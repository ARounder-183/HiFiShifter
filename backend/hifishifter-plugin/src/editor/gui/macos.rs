//! macOS 原语：子 `NSView` + `WKWebView`。
//!
//! 【为什么是"加一个子视图"而不是开窗口】宿主给的是它自己的 `NSView*`
//! （`kPlatformTypeNSView`，坐标是**逻辑点**）。我们在它里面放一个容器视图，
//! 容器里放 `WKWebView` —— 于是宿主的布局、缩放、隐藏、Tab 顺序全都照旧生效，
//! 我们不需要管任何窗口生命周期。
//!
//! 【为什么不跑自己的 run loop】`NSApplication` 已经在跑。回执泵挂在
//! `NSTimer`（20 ms，与 Windows 的 `WM_TIMER` 同频）上，加入
//! `NSRunLoopCommonModes` —— 默认模式在拖拽/菜单跟踪时会停，回执就会卡住。
//!
//! 【为什么用 `hfs-plugin://` 而不是 `file://`】三条硬理由：ES module 的跨源加载需要
//! 非公开偏好；`file://` 的 origin 是 opaque，`localStorage` 直接抛；CSP 里的
//! `'self'` 在 `file://` 下语义含混。见方案 `5.3`。

use super::client::{self, EditorClient};
use super::port::{EditorHost, EditorPort, GuiFidelity, NativeWindow};
use super::protocol;
use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject, NSObjectProtocol, ProtocolObject};
use objc2::{define_class, msg_send, DefinedClass, MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{NSAutoresizingMaskOptions, NSView};
use objc2_foundation::{
    NSData, NSHTTPURLResponse, NSMutableDictionary, NSRunLoop, NSRunLoopCommonModes, NSString,
    NSTimer, NSURLRequest, NSURL,
};
use objc2_web_kit::{
    WKNavigation, WKNavigationAction, WKNavigationActionPolicy, WKNavigationDelegate,
    WKScriptMessage, WKScriptMessageHandler, WKURLSchemeHandler, WKURLSchemeTask,
    WKUserContentController, WKUserScript, WKUserScriptInjectionTime, WKWebView,
    WKWebViewConfiguration, WKWebsiteDataStore,
};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// 回执泵的节拍：与 Windows 的 `WM_TIMER`、Linux 的 `g_timeout_add` 同频。
const TICK: f64 = 0.02;

static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);
static LIVE_VIEWS: AtomicUsize = AtomicUsize::new(0);

/// 脚本消息通道名。必须与 `gui/bridge.js` 里的 `messageHandlers.hfs` 一致。
const HANDLER: &str = "hfs";

// ─────────────────────── 四个 ObjC 小类 ───────────────────────
//
// 【为什么每个类只持有一个 `Rc<RefCell<EditorClient>>`】回调必须能找回判定状态，
// 而 Policy 是 `!Send` 的普通 Rust 结构。VST3 保证 `IPlugView` 与 WebKit 的
// 主线程回调都在同一线程上，所以 `Rc` 在这里是安全的；`MainThreadOnly` 把它钉在
// 类型层面（这类对象只能在主线程创建/使用）。

struct ScriptIvars {
    client: Rc<RefCell<EditorClient>>,
}
define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[ivars = ScriptIvars]
    struct HFSScriptHandler;

    unsafe impl NSObjectProtocol for HFSScriptHandler {}

    unsafe impl WKScriptMessageHandler for HFSScriptHandler {
        #[unsafe(method(userContentController:didReceiveScriptMessage:))]
        fn did_receive(&self, _controller: &WKUserContentController, message: &WKScriptMessage) {
            // 只接受主 frame 的字符串消息：与 Windows 的 `Source` 来源校验同义。
            if !message.frameInfo().isMainFrame() {
                return;
            }
            let body = message.body();
            let Ok(text) = body.downcast::<NSString>() else {
                return;
            };
            let text = text.to_string();
            if text.len() > protocol::MAX_MESSAGE_BYTES {
                return;
            }
            // 【为什么回调里什么都不做，只转给 Policy】与 Windows 的
            // `WebMessageReceived` 同一条纪律：UI 回调不得分析、推理、读文件或等待 worker。
            let client = self.ivars().client.clone();
            client::on_message(&client, &text);
        }
    }
);

struct NavigationIvars {
    client: Rc<RefCell<EditorClient>>,
}
define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[ivars = NavigationIvars]
    struct HFSNavigation;

    unsafe impl NSObjectProtocol for HFSNavigation {}

    unsafe impl WKNavigationDelegate for HFSNavigation {
        #[unsafe(method(webView:decidePolicyForNavigationAction:decisionHandler:))]
        fn decide(
            &self,
            _webview: &WKWebView,
            action: &WKNavigationAction,
            handler: &block2::Block<dyn Fn(WKNavigationActionPolicy)>,
        ) {
            let allowed = unsafe { action.request() }
                .URL()
                .and_then(|url| url.absoluteString())
                .is_some_and(|uri| protocol::is_allowed_navigation(&uri.to_string()));
            let policy = if allowed {
                WKNavigationActionPolicy::Allow
            } else {
                WKNavigationActionPolicy::Cancel
            };
            handler.call((policy,));
        }

        #[unsafe(method(webView:didFinishNavigation:))]
        fn did_finish(&self, _webview: &WKWebView, _navigation: &WKNavigation) {
            let client = self.ivars().client.clone();
            client.borrow_mut().on_page_ready();
        }

        /// `WKWebView` 的渲染进程崩溃（内存压力、GPU 进程重启）。**不要**自动 reload：
        /// 那会重放一整套 `plugin_health` 握手，用户看到界面闪一下并丢失当前选中。
        #[unsafe(method(webViewWebContentProcessDidTerminate:))]
        fn terminated(&self, _webview: &WKWebView) {
            let client = self.ivars().client.clone();
            client
                .borrow_mut()
                .mark_dead("the web content process terminated");
        }
    }
);

struct SchemeIvars {
    root: std::path::PathBuf,
}
define_class!(
    #[unsafe(super(NSObject))]
    #[ivars = SchemeIvars]
    struct HFSSchemeHandler;

    unsafe impl NSObjectProtocol for HFSSchemeHandler {}

    unsafe impl WKURLSchemeHandler for HFSSchemeHandler {
        #[unsafe(method(webView:startURLSchemeTask:))]
        fn start(&self, _webview: &WKWebView, task: &ProtocolObject<dyn WKURLSchemeTask>) {
            serve_asset(&self.ivars().root, task);
        }

        #[unsafe(method(webView:stopURLSchemeTask:))]
        fn stop(&self, _webview: &WKWebView, _task: &ProtocolObject<dyn WKURLSchemeTask>) {}
    }
);

/// 把 `hfs-plugin://local/<path>` 映射到 bundle 里的真实文件。
fn serve_asset(root: &std::path::Path, task: &ProtocolObject<dyn WKURLSchemeTask>) {
    let request = unsafe { task.request() };
    let Some(url) = request.URL() else {
        return;
    };
    let path = url.path().map(|path| path.to_string()).unwrap_or_default();
    let (status, mime, body) = match protocol::resolve_asset(root, &path) {
        Some(file) => match std::fs::read(&file) {
            Ok(bytes) => (200_i32, protocol::mime_for(&file), bytes),
            Err(error) => (
                500,
                "text/plain",
                format!("unreadable: {path} ({error})").into_bytes(),
            ),
        },
        // 【为什么给一条明确的 body 而不是静默失败】缺一个 asset 的包是最常见的
        // "打包漏了"，而空白响应会让它表现成"页面什么都不显示"。
        None => (404, "text/plain", format!("not found: {path}").into_bytes()),
    };
    let headers = NSMutableDictionary::<NSString, NSString>::new();
    headers.insert(
        &NSString::from_str("Content-Type"),
        &NSString::from_str(mime),
    );
    headers.insert(
        &NSString::from_str("Content-Length"),
        &NSString::from_str(&body.len().to_string()),
    );
    let version = NSString::from_str("HTTP/1.1");
    let response = NSHTTPURLResponse::initWithURL_statusCode_HTTPVersion_headerFields(
        NSHTTPURLResponse::alloc(),
        &url,
        status,
        Some(&version),
        Some(&headers),
    );
    let data = NSData::from_vec(body);
    // SAFETY: 三个调用都发生在同一个 `WKURLSchemeTask` 生命周期内，且顺序正确
    // （先 response，再 data，最后 finish）。
    unsafe {
        task.didReceiveResponse(&response);
        task.didReceiveData(&data);
        task.didFinish();
    }
}

// ─────────────────────── 端口与视图 ───────────────────────

struct MacPort {
    webview: RefCell<Option<Retained<WKWebView>>>,
    closed: Cell<bool>,
}

impl EditorPort for MacPort {
    fn post_message(&self, text: &str) -> bool {
        if self.closed.get() {
            return false;
        }
        let webview = self.webview.borrow().clone();
        let Some(webview) = webview else {
            return false;
        };
        let script = NSString::from_str(&protocol::js_expression(text));
        // SAFETY: 回调只记日志；它不持有任何 Rust 借用。
        unsafe {
            let handler = RcBlock::new(move |_value: *mut AnyObject, error: *mut AnyObject| {
                if !error.is_null() {
                    crate::log_line("[gui] page delivery failed");
                }
            });
            webview.evaluateJavaScript_completionHandler(&script, Some(&handler));
        }
        true
    }

    fn show_error(&self, text: &str) {
        crate::log_line(&format!("Native editor error: {text}"));
        // WebKit 没有"画一个错误控件"的接口；把错误写进页面，用户至少能看到一行字。
        let webview = self.webview.borrow().clone();
        if let Some(webview) = webview {
            let text = text.replace('\\', " ").replace('\'', " ");
            let script = NSString::from_str(&format!(
                "document.title = 'HiFiShifter editor: {text}'; if (document.body) {{ document.body.textContent = 'HiFiShifter editor: {text}'; }}"
            ));
            unsafe {
                webview.evaluateJavaScript_completionHandler(&script, None);
            }
        }
    }

    fn close(&self) {
        self.closed.set(true);
        *self.webview.borrow_mut() = None;
    }
}

/// 宿主持有的句柄。只含一个 token —— 真实对象在 `VIEWS`（主线程 thread_local）里。
pub(super) struct MacHost {
    token: usize,
}

impl EditorHost for MacHost {
    fn fidelity(&self) -> GuiFidelity {
        GuiFidelity::Embedded
    }

    fn resize(&self, width: i32, height: i32) -> Result<(), String> {
        with_view(self.token, |view| {
            // 【为什么是逻辑点】`iplugview.h` 对 macOS 明写坐标是逻辑单位；
            // Windows/Linux 是物理像素。这里**不要**乘 `backingScaleFactor`。
            let size = objc2_foundation::NSSize {
                width: width as f64,
                height: height as f64,
            };
            view.container.setFrameSize(size);
            Ok(())
        })
        .unwrap_or(Ok(()))
    }

    fn focus(&self) -> Result<(), String> {
        with_view(self.token, |view| {
            // 让 `WKWebView` 自己成为第一响应者；不改变宿主窗口的 key 状态。
            if let Some(window) = view.webview.window() {
                let _ = window.makeFirstResponder(Some(&view.webview));
            }
            Ok(())
        })
        .unwrap_or(Ok(()))
    }

    fn forward_edit_key(&self, _key: char, _mods: i16, _down: bool) -> bool {
        // macOS 上键盘事件沿响应链直达 `WKWebView`，宿主不会先把 Ctrl+C 交给
        // `IPlugView::keyDown` —— 这条通路在本平台不存在（与 Linux 同）。
        false
    }

    fn host_tick(&self) {
        // 定时器就在主线程上，已经由 `NSTimer` 做过。
    }

    fn close(&self) {
        close_view(self.token);
    }
}

struct View {
    client: Rc<RefCell<EditorClient>>,
    port: Rc<MacPort>,
    container: Retained<NSView>,
    webview: Retained<WKWebView>,
    timer: Retained<NSTimer>,
    /// 委托对象必须被**我们自己**持有：`WKUserContentController` 与 `WKWebView` 的
    /// delegate 都是 `weak`（`addScriptMessageHandler` 除外，那是强引用）。
    delegates: Vec<Retained<AnyObject>>,
}

thread_local! {
    static VIEWS: RefCell<HashMap<usize, View>> = RefCell::new(HashMap::new());
}

fn with_view<T>(token: usize, action: impl FnOnce(&View) -> T) -> Option<T> {
    VIEWS.with(|views| views.borrow().get(&token).map(action))
}

impl MacHost {
    pub(super) fn attach(
        parent: NativeWindow,
        width: i32,
        height: i32,
        link: std::sync::Arc<crate::editor::routing::EditorLink>,
    ) -> Result<Self, String> {
        let marker = MainThreadMarker::new()
            .ok_or_else(|| "the editor view must be created on the main thread".to_string())?;
        let root = crate::editor::resources::frontend_directory()?;
        let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed) as usize;

        let configuration = unsafe { WKWebViewConfiguration::new(marker) };
        // 【为什么用非持久 store】插件窗口的用户数据是**每进程临时**的（与 Windows 的
        // 每进程 profile 目录同义）：偏好靠后端配置文件持久化，WebView 自己不该往
        // 宿主的容器目录里写东西。
        unsafe {
            configuration.setWebsiteDataStore(&WKWebsiteDataStore::nonPersistentDataStore(marker));
        }
        let scheme = HFSSchemeHandler::alloc(marker).set_ivars(SchemeIvars { root: root.clone() });
        let scheme: Retained<HFSSchemeHandler> = unsafe { msg_send![super(scheme), init] };
        unsafe {
            configuration.setURLSchemeHandler_forURLScheme(
                ProtocolObject::from_ref(&*scheme),
                &NSString::from_str(protocol::SCHEME),
            );
        }

        let frame = objc2_foundation::NSRect {
            origin: objc2_foundation::NSPoint { x: 0.0, y: 0.0 },
            size: objc2_foundation::NSSize {
                width: width as f64,
                height: height as f64,
            },
        };
        let webview = unsafe {
            WKWebView::initWithFrame_configuration(WKWebView::alloc(marker), frame, &configuration)
        };
        let container = unsafe { NSView::initWithFrame(NSView::alloc(marker), frame) };
        container.setAutoresizesSubviews(true);
        container.setAutoresizingMask(
            NSAutoresizingMaskOptions::ViewWidthSizable
                | NSAutoresizingMaskOptions::ViewHeightSizable,
        );
        webview.setAutoresizingMask(
            NSAutoresizingMaskOptions::ViewWidthSizable
                | NSAutoresizingMaskOptions::ViewHeightSizable,
        );
        container.addSubview(&webview);

        // 判定状态与端口
        let client = EditorClient::open(link);
        let port = Rc::new(MacPort {
            webview: RefCell::new(Some(webview.clone())),
            closed: Cell::new(false),
        });
        let dyn_port: Rc<dyn EditorPort> = port.clone();
        {
            let mut state = client.borrow_mut();
            state.set_port(dyn_port);
            state.set_window(token);
        }

        // 消息通道 + 两份 document-start 脚本（桥接**先于** bootstrap）
        let controller = configuration.userContentController();
        let script_handler = HFSScriptHandler::alloc(marker).set_ivars(ScriptIvars {
            client: client.clone(),
        });
        let script_handler: Retained<HFSScriptHandler> =
            unsafe { msg_send![super(script_handler), init] };
        unsafe {
            controller.addScriptMessageHandler_name(
                ProtocolObject::from_ref(&*script_handler),
                &NSString::from_str(HANDLER),
            );
        }
        add_script(&controller, marker, BRIDGE);
        let boot = {
            let state = client.borrow();
            client::bootstrap_json(&state.link, &state.view_id)
        };
        add_script(
            &controller,
            marker,
            &format!("window.__HFS_PLUGIN_BOOTSTRAP__={boot};"),
        );

        // 导航策略
        let navigation = HFSNavigation::alloc(marker).set_ivars(NavigationIvars {
            client: client.clone(),
        });
        let navigation: Retained<HFSNavigation> = unsafe { msg_send![super(navigation), init] };
        unsafe {
            webview.setNavigationDelegate(Some(ProtocolObject::from_ref(&*navigation)));
        }

        // 20 ms 回执泵
        let weak = Rc::downgrade(&client);
        let block = RcBlock::new(move |_timer: *mut NSTimer| {
            if let Some(client) = weak.upgrade() {
                client::tick(&client);
            }
        });
        let timer = unsafe { NSTimer::timerWithTimeInterval_repeats_block(TICK, true, &block) };
        // 【为什么加入 CommonModes】默认模式在拖拽/菜单跟踪时会停 —— 那时回执会卡住，
        // 而用户正按着鼠标拖片段。
        NSRunLoop::mainRunLoop().addTimer_forMode(&timer, unsafe { NSRunLoopCommonModes });

        VIEWS.with(|views| {
            views.borrow_mut().insert(
                token,
                View {
                    client: client.clone(),
                    port: port.clone(),
                    container: container.clone(),
                    webview: webview.clone(),
                    timer,
                    delegates: vec![script_handler.into(), navigation.into(), scheme.into()],
                },
            );
        });
        LIVE_VIEWS.fetch_add(1, Ordering::AcqRel);

        // 挂到宿主给的父视图里。
        let parent = parent as *mut NSView;
        if parent.is_null() {
            return Err("parent is not a live NSView".into());
        }
        // SAFETY: `attached` 的 parent 由宿主保证在本次调用期间存活（VST3 契约）。
        let parent = unsafe { &*parent };
        parent.addSubview(&container);

        // 页面
        let url = NSURL::URLWithString(&NSString::from_str(protocol::PAGE_URL))
            .ok_or_else(|| "invalid editor page URL".to_string())?;
        unsafe {
            webview.loadRequest(&NSURLRequest::requestWithURL(&url));
        }
        crate::log_line(&format!(
            "[gui] wkwebview {token} created ({}x{})",
            width, height
        ));
        Ok(Self { token })
    }
}

fn add_script(controller: &WKUserContentController, marker: MainThreadMarker, source: &str) {
    // SAFETY: 脚本对象只在 documentStart 注入到主 frame；字符串在调用期间存活。
    unsafe {
        let script = WKUserScript::initWithSource_injectionTime_forMainFrameOnly(
            WKUserScript::alloc(marker),
            &NSString::from_str(source),
            WKUserScriptInjectionTime::AtDocumentStart,
            true,
        );
        controller.addUserScript(&script);
    }
}

fn close_view(token: usize) {
    let view = VIEWS.with(|views| views.borrow_mut().remove(&token));
    let Some(view) = view else {
        return;
    };
    view.timer.invalidate();
    view.client.borrow_mut().close();
    view.port.close();
    // 【顺序】先摘 `messageHandler` 再拆视图：`addScriptMessageHandler` 是**强引用**，
    // 不摘就会每开一次窗口泄漏一个 webview（方案 `5.4` 的循环引用警告）。
    if let Some(webview) = view.port.webview.borrow().clone() {
        let controller = webview.configuration().userContentController();
        unsafe {
            controller.removeScriptMessageHandlerForName(&NSString::from_str(HANDLER));
        }
        unsafe {
            webview.stopLoading();
            webview.setNavigationDelegate(None);
        }
    }
    view.webview.removeFromSuperview();
    view.container.removeFromSuperview();
    drop(view.delegates);
    LIVE_VIEWS.fetch_sub(1, Ordering::AcqRel);
}

/// 卸载前收尾：还有活视图就明确拒绝（留着一个指向已卸载代码段的委托 = 野跳转）。
pub(super) fn shutdown() -> bool {
    if LIVE_VIEWS.load(Ordering::Acquire) != 0 {
        crate::log_line("ExitDll refused: native editor views still alive");
        return false;
    }
    true
}

/// 平台垫片：让 WebKit 假装成 WebView2 的那个 `chrome.webview`。
const BRIDGE: &str = include_str!("bridge.js");
