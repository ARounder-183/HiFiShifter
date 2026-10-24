//! Linux 原语：`GtkPlug`（XEmbed）+ `WebKitWebView`。
//!
//! 这一份是三个平台里唯一**必须**处理"GTK 跑在哪条线程上"的实现，而 GTK 的铁律是
//! "所有 GTK 调用都来自初始化它的那条线程"。VST3 的 `attached()` 在**宿主**的 UI
//! 线程上，两者能不能对齐取决于宿主，所以判据是**运行时**的：
//!
//! | 情形 | 判定 | 视图线程 | 自报 `GuiFidelity` |
//! |---|---|---|---|
//! | A 寄宿主的 GTK | 进程里 GDK 已经初始化 | 宿主 UI 线程（= 当前线程） | `Embedded` |
//! | B 自己起 GTK | 没有已初始化的 GDK | 本模块自建的唯一 GUI 线程 | `FloatingWindow` |
//!
//! 【为什么情形 B 是降级而不是等价】泵不在宿主 UI 线程上，于是 `flush_dirty` /
//! `host_undo.tick()` 只能由宿主的下一次 `IPlugView` 回调补做 —— 宿主 Undo 的合并
//! 粒度会变粗。功能可用，语义有损，所以它要能被看见（日志一行）。
//!
//! 【为什么 Wayland 只能浮窗】WebKitGTK 没有"把 `WebKitWebView` 渲染到外部
//! `wl_surface`"的公开 API，`GdkSurface` 也不能挂在别人的 `wl_display` 上。
//! 于是 `WaylandSurfaceID` 一律走 `FloatingWindow`（见 `view.rs` 的 `platform()`）。

use super::client::{self, EditorClient};
use super::port::{EditorHost, EditorPort, GuiFidelity, NativeWindow};
use super::protocol;
use gtk::gio;
use gtk::glib;
use gtk::prelude::*;
use javascriptcore::ValueExt;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender, SyncSender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use webkit2gtk::{
    NavigationPolicyDecisionExt, PolicyDecisionExt, SecurityManagerExt, URIRequestExt,
    URISchemeRequestExt, UserContentManagerExt, WebContextExt, WebViewExt,
};

/// 回执泵的节拍：与 Windows 的 `WM_TIMER`、macOS 的 `NSTimer` 同频。
///
/// 【为什么不能为了省电调大】回执延迟直接体现在旋钮手感上（`pluginHost` 的
/// `invoke` 是请求-响应式）。
const TICK: Duration = Duration::from_millis(20);

/// `attached()` 等 GTK 线程建视图的上限。
///
/// 【为什么宁可不给 GUI 也不能等】超时就返回失败：宿主看到空面板 + 一行日志，
/// 总好过把宿主的 UI 线程挂在这里。
const ATTACH_TIMEOUT: Duration = Duration::from_secs(2);

/// `ExitDll` 等 GTK 线程确认退出的上限。
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);
/// 存活视图数（跨线程可读；`VIEWS` 是 thread_local，问不出别的线程）。
static LIVE_VIEWS: AtomicUsize = AtomicUsize::new(0);

// 视图在**拥有它的那条线程**上的全部状态。
//
// 【为什么是 thread_local 而不是全局表】`GtkPlug` / `WebKitWebView` 都是 `!Send` 的
// GObject；把它们塞进全局表会让整个模块变成 `unsafe impl Send` 的谎言。
// thread_local 把"只能在创建它的线程上碰"这条不变式交给类型系统。
thread_local! {
    static VIEWS: RefCell<HashMap<usize, View>> = RefCell::new(HashMap::new());
}

struct View {
    client: Rc<RefCell<EditorClient>>,
    port: Rc<LinuxPort>,
    /// `GtkPlug`（XEmbed）或 `GtkWindow`（浮窗）。
    container: gtk::Widget,
    webview: webkit2gtk::WebView,
}

/// WebKitGTK 的出站端口。
struct LinuxPort {
    /// `WebView` 在 `attach_view` 里才建好；之前投递只能报"投不出去"。
    webview: RefCell<Option<webkit2gtk::WebView>>,
    closed: Cell<bool>,
}

impl EditorPort for LinuxPort {
    /// 【为什么这里总是 `true`】WebKitGTK 没有同步投递接口：
    /// `evaluate_javascript` 是异步的，返回时还看不出成败。真正的失败由两条信号报上来
    /// ——`web-process-terminated` 与 `load-failed`（见 `attach_view`），它们直接
    /// 把视图标记为死亡，比"数连续投递失败"更准。Windows 的 WebView2 有同步返回值，
    /// 所以那条路继续用计数判据。
    fn post_message(&self, text: &str) -> bool {
        if self.closed.get() {
            return false;
        }
        let webview = self.webview.borrow().clone();
        let Some(webview) = webview else {
            return false;
        };
        let expression = protocol::js_expression(text);
        webview.evaluate_javascript(
            &expression,
            None,
            None,
            None::<&gio::Cancellable>,
            |result| {
                if let Err(error) = result {
                    crate::log_line(&format!("[gui] page delivery failed: {error}"));
                }
            },
        );
        true
    }

    fn show_error(&self, text: &str) {
        crate::log_line(&format!("Native editor error: {text}"));
        // WebKit 没有"画一个错误控件"的接口；把错误写进 `document.title` 与 body，
        // 用户至少能看到一行字，而不是对着白屏猜。
        let webview = self.webview.borrow().clone();
        if let Some(webview) = webview {
            let text = text.replace('\\', " ").replace('\'', " ");
            let script = format!(
                "document.title = 'HiFiShifter editor: {text}'; if (document.body) {{ document.body.textContent = 'HiFiShifter editor: {text}'; }}"
            );
            webview.evaluate_javascript(&script, None, None, None::<&gio::Cancellable>, |_| {});
        }
    }

    fn close(&self) {
        self.closed.set(true);
        *self.webview.borrow_mut() = None;
    }
}

/// 宿主持有的句柄。**只含可跨线程的东西**（token / 线程身份 / 通道 / 路由）。
pub(super) struct LinuxHost {
    token: usize,
    owner: std::thread::ThreadId,
    /// 情形 B 才有：命令投给私有 GTK 线程。情形 A 走内联。
    control: Option<Sender<Control>>,
    fidelity: GuiFidelity,
    link: Arc<crate::editor::routing::EditorLink>,
}

impl EditorHost for LinuxHost {
    fn fidelity(&self) -> GuiFidelity {
        self.fidelity
    }

    fn resize(&self, width: i32, height: i32) -> Result<(), String> {
        self.submit(Control::Resize {
            token: self.token,
            width,
            height,
        })
    }

    fn focus(&self) -> Result<(), String> {
        self.submit(Control::Focus { token: self.token })
    }

    /// 【为什么 Linux 不转交编辑键】键盘事件直达 `WebKitWebView`，宿主不会先把
    /// Ctrl+C 交给 `IPlugView::keyDown` —— 这条通路在本平台不存在。
    fn forward_edit_key(&self, _key: char, _mods: i16, _down: bool) -> bool {
        false
    }

    fn host_tick(&self) {
        // 情形 A：节拍就在宿主线程上，已经由 `g_timeout_add` 做过。
        // 情形 B：泵在私有线程上，宿主侧的事只能在这里补做。
        if matches!(self.fidelity, GuiFidelity::FloatingWindow) {
            client::host_tick(&self.link);
        }
    }

    fn close(&self) {
        let _ = self.submit(Control::Close { token: self.token });
    }
}

impl LinuxHost {
    /// 命令要么**就地**执行（我们已经在这条线程上），要么投给拥有视图的那条线程。
    fn submit(&self, command: Control) -> Result<(), String> {
        if std::thread::current().id() == self.owner {
            run_inline(command);
            return Ok(());
        }
        match &self.control {
            Some(control) => control
                .send(command)
                .map_err(|_| "the GTK thread is gone".to_string()),
            // 情形 A 里"当前线程不是视图线程"意味着宿主在别的线程上调用 `IPlugView`，
            // 那已经违反 VST3 的契约；明确拒绝好过在错误的线程上碰 GTK。
            None => Err("GTK view touched off its own thread".into()),
        }
    }
}

/// 跨线程命令。
enum Control {
    Resize {
        token: usize,
        width: i32,
        height: i32,
    },
    Focus {
        token: usize,
    },
    Close {
        token: usize,
    },
    /// 只在情形 B：建视图并回报 token。
    Attach {
        request: AttachRequest,
        reply: SyncSender<Result<usize, String>>,
    },
    /// `ExitDll`：停掉 GTK 主循环。
    Shutdown {
        reply: SyncSender<()>,
    },
}

struct AttachRequest {
    parent: NativeWindow,
    width: i32,
    height: i32,
    embedded: bool,
    link: Arc<crate::editor::routing::EditorLink>,
}

/// 进程级的私有 GUI 线程（情形 B）。
struct GuiThread {
    control: Sender<Control>,
    owner: std::thread::ThreadId,
    handle: Mutex<Option<std::thread::JoinHandle<()>>>,
}

static GUI_THREAD: OnceLock<Option<GuiThread>> = OnceLock::new();

pub(super) fn create(
    parent: NativeWindow,
    width: i32,
    height: i32,
    embedded: bool,
    link: Arc<crate::editor::routing::EditorLink>,
) -> Result<Rc<dyn EditorHost>, String> {
    let request = AttachRequest {
        parent,
        width,
        height,
        embedded,
        link: link.clone(),
    };
    if gdk_is_initialized() {
        // 情形 A：宿主的 GTK 已经在这条线程上跑着。
        //
        // 【为什么这里什么都不用做】本 crate 打开了 gtk 的
        // `unsafe-assume-initialized`：gtk-rs 因此认为"GTK 已在本线程初始化"，
        // `assert_initialized_main_thread!` 全部放行。代价是它不再替我们守线程纪律 ——
        // 由 `VIEWS`（thread_local）与 `LinuxHost::submit`（先比线程身份）来守。
        let token = attach_view(request)?;
        crate::log_line("[gui] hosting the host's GTK main loop (embedded)");
        return Ok(Rc::new(LinuxHost {
            token,
            owner: std::thread::current().id(),
            control: None,
            fidelity: GuiFidelity::Embedded,
            link,
        }));
    }
    // 情形 B：进程里没有 GTK 主循环，只能自己起一条。
    let thread = gui_thread()?;
    let (reply, replies) = std::sync::mpsc::sync_channel(1);
    thread
        .control
        .send(Control::Attach { request, reply })
        .map_err(|_| "the GTK thread is gone".to_string())?;
    match replies.recv_timeout(ATTACH_TIMEOUT) {
        Ok(Ok(token)) => {
            crate::log_line("[gui] hosting our own GTK thread (floating window)");
            Ok(Rc::new(LinuxHost {
                token,
                owner: thread.owner,
                control: Some(thread.control.clone()),
                fidelity: GuiFidelity::FloatingWindow,
                link,
            }))
        }
        Ok(Err(error)) => Err(error),
        Err(_) => Err("the GTK thread did not create the editor view in time".into()),
    }
}

/// 卸载前收尾：停掉私有 GUI 线程。
///
/// 【为什么不 `join` 到底】`gtk_main_quit` 只是置一个标志；如果此刻某条 GTK 回调正
/// 卡在宿主 API 上，`join` 会带着 GTK 的栈帧死锁在 `ExitDll` 里 —— 那比"漏一条线程"
/// 严重得多。所以只等一个"命令已收到"的回执，超时就放弃。
pub(super) fn shutdown() -> bool {
    if LIVE_VIEWS.load(Ordering::Acquire) != 0 {
        crate::log_line("ExitDll refused: native editor views still alive");
        return false;
    }
    let Some(Some(thread)) = GUI_THREAD.get() else {
        return true;
    };
    let (reply, replies) = std::sync::mpsc::sync_channel(1);
    if thread.control.send(Control::Shutdown { reply }).is_err() {
        return true;
    }
    if replies.recv_timeout(SHUTDOWN_TIMEOUT).is_err() {
        crate::log_line("ExitDll: the GTK thread did not stop in time; abandoning it");
    }
    // 句柄直接丢掉（线程分离），见上面的理由。
    let _ = thread
        .handle
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take();
    true
}

/// GDK 是否已经在本进程里初始化过。
///
/// 【为什么用 `dlopen(NULL)` + `dlsym` 而不是链接期符号】我们**不能**为了探测去
/// 初始化 GTK（那正是要判断的事）；`RTLD_DEFAULT` 只查"已经装载的模块"，
/// 不加载任何东西。
fn gdk_is_initialized() -> bool {
    type GetDefault = unsafe extern "C" fn() -> *mut std::ffi::c_void;
    // SAFETY: dlopen/dlsym 只读进程的模块表；符号名与签名由 GDK 3 固定
    // （`GdkDisplay *gdk_display_get_default(void)`）。
    unsafe {
        let handle = libc::dlopen(std::ptr::null(), libc::RTLD_LAZY);
        if handle.is_null() {
            return false;
        }
        let symbol = libc::dlsym(handle, c"gdk_display_get_default".as_ptr());
        libc::dlclose(handle);
        if symbol.is_null() {
            return false;
        }
        let get_default: GetDefault = std::mem::transmute(symbol);
        !get_default().is_null()
    }
}

fn gui_thread() -> Result<&'static GuiThread, String> {
    GUI_THREAD
        .get_or_init(|| {
            start_gui_thread()
                .map_err(|error| {
                    crate::log_line(&format!("[gui] GTK thread unavailable: {error}"));
                    error
                })
                .ok()
        })
        .as_ref()
        .ok_or_else(|| "GTK could not be initialized in this process".to_string())
}

fn start_gui_thread() -> Result<GuiThread, String> {
    let (control, commands) = channel::<Control>();
    let (ready, ready_reply) = std::sync::mpsc::sync_channel::<Result<(), String>>(1);
    let handle = std::thread::Builder::new()
        .name("hfs-gui".into())
        .spawn(move || {
            // 【为什么不调 §gtk::init()§】本 crate 打开了 §unsafe-assume-initialized§，
            // 于是 §gtk::init()§ 会认为"已经初始化"并直接返回 —— 它**不会**真的去初始化。
            // 情形 B 里我们才是那个初始化者，所以显式调 §gtk_init_check§（argc/argv 允许
            // 为 NULL，GTK 文档明写）。
            //
            // SAFETY: 只在尚未初始化 GDK 的进程里、在本模块自建的线程上调用一次。
            let initialized =
                unsafe { gtk::ffi::gtk_init_check(std::ptr::null_mut(), std::ptr::null_mut()) };
            if initialized == glib::ffi::GFALSE {
                let _ = ready.send(Err("gtk_init_check failed (no display?)".into()));
                return;
            }
            let _ = ready.send(Ok(()));
            CONTROL.with(|slot| *slot.borrow_mut() = Some(commands));
            ensure_tick();
            gtk::main();
            // 收尾：销毁残留视图（正常路径下 `ExitDll` 已经要求它们先关掉）。
            VIEWS.with(|views| {
                for (_, view) in views.borrow_mut().drain() {
                    view.port.close();
                    view.webview.stop_loading();
                }
            });
        })
        .map_err(|error| error.to_string())?;
    match ready_reply.recv_timeout(ATTACH_TIMEOUT) {
        Ok(Ok(())) => Ok(GuiThread {
            control,
            owner: handle.thread().id(),
            handle: Mutex::new(Some(handle)),
        }),
        Ok(Err(error)) => Err(error),
        Err(_) => Err("GTK did not initialize in time".into()),
    }
}

/// 拥有视图的那条线程上：执行一条命令。
fn run_inline(command: Control) {
    match command {
        Control::Resize {
            token,
            width,
            height,
        } => VIEWS.with(|views| {
            if let Some(view) = views.borrow().get(&token) {
                // 坐标是**物理像素**（iplugview.h 对 X11 的约定），与 Windows 一致。
                view.container.set_size_request(width, height);
            }
        }),
        Control::Focus { token } => VIEWS.with(|views| {
            if let Some(view) = views.borrow().get(&token) {
                view.webview.grab_focus();
            }
        }),
        Control::Close { token } => close_view(token),
        Control::Attach { request, reply } => {
            let _ = reply.send(attach_view(request));
        }
        Control::Shutdown { reply } => {
            let _ = reply.send(());
            gtk::main_quit();
        }
    }
}

/// 20 ms 节拍：先收命令，再推进每条视图的回执泵。
///
/// 【为什么只 `pump` 不 `tick`】`client::tick` 会碰宿主 API，而这里可能不在宿主 UI
/// 线程上（情形 B）。宿主侧的事由 `LinuxHost::host_tick` 在宿主线程上补做。
fn tick() -> glib::ControlFlow {
    CONTROL.with(|slot| {
        if let Some(commands) = slot.borrow().as_ref() {
            while let Ok(command) = commands.try_recv() {
                run_inline(command);
            }
        }
    });
    let clients = VIEWS.with(|views| {
        views
            .borrow()
            .values()
            .map(|view| (view.client.clone(), view.container.clone()))
            .collect::<Vec<_>>()
    });
    for (client, container) in clients {
        client::pump(&client);
        if client.borrow().is_dead() {
            let dead = VIEWS.with(|views| {
                views
                    .borrow()
                    .iter()
                    .find(|(_, view)| Rc::ptr_eq(&view.client, &client))
                    .map(|(token, _)| *token)
            });
            if let Some(token) = dead {
                crate::log_line("[gui] closing the editor view: the page is gone");
                close_view(token);
            }
        }
        let _ = container;
    }
    glib::ControlFlow::Continue
}

fn ensure_tick() {
    TICK_SOURCE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(glib::timeout_add_local(TICK, tick));
        }
    })
}

fn close_view(token: usize) {
    let view = VIEWS.with(|views| views.borrow_mut().remove(&token));
    let Some(view) = view else {
        return;
    };
    view.client.borrow_mut().close();
    view.port.close();
    view.webview.stop_loading();
    // SAFETY: 容器由本视图独占；GTK 的 destroy 只要求在同一线程上调用。
    unsafe { view.container.destroy() };
    LIVE_VIEWS.fetch_sub(1, Ordering::AcqRel);
}

/// 在拥有视图的线程上真正建视图。
fn attach_view(request: AttachRequest) -> Result<usize, String> {
    let assets = crate::editor::resources::frontend_directory()?;
    let context = web_context(&assets);
    let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed) as usize;

    let webview = webkit2gtk::WebView::with_context(&context);
    let manager = webview
        .user_content_manager()
        .ok_or_else(|| "the web view has no user content manager".to_string())?;
    // 桥接脚本必须**先于** bootstrap 注入：`pluginHost.ts` 读 `window.chrome.webview`
    // 的那一刻它就得存在。
    manager.add_script(&webkit2gtk::UserScript::new(
        BRIDGE,
        webkit2gtk::UserContentInjectedFrames::TopFrame,
        webkit2gtk::UserScriptInjectionTime::Start,
        &[],
        &[],
    ));

    // 判定状态先建：bootstrap 里的 `viewId` 由它生成，消息回调也要它。
    let client = EditorClient::open(request.link);
    let boot = {
        let state = client.borrow();
        client::bootstrap_json(&state.link, &state.view_id)
    };
    manager.add_script(&webkit2gtk::UserScript::new(
        &format!("window.__HFS_PLUGIN_BOOTSTRAP__={boot};"),
        webkit2gtk::UserContentInjectedFrames::TopFrame,
        webkit2gtk::UserScriptInjectionTime::Start,
        &[],
        &[],
    ));

    let port = Rc::new(LinuxPort {
        webview: RefCell::new(Some(webview.clone())),
        closed: Cell::new(false),
    });
    let dyn_port: Rc<dyn EditorPort> = port.clone();
    {
        let mut state = client.borrow_mut();
        state.set_port(dyn_port);
        // 句柄是一个**不透明 token**：文件对话框与让路原语都只需要"这是哪个视图"。
        state.set_window(token);
    }

    install_messages(&manager, &client);
    install_policy(&webview, &client);
    install_drop(&webview, &client);

    let container: gtk::Widget = if request.embedded {
        let plug = gtk::Plug::new(request.parent as gtk::xlib::Window);
        plug.add(&webview);
        plug.upcast()
    } else {
        // 浮窗：宿主没有可嵌入的 surface（Wayland）或我们必须用私有 GTK 线程。
        let window = gtk::Window::new(gtk::WindowType::Toplevel);
        window.set_title("HiFiShifter");
        window.set_default_size(request.width, request.height);
        window.add(&webview);
        let weak = Rc::downgrade(&client);
        window.connect_delete_event(move |_, _| {
            if let Some(client) = weak.upgrade() {
                client.borrow_mut().close();
            }
            // 【为什么 `Propagate`】窗口的销毁交给 `close_view`（它要同时摘掉注册表
            // 与节拍）；这里只拒绝"由 GTK 自己删窗口"那条路。
            gtk::glib::Propagation::Proceed
        });
        window.upcast()
    };
    container.set_size_request(request.width, request.height);
    container.show_all();

    VIEWS.with(|views| {
        views.borrow_mut().insert(
            token,
            View {
                client: client.clone(),
                port: port.clone(),
                container: container.clone(),
                webview: webview.clone(),
            },
        );
    });
    LIVE_VIEWS.fetch_add(1, Ordering::AcqRel);
    ensure_tick();

    webview.load_uri(protocol::PAGE_URL);
    crate::log_line(&format!(
        "[gui] webkit view {token} created ({}x{})",
        request.width, request.height
    ));
    Ok(token)
}

/// 进程级唯一的 WebContext：自定义 scheme 只注册一次。
///
/// 【为什么是 ephemeral】插件窗口的用户数据是**每进程临时**的（与 Windows 的
/// 每进程 profile 目录同义）：偏好靠后端配置文件持久化，WebView 自己不该往
/// 用户的 `~/.local/share` 里写东西。
fn web_context(assets: &std::path::Path) -> webkit2gtk::WebContext {
    // 【为什么是 thread_local 而不是 §static§】§WebContext§ 是 GObject（§!Send§），
    // 塞进 §OnceLock§ 会要求它 §Send + Sync§ —— 那等于对 GObject 撒谎。
    // 每个拥有视图的线程各一份，正好也是"GTK 对象只属于一条线程"的写照。
    CONTEXT.with(|slot| {
        let mut slot = slot.borrow_mut();
        slot.get_or_insert_with(|| {
            let context = webkit2gtk::WebContext::new_ephemeral();
            let security = context
                .security_manager()
                .expect("a WebContext always has a security manager");
            // 【为什么必须显式登记】自定义 scheme 默认**不是** secure context，
            // 于是 `crypto.subtle` 之类的 API 不可用。今天的前端没用到，但留下这个坑
            // 只会让下一个人以为"WebKit 就是不行"。
            security.register_uri_scheme_as_secure(protocol::SCHEME);
            security.register_uri_scheme_as_cors_enabled(protocol::SCHEME);
            let root = assets.to_path_buf();
            context.register_uri_scheme(protocol::SCHEME, move |request| {
                serve_asset(&root, request);
            });
            context
        })
        .clone()
    })
}

/// 把 `hfs-plugin://local/<path>` 映射到 bundle 里的真实文件。
fn serve_asset(root: &std::path::Path, request: &webkit2gtk::URISchemeRequest) {
    let path = request.path().unwrap_or_default();
    let Some(file) = protocol::resolve_asset(root, &path) else {
        // 【为什么给一条明确的 body 而不是静默失败】缺一个 asset 的包是最常见的
        // "打包漏了"，而空白响应会让它表现成"页面什么都不显示"。
        let body = format!("not found: {path}");
        let stream = gio::MemoryInputStream::from_bytes(&glib::Bytes::from(body.as_bytes()));
        request.finish(&stream, body.len() as i64, Some("text/plain"));
        return;
    };
    let Ok(bytes) = std::fs::read(&file) else {
        let body = format!("unreadable: {path}");
        let stream = gio::MemoryInputStream::from_bytes(&glib::Bytes::from(body.as_bytes()));
        request.finish(&stream, body.len() as i64, Some("text/plain"));
        return;
    };
    let mime = protocol::mime_for(&file);
    let stream = gio::MemoryInputStream::from_bytes(&glib::Bytes::from(&bytes));
    request.finish(&stream, bytes.len() as i64, Some(mime));
}

/// JS → 原生：只接受字符串，限长后交给 Policy。
fn install_messages(manager: &webkit2gtk::UserContentManager, client: &Rc<RefCell<EditorClient>>) {
    if !manager.register_script_message_handler("hfs") {
        crate::log_line("[gui] could not register the script message handler");
        return;
    }
    let weak = Rc::downgrade(client);
    manager.connect_script_message_received(None, move |_, result| {
        let Some(client) = weak.upgrade() else {
            return;
        };
        let Some(value) = result.js_value() else {
            return;
        };
        // 我们只接受字符串（见 protocol.rs 的理由）；结构化对象在这里被拒绝。
        if !value.is_string() {
            return;
        }
        let text = value.to_str();
        if text.len() > protocol::MAX_MESSAGE_BYTES {
            return;
        }
        client::on_message(&client, text.as_str());
    });
}

/// 导航白名单、新窗口、页面就绪、页面死亡。
fn install_policy(webview: &webkit2gtk::WebView, client: &Rc<RefCell<EditorClient>>) {
    webview.connect_decide_policy(|_, decision, kind| {
        if kind == webkit2gtk::PolicyDecisionType::NavigationAction {
            if let Some(navigation) =
                decision.downcast_ref::<webkit2gtk::NavigationPolicyDecision>()
            {
                let allowed = navigation
                    .request()
                    .and_then(|request| request.uri())
                    .is_some_and(|uri| protocol::is_allowed_navigation(&uri));
                if !allowed {
                    decision.ignore();
                    return true;
                }
            }
        }
        false
    });
    // 新窗口一律拒绝：本编辑器只有一个页面。
    webview.connect_create(|_, _| None::<gtk::Widget>);
    let weak = Rc::downgrade(client);
    webview.connect_load_changed(move |_, event| {
        if event == webkit2gtk::LoadEvent::Finished {
            if let Some(client) = weak.upgrade() {
                client.borrow_mut().on_page_ready();
            }
        }
    });
    // 【为什么"页面死亡"要直接标记】WebKitGTK 的 web process 崩溃之后所有投递都会
    // 静默失败，界面成了一张死图。继续假装可用只会让后续命令排进预算、把队列占满。
    let weak = Rc::downgrade(client);
    webview.connect_web_process_terminated(move |_, reason| {
        if let Some(client) = weak.upgrade() {
            client
                .borrow_mut()
                .mark_dead(&format!("the web process terminated ({reason:?})"));
        }
    });
    let weak = Rc::downgrade(client);
    webview.connect_load_failed(move |_, _, uri, error| {
        crate::log_line(&format!("[gui] page load failed for {uri}: {error}"));
        if let Some(client) = weak.upgrade() {
            client
                .borrow_mut()
                .mark_dead("the editor page could not be loaded");
        }
        false
    });
}

/// 真实文件拖入：把 URI 列表换成绝对路径，作为一条事件推给页面。
///
/// 【为什么不复制字节】与 Windows 的 `import_native_audio_file` 落点相同：页面收到
/// `native_drop` 之后调用既有的导入命令（参数是路径）。WebKit 的 `File` 没有磁盘
/// 路径，所以"路径从哪来"只能由原生侧回答。
fn install_drop(webview: &webkit2gtk::WebView, client: &Rc<RefCell<EditorClient>>) {
    let targets = [gtk::TargetEntry::new(
        "text/uri-list",
        gtk::TargetFlags::OTHER_APP,
        0,
    )];
    webview.drag_dest_set(gtk::DestDefaults::ALL, &targets, gtk::gdk::DragAction::COPY);
    let weak = Rc::downgrade(client);
    webview.connect_drag_data_received(move |_, _, _, _, data, _, _| {
        let Some(client) = weak.upgrade() else {
            return;
        };
        let paths = data
            .uris()
            .iter()
            .filter_map(|uri| gio::File::for_uri(uri).path())
            // 拖一个已删除的文件进来是常见操作；不存在就安静跳过，别报错。
            .filter(|path| path.is_file())
            .map(|path| path.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        if paths.is_empty() {
            return;
        }
        let view_id = client.borrow().view_id.clone();
        client.borrow_mut().post(serde_json::json!({
            "version": 1,
            "viewId": view_id,
            "event": "native_drop",
            "payload": { "paths": paths },
        }));
    });
}

/// 平台垫片：让 WebKit 假装成 WebView2 的那个 `chrome.webview`。
const BRIDGE: &str = include_str!("bridge.js");

// 每个拥有视图的线程各有一份：20 ms 节拍、跨线程命令、WebContext。
//
// 【为什么 WebContext 也在这里】它是 GObject（!Send），塞进 static OnceLock 会要求它
// Send + Sync —— 那等于对 GObject 撒谎。每线程一份正好也是"GTK 对象只属于一条线程"的写照。
thread_local! {
    static TICK_SOURCE: RefCell<Option<glib::SourceId>> = RefCell::new(None);
    static CONTROL: RefCell<Option<std::sync::mpsc::Receiver<Control>>> = RefCell::new(None);
    static CONTEXT: RefCell<Option<webkit2gtk::WebContext>> = RefCell::new(None);
}
