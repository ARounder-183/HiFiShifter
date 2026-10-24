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
    NavigationPolicyDecisionExt, PolicyDecisionExt, SecurityManagerExt, SettingsExt, URIRequestExt,
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

/// 手写几个 `gdkx.h` / `Xlib` 函数。
///
/// 【为什么手写】gdk 0.18 没有 `x11` 模块，而"把一个 GtkWindow 的 X 窗口塞进宿主窗口"
/// 只有这几个函数表达得了。它们不随 gtk-rs 版本变化，比升级整条依赖链便宜得多。
mod x11 {
    use std::ffi::c_void;

    pub type XWindow = libc::c_ulong;

    extern "C" {
        /// `Display *gdk_x11_get_default_xdisplay (void)`
        pub fn gdk_x11_get_default_xdisplay() -> *mut c_void;
        /// `Window gdk_x11_window_get_xid (GdkWindow*)`
        pub fn gdk_x11_window_get_xid(window: *mut c_void) -> XWindow;
        /// `void gdk_window_set_override_redirect (GdkWindow*, gboolean)`
        pub fn gdk_window_set_override_redirect(window: *mut c_void, override_redirect: i32);
        /// `int XReparentWindow (Display*, Window, Window, int, int)`
        pub fn XReparentWindow(
            display: *mut c_void,
            window: XWindow,
            parent: XWindow,
            x: i32,
            y: i32,
        ) -> i32;
        /// `int XMoveResizeWindow (Display*, Window, int, int, unsigned, unsigned)`
        pub fn XMoveResizeWindow(
            display: *mut c_void,
            window: XWindow,
            x: i32,
            y: i32,
            width: u32,
            height: u32,
        ) -> i32;
        /// `int XSync (Display*, Bool)`
        pub fn XSync(display: *mut c_void, discard: i32) -> i32;
        /// `Status XGetWindowAttributes (Display*, Window, XWindowAttributes*)`
        ///
        /// 只用到结构体开头的 `width/height`，所以按"足够大的不透明缓冲"接收 —— 这比
        /// 逐字段复刻 `XWindowAttributes`（后面还跟着 `Colormap`、`Visual*` 等）稳。
        pub fn XGetWindowAttributes(
            display: *mut c_void,
            window: XWindow,
            attributes: *mut c_void,
        ) -> i32;
    }
}

/// 手写 `cairo_surface_write_to_png`。
///
/// 【为什么手写】cairo 0.18 的 `Surface` 是个 struct，只有 `type_()`/`flush()`；
/// `write_to_png` 长在 `ImageSurface` 上，而 WebKit 的 snapshot 回调给的是通用
/// `Surface`。一个 `extern` 声明比"猜怎么转类型"便宜，也不会随 cairo-rs 版本漂移。
mod cairo_ffi {
    extern "C" {
        pub fn cairo_surface_write_to_png(
            surface: *mut std::ffi::c_void,
            filename: *const std::ffi::c_char,
        ) -> i32;
    }
}

static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

/// 只记一次"宿主回调不再做宿主侧工作"的说明。
static ONCE_HOST_TICK: std::sync::Once = std::sync::Once::new();

/// 回执泵已跑的节拍数（见 §tick§ 里的说明）。
static TICKS: AtomicU64 = AtomicU64::new(0);
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
    /// 嵌进宿主时我们自己的 X 窗口 id（§0§ = 浮窗）。resize 要直接对它发 X 请求。
    embedded_xid: u64,
    /// 泵是否就跑在**宿主 UI 线程**上（情形 A）。
    ///
    /// 【为什么必须区分】§client::tick§ 会碰宿主 API（§flush_dirty§ / 宿主时钟 /
    /// 让路状态机），只有宿主线程能调；情形 B 的节拍在私有 GTK 线程上，只能
    /// §client::pump§（纯投递）。搞反了是"在错误线程上调宿主"—— 最难查的一类。
    on_host_thread: bool,
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
                    crate::gui_log_line(&format!("page delivery failed: {error}"));
                }
            },
        );
        true
    }

    fn show_error(&self, text: &str) {
        crate::gui_log_line(&format!("Native editor error: {text}"));
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
    #[allow(dead_code)]
    fn forward_edit_key(&self, _key: char, _mods: i16, _down: bool) -> bool {
        false
    }

    fn host_tick(&self) {
        // ── 这里什么都不做，是刻意的 ─────────────────────────────────────────
        //
        // 这个函数被 `onSize` / `onFocus` / `setFrame` 调用 —— 也就是说它跑在
        // **宿主的 UI 线程上、宿主自己的回调栈里**。此前它会做一整轮宿主侧工作
        // （`flush_dirty`、宿主时钟、私有状态通知），那些都会回头调宿主的 API。
        // 在宿主的回调栈里回调宿主，等于把一条本来就很脆的链子再套一圈：宿主
        // 如果在那条 API 里同步等我们另一条线程（GTK/WebKit），两边就互等到死。
        //
        // 实测症状正是"宿主整体冻住、我们的界面照常响应、CPU 只有 13%"。
        //
        // 现在宿主线程上只做**记录**，工作留给回执泵。代价是宿主侧的 flush 晚一个
        // 节拍（20 ms 量级），换来的是宿主永远不会被我们的回调链卡住。
        ONCE_HOST_TICK.call_once(|| {
            crate::gui_log_line("host callbacks no longer run host-side work; the pump owns it");
        });
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
    /// 只在情形 B：建视图并回报 §(token, 实际做到的保真度)§。
    Attach {
        request: AttachRequest,
        reply: SyncSender<Result<(usize, GuiFidelity), String>>,
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

/// 渲染进程（`hfs-gui`）的宿主侧代理。
///
/// 【为什么整块都没有 GTK】实测：**只要在 REAPER 的进程里创建一个 GTK 窗口**，那个
/// 进程的绘制就整个废掉 —— 判据是 `xwd -id <REAPER 主窗口>` 从 3.1 MB 变成 0 字节，
/// 连 REAPER 自己的对话框都变黑、宿主 UI 线程空转、输入全无。窗口里只放一个
/// `gtk::Label`（完全没有 WebKit）也照样复现。
///
/// 所以这里一行 GTK 都不碰：策略（`EditorClient`）跑在**我们自己的线程**上，
/// 页面在**另一个进程**里，两者用管道上的 JSON 行对话。
mod remote {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::sync::mpsc::{channel, Receiver, Sender};

    /// 宿主线程 → 驱动线程。
    pub(super) enum Command {
        Resize(i32, i32),
        Focus,
        Close,
    }

    /// 渲染进程的出站端口：把一条已序列化的 JSON 交给页面。
    struct RemotePort {
        out: Sender<String>,
    }

    impl EditorPort for RemotePort {
        fn post_message(&self, text: &str) -> bool {
            let expression = protocol::js_expression(text);
            self.out
                .send(serde_json::json!({"type":"eval","js":expression}).to_string())
                .is_ok()
        }

        fn show_error(&self, text: &str) {
            crate::gui_log_line(&format!("Native editor error: {text}"));
            let text = text.replace('\\', " ").replace('\'', " ");
            let script = format!(
                "document.title = 'HiFiShifter editor: {text}'; if (document.body) {{ document.body.textContent = 'HiFiShifter editor: {text}'; }}"
            );
            let _ = self
                .out
                .send(serde_json::json!({"type":"eval","js":script}).to_string());
        }

        fn close(&self) {
            let _ = self
                .out
                .send(serde_json::json!({"type":"close"}).to_string());
        }
    }

    /// 宿主持有的句柄：只有通道与身份，可跨线程。
    pub(super) struct RemoteHost {
        pub(super) control: Sender<Command>,
        pub(super) fidelity: GuiFidelity,
    }

    impl EditorHost for RemoteHost {
        fn fidelity(&self) -> GuiFidelity {
            self.fidelity
        }

        fn resize(&self, width: i32, height: i32) -> Result<(), String> {
            self.control
                .send(Command::Resize(width, height))
                .map_err(|_| "the editor process is gone".to_string())
        }

        fn focus(&self) -> Result<(), String> {
            self.control
                .send(Command::Focus)
                .map_err(|_| "the editor process is gone".to_string())
        }

        fn forward_edit_key(&self, _key: char, _mods: i16, _down: bool) -> bool {
            false
        }

        fn host_tick(&self) {}

        fn close(&self) {
            let _ = self.control.send(Command::Close);
        }
    }

    /// 找 `hfs-gui`：环境变量优先，其次与 `HiFiShifter.so` 同目录。
    ///
    /// 【为什么不用 `current_exe`】插件是加载进宿主的，`current_exe` 返回的是 REAPER
    /// 的路径。我们的 .so 路径只能从 `/proc/self/maps` 里找。
    pub(super) fn helper_path() -> Option<std::path::PathBuf> {
        if let Some(path) = std::env::var_os("HIFISHIFTER_GUI_HELPER") {
            return Some(path.into());
        }
        let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
        for line in maps.lines() {
            let Some(path) = line.split_whitespace().last() else {
                continue;
            };
            if !path.contains("HiFiShifter.vst3") || !path.ends_with(".so") {
                continue;
            }
            let dir = std::path::Path::new(path).parent()?;
            for name in ["hfs-gui", "hfs-gui.bin"] {
                let candidate = dir.join(name);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
        None
    }

    pub(super) fn attach(request: AttachRequest) -> Result<Rc<dyn EditorHost>, String> {
        let assets = crate::editor::resources::frontend_directory()?;
        let helper = helper_path()
            .ok_or_else(|| "the editor process (hfs-gui) is not next to the plug-in".to_string())?;
        let (commands, incoming) = channel::<Command>();
        let link = request.link.clone();
        let (parent, width, height) = (request.parent, request.width, request.height);

        std::thread::Builder::new()
            .name("hfs-editor".into())
            .spawn(move || driver(link, helper, assets, parent, width, height, incoming))
            .map_err(|error| format!("could not start the editor thread: {error}"))?;
        // 其余通道都归驱动线程；这里立刻返回，宿主不会等任何东西。

        crate::gui_log_line("editor runs out of process (hfs-gui)");
        Ok(Rc::new(RemoteHost {
            control: commands,
            fidelity: GuiFidelity::Embedded,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn driver(
        link: Arc<crate::editor::routing::EditorLink>,
        helper: std::path::PathBuf,
        assets: std::path::PathBuf,
        parent: NativeWindow,
        width: i32,
        height: i32,
        commands: Receiver<Command>,
    ) {
        let (events_tx, events) = channel::<serde_json::Value>();
        let (out_tx, out_rx) = channel::<String>();
        let client = EditorClient::open(link);
        let boot = {
            let state = client.borrow();
            client::bootstrap_json(&state.link, &state.view_id)
        };
        let boot_path = std::env::temp_dir().join(format!("hfs-boot-{}.json", std::process::id()));
        if let Err(error) = std::fs::write(&boot_path, boot.to_string()) {
            crate::gui_log_line(&format!("could not write the bootstrap file: {error}"));
            client
                .borrow_mut()
                .mark_dead("the editor bootstrap could not be written");
            return;
        }
        let mut child = match std::process::Command::new(&helper)
            .arg("--parent")
            .arg(format!("0x{parent:x}"))
            .arg("--width")
            .arg(width.to_string())
            .arg("--height")
            .arg(height.to_string())
            .arg("--root")
            .arg(&assets)
            .arg("--bootstrap-file")
            .arg(&boot_path)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(error) => {
                crate::gui_log_line(&format!("could not start {helper:?}: {error}"));
                client
                    .borrow_mut()
                    .mark_dead("the editor process could not be started");
                return;
            }
        };
        let Some(stdin) = child.stdin.take() else {
            return;
        };
        let Some(stdout) = child.stdout.take() else {
            return;
        };
        {
            let mut stdin = stdin;
            std::thread::Builder::new()
                .name("hfs-editor-in".into())
                .spawn(move || {
                    for line in out_rx {
                        if writeln!(stdin, "{line}").is_err() || stdin.flush().is_err() {
                            break;
                        }
                    }
                })
                .ok();
        }
        {
            std::thread::Builder::new()
                .name("hfs-editor-out".into())
                .spawn(move || {
                    for line in BufReader::new(stdout).lines() {
                        let Ok(line) = line else { break };
                        let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
                            continue;
                        };
                        if events_tx.send(value).is_err() {
                            break;
                        }
                    }
                })
                .ok();
        }

        let port: Rc<dyn EditorPort> = Rc::new(RemotePort {
            out: out_tx.clone(),
        });
        client.borrow_mut().set_port(port);

        let mut closed = false;
        while !closed {
            for value in events.try_iter() {
                match value["type"].as_str().unwrap_or_default() {
                    "message" => {
                        if let Some(text) = value["text"].as_str() {
                            if text.len() <= protocol::MAX_MESSAGE_BYTES {
                                client::on_message(&client, text);
                            }
                        }
                    }
                    "page-ready" => client.borrow_mut().on_page_ready(),
                    "ready" => crate::gui_log_line(&format!(
                        "the editor process embedded its window ({})",
                        value["xid"]
                    )),
                    "error" => {
                        let text = value["text"].as_str().unwrap_or("unknown error");
                        crate::gui_log_line(&format!("the editor process reported: {text}"));
                        client.borrow_mut().mark_dead(text);
                    }
                    _ => {}
                }
            }
            while let Ok(command) = commands.try_recv() {
                match command {
                    Command::Resize(width, height) => {
                        let _ = out_tx.send(
                            serde_json::json!({"type":"resize","width":width,"height":height})
                                .to_string(),
                        );
                    }
                    Command::Focus => {
                        let _ = out_tx.send(serde_json::json!({"type":"focus"}).to_string());
                    }
                    Command::Close => {
                        let _ = out_tx.send(serde_json::json!({"type":"close"}).to_string());
                        closed = true;
                    }
                }
            }
            client::pump(&client);
            if client.borrow().is_dead() {
                break;
            }
            if let Ok(Some(_)) = child.try_wait() {
                client.borrow_mut().mark_dead("the editor process exited");
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = child.kill();
        let _ = child.wait();
        let _ = std::fs::remove_file(&boot_path);
    }
}

pub(super) fn create(
    parent: NativeWindow,
    width: i32,
    height: i32,
    embedded: bool,
    frame: usize,
    link: Arc<crate::editor::routing::EditorLink>,
) -> Result<Rc<dyn EditorHost>, String> {
    let request = AttachRequest {
        parent,
        width,
        height,
        embedded,
        link: link.clone(),
    };
    // ── 首选：渲染进程 ───────────────────────────────────────────────────────
    //
    // 宿主进程里**一个 GTK 符号都不碰**（见 `remote` 顶部的实测说明）。
    if !std::env::var("HIFISHIFTER_INPROCESS_GUI").is_ok_and(|v| v.trim() == "1") {
        match remote::attach(request) {
            Ok(host) => return Ok(host),
            Err(error) => crate::gui_log_line(&format!(
                "the out-of-process editor is unavailable ({error}); falling back in-process"
            )),
        }
    }
    let request = AttachRequest {
        parent,
        width,
        height,
        embedded,
        link: link.clone(),
    };
    // 宿主是否提供 `Linux::IRunLoop`（"正规解"，见方案 `10.3）。
    //
    // 【为什么现在只探测、不注册】IRunLoop 让 GTK 跑在**宿主线程**上、由宿主的
    // 定时器驱动 —— 但它要求我们同时把 Gdk 的 X 连接 fd 注册给宿主（否则 GTK 收不到
    // 输入，窗口会"画一次就冻住"）。这两步里任何一步在真宿主上不如预期，代价都是
    // 一个**冻住的界面**，而现在的情形 B（私有 GTK 线程 + 浮窗）虽然降级但确实能用。
    // 本机没有任何提供 IRunLoop 的宿主可以验证，所以这里只把"宿主有没有这条路"
    // 记进日志 —— 报障日志里有这一行，下一次就能拿着真实证据决定要不要翻这个开关。
    //
    // 【抄错 IID 会怎样】只会让探测恒为 §false§，也就是永远走情形 B —— 不会崩、
    // 不会静默改行为。这条探测是纯只读的：拿到接口立刻释放，不注册任何东西。
    let run_loop = probe_run_loop(frame);
    crate::gui_log_line(&format!(
        "host IRunLoop: {} (frame={}, embedded={embedded})",
        if run_loop { "available" } else { "absent" },
        if frame == 0 { "unset" } else { "set" }
    ));
    // ── 默认**不**借用宿主的 GTK 主循环 ──────────────────────────────────────
    //
    // 【为什么把这个开关反过来】实测：只要编辑器窗口一出现，宿主就整体冻住 ——
    // CPU 13%、状态 `S`（在等不是在算）、我们的回执泵照常 20 ms 一拍、没有任何
    // 慢调用，而且**浮窗形态也一样**（所以不是窗口搬运的问题）。
    //
    // 剩下的共同因素只有一个：GTK/WebKit 跑在**宿主的 UI 线程**上，于是宿主的
    // 消息泵被 GTK 的事件源接管，再也回不到它自己的输入处理。
    //
    // 所以默认改成情形 B：我们自己起一条 GTK 线程、自己的主循环，宿主的线程
    // 一根手指都不碰。`HIFISHIFTER_GUI_HOST_LOOP=1` 可以退回情形 A。
    let host_loop =
        std::env::var("HIFISHIFTER_GUI_HOST_LOOP").is_ok_and(|value| value.trim() == "1");
    if gtk3_initialized() && host_loop {
        // 情形 A：宿主的 GTK 已经在这条线程上跑着。
        //
        // 【为什么这里什么都不用做】本 crate 打开了 gtk 的
        // `unsafe-assume-initialized`：gtk-rs 因此认为"GTK 已在本线程初始化"，
        // `assert_initialized_main_thread!` 全部放行。代价是它不再替我们守线程纪律 ——
        // 由 `VIEWS`（thread_local）与 `LinuxHost::submit`（先比线程身份）来守。
        let (token, fidelity) = attach_view(request, true)?;
        crate::gui_log_line(&format!(
            "hosting the host's GTK main loop ({})",
            fidelity.as_str()
        ));
        return Ok(Rc::new(LinuxHost {
            token,
            owner: std::thread::current().id(),
            control: None,
            fidelity,
            link,
        }));
    }
    // 情形 B：自己起一条 GTK 线程与主循环。
    //
    // 【注意这里是一个赌注】宿主可能已经初始化过 GTK（`gtk3_initialized()` 为真）。
    // `gtk_init_check` 在那种情况下直接返回 TRUE，而我们随后在**自己的线程**上建
    // widget —— 严格说是 UB。但两害相权：借用宿主的主循环会让宿主的消息泵再也
    // 回不到自己的输入处理（实测整体冻住），而这条路的代价只是"可能不稳"。
    // 一旦确认稳定，应当反过来把宿主的 GTK 换成 XEMBED 之外的方案。
    if gtk3_initialized() {
        crate::gui_log_line(
            "the host had already initialised GTK; running the editor on a private thread anyway",
        );
    }
    //
    // 【为什么先把 GDK 钉到 X11】宿主给的是一个 **X11 Window ID**
    // （`kPlatformTypeX11EmbedWindowID`）。而 GDK 在 `WAYLAND_DISPLAY` 存在时
    // **默认选 Wayland** —— WSLg 与 GNOME Wayland 会话下都是这样，实测
    // `gdk_display_get_default()` 返回 `GdkWaylandDisplay`。Wayland 后端上没有
    // "外来窗口"这个概念，嵌入必然失败；而且失败得**非常安静**：
    // `gtk_plug_new` 会返回一个非空、永不映射、不报错的 widget —— 用户看到的就是
    // 一块空面板。只有在"我们自己初始化 GTK"时才能改这个选择；宿主已经初始化过的
    // 情形 A 改也没用，那时走浮窗（`attach_view` 会自报 `FloatingWindow`）。
    if request.embedded && std::env::var_os("GDK_BACKEND").is_none() {
        std::env::set_var("GDK_BACKEND", "x11");
        crate::gui_log_line("pinning GDK_BACKEND=x11 so the host's X11 window can be embedded");
    }
    //
    // 【为什么这里不返回 §Err§】"这台机器上没有可用的 GTK/WebKit"是一个**环境事实**，
    // 不是一个错误：宿主应当看到空面板 + 一行可搜到的日志，而不是一个没人读的字符串。
    // 真正的编程错误（父窗口为空、线程身份不对）仍然走 §Err§。
    let thread = match gui_thread() {
        Ok(thread) => thread,
        Err(_) => {
            crate::gui_log_line(
                "no usable GTK/WebKit in this process; the editor panel stays empty",
            );
            return Ok(Rc::new(super::unavailable::UnavailableHost::new(
                "no usable GTK/WebKit in this host",
            )));
        }
    };
    let (reply, replies) = std::sync::mpsc::sync_channel(1);
    thread
        .control
        .send(Control::Attach { request, reply })
        .map_err(|_| "the GTK thread is gone".to_string())?;
    match replies.recv_timeout(ATTACH_TIMEOUT) {
        Ok(Ok((token, fidelity))) => {
            crate::gui_log_line(&format!(
                "hosting our own GTK thread ({})",
                fidelity.as_str()
            ));
            Ok(Rc::new(LinuxHost {
                token,
                owner: thread.owner,
                control: Some(thread.control.clone()),
                fidelity,
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
        crate::gui_log_line("ExitDll refused: native editor views still alive");
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
        crate::gui_log_line("ExitDll: the GTK thread did not stop in time; abandoning it");
    }
    // 句柄直接丢掉（线程分离），见上面的理由。
    let _ = thread
        .handle
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take();
    true
}

/// **GTK 3** 是否已经在本进程里初始化过。
///
/// 【为什么必须区分 GTK2 与 GTK3】`gdk_display_get_default` 这个符号 **GTK2 也有**
/// （`libgdk-x11-2.0.so.0`）。而 Linux 上的宿主（REAPER 的 SWELL、Ardour 的插件窗口）
/// 很可能已经装载了 GTK2 并打开了它的 display —— 于是朴素的 `dlsym` 探测会说
/// "GDK 已初始化"，我们便跳过自己的 `gtk_init_check`，接着用 **GTK3** 的
/// `gtk_widget_*` 去操作一个**从未初始化过**的工具箱。那不会报错，只会产出一个
/// 不映射的窗口 —— 也就是"空面板、日志里什么都没有"。
///
/// 判据因此收紧成两步：先 `dlsym` 找到符号，再用 `dladdr` 问"它是哪个库提供的"，
/// 要求路径里是 `libgtk-3` / `libgdk-3`。
///
/// 【为什么用 `dlopen(NULL)` + `dlsym` 而不是链接期符号】我们**不能**为了探测去
/// 初始化 GTK（那正是要判断的事）；`RTLD_DEFAULT` 只查"已经装载的模块"，
/// 不加载任何东西。
fn gtk3_initialized() -> bool {
    type GetDefault = unsafe extern "C" fn() -> *mut std::ffi::c_void;
    // SAFETY: dlopen/dlsym/dladdr 只读进程的模块表；符号名与签名由 GDK 3 固定
    // （`GdkDisplay *gdk_display_get_default(void)`）。
    unsafe {
        let handle = libc::dlopen(std::ptr::null(), libc::RTLD_LAZY);
        if handle.is_null() {
            return false;
        }
        let symbol = libc::dlsym(handle, c"gdk_display_get_default".as_ptr());
        if symbol.is_null() {
            libc::dlclose(handle);
            return false;
        }
        let mut info = std::mem::MaybeUninit::<libc::Dl_info>::uninit();
        let located = libc::dladdr(symbol, info.as_mut_ptr());
        libc::dlclose(handle);
        if located == 0 {
            return false;
        }
        let info = info.assume_init();
        if info.dli_fname.is_null() {
            return false;
        }
        let library = std::ffi::CStr::from_ptr(info.dli_fname)
            .to_string_lossy()
            .into_owned();
        let is_gtk3 = library.contains("libgtk-3") || library.contains("libgdk-3");
        if !is_gtk3 {
            crate::gui_log_line(&format!(
                "a GDK from {library} is loaded, but that is not GTK 3; initializing our own"
            ));
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
                    crate::gui_log_line(&format!("GTK thread unavailable: {error}"));
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
            // 【为什么不调 `gtk::init()`】本 crate 打开了 `unsafe-assume-initialized`，
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
                if view.embedded_xid != 0 {
                    // 【为什么还要直接发 X 请求】被 reparent 之后窗口不再由 WM 摆放，
                    // GTK 的 resize 只更新它自己的记录；尺寸真正生效要靠 X。
                    // SAFETY: embedded_xid 是本进程在同一个 display 上创建的窗口。
                    unsafe {
                        let dpy = x11::gdk_x11_get_default_xdisplay();
                        x11::XMoveResizeWindow(
                            dpy,
                            view.embedded_xid,
                            0,
                            0,
                            width as u32,
                            height as u32,
                        );
                        x11::XSync(dpy, 0);
                    }
                }
            }
        }),
        Control::Focus { token } => VIEWS.with(|views| {
            if let Some(view) = views.borrow().get(&token) {
                view.webview.grab_focus();
            }
        }),
        Control::Close { token } => close_view(token),
        Control::Attach { request, reply } => {
            let _ = reply.send(attach_view(request, false));
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
/// 看门狗：任何一个跑在**宿主 UI 线程**上的入口超过这个时长，就落一行 WARN。
///
/// 【为什么需要它】"宿主卡住"最难受的地方是无法归因：CPU 不高、X 也没被 grab，
/// 但宿主就是不刷新。唯一能把"我们挡住了它"和"它在等别的东西"分开的，是**我们
/// 自己每个入口的耗时**。50 ms 已经远超一次正常回执的成本。
const SLOW_ENTRY: Duration = Duration::from_millis(50);

fn tick() -> glib::ControlFlow {
    // 【为什么要数节拍】宿主线程上的 `g_timeout_add` **只在宿主迭代 GTK 主循环时才跑**。
    // 宿主迭代得慢，回执泵就慢，前端就会报 "Backend invoke failed"、手感像卡死 ——
    // 而这个数字是唯一能把"宿主没在转"和"我们算得慢"分开的证据。
    let ticks = TICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let started = std::time::Instant::now();
    if ticks == 250 || ticks == 1000 {
        crate::gui_log_line(&format!(
            "pump: {ticks} ticks (~{} ms of wall clock at 20 ms)",
            ticks * 20
        ));
    }
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
            .map(|view| (view.client.clone(), view.on_host_thread))
            .collect::<Vec<_>>()
    });
    for (client, on_host_thread) in clients {
        if on_host_thread {
            // 情形 A：节拍就在宿主线程上，于是宿主侧的事（私有状态通知、宿主时钟、
            // 让路状态机）能在这里做 —— 与 Windows 的 §WM_TIMER§、macOS 的 §NSTimer§ 等价。
            client::tick(&client);
        } else {
            // 情形 B：这里不是宿主线程，只推进投递；宿主侧由 §LinuxHost::host_tick§ 补做。
            client::pump(&client);
        }
        if client.borrow().is_dead() {
            let dead = VIEWS.with(|views| {
                views
                    .borrow()
                    .iter()
                    .find(|(_, view)| Rc::ptr_eq(&view.client, &client))
                    .map(|(token, _)| *token)
            });
            if let Some(token) = dead {
                crate::gui_log_line("closing the editor view: the page is gone");
                close_view(token);
            }
        }
    }
    // 【为什么在这里也量】回执泵跑在宿主 UI 线程上：它一旦变慢，宿主就跟着卡。
    let elapsed = started.elapsed();
    if elapsed > SLOW_ENTRY {
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            "gui.slow.pump",
            format_args!("[gui] slow entry: pump took {} ms", elapsed.as_millis()),
        );
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
    // 【为什么要显式 destroy】只 drop Rust 包装只是减一次引用；只要 GTK 侧还有一份，
    // 这个 override-redirect 窗口就会留在合成器里 —— 一块盖在宿主上的黑矩形，
    // 而且会吃掉点击。必须确定地销毁。
    if let Some(window) = view.container.downcast_ref::<gtk::Window>() {
        // SAFETY: 这个窗口只由我们持有，销毁发生在摘掉注册表之后。
        unsafe { window.destroy() };
    }
    view.webview.stop_loading();
    // SAFETY: 容器由本视图独占；GTK 的 destroy 只要求在同一线程上调用。
    unsafe { view.container.destroy() };
    LIVE_VIEWS.fetch_sub(1, Ordering::AcqRel);
}

// ── Steinberg::Linux::IRunLoop 的探测 ───────────────────────────────────────
//
// 【为什么手写】本仓的 vst3sdk 是 sparse checkout（只有 `pluginterfaces/{base,gui,vst}`），
// 里面**没有** `base/irunloop.h`。手写 30 行比去改 Windows 的 SDK 拉取脚本便宜得多，
// 也与 `vst3.rs` / `view.rs` 手写 `iplugview` 的风格一致。
//
// 【抄错的后果是什么】只有一个：`queryInterface` 返回失败 ⇒ 探测恒为 `false` ⇒
// 永远走情形 B（现状）。不会崩、不会改变任何行为 —— 这正是"先只探测"的前提。

/// `IRunLoop` 的 IID：`18C35366-9776-4F1A-9C5B-83857A871389`
/// （`DECLARE_CLASS_IID (IRunLoop, 0x18C35366, 0x97764F1A, 0x9C5B8385, 0x7A871389)`）。
const IID_RUN_LOOP: [u32; 4] = [0x18C35366, 0x97764F1A, 0x9C5B8385, 0x7A871389];

/// `FUnknown` 的三项（`IPlugFrame` 与 `IRunLoop` 都从它派生，vtable 前缀一致）。
#[repr(C)]
struct UnknownVtbl {
    query: unsafe extern "system" fn(
        *mut std::ffi::c_void,
        *const u8,
        *mut *mut std::ffi::c_void,
    ) -> i32,
    add: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
    release: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
}

/// 宿主给的 `IPlugFrame*` 上有没有 `Linux::IRunLoop`。
///
/// 【为什么拿到就立刻 `release`】探测不该改变任何引用计数；宿主在 `attached` 之后
/// 可能还会换 frame，留着一个引用只会让"谁持有它"变得说不清。
fn probe_run_loop(frame: usize) -> bool {
    if frame == 0 {
        return false;
    }
    // SAFETY: frame 由宿主在 `IPlugFrame::queryInterface` 的契约下给出（本次调用期间
    // 有效）。我们只读它的 vtable 前两项，拿到结果立刻 release。
    unsafe {
        let object = frame as *mut std::ffi::c_void;
        let vtbl = *(frame as *const *const UnknownVtbl);
        if vtbl.is_null() {
            return false;
        }
        let mut out: *mut std::ffi::c_void = std::ptr::null_mut();
        let result = ((*vtbl).query)(object, IID_RUN_LOOP.as_ptr().cast(), &mut out);
        if result != 0 || out.is_null() {
            return false;
        }
        let vtbl = *(out as *const *const UnknownVtbl);
        if !vtbl.is_null() {
            ((*vtbl).release)(out);
        }
        true
    }
}

/// 在拥有视图的线程上真正建视图。
fn attach_view(
    request: AttachRequest,
    on_host_thread: bool,
) -> Result<(usize, GuiFidelity), String> {
    let assets = crate::editor::resources::frontend_directory()?;
    let context = web_context(&assets);
    let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed) as usize;

    let webview = webkit2gtk::WebView::with_context(&context);
    // ── 渲染后端：默认**关掉**加速合成 ───────────────────────────────────────
    //
    // 【为什么默认关】插件 UI 跑在宿主进程里，宿主的 GL/EGL 状态不是我们能假设的。
    // 现场证据（Linux REAPER + WSLg）：打开插件窗口时 WebKit 的 EGL 初始化失败 ——
    //
    //     libEGL warning: DRI3 error: Could not get DRI3 device
    //     libEGL warning: Ensure your X server supports DRI3 to get accelerated rendering
    //
    // 加速合成拿不到 DRI3 时的失败形态**不是报错，是黑屏**：页面照样 load-finished、
    // 日志全绿，屏幕上一片黑。而关掉它（CPU 合成）在同样的环境里画得好好的 ——
    // 独立探针里同一个 bundle 抓出 1306 种颜色、可辨认的工具栏布局。
    //
    // 【为什么不干脆全平台都关】有可用 GPU 时加速合成本来更省电也更顺滑；所以
    // `HIFISHIFTER_WEBKIT_ACCEL=1` 可以把它打开，日志里会写明用的是哪一种。
    let accelerate =
        std::env::var("HIFISHIFTER_WEBKIT_ACCEL").is_ok_and(|value| value.trim() == "1");
    if let Some(settings) = WebViewExt::settings(&webview) {
        settings.set_hardware_acceleration_policy(if accelerate {
            webkit2gtk::HardwareAccelerationPolicy::OnDemand
        } else {
            webkit2gtk::HardwareAccelerationPolicy::Never
        });
        crate::gui_log_line(&format!(
            "webkit hardware acceleration: {}",
            if accelerate {
                "on-demand"
            } else {
                "never (CPU compositing)"
            }
        ));
    }
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

    // ── 把视图塞进宿主给的窗口（或者只能自己开窗）──────────────────────────
    //
    // 【为什么既不是 `GtkPlug` 也不是 `set_parent_window`】两条路都实测过：
    //   * `set_parent_window`：GTK3 拒绝 realize（`widget->priv->anchored` 断言），
    //     widget 永远拿不到 GdkWindow；
    //   * `GtkPlug`：能映射、X 侧一切正常（`IsViewable`、尺寸正确、父子关系正确），
    //     但 **WebKit 在里面渲染不出来** —— 用 `webkit_web_view_get_snapshot` 一测就现形：
    //     toplevel 里 `snapshot OK`，GtkPlug 里 `snapshot FAILED: There was an error
    //     creating the snapshot`。屏幕上是黑的，而所有别的判据全绿。
    //
    // 正确的做法是**让 GTK 以为自己在管一个普通 toplevel**，再在 X 层把那个窗口搬进
    // 宿主窗口：`GtkWindow(override-redirect)` + `XReparentWindow`。实测：
    // `snapshot OK`、`XQueryTree(host)` 里能看到我们的 XID、WebKit 正常出图。
    let display =
        gdk::Display::default().ok_or_else(|| "GTK has no default display".to_string())?;
    let display_kind = display.type_().name().to_string();
    // ── 默认**不**嵌入 ────────────────────────────────────────────────────────
    //
    // 【为什么默认退回浮窗】把我们的窗口 `XReparentWindow` 进宿主的窗口，在
    // XWayland + 合成器（WSLg 的 Weston）下会把合成器场景弄坏：宿主从此不重绘
    // （拖动时满屏黑块），而且输入也没了 —— 实测宿主进程 CPU 只有 13%、状态 `S`
    // （在等，不是在算），我们自己的回执泵照常在跑、没有任何慢调用。
    //
    // 也就是说：卡的不是我们的线程，是**别的东西被我们弄坏了**。在这条路被证明
    // 可靠之前，默认走浮窗（GTK 的普通 toplevel，交给窗口管理器管，合成器不会乱），
    // 嵌入留给 `HIFISHIFTER_GUI_EMBED=1` 显式开启。
    let embed_requested =
        std::env::var("HIFISHIFTER_GUI_EMBED").is_ok_and(|value| value.trim() == "1");
    let embedded_ok = request.embedded && display_kind.contains("X11") && embed_requested;
    if request.embedded && !embed_requested {
        crate::gui_log_line(
            "opening the editor as a floating window (set HIFISHIFTER_GUI_EMBED=1 to embed)",
        );
    }
    let fidelity = if embedded_ok {
        GuiFidelity::Embedded
    } else {
        GuiFidelity::FloatingWindow
    };
    // 【为什么用宿主窗口的几何而不是 `request.width/height`】宿主给的 `ViewRect`
    // 可以是 0x0（"你看着办"），而我们默认的 1100x720 只是"一个够大的窗口"。两者都会
    // 让我们**盖不满**宿主留出的那块区域 —— 屏幕上就是一圈黑边。父窗口的实际尺寸才是
    // 我们要填满的东西。
    let mut width = request.width;
    let mut height = request.height;
    if embedded_ok {
        // SAFETY: parent 是宿主给的有效 XID；缓冲区按 `XWindowAttributes` 的真实大小开。
        unsafe {
            let mut attributes = [0u8; 256];
            let dpy = x11::gdk_x11_get_default_xdisplay();
            if x11::XGetWindowAttributes(
                dpy,
                request.parent as x11::XWindow,
                attributes.as_mut_ptr().cast(),
            ) != 0
            {
                // `XWindowAttributes` 的第 4、5 个字段是 `int width, height`（前面是
                // x, y, width, height —— 各 4 字节，偏移 8 和 12）。
                let read = |offset: usize| {
                    i32::from_ne_bytes([
                        attributes[offset],
                        attributes[offset + 1],
                        attributes[offset + 2],
                        attributes[offset + 3],
                    ])
                };
                let (host_width, host_height) = (read(8), read(12));
                if host_width > 0 && host_height > 0 {
                    width = host_width as i32;
                    height = host_height as i32;
                }
            }
        }
    }
    let window = gtk::Window::new(gtk::WindowType::Toplevel);
    window.set_decorated(false);
    window.set_title("HiFiShifter");
    // 【为什么关掉焦点】我们的窗口只是宿主窗口里的一块画布，不是独立的应用窗口。
    // 让它抢焦点会把手感变成"点了插件之后宿主不再响应键盘"。
    window.set_accept_focus(false);
    window.set_focus_on_map(false);
    // 【已停用：见下方 probe】`accept-focus` 只管"映射时要不要焦点"。我们这块画布在
    // GTK 眼里仍是 toplevel：用户在里面点一下，GTK 会走 `gtk_window_present` ——
    // 向窗口管理器请求激活、把输入焦点从宿主手里拿走。Weston/XWayland 照办之后，
    // 宿主就"点不动了"（而且它自己的重绘也停摆，拖窗口时一片黑）。
    //
    // `Dock` 提示 + 跳过任务栏/分页器，让窗口管理器从一开始就不把这块当应用窗口看。
    // 【为什么全部去掉】这三个（Dock 类型提示 + 跳过任务栏/分页器）是"让窗口管理器
    // 别管我们"的尝试。实测它们会把 WSLg 的合成器场景弄坏：从那一刻起**所有**窗口
    // 都取不到内容（xwd 对 REAPER 自己的窗口也返回 0 字节），宿主自己的对话框变成
    // 黑块，整个进程像冻住一样。窗口保持最朴素的形态。
    window.set_default_size(width, height);
    // 【探针】HIFISHIFTER_NO_WEBKIT=1 时窗口里只放一个标签，完全不创建 WebKit ——
    // 用来判定"整个进程画不出东西"是 GTK 窗口本身，还是 WebKit。
    let no_webkit = std::env::var("HIFISHIFTER_NO_WEBKIT").is_ok_and(|v| v.trim() == "1");
    if no_webkit {
        crate::gui_log_line("webkit disabled by HIFISHIFTER_NO_WEBKIT (plain GTK window)");
        window.add(&gtk::Label::new(Some("HiFiShifter (no webkit)")));
    } else {
        window.add(&webview);
    }
    let mut embedded_xid = 0u64;
    if embedded_ok {
        // 先 realize 拿到 GdkWindow，再标 override-redirect（WM 就不会去管它），
        // 最后 show 并把 X 窗口搬进宿主。
        window.realize();
        if let Some(gdk_window) = window.window() {
            use gtk::glib::translate::ToGlibPtr;
            let raw: *mut gdk::ffi::GdkWindow = gdk_window.to_glib_none().0;
            // SAFETY: raw 是本窗口的 GdkWindow*；两个调用都只读写它自己的状态。
            unsafe {
                x11::gdk_window_set_override_redirect(raw.cast(), 1);
                embedded_xid = x11::gdk_x11_window_get_xid(raw.cast());
            }
        }
        // 【为什么必须在 `show_all` **之前** reparent】先 `show_all` 再搬，窗口会以
        // "override-redirect 的独立窗口"身份在合成器里存在一小会儿。Weston 记下它，
        // 之后它被搬进宿主，WM 的场景就永久错位 —— 现象正是用户报的：宿主自己的窗口
        // 大块变黑、拖一下满屏黑块、点哪里都没反应。
        //
        // 未映射的窗口照样可以 `XReparentWindow`；搬完再 map，窗口管理器从头到尾
        // 都没见过它。
        if embedded_xid != 0 {
            // SAFETY: 两个 XID 都来自同一个 display 连接，且宿主窗口在 attached 期间有效。
            unsafe {
                let dpy = x11::gdk_x11_get_default_xdisplay();
                x11::XReparentWindow(dpy, embedded_xid, request.parent as x11::XWindow, 0, 0);
                x11::XMoveResizeWindow(dpy, embedded_xid, 0, 0, width as u32, height as u32);
                x11::XSync(dpy, 0);
            }
            crate::gui_log_line(&format!(
                "reparented editor window 0x{embedded_xid:x} into host 0x{:x} (before mapping)",
                request.parent
            ));
        } else {
            crate::gui_log_line("could not resolve our X window id; the editor may stay hidden");
        }
        window.show_all();
    } else {
        let reason = if !request.embedded {
            "the host did not hand us an embeddable X11 window"
        } else if !display_kind.contains("X11") {
            // 【这一行就是 WSLg / Wayland 会话下的现场证据】非 X11 后端上没有外来窗口。
            "GTK is running on a non-X11 backend, so the host's X11 window cannot be embedded"
        } else {
            "embedding is opt-in on Linux (HIFISHIFTER_GUI_EMBED=1); the reparented window breaks the compositor"
        };
        crate::gui_log_line(&format!("falling back to a floating window: {reason}"));
        let weak = Rc::downgrade(&client);
        window.connect_delete_event(move |_, _| {
            if let Some(client) = weak.upgrade() {
                client.borrow_mut().close();
            }
            // 【为什么 `Proceed`】窗口的销毁交给 `close_view`（它要同时摘掉注册表与节拍）。
            gtk::glib::Propagation::Proceed
        });
        window.show_all();
    }
    let container: gtk::Widget = window.clone().upcast();
    container.set_size_request(request.width, request.height);
    container.show_all();

    // ── 一次性自检：视图真的到屏幕上了吗 ─────────────────────────────────────
    //
    // 【为什么必须自检】"`attached` 成功 + 页面加载完成"与"用户看得见"是**两件事**。
    // 实测过的最坏形态：宿主给的父窗口还没映射、或者我们的 GdkWindow 建在了它下面却
    // 始终没被 map —— 页面照样加载完成、日志照样全绿，屏幕上一片空白。
    // 这三行把"到底映射没有、尺寸多大、父窗口可见不可见"变成日志里的事实。
    {
        let widget = container.clone();
        glib::timeout_add_local_once(std::time::Duration::from_millis(1500), move || {
            let window = widget.window();
            crate::gui_log_line(&format!(
                "view self-check: widget_mapped={} window_viewable={} size={}x{}",
                widget.is_mapped(),
                window.as_ref().is_some_and(|w| w.is_viewable()),
                window.as_ref().map(|w| w.width()).unwrap_or(0),
                window.as_ref().map(|w| w.height()).unwrap_or(0),
            ));
        });
    }

    VIEWS.with(|views| {
        views.borrow_mut().insert(
            token,
            View {
                client: client.clone(),
                port: port.clone(),
                container: container.clone(),
                webview: webview.clone(),
                embedded_xid,
                on_host_thread,
            },
        );
    });
    LIVE_VIEWS.fetch_add(1, Ordering::AcqRel);
    ensure_tick();

    // ── 绘制自检：把 WebKit 的渲染结果写成 PNG ───────────────────────────────
    //
    // 【为什么必须这么做】"映射成功 + 页面就绪"仍不等于"画出来了"，而宿主进程里的
    // 像素**从外面拿不到**：WSLg（XWayland + Weston）下 `xwd` 连 REAPER 自己的
    // FX 链窗口都读成纯黑（实测 `unique=1`），"黑屏"这个读数没有信息量。
    //
    // `snapshot` 取的是 WebKit 自己的渲染结果，与 X 合成无关。写成 PNG 之后：
    // 纯色 1100x720 压出来是**几百字节**，真实 UI 是**几十 KB** —— 一个数字就能回答
    // "WebKit 到底画了没有"，而且这个文件还能被外部工具打开来看。
    {
        let view = webview.clone();
        glib::timeout_add_local_once(std::time::Duration::from_millis(3000), move || {
            view.snapshot(
                webkit2gtk::SnapshotRegion::Visible,
                webkit2gtk::SnapshotOptions::NONE,
                None::<&gio::Cancellable>,
                move |result| {
                    let surface = match result {
                        Ok(surface) => surface,
                        Err(error) => {
                            // 【为什么要打错误文本】§snapshot§ 失败就是"WebKit 拿不出渲染
                            // 结果"，那正是黑屏的机制。错误信息里写着它卡在哪一步
                            // （进程崩溃 / 没有可绘制的帧 / 区域为空）。
                            crate::gui_log_line(&format!(
                                "paint self-check: snapshot failed: {error}"
                            ));
                            return;
                        }
                    };
                    // 固定文件名：每开一次窗口覆盖上一次，不在日志目录里堆积。
                    let path = crate::diagnostics::log_directory().join("gui-snapshot.png");
                    let Ok(c_path) = std::ffi::CString::new(path.to_string_lossy().as_bytes())
                    else {
                        return;
                    };
                    use gtk::glib::translate::ToGlibPtr;
                    let status = unsafe {
                        cairo_ffi::cairo_surface_write_to_png(
                            surface.to_glib_none().0.cast(),
                            c_path.as_ptr(),
                        )
                    };
                    let bytes = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
                    crate::gui_log_line(&format!(
                        "paint self-check: status={status} png_bytes={bytes} path={}",
                        path.display()
                    ));
                },
            );
        });
    }

    webview.load_uri(protocol::PAGE_URL);
    crate::gui_log_line(&format!(
        "webkit view {token} created ({}x{}, backend={display_kind}, {})",
        request.width,
        request.height,
        fidelity.as_str()
    ));
    Ok((token, fidelity))
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
        crate::gui_log_line("could not register the script message handler");
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
                // §request()§ 自 WebKit 2.6 起废弃，用 §navigation_action().request()§。
                let allowed = navigation
                    .navigation_action()
                    .and_then(|action| action.request())
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
        crate::gui_log_line(&format!("page load failed for {uri}: {error}"));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// **不变量：`IRunLoop` 的 IID 在内存里逐字节等于 SDK 声明的那 16 个字节。**
    ///
    /// 【为什么钉死】`queryInterface` 是**逐字节**比较 IID 的。抄错一个十六进制位，
    /// 后果不是崩溃，而是"探测恒为 false" ⇒ 永远走情形 B ⇒ 宿主明明有正规解却没人用，
    /// 而日志里看不出来 —— 这是本仓最讨厌的一类静默退化。
    ///
    /// 期望值逐字来自 `DECLARE_CLASS_IID (IRunLoop, 0x18C35366, 0x97764F1A,
    /// 0x9C5B8385, 0x7A871389)`，按 VST3 的 FUID 约定以四个 u32 依次存放（主机字节序）。
    #[test]
    fn the_run_loop_iid_is_byte_identical_to_the_sdk_declaration() {
        let expected: [u8; 16] = [
            0x66, 0x53, 0xC3, 0x18, // 0x18C35366
            0x1A, 0x4F, 0x76, 0x97, // 0x97764F1A
            0x85, 0x83, 0x5B, 0x9C, // 0x9C5B8385
            0x89, 0x13, 0x87, 0x7A, // 0x7A871389
        ];
        // SAFETY: IID_RUN_LOOP 是 [u32; 4]，长度恰好 16 字节，读成 [u8; 16] 是安全的。
        let actual: &[u8; 16] = unsafe { &*(IID_RUN_LOOP.as_ptr().cast::<[u8; 16]>()) };
        assert_eq!(actual, &expected);
        assert_eq!(std::mem::size_of_val(&IID_RUN_LOOP), 16);
    }

    /// 没有 frame 时探测必须直接说"没有"，不能解引用空指针。
    #[test]
    fn probing_without_a_host_frame_is_a_no_op() {
        assert!(!probe_run_loop(0));
    }
}
