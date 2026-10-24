//! Windows 原语：一个子 `HWND` + WebView2 + 20 ms `WM_TIMER` 泵。
//!
//! 【这一份与 `webview.rs` 的差别必须逐块可对照】搬迁是**纯机械**的：判定逻辑一个字
//! 都没改，只是搬到了 `client.rs`。这里保留的三件事全是"只有 Windows 会做"的：
//! 建子窗口、把虚拟主机映射到 `Resources/frontend`、把 `WebMessageReceived` 转给
//! Policy。评审时对着 `git show` 逐块比即可。
//!
//! 【为什么 `ORIGIN` 只在 Windows 上存在】WebView2 用
//! `SetVirtualHostNameToFolderMapping` 把 `hifishifter.invalid` 映射到一个本地目录，
//! 于是"资源在哪"由宿主进程决定；WebKit 两个平台走自定义 scheme（见 `protocol.rs`）。

use super::client::{self, EditorClient};
use super::port::{EditorHost, EditorPort, GuiFidelity, NativeWindow};
use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use webview2_com::Microsoft::Web::WebView2::Win32::*;
use webview2_com::{
    AddScriptToExecuteOnDocumentCreatedCompletedHandler, CoTaskMemPWSTR,
    CreateCoreWebView2ControllerCompletedHandler, CreateCoreWebView2EnvironmentCompletedHandler,
    NavigationCompletedEventHandler, NavigationStartingEventHandler,
    NewWindowRequestedEventHandler, WebMessageReceivedEventHandler,
};
use windows::core::{w, Interface, PCWSTR, PWSTR};
use windows::Win32::Foundation::{HANDLE, HINSTANCE, HMODULE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_APARTMENTTHREADED};
use windows::Win32::System::LibraryLoader::GET_MODULE_HANDLE_EX_FLAG_PIN;
use windows::Win32::System::LibraryLoader::{
    GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
};
use windows::Win32::System::Threading::GetCurrentThreadId;
use windows::Win32::UI::Input::KeyboardAndMouse::{GetFocus, SetFocus};
use windows::Win32::UI::WindowsAndMessaging::*;

/// 内嵌 WebView 的虚拟 origin。
///
/// 【为什么用 `.invalid`】RFC 2606 把 `.invalid` 保留为**永不解析**的 TLD，所以即使宿主
/// 机器上有恶意 DNS/代理，这个 origin 也不可能指向真实站点 —— 它只是本进程内
/// `WebResourceRequested` 拦截器（见下面的路由表）用的命名空间。不要改成任何可注册域名。
const ORIGIN: &str = "https://hifishifter.invalid/";
static WINDOWS: AtomicUsize = AtomicUsize::new(0);
/// 窗口属性 token：只用来判定 "这个句柄还是不是我的那个窗口"。
static NEXT_TOKEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static CLASS_LOCK: Mutex<bool> = Mutex::new(false);
static WEBVIEW_MODULE_PIN: OnceLock<Result<(), String>> = OnceLock::new();

/// 挂在 `GWLP_USERDATA` 上的窗口数据：它是**唯一的**"从 HWND 找回判定状态"的入口。
///
/// 【为什么仍然用窗口属性而不是全局表】`WM_NCDESTROY` 是系统给的、与窗口同寿的析构
/// 点；用它释放数据，就不存在"窗口没了而表里还留着"的悬垂项。
struct WindowData {
    client: Rc<RefCell<EditorClient>>,
    port: Rc<WindowsPort>,
    capturered: Rc<Cell<bool>>,
    token: usize,
}
impl Drop for WindowData {
    fn drop(&mut self) {
        // 【顺序】先让判定层收尾（撤销本视图持有的宿主 Undo 记账、停掉出站），
        // 再关 COM。反过来的话，关闭过程中的迟到回调会撞上一个已经宣称关闭的状态。
        self.client.borrow_mut().close();
        self.port.close();
        let _ = unsafe { KillTimer(Some(self.port.hwnd.get()), 0x4853) };
        // 【为什么不在这里删用户数据目录】目录现在是**每进程**一份、被本进程的其它视图
        // 共享（见 `begin_browser`），关掉一个视图不能把还在用的那份删掉。回收改由
        // `temp_sweep::sweep_stale_temp_dirs` 在下一次启动时按"属主进程已退出"进行。
        unsafe {
            CoUninitialize();
        }
    }
}

/// WebView2 的出站端口：页面还没建好时 `post_message` 返回 `false`。
pub(super) struct WindowsPort {
    /// 建窗之后才填（§WindowData§ 必须先于 §CreateWindowExW§ 存在）。
    hwnd: Cell<HWND>,
    /// 控制器是**异步**到的；它到达之前一切投递都只能报"投不出去"。
    controller: RefCell<Option<ICoreWebView2Controller>>,
    /// 错误提示用的 `STATIC` 子窗口（自有 child 没有标题栏，不能只写 window text）。
    error_window: RefCell<Option<HWND>>,
}
impl EditorPort for WindowsPort {
    fn post_message(&self, text: &str) -> bool {
        let controller = self.controller.borrow().clone();
        let Some(controller) = controller else {
            return false;
        };
        let Ok(browser) = (unsafe { controller.CoreWebView2() }) else {
            return false;
        };
        let text = wide(text);
        unsafe { browser.PostWebMessageAsJson(PCWSTR(text.as_ptr())) }.is_ok()
    }

    fn show_error(&self, text: &str) {
        let hwnd = self.hwnd.get();
        let text = wide(&format!("HiFiShifter editor: {text}"));
        let existing = *self.error_window.borrow();
        if let Some(existing) = existing {
            let _ = unsafe { SetWindowTextW(existing, PCWSTR(text.as_ptr())) };
            return;
        }
        // 自有 child 没有标题栏；用真实 STATIC 控件显示错误，不能只写不可见 window text。
        if let Ok(window) = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE::default(),
                w!("STATIC"),
                PCWSTR(text.as_ptr()),
                WS_CHILD | WS_VISIBLE,
                12,
                12,
                600,
                120,
                Some(hwnd),
                None,
                None,
                None,
            )
        } {
            *self.error_window.borrow_mut() = Some(window);
        }
    }

    fn close(&self) {
        // Close 会同步回调；此时不持有任何 `RefCell` 借用。
        let controller = self.controller.borrow_mut().take();
        if let Some(controller) = controller {
            let _ = unsafe { controller.Close() };
        }
    }
}

/// 只保存窗口身份，不把非 Send COM 接口从 UI 线程带走。资源由自有 WndProc 析构。
pub(super) struct WindowsHost {
    key: WindowKey,
}
#[derive(Clone, Copy)]
pub(super) struct WindowKey {
    hwnd: usize,
    thread: u32,
    token: usize,
}
impl WindowKey {
    /// 句柄被系统复用后不能移动/关闭另一实例窗口；属性 token 在 `WM_NCDESTROY` 时撤销。
    fn valid(self) -> bool {
        let hwnd = HWND(self.hwnd as *mut c_void);
        unsafe {
            IsWindow(Some(hwnd)).as_bool()
                && GetPropW(hwnd, w!("HiFiShifter.ARA.ViewToken")).0 as usize == self.token
        }
    }
    /// 从窗口属性取回判定状态（**只克隆 `Rc`**，不返回指向窗口数据的引用）。
    fn client(self) -> Option<Rc<RefCell<EditorClient>>> {
        if !self.valid() {
            return None;
        }
        let hwnd = HWND(self.hwnd as *mut c_void);
        let pointer = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *const WindowData;
        if pointer.is_null() {
            return None;
        }
        Some(unsafe { (*pointer).client.clone() })
    }
    /// 只在宿主 UI 线程把焦点交给自有编辑器；已在浏览器内时不重置 DOM 焦点。
    fn focus(self) -> Result<(), String> {
        if unsafe { GetCurrentThreadId() } != self.thread || !self.valid() {
            return Err("focus on stale window or off UI thread".into());
        }
        let hwnd = HWND(self.hwnd as *mut c_void);
        let current = unsafe { GetFocus() };
        if current == hwnd || unsafe { IsChild(hwnd, current) }.as_bool() {
            return Ok(());
        }
        let result = unsafe { SetFocus(Some(hwnd)) };
        // 首次 SetFocus 的旧句柄可能为 null，Win32 包装返回 Err 但真实焦点已成功设置。
        let focused = unsafe { GetFocus() };
        if focused == hwnd || unsafe { IsChild(hwnd, focused) }.as_bool() {
            Ok(())
        } else {
            result.map(|_| ()).map_err(|e| e.to_string())
        }
    }
    /// 可复制的窗口身份不持有 COM 资源，允许调用前释放 view 锁以防 Win32 同步重入。
    fn resize(self, width: i32, height: i32) -> Result<(), String> {
        if unsafe { GetCurrentThreadId() } != self.thread || !self.valid() {
            return Err("resize on stale window or off UI thread".into());
        }
        unsafe { MoveWindow(HWND(self.hwnd as *mut c_void), 0, 0, width, height, true) }
            .map_err(|e| e.to_string())
    }
}
impl EditorHost for WindowsHost {
    fn fidelity(&self) -> GuiFidelity {
        GuiFidelity::Embedded
    }

    fn resize(&self, width: i32, height: i32) -> Result<(), String> {
        self.key.resize(width, height)
    }

    fn focus(&self) -> Result<(), String> {
        self.key.focus()
    }

    /// 只向真实活 WebView 交付共享编辑键；不向外层 HWND 伪造无效 Win32 键盘消息。
    fn forward_edit_key(&self, key: char, mods: i16, down: bool) -> bool {
        if unsafe { GetCurrentThreadId() } != self.key.thread {
            return false;
        }
        let Some(client) = self.key.client() else {
            return false;
        };
        let handled = client.borrow_mut().forward_edit_key(key, mods, down);
        handled
    }

    /// 20 ms 定时器本来就在宿主 UI 线程上，不需要"补做"。
    fn host_tick(&self) {}

    fn close(&self) {
        // 析构由 `Drop` 负责（它知道该 DestroyWindow 还是 PostMessage 给窗口线程）。
    }
}
impl Drop for WindowsHost {
    fn drop(&mut self) {
        let hwnd = HWND(self.key.hwnd as *mut c_void);
        if !self.key.valid() {
            return;
        }
        if unsafe { GetCurrentThreadId() } == self.key.thread {
            let _ = unsafe { DestroyWindow(hwnd) };
        } else {
            // COM 由自有窗口线程 Close，不在错误线程释放。正常 REAPER UI 析构走上分支。
            let _ = unsafe { PostMessageW(Some(hwnd), WM_CLOSE, WPARAM(0), LPARAM(0)) };
        }
    }
}

/// 获取本DLL而非REAPER模块；不依据项目CWD定位资产。
fn module() -> Result<HMODULE, String> {
    let mut module = HMODULE::default();
    unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            PCWSTR(module_address as *const () as *const u16),
            &mut module,
        )
    }
    .map_err(|e| e.to_string())?;
    Ok(module)
}
fn module_address() {}
/// WebView2的环境创建没有取消API，迟到COM回调仍会进入本DLL；固定模块到宿主退出。
/// 只固定这一模块一次，不保留窗口/浏览器/worker；开发升级需要正常退出REAPER。
fn pin_callback_code() -> Result<(), String> {
    WEBVIEW_MODULE_PIN
        .get_or_init(|| {
            let mut handle = HMODULE::default();
            unsafe {
                GetModuleHandleExW(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                    PCWSTR(module_address as *const () as *const u16),
                    &mut handle,
                )
            }
            .map_err(|e| e.to_string())
        })
        .clone()
}
/// 前端产物目录：与 macOS/Linux **同一份**实现（`editor::resources`）。
///
/// 【为什么不在这里再写一遍 `GetModuleFileNameW`】"资源在哪"只该有一个答案；
/// 两份实现里只要有一份忘了往上两级，另一个平台就会白屏，而日志里看不出来。
fn assets() -> Result<PathBuf, String> {
    crate::editor::resources::frontend_directory()
}

fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(Some(0)).collect()
}

fn register_class() -> Result<(), String> {
    let mut registered = CLASS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if *registered {
        return Ok(());
    }
    let class = WNDCLASSW {
        lpfnWndProc: Some(window_proc),
        hInstance: HINSTANCE(module()?.0),
        lpszClassName: w!("HiFiShifter.ARA.Editor"),
        ..Default::default()
    };
    if unsafe { RegisterClassW(&class) } == 0 {
        // windows 0.62 把 `Error::from_win32` 改名为 `from_thread`（语义不变：
        // 取本线程的 GetLastError）。
        return Err(windows::core::Error::from_thread().to_string());
    }
    *registered = true;
    Ok(())
}
/// 不留指向已卸载DLL的window class；打开中的child明确阻止正常卸载。
pub(super) fn shutdown() -> bool {
    if WINDOWS.load(Ordering::Acquire) != 0 {
        crate::gui_log_line("ExitDll refused: native editor windows still alive");
        return false;
    }
    let mut registered = CLASS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if *registered {
        let Ok(module) = module() else {
            return false;
        };
        if unsafe { UnregisterClassW(w!("HiFiShifter.ARA.Editor"), Some(HINSTANCE(module.0))) }
            .is_err()
        {
            return false;
        }
        *registered = false;
    }
    true
}

impl WindowsHost {
    /// 同线程创建子窗口，异步浏览器由宿主消息循环完成，禁止嵌套消息泵。
    pub(super) fn attach(
        parent: NativeWindow,
        width: i32,
        height: i32,
        link: Arc<crate::editor::routing::EditorLink>,
    ) -> Result<Self, String> {
        let parent = HWND(parent as *mut c_void);
        if !unsafe { IsWindow(Some(parent)) }.as_bool() {
            return Err("parent is not a live HWND".into());
        }
        let thread = unsafe { GetCurrentThreadId() };
        if unsafe { GetWindowThreadProcessId(parent, None) } != thread {
            return Err("IPlugView attached off the parent UI thread".into());
        }
        register_class()?;
        let instance = HINSTANCE(module()?.0);
        let initialized = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED) };
        initialized
            .ok()
            .map_err(|e| format!("WebView2 STA unavailable: {e}"))?;
        let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed) as usize;
        // 判定状态先建：窗口数据要持有它，而窗口数据在 CreateWindowExW 之前就要存在
        // （WM_NCCREATE 会立刻取它）。
        let client = EditorClient::open(link);
        let port = Rc::new(WindowsPort {
            hwnd: Cell::new(HWND::default()),
            controller: RefCell::new(None),
            error_window: RefCell::new(None),
        });
        let capturered = Rc::new(Cell::new(false));
        let data = Box::into_raw(Box::new(WindowData {
            client: client.clone(),
            port: port.clone(),
            capturered: capturered.clone(),
            token,
        }));
        let result = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE::default(),
                w!("HiFiShifter.ARA.Editor"),
                w!("HiFiShifter plugin editor initializing…"),
                WS_CHILD | WS_VISIBLE | WS_TABSTOP | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
                0,
                0,
                width,
                height,
                Some(parent),
                None,
                Some(instance),
                Some(data.cast()),
            )
        };
        let hwnd = match result {
            Ok(hwnd) => hwnd,
            Err(error) => {
                // WM_NCCREATE 之前失败时系统未接管 data；之后失败则 WM_NCDESTROY 已负责释放。
                if !capturered.get() {
                    unsafe {
                        drop(Box::from_raw(data));
                    }
                }
                return Err(format!("create child window: {error}"));
            }
        };
        port.hwnd.set(hwnd);
        client.borrow_mut().set_window(hwnd.0 as usize);
        let host = Self {
            key: WindowKey {
                hwnd: hwnd.0 as usize,
                thread,
                token,
            },
        };
        if unsafe { SetTimer(Some(hwnd), 0x4853, 20, None) } == 0 {
            port.show_error("UI response timer unavailable");
        }
        if let Err(error) = begin_browser(&client, &port) {
            port.show_error(&error);
            // attach 半失败：没有可投递的 browser，泵无事可做、视图已死，
            // 20ms 定时器只会空转（50Hz 全速泵一块错误 STATIC）。停掉它。
            unsafe {
                let _ = KillTimer(Some(hwnd), 0x4853);
            }
        }
        Ok(host)
    }
}

unsafe extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // 所有可能 panic 的自有逻辑都在边界内，不能越过 Win32 回调栈。
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if message == WM_NCCREATE {
            let create = &*(lparam.0 as *const CREATESTRUCTW);
            let data = &*(create.lpCreateParams as *const WindowData);
            data.capturered.set(true);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, create.lpCreateParams as isize);
            WINDOWS.fetch_add(1, Ordering::AcqRel);
            if SetPropW(
                hwnd,
                w!("HiFiShifter.ARA.ViewToken"),
                Some(HANDLE(data.token as *mut c_void)),
            )
            .is_err()
            {
                return LRESULT(0);
            }
        }
        let pointer = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut WindowData;
        if !pointer.is_null() {
            if message == WM_TIMER && wparam.0 == 0x4853 {
                // 宿主 getter 可能重入并关闭窗口：先持 Rc，不再在调用后解引用 WindowData。
                let client = (*pointer).client.clone();
                client::tick(&client);
                // 【为什么在定时器上关窗】页面进程崩溃之后所有投递都会失败，界面已是
                // 一张死图。继续留着它只会让预算被占满、后续命令全部被拒 —— 那比
                // "明确关掉"更难排查。
                if client.borrow().is_dead() {
                    crate::gui_log_line(
                        "closing the editor view: the embedded page stopped responding",
                    );
                    let _ = DestroyWindow(hwnd);
                }
                return LRESULT(0);
            } else if message == WM_SETFOCUS {
                // 宿主 Tab/onFocus 进入自有 HWND 后，使用标准 WebView2 接口进入 HTML 控件。
                // MoveFocus 可能同步通知宿主；调用前不持任何借用。
                let port = (*pointer).port.clone();
                port.focus_page();
                return LRESULT(0);
            } else if message == WM_SIZE {
                let port = (*pointer).port.clone();
                port.sync_bounds(hwnd);
            } else if message == WM_CLOSE {
                let _ = DestroyWindow(hwnd);
                return LRESULT(0);
            } else if message == WM_NCDESTROY {
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
                let _ = RemovePropW(hwnd, w!("HiFiShifter.ARA.ViewToken"));
                drop(Box::from_raw(pointer));
                WINDOWS.fetch_sub(1, Ordering::AcqRel);
            }
        }
        DefWindowProcW(hwnd, message, wparam, lparam)
    }))
    .unwrap_or(LRESULT(0))
}

/// 把一条 `import_native_audio_file` 改写成普通的 `import_audio_item`。
///
/// 【为什么必须在平台侧做】真实磁盘路径只在 WebView2 的 `AdditionalObjects` 上，
/// 而那是 COM 参数 —— Policy 只收字符串。改写在**进入** Policy 之前完成，于是
/// "拖进来的文件"和"从菜单选的路径"在 Policy 里是同一条命令。
///
/// 返回 `None` = 这条消息不是拖放导入，原样交给 Policy。
fn native_file_request(
    args: &ICoreWebView2WebMessageReceivedEventArgs,
    view_id: &str,
    text: &str,
) -> Option<Result<String, serde_json::Value>> {
    if !text.contains("import_native_audio_file") {
        return None;
    }
    let mut request = serde_json::from_str::<serde_json::Value>(text).ok()?;
    if request["command"] != "import_native_audio_file" {
        return None;
    }
    let path = (|| -> Result<String, String> {
        if request
            .get("args")
            .is_some_and(|value| !value.is_object() && !value.is_null())
        {
            return Err("native File args must be an object".into());
        }
        let extra: ICoreWebView2WebMessageReceivedEventArgs2 =
            args.cast().map_err(|_| "native File bridge unavailable")?;
        let objects = unsafe { extra.AdditionalObjects() }.map_err(|e| e.to_string())?;
        let mut count = 0;
        unsafe { objects.Count(&mut count) }.map_err(|e| e.to_string())?;
        if count != 1 {
            return Err("exactly one native File is required".into());
        }
        let file: ICoreWebView2File = unsafe { objects.GetValueAtIndex(0) }
            .map_err(|e| e.to_string())?
            .cast()
            .map_err(|_| "additional object is not a disk File")?;
        let mut path = PWSTR::null();
        unsafe { file.Path(&mut path) }.map_err(|e| e.to_string())?;
        let path = CoTaskMemPWSTR::from(path).to_string();
        if !std::path::Path::new(&path).is_absolute() {
            return Err("dragged File has no native disk path; use File menu import".into());
        }
        Ok(path)
    })();
    match path {
        Ok(path) => {
            request["command"] = serde_json::json!("import_audio_item");
            request["args"]["audioPath"] = serde_json::json!(path);
            Some(Ok(request.to_string()))
        }
        Err(error) => {
            // 失败也走同一条回执通道 —— Policy 不知道 Windows 的 COM 参数，
            // 所以这条错误只能在这里成形。
            let id = request["id"].as_u64().unwrap_or_default();
            Some(Err(
                serde_json::json!({"version":1,"viewId":view_id,"id":id,
                "ok":false,"errorKind":"transient","error":error}),
            ))
        }
    }
}

/// WebView 异步回调仅捕获 weak state，removed 后不会重新创建或访问裸 view。
fn begin_browser(state: &Rc<RefCell<EditorClient>>, port: &Rc<WindowsPort>) -> Result<(), String> {
    let folder = assets()?;
    pin_callback_code()?;
    // 【为什么不再用 `%TEMP%`】磁盘清理会在会话进行中删掉 `%TEMP%` 下的内容，
    // 正在运行的 WebView2 用户数据目录被删掉会直接崩 —— 而不只是"缓存丢了"。
    // 放在本机数据目录下，生命周期由我们自己掌握。
    //
    // 【为什么是**每进程**一个目录，而不是每视图一个】WebView2 不支持两个进程同时打开
    // 同一个用户数据目录（所以必须带 PID），但同进程内多个控制器可以共享一个环境。
    // 此前每个视图各建一个目录，视图一多就攒出一堆（实测 41 个目录 838 MB），而且只在
    // `Drop` 时清理 —— 宿主被强杀就永久留下。现在按进程复用一份，并由
    // `temp_sweep::sweep_stale_temp_dirs` 在下一次启动时回收已退出进程的那份。
    let profile = hifishifter_kernel::config_location::local_data_subdir("webview")
        .join(format!("view-{}", std::process::id()));
    std::fs::create_dir_all(&profile).map_err(|e| e.to_string())?;
    let profile = wide(&profile.to_string_lossy());
    let weak = Rc::downgrade(state);
    let outer_port = port.clone();
    let completed = CreateCoreWebView2EnvironmentCompletedHandler::create(Box::new(
        move |result, environment| {
            let Some(state) = weak.upgrade() else {
                return Ok(());
            };
            if state.borrow().closed {
                return Ok(());
            }
            let outcome = (|| -> windows::core::Result<()> {
                result?;
                let environment = environment.ok_or_else(|| {
                    windows::core::Error::from_hresult(windows::core::HRESULT(0x80004003u32 as i32))
                })?;
                let weak = Rc::downgrade(&state);
                let port = outer_port.clone();
                let folder = folder.clone();
                let controller_ready = CreateCoreWebView2ControllerCompletedHandler::create(
                    Box::new(move |result, controller| {
                        let Some(state) = weak.upgrade() else {
                            if let Some(controller) = controller {
                                let _ = unsafe { controller.Close() };
                            }
                            return Ok(());
                        };
                        if state.borrow().closed {
                            if let Some(controller) = controller {
                                let _ = unsafe { controller.Close() };
                            }
                            return Ok(());
                        }
                        if let Err(error) = result {
                            port.show_error(&error.to_string());
                            return Ok(());
                        }
                        if let Some(controller) = controller {
                            port.attach(controller.clone());
                            // 【为什么端口在这里才挂上】页面就绪前的出站消息由 Policy
                            // 排队；没有端口时它连"投不出去"都判断不了。
                            let dyn_port: Rc<dyn EditorPort> = port.clone();
                            state.borrow_mut().set_port(dyn_port);
                            if let Err(error) =
                                configure_browser(&state, &port, &controller, &folder)
                            {
                                port.show_error(&error.to_string());
                            }
                        } else {
                            port.show_error("WebView2 returned no controller");
                        }
                        Ok(())
                    }),
                );
                let hwnd = outer_port.hwnd.get();
                unsafe { environment.CreateCoreWebView2Controller(hwnd, &controller_ready) }
            })();
            if let Err(error) = outcome {
                outer_port.show_error(&error.to_string());
            }
            Ok(())
        },
    ));
    unsafe {
        CreateCoreWebView2EnvironmentWithOptions(
            PCWSTR::null(),
            PCWSTR(profile.as_ptr()),
            None::<&ICoreWebView2EnvironmentOptions>,
            &completed,
        )
    }
    .map_err(|e| e.to_string())
}

impl WindowsPort {
    fn attach(&self, controller: ICoreWebView2Controller) {
        *self.controller.borrow_mut() = Some(controller);
    }

    /// 宿主把焦点交给本窗口后，让 WebView2 把焦点放进 HTML 控件。
    fn focus_page(&self) {
        let controller = self.controller.borrow().clone();
        if let Some(controller) = controller {
            let _ = unsafe { controller.MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_NEXT) };
        }
    }

    /// 子窗口尺寸变了：把同一个矩形交给控制器（客户区坐标，物理像素）。
    fn sync_bounds(&self, hwnd: HWND) {
        let controller = self.controller.borrow().clone();
        if let Some(controller) = controller {
            let mut bounds = RECT::default();
            if unsafe { GetClientRect(hwnd, &mut bounds) }.is_ok() {
                let _ = unsafe { controller.SetBounds(bounds) };
            }
        }
    }
}

fn configure_browser(
    state: &Rc<RefCell<EditorClient>>,
    port: &Rc<WindowsPort>,
    controller: &ICoreWebView2Controller,
    folder: &std::path::Path,
) -> windows::core::Result<()> {
    let hwnd = port.hwnd.get();
    let mut bounds = RECT::default();
    unsafe {
        GetClientRect(hwnd, &mut bounds)?;
        controller.SetBounds(bounds)?;
        controller.SetIsVisible(true)?;
    }
    // 异步创建完成时仅延续宿主已交给本窗口的焦点，不能抢另一个 FX/应用的焦点。
    if unsafe { GetFocus() } == hwnd {
        unsafe {
            controller.MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_NEXT)?;
        }
    }
    let browser = unsafe { controller.CoreWebView2()? };
    // 真实文件拖入由 WebView2 原生 OLE 接收，再通过 AdditionalObjects/File.Path 交给宿主。
    if let Ok(external) = controller.cast::<ICoreWebView2Controller4>() {
        unsafe {
            external.SetAllowExternalDrop(true)?;
        }
    }
    let mapping: ICoreWebView2_3 = browser.cast()?;
    let path = wide(&folder.to_string_lossy());
    unsafe {
        mapping.SetVirtualHostNameToFolderMapping(
            w!("hifishifter.invalid"),
            PCWSTR(path.as_ptr()),
            COREWEBVIEW2_HOST_RESOURCE_ACCESS_KIND_DENY_CORS,
        )?;
    }
    let settings = unsafe { browser.Settings()? };
    unsafe {
        settings.SetAreHostObjectsAllowed(false)?;
        settings.SetAreDefaultContextMenusEnabled(false)?;
    }
    let mut token = 0;
    unsafe {
        browser.add_NavigationStarting(
            &NavigationStartingEventHandler::create(Box::new(|_, args| {
                if let Some(args) = args {
                    let mut uri = PWSTR::null();
                    args.Uri(&mut uri)?;
                    if !CoTaskMemPWSTR::from(uri).to_string().starts_with(ORIGIN) {
                        args.SetCancel(true)?;
                    }
                }
                Ok(())
            })),
            &mut token,
        )?;
        browser.add_NewWindowRequested(
            &NewWindowRequestedEventHandler::create(Box::new(|_, args| {
                if let Some(args) = args {
                    args.SetHandled(true)?;
                }
                Ok(())
            })),
            &mut token,
        )?;
    }
    // 就绪门（见方案 `4.3`）：导航完成之前投递会被丢弃，所以 Policy 先排队。
    //
    // 【为什么用 NavigationCompleted 而不是"脚本注入完成"】后者只证明脚本装好了，
    // 页面本身还在解析；前者才是"监听器已在运行"的可靠信号。
    let weak = Rc::downgrade(state);
    unsafe {
        browser.add_NavigationCompleted(
            &NavigationCompletedEventHandler::create(Box::new(move |_, args| {
                let success = args.is_some_and(|args| {
                    let mut ok = windows::core::BOOL(0);
                    args.IsSuccess(&mut ok).is_ok() && ok.as_bool()
                });
                if success {
                    if let Some(state) = weak.upgrade() {
                        state.borrow_mut().on_page_ready();
                    }
                }
                Ok(())
            })),
            &mut token,
        )?;
    }
    let weak = Rc::downgrade(state);
    unsafe {
        browser.add_WebMessageReceived(
            &WebMessageReceivedEventHandler::create(Box::new(move |_, args| {
                let (Some(state), Some(args)) = (weak.upgrade(), args) else {
                    return Ok(());
                };
                let view_id = {
                    let state = state.borrow();
                    if state.closed {
                        return Ok(());
                    }
                    state.view_id.clone()
                };
                // 来源校验：只有本 bundle 的虚拟主机可以发消息。
                let mut source = PWSTR::null();
                args.Source(&mut source)?;
                if !CoTaskMemPWSTR::from(source).to_string().starts_with(ORIGIN) {
                    return Ok(());
                }
                let mut json = PWSTR::null();
                args.WebMessageAsJson(&mut json)?;
                let text = CoTaskMemPWSTR::from(json).to_string();
                // 【Windows 专属的入口改写】拖进来的真实磁盘文件只能从 COM 参数读到。
                match native_file_request(&args, &view_id, &text) {
                    None => client::on_message(&state, &text),
                    Some(Ok(rewritten)) => client::on_message(&state, &rewritten),
                    Some(Err(reply)) => {
                        state.borrow_mut().post(reply);
                    }
                }
                Ok(())
            })),
            &mut token,
        )?;
    }
    let boot = {
        let state = state.borrow();
        client::bootstrap_json(&state.link, &state.view_id)
    };
    let script = wide(&format!("window.__HFS_PLUGIN_BOOTSTRAP__={boot};"));
    let weak = Rc::downgrade(state);
    let navigate = browser.clone();
    let ready_port = port.clone();
    let ready =
        AddScriptToExecuteOnDocumentCreatedCompletedHandler::create(Box::new(move |result, _| {
            let Some(state) = weak.upgrade() else {
                return Ok(());
            };
            if state.borrow().closed {
                return Ok(());
            }
            if let Err(error) = result {
                ready_port.show_error(&error.to_string());
                return Ok(());
            }
            unsafe {
                navigate.Navigate(w!("https://hifishifter.invalid/plugin.html"))?;
            }
            crate::gui_log_line("frontend navigation started (commands pending session binding)");
            Ok(())
        }));
    unsafe { browser.AddScriptToExecuteOnDocumentCreated(PCWSTR(script.as_ptr()), &ready) }
}

#[cfg(test)]
mod focus_tests {
    use super::*;
    use std::ffi::c_void;
    use windows::Win32::UI::Input::KeyboardAndMouse::{GetActiveWindow, SetActiveWindow};

    /// 用真实自有HWND验证Tab边界，不凭源码是否包含某个style判断可达性。
    #[test]
    fn native_editor_is_a_tab_stop_in_its_host_parent() {
        // SAFETY: 夹具父窗口仅在本线程创建/销毁，不操作任何用户或REAPER窗口。
        let parent = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE::default(),
                w!("STATIC"),
                w!("HiFiShifter focus fixture"),
                WS_OVERLAPPEDWINDOW,
                0,
                0,
                1100,
                720,
                None,
                None,
                None,
                None,
            )
        }
        .unwrap();
        let host = WindowsHost::attach(parent.0 as usize, 1100, 720, Arc::new(Default::default()))
            .unwrap();
        let child = HWND(host.key.hwnd as *mut c_void);
        // `SetFocus` 的前置条件是目标窗口所属线程有一个**活动窗口**：刚创建的
        // 夹具窗口尚未激活，此时 `SetFocus` 返回 `ERROR_INVALID_PARAMETER`
        // (0x80070057)。真实宿主里插件窗口始终位于活动窗口内，这里补上同一前置
        // 条件。
        //
        // 但"能否激活"取决于会话：当本进程不是前台进程时（无头/服务/沙箱，或
        // 用户正聚焦别的应用）`SetActiveWindow` 会被系统拒绝。此时**焦点投递**
        // 类断言无从成立，跳过它们；本用例的本职 —— Tab 顺序可达性 —— 不依赖
        // 激活，照常执行。这样 CI（有交互桌面）保持全量覆盖，无前台窗口的会话
        // 也不会假失败。
        // `SetActiveWindow` 返回先前的活动窗口（没有则为 null → Err），仅用于还原。
        let previous_active = unsafe { SetActiveWindow(parent) }.ok();
        let can_take_activation = unsafe { GetActiveWindow() } == parent;
        if !can_take_activation {
            eprintln!(
                "[focus_tests] 本会话不允许本进程激活窗口，跳过焦点投递断言（Tab 顺序断言仍执行）"
            );
        }
        // GetNextDlgTabItem在没有任何Tab stop时会回退到第一个child，单看句柄会假通过。
        let tab_stop = unsafe { GetWindowLongPtrW(child, GWL_STYLE) } as u32 & WS_TABSTOP.0 != 0;
        let key = host.key;
        assert!(
            std::thread::spawn(move || key.focus().is_err())
                .join()
                .unwrap(),
            "不能从非UI线程触碰宿主焦点"
        );
        let focus_result = key.focus();
        let focused = unsafe { GetFocus() };
        // SAFETY: 两个窗口均为本线程且存活；仅查询标准Windows对话框导航结果。
        let target = unsafe { GetNextDlgTabItem(parent, None, false) }.ok();
        drop(host);
        assert!(key.focus().is_err(), "关闭后的窗口租约不能再夺取焦点");
        // 把线程的活动窗口还回去，免得把即将销毁的夹具窗口留给后续用例。
        if let Some(previous) = previous_active {
            // SAFETY: 还原本线程先前的活动窗口。
            let _ = unsafe { SetActiveWindow(previous) };
        }
        // SAFETY: 清理测试创建的唯一父窗口，子窗口已由NativeEditor析构。
        unsafe { DestroyWindow(parent) }.unwrap();
        assert!(
            tab_stop,
            "真实编辑器HWND必须接受宿主Tab停留，不能依赖导航的首child回退"
        );
        if can_take_activation {
            assert!(
                focus_result.is_ok(),
                "首次无旧焦点时仍应成功: {focus_result:?}"
            );
            assert_eq!(focused, child, "宿主焦点请求必须进入自有窗口");
        }
        assert_eq!(target, Some(child), "宿主Tab顺序必须能到达真实编辑器子窗口");
    }
}
