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
    CoreWebView2EnvironmentOptions, CreateCoreWebView2ControllerCompletedHandler,
    CreateCoreWebView2EnvironmentCompletedHandler, NavigationCompletedEventHandler,
    NavigationStartingEventHandler, NewWindowRequestedEventHandler, ProcessFailedEventHandler,
    WebMessageReceivedEventHandler,
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

// 进程内**唯一**的 WebView2 环境。
//
// 【为什么必须池化】此前每个视图各调一次 `CreateCoreWebView2EnvironmentWithOptions`
// —— 一次会话 31 次开窗就是 31 次环境创建。每次创建都会拉起（或接管）一个浏览器
// 进程并初始化一整套缓存；而 WebView2 的环境本身就是**可以**在同进程内被多个控制器
// 共享的，文档里的约束只是"一个用户数据目录不能被两个**进程**同时打开"。于是那 31 次
// 里只有第 1 次是必要的，其余 30 次只是在重建同一份东西。
//
// 池化之后还有两个附带收益：① 浏览器进程、HTTP 缓存、GPU shader 缓存跨窗口复用，
// 关掉再打开插件窗口不再从冷缓存起步；② 渲染进程死亡后的自动重建（见 [`rebuild_page`]）
// 变成毫秒级 —— 不必再等一次环境初始化。
//
// 【为什么是 thread_local】`ICoreWebView2Environment` 不是 `Send`/`Sync`：COM 的
// 单元线程模型要求它在创建它的 STA 上使用。而所有视图都在宿主 UI 线程上创建，
// 所以"UI 线程独占一份"既满足 COM 的约束，也满足共享的需要。
thread_local! {
    static ENVIRONMENT: RefCell<Option<ICoreWebView2Environment>> = const { RefCell::new(None) };
    /// 本窗口已经自动重建过几次（连续计数，页面成功加载后归零）。
    static PAGE_REBUILDS: Cell<u32> = const { Cell::new(0) };
}

/// 连续自动重建的上限。
///
/// 【为什么要有上限】重建解决的是"渲染进程被 OOM 杀掉"这种**偶发**故障。若页面每次
/// 都立刻再死一次（比如某个必现的崩溃），无限重建就是一块永远在闪的面板加一份永远在
/// 涨的日志。三次之后停下来并明确告诉用户"重开编辑器"，比空转诚实。
const MAX_PAGE_REBUILDS: u32 = 3;

/// 交给 Chromium 的附加开关。
///
/// 【`--enable-precise-memory-info`】没有它 `performance.memory` 会被模糊到 100 MiB
/// 粒度，看门狗只能看出"涨了 100 MiB 以上"，看不出斜率。它只影响我们自己创建的环境，
/// 与宿主无关。
///
/// 【`--disk-cache-size`】单个 profile 的磁盘缓存上限。实测单份最大 115 MiB，而这个
/// 数字乘上"曾经攒下的 147 份"就是 7 GiB。给一个明确上限之后，即使清扫失效，一份
/// profile 也不会再长到那个量级。
const BROWSER_ARGUMENTS: &str = "--enable-precise-memory-info --disk-cache-size=67108864";

fn cached_environment() -> Option<ICoreWebView2Environment> {
    ENVIRONMENT.with(|slot| slot.borrow().clone())
}

fn store_environment(environment: &ICoreWebView2Environment) {
    ENVIRONMENT.with(|slot| *slot.borrow_mut() = Some(environment.clone()));
}

/// 丢掉池化的环境（它随浏览器进程一起死掉时）。
fn drop_environment() {
    ENVIRONMENT.with(|slot| *slot.borrow_mut() = None);
}

/// 构造环境选项：附加开关走这里。
fn environment_options() -> ICoreWebView2EnvironmentOptions {
    let options = CoreWebView2EnvironmentOptions::default();
    // SAFETY: 只写本对象自己的 `UnsafeCell<String>`，没有别名借用。
    unsafe { options.set_additional_browser_arguments(BROWSER_ARGUMENTS.to_string()) };
    options.into()
}

/// 本进程的 WebView2 用户数据目录。
///
/// 【为什么不再用 `%TEMP%`】磁盘清理会在会话进行中删掉 `%TEMP%` 下的内容，
/// 正在运行的 WebView2 用户数据目录被删掉会直接崩 —— 而不只是"缓存丢了"。
/// 放在本机数据目录下，生命周期由我们自己掌握。
///
/// 【为什么是**每进程**一个目录，而不是每视图一个】WebView2 不支持两个进程同时打开
/// 同一个用户数据目录（所以必须带 PID），但同进程内多个控制器可以共享一个环境。
/// 此前每个视图各建一个目录，视图一多就攒出一堆（实测 41 个目录 838 MB），而且只在
/// `Drop` 时清理 —— 宿主被强杀就永久留下。现在按进程复用一份，并由
/// `temp_sweep::sweep_step` 周期回收已退出进程的那份。
fn profile_directory() -> Result<Vec<u16>, String> {
    let profile = hifishifter_kernel::config_location::local_data_subdir("webview")
        .join(format!("view-{}", std::process::id()));
    std::fs::create_dir_all(&profile).map_err(|e| e.to_string())?;
    Ok(wide(&profile.to_string_lossy()))
}

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
    /// 建窗之后才填（`WindowData` 必须先于 `CreateWindowExW` 存在）。
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
                // 【为什么先重建而不是直接关窗】页面进程崩溃 / OOM 之后所有投递都会
                // 失败，界面已是一张死图。此前唯一的处置是**关掉窗口** —— 用户看到的
                // 是一块空白，而且必须自己重开插件（这正是用户报的"关掉再开就好"）。
                // 现在先在同一个 HWND 上重建（毫秒级，因为环境是池化的）；只有连续
                // 失败到上限才真的关，并把原因写在界面与日志里。
                if client.borrow().is_dead() {
                    let port = (*pointer).port.clone();
                    if let Err(reason) = rebuild_page(&client, &port) {
                        crate::gui_log_line(&format!("closing the editor view: {reason}"));
                        let _ = DestroyWindow(hwnd);
                    }
                }
                return LRESULT(0);
            } else if message == WM_KILLFOCUS {
                // 【为什么记这里】wParam 是**即将获得焦点**的那个窗口 —— 用户点了宿主
                // 的哪一块，这里就是哪一块。"把键盘交给宿主"要把焦点还回原处，靠的就是
                // 这一笔（见 `EditorClient::previous_focus`）。
                (*pointer).client.borrow_mut().note_previous_focus(wparam.0);
            } else if message == WM_SETFOCUS {
                // 宿主 Tab/onFocus 进入自有 HWND 后，使用标准 WebView2 接口进入 HTML 控件。
                // MoveFocus 可能同步通知宿主；调用前不持任何借用。
                //
                // 【为什么在这里清"宿主自己处理这个键"的记账】用户从别的窗口回到插件的
                // 这一刻，是"我接下来想让插件处理这些键"最明确的一刻。那条记账来自一次
                // 只看 `playing` 翻转的探针，会被工程播完 / 循环回卷污染 —— 不清它，
                // 一次误判就让某个键在整场会话里都送不出去。
                let client = (*pointer).client.clone();
                client.borrow_mut().note_focus_regained();
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

/// 建（或复用）WebView2 环境，然后在本窗口的 HWND 上挂一个控制器。
///
/// 【为什么分成"环境"与"控制器"两段】环境是**进程级**资源（见 [`ENVIRONMENT`]），
/// 控制器是**窗口级**资源。拆开之后，重建一个死掉的页面只需要重做第二段。
fn begin_browser(state: &Rc<RefCell<EditorClient>>, port: &Rc<WindowsPort>) -> Result<(), String> {
    let folder = assets()?;
    pin_callback_code()?;
    // 已经有环境：直接建控制器。这条路是"开第二个窗口"与"重建"的共同路径。
    if let Some(environment) = cached_environment() {
        return create_controller(&environment, state, port, folder, true);
    }
    let profile = profile_directory()?;
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
            let environment = match (result, environment) {
                (Ok(()), Some(environment)) => environment,
                (Err(error), _) => {
                    outer_port.show_error(&error.to_string());
                    return Ok(());
                }
                _ => {
                    outer_port.show_error("WebView2 returned no environment");
                    return Ok(());
                }
            };
            store_environment(&environment);
            if let Err(error) =
                create_controller(&environment, &state, &outer_port, folder.clone(), false)
            {
                outer_port.show_error(&error.to_string());
            }
            Ok(())
        },
    ));
    let options = environment_options();
    unsafe {
        CreateCoreWebView2EnvironmentWithOptions(
            PCWSTR::null(),
            PCWSTR(profile.as_ptr()),
            Some(&options),
            &completed,
        )
    }
    .map_err(|e| e.to_string())
}

/// 在给定环境上为本窗口创建控制器（异步；结果在回调里挂端口并配置浏览器）。
///
/// `from_cache` 记录这次用的是不是池化的环境 —— 若是，而控制器创建失败，那只可能是
/// 那个环境已经随浏览器进程一起死了，必须丢掉它，否则之后每次重建都会撞上同一个死环境。
fn create_controller(
    environment: &ICoreWebView2Environment,
    state: &Rc<RefCell<EditorClient>>,
    port: &Rc<WindowsPort>,
    folder: PathBuf,
    from_cache: bool,
) -> Result<(), String> {
    let weak = Rc::downgrade(state);
    let outer_port = port.clone();
    let controller_ready = CreateCoreWebView2ControllerCompletedHandler::create(Box::new(
        move |result, controller| {
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
                // 【为什么这里必须把视图重新判死】池化的环境可能已经随浏览器进程一起
                // 死了；控制器建不出来时页面不会自己好起来，而"没死"就意味着定时器
                // 再也不会来重建。判死 + 丢掉环境，下一次 tick 就会用新环境重试。
                if from_cache {
                    drop_environment();
                    state.borrow_mut().mark_dead("controller creation failed");
                }
                outer_port.show_error(&error.to_string());
                return Ok(());
            }
            let Some(controller) = controller else {
                outer_port.show_error("WebView2 returned no controller");
                return Ok(());
            };
            outer_port.attach(controller.clone());
            // 控制器起来了 = 这一代页面已经接手这块矩形；旧的错误提示必须撤掉，
            // 否则它会一直盖在刚恢复的界面上。
            outer_port.clear_error();
            // 【为什么端口在这里才挂上】页面就绪前的出站消息由 Policy 排队；
            // 没有端口时它连"投不出去"都判断不了。
            let dyn_port: Rc<dyn EditorPort> = outer_port.clone();
            state.borrow_mut().set_port(dyn_port);
            if let Err(error) = configure_browser(&state, &outer_port, &controller, &folder) {
                outer_port.show_error(&error.to_string());
            }
            Ok(())
        },
    ));
    let hwnd = port.hwnd.get();
    unsafe { environment.CreateCoreWebView2Controller(hwnd, &controller_ready) }
        .map_err(|e| e.to_string())
}

/// 页面死了：关掉旧控制器，在**同一个 HWND** 上重新走一次引导。
///
/// 【为什么不重建窗口】窗口是宿主交给我们的、由 `IPlugView` 管尺寸与焦点的那一块矩形。
/// 换一个 HWND 就等于把宿主手里的句柄换掉，而宿主并不知道。控制器是"谁在这个矩形里
/// 画东西"，它可以随便换 —— 换的时候用户只会看到画面闪一下。
///
/// 【为什么不重建环境】见 [`ENVIRONMENT`]：环境是进程级的，重建它反而要等一次初始化。
fn rebuild_page(state: &Rc<RefCell<EditorClient>>, port: &Rc<WindowsPort>) -> Result<(), String> {
    let attempt = PAGE_REBUILDS.with(|count| {
        let next = count.get().saturating_add(1);
        count.set(next);
        next
    });
    if attempt > MAX_PAGE_REBUILDS {
        return Err(format!(
            "the embedded page failed {MAX_PAGE_REBUILDS} times in a row; reopen the editor to try again"
        ));
    }
    crate::gui_log_line(&format!(
        "rebuilding the embedded page (attempt {attempt}/{MAX_PAGE_REBUILDS})"
    ));
    // 【顺序】先让判定层放掉旧端口，再关控制器，最后重置就绪门。反过来的话，关闭
    // 过程中的迟到回调会往一个"已经准备好"的队列里塞消息。
    port.close();
    state.borrow_mut().reset_page();
    begin_browser(state, port)
}

impl WindowsPort {
    fn attach(&self, controller: ICoreWebView2Controller) {
        *self.controller.borrow_mut() = Some(controller);
    }

    /// 撤掉错误提示。
    ///
    /// 【为什么必须显式撤】错误提示是一块盖在 WebView 上的 `STATIC` 子窗口。
    /// 页面重建成功之后它仍然留在最上层 —— 于是"已经恢复"的界面看起来还是一块
    /// 错误文字，用户唯一的结论仍然是"得重开插件"。重建成功即撤掉它。
    fn clear_error(&self) {
        let existing = self.error_window.borrow_mut().take();
        if let Some(window) = existing {
            let _ = unsafe { DestroyWindow(window) };
        }
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

/// `COREWEBVIEW2_PROCESS_FAILED_KIND` → 可 grep 的名字。
///
/// 【为什么不用 `{:?}`】这些是 `windows-core` 的 newtype，`Debug` 打出来是
/// `COREWEBVIEW2_PROCESS_FAILED_KIND(5)` —— 一条要靠查表才能读懂的日志。名字是给人
/// 看的，序号随时可能随 SDK 版本变。
fn process_failed_kind(args: &ICoreWebView2ProcessFailedEventArgs) -> &'static str {
    let mut raw = COREWEBVIEW2_PROCESS_FAILED_KIND::default();
    if unsafe { args.ProcessFailedKind(&mut raw) }.is_err() {
        return "unknown";
    }
    let value = raw.0;
    if value == COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_EXITED.0 {
        "render-process-exited"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_UNRESPONSIVE.0 {
        "render-process-unresponsive"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_BROWSER_PROCESS_EXITED.0 {
        "browser-process-exited"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_GPU_PROCESS_EXITED.0 {
        "gpu-process-exited"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_FRAME_RENDER_PROCESS_EXITED.0 {
        "frame-render-process-exited"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_UTILITY_PROCESS_EXITED.0 {
        "utility-process-exited"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_SANDBOX_HELPER_PROCESS_EXITED.0 {
        "sandbox-helper-process-exited"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_PPAPI_PLUGIN_PROCESS_EXITED.0 {
        "ppapi-plugin-process-exited"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_PPAPI_BROKER_PROCESS_EXITED.0 {
        "ppapi-broker-process-exited"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_UNKNOWN_PROCESS_EXITED.0 {
        "unknown-process-exited"
    } else {
        "unknown"
    }
}

/// `COREWEBVIEW2_PROCESS_FAILED_REASON` → 可 grep 的名字。
///
/// 【为什么 OOM 这一条最关键】用户报的就是 "Out of Memory"。把它单独命名之后，
/// "这一次到底是内存耗尽、还是崩溃、还是被外部杀掉"在日志里一眼可辨 —— 而这三者的
/// 处置完全不同（前者要降载荷，后者要查崩溃，第三种可能是宿主主动结束）。
fn process_failed_reason(raw: COREWEBVIEW2_PROCESS_FAILED_REASON) -> &'static str {
    let value = raw.0;
    if value == COREWEBVIEW2_PROCESS_FAILED_REASON_OUT_OF_MEMORY.0 {
        "out-of-memory"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_REASON_CRASHED.0 {
        "crashed"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_REASON_TERMINATED.0 {
        "terminated"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_REASON_UNRESPONSIVE.0 {
        "unresponsive"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_REASON_LAUNCH_FAILED.0 {
        "launch-failed"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_REASON_PROFILE_DELETED.0 {
        "profile-deleted"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_REASON_UNEXPECTED.0 {
        "unexpected"
    } else {
        "unknown"
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
        // 【为什么必须关掉浏览器加速键】默认值下 WebView2 自己消费一批按键
        // （`F5` 重载页面、`F3` / `F7`、`Ctrl+F`、`Alt+←` …），页面根本收不到
        // `keydown`。而 `F1-F24` 明确在可让路键表里 —— 用户按 F5 期望宿主动作，
        // 实际发生的是**插件页面被重载**（顺带清掉前端的"本会话不再尝试"闩锁，
        // 于是同一个 bug 会"偶尔自愈"，更难查）。
        //
        // 【为什么选"关掉"而不是订阅 `AcceleratorKeyPressed`】两者都能到达 DOM，
        // 但订阅那条路要自己维护一份"哪些键允许浏览器处理"的白名单，而那份名单
        // 会与页面里的键表漂移成两份；关掉之后**所有**按键都进页面，让路的判据
        // 只剩一处（前端 `hostKeyRelay`）。方案 §3.5.5 允许二选一。
        // 该成员在 `ICoreWebView2Settings3` 上（WebView2 ≥ 1.0.864），基础接口没有。
        let settings3: webview2_com::Microsoft::Web::WebView2::Win32::ICoreWebView2Settings3 =
            settings.cast()?;
        settings3.SetAreBrowserAcceleratorKeysEnabled(false)?;
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
    // 【为什么必须订阅 ProcessFailed】这是 WebView2 唯一一条"渲染进程 / GPU 进程 /
    // 浏览器进程没了"的**显式**信号。没有它，"页面被 OOM 杀掉"在插件侧与"用户一直没
    // 操作"完全同形 —— 全仓此前 0 次出现，所以用户报的 Out of Memory 在日志里连一行
    // 痕迹都没有。它同时也是自动重建的触发器。
    let weak = Rc::downgrade(state);
    unsafe {
        browser.add_ProcessFailed(
            &ProcessFailedEventHandler::create(Box::new(move |_, args| {
                let Some(args) = args else {
                    return Ok(());
                };
                let kind = process_failed_kind(&args);
                // 详细原因与退出码在 ICoreWebView2ProcessFailedEventArgs2 上；
                // 老版本运行时没有它，那就只报 kind。
                let (reason, exit) = match args.cast::<ICoreWebView2ProcessFailedEventArgs2>() {
                    Ok(detailed) => {
                        let mut raw = COREWEBVIEW2_PROCESS_FAILED_REASON::default();
                        let reason = if detailed.Reason(&mut raw).is_ok() {
                            process_failed_reason(raw)
                        } else {
                            "unknown"
                        };
                        let mut exit = 0i32;
                        let exit = if detailed.ExitCode(&mut exit).is_ok() {
                            exit
                        } else {
                            0
                        };
                        (reason, exit)
                    }
                    Err(_) => ("unknown", 0),
                };
                if let Some(state) = weak.upgrade() {
                    state.borrow_mut().on_process_failed(kind, reason, exit);
                } else {
                    crate::gui_log_line(&format!(
                        "renderer failed kind={kind} reason={reason} exit=0x{exit:08X}"
                    ));
                }
                Ok(())
            })),
            &mut token,
        )?;
    }
    let weak = Rc::downgrade(state);
    unsafe {
        browser.add_NavigationCompleted(
            &NavigationCompletedEventHandler::create(Box::new(move |_, args| {
                let Some(args) = args else {
                    return Ok(());
                };
                let mut ok = windows::core::BOOL(0);
                let success = args.IsSuccess(&mut ok).is_ok() && ok.as_bool();
                if success {
                    // 页面真的跑起来了：连续重建计数归零（下一次故障重新给三次机会）。
                    PAGE_REBUILDS.with(|count| count.set(0));
                    if let Some(state) = weak.upgrade() {
                        state.borrow_mut().on_page_ready();
                    }
                } else {
                    // 【为什么失败也要落一行】导航失败此前完全静默：界面停在上一帧，
                    // 日志里一行都没有，于是"白屏"只能靠猜。状态码是唯一能区分
                    // "资源缺失"与"渲染进程没了"的读数。
                    let mut status = COREWEBVIEW2_WEB_ERROR_STATUS::default();
                    let status = if args.WebErrorStatus(&mut status).is_ok() {
                        format!("0x{:08X}", status.0)
                    } else {
                        "unknown".to_string()
                    };
                    crate::gui_log_line(&format!("navigation failed status={status}"));
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
