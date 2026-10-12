//! 原生WebView2子窗口：借用REAPER消息循环，取消异步创建，不启动Tauri/app。
use crate::sync::MutexExt;
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::ffi::c_void;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::sync::{Mutex, OnceLock};
use webview2_com::Microsoft::Web::WebView2::Win32::*;
use webview2_com::{
    AddScriptToExecuteOnDocumentCreatedCompletedHandler, CoTaskMemPWSTR,
    CreateCoreWebView2ControllerCompletedHandler, CreateCoreWebView2EnvironmentCompletedHandler,
    NavigationStartingEventHandler, NewWindowRequestedEventHandler, WebMessageReceivedEventHandler,
};
use windows::core::{w, Interface, PCWSTR, PWSTR};
use windows::Win32::Foundation::{HANDLE, HINSTANCE, HMODULE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_APARTMENTTHREADED};
use windows::Win32::System::LibraryLoader::GET_MODULE_HANDLE_EX_FLAG_PIN;
use windows::Win32::System::LibraryLoader::{
    GetModuleFileNameW, GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
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
static NEXT_VIEW: AtomicU64 = AtomicU64::new(1);
static WINDOWS: AtomicUsize = AtomicUsize::new(0);
static CLASS_LOCK: Mutex<bool> = Mutex::new(false);
static WEBVIEW_MODULE_PIN: OnceLock<Result<(), String>> = OnceLock::new();

/// 在途请求上限。
const PENDING_BUDGET: usize = 32;
/// 在途请求的存活时间。超过它仍未收到回复的请求视为"永远不会回"。
///
/// 【为什么是 30 秒】与各处 `HostReply::until` 的等待窗口一致：正常的宿主回流
/// （一次 `get_timeline_state` + ARA 重认领）在数百毫秒内完成；30 秒仍无回复只可能是
/// 请求已被丢弃。超时后剔除该条目，避免永久占用预算。
const PENDING_TTL: std::time::Duration = std::time::Duration::from_secs(30);

/// 单个编辑请求最多重投多少次（等待宿主几何/媒体追上）。
///
/// 【为什么必须有上限】回复泵由 20 ms 定时器驱动。没有上限时，一个**永不收敛**的
/// 等待会以 50 Hz 空转到 `until`（30 s ≈ 1500 次 `get_timeline_state`），每次都争
/// 同一把文档事务。有了预算，失败在几秒内转成明确的终态错误，用户可以据此处置。
const HOST_REPLY_RETRY_BUDGET: u32 = 24;

/// 第 `attempt` 次重投前的等待：20 ms 起指数退避，上限 500 ms。
///
/// 【为什么起点仍是 20 ms】宿主追上几何通常只要一两帧；第一档保持与定时器同频，
/// 不牺牲正常情况的响应速度，只把"长期不收敛"从 50 Hz 压到 2 Hz。
fn host_reply_retry_backoff(attempt: u32) -> std::time::Duration {
    // attempt 从 1 起算（第一次重投前），所以第 1 档就是 20 ms。
    let shift = attempt.saturating_sub(1).min(5);
    let millis = 20_u64.saturating_mul(1_u64 << shift).min(500);
    std::time::Duration::from_millis(millis)
}

/// 把前端上报的错误文本折成一个**稳定、有界**的限流键片段。
///
/// 【为什么截断】键里带的是运行期数据（方法名、错误摘要、栈）。不截断就等于把
/// 整条栈当成键 —— 而栈里往往含地址/行号这类每次都变的内容，会让"同一类错误"
/// 每次都落进新桶，限流器形同虚设。64 字符足以区分 invoke 方法名与错误分类，
/// 又不会让键本身变成新的日志体量。
fn truncate_key(text: &str) -> String {
    const MAX: usize = 64;
    if text.len() <= MAX {
        return text.to_owned();
    }
    // 按字符边界截断，避免在多字节 UTF-8 中间切开造成 panic。
    let mut end = MAX;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_owned()
}

/// 一组 clip id → 宿主 `I_GROUPID`（`1..=host_edit::HOST_GROUP_ID_MAX`）。
///
/// 【为什么是哈希而不是"显式序号"】序号要从当前编组集合取 `max+1`，而本处理器在
/// UI 线程读到的 `editor.timeline` 未必已含刚提交的上一次编组 ⇒ 两次快速编组会取到
/// 同一个序号，反而把两组并成一组。哈希只依赖**本次选择**：同一选择恒等（幂等），
/// 不受提交时序影响。
///
/// 【为什么必须折进区间】旧实现 `raw & 0x7fff_ffff` 会给出到 `2^31-1` 的值，而宿主
/// 只收 `0..=HOST_GROUP_ID_MAX`：约 7% 的编组会撞上 "invalid host item group id"，
/// 整条 `set_clips_state_bulk` 被拒。取模后用到全部 32 位熵，同时保证落在区间内。
fn host_group_id_for(clip_ids: &[String]) -> i32 {
    let joined = clip_ids.join("\n");
    let digest = blake3::hash(joined.as_bytes());
    let mut bytes = [0_u8; 4];
    bytes.copy_from_slice(&digest.as_bytes()[..4]);
    (u32::from_le_bytes(bytes) % crate::host::host_edit::HOST_GROUP_ID_MAX as u32 + 1) as i32
}

struct BrowserState {
    hwnd: HWND,
    closed: bool,
    view_id: String,
    controller: Option<ICoreWebView2Controller>,
    error_window: Option<HWND>,
    link: std::sync::Arc<super::routing::EditorLink>,
    sink: super::session::UiSink,
    replies: mpsc::Receiver<serde_json::Value>,
    events: mpsc::Receiver<serde_json::Value>,
    /// 在途请求 id → 插入时刻。
    ///
    /// 【为什么带时刻】`pending` 只在**收到回复**时移除（`deliver` 的 `retain_mut`）。
    /// 若某个请求永远不回（文档关闭、`PostWebMessageAsJson` 失败、actor 丢弃），条目会
    /// 永久残留；累积到 32（各处的预算上限）后，本视图会**拒绝一切后续命令**。
    /// 带时刻就能在定时器里超时剔除，把"永久失效"降级成"这一次超时"。
    pending: std::collections::HashMap<u64, std::time::Instant>,
    forwarded_keys: HashSet<char>,
    host_replies: std::collections::HashMap<u64, HostReply>,
    undo_requests:
        std::collections::HashMap<u64, std::sync::Weak<crate::render::document::DocumentSession>>,
    undo_document: Option<std::sync::Weak<crate::render::document::DocumentSession>>,
    history_requests: HashSet<u64>,
    // 首位是连接绑定标识（route token + 重绑代数，见 `EditorLink::route_identity`），
    // 不是 `Arc::as_ptr`：地址回收复用后旧指针可能命中陈旧缓存（撤销历史来自
    // 上一份文档）。
    history_cache: Option<((String, u64, i32, i32), serde_json::Value)>,
    last_history: Option<serde_json::Value>,
}

impl BrowserState {
    /// 记录一个在途请求。
    fn mark_pending(&mut self, id: u64) {
        self.pending.insert(id, std::time::Instant::now());
    }

    /// 在途请求是否已满（或该 id 已在途）。
    ///
    /// 先剔除超时条目，再判预算 —— 否则一个永不回复的请求会永久占用一个名额。
    fn pending_full(&mut self, id: u64) -> bool {
        self.expire_pending();
        self.pending.len() >= PENDING_BUDGET || self.pending.contains_key(&id)
    }

    /// 剔除超过 [`PENDING_TTL`] 仍无回复的请求，并同步释放它占用的关联状态。
    ///
    /// 只删 `pending` 条目还不够：`host_replies` / `undo_requests` / `history_requests`
    /// 都以同一 id 为键，留着会泄漏内存，也会让宿主 Undo 块一直开着（`finish_request`
    /// 收不到）。所以这里把它们一并清掉。
    fn expire_pending(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let now = std::time::Instant::now();
        let expired: Vec<u64> = self
            .pending
            .iter()
            .filter(|(_, at)| now.duration_since(**at) >= PENDING_TTL)
            .map(|(id, _)| *id)
            .collect();
        if expired.is_empty() {
            return;
        }
        let view_id = self.view_id.clone();
        for id in &expired {
            self.pending.remove(id);
            let reply = self.host_replies.remove(id);
            let document = reply
                .as_ref()
                .and_then(|reply| reply.document.upgrade())
                .or_else(|| self.undo_document.as_ref().and_then(|weak| weak.upgrade()));
            let undo = self
                .undo_requests
                .remove(id)
                .and_then(|weak| weak.upgrade());
            let history = self.history_requests.remove(id);
            if let Some(document) = undo {
                document.host_undo.finish_request(&view_id, *id);
            }
            if history {
                if let Some(document) = &document {
                    document.host_undo.finish_history(&view_id, *id);
                }
            }
            log::warn!("[ara] editor request {id} expired without a reply");
        }
    }
}
#[derive(Clone, Copy)]
enum HistoryJump {
    Undo,
    Redo,
    Position(i32),
}
#[derive(Clone)]
struct MediaAction {
    command: String,
    input: serde_json::Value,
}
/// 宿主写完后只重试读取ARA回流，不重复setter；保存最初的document/route租约。
struct HostReply {
    editor: Arc<super::session::EditorSession>,
    document: std::sync::Weak<crate::render::document::DocumentSession>,
    lease: u64,
    until: std::time::Instant,
    /// 已重投次数与下一次允许重投的时刻。
    ///
    /// 【为什么需要退避】回复泵由 20 ms `WM_TIMER` 驱动，此前只要判据仍成立就**每一 tick
    /// 重投一次**（50 Hz，直到 30 s 的 `until`）。宿主追上几何通常只需一两帧，这种
    /// 线性热重投让"等待收敛"变成 1500 次无意义的 `get_timeline_state`，与渲染线程争
    /// 同一把文档事务。改为指数退避（20→40→80…≤500 ms）并设总次数上限，超限即走明确的
    /// 终态错误，而不是继续空转。
    attempts: u32,
    next_attempt: std::time::Instant,
    jump: Option<HistoryJump>,
    imported: Option<String>,
    geometry: Option<crate::host::host_edit::HostEditReceipt>,
    split: Option<crate::host::host_split::SplitReceipt>,
    media: Option<crate::host::clipboard::MediaReceipt>,
    action: Option<MediaAction>,
}
struct WindowData {
    state: Rc<RefCell<BrowserState>>,
    transferred: Rc<Cell<bool>>,
    token: usize,
}
impl Drop for WindowData {
    fn drop(&mut self) {
        let (documents, view_id) = {
            let state = self.state.borrow();
            let documents = state
                .undo_document
                .iter()
                .cloned()
                .chain(state.undo_requests.values().cloned())
                .chain(
                    state
                        .host_replies
                        .values()
                        .map(|reply| reply.document.clone()),
                )
                .collect::<Vec<_>>();
            (documents, state.view_id.clone())
        };
        let controller = {
            let mut state = self.state.borrow_mut();
            state.closed = true;
            state.sink.closed.store(true, Ordering::Release);
            state.controller.take()
        };
        for document in documents.into_iter().filter_map(|doc| doc.upgrade()) {
            document.host_undo.release_view(&view_id);
        }
        // Close会同步回调；此时不持有RefCell borrow。
        if let Some(controller) = controller {
            let _ = unsafe { controller.Close() };
        }
        // 【为什么不在这里删用户数据目录】目录现在是**每进程**一份、被本进程的其它视图
        // 共享（见 `begin_browser`），关掉一个视图不能把还在用的那份删掉。回收改由
        // `temp_sweep::sweep_stale_temp_dirs` 在下一次启动时按"属主进程已退出"进行。
        let _ = unsafe { KillTimer(Some(self.state.borrow().hwnd), 0x4853) };
        unsafe {
            CoUninitialize();
        }
    }
}

/// 只保存窗口身份，不把非Send COM接口从UI线程带走。资源由自有WndProc析构。
pub(super) struct NativeEditor {
    key: WindowKey,
}
#[derive(Clone, Copy)]
pub(super) struct WindowKey {
    hwnd: usize,
    thread: u32,
    token: usize,
}
impl WindowKey {
    /// 句柄被系统复用后不能移动/关闭另一实例窗口；属性token在NCDESTROY时撤销。
    fn valid(self) -> bool {
        let hwnd = HWND(self.hwnd as *mut c_void);
        unsafe {
            IsWindow(Some(hwnd)).as_bool()
                && GetPropW(hwnd, w!("HiFiShifter.ARA.ViewToken")).0 as usize == self.token
        }
    }
    /// 只在宿主UI线程把焦点交给自有编辑器；已在浏览器内时不重置DOM焦点。
    pub(super) fn focus(self) -> Result<(), String> {
        if unsafe { GetCurrentThreadId() } != self.thread || !self.valid() {
            return Err("focus on stale window or off UI thread".into());
        }
        let hwnd = HWND(self.hwnd as *mut c_void);
        let current = unsafe { GetFocus() };
        if current == hwnd || unsafe { IsChild(hwnd, current) }.as_bool() {
            return Ok(());
        }
        let result = unsafe { SetFocus(Some(hwnd)) };
        // 首次SetFocus的旧句柄可能为null，Win32包装返回Err但真实焦点已成功设置。
        let focused = unsafe { GetFocus() };
        if focused == hwnd || unsafe { IsChild(hwnd, focused) }.as_bool() {
            Ok(())
        } else {
            result.map(|_| ()).map_err(|e| e.to_string())
        }
    }
    /// 可复制的窗口身份不持有COM资源，允许调用前释放view锁以防Win32同步重入。
    pub(super) fn resize(self, width: i32, height: i32) -> Result<(), String> {
        if unsafe { GetCurrentThreadId() } != self.thread || !self.valid() {
            return Err("resize on stale window or off UI thread".into());
        }
        unsafe { MoveWindow(HWND(self.hwnd as *mut c_void), 0, 0, width, height, true) }
            .map_err(|e| e.to_string())
    }
    /// 只向真实活WebView交付共享编辑键；不向外层HWND伪造无效Win32键盘消息。
    pub(super) fn forward_edit_key(self, key: char, mods: i16, down: bool) -> bool {
        if unsafe { GetCurrentThreadId() } != self.thread || !self.valid() {
            return false;
        }
        let hwnd = HWND(self.hwnd as *mut c_void);
        let pointer = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *const WindowData;
        if pointer.is_null() {
            return false;
        }
        let state = unsafe { (*pointer).state.clone() };
        let (controller, view_id) = {
            let state = state.borrow();
            if state.closed {
                return false;
            }
            (state.controller.clone(), state.view_id.clone())
        };
        let Some(controller) = controller else {
            return false;
        };
        let Ok(browser) = (unsafe { controller.CoreWebView2() }) else {
            return false;
        };
        let repeat = {
            let mut state = state.borrow_mut();
            if !down && !state.forwarded_keys.contains(&key) {
                return false;
            }
            if down {
                !state.forwarded_keys.insert(key)
            } else {
                state.forwarded_keys.remove(&key);
                false
            }
        };
        let message=wide(&serde_json::json!({"version":1,"viewId":view_id,"event":"plugin_keyboard",
            "payload":{"type":if down {"keydown"} else {"keyup"},"key":key.to_string(),
                "ctrlKey":true,"shiftKey":mods&1!=0,"altKey":mods&2!=0,"metaKey":mods&8!=0,"repeat":repeat}}).to_string());
        unsafe { browser.PostWebMessageAsJson(PCWSTR(message.as_ptr())) }.is_ok()
    }
}
impl NativeEditor {
    /// 同线程创建子窗口，异步浏览器由宿主消息循环完成，禁止嵌套消息泵。
    pub(super) fn attach(
        parent: *mut c_void,
        width: i32,
        height: i32,
        link: std::sync::Arc<super::routing::EditorLink>,
    ) -> Result<Self, String> {
        let parent = HWND(parent);
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
        let token = NEXT_VIEW.fetch_add(1, Ordering::Relaxed) as usize;
        let view_id = format!("view-{}-{token}", std::process::id());
        let (reply, replies) = mpsc::channel();
        let (event_sender, events) = mpsc::sync_channel(128);
        let sink = super::session::UiSink {
            view_id: view_id.clone(),
            reply,
            events: event_sender,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let state = Rc::new(RefCell::new(BrowserState {
            hwnd: HWND::default(),
            closed: false,
            view_id,
            controller: None,
            error_window: None,
            link,
            sink,
            replies,
            events,
            pending: std::collections::HashMap::new(),
            forwarded_keys: HashSet::new(),
            host_replies: std::collections::HashMap::new(),
            undo_requests: std::collections::HashMap::new(),
            undo_document: None,
            history_requests: HashSet::new(),
            history_cache: None,
            last_history: None,
        }));
        let transferred = Rc::new(Cell::new(false));
        let data = Box::into_raw(Box::new(WindowData {
            state: state.clone(),
            transferred: transferred.clone(),
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
                // NCCREATE之前失败时系统未接管data；之后失败则NCDESTROY已负责释放。
                if !transferred.get() {
                    unsafe {
                        drop(Box::from_raw(data));
                    }
                }
                return Err(format!("create child window: {error}"));
            }
        };
        let native = Self {
            key: WindowKey {
                hwnd: hwnd.0 as usize,
                thread,
                token,
            },
        };
        if unsafe { SetTimer(Some(hwnd), 0x4853, 20, None) } == 0 {
            show_error(&state, "UI response timer unavailable");
        }
        if let Err(error) = begin_browser(&state) {
            show_error(&state, &error);
            // attach 半失败：没有可投递的 browser，`deliver` 无事可做、视图已死，
            // 20ms 定时器只会空转（50Hz 全速泵一块错误 STATIC）。停掉它。
            unsafe {
                let _ = KillTimer(Some(hwnd), 0x4853);
            }
        }
        Ok(native)
    }
    /// 提供不持COM资源的几何操作身份，供view在释放锁后使用。
    pub(super) fn window_key(&self) -> WindowKey {
        self.key
    }
}
impl Drop for NativeEditor {
    fn drop(&mut self) {
        let hwnd = HWND(self.key.hwnd as *mut c_void);
        if !self.key.valid() {
            return;
        }
        if unsafe { GetCurrentThreadId() } == self.key.thread {
            let _ = unsafe { DestroyWindow(hwnd) };
        } else {
            // COM由自有窗口线程Close，不在错误线程释放。正常REAPER UI析构走上分支。
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
fn assets() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("HIFISHIFTER_ARA_EDITOR_ASSETS") {
        let path = PathBuf::from(path);
        if !path.is_absolute() {
            return Err("editor asset override must be absolute".into());
        }
        let path = path.canonicalize().map_err(|e| e.to_string())?;
        if !path.join("plugin.html").is_file() {
            return Err("editor plugin.html missing".into());
        }
        return Ok(path);
    }
    let mut buffer = vec![0u16; 32768];
    let length = unsafe { GetModuleFileNameW(Some(module()?), &mut buffer) } as usize;
    if length == 0 || length >= buffer.len() {
        return Err("module path unavailable".into());
    }
    let file = PathBuf::from(String::from_utf16_lossy(&buffer[..length]));
    let folder = file
        .parent()
        .and_then(|p| p.parent())
        .ok_or("VST3 Contents directory unavailable")?
        .join("Resources")
        .join("frontend");
    if !folder.join("plugin.html").is_file() {
        return Err(format!(
            "plugin frontend assets missing: {}",
            folder.display()
        ));
    }
    Ok(folder)
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
        return Err(windows::core::Error::from_win32().to_string());
    }
    *registered = true;
    Ok(())
}
/// 不留指向已卸载DLL的window class；打开中的child明确阻止正常卸载。
pub(super) fn shutdown() -> bool {
    if WINDOWS.load(Ordering::Acquire) != 0 {
        crate::log_line("ExitDll refused: native editor windows still alive");
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

unsafe extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // 所有可能panic的自有逻辑都在边界内，不能越过Win32回调栈。
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if message == WM_NCCREATE {
            let create = &*(lparam.0 as *const CREATESTRUCTW);
            let data = &*(create.lpCreateParams as *const WindowData);
            data.transferred.set(true);
            data.state.borrow_mut().hwnd = hwnd;
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
                // host getter可能重入并关闭窗口：先持Rc、释放RefCell借用，不再在调用后解引用WindowData。
                let state = (*pointer).state.clone();
                let link = state.borrow().link.clone();
                if let Ok(owner) = link.owner() {
                    owner.refresh_host_transport_tick();
                }
                if !state.borrow().closed {
                    deliver(&state);
                }
                return LRESULT(0);
            } else if message == WM_SETFOCUS {
                // 宿主Tab/onFocus进入自有HWND后，使用标准WebView2接口进入HTML控件。
                // MoveFocus可能同步通知宿主；调用前释放BrowserState借用。
                let controller = (*pointer).state.borrow().controller.clone();
                if let Some(controller) = controller {
                    let _ = controller.MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_NEXT);
                }
                return LRESULT(0);
            } else if message == WM_SIZE {
                let controller = (*pointer).state.borrow().controller.clone();
                if let Some(controller) = controller {
                    let mut bounds = RECT::default();
                    if GetClientRect(hwnd, &mut bounds).is_ok() {
                        let _ = controller.SetBounds(bounds);
                    }
                }
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

/// worker只写JSON邮箱；COM响应仅在本窗口UI线程发送，且不持RefCell borrow。
fn deliver(state: &Rc<RefCell<BrowserState>>) {
    let link = state.borrow().link.clone();
    // 私有状态通知必须先于可能收尾的Undo块；反过来会在宿主已记录状态后才标脏。
    link.flush_dirty();
    if let Ok(owner) = link.owner() {
        if let Ok(document) = owner.editor_document() {
            document.host_undo.tick();
        }
    }
    let (controller, mut replies, events) = {
        let state = state.borrow();
        if state.closed || state.controller.is_none() {
            return;
        }
        let replies = state.replies.try_iter().take(32).collect::<Vec<_>>();
        let events = state.events.try_iter().take(64).collect::<Vec<_>>();
        (state.controller.clone(), replies, events)
    };
    let Some(controller) = controller else {
        return;
    };
    let Ok(browser) = (unsafe { controller.CoreWebView2() }) else {
        return;
    };
    // 在宿主模型/PCM尚未ready的过渡期，保留原Promise。只排actor读取，不在UI做文件/推理。
    replies.retain_mut(|reply| {
        let Some(id)=reply["id"].as_u64() else {return true;};
        // 媒体复制需要先越过actor写入屏障；API仍在本UI线程，调用期间不借用BrowserState。
        // 【为什么用单次 `get` 而不是 `get` 之后再 `[&id]`】后者是 `Index` panic 形状：一旦
        // 将来这段改成跨借用边界，缺失的条目会直接 panic 在宿主进程里。这里一次取到
        // 需要的字段再克隆出来。
        let action={let state=state.borrow();state.host_replies.get(&id).and_then(|host|{
            host.action.clone().map(|action|(action,host.document.clone(),host.lease,host.editor.clone(),state.sink.clone(),state.view_id.clone()))})};
        if let Some((action,weak,lease,editor,sink,view_id))=action.filter(|_|reply["ok"]==true) {
            let outcome=(||->Result<bool,String>{
                let document=weak.upgrade().ok_or("media document closed")?;
                if state.borrow().host_replies.get(&id).is_some_and(|host|std::time::Instant::now()>=host.until) {return Err("media request timed out".into());}
                let allowed=||link.authorize(&document).is_ok_and(|current|current==lease);
                if !allowed() {return Err("media editor lease changed".into());}
                state.borrow_mut().host_replies.get_mut(&id).ok_or("media request closed")?.action=None;
                let owner=link.owner()?;
                if action.command=="copy_timeline_clips" {reply["value"]=crate::host::clipboard::copy(&owner,&editor,&action.input,&allowed)?;return Ok(true);}
                let host=owner.project_history_host().filter(|host|host.can_clipboard_items()).ok_or("host media clipboard API unavailable")?;
                let duplicated=action.command=="duplicate_clips_bulk";
                let pasted=action.command=="paste_timeline_clipboard"||duplicated;
                // 计划只读预检；native Undo开始后才创建/删除，失败也必须结束请求。
                let paste=if duplicated {Some(crate::host::clipboard::plan_duplicate(&owner,&editor,&action.input,&allowed)?)} else if pasted {Some(crate::host::clipboard::plan_paste(&owner,&editor,&action.input,&allowed)?)} else {None};
                let delete=if pasted {None} else {Some(crate::host::clipboard::plan_delete(&owner,&editor,&action.input,action.command=="remove_clip",&allowed)?)};
                document.host_undo.begin_request(&view_id,id,true,true,&host,&allowed)?;
                state.borrow_mut().undo_requests.insert(id,Arc::downgrade(&document));
                let receipt=if let Some(plan)=paste {crate::host::clipboard::execute_paste(&owner,plan,&allowed)?}
                    else {crate::host::clipboard::execute_delete(&owner,delete.unwrap(),&allowed)?};
                // 所有native写入已结束；回执只读等待不能撑住宿主Undo块/自动准备，形成相互等待。
                // 失败仍由下方统一收尾；成功回执继续核对真实GUID，不重复创建item。
                state.borrow_mut().undo_requests.remove(&id);
                document.host_undo.finish_request(&view_id,id);
                state.borrow_mut().host_replies.get_mut(&id).ok_or("media view closed during write")?.media=Some(receipt);
                editor.enqueue(super::session::UiRequest {id,command:"get_timeline_inventory".into(),args:serde_json::json!({}),sink,link:Some(link.clone())})?;
                Ok(false)
            })();
            match outcome {Ok(false)=>return false,Ok(true)=>{},Err(error)=>{reply["ok"]=serde_json::json!(false);reply["error"]=serde_json::json!(error);}}
        }
        // Undo/Redo的actor屏障到达后才调用宿主；该外部调用期间不持BrowserState借用。
        let jump={let state=state.borrow();state.host_replies.get(&id).and_then(|host|{
            host.jump.map(|redo|(redo,host.document.clone(),host.lease,host.editor.clone(),state.sink.clone()))})};
        if let Some((jump,weak,lease,editor,sink))=jump {
            let result=(||->Result<bool,String>{
                let document=weak.upgrade().ok_or("undo document closed")?;
                if state.borrow().host_replies.get(&id).is_some_and(|host|std::time::Instant::now()>=host.until) {return Err("host history request timed out".into());}
                let allowed=||link.authorize(&document).is_ok_and(|current|current==lease);
                if !allowed() {return Err("undo editor lease changed".into());}
                if !document.host_undo.finish_for_history() {
                    editor.enqueue(super::session::UiRequest {id,command:"plugin_history_barrier".into(),args:serde_json::json!({}),sink,link:Some(link.clone())})?;
                    return Ok(false);
                }
                let host=link.owner()?.project_history_host().ok_or("host history unavailable")?;
                match jump {HistoryJump::Undo=>{host.history_jump(false,&allowed)?;},HistoryJump::Redo=>{host.history_jump(true,&allowed)?;},HistoryJump::Position(index)=>{host.history_jump_to(index,&allowed)?;}}
                // 【为什么用 `get_mut` 而不是 `[&id]`/`unwrap`】上面的 `history_jump` 是
                // **宿主回调**，其间不持 BrowserState 借用（见本段的注释），因此它可能
                // 重入并改动 `host_replies`。用索引会 panic，而这里跑在宿主进程里。
                if let Some(entry)=state.borrow_mut().host_replies.get_mut(&id) {entry.jump=None;}
                editor.enqueue(super::session::UiRequest {id,command:"get_timeline_state".into(),args:serde_json::json!({}),sink,link:Some(link.clone())})?;
                Ok(false)
            })();
            match result {Ok(false)=>return false,Ok(true)=>{},Err(error)=>{reply["ok"]=serde_json::json!(false);reply["error"]=serde_json::json!(error);}}
        }
        let mut state=state.borrow_mut();
        if let Some(host)=state.host_replies.get(&id) {
            let error=reply["error"].as_str().unwrap_or("").to_owned();
            let imported=host.imported.as_ref().and_then(|item|host.document.upgrade().and_then(|doc|doc.clip_for_host_item(item))).map(|clip|host.editor.ui_clip_id(&clip));
            let waiting=host.imported.is_some()&&(imported.is_none()||!reply["value"]["clips"].as_array().is_some_and(|clips|clips.iter().any(|clip|Some(clip["id"].as_str().unwrap_or(""))==imported.as_deref())));
            let geometry_waiting=reply["ok"]==true&&host.geometry.as_ref().is_some_and(|geometry|!geometry.matches(&reply["value"]));
            let split_waiting=reply["ok"]==true&&host.split.as_ref().is_some_and(|split|!host.document.upgrade().is_some_and(|doc|split.matches(&doc,&host.editor.namespace,&mut reply["value"])));
            let media_waiting=reply["ok"]==true&&host.media.as_ref().is_some_and(|media|!host.document.upgrade().is_some_and(|doc|media.matches(&doc,&host.editor.namespace,&mut reply["value"])));
            let now = std::time::Instant::now();
            let retryable = waiting
                || geometry_waiting
                || split_waiting
                || media_waiting
                || reply["ok"] == false
                    && (error.starts_with(crate::editor::errors::CONFLICT_PREFIX)
                        || error.contains(crate::editor::errors::HOST_PCM_UNAVAILABLE)
                        || error.contains(crate::editor::errors::HOST_MODEL_NOT_READY));
            // 退避：只在到点且未超预算时重投。预算与 `until` 双保险 —— 前者挡住
            // "永不收敛的失败"（否则 50 Hz 空转到 30 s），后者挡住"慢宿主"。
            let can_retry = host.attempts < HOST_REPLY_RETRY_BUDGET && now >= host.next_attempt;
            if retryable && now < host.until && can_retry {
                let allowed = host.document.upgrade().is_some_and(|doc| {
                    state
                        .link
                        .authorize(&doc)
                        .is_ok_and(|lease| lease == host.lease)
                });
                let command = if host.media.is_some() {
                    "get_timeline_inventory"
                } else {
                    "get_timeline_state"
                };
                let editor = host.editor.clone();
                let sink = state.sink.clone();
                let link = state.link.clone();
                let enqueued = allowed
                    && editor
                        .enqueue(super::session::UiRequest {
                            id,
                            command: command.into(),
                            args: serde_json::json!({}),
                            sink,
                            link: Some(link),
                        })
                        .is_ok();
                if enqueued {
                    // 单独取可变借用推进退避状态（上面的只读借用到此结束）。
                    if let Some(host) = state.host_replies.get_mut(&id) {
                        host.attempts = host.attempts.saturating_add(1);
                        host.next_attempt = now + host_reply_retry_backoff(host.attempts);
                    }
                    return false;
                }
            }
            if waiting {reply["ok"]=serde_json::json!(false);reply["error"]=serde_json::json!("item was created in the host but its ARA audio is not ready; use host Undo if canceling");}
            if geometry_waiting {reply["ok"]=serde_json::json!(false);reply["error"]=serde_json::json!("clip was edited in the host but its ARA geometry has not caught up; host Undo remains available");}
            if split_waiting {reply["ok"]=serde_json::json!(false);reply["error"]=serde_json::json!("item was split in the host but both clips have not caught up; host Undo remains available");}
            if media_waiting {reply["ok"]=serde_json::json!(false);reply["error"]=serde_json::json!("media was edited in the host but its GUI/ARA identities have not caught up; host Undo remains available");}
            if !waiting&&host.imported.is_some() {reply["value"]["imported_clip_id"]=serde_json::json!(imported);}
        }
        let done_history=state.history_requests.remove(&id);let view_id=state.view_id.clone();
        let finished_undo=state.undo_requests.remove(&id).and_then(|document|document.upgrade());
        let document=state.host_replies.remove(&id).and_then(|host|host.document.upgrade());state.pending.remove(&id);drop(state);
        if let Some(document)=finished_undo {document.host_undo.finish_request(&view_id,id);}
        if done_history {if let Some(document)=document {document.host_undo.finish_history(&view_id,id);}}
        true
    });
    replies.extend(events);
    let history = link.owner().ok().and_then(|owner| {
        let document = owner.editor_document().ok()?;
        let lease = link.authorize(&document).ok()?;
        let host = owner.project_history_host()?;
        let allowed = || {
            link.authorize(&document)
                .is_ok_and(|current| current == lease)
        };
        let token = host.history_token(&allowed).ok()?;
        let identity = link.route_identity()?;
        let key = (identity.0, identity.1, token.0, token.1);
        if let Some((_, cached)) = state
            .borrow()
            .history_cache
            .as_ref()
            .filter(|(known, _)| *known == key)
        {
            return Some(cached.clone());
        }
        let value = host.project_history(&allowed).ok()?;
        state.borrow_mut().history_cache = Some((key, value.clone()));
        Some(value)
    });
    if let Some(history) = &history {
        for response in &mut replies {
            if response["event"] == "history_state" {
                response["payload"] = history.clone();
            }
            if response["value"]["clips"].is_array() {
                response["value"]["undo_depth"] = history["undoDepth"].clone();
                response["value"]["redo_depth"] = history["redoDepth"].clone();
            }
        }
        let changed = state.borrow().last_history.as_ref() != Some(history);
        if changed {
            state.borrow_mut().last_history = Some(history.clone());
            replies.push(serde_json::json!({"version":1,"viewId":state.borrow().view_id,"event":"history_state","payload":history}));
        }
    }
    for response in replies {
        let mut text = response.to_string();
        if text.len() > 8 * 1024 * 1024 {
            text = serde_json::json!({"version":1,"viewId":response["viewId"],"id":response["id"],
                "ok":false,"error":"native response budget exceeded"})
            .to_string();
        }
        let text = wide(&text);
        if unsafe { browser.PostWebMessageAsJson(PCWSTR(text.as_ptr())) }.is_err() {
            break;
        }
    }
}

fn show_error(state: &Rc<RefCell<BrowserState>>, error: &str) {
    crate::log_line(&format!("Native editor error: {error}"));
    let (closed, hwnd) = {
        let state = state.borrow();
        (state.closed, state.hwnd)
    };
    if !closed {
        let text = wide(&format!("HiFiShifter editor: {error}"));
        let existing = state.borrow().error_window;
        if let Some(existing) = existing {
            let _ = unsafe { SetWindowTextW(existing, PCWSTR(text.as_ptr())) };
        } else {
            // 自有child没有标题栏；用真实STATIC控件显示错误，不能只写不可见window text。
            if let Ok(error_window) = unsafe {
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
                state.borrow_mut().error_window = Some(error_window);
            }
        }
    }
}

/// WebView异步回调仅捕获weak state，removed后不会重新创建或访问裸view。
fn begin_browser(state: &Rc<RefCell<BrowserState>>) -> Result<(), String> {
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
                            show_error(&state, &error.to_string());
                            return Ok(());
                        }
                        if let Some(controller) = controller {
                            state.borrow_mut().controller = Some(controller.clone());
                            if let Err(error) = configure_browser(&state, &controller, &folder) {
                                show_error(&state, &error.to_string());
                            }
                        } else {
                            show_error(&state, "WebView2 returned no controller");
                        }
                        Ok(())
                    }),
                );
                let hwnd = state.borrow().hwnd;
                unsafe { environment.CreateCoreWebView2Controller(hwnd, &controller_ready) }
            })();
            if let Err(error) = outcome {
                show_error(&state, &error.to_string());
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

/// WebView 启动时注入的能力标志（`window.__HFS_PLUGIN_BOOTSTRAP__`）。
///
/// 【为什么单独成类型】前端按这些标志决定开放哪些操作（`services/hostCapabilities.ts`）。
/// 内联在一个 300 行的浏览器配置函数里时，**没人能对它写测试** —— 而这里恰好出过一次
/// 错（`trackGrouping` 被写成复用 `audioImport`，见 [`BootstrapFlags::to_json`]）。
///
/// 【为什么宿主写能力不再各占一个布尔】`clip_editing` / `clip_splitting` /
/// `clip_clipboard` / `audio_import` / `track_creation` 曾经与 `hostCapabilities` 并存，
/// 由**两处**各自计算（一处按本实例的原生扩展、一处按 `project_history_host()`）——
/// 两份判据必然漂移。现在它们只存在于 `host_capabilities`（由
/// [`crate::render::extension::ExtensionOwner::effective_host_capabilities`] 统一算出），
/// 前端只读一处。
#[derive(Clone, Copy, Debug)]
struct BootstrapFlags {
    /// 宿主提供 ARA 播放请求（`ARAPlaybackControllerInterface`）。
    ///
    /// 【为什么**不**放进 `hostCapabilities`】它描述的是**插件**能否向宿主发播放请求，
    /// 不是宿主能否提供 transport 读数（后者是 `HostCapabilities::transport`）。两者是
    /// 不同的东西：ARA 播放控制器可用不代表宿主把播放位置喂给了插件，反之亦然。
    transport_control: bool,
    /// 宿主用哪一套淡化轴：`Some("legacy")`（≤7.80，`C_FADE*SHAPE` 决定形状）
    /// 或 `Some("continuous")`（≥7.81，curvature/S 两轴决定形状）。
    ///
    /// 【为什么是枚举而不是布尔】"能不能编辑"与"编辑的是哪套轴"是两件事：新轴宿主
    /// 上没有形状号可写，界面得把预设翻译成 `(curvature, S)` 一对分量。`None` =
    /// 版本读不出来 → 前端整块保持只读（不猜）。
    ///
    /// 这套翻译的映射是实测的（`hifishifter_kernel::fade_axes`，证据
    /// `docs/2026-10-09-fade-axis-findings.md`），所以新轴宿主上**也**摆七个预设按钮；
    /// 此前这里的注释写着"预设映射尚未校准，所以只给连续滑杆"，那条限制已随实测消失。
    fade_axes: Option<&'static str>,
    /// 撤销历史由谁承载：`"host"`（宿主权威栈）或 `"local"`（插件自管栈）。
    ///
    /// 【为什么前端需要它】两条栈共用同一个「操作记录」面板，但 `label` 的含义不同：
    /// 本地栈是 op key（按 `history_op_*` 本地化），宿主栈是宿主自己的可读字符串
    /// （原样显示）。缺了这个判别式，前端只能无差别加前缀 —— 那正是
    /// `history_op_Move items` 这类原始键名被摊给用户的原因。
    history_backend: &'static str,
    /// 当前后端身份：`"reaper"` 或 `"ara_only"`。
    ///
    /// 【为什么前端需要它】非 REAPER 宿主上片段几何**只读**（ARA 2.0 不允许插件改
    /// region 几何）。前端据此显示只读横幅，而不是让用户点了写操作再撞一个错误。
    host_kind: &'static str,
    /// **生效的**能力位快照：写片段 / 分割 / 剪贴板 / 导入 / 建轨 / 撤销栈的唯一来源。
    host_capabilities: hifishifter_kernel::editor::host::HostCapabilities,
}

impl BootstrapFlags {
    fn to_json(self, view_id: &str) -> serde_json::Value {
        serde_json::json!({
            "version": 1,
            "viewId": view_id,
            "transportControl": self.transport_control,
            "fadeAxes": self.fade_axes,
            "historyBackend": self.history_backend,
            // 后端身份与能力：前端据此显示只读横幅、禁用不可用的入口。
            // 【这是写能力的唯一来源】clipWrite / split / itemClipboard / mediaImport /
            // trackCreate / undoStack 都从这里读，不再有并存的顶层布尔。
            "hostKind": self.host_kind,
            "hostCapabilities": self.host_capabilities,
            // 【为什么是常量 true，而不是复用 mediaImport】它守卫的是
            // `move_private_track`（`editor/session.rs`）—— 那条路径只改插件自己的
            // timeline 与私有参数分组，注释里明写"不调用任何宿主轨道 setter"，因此
            // **不需要任何宿主写能力**。此前它写成 `"trackGrouping": audio_import`，
            // 于是一个纯插件的参数分组功能被"能不能导入音频"门住：宿主缺少媒体 API
            // 时，插件里的轨道拖动会无声失效，而它本来根本不需要那个 API。
            "trackGrouping": true,
        })
    }
}

fn configure_browser(
    state: &Rc<RefCell<BrowserState>>,
    controller: &ICoreWebView2Controller,
    folder: &std::path::Path,
) -> windows::core::Result<()> {
    let hwnd = state.borrow().hwnd;
    let mut bounds = RECT::default();
    unsafe {
        GetClientRect(hwnd, &mut bounds)?;
        controller.SetBounds(bounds)?;
        controller.SetIsVisible(true)?;
    }
    // 异步创建完成时仅延续宿主已交给本窗口的焦点，不能抢另一个FX/应用的焦点。
    if unsafe { GetFocus() } == hwnd {
        unsafe {
            controller.MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_NEXT)?;
        }
    }
    let browser = unsafe { controller.CoreWebView2()? };
    // 真实文件拖入由WebView2原生OLE接收，再通过AdditionalObjects/File.Path交给宿主。
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
    let weak = Rc::downgrade(state);
    unsafe {
        browser.add_WebMessageReceived(&WebMessageReceivedEventHandler::create(Box::new(move |browser,args| {
        let (Some(state),Some(browser),Some(args)) = (weak.upgrade(),browser,args) else { return Ok(()); };
        let view_id = {
            let state = state.borrow();
            if state.closed { return Ok(()); }
            state.view_id.clone()
        };
        let mut source = PWSTR::null();
        args.Source(&mut source)?;
        if !CoTaskMemPWSTR::from(source).to_string().starts_with(ORIGIN) { return Ok(()); }
        let mut json = PWSTR::null();
        args.WebMessageAsJson(&mut json)?;
        let text = CoTaskMemPWSTR::from(json).to_string();
        if text.len() > 1024*1024 { return Ok(()); }
        let Ok(mut request) = serde_json::from_str::<serde_json::Value>(&text) else { return Ok(()); };
        if request["version"] != 1 || request["viewId"] != view_id { return Ok(()); }
        let Some(id) = request["id"].as_u64().filter(|id| *id > 0 && *id <= 9_007_199_254_740_991) else { return Ok(()); };
        // 编组/解组仍使用同一宿主item setter与Undo块；把UI字符串groupId映射为有界REAPER I_GROUPID。
        if matches!(request["command"].as_str(),Some("group_clips")|Some("ungroup_clips")) {
            let ids=request["args"]["clipIds"].as_array().cloned().unwrap_or_default();
            if ids.is_empty()||ids.len()>512||ids.iter().any(|id|id.as_str().is_none()) {return Ok(());}
            let group=if request["command"]=="ungroup_clips" {0_i32} else {
                let names=ids.iter().filter_map(|id|id.as_str()).map(str::to_owned).collect::<Vec<_>>();
                host_group_id_for(&names)
            };
            request["command"]=serde_json::json!("set_clips_state_bulk");request["args"]=serde_json::json!({"updates":ids.into_iter().map(|id|serde_json::json!({"clipId":id,"hostGroupId":group})).collect::<Vec<_>>(),"checkpoint":true});
        }
        if request["command"]=="close_track_gaps" {
            let track=request["args"]["trackId"].as_str().unwrap_or("").to_owned();let from=request["args"]["fromSec"].as_f64().unwrap_or(f64::NAN);
            let link=state.borrow().link.clone();
            if let Ok(owner)=link.owner() {if let Ok(editor)=owner.editor_session() {
                let moves=editor.timeline.lock_or_recover().close_track_gaps_moves(&track,from);
                request["command"]=if moves.is_empty(){serde_json::json!("get_timeline_state")}else{serde_json::json!("move_clips")};
                request["args"]=if moves.is_empty(){serde_json::json!({})}else{serde_json::json!({"moves":moves,"moveLinkedParams":true})};
            }}
        }
        if request["command"]=="rename_clip_take" {
            let clip_id=request["args"]["clipId"].as_str().unwrap_or("").to_owned();
            let take_id=request["args"]["takeId"].as_str().unwrap_or("").to_owned();
            let name=request["args"]["name"].as_str().unwrap_or("").to_owned();let link=state.borrow().link.clone();
            if let Ok(owner)=link.owner() {if let Ok(editor)=owner.editor_session() {
                let active=editor.timeline.lock_or_recover().clips.iter().find(|clip|clip.id==clip_id).and_then(|clip|clip.active_take_id.clone());
                if active.as_deref()==Some(take_id.as_str()) {
                    // 只有 active take 的名字能落到宿主：`set_clip_state{name}` 写的是
                    // 宿主 item 的当前 take 名。非 active take 的重命名没有宿主对应
                    // （`SetTakeName` 只作用于 take 指针，而 ARA 侧只认 active take）。
                    request["command"]=serde_json::json!("set_clip_state");
                    request["args"]=serde_json::json!({"clipId":clip_id,"name":name,"checkpoint":request["args"]["checkpoint"]});
                } else {
                    // 【为什么给明确错误而不是让它掉进 catch-all】此前这条命令原样落到
                    // `Command unavailable in ARA plugin mode: rename_clip_take` ——
                    // 用户看到的是"命令不可用"，而不是"只有当前 Take 能改名"。后者才是
                    // 可操作的。
                    let reply=wide(&serde_json::json!({"version":1,"viewId":view_id,"id":id,"ok":false,
                        "error":"only the active take can be renamed in the ARA plugin; switch to it in the host first"}).to_string());
                    browser.PostWebMessageAsJson(PCWSTR(reply.as_ptr()))?;
                    return Ok(());
                }
            }}
        }
        // DOM File不经base64复制音频；路径仅从WebView2提供的真实File对象读取。
        if request["command"]=="import_native_audio_file" {
            let path=(||->Result<String,String>{
                if request.get("args").is_some_and(|value|!value.is_object()&&!value.is_null()) {return Err("native File args must be an object".into());}
                let extra:ICoreWebView2WebMessageReceivedEventArgs2=args.cast().map_err(|_|"native File bridge unavailable")?;
                let objects=extra.AdditionalObjects().map_err(|e|e.to_string())?;let mut count=0;objects.Count(&mut count).map_err(|e|e.to_string())?;
                if count!=1 {return Err("exactly one native File is required".into());}
                let file:ICoreWebView2File=objects.GetValueAtIndex(0).map_err(|e|e.to_string())?.cast().map_err(|_|"additional object is not a disk File")?;
                let mut path=PWSTR::null();file.Path(&mut path).map_err(|e|e.to_string())?;
                let path=CoTaskMemPWSTR::from(path).to_string();if !std::path::Path::new(&path).is_absolute() {return Err("dragged File has no native disk path; use File menu import".into());}Ok(path)
            })();
            match path {Ok(path)=>{request["command"]=serde_json::json!("import_audio_item");request["args"]["audioPath"]=serde_json::json!(path);},
                Err(error)=>{let reply=wide(&serde_json::json!({"version":1,"viewId":view_id,"id":id,"ok":false,"error":error}).to_string());browser.PostWebMessageAsJson(PCWSTR(reply.as_ptr()))?;return Ok(());}}
        }
        // 普通命令只排队；UI回调不得分析、推理、读文件或等待worker。
        let result = match request["command"].as_str() {
            Some("ping") => Ok(serde_json::json!({"ok":true,"mode":"plugin","view_id":view_id,
                "connected":state.borrow().link.owner().is_ok()})),
            Some("log_frontend_error") => {
                // 【为什么按 (message, detail) 分桶限流，而不是原样落盘】
                // 这个入口是**运行期数据驱动**的：每个失败的方法名是一类新错误。
                // 此前每条都原样写进日志，实测把一个终态错误（组件已关闭而页面仍在
                // 轮询）放大成 50 分钟 19,056 行、轮转文件合计约 32 MB 纯噪声 ——
                // 用户报障时真正有用的线索全被冲掉。
                // 现在每类错误各自 10 秒最多一条，且窗口内被抑制的条数会在下一条
                // 输出前以 `[throttled]` 汇总行补记（不丢信息，只丢重复）。
                let args = &request["args"];
                let message = args["message"].as_str().unwrap_or("");
                let detail = args["detail"].as_str().unwrap_or("");
                let key = format!(
                    "frontend:{}:{}",
                    truncate_key(message),
                    truncate_key(detail)
                );
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Info,
                    &key,
                    format_args!("Embedded frontend: {}", args),
                );
                Ok(serde_json::Value::Null)
            }
            Some(command @ ("open_audio_dialog"|"open_audio_dialog_multi"))=>{
                let link=state.borrow().link.clone();( ||->Result<serde_json::Value,String>{
                    let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                    let host=owner.project_history_host().filter(|host|host.can_import_audio()).ok_or("host audio import is unavailable")?;
                    let paths=host.pick_audio_paths(command=="open_audio_dialog_multi",&||link.authorize(&document).is_ok_and(|current|current==lease))?;
                    Ok(if command=="open_audio_dialog_multi" {serde_json::json!({"ok":true,"canceled":paths.is_empty(),"paths":paths})}
                        else {serde_json::json!({"ok":true,"canceled":paths.is_empty(),"path":paths.first()})})
                })()
            },
            Some("pick_directory")=>{
                let (link,hwnd)={let state=state.borrow();(state.link.clone(),state.hwnd)};
                (||->Result<serde_json::Value,String>{
                    let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                    let editor=owner.editor_session()?;
                    let path=super::browser_files::pick_directory(hwnd)?;
                    if link.authorize(&document)?!=lease {return Err("folder picker editor lease changed".into());}
                    match path {Some(path)=>Ok(serde_json::json!({"ok":true,"path":editor.grant_browser_directory(&path)?})),
                        None=>Ok(serde_json::json!({"ok":true,"canceled":true}))}
                })()
            },
            Some("import_audio_item")=>{
                let (link,sink,full)={let mut state=state.borrow_mut();(state.link.clone(),state.sink.clone(),state.pending_full(id))};
                let outcome=(||->Result<HostReply,String>{
                    if full {return Err("native import request budget exceeded".into());}
                    let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;let editor=owner.editor_session()?;
                    let input=&request["args"];let path=input["audioPath"].as_str().ok_or("audio path missing")?;
                    if input["mediaAudioStreamIndex"].as_u64().is_some_and(|index|index!=0) {return Err("specific container audio stream import is not supported yet".into());}
                    let track=input["trackId"].as_str().map(|id|editor.host_track_id(id)).transpose()?;
                    let allowed=||link.authorize(&document).is_ok_and(|current|current==lease);
                    let host=owner.project_history_host().ok_or("import project history missing")?;
                    let new_track=input["trackId"].is_null()&&input.as_object().is_some_and(|object|object.contains_key("trackId"));
                    let existing=if new_track {None} else {Some(owner.audio_import_target(track.as_deref(),&allowed)?)};
                    state.borrow_mut().undo_document=Some(Arc::downgrade(&document));
                    document.host_undo.begin_request(&view_id,id,true,true,&host,&allowed)?;
                    let result=(||->Result<String,String>{
                        let mut created=if new_track {Some(host.create_audio_track(std::path::Path::new(path).file_stem().and_then(|name|name.to_str()).unwrap_or("Imported audio"),None,&allowed)?)} else {None};
                        let target=existing.as_ref().or_else(||created.as_ref().map(|track|&track.target)).ok_or("import target missing")?;
                        let id=target.import_audio(path,input["startSec"].as_f64().unwrap_or_else(||owner.clock.get().map(|clock|clock.read().0).unwrap_or(0.)),&allowed)?;
                        if let Some(track)=&mut created {track.commit();}Ok(id)
                    })();
                    document.host_undo.finish_request(&view_id,id);let imported=result?;
                    owner.refresh_host_transport();editor.notify_timeline();
                    editor.enqueue(super::session::UiRequest {id,command:"get_timeline_state".into(),args:serde_json::json!({}),sink,link:Some(link.clone())})?;
                    Ok(HostReply {editor,document:Arc::downgrade(&document),lease,until:std::time::Instant::now()+std::time::Duration::from_secs(30),attempts:0,next_attempt:std::time::Instant::now(),jump:None,imported:Some(imported),geometry:None,split:None,media:None,action:None})
                })();
                match outcome {Ok(reply)=>{let mut state=state.borrow_mut();state.mark_pending(id);state.host_replies.insert(id,reply);return Ok(());},Err(error)=>Err(error)}
            },
            // 独立的"添加宿主轨道"入口。
            //
            // 【为什么不是 `add_track`】`add_track` 建的是**插件自己的**轨道，且被
            // `pluginAllowsAction` 的 `track.*` 规则一刀切掉；这条建的是**宿主**轨道。
            // 两者语义不同，不能共用一个名字。
            //
            // 【为什么插入位置要锚定】用户说的"最下方"是**本实例能看到的最下方**，
            // 不是整个工程的最底下 —— 后者会把新轨道丢到别的 FX 实例的轨道下面。
            Some("create_host_track")=>{
                let (link,sink,full)={let mut state=state.borrow_mut();(state.link.clone(),state.sink.clone(),state.pending_full(id))};
                let outcome=(||->Result<HostReply,String>{
                    if full {return Err("native track request budget exceeded".into());}
                    let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                    let editor=owner.editor_session()?;
                    let allowed=||link.authorize(&document).is_ok_and(|current|current==lease);
                    let host=owner.project_history_host().ok_or("host project history missing")?;
                    if !host.can_create_audio_track() {return Err("host track creation is unavailable".into());}
                    let anchor=host.folder_anchor_order(&allowed);
                    state.borrow_mut().undo_document=Some(Arc::downgrade(&document));
                    document.host_undo.begin_request(&view_id,id,true,true,&host,&allowed)?;
                    let result=(||->Result<(),String>{
                        let mut created=host.create_audio_track("HiFiShifter",anchor,&allowed)?;
                        created.commit();
                        Ok(())
                    })();
                    document.host_undo.finish_request(&view_id,id);result?;
                    owner.refresh_host_transport();editor.notify_timeline();
                    editor.enqueue(super::session::UiRequest {id,command:"get_timeline_state".into(),args:serde_json::json!({}),sink,link:Some(link.clone())})?;
                    Ok(HostReply {editor,document:Arc::downgrade(&document),lease,until:std::time::Instant::now()+std::time::Duration::from_secs(30),attempts:0,next_attempt:std::time::Instant::now(),jump:None,imported:None,geometry:None,split:None,media:None,action:None})
                })();
                match outcome {Ok(reply)=>{let mut state=state.borrow_mut();state.mark_pending(id);state.host_replies.insert(id,reply);return Ok(());},Err(error)=>Err(error)}
            },
            // 精简诊断导出（Help 菜单）。命令名与独立 App **相同** —— 前端不需要按
            // 模式分支，也不需要一个"插件能不能导出"的能力标志。
            //
            // 【为什么是精简版】App 的完整包会跑推理设备基准测试，而那条路径在
            // GPU/驱动异常时可能硬崩；在 DAW 进程里崩会带走用户整个会话。基准测试
            // 在插件里**保持拒绝**（见 `commands.rs`），这里只做不会崩的部分。
            Some("pick_diagnostics_output_path")=>{
                let (link,hwnd)={let state=state.borrow();(state.link.clone(),state.hwnd)};
                (||->Result<serde_json::Value,String>{
                    let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                    let name=super::plugin_diagnostics::default_file_name();
                    let path=super::browser_files::pick_save_path(hwnd,&name,"zip")?;
                    // 对话框期间用户可能关窗/切实例；租约变了就不再交付这个路径。
                    if link.authorize(&document)?!=lease {return Err("diagnostics picker editor lease changed".into());}
                    match path {Some(path)=>Ok(serde_json::json!({"ok":true,"path":path.to_string_lossy()})),
                        None=>Ok(serde_json::json!({"ok":false,"canceled":true}))}
                })()
            },
            // 「导出 MIDI」的另存为对话框。命令名与独立 App **相同**，前端不必按模式
            // 分支；真正的导出（读曲线 → 写文件）走 actor 线程的 `export_pitch_to_midi`。
            //
            // 【为什么对话框必须在 UI 线程】`IFileSaveDialog` 需要一个父 HWND，而
            // HWND 只存在于 UI 线程（与诊断导出、记事本另存为同一条理由）。
            Some("pick_midi_output_path")=>{
                let (link,hwnd)={let state=state.borrow();(state.link.clone(),state.hwnd)};
                (||->Result<serde_json::Value,String>{
                    let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                    let path=super::browser_files::pick_save_path(hwnd,"export.mid","mid")?;
                    // 对话框期间用户可能关窗/切实例；租约变了就不再交付这个路径。
                    if link.authorize(&document)?!=lease {return Err("MIDI picker editor lease changed".into());}
                    match path {
                        Some(path)=>Ok(serde_json::json!({"ok":true,"canceled":false,"path":path.to_string_lossy()})),
                        None=>Ok(serde_json::json!({"ok":true,"canceled":true})),
                    }
                })()
            },
            // 「导入 MIDI」的打开对话框。命令名与独立 App **相同**，前端不必按模式
            // 分支；真正的解析/写入走 actor 线程的 `get_midi_tracks` /
            // `import_midi_to_pitch`。
            //
            // 【为什么对话框必须在 UI 线程】`IFileOpenDialog` 需要一个父 HWND，而
            // HWND 只存在于 UI 线程（与文件夹选择器、另存为对话框同一条理由）。
            Some("open_midi_dialog")=>{
                let (link,hwnd)={let state=state.borrow();(state.link.clone(),state.hwnd)};
                (||->Result<serde_json::Value,String>{
                    let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                    let path=super::browser_files::pick_open_path(hwnd,"MIDI","mid")?;
                    // 对话框期间用户可能关窗/切实例；租约变了就不再交付这个路径。
                    if link.authorize(&document)?!=lease {return Err("MIDI picker editor lease changed".into());}
                    match path {
                        Some(path)=>Ok(serde_json::json!({"ok":true,"canceled":false,"path":path.to_string_lossy()})),
                        None=>Ok(serde_json::json!({"ok":true,"canceled":true})),
                    }
                })()
            },
            Some("export_diagnostics")=>{
                let link=state.borrow().link.clone();
                (||->Result<serde_json::Value,String>{
                    let output=request["args"]["outputPath"].as_str().ok_or("diagnostics output path missing")?;
                    let out=std::path::PathBuf::from(output);
                    if !out.is_absolute() {return Err("diagnostics output path must be absolute".into());}
                    let owner=link.owner()?;let document=owner.editor_document()?;
                    let backend=owner.host_backend();
                    let system=serde_json::json!({
                        "product":"HiFiShifter ARA plugin",
                        "about":hifishifter_kernel::build_info::about_payload("ARA plugin",crate::VERSION),
                        "os":std::env::consts::OS,"arch":std::env::consts::ARCH,
                        "generatedAt":chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
                        // 用户报障时"你是哪个宿主"是第一个要问的问题。宿主名来自
                        // `IHostApplication::getName`（只诊断用，绝不参与分支）。
                        "host":{"appName":backend.app_name().map(str::to_owned)
                                .or_else(super::connection::host_application_name),
                            "appVersion":backend.app_version()},
                        // 当前编辑后端与能力位：报障时能直接读出"是不是 ARA-only 只读后端"。
                        "backend":{"kind":backend.kind().as_str(),
                            "nativeAdapter":backend.native().is_some(),
                            "capabilities":backend.capabilities()},
                        "transport":document.clock.diagnostics(),
                        // 音频线程被 catch_unwind 拦下的 panic 次数。音频线程不能写日志，
                        // 只能计数；这里（UI 线程）把它落进诊断包，否则"崩过但没痕迹"。
                        "audioPanicCount":crate::vst3::audio_panic_count(),
                        "renderers":document.renderer_diagnostics(),
                    });
                    let settings=serde_json::json!({
                        "plugin":crate::settings_store::settings(),
                        "frontendPrefs":crate::settings_store::frontend_prefs(),
                        "frontend":request["args"]["frontendSettings"].clone(),
                    });
                    super::plugin_diagnostics::write_package(&out,&system,&settings)?;
                    crate::log_line(&format!("[diagnostics] plugin package exported: {}",out.display()));
                    Ok(serde_json::json!({"ok":true,"path":output}))
                })()
            },
            // 记事本导出 / 附件另存为：需要系统保存对话框（HWND 只有 UI 线程有），
            // 因此不走 actor dispatch。
            Some(command @ ("notebook_export_document"|"notebook_save_asset_as"))=>{
                let (link,hwnd)={let state=state.borrow();(state.link.clone(),state.hwnd)};
                (||->Result<serde_json::Value,String>{
                    // 对话框是模态的：期间用户可能关窗或切实例，所以前后各核对一次租约。
                    let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                    let result=super::notebook::export_with_dialog(hwnd,command,&request["args"])?;
                    if link.authorize(&document)?!=lease {return Err("notebook dialog editor lease changed".into());}
                    Ok(result)
                })()
            },
            Some(command @ ("play_original"|"play_synthesized"|"stop_audio"))=>{
                // host callback可能同步重入；调用期间不持BrowserState/会话锁。
                let link=state.borrow().link.clone();
                (||->Result<serde_json::Value,String> {
                    let owner=link.owner()?;
                    let playback=owner.host_playback().ok_or("host does not provide ARA playback control")?;
                    let position=owner.clock.get().map(|clock|clock.read().0).unwrap_or(0.);
                    if command=="stop_audio" {playback.stop().map_err(|e|e.to_string())?;
                        // ARA只有Stop请求；暂停后定位到实际停止点，Stop按钮的回锚由原thunk另发seek。
                        playback.set_position(position).map_err(|e|e.to_string())?;
                        Ok(serde_json::json!({"ok":true,"stopped_at_sec":position}))
                    } else {playback.start().map_err(|e|e.to_string())?;
                        Ok(serde_json::json!({"ok":true,"host_request":true,"anchorSec":position,"start_sec":position}))}
                })()
            },
            Some(command @ ("split_clip"|"split_clips_at"))=>{
                let (link,sink,full)={let mut state=state.borrow_mut();(state.link.clone(),state.sink.clone(),state.pending_full(id))};
                let outcome=(||->Result<HostReply,String> {
                    if full {return Err("native split request budget exceeded".into());}
                    let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                    let editor=owner.editor_session()?;let input=request.get("args").cloned().unwrap_or_else(||serde_json::json!({}));
                    let plan=editor.plan_host_split(command,&input)?;
                    let host=owner.project_history_host().ok_or("host project history missing")?;
                    if !host.can_split_clips() {return Err("host split capability unavailable".into());}
                    let allowed=||link.authorize(&document).is_ok_and(|current|current==lease);
                    state.borrow_mut().undo_document=Some(Arc::downgrade(&document));
                    document.host_undo.begin_request(&view_id,id,true,true,&host,&allowed)?;
                    let outcome=crate::host::host_split::execute(&owner,plan,&allowed);
                    document.host_undo.finish_request(&view_id,id);let split=Some(outcome?);
                    editor.enqueue(super::session::UiRequest {id,command:"get_timeline_state".into(),args:serde_json::json!({}),sink,link:Some(link.clone())})?;
                    Ok(HostReply {editor,document:Arc::downgrade(&document),lease,until:std::time::Instant::now()+std::time::Duration::from_secs(30),attempts:0,next_attempt:std::time::Instant::now(),jump:None,imported:None,geometry:None,split,media:None,action:None})
                })();
                match outcome {Ok(reply)=>{let mut state=state.borrow_mut();state.mark_pending(id);state.host_replies.insert(id,reply);return Ok(());},Err(error)=>Err(error)}
            },
            Some(command) if crate::host::host_edit::is_clip_edit(command)=>{
                let (link,sink,full)={let mut state=state.borrow_mut();(state.link.clone(),state.sink.clone(),state.pending_full(id))};
                let outcome=(||->Result<HostReply,String> {
                    if full {return Err("native host edit request budget exceeded".into());}
                    let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                    let editor=owner.editor_session()?;let input=request.get("args").cloned().unwrap_or_else(||serde_json::json!({}));
                    let plan=editor.plan_host_edit(command,&input)?;
                    // 完成门需要知道宿主的淡化轴版本：形状预设在新轴/旧轴上落盘的字段不同，
                    // 期望值也就不同（见 `HostEditPlan::receipt`）。
                    let fade_axes_v2=owner.host_fade_axes();
                    let geometry=Some(plan.receipt(&editor.namespace,fade_axes_v2));
                    state.borrow_mut().undo_document=Some(Arc::downgrade(&document));
                    let host=owner.project_history_host().ok_or("host project history missing")?;
                    let allowed=||link.authorize(&document).is_ok_and(|current|current==lease);
                    // checkpoint 由调用方声明（流式提交的末笔才收尾批次），此前这里写死
                    // `true`，任何按帧分块提交的调用方会把宿主撤销栈打爆（对照通用命令
                    // 分支对 `args.checkpoint` 的读法）。缺省仍是 true：整笔提交是常态。
                    let checkpoint=input["checkpoint"].as_bool().unwrap_or(true);
                    document.host_undo.begin_request(&view_id,id,checkpoint,true,&host,&allowed)?;
                    let outcome=crate::host::host_edit::execute_managed(&owner,plan,allowed,true);
                    document.host_undo.finish_request(&view_id,id);outcome?;
                    editor.enqueue(super::session::UiRequest {id,command:"get_timeline_state".into(),args:serde_json::json!({}),sink,link:Some(link.clone())})?;
                    Ok(HostReply {editor,document:Arc::downgrade(&document),lease,until:std::time::Instant::now()+std::time::Duration::from_secs(30),attempts:0,next_attempt:std::time::Instant::now(),jump:None,imported:None,geometry,split:None,media:None,action:None})
                })();
                match outcome {
                    Ok(reply)=>{let mut state=state.borrow_mut();state.mark_pending(id);state.host_replies.insert(id,reply);return Ok(());},
                    Err(error)=>Err(error),
                }
            },
            Some("has_timeline_clipboard")=>crate::host::clipboard::available(),
            Some(command @ ("copy_timeline_clips"|"paste_timeline_clipboard"|"duplicate_clips_bulk"|"remove_clip"|"remove_clips"))=>{
                let (link,sink,full)={let mut state=state.borrow_mut();(state.link.clone(),state.sink.clone(),state.pending_full(id))};
                let outcome=(||->Result<HostReply,String>{
                    if full {return Err("native media request budget exceeded".into());}
                    let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                    if !owner.project_history_host().is_some_and(|host|host.can_clipboard_items()) {return Err("host media clipboard API unavailable".into());}
                    let editor=owner.editor_session()?;
                    document.host_undo.begin_history(&view_id,id)?;
                    if let Err(error)=editor.enqueue(super::session::UiRequest {id,command:"plugin_editor_barrier".into(),args:serde_json::json!({}),sink,link:Some(link.clone())}) {
                        document.host_undo.finish_history(&view_id,id);return Err(error);
                    }
                    Ok(HostReply {editor,document:Arc::downgrade(&document),lease,until:std::time::Instant::now()+std::time::Duration::from_secs(30),attempts:0,next_attempt:std::time::Instant::now(),jump:None,
                        imported:None,geometry:None,split:None,media:None,action:Some(MediaAction {command:command.into(),input:request.get("args").cloned().unwrap_or_else(||serde_json::json!({}))})})
                })();
                match outcome {Ok(reply)=>{let mut state=state.borrow_mut();state.undo_document=Some(reply.document.clone());state.mark_pending(id);state.history_requests.insert(id);state.host_replies.insert(id,reply);return Ok(());},Err(error)=>Err(error)}
            },
            // 【无条件处理，不再用 `is_some()` 做 guard】此前这两条 arm 只在宿主提供撤销栈
            // 时才匹配，非 REAPER 宿主会落到下面的 `Some(command)` 分支并被静默丢弃 ——
            // 用户按 Ctrl+Z 什么都不发生。现在"没有宿主撤销栈"是 `LocalHistory` 这个
            // **实现**，`history_authority()` 永远返回一个权威（见 `host::history`）。
            Some("get_history_state")=>{
                let link=state.borrow().link.clone();( ||->Result<serde_json::Value,String>{
                    let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                    owner.history_authority().snapshot(&||link.authorize(&document).is_ok_and(|current|current==lease))
                })()
            },
            Some(command @ ("undo_timeline"|"redo_timeline"|"set_history_position"))=>{
                let (link,sink)={let state=state.borrow();(state.link.clone(),state.sink.clone())};
                // 宿主权威栈 ⇒ 走异步路径（必须等在途写入落定再跳，否则会跳到一半的状态）。
                //
                // 【为什么比 `"host"` 而不是 `"reaper"`】`ProjectHistory::backend()` 返回
                // `"host"`（`host::history`，随后端去品牌化改名）。此前这里比的是旧值
                // `"reaper"`，于是 `uses_host` 恒为 `false` —— REAPER 里宿主撤销/重做
                // 实际走了本地栈，`history_jump*` 整条路径不可达。
                let uses_host = link
                    .owner()
                    .is_ok_and(|owner| owner.history_authority().backend() == "host");
                if !uses_host {
                    // 本地权威栈 ⇒ 与普通命令同一条队列：本地命令天然排在已入队写入之后，
                    // 不需要额外的 barrier，也不会被 `host_undo.pending` 的门控挡住。
                    let full=state.borrow_mut().pending_full(id);
                    let outcome=if full {Err("pending request budget exceeded".into())} else {
                        link.owner().and_then(|owner|owner.editor_session()?.enqueue(super::session::UiRequest {
                            id,command:command.into(),args:request.get("args").cloned().unwrap_or_else(||serde_json::json!({})),sink,link:Some(link.clone()),
                        }))
                    };
                    match outcome {Ok(())=>{state.borrow_mut().mark_pending(id);Ok(serde_json::Value::Null)},Err(error)=>Err(error)}
                } else {
                let outcome=(||->Result<HostReply,String>{
                    if state.borrow_mut().pending_full(id) {return Err("native history request budget exceeded".into());}
                    let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;let editor=owner.editor_session()?;
                    let jump=match command {"undo_timeline"=>HistoryJump::Undo,"redo_timeline"=>HistoryJump::Redo,_=>HistoryJump::Position(request["args"]["position"].as_i64().filter(|v|(0..10000).contains(v)).ok_or("invalid host history position")? as i32)};
                    state.borrow_mut().undo_document=Some(Arc::downgrade(&document));
                    document.host_undo.begin_history(&view_id,id)?;
                    if let Err(error)=editor.enqueue(super::session::UiRequest {id,command:"plugin_history_barrier".into(),args:serde_json::json!({}),sink,link:Some(link.clone())}) {
                        document.host_undo.finish_history(&view_id,id);return Err(error);
                    }
                    Ok(HostReply {editor,document:Arc::downgrade(&document),lease,until:std::time::Instant::now()+std::time::Duration::from_secs(30),attempts:0,next_attempt:std::time::Instant::now(),jump:Some(jump),imported:None,geometry:None,split:None,media:None,action:None})
                })();
                match outcome {Ok(reply)=>{let mut state=state.borrow_mut();state.mark_pending(id);state.history_requests.insert(id);state.host_replies.insert(id,reply);return Ok(());},Err(error)=>Err(error)}
                }
            },
            Some(command)=>{
                if command=="set_transport" && request["args"]["playheadSec"].is_number() {
                    let link=state.borrow().link.clone();
                    let result=(||->Result<(),String> {
                        let position=request["args"]["playheadSec"].as_f64().ok_or("invalid seek position")?;
                        let owner=link.owner()?;
                        owner.host_playback().ok_or("host does not provide ARA playback control")?
                            .set_position(position).map_err(|e|e.to_string())?;
                        // seek回执前同步真实宿主光标，避免继续发布上次UI tick的旧位置。
                        owner.refresh_host_transport();Ok(())
                    })();
                    if let Err(error)=result {
                        let text=wide(&serde_json::json!({"version":1,"viewId":view_id,"id":id,"ok":false,"error":error}).to_string());
                        browser.PostWebMessageAsJson(PCWSTR(text.as_ptr()))?;return Ok(());
                    }
                }
                let (link,sink,full)={let mut state=state.borrow_mut();(state.link.clone(),state.sink.clone(),state.pending_full(id))};
                let history_context=link.owner().ok().and_then(|owner|{
                    let document=owner.editor_document().ok()?;let host=owner.project_history_host()?;
                    let lease=link.authorize(&document).ok()?;Some((document,host,lease))
                });
                let mut undo_started=false;
                if let Some((document,host,lease))=&history_context {
                    state.borrow_mut().undo_document=Some(Arc::downgrade(document));
                    if command=="begin_undo_group" {document.host_undo.begin_group(&view_id);}
                    if command=="end_undo_group" {document.host_undo.end_group(&view_id);}
                    if super::commands::mutates_audio(command) {
                        let checkpoint=request["args"]["checkpoint"].as_bool().unwrap_or(true);
                        match document.host_undo.begin_request(&view_id,id,checkpoint,false,host,&||link.authorize(document).is_ok_and(|current|current==*lease)) {
                            Ok(())=>undo_started=true,
                            Err(error)=>{let text=wide(&serde_json::json!({"version":1,"viewId":view_id,"id":id,"ok":false,"error":error}).to_string());browser.PostWebMessageAsJson(PCWSTR(text.as_ptr()))?;return Ok(());},
                        }
                    }
                }
                let outcome=if full {Err("native pending request budget exceeded".into())} else {
                    link.owner().and_then(|owner|owner.editor_session()?.enqueue(super::session::UiRequest {
                        id,command:command.into(),args:request.get("args").cloned().unwrap_or_else(||serde_json::json!({})),sink,link:Some(link.clone()),
                    }))
                };
                match outcome {
                    Ok(())=>{let mut state=state.borrow_mut();state.mark_pending(id);if undo_started {if let Some((document,_,_))=&history_context {state.undo_requests.insert(id,Arc::downgrade(document));}}return Ok(());},
                    Err(error)=>{
                        if let Some((document,_,_))=&history_context {
                            if undo_started {document.host_undo.finish_request(&view_id,id);}
                            if command=="begin_undo_group" {document.host_undo.end_group(&view_id);}
                        }Err(error)
                    },
                }
            },
            None=>Err(String::from("native command missing")),
        };
        let response = match result {
            Ok(value) => serde_json::json!({"version":1,"viewId":view_id,"id":id,"ok":true,"value":value}),
            // `errorKind` 让前端能区分"暂时失败"与"会话已终结"。后者必须让前端
            // **停止轮询**并如实显示，而不是每 250 ms 重问一个永远不会好的问题
            // （见 `editor::classify_error` 的说明）。
            Err(error) => serde_json::json!({"version":1,"viewId":view_id,"id":id,"ok":false,
                "errorKind":super::classify_error(&error),
                "errorCode":super::errors::terminal_error_code(&error),"error":error}),
        };
        let response = wide(&response.to_string());
        browser.PostWebMessageAsJson(PCWSTR(response.as_ptr()))?;
        Ok(())
    })),&mut token)?;
    }
    let transport_control = state
        .borrow()
        .link
        .owner()
        .is_ok_and(|owner| owner.host_playback().is_some());
    // 宿主用哪一套淡化轴。版本读不出来时是 `None`，前端据此整块保持只读。
    let fade_axes = state
        .borrow()
        .link
        .owner()
        .ok()
        .and_then(|owner| owner.host_fade_axes())
        .map(|axes_new| if axes_new { "continuous" } else { "legacy" });
    let boot = BootstrapFlags {
        transport_control,
        fade_axes,
        history_backend: state
            .borrow()
            .link
            .owner()
            .map(|owner| owner.history_authority().backend())
            .unwrap_or("local"),
        host_kind: state
            .borrow()
            .link
            .owner()
            .map(|owner| owner.host_backend().kind().as_str())
            .unwrap_or("ara_only"),
        // 写能力/分割/剪贴板/导入/建轨/撤销栈的唯一来源（见 `effective_host_capabilities`）。
        host_capabilities: state
            .borrow()
            .link
            .owner()
            .map(|owner| owner.effective_host_capabilities())
            .unwrap_or_else(|_| hifishifter_kernel::editor::host::HostCapabilities::none()),
    }
    .to_json(&state.borrow().view_id);
    let script = wide(&format!("window.__HFS_PLUGIN_BOOTSTRAP__={boot};"));
    let weak = Rc::downgrade(state);
    let navigate = browser.clone();
    let ready =
        AddScriptToExecuteOnDocumentCreatedCompletedHandler::create(Box::new(move |result, _| {
            let Some(state) = weak.upgrade() else {
                return Ok(());
            };
            if state.borrow().closed {
                return Ok(());
            }
            if let Err(error) = result {
                show_error(&state, &error.to_string());
                return Ok(());
            }
            unsafe {
                navigate.Navigate(w!("https://hifishifter.invalid/plugin.html"))?;
            }
            crate::log_line(
                "Native editor frontend navigation started (commands pending session binding)",
            );
            Ok(())
        }));
    unsafe { browser.AddScriptToExecuteOnDocumentCreated(PCWSTR(script.as_ptr()), &ready) }
}

#[cfg(test)]
mod focus_tests {
    use super::*;
    use windows::Win32::UI::Input::KeyboardAndMouse::{GetActiveWindow, SetActiveWindow};

    /// **不变量：重投必须退避且封顶。**
    ///
    /// 【为什么这是回归测试】回复泵由 20 ms 定时器驱动，此前只要等待判据仍成立就
    /// **每一 tick 重投一次**（50 Hz，直到 30 s 的 `until`）。一个永不收敛的等待会变成
    /// ~1500 次 `get_timeline_state`，每次都与渲染线程争同一把文档事务。退避必须
    /// 单调不减、首档与定时器同频（不牺牲正常情况的响应），并封顶。
    #[test]
    fn host_reply_retry_backoff_grows_and_is_capped() {
        let first = host_reply_retry_backoff(1);
        assert_eq!(
            first,
            std::time::Duration::from_millis(20),
            "第一档保持与 20 ms 定时器同频，正常情况不受影响"
        );
        let mut previous = std::time::Duration::ZERO;
        for attempt in 1..=HOST_REPLY_RETRY_BUDGET {
            let delay = host_reply_retry_backoff(attempt);
            assert!(delay >= previous, "退避不得回退（attempt={attempt}）");
            assert!(
                delay <= std::time::Duration::from_millis(500),
                "退避必须封顶，否则收敛会慢到不可接受（attempt={attempt}）"
            );
            previous = delay;
        }
    }

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
        let native =
            NativeEditor::attach(parent.0, 1100, 720, Arc::new(Default::default())).unwrap();
        let child = HWND(native.key.hwnd as *mut c_void);
        // `SetFocus` 的前置条件是目标窗口所属线程有一个**活动窗口**：刚创建的
        // 夹具窗口尚未激活，此时 `SetFocus` 返回 `ERROR_INVALID_PARAMETER`
        // (0x80070057) —— 于是本用例在"进程没有前台窗口"的会话里（无头/服务/
        // 沙箱）会假失败，而真实宿主里插件窗口始终位于活动窗口内。这里显式补上
        // 同一前置条件，让用例自足、与运行会话无关。
        // `SetActiveWindow` 返回先前的活动窗口（没有则为 null → Err），仅用于还原。
        let previous_active = unsafe { SetActiveWindow(parent) }.ok();
        assert!(
            unsafe { GetActiveWindow() } == parent,
            "夹具父窗口必须能成为活动窗口；不能则说明本会话没有可交互的窗口站"
        );
        // GetNextDlgTabItem在没有任何Tab stop时会回退到第一个child，单看句柄会假通过。
        let tab_stop = unsafe { GetWindowLongPtrW(child, GWL_STYLE) } as u32 & WS_TABSTOP.0 != 0;
        let key = native.key;
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
        drop(native);
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
        assert!(
            focus_result.is_ok(),
            "首次无旧焦点时仍应成功: {focus_result:?}"
        );
        assert_eq!(focused, child, "宿主焦点请求必须进入自有窗口");
        assert_eq!(target, Some(child), "宿主Tab顺序必须能到达真实编辑器子窗口");
    }
}

#[cfg(test)]
mod bootstrap_tests {
    use super::BootstrapFlags;

    /// `trackGrouping` 不随任何宿主能力变化 —— 它守卫的是纯插件的参数分组。
    ///
    /// 【回归】它此前被写成复用 `audioImport`，于是宿主缺少媒体 API 时，插件里
    /// 本不需要任何宿主写能力的轨道拖动会无声失效。
    #[test]
    fn track_grouping_does_not_depend_on_host_media_apis() {
        for media_import in [true, false] {
            let flags = BootstrapFlags {
                transport_control: false,
                fade_axes: None,
                history_backend: "local",
                host_kind: "ara_only",
                host_capabilities: hifishifter_kernel::editor::host::HostCapabilities {
                    media_import,
                    ..hifishifter_kernel::editor::host::HostCapabilities::ara_only()
                },
            };
            assert_eq!(
                flags.to_json("view-7")["trackGrouping"],
                serde_json::json!(true),
                "mediaImport={media_import} 时 trackGrouping 被错误地关掉了"
            );
        }
    }

    /// 宿主能力标志必须**原样**透传：前端按它们决定开放哪些操作。
    #[test]
    fn host_capabilities_pass_through_unchanged() {
        let flags = BootstrapFlags {
            transport_control: true,
            fade_axes: Some("legacy"),
            history_backend: "local",
            host_kind: "ara_only",
            host_capabilities: hifishifter_kernel::editor::host::HostCapabilities {
                split: true,
                media_import: true,
                track_create: true,
                ..hifishifter_kernel::editor::host::HostCapabilities::ara_only()
            },
        };
        let json = flags.to_json("view-42");
        assert_eq!(json["version"], serde_json::json!(1));
        assert_eq!(json["viewId"], serde_json::json!("view-42"));
        assert_eq!(json["transportControl"], serde_json::json!(true));
        assert_eq!(json["fadeAxes"], serde_json::json!("legacy"));
        // 后端身份与能力位必须原样透传：前端据此显示只读横幅。
        assert_eq!(json["hostKind"], serde_json::json!("ara_only"));
        assert_eq!(
            json["hostCapabilities"]["clipWrite"],
            serde_json::json!(false)
        );
        assert_eq!(
            json["hostCapabilities"]["clipGeometry"],
            serde_json::json!(true)
        );
        assert_eq!(json["hostCapabilities"]["split"], serde_json::json!(true));
        assert_eq!(
            json["hostCapabilities"]["mediaImport"],
            serde_json::json!(true)
        );
        assert_eq!(
            json["hostCapabilities"]["trackCreate"],
            serde_json::json!(true)
        );
    }

    /// 写能力只有**一个**来源：顶层不得再出现 `clipEditing` / `audioImport` 这类副本。
    ///
    /// 【回归】两处并存时（一处按本实例原生扩展、一处按 `project_history_host()`）
    /// 必然漂移，而"哪一份算数"没人说得清。
    #[test]
    fn host_write_capabilities_have_a_single_source() {
        let flags = BootstrapFlags {
            transport_control: false,
            fade_axes: None,
            history_backend: "local",
            host_kind: "ara_only",
            host_capabilities: hifishifter_kernel::editor::host::HostCapabilities::ara_only(),
        };
        let json = flags.to_json("v");
        for stale in [
            "clipEditing",
            "clipSplitting",
            "clipClipboard",
            "audioImport",
            "trackCreation",
        ] {
            assert!(
                json.get(stale).is_none(),
                "顶层不应再出现 `{stale}`：写能力只由 `hostCapabilities` 提供"
            );
        }
    }

    /// `trackCreate` 与 `mediaImport` 是独立能力：导入音频蕴含建轨，建轨不蕴含导入。
    ///
    /// 【为什么单独钉住】合成一个标志会让"添加轨道"随"能不能导入音频"一起开关 ——
    /// 而这两件事的门槛不同（建轨还需要 `InsertTrackInProject` / FX 接口）。
    #[test]
    fn track_creation_is_independent_of_audio_import() {
        let flags = BootstrapFlags {
            transport_control: false,
            fade_axes: None,
            history_backend: "local",
            host_kind: "ara_only",
            host_capabilities: hifishifter_kernel::editor::host::HostCapabilities {
                media_import: false,
                track_create: true,
                ..hifishifter_kernel::editor::host::HostCapabilities::ara_only()
            },
        };
        let json = flags.to_json("v");
        assert_eq!(
            json["hostCapabilities"]["mediaImport"],
            serde_json::json!(false)
        );
        assert_eq!(
            json["hostCapabilities"]["trackCreate"],
            serde_json::json!(true)
        );
    }

    /// 未知轴语义必须如实报 `null`，前端据此保持只读（不猜）。
    #[test]
    fn an_unknown_fade_axis_set_is_reported_as_null() {
        let flags = BootstrapFlags {
            transport_control: false,
            fade_axes: None,
            history_backend: "local",
            host_kind: "ara_only",
            host_capabilities: hifishifter_kernel::editor::host::HostCapabilities::ara_only(),
        };
        assert_eq!(flags.to_json("v")["fadeAxes"], serde_json::Value::Null);
    }

    /// 新轴宿主如实报 `continuous`：前端据此只给连续滑杆、不摆预设按钮。
    #[test]
    fn a_continuous_fade_axis_host_is_reported_as_such() {
        let flags = BootstrapFlags {
            transport_control: false,
            fade_axes: Some("continuous"),
            history_backend: "local",
            host_kind: "ara_only",
            host_capabilities: hifishifter_kernel::editor::host::HostCapabilities::ara_only(),
        };
        assert_eq!(
            flags.to_json("v")["fadeAxes"],
            serde_json::json!("continuous")
        );
    }
}

#[cfg(test)]
mod group_id_tests {
    use super::host_group_id_for;
    use crate::host::host_edit::HOST_GROUP_ID_MAX;

    fn raw_first_four(key: &str) -> u32 {
        let digest = blake3::hash(key.as_bytes());
        u32::from_le_bytes(digest.as_bytes()[..4].try_into().unwrap())
    }

    /// 编组 id 必须落在宿主接受区间内 —— 旧实现 `& 0x7fff_ffff` 会给出到 `2^31-1`
    /// 的值，约 7% 的编组被 `host_edit` 的 "invalid host item group id" 整条拒绝。
    #[test]
    fn host_group_ids_stay_inside_the_host_accepted_range() {
        let mut highest = 0_i32;
        for i in 0..20_000 {
            let id = host_group_id_for(&[format!("clip-{i}")]);
            assert!((1..=HOST_GROUP_ID_MAX).contains(&id), "越界 id: {id}");
            highest = highest.max(id);
        }
        // 分布应覆盖到区间上半，否则本测试没证明折进区间后仍用满了熵。
        assert!(
            highest > HOST_GROUP_ID_MAX / 2,
            "取值上界偏低，折进区间可能压掉了熵：{highest}"
        );
    }

    /// 直接钉住回归：样本里存在旧实现会拒绝的取值，新实现必须把它们收进区间。
    #[test]
    fn values_the_old_mask_would_have_rejected_are_now_in_range() {
        let mut rejected_by_old = 0;
        for i in 0..20_000 {
            let key = format!("clip-{i}");
            if raw_first_four(&key) & 0x7fff_ffff > HOST_GROUP_ID_MAX as u32 {
                rejected_by_old += 1;
                let id = host_group_id_for(&[key]);
                assert!((1..=HOST_GROUP_ID_MAX).contains(&id), "仍越界: {id}");
            }
        }
        assert!(
            rejected_by_old > 0,
            "样本里应存在旧实现会拒绝的取值，否则本测试没覆盖到回归"
        );
    }

    /// 同一选择必须恒等映射：重复编组不得不断改号。
    #[test]
    fn the_same_selection_always_maps_to_the_same_group() {
        let first = host_group_id_for(&["b".into(), "a".into()]);
        let second = host_group_id_for(&["b".into(), "a".into()]);
        assert_eq!(first, second);
    }
}
