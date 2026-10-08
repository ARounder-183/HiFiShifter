//! 原生WebView2子窗口：借用REAPER消息循环，取消异步创建，不启动Tauri/app。
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

const ORIGIN: &str = "https://hifishifter.invalid/";
static NEXT_VIEW: AtomicU64 = AtomicU64::new(1);
static WINDOWS: AtomicUsize = AtomicUsize::new(0);
static CLASS_LOCK: Mutex<bool> = Mutex::new(false);
static WEBVIEW_MODULE_PIN: OnceLock<Result<(), String>> = OnceLock::new();

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
    pending: HashSet<u64>,
    forwarded_keys: HashSet<char>,
    host_replies: std::collections::HashMap<u64, HostReply>,
    undo_requests:
        std::collections::HashMap<u64, std::sync::Weak<crate::render::document::DocumentSession>>,
    undo_document: Option<std::sync::Weak<crate::render::document::DocumentSession>>,
    history_requests: HashSet<u64>,
    history_cache: Option<((usize, i32, i32), serde_json::Value)>,
    last_history: Option<serde_json::Value>,
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
    jump: Option<HistoryJump>,
    imported: Option<String>,
    geometry: Option<super::host_edit::HostEditReceipt>,
    split: Option<super::host_split::SplitReceipt>,
    media: Option<super::host_clipboard::MediaReceipt>,
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
        // 尽力清理本次视图的用户数据目录：WebView2 在 Close 之后释放文件句柄，
        // 但"释放"没有同步保证 —— 删不掉就留着（下次同名目录会复用），绝不因为
        // 清理失败而影响关闭流程。放在本机数据目录而不是 `%TEMP%`，就是为了让
        // 这些目录有一个由我们负责的回收点（见 config_location 的说明）。
        let _ = std::fs::remove_dir_all(
            hifishifter_kernel::config_location::local_data_subdir("webview").join(&view_id),
        );
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
            pending: HashSet::new(),
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
                    owner.refresh_reaper_transport();
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
        let action={let state=state.borrow();state.host_replies.get(&id).and_then(|host|host.action.clone())
            .map(|action|(action,state.host_replies[&id].document.clone(),state.host_replies[&id].lease,state.host_replies[&id].editor.clone(),state.sink.clone(),state.view_id.clone()))};
        if let Some((action,weak,lease,editor,sink,view_id))=action.filter(|_|reply["ok"]==true) {
            let outcome=(||->Result<bool,String>{
                let document=weak.upgrade().ok_or("media document closed")?;
                if state.borrow().host_replies.get(&id).is_some_and(|host|std::time::Instant::now()>=host.until) {return Err("media request timed out".into());}
                let allowed=||link.authorize(&document).is_ok_and(|current|current==lease);
                if !allowed() {return Err("media editor lease changed".into());}
                state.borrow_mut().host_replies.get_mut(&id).ok_or("media request closed")?.action=None;
                let owner=link.owner()?;
                if action.command=="copy_timeline_clips" {reply["value"]=super::host_clipboard::copy(&owner,&editor,&action.input,&allowed)?;return Ok(true);}
                let host=owner.project_history_host().filter(|host|host.can_clipboard_items()).ok_or("host media clipboard API unavailable")?;
                let duplicated=action.command=="duplicate_clips_bulk";
                let pasted=action.command=="paste_timeline_clipboard"||duplicated;
                // 计划只读预检；native Undo开始后才创建/删除，失败也必须结束请求。
                let paste=if duplicated {Some(super::host_clipboard::plan_duplicate(&owner,&editor,&action.input,&allowed)?)} else if pasted {Some(super::host_clipboard::plan_paste(&owner,&editor,&action.input,&allowed)?)} else {None};
                let delete=if pasted {None} else {Some(super::host_clipboard::plan_delete(&owner,&editor,&action.input,action.command=="remove_clip",&allowed)?)};
                document.host_undo.begin_request(&view_id,id,true,true,&host,&allowed)?;
                state.borrow_mut().undo_requests.insert(id,Arc::downgrade(&document));
                let receipt=if let Some(plan)=paste {super::host_clipboard::execute_paste(&owner,plan,&allowed)?}
                    else {super::host_clipboard::execute_delete(&owner,delete.unwrap(),&allowed)?};
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
        let jump={let state=state.borrow();state.host_replies.get(&id).and_then(|host|host.jump)
            .map(|redo|(redo,state.host_replies[&id].document.clone(),state.host_replies[&id].lease,state.host_replies[&id].editor.clone(),state.sink.clone()))};
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
                state.borrow_mut().host_replies.get_mut(&id).unwrap().jump=None;
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
            if (waiting||geometry_waiting||split_waiting||media_waiting||reply["ok"]==false&&(error.starts_with("Conflict: host")||error.contains("host PCM unavailable")||error.contains("host model not ready")))&&std::time::Instant::now()<host.until {
                let allowed=host.document.upgrade().is_some_and(|doc|state.link.authorize(&doc).is_ok_and(|lease|lease==host.lease));
                let command=if host.media.is_some() {"get_timeline_inventory"} else {"get_timeline_state"};
                if allowed&&host.editor.enqueue(super::session::UiRequest {id,command:command.into(),args:serde_json::json!({}),sink:state.sink.clone(),link:Some(state.link.clone())}).is_ok() {return false;}
            }
            if waiting {reply["ok"]=serde_json::json!(false);reply["error"]=serde_json::json!("item was created in REAPER but its ARA audio is not ready; use host Undo if canceling");}
            if geometry_waiting {reply["ok"]=serde_json::json!(false);reply["error"]=serde_json::json!("clip was edited in REAPER but its ARA geometry has not caught up; host Undo remains available");}
            if split_waiting {reply["ok"]=serde_json::json!(false);reply["error"]=serde_json::json!("item was split in REAPER but both clips have not caught up; host Undo remains available");}
            if media_waiting {reply["ok"]=serde_json::json!(false);reply["error"]=serde_json::json!("media was edited in REAPER but its GUI/ARA identities have not caught up; host Undo remains available");}
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
        let key = (Arc::as_ptr(&host) as usize, token.0, token.1);
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
    let id = state.borrow().view_id.clone();
    // 【为什么不再用 `%TEMP%`】磁盘清理会在会话进行中删掉 `%TEMP%` 下的内容，
    // 正在运行的 WebView2 用户数据目录被删掉会直接崩 —— 而不只是"缓存丢了"。
    // 放在本机数据目录下，生命周期由我们自己掌握。
    //
    // 【为什么仍然每进程一个目录】WebView2 不支持两个进程同时打开同一个用户数据
    // 目录，而 REAPER 可能把插件放在独立的宿主进程里。耐久性早已不依赖它（设置
    // 在配置文件里），所以这里只要"稳定且可清理"就够。
    let profile = hifishifter_kernel::config_location::local_data_subdir("webview").join(&id);
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
#[derive(Clone, Copy, Debug)]
struct BootstrapFlags {
    /// 宿主提供 ARA 播放控制。
    transport_control: bool,
    /// 宿主允许写片段几何。
    clip_editing: bool,
    /// 宿主允许分割片段。
    clip_splitting: bool,
    /// 宿主提供完整 item 剪贴板。
    clip_clipboard: bool,
    /// 宿主允许导入音频（会建轨、建 item）。
    audio_import: bool,
    /// 宿主用哪一套淡化轴：`Some("legacy")`（≤7.80，`C_FADE*SHAPE` 决定形状）
    /// 或 `Some("continuous")`（≥7.81，curvature/S 两轴决定形状）。
    ///
    /// 【为什么是枚举而不是布尔】"能不能编辑"与"能不能选预设"是两件事：新轴宿主
    /// 上曲率可以改，但预设到 (curvature, S) 的映射尚未校准，所以只给连续滑杆。
    /// `None` = 版本读不出来 → 前端整块保持只读（不猜）。
    fade_axes: Option<&'static str>,
}

impl BootstrapFlags {
    fn to_json(self, view_id: &str) -> serde_json::Value {
        serde_json::json!({
            "version": 1,
            "viewId": view_id,
            "transportControl": self.transport_control,
            "clipEditing": self.clip_editing,
            "clipSplitting": self.clip_splitting,
            "clipClipboard": self.clip_clipboard,
            "audioImport": self.audio_import,
            "fadeAxes": self.fade_axes,
            // 【为什么是常量 true，而不是复用 audioImport】它守卫的是
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
                let mut bytes=[0_u8;4];bytes.copy_from_slice(&blake3::hash(ids.iter().filter_map(|id|id.as_str()).collect::<Vec<_>>().join("\n").as_bytes()).as_bytes()[..4]);
                (u32::from_le_bytes(bytes)&0x7fff_ffff).max(1) as i32
            };
            request["command"]=serde_json::json!("set_clips_state_bulk");request["args"]=serde_json::json!({"updates":ids.into_iter().map(|id|serde_json::json!({"clipId":id,"hostGroupId":group})).collect::<Vec<_>>(),"checkpoint":true});
        }
        if request["command"]=="close_track_gaps" {
            let track=request["args"]["trackId"].as_str().unwrap_or("").to_owned();let from=request["args"]["fromSec"].as_f64().unwrap_or(f64::NAN);
            let link=state.borrow().link.clone();
            if let Ok(owner)=link.owner() {if let Ok(editor)=owner.editor_session() {
                let moves=editor.timeline.lock().unwrap().close_track_gaps_moves(&track,from);
                request["command"]=if moves.is_empty(){serde_json::json!("get_timeline_state")}else{serde_json::json!("move_clips")};
                request["args"]=if moves.is_empty(){serde_json::json!({})}else{serde_json::json!({"moves":moves,"moveLinkedParams":true})};
            }}
        }
        if request["command"]=="rename_clip_take" {
            let clip_id=request["args"]["clipId"].as_str().unwrap_or("").to_owned();
            let take_id=request["args"]["takeId"].as_str().unwrap_or("").to_owned();
            let name=request["args"]["name"].as_str().unwrap_or("").to_owned();let link=state.borrow().link.clone();
            if let Ok(owner)=link.owner() {if let Ok(editor)=owner.editor_session() {
                let active=editor.timeline.lock().unwrap().clips.iter().find(|clip|clip.id==clip_id).and_then(|clip|clip.active_take_id.clone());
                if active.as_deref()==Some(take_id.as_str()) {request["command"]=serde_json::json!("set_clip_state");request["args"]=serde_json::json!({"clipId":clip_id,"name":name,"checkpoint":request["args"]["checkpoint"]});}
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
                crate::log_line(&format!("Embedded frontend: {}",request["args"])); Ok(serde_json::Value::Null)
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
                let (link,sink,full)={let state=state.borrow();(state.link.clone(),state.sink.clone(),state.pending.len()>=32||state.pending.contains(&id))};
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
                        let mut created=if new_track {Some(host.create_audio_track(std::path::Path::new(path).file_stem().and_then(|name|name.to_str()).unwrap_or("Imported audio"),&allowed)?)} else {None};
                        let target=existing.as_ref().or_else(||created.as_ref().map(|track|&track.target)).ok_or("import target missing")?;
                        let id=target.import_audio(path,input["startSec"].as_f64().unwrap_or_else(||owner.clock.get().map(|clock|clock.read().0).unwrap_or(0.)),&allowed)?;
                        if let Some(track)=&mut created {track.commit();}Ok(id)
                    })();
                    document.host_undo.finish_request(&view_id,id);let imported=result?;
                    owner.refresh_reaper_transport();editor.notify_timeline();
                    editor.enqueue(super::session::UiRequest {id,command:"get_timeline_state".into(),args:serde_json::json!({}),sink,link:Some(link.clone())})?;
                    Ok(HostReply {editor,document:Arc::downgrade(&document),lease,until:std::time::Instant::now()+std::time::Duration::from_secs(30),jump:None,imported:Some(imported),geometry:None,split:None,media:None,action:None})
                })();
                match outcome {Ok(reply)=>{let mut state=state.borrow_mut();state.pending.insert(id);state.host_replies.insert(id,reply);return Ok(());},Err(error)=>Err(error)}
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
                let (link,sink,full)={let state=state.borrow();(state.link.clone(),state.sink.clone(),state.pending.len()>=32||state.pending.contains(&id))};
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
                    let outcome=super::host_split::execute(&owner,plan,&allowed);
                    document.host_undo.finish_request(&view_id,id);let split=Some(outcome?);
                    editor.enqueue(super::session::UiRequest {id,command:"get_timeline_state".into(),args:serde_json::json!({}),sink,link:Some(link.clone())})?;
                    Ok(HostReply {editor,document:Arc::downgrade(&document),lease,until:std::time::Instant::now()+std::time::Duration::from_secs(30),jump:None,imported:None,geometry:None,split,media:None,action:None})
                })();
                match outcome {Ok(reply)=>{let mut state=state.borrow_mut();state.pending.insert(id);state.host_replies.insert(id,reply);return Ok(());},Err(error)=>Err(error)}
            },
            Some(command) if super::host_edit::is_clip_edit(command)=>{
                let (link,sink,full)={let state=state.borrow();(state.link.clone(),state.sink.clone(),state.pending.len()>=32||state.pending.contains(&id))};
                let outcome=(||->Result<HostReply,String> {
                    if full {return Err("native host edit request budget exceeded".into());}
                    let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                    let editor=owner.editor_session()?;let input=request.get("args").cloned().unwrap_or_else(||serde_json::json!({}));
                    let plan=editor.plan_host_edit(command,&input)?;
                    let geometry=Some(plan.receipt(&editor.namespace));
                    state.borrow_mut().undo_document=Some(Arc::downgrade(&document));
                    let host=owner.project_history_host().ok_or("host project history missing")?;
                    let allowed=||link.authorize(&document).is_ok_and(|current|current==lease);
                    document.host_undo.begin_request(&view_id,id,true,true,&host,&allowed)?;
                    let outcome=super::host_edit::execute_managed(&owner,plan,allowed,true);
                    document.host_undo.finish_request(&view_id,id);outcome?;
                    editor.enqueue(super::session::UiRequest {id,command:"get_timeline_state".into(),args:serde_json::json!({}),sink,link:Some(link.clone())})?;
                    Ok(HostReply {editor,document:Arc::downgrade(&document),lease,until:std::time::Instant::now()+std::time::Duration::from_secs(30),jump:None,imported:None,geometry,split:None,media:None,action:None})
                })();
                match outcome {
                    Ok(reply)=>{let mut state=state.borrow_mut();state.pending.insert(id);state.host_replies.insert(id,reply);return Ok(());},
                    Err(error)=>Err(error),
                }
            },
            Some("has_timeline_clipboard")=>super::host_clipboard::available(),
            Some(command @ ("copy_timeline_clips"|"paste_timeline_clipboard"|"duplicate_clips_bulk"|"remove_clip"|"remove_clips"))=>{
                let (link,sink,full)={let state=state.borrow();(state.link.clone(),state.sink.clone(),state.pending.len()>=32||state.pending.contains(&id))};
                let outcome=(||->Result<HostReply,String>{
                    if full {return Err("native media request budget exceeded".into());}
                    let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                    if !owner.project_history_host().is_some_and(|host|host.can_clipboard_items()) {return Err("host media clipboard API unavailable".into());}
                    let editor=owner.editor_session()?;
                    document.host_undo.begin_history(&view_id,id)?;
                    if let Err(error)=editor.enqueue(super::session::UiRequest {id,command:"plugin_editor_barrier".into(),args:serde_json::json!({}),sink,link:Some(link.clone())}) {
                        document.host_undo.finish_history(&view_id,id);return Err(error);
                    }
                    Ok(HostReply {editor,document:Arc::downgrade(&document),lease,until:std::time::Instant::now()+std::time::Duration::from_secs(30),jump:None,
                        imported:None,geometry:None,split:None,media:None,action:Some(MediaAction {command:command.into(),input:request.get("args").cloned().unwrap_or_else(||serde_json::json!({}))})})
                })();
                match outcome {Ok(reply)=>{let mut state=state.borrow_mut();state.undo_document=Some(reply.document.clone());state.pending.insert(id);state.history_requests.insert(id);state.host_replies.insert(id,reply);return Ok(());},Err(error)=>Err(error)}
            },
            Some("get_history_state") if state.borrow().link.owner().is_ok_and(|owner|owner.project_history_host().is_some())=>{
                let link=state.borrow().link.clone();( ||->Result<serde_json::Value,String>{
                    let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                    owner.project_history_host().ok_or("host history missing")?.project_history(&||link.authorize(&document).is_ok_and(|current|current==lease))
                })()
            },
            Some(command @ ("undo_timeline"|"redo_timeline"|"set_history_position")) if state.borrow().link.owner().is_ok_and(|owner|owner.project_history_host().is_some())=>{
                let (link,sink)={let state=state.borrow();(state.link.clone(),state.sink.clone())};
                let outcome=(||->Result<HostReply,String>{
                    if state.borrow().pending.len()>=32||state.borrow().pending.contains(&id) {return Err("native history request budget exceeded".into());}
                    let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;let editor=owner.editor_session()?;
                    let jump=match command {"undo_timeline"=>HistoryJump::Undo,"redo_timeline"=>HistoryJump::Redo,_=>HistoryJump::Position(request["args"]["position"].as_i64().filter(|v|(0..10000).contains(v)).ok_or("invalid host history position")? as i32)};
                    state.borrow_mut().undo_document=Some(Arc::downgrade(&document));
                    document.host_undo.begin_history(&view_id,id)?;
                    if let Err(error)=editor.enqueue(super::session::UiRequest {id,command:"plugin_history_barrier".into(),args:serde_json::json!({}),sink,link:Some(link.clone())}) {
                        document.host_undo.finish_history(&view_id,id);return Err(error);
                    }
                    Ok(HostReply {editor,document:Arc::downgrade(&document),lease,until:std::time::Instant::now()+std::time::Duration::from_secs(30),jump:Some(jump),imported:None,geometry:None,split:None,media:None,action:None})
                })();
                match outcome {Ok(reply)=>{let mut state=state.borrow_mut();state.pending.insert(id);state.history_requests.insert(id);state.host_replies.insert(id,reply);return Ok(());},Err(error)=>Err(error)}
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
                        owner.refresh_reaper_transport();Ok(())
                    })();
                    if let Err(error)=result {
                        let text=wide(&serde_json::json!({"version":1,"viewId":view_id,"id":id,"ok":false,"error":error}).to_string());
                        browser.PostWebMessageAsJson(PCWSTR(text.as_ptr()))?;return Ok(());
                    }
                }
                let (link,sink,full)={let state=state.borrow();(state.link.clone(),state.sink.clone(),state.pending.len()>=32 || state.pending.contains(&id))};
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
                    Ok(())=>{let mut state=state.borrow_mut();state.pending.insert(id);if undo_started {if let Some((document,_,_))=&history_context {state.undo_requests.insert(id,Arc::downgrade(document));}}return Ok(());},
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
            Err(error) => serde_json::json!({"version":1,"viewId":view_id,"id":id,"ok":false,"error":error}),
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
    let clip_editing = state
        .borrow()
        .link
        .owner()
        .is_ok_and(|owner| owner.host_clip_editing_available());
    let audio_import = state.borrow().link.owner().is_ok_and(|owner| {
        owner
            .project_history_host()
            .is_some_and(|host| host.can_import_audio())
    });
    let clip_splitting = state.borrow().link.owner().is_ok_and(|owner| {
        owner
            .project_history_host()
            .is_some_and(|host| host.can_split_clips())
    });
    let clip_clipboard = state.borrow().link.owner().is_ok_and(|owner| {
        owner
            .project_history_host()
            .is_some_and(|host| host.can_clipboard_items())
    });
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
        clip_editing,
        clip_splitting,
        clip_clipboard,
        audio_import,
        fade_axes,
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
        for audio_import in [true, false] {
            let flags = BootstrapFlags {
                transport_control: false,
                clip_editing: false,
                clip_splitting: false,
                clip_clipboard: false,
                audio_import,
                fade_axes: None,
            };
            assert_eq!(
                flags.to_json("view-7")["trackGrouping"],
                serde_json::json!(true),
                "audioImport={audio_import} 时 trackGrouping 被错误地关掉了"
            );
        }
    }

    /// 宿主能力标志必须**原样**透传：前端按它们决定开放哪些操作。
    #[test]
    fn host_capabilities_pass_through_unchanged() {
        let flags = BootstrapFlags {
            transport_control: true,
            clip_editing: false,
            clip_splitting: true,
            clip_clipboard: false,
            audio_import: true,
            fade_axes: Some("legacy"),
        };
        let json = flags.to_json("view-42");
        assert_eq!(json["version"], serde_json::json!(1));
        assert_eq!(json["viewId"], serde_json::json!("view-42"));
        assert_eq!(json["transportControl"], serde_json::json!(true));
        assert_eq!(json["clipEditing"], serde_json::json!(false));
        assert_eq!(json["clipSplitting"], serde_json::json!(true));
        assert_eq!(json["clipClipboard"], serde_json::json!(false));
        assert_eq!(json["audioImport"], serde_json::json!(true));
        assert_eq!(json["fadeAxes"], serde_json::json!("legacy"));
    }

    /// 未知轴语义必须如实报 `null`，前端据此保持只读（不猜）。
    #[test]
    fn an_unknown_fade_axis_set_is_reported_as_null() {
        let flags = BootstrapFlags {
            transport_control: false,
            clip_editing: true,
            clip_splitting: false,
            clip_clipboard: false,
            audio_import: false,
            fade_axes: None,
        };
        assert_eq!(flags.to_json("v")["fadeAxes"], serde_json::Value::Null);
    }

    /// 新轴宿主如实报 `continuous`：前端据此只给连续滑杆、不摆预设按钮。
    #[test]
    fn a_continuous_fade_axis_host_is_reported_as_such() {
        let flags = BootstrapFlags {
            transport_control: false,
            clip_editing: true,
            clip_splitting: false,
            clip_clipboard: false,
            audio_import: false,
            fade_axes: Some("continuous"),
        };
        assert_eq!(
            flags.to_json("v")["fadeAxes"],
            serde_json::json!("continuous")
        );
    }
}
