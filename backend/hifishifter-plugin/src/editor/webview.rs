//! 原生WebView2子窗口：借用REAPER消息循环，取消异步创建，不启动Tauri/app。
use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::sync::{Arc,mpsc};
use std::sync::atomic::AtomicBool;
use std::collections::HashSet;
use webview2_com::Microsoft::Web::WebView2::Win32::*;
use webview2_com::{AddScriptToExecuteOnDocumentCreatedCompletedHandler, CoTaskMemPWSTR,
    CreateCoreWebView2ControllerCompletedHandler, CreateCoreWebView2EnvironmentCompletedHandler,
    NavigationStartingEventHandler, NewWindowRequestedEventHandler, WebMessageReceivedEventHandler};
use windows::core::{w, Interface, PCWSTR, PWSTR};
use windows::Win32::Foundation::{HANDLE, HINSTANCE, HMODULE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_APARTMENTTHREADED};
use windows::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleExW,
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT};
use windows::Win32::System::LibraryLoader::GET_MODULE_HANDLE_EX_FLAG_PIN;
use windows::Win32::System::Threading::GetCurrentThreadId;
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::Win32::UI::Input::KeyboardAndMouse::{GetFocus,SetFocus};

const ORIGIN: &str = "https://hifishifter.invalid/";
static NEXT_VIEW: AtomicU64 = AtomicU64::new(1);
static WINDOWS: AtomicUsize = AtomicUsize::new(0);
static CLASS_LOCK: Mutex<bool> = Mutex::new(false);
static WEBVIEW_MODULE_PIN: OnceLock<Result<(),String>> = OnceLock::new();

struct BrowserState {
    hwnd: HWND,
    closed: bool,
    view_id: String,
    controller: Option<ICoreWebView2Controller>,
    error_window: Option<HWND>,
    link:std::sync::Arc<super::routing::EditorLink>,
    sink:super::session::UiSink,
    replies:mpsc::Receiver<serde_json::Value>,
    events:mpsc::Receiver<serde_json::Value>,
    pending:HashSet<u64>,
}
struct WindowData {
    state: Rc<RefCell<BrowserState>>,
    transferred: Rc<Cell<bool>>,
    token: usize,
}
impl Drop for WindowData {
    fn drop(&mut self) {
        let controller = {
            let mut state = self.state.borrow_mut();
            state.closed = true;
            state.sink.closed.store(true,Ordering::Release);
            state.controller.take()
        };
        // Close会同步回调；此时不持有RefCell borrow。
        if let Some(controller) = controller { let _ = unsafe { controller.Close() }; }
        let _=unsafe {KillTimer(Some(self.state.borrow().hwnd),0x4853)};
        unsafe { CoUninitialize(); }
    }
}

/// 只保存窗口身份，不把非Send COM接口从UI线程带走。资源由自有WndProc析构。
pub(super) struct NativeEditor { key: WindowKey }
#[derive(Clone, Copy)]
pub(super) struct WindowKey { hwnd: usize, thread: u32, token: usize }
impl WindowKey {
    /// 句柄被系统复用后不能移动/关闭另一实例窗口；属性token在NCDESTROY时撤销。
    fn valid(self) -> bool {
        let hwnd = HWND(self.hwnd as *mut c_void);
        unsafe { IsWindow(Some(hwnd)).as_bool() &&
            GetPropW(hwnd,w!("HiFiShifter.ARA.ViewToken")).0 as usize == self.token }
    }
    /// 只在宿主UI线程把焦点交给自有编辑器；已在浏览器内时不重置DOM焦点。
    pub(super) fn focus(self) -> Result<(),String> {
        if unsafe {GetCurrentThreadId()} != self.thread || !self.valid() {
            return Err("focus on stale window or off UI thread".into());
        }
        let hwnd=HWND(self.hwnd as *mut c_void);
        let current=unsafe {GetFocus()};
        if current==hwnd || unsafe {IsChild(hwnd,current)}.as_bool() {return Ok(());}
        let result=unsafe {SetFocus(Some(hwnd))};
        // 首次SetFocus的旧句柄可能为null，Win32包装返回Err但真实焦点已成功设置。
        let focused=unsafe {GetFocus()};
        if focused==hwnd || unsafe {IsChild(hwnd,focused)}.as_bool() {Ok(())}
        else {result.map(|_|()).map_err(|e|e.to_string())}
    }
    /// 可复制的窗口身份不持有COM资源，允许调用前释放view锁以防Win32同步重入。
    pub(super) fn resize(self, width: i32, height: i32) -> Result<(),String> {
        if unsafe { GetCurrentThreadId() } != self.thread || !self.valid() {
            return Err("resize on stale window or off UI thread".into());
        }
        unsafe { MoveWindow(HWND(self.hwnd as *mut c_void),0,0,width,height,true) }.map_err(|e| e.to_string())
    }
}
impl NativeEditor {
    /// 同线程创建子窗口，异步浏览器由宿主消息循环完成，禁止嵌套消息泵。
    pub(super) fn attach(parent: *mut c_void, width: i32, height: i32,
        link:std::sync::Arc<super::routing::EditorLink>) -> Result<Self, String> {
        let parent = HWND(parent);
        if !unsafe { IsWindow(Some(parent)) }.as_bool() { return Err("parent is not a live HWND".into()); }
        let thread = unsafe { GetCurrentThreadId() };
        if unsafe { GetWindowThreadProcessId(parent, None) } != thread {
            return Err("IPlugView attached off the parent UI thread".into());
        }
        register_class()?;
        let instance = HINSTANCE(module()?.0);
        let initialized = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED) };
        initialized.ok().map_err(|e| format!("WebView2 STA unavailable: {e}"))?;
        let token = NEXT_VIEW.fetch_add(1,Ordering::Relaxed) as usize;
        let view_id = format!("view-{}-{token}", std::process::id());
        let (reply,replies)=mpsc::channel();
        let (event_sender,events)=mpsc::sync_channel(128);
        let sink=super::session::UiSink {view_id:view_id.clone(),reply,events:event_sender,closed:Arc::new(AtomicBool::new(false))};
        let state = Rc::new(RefCell::new(BrowserState { hwnd: HWND::default(), closed: false,
            view_id, controller: None, error_window: None, link,sink,replies,events,pending:HashSet::new() }));
        let transferred = Rc::new(Cell::new(false));
        let data = Box::into_raw(Box::new(WindowData { state: state.clone(), transferred: transferred.clone(), token }));
        let result = unsafe { CreateWindowExW(WINDOW_EX_STYLE::default(), w!("HiFiShifter.ARA.Editor"),
            w!("HiFiShifter plugin editor initializing…"),
            WS_CHILD | WS_VISIBLE | WS_TABSTOP | WS_CLIPSIBLINGS | WS_CLIPCHILDREN, 0,0,width,height,
            Some(parent), None, Some(instance), Some(data.cast())) };
        let hwnd = match result {
            Ok(hwnd) => hwnd,
            Err(error) => {
                // NCCREATE之前失败时系统未接管data；之后失败则NCDESTROY已负责释放。
                if !transferred.get() { unsafe { drop(Box::from_raw(data)); } }
                return Err(format!("create child window: {error}"));
            }
        };
        let native = Self { key:WindowKey { hwnd: hwnd.0 as usize, thread, token } };
        if unsafe {SetTimer(Some(hwnd),0x4853,20,None)}==0 {show_error(&state,"UI response timer unavailable");}
        if let Err(error) = begin_browser(&state) { show_error(&state, &error); }
        Ok(native)
    }
    /// 提供不持COM资源的几何操作身份，供view在释放锁后使用。
    pub(super) fn window_key(&self) -> WindowKey { self.key }
}
impl Drop for NativeEditor {
    fn drop(&mut self) {
        let hwnd = HWND(self.key.hwnd as *mut c_void);
        if !self.key.valid() { return; }
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
    unsafe { GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        PCWSTR(module_address as *const () as *const u16), &mut module) }.map_err(|e| e.to_string())?;
    Ok(module)
}
fn module_address() {}
/// WebView2的环境创建没有取消API，迟到COM回调仍会进入本DLL；固定模块到宿主退出。
/// 只固定这一模块一次，不保留窗口/浏览器/worker；开发升级需要正常退出REAPER。
fn pin_callback_code() -> Result<(),String> {
    WEBVIEW_MODULE_PIN.get_or_init(|| {
        let mut handle=HMODULE::default();
        unsafe { GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
            PCWSTR(module_address as *const () as *const u16),&mut handle) }.map_err(|e|e.to_string())
    }).clone()
}
fn assets() -> Result<PathBuf,String> {
    if let Some(path) = std::env::var_os("HIFISHIFTER_ARA_EDITOR_ASSETS") {
        let path = PathBuf::from(path);
        if !path.is_absolute() { return Err("editor asset override must be absolute".into()); }
        let path = path.canonicalize().map_err(|e| e.to_string())?;
        if !path.join("plugin.html").is_file() { return Err("editor plugin.html missing".into()); }
        return Ok(path);
    }
    let mut buffer = vec![0u16; 32768];
    let length = unsafe { GetModuleFileNameW(Some(module()?), &mut buffer) } as usize;
    if length == 0 || length >= buffer.len() { return Err("module path unavailable".into()); }
    let file = PathBuf::from(String::from_utf16_lossy(&buffer[..length]));
    let folder = file.parent().and_then(|p| p.parent()).ok_or("VST3 Contents directory unavailable")?
        .join("Resources").join("frontend");
    if !folder.join("plugin.html").is_file() { return Err(format!("plugin frontend assets missing: {}", folder.display())); }
    Ok(folder)
}
fn wide(text: &str) -> Vec<u16> { text.encode_utf16().chain(Some(0)).collect() }

fn register_class() -> Result<(),String> {
    let mut registered = CLASS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if *registered { return Ok(()); }
    let class = WNDCLASSW { lpfnWndProc: Some(window_proc), hInstance: HINSTANCE(module()?.0),
        lpszClassName: w!("HiFiShifter.ARA.Editor"), ..Default::default() };
    if unsafe { RegisterClassW(&class) } == 0 { return Err(windows::core::Error::from_win32().to_string()); }
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
        let Ok(module) = module() else { return false; };
        if unsafe { UnregisterClassW(w!("HiFiShifter.ARA.Editor"), Some(HINSTANCE(module.0))) }.is_err() { return false; }
        *registered = false;
    }
    true
}

unsafe extern "system" fn window_proc(hwnd: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    // 所有可能panic的自有逻辑都在边界内，不能越过Win32回调栈。
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if message == WM_NCCREATE {
            let create = &*(lparam.0 as *const CREATESTRUCTW);
            let data = &*(create.lpCreateParams as *const WindowData);
            data.transferred.set(true);
            data.state.borrow_mut().hwnd = hwnd;
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, create.lpCreateParams as isize);
            WINDOWS.fetch_add(1, Ordering::AcqRel);
            if SetPropW(hwnd,w!("HiFiShifter.ARA.ViewToken"),Some(HANDLE(data.token as *mut c_void))).is_err() {
                return LRESULT(0);
            }
        }
        let pointer = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut WindowData;
        if !pointer.is_null() {
            if message==WM_TIMER && wparam.0==0x4853 {
                deliver(&(*pointer).state);
                return LRESULT(0);
            } else if message == WM_SETFOCUS {
                // 宿主Tab/onFocus进入自有HWND后，使用标准WebView2接口进入HTML控件。
                // MoveFocus可能同步通知宿主；调用前释放BrowserState借用。
                let controller=(*pointer).state.borrow().controller.clone();
                if let Some(controller)=controller {let _=controller.MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_NEXT);}
                return LRESULT(0);
            } else if message == WM_SIZE {
                let controller = (*pointer).state.borrow().controller.clone();
                if let Some(controller) = controller {
                    let mut bounds = RECT::default();
                    if GetClientRect(hwnd,&mut bounds).is_ok() { let _ = controller.SetBounds(bounds); }
                }
            } else if message == WM_CLOSE {
                let _ = DestroyWindow(hwnd);
                return LRESULT(0);
            } else if message == WM_NCDESTROY {
                SetWindowLongPtrW(hwnd,GWLP_USERDATA,0);
                let _ = RemovePropW(hwnd,w!("HiFiShifter.ARA.ViewToken"));
                drop(Box::from_raw(pointer));
                WINDOWS.fetch_sub(1,Ordering::AcqRel);
            }
        }
        DefWindowProcW(hwnd,message,wparam,lparam)
    })).unwrap_or(LRESULT(0))
}

/// worker只写JSON邮箱；COM响应仅在本窗口UI线程发送，且不持RefCell borrow。
fn deliver(state:&Rc<RefCell<BrowserState>>) {
    let (controller,mut replies,events)={
        let mut state=state.borrow_mut();
        if state.closed || state.controller.is_none() {return;}
        let replies=state.replies.try_iter().take(32).collect::<Vec<_>>();
        let events=state.events.try_iter().take(64).collect::<Vec<_>>();
        for reply in &replies {if let Some(id)=reply["id"].as_u64() {state.pending.remove(&id);}}
        (state.controller.clone(),replies,events)
    };
    let Some(controller)=controller else {return;};
    let Ok(browser)=(unsafe {controller.CoreWebView2()}) else {return;};
    replies.extend(events);
    for response in replies {
        let mut text=response.to_string();
        if text.len()>8*1024*1024 {
            text=serde_json::json!({"version":1,"viewId":response["viewId"],"id":response["id"],
                "ok":false,"error":"native response budget exceeded"}).to_string();
        }
        let text=wide(&text);
        if unsafe {browser.PostWebMessageAsJson(PCWSTR(text.as_ptr()))}.is_err() {break;}
    }
}

fn show_error(state: &Rc<RefCell<BrowserState>>, error: &str) {
    crate::log_line(&format!("Native editor error: {error}"));
    let (closed, hwnd) = { let state = state.borrow(); (state.closed,state.hwnd) };
    if !closed {
        let text = wide(&format!("HiFiShifter editor: {error}"));
        let existing = state.borrow().error_window;
        if let Some(existing) = existing {
            let _ = unsafe { SetWindowTextW(existing, PCWSTR(text.as_ptr())) };
        } else {
            // 自有child没有标题栏；用真实STATIC控件显示错误，不能只写不可见window text。
            if let Ok(error_window) = unsafe { CreateWindowExW(WINDOW_EX_STYLE::default(),w!("STATIC"),
                PCWSTR(text.as_ptr()),WS_CHILD | WS_VISIBLE,12,12,600,120,Some(hwnd),None,None,None) } {
                state.borrow_mut().error_window = Some(error_window);
            }
        }
    }
}

/// WebView异步回调仅捕获weak state，removed后不会重新创建或访问裸view。
fn begin_browser(state: &Rc<RefCell<BrowserState>>) -> Result<(),String> {
    let folder = assets()?;
    pin_callback_code()?;
    let id = state.borrow().view_id.clone();
    let profile = std::env::temp_dir().join("hifishifter-plugin-webview").join(&id);
    std::fs::create_dir_all(&profile).map_err(|e| e.to_string())?;
    let profile = wide(&profile.to_string_lossy());
    let weak = Rc::downgrade(state);
    let completed = CreateCoreWebView2EnvironmentCompletedHandler::create(Box::new(move |result, environment| {
        let Some(state) = weak.upgrade() else { return Ok(()); };
        if state.borrow().closed { return Ok(()); }
        let outcome = (|| -> windows::core::Result<()> {
            result?;
            let environment = environment.ok_or_else(|| windows::core::Error::from_hresult(windows::core::HRESULT(0x80004003u32 as i32)))?;
            let weak = Rc::downgrade(&state);
            let controller_ready = CreateCoreWebView2ControllerCompletedHandler::create(Box::new(move |result, controller| {
                let Some(state) = weak.upgrade() else {
                    if let Some(controller) = controller { let _ = unsafe { controller.Close() }; }
                    return Ok(());
                };
                if state.borrow().closed {
                    if let Some(controller) = controller { let _ = unsafe { controller.Close() }; }
                    return Ok(());
                }
                if let Err(error) = result { show_error(&state,&error.to_string()); return Ok(()); }
                if let Some(controller) = controller {
                    state.borrow_mut().controller = Some(controller.clone());
                    if let Err(error) = configure_browser(&state,&controller,&folder) {
                        show_error(&state,&error.to_string());
                    }
                } else { show_error(&state,"WebView2 returned no controller"); }
                Ok(())
            }));
            let hwnd = state.borrow().hwnd;
            unsafe { environment.CreateCoreWebView2Controller(hwnd,&controller_ready) }
        })();
        if let Err(error) = outcome { show_error(&state,&error.to_string()); }
        Ok(())
    }));
    unsafe { CreateCoreWebView2EnvironmentWithOptions(PCWSTR::null(),PCWSTR(profile.as_ptr()),
        None::<&ICoreWebView2EnvironmentOptions>, &completed) }.map_err(|e| e.to_string())
}

fn configure_browser(state: &Rc<RefCell<BrowserState>>, controller: &ICoreWebView2Controller,
    folder: &std::path::Path) -> windows::core::Result<()> {
    let hwnd = state.borrow().hwnd;
    let mut bounds = RECT::default();
    unsafe { GetClientRect(hwnd,&mut bounds)?; controller.SetBounds(bounds)?; controller.SetIsVisible(true)?; }
    // 异步创建完成时仅延续宿主已交给本窗口的焦点，不能抢另一个FX/应用的焦点。
    if unsafe {GetFocus()}==hwnd {unsafe {controller.MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_NEXT)?;}}
    let browser = unsafe { controller.CoreWebView2()? };
    let mapping: ICoreWebView2_3 = browser.cast()?;
    let path = wide(&folder.to_string_lossy());
    unsafe { mapping.SetVirtualHostNameToFolderMapping(w!("hifishifter.invalid"),PCWSTR(path.as_ptr()),
        COREWEBVIEW2_HOST_RESOURCE_ACCESS_KIND_DENY_CORS)?; }
    let settings = unsafe { browser.Settings()? };
    unsafe { settings.SetAreHostObjectsAllowed(false)?; settings.SetAreDefaultContextMenusEnabled(false)?; }
    let mut token = 0;
    unsafe {
        browser.add_NavigationStarting(&NavigationStartingEventHandler::create(Box::new(|_,args| {
            if let Some(args) = args {
                let mut uri = PWSTR::null();
                args.Uri(&mut uri)?;
                if !CoTaskMemPWSTR::from(uri).to_string().starts_with(ORIGIN) { args.SetCancel(true)?; }
            }
            Ok(())
        })),&mut token)?;
        browser.add_NewWindowRequested(&NewWindowRequestedEventHandler::create(Box::new(|_,args| {
            if let Some(args) = args { args.SetHandled(true)?; }
            Ok(())
        })),&mut token)?;
    }
    let weak = Rc::downgrade(state);
    unsafe { browser.add_WebMessageReceived(&WebMessageReceivedEventHandler::create(Box::new(move |browser,args| {
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
        let Ok(request) = serde_json::from_str::<serde_json::Value>(&text) else { return Ok(()); };
        if request["version"] != 1 || request["viewId"] != view_id { return Ok(()); }
        let Some(id) = request["id"].as_u64().filter(|id| *id > 0 && *id <= 9_007_199_254_740_991) else { return Ok(()); };
        // 普通命令只排队；UI回调不得分析、推理、读文件或等待worker。
        let result = match request["command"].as_str() {
            Some("ping") => Ok(serde_json::json!({"ok":true,"mode":"plugin","view_id":view_id,
                "connected":state.borrow().link.owner().is_ok()})),
            Some("log_frontend_error") => {
                crate::log_line(&format!("Embedded frontend: {}",request["args"])); Ok(serde_json::Value::Null)
            }
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
            Some(command)=>{
                if command=="set_transport" && request["args"]["playheadSec"].is_number() {
                    let link=state.borrow().link.clone();
                    let result=(||->Result<(),String> {
                        let position=request["args"]["playheadSec"].as_f64().ok_or("invalid seek position")?;
                        link.owner()?.host_playback().ok_or("host does not provide ARA playback control")?
                            .set_position(position).map_err(|e|e.to_string())
                    })();
                    if let Err(error)=result {
                        let text=wide(&serde_json::json!({"version":1,"viewId":view_id,"id":id,"ok":false,"error":error}).to_string());
                        browser.PostWebMessageAsJson(PCWSTR(text.as_ptr()))?;return Ok(());
                    }
                }
                let (link,sink,full)={let state=state.borrow();(state.link.clone(),state.sink.clone(),state.pending.len()>=32 || state.pending.contains(&id))};
                let outcome=if full {Err("native pending request budget exceeded".into())} else {
                    link.owner().and_then(|owner|owner.editor_session()?.enqueue(super::session::UiRequest {
                        id,command:command.into(),args:request.get("args").cloned().unwrap_or_else(||serde_json::json!({})),sink,
                    }))
                };
                match outcome {
                    Ok(())=>{state.borrow_mut().pending.insert(id);return Ok(());},
                    Err(error)=>Err(error),
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
    })),&mut token)?; }
    let transport_control=state.borrow().link.owner().is_ok_and(|owner|owner.host_playback().is_some());
    let boot = serde_json::json!({"version":1,"viewId":state.borrow().view_id,"transportControl":transport_control});
    let script = wide(&format!("window.__HFS_PLUGIN_BOOTSTRAP__={boot};"));
    let weak = Rc::downgrade(state);
    let navigate = browser.clone();
    let ready = AddScriptToExecuteOnDocumentCreatedCompletedHandler::create(Box::new(move |result,_| {
        let Some(state) = weak.upgrade() else { return Ok(()); };
        if state.borrow().closed { return Ok(()); }
        if let Err(error) = result { show_error(&state,&error.to_string()); return Ok(()); }
        unsafe { navigate.Navigate(w!("https://hifishifter.invalid/plugin.html"))?; }
        crate::log_line("Native editor frontend navigation started (commands pending session binding)");
        Ok(())
    }));
    unsafe { browser.AddScriptToExecuteOnDocumentCreated(PCWSTR(script.as_ptr()),&ready) }
}

#[cfg(test)]
mod focus_tests {
    use super::*;

    /// 用真实自有HWND验证Tab边界，不凭源码是否包含某个style判断可达性。
    #[test]
    fn native_editor_is_a_tab_stop_in_its_host_parent() {
        // SAFETY: 夹具父窗口仅在本线程创建/销毁，不操作任何用户或REAPER窗口。
        let parent=unsafe {CreateWindowExW(WINDOW_EX_STYLE::default(),w!("STATIC"),w!("HiFiShifter focus fixture"),
            WS_OVERLAPPEDWINDOW,0,0,1100,720,None,None,None,None)}.unwrap();
        let native=NativeEditor::attach(parent.0,1100,720,Arc::new(Default::default())).unwrap();
        let child=HWND(native.key.hwnd as *mut c_void);
        // GetNextDlgTabItem在没有任何Tab stop时会回退到第一个child，单看句柄会假通过。
        let tab_stop=unsafe {GetWindowLongPtrW(child,GWL_STYLE)} as u32 & WS_TABSTOP.0 != 0;
        let key=native.key;
        assert!(std::thread::spawn(move||key.focus().is_err()).join().unwrap(),"不能从非UI线程触碰宿主焦点");
        let focus_result=key.focus();
        let focused=unsafe {GetFocus()};
        // SAFETY: 两个窗口均为本线程且存活；仅查询标准Windows对话框导航结果。
        let target=unsafe {GetNextDlgTabItem(parent,None,false)}.ok();
        drop(native);
        assert!(key.focus().is_err(),"关闭后的窗口租约不能再夺取焦点");
        // SAFETY: 清理测试创建的唯一父窗口，子窗口已由NativeEditor析构。
        unsafe {DestroyWindow(parent)}.unwrap();
        assert!(tab_stop,"真实编辑器HWND必须接受宿主Tab停留，不能依赖导航的首child回退");
        assert!(focus_result.is_ok(),"首次无旧焦点时仍应成功: {focus_result:?}");
        assert_eq!(focused,child,"宿主焦点请求必须进入自有窗口");
        assert_eq!(target,Some(child),"宿主Tab顺序必须能到达真实编辑器子窗口");
    }
}
