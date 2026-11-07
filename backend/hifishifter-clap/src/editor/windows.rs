//! Windows 内嵌视图：子 HWND + WebView2 + 同一份前端产物。
//!
//! 【为什么不是复制 VST3 的 windows.rs】那份 1188 行里大半是 `EditorClient` /
//! `EditorLink` / ARA 会话 / 219 条命令的策略。CLAP 阶段 1 没有那些后端，
//! 只需要"把 `frontend/dist` 真正画进宿主给的矩形"。这里只保留四件事：
//! 注册窗口类、建子 HWND、把 `hifishifter.invalid` 映射到资源目录、把页面消息
//! 交给 `protocol` 回执。
//!
//! 【为什么用进程级单视图】`clap.gui` 的回调只拿到 `clap_plugin_t*`，而
//! `clap_host_gui` 不携带每窗口上下文；一个 CLAP 实例同时只有一个 GUI，
//! 所以把视图状态放在 `Instance` 里、由 `plugin` 指针索引最贴近契约。
//! 本模块用 `OnceLock<Mutex<HashMap<usize, ViewState>>>` 保存它们，
//! `destroy` 时按 plugin 指针移除，不留悬垂 HWND。

#![cfg(windows)]

use super::protocol;
use crate::abi::clap_plugin_t;
use crate::gui::clap_window_t;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use webview2_com::Microsoft::Web::WebView2::Win32::*;
use webview2_com::{
    AddScriptToExecuteOnDocumentCreatedCompletedHandler, CoTaskMemPWSTR,
    CoreWebView2EnvironmentOptions, CreateCoreWebView2ControllerCompletedHandler,
    CreateCoreWebView2EnvironmentCompletedHandler, NavigationStartingEventHandler,
    NewWindowRequestedEventHandler, WebMessageReceivedEventHandler,
};
use windows::core::{w, Interface, PCWSTR, PWSTR};
use windows::Win32::Foundation::{HINSTANCE, HMODULE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_APARTMENTTHREADED};
use windows::Win32::System::LibraryLoader::{
    GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
};
use windows::Win32::System::Threading::GetCurrentThreadId;
use windows::Win32::UI::WindowsAndMessaging::*;

/// 与 VST3 同一个虚拟 origin：前端 CSP 与资源路径都按它写死，不能各用一套。
const ORIGIN: &str = "https://hifishifter.invalid/";
const PAGE_URL: &str = "https://hifishifter.invalid/plugin.html";
const CLASS_NAME: &str = "HiFiShifter.CLAP.Editor";
const WINDOW_TITLE: &str = "HiFiShifter CLAP editor";

// 本进程的 WebView2 环境与视图表（STA 线程独占；所有 CLAP 视图都在宿主 UI 线程创建）。
thread_local! {
    // 【为什么不是全局 Mutex】HWND 与 COM 接口都不是 Send；而 CLAP 的 GUI 回调
    // 契约本来就全在宿主主线程上，thread_local 既满足 Rust，也符合真实线程模型。
    static ENVIRONMENT: RefCell<Option<ICoreWebView2Environment>> = const { RefCell::new(None) };
    static VIEWS: RefCell<HashMap<usize, ViewState>> = RefCell::new(HashMap::new());
    /// 本线程处理过的页面消息总数（销毁时汇总，一条日志回答"页面跑通没有"）。
    static PAGE_MESSAGES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

struct ViewState {
    view_id: String,
    hwnd: HWND,
    parent: HWND,
    controller: Option<ICoreWebView2Controller>,
}

fn with_view<R>(plugin: *const clap_plugin_t, f: impl FnOnce(&mut ViewState) -> R) -> Option<R> {
    let key = plugin as usize;
    VIEWS.with(|views| views.borrow_mut().get_mut(&key).map(f))
}

fn class_name() -> &'static [u16] {
    static NAME: OnceLock<&'static [u16]> = OnceLock::new();
    NAME.get_or_init(|| Box::leak(wide(CLASS_NAME).into_boxed_slice()))
}

fn window_title() -> &'static [u16] {
    static TITLE: OnceLock<&'static [u16]> = OnceLock::new();
    TITLE.get_or_init(|| Box::leak(wide(WINDOW_TITLE).into_boxed_slice()))
}

fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(Some(0)).collect()
}

fn module() -> Result<HMODULE, String> {
    let mut module = HMODULE::default();
    // SAFETY: 取本模块内一个函数的地址作为锚点，UNCHANGED_REFCOUNT 不改引用计数。
    let ok = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            PCWSTR(module_address as *const () as *const u16),
            &mut module,
        )
    };
    if ok.is_err() || module.0.is_null() {
        return Err("cannot resolve the CLAP module handle".into());
    }
    Ok(module)
}
fn module_address() {}

/// 窗口过程：只需处理尺寸与销毁，其余交给默认过程（宿主泵负责消息循环）。
unsafe extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // 【为什么必须 guard_ffi】窗口过程是宿主消息泵调用的 C ABI 入口，
    // Rust panic 越过它即 abort 整个宿主进程。实测 WM_SIZE 在 set_parent
    // 持 RefMut 时重入就触发过一次；守卫把任何意外收敛成默认窗口过程。
    hifishifter_plugin_core::guard_ffi(LRESULT(0), || {
        match message {
            WM_SIZE => {
                let width = (lparam.0 as u32) & 0xffff;
                let height = ((lparam.0 as u32) >> 16) & 0xffff;
                let bounds = RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                };
                // 只取控制器快照，绝不在这里回调 Win32 API（会再派发消息）。
                let controllers: Vec<ICoreWebView2Controller> = VIEWS.with(|views| {
                    views
                        .borrow()
                        .values()
                        .filter(|view| view.hwnd == hwnd)
                        .filter_map(|view| view.controller.clone())
                        .collect()
                });
                for controller in controllers {
                    let _ = unsafe { controller.SetBounds(bounds) };
                }
                LRESULT(0)
            }
            WM_DESTROY => LRESULT(0),
            _ => unsafe { DefWindowProcW(hwnd, message, wparam, lparam) },
        }
    })
}
fn register_class() -> Result<(), String> {
    static REGISTERED: OnceLock<Result<(), String>> = OnceLock::new();
    REGISTERED
        .get_or_init(|| {
            // 窗口类名必须与窗口类同寿；OnceLock 只跑一次，泄漏到进程结束。
            let name: &'static [u16] = Box::leak(wide(CLASS_NAME).into_boxed_slice());
            let class = WNDCLASSW {
                lpfnWndProc: Some(window_proc),
                hInstance: HINSTANCE(module()?.0),
                lpszClassName: PCWSTR(name.as_ptr()),
                ..Default::default()
            };
            if unsafe { RegisterClassW(&class) } == 0 {
                return Err(windows::core::Error::from_thread().to_string());
            }
            Ok(())
        })
        .clone()
}
/// CLAP `clap.gui`：`is_api_supported` 的判据。只认 win32 嵌入。
pub fn is_api_supported(api: &str, floating: bool) -> bool {
    !floating && api == "win32"
}

pub fn preferred_api() -> Option<(&'static str, bool)> {
    Some(("win32", false))
}

/// `create`：建子 HWND（同步），随后异步初始化 WebView2 并导航。
pub fn create(plugin: *const clap_plugin_t, floating: bool) -> Result<(), String> {
    if floating {
        return Err("a floating CLAP window is not supported; embed with win32".into());
    }
    if plugin.is_null() {
        return Err("null CLAP plug-in identity".into());
    }
    // 视图状态先占位，避免并发 create 时重复建窗。
    let key = plugin as usize;
    {
        VIEWS.with(|v| {
            let mut views = v.borrow_mut();
            if views.contains_key(&key) {
                return Err("GUI already created for this CLAP instance".into());
            }
            views.insert(
                key,
                ViewState {
                    view_id: format!("clap-view-{}-{}", std::process::id(), key),
                    hwnd: HWND::default(),
                    parent: HWND::default(),
                    controller: None,
                },
            );
            Ok::<(), String>(())
        })?;
    }
    if let Err(error) = create_inner(plugin) {
        VIEWS.with(|v| {
            v.borrow_mut().remove(&key);
        });
        return Err(error);
    }
    Ok(())
}

fn create_inner(plugin: *const clap_plugin_t) -> Result<(), String> {
    register_class()?;
    let initialized = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED) };
    initialized
        .ok()
        .map_err(|e| format!("WebView2 STA unavailable: {e}"))?;
    let hwnd = unsafe {
        CreateWindowExW(
            WINDOW_EX_STYLE::default(),
            PCWSTR(class_name().as_ptr()),
            PCWSTR(window_title().as_ptr()),
            // 先建无父的 popup：CLAP 的 create() 早于 set_parent()，
            // 此刻宿主还没给 HWND，WS_CHILD 配 NULL 父窗口会被系统拒绝
            // （"无法创建最上层子窗口"）。set_parent 里再改成 WS_CHILD 并挂上去。
            WS_POPUP | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
            0,
            0,
            960,
            640,
            None,
            None,
            Some(HINSTANCE(module()?.0)),
            None,
        )
    }
    .map_err(|e| format!("CreateWindowExW failed: {e}"))?;
    with_view(plugin, |view| {
        view.hwnd = hwnd;
    })
    .ok_or("GUI view disappeared during creation")?;
    let folder = frontend_dir()?;
    begin_browser(plugin, hwnd, folder)
}

fn frontend_dir() -> Result<PathBuf, String> {
    let dir = hifishifter_plugin_core::resource_root::frontend_dir()?;
    log::info!("[clap][gui] frontend assets: {}", dir.display());
    Ok(dir)
}

fn begin_browser(plugin: *const clap_plugin_t, hwnd: HWND, folder: PathBuf) -> Result<(), String> {
    if let Some(environment) = ENVIRONMENT.with(|slot| slot.borrow_mut().take()) {
        ENVIRONMENT.with(|slot| *slot.borrow_mut() = Some(environment.clone()));
        return create_controller(plugin, hwnd, folder, &environment);
    }
    let completed = CreateCoreWebView2EnvironmentCompletedHandler::create(Box::new(
        move |result, environment| {
            let environment = match (result, environment) {
                (Ok(()), Some(environment)) => environment,
                (Err(error), _) => {
                    log::error!("[clap][gui] WebView2 environment failed: {error}");
                    return Ok(());
                }
                _ => {
                    log::error!("[clap][gui] WebView2 returned no environment");
                    return Ok(());
                }
            };
            ENVIRONMENT.with(|slot| *slot.borrow_mut() = Some(environment.clone()));
            if let Err(error) = create_controller(plugin, hwnd, folder.clone(), &environment) {
                log::error!("[clap][gui] controller failed: {error}");
            }
            Ok(())
        },
    ));
    let options: ICoreWebView2EnvironmentOptions = CoreWebView2EnvironmentOptions::default().into();
    // 用户数据目录用进程 ID 隔离，避免两个宿主进程抢同一份 profile。
    let profile = hifishifter_plugin_core::resource_root::resources_dir()
        .map(|dir| {
            dir.join("webview")
                .join(format!("view-{}", std::process::id()))
        })
        .unwrap_or_else(|_| std::env::temp_dir().join(format!("hfs-clap-{}", std::process::id())));
    let _ = std::fs::create_dir_all(&profile);
    let profile = wide(&profile.to_string_lossy());
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

fn create_controller(
    plugin: *const clap_plugin_t,
    hwnd: HWND,
    folder: PathBuf,
    environment: &ICoreWebView2Environment,
) -> Result<(), String> {
    let ready = CreateCoreWebView2ControllerCompletedHandler::create(Box::new(
        move |result, controller| {
            let controller = match (result, controller) {
                (Ok(()), Some(controller)) => controller,
                (Err(error), _) => {
                    log::error!("[clap][gui] controller creation failed: {error}");
                    return Ok(());
                }
                _ => {
                    log::error!("[clap][gui] WebView2 returned no controller");
                    return Ok(());
                }
            };
            with_view(plugin, |view| {
                view.controller = Some(controller.clone());
            });
            if let Err(error) = configure_browser(plugin, &controller, &folder) {
                log::error!("[clap][gui] browser setup failed: {error}");
            }
            Ok(())
        },
    ));
    unsafe { environment.CreateCoreWebView2Controller(hwnd, &ready) }.map_err(|e| e.to_string())
}

fn configure_browser(
    plugin: *const clap_plugin_t,
    controller: &ICoreWebView2Controller,
    folder: &std::path::Path,
) -> windows::core::Result<()> {
    let hwnd = with_view(plugin, |v| v.hwnd).unwrap_or_default();
    let mut bounds = RECT::default();
    unsafe {
        GetClientRect(hwnd, &mut bounds)?;
        controller.SetBounds(bounds)?;
        controller.SetIsVisible(true)?;
    }
    let browser = unsafe { controller.CoreWebView2()? };
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
        let message_browser = browser.clone();
        browser.clone().add_WebMessageReceived(
            &WebMessageReceivedEventHandler::create(Box::new(move |_, args| {
                let Some(args) = args else {
                    return Ok(());
                };
                let mut json = PWSTR::null();
                args.WebMessageAsJson(&mut json)?;
                let text = CoTaskMemPWSTR::from(json).to_string();
                // 【为什么记这一行】"窗口开了但页面没跑起来"与"页面在跑、只是没内容"
                // 在宿主侧完全同形。第一条页面消息是**JS 真的执行了**的唯一证据，
                // 而用户报障时我们只有日志——所以这条只打一次，带命令名。
                // 【为什么是前 8 条 + 总数】页面启动会依次问偏好、探活、拉时间线；
                // 只看第一条分不清"页面起来了"与"页面卡在第二步"，而逐条打印
                // 又会把日志刷满。前 8 条给序列，销毁时给总数。
                let seen = PAGE_MESSAGES.with(|count| {
                    let seen = count.get();
                    count.set(seen + 1);
                    seen
                });
                if seen < 8 {
                    // 命令名来自页面（外部输入），只截取前 64 字节且按 debug 打印；
                    // 它回答的是"页面到底跑到哪一步了"。
                    let command = text
                        .split("\"command\":\"")
                        .nth(1)
                        .and_then(|rest| rest.split('"').next())
                        .unwrap_or("?");
                    log::info!(
                        "[clap][gui] page message #{} ({} bytes, command={command})",
                        seen + 1,
                        text.len()
                    );
                }
                let Some(reply) = with_view(plugin, |view| {
                    protocol::parse_request(&text, &view.view_id)
                        .map(|request| protocol::reply(&view.view_id, &request))
                })
                .flatten() else {
                    return Ok(());
                };
                // 【为什么是 PostWebMessageAsJson】这是 WebView2 的原生投递通道，
                // 页面从 `chrome.webview` 的 message 事件收到它。`__hfsDeliver` 是
                // WebKit 垫片里的函数，Windows 上没有 —— 用它会让每条回执
                // 静默丢失，表现是页面发了一条消息就再没动静（实测）。
                let payload = wide(&reply.to_string());
                let _ = message_browser.PostWebMessageAsJson(PCWSTR(payload.as_ptr()));
                Ok(())
            })),
            &mut token,
        )?;
    }
    // bootstrap 必须在任何页面脚本之前注入，且导航要等注入完成。
    let boot = with_view(plugin, |view| protocol::bootstrap(&view.view_id))
        .unwrap_or_else(|| protocol::bootstrap("clap-view"));
    let script = wide(&format!("window.__HFS_PLUGIN_BOOTSTRAP__={boot};"));
    let nav_browser = browser.clone();
    let ready =
        AddScriptToExecuteOnDocumentCreatedCompletedHandler::create(Box::new(move |result, _| {
            if let Err(error) = result {
                log::error!("[clap][gui] bootstrap injection failed: {error}");
                return Ok(());
            }
            let url = wide(PAGE_URL);
            unsafe { nav_browser.Navigate(PCWSTR(url.as_ptr()))? };
            log::info!("[clap][gui] frontend navigation started");
            Ok(())
        }));
    unsafe { browser.AddScriptToExecuteOnDocumentCreated(PCWSTR(script.as_ptr()), &ready) }
}

/// `set_parent`：把子窗口挂到宿主给的 HWND 上。
///
/// # Safety
/// `window` 必须指向宿主给出的、在调用期间有效的 `clap_window_t`；
/// 其 `handle` 必须是当前 UI 线程上的有效 HWND。
pub unsafe fn set_parent(plugin: *const clap_plugin_t, window: *const clap_window_t) -> bool {
    if window.is_null() {
        return false;
    }
    let parent = HWND(unsafe { (*window).handle });
    if parent.0.is_null() || !unsafe { IsWindow(Some(parent)) }.as_bool() {
        return false;
    }
    let thread = unsafe { GetCurrentThreadId() };
    if unsafe { GetWindowThreadProcessId(parent, None) } != thread {
        log::error!("[clap][gui] set_parent called off the parent UI thread");
        return false;
    }
    // 【为什么先把数据拷出来】SetWindowPos / SetParent 会同步派发 WM_SIZE，
    // 而 window_proc 也要借用 VIEWS。持着 RefMut 调 Win32 API 必然触发
    // "RefCell already borrowed" 并在 extern "system" 里 abort（实测）。
    let snapshot = with_view(plugin, |view| {
        view.parent = parent;
        (view.hwnd, view.controller.clone())
    });
    let Some((hwnd, controller)) = snapshot else {
        return false;
    };
    unsafe {
        let _ = SetWindowLongPtrW(
            hwnd,
            GWL_STYLE,
            (WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN).0 as isize,
        );
        let _ = SetParent(hwnd, Some(parent));
        let mut bounds = RECT::default();
        let _ = GetClientRect(parent, &mut bounds);
        let _ = SetWindowPos(
            hwnd,
            None,
            0,
            0,
            bounds.right - bounds.left,
            bounds.bottom - bounds.top,
            SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
        );
        if let Some(controller) = &controller {
            let _ = controller.SetBounds(bounds);
        }
    }
    true
}

pub fn set_size(plugin: *const clap_plugin_t, width: u32, height: u32) -> bool {
    let snapshot = with_view(plugin, |view| (view.hwnd, view.controller.clone()));
    let Some((hwnd, controller)) = snapshot else {
        return false;
    };
    unsafe {
        let _ = SetWindowPos(
            hwnd,
            None,
            0,
            0,
            width as i32,
            height as i32,
            SWP_NOZORDER | SWP_NOACTIVATE,
        );
    }
    let bounds = RECT {
        left: 0,
        top: 0,
        right: width as i32,
        bottom: height as i32,
    };
    if let Some(controller) = &controller {
        let _ = unsafe { controller.SetBounds(bounds) };
    }
    true
}

pub fn get_size(plugin: *const clap_plugin_t) -> Option<(u32, u32)> {
    with_view(plugin, |view| unsafe {
        let mut bounds = RECT::default();
        let _ = GetClientRect(view.hwnd, &mut bounds);
        (
            (bounds.right - bounds.left) as u32,
            (bounds.bottom - bounds.top) as u32,
        )
    })
}

pub fn show(plugin: *const clap_plugin_t) -> bool {
    let snapshot = with_view(plugin, |view| (view.hwnd, view.controller.clone()));
    let Some((hwnd, controller)) = snapshot else {
        return false;
    };
    unsafe {
        let _ = ShowWindow(hwnd, SW_SHOW);
        if let Some(controller) = &controller {
            let _ = controller.SetIsVisible(true);
        }
    }
    true
}

pub fn hide(plugin: *const clap_plugin_t) -> bool {
    let snapshot = with_view(plugin, |view| (view.hwnd, view.controller.clone()));
    let Some((hwnd, controller)) = snapshot else {
        return false;
    };
    unsafe {
        let _ = ShowWindow(hwnd, SW_HIDE);
        if let Some(controller) = &controller {
            let _ = controller.SetIsVisible(false);
        }
    }
    true
}

pub fn is_created(plugin: *const clap_plugin_t) -> bool {
    VIEWS.with(|views| views.borrow().contains_key(&(plugin as usize)))
}

/// `destroy`：关控制器、销毁窗口、从表里移除。可重入。
pub fn destroy(plugin: *const clap_plugin_t) {
    let key = plugin as usize;
    let view = VIEWS.with(|views| views.borrow_mut().remove(&key));
    if let Some(mut view) = view {
        if let Some(controller) = view.controller.take() {
            let _ = unsafe { controller.Close() };
        }
        if !view.hwnd.0.is_null() {
            unsafe {
                let _ = DestroyWindow(view.hwnd);
            }
        }
        let handled = PAGE_MESSAGES.with(|count| count.get());
        log::info!("[clap][gui] view destroyed after {handled} page message(s)");
    }
}

/// 模块卸载前的收尾：所有视图必须已经关掉。
pub fn shutdown() -> bool {
    let alive = VIEWS.with(|views| views.borrow().len());
    if alive != 0 {
        log::warn!("[clap][gui] shutdown refused: {alive} view(s) still alive");
        return false;
    }
    unsafe { CoUninitialize() };
    true
}
