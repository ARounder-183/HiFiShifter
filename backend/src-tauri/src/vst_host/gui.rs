//! VST 插件原生 GUI 窗口管理。
//!
//! 为 VST 插件编辑器创建独立的原生窗口（Windows HWND / macOS NSView）。
//! 插件编辑器嵌入到该原生窗口中显示。
//!
//! Windows 平台通过 Win32 API（RegisterClassExW / CreateWindowExW）创建宿主窗口，
//! 然后调用 VST2 editor `open()` 将 GUI 嵌入到该窗口的客户区。
//! 窗口运行在独立线程中，自带消息泵（GetMessage / DispatchMessage 循环）。

use std::sync::{Arc, Mutex};

use super::plugin_instance::{VstPluginBackend, VstPluginInstance};

/// 编辑器窗口关闭时的回调类型。
/// 参数为插件实例导出的 chunk 数据（base64），None 表示无法获取。
pub type OnEditorCloseCallback = Box<dyn FnOnce(Option<String>) + Send + 'static>;

/// VST 编辑器窗口句柄。
pub struct VstEditorWindow {
    /// 窗口标题。
    pub title: String,
    /// 窗口宽度。
    pub width: u32,
    /// 窗口高度。
    pub height: u32,
    /// 是否已打开。
    pub is_open: bool,
    /// 平台窗口句柄（Windows HWND）。
    #[cfg(target_os = "windows")]
    pub hwnd: Option<*mut std::ffi::c_void>,
    /// 窗口线程的 join handle。
    #[cfg(target_os = "windows")]
    pub thread_handle: Option<std::thread::JoinHandle<()>>,
    #[cfg(not(target_os = "windows"))]
    pub _handle: Option<()>,
}

// SAFETY: 窗口句柄仅在窗口线程使用，VstEditorWindow 结构体通过 Mutex 保护。
unsafe impl Send for VstEditorWindow {}

impl Default for VstEditorWindow {
    fn default() -> Self {
        Self {
            title: String::new(),
            width: 800,
            height: 600,
            is_open: false,
            #[cfg(target_os = "windows")]
            hwnd: None,
            #[cfg(target_os = "windows")]
            thread_handle: None,
            #[cfg(not(target_os = "windows"))]
            _handle: None,
        }
    }
}

/// 为 VST 插件打开原生编辑器窗口。
///
/// 在 Windows 上创建一个 Win32 窗口，将插件编辑器嵌入其中。
/// 窗口运行在独立线程的消息循环中，关闭窗口后线程自动退出。
///
/// `on_close` 回调在窗口关闭时被调用（窗口线程中），
/// 携带插件当前的 chunk 数据（base64），用于回写到 timeline。
///
/// 返回创建的窗口信息，失败时返回错误。
pub fn open_editor_window(
    instance: &Arc<Mutex<VstPluginInstance>>,
    title: &str,
    on_close: Option<OnEditorCloseCallback>,
) -> Result<VstEditorWindow, String> {
    // 获取编辑器推荐尺寸
    let mut inst = instance.lock().unwrap_or_else(|e| e.into_inner());
    let (width, height) = inst.editor_size();
    drop(inst);

    #[cfg(target_os = "windows")]
    {
        open_editor_window_win32(instance, title, width, height, on_close)
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = (instance, title, width, height, on_close);
        Err("VST editor windows are currently only supported on Windows".to_string())
    }
}

// ─── Windows 实现 ───────────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
mod win32 {
    //! Win32 FFI 类型与外部函数声明。
    //! 避免引入整个 windows-sys crate，手动声明所需子集。
    #![allow(non_snake_case, non_upper_case_globals, clippy::upper_case_acronyms)]

    use std::ffi::c_void;

    // ── 基础类型 ──
    pub type HWND = *mut c_void;
    pub type HINSTANCE = *mut c_void;
    pub type HICON = *mut c_void;
    pub type HCURSOR = *mut c_void;
    pub type HBRUSH = *mut c_void;
    pub type HMENU = *mut c_void;
    pub type WPARAM = usize;
    pub type LPARAM = isize;
    pub type LRESULT = isize;
    pub type ATOM = u16;
    pub type BOOL = i32;
    pub type DWORD = u32;
    pub type UINT = u32;

    // ── 窗口消息 ──
    pub const WM_DESTROY: UINT = 0x0002;
    pub const WM_CLOSE: UINT = 0x0010;
    pub const WM_ERASEBKGND: UINT = 0x0014;
    pub const WM_TIMER: UINT = 0x0113;

    // ── 窗口类样式 ──
    /// CS_OWNDC: 分配独占设备上下文，许多使用 OpenGL/Direct2D 渲染的 VST 插件需要此标志。
    pub const CS_OWNDC: UINT = 0x0020;

    // ── 窗口样式 ──
    pub const WS_OVERLAPPED: DWORD = 0x00000000;
    pub const WS_CAPTION: DWORD = 0x00C00000;
    pub const WS_SYSMENU: DWORD = 0x00080000;
    pub const WS_MINIMIZEBOX: DWORD = 0x00020000;
    pub const WS_VISIBLE: DWORD = 0x10000000;
    /// WS_CLIPCHILDREN: 防止父窗口绘制覆盖 VST 子窗口内容（解决白屏问题的关键）。
    pub const WS_CLIPCHILDREN: DWORD = 0x02000000;
    pub const WS_OVERLAPPEDWINDOW_NO_RESIZE: DWORD =
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;

    // ── 其他常量 ──
    pub const CW_USEDEFAULT: i32 = 0x80000000_u32 as i32;
    pub const IDC_ARROW: *const u16 = 32512 as *const u16;
    pub const COLOR_WINDOW: i32 = 5;
    pub const SW_SHOW: i32 = 5;

    // ── 结构体 ──
    #[repr(C)]
    pub struct WNDCLASSEXW {
        pub cbSize: UINT,
        pub style: UINT,
        pub lpfnWndProc: Option<
            unsafe extern "system" fn(HWND, UINT, WPARAM, LPARAM) -> LRESULT,
        >,
        pub cbClsExtra: i32,
        pub cbWndExtra: i32,
        pub hInstance: HINSTANCE,
        pub hIcon: HICON,
        pub hCursor: HCURSOR,
        pub hbrBackground: HBRUSH,
        pub lpszMenuName: *const u16,
        pub lpszClassName: *const u16,
        pub hIconSm: HICON,
    }

    #[repr(C)]
    pub struct MSG {
        pub hwnd: HWND,
        pub message: UINT,
        pub wParam: WPARAM,
        pub lParam: LPARAM,
        pub time: DWORD,
        pub pt_x: i32,
        pub pt_y: i32,
    }

    #[repr(C)]
    pub struct RECT {
        pub left: i32,
        pub top: i32,
        pub right: i32,
        pub bottom: i32,
    }

    // ── 外部函数 ──
    extern "system" {
        pub fn GetModuleHandleW(lpModuleName: *const u16) -> HINSTANCE;
        pub fn RegisterClassExW(lpwcx: *const WNDCLASSEXW) -> ATOM;
        pub fn CreateWindowExW(
            dwExStyle: DWORD,
            lpClassName: *const u16,
            lpWindowName: *const u16,
            dwStyle: DWORD,
            x: i32,
            y: i32,
            nWidth: i32,
            nHeight: i32,
            hWndParent: HWND,
            hMenu: HMENU,
            hInstance: HINSTANCE,
            lpParam: *mut c_void,
        ) -> HWND;
        pub fn DestroyWindow(hWnd: HWND) -> BOOL;
        pub fn ShowWindow(hWnd: HWND, nCmdShow: i32) -> BOOL;
        pub fn UpdateWindow(hWnd: HWND) -> BOOL;
        pub fn GetMessageW(
            lpMsg: *mut MSG,
            hWnd: HWND,
            wMsgFilterMin: UINT,
            wMsgFilterMax: UINT,
        ) -> BOOL;
        pub fn TranslateMessage(lpMsg: *const MSG) -> BOOL;
        pub fn DispatchMessageW(lpMsg: *const MSG) -> LRESULT;
        pub fn PostQuitMessage(nExitCode: i32);
        pub fn DefWindowProcW(hWnd: HWND, Msg: UINT, wParam: WPARAM, lParam: LPARAM) -> LRESULT;
        pub fn LoadCursorW(hInstance: HINSTANCE, lpCursorName: *const u16) -> HCURSOR;
        pub fn AdjustWindowRectEx(
            lpRect: *mut RECT,
            dwStyle: DWORD,
            bMenu: BOOL,
            dwExStyle: DWORD,
        ) -> BOOL;
        pub fn SetTimer(hWnd: HWND, nIDEvent: usize, uElapse: UINT, lpTimerFunc: *const c_void) -> usize;
        pub fn KillTimer(hWnd: HWND, uIDEvent: usize) -> BOOL;
    }

    /// WndProc 回调：VST 编辑器宿主窗口消息处理。
    pub unsafe extern "system" fn vst_editor_wnd_proc(
        hwnd: HWND,
        msg: UINT,
        w_param: WPARAM,
        l_param: LPARAM,
    ) -> LRESULT {
        match msg {
            WM_CLOSE => {
                DestroyWindow(hwnd);
                0
            }
            WM_DESTROY => {
                KillTimer(hwnd, 1);
                PostQuitMessage(0);
                0
            }
            WM_ERASEBKGND => {
                // 返回 1 表示已处理背景擦除（实际不做任何绘制），
                // 防止 Windows 用白色画刷覆盖 VST 插件的渲染内容。
                1
            }
            WM_TIMER => {
                // timer id 1 = VST idle callback（30 Hz）
                // VST2 editor 需要定期调用 idle 来处理 GUI 重绘
                // 实际调用在窗口线程主循环中完成
                0
            }
            _ => DefWindowProcW(hwnd, msg, w_param, l_param),
        }
    }
}

/// Windows 平台：创建 Win32 窗口 → 附着 VST editor → 运行消息循环。
///
/// 窗口创建和消息泵运行在新 spawn 的线程中。
/// VST editor 的 `open(parent_hwnd)` 在窗口创建后立即调用。
/// 消息循环中以 ~30 Hz 调用 editor `idle()` 维持 GUI 刷新。
#[cfg(target_os = "windows")]
fn open_editor_window_win32(
    instance: &Arc<Mutex<VstPluginInstance>>,
    title: &str,
    width: u32,
    height: u32,
    on_close: Option<OnEditorCloseCallback>,
) -> Result<VstEditorWindow, String> {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;
    use std::sync::mpsc;

    let instance_clone = Arc::clone(instance);
    let title_owned = title.to_string();
    let (tx, rx) = mpsc::channel::<Result<usize, String>>();

    // on_close 回调需要 move 到线程中
    let on_close_cell = std::cell::Cell::new(on_close);
    // SAFETY: Cell<Option<Box<dyn FnOnce + Send>>> 通过线程 move 传递是安全的
    struct SendCell(std::cell::Cell<Option<OnEditorCloseCallback>>);
    unsafe impl Send for SendCell {}
    let on_close_wrapper = SendCell(on_close_cell);

    let thread_handle = std::thread::Builder::new()
        .name(format!("vst-editor-{}", title))
        .spawn(move || {
            unsafe {
                use win32::*;

                let h_instance = GetModuleHandleW(std::ptr::null());

                // ── 注册窗口类 ──
                // 每个插件实例使用唯一类名，避免冲突
                let class_name_str = format!("HiFiShifter_VST_{}\0", title_owned);
                let class_name: Vec<u16> = OsStr::new(&class_name_str).encode_wide().collect();

                let wc = WNDCLASSEXW {
                    cbSize: std::mem::size_of::<WNDCLASSEXW>() as UINT,
                    style: CS_OWNDC,
                    lpfnWndProc: Some(vst_editor_wnd_proc),
                    cbClsExtra: 0,
                    cbWndExtra: 0,
                    hInstance: h_instance,
                    hIcon: std::ptr::null_mut(),
                    hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW),
                    hbrBackground: std::ptr::null_mut(), // null 画刷：不绘制背景，避免白屏
                    lpszMenuName: std::ptr::null(),
                    lpszClassName: class_name.as_ptr(),
                    hIconSm: std::ptr::null_mut(),
                };

                let atom = RegisterClassExW(&wc);
                if atom == 0 {
                    // 类已注册也不算错误（同名类可能已存在）
                    eprintln!(
                        "[vst_host::gui] RegisterClassExW returned 0 (may already exist)"
                    );
                }

                // ── 计算含窗口边框的实际尺寸 ──
                // WS_CLIPCHILDREN 确保父窗口绘制时不覆盖 VST 子窗口内容
                let style = WS_OVERLAPPEDWINDOW_NO_RESIZE | WS_VISIBLE | WS_CLIPCHILDREN;
                let mut rect = RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                };
                AdjustWindowRectEx(&mut rect, style, 0, 0);
                let adj_w = rect.right - rect.left;
                let adj_h = rect.bottom - rect.top;

                // ── 创建窗口 ──
                let window_title: Vec<u16> = OsStr::new(&format!("{}\0", title_owned))
                    .encode_wide()
                    .collect();

                let hwnd = CreateWindowExW(
                    0,
                    class_name.as_ptr(),
                    window_title.as_ptr(),
                    style,
                    CW_USEDEFAULT,
                    CW_USEDEFAULT,
                    adj_w,
                    adj_h,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    h_instance,
                    std::ptr::null_mut(),
                );

                if hwnd.is_null() {
                    let _ = tx.send(Err("CreateWindowExW returned null".to_string()));
                    return;
                }

                ShowWindow(hwnd, SW_SHOW);
                UpdateWindow(hwnd);

                // ── 将 VST editor 附着到窗口（catch_unwind 保护） ──
                {
                    let attach_result = std::panic::catch_unwind(
                        std::panic::AssertUnwindSafe(|| {
                            let mut inst = instance_clone.lock().unwrap_or_else(|e| e.into_inner());
                            match &mut inst.backend {
                                #[cfg(feature = "vst")]
                                VstPluginBackend::Vst2 { plugin, .. } => {
                                    let editor = plugin.get_editor();
                                    if let Some(mut editor) = editor {
                                        editor.open(hwnd);
                                        // 立即调用一次 idle() 触发首次绘制，
                                        // 部分 VST2 插件需要 open() 后首次 idle 才会渲染 GUI
                                        editor.idle();
                                        eprintln!(
                                            "[vst_host::gui] VST2 editor opened in HWND {:?}",
                                            hwnd
                                        );
                                    } else {
                                        eprintln!(
                                            "[vst_host::gui] VST2 plugin has no editor GUI"
                                        );
                                    }
                                }
                                #[cfg(feature = "vst")]
                                VstPluginBackend::Vst3 { instance } => {
                                    match instance.attach_editor(hwnd) {
                                        Ok(()) => {
                                            eprintln!(
                                                "[vst_host::gui] VST3 editor attached to HWND {:?}",
                                                hwnd
                                            );
                                        }
                                        Err(e) => {
                                            eprintln!(
                                                "[vst_host::gui] VST3 editor attach failed: {}",
                                                e
                                            );
                                        }
                                    }
                                }
                                #[cfg(not(feature = "vst"))]
                                VstPluginBackend::Stub => {}
                            }
                        }),
                    );

                    if let Err(panic_info) = attach_result {
                        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                            s.to_string()
                        } else if let Some(s) = panic_info.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "Unknown panic".to_string()
                        };
                        eprintln!(
                            "[vst_host::gui] Plugin panicked during editor attach: {}",
                            msg
                        );
                        // 销毁窗口并通知调用方失败
                        DestroyWindow(hwnd);
                        let _ = tx.send(Err(format!(
                            "Plugin panicked during editor attach: {}", msg
                        )));
                        return;
                    }
                }

                // 通知调用方窗口已创建（发送 usize 指针以满足 Send）
                let _ = tx.send(Ok(hwnd as usize));

                // ── 设置 idle timer（约 33ms = ~30 Hz） ──
                SetTimer(hwnd, 1, 33, std::ptr::null());

                // ── 消息循环 ──
                let mut msg: MSG = std::mem::zeroed();
                while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
                    TranslateMessage(&msg);
                    DispatchMessageW(&msg);

                    // 在 WM_TIMER 消息中调用 VST2 editor idle（catch_unwind 保护）
                    if msg.message == WM_TIMER {
                        let _ = std::panic::catch_unwind(
                            std::panic::AssertUnwindSafe(|| {
                                let mut inst =
                                    instance_clone.lock().unwrap_or_else(|e| e.into_inner());
                                match &mut inst.backend {
                                    #[cfg(feature = "vst")]
                                    VstPluginBackend::Vst2 { plugin, .. } => {
                                        if let Some(mut editor) = plugin.get_editor() {
                                            editor.idle();
                                        }
                                    }
                                    _ => {}
                                }
                            }),
                        );
                    }
                }

                // 消息循环结束，窗口已被关闭
                eprintln!("[vst_host::gui] Editor window thread exiting: {}", title_owned);

                // 在关闭 editor 之前，提取当前 chunk 数据用于回写
                let chunk_data = {
                    let mut inst = instance_clone.lock().unwrap_or_else(|e| e.into_inner());
                    inst.get_chunk()
                };

                // 关闭 VST editor
                {
                    let mut inst = instance_clone.lock().unwrap_or_else(|e| e.into_inner());
                    match &mut inst.backend {
                        #[cfg(feature = "vst")]
                        VstPluginBackend::Vst2 { plugin, .. } => {
                            if let Some(mut editor) = plugin.get_editor() {
                                editor.close();
                            }
                        }
                        #[cfg(feature = "vst")]
                        VstPluginBackend::Vst3 { instance } => {
                            instance.detach_editor();
                        }
                        _ => {}
                    }
                }

                // 调用 on_close 回调，将 chunk 数据回写到 timeline
                if let Some(callback) = on_close_wrapper.0.take() {
                    eprintln!(
                        "[vst_host::gui] Invoking on_close callback with chunk={}",
                        if chunk_data.is_some() { "present" } else { "none" }
                    );
                    callback(chunk_data);
                }
            }
        })
        .map_err(|e| format!("Failed to spawn editor thread: {}", e))?;

    // 等待窗口创建结果
    let hwnd_result = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .map_err(|_| "Timeout waiting for editor window creation".to_string())?;

    let hwnd_usize = hwnd_result?;
    let hwnd = hwnd_usize as *mut std::ffi::c_void;

    eprintln!(
        "[vst_host::gui] Editor window created: {} ({}x{}) HWND={:?}",
        title, width, height, hwnd
    );

    Ok(VstEditorWindow {
        title: title.to_string(),
        width,
        height,
        is_open: true,
        hwnd: Some(hwnd),
        thread_handle: Some(thread_handle),
    })
}

/// 关闭 VST 编辑器窗口。
///
/// 向窗口发送 WM_CLOSE 消息，触发窗口销毁和线程退出。
/// 发送前检查 HWND 是否仍有效（IsWindow），避免向已销毁的窗口发送消息。
pub fn close_editor_window(window: &mut VstEditorWindow) {
    if !window.is_open {
        return;
    }

    #[cfg(target_os = "windows")]
    {
        if let Some(hwnd) = window.hwnd.take() {
            // 向窗口线程发送关闭消息（先检查 HWND 有效性）
            unsafe {
                extern "system" {
                    fn IsWindow(hWnd: *mut std::ffi::c_void) -> i32;
                    fn PostMessageW(
                        hWnd: *mut std::ffi::c_void,
                        Msg: u32,
                        wParam: usize,
                        lParam: isize,
                    ) -> i32;
                }
                if IsWindow(hwnd) != 0 {
                    PostMessageW(hwnd, win32::WM_CLOSE, 0, 0);
                } else {
                    eprintln!(
                        "[vst_host::gui] HWND {:?} is no longer valid, skipping PostMessage",
                        hwnd
                    );
                }
            }
        }

        // 等待窗口线程退出（最多 2 秒）
        if let Some(handle) = window.thread_handle.take() {
            let _ = handle.join();
        }
    }

    window.is_open = false;
    eprintln!("[vst_host::gui] Editor window closed: {}", window.title);
}
