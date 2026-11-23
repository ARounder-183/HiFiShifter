//! 主窗口的形态相关设置。
//!
//! 【为什么单独一个模块】桌面窗口有一整套"用户可以摆布它"的语义：恢复上次的位置与
//! 尺寸、记住最大化/全屏、多显示器可见性校验、可缩放。移动端一条都不成立 ——
//! 窗口由系统给、尺寸随旋转变化、位置不存在。这些差异如果散落在 `lib.rs` 的 setup
//! 里，就会变成一串 `if !mobile` 把 300 行窗口恢复逻辑包起来；收在这里之后，
//! `lib.rs` 只需要问一句 [`supports_window_state`]。

/// 主窗口是否需要"恢复上次的几何状态"。
///
/// 【为什么移动端必须是 false】移动端的窗口几何由系统决定，`set_size` / `set_position`
/// 要么被忽略、要么在多窗口（分屏）状态下造成奇怪的行为。更重要的是：恢复逻辑里
/// 有一段"与显示器求交集"的算法，移动端没有可枚举的显示器 —— 它会退化成"居中"，
/// 也就是每次冷启动都做一次无意义的窗口操作。
pub fn supports_window_state() -> bool {
    !crate::platform::is_mobile()
}

/// 窗口是否可以被用户缩放（决定前端是否渲染分隔条与拖动边缘）。
pub fn resizable() -> bool {
    !crate::platform::is_mobile()
}

/// 启动期的窗口准备。
///
/// 【为什么移动分支几乎什么都不做】`tauri` 在移动目标上**根本不编译** `set_fullscreen`
/// 与 `maximize`（它们是 `#[cfg(desktop)]`）—— "移动端没有全屏这个概念"由上游的 API
/// 面本身表达，不需要我们再补一次。窗口尺寸与位置同样由系统决定，`set_resizable` 在
/// 那里也没有意义。这里留一行日志，让"窗口几何归谁管"在用户报障时有据可查。
pub fn prepare_main_window(_app: &tauri::AppHandle) {
    if crate::platform::is_mobile() {
        log::debug!("[window] mobile surface: window geometry is owned by the system");
    }
}

/// 应用从窗口状态里恢复出来的"模式位"（全屏 / 最大化 / 普通）。
///
/// 【为什么必须收进收敛层】`set_fullscreen` 与 `maximize` 在移动目标上**不存在**，
/// 而 `is_fullscreen` 却存在 —— 上游没有提供"移动端调用会被忽略"的退路，代码必须在
/// **编译期**分叉。分叉留在 `lib.rs` 里就是业务代码里的平台判断（见 `platform` 模块
/// 的三条纪律），因此收在这里。
pub fn apply_window_mode(window: &tauri::WebviewWindow, fullscreen: bool, maximized: bool) {
    #[cfg(desktop)]
    {
        if fullscreen {
            let _ = window.set_fullscreen(true);
        } else if maximized {
            let _ = window.maximize();
        } else {
            let _ = window.set_fullscreen(false);
        }
    }
    #[cfg(not(desktop))]
    {
        // 【为什么移动端是空操作】移动端的窗口永远整屏、由系统摆放，没有"最大化"这个
        // 状态。调用点照常把恢复出来的布尔值传进来，这里只是没有可做的动作 —— 这样
        // 调用点就不必知道移动端不存在这些 API。
        let _ = (window, fullscreen, maximized);
    }
}
// ── 应用菜单与 WebView 快捷键 ────────────────────────────────────────────────

/// 是否由应用自建菜单（macOS 需要，其余平台由 Tauri 决定）。
pub fn uses_custom_app_menu() -> bool {
    #[cfg(target_os = "macos")]
    {
        true
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// 安装应用菜单。
///
/// 【为什么收进收敛层】`macos_menu` 模块本身是 `#[cfg(target_os = "macos")]` 的，
/// 调用点若留在 `lib.rs` 里，业务代码就多出一处平台判断。收敛层内部判断，
/// 调用点因此对平台无感知 —— 这也让"新增一个平台"只需要改这一处。
pub fn install_app_menu(app: &tauri::App) -> tauri::Result<()> {
    #[cfg(target_os = "macos")]
    {
        let menu = crate::macos_menu::build(app.handle())?;
        app.set_menu(menu)?;
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = app;
    }
    Ok(())
}

/// 主 WebView 的宿主级设置。
///
/// 【为什么必须有这一步】WebView2 把 Ctrl+Plus/Minus、Ctrl+F、F5 这类按键当作**浏览器
/// 快捷键**处理，它们不会作为可取消的 keydown 交给页面 —— 于是参数线微调的
/// `Ctrl+=` / `Ctrl+-` 会被页面缩放吃掉。wry 提供
/// `with_browser_accelerator_keys(false)`，但 Tauri 2 的 builder 未透传，
/// 只能在窗口创建后经 `with_webview` 设置。
pub fn configure_main_webview(window: &tauri::WebviewWindow) {
    #[cfg(target_os = "windows")]
    {
        crate::webview2_accelerators::disable_browser_accelerator_keys(window);
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = window;
    }
}
