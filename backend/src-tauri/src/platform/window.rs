//! 主窗口的形态相关设置。
//!
//! 【为什么单独一个模块】桌面窗口有一整套"用户可以摆布它"的语义：恢复上次的位置与
//! 尺寸、记住最大化/全屏、多显示器可见性校验、可缩放。移动端一条都不成立 ——
//! 窗口由系统给、尺寸随旋转变化、位置不存在。这些差异如果散落在 `lib.rs` 的 setup
//! 里，就会变成一串 `if !mobile` 把 300 行窗口恢复逻辑包起来；收在这里之后，
//! `lib.rs` 只需要问一句 [`supports_window_state`]。

use tauri::Manager;

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
pub fn prepare_main_window(app: &tauri::AppHandle) {
    if !crate::platform::is_mobile() {
        // 桌面：几何由 `tauri.conf.json` 与 setup 里的恢复逻辑决定，这里不碰。
        return;
    }
    let Some(window) = app.get_webview_window("main") else {
        log::warn!("[window] main webview not found during mobile setup");
        return;
    };
    // 移动端的窗口应当**始终**是整屏：配置里若残留桌面尺寸（例如 1200x800），
    // 某些设备会按逻辑像素摆放它，右侧与底部留下一圈黑边。这里显式改回全屏。
    if let Err(error) = window.set_fullscreen(true) {
        log::warn!("[window] set_fullscreen(true) failed: {error}");
    }
    if let Err(error) = window.set_resizable(false) {
        log::warn!("[window] set_resizable(false) failed: {error}");
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
