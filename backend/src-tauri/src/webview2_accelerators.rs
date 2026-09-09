//! webview2_accelerators.rs — 禁用 WebView2 的浏览器级快捷键（仅 Windows）。
//!
//! WebView2 默认把一批按键当作「浏览器快捷键」处理（
//! `ICoreWebView2Settings3::AreBrowserAcceleratorKeysEnabled`，默认 TRUE），
//! 其中包括 Ctrl+Plus / Ctrl+Minus（页面缩放）、Ctrl+F（查找）、Ctrl+P
//! （打印）、Ctrl+R / F5（刷新）等。这类按键不会作为可取消的 keydown
//! 交给页面 —— 浏览器行为直接生效，会抢占应用自身的快捷键（例如参数线
//! 微调的 Ctrl+= / Ctrl+- 会被页面缩放吃掉）。
//!
//! wry 提供 `with_browser_accelerator_keys(false)`，但 Tauri 2 的
//! WebviewWindowBuilder 未透传该选项，因此在窗口创建后经 `with_webview`
//! 取出 WebView2 控制器，把该设置置为 FALSE（与 wry 的做法一致，且在
//! 首次导航前生效）。禁用后这些按键作为普通输入交给页面，由前端的
//! 全局 keydown 路由（useKeybindings）正常处理。

use tauri::WebviewWindow;
use webview2_com::Microsoft::Web::WebView2::Win32::{ICoreWebView2Settings3};
use windows_core_webview2::Interface;

/// 禁用主窗口 WebView2 的浏览器级快捷键。失败仅记录警告，不影响启动。
pub fn disable_browser_accelerator_keys(win: &WebviewWindow) {
    let dispatched = win.with_webview(|webview| {
        let outcome = (|| -> windows_core_webview2::Result<()> {
            let controller = webview.controller();
            // SAFETY: COM 调用均在 WebView2 控制器存活期内执行；
            // with_webview 的闭包在 UI 线程上运行，满足 WebView2 线程模型。
            unsafe {
                let core = controller.CoreWebView2()?;
                let settings = core.Settings()?;
                let settings3: ICoreWebView2Settings3 = settings.cast()?;
                settings3.SetAreBrowserAcceleratorKeysEnabled(false)?;
            }
            Ok(())
        })();
        if let Err(err) = outcome {
            log::warn!("[webview2] failed to disable browser accelerator keys: {err}");
        } else {
            log::debug!(
                "[webview2] browser accelerator keys disabled (Ctrl+/- zoom, Ctrl+F, F5 etc. are handed back to the app)"
            );
        }
    });
    if let Err(err) = dispatched {
        log::warn!("[webview2] with_webview dispatch failed: {err}");
    }
}
