//! 内嵌编辑器视图：**Policy / Port / Primitive** 三层，与 `host_key_relay` 同构。
//!
//! `client.rs`    —— Policy：校验、分流、回执泵、预算、撤销门、事件队列。**无 `cfg`**。
//! `port.rs`      —— Port：两个 trait（`EditorPort` / `EditorHost`）+ `GuiFidelity`。**无 `cfg`**。
//! `protocol.rs`  —— 纯函数：消息形状、资产路径、MIME、导航白名单。**无 `cfg`**。
//! `windows.rs`   —— 原语：子 HWND + WebView2。
//! `macos.rs`     —— 原语：子 NSView + WKWebView。
//! `linux.rs`     —— 原语：GtkPlug + WebKitGTK（必要时自建 GTK 线程）。
//! `unavailable.rs` —— 明确"没有 GUI"的句柄（缺 GTK/WebKit 时宿主看到空面板 + 一行日志）。
//!
//! 【为什么入口只有两个函数】上层（`view.rs` / `editor::shutdown`）只该知道
//! "建一个视图"和"卸载前收尾"；平台的差异（异步 COM、GTK 线程、scheme handler）
//! 全部留在原语里。

mod client;
mod port;
mod protocol;
mod unavailable;

#[cfg(windows)]
mod windows;

#[cfg(target_os = "macos")]
mod macos;

#[cfg(all(unix, not(target_os = "macos")))]
mod linux;

use std::rc::Rc;

pub(crate) use port::{EditorHost, GuiFidelity, NativeWindow};

/// 建一个内嵌视图，并把 JS 消息接到 Policy。
///
/// 【为什么返回 `Result` 而不是 `Option`】"父窗口不可用""宿主不在 UI 线程"这类失败
/// 必须能被日志说清楚 —— 宿主只看到 `K_RESULT_FALSE` 加一块空面板，日志是唯一线索。
pub(crate) fn create_for(
    parent: NativeWindow,
    width: i32,
    height: i32,
    // 宿主给的是不是"可嵌入的父窗口"：Linux 上 `X11EmbedWindowID` 才能内嵌，
    // `WaylandSurfaceID` 只能开浮窗（见 `view.rs` 的 `platform()`）。Windows/macOS 恒为 `true`。
    embedded: bool,
    // 宿主的 `IPlugFrame*`（`0` = 还没给）。Linux 用它查 `Linux::IRunLoop`；
    // 另外两个平台用不上 —— 它们的主循环是现成的（`NSApplication` / 宿主自己的消息泵）。
    frame: usize,
    link: std::sync::Arc<crate::editor::routing::EditorLink>,
) -> Result<Rc<dyn EditorHost>, String> {
    #[cfg(windows)]
    {
        let _ = (embedded, frame);
        Ok(Rc::new(windows::WindowsHost::attach(
            parent, width, height, link,
        )?))
    }
    #[cfg(target_os = "macos")]
    {
        let _ = embedded;
        Ok(Rc::new(macos::MacHost::attach(
            parent, width, height, link,
        )?))
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        linux::create(parent, width, height, embedded, frame, link)
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = (parent, width, height, embedded, frame, link);
        Err(unavailable::reason().to_owned())
    }
}

/// 卸载前收尾。
///
/// 【为什么返回 `bool`】`ExitDll` 只能在"确实拆干净"时返回成功：留着一个指向已卸载
/// 代码段的窗口类/线程，下一次窗口消息就是野跳转。
pub(crate) fn shutdown() -> bool {
    #[cfg(windows)]
    {
        windows::shutdown()
    }
    #[cfg(target_os = "macos")]
    {
        macos::shutdown()
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        linux::shutdown()
    }
    #[cfg(not(any(windows, unix)))]
    {
        true
    }
}
