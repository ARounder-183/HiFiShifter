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

pub(crate) mod client;
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

// 宿主主线程上的周期工作由 `editor::host_main_thread_tick` 统一暴露：VST3 挂在
// REAPER 的 `plugin_register("timer")`、GTK 定时器上，CLAP 挂在 `clap.timer-support` 上，
// 但每次节拍要做的事只有 `client::host_heartbeat` 这一份。
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
        let _ = (embedded, frame);
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

/// 一个已建立的嵌入式编辑器视图（**格式无关**的句柄）。
///
/// 【为什么要有这一层】VST3 的 `IPlugView` 与 CLAP 的 `clap_plugin_gui` 是两套 ABI，
/// 但它们对视图的要求完全相同：一个父窗口、一次尺寸、一次焦点、一次关闭。把这三件事
/// 收成一个句柄，两个外壳就只剩"把父窗口句柄翻译成 `usize`"这一行差异 —— 而 WebView2
/// 的创建、页面重建、原生文件对话框、键盘让路、20 ms 泵都只有这一份实现。
///
/// 【为什么不是 `Box<dyn EditorHost>`】句柄在宿主 UI 线程上创建与使用，`Rc` 已经足够；
/// 给它 `Send` 会逼三个平台把真正的线程亲和性藏起来（见 `port.rs` 的说明）。
pub struct EditorView {
    host: Rc<dyn EditorHost>,
}

impl EditorView {
    /// 视图尺寸变化（宿主拖拽边框、DPI 变化都走这里）。
    pub fn resize(&self, width: i32, height: i32) -> Result<(), String> {
        self.host.resize(width, height)
    }

    /// 宿主明确把焦点交给本视图。
    pub fn focus(&self) -> Result<(), String> {
        self.host.focus()
    }

    /// 平台自报的嵌入形态（诊断用稳定短名）。
    pub fn fidelity(&self) -> &'static str {
        self.host.fidelity().as_str()
    }

    /// 拆掉原生侧的一切。可重入。
    pub fn close(&self) {
        self.host.close();
    }
}

/// 本目标是否提供内嵌编辑器。
pub const fn available() -> bool {
    cfg!(any(windows, unix))
}

/// 平台不可用时给宿主/日志的一句话原因。
pub fn unavailable_reason() -> &'static str {
    unavailable::reason()
}

/// 在宿主给的父窗口里建一个内嵌视图（**两个格式唯一的建视图入口**）。
///
/// `parent` 是平台父句柄的不透明整数：Windows `HWND`、macOS `NSView*`、Linux X11 Window。
/// `embedded` / `frame` 只对 Linux 有意义（见 [`create_for`]）；CLAP 没有 `IPlugFrame`，
/// 传 `0` 表示"没有宿主 run loop"，平台会走自己的线程。
pub fn create_view(
    parent: usize,
    width: i32,
    height: i32,
    link: std::sync::Arc<crate::editor::routing::EditorLink>,
) -> Result<EditorView, String> {
    Ok(EditorView {
        host: create_for(parent, width, height, true, 0, link)?,
    })
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
