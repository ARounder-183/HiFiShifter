//! 插件内原GUI宿主；原生视图不启动独立app，也不接管宿主消息循环。
mod view;
pub(crate) mod routing;
pub(crate) mod connection;
pub(crate) mod session;
mod commands;
mod events;
pub(crate) mod resources;
#[cfg(windows)]
mod webview;
pub(crate) use view::create_view_with_link;

/// 卸载前撤销本模块注册的窗口类；尚有窗口时明确拒绝正常卸载。
pub(crate) fn shutdown() -> bool {
    #[cfg(windows)]
    { webview::shutdown() }
    #[cfg(not(windows))]
    { true }
}
