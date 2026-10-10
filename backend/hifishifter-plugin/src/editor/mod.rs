//! 插件内原GUI宿主；原生视图不启动独立app，也不接管宿主消息循环。
mod browser_files;
pub(crate) mod commands;
pub(crate) mod connection;
mod events;
mod host_clipboard;
pub(crate) mod host_edit;
pub(crate) mod host_split;
mod notebook;
pub(crate) mod parameter_atlas;
mod plugin_diagnostics;
pub(crate) mod private_groups;
pub(crate) mod resources;
pub(crate) mod routing;
pub(crate) mod session;
mod view;
#[cfg(windows)]
mod webview;
pub(crate) mod workspace;
pub(crate) use view::create_view_with_link;

/// 卸载前撤销本模块注册的窗口类；尚有窗口时明确拒绝正常卸载。
pub(crate) fn shutdown() -> bool {
    #[cfg(windows)]
    {
        webview::shutdown()
    }
    #[cfg(not(windows))]
    {
        true
    }
}
