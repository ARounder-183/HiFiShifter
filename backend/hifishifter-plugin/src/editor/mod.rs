//! 插件内原GUI宿主；原生视图不启动独立app，也不接管宿主消息循环。
mod browser_files;
pub(crate) mod commands;
pub(crate) mod connection;
pub(crate) mod errors;
mod events;
mod formant;
pub(crate) mod media_identity;
mod notebook;
pub(crate) mod parameter_atlas;
mod plugin_diagnostics;
pub(crate) mod private_groups;
pub(crate) mod resources;
pub(crate) mod routing;
pub(crate) mod session;
mod temp_sweep;
pub(crate) mod capture_edits;
mod view;
#[cfg(windows)]
mod webview;
pub(crate) mod workspace;
pub(crate) use errors::classify_error;
pub(crate) use view::create_view_with_link;

#[cfg(test)]
mod tests {
    use super::classify_error;

    /// 【为什么钉死】前端据 `terminal` **永久停止轮询**。一旦这里把终态错误判成
    /// `transient`，那个"把终态错误答 19,056 次"的缺陷就会以另一种形式回来。
    #[test]
    fn session_teardown_errors_are_terminal() {
        for error in [
            "document closed",
            "FX processor closed",
            "editor view closed or unknown",
            "editor view identity mismatch",
            "editor route changed while request queued",
            "editor route belongs to another document",
            "editor lease revoked",
            "plugin instance closed",
            "plug-in instance is not bound to an ARA document",
        ] {
            assert_eq!(
                classify_error(error),
                "terminal",
                "{error} must be terminal"
            );
        }
    }

    /// 【为什么钉死】"尚未连接"与"宿主拒绝"这类是**暂时**的：编辑器视图在组件
    /// 创建与绑定之间必然经历它，判成终态会让界面刚打开就宣称"会话已结束"。
    #[test]
    fn recoverable_conditions_stay_transient() {
        for error in [
            "FX editor not connected to its processor yet",
            "host clip changed before GUI commit; refresh required",
            "host does not provide ARA playback control",
            "invalid editor arguments",
        ] {
            assert_eq!(
                classify_error(error),
                "transient",
                "{error} must be transient"
            );
        }
    }
}

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
