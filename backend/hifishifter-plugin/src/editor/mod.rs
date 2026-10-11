//! 插件内原GUI宿主；原生视图不启动独立app，也不接管宿主消息循环。
mod browser_files;
pub(crate) mod commands;
pub(crate) mod connection;
mod events;
mod formant;
mod notebook;
pub(crate) mod parameter_atlas;
mod plugin_diagnostics;
pub(crate) mod private_groups;
pub(crate) mod resources;
pub(crate) mod routing;
pub(crate) mod session;
mod temp_sweep;
mod view;
#[cfg(windows)]
mod webview;
pub(crate) mod workspace;
pub(crate) use view::create_view_with_link;

/// 命令失败的**机器可读分类**，随 IPC 回执一起发给前端。
///
/// 【为什么需要它】此前前端只能拿到一个裸字符串。于是它无法区分"这次失败是暂时的、
/// 稍后会好"与"这个编辑会话已经**永久**结束、再问一万次也一样" —— 用户报障的
/// Cubase 场景正是后者：文档 `close()` 是终态（`alive` 原子量一旦 false 永不再活），
/// 而页面仍在每 250 ms 轮询，把一个终态错误放大成 19,056 条日志。
/// 有了分类，前端可以**停止**轮询并如实显示"会话已结束"，而不是无限重试。
///
/// 【为什么用"包含"而不是全等】这些串来自多个模块（`render::extension` 的
/// `editor_document`、`editor::routing` 的 `EditorLink`、`editor::session` 的
/// worker），措辞略有差异但语义同类。用子串匹配能覆盖它们而不必让每个产生点都改。
/// 这是**协议级**的固定映射，不是"随手解析错误文本"。
pub(crate) fn classify_error(error: &str) -> &'static str {
    /// 会话已终结：再重试也不会成功，前端应当停止轮询。
    const TERMINAL: &[&str] = &[
        "document closed",
        "FX processor closed",
        "editor view closed or unknown",
        "editor view identity mismatch",
        "editor route changed while request queued",
        "editor route belongs to another document",
        "editor lease revoked",
        "plugin instance closed",
    ];
    if TERMINAL.iter().any(|needle| error.contains(needle)) {
        "terminal"
    } else {
        "transient"
    }
}

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
        ] {
            assert_eq!(classify_error(error), "terminal", "{error} must be terminal");
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
            assert_eq!(classify_error(error), "transient", "{error} must be transient");
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
