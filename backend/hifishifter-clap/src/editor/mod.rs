//! CLAP 编辑器窗口：`clap.gui` 的真实宿主视图。
//!
//! 【为什么独立一层】VST3 的 `editor/gui` 与 ARA 会话、219 条命令、REAPER 原生
//! 扩展绑在一起；CLAP 阶段 1 没有那套后端。这里只做三件事：建原生视图、加载
//! 同一份前端产物、把页面消息交给 `protocol` 回执。

pub mod protocol;
pub mod timeline_host;

#[cfg(windows)]
pub mod windows;

/// 平台是否提供内嵌视图。非 Windows 上如实返回 false（不假装）。
pub const fn is_supported() -> bool {
    cfg!(windows)
}

/// 平台不可用时给宿主/日志的一句话原因。
pub const fn unavailable_reason() -> &'static str {
    "embedded WebView is available on Windows only in the CLAP Capture-only shell"
}
