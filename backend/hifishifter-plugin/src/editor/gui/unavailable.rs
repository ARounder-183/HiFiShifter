//! "没有 GUI"这条路的唯一实现。
//!
//! 【为什么它不是错误分支】拿不到 GTK / WebKit 是**可预期的环境事实**（宿主没装依赖、
//! 无显示服务、Wayland 会话下无可嵌入 surface）。上层要的是"宿主显示空面板 + 日志里
//! 有一行可搜到的原因"，而不是让 `attached` 抛一个没人看的字符串。
//!
//! 【为什么要有一个真句柄】`view.rs` 在 `attached` 成功后会在 `onSize`/`focus` 上
//! 调用它；返回 `None` 会让那几处再各加一个判空分支 —— 而它们本来只需要"什么都不做"。

#![cfg_attr(windows, allow(dead_code))]

use super::port::{EditorHost, GuiFidelity};

/// 明确"本平台没有内嵌编辑器"的句柄。
pub(crate) struct UnavailableHost {
    reason: &'static str,
}

impl UnavailableHost {
    pub(crate) fn new(reason: &'static str) -> Self {
        Self { reason }
    }
}

impl EditorHost for UnavailableHost {
    fn fidelity(&self) -> GuiFidelity {
        GuiFidelity::Unavailable(self.reason)
    }

    fn resize(&self, _width: i32, _height: i32) -> Result<(), String> {
        Ok(())
    }

    fn focus(&self) -> Result<(), String> {
        Err(self.reason.to_owned())
    }

    fn forward_edit_key(&self, _key: char, _mods: i16, _down: bool) -> bool {
        false
    }

    fn host_tick(&self) {}

    fn close(&self) {}
}

/// 兜底原因（既不是 Windows 也不是 unix 的目标）。
#[cfg_attr(any(windows, unix), allow(dead_code))]
pub(crate) fn reason() -> &'static str {
    "the embedded editor is not implemented for this target"
}
