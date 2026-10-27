//! 没有"把键交回宿主"这条路的平台。
//!
//! 【哪些平台落在这里】Wayland（VST3 3.7.x 没有 Wayland 视图类型，也没有向别的客户端
//! 注入按键的公开协议），以及**视图还在但让路接缝未接**的组合 —— macOS / Linux 的
//! 编辑器视图已经落地（`gui/macos.rs` / `gui/linux.rs`），而让路只在 Windows 有
//! 原生实现；Linux 上 Fidelity 是 `Unavailable`（经 `escape_for_platform` 的
//! `not(windows)` 分支），不是"没有视图"。
//!
//! 【为什么 Unavailable 也要"接手"】回一句"我不接手"会让前端回落那句"当前宿主没有提供
//! 这个接口" —— 那是把用户往死路上指。接手之后什么都不做、只留一行日志：本平台没有
//! 让路原语，这是能力事实，不是操作失败。
//!
//! 【新平台怎么接进来】在 `mod.rs` 的 `escape_for_platform` 里加一条 `cfg` 分支，
//! 写一个 `HostKeyEscape` 实现，自报 `Fidelity::PassThrough`（能事前让路，如
//! macOS 的响应链）或 `Fidelity::Synthetic`（只能事后补一次，如 X11 的合成事件）。
//! **Policy 一行都不用改** —— 这就是把"怎么让"和"什么时候让"分开的全部收益。

use super::{EscapeError, Fidelity, HostKeyEscape, Stroke, TargetKind};

/// 【为什么在 Windows 上也算"未构造"却仍保留】把它 cfg 掉能让警告消失，代价是这份
/// 文件在 Windows 上**不再被编译** —— 一个只在别的平台编译的文件，等真到那个平台时
/// 往往已经腐烂。留着它编译，比一条干净的构建日志值钱。
#[allow(dead_code)]
pub(crate) struct UnavailableEscape;

impl HostKeyEscape for UnavailableEscape {
    fn fidelity(&self) -> Fidelity {
        Fidelity::Unavailable
    }

    fn escape(&mut self, _window: usize, _stroke: &Stroke) -> Result<TargetKind, EscapeError> {
        // 状态机在 Fidelity::Unavailable 下根本不会走到这里（它直接记一行日志）；保留一个
        // 明确的失败值，是为了让"忘了看 Fidelity"的新调用方也拿不到假成功。
        Err(EscapeError::NoHostWindow)
    }

    fn focus_host(&mut self, _window: usize, _previous: Option<usize>) -> Result<(), EscapeError> {
        // 连"宿主窗口在哪"都不知道，就没有焦点可交还。
        Err(EscapeError::NoHostWindow)
    }
}
