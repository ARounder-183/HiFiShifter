//! 音频会话（移动端的系统级音频生命周期）。
//!
//! ## 为什么这不是"锦上添花"
//!
//! cpal 0.18 的 iOS 后端**从不设置 `AVAudioSession` 的类别**（整个 crate 只有一处
//! `setActive(true)`，见 `cpal-0.18.2/src/host/coreaudio/ios/session_event_manager.rs`）。
//! 系统因此给应用的是默认类别 `.soloAmbient` —— 而 `.soloAmbient` 会被**静音键**
//! 与**锁屏**直接掐掉。症状是"插着耳机、音量拉满，锁屏之后什么都没了"，且没有任何
//! 报错。这不是可选的打磨项：不设类别，移动端就不算能出声。
//!
//! ## 本模块当前的边界
//!
//! 桌面分支是空操作（桌面没有"音频会话"这个系统概念）。移动分支需要
//! `objc2-av-foundation`（iOS）与 `AudioManager`（Android）两条原生通道，而它们
//! **必须先在真机上验证**（见设计文档 §P0-2 与 §P1-4 的门）。在那之前这里只留
//! [`report_pending`]，让"为什么静音键会杀掉声音"在日志里有一行明确记录，而不是
//! 变成一个只能靠猜的现象。

/// 音频会话的用途。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Usage {
    /// 只播放：iOS 应设 `AVAudioSessionCategoryPlayback`。
    Playback,
    /// 播放 + 录音：iOS 应设 `AVAudioSessionCategoryPlayAndRecord`（并处理路由变化，
    /// 否则录音结束后声音会从听筒出）。
    PlayAndRecord,
}

/// 启动期配置。
pub fn configure(app: &tauri::AppHandle) {
    if !crate::platform::is_mobile() {
        return;
    }
    report_pending(app, Usage::Playback);
}

/// 切换到另一个用途（开始 / 结束录音时调用）。
pub fn set_usage(app: &tauri::AppHandle, usage: Usage) {
    if !crate::platform::is_mobile() {
        return;
    }
    report_pending(app, usage);
}

/// 把"尚未接入原生会话"这件事写进日志与诊断。
///
/// 【为什么不是 `todo!()` 也不是静默 no-op】前者会让移动端一启动就崩；后者会让
/// "锁屏后没声音"永远没有线索。一行 warn 是这两者之间唯一诚实的选择。
fn report_pending(app: &tauri::AppHandle, usage: Usage) {
    let os = crate::platform::os();
    log::warn!(
        "[audio_session] native audio session not configured yet on {os:?} (usage={usage:?}); \
         iOS defaults to .soloAmbient, so the silent switch and screen lock will mute playback. \
         See docs/plans/2026-11-24-mobile-adaptation-design.md §5.4 / P1-4."
    );
    // 诊断包里也要看得到：用户报"锁屏后没声音"时，这一行是唯一能证明
    // "类别从未被设置"的证据。
    let _ = app;
}

/// 系统是否会把"被来电 / 其它应用打断"告诉我们。
pub fn focus_events_available() -> bool {
    false
}
