//! 传输会话（TransportSession）：一次“播放”从开始到停止的显式生命周期。
//!
//! 旧代码用四个互相独立的信号推断“在不在播”：kPlaying、位置有没有动、输入内容
//! 有没有变、prefetch 播放位。停播后宿主仍送块时，这四个信号可以同时给出“在走”的
//! 假象。这里把它们收敛成一个状态机：只有 Starting（有界起播窗口）或 Rolling
//! 允许时间前进；Paused 与 Stopped 一律冻结。
//!
//! 【本轮的职责收缩】它只回答“这是不是同一次播放、何时冻结”。坐标域由
//! `HostClock` 回答，离线/冻结预览由 `ProcessingPass` 回答。三者顺序固定：
//! ProcessingPass -> HostClock -> TransportSession。
//!
//! 【为什么 Starting -> Rolling 不递增 epoch】VEGAS Event FX 的域切换恰好发生在
//! kPlaying 上升沿：pre-roll 的绝对域块先打开 Starting 窗口，下一块 kPlaying 置位
//! 并切到相对域。若这里把上升沿判成新会话，域切换就永远没有机会执行 —— 现场
//! 那 0.68 秒绝对域首段正是这样留下的。

/// 传输会话的相位。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SessionPhase {
    #[default]
    Idle,
    /// 已经收到块，但宿主的 kPlaying 还没置位；只允许一个有界的起点窗口。
    Starting,
    /// 宿主明确在播放。
    Rolling,
    /// kPlaying 已落，但捕获仍 armed：可以续录，位置必须冻结。
    Paused,
    /// 用户停止 / 离线 pass 结束。
    Stopped,
    /// 离线 pass 的本地游标（由 ProcessingPass 驱动，保留编码兼容）。
    OfflineRolling,
}

impl SessionPhase {
    pub fn code(self) -> u8 {
        match self {
            Self::Idle => 0,
            Self::Starting => 1,
            Self::Rolling => 2,
            Self::Paused => 3,
            Self::Stopped => 4,
            Self::OfflineRolling => 5,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Idle),
            1 => Some(Self::Starting),
            2 => Some(Self::Rolling),
            3 => Some(Self::Paused),
            4 => Some(Self::Stopped),
            5 => Some(Self::OfflineRolling),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub fn is_advancing(self) -> bool {
        matches!(self, Self::Starting | Self::Rolling | Self::OfflineRolling)
    }

    #[allow(dead_code)]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Starting => "starting",
            Self::Rolling => "rolling",
            Self::Paused => "paused",
            Self::Stopped => "stopped",
            Self::OfflineRolling => "offline",
        }
    }
}

/// 一个传输会话的纯状态。
#[derive(Clone, Copy, Debug)]
pub struct TransportSession {
    pub epoch: u64,
    pub phase: SessionPhase,
    pub session_frames: u64,
    pub last_playing: bool,
    pub max_start_window_frames: u64,
}

impl Default for TransportSession {
    fn default() -> Self {
        Self {
            epoch: 0,
            phase: SessionPhase::Idle,
            session_frames: 0,
            last_playing: false,
            max_start_window_frames: 48_000,
        }
    }
}

/// 本块看到的传输事实。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionInput {
    pub playing: bool,
    pub playing_rising: bool,
    pub recording_rising: bool,
    pub host_reset: bool,
    pub offline: bool,
    pub frames: u32,
}

/// 状态机结论。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionDecision {
    pub phase: SessionPhase,
    pub new_epoch: bool,
    /// 本块是否有权推进位置 / 写捕获。
    pub transport_active: bool,
    /// 起点窗口还开着：窗口内的块可以按块长前进，即使 kPlaying 还是 0。
    pub start_window_open: bool,
}

impl TransportSession {
    #[cfg(test)]
    pub fn with_rate(rate: f64) -> Self {
        let mut session = Self::default();
        if rate.is_finite() && rate > 0.0 {
            session.max_start_window_frames = (rate * 1.0).round().max(1.0) as u64;
        }
        session
    }

    /// 前进一步。host_reset = 宿主位置倒退或跳变；它是新会话的强证据之一。
    pub fn observe(&mut self, input: SessionInput) -> SessionDecision {
        let mut new_epoch = false;
        let mut transport_active = false;
        let mut start_window_open = false;

        if input.offline {
            if self.phase != SessionPhase::OfflineRolling {
                self.epoch = self.epoch.saturating_add(1);
                self.phase = SessionPhase::OfflineRolling;
                self.session_frames = 0;
                new_epoch = true;
            }
            self.session_frames = self.session_frames.saturating_add(u64::from(input.frames));
            self.last_playing = true;
            return SessionDecision {
                phase: self.phase,
                new_epoch,
                transport_active: true,
                start_window_open: false,
            };
        }

        if input.playing {
            match self.phase {
                // 真正的新起播：从 Idle / Paused / Stopped 进入 Rolling。
                SessionPhase::Idle | SessionPhase::Paused | SessionPhase::Stopped => {
                    self.epoch = self.epoch.saturating_add(1);
                    self.session_frames = 0;
                    self.phase = SessionPhase::Rolling;
                    new_epoch = true;
                }
                // 【关键】pre-roll 与正文是同一次演奏：不新开 epoch。
                SessionPhase::Starting => {
                    self.phase = SessionPhase::Rolling;
                }
                SessionPhase::Rolling | SessionPhase::OfflineRolling => {
                    self.phase = SessionPhase::Rolling;
                }
            }
            self.session_frames = self.session_frames.saturating_add(u64::from(input.frames));
            transport_active = true;
        } else if input.playing_rising || input.recording_rising {
            if self.phase != SessionPhase::Starting {
                self.epoch = self.epoch.saturating_add(1);
                self.session_frames = 0;
                self.phase = SessionPhase::Starting;
                new_epoch = true;
            }
            start_window_open = self.session_frames < self.max_start_window_frames;
            transport_active = start_window_open;
            self.session_frames = self.session_frames.saturating_add(u64::from(input.frames));
        } else if self.last_playing || self.phase == SessionPhase::Rolling {
            // kPlaying 从 1 落到 0：立即冻结。绝不再用内容变化/静音重新解释。
            self.phase = SessionPhase::Paused;
        } else if self.phase == SessionPhase::Starting
            && self.session_frames < self.max_start_window_frames
        {
            // 起播窗口：宿主还来不及置 kPlaying，但窗口有严格上限。
            start_window_open = true;
            transport_active = true;
            self.session_frames = self.session_frames.saturating_add(u64::from(input.frames));
        } else if self.phase != SessionPhase::Idle {
            self.phase = SessionPhase::Paused;
        }

        if input.host_reset && self.phase == SessionPhase::Rolling {
            self.epoch = self.epoch.saturating_add(1);
            self.session_frames = 0;
            new_epoch = true;
        }
        self.last_playing = input.playing;
        SessionDecision {
            phase: self.phase,
            new_epoch,
            transport_active,
            start_window_open,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(playing: bool, rising: bool, frames: u32) -> SessionInput {
        SessionInput {
            playing,
            playing_rising: rising,
            recording_rising: false,
            host_reset: false,
            offline: false,
            frames,
        }
    }

    #[test]
    fn k_playing_zero_freezes_even_when_input_keeps_arriving() {
        let mut session = TransportSession::with_rate(48_000.0);
        session.observe(input(true, false, 512));
        let decision = session.observe(input(false, false, 512));
        assert_eq!(decision.phase, SessionPhase::Paused);
        assert!(!decision.transport_active);
        assert!(!decision.start_window_open);
    }

    #[test]
    fn a_new_playing_edge_from_paused_is_a_new_epoch() {
        let mut session = TransportSession::with_rate(48_000.0);
        let first = session.observe(input(true, true, 512));
        assert!(first.new_epoch);
        session.observe(input(false, false, 512));
        let second = session.observe(input(true, true, 512));
        assert!(second.new_epoch, "每次重新起播都必须开新 epoch");
        assert_ne!(second.phase, SessionPhase::Paused);
    }

    /// 问题 1 的直接回归：pre-roll 打开 Starting，kPlaying 上升沿进入 Rolling
    /// **不得**递增 epoch，否则 HostClock 会把它当成新会话而跳过域切换。
    #[test]
    fn starting_to_rolling_keeps_the_epoch() {
        let mut session = TransportSession::with_rate(48_000.0);
        let first = session.observe(input(false, true, 848));
        assert_eq!(first.phase, SessionPhase::Starting);
        assert!(first.new_epoch);
        let second = session.observe(input(true, true, 848));
        assert_eq!(second.phase, SessionPhase::Rolling);
        assert!(!second.new_epoch);
        assert!(second.transport_active);
    }

    #[test]
    fn the_start_window_is_bounded_to_one_second() {
        let mut session = TransportSession::with_rate(48_000.0);
        let first = session.observe(input(false, true, 480));
        assert!(first.start_window_open);
        let mut last = first;
        for _ in 0..120 {
            last = session.observe(input(false, false, 480));
        }
        assert!(!last.transport_active, "超过 1 秒的 kPlaying=0 必须冻结");
    }

    #[test]
    fn offline_blocks_always_advance_and_create_one_epoch() {
        let mut session = TransportSession::with_rate(48_000.0);
        let mut offline = input(false, false, 512);
        offline.offline = true;
        let first = session.observe(offline);
        assert!(first.new_epoch);
        assert!(first.transport_active);
        let second = session.observe(offline);
        assert!(!second.new_epoch);
        assert_eq!(second.phase, SessionPhase::OfflineRolling);
    }
}
