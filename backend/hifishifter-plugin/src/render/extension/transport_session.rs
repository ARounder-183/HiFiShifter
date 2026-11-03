//! 传输会话（TransportSession）：一次“播放”从开始到停止的显式生命周期。
//!
//! 旧代码用四个互相独立的信号推断“在不在播”：kPlaying、位置有没有动、输入内容
//! 有没有变、prefetch 播放位。停播后宿主仍送块时，这四个信号可以同时给出“在走”的
//! 假象。这里把它们收敛成一个状态机：只有 Starting（有界起播窗口）或 Rolling
//! 允许时间前进；Paused 与 Stopped 一律冻结。
//!
//! 本模块是纯函数，不碰宿主、不取锁；音频线程只需要把上一相位与本块事实喂进来。

/// 传输会话的相位。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum SessionPhase {
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
    /// 离线 pass 的本地游标。
    OfflineRolling,
}

impl SessionPhase {
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::Idle => 0,
            Self::Starting => 1,
            Self::Rolling => 2,
            Self::Paused => 3,
            Self::Stopped => 4,
            Self::OfflineRolling => 5,
        }
    }

    pub(crate) fn from_code(code: u8) -> Option<Self> {
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
    pub(crate) fn is_advancing(self) -> bool {
        matches!(self, Self::Starting | Self::Rolling | Self::OfflineRolling)
    }

    #[allow(dead_code)]
    pub(crate) fn as_str(self) -> &'static str {
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
pub(crate) struct TransportSession {
    pub(crate) epoch: u64,
    pub(crate) phase: SessionPhase,
    pub(crate) session_frames: u64,
    pub(crate) last_playing: bool,
    pub(crate) max_start_window_frames: u64,
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
pub(crate) struct SessionInput {
    pub(crate) playing: bool,
    pub(crate) playing_rising: bool,
    pub(crate) recording_rising: bool,
    pub(crate) host_reset: bool,
    pub(crate) offline: bool,
    pub(crate) frames: u32,
}

/// 状态机结论。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SessionDecision {
    pub(crate) phase: SessionPhase,
    pub(crate) new_epoch: bool,
    /// 本块是否有权推进位置 / 写捕获。
    pub(crate) transport_active: bool,
    /// 起点窗口还开着：窗口内的块可以按块长前进，即使 kPlaying 还是 0。
    pub(crate) start_window_open: bool,
}

impl TransportSession {
    pub(crate) fn with_rate(rate: f64) -> Self {
        let mut session = Self::default();
        if rate.is_finite() && rate > 0.0 {
            session.max_start_window_frames = (rate * 1.0).round().max(1.0) as u64;
        }
        session
    }

    /// 前进一步。host_reset = 宿主位置倒退或跳变；它是新会话的强证据之一。
    pub(crate) fn observe(&mut self, input: SessionInput) -> SessionDecision {
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
            if !self.last_playing
                || self.phase == SessionPhase::Idle
                || self.phase == SessionPhase::Stopped
            {
                self.epoch = self.epoch.saturating_add(1);
                self.session_frames = 0;
                new_epoch = true;
            }
            self.phase = SessionPhase::Rolling;
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
        } else {
            if self.phase != SessionPhase::Idle {
                self.phase = SessionPhase::Paused;
            }
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

    /// 强制停止（setProcessing(false) / setActive(false)）。
    #[allow(dead_code)]
    pub(crate) fn stop(&mut self) {
        self.phase = SessionPhase::Stopped;
        self.last_playing = false;
        self.session_frames = 0;
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
    fn a_new_playing_edge_is_a_new_epoch() {
        let mut session = TransportSession::with_rate(48_000.0);
        let first = session.observe(input(true, true, 512));
        assert!(first.new_epoch);
        session.observe(input(false, false, 512));
        let second = session.observe(input(true, true, 512));
        assert!(second.new_epoch, "每次重新起播都必须开新 epoch");
        assert_ne!(first.phase, SessionPhase::Paused);
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
