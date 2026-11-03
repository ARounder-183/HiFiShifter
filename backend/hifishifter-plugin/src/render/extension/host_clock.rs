//! 宿主时钟（HostClock）：把 ProcessContext 的原始位置映射到**唯一**的插件本地时间轴。
//!
//! 这个模块取代旧的 `classify_domain_switch`。旧分类器把三件事混在一起：
//! 坐标域、传输会话、位置记账；而 VEGAS Event FX 的域切换恰好发生在 kPlaying
//! 上升沿，三条规则互相排斥，域切换被永久跳过。现在职责拆开：
//!
//! * `TransportSession` 回答“这是不是同一次播放”；
//! * `HostClock` 回答“这一块在本地时间轴的哪里、属于哪个坐标域”；
//! * `ProcessingPass` 回答“这是不是一个离线/冻结预览 pass”。
//!
//! 本模块是纯状态机：不碰宿主、不取锁、不分配，因此可以在音频线程调用，也可以
//! 脱离宿主单测。

use super::transport_session::SessionPhase;

/// 宿主原始读数的坐标域。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum RawDomain {
    /// 工程绝对位置。
    #[default]
    Absolute,
    /// 事件内相对位置。
    Relative,
}

/// 当前生效的宿主坐标域。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum ClockDomain {
    /// 还没有足够证据。
    #[default]
    Unknown,
    /// Track FX / 普通插入：工程绝对位置。
    Project,
    /// Event FX：事件坐标域。`origin` 是事件在工程里的绝对起点。
    Event { origin: i64, raw_domain: RawDomain },
}

impl ClockDomain {
    pub(crate) fn is_event(self) -> bool {
        matches!(self, Self::Event { .. })
    }

    /// 稳定编码（只用于诊断与原子存储）。**只增不改**。
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::Project => 1,
            Self::Event {
                raw_domain: RawDomain::Absolute,
                ..
            } => 2,
            Self::Event {
                raw_domain: RawDomain::Relative,
                ..
            } => 3,
        }
    }
}

/// 音频边界连续性。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum Continuity {
    #[default]
    Unknown,
    Continuous,
    Discontinuous,
}

impl Continuity {
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::Continuous => 1,
            Self::Discontinuous => 2,
        }
    }

    pub(crate) fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Continuous,
            2 => Self::Discontinuous,
            _ => Self::Unknown,
        }
    }
}

/// 域切换的观测窗口：只在一次传输会话的最初 2 秒内猜测。
pub(crate) const DOMAIN_SWITCH_WINDOW_SEC: f64 = 2.0;

/// 域切换的最小幅度：小于半秒的差值属于块边界/记账误差，不是换了一把尺子。
pub(crate) const DOMAIN_SWITCH_MIN_SEC: f64 = 0.5;

/// 把“同一段音频在两种坐标域里的读数”视为同一个本地位置的容差。
pub(crate) const MAX_MAPPING_ERROR_SEC: f64 = 0.25;

/// 一个本地时间轴的纯状态。
#[derive(Clone, Copy, Debug)]
pub(crate) struct ClockState {
    pub(crate) domain: ClockDomain,
    /// 实例级事实：事件在工程里的绝对起点。跨 epoch 保留。
    pub(crate) event_origin: Option<i64>,
    /// 本次 take 第一个有权块的原始位置。跨块保留，`start()` 时由捕获侧清空。
    pub(crate) capture_origin_raw: Option<i64>,
    /// 本传输会话第一块的原始位置。
    pub(crate) session_first_raw: Option<i64>,
    /// 事件内已经见过的最大本地终点（只用于候选评分）。
    pub(crate) event_max_local: i64,
    pub(crate) last_raw: i64,
    pub(crate) last_local_end: i64,
    pub(crate) last_frames: u32,
    pub(crate) session_frames: u64,
    pub(crate) last_continuity: Continuity,
    pub(crate) provisional: bool,
    pub(crate) last_playing: bool,
    pub(crate) start_window_open: bool,
}

impl Default for ClockState {
    fn default() -> Self {
        Self {
            domain: ClockDomain::Unknown,
            event_origin: None,
            capture_origin_raw: None,
            session_first_raw: None,
            event_max_local: 0,
            last_raw: 0,
            last_local_end: 0,
            last_frames: 0,
            session_frames: 0,
            last_continuity: Continuity::Unknown,
            provisional: true,
            last_playing: false,
            start_window_open: false,
        }
    }
}

/// 本块看到的宿主事实。
#[derive(Clone, Copy, Debug)]
pub(crate) struct ClockInput {
    pub(crate) raw_start: i64,
    pub(crate) frames: usize,
    pub(crate) rate: f64,
    pub(crate) playing: bool,
    pub(crate) playing_rising: bool,
    pub(crate) recording_rising: bool,
    pub(crate) cycle_active: bool,
    pub(crate) cycle_wrap: bool,
    pub(crate) continuity: Continuity,
    pub(crate) had_previous: bool,
    /// 本块**处理之前**的传输相位。
    pub(crate) session_phase: SessionPhase,
    /// 本块**处理之前**本会话已经收到的帧数。
    pub(crate) session_frames: u64,
    /// TransportSession 是否把本块判成新 epoch。
    pub(crate) session_new_epoch: bool,
    pub(crate) capture_origin_raw: Option<i64>,
    pub(crate) persisted_event_origin: Option<i64>,
    pub(crate) music: Option<f64>,
    pub(crate) tempo: Option<f64>,
}

/// 本块的时钟结论。
#[derive(Clone, Copy, Debug)]
pub(crate) struct ClockDecision {
    /// 本块应落到的本地位置（R2 时是“上一块终点”，调用方可再用 continuous/music 校正）。
    pub(crate) local_start: i64,
    pub(crate) domain: ClockDomain,
    /// 坐标域发生了变化（不一定是 seek）。
    pub(crate) switched: bool,
    /// 整条 take 需要平移的帧数；0 = 不平移。
    pub(crate) rebase_frames: i64,
    /// 这是一次真 seek（不是域切换）。
    pub(crate) seek: bool,
    pub(crate) timebase_offset: i64,
    pub(crate) event_origin: Option<i64>,
    #[allow(dead_code)]
    pub(crate) provisional: bool,
}

fn offset_for(domain: ClockDomain) -> i64 {
    match domain {
        ClockDomain::Event {
            origin,
            raw_domain: RawDomain::Absolute,
        } => -origin,
        _ => 0,
    }
}

fn local_for_raw(domain: ClockDomain, raw: i64) -> i64 {
    raw.saturating_add(offset_for(domain))
}

fn sane_rate(rate: f64) -> f64 {
    if rate.is_finite() && rate > 0.0 {
        rate
    } else {
        48_000.0
    }
}

fn frames_for(rate: f64, seconds: f64) -> i64 {
    (sane_rate(rate) * seconds).round().max(1.0) as i64
}

impl ClockState {
    /// 前进一步。纯函数：只改 `self`，不碰任何外部状态。
    pub(crate) fn observe(&mut self, input: ClockInput) -> ClockDecision {
        let rate = sane_rate(input.rate);
        let frames = input.frames as i64;
        let had = input.had_previous;
        let raw = input.raw_start;

        // 新 take：捕获侧给了新的 origin（或明确清空）。
        if input.session_new_epoch {
            self.capture_origin_raw = input.capture_origin_raw;
            self.session_first_raw = Some(raw);
        } else {
            if input.capture_origin_raw.is_some() {
                self.capture_origin_raw = input.capture_origin_raw;
            }
            if self.session_first_raw.is_none() {
                self.session_first_raw = Some(raw);
            }
        }
        if self.event_origin.is_none() {
            self.event_origin = input.persisted_event_origin;
        }

        let previous_local_end = self.last_local_end;
        let delta = raw.saturating_sub(previous_local_end);
        let min_frames = frames_for(rate, DOMAIN_SWITCH_MIN_SEC);
        let window_frames = frames_for(rate, DOMAIN_SWITCH_WINDOW_SEC).max(1) as u64;
        let max_error = frames_for(rate, MAX_MAPPING_ERROR_SEC);
        let session_in_window = input.session_frames <= window_frames;
        let continuous = input.continuity != Continuity::Discontinuous;
        let same_session_edge = (input.playing_rising || input.recording_rising)
            && !input.session_new_epoch
            && input.session_phase == SessionPhase::Starting;

        // ① 本会话第一块：若已知事件起点，先解析“这一块报的是绝对还是相对”。
        if !had && self.event_origin.is_some() {
            let origin = self.event_origin.expect("checked");
            let (domain, local) = self.resolve_first_block(raw, origin, &input, rate);
            self.domain = domain;
            self.provisional = false;
            let decision = ClockDecision {
                local_start: local,
                domain,
                switched: true,
                rebase_frames: 0,
                seek: false,
                timebase_offset: offset_for(domain),
                event_origin: self.event_origin,
                provisional: false,
            };
            self.commit(local, raw, frames, input.continuity, input.session_frames);
            return decision;
        }

        // ② 同一事件里“绝对读数 -> 相对读数”但本地位置连续：读数换了尺子，
        //    本地位置不该动。VEGAS 从事件中间起播时会先报绝对、再报相对。
        if let ClockDomain::Event {
            origin,
            raw_domain: RawDomain::Absolute,
        } = self.domain
        {
            if had
                && !input.cycle_active
                && continuous
                && raw < self.last_raw
                && (raw - previous_local_end).abs() <= max_error
            {
                self.domain = ClockDomain::Event {
                    origin,
                    raw_domain: RawDomain::Relative,
                };
                self.provisional = false;
                let decision = ClockDecision {
                    local_start: raw,
                    domain: self.domain,
                    switched: true,
                    rebase_frames: 0,
                    seek: false,
                    timebase_offset: 0,
                    event_origin: Some(origin),
                    provisional: false,
                };
                self.commit(raw, raw, frames, input.continuity, input.session_frames);
                return decision;
            }
        }

        // ③ 相对 -> 绝对：映射回本地后与上一块终点连续。
        if let ClockDomain::Event {
            origin,
            raw_domain: RawDomain::Relative,
        } = self.domain
        {
            if had && !input.cycle_active && continuous && delta > min_frames && session_in_window {
                let mapped = raw.saturating_sub(origin);
                if mapped >= 0 && (mapped - previous_local_end).abs() <= max_error {
                    self.domain = ClockDomain::Event {
                        origin,
                        raw_domain: RawDomain::Absolute,
                    };
                    self.provisional = false;
                    let decision = ClockDecision {
                        local_start: mapped,
                        domain: self.domain,
                        switched: true,
                        rebase_frames: 0,
                        seek: false,
                        timebase_offset: -origin,
                        event_origin: Some(origin),
                        provisional: false,
                    };
                    self.commit(mapped, raw, frames, input.continuity, input.session_frames);
                    return decision;
                }
            }
        }

        // ④ 工程/未知域 -> 事件相对域：一次“换尺子”的巨大回退。
        //    只允许在域切换窗口内，且必须有事件证据（已知 origin 或同一 Starting
        //    会话里的 kPlaying 上升沿）。Track FX 的普通 seek 没有事件证据。
        let provisional_shape = input.playing
            && self.event_origin.is_none()
            && self.capture_origin_raw.is_none()
            && previous_local_end >= frames_for(rate, 2.0)
            && self.session_first_raw != Some(raw)
            && raw >= 0
            && raw < frames_for(rate, 5.0);
        let event_evidence = self.event_origin.is_some() || same_session_edge || provisional_shape;
        if matches!(self.domain, ClockDomain::Unknown | ClockDomain::Project)
            && had
            && !input.cycle_active
            && continuous
            && delta <= -min_frames
            && session_in_window
            && event_evidence
        {
            if let Some(origin) = self.pick_event_origin(previous_local_end, raw, max_error) {
                if origin >= 0 {
                    self.domain = ClockDomain::Event {
                        origin,
                        raw_domain: RawDomain::Relative,
                    };
                    self.event_origin = Some(origin);
                    self.provisional = false;
                    let decision = ClockDecision {
                        local_start: raw,
                        domain: self.domain,
                        switched: true,
                        rebase_frames: -origin,
                        seek: false,
                        timebase_offset: 0,
                        event_origin: Some(origin),
                        provisional: false,
                    };
                    self.commit(raw, raw, frames, input.continuity, input.session_frames);
                    return decision;
                }
            }
        }

        // ⑤ 真 seek：没有域切换证据的位置跳变。
        let seek = had
            && raw != self.last_raw
            && !input.cycle_active
            && !input.cycle_wrap
            && (delta.unsigned_abs() > max_error as u64)
            && (input.continuity == Continuity::Discontinuous
                || input.session_frames > window_frames);

        let local = if !had {
            local_for_raw(self.domain, raw)
        } else if raw == self.last_raw {
            // R2：宿主没报新位置，本地位置按上一块终点前进。
            previous_local_end
        } else {
            local_for_raw(self.domain, raw)
        };
        let decision = ClockDecision {
            local_start: local,
            domain: self.domain,
            switched: false,
            rebase_frames: 0,
            seek,
            timebase_offset: offset_for(self.domain),
            event_origin: self.event_origin,
            provisional: self.provisional,
        };
        self.commit(local, raw, frames, input.continuity, input.session_frames);
        decision
    }

    /// 本会话第一块的候选解析。
    fn resolve_first_block(
        &self,
        raw: i64,
        origin: i64,
        input: &ClockInput,
        rate: f64,
    ) -> (ClockDomain, i64) {
        let absolute = raw.saturating_sub(origin);
        let relative = raw;
        let choose_absolute = if absolute < 0 {
            false
        } else if let (Some(music), Some(tempo)) = (input.music, input.tempo) {
            let local_sec = relative as f64 / rate;
            let absolute_sec = absolute as f64 / rate;
            let expected_beats = |seconds: f64| seconds * tempo / 60.0;
            (expected_beats(absolute_sec) - music).abs()
                <= (expected_beats(local_sec) - music).abs()
        } else if raw < origin {
            false
        } else if self.event_max_local == 0 {
            // 【没有历史长度时的唯一可用判据】raw 已经在事件起点之后，且把它读成
            // 绝对位置会得到更小的本地位置。真实的长事件（相对读数 > origin）与
            // 从事件中间起播（绝对读数）在这里同形，只能靠 projectTimeMusic 区分；
            // 没有 music 时优先相信“宿主先报绝对”这一实测形状。
            absolute < raw
        } else {
            absolute <= self.event_max_local.saturating_add(frames_for(rate, 2.0))
        };
        if choose_absolute {
            (
                ClockDomain::Event {
                    origin,
                    raw_domain: RawDomain::Absolute,
                },
                absolute,
            )
        } else {
            (
                ClockDomain::Event {
                    origin,
                    raw_domain: RawDomain::Relative,
                },
                relative,
            )
        }
    }

    /// 为一次“绝对 -> 相对”切换挑选事件起点。
    ///
    /// 首选本次 take 第一个有权块的位置（`capture_origin_raw`）：它不受 lookbehind
    /// 丢块影响。若它与“上一块终点 - 相对读数”差得太远（说明这次演奏是从事件中间
    /// 开始的），退化为后者 —— 它才是这一段音频真正的绝对起点。
    fn pick_event_origin(&self, previous_local_end: i64, raw: i64, max_error: i64) -> Option<i64> {
        let from_capture = self.capture_origin_raw.filter(|origin| {
            *origin >= 0
                && (previous_local_end
                    .saturating_sub(*origin)
                    .saturating_sub(raw))
                .abs()
                    <= max_error
        });
        let from_end = previous_local_end.saturating_sub(raw);
        from_capture
            .or_else(|| (from_end >= 0).then_some(from_end))
            .or(self.event_origin)
            .or(self.session_first_raw)
            .or(Some(raw))
    }

    /// 记账：下一块的“上一块终点”。
    pub(crate) fn commit(
        &mut self,
        local_start: i64,
        raw: i64,
        frames: i64,
        continuity: Continuity,
        session_frames_before: u64,
    ) {
        let end = local_start.saturating_add(frames);
        self.last_raw = raw;
        self.last_local_end = end;
        self.last_frames = u32::try_from(frames).unwrap_or(u32::MAX);
        self.session_frames = session_frames_before.saturating_add(frames.max(0) as u64);
        self.last_continuity = continuity;
        if self.domain.is_event() {
            self.event_max_local = self.event_max_local.max(end);
        }
        if !self.domain.is_event() && self.event_origin.is_none() {
            self.provisional = true;
        }
    }

    /// 本块最终采用的位置（R2 的 continuous/music 校正可能不同于默认值）。
    pub(crate) fn note_final_local(&mut self, local_start: i64, frames: i64) {
        self.last_local_end = local_start.saturating_add(frames);
        self.last_frames = u32::try_from(frames).unwrap_or(u32::MAX);
        if self.domain.is_event() {
            self.event_max_local = self.event_max_local.max(self.last_local_end);
        }
    }

    /// 进入一个已确认的工程域（第一次有权块、且没有任何事件证据）。
    pub(crate) fn confirm_project(&mut self) {
        if self.domain == ClockDomain::Unknown {
            self.domain = ClockDomain::Project;
            self.provisional = false;
        }
    }

    /// 会话边界：保留实例级 `event_origin`，清空会话级读数。
    pub(crate) fn reset_session(&mut self) {
        self.capture_origin_raw = None;
        self.session_first_raw = None;
        self.last_raw = 0;
        self.last_local_end = 0;
        self.last_frames = 0;
        self.session_frames = 0;
        self.last_continuity = Continuity::Unknown;
        self.last_playing = false;
        self.start_window_open = false;
        // 域保留：同一个 Event 的下一次播放仍在同一域；Track 仍是 Project。
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f64 = 48_000.0;

    fn input(raw: i64, frames: usize, playing: bool, rising: bool) -> ClockInput {
        ClockInput {
            raw_start: raw,
            frames,
            rate: RATE,
            playing,
            playing_rising: rising,
            recording_rising: false,
            cycle_active: false,
            cycle_wrap: false,
            continuity: Continuity::Continuous,
            had_previous: false,
            session_phase: SessionPhase::Starting,
            session_frames: 0,
            session_new_epoch: true,
            capture_origin_raw: None,
            persisted_event_origin: None,
            music: None,
            tempo: None,
        }
    }

    fn feed(state: &mut ClockState, mut block: ClockInput) -> ClockDecision {
        block.had_previous = state.last_frames > 0;
        block.session_new_epoch = false;
        block.session_phase = SessionPhase::Starting;
        block.session_frames = state.session_frames;
        state.observe(block)
    }

    #[test]
    fn event_switch_at_playing_edge() {
        let mut state = ClockState::default();
        let first = state.observe(input(906_000, 32_768, false, true));
        assert_eq!(first.local_start, 906_000);
        assert!(!first.switched);
        let mut second = input(32_768, 32_768, true, true);
        second.capture_origin_raw = Some(906_000);
        let decision = feed(&mut state, second);
        assert_eq!(
            decision.domain,
            ClockDomain::Event {
                origin: 906_000,
                raw_domain: RawDomain::Relative
            }
        );
        assert_eq!(decision.local_start, 32_768);
        assert_eq!(decision.rebase_frames, -906_000);
        assert!(!decision.seek);
    }

    #[test]
    fn event_sidecar_origin_is_not_end_minus_raw() {
        let mut state = ClockState::default();
        state.observe(input(906_000, 32_768, false, true));
        // 第一块相对读数被 lookbehind 暂存，下一块才是 38064。
        let mut second = input(38_064, 5_296, true, true);
        second.capture_origin_raw = Some(906_000);
        let decision = feed(&mut state, second);
        assert_eq!(decision.rebase_frames, -906_000);
        // 若按 end - raw 会得到 900704，留下 5296 帧假洞。
        assert_ne!(decision.rebase_frames, -(938_768 - 38_064));
    }

    #[test]
    fn event_start_at_half_second() {
        let mut state = ClockState::default();
        // 事件起点 0.5 s；首块 0.5..1.1 s，下一块相对读数 0.6 s。
        state.observe(input(24_000, 28_800, false, true));
        let mut second = input(28_800, 24_000, true, true);
        second.capture_origin_raw = Some(24_000);
        let decision = feed(&mut state, second);
        assert_eq!(decision.rebase_frames, -24_000);
        assert_eq!(decision.local_start, 28_800);
    }

    #[test]
    fn event_mid_playback_at_10s_keeps_local_position() {
        // 已知事件起点 31 s；宿主先报绝对 41 s，再报相对 10 s。
        let mut state = ClockState::default();
        state.event_origin = Some(31 * 48_000);
        let first = state.observe(input(41 * 48_000, 1_000, true, false));
        assert_eq!(
            first.domain,
            ClockDomain::Event {
                origin: 31 * 48_000,
                raw_domain: RawDomain::Absolute
            }
        );
        assert_eq!(first.local_start, 10 * 48_000);
        let second = feed(&mut state, input(10 * 48_000, 1_000, true, false));
        assert_eq!(
            second.domain,
            ClockDomain::Event {
                origin: 31 * 48_000,
                raw_domain: RawDomain::Relative
            }
        );
        assert_eq!(second.local_start, 10 * 48_000);
        assert_eq!(second.rebase_frames, 0);
    }

    #[test]
    fn event_reverse_switch_keeps_audio_continuity() {
        let mut state = ClockState::default();
        state.domain = ClockDomain::Event {
            origin: 31 * 48_000,
            raw_domain: RawDomain::Relative,
        };
        state.event_origin = Some(31 * 48_000);
        state.last_raw = 24_000;
        state.last_local_end = 24_000;
        state.last_frames = 1;
        let decision = feed(&mut state, input(31 * 48_000 + 24_000, 1_000, true, false));
        assert_eq!(
            decision.domain,
            ClockDomain::Event {
                origin: 31 * 48_000,
                raw_domain: RawDomain::Absolute
            }
        );
        assert_eq!(decision.local_start, 24_000);
        assert_eq!(decision.rebase_frames, 0);
    }

    #[test]
    fn track_playback_is_never_event() {
        let mut state = ClockState::default();
        let first = state.observe(input(906_000, 848, true, true));
        assert_eq!(first.domain, ClockDomain::Unknown);
        let second = feed(&mut state, input(942_016, 848, true, false));
        assert_eq!(second.domain, ClockDomain::Unknown);
        assert_eq!(second.rebase_frames, 0);
        assert!(!second.switched);
    }

    #[test]
    fn track_start_seek_is_seek() {
        let mut state = ClockState::default();
        state.confirm_project();
        state.observe(input(31 * 48_000, 848, true, true));
        let mut seek = input(5 * 48_000, 848, true, false);
        seek.continuity = Continuity::Discontinuous;
        let decision = feed(&mut state, seek);
        assert!(decision.seek);
        assert!(!decision.switched);
        assert_eq!(decision.rebase_frames, 0);
    }

    #[test]
    fn event_origin_persists_across_epochs() {
        let mut state = ClockState::default();
        state.event_origin = Some(31 * 48_000);
        state.domain = ClockDomain::Event {
            origin: 31 * 48_000,
            raw_domain: RawDomain::Relative,
        };
        let decision = state.observe(input(10 * 48_000, 1_000, true, true));
        assert_eq!(decision.local_start, 10 * 48_000);
        assert_eq!(decision.rebase_frames, 0);
    }

    #[test]
    fn event_origin_from_component_state_maps_absolute_raw() {
        let mut state = ClockState::default();
        state.event_origin = Some(31 * 48_000);
        let decision = state.observe(input(41 * 48_000, 1_000, true, true));
        assert_eq!(decision.local_start, 10 * 48_000);
        assert_eq!(
            decision.domain,
            ClockDomain::Event {
                origin: 31 * 48_000,
                raw_domain: RawDomain::Absolute
            }
        );
    }

    #[test]
    fn event_origin_unknown_keeps_project() {
        let mut state = ClockState::default();
        state.confirm_project();
        state.observe(input(41 * 48_000, 1_000, true, true));
        let mut jump = input(10 * 48_000, 1_000, true, false);
        jump.continuity = Continuity::Discontinuous;
        let decision = feed(&mut state, jump);
        assert_eq!(decision.domain, ClockDomain::Project);
        assert!(!decision.switched);
        assert_eq!(decision.rebase_frames, 0);
    }

    #[test]
    fn a_playing_edge_from_paused_is_not_a_domain_switch() {
        let mut state = ClockState::default();
        state.confirm_project();
        state.observe(input(31 * 48_000, 1_000, true, true));
        let mut edge = input(0, 1_000, true, true);
        edge.session_new_epoch = true;
        edge.session_phase = SessionPhase::Paused;
        edge.capture_origin_raw = Some(31 * 48_000);
        let decision = state.observe(edge);
        assert!(!decision.switched);
        assert_eq!(decision.rebase_frames, 0);
    }

    #[test]
    fn preview_candidate_is_cancelled_by_playing_edge() {
        // 这条由 ProcessingPass 的单测覆盖；这里只钉住 HostClock 不把
        // “固定 raw + kPlaying=0”当成域切换。
        let mut state = ClockState::default();
        state.confirm_project();
        state.observe(input(1_295_764, 1_200, false, true));
        let decision = feed(&mut state, input(1_295_764, 1_200, false, false));
        assert!(!decision.switched);
        assert_eq!(decision.rebase_frames, 0);
    }
}
