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
//!
//! # 位置可信度是显式的
//!
//! 首块在“已武装 + 未播放 + 绝对域”时**不再直接落成工程域**，而是进入
//! `PendingAbsolute`：位置照常按 raw 写入捕获（Track 的预卷必须保留），但
//! **不发布**（`position_trusted = false`、`publish_deferred = true`）。下一块如果
//! `kPlaying` 升起且 raw 回退，就按 `choose_origin` 定出 Event 原点、整条 take
//! 平移并裁剪不连续前缀；如果 raw 继续绝对前进，就落成 Project，预卷原样保留。
//!
//! 这样“猜一个 0..5 s 的候选”不再参与任何决策：证据不足时位置不可发布，证据
//! 充分时原点由 `capture_origin_raw` 与 `projectTimeMusic` 的跳变共同给出。

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
    /// 已收到有权块，但还没有足够证据区分 Project 与 Event。
    ///
    /// 【为什么需要它】VEGAS 的 Event FX 起播会先送一段 `kPlaying = 0` 的绝对域
    /// 预卷块，再在 `kPlaying` 上升沿切到事件相对域。首块若直接落成 Project，
    /// 这段预卷就会被当成正文发布出去（现场：光标从 60.000 s 跳到 0.750 s）。
    /// 它只表示“捕获可以按 raw 写、但位置不可发布”，不是一种猜测。
    PendingAbsolute,
    /// Track FX / 普通插入：工程绝对位置。
    Project,
    /// Event FX：事件坐标域。`origin` 是事件在工程里的绝对起点。
    Event { origin: i64, raw_domain: RawDomain },
}

impl ClockDomain {
    pub(crate) fn is_event(self) -> bool {
        matches!(self, Self::Event { .. })
    }

    /// 本域的 raw → local 偏移（Project/Pending 为 0，Event 绝对域为 -origin）。
    pub(crate) fn local_offset(self) -> i64 {
        offset_for(self)
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
            Self::PendingAbsolute => 4,
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

/// `PendingAbsolute` 最多保留多久（秒）。
///
/// 【为什么需要上界】一直停在 PendingAbsolute 会让位置永远不可发布。VEGAS 的预卷
/// 实测在 1 秒内结束；2 秒之后仍没有 `kPlaying`，那它就是一次真实的工程域
/// 停播（Track 停在光标处），必须如实落成 Project。
pub(crate) const DOMAIN_PROBE_SEC: f64 = 2.0;

/// `projectTimeMusic` 回退超过这么多拍，才认为读数换了尺子。
///
/// 【为什么是一拍】现场 Event 预卷 `24414.06` → 相对 `1.50`，远超阈值；
/// 而同一域内的正常推进只会前进，不会回退。一拍足以把“换域”与“块边界量化”分开。
pub(crate) const MUSIC_DROP_BEATS: f64 = 1.0;

/// `origin_valid` 允许的音乐时间误差（拍）。
///
/// 【为什么是 4 拍】`projectTimeMusic` 与采样位置之间隔着宿主的块量化与
/// 速度映射；4 拍（2 秒 @120 BPM）足以容忍它们，又远小于“工程级音乐时间”与
/// “事件内音乐时间”的差（现场 `28635` vs `1.5`）。
pub(crate) const MUSIC_VALID_TOLERANCE_BEATS: f64 = 4.0;

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
    /// `PendingAbsolute` 的首块 raw 与同一块的 `projectTimeMusic`。
    pub(crate) pending_raw: i64,
    pub(crate) pending_music: Option<f64>,
    /// 上一块的 `projectTimeMusic`（只用于判断“读数换了尺子”）。
    pub(crate) last_music: Option<f64>,
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
            pending_raw: 0,
            pending_music: None,
            last_music: None,
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
    /// 本实例已武装（Capture 正在等宿主播放）。
    ///
    /// 【为什么由调用方给】`HostClock` 不取捕获锁；武装状态是捕获侧的
    /// 事实，在音频线程上本来就已经读成局部变量。
    pub(crate) capture_armed: bool,
    /// 本块是“这一次播放的第一块”（首块 / 上升沿）。
    pub(crate) transport_start: bool,
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
    /// 域切换后需要丢弃的、位于该本地位置之前的不连续前缀（0 = 不裁剪）。
    pub(crate) prune_before: i64,
    /// 这是一次真 seek（不是域切换）。
    pub(crate) seek: bool,
    pub(crate) timebase_offset: i64,
    pub(crate) event_origin: Option<i64>,
    /// 本块位置可以发布给 UI / 输出仲裁。
    pub(crate) position_trusted: bool,
    /// 本块位置**暂不发布**（`PendingAbsolute` 期间为真）。
    pub(crate) publish_deferred: bool,
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

/// `projectTimeMusic` 是否发生了一次“换尺子”式的回退。
pub(crate) fn music_dropped(before: Option<f64>, now: Option<f64>) -> bool {
    before
        .zip(now)
        .is_some_and(|(before, now)| before - now > MUSIC_DROP_BEATS)
}

/// 事件起点是否与这一块的宿主读数自洽。
///
/// 【为什么必须校验】`eventOriginSamples` 会跟着组件 state 复制/粘贴到另一个
/// Event。一个陈旧的 `origin = 0` 与工程级 `projectTimeMusic` 组合起来会
/// 把整条 take 拉到工程 0，而现场看起来只是“从宿主光标开始”。
///
/// 【两种读数都要接受】Event 的绝对读数满足 `raw >= origin`；相对读数满足
/// `raw < origin`（`raw` 是事件内偏移）。前者用
/// `(raw - origin) / rate * tempo / 60` 预测拍数，后者用 `raw` 本身。
pub(crate) fn origin_valid(
    origin: i64,
    raw: i64,
    music: Option<f64>,
    tempo: Option<f64>,
    rate: f64,
) -> bool {
    if origin < 0 {
        return false;
    }
    let (Some(music), Some(tempo)) = (music, tempo) else {
        return true;
    };
    let rate = sane_rate(rate);
    if !(tempo.is_finite() && tempo > 0.0) {
        return true;
    }
    let local = if raw >= origin {
        raw.saturating_sub(origin)
    } else {
        raw
    };
    let expected = local as f64 / rate * tempo / 60.0;
    if (expected - music).abs() > MUSIC_VALID_TOLERANCE_BEATS {
        return false;
    }
    // 工程级音乐时间远大于事件内拍数。用“比预期大一个数量级”拒绝陈旧 origin，
    // 而不是写死一个秒数上限 —— 很长的 Event 仍然合法。
    if music > expected + 1000.0 && music > 1000.0 {
        return false;
    }
    true
}

/// 为一次“绝对 -> 相对”切换挑选事件起点。
///
/// 【为什么不再用 `previous_end - raw`】预卷与正文之间可能本来就不连续（现场
/// 差 3248 帧），那个差值得到的是 `origin - gap`，会在 take 开头留下一段假洞。
/// 现在只有两种证据：
///
/// 1. `capture_origin_raw`：本次 take 第一个有权块的 raw（预卷块）。VEGAS 的
///    预卷块报的是 **Event 起点**；音乐时间发生回退时直接采信它。
/// 2. 没有音乐回退时，预卷块报的是**宿主光标**，事件起点 = 光标 − 事件内偏移。
pub(crate) fn choose_origin(
    pending_raw: i64,
    capture_origin: Option<i64>,
    relative_raw: i64,
    music_before: Option<f64>,
    music_now: Option<f64>,
) -> i64 {
    let dropped = music_dropped(music_before, music_now);
    if let Some(origin) = capture_origin {
        if dropped {
            return origin;
        }
        if origin >= relative_raw {
            return origin.saturating_sub(relative_raw);
        }
        return origin;
    }
    if dropped {
        return pending_raw;
    }
    pending_raw.saturating_sub(relative_raw).max(0)
}

impl ClockState {
    #[allow(clippy::too_many_arguments)]
    fn decision(
        &self,
        local_start: i64,
        switched: bool,
        rebase_frames: i64,
        seek: bool,
        prune_before: i64,
        position_trusted: bool,
        publish_deferred: bool,
    ) -> ClockDecision {
        ClockDecision {
            local_start,
            domain: self.domain,
            switched,
            rebase_frames,
            prune_before,
            seek,
            timebase_offset: offset_for(self.domain),
            event_origin: self.event_origin,
            position_trusted,
            publish_deferred,
            provisional: self.provisional,
        }
    }

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
        // 【陈旧 origin 必须在这里失效】复制/粘贴 Event FX 会把上一事件的 origin 带过来；
        // 它一旦永远赢，整条 take 就被拉到工程 0。
        if let Some(origin) = self.event_origin {
            if !origin_valid(origin, raw, input.music, input.tempo, rate) {
                self.event_origin = None;
                if self.domain.is_event() {
                    self.domain = ClockDomain::Unknown;
                }
            }
        }

        let previous_local_end = self.last_local_end;
        let delta = raw.saturating_sub(previous_local_end);
        let min_frames = frames_for(rate, DOMAIN_SWITCH_MIN_SEC);
        let window_frames = frames_for(rate, DOMAIN_SWITCH_WINDOW_SEC).max(1) as u64;
        let probe_frames = frames_for(rate, DOMAIN_PROBE_SEC).max(1) as u64;
        let max_error = frames_for(rate, MAX_MAPPING_ERROR_SEC);
        let session_in_window = input.session_frames <= window_frames;
        let continuous = input.continuity != Continuity::Discontinuous;
        let same_session_edge = (input.playing_rising || input.recording_rising)
            && !input.session_new_epoch
            && input.session_phase == SessionPhase::Starting;
        let music_drop = music_dropped(self.last_music, input.music);

        // ① 本会话第一块。
        if !had {
            if let Some(origin) = self.event_origin {
                let (domain, local) = self.resolve_first_block(raw, origin, &input, rate);
                self.domain = domain;
                self.provisional = false;
                let decision = self.decision(local, true, 0, false, 0, true, false);
                self.commit(
                    local,
                    raw,
                    frames,
                    input.continuity,
                    input.session_frames,
                    input.music,
                );
                return decision;
            }
            // 【PendingAbsolute】已武装、还没播放、首块是绝对域：这既可能是 Track 的
            // 预卷（同域连续），也可能是 Event 的预卷（下一块会回退到事件相对域）。
            // 位置照常按 raw 写入捕获，但**不发布**。
            if input.capture_armed && !input.playing && input.transport_start {
                self.domain = ClockDomain::PendingAbsolute;
                self.pending_raw = raw;
                self.pending_music = input.music;
                self.provisional = true;
                let decision = self.decision(raw, false, 0, false, 0, false, true);
                self.commit(
                    raw,
                    raw,
                    frames,
                    input.continuity,
                    input.session_frames,
                    input.music,
                );
                return decision;
            }
            self.domain = ClockDomain::Project;
            self.provisional = false;
            let decision = self.decision(raw, false, 0, false, 0, true, false);
            self.commit(
                raw,
                raw,
                frames,
                input.continuity,
                input.session_frames,
                input.music,
            );
            return decision;
        }

        // ② PendingAbsolute 的解析：回退 ⇒ Event；继续绝对 ⇒ Project；仍停着 ⇒ 继续等。
        if self.domain == ClockDomain::PendingAbsolute {
            let rewound = delta <= -min_frames;
            let evidence = music_drop || same_session_edge || self.capture_origin_raw.is_some();
            if input.playing && rewound && !input.cycle_active && evidence {
                let origin = choose_origin(
                    self.pending_raw,
                    self.capture_origin_raw,
                    raw,
                    self.pending_music,
                    input.music,
                );
                // 【origin 可以大于相对读数】Event 起点是绝对位置，而 raw 是事件内
                // 偏移；只要求它不晚于预卷块（否则是另一个事件）。
                if origin >= 0 && origin <= self.pending_raw.saturating_add(max_error) {
                    self.domain = ClockDomain::Event {
                        origin,
                        raw_domain: RawDomain::Relative,
                    };
                    self.event_origin = Some(origin);
                    self.event_max_local = 0;
                    self.provisional = false;
                    let decision = self.decision(raw, true, -origin, false, raw, true, false);
                    self.commit(
                        raw,
                        raw,
                        frames,
                        input.continuity,
                        input.session_frames,
                        input.music,
                    );
                    return decision;
                }
            }
            if input.playing {
                self.domain = ClockDomain::Project;
                self.provisional = false;
                let decision = self.decision(raw, true, 0, false, 0, true, false);
                self.commit(
                    raw,
                    raw,
                    frames,
                    input.continuity,
                    input.session_frames,
                    input.music,
                );
                return decision;
            }
            if input.session_frames < probe_frames {
                // R2：宿主把位置钉住时，本地位置按上一块终点前进（预卷也是播放时间）。
                let local = if raw == self.last_raw {
                    previous_local_end
                } else {
                    raw
                };
                let decision = self.decision(local, false, 0, false, 0, false, true);
                self.commit(
                    local,
                    raw,
                    frames,
                    input.continuity,
                    input.session_frames,
                    input.music,
                );
                return decision;
            }
            // 探针到期：它确实是一次工程域停播（Track 停在光标处），如实落成 Project。
            self.domain = ClockDomain::Project;
            self.provisional = false;
            let decision = self.decision(raw, true, 0, false, 0, true, false);
            self.commit(
                raw,
                raw,
                frames,
                input.continuity,
                input.session_frames,
                input.music,
            );
            return decision;
        }

        // ③ 同一事件里“绝对读数 -> 相对读数”但本地位置连续：读数换了尺子，
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
                let decision = self.decision(raw, true, 0, false, 0, true, false);
                self.commit(
                    raw,
                    raw,
                    frames,
                    input.continuity,
                    input.session_frames,
                    input.music,
                );
                return decision;
            }
        }

        // ④ 相对 -> 绝对：映射回本地后与上一块终点连续。
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
                    let decision = self.decision(mapped, true, 0, false, 0, true, false);
                    self.commit(
                        mapped,
                        raw,
                        frames,
                        input.continuity,
                        input.session_frames,
                        input.music,
                    );
                    return decision;
                }
            }
        }

        // ⑤ 工程/未知域 -> 事件相对域：一次“换尺子”的巨大回退。
        //
        // 只允许有事件证据的场合：
        //   * 已知 origin；
        //   * 同一 Starting 会话里的 kPlaying 上升沿；
        //   * 本次 take 已记到预卷 origin，且音乐时间发生回退（VEGAS 的确定形状）。
        // Track FX 的普通 seek 没有这些证据。
        let provisional_shape = input.playing
            && self.event_origin.is_none()
            && (self.capture_origin_raw.is_none() || same_session_edge)
            && previous_local_end >= frames_for(rate, 2.0)
            && self.session_first_raw != Some(raw)
            && raw >= 0
            && raw < frames_for(rate, 5.0);
        let event_evidence = self.event_origin.is_some()
            || same_session_edge
            || (self.capture_origin_raw.is_some() && music_drop)
            || provisional_shape;
        if matches!(self.domain, ClockDomain::Unknown | ClockDomain::Project)
            && had
            && !input.cycle_active
            && continuous
            && delta <= -min_frames
            && event_evidence
            && (session_in_window || music_drop)
        {
            let origin = choose_origin(
                previous_local_end,
                self.capture_origin_raw,
                raw,
                self.last_music,
                input.music,
            );
            if origin >= 0 {
                self.domain = ClockDomain::Event {
                    origin,
                    raw_domain: RawDomain::Relative,
                };
                self.event_origin = Some(origin);
                self.event_max_local = 0;
                self.provisional = false;
                let decision = self.decision(raw, true, -origin, false, raw, true, false);
                self.commit(
                    raw,
                    raw,
                    frames,
                    input.continuity,
                    input.session_frames,
                    input.music,
                );
                return decision;
            }
        }

        // ⑥ 真 seek：没有域切换证据的位置跳变。
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
        let decision = self.decision(local, false, 0, seek, 0, true, false);
        self.commit(
            local,
            raw,
            frames,
            input.continuity,
            input.session_frames,
            input.music,
        );
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

    /// 记账：下一块的“上一块终点”。
    pub(crate) fn commit(
        &mut self,
        local_start: i64,
        raw: i64,
        frames: i64,
        continuity: Continuity,
        session_frames_before: u64,
        music: Option<f64>,
    ) {
        let end = local_start.saturating_add(frames);
        self.last_raw = raw;
        self.last_local_end = end;
        self.last_frames = u32::try_from(frames).unwrap_or(u32::MAX);
        self.session_frames = session_frames_before.saturating_add(frames.max(0) as u64);
        self.last_continuity = continuity;
        self.last_music = music;
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
        if matches!(
            self.domain,
            ClockDomain::Unknown | ClockDomain::PendingAbsolute
        ) {
            self.domain = ClockDomain::Project;
            self.provisional = false;
        }
    }

    /// 会话边界：保留实例级 `event_origin`，清空会话级读数。
    pub(crate) fn reset_session(&mut self) {
        self.capture_origin_raw = None;
        self.session_first_raw = None;
        self.pending_raw = 0;
        self.pending_music = None;
        self.last_music = None;
        self.last_raw = 0;
        self.last_local_end = 0;
        self.last_frames = 0;
        self.session_frames = 0;
        self.last_continuity = Continuity::Unknown;
        self.last_playing = false;
        self.start_window_open = false;
        // 【PendingAbsolute 不是实例级事实】它描述“这一次播放还没解析出来”；
        // 会话一断，证据就作废，否则下一次播放会继承一段不可发布的死状态。
        if self.domain == ClockDomain::PendingAbsolute {
            self.domain = ClockDomain::Unknown;
        }
        // 域保留：同一个 Event 的下一次播放仍在同一域；Track 仍是 Project。
        // 但 Event 域必须有 origin 支撑，否则如实回落到 Unknown。
        if self.domain.is_event() && self.event_origin.is_none() {
            self.domain = ClockDomain::Unknown;
        }
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
            capture_armed: true,
            transport_start: true,
        }
    }

    fn feed(state: &mut ClockState, mut block: ClockInput) -> ClockDecision {
        block.had_previous = state.last_frames > 0;
        block.session_new_epoch = false;
        block.session_phase = SessionPhase::Starting;
        block.session_frames = state.session_frames;
        state.observe(block)
    }

    fn event_origin(decision: ClockDecision) -> Option<i64> {
        match decision.domain {
            ClockDomain::Event { origin, .. } => Some(origin),
            _ => None,
        }
    }

    #[test]
    fn event_pre_roll_is_pending_not_project() {
        let mut state = ClockState::default();
        let first = state.observe(input(2_880_000, 848, false, false));
        assert_eq!(first.domain, ClockDomain::PendingAbsolute);
        assert!(!first.position_trusted);
        assert!(first.publish_deferred);
        assert_eq!(first.local_start, 2_880_000);
        assert_eq!(first.rebase_frames, 0);
    }

    #[test]
    fn event_switch_drops_disconnected_pre_roll() {
        let mut state = ClockState::default();
        let mut first = input(2_880_000, 848, false, false);
        first.capture_origin_raw = Some(2_880_000);
        first.music = Some(24_414.0);
        let pending = state.observe(first);
        assert_eq!(pending.domain, ClockDomain::PendingAbsolute);
        // 宿主把预卷位置钉住：本地位置按块长前进（与现场 pos=2,911,920 同形）。
        let mut pinned = input(2_880_000, 848, false, false);
        pinned.capture_origin_raw = Some(2_880_000);
        pinned.music = Some(24_414.0);
        let second = feed(&mut state, pinned);
        assert!(!second.position_trusted);
        assert_eq!(second.local_start, 2_880_848);
        // 第一个相对域块：raw=36,016、kPlaying=1、music 回退。
        let mut switched = input(36_016, 848, true, true);
        switched.capture_origin_raw = Some(2_880_000);
        switched.music = Some(1.5);
        let decision = feed(&mut state, switched);
        assert_eq!(event_origin(decision), Some(2_880_000));
        assert_eq!(decision.local_start, 36_016);
        assert_eq!(decision.rebase_frames, -2_880_000);
        assert_eq!(decision.prune_before, 36_016);
        assert!(decision.position_trusted);
        assert!(!decision.publish_deferred);
        assert!(!decision.seek);
    }

    #[test]
    fn event_switch_keeps_adjacent_pre_roll() {
        let mut state = ClockState::default();
        let mut first = input(2_880_000, 848, false, false);
        first.capture_origin_raw = Some(2_880_000);
        first.music = Some(24_414.0);
        state.observe(first);
        let mut switched = input(848, 848, true, true);
        switched.capture_origin_raw = Some(2_880_000);
        switched.music = Some(1.5);
        let decision = feed(&mut state, switched);
        assert_eq!(event_origin(decision), Some(2_880_000));
        assert_eq!(decision.local_start, 848);
        assert_eq!(decision.prune_before, 848);
    }

    #[test]
    fn event_origin_zero_is_legal() {
        assert!(origin_valid(0, 0, Some(0.0), Some(120.0), RATE));
        assert!(origin_valid(0, 24_000, Some(1.0), Some(120.0), RATE));
        let mut state = ClockState::default();
        state.event_origin = Some(0);
        let decision = state.observe(input(24_000, 1_000, true, false));
        assert_eq!(event_origin(decision), Some(0));
        assert_eq!(decision.local_start, 24_000);
    }

    #[test]
    fn track_pre_roll_resolves_project() {
        let mut state = ClockState::default();
        let first = state.observe(input(2_880_000, 848, false, false));
        assert_eq!(first.domain, ClockDomain::PendingAbsolute);
        let mut second = input(2_916_016, 848, true, true);
        second.music = Some(24_414.0);
        let decision = feed(&mut state, second);
        assert_eq!(decision.domain, ClockDomain::Project);
        assert_eq!(decision.local_start, 2_916_016);
        assert_eq!(decision.rebase_frames, 0);
        assert_eq!(decision.prune_before, 0);
        assert!(decision.position_trusted);
        assert!(!decision.publish_deferred);
    }

    #[test]
    fn track_seek_to_zero_is_not_event() {
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
    fn stale_zero_origin_is_rejected() {
        // 工程级 music=28635，raw=3,378,000：origin=0 会把整条 take 拉到工程 0。
        assert!(!origin_valid(
            0,
            3_378_000,
            Some(28_635.0),
            Some(120.0),
            RATE
        ));
        let mut state = ClockState::default();
        state.event_origin = Some(0);
        let mut first = input(3_378_000, 1_200, false, false);
        first.music = Some(28_635.0);
        first.tempo = Some(120.0);
        let decision = state.observe(first);
        assert_eq!(decision.domain, ClockDomain::PendingAbsolute);
        assert!(!decision.position_trusted);
    }

    #[test]
    fn event_mid_playback_keeps_offset() {
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
    fn event_sidecar_origin_is_not_end_minus_raw() {
        let mut state = ClockState::default();
        let mut first = input(906_000, 32_768, false, false);
        first.capture_origin_raw = Some(906_000);
        first.music = Some(24_414.0);
        state.observe(first);
        // 第一块相对读数被 lookbehind 暂存，下一块才是 38064。
        let mut second = input(38_064, 5_296, true, true);
        second.capture_origin_raw = Some(906_000);
        second.music = Some(1.5);
        let decision = feed(&mut state, second);
        assert_eq!(decision.rebase_frames, -906_000);
        // 若按 end - raw 会得到 900704，留下 5296 帧假洞。
        assert_ne!(decision.rebase_frames, -(938_768 - 38_064));
    }

    #[test]
    fn event_start_at_half_second() {
        let mut state = ClockState::default();
        // 事件起点 0.5 s；首块 0.5..1.1 s，下一块相对读数 0.6 s。
        let mut first = input(24_000, 28_800, false, false);
        first.capture_origin_raw = Some(24_000);
        first.music = Some(24_414.0);
        state.observe(first);
        let mut second = input(28_800, 24_000, true, true);
        second.capture_origin_raw = Some(24_000);
        second.music = Some(1.2);
        let decision = feed(&mut state, second);
        assert_eq!(decision.rebase_frames, -24_000);
        assert_eq!(decision.local_start, 28_800);
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
    fn a_playing_edge_from_paused_is_not_a_domain_switch() {
        let mut state = ClockState::default();
        state.confirm_project();
        state.observe(input(31 * 48_000, 1_000, true, true));
        let mut edge = input(0, 1_000, true, true);
        edge.session_new_epoch = true;
        edge.session_phase = SessionPhase::Paused;
        edge.capture_origin_raw = Some(31 * 48_000);
        let decision = state.observe(edge);
        // 已知 Event origin 时，首块按绝对/相对解析；没有 origin 时它必须是 Project。
        assert!(!decision.switched || decision.rebase_frames == 0);
    }

    #[test]
    fn pending_expires_to_project() {
        let mut state = ClockState::default();
        state.observe(input(2_880_000, 848, false, false));
        let mut block = input(2_880_000, 848, false, false);
        block.had_previous = true;
        block.session_new_epoch = false;
        block.session_frames = 48_000 * 3;
        let decision = state.observe(block);
        assert_eq!(decision.domain, ClockDomain::Project);
        assert!(decision.position_trusted);
    }

    #[test]
    fn origin_valid_accepts_relative_reading() {
        // Event 起点 2,880,000；相对读数 36,016（music 1.5 拍）。
        assert!(origin_valid(
            2_880_000,
            36_016,
            Some(1.5),
            Some(120.0),
            RATE
        ));
    }

    #[test]
    fn reset_session_drops_pending_and_keeps_event_origin() {
        let mut state = ClockState::default();
        state.observe(input(2_880_000, 848, false, false));
        state.reset_session();
        assert_eq!(state.domain, ClockDomain::Unknown);
        state.event_origin = Some(2_880_000);
        state.domain = ClockDomain::Event {
            origin: 2_880_000,
            raw_domain: RawDomain::Relative,
        };
        state.reset_session();
        assert_eq!(
            state.domain,
            ClockDomain::Event {
                origin: 2_880_000,
                raw_domain: RawDomain::Relative
            }
        );
    }
}
