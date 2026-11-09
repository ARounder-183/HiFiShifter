//! 处理 pass（ProcessingPass）：离线渲染与“冻结预览”的本地游标。
//!
//! 【为什么需要它】离线渲染（process_mode=2）没有可信的宿主播放头；VEGAS 的
//! “应用非实时事件 FX -> 预览”更隐蔽：它是 **process_mode=0 + kPlaying=0 +
//! projectTimeSamples 固定**，但输入音频在连续到达、内容在变化。旧实现把它当成
//! 实时停播，1 秒起播窗口之后冻结 —— 用户看到的是“只录了 1 秒”。
//!
//! # 预览从第一块开始
//!
//! 旧实现要求候选连续 0.75 秒才确认，确认之后才用 `rebase` 把已经写下的绝对段
//! 平移回事件域。现场 `rebases=0` 说明它根本没有确认；即使确认，平移也会
//! 留下不连续前缀。现在第一块就进入候选、就以本地 cursor 写入（事件起点时 = 0），
//! 确认只改变状态，不改变已经写下的位置。
//!
//! # 候选被证伪时回滚，而不是丢弃
//!
//! 预览与“真实传输的第一块”在首块上同形（`kPlaying=0` + 固定 raw）。候选期间
//! 按 cursor 写入；一旦下一块证明这是真实传输（`kPlaying` 升起或 raw 移动），
//! 就把整条 take **平移回 raw 域**（`rebase_take(+origin)`）。音频一帧不丢，
//! 也不需要丢弃已经录下的预卷 —— 这比“证伪就丢整条 take”安全得多。
use super::transport_session::SessionPhase;
use super::*;

/// 处理 pass 的种类。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum ProcessingKind {
    #[default]
    None,
    /// 显式离线渲染（process_mode=2）。
    Offline,
    /// 冻结预览（mode=0/1 + kPlaying=0 + raw 固定 + 内容连续）。
    FrozenPreview,
}

impl ProcessingKind {
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Offline => 1,
            Self::FrozenPreview => 2,
        }
    }

    pub(crate) fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::None),
            1 => Some(Self::Offline),
            2 => Some(Self::FrozenPreview),
            _ => None,
        }
    }
}

/// 预览 pass 的相位。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum PreviewPhase {
    #[default]
    Idle,
    /// 第一块就进入；cursor 从 0 或 Event 内偏移开始。
    Candidate,
    /// 0.75 s 确认后；cursor 连续。
    Active,
    /// `setProcessing(false)` / 回调中断：保留 cursor，等待恢复或 quiet timeout。
    Suspended,
}

impl PreviewPhase {
    fn code(self) -> u8 {
        match self {
            Self::Idle => 0,
            Self::Candidate => 1,
            Self::Active => 2,
            Self::Suspended => 3,
        }
    }

    fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Candidate,
            2 => Self::Active,
            3 => Self::Suspended,
            _ => Self::Idle,
        }
    }
}

/// 本块在 pass 里的结论。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProcessingBlock {
    pub(crate) kind: ProcessingKind,
    pub(crate) pass_id: u64,
    pub(crate) local_start: i64,
    /// 保留字段：确认不再平移（恒为 0）。候选被证伪时由调用方单独回滚。
    pub(crate) rebase_frames: i64,
    pub(crate) new_pass: bool,
}

/// 预览判定。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreviewDecision {
    /// 本块不是处理 pass（候选期间也走实时路径，由 HostClock 按 raw 写入）。
    None,
    /// 本块属于处理 pass。
    Block(ProcessingBlock),
}

/// 预览确认所需的连续时长（秒）。
pub(crate) const PREVIEW_CONFIRM_SEC: f64 = 0.75;

/// 预览候选至少需要的连续块数（防止单块抖动）。
pub(crate) const PREVIEW_MIN_BLOCKS: u32 = 2;

/// 候选在没有确认、也没有信号时最多保留多久（秒）。
///
/// 【为什么需要它】宿主停播后会持续送静音块；没有上界，候选会把整段停播写成
/// 一段越来越长的“预览”。超过这个时长仍没有内容，就回滚到 raw 域并退出。
pub(crate) const PREVIEW_CANDIDATE_CAP_SEC: f64 = 1.5;

/// 连续静音超过这个时长就结束预览（秒）。
pub(crate) const PREVIEW_SILENT_END_SEC: f64 = 2.0;

/// 连续“逐块相同且非零”的容忍块数；超过就判成停播重复投递。
pub(crate) const PREVIEW_REPEAT_TOLERANCE: u32 = 2;

/// 回调中断超过这个间隔就结束预览（毫秒）。**只由非音频线程使用**。
pub(crate) const PREVIEW_QUIET_MS: u64 = 2_000;

/// 处理 pass 的 epoch 高位。
///
/// 【为什么需要】`Capture::begin_transport_pass` 只比较 epoch 数值，而处理 pass 与
/// `TransportSession` 各有自己的计数器；两者都从 1 开始时会撞号，新 pass 被
/// 当成同一个 pass 而跳过封口/清 pending。把处理 pass 的高位钉死，命名空间就分开了。
pub(crate) const PROCESSING_PASS_BIT: u64 = 0x8000_0000_0000_0000;

/// “未知位置”的哨兵（与 `stream::NO_POSITION` 同值）。
const NO_POSITION: i64 = i64::MIN;

fn now_ms() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
        + 1
}

fn sane_rate(sample_rate: f64) -> f64 {
    if sample_rate.is_finite() && sample_rate > 0.0 {
        sample_rate
    } else {
        48_000.0
    }
}

/// 预览 pass 的原点。
///
/// 只采信通过合理性检查的 `event_origin`；否则以首块为原点（cursor = 0）。
/// 一次从宿主光标开始的预览绝不能把错误 origin 留给后续实时播放 —— 它只用于本 pass。
pub(crate) fn choose_preview_origin(
    event_origin: Option<i64>,
    first_raw: i64,
    music: Option<f64>,
    tempo: Option<f64>,
    rate: f64,
) -> i64 {
    match event_origin {
        Some(origin)
            if origin >= 0
                && first_raw >= origin
                && preview_origin_plausible(origin, first_raw, music, tempo, rate) =>
        {
            origin
        }
        _ => first_raw,
    }
}

/// 预览 origin 的合理性：不是工程级音乐时间、不超过 24 小时。
fn preview_origin_plausible(
    origin: i64,
    first_raw: i64,
    music: Option<f64>,
    tempo: Option<f64>,
    rate: f64,
) -> bool {
    if origin < 0 || first_raw < origin {
        return false;
    }
    let rate = sane_rate(rate);
    let offset = first_raw.saturating_sub(origin) as f64 / rate;
    if offset > 24.0 * 3600.0 {
        return false;
    }
    if let (Some(music), Some(tempo)) = (music, tempo) {
        if tempo.is_finite() && tempo > 0.0 {
            let expected = offset * tempo / 60.0;
            // 工程级音乐时间远大于事件内拍数：拒绝“origin 声称事件在 0、而宿主报的
            // 是工程级音乐时间”的组合。
            if music > expected + 1000.0 && music > 1000.0 {
                return false;
            }
        }
    }
    true
}

/// 一次处理 pass 的纯状态（音频线程只读写原子量）。
#[derive(Default)]
pub(crate) struct ProcessingPass {
    phase: AtomicU8,
    kind: AtomicU8,
    pass_id: AtomicU64,
    cursor_frames: AtomicI64,
    origin_raw: AtomicI64,
    first_raw: AtomicI64,
    sample_rate: AtomicU32,
    candidate_frames: AtomicU64,
    stable_blocks: AtomicU32,
    candidate_signal: AtomicBool,
    last_raw: AtomicI64,
    last_fingerprint: AtomicU64,
    fingerprint_valid: AtomicBool,
    last_mode: AtomicU8,
    last_at_ms: AtomicU64,
    repeat_run: AtomicU32,
    silent_frames: AtomicU64,
    suspended_phase: AtomicU8,
    /// 候选开始时 HostClock 是否已经在本地域（Event），以及它的 raw→local 偏移。
    ///
    /// 【为什么必须存下来】候选期间域可能从 Pending 落成 Project；确认时的域不能
    /// 代表候选块当时写在哪。存住它，确认才能算出正确的本地起点与是否需要平移。
    candidate_local: AtomicBool,
    candidate_offset: AtomicI64,
}

impl ProcessingPass {
    fn kind(&self) -> ProcessingKind {
        ProcessingKind::from_code(self.kind.load(Ordering::Acquire)).unwrap_or_default()
    }

    pub(crate) fn phase(&self) -> PreviewPhase {
        PreviewPhase::from_code(self.phase.load(Ordering::Acquire))
    }

    /// 候选或活动（`Suspended` 不算：它不在推进）。
    pub(crate) fn is_active(&self) -> bool {
        matches!(self.phase(), PreviewPhase::Candidate | PreviewPhase::Active)
    }

    /// 本 pass 是冻结预览（含 `Suspended`）。
    pub(crate) fn is_preview(&self) -> bool {
        self.kind() == ProcessingKind::FrozenPreview
    }

    pub(crate) fn is_suspended_preview(&self) -> bool {
        self.is_preview() && self.phase() == PreviewPhase::Suspended
    }

    pub(crate) fn is_offline(&self) -> bool {
        self.kind() == ProcessingKind::Offline && self.is_active()
    }

    #[cfg(test)]
    pub(crate) fn cursor_frames(&self) -> i64 {
        self.cursor_frames.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(crate) fn origin_raw(&self) -> Option<i64> {
        let value = self.origin_raw.load(Ordering::Acquire);
        (value != NO_POSITION).then_some(value)
    }

    pub(crate) fn pass_id(&self) -> u64 {
        self.pass_id.load(Ordering::Acquire)
    }

    /// 非音频线程的诊断读数：`(phase, origin, cursor)`。
    pub(crate) fn preview_state(&self) -> (u8, Option<i64>, i64) {
        let origin = self.origin_raw.load(Ordering::Acquire);
        (
            self.phase().code(),
            (origin != NO_POSITION).then_some(origin),
            self.cursor_frames.load(Ordering::Acquire),
        )
    }

    /// 显式离线块（process_mode=2）。返回本块的本地 cursor。
    pub(crate) fn note_offline_block(
        &self,
        mode: i32,
        raw_start: i64,
        frames: usize,
        sample_rate: f64,
        pass_sequence: &AtomicU64,
    ) -> Option<ProcessingBlock> {
        let previous_mode = self
            .last_mode
            .swap(mode.clamp(0, 2) as u8, Ordering::AcqRel) as i32;
        if mode != 2 {
            if previous_mode == 2 && self.kind() == ProcessingKind::Offline {
                self.end();
            }
            return None;
        }
        if !self.is_active() || self.kind() != ProcessingKind::Offline {
            self.begin_offline(raw_start, sample_rate, pass_sequence);
        }
        let position = self.cursor_frames.load(Ordering::Acquire);
        self.cursor_frames
            .fetch_add(frames as i64, Ordering::AcqRel);
        self.last_at_ms.store(now_ms(), Ordering::Release);
        Some(ProcessingBlock {
            kind: ProcessingKind::Offline,
            pass_id: self.pass_id(),
            local_start: position,
            rebase_frames: 0,
            new_pass: position == 0,
        })
    }

    /// 每个实时块调用一次：识别并推进“冻结预览”。
    ///
    /// 【调用顺序】必须在 HostClock 之前。否则 mode=0 预览会先被实时起播窗口接走，
    /// 1 秒后冻结，永远走不到这里。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn note_preview_block(
        &self,
        mode: i32,
        raw_start: i64,
        frames: usize,
        sample_rate: f64,
        fingerprint: u64,
        silent: bool,
        playing: bool,
        allow_preview: bool,
        phase_before: SessionPhase,
        capture_started: bool,
        setup_mode: i32,
        event_origin: Option<i64>,
        music: Option<f64>,
        tempo: Option<f64>,
        clock_is_local: bool,
        clock_offset: i64,
        clock_pending: bool,
        pass_sequence: &AtomicU64,
    ) -> PreviewDecision {
        if frames == 0 || mode == 2 {
            return PreviewDecision::None;
        }
        let rate = sane_rate(sample_rate);
        let now = now_ms();
        let mut phase = self.phase();

        // ── Suspended：raw/mode 未变就恢复，变了就放弃 ────────────────────
        if phase == PreviewPhase::Suspended {
            let last_raw = self.last_raw.load(Ordering::Acquire);
            if (mode == 0 || mode == 1) && (last_raw == NO_POSITION || last_raw == raw_start) {
                phase = PreviewPhase::from_code(self.suspended_phase.load(Ordering::Acquire));
                if phase == PreviewPhase::Idle {
                    phase = PreviewPhase::Active;
                }
                self.phase.store(phase.code(), Ordering::Release);
                self.last_at_ms.store(now, Ordering::Release);
            } else {
                self.end();
                return PreviewDecision::None;
            }
        }

        // ── Candidate / Active ───────────────────────────────────────────
        if matches!(phase, PreviewPhase::Candidate | PreviewPhase::Active) {
            if playing || (mode != 0 && mode != 1) {
                self.end();
                return PreviewDecision::None;
            }
            let last_raw = self.last_raw.load(Ordering::Acquire);
            if last_raw != NO_POSITION && last_raw != raw_start {
                self.end();
                return PreviewDecision::None;
            }

            // 重复块容忍：0.5 s 唯一内容循环两次时，相邻块并不相同；只有真正
            // “逐块完全相同”的停播重复投递才会累积。
            let previous_valid = self.fingerprint_valid.swap(true, Ordering::AcqRel);
            let previous = self.last_fingerprint.swap(fingerprint, Ordering::AcqRel);
            let same = previous_valid && previous == fingerprint;
            let repeat = if same && !silent {
                self.repeat_run.fetch_add(1, Ordering::AcqRel) + 1
            } else {
                self.repeat_run.store(0, Ordering::Release);
                0
            };
            if silent {
                self.silent_frames
                    .fetch_add(frames as u64, Ordering::AcqRel);
            } else {
                self.silent_frames.store(0, Ordering::Release);
            }
            if repeat > PREVIEW_REPEAT_TOLERANCE {
                self.end();
                return PreviewDecision::None;
            }

            if phase == PreviewPhase::Candidate {
                let stable = self.stable_blocks.fetch_add(1, Ordering::AcqRel) + 1;
                let prior = self.candidate_frames.load(Ordering::Acquire);
                let candidate = prior.saturating_add(frames as u64);
                self.candidate_frames.store(candidate, Ordering::Release);
                if !silent {
                    self.candidate_signal.store(true, Ordering::Release);
                }
                let confirm_frames = (rate * PREVIEW_CONFIRM_SEC).round().max(1.0) as u64;
                let cap_frames = (rate * PREVIEW_CANDIDATE_CAP_SEC).round().max(1.0) as u64;
                if stable >= PREVIEW_MIN_BLOCKS
                    && candidate >= confirm_frames
                    && self.candidate_signal.load(Ordering::Acquire)
                    && !clock_pending
                {
                    // 【确认：一次性把已经写下的 raw 段平移回本地域】
                    // 候选期间块由 HostClock 按 raw 写入（Event 预卷因此仍然走
                    // PendingAbsolute，不会被预览抢走）；确认时整条 take 平移
                    // -origin，第一个候选块正好落在 cursor。
                    self.phase
                        .store(PreviewPhase::Active.code(), Ordering::Release);
                    let first_raw = self.first_raw.load(Ordering::Acquire);
                    let origin = self.origin_raw.load(Ordering::Acquire);
                    let origin = if origin == NO_POSITION {
                        first_raw
                    } else {
                        origin
                    };
                    let candidate_local = self.candidate_local.load(Ordering::Acquire);
                    let candidate_offset = self.candidate_offset.load(Ordering::Acquire);
                    let base = if candidate_local {
                        first_raw.saturating_add(candidate_offset)
                    } else {
                        first_raw.saturating_sub(origin)
                    }
                    .max(0);
                    let local = base.saturating_add(prior as i64);
                    self.cursor_frames
                        .store(local.saturating_add(frames as i64), Ordering::Release);
                    self.last_at_ms.store(now, Ordering::Release);
                    self.last_raw.store(raw_start, Ordering::Release);
                    return PreviewDecision::Block(ProcessingBlock {
                        kind: ProcessingKind::FrozenPreview,
                        pass_id: self.pass_id(),
                        local_start: local,
                        // 时钟已经在 Event 域里写本地位置（raw - origin）时不能再平移；
                        // 只有候选块按 raw 写入（Project / Pending）才需要 -origin。
                        rebase_frames: if candidate_local { 0 } else { -origin },
                        new_pass: true,
                    });
                }
                if !clock_pending
                    && candidate > cap_frames
                    && !self.candidate_signal.load(Ordering::Acquire)
                {
                    // 候选一直没有内容：宿主在停播时送静音块。
                    self.end();
                    return PreviewDecision::None;
                }
                // 候选期间不写 cursor：块由 HostClock 按 raw 写入。
                self.last_at_ms.store(now, Ordering::Release);
                self.last_raw.store(raw_start, Ordering::Release);
                return PreviewDecision::None;
            }

            let silent_limit = (rate * PREVIEW_SILENT_END_SEC).round().max(1.0) as u64;
            if self.silent_frames.load(Ordering::Acquire) > silent_limit {
                self.end();
                return PreviewDecision::None;
            }

            let position = self.cursor_frames.load(Ordering::Acquire);
            self.cursor_frames
                .fetch_add(frames as i64, Ordering::AcqRel);
            self.last_at_ms.store(now, Ordering::Release);
            self.last_raw.store(raw_start, Ordering::Release);
            return PreviewDecision::Block(ProcessingBlock {
                kind: ProcessingKind::FrozenPreview,
                pass_id: self.pass_id(),
                local_start: position,
                rebase_frames: 0,
                new_pass: false,
            });
        }

        // ── Idle：候选首块（不写 cursor，块仍走 HostClock）─────────────────
        if !allow_preview || playing || (mode != 0 && mode != 1) {
            return PreviewDecision::None;
        }
        let eligible = setup_mode == 2
            || phase_before == SessionPhase::Paused
            || (!capture_started && phase_before != SessionPhase::Rolling);
        if !eligible {
            return PreviewDecision::None;
        }
        let origin = choose_preview_origin(event_origin, raw_start, music, tempo, rate);
        let cursor = if clock_is_local {
            raw_start.saturating_add(clock_offset)
        } else {
            raw_start.saturating_sub(origin)
        }
        .max(0);
        self.begin_preview(
            origin,
            raw_start,
            rate,
            cursor,
            clock_is_local,
            clock_offset,
            pass_sequence,
        );
        self.candidate_signal.store(!silent, Ordering::Release);
        self.fingerprint_valid.store(true, Ordering::Release);
        self.last_fingerprint.store(fingerprint, Ordering::Release);
        if silent {
            self.silent_frames.store(frames as u64, Ordering::Release);
        }
        // 【首块也要计入候选帧数】确认时 prior 是“本块之前的候选帧数”。
        self.candidate_frames
            .store(frames as u64, Ordering::Release);
        self.last_at_ms.store(now, Ordering::Release);
        PreviewDecision::None
    }

    fn begin_offline(&self, origin_raw: i64, sample_rate: f64, pass_sequence: &AtomicU64) {
        self.begin(
            ProcessingKind::Offline,
            origin_raw,
            sample_rate,
            0,
            pass_sequence,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_preview(
        &self,
        origin_raw: i64,
        first_raw: i64,
        sample_rate: f64,
        cursor: i64,
        clock_is_local: bool,
        clock_offset: i64,
        pass_sequence: &AtomicU64,
    ) {
        self.begin(
            ProcessingKind::FrozenPreview,
            origin_raw,
            sample_rate,
            cursor,
            pass_sequence,
        );
        self.first_raw.store(first_raw, Ordering::Release);
        self.last_raw.store(first_raw, Ordering::Release);
        self.phase
            .store(PreviewPhase::Candidate.code(), Ordering::Release);
        self.silent_frames.store(0, Ordering::Release);
        self.candidate_local
            .store(clock_is_local, Ordering::Release);
        self.candidate_offset.store(clock_offset, Ordering::Release);
    }

    fn begin(
        &self,
        kind: ProcessingKind,
        origin_raw: i64,
        sample_rate: f64,
        cursor: i64,
        pass_sequence: &AtomicU64,
    ) {
        let sequence = pass_sequence.fetch_add(1, Ordering::AcqRel) + 1;
        self.pass_id
            .store(PROCESSING_PASS_BIT | sequence, Ordering::Release);
        self.kind.store(kind.code(), Ordering::Release);
        self.origin_raw.store(origin_raw, Ordering::Release);
        self.first_raw.store(origin_raw, Ordering::Release);
        self.cursor_frames.store(cursor, Ordering::Release);
        self.candidate_frames.store(0, Ordering::Release);
        self.stable_blocks.store(1, Ordering::Release);
        self.candidate_signal.store(false, Ordering::Release);
        self.fingerprint_valid.store(false, Ordering::Release);
        self.repeat_run.store(0, Ordering::Release);
        self.silent_frames.store(0, Ordering::Release);
        self.last_raw.store(NO_POSITION, Ordering::Release);
        if sample_rate.is_finite() && sample_rate > 0.0 {
            self.sample_rate
                .store(sample_rate.round() as u32, Ordering::Release);
        }
        self.last_at_ms.store(now_ms(), Ordering::Release);
        // 离线从 Active 开始；预览由调用方覆盖成 Candidate。
        self.phase
            .store(PreviewPhase::Active.code(), Ordering::Release);
    }

    /// `setProcessing(false)` / 组件失活：保留 cursor 与 origin。
    pub(crate) fn suspend(&self) {
        let phase = self.phase();
        if matches!(phase, PreviewPhase::Candidate | PreviewPhase::Active) {
            self.suspended_phase.store(phase.code(), Ordering::Release);
            self.phase
                .store(PreviewPhase::Suspended.code(), Ordering::Release);
            self.last_at_ms.store(now_ms(), Ordering::Release);
        }
    }

    /// `setProcessing(true)` / 组件重新激活：回到挂起前的相位。
    pub(crate) fn resume(&self) {
        if self.phase() != PreviewPhase::Suspended {
            return;
        }
        let mut phase = PreviewPhase::from_code(self.suspended_phase.load(Ordering::Acquire));
        if phase == PreviewPhase::Idle {
            phase = PreviewPhase::Active;
        }
        self.phase.store(phase.code(), Ordering::Release);
        self.last_at_ms.store(now_ms(), Ordering::Release);
    }

    /// 非音频线程：挂起/静止超过 `PREVIEW_QUIET_MS` 就结束预览。
    ///
    /// 返回 true 表示本次调用结束了预览，调用方应 promote 当前 take 并清时钟。
    pub(crate) fn end_if_quiet(&self) -> bool {
        if !self.is_preview() {
            return false;
        }
        let phase = self.phase();
        if phase == PreviewPhase::Idle {
            return false;
        }
        let last = self.last_at_ms.load(Ordering::Acquire);
        if last == 0 || now_ms().saturating_sub(last) <= PREVIEW_QUIET_MS {
            return false;
        }
        self.end();
        true
    }

    pub(crate) fn end(&self) {
        self.phase
            .store(PreviewPhase::Idle.code(), Ordering::Release);
        self.kind
            .store(ProcessingKind::None.code(), Ordering::Release);
        self.cursor_frames.store(0, Ordering::Release);
        self.origin_raw.store(NO_POSITION, Ordering::Release);
        self.first_raw.store(NO_POSITION, Ordering::Release);
        self.candidate_frames.store(0, Ordering::Release);
        self.stable_blocks.store(0, Ordering::Release);
        self.candidate_signal.store(false, Ordering::Release);
        self.fingerprint_valid.store(false, Ordering::Release);
        self.repeat_run.store(0, Ordering::Release);
        self.silent_frames.store(0, Ordering::Release);
        self.last_raw.store(NO_POSITION, Ordering::Release);
        self.last_mode.store(0, Ordering::Release);
        self.last_at_ms.store(0, Ordering::Release);
        self.suspended_phase.store(0, Ordering::Release);
    }
}

impl ExtensionOwner {
    /// 记录 setupProcessing 的 process mode。
    pub(crate) fn set_setup_process_mode(&self, mode: i32) {
        self.setup_process_mode.store(mode, Ordering::Release);
    }

    /// 宿主声明的进程模式（0 = 实时，1 = 预取，2 = 离线）。
    ///
    /// 【为什么公开】它是两种格式都要在诊断里对齐的事实，也是"离线准备到底有没有
    /// 发生"的唯一直接读数（外壳的集成测试据此断言 `clap.render.set` 真的走到了
    /// 共享 setup 序列）。
    pub fn setup_process_mode(&self) -> i32 {
        self.setup_process_mode.load(Ordering::Acquire)
    }

    /// 每个 process 块调用一次：离线时返回本块本地 cursor。
    ///
    /// 【为什么只在未绑定实例上生效】ARA 的离线渲染由文档/区域驱动，仍应使用宿主
    /// 给的 projectTimeSamples；Capture 才需要自己的 0 起点。
    ///
    /// 【为什么参数是中立的三个事实】它此前吃 VST3 的 ProcessContext —— 共享层
    /// 因此认识了一种格式的 ABI，CLAP 想接就只能伪造一个 VST3 结构（于是它干脆
    /// 没接，见 render_block.rs 的 observe_input_block）。离线 pass 只需要"宿主报的
    /// 工程位置 / 本块采样率 / 进程模式 / 帧数"，四个数与格式无关。
    pub(crate) fn note_offline_block(
        &self,
        raw_start: i64,
        sample_rate: f64,
        mode: i32,
        frames: usize,
    ) -> Option<i64> {
        if self.is_bound() || self.bound_once.load(Ordering::Acquire) {
            return None;
        }
        self.processing_pass
            .note_offline_block(mode, raw_start, frames, sample_rate, &self.pass_sequence)
            .map(|block| block.local_start)
    }

    /// 宿主停止处理：捕获暂停、处理 pass 结束、传输时钟冻结。
    pub(crate) fn note_processing_stopped(&self) {
        self.processing_pass.end();
        self.capture().set_preview_active(false);
        self.capture().pause();
        self.reset_transport_session("processing_stopped");
        if let Some(clock) = self.clock.get() {
            clock.freeze_transport();
            clock.clear_capture();
        }
    }

    /// 组件失活：预览只挂起（保留 cursor），其它通路与停止处理同一处置。
    pub(crate) fn note_component_deactivated(&self) {
        if self.processing_pass.is_preview() {
            self.suspend_processing_preview("component_deactivated");
        } else {
            self.note_processing_stopped();
        }
    }

    /// 预览挂起：保留 cursor/origin/phase，只冻结 take 与 UI。
    ///
    /// 【为什么不能走 `note_processing_stopped`】VEGAS 的非实时预览完全可能在
    /// 每个 1 s chunk 之间调用 `setProcessing(false)`。结束 pass 会让 cursor 清零、
    /// 流几何清空，下一块只能重新从候选开始 —— 永远接不上，看起来就是“只录了 1 秒”。
    pub(crate) fn suspend_processing_preview(&self, reason: &str) {
        let _ = reason;
        self.processing_pass.suspend();
        self.capture().pause();
        if let Some(clock) = self.clock.get() {
            clock.hold_capture();
            clock.freeze_transport();
        }
    }

    /// 预览恢复；返回 true 表示确实恢复了一个挂起的预览。
    pub(crate) fn resume_processing_preview(&self) -> bool {
        if !self.processing_pass.is_suspended_preview() {
            return false;
        }
        self.processing_pass.resume();
        self.capture().set_preview_active(true);
        true
    }

    /// 非音频线程心跳：挂起/静止超过 2 秒就结束预览并 promote。
    ///
    /// 【为什么公开】它是共享生命周期序列的一环（`editor::session` 的 actor 心跳里
    /// 与 `capture().service()` 并列调用）。格式外壳的集成测试要按真实心跳复现宿主
    /// 停播行为，就必须能调到同一条入口，而不是复制一份"安静判据"。
    pub fn tick_preview_quiet(&self) {
        if !self.processing_pass.end_if_quiet() {
            return;
        }
        match self.capture().end_preview() {
            Ok(id) => {
                log::warn!("[capture] preview pass ended quietly; take {id} promoted");
                // 新 take 需要一次合成；actor 心跳会消费这个意图并排一次渲染。
                self.render_wanted.store(true, Ordering::Release);
            }
            Err(error) => log::debug!("[capture] preview pass ended quietly: {error}"),
        }
        if let Some(clock) = self.clock.get() {
            clock.clear_capture();
        }
        self.reset_transport_session("preview_quiet");
    }

    /// 离线/预览 pass 是否活跃。
    pub(crate) fn offline_pass_active(&self) -> bool {
        self.processing_pass.is_active()
    }

    /// 处理 pass（候选/活动/挂起预览）是否占用本实例。
    pub(crate) fn processing_pass_active(&self) -> bool {
        self.processing_pass.is_active() || self.processing_pass.is_suspended_preview()
    }

    /// 本实例当前是否处于冻结预览（含挂起）。
    pub(crate) fn processing_pass_is_preview(&self) -> bool {
        self.processing_pass.is_preview()
    }

    /// 用户动作（开始/停止/取消/清空捕获）结束当前处理 pass。
    pub(crate) fn end_processing_pass(&self) {
        self.processing_pass.end();
        self.capture().set_preview_active(false);
    }

    /// 实例级事件起点（组件 state 恢复 / 首次域切换学到）。
    pub(crate) fn event_origin_samples(&self) -> Option<i64> {
        // 0 是“未知”的哨兵；存储值是 origin + 1。origin 本身可以是 0（Event 在工程 0）。
        let value = self.event_origin_samples.load(Ordering::Acquire);
        (value != 0).then_some(value - 1)
    }

    /// 记录事件起点。
    ///
    /// 【为什么需要置信度】第一次学到的值不一定是真的：复制/粘贴 Event FX 会把上
    /// 一个事件的 origin 带过来，误判也会写下一个 0。`confidence = 2`（组件 state /
    /// 确认过的域切换）可以覆盖旧值；`confidence = 1` 只在完全未知时写入。
    pub(crate) fn note_event_origin(&self, origin: i64, confidence: u8) {
        if origin < 0 {
            return;
        }
        let current = self.event_origin_samples.load(Ordering::Acquire);
        let current_confidence = self.event_origin_confidence.load(Ordering::Acquire);
        if confidence >= 2 && current_confidence < 2 {
            self.event_origin_samples
                .store(origin.saturating_add(1), Ordering::Release);
            self.event_origin_confidence.store(2, Ordering::Release);
        } else if confidence == 1 && current == 0 {
            self.event_origin_samples
                .store(origin.saturating_add(1), Ordering::Release);
            self.event_origin_confidence.store(1, Ordering::Release);
        }
    }

    /// 是否允许把“mode=0 + kPlaying=0 + raw 固定 + 内容连续”识别成冻结预览。
    ///
    /// 【为什么不再依赖 initialize 的宿主名】包装器宿主（sfvstwrap）会让
    /// `getName` 返回 unknown；身份现在在 `Processor::create` / `setupProcessing`
    /// 解析，那两处一定会被调用。
    ///
    /// 【为什么不做通用兜底】同样的形状在别的宿主里也可能是停播/预卷；通用兜底
    /// 会把 Track 的预卷误判成预览，改变选举与输出语义。预览只对 VEGAS 家族与
    /// setup mode 2 开放。
    pub(crate) fn capture_preview_allowed(&self) -> bool {
        if self.is_bound() || self.bound_once.load(Ordering::Acquire) {
            return false;
        }
        // 【预览是捕获功能】未武装时识别预览只会让处理 pass 抢走 HostClock 与
        // 位置发布，却录不到任何东西。
        if !self.capture().is_armed() {
            return false;
        }
        if self.setup_process_mode() == 2 {
            return true;
        }
        // 【只对 VEGAS 家族开放】“mode 0 + kPlaying=0 + raw 固定”在别的宿主里
        // 也可能是停播/预卷的形状；通用兜底会把 Track 的预卷误判成预览，进而
        // 改变选举与输出语义。身份现在在 create/setup 解析，不再依赖 initialize。
        crate::host_identity::is_vegas_family()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    fn sequence() -> AtomicU64 {
        AtomicU64::new(0)
    }

    #[allow(clippy::too_many_arguments)]
    fn feed(
        pass: &ProcessingPass,
        sequence: &AtomicU64,
        raw: i64,
        frames: usize,
        fingerprint: u64,
        silent: bool,
        playing: bool,
        phase_before: SessionPhase,
        capture_started: bool,
        origin: Option<i64>,
        music: Option<f64>,
    ) -> PreviewDecision {
        pass.note_preview_block(
            0,
            raw,
            frames,
            48_000.0,
            fingerprint,
            silent,
            playing,
            true,
            phase_before,
            capture_started,
            0,
            origin,
            music,
            Some(120.0),
            false,
            0,
            false,
            sequence,
        )
    }

    fn candidate_block(
        pass: &ProcessingPass,
        sequence: &AtomicU64,
        raw: i64,
        fingerprint: u64,
    ) -> PreviewDecision {
        feed(
            pass,
            sequence,
            raw,
            1_200,
            fingerprint,
            false,
            false,
            SessionPhase::Paused,
            false,
            None,
            None,
        )
    }

    #[test]
    fn candidate_starts_on_the_first_block_without_taking_over_the_clock() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        let decision = candidate_block(&pass, &sequence, 3_378_000, 1);
        assert_eq!(decision, PreviewDecision::None, "候选首块仍走实时路径");
        assert_eq!(pass.phase(), PreviewPhase::Candidate);
        assert!(pass.is_preview());
        assert_eq!(pass.origin_raw(), Some(3_378_000));
        assert_eq!(pass.cursor_frames(), 0);
    }

    #[test]
    fn candidate_confirms_after_three_quarters_of_a_second() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        let mut confirmed = None;
        for index in 0..45 {
            if let PreviewDecision::Block(block) =
                candidate_block(&pass, &sequence, 3_378_000, index as u64 + 1)
            {
                confirmed = Some(block);
                break;
            }
        }
        let block = confirmed.expect("0.75 s 之后必须确认预览");
        assert!(block.new_pass);
        assert_eq!(block.rebase_frames, -3_378_000);
        // 第 30 块（index 29）确认：它从 29 * 1200 = 34,800 开始。
        assert_eq!(block.local_start, 34_800);
        assert_eq!(pass.phase(), PreviewPhase::Active);
        assert_eq!(pass.cursor_frames(), 36_000);
    }

    #[test]
    fn candidate_survives_a_repeated_half_second_loop() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        for index in 0..40 {
            let _ = feed(
                &pass,
                &sequence,
                3_378_000,
                1_200,
                (index % 20) as u64,
                false,
                false,
                SessionPhase::Paused,
                false,
                None,
                None,
            );
        }
        assert_eq!(pass.phase(), PreviewPhase::Active);
        assert_eq!(pass.cursor_frames(), 48_000);
    }

    #[test]
    fn candidate_uses_first_raw_when_origin_is_unknown() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        let _ = candidate_block(&pass, &sequence, 3_378_000, 1);
        assert_eq!(pass.origin_raw(), Some(3_378_000));
        assert_eq!(pass.cursor_frames(), 0);
    }

    #[test]
    fn candidate_uses_the_event_origin_when_valid() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        let _ = feed(
            &pass,
            &sequence,
            3_378_000,
            1_200,
            0,
            false,
            false,
            SessionPhase::Paused,
            false,
            Some(2_880_000),
            None,
        );
        assert_eq!(pass.origin_raw(), Some(2_880_000));
        assert_eq!(pass.cursor_frames(), 498_000);
    }

    #[test]
    fn candidate_rejects_a_stale_zero_origin() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        let _ = feed(
            &pass,
            &sequence,
            3_378_000,
            1_200,
            0,
            false,
            false,
            SessionPhase::Paused,
            false,
            Some(0),
            Some(28_635.0),
        );
        assert_eq!(pass.origin_raw(), Some(3_378_000));
        assert_eq!(pass.cursor_frames(), 0);
    }

    #[test]
    fn preview_survives_set_processing_false() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        for index in 0..40 {
            let _ = feed(
                &pass,
                &sequence,
                3_378_000,
                1_200,
                index,
                false,
                false,
                SessionPhase::Paused,
                false,
                None,
                None,
            );
        }
        let before = pass.cursor_frames();
        assert_eq!(before, 48_000);
        pass.suspend();
        assert!(pass.is_suspended_preview());
        pass.resume();
        assert_eq!(pass.phase(), PreviewPhase::Active);
        let block = match feed(
            &pass,
            &sequence,
            3_378_000,
            1_200,
            99,
            false,
            false,
            SessionPhase::Paused,
            false,
            None,
            None,
        ) {
            PreviewDecision::Block(block) => block,
            other => panic!("expected a block, got {other:?}"),
        };
        assert_eq!(block.local_start, before);
        assert_eq!(pass.cursor_frames(), before + 1_200);
    }

    #[test]
    fn preview_ends_on_a_playing_edge() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        for index in 0..4 {
            let _ = feed(
                &pass,
                &sequence,
                3_378_000,
                1_200,
                index,
                false,
                false,
                SessionPhase::Paused,
                false,
                None,
                None,
            );
        }
        let decision = feed(
            &pass,
            &sequence,
            3_378_000,
            1_200,
            9,
            false,
            true,
            SessionPhase::Paused,
            false,
            None,
            None,
        );
        assert_eq!(decision, PreviewDecision::None);
        assert!(!pass.is_preview());
    }

    #[test]
    fn preview_ends_when_raw_moves() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        let _ = candidate_block(&pass, &sequence, 3_378_000, 1);
        let decision = feed(
            &pass,
            &sequence,
            3_379_200,
            1_200,
            2,
            false,
            false,
            SessionPhase::Paused,
            false,
            None,
            None,
        );
        assert_eq!(decision, PreviewDecision::None);
        assert!(!pass.is_preview());
    }

    #[test]
    fn a_silent_candidate_ends_after_the_cap() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        for index in 0..200 {
            let _ = feed(
                &pass,
                &sequence,
                3_378_000,
                1_200,
                index,
                true,
                false,
                SessionPhase::Paused,
                false,
                None,
                None,
            );
            if !pass.is_preview() {
                return;
            }
        }
        panic!("a silent candidate must not grow forever");
    }

    #[test]
    fn preview_ten_seconds_has_480k_frames() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        for index in 0..400 {
            let _ = feed(
                &pass,
                &sequence,
                3_378_000,
                1_200,
                index,
                false,
                false,
                SessionPhase::Paused,
                false,
                None,
                None,
            );
        }
        assert_eq!(pass.cursor_frames(), 480_000);
        assert_eq!(pass.phase(), PreviewPhase::Active);
    }

    #[test]
    fn offline_cursor_starts_at_zero_and_advances_by_block_frames() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        assert_eq!(
            pass.note_offline_block(2, 1_500_000, 512, 48_000.0, &sequence)
                .map(|block| block.local_start),
            Some(0)
        );
        assert_eq!(
            pass.note_offline_block(2, 1_500_000, 512, 48_000.0, &sequence)
                .map(|block| block.local_start),
            Some(512)
        );
        assert_eq!(pass.cursor_frames(), 1_024);
        assert_eq!(
            pass.note_offline_block(0, 0, 512, 48_000.0, &sequence),
            None
        );
        assert!(!pass.is_active());
    }

    #[test]
    fn processing_pass_ids_do_not_collide_with_transport_epochs() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        let _ = pass.note_offline_block(2, 0, 512, 48_000.0, &sequence);
        let first = pass.pass_id();
        pass.end();
        let _ = pass.note_offline_block(2, 0, 512, 48_000.0, &sequence);
        let second = pass.pass_id();
        assert_ne!(first, second);
        assert_ne!(first & PROCESSING_PASS_BIT, 0);
        assert_ne!(second & PROCESSING_PASS_BIT, 0);
        assert!(first > 1 << 32);
    }
}
