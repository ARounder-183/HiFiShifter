//! 处理 pass（ProcessingPass）：离线渲染与“冻结预览”的本地游标。
//!
//! 【为什么需要它】离线渲染（process_mode=2）没有可信的宿主播放头；VEGAS 的
//! “应用非实时事件 FX -> 预览”更隐蔽：它是 **process_mode=0 + kPlaying=0 +
//! projectTimeSamples 固定**，但输入音频在连续到达、内容在变化。旧实现把它当成
//! 实时停播，1 秒起播窗口之后冻结 —— 用户看到的是“只录了 1 秒”。
//!
//! 这里把两种形状收敛成同一个 pass 概念：每个 pass 有自己的 `pass_id` 和从 0 开始的
//! cursor；pass 内所有块（包括候选首块）都读 cursor。离线与预览的区别只在结束条件
//! 与输出监听策略。
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

/// 本块在 pass 里的结论。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProcessingBlock {
    pub(crate) kind: ProcessingKind,
    pub(crate) pass_id: u64,
    pub(crate) local_start: i64,
    /// 确认预览时把已经写下的候选段平移回事件本地域。
    pub(crate) rebase_frames: i64,
    pub(crate) new_pass: bool,
}

/// 预览确认所需的连续时长（秒）。
pub(crate) const PREVIEW_CONFIRM_SEC: f64 = 0.75;

/// 预览候选至少需要的连续块数（防止单块抖动）。
pub(crate) const PREVIEW_MIN_BLOCKS: u32 = 2;

/// 回调中断超过这个间隔就结束预览（毫秒）。
pub(crate) const PREVIEW_GAP_MS: u64 = 250;

fn now_ms() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
        + 1
}

/// 一次处理 pass 的纯状态（音频线程只读写原子量）。
#[derive(Default)]
pub(crate) struct ProcessingPass {
    active: AtomicBool,
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
}

impl ProcessingPass {
    fn kind(&self) -> ProcessingKind {
        ProcessingKind::from_code(self.kind.load(Ordering::Acquire)).unwrap_or_default()
    }

    pub(crate) fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    pub(crate) fn is_preview(&self) -> bool {
        self.kind() == ProcessingKind::FrozenPreview
    }

    pub(crate) fn is_offline(&self) -> bool {
        self.kind() == ProcessingKind::Offline
    }

    #[cfg(test)]
    pub(crate) fn cursor_frames(&self) -> i64 {
        self.cursor_frames.load(Ordering::Acquire)
    }

    pub(crate) fn pass_id(&self) -> u64 {
        self.pass_id.load(Ordering::Acquire)
    }

    /// 显式离线块（process_mode=2）。返回本块的本地 cursor。
    pub(crate) fn note_offline_block(
        &self,
        mode: i32,
        raw_start: i64,
        frames: usize,
        sample_rate: f64,
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
        if !self.active.load(Ordering::Acquire) || self.kind() != ProcessingKind::Offline {
            self.begin(ProcessingKind::Offline, raw_start, sample_rate, 0);
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
        repeated_nonzero: bool,
        playing: bool,
        allow_preview: bool,
        event_origin: Option<i64>,
    ) -> Option<ProcessingBlock> {
        if frames == 0 || mode == 2 {
            return None;
        }
        // 已经在预览：kPlaying 升起、模式离开、或回调中断 ⇒ 结束。
        if self.kind() == ProcessingKind::FrozenPreview && self.active.load(Ordering::Acquire) {
            let last_at = self.last_at_ms.load(Ordering::Acquire);
            let now = now_ms();
            if playing
                || (mode != 0 && mode != 1)
                || (last_at != 0 && now.saturating_sub(last_at) > PREVIEW_GAP_MS)
            {
                self.end();
                return None;
            }
            let position = self.cursor_frames.load(Ordering::Acquire);
            self.cursor_frames
                .fetch_add(frames as i64, Ordering::AcqRel);
            self.last_at_ms.store(now, Ordering::Release);
            self.last_raw.store(raw_start, Ordering::Release);
            return Some(ProcessingBlock {
                kind: ProcessingKind::FrozenPreview,
                pass_id: self.pass_id(),
                local_start: position,
                rebase_frames: 0,
                new_pass: false,
            });
        }

        if !allow_preview || playing || (mode != 0 && mode != 1) {
            self.reset_candidate();
            return None;
        }

        // 候选必须固定 raw；一移动就说明宿主在走真实时间轴，不是冻结预览。
        let last_raw = self.last_raw.load(Ordering::Acquire);
        if last_raw != raw_start {
            self.reset_candidate();
            self.last_raw.store(raw_start, Ordering::Release);
            self.first_raw.store(raw_start, Ordering::Release);
            self.last_at_ms.store(now_ms(), Ordering::Release);
        }
        if repeated_nonzero {
            self.reset_candidate();
            return None;
        }
        let previous_valid = self.fingerprint_valid.swap(true, Ordering::AcqRel);
        let previous = self.last_fingerprint.swap(fingerprint, Ordering::AcqRel);
        let changed = !previous_valid || previous != fingerprint;
        let prior_frames = self.candidate_frames.load(Ordering::Acquire);
        let stable = self.stable_blocks.fetch_add(1, Ordering::AcqRel) + 1;
        self.candidate_frames
            .fetch_add(frames as u64, Ordering::AcqRel);
        if !silent {
            self.candidate_signal.store(true, Ordering::Release);
        }
        self.last_at_ms.store(now_ms(), Ordering::Release);

        // 内容逐块完全相同（且非静音）⇒ 停播重复投递，不是预览。
        if !changed && !silent && stable > 1 {
            self.reset_candidate();
            return None;
        }

        let rate = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate
        } else {
            48_000.0
        };
        let confirm_frames = (rate * PREVIEW_CONFIRM_SEC).round().max(1.0) as u64;
        if stable < PREVIEW_MIN_BLOCKS
            || prior_frames.saturating_add(frames as u64) < confirm_frames
            || !self.candidate_signal.load(Ordering::Acquire)
        {
            return None;
        }

        // 确认：以事件起点为 pass 原点（未知时退化为首块），cursor 从“首块相对
        // 事件起点的偏移 + 已经写下的候选帧数”开始，保证确认块与候选段首尾相接。
        let first_raw = self.first_raw.load(Ordering::Acquire);
        let origin = match event_origin {
            Some(origin) if first_raw >= origin => origin,
            _ => first_raw,
        };
        let local = first_raw.saturating_sub(origin).max(0) + prior_frames as i64;
        self.begin(ProcessingKind::FrozenPreview, origin, rate, local);
        self.cursor_frames
            .fetch_add(frames as i64, Ordering::AcqRel);
        Some(ProcessingBlock {
            kind: ProcessingKind::FrozenPreview,
            pass_id: self.pass_id(),
            local_start: local,
            rebase_frames: -origin,
            new_pass: true,
        })
    }

    fn begin(&self, kind: ProcessingKind, origin_raw: i64, sample_rate: f64, cursor: i64) {
        self.pass_id.fetch_add(1, Ordering::AcqRel);
        self.kind.store(kind.code(), Ordering::Release);
        self.origin_raw.store(origin_raw, Ordering::Release);
        self.first_raw.store(origin_raw, Ordering::Release);
        self.cursor_frames.store(cursor, Ordering::Release);
        self.candidate_frames.store(0, Ordering::Release);
        self.stable_blocks.store(0, Ordering::Release);
        self.candidate_signal.store(false, Ordering::Release);
        self.fingerprint_valid.store(false, Ordering::Release);
        if sample_rate.is_finite() && sample_rate > 0.0 {
            self.sample_rate
                .store(sample_rate.round() as u32, Ordering::Release);
        }
        self.active.store(true, Ordering::Release);
    }

    fn reset_candidate(&self) {
        self.candidate_frames.store(0, Ordering::Release);
        self.stable_blocks.store(0, Ordering::Release);
        self.candidate_signal.store(false, Ordering::Release);
        self.fingerprint_valid.store(false, Ordering::Release);
    }

    pub(crate) fn end(&self) {
        self.active.store(false, Ordering::Release);
        self.kind
            .store(ProcessingKind::None.code(), Ordering::Release);
        self.cursor_frames.store(0, Ordering::Release);
        self.candidate_frames.store(0, Ordering::Release);
        self.stable_blocks.store(0, Ordering::Release);
        self.candidate_signal.store(false, Ordering::Release);
        self.fingerprint_valid.store(false, Ordering::Release);
        self.last_mode.store(0, Ordering::Release);
        self.last_at_ms.store(0, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preview_block(
        pass: &ProcessingPass,
        raw: i64,
        frames: usize,
        fingerprint: u64,
    ) -> Option<ProcessingBlock> {
        pass.note_preview_block(
            0,
            raw,
            frames,
            48_000.0,
            fingerprint,
            false,
            false,
            false,
            true,
            None,
        )
    }

    #[test]
    fn offline_cursor_starts_at_zero_and_advances_by_block_frames() {
        let pass = ProcessingPass::default();
        assert_eq!(
            pass.note_offline_block(2, 1_500_000, 512, 48_000.0)
                .map(|block| block.local_start),
            Some(0)
        );
        assert_eq!(
            pass.note_offline_block(2, 1_500_000, 512, 48_000.0)
                .map(|block| block.local_start),
            Some(512)
        );
        assert_eq!(pass.cursor_frames(), 1_024);
        assert_eq!(pass.note_offline_block(0, 0, 512, 48_000.0), None);
        assert!(!pass.is_active());
    }

    #[test]
    fn preview_is_confirmed_after_three_quarters_of_a_second() {
        let pass = ProcessingPass::default();
        let mut confirmed = None;
        for index in 0..40 {
            let block = preview_block(&pass, 1_295_764, 1_200, index as u64);
            if block.is_some() {
                confirmed = block;
                break;
            }
        }
        let block = confirmed.expect("0.75 s 之后必须确认预览");
        assert_eq!(block.kind, ProcessingKind::FrozenPreview);
        assert_eq!(block.local_start, 34_800);
        assert_eq!(block.rebase_frames, -1_295_764);
        assert!(pass.is_active());
        assert!(pass.is_preview());
    }

    #[test]
    fn a_playing_edge_cancels_the_preview() {
        let pass = ProcessingPass::default();
        for index in 0..10 {
            let _ = preview_block(&pass, 1_000, 1_200, index);
        }
        let block = pass.note_preview_block(
            0, 1_000, 1_200, 48_000.0, 999, false, false, true, true, None,
        );
        assert!(block.is_none());
        assert!(!pass.is_active());
    }

    #[test]
    fn a_moving_raw_cancels_the_preview() {
        let pass = ProcessingPass::default();
        for index in 0..10 {
            let _ = preview_block(&pass, 1_000 + index as i64 * 1_200, 1_200, index);
        }
        assert!(!pass.is_active());
    }

    #[test]
    fn a_repeated_nonzero_block_cancels_the_preview() {
        let pass = ProcessingPass::default();
        for index in 0..10 {
            let _ = preview_block(&pass, 1_000, 1_200, index);
        }
        let block =
            pass.note_preview_block(0, 1_000, 1_200, 48_000.0, 9, false, true, false, true, None);
        assert!(block.is_none());
        assert!(!pass.is_active());
    }

    #[test]
    fn preview_uses_the_event_origin_when_known() {
        let pass = ProcessingPass::default();
        let mut confirmed = None;
        for index in 0..40 {
            let block = pass.note_preview_block(
                0,
                1_295_764,
                1_200,
                48_000.0,
                index as u64,
                false,
                false,
                false,
                true,
                Some(906_000),
            );
            if block.is_some() {
                confirmed = block;
                break;
            }
        }
        let block = confirmed.expect("must confirm");
        assert_eq!(block.rebase_frames, -906_000);
        assert_eq!(block.local_start, 1_295_764 - 906_000 + 34_800);
    }

    #[test]
    fn ending_the_pass_clears_the_cursor() {
        let pass = ProcessingPass::default();
        assert!(pass.note_offline_block(2, 0, 512, 48_000.0).is_some());
        pass.end();
        assert!(!pass.is_active());
        assert_eq!(pass.cursor_frames(), 0);
    }
}

impl ExtensionOwner {
    /// 记录 setupProcessing 的 process mode。
    pub(crate) fn set_setup_process_mode(&self, mode: i32) {
        self.setup_process_mode.store(mode, Ordering::Release);
    }

    pub(crate) fn setup_process_mode(&self) -> i32 {
        self.setup_process_mode.load(Ordering::Acquire)
    }

    /// 每个 process 块调用一次：离线时返回本块本地 cursor。
    ///
    /// 【为什么只在未绑定实例上生效】ARA 的离线渲染由文档/区域驱动，仍应使用宿主
    /// 给的 projectTimeSamples；Capture 才需要自己的 0 起点。
    pub(crate) fn note_offline_block(
        &self,
        context: &crate::audio_abi::ProcessContext,
        mode: i32,
        frames: usize,
    ) -> Option<i64> {
        if self.is_bound() || self.bound_once.load(Ordering::Acquire) {
            return None;
        }
        self.processing_pass
            .note_offline_block(
                mode,
                context.project_time_samples,
                frames,
                context.sample_rate,
            )
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

    /// 组件失活：与停止处理同一处置，但不丢素材。
    pub(crate) fn note_component_deactivated(&self) {
        self.note_processing_stopped();
    }

    /// 离线/预览 pass 是否活跃。
    pub(crate) fn offline_pass_active(&self) -> bool {
        self.processing_pass.is_active()
    }

    /// 实例级事件起点（组件 state 恢复 / 首次域切换学到）。
    pub(crate) fn event_origin_samples(&self) -> Option<i64> {
        // 0 是“未知”的哨兵（`AtomicI64::default()` 就是 0）；存储值是 origin + 1。
        let value = self.event_origin_samples.load(Ordering::Acquire);
        (value != 0).then_some(value - 1)
    }

    /// 记录事件起点。只写一次：第一次学到的值就是事件在工程里的绝对起点。
    pub(crate) fn note_event_origin(&self, origin: i64) {
        if origin < 0 {
            return;
        }
        let _ = self.event_origin_samples.compare_exchange(
            0,
            origin.saturating_add(1),
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    /// 是否允许把“mode=0 + kPlaying=0 + raw 固定 + 内容连续”识别成冻结预览。
    ///
    /// 【为什么默认关闭】它不是通用形状：真实 Track FX 停播但仍有 live input 时
    /// 也可能长成这样。只有两类证据允许进入：宿主在 setupProcessing 明确声明
    /// mode=2（任何宿主），或宿主属于 VEGAS 家族且本实例已武装（问题 2 的实际入口）。
    pub(crate) fn capture_preview_allowed(&self) -> bool {
        if self.is_bound() || self.bound_once.load(Ordering::Acquire) {
            return false;
        }
        if self.setup_process_mode() == 2 {
            return true;
        }
        self.capture().is_armed() && crate::host_identity::is_vegas_family()
    }

    pub(crate) fn processing_pass_active(&self) -> bool {
        self.processing_pass.is_active()
    }
}
