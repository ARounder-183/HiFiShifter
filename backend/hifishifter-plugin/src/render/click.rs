//! 输出接缝的**块账本**：把"咔"从一句主观描述变成一行可判定的读数。
//!
//! # 为什么需要它
//!
//! round 18~22 的每一次修复都靠"用户再听一遍"来判定对错，因为插件里**没有任何**
//! 关于"这一块输出里有没有阶跃"的读数。于是同一个症状被反复归因到不同机理，
//! 而每次改完都无法证伪。
//!
//! 这个账本只做一件事：音频线程把每块的**形状**记进一个定长环（全原子、不分配、
//! 不取锁），非实时线程定期扫一遍，把"阶跃显著大于邻域"的块打印成 `[click]` 行。
//! 每行自带足够定位的信息 —— 哪个流、哪种输出源、什么接缝、这一块里有多少帧
//! **不属于我们**（未覆盖）、覆盖边是否落在块内、宿主输入在那一段有没有信号。
//!
//! 有了这五个数，R1（假静音）/ R2（回退到空输入）/ R3（线性交叉淡化）/
//! R4（首块原样输出）/ R5（位置跳变）就能被**互相区分**，而不再靠猜。
//!
//! # 为什么阈值判定不在音频线程做
//!
//! "显著"是相对**邻域中位数**的，而求中位数要排序、要缓冲 —— 两者在音频线程上
//! 都是禁止的。音频线程只记录原始读数；判定留给非实时线程（它顺手还能把判定
//! 所需的上下文补齐）。

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// 环的容量。
///
/// 【为什么是 256】48 kHz / 512 帧下约 2.7 秒的历史。阶跃的成因（渲染追上、
/// 接缝、起播窗口收口）都在这个尺度内成对出现，再长的历史只会让报告行变多。
pub(crate) const CLICK_RING: usize = 256;

/// 绝对阶跃阈值（满幅 = 1.0）。
///
/// 【为什么是 0.25】−12 dBFS 的单样本跳变在音乐内容里几乎不可能自然出现；
/// 而人耳对它的感知就是一次"咔"。低于它的阶跃被相对阈值覆盖。
const MAX_STEP_ABS: f32 = 0.25;

/// 相对阈值：本块阶跃超过邻域中位数的这么多倍就算可疑。
///
/// 【为什么还要相对判据】安静段落里的阶跃绝对值可能只有 0.05，但相对该段的
/// 内容仍然是满幅跳变 —— 用户听到的"咔"一样清楚。
const MAX_STEP_RATIO: f32 = 8.0;

/// 一次报告最多打印多少行（其余只计数）。
const REPORT_LIMIT: usize = 8;

/// 一条接缝的种类（与 `extension::Seam` 一一对应，编码成 u8 以便跨线程）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SeamCode {
    /// 本块没有发生源切换。
    None,
    /// 异源：旧信号被掩成 0，新信号淡入。
    FadeIn,
    /// 同族切换：短交叉淡化。
    Crossfade,
    /// Raw ↔ Processed：宽等功率交叉淡化。
    WideCrossfade,
}

impl SeamCode {
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::None => 0,
            Self::FadeIn => 1,
            Self::Crossfade => 2,
            Self::WideCrossfade => 3,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::FadeIn => "fade-in",
            Self::Crossfade => "crossfade",
            Self::WideCrossfade => "wide",
        }
    }
}

/// 一块输出的形状读数（音频线程算好，交给 [`ClickLedger::record`]）。
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockSample {
    pub stream: u8,
    /// 本块的输出源种类（`TransferOutput::code()`）。
    pub kind: u8,
    pub seam: SeamCode,
    pub fade_len: u32,
    pub frames: u32,
    pub first: f32,
    pub last: f32,
    /// 块内相邻样本的最大绝对差（含与上一块尾巴的接缝）。
    pub max_step: f32,
    pub peak: f32,
    /// 本块**未覆盖**帧上宿主输入的峰值（`NaN` = 本块没有未覆盖帧）。
    pub input_peak: f32,
    /// 本块里**不属于我们**的帧数（快照未覆盖 / 无快照）。
    pub uncovered: u32,
    /// 覆盖边（覆盖 ↔ 未覆盖的翻转点）是否落在本块内。
    pub edge: bool,
    /// 本流的第一块（`primed_kind` 从 0 开始）。
    pub first_of_stream: bool,
}

/// 环里的一格。全部是原子量：音频线程写，非实时线程读。
#[derive(Default)]
struct Slot {
    seq: AtomicU64,
    stream: AtomicU32,
    kind: AtomicU32,
    seam: AtomicU32,
    fade_len: AtomicU32,
    frames: AtomicU32,
    first: AtomicU32,
    last: AtomicU32,
    max_step: AtomicU32,
    peak: AtomicU32,
    input_peak: AtomicU32,
    uncovered: AtomicU32,
    edge: AtomicBool,
    first_of_stream: AtomicBool,
}

/// 一份从环里读出来的快照（非实时线程用）。
#[derive(Clone, Copy, Debug)]
pub(crate) struct RecordedBlock {
    pub seq: u64,
    pub sample: BlockSample,
}

/// 输出接缝的块账本。
pub(crate) struct ClickLedger {
    slots: Vec<Slot>,
    /// 下一格写到哪（`fetch_add`，环绕）。
    next: AtomicUsize,
    /// 单调递增的块序号（`0` = 从未记录）。
    seq: AtomicU64,
    /// 被判为可疑的块总数（诊断读数）。
    suspicious: AtomicU64,
    /// 已经被报告过的最大序号（避免同一块被反复打印）。
    reported: AtomicU64,
}

impl Default for ClickLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl ClickLedger {
    /// 建一份空账本。**只在非实时线程调用**（它分配环）。
    pub(crate) fn new() -> Self {
        Self {
            slots: (0..CLICK_RING).map(|_| Slot::default()).collect(),
            next: AtomicUsize::new(0),
            seq: AtomicU64::new(0),
            suspicious: AtomicU64::new(0),
            reported: AtomicU64::new(0),
        }
    }

    /// 记一块。**音频线程**：只写原子量，不分配、不取锁。
    pub(crate) fn record(&self, sample: BlockSample) {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % CLICK_RING;
        let seq = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        let slot = &self.slots[index];
        slot.stream.store(u32::from(sample.stream), Ordering::Relaxed);
        slot.kind.store(u32::from(sample.kind), Ordering::Relaxed);
        slot.seam.store(u32::from(sample.seam.code()), Ordering::Relaxed);
        slot.fade_len.store(sample.fade_len, Ordering::Relaxed);
        slot.frames.store(sample.frames, Ordering::Relaxed);
        slot.first.store(sample.first.to_bits(), Ordering::Relaxed);
        slot.last.store(sample.last.to_bits(), Ordering::Relaxed);
        slot.max_step.store(sample.max_step.to_bits(), Ordering::Relaxed);
        slot.peak.store(sample.peak.to_bits(), Ordering::Relaxed);
        slot.input_peak
            .store(sample.input_peak.to_bits(), Ordering::Relaxed);
        slot.uncovered.store(sample.uncovered, Ordering::Relaxed);
        slot.edge.store(sample.edge, Ordering::Relaxed);
        slot.first_of_stream
            .store(sample.first_of_stream, Ordering::Relaxed);
        // 序号最后写（Release）：读者看到非零 seq 时，其余字段必然已经写好。
        slot.seq.store(seq, Ordering::Release);
    }

    /// 读一遍环，按序号升序返回（非实时线程）。
    pub(crate) fn recorded(&self) -> Vec<RecordedBlock> {
        let mut out = Vec::with_capacity(CLICK_RING);
        for slot in &self.slots {
            let seq = slot.seq.load(Ordering::Acquire);
            if seq == 0 {
                continue;
            }
            out.push(RecordedBlock {
                seq,
                sample: BlockSample {
                    stream: slot.stream.load(Ordering::Relaxed) as u8,
                    kind: slot.kind.load(Ordering::Relaxed) as u8,
                    seam: seam_from_code(slot.seam.load(Ordering::Relaxed) as u8),
                    fade_len: slot.fade_len.load(Ordering::Relaxed),
                    frames: slot.frames.load(Ordering::Relaxed),
                    first: f32::from_bits(slot.first.load(Ordering::Relaxed)),
                    last: f32::from_bits(slot.last.load(Ordering::Relaxed)),
                    max_step: f32::from_bits(slot.max_step.load(Ordering::Relaxed)),
                    peak: f32::from_bits(slot.peak.load(Ordering::Relaxed)),
                    input_peak: f32::from_bits(slot.input_peak.load(Ordering::Relaxed)),
                    uncovered: slot.uncovered.load(Ordering::Relaxed),
                    edge: slot.edge.load(Ordering::Relaxed),
                    first_of_stream: slot.first_of_stream.load(Ordering::Relaxed),
                },
            });
        }
        out.sort_by_key(|block| block.seq);
        out
    }

    /// 已记录的块总数。
    pub(crate) fn blocks(&self) -> u64 {
        self.seq.load(Ordering::Acquire)
    }

    /// 被判为可疑的块总数。
    pub(crate) fn suspicious(&self) -> u64 {
        self.suspicious.load(Ordering::Acquire)
    }

    /// 取出**尚未报告过**的可疑块，并格式化成一行的主体（不含级别前缀）。
    ///
    /// 【为什么"报告过"要单独记】本函数由 UI 定时器驱动（20 ms 一次）。没有游标时
    /// 同一块会在环里存活 2.7 秒、被打印上百遍，日志立刻失去价值。
    pub(crate) fn drain_suspicious(&self) -> Vec<String> {
        let blocks = self.recorded();
        if blocks.is_empty() {
            return Vec::new();
        }
        // 邻域中位数：把全部块的最大阶跃排序取中位。它代表"这个会话的常态"，
        // 于是"安静段落里的相对跳变"与"响段落里的绝对跳变"用同一把尺子。
        let mut steps: Vec<f32> = blocks.iter().map(|b| b.sample.max_step).collect();
        steps.sort_by(|a, b| a.total_cmp(b));
        let median = steps[steps.len() / 2];
        let reported = self.reported.load(Ordering::Acquire);
        let mut fresh: Vec<&RecordedBlock> = blocks
            .iter()
            .filter(|block| block.seq > reported && is_suspicious(block.sample.max_step, median))
            .collect();
        if fresh.is_empty() {
            return Vec::new();
        }
        let highest = fresh.iter().map(|block| block.seq).max().unwrap_or(reported);
        self.reported.store(highest, Ordering::Release);
        self.suspicious
            .fetch_add(fresh.len() as u64, Ordering::Relaxed);
        fresh.truncate(REPORT_LIMIT);
        fresh.iter().map(|block| format_block(block, median)).collect()
    }
}

/// 这一块是否可疑。**纯函数**：判定规则本身要被单测钉住。
pub(crate) fn is_suspicious(max_step: f32, median: f32) -> bool {
    if !max_step.is_finite() {
        return false;
    }
    if max_step > MAX_STEP_ABS {
        return true;
    }
    // 中位数为 0（整段静音）时只看绝对阈值：否则任何非零都会被判成"相对过大"。
    median.is_finite() && median > 1e-6 && max_step > median * MAX_STEP_RATIO
}

fn seam_from_code(code: u8) -> SeamCode {
    match code {
        1 => SeamCode::FadeIn,
        2 => SeamCode::Crossfade,
        3 => SeamCode::WideCrossfade,
        _ => SeamCode::None,
    }
}

/// 把一块格式化成一行的主体。
///
/// 【为什么每个字段都要打】排查"这一下咔是什么"时，五个读数各自排除一条机理：
///   * `seam` 非 `none` ⇒ 是**源切换**（R3/R4），不是覆盖问题；
///   * `uncovered` > 0 且 `edge=true` ⇒ 覆盖边落在块内 ⇒ 覆盖语义（R1/R2）；
///   * `input_peak≈0` 且 `uncovered>0` ⇒ 回退目标本身是空的（R2）；
///   * `first_of_stream` ⇒ 首块（R4）；
///   * `pos` 与 `corr` 的差 ⇒ 位置跳变（R5）。
fn format_block(block: &RecordedBlock, median: f32) -> String {
    let sample = block.sample;
    let input_peak = if sample.input_peak.is_finite() {
        format!("{:.4}", sample.input_peak)
    } else {
        "n/a".to_owned()
    };
    format!(
        "block={} stream={} kind={} seam={} fade={} frames={} max_step={:.4} (median {:.4}) \
         peak={:.4} first={:.4} last={:.4} uncovered={} edge={} input_peak={} first_of_stream={}",
        block.seq,
        sample.stream,
        sample.kind,
        sample.seam.name(),
        sample.fade_len,
        sample.frames,
        sample.max_step,
        median,
        sample.peak,
        sample.first,
        sample.last,
        sample.uncovered,
        sample.edge,
        input_peak,
        sample.first_of_stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(max_step: f32) -> BlockSample {
        BlockSample {
            stream: 0,
            kind: 3,
            seam: SeamCode::None,
            fade_len: 0,
            frames: 512,
            first: 0.0,
            last: 0.0,
            max_step,
            peak: max_step,
            input_peak: f32::NAN,
            uncovered: 0,
            edge: false,
            first_of_stream: false,
        }
    }

    /// 绝对阈值：一次满幅跳变必须被判成可疑，且带得出它的全部读数。
    #[test]
    fn a_full_scale_step_is_reported_with_its_readings() {
        let ledger = ClickLedger::new();
        for _ in 0..4 {
            ledger.record(sample(0.001));
        }
        ledger.record(BlockSample {
            seam: SeamCode::WideCrossfade,
            uncovered: 96,
            edge: true,
            input_peak: 0.0,
            ..sample(0.8)
        });
        let lines = ledger.drain_suspicious();
        assert_eq!(lines.len(), 1, "只应有那一条：{lines:?}");
        let line = &lines[0];
        assert!(line.contains("max_step=0.8000"), "{line}");
        assert!(line.contains("seam=wide"), "{line}");
        assert!(line.contains("uncovered=96"), "{line}");
        assert!(line.contains("edge=true"), "{line}");
        assert!(line.contains("input_peak=0.0000"), "{line}");
    }

    /// 同一块只报告一次（UI 定时器 20 ms 一次，环里会存活 2.7 s）。
    #[test]
    fn a_block_is_reported_once() {
        let ledger = ClickLedger::new();
        ledger.record(sample(0.9));
        assert_eq!(ledger.drain_suspicious().len(), 1);
        assert!(ledger.drain_suspicious().is_empty(), "同一块不得重复打印");
        ledger.record(sample(0.9));
        assert_eq!(ledger.drain_suspicious().len(), 1, "新块仍要打印");
        assert_eq!(ledger.suspicious(), 2);
    }

    /// 平滑内容不得触发。
    #[test]
    fn smooth_content_never_trips_the_ledger() {
        let ledger = ClickLedger::new();
        for index in 0..200 {
            let step = 0.002 + (index % 7) as f32 * 0.0005;
            ledger.record(sample(step));
        }
        assert!(ledger.drain_suspicious().is_empty());
    }

    /// 相对阈值：安静段落里的小阶跃也要报（绝对阈值够不着它）。
    #[test]
    fn a_small_step_in_a_quiet_passage_is_still_relative_suspicious() {
        assert!(is_suspicious(0.06, 0.002));
        assert!(!is_suspicious(0.003, 0.002));
    }

    /// 整段静音（中位数为 0）时只看绝对阈值 —— 否则任何非零都会被判成"相对过大"。
    #[test]
    fn silence_uses_the_absolute_threshold_only() {
        assert!(!is_suspicious(0.01, 0.0));
        assert!(is_suspicious(0.3, 0.0));
    }

    /// 环满之后仍按序号升序返回，且不会重复。
    #[test]
    fn the_ring_wraps_without_reordering() {
        let ledger = ClickLedger::new();
        for _ in 0..(CLICK_RING + 10) {
            ledger.record(sample(0.001));
        }
        let blocks = ledger.recorded();
        assert_eq!(blocks.len(), CLICK_RING);
        assert!(blocks.windows(2).all(|pair| pair[0].seq < pair[1].seq));
        assert_eq!(ledger.blocks(), (CLICK_RING + 10) as u64);
    }
}
