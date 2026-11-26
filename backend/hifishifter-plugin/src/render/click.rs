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

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering};

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

/// 一次排水最多格式化多少条读数（其余只计数）。
///
/// 【为什么从 8 提到 32】排放方现在会把同类读数**合并成一行**（见
/// `capture::log_click_report`），所以"格式化多少条"不再等于"日志里多几行"。
/// 反过来，8 太小了：界面定时器一拍能攒下十几块，超出的部分会被直接标成
/// "已报告"而永远消失 —— 一个会静默吞事件的诊断，比没有诊断更坏。
const REPORT_LIMIT: usize = 32;

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
    /// 本块内发生了**覆盖边**（"宿主原声 ↔ 处理结果"的翻转点），没有源切换。
    ///
    /// 【为什么单独一档】它与 `FadeIn` 的听感几乎一样（都是一次信号更换），但机理
    /// 完全不同：`FadeIn` 是**输出源**换了（Raw/Processed/透传之间），本档是**同一份
    /// 源内部**的覆盖归属换了。把两者混成一个 `none` 会让"覆盖表宣称了不该宣称的
    /// 帧"这类问题在读数上完全隐形 —— round 23 的假洞就是这样躲过了一轮验证。
    CoverageEdge,
}

impl SeamCode {
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::None => 0,
            Self::FadeIn => 1,
            Self::Crossfade => 2,
            Self::WideCrossfade => 3,
            Self::CoverageEdge => 4,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::FadeIn => "fade-in",
            Self::Crossfade => "crossfade",
            Self::WideCrossfade => "wide",
            Self::CoverageEdge => "coverage-edge",
        }
    }
}

/// 一条读数的类别。
///
/// 【为什么类名要独立成一个枚举】输出的每一行都要能被"按类过滤"：现场排查时人只会
/// 看某一类（"有没有换源" / "有没有洞"）。把类名拼进正文里，过滤就只能靠前缀匹配
/// 字符串 —— 而汇总行（同类合并成一行）又需要它单独存在以便计数。两件事要的是同一个
/// 事实，它就该有同一个来源。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClickClass {
    /// 换源 / 首块 / 覆盖边。
    Switch,
    /// "有声 → 整块零 → 有声"。
    Hole,
    /// 阶跃显著。
    Step,
    /// 相邻块能量比离开健康带。
    Envelope,
}

impl ClickClass {
    /// 全部类别（顺序与 [ClickClass::index] 一致）。
    ///
    /// 【为什么要有它】输出方按"类别 × 归因"分桶，冲出去时要用下标反查类名。
    /// 让下标与类名来自同一张表，就不会出现"第 3 个桶印成 step"这种错位。
    pub(crate) const ALL: [Self; 4] = [Self::Switch, Self::Hole, Self::Step, Self::Envelope];

    /// 类别的总数（数组下标用）。
    pub(crate) const COUNT: usize = Self::ALL.len();

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Switch => "switch",
            Self::Hole => "hole",
            Self::Step => "step",
            Self::Envelope => "envelope",
        }
    }

    /// 数组下标（顺序与 [ClickClass::ALL] 一致）。
    pub(crate) fn index(self) -> usize {
        match self {
            Self::Switch => 0,
            Self::Hole => 1,
            Self::Step => 2,
            Self::Envelope => 3,
        }
    }
}

/// 一块输出的形状读数（音频线程算好，交给 [`ClickLedger::record`]）。
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockSample {
    pub stream: u8,
    /// 本块的输出源种类（`CaptureOutput::code()`）。
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
    /// 本块的 RMS（左右声道的均方根）。
    ///
    /// 【为什么阶跃之外还要一个能量读数】20 ms 的等功率过渡**不掉阶跃**：它把
    /// 两段不相关的信号按 cos/sin 叠起来，波形逐样本连续，`max_step` 只有 0.01 量级，
    /// 而听感是一次明显的"抿"。那种事件只能从**包络**上看出来 —— 相邻块的 RMS
    /// 比值离开 [0.5, 2] 就说明这一块的能量被换掉了。
    pub rms: f32,
    /// 覆盖边（覆盖 ↔ 未覆盖的翻转点）是否落在本块内。
    pub edge: bool,
    /// 本流的第一块（`primed_kind` 从 0 开始）。
    pub first_of_stream: bool,
    /// 本块**读快照的起点**（工程帧；[`NO_POSITION`] = 未知）。
    ///
    /// 【为什么这张账本必须记位置】此前它只记"第几块"，而块序号只有在**块与块严格
    /// 相接**时才能换算成时间位置。宿主不必按连续位置回调：预取会跳到播放头、循环
    /// 折返会回卷、起播会带 pre-roll。于是同一个读数既可能意味着"那段内容本来就是
    /// 静音"，也可能意味着"我们根本没写出内容" —— 两种结论完全相反，而现场无从
    /// 区分，只能靠用户再听一遍。位置进了账本，读数第一次能对着工程时间轴念出来。
    pub pos: i64,
    /// 同一位置换算成秒（`NaN` = 未知）。仅为了让人读日志时不必心算帧号。
    pub pos_sec: f32,
}

/// 账本里"不知道位置"的哨兵。`i64::MIN` 与任何合法工程帧都不可能相等。
pub(crate) const NO_POSITION: i64 = i64::MIN;

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
    rms: AtomicU32,
    edge: AtomicBool,
    first_of_stream: AtomicBool,
    pos: AtomicI64,
    pos_sec: AtomicU32,
    jump: AtomicI64,
}

/// 一份从环里读出来的快照（非实时线程用）。
#[derive(Clone, Copy, Debug)]
pub(crate) struct RecordedBlock {
    pub seq: u64,
    pub sample: BlockSample,
    /// 与"上一块本应接上的位置"之差（帧）；`0` = 严格相接。
    ///
    /// 【为什么它比 `pos` 更重要】`max_step` 与 `envelope` 两类判据都是**跨块**
    /// 比较：两块不相接时，它们量的根本不是同一段音频的接缝，报出来的每个数都无
    /// 意义。有了它，"这条账本能不能按顺序读"本身就成了一个可断言的事实。
    pub jump: i64,
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
    /// 每条流"下一块本应开始的位置"（`NO_POSITION` = 还没记过）。
    ///
    /// 【为什么按流分开】两条流的位置是两条独立的时间轴；混在一起算，"跳变"恒为真。
    expected: [AtomicI64; 3],
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
            expected: std::array::from_fn(|_| AtomicI64::new(NO_POSITION)),
        }
    }

    /// 记一块。**音频线程**：只写原子量，不分配、不取锁。
    pub(crate) fn record(&self, sample: BlockSample) {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % CLICK_RING;
        let seq = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        let slot = &self.slots[index];
        slot.stream
            .store(u32::from(sample.stream), Ordering::Relaxed);
        slot.kind.store(u32::from(sample.kind), Ordering::Relaxed);
        slot.seam
            .store(u32::from(sample.seam.code()), Ordering::Relaxed);
        slot.fade_len.store(sample.fade_len, Ordering::Relaxed);
        slot.frames.store(sample.frames, Ordering::Relaxed);
        slot.first.store(sample.first.to_bits(), Ordering::Relaxed);
        slot.last.store(sample.last.to_bits(), Ordering::Relaxed);
        slot.max_step
            .store(sample.max_step.to_bits(), Ordering::Relaxed);
        slot.peak.store(sample.peak.to_bits(), Ordering::Relaxed);
        slot.input_peak
            .store(sample.input_peak.to_bits(), Ordering::Relaxed);
        slot.uncovered.store(sample.uncovered, Ordering::Relaxed);
        slot.rms.store(sample.rms.to_bits(), Ordering::Relaxed);
        slot.edge.store(sample.edge, Ordering::Relaxed);
        slot.first_of_stream
            .store(sample.first_of_stream, Ordering::Relaxed);
        // 【位置与接续】见 `BlockSample::pos` 与 `RecordedBlock::jump`。`expected`
        // 存的是"这一块之后下一块该在哪里"，所以它写入的是 `pos + frames`。
        let stream = usize::from(sample.stream).min(2);
        let expected = if sample.pos == NO_POSITION {
            NO_POSITION
        } else {
            sample.pos.saturating_add(i64::from(sample.frames))
        };
        let previous = self.expected[stream].swap(expected, Ordering::Relaxed);
        let jump = if sample.pos == NO_POSITION || previous == NO_POSITION {
            0
        } else {
            sample.pos.saturating_sub(previous)
        };
        slot.pos.store(sample.pos, Ordering::Relaxed);
        slot.pos_sec
            .store(sample.pos_sec.to_bits(), Ordering::Relaxed);
        slot.jump.store(jump, Ordering::Relaxed);
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
                jump: slot.jump.load(Ordering::Relaxed),
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
                    rms: f32::from_bits(slot.rms.load(Ordering::Relaxed)),
                    edge: slot.edge.load(Ordering::Relaxed),
                    first_of_stream: slot.first_of_stream.load(Ordering::Relaxed),
                    pos: slot.pos.load(Ordering::Relaxed),
                    pos_sec: f32::from_bits(slot.pos_sec.load(Ordering::Relaxed)),
                },
            });
        }
        out.sort_by_key(|block| block.seq);
        out
    }

    /// 已记录的块总数（测试断言用）。
    #[cfg(test)]
    pub(crate) fn blocks(&self) -> u64 {
        self.seq.load(Ordering::Acquire)
    }

    /// 被判为可疑的块总数（测试断言用）。
    #[cfg(test)]
    pub(crate) fn suspicious(&self) -> u64 {
        self.suspicious.load(Ordering::Acquire)
    }

    /// 取出**尚未报告过**的读数。
    ///
    /// 【为什么"报告过"要单独记】本函数由周期定时器驱动（约 100 ms 一次）。没有游标时
    /// 同一块会在环里存活 2.7 秒、被打印上百遍，日志立刻失去价值。
    ///
    /// # 四类事件，各自回答一个不同的问题
    ///
    /// 【为什么必须分四类】此前只有"阶跃显著"一类，于是：
    ///   * **包络类**（20 ms 等功率过渡）逐样本连续，`max_step` 只有 0.01 量级 ⇒
    ///     一条都不报。用户的"咔"在读数上**完全隐形**，我们只能靠猜；
    ///   * **源切换**本身就是用户听到那一下的位置，却被"可疑"这个过滤器挡在门外 ——
    ///     只有当它恰好也产生大阶跃时才露头。
    ///
    /// 现在：`switch` 无条件产出（一次播放本该只有极少几次），`envelope` 看相邻块的
    /// 能量比，`step` 保留原有的阶跃判据，`hole` 抓"上一块有信号、这一块塌到近乎零"
    /// —— 也就是用户说的"静音断点"本身。
    ///
    /// # 产出的是读数，不是结论
    ///
    /// 每条读数都带一个 `blamed`（见 [blames_us]）：**只有**"这一块里有不属于我们的
    /// 帧"才是账本自己能证明的事。级别由它决定 —— 有证据的升级为 WARN，其余留在
    /// DEBUG。这个分工是这条账本能长期留在日志里的前提：现场 30 秒的渲染就有 200 多条
    /// 这一类读数，全部以 WARN 打出来时，真正的故障反而没有人看得见。
    pub(crate) fn drain_reports(&self) -> Vec<ClickReport> {
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
        let mut out = Vec::new();
        let mut highest = reported;
        let mut counted = 0_u64;
        for (index, block) in blocks.iter().enumerate() {
            if block.seq <= reported {
                continue;
            }
            highest = highest.max(block.seq);
            let sample = &block.sample;
            // `switch`：源切换 / 首块 / 覆盖边 —— 它们本身就是"边界在哪"，无条件产出。
            let switching = sample.seam != SeamCode::None || sample.first_of_stream;
            if switching {
                counted += 1;
                out.push(report(
                    ClickClass::Switch,
                    sample.max_step,
                    format_block(block, median),
                    sample,
                ));
                continue;
            }
            if is_suspicious(sample.max_step, median) {
                counted += 1;
                out.push(report(
                    ClickClass::Step,
                    sample.max_step,
                    format_block(block, median),
                    sample,
                ));
                continue;
            }
            // `hole`：上一块有信号、这一块塌到近乎零 —— "有声 → 静音 → 有声"就是用户
            // 说的"静音断点"。
            //
            // 【为什么这一档必须存在】`envelope_ratio` 在 `rms <= 1e-5` 时直接放弃
            // （整块是零时没有比值可言），`is_suspicious` 只看块内的阶跃（整块零的
            // `max_step` 恰好是 0）。于是**真正的洞恰好是唯一看不见的一类**：一块纯零
            // 既不进 `step` 也不进 `envelope`。此前两轮"读数全绿、耳朵有洞"的僵局，
            // 缺的就是这一档读数。
            if let Some(previous) = previous_peak(&blocks, index) {
                if previous >= HOLE_PREVIOUS_PEAK && sample.peak < previous * HOLE_RATIO {
                    counted += 1;
                    // 掉得越深"更强"（汇总行只留同类里最强的一条）。
                    let severity = previous / sample.peak.max(1e-9);
                    out.push(report(
                        ClickClass::Hole,
                        severity,
                        format_block(block, median),
                        sample,
                    ));
                    continue;
                }
            }
            // `envelope`：同一条流、上一块非静音时的能量比。
            if let Some(ratio) = envelope_ratio(&blocks, index) {
                if !(ENVELOPE_MIN..=ENVELOPE_MAX).contains(&ratio) {
                    counted += 1;
                    // 离 1 越远越强：0.25 与 4 一样糟。
                    let severity = ratio.max(1.0 / ratio.max(f32::MIN_POSITIVE));
                    out.push(report(
                        ClickClass::Envelope,
                        severity,
                        format!("ratio={ratio:.3} {}", format_block(block, median)),
                        sample,
                    ));
                }
            }
        }
        if out.is_empty() {
            return Vec::new();
        }
        self.reported.store(highest, Ordering::Release);
        self.suspicious.fetch_add(counted, Ordering::Relaxed);
        out.truncate(REPORT_LIMIT);
        out
    }
}

/// 一条待输出的读数。
pub(crate) struct ClickReport {
    /// 是哪一类事件。
    pub class: ClickClass,
    /// 这一条能否**归因于我们自己**（见 [blames_us]）—— 级别由它决定。
    pub blamed: bool,
    /// 同类里"更强"的那一条会被留在汇总行里（越大越强）。
    pub severity: f32,
    /// 读数主体，**不含**类名（类名由输出方拼，汇总行要用它计数）。
    pub body: String,
}

/// 把一条读数连同它的归因判断装起来。
fn report(class: ClickClass, severity: f32, body: String, sample: &BlockSample) -> ClickReport {
    ClickReport {
        class,
        blamed: blames_us(sample),
        severity,
        body,
    }
}

/// 这一块能否**归因于我们自己**。
///
/// 【判据为什么只有这两个位】账本量的是**输出波形**，它手里没有内容的"标准答案"。
/// 一次起伏既可能是音乐换了口气，也可能是我们换错了源 —— 只看波形永远分不开。唯一
/// 分得开的事实是**覆盖**：`uncovered > 0`（这一块里有帧不是我们写的）与 `edge`
/// （覆盖边落在块内，宿主原声与处理结果在块内翻转）。这两件事只可能由我们造成。
///
/// 【为什么其余读数一律不升级为 WARN】现场 30 秒的渲染里有 182 条 `envelope` 与 30
/// 条 `step`，全部来自 0.5~2.0 这条健康带之外的**正常音乐动态**。把"内容在动"当成
/// "我们出错"，WARN 渠道就失去了检索价值 —— 而它正是现场唯一会被人看的渠道。
fn blames_us(sample: &BlockSample) -> bool {
    sample.uncovered > 0 || sample.edge
}

/// 相邻块能量比的**健康带**。离开它就说明这一块的能量被换掉了。
///
/// 【为什么是 0.5 ~ 2.0】正常音乐在 10 ms 尺度上的块间 RMS 比值极少超出 ±6 dB；
/// 而一次 20 ms 等功率过渡（两侧互不相关）在中点掉 3 dB，叠加相位抵消后实测可以
/// 掉到 0.5 以下。带宽放到 ±6 dB 是为了让"真正的包络事件"露头，而不把普通音符
/// 起落算进来。
const ENVELOPE_MIN: f32 = 0.5;
const ENVELOPE_MAX: f32 = 2.0;

/// "上一块确实有信号"的门限（满幅 = 1.0）。低于它就是安静段落，没有"断"可言。
const HOLE_PREVIOUS_PEAK: f32 = 0.01;

/// 本块峰值跌到上一块的这么多倍以下 ⇒ 判为一次洞。
///
/// 【为什么是 −34 dB】音乐内容在 10 ms 尺度上不可能自然掉这么多；而人耳听到的
/// 就是一次"静音断点"。比它更严会漏掉"塌到残余噪声"的洞，更松会把正常换气算进来。
const HOLE_RATIO: f32 = 0.02;

/// 本块相对**同一条流上一块**的 RMS 比值（`None` = 没有可比的上文）。
///
/// 【为什么必须限定同一条流】实时流与预取流描述的是两段互不相邻的时间；把它们
/// 互相比较得到的比值毫无意义，却会稳定地落在健康带之外 —— 那会让每一块都被报一次。
fn envelope_ratio(blocks: &[RecordedBlock], index: usize) -> Option<f32> {
    let current = &blocks.get(index)?.sample;
    if !(current.rms.is_finite() && current.rms > 1e-5) {
        return None;
    }
    let previous = blocks[..index]
        .iter()
        .rev()
        .find(|block| block.sample.stream == current.stream)?
        .sample;
    if !(previous.rms.is_finite() && previous.rms > 1e-5) {
        return None;
    }
    Some(current.rms / previous.rms)
}

/// 同一条流上一块的峰值（`None` = 没有可比的上文）。
///
/// 【为什么按流取】实时流与预取流描述的是两段互不相邻的时间；跨流比较会把
/// "另一条流刚播完"读成一次洞。
fn previous_peak(blocks: &[RecordedBlock], index: usize) -> Option<f32> {
    let current = &blocks.get(index)?.sample;
    let previous = blocks[..index]
        .iter()
        .rev()
        .find(|block| block.sample.stream == current.stream)?
        .sample;
    previous.peak.is_finite().then_some(previous.peak)
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

/// 位置的人读形式。未知时打 `n/a`，与 `input_peak` 同一套写法。
fn position_text(pos_sec: f32) -> String {
    if pos_sec.is_finite() {
        format!("{pos_sec:.3}s")
    } else {
        "n/a".to_owned()
    }
}

fn seam_from_code(code: u8) -> SeamCode {
    match code {
        1 => SeamCode::FadeIn,
        2 => SeamCode::Crossfade,
        3 => SeamCode::WideCrossfade,
        4 => SeamCode::CoverageEdge,
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
    // 【位置为什么必须在行首】`block=` 只是一个序号：它只有在块严格相接时才对得上
    // 工程时间。现场读日志的人和事后读日志的人，第一件事都是问"这一下在哪儿"。
    format!(
        "block={} at={} jump={} stream={} kind={} seam={} fade={} frames={} max_step={:.4} (median {:.4}) \
         peak={:.4} first={:.4} last={:.4} uncovered={} edge={} input_peak={} \
         first_of_stream={}",
        block.seq,
        position_text(block.sample.pos_sec),
        block.jump,
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

    /// 把读数还原成与日志同一形状的一行（`<class> <body>`）。
    ///
    /// 【为什么测试不直接比对 `ClickReport`】现场读的是**日志行**：类名与主体拼起来
    /// 才是读者看到的东西。断言那一行，等于断言"读者能不能读懂"。
    fn lines(ledger: &ClickLedger) -> Vec<String> {
        ledger
            .drain_reports()
            .into_iter()
            .map(|report| format!("{} {}", report.class.as_str(), report.body))
            .collect()
    }

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
            rms: max_step.max(0.01),
            edge: false,
            first_of_stream: false,
            pos: NO_POSITION,
            pos_sec: f32::NAN,
        }
    }

    /// **"有声 → 整块零 → 有声"必须报成 `hole`，而不是悄无声息。**
    ///
    /// 【为什么要单独钉这一条】块内阶跃判据对整块零的 `max_step` 恰好是 0，包络判据
    /// 在 `rms <= 1e-5` 时直接放弃 —— 于是"真正的静音断点"是账本里唯一看不见的一类，
    /// 而它正是用户反复报的那个症状。
    #[test]
    fn a_block_that_collapses_to_silence_is_reported_as_a_hole() {
        let ledger = ClickLedger::new();
        for _ in 0..4 {
            ledger.record(BlockSample {
                peak: 0.4,
                rms: 0.2,
                ..sample(0.01)
            });
        }
        ledger.record(BlockSample {
            peak: 0.0,
            rms: 0.0,
            first: 0.0,
            last: 0.0,
            ..sample(0.0)
        });
        let lines = lines(&ledger);
        assert_eq!(lines.len(), 1, "只应有那一条：{lines:?}");
        assert!(lines[0].starts_with("hole "), "{}", lines[0]);
        assert!(lines[0].contains("peak=0.0000"), "{}", lines[0]);
    }

    /// 绝对阈值：一次满幅跳变必须被判成可疑，且带得出它的全部读数。
    #[test]
    fn a_full_scale_step_is_reported_with_its_readings() {
        let ledger = ClickLedger::new();
        for _ in 0..4 {
            ledger.record(sample(0.001));
        }
        // 【为什么这里不带 `seam`】带上的话它会被归成 `switch` 类（那一类无条件
        // 打印），本条要钉的是 `step` 判据本身。
        ledger.record(BlockSample {
            uncovered: 96,
            edge: false,
            input_peak: 0.0,
            ..sample(0.8)
        });
        let lines = lines(&ledger);
        assert_eq!(lines.len(), 1, "只应有那一条：{lines:?}");
        let line = &lines[0];
        // 每一行都以**事件类**开头：`step` / `envelope` / `switch`。没有这个前缀，
        // 一条日志无法回答"这一下是阶跃、包络还是换源"，而那正是排查的第一步。
        assert!(line.starts_with("step "), "{line}");
        assert!(line.contains("max_step=0.8000"), "{line}");
        assert!(line.contains("uncovered=96"), "{line}");
        assert!(line.contains("edge=false"), "{line}");
        assert!(line.contains("input_peak=0.0000"), "{line}");
        // 位置与接续必须在每一行里：没有它们，`block=` 只是一个序号，无法换算成
        // 时间，也就无法回答"这一下在工程第几秒"。
        assert!(line.contains(" at="), "{line}");
        assert!(line.contains(" jump="), "{line}");
    }

    /// **位置与接续**：连续块的 `jump` 必须为 0，断开时必须报出差值。
    ///
    /// 【为什么这条要钉】`max_step` / `envelope` 都是**跨块**比较：两块不相接时它们
    /// 量的根本不是同一段音频的接缝，报出来的每个数都无意义。"这条账本能不能按顺序
    /// 读"本身必须先是一个可断言的事实。
    #[test]
    fn a_position_gap_is_reported_and_a_contiguous_run_is_not() {
        let ledger = ClickLedger::new();
        for index in 0..4 {
            ledger.record(BlockSample {
                pos: 48_000 + index * 512,
                ..sample(0.001)
            });
        }
        let blocks = ledger.recorded();
        assert!(
            blocks.iter().all(|block| block.jump == 0),
            "严格相接的块不该报跳变：{blocks:?}"
        );
        // 跳到别处：差值必须立刻可见（正数 = 向前，负数 = 回卷）。
        ledger.record(BlockSample {
            pos: 96_000,
            ..sample(0.001)
        });
        let last = ledger
            .recorded()
            .into_iter()
            .max_by_key(|block| block.seq)
            .unwrap();
        assert_eq!(last.jump, 96_000 - (48_000 + 4 * 512));
        assert!(
            last.sample.pos_sec.is_nan(),
            "夹具没有给秒，读数必须诚实地说不知道"
        );
    }

    /// **源切换无条件报告**：它是"用户听到那一下"的位置，不能被"可疑"过滤器挡在门外。
    ///
    /// 【为什么这条必须单独钉住】旧账本只有"阶跃显著"一类判据。而一次 20 ms 等功率
    /// 过渡逐样本连续（`max_step` 只有 0.01 量级）—— 一条都不报。于是用户的"咔"在
    /// 读数上完全隐形，每一轮修复都只能靠"再听一遍"来判定。
    #[test]
    fn a_source_switch_is_always_reported() {
        let ledger = ClickLedger::new();
        for _ in 0..8 {
            ledger.record(sample(0.001));
        }
        // 一个平滑、但**换了源**的块：阶跃很小，包络也正常。
        ledger.record(BlockSample {
            seam: SeamCode::WideCrossfade,
            ..sample(0.002)
        });
        let reports = ledger.drain_reports();
        assert_eq!(reports.len(), 1, "换源必须报一次");
        assert_eq!(reports[0].class, ClickClass::Switch);
        assert!(reports[0].body.contains("seam=wide"), "{}", reports[0].body);
    }

    /// **包络类事件**：20 ms 等功率过渡掉 3 dB 就是这一类。
    ///
    /// 【为什么阶跃判据看不见它】等功率混合的两侧信号是连续的，`max_step` 只有
    /// 0.01 量级；但能量在中点掉一半，听感就是一次"抿"。只有 RMS 比值能抓住它。
    #[test]
    fn an_envelope_dip_is_reported_even_without_a_step() {
        let ledger = ClickLedger::new();
        for _ in 0..6 {
            ledger.record(BlockSample {
                rms: 0.2,
                ..sample(0.001)
            });
        }
        ledger.record(BlockSample {
            rms: 0.05,
            ..sample(0.001)
        });
        let reports = ledger.drain_reports();
        assert_eq!(reports.len(), 1, "包络塌陷必须报一次");
        assert_eq!(reports[0].class, ClickClass::Envelope);
    }

    /// 同一块只报告一次（UI 定时器 20 ms 一次，环里会存活 2.7 s）。
    #[test]
    fn a_block_is_reported_once() {
        let ledger = ClickLedger::new();
        ledger.record(sample(0.9));
        assert_eq!(lines(&ledger).len(), 1);
        assert!(lines(&ledger).is_empty(), "同一块不得重复打印");
        ledger.record(sample(0.9));
        assert_eq!(lines(&ledger).len(), 1, "新块仍要打印");
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
        assert!(lines(&ledger).is_empty());
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

    /// **只有"这一块里有不属于我们的帧"才升级为 WARN。**
    ///
    /// 【为什么这条必须钉住】账本量的是输出波形，它分不开"音乐换了口气"与"我们换错
    /// 了源"。唯一分得开的事实是覆盖：`uncovered > 0` 与 `edge`。现场 30 秒的渲染
    /// 有 200 多条 `envelope` / `step` 读数，它们全是正常音乐动态 —— 一旦这些也进
    /// WARN，真正的故障就没人看得见了。
    #[test]
    fn only_a_block_that_claims_frames_it_did_not_read_is_blamed() {
        let ledger = ClickLedger::new();
        for _ in 0..4 {
            ledger.record(sample(0.001));
        }
        // 完全覆盖、只是内容在动：读数，不是故障。
        ledger.record(BlockSample {
            rms: 0.05,
            ..sample(0.001)
        });
        // 同样的形状，但这一块里有 96 帧不是我们写的：证据。
        // （`rms` 再抬一档，让它在包络判据上同样成立 —— 本条问的是归因，
        //   不是判据，两块都必须先被产出。）
        ledger.record(BlockSample {
            rms: 0.5,
            uncovered: 96,
            ..sample(0.001)
        });
        let reports = ledger.drain_reports();
        assert_eq!(reports.len(), 2, "两条都要产出");
        assert!(
            !reports[0].blamed,
            "覆盖完整的包络起伏只是读数：{:?}",
            reports[0].class
        );
        assert!(
            reports[1].blamed,
            "有未覆盖帧的块必须能归因于我们：{:?}",
            reports[1].class
        );
    }

    /// 覆盖边落在块内同样是一份证据（宿主原声与处理结果在块内翻转）。
    #[test]
    fn a_coverage_edge_inside_the_block_is_blamed() {
        let ledger = ClickLedger::new();
        for _ in 0..4 {
            ledger.record(sample(0.001));
        }
        ledger.record(BlockSample {
            rms: 0.05,
            edge: true,
            ..sample(0.001)
        });
        let reports = ledger.drain_reports();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].blamed, "{:?}", reports[0].class);
    }

    /// 类名与下标必须一一对应：汇总行按 `ClickClass::index` 分桶，错一位就会把
    /// "洞"记成"阶跃"。
    #[test]
    fn every_class_has_its_own_bucket_and_name() {
        let classes = [
            ClickClass::Switch,
            ClickClass::Hole,
            ClickClass::Step,
            ClickClass::Envelope,
        ];
        assert_eq!(classes.len(), ClickClass::COUNT);
        for (expected, class) in classes.iter().enumerate() {
            assert_eq!(class.index(), expected, "{}", class.as_str());
        }
        let names: Vec<&str> = classes.iter().map(|class| class.as_str()).collect();
        assert_eq!(names, vec!["switch", "hole", "step", "envelope"]);
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
