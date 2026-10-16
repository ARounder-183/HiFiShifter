//! 音频线程上的捕获：把宿主播放中的音频按**工程位置**写进定长段。
//!
//! # 为什么是"定长段 + 预分配备用缓冲"
//!
//! 音频线程上不能分配、不能阻塞、不能取会与 UI 争用的锁。而捕获天然需要"越录越长"，
//! 任何"往一个 Vec 里追加"的表示都会在某一块上 realloc —— 那是一次不确定时长的
//! 内存操作，发生在 DAW 的音频回调里。
//!
//! 所以这里的容器是**定长**的：段的平面在分配时就是最终长度，只有 `filled` 这个
//! 整数在动。段满（或宿主位置跳跃）时，音频线程把它封口、从备用缓冲里取一个新的
//! 继续写。备用缓冲由非音频线程预先分配并补充；取不到时**停止捕获并记原因**，
//! 绝不阻塞、绝不分配、绝不覆盖已封口的数据。
//!
//! # 为什么按"块首区间是否首尾相接"切段
//!
//! `projectTimeSamples` 是 SDK 唯一保证始终有效的工程位置（见
//! `audio_abi::ProcessContext`）。它在一趟连续播放内单调，块首的读数权威。
//! 于是"上一块 `[a,b)` 与当前块 `[c,d)` 连续"就等价于 `c == b` —— 一个整数比较。
//!
//! 【为什么不用 `continuousTimeSamples`】SDK 对它的全部表述只有 "project time,
//! without loop" 五个词。宿主绑不绑、跳不跳全看实现，是一张空白支票：在绑了的宿主
//! 上会**少切段**（循环回卷不切，覆盖关系就丢了），在不绑的宿主上会**多切段**。
//! 它只作为诊断读数采集，不参与判据。
//!
//! # 覆盖语义
//!
//! 段之间可以重叠（宿主 seek 回退、循环回卷）。这里不做合并 —— 合并是**时间线视图**
//! 的事（`timeline.rs`），规则是"后来的段胜出"。音频线程只负责把每一块准确地写到
//! 它该在的工程位置上。
//!
//! # 素材位置由**播放**决定，不由按键决定
//!
//! 一次捕获有两个阶段：
//!
//! | 阶段 | 条件 | 音频线程做什么 |
//! |---|---|---|
//! | **待命**（standby） | 已武装，但宿主还没播过 | **什么都不写** |
//! | **录制中** | 宿主播放过至少一块 | 按工程位置写 |
//!
//! 起播的那一刻（第一块 `playing == true`）记下 [`Capture::anchor`] —— 素材的第一帧
//! 就在那里。**这是本模块最重要的一条语义**：用户按录制键的时机与宿主播放的时机是
//! 两件独立的事，而"素材从哪开始"只由后者决定。否则同一个动作（先按录制、再按播放）
//! 会得到两种结果，取决于用户按得多快 —— 行为随按键时机漂移是最难被用户理解的一类缺陷。
//!
//! 停播**不结束**这次捕获：它只是让素材停止增长。用户会停下来听一遍、再接着录，
//! 那是同一次录制的继续。因此 `playing == false` 的块一律跳过 —— 既包括待命期的块，
//! 也包括停播后宿主仍会送来的那些"位置钉住不动"的块（REAPER 就是这样）。

use std::ops::Range;

/// 距单次录制上限还有这么多秒时开始预警。
///
/// 【为什么是 10 秒】它要足够长到用户能**做完手上的事**（唱完这一句、把光标挪开），
/// 又要足够短到不会被当成常驻噪声。10 秒在 120 BPM 下是 5 小节，覆盖绝大多数乐句。
const LIMIT_WARNING_SEC: u64 = 10;

/// 一个段的**结束原因**。它是用户可见诊断的一部分：为什么录到这里就断了。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClosedReason {
    /// 段满（正常切段，不是异常）。
    Full,
    /// 宿主位置跳跃（seek / 循环回卷 / 录音中移动光标）。
    Jump,
    /// 用户停止。
    Stopped,
    /// 到达用户设定的单次录制上限（`TransferOptions::max_duration_sec`）。
    ///
    /// 【为什么它必须是一个**结束**原因】上限的意义就是"不要超过它"：到点必须真的停，
    /// 而不是记一笔然后继续录。它是本文件里唯一由**用户设置**触发的中止，因此也是唯一
    /// 需要提前预警（见 [`Capture::limit_warning`]）的那一个。
    LimitReached,
    /// 全局 PCM 预算耗尽。
    BudgetFull,
    /// 备用缓冲用尽（非音频线程没跟上）。
    NoSpare,
    /// 宿主工程的采样率变了：本 take 的样本域不再一致。
    RateChanged,
}

impl ClosedReason {
    /// 语言无关的分类名；前端按此查 catalog。
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ClosedReason::Full => "full",
            ClosedReason::Jump => "jump",
            ClosedReason::Stopped => "stopped",
            ClosedReason::LimitReached => "limit_reached",
            ClosedReason::BudgetFull => "budget_full",
            ClosedReason::NoSpare => "no_spare",
            ClosedReason::RateChanged => "rate_changed",
        }
    }

    /// 这个原因是否意味着"用户需要知道"（正常切段不需要提示）。
    ///
    /// 【为什么 `LimitReached` 算"需要知道"】录制自己停了，而原因是用户自己设的一个
    /// 数值 —— 不说清就只能是"录着录着就断了"。它与 `NoSpare`（内存没跟上）同一类：
    /// 都是"不是用户按的停"，都必须能被归因。
    pub(crate) fn is_notable(self) -> bool {
        !matches!(self, ClosedReason::Full | ClosedReason::Stopped)
    }

    /// 稳定的编码值（`0` = 没有）。**只增不改**。
    ///
    /// 【为什么不用 `as u8`】显式编码让"哪个数字是哪一条"成为契约的一部分：
    /// 将来插入或重排枚举成员不会静默改变已写出去的数字。
    pub(crate) fn code(self) -> u8 {
        match self {
            ClosedReason::Full => 1,
            ClosedReason::Jump => 2,
            ClosedReason::Stopped => 3,
            ClosedReason::LimitReached => 4,
            ClosedReason::BudgetFull => 5,
            ClosedReason::NoSpare => 6,
            ClosedReason::RateChanged => 7,
        }
    }

    /// [`Self::code`] 的逆。未知编码返回 `None`（而不是猜一个）。
    pub(crate) fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(ClosedReason::Full),
            2 => Some(ClosedReason::Jump),
            3 => Some(ClosedReason::Stopped),
            4 => Some(ClosedReason::LimitReached),
            5 => Some(ClosedReason::BudgetFull),
            6 => Some(ClosedReason::NoSpare),
            7 => Some(ClosedReason::RateChanged),
            _ => None,
        }
    }
}

/// 一个段与它前一段之间的关系。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SegmentLoss {
    /// 与上一段首尾相接（正常）。
    None,
    /// 与上一段之间有跳跃：中间那段工程时间**没有被录到**。
    Gap {
        /// 跳过的工程帧数。
        skipped_frames: u64,
    },
    /// 宿主位置倒退了（seek 回退 / 循环回卷）。
    Backward,
    /// 这是本 take 的第一段。
    First,
}

/// 一段**物理连续**捕获的音频。
///
/// 【不变式】封口后 `span().len() == planes[ch].len()`，且所有平面等长。
/// 活跃期间平面长度是**容量**，`filled` 才是有效长度 —— 封口时截断。
#[derive(Debug)]
pub(crate) struct Segment {
    /// 工程采样域起点（`projectTimeSamples` 的域）。
    pub origin: i64,
    pub sample_rate: u32,
    /// 每声道一条平面。
    pub planes: Vec<Vec<f32>>,
    /// 已写入的帧数。
    pub filled: usize,
    pub loss: SegmentLoss,
    /// 每声道峰值（绝对值的最大值）。
    pub peak: [f32; 2],
    pub closed: ClosedReason,
    /// 该段 PCM 占用的全局预算额度。
    ///
    /// 【为什么额度跟着段走而不是由 store 统一持有】段一旦封口就不再增长，
    /// 它的额度也就固定了。跟着段走意味着"丢弃一个段"立刻归还额度，
    /// 不需要 store 再维护一张并行的账。
    pub _reservation: Option<crate::render::budget::Reservation>,
}

impl Segment {
    /// 本段覆盖的工程采样区间（左闭右开）。
    pub(crate) fn span(&self) -> Range<i64> {
        self.origin..self.origin + self.filled as i64
    }

    /// 段的采样区间是否**全部**为零（且没有非有限值）。
    ///
    /// 【为什么要这个判据】全零段在时间线视图里与相邻内容合并，不单独成片 ——
    /// 这既省内存，也让"用户录了一段静音"与"用户没录"在界面上可区分。
    ///
    /// 段内是否含非有限采样。含 NaN/Inf 的段不能进渲染。
    pub(crate) fn has_nonfinite(&self) -> bool {
        self.planes
            .iter()
            .any(|plane| plane.iter().any(|sample| !sample.is_finite()))
    }
}

/// 一个音频块的输入视图（音频线程构造，**不拥有**数据，因此不分配）。
#[derive(Clone, Copy)]
pub(crate) struct InputBlock<'a> {
    /// 块首的工程位置（`projectTimeSamples`）。
    pub start: i64,
    /// 帧数（`num_samples`）。
    pub frames: usize,
    /// 宿主在 `ProcessContext` 里报的采样率。
    pub sample_rate: f64,
    /// 至多两条输入平面；`None` = 该声道缺失或宿主标记为静音。
    pub planes: [Option<&'a [f32]>; 2],
    /// 宿主此刻是否在播放（`ProcessContext::state` 的 `kPlaying`）。
    ///
    /// 【为什么它必须与音频一起给出来】位置、采样率、样本与"在不在播放"是**同一个块**的
    /// 属性，而捕获对四者缺一不可：没有播放位就分不出"待命"与"录制中"，也就无法保证
    /// "素材从宿主起播那一帧开始"（见文件头的说明）。
    pub playing: bool,
}

impl InputBlock<'_> {
    /// 本块有没有**任何**可用输入。
    pub(crate) fn has_audio(&self) -> bool {
        self.planes.iter().any(|plane| plane.is_some())
    }
}

/// 一次 `on_block` 的结果（给调用方决定输出）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CaptureOutcome {
    /// 已写入（可能同时封了口）。
    Captured,
    /// 本块没有可用输入：写了等长的零，保持时间轴连续。
    SilentInput,
    /// 宿主没给可用的工程位置或采样率：**丢弃**本块（写进去会污染时间轴）。
    NoPosition,
    /// 宿主此刻没在播放：**不写**本块。
    ///
    /// 待命期（还没起播）与停播期（已录了一段、用户按了暂停）都是这一条 ——
    /// 两者的**动作**相同（什么都不写），区别只在状态读数上，由 [`Capture::is_started`]
    /// 回答。分开成两个枚举值会让调用方为同一件事写两个分支。
    NotPlaying,
    /// 采样率与本次捕获不一致：已停止捕获。
    RateChanged,
    /// 备用缓冲用尽：已停止捕获。
    OutOfSpares,
    /// 到达用户设定的单次录制上限：已停止捕获。
    LimitReached,
}

/// 捕获状态机。**只有音频线程写**；非音频线程在锁下补充备用缓冲与收走封口的段。
pub(crate) struct Capture {
    /// 正在写的段；`None` = 尚未开始或已停止。
    active: Option<Segment>,
    /// 预分配的备用缓冲（非音频线程补充）。
    spares: Vec<Segment>,
    /// 已封口、等待非音频线程收走的段。
    sealed: Vec<Segment>,
    /// 上一块的区间终点（连续性判据）。
    expected_next: Option<i64>,
    /// 本次捕获的工程采样率（第一块确定）；`0` = 尚未确定。
    sample_rate: u32,
    /// 声道数（第一块确定）。
    channels: u8,
    /// 单段容量（帧）。
    capacity: usize,
    /// 停止原因；`Some` = 捕获已结束，等非音频线程 promote。
    stopped: Option<ClosedReason>,
    /// 已写入的工程帧总数（含静音块）。
    captured_frames: u64,
    /// 最后写入的工程位置终点。
    last_position: Option<i64>,
    /// 本次捕获的峰值（各段峰值的最大值）。
    peak: [f32; 2],
    /// 宿主是否**已经播过**至少一块（`false` = 待命）。
    ///
    /// 【为什么是一个单调位】它只从假变真一次（用户按下录制之后、宿主第一次播放）。
    /// 待命与录制中的区别就是它 —— 而这两个状态对用户是**完全不同**的两件事
    /// （一个在等外部事件、一个在积累素材），界面必须分得开。
    started: bool,
    /// 单次录制的帧数上限（`0` = 不限）。
    max_frames: u64,
    /// 诊断计数。
    pub gaps: u64,
    pub backwards: u64,
    pub dropouts: u64,
    pub oversized_blocks: u64,
    pub silent_blocks: u64,
}

impl Default for Capture {
    fn default() -> Self {
        Self {
            active: None,
            spares: Vec::new(),
            sealed: Vec::new(),
            expected_next: None,
            sample_rate: 0,
            channels: 0,
            capacity: 0,
            stopped: None,
            captured_frames: 0,
            last_position: None,
            peak: [0.0; 2],
            started: false,
            max_frames: 0,
            gaps: 0,
            backwards: 0,
            dropouts: 0,
            oversized_blocks: 0,
            silent_blocks: 0,
        }
    }
}

impl Capture {
    /// 开始一次捕获。
    ///
    /// `capacity_frames` 是单段容量；`spares` 是调用方**预先分配好**的备用缓冲
    /// （每个都是空的、长度为容量的平面）；`max_frames` 是本次录制的帧数上限
    /// （`0` = 不限）。非音频线程调用。
    pub(crate) fn start(
        &mut self,
        channels: u8,
        capacity_frames: usize,
        mut spares: Vec<Segment>,
        max_frames: u64,
    ) {
        *self = Self {
            spares: std::mem::take(&mut spares),
            channels,
            capacity: capacity_frames,
            max_frames,
            ..Self::default()
        };
    }

    /// 宿主是否**已经播过**至少一块（`false` = 待命）。
    ///
    /// 【为什么界面需要它】待命与录制中是两种不同的等待：前者等的是**用户去按播放**，
    /// 后者等的是**时间积累**。合成一个"已录 0.0 秒"会让待命看起来像失败。
    pub(crate) fn is_started(&self) -> bool {
        self.started
    }

    /// 距单次录制上限不足 [`LIMIT_WARNING_SEC`] 秒。
    ///
    /// 【为什么要有预警】上限到点会**自己停**，而那是用户设的一个数值 —— 不预警的话
    /// 用户只能看到"录着录着就断了"。预警必须早于上限，且必须由后端给（只有它知道
    /// 已捕获多少帧）。
    pub(crate) fn limit_warning(&self) -> bool {
        if self.max_frames == 0 || self.sample_rate == 0 {
            return false;
        }
        let remaining = self.max_frames.saturating_sub(self.captured_frames);
        remaining < u64::from(self.sample_rate) * LIMIT_WARNING_SEC
    }

    /// 是否已经结束（等 promote）。
    pub(crate) fn is_stopped(&self) -> bool {
        self.stopped.is_some()
    }

    /// 停止原因。
    pub(crate) fn stopped_reason(&self) -> Option<ClosedReason> {
        self.stopped
    }

    /// 备用缓冲还剩几个（非音频线程据此补充）。
    pub(crate) fn spare_count(&self) -> usize {
        self.spares.len()
    }

    /// 已封口、尚未被收走的段数。
    pub(crate) fn sealed_len(&self) -> usize {
        self.sealed.len()
    }

    /// 本次捕获**已经覆盖**的工程区间（并集，含正在写的那一段）。
    ///
    /// 【为什么取并集而不是"首段起点 .. 最后写入位置"】宿主 seek 回退或循环回卷会让
    /// 后面的段落在前面：那时"最后写入位置"反而在起点左侧，用区间端点算出来的带子会
    /// 缩回去甚至翻号。界面要的是"我录过哪些地方"，并集才是那个问题的答案。
    pub(crate) fn span(&self) -> Option<Range<i64>> {
        let mut start = i64::MAX;
        let mut end = i64::MIN;
        for segment in self.sealed.iter().chain(self.active.iter()) {
            start = start.min(segment.origin);
            end = end.max(segment.origin + segment.filled as i64);
        }
        if let Some(position) = self.last_position {
            start = start.min(position);
            end = end.max(position);
        }
        (start < end).then_some(start..end)
    }

    /// 已写入的工程帧总数。
    pub(crate) fn captured_frames(&self) -> u64 {
        self.captured_frames
    }

    /// 最后写入的工程位置终点。
    pub(crate) fn last_position(&self) -> Option<i64> {
        self.last_position
    }

    /// 本次捕获的峰值。
    pub(crate) fn peak(&self) -> [f32; 2] {
        self.peak
    }

    /// 本次捕获的工程采样率（`0` = 尚未确定）。
    pub(crate) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// 补充一个备用缓冲（非音频线程）。
    ///
    /// 【为什么要"补充"而不是"预留一大把"】预留 N 个等于预留 N×8 秒的内存，而绝大多数
    /// 录制只用得起一两个。按需补充让内存占用跟随实际录制长度，同时保证音频线程手上
    /// **总有**一个可用的。
    pub(crate) fn push_spare(&mut self, spare: Segment) {
        self.spares.push(spare);
    }

    /// 收走已封口的段（非音频线程）。
    pub(crate) fn drain_sealed(&mut self) -> Vec<Segment> {
        std::mem::take(&mut self.sealed)
    }

    /// 收走全部段：封口当前段 + 已封口的（promote 用，非音频线程）。
    pub(crate) fn finish(&mut self, reason: ClosedReason) -> Vec<Segment> {
        if let Some(active) = self.active.take() {
            self.sealed.push(seal(active, reason));
        }
        if self.stopped.is_none() {
            self.stopped = Some(reason);
        }
        self.drain_sealed()
    }

    /// 音频线程入口。
    pub(crate) fn on_block(&mut self, block: InputBlock<'_>) -> CaptureOutcome {
        if self.stopped.is_some() {
            return CaptureOutcome::Captured;
        }
        // 【为什么"没在播放"必须排在位置与采样率之前】停播后宿主仍会送块（REAPER 会把
        // 位置钉在光标处反复 process），而那些块的位置往往**与已录内容不相邻** ——
        // 让它们走进下面的连续性判据会凭空多出一次"跳跃"，把一次正常暂停记成一次异常。
        // 先按播放位挡掉，暂停就是暂停。
        if !block.playing {
            return CaptureOutcome::NotPlaying;
        }
        // 【为什么非法位置要丢弃而不是当 0】把一块音频写到工程位置 0 会**污染**时间轴：
        // 用户听到的那一小段会出现在开头。丢弃只损失这一块，且被计数。
        let Some(rate) = normalize_rate(block.sample_rate) else {
            self.dropouts = self.dropouts.saturating_add(1);
            return CaptureOutcome::NoPosition;
        };
        // 起播：素材从这一块开始。**待命期一个样本都不写** —— 这就是"素材位置由播放
        // 决定、不由按键决定"的落点（见文件头）。
        //
        // 【为什么排在采样率校验之后】一块"在播放但报不出采样率"的块不算起播：
        // 那时我们连样本域都不知道，没有资格宣称素材已经开始。
        if !self.started {
            self.started = true;
        }
        if block.frames == 0 {
            return CaptureOutcome::Captured;
        }
        if self.sample_rate == 0 {
            self.sample_rate = rate;
            self.channels = channels_of(&block);
        } else if self.sample_rate != rate {
            // 工程率变了：本 take 里已录样本的"秒"与后续块的"秒"不再可比。
            self.stop(ClosedReason::RateChanged);
            return CaptureOutcome::RateChanged;
        }

        // 连续性：上一块的终点是否正好接上这一块的首点。
        let loss = match self.expected_next {
            None => SegmentLoss::First,
            Some(expected) if expected == block.start => SegmentLoss::None,
            Some(expected) if block.start > expected => {
                self.gaps = self.gaps.saturating_add(1);
                SegmentLoss::Gap {
                    skipped_frames: (block.start - expected) as u64,
                }
            }
            Some(_) => {
                // 位置倒退：宿主 seek 回退或**循环回卷**。
                //
                // 【为什么是"另起一段"而不是"丢弃"】循环录制是本功能的主要用法之一：
                // 用户让宿主循环播放，唱一遍、再唱一遍。每一圈都是**新内容**，
                // 丢掉第二圈等于把用户刚唱的那一遍扔了。另起一段之后，覆盖关系交给
                // 时间线视图按"后来的胜出"解决（`timeline.rs`），默认留下最后一圈 ——
                // 这正是 Melodyne 式 Transfer 的默认语义。
                self.backwards = self.backwards.saturating_add(1);
                SegmentLoss::Backward
            }
        };

        // 段满或跳跃 ⇒ 换段。跳跃必须换段，否则一个段里会含两段不相邻的工程时间。
        let needs_new = match &self.active {
            None => true,
            Some(active) => {
                active.filled + block.frames > self.capacity
                    || !matches!(loss, SegmentLoss::None)
            }
        };
        if needs_new {
            if let Some(active) = self.active.take() {
                let reason = if matches!(loss, SegmentLoss::None) {
                    ClosedReason::Full
                } else {
                    ClosedReason::Jump
                };
                self.sealed.push(seal(active, reason));
            }
            let Some(mut spare) = self.spares.pop() else {
                // 备用缓冲用尽：停止而不是阻塞或分配。用户看到的是"录制已停止"，
                // 而不是 DAW 卡顿或数据被覆盖。
                self.stop(ClosedReason::NoSpare);
                return CaptureOutcome::OutOfSpares;
            };
            spare.origin = block.start;
            spare.sample_rate = rate;
            spare.loss = loss;
            spare.filled = 0;
            spare.peak = [0.0; 2];
            spare.closed = ClosedReason::Full;
            self.active = Some(spare);
        }

        let Some(active) = self.active.as_mut() else {
            // 上面刚建过；走到这里说明 `capacity == 0`（未正确 start）。
            self.stop(ClosedReason::NoSpare);
            return CaptureOutcome::OutOfSpares;
        };

        // 【为什么超大块只截断并计数】容量是 8 秒而宿主单块通常 ≤ 6 秒；超过说明宿主
        // 行为异常。截断只损失超出部分，而"整块丢弃"会留下一个洞。
        let writable = block.frames.min(self.capacity.saturating_sub(active.filled));
        if writable < block.frames {
            self.oversized_blocks = self.oversized_blocks.saturating_add(1);
        }
        if writable == 0 {
            return CaptureOutcome::Captured;
        }
        let silent = !block.has_audio();
        if silent {
            self.silent_blocks = self.silent_blocks.saturating_add(1);
        }
        for (channel, plane) in active.planes.iter_mut().enumerate() {
            let target = &mut plane[active.filled..active.filled + writable];
            match block.planes.get(channel).copied().flatten() {
                Some(source) if source.len() >= writable => {
                    target.copy_from_slice(&source[..writable]);
                }
                // 声道缺失 / 宿主标记静音 / 平面比声明的帧数短：写零。
                // 【为什么不能跳过不写】跳过会把**上一次使用的残留**留在缓冲里 ——
                // 那听起来是随机的爆音，且完全不可归因。
                _ => target.fill(0.0),
            }
            let peak = target.iter().fold(0.0_f32, |acc, value| {
                if value.is_finite() {
                    acc.max(value.abs())
                } else {
                    acc
                }
            });
            active.peak[channel] = active.peak[channel].max(peak);
        }
        active.filled += writable;
        self.expected_next = Some(block.start + writable as i64);
        self.captured_frames = self.captured_frames.saturating_add(writable as u64);
        self.last_position = self.expected_next;
        self.peak[0] = self.peak[0].max(active.peak[0]);
        self.peak[1] = self.peak[1].max(active.peak[1]);

        // 恰好写满 ⇒ 立刻封口，让非音频线程尽早拿到完整段。
        if active.filled >= self.capacity {
            if let Some(active) = self.active.take() {
                self.sealed.push(seal(active, ClosedReason::Full));
            }
        }
        // 【为什么上限在音频线程判】只有这里知道"已经捕获了多少帧"，而判据只是一次
        // 整数比较。放到非音频线程会有一个块的滞后 —— 而上限的全部意义就是**不要超**。
        if self.max_frames > 0 && self.captured_frames >= self.max_frames {
            self.stop(ClosedReason::LimitReached);
            return CaptureOutcome::LimitReached;
        }
        if silent {
            CaptureOutcome::SilentInput
        } else {
            CaptureOutcome::Captured
        }
    }

    /// 停止捕获（幂等）。
    fn stop(&mut self, reason: ClosedReason) {
        if let Some(active) = self.active.take() {
            self.sealed.push(seal(active, reason));
        }
        if self.stopped.is_none() {
            self.stopped = Some(reason);
        }
    }
}

/// 封口：把平面截到实际写入长度，让"平面长度 == 段长度"成为交付不变式。
///
/// 【为什么 `truncate` 可以在音频线程做】对 `f32` 它只改长度字段，不释放、不分配、
/// 不移动数据。
fn seal(mut segment: Segment, reason: ClosedReason) -> Segment {
    let filled = segment.filled;
    for plane in &mut segment.planes {
        plane.truncate(filled);
    }
    segment.closed = reason;
    segment
}

/// 采样率必须是有限的正常值，否则本块没有可用的时间基。
fn normalize_rate(rate: f64) -> Option<u32> {
    if !rate.is_finite() || rate <= 0.0 {
        return None;
    }
    let rounded = rate.round();
    if !(8000.0..=384_000.0).contains(&rounded) {
        return None;
    }
    Some(rounded as u32)
}

/// 本块实际有几个声道（只取前两个）。
fn channels_of(block: &InputBlock<'_>) -> u8 {
    let count = block.planes.iter().filter(|plane| plane.is_some()).count();
    count.clamp(1, 2) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 造一个长度为 `frames` 的备用缓冲。
    fn spare(capacity: usize, channels: u8) -> Segment {
        Segment {
            origin: 0,
            sample_rate: 0,
            planes: vec![vec![0.0; capacity]; channels as usize],
            filled: 0,
            loss: SegmentLoss::First,
            peak: [0.0; 2],
            closed: ClosedReason::Full,
            _reservation: None,
        }
    }

    /// 造一个只含一条声道的块，样本值等于帧号。
    fn block(start: i64, frames: usize, rate: f64) -> (Vec<f32>, InputBlock<'static>) {
        // 泄漏一份样本缓冲，让 `InputBlock` 的借用满足 'static（测试专用）。
        let samples: &'static [f32] = Box::leak(
            (0..frames)
                .map(|index| index as f32 / 1024.0)
                .collect::<Vec<f32>>()
                .into_boxed_slice(),
        );
        (
            samples.to_vec(),
            InputBlock {
                start,
                frames,
                sample_rate: rate,
                planes: [Some(samples), None],
                playing: true,
            },
        )
    }

    /// 同一个块，但宿主**没在播放**（停播后仍会 process 的那种）。
    fn paused(view: InputBlock<'_>) -> InputBlock<'_> {
        InputBlock {
            playing: false,
            ..view
        }
    }

    fn started(capacity: usize, spares: usize) -> Capture {
        let mut capture = Capture::default();
        capture.start(
            1,
            capacity,
            (0..spares).map(|_| spare(capacity, 1)).collect(),
            0,
        );
        capture
    }

    /// 连续块必须写进**同一个**段，且样本落在正确的段内偏移上。
    #[test]
    fn contiguous_blocks_accumulate_into_one_segment() {
        let mut capture = started(16, 2);
        for start in [0, 4, 8] {
            let (_, view) = block(start, 4, 48_000.0);
            assert_eq!(capture.on_block(view), CaptureOutcome::Captured);
        }
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed.len(), 1);
        let segment = &sealed[0];
        assert_eq!(segment.span(), 0..12);
        assert_eq!(segment.filled, 12);
        assert_eq!(segment.planes[0].len(), 12, "封口必须截到实际写入长度");
        // 第 0 块的样本值 = 帧号/1024，因此段内偏移 i 的值就是 i/1024。
        assert!((segment.planes[0][5] - 1.0 / 1024.0).abs() < 1e-9);
        assert_eq!(segment.loss, SegmentLoss::First);
    }

    /// **跳跃必须切段**：一个段里不能含两段不相邻的工程时间。
    ///
    /// 【为什么这是核心不变量】段的定义就是"物理连续"。若跳跃不切段，段的
    /// `origin + 偏移` 与真实工程位置就对不上，整条时间轴会静默错位。
    #[test]
    fn a_position_jump_starts_a_new_segment_and_records_the_gap() {
        let mut capture = started(16, 2);
        let (_, first) = block(0, 4, 48_000.0);
        assert_eq!(capture.on_block(first), CaptureOutcome::Captured);
        // 跳到工程位置 100：中间 96 帧没有被录到。
        let (_, second) = block(100, 4, 48_000.0);
        assert_eq!(capture.on_block(second), CaptureOutcome::Captured);
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed.len(), 2);
        assert_eq!(sealed[0].span(), 0..4);
        assert_eq!(sealed[0].closed, ClosedReason::Jump);
        assert_eq!(sealed[1].span(), 100..104);
        assert_eq!(
            sealed[1].loss,
            SegmentLoss::Gap {
                skipped_frames: 96
            }
        );
        assert_eq!(capture.gaps, 1);
    }

    /// 段满自动封口，下一块落到新段上（备用缓冲被消费）。
    #[test]
    fn a_full_segment_is_sealed_and_replaced_from_the_spare_pool() {
        let mut capture = started(8, 2);
        for start in [0, 4, 8, 12] {
            let (_, view) = block(start, 4, 48_000.0);
            capture.on_block(view);
        }
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed.len(), 2, "8 帧一段，共 16 帧 ⇒ 2 段");
        assert_eq!(sealed[0].span(), 0..8);
        assert_eq!(sealed[1].span(), 8..16);
        assert_eq!(sealed[0].closed, ClosedReason::Full);
    }

    /// 备用缓冲用尽时**停止捕获**，而不是阻塞或分配。
    #[test]
    fn running_out_of_spares_stops_capture_instead_of_blocking() {
        let mut capture = started(8, 1);
        // 第一段用掉唯一的备用；第二段无处可去。
        for start in [0, 4, 8, 12] {
            let (_, view) = block(start, 4, 48_000.0);
            capture.on_block(view);
        }
        assert!(capture.is_stopped());
        assert_eq!(capture.stopped_reason(), Some(ClosedReason::NoSpare));
    }

    /// 位置倒退（循环回卷）**另起一段**，覆盖关系交给时间线视图按"后来者胜出"解决。
    ///
    /// 【为什么不是丢弃】循环录制是本功能的主要用法：用户让宿主循环播放，唱一遍、
    /// 再唱一遍。丢掉第二圈等于把用户刚唱的那一遍扔了。
    #[test]
    fn a_backward_jump_starts_a_new_segment_so_the_last_pass_can_win() {
        let mut capture = started(64, 2);
        let (_, first) = block(100, 4, 48_000.0);
        capture.on_block(first);
        let (_, backward) = block(0, 4, 48_000.0);
        capture.on_block(backward);
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed.len(), 2, "倒退的块必须另起一段");
        assert_eq!(sealed[0].span(), 100..104);
        assert_eq!(sealed[1].span(), 0..4);
        assert_eq!(sealed[1].loss, SegmentLoss::Backward);
        assert_eq!(capture.backwards, 1);
    }

    /// 采样率变化必须停止捕获：已录样本的"秒"与新块不再可比。
    #[test]
    fn a_sample_rate_change_stops_the_capture() {
        let mut capture = started(64, 2);
        let (_, first) = block(0, 4, 48_000.0);
        capture.on_block(first);
        let (_, changed) = block(4, 4, 44_100.0);
        assert_eq!(capture.on_block(changed), CaptureOutcome::RateChanged);
        assert_eq!(capture.stopped_reason(), Some(ClosedReason::RateChanged));
    }

    /// 没有可用输入时写**等长的零**，保持时间轴连续。
    ///
    /// 【为什么不能跳过不写】跳过会留下上一次使用的残留（爆音），或让段长度与工程
    /// 时间不符（后面所有内容错位）。
    #[test]
    fn a_block_without_input_writes_silence_of_the_same_length() {
        let mut capture = started(64, 2);
        let (_, view) = block(0, 4, 48_000.0);
        capture.on_block(view);
        let silent = InputBlock {
            start: 4,
            frames: 4,
            sample_rate: 48_000.0,
            planes: [None, None],
            playing: true,
        };
        assert_eq!(capture.on_block(silent), CaptureOutcome::SilentInput);
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed[0].span(), 0..8);
        assert_eq!(&sealed[0].planes[0][4..8], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(capture.silent_blocks, 1);
    }

    /// 非法位置/采样率的块被丢弃，不污染时间轴。
    #[test]
    fn a_block_without_a_usable_position_is_dropped() {
        let mut capture = started(64, 2);
        let bad = InputBlock {
            start: 0,
            frames: 4,
            sample_rate: f64::NAN,
            planes: [None, None],
            playing: true,
        };
        assert_eq!(capture.on_block(bad), CaptureOutcome::NoPosition);
        assert_eq!(capture.dropouts, 1);
        assert!(capture.finish(ClosedReason::Stopped).is_empty());
    }

    /// 峰值按声道分别跟踪。
    #[test]
    fn peak_is_tracked_per_channel() {
        let left: &'static [f32] = Box::leak(vec![-0.5, 0.25].into_boxed_slice());
        let right: &'static [f32] = Box::leak(vec![0.75, -0.1].into_boxed_slice());
        let mut capture = Capture::default();
        capture.start(2, 64, vec![spare(64, 2)], 0);
        let view = InputBlock {
            start: 0,
            frames: 2,
            sample_rate: 44_100.0,
            planes: [Some(left), Some(right)],
            playing: true,
        };
        capture.on_block(view);
        let sealed = capture.finish(ClosedReason::Stopped);
        assert!((sealed[0].peak[0] - 0.5).abs() < 1e-6);
        assert!((sealed[0].peak[1] - 0.75).abs() < 1e-6);
    }

    /// 没有输入时写下的段**全为零**，且不含非有限值。
    ///
    /// 【为什么这条仍然要钉住】投影层按素材带上的窗口判静音（`timeline::Tape::is_silent`），
    /// 而那个判据读的就是这里写下的样本 —— 如果捕获在这里留下的是残留值而不是零，
    /// "全静音的录制"就会在时间轴上变成一段噪声。
    #[test]
    fn a_segment_without_input_is_all_zero() {
        let mut capture = started(64, 2);
        let silent = InputBlock {
            start: 0,
            frames: 8,
            sample_rate: 48_000.0,
            planes: [None, None],
            playing: true,
        };
        capture.on_block(silent);
        let sealed = capture.finish(ClosedReason::Stopped);
        assert!(sealed[0]
            .planes
            .iter()
            .all(|plane| plane.iter().all(|sample| *sample == 0.0)));
        assert!(!sealed[0].has_nonfinite());
    }

    /// **待命期一个样本都不写**：宿主没播放时按下的录制不产生任何素材。
    ///
    /// 【为什么这是本模块最重要的回归点】"素材位置由播放决定、不由按键决定"这条语义
    /// 全靠它。待命期若写了东西，素材就会落在"按录制时宿主恰好停在哪"这个与用户
    /// 意图无关的位置上，而同一个动作（先按录制、再按播放）会因按键快慢得到两种结果。
    #[test]
    fn standby_writes_nothing_until_the_host_plays() {
        let mut capture = started(64, 2);
        // 宿主没播放：反复送块也不该有任何素材。
        for start in [0, 4, 8, 12] {
            let (_, view) = block(start, 4, 48_000.0);
            assert_eq!(
                capture.on_block(paused(view)),
                CaptureOutcome::NotPlaying,
                "宿主没在播放时不该写任何块"
            );
        }
        assert!(!capture.is_started(), "没播过就还是待命");
        assert_eq!(capture.captured_frames(), 0);
        assert!(capture.finish(ClosedReason::Stopped).is_empty());
    }

    /// **素材的第一帧是宿主起播的那一帧，不是按下录制时宿主停在哪。**
    ///
    /// 【为什么这条钉的是"位置随播放走"】它是用户反馈"先点录制再播放录不到"的直接
    /// 修复点：宿主从 20 秒处起播，素材就必须从 20 秒开始 —— 哪怕用户在宿主停在
    /// 0 秒时就按下了录制。
    #[test]
    fn the_first_frame_follows_the_host_not_the_button() {
        let mut capture = started(64, 2);
        // 用户按下录制时宿主停在 0，并在那里反复 process。
        for start in [0, 0, 0] {
            let (_, view) = block(start, 4, 48_000.0);
            assert_eq!(capture.on_block(paused(view)), CaptureOutcome::NotPlaying);
        }
        assert!(!capture.is_started(), "宿主没播放就还是待命");
        // 用户按了播放，宿主从 20 帧处开始。
        let (_, first) = block(20, 4, 48_000.0);
        assert_eq!(capture.on_block(first), CaptureOutcome::Captured);
        assert!(capture.is_started());
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(
            sealed[0].span(),
            20..24,
            "素材起点必须是宿主起播的位置，而不是按下录制时宿主停的位置"
        );
    }

    /// **停播不封口**：暂停只让素材停止增长，重新起播且位置相接时无缝续录。
    ///
    /// 【为什么停播不能结束本次录制】用户会停下来听一遍、再接着录，那是同一次录制的
    /// 继续。若停播就 promote，用户必须重新按一次录制 —— 而那一次会重新待命，
    /// 于是"接着录"这个动作变得不可能。
    #[test]
    fn a_paused_transport_keeps_the_take_and_resumes_seamlessly() {
        let mut capture = started(64, 2);
        let (_, first) = block(0, 4, 48_000.0);
        capture.on_block(first);
        // 停播：宿主把位置钉在 4 处反复 process（REAPER 的行为）。
        for _ in 0..5 {
            let (_, stalled) = block(4, 4, 48_000.0);
            assert_eq!(capture.on_block(paused(stalled)), CaptureOutcome::NotPlaying);
        }
        assert!(!capture.is_stopped(), "停播不该结束这次捕获");
        // 重新起播，位置正好接上。
        let (_, resumed) = block(4, 4, 48_000.0);
        assert_eq!(capture.on_block(resumed), CaptureOutcome::Captured);
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed.len(), 1, "续录必须落在同一个段里");
        assert_eq!(sealed[0].span(), 0..8);
        assert_eq!(capture.backwards, 0, "暂停不该被记成一次位置倒退");
        assert_eq!(capture.gaps, 0, "暂停不该被记成一次跳跃");
    }

    /// 停播后从**别处**续录：位置不相接 ⇒ 按跳跃切段，而不是被当成连续。
    #[test]
    fn resuming_somewhere_else_is_a_jump_not_a_continuation() {
        let mut capture = started(64, 2);
        let (_, first) = block(0, 4, 48_000.0);
        capture.on_block(first);
        let (_, stalled) = block(4, 4, 48_000.0);
        capture.on_block(paused(stalled));
        let (_, elsewhere) = block(100, 4, 48_000.0);
        assert_eq!(capture.on_block(elsewhere), CaptureOutcome::Captured);
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed.len(), 2);
        assert_eq!(sealed[1].loss, SegmentLoss::Gap { skipped_frames: 96 });
    }

    /// **到达上限就真的停**，且原因可归因。
    ///
    /// 【为什么必须真的停】上限的意义就是"不要超过它"。记一笔然后继续录等于没有上限，
    /// 而用户会据此以为"10 分钟后自动停"是可信的 —— 一个不生效的上限比没有上限更糟。
    #[test]
    fn the_duration_limit_stops_the_capture() {
        let mut capture = Capture::default();
        capture.start(1, 64, vec![spare(64, 1)], 8);
        let (_, first) = block(0, 4, 48_000.0);
        assert_eq!(capture.on_block(first), CaptureOutcome::Captured);
        let (_, second) = block(4, 4, 48_000.0);
        assert_eq!(capture.on_block(second), CaptureOutcome::LimitReached);
        assert!(capture.is_stopped());
        assert_eq!(capture.stopped_reason(), Some(ClosedReason::LimitReached));
        // 上限是**结束**原因：它必须让用户知道。
        assert!(ClosedReason::LimitReached.is_notable());
    }

    /// 上限预警在到达前 10 秒触发。
    ///
    /// 【为什么要走真实采样率】采样率是"帧 ↔ 秒"换算的唯一依据，而捕获只接受
    /// `8000..=384000`（`normalize_rate`）—— 用一个玩具率（如 1 Hz）写测试会让每一块
    /// 都被判成"没有可用位置"，于是测的其实是别的东西。
    #[test]
    fn the_limit_warning_fires_ten_seconds_early() {
        let rate = 48_000.0;
        // 上限 20 秒 = 960000 帧；段容量够大，让 11 秒素材落在一个段里。
        let mut capture = Capture::default();
        capture.start(1, 600_000, vec![spare(600_000, 1)], 960_000);
        // 1 秒一块。
        for second in 0..11_i64 {
            let (_, view) = block(second * 48_000, 48_000, rate);
            capture.on_block(view);
            if second == 0 {
                assert!(
                    !capture.limit_warning(),
                    "还剩 19 秒时不该预警（预警线是 10 秒）"
                );
            }
        }
        // 已录 11 秒 ⇒ 还剩 9 秒 < 10 秒。
        assert!(capture.limit_warning(), "还剩 9 秒时必须预警");
        assert!(!capture.is_stopped(), "预警不是停止：还没到上限");
    }

    /// 不限时长（`max_frames == 0`）时永不预警、永不自停。
    #[test]
    fn an_unlimited_capture_never_warns() {
        let mut capture = started(64, 2);
        for start in [0, 4, 8, 12] {
            let (_, view) = block(start, 4, 48_000.0);
            capture.on_block(view);
        }
        assert!(!capture.limit_warning());
        assert!(!capture.is_stopped());
    }
}
