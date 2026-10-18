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

/// resync 接缝处的去咔长度（帧）。48 kHz 下约 5.3 ms。
///
/// 【为什么必须在捕获侧做】空洞两端的样本都在源 PCM 里；在源上做淡入淡出后，
/// 处理结果与原始磁带都会听到同一个平滑边界，不需要输出层再猜哪里是接缝。
const DECLICK_FRAMES: usize = 256;
/// 向前桥接的上限（秒）。超过它就不是"记账纠正"，而是真实 seek。
const MAX_BRIDGE_SEC: u64 = 2;

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
    /// 本段所属循环 pass 的工程起点（`None` = 不是循环回卷产生的段）。
    ///
    /// 【为什么跟着段走】一个 pass 可能被容量切成多段；投影需要把它们认成同一圈，
    /// 才能把“最后一圈原子替换整圈”落在同一组里。它不影响音频，只影响投影分组。
    pub loop_start: Option<i64>,
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

/// 本块属于 VST3 的哪条进程流。
///
/// 【为什么这不是诊断字段】SDK 明确允许宿主在 `kRealtime` 与 `kPrefetch` 之间切换，
/// 且切换不需要 `setupProcessing`（`ivstaudioprocessor.h:138-143`）。两条流的
/// `projectTimeSamples` 节奏、块长与“此刻是否被听到”都不同；把它们塞进同一份
/// `last_block_*` / `expected_next` 会让一条流替另一条流作证。捕获与输出都必须
/// 按流隔离状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProcessStream {
    /// `kRealtime`（0）。
    Realtime,
    /// `kPrefetch`（1）。
    Prefetch,
    /// `kOffline`（2）。
    Offline,
}

impl ProcessStream {
    pub(crate) fn from_mode(mode: i32) -> Self {
        match mode {
            1 => Self::Prefetch,
            2 => Self::Offline,
            _ => Self::Realtime,
        }
    }

    pub(crate) fn index(self) -> usize {
        match self {
            Self::Realtime => 0,
            Self::Prefetch => 1,
            Self::Offline => 2,
        }
    }

    /// 捕获侧预分配缓冲的下标：离线与实时互斥，共用 0。
    fn pending_slot(self) -> usize {
        match self {
            Self::Prefetch => 1,
            Self::Realtime | Self::Offline => 0,
        }
    }

    /// 稳定编码（只增不改，诊断用）。
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::Realtime => 1,
            Self::Prefetch => 2,
            Self::Offline => 3,
        }
    }

    pub(crate) fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Realtime),
            2 => Some(Self::Prefetch),
            3 => Some(Self::Offline),
            _ => None,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Realtime => "realtime",
            Self::Prefetch => "prefetch",
            Self::Offline => "offline",
        }
    }
}

/// 一块音频相对“同一流上一块”的传输相位。
///
/// 【为什么不能只看 `kPlaying`】宿主可以在停播时仍置 `kPlaying`（FL Studio），也可以
/// 在起播第一块滞后置位（VEGAS/REAPER）。真正能跨宿主成立的判据只有“位置是否在
/// 推进”。`Resting` 表示同一块首已经出现过：这不是新素材，也不是新的输出位置。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum TransportPhase {
    /// 本流首块，或 `kPlaying`/`kRecording` 上升沿后的第一块。
    #[default]
    Start,
    /// 块首相对上一块发生了变化（前进、后退、seek、循环回卷）。
    Rolling,
    /// 同一块首已经至少出现过一次：位置钉住 / 停播重复。
    Resting,
}

impl TransportPhase {
    /// 稳定编码（只增不改，诊断用）。
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::Start => 1,
            Self::Rolling => 2,
            Self::Resting => 3,
        }
    }
}

/// 本块有效工程位置的来源。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum PositionSource {
    /// `projectTimeSamples`（正常路径）。
    #[default]
    Project,
    /// 工程位置钉住、`continuousTimeSamples` 在前进：用增量锚定。
    Continuous,
    /// 工程位置钉住、`projectTimeMusic` 在前进：用音乐时间增量锚定。
    Music,
    /// 循环播放：使用会回卷的工程位置（不是 continuous）。
    Cycle,
    /// 工程位置钉住、但输入音频在流动：位置由本插件按块长自行推进。
    ///
    /// 【为什么这是合法的记账】宿主在起播窗口里可能把 `projectTimeSamples` 钉在
    /// 起点不动，却已经按自己的块长连续送时间上相邻的样本。此时"每块前进 frames"
    /// 是唯一自洽的换算 —— 它同时也是**可核对**的：宿主位置一恢复前进就立刻以
    /// 宿主为准（记一次 `position_resyncs`）。
    Synthesized,
}

impl PositionSource {
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::Project => 1,
            Self::Continuous => 2,
            Self::Music => 3,
            Self::Cycle => 4,
            Self::Synthesized => 5,
        }
    }

    pub(crate) fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Project),
            2 => Some(Self::Continuous),
            3 => Some(Self::Music),
            4 => Some(Self::Cycle),
            5 => Some(Self::Synthesized),
            _ => None,
        }
    }
}

/// 一个宿主块在循环边界处切分出的一段。
///
/// 【为什么需要它】SDK 明确允许宿主不按 cycle 边界切块；一个块的后半段可能已经是
/// 下一圈的开头。按原位置整块写入会把那一段放到 loop 终点之后，形成悬挂尾巴。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SubBlock {
    /// 这一段在工程时间轴上的起点（回卷后是 cycle start）。
    pub position: i64,
    /// 这一段在宿主块内的帧偏移。
    pub input_offset: usize,
    pub frames: usize,
}

impl SubBlock {
    pub(crate) const EMPTY: Self = Self {
        position: 0,
        input_offset: 0,
        frames: 0,
    };
}

/// 一个宿主块的切分结果（1 或 2 段）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockSplit {
    pub parts: [SubBlock; 2],
    pub count: u8,
}

impl Default for BlockSplit {
    fn default() -> Self {
        Self {
            parts: [SubBlock::EMPTY; 2],
            count: 1,
        }
    }
}

impl BlockSplit {
    pub(crate) fn whole(position: i64, frames: usize) -> Self {
        Self {
            parts: [
                SubBlock {
                    position,
                    input_offset: 0,
                    frames,
                },
                SubBlock::EMPTY,
            ],
            count: 1,
        }
    }

    pub(crate) fn split(first: SubBlock, second: SubBlock) -> Self {
        Self {
            parts: [first, second],
            count: 2,
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &SubBlock> {
        self.parts[..self.count.clamp(1, 2) as usize].iter()
    }
}

/// 一块音频的进程流上下文（与样本视图分开：`InputBlock` 仍然只描述音频）。
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockContext {
    pub stream: ProcessStream,
    /// 本块的 `kPlaying`（与样本一起使用的强证据）。
    pub playing: bool,
    /// `ProcessContext::state & kRecording`。
    pub recording: bool,
    /// 本流内：块首、块首移动、或同一块首的重复。
    pub phase: TransportPhase,
    /// `kPlaying` 从 0 变 1 的那一块。
    pub playing_rising: bool,
    /// `kRecording` 从 0 变 1 的那一块。
    pub recording_rising: bool,
    /// 本块由一次循环回卷开始（块首可以与上一块相同）。
    pub cycle_wrap: bool,
    /// 循环起点（工程采样域；`None` = 宿主没给可用的 cycle 元数据）。
    pub cycle_start: Option<i64>,
    /// 本块开始了一个新的 pass（回卷或位置后退）。
    pub new_pass: bool,
    /// 本段属于循环切分的第几段（0 = 第一段；只有 0 更新候选/几何计数）。
    pub part: u8,
    /// 本流最近是否观测到"输入内容在变化"（音频在流动）。
    ///
    /// 【为什么捕获需要它】停播时宿主会重复投递同一区间（内容逐块**相同**）；
    /// 循环回卷时块首也会回到同一位置，但内容仍在变化。只按几何无法区分两者 ——
    /// 前者必须丢弃，后者必须另起一段。
    pub advancing: bool,
    /// 本块是**位置回退结束后的第一块**（`Synthesized`/`Continuous`/`Music` → `Project`）。
    ///
    /// 【为什么捕获需要它】起播窗口里我们自己推进的位置与宿主恢复后的位置可能差
    /// 一小段。那不是 seek，是我们记账的纠正：向前用静音桥接、向后裁掉多记的尾巴，
    /// 两种都**继续同一段**，不能切 clip。
    pub resync: bool,
    /// 允许当作"记账纠正"的最大帧差（本流钉住期间送来的帧数 + 本块）。
    pub resync_limit_frames: u64,
}

impl BlockContext {
    pub(crate) fn realtime(flow: bool) -> Self {
        Self {
            stream: ProcessStream::Realtime,
            playing: false,
            recording: false,
            phase: if flow {
                TransportPhase::Rolling
            } else {
                TransportPhase::Start
            },
            playing_rising: false,
            recording_rising: false,
            cycle_wrap: false,
            cycle_start: None,
            new_pass: false,
            part: 0,
            advancing: flow,
            resync: false,
            resync_limit_frames: 0,
        }
    }
}

/// 交接所需的最少连续块数。
///
/// 【为什么不是 8】块长由宿主决定；0.7 s 的块长下 8 块是 5.6 s，静音/错误的流
/// 会长期占着捕获。块数只作为“避免单块交错”的下限，真正的量级由
/// [`HANDOVER_SEC`] 决定。
const HANDOVER_BLOCKS: u32 = 2;
/// 交接所需的连续供料时长（秒）。长块宿主上一块即可超过它。
const HANDOVER_SEC: f64 = 0.12;

/// 单条进程流的候选状态（只存整数，不分配）。
#[derive(Debug, Clone, Copy, Default)]
struct StreamCandidate {
    seen: bool,
    last_start: i64,
    last_frames: usize,
    last_end: i64,
    /// 本流内部首尾相接的连续块数。
    contiguous: u32,
    /// 本流内部首尾相接的连续帧数（长块宿主上比块数更有意义）。
    contiguous_frames: u32,
    playing_blocks: u32,
    recording_blocks: u32,
    non_silent_blocks: u32,
    /// 本流内位置真的在推进（相位非 Resting）的块数 / 帧数。
    moving_blocks: u32,
    moving_frames: u32,
    /// 本流内非静音的帧数。
    non_silent_frames: u32,
    playing_frames: u32,
    recording_frames: u32,
    /// 最后一次见到的全局块序号（诊断用）。
    last_seen_seq: u64,
}

/// 候选流的证据强度（用于首次选举的平局）。
///
/// 【为什么把 recording/playing 放在最前】它们是宿主对“这一块是不是真实播放/录音”
/// 的直接声明；位置推进与非静音是辅助证据。元组按字典序比较，越靠后的项只在
/// 前面完全相等时才起作用。
fn candidate_evidence(candidate: &StreamCandidate) -> (u8, u32, u32, u32) {
    let tier = if candidate.recording_frames > 0 {
        3
    } else if candidate.playing_frames > 0 {
        2
    } else if candidate.moving_frames > 0 {
        1
    } else {
        0
    };
    (
        tier,
        candidate.moving_frames,
        candidate.non_silent_frames,
        candidate.contiguous_frames,
    )
}

/// 一条流的 lookbehind 缓冲。
///
/// 【为什么每流一份】候选首块在选举之前就要暂存；两条流共用一个 `pending` 会让
/// 后到的流覆盖先到的流。备用缓冲在 `start()` 里一次性分配，音频线程只改 `filled`。
struct Pending {
    buffer: Option<Segment>,
    ready: bool,
    key: Option<(i64, u32, u32)>,
    /// 这块候选属于哪条流。离线与实时共用 0 号槽，因此不能只看槽位。
    stream: Option<ProcessStream>,
}

impl Default for Pending {
    fn default() -> Self {
        Self {
            buffer: None,
            ready: false,
            key: None,
            stream: None,
        }
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
    /// 同一流把**同一个工程区间**又送了一遍：不是新素材，跳过（不产生段）。
    ///
    /// 【为什么不是 `NotPlaying`】两者都“没写”，但原因不同：后者是宿主没在播，
    /// 前者是宿主在播（或预取）却重复投递同一块。诊断必须能分开，否则
    /// “开头为什么有重复内容”无法归因。
    Repeated,
    /// 采样率与本次捕获不一致：已停止捕获。
    RateChanged,
    /// 备用缓冲用尽：已停止捕获。
    OutOfSpares,
    /// 到达用户设定的单次录制上限：已停止捕获。
    LimitReached,
}

/// 捕获侧诊断快照（`transfer_status` 与诊断包读取）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CaptureDiagnostics {
    pub preroll_frames_dropped: u64,
    pub pending_stashed: u64,
    pub pending_committed: u64,
    pub pending_discarded: u64,
    pub pending_ready: bool,
    pub pending_available: bool,
    pub gaps: u64,
    pub backwards: u64,
    pub dropouts: u64,
    pub oversized_blocks: u64,
    pub silent_blocks: u64,
    pub spared: usize,
    /// 当前为捕获供料的进程流（`None` = 尚未选举）。
    pub elected: Option<ProcessStream>,
    /// 进程流交接次数。
    pub stream_switches: u64,
    /// 同一区间被重复投递而跳过的块数。
    pub repeated_blocks: u64,
    /// 交接/重叠处被裁掉的前缀帧数。
    pub overlapped_frames: u64,
    /// 因 `Resting` 被抑制的块数 / 帧数。
    pub resting_blocks: u64,
    pub resting_frames: u64,
    /// 上升沿覆盖写：块数 / 帧数。
    pub overwritten_blocks: u64,
    pub overwritten_frames: u64,
    /// 回卷时裁掉的悬挂尾巴帧数。
    pub overhang_trimmed_frames: u64,
    /// 循环边界切分次数。
    pub cycle_splits: u64,
    /// 起播 resync 时用静音桥接的帧数。
    pub resync_bridged_frames: u64,
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
    /// 每条流自己的候选首块（一块 lookbehind）。
    ///
    /// 【为什么需要它】宿主可以在 `kPlaying` 置位之前就送来起播的第一块
    /// （ASIO Guard / anticipative FX / pre-roll）。只按 `kPlaying` 判断会丢掉这块；
    /// 一看到位置推进就写又会把“停播时 seek 的一块”录进来。先暂存一块，等下一块
    /// **首尾相接**才提交，才能同时满足两边。
    ///
    /// 【为什么每流一份】选举之前两条流都可能送来首块；共用一份会让后到的流
    /// 覆盖先到的流（见 [`Pending`]）。
    pendings: [Pending; 2],
    /// 当前为捕获供料的流。
    elected: Option<ProcessStream>,
    /// 每条流自己的候选读数（选举与交接）。
    candidates: [StreamCandidate; 3],
    /// 当前流连续供料了多少块。
    elected_run: u32,
    /// 当前流之后又有多少块来自别的流（用于判断“另一条流是否接管”）。
    elected_idle: u32,
    /// 当前流之后别的流累计供料的帧数（长块宿主上的交接证据）。
    elected_idle_frames: u32,
    /// 当前流连续给出静音块的次数（用于“真正的输入在另一条流上”的接管）。
    elected_silent_blocks: u32,
    /// 当前流连续静音的帧数。
    elected_silent_frames: u32,
    /// 全局块序号（只用于诊断）。
    block_seq: u64,
    /// 当前 pass 的循环起点；容量切段时由后续段继承。
    current_loop_start: Option<i64>,
    /// 当前流上一块**实际写入**的区间；用于识别同一区间的重复投递。
    last_written: Option<(i64, usize)>,
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
    /// 负时间（宿主 pre-roll）被夹掉的帧数。工程时间轴从 0 开始，这部分不可听。
    pub preroll_frames_dropped: u64,
    /// lookbehind 的三个结果（诊断用）。
    pub pending_stashed: u64,
    pub pending_committed: u64,
    pub pending_discarded: u64,
    /// 流交接次数。
    pub stream_switches: u64,
    /// 同一区间被重复投递而跳过的块数。
    pub repeated_blocks: u64,
    /// 交接/重叠处被裁掉的前缀帧数。
    pub overlapped_frames: u64,
    /// 因 `Resting` 被抑制的块数 / 帧数。
    pub resting_blocks: u64,
    pub resting_frames: u64,
    /// 上升沿覆盖写：块数 / 帧数。
    pub overwritten_blocks: u64,
    pub overwritten_frames: u64,
    /// 回卷时裁掉的悬挂尾巴帧数。
    pub overhang_trimmed_frames: u64,
    /// 循环边界切分次数。
    pub cycle_splits: u64,
    /// 起播 resync 时用静音桥接的帧数。
    pub resync_bridged_frames: u64,
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
            pendings: Default::default(),
            elected: None,
            candidates: Default::default(),
            elected_run: 0,
            elected_idle: 0,
            elected_idle_frames: 0,
            elected_silent_blocks: 0,
            elected_silent_frames: 0,
            block_seq: 0,
            current_loop_start: None,
            last_written: None,
            peak: [0.0; 2],
            started: false,
            max_frames: 0,
            gaps: 0,
            backwards: 0,
            dropouts: 0,
            oversized_blocks: 0,
            silent_blocks: 0,
            preroll_frames_dropped: 0,
            pending_stashed: 0,
            pending_committed: 0,
            pending_discarded: 0,
            stream_switches: 0,
            repeated_blocks: 0,
            overlapped_frames: 0,
            resting_blocks: 0,
            resting_frames: 0,
            overwritten_blocks: 0,
            overwritten_frames: 0,
            overhang_trimmed_frames: 0,
            cycle_splits: 0,
            resync_bridged_frames: 0,
        }
    }
}

impl Capture {
    /// 开始一次捕获。
    ///
    /// `capacity_frames` 是单段容量；`spares` 是调用方**预先分配好**的备用缓冲
    /// （每个都是空的、长度为容量的平面）；`pendings` 是两条流各自的 lookbehind
    /// 专用缓冲（`None` = 本机预算不足，关闭该流的 lookbehind）；`max_frames` 是
    /// 本次录制的帧数上限（`0` = 不限）。非音频线程调用。
    pub(crate) fn start(
        &mut self,
        channels: u8,
        capacity_frames: usize,
        mut spares: Vec<Segment>,
        pendings: [Option<Segment>; 2],
        max_frames: u64,
    ) {
        *self = Self {
            spares: std::mem::take(&mut spares),
            pendings: pendings.map(|buffer| Pending {
                buffer,
                ..Pending::default()
            }),
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
            let segment_start = segment.origin.max(0);
            let segment_end = segment.origin.saturating_add(segment.filled as i64);
            if segment_end <= segment_start {
                continue;
            }
            start = start.min(segment_start);
            end = end.max(segment_end);
        }
        if let Some(position) = self.last_position {
            let position = position.max(0);
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

    /// promote 之后重置捕获状态，但保留诊断计数。
    ///
    /// 【为什么保留】录音结束后正是用户最需要看"开头丢了多少 pre-roll / lookbehind
    /// 补了几块"的时刻；`*capture = default()` 会把这些读数一起清掉，诊断就永远只能
    /// 在录制**过程中**读到。
    pub(crate) fn reset_after_promote(&mut self) {
        let diagnostics = self.diagnostics();
        *self = Self {
            // 保留“这次录制最后用哪条流供料”的读数：用户按下停止后仍要能诊断
            // 起播窗口，而不是一停止就变成 `None`。
            elected: diagnostics.elected,
            gaps: diagnostics.gaps,
            backwards: diagnostics.backwards,
            dropouts: diagnostics.dropouts,
            oversized_blocks: diagnostics.oversized_blocks,
            silent_blocks: diagnostics.silent_blocks,
            preroll_frames_dropped: diagnostics.preroll_frames_dropped,
            pending_stashed: diagnostics.pending_stashed,
            pending_committed: diagnostics.pending_committed,
            pending_discarded: diagnostics.pending_discarded,
            stream_switches: diagnostics.stream_switches,
            repeated_blocks: diagnostics.repeated_blocks,
            overlapped_frames: diagnostics.overlapped_frames,
            resting_blocks: diagnostics.resting_blocks,
            resting_frames: diagnostics.resting_frames,
            overwritten_blocks: diagnostics.overwritten_blocks,
            overwritten_frames: diagnostics.overwritten_frames,
            overhang_trimmed_frames: diagnostics.overhang_trimmed_frames,
            cycle_splits: diagnostics.cycle_splits,
            resync_bridged_frames: diagnostics.resync_bridged_frames,
            ..Self::default()
        };
    }

    /// 捕获侧诊断快照（非音频线程调用）。
    pub(crate) fn diagnostics(&self) -> CaptureDiagnostics {
        CaptureDiagnostics {
            preroll_frames_dropped: self.preroll_frames_dropped,
            pending_stashed: self.pending_stashed,
            pending_committed: self.pending_committed,
            pending_discarded: self.pending_discarded,
            pending_ready: self.pendings.iter().any(|pending| pending.ready),
            pending_available: self.pendings.iter().any(|pending| pending.buffer.is_some()),
            gaps: self.gaps,
            backwards: self.backwards,
            dropouts: self.dropouts,
            oversized_blocks: self.oversized_blocks,
            silent_blocks: self.silent_blocks,
            spared: self.spares.len(),
            elected: self.elected,
            stream_switches: self.stream_switches,
            repeated_blocks: self.repeated_blocks,
            overlapped_frames: self.overlapped_frames,
            resting_blocks: self.resting_blocks,
            resting_frames: self.resting_frames,
            overwritten_blocks: self.overwritten_blocks,
            overwritten_frames: self.overwritten_frames,
            overhang_trimmed_frames: self.overhang_trimmed_frames,
            cycle_splits: self.cycle_splits,
            resync_bridged_frames: self.resync_bridged_frames,
        }
    }

    /// 当前为捕获供料的进程流（`None` = 尚未选举）。
    pub(crate) fn elected_stream(&self) -> Option<ProcessStream> {
        self.elected
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
    ///
    /// 【为什么 pending 直接丢弃】它只是“尚未被下一块证明”的候选首块；用户按下
    /// stop 时，没有证据的那一块不能算素材（否则一次误按会产生一个幽灵 take）。
    pub(crate) fn finish(&mut self, reason: ClosedReason) -> Vec<Segment> {
        for pending in &mut self.pendings {
            pending.buffer = None;
            pending.ready = false;
            pending.key = None;
            pending.stream = None;
        }
        if let Some(active) = self.active.take() {
            self.sealed.push(seal(active, reason));
        }
        if self.stopped.is_none() {
            self.stopped = Some(reason);
        }
        self.drain_sealed()
    }

    /// 音频线程入口（兼容既有调用：没有块流证据时等价于“只在 `kPlaying` 时写”）。
    pub(crate) fn on_block(&mut self, block: InputBlock<'_>) -> CaptureOutcome {
        self.on_block_with_flow(block, false)
    }

    /// 音频线程入口，带“本块相对上一块是否在推进”的弱证据。
    ///
    /// 【为什么 `flow` 不能直接等于“写”】位置推进既可能是起播，也可能是停播时
    /// 拖了一次播放头。弱证据先暂存一块，下一块**首尾相接**才提交（lookbehind）。
    pub(crate) fn on_block_with_flow(
        &mut self,
        block: InputBlock<'_>,
        flow: bool,
    ) -> CaptureOutcome {
        self.on_block_with_context(block, BlockContext::realtime(flow))
    }

    /// 音频线程入口：带进程流与块流证据。
    ///
    /// 【职责划分】本函数只做三件事：夹掉 pre-roll、更新**本流**候选、决定是否
    /// 选举/交接；真正的 lookbehind 与写入是 [`Self::capture_elected`]。这样
    /// “哪条流供料”与“这一块怎么写”不会在同一个 if 树里互相污染。
    pub(crate) fn on_block_with_context(
        &mut self,
        block: InputBlock<'_>,
        ctx: BlockContext,
    ) -> CaptureOutcome {
        if self.stopped.is_some() {
            return CaptureOutcome::Captured;
        }
        self.block_seq = self.block_seq.wrapping_add(1);
        // 【为什么零帧块要单独处理】旧的语义是“在播放的零帧块 ⇒ 起播”，
        // 它不产生素材但会把状态从待命推进到录制中。负时间裁剪不能把这个语义吃掉。
        if block.frames == 0 {
            if block.playing {
                self.started = true;
                return CaptureOutcome::Captured;
            }
            return CaptureOutcome::NotPlaying;
        }
        // 【为什么非法位置要丢弃而不是当 0】把一块音频写到工程位置 0 会**污染**时间轴：
        // 用户听到的那一小段会出现在开头。丢弃只损失这一块，且被计数。
        let Some(rate) = normalize_rate(block.sample_rate) else {
            self.dropouts = self.dropouts.saturating_add(1);
            return CaptureOutcome::NoPosition;
        };
        // 【为什么负时间必须在这里夹掉】`projectTimeSamples < 0` 是宿主 pre-roll
        // （Cubase 的 ASIO Guard / PDC 会给出负位置）。工程时间轴从 0 开始，
        // 负的部分不可听；若原样存进段，投影会生成负起点 clip，而渲染侧把窗口
        // 起点夹到 0 时**不会**同步平移源窗口 —— 整段内容会前移一个 pre-roll。
        // 夹掉是对齐的唯一正确做法：源偏移跟着跳过同样的帧数。
        let skip = block.start.saturating_neg().min(block.frames as i64).max(0) as usize;
        let start = block.start.saturating_add(skip as i64);
        let frames = block.frames - skip;
        if skip > 0 {
            self.preroll_frames_dropped = self.preroll_frames_dropped.saturating_add(skip as u64);
        }
        if frames == 0 {
            // 整块都在工程 0 之前：没有可听内容，也不把状态推进到“录制中”。
            return CaptureOutcome::NotPlaying;
        }

        let audible = block_has_signal(&block, skip, frames);
        let ctx = BlockContext {
            playing: block.playing,
            ..ctx
        };
        // 循环 pass 的元数据：新一圈开始时记下 cycle start；同一 pass 的容量切段
        // 继续继承它。没有 cycle 元数据时保持 None（投影会按相同起点隐式分组）。
        if ctx.part > 0 {
            if ctx.cycle_start.is_some() {
                self.current_loop_start = ctx.cycle_start;
            }
        } else if ctx.new_pass || self.active.is_none() || ctx.cycle_wrap {
            self.current_loop_start = ctx.cycle_start;
        }
        // 【Resting 必须在候选/选举/写入之前拦下】同一块首的重复既不是新素材，
        // 也不能替另一条流作证。上升沿例外：真正的起播首块可能与停播块同几何。
        if ctx.phase == TransportPhase::Resting && !ctx.playing_rising && !ctx.recording_rising {
            self.resting_blocks = self.resting_blocks.saturating_add(1);
            self.resting_frames = self.resting_frames.saturating_add(frames as u64);
            return CaptureOutcome::NotPlaying;
        }
        if ctx.part > 0 {
            // 循环切分的第二段复用第一段选出的流；不重复更新候选/交接计数。
            if self.elected == Some(ctx.stream) {
                return self.capture_elected(
                    ctx.stream,
                    &block,
                    start,
                    frames,
                    skip,
                    rate,
                    true,
                    ctx.playing_rising || ctx.recording_rising,
                    ctx.part,
                    ctx.advancing,
                    false,
                    0,
                );
            }
            return CaptureOutcome::NotPlaying;
        }
        self.note_candidate(ctx, start, frames, audible);

        match self.elected {
            Some(elected) if elected == ctx.stream => {
                self.elected_idle = 0;
                self.elected_idle_frames = 0;
                self.elected_run = self.elected_run.saturating_add(1);
                if audible {
                    self.elected_silent_blocks = 0;
                    self.elected_silent_frames = 0;
                } else {
                    self.elected_silent_blocks = self.elected_silent_blocks.saturating_add(1);
                    self.elected_silent_frames = self
                        .elected_silent_frames
                        .saturating_add(u32::try_from(frames).unwrap_or(u32::MAX));
                }
            }
            Some(_) => {
                self.elected_idle = self.elected_idle.saturating_add(1);
                self.elected_idle_frames = self
                    .elected_idle_frames
                    .saturating_add(u32::try_from(frames).unwrap_or(u32::MAX));
                if !self.should_handover(ctx, rate) {
                    // 非供料流的块只更新候选；它自己的 pending 留给之后的交接。
                    self.stash_if_needed(ctx.stream, &block, start, frames, skip, rate);
                    return CaptureOutcome::NotPlaying;
                }
                self.switch_to(ctx.stream, start, frames, rate);
            }
            None => {
                if !self.try_elect(ctx.stream) {
                    // 选举之前也要暂存：首块的 lookbehind 是下一块能提交它的前提。
                    self.stash_if_needed(ctx.stream, &block, start, frames, skip, rate);
                    return CaptureOutcome::NotPlaying;
                }
            }
        }

        self.capture_elected(
            ctx.stream,
            &block,
            start,
            frames,
            skip,
            rate,
            ctx.phase == TransportPhase::Rolling,
            ctx.playing_rising || ctx.recording_rising,
            ctx.part,
            ctx.advancing,
            ctx.resync,
            ctx.resync_limit_frames,
        )
    }

    /// 更新**本流**的候选读数；绝不与另一条流比较。
    fn note_candidate(&mut self, ctx: BlockContext, start: i64, frames: usize, audible: bool) {
        let frames32 = u32::try_from(frames).unwrap_or(u32::MAX);
        let candidate = &mut self.candidates[ctx.stream.index()];
        if candidate.seen && candidate.last_end == start {
            candidate.contiguous = candidate.contiguous.saturating_add(1);
            candidate.contiguous_frames = candidate.contiguous_frames.saturating_add(frames32);
        } else {
            candidate.contiguous = 1;
            candidate.contiguous_frames = frames32;
        }
        candidate.seen = true;
        candidate.last_start = start;
        candidate.last_frames = frames;
        candidate.last_end = start.saturating_add(frames as i64);
        if ctx.playing {
            candidate.playing_blocks = candidate.playing_blocks.saturating_add(1);
            candidate.playing_frames = candidate.playing_frames.saturating_add(frames32);
        }
        if ctx.recording {
            candidate.recording_blocks = candidate.recording_blocks.saturating_add(1);
            candidate.recording_frames = candidate.recording_frames.saturating_add(frames32);
        }
        if ctx.phase != TransportPhase::Resting {
            candidate.moving_blocks = candidate.moving_blocks.saturating_add(1);
            candidate.moving_frames = candidate.moving_frames.saturating_add(frames32);
        }
        if audible {
            candidate.non_silent_blocks = candidate.non_silent_blocks.saturating_add(1);
            candidate.non_silent_frames = candidate.non_silent_frames.saturating_add(frames32);
        }
        candidate.last_seen_seq = self.block_seq;
    }

    /// 首次选举。返回是否成功选出一条流。
    ///
    /// 【证据顺序】离线（显式）> 录音中 > 播放位 > 位置真的在推进 >
    /// 非静音。非静音不能单独当选：用户真实的静音演奏也必须能被录下来。
    fn try_elect(&mut self, stream: ProcessStream) -> bool {
        if stream == ProcessStream::Offline {
            self.elect(stream);
            return true;
        }
        let candidate = self.candidates[stream.index()];
        let viable = candidate.playing_blocks > 0
            || candidate.recording_blocks > 0
            || candidate.contiguous >= 2;
        if !viable {
            return false;
        }
        let mine = candidate_evidence(&candidate);
        let better = self.candidates.iter().enumerate().any(|(index, other)| {
            index != stream.index()
                && other.seen
                && other.contiguous >= 2
                && candidate_evidence(other) > mine
        });
        if better {
            return false;
        }
        self.elect(stream);
        true
    }

    fn elect(&mut self, stream: ProcessStream) {
        self.elected = Some(stream);
        self.elected_run = 1;
        self.elected_idle = 0;
        self.elected_idle_frames = 0;
        self.elected_silent_blocks = 0;
        self.elected_silent_frames = 0;
    }

    /// 另一条流是否已经提供了“可以接管”的持续证据。
    fn should_handover(&self, ctx: BlockContext, rate: u32) -> bool {
        if ctx.stream == ProcessStream::Offline {
            return true;
        }
        let candidate = self.candidates[ctx.stream.index()];
        let frames_needed = ((rate as f64) * HANDOVER_SEC)
            .round()
            .clamp(1.0, u32::MAX as f64) as u32;
        // `kRecording` 是强证据：录音轨上的真实输入在实时路径上，晚接管会丢掉正在录的
        // 第一块。其余证据要求“块数下限 + 帧数阈值”，长块宿主不会退化成按块数等数秒。
        ctx.recording
            || (ctx.playing
                && candidate.contiguous >= HANDOVER_BLOCKS
                && candidate.contiguous_frames >= frames_needed)
            || (self.elected_idle >= HANDOVER_BLOCKS
                && self.elected_idle_frames >= frames_needed
                && candidate.contiguous >= HANDOVER_BLOCKS
                && candidate.contiguous_frames >= frames_needed)
            || (self.elected_silent_blocks >= HANDOVER_BLOCKS
                && self.elected_silent_frames >= frames_needed
                && candidate.non_silent_blocks >= HANDOVER_BLOCKS
                && candidate.non_silent_frames >= frames_needed
                && candidate.contiguous >= HANDOVER_BLOCKS)
    }

    /// 把供料流切换到 `next`。
    ///
    /// 【交接时的四条纪律】1) 新流接着当前时间线时**不封口**（同一遍播放不能因为
    /// 流切换被切成两段）；2) 只有真正的跳跃才封口；3) 新流在选举前暂存的首块，
    /// 若与当前块连续/半重叠就提交；4) 其他流的候选一律丢弃。
    fn switch_to(&mut self, next: ProcessStream, start: i64, frames: usize, rate: u32) {
        let continues = match (
            self.active.as_ref().map(|active| active.origin),
            self.expected_next,
        ) {
            (Some(origin), Some(end)) => start >= origin && start <= end,
            _ => false,
        };
        for stream in [
            ProcessStream::Realtime,
            ProcessStream::Prefetch,
            ProcessStream::Offline,
        ] {
            if stream != next {
                self.discard_pending(stream);
            }
        }
        if continues {
            // 新流覆盖的是 active 已经写过的区间（或正好接在终点）：同一遍播放。
            // pending 只会与 active 重叠，丢弃它比把重复内容再写一次更安全。
            self.discard_pending(next);
            self.elected = Some(next);
            self.elected_run = 1;
            self.elected_idle = 0;
            self.elected_idle_frames = 0;
            self.elected_silent_blocks = 0;
            self.elected_silent_frames = 0;
            self.stream_switches = self.stream_switches.saturating_add(1);
            return;
        }
        let slot = next.pending_slot();
        let commit = self.pendings[slot].ready
            && self.pendings[slot].stream == Some(next)
            && self.pendings[slot].buffer.as_ref().is_some_and(|pending| {
                let end = pending.origin.saturating_add(pending.filled as i64);
                pending.sample_rate == rate
                    && (self.sample_rate == 0 || self.sample_rate == pending.sample_rate)
                    && (end == start || (start < end && start + frames as i64 > pending.origin))
            });
        if let Some(active) = self.active.take() {
            self.sealed.push(seal(active, ClosedReason::Jump));
        }
        self.expected_next = None;
        self.last_written = None;
        if commit {
            let _ = self.commit_pending(next);
        } else {
            self.discard_pending(next);
        }
        self.elected = Some(next);
        self.elected_run = 1;
        self.elected_idle = 0;
        self.elected_idle_frames = 0;
        self.elected_silent_blocks = 0;
        self.elected_silent_frames = 0;
        self.stream_switches = self.stream_switches.saturating_add(1);
    }

    /// 选举之前暂存候选首块；同一流已有连续候选时保留最早那块。
    fn stash_if_needed(
        &mut self,
        stream: ProcessStream,
        block: &InputBlock<'_>,
        start: i64,
        frames: usize,
        skip: usize,
        rate: u32,
    ) {
        if self.started {
            return;
        }
        if self.pending_ready_for(stream) {
            if self.pending_is_contiguous(stream, start, rate) {
                // 首块已经证明是这段连续播放的起点；不要用后续块覆盖它。
                return;
            }
            self.discard_pending(stream);
        }
        self.stash_pending(stream, block, start, frames, skip, rate);
    }

    /// 被选举流上的 lookbehind + 写入判定。
    fn capture_elected(
        &mut self,
        stream: ProcessStream,
        block: &InputBlock<'_>,
        start: i64,
        frames: usize,
        skip: usize,
        rate: u32,
        flow: bool,
        rising: bool,
        part: u8,
        advancing: bool,
        resync: bool,
        resync_limit_frames: u64,
    ) -> CaptureOutcome {
        // 强证据：宿主自己说在播放。直接开始（若 pending 恰好连续，先把那一块补上）。
        if block.playing {
            if self.pending_ready_for(stream) {
                if self.pending_is_contiguous(stream, start, rate) {
                    let _ = self.commit_pending(stream);
                } else {
                    self.discard_pending(stream);
                }
            }
            self.started = true;
            return self.write_block(
                block,
                start,
                frames,
                skip,
                rate,
                rising,
                part,
                advancing,
                resync,
                resync_limit_frames,
            );
        }

        // 弱证据：位置流在推进。第一块只暂存，第二块连续才提交。
        if flow {
            if self.pending_ready_for(stream) {
                if self.pending_is_contiguous(stream, start, rate) {
                    if self.commit_pending(stream) {
                        return self.write_block(
                            block,
                            start,
                            frames,
                            skip,
                            rate,
                            rising,
                            part,
                            advancing,
                            resync,
                            resync_limit_frames,
                        );
                    }
                } else {
                    self.discard_pending(stream);
                }
            }
            if !self.started {
                self.stash_pending(stream, block, start, frames, skip, rate);
                return CaptureOutcome::NotPlaying;
            }
            // 已经开录之后，只有“接着上一块”的滞后续录才写；停播时的 seek 块不写。
            // `expected_next == None` 是流交接后的第一块：没有可比的终点，直接写。
            if self.expected_next.is_none() || self.expected_next == Some(start) {
                return self.write_block(
                    block,
                    start,
                    frames,
                    skip,
                    rate,
                    rising,
                    part,
                    advancing,
                    resync,
                    resync_limit_frames,
                );
            }
            return CaptureOutcome::NotPlaying;
        }

        // 没有播放位、也没有位置推进：可能是钉住的停播块，也可能是“同头同长”的
        // 起播第一块。先暂存；下一块连续时由上面的分支提交。
        if self.started && self.expected_next.is_none() {
            // 刚交接：新流的第一块没有可比终点，即使没有 flow 也要写。
            return self.write_block(
                block,
                start,
                frames,
                skip,
                rate,
                rising,
                part,
                advancing,
                resync,
                resync_limit_frames,
            );
        }
        if !self.started {
            self.stash_pending(stream, block, start, frames, skip, rate);
        }
        CaptureOutcome::NotPlaying
    }

    fn pending_slot_ready(&self, slot: usize, stream: ProcessStream) -> bool {
        let pending = &self.pendings[slot];
        pending.ready && pending.stream == Some(stream)
    }

    fn pending_ready_for(&self, stream: ProcessStream) -> bool {
        self.pending_slot_ready(stream.pending_slot(), stream)
    }

    fn pending_is_contiguous(&self, stream: ProcessStream, start: i64, rate: u32) -> bool {
        let pending = &self.pendings[stream.pending_slot()];
        pending.ready
            && pending.stream == Some(stream)
            && pending.buffer.as_ref().is_some_and(|pending| {
                pending.sample_rate == rate
                    && pending.origin.saturating_add(pending.filled as i64) == start
            })
    }

    fn take_pending(&mut self, stream: ProcessStream) -> Option<Segment> {
        let pending = &mut self.pendings[stream.pending_slot()];
        if !pending.ready || pending.stream != Some(stream) {
            return None;
        }
        pending.ready = false;
        pending.key = None;
        pending.stream = None;
        pending.buffer.take()
    }

    /// 把一块写成某条流的 pending（只拷贝，不改变“是否已开始”的状态）。
    fn stash_pending(
        &mut self,
        stream: ProcessStream,
        block: &InputBlock<'_>,
        start: i64,
        frames: usize,
        skip: usize,
        rate: u32,
    ) {
        let slot = stream.pending_slot();
        // 槽位可能被另一条流占着（离线与实时共用 0 号）：先清掉，避免把它的内容
        // 冒充成当前流的候选。
        if self.pendings[slot].ready && self.pendings[slot].stream != Some(stream) {
            self.pendings[slot].ready = false;
            self.pendings[slot].key = None;
            self.pendings[slot].stream = None;
            self.pending_discarded = self.pending_discarded.saturating_add(1);
        }
        let capacity = self.capacity;
        let Some(pending) = self.pendings[slot].buffer.as_mut() else {
            return;
        };
        let writable = frames
            .min(capacity)
            .min(pending.planes.first().map_or(0, Vec::len));
        if writable == 0 {
            return;
        }
        let key = (start, writable as u32, rate);
        if self.pendings[slot].key == Some(key) && pending.filled == writable {
            // 宿主停在一个光标处反复送同一块：内容与长度都没变，不必每块重拷。
            return;
        }
        pending.origin = start;
        pending.sample_rate = rate;
        pending.filled = writable;
        pending.loss = SegmentLoss::First;
        pending.closed = ClosedReason::Full;
        pending.peak = [0.0; 2];
        for (channel, plane) in pending.planes.iter_mut().enumerate() {
            let target = &mut plane[..writable];
            match block.planes.get(channel).copied().flatten() {
                Some(source) if source.len() >= skip + writable => {
                    target.copy_from_slice(&source[skip..skip + writable]);
                }
                _ => target.fill(0.0),
            }
            let peak = target.iter().fold(0.0_f32, |acc, value| {
                if value.is_finite() {
                    acc.max(value.abs())
                } else {
                    acc
                }
            });
            pending.peak[channel] = peak;
        }
        self.pendings[slot].key = Some(key);
        self.pendings[slot].ready = true;
        self.pendings[slot].stream = Some(stream);
        self.pending_stashed = self.pending_stashed.saturating_add(1);
    }

    /// 把某条流的 pending 提升为 active；下一块的连续性从它的终点开始。
    ///
    /// 返回是否提交。采样率与既有捕获不一致时丢弃（本 take 的样本域不再一致，
    /// 交给 `write_block` 的 `RateChanged` 处理，而不是把两份不同率的样本拼进一段）。
    fn commit_pending(&mut self, stream: ProcessStream) -> bool {
        let Some(mut pending) = self.take_pending(stream) else {
            return false;
        };
        if self.sample_rate != 0 && self.sample_rate != pending.sample_rate {
            self.pending_discarded = self.pending_discarded.saturating_add(1);
            return false;
        }
        pending.loop_start = self.current_loop_start;
        self.sample_rate = pending.sample_rate;
        self.channels = pending.planes.len().clamp(1, 2) as u8;
        self.expected_next = Some(pending.origin.saturating_add(pending.filled as i64));
        self.captured_frames = self.captured_frames.saturating_add(pending.filled as u64);
        self.last_position = self.expected_next;
        self.peak[0] = self.peak[0].max(pending.peak[0]);
        self.peak[1] = self.peak[1].max(pending.peak[1]);
        self.last_written = Some((pending.origin, pending.filled));
        self.active = Some(pending);
        self.started = true;
        self.pending_committed = self.pending_committed.saturating_add(1);
        true
    }

    /// 丢弃某条流的 pending（非音频线程不参与；只由音频线程在状态切换时调用）。
    fn discard_pending(&mut self, stream: ProcessStream) {
        let pending = &mut self.pendings[stream.pending_slot()];
        if pending.ready && pending.stream == Some(stream) {
            pending.ready = false;
            pending.key = None;
            pending.stream = None;
            self.pending_discarded = self.pending_discarded.saturating_add(1);
        }
    }

    /// 用本块的样本覆盖 active 段内一段**已经写过**的区间。
    ///
    /// 【为什么需要它】停播/预卷时宿主可能先送一块同几何的块（被写成素材），真正
    /// 起播时再送同一几何、不同内容的首块。几何判重会把真正的首块当成重复丢掉；
    /// 上升沿时改为覆盖写，才是“这一遍真的开始了”。
    ///
    /// 覆盖不增加 `captured_frames`，也不改变 `expected_next`。目标必须完全落在
    /// active 的已写范围内；落在已封口段时返回 false（由调用方退化为 Repeated）。
    fn overwrite_range(
        &mut self,
        start: i64,
        frames: usize,
        block: &InputBlock<'_>,
        skip: usize,
    ) -> bool {
        if frames == 0 {
            return false;
        }
        let Some(active) = self.active.as_mut() else {
            return false;
        };
        let origin = active.origin;
        if start < origin {
            return false;
        }
        let Ok(offset) = usize::try_from(start - origin) else {
            return false;
        };
        let Some(end) = offset.checked_add(frames) else {
            return false;
        };
        if end > active.filled {
            return false;
        }
        for (channel, plane) in active.planes.iter_mut().enumerate() {
            let target = &mut plane[offset..end];
            match block.planes.get(channel).copied().flatten() {
                Some(source) if source.len() >= skip.saturating_add(frames) => {
                    target.copy_from_slice(&source[skip..skip + frames]);
                }
                _ => target.fill(0.0),
            }
        }
        // 峰值只服务 UI；覆盖后保守地重算整段峰值，避免显示一个已经不存在的峰值。
        for (channel, plane) in active.planes.iter().enumerate() {
            let peak = plane[..active.filled].iter().fold(0.0_f32, |acc, value| {
                if value.is_finite() {
                    acc.max(value.abs())
                } else {
                    acc
                }
            });
            active.peak[channel.min(1)] = peak;
        }
        let peak = active.peak;
        self.peak[0] = self.peak[0].max(peak[0]);
        self.peak[1] = self.peak[1].max(peak[1]);
        true
    }

    /// 回卷/向后跳跃时裁掉 active 末尾的“悬挂尾巴”。
    ///
    /// 【裁到哪里】只从**最后一次写入的块首**开始裁：更早的块一定属于上一遍的有效
    /// 内容，不能因为新一遍从更早的位置开始就把它们删掉。代价是至多损失一个块在
    /// 回卷点之前的有效音频；换来的是不会把错位的下一圈开头留在 loop 终点之后。
    fn trim_active_tail_to(&mut self, new_start: i64) -> bool {
        let last_written = self.last_written;
        let Some(active) = self.active.as_ref() else {
            return false;
        };
        let end = active.origin.saturating_add(active.filled as i64);
        if new_start >= end {
            return false;
        }
        let from = match last_written {
            Some((last_start, _)) => new_start.max(last_start),
            None => new_start,
        };
        if from <= active.origin || from >= end {
            return false;
        }
        let keep = (from - active.origin) as usize;
        let dropped = active.filled - keep;
        if let Some(active) = self.active.as_mut() {
            active.filled = keep;
        }
        self.captured_frames = self.captured_frames.saturating_sub(dropped as u64);
        self.expected_next = Some(from);
        self.last_position = Some(from);
        self.last_written = None;
        self.overhang_trimmed_frames = self.overhang_trimmed_frames.saturating_add(dropped as u64);
        if keep == 0 {
            self.active = None;
        }
        true
    }

    /// 把 active 裁到**精确**的工程位置（resync 向后纠正）。
    ///
    /// 【为什么不能用 `trim_active_tail_to`】那个函数只从"最后一次写入的块首"开始裁，
    /// 目的是保住循环回卷前的有效内容；resync 向后纠正时，多记的整段都是我们按块长
    /// 推出来的，宿主已经明确告诉我们真实位置在更前面 —— 必须精确裁到 `new_start`。
    fn trim_active_tail_exact(&mut self, new_start: i64) -> bool {
        let Some(active) = self.active.as_mut() else {
            return false;
        };
        let end = active.origin.saturating_add(active.filled as i64);
        if new_start <= active.origin || new_start >= end {
            return false;
        }
        let keep = (new_start - active.origin) as usize;
        let dropped = active.filled - keep;
        active.filled = keep;
        self.captured_frames = self.captured_frames.saturating_sub(dropped as u64);
        self.expected_next = Some(new_start);
        self.last_position = Some(new_start);
        self.last_written = None;
        self.overhang_trimmed_frames = self.overhang_trimmed_frames.saturating_add(dropped as u64);
        true
    }

    /// 用静音桥接 `frames` 帧（resync 向前纠正）。
    ///
    /// 返回是否全部写入；备用缓冲不够时返回 false，调用方退回"新开段"的旧行为。
    /// 桥接帧计入 `captured_frames`（它是 take 的一部分），也计入
    /// `resync_bridged_frames`（诊断）。
    fn bridge_silence(&mut self, frames: usize) -> bool {
        let mut remaining = frames;
        while remaining > 0 {
            if self
                .active
                .as_ref()
                .is_some_and(|active| active.filled >= self.capacity)
            {
                if let Some(active) = self.active.take() {
                    self.sealed.push(seal(active, ClosedReason::Full));
                }
            }
            if self.active.is_none() {
                let Some(mut spare) = self.spares.pop() else {
                    return false;
                };
                spare.origin = self.expected_next.unwrap_or(0);
                spare.sample_rate = self.sample_rate;
                spare.loss = SegmentLoss::None;
                spare.loop_start = self.current_loop_start;
                spare.filled = 0;
                spare.peak = [0.0; 2];
                spare.closed = ClosedReason::Full;
                self.active = Some(spare);
            }
            let Some(active) = self.active.as_mut() else {
                return false;
            };
            let writable = remaining.min(self.capacity.saturating_sub(active.filled));
            if writable == 0 {
                return false;
            }
            let start = active.origin.saturating_add(active.filled as i64);
            for plane in active.planes.iter_mut() {
                plane[active.filled..active.filled + writable].fill(0.0);
            }
            active.filled += writable;
            let next = active.origin.saturating_add(active.filled as i64);
            self.expected_next = Some(next);
            self.last_position = Some(next);
            self.last_written = Some((start, writable));
            self.captured_frames = self.captured_frames.saturating_add(writable as u64);
            self.resync_bridged_frames = self
                .resync_bridged_frames
                .saturating_add(writable as u64);
            remaining -= writable;
        }
        true
    }

    /// 对 active 末尾 `frames` 帧做线性淡出（桥接前的去咔）。
    fn fade_active_tail_out(&mut self, frames: usize) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        let end = active.filled;
        let fade = frames.min(end);
        if fade == 0 {
            return;
        }
        let start = end - fade;
        for plane in active.planes.iter_mut() {
            for (index, sample) in plane[start..end].iter_mut().enumerate() {
                let gain = 1.0 - (index + 1) as f32 / fade as f32;
                *sample *= gain;
            }
        }
    }

    /// 真正写入一块（裁剪后的 `start`/`frames`，源偏移 `skip`）。
    ///
    /// `rising` = 本块是 `kPlaying`/`kRecording` 的上升沿：允许用新内容覆盖停播时
    /// 写过的同几何块。`part` = 循环切分段号；第 1 段是**有意**的向后换位，不参与
    /// “重复/裁剪”判定。
    #[allow(clippy::too_many_arguments)]
    fn write_block(
        &mut self,
        block: &InputBlock<'_>,
        start: i64,
        frames: usize,
        skip: usize,
        rate: u32,
        rising: bool,
        part: u8,
        advancing: bool,
        resync: bool,
        resync_limit_frames: u64,
    ) -> CaptureOutcome {
        if self.sample_rate == 0 {
            self.sample_rate = rate;
            self.channels = channels_of(block);
        } else if self.sample_rate != rate {
            // 工程率变了：本 take 里已录样本的"秒"与后续块的"秒"不再可比。
            self.stop(ClosedReason::RateChanged);
            return CaptureOutcome::RateChanged;
        }

        // 【为什么先裁半重叠】两条流在交接处经常给出首尾重叠的块（prefetch 已铺到
        // `P+N`，实时流从 `P+N-M` 开始）。把整个块当 `Backward` 另起一段会让同一段
        // 内容被写两次；裁掉已经录过的前缀、只写新增部分，才是“同一遍的延续”。
        //
        // 完全落在已录区间内的块：宿主重复投递同一区间（跳过），或真正的循环回卷/
        // seek（另起一段、后来者胜出）。循环切分的第 1 段是有意换位，跳过这一整段
        // 判定，直接按 Backward 开新段。
        let raw = (start, frames);
        let mut start = start;
        let mut frames = frames;
        let mut skip = skip;
        if part == 1 {
            self.cycle_splits = self.cycle_splits.saturating_add(1);
        }
        // 【resync 纠正：这不是 seek】起播窗口里我们自己推进的位置与宿主恢复后的位置
        // 可能差一小段。向前用静音桥接（缺的正是真实播放的静音），向后裁掉多记的
        // 尾巴；两种都继续同一段，不能因为一个记账误差把 take 切成两个 clip。
        let mut resync_continued = false;
        if part == 0 && resync {
            if let Some(expected) = self.expected_next {
                let delta = start.saturating_sub(expected);
                let limit = resync_limit_frames.max(64);
                if delta > 0
                    && delta as u64 <= limit
                    && (delta as u64) <= u64::from(rate).saturating_mul(MAX_BRIDGE_SEC)
                {
                    self.fade_active_tail_out(DECLICK_FRAMES);
                    resync_continued = self.bridge_silence(delta as usize);
                } else if delta < 0 && delta.unsigned_abs() <= limit {
                    resync_continued = self.trim_active_tail_exact(start);
                }
            }
        }
        if part == 0 {
            if let Some(expected) = self.expected_next {
                if start < expected {
                    let covered = usize::try_from(expected - start).unwrap_or(usize::MAX);
                    if covered < frames {
                        // 上升沿：先覆盖已写前缀（若仍在 active 内），再写新增尾部。
                        if rising {
                            let _ = self.overwrite_range(start, covered, block, skip);
                        }
                        skip = skip.saturating_add(covered);
                        frames -= covered;
                        start = expected;
                        self.overlapped_frames =
                            self.overlapped_frames.saturating_add(covered as u64);
                    } else if self.last_written == Some(raw) {
                        if rising && self.overwrite_range(start, frames, block, skip) {
                            self.overwritten_blocks = self.overwritten_blocks.saturating_add(1);
                            self.overwritten_frames =
                                self.overwritten_frames.saturating_add(frames as u64);
                            return CaptureOutcome::Captured;
                        }
                        self.repeated_blocks = self.repeated_blocks.saturating_add(1);
                        return CaptureOutcome::Repeated;
                    } else if rising {
                        if self.overwrite_range(start, frames, block, skip) {
                            self.overwritten_blocks = self.overwritten_blocks.saturating_add(1);
                            self.overwritten_frames =
                                self.overwritten_frames.saturating_add(frames as u64);
                            return CaptureOutcome::Captured;
                        }
                        self.repeated_blocks = self.repeated_blocks.saturating_add(1);
                        return CaptureOutcome::Repeated;
                    } else if !advancing
                        && self
                            .active
                            .as_ref()
                            .is_some_and(|active| start > active.origin)
                    {
                        // 落在 active 内部（不是段首）的完全重复：不是新素材。
                        // 段首的向后跳跃可能是循环回卷/seek，必须继续走到 Backward 分支。
                        self.repeated_blocks = self.repeated_blocks.saturating_add(1);
                        return CaptureOutcome::Repeated;
                    }
                }
            }
        }
        if frames == 0 {
            self.repeated_blocks = self.repeated_blocks.saturating_add(1);
            return CaptureOutcome::Repeated;
        }

        // 循环回卷的兜底：没有 cycle 元数据时，把上一段末尾“越过新起点”的尾巴裁掉。
        // 只从**最后一次写入的块首**开始裁：更早的块一定属于上一遍的有效内容。
        let trimmed = if part == 0 && !resync_continued {
            self.trim_active_tail_to(start)
        } else {
            false
        };

        // 连续性：上一块的终点是否正好接上这一块的首点。
        let loss = if resync_continued {
            // 桥接/精确裁剪已经把 `expected_next` 对齐到 `start`；这是同一段的延续。
            SegmentLoss::None
        } else if trimmed {
            self.backwards = self.backwards.saturating_add(1);
            SegmentLoss::Backward
        } else {
            match self.expected_next {
                None => SegmentLoss::First,
                Some(expected) if expected == start => SegmentLoss::None,
                Some(expected) if start > expected => {
                    self.gaps = self.gaps.saturating_add(1);
                    SegmentLoss::Gap {
                        skipped_frames: (start - expected) as u64,
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
            }
        };
        // 正向跳跃（seek 到别处）退出当前循环 pass：后续段不能继承 loop_start，
        // 否则投影会把一段非循环内容误并进循环组。
        //
        // 【为什么小跳跃不算】起播窗口里宿主把位置钉住、恢复前进时可能留下一个
        // 块级的小缺口（本插件按块长记账的终点与宿主真实位置之间的取整差）。那不是
        // "跳到了别处"，只是记账误差；按它清掉循环身份会把同一圈拆成两组。
        let small_gap = match loss {
            SegmentLoss::Gap { skipped_frames } => {
                skipped_frames <= (frames as u64).saturating_mul(4).max(64)
            }
            _ => false,
        };
        if matches!(loss, SegmentLoss::Gap { .. }) && !small_gap {
            self.current_loop_start = None;
        }

        // 段满或跳跃 ⇒ 换段。跳跃必须换段，否则一个段里会含两段不相邻的工程时间。
        let needs_new = match &self.active {
            None => true,
            Some(active) => {
                active.filled + frames > self.capacity || !matches!(loss, SegmentLoss::None)
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
            spare.origin = start;
            spare.sample_rate = rate;
            spare.loss = loss;
            spare.loop_start = self.current_loop_start;
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
        let writable = frames.min(self.capacity.saturating_sub(active.filled));
        if writable < frames {
            self.oversized_blocks = self.oversized_blocks.saturating_add(1);
        }
        if writable == 0 {
            return CaptureOutcome::Captured;
        }
        let silent = !block.has_audio();
        if silent {
            self.silent_blocks = self.silent_blocks.saturating_add(1);
        }
        let written_offset = active.filled;
        for (channel, plane) in active.planes.iter_mut().enumerate() {
            let target = &mut plane[active.filled..active.filled + writable];
            match block.planes.get(channel).copied().flatten() {
                Some(source) if source.len() >= skip + writable => {
                    target.copy_from_slice(&source[skip..skip + writable]);
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
        self.expected_next = Some(start + writable as i64);
        self.last_written = Some(raw);
        self.captured_frames = self.captured_frames.saturating_add(writable as u64);
        self.last_position = self.expected_next;
        self.peak[0] = self.peak[0].max(active.peak[0]);
        self.peak[1] = self.peak[1].max(active.peak[1]);
        if resync_continued {
            // 桥接/裁剪后的第一块开头做淡入；与桥接前的淡出配对，接缝不再有阶跃。
            let fade = DECLICK_FRAMES.min(writable);
            if fade > 0 {
                for plane in active.planes.iter_mut() {
                    for (index, sample) in plane[written_offset..written_offset + fade]
                        .iter_mut()
                        .enumerate()
                    {
                        *sample *= (index + 1) as f32 / fade as f32;
                    }
                }
            }
        }

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
/// 本块是否有任何非零样本（选举的辅助证据）。
///
/// 【为什么可以扫全块】它只在音频线程上做一次顺序读，和随后的拷贝同一量级；
/// 而“哪条流有真正的输入”是判断 ASIO Guard 静音流不能覆盖 prefetch 内容的关键。
/// 任意声道有非零即算有信号；NaN/Inf 也按“有内容”处理（它们会走后续的丢弃规则）。
fn block_has_signal(block: &InputBlock<'_>, skip: usize, frames: usize) -> bool {
    block.planes.iter().flatten().any(|plane| {
        let end = skip.saturating_add(frames).min(plane.len());
        plane
            .get(skip..end)
            .is_some_and(|samples| samples.iter().any(|sample| *sample != 0.0))
    })
}

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
            loop_start: None,
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
            [Some(spare(capacity, 1)), None],
            0,
        );
        capture
    }

    fn ctx(stream: ProcessStream, playing: bool, recording: bool, flow: bool) -> BlockContext {
        BlockContext {
            stream,
            playing,
            recording,
            phase: if flow {
                TransportPhase::Rolling
            } else {
                TransportPhase::Start
            },
            playing_rising: false,
            recording_rising: false,
            cycle_wrap: false,
            cycle_start: None,
            new_pass: false,
            part: 0,
            advancing: flow,
            resync: false,
            resync_limit_frames: 0,
        }
    }

    fn ctx_phase(
        stream: ProcessStream,
        playing: bool,
        phase: TransportPhase,
        playing_rising: bool,
    ) -> BlockContext {
        BlockContext {
            phase,
            playing_rising,
            ..ctx(stream, playing, false, false)
        }
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
        assert_eq!(sealed[1].loss, SegmentLoss::Gap { skipped_frames: 96 });
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
        capture.start(2, 64, vec![spare(64, 2)], [None, None], 0);
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
            assert_eq!(
                capture.on_block(paused(stalled)),
                CaptureOutcome::NotPlaying
            );
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
        capture.start(1, 64, vec![spare(64, 1)], [None, None], 8);
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
        capture.start(1, 600_000, vec![spare(600_000, 1)], [None, None], 960_000);
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
    /// **宿主 pre-roll（负 `projectTimeSamples`）在写入前被夹掉。**
    ///
    /// 【为什么这是数据正确性】负起点会被投影成负 clip；渲染侧只夹窗口、不平移
    /// 源窗口，整段内容会前移一个 pre-roll。夹掉负部分、源偏移同步跳过，才是正确对齐。
    #[test]
    fn a_negative_preroll_block_is_clamped_into_the_project_domain() {
        let mut capture = started(64, 2);
        let (_, view) = block(-8, 16, 48_000.0);
        assert_eq!(capture.on_block(view), CaptureOutcome::Captured);
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed.len(), 1);
        assert_eq!(sealed[0].span(), 0..8, "只能写工程 0 之后的部分");
        assert!(
            (sealed[0].planes[0][0] - 8.0 / 1024.0).abs() < 1e-9,
            "源偏移必须跳过与 pre-roll 等长的帧数，否则内容整体前移"
        );
        assert_eq!(capture.preroll_frames_dropped, 8);
    }

    /// **整块都在工程 0 之前：不写、也不推进到“录制中”。**
    #[test]
    fn a_fully_negative_block_does_not_start_the_capture() {
        let mut capture = started(64, 2);
        let (_, view) = block(-16, 16, 48_000.0);
        assert_eq!(capture.on_block(view), CaptureOutcome::NotPlaying);
        assert!(!capture.is_started());
        assert_eq!(capture.captured_frames(), 0);
        assert!(capture.finish(ClosedReason::Stopped).is_empty());
        assert_eq!(capture.preroll_frames_dropped, 16);
    }

    /// **`kPlaying` 滞后但位置连续推进：第一块由 lookbehind 补回。**
    ///
    /// 【为什么这条是用户报障的回归】宿主在起播窗口里用 ASIO Guard / anticipative
    /// FX 提前处理时，前几块可能 `playing == false`。只按播放位判断会把开头整段丢掉。
    #[test]
    fn a_lagging_playing_bit_still_captures_the_first_block() {
        let mut capture = started(64, 2);
        let (_, first) = block(100, 8, 48_000.0);
        assert_eq!(
            capture.on_block_with_flow(paused(first), true),
            CaptureOutcome::NotPlaying,
            "第一块只暂存，等下一块证明连续性"
        );
        assert!(!capture.is_started());
        assert_eq!(capture.span(), None, "pending 不是素材");
        let (_, second) = block(108, 8, 48_000.0);
        assert_eq!(
            capture.on_block_with_flow(paused(second), true),
            CaptureOutcome::Captured,
            "第二块连续 ⇒ 提交 pending 并继续写"
        );
        assert!(capture.is_started());
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed.len(), 1);
        assert_eq!(sealed[0].span(), 100..116);
        assert_eq!(sealed[0].filled, 16, "开头那一块不能丢");
        assert_eq!(capture.pending_committed, 1);
    }

    /// **`kPlaying` 在第二块才置位：第一块同样从 pending 补回。**
    #[test]
    fn a_playing_edge_on_the_second_block_commits_the_pending_first_one() {
        let mut capture = started(64, 2);
        let (_, first) = block(20, 8, 48_000.0);
        assert_eq!(
            capture.on_block_with_flow(paused(first), true),
            CaptureOutcome::NotPlaying
        );
        let (_, second) = block(28, 8, 48_000.0);
        assert_eq!(capture.on_block(second), CaptureOutcome::Captured);
        assert!(capture.is_started());
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed[0].span(), 20..36);
        assert_eq!(sealed[0].filled, 16);
    }

    /// **停播时向前 seek 一块，再继续钉住：不能把它录成素材。**
    #[test]
    fn a_seek_block_while_stopped_is_discarded_by_the_lookbehind() {
        let mut capture = started(64, 2);
        let (_, seek) = block(100, 8, 48_000.0);
        assert_eq!(
            capture.on_block_with_flow(paused(seek), true),
            CaptureOutcome::NotPlaying
        );
        // 下一块换了位置（不是首尾相接）⇒ pending 被替换，丢弃旧的候选。
        let (_, elsewhere) = block(200, 8, 48_000.0);
        assert_eq!(
            capture.on_block_with_flow(paused(elsewhere), true),
            CaptureOutcome::NotPlaying
        );
        assert!(!capture.is_started());
        assert_eq!(capture.captured_frames(), 0);
        assert!(capture.finish(ClosedReason::Stopped).is_empty());
        assert!(capture.pending_discarded >= 1);
    }

    fn capture_with_both_streams(capacity: usize, spares: usize) -> Capture {
        let mut capture = Capture::default();
        capture.start(
            1,
            capacity,
            (0..spares).map(|_| spare(capacity, 1)).collect(),
            [Some(spare(capacity, 1)), Some(spare(capacity, 1))],
            0,
        );
        capture
    }

    /// **宿主只用 prefetch：捕获必须跟随它，而不是一直等实时流。**
    #[test]
    fn a_prefetch_only_host_is_captured_with_its_own_lookbehind() {
        let mut capture = capture_with_both_streams(64, 2);
        let (_, first) = block(0, 4, 48_000.0);
        assert_eq!(
            capture.on_block_with_context(
                paused(first),
                ctx(ProcessStream::Prefetch, false, false, true)
            ),
            CaptureOutcome::NotPlaying,
            "首块先暂存"
        );
        let (_, second) = block(4, 4, 48_000.0);
        assert_eq!(
            capture.on_block_with_context(
                paused(second),
                ctx(ProcessStream::Prefetch, false, false, true)
            ),
            CaptureOutcome::Captured,
            "同一流连续两块 ⇒ 选举并提交首块"
        );
        assert_eq!(capture.elected_stream(), Some(ProcessStream::Prefetch));
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed.len(), 1);
        assert_eq!(sealed[0].span(), 0..8, "开头那一块不能丢");
    }

    /// **两条流交错：每块都另起一段的旧行为必须消失。**
    #[test]
    fn interleaved_streams_do_not_fragment_the_capture() {
        let mut capture = capture_with_both_streams(64, 2);
        let feeds = [
            (ProcessStream::Prefetch, 0, 4),
            (ProcessStream::Realtime, 0, 4),
            (ProcessStream::Prefetch, 4, 4),
            (ProcessStream::Realtime, 4, 4),
        ];
        for (stream, start, frames) in feeds {
            let (_, view) = block(start, frames, 48_000.0);
            let _ = capture.on_block_with_context(paused(view), ctx(stream, false, false, true));
        }
        assert_eq!(capture.elected_stream(), Some(ProcessStream::Prefetch));
        assert_eq!(capture.backwards, 0, "另一条流不能把 expected_next 撕碎");
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed.len(), 1);
        assert_eq!(sealed[0].span(), 0..8);
        assert_eq!(capture.stream_switches, 0);
    }

    /// **交接处的半重叠必须裁剪前缀，而不是另起一段。**
    #[test]
    fn a_partially_overlapping_block_continues_the_same_segment() {
        let mut capture = started(64, 2);
        let (_, first) = block(0, 4, 48_000.0);
        assert_eq!(capture.on_block(first), CaptureOutcome::Captured);
        let (_, overlap) = block(2, 4, 48_000.0);
        assert_eq!(capture.on_block(overlap), CaptureOutcome::Captured);
        assert_eq!(capture.overlapped_frames, 2);
        assert_eq!(capture.captured_frames(), 6);
        assert_eq!(capture.span(), Some(0..6));
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed.len(), 1, "半重叠不是跳跃，不得切段");
        assert_eq!(sealed[0].span(), 0..6);
    }

    /// **同一区间的整块重复不是新素材。**
    #[test]
    fn an_exact_repeat_of_the_same_interval_is_skipped() {
        let mut capture = started(64, 2);
        let (_, first) = block(0, 4, 48_000.0);
        assert_eq!(capture.on_block(first), CaptureOutcome::Captured);
        let (_, repeat) = block(0, 4, 48_000.0);
        assert_eq!(capture.on_block(repeat), CaptureOutcome::Repeated);
        assert_eq!(capture.repeated_blocks, 1);
        assert_eq!(capture.captured_frames(), 4);
        assert_eq!(capture.span(), Some(0..4));
    }

    /// **`kRecording` 出现时立即从 prefetch 交接给实时流。**
    #[test]
    fn a_recording_block_hands_over_to_the_realtime_stream_immediately() {
        let mut capture = capture_with_both_streams(64, 2);
        for start in [0, 4] {
            let (_, view) = block(start, 4, 48_000.0);
            let _ = capture.on_block_with_context(
                paused(view),
                ctx(ProcessStream::Prefetch, false, false, true),
            );
        }
        assert_eq!(capture.elected_stream(), Some(ProcessStream::Prefetch));
        let (_, live) = block(8, 4, 48_000.0);
        assert_eq!(
            capture.on_block_with_context(
                paused(live),
                ctx(ProcessStream::Realtime, false, true, true)
            ),
            CaptureOutcome::Captured,
            "录音流的第一块不能被一条安静的首块延迟掉"
        );
        assert_eq!(capture.elected_stream(), Some(ProcessStream::Realtime));
        assert_eq!(capture.stream_switches, 1);
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(
            sealed.len(),
            1,
            "连续交接必须留在同一段里，不能因换流而切段"
        );
        assert_eq!(sealed[0].span(), 0..12);
    }

    /// **Resting 块即使带着 `kPlaying` 也不写。**
    #[test]
    fn a_resting_block_is_not_written_even_when_playing() {
        let mut capture = started(64, 2);
        let (_, first) = block(0, 4, 48_000.0);
        assert_eq!(capture.on_block(first), CaptureOutcome::Captured);
        let (_, repeat) = block(0, 4, 48_000.0);
        assert_eq!(
            capture.on_block_with_context(
                repeat,
                ctx_phase(
                    ProcessStream::Realtime,
                    true,
                    TransportPhase::Resting,
                    false
                )
            ),
            CaptureOutcome::NotPlaying
        );
        assert_eq!(capture.captured_frames(), 4);
        assert_eq!(capture.resting_blocks, 1);
    }

    /// **上升沿用新内容覆盖停播时写过的同几何块，而不是把它当成重复丢掉。**
    #[test]
    fn a_rising_edge_overwrites_the_stopped_same_geometry_block() {
        let mut capture = started(64, 2);
        let (_, first) = block(0, 4, 48_000.0);
        assert_eq!(capture.on_block(first), CaptureOutcome::Captured);
        let samples: &'static [f32] = Box::leak(vec![0.5_f32; 4].into_boxed_slice());
        let replacement = InputBlock {
            start: 0,
            frames: 4,
            sample_rate: 48_000.0,
            planes: [Some(samples), None],
            playing: true,
        };
        assert_eq!(
            capture.on_block_with_context(
                replacement,
                ctx_phase(ProcessStream::Realtime, true, TransportPhase::Resting, true)
            ),
            CaptureOutcome::Captured
        );
        assert_eq!(capture.captured_frames(), 4, "覆盖写不是新增帧");
        assert_eq!(capture.overwritten_blocks, 1);
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed[0].planes[0], vec![0.5; 4]);
    }

    /// **回卷时裁掉上一段越过新起点的尾巴。**
    #[test]
    fn a_backward_wrap_trims_the_overhang_tail() {
        let mut capture = started(64, 2);
        for start in [100, 104] {
            let (_, view) = block(start, 4, 48_000.0);
            assert_eq!(capture.on_block(view), CaptureOutcome::Captured);
        }
        // 同一循环起点重新开始：上一段 [100,108) 的 [104,108) 是越界的尾巴。
        let (_, wrap) = block(100, 4, 48_000.0);
        assert_eq!(capture.on_block(wrap), CaptureOutcome::Captured);
        assert_eq!(capture.overhang_trimmed_frames, 4);
        assert_eq!(capture.captured_frames(), 8, "4 - 4 + 4 + 4");
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed.len(), 2);
        assert_eq!(sealed[0].span(), 100..104);
        assert_eq!(sealed[1].span(), 100..104);
    }

    /// **回卷块不能被"完全覆盖"判定吞掉。**
    ///
    /// 【为什么这是问题三的一个直接成因】停播时宿主会重复投递同一区间（内容相同），
    /// 那必须跳过；循环回卷时块首也会回到同一位置，但**音频仍在流动**。此前两者
    /// 共用"完全落在已录区间内 ⇒ Repeated"，于是每一圈的首块都被丢掉，那一圈永远
    /// 开不了头。
    #[test]
    fn a_wrap_block_with_live_audio_starts_a_new_pass() {
        let mut capture = started(16, 2);
        for start in [0, 4, 8] {
            let (_, view) = block(start, 4, 48_000.0);
            assert_eq!(capture.on_block(view), CaptureOutcome::Captured);
        }
        // 回卷：位置回到 0，音频仍在流动。
        let (_, view) = block(0, 4, 48_000.0);
        let mut context = ctx(ProcessStream::Realtime, true, false, true);
        context.cycle_wrap = true;
        context.new_pass = true;
        assert_eq!(
            capture.on_block_with_context(view, context),
            CaptureOutcome::Captured,
            "回卷首块必须写入"
        );
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed.len(), 2, "回卷必须另起一段");
        assert_eq!(sealed[1].origin, 0, "新一圈从回卷点开始");
    }

    /// 停播时同一区间的重复投递仍然必须跳过（与上一条互为反例）。
    #[test]
    fn a_repeat_without_live_audio_is_still_skipped() {
        let mut capture = started(16, 2);
        for start in [0, 4, 8] {
            let (_, view) = block(start, 4, 48_000.0);
            capture.on_block(view);
        }
        let (_, view) = block(4, 4, 48_000.0);
        let mut context = ctx(ProcessStream::Realtime, true, false, false);
        context.phase = TransportPhase::Rolling;
        context.advancing = false;
        assert_eq!(
            capture.on_block_with_context(view, context),
            CaptureOutcome::Repeated,
            "没有音频流动的证据时，落在段内部的同一区间就是重复"
        );
    }

    /// **起播窗口结束时的小缺口不得清掉循环身份。**
    ///
    /// 【为什么】位置被钉住期间由本插件按块长记账，恢复前进时可能留下一个块级的取整
    /// 差。那不是"跳到了别处"，按它清掉 `loop_start` 会把同一圈拆成两个组。
    #[test]
    fn a_small_start_window_gap_keeps_the_loop_identity() {
        let mut capture = started(64, 2);
        let mut context = ctx(ProcessStream::Realtime, true, false, true);
        context.cycle_start = Some(0);
        context.new_pass = true;
        let (_, view) = block(0, 4, 48_000.0);
        capture.on_block_with_context(view, context);

        // 记账取整差：期望 4，实际 8 —— 只差一个块。
        let (_, view) = block(8, 4, 48_000.0);
        let mut context = ctx(ProcessStream::Realtime, true, false, true);
        context.cycle_start = Some(0);
        capture.on_block_with_context(view, context);

        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed.len(), 2);
        assert_eq!(sealed[0].loop_start, Some(0));
        assert_eq!(
            sealed[1].loop_start,
            Some(0),
            "块级缺口不是 seek，循环身份必须保留"
        );
    }

    /// 真正的大跳跃（seek 到别处）仍然退出循环 pass。
    #[test]
    fn a_large_jump_leaves_the_loop_pass() {
        let mut capture = started(64, 2);
        let mut context = ctx(ProcessStream::Realtime, true, false, true);
        context.cycle_start = Some(0);
        context.new_pass = true;
        let (_, view) = block(0, 4, 48_000.0);
        capture.on_block_with_context(view, context);

        let (_, view) = block(1_000, 4, 48_000.0);
        let mut context = ctx(ProcessStream::Realtime, true, false, true);
        context.cycle_start = Some(0);
        capture.on_block_with_context(view, context);

        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed[1].loop_start, None, "大跳跃必须退出循环组");
    }

    /// **起播 resync 的小缺口用静音桥接，且不新开段。**
    #[test]
    fn a_resync_gap_is_bridged_into_one_segment() {
        let mut capture = started(64, 2);
        let (_, view) = block(0, 4, 48_000.0);
        capture.on_block_with_context(view, ctx(ProcessStream::Realtime, true, false, true));

        // 宿主恢复后的位置比我们自己记账的终点晚 12 帧。
        let mut context = ctx(ProcessStream::Realtime, true, false, true);
        context.resync = true;
        context.resync_limit_frames = 100;
        let (_, view) = block(16, 4, 48_000.0);
        capture.on_block_with_context(view, context);

        assert_eq!(capture.resync_bridged_frames, 12);
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed.len(), 1, "记账纠正不能把 take 切成两段");
        assert_eq!(sealed[0].span(), 0..20);
    }

    /// **起播 resync 的向后纠正裁掉多记的尾巴，并继续同一段。**
    #[test]
    fn a_backward_resync_continues_the_same_segment() {
        let mut capture = started(64, 2);
        let (_, view) = block(0, 16, 48_000.0);
        capture.on_block_with_context(view, ctx(ProcessStream::Realtime, true, false, true));

        let mut context = ctx(ProcessStream::Realtime, true, false, true);
        context.resync = true;
        context.resync_limit_frames = 100;
        let (_, view) = block(8, 4, 48_000.0);
        capture.on_block_with_context(view, context);

        assert!(capture.overhang_trimmed_frames >= 8);
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed.len(), 1, "向后纠正不能把 take 切成两段");
        assert_eq!(sealed[0].span(), 0..12);
    }

    /// **真实 seek 不会被 resync 桥接吞掉。**
    #[test]
    fn a_real_seek_still_starts_a_new_segment() {
        let mut capture = started(64, 2);
        let (_, view) = block(0, 4, 48_000.0);
        capture.on_block_with_context(view, ctx(ProcessStream::Realtime, true, false, true));

        let mut context = ctx(ProcessStream::Realtime, true, false, true);
        context.resync = true;
        context.resync_limit_frames = 10;
        let (_, view) = block(1_000, 4, 48_000.0);
        capture.on_block_with_context(view, context);

        assert_eq!(capture.resync_bridged_frames, 0);
        let sealed = capture.finish(ClosedReason::Stopped);
        assert_eq!(sealed.len(), 2, "超出纠正上限的跳跃仍然另起一段");
    }
}
