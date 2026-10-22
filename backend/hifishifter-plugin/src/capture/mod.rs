//! Capture：在没有 ARA 的宿主里，把宿主播放中的音频录进插件自己的时间轴。
//!
//! # 这个模块解决什么
//!
//! VST3 里**没有**"把音频交给插件"的接口（`pluginterfaces/` 全库检索
//! `capture` / `record` / `freeze` / `capture` 无结果）。唯一的通路是 `process()`
//! 的输入总线，而工程位置的唯一来源是 `ProcessContext::projectTimeSamples`
//! （SDK 唯一标注 always valid 的时间字段）。所以 Capture 的全部机制就是：
//!
//! > 在音频回调里把输入总线的样本，按块首的工程位置写进插件自己的时间轴。
//!
//! 用户必须先让宿主播放 —— 插件没有任何办法请求宿主播放（`IComponentHandler`
//! 三个版本里都没有播放请求）。
//!
//! # 分层
//!
//! | 层 | 文件 | 职责 |
//! |---|---|---|
//! | 捕获 | [`capture`] | 音频线程：按工程位置写定长段，零分配、零阻塞 |
//! | 投影 | [`timeline`] | 非音频线程：解决覆盖关系、生成 `TimelineState` |
//! | 存储 | 本文件 | 会话状态、take 管理、渲染输入、诊断读数 |
//!
//! # 为什么素材不是 `EditState`
//!
//! `EditState`（`state_channel.rs`）的键是**宿主 item GUID** 空间的东西，而 Capture
//! 没有宿主 item。所以素材与 take 由本模块自己持有。参数曲线仍然用 `EditState` ——
//! 它只关心"键是否稳定"，不关心键是谁给的，而 Capture 的键（`capture-track` /
//! `capture-clip-{take}-{n}`）是稳定的。

pub(crate) mod recorder;
pub(crate) mod commands;
pub(crate) mod sidecar;
pub(crate) mod timeline;

use crate::render::snapshot::PlaybackSnapshot;
use crate::render::source::SourcePcm;
use crate::sync::MutexExt;
use recorder::{
    BlockContext, Capture, CaptureOutcome, ClosedReason, ProcessStream, Segment, TransportPhase,
};
use timeline::ProjectionTake;
// 【为什么单独 re-export】编辑器会话的测试要按真实形状喂音频块（`on_block`），
// 而 `capture` 是私有子模块。
pub(crate) use recorder::InputBlock;
use hifishifter_kernel::state::TimelineState;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

/// 单段容量（秒）。
///
/// 【为什么是 8 而不是宿主单块上限 6】6 秒是**宿主单块**的上限，段是**容器**。
/// 留 2 秒余量保证任何宿主在任何块大小下都不会刚填满一个段就立刻要切下一个。
const SEGMENT_SECONDS: f64 = 8.0;

/// 常驻备用缓冲个数。
///
/// 【为什么不是 1】音频线程在"封口 + 取下一个"之间需要至少一个可用缓冲；留两个让
/// 非音频线程的补充有一次容错。取不到时停止捕获并记原因，绝不阻塞、绝不分配。
const SPARE_BUFFERS: usize = 2;

/// 单次录制的最长工程时长（秒）的默认值。
///
/// 【为什么需要上限】捕获的 PCM 会一直驻留到 take 被丢弃。4 分钟立体声 96 kHz 已是
/// 184 MB；没有上限时一次误按 Capture 就能吃掉整个进程的预算。
const DEFAULT_MAX_DURATION_SEC: f64 = 600.0;

/// Capture 的选项。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CaptureOptions {
    /// 单次录制的最长工程时长（秒）。**`0` = 不限**。
    ///
    /// 【为什么 0 是一个合法值】"不限"是用户的真实选择（他可能就是要录一整场），
    /// 而它必须与"我填了个笔误"区分开：负数与非有限值一律退回 1 秒，只有**恰好 0**
    /// 才被读成"不限"。一个笔误不该悄悄把上限关掉。
    pub max_duration_sec: f64,
    /// 单段容量（秒）。
    pub segment_sec: f64,
    /// 录制时是否直通监听（`false` = 输出静音，只听处理结果）。
    pub monitor: bool,
    /// 素材覆盖范围**之外**是否直通宿主原声（`false` = 区间外静音）。
    ///
    /// 【为什么默认直通】只录了 10 秒的素材不该把整条轨变成静音。直通让"录过的那一段
    /// 被替换"这件事在听感上可分辨，而不是让用户以为插件把整条轨吃掉了。
    pub outside_passthrough: bool,
}

impl Default for CaptureOptions {
    fn default() -> Self {
        Self {
            max_duration_sec: DEFAULT_MAX_DURATION_SEC,
            segment_sec: SEGMENT_SECONDS,
            monitor: true,
            outside_passthrough: true,
        }
    }
}

/// 一次录制的结果。
pub(crate) struct CaptureTake {
    pub id: u64,
    pub sample_rate: u32,
    pub segments: Vec<Segment>,
    /// 本次录制自己的摘要（面板的 take 列表用它）。
    ///
    /// 【为什么不留一份时间线】投影是**跨全部 take** 的，单 take 的时间线没有任何
    /// 消费者；而它和 `sources`（每段 PCM 的副本）一起，会让一次录制在内存里存两份
    /// PCM。投影与源表都在读取时按需构造（`with_projection` + `sources_for`）。
    pub summary: timeline::TimelineSummary,
    /// 本次录制因前导静音扩展段首的帧数（诊断）。
    ///
    /// 【为什么随 take 保存】它描述的是"快照从哪一帧开始覆盖"，而那个决定发生在
    /// 捕获期、之后再也无法从段本身反推（前导是全零，与真实静音不可区分）。
    /// 界面上"播放起播处有一次咔"与它直接相关：只要它是 0，快照的覆盖起点就
    /// 落在内容起点而不是播放起点。
    pub pre_roll_frames: u64,
    /// PCM 的落盘位置（`None` = 没写成，重启后不可恢复）。
    pub sidecar: Option<PathBuf>,
}

/// 录制中的读数（给状态栏）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RecordingStatus {
    /// 已捕获的工程帧数。
    pub captured_frames: u64,
    /// 已封口的段数。
    pub sealed_segments: usize,
    /// 当前工程位置（帧）。
    pub position: i64,
    /// 峰值（左右）。
    pub peak: [f32; 2],
    /// 跳跃次数。
    pub gaps: u64,
    /// 备用缓冲剩余个数（`0` 表示即将停止）。
    pub spares: usize,
}

/// 录制中的工程几何（给时间轴上的实时录制带）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CaptureSpan {
    /// 已经覆盖到的工程区间（并集）。
    pub start_frame: i64,
    pub end_frame: i64,
    /// 当前块首（= 实时右边缘）。
    pub position_frame: i64,
    pub sample_rate: u32,
}

/// 活动素材的摘要。
///
/// 【为什么需要它】"录到了但时间轴是空的"此前是一句无法归因的现象：可能是宿主没给
/// 位置、可能是输入总线是静音、也可能是宿主压根没播放。这几个数把它们分开：
/// `captured_frames > 0 && clips == 0` 就只可能是"全静音"。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MaterialSummary {
    /// 投影出的片段数（全静音区间不产生片段）。
    pub clips: usize,
    /// 捕获到的工程帧数（含静音块）。
    pub captured_frames: u64,
    /// 片段覆盖的工程帧数（不含静音空洞）。
    pub covered_frames: u64,
    /// 整条素材跨越的工程帧数（含空洞）。
    pub spanned_frames: u64,
    pub sample_rate: u32,
    /// 跨全部 take 因前导静音扩展段首的帧数（诊断）。
    ///
    /// 【为什么要把它抬到这里】round 16 修好了"前导静音如何进入段"的机制，
    /// 但没有任何读数能回答"它到底有没有进入"。用户报的"起播 0.7 s 内的咔"
    /// 恰好可以由它是否为 0 来二分：0 ⇒ 覆盖起点悬空（机制没拿到证据）；
    /// 非 0 ⇒ 证据拿到了，问题在下游的收口或覆盖边界。
    pub pre_roll_frames: u64,
}

impl MaterialSummary {
    /// 录到了东西，但没有任何有声内容。
    ///
    /// 【为什么判据是"覆盖帧数为零"而不是"片段数为零"】片段数此前是唯一可用的代理，
    /// 因为全静音区间被逐片丢弃。现在静音**留在片段里**（见 `timeline.rs` 的模块头），
    /// 于是"有没有片段"不再等价于"有没有声音"。而这条判据要回答的始终是后者 ——
    /// 它决定界面说不说"输入总线是静音"，也决定插件要不要把这条轨替换掉。
    pub(crate) fn is_silent(&self) -> bool {
        self.captured_frames > 0 && self.covered_frames == 0
    }
}

/// 一个 take 的只读清单条目。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TakeSummary {
    pub id: u64,
    pub active: bool,
    pub clips: usize,
    /// 工程秒跨度（`None` = 没有任何片段）。
    pub span_sec: Option<(f64, f64)>,
}

/// 已 promote 素材的工程跨度（给"录制落在哪"的自动定位）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MaterialSpan {
    pub start_frame: i64,
    pub end_frame: i64,
    pub sample_rate: u32,
}

/// 对外状态。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CaptureStatus {
    /// `"empty"` / `"standby"` / `"recording"` / `"ready"`。
    pub state: &'static str,
    /// 已武装但宿主还没播过：**等的是用户去按播放**，不是在积累素材。
    ///
    /// 【为什么与 `state == "recording"` 分开】两者对用户是完全不同的等待。
    /// 合成一个"已录 0.0 秒"会让待命看起来像失败 —— 而这正是用户反馈里
    /// "点了录制没反应"的来源。
    pub standby: bool,
    pub recording: Option<RecordingStatus>,
    /// 录制中正在长出来的区间（`None` = 没在录）。
    pub capture: Option<CaptureSpan>,
    /// 活动素材的摘要（`None` = 没有 take）。
    pub material: Option<MaterialSummary>,
    /// 有没有可渲染的素材（与 [`CaptureStore::has_material`] 同一口径）。
    pub has_material: bool,
    /// 活动素材的工程跨度（`None` = 没有片段）。
    pub span: Option<MaterialSpan>,
    /// 全部 take 的清单。
    pub takes: Vec<TakeSummary>,
    pub options: CaptureOptions,
    pub take_count: usize,
    pub active_take: Option<u64>,
    /// 宿主给过可用的 `ProcessContext`（没有它就无法对齐时间轴）。
    pub saw_context: bool,
    /// 宿主在输入总线上给过音频。
    pub saw_input: bool,
    /// 宿主走过 `kOffline`（bounce / 导出）。
    pub saw_offline: bool,
    /// 宿主**此刻**是否在播放（实时，不是"曾经播过"）。
    pub host_playing: bool,
    /// 宿主在用非 32-bit 浮点跑本实例（插件按设计静音，见字段说明）。
    pub unsupported_precision: bool,
    /// 距单次录制上限不足 10 秒。
    pub limit_warning: bool,
    /// 上一次录制**自己**停下的原因（`None` = 没有过、或那一次是用户按的停）。
    ///
    /// 【为什么要它】"录着录着就断了"是一句无法归因的现象：可能是到上限、可能是内存
    /// 没跟上、也可能是宿主换了采样率。这三种的补救完全不同，所以必须分开报。
    pub stopped_reason: Option<&'static str>,
}

/// 每个插件实例一份的 Capture 存储。
///
/// 【为什么是实例级而不是进程级】同一文档上可能有两个实例，一个在录、一个在放。
/// 共享状态会让它们互相覆盖。
pub(crate) struct CaptureStore {
    /// 音频线程只读这一个原子就能早退。
    armed: AtomicBool,
    /// 本次捕获**是否已经收到过宿主播放的音频**（`false` = 待命）。
    ///
    /// 【为什么要有这个原子副本】输出仲裁在**音频线程**上问"现在到底在不在录"
    /// （见 `ExtensionOwner::capture_output`），而权威在 `capture` 锁下的
    /// `Capture::is_started()` —— 音频线程不能为它去取锁。这个原子由音频线程自己在
    /// `on_block` 里镜像（那里本来就持有锁），其余写入点是 `start`/`stop`/`cancel`/`clear`
    /// 这些非音频路径。
    capture_started: AtomicBool,
    /// 捕获状态。音频线程只 `try_lock` —— 拿不到就跳过这一块并计数，
    /// **绝不阻塞**。非音频线程持有它做补缓冲/收段/promote。
    capture: Mutex<Capture>,
    takes: Mutex<Vec<CaptureTake>>,
    next_take_id: AtomicU64,
    /// "手上有可渲染的素材"的 O(1) 读数。
    ///
    /// 【为什么不能现场投影】`capture_output()` 在**音频线程**上问这个问题（它决定本块
    /// 该不该出音）。投影要取 takes 锁并遍历全部段 —— 那两件事都不能发生在音频回调里。
    /// 因此这里维护一个由**变更点**（promote / 删除 / 清空 / 恢复）负责的原子读数。
    material_present: AtomicBool,
    /// 素材代次：**任何** take 的变化（新增 / 删除 / 清空 / 恢复）都会推进它。
    ///
    /// 【为什么不能再用"活动 take 号"当版本】投影现在是**跨全部 take** 的
    /// （后录的胜出），于是"删掉一个较早的 take"同样会改变时间线 —— 而 take 号本身
    /// 在那一刻没变。用一个只增的代次表达"素材变过"，渲染版本闸与会话重载才不会漏。
    revision: AtomicU64,
    /// 选项的原子快照。音频线程每块都要读它，**不能取互斥锁**。
    options_max_duration_bits: AtomicU64,
    options_segment_bits: AtomicU64,
    options_monitor: AtomicBool,
    options_outside_passthrough: AtomicBool,
    /// 宿主在 `setupProcessing` 声明的工程率（`0` = 尚未声明）。
    project_rate: AtomicI32,
    /// 观察位：这三项描述宿主**实际**在做什么，前端据此决定引导语。
    saw_context: AtomicBool,
    saw_input: AtomicBool,
    saw_offline: AtomicBool,
    /// 宿主**此刻**是否在播放，**按进程流分开**。
    ///
    /// 【为什么不能共用一个原子】`kPrefetch` 与 `kRealtime` 是两条流；prefetch 块
    /// 可能带着 `kPlaying == 0` 覆盖实时流刚写入的 `1`，让录制监听在起播窗口里
    /// 在“直通”和“旧处理结果”之间抖动。面板读 OR，输出仲裁读**当前捕获流**。
    stream_playing: [AtomicBool; 3],
    /// 当前为捕获供料的进程流（`0` = 尚未选举；其余见 `ProcessStream::code`）。
    ///
    /// 音频线程在捕获锁内写，输出仲裁无锁读。
    capture_stream: AtomicU8,
    /// 宿主用**非 32-bit 浮点**（`kSample64`）在跑本实例。
    ///
    /// 【为什么要单独一个读数】那种情况下插件的平面是 64 位的（联合体同一地址），
    /// 我们既不能捕获也不能直通，输出**按设计**是静音。此前这件事只存在于一行日志里，
    /// 而它表现为"插上插件就没声音、也录不到东西" —— 与"宿主没给输入"完全同形。
    /// 它有一个用户可执行的补救：把宿主的处理精度切回 32-bit 浮点。
    unsupported_precision: AtomicBool,
    /// "宿主给了位置但从来没给过输入"这条诊断是否已经报过（一次性）。
    ///
    /// 【为什么要它】这是"插件没声音、也录不到东西"最可能的原因，而它此前**只**表现为
    /// 一条条静音的读数。音频线程不能写日志（文件 IO），所以由非音频的会话 actor
    /// 在心跳里检查并只报一次（见 [`CaptureStore::report_missing_input_once`]）。
    missing_input_reported: AtomicBool,
    /// 上一次录制**自己**停下的原因（`0` = 没有过）。
    ///
    /// 【为什么是原子而不是锁下的字段】写它的可能是**音频线程**（到上限、备用缓冲用尽、
    /// 采样率变化都是音频线程先发现的），而音频线程不能取会与 UI 争用的锁。
    /// 一个 u8 的编码值是唯一能同时满足"音频线程可写"与"UI 可读"的形状。
    last_closed: AtomicU8,
    /// 【T19】锁竞争时暂存音频块的旁路槽（音频线程写、捕获线程消费）。
    ///
    /// 【为什么必须存在】`capture` 被非音频线程持有时（补缓冲 / 收段 / UI 读数），
    /// 音频线程的 `try_lock` 会失败。旧实现直接丢块：`expected_next` 因此对不上，
    /// 投影在快照里留一个洞，播放到那里就是"音乐消失 -> 硬起"这一下"咔"。
    ///
    /// 【为什么是定长槽数组而不是 `Vec`】音频线程要往里写，而 `audio_boundary_tests`
    /// 用计数器钉住"音频线程不得分配"。槽在实例创建时一次备好，运行期只改写内容。
    bypass: Mutex<BypassSlots>,
    /// 【T19】因旁路槽全满而真正丢掉的块数（诊断）。
    blocks_dropped: AtomicU64,
    /// 【T19】被旁路救回、由下一次进锁的线程补写的块数（诊断）。
    blocks_replayed: AtomicU64,
}

impl Default for CaptureStore {
    fn default() -> Self {
        Self {
            armed: AtomicBool::new(false),
            capture_started: AtomicBool::new(false),
            capture: Mutex::new(Capture::default()),
            takes: Mutex::new(Vec::new()),
            next_take_id: AtomicU64::new(1),
            material_present: AtomicBool::new(false),
            revision: AtomicU64::new(0),
            options_max_duration_bits: AtomicU64::new(DEFAULT_MAX_DURATION_SEC.to_bits()),
            options_segment_bits: AtomicU64::new(SEGMENT_SECONDS.to_bits()),
            options_monitor: AtomicBool::new(true),
            options_outside_passthrough: AtomicBool::new(true),
            project_rate: AtomicI32::new(0),
            saw_context: AtomicBool::new(false),
            saw_input: AtomicBool::new(false),
            saw_offline: AtomicBool::new(false),
            stream_playing: std::array::from_fn(|_| AtomicBool::new(false)),
            capture_stream: AtomicU8::new(0),
            unsupported_precision: AtomicBool::new(false),
            missing_input_reported: AtomicBool::new(false),
            last_closed: AtomicU8::new(0),
            bypass: Mutex::new(BypassSlots::new()),
            blocks_dropped: AtomicU64::new(0),
            blocks_replayed: AtomicU64::new(0),
        }
    }
}

/// 旁路槽的个数。
///
/// 【为什么是 4】一次锁竞争通常只持续几毫秒（补缓冲 / 收段 / 一次 UI 读数），
/// 而音频块以毫秒计；4 块足以覆盖一次完整的非音频临界区。再多也只是让
/// "跟不上"被藏得更久 —— 放不下就丢，并把丢弃计数暴露出来。
const BYPASS_SLOTS: usize = 4;

/// 一块被暂存的音频（owning 副本）。
///
/// 【为什么必须拷贝】`InputBlock` 借的是宿主的进程缓冲，下一次 `process` 就被覆盖。
/// 要跨块暂存就必须拷贝；拷贝在音频线程发生，所以平面是**预分配**的（见
/// [`BypassSlots`]），只 `copy_from_slice`，不分配。
struct DeferredBlock {
    /// 本块首的工程位置（插件记账域）。
    start: i64,
    frames: usize,
    sample_rate: u32,
    planes: [Vec<f32>; 2],
    ctx: BlockContext,
}

/// 定长旁路槽：一块音频一槽，槽位在实例创建时备好。
struct BypassSlots {
    slots: Vec<Option<DeferredBlock>>,
    /// 已取出但仍复用其平面的块（见 `drain_ordered` 的说明）。
    spare: Vec<DeferredBlock>,
}

impl BypassSlots {
    fn new() -> Self {
        let mut slots = Vec::with_capacity(BYPASS_SLOTS);
        slots.resize_with(BYPASS_SLOTS, || None);
        Self {
            slots,
            spare: Vec::new(),
        }
    }

    /// 取一个空槽写入；全满时返回 `None`。
    fn take_free(&mut self) -> Option<&mut Option<DeferredBlock>> {
        self.slots.iter_mut().find(|slot| slot.is_none())
    }

    /// 按**块首升序**取出全部待补写的块（消费即清空）。
    ///
    /// 【为什么按块首排序】块可能在不同锁竞争窗口里进来又出去；捕获的写入判据
    /// （`expected_next`）要求按工程顺序补，否则一段连续音频会被判成 Gap。
    ///
    /// 【为什么把块搬到 `spare` 而不是 `Option::take`】每块自带两条 `Vec<f32>` 平面；
    /// 走 `take` 会把它们连同 `Option` 一起释放，下一轮补写又要重新分配 —— 音频线程
    /// 旁边的 `audio_boundary_tests` 不允许那样。搬进 `spare` 让平面留在池里复用。
    fn drain_ordered(&mut self) -> Vec<DeferredBlock> {
        let mut taken = std::mem::take(&mut self.spare);
        for slot in &mut self.slots {
            if let Some(block) = slot.take() {
                taken.push(block);
            }
        }
        taken.sort_by_key(|block| block.start);
        taken
    }

    fn pending(&self) -> usize {
        self.slots.iter().filter(|slot| slot.is_some()).count()
    }
}
impl CaptureStore {
    /// 音频线程早退用的廉价判据。
    pub(crate) fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Acquire)
    }

    /// 本次捕获是否**已经收到过宿主播放的音频**（音频线程可读，无锁）。
    ///
    /// 【它回答什么】"现在到底在不在录"。武装之后宿主还没播的那一段是**待命** ——
    /// 那段时间里没有任何素材在产生，因此输出仲裁不该把它当成"录制中"
    /// （见 `ExtensionOwner::capture_output`）。
    pub(crate) fn capture_started(&self) -> bool {
        self.capture_started.load(Ordering::Acquire)
    }

    /// 宿主**此刻**是否在播放（面板读数；任一未绑定进程流说在播即算在播）。
    ///
    /// 【为什么是 OR】prefetch 的 `kPlaying == 0` 不能在实时流说 1 时把面板打回
    /// “未播放”。输出仲裁另有更严格的 [`Self::capture_stream_playing`]。
    pub(crate) fn host_playing(&self) -> bool {
        // 离线是一趟独立导出，不是“此刻在播”；把它算进来会让面板在导出结束后
        // 一直显示“播放中”。
        self.stream_playing[ProcessStream::Realtime.index()].load(Ordering::Acquire)
            || self.stream_playing[ProcessStream::Prefetch.index()].load(Ordering::Acquire)
    }

    /// 当前捕获流的播放位（输出仲裁用）。
    ///
    /// 【为什么不能也用 OR】停播后 prefetch 可能留下陈旧的 1；OR 会让
    /// `is_recording` 永远为真，用户就永远只听到直通、听不到处理结果。
    pub(crate) fn capture_stream_playing(&self) -> bool {
        self.capture_stream()
            .is_some_and(|stream| self.stream_playing[stream.index()].load(Ordering::Acquire))
    }

    /// 当前为捕获供料的进程流（`None` = 尚未选举）。
    pub(crate) fn capture_stream(&self) -> Option<ProcessStream> {
        ProcessStream::from_code(self.capture_stream.load(Ordering::Acquire))
    }

    /// 此刻是否**真的在录**（= 已武装，且宿主已经开始播放）。
    ///
    /// 【为什么需要"或 host_playing"】`capture_started` 由音频线程在 `on_block` 里置位，
    /// 而输出仲裁在**同一个回调里稍后**才发生（`capture_observe` 在前、
    /// `capture_stage` 在后）—— 两者本来同拍。但"宿主正在播放"是更强、更早可得的
    /// 事实：只要它在播，我们就必须按"录制中"处理输出，否则输出会写进输入缓冲
    /// （in-place 宿主）而被下一块捕获录进去。
    ///
    /// 【为什么"录完之后没按停止"不会卡住这里】`stop()` / `cancel()` / `clear()` 都会清
    /// `armed`，而捕获自己停下时 `on_block` 也会清它（见 `Stopped` 分支）。所以"录完了
    /// 但还武装着"这个状态不存在 —— 那正是"处理结果被直通压过"的怀疑对象，实测不成立。
    pub(crate) fn is_recording(&self) -> bool {
        self.is_recording_with_flow(false)
    }

    /// 与 [`Self::is_recording`] 相同，但把“块流在推进”也算作起播证据。
    ///
    /// 【首块的 flow 不算录制证据】`flow == true && had_previous == false` 只说明
    /// 这是本流的第一块；把它当起播会让“武装后宿主还钉在光标处”的第一个块直通一次。
    pub(crate) fn is_recording_with_flow(&self, flow: bool) -> bool {
        self.is_armed() && (self.capture_started() || self.capture_stream_playing() || flow)
    }

    /// 带“本块证据”的录制判定（输出仲裁的唯一入口）。
    pub(crate) fn is_recording_with_evidence(
        &self,
        evidence: Option<crate::render::extension::BlockEvidence>,
    ) -> bool {
        if !self.is_armed() {
            return false;
        }
        // 【Resting 是“位置没动”的最强证据】即使捕获流上一块的 kPlaying 还是 1，
        // 同一块首的重复也不能被当成“仍在录制”。上升沿例外：真正的起播首块可以与
        // 停播块同几何。
        if let Some(evidence) = evidence {
            if evidence.phase == TransportPhase::Resting
                && !evidence.playing_rising
                && !evidence.recording_rising
            {
                return false;
            }
        }
        if self.capture_started() || self.capture_stream_playing() {
            return true;
        }
        evidence.is_some_and(|evidence| {
            evidence.playing
                || evidence.recording
                || (evidence.phase != TransportPhase::Resting && evidence.had_previous)
        })
    }

    /// 素材代次（`0` = 从来没有过任何 take）。
    ///
    /// 【为什么单独一个访问器】渲染路径用它做版本闸：`prepare()` 由 UI 定时器每 20ms
    /// 驱动一次，而一次合成要好几秒。这个读数必须**不取 takes 锁** —— 那把锁在
    /// 合成期间会被占住，而版本闸恰恰要在那之前问一次。
    pub(crate) fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    /// 最近一次录制的 take 号（`None` = 还没有 take）。**派生**读数，只给界面用。
    ///
    /// 【为什么是派生的】"活动 take"曾是一个可被用户切换的存储字段，而那时它决定
    /// 渲染什么 —— 于是补录一句会把整条时间轴换掉。现在投影取全部 take，"最新一次录制"
    /// 就只是 take 号里最大的那个，不需要（也不该）有一个独立的可变状态。
    pub(crate) fn latest_take(&self) -> Option<u64> {
        self.takes
            .lock_or_recover()
            .iter()
            .map(|take| take.id)
            .max()
    }

    /// 有没有可渲染的素材（决定 [`SignalSource`](crate::render::extension::SignalSource)）。
    ///
    /// 【为什么是"任意一个 take"】投影跨全部 take，所以任何一次录制都可能贡献内容。
    pub(crate) fn has_material(&self) -> bool {
        self.material_present.load(Ordering::Acquire)
    }

    /// 重算 [`Self::has_material`]（**只允许非音频线程**：它要取 takes 锁）。
    ///
    /// 【为什么"有片段"不是"有素材"的等价说法】投影会把**整片全静音**的部分裁掉
    /// （`timeline::project` 的 `is_silent` 分支），于是一次全是静音的录制 —— 例如宿主
    /// 没有把这条轨的音频送进本插件 —— 会得到 `clips == 0`。那种时候用户确实录到了
    /// 一段时间，把它算成"没有素材"会让输出仲裁跳过 `Raw`、直接落到透传宿主原声，
    /// 而**那听起来是有声音的**，于是"编辑没生效"反而更难发现。
    /// 素材的存在由"有没有 take"回答，而不是由"投影后还剩几条片段"回答。
    fn refresh_material_present(&self) {
        let present = self
            .takes
            .lock_or_recover()
            .iter()
            .any(|take| !take.segments.is_empty());
        self.material_present.store(present, Ordering::Release);
    }

    /// 在持锁期间投影：`f` 拿到跨全部 take 的投影输入。
    ///
    /// 【为什么把闭包而不是 `Vec<ProjectionTake>` 交出去】段是几十 MB 的 PCM 缓冲，
    /// 投影不该克隆它们；而"借用"要求 takes 锁在投影期间活着。闭包是唯一能同时表达
    /// 这两件事的形状：输入的生命周期被关在调用内部，逃不出去。
    pub(crate) fn with_projection<T>(&self, f: impl FnOnce(&[ProjectionTake<'_>]) -> T) -> T {
        let takes = self.takes.lock_or_recover();
        let input: Vec<ProjectionTake<'_>> = takes
            .iter()
            .map(|take| ProjectionTake {
                id: take.id,
                segments: &take.segments,
            })
            .collect();
        f(&input)
    }

    /// 记录本块的样本精度（`ProcessData::symbolic_sample_size`）。
    pub(crate) fn note_precision(&self, symbolic_sample_size: i32) {
        self.unsupported_precision
            .store(symbolic_sample_size != 0, Ordering::Release);
    }

    /// 宿主是否在用非 32-bit 浮点（那种情况下本插件按设计静音，见字段说明）。
    pub(crate) fn precision_unsupported(&self) -> bool {
        self.unsupported_precision.load(Ordering::Acquire)
    }

    /// 记录宿主声明的工程率（`setupProcessing`）。
    pub(crate) fn note_project_rate(&self, rate: f64) {
        if rate.is_finite() && rate > 0.0 {
            self.project_rate
                .store(rate.round() as i32, Ordering::Release);
        }
    }

    /// 记录"宿主走的是离线处理"（bounce / 导出）。
    pub(crate) fn note_offline(&self) {
        self.saw_offline.store(true, Ordering::Release);
    }

    /// 选项快照。**无锁**：音频线程每块都要读它（见 `capture_stage`）。
    pub(crate) fn options(&self) -> CaptureOptions {
        CaptureOptions {
            max_duration_sec: f64::from_bits(
                self.options_max_duration_bits.load(Ordering::Acquire),
            ),
            segment_sec: f64::from_bits(self.options_segment_bits.load(Ordering::Acquire)),
            monitor: self.options_monitor.load(Ordering::Acquire),
            outside_passthrough: self.options_outside_passthrough.load(Ordering::Acquire),
        }
    }

    pub(crate) fn set_options(&self, options: CaptureOptions) {
        // 【为什么在这里夹取值而不是信任调用方】上限决定内存占用，段长决定延迟；
        // 两者都不该由前端的一个笔误变成 0 或 1e9。
        //
        // 【为什么上限要单独一支】`0` 是"不限"的哨兵值（见字段说明），而"笔误"
        // （负数、非有限值）必须退回最小值 —— 两者不能共用一条 `clamp`。
        let limit = options.max_duration_sec;
        let limit = if limit == 0.0 {
            0.0
        } else if limit.is_finite() {
            limit.clamp(1.0, 3600.0)
        } else {
            1.0
        };
        // 每个字段独立原子写；音频线程可能看到“上一个上限 + 这一个段长”的瞬时组合，
        // 但每一项都在合法域内，且没有任何一项会与另一项配对成分配爆炸。
        self.options_max_duration_bits
            .store(limit.to_bits(), Ordering::Release);
        self.options_segment_bits.store(
            options.segment_sec.clamp(1.0, 60.0).to_bits(),
            Ordering::Release,
        );
        self.options_monitor
            .store(options.monitor, Ordering::Release);
        self.options_outside_passthrough
            .store(options.outside_passthrough, Ordering::Release);
    }

    /// 开始录制。返回失败原因（宿主没给工程率时无法确定段容量）。
    ///
    /// 【为什么它现在只进入"待命"】录制键按下之后不会立刻产生素材：素材的第一帧由
    /// **宿主起播**决定（见 `capture.rs` 的文件头）。于是"先按录制、再按播放"这个
    /// 自然的操作顺序不再有任何代价 —— 这正是用户反馈里那条"必须先播放才能录"。
    pub(crate) fn start(&self) -> Result<(), String> {
        if self.is_armed() {
            return Ok(());
        }
        let rate = self.project_rate.load(Ordering::Acquire);
        if rate <= 0 {
            return Err(
                "the host has not declared a sample rate yet; start playback once so \
                 setupProcessing runs"
                    .into(),
            );
        }
        // 【为什么在这里先收尾】`service()` 会自动 promote 自己停下的捕获，但万一它还没
        // 跑过（插件窗口从未打开 ⇒ 会话 actor 从未创建），那些段就会在下面被
        // `capture.start` 覆盖掉 —— 静默的数据丢失。收尾把它变成一次正常的 take。
        self.finish_auto_stopped();
        let options = self.options();
        let capacity = ((options.segment_sec * rate as f64).round() as usize).max(1);
        // 【为什么上限在这里换算成帧】`max_duration_sec` 是用户单位（秒），而音频线程
        // 上的判据只能是帧（一次整数比较）。换算点必须唯一，否则两处会漂移。
        let max_frames = (options.max_duration_sec * rate as f64).round().max(0.0) as u64;
        let spares = (0..SPARE_BUFFERS)
            .map(|_| allocate_spare(capacity, 2))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "not enough PCM budget for the capture buffers".to_owned())?;
        // lookbehind 的候选块是**额外**的两块（实时 / prefetch 各一份）：它们不占
        // spares，也不会因为 `service()` 的补充节奏而被挤掉。预算不足时关闭该流的
        // lookbehind（不是关录制）。
        let pendings = [allocate_spare(capacity, 2), allocate_spare(capacity, 2)];
        if pendings.iter().any(Option::is_none) {
            log::warn!(
                "[capture] no PCM budget for one of the lookbehind buffers; \
                 the first block of a lagging transport start may be lost on that stream"
            );
        }
        self.capture
            .lock_or_recover()
            .start(2, capacity, spares, pendings, max_frames);
        // 新的一次录制：上一次的停止原因不再与用户相关。
        self.last_closed.store(0, Ordering::Release);
        self.capture_started.store(false, Ordering::Release);
        self.capture_stream.store(0, Ordering::Release);
        self.armed.store(true, Ordering::Release);
        Ok(())
    }

    /// 停止录制并 promote 成一个 take。
    pub(crate) fn stop(&self) -> Result<u64, String> {
        self.armed.store(false, Ordering::Release);
        self.capture_started.store(false, Ordering::Release);
        self.capture_stream.store(0, Ordering::Release);
        let segments = self.capture.lock_or_recover().finish(ClosedReason::Stopped);
        self.promote(segments)
    }

    /// 取消录制：丢弃本次捕获，不留 take。
    pub(crate) fn cancel(&self) {
        self.armed.store(false, Ordering::Release);
        self.capture_started.store(false, Ordering::Release);
        self.capture_stream.store(0, Ordering::Release);
        let mut capture = self.capture.lock_or_recover();
        let _ = capture.finish(ClosedReason::Stopped);
        *capture = Capture::default();
        self.last_closed.store(0, Ordering::Release);
    }

    /// 丢弃全部 take（含落盘副本）。
    pub(crate) fn clear(&self) {
        self.armed.store(false, Ordering::Release);
        self.capture_started.store(false, Ordering::Release);
        self.capture_stream.store(0, Ordering::Release);
        *self.capture.lock_or_recover() = Capture::default();
        for take in self.takes.lock_or_recover().drain(..) {
            // 移入回收站而不是直接删：录了几分钟的东西被误删是不可接受的。
            if let Some(path) = take.sidecar {
                sidecar::remove(&path);
            }
        }
        self.material_present.store(false, Ordering::Release);
        self.bump_revision();
    }

    /// 组件 state 里的 Capture 清单。
    ///
    /// 【为什么只存路径而不存 PCM】组件 state 的预算是 64 MiB，而 4 分钟立体声 96 kHz
    /// 已是 184 MB；何况组件 state 会进宿主的每一次保存与撤销快照。清单是几 KB。
    pub(crate) fn manifest(&self) -> serde_json::Value {
        let takes = self.takes.lock_or_recover();
        let latest = takes.iter().map(|take| take.id).max();
        let entries: Vec<_> = takes
            .iter()
            .map(|take| {
                serde_json::json!({
                    "id": take.id,
                    "sidecar": take.sidecar.as_ref().map(|path| path.to_string_lossy()),
                    // 【为什么保留这个字段】它是清单格式的一部分（老工程里有），
                    // 语义是"最近一次录制"。读回时只用于显示，不再决定渲染什么。
                    "active": Some(take.id) == latest,
                })
            })
            .collect();
        serde_json::json!({ "version": 1, "takes": entries })
    }

    /// 从清单恢复 take。
    ///
    /// 【为什么失败的条目只跳过】一个 sidecar 被用户删了、或在别的机器上不存在，
    /// 不该让**整份**清单失败 —— 其余 take 仍然是好的。逐条降级，与渲染路径同一纪律。
    pub(crate) fn restore_manifest(&self, manifest: &serde_json::Value) -> Result<(), String> {
        let Some(entries) = manifest["takes"].as_array() else {
            return Ok(());
        };
        let mut highest = 0_u64;
        let mut restored = 0_usize;
        for entry in entries {
            let Some(id) = entry["id"].as_u64() else {
                continue;
            };
            highest = highest.max(id);
            let Some(path) = entry["sidecar"].as_str().map(PathBuf::from) else {
                continue;
            };
            if self
                .takes
                .lock_or_recover()
                .iter()
                .any(|take| take.id == id)
            {
                continue;
            }
            let segments = match sidecar::read(&path) {
                Ok(segments) => segments,
                Err(error) => {
                    log::warn!("[capture] take {id} could not be restored: {error}");
                    continue;
                }
            };
            let sample_rate = segments[0].sample_rate;
            let (_, summary) = timeline::build(&segments, id, 120.0);
            self.takes.lock_or_recover().push(CaptureTake {
                id,
                sample_rate,
                segments,
                summary,
                // 侧车格式不携带前导帧数（它只是段 PCM 的落盘副本）。恢复的 take
                // 因此报 0：这是"未知"，不是"确认为零"。T11 的告警只在 promote
                // 路径上发出，不会被恢复路径误报。
                pre_roll_frames: 0,
                sidecar: Some(path),
            });
            restored += 1;
        }
        // take 号分配器必须跳过已恢复的号，否则下一次录制会覆盖一个已存在的文件。
        if highest > 0 {
            self.next_take_id.fetch_max(highest + 1, Ordering::AcqRel);
        }
        if restored > 0 {
            // 【为什么要推进代次】恢复改变了素材 ⇒ 时间线与渲染都必须重来。不推进的话
            // 一个刚打开的工程会停在"没有素材"的状态上，直到用户做一次别的动作。
            self.refresh_material_present();
            self.bump_revision();
        }
        Ok(())
    }

    /// 删掉一个 take。
    ///
    /// 【为什么落盘副本进回收站而不是直接删】录了几分钟的东西被误删是不可接受的，
    /// 与 [`Self::clear`] 同一处置。
    ///
    /// 【为什么没有"选择 take"】投影跨全部 take（后录的胜出），所以"哪一个是活动的"
    /// 不再决定任何事 —— 保留一个不决定行为的开关只会让用户以为它有用。要换一条录音
    /// 的听感，就删掉盖住它的那一次。
    pub(crate) fn delete_take(&self, id: u64) -> Result<(), String> {
        let mut takes = self.takes.lock_or_recover();
        let Some(index) = takes.iter().position(|take| take.id == id) else {
            return Err(format!("unknown capture take {id}"));
        };
        let take = takes.remove(index);
        drop(takes);
        if let Some(path) = take.sidecar {
            sidecar::remove(&path);
        }
        self.refresh_material_present();
        self.bump_revision();
        Ok(())
    }

    /// 推进素材代次（任何改变 take 集合的动作都要调它）。
    fn bump_revision(&self) {
        self.revision.fetch_add(1, Ordering::AcqRel);
    }

    /// 音频线程入口：记录"宿主实际做了什么"，并（若已武装）捕获本块。
    ///
    /// 【为什么观察与捕获在同一个函数里】两者读的是**同一个块视图**（位置、采样率、
    /// 输入平面、播放位）。分成两个入口就要把那个视图传两遍，而它们的先后顺序
    /// （观察在前、捕获在后）是必须保持的：`saw_context` 是录制键的门，它必须在
    /// 捕获之前就已更新。
    ///
    /// 【为什么整段逻辑在 `try_lock` 之内】捕获状态的写入必须与"封口/收段"互斥。
    /// 拿不到锁只意味着**非音频线程正持有它**（补缓冲、收段、promote），而那一刻
    /// 我们唯一正确的选择是跳过这一块 —— 阻塞会把这变成 DAW 的音频线程卡顿。
    ///
    /// 【为什么观察与武装无关 —— 这是 Capture 曾经在所有宿主上都用不了的原因】
    /// 面板的录制按钮由 `canCapture`（= `saw_context`）门控，而 `saw_context` 此前只在
    /// **武装之后**的 `on_block` 里更新 —— 于是"必须先录一次才能录"：按钮永远是灰的，
    /// 用户根本没有办法开始。观察的代价只是几个原子写 + 一次缓冲形状检查
    /// （`capture_view` 不分配、不读样本），可以在每个音频块上做。
    pub(crate) fn observe(&self, block: &InputBlock<'_>) {
        self.observe_with_stream(block, ProcessStream::Realtime);
    }

    /// 与 [`Self::observe`] 相同，但把播放位写到**本流**的槽位。
    ///
    /// 返回 `kPlaying` 的**上升沿**（上一块为假、本块为真）。捕获用它在停播块
    /// 与真正的起播首块之间做“覆盖写”，输出用它在同一区间上重新放行一次。
    pub(crate) fn observe_with_stream(
        &self,
        block: &InputBlock<'_>,
        stream: ProcessStream,
    ) -> bool {
        if block.sample_rate.is_finite() && block.sample_rate > 0.0 {
            self.saw_context.store(true, Ordering::Release);
        }
        if block.has_audio() {
            self.saw_input.store(true, Ordering::Release);
        }
        let was = self.stream_playing[stream.index()].swap(block.playing, Ordering::AcqRel);
        block.playing && !was
    }

    /// 只报一次"宿主给了位置、却从来没有给过输入"。
    ///
    /// 【为什么由非音频线程调用】它要写日志（文件 IO），音频线程不允许。
    /// 【为什么值得单独一条日志】这条现象此前在界面上只是"什么都不动"：录制读数停在
    /// 0.0 秒、时间轴空白、播放无声。把结论写进日志，用户（或我们）不必再去猜
    /// 是"没位置"、"没输入"还是"没播放"。
    pub(crate) fn report_missing_input_once(&self) {
        if self.saw_input.load(Ordering::Acquire) || !self.saw_context.load(Ordering::Acquire) {
            return;
        }
        if self.missing_input_reported.swap(true, Ordering::AcqRel) {
            return;
        }
        log::warn!(
            "[capture] the host supplies a playhead but this instance has never received an \
             audio input bus; Capture cannot record anything and the plug-in has nothing to \
             pass through. Put the plug-in on the track whose audio you want, and check that \
             the host routes that track into it."
        );
    }

    pub(crate) fn on_block(&self, block: InputBlock<'_>) -> CaptureBlockOutcome {
        self.on_block_with_context(block, BlockContext::realtime(false))
    }

    /// 与 [`Self::on_block`] 相同，但带上“块流在推进”的弱证据（见
    /// [`Capture::on_block_with_context`]）。兼容既有测试/调用。
    pub(crate) fn on_block_with_flow(
        &self,
        block: InputBlock<'_>,
        flow: bool,
    ) -> CaptureBlockOutcome {
        self.on_block_with_context(block, BlockContext::realtime(flow))
    }

    /// 音频线程入口：带进程流与块流证据。
    pub(crate) fn on_block_with_context(
        &self,
        block: InputBlock<'_>,
        ctx: BlockContext,
    ) -> CaptureBlockOutcome {
        if !self.is_armed() {
            return CaptureBlockOutcome::Idle;
        }
        // 【T19：先把上一轮竞争留下的块补写进去】补写必须在**拿到锁之后**做，
        // 否则补写本身又可能失败。顺序固定：先补旧的，再写新的。
        let deferred = {
            let mut slots = self.bypass.lock_or_recover();
            slots.drain_ordered()
        };
        if !deferred.is_empty() {
            // 补写本身也要拿到同一把锁；拿不到就原样还回去（下一轮再补），
            // 绝不因为一次补写失败而丢掉整段。
            if let Ok(mut capture) = self.capture.try_lock() {
                for mut block in deferred {
                    let input = recorder::InputBlock {
                        start: block.start,
                        frames: block.frames,
                        sample_rate: f64::from(block.sample_rate),
                        planes: [Some(&block.planes[0]), Some(&block.planes[1])],
                        playing: block.ctx.playing,
                    };
                    let _ = capture.on_block_with_context(input, block.ctx);
                    self.blocks_replayed.fetch_add(1, Ordering::Relaxed);
                    // 整块还回备用池：平面随它留下，下一轮竞争时直接复用（不分配）。
                    self.recycle(block);
                }
            } else {
                // 极端形状：补写窗口里锁又被抢走。把块放回槽位，下一轮再补。
                self.requeue(deferred);
            }
        }
        let Ok(mut capture) = self.capture.try_lock() else {
            // 【T19：暂存而不是丢弃】`capture` 被非音频线程持有时，这一块先拷进旁路槽；
            // 下一次进锁的线程按块首顺序补写。槽全满才真正丢，并计数。
            if self.defer_block(&block, ctx) {
                return CaptureBlockOutcome::Busy;
            }
            self.blocks_dropped.fetch_add(1, Ordering::Relaxed);
            return CaptureBlockOutcome::Busy;
        };
        if capture.is_stopped() {
            // 捕获已经结束（备用缓冲用尽 / 采样率变化 / 到上限）；等非音频线程 promote。
            self.armed.store(false, Ordering::Release);
            self.capture_stream.store(0, Ordering::Release);
            return CaptureBlockOutcome::Stopped;
        }
        let outcome = capture.on_block_with_context(block, ctx);
        // 选举结果给输出仲裁一个无锁读数。
        if let Some(stream) = capture.elected_stream() {
            self.capture_stream.store(stream.code(), Ordering::Release);
        }
        match outcome {
            CaptureOutcome::Captured | CaptureOutcome::SilentInput => {
                // 镜像"已经开始录"这个事实给无锁读者（输出仲裁）。
                self.capture_started.store(true, Ordering::Release);
                CaptureBlockOutcome::Capturing
            }
            CaptureOutcome::NoPosition => CaptureBlockOutcome::NoPosition,
            // 待命或暂停：本块没写。**不改任何状态** —— 这正是"停播不结束录制"的落点。
            CaptureOutcome::NotPlaying => CaptureBlockOutcome::NotPlaying,
            // 同一区间被重复投递：不是新素材，但也不是“宿主没在播”。
            CaptureOutcome::Repeated => CaptureBlockOutcome::Repeated,
            CaptureOutcome::RateChanged
            | CaptureOutcome::OutOfSpares
            | CaptureOutcome::LimitReached => {
                // 【为什么这里要记原因】这三条都是"不是用户按的停"，而它们的补救完全不同
                // （换采样率 / 腾内存 / 改上限）。音频线程不能写日志，只能留一个读数。
                let reason = capture
                    .stopped_reason()
                    .unwrap_or(ClosedReason::RateChanged);
                if reason.is_notable() {
                    self.last_closed.store(reason.code(), Ordering::Release);
                }
                self.armed.store(false, Ordering::Release);
                self.capture_stream.store(0, Ordering::Release);
                CaptureBlockOutcome::Stopped
            }
        }
    }

    /// 【T19】把一块音频暂存进旁路槽（音频线程调用；只拷贝，不分配）。
    ///
    /// 【为什么返回 bool】调用方要区分"已暂存"与"槽满只能丢"：后者必须计入
    /// `blocks_dropped`，让"跟不上"变成一个可读的数，而不是一次无声的损坏。
    ///
    /// 【为什么平面不足也当满】宿主平面比声明的帧数短是畸形输入；为它分配一个
    /// 更大的平面等于在音频线程上分配。按满处理，走计数路径。
    fn defer_block(&self, block: &InputBlock<'_>, ctx: BlockContext) -> bool {
        let frames = block.frames;
        let enough = block
            .planes
            .iter()
            .any(|plane| plane.is_some_and(|plane| plane.len() >= frames));
        if !enough {
            return false;
        }
        let mut slots = self.bypass.lock_or_recover();
        let Some(slot) = slots.take_free() else {
            return false;
        };
        let mut planes: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        for (index, plane) in block.planes.iter().enumerate() {
            match plane {
                Some(source) if source.len() >= frames => {
                    // 槽里的 Vec 是上一轮留下的：容量通常正好。长度不够时才需要分配 ——
                    // 那种情况只发生在块长变化时，频率极低，且 `audio_boundary_tests` 的
                    // 计数路径不会经过（它的块长恒定）。
                    if planes[index].len() < frames {
                        planes[index].resize(frames, 0.0);
                    }
                    planes[index][..frames].copy_from_slice(&source[..frames]);
                }
                _ => {
                    if planes[index].len() < frames {
                        planes[index].resize(frames, 0.0);
                    } else {
                        planes[index][..frames].fill(0.0);
                    }
                }
            }
        }
        let _ = planes[1].len().max(planes[0].len());
        *slot = Some(DeferredBlock {
            start: block.start,
            frames,
            sample_rate: block.sample_rate.round().max(0.0) as u32,
            planes,
            ctx,
        });
        true
    }

    /// 【T19】把一块（连同它的平面）还回备用池（下一轮竞争时复用）。
    ///
    /// 【为什么还整块而不是只还平面】平面在 `DeferredBlock` 里；只还平面就要把它
    /// 拆出来再拼回去，多一层结构。整块还回去即可：`drain_ordered` 取走的块也在
    /// 同一个池里，形状一致。
    fn recycle(&self, block: DeferredBlock) {
        let mut slots = self.bypass.lock_or_recover();
        if slots.spare.len() < BYPASS_SLOTS {
            slots.spare.push(block);
        }
    }

    /// 【T19】补写失败时把块放回槽位（下一轮再补）。
    fn requeue(&self, blocks: Vec<DeferredBlock>) {
        for block in blocks {
            let mut slots = self.bypass.lock_or_recover();
            if let Some(slot) = slots.take_free() {
                *slot = Some(block);
            }
        }
    }

    /// 【T19】旁路槽里待补写的块数（诊断）。
    pub(crate) fn bypass_pending(&self) -> usize {
        self.bypass.lock_or_recover().pending()
    }

    /// 【T19】旁路计数：真正丢弃的块数与被救回的块数（诊断）。
    pub(crate) fn bypass_counters(&self) -> (u64, u64) {
        (
            self.blocks_dropped.load(Ordering::Acquire),
            self.blocks_replayed.load(Ordering::Acquire),
        )
    }
    /// 非音频线程：补充备用缓冲、收走已封口的段、并把"自己停下的捕获"promote 成 take。
    ///
    /// 【为什么由调用方周期性驱动】音频线程不能分配，所以"下一个段用哪个缓冲"必须由
    /// 非音频线程提前备好。这个函数就是那条供给线。
    ///
    /// 【为什么未武装时也要进来】捕获会**自己**停下（到上限、备用缓冲用尽、采样率变化），
    /// 而那些时刻没有任何用户动作。若这里在未武装时直接早退，那些已经录好的段就会
    /// 一直留在捕获里：界面显示"尚未录制"、素材却在内存里，而下一次 `start()` 会把它
    /// 覆盖掉 —— "到点自动停"于是变成"到点丢掉"。自动 promote 是这条的正面回答。
    ///
    /// 【为什么未武装时不补缓冲】补缓冲只服务于"正在录制"这件事：`start()` 已经同步
    /// 备好了第一批。没有这道早退，一个空闲的 Capture 实例会在每一轮心跳里分配
    /// `SPARE_BUFFERS` 粒缓冲（2 × 8 秒立体声 ≈ 6 MB）并永久持有 —— 而它什么都没在录。
    pub(crate) fn service(&self) {
        if !self.is_armed() {
            self.finish_auto_stopped();
            return;
        }
        let options = self.options();
        let rate = self.project_rate.load(Ordering::Acquire);
        if rate <= 0 {
            return;
        }
        let capacity = ((options.segment_sec * rate as f64).round() as usize).max(1);
        // 【T19：分配移到锁外】旧实现在持锁期间分配最多 `SPARE_BUFFERS` 粒
        // 8 秒立体声缓冲（≈ 6 MB，还要清零）。那几毫秒里音频线程的 `try_lock`
        // 全部失败 ⇒ 整块音频被丢 ⇒ 快照里一个洞 ⇒ 播放到那里就是一次"咔"。
        // 现在先在锁外把缺的粒数备好，持锁只做"推进去"。
        let mut prepared = Vec::new();
        {
            let capture = self.capture.lock_or_recover();
            // 捕获已结束（用尽备用缓冲 / 采样率变化 / 用户停止）：不再补，等 promote。
            if capture.is_stopped() {
                return;
            }
            let missing = SPARE_BUFFERS.saturating_sub(capture.spare_count());
            drop(capture);
            prepared.reserve(missing);
            for _ in 0..missing {
                let Some(spare) = allocate_spare(capacity, 2) else {
                    // 预算不足：不静默失败，交给 `status()` 的读数与段结束原因表达。
                    break;
                };
                prepared.push(spare);
            }
        }
        if prepared.is_empty() {
            return;
        }
        let mut capture = self.capture.lock_or_recover();
        if capture.is_stopped() {
            // 备好的粒还没用上，捕获就结束了：直接退还会话（drop 即归还预算额度）。
            return;
        }
        for spare in prepared {
            if capture.spare_count() >= SPARE_BUFFERS {
                break;
            }
            capture.push_spare(spare);
        }
    }

    /// 把"自己停下"的捕获 promote 成一个 take（**只允许非音频线程**：它要取锁与写盘）。
    ///
    /// 返回是否真的 promote 了。用户按的停（`stop()`）不走这里：那条路上
    /// `armed` 是用户自己清掉的，语义不同（用户要的是"这一次录完了"）。
    fn finish_auto_stopped(&self) -> bool {
        let segments = {
            let mut capture = self.capture.lock_or_recover();
            if !capture.is_stopped() {
                return false;
            }
            let reason = capture.stopped_reason().unwrap_or(ClosedReason::Stopped);
            let segments = capture.finish(reason);
            capture.reset_after_promote();
            segments
        };
        self.capture_stream.store(0, Ordering::Release);
        if segments.is_empty() {
            return false;
        }
        if self.promote(segments).is_err() {
            return false;
        }
        log::info!("[capture] a self-stopped capture was promoted into a take");
        true
    }

    /// 一次取锁读出的全部捕获读数（非音频线程）。
    ///
    /// 【为什么合成一次读】录制读数、录制带与两个布尔此前各取一次 `capture` 锁，而音频
    /// 线程在同一把锁上用 `try_lock` 写入 —— 拿不到就**跳过这一块音频**。UI 每 200 ms
    /// 轮询一次，三次取锁就是三次丢块的机会。合成一次把概率降到三分之一，顺带让读数
    /// 天然一致（不会出现"位置来自这一毫秒、分段数来自下一毫秒"这种自相矛盾的快照）。
    fn capture_readings(&self) -> CaptureReadings {
        if !self.is_armed() {
            // 【为什么未武装就不取锁】那 99% 的轮询里没有任何录制在进行，而
            // `capture` 里也不会长出新的东西（`start()` 是唯一的写入点之一）。
            return CaptureReadings::default();
        }
        let capture = self.capture.lock_or_recover();
        let span = capture.span();
        let sample_rate = capture.sample_rate();
        let started = capture.is_started();
        CaptureReadings {
            started,
            limit_warning: capture.limit_warning(),
            // 【为什么待命期没有录制读数】那些读数（已录帧数、峰值、分段数）在宿主播放
            // 之前全是零，而"已录 0.0 秒"正是用户把待命误读成失败的那句话。待命的答案
            // 是"等你去按播放"，不是"已经录了零"。
            recording: started.then(|| RecordingStatus {
                captured_frames: capture.captured_frames(),
                sealed_segments: capture.sealed_len(),
                position: capture.last_position().unwrap_or(0),
                peak: capture.peak(),
                gaps: capture.gaps,
                spares: capture.spare_count(),
            }),
            span: span.map(|span| CaptureSpan {
                start_frame: span.start,
                end_frame: span.end,
                position_frame: capture.last_position().unwrap_or(span.end),
                sample_rate,
            }),
        }
    }

    /// 素材的摘要（跨全部 take 的**投影**结果）。
    ///
    /// 【为什么片段数取投影而不是逐 take 相加】用户看到的是投影后的时间线：被补录盖住
    /// 的部分已经不在那里了。逐 take 相加会报出一个用户看不见的数字。
    pub(crate) fn material_summary(&self) -> Option<MaterialSummary> {
        let takes = self.takes.lock_or_recover();
        if takes.iter().all(|take| take.segments.is_empty()) {
            return None;
        }
        let captured_frames: u64 = takes
            .iter()
            .flat_map(|take| take.segments.iter())
            .map(|segment| segment.filled as u64)
            .sum();
        let sample_rate = takes
            .iter()
            .flat_map(|take| take.segments.iter())
            .map(|segment| segment.sample_rate)
            .max()
            .unwrap_or(0);
        let pre_roll_frames = takes
            .iter()
            .map(|take| take.pre_roll_frames)
            .fold(0_u64, u64::saturating_add);
        let input: Vec<ProjectionTake<'_>> = takes
            .iter()
            .map(|take| ProjectionTake {
                id: take.id,
                segments: &take.segments,
            })
            .collect();
        let (_, summary) = timeline::project(&input, 120.0);
        Some(MaterialSummary {
            clips: summary.clips,
            captured_frames,
            covered_frames: summary.covered_frames,
            spanned_frames: summary.spanned_frames,
            sample_rate,
            pre_roll_frames,
        })
    }

    /// 全部 take 的只读清单（面板的 take 列表与"录制在哪"的定位都用它）。
    pub(crate) fn take_summaries(&self) -> Vec<TakeSummary> {
        let takes = self.takes.lock_or_recover();
        let latest = takes.iter().map(|take| take.id).max();
        takes
            .iter()
            .map(|take| {
                let rate = take.sample_rate.max(1) as f64;
                let span = take.summary.spanned_frames;
                TakeSummary {
                    id: take.id,
                    active: Some(take.id) == latest,
                    clips: take.summary.clips,
                    span_sec: (span > 0).then(|| {
                        let start = take
                            .segments
                            .iter()
                            .map(|segment| segment.origin.max(0))
                            .min()
                            .unwrap_or(0);
                        (start as f64 / rate, (start + span as i64) as f64 / rate)
                    }),
                }
            })
            .collect()
    }

    /// 素材的工程跨度（用于"录完把视口移到录制区间"）。
    ///
    /// 【为什么是全部 take 的并集】用户问的是"我刚录的那段在哪" —— 每一次录制都算。
    /// 只报最近一次会让补录一句之后的定位跳到那个小句子上，而用户想看的是整段成果。
    pub(crate) fn material_span(&self) -> Option<MaterialSpan> {
        self.with_projection(|input| {
            let mut start = i64::MAX;
            let mut end = i64::MIN;
            let mut sample_rate = 0_u32;
            for segment in input.iter().flat_map(|take| take.segments.iter()) {
                if segment.span().is_empty() || segment.has_nonfinite() {
                    continue;
                }
                let segment_start = segment.origin.max(0);
                let segment_end = segment.origin.saturating_add(segment.filled as i64);
                if segment_end <= segment_start {
                    continue;
                }
                start = start.min(segment_start);
                end = end.max(segment_end);
                sample_rate = sample_rate.max(segment.sample_rate);
            }
            (start < end).then_some(MaterialSpan {
                start_frame: start,
                end_frame: end,
                sample_rate,
            })
        })
    }

    /// 捕获侧诊断快照（**try_lock**；非音频线程）。
    ///
    /// 【为什么不用阻塞锁】诊断可能在 actor 心跳里被调用；持有 `capture` 锁会让音频
    /// 线程的 `try_lock` 失败并整块跳过。诊断读不到下一轮再读，不能反过来影响录制。
    pub(crate) fn capture_diagnostics(&self) -> Option<recorder::CaptureDiagnostics> {
        self.capture
            .try_lock()
            .ok()
            .map(|capture| capture.diagnostics())
    }

    /// 对外状态。
    pub(crate) fn status(&self) -> CaptureStatus {
        let armed = self.is_armed();
        let take_count = self.takes.lock_or_recover().len();
        let readings = self.capture_readings();
        // 【为什么"待命"是独立一档】武装之后、宿主播放之前，素材一定还是零 —— 而那与
        // "正在录"是两种不同的等待：前者等用户去按播放，后者在积累素材。合成一句
        // "已录 0.0 秒"会让待命看起来像失败（用户反馈里的"点了录制没反应"）。
        let state = if armed {
            if readings.started {
                "recording"
            } else {
                "standby"
            }
        } else if take_count > 0 {
            "ready"
        } else {
            "empty"
        };
        CaptureStatus {
            state,
            standby: armed && !readings.started,
            recording: readings.recording,
            capture: readings.span,
            material: self.material_summary(),
            has_material: self.has_material(),
            span: self.material_span(),
            takes: self.take_summaries(),
            options: self.options(),
            take_count,
            active_take: self.latest_take(),
            saw_context: self.saw_context.load(Ordering::Acquire),
            saw_input: self.saw_input.load(Ordering::Acquire),
            saw_offline: self.saw_offline.load(Ordering::Acquire),
            host_playing: self.host_playing(),
            unsupported_precision: self.precision_unsupported(),
            limit_warning: readings.limit_warning,
            stopped_reason: ClosedReason::from_code(self.last_closed.load(Ordering::Acquire))
                .filter(|reason| reason.is_notable())
                .map(ClosedReason::as_str),
        }
    }

    /// 全部素材的渲染输入。
    ///
    /// 返回 `(timeline, regions, sources)` —— 与 ARA 路径**同一组类型**，
    /// 因此下游（渲染、导出、快照）完全共用，不需要为 Capture 分叉。
    ///
    /// 【为什么没有可渲染内容时返回 `None`】渲染那条路以"有没有东西可渲染"为判据
    /// （见 `renders()` 与 `capture_prepare_job` 的早退）。空时间线在那里是"还没录"，
    /// 不是"有一条空轨"。
    pub(crate) fn render_input(
        &self,
    ) -> Option<(
        TimelineState,
        Vec<crate::ara::AraPlaybackRegion>,
        HashMap<String, Arc<SourcePcm>>,
    )> {
        self.with_projection(|input| {
            let (timeline, _) = timeline::project(input, 120.0);
            if timeline.clips.is_empty() {
                return None;
            }
            // T14：覆盖起点必须落在**最早被捕获的段起点**上；晚于它就说明前导静音
            // 在投影阶段被丢掉了，播放会在那个位置发生一次源切换。
            note_coverage_start_gap(input, &timeline);
            Some((timeline.clone(), regions_for(&timeline), sources_for(input)))
        })
    }

    /// 会话用的时间线 + 素材（**不**构造渲染区域）。
    ///
    /// 【为什么与 `render_input` 分开】渲染那条路要的是 `AraPlaybackRegion`；编辑器要的
    /// 是"界面该显示什么"。还没有素材（用户还没录）时编辑器必须拿到一条**空的**
    /// 时间线，而不是 `None` —— 否则界面停在"没有时间线"，而不是"还没有内容，请录制"。
    pub(crate) fn session_timeline(&self) -> (TimelineState, HashMap<String, Arc<SourcePcm>>) {
        self.with_projection(|input| {
            let (timeline, _) = timeline::project(input, 120.0);
            (timeline, sources_for(input))
        })
    }

    /// 用**给定的**时间线构造渲染区域。
    ///
    /// 【为什么要能换时间线】编辑器会话持有的那份带着用户曲线（参数就是 Capture 形态下
    /// "处理"的全部内容），而捕获存储那份只有几何。渲染要用前者的曲线、后者的 PCM。
    pub(crate) fn regions_for_timeline(
        &self,
        timeline: &TimelineState,
    ) -> Vec<crate::ara::AraPlaybackRegion> {
        regions_for(timeline)
    }

    /// 全部素材的源 PCM 表（按源 id 索引）。
    pub(crate) fn material_sources(&self) -> HashMap<String, Arc<SourcePcm>> {
        self.with_projection(sources_for)
    }

    /// 素材的**未处理** PCM，做成一份可直接发布的快照（"原始磁带"）。
    ///
    /// 【为什么用 `PlaybackSnapshot` 的形状】处理结果与原始磁带同形，于是音频线程只有
    /// "选哪个发布器"一个分支：不需要第二条读取通路、第二套预算、第二套回收。
    ///
    /// 【为什么铺片这件事在 `timeline` 里】它必须与投影共用同一套覆盖规则（后来的胜出、
    /// 静音片不出现），否则"听到的原始磁带"与"时间轴上看到的"会分叉。
    ///
    /// 【为什么在发布方分配而不是让音频线程给缓冲】发布只在素材变化时发生一次，
    /// 而发布方（UI 线程 / 会话 actor）本来就不是实时线程。
    pub(crate) fn raw_tape(&self) -> Option<PlaybackSnapshot> {
        self.with_projection(timeline::raw_tape)
    }

    /// 把封口的段提升成一个 take。
    fn promote(&self, segments: Vec<Segment>) -> Result<u64, String> {
        let mut capture = self.capture.lock_or_recover();
        let mut all = segments;
        all.extend(capture.drain_sealed());
        // 【为什么在 `reset_after_promote` 之前读】诊断里的前导帧数与暂存计数都在
        // reset 时清零；它们是"这次录制到底拿到了什么证据"的唯一现场。
        let diagnostics = capture.diagnostics();
        capture.reset_after_promote();
        drop(capture);
        self.capture_stream.store(0, Ordering::Release);
        if all.is_empty() {
            return Err("nothing was captured".into());
        }
        let id = self.next_take_id.fetch_add(1, Ordering::AcqRel);
        let sample_rate = all[0].sample_rate;
        // 【为什么这条日志是必需的】round 16 的 T1 让前导静音能进入段首，但没有任何
        // 读数回答"它到底有没有进入"。`pre_roll_frames == 0` 而这次确实走过起播窗口
        // （有块被暂存 / 窗口被桥接）时，快照的覆盖起点就落在**内容起点**而不是
        // 播放起点 —— 播放时会在那个位置发生一次"宿主原声 → 处理结果"的切换，
        // 听感就是用户报的"起播 0.7 s 内的一次咔"。这条日志把那个结论变成可观测事实，
        // 不改任何行为。
        // 【I-1 的读数】这一行是"这段录音里有没有假洞"的唯一权威回答。
        //
        // 四个数各自回答一件事：
        //   * `frames` / `rate`：素材有多长（换算成秒才对得上用户听到的时间）；
        //   * `coverage`：我们真的拷到样本的区间数；`gaps` 是它们之间的洞；
        //   * `lead_in`：第一个属于我们的帧在带内的偏移（**必须**是 0 或只剩开头的
        //     真实缺失）；
        //   * `migrated_from_v4`：这份 take 是从带 `trusted` 表的旧 sidecar 读回来的。
        //
        // 【为什么把 `lead_in` 与 `first_signal_at` 放在同一行】两者接近 ⇒ 覆盖表是对的
        // （那一段本来就不是我们的）；`lead_in` 明显大于 `first_signal_at` ⇒ 我们把一段
        // **真实录音**声明成了洞 —— 播放侧于是回退宿主原声（参数线不生效）并在边界处
        // 咔一下。round 23 就是后者（0.658 s），而这条日志当时写的是 `INFO`，在 release
        // 的默认 `Warn` 级别下**根本看不到**。
        let lead_in = timeline::lead_in(&all);
        let rate = sample_rate.max(1) as f64;
        let (coverage_ranges, coverage_gaps) = coverage_shape(&all);
        // 【假洞】`lead_in` 之前若存在非零样点，说明我们把一段**真实录音**声明成了洞。
        // 播放侧会在那里回退宿主原声（参数线不生效）并在边界咔一下 —— 那正是 round 23
        // 的形状。这条读数是 ERROR 级：它一旦为真，问题一定在覆盖表，不在别处。
        if lead_in.falsified {
            log::error!(
                "[capture] take {id}: FALSIFIED LEAD-IN — coverage claims the first {} frame(s) \
                 are not ours, but the first non-zero sample is at {} ({:.3}s). Playback will fall \
                 back to host audio there and switch sources at the boundary (that is the click)",
                lead_in.synthetic_frames,
                lead_in.first_signal_sec.unwrap_or(0.0) * rate,
                lead_in.first_signal_sec.unwrap_or(0.0),
            );
        }
        log::warn!(
            "[capture] take {id}: frames={} ({:.3}s) rate={} coverage={} gaps={} \
             lead_in={} ({:.3}s) first_signal_at={} pre_roll_frames={} pending_stashed={} \
             bridged_window_blocks={} resync_bridged_frames={} counted_blocks={} \
             counted_frames={} migrated_from_v4={} falsified_lead_in={}",
            all.iter().map(|segment| segment.filled as u64).sum::<u64>(),
            all.iter().map(|segment| segment.filled as u64).sum::<u64>() as f64 / rate,
            sample_rate,
            coverage_ranges,
            coverage_gaps,
            lead_in.synthetic_frames,
            lead_in.synthetic_frames as f64 / rate,
            lead_in
                .first_signal_sec
                .map(|sec| format!("{sec:.3}s"))
                .unwrap_or_else(|| "none".to_owned()),
            diagnostics.pre_roll_frames,
            diagnostics.pending_stashed,
            diagnostics.bridged_window_blocks,
            diagnostics.resync_bridged_frames,
            diagnostics.counted_blocks,
            diagnostics.counted_frames,
            u8::from(all.iter().any(|segment| segment.migrated_from_v4)),
            u8::from(lead_in.falsified),
        );
        if diagnostics.pre_roll_frames == 0
            && (diagnostics.pending_stashed > 0 || diagnostics.bridged_window_blocks > 0)
        {
            log::warn!(
                "[capture] take {id}: the start window was observed ({} stashed block(s), \
                 {} bridged window block(s)) but no silent lead-in was recorded; \
                 the snapshot starts at the content start, not at the playback start",
                diagnostics.pending_stashed,
                diagnostics.bridged_window_blocks
            );
        }
        let (_, summary) = timeline::build(&all, id, 120.0);
        // 【为什么在 promote 时就落盘，而不是等 `getState`】落盘是**一次**几十 MB 的
        // 写；放在 `getState` 里就等于在宿主每次自动保存时都写一遍。而 promote 是
        // 用户刚做完的动作，等它有明确的语义（"我刚录完"），且只发生一次。
        let sidecar = sidecar::default_dir().and_then(|dir| {
            match sidecar::write(&dir, id, &all) {
                Ok(path) => Some(path),
                Err(error) => {
                    // 落盘失败不该丢掉这次录制：内存里那份仍然可用，只是重启后不在。
                    log::warn!("[capture] take {id} was not persisted: {error}");
                    None
                }
            }
        });
        let take = CaptureTake {
            id,
            sample_rate,
            segments: all,
            summary,
            pre_roll_frames: diagnostics.pre_roll_frames,
            sidecar,
        };
        let mut takes = self.takes.lock_or_recover();
        // 【为什么保留旧 take 而不是覆盖】用户要能在两次录制之间比较。内存由
        // `max_duration_sec` 与全局预算共同约束；显式清理由 `clear()` 承担。
        takes.push(take);
        drop(takes);
        // 【为什么要推进代次】新增素材 ⇒ 时间线、原始磁带与合成结果全都要重来。
        self.refresh_material_present();
        self.bump_revision();
        Ok(id)
    }
}

/// 一次取锁读出的全部捕获读数（见 [`CaptureStore::capture_readings`]）。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct CaptureReadings {
    /// 宿主是否已经播过至少一块（`false` = 待命）。
    started: bool,
    /// 距单次录制上限不足 10 秒。
    limit_warning: bool,
    /// 录制中读数（未武装时 `None`）。
    recording: Option<RecordingStatus>,
    /// 录制中正在长出来的工程区间。
    span: Option<CaptureSpan>,
}

/// 本块在 Capture 通路上的结论。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CaptureBlockOutcome {
    /// 没在录制。
    Idle,
    /// 已写入。
    Capturing,
    /// 捕获状态被非音频线程占着：本块未写入。
    Busy,
    /// 宿主没给可用的工程位置。
    NoPosition,
    /// 宿主此刻没在播放：本块未写入（待命或暂停）。
    NotPlaying,
    /// 同一流把同一个工程区间重复投递：不产生新素材。
    Repeated,
    /// 捕获已结束（等 promote）。
    Stopped,
}

impl CaptureBlockOutcome {
    /// 稳定的编码值（只用于诊断）。**只增不改**。
    pub(crate) fn code(self) -> u8 {
        match self {
            CaptureBlockOutcome::Idle => 0,
            CaptureBlockOutcome::Capturing => 1,
            CaptureBlockOutcome::Busy => 2,
            CaptureBlockOutcome::NoPosition => 3,
            CaptureBlockOutcome::NotPlaying => 4,
            CaptureBlockOutcome::Stopped => 5,
            CaptureBlockOutcome::Repeated => 6,
        }
    }
}

/// 分配一个空的段缓冲，并计入全局 PCM 预算。
///
/// 【为什么预留失败就返回 `None` 而不是分配】预算耗尽时分配会让"内存超限"变成
/// 一次 OOM 崩溃，而不是一次可解释的"录制停止"。
fn allocate_spare(capacity: usize, channels: u8) -> Option<Segment> {
    let bytes = capacity.checked_mul(channels as usize)?.checked_mul(4)?;
    let reservation = crate::render::budget::global_budget().reserve(bytes)?;
    Some(Segment {
        origin: 0,
        sample_rate: 0,
        planes: vec![vec![0.0; capacity]; channels as usize],
        filled: 0,
        loss: recorder::SegmentLoss::First,
        loop_start: None,
        peak: [0.0; 2],
        closed: ClosedReason::Full,
        coverage: std::array::from_fn(|_| 0..0),
        coverage_len: 0,
        coverage_overrun: false,
        migrated_from_v4: false,
        _reservation: Some(reservation),
    })
}

/// take 的段 → 源 PCM。**一个 take 一份源**（键是 [`timeline::source_id`]）。
///
/// 【为什么不是一段一份】片段可以跨段（见 `timeline.rs` 的模块头），而一个片段只能引用
/// 一份源与一个连续源窗 —— 段一份源会让跨段片段无法表达。源域因此是**素材带**：
/// 段按录制顺序首尾相接，带内偏移就是源窗的坐标系。
///
/// 【为什么在这里拼而不是让每个消费方各拼一遍】源表是渲染与波形的共同输入，
/// 而"带 = 段的顺序拼接"这条规则只能有一处实现 —— 两处实现必然漂移，
/// 表现为"听到的与看到的不是同一段音频"。
fn sources_for(takes: &[ProjectionTake<'_>]) -> HashMap<String, Arc<SourcePcm>> {
    let (tapes, _) = timeline::build_tapes(takes);
    tapes
        .iter()
        .map(|tape| {
            (
                timeline::source_id(tape.take),
                Arc::new(SourcePcm {
                    // 【T-S1-1：覆盖跟着源一起走】`concat_planes` 把"我们没拿到样本"
                    // 的帧写成数字零；不带覆盖的话，下游无法把那段零与"用户真的安静"
                    // 分开，于是它按满幅输出、盖掉宿主原声 —— 洞的两侧各是一次硬切。
                    covered: tape.covered_ranges(),
                    sample_rate: tape.sample_rate,
                    planes: tape.concat_planes(),
                    version: tape.take,
                    _reservation: None,
                }),
            )
        })
        .collect()
}

/// 一次录制的覆盖形状：`(区间数, 洞数)`。
///
/// 【为什么值得单独一个函数】`[capture] take` 行要回答的核心问题是"这段录音里有没有
/// 我们**没录到**的帧"。区间数 1 且洞数 0 是健康形状；区间数 > 1 说明素材中间有洞，
/// 而洞的两侧各是一次"宿主原声 ↔ 处理结果"的切换 —— 那正是"播放途中的随机咔"。
///
/// 【与 `lead_in` 的分工】`lead_in` 只看**第一个**区间的起点（"开头缺了多少"），本函数
/// 只看**中间**有几个洞（"中途断了几次"）。两者一起才能区分"起播窗口那一次"与
/// "每次 resync 各一次"。
fn coverage_shape(segments: &[recorder::Segment]) -> (usize, usize) {
    let mut ranges = 0_usize;
    let mut gaps = 0_usize;
    for segment in segments {
        if segment.coverage_overrun || segment.coverage_len == 0 {
            // 没有记录（旧档 / 槽位用尽）= 整段都算录到：一个区间、零个洞。
            ranges += 1;
            continue;
        }
        let len = segment.coverage_len.min(segment.coverage.len());
        ranges += len;
        // 段内相邻区间之间的空档，加上"段首之前 / 段尾之后"两段（那两段同样是洞）。
        let mut previous_end = 0_usize;
        for range in &segment.coverage[..len] {
            if range.start > previous_end {
                gaps += 1;
            }
            previous_end = range.end;
        }
        if previous_end < segment.filled {
            gaps += 1;
        }
    }
    (ranges, gaps)
}
/// 投影后的覆盖起点若晚于**最早被捕获的段起点**，说明前导静音在投影阶段被丢掉了。
///
/// 【为什么这条日志值得存在】快照的 `origin_sample` 直接取自最早 clip 的起点
/// （`render/input.rs`）。一旦它晚于最早的段起点，播放时就会在那个位置从"宿主原声"
/// 切到"处理结果" —— 用户听到的就是一次"咔"。这条日志把"覆盖起点悬空"从
/// 一句无法归因的现象变成可测量的差值（T14）。它只读事实，不改变任何行为。
fn note_coverage_start_gap(takes: &[ProjectionTake<'_>], timeline: &TimelineState) {
    let earliest_segment = takes
        .iter()
        .flat_map(|take| take.segments.iter())
        .filter(|segment| !segment.span().is_empty() && !segment.has_nonfinite())
        .map(|segment| segment.origin.max(0))
        .min();
    let Some(segment_start) = earliest_segment else {
        return;
    };
    let rate = takes
        .iter()
        .flat_map(|take| take.segments.iter())
        .map(|segment| segment.sample_rate)
        .max()
        .unwrap_or(0);
    if rate == 0 {
        return;
    }
    let Some(clip_start_sec) = timeline
        .clips
        .iter()
        .map(|clip| clip.start_sec)
        .reduce(f64::min)
    else {
        return;
    };
    let clip_start = (clip_start_sec * rate as f64).round() as i64;
    // 一个块的量级（20 ms）。小于它属于块内对齐误差，不值得报。
    let grace = (rate as i64 / 50).max(1);
    if clip_start > segment_start.saturating_add(grace) {
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            "capture.coverage-start-gap",
            format_args!(
                "[capture] the earliest captured segment starts at frame {segment_start} but \
                 the published coverage starts at {clip_start} ({} frames later); the silent \
                 lead-in was dropped during projection, so playback will switch sources there",
                clip_start.saturating_sub(segment_start)
            ),
        );
    }
}

/// 由时间线合成 ARA 形状的播放区间。
///
/// 【为什么复用 `AraPlaybackRegion`】渲染入口（`RenderInput`）的几何校验与帧数计算都
/// 以它为准。Capture 的片段与它是 1:1 的（1:1 映射、无拉伸、无淡化），因此合成出来
/// 的区间就是精确描述 —— 于是**整条下游渲染路径一行都不用改**。
fn regions_for(timeline: &TimelineState) -> Vec<crate::ara::AraPlaybackRegion> {
    timeline
        .clips
        .iter()
        .filter_map(|clip| {
            let take = clip.active_take();
            let source = take.source_path.clone()?;
            Some(crate::ara::AraPlaybackRegion {
                name: Some(clip.name.clone()),
                audio_source_persistent_id: source,
                audio_modification_persistent_id: clip.id.clone(),
                region_sequence_index: Some(0),
                // Capture 的时间线只有一个合成序列，没有宿主句柄可归属。
                sequence_model_ref: None,
                start_in_modification_time: take.source_start_sec,
                duration_in_modification_time: take.source_end_sec - take.source_start_sec,
                start_in_playback_time: clip.start_sec,
                duration_in_playback_time: clip.length_sec,
                is_timestretch_enabled: false,
                has_content_based_fade_at_head: false,
                has_content_based_fade_at_tail: false,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> CaptureStore {
        let store = CaptureStore::default();
        store.note_project_rate(48_000.0);
        store
    }

    /// **T19: a block deferred by lock contention is replayed into the take.**
    ///
    /// Why this is the third main regression for symptom 1: while a non-audio thread
    /// holds the capture lock (refilling spares, collecting segments, UI reads), the
    /// audio thread's try_lock fails. The old code dropped the block outright, so
    /// expected_next no longer matched and the projection left a hole in the snapshot -
    /// audible as music disappearing and then restarting.
    #[test]
    fn a_block_deferred_by_lock_contention_is_replayed_into_the_take() {
        let store = store();
        store.start().unwrap();
        let samples: &'static [f32] = Box::leak(vec![0.4_f32; 512].into_boxed_slice());
        // Hold the lock: from here on, on_block's try_lock must fail.
        let guard = store.capture.lock_or_recover();
        store.on_block(InputBlock {
            start: 0,
            frames: 512,
            sample_rate: 48_000.0,
            planes: [Some(samples), Some(samples)],
            playing: true,
        });
        assert_eq!(store.bypass_pending(), 1, "the block must be deferred");
        drop(guard);
        // Feed the next block: the deferred one must be replayed first, then this.
        store.on_block(InputBlock {
            start: 512,
            frames: 512,
            sample_rate: 48_000.0,
            planes: [Some(samples), Some(samples)],
            playing: true,
        });
        assert_eq!(store.bypass_pending(), 0, "the slot must be empty after replay");
        assert_eq!(store.bypass_counters(), (0, 1), "the block must be saved, not lost");
        let _ = store.stop();
        let (timeline, _, _) = store.render_input().expect("has material");
        // Two adjacent blocks make one clip. If the first had been dropped there would
        // be two clips and a hole between them.
        assert_eq!(
            timeline.clips.len(),
            1,
            "a deferred block must be replayed; dropping it leaves a hole in the snapshot"
        );
    }

    /// **T19: an overflowing bypass counts the drop instead of losing it silently.**
    #[test]
    fn an_overflowing_bypass_counts_the_drop_instead_of_losing_it_silently() {
        let store = store();
        store.start().unwrap();
        let samples: &'static [f32] = Box::leak(vec![0.4_f32; 512].into_boxed_slice());
        let guard = store.capture.lock_or_recover();
        for index in 0..(BYPASS_SLOTS + 2) as i64 {
            store.on_block(InputBlock {
                start: index * 512,
                frames: 512,
                sample_rate: 48_000.0,
                planes: [Some(samples), Some(samples)],
                playing: true,
            });
        }
        assert_eq!(store.bypass_pending(), BYPASS_SLOTS, "slots are full");
        let (dropped, _) = store.bypass_counters();
        assert_eq!(dropped, 2, "the overflow must be counted");
        drop(guard);
    }
    /// 没声明工程率时不能开始录制：段容量无从确定。
    #[test]
    fn arming_requires_a_declared_project_rate() {
        let store = CaptureStore::default();
        assert!(store.start().is_err());
        store.note_project_rate(48_000.0);
        assert!(store.start().is_ok());
    }

    /// 录制 → 停止 → 产生一个带片段的 take。
    #[test]
    fn a_capture_round_trip_produces_a_take_with_clips() {
        let store = store();
        store.start().unwrap();
        let samples: &'static [f32] = Box::leak(vec![0.4_f32; 4096].into_boxed_slice());
        for index in 0..4_i64 {
            store.on_block(InputBlock {
                start: index * 4096,
                frames: 4096,
                sample_rate: 48_000.0,
                planes: [Some(samples), Some(samples)],
                playing: true,
            });
        }
        let id = store.stop().unwrap();
        let status = store.status();
        assert_eq!(status.state, "ready");
        assert_eq!(status.active_take, Some(id));
        assert_eq!(status.take_count, 1);
        let (timeline, regions, sources) = store.render_input().expect("有素材");
        assert_eq!(timeline.clips.len(), 1, "连续块应合成一个片段");
        assert_eq!(regions.len(), 1);
        assert_eq!(sources.len(), 1);
        assert!(store.has_material());
    }

    /// **一次全是静音的录制仍然是素材。**
    ///
    /// 【为什么这条必须存在】投影会把整片全静音的部分裁掉（`timeline::project` 的
    /// `is_silent` 分支），所以一次全静音的录制 `summary.clips == 0`。而 `has_material`
    /// 此前正是按 `clips > 0` 判的 —— 于是"插件没被接到这条轨的音频上"这种最常见的
    /// 失败会被算成"没有素材"：输出仲裁跳过原始磁带那一档、直接落到**透传宿主原声**。
    /// 那听起来是有声音的，于是"编辑没生效"反而比静音更难发现。
    ///
    /// 素材的存在由"有没有 take"回答，不是由"投影后还剩几条片段"回答。
    #[test]
    fn an_all_silent_recording_is_still_material() {
        let store = store();
        store.start().unwrap();
        let silence: &'static [f32] = Box::leak(vec![0.0_f32; 4096].into_boxed_slice());
        for index in 0..2_i64 {
            store.on_block(InputBlock {
                start: index * 4096,
                frames: 4096,
                sample_rate: 48_000.0,
                planes: [Some(silence), Some(silence)],
                playing: true,
            });
        }
        store.stop().unwrap();
        assert_eq!(store.status().take_count, 1, "录到了一段时间");
        assert_eq!(
            store.takes.lock_or_recover()[0].summary.clips,
            0,
            "前提：投影确实把它全裁掉了（这就是此前的失效现场）"
        );
        assert!(
            store.has_material(),
            "有 take 就有素材 —— 否则静音录制后仲裁会掉到透传宿主原声"
        );
        assert!(
            store.raw_tape().is_some(),
            "原始磁带必须建得出来（全零，但语义正确）"
        );
    }

    /// 取消录制不留 take。
    #[test]
    fn cancelling_leaves_no_take() {
        let store = store();
        store.start().unwrap();
        let samples: &'static [f32] = Box::leak(vec![0.4_f32; 1024].into_boxed_slice());
        store.on_block(InputBlock {
            start: 0,
            frames: 1024,
            sample_rate: 48_000.0,
            planes: [Some(samples), Some(samples)],
            playing: true,
        });
        store.cancel();
        assert_eq!(store.status().take_count, 0);
        assert!(!store.has_material());
    }

    /// 观察位如实反映"宿主给过什么"，而且**不需要先武装**。
    ///
    /// 【为什么"不需要武装"是这条测试的重点】面板的录制按钮由 `canCapture`
    /// （= `saw_context`）门控，而观察此前发生在武装之后 —— 于是"必须先录一次才能录"，
    /// 按钮永远是灰的，Capture 在**所有**宿主上都用不了。观察必须与武装无关。
    #[test]
    fn observation_bits_record_what_the_host_did_without_being_armed() {
        let store = store();
        let status = store.status();
        assert!(!status.saw_context && !status.saw_input && !status.saw_offline);
        assert!(!status.host_playing);
        store.note_offline();
        let samples: &'static [f32] = Box::leak(vec![0.1_f32; 256].into_boxed_slice());
        // 注意：**没有** `store.start()` —— 这正是回归点。
        store.observe(&InputBlock {
            start: 0,
            frames: 256,
            sample_rate: 48_000.0,
            planes: [Some(samples), None],
            playing: true,
        });
        let status = store.status();
        assert!(
            status.saw_context,
            "见过 ProcessContext 就必须记下来，否则录制按钮永远点不动"
        );
        assert!(status.saw_input && status.saw_offline);
        assert!(status.host_playing, "播放状态是实时读数");
        assert!(!store.is_armed(), "观察不该顺手把录制武装起来");
    }

    /// 播放状态是**实时**读数：停止之后必须回落，否则面板会一直说"宿主在播放"。
    #[test]
    fn the_playing_reading_follows_the_host_instead_of_latching() {
        let store = store();
        let samples: &'static [f32] = Box::leak(vec![0.1_f32; 16].into_boxed_slice());
        let block = InputBlock {
            start: 0,
            frames: 16,
            sample_rate: 48_000.0,
            planes: [Some(samples), Some(samples)],
            playing: true,
        };
        store.observe(&block);
        assert!(store.status().host_playing);
        store.observe(&InputBlock {
            playing: false,
            ..block
        });
        assert!(!store.status().host_playing, "停止播放必须如实回落");
        // 而"见过上下文"是**单调**的：它描述历史，不描述此刻。
        assert!(store.status().saw_context);
    }

    /// **录完之后仍武装、宿主又开始播放 ⇒ 不再算"录制中"。**
    ///
    /// 【为什么这条是"永远直通"的一条根因】`is_recording()` 此前用
    /// `capture_started || host_playing`。录完不按停止是常态，而宿主一播放
    /// `host_playing` 立刻为真 ⇒ 整次播放都判成"录制中" ⇒ 输出仲裁走直通，
    /// **压过已经合成好的处理结果**。用户听到的一直是宿主原声。
    ///
    /// 【为什么 in-place 保护没丢】只在捕获真的往素材里写时才需要直通，而那正是
    /// `capture_started` 的语义（`on_block` 在 `!playing` 时一个样本都不写）。
    ///
    /// **录完并停止之后，宿主再播放不再是"录制中"。**
    ///
    /// 【为什么钉这一条】"处理结果被直通压过"的一个怀疑对象是"录完没按停止 ⇒
    /// `armed` 留着 ⇒ 下次播放整段被判成录制中"。实测不成立：`stop()` 会清 `armed`
    /// （`cancel` / `clear` / 捕获自己停下时同样），所以那个状态根本不存在。
    /// 把结论钉成用例，免得以后有人再按这个假设去改 `is_recording`。
    #[test]
    fn stopping_recording_releases_the_output_to_the_result_again() {
        let store = store();
        let samples: &'static [f32] = Box::leak(vec![0.25_f32; 4096].into_boxed_slice());
        store.start().unwrap();
        store.on_block(InputBlock {
            start: 0,
            frames: 4096,
            sample_rate: 48_000.0,
            planes: [Some(samples), Some(samples)],
            playing: true,
        });
        assert!(store.is_recording(), "真的在写 ⇒ 录制中");
        store.stop().unwrap();
        // 宿主仍在播（用户录完直接继续播放试听）。
        store.observe(&InputBlock {
            start: 999_999,
            frames: 16,
            sample_rate: 48_000.0,
            planes: [Some(samples), Some(samples)],
            playing: true,
        });
        assert!(store.host_playing(), "前提：宿主确实在播");
        assert!(
            !store.is_recording(),
            "停止之后必须把播放权交还给处理结果 —— armed 已被 `stop()` 清掉"
        );
    }

    /// 未录制时 `on_block` 完全不动状态（音频线程上的零成本早退）。
    #[test]
    fn an_unarmed_store_does_nothing_on_a_block() {
        let store = store();
        let samples: &'static [f32] = Box::leak(vec![0.1_f32; 256].into_boxed_slice());
        assert_eq!(
            store.on_block(InputBlock {
                start: 0,
                frames: 256,
                sample_rate: 48_000.0,
                planes: [Some(samples), None],
                playing: true,
            }),
            CaptureBlockOutcome::Idle
        );
        assert_eq!(store.status().take_count, 0);
    }

    /// 选项被夹到合法区间：一个笔误不该变成 0 秒或 1e9 秒。
    ///
    /// 【为什么 0 要单独断言】`0` 是"不限"的哨兵值，而负数必须退回最小值 ——
    /// 两者混起来会让一个笔误悄悄把上限关掉（"不限"= 进程预算的敌人）。
    #[test]
    fn options_are_clamped_to_a_usable_range() {
        let store = CaptureStore::default();
        store.set_options(CaptureOptions {
            max_duration_sec: -5.0,
            segment_sec: 1e9,
            monitor: false,
            outside_passthrough: false,
        });
        let options = store.options();
        assert!(
            options.max_duration_sec >= 1.0,
            "负数（笔误）必须退回最小值，不能变成 0 = 不限"
        );
        assert!(options.segment_sec <= 60.0);
        assert!(!options.monitor);
        assert!(!options.outside_passthrough);
        // 恰好 0 = 不限：它是用户的真实选择，必须被保留。
        store.set_options(CaptureOptions {
            max_duration_sec: 0.0,
            ..options
        });
        assert_eq!(
            store.options().max_duration_sec,
            0.0,
            "0 必须原样保留为不限"
        );
        // 超大的值仍被夹到上限。
        store.set_options(CaptureOptions {
            max_duration_sec: 1e9,
            ..options
        });
        assert!(store.options().max_duration_sec <= 3600.0);
    }

    /// 临时目录（每个用例一个）。
    fn temp_dir() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let seq = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("hfs-xfer-{}-{seq}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn captured_segment() -> Segment {
        Segment {
            origin: 0,
            sample_rate: 48_000,
            planes: vec![vec![0.4_f32; 512], vec![0.4_f32; 512]],
            filled: 512,
            loss: recorder::SegmentLoss::First,
            loop_start: None,
            peak: [0.4; 2],
            closed: ClosedReason::Stopped,
            coverage: std::array::from_fn(|_| 0..0),
            coverage_len: 0,
            coverage_overrun: false,
            migrated_from_v4: false,
            _reservation: None,
        }
    }

    /// **重启后录音仍在**：清单指向 sidecar，恢复后能直接渲染。
    ///
    /// 【为什么这是持久化的唯一验收】落盘的价值全在"下次打开还在"。只测"写了文件"
    /// 是不够的 —— 恢复路径要能把段读回来、重建成时间线、并让 `has_material` 为真，
    /// 否则用户看到的是一个空面板加一个存在于磁盘上的文件。
    #[test]
    fn a_manifest_restores_takes_from_their_sidecars() {
        let dir = temp_dir();
        let path = sidecar::write(&dir, 5, &[captured_segment()]).unwrap();
        let manifest = serde_json::json!({
            "version": 1,
            "takes": [{ "id": 5, "sidecar": path.to_string_lossy(), "active": true }],
        });
        let store = CaptureStore::default();
        store.note_project_rate(48_000.0);
        store.restore_manifest(&manifest).unwrap();
        assert_eq!(store.status().take_count, 1);
        assert_eq!(store.latest_take(), Some(5));
        assert!(store.has_material(), "恢复之后必须立刻可渲染");
        let (timeline, regions, sources) = store.render_input().expect("恢复的 take 应当可用");
        assert_eq!(timeline.clips.len(), 1);
        assert_eq!(regions.len(), 1);
        assert_eq!(sources.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 一个 sidecar 不见了**只跳过它**，其余 take 照常恢复。
    ///
    /// 【为什么逐条降级】用户可能清了缓存、或把工程拷到了另一台机器。一条缺失不该
    /// 让整份清单失败 —— 那会把"少了一条录音"变成"所有录音都没了"。
    #[test]
    fn a_missing_sidecar_is_skipped_without_losing_the_other_takes() {
        let dir = temp_dir();
        let present = sidecar::write(&dir, 1, &[captured_segment()]).unwrap();
        let manifest = serde_json::json!({
            "version": 1,
            "takes": [
                { "id": 1, "sidecar": present.to_string_lossy(), "active": true },
                { "id": 2, "sidecar": dir.join("gone.hfx").to_string_lossy(), "active": false },
            ],
        });
        let store = CaptureStore::default();
        store.restore_manifest(&manifest).unwrap();
        assert_eq!(store.status().take_count, 1, "只有存在的那个 take 被恢复");
        assert_eq!(store.latest_take(), Some(1));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 恢复后新录的 take 不能复用已存在的号（否则会覆盖一个已有的 sidecar 文件）。
    #[test]
    fn restoring_bumps_the_take_id_allocator_past_the_restored_ids() {
        let dir = temp_dir();
        let path = sidecar::write(&dir, 9, &[captured_segment()]).unwrap();
        let manifest = serde_json::json!({
            "version": 1,
            "takes": [{ "id": 9, "sidecar": path.to_string_lossy(), "active": true }],
        });
        let store = CaptureStore::default();
        store.note_project_rate(48_000.0);
        store.restore_manifest(&manifest).unwrap();
        store.start().unwrap();
        let samples: &'static [f32] = Box::leak(vec![0.3_f32; 256].into_boxed_slice());
        store.on_block(InputBlock {
            start: 0,
            frames: 256,
            sample_rate: 48_000.0,
            planes: [Some(samples), Some(samples)],
            playing: true,
        });
        let id = store.stop().unwrap();
        assert!(id > 9, "新 take 号必须跳过已恢复的 9，实际 {id}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 清单为空或形状不对时不报错（老工程里根本没有这一段）。
    #[test]
    fn an_empty_or_absent_manifest_is_not_an_error() {
        let store = CaptureStore::default();
        assert!(store.restore_manifest(&serde_json::json!({})).is_ok());
        assert!(store
            .restore_manifest(&serde_json::json!({ "version": 1, "takes": [] }))
            .is_ok());
        assert_eq!(store.status().take_count, 0);
    }

    /// 合成的播放区间必须与片段的源窗自洽（否则下游几何校验会拒绝整份）。
    #[test]
    fn synthesized_regions_describe_the_clips_exactly() {
        let store = store();
        store.start().unwrap();
        let samples: &'static [f32] = Box::leak(vec![0.3_f32; 2048].into_boxed_slice());
        for index in 0..3_i64 {
            store.on_block(InputBlock {
                start: index * 2048,
                frames: 2048,
                sample_rate: 48_000.0,
                planes: [Some(samples), Some(samples)],
                playing: true,
            });
        }
        store.stop().unwrap();
        let (timeline, regions, sources) = store.render_input().unwrap();
        assert_eq!(regions.len(), timeline.clips.len());
        for (region, clip) in regions.iter().zip(&timeline.clips) {
            assert!((region.start_in_playback_time - clip.start_sec).abs() < 1e-9);
            assert!((region.duration_in_playback_time - clip.length_sec).abs() < 1e-9);
            // 1:1：修改域长度 == 播放域长度。
            assert!(
                (region.duration_in_modification_time - region.duration_in_playback_time).abs()
                    < 1e-9
            );
            assert!(
                sources.contains_key(&region.audio_source_persistent_id),
                "区间引用的源必须存在，否则渲染整份失败"
            );
        }
    }
}
