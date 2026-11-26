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

/// 这一遍是**渲染**还是**实时监听**。
///
/// 【为什么不能只看 `kOffline`】VEGAS 的 Render As **不置** `kOffline`：现场整段渲染
/// 里 `setupProcessing mode=0`、`stream=realtime`，一次 `mode=2` 都没有。于是
/// "离线"这一位在渲染那一趟是假的，而它决定的输出是"直通 vs 静音" ——
/// 直通会把**输入原声**写进产物，用户会把它当成"处理后的结果"。
///
/// 【为什么这三个判据是"可观测证据"而不是猜测】三条都是宿主行为的直接读数：
///
///   * **喂入速度**：单位墙钟时间内喂进来的帧数 / 采样率。现场一次渲染是
///     1 520 704 帧 / 365 ms ≈ **4 165 倍实时**；实时播放恒为 1×。
///   * **喂入长度**：`kPlaying` 全程为 0、而这条喂入流已经超过起点窗口（1 s）。
///     实时监听不会有这个形状（真播放会升起 `kPlaying`）。
///   * **宿主置位**：`is_offline` / `setupProcessMode == 2`（愿意说真话的宿主）。
///
/// 三条任一成立即判渲染，且**整遍不变**（与 `render_run` 同一生命周期）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum CapturePassKind {
    #[default]
    Realtime,
    Render,
}

impl CapturePassKind {
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::Realtime => 0,
            Self::Render => 1,
        }
    }

    pub(crate) fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Render,
            _ => Self::Realtime,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Realtime => "realtime",
            Self::Render => "render",
        }
    }
}

/// 判"这一遍是渲染"的喂入速度阈值（倍实时）。
///
/// 【为什么是 4】实时播放的抖动不会超过 2×（宿主缓冲 + 一次 seek 的追赶）；4× 已经
/// 只可能来自"不按墙钟喂"的渲染。现场实测 4 165×，与 4 之间有三个数量级的余量。
pub(crate) const RENDER_SPEED_X: f64 = 4.0;

/// 速度判据的采样窗口（毫秒）。太短会被单块抖动放大，太长会让判据迟迟不生效。
const FEED_SPEED_WINDOW_MS: u64 = 250;

/// 判"这一遍是渲染"的喂入长度（秒）。见 `CapturePassKind` 的第二条判据。
const RENDER_LENGTH_SEC: f64 = 1.0;

/// 这一组判据的行为开关（环境变量 `HFS_PASS_KIND=off` 关闭）。
///
/// 【为什么要有】`CapturePassKind` 决定"录制中的输出是直通还是静音"。判据有三条、
/// 全部来自宿主读数，但"某个宿主的块序列恰好长得像渲染"这种误判在离线环境里无法
/// 穷举。留一个一键退回旧判据（只看 `kOffline`）的开关，与 `HFS_PINNED_FEED` /
/// `HFS_RENDER_RUN` 同一条纪律。
pub(crate) fn pass_kind_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var("HFS_PASS_KIND").is_ok_and(|value| value == "0" || value == "off")
    })
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

/// 渲染 run 的空闲上界（毫秒）：上一次 pass 之后静默超过这个时长，就认定宿主已经
/// 不再做同一件事，收尾并**丢弃**可延续的尾巴。
///
/// 【为什么不再有"间隔上界"】旧实现用 2 秒的墙钟把 pass 隔开，理由是"VEGAS 的 chunk
/// 之间只隔几十毫秒"。现场证据推翻了它：chunk 之间是 **4 秒**（磁盘写入 + 宿主自己的
/// 重排），于是同一次渲染的每个 chunk 都被判成新 run、游标回到 0，输出就成了"前 0~1 s
/// 反复重放"。墙钟在这里本来就是错的判据 —— 位置信息一直都在（见
/// `render_run_continues`），只有"完全没有位置可依"时才需要时间兜底。
///
/// 30 秒的选取：一次 30 s 工程的 Render As 里，chunk 之间的间隔是秒级；而"用户停手、
/// 想一想、再按一次渲染"通常远长于 30 秒。真正需要"立刻断开"的场合（播放、编辑、
/// 用户停止）都有各自的显式信号，不依赖这个上界。
pub(crate) const RENDER_RUN_IDLE_MS: u64 = 30_000;

/// 一次 run 最多容纳多少个 pass。
///
/// 【为什么需要上界】续算的判据是"宿主位置连续"，而**同一段**渲染两次时第二次的首块
/// 位置也可能与上一次连续（尤其是被钉住的位置）。上界让"判据失效"退化成一枚可观测的
/// 兜底：超过就开新 run 并记一条 Warn，而不是无限接下去。
pub(crate) const RENDER_RUN_MAX_PASSES: u32 = 4_096;

/// 相邻 pass 的宿主位置差容差（以**块**为单位）。见 R-run-1。
pub(crate) const RENDER_RUN_BLOCK_TOLERANCE: i64 = 2;

/// 上一段渲染 run 的尾巴：判"新 pass 是否延续它"的全部输入。
#[derive(Clone, Copy, Debug)]
pub(crate) struct RenderRunTail {
    /// 上一个 pass 最后一块的宿主位置。
    ///
    /// 【为什么它能代表整个 pass】VEGAS 的冻结预览在一个 chunk 内把
    /// `projectTimeSamples` 钉死（位置全在本地游标里推进），所以"最后一块的位置"
    /// 与"这个 pass 的起始位置"是同一个数 —— 判定连续性正是靠这一点。
    pub(crate) raw_start: i64,
    /// 上一个 pass 结束时的本地游标（= 下一次的起点）。
    pub(crate) cursor_end: i64,
    /// 上一个 pass 消费掉的帧数。
    ///
    /// 【为什么单独记】下一个 pass 的宿主位置应当正好是
    /// `raw_start + pass_frames`（宿主按"我渲染了这么多"推进），这是唯一与采样率、
    /// 块大小、chunk 长度都无关的连续性判据。
    pub(crate) pass_frames: i64,
    /// 采样率（取整）。
    pub(crate) sample_rate: u32,
    /// 上一个 pass 最后一块的帧数（容差按它换算）。
    pub(crate) block_frames: i64,
}

/// **R-run-1 / R-run-2**：新 pass 是否延续上一次渲染 run。
///
/// 【为什么单独成纯函数】这条判据决定"渲染出来的音频对不对"：判错一次，整个文件就是
/// 重复的前几秒。纯函数可单测，也能在 `[capture] pass begin` 的诊断行里直接复现。
///
/// 【判据里没有墙钟】run 的边界只能由"宿主是否还在做同一件事"决定。位置信息永远存在，
/// 所以这里只看位置：chunk 之间的秒级间隔、`setActive` 循环、磁盘写入造成的停顿，
/// 都不是边界（它们曾经都是 —— 见 `RENDER_RUN_IDLE_MS` 的说明）。
///
/// * 没有尾巴 / 采样率变了 ⇒ 新 run；
/// * 宿主位置**被钉住**（与上一 pass 相同）⇒ 同一次 run，游标必须接着走（R-run-2）；
/// * 位置**按上一 pass 的消费量前进**（|Δ - pass_frames| ≤ 容差）⇒ 同一次 run
///   （R-run-1：VEGAS 的 chunk 正是这样推进的）；
/// * 其余位置跳变（真 seek / 新一次渲染）⇒ 新 run。
pub(crate) fn render_run_continues(
    tail: Option<RenderRunTail>,
    raw_start: i64,
    sample_rate: f64,
) -> bool {
    let Some(tail) = tail else { return false };
    let rate = sane_rate(sample_rate).round().max(1.0) as u32;
    if rate != tail.sample_rate {
        return false;
    }
    // R-run-2：位置被钉住 —— 位置全在本地游标里，只能靠游标连续来保证正确。
    if raw_start == tail.raw_start {
        return true;
    }
    // R-run-1：宿主位置按"上一 pass 消费了多少"前进。
    let expected = tail.raw_start.saturating_add(tail.pass_frames);
    let tolerance = tail
        .block_frames
        .saturating_mul(RENDER_RUN_BLOCK_TOLERANCE)
        .max(1);
    raw_start.saturating_sub(expected).abs() <= tolerance
}

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

/// 静止多久算"这次预览结束了"（毫秒）。
///
/// 【为什么是两档】没有 run 在跑时，2 s 静默就是"用户停下了"，收尾越早越好；run 还开着
/// 时（chunk 之间）2 s 会把一次渲染切成 N 段 —— 现场 VEGAS 的 chunk 间隔是 4 s，最长
/// 16.6 s。两档共用一条判据，是为了让"哪一种静止"这件事只在一处决定。
pub(crate) fn preview_idle_limit(run_open: bool) -> u64 {
    if run_open {
        RENDER_RUN_IDLE_MS
    } else {
        PREVIEW_QUIET_MS
    }
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
    // ── 渲染 run（见 render_run_continues）────────────────────────────────
    /// 当前 run 的序号（新 run 才自增）。
    run_id: AtomicU64,
    /// 本 pass 是否延续了上一次 run。
    run_continues: AtomicBool,
    /// 上一段 run 的尾巴；字段语义见 {@link RenderRunTail}。
    run_tail_raw: AtomicI64,
    run_tail_cursor: AtomicI64,
    run_tail_rate: AtomicU32,
    run_tail_frames: AtomicI64,
    /// 上一段 pass 消费的帧数（连续性判据的期望位移）。
    run_tail_pass_frames: AtomicI64,
    /// 本 run 累计写下的帧数（只用于 `run end` 的诊断行）。
    render_run_frames: AtomicI64,
    /// 本 run 的**第一块**是否存在，以及它的 `(raw, local)`。
    ///
    /// 【为什么要"第一块"这个读数】`run end … frames=0` 是个需要判读的形状：它可能是
    /// "run 开了但一块都没写"（预览 pass 立刻被播放打断），也可能是"写了却记错账"。
    /// 现场那条 `run end id=1 passes=1 frames=0 reason=play` 就分不清是哪一种。
    /// 有了第一块的 raw/local，"0 帧"立刻能区分"还没开始"与"开始了但没写"。
    run_first_seen: AtomicBool,
    run_first_raw: AtomicI64,
    run_first_local: AtomicI64,
    /// 最近一块的帧数（尾巴的容差换算要用）。
    last_block_frames: AtomicI64,
    /// 本 pass 开始时的本地游标（`begin` 写入）。
    ///
    /// 【为什么需要】"宿主位置前进了 X 帧"只有在与"本 pass 已经消费了 X 帧"相等时才是
    /// 同一段渲染的下一块；否则就是 seek。两个数都相对于**本 pass 的起点**，所以必须
    /// 存下起点，不能拿"当前游标"直接比。
    pass_start_cursor: AtomicI64,
    /// 当前 run 里已经开过多少个 pass（`RENDER_RUN_MAX_PASSES` 兜底）。
    run_passes: AtomicU32,
    /// 上一次 `begin` 的时刻（`gap_ms` 诊断）。
    last_begin_ms: AtomicU64,
    /// 距上一次 `begin` 的间隔（毫秒；`pass begin` 的诊断字段）。
    pass_gap_ms: AtomicU64,
    // ── 这一遍是渲染还是实时监听（见 `CapturePassKind`）──────────────────
    /// 本遍的判定结果（`CapturePassKind::code`）。一旦置成 Render 就不再回落。
    pass_kind: AtomicU8,
    /// 速度判据的窗口起点（相对进程纪元的毫秒；0 = 窗口还没开）。
    feed_window_started_ms: AtomicU64,
    /// 本窗口内已经喂进来的帧数。
    feed_window_frames: AtomicU64,
    /// 本遍累计喂进来的帧数（第三条判据与诊断用）。
    feed_total_frames: AtomicU64,
    /// 最近一次算出的喂入速度（× 100，即 `416500` = 4165×；0 = 还没算出来）。
    feed_speed_x100: AtomicU64,
}

/// 渲染 run 的**行为**开关（环境变量 `HFS_RENDER_RUN=0` 关闭）。
///
/// 【为什么现在默认打开】上一轮把它默认关着，等的是"每个 pass 的 `local` 都从 0 开始"
/// 这一条取证 —— 现场日志已经给足（六个 pass 全为 `local=0`、`same_run=false`，`run`
/// 每次自增）。继续默认关闭等于"明知机制成立、却把缺陷留给用户"。
///
/// 保留开关只为一件事：**现场对照**。设 `HFS_RENDER_RUN=0` 即回到"每个 pass 都从 0 读
/// 快照"的旧行为，不需要重新出包。
///
/// 【为什么用 OnceLock 而不是每次读环境变量】本函数在音频线程上被调用；getenv 只在
/// 第一次（进程内首个处理块）发生一次，之后是一次原子读。
pub(crate) fn render_run_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var("HFS_RENDER_RUN").is_ok_and(|value| value == "0" || value == "false")
    })
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

    /// 当前渲染 run 的序号（诊断）。
    pub(crate) fn run_id(&self) -> u64 {
        self.run_id.load(Ordering::Acquire)
    }

    /// 本 pass 是否延续了上一次渲染 run（诊断；与行为开关无关，永远如实反映判据）。
    pub(crate) fn continues_render_run(&self) -> bool {
        self.run_continues.load(Ordering::Acquire)
    }

    /// 取出并清空上一段 run 的尾巴。
    fn take_run_tail(&self) -> Option<RenderRunTail> {
        let raw = self.run_tail_raw.swap(NO_POSITION, Ordering::AcqRel);
        if raw == NO_POSITION {
            return None;
        }
        Some(RenderRunTail {
            raw_start: raw,
            cursor_end: self.run_tail_cursor.load(Ordering::Acquire),
            pass_frames: self.run_tail_pass_frames.load(Ordering::Acquire),
            sample_rate: self.run_tail_rate.load(Ordering::Acquire),
            block_frames: self.run_tail_frames.load(Ordering::Acquire),
        })
    }

    /// 记下本段 run 的尾巴（**chunk 边界**：end / suspend 时调用）。
    ///
    /// 【为什么要记】下一次 `begin` 靠它判定"这是同一个 run 的下一段"还是"新的一次
    /// 渲染"。判错的后果是整段音频重复（游标每次归零）。
    fn record_run_tail(&self) {
        let raw = self.last_raw.load(Ordering::Acquire);
        if raw == NO_POSITION {
            // 一个块都没见过：没有可延续的位置，不记。
            //
            // 【这条顺带挡掉了"同一段渲染两次"的一半风险】第二次渲染的第一段若一个块都
            // 没产出就结束，它不会留下尾巴，于是永远不会把上一次的游标接过去。
            return;
        }
        let cursor = self.cursor_frames.load(Ordering::Acquire);
        let start = self.pass_start_cursor.load(Ordering::Acquire);
        self.run_tail_raw.store(raw, Ordering::Release);
        self.run_tail_cursor.store(cursor, Ordering::Release);
        self.run_tail_pass_frames
            .store(cursor.saturating_sub(start), Ordering::Release);
        self.run_tail_rate
            .store(self.sample_rate.load(Ordering::Acquire), Ordering::Release);
        self.run_tail_frames.store(
            self.last_block_frames.load(Ordering::Acquire).max(1),
            Ordering::Release,
        );
    }

    /// 丢弃"可以延续的尾巴"：下一次 `begin` 一定开新 run。
    ///
    /// 【什么时候必须调】凡是"这次渲染**结束**了"的信号：用户停止 / 清空捕获 / 播放升起
    /// / 时间线编辑 / 空闲超时。区分它与 chunk 边界是本轮的核心 —— 旧实现把
    /// `setActive(false)`（VEGAS 的 chunk 边界！）当成了结束信号，于是每个 chunk 都成了
    /// 新 run、游标每次归零，渲染文件就是"前 0~1 s 反复重放"。
    pub(crate) fn invalidate_render_run(&self) {
        self.run_tail_raw.store(NO_POSITION, Ordering::Release);
        self.run_tail_pass_frames.store(0, Ordering::Release);
    }

    /// 本 run 是否还"开着"（有可延续的尾巴或正在推进）。
    fn render_run_is_open(&self) -> bool {
        self.run_tail_raw.load(Ordering::Acquire) != NO_POSITION || self.is_active()
    }

    /// 结束本 pass，并**宣告本次渲染结束**（丢尾巴 + 一条可归因的 `run end`）。
    ///
    /// 【为什么单独一个入口】`end()` 是 chunk 边界的语义（"这一段写完了，尾巴留给下一
    /// 段"）。终止语义必须显式分开，否则"哪种结束"就只剩调用点的注释来区分 —— 那正是
    /// R-run-3 出错的方式。
    pub(crate) fn end_run(&self, reason: &'static str) {
        let run = self.run_id.load(Ordering::Acquire);
        let passes = self.run_passes.load(Ordering::Acquire);
        let frames = self.render_run_frames.load(Ordering::Acquire);
        if self.is_preview() || self.kind() == ProcessingKind::Offline {
            // 【`first_block` 与 `ended_at_pass_local` 是给 `frames=0` 用的】前者回答
            // "这个 run 到底写过块没有"，后者回答"本 pass 结束时游标停在哪" —— 两个数
            // 一起看，"开了 pass 却 0 帧"就不会再是一句需要猜的话。
            let first_block = if self.run_first_seen.load(Ordering::Acquire) {
                format!(
                    "{}:{}",
                    self.run_first_raw.load(Ordering::Acquire),
                    self.run_first_local.load(Ordering::Acquire)
                )
            } else {
                "none".to_string()
            };
            let ended_at = self.cursor_frames.load(Ordering::Acquire);
            log::info!(
                "[capture] run end id={run} passes={passes} frames={frames} first_block={first_block} ended_at_pass_local={ended_at} reason={reason}"
            );
        }
        self.end();
        self.invalidate_render_run();
        self.run_passes.store(0, Ordering::Release);
        self.render_run_frames.store(0, Ordering::Release);
        self.run_first_seen.store(false, Ordering::Release);
    }

    /// 记下本 run 的**第一块**（只在第一次生效）。
    ///
    /// 【为什么用"见过"这一位而不是哨兵值】`raw` 与 `local` 都可能合法地等于 0，
    /// 拿 0 当"没写过"会让"第一块正好在 0"的 run 报成 `none`。
    fn note_run_first_block(&self, raw: i64, local: i64) {
        if self.run_first_seen.load(Ordering::Acquire) {
            return;
        }
        self.run_first_raw.store(raw, Ordering::Release);
        self.run_first_local.store(local, Ordering::Release);
        self.run_first_seen.store(true, Ordering::Release);
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
        let new_pass = !self.is_active() || self.kind() != ProcessingKind::Offline;
        if new_pass {
            self.begin_offline(raw_start, sample_rate, pass_sequence);
        }
        let position = self.cursor_frames.load(Ordering::Acquire);
        self.last_block_frames
            .store(frames as i64, Ordering::Release);
        // 尾巴要用它：离线块同样有"最后一块的宿主位置"。
        self.last_raw.store(raw_start, Ordering::Release);
        if new_pass {
            log::info!(
                "[capture] pass begin run={} kind=offline raw={} local={} frames={} same_run={}",
                self.run_id(),
                raw_start,
                position,
                frames,
                self.continues_render_run()
            );
        }
        self.note_run_first_block(raw_start, position);
        self.cursor_frames
            .fetch_add(frames as i64, Ordering::AcqRel);
        self.render_run_frames
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
                // 宿主开始真实播放 / 换了处理模式：这次渲染结束了（不是 chunk 边界）。
                self.end_run("play");
                return PreviewDecision::None;
            }
            let last_raw = self.last_raw.load(Ordering::Acquire);
            if last_raw != NO_POSITION && last_raw != raw_start {
                // 【宿主位置动了：两种可能，只有一种该结束】旧实现一律 end()，而 VEGAS
                // 的 chunk 恰好是这样推进的（位置按"上一段消费了多少"前移），于是每个
                // chunk 都成了一个新 pass、游标回到 0 —— 输出就是"前 0~1 s 反复重放"。
                // 前进量与本 pass 已消费的帧数一致 ⇒ 同一段渲染的下一块，游标接着走。
                if !self.position_progress_is_continuous(raw_start, frames) {
                    self.end();
                    return PreviewDecision::None;
                }
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
                    // 【续算的 pass 不能被"按 origin 重算"抹回 0】本 pass 若是从上一段
                    // run 的游标接着走的，候选块的本地起点就是那个游标；按 origin 重算会
                    // 把它变成 first_raw - origin（chunk 里恒为 0），续算等于没发生 ——
                    // 输出又回到"每个 chunk 都从 0 开始"。
                    let base = if self.continues_render_run() {
                        self.pass_start_cursor.load(Ordering::Acquire)
                    } else if candidate_local {
                        first_raw.saturating_add(candidate_offset)
                    } else {
                        first_raw.saturating_sub(origin)
                    }
                    .max(0);
                    let local = base.saturating_add(prior as i64);
                    self.note_run_first_block(raw_start, local);
                    self.cursor_frames
                        .store(local.saturating_add(frames as i64), Ordering::Release);
                    self.render_run_frames
                        .fetch_add(frames as i64, Ordering::AcqRel);
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
            self.note_run_first_block(raw_start, position);
            self.last_block_frames
                .store(frames as i64, Ordering::Release);
            self.cursor_frames
                .fetch_add(frames as i64, Ordering::AcqRel);
            self.render_run_frames
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
        // 【取证】每个 pass 的第一块一行：run 序号、判据结论、宿主位置与本地起点。
        // 现场判据是"hash 按 pass 周期性重复且 local 每 pass 从 0 开始"（见计划 §5.3.3）。
        log::info!(
            "[capture] pass begin run={} pass={} gap_ms={} kind=frozen raw={} local={} frames={} silent={} same_run={}",
            self.run_id(),
            self.run_passes.load(Ordering::Acquire),
            self.pass_gap_ms.load(Ordering::Acquire),
            raw_start,
            self.cursor_frames.load(Ordering::Acquire),
            frames,
            silent,
            self.continues_render_run()
        );
        self.last_block_frames
            .store(frames as i64, Ordering::Release);
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

    /// 宿主位置的前进量是否等于本 pass 已消费的帧数（= 同一段渲染的下一块）。
    ///
    /// 【为什么用"消费量"而不是"块数"】VEGAS 的 chunk 长度不固定（现场 0.5 s、0.79 s、
    /// 2.25 s 都有），块大小也不固定（1200 / 848 交替）。唯一恒等的关系是：宿主按
    /// "我刚才渲染了多少"推进它的位置 —— 那正好是我们写进游标的帧数。
    fn position_progress_is_continuous(&self, raw_start: i64, frames: usize) -> bool {
        // 【只在 Active 上成立】候选期间块由 HostClock 按 raw 写入、本地游标**不推进**，
        // 于是"推进量 == 消费量"这个等式没有意义：raw 一动就是真实传输开始了，必须结束。
        if self.phase() != PreviewPhase::Active {
            return false;
        }
        let first = self.first_raw.load(Ordering::Acquire);
        if first == NO_POSITION {
            return false;
        }
        let progressed = raw_start.saturating_sub(first);
        let consumed = self
            .cursor_frames
            .load(Ordering::Acquire)
            .saturating_sub(self.pass_start_cursor.load(Ordering::Acquire));
        let tolerance = (frames as i64).max(1);
        progressed > 0 && progressed.saturating_sub(consumed).abs() <= tolerance
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
        // ── 渲染 run：这一 pass 是不是上一次渲染的延续 ──────────────────────
        //
        // 判据永远算（诊断要如实），**行为**由 HFS_RENDER_RUN 决定（见
        // render_run_enabled；默认打开，`=0` 回到旧行为）。
        let tail = self.take_run_tail();
        let mut continues = render_run_continues(tail, origin_raw, sample_rate);
        // 【pass 数上界】判据失效时（例如"同一段渲染两次"且位置恰好连续）不能让一个 run
        // 无限接下去：超过 `RENDER_RUN_MAX_PASSES` 就开新 run 并留下一条 Warn。
        if continues && self.run_passes.load(Ordering::Acquire) >= RENDER_RUN_MAX_PASSES {
            log::warn!(
                "[capture] render run {run} reached {RENDER_RUN_MAX_PASSES} passes; \
                 starting a new run (the continuation predicate is no longer trusted)",
                run = self.run_id.load(Ordering::Acquire)
            );
            continues = false;
        }
        self.run_continues.store(continues, Ordering::Release);
        let new_run = !continues;
        if new_run {
            self.run_id.fetch_add(1, Ordering::AcqRel);
            self.run_passes.store(0, Ordering::Release);
            self.render_run_frames.store(0, Ordering::Release);
            self.run_first_seen.store(false, Ordering::Release);
        }
        // R-run-2：同一次 run 内本地游标接着走，**不是**从 0 重新开始。
        let cursor = match (continues && render_run_enabled(), tail) {
            (true, Some(tail)) => tail.cursor_end,
            _ => cursor,
        };
        self.run_passes.fetch_add(1, Ordering::AcqRel);
        self.pass_start_cursor.store(cursor, Ordering::Release);
        // 【诊断：本 run 内的 pass 序号与距上一次 pass 的间隔】
        // 现场只要看到这两个数，就能判断"这是同一次渲染的下一段"还是"新的一次渲染"。
        let begun = now_ms();
        let previous_begin = self.last_begin_ms.swap(begun, Ordering::AcqRel);
        self.pass_gap_ms.store(
            if previous_begin == 0 {
                0
            } else {
                begun.saturating_sub(previous_begin)
            },
            Ordering::Release,
        );
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
        // 【这一遍的种类从零开始判】它是"整遍不变"的结论，不能从上一条流继承。
        self.pass_kind.store(CapturePassKind::Realtime.code(), Ordering::Release);
        self.feed_window_started_ms.store(0, Ordering::Release);
        self.feed_window_frames.store(0, Ordering::Release);
        self.feed_total_frames.store(0, Ordering::Release);
        self.feed_speed_x100.store(0, Ordering::Release);
        // 离线从 Active 开始；预览由调用方覆盖成 Candidate。
        self.phase
            .store(PreviewPhase::Active.code(), Ordering::Release);
    }

    /// 本遍是渲染还是实时监听。
    pub(crate) fn pass_kind(&self) -> CapturePassKind {
        CapturePassKind::from_code(self.pass_kind.load(Ordering::Acquire))
    }

    /// 最近一次算出的喂入速度（倍实时；`None` = 还没算出来）。
    pub(crate) fn feed_speed(&self) -> Option<f64> {
        let value = self.feed_speed_x100.load(Ordering::Acquire);
        (value > 0).then(|| value as f64 / 100.0)
    }

    /// 本遍累计喂进来的帧数。
    pub(crate) fn feed_total_frames(&self) -> u64 {
        self.feed_total_frames.load(Ordering::Acquire)
    }

    /// 记一次喂入进度，并（在有证据时）把这一遍判成渲染。
    ///
    /// 【为什么在音频线程上只做原子写】这是每个音频块都会走到的路径。窗口到期时才
    /// 算一次除法（每 250 ms 一次），其余时候只是两次 `fetch_add`。
    pub(crate) fn note_feed_progress(
        &self,
        frames: u32,
        sample_rate: f64,
        offline: bool,
        playing: bool,
    ) {
        // 【实时播放永远不是渲染】这一句必须排在最前：宿主真实播放时 `kPlaying` 是 1，
        // 而渲染那一趟全程是 0。把它当"可能是渲染"去算速度，会让一次正常的缓冲突发
        // 把实时监听判成渲染 ⇒ 录制中输出被静音。
        if playing && !offline {
            self.pass_kind
                .store(CapturePassKind::Realtime.code(), Ordering::Release);
            self.feed_window_started_ms.store(0, Ordering::Release);
            self.feed_window_frames.store(0, Ordering::Release);
            self.feed_total_frames.store(0, Ordering::Release);
            return;
        }
        if offline {
            self.pass_kind
                .store(CapturePassKind::Render.code(), Ordering::Release);
        }
        let frames = u64::from(frames);
        let total = self.feed_total_frames.fetch_add(frames, Ordering::AcqRel) + frames;
        let rate = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate
        } else {
            48_000.0
        };
        // 【第二条判据：喂入长度】`kPlaying` 全程为 0 的喂入流已经超过 1 秒的起点
        // 窗口 —— 实时监听不会有这个形状（真播放会升起 kPlaying）。
        if total as f64 > rate * RENDER_LENGTH_SEC {
            self.pass_kind
                .store(CapturePassKind::Render.code(), Ordering::Release);
        }
        // 【第一条判据：喂入速度】窗口到期时算一次。
        let now = now_ms();
        let started = self.feed_window_started_ms.load(Ordering::Acquire);
        if started == 0 {
            self.feed_window_started_ms.store(now, Ordering::Release);
            self.feed_window_frames.store(frames, Ordering::Release);
            return;
        }
        let window_frames = self
            .feed_window_frames
            .fetch_add(frames, Ordering::AcqRel)
            .saturating_add(frames);
        let elapsed = now.saturating_sub(started);
        if elapsed < FEED_SPEED_WINDOW_MS {
            return;
        }
        let speed = window_frames as f64 / (elapsed as f64 / 1000.0) / rate;
        self.feed_speed_x100
            .store((speed * 100.0) as u64, Ordering::Release);
        if speed > RENDER_SPEED_X {
            self.pass_kind
                .store(CapturePassKind::Render.code(), Ordering::Release);
        }
        self.feed_window_started_ms.store(now, Ordering::Release);
        self.feed_window_frames.store(0, Ordering::Release);
    }

    /// `setProcessing(false)` / 组件失活：保留 cursor 与 origin。
    pub(crate) fn suspend(&self) {
        let phase = self.phase();
        if matches!(phase, PreviewPhase::Candidate | PreviewPhase::Active) {
            // 挂起时游标与 origin 都保留，尾巴也照记：chunk 之间的 setProcessing(false)
            // 不能让下一次 begin 以为"这是全新的一次渲染"。
            self.record_run_tail();
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

    /// 非音频线程：静止足够久就结束预览。
    ///
    /// 返回 true 表示本次调用结束了预览，调用方应 promote 当前 take 并清时钟。
    ///
    /// 【两种静止必须分开】
    ///   * **没有 run 在跑**：这是一次普通预览静了下来 —— 2 s（`PREVIEW_QUIET_MS`）收尾，
    ///     与今天一致；
    ///   * **run 还开着**（chunk 之间）：宿主只是暂时不送块（现场 VEGAS 的 chunk 间隔是
    ///     4 s，最长 16.6 s）。这时按 2 s 收尾会把一次渲染切成 N 段、每段游标归零 ——
    ///     输出就是"前 0~1 s 反复重放"。改用 `RENDER_RUN_IDLE_MS` 兜底。
    ///
    /// 【收尾必须连带丢尾巴】走到这里说明宿主已经不再做同一件事了：留着可延续的尾巴，
    /// 下一次渲染的首块就会把这次的游标接过去（新文件从中间开始）。
    pub(crate) fn end_if_quiet(&self) -> bool {
        if !self.is_preview() {
            return false;
        }
        let phase = self.phase();
        if phase == PreviewPhase::Idle {
            return false;
        }
        let last = self.last_at_ms.load(Ordering::Acquire);
        if last == 0 {
            return false;
        }
        let idle = now_ms().saturating_sub(last);
        if idle <= preview_idle_limit(self.render_run_is_open()) {
            return false;
        }
        self.end_run("idle");
        true
    }

    pub(crate) fn end(&self) {
        // 先记尾巴再清状态：清了就再也算不出"下一个 pass 能不能接上"。
        self.record_run_tail();
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
        self.last_block_frames.store(0, Ordering::Release);
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
    ///
    /// 【为什么这里**不**结束渲染 run】R-run-3 曾把 `setActive(false)` 当成"这次渲染结束"
    /// 的硬信号。现场证据推翻了它：**VEGAS 的 chunk 边界就是一次 setActive 循环**
    /// （16:19:31.962 `setActive(0)` → 32.671 `setupProcessing` → `setActive(1)` → 下一个
    /// chunk，随后 `pass begin` 的 `local` 又是 0）。把 chunk 信号当结束信号，续算在
    /// 结构上就不可能生效。
    ///
    /// 真正该结束 run 的信号都由各自的入口显式给出：播放升起、无法解释的位置跳变、
    /// 采样率变化、时间线编辑、用户停止、空闲超时。
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
    ///
    /// 【为什么是 `end_run` 而不是 `end`】用户停手就是"这次渲染结束了"最明确的信号。
    /// `end()` 会把尾巴留给下一次（那是 chunk 边界的语义），而用户的下一次可能是**另一
    /// 段**渲染 —— 接着上一次的游标写，新文件会从中间开始。
    pub(crate) fn end_processing_pass(&self) {
        self.processing_pass.end_run("stop");
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

    /// **判据：喂 30 秒音频只花 1/1000 的墙钟时间 ⇒ 这是渲染。**
    ///
    /// 【现场原形】VEGAS Render As 把 1 520 704 帧（31.68 s）在 365 ms 内喂完
    /// ≈ 4 165 倍实时，而 `kOffline` 全程为假。
    #[test]
    fn a_fast_feed_is_a_render() {
        let pass = ProcessingPass::default();
        assert_eq!(pass.pass_kind(), CapturePassKind::Realtime);
        // 250 ms 的窗口里喂进 48 000 × 30 帧 = 30 秒音频 ⇒ 120 倍实时。
        pass.note_feed_progress(48_000 * 30, 48_000.0, false, false);
        std::thread::sleep(std::time::Duration::from_millis(260));
        pass.note_feed_progress(48_000, 48_000.0, false, false);
        assert_eq!(pass.pass_kind(), CapturePassKind::Render);
        assert!(pass.feed_speed().is_some_and(|speed| speed > RENDER_SPEED_X));
    }

    /// **判据：实时播放不是渲染。**
    ///
    /// 【为什么这条必须有】把实时播放误判成渲染的代价是**录制中输出被静音** ——
    /// 用户会听到"一按录制就没声音"。所以 `kPlaying=1` 必须直接把结论钉回实时。
    #[test]
    fn a_realtime_feed_is_not_a_render() {
        let pass = ProcessingPass::default();
        for _ in 0..40 {
            pass.note_feed_progress(48_000, 48_000.0, false, true);
        }
        std::thread::sleep(std::time::Duration::from_millis(260));
        pass.note_feed_progress(48_000, 48_000.0, false, true);
        assert_eq!(pass.pass_kind(), CapturePassKind::Realtime);
    }

    /// **判据：不报播放、却已经喂满 1 秒以上 ⇒ 这是渲染。**
    #[test]
    fn a_long_silent_feed_without_k_playing_is_a_render() {
        let pass = ProcessingPass::default();
        // 1 秒 = 48 000 帧；喂 2 秒。
        pass.note_feed_progress(48_000, 48_000.0, false, false);
        assert_eq!(pass.pass_kind(), CapturePassKind::Realtime);
        pass.note_feed_progress(48_000, 48_000.0, false, false);
        pass.note_feed_progress(48_000, 48_000.0, false, false);
        assert_eq!(pass.pass_kind(), CapturePassKind::Render);
    }

    /// 宿主明说离线 ⇒ 立刻是渲染，不需要任何速度证据。
    #[test]
    fn an_offline_host_is_a_render_immediately() {
        let pass = ProcessingPass::default();
        pass.note_feed_progress(960, 48_000.0, true, false);
        assert_eq!(pass.pass_kind(), CapturePassKind::Render);
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

    // ── 渲染 run（R-run-1 / R-run-2）────────────────────────────────────
    //
    // 判据判错的后果不是"慢一点"，而是**渲染出来的音频整段重复**：每个 pass 的本地
    // 游标都从 0 开始，输出就一遍遍读前几秒。所以这些用例钉的是判据本身。

    fn tail(raw: i64, cursor: i64, pass_frames: i64) -> RenderRunTail {
        RenderRunTail {
            raw_start: raw,
            cursor_end: cursor,
            pass_frames,
            sample_rate: 48_000,
            block_frames: 512,
        }
    }

    /// 喂一个 chunk：两块 0.42 s 的块刚好凑够候选确认（0.75 s / 至少 2 块）。
    fn feed_chunk(pass: &ProcessingPass, sequence: &AtomicU64, raw: i64, salt: u64) {
        for block in 0..2 {
            let _ = feed(
                pass,
                sequence,
                raw,
                20_000,
                salt * 16 + block + 1,
                false,
                false,
                SessionPhase::Paused,
                false,
                None,
                None,
            );
        }
    }

    #[test]
    fn a_render_run_needs_a_previous_tail() {
        assert!(!render_run_continues(None, 0, 48_000.0));
    }

    /// **回归：chunk 边界不是 run 边界。**
    ///
    /// 【现场】VEGAS 的每个 chunk 都以一次 `setActive(0)` → `setupProcessing` →
    /// `setActive(1)` 开头，`pass begin` 的 `local` 每次都是 0，而 chunk 之间隔着 4 s
    /// 甚至 16.6 s。旧实现的三道闸（setActive 即结束、2 s 间隔即结束、2 s 静默即收尾）
    /// 各判它一次，于是每个 chunk 都成了新 run —— 渲染文件就是"前 0~1 s 反复重放"。
    /// 这条用八个 chunk 走完整条通路：run 号不变、游标单调、每段正好接上。
    #[test]
    fn chunk_boundaries_do_not_break_the_run() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        let mut raw = 720_000_i64;
        feed_chunk(&pass, &sequence, raw, 0);
        let run = pass.run_id();
        assert_eq!(
            pass.cursor_frames(),
            40_000,
            "第一个 chunk 必须写满两块的帧数"
        );
        for chunk in 1..8_u64 {
            // chunk 边界：pass 结束（宿主 setProcessing(false) 的**硬**形态）。
            // 游标与"可以延续的尾巴"都必须留下来。
            pass.end();
            // 宿主按"我渲染了多少"推进位置，然后开下一段。
            raw += 40_000;
            feed_chunk(&pass, &sequence, raw, chunk);
            assert!(pass.continues_render_run(), "chunk {chunk} 必须续算");
            assert_eq!(pass.run_id(), run, "chunk {chunk} 不得开新 run");
            assert_eq!(
                pass.cursor_frames(),
                40_000 * (chunk as i64 + 1),
                "chunk {chunk} 的游标必须接着上一段走，不能归零"
            );
        }
    }

    /// **更软的一种 chunk 边界：只挂起，不结束 pass。**
    ///
    /// 【为什么也要钉】宿主在 chunk 之间给的往往是 setProcessing(false)（只挂起）。
    /// 这种情形下 pass 根本不该结束 —— 位置前移量与本 pass 的消费量相同，就直接接着写。
    /// 少一次"结束再重建"，也就少一个丢块的窗口。
    #[test]
    fn a_suspended_chunk_boundary_does_not_even_end_the_pass() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        let mut raw = 720_000_i64;
        feed_chunk(&pass, &sequence, raw, 0);
        let run = pass.run_id();
        let pass_id = pass.pass_id();
        for chunk in 1..4_u64 {
            pass.suspend();
            pass.resume();
            raw += 40_000;
            feed_chunk(&pass, &sequence, raw, chunk);
            assert_eq!(pass.run_id(), run);
            assert_eq!(pass.pass_id(), pass_id, "挂起-恢复不得开新 pass");
            assert_eq!(
                pass.cursor_frames(),
                40_000 * (chunk as i64 + 1),
                "挂起-恢复之后游标必须接着走"
            );
        }
    }

    #[test]
    fn a_render_run_does_not_survive_a_sample_rate_change() {
        let previous = tail(0, 24_000, 24_000);
        assert!(!render_run_continues(Some(previous), 0, 44_100.0));
    }

    #[test]
    fn a_pinned_host_position_continues_the_run() {
        // 事件内的预览：raw 恒为同一个数，位置全在本地游标里。
        let previous = tail(0, 24_000, 24_000);
        assert!(render_run_continues(Some(previous), 0, 48_000.0));
    }

    #[test]
    fn a_contiguous_host_position_continues_the_run() {
        // 宿主按上一段消费的帧数前移 —— 这是 VEGAS chunk 的真实形状。
        let previous = tail(24_000, 48_000, 24_000);
        assert!(render_run_continues(Some(previous), 48_000, 48_000.0));
    }

    #[test]
    fn a_real_seek_starts_a_new_run() {
        let previous = tail(0, 48_000, 48_000);
        assert!(!render_run_continues(Some(previous), 48_000 * 30, 48_000.0));
    }

    /// **回归：播放升起就是"这次渲染结束了"。**
    ///
    /// 【为什么这条必须钉住】续算的判据只看位置；而"用户开始播放"与"渲染的下一段"
    /// 在位置上可能完全连续。播放是显式信号，必须由它来断开，不能等位置对不上。
    #[test]
    fn a_real_play_ends_the_run() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        feed_chunk(&pass, &sequence, 0, 0);
        let run = pass.run_id();
        // 下一块带着 kPlaying=1 到达：这是真实传输，不是渲染的下一段。
        let decision = feed(
            &pass,
            &sequence,
            0,
            20_000,
            99,
            false,
            true,
            SessionPhase::Rolling,
            true,
            None,
            None,
        );
        assert_eq!(decision, PreviewDecision::None);
        assert!(!pass.is_preview(), "播放升起必须结束预览 pass");
        feed_chunk(&pass, &sequence, 0, 1);
        assert!(!pass.continues_render_run());
        assert_ne!(pass.run_id(), run, "播放之后必须开新 run");
    }

    /// **回归：无法解释的位置跳变结束 run。**
    #[test]
    fn an_unexplained_jump_ends_the_run() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        feed_chunk(&pass, &sequence, 0, 0);
        let run = pass.run_id();
        pass.end();
        // 跳了 5 s，而上一个 pass 只消费了 40 000 帧：这不是"下一段"。
        feed_chunk(&pass, &sequence, 48_000 * 5, 1);
        assert!(!pass.continues_render_run());
        assert_ne!(pass.run_id(), run);
    }

    /// **回归：同一段渲染两次不得并成一个 run。**
    ///
    /// 【为什么会被并上】第二次渲染的首块位置与第一次的最后一块完全相同（同一段起止），
    /// "位置被钉住"这一支会判成续算，新文件于是从中间开始。用户停手时 `end_run` 会丢掉
    /// 尾巴；这条把那个动作钉住。
    #[test]
    fn two_renders_of_the_same_range_do_not_merge() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        feed_chunk(&pass, &sequence, 0, 0);
        let run = pass.run_id();
        pass.end_run("stop");
        feed_chunk(&pass, &sequence, 0, 1);
        assert!(!pass.continues_render_run());
        assert_ne!(pass.run_id(), run);
        assert_eq!(pass.cursor_frames(), 40_000, "新 run 必须从 0 起算");
    }

    /// **兜底：pass 数超过上界就开新 run。**
    #[test]
    fn passes_are_capped() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        feed_chunk(&pass, &sequence, 0, 0);
        let run = pass.run_id();
        let mut raw = 0_i64;
        let mut restarted = None;
        for index in 1..=(RENDER_RUN_MAX_PASSES + 2) {
            pass.end();
            raw += 40_000;
            feed_chunk(&pass, &sequence, raw, index as u64);
            if pass.run_id() != run {
                restarted = Some(index);
                break;
            }
        }
        assert!(
            restarted.is_some(),
            "超过 {RENDER_RUN_MAX_PASSES} 个 pass 之后必须开新 run"
        );
    }

    #[test]
    fn consecutive_offline_passes_with_a_pinned_raw_share_one_run() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        let _ = pass.note_offline_block(2, 0, 512, 48_000.0, &sequence);
        let first_run = pass.run_id();
        pass.end();
        let _ = pass.note_offline_block(2, 0, 512, 48_000.0, &sequence);
        assert!(pass.continues_render_run());
        assert_eq!(pass.run_id(), first_run);
    }

    #[test]
    fn a_seek_between_offline_passes_starts_a_new_run() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        let _ = pass.note_offline_block(2, 0, 512, 48_000.0, &sequence);
        let first_run = pass.run_id();
        pass.end();
        let _ = pass.note_offline_block(2, 48_000 * 30, 512, 48_000.0, &sequence);
        assert!(!pass.continues_render_run());
        assert_ne!(pass.run_id(), first_run);
    }

    #[test]
    fn invalidating_a_run_breaks_the_continuation() {
        let pass = ProcessingPass::default();
        let sequence = sequence();
        let _ = pass.note_offline_block(2, 0, 512, 48_000.0, &sequence);
        pass.end();
        pass.invalidate_render_run();
        let _ = pass.note_offline_block(2, 0, 512, 48_000.0, &sequence);
        assert!(!pass.continues_render_run());
    }

    /// **两档静默阈值**：没有 run 在跑时 2 s 收尾；run 还开着（chunk 之间）等 30 s。
    #[test]
    fn a_quiet_preview_only_waits_long_when_a_run_is_open() {
        assert_eq!(preview_idle_limit(false), PREVIEW_QUIET_MS);
        assert_eq!(preview_idle_limit(true), RENDER_RUN_IDLE_MS);
        // 现场最长的 chunk 间隔是 16.6 s，阈值必须高于它。（常量比较，编译期断言即可。）
        const { assert!(RENDER_RUN_IDLE_MS > 16_600) };
    }
}
