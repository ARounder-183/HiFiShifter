//! 每个 VST3 entry 的扩展所有权。强引用由 entry builder 保留，不在组件销毁时悬空。
//!
//! # 为什么是一个模块目录
//!
//! 这个模块曾经是**一个 8597 行的文件**：输出仲裁、门控、流几何、捕获输出、宿主绑定、
//! 持久化状态与诊断环挤在一处，改任意一件事都要先在同一份文件里翻几千行。现在按
//! **职责**拆成同目录的子模块，`ExtensionOwner` 与它的常量留在这里。
//!
//! 【移动是逐字的】除每条 item 的可见性与每个文件的说明之外，没有一行被改写：每个
//! `impl ExtensionOwner` 块都由原来那一个块按方法组切开，组内顺序不变。
//!
//! 【可见性为什么是 `pub(super)`】拆分前这些 item 是**本模块私有**的，即可见集合是
//! "`render::extension` 及其后代"。子模块里的 `pub(super)` 表示**同一个集合**，因此
//! 可达性一点没变 —— `dead_code` / `private_interfaces` 报的还是原来那几条，不会因为
//! 搬家而静默消失或凭空出现。原来就写着 `pub(super)`（= `pub(in crate::render)`）的那
//! 一个方法改写成显式路径，因为子模块里的 `pub(super)` 会窄成一层。
//!
//! 【子模块之间怎么互相看得见】每个子模块 `use super::*`：它拿到的是本模块的私有名字
//! （`use` 绑定、`pub(super)` 类型与常量），因为子模块本来就能看见祖先的私有 item。
//! 搬走的代码里 `super::document` 这类写法沿用拆分前的语义（拆分前 `super` 就是
//! `render`），由下面那条 `use super::{document, …}` 维持。

use super::ownership::{region_owners, RegionKey};
use crate::capture::recorder::{
    BlockContext, BlockSplit, PositionSource, ProcessStream, SubBlock, TransportPhase,
};
use crate::sync::MutexExt;
use ara2_bridge::core::{ApiGeneration, AraError};
use ara2_bridge::plugin::{ExtensionBinding, ExtensionRoles};
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{
    AtomicBool, AtomicI32, AtomicI64, AtomicU32, AtomicU64, AtomicU8, Ordering,
};
use std::sync::{Arc, Mutex};

// 被搬走的代码沿用了拆分前的 `super::document` 写法（拆分前 `super` 就是 `render`）。
// 这里把用到的兄弟模块重新导出成本模块的名字，让移动保持逐字、可逐 token 核对。
use super::{document, input, ownership, preparation, snapshot, transport};

mod ara;
mod binding;
mod capture;
mod declick;
mod diagnostics;
mod host_link;
mod playback;
mod restore;
mod roles;
mod session;
mod stream;
mod trace;
mod types;

// 值类型与诊断环的读数在本模块名下重新导出：子模块与测试模块都按裸名引用它们。
use restore::*;
use stream::*;
use trace::*;
use types::*;

// 这些类型本来就是 `pub(crate)`（别的模块按 `render::extension::X` 引用），搬家后
// 必须在本模块名下重新导出，否则它们的可见面会静默窄一层。
pub(crate) use types::{
    BlockEvidence, CaptureOutput, CaptureStage, HostAudioState, InstanceMode, ScopeSource,
    SnapshotSource,
};

/// 原生接口与只读元数据属于真实组件，缓存不跨其文档/分配/host变更复用。
#[derive(Default)]
pub(crate) struct ExtensionOwner {
    binding: Mutex<Option<ExtensionBinding>>,
    /// 已退役的扩展实例存储。
    ///
    /// 【为什么不能立刻 drop】宿主可能仍持有上一份文档的
    /// `ARAPlugInExtensionInstance*`，并在同一次 bind 回调之后补发最后几次
    /// `removePlaybackRegion`。立刻释放是 use-after-free；把它们留到组件销毁
    /// （`ExtensionOwner` 析构）再统一释放，代价是"一个处理器一生绑定过的文档数"
    /// 条额外存储，上界极小。
    retired: Mutex<Vec<ExtensionBinding>>,
    /// 最近一次 bind 的目标文档 id（仅诊断；`0` = 从未绑定）。
    ///
    /// 【为什么不用 `document` 反查】文档关闭后 `document` 被清空，而诊断需要
    /// 回答"上一次绑的是谁、这次绑的是谁"。
    bound_document_id: AtomicU64,
    document: Mutex<Option<std::sync::Weak<super::document::DocumentSession>>>,
    /// 本实例**曾经**成功绑定过 ARA 文档（一旦置真永不回落）。
    ///
    /// 【为什么需要它】`document` 字段同时表达两种完全不同的处境：`None` 既可能是
    /// "宿主从未绑定"（Cubase 实测：从不查询 ARA 入口点），也可能是"绑定过、文档已
    /// 销毁"。两者此前共用一句 "document closed"，前端据此提示"关闭并重新打开插件
    /// 窗口"—— 而前者重开一万次也一样。用一个单调标志把两者分开，让错误措辞说实话。
    ///
    /// 【为什么单调不回落】绑定过又关闭就是**真的**关闭了；重置它会让措辞静默退化回
    /// 那句误导性的提示。`document_closed()` 因此刻意不清它。
    bound_once: std::sync::atomic::AtomicBool,
    assignments: Mutex<HashMap<i32, Vec<RegionKey>>>,
    sequences: Mutex<HashMap<i32, Vec<u64>>>,
    role: AtomicI32,
    roles: AtomicI32,
    /// 本实例的运行形态（[`InstanceMode`] 的 `raw()`）。
    ///
    /// 【为什么是原子而不是锁下的字段】`audio_process_inner` 在**音频线程**上每块都要
    /// 问一次"要不要走捕获通路"，那里不能取锁。而写入点只有两个（ARA 绑定成功、
    /// 绑定窗口关闭），频率是"每次会话一次"，所以原子读写绰绰有余。
    ///
    /// 【为什么单调（只前进、不回落）】两个取值都是**关于过去**的事实：
    /// "宿主绑过我们"与"宿主过了宽限期仍未绑"。它们不会因为后来的事件变成假的。
    /// 唯一的例外是宿主违反 ARA 契约、在 `setActive` 之后才绑定 —— 那时写入点会
    /// 覆盖成 `Ara`，而 `Ara` 本来就该赢（`signal_source` 也先查实时绑定）。
    binding_mode: AtomicI32,
    /// ARA 绑定发生后，需要给已经打开的视图补发一次 bootstrap。
    ///
    /// 【为什么需要】窗口可能在绑定之前打开，那时 `hostKind` 只能报
    /// `ara_only`/`capture`；绑定后能力位与形态都变了，而 bootstrap 是导航前注入的
    /// 静态快照。视图定时器看到这个标记就推一条 `plugin_bootstrap_changed`。
    bootstrap_changed: AtomicBool,
    /// companion 层"已绑定过"被复位重建的次数（诊断）。
    ///
    /// 【为什么值得单独计数】正常宿主一次工程生命周期只复位一次（重开工程的那一刻）。
    /// 持续增长说明宿主在反复重绑，而复位本身要丢弃控制器绑定 —— 那是需要被看见的事实。
    companion_resets: std::sync::atomic::AtomicU64,
    /// 上一次"恢复宿主持久化状态"的结果（`RESTORE_*`）。
    ///
    /// 【为什么需要它】参数/曲线的恢复有两条载体（ARA 存档与 VST3 `setState`），
    /// 而合并失败此前**只写一行 `log::warn!`** —— 用户无法回答"参数到底读回来没有"，
    /// 诊断包里也找不到答案。把结果变成状态之后，界面与诊断包都能说真话。
    restore_status: std::sync::atomic::AtomicU8,
    /// 最近一次恢复失败的真因（有界；成功时清空）。
    restore_error: Mutex<Option<String>>,
    /// 恢复载荷仍挂着、但还没到可合并时机的可读原因（成功/失败时清空）。
    ///
    /// 【为什么需要它】`host timeline unavailable` 这类"文档还没就绪"不是失败；
    /// 把它报成 failed 会让用户在正常重开过程中看到一条误导性的终态。这里把
    /// "在等什么"单独记下来，UI 报 `restore_pending`，诊断包能回答"卡在哪一步"。
    restore_pending_reason: Mutex<Option<String>>,
    /// 成功合并过一次持久化状态的次数（诊断）。
    restore_applied_generation: std::sync::atomic::AtomicU64,
    /// **文档上是否有别的实例持有 `PLAYBACK_RENDERER` 角色**。
    ///
    /// 【为什么需要这个原子量】`is_editor_only()` 要在**音频线程**上无锁求值
    /// （`vst3.rs::audio_process`），因而不可能现场去查文档（那要取锁）。而"文档上还有没有
    /// 别的 playback 实例"是**文档级**事实，由 [`super::document::DocumentSession`]
    /// 在每次绑定/分配变化时算一次并推送到这里；变化频率是"用户增删插件实例"，
    /// 不是音频帧率。
    ///
    /// 【为什么这解决了 Cubase 的"编辑不改变声音"】实测 Cubase 15 只给插件**一个**
    /// 实例、角色是 `[editor, view]`（日志 `playback ownership: 1 live renderer
    /// instance(s) but NONE holds the playback role`），于是 `process` 全部走
    /// `pass_through` —— 用户听到的永远是宿主原声，再正确的编辑也听不出来。
    /// 有了这个读数就能区分两种在文档级**不同形**的形态：
    ///
    ///   * REAPER：编辑器实例 + 播放实例并存 ⇒ 本实例 `playback_covered = true` ⇒ 继续透传
    ///     （`vst3/audio_boundary_tests` 的既有契约不变）；
    ///
    ///   * Cubase 单实例：没有任何实例持 playback ⇒ `false` ⇒ 该实例接管渲染。
    playback_covered: std::sync::atomic::AtomicBool,
    /// 文档关闭后重绑到另一份文档的次数（诊断）。
    ///
    /// 【为什么必须是计数】"重开工程复用同一实例"正是用户报的
    /// "只有删掉插件再添加才好"的机制。计数让诊断包能回答"发生过几次重绑"，
    /// 也用来验证 T6 真的接上了（不重绑时永远是 0）。
    rebind_generation: std::sync::atomic::AtomicU64,
    /// 文档级"没有任何实例持 playback 角色"时被提升为渲染者的次数（诊断）。
    ///
    /// 【为什么必须是计数】这是"宿主没有给我们 playback renderer"的现场证据，
    /// 而它每次都只打一条限流的 WARN。计数让诊断包能回答"发生过几次"，
    /// 也用来验证 T11 的提升真的发生了。
    promoted_to_renderer: std::sync::atomic::AtomicU64,
    closed: std::sync::atomic::AtomicBool,
    writer_id: AtomicU64,
    /// 当前编辑后端；`None` ⇒ 尚未绑定任何原生扩展（与 `HostBackend::AraOnly` 不同：
    /// 后者是"宿主明确只给 ARA"，前者是"还没接上"）。
    ///
    /// 【为什么字段不是具体类型 `Arc<NativeHost>`】通用层（本文件）不该按 REAPER 类型
    /// 书写。存 [`crate::host::adapter::HostBackend`] 之后，任何需要原生扩展的调用点都
    /// 必须显式 `reaper()` 并处理 `None` —— REAPER 依赖变成**可见且可能力门控**的一步，
    /// 而不是一个默认就在的具体类型。
    host: Mutex<Option<crate::host::adapter::HostBackend>>,
    host_geometry: Mutex<Option<CachedHostGeometry>>,
    /// 每个真实单region播放实例的有效mute；process仅读原子，不查询宿主或清修音参数。
    host_item_muted: std::sync::atomic::AtomicBool,
    /// 宿主在 `setupProcessing` 声明的采样率（Hz；`0` = 尚未声明）。
    ///
    /// 【为什么必须记下来】`process` 此前用 `ProcessContext::sample_rate` 选快照，且只认
    /// 44.1/48k，其余一律 `kResultFalse` —— 而 `setupProcessing` 已接受 44.1..192k。两者
    /// 不一致时（96k 宿主、或宿主在两处报出不同率）音频线程会持续返回错误码，用户听到
    /// 静音甚至被踢出链路。改按 `setupProcessing` 的权威率选快照，非原生率交给
    /// `PlaybackSnapshot::copy_block` 线性重采样（与 `setupProcessing` 的声明一致）。
    setup_sample_rate: AtomicI32,
    /// 宿主在 `setBusArrangements` 协商出的扬声器编排（`kMono` = 1 / `kStereo` = 3）。
    ///
    /// 【为什么必须记下来】`setBusArrangements` 接受 mono，但 `getBusArrangement` 此前
    /// **无条件**回 `kStereo` —— 对宿主撒谎。宿主据此重新协商或按错误声道数组织缓冲，
    /// 单声道轨（人声轨极常见）上表现为编排反复或静音。默认 `kStereo`。
    bus_arrangement: AtomicU64,
    prepared: Mutex<Option<PreparedVersion>>,
    /// 测试专用：`prepare()` 真正进入重排的次数。
    ///
    /// 【为什么需要它】"渲染输入失效必须通知**全部** owner"这条不变量，在单实例下
    /// 无法与"只通知自己"区分（`self` 恰好就是播放实例）。只有让编辑实例触发、断言
    /// 播放实例被通知，才能锁住回归。计数在 `prepare()` 的 `renders_playback()` 之后
    /// 自增 —— 那正是此前漏掉播放实例的位置。
    #[cfg(test)]
    prepare_requests: AtomicU64,
    pub snapshots: [super::snapshot::SnapshotPublisher; 2],
    /// 活动 Capture take 的**未处理** PCM，与处理结果同形（一份 `PlaybackSnapshot`）。
    ///
    /// 【为什么需要它】处理结果要经过神经合成（数秒），而"录完就想听到东西"是用户的
    /// 最低预期。有了这一份，输出通路的兜底顺序就是"处理后 → 原始 → 区间外直通"，
    /// 于是**任何**中间状态下都有声音；"静音"于是只剩"确实没有素材"这一个含义。
    ///
    /// 【为什么与 `snapshots` 分开而不是塞进去】两者更新节奏不同：处理结果随曲线编辑
    /// 重发，原始磁带只在 take 变化时重发。合成失败或还没完成时这一份仍然有效。
    pub raw_tape: super::snapshot::SnapshotPublisher,
    /// 本块输出源的稳定 token（种类 + 快照代次）；音频线程据此识别源切换。
    ///
    /// 【为什么按流分开】prefetch 与 realtime 的块会交替到达；共用一个 token 会让
    /// prefetch 的块改写 realtime 流看到的"上一块源"，于是实时流在**没有真正切换**
    /// 的时刻做一次淡化（或反过来漏掉真正的切换）。与 `declick` 同一粒度。
    output_token: [AtomicU64; 3],
    /// 每条进程流的输出接缝状态。
    declick: [OutputDeclickState; 3],
    /// 【B0】输出接缝的块账本：音频线程只写原子量，UI 线程读它打印 `[click]`。
    ///
    /// 【为什么它是账本而不是计数器】"咔"的五个可能机理（假静音、空回退、线性交叉
    /// 淡化、首块原样输出、位置跳变）只能靠**同一块上的多个读数**互相区分。只数
    /// "发生过几次"的计数器永远回答不了"这一次是哪一种"。
    click: crate::render::click::ClickLedger,
    /// 【B0】本块分层拷贝的覆盖读数（音频线程写，`apply_output_declick` 读走）。
    ///
    /// 【为什么按流分开】prefetch 与 realtime 交替到达；共用一格会让一条流的
    /// "未覆盖帧数"被另一条流当成自己的。
    layered_probe: [LayeredProbeCell; 3],
    /// 已发布处理结果对应的**素材代次**（`0` = 还没发过）。
    ///
    /// 【为什么单独记而不是从 `capture_prepared` 里拆位】那个版本号是
    /// `revision ^ (edit << 32)` 的位运算结果；音频线程要回答的是"已发布的那份是不是
    /// **当前**素材"，一个显式读数比拆位可靠得多（也免得代次增长过 32 位时出错）。
    capture_ready_revision: AtomicU64,
    /// 已发布原始磁带对应的素材代次（`0` = 还没发过）。
    raw_tape_revision: AtomicU64,
    /// 上一次"有素材但没有处理结果"已经警告过的素材代次（`0` = 还没警告过）。
    ///
    /// 【为什么按代次只报一次】这是一个**状态**，不是事件：每 100ms 心跳都写一行
    /// 日志会把日志刷满，而按代次去重之后，一次真正的卡死只会留下一条可检索的记录，
    /// 合成成功后自然不再出现。
    pending_reported_revision: AtomicU64,
    /// 每条进程流自己的块几何。**绝不跨流比较**。
    ///
    /// 【为什么必须分开】SDK 允许 `kRealtime` 与 `kPrefetch` 交替调用同一实例
    /// （`ivstaudioprocessor.h:138-143`）。共用一份 `last_block_start` 时，prefetch
    /// 的位置移动会让实时流的“同一块重复”被误判成“在前进”，门控于是反复放行同一
    /// 小段（用户听到的“开头 0.7 秒反复播放”）；捕获的 `expected_next` 同样会被
    /// 两条流撕成每块一段。
    streams: [StreamGeometry; 3],
    /// 仅实时流的最近一次推进结论（兼容旧测试与诊断；ARA 实例恒为假）。
    realtime_flow: AtomicBool,
    /// 实时流上一次**实际写出内容**的区间；用于抑制停播光标处的同区间重复。
    rt_last_emitted_start: AtomicI64,
    /// 【T3】实时流上一次**实际读快照**的位置（用于发现读游标跳变）。
    rt_last_read_position: AtomicI64,
    /// 【T3】上一次实际读的**帧数**：判据必须拿"上一块起点 + 它的帧数"当预期续接，
    /// 否则任何正常推进都会被算成跳变（那是"每块都直通"的成因）。
    rt_last_read_frames: AtomicU32,
    /// 本块要求做一次**强制**接缝淡化（同一源内部的读位置跳变）。
    force_declick: [AtomicBool; 3],
    /// 上一次实际写出内容时所在的 **pass 序号**（`StreamGeometry::pass`）。
    ///
    /// 【为什么需要它】`read_position_jump` 的判据是"本块起点 != 上一块起点 + 上一块
    /// 帧数"。跨 pass 时那个差值必然巨大（一次起播/回退），但**那不是读游标跳变**：
    /// pass 边界本来就要重新开始，而上一块输出的尾巴属于**另一个位置**的内容。拿它
    /// 做交叉淡化等于把两段不相干的信号叠在一起 —— 反而制造了一次咔。
    rt_last_read_pass: AtomicU64,
    rt_last_emitted_frames: AtomicU32,
    rt_last_emitted_valid: AtomicBool,
    /// 每条进程流的上一块 `kRecording` 位（用于上升沿覆盖写）。
    stream_recording: [AtomicBool; 3],
    /// 块级诊断计数：收到的块、`kPlaying` 块、位置推进但未置播放位的块、被门控清掉的块。
    blocks_seen: AtomicU64,
    blocks_playing: AtomicU64,
    blocks_flow_without_playing: AtomicU64,
    gate_cleared: AtomicU64,
    /// ARA 实例在快照未就绪时退到 layered（宿主原声兜底）的块数。
    ///
    /// 【为什么必须是诊断】它等于"本该由我们出声、实际让宿主原声顶替"的次数。
    /// T9/T10 修好后这个数应该长期为 0；一旦上升，说明快照清空/合成丢弃又回来了。
    bound_snapshot_fallback: AtomicU64,
    /// 按进程流分开的同一组计数。
    stream_blocks: [AtomicU64; 3],
    stream_playing_blocks: [AtomicU64; 3],
    stream_flow_without_playing: [AtomicU64; 3],
    stream_gate_cleared: [AtomicU64; 3],
    /// 每流的相位计数与位置来源计数（诊断）。
    stream_resting_blocks: [AtomicU64; 3],
    stream_rolling_blocks: [AtomicU64; 3],
    continuous_used: AtomicU64,
    music_used: AtomicU64,
    /// 【I-2】上一次输出仲裁的档位，以及"是否已经记录过"（`0` 是 `Untouched` 的编码，
    /// 不能兼作"从未记录"）。
    rung_previous: AtomicU32,
    rung_valid: AtomicBool,
    /// 档位切换的总次数 / 最近一次切换的两端 / 本次 pass 内的切换次数。
    rung_switches: AtomicU64,
    rung_last_from: AtomicU32,
    rung_last_to: AtomicU32,
    pass_switches: AtomicU64,
    /// 【I-2】本 pass 锁定的内容源（`0` = 未锁定）。
    ///
    /// 【为什么必须锁存】旧实现每块都重新仲裁，于是 `Raw → Processed` 可以发生在
    /// **音乐中途** —— 一次换源就是一次咔，而它恰好出现在"刚录完、后台合成刚好跑完"
    /// 的那一刻，也就是用户起播后的 0.4–0.7 s。锁存之后，一次播放只有一个内容源，
    /// 换源只可能发生在 pass 边界（那时本来就要重新开始）。
    rung_locked: AtomicU8,
    /// 锁存所属的 pass 序号（与 `StreamGeometry::pass` 同一口径）。
    rung_pass: AtomicU64,
    position_resyncs: AtomicU64,
    cycle_wraps: AtomicU64,
    cycle_wrap_unknown: AtomicU64,
    cycle_splits: AtomicU64,
    /// 因“同一区间重复”被抑制输出的实时块数。
    /// 【T3】读游标被纠正的次数（pass 边界、宿主纠正记账、流交接）。
    ///
    /// 【为什么必须是诊断】输出侧按块首 `position` 读快照，而捕获侧另有自己的一套
    /// 记账（`expected_next`）。两者出现几帧到几十帧的差时，读游标就会跳一次 ——
    /// 回退=重播已听内容，前进=跳过内容。记下来才能在下一次复现时回答"跳了多少帧"。
    position_corrections: AtomicU64,
    /// 【T3】最近一次纠正的帧差（绝对值）。
    max_correction_frames: AtomicU64,
    /// 【T-S1-6】最近一次发布里，素材**带覆盖信息**的源有几个洞、洞有多少帧。
    ///
    /// 【为什么必须可读】"起播 0.4~0.7 s 一次咔"的直接原因是素材开头有一段我们
    /// 没录到的洞，而那段洞在**处理结果**里一度被当成"我们录到的静音"。把洞的个数
    /// 与总帧数打出来之后，"这里到底有没有洞"不再需要靠耳朵判断 —— 而 `holes=0`
    /// 正是"洞假设不成立、该去查别的原因"的判据。
    tape_holes: AtomicU64,
    tape_hole_frames: AtomicU64,
    /// 上一次打印的素材覆盖读数（去重用）。
    last_tape_diag: Mutex<Option<String>>,
    /// `[capture] pass` / `[capture] clock` 两行的去重指纹（非实时线程用）。
    last_pass_diag: Mutex<Option<String>>,
    last_clock_diag: Mutex<Option<String>>,
    /// `[capture] ctx` 的去重指纹（非实时线程用）。
    last_ctx_diag: Mutex<Option<String>>,
    /// 输出写满审计的读数（`HIFISHIFTER_BUFFER_AUDIT=1`；音频线程只写原子）。
    buffer_audit_channels: AtomicU32,
    buffer_audit_frames: AtomicU32,
    buffer_audit_unwritten: AtomicU64,
    buffer_audit_active: AtomicBool,
    last_buffer_diag: Mutex<Option<String>>,
    repeats_suppressed: AtomicU64,
    /// 连续多少块被判定为"不该出声"（停播 / 重复区间）。
    ///
    /// 【为什么需要计数而不是逐块判定】逐块判定的后果是：播放中途只要**一块**被
    /// 判成 Resting（一次调度抖动、一次宿主复用同一区间），就会触发一次
    /// `apply_gate_fade_out` —— 那是 64 帧淡到零、下一块再淡回来（约 1.3 ms 的掉零）。
    /// 听感正是"播放过程中随机地咔"。真实停播会**连续**满足判据，所以用"连续两块"
    /// 作为分界：单块抖动继续出声，连续不推进才静音（代价是停播后多 10 ms 声音）。
    gate_resting_run: AtomicU32,
    /// 宿主在 `setupProcessing` 声明的单块上限（`0` = 未声明）。
    max_samples_per_block: AtomicI32,
    /// 音频线程发现值得留证的起播窗口时置位；actor 心跳消费并落一次日志。
    trace_wanted: AtomicBool,
    /// 本次可疑起播窗口是否已经落过日志（避免每 100ms 刷同一段 trace）。
    trace_logged: AtomicBool,
    /// 预分配的块级环形记录（非音频线程初始化；音频线程只写，不分配）。
    block_trace: std::sync::OnceLock<Box<BlockTraceRing>>,
    /// 音频线程发现"有素材、但处理结果不是当前代次"时置位；actor 心跳消费它并
    /// 重新排一次渲染（见 [`Self::capture_stage`]）。
    ///
    /// 【为什么音频线程只置位、不调度】合成要数秒且要取锁/分配，绝不能进音频线程。
    /// 这里只是一次原子写；真正的调度仍由会话 actor 的渲染任务完成 —— 它自带
    /// 在途检查、取消与重试预算，比从音频线程另开一条合成通路安全得多。
    render_wanted: AtomicBool,
    pub(crate) edits: Arc<Mutex<crate::state_channel::EditState>>,
    pending_restore: Mutex<Option<crate::state_channel::EditState>>,
    /// 等待账：起点与签名（见 [`RestoreProgress`]）。**唯一**的时间来源。
    restore_progress: Mutex<RestoreProgress>,
    /// 报过"没等到"之后的下一次重试时刻（`None` = 不在退避中）。
    restore_next_attempt: Mutex<Option<std::time::Instant>>,
    /// 已尝试合并的次数（**只是诊断**：不再参与任何预算判断）。
    restore_attempts: std::sync::atomic::AtomicU32,
    /// 本实例最近一次**成功合入**的载荷摘要（T-D5）。
    ///
    /// 【为什么记摘要】同一份存档会被宿主重复交回（重开编辑器、撤销后重做、文档重建）。
    /// 已经合入过的那一份再走一遍 reconcile 只会重复劳动，并在失败时重新点燃横幅。
    /// 摘要相同且手里没有未合入载荷 ⇒ 直接认作已合入。
    restore_applied_digest: Mutex<Option<u64>>,
    /// 最后一次成功编码的 ARA `EditState` 字节。
    ///
    /// 【为什么需要】文档已经关闭（`bound_once == true && document == None`）时，
    /// 宿主仍可能调用 `getState`。此前会落到 Capture 分支，序列化一份从未被 ARA
    /// 写过的空 `EditState`，把用户的 ARA 编辑覆盖掉。这里保留最后一份成功载荷。
    last_ara_state: Mutex<Option<Vec<u8>>>,
    /// 后台准备队列。
    ///
    /// 【为什么不是 `OnceLock`】`stop_editor()` 会 `close()` 它（`mailbox.closed`
    /// 永久置真且 worker 已 join），而 `OnceLock::get_or_init` 在文档关闭后重绑时
    /// 只会把**同一个死队列**交回来 —— 重绑成功的实例可以收 region、可以绑定，
    /// 却永远排不出快照（`request()` 恒返回 `"host preparation closed"`），
    /// 表现就是重开工程后时间线空、播放无声。队列因此必须是可替换的。
    preparation: Mutex<Option<Arc<super::preparation::PreparationQueue>>>,
    prepare_owner: Mutex<Option<std::sync::Weak<ExtensionOwner>>>,
    pub(crate) clock: std::sync::OnceLock<Arc<super::transport::TransportClock>>,
    /// 最近一次清单刷新得出的宿主音频读数；GUI 载荷读它（见 [`HostAudioStatus`]）。
    host_audio: Mutex<HostAudioStatus>,
    /// 上一次"重"宿主枚举（几何 + 清单）的时刻。
    ///
    /// 【为什么把时钟与枚举分开节流】UI 定时器是 20ms（50Hz）。播放光标必须跟得上
    /// （`host.sample` 很便宜），但几何/清单枚举即使有代次缓存，每 tick 仍要付
    /// `GetProjectStateChangeCount` 等若干宿主调用。50Hz 下这些调用在真实工程里会累积
    /// 成可观的 UI 线程占用（且与 actor/渲染线程争同一把 document 事务）。所以时钟保持
    /// 20ms，枚举降到 [`HOST_ENUMERATION_INTERVAL`]。宿主真的变了时，下一次节流点就会
    /// 捕捉到（≤100ms 延迟），用户不可感。
    last_enumeration: Mutex<Option<std::time::Instant>>,
    /// 上一次视图定时器 tick 的时刻（见 [`Self::refresh_host_transport_tick`]）。
    ///
    /// 【为什么需要它】宿主把插件放到后台时，20 ms 的视图定时器会被挂起或大幅降频；
    /// 回到前台后的第一次 tick 与上一次之间会出现一个远大于 20 ms 的间隔。那个间隔
    /// 是"刚刚恢复前台"的**本地**证据 —— 不依赖宿主是否重发 `setActive`，也不依赖
    /// 子窗口收不收得到 `WM_ACTIVATEAPP`（本编辑器窗口是 `WS_CHILD`，收不到它）。
    last_view_tick: Mutex<Option<std::time::Instant>>,
    /// 上一次对外报告过的绑定读数（见 [`BindingReport`]）。
    binding_report: Mutex<Option<BindingReport>>,
    /// 连续"结构性绑定失败"的次数（见 [`ExtensionOwner::enumeration_interval`]）。
    ///
    /// 【为什么必须退避】`HOST_ENUMERATION_INTERVAL`（100ms）是研究"宿主刚改了几何"
    /// 定出来的节奏，把它用在**永不成功**的重试上，等于每 110ms 完整枚举一遍宿主工程
    /// （folder 树 + 每条后代轨道的 item × take）。实测 35 分钟 ≈ 22,000 次 ——
    /// 日志刷屏只是这件事可见的那一部分。
    enumeration_backoff: std::sync::atomic::AtomicU32,
    /// 上一次结构性失败时看到的宿主几何代次；用于"工程真变了就立刻回到全速"。
    ///
    /// 【为什么不能省】退避不能牺牲响应速度：用户拖完 item 必须立刻看到新几何，
    /// 而不是等 4 秒。代次一变就清零档位，这条是唯一的保证。
    last_failed_revision: Mutex<Option<Option<i32>>>,
    /// Capture 形态的编辑器会话（**没有** ARA 文档的那一份）。
    ///
    /// 【为什么挂在 owner 上而不是文档上】文档是 ARA 的容器，而 Capture 实例永远没有
    /// 文档 —— 会话必须有一个不依赖文档的归属。它与 ARA 会话是同一个类型：时间线、历史、
    /// 参数、分析、命令分发全部共用，差别只在 `SessionBacking`（见 `editor::session`）。
    capture_session:
        std::sync::OnceLock<Result<Arc<crate::editor::session::EditorSession>, String>>,
    /// 组件 state 里恢复出来、但**会话还没创建**的 Capture 用户编辑。
    ///
    /// 【为什么需要 owner 级暂存】`IComponent::setState` 到达时用户可能从没开过编辑器
    /// 窗口 —— 那一刻没有会话可投递。而 `getState` 又**不能**为了存档去创建一个会话
    /// （那会拉起 actor 与文件 I/O）。所以字节块先落在这里，由
    /// [`Self::capture_session`] 在会话真正创建时补投一次。
    pending_capture_edits: Mutex<Option<Vec<u8>>>,
    /// 宿主调用 `setActive(true)` 的时刻；`None` = 还没激活。
    ///
    /// 【为什么必须记下来】ARA 规定 `bindToDocumentControllerWithRoles` 只能在
    /// `setActive` **之前**调用。宿主一旦激活组件就再也不会给这个实例绑定机会 ——
    /// 那一刻之后仍无绑定，就是"宿主把本插件当普通插入"的确证，也就是 Capture 形态
    /// （见 [`InstanceMode`]）。此前这件事只能靠读者对比两个会话的日志才能推断出来。
    set_active_at: Mutex<Option<std::time::Instant>>,
    /// "激活后仍无 ARA 绑定"这条诊断是否已经报过（一次性）。
    ara_binding_absent_reported: std::sync::atomic::AtomicBool,
    /// Capture 通路的捕获存储。
    ///
    /// 【为什么每个实例一份而不是进程一份】同一文档上可能有两个实例，一个在录、
    /// 一个在放；共享存储会让它们互相覆盖对方的 take。
    capture: crate::capture::CaptureStore,
    /// 最近一次为 Capture 发布快照的 take 号（`0` = 还没发过）。
    ///
    /// 【为什么需要它】`prepare()` 由 UI 定时器每 20ms 驱动一次，而一次合成要好几秒。
    /// 没有这道闸，同一次录制会被反复排队 —— 每次都在音频线程之外占满 CPU，
    /// 用户看到的是界面变卡而声音没有任何变化。
    capture_prepared: AtomicU64,
}

/// `setActive(true)` 之后允许 ARA 绑定出现的宽限时间（见 [`ExtensionOwner::note_editor_tick`]）。
///
/// 【为什么不是 0】有些宿主在 `setActive` 前后几毫秒内完成绑定；立刻报"没有绑定"会
/// 产生假警报。5 秒足以覆盖实测过的全部宿主，而真出问题时用户也不会等得不耐烦。
const ARA_BINDING_GRACE: std::time::Duration = std::time::Duration::from_secs(5);

/// 宿主几何/清单枚举的节流间隔（见 `ExtensionOwner::last_enumeration`）。
const HOST_ENUMERATION_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

/// 强制重采宿主清单的最长间隔（见 `DocumentSession::ui_inventory_sampled_at`）。
///
/// 【为什么需要一条与 change count 无关的下限】`ui_inventory_stamp` 早退以
/// `GetProjectStateChangeCount` 为主。方向（倒放）/ 声道模式是 **take / item 级**属性，
/// 宿主未必为它们推进项目 change count；只靠令牌早退会让"倒放 Take"连采样这一步都被
/// 跳过（`render_facts` 不更新 ⇒ 渲染输入不失效 ⇒ 波形变了、声音没变）。超过这个间隔
/// 就强制重采一轮，保证任何 take 级事实变化都必然穿透早退。
///
/// 【为什么是 1s】宿主真变了时令牌仍走原路（≤100ms 捕捉）；这条只兜住"令牌不动"的
/// 情况，把最坏延迟钉在 1s，同时把空闲时的完整枚举压到 1Hz —— 相比 50Hz 定时器可以
/// 忽略，又足以让用户在放手后立刻看到倒放生效。
pub(crate) const FACTS_REFRESH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

/// 超过这个时间没有成功完整采样过，就认为 UI 定时器曾停摆（后台/挂起）。
///
/// 【为什么需要】`resume_resync` 只挂在 `IComponent::setActive(true)` 上，而宿主
/// 仅切换窗口焦点时不会再调 `setActive`。后台期间定时器被降频或停摆后，前台回来
/// 的第一轮仍会命中"令牌未变 + 1s 内采样过"的早退，`ui_tracks` / `region_items`
/// 停在暂停前的快照 —— 用户看到 Clip 重新变成"等待宿主完成音频分配"。
/// 与 `FACTS_REFRESH_INTERVAL` 同量级：正常路径每 1s 必然重采一次，超过它说明
/// 定时器确实漏跑了。
const STALE_SAMPLE_GRACE: std::time::Duration = std::time::Duration::from_secs(1);

/// 视图定时器两次 tick 之间超过这个间隔，就认为它曾停摆（宿主在后台）。
///
/// 【为什么是 500 ms】正常周期是 20 ms；25 个周期足够把"一次调度抖动"与
/// "进程被挂起/降频"分开，又短到用户切回前台后的第一帧就能触发重同步。
/// 与 `STALE_SAMPLE_GRACE` 分工不同：后者是"采样数据陈旧"，这条是"**事件源**
/// 曾经停止"。前者可以靠其它路径补，后者只能由恢复后的第一次 tick 自己发现。
const FOREGROUND_RESUME_GAP: std::time::Duration = std::time::Duration::from_millis(500);

/// 连续多少块"不推进"才允许门控静音（见 `ExtensionOwner::gate_resting_run`）。
///
/// 【为什么是 2】真实停播会**连续**满足判据；而一次调度抖动、一次宿主复用同一区间
/// 只会命中**单块**。逐块判定会把后者变成一次 1.3 ms 的掉零 —— 用户听到的就是
/// "播放过程中随机地咔"。取 2 的代价是停播后多出约一个块（10 ms）的声音，
/// 听感上察觉不到。
const GATE_SETTLE_BLOCKS: u32 = 2;

/// VST3 离线 `setupProcessing` 里等待后台准备的最长时间。
///
/// 【为什么是 15 s 而不是更短】这个调用在宿主的 prepare 线程/UI 线程上同步等待，而
/// 离线渲染（冻结 / 胶合 / 导出）会**把当时听不到的声音烘焙进产物**：等得不够久 ⇒
/// 产物里是静音，这是数据正确性失败，比 UI 卡一下严重得多。因此宁可等。15 s 只是
/// **上限**：正常路径上编辑已经触发过后台准备，这里通常只等剩余的推理时间。
///
/// 【为什么不更长】180 s 会把宿主界面卡死三分钟。真正消除这次同步等待需要异步完成 +
/// 进度回报（独立一期）；在那之前，收敛失败由 [`ExtensionOwner::prepare_offline_until`]
/// 的尝试次数上限提前退出，不必空等到这个上限。
pub(crate) const OFFLINE_PREPARE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

/// 一份合法的空 `EditState` 字节；编码失败时返回空块（`restore` 把空块当 no-op）。
fn default_edit_state_bytes() -> Vec<u8> {
    crate::state_channel::EditState::default()
        .encode()
        .unwrap_or_default()
}

#[cfg(test)]
mod assigned_timeline_tests;
#[cfg(test)]
mod bound_tests;
#[cfg(test)]
mod capture_output_tests;
#[cfg(test)]
mod editor_document_tests;
#[cfg(test)]
mod extension_roles_tests;
#[cfg(test)]
mod host_audio_state_tests;
#[cfg(test)]
mod host_geometries_tests;
#[cfg(test)]
mod region_geometry_match_tests;
