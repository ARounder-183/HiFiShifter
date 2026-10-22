//! 每个 VST3 entry 的扩展所有权。强引用由 entry builder 保留，不在组件销毁时悬空。

use super::ownership::{region_owners, RegionKey};
use crate::sync::MutexExt;
use crate::capture::recorder::{
    BlockContext, BlockSplit, PositionSource, ProcessStream, SubBlock, TransportPhase,
};
use ara2_bridge::core::{ApiGeneration, AraError};
use ara2_bridge::plugin::{ExtensionBinding, ExtensionRoles};
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{
    AtomicBool, AtomicI32, AtomicI64, AtomicU32, AtomicU64, AtomicU8, Ordering,
};
use std::sync::{Arc, Mutex};

/// 只读元数据供非实时读者消费；UI/model采集，音频线程不访问此缓存或锁owner。
///
/// `value` 是**逐 region** 的绑定列表，不是一个实例一个几何：挂在 REAPER folder
/// 轨上的 FX 会被 ARA 分配整组 region，而 `parent(2)`（实例自己的 take）结构上只
/// 能覆盖其中一个。见 [`ExtensionOwner::host_geometries`]。
#[derive(Clone)]
struct CachedHostGeometry {
    model: u64,
    scope: u64,
    change: Option<i32>,
    value: Result<Vec<crate::host::geometry::BoundHostGeometry>, String>,
}

/// 一次 region → 宿主 item 绑定求解的**可判等读数**。
///
/// 【为什么需要指纹】"2 个 region 对 1 个候选 item"是 **folder 父轨上的结构性不匹配**：
/// 宿主工程不变，它就永远匹配不上。而无条件日志会跟着 UI 定时器以 10Hz 无限重放 ——
/// 实测 2026-10-12 那一次 35 分钟输出了 16,769 条，把三个 8MB 轮转件全部写满。
/// 指纹相同即"没有任何新信息"，不输出。
///
/// 【为什么不放几何本身】几何每次刷新都可能有微小差异（宿主浮点读数），放进去会让
/// 指纹永远不同、去重彻底失效。这里只放**身份与数量**：哪些 region 被分配、哪些绑上了、
/// 哪些没绑上、候选有多少个。
#[derive(Clone, PartialEq, Eq)]
struct BindingReport {
    assigned: Vec<u64>,
    bound: Vec<u64>,
    unmatched: Vec<u64>,
    dropped: usize,
    candidates: usize,
}

/// 绑定求解失败的两条**结构性**措辞（产生点与分类点共用同一常量）。
///
/// 【为什么不用字面量】与 `editor::errors` 的终态常量同一条纪律：产生点与分类点
/// 分处不同函数，两边各写字面量时任何一处改措辞都会让分类**静默失效**（退避停摆，
/// 或对状态变化误退避），且没有编译期信号。
const BINDING_NO_UNIQUE_ITEM: &str = "no assigned ARA region matched a unique host item";
const BINDING_GEOMETRY_INCOMPATIBLE: &str =
    "direct take geometry incompatible with assigned ARA playback window";

/// 绑定失败是否属于"由宿主工程结构决定、不会自行变化"的一类。
///
/// 【为什么必须区分】只有这一类才值得退避：它是"重试一万次也一样"的读数。授权撤销、
/// 文档关闭、扩展接口消失都是**状态变化**，它们要么很快自愈、要么已经是终态 ——
/// 对它们退避只会拖慢恢复。
///
/// 【为什么用"包含"而不是全等】与 `editor::errors::classify_error` 同一理由：措辞
/// 可能带上下文前缀（`host split: …`），常量保证的是**子串出现**。
fn is_structural_binding_failure(error: &str) -> bool {
    error.contains(BINDING_NO_UNIQUE_ITEM) || error.contains(BINDING_GEOMETRY_INCOMPATIBLE)
}

/// 两个宿主/ARA 浮点量是否相容；判据全局唯一，见 `host::geometry::host_value_compatible`
/// （绑定 / 分割规划 / 写回前置检查 / 剪贴板共用，避免"规划接受、绑定拒绝"的不对称）。
fn compatible(a: f64, b: f64) -> bool {
    crate::host::geometry::host_value_compatible(a, b)
}

/// 在候选宿主 item 里为一条 ARA region 找**唯一**匹配的几何。
///
/// 【为什么必须"唯一才采信"】ARA 侧没有 region → item 的显式边（见
/// `crate::ara::mapping` 的 `LOST_FIELDS`）：`parent(2)` 只覆盖 FX 自己的那一个
/// take，源 persistentID 会被多个 item 共享，名字/轨道序号按既有纪律不得用于猜
/// 关联。因此只能按**几何**比对；而后续所有宿主写回都沿这条边定位 item —— 猜错
/// 就是写到别的 item 上。所以 0 个匹配或多个匹配一律返回 `None`：宁可不绑定。
///
/// 四项都要相容：播放位置、播放时长、源内起点、播放速率。只比时间窗会在
/// "同一位置的两条子轨各有一个 item"这种常见布局上直接产生歧义。
///
/// 【`preferred` 是消歧而不是放宽】`preferred` 是这条 region **上一次**绑定的 item GUID。
/// 复制粘贴的安全副本、同一素材的多个 item 会给出多个几何全等的候选；纯"唯一才采信"
/// 会把它们**全部**判为歧义、两个都丢（用户看到片段凭空消失）。当且仅当候选里恰好有
/// 一个 GUID 等于上次绑定时采信它 —— 这是**稳定性**，不是猜测：上一次的绑定本身就来自
/// 一次唯一匹配，保持它不会把写回引到新的对象上。
fn unique_geometry_for_region(
    region: &crate::ara::AraPlaybackRegion,
    candidates: &[crate::host::geometry::HostClipGeometry],
    preferred: Option<&str>,
) -> Option<crate::host::geometry::HostClipGeometry> {
    let playback_rate = if region.duration_in_playback_time > 0.0 {
        region.duration_in_modification_time / region.duration_in_playback_time
    } else {
        f64::NAN
    };
    let matched: Vec<&crate::host::geometry::HostClipGeometry> = candidates
        .iter()
        .filter(|candidate| {
            compatible(candidate.start_sec, region.start_in_playback_time)
                && compatible(candidate.duration_sec, region.duration_in_playback_time)
                && compatible(
                    candidate.source_start_sec,
                    region.start_in_modification_time,
                )
                && compatible(candidate.playback_rate, playback_rate)
        })
        .collect();
    let first = *matched.first()?;
    if matched.len() == 1 {
        return Some(first.clone());
    }
    // 有歧义：只有"上次绑定的那个"能打破它 —— 且它必须在候选里。
    if let Some(preferred) = preferred {
        if let Some(found) = matched
            .iter()
            .find(|candidate| candidate.item_id == preferred)
        {
            return Some((*found).clone());
        }
    }
    None
}

#[derive(Clone, PartialEq, Eq)]
struct PreparedVersion {
    model: u64,
    edit: u64,
    epoch: u64,
    scope: u64,
    keys: Vec<u64>,
}

/// 宿主音频是否已经到达本实例 —— **语言无关**的分类，文案由前端 catalog 本地化。
///
/// 【为什么不是"有没有音频"这个布尔】用户侧有两种完全不同的处境，处置方式也不同：
/// 插件挂在了 folder 父轨上是**用法问题**，必须给出具体指引（REAPER 按轨道管理 ARA
/// 插件）；普通等待则是宿主还没分配 region，只需说明"等待中"。把两者压成一个布尔，
/// 前端就只能给出一句对谁都不准的泛泛提示 —— 而"轨道和 item 都看得见、能拖、但没
/// 内容"这种症状极具误导性，泛泛提示等于没说。
///
/// 【措辞红线】不得表述为"ARA 规范不支持跨轨"：ARA 2.0 规范**允许**一个实例服务
/// 多个 region sequence，是 **REAPER 选择按轨道管理 ARA 实例**。错的解释会把日后
/// 真正可行的改进方向带偏。
/// 本实例这一轮"该看到哪些 region"的结论，连同**它是怎么得出来的**。
///
/// 【为什么要把来源也带出来】"宿主没给"与"宿主给了空的"在界面上长得一模一样，
/// 但处置完全相反：前者是**暂缓**（下一次回调会补上），后者才是**真空**（这份工程
/// 真的没有 region）。只返回一个集合，调用方只能二选一地猜 —— 那正是"重开工程后
/// 界面停在等待宿主音频、却没有任何可行动提示"的成因之一。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScopeSource {
    /// 宿主明确告知过（`addPlaybackRegion` / 分配观察器）。
    HostAssignment,
    /// 宿主还没告知，从文档级 `sequence_regions` 派生（只服务并集语义的调用方）。
    DocumentDerived,
    /// 宿主告知过、结果确实是空的；或派生之后仍然为空。
    Empty,
}

impl ScopeSource {
}

/// [`ScopeSource`] 与 region 集合的配对。
#[derive(Clone, Debug)]
pub(crate) struct AssignedScope {
    pub keys: Vec<u64>,
    pub source: ScopeSource,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum HostAudioState {
    /// 没有等待中的 item（含空工程）—— 无可抱怨。
    #[default]
    Ready,
    /// 清单里有 item，但本实例一个 region 都没拿到：宿主尚未分配。
    AwaitingRegions,
    /// 同上，且本 FX 所在轨道是 folder 父轨：组内子轨的音频不会交给这个实例。
    FolderParentWithoutRegions,
    /// 宿主**分配了** region，但没有一条能绑到唯一宿主 item。
    ///
    /// 【为什么必须与 `FolderParentWithoutRegions` 分开】两者的用户可见症状相同（片段
    /// 是空的、能拖、没内容），但事实不同：那一态是"一个 region 都没拿到"（宿主尚未
    /// 分配，或 FX 挂在 folder 父轨上）；这一态是"拿到了，但按几何唯一匹配对不上"
    /// （folder 父轨的 FX 会拿到整组 region，而候选 item 数少于 region 数，或几何不
    /// 相等）。混为一谈会让提示说不准。
    ///
    /// 【为什么必须让用户看见】这个读数此前**只存在于日志里** —— 而且是刷屏的那两条
    /// （见 [`BindingReport`]）。用户界面上没有任何对应物：时间线是空的，没有任何说明。
    /// 把它接进状态链之后，日志可以安静，用户拿到的是**一句可读的说明**。
    UnmatchedRegions,
    /// region 已分配，但宿主还没开启样本访问。
    AwaitingSamplesAccess,
    /// 宿主开启了访问，但样本读取失败（真故障，终态）。
    SourceReadFailed,
    /// 宿主交付了素材，但本插件**拒绝**了 region 创建（真故障，终态）。
    ///
    /// 【为什么必须与 `AwaitingRegions` 分开】"等待"暗示用户应该继续等，而这一态
    /// 说的是"插件自己拒了，再等也不会来"。混为一谈会把一个可行动的故障显示成
    /// 一个无限期的等待 —— 用户唯一能做的是重载插件，而他不知道原因。
    ///
    /// 【为什么 T4/T5 修好后应该永不出现】跨文档认领 + 强制销毁之后，
    /// `create_playback_region` 不再因为遗留身份而失败。真出现时
    /// `DocumentSession::region_rejections` 里带着第一条原因。
    RegionRejected,
    /// 宿主与绑定都正常，但本插件未能把上一次保存的状态合并进来（可行动）。
    ///
    /// 【为什么必须与 `Ready` 分开】这一态的音频通路是通的（时间线会出来、播放有声），
    /// 唯独参数/曲线停在上一次成功合并的版本。用户看到的不是"没声音"，而是"设置没回来"——
    /// 而此前这件事只写一行 `log::warn!`，界面上完全不可见。
    RestoreUnresolved,
    /// 本实例根本没有可用的 ARA renderer / 文档绑定。
    ///
    /// 【为什么必须单列】"等待宿主分配 region"暗示"继续等会来"；这一态说明
    /// bind 失败、文档控制器缺失或旧文档仍占着实例 —— 再等也不会来，用户唯一
    /// 能做的是重载插件。此前界面沿用上一次的 `awaiting_regions`，把一个终态
    /// 显示成无限期等待。
    ///
    /// 【⚠ 当前没有生产者】`classify_host_audio` 的所有调用点都在"某份文档正在刷新"
    /// 的路径上，因此它永远拿不到"这个实例根本没有文档"这个事实。要真正产出这一态，
    /// 需要先回答一个产品问题：**捕获形态本来就没有文档**，若按"无文档 ⇒ NotBound"
    /// 分类，捕获用户会被告知"插件没有绑定到 ARA 文档"——那是误导。正确的判据是
    /// "这个宿主本应提供 ARA 却没绑上"，而那需要 `IHostApplication` 侧的证据。
    /// 前端（`hostAudio.ts` 的 `isNotBound` 与 `HostAudioNotice`）已经按这一态
    /// 做好了界面，所以**保留定义**而不是删掉；等生产者落地时只需接线。
    #[allow(dead_code)]
    NotBound,
}

impl HostAudioState {
    /// 语言无关的分类名；前端按此查 catalog（与 `ara_host_fields:` 同一原则）。
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            HostAudioState::Ready => "ready",
            HostAudioState::AwaitingRegions => "awaiting_regions",
            HostAudioState::FolderParentWithoutRegions => "folder_parent_without_regions",
            HostAudioState::UnmatchedRegions => "unmatched_regions",
            HostAudioState::AwaitingSamplesAccess => "awaiting_samples_access",
            HostAudioState::SourceReadFailed => "source_read_failed",
            HostAudioState::RegionRejected => "region_rejected",
            HostAudioState::RestoreUnresolved => "restore_unresolved",
            HostAudioState::NotBound => "not_bound",
        }
    }
}

/// 【T18】`reconcile` 身份失败的两种归属。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RestoreIdentityVerdict {
    /// 宿主的图还没长成（重开工程时 region 尚未交付）：继续等，报 `pending`。
    AwaitingHostGraph,
    /// 真失败（存档损坏、修订号耗尽）：报 `failed`。
    Fatal,
}
/// 最近一次清单刷新得出的宿主音频读数。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HostAudioStatus {
    pub state: HostAudioState,
    /// 清单里**尚未被任何已分配 region 认领**的 item 数（"还在等音频"的 clip 数）。
    pub waiting_items: usize,
}

/// 纯函数：由宿主侧事实判定状态。可独立单测，不需要 REAPER。
///
/// 判据只用可核实的事实：本 FX 轨是不是 folder 父轨、清单里有多少 item 认领到了
/// region、以及宿主**分配**了多少 region、其中多少条没绑上。不推断、不猜 ——
/// 读不出 folder 结构时调用方传 `false`，读不出绑定读数时两个 region 计数传 `0`，
/// 此时只会退化成"等待中"，不会给出一个可能指错方向的提示。
///
/// 【为什么 region 计数与 `claimed_items` 都要】`claimed_items` 来自**清单侧**（item
/// 被谁认领），region 计数来自**绑定侧**（求解结果）。"分配了但一条都没绑上"这个事实
/// 只有绑定侧有，而它正是 folder 父轨上片段凭空消失的成因。
pub(crate) fn classify_host_audio(
    fx_is_folder_parent: bool,
    item_count: usize,
    claimed_items: usize,
    regions_assigned: usize,
    regions_unmatched: usize,
) -> HostAudioStatus {
    let waiting_items = item_count.saturating_sub(claimed_items);
    let state = if regions_assigned > 0 && regions_unmatched > 0 && claimed_items == 0 {
        // 拿到了 region，但一条都没绑上 —— 比"还没拿到"更具体，优先报它。
        HostAudioState::UnmatchedRegions
    } else if claimed_items > 0 || item_count == 0 {
        // 有认领 = 正常；一个 item 都没有 = 没有"等待"可言（空工程不该报障）。
        HostAudioState::Ready
    } else if fx_is_folder_parent {
        HostAudioState::FolderParentWithoutRegions
    } else {
        HostAudioState::AwaitingRegions
    };
    HostAudioStatus {
        state,
        waiting_items: if state == HostAudioState::Ready {
            0
        } else {
            waiting_items
        },
    }
}

/// 图账本的一行读数（T8）。缺席时给一行可读的"没有接管账本"，而不是空串。
fn graph_summary_of(document: &super::document::DocumentSession) -> String {
    document
        .graph_ledger()
        .map_or_else(|| "graph=unbound".to_owned(), |ledger| ledger.summary())
}
/// 本实例的**运行形态**：宿主把音频交给我们的方式。
///
/// 【为什么需要一个显式形态，而不是"一堆布尔"】它要回答的问题在音频线程上被问
/// （`audio_process_inner` 每块一次），因此必须能**无锁**求值；而"绑定过没有"、
/// "绑定窗口关没关"这些事实各自都在别处、各自都可能取锁。把结论收敛成一个原子读数，
/// 音频线程只读它，判断点就不会散开。
///
/// 【为什么 `Undetermined` 是一等状态】ARA 绑定**允许**发生在 `setActive` 之前，
/// 而本插件在 `initialize` 时就已经存在。在窗口关闭之前断言"宿主不给我们 ARA"
/// 是**早的** —— 多数支持 ARA 的宿主正在这段时间里完成绑定。此前把"没绑"直接当成
/// "ARA-only"，正是 `bound_once` 那套措辞分支要解决的问题的同一个根源。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InstanceMode {
    /// 绑定窗口仍开着：宿主还可能把本实例绑到 ARA 文档。
    Undetermined,
    /// 宿主把本实例绑到了 ARA 文档：素材由 ARA 逐个 region 授权。
    Ara,
    /// 宿主没有绑定 ARA：素材只能从 VST3 输入总线捕获（Capture 通路）。
    Capture,
}

impl InstanceMode {
    const fn from_raw(raw: i32) -> Self {
        match raw {
            1 => InstanceMode::Ara,
            2 => InstanceMode::Capture,
            _ => InstanceMode::Undetermined,
        }
    }

    const fn raw(self) -> i32 {
        match self {
            InstanceMode::Undetermined => 0,
            InstanceMode::Ara => 1,
            InstanceMode::Capture => 2,
        }
    }
}

/// 本块在 Capture 通路上的输出结论（见 [`ExtensionOwner::capture_output`]）。
///
/// 【为什么把它做成一个值而不是几个布尔】"没素材时会听到什么"是这条通路唯一的对外
/// 行为，此前散在 `capture_stage` 的早退与 `vst3` 的后续分支里。收敛成一个值之后，
/// 每个分支都能被单独断言，而"静音"只剩"确实没有素材"一个含义。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CaptureOutput {
    /// 本阶段**不介入**：继续走既有快照通路。
    ///
    /// 两种场合落在这里：绑在 ARA 文档上的实例（素材由 ARA 授权），以及形态还没落定
    /// 且**没有**素材的实例。后者是刻意的：那时宿主可能正要绑定 ARA，而
    /// "未录制就不动输出"是既有契约（见 `an_unarmed_capture_path_leaves_the_snapshot_output_alone`）。
    Untouched,
    /// 正在录制：输出直通（监听）或静音。
    Recording,
    /// 有素材，且处理结果就是**当前** take：输出处理结果。
    Processed,
    // 【这一档被删掉了】它曾经表示"有已发布的处理结果，但那一份属于**上一个** take"。
    //
    // 删除的理由是它违反 I-2（一次播放只有一个内容源）：那一档的产物是
    // `SnapshotSource::Processed`，读的就是 `snapshots[index]` —— 也就是**上一段素材**。
    // 换 take 之后播它，等于用旧录音盖住新演奏，而用户会把它当成"编辑生效了"。
    //
    // 它当初存在的理由是"`promote()` 会 `bump_revision()`，于是换 take 的瞬间
    // `processed_ready()` 立刻变假，空档会掉到透传宿主原声"。而这条空档现在由
    // `Raw` 覆盖：`sync_raw_tape()` 在 promote 的同一条路径上**同步**发布当前 take 的
    // 原始捕获，于是那段空档播的是"这一份 take 的内容，只是还没处理"—— 同源、
    // 同 take、可解释。
    /// 有素材，但处理结果还不是当前 take：输出**原始磁带**（未处理的捕获）。
    Raw,
    /// 有素材，但**两份发布器都还没就绪**（刚录完、后台合成尚未完成）。
    ///
    /// 【为什么必须与 `PassThrough` 分开】两者的输出内容相同（都透传宿主原声），
    /// 但含义完全不同：`PassThrough` 是"还没有素材，插件保持透明"，本档是
    /// "素材已经在了，只是处理结果还没算出来"。用户听到的都是宿主原声，但前者是
    /// 正常状态，后者是**该有结果却没有**。此前两者共用一个名字，于是"合成从来没被
    /// 调度过"这件事在界面上完全看不出来（用户报障时只能看到"宿主原声"）。
    PendingSynthesis,
    /// 已确知本实例走 Capture、但还没有素材：透传宿主原声。
    PassThrough,
}

impl CaptureOutput {
    /// 语言无关的分类名；前端按此查 catalog。
    ///
    /// 【为什么要有它】"现在听到的是哪一版"是用户**无法自己判断**的一件事：处理结果、
    /// 原始录音与宿主原声听起来都可以"像那么回事"，而它们之间的差别正是"我的编辑生效了
    /// 没有"。把仲裁结论如实报给界面，用户就不必猜。
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            CaptureOutput::Untouched => "untouched",
            CaptureOutput::Recording => "recording",
            CaptureOutput::Processed => "processed",
            CaptureOutput::Raw => "raw",
            CaptureOutput::PendingSynthesis => "pending_synthesis",
            CaptureOutput::PassThrough => "passthrough",
        }
    }

    /// 编码值 → 档位。未知编码（含已删除的 `3`）返回 `None`。
    pub(crate) fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(CaptureOutput::Untouched),
            1 => Some(CaptureOutput::Recording),
            2 => Some(CaptureOutput::Processed),
            4 => Some(CaptureOutput::Raw),
            5 => Some(CaptureOutput::PendingSynthesis),
            6 => Some(CaptureOutput::PassThrough),
            _ => None,
        }
    }

    /// 稳定的编码值（只用于块级诊断）。**只增不改**。
    pub(crate) fn code(self) -> u8 {
        match self {
            CaptureOutput::Untouched => 0,
            CaptureOutput::Recording => 1,
            CaptureOutput::Processed => 2,
            // 3 是已删除的 `StaleProcessed`；编码值只增不改，故留空。
            CaptureOutput::Raw => 4,
            CaptureOutput::PendingSynthesis => 5,
            CaptureOutput::PassThrough => 6,
        }
    }
}

/// [`ExtensionOwner::capture_stage`] 给调用方的结论。
pub(crate) enum CaptureStage {
    /// 本阶段已经写完输出：调用方直接返回，不要再走快照通路。
    Handled,
    /// 继续走既有快照通路。
    Snapshot {
        /// 读哪个发布器。
        source: SnapshotSource,
        /// 快照覆盖范围**之外**的帧是否直通宿主原声。
        ///
        /// Capture 是（只替换录过的那一段）；ARA 不是（整条轨由 ARA 接管，区间外静音）。
        layered: bool,
    },
}

/// 快照通路该读哪个发布器。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SnapshotSource {
    /// 合成结果（`ExtensionOwner::snapshots`）。
    Processed,
    /// 活动 take 的未处理捕获（`ExtensionOwner::raw_tape`）。
    Raw,
}

/// 一次 `process` 回调里**本块**的完整证据。
///
/// 【为什么在同一个回调内传递】门控（停播清空）、输出仲裁（直通/处理结果）与捕获
/// （lookbehind）必须看到**同一块**。此前它们分别读实例级 `position_advancing` /
/// `host_playing`，而另一条进程流或另一个线程可以在两次读之间覆盖结论。这个值由
/// `capture_observe` 算出，原样交给 `capture_stage` 与记录器。
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockEvidence {
    pub stream: ProcessStream,
    /// 本块的有效工程起点（可能来自 continuous/music 回退）。
    ///
    /// 有切分时这是**第一段**的起点；逐段几何以 `split` 为准。
    pub position: i64,
    pub frames: usize,
    pub playing: bool,
    pub recording: bool,
    /// `kPlaying` / `kRecording` 的上升沿。
    pub playing_rising: bool,
    pub recording_rising: bool,
    /// 本块相对同流上一块的传输相位。
    pub phase: TransportPhase,
    /// 本流内是否见过上一块。
    pub had_previous: bool,
    /// 门控是否允许本块写出内容（仅实时流会因 `Resting` 为假）。
    pub allow_output: bool,
    /// 起播/录音上升沿打开的钉住窗口仍开着。
    ///
    /// 【为什么门控需要它】窗口内的前段可能是静音；按 `allow_output == false`
    /// 清掉它，内容出现时就是一次零→内容的阶跃（"咔"）。窗口本身就是"宿主刚
    /// 开始播放"的证据，窗口开着时输出应当连续。
    pub start_window: bool,
    /// 本块与上一块逐块完全相同且非零（FL 停播重复投递的形状）。
    ///
    /// 【为什么门控需要它】`start_window` 放行静音时不能把 FL 的停播重复内容
    /// 一起放行；这条是唯一能区分两者的证据。
    pub repeated_nonzero: bool,
    /// 循环边界切分（1 或 2 段）。
    pub split: BlockSplit,
}

/// 一条进程流自己的块几何（音频线程写，无锁）。
///
/// # 位置记账只有一只钟
///
/// 这里曾经有**两只**：宿主报的 `projectTimeSamples`，和插件按块长数出来的累积时钟
/// （`virtual_position` + `host_clock_offset` + `pinned_frames` + 4 个窗口原子）。两者
/// 之间的差用一个 1.5 s 的容差仲裁，容差之内一律相信累积时钟。
///
/// 那个设计的后果是：**用户回到起点再播（0.3–1.2 s 的循环）落在容差之内**，于是被
/// 判成"宿主读数滞后"，位置被强制成"上一遍末尾 + 块长"，整遍内容偏移恰好一个循环
/// 长度。这不是一个可以靠调容差修好的参数问题 —— 只要存在两只钟，就必须有一个仲裁，
/// 而任何仲裁都会在某一种真实操作上判错。
///
/// 现在只有一条规则（见 `note_stream_geometry` 的 R1–R4）：
/// **宿主每报一次位置就以它为准；两次报告之间按收到的帧数前进；跨 pass 一律归零。**
#[derive(Default)]
struct StreamGeometry {
    seen: AtomicBool,
    last_start: AtomicI64,
    last_frames: AtomicU32,
    last_end: AtomicI64,
    /// 同一个块首连续出现的次数。
    same_start: AtomicU32,
    /// 本块首上是否已经允许过一次“显著更大”的扩展块。
    extension_used: AtomicBool,

    // ── PassClock ──────────────────────────────────────────────────────
    /// 本 pass 的锚点：宿主**最后一次**报告的位置。
    ///
    /// 【为什么只需要一个 i64】`R1`（新 pass）与 `R3`（宿主报了新位置）都会把它覆盖成
    /// 宿主刚给的值；`R2`（宿主没动）只读不写。于是"锚点是否有效"由 `seen` 与
    /// `pass` 共同回答，不需要一个独立的 `anchor_valid`。
    anchor: AtomicI64,
    /// 自 `anchor` 以来**已经送来**的帧数（不含当前块）。
    ///
    /// 【为什么是"已送来的帧数"而不是"上一位置 + 本块长"】钉住的起点不一定落在第一个
    /// 块上；累加"已送来的帧数"让位置从一开始就对齐，而不必依赖 latch 在哪一块打开。
    frames_since_anchor: AtomicU64,
    /// 本 pass 里"**内容没变也可以按块长推进**"的剩余预算（帧）。`0` = 不推进。
    ///
    /// 【为什么必须有它】宿主停播时会用**同一个位置**反复调 `process`。那时
    /// `raw_start` 不动，但也没有任何证据说明时间在走 —— 按块长推进会让停播的光标
    /// 一直"前进"，捕获于是把同一小段内容反复写成新素材（FL Studio 的既有回归）。
    ///
    /// 唯一的例外是**起播窗口**：`kPlaying` 上升沿之后的前 1 秒里，前段往往是静音或
    /// 持续音，内容证据还没到，但那段时间确实是播放时间。所以预算只在上升沿打开，
    /// 并且只在**内容在变化**或**预算未耗尽**时才允许计数器前进。
    ///
    /// 【为什么它是帧数而不是布尔】"窗口有多宽"与"已经用掉多少"是同一件事的两面；
    /// 存一个剩余量让两者不可能互相漂移。
    advance_limit: AtomicU32,
    /// "内容在流动"这条证据还剩多少帧可信（只在 `playing == false` 时消耗）。
    ///
    /// 【为什么与 `advance_limit` 分开】两者管的是**不同的通路**：
    ///
    ///   * `advance_limit` 只在上升沿打开，管"宿主还没开始报位置"；
    ///   * `activity_budget` 在每次 `new_pass` 充满，管"宿主钉住位置但音频在流动"。
    ///
    /// 后者此前完全没有额度，于是宿主停播后仍在送预取音频时，计数器可以一路
    /// 前进 —— 实测 3.4 s。把它绑到 pass 上（一次播放只有一个开头），语义与它
    /// 所服务的起播窗口完全一致。
    activity_budget: AtomicU64,
    /// 本流的 pass 序号（R1 递增）。`0` = 还没有过任何一块。
    pass: AtomicU64,
    /// 锚点那一刻 `continuousTimeSamples` / `projectTimeMusic` 的读数。
    ///
    /// 【为什么存锚而不是累加增量】两者都是**绝对**读数（相对工程起点）。把"本块与
    /// 上一块的差"再叠到计数器上，会把同一段时间算两遍 —— 实测就是"位置钉住 +
    /// continuous 前进"时位置翻倍。存锚之后位置恒为
    /// `anchor + (continuous − continuous_anchor)`，与计数器互不叠加。
    continuous_anchor: AtomicI64,
    music_anchor: AtomicU64,

    last_raw_start: AtomicI64,
    last_continuous: AtomicI64,
    last_music_bits: AtomicU64,
    last_position: AtomicI64,
    last_source: AtomicU8,

    // ── 读数（I-3：位置是否逐块续接）─────────────────────────────────
    /// 位置**倒退**的块数。稳态下恒为 0。
    clock_rewinds: AtomicU64,
    /// 位置增量绝对值的最小/最大（帧）。稳态下两者都等于块长。
    clock_delta_min: AtomicU64,
    clock_delta_max: AtomicU64,
    /// 已经量过增量的块数（`delta_min` 的惰性初始化用）。
    clock_deltas: AtomicU64,
    /// `R3` 里"宿主报的位置与计数器不同"的次数。
    jumps: AtomicU64,
    /// 最近一次 jump 的带符号幅度（帧）。正 = 计数器落后于宿主。
    last_jump: AtomicI64,
    /// 位置由**计数器**推进的块数 / 帧数（R2 且没有更好的宿主时钟）。
    ///
    /// 【为什么它必须与 `position_source` 分开】"位置由我们数"与"位置来自哪条宿主
    /// 时钟"是两个正交的事实。混进一个枚举，就得到了 round 24 那个粘住整场会话的
    /// `Carried` 标签 —— 它让"宿主钉住过位置"这件事在读数上无法与"宿主一直在报位置"
    /// 区分。
    counted_blocks: AtomicU64,
    counted_frames: AtomicU64,

    blocks: AtomicU64,
    rolling_blocks: AtomicU64,
    resting_blocks: AtomicU64,
    rising_edges: AtomicU64,
    continuous_used: AtomicU64,
    music_used: AtomicU64,
    position_resyncs: AtomicU64,
    cycle_wraps: AtomicU64,
    cycle_wrap_unknown: AtomicU64,

    /// 最近 8 个块里"输入内容与上一块不同"的位图。
    ///
    /// 【为什么是"内容变化"而不是"有信号"】停播时宿主也可能持续送同一份非零内容
    /// （FL Studio 就是如此）。"有信号"会把停播判成播放；"内容在变"才是音频**正在
    /// 流动**的证据 —— 同一段内容反复出现说明时间没有前进。
    activity_changed: AtomicU32,
    /// 已经观测过的块数（饱和）。
    activity_seen: AtomicU32,
    /// 上一块的内容指纹。
    activity_hash: AtomicU64,
    activity_hash_valid: AtomicBool,
    /// 连续"内容没变"的块数（有变化就清零）。
    unchanged_run: AtomicU32,
    /// 本块处理**之前**是否处在"内容连续不变"的静止段。
    ///
    /// 【为什么要单独记】`unchanged_run` 在内容一变就清零，而相位判定发生在那之后；
    /// 想回答"这一块是不是刚从静止转入流动"就必须把结论留到那时。
    was_stationary: AtomicBool,
    /// 本流在位置钉住期间**紧接在本块之前**的连续静音帧数。
    ///
    /// 【为什么需要】宿主可能先送 0.4 s 静音再送内容（VEGAS 起播窗口）。这段静音
    /// 没有被写入捕获时，快照会从内容起点才开始覆盖，播放时就在 0.4 s 发生一次
    /// 块内源切换（"咔"）。捕获侧只需要帧数，不需要 PCM —— 它可以在第一段开头
    /// 合成等长静音，把段起点前移到播放起点。
    trailing_silent_frames: AtomicU64,
    /// 本块是否全零。
    ///
    /// 【为什么还需要它】它不参与位置推进，只作为诊断读数与"停播重复投递"的
    /// 判据：窗口内有多少块是静音，用来判断宿主起播窗口的形状。
    last_block_silent: AtomicBool,
}

/// "音频在流动"的观测窗口（块数）。
///
/// 【为什么是 8】窗口同时决定两件事：判据的灵敏度（至少 1 块有变化才成立）与
/// **停播后的尾巴**（最多 8 块 ≈ 170 ms @ 1024/48k）。再大就会让"停下之后"继续出声；
/// 再小则一次气口（静音块）就能把"在流动"判成"停了"。
const ACTIVITY_WINDOW_BLOCKS: u32 = 8;
/// 窗口内至少这么多块的内容有变化，才认为音频在流动。
///
/// 【为什么是 1】停播时宿主重复投递的是**同一份**内容（逐块相同），而真正在播放
/// 时相邻块几乎不可能完全相同。要求 2 块会让起播第一块被判成停播而丢失（那正是
/// 用户报的"开头一小段录不到"）。单块误判的代价只是多出一块（约 21 ms），
/// 与丢一块的代价同量级，而"宁可多录"是本轮的既定方向。
const ACTIVITY_MIN_CHANGES: u32 = 1;
/// 持久化状态从未尝试恢复。
pub(crate) const RESTORE_NEVER: u8 = 0;
/// 持久化状态已成功合并（`0` = 从未尝试）。
pub(crate) const RESTORE_APPLIED: u8 = 1;
/// 持久化状态已到达，但文档还没就绪 / 仍在重试（不是故障）。
pub(crate) const RESTORE_PENDING: u8 = 2;
/// 持久化状态合并失败，且没有可重试的路径（真故障）。
pub(crate) const RESTORE_FAILED: u8 = 3;
/// 宿主图**一次都没变**的时长上界：超过它就改报"没等到"并转入慢重试。
///
/// 【为什么必须有上界】"身份对不上 ⇒ 继续等"如果没有截止时刻，载荷永远挂着、界面
/// 永远说"正在等待"，而 `encode_archive` 在此期间会把这份**没合入**的状态当成权威
/// 写回磁盘 —— 用户看到的是"参数没了"，日志里一行错都没有。
///
/// 【为什么是 10 秒，且必须按"宿主没动"计】重开工程时宿主交付几何的典型跨度是
/// 几百毫秒到几秒。旧实现按**我们自己 poll 的次数**计时（`RESTORE_GIVE_UP_ATTEMPTS`），
/// 于是界面刷新得越快、预算烧得越快：10 秒的窗口能打印出 38.2 秒，那个数字既不是
/// "等了多久"也不是"宿主多久没动"。现在只有一个起点 —— 图签名最后一次变化的时刻。
pub(crate) const RESTORE_WAIT_WINDOW: std::time::Duration = std::time::Duration::from_secs(10);
/// 报过"没等到"之后的重试间隔。
///
/// 【为什么"放弃"还要重试】放弃的只是**界面读数**（从"等待"改成"没等到"），载荷本身
/// 必须留着：宿主随时可能补发几何，而那时合入一次就能把用户参数全部找回来。
pub(crate) const RESTORE_RETRY_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);

/// 一份恢复载荷的等待账：**只由宿主图的变化推进**。
///
/// 【为什么是一个结构而不是三个原子】"等了多久 / 变过没有 / 报过没有"必须一起读、
/// 一起写，否则又会出现上一版那种"三个计数器凑出一个数字"的状态。
#[derive(Default)]
struct RestoreProgress {
    /// 图签名最后一次变化的时刻（`None` = 还没有载荷）。
    since: Option<std::time::Instant>,
    /// 上一次看到的图签名。
    signature: u64,
    /// 这一次等待是否已经报过"没等到"（避免每轮重报）。
    reported: bool,
}
/// 一个 pass 里允许累积的**前导静音**上限（秒）。
///
/// 【它约束什么，不约束什么】只约束 `trailing_silent_frames` 的累积与 `start_window`
/// 的宽度（两者都是"这次播放刚开始"的证据）；**不约束位置推进**。VEGAS 的起播窗口
/// 约 0.66 s，1 秒足够覆盖；超过 1 秒的静音由捕获侧的 resync 零桥接补齐。
const PRE_ROLL_CAP_SEC: f64 = 1.0;
impl StreamGeometry {
    /// 记录本块的内容指纹，回答"这一块与上一块是不是同一份内容"。
    fn note_activity(&self, hash: u64) -> bool {
        let previous_valid = self.activity_hash_valid.swap(true, Ordering::AcqRel);
        let previous = self.activity_hash.swap(hash, Ordering::AcqRel);
        let changed = previous_valid && previous != hash;
        self.was_stationary.store(
            self.unchanged_run.load(Ordering::Acquire) > 0,
            Ordering::Release,
        );
        if changed {
            self.unchanged_run.store(0, Ordering::Release);
        } else if previous_valid {
            self.unchanged_run.fetch_add(1, Ordering::Relaxed);
        }
        // 只保留最近 8 位；`changed` 占最低位。
        let window = self.activity_changed.load(Ordering::Acquire);
        let window = (window << 1) | u32::from(changed);
        self.activity_changed.store(
            window & ((1 << ACTIVITY_WINDOW_BLOCKS) - 1),
            Ordering::Release,
        );
        self.activity_seen
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |seen| {
                Some(seen.saturating_add(1))
            })
            .ok();
        changed
    }

    /// 记录本块是否全零（由 `capture_observe` 在几何判定之前调用）。
    fn note_block_silence(&self, silent: bool) {
        self.last_block_silent.store(silent, Ordering::Release);
    }

    /// 最近窗口里是否有足够多的"内容变化"块。
    fn activity_moving(&self) -> bool {
        self.activity_changed.load(Ordering::Acquire).count_ones() >= ACTIVITY_MIN_CHANGES
    }
}

/// 输入块的内容指纹（只用来回答"是不是同一份内容"）。
///
/// 【为什么可以抽样】它不参与任何音频计算，只区分"这一块与上一块不同"。步长 8 让
/// 成本降到每声道 `frames/8` 次读；而任何真正在流动的音频在 20 ms 内都不可能只在
/// 未抽样的位置上变化。
///
/// 【为什么把块长也拌进去】宿主在停播时可能把块长缩小/放大而内容不变 —— 那不是
/// 时间在前进。
/// 本块是否全零（锁存推进只对静音块生效）。
fn block_is_silent(block: &crate::capture::recorder::InputBlock<'_>) -> bool {
    block
        .planes
        .iter()
        .flatten()
        .all(|plane| plane.iter().all(|sample| *sample == 0.0))
}

fn block_activity_hash(block: &crate::capture::recorder::InputBlock<'_>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for plane in block.planes.iter().flatten() {
        let mut index = 0;
        while index < plane.len() {
            hash ^= u64::from(plane[index].to_bits());
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            index += 8;
        }
    }
    hash ^ (block.frames as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// 一条流在本块上的位置/相位结论。
#[derive(Clone, Copy, Debug)]
struct StreamBlock {
    /// 本块的有效工程起点（可能来自 continuous/music 回退）。
    position: i64,
    /// 宿主原始 `projectTimeSamples`。
    raw_start: i64,
    phase: TransportPhase,
    had_previous: bool,
    cycle_wrap: bool,
    /// 本块开始了一个新的 pass（循环回卷或位置后退）。
    new_pass: bool,
    /// 本流最近是否观测到"输入内容在变化"（音频在流动）。
    activity_moving: bool,
    /// 本块是宿主**纠正了我们数出来的位置**（见 `note_stream_geometry` 的 R3）。
    resync: bool,
    /// 允许当作"记账纠正"的最大帧差（钉住期间送来的帧数 + 本块）。
    resync_limit_frames: u64,
    /// 本块之前连续静音的帧数（见 `StreamGeometry::trailing_silent_frames`）。
    preceding_silent_frames: u64,
    /// 本块位置由**计数器**推进（宿主没有报新位置）。
    ///
    /// 【为什么与 `position_source` 分开】"位置由我们数"与"位置来自哪条宿主时钟"
    /// 是两个正交的事实：计数器推进时标签仍是 `Project`（宿主给的那个起点确实是
    /// `Project`），只是这一块没有新的宿主报告。混进一个枚举就得到了 round 24 那个
    /// 粘住整场会话的 `Carried`。
    counted: bool,
    /// 本块是停播很久之后的上升沿：此前累积的静音必须从 take 前导里扣掉。
    discard_pre_roll: bool,
    /// 起播/录音上升沿打开的窗口仍开着（门控据此放行窗口内静音）。
    start_window: bool,
    /// 本块与上一块逐块完全相同且非零（FL 停播重复投递的形状）。
    repeated_nonzero: bool,
    /// 本块有效位置来自哪条**宿主时钟**（见 `PositionSource`）。
    ///
    /// 【它不再表示"位置可不可信"】位置只有一只钟（见 `note_stream_geometry` 的
    /// R1–R4），所以这个字段只回答"宿主用哪个字段给的"。它**不**影响覆盖 ——
    /// 覆盖只回答"这一段样本我们拷过吗"。
    position_source: PositionSource,
}

/// 本块可用的宿主时间字段（全部可选；缺失时退回工程位置）。
///
/// 【为什么是 `pub(crate)`】它出现在 `ExtensionOwner::note_block_trace` 的参数里，
/// 而那个方法是 `pub(crate)`（`vst3` 的音频回调要调它）。私有类型泄进更宽可见度的
/// 签名是 `private_interfaces` 告警的来源；把类型提到同一档即可，不需要放宽方法。
#[derive(Clone, Copy, Default)]
pub(crate) struct StreamTime {
    continuous: Option<i64>,
    music: Option<f64>,
    tempo: Option<f64>,
    cycle_active: bool,
    cycle_valid: bool,
    cycle_start_music: f64,
    cycle_end_music: f64,
}

impl StreamTime {
    fn from_context(context: &crate::audio_abi::ProcessContext) -> Self {
        let state = context.state;
        let continuous_ok = state & crate::audio_abi::PROCESS_CONTEXT_CONTINUOUS_VALID != 0;
        let music_ok = state & crate::audio_abi::PROCESS_CONTEXT_PROJECT_TIME_MUSIC_VALID != 0;
        let tempo_ok = state & crate::audio_abi::PROCESS_CONTEXT_TEMPO_VALID != 0;
        let cycle_ok = state & crate::audio_abi::PROCESS_CONTEXT_CYCLE_VALID != 0;
        Self {
            continuous: continuous_ok.then_some(context.continuous_time_samples),
            music: music_ok.then_some(context.project_time_music),
            tempo: (tempo_ok && context.tempo.is_finite() && context.tempo > 0.0)
                .then_some(context.tempo),
            cycle_active: state & crate::audio_abi::PROCESS_CONTEXT_CYCLE_ACTIVE != 0,
            cycle_valid: cycle_ok,
            cycle_start_music: context.cycle_start_music,
            cycle_end_music: context.cycle_end_music,
        }
    }
}

/// 输出源切换的去咔长度（帧）。48 kHz 下约 1.3 ms，足以消除阶跃。
///
/// 【它现在是**默认**宽度，不再是唯一宽度】异源切换（透传 ↔ 处理结果）只能淡入，
/// 1.3 ms 足够躲开阶跃；而**同一段素材的两种处理**（原始磁带 ↔ 处理结果）内容高度
/// 相关，正确处置是 20 ms 等功率交叉 —— 见 [`OUTPUT_WIDE_CROSSFADE_FRAMES`]。
const OUTPUT_DECLICK_FRAMES: usize = 64;

/// 接缝状态保留的历史长度（帧）。48 kHz 下 20 ms。
///
/// 【为什么缓冲区要比默认淡化宽】要交叉淡化 20 ms，就必须手里有上一块输出的
/// 20 ms 尾巴。缓冲区按最大宽度开，具体一次淡化用多宽由 `fade_len` 决定 ——
/// 于是"64 帧淡入"与"960 帧交叉"共用同一份状态，音频线程仍然只写原子、不分配。
const DECLICK_HISTORY_FRAMES: usize = 960;

/// `Raw ↔ Processed` 的交叉淡化宽度（帧）。48 kHz 下 20 ms。
///
/// 【为什么这一对要宽得多】两者是**同一段素材的两种处理**：原始磁带是捕获原样，
/// 处理结果是声码器输出。它们高度相关，所以等功率交叉**不会**叠出双声；而 1.3 ms
/// 只够躲开阶跃，躲不开"音色突变"的听感 —— 用户描述的"播放中途随机咔"里有一份
/// 正是渲染恰好在起播后 0.4~0.7 s 完成、把 Raw 换成 Processed 的那一下。
const OUTPUT_WIDE_CROSSFADE_FRAMES: usize = 960;
/// 本流**第一块**的淡入宽度（帧）。48 kHz 下 5 ms。
///
/// 【为什么比 `OUTPUT_DECLICK_FRAMES` 宽】接缝淡化有"上一块输出的尾巴"作为旧侧，
/// 1.3 ms 就够把两条连续信号接上；首块**没有任何旧侧**，1.3 ms 在听感上仍是一次
/// 突变。5 ms 是"听不出渐强、但足以消掉阶跃"的那一档。
///
/// 【为什么不是 20 ms】首块之后马上就是真实内容；20 ms 的起播淡入会被听成
/// "开头有点闷"。
const OUTPUT_FIRST_BLOCK_FADE_FRAMES: usize = 240;

/// 输出源的**内容家族**（接缝判据的输入，见 `ExtensionOwner::family`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContentFamily {
    /// 宿主原声（透传 / 待合成 / ARA 未接管）。
    Passthrough,
    /// 原始捕获磁带（未处理）。
    Raw,
    /// 算出来的处理结果。
    Processed,
}

/// 一次接缝该怎么处理。
///
/// 【为什么是一个枚举而不是布尔】此前只有"淡入 / 交叉淡化"两档，于是"同一段素材的
/// 两种处理"被迫归到淡入 —— 那会在 1.3 ms 里把处理结果从零拉满，听感仍是一次突变。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Seam {
    /// 异源（任一侧是透传）：对当前结果线性淡入，宽度 `OUTPUT_DECLICK_FRAMES`。
    FadeIn,
    /// 同一族内的版本切换：交叉淡化，宽度 `OUTPUT_DECLICK_FRAMES`。
    Crossfade,
    /// 同一段素材的两种处理（Raw ↔ Processed）：等功率交叉，宽度 20 ms。
    WideCrossfade,
    /// 本流的**第一块**：没有"上一块输出的尾巴"，按零起点淡入。
    ///
    /// 【为什么需要它】"真首块原样输出"这条旧契约的理由是"从零淡入会改变宿主直通
    /// 的样本"。那只对**透传**成立（透传的样本就是宿主正在播的东西，它自己已经
    /// 连续）。当第一块是**我们算出来的内容**（Raw / Processed）时，它从哪一相位
    /// 开始是任意的 —— 从零直接跳到块内波形就是一次标准的起播咔。
    FirstBlock,
}

impl Seam {
    /// 诊断编码（进块账本）。**只增不改**：日志里已经出现过的数字必须保持含义。
    fn code(self) -> crate::render::click::SeamCode {
        match self {
            Self::FadeIn => crate::render::click::SeamCode::FadeIn,
            Self::Crossfade => crate::render::click::SeamCode::Crossfade,
            Self::WideCrossfade => crate::render::click::SeamCode::WideCrossfade,
            Self::FirstBlock => crate::render::click::SeamCode::FadeIn,
        }
    }

    /// 这一次淡化是否用等功率权重（见 `arm_seam` 的说明）。
    fn uses_equal_power(self) -> bool {
        matches!(self, Self::FadeIn | Self::WideCrossfade | Self::FirstBlock)
    }
}

/// 一条进程流的输出接缝状态（音频线程写，无锁）。
///
/// 【为什么按流分开】prefetch 与 realtime 的块会交替到达；共用一份 tail 会让
/// 一条流把另一条流的"上一块"当成自己的接缝。
struct OutputDeclickState {
    /// 上一块输出的最后 `DECLICK_HISTORY_FRAMES` 帧（f32 位模式）。
    tail: [Vec<AtomicU32>; 2],
    /// 源切换那一刻抓下来的旧信号，供跨块淡入淡出（前 `fade_len` 项有效）。
    fade_from: [Vec<AtomicU32>; 2],
    fade_pos: AtomicU32,
    fade_active: AtomicBool,
    /// 本次淡化用多宽（帧）。0 = 还没设过（按 `OUTPUT_DECLICK_FRAMES` 处理）。
    fade_len: AtomicU32,
    /// 本次淡化用哪条权重（`1` = 等功率，`0` = 线性）。见 `Seam::uses_equal_power`。
    ///
    /// 【为什么权重也要进状态】淡化可以跨块继续，而"两条不同源"与"同一份内容的
    /// 两次渲染"需要**不同**的数学。把结论存下来，跨块的每一帧才用同一条曲线。
    fade_curve: AtomicU32,
    last_token: AtomicU64,
    /// 本流是否已经输出过**至少一块**。
    ///
    /// 【为什么不能用 `primed_kind == 0` 表达"没见过"】`CaptureOutput::Untouched` 的
    /// 编码值**就是 0**，而 ARA 实例的输出仲裁恒为它。于是"从没见过"与"一直见到 0"
    /// 在旧判据下无法区分：ARA 的每一块都会被当成"真首块"。
    primed: AtomicBool,
    /// 本流上一次输出源的**种类**（`primed` 为假时无意义）。
    ///
    /// 【为什么记种类而不是一个 bool】真首块没有"上一块输出的尾巴"，从零淡入会改变
    /// 宿主直通/录制监听的样本（测试与真实行为都要求第一块原样输出）。但"上一轮播放
    /// 收在 `Raw`、这一轮首块是 `Processed`"仍然是一次源切换 —— 那种场合必须去咔，
    /// 否则用户听到的正是"咔声位置不固定"。种类相同才允许首块原样通过。
    primed_kind: AtomicU32,
}

impl Default for OutputDeclickState {
    fn default() -> Self {
        let zeros = || {
            (0..DECLICK_HISTORY_FRAMES)
                .map(|_| AtomicU32::new(0))
                .collect::<Vec<_>>()
        };
        Self {
            tail: [zeros(), zeros()],
            fade_from: [zeros(), zeros()],
            fade_pos: AtomicU32::new(0),
            fade_active: AtomicBool::new(false),
            fade_len: AtomicU32::new(0),
            fade_curve: AtomicU32::new(0),
            last_token: AtomicU64::new(0),
            primed: AtomicBool::new(false),
            primed_kind: AtomicU32::new(0),
        }
    }
}

/// 一块分层拷贝的**覆盖读数**（音频线程写、同块读走）。
///
/// 【为什么用原子而不是局部变量】写它的地方在 `snapshot.rs`（只知道"哪一帧属于
/// 快照"），读它的地方在 `apply_output_declick`（只知道"这一块输出了什么"）。
/// 两者之间隔着一次跨模块调用，而那块缓冲的生命周期不允许我们把闭包传出去 ——
/// 一个每流一格、每块重置的原子单元是最小耦合。
#[derive(Default)]
struct LayeredProbeCell {
    /// 本块里**不属于我们**的帧数。
    uncovered: AtomicU32,
    /// 那些帧上宿主输入的峰值（`NaN` 位模式 = 本块没有未覆盖帧）。
    input_peak: AtomicU32,
    /// 覆盖边（覆盖 ↔ 未覆盖的翻转点）是否落在本块内。
    edge: AtomicBool,
    /// 本块写过这份读数（否则读走的是上一块的值）。
    valid: AtomicBool,
}

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

impl ExtensionOwner {
    /// 组件终止独立于Arc尚存与否，旧native entry不能维持可编辑授权。
    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
    /// 已绑定且实例仍存活时的扩展实例指针。
    ///
    /// 【用途】宿主重复 `bindDocumentToExtension` 时，companion 层会先拒绝第二次绑定
    /// （`companion processor is already bound`），`ara_entry::bind` 据此**幂等复用**
    /// 既有实例而不是返回 null —— 否则宿主认为插件未绑定、插件完全无法加载。这里只
    /// 暴露"复用"所需的只读指针，**不改绑定语义**。
    ///
    /// 【为什么还需要 `_for(document)`】只检查 `is_closed()` 会漏掉更常见的一种
    /// 顺序：宿主先绑新文档、旧文档稍后才关闭。此时实例仍绑在旧文档上，
    /// `existing_binding_raw()` 会把**旧文档的扩展实例指针**交回给新文档 ——
    /// 宿主随后对新文档调用 `addPlaybackRegion`，却落在旧文档的 observer 上。
    /// 调用方必须比对文档身份；`existing_binding_raw()` 只保留给"同文档重复绑定"
    /// 的诊断/测试路径。
    pub(crate) fn existing_binding_raw(
        &self,
    ) -> Option<*const ara2_bridge::sys::ARAPlugInExtensionInstance> {
        if self.is_closed() {
            return None;
        }
        self.binding
            .lock_or_recover()
            .as_ref()
            .map(ara2_bridge::plugin::ExtensionBinding::as_raw)
    }
    /// 仅当"未关闭、且确实绑定在 `document` 上"时复用既有扩展实例。
    pub(crate) fn existing_binding_raw_for(
        &self,
        document: super::ownership::DocumentId,
    ) -> Option<*const ara2_bridge::sys::ARAPlugInExtensionInstance> {
        if self.is_closed()
            || self.bound_document_id.load(Ordering::Acquire) != document
            || self.bound_document_id.load(Ordering::Acquire) == 0
        {
            return None;
        }
        self.existing_binding_raw()
    }
    /// 当前绑定的**活**文档 id；已关闭/从未绑定为 `None`。
    pub(crate) fn bound_document_id(&self) -> Option<super::ownership::DocumentId> {
        let id = self.bound_document_id.load(Ordering::Acquire);
        (id != 0).then_some(id)
    }
    /// 是否还持有一份扩展绑定（不区分它是哪份文档）。
    pub(crate) fn has_binding(&self) -> bool {
        self.binding.lock_or_recover().is_some()
    }
    /// 本实例**是否绑定过** ARA 文档（历史事实，不随文档关闭回落）。
    ///
    /// 【为什么 ara_entry 需要它】companion 层的"already bound"既可能是"宿主重复绑定
    /// 同一文档"（幂等复用即可），也可能是"宿主重开工程、同一实例要求绑新文档"
    /// （必须复位 companion 再绑）。前者复用，后者复位 —— 两种处理不能互换：
    /// 复用会把已销毁文档的绑定交给宿主，复位则会把还在用的绑定拆掉。
    pub(crate) fn was_bound_before(&self) -> bool {
        self.bound_once.load(Ordering::Acquire)
    }
    /// 实时流此刻是否在推进（供"内容更新要不要撤渲染许可"判断）。
    pub(crate) fn realtime_flow_active(&self) -> bool {
        self.realtime_flow.load(Ordering::Acquire)
    }
    /// 读游标接缝读数（诊断线程读；音频线程只写原子）。
    ///
    /// 【为什么要有它】"起播 0.4~0.7 s 一次咔"最直接的量是"读位置跳了多少帧"。
    /// 没有这一行时，判断只能靠听感与代码推理；有了它，一次复现就能回答
    /// "跳了几次、最大多少、宿主纠正过几次"。
    pub(crate) fn seam_summary(&self) -> String {
        let geometry = &self.streams[crate::capture::recorder::ProcessStream::Realtime.index()];
        format!(
            "seam(pos={} corr={} max={} jumps={} last_jump={} counted={} cov(holes={} frames={}))",
            self.rt_last_read_position.load(Ordering::Acquire),
            self.position_corrections.load(Ordering::Relaxed),
            self.max_correction_frames.load(Ordering::Relaxed),
            geometry.jumps.load(Ordering::Acquire),
            geometry.last_jump.load(Ordering::Acquire),
            geometry.counted_blocks.load(Ordering::Acquire),
            self.tape_holes.load(Ordering::Relaxed),
            self.tape_hole_frames.load(Ordering::Relaxed),
        )
    }
    /// 记录一次 companion 绑定复位（见 [`Self::companion_reset_count`]）。
    pub(crate) fn note_companion_reset(&self) {
        self.companion_resets.fetch_add(1, Ordering::Relaxed);
    }
    /// companion 绑定复位的累计次数（诊断）。
    pub(crate) fn companion_reset_count(&self) -> u64 {
        self.companion_resets.load(Ordering::Relaxed)
    }
    /// 记录"有一份持久化状态待恢复"。
    pub(crate) fn note_restore_pending(&self) {
        // 已经从"失败"进入"重试中"时保留错误原文，直到真正成功才清 ——
        // 用户看到的应当是"仍在尝试（上次失败原因：…）"。
        self.restore_status
            .store(RESTORE_PENDING, Ordering::Release);
    }
    /// 记录"仍在等一个明确的时机，不是失败"。
    ///
    /// 【为什么与 `note_restore_pending` 分开】pending 只说明"有载荷"；这一条
    /// 回答"在等什么"（时间线未发布 / 轨道为空 / fade 元数据未就绪）。
    pub(crate) fn note_restore_waiting(&self, reason: &str) {
        self.restore_status
            .store(RESTORE_PENDING, Ordering::Release);
        *self.restore_pending_reason.lock_or_recover() = Some(reason.to_owned());
    }
    /// 记录"持久化状态已成功合并"。
    fn note_restore_applied(&self) {
        self.restore_status
            .store(RESTORE_APPLIED, Ordering::Release);
        *self.restore_error.lock_or_recover() = None;
        *self.restore_pending_reason.lock_or_recover() = None;
        self.restore_applied_generation
            .fetch_add(1, Ordering::Relaxed);
    }
    /// 记录"持久化状态合并失败"。
    pub(crate) fn note_restore_failed(&self, error: &str) {
        self.restore_status.store(RESTORE_FAILED, Ordering::Release);
        *self.restore_error.lock_or_recover() = Some(error.to_owned());
        *self.restore_pending_reason.lock_or_recover() = None;
    }
    /// 持久化状态的恢复读数：`(status, error, applied_generation)`。
    pub(crate) fn restore_report(&self) -> (u8, Option<String>, u64) {
        (
            self.restore_status.load(Ordering::Acquire),
            self.restore_error.lock_or_recover().clone(),
            self.restore_applied_generation.load(Ordering::Relaxed),
        )
    }
    /// 恢复状态的语言无关名（前端按它查 catalog）。
    pub(crate) fn restore_status_name(&self) -> &'static str {
        match self.restore_status.load(Ordering::Acquire) {
            RESTORE_APPLIED => "applied",
            RESTORE_PENDING => "pending",
            RESTORE_FAILED => "failed",
            _ => "never",
        }
    }
    /// 恢复仍挂着时的可读原因（成功/失败/无载荷时为 `None`）。
    pub(crate) fn restore_pending_reason(&self) -> Option<String> {
        self.restore_pending_reason.lock_or_recover().clone()
    }
    /// 清空本实例的恢复读数（重绑到新文档时用）。
    fn reset_restore_report(&self) {
        self.restore_status.store(RESTORE_NEVER, Ordering::Release);
        *self.restore_error.lock_or_recover() = None;
        *self.restore_pending_reason.lock_or_recover() = None;
        self.restore_applied_generation.store(0, Ordering::Relaxed);
    }
    /// 状态是否落在"失败"档（供 `HostAudioState` 分类）。
    pub(crate) fn restore_failed(&self) -> bool {
        self.restore_status.load(Ordering::Acquire) == RESTORE_FAILED
    }
    pub(crate) fn editor_document(&self) -> Result<Arc<super::document::DocumentSession>, String> {
        if self.is_closed() {
            return Err(crate::editor::errors::FX_PROCESSOR_CLOSED.error());
        }
        self.document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .filter(|document| document.is_alive())
            .ok_or_else(|| {
                // "从未绑定"与"绑定过又关闭"是两件不同的事，措辞必须分开：
                // 前者在 Cubase 上重开一万次窗口也一样（宿主根本不走 ARA 路径），
                // 后者才是"会话已结束"。共用一句会把用户引向必然失败的补救动作。
                if self.bound_once.load(Ordering::Acquire) {
                    crate::editor::errors::DOCUMENT_CLOSED.error()
                } else {
                    crate::editor::errors::ARA_NOT_BOUND.error()
                }
            })
    }
    /// 宿主最近一次编辑器视图选择（`notifySelection`），指针键的副本。
    ///
    /// 【为什么可以从任意线程读】`ExtensionState` 只在写入时持锁，这里读的是一份
    /// 已经拷好的值的克隆 —— 与 `assigned_regions()` 同一个口径。宿主的选择通知本身
    /// 不触发 `end_editing`，因此**不能**只在模型线程上采：那样它要等到下一次模型变化
    /// 才可见。编辑器 actor 每次 `ensure_loaded` 采一次即可。
    pub(crate) fn host_view_selection(
        &self,
    ) -> Option<ara2_bridge::plugin::ExtensionViewSelection> {
        if self.is_closed() {
            return None;
        }
        self.binding
            .lock_or_recover()
            .as_ref()
            .and_then(ara2_bridge::plugin::ExtensionBinding::view_selection)
    }
    /// 健康探活快照：**只读原子/弱引用，不创建会话、不取文档事务、不排队**。
    ///
    /// 【为什么需要它】启动探活此前用 `get_runtime_info`，而那条命令会走
    /// `editor_session()` + actor 队列。重开工程时 `EditorSession::new` 的初始化
    /// （以及曾经的同步临时目录清扫）可能远超 30 s IPC 超时，于是用户看到
    /// `Backend invoke failed: plugin:get_runtime_info`。探活只需要回答"会话还在不在、
    /// 形态是什么、有没有片段"，这些都能从原子读数得到。
    pub(crate) fn health_snapshot(&self) -> serde_json::Value {
        let capture = self.capture_session_if_initialized();
        let document = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade);
        let document_alive = document
            .as_ref()
            .is_some_and(|document| document.is_alive());
        let clip_count = capture
            .as_ref()
            .map_or(0, |session| session.clip_count())
            .max(
                document
                    .as_ref()
                    .map_or(0, |document| document.editor_clip_count()),
            );
        let mode = match self.mode() {
            InstanceMode::Ara => "ara",
            InstanceMode::Capture => "capture",
            InstanceMode::Undetermined => "undetermined",
        };
        serde_json::json!({
            "ok": true,
            "closed": self.is_closed(),
            "bound": self.is_bound(),
            "bound_once": self.bound_once.load(Ordering::Acquire),
            "mode": mode,
            "document_alive": document_alive,
            "session_initialized": capture.is_some()
                || document.as_ref().is_some_and(|document| document.editor_initialized()),
            "host_kind": self.host_backend().kind().as_str(),
            "model_available": hifishifter_kernel::world_vocoder::is_available(),
            "clip_count": clip_count,
        })
    }

    /// 本实例当前是否**真的**挂在一份活的 ARA 文档上。
    ///
    /// 与 `bound_once` 的区别：那个是"曾经绑定过"的单调历史，这个是"此刻有效"。
    /// 诊断（"宿主是否走了 ARA 路径"）要的是后者。
    pub(crate) fn is_ara_bound(&self) -> bool {
        self.document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .is_some_and(|document| document.is_alive())
    }

    /// 本实例的运行形态。**无锁**，可从音频线程调用。
    ///
    /// 【为什么先看 `bound_once` 再看原子】两者本该一致，但 `bound_once` 是既有的、
    /// 已被测试覆盖的读数，而 `binding_mode` 是本轮新增的。取"任一说 ARA 就是 ARA"
    /// 的交集语义，让新字段在任何写入点漏掉时**退化成旧行为**（AraOnly），
    /// 而不是退化成 Capture —— 后者会让一个正常绑定的实例去录制自己的输出。
    pub(crate) fn mode(&self) -> InstanceMode {
        if self.bound_once.load(Ordering::Acquire) {
            return InstanceMode::Ara;
        }
        InstanceMode::from_raw(self.binding_mode.load(Ordering::Acquire))
    }

    /// 记录"宿主把本实例绑到了 ARA 文档"。
    pub(crate) fn note_ara_bound(&self) {
        self.binding_mode
            .store(InstanceMode::Ara.raw(), Ordering::Release);
        // 已打开的视图可能是在绑定之前创建的：让视图定时器补发一次 bootstrap。
        self.bootstrap_changed.store(true, Ordering::Release);
    }

    /// 取走"bootstrap 需要补发"的标记（视图定时器调用）。
    pub(crate) fn take_bootstrap_changed(&self) -> bool {
        self.bootstrap_changed.swap(false, Ordering::AcqRel)
    }

    /// 记录"绑定窗口已关闭且没有绑定" —— 本实例是 Capture 形态。
    ///
    /// 【为什么由调用方判定"窗口已关闭"而不是这里自己看时钟】判据是
    /// `setActive` 之后过了宽限期（见 [`Self::note_editor_tick`]），那需要读
    /// `set_active_at` 这个锁下的值。这里只负责**写入结论**，判定留在唯一的调用点，
    /// 避免同一判据出现两份。
    pub(crate) fn note_binding_window_closed_without_ara(&self) {
        // 已经绑过就不回落：那会把一个正常工作的 ARA 实例降级成 Capture。
        if self.bound_once.load(Ordering::Acquire) {
            return;
        }
        let _ = self.binding_mode.compare_exchange(
            InstanceMode::Undetermined.raw(),
            InstanceMode::Capture.raw(),
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    /// 记录宿主调用 `setActive(true)` 的时刻。
    ///
    /// 【为什么在这里记而不是轮询宿主】`setActive` 是 ARA 绑定窗口的**关闭点**：
    /// 契约规定绑定只能发生在它之前。把时刻记下来，UI 定时器就能在宽限期后给出
    /// 一条确定性的诊断，而不是让读者去对比两个会话的日志。
    pub(crate) fn note_component_activated(&self) {
        let mut at = self.set_active_at.lock_or_recover();
        if at.is_none() {
            *at = Some(std::time::Instant::now());
        }
    }

    /// UI 定时器每 tick 调用：宿主激活后仍无 ARA 绑定时，**一次性**给出诊断。
    ///
    /// 【为什么必须有这一条】用户报障：Cubase 里窗口能开、每个命令都失败，但日志里
    /// 没有任何一行说明"宿主根本没走 ARA"。要得出这个结论，得把 Cubase 会话与 REAPER
    /// 会话逐行对比（前者零次 `queryInterface ARA hit`、零次 `document controller
    /// created`）。一条确定性的警告把这件事变成一眼可见。
    ///
    /// 【为什么在 UI 定时器上做而不是开线程】插件 DLL 可能被宿主卸载；一个 sleep 5 秒
    /// 的分离线程会在模块卸载后回到已消失的代码。UI 定时器天然与视图同寿，且已经在
    /// 每 20ms 跑一次。
    /// 绑定窗口是否已经关闭：宿主激活过，且过了宽限期仍没有 ARA 绑定。
    ///
    /// 【为什么不能只看 UI 定时器】UI 定时器要等 WebView 起来才跑，而 bootstrap
    /// （`hostKind` / `captureCaps`）与编辑器会话都在**那之前**就要读形态 ——
    /// 于是"窗口打开的那一刻"读到的永远是 `Undetermined`，`host_backend()` 退成
    /// ARA-only，前端所有按形态分流的地方（面板、探活）在前 5 秒里全是错的。
    /// 判定因此必须能按需求值，而不是只在一个定时器里发生。
    fn binding_window_closed(&self) -> bool {
        self.set_active_at
            .lock_or_recover()
            .is_some_and(|at| at.elapsed() >= ARA_BINDING_GRACE)
    }

    /// 在**非音频线程**上把形态判定落定（幂等，可在任何读形态之前调用）。
    ///
    /// 读 `set_active_at` 要取锁，因此绝不能在音频线程调用 —— 音频线程只读
    /// [`Self::mode`] 那个原子结论。
    pub(crate) fn resolve_binding_mode(&self) {
        if self.is_ara_bound() {
            return;
        }
        if self.binding_window_closed() {
            self.note_binding_window_closed_without_ara();
        }
    }

    pub(crate) fn note_editor_tick(self: &Arc<Self>) {
        if self.is_ara_bound() {
            return;
        }
        // 【为什么形态判定在这里落地】这里正是"过了宽限期仍未绑定"这个结论的判定点：
        // `set_active_at` 只有这里与 `binding_window_closed` 读，宽限常数只有这两处用。
        // 形态一旦确定，后端选择、能力位与前端面板都会跟着翻到 Capture（见
        // `host_backend`）。判定写在"只报一次"的早退**之前** —— 形态是事实，与是否要
        // 打日志无关。
        if !self.binding_window_closed() {
            // 宿主还没激活，或还在宽限期内：绑定窗口仍然开着，任何结论都是早的。
            return;
        }
        self.note_binding_window_closed_without_ara();
        // 【为什么这里不再做任何维护】备用缓冲与宿主时钟此前挂在 UI 定时器上，于是
        // "关掉窗口就断录"（备用缓冲 2 粒 × 8 秒 = 24 秒后 `NoSpare`）而且没开窗口的
        // 会话里宿主位置根本没有地方落。现在两者都有与窗口无关的归属：
        //   * 时钟/合成队列 → `setupProcessing`（见 `arm_capture_rendering` 的调用点）；
        //   * 备用缓冲       → 会话 actor（见 `EditorSession::run` 的 `top_up` 调用）。
        // 这里只剩"形态落定 + 一次性说明"，形态本身仍然是这个 tick 的职责 ——
        // 它是唯一同时握有 `setActive` 时刻与宽限期的地方。
        if self
            .ara_binding_absent_reported
            .swap(true, Ordering::AcqRel)
        {
            return;
        }
        let backend = self.host_backend();
        let host = backend.app_name().unwrap_or("this host");
        // 诊断里要报"激活之后过了多久"，判据本身只问"窗口关没关"（见 `binding_window_closed`）。
        let elapsed = (*self.set_active_at.lock_or_recover())
            .map(|at| at.elapsed())
            .unwrap_or_default();
        // 【为什么措辞变了、级别也变了】此前这条是 `warn` 且说"宿主把本插件当普通插入
        // 效果器运行"，并把用户引向"请从 ARA 入口插入"。现在普通插入是**受支持**的
        // 形态（Capture 通路，见 `docs/plans/2026-10-15-vst-capture-mode.md`）：
        // 音频从输入总线捕获、按工程时间对到插件时间轴、处理后输出。
        // 因此这不再是一条错误，而是一条**形态说明** —— 级别降到 `info`，
        // 内容改为"本实例走 Capture；想直接编辑宿主片段就从 ARA 入口插入"。
        log::info!(
            "[ara] no ARA binding {:.0}s after setActive: {host} is running this instance as a \
             plain insert, so this instance uses the Capture path (capture the host's playback \
             into the plug-in). Insert the plug-in from the host's ARA entry instead if you want \
             to edit the host's clips directly.",
            elapsed.as_secs_f32()
        );
    }

    /// 初始化在宿主UI/model线程保存有引用的typed扩展，不在actor或process调用REAPER API。
    /// # Safety
    /// context须为宿主初始化期间活FUnknown。
    pub(crate) unsafe fn bind_host_adapter(&self, context: *mut std::ffi::c_void) {
        // SAFETY: 调用方保证 context 是宿主初始化期间存活的 FUnknown。
        unsafe { self.bind_host_adapter_inner(context, false) }
    }

    /// 在 `setActive` 等"宿主明确进入新阶段"的入口强制重建宿主适配器。
    ///
    /// 【为什么需要强制变体】`bind_host_adapter` 在文档已绑定时要求
    /// `host_query_stamp()` 成功；工程刚加载、模型还没发布时它必然失败，于是
    /// 适配器不会被创建。而宿主可能在另一个线程上调用 `setActive`，
    /// 旧适配器的线程亲和性不再成立。强制变体只要求组件未关闭：宿主接口属于
    /// 组件，不属于文档，可以在任何阶段重建。
    ///
    /// # Safety
    /// `context` 必须是本组件持有的、在组件生命周期内有效的宿主接口。
    pub(crate) unsafe fn rebind_host_adapter(&self, context: *mut std::ffi::c_void) {
        // SAFETY: 同 `bind_host_adapter`。
        unsafe { self.bind_host_adapter_inner(context, true) }
    }

    /// 当前宿主适配器是否与调用线程不匹配（`false` = 没有适配器或线程一致）。
    pub(crate) fn host_adapter_thread_mismatch(&self) -> bool {
        self.native_host()
            .is_some_and(|host| host.bound_thread() != std::thread::current().id())
    }

    unsafe fn bind_host_adapter_inner(&self, context: *mut std::ffi::c_void, force: bool) {
        let bound = self.document.lock_or_recover().is_some();
        let stamp = if bound && !force {
            match self.host_query_stamp() {
                Ok(stamp) => Some(stamp),
                Err(_) => return,
            }
        } else {
            None
        };
        let authorized = || {
            !self.is_closed()
                && match &stamp {
                    Some(stamp) => self.host_query_authorized(stamp),
                    // 非强制路径保持既有语义：初始化期间若文档已经出现，说明
                    // 宿主正在绑定，此时不抢建适配器；强制路径（setActive）则
                    // 只要求组件未关闭。
                    None => force || self.document.lock_or_recover().is_none(),
                }
        };
        let host = unsafe { crate::host::NativeHost::from_context(context, authorized) }
            .map(|host| crate::host::adapter::HostBackend::Native(Arc::new(host)));
        if !authorized() {
            return;
        }
        crate::log_line(&format!(
            "native host extension available={} (force={force}, thread={:?})",
            host.is_some(),
            std::thread::current().id()
        ));
        let old = std::mem::replace(&mut *self.host.lock_or_recover(), host);
        drop(old);
        self.host_geometry.lock_or_recover().take();
    }
    /// 短事务冻结真实owner/document/model/scope；整个host调用链都不持内部锁。
    fn host_query_stamp(
        &self,
    ) -> Result<(Arc<super::document::DocumentSession>, u64, u64, Vec<u64>), String> {
        let document = match self.editor_document() {
            Ok(document) => document,
            Err(error) => {
                // 【为什么这条静默早退必须留痕】它是枚举、几何刷新与前台自愈的**共同**
                // 第一行。此前它失败时什么都不写，于是"重开工程后界面停在等待宿主音频"
                // 与"宿主没给原生扩展"在日志里完全同形（T6）。
                //
                // 【为什么 Capture 实例不报】它**永远**没有 ARA 文档 —— 那不是故障，
                // 是形态。此前这一行每 10 秒报一次，40 分钟的现场日志里刷了 480 行，
                // 把真正的错误整片淹掉；而"看不到读数"正是本轮要修的第一件事。
                if self.mode() != InstanceMode::Capture {
                    hifishifter_kernel::log_limiter::emit_limited_key(
                        log::Level::Warn,
                        "ara.stamp.no-document",
                        format_args!("[ara] host query skipped (no document): {error}"),
                    );
                }
                return Err(error);
            }
        };
        let _transaction = document.transaction.lock_or_recover();
        if self.is_closed() || !document.is_alive() {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.stamp.closed",
                format_args!(
                    "[ara] host query skipped: owner closed={} document alive={}",
                    self.is_closed(),
                    document.is_alive()
                ),
            );
            return Err("host query document/owner closed".into());
        }
        let keys = match self.host_assigned_regions(&document) {
            Ok(keys) => keys,
            Err(error) => {
                // 【为什么单独一条 key】`assignments` 为空是本轮症状二/三最可疑的
                // 静默早退点（见 `bind_to_document` 的 rebind 分支）。把它与
                // "文档已关闭"分开计数，真机日志才能直接指认。
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Warn,
                    "ara.stamp.no-assignment",
                    format_args!("[ara] host query skipped (no usable assignment): {error}"),
                );
                return Err(error);
            }
        };
        let model = document.revision.load(Ordering::Acquire);
        let scope = document.scope_revision.load(Ordering::Acquire);
        drop(_transaction);
        Ok((document, model, scope, keys))
    }
    fn host_query_authorized(
        &self,
        stamp: &(Arc<super::document::DocumentSession>, u64, u64, Vec<u64>),
    ) -> bool {
        let (document, model, scope, keys) = stamp;
        let _transaction = document.transaction.lock_or_recover();
        !self.is_closed()
            && document.is_alive()
            && self
                .editor_document()
                .is_ok_and(|current| Arc::ptr_eq(&current, document))
            && document.revision.load(Ordering::Acquire) == *model
            && document.scope_revision.load(Ordering::Acquire) == *scope
            && self
                .host_assigned_regions(document)
                .is_ok_and(|current| current == *keys)
    }
    /// 本实例这一轮应当看到哪些 region —— **全仓库唯一的判据**（T1）。
    ///
    /// 【为什么必须收敛成一个函数】此前有两个各写一遍的实现：`host_assigned_regions`
    /// （inventory / 几何 / 传输的入口，round 17 给它加了"文档级派生"兜底）与
    /// `assigned_regions`（**时间线**与**渲染输入**的入口，没有兜底）。两者在同一个
    /// tick 里给出**互相矛盾**的结论：几何刷新说"这个 item 我认得"，工作区投影说
    /// "这条 region 不归你"。界面于是显示"有 item、有轨道、但 clip 没内容"。
    ///
    /// 【三条纪律】
    /// 1. **逐条过滤，不整批作废**：宿主在一次分割里会把已销毁父段的 key 一并重发
    ///    （`workspace_scope_locked` 的注释早已记录这个形状），整批 `resolve` 会让
    ///    **一条**陈旧 key 把**整份**工作区打成空。
    /// 2. **空集合要区分"宿主没给"与"宿主给了空的"**：前者回退文档级，后者才是真空。
    /// 3. **不可用不是错误**：返回 `AssignedScope`（带来源），由调用方决定怎么呈现。
    ///
    /// 【`allow_document_derived` 为什么是参数而不是常量】只有**并集语义**的调用方
    /// 可以派生：`workspace_scope_locked` 本来就是"本文档全部 owner 的并集"，
    /// 而 REAPER 的一个 ARA 文档上同时挂着多条轨道的实例，派生出来的并集正是它要的。
    /// **渲染输入不能派生** —— 每个实例只许渲染自己的分配，否则同一段音频会被渲染两次。
    /// `role` 为 `None` 时取**全部角色**的并集（枚举/几何路径：它们服务整条轨道），
    /// `Some(bits)` 时只取该角色（**渲染路径**：playback 实例只许渲染自己那一份）。
    pub(crate) fn assigned_scope(
        &self,
        document: &super::document::DocumentSession,
        role: Option<u32>,
        allow_document_derived: bool,
    ) -> Result<AssignedScope, String> {
        let pick = |map: &std::collections::HashMap<i32, Vec<u64>>| -> BTreeSet<u64> {
            match role {
                Some(bits) => map
                    .get(&(bits as i32))
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .collect(),
                None => map.values().flatten().copied().collect(),
            }
        };
        let mut keys = {
            let assignments = self.assignments.lock_or_recover();
            pick(&assignments)
        };
        let declared = {
            let sequences = self.sequences.lock_or_recover();
            pick(&sequences)
        };
        let members = document.sequence_regions.lock_or_recover();
        // 【为什么两者都空才派生】只要宿主给过任何一条线索（region key 或 sequence），
        // 就以它为准 —— 派生只补"本实例还没被告知"，不覆盖"已经被告知为空"。
        let derived = keys.is_empty() && declared.is_empty() && allow_document_derived;
        let mut source = if derived {
            ScopeSource::DocumentDerived
        } else {
            ScopeSource::HostAssignment
        };
        if derived {
            for regions in members.values() {
                keys.extend(regions.iter().copied());
            }
        } else {
            for sequence in declared {
                // 【为什么未知 sequence 只跳过】宿主先建 sequence、后建 region 是正常
                // 时序，其间该键必然缺席。此前 `ok_or(...)?` 让整份 scope 失败。
                if let Some(regions) = members.get(&sequence) {
                    keys.extend(regions.iter().copied());
                }
            }
        }
        drop(members);
        // 【纪律 1：逐条过滤】`resolve` 是**整批**语义，一条坏 key 就整体 Err。
        // 写路径（assignment observer）早已改成逐条 `owner_of`，读路径必须同口径，
        // 否则同一次分割在两处得到不同结论。
        let mut dropped = 0usize;
        let keys = {
            let owners = region_owners().lock_or_recover();
            keys.into_iter()
                .filter(|key| {
                    if owners.owner_of(*key) == Some(document.id) {
                        true
                    } else {
                        dropped += 1;
                        false
                    }
                })
                .collect::<Vec<_>>()
        };
        if dropped > 0 {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.scope.dropped",
                format_args!(
                    "[ara] {dropped} assigned region key(s) are not owned by this document \
                     (destroyed, or re-sent by the host after a split); the rest of the scope \
                     is unaffected"
                ),
            );
        }
        if keys.is_empty() {
            source = ScopeSource::Empty;
        }
        Ok(AssignedScope { keys, source })
    }

    /// 几何绑定必须覆盖owner的全部已分配角色，不能只拿playback子集隐藏editor歧义。
    fn host_assigned_regions(
        &self,
        document: &super::document::DocumentSession,
    ) -> Result<Vec<u64>, String> {
        // 枚举 / 几何 / 传输三条路径都是**读**：它们可以接受文档级派生（见
        // `assigned_scope` 的说明），但绝不能因为一条陈旧 key 就整份失败。
        Ok(self.assigned_scope(document, None, true)?.keys)
    }
    /// 仅UI/model线程读取；唯一assignment先成立，位置/长度仅用于绑定后相容核对。
    ///
    /// 这条路径的身份来自宿主直接 parent take —— 是强证据，因此**只在恰好一个
    /// region 时**成立。多 region（folder 轨上的 FX）走
    /// [`Self::host_geometries`]，那里没有直接 parent 可用，只能按几何唯一匹配。
    pub(crate) fn host_geometry(&self) -> Result<crate::host::geometry::BoundHostGeometry, String> {
        let stamp = self.host_query_stamp()?;
        let [region_key] = stamp.3.as_slice() else {
            return Err("host geometry requires exactly one assigned ARA region".into());
        };
        let region = stamp
            .0
            .regions
            .lock_or_recover()
            .get(region_key)
            .cloned()
            .ok_or("assigned ARA region unavailable")?;
        let host = self.native_host().ok_or("host extension unavailable")?;
        let geometry = host.geometry(|| self.host_query_authorized(&stamp))?;
        if !compatible(geometry.start_sec, region.start_in_playback_time)
            || !compatible(geometry.duration_sec, region.duration_in_playback_time)
        {
            return Err(BINDING_GEOMETRY_INCOMPATIBLE.into());
        }
        if !self.host_query_authorized(&stamp) {
            return Err("host geometry authorization revoked".into());
        }
        Ok(crate::host::geometry::BoundHostGeometry {
            region_key: *region_key,
            geometry,
        })
    }

    /// 多 region owner（挂在 REAPER folder 轨上的 FX）逐 region 解析宿主几何。
    ///
    /// 【为什么不能退回"取第一个"】后续所有宿主写回都沿 region → item 这条边定位
    /// 对象。拿一个 take 的几何去冒充整组，会把 item GUID 绑到**错误的** region 上，
    /// 于是移动/切分/fade/音量全部可能落到别的 item。因此这里对每个 region 单独
    /// 求唯一匹配，匹配不到就**不绑定**，并在日志里如实报出未绑定的数量。
    ///
    /// 候选 item 与 GUI 清单同源（FX 自己的轨道 + folder 全部后代轨道），沿既有
    /// `ui_folder_tracks` 的线程/授权/预算纪律取得。
    fn host_geometries(&self) -> Result<Vec<crate::host::geometry::BoundHostGeometry>, String> {
        let stamp = self.host_query_stamp()?;
        if stamp.3.is_empty() {
            return Err("no assigned ARA region".into());
        }
        let host = self.native_host().ok_or("host extension unavailable")?;
        let authorized = || self.host_query_authorized(&stamp);
        let candidates = host
            .ui_folder_tracks(&authorized)?
            .into_iter()
            .flat_map(|track| track.items.into_iter().map(|item| item.geometry))
            .collect::<Vec<_>>();
        let mut bound = Vec::new();
        let mut unmatched = Vec::new();
        for key in &stamp.3 {
            let region = stamp
                .0
                .regions
                .lock_or_recover()
                .get(key)
                .cloned()
                .ok_or("assigned ARA region unavailable")?;
            // 上一次绑定：给几何全等的候选（复制出来的安全副本等）一个稳定性偏好。
            let preferred = stamp.0.region_items.lock_or_recover().get(key).cloned();
            match unique_geometry_for_region(&region, &candidates, preferred.as_deref()) {
                Some(geometry) => bound.push(crate::host::geometry::BoundHostGeometry {
                    region_key: *key,
                    geometry,
                }),
                None => unmatched.push(*key),
            }
        }
        // 一个宿主 item 只能被**一个** region 认领。两条 region 绑到同一个 item 会让
        // 后续所有写回指向同一对象：移动、切分、fade、音量互相覆盖，且没有任何一侧
        // 会报错。
        //
        // 【为什么不再"全部丢弃"】旧实现把重复认领的候选**全丢**，于是"同一位置两个
        // 几何全等的 item"会两个都不绑定、两个 clip 一起消失（用户报障："片段凭空
        // 没了"）。现在按 region key 的**稳定顺序**保留第一个认领者，其余照常记账 ——
        // 至少不会一次丢掉全部；剩下的那个 region 仍走 unmatched，如实报出。
        let mut seen = std::collections::HashSet::<String>::new();
        let claimed = bound.len();
        bound.retain(|entry| seen.insert(entry.geometry.item_id.clone()));
        // 一次说清、且只在读数变化时输出（见 `report_binding`）。此前这里是两条无条件
        // 日志（INFO + WARN），在 folder 父轨场景下以 10Hz 重放 35 分钟共 16,769 条。
        self.report_binding(&BindingReport {
            assigned: stamp.3.clone(),
            bound: bound.iter().map(|entry| entry.region_key).collect(),
            unmatched: unmatched.clone(),
            dropped: claimed.saturating_sub(bound.len()),
            candidates: candidates.len(),
        });
        if bound.is_empty() {
            return Err(BINDING_NO_UNIQUE_ITEM.into());
        }
        if !self.host_query_authorized(&stamp) {
            return Err("host geometry authorization revoked".into());
        }
        Ok(bound)
    }

    /// 报告一次 region → 宿主 item 的绑定读数；**只在读数变化时**输出。
    ///
    /// 【为什么必须去重】见 [`BindingReport`]：folder 父轨上的"2 个 region 对 1 个候选"
    /// 是结构性不匹配，100ms 内不会自愈。无条件日志会跟着 UI 定时器无限重放。
    ///
    /// 【为什么还要限流兜底】指纹是"变化检测"，限流是"变化速率上限"。两者叠加覆盖两种
    /// 失败模式：指纹包含会漂移的字段 ⇒ 变化检测失效，由限流兜住；读数在两个值之间
    /// 高频抖动 ⇒ 变化检测每次都为真，同样由限流兜住（键含指纹，两个值各自计数）。
    /// 最坏情况是 10 秒一条，与"刷屏"已经不沾边。
    fn report_binding(&self, report: &BindingReport) {
        if !self.binding_report_is_new(report) {
            return;
        }
        let detail = format!(
            "{} assigned, {} bound, {} unmatched, {} dropped (candidates={})",
            report.assigned.len(),
            report.bound.len(),
            report.unmatched.len(),
            report.dropped,
            report.candidates
        );
        // 键取**读数本身**：不同读数各自限流，因此真实变化不会被别的读数挤掉。
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            &format!("ara:binding-report:{detail}"),
            format_args!("[ara] region→item binding: {detail}"),
        );
    }

    /// 这份读数是否**尚未报告过**；是则记下它。纯状态判定，可独立单测。
    ///
    /// 【为什么与输出分开】去重逻辑是这条修复的核心（它决定"刷屏"会不会回来），
    /// 必须能脱离日志后端被断言 —— 与 `log_limiter::limited_decision` 同一手法。
    fn binding_report_is_new(&self, report: &BindingReport) -> bool {
        // 全部绑上 = 没有可抱怨的读数。顺手清掉指纹，让下一次真实失败**必然**输出
        // （否则"失败 → 成功 → 同样的失败"会被去重吞掉第二条）。
        if report.unmatched.is_empty() && report.dropped == 0 {
            self.binding_report.lock_or_recover().take();
            return false;
        }
        let mut last = self.binding_report.lock_or_recover();
        if last.as_ref() == Some(report) {
            return false;
        }
        *last = Some(report.clone());
        true
    }

    /// 当前的重枚举间隔：无失败时是 [`HOST_ENUMERATION_INTERVAL`]，连续结构性失败时退避。
    ///
    /// 【为什么用绝对档位而不是乘法】档位表要能一眼看出"最坏情况多久枚举一次"。乘法
    /// 形式（100ms × 2^n）在 n 大时会越过 4 秒上限，得再补一次钳制 —— 不如直接查表。
    fn enumeration_interval(&self) -> std::time::Duration {
        match self
            .enumeration_backoff
            .load(std::sync::atomic::Ordering::Acquire)
        {
            0 => HOST_ENUMERATION_INTERVAL,
            1..=2 => std::time::Duration::from_millis(250),
            3..=5 => std::time::Duration::from_secs(1),
            _ => std::time::Duration::from_secs(4),
        }
    }

    /// 按一次几何求解的结果更新退避档位（见 [`ExtensionOwner::enumeration_interval`]）。
    fn note_geometry_outcome(
        &self,
        geometry: &Result<Vec<crate::host::geometry::BoundHostGeometry>, String>,
        revision: Option<i32>,
    ) {
        let Err(error) = geometry else {
            self.enumeration_backoff
                .store(0, std::sync::atomic::Ordering::Release);
            *self.last_failed_revision.lock_or_recover() = None;
            return;
        };
        if !is_structural_binding_failure(error) {
            // 授权撤销 / 文档关闭 / 扩展接口消失：状态变化，不是结构性问题。
            // 退避它们只会拖慢恢复，且它们本来就不会以 10Hz 重放。
            return;
        }
        let mut last = self.last_failed_revision.lock_or_recover();
        if *last != Some(revision) {
            // 宿主几何代次变了（用户真的动了工程）⇒ 回到全速，本次不计档。
            // 【为什么不在这里也 +1】那会让"改完工程"的第一次重试落在 250ms 档，
            // 而不是 100ms —— 退避不得牺牲响应速度。
            *last = Some(revision);
            self.enumeration_backoff
                .store(0, std::sync::atomic::Ordering::Release);
            return;
        }
        let _ = self.enumeration_backoff.fetch_update(
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
            |level| Some(level.saturating_add(1)),
        );
    }

    /// actor/worker只可消费UI冻结的Rust值，不沿此访问器调用host；代次变化明确不可用。
    // 只读几何元数据访问器保留，供worker消费冻结值。
    #[cfg(test)]
    pub(crate) fn host_geometry_metadata(
        &self,
    ) -> Result<Vec<crate::host::geometry::BoundHostGeometry>, String> {
        let stamp = self.host_query_stamp()?;
        let cached = self
            .host_geometry
            .lock_or_recover()
            .clone()
            .ok_or("host geometry has not been sampled on model/UI thread")?;
        if cached.model != stamp.1 || cached.scope != stamp.2 || !self.host_query_authorized(&stamp)
        {
            return Err("host geometry metadata superseded".into());
        }
        cached.value
    }
    /// 调用者已持所属doc事务，只消费已授权Rust缓存，不嵌套加锁或调用宿主。
    ///
    /// 返回**当前仍被分配**的 region 各自的绑定；不再要求"恰好一个" —— folder 轨上
    /// 的 FX 会被分配整组 region，要求唯一等于让整条投影链（fade / mute / item 身份）
    /// 全部静默失效。仍然只返回 region 与缓存都同意的那部分：缓存过期、region 已撤销、
    /// 或该 region 没匹配到唯一 item 的，一律不出现。
    pub(crate) fn host_geometries_locked(
        &self,
        document: &super::document::DocumentSession,
    ) -> Vec<crate::host::geometry::BoundHostGeometry> {
        if self.is_closed()
            || !document.is_alive()
            || !self
                .editor_document()
                .is_ok_and(|current| std::ptr::eq(Arc::as_ptr(&current), document))
        {
            return Vec::new();
        }
        let Some(cached) = self.host_geometry.lock_or_recover().clone() else {
            return Vec::new();
        };
        if cached.model != document.revision.load(Ordering::Acquire)
            || cached.scope != document.scope_revision.load(Ordering::Acquire)
        {
            return Vec::new();
        }
        let Ok(keys) = self.host_assigned_regions(document) else {
            return Vec::new();
        };
        match cached.value {
            Ok(bound) => bound
                .into_iter()
                .filter(|entry| keys.contains(&entry.region_key))
                .collect(),
            Err(_) => Vec::new(),
        }
    }
    /// 最近一次清单刷新得出的宿主音频读数；GUI 载荷读它。
    ///
    /// 只读快照，不触发宿主调用 —— 与 `host_geometries_locked` 同一纪律：调用方已经
    /// 在事务里，这里不得再枚举宿主。
    pub(crate) fn host_audio_status(&self) -> HostAudioStatus {
        *self.host_audio.lock_or_recover()
    }
    /// 完整刷新：时钟 + 几何 + 清单。写回路径与测试用这个（必须**同步**生效）。
    pub(crate) fn refresh_host_transport(&self) {
        self.refresh_host_transport_inner(false);
    }

    /// 前台恢复/重新激活时主动重同步一次。
    ///
    /// 【为什么不能等 UI 定时器】后台期间定时器可能停摆；回来后第一次 tick 仍会
    /// 命中"令牌未变 + 1 s 内采样过"的早退，`ui_tracks` 与 `region_items` 可能是
    /// 暂停前的快照。这里清掉采样时刻，强制一次完整枚举，并让每个 renderer 重排
    /// 快照 —— 用户切回前台后不必再等下一次编辑才恢复。
    pub(crate) fn resume_resync(&self) {
        if let Ok(document) = self.editor_document() {
            *document.ui_inventory_sampled_at.lock_or_recover() = None;
        }
        self.refresh_host_geometry();
        self.refresh_ui_inventory();
        if let Ok(document) = self.editor_document() {
            for owner in document.renderer_owners() {
                owner.prepare();
            }
        }
    }

    /// UI 定时器路径：时钟每次跟，几何/清单按 [`ExtensionOwner::enumeration_interval`] 节流。
    ///
    /// 【为什么时钟与枚举分开节流】定时器是 20ms（50Hz）。播放光标必须跟得上
    /// （`host.sample` 很便宜），但几何/清单枚举即使有代次缓存，每 tick 仍要付
    /// `GetProjectStateChangeCount` 等若干宿主调用，还会与 actor/渲染线程争同一把
    /// document 事务。宿主真变了时下一次节流点会捕捉到（≤100ms），用户不可感。
    ///
    /// 【为什么节流间隔是动态的】连续的结构性绑定失败（folder 父轨上"2 个 region 对
    /// 1 个候选"）不会在 100ms 内自愈，却会以这个节奏完整枚举宿主工程。退避把这种
    /// "永不成功"的重试降到 4 秒一次，而宿主几何代次一变就立刻回到 100ms。
    ///
    /// 【为什么不把节流放进 `refresh_host_transport` 本身】写回路径（导入/分割/
    /// 剪贴板）与测试依赖它**同步**把新几何取回来；节流在那里会把"刚写完就读到新值"
    /// 变成"下次再看"。所以节流只属于定时器这一个调用方。
    pub(crate) fn refresh_host_transport_tick(&self) {
        // 【T5：前台恢复的本地信号】编辑器窗口是 `WS_CHILD`，收不到 `WM_ACTIVATEAPP`；
        // 而 `resume_resync()` 的唯一调用点是 `setActive(true)`，宿主切窗口焦点时不会重发。
        // 于是"后台→前台"没有任何入口。这里用**定时器自身的间隔**补上：20 ms 的定时器
        // 被挂起/降频后，恢复后的第一次 tick 与上一次之间的间隔必然远大于一个周期。
        // 它不需要任何宿主配合，也不改变线程模型。
        let resumed = {
            let mut last = self.last_view_tick.lock_or_recover();
            let now = std::time::Instant::now();
            let resumed = last.is_some_and(|at| now.duration_since(at) >= FOREGROUND_RESUME_GAP);
            *last = Some(now);
            resumed
        };
        if resumed {
            log::info!(
                "[ara] the view timer resumed after a long gap; resyncing host geometry and inventory"
            );
            self.resume_resync();
        }
        // 【T4：自愈必须先于 gating】`refresh_host_transport_inner` 的第一行是
        // `host_query_stamp()` 的早退，而自愈原本写在它**之后**。恰恰在重开工程 /
        // `assignments` 为空时 stamp 拿不到 —— 于是最需要自愈的场景一行都不执行，
        // 且完全沉默（这是"round 16 加了自愈却仍然复现"的直接原因）。
        //
        // "我们多久没成功采过样"不需要任何授权：它是本侧的时钟事实。把"记录需求"
        // 与"执行动作"分开之后，随后的枚举仍会被原来的 gating 拦下（正确），
        // 但恢复需求已经被记下来了，下一次 stamp 可用时会真正执行。
        self.heal_stale_inventory_stamp();
        // 【T23】视图定时器是"宿主不再回调"时唯一还会跑的节拍：在这里提出收口请求。
        self.request_settle_if_overdue();
        self.refresh_host_transport_inner(true);
    }

    /// **T23**：把"宿主不再收口"这件事从插件侧提出来。
    ///
    /// 【为什么需要它】`ready` 回到真只依赖宿主发起的四个入口。宿主一旦 `beginEditing`
    /// 之后再也不收口，或者积压了模型变化却不再调 `notifyModelUpdates`，插件就永久
    /// 停在未就绪 —— 用户唯一的出路是删掉插件再加。
    ///
    /// 本函数只**置位**（文档上的两个原子量），真正的收口由 ARA 控制器在它下一次
    /// 回调里执行（见 `ControllerAdapter::service_forced_settle`）。桥的 `PluginRuntime`
    /// 不是 `Send`/`Sync`，从这条线程直接调用它是未定义行为。
    ///
    /// 【为什么用 `try_lock`】音频线程也会调它。拿不到文档槽就跳过 —— 下一次块
    /// （10 ms 后）会再来，而"宽限期"本来就是秒级，漏一拍没有影响。
    pub(crate) fn request_settle_if_overdue(&self) {
        let Ok(slot) = self.document.try_lock() else {
            return;
        };
        let document = slot.as_ref().and_then(std::sync::Weak::upgrade);
        drop(slot);
        let Some(document) = document else {
            return;
        };
        document.request_settle_if_overdue();
    }

    /// 距上次成功采样超过 `STALE_SAMPLE_GRACE` 就清掉采样时刻，让下一次真正执行强制重采。
    ///
    /// 【为什么它不取 `host_query_stamp`】那条路径会因为"没有 document / 没有分配 /
    /// 已关闭"而失败，而这三者正是"后台恢复后界面不刷新"的常见形状。时钟维护
    /// 不能依赖它们。
    fn heal_stale_inventory_stamp(&self) {
        let Ok(document) = self.editor_document() else {
            return;
        };
        let stale = document
            .ui_inventory_sampled_at
            .lock_or_recover()
            .is_none_or(|at| at.elapsed() >= STALE_SAMPLE_GRACE);
        if stale {
            *document.ui_inventory_sampled_at.lock_or_recover() = None;
        }
    }

    fn refresh_host_transport_inner(&self, throttled: bool) {
        let Ok(stamp) = self.host_query_stamp() else {
            return;
        };
        let host = self.native_host();
        if let Some(host) = host {
            if let Ok((position, playing)) = host.sample(|| self.host_query_authorized(&stamp)) {
                // getter可能重入关闭/改scope，发布与撤销共用最终短事务。
                let _transaction = stamp.0.transaction.lock_or_recover();
                if !self.is_closed()
                    && stamp.0.is_alive()
                    && stamp.0.revision.load(Ordering::Acquire) == stamp.1
                    && stamp.0.scope_revision.load(Ordering::Acquire) == stamp.2
                {
                    if let Some(clock) = self.clock.get() {
                        clock.publish_host(position, playing);
                    }
                }
            }
        }
        if !self.host_query_authorized(&stamp) {
            return;
        }
        // 【前台恢复不依赖 setActive】后台期间定时器可能停摆；超过一个刷新周期
        // 没有成功采样过，就强制一次完整枚举（清掉采样时刻，让 `refresh_ui_inventory`
        // 跳过"令牌未变"的早退），并跳过实例级节流。强制重采在事实未变时无副作用
        // （见 `refresh_ui_inventory` 的 `token_changed` 分支）。
        let stale_sample = stamp
            .0
            .ui_inventory_sampled_at
            .lock_or_recover()
            .is_none_or(|at| at.elapsed() >= STALE_SAMPLE_GRACE);
        if stale_sample {
            *stamp.0.ui_inventory_sampled_at.lock_or_recover() = None;
        }
        if throttled && !stale_sample {
            let mut last = self.last_enumeration.lock_or_recover();
            let now = std::time::Instant::now();
            if last.is_some_and(|last| now.duration_since(last) < self.enumeration_interval()) {
                return;
            }
            *last = Some(now);
        } else if throttled {
            *self.last_enumeration.lock_or_recover() = Some(std::time::Instant::now());
        }
        self.refresh_host_geometry();
        // editor-only通常没有所属take；一个工程GUI必须驱动真正隐藏playback实例的只读采集。
        // renderer_owners仅返回本真实文档的活租约，不扫描全局实例，也不借名称/位置找take。
        for owner in stamp.0.renderer_owners() {
            if !self.host_query_authorized(&stamp) {
                break;
            }
            if owner.renders_playback() && !std::ptr::eq(self, Arc::as_ptr(&owner)) {
                owner.refresh_host_geometry();
            }
        }
        self.refresh_ui_inventory();
    }
    /// GUI清单按项目change/model/scope代次刷新，不每20ms枚举所有item，也不借活动工程。
    pub(crate) fn refresh_ui_inventory(&self) {
        let Ok(stamp) = self.host_query_stamp() else {
            return;
        };
        let Some(host) = self.native_host() else {
            // 没有原生扩展接口 —— 这是 **ARA-only 宿主**（Studio One / Cubase / Nuendo /
            // Logic …）的正常形态，不是"不受支持"：时间线有内容、音频能出、参数能调，
            // 只读的只有**片段几何写回**，而那由能力位门控（`HostCapabilities`），
            // 不需要任何横幅。
            //
            // 【为什么这里不再区分"未识别的宿主"】两者此前共用一个三元表达式，而
            // `ara_only` 那一支**恒为真**：`bind_host_adapter` 只会存 `Some(Native)` 或
            // `None`，而 `host_backend()` 把 `None` 映射成 `AraOnly`。所以"未识别"那一态
            // 从来没有产出者，而它的文案（「本宿主尚不支持 / 时间线会是空的」）对每一个
            // 真能走到编辑器里的宿主都是**假的**。真正"既不提供原生扩展、也不提供 ARA"
            // 的宿主根本不会有文档，插件报的是 `ARA_NOT_BOUND` 那条终态错误，
            // 不会表现为"空白时间线 + 一条猜错方向的横幅"。
            //
            // 因此这里只把读数复位为"无可抱怨"，不再产生任何宿主支持类提示。
            *self.host_audio.lock_or_recover() = HostAudioStatus::default();
            // 【ARA-only 也必须推进 render_epoch】原生扩展缺席不代表"没有渲染输入"：
            // ARA 模型自己就能给出时间拉伸与内容淡化（`RenderItemFacts` 的 ARA 三字段）。
            // 此前这里直接 return ⇒ `render_facts` 永不更新、`render_epoch` 永不推进 ⇒
            // 宿主在 ARA 侧改了这两项，插件继续用旧配置渲染（静默的正确性 bug）。
            // ARA 拿不到方向/循环/声道（ARA 表达不到，见 `ara::mapping::LOST_FIELDS`），
            // 所以这里只登记 ARA 能给出的字段，不假装有原生事实。
            self.refresh_ara_render_facts(&stamp);
            return;
        };
        let allowed = || self.host_query_authorized(&stamp);
        let Ok(change) = host.geometry_revision(allowed) else {
            return;
        };
        let token = (change, stamp.1, stamp.2);
        // 令牌相同 **且** 距上次完整采集还没超过 `FACTS_REFRESH_INTERVAL` 才早退。
        // 光看令牌会漏掉 take/item 级事实变化（倒放 / 声道模式），因为宿主未必为它们
        // 推进 `GetProjectStateChangeCount` —— 那会让 `render_facts` 永不更新，正是
        // "倒放后波形变了、声音没变"的第二层缺口（见 `ui_inventory_sampled_at` 与
        // `FACTS_REFRESH_INTERVAL`）。
        let fresh = stamp
            .0
            .ui_inventory_sampled_at
            .lock_or_recover()
            .is_some_and(|at| at.elapsed() < FACTS_REFRESH_INTERVAL);
        let stored = *stamp.0.ui_inventory_stamp.lock_or_recover();
        if self.host_query_authorized(&stamp) && fresh && stored == Some(token) {
            return;
        }
        // 令牌本身是否变了。强制重采（令牌没变、只是采样过期）时若事实也没变，
        // 就不发布、不 bump `ui_geometry_revision`、不失效渲染 —— 兜底采样必须**无副作用**，
        // 否则空闲工程会被 1Hz 的假变更搅得前端反复重绘。
        let token_changed = stored != Some(token);
        let mut tracks = std::collections::BTreeMap::new();
        let mut seen = std::collections::BTreeSet::new();
        // 诊断计数：多少 item 认领到了 region（没认领的只能是无源占位）。
        let mut item_count = 0usize;
        let mut claimed_items = 0usize;
        // 本实例所在轨道的 `I_FOLDERDEPTH`；读不出来保持 `None`（见 `HostAudioState`）。
        let mut fx_folder_depth: Option<i32> = None;
        for owner in stamp.0.renderer_owners() {
            if !allowed() {
                return;
            }
            let Some(host) = owner.native_host() else {
                continue;
            };
            let own_allowed = || {
                allowed()
                    && !owner.is_closed()
                    && owner
                        .editor_document()
                        .is_ok_and(|doc| Arc::ptr_eq(&doc, &stamp.0))
            };
            let Ok(parent) = host.direct_track_target(&own_allowed) else {
                continue;
            };
            // 只读本实例那条轨道的单值，不枚举整个工程。放在 `seen` 去重之前：本实例
            // 可能不是本轮第一个被处理的 owner，去重会让它被跳过。
            if std::ptr::eq(self, Arc::as_ptr(&owner)) {
                fx_folder_depth = parent.folder_depth(&own_allowed).ok();
            }
            // 父轨已枚举过 ⇒ 它的全部后代也已经在 `tracks` 里，不必重走一遍。
            if seen.contains(parent.inventory_guid()) {
                continue;
            }
            // FX 挂在 folder 轨上时，folder 轨自身通常没有 item：连同后代一起枚举，
            // 否则整组在插件里"看不见"（ARA 侧没有 folder 概念，只能从宿主清单取）。
            let Ok(own_tracks) = host.ui_folder_tracks(&own_allowed) else {
                continue;
            };
            for mut track in own_tracks {
                if !seen.insert(track.guid.clone()) {
                    continue;
                }
                // 【锁序】按 `document.rs` 声明的层级取锁：`timeline`(2) 先于
                // `clip_ids`(4)。此前这里是反的 —— 与 `session.rs::analysis_timeline`
                // 的正序共存，构成真实的死锁窗口。
                let timeline = stamp.0.timeline.lock_or_recover();
                let ids = stamp.0.clip_ids.lock_or_recover();
                item_count += track.items.len();
                for candidate in stamp.0.renderer_owners() {
                    // 该 owner 可能同时持有多个 region（folder 轨上的 FX）：只要本轨有
                    // 任一 item 被它的某个 region 绑定，就把轨道身份认领到那个 clip 上。
                    let matched =
                        candidate
                            .host_geometries_locked(&stamp.0)
                            .into_iter()
                            .find(|bound| {
                                track
                                    .items
                                    .iter()
                                    .any(|item| item.geometry.item_id == bound.geometry.item_id)
                            });
                    if let Some(bound) = matched {
                        claimed_items += track
                            .items
                            .iter()
                            .filter(|item| item.geometry.item_id == bound.geometry.item_id)
                            .count();
                        if let Some(id) = ids.get(&bound.region_key) {
                            if let Some(clip) = timeline
                                .as_ref()
                                .and_then(|t| t.clips.iter().find(|c| &c.id == id))
                            {
                                track.id = clip.track_id.clone();
                                break;
                            }
                        }
                    }
                }
                drop(timeline);
                drop(ids);
                if track.id.starts_with("host-track-") {
                    if let Some(old) = stamp.0.ui_tracks.lock_or_recover().get(&track.guid) {
                        track.id = old.id.clone();
                    }
                }
                tracks.insert(track.guid.clone(), track);
            }
        }
        // 分类只用可核实的事实（见 `classify_host_audio`）。folder 结构读不出来时
        // `None` ⇒ 退化成"等待中"，绝不给出一个可能指错方向的 folder 提示。
        //
        // 绑定侧读数取自 `binding_report`：它与刷屏日志**同源** —— 同一份事实既决定
        // 日志输出，也决定用户看到的提示。`None` 表示"全部绑上了"（或还没求解过），
        // 此时两个 region 计数传 0，分类退回既有的 folder / awaiting 判定。
        let (regions_assigned, regions_unmatched) = self
            .binding_report
            .lock_or_recover()
            .as_ref()
            .map(|report| (report.assigned.len(), report.unmatched.len()))
            .unwrap_or((0, 0));
        let status = classify_host_audio(
            fx_folder_depth.is_some_and(|depth| depth >= 1),
            item_count,
            claimed_items,
            regions_assigned,
            regions_unmatched,
        );
        if !allowed() || host.geometry_revision(allowed).ok() != Some(change) {
            return;
        }
        // 采样会改变渲染结果的宿主事实。**两个来源**：
        //  - 原生扩展：方向 / 循环源 / 声道模式（ARA 表达不到，见 `ara::mapping::LOST_FIELDS`）；
        //  - ARA 模型：时间拉伸 / 基于内容的淡化（ARA region 自己就给出）。
        // 与 `project_host_take_facts_locked` 用**同一个** active-take 口径，否则"清单
        // 说倒放、渲染播种说正放"会再次分叉。
        let mut facts =
            crate::host::geometry::RenderFacts::from_items(tracks.values().flat_map(|track| {
                track.items.iter().map(|item| {
                    let active = item.takes.iter().find(|take| take.active);
                    let geometry = active.map(|take| &take.geometry).unwrap_or(&item.geometry);
                    (
                        item.geometry.item_id.clone(),
                        crate::host::geometry::RenderItemFacts {
                            reversed: active.is_some_and(|take| take.reversed),
                            loop_enabled: geometry.loop_source,
                            channel_mode: geometry.channel_mode,
                            // ARA 侧字段在下面按 region 叠加；这里先给安全默认。
                            timestretch: false,
                            content_fade_head: false,
                            content_fade_tail: false,
                        },
                    )
                })
            }));
        // ARA 侧的细化：把 `region_items` 里已认领的 region 事实叠加到对应 item 上。
        // 这份叠加让"宿主在 ARA 侧改了时间拉伸 / 内容淡化"也进 `RenderFacts` —— 与原生
        // 字段走同一条失效通路，不必再依赖某一处记得手动 bump。
        {
            let regions = stamp.0.regions.lock_or_recover();
            let items = stamp.0.region_items.lock_or_recover();
            for (key, item_id) in items.iter() {
                let (Some(entry), Some(region)) = (facts.items.get_mut(item_id), regions.get(key))
                else {
                    continue;
                };
                entry.timestretch = region.is_timestretch_enabled;
                entry.content_fade_head = region.has_content_based_fade_at_head;
                entry.content_fade_tail = region.has_content_based_fade_at_tail;
            }
        }
        let _transaction = stamp.0.transaction.lock_or_recover();
        if self.is_closed()
            || !stamp.0.is_alive()
            || stamp.0.revision.load(Ordering::Acquire) != stamp.1
            || stamp.0.scope_revision.load(Ordering::Acquire) != stamp.2
        {
            return;
        }
        // 【这是"宿主改了、插件立刻重渲染"的唯一通路】任何一项渲染事实变化都推进
        // `render_epoch`，于是 `PreparedVersion`（渲染缓存键）失效、`prepare_job` 不再
        // 早退。此前只有**淡变长度**有这条通路，倒放/声道/循环源都没有 —— 用户报障
        // "倒放后波形更新了、声音没变，编辑一下才对"正是这个缺口。
        //
        // 【为什么只置标志、通知留给文档级入口】发现变化的实例不一定是持有快照的实例
        // （REAPER 按轨道管理 ARA 实例：定时器在编辑实例上，快照在播放实例上）。
        // `invalidate_render_input()` 会推进 epoch 并通知**全部 owner**；此处只在本
        // 事务内记下"变了"，出事务后再调用它。
        let mut facts_changed = false;
        {
            let mut previous = stamp.0.render_facts.lock_or_recover();
            if *previous != facts {
                *previous = facts;
                facts_changed = true;
            }
        }
        // 强制重采但令牌与事实都没变 ⇒ 这一轮什么也没发生：只刷新采样时刻并返回，
        // 不发布清单、不 bump `ui_geometry_revision`、不失效渲染（见上面的 `token_changed`）。
        if !token_changed && !facts_changed {
            *stamp.0.ui_inventory_sampled_at.lock_or_recover() = Some(std::time::Instant::now());
            return;
        }
        stamp
            .0
            .ui_known_tracks
            .lock_or_recover()
            .extend(tracks.values().map(|track| track.id.clone()));
        *stamp.0.ui_tracks.lock_or_recover() = tracks;
        *stamp.0.ui_inventory_stamp.lock_or_recover() = Some(token);
        // 完整采集已完成：记下时刻，让"令牌不动但 take 级事实变了"这条兜底路径
        // 在下一次 `FACTS_REFRESH_INTERVAL` 到期时强制重采（见 `ui_inventory_sampled_at`）。
        *stamp.0.ui_inventory_sampled_at.lock_or_recover() = Some(std::time::Instant::now());
        stamp.0.ui_geometry_revision.fetch_add(1, Ordering::AcqRel);
        *self.host_audio.lock_or_recover() = status;
        // 用户报障时最需要的一行：宿主清单里有多少 item，其中多少被 ARA region 认领。
        // 认领不到的那些只能以无源占位出现（见方案 Task 5.1 / R4）。
        if item_count > 0 {
            log::info!(
                "[ara] host inventory: {} track(s), {} item(s), {} claimed by an assigned region",
                stamp.0.ui_tracks.lock_or_recover().len(),
                item_count,
                claimed_items
            );
        }
        if status.state != HostAudioState::Ready {
            // 直读病灶：本实例到底拿到几条 region、FX 轨是不是 folder 父轨。下一次同样
            // 的报障不必再从位掩码与清单反推（见方案 Task 3.3）。
            let assigned = self
                .host_assigned_regions(&stamp.0)
                .map(|keys| keys.len())
                .unwrap_or(0);
            let depth = match fx_folder_depth {
                Some(depth) => depth.to_string(),
                None => "unknown".to_owned(),
            };
            log::info!("[ara] fx track: folderDepth={depth}; assigned regions={assigned}");
            log::info!(
                "[ara] {} clip(s) still waiting for host audio ({})",
                status.waiting_items,
                status.state.as_str()
            );
        }
        drop(_transaction);
        if facts_changed {
            // 渲染输入变了 ⇒ 文档级失效：推进 epoch 并通知**全部 owner**。
            // 不持 `transaction` 调用，避免与 `prepare_job` 的事务重入。
            stamp.0.invalidate_render_input();
        }
    }
    /// ARA-only 后端的渲染事实来源：只登记 ARA region 自己能给出的字段。
    ///
    /// 键用 region key 而非 item GUID —— ARA-only 下没有宿主 item 清单，item GUID 不
    /// 存在；region key 是这一后端里唯一稳定的 region 身份。原生后端另有一套按 item
    /// GUID 的键空间，两者不会同时出现（一个实例的后端在生命周期内固定）。
    fn refresh_ara_render_facts(
        &self,
        stamp: &(Arc<super::document::DocumentSession>, u64, u64, Vec<u64>),
    ) {
        let facts = {
            let regions = stamp.0.regions.lock_or_recover();
            crate::host::geometry::RenderFacts::from_items(regions.iter().map(|(key, region)| {
                (
                    format!("ara-region-{key}"),
                    crate::host::geometry::RenderItemFacts {
                        // 方向 / 循环 / 声道是**原生扩展专属**事实，ARA 表达不到 ⇒ 恒为安全默认。
                        reversed: false,
                        loop_enabled: false,
                        channel_mode: 0,
                        timestretch: region.is_timestretch_enabled,
                        content_fade_head: region.has_content_based_fade_at_head,
                        content_fade_tail: region.has_content_based_fade_at_tail,
                    },
                )
            }))
        };
        let _transaction = stamp.0.transaction.lock_or_recover();
        if self.is_closed()
            || !stamp.0.is_alive()
            || stamp.0.revision.load(Ordering::Acquire) != stamp.1
            || stamp.0.scope_revision.load(Ordering::Acquire) != stamp.2
        {
            return;
        }
        let changed = {
            let mut previous = stamp.0.render_facts.lock_or_recover();
            if *previous != facts {
                *previous = facts;
                true
            } else {
                false
            }
        };
        drop(_transaction);
        if changed {
            stamp.0.invalidate_render_input();
        }
    }
    /// 元数据单独采集，不让每个隐藏实例重复写工程clock；宿主调用始终不持内部锁。
    pub(crate) fn refresh_host_state_for_model(&self) {
        self.refresh_host_geometry();
    }
    fn refresh_host_geometry(&self) {
        let Ok(stamp) = self.host_query_stamp() else {
            return;
        };
        let host = self.native_host();
        // 【为什么没有原生扩展就直接返回】本函数采集的是**宿主 item 几何**，而
        // `document.region_items` 的唯一写者就在它里面。没有原生扩展时
        // `host_geometry()` / `host_geometries()` 的第一行就
        // `Err("host extension unavailable")` —— 什么都写不成。此前这里依赖
        // `before.is_some()`（`before` 来自原生宿主，缺席时为 `None`）让早退**永不成立**，
        // 于是每个 UI tick 都完整走一遍这条注定失败的路径。早返回等价而省事。
        if host.is_none() {
            return;
        }
        let authorized = || self.host_query_authorized(&stamp);
        let before = host
            .as_ref()
            .and_then(|host| host.geometry_revision(authorized).ok());
        if !authorized() {
            return;
        }
        let unchanged = before.is_some()
            && self
                .host_geometry
                .lock_or_recover()
                .as_ref()
                .is_some_and(|cached| {
                    cached.model == stamp.1
                        && cached.scope == stamp.2
                        && cached.change == before
                        && cached.value.is_ok()
                });
        if unchanged {
            let after = host
                .as_ref()
                .and_then(|host| host.geometry_revision(authorized).ok());
            if before == after && authorized() {
                return;
            }
        }
        if !authorized() {
            return;
        }
        // 唯一 region 时走强身份路径（宿主直接 parent take）；多 region（folder 轨上
        // 的 FX）结构上不可能有直接 parent，只能逐 region 按几何唯一匹配。
        let mut geometry = if stamp.3.len() == 1 {
            self.host_geometry().map(|bound| vec![bound])
        } else {
            self.host_geometries()
        };
        let after = host
            .as_ref()
            .and_then(|host| host.geometry_revision(authorized).ok());
        if geometry.is_ok() && (before.is_none() || before != after) {
            geometry = Err("host project changed during cached geometry refresh".into());
        }
        // 退避档位按**求解结果**更新（见 `enumeration_interval`）。放在缓存事务之前：
        // 它只读结果与代次，不碰文档锁。`after` 是本次尝试结束时的宿主几何代次，
        // 用它判断"宿主工程真的变了"。
        self.note_geometry_outcome(&geometry, after);
        let _transaction = stamp.0.transaction.lock_or_recover();
        let mut prepare_fades = false;
        if !self.is_closed()
            && stamp.0.is_alive()
            && stamp.0.revision.load(Ordering::Acquire) == stamp.1
            && stamp.0.scope_revision.load(Ordering::Acquire) == stamp.2
        {
            let mut cached = self.host_geometry.lock_or_recover();
            let changed = cached.as_ref().is_none_or(|old| {
                old.model != stamp.1 || old.scope != stamp.2 || old.value != geometry
            });
            let delegated = stamp.3.iter().any(|key| {
                stamp
                    .0
                    .regions
                    .lock_or_recover()
                    .get(key)
                    .is_some_and(|region| {
                        region.has_content_based_fade_at_head
                            || region.has_content_based_fade_at_tail
                    })
            });
            if delegated && changed && geometry.is_ok() {
                // 比较**整组**的淡变输入：此前只比四个 fade **长度**，于是改形状 /
                // 曲率 / S 都不推进 `render_epoch` —— 委托端（插件自己渲这条淡变）
                // 改了曲线而声音不变。形状与两根轴同样进指纹。
                let fade_inputs = |value: &[crate::host::geometry::BoundHostGeometry]| {
                    value
                        .iter()
                        .flat_map(|entry| {
                            let g = &entry.geometry;
                            [
                                g.fade_in_sec,
                                g.fade_out_sec,
                                g.auto_fade_in_sec,
                                g.auto_fade_out_sec,
                                g.fade_in_shape,
                                g.fade_out_shape,
                                g.fade_in_dir,
                                g.fade_out_dir,
                                g.fade_in_dir_new,
                                g.fade_out_dir_new,
                                g.fade_in_dir2_new,
                                g.fade_out_dir2_new,
                            ]
                        })
                        .collect::<Vec<_>>()
                };
                prepare_fades = cached
                    .as_ref()
                    .and_then(|old| old.value.as_ref().ok())
                    .is_none_or(|old| fade_inputs(old) != fade_inputs(geometry.as_ref().unwrap()));
                // `prepare_fades` 只记"渲染输入变了"；`render_epoch` 的推进与对全部
                // owner 的通知统一由下面的 `invalidate_render_input()` 完成。
            }
            // 门控是**实例级**的：单 region 时等同于该 item 的 mute（与既有语义一致）；
            // 多 region 时只有全部绑定 item 都静音才静音整组 —— 一个子轨静音不该让
            // 整组哑掉。逐 item 的独立门控需要渲染路径支持，见方案 Task 5.2 的已知边界。
            // 【为什么不给这个读数去抖】"静音 ⇒ 不出声"是一条**基本契约**，
            // 每一次迟滞都会让用户多听一段本该静音的内容。take 切换瞬间的误读是
            // 宿主侧异常，用状态去掩盖它会把"多响了"换成"该静音时还在响"——
            // 后者更难定位。宁可忠实地按宿主读数门控。
            self.host_item_muted.store(
                geometry
                    .as_ref()
                    .is_ok_and(|bound| !bound.is_empty() && bound.iter().all(|e| e.geometry.muted)),
                Ordering::Release,
            );
            if let Ok(bound) = &geometry {
                // 【为什么按**存活 region 集合**回收，而不是按本轮 `bound`】`bound`
                // 是**本实例**分配到的子集：同一文档上的编辑实例与播放实例各有自己
                // 的分配，用其中一个子集做 retain 会把另一个实例的边整批删掉。
                // `regions` 是文档级权威（`destroy_playback_region` 已按 key 删除），
                // 按它回收既清掉陈旧边，又不会误伤兄弟实例。
                let live: std::collections::BTreeSet<u64> =
                    stamp.0.regions.lock_or_recover().keys().copied().collect();
                let mut items = stamp.0.region_items.lock_or_recover();
                items.retain(|key, _| live.contains(key));
                // 【T22：授权记录在这里也有一个写者】此前 `authorized_takes` /
                // `authorized_sources` 只有一个写者（`remap_and_log`），而它要求
                // `region_items` 与 `ui_tracks` **在同一刻都就绪**。重开工程时这个条件
                // 常常不成立（几何缓存被 `clear_renderers` 的 revision 推进作废、
                // `assignments` 还没重发），于是那张表被写成空的 —— 而清单侧把"没有记录"
                // 当成"未授权"，一次性剥掉全部 clip 的源。
                //
                // 【为什么用 `or_insert` 而不是覆盖】这张表的语义是"ARA **授权那一刻**的
                // active take"。每轮清单都覆盖的话，用户在 REAPER 里切换 active take
                // 会被静默接受（GUID 永远相等），`take_switched` 这条真因就再也报不出来。
                // 只在**缺席**时补记 —— 那正是"这一刻才第一次观察到它"的语义。
                //
                // 【锁序】`region_items` → `authorized_takes` → `authorized_sources`，
                // 与 `document.rs` 声明的层级一致。
                let mut takes = stamp.0.authorized_takes.lock_or_recover();
                let mut sources = stamp.0.authorized_sources.lock_or_recover();
                for entry in bound {
                    let key = entry.region_key;
                    let item = entry.geometry.item_id.clone();
                    takes
                        .entry(key)
                        .or_insert_with(|| entry.geometry.take_id.clone());
                    if let Some(name) = entry.geometry.source_file_name.clone() {
                        sources.entry(key).or_insert(name);
                    }
                    items.insert(key, item);
                }
            }
            *cached = Some(CachedHostGeometry {
                model: stamp.1,
                scope: stamp.2,
                change: after,
                value: geometry,
            });
            drop(cached);
            if changed {
                stamp.0.ui_geometry_revision.fetch_add(1, Ordering::AcqRel);
            }
        }
        drop(_transaction);
        if prepare_fades {
            stamp.0.invalidate_render_input();
        }
    }
    /// 本实例是否负责把处理结果作为音频输出（而非透传宿主原声）。
    ///
    /// 判据见 [`ExtensionOwner::is_editor_only`]。**未绑定实例（`roles == 0`）恒为
    /// `false`** —— 这与改动前的 `role == 1` 一致，是 `process_outputs_the_snapshot_*`
    /// 等用例依赖的既有行为。
    pub(crate) fn renders_playback(&self) -> bool {
        self.is_bound() && !self.is_editor_only()
    }
    /// 本实例收到过多少次"重新准备"通知（仅测试构建可见）。
    ///
    /// 【为什么暴露成方法而不是把字段改成 pub(crate)】`prepare_requests` 只在
    /// `#[cfg(test)]` 下存在，字段本身不能带跨模块的 pub 可见性而不连带改变
    /// 非测试构建的形状。给出一个同样只在 test 下编译的读取器，生产构建里
    /// 它随之消失，不会多出一个可见的原子量。
    #[cfg(test)]
    pub(crate) fn prepare_request_count(&self) -> u64 {
        self.prepare_requests.load(Ordering::Acquire)
    }
    /// 等后台准备任务排空（仅测试构建可见）。
    ///
    /// 【为什么需要它】本模块的测试会经 `prepare_renderers` 触发后台任务，而夹具文档
    /// 在用例结束时就被丢弃。给出一个有界的等待，让用例结尾不与仍在运行的任务抢生命周期
    /// （不等待的话会出现偶发的访问违例，且与真正的回归难以区分）。
    #[cfg(test)]
    pub(crate) fn await_preparation_idle(&self, timeout: std::time::Duration) {
        if let Some(worker) = self.preparation() {
            let _ = worker.wait_idle_until(std::time::Instant::now() + timeout);
        }
    }
    /// 当前准备队列（取自锁后的克隆；没有则 `None`）。
    ///
    /// 【为什么返回 `Arc` 而不是 guard】调用方要在队列表上做 `state()` / `request()`
    /// 这类可能短暂阻塞的操作；持着槽位锁做它们会把"换队列"（重绑）拖住。
    fn preparation(&self) -> Option<Arc<super::preparation::PreparationQueue>> {
        self.preparation.lock_or_recover().clone()
    }

    /// 确保存在一条**可用**的准备队列；已被 `close()` 的旧队列在这里被替换。
    ///
    /// 【为什么必须检查 `is_closed()`】`stop_editor()` 关闭队列后，`Arc` 仍可能被
    /// 队列自身的 worker 或 UI 读数短暂持有。只判 `Option::is_none()` 会漏掉
    /// "槽位里躺着一个死队列"这一档，而那正是重绑时最常见的形状。
    fn ensure_preparation(&self) {
        let mut slot = self.preparation.lock_or_recover();
        if slot.as_ref().is_some_and(|queue| !queue.is_closed()) {
            return;
        }
        match super::preparation::PreparationQueue::new() {
            Ok(queue) => *slot = Some(Arc::new(queue)),
            Err(error) => {
                // 【为什么只记日志】绑定本身仍然有效；没有准备队列只会让快照停在
                // "尚未就绪"，那由 `HostAudioState` 的合成中读数暴露，不需要让绑定失败。
                log::warn!("[ara] preparation worker unavailable: {error}");
                *slot = None;
            }
        }
    }

    /// 本实例是否已拿到角色分配（`roles == 0` 表示尚未绑定）。
    fn is_bound(&self) -> bool {
        self.roles.load(Ordering::Acquire) != 0
    }
    /// 本实例是否拿到了宿主分配的 playback 角色。
    pub(crate) fn has_playback_role(&self) -> bool {
        self.roles.load(Ordering::Acquire) & ExtensionRoles::PLAYBACK_RENDERER.bits() != 0
    }
    /// 本实例已拿到角色分配，但其中**没有** playback 角色（等价于旧代码的 `role == 2`）。
    ///
    /// 【为什么与 `is_editor_only()` 分开】后者还要求"文档上另有 playback 实例"，
    /// 因为那才决定**要不要透传**。而"哪个实例挂着用户的窗口"只取决于角色本身 ——
    /// Cubase 的单实例形态下这个实例同时是"唯一的 GUI"和"接管渲染的那个"，
    /// 两个问题在这里答案相同、但判据不同。
    pub(crate) fn is_non_playback(&self) -> bool {
        self.is_bound() && !self.has_playback_role()
    }
    /// 宿主是否把"渲染"这件事交给了**别的**实例。
    ///
    /// 由 [`super::document::DocumentSession::publish_playback_coverage`] 写入；见该函数的说明。
    pub(crate) fn set_playback_covered(&self, covered: bool) {
        self.playback_covered.store(covered, Ordering::Release);
    }
    /// 本实例是否已经有可发布的快照。
    ///
    /// 【为什么提升前要问它】文档级"没有任何实例持 playback 角色"时需要挑一个实例
    /// 接管渲染。挑一个从未 `prepare()` 过的实例只会把静音从"一个实例"搬到"另一个
    /// 实例"；只有已经发布过快照的实例才真的能出声。见
    /// [`super::document::DocumentSession::report_playback_ownership`]。
    pub(crate) fn has_prepared_snapshot(&self) -> bool {
        self.snapshots.iter().any(|publisher| publisher.is_ready())
    }
    /// 处理结果发布器是否对**实际会被读取的那一份**就绪。
    ///
    /// 【为什么不能固定查 `snapshots[0]`】真正被读的是
    /// `snapshots[snapshot_index_for_rate(...)]`：48 kHz 及倍率工程读索引 1。
    /// 探测错发布器会让"实际那份还没就绪"被判成就绪，于是 `layered == false`、
    /// 输出被预清后逐样本写零（用户听到的"有概率完全没声音"）。
    pub(crate) fn processed_snapshot_ready(&self) -> bool {
        let index = self.snapshot_index_for_rate(self.setup_sample_rate());
        self.snapshots[index].is_ready()
    }
    /// 把本实例标记为"接管渲染"（诊断计数 + 日志）。
    pub(crate) fn promote_to_renderer(&self) {
        self.promoted_to_renderer.fetch_add(1, Ordering::Relaxed);
        log::warn!(
            "[ara] writer {} takes over rendering: no instance on this document holds the \
             playback role (roles={:#x})",
            self.writer_id(),
            self.roles.load(Ordering::Acquire)
        );
    }
    /// 文档关闭后重绑到另一份文档的次数（诊断；测试断言用）。
    #[cfg(test)]
    pub(crate) fn rebind_generation_count(&self) -> u64 {
        self.rebind_generation.load(Ordering::Acquire)
    }
    /// 本实例的创建序号（越小越早）；诊断时用来指认实例。
    pub(crate) fn writer_id(&self) -> u64 {
        self.writer_id.load(Ordering::Acquire)
    }
    /// 本实例的角色位集合（诊断用）。
    ///
    /// 【为什么需要它】`bind document` 那一行是 `INFO`，而现场日志可能是 `WARN` 级
    /// （实测 Cubase 会话整段没有一行 INFO）。于是"这个实例到底拿到了什么角色"只在
    /// 报警那条 `WARN` 里才可见 —— 那就得把角色位带上，一条日志说清全部。
    pub(crate) fn roles_bits(&self) -> i32 {
        self.roles.load(Ordering::Acquire)
    }
    /// 记录 `setupProcessing` 声明的采样率（Hz）。非有限/非正数忽略（保持未声明）。
    pub(crate) fn set_setup_sample_rate(&self, rate: f64) {
        let hz = rate.round();
        if hz.is_finite() && hz > 0.0 && hz <= f64::from(i32::MAX) {
            self.setup_sample_rate.store(hz as i32, Ordering::Release);
            // Capture 的段容量也按这个率算（一"段"是 8 **秒**，不是 8 万帧）。
            // 在这里转发而不是让调用点各记一次：宿主声明的工程率只有一个来源，
            // 两处分别记录必然漂移。
            self.capture.note_project_rate(hz);
        }
    }
    /// `setupProcessing` 声明的采样率（Hz；未声明为 `0`）。
    pub(crate) fn setup_sample_rate(&self) -> f64 {
        f64::from(self.setup_sample_rate.load(Ordering::Acquire))
    }
    /// 记录 `setBusArrangements` 协商出的扬声器编排（`kMono` = 1 / `kStereo` = 3），
    /// 打包为 `(输入 << 32) | 输出`。
    pub(crate) fn set_bus_arrangements(&self, inputs: u64, outputs: u64) {
        self.bus_arrangement
            .store((inputs << 32) | (outputs & 0xffff_ffff), Ordering::Release);
    }
    /// 协商出的扬声器编排位掩码（`direction`：`kInput` = 0 / `kOutput` = 1）。
    ///
    /// 声道数 = 掩码置位数（`count_ones`）：kMono=1 ⇒ 1，kStereo=3 ⇒ 2，
    /// 5.1=63 ⇒ 6。尚未协商（`0`）时按默认 `kStereo`(3) 报告 —— 与
    /// `component_get_bus_info` 声明的 2 声道一致，不与宿主协商结果矛盾。
    pub(crate) fn bus_arrangement(&self, direction: i32) -> u64 {
        let packed = self.bus_arrangement.load(Ordering::Acquire);
        if packed == 0 {
            return 3;
        }
        if direction == 0 {
            packed >> 32
        } else {
            packed & 0xffff_ffff
        }
    }
    /// 快照发布者下标：原生 44.1/48k 各一个，其余率取**最合适的已声明原生率**。
    ///
    /// 优先用 `setupProcessing` 的率（权威），未声明时退回 `context_rate`。这里**不**按
    /// 宿主率拒绝：`copy_block` 会把宿主帧线性重采样到该快照的原生率，与
    /// `setupProcessing` 里对非原生率的声明一致。
    ///
    /// 选择次序：先看输出率是不是某个原生率的**整数倍**（整数倍重采样无插值误差，
    /// 如 88.2k→44.1k、96k→48k），再退到最近的原生率。
    pub(crate) fn snapshot_index_for_rate(&self, context_rate: f64) -> usize {
        let declared = self.setup_sample_rate();
        let rate = if declared > 0.0 {
            declared
        } else {
            context_rate
        };
        if !rate.is_finite() || rate <= 0.0 {
            return 0;
        }
        let integral_multiple = |base: f64| {
            let ratio = rate / base;
            ratio >= 1.0 && (ratio - ratio.round()).abs() <= 1e-6
        };
        if integral_multiple(44100.0) {
            0
        } else if integral_multiple(48000.0) {
            1
        } else if (rate - 44100.0).abs() <= (rate - 48000.0).abs() {
            0
        } else {
            1
        }
    }
    /// 最终输出前的有效item静音门；与编辑器内轨道mute不同，不从输入是否全零猜静音。
    pub(crate) fn host_item_muted(&self) -> bool {
        self.host_item_muted.load(Ordering::Acquire)
    }
    /// editor-only必须透传；未绑定角色不在这里虚构为editor，保留原未绑定安全行为。
    ///
    /// 【为什么判据必须是 `role == 2` 而不是"没有 playback 角色"】未绑定的实例
    /// `role == 0`。`role != 1` 会把**未绑定**也判成 editor-only ⇒ `process` 走透传 ⇒
    /// 真正该输出快照的实例什么都不输出（`vst3/audio_boundary_tests` 的
    /// `process_outputs_the_snapshot_...` 等用例正是这个形状）。只有宿主**明确**
    /// 分配了非 playback 角色（`role == 2`）才是 editor-only。
    pub(crate) fn is_editor_only(&self) -> bool {
        self.is_non_playback() && self.playback_covered.load(Ordering::Acquire)
    }
    /// process只写有界原子诊断，JSON/日志全部由actor读取，暂不把候选计数当作根因。
    pub(crate) fn observe_transport(&self, context: &crate::audio_abi::ProcessContext, mode: i32) {
        if let Some(clock) = self.clock.get() {
            clock.observe(
                context,
                self.writer_id.load(Ordering::Relaxed),
                self.roles.load(Ordering::Relaxed),
                mode,
            );
            if mode != 2 {
                clock.update(context);
            }
        }
    }
    /// SDK允许从音频线程调用；记录来源但不做文件IO。
    pub(crate) fn record_processing_stop(&self) {
        if let Some(clock) = self.clock.get() {
            clock.observe_stop(
                self.writer_id.load(Ordering::Relaxed),
                self.roles.load(Ordering::Relaxed),
            );
            clock.stopped();
        }
    }
    /// 真实组件仅作为授权入口，原GUI共享同文档唯一actor/history。
    ///
    /// 【为什么 Capture 走另一支】宿主没给 ARA 时 `editor_document()` 必然失败
    /// （那是 `ARA_NOT_BOUND` 的语义），而 Capture 的编辑界面**必须**能用 ——
    /// 否则"录进来由插件处理"无从发生：用户改不了音高，也就没有处理可言。
    /// 两个分支产出同一个类型，因此命令分发、历史、参数、分析全部共用。
    pub(crate) fn editor_session(
        self: &Arc<Self>,
    ) -> Result<Arc<crate::editor::session::EditorSession>, String> {
        if !self.is_bound() && !self.bound_once.load(Ordering::Acquire) {
            // 【为什么"形态还没落定"也要给会话】形态要等 `setActive` 之后
            // `ARA_BINDING_GRACE`（5 秒）才落定，而**窗口打开的那一刻必然早于它**：
            // 那时拒绝服务就等于"非 ARA 宿主永远打不开界面"—— 探活拿到
            // `ARA_NOT_BOUND`（终态），整页替换成"插件会话不可用"，而且不可恢复。
            // 从未绑定过 ARA 的实例只有一条能服务界面的路（Capture 会话），所以这里
            // 不按形态分支。宿主**真的**绑上 ARA 之后 `is_bound()` 为真，自然走文档那份。
            return self.capture_session();
        }
        // 绑定过又关闭：那才是真的"会话已结束"，必须如实报错而不是悄悄变成 Capture。
        self.editor_document()?.editor_session()
    }

    /// Capture 形态的会话；每个实例唯一（`OnceLock`）。
    ///
    /// 【为什么与 ARA 会话分开持有】ARA 会话按**文档**唯一（文档上多个实例共享一个
    /// actor，这样同文档的视图能看到同一份编辑）。Capture 实例不在任何文档上，
    /// 因此"每个实例一份"才是与 ARA 同构的粒度（见 Capture 方案 §5.1 第 3 条：
    /// 同一文档上两个实例必须能独立工作）。
    pub(crate) fn capture_session(
        self: &Arc<Self>,
    ) -> Result<Arc<crate::editor::session::EditorSession>, String> {
        let session = self
            .capture_session
            .get_or_init(|| crate::editor::session::EditorSession::new_capture(self))
            .clone();
        // 【为什么在这里补投】`setState` 可能早于会话创建（用户从没开过窗口）。
        // 取走即清，幂等；失败只记日志 —— 一份读不出来的存档不该让界面打不开。
        let session = session?;
        if let Some(bytes) = self.pending_capture_edits.lock_or_recover().take() {
            if let Err(error) = session.stash_serialized_capture_edits(&bytes) {
                log::warn!("[capture] restored edits could not be applied: {error}");
            } else {
                session.request_reload();
            }
        }
        Ok(session)
    }

    /// 已经创建过的 Capture 会话；**不会**创建。
    ///
    /// 【为什么 `getState` 必须用这个】存档发生在宿主的主线程上，而创建一个会话会拉起
    /// actor 线程、扫临时目录、初始化模型资源。用户没开过窗口时"没有曲线可存"是正确
    /// 结论，不需要为此付出一次会话的代价。
    pub(crate) fn capture_session_if_initialized(
        &self,
    ) -> Option<Arc<crate::editor::session::EditorSession>> {
        self.capture_session
            .get()
            .and_then(|result| result.as_ref().ok())
            .cloned()
    }

    /// 已经存在的编辑器会话（Capture 或 ARA），**不创建**。
    pub(crate) fn existing_editor_session(
        &self,
    ) -> Option<Arc<crate::editor::session::EditorSession>> {
        if let Some(session) = self.capture_session_if_initialized() {
            return Some(session);
        }
        self.document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .and_then(|document| document.existing_editor_session())
    }

    /// 让 Capture 编辑器会话重新加载时间线并重渲染（录制结束 / 切换 take / 清空之后）。
    ///
    /// 【为什么不在命令里直接做】`capture_*` 命令在视图层（UI 线程）执行，而加载要读
    /// 捕获存储、物化分析副本 —— 那是文件 I/O。这里只置一个标志，由会话的 actor 线程
    /// 在下一轮自己加载（加载成功会顺带排一次渲染）。
    pub(crate) fn request_capture_reload(self: &Arc<Self>) {
        // 原始磁带要**立刻**跟着 take 走，不能等合成：用户按下停止之后马上就想听到东西，
        // 而一次神经合成要数秒。这一份也是"换 take 之后绝不播上一份"的保证
        // （见 `raw_tape_ready` 的 take 比对）。
        self.sync_raw_tape();
        // 换 take / 清空 / 录制结束都会走到这里：内容换了，本 pass 锁存的内容源不再有效。
        self.clear_rung_latch();
        if let Ok(session) = self.capture_session() {
            session.request_reload();
        }
    }

    /// 把活动 take 的原始捕获发布成一份**未处理**的快照（见 [`Self::raw_tape`]）。
    ///
    /// 【为什么幂等】它由命令（UI 线程）与合成准备（actor）两处调用；take 号没变时
    /// 直接返回，不重复拷贝几十 MB 的 PCM。
    pub(crate) fn sync_raw_tape(&self) {
        if !self.capture.has_material() {
            // 没有素材（清空 / 还没录 / 全是静音）：撤掉指针，让仲裁落到透传。
            self.raw_tape.clear();
            self.raw_tape_revision.store(0, Ordering::Release);
            return;
        }
        let revision = self.capture.revision();
        if self.raw_tape_revision.load(Ordering::Acquire) == revision {
            return;
        }
        let Some(snapshot) = self.capture.raw_tape() else {
            return;
        };
        self.raw_tape.collect_retired();
        if !self.raw_tape.has_capacity(&snapshot) {
            log::warn!(
                "[capture] raw tape does not fit the snapshot budget; keeping the previous one"
            );
            return;
        }
        match self.raw_tape.publish(snapshot) {
            Ok(()) => self.raw_tape_revision.store(revision, Ordering::Release),
            Err(error) => log::warn!("[capture] raw tape publish failed: {error:?}"),
        }
    }
    /// GUI能力标记只回答是否有官方写API；实际每个clip仍须唯一真实take绑定。
    /// 宿主淡化轴的版本语义；`None` 表示版本读不出来（前端应保持只读）。
    ///
    /// 取**任一** owner 的读数：`fade_axes_v2` 来自 `GetAppVersion`，同一进程里
    /// 所有实例必然一致，不存在"这个 owner 是新轴、那个是旧轴"的情况。
    pub(crate) fn host_fade_axes(&self) -> Option<bool> {
        self.editor_document().ok().and_then(|document| {
            document
                .renderer_owners()
                .iter()
                .find_map(|owner| owner.native_host().and_then(|host| host.fade_axes_v2()))
        })
    }

    /// 只返回同文档活入口的拥有引用接口；其project()仍钉在初始化时的真实所属工程。
    pub(crate) fn project_history_host(&self) -> Option<Arc<crate::host::NativeHost>> {
        let document = self.editor_document().ok()?;
        if let Some(host) = self.native_host().filter(|host| host.has_project_history()) {
            return Some(host);
        }
        document.renderer_owners().iter().find_map(|owner| {
            owner
                .native_host()
                .filter(|host| host.has_project_history())
        })
    }
    /// 当前编辑后端：宿主给了原生扩展就用它，否则退回 ARA-only。
    ///
    /// 【为什么是"一等后端"而不是 `None`】ARA-only 后端照样能看 / 能听 / 能渲染 /
    /// 能调参；它的退化只体现在"片段几何只读"。把这条结论收敛成一个后端值，前端与
    /// 诊断包就能直接读到"我是哪个后端、我能做什么"，而不是从一堆 `Option` 反推。
    /// Capture 通路的捕获存储。
    pub(crate) fn capture(&self) -> &crate::capture::CaptureStore {
        &self.capture
    }

    /// 本实例是否**应当**发布快照。
    ///
    /// 【为什么不能直接改 `renders_playback()`】那个读数的语义是"本文档的播放由本实例
    /// 渲染"，它被 `report_playback_ownership` 与诊断用于回答"文档上还有没有人在渲染"。
    /// Capture 实例**不在任何 ARA 文档上**，把它算进去会让那两处以为文档有播放实例。
    /// 这里另立一个"要不要出音"的判据，只在渲染准入点使用。
    pub(crate) fn renders(&self) -> bool {
        if self.is_bound() {
            return !self.is_editor_only();
        }
        // 【为什么"有素材"本身就是充分证据】素材只有一条产生路径：`capture_start`
        // （ARA 绑定时被显式拒绝，见 `capture/commands.rs`）或组件 state 恢复（绑定的
        // 实例走上面那一支）。因此"未绑定 + 有素材"只可能是 Capture。
        //
        // 【为什么不能只问 `mode()`】形态判定此前完全依赖编辑器窗口的 20ms 定时器
        // （`note_editor_tick`）与 `setActive` 之后 5 秒的宽限期。宿主把插件当普通插入
        // 用时用户很可能**根本没开窗口**，于是 `mode()` 永远是 `Undetermined` ⇒ 这里
        // 恒假 ⇒ 一条快照都不发布 ⇒ 录制正常、界面正常、**播放时全静音**。
        // 素材是比时钟更早、更硬的证据：它已经存在，就说明这个实例在录宿主。
        self.capture.has_material()
    }

    /// Capture 形态的渲染准备（不经过 ARA 文档）。
    ///
    /// 【为什么必须有一份独立的实现】`prepare_job` 的第一件事就是取 ARA 文档，
    /// 取不到直接返回错误。Capture 实例**永远**没有文档，所以那条路对它等于
    /// "永远不准备 ⇒ 永远静音"。
    ///
    /// 【为什么复用 `RenderInput` 而不是另写一条】Capture 的投影产出的正是
    /// `(TimelineState, Vec<AraPlaybackRegion>, sources)` —— 与 ARA 路径同一组类型
    /// （见 `capture::CaptureStore::render_input`）。于是校验、渲染、压缩、双率发布
    /// 全部是同一份代码，Capture 不需要第二条渲染通路。
    /// Capture 形态的渲染输入，以及它的版本 `(take, edit)`。
    ///
    /// 【为什么只有 owner 够得着两边】几何与 PCM 在捕获存储上，用户曲线在编辑器会话上。
    /// **播放与导出共用这一份** —— 于是"导出的音频与听到的不一样"这一类漂移不可能发生。
    ///
    /// 【为什么优先用会话的那份时间线】它带着用户的参数曲线，而曲线就是 Capture 形态下
    /// "处理"的全部内容。用捕获存储那份（只有几何）会让"改了音高却不重算"变成一个静默的
    /// 失效：界面变了，声音还是旧的。
    /// 返回 `(素材代次, 编辑代次, 这份时间线带不带用户曲线, 渲染输入)`。
    ///
    /// 【为什么"带不带曲线"要作为一项返回】会话还没加载时拿到的是**不含用户曲线**的
    /// 几何，而那时 `edit` 恒为 0 —— 若调用方把它当成"已准备"，会话加载之后版本不变、
    /// 这次渲染被跳过，用户画完线永远听不到变化（见 `capture_prepare_job` 的版本闸）。
    pub(crate) fn capture_render_input(
        &self,
    ) -> Result<(u64, u64, bool, super::input::RenderInput), String> {
        if !self.capture.has_material() {
            return Err("no Capture material".into());
        }
        let revision = self.capture.revision();
        let session = self
            .capture_session
            .get()
            .and_then(|session| session.as_ref().ok())
            .filter(|session| session.is_loaded())
            .cloned();
        let (mut timeline, edit, with_curves) = match &session {
            // 【为什么用"发布代次"而不是"编辑代次"】渲染读的就是这份被发布的时间线，
            // 而 `generation` 只由 `mark_dirty` 推进 —— 有命令改了曲线却不标脏
            // （`convert_mix_param`），只认 `generation` 会让那种改动不推进版本 ⇒
            // 渲染被跳过 ⇒ 曲线永远不生效。
            Some(session) => (
                session.render_timeline()?,
                session.timeline_revision(),
                true,
            ),
            // 会话还没加载（窗口还没开、或前端还没问过时间线）：退回捕获存储的几何。
            // 那时也没有用户在编辑，两者等价 —— 但**必须把这个事实带进版本号**
            // （见 `capture_prepare_job` 的 `with_curves` 位）。
            None => (self.capture.session_timeline().0, 0, false),
        };
        let trimmed = crate::capture::timeline::clamp_timeline_to_project(&mut timeline);
        if trimmed > 0 {
            log::warn!(
                "[capture] removed pre-roll from {trimmed} clip(s) before rendering; the material domain starts at project time 0"
            );
        }
        let input = super::input::RenderInput {
            regions: self.capture.regions_for_timeline(&timeline),
            sources: self.capture.material_sources(),
            timeline: Some(timeline),
            project_sample_rate: Some(self.setup_sample_rate()).filter(|rate| *rate > 0.0),
            // 【为什么参数曲线这里为空】Capture 的曲线就在时间线里（它**就是**参数权威），
            // 于是 `mix_interleaved` 走"直接按时间线合成"那一支。ARA 需要图谱投影是因为
            // 它的时间线每次都被宿主重建、clip 身份由 region key 派生 —— Capture 没有
            // 那个问题（见 `render/input.rs::mix_interleaved` 的第一支）。
            clip_parameters: std::collections::BTreeMap::new(),
        };
        Ok((revision, edit, with_curves, input))
    }

    pub(crate) fn capture_prepare_job(
        &self,
        cancel: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<(), String> {
        if !self.capture.has_material() {
            return Ok(());
        }
        // 【为什么这里也要补一次】命令那条路（`request_capture_reload`）只在用户动作时
        // 发生；从组件 state 恢复 take 的会话不走那里。这一份是兜底，且幂等。
        self.sync_raw_tape();
        let (revision, edit, with_curves, input) = self.capture_render_input()?;
        // 【T-S1-6】发布前把"素材里有多少个洞"变成可读的数（去重打印）。
        self.note_tape_coverage(&input);
        // 【为什么要版本闸】它被 UI 定时器与命令两条路径驱动；没有这道闸，一次 8 秒的
        // 合成会被排队几百次。版本里必须**带上编辑代次**：只认 take 会让参数改动
        // 永远不重算。
        //
        // 【为什么还要带上"这份时间线有没有曲线"】会话没加载时拿到的是**不含用户曲线**
        // 的几何（`edit` 恒为 0）。若把它算作"已准备"，会话加载之后版本不变 ⇒ 这次
        // 渲染被跳过 ⇒ 用户画完线永远听不到变化。把这个事实编进版本号，会话一加载
        // 版本就变，于是恰好补一次渲染，而没加载时也不会反复重算。
        let version = revision ^ (edit << 32) ^ if with_curves { 0 } else { 1 << 63 };
        if self.capture_prepared.load(Ordering::Acquire) == version
            && self.snapshots.iter().all(|snapshot| snapshot.is_ready())
        {
            return Ok(());
        }
        if input
            .timeline
            .as_ref()
            .is_none_or(|timeline| timeline.clips.is_empty())
        {
            return Ok(());
        }
        for publisher in &self.snapshots {
            publisher.collect_retired();
        }
        // 【为什么要量这一步】"起播 0.4–0.7 s 一次咔"的两个候选成因（起播窗口 vs
        // 处理结果晚到）在这一个数上就能分开：若 `render_ms` 与用户听到的时间同量级，
        // 那一段就是"素材在、结果还没算出来"；若它远大于 0.7 s，那 0.4 s 只能来自
        // 起播窗口。没有这个数时，两者在日志里完全同形。
        let render_started = std::time::Instant::now();
        let mut snapshots = input.render(cancel)?;
        super::snapshot::compact_prepared(&mut snapshots)?;
        let render_ms = render_started.elapsed().as_millis() as u64;
        if self.is_closed() {
            return Ok(());
        }
        if !self
            .snapshots
            .iter()
            .zip(&snapshots)
            .all(|(publisher, snapshot)| publisher.has_capacity(snapshot))
        {
            return Err("retired snapshot budget exhausted".into());
        }
        for (publisher, snapshot) in self.snapshots.iter().zip(snapshots) {
            publisher
                .publish(snapshot)
                .map_err(|e| format!("snapshot publish failed: {e:?}"))?;
        }
        self.capture_prepared.store(version, Ordering::Release);
        // 处理结果与当前 take 的对应关系：音频线程据此在"处理结果"与"原始磁带"之间选
        // （见 `processed_ready`）。换 take 之后这里才更新，所以中间那段空窗由原始磁带兜底。
        self.capture_ready_revision
            .store(revision, Ordering::Release);
        log::warn!("[capture] prepared material revision {revision} edit {edit} in {render_ms} ms");
        Ok(())
    }

    /// 把这一份渲染输入里"我们没录到"的帧数记成读数（诊断线程读）。
    ///
    /// 【为什么不打印逐源的明细】一条 take 通常只有一份源；有洞时真正想知道的是
    /// "有几个洞、一共多少帧"。源数为 0（全是 ARA 通路那种"整份都算"）时这一行
    /// 不会出现 —— 那正是"没有洞可谈"的正确表达。
    fn note_tape_coverage(&self, input: &super::input::RenderInput) {
        let mut holes = 0_u64;
        let mut hole_frames = 0_u64;
        let mut covered_sources = 0_u64;
        for pcm in input.sources.values() {
            if pcm.covered.is_empty() {
                continue;
            }
            covered_sources += 1;
            let total = pcm.planes.first().map_or(0, Vec::len) as i64;
            let covered: i64 = pcm
                .covered
                .iter()
                .map(|range| (range.end - range.start) as i64)
                .sum();
            let gap = total.saturating_sub(covered);
            if gap <= 0 {
                continue;
            }
            hole_frames = hole_frames.saturating_add(gap as u64);
            // 首/尾之外的每一段都算一个洞；段与段之间的空档也算一个。
            holes = holes.saturating_add((pcm.covered.len() as u64).saturating_sub(1));
            if pcm.covered.first().is_some_and(|range| range.start > 0) {
                holes = holes.saturating_add(1);
            }
            if pcm
                .covered
                .last()
                .is_some_and(|range| (range.end as i64) < total)
            {
                holes = holes.saturating_add(1);
            }
        }
        self.tape_holes.store(holes, Ordering::Relaxed);
        self.tape_hole_frames.store(hole_frames, Ordering::Relaxed);
        if covered_sources == 0 {
            return;
        }
        let line = format!(
            "[capture] tape coverage: sources={covered_sources} holes={holes} \
             hole_frames={hole_frames}"
        );
        let mut last = self.last_tape_diag.lock_or_recover();
        if last.as_deref() != Some(line.as_str()) {
            *last = Some(line.clone());
            log::info!("{line}");
        }
    }

    /// 本块在 Capture 通路上该怎么输出。**无锁、无分配**，可从音频线程调用。
    ///
    /// 【为什么要有这个枚举】"没素材时会听到什么"此前要读者把 `capture_stage` 的早退
    /// 与 `vst3` 的后续分支拼起来才能回答。收敛成一个值之后，每个分支都能被单独断言，
    /// 而"静音"只剩下"确实没有素材"这一个含义。
    pub(crate) fn capture_output(&self) -> CaptureOutput {
        self.capture_output_with(None)
    }

    /// 与 [`Self::capture_output`] 相同，但使用**本块**的流/播放证据。
    pub(crate) fn capture_output_with(&self, evidence: Option<BlockEvidence>) -> CaptureOutput {
        // 【为什么 ARA 实例完全不走这里】它的素材由 ARA 逐个 region 授权，"区间外静音"
        // 是那条通路的既有语义（宿主自己的原声已被 ARA 取代）。这一条必须排在武装之前：
        // 形态一旦是 ARA，任何 Capture 结论都不适用。
        //
        // 【已知边界】`bound_once` 为真而文档已关的实例（宿主绑定过 ARA、文档后来又关掉）
        // 在这里仍然判成"不介入"。那种实例在界面上本来就是终态（`editor_session` 会以
        // DOCUMENT_CLOSED 收场，整页显示"会话不可用"），所以它不是"有界面却不出声"的
        // 现场；要收掉这条边界得先给"文档是否还活着"一个**无锁**读数（音频线程不能取
        // `document` 那把锁），那是独立一期的事。
        if self.is_bound() || self.bound_once.load(Ordering::Acquire) {
            return CaptureOutput::Untouched;
        }
        /*
         * 【为什么"武装"不再直接等于"录制中"】武装之后、宿主开始播放之前是**待命**：
         * 那段时间里没有任何素材在产生，所以"用旧内容盖住新演奏"这个风险根本不存在
         * （那正是录制中要直通的理由）。而此前 `is_armed()` 直接返回 `Recording` ⇒
         * 待命期输出直通宿主原声，用户画完参数线听不到任何变化 —— 表现就是
         * **"参数线不生效"**，而界面只说"等待播放"，两件事对不上。
         *
         * 一旦宿主开始播放就立刻回到 `Recording`：捕获读的是输入总线，而输出可能写进
         * 同一块缓冲（in-place 宿主），直通才不会让捕获录到我们自己的输出。
         * `capture_observe` 在 `capture_stage` 之前更新 `host_playing`，所以起播的
         * 第一块就已经是 `Recording`，捕获读到的是宿主原声。
         */
        // 【为什么把块流证据也算进"录制中"】`kPlaying` 可能在起播第一块才滞后置位；
        // 只按 `capture_started || host_playing` 判断时，那一块会被当成"待命"，
        // 输出上一版 `Processed` —— 旧内容盖在新演奏上。位置流在推进时按录制中直通，
        // 最多在几何变化的瞬间多直通一块；重复块随即回到 `Processed`，待命语义不变。
        // 没有本块证据的调用（旧测试/旧入口）退回实例级实时流读数；生产路径
        // 永远由 `capture_observe` 提供证据，因此不会读到另一条流的结论。
        let recording = if evidence.is_some() {
            self.capture.is_recording_with_evidence(evidence)
        } else {
            self.capture
                .is_recording_with_flow(self.capture_transport_running())
        };
        if recording {
            return CaptureOutput::Recording;
        }
        if !self.capture.has_material() {
            // 【为什么没有素材就一定透传】"不要覆盖一个正常渲染的 ARA 实例的输出"
            // 这条契约由上面的 `is_bound()` 守卫负责：有活的文档时这里根本走不到。
            // 此前这里还额外要求"形态已落定为 Capture"，于是窗口没开过、宽限期未过的
            // 会话（`Undetermined`）在**任何素材之前**都是静音 —— 用户把插件插到一条
            // 有声音的轨道上，得到的是"整条轨哑了"。
            // 插件的诚实默认是**透明**：直到它真的录到了东西，才对声音负责。
            return CaptureOutput::PassThrough;
        }
        let candidate = if self.processed_ready() {
            CaptureOutput::Processed
        } else if self.raw_tape_ready() {
            // 【为什么在这里置位】有素材、但处理结果不是当前代次 ⇒ 该有一次合成在跑。
            // 音频线程只记一个原子；调度仍由会话 actor 完成（见 `render_wanted`）。
            self.render_wanted.store(true, Ordering::Release);
            CaptureOutput::Raw
        } else {
            // 素材存在但两份都还没发布（刚录完、后台还没跑完）：透传宿主原声，但**如实**
            // 报成"等待合成"，而不是与"没有素材"共用同一个名字。
            self.render_wanted.store(true, Ordering::Release);
            CaptureOutput::PendingSynthesis
        };
        self.latch_rung(candidate, evidence)
    }

    /// 撤掉本 pass 锁存的内容源（换 take / 清空之后）。
    ///
    /// 【为什么必须在换 take 时清】锁存的意义是"这一遍只播一个源"，而"这一遍"
    /// 的前提是内容没变。换了 take 之后，上一遍锁存的 `Raw`/`Processed` 指向的是
    /// **另一份素材** —— 继续沿用等于用旧录音盖住新演奏。
    pub(crate) fn clear_rung_latch(&self) {
        self.rung_locked.store(0, Ordering::Release);
    }

    /// 把本块的候选档位**锁存到本 pass**（I-2：一次播放只有一个内容源）。
    ///
    /// # 为什么必须锁存
    ///
    /// 旧实现每块都重新仲裁，于是 `Raw → Processed` 可以发生在**音乐中途** ——
    /// 而那恰好出现在"刚录完、后台合成刚好跑完"的那一刻，也就是用户起播后的
    /// 0.4–0.7 s。一次换源就是一次咔，而它不是用户做错了什么。
    ///
    /// 锁存之后，一次 pass 只有一个内容源；换源只可能发生在 pass 边界，而那时
    /// 本来就要重新开始（第一块走 `Seam::FirstBlock` / `Seam::FadeIn`）。
    ///
    /// # 代价与它为什么值得
    ///
    /// 若某一遍恰好在"处理结果还没算出来"时起播，那一遍会全程播 `Raw`（原始捕获，
    /// **仍然是这一份 take 的内容**，只是没处理）。用户听到的是"这一遍没生效"，
    /// 而不是"听到一半突然换了"。前者可解释、可重试；后者不可解释。
    /// 真正的解法是让处理结果在播放头之前就绪（见方案 §4.2.8 的水位渲染），
    /// 那一步落地之后这个代价会消失。
    ///
    /// # 哪些档位不锁存
    ///
    /// * `Untouched`：绑定的 ARA 实例，根本不走这条通路（上面已早退）。
    /// * `Recording`：录制中必须直通输入，而且它本身意味着一次新的传输 ——
    ///   顺便把锁存清掉，让录完之后的那一遍重新选。
    fn latch_rung(&self, candidate: CaptureOutput, evidence: Option<BlockEvidence>) -> CaptureOutput {
        // 没有本块证据（离线 / 预取 / 旧测试入口）时不锁存：它们的输出稍后才被播放，
        // "此刻的传输状态"对它们不适用。
        let Some(evidence) = evidence else {
            return candidate;
        };
        if evidence.stream != ProcessStream::Realtime {
            return candidate;
        }
        if candidate == CaptureOutput::Recording {
            self.rung_locked.store(0, Ordering::Release);
            return candidate;
        }
        let pass = self.streams[ProcessStream::Realtime.index()]
            .pass
            .load(Ordering::Acquire);
        let locked = self.rung_locked.load(Ordering::Acquire);
        if locked != 0 && self.rung_pass.load(Ordering::Acquire) == pass {
            // 本 pass 已经选过：沿用，不重新仲裁。
            return CaptureOutput::from_code(locked).unwrap_or(candidate);
        }
        self.rung_locked.store(candidate.code(), Ordering::Release);
        self.rung_pass.store(pass, Ordering::Release);
        candidate
    }

    /// 输出源 token：种类 + 快照代次。任一项变化都意味着"用户听到的信号换了源"。
    fn output_token_for(&self, output: CaptureOutput) -> u64 {
        let kind = u64::from(output.code());
        let revision = match output {
            // 处理结果换版（曲线编辑 / 换 take）也要去咔。
            //
            // 【为什么 `Untouched` 也要编进代次】bound 实例的输出仲裁恒为 `Untouched`
            // （它走既有快照通路），而 `capture_stage` 存下的 token 只含"种类"。
            // 于是快照第一次发布（此前逐样本写零）或换版时 token 不变 ⇒
            // `apply_output_declick` 认为"没有切换"⇒ 用户听到一次从零到有声的硬切。
            // 把两份发布器的代次编进来，ARA 的接缝也能拿到同样的 1.3 ms 淡化。
            //
            // 【为什么两份都编】真正被读的是 `snapshots[snapshot_index_for_rate(...)]`
            // （48 kHz 及倍率工程读索引 1）。只编索引 0 会让"实际读的是另一份"的宿主
            // 在换版时拿不到去咔。
            // 【为什么用 `content` 而不是 `generation`】token 的用途只有一个：
            // 让 `apply_output_declick` 判断"这一块与上一块是不是同一份输出"。
            // `generation` 回答的是"发布过几次"，于是同一份素材被重新合成一次
            // （曲线没变、只是版本号动了）就会让 token 变、装填一次 64 帧接缝 ——
            // 实测每 0.68 s 一次，正是用户听到的那个周期性"咔"。
            //
            // 【为什么仍然保留 `kind`】种类（Raw / Processed / Passthrough）回答的是
            // "这一侧是什么内容"，与内容指纹正交：两者都必须参与，缺一个就会把
            // 一次真实换源或一次重新合成判反。
            CaptureOutput::Processed | CaptureOutput::Untouched => {
                self.snapshots[0].content() ^ self.snapshots[1].content().rotate_left(32)
            }
            CaptureOutput::Raw => self.raw_tape.content(),
            _ => 0,
        };
        kind | (revision << 8)
    }

    /// 【B0】记下本块分层拷贝的覆盖读数（音频线程）。
    pub(crate) fn note_layered_probe(
        &self,
        stream: ProcessStream,
        probe: crate::render::snapshot::LayeredProbe,
    ) {
        let Some(cell) = self.layered_probe.get(stream.index()) else {
            return;
        };
        cell.uncovered.store(probe.uncovered as u32, Ordering::Relaxed);
        cell.input_peak.store(
            probe.input_peak.unwrap_or(f32::NAN).to_bits(),
            Ordering::Relaxed,
        );
        cell.edge.store(probe.edge, Ordering::Relaxed);
        cell.valid.store(true, Ordering::Release);
    }

    /// 【B0】取走本块的覆盖读数（音频线程；取走即失效，避免下一块读到陈旧值）。
    fn take_layered_probe(&self, stream: ProcessStream) -> Option<(u32, f32, bool)> {
        let cell = self.layered_probe.get(stream.index())?;
        if !cell.valid.swap(false, Ordering::AcqRel) {
            return None;
        }
        Some((
            cell.uncovered.load(Ordering::Relaxed),
            f32::from_bits(cell.input_peak.load(Ordering::Relaxed)),
            cell.edge.load(Ordering::Relaxed),
        ))
    }

    /// 【B0】量一块输出的形状并记进账本。**音频线程**：只读缓冲、只写原子量。
    ///
    /// 【为什么量 `max_step` 而不是别的】"咔"在波形上就是相邻样本的一次大跳变。
    /// 峰值、RMS、过零率都看不见它 —— 一个 20 ms 的音乐片段可以峰值不变而中间
    /// 跳一下。相邻差分是唯一直接对应听感的读数。
    fn record_click_sample(
        &self,
        stream: ProcessStream,
        kind: u8,
        seam: crate::render::click::SeamCode,
        first_of_stream: bool,
        output: &crate::audio_abi::AudioBusBuffers,
        frames: usize,
    ) {
        let channels = (output.num_channels as usize).min(2);
        let mut max_step = 0.0_f32;
        let mut peak = 0.0_f32;
        let mut first = 0.0_f32;
        let mut last = 0.0_f32;
        let mut energy = 0.0_f64;
        let mut energy_frames = 0_u32;
        for channel in 0..channels {
            // SAFETY: 调用方保证总线有 `channels` 个可读平面，且至少 `frames` 帧。
            let plane = unsafe { *output.channel_buffers.add(channel) };
            if plane.is_null() {
                continue;
            }
            let mut previous = 0.0_f32;
            for index in 0..frames {
                // SAFETY: `index < frames`。
                let value = unsafe { *plane.add(index) };
                if !value.is_finite() {
                    continue;
                }
                peak = peak.max(value.abs());
                max_step = max_step.max((value - previous).abs());
                energy += f64::from(value) * f64::from(value);
                energy_frames += 1;
                if channel == 0 {
                    if index == 0 {
                        first = value;
                    }
                    last = value;
                }
                previous = value;
            }
        }
        let (uncovered, input_peak, edge) = self
            .take_layered_probe(stream)
            .unwrap_or((0, f32::NAN, false));
        // 【覆盖边也是一种接缝】没有源切换、但块内有"宿主原声 ↔ 处理结果"的翻转点：
        // 那一下同样是用户听到的一次信号更换。把它标进账本，round 23 那种"假洞"
        // 就不必再靠用户复述才能被发现。
        let seam = if seam == crate::render::click::SeamCode::None && edge {
            crate::render::click::SeamCode::CoverageEdge
        } else {
            seam
        };
        let rms = if energy_frames == 0 {
            0.0
        } else {
            (energy / f64::from(energy_frames)).sqrt() as f32
        };
        self.click.record(crate::render::click::BlockSample {
            stream: stream.index() as u8,
            kind,
            seam,
            fade_len: self.declick[stream.index()]
                .fade_len
                .load(Ordering::Acquire),
            frames: frames as u32,
            first,
            last,
            max_step,
            peak,
            input_peak,
            uncovered,
            rms,
            edge,
            first_of_stream,
        });
    }

    /// 【B0】把尚未报告过的可疑块打印成 `[click]` 行。**非实时线程**（UI 定时器）。
    ///
    /// 【为什么由 UI 定时器驱动】它是唯一稳定存在、且不在音频线程上的周期点；
    /// 音频线程只负责记录。报告本身是"扫 256 个原子 + 格式化"，与一次界面刷新同量级。
    pub(crate) fn log_click_report(&self) {
        for line in self.click.drain_suspicious() {
            log::warn!("[click] {line}");
        }
    }

    /// 把**最近一块**的宿主上下文打成一行（非实时线程；由会话 actor 驱动）。
    ///
    /// 【为什么必须有这一行】排查"起播窗口/位置钉住"时，唯一的硬事实是**宿主到底
    /// 给了什么**：`state` 的哪一位、`projectTimeSamples` 动不动、
    /// `continousTimeSamples` 有没有值、`projectTimeMusic` 有没有值。此前这些字段
    /// 一个都没有被打出来，于是每一轮都只能靠"用户再听一遍"来猜是哪一种形状。
    ///
    /// 【为什么读环形记录而不是新开一组原子】环里已经逐块记下了全部字段（音频线程
    /// 只写原子、不分配），复用它就不必再维护第二份会漂移的读数。
    pub(crate) fn log_capture_context(&self) {
        let Some(ring) = self.block_trace.get() else {
            return;
        };
        let Some(line) = ring.last_context_line() else {
            return;
        };
        let mut last = self.last_ctx_diag.lock_or_recover();
        if last.as_deref() != Some(line.as_str()) {
            *last = Some(line.clone());
            log::warn!("{line}");
        }
    }

    /// 【B0】把 I-2 / I-3 的读数打成两行。**非实时线程**（UI 定时器）。
    ///
    /// 【为什么要这两行】本轮之前，排查"咔"只能靠"用户再听一遍"：插件里没有
    /// "这一块换了源"与"这一块位置回退了"的读数，于是同一个症状被反复归因到
    /// 不同机理。这两行把它变成可判定的数字：
    ///
    ///   * `[capture] pass` —— 一次播放里换了几次内容源（I-2，理想为 0）；
    ///
    ///   * `[capture] clock` —— 位置是否逐块续接（I-3，`rewinds` 理想为 0）。
    ///
    /// 【为什么去重】它们由 20 ms 的界面定时器驱动；不去重就是每秒 50 行。判据取
    /// "内容变没变"：所有计数都进了指纹，变了才打印。
    pub(crate) fn log_capture_report(&self) {
        if self.is_bound() || self.bound_once.load(Ordering::Acquire) {
            // ARA 实例不走 Capture 仲裁：这两行的数字对它是无意义的零。
            return;
        }
        if !self.capture.has_material() {
            return;
        }
        let switches = self.rung_switches.load(Ordering::Acquire);
        let from = self.rung_last_from.load(Ordering::Acquire);
        let to = self.rung_last_to.load(Ordering::Acquire);
        let pass_switches = self.pass_switches.load(Ordering::Acquire);
        let output = self.capture_output();
        let prepared = self.capture_prepared.load(Ordering::Acquire);
        let ready = self.capture_ready_revision.load(Ordering::Acquire);
        let line = format!(
            "[capture] pass: take={} prepared={prepared} ready_revision={ready} \
             output={} switches={switches} pass_switches={pass_switches} last={}->{}",
            self.capture.revision(),
            output.as_str(),
            CaptureOutput::from_code(from as u8).map_or("none", |value| value.as_str()),
            CaptureOutput::from_code(to as u8).map_or("none", |value| value.as_str()),
        );
        {
            let mut last = self.last_pass_diag.lock_or_recover();
            if last.as_deref() != Some(line.as_str()) {
                *last = Some(line.clone());
                log::warn!("{line}");
            }
        }
        let index = self.snapshot_index_for_rate(self.setup_sample_rate());
        let geometry = &self.streams[index.min(2)];
        let blocks = geometry.blocks.load(Ordering::Acquire);
        if blocks == 0 {
            return;
        }
        // 【这一行的判读方式】稳态下 `delta_min == delta_max == 块长` 且
        // `rewinds == 0`。任何别的形状都说明位置记账漏了一项 —— 而那就是咔。
        //
        // `counted=` 是"位置由计数器推进的块数"，`jumps=` 是"宿主报的位置与计数器
        // 不同"的次数。两者都大而 `rewinds` 为 0 是**健康**的：说明宿主确实钉住过
        // 位置，而每次它恢复报告时我们都跟上了。
        let line = format!(
            "[capture] clock: stream={} blocks={blocks} pass={} counted={} frames={} \
             jumps={} last_jump={} rewinds={} delta_min={} delta_max={} resyncs={} \
             rising={} continuous={} music={}",
            index.min(2),
            geometry.pass.load(Ordering::Acquire),
            geometry.counted_blocks.load(Ordering::Acquire),
            geometry.counted_frames.load(Ordering::Acquire),
            geometry.jumps.load(Ordering::Acquire),
            geometry.last_jump.load(Ordering::Acquire),
            geometry.clock_rewinds.load(Ordering::Acquire),
            geometry.clock_delta_min.load(Ordering::Acquire),
            geometry.clock_delta_max.load(Ordering::Acquire),
            geometry.position_resyncs.load(Ordering::Acquire),
            geometry.rising_edges.load(Ordering::Acquire),
            geometry.continuous_used.load(Ordering::Acquire),
            geometry.music_used.load(Ordering::Acquire),
        );
        let mut last = self.last_clock_diag.lock_or_recover();
        if last.as_deref() != Some(line.as_str()) {
            *last = Some(line.clone());
            log::warn!("{line}");
        }
    }

    /// 记录本块输出源，并做接缝去咔。
    ///
    /// 【为什么在写完输出之后调用】先用新源写满整块，再把前 `OUTPUT_DECLICK_FRAMES`
    /// 帧与"上一块输出的尾巴"交叉淡化；尾巴就是用户刚刚听到的旧源信号，与旧源连续，
    /// 不需要再持有旧快照。所有状态都是原子 + 预分配缓冲，音频线程不分配、不取锁。
    ///
    /// # Safety
    /// `output` 必须是本回调内有效的 VST3 输出总线；`frames` 不超过其平面长度。
    /// 从一个源切到另一个源时，是否应该**淡入**而不是交叉淡化。
    ///
    /// 【为什么看两侧而不只看上一侧】透传源（宿主原声）与处理结果**不同源**：前者是
    /// 宿主正在播的东西，后者是插件算出来的东西。两者交叉淡化叠出的是"两声并发"，
    /// 不是一次接缝 —— 那种场合必须淡入。但**两个处理结果之间**（Raw ↔ Processed）
    /// 同样是不同源：Raw 是原始捕获，Processed 是神经声码器的输出，内容毫不相干。
    /// 只看上一侧会把 `Processed → Raw` 判成"同源切换"，于是 64 帧里把两段不相干的
    /// 信号线性叠在一起 —— 用户听到的随机"咔"正是它。
    ///
    /// 因此判据是**异源**：只要两侧有一个是透传类，或者两侧不属于同一个"已算出的"
    /// 内容族（Raw 与 Processed 分属不同族），就淡入。
    fn seam(previous_kind: u8, kind: u8) -> Seam {
        match (Self::family(previous_kind), Self::family(kind)) {
            // 两侧都是宿主原声：同一份信号，短交叉就够。
            (ContentFamily::Passthrough, ContentFamily::Passthrough) => Seam::Crossfade,
            // 任一侧是透传 ⇒ 异源：另一侧是我们算出来的内容，叠起来就是"两声并发"。
            (ContentFamily::Passthrough, _) | (_, ContentFamily::Passthrough) => Seam::FadeIn,
            // Raw 与 Processed 是**同一段素材的两种处理**：内容相关但相位不同，
            // 该用 20 ms 等功率交叉，而不是 1.3 ms 的淡入。
            (ContentFamily::Raw, ContentFamily::Processed)
            | (ContentFamily::Processed, ContentFamily::Raw) => Seam::WideCrossfade,
            // 同族、同源、只是版本变了：内容几乎相同，短交叉即可。
            _ => Seam::Crossfade,
        }
    }

    /// 输出源的内容家族。接缝判据只关心"这一侧是什么内容"，不关心具体是哪一档。
    ///
    /// 【为什么收敛成三档】`CaptureOutput` 有七个取值，但接缝只需要回答一个问题：
    /// 两侧的信号**是否同源**。透传类（宿主原声）与算出来的内容之间必须淡入；
    /// Raw 与 Processed 是同一段素材的两种处理，可以交叉；其余同族切换同理。
    /// 按七个取值逐个写分支，每加一档就要重审一遍全部组合 —— 那正是这里曾经出过
    /// 两次错的原因：`Processed` 与 `Raw` 曾被当成"同源切换"，于是 64 帧里把两段
    /// 不相干的信号线性叠在一起。
    fn family(kind: u8) -> ContentFamily {
        if kind == CaptureOutput::Raw.code() {
            ContentFamily::Raw
        } else if kind == CaptureOutput::PendingSynthesis.code()
            || kind == CaptureOutput::PassThrough.code()
            // 录制监听送出去的也是**宿主原声**：`Recording` 分支把输入总线原样写出，
            // 它从哪一相位开始是宿主的事，我们不该给它加淡入（那会把录制监听的前
            // 5 ms 削掉）。
            || kind == CaptureOutput::Recording.code()
        {
            ContentFamily::Passthrough
        } else {
            ContentFamily::Processed
        }
    }

    /// 按接缝种类装填一次淡化：抓下旧信号、定宽度、复位进度。
    ///
    /// 【为什么宽度也进状态】三种接缝的宽度不同（1.3 ms / 1.3 ms / 20 ms），而淡化
    /// 可能跨块继续。把宽度存进原子量之后，淡化的**每一块**都读到同一个值。
    fn arm_seam(&self, state: &OutputDeclickState, seam: Seam) {
        let len = match seam {
            Seam::FadeIn | Seam::Crossfade => OUTPUT_DECLICK_FRAMES,
            Seam::WideCrossfade => OUTPUT_WIDE_CROSSFADE_FRAMES,
            Seam::FirstBlock => OUTPUT_FIRST_BLOCK_FADE_FRAMES,
        };
        // 淡入 = 把旧信号掩成 0；交叉淡化保留旧信号。
        let keep = match seam {
            Seam::FadeIn | Seam::FirstBlock => 0,
            _ => u32::MAX,
        };
        // 【等功率只在"两条不同源"时用】FadeIn / FirstBlock 的旧侧恒为 0，等功率
        // 退化成一条两端平缓的斜坡（比线性更平滑）；WideCrossfade 两侧是互不相关的
        // 信号，线性会在中点掉 6 dB。Crossfade 两侧是**同一份内容**的两次渲染
        // （相关性接近 1），等功率会在中点叠出 +3 dB 的鼓包 —— 那种场合线性才对。
        state
            .fade_curve
            .store(u32::from(seam.uses_equal_power()), Ordering::Release);
        // 旧信号是"上一块输出的最后 `len` 帧"，在历史缓冲里位于末尾。
        let base = DECLICK_HISTORY_FRAMES - len;
        for channel in 0..2 {
            for index in 0..len {
                let old = state.tail[channel][base + index].load(Ordering::Relaxed);
                state.fade_from[channel][index].store(old & keep, Ordering::Relaxed);
            }
        }
        state.fade_len.store(len as u32, Ordering::Release);
        state.fade_pos.store(0, Ordering::Release);
        state.fade_active.store(true, Ordering::Release);
    }

    pub(crate) unsafe fn apply_output_declick(
        &self,
        stream: ProcessStream,
        output: &mut crate::audio_abi::AudioBusBuffers,
        frames: usize,
    ) {
        if frames == 0 || output.num_channels <= 0 || output.channel_buffers.is_null() {
            return;
        }
        let token = self.output_token[stream.index()].load(Ordering::Acquire);
        let Some(state) = self.declick.get(stream.index()) else {
            return;
        };
        // 【为什么按"种类"判首块】种类变化（Raw ↔ Processed、Processed →
        // PendingSynthesis）才是用户听得出的接缝，同种类的版本更新（曲线编辑后重渲）
        // 由下面的 `last_token` 分支处理。第一块的种类若与本流已知的最后种类不同，
        // 说明上一轮播放停在另一种源上 —— 那一跳正是"咔声位置不固定"的来源之一。
        let kind = (token & 0xFF) as u32;
        // `None` = 本流还没输出过任何一块（真首块）；`Some(prev)` = 上一块的种类。
        // 【为什么两行而不是 `then(|| …)`】`bool::then` 在条件为假时**不调用**闭包，
        // 于是首块那一轮 `primed_kind` 永远不会被写入 —— 下一块就会拿"0"当上一块的
        // 种类（而 0 恰好是 `Untouched` 的编码），把一次真实的源切换判成"同族"。
        let was_primed = state.primed.swap(true, Ordering::AcqRel);
        let previous_kind = state.primed_kind.swap(kind, Ordering::AcqRel);
        let first_kind = was_primed.then_some(previous_kind);
        // 【B0】本块装填的是哪一种接缝（`None` = 没有源切换）。账本要它来区分
        // "这一下咔来自源切换"与"这一下咔来自覆盖边界"。
        let mut armed = crate::render::click::SeamCode::None;
        if first_kind.is_none() {
            // 真首块：没有"上一块输出的尾巴"。
            //
            // 【为什么不是无条件原样输出】旧契约是"从零淡入会改变宿主直通的样本"，
            // 而那**只对透传成立**：透传的样本就是宿主正在播的东西，它自己已经连续。
            // 当第一块是**我们算出来的内容**（Raw / Processed）时，它从哪一相位开始
            // 是任意的 —— 从零直接跳到块内波形就是一次标准的起播咔。
            // 【作用域：只给 Capture 的播放通路】首块淡入要解决的是"我们自己算出来的
            // 内容从任意相位开始"—— 那是 Capture 通路（插件直接对着一条轨出声）的形状。
            // ARA 通路由宿主按 region 调度，区块起点与 item 淡化都是宿主的事；在那里
            // 再叠一层淡入会改动宿主自己的包络（且没有任何报障指向它）。
            let capture_content = !self.is_bound() && self.capture.has_material();
            if capture_content && Self::family(kind as u8) != ContentFamily::Passthrough {
                self.arm_seam(state, Seam::FirstBlock);
                armed = crate::render::click::SeamCode::FadeIn;
            }
            state.last_token.store(token, Ordering::Release);
        } else if first_kind != Some(kind) {
            // 本轮从未去过咔、但上一轮停在另一种源上：仍然要去咔。至于是交叉淡化
            // 还是淡入，取决于上一个源是不是透传（见 `seam`）。
            let previous_kind = first_kind.unwrap_or(kind) as u8;
            let seam = Self::seam(previous_kind, kind as u8);
            self.arm_seam(state, seam);
            armed = seam.code();
            state.last_token.store(token, Ordering::Release);
        } else if self.force_declick[stream.index()].swap(false, Ordering::AcqRel) {
            // 【T-A1-1：同一份源内部的读位置跳变】token 没变，但音频在块边界上确实
            // 错开了一小段（起播窗口收口、循环折返、流交接）。尾巴里存的是上一块真正
            // 输出的样本，与旧位置连续 —— 用它做一次等功率交叉，比"整块换源"轻得多。
            let previous_token = state.last_token.load(Ordering::Acquire);
            let previous_kind = (previous_token & 0xFF) as u8;
            let seam = Self::seam(previous_kind, kind as u8);
            self.arm_seam(state, seam);
            armed = seam.code();
            state.last_token.store(token, Ordering::Release);
        } else {
            // 【为什么先取旧的再交换】"要不要交叉淡化"取决于**上一个**源是什么；
            // `swap` 一执行旧值就没了，所以必须先读。
            let previous_token = state.last_token.load(Ordering::Acquire);
            if state.last_token.swap(token, Ordering::AcqRel) == token {
                // 同一份快照、同一个版本：没有切换，不做任何淡化。
            } else {
                // 【为什么"透传 → 处理结果"不交叉淡化】`PendingSynthesis` / `PassThrough`
                // 输出的是**宿主原声**，`Processed` 输出的是神经声码器结果。两者内容根本
                // 不同源，交叉淡化会把两段不同内容叠在一起（听声是"两声并发"）。
                // 那种场合改为对结果做一次淡入。
                let previous_kind = (previous_token & 0xFF) as u8;
                let current_kind = (token & 0xFF) as u8;
                let seam = Self::seam(previous_kind, current_kind);
                self.arm_seam(state, seam);
                armed = seam.code();
            }
        }
        let channels = (output.num_channels as usize).min(2);
        if state.fade_active.load(Ordering::Acquire) {
            // 宽度由 `arm_seam` 定：透传接缝与同族切换 64 帧，Raw ↔ Processed 960 帧。
            let len = match state.fade_len.load(Ordering::Acquire) as usize {
                0 => OUTPUT_DECLICK_FRAMES,
                len => len.min(DECLICK_HISTORY_FRAMES),
            };
            let position = state.fade_pos.load(Ordering::Acquire) as usize;
            let remaining = len.saturating_sub(position);
            let count = remaining.min(frames);
            for channel in 0..channels {
                // SAFETY: 调用方保证输出总线有 `channels` 个可写平面。
                let plane = unsafe { *output.channel_buffers.add(channel) };
                if plane.is_null() {
                    continue;
                }
                let equal_power = state.fade_curve.load(Ordering::Acquire) != 0;
                for index in 0..count {
                    let t = (position + index + 1) as f32 / len as f32;
                    let old = f32::from_bits(
                        state.fade_from[channel][position + index].load(Ordering::Relaxed),
                    );
                    // SAFETY: `index < frames`，调用方保证平面至少这么长。
                    let current = unsafe { *plane.add(index) };
                    // 【为什么权重不能只有一套】见 `Seam::uses_equal_power`：
                    // 两条不相关信号的线性混合在中点掉 6 dB（听感是一次凹陷/咔），
                    // 而同一份内容的两次渲染用等功率会在中点叠出 +3 dB 的鼓包。
                    let mixed = if equal_power {
                        let (out_gain, in_gain) = crate::render::crossfade::equal_power(t);
                        old * out_gain + current * in_gain
                    } else {
                        old * (1.0 - t) + current * t
                    };
                    unsafe {
                        *plane.add(index) = mixed;
                    }
                }
            }
            let next = position + count;
            state.fade_pos.store(next as u32, Ordering::Release);
            if next >= len {
                state.fade_active.store(false, Ordering::Release);
            }
        }
        // 保存本块输出的尾巴，供下一次源切换使用。
        //
        // 【为什么短块要左移历史、把新样本追加在末尾】`tail` 的索引语义是
        // "0 = 最近 64 帧里最旧的一帧、63 = 最新一帧"。块长不足 64 时，若像旧实现
        // 那样把样本放在 0 起、右边补零，淡出会从**本块第一帧**开始、然后读到零，
        // 与"边界处的最新样本"差出整整一个块长 —— 短块宿主上每次接缝都会多一次
        // 阶跃。左移历史再追加，才能保持"最后 64 帧"这一不变量。
        if frames >= DECLICK_HISTORY_FRAMES {
            let start = frames - DECLICK_HISTORY_FRAMES;
            for channel in 0..channels {
                // SAFETY: 同上。
                let plane = unsafe { *output.channel_buffers.add(channel) };
                if plane.is_null() {
                    continue;
                }
                for index in 0..DECLICK_HISTORY_FRAMES {
                    unsafe {
                        state.tail[channel][index]
                            .store((*plane.add(start + index)).to_bits(), Ordering::Relaxed);
                    }
                }
            }
        } else {
            let written = frames;
            for channel in 0..channels {
                // SAFETY: 同上。
                let plane = unsafe { *output.channel_buffers.add(channel) };
                if plane.is_null() {
                    continue;
                }
                for index in 0..(DECLICK_HISTORY_FRAMES - written) {
                    let value = state.tail[channel][index + written].load(Ordering::Relaxed);
                    state.tail[channel][index].store(value, Ordering::Relaxed);
                }
                for index in 0..written {
                    unsafe {
                        state.tail[channel][DECLICK_HISTORY_FRAMES - written + index].store(
                            (*plane.add(index)).to_bits(),
                            Ordering::Relaxed,
                        );
                    }
                }
            }
        }

        // 【B0】把这一块的形状记进账本。音频线程：只写原子量。
        //
        // 【为什么放在最后】这里读到的就是**已经输出**的样本（去咔之后），于是
        // `max_step` 描述的正是用户听到的那一段。放在去咔之前会量到被淡化前的阶跃，
        // 那对"到底有没有响一下"没有意义。
        self.record_click_sample(
            stream,
            kind as u8,
            armed,
            first_kind.is_none(),
            output,
            frames,
        );
    }

    /// 门控清零时的去咔：把上一块的尾巴在首段内线性淡到零，其余帧写零。
    ///
    /// 【为什么必须独立于 `apply_output_declick`】后者就地做
    /// `old*(1-t) + current*t`，依赖"当前缓冲里已经写着本块的新内容"。门控分支里
    /// 缓冲里是**上一块的残留**（in-place 宿主下更是等于输入），而且紧跟着的
    /// `clear_outputs` 会把整块（含刚淡好的 64 帧）一起写零 —— 先淡后清等于白淡，
    /// 先清后淡则 `current` 恒为 0。两条都得不到"从上一块尾巴淡到零"。
    ///
    /// 本函数自己写满整块：前 `min(frames, OUTPUT_DECLICK_FRAMES)` 帧按尾巴淡出，
    /// 其余为零，并把尾巴清零。调用方因此**不要**再调 `clear_outputs`。
    ///
    /// # Safety
    /// `output` 必须是本回调内有效的 VST3 输出总线；`frames` 不超过其平面长度。
    pub(crate) unsafe fn apply_gate_fade_out(
        &self,
        stream: ProcessStream,
        output: &mut crate::audio_abi::AudioBusBuffers,
        frames: usize,
    ) {
        if frames == 0 || output.num_channels <= 0 || output.channel_buffers.is_null() {
            return;
        }
        let Some(state) = self.declick.get(stream.index()) else {
            return;
        };
        let declared = output.num_channels as usize;
        let fade = frames.min(OUTPUT_DECLICK_FRAMES);
        // `tail` 里 0 是最旧的、63 是最新的；短块淡出必须从**最新的 fade 帧**开始，
        // 否则边界处仍会从一段旧样本跳到零。
        let base = DECLICK_HISTORY_FRAMES - fade;
        let mut any_signal = false;
        for channel in 0..declared {
            // SAFETY: 宿主声明 `num_channels` 个平面指针，非空平面至少含 `frames` 帧。
            let plane = unsafe { *output.channel_buffers.add(channel) };
            if plane.is_null() {
                continue;
            }
            if channel < 2 {
                for index in 0..fade {
                    let old = f32::from_bits(
                        state.tail[channel][base + index].load(Ordering::Relaxed),
                    );
                    if old != 0.0 {
                        any_signal = true;
                    }
                    // 分母用 `fade` 而不是常量：块短于 64 帧时也必须在本块内淡到零，
                    // 否则块尾留下非零值，下一块从零起就是一次新的阶跃。
                    let gain = 1.0 - (index + 1) as f32 / fade as f32;
                    // SAFETY: index < fade ≤ frames。
                    unsafe { *plane.add(index) = old * gain };
                }
                for index in fade..frames {
                    // SAFETY: index < frames。
                    unsafe { *plane.add(index) = 0.0 };
                }
            } else {
                // SAFETY: 同上；多声道总线的后几个平面不在尾巴里，直接写零。
                unsafe { std::ptr::write_bytes(plane, 0, frames) };
            }
        }
        // 【为什么按实际输出报静音位】本块通常含一段淡出（非静音）；但尾巴本来就
        // 全零时（首次静音、或连续静音的第二块），整块确实全是零 —— 这时如实报
        // 静音，宿主才能跳过下游处理。恒报"非静音"会在每次静音开始时白跑一整条链路。
        output.silence_flags = if any_signal {
            0
        } else {
            (1_u64 << declared.min(64)) - 1
        };
        // 尾巴与输出源一起归零：下一块从静音起，不需要（也不该）再交叉淡化。
        for channel in 0..declared.min(2) {
            for index in 0..OUTPUT_DECLICK_FRAMES {
                state.tail[channel][index].store(0, Ordering::Relaxed);
            }
        }
        state.fade_active.store(false, Ordering::Release);
        // 【为什么不是 0】`primed_kind == 0` 的语义是"这个流从未输出过任何东西"，
        // 下一块会被当成真首块**原样输出** —— 而本块刚刚输出了一整段淡出到零的
        // 静音。从零直接跳到完整内容就是用户听到的"咔"。借用 PassThrough 的语义：
        // 它让 `apply_output_declick` 把 `fade_from` 视为全零，于是下一块做一次
        // 线性淡入。
        state.primed_kind.store(
            u32::from(CaptureOutput::PassThrough.code()),
            Ordering::Release,
        );
        state.last_token.store(0, Ordering::Release);
    }

    /// 取走"该重新合成一次"的意图（非音频线程调用；见 [`Self::render_wanted`]）。
    pub(crate) fn take_render_wanted(&self) -> bool {
        self.render_wanted.swap(false, Ordering::AcqRel)
    }

    /// 一次性诊断：有素材，但处理结果不是当前代次（非音频线程调用）。
    ///
    /// 【为什么需要它】"合成从来没被调度过"此前在界面上与"没有素材"共用同一个
    /// `passthrough` 读数，用户和我们都看不出区别。这条日志把结论直接写下来：
    /// 素材代次是多少、处理结果代次停在哪。合成成功后它自然不再出现。
    pub(crate) fn report_pending_synthesis_once(&self) {
        if !self.capture.has_material() {
            return;
        }
        let revision = self.capture.revision();
        let ready = self.capture_ready_revision.load(Ordering::Acquire);
        if ready == revision {
            return;
        }
        // 按代次去重：同一份素材只报一次。
        if self
            .pending_reported_revision
            .swap(revision, Ordering::AcqRel)
            == revision
        {
            return;
        }
        log::warn!(
            "[capture] material revision {revision} has no processed result yet \
             (last published revision {ready}); the render scheduler has not caught up"
        );
    }

    /// 记录本块的几何，并回答"位置流是否在推进"（音频线程，无锁无分配）。
    ///
    /// 【判据：块首变化，或同头块变长】块长由宿主决定（SDK 只承诺
    /// `numSamples <= maxSamplesPerBlock`），起播第一个块可以很大，也可以与停播块
    /// 同头。因此：
    ///
    ///   * 块首移动 ⇒ 时间轴在走（seek / 循环回卷也算，代价是至多一块瞬态）；
    ///
    ///   * 块首没动但块变长 ⇒ 起播给了更大的块，也是在走；
    ///
    ///   * 块首没动且块不变长/变短 ⇒ 钉住/重复（或宿主在停播时缩小了块）。
    ///
    /// 【为什么不比较块尾/历史高水位】那会让停播/预渲染留下的旧位置把起播块
    /// 压住：播放头还没越过旧块尾时，每一块都被判成"没前进"（用户报障：开头约
    /// 0.7 秒静音/卡顿）。当前块与上一块的关系不会跨停播残留。
    ///
    /// 【首块】`flow == true`（本流第一次出现）。门控允许它出音一次；重复抑制由
    /// [`Self::rt_last_emitted`] 负责，不再靠“首块保守静音”。
    #[cfg(test)]
    pub(crate) fn note_block_position(&self, start: i64, frames: usize) -> bool {
        self.note_stream_position(ProcessStream::Realtime, start, frames)
            .0
    }

    /// 只更新**本流**的几何并回答 `(flow, had_previous)`。
    ///
    /// 【为什么必须带 stream】`kPrefetch` 与 `kRealtime` 的位置节奏不同；跨流比较
    /// 会让 prefetch 的移动给实时块的重复投递作“在前进”的伪证。
    #[cfg(test)]
    pub(crate) fn note_stream_position(
        &self,
        stream: ProcessStream,
        start: i64,
        frames: usize,
    ) -> (bool, bool) {
        let block = self.note_stream_geometry(
            stream,
            start,
            frames,
            48_000.0,
            StreamTime::default(),
            false,
            false,
            // 【这个入口没有 transport 状态】它只回答"这一块的几何形状变了没有"
            // （`(flow, had_previous)`），调用点全在测试里。没有 `ProcessContext`
            // 就没有 `kPlaying`，取"transport 在走"是与它原有契约一致的假设 ——
            // 生产路径走 `note_stream_block`，那里的 `playing` 来自真实上下文。
            true,
        );
        (block.phase != TransportPhase::Resting, block.had_previous)
    }

    /// 生产入口：从完整 `ProcessContext` 计算有效位置与相位。
    fn note_stream_block(
        &self,
        stream: ProcessStream,
        context: &crate::audio_abi::ProcessContext,
        frames: usize,
        playing_rising: bool,
        recording_rising: bool,
    ) -> StreamBlock {
        let rate = if context.sample_rate.is_finite() && context.sample_rate > 0.0 {
            context.sample_rate
        } else {
            self.setup_sample_rate()
        };
        self.note_stream_geometry(
            stream,
            context.project_time_samples,
            frames,
            rate,
            StreamTime::from_context(context),
            playing_rising,
            recording_rising,
            // kPlaying 是 SDK 对"transport 是否在走"的权威回答，与位置、采样率、
            // 样本同属**同一个块**。计数器的推进权必须以它为准（见 may_advance）。
            context.state & crate::audio_abi::PROCESS_CONTEXT_PLAYING != 0,
        )
    }

    /// 每流几何 + 有效位置 + 相位。音频线程调用：只读写原子，不分配。
    ///
    /// # 四条规则（没有第五条）
    ///
    /// ```text
    /// R1  新 pass（!had / playing_rising / recording_rising / raw_start < last_raw）
    ///         ⇒ anchor = raw_start，计数器归零，position = raw_start
    /// R2  宿主没报新位置（raw_start == last_raw）
    ///         ⇒ position = anchor + 已送来的帧数
    ///            （若 continuousTimeSamples / projectTimeMusic 在前进，优先用它们）
    /// R3  宿主报了新位置（raw_start != last_raw）
    ///         ⇒ position = raw_start。宿主永远赢，永不"纠正"它
    /// R4  任何 raw_start < last_raw
    ///         ⇒ 按 R1 处理（新 pass）。**永不**当成"宿主读数滞后"
    /// ```
    ///
    /// # 为什么删掉了 1.5 s 容差与"锁存偏移"
    ///
    /// 旧实现把"宿主报的位置比我们的累积时钟落后不超过 `POSITION_DRIFT_TOLERANCE_SEC`"
    /// 判成"宿主读数滞后"，于是继续用累积时钟，并把差值**锁存**下来供后续每一块复用。
    ///
    /// 用户回到起点再播 0.7 s 时，那个差恰好是 −0.7 s —— **在容差之内** ⇒ 位置被强制成
    /// "上一遍末尾 + 块长"，整遍内容偏移一个循环长度；而且锁存一生效，`offset_valid`
    /// 就再也不清（只有新起播/循环/一次 >250 ms 的跳变能清它），于是**整场会话**都走
    /// 这条路。这就是用户说的"反复播放 0.7 s，内容完全与它应该输出的对不上"。
    ///
    /// 教训是结构性的：**只要存在两只钟，就必须有一个仲裁，而任何仲裁都会在某一种
    /// 真实操作上判错**。所以这里不仲裁 —— 宿主报了，就听宿主的（R3）；宿主没报，
    /// 才按帧数接着数（R2），而那个数**可核对**：宿主下次报位置时 R3 立刻接管，
    /// 差值记进 `jumps` 并交给捕获侧做记账纠正（桥接/裁剪），不会变成 seek。
    ///
    /// # 为什么 R2 仍然必须存在
    ///
    /// 实测 VEGAS 在 Capture 播放期间会让 `projectTimeSamples` 停住不动。那时
    /// "每块前进 frames"是唯一自洽的换算 —— 它是合法记账，不是猜测。把它删掉不会让
    /// 宿主变好，只会让 take 变成一大片洞。
    #[allow(clippy::too_many_arguments)]
    fn note_stream_geometry(
        &self,
        stream: ProcessStream,
        raw_start: i64,
        frames: usize,
        rate: f64,
        time: StreamTime,
        playing_rising: bool,
        recording_rising: bool,
        playing: bool,
    ) -> StreamBlock {
        let geometry = &self.streams[stream.index()];
        geometry.blocks.fetch_add(1, Ordering::Relaxed);
        let had = geometry.seen.load(Ordering::Acquire);
        let rate = if rate.is_finite() && rate > 0.0 {
            rate
        } else {
            48_000.0
        };
        let pre_roll_cap = (rate * PRE_ROLL_CAP_SEC).round().max(1.0) as u64;
        if frames == 0 {
            // 零帧块是参数 flush，不代表时间轴在走。
            return StreamBlock {
                position: raw_start,
                raw_start,
                phase: if had {
                    TransportPhase::Resting
                } else {
                    TransportPhase::Start
                },
                had_previous: had,
                cycle_wrap: false,
                new_pass: false,
                activity_moving: false,
                resync: false,
                resync_limit_frames: 0,
                preceding_silent_frames: geometry.trailing_silent_frames.load(Ordering::Acquire),
                counted: false,
                discard_pre_roll: false,
                start_window: false,
                repeated_nonzero: false,
                // 零帧块不写任何样本，位置来源对它没有意义；沿用上一块的结论即可。
                position_source: PositionSource::from_code(
                    geometry.last_source.load(Ordering::Acquire),
                )
                .unwrap_or(PositionSource::Project),
            };
        }
        let frames32 = u32::try_from(frames).unwrap_or(u32::MAX);

        let last_position = geometry.last_position.load(Ordering::Acquire);
        let last_raw = geometry.last_raw_start.load(Ordering::Acquire);
        let last_continuous = geometry.last_continuous.load(Ordering::Acquire);
        let last_music = f64::from_bits(geometry.last_music_bits.load(Ordering::Acquire));
        let last_source = PositionSource::from_code(geometry.last_source.load(Ordering::Acquire))
            .unwrap_or(PositionSource::Project);
        let anchor_before = geometry.anchor.load(Ordering::Acquire);
        let advance_before = geometry.frames_since_anchor.load(Ordering::Acquire);
        // 【为什么先读再判】`resync_limit_frames` 的语义是"计数器**可能**错过的范围"，
        // 必须用本块处理**之前**的累积值；本块一写就被清零了。
        let counted_before = advance_before.saturating_add(frames as u64);

        // 【循环回卷】只有循环播放才有"回卷"这个形状：块首回退，或块首不变而时间前进。
        let cycle_wrap = if time.cycle_active {
            let continuous_advanced = time
                .continuous
                .is_some_and(|continuous| continuous > last_continuous);
            let music_wrapped = time.music.is_some_and(|music| music + 1e-9 < last_music);
            let music_boundary = time.cycle_valid
                && time.continuous.is_none()
                && time.music.is_some_and(|music| {
                    music <= time.cycle_start_music + 1e-6
                        && last_music >= time.cycle_end_music - 1e-6
                });
            had && (raw_start < last_raw
                || (raw_start == last_raw
                    && (continuous_advanced || music_wrapped || music_boundary)))
        } else {
            false
        };

        let host_rewound = had && raw_start < last_raw;
        let host_moved = had && raw_start != last_raw;
        // R1 + R4：新一次播放。回退**永远**是新 pass —— 它不是"读数滞后"。
        let new_pass = !had || playing_rising || recording_rising || host_rewound;

        // 本块位置的"锚 + 偏移"。R1/R3 时锚就是宿主刚报的值、偏移为 0；
        // R2 时锚保持不动、偏移是已送来的帧数。两条路径因此共用同一句
        // `anchor + advance`，不存在"两个域"。
        let (anchor, advance) = if new_pass || host_moved {
            (raw_start, 0_u64)
        } else {
            (anchor_before, advance_before)
        };

        // 【停播重复投递的判据】同一份**非零**内容逐块重复 ⇒ 宿主在停播时反复送同一
        // 段缓冲，那不是播放时间。它与 VEGAS 起播窗口的区分点是内容：窗口送的是静音，
        // FL Studio 停播送的是同一份非零内容。
        let silent = geometry.last_block_silent.load(Ordering::Acquire);
        let repeated_nonzero = geometry.unchanged_run.load(Ordering::Acquire) > 0 && !silent;

        let mut position = raw_start;
        let mut source = if time.cycle_active {
            PositionSource::Cycle
        } else {
            PositionSource::Project
        };
        let mut counted = false;

        if new_pass || host_moved {
            // R1 / R3 / R4：宿主给了位置，它就是权威。
            //
            // R3 的差值只做**记账**（`jumps` + `last_jump`），绝不参与位置的选取 ——
            // 那正是旧实现出错的地方。真正的纠正发生在捕获侧：它持有 `expected_next`，
            // 才判得出"这个差值是桥接还是裁剪"。
            if host_moved && !new_pass {
                let expected = anchor_before.saturating_add(advance_before as i64);
                let jump = raw_start.saturating_sub(expected);
                if jump != 0 {
                    geometry.jumps.fetch_add(1, Ordering::Relaxed);
                    geometry.last_jump.store(jump, Ordering::Release);
                }
            }
            geometry.anchor.store(raw_start, Ordering::Release);
            geometry.continuous_anchor.store(
                time.continuous.unwrap_or(last_continuous),
                Ordering::Release,
            );
            geometry.music_anchor.store(
                time.music.unwrap_or(last_music).to_bits(),
                Ordering::Release,
            );
            // 【锚点在本块起点上】本块的偏移是 0，下一块（若连续）是一块长。
            geometry
                .frames_since_anchor
                .store(frames as u64, Ordering::Release);
            if new_pass {
                // 【窗口只在上升沿打开】`!had` 单独不打开：第一块的位置本来就是宿主
                // 给的，不需要推进；而"宿主重复投递同一位置"必须以不推进告终。
                geometry.advance_limit.store(
                    if playing_rising || recording_rising {
                        pre_roll_cap.min(u64::from(u32::MAX)) as u32
                    } else {
                        0
                    },
                    Ordering::Release,
                );
                // 【活动额度随 pass 充满】一次播放只有一个开头，所以"内容在流动"这条
                // 证据的可信长度也只有一个开头那么多。用完就不再借它。
                geometry
                    .activity_budget
                    .store(pre_roll_cap, Ordering::Release);
                geometry.pass.fetch_add(1, Ordering::Relaxed);
            } else {
                // R3：宿主报了新位置 ⇒ 计数器不再有权推进，窗口一并作废。
                geometry.advance_limit.store(0, Ordering::Release);
            }
            position = raw_start;
        } else {
            // R2：宿主没报新位置。用最好的宿主时钟，最后才是计数器。
            let sane = (frames as i64).saturating_mul(4).max(64);
            //
            // 见到"逐块完全相同的非零内容"就当场把窗口作废 —— 那是停播重复投递的
            // 形状，继续推进会让停播的光标一直"前进"（FL Studio 的既有回归）。
            if repeated_nonzero && advance < u64::from(geometry.advance_limit.load(Ordering::Acquire)) {
                geometry.advance_limit.store(0, Ordering::Release);
            }
            let limit = u64::from(geometry.advance_limit.load(Ordering::Acquire));
            let window_open = advance < limit;
            // 【计数器凭什么有权推进】只有两种情形，且**两者都有明确预算**：
            //
            //   (a) 起播窗口（window_open）—— 本 pass 刚开始，宿主还没接管位置，
            //       预算由 advance_limit 给（PRE_ROLL_CAP_SEC）。
            //   (b) 真实播放（playing）且内容在流动 —— 宿主说 transport 在走，
            //       位置读数可能滞后，用块长接着数是对的。
            //
            // 【为什么 (b) 必须带 playing】旧实现只看"内容在流动"，于是宿主
            // **停播**（kPlaying 为假）却仍在送预取音频时，计数器照样一路前进 ——
            // 实测一次跑掉 162992 帧（3.4 s），而宿主报的位置只挪了 1220 帧。
            // 那 3.4 s 的偏移正是"反复播 0.7 s 内容完全对不上"：重播时我们从
            // anchor + 偏移 读，读到的是素材里另一段。
            //
            // 【为什么不是"按秒数封顶"】封顶需要一个固定常数（比如 0.25 s），
            // 而实测的 VEGAS 起播窗口是 0.66 s —— 任何小于它的常数都会把起播切断。
            // 预算本来就该由"本 pass 是不是刚开始"给出，而不是一个与宿主无关的秒数。
            // 【"内容在流动"这条证据也必须有限额】它是起播窗口的第二条通路：
            // VEGAS 的起播窗口里宿主把位置钉在 0、却已经在按块长送音频，那时
            // `playing_rising` 早已过去，`window_open` 也已用完，只有"内容在变"
            // 还能说明时间在走。
            //
            // 但这条证据**无限用**就成了缺陷：宿主停播后仍在送预取音频时，内容
            // 同样在变，计数器于是一路前进 —— 实测一次跑掉 162992 帧（3.4 s），
            // 而宿主报的位置只挪了 1220 帧。那正是"反复播 0.7 s 内容完全对不上"。
            //
            // 所以给它与本 pass 起播窗口**同一份额度**：`activity_budget` 在
            // `new_pass` 时充满，用一帧扣一帧。额度用完，位置就回到宿主报的那个点，
            // 等宿主恢复报数由 R3 接管。
            //
            // 【为什么额度不能是"每次 pass 都重新充满"之外的东西】起播窗口本身就是
            // 一次性的：一次播放只有一个开头。额度绑在 pass 上，语义与它完全一致。
            // 【`kPlaying` 在这里的作用】它把"内容在流动"这条证据分成两种用法：
            //
            //   * 宿主说 transport 在走（`playing`）⇒ 位置读数滞后是常态，
            //     计数器就是正确来源，**不设额度**；
            //   * 宿主说没在走（停播 / 起播窗口）⇒ 位置才是权威，
            //     只在"本 pass 刚开始"的额度内才允许猜。
            //
            // 现场日志里跑飞的那 3.4 s 全部发生在 `playing=0` 的块上，所以第二档
            // 一收紧，漂移就没有了；而真实播放（`playing=1`）一个行为都没变。
            let mut may_advance = window_open;
            let mut uses_activity_budget = false;
            if !may_advance && geometry.activity_moving() {
                if playing {
                    may_advance = true;
                } else if geometry.activity_budget.load(Ordering::Acquire) >= frames as u64 {
                    may_advance = true;
                    uses_activity_budget = true;
                }
            }
            // 本块相对锚点的偏移；`advance` 是"按块长算，本块应该在哪"。
            let mut offset = advance as i64;
            let mut advanced = false;
            if !time.cycle_active {
                // 【宿主时钟优先，且它是**绝对**的】位置 = 锚点 + (当前读数 − 锚点读数)。
                // 不与计数器叠加 —— 叠加会把同一段时间算两遍。
                if let Some(continuous) = time.continuous {
                    let from_anchor = continuous.saturating_sub(
                        geometry.continuous_anchor.load(Ordering::Acquire),
                    );
                    let step = from_anchor.saturating_sub(offset);
                    if from_anchor >= 0 && step >= 0 && step <= sane {
                        offset = from_anchor;
                        position = anchor.saturating_add(from_anchor);
                        source = PositionSource::Continuous;
                        advanced = true;
                        self.continuous_used.fetch_add(1, Ordering::Relaxed);
                        geometry.continuous_used.fetch_add(1, Ordering::Relaxed);
                    }
                }
                if !advanced {
                    if let (Some(music), Some(tempo)) = (time.music, time.tempo) {
                        let anchor_music =
                            f64::from_bits(geometry.music_anchor.load(Ordering::Acquire));
                        let beats = music - anchor_music;
                        if beats > 0.0 {
                            let from_anchor = (beats * 60.0 / tempo * rate).round() as i64;
                            let step = from_anchor.saturating_sub(offset);
                            if step >= 0 && step <= sane {
                                offset = from_anchor;
                                position = anchor.saturating_add(from_anchor);
                                source = PositionSource::Music;
                                advanced = true;
                                self.music_used.fetch_add(1, Ordering::Relaxed);
                                geometry.music_used.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            }
            if !advanced && may_advance {
                // 计数器：宿主没说，就按"收到了多少帧"接着数 —— 但只在有证据的时候。
                position = anchor.saturating_add(offset);
                counted = true;
                advanced = true;
                if uses_activity_budget {
                    geometry.activity_budget.fetch_update(
                        Ordering::AcqRel,
                        Ordering::Acquire,
                        |budget| Some(budget.saturating_sub(frames as u64)),
                    )
                    .ok();
                }
                geometry.counted_blocks.fetch_add(1, Ordering::Relaxed);
                geometry.counted_frames.fetch_add(frames as u64, Ordering::Relaxed);
            }
            if !advanced {
                // 【停播：位置回到宿主报的那个点】没有任何证据说明时间在走，而宿主
                // 反复投递的是同一个位置 —— 继续累加等于把停播时间算成播放时间，
                // 捕获于是把同一小段反复写成新素材（FL Studio 的既有回归）。
                position = anchor;
                offset = 0;
            }
            geometry.frames_since_anchor.store(
                offset.saturating_add(frames as i64).max(0) as u64,
                Ordering::Release,
            );
        }

        // ── 前导静音（诊断 + 门控）────────────────────────────────────
        let mut preceding_silent_frames = geometry.trailing_silent_frames.load(Ordering::Acquire);
        let mut discard_pre_roll = false;
        // 【停播静音 vs 起播窗口静音】两者都是"全零块"，但只有后者是播放时间。
        //
        // 真正能区分它们的是**上升沿之前那一块的内容**：
        //   * 起播窗口：宿主先送**静音**，内容出现时才把播放位置置 1；
        //   * 停播重复投递：宿主把**同一段有内容的音频**反复送。
        // 所以判据是"计数器已经累积超过一个窗口 **且** 上一块不是静音"。
        if (playing_rising || recording_rising) && advance_before > pre_roll_cap && !silent {
            preceding_silent_frames = 0;
            discard_pre_roll = true;
        }
        // 宿主没报新位置时，静音块才是"播放中的静音"；停播重复投递不累积。
        let pinned_since_last = !had || raw_start == last_raw;
        if silent && pinned_since_last {
            geometry.trailing_silent_frames.store(
                preceding_silent_frames
                    .saturating_add(frames as u64)
                    .min(pre_roll_cap),
                Ordering::Release,
            );
        } else if !silent {
            geometry.trailing_silent_frames.store(0, Ordering::Release);
        }

        // ── 相位 ──────────────────────────────────────────────────────
        let activity_moving = geometry.activity_moving();
        let phase = if !had {
            geometry.same_start.store(0, Ordering::Release);
            geometry.extension_used.store(false, Ordering::Release);
            TransportPhase::Start
        } else if cycle_wrap || position != last_position || activity_moving {
            // 【为什么音频在流动也算 Rolling】起播窗口里宿主可以把位置钉住，而音频
            // 已经在按块长流动。此时"位置没变"不成立 —— 我们刚刚按块长推进了它。
            geometry.same_start.store(0, Ordering::Release);
            geometry.extension_used.store(false, Ordering::Release);
            TransportPhase::Rolling
        } else if playing_rising || recording_rising {
            geometry.same_start.store(0, Ordering::Release);
            geometry.extension_used.store(false, Ordering::Release);
            TransportPhase::Start
        } else {
            geometry.same_start.fetch_add(1, Ordering::AcqRel);
            let extension_min = (rate / 20.0).round().max(1.0) as u32;
            let last_frames = geometry.last_frames.load(Ordering::Acquire);
            if frames32 > last_frames.saturating_add(extension_min)
                && !geometry.extension_used.swap(true, Ordering::AcqRel)
            {
                TransportPhase::Rolling
            } else {
                TransportPhase::Resting
            }
        };

        if playing_rising || recording_rising {
            geometry.rising_edges.fetch_add(1, Ordering::Relaxed);
        }
        match phase {
            TransportPhase::Resting => {
                geometry.resting_blocks.fetch_add(1, Ordering::Relaxed);
                self.stream_resting_blocks[stream.index()].fetch_add(1, Ordering::Relaxed);
            }
            TransportPhase::Start | TransportPhase::Rolling => {
                geometry.rolling_blocks.fetch_add(1, Ordering::Relaxed);
                self.stream_rolling_blocks[stream.index()].fetch_add(1, Ordering::Relaxed);
            }
        }
        if cycle_wrap {
            self.cycle_wraps.fetch_add(1, Ordering::Relaxed);
            geometry.cycle_wraps.fetch_add(1, Ordering::Relaxed);
        } else if time.cycle_active && had && raw_start == last_raw && !playing_rising {
            // 同一块首、cycle 有效但没有可用的回卷证据：可能是短循环，也可能是停播。
            // 不能据此写入；只记一个读数，交给捕获侧的判据解释。
            geometry.cycle_wrap_unknown.fetch_add(1, Ordering::Relaxed);
        }

        // ── I-3 的读数：位置是否逐块续接 ─────────────────────────────
        //
        // 稳态下 `position - last_position` 恒等于块长。它不等于块长只有三种解释：
        // ① 宿主真的 seek（用户动作，可接受）；② pass 边界（可接受）；
        // ③ **我们的记账漏了一项**（不可接受，且它就是咔）。
        if had && !new_pass && !cycle_wrap {
            let step = position.saturating_sub(last_position);
            if step < 0 {
                geometry.clock_rewinds.fetch_add(1, Ordering::Relaxed);
            }
            let magnitude = step.unsigned_abs();
            if geometry.clock_deltas.load(Ordering::Acquire) == 0 {
                geometry.clock_delta_min.store(magnitude, Ordering::Release);
            } else {
                geometry.clock_delta_min.fetch_min(magnitude, Ordering::Relaxed);
            }
            geometry.clock_delta_max.fetch_max(magnitude, Ordering::Relaxed);
            geometry.clock_deltas.fetch_add(1, Ordering::Relaxed);
        }

        geometry.seen.store(true, Ordering::Release);
        geometry.last_start.store(position, Ordering::Release);
        geometry.last_frames.store(frames32, Ordering::Release);
        geometry
            .last_end
            .store(position.saturating_add(frames as i64), Ordering::Release);
        geometry.last_position.store(position, Ordering::Release);
        geometry.last_raw_start.store(raw_start, Ordering::Release);
        geometry.last_continuous.store(
            time.continuous.unwrap_or(last_continuous),
            Ordering::Release,
        );
        geometry.last_music_bits.store(
            time.music.unwrap_or(last_music).to_bits(),
            Ordering::Release,
        );
        geometry.last_source.store(source.code(), Ordering::Release);
        if stream == ProcessStream::Realtime {
            self.realtime_flow
                .store(phase != TransportPhase::Resting, Ordering::Release);
        }

        // 【resync：宿主纠正了我们数出来的位置】这不是 seek，是记账纠正 —— 真正的
        // 差值判定在捕获侧（它才持有 `expected_next`），这里只给上限与意图。
        let resync = had
            && !time.cycle_active
            && !new_pass
            && host_moved
            && (counted_before > 0
                || matches!(
                    last_source,
                    PositionSource::Continuous | PositionSource::Music
                ));
        StreamBlock {
            position,
            raw_start,
            phase,
            had_previous: had,
            cycle_wrap,
            new_pass,
            activity_moving,
            resync,
            resync_limit_frames: counted_before,
            preceding_silent_frames,
            counted,
            discard_pre_roll,
            start_window: advance
                < u64::from(geometry.advance_limit.load(Ordering::Acquire)),
            repeated_nonzero,
            position_source: source,
        }
    }

    /// 宿主此刻**确实在推进工程位置**（无锁、无分配，可从音频线程调用）。
    ///
    /// 【为什么要这个判据而不是 `kPlaying`】`kPlaying` 与 `projectTimeSamples` 不是原子
    /// 更新的：宿主从"停止"切到"播放"的那几块里，位置已经在走、播放位可能还没置上。
    /// 而"位置在前进"是宿主无关的硬事实 —— 真在播就必然前进，钉住不动就是在停。
    ///
    /// 它服务的唯一决定是"停播时要不要清输出"：清是对的（固定光标处不清就会把同一小段
    /// 反复送出去，听起来是个嗡嗡声），但**不能拿播放位当唯一的判据** —— 那会让每次
    /// 播放都吞掉开头（用户报障：播放后约 1 秒没有输出）。
    ///
    /// 【判据的粒度是"当前块 vs 上一块"】见 [`Self::note_block_position`]：
    /// 块首移动或同头块变长即算在走。门控**不会**被停播/预渲染留下的历史位置卡住。
    ///
    /// 【为什么 ARA 实例恒为假】它的输出语义由 ARA 通路决定，这个守卫必须对它完全无效
    /// （见 `a_bound_instance_keeps_its_output`）。返回假 ⇒ 保持"停播即静音"的既有行为。
    pub(crate) fn capture_transport_running(&self) -> bool {
        if self.is_bound() || self.bound_once.load(Ordering::Acquire) {
            return false;
        }
        self.realtime_flow.load(Ordering::Acquire)
    }

    /// 门控问“本块是否可以不清空”。
    ///
    /// 【真值表】ARA 实例：`false`（保持“停播即静音”的既有语义，由调用方的
    /// `!kPlaying` 决定）。未绑定实例：offline/prefetch 恒真（它们的输出稍后才被
    /// 播放，不能用此刻的传输状态挖洞）；实时流看本块证据。
    pub(crate) fn capture_should_emit(
        &self,
        evidence: Option<BlockEvidence>,
        playing: bool,
    ) -> bool {
        if self.is_bound() || self.bound_once.load(Ordering::Acquire) {
            // 【为什么 ARA 实例也要区分流】REAPER 在预取 / 擦洗 / 循环折返时会
            // **合法地**以 `kPlaying == 0` 调用 `process`，而预取块会被它缓存并真的
            // 播出来 —— 按 `kPlaying` 清零等于往缓存里写静音（"有概率完全没声音"）。
            // offline / prefetch 与 Capture 实例同一判据：输出稍后才被播放，此刻的
            // 传输状态不适用。实时流仍保持 ARA 的"停播即静音"语义：光标处的重复块
            // 不该循环出声，而真实起播由 `kPlaying` 与捕获侧证据驱动。
            return match evidence {
                Some(evidence) => match evidence.stream {
                    ProcessStream::Offline | ProcessStream::Prefetch => true,
                    ProcessStream::Realtime => playing,
                },
                None => playing,
            };
        }
        match evidence {
            Some(evidence) => match evidence.stream {
                ProcessStream::Offline => true,
                // prefetch 的输出稍后才被播放；重复请求可能是参数变化后的重算，
                // 抑制会留下旧结果。传输状态与重复判定都不适用于它。
                ProcessStream::Prefetch => true,
                ProcessStream::Realtime => {
                    // 【为什么起播窗口要放行】窗口内的前段可能是静音；按
                    // `allow_output == false` 清掉它，内容出现时就是一次零→内容的
                    // 阶跃（"咔"）。窗口本身是"宿主刚开始播放"的证据，输出应当
                    // 连续。`repeated_nonzero` 仍然阻止 FL 停播继续出声。
                    evidence.allow_output || (evidence.start_window && !evidence.repeated_nonzero)
                }
            },
            // 没有上下文证据时退回宿主播放位（旧测试/退化路径）。
            None => playing,
        }
    }

    /// 已发布的**处理结果**是否对应当前活动 take。
    ///
    /// 【为什么只比 take、不比编辑代次】编辑代次在会话那一侧，音频线程取不到。而"曲线
    /// 改了但合成还没跑完"这一档若回落到原始磁带，用户刚调完音高就会听到**未处理的原声**
    /// —— 那是倒退。只比 take 号：曲线编辑不改 take，于是仍然读处理结果（听到上一版处理，
    /// 与既有行为一致）；换 take 才回落。
    fn processed_ready(&self) -> bool {
        if !self.capture.has_material() {
            return false;
        }
        let revision = self.capture.revision();
        self.capture_ready_revision.load(Ordering::Acquire) == revision
            // 【为什么不能 any()】实际读取的是
            // `snapshots[snapshot_index_for_rate(...)]`；只就绪另一份发布器时
            // `any()` 会误判"就绪"，随后在空发布器上逐样本写零 ⇒ 从有声阶跃到零。
            && self.processed_snapshot_ready()
    }

    /// 已发布的**原始磁带**是否对应当前活动 take。
    ///
    /// 【为什么要比 take 号】否则换 take 之后会继续播上一份原始磁带 —— 用户听到的是
    /// 上一次录制的内容，而界面上已经是新 take。
    fn raw_tape_ready(&self) -> bool {
        if !self.capture.has_material() {
            return false;
        }
        self.raw_tape_revision.load(Ordering::Acquire) == self.capture.revision()
            && self.raw_tape.is_ready()
    }

    /// 每个音频块的**观察与捕获**：只依赖输入总线与工程位置，**不依赖输出形状**。
    ///
    /// 【为什么必须与输出阶段分开 —— 这是"录制与输出全静音"的一个直接原因】
    /// 这两件事此前都长在 [`Self::capture_stage`] 里，而后者被 `process` 在
    /// `num_samples == 0 || num_outputs == 0` 的早退**之后**才调用。于是"宿主没接我们的
    /// 输出总线"这一个与录制**无关**的事实，会让插件：
    ///
    ///   * 永远捕获不到任何东西（录制静音）；
    ///   * `saw_context` 永不置位 ⇒ 界面上的录制键一直是灰的（连录都录不了）。
    ///
    /// 录制只关心两件事：宿主给的**工程位置**、宿主给的**输入**。
    ///
    /// 前置条件（由调用方保证）：`data` 的样本是 32-bit 浮点、`process_context` 非空。
    /// 非 32-bit 时平面是 64 位（联合体同地址），按 `f32` 读会读到垃圾。
    pub(crate) fn capture_observe(
        &self,
        data: *mut crate::audio_abi::ProcessData,
        context: &crate::audio_abi::ProcessContext,
    ) -> Option<BlockEvidence> {
        let capture = self.capture();
        // SAFETY: 调用方保证 data 在本回调期间可读。
        let mode = unsafe { (*data).process_mode };
        let stream = ProcessStream::from_mode(mode);
        // 【为什么离线也要记】它决定前端显示哪一条引导：走过 bounce 的宿主可以
        // "点一次 Capture 然后导出"，而实时宿主必须真的播一遍。
        if stream == ProcessStream::Offline {
            capture.note_offline();
        }
        // 【为什么采样率有兜底】`ProcessContext::sample_rate` 是 SDK 标注 always valid 的
        // 字段，但真出现过宿主不填它的情形；而**没有率就无法把工程帧换算成秒**，
        // 捕获会把每一块都当成"没有位置"丢掉（用户看到的是"录了 0.0 秒"）。
        // `setupProcessing` 声明过的率是同一件事的权威来源，用它兜底不引入新语义。
        let declared = self.setup_sample_rate();
        let rate = if context.sample_rate.is_finite() && context.sample_rate > 0.0 {
            context.sample_rate
        } else {
            declared
        };
        let playing = context.state & crate::audio_abi::PROCESS_CONTEXT_PLAYING != 0;
        let recording = context.state & crate::audio_abi::PROCESS_CONTEXT_RECORDING != 0;
        // SAFETY: 与 `audio_process_inner` 同一份宿主提供的 ProcessData；`capture_view`
        // 的切片只在本次回调内有效，而消费方同步读完就返回。
        let block = unsafe {
            crate::audio_abi::capture_view(data, context.project_time_samples, rate, playing)
        };
        // 【为什么未武装也要观察】观察位描述"宿主实际做了什么"，而录制按钮的门
        // （`canCapture`）读的正是 `saw_context`。此前观察发生在武装**之后**，于是
        // "必须先录一次才能录"——按钮永远是灰的，Capture 在所有宿主上都用不了。
        // 代价是几个原子写 + 一次缓冲形状检查（不分配、不读样本）。
        let playing_rising = capture.observe_with_stream(&block, stream);
        let was_recording = self.stream_recording[stream.index()].swap(recording, Ordering::AcqRel);
        let recording_rising = recording && !was_recording;
        // 【为什么要在相位判定之前记】`note_stream_geometry` 的"音频在流动"证据
        // 来自本块的指纹与上一块的比较；晚一步记录就永远读到上一块的结论。
        //
        // 【为什么零帧块不记】它是参数 flush，不代表时间轴在走。把它记成一次"内容
        // 变化"会让紧随其后的真实块被当成新素材（多出一块本不该有的位移）。
        if block.frames > 0 {
            self.streams[stream.index()].note_activity(block_activity_hash(&block));
            // 【T12：不再要求 `has_audio()`】宿主在 pre-roll 期间标 `silence_flags` /
            // 给空输入平面是**正常**的（`capture.rs` 的 `let silent = !block.has_audio();`
            // 就在用同一语义判定"这一块是静音"）。把它当成"没有证据"，等于告诉自己
            // "宿主标了静音 ⇒ 不是静音" —— 整段起播窗口的静音一次都记不下来，
            // `take_pre_roll` 永远是 0。空平面与全零平面对本判据是同一件事。
            //
            // 注意：`has_audio()` 的其它读点（"这条总线接没接"）保持原样，本次只改
            // 沉默累积这一个调用点。
            self.streams[stream.index()].note_block_silence(block_is_silent(&block));
        }
        // 【为什么位置是否推进要按流记】它决定"停播时要不要清输出"与捕获的
        // lookbehind。两条流的块会交替到达；跨流比较会把 prefetch 的移动当成
        // 实时流“在前进”的证据（用户听到的开头小段循环）。
        let stream_block = self.note_stream_block(
            stream,
            context,
            block.frames,
            playing_rising,
            recording_rising,
        );
        // `kPlaying` 上升沿意味着一次新的播放会话：允许同一区间再出音一次。
        if stream == ProcessStream::Realtime && playing_rising {
            self.rt_last_emitted_valid.store(false, Ordering::Release);
            // 【为什么帧数也要清】起播第一块与停播最后一块可能同起点、同长度；
            // 只清 `valid` 已经足够，但把帧数一并归零让"下一次比较"不依赖旧值。
            self.rt_last_emitted_frames.store(0, Ordering::Release);
            // 【T3：新的播放会话从零开始】读位置也必须清：上一轮停在任意位置，
            // 新会话第一块的位置天然与它不同 —— 那不是读游标跳变，是一次新的起播。
            self.rt_last_read_position.store(0, Ordering::Release);
        }
        let split = self.block_split(context, block.frames, rate, stream_block.position);
        let cycle_start = self.cycle_start_position(context, rate);
        let allow_output = self.allow_output_for(
            stream,
            &block,
            stream_block.position,
            stream_block.phase,
            playing_rising,
            recording_rising,
        );
        // 【为什么捕获不再需要外面的门】`on_block` 自己先读一次 `is_armed` 就早退，
        // 代价与这里再判一次相同；而少一层门就少一处"两个判据漂移"的机会。
        let mut outcome = crate::capture::CaptureBlockOutcome::Idle;
        for (index, part) in split.iter().enumerate() {
            let planes = [
                slice_plane(block.planes[0], part),
                slice_plane(block.planes[1], part),
            ];
            let sub = crate::capture::recorder::InputBlock {
                start: part.position,
                frames: part.frames,
                sample_rate: block.sample_rate,
                planes,
                playing,
            };
            let part_outcome = capture.on_block_with_context(
                sub,
                BlockContext {
                    stream,
                    playing,
                    recording,
                    phase: stream_block.phase,
                    playing_rising,
                    recording_rising,
                    cycle_wrap: stream_block.cycle_wrap,
                    cycle_start,
                    new_pass: stream_block.new_pass,
                    part: index.min(u8::MAX as usize) as u8,
                    advancing: stream_block.activity_moving,
                    resync: stream_block.resync,
                    resync_limit_frames: stream_block.resync_limit_frames,
                    // 循环切分的第二段不是起播起点：前导静音/钉住纠正只属于第一段。
                    preceding_silent_frames: if index == 0 {
                        stream_block.preceding_silent_frames
                    } else {
                        0
                    },
                    counted: index == 0 && stream_block.counted,
                    position_source: stream_block.position_source,
                    discard_pre_roll: index == 0 && stream_block.discard_pre_roll,
                    // 记账域位置：起播窗口里宿主位置被钉住，`part.position` 已经是插件
                    // 按块长推进后的值，与宿主管道域不同（见 `BlockContext` 的说明）。
                },
            );
            outcome = if index == 0 {
                part_outcome
            } else {
                combine_capture_outcomes(outcome, part_outcome)
            };
        }
        self.note_block_trace(
            context.state,
            mode,
            stream,
            stream_block.raw_start,
            stream_block.position,
            block.frames,
            playing,
            stream_block.phase,
            playing_rising,
            stream_block.cycle_wrap,
            split.count,
            stream_block.had_previous,
            allow_output,
            outcome.code(),
            capture.capture_started(),
            StreamTime::from_context(context),
            rate,
            stream_block.counted,
        );
        Some(BlockEvidence {
            stream,
            position: stream_block.position,
            frames: block.frames,
            playing,
            recording,
            playing_rising,
            recording_rising,
            phase: stream_block.phase,
            had_previous: stream_block.had_previous,
            allow_output,
            start_window: stream_block.start_window,
            repeated_nonzero: stream_block.repeated_nonzero,
            split,
        })
    }

    /// 循环起点在工程采样域中的位置（`None` = 元数据不足）。
    ///
    /// 只用于把同一圈的多个段标记成同一组；不参与音频定位（音频定位由
    /// `block_split` 的逐段位置负责）。
    fn cycle_start_position(
        &self,
        context: &crate::audio_abi::ProcessContext,
        rate: f64,
    ) -> Option<i64> {
        let time = StreamTime::from_context(context);
        if !time.cycle_active || !time.cycle_valid {
            return None;
        }
        let (Some(tempo), Some(music)) = (time.tempo, time.music) else {
            return None;
        };
        let beats = music - time.cycle_start_music;
        let frames = (beats * 60.0 / tempo * rate).round() as i64;
        Some(context.project_time_samples.saturating_sub(frames).max(0))
    }

    /// 计算循环边界切分。只有在 cycle 元数据完整、且本块确实横跨 loop 终点时才切。
    ///
    /// 【为什么用音乐时间算】`ProcessContext` 不提供“循环终点对应的采样位置”，但
    /// 给了 `projectTimeMusic` / `cycleEndMusic` / `tempo`。三者有效时，
    /// `(cycle_end − project_time_music) × 60/tempo × rate` 就是本块到终点的帧数。
    fn block_split(
        &self,
        context: &crate::audio_abi::ProcessContext,
        frames: usize,
        rate: f64,
        position: i64,
    ) -> BlockSplit {
        let time = StreamTime::from_context(context);
        let (Some(tempo), Some(music)) = (time.tempo, time.music) else {
            return BlockSplit::whole(position, frames);
        };
        if !time.cycle_active || !time.cycle_valid {
            return BlockSplit::whole(position, frames);
        }
        let loop_beats = time.cycle_end_music - time.cycle_start_music;
        // 【为什么写成 `<= 0.0` 而不是 `!(x > 0.0)`】两者在有限数上等价，但否定的
        // 比较让"这里到底要不要 NaN"变得要读者自己想。这里要的就是"循环长度必须是正的"
        // 这一个陈述。
        if loop_beats <= 0.0 {
            return BlockSplit::whole(position, frames);
        }
        let beats_to_end = time.cycle_end_music - music;
        let to_end = (beats_to_end * 60.0 / tempo * rate).round();
        // 同上：直接陈述"到循环尾的距离必须是正的、且必须在本块之内"。
        if to_end <= 0.0 || to_end >= frames as f64 {
            return BlockSplit::whole(position, frames);
        }
        let length = (loop_beats * 60.0 / tempo * rate).round() as i64;
        if length <= 0 {
            return BlockSplit::whole(position, frames);
        }
        let to_end = to_end as usize;
        let wrapped = context
            .project_time_samples
            .saturating_add(to_end as i64)
            .saturating_sub(length)
            .max(0);
        self.cycle_splits.fetch_add(1, Ordering::Relaxed);
        BlockSplit::split(
            SubBlock {
                position: context.project_time_samples,
                input_offset: 0,
                frames: to_end,
            },
            SubBlock {
                position: wrapped,
                input_offset: to_end,
                frames: frames - to_end,
            },
        )
    }

    /// 本块是否允许写出内容。
    ///
    /// 【真值表】离线/prefetch：永远允许（它们的输出稍后才被播放，此刻的传输状态
    /// 与内容正确性无关）。实时：`kPlaying`、本流推进、或“这个区间还没出过音”
    /// 允许；同一区间在停播光标处重复到达则抑制（避免小段循环/嗡嗡声）。
    fn allow_output_for(
        &self,
        stream: ProcessStream,
        block: &crate::capture::recorder::InputBlock<'_>,
        position: i64,
        phase: TransportPhase,
        playing_rising: bool,
        recording_rising: bool,
    ) -> bool {
        // 【为什么 bound 实例不参与这里的判据】它的"停播即静音"由
        // `capture_should_emit` 直接读 `kPlaying` 决定（见那里的说明：预取/offline
        // 恒真、实时流看播放位）。若这里也参与，`evidence.allow_output` 会在"停播
        // 光标处的重复块"上翻假，把 ARA 的既有语义改掉。
        if self.is_bound() || self.bound_once.load(Ordering::Acquire) {
            return true;
        }
        match stream {
            ProcessStream::Offline => true,
            // prefetch 的输出稍后才被播放；重复请求可能是参数变化后的重算，
            // 抑制会留下旧结果。
            ProcessStream::Prefetch => true,
            ProcessStream::Realtime => {
                if playing_rising || recording_rising {
                    self.gate_resting_run.store(0, Ordering::Release);
                    return true;
                }
                let same = self.rt_last_emitted_valid.load(Ordering::Acquire)
                    && self.rt_last_emitted_start.load(Ordering::Acquire) == position
                    && self.rt_last_emitted_frames.load(Ordering::Acquire)
                        == u32::try_from(block.frames).unwrap_or(u32::MAX);
                let quiet = phase == TransportPhase::Resting || same;
                if !quiet {
                    self.gate_resting_run.store(0, Ordering::Release);
                    return true;
                }
                // 【T20：单块抖动不静音】见 `gate_resting_run` 的字段说明。
                //
                // 【为什么不用 `kPlaying` 当分界】试过，会把 FL Studio 的回归打破：
                // 那边停播时 `kPlaying` 常置 1，而同一位置的重复块**必须**被淡出，
                // 否则固定光标处同一小段会被反复重写（用户听到嗡嗡声）。因此判据
                // 只能是"连续多少块不推进"——`a_repeated_realtime_block_is_silenced`
                // 与 `a_pinned_block_with_playing_one_is_silenced` 两条用例把这个
                // 契约钉在两侧：单块抖动必须出声，连续不推进必须淡出。
                let run = self.gate_resting_run.fetch_add(1, Ordering::AcqRel) + 1;
                if run < GATE_SETTLE_BLOCKS {
                    return true;
                }
                self.repeats_suppressed.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// 记录“本块确实写出了内容”（由 [`Self::capture_stage`] 在写入分支调用）。
    fn note_emitted(&self, evidence: Option<BlockEvidence>) {
        let Some(evidence) = evidence else {
            return;
        };
        if evidence.stream != ProcessStream::Realtime
            || self.is_bound()
            || self.bound_once.load(Ordering::Acquire)
        {
            return;
        }
        // 【T3：读位置单独记一份】输出侧按 `position` 读快照；跳变就是读游标阶跃。
        // 【为什么连帧数一起记】下一块的"预期续接"是 `位置 + 帧数`；只记位置会把
        // 每一次正常推进都算成跳变（见 `read_position_jump`）。
        self.rt_last_read_position
            .store(evidence.position, Ordering::Release);
        self.rt_last_read_frames.store(
            u32::try_from(evidence.frames).unwrap_or(u32::MAX),
            Ordering::Release,
        );
        self.rt_last_read_pass.store(
            self.streams[ProcessStream::Realtime.index()]
                .pass
                .load(Ordering::Acquire),
            Ordering::Release,
        );
        self.rt_last_emitted_start
            .store(evidence.position, Ordering::Release);
        self.rt_last_emitted_frames.store(
            u32::try_from(evidence.frames).unwrap_or(u32::MAX),
            Ordering::Release,
        );
        self.rt_last_emitted_valid.store(true, Ordering::Release);
    }

    /// 测试专用：直接写"上一块读到哪个位置"（T3 判据的唯一输入）。
    #[cfg(test)]
    pub(crate) fn note_read_position_for_test(&self, position: i64) {
        self.rt_last_read_position.store(position, Ordering::Release);
    }

    /// 测试专用：写"上一块读了多少帧"（预期续接的第二个输入）。
    #[cfg(test)]
    pub(crate) fn note_read_frames_for_test(&self, frames: u32) {
        self.rt_last_read_frames.store(frames, Ordering::Release);
    }
    /// 测试专用：标记"上一块写出过内容"（T3 判据的第二个输入）。
    #[cfg(test)]
    pub(crate) fn mark_emitted_for_test(&self) {
        self.rt_last_emitted_valid.store(true, Ordering::Release);
    }
    /// 【T3】这一块的读位置与上一块差了多少帧（`None` = 不存在跳变）。
    ///
    /// 四条纪律（都是纯函数，可单测）：
    /// 1. **没有上一块就不存在跳变**：`rt_last_read_position <= 0` 或上一块没写出过
    ///    内容 ⇒ `None`。否则会话刚开始的两次不同位置会被当成跳变。
    /// 2. **跨 pass 不是跳变**：那是"重新开始"，由起播接缝处理。
    /// 3. **位置相同不是跳变**。
    /// 4. **小于一个淡化带的差被淡化吸收**：那种场合读快照的那一小段与上一块只差几帧，
    ///    64 帧的交叉淡化足够；再淡一次反而多一次能量塌陷。
    fn read_position_jump(&self, evidence: Option<BlockEvidence>) -> Option<u64> {
        let evidence = evidence?;
        let previous = self.rt_last_read_position.load(Ordering::Acquire);
        let previous_frames = i64::from(self.rt_last_read_frames.load(Ordering::Acquire));
        if previous <= 0 || !self.rt_last_emitted_valid.load(Ordering::Acquire) {
            return None;
        }
        // 【跨 pass 不是跳变】pass 边界是"重新开始"，不是"读游标跳了"。上一块输出的
        // 尾巴属于另一个位置的内容，拿它做交叉淡化只会叠出两段不相干信号。
        // 起播的接缝由 `Seam::FirstBlock` / `Seam::FadeIn` 在种类变化时处理。
        let pass = self.streams[ProcessStream::Realtime.index()]
            .pass
            .load(Ordering::Acquire);
        if pass != self.rt_last_read_pass.load(Ordering::Acquire) {
            return None;
        }
        // 【核心判据：预期续接，而不是"上一块起点"】上一块读了 `previous_frames` 帧，
        // 那么本块的起点**应当**是 `previous + previous_frames`。旧实现直接拿本块起点
        // 减上一块起点，于是任何连续播放（差值 = 块长 > 64）都被判成跳变 ⇒ 整块直通
        // 宿主原声 ⇒ 每一块都在干声与合成结果之间硬接一次。那就是"咔"。
        let expected = previous.saturating_add(previous_frames);
        let jump = evidence.position.saturating_sub(expected).unsigned_abs();
        (jump > u64::try_from(OUTPUT_DECLICK_FRAMES).unwrap_or(u64::MAX)).then_some(jump)
    }
    /// 音频线程上的 Capture **输出**阶段。
    ///
    /// 【为什么未录制时要早退而不是统一处理】本函数在**每个音频块**上被调用。
    /// 未录制时它只读几个原子就返回，代价与"没写这段代码"相同 —— 这是它能被放在
    /// 所有既有逻辑之前的前提。
    ///
    /// 【为什么录制中不走快照输出】正在录的那一段，插件手上还没有对应的素材；
    /// 输出上一次的快照会让用户听到**旧内容盖在新演奏上**。所以录制中按选项输出
    /// 直通（默认，让用户听到宿主原声）或静音。
    ///
    /// 【为什么捕获不在这里】见 [`Self::capture_observe`]：它必须发生在任何输出形状
    /// 判定之前，否则"宿主没接输出"会让录制一起失效。
    pub(crate) fn capture_stage(
        &self,
        data: *mut crate::audio_abi::ProcessData,
        evidence: Option<BlockEvidence>,
    ) -> CaptureStage {
        let capture = self.capture();
        // SAFETY: 调用方保证 data 在本回调期间可读。
        let offline = unsafe { (*data).process_mode } == 2;
        // 选项是无锁快照（见 `CaptureStore::options`）。
        let options = capture.options();
        // 去咔状态按进程流隔离；块长取宿主本回调声明的帧数。
        let stream = evidence.map_or(ProcessStream::Realtime, |evidence| evidence.stream);
        let frames = unsafe { (*data).num_samples as usize };
        // 【为什么先仲裁、再做防御性清空】录制中的输出是**宿主正在送进来的原声**
        // （用户监听自己），它不该被"停播重复抑制"影响 —— 那个判据描述的是"同一段
        // 内容又来了"，而录制中来的每一块都是要录的新内容。此前防御清空排在仲裁之前，
        // 于是一次误判就能把监听整块抹掉。
        let output = self.capture_output_with(evidence);
        // 【T3：读游标跳变 ⇒ 这一块做一次去咔】种类与代次都没变，但这一块读的位置与
        // 上一块不同（pass 边界、宿主纠正记账、流交接）。那时读快照等于把**另一段**
        // 内容写进总线 —— 一次单样本阶跃。判据在 `read_position_jump`，纯函数、可单测。
        //
        // 【为什么不再"整块直通"】直通本身就是**换了一个信号源**：一个块最长可达 6 s，
        // 把它整块换成宿主原声，前后各是一次硬切换 —— 用户听到的那一下"咔"其实是这个
        // 保护动作自己（代码自己的注释早就这么写过）。现在位置只有一个权威
        // （`note_stream_geometry` 的 R1–R4），跳变只可能发生在 pass 边界，而 pass
        // 边界本来就要重新开始，一次交叉淡化就够。
        if let Some(jump) = self.read_position_jump(evidence) {
            self.position_corrections.fetch_add(1, Ordering::Relaxed);
            self.max_correction_frames.fetch_max(jump, Ordering::Relaxed);
            self.force_declick[stream.index()].store(true, Ordering::Release);
        }
        self.output_token[stream.index()]
            .store(self.output_token_for(output), Ordering::Release);
        self.note_output_rung(output);
        // 【防御】门控已经在 `vst3.rs` 里判过一次；这里再判一次实时流，避免未来
        // 改动绕过门控时把停播重复块写出去。ARA 实例保持原语义（不参与）。
        if output != CaptureOutput::Recording {
            if let Some(evidence) = evidence {
                if evidence.stream == ProcessStream::Realtime
                    && !evidence.allow_output
                    && !self.is_bound()
                    && !self.bound_once.load(Ordering::Acquire)
                {
                    // 从上一块尾巴淡到零。本函数写满整块，**不能**再调 `clear_outputs`
                    // —— 那会把刚写好的淡出一起抹掉（见其文档）。
                    unsafe { self.apply_gate_fade_out(stream, &mut *(*data).outputs, frames) };
                    return CaptureStage::Handled;
                }
            }
        }
        match output {
            // 不介入：与改动前的 `return false` 完全等价，继续走既有快照通路。
            //
            // 【为什么快照没就绪时要 layered】这是 ARA 实例（`is_bound()` / `bound_once`）
            // 唯一会走到的输出分支。快照被 `clear_renderers()` / `stop_editor()` /
            // `document_closed()` 清空后，`copy_block_at` 的读闭包逐样本返回 `0.0`，
            // 而 `layered == false` 时 `audio_process_inner` 既不预清也不兜底 ——
            // 于是每个音频块都是纯零，用户听到的就是"有概率完全没声音"。
            // layered 让 `copy_block_layered_at` 在快照缺失处回退取宿主输入
            // （`snapshot.rs` 的 `input_at`），至少保住宿主原声而不是静音。
            CaptureOutput::Untouched => {
                let ready = self.processed_snapshot_ready();
                self.note_bound_snapshot_fallback(!ready);
                CaptureStage::Snapshot {
                    source: SnapshotSource::Processed,
                    layered: !ready,
                }
            }
            CaptureOutput::Recording => {
                // 【为什么离线那一趟不监听】离线渲染（bounce / 导出）若只有一趟，直通会把
                // **输入原声**写进产物 —— 而用户会把它当成"处理后的结果"，那是最难发现的一类
                // 错误。静音是诚实的：这一趟的唯一作用是把素材拿进来；第二趟（解除武装后再
                // 导出）才是结果。实时播放则照旧监听，因为用户要听到自己正在录的东西。
                if options.monitor && !offline {
                    // SAFETY: 与上面同一份数据；`pass_through` 自己会校验形状。
                    let _ = unsafe { crate::audio_abi::pass_through(data) };
                    self.note_emitted(evidence);
                } else {
                    // 【为什么"不监听"必须显式清空】本分支**整块都不写**，而调用方拿到
                    // `Handled` 就立刻返回（见 `audio_process_inner`）—— 于是输出缓冲里
                    // 留着上一次的内容：in-place 宿主下那是**输入原声**（关掉监听反而
                    // 听得见），非 in-place 宿主下是未定义字节。本文件头那条纪律正是
                    // "只有本通路确实不写整块时才清"，这里就是那种情形。
                    // SAFETY: 与上面同一份数据；`clear_outputs` 自己会校验形状。
                    let _ = unsafe { crate::audio_abi::clear_outputs(data) };
                }
                unsafe { self.apply_output_declick(stream, &mut *(*data).outputs, frames) };
                CaptureStage::Handled
            }
            // 【为什么区间外要直通】只录了 10 秒的素材不该把整条轨变成静音：快照覆盖的
            // 那段被替换成处理结果，其余保持宿主原声。见 `copy_block_layered`。
            CaptureOutput::Processed => {
                self.note_emitted(evidence);
                CaptureStage::Snapshot {
                    source: SnapshotSource::Processed,
                    layered: options.outside_passthrough,
                }
            }
            CaptureOutput::Raw => {
                self.note_emitted(evidence);
                CaptureStage::Snapshot {
                    source: SnapshotSource::Raw,
                    layered: options.outside_passthrough,
                }
            }
            // 【为什么与 `PassThrough` 输出相同】素材在，但还没有任何可播的版本；
            // 唯一诚实的输出仍是宿主原声（静音会让整条轨"哑掉"）。区别只在**读数**：
            // 这一档让界面能说"素材已就绪，等待合成"，从而把"合成没跑"暴露出来。
            // `capture_output` 已置好 `render_wanted`，actor 会补排一次合成。
            CaptureOutput::PendingSynthesis | CaptureOutput::PassThrough => {
                // SAFETY: 与上面同一份数据；`pass_through` 自己会校验形状。
                let _ = unsafe { crate::audio_abi::pass_through(data) };
                self.note_emitted(evidence);
                unsafe { self.apply_output_declick(stream, &mut *(*data).outputs, frames) };
                CaptureStage::Handled
            }
        }
    }

    /// 当前生效的编辑后端。
    ///
    /// 【为什么不能只返回字段里的那份】字段在 `initialize` 期间写入，而那时 ARA 绑定
    /// 窗口还开着（见 `HostBackend::from_native` 的说明），所以字段里那份在
    /// "宿主会绑定 ARA"的场合是 `AraOnly`、在"宿主不会绑定"的场合**也是** `AraOnly` ——
    /// 它区分不了这两种形态。形态由 [`Self::mode`] 在窗口关闭后给出，这里据此把
    /// "无原生扩展且无 ARA" 解析成 [`crate::host::adapter::HostBackend::Capture`]。
    ///
    /// 【为什么原生扩展优先】REAPER 既有原生扩展**又**会绑定 ARA；只要字段里有
    /// `Native`，形态问题就不存在（它一定走 ARA），直接返回它。
    pub(crate) fn host_backend(&self) -> crate::host::adapter::HostBackend {
        let stored = self.host.lock_or_recover().clone();
        if let Some(backend) = stored {
            if backend.native().is_some() {
                return backend;
            }
        }
        match self.mode() {
            InstanceMode::Capture => crate::host::adapter::HostBackend::Capture(
                crate::host::capture::CaptureHost::new(),
            ),
            // 【为什么 `Undetermined` 也要看素材】形态要等 `setActive` 之后 5 秒才落定，
            // 而"宿主把本插件当普通插入"的会话里用户可能从不打开窗口 —— 那时形态永远
            // 停在 `Undetermined`，按 ARA-only 准备会让 Capture 的渲染准入失败。
            // 素材存在本身就是 Capture 的证明（见 `renders`）。
            InstanceMode::Undetermined if self.capture.has_material() => {
                crate::host::adapter::HostBackend::Capture(
                    crate::host::capture::CaptureHost::new(),
                )
            }
            // 剩下的 `Undetermined` 是"还没确定"（此时按 ARA 准备是安全的，ARA 是更丰富的
            // 形态），`Ara` 本来就是这个后端。
            InstanceMode::Undetermined | InstanceMode::Ara => {
                crate::host::adapter::HostBackend::from_native(None)
            }
        }
    }
    /// 原生宿主适配器（未绑定时 `None`）。
    ///
    /// 【为什么单独一个访问器】它是通用层唯一允许取具体 `NativeHost` 的地方 —— 所有
    /// 需要宿主扩展的操作都从这里取、并显式处理 `None`，让"这一步依赖原生扩展"写在
    /// 调用点上，而不是藏在字段类型里。`NativeHost` 自身不带品牌：它只是一组宿主扩展
    /// API 的适配器，当前实现是 REAPER，但通用层从不命名具体宿主。
    pub(crate) fn native_host(&self) -> Option<Arc<crate::host::NativeHost>> {
        self.host
            .lock_or_recover()
            .clone()
            .and_then(|backend| backend.native().cloned())
    }
    /// **生效的**宿主能力位：唯一来源，前端只读这一份。
    ///
    /// 【为什么不能直接用 `host_backend().capabilities()`】后者按 `self.host`（本实例自己
    /// 的原生扩展）计算，而写能力实际由**本文档里任何**一个持有 project history 的实例
    /// 提供（`project_history_host` 的解析口径）。此前前端另有一组 `BootstrapFlags` 布尔，
    /// 用 `project_history_host().can_*()` 各算一遍 —— 两份判据必然漂移，而且"到底哪一份
    /// 算数"没人说得清。这里把同一口径收敛成一个值，布尔与能力位不再各说各话。
    ///
    /// 【为什么 `clip_write` 与 `undo_stack` 分开报】`has_project_history()` 等于
    /// "有撤销栈 **且** 可写片段"，前端 `canEditHostClips()` 用 `clipWrite && undoStack`
    /// 即可精确复现"没有历史 API 就不给拖拽"这条既有语义，而不是把它折进 `clip_write`。
    pub(crate) fn effective_host_capabilities(
        &self,
    ) -> hifishifter_kernel::editor::host::HostCapabilities {
        let mut caps = self.host_backend().capabilities();
        if let Some(host) = self.project_history_host() {
            caps.clip_write = host.can_edit_clips();
            caps.split = host.can_split_clips();
            caps.item_clipboard = host.can_clipboard_items();
            caps.media_import = host.can_import_audio();
            caps.track_create = host.can_create_audio_track();
            caps.undo_stack = host.has_project_history();
        }
        caps
    }
    /// 撤销历史的**权威**：宿主提供撤销栈时用宿主的，否则用插件自管的那份。
    ///
    /// 【为什么永远返回 `Some`】把"没有宿主历史"变成一个实现（`LocalHistory`）而不是
    /// 一条被跳过的分支，是 `undo_timeline` / `redo_timeline` / `set_history_position`
    /// 在非 REAPER 宿主上不再静默失效的关键。只有在文档/会话都还没建起来时才退到
    /// `NullHistory`（诚实的空栈），仍然不是 `None`。
    /// 撤销历史的承载者分类（`"host"` / `"local"`），**不创建会话**。
    ///
    /// 【为什么与 `history_authority` 分开】bootstrap/健康探活只需要一个字符串；
    /// `history_authority` 在没有宿主历史时会创建编辑器会话（见其实现），
    /// 而探活必须能在会话尚未创建时回答。
    pub(crate) fn history_backend(&self) -> &'static str {
        if self.project_history_host().is_some() {
            "host"
        } else {
            "local"
        }
    }

    pub(crate) fn history_authority(
        self: &Arc<Self>,
    ) -> Arc<dyn crate::host::history::HostHistory> {
        if let Some(host) = self.project_history_host() {
            return Arc::new(crate::host::history::ProjectHistory { host });
        }
        match self.editor_session() {
            Ok(editor) => Arc::new(crate::host::history::LocalHistory {
                editor: Arc::downgrade(&editor),
            }),
            Err(_) => Arc::new(crate::host::history::NullHistory),
        }
    }
    /// 明确指定轨道须已有真实assigned item；空轨未指定时仅允许本FX的直接parent轨道。
    pub(crate) fn audio_import_target(
        &self,
        track: Option<&str>,
        authorized: &impl Fn() -> bool,
    ) -> Result<crate::host::HostTrackTarget, String> {
        if let Some(id) = track {
            let document = self.editor_document()?;
            let target = {
                document
                    .ui_tracks
                    .lock_or_recover()
                    .values()
                    .find(|t| t.id == id)
                    .map(|t| t.target.clone())
            };
            if let Some(target) = target {
                return Ok(target);
            }
        }
        if let Some(track) = track {
            let document = self.editor_document()?;
            let clips = {
                let timeline = document.timeline.lock_or_recover();
                timeline
                    .as_ref()
                    .ok_or("host timeline unavailable")?
                    .clips
                    .iter()
                    .filter(|clip| clip.track_id == track)
                    .map(|clip| clip.id.clone())
                    .collect::<Vec<_>>()
            };
            for clip in clips {
                if let Ok(target) = self.host_edit_target(&clip) {
                    return target.import_target(authorized);
                }
            }
            return Err("import track has no directly bound host region".into());
        }
        let host = self.native_host().ok_or("primary host interface missing")?;
        host.direct_track_target(authorized)
    }
    /// 同文档实际renderer唯一region边解析写对象，不能按名字/位置/源路径查item。
    pub(crate) fn host_edit_target(
        &self,
        clip_id: &str,
    ) -> Result<crate::host::HostClipTarget, String> {
        self.refresh_ui_inventory();
        let document = self.editor_document()?;
        if let Some(item) = clip_id.strip_prefix("ara-item-") {
            let target = document
                .ui_tracks
                .lock_or_recover()
                .values()
                .flat_map(|track| &track.items)
                .find(|entry| entry.geometry.item_id == item)
                .map(|entry| entry.target.clone());
            if let Some(target) = target {
                return target.current(&|| {
                    !self.is_closed()
                        && document.is_alive()
                        && self
                            .editor_document()
                            .is_ok_and(|doc| Arc::ptr_eq(&doc, &document))
                });
            }
        }
        let key = {
            let ids = document.clip_ids.lock_or_recover();
            ids.iter()
                .find(|(_, id)| id.as_str() == clip_id)
                .map(|(key, _)| *key)
                .ok_or("unknown ARA clip identity")?
        };
        let mut target = None;
        for owner in document.renderer_owners() {
            if !owner.renders_playback()
                || !owner
                    .assigned_regions()
                    .is_ok_and(|keys| keys.as_slice() == [key])
            {
                continue;
            }
            let stamp = owner.host_query_stamp()?;
            let bound = owner.host_geometry()?;
            if bound.region_key != key {
                return Err("host region binding changed".into());
            }
            let host = owner.native_host().ok_or("host interface missing")?;
            let next = host.clip_target(|| owner.host_query_authorized(&stamp))?;
            if target.is_some() {
                return Err("multiple host writers for one ARA clip".into());
            }
            target = Some(next);
        }
        target.ok_or_else(|| "clip has no unique directly bound host take".into())
    }
    /// 组件终止时停止实例 worker 并释放宿主适配器；不要在音频 process 调用。
    ///
    /// 【为什么与文档关闭分开】`stop_editor()` 同时服务两种语义：文档结束
    /// （组件还活着，下一份文档还会来）与组件结束（宿主适配器确实该释放）。
    /// 合并在一起会在一份文档关闭时把 REAPER 原生扩展适配器永久丢掉 ——
    /// 宿主复用处理器实例重开工程时，重绑后 `native_host()` 永远是 `None`，
    /// 于是不枚举 item、不填 `region_items`、界面停在"等待宿主音频"。
    /// 文档关闭必须走 [`Self::stop_editor_for_document`]。
    pub(crate) fn stop_editor(&self) {
        self.stop_editor_inner(false);
    }

    /// 文档结束：停止 worker、清快照与 per-document 缓存，**保留**宿主适配器。
    pub(crate) fn stop_editor_for_document(&self) {
        self.stop_editor_inner(true);
    }

    fn stop_editor_inner(&self, keep_host: bool) {
        let document = {
            self.document
                .lock_or_recover()
                .as_ref()
                .and_then(std::sync::Weak::upgrade)
        };
        if let Some(document) = document {
            {
                let _transaction = document.transaction.lock_or_recover();
                if !self.closed.swap(true, Ordering::AcqRel) {
                    document.scope_revision.fetch_add(1, Ordering::AcqRel);
                }
                self.snapshots.iter().for_each(|snapshot| snapshot.clear());
            }
            // 不持文档事务取得views锁，避免enqueue/worker授权反向等待。
            document.revoke_editor_views();
        } else {
            self.closed.store(true, Ordering::Release);
            self.snapshots.iter().for_each(|snapshot| snapshot.clear());
        }
        if !keep_host {
            let host = self.host.lock_or_recover().take();
            drop(host);
        }
        self.host_geometry.lock_or_recover().take();
        *self.host_audio.lock_or_recover() = HostAudioStatus::default();
        self.binding_report.lock_or_recover().take();
        *self.last_enumeration.lock_or_recover() = None;
        self.cancel_preparation();
        // 【为什么 take 掉而不是留在槽位里】`close()` 是终结性的；留下一个
        // `closed == true` 的队列表只会让下一次重绑再次踩到同一个死队列。
        // 重绑时 `ensure_preparation()` 会建一条新的。
        if let Some(worker) = self.preparation.lock_or_recover().take() {
            worker.close();
        }
    }

    /// 把本实例的**每份文档**缓存清到"尚未绑定"状态（宿主切换文档时用）。
    ///
    /// 【为什么不能只清 `assignments / sequences / document`】这些字段的名字都带
    /// "当前文档"的语义，但它们的读者分布在音频线程、UI 定时器与诊断包里；
    /// 留下旧文档的恢复状态、几何缓存、宿主音频读数或快照，会让新文档的第一帧
    /// 输出/第一份 UI 载荷读到上一份工程的结论。宿主适配器（`host`）**不在**
    /// 这里清 —— 它属于组件，不属于文档。
    fn prepare_rebind_from_previous_document(&self) {
        self.snapshots.iter().for_each(|snapshot| snapshot.clear());
        self.raw_tape.clear();
        *self.prepared.lock_or_recover() = None;
        self.reset_restore_report();
        self.binding_report.lock_or_recover().take();
        *self.host_audio.lock_or_recover() = HostAudioStatus::default();
        *self.last_enumeration.lock_or_recover() = None;
        *self.last_failed_revision.lock_or_recover() = None;
        self.enumeration_backoff.store(0, Ordering::Release);
        self.host_item_muted.store(false, Ordering::Release);
        self.playback_covered.store(false, Ordering::Release);
        self.rt_last_emitted_valid.store(false, Ordering::Release);
        self.rt_last_emitted_frames.store(0, Ordering::Release);
        self.trace_wanted.store(false, Ordering::Release);
        self.trace_logged.store(false, Ordering::Release);
        self.render_wanted.store(false, Ordering::Release);
        // 【为什么不在这里清 `pending_restore`】组件 `setState` 可能先于 ARA bind
        // 到达（宿主在 setActive 之前恢复组件状态）。那份载荷属于**即将到来的**
        // 新文档，清掉等于把用户参数丢掉。旧文档遗留的载荷由下一次成功合并覆盖。
        if self.pending_restore.lock_or_recover().is_some() {
            self.note_restore_waiting("waiting for the new document graph");
        }
    }
    /// 模型撤销时立即取消待准备作业；保持worker可供后续重新授权使用。
    pub(crate) fn cancel_preparation(&self) {
        if let Some(worker) = self.preparation() {
            worker.cancel();
        }
    }
    /// 原GUI合并后台宿主准备状态，不能把冷恢复尚未准备完写成已应用。
    pub(crate) fn preparation_state(&self) -> (bool, Option<String>) {
        if self.is_editor_only() {
            if let Some(document) = self
                .document
                .lock_or_recover()
                .as_ref()
                .and_then(std::sync::Weak::upgrade)
            {
                let states = document
                    .renderer_owners()
                    .into_iter()
                    .filter(|owner| owner.renders_playback())
                    .map(|owner| owner.local_preparation_state())
                    .collect::<Vec<_>>();
                return (
                    states.iter().any(|state| state.0),
                    states.into_iter().find_map(|state| state.1),
                );
            }
        }
        self.local_preparation_state()
    }
    pub(crate) fn local_preparation_state(&self) -> (bool, Option<String>) {
        match self.preparation() {
            Some(worker) => worker.state(),
            None => (false, None),
        }
    }
    /// 渲染器的准备状态快照（诊断包用）：模型/编辑代次、作用域与区间数。
    ///
    /// 【为什么单独抽出来】`Snapshot` 里那段诊断需要当前分配区间才能算逐输出读数；
    /// 诊断包在任意时刻被导出，不值得为它触发一次完整快照。这里只给"准备好了没有、
    /// 准备的是哪一版"，那正是排查"渲染结果陈旧"时要看的东西。
    pub(crate) fn prepared_snapshot(&self) -> Option<serde_json::Value> {
        self.prepared.lock_or_recover().as_ref().map(|version| {
            serde_json::json!({
                "model": version.model,
                "edit": version.edit,
                "epoch": version.epoch,
                "scope": version.scope,
                "regions": version.keys.len(),
            })
        })
    }
    /// native主线程取得宿主给本ARA文档的可撤销播放租约；不寻找全局REAPER窗口。
    pub(crate) fn host_playback(&self) -> Option<ara2_bridge::plugin::PlaybackRequestHandle> {
        self.document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .and_then(|document| document.playback.lock_or_recover().clone())
    }
    /// 组件释放/文档关闭前结束后台服务，避免DLL卸载后线程仍执行插件代码。
    /// 未绑定时暂存组件state；绑定后所有处理器读取同一文档的参数权威。
    // 组件state访问器保留，供未绑定路径读取。
    #[cfg(test)]
    pub fn edit_state(&self) -> Arc<Mutex<crate::state_channel::EditState>> {
        self.document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .map(|document| document.edits.clone())
            .unwrap_or_else(|| self.edits.clone())
    }

    /// 恢复/undo组件状态后刷新整张文档，各renderer保持原分配。
    ///
    /// 【为什么拆成两个】这个方法会在 `transaction` 之外被调用（`restore_state`），
    /// 也会在**已经持有** `transaction` 的地方被调用（`DocumentSession::prepare_renderers`
    /// 的存档合入路径）。`std::sync::Mutex` 不可重入，两条路径共用一个"自己取锁"的
    /// 实现就会自锁 —— 用户报的"重开 ARA 工程宿主未响应"正是那一次。
    ///
    /// 现在的契约：本方法是**无锁入口**，自己取 `transaction`；
    /// [`Self::refresh_document_locked`] 要求调用方已经持有它。任何未来调用点必须
    /// 显式二选一，不存在"隐含取锁"的版本。
    pub fn refresh_document(&self) {
        let document = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade);
        if let Some(document) = document {
            let _transaction = document.transaction.lock_or_recover();
            self.refresh_document_locked(&document);
        }
    }

    /// [`Self::refresh_document`] 的持锁版本：**调用方必须已持有** `document.transaction`。
    ///
    /// 【为什么参数是 `&DocumentSession` 而不是自己 upgrade】持锁调用方本来就有那个
    /// 引用（`prepare_renderers` 就是 `self`），传进来既省一次 Weak 升级，也让
    /// "哪份文档的锁被持有"在签名上是唯一的。
    pub(crate) fn refresh_document_locked(&self, document: &super::document::DocumentSession) {
        let owners = document.renderer_owners();
        for owner in &owners {
            if let Err(error) = owner.merge_pending_restore(document) {
                log::warn!("[ara] instance state unresolved: {error}");
            }
        }
        for owner in owners {
            owner.prepare();
        }
    }

    /// setState只暂存本组件恢复；宿主明确分配区域且模型ready后再合入共享权威。
    pub(crate) fn restore_state(&self, bytes: &[u8]) -> Result<(), String> {
        if bytes.is_empty() {
            return Ok(());
        }
        // 容器解析对**历史格式**（裸 `EditState`）是透明的：没有魔数就整块按旧格式读。
        // 读不出来的后果是把用户已有的编辑弄丢，所以这条兼容不是可选项。
        let state = crate::component_state::ComponentState::decode(bytes)?;
        if let Some(manifest) = state.capture.as_deref() {
            match serde_json::from_slice::<serde_json::Value>(manifest) {
                // 【为什么清单失败只记日志】PCM 丢了不该让参数曲线也读不出来 ——
                // 那是两件独立的用户资产，而参数是更贵的那一份。
                Ok(value) => {
                    if let Err(error) = self.capture.restore_manifest(&value) {
                        log::warn!("[capture] manifest restore failed: {error}");
                    }
                }
                Err(error) => log::warn!("[capture] manifest is not valid JSON: {error}"),
            }
        }
        // 【Capture 的用户曲线走会话，不走 `pending_restore`】`pending_restore` 的读者
        // 是 `merge_pending_restore`，它只服务 ARA 文档；没有文档时那条路是 no-op
        // （见 `refresh_document`）。把曲线投给会话，它会在下一次时间线加载时叠上去。
        //
        // 会话可能还不存在（用户从没开过窗口）—— 那时先暂存在 owner 上，
        // `capture_session()` 创建会话时补投。
        if let Some(capture_edits) = state.capture_edits.as_deref() {
            match self.capture_session_if_initialized() {
                Some(session) => {
                    session.stash_serialized_capture_edits(capture_edits)?;
                    session.request_reload();
                }
                None => {
                    *self.pending_capture_edits.lock_or_recover() = Some(capture_edits.to_vec());
                }
            }
        }
        let mut restored = crate::state_channel::EditState::default();
        restored.restore(&state.edits)?;
        // 【G5】它到达时我们在不在一次编辑会话里（见 `EditState::arrived_in_edit_session`）。
        restored.arrived_in_edit_session = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .is_some_and(|document| document.in_edit_session());
        // 组件通道：这份载荷是**本实例**的存档，宿主用它做 undo / 复制轨 —— 恢复即权威。
        restored.from_archive = false;
        // 【T-D5】同一份载荷已经合入过就别再排队：宿主重复交回同一份存档时（重开编辑器、
        // 撤销后重做、文档重建），重放只会重复劳动，并在失败时重新点燃"没读回来"。
        if Self::restore_digest(&restored).is_some_and(|digest| {
            *self.restore_applied_digest.lock_or_recover() == Some(digest)
                && !self.has_pending_restore()
        }) {
            log::info!("[ara] persisted state unchanged since the last merge; not replaying it");
            self.refresh_document();
            return Ok(());
        }
        *self.pending_restore.lock_or_recover() = Some(restored);
        self.reset_restore_budget();
        self.note_restore_pending();
        self.refresh_document();
        Ok(())
    }

    /// 暂存 ARA 存档恢复出来的编辑，交给本 owner 的既有合并路径。
    ///
    /// 【为什么不直接写 `document.edits`】合并要做身份重绑、冲突判定、`needs_rebind`
    /// 处理 —— 这些都在 `merge_pending_restore` 里，而且已经过测试。存档通道只是另一个
    /// 来源，走同一条路才不会出现两份会漂移的实现。
    ///
    /// 【为什么覆盖已有的 `pending_restore`】VST3 组件状态与 ARA 存档携带的是同一份
    /// `EditState`。两者同时到达时后到的应当胜出（它是宿主更晚保存的那份）；而 ARA 存档
    /// 在宿主重载工程时是权威来源。
    ///
    /// 【为什么这里**不**刷新文档】唯一的调用点是 `DocumentSession::prepare_renderers`，
    /// 它已经持有 `transaction`，而刷新要取同一把不可重入的锁 —— 那就是用户报的
    /// "重开工程宿主未响应"。合入由 `prepare_renderers` 紧随其后的
    /// `merge_pending_restore` 循环完成，所以这里只负责**暂存**。
    pub(crate) fn stash_archive_restore(&self, restored: crate::state_channel::EditState) {
        // 【T-D5】与 `restore_state` 同一条纪律：已经合入过的同一份载荷不再重放。
        if Self::restore_digest(&restored).is_some_and(|digest| {
            *self.restore_applied_digest.lock_or_recover() == Some(digest)
                && !self.has_pending_restore()
        }) {
            log::info!("[ara] archive payload unchanged since the last merge; not replaying it");
            return;
        }
        *self.pending_restore.lock_or_recover() = Some(restored);
        self.reset_restore_budget();
        self.note_restore_pending();
    }
    /// 新载荷到达 ⇒ 等待账从头开始（否则上一份载荷用完的 10 s 会立刻压到这一份上）。
    fn reset_restore_budget(&self) {
        self.restore_attempts.store(0, Ordering::Relaxed);
        *self.restore_progress.lock_or_recover() = RestoreProgress::default();
        *self.restore_next_attempt.lock_or_recover() = None;
    }
    /// 一份载荷的摘要（T-D5）：同一份存档重复交回时不再重放。
    ///
    /// 【为什么用编码后的字节】它能覆盖"用户真的改了参数"与"宿主只是把同一份又给了
    /// 一遍"这两种形状，而字段级比较会因为 schema 演进而漏判。
    fn restore_digest(state: &crate::state_channel::EditState) -> Option<u64> {
        use std::hash::{Hash, Hasher};
        let bytes = state.encode().ok()?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut hasher);
        Some(hasher.finish())
    }
    /// 宿主图的形状摘要：**它是等待时间的唯一时钟**（T-D3）。
    ///
    /// 【为什么是摘要而不是计数】"宿主有没有动过"要一次比较得出结论。把
    /// `track_bindings` / `regions` / `region_items` 一起哈希，任何一次真实的交付
    /// 都会改变它，而我们的轮询不会 —— 于是"十秒"第一次真的是"十秒宿主没动"。
    fn restore_signature(&self, document: &super::document::DocumentSession) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        {
            let bindings = document.track_bindings.lock_or_recover();
            for (id, members) in bindings.iter() {
                id.hash(&mut hasher);
                members.hash(&mut hasher);
            }
        }
        document.regions.lock_or_recover().len().hash(&mut hasher);
        document.region_items.lock_or_recover().len().hash(&mut hasher);
        hasher.finish()
    }
    /// 是否还有一份**没合入**的持久化载荷（组件状态或 ARA 存档）。
    ///
    /// 【为什么需要这个只读问题】存档路径必须能判断"内存里的编辑是不是用户的数据"。
    /// 载荷还在 ⇒ 不是。见 `DocumentSession::has_unmerged_restore`。
    pub(crate) fn has_pending_restore(&self) -> bool {
        self.pending_restore.lock_or_recover().is_some()
    }
    /// 未合入载荷的编码形式（存档路径的兜底：宁可交回它，也不交空块）。
    pub(crate) fn pending_restore_bytes(&self) -> Option<Vec<u8>> {
        self.pending_restore
            .lock_or_recover()
            .as_ref()
            .and_then(|state| state.encode().ok())
    }
    /// 调用方持文档transaction；恢复候选集由宿主区域分配收窄，不按名称/旧序号猜测。
    ///
    /// 【为什么外面再包一层】恢复失败此前只有调用点的 `log::warn!`，用户与诊断包都
    /// 看不到"参数到底读回来没有"。包装层把结果写进 `restore_status`，四个调用点
    /// 不必各自重复这件事。
    ///
    /// 【T-D3：等待只由宿主图的变化推进】旧实现用"我们 poll 了多少次"当预算，
    /// 于是界面刷新得越快、预算烧得越快，10 秒的窗口能打印出 38.2 秒。现在只有一个
    /// 起点：图签名最后一次变化的时刻。签名不变且超过 `RESTORE_WAIT_WINDOW` 才报一次
    /// "没等到"，之后**不再重报**，直到宿主图再次变化。
    pub(super) fn merge_pending_restore(
        &self,
        document: &super::document::DocumentSession,
    ) -> Result<(), String> {
        // 【为什么要先看有没有载荷】成功路径会 `take()` 掉 pending；只有"进来时有、
        // 出去时没有"才说明这一次真的把状态合进去了。没有这个前提，一个从未收到
        // 载荷的实例也会被记成"恢复成功"。
        if !self.pending_restore.lock_or_recover().is_some() {
            return Ok(());
        }
        let now = std::time::Instant::now();
        let signature = self.restore_signature(document);
        let (waited, reported, changed) = {
            let mut progress = self.restore_progress.lock_or_recover();
            let changed = progress.signature != signature;
            if progress.since.is_none() || changed {
                progress.signature = signature;
                progress.since = Some(now);
                progress.reported = false;
            }
            (
                now.duration_since(progress.since.unwrap_or(now)),
                progress.reported,
                changed,
            )
        };
        if changed {
            // 宿主动了 ⇒ 退避失效，立刻重试一次（否则"改完工程还得再等十秒"）。
            *self.restore_next_attempt.lock_or_recover() = None;
            *self.restore_pending_reason.lock_or_recover() = None;
        }
        {
            let mut next = self.restore_next_attempt.lock_or_recover();
            if let Some(at) = *next {
                if now < at {
                    return Ok(());
                }
                *next = None;
            }
        }
        self.restore_attempts.fetch_add(1, Ordering::Relaxed);
        // 摘要要在 inner 之前取：成功路径会把 pending `take()` 掉。
        let pending_digest = self
            .pending_restore
            .lock_or_recover()
            .as_ref()
            .and_then(Self::restore_digest);
        let result = self.merge_pending_restore_inner(document);
        match &result {
            Ok(()) => {
                if self.pending_restore.lock_or_recover().is_some() {
                    if waited >= RESTORE_WAIT_WINDOW && !reported {
                        let reason = self
                            .restore_pending_reason
                            .lock_or_recover()
                            .clone()
                            .unwrap_or_else(|| "waiting-for-host-model".to_owned());
                        self.note_restore_failed(&format!(
                            "the host graph has not changed for {:.1}s ({reason}); the saved payload is kept and retried every {}s",
                            waited.as_secs_f64(),
                            RESTORE_RETRY_INTERVAL.as_secs()
                        ));
                        self.restore_progress.lock_or_recover().reported = true;
                        *self.restore_next_attempt.lock_or_recover() =
                            Some(now + RESTORE_RETRY_INTERVAL);
                    } else if !reported {
                        // 仍有载荷挂着（文档未就绪 / 空轨 / 缺 fade 元数据）：不是失败，
                        // 只是还不能收口。
                        self.note_restore_pending();
                        if self.restore_pending_reason.lock_or_recover().is_none() {
                            *self.restore_pending_reason.lock_or_recover() =
                                Some("waiting-for-host-model".to_owned());
                        }
                    }
                } else {
                    self.note_restore_applied();
                    if let Some(digest) = pending_digest {
                        *self.restore_applied_digest.lock_or_recover() = Some(digest);
                    }
                    self.reset_restore_budget();
                }
            }
            Err(error) => {
                self.note_restore_failed(error);
                self.restore_progress.lock_or_recover().reported = true;
                *self.restore_next_attempt.lock_or_recover() =
                    Some(now + RESTORE_RETRY_INTERVAL);
            }
        }
        result
    }

    /// 【T18】`reconcile` 的身份失败属于"等宿主图重建"还是"真失败"。
    ///
    /// 【为什么抽成纯函数】它是本轮的**唯一判据**，而它决定用户看到"还在等"还是
    /// "没读回来" —— 那两句话的可执行动作完全相反。纯函数让六个边界可单测，
    /// 不必构造一整份 ARA 文档。
    ///
    /// 【为什么按措辞判而不是错误类型】`reconcile` 返回 `Result<(), String>`
    /// （`state_channel` 不依赖本模块的错误类型），三条身份错误都带 `identity`：
    /// "ARA edit identity missing…" / "…missing or ambiguous in rebuilt host graph" /
    /// "ARA saved edit identities are ambiguous…"。其余错误（修订号耗尽）是真失败。
    fn classify_restore_identity_error(error: &str) -> RestoreIdentityVerdict {
        // 【为什么是 `identit` 而不是 `identity`】`state_channel` 的三条措辞分别是
        // "identity missing"、"identity missing or ambiguous" 与
        // **"identities are ambiguous"** —— 第三条是复数，用全词会漏掉它，
        // 而那正是"两条存档记录指向同一条宿主轨"这个最需要继续等的形状。
        if error.contains("identit") {
            RestoreIdentityVerdict::AwaitingHostGraph
        } else {
            RestoreIdentityVerdict::Fatal
        }
    }
    fn merge_pending_restore_inner(
        &self,
        document: &super::document::DocumentSession,
    ) -> Result<(), String> {
        let mut pending = self.pending_restore.lock_or_recover();
        let Some(saved) = pending.as_ref() else {
            return Ok(());
        };
        if !document.ready.load(Ordering::Acquire) {
            self.note_restore_waiting("document-not-ready");
            return Ok(());
        }
        // 【为什么先看 timeline】`assigned_timeline` 在时间线还没发布时会返回
        // "host timeline unavailable"；把它当失败会让正常重开过程显示
        // `RestoreUnresolved`。这里把它归为"还在等宿主收口"，并留下可读原因。
        if document.timeline.lock_or_recover().is_none() {
            self.note_restore_waiting("host-timeline-not-ready");
            return Ok(());
        }
        let host = self.assigned_timeline(document)?;
        if host.tracks.is_empty() {
            // 空轨组件仍能恢复私有参数分组；不借零音频scope恢复其它轨的源曲线。
            if saved.params.is_empty() && saved.tracks.is_empty() && saved.atlas.is_empty() {
                let mut edits = document.edits.lock_or_recover();
                edits.groups = saved.groups.clone();
                edits.revision = edits
                    .revision
                    .checked_add(1)
                    .ok_or("edit revision exhausted")?;
                *pending = None;
            } else {
                self.note_restore_waiting("host-timeline-empty");
            }
            return Ok(());
        }
        let allowed: BTreeSet<_> = host.tracks.iter().map(|t| t.id.clone()).collect();
        let bindings = document
            .track_bindings
            .lock_or_recover()
            .iter()
            .filter(|(id, _)| allowed.contains(*id))
            .map(|(id, identity)| (id.clone(), identity.clone()))
            .collect();
        let mut restored = saved.clone();
        // 【T18：身份对不上是"还在等"，不是"失败"】
        //
        // `reconcile` 的契约是严格的：任何一条旧轨在新图里找不到唯一身份，它就返回
        // `Err`（这条契约由 `restored_edits_refuse_ambiguous_empty_and_missing_host_identity`
        // 等用例钉住，本轮**不动**它 —— 猜测归属会把参数挂到别人的轨道上）。
        //
        // 但"对不上"有两种成因，处置相反：
        //   * 存档坏了 / 身份本身为空 —— 真失败，报 `failed`；
        //   * 宿主**这一次**的图还没长成（重开工程时 region 尚未交付、轨道还在重建）——
        //     该继续等，报 `pending` 并说明在等什么。
        // 此前两者都走 `?` ⇒ `note_restore_failed` ⇒ 界面显示"没能读回参数"，而实际
        // 上再等一轮（或用户在宿主里做一次编辑）就好了。
        if let Err(error) = restored.reconcile(&bindings) {
            match Self::classify_restore_identity_error(&error) {
                RestoreIdentityVerdict::AwaitingHostGraph => {
                    self.note_restore_waiting(&format!("host-graph-not-rebuilt: {error}"));
                    return Ok(());
                }
                RestoreIdentityVerdict::Fatal => return Err(error),
            }
        }
        restored.atlas.migrate_legacy_gaps(&restored.params)?;
        let mut changed_geometry = false;
        if !restored.atlas.is_empty() {
            let rebound = restored
                .atlas
                .rebind(&host, &document.parameter_identities_locked(&host)?)?;
            changed_geometry = !restored.atlas.same_layout(&rebound);
            restored.atlas = rebound;
        }
        let mut client = host.clone();
        restored.apply(&mut client);
        if changed_geometry {
            crate::editor::parameter_atlas::merge_root_projection(
                &mut client.params_by_root_track,
                restored
                    .atlas
                    .project_root_records(&host, &document.parameter_identities_locked(&host)?)?,
            );
        } else if restored.atlas.is_empty() && !restored.params.is_empty() {
            // 旧v2只有项目帧：在首次真实绑定的宿主布局建立source basis，随后移动/拉伸沿此迁移。
            // 不猜旧会话key/名字，也不能回推旧状态未保存的、首次加载前已改变的布局。
            restored.atlas = crate::editor::parameter_atlas::ParameterAtlas::default()
                .capture(&client, &document.parameter_identities_locked(&host)?)?;
        }
        let mut edits = document.edits.lock_or_recover();
        let mut merged = edits.merge(&host, &client, edits.revision)?;
        merged.groups = restored.groups.clone();
        let clip_ids = host
            .clips
            .iter()
            .map(|clip| clip.id.clone())
            .collect::<BTreeSet<_>>();
        // 【G5：权威侧合并分两种，不能一律覆盖】
        //
        // 旧规则是"恢复的那一份更权威"，理由写得很直白 —— 它只在**重开工程**时成立
        // （在线状态是空的）。而切屏重载 take 时交回来的载荷携带的是**更旧**的参数，
        // 那时"恢复的更权威"就等于"回退"。
        //
        // 三种情况分开处置：
        //   * 在线状态是**空的**（冷启动）⇒ 整份装上。没有别的数据源。
        //   * 载荷到达时我们正处在**一次编辑会话**里 ⇒ 覆盖。撤销/重做在 ARA 里就是
        //     一次还原会话（`ARAInterface.h:2938-2943`），undo 必须生效。
        //   * 其余（宿主交回一份旧载荷）⇒ **按键补缺**：只补在线状态里没有的键。
        let authoritative = !restored.from_archive
            || edits.is_pristine()
            || restored.arrived_in_edit_session
            || document.authoritative_restore();
        let mut kept_live = 0_usize;
        for (key, params) in restored.atlas.authorities.clone() {
            if authoritative || !merged.atlas.authorities.contains_key(&key) {
                merged.atlas.set_authority(&key, params);
            } else {
                kept_live += 1;
            }
        }
        if kept_live > 0 {
            log::info!(
                "[ara] restore kept {kept_live} live parameter set(s); the payload carries older \
                 values for them and the live state is newer"
            );
        }
        merged.atlas.regions.retain(|id, _| !clip_ids.contains(id));
        merged.atlas.regions.extend(restored.atlas.regions);
        merged
            .atlas
            .copy_seeds
            .retain(|_, seed| !allowed.contains(&seed.root));
        merged.atlas.copy_seeds.extend(restored.atlas.copy_seeds);
        merged.atlas.gaps.retain(|root, _| !allowed.contains(root));
        merged.atlas.gaps.extend(restored.atlas.gaps);
        let items = document.fade_items_locked(&clip_ids);
        if !merged.fades.is_empty() && items.is_empty() && !clip_ids.is_empty() {
            // 【为什么是 pending 而不是失败】fade 归属元数据要等宿主几何刷新后
            // 才能算出来；这是一次可重试的暂缓，不是身份冲突。保留载荷，下一次
            // `prepare_renderers` / 分配变化会再试。
            self.note_restore_waiting("fade-metadata-pending");
            return Ok(());
        }
        merged.fades.retain(|item, _| !items.contains(item));
        merged.fades.extend(restored.fades);
        merged.reconcile(&document.track_bindings.lock_or_recover())?;
        *edits = merged;
        *pending = None;
        Ok(())
    }

    /// 保存前重新核对完整宿主图；歧义或中间编辑态不能伪装成可恢复的state。
    ///
    /// 【契约】只有"调用方给的 state stream 非法"才可能失败；**任何**图未就绪、
    /// flush 超时、reconcile 失败、fade 元数据在途、payload 超限都降级为"返回当前
    /// 可用的最佳载荷"。宿主把 `getState` 的失败当成"插件存档失败"，而那比保存一份
    /// 稍旧的快照糟得多。
    pub(crate) fn encode_state(&self) -> Result<Vec<u8>, String> {
        let document = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade);
        let (edits, capture_edits) = if let Some(document) = document {
            // 【为什么 flush 失败不再传播】actor 可能正在跑一次长合成；等 10 秒再失败
            // 会把宿主保存线程冻住，而且仍然存不下。短预算 + 用当前权威继续编码。
            if let Err(error) =
                document.flush_editor_with_budget(std::time::Duration::from_millis(200))
            {
                log::warn!("[ara] flush before save failed: {error}; saving the committed state");
            }
            let _transaction = document.transaction.lock_or_recover();
            if !document.ready.load(Ordering::Acquire) {
                log::warn!(
                    "[ara] save requested before the host graph is ready; using the last good \
                     snapshot"
                );
                let edits = self
                    .pending_restore
                    .lock_or_recover()
                    .as_ref()
                    .and_then(|state| state.encode().ok())
                    .or_else(|| document.archive_cache.lock_or_recover().clone())
                    .or_else(|| document.edits.lock_or_recover().encode().ok())
                    .unwrap_or_else(default_edit_state_bytes);
                return self.wrap_component_state(edits, None);
            }
            if let Err(error) = self.merge_pending_restore(&document) {
                log::warn!("[ara] merge pending restore before save failed: {error}");
            }
            let mut edits = document.edits.lock_or_recover();
            if let Err(error) = edits.reconcile(&document.track_bindings.lock_or_recover()) {
                log::warn!(
                    "[ara] reconcile before save failed: {error}; saving the unpruned state"
                );
            }
            let mut local = edits.clone();
            match self.assigned_timeline(&document) {
                Ok(host) => {
                    let allowed: BTreeSet<_> = host.tracks.iter().map(|t| t.id.clone()).collect();
                    local.params.retain(|id, _| allowed.contains(id));
                    local.tracks.retain(|t| allowed.contains(&t.id));
                    let clips = host
                        .clips
                        .iter()
                        .map(|clip| clip.id.clone())
                        .collect::<BTreeSet<_>>();
                    local.atlas.regions.retain(|id, _| clips.contains(id));
                    // 权威表必须按**同一口径**裁剪：键从已裁剪的 regions 派生 —— 那正是本组件
                    // 真实持有的片段身份。不裁剪会让一个组件把另一个组件的参数写进自己的存档
                    // （两个组件共享同一 source/modification 身份时尤其明显）。
                    let allowed_authorities = local
                        .atlas
                        .regions
                        .values()
                        .filter_map(|record| {
                            crate::editor::parameter_atlas::authority_key(
                                record.identity.item.as_deref(),
                                &record.identity.modification,
                            )
                        })
                        .collect::<BTreeSet<_>>();
                    local
                        .atlas
                        .authorities
                        .retain(|key, _| allowed_authorities.contains(key));
                    local
                        .atlas
                        .copy_seeds
                        .retain(|_, seed| allowed.contains(&seed.root));
                    local.atlas.gaps.retain(|root, _| allowed.contains(root));
                    if !local.fades.is_empty() {
                        let items = document.fade_items_locked(&clips);
                        // 【为什么这里不再失败】fade 元数据在途时保留整份 fade 表；
                        // 恢复端会按 clip 身份逐条 reconcile，丢几条好过整份存不下。
                        if items.is_empty() && !clips.is_empty() {
                            log::warn!(
                                "[ara] fade ownership metadata pending; saving the unpruned fades"
                            );
                        } else {
                            local.fades.retain(|item, _| items.contains(item));
                        }
                    }
                    local.bindings.retain(|id, _| allowed.contains(id));
                }
                Err(error) => {
                    log::warn!(
                        "[ara] assigned timeline unavailable during save: {error}; saving the \
                         full state"
                    );
                }
            }
            match local.encode() {
                Ok(bytes) => {
                    *self.last_ara_state.lock_or_recover() = Some(bytes.clone());
                    (bytes, None)
                }
                Err(error) => {
                    log::error!("[ara] state encode failed: {error}; using the last good snapshot");
                    (
                        self.last_ara_state
                            .lock_or_recover()
                            .clone()
                            .unwrap_or_else(default_edit_state_bytes),
                        None,
                    )
                }
            }
        } else if self.bound_once.load(Ordering::Acquire) {
            // 【为什么不能落到 Capture 分支】这个实例曾经绑定过 ARA，只是文档已经
            // 关闭。Capture 的 `self.edits` 从未被 ARA 写过；序列化它等于用一份空
            // 状态覆盖用户全部编辑。保留最后一份成功载荷（或尚未合入的恢复载荷）。
            let edits = self
                .last_ara_state
                .lock_or_recover()
                .clone()
                .or_else(|| {
                    self.pending_restore
                        .lock_or_recover()
                        .as_ref()
                        .and_then(|state| state.encode().ok())
                })
                .unwrap_or_else(default_edit_state_bytes);
            (edits, None)
        } else {
            // 【Capture 形态：用户曲线在会话上，不在这里】`self.edits` 从未被任何
            // Capture 生产路径写过 —— 序列化它只会得到一个语法合法、语义为空的
            // `EditState`（那正是"重开工程参数全丢"的直接原因）。真正的权威是
            // `EditorSession::timeline`，见 `editor::capture_edits`。
            //
            // `edits` 仍然写一份默认 `EditState`：`restore_state` 要能把它读回来，
            // 而 ARA 字段在 Capture 下本来就该是空的。
            let edits = self
                .pending_restore
                .lock_or_recover()
                .as_ref()
                .and_then(|state| state.encode().ok())
                .or_else(|| self.edits.lock_or_recover().encode().ok())
                .unwrap_or_else(default_edit_state_bytes);
            let capture_edits = self.capture_edits_for_save();
            (edits, capture_edits)
        };
        self.wrap_component_state(edits, capture_edits)
    }

    /// 保存 Capture 用户编辑：优先 live 会话，其次恢复载荷，最后 owner 暂存。
    ///
    /// 【为什么顺序不能反】live 编辑是用户刚做的事，最权威；恢复载荷是"还没合入
    /// 时间线"的存档；owner 暂存是"连会话都还没建"。反过来会让一次保存把用户刚
    /// 画的曲线覆盖成旧存档。
    fn capture_edits_for_save(&self) -> Option<Vec<u8>> {
        match self.capture_session_if_initialized() {
            Some(session) => {
                // 【为什么不是只看 `is_loaded`】恢复后、actor 第一次加载前，时间线可能
                // 已经被写入（恢复路径/测试）；那仍是 live 权威，不能回落到空载荷。
                let live = if session.is_loaded() || session.has_live_capture_edits() {
                    if let Err(error) =
                        session.flush_with_budget(std::time::Duration::from_millis(200))
                    {
                        log::warn!(
                            "[capture] flush before save exceeded the budget: {error}; using \
                             the last committed curves"
                        );
                    }
                    match session.serialize_capture_edits() {
                        Ok(bytes) => Some(bytes),
                        Err(error) => {
                            log::error!("[capture] serialize before save failed: {error}");
                            None
                        }
                    }
                } else {
                    None
                };
                live.or_else(|| session.serialize_pending_capture_edits())
            }
            None => self.pending_capture_edits.lock_or_recover().clone(),
        }
    }

    /// 把 `EditState` 字节与 Capture 清单包成组件 state。
    ///
    /// 【为什么清单只带路径】PCM 在 sidecar 文件里（见 `capture::sidecar`）：
    /// 组件 state 的预算是 `MAX_COMPONENT_STATE`（64 MiB），而 4 分钟立体声 96 kHz 已是 184 MB，何况组件 state
    /// 会进宿主的每一次保存与撤销快照。清单是几 KB。
    fn wrap_component_state(
        &self,
        edits: Vec<u8>,
        capture_edits: Option<Vec<u8>>,
    ) -> Result<Vec<u8>, String> {
        let capture = self.capture.manifest();
        let has_takes = capture["takes"]
            .as_array()
            .is_some_and(|takes| !takes.is_empty());
        // 【为什么没有 Capture 数据时返回**裸** `EditState` 字节】两重理由：
        //
        // 1. 那些实例没有任何额外信息要存，套一层容器纯是开销；
        // 2. 更重要的是**兼容性**：此前 `encode_state` 的输出就是 `EditState` 的
        //    编码（ARA 路径下是一段 JSON）。加容器会让每一个既有消费者与既有工程
        //    都读到看不懂的东西 —— 而它们本来完全正常。
        //
        // 于是格式变成"有 Capture 数据才是容器，否则是历史格式"，而 `decode`
        // 两种都认。这条不对称是有意的：新信息需要新格式，旧信息不需要。
        let capture_edits = capture_edits.filter(|bytes| !bytes.is_empty());
        if !has_takes && capture_edits.is_none() {
            return Ok(edits);
        }
        Ok(crate::component_state::ComponentState {
            edits,
            capture: has_takes.then(|| capture.to_string().into_bytes()),
            capture_edits,
        }
        .encode())
    }

    /// 文档级的宿主时间线（**不**按实例收窄）。
    ///
    /// 【为什么与 `assigned_timeline` 分开】后者把 clip 收窄到本实例被分配的 region，
    /// 因为渲染必须只碰自己的那一份。而导出与几何判据要的是**整个文档**：
    /// 用户的心智模型是「这个工程」，不是「这个插件窗口」。
    fn document_timeline(
        &self,
        document: &super::document::DocumentSession,
    ) -> Result<hifishifter_kernel::state::TimelineState, String> {
        let mut timeline = document
            .timeline
            .lock_or_recover()
            .clone()
            .ok_or("host timeline unavailable")?;
        if let Some(tempo) = document.clock.tempo() {
            timeline.bpm = tempo;
        }
        document.project_plugin_musical_context_locked(&mut timeline);
        Ok(timeline)
    }

    /// 把**整个文档**导出成一个工作区。
    ///
    /// 【为什么是文档级而不是实例级】管道时代的服务端挂在实例上，于是导出的 clip
    /// 被收窄到「本实例拥有的 region」，10 条轨就有 10 个候选，而 `nMaxInstances = 1`
    /// 只能连一个。内核侧早就有这个唯一性（一个文档一个 `EditorSession`）。
    /// ARA 契约关心的是 per playback region 只有一个 editor renderer ——
    /// 导出不新增任何 renderer。
    ///
    /// 【为什么素材缺音频不再是失败】没有授权 PCM 的素材以
    /// [`hifishifter_host_link::AudioState::Partial`] 登记：几何保留、音频槽留空。
    /// 管道时代这会让**整份导入**失败，而用户看到的是「连接失败」。
    pub(crate) fn export_link(
        &self,
        root: &std::path::Path,
    ) -> Result<
        (
            hifishifter_host_link::LinkManifest,
            hifishifter_host_link::Workspace,
        ),
        String,
    > {
        use hifishifter_host_link as link;
        let document = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .ok_or_else(|| crate::editor::errors::DOCUMENT_CLOSED.error())?;
        let _transaction = document.transaction.lock_or_recover();
        self.merge_pending_restore(&document)?;
        if !document.ready.load(Ordering::Acquire) {
            return Err(format!(
                "{}; edit something in the host and try again",
                crate::editor::errors::HOST_MODEL_NOT_READY
            ));
        }
        let model_revision = document.revision.load(Ordering::Acquire);
        let mut timeline = self.document_timeline(&document)?;
        let edits_revision = {
            let edits = document.edits.lock_or_recover();
            // 导出的是**用户现在看到的**那一份，所以要把已保存的曲线叠上去；
            // 否则每次重新导出都会让 App 丢掉它上次提交的编辑。
            edits.apply(&mut timeline);
            edits.revision
        };

        let host_label = self.host_backend().app_name().unwrap_or("Host").to_string();
        let doc_key = self.document_key(&document);
        let identity = link::link_id(link::SourceKind::Ara, &host_label, &doc_key);
        // 【为什么目录名就是身份】重新导出必须落在**同一个**工作区上，否则每次导出
        // 都在用户磁盘上留一个新目录。身份稳定，目录名就稳定。
        let workspace = link::Workspace::at(root.join(identity.replace(':', "-")));

        let mut sources = Vec::new();
        {
            let available = document.edit_sources.lock_or_recover();
            let keys: BTreeSet<String> = timeline
                .clips
                .iter()
                .filter_map(|clip| clip.source_path.clone())
                .collect();
            for key in keys {
                let Some(pcm) = available.get(&key) else {
                    sources.push(link::SourceEntry {
                        key,
                        sample_rate: 0,
                        channels: 0,
                        frames: 0,
                        state: link::AudioState::Partial,
                        file: None,
                        blake3: None,
                        bytes: None,
                        note: Some(crate::editor::errors::HOST_PCM_UNAVAILABLE.to_string()),
                    });
                    continue;
                };
                match workspace.write_audio(&key, pcm.sample_rate, &pcm.planes) {
                    Ok(audio) => sources.push(link::SourceEntry {
                        key,
                        sample_rate: pcm.sample_rate,
                        channels: pcm.planes.len() as u32,
                        frames: pcm.planes.first().map_or(0, Vec::len) as u64,
                        state: link::AudioState::Complete,
                        file: Some(audio.file),
                        blake3: Some(audio.blake3),
                        bytes: Some(audio.bytes),
                        note: None,
                    }),
                    Err(error) => {
                        log::warn!("[link] cannot write audio for {key}: {error}");
                        sources.push(link::SourceEntry {
                            key,
                            sample_rate: pcm.sample_rate,
                            channels: pcm.planes.len() as u32,
                            frames: pcm.planes.first().map_or(0, Vec::len) as u64,
                            state: link::AudioState::Partial,
                            file: None,
                            blake3: None,
                            bytes: None,
                            note: Some(error),
                        });
                    }
                }
            }
        }

        // 逐条降级报告：宿主里被跳过、进不来的东西。空表示**没有丢掉任何东西**。
        let losses: Vec<link::LossEntry> = document
            .skipped_regions
            .lock_or_recover()
            .iter()
            .map(|(index, reason)| link::LossEntry {
                id: format!("region #{index}"),
                reason: reason.clone(),
                detail: None,
            })
            .collect();

        let manifest = link::LinkManifest {
            format: link::FORMAT.into(),
            format_version: link::FORMAT_VERSION,
            link_id: identity.clone(),
            source_kind: link::SourceKind::Ara,
            host_label,
            created_unix_ms: link::now_ms(),
            producer: link::Producer {
                implementation: format!("hifishifter-ara {}", crate::VERSION),
                pid: Some(std::process::id()),
            },
            // 【为什么不写 `project_sec` 以外的推导值】时间轴就是宿主事实的投影；
            // 这里是唯一一处把 `TimelineState` 交给外部的出口。
            timeline: serde_json::to_value(&timeline).map_err(|error| error.to_string())?,
            // 【为什么在这里算，而不是让消费者算】见 `LinkManifest::host_geometry`：
            // 判据的两边必须出自同一次计算，否则任何浮点归一化差异都会变成假漂移。
            host_geometry: hifishifter_kernel::editor::host_timeline::host_geometry_projection(
                &timeline,
            ),
            sources,
            granted: vec![link::Capability::Commit, link::Capability::Audio],
            revisions: link::Revisions {
                edits: edits_revision,
                model: model_revision,
            },
            admission: link::Admission::Strict,
            losses,
            diagnostics: Some(self.link_diagnostics(&document, &timeline)),
        };
        workspace.write_manifest(&manifest)?;
        workspace.write_status(&link::LinkStatus::initial(&identity, manifest.revisions))?;
        Ok((manifest, workspace))
    }

    /// 文档身份：`TrackBindings` 的 `(modification, source)` 签名。
    ///
    /// 【为什么不用 region key】region key 只在会话内稳定。`TrackBindings` 是既有的、
    /// 被文档注释称为「跨会话身份」的那一份（见 `ara/model.rs` 的 `publish_track_bindings`），
    /// 它由宿主给的两个 persistentID 组成，重开宿主工程后仍然相同。
    fn document_key(&self, document: &super::document::DocumentSession) -> String {
        let bindings = document.track_bindings.lock_or_recover();
        if bindings.is_empty() {
            return "unknown-document".to_string();
        }
        let mut pairs: Vec<String> = bindings
            .values()
            .flatten()
            .map(|(modification, source)| format!("{modification}|{source}"))
            .collect();
        pairs.sort();
        pairs.dedup();
        let joined = pairs.join(",");
        // 摘要而非原文：身份要短、要稳定、要能进文件名，而原文可以很长。
        blake3::hash(joined.as_bytes()).to_hex().to_string()[..16].to_string()
    }

    /// 只读现场诊断：渲染器就绪度，以及**已发布输出对片段范围的覆盖**。
    ///
    /// 【为什么这次真的会有人读】管道时代它被生产出来后在 App 侧**没有任何消费者** ——
    /// 一条只写不读的线。现在它随清单落盘，面板上直接显示
    /// 「宿主渲染器尚未就绪」「该片段在宿主侧是静音」。
    ///
    /// 【为什么要按片段给覆盖与峰值，而不是只报 ready】用户真正的疑问是
    /// 「我导出的是不是有声音」。就绪只说明快照发布了，不说明那一段**有内容**。
    fn link_diagnostics(
        &self,
        document: &super::document::DocumentSession,
        timeline: &hifishifter_kernel::state::TimelineState,
    ) -> serde_json::Value {
        // 与面板显示的顺序一致：按起点排前 8 条，避免长工程把清单撑大。
        let mut clips = timeline.clips.iter().collect::<Vec<_>>();
        clips.sort_by(|a, b| a.start_sec.total_cmp(&b.start_sec));
        let ranges = clips
            .iter()
            .take(8)
            .map(|clip| (clip.id.clone(), clip.start_sec, clip.length_sec))
            .collect::<Vec<_>>();
        let renderers = document
            .renderer_owners()
            .into_iter()
            .filter(|owner| owner.renders_playback())
            .map(|owner| {
                let (busy, error) = owner.local_preparation_state();
                serde_json::json!({
                    "busy": busy,
                    "error": error,
                    "prepared": owner.prepared_snapshot(),
                    "outputs": owner
                        .snapshots
                        .iter()
                        .map(|snapshot| snapshot.diagnose(&ranges))
                        .collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({ "renderers": renderers })
    }

    /// 合并一份**回写**的编辑，并返回要落盘的状态。
    ///
    /// 【几何归属判据为什么在这里算】消费者只有宿主几何的一份拷贝，生产者才有权威。
    /// 判据放在生产者侧之后，消费者侧的**活状态**（后台的假立体声折叠扫描、
    /// 波形回填、音高分析都在写它）再也不可能打穿判据 —— 于是那个用于压制后台子系统
    /// 的进程级全局开关也随之消失。
    pub(crate) fn apply_link_edit(
        &self,
        identity: &str,
        edit: &hifishifter_host_link::EditManifest,
    ) -> hifishifter_host_link::LinkStatus {
        use hifishifter_host_link as link;
        let mut status = link::LinkStatus {
            format: link::STATUS_FORMAT.into(),
            format_version: link::FORMAT_VERSION,
            link_id: identity.to_string(),
            ok: false,
            code: None,
            error: None,
            revisions: link::Revisions::default(),
            updated_unix_ms: link::now_ms(),
        };
        let outcome = (|| -> Result<(), (u32, String)> {
            let document = self
                .document
                .lock_or_recover()
                .as_ref()
                .and_then(std::sync::Weak::upgrade)
                .ok_or((
                    link::error_code::DOCUMENT_CLOSED,
                    crate::editor::errors::DOCUMENT_CLOSED.error(),
                ))?;
            if edit.link_id != identity {
                return Err((
                    link::error_code::LINK_MISMATCH,
                    format!("this edit belongs to {}, not {identity}", edit.link_id),
                ));
            }
            // ── 捕获：短事务内冻结几何、参数与渲染输入 ─────────────────────
            //
            // 【为什么必须是一个独立的作用域】`transaction` 是**不可重入**的互斥量，
            // 而下面"提交"那一半还要再取一次。第一份守护必须在这里释放 ——
            // 否则第二次获取就是一次必然的死锁（而且它阻塞的是宿主回调）。
            // 同理，渲染也不能在事务里做：那会把宿主挡住整个渲染时长。
            let (base_model, candidate, epoch, scope, inputs) = {
                let _transaction = document.transaction.lock_or_recover();
                self.merge_pending_restore(&document)
                    .map_err(Self::invalid_edit)?;
                if !document.ready.load(Ordering::Acquire) {
                    return Err((
                        link::error_code::MODEL_NOT_READY,
                        format!(
                            "{}; edit something in the host",
                            crate::editor::errors::HOST_MODEL_NOT_READY
                        ),
                    ));
                }
                let base_model = document.revision.load(Ordering::Acquire);
                // 几何归属：宿主拥有的字段变了就拒绝，并**指名字段**。
                // 判据在这里算而不是在消费者侧 —— 只有生产者有权威，
                // 而消费者的活状态（后台的假立体声折叠、波形回填）因此再也打不穿它。
                let current = self
                    .document_timeline(&document)
                    .map_err(Self::invalid_edit)?;
                let differences = hifishifter_kernel::editor::host_timeline::geometry_differences(
                    &edit.host_geometry,
                    &hifishifter_kernel::editor::host_timeline::host_geometry_projection(&current),
                );
                if !differences.is_empty() {
                    return Err((
                        link::error_code::GEOMETRY_CHANGED,
                        format!("ara_host_fields: {}", differences.join("; ")),
                    ));
                }
                if base_model != edit.base_revisions.model {
                    return Err((
                        link::error_code::CONFLICT,
                        format!(
                            "{} the host model is {base_model} but this edit was written against {}",
                            crate::editor::errors::CONFLICT_PREFIX,
                            edit.base_revisions.model
                        ),
                    ));
                }
                let params: std::collections::BTreeMap<
                    String,
                    hifishifter_kernel::state::TrackParamsState,
                > = serde_json::from_value(edit.params.clone()).map_err(Self::invalid_edit)?;
                let tracks: Vec<hifishifter_kernel::state::Track> =
                    serde_json::from_value(edit.tracks.clone()).map_err(Self::invalid_edit)?;
                let mut edits = document.edits.lock_or_recover();
                edits
                    .reconcile(&document.track_bindings.lock_or_recover())
                    .map_err(Self::invalid_edit)?;
                let mut candidate = edits
                    .merge_params(
                        &self
                            .assigned_timeline(&document)
                            .map_err(Self::invalid_edit)?,
                        &params,
                        &tracks,
                        edit.base_revisions.edits,
                    )
                    .map_err(Self::invalid_edit)?;
                let mut curve_timeline = self
                    .assigned_timeline(&document)
                    .map_err(Self::invalid_edit)?;
                candidate.apply(&mut curve_timeline);
                candidate.atlas = edits
                    .atlas
                    .capture(
                        &curve_timeline,
                        &document
                            .parameter_identities_locked(&curve_timeline)
                            .map_err(Self::invalid_edit)?,
                    )
                    .map_err(Self::invalid_edit)?;
                candidate
                    .reconcile(&document.track_bindings.lock_or_recover())
                    .map_err(Self::invalid_edit)?;
                drop(edits);
                let inputs = document
                    .renderer_owners()
                    .into_iter()
                    .filter(|owner| owner.renders_playback())
                    .map(|owner| {
                        let (keys, input) =
                            owner.ara_render_input(&document, &candidate, true)?;
                        Ok((owner, keys, input))
                    })
                    .collect::<Result<Vec<_>, String>>()
                    .map_err(Self::invalid_edit)?;
                (
                    base_model,
                    candidate,
                    document.render_epoch.load(Ordering::Acquire),
                    document.scope_revision.load(Ordering::Acquire),
                    inputs,
                )
            };
            // ── 渲染：**不持事务锁** ──────────────────────────────────────
            let mut prepared = Vec::new();
            for (owner, keys, input) in inputs {
                for publisher in &owner.snapshots {
                    publisher.collect_retired();
                }
                let mut snapshots = input
                    .render(Arc::new(std::sync::atomic::AtomicBool::new(false)))
                    .map_err(Self::invalid_edit)?;
                super::snapshot::compact_prepared(&mut snapshots).map_err(Self::invalid_edit)?;
                prepared.push((owner, keys, snapshots));
            }
            // ── 提交：短事务内复核（渲染期间一切可能都变了）并发布 ──────────
            let _transaction = document.transaction.lock_or_recover();
            if !document.ready.load(Ordering::Acquire)
                || document.revision.load(Ordering::Acquire) != base_model
            {
                return Err((
                    link::error_code::CONFLICT,
                    "host model changed while the edit was being applied; export again".into(),
                ));
            }
            let mut edits = document.edits.lock_or_recover();
            if edits.revision != edit.base_revisions.edits {
                return Err((
                    link::error_code::CONFLICT,
                    "edit revision changed while the edit was being applied; export again".into(),
                ));
            }
            if document.render_epoch.load(Ordering::Acquire) != epoch
                || document.scope_revision.load(Ordering::Acquire) != scope
            {
                return Err((
                    link::error_code::CONFLICT,
                    "host audio access changed while the edit was being applied; try again".into(),
                ));
            }
            for (owner, keys, snapshots) in &prepared {
                if owner.assigned_regions().map_err(Self::invalid_edit)? != *keys {
                    return Err((
                        link::error_code::CONFLICT,
                        "assigned regions changed while the edit was being applied; try again"
                            .into(),
                    ));
                }
                if !owner
                    .snapshots
                    .iter()
                    .zip(snapshots)
                    .all(|(publisher, snapshot)| publisher.has_capacity(snapshot))
                {
                    return Err((
                        link::error_code::CONFLICT,
                        "retired snapshot budget exhausted; reopen the plug-in window".into(),
                    ));
                }
            }
            for (owner, keys, snapshots) in prepared {
                for (publisher, snapshot) in owner.snapshots.iter().zip(snapshots) {
                    publisher
                        .publish(snapshot)
                        .map_err(|error| format!("snapshot publish failed: {error:?}"))
                        .map_err(Self::invalid_edit)?;
                }
                owner.record_prepared(base_model, candidate.revision, epoch, scope, keys);
            }
            *edits = candidate;
            log::info!(
                "[link] applied an edit revision={} model={base_model}",
                edits.revision
            );
            Ok(())
        })();
        match outcome {
            Ok(()) => status.ok = true,
            Err((code, error)) => {
                log::warn!("[link] rejected an edit: {error}");
                status.code = Some(code);
                status.error = Some(error);
            }
        }
        status.revisions = self.link_revisions();
        status
    }

    fn link_revisions(&self) -> hifishifter_host_link::Revisions {
        let document = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade);
        match document {
            Some(document) => hifishifter_host_link::Revisions {
                edits: document.edits.lock_or_recover().revision,
                model: document.revision.load(Ordering::Acquire),
            },
            None => hifishifter_host_link::Revisions::default(),
        }
    }

    /// 把一条内部错误规整成「协议码 + 文本」。
    ///
    /// 【为什么统一走 `INVALID_EDIT`】能走到这里的失败都属于「这份编辑本身不合格」：
    /// 曲线非有限、轨道未知、音量越界、JSON 形状不对。消费者据此**不要重试**，
    /// 而是把 `error` 显示给用户 —— 与 `CONFLICT`（重导一次就好）处置相反。
    fn invalid_edit(error: impl std::fmt::Display) -> (u32, String) {
        (
            hifishifter_host_link::error_code::INVALID_EDIT,
            error.to_string(),
        )
    }
    /// 调用方在短事务内冻结几何/参数/PCM，只复制本renderer授权源，不执行合成。
    pub(crate) fn ara_render_input(
        &self,
        document: &super::document::DocumentSession,
        edits: &crate::state_channel::EditState,
        edited: bool,
    ) -> Result<(Vec<u64>, super::input::RenderInput), String> {
        if !document.ready.load(Ordering::Acquire) {
            return Err(format!(
                "{}; {}",
                crate::editor::errors::HOST_PCM_UNAVAILABLE,
                crate::editor::errors::HOST_MODEL_NOT_READY
            ));
        }
        let keys = self.assigned_regions().map_err(|e| e.to_string())?;
        let mut timeline = self.assigned_timeline(document)?;
        // 所有授权轨道保留原kernel的全局solo/父链判定，clip仍只有本renderer分配区域。
        timeline.tracks = document.workspace_timeline_locked()?.tracks;
        // 【为什么这里也要播种宿主 take 事实】渲染路径**不经过** `workspace_timeline_locked`
        // 的 clip 过滤（见上一行的注释：只借了 tracks）。而 `assigned_timeline` 读的是
        // `document.timeline`（ARA 映射产物），ARA **没有**方向位/循环源/声道模式
        // （`ara::mapping::LOST_FIELDS`）—— 不播种，倒放片段会被按正放合成（用户报障：
        // "在 HiFiShifter 中，倒放被识别为正放"）。与音阶播种（`document.timeline` 的
        // `project_scale_notes`）是同一条纪律、同一个理由。
        document.project_host_take_facts_locked(&mut timeline);
        let mut resolved = edits.clone();
        resolved.reconcile(&document.track_bindings.lock_or_recover())?;
        resolved.apply(&mut timeline);
        if !resolved.groups.is_empty() {
            // 空父轨也能承载参数组算法；不加入其音频clip，更不混入其它renderer的区域。
            let mut grouped_view = timeline.clone();
            document.project_private_group_view(&mut grouped_view, &resolved)?;
            timeline.tracks = grouped_view.tracks;
            for track in &mut timeline.tracks {
                track.parent_id = None;
            }
            resolved.groups.processing_settings(
                &mut timeline,
                &document.group_aliases(""),
                &std::collections::BTreeSet::new(),
            );
        }
        let geometry = document.regions.lock_or_recover();
        // 【为什么消失的 region 只跳过】此前这里 `ok_or("assigned region disappeared")?`
        // + `collect::<Result<_,_>>()?`：只要**一条**已分配的 key 不在 `regions` 表里
        // （宿主销毁了 region、分配还没撤销的窗口），整份 `ara_render_input` 失败 ⇒
        // `prepare_job` 不发布快照 ⇒ 声音停在旧版本，且界面上只有一条内部错误。
        // 逐条过滤：消失的那条本次不参与渲染，其余照常。`keys` 与 `regions` 必须
        // **同长同序**（它们一起进 `RenderInput`），所以两者一起筛。
        let mut resolved_keys = Vec::with_capacity(keys.len());
        let mut regions = Vec::with_capacity(keys.len());
        let mut vanished = 0_usize;
        for key in &keys {
            match geometry.get(key) {
                Some(region) => {
                    resolved_keys.push(*key);
                    regions.push(region.clone());
                }
                None => vanished += 1,
            }
        }
        drop(geometry);
        if vanished > 0 {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.capture.vanished-region",
                format_args!(
                    "[ara] {vanished} assigned region(s) are gone from the model this round; \
                     they are omitted from the render input"
                ),
            );
        }
        let keys = resolved_keys;
        let stretch = regions.iter().any(|region| {
            (region.duration_in_modification_time - region.duration_in_playback_time).abs() > 1e-9
        });
        let owned_fade = regions.iter().any(|region| {
            region.has_content_based_fade_at_head || region.has_content_based_fade_at_tail
        });
        if owned_fade {
            document.project_audio_fades_locked(&mut timeline, &resolved)?;
        }
        // 【倒放也必须走内核】未编辑、无拉伸、无淡化委托的片段本来走
        // `mix_plain_regions` 直通（`RenderInput.timeline = None`）—— 那是"原样放 ARA
        // 授权 PCM"。但倒放片段**不能**直通：ARA 给的是正向 PCM，方向翻转只发生在内核
        // 的 `render_mixdown_internal`（`reverse_interleaved_frames`）。不把倒放片段推进
        // 内核，它就会被原样按正向放出来（实测：输出 = 源窗口正放）。
        let reversed_clip = timeline.clips.iter().any(|clip| clip.reversed);
        // 【为什么"事实还没到"也必须走内核】方向 / 循环源 / 声道模式是 ARA **表达不到**
        // 的输入（`ara::mapping::LOST_FIELDS`），只有宿主清单能给。直通路径
        // （`mix_plain_regions`）不做这些处理 —— 于是在"宿主有原生扩展、清单却还没刷新"
        // 这个窗口里，clip 的 `reversed` 还是默认 false，一个倒放片段会被原样按正放放出来：
        // 波形、按钮都正常，声音是错的，且没有任何线索。
        //
        // 【为什么条件不是"清单为空"】宿主**根本不给**这些概念时（ARA-only），强制走内核
        // 换不来任何信息，只会白白多算一遍、多占一份工作缓冲。真正会出错的是上面那个
        // 中间窗口，所以判据是"有原生扩展 且 清单还没到"。
        let facts_pending =
            self.native_host().is_some() && document.ui_tracks.lock_or_recover().is_empty();
        let kernel_render = edited
            || stretch
            || owned_fade
            || reversed_clip
            || !resolved.groups.is_empty()
            || facts_pending;
        let clip_parameters = if resolved.atlas.is_empty() {
            Default::default()
        } else {
            resolved
                .atlas
                .project_local(&timeline, &document.parameter_identities_locked(&timeline)?)?
        };
        let available = if kernel_render {
            document.edit_sources.lock_or_recover()
        } else {
            document.sources.lock_or_recover()
        };
        let sources = regions
            .iter()
            .map(|region| region.audio_source_persistent_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter_map(|id| available.get(&id).map(|pcm| (id.clone(), pcm.clone())))
            .collect();
        Ok((
            keys,
            super::input::RenderInput {
                timeline: kernel_render.then_some(timeline),
                regions,
                sources,
                // 宿主通报的工程速率。没有内核渲染时 `mix_plain_regions` 要靠它把宿主的
                // "工程速率秒"折回源自己的秒（见 `RenderInput::project_sample_rate`）。
                project_sample_rate: Some(self.setup_sample_rate()).filter(|rate| *rate > 0.0),
                clip_parameters,
            },
        ))
    }

    /// 从宿主时间线筛选实际分配的区域，而不是让每个处理器混整张文档。
    fn assigned_timeline(
        &self,
        document: &super::document::DocumentSession,
    ) -> Result<hifishifter_kernel::state::TimelineState, String> {
        let keys = self.assigned_regions().map_err(|e| e.to_string())?;
        // 身份缺席时按 region key 派生（见 `clip_id_for`）：region 分配早于
        // `clip_ids` 重算，这个窗口里的渲染不该失败。
        //
        // 【为什么无效键只跳过】`clip_id_for` 只对 key==0 返回 `None`。此前这里的
        // `ok_or(...)?` + `collect::<Result>()` 会让**一个**无效键把整份
        // `assigned_timeline` 打失败 —— 而它是 `ara_render_input` /
        // `merge_pending_restore` / `encode_state` 的共同入口 ⇒ 快照不发布 +
        // 存档保存失败。逐条过滤：那一条 region 本次不参与渲染，其余照常。
        let mut invalid_keys = 0_usize;
        let ids = keys
            .iter()
            .filter_map(|key| match document.clip_id_for(*key) {
                Some(id) => Some(id),
                None => {
                    invalid_keys += 1;
                    None
                }
            })
            .collect::<BTreeSet<_>>();
        if invalid_keys > 0 {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.assigned.invalid-region-key",
                format_args!(
                    "[ara] {invalid_keys} assigned region(s) have no usable key this round; \
                     they are omitted from the render input"
                ),
            );
        }
        let mut timeline = document
            .timeline
            .lock_or_recover()
            .clone()
            .ok_or("host timeline unavailable")?;
        if let Some(tempo) = document.clock.tempo() {
            timeline.bpm = tempo;
        }
        // 【为什么这里也要播种音阶】渲染路径**不经过** `workspace_timeline_locked`
        // 的 clip 分支：它克隆原始 ARA 时间线后只借用后者的 `tracks`。所以只在
        // `workspace_timeline_locked` 里播种，渲染拿到的仍是默认 C 大调 ——
        // 正是"界面显示 Gb、内核按 C 渲染"那条缺陷。
        document.project_plugin_musical_context_locked(&mut timeline);
        timeline.clips.retain(|clip| ids.contains(&clip.id));
        // 被分配了 sequence、当前却没有 clip 的轨道仍然留下：空 folder 轨、只有静音
        // item 或 PCM 尚未授权的子轨都属此类，而它们本该是**参数根** —— 与
        // `add_group_view_tracks` 的既有口径一致（空父轨也能承载参数组算法）。
        //
        // 【为什么这不会泄漏无关轨道】这条边只来自宿主 `createRegionSequence` 的真实
        // 键（`sequence_track_index`），且只取**本 owner 已分配**的那些 sequence。
        // 绝不按轨名/位置/序号凭空造轨道。
        let assigned_sequences = self
            .sequences
            .lock_or_recover()
            .values()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>();
        let empty_roots = document
            .sequence_track_index
            .lock_or_recover()
            .iter()
            .filter(|(key, _)| assigned_sequences.contains(key))
            .filter_map(|(_, index)| timeline.tracks.get(*index).map(|track| track.id.clone()))
            .collect::<BTreeSet<_>>();
        timeline.tracks.retain(|track| {
            timeline.clips.iter().any(|clip| clip.track_id == track.id)
                || empty_roots.contains(&track.id)
        });
        Ok(timeline)
    }

    /// 把租约关联到真实文档；关闭文档时由其同步撤销。
    pub fn bind_to_document(
        self: &Arc<Self>,
        document: Arc<super::document::DocumentSession>,
        generation: ApiGeneration,
        known: ExtensionRoles,
        assigned: ExtensionRoles,
        companion: Option<ara2_bridge::companion::CompanionControllerBinding<'static>>,
    ) -> Result<*const ara2_bridge::sys::ARAPlugInExtensionInstance, AraError> {
        let mut current = self.binding.lock().unwrap_or_else(|p| p.into_inner());
        if current.is_some() {
            // 已绑定再绑定：只有"同一份仍然活着的文档"才是重复绑定。
            //
            // 【为什么不变量从"一个实例只绑一个文档"改成"同一时刻只服务一份活文档"】
            // 宿主顺序不可控：REAPER 可能在旧文档还没关闭时就为新工程发出 bind。
            // 此时 `closed` 仍是 false，若一律拒绝，新文档永远拿不到 renderer
            // （用户报的"重开只显示等待宿主音频，删掉插件再加就好"）。
            //
            // 【为什么不清 `bound_once`】它是历史事实："这个实例绑定过 ARA"。清了它
            // 会让 "document closed" 静默退化成 `ARA_NOT_BOUND`，那是在陈述一件
            // 没发生过的事。
            let same_live_document = self.bound_document_id.load(Ordering::Acquire) == document.id
                && !self.closed.load(Ordering::Acquire);
            if same_live_document {
                return Err(AraError::InvalidState("extension already bound"));
            }
            let previous_document = self
                .document
                .lock_or_recover()
                .as_ref()
                .and_then(std::sync::Weak::upgrade);
            let previous_id = previous_document.as_ref().map(|document| document.id);
            if let Some(previous) = previous_document {
                // 旧文档可能仍活着：只摘掉本实例的租约，不关闭整份文档。
                previous.detach_owner(self);
                previous.report_playback_ownership();
            }
            log::warn!(
                "[ara] writer {} moves from document {:?} to {} (previous document {}); \
                 retiring the old extension binding",
                self.writer_id(),
                previous_id,
                document.id,
                if self.closed.load(Ordering::Acquire) {
                    "already closed"
                } else {
                    "still alive"
                }
            );
            self.stop_editor_for_document();
            // 旧扩展实例不能立刻 drop：宿主可能还有最后一次合法回调。
            if let Some(old) = current.take() {
                self.retired.lock_or_recover().push(old);
            }
            self.prepare_rebind_from_previous_document();
            self.assignments.lock_or_recover().clear();
            self.sequences.lock_or_recover().clear();
            self.document.lock_or_recover().take();
            self.rebind_generation.fetch_add(1, Ordering::Relaxed);
            self.closed.store(false, Ordering::Release);
        }
        let owner = Arc::downgrade(self);
        let document_id = document.id;
        let observer = Arc::new(
            move |role: ExtensionRoles, keys: &[usize], sequences: &[usize]| {
                if let Some(owner) = owner.upgrade() {
                    let document = owner
                        .document
                        .lock_or_recover()
                        .as_ref()
                        .and_then(std::sync::Weak::upgrade);
                    let Some(document) = document else {
                        return;
                    };
                    // assignment写入、撤销和最终发布必须共用此短事务，消除check→publish竞态。
                    let _transaction = document.transaction.lock_or_recover();
                    let keys = keys.iter().map(|key| *key as RegionKey).collect::<Vec<_>>();
                    // 【为什么逐条过滤而不是整批作废】`resolve` 是整批语义：一条坏 key
                    // （跨文档/已销毁）就让**整批**返回 `Err`。而宿主会把当前分配整体重发，
                    // 分割后它常把已销毁父段的 key 也带进来 —— 整批作废会连**新 region**
                    // 一起丢掉，表现就是"分割后既不出声也不显示"，且没有任何提示。
                    // 只剔除坏的那条：其余（包括新 region）照常。
                    let (next_keys, dropped): (Vec<RegionKey>, usize) = {
                        let owners = region_owners().lock().unwrap_or_else(|p| p.into_inner());
                        let mut kept = Vec::with_capacity(keys.len());
                        let mut dropped = 0_usize;
                        for key in keys.iter().copied() {
                            if owners.owner_of(key) == Some(document_id) {
                                kept.push(key);
                            } else {
                                dropped += 1;
                            }
                        }
                        (kept, dropped)
                    };
                    let valid = dropped == 0;
                    let count = next_keys.len();
                    let next_sequences =
                        sequences.iter().map(|key| *key as u64).collect::<Vec<_>>();
                    let changed = owner.assignments.lock_or_recover().get(&role.bits())
                        != Some(&next_keys)
                        || owner.sequences.lock_or_recover().get(&role.bits())
                            != Some(&next_sequences);
                    if changed {
                        document.scope_revision.fetch_add(1, Ordering::AcqRel);
                    }
                    owner
                        .assignments
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .insert(role.bits(), next_keys);
                    owner.sequences.lock_or_recover().insert(
                        role.bits(),
                        sequences.iter().map(|key| *key as u64).collect(),
                    );
                    if valid {
                        log::info!(
                            "[ara] renderer assignment role={} regions={count}",
                            role.bits()
                        );
                    } else {
                        log::warn!(
                            "[ara] renderer assignment role={} kept {count} region(s) and \
                             dropped {dropped} unknown or cross-document key(s)",
                            role.bits()
                        );
                    }
                    owner.cancel_preparation();
                    owner.snapshots.iter().for_each(|snapshot| snapshot.clear());
                    let owners = document.renderer_owners();
                    for renderer in &owners {
                        if let Err(error) = renderer.merge_pending_restore(&document) {
                            log::warn!("[ara] instance state unresolved: {error}");
                        }
                    }
                    // 某轨首次获得分配可能恢复其state并推进共享revision；其它轨也需要最新任务。
                    drop(_transaction);
                    // 分配变化是角色集合可能改变的第二个时刻：宿主可能在这一步才把
                    // playback 角色交给某个实例。报一次读数，便于从日志判断
                    // "到底有没有实例在渲染"。
                    document.report_playback_ownership();
                    for renderer in owners {
                        renderer.refresh_host_geometry();
                        renderer.prepare();
                    }
                }
            },
        );
        // 【为什么 `EDITOR_VIEW` 必须在 `supported` 里】ARA SDK 对
        // `ARAPlugInExtensionInstance` 的契约是**双向**的：
        //
        //   "Plug-ins **must provide all interfaces that have been requested by the host
        //    through the role assignment**, and suppress interfaces explicitly excluded by
        //    the roles — e.g. if the host did not assign kARAEditorRendererRole even it was
        //    known, editorRendererInterface will be NULL."
        //
        // 此前这里只声明 playback + editor，于是 `enabled = supported & (!known | assigned)`
        // **永远不可能**包含 view ⇒ `editorViewInterface` 恒为 NULL。而 Cubase 实测分配的
        // 正是 `[editor,view]`：它按契约拿 `editorViewInterface` 来通知选区与隐藏
        // （"Selection and hiding of host entities is communicated to editor view plug-ins"），
        // 那正好发生在**打开编辑器**的那一刻 —— 实测崩点就在
        // `IEditController::createView` 之后 7ms，`araservice.dll` 内读空指针。
        //
        // 库那边早就实现好了（`ARAEditorViewInterface` 的 notifySelection /
        // notifyHideRegionSequences），缺的只是我们这一句声明。
        let supported = ExtensionRoles::PLAYBACK_RENDERER
            | ExtensionRoles::EDITOR_RENDERER
            | ExtensionRoles::EDITOR_VIEW;
        let enabled = ExtensionRoles::resolve(known, assigned, supported)?;
        let owned = ExtensionBinding::new_with_renderer_observer(
            generation, known, assigned, supported, observer,
        )?;
        let raw = owned.0.as_raw();
        document.attach(self, owned.1, companion)?;
        *self.document.lock_or_recover() = Some(Arc::downgrade(&document));
        let weak = Arc::downgrade(self);
        // 【为什么不再只对 playback 角色初始化】文档上没有任何 playback 实例时，编辑实例会
        // **接管渲染**（见 [`ExtensionOwner::is_editor_only`]），那时它也必须有这条准备队列 ——
        // 只对 playback 初始化会让接管的实例永远排不出任务（`prepare()` 的 `None => {}`
        // 分支会静默吞掉），于是"接管"看起来生效了、声音却一直不更新。
        if !enabled.is_empty() {
            // 【为什么是 ensure 而不是 get_or_init】见 `preparation` 字段说明：
            // 上一份文档关闭时队列已被 close，重绑必须拿到一条**新**队列。
            self.ensure_preparation();
        }
        // 只登记weak供宿主回调排队；后台计算不形成owner自循环。
        *self.prepare_owner.lock_or_recover() = Some(weak);
        let _ = self.clock.set(document.clock.clone());
        static NEXT_WRITER: AtomicU64 = AtomicU64::new(1);
        self.writer_id.store(
            NEXT_WRITER.fetch_add(1, Ordering::Relaxed),
            Ordering::Release,
        );
        *current = Some(owned.0);
        self.bound_document_id.store(document.id, Ordering::Release);
        // 【T-B0-1】账本里的 `bound` 此前没有任何写者，于是它恒为 false —— 读数在说谎。
        if let Some(ledger) = document.graph_ledger() {
            ledger.note_bound();
        }
        // 绑定成功即置"曾经绑定过"。此后 `editor_document()` 的措辞走 DOCUMENT_CLOSED
        // 而不是 ARA_NOT_BOUND —— 两者对用户的含义完全不同（见该访问器的说明）。
        self.bound_once.store(true, Ordering::Release);
        // 形态也随之确定：这个实例永远走 ARA 通路（`mode()` 先看 `bound_once`，
        // 这里再写原子是为了让不查 `bound_once` 的读者也拿到一致读数）。
        self.note_ara_bound();
        self.roles.store(enabled.bits(), Ordering::Release);
        self.role.store(
            if enabled.contains(ExtensionRoles::PLAYBACK_RENDERER) {
                1
            } else {
                2
            },
            Ordering::Release,
        );
        drop(current);
        // 【为什么放在最后】这份读数要看到**全部**实例，包括本实例刚写进去的
        // `writer_id`。角色集合只有两个时刻会变：绑定（这里）与分配变化
        // （见 `new_with_renderer_observer` 的观察器），两处都要报。
        document.report_playback_ownership();
        // 【新文档必须从一次完整枚举开始】`ui_tracks` / `region_items` 是文档级
        // 缓存，重绑后是一份空表；只靠 20ms 定时器要等到节流窗口过期才会填。
        // 这里同步采一次，界面在绑定成功的当帧就能看到 item 与授权媒体。
        for renderer in document.renderer_owners() {
            renderer.refresh_host_geometry();
            renderer.refresh_ui_inventory();
            renderer.prepare();
        }
        // 【T8：把图读数打出来】绑定成功的那一刻宿主的图通常还没建（region 在之后才到），
        // 所以这一行几乎总是全零 —— 它的价值在于**当它停在零**：下一次重启报障时，
        // 若这里 ctx=0，说明宿主要么没发图、要么被我们拒了，不必再从位掩码反推。
        log::info!("[ara] {} (document={})", graph_summary_of(&document), document.id);
        Ok(raw)
    }

    /// 文档侧已撤销 lease；保留 binding 以供宿主最后几次合法移除/释放回调访问。
    pub fn document_closed(&self) {
        // 【为什么不是 `stop_editor`】文档结束不等于组件结束：宿主复用同一个
        // 处理器重开工程时，原生扩展适配器必须保留，否则重绑后永远枚举不到
        // 宿主 item（见 `stop_editor_for_document` 的说明）。
        self.stop_editor_for_document();
        // 【为什么这里要重算覆盖】本实例退出后，"文档上还有没有 playback 实例"可能翻转
        // （播放实例先死、留下编辑实例）。`stop_editor` 已把 `closed` 置真，所以
        // `renderer_owners()` 不再包含自己 —— 重算后留下的那个实例会接管渲染，
        // 而不是整档静音。
        let document = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade);
        if let Some(document) = document {
            document.report_playback_ownership();
        }
        self.assignments.lock_or_recover().clear();
        self.sequences.lock_or_recover().clear();
        self.document.lock_or_recover().take();
        self.snapshots.iter().for_each(|snapshot| snapshot.clear());
        // 【刻意不清 `bound_once`】绑定过又关闭就是真的关闭了，措辞必须保持
        // DOCUMENT_CLOSED。清掉它会让 "document closed" 静默退化成
        // ARA_NOT_BOUND（"宿主从未绑定"）—— 那是在陈述一件没发生过的事。
    }

    /// 模型线程展开当前 renderer 的显式区域与 editor sequence，并去重，拒绝跨文档身份。
    pub fn assigned_regions(&self) -> Result<Vec<u64>, AraError> {
        let document = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .ok_or(AraError::InvalidState("document no longer available"))?;
        let role = self.role.load(Ordering::Acquire);
        // 【T1：与枚举/几何路径共用同一个判据】此前这里是第二份实现，且用的是
        // **整批** `resolve` —— 宿主在一次分割里把已销毁父段的 key 一并重发时，
        // 一条陈旧 key 就让整份 scope 失败（→ `ensure_loaded` 整份 Err → 界面回落到
        // 只有宿主清单占位的降级视图）。现在两条路径共用 `assigned_scope`：
        // 逐条过滤、不整批作废。
        //
        // 【为什么这里**不**允许文档级派生】渲染输入是**逐实例**的：REAPER 的一个 ARA
        // 文档上挂着多条轨道的实例，派生出来的并集会让同一段音频被渲染两次。
        // 派生只服务并集语义的调用方（工作区、几何、传输）。
        self.assigned_scope(&document, Some(role as u32), false)
            .map(|scope| scope.keys)
            .map_err(|error| AraError::InvalidState(error.leak()))
    }

    /// 宿主模型callback只撤销/排最新准备任务，不在主线程执行WORLD或PCM混音。
    ///
    /// 【为什么 `fetch_add` 必须在 `renders_playback()` 之后】计数要衡量的是
    /// "播放实例收到了多少次通知" —— 这正是 `render_input_invalidation_…`
    /// 那条不变量测试锁的东西（此前漏掉播放实例的回归就在这里）。放在前面
    /// 会把编辑器实例的 no-op 也数进去，测试就再也分不出"通知了谁"。
    /// 为 Capture 形态准备渲染基础设施。
    ///
    /// 【为什么必须有这一步】`prepare()` 依赖 `prepare_owner` 与 `preparation`，而这两样
    /// 此前**只在 ARA 绑定时**建立（`bind_to_document`）。Capture 实例从不绑定，于是
    /// 它的 `prepare()` 会在第一个早退处静默返回 —— 录制正常、界面正常、就是没有声音。
    /// 这类"线都接好了但没插电"的故障最难归因，所以这里显式补上那部分与 ARA 无关的
    /// 初始化。
    ///
    /// 【为什么可以重复调用】三处都是 `OnceLock` / 覆盖式赋值，重复调用无副作用；
    /// 它由 20ms 的 UI 定时器驱动，因此必须幂等。
    pub(crate) fn arm_capture_rendering(self: &Arc<Self>) {
        self.init_block_trace();
        self.ensure_preparation();
        *self.prepare_owner.lock_or_recover() = Some(Arc::downgrade(self));
        // 时钟：Capture 没有 ARA 文档的共享时钟，自己起一个。宿主播放位置由
        // `audio_process_inner` 的 `observe_transport` 照常写入（它与 ARA 无关）。
        let _ = self
            .clock
            .set(Arc::new(super::transport::TransportClock::default()));
        static NEXT_WRITER: AtomicU64 = AtomicU64::new(1);
        self.writer_id.store(
            NEXT_WRITER.fetch_add(1, Ordering::Relaxed),
            Ordering::Release,
        );
        // 【为什么在这里也要把会话立起来】处理结果的**唯一**生产者是会话 actor
        // （`start_capture_render` → `capture_prepare_job`），而会话此前只在
        // `webview.rs` 的命令分发里惰性创建 —— 于是"从未打开过插件窗口"的会话里
        // 没有任何东西会把带曲线的渲染排出去：录制正常、时间轴有内容、
        // **播放时永远直通宿主原声**（用户报障）。
        // `setupProcessing` 是 SDK 保证在任何 `process` 之前、且在宿主 UI 线程上被
        // 调用的唯一时刻；actor 由会话创建，与窗口开没开无关，所以建在这里就够了。
        //
        // 【为什么只在有素材时才建】会话会起一个线程、初始化模型并建临时目录
        // （`EditorSession::new_with`）。一个插在母线上、用户早已忘记的实例不该在
        // 每次 `setupProcessing` 上都付这份代价；而没有素材时它也没有任何东西可渲染。
        if self.capture.has_material() {
            if let Ok(session) = self.clone().capture_session() {
                // 让 actor 下一轮加载时间线并排一次渲染（幂等：已加载时它只推票据）。
                session.request_reload();
            }
        }
    }

    pub fn prepare(&self) {
        // 【B0】先报告上一批可疑块。放在早退**之前**：Capture 实例在没有素材时
        // `renders()` 为假，而那正是"刚录完、播放中"这个最需要读数的时刻。
        self.log_click_report();
        self.log_capture_report();
        if !self.renders() {
            return;
        }
        // 源/assignment撤销在文档事务内清理；此函数只排队，不能在事务外clear旧任务刚发布的音频。
        #[cfg(test)]
        self.prepare_requests.fetch_add(1, Ordering::AcqRel);
        let weak = self.prepare_owner.lock_or_recover().clone();
        let Some(weak) = weak else {
            return;
        };
        if let Some(worker) = self.preparation() {
            if let Err(error) = worker.request(Box::new(move |cancel| {
                weak.upgrade()
                    .ok_or("renderer closed".to_owned())?
                    .prepare_job(cancel)
            })) {
                log::warn!("[ara] preparation unavailable: {error}");
            }
        }
    }
    /// 调用者持document事务；完整两输出率发布成功后才登记，不额外持有源或快照。
    pub(crate) fn record_prepared(
        &self,
        model: u64,
        edit: u64,
        epoch: u64,
        scope: u64,
        keys: Vec<u64>,
    ) {
        *self.prepared.lock_or_recover() = Some(PreparedVersion {
            model,
            edit,
            epoch,
            scope,
            keys,
        });
        if let Some(queue) = self.preparation() {
            queue.acknowledge_idle_success();
        }
    }
    /// 仅SDK规定UI线程的kOffline setup调用；不在事务锁内等待，不把旧快照当最新版本。
    pub(crate) fn prepare_offline_until(&self, deadline: std::time::Instant) -> Result<(), String> {
        // 【为什么 Capture 要在这里同步准备】bounce / 导出是 Capture 的**第二条通路**，
        // 而它是宿主跑得最快的一次 —— 插件没有第二次机会。这里必须同步把结果准备好，
        // 否则产物里是静音。`deadline` 对它无意义：没有文档事务可等，合成本身有界。
        // 【为什么判据与 `prepare_job` 同一份】离线只跑一趟，没有第二次机会，所以这里
        // 更不能因为"窗口没开过 ⇒ 形态未落定"就把静音烘焙进产物。与实时通路共用
        // "未绑定 + 有素材"这条证据（理由见 `prepare_job` 的同名分支）。
        if !self.is_bound() && self.capture.has_material() {
            return self.capture_prepare_job(Arc::new(std::sync::atomic::AtomicBool::new(false)));
        }
        if !self.renders_playback() {
            return Ok(());
        }
        // SDK离线setup在UI线程；无GUI时也刷新真实item状态，不能依赖WebView timer。
        self.refresh_host_geometry();
        let document = self.editor_document()?;
        // 【为什么需要尝试次数上限】若后台准备**成功返回**却仍不能收敛到当前版本
        // （例如 ready 永远不置位），此前会一路空转到 deadline，把宿主线程白等满
        // `OFFLINE_PREPARE_TIMEOUT`。同一个版本连续排队两次仍不匹配就是不可能收敛，
        // 提前返回明确错误，让上层如实记录，而不是静默冻结 15 s。
        let mut attempts = 0usize;
        loop {
            {
                let _transaction = document.transaction.lock_or_recover();
                if self.is_closed() || !document.is_alive() {
                    return Err("offline renderer closed".into());
                }
                if !document.ready.load(Ordering::Acquire) {
                    return Err("offline host model/PCM not ready".into());
                }
                let keys = self.assigned_regions().map_err(|e| e.to_string())?;
                let version = PreparedVersion {
                    model: document.revision.load(Ordering::Acquire),
                    edit: document.edits.lock_or_recover().revision,
                    epoch: document.render_epoch.load(Ordering::Acquire),
                    scope: document.scope_revision.load(Ordering::Acquire),
                    keys,
                };
                if self.prepared.lock_or_recover().as_ref() == Some(&version)
                    && self.snapshots.iter().all(|snapshot| snapshot.is_ready())
                {
                    return Ok(());
                }
            }
            if std::time::Instant::now() >= deadline {
                return Err("offline preparation timed out".into());
            }
            let worker = self
                .preparation()
                .ok_or("offline preparation worker missing")?;
            if !worker.state().0 {
                attempts += 1;
                if attempts > 2 {
                    return Err(
                        "offline preparation did not converge; the current edit cannot be \
                         rendered (host model/PCM or region assignment keeps changing)"
                            .into(),
                    );
                }
                self.prepare();
            }
            worker.wait_idle_until(deadline)?;
        }
    }
    /// 后台冻结/计算/发布三阶段；撤销标记和全部版本在发布短事务内重新验证。
    fn prepare_job(&self, cancel: Arc<std::sync::atomic::AtomicBool>) -> Result<(), String> {
        // Capture 形态没有 ARA 文档；在取文档**之前**分流，否则它会以
        // "document closed" 收场，而那条路对 Capture 是常态而非错误。
        //
        // 【为什么判据是"未绑定 + 有素材"而不是 `mode()`】形态由 `note_editor_tick`
        // 落定，而它的唯一调用点是 WebView 的 20ms 定时器 —— **窗口不开就没有这个
        // tick**，而且即便开了也要等 `setActive` 之后 5 秒。于是宿主把插件当普通插入、
        // 用户录完关掉窗口再播放的会话里 `mode()` 恒为 `Undetermined` ⇒ 这里永不分流
        // ⇒ 一条处理结果都没合成过 ⇒ 仲裁落到直通。
        // `renders()` 早就因为同一理由绕开了 `mode()`（见它的注释）；准入侧与产出侧
        // 必须问同一个问题，否则"该出声"与"还没到时候"会各自漂移。
        if !self.is_bound() && self.capture.has_material() {
            return self.capture_prepare_job(cancel);
        }
        let Some(document) = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
        else {
            return Err("document closed".into());
        };
        let (model, edit, epoch, scope, keys, input) = {
            let _transaction = document.transaction.lock_or_recover();
            if !document.ready.load(Ordering::Acquire) {
                return Ok(());
            }
            let edits = document.edits.lock_or_recover().clone();
            let (keys, input) = self.ara_render_input(&document, &edits, edits.revision > 0)?;
            // actor可能已发布本代两率结果；同版本重排队不应再争预算或留下伪失败状态。
            let version = PreparedVersion {
                model: document.revision.load(Ordering::Acquire),
                edit: edits.revision,
                epoch: document.render_epoch.load(Ordering::Acquire),
                scope: document.scope_revision.load(Ordering::Acquire),
                keys: keys.clone(),
            };
            if self.prepared.lock_or_recover().as_ref() == Some(&version)
                && self.snapshots.iter().all(|snapshot| snapshot.is_ready())
            {
                return Ok(());
            }
            (
                document.revision.load(Ordering::Acquire),
                edits.revision,
                document.render_epoch.load(Ordering::Acquire),
                document.scope_revision.load(Ordering::Acquire),
                keys,
                input,
            )
        };
        for publisher in &self.snapshots {
            publisher.collect_retired();
        }
        let mut snapshots = input.render(cancel.clone())?;
        super::snapshot::compact_prepared(&mut snapshots)?;
        let _transaction = document.transaction.lock_or_recover();
        if cancel.load(Ordering::Acquire) || !document.ready.load(Ordering::Acquire) {
            return Ok(());
        }
        if document.revision.load(Ordering::Acquire) != model
            || document.edits.lock_or_recover().revision != edit
            || document.render_epoch.load(Ordering::Acquire) != epoch
            || document.scope_revision.load(Ordering::Acquire) != scope
            || self.assigned_regions().map_err(|e| e.to_string())? != keys
        {
            // 跨轨接受新参数不一定有新的model callback；丢弃后补排最新任务，不能永久留空且假idle。
            self.prepare();
            return Ok(());
        }
        if !self
            .snapshots
            .iter()
            .zip(&snapshots)
            .all(|(p, s)| p.has_capacity(s))
        {
            return Err("retired snapshot budget exhausted".into());
        }
        for (publisher, snapshot) in self.snapshots.iter().zip(snapshots) {
            publisher
                .publish(snapshot)
                .map_err(|e| format!("snapshot publish failed: {e:?}"))?;
        }
        self.record_prepared(model, edit, epoch, scope, keys.clone());
        log::info!(
            "[ara] background snapshot ready role={} revision={edit} model={model} regions={}",
            self.role.load(Ordering::Relaxed),
            keys.len()
        );
        Ok(())
    }
}

/// 块级诊断环的容量。256 块在 48 kHz/512 帧下约 2.7 秒的实时历史。
const BLOCK_TRACE_CAPACITY: usize = 256;

/// 一条块级记录。字段全部是原子，读者用 seqlock 校验一致性。
#[derive(Default)]
struct BlockTraceRecord {
    seq: AtomicU64,
    state: AtomicU32,
    mode: AtomicI32,
    stream: AtomicU8,
    start: AtomicI64,
    position: AtomicI64,
    position_source: AtomicU8,
    frames: AtomicU32,
    playing: AtomicBool,
    playing_rising: AtomicBool,
    phase: AtomicU8,
    flow: AtomicBool,
    cycle_wrap: AtomicBool,
    split: AtomicU8,
    had_previous: AtomicBool,
    emitted: AtomicBool,
    capture: AtomicU8,
    capture_started: AtomicBool,
    rung: AtomicU8,
    // ── 宿主时间字段（`[capture] ctx` 用它回答"宿主到底给了什么"）──
    continuous: AtomicI64,
    continuous_valid: AtomicBool,
    music_bits: AtomicU64,
    music_valid: AtomicBool,
    /// 【为什么要 tempo】`[capture] ctx` 只打 `music=` 时，读者无法判断
    /// "Music 分支为什么没生效"：那一条要求 `music` 与 `tempo` **同时**有效。
    /// 现场日志里 `music=841.5(valid)` 连着几块不变，而 `counted=1` —— 只有
    /// 把 tempo 也打出来，才能一眼区分"宿主没给 tempo"与"music 冻结了"。
    tempo_bits: AtomicU64,
    tempo_valid: AtomicBool,
    /// `0` = 非循环；`1` = 循环且元数据完整；`2` = 循环但元数据不完整。
    cycle: AtomicU8,
    rate_bits: AtomicU64,
    /// 本块位置是否由**计数器**推进（见 `StreamBlock::counted`）。
    counted: AtomicBool,
}

/// 预分配的块级环形记录（SPSC：音频线程写、actor/诊断读）。
struct BlockTraceRing {
    head: AtomicU64,
    records: Vec<BlockTraceRecord>,
}

impl BlockTraceRing {
    fn new() -> Self {
        Self {
            head: AtomicU64::new(0),
            records: (0..BLOCK_TRACE_CAPACITY)
                .map(|_| BlockTraceRecord::default())
                .collect(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn write(
        &self,
        state: u32,
        mode: i32,
        stream: ProcessStream,
        raw_start: i64,
        position: i64,
        position_source: PositionSource,
        frames: usize,
        playing: bool,
        playing_rising: bool,
        phase: TransportPhase,
        cycle_wrap: bool,
        split: u8,
        had_previous: bool,
        emitted: bool,
        capture: u8,
        capture_started: bool,
        time: StreamTime,
        rate: f64,
        counted: bool,
    ) {
        let index = self.head.fetch_add(1, Ordering::Relaxed) as usize % self.records.len();
        let record = &self.records[index];
        record.seq.fetch_add(1, Ordering::Relaxed);
        record.state.store(state, Ordering::Relaxed);
        record.mode.store(mode, Ordering::Relaxed);
        record.stream.store(stream.code(), Ordering::Relaxed);
        record.start.store(raw_start, Ordering::Relaxed);
        record.position.store(position, Ordering::Relaxed);
        record
            .position_source
            .store(position_source.code(), Ordering::Relaxed);
        record
            .frames
            .store(u32::try_from(frames).unwrap_or(u32::MAX), Ordering::Relaxed);
        record.playing.store(playing, Ordering::Relaxed);
        record
            .playing_rising
            .store(playing_rising, Ordering::Relaxed);
        record.phase.store(phase.code(), Ordering::Relaxed);
        record
            .flow
            .store(phase != TransportPhase::Resting, Ordering::Relaxed);
        record.cycle_wrap.store(cycle_wrap, Ordering::Relaxed);
        record.split.store(split, Ordering::Relaxed);
        record.had_previous.store(had_previous, Ordering::Relaxed);
        record.emitted.store(emitted, Ordering::Relaxed);
        record.capture.store(capture, Ordering::Relaxed);
        record
            .capture_started
            .store(capture_started, Ordering::Relaxed);
        record.rung.store(0, Ordering::Relaxed);
        record
            .continuous
            .store(time.continuous.unwrap_or(i64::MIN), Ordering::Relaxed);
        record
            .continuous_valid
            .store(time.continuous.is_some(), Ordering::Relaxed);
        record
            .music_bits
            .store(time.music.unwrap_or(f64::NAN).to_bits(), Ordering::Relaxed);
        record.music_valid.store(time.music.is_some(), Ordering::Relaxed);
        record
            .tempo_bits
            .store(time.tempo.unwrap_or(f64::NAN).to_bits(), Ordering::Relaxed);
        record.tempo_valid.store(time.tempo.is_some(), Ordering::Relaxed);
        record.cycle.store(
            if time.cycle_active {
                1 + u8::from(!time.cycle_valid)
            } else {
                0
            },
            Ordering::Relaxed,
        );
        record.rate_bits.store(rate.to_bits(), Ordering::Relaxed);
        record.counted.store(counted, Ordering::Relaxed);
        record.seq.fetch_add(1, Ordering::Release);
    }

    fn update_last_rung(&self, rung: u8) {
        let head = self.head.load(Ordering::Relaxed);
        if head == 0 {
            return;
        }
        let index = (head - 1) as usize % self.records.len();
        let record = &self.records[index];
        record.seq.fetch_add(1, Ordering::Relaxed);
        record.rung.store(rung, Ordering::Relaxed);
        record.seq.fetch_add(1, Ordering::Release);
    }

    /// 最近一块的宿主上下文，格式化成 `[capture] ctx: …` 一行。
    ///
    /// 【为什么只有一块】它回答的是"宿主此刻给的是什么形状"，不是历史。需要历史时
    /// 用 `snapshot`（诊断包里的块级 trace）。
    fn last_context_line(&self) -> Option<String> {
        let head = self.head.load(Ordering::Acquire);
        if head == 0 {
            return None;
        }
        let index = (head - 1) as usize % self.records.len();
        let record = &self.records[index];
        // seqlock：读两遍一致才算稳定（音频线程可能正在写同一个槽）。
        let first = record.seq.load(Ordering::Acquire);
        if !first.is_multiple_of(2) {
            return None;
        }
        let read = || {
            (
                record.state.load(Ordering::Relaxed),
                record.mode.load(Ordering::Relaxed),
                record.stream.load(Ordering::Relaxed),
                record.start.load(Ordering::Relaxed),
                record.position.load(Ordering::Relaxed),
                record.frames.load(Ordering::Relaxed),
                record.phase.load(Ordering::Relaxed),
                record.playing.load(Ordering::Relaxed),
                record.playing_rising.load(Ordering::Relaxed),
                record.cycle_wrap.load(Ordering::Relaxed),
                record.had_previous.load(Ordering::Relaxed),
                record.continuous.load(Ordering::Relaxed),
                record.continuous_valid.load(Ordering::Relaxed),
                record.music_bits.load(Ordering::Relaxed),
                record.music_valid.load(Ordering::Relaxed),
                record.tempo_bits.load(Ordering::Relaxed),
                record.tempo_valid.load(Ordering::Relaxed),
                record.cycle.load(Ordering::Relaxed),
                record.rate_bits.load(Ordering::Relaxed),
                record.counted.load(Ordering::Relaxed),
                record.emitted.load(Ordering::Relaxed),
                record.capture.load(Ordering::Relaxed),
            )
        };
        let values = read();
        if record.seq.load(Ordering::Acquire) != first {
            return None;
        }
        let (
            state,
            mode,
            stream,
            raw_start,
            position,
            frames,
            phase,
            playing,
            playing_rising,
            cycle_wrap,
            had_previous,
            continuous,
            continuous_valid,
            music_bits,
            music_valid,
            tempo_bits,
            tempo_valid,
            cycle,
            rate_bits,
            counted,
            emitted,
            capture,
        ) = values;
        let rate = f64::from_bits(rate_bits);
        let music = f64::from_bits(music_bits);
        let tempo = f64::from_bits(tempo_bits);
        Some(format!(
            "[capture] ctx: stream={} mode={mode} state=0x{state:x} \
             raw={raw_start} pos={position} delta={} frames={frames} phase={phase} \
             playing={} rising={} cycle={} had_prev={} \
             cycle_meta={} cont={}{} music={}{} tempo={}{} rate={rate:.0} counted={} capture={} \
             emitted={}",
            ProcessStream::from_code(stream).map_or("?", |value| value.as_str()),
            position.saturating_sub(raw_start),
            u8::from(playing),
            u8::from(playing_rising),
            cycle_wrap,
            u8::from(had_previous),
            cycle,
            if continuous_valid { continuous } else { 0 },
            if continuous_valid { "(valid)" } else { "(absent)" },
            if music_valid { music } else { 0.0 },
            if music_valid { "(valid)" } else { "(absent)" },
            if tempo_valid { tempo } else { 0.0 },
            if tempo_valid { "(valid)" } else { "(absent)" },
            u8::from(counted),
            emitted,
            capture,
        ))
    }

    /// 最近的 `limit` 条记录（按写入顺序）。撕裂的记录直接跳过。
    fn snapshot(&self, limit: usize) -> Vec<serde_json::Value> {
        let head = self.head.load(Ordering::Acquire);
        let count = usize::try_from(head)
            .unwrap_or(usize::MAX)
            .min(self.records.len())
            .min(limit);
        let mut out = Vec::with_capacity(count);
        for offset in 0..count {
            let index = (head.saturating_sub(1 + offset as u64)) as usize % self.records.len();
            let record = &self.records[index];
            let first = record.seq.load(Ordering::Acquire);
            if first & 1 != 0 {
                continue;
            }
            let state = record.state.load(Ordering::Relaxed);
            let mode = record.mode.load(Ordering::Relaxed);
            let stream = record.stream.load(Ordering::Relaxed);
            let start = record.start.load(Ordering::Relaxed);
            let position = record.position.load(Ordering::Relaxed);
            let position_source = record.position_source.load(Ordering::Relaxed);
            let frames = record.frames.load(Ordering::Relaxed);
            let playing = record.playing.load(Ordering::Relaxed);
            let playing_rising = record.playing_rising.load(Ordering::Relaxed);
            let phase = record.phase.load(Ordering::Relaxed);
            let flow = record.flow.load(Ordering::Relaxed);
            let cycle_wrap = record.cycle_wrap.load(Ordering::Relaxed);
            let split = record.split.load(Ordering::Relaxed);
            let had_previous = record.had_previous.load(Ordering::Relaxed);
            let emitted = record.emitted.load(Ordering::Relaxed);
            let capture = record.capture.load(Ordering::Relaxed);
            let capture_started = record.capture_started.load(Ordering::Relaxed);
            let rung = record.rung.load(Ordering::Relaxed);
            if record.seq.load(Ordering::Acquire) != first {
                continue;
            }
            out.push(serde_json::json!({
                "seq": first / 2,
                "state": state,
                "mode": mode,
                "stream": stream,
                "start": start,
                "position": position,
                "position_source": position_source,
                "frames": frames,
                "playing": playing,
                "playing_rising": playing_rising,
                "phase": phase,
                "flow": flow,
                "cycle_wrap": cycle_wrap,
                "split": split,
                "had_previous": had_previous,
                "emitted": emitted,
                "capture": capture,
                "capture_started": capture_started,
                "rung": rung,
            }));
        }
        out.reverse();
        out
    }
}

impl ExtensionOwner {
    /// 预分配块级诊断环（非音频线程；幂等）。
    pub(crate) fn init_block_trace(&self) {
        let _ = self
            .block_trace
            .get_or_init(|| Box::new(BlockTraceRing::new()));
    }

    /// 记录 `setupProcessing` 声明的单块上限（诊断用）。
    pub(crate) fn note_max_samples_per_block(&self, value: i32) {
        self.max_samples_per_block
            .store(value.max(0), Ordering::Release);
    }

    /// 记一次**输出写满审计**（`HIFISHIFTER_BUFFER_AUDIT=1` 时由 vst3 层调用）。
    ///
    /// 【它回答什么】"这一块有没有哪一帧根本没被写过"。立体声下每条分支都写满整块，
    /// 所以这个数应当恒为 0；非零只有三种来源：>2 声道的总线（`pass_through` 只写前
    /// 两个平面）、短块、或者某条新分支漏写。三种都是"用户听到上一次的内容"。
    ///
    /// 【为什么按档位去重打印】它由音频线程记原子，格式化留给 actor；一次播放里同一档位
    /// 只报一次，避免每块一行。
    pub(crate) fn note_buffer_audit(&self, channels: usize, unwritten: usize, frames: usize) {
        self.buffer_audit_channels.store(channels as u32, Ordering::Relaxed);
        self.buffer_audit_frames.store(frames as u32, Ordering::Relaxed);
        self.buffer_audit_unwritten
            .fetch_max(unwritten as u64, Ordering::Relaxed);
    }

    /// 把写满审计的结论打成一行（非实时线程）。
    pub(crate) fn log_buffer_audit(&self) {
        if !self.buffer_audit_active.swap(true, Ordering::AcqRel) {
            // 第一次调用只置位：没有读数可报。
            return;
        }
        let unwritten = self.buffer_audit_unwritten.load(Ordering::Acquire);
        let line = format!(
            "[buf] audit: channels={} frames={} unwritten={unwritten} output={}",
            self.buffer_audit_channels.load(Ordering::Acquire),
            self.buffer_audit_frames.load(Ordering::Acquire),
            self.capture_output().as_str(),
        );
        let mut last = self.last_buffer_diag.lock_or_recover();
        if last.as_deref() != Some(line.as_str()) {
            *last = Some(line.clone());
            if unwritten > 0 {
                log::warn!("{line}");
            } else {
                log::info!("{line}");
            }
        }
    }

    /// 门控清掉一块（诊断用）。按流分开计数：prefetch/offline 的清零必须是 0。
    pub(crate) fn note_gate_cleared(&self, stream: ProcessStream) {
        self.gate_cleared.fetch_add(1, Ordering::Relaxed);
        self.stream_gate_cleared[stream.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// ARA 实例因快照未就绪而退回宿主原声（见 `capture_stage` 的 `Untouched` 分支）。
    pub(crate) fn note_bound_snapshot_fallback(&self, fell_back: bool) {
        if fell_back {
            self.bound_snapshot_fallback.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// 本块最终走了哪一档输出仲裁（由 `capture_stage` 在决定后调用）。
    ///
    /// 【I-2 的读数在这里攒】"一次播放只有一个内容源"是可以直接数出来的：只要
    /// 相邻两块之间的档位不同，那一次切换就可能在听感上表现为一次咔。音频线程只做
    /// 一次整数比较与两次原子写；报告留给非实时线程（见 `log_capture_report`）。
    pub(crate) fn note_output_rung(&self, output: CaptureOutput) {
        if let Some(ring) = self.block_trace.get() {
            ring.update_last_rung(output.code());
        }
        let code = u32::from(output.code());
        let previous = self.rung_previous.swap(code, Ordering::AcqRel);
        // `0` 是 `Untouched` 的编码，也是"从未记录"的初值 —— 用独立的 valid 位区分，
        // 否则首块会被算成一次切换（与 `apply_output_declick` 里 `primed` 的教训相同）。
        if !self.rung_valid.swap(true, Ordering::AcqRel) {
            return;
        }
        if previous != code {
            self.rung_switches.fetch_add(1, Ordering::Relaxed);
            self.rung_last_from.store(previous, Ordering::Release);
            self.rung_last_to.store(code, Ordering::Release);
            // 起播/换 take 会让"这一次播放"重新开始计数：切换数归零由非实时线程
            // 依据 `pass_epoch` 完成（音频线程不能分配、也不该做字符串格式化）。
            self.pass_switches.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// 取走“该落一次块级 trace”的意图（非音频线程）。
    pub(crate) fn take_trace_wanted(&self) -> bool {
        let wanted = self.trace_wanted.swap(false, Ordering::AcqRel);
        if wanted {
            self.trace_logged.store(true, Ordering::Release);
        }
        wanted
    }

    /// 音频线程：计数并写入一条块级记录。无锁、无分配。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn note_block_trace(
        &self,
        state: u32,
        mode: i32,
        stream: ProcessStream,
        raw_start: i64,
        position: i64,
        frames: usize,
        playing: bool,
        phase: TransportPhase,
        playing_rising: bool,
        cycle_wrap: bool,
        split: u8,
        had_previous: bool,
        allow_output: bool,
        capture: u8,
        capture_started: bool,
        time: StreamTime,
        rate: f64,
        counted: bool,
    ) {
        self.blocks_seen.fetch_add(1, Ordering::Relaxed);
        self.stream_blocks[stream.index()].fetch_add(1, Ordering::Relaxed);
        if playing {
            self.blocks_playing.fetch_add(1, Ordering::Relaxed);
            self.stream_playing_blocks[stream.index()].fetch_add(1, Ordering::Relaxed);
        }
        let flow = phase != TransportPhase::Resting;
        if flow && !playing {
            self.blocks_flow_without_playing
                .fetch_add(1, Ordering::Relaxed);
            self.stream_flow_without_playing[stream.index()].fetch_add(1, Ordering::Relaxed);
        }
        // 只在确实与 Capture 有关的实例上留证，避免给纯 ARA 实例刷日志。
        // 同一次可疑窗口只报一次：`trace_logged` 由 actor 在 dump 后置位，
        // 直到宿主回到“钉住/停播”才复位，下一次起播再报。
        let suspicious = (!playing && flow) || raw_start < 0;
        if (self.capture.is_armed() || self.capture.has_material())
            && suspicious
            && !self.trace_logged.load(Ordering::Relaxed)
        {
            self.trace_wanted.store(true, Ordering::Release);
        }
        if !playing && !flow && (self.capture.is_armed() || self.capture.has_material()) {
            self.trace_logged.store(false, Ordering::Release);
        }
        if let Some(ring) = self.block_trace.get() {
            let source = self
                .streams
                .get(stream.index())
                .and_then(|geometry| {
                    PositionSource::from_code(geometry.last_source.load(Ordering::Relaxed))
                })
                .unwrap_or(PositionSource::Project);
            ring.write(
                state,
                mode,
                stream,
                raw_start,
                position,
                source,
                frames,
                playing,
                playing_rising,
                phase,
                cycle_wrap,
                split,
                had_previous,
                allow_output,
                capture,
                capture_started,
                time,
                rate,
                counted,
            );
        }
    }

    /// 块级诊断的 JSON（非音频线程）。`trace_limit` 控制环形记录条数。
    pub(crate) fn block_diagnostics_json(&self, trace_limit: usize) -> serde_json::Value {
        let capture = self.capture.capture_diagnostics().map(|capture| {
            serde_json::json!({
                "preroll_frames_dropped": capture.preroll_frames_dropped,
                "pending_stashed": capture.pending_stashed,
                "pending_committed": capture.pending_committed,
                "pending_discarded": capture.pending_discarded,
                "pending_ready": capture.pending_ready,
                "pending_available": capture.pending_available,
                "gaps": capture.gaps,
                "backwards": capture.backwards,
                "dropouts": capture.dropouts,
                "oversized_blocks": capture.oversized_blocks,
                "silent_blocks": capture.silent_blocks,
                "spares": capture.spared,
                "elected": capture.elected.map(|stream| stream.as_str()),
                "stream_switches": capture.stream_switches,
                "repeated_blocks": capture.repeated_blocks,
                "overlapped_frames": capture.overlapped_frames,
                "resting_blocks": capture.resting_blocks,
                "resting_frames": capture.resting_frames,
                "overwritten_blocks": capture.overwritten_blocks,
                "overwritten_frames": capture.overwritten_frames,
                "overhang_trimmed_frames": capture.overhang_trimmed_frames,
                "cycle_splits": capture.cycle_splits,
                "resync_bridged_frames": capture.resync_bridged_frames,
                // 起播窗口整段都没写下、由 resync 兜底补出一个段的次数。
                // T1 让窗口内记账连续后这个数应当长期为 0；只要它 > 0，说明暂存/
                // 提交链路上仍有漏，但 T2 保证了时间线不断。
                "bridged_window_blocks": capture.bridged_window_blocks,
                // 起播前导静音：捕获起点晚于播放起点时，这一段被合成进段首。
                "pre_roll_frames": capture.pre_roll_frames,
                // T19：旁路队列。竞争时暂存、下一次进锁时补写；`dropped > 0` 说明
                // 槽全满而竞争又在持续 —— 那时快照里会有洞。
                "bypass_pending": self.capture.bypass_pending(),
            })
        });
        let stream = |index: usize| {
            serde_json::json!({
                "blocks": self.stream_blocks[index].load(Ordering::Acquire),
                "playing": self.stream_playing_blocks[index].load(Ordering::Acquire),
                "flow_without_playing": self.stream_flow_without_playing[index]
                    .load(Ordering::Acquire),
                "gate_cleared": self.stream_gate_cleared[index].load(Ordering::Acquire),
                "resting": self.stream_resting_blocks[index].load(Ordering::Acquire),
                "rolling": self.stream_rolling_blocks[index].load(Ordering::Acquire),
                // 【位置记账的四个数】`counted` = 位置由计数器推进的块数；
                // `jumps` = 宿主报的位置与计数器不同的次数；`rewinds` = 位置倒退的
                // 块数（稳态恒为 0）；`passes` = 这一流经历过几次"新一次播放"。
                "counted": self.streams[index]
                    .counted_blocks
                    .load(Ordering::Acquire),
                "counted_frames": self.streams[index]
                    .counted_frames
                    .load(Ordering::Acquire),
                "jumps": self.streams[index].jumps.load(Ordering::Acquire),
                "last_jump": self.streams[index].last_jump.load(Ordering::Acquire),
                "rewinds": self.streams[index].clock_rewinds.load(Ordering::Acquire),
                "passes": self.streams[index].pass.load(Ordering::Acquire),
            })
        };
        // 【位置记账的汇总读数】三条流加起来：位置由计数器推进的块数、宿主纠正的次数、
        // 以及"这一流经历过几次新一次播放"。
        let (counted_blocks, jumps, passes) = self.streams.iter().fold(
            (0u64, 0u64, 0u64),
            |(counted, jumps, passes), geometry| {
                (
                    counted.saturating_add(geometry.counted_blocks.load(Ordering::Acquire)),
                    jumps.saturating_add(geometry.jumps.load(Ordering::Acquire)),
                    passes.saturating_add(geometry.pass.load(Ordering::Acquire)),
                )
            },
        );
        serde_json::json!({
            "blocks_seen": self.blocks_seen.load(Ordering::Acquire),
            "blocks_playing": self.blocks_playing.load(Ordering::Acquire),
            "blocks_flow_without_playing": self.blocks_flow_without_playing.load(Ordering::Acquire),
            "gate_cleared": self.gate_cleared.load(Ordering::Acquire),
            "bound_snapshot_fallback": self.bound_snapshot_fallback.load(Ordering::Acquire),
            "continuous_used": self.continuous_used.load(Ordering::Acquire),
            "music_used": self.music_used.load(Ordering::Acquire),
            "counted_blocks": counted_blocks,
            "jumps": jumps,
            "passes": passes,
            "position_resyncs": self.position_resyncs.load(Ordering::Acquire),
            "bound_fallback": {
                "snapshot_missing_blocks": self.bound_snapshot_fallback.load(Ordering::Acquire),
                "promoted_to_renderer": self
                    .promoted_to_renderer
                    .load(Ordering::Acquire),
                "rebind_count": self.rebind_generation.load(Ordering::Acquire),
                // companion 的一次性绑定被复位重建了几次。重开工程正常是 1；
                // 持续增长说明宿主在反复重绑。
                "companion_resets": self.companion_reset_count(),
            },
            // 【为什么恢复读数要进诊断包】恢复失败此前只有一行 `log::warn!`，
            // 用户无法回答"参数到底读回来没有"。这里的三个字段就是那个答案：
            // 0 = 从未尝试、1 = 已应用、2 = 待文档就绪/重试中、3 = 失败。
            "restore": {
                "status": self.restore_status.load(Ordering::Acquire),
                "error": self.restore_error.lock_or_recover().clone(),
                "applied_generation": self
                    .restore_applied_generation
                    .load(Ordering::Relaxed),
            },
            "clock": {
                "counted_blocks": counted_blocks,
                "jumps": jumps,
                "passes": passes,
            },
            "cycle_wraps": self.cycle_wraps.load(Ordering::Acquire),
            "cycle_wrap_unknown": self.cycle_wrap_unknown.load(Ordering::Acquire),
            "cycle_splits": self.cycle_splits.load(Ordering::Acquire),
            "streams": {
                "realtime": stream(ProcessStream::Realtime.index()),
                "prefetch": stream(ProcessStream::Prefetch.index()),
                "offline": stream(ProcessStream::Offline.index()),
            },
            "repeats_suppressed": self.repeats_suppressed.load(Ordering::Acquire),
        // T3：读游标纠正。跳变幅度是"起播 0.4~0.7 s 一次咔"最直接的量。
        "position_corrections": self.position_corrections.load(Ordering::Acquire),
        "max_correction_frames": self.max_correction_frames.load(Ordering::Acquire),
            "max_samples_per_block": self.max_samples_per_block.load(Ordering::Acquire),
            "capture": capture,
            "trace": self
                .block_trace
                .get()
                .map(|ring| ring.snapshot(trace_limit))
                .unwrap_or_default(),
        })
    }
}

/// 取宿主缓冲中属于某个循环切分段的那一片。
fn slice_plane<'a>(plane: Option<&'a [f32]>, part: &SubBlock) -> Option<&'a [f32]> {
    plane.and_then(|samples| {
        let end = part.input_offset.checked_add(part.frames)?;
        samples.get(part.input_offset..end)
    })
}

/// 合并循环切分两段的捕获结论：只要有一段真的写了，就报 `Capturing`。
fn combine_capture_outcomes(
    first: crate::capture::CaptureBlockOutcome,
    second: crate::capture::CaptureBlockOutcome,
) -> crate::capture::CaptureBlockOutcome {
    use crate::capture::CaptureBlockOutcome;
    if first == CaptureBlockOutcome::Capturing || second == CaptureBlockOutcome::Capturing {
        return CaptureBlockOutcome::Capturing;
    }
    if first != CaptureBlockOutcome::Idle {
        first
    } else {
        second
    }
}

#[cfg(test)]
mod assigned_timeline_tests;
#[cfg(test)]
mod bound_tests;
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
#[cfg(test)]
mod capture_output_tests;
