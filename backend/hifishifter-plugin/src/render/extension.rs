//! 每个 VST3 entry 的扩展所有权。强引用由 entry builder 保留，不在组件销毁时悬空。

use super::ownership::{region_owners, RegionKey};
use crate::sync::MutexExt;
use ara2_bridge::core::{ApiGeneration, AraError};
use ara2_bridge::plugin::{ExtensionBinding, ExtensionRoles};
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// 全PCM内容参与身份，重开时同路径不同音频不能命中旧缓存。
pub(super) fn pcm_fingerprint(pcm: &super::source::SourcePcm) -> String {
    let mut hash = blake3::Hasher::new();
    hash.update(&pcm.sample_rate.to_le_bytes());
    hash.update(&(pcm.planes.len() as u64).to_le_bytes());
    for plane in &pcm.planes {
        for sample in plane {
            hash.update(&sample.to_le_bytes());
        }
    }
    hash.finalize().to_hex().to_string()
}

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
}

impl HostAudioState {
    /// 语言无关的分类名；前端按此查 catalog（与 `ara_host_fields:` 同一原则）。
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            HostAudioState::Ready => "ready",
            HostAudioState::AwaitingRegions => "awaiting_regions",
            HostAudioState::FolderParentWithoutRegions => "folder_parent_without_regions",
            HostAudioState::UnmatchedRegions => "unmatched_regions",
        }
    }
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

/// 原生接口与只读元数据属于真实组件，缓存不跨其文档/分配/host变更复用。
#[derive(Default)]
pub(crate) struct ExtensionOwner {
    binding: Mutex<Option<ExtensionBinding>>,
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
    pub(crate) edits: Arc<Mutex<crate::state_channel::EditState>>,
    pending_restore: Mutex<Option<crate::state_channel::EditState>>,
    channel: Mutex<Option<hifishifter_ara_ipc::Server>>,
    preparation: std::sync::OnceLock<Result<super::preparation::PreparationQueue, String>>,
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
    /// 宿主调用 `setActive(true)` 的时刻；`None` = 还没激活。
    ///
    /// 【为什么必须记下来】ARA 规定 `bindToDocumentControllerWithRoles` 只能在
    /// `setActive` **之前**调用。宿主一旦激活组件就再也不会给这个实例绑定机会 ——
    /// 那一刻之后仍无绑定，就是"宿主把本插件当普通插入效果器"的确证。此前这件事
    /// 只能靠读者对比两个会话的日志才能推断出来（见 `note_editor_tick`）。
    set_active_at: Mutex<Option<std::time::Instant>>,
    /// "激活后仍无 ARA 绑定"这条诊断是否已经报过（一次性）。
    ara_binding_absent_reported: std::sync::atomic::AtomicBool,
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
    pub(crate) fn note_editor_tick(&self) {
        if self.is_ara_bound() {
            return;
        }
        let Some(at) = *self.set_active_at.lock_or_recover() else {
            // 宿主还没激活：绑定窗口仍然开着，任何结论都是早的。
            return;
        };
        let elapsed = at.elapsed();
        if elapsed < ARA_BINDING_GRACE {
            return;
        }
        if self
            .ara_binding_absent_reported
            .swap(true, Ordering::AcqRel)
        {
            return;
        }
        let backend = self.host_backend();
        let host = backend.app_name().unwrap_or("this host");
        log::warn!(
            "[ara] no ARA binding {:.0}s after setActive: {host} is running this instance as a \
             plain insert effect (no document controller was created, so no clip can be edited). \
             Insert the plug-in from the host's ARA entry rather than a regular effect slot.",
            elapsed.as_secs_f32()
        );
    }

    /// 初始化在宿主UI/model线程保存有引用的typed扩展，不在actor或process调用REAPER API。
    /// # Safety
    /// context须为宿主初始化期间活FUnknown。
    pub(crate) unsafe fn bind_host_adapter(&self, context: *mut std::ffi::c_void) {
        let bound = self.document.lock_or_recover().is_some();
        let stamp = if bound {
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
                    None => self.document.lock_or_recover().is_none(),
                }
        };
        let host = unsafe { crate::host::NativeHost::from_context(context, authorized) }
            .map(|host| crate::host::adapter::HostBackend::Native(Arc::new(host)));
        if !authorized() {
            return;
        }
        crate::log_line(&format!(
            "native host extension available={}",
            host.is_some()
        ));
        let old = std::mem::replace(&mut *self.host.lock_or_recover(), host);
        drop(old);
        self.host_geometry.lock_or_recover().take();
    }
    /// 短事务冻结真实owner/document/model/scope；整个host调用链都不持内部锁。
    fn host_query_stamp(
        &self,
    ) -> Result<(Arc<super::document::DocumentSession>, u64, u64, Vec<u64>), String> {
        let document = self.editor_document()?;
        let _transaction = document.transaction.lock_or_recover();
        if self.is_closed() || !document.is_alive() {
            return Err("host query document/owner closed".into());
        }
        let keys = self.host_assigned_regions(&document)?;
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
    /// 几何绑定必须覆盖owner的全部已分配角色，不能只拿playback子集隐藏editor歧义。
    fn host_assigned_regions(
        &self,
        document: &super::document::DocumentSession,
    ) -> Result<Vec<u64>, String> {
        let mut keys = self
            .assignments
            .lock_or_recover()
            .values()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>();
        let sequences = self
            .sequences
            .lock_or_recover()
            .values()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>();
        let members = document.sequence_regions.lock_or_recover();
        for sequence in sequences {
            keys.extend(
                members
                    .get(&sequence)
                    .ok_or("unknown host assigned sequence")?
                    .iter()
                    .copied(),
            );
        }
        drop(members);
        let keys = keys.into_iter().collect::<Vec<_>>();
        if !keys.is_empty()
            && region_owners()
                .lock_or_recover()
                .resolve(&keys)
                .map_err(|_| "invalid host assigned region")?
                .0
                != document.id
        {
            return Err("host assigned region belongs to another document".into());
        }
        Ok(keys)
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
        self.refresh_host_transport_inner(true);
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
        if throttled {
            let mut last = self.last_enumeration.lock_or_recover();
            let now = std::time::Instant::now();
            if last.is_some_and(|last| now.duration_since(last) < self.enumeration_interval()) {
                return;
            }
            *last = Some(now);
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
            self.host_item_muted.store(
                geometry
                    .as_ref()
                    .is_ok_and(|bound| !bound.is_empty() && bound.iter().all(|e| e.geometry.muted)),
                Ordering::Release,
            );
            if let Ok(bound) = &geometry {
                let mut items = stamp.0.region_items.lock_or_recover();
                for entry in bound {
                    items.insert(entry.region_key, entry.geometry.item_id.clone());
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
    /// realtime只读角色原子值；只有playback角色负责替换歌曲音频。
    pub(crate) fn renders_playback(&self) -> bool {
        self.role.load(Ordering::Acquire) == 1
    }
    /// 本实例是否拿到了宿主分配的 playback 角色。
    ///
    /// 【为什么不在这里做"没有 playback 实例就让 editor 接管"】那个改法看起来能救
    /// "宿主一个 playback 角色都不分配"的情形，但它**无法与合法形态区分**：
    /// 一个只拿到 `EDITOR_RENDERER` 的实例（REAPER 的正常编辑实例）与"宿主没分配
    /// playback"在文档级完全同形 —— 两者都是"本实例 known=all, assigned=editor"。
    /// 接管会让编辑实例开始渲染，从而破坏既有的透传契约（`vst3/audio_boundary_tests`
    /// 明确钉住"editor-only 必须保留宿主音频"）。而 ARA 没有"文档级哪个实例是
    /// 播放渲染器"的查询。
    ///
    /// 所以这里**不改行为**，只把它做成可见的读数（见
    /// `DocumentSession::report_playback_ownership`）：真要判断 Cubase 是否分配了
    /// playback，靠日志而不是靠猜。
    pub(crate) fn has_playback_role(&self) -> bool {
        self.role.load(Ordering::Acquire) == 1
    }
    /// 本实例的创建序号（越小越早）；诊断时用来指认实例。
    pub(crate) fn writer_id(&self) -> u64 {
        self.writer_id.load(Ordering::Acquire)
    }
    /// 记录 `setupProcessing` 声明的采样率（Hz）。非有限/非正数忽略（保持未声明）。
    pub(crate) fn set_setup_sample_rate(&self, rate: f64) {
        let hz = rate.round();
        if hz.is_finite() && hz > 0.0 && hz <= f64::from(i32::MAX) {
            self.setup_sample_rate.store(hz as i32, Ordering::Release);
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
        self.role.load(Ordering::Acquire) == 2
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
    pub(crate) fn editor_session(
        self: &Arc<Self>,
    ) -> Result<Arc<crate::editor::session::EditorSession>, String> {
        self.editor_document()?.editor_session()
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
    pub(crate) fn host_backend(&self) -> crate::host::adapter::HostBackend {
        self.host
            .lock_or_recover()
            .clone()
            .unwrap_or_else(|| crate::host::adapter::HostBackend::from_native(None))
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
    /// 组件/文档终止时停止实例worker；不要在音频process调用。
    pub(crate) fn stop_editor(&self) {
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
        let host = self.host.lock_or_recover().take();
        drop(host);
        self.host_geometry.lock_or_recover().take();
        self.cancel_preparation();
        if let Some(Ok(worker)) = self.preparation.get() {
            worker.close();
        }
    }
    /// 模型撤销时立即取消待准备作业；保持worker可供后续重新授权使用。
    pub(crate) fn cancel_preparation(&self) {
        if let Some(Ok(worker)) = self.preparation.get() {
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
        match self.preparation.get() {
            Some(Ok(worker)) => worker.state(),
            Some(Err(error)) => (false, Some(error.clone())),
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
    pub fn stop_channel(&self) {
        let channel = self.channel.lock_or_recover().take();
        drop(channel);
    }

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
    pub fn refresh_document(&self) {
        let document = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade);
        if let Some(document) = document {
            {
                let _transaction = document.transaction.lock_or_recover();
                let owners = document.renderer_owners();
                for owner in &owners {
                    if let Err(error) = owner.merge_pending_restore(&document) {
                        log::warn!("[ara] instance state unresolved: {error}");
                    }
                }
                for owner in owners {
                    owner.prepare();
                }
            }
        }
    }

    /// setState只暂存本组件恢复；宿主明确分配区域且模型ready后再合入共享权威。
    pub(crate) fn restore_state(&self, bytes: &[u8]) -> Result<(), String> {
        if bytes.is_empty() {
            return Ok(());
        }
        let mut restored = crate::state_channel::EditState::default();
        restored.restore(bytes)?;
        *self.pending_restore.lock_or_recover() = Some(restored);
        self.refresh_document();
        Ok(())
    }

    /// 接收 ARA 存档恢复出来的编辑，交给本 owner 的既有合并路径。
    ///
    /// 【为什么不直接写 `document.edits`】合并要做身份重绑、冲突判定、`needs_rebind`
    /// 处理 —— 这些都在 `merge_pending_restore` 里，而且已经过测试。存档通道只是另一个
    /// 来源，走同一条路才不会出现两份会漂移的实现。
    ///
    /// 【为什么覆盖已有的 `pending_restore`】VST3 组件状态与 ARA 存档携带的是同一份
    /// `EditState`。两者同时到达时后到的应当胜出（它是宿主更晚保存的那份）；而 ARA 存档
    /// 在宿主重载工程时是权威来源。
    pub(crate) fn adopt_archive_restore(&self, restored: crate::state_channel::EditState) {
        *self.pending_restore.lock_or_recover() = Some(restored);
        self.refresh_document();
    }
    /// 调用方持文档transaction；恢复候选集由宿主区域分配收窄，不按名称/旧序号猜测。
    pub(super) fn merge_pending_restore(
        &self,
        document: &super::document::DocumentSession,
    ) -> Result<(), String> {
        let mut pending = self.pending_restore.lock_or_recover();
        let Some(saved) = pending.as_ref() else {
            return Ok(());
        };
        if !document.ready.load(Ordering::Acquire) {
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
        restored.reconcile(&bindings)?;
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
            client.params_by_root_track.extend(
                restored
                    .atlas
                    .project_roots(&host, &document.parameter_identities_locked(&host)?)?,
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
            return Err("fade ownership metadata pending for restore".into());
        }
        merged.fades.retain(|item, _| !items.contains(item));
        merged.fades.extend(restored.fades);
        merged.reconcile(&document.track_bindings.lock_or_recover())?;
        *edits = merged;
        *pending = None;
        Ok(())
    }

    /// 保存前重新核对完整宿主图；歧义或中间编辑态不能伪装成可恢复的state。
    pub(crate) fn encode_state(&self) -> Result<Vec<u8>, String> {
        let document = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade);
        if let Some(document) = document {
            document.flush_editor()?;
            let _transaction = document.transaction.lock_or_recover();
            if !document.ready.load(Ordering::Acquire) {
                return Err("host graph not ready; cannot save ARA edits".into());
            }
            self.merge_pending_restore(&document)?;
            let mut edits = document.edits.lock_or_recover();
            edits.reconcile(&document.track_bindings.lock_or_recover())?;
            let host = self.assigned_timeline(&document)?;
            let allowed: BTreeSet<_> = host.tracks.iter().map(|t| t.id.clone()).collect();
            let mut local = edits.clone();
            local.params.retain(|id, _| allowed.contains(id));
            local.tracks.retain(|t| allowed.contains(&t.id));
            let clips = host
                .clips
                .iter()
                .map(|clip| clip.id.clone())
                .collect::<BTreeSet<_>>();
            local.atlas.regions.retain(|id, _| clips.contains(id));
            local
                .atlas
                .copy_seeds
                .retain(|_, seed| allowed.contains(&seed.root));
            local.atlas.gaps.retain(|root, _| allowed.contains(root));
            if !local.fades.is_empty() {
                let items = document.fade_items_locked(&clips);
                if items.is_empty() && !clips.is_empty() {
                    return Err("fade ownership metadata pending for save".into());
                }
                local.fades.retain(|item, _| items.contains(item));
            }
            local.bindings.retain(|id, _| allowed.contains(id));
            local.encode()
        } else {
            self.pending_restore
                .lock_or_recover()
                .as_ref()
                .unwrap_or(&self.edits.lock_or_recover())
                .encode()
        }
    }

    /// 参数/快照在短事务内校验；外部提交合成同样不能持锁阻塞宿主模型回调。
    pub(crate) fn handle_request(
        &self,
        request: hifishifter_ara_ipc::Request,
    ) -> hifishifter_ara_ipc::Response {
        use hifishifter_ara_ipc::{HostPcm, Request, Response};
        let document = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade);
        let Some(document) = document else {
            return Response {
                error: Some(crate::editor::errors::DOCUMENT_CLOSED.error()),
                code: Some(hifishifter_ara_ipc::error_code::DOCUMENT_CLOSED),
                ..Default::default()
            };
        };
        if let Request::Commit {
            base_revision,
            model_revision,
            schema_version,
            timeline,
        } = request
        {
            // timeline 载荷的结构版本守卫：未知的新版本拒收（旧客户端的 0 = 版本化
            // 之前，按 pre-versioning 接受）。否则一次不认识的 JSON 会被静默合并进
            // 编辑会话。
            if schema_version != 0 && schema_version != hifishifter_ara_ipc::TIMELINE_SCHEMA_VERSION
            {
                return Response {
                    ok: false,
                    error: Some(format!(
                        "unsupported timeline schema version {schema_version} (expected {}); refresh the app",
                        hifishifter_ara_ipc::TIMELINE_SCHEMA_VERSION
                    )),
                    ..Default::default()
                };
            }
            return self.commit_request(&document, base_revision, model_revision, timeline);
        }
        let _transaction = document.transaction.lock_or_recover();
        if let Err(error) = self.merge_pending_restore(&document) {
            return Response {
                code: crate::editor::errors::terminal_error_code(&error),
                error: Some(error),
                ..Default::default()
            };
        }
        let model_revision = document.revision.load(Ordering::Acquire);
        let mut edits = document.edits.lock_or_recover();
        let outcome = (|| {
            if !document.ready.load(Ordering::Acquire) {
                return Err(format!(
                    "{}; refresh after editing",
                    crate::editor::errors::HOST_MODEL_NOT_READY
                ));
            }
            edits.reconcile(&document.track_bindings.lock_or_recover())?;
            let mut timeline = self.assigned_timeline(&document)?;
            match request {
                Request::Snapshot => {
                    edits.apply(&mut timeline);
                    let available = document.edit_sources.lock_or_recover();
                    let mut sources = Vec::new();
                    for id in timeline
                        .clips
                        .iter()
                        .filter_map(|clip| clip.source_path.as_ref())
                        .collect::<BTreeSet<_>>()
                    {
                        let pcm = available.get(id).ok_or_else(|| {
                            format!("{}: {id}", crate::editor::errors::HOST_PCM_UNAVAILABLE)
                        })?;
                        sources.push(HostPcm {
                            persistent_id: id.clone(),
                            sample_rate: pcm.sample_rate,
                            fingerprint: pcm_fingerprint(pcm),
                            planes: pcm.planes.clone(),
                        });
                    }
                    let mut clips = timeline.clips.iter().collect::<Vec<_>>();
                    clips.sort_by(|a, b| a.start_sec.total_cmp(&b.start_sec));
                    let ranges = clips
                        .iter()
                        .take(8)
                        .map(|clip| (clip.id.clone(), clip.start_sec, clip.length_sec))
                        .collect::<Vec<_>>();
                    let diagnostics=document.renderer_owners().into_iter().filter(|owner|owner.renders_playback()).map(|owner| {
                        let (busy,error)=owner.local_preparation_state();
                        let prepared=owner.prepared_snapshot();
                        serde_json::json!({"busy":busy,"error":error,"prepared":prepared,
                            "outputs":owner.snapshots.iter().map(|snapshot|snapshot.diagnose(&ranges)).collect::<Vec<_>>()})
                    }).collect::<Vec<_>>();
                    Ok(Response {
                        ok: true,
                        timeline: Some(serde_json::to_value(&timeline).map_err(|e| e.to_string())?),
                        sources,
                        diagnostics: Some(serde_json::json!({"renderers":diagnostics})),
                        ..Default::default()
                    })
                }
                Request::Commit { .. } => {
                    // 结构性守卫：`Commit` 在进入本分发前就被拦下（见上方早分发）。
                    // 用 `unreachable!` 的话，一旦以后新增请求变体或早分发条件变动，
                    // 这里会在 IPC 工作线程上 panic（`catch_unwind` 会把它降级成
                    // "plugin request panicked"，提交被静默丢掉）。返回错误更诚实。
                    Err("commit request reached the non-commit dispatcher".to_string())
                }
            }
        })();
        let mut response = outcome.unwrap_or_else(|error| Response {
            code: crate::editor::errors::terminal_error_code(&error),
            error: Some(error),
            ..Default::default()
        });
        response.revision = edits.revision;
        response.model_revision = model_revision;
        response
    }

    /// 一期外部客户端保持同步结果契约，但捕获/计算/提交分开，失败不改变参数权威。
    fn commit_request(
        &self,
        document: &Arc<super::document::DocumentSession>,
        base_edit: u64,
        base_model: u64,
        client: serde_json::Value,
    ) -> hifishifter_ara_ipc::Response {
        let outcome = (|| {
            let (candidate, epoch, scope, inputs) = {
                let _transaction = document.transaction.lock_or_recover();
                self.merge_pending_restore(document)?;
                if !document.ready.load(Ordering::Acquire) {
                    return Err(format!(
                        "{}; refresh after editing",
                        crate::editor::errors::HOST_MODEL_NOT_READY
                    ));
                }
                if document.revision.load(Ordering::Acquire) != base_model {
                    return Err(format!(
                        "{} model changed; refresh",
                        crate::editor::errors::CONFLICT_PREFIX
                    ));
                }
                let mut edits = document.edits.lock_or_recover();
                edits.reconcile(&document.track_bindings.lock_or_recover())?;
                let client = serde_json::from_value(client).map_err(|e| e.to_string())?;
                let mut candidate =
                    edits.merge(&self.assigned_timeline(document)?, &client, base_edit)?;
                let mut curve_timeline = self.assigned_timeline(document)?;
                candidate.apply(&mut curve_timeline);
                candidate.atlas = edits.atlas.capture(
                    &curve_timeline,
                    &document.parameter_identities_locked(&curve_timeline)?,
                )?;
                candidate.reconcile(&document.track_bindings.lock_or_recover())?;
                drop(edits);
                let inputs = document
                    .renderer_owners()
                    .into_iter()
                    .filter(|owner| owner.renders_playback())
                    .map(|owner| {
                        let (keys, input) =
                            owner.capture_render_input(document, &candidate, true)?;
                        Ok((owner, keys, input))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                (
                    candidate,
                    document.render_epoch.load(Ordering::Acquire),
                    document.scope_revision.load(Ordering::Acquire),
                    inputs,
                )
            };
            let mut prepared = Vec::new();
            for (owner, keys, input) in inputs {
                for publisher in &owner.snapshots {
                    publisher.collect_retired();
                }
                let mut snapshots =
                    input.render(Arc::new(std::sync::atomic::AtomicBool::new(false)))?;
                super::snapshot::compact_prepared(&mut snapshots)?;
                prepared.push((owner, keys, snapshots));
            }
            let _transaction = document.transaction.lock_or_recover();
            if !document.ready.load(Ordering::Acquire)
                || document.revision.load(Ordering::Acquire) != base_model
            {
                return Err("Conflict: host model changed during commit".into());
            }
            let mut edits = document.edits.lock_or_recover();
            if edits.revision != base_edit {
                return Err("Conflict: edit revision changed during commit".into());
            }
            if document.render_epoch.load(Ordering::Acquire) != epoch
                || document.scope_revision.load(Ordering::Acquire) != scope
            {
                return Err("host audio access/scope changed during commit; retry".into());
            }
            for (owner, keys, snapshots) in &prepared {
                if owner.assigned_regions().map_err(|e| e.to_string())? != *keys {
                    return Err("Conflict: assigned regions changed during commit".into());
                }
                if !owner
                    .snapshots
                    .iter()
                    .zip(snapshots)
                    .all(|(p, s)| p.has_capacity(s))
                {
                    return Err("retired snapshot budget exhausted; reopen instance".into());
                }
            }
            for (owner, keys, snapshots) in prepared {
                for (publisher, snapshot) in owner.snapshots.iter().zip(snapshots) {
                    publisher
                        .publish(snapshot)
                        .map_err(|e| format!("snapshot publish failed: {e:?}"))?;
                }
                owner.record_prepared(base_model, candidate.revision, epoch, scope, keys);
            }
            *edits = candidate;
            log::info!(
                "[ara] GUI commit ready revision={} model={base_model}",
                edits.revision
            );
            Ok::<(), String>(())
        })();
        let error = outcome.err();
        hifishifter_ara_ipc::Response {
            ok: error.is_none(),
            code: error
                .as_deref()
                .and_then(crate::editor::errors::terminal_error_code),
            error,
            revision: document.edits.lock_or_recover().revision,
            model_revision: document.revision.load(Ordering::Acquire),
            ..Default::default()
        }
    }

    /// 调用方在短事务内冻结几何/参数/PCM，只复制本renderer授权源，不执行合成。
    pub(super) fn capture_render_input(
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
        // （宿主销毁了 region、分配还没撤销的窗口），整份 `capture_render_input` 失败 ⇒
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
        // `assigned_timeline` 打失败 —— 而它是 `capture_render_input` /
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
            // 已绑定再绑定是**不变量违反**（一个实例只绑一个文档）。真实宿主的重复
            // `bindDocumentToExtension` 会在更早的 companion 层被拒（"already bound"），
            // 由 `ara_entry::bind` 幂等复用既有实例处理（见那里的说明）—— 所以这里保持
            // 防御性错误，不在此做语义变更。
            return Err(AraError::InvalidState("extension already bound"));
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
        if enabled.contains(ExtensionRoles::PLAYBACK_RENDERER) {
            let _ = self
                .preparation
                .get_or_init(super::preparation::PreparationQueue::new);
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
        // 绑定成功即置"曾经绑定过"。此后 `editor_document()` 的措辞走 DOCUMENT_CLOSED
        // 而不是 ARA_NOT_BOUND —— 两者对用户的含义完全不同（见该访问器的说明）。
        self.bound_once.store(true, Ordering::Release);
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
        if enabled.contains(ExtensionRoles::EDITOR_RENDERER) {
            let weak = Arc::downgrade(self);
            // 【为什么不写死宿主名】IPC 服务器名对所有连接的客户端可见；此前硬编码
            // `"HiFiShifter / REAPER"`，在 Studio One / Cubase 等宿主上会报出错误的
            // 宿主身份。从当前后端派生（`app_name` 只用于诊断，不参与分支）。
            let server_name = match self.host_backend().app_name() {
                Some(app) => format!("HiFiShifter / {app}"),
                None => "HiFiShifter".to_string(),
            };
            match hifishifter_ara_ipc::Server::start(server_name, move |request| {
                weak.upgrade()
                    .map(|owner| owner.handle_request(request))
                    .unwrap_or_else(|| hifishifter_ara_ipc::Response {
                        error: Some(crate::editor::errors::INSTANCE_CLOSED.error()),
                        ..Default::default()
                    })
            }) {
                Ok(server) => {
                    log::info!(
                        "[ara] GUI channel ready instance={}",
                        server.record().instance_id
                    );
                    *self.channel.lock_or_recover() = Some(server);
                }
                Err(error) => log::warn!("[ara] GUI channel unavailable: {error}"),
            }
        }
        // 【为什么放在最后】这份读数要看到**全部**实例，包括本实例刚写进去的
        // `writer_id`。角色集合只有两个时刻会变：绑定（这里）与分配变化
        // （见 `new_with_renderer_observer` 的观察器），两处都要报。
        document.report_playback_ownership();
        Ok(raw)
    }

    /// 文档侧已撤销 lease；保留 binding 以供宿主最后几次合法移除/释放回调访问。
    pub fn document_closed(&self) {
        self.stop_editor();
        self.stop_channel();
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
        let mut keys = self
            .assignments
            .lock_or_recover()
            .get(&role)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeSet<_>>();
        let sequences = self
            .sequences
            .lock_or_recover()
            .get(&role)
            .cloned()
            .unwrap_or_default();
        let members = document.sequence_regions.lock_or_recover();
        let mut unknown_sequences = 0_usize;
        for sequence in sequences {
            // 【为什么未知 sequence 只跳过】`sequence_regions` 只登记**已经有 region**
            // 的 sequence（见 `create_playback_region`）。宿主先建 sequence、后建 region
            // 是正常时序，其间该键必然缺席。此前这里 `ok_or(...)?` 让整份
            // `assigned_regions()` 失败 ⇒ 经 `workspace_scope_locked` /
            // `assigned_timeline` 把整份工作区与渲染输入一起打失败 ——
            // "宿主建了轨道、插件里什么都没有"。跳过即可：该 sequence 本次没有成员。
            match members.get(&sequence) {
                Some(regions) => keys.extend(regions.iter().copied()),
                None => unknown_sequences += 1,
            }
        }
        drop(members);
        if unknown_sequences > 0 {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Debug,
                "ara.assigned.unknown-sequence",
                format_args!(
                    "[ara] {unknown_sequences} assigned region sequence(s) have no playback \
                     region yet; they contribute no members this round"
                ),
            );
        }
        let keys = keys.into_iter().collect::<Vec<_>>();
        if !keys.is_empty() {
            let (owner, _) = region_owners()
                .lock_or_recover()
                .resolve(&keys)
                .map_err(|_| AraError::InvalidState("invalid assigned region"))?;
            if owner != document.id {
                return Err(AraError::InvalidState(
                    "assigned region belongs to another document",
                ));
            }
        }
        Ok(keys)
    }

    /// 宿主模型callback只撤销/排最新准备任务，不在主线程执行WORLD或PCM混音。
    pub fn prepare(&self) {
        if !self.renders_playback() {
            return;
        }
        #[cfg(test)]
        self.prepare_requests.fetch_add(1, Ordering::AcqRel);
        // 源/assignment撤销在文档事务内清理；此函数只排队，不能在事务外clear旧任务刚发布的音频。
        let weak = self.prepare_owner.lock_or_recover().clone();
        let Some(weak) = weak else {
            return;
        };
        match self.preparation.get() {
            Some(Ok(worker)) => {
                if let Err(error) = worker.request(Box::new(move |cancel| {
                    weak.upgrade()
                        .ok_or("renderer closed".to_owned())?
                        .prepare_job(cancel)
                })) {
                    log::warn!("[ara] preparation unavailable: {error}");
                }
            }
            Some(Err(error)) => log::warn!("[ara] preparation unavailable: {error}"),
            None => {}
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
        if let Some(Ok(queue)) = self.preparation.get() {
            queue.acknowledge_idle_success();
        }
    }
    /// 仅SDK规定UI线程的kOffline setup调用；不在事务锁内等待，不把旧快照当最新版本。
    pub(crate) fn prepare_offline_until(&self, deadline: std::time::Instant) -> Result<(), String> {
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
                .preparation
                .get()
                .ok_or("offline preparation worker missing")?
                .as_ref()
                .map_err(Clone::clone)?;
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
            let (keys, input) = self.capture_render_input(&document, &edits, edits.revision > 0)?;
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
