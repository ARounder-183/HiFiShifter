//! `ExtensionOwner` 周边的值类型：几何缓存、绑定读数、作用域、宿主音频读数、形态与捕获结论。
use super::*;
/// 只读元数据供非实时读者消费；UI/model采集，音频线程不访问此缓存或锁owner。
///
/// `value` 是**逐 region** 的绑定列表，不是一个实例一个几何：挂在 REAPER folder
/// 轨上的 FX 会被 ARA 分配整组 region，而 `parent(2)`（实例自己的 take）结构上只
/// 能覆盖其中一个。见 [`ExtensionOwner::host_geometries`]。
#[derive(Clone)]
pub(super) struct CachedHostGeometry {
    pub(super) model: u64,
    pub(super) scope: u64,
    pub(super) change: Option<i32>,
    pub(super) value: Result<Vec<crate::host::geometry::BoundHostGeometry>, String>,
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
pub(super) struct BindingReport {
    pub(super) assigned: Vec<u64>,
    pub(super) bound: Vec<u64>,
    pub(super) unmatched: Vec<u64>,
    pub(super) dropped: usize,
    pub(super) candidates: usize,
}

/// 绑定求解失败的两条**结构性**措辞（产生点与分类点共用同一常量）。
///
/// 【为什么不用字面量】与 `editor::errors` 的终态常量同一条纪律：产生点与分类点
/// 分处不同函数，两边各写字面量时任何一处改措辞都会让分类**静默失效**（退避停摆，
/// 或对状态变化误退避），且没有编译期信号。
pub(super) const BINDING_NO_UNIQUE_ITEM: &str = "no assigned ARA region matched a unique host item";
pub(super) const BINDING_GEOMETRY_INCOMPATIBLE: &str =
    "direct take geometry incompatible with assigned ARA playback window";

/// 绑定失败是否属于"由宿主工程结构决定、不会自行变化"的一类。
///
/// 【为什么必须区分】只有这一类才值得退避：它是"重试一万次也一样"的读数。授权撤销、
/// 文档关闭、扩展接口消失都是**状态变化**，它们要么很快自愈、要么已经是终态 ——
/// 对它们退避只会拖慢恢复。
///
/// 【为什么用"包含"而不是全等】与 `editor::errors::classify_error` 同一理由：措辞
/// 可能带上下文前缀（`host split: …`），常量保证的是**子串出现**。
pub(super) fn is_structural_binding_failure(error: &str) -> bool {
    error.contains(BINDING_NO_UNIQUE_ITEM) || error.contains(BINDING_GEOMETRY_INCOMPATIBLE)
}

/// 两个宿主/ARA 浮点量是否相容；判据全局唯一，见 `host::geometry::host_value_compatible`
/// （绑定 / 分割规划 / 写回前置检查 / 剪贴板共用，避免"规划接受、绑定拒绝"的不对称）。
pub(super) fn compatible(a: f64, b: f64) -> bool {
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
pub(super) fn unique_geometry_for_region(
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
pub(super) struct PreparedVersion {
    pub(super) model: u64,
    pub(super) edit: u64,
    pub(super) epoch: u64,
    pub(super) scope: u64,
    pub(super) keys: Vec<u64>,
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

impl ScopeSource {}

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
pub(super) fn graph_summary_of(document: &super::document::DocumentSession) -> String {
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
    pub(super) const fn from_raw(raw: i32) -> Self {
        match raw {
            1 => InstanceMode::Ara,
            2 => InstanceMode::Capture,
            _ => InstanceMode::Undetermined,
        }
    }

    pub(super) const fn raw(self) -> i32 {
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
