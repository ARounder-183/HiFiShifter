//! 按真实 controller 身份索引文档生命周期，不从最后一个创建的文档猜测归属。

use super::extension::ExtensionOwner;
use super::ownership::DocumentId;
use crate::sync::MutexExt;
use ara2_bridge::companion::CompanionControllerBinding;
use ara2_bridge::core::{ApiGeneration, AraError};
use ara2_bridge::plugin::ExtensionControllerLease;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

struct RendererLease {
    lease: ExtensionControllerLease,
    _companion: Option<CompanionControllerBinding<'static>>,
    owner: Weak<ExtensionOwner>,
}

/// 内嵌GUI同进程冻结值；PCM沿授权Arc共享，不能经旧IPC/JSON复制完整长源。
pub(crate) struct WorkspaceSnapshot {
    pub timeline: hifishifter_kernel::state::TimelineState,
    pub sources: Vec<(String, Arc<super::source::SourcePcm>)>,
    pub revision: u64,
    pub model_revision: u64,
    pub audio_revision: u64,
}

/// 一个 ARA 文档的共享状态。
///
/// # 锁序（必须遵守）
///
/// 多把锁同时持有时，一律按下面的顺序获取；**反向获取会死锁**：
///
/// 1. `transaction` —— 最外层：几何/编辑的短事务
/// 2. `timeline`
/// 3. `edits`
/// 4. 宿主事实：`ui_tracks` → `host_folder_children` / `clip_ids` / `regions` /
///    `region_items` / `authorized_takes` / `authorized_sources` / `render_facts`
/// 5. `region_owners()`（进程级全局表）—— 最内层叶子锁
///
/// 例外与约束：
/// - `region_owners()` 是**叶子**：持有它时不得再取上面任何一把锁。它出现在
///   `ara/model.rs` 的回调里（不持文档锁）与 `render/extension.rs` 的查询里
///   （在 `transaction` 之内）—— 两者都满足"最后获取"。
/// - `RenderFacts` 采样（`refresh_ui_inventory`）在 `transaction` 之内、且在其它
///   文档锁之外，因此它与 `workspace_projection_locked` 里对 `render_facts` 的读取
///   不会互相阻塞到死锁。
/// - 单把锁内的长操作（宿主 API 调用、神经推理）必须先把数据 `clone` 出来再解锁 ——
///   见 `render/extension.rs` 的"宿主调用始终不持内部锁"。
/// 宿主内容变化后等待其收口的宽限期。
///
/// 与 bridge 的 `DEFERRED_EDITING_GRACE` 同量级：一次编辑从内容回调到
/// `endEditing` 通常在毫秒级；超过这个窗口仍未收口，就按"宿主不再收口"处理 ——
/// 保留最后一份已发布的时间线继续刷新界面，而不是把窗口永久锁在冲突状态上。
pub(crate) const EDIT_SETTLE_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// 插件侧强制收口一次**从未关闭**的编辑会话的宽限期。
///
/// 【为什么比 `EDIT_SETTLE_GRACE` 长得多】那一条判的是"宿主已经发过 `endEditing`，
/// 只是被 reader 挡住了"——收敛是必然的，短一点没有风险。这一条判的是"宿主**根本
/// 没打算收口**"，而合法的编辑会话可以持续很久：宿主通常在鼠标按下时
/// `beginEditing`、抬起时 `endEditing`，一次慢速拖拽就能超过 2 秒。用 2 秒去强制
/// 收口，等于在用户拖动到一半时把模型提交一次 —— 曲线会跳、撤销栈会碎。
///
/// 取 8 秒：足够覆盖任何正常交互，又短到"宿主真的挂了"时用户不必等太久。
pub(crate) const FORCED_EDIT_CLOSE_GRACE: std::time::Duration = std::time::Duration::from_secs(8);

/// 一个 ARA 音频源相对本插件的**可读性**。
///
/// 【为什么是三态而不是一个布尔】三种"没有 PCM"的处置完全不同：
/// * `Published` —— 有 PCM，正常；
/// * `Revoked` —— 宿主暂时收回了样本访问（或把源放进了 undo 历史）。**会自己好**：
///   下一次授权 / 插件侧重试都能恢复，界面上不该说成"等待分配"；
/// * `Failed` —— 授权在，但读取失败（真故障，需要留真因）。
///
/// 把它们压成一个布尔，正是"后台切回来之后 Clip 一直等宿主音频"看起来像
/// "宿主还在分配 region"的原因。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SourceState {
    /// 已读到并发布了 PCM。
    Published,
    /// 宿主暂时不可读（样本访问被收回 / 源被停用），但**已发布的 PCM 仍然保留**。
    Revoked,
    /// 授权在、读取失败。
    Failed,
}

impl SourceState {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Published => "published",
            Self::Revoked => "revoked",
            Self::Failed => "failed",
        }
    }
}

/// 输入可用性的**综合判定**。
///
/// 【为什么不是一个布尔】`ready` 只回答"渲染许可恢复过没有"，它不回答"PCM 到底在不在"。
/// 于是"宿主授权了 0 个源"与"一切正常"在插件自己看来是同一种状态，界面也只能说一句
/// 含糊的"等待宿主音频"。这个判据把四种处境分开，每一处都对应一个不同的下一步。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InputVerdict {
    /// 有 region、有授权、有已发布的 PCM。
    Ready,
    /// 宿主还没交付任何音频源（还没分配 region / 还没建 source）。
    AwaitingRegions,
    /// 源有了，但样本访问没开、或还没读出来。
    AwaitingSamples,
    /// 授权在却读不出来，或部分源失败 —— 可行动的真故障。
    Degraded,
}

impl InputVerdict {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::AwaitingRegions => "awaiting_regions",
            Self::AwaitingSamples => "awaiting_samples",
            Self::Degraded => "degraded",
        }
    }
}

/// 一次输入可用性的完整读数（见 [`InputVerdict`]）。
///
/// 【为什么带这么多计数】用户报障时最需要回答的是"卡在哪一步"。只给一个 verdict，
/// 下一轮仍然只能靠猜；把"几条 region、几个源、几个授权、几个已发布、几个失败"
/// 一起给出来，一行日志就能指认。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InputReadiness {
    pub regions: usize,
    pub regions_with_source: usize,
    pub sources: usize,
    pub sources_authorized: usize,
    pub sources_published: usize,
    pub sources_failed: usize,
    pub model_published: bool,
    pub verdict: InputVerdict,
}

impl InputReadiness {
    /// 一行可读的读数（诊断包与 `stuck_report` 共用）。
    pub(crate) fn summary(&self) -> String {
        format!(
            "verdict={} regions={}/{} sources={} authorized={} published={} failed={} model={}",
            self.verdict.as_str(),
            self.regions_with_source,
            self.regions,
            self.sources,
            self.sources_authorized,
            self.sources_published,
            self.sources_failed,
            self.model_published,
        )
    }
}

#[derive(Default)]
pub(crate) struct DocumentSession {    controller: AtomicUsize,
    alive: AtomicBool,
    /// 本文档是否曾经成功 attach 过 renderer（重绑/关闭后用于区分"还没绑"与"绑过又没了"）。
    ever_had_renderer: AtomicBool,
    renderers: Mutex<Vec<RendererLease>>,
    pub generation: Mutex<Option<ApiGeneration>>,
    pub sequence_regions: Mutex<HashMap<u64, HashSet<u64>>>,
    /// sequence 真实键 → 扁平 region sequence 下标（= 映射层轨道下标）。
    ///
    /// 只用于把"被分配了 sequence、当前却没有 clip"的轨道认出来（空 folder 轨、
    /// 只有静音/待授权 item 的子轨）—— 它们仍应是参数根。**不是**按名字/位置猜轨道：
    /// 这条边来自宿主 `createRegionSequence` 的真实键。
    pub sequence_track_index: Mutex<HashMap<u64, usize>>,
    pub regions: Mutex<HashMap<u64, crate::ara::AraPlaybackRegion>>,
    /// region key → clip id 的**缓存**投影。
    ///
    /// 【为什么是缓存而不是权威】权威身份是 region key 本身，clip id 由它派生
    /// （`ara-region-{key}`，见 `ara::mapping::MappingIdentities`）。这张表只是把
    /// "key → 当前时间线里的 id"记下来供反查，**缺席不是错误** —— [`Self::clip_id_for`]
    /// 会按同一个算式补出来。此前三处调用点把缺席当致命错误（`missing assigned clip
    /// identity` 等），而 `clip_ids` 只在 `end_editing` 时整体重写，于是"region 已分配、
    /// identity 还没重算"这个窗口里的每次渲染都必然失败。
    pub clip_ids: Mutex<HashMap<u64, String>>,
    pub region_items: Mutex<HashMap<u64, String>>,
    /// ARA 授权那一刻，每个 region 对应 item 的 **active take GUID**（region_key → take GUID）。
    ///
    /// 【为什么需要】ARA 只授权 active take 的 PCM（见 `probe/ara/MULTI-TAKE-FINDINGS.md`
    /// 与 plan Part 2 的授权边界）。宿主清单重建 take 列表时（`sync_host_takes`），
    /// 只有 GUID 与这里相等的那一个 take 可以带上授权媒体；其余 take 必须保持无源。
    ///
    /// 【为什么记 GUID 而不是"就是 active 那个"】用户在 REAPER 里切换 active take 后，
    /// 这条记录直到 ARA 模型重新认领才更新。这期间**不沿用**旧 PCM —— 否则会把上一个
    /// take 的采样挂到新 take 上，那是听不见的错（渲染读的是同一条 `source_path`）。
    /// 记不下来时宁可不挂：无源占位是看得见的。
    pub authorized_takes: Mutex<HashMap<u64, String>>,
    /// ARA 授权那一刻，每个 region 对应 item 的 active take 的**源文件路径**
    /// （region_key → 文件路径）。与 [`Self::authorized_takes`] 同处写入。
    ///
    /// 【为什么需要】GUID 相等是**最强**的身份判据，但用户在 REAPER 里把 active take
    /// 换成**同一个文件**的另一个 take（复制 take、切换 take）时 GUID 就变了，而内容
    /// 并未改变。此时文件路径相同即可证明"同一份已授权 PCM"，`sync_host_takes` 可以
    /// 安全地把它挂上去，而不是让明明有音频的片段显示占位。
    ///
    /// 【边界不可松动】这只回退到**已被 ARA 授权的那份 PCM**；路径为空或读不到时
    /// 仍然"宁可不挂"（无源占位是看得见的，挂错采样是听不见的错）。
    pub authorized_sources: Mutex<HashMap<u64, String>>,
    /// 每个 ARA 音频源（persistentID → 状态）的**可读性**，供界面与自愈判定。
    ///
    /// 【为什么需要它】此前"源暂时不可读"与"源从来没读到过"在实现里是同一件事：
    /// `refresh_source_pcm` 无论什么原因都先把已发布的 PCM 删掉，再"若可读则重读"。
    /// 于是宿主收回一次样本访问（后台、undo 历史、宿主自己整理缓存）就等于**永久**
    /// 丢掉音频 —— 恢复只依赖"宿主记得再发一次 enable"，而那条回调丢了就没了。
    ///
    /// 把状态记下来之后：已发布的 PCM 只在"源被销毁 / 内容版本换了且新版本就绪 /
    /// 文档关闭"这三种情况下被替换，其余一切暂时不可读都只是状态变化。
    pub source_states: Mutex<HashMap<String, SourceState>>,
    /// 最近一次由**模型侧**算出的输入可用性读数（`None` = 还没算过，按旧行为处理）。
    ///
    /// 【为什么由模型侧写、文档侧读】`audio_sources` 是 `ModelHandle` 的私有数据，
    /// 文档看不到；而判定结果要给 `ensure_loaded` / `capture_render_input` / 诊断包用。
    /// 让模型在每次源变化之后写一份读数，是两者之间唯一不需要反向引用的通道。
    ///
    /// 【`None` 为什么等于旧行为】没有跑过任何源回调的会话（测试夹具、纯几何会话）
    /// 不该因为"没算过"被判成未就绪 —— 那会把未知变成否定，正是本轮要修的那类错误。
    /// 本地补丁（T8）：本份文档控制器的图账本。
    ///
    /// 【为什么文档持有它】桥把账本记在 `ControllerAdapter` 上，而诊断路径在文档侧。
    /// `document_controller` 闭包每建一份模型就运行一次，模型在那里接管句柄并转存到
    /// 这里 —— 一条不需要反向引用的通道。缺席时降级为空账本（账本是纯观测）。
    pub graph_ledger: Mutex<Option<std::sync::Arc<ara2_bridge::plugin::GraphLedger>>>,
    pub readiness: Mutex<Option<InputReadiness>>,
    pub ui_tracks: Mutex<std::collections::BTreeMap<String, crate::host::UiTrack>>,
    pub ui_known_tracks: Mutex<HashSet<String>>,
    /// 由 REAPER folder 结构**得到父级**的轨道 id（见 `host::folder`）。
    ///
    /// 这些轨道的父子边由宿主决定，插件私有分组**不得**改写 —— 否则同一个工程会
    /// 出现两套父子关系：用户在 REAPER 里改了分组，插件却显示另一套，且无法诊断。
    ///
    /// 【为什么只记"有父级"的轨道，而不是"宿主呈现过的所有轨道"】宿主清单同时承载
    /// 轨道身份（GUID↔id 绑定）与 folder 结构两件事。宿主**没有**给出父级的轨道
    /// （folder 之外的普通轨）并不代表"宿主说它是根级"，只是没有那条边 —— 私有分组
    /// 依然可以把它编进一个参数组。把整份清单都算作权威会静默废掉用户的私有分组。
    pub host_folder_children: Mutex<std::collections::BTreeSet<String>>,
    /// 分割谱系：右半段 item GUID → 左半段（父段）item GUID。
    ///
    /// 【为什么需要】宿主分割**不改变音频源** —— 两半指向同一个文件。父段已被 ARA
    /// 授权，所以右半段可以立刻显示波形，不必等宿主为它重新分配 region。缺了这条，
    /// 分割后右半段会停在一段"等待 REAPER 提供音频"的占位里（用户报障过）。
    ///
    /// 【为什么不复用 `ParameterAtlas::split_parents`】那份谱系服务的是**参数曲线**
    /// 继承，键与生命周期都挂在参数权威上；媒体派生是显示层的事，两件事不该共用一个
    /// 记录 —— 否则一方的清理会静默影响另一方。
    pub split_media_from: Mutex<HashMap<String, String>>,
    /// 每个尚未拿到音频的 item 是**从什么时候**开始等的（item GUID → 首次进入
    /// "在途"的时刻）。
    ///
    /// 【为什么曾经需要它】"等待宿主完成音频分配"此前是**吸收态**：一个 item 没有
    /// `source_path`、又不是 folder 父轨，就永远显示"正在等待"，用户做什么都出不来。
    /// 记下时刻是为了"超过 2 s 就转 `unavailable` 并给出原因"。
    ///
    /// 【为什么它现在**不**决定状态】那个终态把"我们这一轮没观察到"说成了"宿主不会
    /// 交付了"，而后者是一句我们无法证明的断言 —— 只要展示侧的身份链断一次，所有片段
    /// 都会被盖上同一句话（用户看到的正是"无论什么情况，所有 Clip 都在等"）。
    /// 现在它只回答"等了多久"（`wait_seconds`），状态恒为 `pending`。
    pub pending_since: Mutex<HashMap<String, std::time::Instant>>,
    /// 最近一次映射里被**跳过**的 playback region（下标 + 语言无关原因码）。
    ///
    /// 【为什么要进 GUI 载荷】此前只 `log::warn`，于是 MIDI / 零长度 item 在插件里
    /// **静默消失**：用户看到"我明明有这一段"，界面却什么都没有，也没有任何提示。
    /// 记在这里，载荷给一句"N 个片段未显示（宿主数据无法映射）"，用户至少知道该查什么。
    pub skipped_regions: Mutex<Vec<(usize, String)>>,
    /// 最近一次**完整**宿主清单采集的代次令牌 `(change_count, model, scope)`。
    ///
    /// 【为什么是文档级、不是实例级】`ui_tracks` / `render_facts` 都是文档级共享状态，
    /// 采集结果写进 `document`，任何实例调用 [`crate::render::extension::ExtensionOwner::refresh_ui_inventory`]
    /// 都命中同一份缓存 —— 于是"谁采的"与"谁要用的"不再分离（多实例下只有第一个带
    /// 新令牌的实例真正枚举，其余早退）。
    pub ui_inventory_stamp: Mutex<Option<(i32, u64, u64)>>,
    /// 最近一次**完整**采集的时刻。
    ///
    /// 【为什么必须与令牌并存】令牌以 `GetProjectStateChangeCount` 为主。方向（倒放）是
    /// **take 级**属性，宿主未必为它推进项目 change count；只靠令牌早退会让"倒放 Take"
    /// 连采样这一步都被跳过（`render_facts` 不更新 ⇒ 渲染输入不失效 ⇒ 波形变了声音没变）。
    /// 超过 [`crate::render::extension::FACTS_REFRESH_INTERVAL`] 就**强制**重采一轮，
    /// 让任何 take 级事实变化都必然穿透这道早退 —— 代价被钉在低频（见该常量）。
    pub ui_inventory_sampled_at: Mutex<Option<std::time::Instant>>,
    /// 上一轮宿主清单里"ARA 表达不到、但会改变渲染结果"的事实快照。
    ///
    /// 【为什么必须有它】`render_epoch` 此前只有**淡变长度**一个写者，于是宿主把一条
    /// take 倒放（或改声道模式、开循环源）之后，`PreparedVersion` 五分量全不变 ⇒
    /// `prepare_job` / `prepare_offline_until` 判定"已是最新"直接早退，继续播旧的正向
    /// 音频。用户表现为"波形更新了、声音没变，编辑一下才对"（编辑是唯一能推动
    /// `PreparedVersion.edit` 的手动途径）。见 `host::geometry::RenderFacts`。
    pub render_facts: Mutex<crate::host::geometry::RenderFacts>,
    pub sources: Mutex<HashMap<String, Arc<super::source::SourcePcm>>>,
    pub edit_sources: Mutex<HashMap<String, Arc<super::source::SourcePcm>>>,
    /// 已授权但**读取失败**的 ARA 音频源（`audioSource.persistentID` → 原因）。
    ///
    /// 【为什么必须与 `ready` 分开】`ready` 的语义是"ARA 是否已重新交付输入"，它还
    /// 参与 `ensure_loaded` 的宿主冲突判据（见 `editing_session`）。把"某个源读不出来"
    /// 折进 `ready=false` 会让一次样本读取失败被报成"宿主冲突"，正是用户报障的
    /// "尚未应用：Conflict: host changed" 的来源。所以这里单记一份：读取失败时条目
    /// 出现、成功时消失，`decorate_host_media_locked` 据此把 clip 的
    /// `host_media_reason` 报成真因（`source_read_failed`），而不是含混的
    /// `awaiting_region`。
    pub source_failures: Mutex<HashMap<String, String>>,
    /// 本插件拒绝创建 region 的原因（有界环形缓冲，最新的在尾部）。
    ///
    /// 【为什么要单独一张表】"等待宿主音频"是用户可见的唯一状态，而它把两种截然
    /// 不同的事实混成一句话：宿主**还没**交付，和插件**拒收**了交付。后者是
    /// 可行动的故障 —— `HostAudioState::RegionRejected` 就是为它加的。
    ///
    /// 【为什么有界】它只在故障路径上写；不设上限的话一个反复失败的宿主会把内存
    /// 吃光。32 条足够覆盖"前因 + 若干次重复"。
    pub region_rejections: Mutex<std::collections::VecDeque<String>>,
    /// PCM 可以晚于结构回调到达；不借用 model/scope 版本触发 GUI 资源补齐。
    pub editor_audio_revision: AtomicU64,
    pub timeline: Mutex<Option<hifishifter_kernel::state::TimelineState>>,
    pub track_bindings: Mutex<crate::state_channel::TrackBindings>,
    pub revision: AtomicU64,
    pub render_epoch: AtomicU64,
    pub scope_revision: AtomicU64,
    /// 纯GUI宿主装饰版本，不改变编辑曲线或神经合成代次。
    pub ui_geometry_revision: AtomicU64,
    pub ready: AtomicBool,
    /// 宿主是否至少给过一个源样本访问授权。
    ///
    /// 【用途】把"宿主还没交付 region"与"region 有了、但样本访问还没开"分开显示；
    /// 两者对用户的可执行动作不同。
    pub samples_access_granted: AtomicBool,
    /// 宿主是否处在一次 `beginEditing` / `endEditing` 会话中。
    ///
    /// 【为什么与 `ready` 分开】`ready` 的语义是"ARA 是否已重新交付可用输入"，
    /// 会被**任何**内容变化按回 `false`；而"宿主正在编辑"是另一件事。此前二者共用
    /// `ready`，于是会话内每一次真实变化都被 `ensure_loaded` 判成"宿主冲突"
    /// （用户报障：每次编辑都闪"尚未应用：Conflict: host changed"）。
    /// 有了这个标志，会话中的未就绪窗口被识别为**暂缓**而不是冲突 —— 它会随
    /// `endEditing` 自然收敛，不需要用户干预，也不该报错。
    pub editing_session: AtomicBool,
    /// 编辑会话开始的**单调毫秒**（0 = 不在会话中）。
    ///
    /// 【为什么与 `editing_session` 并存】那个布尔回答"是不是在会话中"，这个数回答
    /// "在会话中多久了"。而"多久"正是插件侧唯一能自主判断"宿主不会再来收口"的依据
    /// （见 [`Self::request_settle_if_overdue`]）。两者分开是因为布尔在音频线程上读，
    /// 而这个数只在非音频线程上读 —— 语义不同，读者也不同。
    pub editing_since_ms: std::sync::atomic::AtomicU64,
    /// 模型自上次成功发布以来**第一次变脏**的单调毫秒（0 = 干净）。
    ///
    /// 【为什么要它】`ModelHandle::model_dirty` 是模型线程私有的布尔，插件侧看不到；
    /// 而"脏了多久还没被 flush"正是第二条死路的判据（宿主既不 `endEditing` 也不
    /// `notifyModelUpdates` 时，唯一的 flush 入口永远到不了）。
    pub model_dirty_since_ms: std::sync::atomic::AtomicU64,
    /// 插件侧请求"强制收口一次编辑会话"（由非 ARA 线程置位，由 ARA 线程消费）。
    ///
    /// 【为什么必须是**请求**而不是直接调用】桥的 `PluginRuntime` 由文档控制器独占，
    /// 不是 `Send`/`Sync`；从视图定时器直接调用它是未定义行为。这个原子量是两者之间
    /// 唯一的通道：谁都可以置位，控制器在**下一次任何回调**的入口消费它。
    pub forced_edit_close: AtomicBool,
    /// 插件侧请求"把积压的模型变化 flush 一次"（同上的通道）。
    pub forced_model_flush: AtomicBool,
    /// 最近一次因宿主内容变化而撤销渲染许可的时刻（见 [`Self::clear_renderers`]）。
    ///
    /// 【为什么需要它】`ready == false` 有两种完全不同的来源：宿主正在一次编辑
    /// 会话中（`editing_session` 已表达，必然收敛），与宿主改完内容后**不再收口**
    /// （不发 `endEditing`、也不发 `notifyModelUpdates`）。后者此前被
    /// `ensure_loaded` 当成"宿主冲突"永久拒绝，用户每做一次操作就看到一条
    /// `Conflict: host changed`，而窗口再也刷不出来。用这个时刻把"刚刚才变过、
    /// 正在等宿主收口"与"确实没有可用输入"分开。
    pub last_content_change: Mutex<Option<std::time::Instant>>,
    pub transaction: Mutex<()>,
    pub edits: Arc<Mutex<crate::state_channel::EditState>>,
    pub id: DocumentId,
    pub clock: Arc<super::transport::TransportClock>,
    pub playback: Mutex<Option<ara2_bridge::plugin::PlaybackRequestHandle>>,
    pub host_undo: crate::host::undo::HostUndo,
    /// 宿主 BPM / 拍号读数（VST3 进程上下文）。
    ///
    /// 【为什么记在文档上】Tempo Map 的音阶点需要宿主轴：0 位置点**必须**显式带拍号
    /// （前端 `effectiveTimeSignatures` 依赖它），每个点都要 BPM。而音阶投影在
    /// `workspace_timeline_locked` 里跑（`DocumentSession` 上），那里读不到
    /// per-view 的 `ProjectState`。
    pub host_meter: Mutex<HostMeter>,
    /// 宿主经 ARA 音乐上下文给出的真实 tempo map 与拍号。
    ///
    /// 【为什么在文档上】它是**文档级**音乐事实（整个工程一份），不是某个 renderer 的
    /// 属性；而投影缓存（`musical_projection`）与 `host_meter` 也在文档上 —— 三者必须
    /// 同源，否则缓存键会漏掉 tempo map 的变化。
    ///
    /// `None` = 宿主没提供（或读取失败），投影退回 VST3 的单点 tempo。
    pub host_tempo_map: Mutex<Option<crate::ara::model::HostTempoMap>>,
    /// 插件自有音乐上下文的投影缓存，键是"输入代次"（见
    /// [`DocumentSession::project_plugin_musical_context_locked`]）。
    pub musical_projection: Mutex<Option<(MusicalProjectionKey, PluginMusicalProjection)>>,
    /// ARA 存档恢复出来的编辑，等待模型图就绪后合入。
    ///
    /// 【为什么放在文档上而不是 owner 上】`beginRestoringDocumentFromArchive` 在
    /// **模型图重建之前**到达（ARA 的恢复顺序：先给存档，宿主再重新交付模型）。那一刻
    /// 可能一个 renderer 都还没绑定，`renderer_owners()` 是空的 —— 所以载荷必须存在
    /// 文档级，等 `prepare_renderers` 收口时再交给 owner 的既有合并路径。
    ///
    /// 【为什么复用 owner 的 `pending_restore`】合并语义（reconcile、身份重绑、冲突
    /// 判定）已经完整实现并测试过（`ExtensionOwner::merge_pending_restore`）。存档通道
    /// 只是**另一个来源**，不该有第二份实现 —— 两份实现迟早会漂移。
    pub pending_archive_restore: Mutex<Option<crate::state_channel::EditState>>,
    /// ARA 2 **部分存档**的暂存载荷（`restoreObjectsFromArchive` 的 `Selected` 分支）。
    ///
    /// 【为什么不复用 `pending_archive_restore`】那条路径经 `apply` + `merge` 走"整份
    /// 文档状态替换"，而部分载荷**故意**只带一部分（例如只带一条 modification 的参数），
    /// 走那条路会用空值覆盖掉其它状态。这里只做**按键合并**：载荷带了什么就覆盖什么。
    pub pending_partial_restore: Mutex<Vec<crate::state_channel::EditState>>,
    /// 最后一次成功编码的完整存档。
    ///
    /// 【为什么需要】宿主可以在模型图尚未 ready 时调用 `storeDocumentToArchive`
    /// （加载中的自动保存、工程关闭）。此前 `encode_archive` 直接返回
    /// `host graph not ready`，宿主据此认为插件存档失败，重开时编辑全丢。
    /// 现在 `ready == false` 时返回这份快照；恢复到达时也会先写一份，保证
    /// "恢复后还没合入就保存"不会把编辑覆盖成空。
    pub archive_cache: Mutex<Option<Vec<u8>>>,
    /// ARA 变更通知队列，由 `runtime.rs` 在构建插件时注入。
    ///
    /// 【为什么必须有】宿主只保存"它知道变过"的部分。插件从不通知，只存增量的宿主
    /// 就会丢数据 —— ARA 在 `ARAModelUpdateControllerInterface` 的三处文档里反复重申
    /// "必须可靠发送"（`ARAInterface.h:2098-2099` / `:2116-2117` / `:2144-2145`）。
    /// 此前插件**从未**调用过它（`mark_*` 在生产代码里出现次数为 0）。
    update_emitter: Mutex<Option<ara2_bridge::plugin::UpdateEmitter>>,
    /// modification persistentID → 宿主句柄身份 `(index, session)`。
    ///
    /// 通知要按**宿主句柄**投递（桥接层用它查 `ARAAudioModificationHostRef`），
    /// 而插件侧的状态是按 persistentID 索引的。`createAudioModification` /
    /// `cloneAudioModification` 是唯一拿得到句柄的时刻。
    ///
    /// 【为什么两个方向各存一份】通知要 persistentID → 句柄，而部分存档的写出侧
    /// （`StoreFilter` 只给句柄）要句柄 → persistentID。
    modification_handles: Mutex<HashMap<String, ara2_bridge::core::RawHandle>>,
    modification_ids: Mutex<HashMap<(std::num::NonZeroU32, std::num::NonZeroU64), String>>,
    editor: OnceLock<Result<Arc<crate::editor::session::EditorSession>, String>>,
}

/// 从 clip id 反解 region key。
///
/// 【为什么需要反解】clip id 由 region key 派生（`ara-region-{key}`，见
/// `ara::mapping::MappingIdentities`），所以它是自描述的。`clip_ids` 这张缓存表
/// 由 `remap_and_log` 整体重写，而 region 分配发生在它之前；在两者之间的窗口里
/// 反查缓存会落空。此时直接从 id 解析出 key，与缓存命中给出同一个值。
///
/// 位置派生的旧 id（`ara-clip-{N}`）**不可反解** —— 那正是它被替换掉的原因。
pub(crate) fn parse_region_key_from_clip_id(clip_id: &str) -> Option<u64> {
    clip_id
        .strip_prefix("ara-region-")?
        .parse::<u64>()
        .ok()
        .filter(|key| *key != 0)
}

/// 宿主音乐读数的快照；默认 120 BPM 4/4（与 `TimelineState::default` 同口径）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HostMeter {
    pub bpm: f64,
    pub numerator: u32,
    pub denominator: u32,
}

impl Default for HostMeter {
    fn default() -> Self {
        Self {
            bpm: 120.0,
            numerator: 4,
            denominator: 4,
        }
    }
}

/// 音阶投影缓存的键：设置代次 + 宿主音乐读数。
///
/// 【为什么宿主读数也在键里】BPM/拍号变化不经过设置写入，只看设置代次会让
/// Tempo Map 的宿主轴停在旧读数上。
///
/// 【为什么 tempo map 也在键里】宿主经 ARA 给出的多点 tempo map 是**独立于**单点
/// `host_meter` 的另一条读数（`host_meter` 来自 VST3 `ProcessContext`）。它变了而键
/// 不变的话，投影会一直用旧速度 —— 用户改工程速度后插件里的小节网格纹丝不动。
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MusicalProjectionKey {
    settings: u64,
    bpm_bits: u64,
    numerator: u32,
    denominator: u32,
    tempo_map: Option<crate::ara::model::HostTempoMap>,
}

impl MusicalProjectionKey {
    pub(crate) fn new(
        settings: u64,
        meter: HostMeter,
        tempo_map: Option<crate::ara::model::HostTempoMap>,
    ) -> Self {
        Self {
            settings,
            bpm_bits: meter.bpm.to_bits(),
            numerator: meter.numerator,
            denominator: meter.denominator,
            tempo_map,
        }
    }
}

/// 插件自有音阶在时间线上的投影结果。
#[derive(Clone, Debug)]
pub(crate) struct PluginMusicalProjection {
    pub project_scale_notes: Vec<u8>,
    pub tempo_map: Option<Vec<hifishifter_kernel::state::TempoPointData>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ara2_bridge::core::ApiGeneration;
    use ara2_bridge::plugin::{FactoryBuilder, PluginBuilder};
    use ara2_bridge::sys::*;

    /// **每一个 `ready = false` 的写点都必须留下"宿主刚改过内容"的时刻。**
    ///
    /// 【为什么这是重开工程的回归】宿主授权 PCM 走的是
    /// `refresh_source_pcm` → `revoke_renderers()` → `revoke_snapshots()`。
    /// 那条路径此前**不写** `last_content_change`，于是 `ensure_loaded` 把它当成
    /// "宿主声称输入不可用、而我们没有观察到任何变化" —— 直接返回
    /// `Conflict: host changed; local curves preserved, host is not ready`，
    /// 时间线再也刷不出来（用户看到的就是"所有 Clip 只显示等待宿主音频"）。
    #[test]
    fn revoking_renderers_records_a_content_change() {
        let document = Arc::new(DocumentSession::new(1));
        *document.last_content_change.lock_or_recover() = None;
        document.revoke_renderers();
        assert!(
            document.last_content_change.lock_or_recover().is_some(),
            "撤销渲染许可必须记下观察时刻，否则会被误判成宿主冲突"
        );
    }

    /// **T7/T8：源暂时不可读不得删掉已发布的编辑侧 PCM。**
    ///
    /// 【为什么这是"后台切回前台之后 Clip 一直等音频"的回归】旧实现无论什么原因都先
    /// 把 PCM 删掉、再"若可读则重读"。宿主收回一次样本访问（后台、undo 历史、宿主
    /// 自己整理缓存）就等于永久丢掉音频，而恢复只依赖"宿主记得再发一次 enable" ——
    /// 那条回调丢了就再也不会来。
    ///
    /// 钉住三件事：① 收回时**渲染权威表**撤掉（不能继续播旧采样）、**编辑侧副本**保留
    /// （波形不消失）；② 状态记为 `Revoked`（界面据此说"暂时收回"而不是"还在分配"）；
    /// ③ 重新授权后状态回到 `Published`。
    /// **I3-1：解不开的存档也必须原样交回宿主。**
    ///
    /// 【为什么这是"参数全丢"的最后一环】解码失败时 `restore_archive_with_remap` 只
    /// 记一行日志就 `Ok(())` —— 宿主以为恢复成功。此后宿主任何一次保存都会问
    /// `store_document`，而旧实现用空 `edits` 重编码并**覆盖** `archive_cache`：
    /// 磁盘上从此只剩默认值，没有任何一步报错。
    #[test]
    fn an_undecodable_archive_is_still_handed_back() {
        let document = Arc::new(DocumentSession::new(31));
        let payload = b"{ this is not a valid archive".to_vec();
        document.restore_archive(&payload).expect("host restore continues");
        assert_eq!(
            document.archive_cache.lock_or_recover().as_ref(),
            Some(&payload),
            "解码失败不得丢掉宿主交给我们的字节"
        );
        assert_eq!(document.encode_archive().unwrap(), payload);
    }

    /// **I3-2：载荷还没合入时，存档必须交回它，而不是用内存态覆盖。**
    ///
    /// 【为什么连续保存三次】第一次覆盖之后用户数据就不在磁盘上了；只测一次会漏掉
    /// "第一次侥幸还留着、第二次被固化"的那种退化。
    #[test]
    fn a_pending_restore_is_never_overwritten_by_the_in_memory_state() {
        let document = Arc::new(DocumentSession::new(32));
        let mut saved = crate::state_channel::EditState::default();
        saved.revision = 7;
        let bytes = saved.encode().unwrap();
        document.restore_archive(&bytes).expect("archive restores");
        for round in 0..3 {
            assert_eq!(
                document.encode_archive().unwrap(),
                bytes,
                "第 {round} 次保存改写了未合入的载荷"
            );
        }
    }

    #[test]
    fn a_revoked_source_keeps_its_editor_copy_and_reports_revoked() {
        use crate::render::document::SourceState;
        let document = Arc::new(DocumentSession::new(21));
        let id = "ara://source".to_string();
        let pcm = Arc::new(crate::render::source::SourcePcm {
            sample_rate: 44100,
            planes: vec![vec![0.1, 0.2, 0.3, 0.4]],
            version: 0,
            _reservation: None,
        });
        document.publish_source_pcm(id.clone(), pcm);
        document
            .source_states
            .lock_or_recover()
            .insert(id.clone(), SourceState::Published);
        // 宿主收回访问：渲染权威表撤掉，编辑侧副本保留。
        document.sources.lock_or_recover().remove(&id);
        document
            .source_states
            .lock_or_recover()
            .insert(id.clone(), SourceState::Revoked);
        assert!(
            document.edit_sources.lock_or_recover().contains_key(&id),
            "收回样本访问不得让界面丢掉已经物化的副本（那正是'整条轨突然空了'）"
        );
        assert!(
            !document.sources.lock_or_recover().contains_key(&id),
            "渲染权威表必须撤掉：宿主已经收回访问，不能继续播旧采样"
        );
        assert_eq!(
            document.source_states.lock_or_recover().get(&id),
            Some(&SourceState::Revoked),
            "状态必须如实记为 Revoked，界面才能说'暂时收回'而不是'还在分配'"
        );
        document.close();
        assert!(
            document.source_states.lock_or_recover().is_empty(),
            "文档关闭必须清空源状态表"
        );
    }

    /// **T23：宿主不再收口时，插件侧必须能把这件事变成一次可执行的请求。**
    ///
    /// 【为什么这是"删掉插件再加就好"的回归】`ready` 回到真只依赖宿主发起的四个
    /// 入口（`endEditing`、`notifyModelUpdates`、样本访问变更、存档恢复）。宿主一旦
    /// `beginEditing` 之后再也不收口，四个入口**一个都不会来** —— 插件永久停在未就绪，
    /// 而用户唯一能做的就是重装插件。
    ///
    /// 这里钉住三件事：① 会话开始的时刻被记下；② 超过宽限期后请求被置位；
    /// ③ 请求**取走即清**（否则每次回调都会重跑一次全量重映射）。
    #[test]
    fn an_overdue_edit_session_becomes_a_settle_request() {
        let document = Arc::new(DocumentSession::new(11));
        assert!(document.editing_elapsed().is_none(), "初始不在会话中");
        document.note_editing_started();
        assert!(document.editing_session.load(Ordering::Acquire));
        // 刚开始：还没到宽限期，不该请求。
        document.request_settle_if_overdue();
        assert!(!document.forced_edit_close.load(Ordering::Acquire));
        // 把起点推到宽限期之外（模拟"宿主再也不发 endEditing"）。
        document.backdate_settle();
        document.request_settle_if_overdue();
        assert!(
            document.forced_edit_close.load(Ordering::Acquire),
            "超过宽限期的会话必须变成一次可执行的收口请求"
        );
        // 取走即清：同一次请求只消费一次。
        assert!(document.forced_edit_close.swap(false, Ordering::AcqRel));
        assert!(!document.forced_edit_close.swap(false, Ordering::AcqRel));
        // 收口之后截止期归零，不再反复请求。
        document.note_editing_finished();
        document.request_settle_if_overdue();
        assert!(!document.forced_edit_close.load(Ordering::Acquire));
        document.close();
    }

    /// **T23：积压的模型变化同样有截止期。**
    ///
    /// 【为什么要单独一条】它与"会话没关"是两件不同的事：宿主可能根本不开编辑会话，
    /// 却在后台重建了图。那时唯一的 flush 入口是 `notifyModelUpdates`，宿主不调用就
    /// 永远不 flush —— 界面停在旧时间线上，而 `ready` 可能是真的（更隐蔽）。
    #[test]
    fn an_overdue_model_change_becomes_a_flush_request() {
        let document = Arc::new(DocumentSession::new(12));
        assert!(document.model_dirty_elapsed().is_none());
        document.note_model_dirty();
        document.note_model_dirty(); // 重复置脏不得推迟截止期
        let first = document.model_dirty_since_ms.load(Ordering::Acquire);
        assert_eq!(
            document.model_dirty_since_ms.load(Ordering::Acquire),
            first,
            "截止期从**第一次**变脏算起"
        );
        document.request_settle_if_overdue();
        assert!(!document.forced_model_flush.load(Ordering::Acquire));
        document.backdate_settle();
        document.request_settle_if_overdue();
        assert!(document.forced_model_flush.load(Ordering::Acquire));
        // 消费掉这一次请求，再看"干净之后不再请求"。
        assert!(document.forced_model_flush.swap(false, Ordering::AcqRel));
        document.note_model_clean();
        document.request_settle_if_overdue();
        assert!(
            !document.forced_model_flush.load(Ordering::Acquire),
            "已发布的模型不得再触发 flush 请求"
        );
        document.close();
    }

    /// **持着 `transaction` 刷新文档不得自锁。**
    ///
    /// 【为什么必须有这条】用户报的"重新打开含本插件的 ARA 工程，宿主直接未响应"
    /// 就是一次同线程自锁：`prepare_renderers` 持着 `transaction` 调了
    /// `adopt_archive_restore` → `refresh_document`，后者又要取同一把不可重入的锁。
    /// 这里把契约钉在 `refresh_document_locked` 上：它**不得**再取 `transaction`。
    ///
    /// 【为什么用线程 + 超时而不是直接调用】自锁的失败形态是"永远不返回"，直接调用
    /// 会把测试套件一起挂死；有超时才是一个可判定的断言。
    #[test]
    fn refreshing_while_the_transaction_is_held_does_not_deadlock() {
        use std::time::Duration;
        let document = Arc::new(DocumentSession::new(1));
        let owner = Arc::new(ExtensionOwner::default());
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker = {
            let document = Arc::clone(&document);
            let owner = Arc::clone(&owner);
            std::thread::spawn(move || {
                let _transaction = document.transaction.lock_or_recover();
                owner.refresh_document_locked(&document);
                let _ = sender.send(());
            })
        };
        assert!(
            receiver.recv_timeout(Duration::from_secs(5)).is_ok(),
            "持锁刷新必须返回：`refresh_document_locked` 不得再取 `transaction`"
        );
        worker.join().expect("worker must not panic");
    }

    /// 真正的工厂必须在把控制器交给宿主前登记身份，空文档也必须可绑定。
    #[test]
    fn native_factory_registers_the_actual_controller_identity() {
        let factory = FactoryBuilder::new("org.hfs.identity", "org.hfs.identity.archive")
            .display(
                "identity",
                "HiFiShifter",
                "https://github.com/ARounder-183/HiFiShifter",
                "1",
            )
            .document_controller(|| {
                let model = crate::ara::model::ModelHandle::new();
                let session = model.session();
                PluginBuilder::new(model)
                    .controller_identity(move |key| session.register(key))
                    .build()
            })
            .build()
            .unwrap();
        factory
            .entry()
            .initialize(ApiGeneration::V2Final, crate::test_host::assert_address())
            .unwrap();
        let mut fixture = crate::test_host::HostFixture::new(vec![]);
        let host = fixture.instance();
        let properties = ARADocumentProperties {
            structSize: std::mem::size_of::<ARADocumentProperties>(),
            name: c"empty document".as_ptr(),
        };
        // SAFETY: factory、fixture、host 和属性在控制器整个生命周期内存活。
        let raw = unsafe {
            (factory
                .raw_copy()
                .createDocumentControllerWithDocument
                .unwrap())(&host, &properties)
        };
        assert!(!raw.is_null());
        // SAFETY: 工厂成功返回稳定 controller instance。
        let instance = unsafe { raw.read_unaligned() };
        assert!(DocumentSession::lookup(instance.documentControllerRef as usize).is_some());
        // SAFETY: 唯一终止回调；之后不再访问控制器。
        unsafe {
            ((*instance.documentControllerInterface)
                .destroyDocumentController
                .unwrap())(instance.documentControllerRef)
        };
        assert!(DocumentSession::lookup(instance.documentControllerRef as usize).is_none());
        factory.entry().uninitialize().unwrap();
    }
}

fn controllers() -> &'static Mutex<HashMap<usize, Weak<DocumentSession>>> {
    static DOCUMENTS: OnceLock<Mutex<HashMap<usize, Weak<DocumentSession>>>> = OnceLock::new();
    DOCUMENTS.get_or_init(Default::default)
}

impl DocumentSession {
    pub(crate) fn publish_source_pcm(&self, id: String, pcm: Arc<super::source::SourcePcm>) {
        let _transaction = self.transaction.lock_or_recover();
        self.edit_sources
            .lock_or_recover()
            .insert(id.clone(), pcm.clone());
        self.sources.lock_or_recover().insert(id, pcm);
        self.editor_audio_revision.fetch_add(1, Ordering::AcqRel);
    }
    /// region key → clip id；`clip_ids` 缺席时按映射层同一算式补出来。    ///
    /// 【为什么必须补而不是报错】`clip_ids` 由 `remap_and_log` 整体重写，而 region 的
    /// 分配（`addPlaybackRegion`）发生在**那之前**。两者之间任何一次渲染、提交或投影
    /// 都会查到缺席。把这个窗口从"致命错误"降成"按 key 派生"—— 派生的 id 与映射层
    /// 给的完全一致（`ara-region-{key}`），所以下一次重算会把它原样认领回去，
    /// 会话内身份保持单调稳定。
    ///
    /// `key == 0` 是无效键（`region_owners` 也拒它），此时无身份可派生，返回 `None`。
    pub(crate) fn clip_id_for(&self, key: u64) -> Option<String> {
        if key == 0 {
            return None;
        }
        Some(
            self.clip_ids
                .lock_or_recover()
                .get(&key)
                .cloned()
                .unwrap_or_else(|| format!("ara-region-{key}")),
        )
    }
    /// 记录宿主经 ARA 给出的 tempo map；`None` 表示宿主没提供。
    ///
    /// 【为什么要比较再写】投影缓存以它为键的一部分，而宿主可能在每次
    /// `updateMusicalContextContent` 上重复给同一份内容 —— 无条件写会让缓存永远失效、
    /// 每帧重建 tempo 点。
    pub(crate) fn set_host_tempo_map(&self, map: Option<crate::ara::model::HostTempoMap>) {
        let mut slot = self.host_tempo_map.lock_or_recover();
        if *slot == map {
            return;
        }
        *slot = map;
        drop(slot);
        // tempo map 变了 ⇒ 音乐投影必须重建（缓存键含它，这里只是立刻让读数可见）。
        log::info!("[ara] host tempo map updated");
    }

    /// 编码 ARA 存档载荷（文档级）。
    ///
    /// 【为什么文档级而不是 per-renderer】`storeDocumentToArchive` 是**文档控制器**的
    /// 回调，载荷应当覆盖整份文档的编辑；而 `ExtensionOwner::encode_state` 会按
    /// "本 renderer 被分配的 region"收窄 —— 那是 VST3 组件状态的语义（一个组件一份
    /// 状态），不是存档的语义。
    ///
    /// 【为什么先 reconcile】`TrackBindings` 是编辑归属的权威边；存档必须只带当前
    /// 模型图里真实存在的轨道/片段，否则重载后会出现指向不存在对象的编辑。
    pub(crate) fn encode_archive(&self) -> Result<Vec<u8>, String> {
        let _transaction = self.transaction.lock_or_recover();
        // 【I3：只要还有一份**没合入**的载荷，交回宿主的就是那一份】
        //
        // 这是"永不回退"最直接的一条实现，也是本轮把"修不好"与"丢数据"解耦的地方：
        // 内存里的 `edits` 此时**不是**用户的数据（载荷还没合进去），用它重编码再
        // 覆盖 `archive_cache`，等于在磁盘上把用户的参数换成默认值 —— 而宿主此后
        // 每一次保存都会把这个结果固化下去。
        //
        // 判据取"有没有未合入的载荷"，不取"合并成不成功"：合并成功时载荷已被取走，
        // 走下面的正常路径；合并不成功时载荷仍在，这里就把宿主当初给的那份原样还回去。
        if self.has_unmerged_restore() {
            let cached = self.archive_cache.lock_or_recover().clone();
            let pending = self
                .renderer_owners()
                .iter()
                .find_map(|owner| owner.pending_restore_bytes());
            if let Some(bytes) = cached.or(pending) {
                log::warn!(
                    "[ara] archive store returns the unmerged payload ({} byte(s)); the saved \
                     parameters are preserved instead of being replaced by the in-memory state",
                    bytes.len()
                );
                return Ok(bytes);
            }
        }
        // 【为什么不再硬失败】`storeDocumentToArchive` 返回错误时宿主会认为"插件存档
        // 失败"，有些宿主会据此跳过这份工程状态。图未就绪（加载中的自动保存、关闭时
        // 保存）不是"没有编辑可存"，而是"这次还没合并完"；返回最后一份成功快照
        // 永远好过让宿主丢掉整块状态。
        if !self.ready.load(Ordering::Acquire) {
            if let Some(cached) = self.archive_cache.lock_or_recover().clone() {
                log::warn!(
                    "[ara] archive store uses the last good snapshot ({} byte(s)); the host \
                     graph is not ready",
                    cached.len()
                );
                return Ok(cached);
            }
            let edits = self.edits.lock_or_recover();
            return match edits.encode() {
                Ok(bytes) => Ok(bytes),
                Err(error) => {
                    // 【为什么退回"最后一份成功快照"而不是空块】空块是**可解析的空编辑**：
                    // 宿主会把它当成"这个工程没有任何用户参数"写进文件，而恢复端读到的
                    // 是一份合法的、全默认的状态 —— 没有任何一处会报错。那正是用户看到
                    // "重开工程后参数全丢"的最后一环。
                    log::error!("[ara] archive store degraded to the last good snapshot: {error}");
                    Ok(self.archive_cache.lock_or_recover().clone().unwrap_or_default())
                }
            };
        }
        let mut edits = self.edits.lock_or_recover();
        if let Err(error) = edits.reconcile(&self.track_bindings.lock_or_recover()) {
            // 身份暂时对不上不该让整份存档失败；保留未裁剪状态，恢复端会逐条 reconcile。
            log::warn!("[ara] archive store reconcile failed: {error}");
        }
        match edits.encode() {
            Ok(bytes) => {
                *self.archive_cache.lock_or_recover() = Some(bytes.clone());
                Ok(bytes)
            }
            Err(error) => {
                // 同上：能交回上一份成功载荷就绝不交空块。
                log::error!("[ara] archive store degraded to the last good snapshot: {error}");
                Ok(self.archive_cache.lock_or_recover().clone().unwrap_or_default())
            }
        }
    }

    /// 接收 ARA 存档载荷，等模型图就绪后合入。
    ///
    /// 【为什么只暂存】恢复到达时模型图通常还没重建（ARA 先给存档、宿主再重新交付
    /// 模型），此刻没有轨道可绑、也没有 renderer 可合并。暂存到文档级，
    /// `prepare_renderers` 收口时再交给 owner 的既有合并路径。
    pub(crate) fn restore_archive(&self, bytes: &[u8]) -> Result<(), String> {
        // 【为什么整份存档走空映射】ARA 只在**带筛选**的恢复里给出
        // `audioModificationArchiveIDs → audioModificationCurrentIDs`
        // （`ARARestoreObjectsFilter`，`ARAInterface.h:2322-2333`）；整份文档恢复
        // 没有这张表，因为 persistentID 本就是持久的。这里仍走同一条带映射的路径，
        // 于是"将来宿主给得出映射"时只有调用点需要改，两条路不会各自漂移。
        self.restore_archive_with_remap(bytes, &[])
    }

    /// 接收存档载荷并按 `archive → current` 重映射权威键后暂存（两条恢复路径共用）。
    fn restore_archive_with_remap(
        &self,
        bytes: &[u8],
        remap: &[(String, String)],
    ) -> Result<(), String> {
        if bytes.is_empty() {
            return Ok(());
        }
        // 【I3-2：先存后解】在尝试解码**之前**就把宿主给的字节留在 `archive_cache` 里。
        //
        // 解码失败时下面的 `return Ok(())` 会让宿主以为恢复成功，而内存里什么都没有；
        // 此时若宿主再保存一次，旧实现会用空 `edits` 重编码并覆盖 cache —— 磁盘上
        // 从此只剩默认值。先把原始字节存下来，`encode_archive` 才有东西可以还回去。
        *self.archive_cache.lock_or_recover() = Some(bytes.to_vec());
        let mut restored = crate::state_channel::EditState::default();
        // 【为什么解码失败也只记日志】宿主把 `beginRestoringDocumentFromArchive`
        // 的失败当成"整份文档恢复失败"：region 不再交付、界面永远停在
        // `awaiting_regions`。丢一份编辑是局部损失，让宿主中止恢复是整份工程不可用。
        if let Err(error) = restored.restore(bytes) {
            log::error!(
                "[ara] archive restore ignored: {error}; the host document restore continues \
                 without these edits"
            );
            return Ok(());
        }
        // 【为什么必须重映射】persistentID 只保证**文档内**唯一，跨文档导入时宿主
        // 要求把存档 ID 映射到当前图里的 ID。不映射会把参数挂到一个不存在的键上 ——
        // 静默丢失，症状正是"重载工程后参数被重置"。
        restored.atlas.remap_authorities(remap);
        // 【为什么不再把 `restored` 重编码后写回 cache】恢复后到合入前，宿主可能再次
        // 保存（甚至还没等模型图 ready），需要一份"能交回去的东西" —— 那已经在函数
        // 开头存好了：**宿主给的原始字节**。再写一份重编码的副本有两个害处：
        // ① 它经过 `restore` 的 revision 前进，与宿主手里的那份不再逐字相同，
        //    "原样还回去"这条不变量就断了；
        // ② 它是"我们的"字节，而宿主下一次保存会把它当成权威 —— 一旦合并逻辑还有
        //    缺陷，被固化的就是我们的误解，而不是用户的数据。
        *self.pending_archive_restore.lock_or_recover() = Some(restored);
        Ok(())
    }

    /// 调用方持document事务；仅从真实region边创建曲线身份，不从轨名/源路径推断。
    ///
    /// 【为什么不再对单个 clip 硬失败】此前这里对**任意一条** clip 的身份缺席直接
    /// `Err`，而它在 `map` 内 ⇒ 整份身份表失败 ⇒ 所有 `project_*` / `follow_geometry`
    /// 调用全部失败 ⇒ 整份投影与整份编辑被拒。这与本文件写在 `clip_id_for` 上的口径
    /// 自相矛盾：那里已经把"region 已分配、`clip_ids` 还没重算"的窗口降级为按 key
    /// 派生，而这里仍把它当致命错误。
    ///
    /// 现在逐条跳过并限流留痕：缺席的那一条 clip 不参与曲线投影（几何与音频不受影响），
    /// 其余 clip 照常。返回类型保留 `Result` 是为了调用方签名不变 —— 本函数已不再
    /// 因单条 clip 而失败。
    pub(crate) fn parameter_identities_locked(
        &self,
        timeline: &hifishifter_kernel::state::TimelineState,
    ) -> Result<
        std::collections::BTreeMap<String, crate::editor::parameter_atlas::RegionIdentity>,
        String,
    > {
        let ids = self.clip_ids.lock_or_recover();
        let regions = self.regions.lock_or_recover();
        let items = self.region_items.lock_or_recover();
        let mut unresolved = 0_usize;
        let mut first_unresolved: Option<String> = None;
        let identities = timeline
            .clips
            .iter()
            .filter_map(|clip| {
                // 反查 region key：先查缓存，未命中时从 id 本身解析 —— clip id 由 key 派生
                // （`ara-region-{key}`），所以它是自描述的。此前的硬错误
                // （`missing actual region parameter edge`）会让"region 已分配、`clip_ids`
                // 还没重算"这个窗口里的每次投影都失败。
                let key = ids
                    .iter()
                    .find(|(_, id)| **id == clip.id)
                    .map(|(key, _)| *key)
                    .or_else(|| parse_region_key_from_clip_id(&clip.id));
                let region = key
                    .and_then(|key| regions.get(&key).map(|region| (key, region)))
                    // 【按 key 直查缺席时，按"源 + 播放几何"反查】
                    //
                    // 少数情况下 clip id 派生出的 key 不在派生表里（宿主在收口前重投模型，
                    // 或 region 刚被重建而 `region_keys` 还没跟上）。此前这种情况直接判
                    // "无身份"，于是这一条 clip 本轮**不投影任何曲线** —— 用户看到的正是
                    // "编辑之后这一段参数被重置"。
                    //
                    // clip 携带的 `source_path` 就是源 persistentID（映射层如此播种），
                    // `start_sec` / `length_sec` 就是播放起止，三者足以在派生表里确定那条
                    // region。多于一条匹配说明猜不出（同一源、同一窗口被重复放置），
                    // 此时仍放弃 —— 不猜。
                    .or_else(|| {
                        let source = clip.source_path.as_deref()?;
                        let mut matches = regions.iter().filter(|(_, region)| {
                            region.audio_source_persistent_id == source
                                && (region.start_in_playback_time - clip.start_sec).abs() <= 1e-6
                                && (region.duration_in_playback_time - clip.length_sec).abs()
                                    <= 1e-6
                        });
                        let (key, region) = matches.next()?;
                        if matches.next().is_some() {
                            return None;
                        }
                        Some((*key, region))
                    });
                let Some((key, region)) = region else {
                    unresolved += 1;
                    first_unresolved.get_or_insert_with(|| clip.id.clone());
                    return None;
                };
                Some((
                    clip.id.clone(),
                    crate::editor::parameter_atlas::RegionIdentity {
                        key,
                        item: items.get(&key).cloned(),
                        source: region.audio_source_persistent_id.clone(),
                        modification: region.audio_modification_persistent_id.clone(),
                        // 跨会话签名：只用 ARA 自己给的量，ARA-only 宿主上也有值
                        // （`item` 在那种宿主上恒为 `None`，谱系全靠它）。
                        signature:
                            crate::editor::parameter_atlas::ParameterAtlas::ara_region_signature(
                                &region.audio_source_persistent_id,
                                &region.audio_modification_persistent_id,
                                clip.start_sec,
                                clip.length_sec,
                            ),
                    },
                ))
            })
            .collect();
        if unresolved > 0 {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.identity.unresolved-clip",
                format_args!(
                    "[ara] {unresolved} of {} clip(s) have no region identity this round \
                     (first: {:?}); they project no curves until the derived tables catch up",
                    timeline.clips.len(),
                    first_unresolved
                ),
            );
        }
        Ok(identities)
    }
    /// 文档惰性持有唯一原编辑actor；actor只弱引用文档，组件不持第二份history。
    pub(crate) fn editor_session(
        self: &Arc<Self>,
    ) -> Result<Arc<crate::editor::session::EditorSession>, String> {
        // 【为什么先释放事务再建会话】`EditorSession::new` 会加载模型、创建 actor、
        // 扫临时目录；这些都不是"读文档状态"，却会把 `transaction` 按住数秒。
        // ARA 模型线程（`remap_and_log` / `prepare_renderers`）等同一把锁，宿主于是
        // 表现为未响应；UI 线程上的 `get_runtime_info` 也会排在这段时间后面。
        // 事务只用来回答"文档此刻还活着吗"这一个原子问题。
        {
            let _transaction = self.transaction.lock_or_recover();
            if !self.is_alive() {
                return Err(crate::editor::errors::DOCUMENT_CLOSED.error());
            }
        }
        // 初始化期间文档可能被关闭：会话只弱引用文档，后续命令会自然返回
        // `DOCUMENT_CLOSED`，不需要在这里把构造失败当成终态。
        self.editor
            .get_or_init(|| crate::editor::session::EditorSession::new(self))
            .clone()
    }
    /// 编辑器会话是否已经创建（不创建、不取事务）。
    pub(crate) fn editor_initialized(&self) -> bool {
        self.editor
            .get()
            .is_some_and(|result| result.as_ref().is_ok())
    }

    /// 已发布片段数（健康探活用；不创建会话）。
    pub(crate) fn editor_clip_count(&self) -> u64 {
        self.editor
            .get()
            .and_then(|result| result.as_ref().ok())
            .map_or(0, |session| session.clip_count())
    }

    /// 已经创建的编辑器会话（**不创建**）。
    pub(crate) fn existing_editor_session(
        &self,
    ) -> Option<Arc<crate::editor::session::EditorSession>> {
        self.editor
            .get()
            .and_then(|result| result.as_ref().ok())
            .cloned()
    }

    /// 有预算的编辑器 flush；用于宿主保存线程，不能等 10 秒。
    pub(crate) fn flush_editor_with_budget(
        &self,
        budget: std::time::Duration,
    ) -> Result<(), String> {
        if let Some(Ok(editor)) = self.editor.get() {
            editor.flush_with_budget(budget)?;
        }
        Ok(())
    }
    /// 组件撤销只关闭失效入口的view，不创建actor也不停止同文档其它入口。
    pub(crate) fn revoke_editor_views(&self) {
        if let Some(Ok(editor)) = self.editor.get() {
            editor.revoke_closed_views();
        }
    }
    /// edit/model/scope分开计数；assignment变更不能由相同model误判成未变。
    pub(crate) fn editor_versions(&self) -> Result<(u64, u64, u64), String> {
        let _transaction = self.transaction.lock_or_recover();
        if !self.is_alive() {
            return Err(crate::editor::errors::DOCUMENT_CLOSED.error());
        }
        Ok((
            self.edits.lock_or_recover().revision,
            self.revision.load(Ordering::Acquire),
            self.scope_revision.load(Ordering::Acquire),
        ))
    }
    fn workspace_projection_locked(
        &self,
        edits: &crate::state_channel::EditState,
    ) -> Result<String, String> {
        let mut timeline = self.workspace_timeline_locked()?;
        edits.apply(&mut timeline);
        // 【为什么音阶也在指纹里】`ensure_loaded` 靠"版本没变 + 本指纹相同"早退
        // （`editor/session.rs:819-843`）。音阶与音阶变化点都是**会影响渲染结果**的
        // 插件自有状态（`scale-signature` 进渲染缓存键），却不在版本号里 —— 不进指纹
        // 就会出现"改了音阶，早退成立，于是既不重载也不重渲染"。与 `fades` 同列，
        // 理由相同。
        //
        // 【为什么可以整体哈希】音阶点的 id 是确定性的（见
        // `plugin_musical_projection`），宿主读数带 1e-6 阈值，所以"内容没变 ⇒
        // 指纹相同"成立，不会造成每轮重载。
        let bytes = serde_json::to_vec(&(
            timeline.params_by_root_track,
            timeline.tracks,
            &edits.fades,
            &edits.groups,
            &timeline.project_scale_notes,
            &timeline.tempo_map,
            // 【为什么渲染事实也进指纹】`ensure_loaded` 靠"版本没变 + 本指纹相同"早退。
            // 方向 / 循环源 / 声道模式由 `project_host_take_facts_locked` 播种进**clips**
            // —— 而 clips 不在上面的哈希里。不进指纹就会出现"宿主倒放了，早退成立，
            // 于是既不重载也不重渲染"，与音阶进指纹是同一条理由。
            &*self.render_facts.lock_or_recover(),
        ))
        .map_err(|e| e.to_string())?;
        Ok(format!(
            "{}:{}",
            self.scope_revision.load(Ordering::Acquire),
            blake3::hash(&bytes).to_hex()
        ))
    }
    /// scope版本参与投影指纹；区域退出后即便edit/model没动也不能接受旧工作区。
    pub(crate) fn workspace_projection(&self) -> Result<String, String> {
        let _transaction = self.transaction.lock_or_recover();
        self.workspace_projection_locked(&self.edits.lock_or_recover())
    }
    /// 原GUI改变素材选择只换可见source投影；不修改参数authority、history或渲染generation。
    pub(crate) fn selected_source_parameters(
        &self,
        selection: &hifishifter_kernel::state::TimelineState,
    ) -> Result<
        std::collections::BTreeMap<String, hifishifter_kernel::state::TrackParamsState>,
        String,
    > {
        let _transaction = self.transaction.lock_or_recover();
        if !self.is_alive() {
            return Err(crate::editor::errors::DOCUMENT_CLOSED.error());
        }
        let host = self.workspace_timeline_locked()?;
        Self::validate_workspace_geometry(&host, selection)?;
        let edits = self.edits.lock_or_recover();
        let mut grouped = host;
        edits.apply(&mut grouped);
        grouped.selected_clip_id = selection.selected_clip_id.clone();
        self.project_private_group_view(&mut grouped, &edits)?;
        Ok(grouped.params_by_root_track)
    }
    /// 短事务同时冻结授权、参数、PCM和三个版本，分析副本的文件IO由actor随后执行。
    pub(crate) fn workspace_snapshot(&self) -> Result<(WorkspaceSnapshot, u64, String), String> {
        let _transaction = self.transaction.lock_or_recover();
        for owner in self.renderer_owners() {
            owner.merge_pending_restore(self)?;
        }
        // 【为什么先取 `edits` 再取时间线】文档声明的锁序是
        // `transaction → timeline → edits → host facts`（见 `DocumentSession` 的说明），
        // 而 `workspace_timeline_locked` 内部要取 `clip_ids` / `ui_tracks`（host facts）。
        // 先取时间线就是 `host facts → edits`，与 `accept_workspace_edits_with_groups`
        // 的 `edits → host facts`（它持 `edits` 时调 `parameter_identities_locked`）
        // 构成 ABBA —— 两条路径分别跑在编辑器 actor 与 ARA 模型线程上，是真死锁候选。
        // `workspace_timeline_locked` 本身不碰 `edits`，所以把顺序倒过来是安全的。
        let mut edits = self.edits.lock_or_recover();
        let mut timeline = self.workspace_timeline_locked()?;
        edits.reconcile(&self.track_bindings.lock_or_recover())?;
        edits.apply(&mut timeline);
        // 【为什么先只取一份 id 集合、立刻释放锁】`source_failures` 与 `edit_sources`
        // 此前从不同时持有；`media_in_flight_items` 的锁序是
        // `regions → region_items → source_failures`。在这里引入 `edit_sources →
        // source_failures` 会给锁序图加一条新边。id 集合很小（每条 clip 一个），
        // 先取出来再释放，既省锁也避免新的 ABBA 候选。
        let available_ids: std::collections::BTreeSet<String> = {
            let available = self.edit_sources.lock_or_recover();
            available.keys().cloned().collect()
        };
        // 【为什么逐条降级而不是整份失败】此前这里对**每一条** clip 的源做
        // `ok_or(HOST_PCM_UNAVAILABLE)?` 再 `collect::<Result<_,_>>()?` —— 只要工程里有
        // **一条** clip 的源还没被 ARA 授权，整个 `workspace_snapshot` 就返回 `Err`，
        // 而它唯一的调用方 `EditorSession::ensure_loaded` 直接 `?` 上去 ⇒ **插件界面
        // 一条片段都不更新**。用户看到的是"宿主里做了操作，插件毫无反应"。
        //
        // 这与本文件自己写在 `source_failures` 上的纪律直接冲突（见其字段说明：
        // "把'某个源读不出来'折进 `ready=false` 会让一次样本读取失败被报成'宿主冲突'，
        // 所以这里单记一份"），也违反项目既有的"局部降级、不整份失败"口径 ——
        // `mapping.rs` 的逐 region 跳过、`project_audio_fades_locked` 的缺几何降级
        // 都是同一纪律的既有实现。
        //
        // 降级方式沿用内核 `host_pcm` 的"隔离"口径：缺源的 clip **清掉源引用**，
        // 界面按无源占位渲染（不是消失），原因记进 `source_failures` 供前端显示
        // "源未授权"。这样其余的 clip、曲线、几何全部照常。
        {
            for clip in &mut timeline.clips {
                for take in &mut clip.takes {
                    let Some(id) = take.source_path.as_deref() else {
                        continue;
                    };
                    if available_ids.contains(id) {
                        continue;
                    }
                    // 【B2：这里**不再**写 `source_failures`】
                    //
                    // 旧实现在这里给每个"这一轮不在 `edit_sources` 里"的源记一条
                    // `HOST_PCM_UNAVAILABLE`，而 `decorate_host_media_locked` 把"有失败记录"
                    // 判成 `unavailable:source_read_failed` —— 一个**终态**。于是"这一轮还没读到"
                    // 被永久写成了"读不出来"，而它本来下一个 tick 就会自愈（插件侧会重读）。
                    //
                    // 真正读不出来的真因只有一个写者：`refresh_source_pcm` 的 `Err` 分支。
                    // 让那一条独占这张表，"等"与"坏"才分得开。
                    take.source_path = None;
                    take.source_path_relative = None;
                }
                // 活跃 take 的源被清掉后，扁平投影必须跟着清 —— 否则下游会拿一个
                // 指向已消失源的 `clip.source_path` 去物化，那正是"missing host PCM"。
                clip.normalize_takes();
            }
        }
        let available = self.edit_sources.lock_or_recover();
        let ids = timeline
            .clips
            .iter()
            .flat_map(|clip| {
                std::iter::once(&clip.source_path)
                    .chain(clip.takes.iter().map(|take| &take.source_path))
            })
            .filter_map(Option::as_ref)
            .collect::<std::collections::BTreeSet<_>>();
        // 到这里 `ids` 只含已授权的源；仍用 `filter_map` 兜住"授权表在两步之间变化"
        // 的竞态 —— 那种情况下也只是少一条 PCM，不该赔上整份快照。
        let sources = ids
            .into_iter()
            .filter_map(|id| available.get(id).map(|pcm| (id.clone(), pcm.clone())))
            .collect::<Vec<_>>();
        let projection = self.workspace_projection_locked(&edits)?;
        self.project_ui_fades_locked(&mut timeline, &edits.fades);
        self.project_private_group_view(&mut timeline, &edits)?;
        Ok((
            WorkspaceSnapshot {
                timeline,
                sources,
                revision: edits.revision,
                model_revision: self.revision.load(Ordering::Acquire),
                audio_revision: self.editor_audio_revision.load(Ordering::Acquire),
            },
            self.scope_revision.load(Ordering::Acquire),
            projection,
        ))
    }
    /// 原分析副本可补媒体元信息，除此之外clip、take及轨道结构全部必须与宿主一致。
    fn validate_workspace_geometry(
        host: &hifishifter_kernel::state::TimelineState,
        client: &hifishifter_kernel::state::TimelineState,
    ) -> Result<(), String> {
        let normalize=|timeline:&hifishifter_kernel::state::TimelineState|->Result<serde_json::Value,String> {
            let mut timeline=timeline.clone();for clip in &mut timeline.clips {clip.normalize_takes();}
            let mut value=serde_json::to_value(timeline).map_err(|e|e.to_string())?;
            let metadata=["source_path_relative","duration_sec","duration_frames","source_sample_rate","source_channels",
                "source_file_fingerprint","source_file_mtime","source_file_size","waveform_preview","pitch_range"];
            if let Some(clips)=value["clips"].as_array_mut() {for clip in clips {
                for key in metadata {clip.as_object_mut().unwrap().remove(key);}
                if let Some(takes)=clip["takes"].as_array_mut() {for take in takes {for key in metadata {take.as_object_mut().unwrap().remove(key);}}}
            }}
            if let Some(tracks)=value["tracks"].as_array_mut() {for track in tracks {for key in ["volume","muted","solo","compose_enabled","pitch_analysis_algo"] {track.as_object_mut().unwrap().remove(key);}}}
            Ok(serde_json::json!({"tracks":value["tracks"],"clips":value["clips"],"project_sec":value["project_sec"]}))
        };
        if normalize(host)? != normalize(client)? {
            return Err("host workspace geometry is read-only".into());
        }
        Ok(())
    }
    /// 全工作区在单事务校验scope/模型/曲线并接受；未知范围或几何不静默忽略。
    // 整批接受入口保留，供不分组调用方使用。
    #[cfg(test)]
    pub(crate) fn accept_workspace_edits(
        &self,
        base_edit: u64,
        base_model: u64,
        client: &hifishifter_kernel::state::TimelineState,
        previous: &str,
    ) -> Result<(u64, u64, String), String> {
        self.accept_workspace_edits_with_groups(base_edit, base_model, client, previous, None)
    }
    /// 组根处理设置与参数delta同一事务提交，避免空父轨设置在音频准备时退回默认值。
    pub(crate) fn accept_workspace_edits_with_groups(
        &self,
        base_edit: u64,
        base_model: u64,
        client: &hifishifter_kernel::state::TimelineState,
        previous: &str,
        groups: Option<crate::editor::private_groups::TrackGroups>,
    ) -> Result<(u64, u64, String), String> {
        let _transaction = self.transaction.lock_or_recover();
        let model = self.revision.load(Ordering::Acquire);
        if model != base_model {
            return Err(format!(
                "{} model changed; local curves preserved",
                crate::editor::errors::CONFLICT_PREFIX
            ));
        }
        let host = self.workspace_timeline_locked()?;
        // 【为什么这里用**检查版**取锁】这是"读改写并影响宿主"的路径：`candidate` 由
        // `edits` 推导，最后写回 `*edits`。若上一次持锁者在 `merge`/`capture_changes`
        // 中途 panic，`edits` 可能停在半写状态 —— 拿它推导再写回，会把一次局部 panic
        // 升级成工程级的编辑损坏（而它随后会经 VST3 `getState` 落进宿主工程）。
        // 宁可放弃这一次编辑并留痕，也不要拿半写状态去改宿主。
        let (mut edits, poisoned) = self.edits.lock_or_recover_checked();
        if poisoned {
            log::error!(
                "[ara] workspace edit refused: the edit authority was left half-written by an                  earlier panic; this edit is dropped instead of being derived from it"
            );
            return Err(
                "Conflict: edit authority is in an unknown state after a panic; the edit was                  not applied"
                    .into(),
            );
        }
        if self.workspace_projection_locked(&edits)? != previous || base_edit > edits.revision {
            return Err(
                "Conflict: workspace scope or curves changed; local curves preserved".into(),
            );
        }
        Self::validate_workspace_geometry(&host, client)?;
        let mut candidate = edits.merge(&host, client, edits.revision)?;
        candidate.reconcile(&self.track_bindings.lock_or_recover())?;
        let identities = self.parameter_identities_locked(&host)?;
        let mut previous_view = edits.params.clone();
        if !edits.atlas.is_empty() {
            let mut selected = host.clone();
            selected.selected_clip_id = client.selected_clip_id.clone();
            previous_view.extend(edits.atlas.project_roots(&selected, &identities)?);
        }
        candidate.atlas = edits
            .atlas
            .capture_changes(client, &identities, &previous_view)?;
        if let Some(groups) = groups {
            groups.validate()?;
            candidate.groups = groups;
        }
        let projection = self.workspace_projection_locked(&candidate)?;
        *edits = candidate;
        // 用户的参数编辑落盘了 ⇒ 告诉宿主"这些 modification 的持久状态变了"。
        // 宿主只保存它知道变过的部分；不通知就会在它做增量保存时丢掉这次编辑。
        self.notify_parameters_changed();
        Ok((edits.revision, model, projection))
    }
    /// 合成只在短事务外计算；最终再次核对文档、scope、assignment与编辑代次。
    // 无进度回调的合成入口保留，供简单调用方使用。
    #[cfg(test)]
    pub(crate) fn apply_workspace_edits(
        &self,
        base_edit: u64,
        base_model: u64,
        previous: &str,
        cancel: Arc<AtomicBool>,
        current: impl Fn() -> bool,
    ) -> Result<(), String> {
        self.apply_workspace_edits_with_progress(
            base_edit, base_model, previous, cancel, current, None,
        )
    }
    pub(crate) fn apply_workspace_edits_with_progress(
        &self,
        base_edit: u64,
        base_model: u64,
        previous: &str,
        cancel: Arc<AtomicBool>,
        current: impl Fn() -> bool,
        progress: Option<hifishifter_kernel::mixdown::ProgressCallback>,
    ) -> Result<(), String> {
        let (edit, epoch, scope, inputs) = {
            let _transaction = self.transaction.lock_or_recover();
            // 【为什么不再比较 `revision != base_model`】`revision` 是"模型被失效过几次"的
            // 计数，宿主一次编辑会把它推进多次（分割在 ARA 侧是 destroy + create + create，
            // 每次 `clear_renderers()` 都 +1）。用它做冲突判据，等于把"宿主正常的多次交付"
            // 判成冲突：实测 Cubase 15 上这就报成 `Conflict: host model changed during
            // automatic apply`，8 次重试后自动渲染永久停摆。
            //
            // 真正的正确性判据在下方复核段：`edits.revision` / `render_epoch` /
            // `scope_revision` / `assigned_regions()` 任一变化都会走 `superseded`（可重试、
            // 不写 error）。而 `render_epoch` 与 `revision` 是**同时**推进的
            // （`clear_renderers` 是唯一同时推进两者的原语），所以这里去掉 `revision`
            // 不丢任何信息 —— 只是不再做那个会误报的快速失败。
            if !self.is_alive() || !self.ready.load(Ordering::Acquire) {
                return Err(format!(
                    "{} model changed during automatic apply",
                    crate::editor::errors::CONFLICT_PREFIX
                ));
            }
            let edits = self.edits.lock_or_recover().clone();
            if edits.revision != base_edit || self.workspace_projection_locked(&edits)? != previous
            {
                return Err("Conflict: workspace changed during automatic apply".into());
            }
            let inputs = self
                .renderer_owners()
                .into_iter()
                .filter(|owner| owner.renders_playback())
                .map(|owner| {
                    let (keys, input) = owner.capture_render_input(self, &edits, true)?;
                    Ok((owner, keys, input))
                })
                .collect::<Result<Vec<_>, String>>()?;
            (
                edits.revision,
                self.render_epoch.load(Ordering::Acquire),
                self.scope_revision.load(Ordering::Acquire),
                inputs,
            )
        };
        if !current() {
            return Err("automatic apply superseded".into());
        }
        let mut prepared = Vec::new();
        for (owner, keys, input) in inputs {
            for publisher in &owner.snapshots {
                publisher.collect_retired();
            }
            let mut snapshots = input.render_with_progress(cancel.clone(), progress.clone())?;
            super::snapshot::compact_prepared(&mut snapshots)?;
            prepared.push((owner, keys, snapshots));
        }
        let _transaction = self.transaction.lock_or_recover();
        // 复核只看"这份已算好的东西还能不能用"：活着、仍就绪，且下面那组
        // 版本/分配没有变。`revision` 不参与 —— 理由同前一处（见上面的说明）。
        if !self.is_alive() || !self.ready.load(Ordering::Acquire) {
            return Err(format!(
                "{} model changed during automatic apply",
                crate::editor::errors::CONFLICT_PREFIX
            ));
        }
        if !current()
            || self.edits.lock_or_recover().revision != edit
            || self.render_epoch.load(Ordering::Acquire) != epoch
            || self.scope_revision.load(Ordering::Acquire) != scope
        {
            return Err("automatic apply superseded".into());
        }
        for (owner, keys, snapshots) in &prepared {
            if owner.is_closed() || owner.assigned_regions().map_err(|e| e.to_string())? != *keys {
                return Err("automatic apply superseded".into());
            }
            if !owner
                .snapshots
                .iter()
                .zip(snapshots)
                .all(|(publisher, snapshot)| publisher.has_capacity(snapshot))
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
            owner.record_prepared(base_model, edit, epoch, scope, keys);
        }
        Ok(())
    }
    /// 宿主专属API调用前核对真实文档仍存活，销毁后的view不能沿旧project指针查询。
    pub(crate) fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }
    /// 会话与模型同寿，controller 地址在工厂 allocation 完成后登记。
    pub fn new(id: DocumentId) -> Arc<Self> {
        Arc::new(Self {
            id,
            alive: AtomicBool::new(true),
            ..Default::default()
        })
    }

    /// 按 ARA 2 的部分存档筛选产出载荷。
    ///
    /// 【为什么需要它】整份存档在"只改了一条 modification"时也要重写全部状态，而
    /// `storeObjectsToArchive` 的筛选正是为了按对象粒度保存（`ARAInterface.h:2979-2988`）。
    /// 不支持它时宿主会拿到 `kARAFalse`，于是它做增量保存时**丢失**这次编辑。
    ///
    /// `modifications == None` 表示"全部选中"；`document_data` 表示是否带上文档级状态
    /// （参数表 / 分组 / 空白编辑 / 逐片段私有属性）。
    pub(crate) fn encode_partial(
        &self,
        modifications: Option<&std::collections::BTreeSet<String>>,
        document_data: bool,
    ) -> Result<Vec<u8>, String> {
        let _transaction = self.transaction.lock_or_recover();
        if !self.ready.load(Ordering::Acquire) {
            // 部分存档没有"最后一份同筛选快照"可复用；返回完整缓存是安全的超集，
            // 宿主恢复时会按文档状态合入。没有缓存才返回空载荷（而不是失败）。
            if let Some(cached) = self.archive_cache.lock_or_recover().clone() {
                log::warn!(
                    "[ara] partial archive store uses the last full snapshot ({} byte(s)); \
                     the host graph is not ready",
                    cached.len()
                );
                return Ok(cached);
            }
            log::warn!(
                "[ara] partial archive store degraded to an empty payload; the host graph is \
                 not ready"
            );
            return Ok(Vec::new());
        }
        let mut edits = self.edits.lock_or_recover();
        if let Err(error) = edits.reconcile(&self.track_bindings.lock_or_recover()) {
            log::warn!("[ara] partial archive reconcile failed: {error}");
        }
        // 【为什么先整份克隆再裁剪】从 `EditState::default()` 起手会把**所有**状态
        // 一起丢掉 —— 包括本次真正要保存的那几条 modification 参数。裁剪必须是
        // "从完整状态里挑出被选中的部分"，而不是"从空状态里填"。
        let mut partial = edits.clone();
        // 参数权威是**逐 modification** 的状态：按筛选裁剪。
        partial.atlas.retain_authorities_for(modifications);
        partial.atlas.regions.clear();
        if !document_data {
            // 文档级状态没被选中 ⇒ 载荷里不带它，恢复端因此不会覆盖它。
            partial.params.clear();
            partial.tracks.clear();
            partial.bindings.clear();
            partial.fades.clear();
            partial.clip_local.clear();
            partial.groups = Default::default();
            partial.atlas.gaps.clear();
        }
        match partial.encode() {
            Ok(bytes) => Ok(bytes),
            Err(error) => {
                log::error!("[ara] partial archive store degraded to an empty payload: {error}");
                Ok(Vec::new())
            }
        }
    }

    /// 接收 ARA 2 的部分存档载荷，按 `archive → current` 重映射后暂存。
    pub(crate) fn restore_archive_partial(
        &self,
        bytes: &[u8],
        remap: &[(String, String)],
    ) -> Result<(), String> {
        if bytes.is_empty() {
            return Ok(());
        }
        let mut restored = crate::state_channel::EditState::default();
        // 与整份恢复同一条纪律：解码失败不能让宿主中止恢复。
        if let Err(error) = restored.restore(bytes) {
            log::error!(
                "[ara] partial archive restore ignored: {error}; the host document restore \
                 continues without these edits"
            );
            return Ok(());
        }
        restored.atlas.remap_authorities(remap);
        self.pending_partial_restore
            .lock_or_recover()
            .push(restored);
        Ok(())
    }
    /// 把暂存的部分载荷**按键合并**进编辑权威（`prepare_renderers` 收口时调用）。
    ///
    /// 【为什么是合并而不是替换】部分载荷只带它被选中的那一部分；替换会把未选中的
    /// 状态（其它 modification 的参数、文档级参数表）清空。
    fn merge_partial_restores(&self) {
        let pending = self
            .pending_partial_restore
            .lock_or_recover()
            .drain(..)
            .collect::<Vec<_>>();
        if pending.is_empty() {
            return;
        }
        let mut edits = self.edits.lock_or_recover();
        for partial in pending {
            for (key, params) in partial.atlas.authorities {
                edits.atlas.set_authority(&key, params);
            }
            edits.atlas.gaps.extend(partial.atlas.gaps);
            edits.params.extend(partial.params);
            if !partial.groups.is_empty() {
                edits.groups = partial.groups;
            }
            edits.fades.extend(partial.fades);
            edits.clip_local.extend(partial.clip_local);
            edits.revision = edits.revision.saturating_add(1);
        }
    }

    /// 本地补丁（T8）：接管本份文档控制器的图账本。
    ///
    /// 【为什么在文档上而不是各处传】诊断路径（`stuck_report` / 证据包）只需要
    /// "这一份文档的图走到哪一步"。记在这里就不必让每个调用方持有桥的类型。
    /// 【幂等】宿主每建一个控制器就接管一次；重绑时换成新账本，旧读数随之失效 ——
    /// 那正是我们要的（新文档从零开始记账）。
    pub(crate) fn set_graph_ledger(
        &self,
        ledger: std::sync::Arc<ara2_bridge::plugin::GraphLedger>,
    ) {
        *self.graph_ledger.lock_or_recover() = Some(ledger);
    }

    /// 图账本（没有接管过时为 `None`）。
    pub(crate) fn graph_ledger(&self) -> Option<std::sync::Arc<ara2_bridge::plugin::GraphLedger>> {
        self.graph_ledger.lock_or_recover().clone()
    }
    /// 注入 ARA 变更通知队列（`runtime.rs` 构建插件时调用一次）。
    pub(crate) fn set_update_emitter(&self, emitter: ara2_bridge::plugin::UpdateEmitter) {
        *self.update_emitter.lock_or_recover() = Some(emitter);
    }

    /// 登记一个 modification 的宿主句柄（`createAudioModification` / clone 时）。
    pub(crate) fn register_modification_handle(
        &self,
        persistent_id: &str,
        handle: ara2_bridge::core::RawHandle,
    ) {
        if persistent_id.is_empty() {
            return;
        }
        self.modification_ids
            .lock_or_recover()
            .insert(handle.identity(), persistent_id.to_owned());
        self.modification_handles
            .lock_or_recover()
            .insert(persistent_id.to_owned(), handle);
    }

    /// 宿主销毁 / 换掉一个 modification 的 persistentID 时同步。
    pub(crate) fn forget_modification_handle(&self, persistent_id: &str) {
        if let Some(handle) = self
            .modification_handles
            .lock_or_recover()
            .remove(persistent_id)
        {
            self.modification_ids
                .lock_or_recover()
                .remove(&handle.identity());
        }
    }

    /// 宿主句柄 → persistentID（部分存档的写出侧筛选）。
    pub(crate) fn modification_id_for_handle(
        &self,
        index: std::num::NonZeroU32,
        session: std::num::NonZeroU64,
    ) -> Option<String> {
        self.modification_ids
            .lock_or_recover()
            .get(&(index, session))
            .cloned()
    }

    /// **参数变了 ⇒ 告诉宿主这些 modification 的持久状态变了。**
    ///
    /// 【为什么宁可多报】通知是幂等且会被合并的（桥接层的 `UpdateTracker`），而漏报
    /// 会让"只存增量"的宿主丢数据。因此这里按当前登记的全部句柄投递，不做精确差分。
    ///
    /// 【为什么信号位为空】音高 / 共振峰 / 气声一变，信号本身就变了，因此
    /// `ContentUpdateScopes` 不得置 `SIGNAL_REMAINS_UNCHANGED`。
    pub(crate) fn notify_parameters_changed(&self) {
        let emitter = self.update_emitter.lock_or_recover().clone();
        let Some(emitter) = emitter else {
            return;
        };
        let handles = self
            .modification_handles
            .lock_or_recover()
            .values()
            .copied()
            .collect::<Vec<_>>();
        let mut refused = 0_usize;
        for handle in handles {
            if emitter
                .mark_modification(
                    handle,
                    None,
                    ara2_bridge::core::ContentUpdateScopes::empty(),
                    ara2_bridge::plugin::UpdateOrigin::Application,
                )
                .is_err()
            {
                refused += 1;
            }
        }
        if refused > 0 {
            // 队列是进程内的，失败只可能是"这个句柄不属于当前会话"（宿主换了图）。
            // 限流留痕：静默吞掉会让"宿主没保存"无从复盘。
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.update.refused",
                format_args!(
                    "[ara] {refused} modification update notification(s) were refused by the                      queue; the host will not be told those objects changed"
                ),
            );
        }
        // 文档级私有状态（分组 / 空白编辑 / 参数权威本身）也在同一次编辑里变了。
        emitter.mark_document(ara2_bridge::plugin::UpdateOrigin::Application);
    }

    /// 工厂成功返回前登记真实 controllerRef，失败或销毁时撤销。
    ///
    /// 【为什么在这里清扫】region 身份表是**进程级**的，而上一次文档可能没有走完
    /// `destroy_document`（宿主不按 leaf-first 销毁）就留下了无主 key。新文档注册时
    /// 主动扫掉"归属文档已死"的登记，让"重开工程"不依赖后续创建顺序就能摆脱上一次
    /// 的残留 —— 否则 `create_playback_region` 会在同一批地址上撞车，
    /// 宿主从此不再交付任何 region，界面永远停在"等待宿主音频"。
    ///
    /// 【为什么逐个问存活】同一进程里可以同时有多份文档（多个工程窗口）。只清死
    /// 文档的登记，绝不能全表清空。
    pub fn register(self: &Arc<Self>, key: usize) {
        self.controller.store(key, Ordering::Release);
        let mut table = controllers().lock_or_recover();
        let dead: Vec<usize> = table
            .iter()
            .filter(|(_, weak)| {
                weak.upgrade()
                    .is_none_or(|session| !session.alive.load(Ordering::Acquire))
            })
            .map(|(controller, _)| *controller)
            .collect();
        for controller in &dead {
            table.remove(controller);
        }
        if !dead.is_empty() {
            log::info!(
                "[ara] swept {} controller entrie(s) of dead document(s) before registering \
                 document {}",
                dead.len(),
                self.id
            );
        }
        // 同一份存活判定扫 region 表：controller 表活着但文档已 close 的也算死
        // （`close()` 已经把它从表里摘掉，因此这里看到的一定是活的）。
        let alive_ids: Vec<u64> = table
            .values()
            .filter_map(|weak| weak.upgrade())
            .filter(|session| session.alive.load(Ordering::Acquire))
            .map(|session| session.id)
            .collect();
        drop(table);
        crate::render::ownership::region_owners()
            .lock_or_recover()
            .retain_alive_documents(&|document| alive_ids.contains(&document));
        controllers()
            .lock_or_recover()
            .insert(key, Arc::downgrade(self));
    }

    /// 同步关闭文档；保持 renderer 的原生接口存储，但撤销模型操作许可。
    pub fn close(&self) {
        {
            let _transaction = self.transaction.lock_or_recover();
            if !self.alive.swap(false, Ordering::AcqRel) {
                return;
            }
            self.scope_revision.fetch_add(1, Ordering::AcqRel);
        }
        // 不持transaction join：actor可能正在收尾短事务；先停止全部编辑/分析再清宿主图。
        self.host_undo.close();
        if let Some(Ok(editor)) = self.editor.get() {
            editor.close();
        }
        self.render_epoch.fetch_add(1, Ordering::AcqRel);
        self.playback.lock_or_recover().take();
        self.ready.store(false, Ordering::Release);
        // 【T1】文档关闭也是"渲染许可失效"的一种。对这份文档本身已无意义，
        // 但让三个置假点语义一致是显式不变量，避免后来者只改其中两个。
        self.note_content_change();
        // 文档关闭时不存在"进行中的宿主会话"；留着会让下一次 ensure_loaded 误判为暂缓。
        self.editing_session.store(false, Ordering::Release);
        let leases = {
            let mut renderers = self.renderers.lock_or_recover();
            self.alive.store(false, Ordering::Release);
            std::mem::take(&mut *renderers)
        };
        let mut controllers = controllers().lock_or_recover();
        let key = self.controller.load(Ordering::Acquire);
        if controllers
            .get(&key)
            .and_then(Weak::upgrade)
            .is_some_and(|current| current.id == self.id)
        {
            controllers.remove(&key);
        }
        drop(controllers);
        for renderer in leases {
            renderer.lease.destroy();
            if let Some(owner) = renderer.owner.upgrade() {
                owner.document_closed();
            }
            // companion guard 随 renderer 释放，禁止继续借用即将销毁的 controller。
        }
        self.sequence_regions.lock_or_recover().clear();
        self.sequence_track_index.lock_or_recover().clear();
        self.regions.lock_or_recover().clear();
        // 【为什么先取本文档的 clip id 再清表】共振峰代次表在内核里是**进程级**的，
        // 同一进程可以同时开着多份工程（多个 ARA 文档）。整表清会把另一份仍在编辑的
        // 工程的活跃代次一并抹掉 —— 那边的后台重建随后会以为代次过期而静默放弃，
        // 表现为"另一个工程里的波形/共振峰不再更新"。只清本文档持有的那些。
        let owned_clip_ids = self
            .clip_ids
            .lock_or_recover()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        self.clip_ids.lock_or_recover().clear();
        self.region_items.lock_or_recover().clear();
        // 授权/事实映射随文档一起释放：此前 `region_items` / `authorized_takes` /
        // `authorized_sources` / `split_media_from` 只在文档销毁时清，而 region 销毁
        // 回调并不回收它们 —— 长会话里每次分割/移动/粘贴都会留下永不释放的条目。
        self.authorized_takes.lock_or_recover().clear();
        self.authorized_sources.lock_or_recover().clear();
        self.split_media_from.lock_or_recover().clear();
        self.source_failures.lock_or_recover().clear();
        self.source_states.lock_or_recover().clear();
        self.region_rejections.lock_or_recover().clear();
        self.pending_since.lock_or_recover().clear();
        self.skipped_regions.lock_or_recover().clear();
        self.ui_tracks.lock_or_recover().clear();
        self.ui_known_tracks.lock_or_recover().clear();
        *self.ui_inventory_stamp.lock_or_recover() = None;
        *self.ui_inventory_sampled_at.lock_or_recover() = None;
        *self.render_facts.lock_or_recover() = Default::default();
        self.timeline.lock_or_recover().take();
        self.track_bindings.lock_or_recover().clear();
        *self.edits.lock_or_recover() = Default::default();
        self.sources.lock_or_recover().clear();
        self.edit_sources.lock_or_recover().clear();
        *self.host_tempo_map.lock_or_recover() = None;
        *self.pending_archive_restore.lock_or_recover() = None;
        // 【为什么文档关闭时要清共振峰代次表】那张表在**内核里是进程级**的
        // （`FORMANT_REBUILD_GENERATIONS`），按 clip_id 索引且只增不减：条目仅在同一个
        // clip 被再次重建时覆写，而工程切换后旧 clip_id 不会再出现 —— 它的清理函数
        // 此前**没有任何调用者**。单条很小，但反复导入工程会一直涨。
        // 只清本文档的 id，别动同进程里的另一份工程。
        hifishifter_kernel::formant_cache::forget_formant_rebuild_generations(
            owned_clip_ids.iter().map(String::as_str),
        );
    }

    /// 控制器侧独立保留 lease，companion 先释放时 raw extension 存储仍须存活。
    pub fn attach(
        &self,
        owner: &Arc<ExtensionOwner>,
        lease: ExtensionControllerLease,
        companion: Option<CompanionControllerBinding<'static>>,
    ) -> Result<(), AraError> {
        let mut renderers = self.renderers.lock_or_recover();
        if !self.alive.load(Ordering::Acquire) {
            return Err(AraError::InvalidState("document is destroyed"));
        }
        renderers.push(RendererLease {
            lease,
            _companion: companion,
            owner: Arc::downgrade(owner),
        });
        self.ever_had_renderer.store(true, Ordering::Release);
        Ok(())
    }

    /// 本文档是否曾经有过 renderer（用于区分"还没绑定"与"绑定过又丢了"）。
    pub(crate) fn ever_had_renderer(&self) -> bool {
        self.ever_had_renderer.load(Ordering::Acquire)
    }

    /// 把某个 owner 的 lease 从本文档摘掉（文档本身仍存活）。
    ///
    /// 【为什么需要它】宿主复用同一个处理器、把实例从旧文档迁到新文档时，旧文档
    /// 可能仍然活着。此时必须只摘掉**这一个** owner 的租约，不能关闭整份文档，
    /// 也不能打断同文档里其他实例（多工程窗口/多实例）。
    pub(crate) fn detach_owner(&self, owner: &Arc<ExtensionOwner>) {
        let mut renderers = self.renderers.lock_or_recover();
        renderers.retain(|lease| {
            !lease
                .owner
                .upgrade()
                .is_some_and(|current| Arc::ptr_eq(&current, owner))
        });
    }

    /// 模型线程刷新所有仍存活 renderer，任何 source/geometry/访问变化都触发重新准备。
    pub fn prepare_renderers(&self) {
        let _transaction = self.transaction.lock_or_recover();
        // 【T2：`ready` 的语义收窄为"模型已收口"，不再假装"输入可用"】
        //
        // 它此前同时承担两件事："我们跑过一次 settle"与"宿主交付了可用输入"。
        // 后者根本无从判断（宿主可以授权 0 个源而一切"成功"），于是重开工程后
        // 界面一片"等待宿主音频"、而日志里没有任何异常。
        //
        // 【为什么**不**在这里按 `input_verdict` 置假】那样会把"宿主还没交音频"
        // 变成"渲染许可失效"，让 `capture_render_input` 在每一次正常的加载窗口里
        // 都失败一次 —— 而失败的下游是"快照停在上一版"，比静音更难排查。
        // 判据的真正消费者是**呈现**：`ensure_loaded` 据此区分"等音频"与"真冲突"，
        // `decorate_host_audio` 据此给出可行动的分类（见 `input_verdict` 的读者）。
        self.ready.store(true, Ordering::Release);
        let owners = self
            .renderers
            .lock_or_recover()
            .iter()
            .filter_map(|lease| lease.owner.upgrade())
            .filter(|owner| !owner.is_closed())
            .collect::<Vec<_>>();
        // 【ARA 存档的落点】存档恢复在模型图重建之前到达，那时还没有 owner 可合并，
        // 所以载荷暂存在文档级。现在图已就绪，把它交给**第一个** owner 的既有合并
        // 路径 —— 编辑权威是文档共享的，交给谁合并都一样（合并写的是 `document.edits`）。
        // 部分存档先合入（它是按键合并，不会覆盖整份状态），再走整份存档的合并路径。
        self.merge_partial_restores();
        // 【为什么这里要把 `take()` 与"放回去"分成两次独立的取锁】此前写成
        // `if let Some(restored) = self.pending_archive_restore.lock_or_recover().take()`
        // —— 那个 guard 活到整个 `if let` 块结束，而 `None` 分支里又要
        // `*self.pending_archive_restore.lock_or_recover() = Some(restored)`。
        // `std::sync::Mutex` 不可重入，于是"没有任何 renderer 绑定时"当场自锁：
        // `prepare_renderers` 永久挂住，本文档再不响应任何宿主编缉。
        // 用户报的症状正是"无论宿主里怎么编辑，ARA 中都不能同步，只有把插件删掉再
        // 添加才好" —— 撤销/重载会送来存档还原，而还原到达时经常还没有 renderer。
        let held_back = self.pending_archive_restore.lock_or_recover().take();
        if let Some(restored) = held_back {
            match owners.first() {
                // 【为什么只暂存、不在这里刷新】本函数已经持有 `transaction`，而刷新要取
                // 同一把不可重入的锁 —— 那会当场自锁（用户报的"重开 ARA 工程宿主未
                // 响应"）。合入由紧随其后的 `merge_pending_restore` 循环完成。
                Some(owner) => owner.stash_archive_restore(restored),
                None => {
                    // 没有任何 owner：把载荷放回去，等下一次收口再试。
                    log::warn!(
                        "[ara] archive restore held back: no renderer is bound yet; it will be \
                         applied on the next prepare"
                    );
                    *self.pending_archive_restore.lock_or_recover() = Some(restored);
                }
            }
        }
        // 先统一恢复全部组件权威，再允许任何后台任务捕获共享revision。
        for owner in &owners {
            if let Err(error) = owner.merge_pending_restore(self) {
                log::warn!("[ara] instance state unresolved: {error}");
            }
        }
        drop(_transaction);
        for owner in owners {
            owner.refresh_host_state_for_model();
            owner.prepare();
        }
    }

    /// 是否还有一份**没合入**的持久化载荷（文档级 ARA 存档或任一实例的组件状态）。
    ///
    /// 【为什么存档路径要问它】`store_document` 唯一正确的回答是"把宿主交给我们的
    /// 那一份还回去"。只要还有载荷挂着，内存里的 `edits` 就**不是**用户的数据；
    /// 用它在覆盖 `archive_cache`，等于在磁盘上把用户参数换成默认值，而宿主此后
    /// 每一次保存都会把这个结果固化下去。
    pub(crate) fn has_unmerged_restore(&self) -> bool {
        if self.pending_archive_restore.lock_or_recover().is_some() {
            return true;
        }
        self.renderer_owners()
            .iter()
            .any(|owner| owner.has_pending_restore())
    }
    /// 同一ARA文档的编辑权威共享，输出仍按各renderer分配隔离。
    pub fn renderer_owners(&self) -> Vec<Arc<ExtensionOwner>> {
        self.renderers
            .lock_or_recover()
            .iter()
            .filter_map(|lease| lease.owner.upgrade())
            .filter(|owner| !owner.is_closed())
            .collect()
    }

    /// 把"本档有没有实例持 playback 角色"这一文档级事实推给每个实例，并报告读数。
    ///
    /// 【为什么必须推送而不是让每个实例自己查】`is_editor_only()` 在**音频线程**上求值
    /// （`vst3.rs::audio_process`），只能读本实例的一个原子量；而"文档上还有没有别的
    /// playback 实例"是文档级事实。这里在每次绑定/分配变化时算一次并推过去 ——
    /// 变化频率是"用户增删插件实例"，不是音频帧率。
    ///
    /// 【为什么这不再破坏透传契约】此前只能比"本实例是不是 editor"，而 REAPER 的正常
    /// 编辑实例与"Cubase 没分配 playback"在**实例级**确实同形。改比**文档级**事实之后就
    /// 区分开了：REAPER 的编辑实例另有 playback 兄弟 ⇒ 仍透传；Cubase 的单实例
    /// 没有任何 playback 兄弟 ⇒ 接管渲染（实测日志：`playback ownership: 1 live
    /// renderer instance(s) but NONE holds the playback role`，用户报的"编辑不改变声音"）。
    ///
    /// 调用点：绑定完成时、分配变化时（角色集合唯一可能改变的两个时刻）。
    pub(crate) fn report_playback_ownership(&self) {
        let owners = self.renderer_owners();
        if owners.is_empty() {
            return;
        }
        let covered = owners.iter().any(|owner| owner.has_playback_role());
        if covered {
            for owner in &owners {
                owner.set_playback_covered(true);
            }
            return;
        }
        // 没有任何实例带 playback 角色：把**已经准备好快照**的 editor 实例提为渲染者。
        //
        // 【为什么必须提升】`set_playback_covered(false)` 会让编辑实例的
        // `is_editor_only()` 翻真，于是它从"透传宿主原声"改走快照通路；而它很可能
        // 从未 `prepare()` 过（`prepare()` 在 `!renders_playback()` 时直接返回），
        // 快照是空的 —— 此时每一个音频块都是纯零。反过来，若播放实例先关闭而编辑
        // 实例没被提升，同样没人出声。两侧都指向同一个动作：挑一个能出声的实例。
        //
        // 【为什么要求"已经准备好"】没准备过的实例没有可发布的快照，提升它只会把
        // 静音从"一个实例"扩大成"另一个实例"。挑不到的场合保持 `covered = false`，
        // 让所有实例都透传宿主原声 —— 至少不是静音。
        let promoted = owners
            .iter()
            .find(|owner| owner.has_prepared_snapshot())
            .or_else(|| owners.first());
        for owner in &owners {
            owner.set_playback_covered(false);
        }
        if let Some(promoted) = promoted {
            promoted.set_playback_covered(false);
            promoted.promote_to_renderer();
        }
        // 限流：绑定与分配都可能连续触发，这条读数不需要每次都打。
        //
        // 【为什么把角色位一起打出来】`bind document` 那一行是 `INFO`，而现场日志可能被
        // 滤成 `WARN`（实测 Cubase 会话整段零行 INFO）。角色分配恰恰是"编辑听不出来"
        // 的头号线索，所以这一条必须自带全部信息：每个实例的 writer_id 与角色位。
        // 1 = playback、2 = editor、4 = view、3 = playback|editor。
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            "ara.playback.no-renderer",
            format_args!(
                "[ara] playback ownership: {} live renderer instance(s) but NONE holds the \
                 playback role (writers/roles: {:?}); an instance takes over rendering so that \
                 edits are audible. If this host meant to provide a playback renderer, its role \
                 assignment is the thing to check",
                owners.len(),
                owners
                    .iter()
                    .map(|owner| (owner.writer_id(), owner.roles_bits()))
                    .collect::<Vec<_>>()
            ),
        );
    }

    /// 播放渲染器的诊断快照（诊断包用）。
    ///
    /// 【为什么只要"准备状态"】逐输出的区间读数需要当前分配区间，而诊断包在任意
    /// 时刻被导出 —— 不值得为它触发一次完整快照。这里给的是"忙不忙、有没有错、
    /// 准备的是哪一版"，正好覆盖"渲染结果陈旧/一直没准备好"这两类报障。
    pub fn renderer_diagnostics(&self) -> serde_json::Value {
        let renderers = self
            .renderer_owners()
            .into_iter()
            .filter(|owner| owner.renders_playback())
            .map(|owner| {
                let (busy, error) = owner.local_preparation_state();
                serde_json::json!({
                    "busy": busy,
                    "error": error,
                    "prepared": owner.prepared_snapshot(),
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({ "renderers": renderers })
    }

    /// 记一次"宿主侧发生了变化，渲染许可随之失效"。
    ///
    /// 【为什么必须共用一个 helper】`ready` 有三个置假点（`close` / `clear_renderers` /
    /// `revoke_snapshots`），而 `ensure_loaded` 靠这个时刻把"宿主正在收敛"与
    /// "宿主永不收敛"分开（见字段说明）。漏写任何一个，就把那一类变化误报成
    /// 用户级冲突 —— 正是重开工程后仍然出现的那条 `Conflict: host changed`：
    /// 宿主授权 PCM 走的是 `refresh_source_pcm` → `revoke_renderers()` →
    /// `revoke_snapshots()`，而它此前**没有**落这个时刻。
    fn note_content_change(&self) {
        *self.last_content_change.lock_or_recover() = Some(std::time::Instant::now());
    }

    /// 进程内单调毫秒。0 保留给"没有事件"。
    ///
    /// 【为什么不用 `Instant` + `Mutex`】这几个时刻要在**音频线程**上读
    /// （`request_settle_if_overdue` 由 `process` 顺带调用），而 `Mutex` 在实时线程上
    /// 是不可接受的。整数毫秒既无锁又足够（宽限期是秒级）。
    fn monotonic_ms() -> u64 {
        static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let epoch = EPOCH.get_or_init(std::time::Instant::now);
        // +1 保证"刚发生"也不等于 0（0 是哨兵）。
        (epoch.elapsed().as_millis() as u64).saturating_add(1)
    }

    /// 记下编辑会话的开始（`begin_editing`）。
    pub(crate) fn note_editing_started(&self) {
        self.editing_session.store(true, Ordering::Release);
        self.editing_since_ms
            .store(Self::monotonic_ms(), Ordering::Release);
    }

    /// 记下编辑会话的结束（`end_editing`）。
    pub(crate) fn note_editing_finished(&self) {
        self.editing_session.store(false, Ordering::Release);
        self.editing_since_ms.store(0, Ordering::Release);
    }

    /// 记下"模型变脏了"（首次才落时刻：后续脏不推迟截止期）。
    pub(crate) fn note_model_dirty(&self) {
        let now = Self::monotonic_ms();
        let _ = self.model_dirty_since_ms.compare_exchange(
            0,
            now,
            Ordering::AcqRel,
            Ordering::Relaxed,
        );
    }

    /// 记下"模型已成功发布"。
    pub(crate) fn note_model_clean(&self) {
        self.model_dirty_since_ms.store(0, Ordering::Release);
    }

    /// 距模型变脏已经过去多久（`None` = 干净）。
    pub(crate) fn model_dirty_elapsed(&self) -> Option<std::time::Duration> {
        let since = self.model_dirty_since_ms.load(Ordering::Acquire);
        (since != 0).then(|| Self::elapsed_since(since))
    }

    /// 距编辑会话开始已经过去多久（`None` = 不在会话中）。
    pub(crate) fn editing_elapsed(&self) -> Option<std::time::Duration> {
        let since = self.editing_since_ms.load(Ordering::Acquire);
        (since != 0).then(|| Self::elapsed_since(since))
    }

    /// **T23：给"宿主不再回调"的三条死路一个插件侧出口。**
    ///
    /// 【为什么必须有它】`ready` 回到真只依赖四个入口：`endEditing`、
    /// `notifyModelUpdates`（→ flush）、`refresh_source_pcm`（要求 `!editing`）、
    /// ARA 存档恢复。**四个全在宿主手里。** 宿主一旦 `beginEditing` 之后再也不收口，
    /// 或者积压了模型变化却不再调 `notifyModelUpdates`，插件就永久停在未就绪 ——
    /// 用户唯一的出路是删掉插件再加。这不是"某个 bug"，是**没有出口**。
    ///
    /// 【为什么是"置位请求"而不是直接收口】桥的 `PluginRuntime` 由文档控制器独占、
    /// 不是 `Send`/`Sync`，从视图定时器直接调它是未定义行为。这里只置两个原子量，
    /// 由控制器在**下一次任何回调**的入口消费（`runtime_mut` 与 `notifyModelUpdates`）。
    ///
    /// 【为什么可以重复调用】纯原子读 + 幂等置位，音频线程也可以调。
    ///
    /// 【边界】宿主连一次回调都不再发时，这里也只能等到下一次回调。那一段的兜底是
    /// VST3 侧的 `getState` / `setActive`（宿主存盘与激活）—— 本函数不假装能覆盖它。
    /// 单调毫秒差的哨兵：`u64::MAX` 表示"起点被回拨到无穷远"（仅测试使用）。
    const BACKDATED: u64 = u64::MAX;

    /// 由起点算已过去多久；`BACKDATED` 视为远超任何宽限期。
    fn elapsed_since(since: u64) -> std::time::Duration {
        if since == Self::BACKDATED {
            // 比任何宽限期都大，但仍是有限值 —— 调用方可能对它做加法。
            return std::time::Duration::from_secs(86_400);
        }
        std::time::Duration::from_millis(Self::monotonic_ms().saturating_sub(since))
    }

    /// 发布一份由模型侧算出的输入可用性读数（见 [`InputReadiness`]）。
    pub(crate) fn note_readiness(&self, readiness: InputReadiness) {
        *self.readiness.lock_or_recover() = Some(readiness);
    }

    /// 当前读数（`None` = 还没算过）。
    pub(crate) fn readiness(&self) -> Option<InputReadiness> {
        *self.readiness.lock_or_recover()
    }

    /// 当前判据；`None` 表示"未知"，调用方按旧行为处理。
    pub(crate) fn input_verdict(&self) -> Option<InputVerdict> {
        self.readiness().map(|readiness| readiness.verdict)
    }

    /// **T6：把"卡住了"变成一份自我描述的读数。**
    ///
    /// 【为什么必须存在】这四类症状反复复发的真正原因不是某个 bug，而是**失败不可
    /// 归因**：十几条路径都能静默早退，而它们留下的现场几乎同形。这一行把每一条早退
    /// 都变成一个具名的数，下一轮不必再从位掩码与调用顺序反推。
    ///
    /// 只在 `verdict != Ready` 时打印，并由调用方限流。
    pub(crate) fn stuck_report(&self) -> String {
        let readiness = self.readiness();
        let scope_regions = self.sequence_regions.lock_or_recover().len();
        let assignments = self.renderer_owners().len();
        let ui_tracks = self.ui_tracks.lock_or_recover().len();
        let region_items = self.region_items.lock_or_recover().len();
        let editing = self.editing_elapsed().map(|elapsed| elapsed.as_millis());
        // 【T11：图账本进这一行】没有它，"宿主没发图"与"我们拒了图"在这里完全同形 ——
        // 而那正是四类症状反复复发的共同形状：两条路径共用一句读数。
        let graph = self
            .graph_ledger
            .lock_or_recover()
            .as_ref()
            .map(|ledger| ledger.summary());
        format!(
            "doc={} {} ready={} editing={:?} dirty={:?} sequences={} renderers={} \
             ui_tracks={} region_items={} {}",
            self.id,
            readiness.map_or_else(|| "verdict=unknown".to_owned(), |r| r.summary()),
            self.ready.load(Ordering::Acquire),
            editing,
            self.model_dirty_elapsed().map(|elapsed| elapsed.as_millis()),
            scope_regions,
            assignments,
            ui_tracks,
            region_items,
            graph.map_or_else(|| "graph=unbound".to_owned(), |line| line),
        )
    }

    /// 测试专用：把两个截止期的起点回拨到"早已超期"。
    ///
    /// 【为什么需要】真实场景里"宿主卡了 30 秒"无法在单测中等待；把起点回拨等价，
    /// 而且不会引入 `sleep`（那会让整个套件变慢且不稳定）。
    #[cfg(test)]
    pub(crate) fn backdate_settle(&self) {
        for slot in [&self.editing_since_ms, &self.model_dirty_since_ms] {
            if slot.load(Ordering::Acquire) != 0 {
                slot.store(Self::BACKDATED, Ordering::Release);
            }
        }
    }

    pub fn request_settle_if_overdue(&self) {
        if self
            .editing_elapsed()
            .is_some_and(|elapsed| elapsed >= FORCED_EDIT_CLOSE_GRACE)
        {
            self.forced_edit_close.store(true, Ordering::Release);
        }
        if self
            .model_dirty_elapsed()
            .is_some_and(|elapsed| elapsed >= EDIT_SETTLE_GRACE)
        {
            self.forced_model_flush.store(true, Ordering::Release);
        }
    }

    /// 在旧内容可能被改变前立即撤销**渲染许可与代次**；不撤已发布快照本身。
    ///
    /// 【为什么不再清空快照】`clear_renderers()` 被一次宿主编辑触发**多次**
    /// （`ara/model.rs` 里 8 处调用）。清空 `current` 之后，`copy_block_at` 的读闭包
    /// 逐样本返回 `0.0`，而 bound 实例固定 `layered == false`，于是**每个音频块都是
    /// 纯零** —— 用户听到的就是"有概率完全没声音"。清空与后台合成还要再叠一层
    /// `prepare_job` 的版本重检丢弃（`extension.rs`），空窗最长可达一次神经合成的时长。
    ///
    /// 保留上一份快照永远好过静音：它对应上一次稳定状态，`publish` 会在新快照算完后
    /// **原子**替换 `current`，音频线程的 `enter()` 要么拿到旧的、要么拿到新的，
    /// 不会拿到空。真正该清空的是"这个实例从此不再渲染"——那由 [`Self::close`]
    /// 与 `ExtensionOwner::stop_editor` 负责，它们会调 [`Self::revoke_snapshots`]。
    ///
    /// 【为什么仍然按回落 `ready`】`ready` 回答的是"现在能不能重排并发布"，宿主编辑
    /// 期间答案确实是否；把它留在 true 会让"还没重排完"被误判成"已经就绪"。
    /// 它与"已发布的快照还在不在"是两个问题，本轮只改后者。
    pub fn clear_renderers(&self) {
        let _transaction = self.transaction.lock_or_recover();
        self.revision.fetch_add(1, Ordering::AcqRel);
        self.render_epoch.fetch_add(1, Ordering::AcqRel);
        self.ready.store(false, Ordering::Release);
        // 记下"宿主刚刚改过内容"：`ensure_loaded` 据此把紧随其后的未就绪窗口
        // 当成"等宿主收口"而不是"宿主冲突"（见字段说明）。
        self.note_content_change();
        for owner in self.renderer_owners() {
            owner.cancel_preparation();
            // 让 revision 门（`processed_ready` / `raw_tape_ready`）与输出代次失效，
            // 但不动 `current` —— 上一份快照继续可听。
            owner
                .snapshots
                .iter()
                .for_each(|snapshot| snapshot.invalidate_generation());
            owner.raw_tape.invalidate_generation();
        }
    }

    /// 授权开关只撤销播放许可，不改变宿主内容/布局的乐观并发版本。
    pub fn revoke_renderers(&self) {
        let _transaction = self.transaction.lock_or_recover();
        self.revoke_snapshots();
    }

    /// 渲染输入失效：推进 `render_epoch` 并**通知本文档全部 owner** 重排。
    ///
    /// 【为什么必须是文档级、必须遍历全部 owner】`render_epoch` 是本文档的共享计数器，
    /// 但"谁发现了变化"与"谁持有快照"在多实例下**不是同一个实例**。REAPER 按轨道管理
    /// ARA 实例：持有 webview 与 20 ms 定时器的是编辑/视图实例（`role == 2`），真正被
    /// 音频回调读取快照的是播放渲染实例（`role == 1`）。若发现变化的一侧只
    /// `self.prepare()`，编辑实例会因 `prepare()` 首行的 `!renders_playback()` 直接返回
    /// （`extension.rs`），播放实例则**从未被通知** —— 于是波形更新、声音不变，直到
    /// 用户做一次编辑（编辑路径恰好遍历了全部 owner）才"碰巧"恢复。
    ///
    /// 【为什么不在这里清快照】失效只需要让 `PreparedVersion` 不再匹配（它含 `epoch`），
    /// `prepare_job` 便不会早退；清快照会制造一段静音空窗，而重排是异步的。
    ///
    /// 【为什么**不**取 `transaction`】这是一次通知，不是与文档状态必须原子提交的变更：
    /// 只推进一个原子计数并枚举 owner。`prepare_job` 会在自己的事务里重新校验全部版本，
    /// 因此这里取事务没有正确性收益，却会在这个 20 ms 定时器路径上多抢一次文档锁，
    /// 并让调用方无法"持事务阻止后台渲染"（测试与离线 setup 都依赖这个手法）。
    /// 锁序也因此更简单：本函数只取 `renderers`，不参与"事务 → renderers"的嵌套。
    pub fn invalidate_render_input(&self) {
        self.render_epoch.fetch_add(1, Ordering::AcqRel);
        let owners = self
            .renderers
            .lock_or_recover()
            .iter()
            .filter_map(|lease| lease.owner.upgrade())
            .filter(|owner| !owner.is_closed())
            .collect::<Vec<_>>();
        for owner in owners {
            owner.prepare();
        }
    }

    fn revoke_snapshots(&self) {
        self.render_epoch.fetch_add(1, Ordering::AcqRel);
        self.ready.store(false, Ordering::Release);
        // 【T1】这里同样是一次"宿主侧变化"（源权限撤销 / 源内容变了），必须与
        // `clear_renderers()` 一样留下观察时刻。漏掉它会让 `ensure_loaded` 把
        // `refresh_source_pcm` 造成的未就绪窗口判成真实冲突。
        self.note_content_change();
        let owners = self
            .renderers
            .lock_or_recover()
            .iter()
            .filter_map(|lease| lease.owner.upgrade())
            .collect::<Vec<_>>();
        for owner in owners {
            owner.cancel_preparation();
            // 【为什么这里是**硬清**而不是只推进代次】`revoke_renderers()` 的语义是
            // "宿主撤销了源权限 / 源内容变了"：继续播旧快照等于播一份已经失效的
            // 渲染结果。真正被频繁调用、且**不该**造成静音的是 `clear_renderers()`
            // （一次宿主编辑触发多次）—— 那一条已经改成只 `invalidate_generation()`。
            owner.snapshots.iter().for_each(|snapshot| snapshot.clear());
        }
    }

    /// 记录一次"本插件拒绝了 region 创建"。
    ///
    /// 【为什么必须留下真因】`create_playback_region` 失败对宿主只表现为一个 null
    /// region，插件侧唯一的痕迹是一条 error 日志。把它接进状态链之后，
    /// `decorate_host_audio` 能把 `awaiting_regions` 换成 `region_rejected`，
    /// 用户看到的是"插件拒收了"而不是"还在等"。
    pub fn note_region_rejection(&self, reason: impl Into<String>) {
        const REGION_REJECTION_LIMIT: usize = 32;
        let mut rejections = self.region_rejections.lock_or_recover();
        if rejections.len() == REGION_REJECTION_LIMIT {
            rejections.pop_front();
        }
        rejections.push_back(reason.into());
    }

    /// 最近一条 region 拒绝原因（没有则为 `None`）。
    pub fn last_region_rejection(&self) -> Option<String> {
        self.region_rejections.lock_or_recover().back().cloned()
    }

    /// 只借出仍存活的文档，拒绝已销毁或非本插件的控制器。
    pub fn lookup(key: usize) -> Option<Arc<Self>> {
        controllers()
            .lock_or_recover()
            .get(&key)
            .and_then(Weak::upgrade)
            .filter(|session| session.alive.load(Ordering::Acquire))
    }

    /// 当前进程里仍存活的控制器键（诊断；有界）。
    ///
    /// 【为什么需要它】`bind` 拿到一个不在表里的 controller ref 时，必须能一眼看出
    /// "是表里根本没有新文档"还是"新文档用了另一个 ref"。只返回前若干条，避免
    /// 多工程/异常宿主把日志撑爆。
    pub(crate) fn live_controller_keys() -> Vec<usize> {
        const MAX_KEYS: usize = 16;
        controllers()
            .lock_or_recover()
            .iter()
            .filter(|(_, weak)| {
                weak.upgrade()
                    .is_some_and(|session| session.alive.load(Ordering::Acquire))
            })
            .map(|(key, _)| *key)
            .take(MAX_KEYS)
            .collect()
    }
}
