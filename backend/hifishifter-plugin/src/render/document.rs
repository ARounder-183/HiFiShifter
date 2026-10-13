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
#[derive(Default)]
pub(crate) struct DocumentSession {
    controller: AtomicUsize,
    alive: AtomicBool,
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
    /// 【为什么必须记时刻】"等待 REAPER 完成音频分配"此前是**吸收态**：只要一个 item
    /// 没有 `source_path`、又不是 folder 父轨、方向位也读不出来，它就永远显示"正在
    /// 等待"，无论等多久、用户做什么都出不来（典型触发：REAPER 的"倒放 Item 为新
    /// Take"换了 active take，而 ARA 不再重发模型 ⇒ `authorized_takes` 永久陈旧）。
    /// 记下时刻就能把"在途"与"等不到了"分开：超过阈值仍无源 ⇒ 转 `unavailable` 并给出
    /// 可执行的原因，而不是让用户对着一个永远转不完的占位。
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
    /// 宿主是否处在一次 `beginEditing` / `endEditing` 会话中。
    ///
    /// 【为什么与 `ready` 分开】`ready` 的语义是"ARA 是否已重新交付可用输入"，
    /// 会被**任何**内容变化按回 `false`；而"宿主正在编辑"是另一件事。此前二者共用
    /// `ready`，于是会话内每一次真实变化都被 `ensure_loaded` 判成"宿主冲突"
    /// （用户报障：每次编辑都闪"尚未应用：Conflict: host changed"）。
    /// 有了这个标志，会话中的未就绪窗口被识别为**暂缓**而不是冲突 —— 它会随
    /// `endEditing` 自然收敛，不需要用户干预，也不该报错。
    pub editing_session: AtomicBool,
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
        if !self.ready.load(Ordering::Acquire) {
            return Err("host graph not ready; cannot archive ARA edits".into());
        }
        let mut edits = self.edits.lock_or_recover();
        edits.reconcile(&self.track_bindings.lock_or_recover())?;
        edits.encode()
    }

    /// 接收 ARA 存档载荷，等模型图就绪后合入。
    ///
    /// 【为什么只暂存】恢复到达时模型图通常还没重建（ARA 先给存档、宿主再重新交付
    /// 模型），此刻没有轨道可绑、也没有 renderer 可合并。暂存到文档级，
    /// `prepare_renderers` 收口时再交给 owner 的既有合并路径。
    pub(crate) fn restore_archive(&self, bytes: &[u8]) -> Result<(), String> {
        if bytes.is_empty() {
            return Ok(());
        }
        let mut restored = crate::state_channel::EditState::default();
        restored.restore(bytes)?;
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
                let region = key.and_then(|key| regions.get(&key).map(|region| (key, region)));
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
        let _transaction = self.transaction.lock_or_recover();
        if !self.is_alive() {
            return Err(crate::editor::errors::DOCUMENT_CLOSED.error());
        }
        self.editor
            .get_or_init(|| crate::editor::session::EditorSession::new(self))
            .clone()
    }
    pub(crate) fn flush_editor(&self) -> Result<(), String> {
        if let Some(Ok(editor)) = self.editor.get() {
            editor.flush()?;
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
            let mut failures = self.source_failures.lock_or_recover();
            for clip in &mut timeline.clips {
                for take in &mut clip.takes {
                    let Some(id) = take.source_path.as_deref() else {
                        continue;
                    };
                    if available_ids.contains(id) {
                        continue;
                    }
                    // 只在没有更具体的原因时记通用码：`refresh_source_pcm` 已经把
                    // "授权了但读不出来"的真因写在这里，别覆盖它。
                    failures
                        .entry(id.to_owned())
                        .or_insert_with(|| crate::editor::errors::HOST_PCM_UNAVAILABLE.to_owned());
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
            return Err("host graph not ready; cannot archive ARA edits".into());
        }
        let mut edits = self.edits.lock_or_recover();
        edits.reconcile(&self.track_bindings.lock_or_recover())?;
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
        partial.encode()
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
        restored.restore(bytes)?;
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
    pub fn register(self: &Arc<Self>, key: usize) {
        self.controller.store(key, Ordering::Release);
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
        self.clip_ids.lock_or_recover().clear();
        self.region_items.lock_or_recover().clear();
        // 授权/事实映射随文档一起释放：此前 `region_items` / `authorized_takes` /
        // `authorized_sources` / `split_media_from` 只在文档销毁时清，而 region 销毁
        // 回调并不回收它们 —— 长会话里每次分割/移动/粘贴都会留下永不释放的条目。
        self.authorized_takes.lock_or_recover().clear();
        self.authorized_sources.lock_or_recover().clear();
        self.split_media_from.lock_or_recover().clear();
        self.source_failures.lock_or_recover().clear();
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
        hifishifter_kernel::formant_cache::clear_formant_rebuild_generations();
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
        Ok(())
    }

    /// 模型线程刷新所有仍存活 renderer，任何 source/geometry/访问变化都触发重新准备。
    pub fn prepare_renderers(&self) {
        let _transaction = self.transaction.lock_or_recover();
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
        if let Some(restored) = self.pending_archive_restore.lock_or_recover().take() {
            match owners.first() {
                Some(owner) => owner.adopt_archive_restore(restored),
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
        for owner in &owners {
            owner.set_playback_covered(covered);
        }
        if covered {
            return;
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
                 playback role (writers/roles: {:?}); the sole editor instance takes over \
                 rendering so that edits are audible. If this host meant to provide a playback \
                 renderer, its role assignment is the thing to check",
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

    /// 在旧内容可能被改变前立即撤销发布；不回收实时读者可能仍持有的旧快照。
    pub fn clear_renderers(&self) {
        let _transaction = self.transaction.lock_or_recover();
        self.revision.fetch_add(1, Ordering::AcqRel);
        self.revoke_snapshots();
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
        let owners = self
            .renderers
            .lock_or_recover()
            .iter()
            .filter_map(|lease| lease.owner.upgrade())
            .collect::<Vec<_>>();
        for owner in owners {
            owner.cancel_preparation();
            owner.snapshots.iter().for_each(|snapshot| snapshot.clear());
        }
    }

    /// 只借出仍存活的文档，拒绝已销毁或非本插件的控制器。
    pub fn lookup(key: usize) -> Option<Arc<Self>> {
        controllers()
            .lock_or_recover()
            .get(&key)
            .and_then(Weak::upgrade)
            .filter(|session| session.alive.load(Ordering::Acquire))
    }
}
