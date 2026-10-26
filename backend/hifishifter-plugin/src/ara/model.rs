//! 把宿主推来的 ARA 模型回调**累积成一份 [`AraDocument`]**。
//!
//! 【为什么是"累积"而不是"边回调边映射"】ARA 的一次用户操作会拆成多次回调
//! （例如"复制 item" = `createPlaybackRegion` × N）。逐次映射会得到中间态，
//! 而 `end_editing` 是 ARA 约定的一次编辑收口点 —— 那里才是映射的正确时机。
//!
//! 【句柄 → 我们的 id】ARA 回调给的是**句柄**（`RawHandle`），而映射层要的是
//! `persistentID` 这样的稳定 id。所有对应关系都在这里维护，映射层因此可以
//! 保持纯数据、可离线逐样本测试。

use crate::ara::mapping::{
    AraAudioModification, AraAudioSource, AraDocument, AraMusicalContext, AraPlaybackRegion,
    AraRegionSequence,
};
use crate::render::document::DocumentSession;
use crate::render::ownership::{region_owners, DocumentId};
use crate::sync::MutexExt;
use ara2_bridge::core::{
    ApiGeneration, AraError, AudioModificationProperties, AudioSourceProperties, BarSignatures,
    ContentTimeRange, ContentUpdateScopes, DocumentProperties, KeySignatures,
    MusicalContextProperties, PlaybackRegionProperties, RegionSequenceProperties, Tempo,
};
use ara2_bridge::plugin::{
    AudioModifications, AudioSources, CreateContext, DocumentLifecycle, HostContentScope,
    MusicalContexts, PlaybackRegions, RegionSequences,
};
use hifishifter_kernel::state::TimelineState;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// 句柄的可用作键的形式。
///
/// `RawHandle` 自带 `Copy + Eq + Hash`（字段私有但不影响做键），所以直接用它。
type HandleKey = ara2_bridge::core::RawHandle;

/// 一个 audio source 的会话态。
#[derive(Clone, Default)]
pub struct SourceState {
    /// 宿主给它的稳定 id（REAPER 实测就是素材绝对路径）。
    pub persistent_id: String,
    /// 宿主是否已授予样本访问权。
    ///
    /// 【为什么单独记】探针实测：绑定后宿主调 `enable=true`，停用时 `enable=false`。
    /// 所以 `false` 是**撤销**语义，不是拒绝 —— 采样到 `false` 时不能判定"宿主不给样本"。
    pub sample_access_enabled: bool,
    /// 宿主报告的内容版本，每次 `update_audio_source_content` 自增。
    ///
    /// 【为什么必须记】渲染缓存不能复用旧内容（设计 §4.5）：这个版本号会被折进
    /// `Clip.source_file_fingerprint`，而那是渲染键的既有输入。
    pub content_version: u64,
}

/// 一个 audio modification 的会话态。
#[derive(Clone, Default)]
pub struct ModificationState {
    /// 宿主给它的稳定 id。
    pub persistent_id: String,
    /// 它基于哪个 source。
    pub source_persistent_id: String,
}

/// 一个 playback region 的会话态。
#[derive(Clone, Default)]
pub struct RegionState {
    /// 它属于哪个 region sequence（按累积顺序的下标）。
    pub sequence_index: Option<usize>,
}

/// 每个文档控制器一份的模型。
///
/// 【为什么不是进程共享】`document_controller(|| …)` 会**每个文档控制器**调用一次，
/// 每份模型对应一个文档。共享累积器会让两份文档互相污染 —— 而 v1 的单实例语义
/// 本来就是"一个实例 = 一条编辑轨"，一份文档一份模型才是对的。
pub struct ModelHandle {
    /// 会话内唯一文档身份；宿主的 model-ref 键只在所属文档生命周期内有效。
    document_id: DocumentId,
    session: Arc<DocumentSession>,
    region_keys: Vec<u64>,
    /// 每条 region sequence 的宿主 persistentID，下标与
    /// `document.musical_contexts[].region_sequences` 的扁平顺序一致。
    ///
    /// 【为什么要单独存】`AraRegionSequence.persistent_id` 已经带这个值，但序列可能被
    /// 宿主销毁（`destroyRegionSequence`）；销毁后该槽位要能标成"已消失"而不把后面的
    /// 序列身份顶上来。这份平行表用空串表示"该槽位的宿主不再承认"。
    sequence_persistent_ids: Vec<String>,
    /// 已销毁的 region sequence 槽位（与 `destroyed_regions` 同口径）。
    destroyed_sequences: HashSet<usize>,
    /// 【G3】已销毁序列的**待认领身份**：`(order_index, name, persistent key)`。
    ///
    /// 【为什么需要它】轨道 id 来自 `context.realtime_key()`（ARA 的 region sequence
    /// **没有** persistentID），而宿主重建序列时这个 key 必然改变。于是一次"切屏回来
    /// 重载 take"就会让整条轨的参数因为"查不到宿主身份"被丢掉。
    ///
    /// 认领规则：新序列若与一条待认领的死亡序列在 `(order_index, name)` 上**唯一**
    /// 匹配，就沿用它的 key —— 轨道 id 不变，参数自动续上。匹配不到（或并列）就用
    /// 新 key，与改动前完全一致。
    ///
    /// 【为什么不进存档】它是**本次会话**的重建窗口，跨会话没有任何意义（进程重启后
    /// 所有 key 都会重新分配）。
    pending_sequence_deaths: Vec<(i32, Option<String>, String)>,
    head_tail: Option<Arc<ara2_bridge::plugin::RealtimeHeadTailAdapter>>,
    /// 正在累积的 ARA 文档。
    document: AraDocument,
    /// 映射好的时间线（`end_editing` 之后可用）。
    timeline: Option<TimelineState>,
    /// 协商到的 ARA 版本。
    generation: Option<ApiGeneration>,
    /// 源授权可能在宿主事务中回调；完整图收口前不能准备/关联恢复编辑。
    editing: bool,
    /// 自上次成功发布时间线以来，模型是否发生过变化。
    ///
    /// 【为什么需要它】`remap_and_log()` 的唯一调用点是 `end_editing()`。宿主若在
    /// 工程加载/后台恢复时直接重建图却不发 `beginEditing/endEditing`，时间线永远
    /// 不会发布，region→item 认领与参数恢复都停在半路。`notifyModelUpdates` 是
    /// SDK 保证宿主在空闲时会周期性调用的入口；靠这个标记只在真有变化时重算。
    model_dirty: bool,
    /// region sequence 句柄 → 我们数组里的下标。
    sequence_index_by_handle: HashMap<HandleKey, usize>,
    /// source 句柄 → 数组下标。
    source_index_by_handle: HashMap<HandleKey, usize>,
    /// modification 句柄 → 数组下标。
    modification_index_by_handle: HashMap<HandleKey, usize>,
    /// 逐源的宿主内容版本（下标与 `document.audio_sources` 对齐）。
    content_versions: Vec<u64>,
    /// 已销毁的区间槽位；保留编号，避免其余宿主对象的状态下标变化。
    destroyed_regions: HashSet<usize>,
    /// region sequence 的 ModelRef 指针 → 下标（按首次出现顺序分配）。
    ///
    /// 【为什么按指针】高层 trait 不交出 region → sequence 的那条边（见
    /// `create_playback_region` 的说明），只能靠"ARA 在同一文档内给同一对象稳定指针"
    /// 这一事实来推。顺序按首次出现，与宿主创建顺序一致。
    sequence_index_by_model_ref: HashMap<usize, usize>,
    /// 还没收齐的宿主分割（跨帧候选，见 `SplitCandidate`）。
    split_candidates: Vec<SplitCandidate>,
    /// 上一帧映射时的 region 集合快照，供宿主分割的**结构 diff** 使用。
    ///
    /// 【为什么需要它】GUI 分割会显式 `register_split`，而宿主分割（REAPER 里按 S）
    /// 只表现为"一个 region 消失 + 两个 region 出现"，没有任何显式谱系 ⇒ 参数曲线
    /// 只能靠几何兜底猜祖先，兄弟半段会互相污染（用户报障：分割后 Clip 内产生偏移）。
    /// 记下上一帧就能按**只用 ARA 自己给的量**识别分割（见 `register_host_split_lineage`）。
    lineage_frame: Vec<RegionFrame>,
    /// 上一次成功重算所消费的模型指纹（见 [`ModelHandle::remap_and_log`]）。
    ///
    /// 【为什么需要它】宿主的编辑事务**不是原子的**：一次分割会在同一毫秒里送来多对
    /// `beginEditing` / `endEditing`，而每次 `end_editing` 都会跑全量重算。实测 Cubase 15
    /// 的一次分割里同一份模型被重算 **4 次**（日志里同一条 `playback_region #0` 的映射读数
    /// 连着出现 4 遍，见 2026-10-13 会话 22:20:54）。每次重算都会重跑参数投影、
    /// `refresh_ui_inventory`、GUID 认领，并可能失效渲染输入 —— 那正是自动应用的
    /// "冻结 → 发布"窗口被自己打断、报 `Conflict: host model changed during automatic apply`
    /// 的直接来源。模型没变就不重算。
    remap_stamp: Option<Vec<RemapInput>>,
    /// 本地补丁：本模型所属文档控制器的图账本（T8）。
    ///
    /// 【为什么模型持有它】桥的账本在 `ControllerAdapter` 里，而诊断路径在插件侧。
    /// `document_controller(|| ...)` 闭包每建一份模型就跑一次，正好在那里把句柄取回来。
    /// 缺席时降级为空账本：图账本是纯观测，拿不到也必须让插件继续工作。
    ledger: std::sync::Arc<ara2_bridge::plugin::GraphLedger>,
}

/// 一次重算所消费的模型输入（每条活 region 一行）。
///
/// 【为什么是这些字段】只取**映射真正消费**的量：槽位与 region key（身份）、宿主 item GUID
/// （`ara-item-*` 认领）、源与修改身份（归属）、序列下标（轨道归属）、四段几何、
/// 以及源侧 PCM 授权位（决定这条 region 能否出声）。**不含**任何会自行漂移的量
/// （时间戳、采样计数），否则去重会永久失效。
///
/// 【为什么第一个字段是槽位而不是 key】key 由 `allocate_region` 分配，而夹具/恢复路径
/// 可以直接构造 `playback_regions` 而**不**碰 `region_keys`。若以 key 起头并用 `?` 过滤，
/// 那些行会被整个丢掉 —— 指纹退化成"空 == 空"，于是**真实变化被当成没变**（这正是
/// `persisted_b_edits_follow_host_identity_…` 暴露的问题）。以槽位起头、key 允许为 0
/// （"尚未分配身份"），每一行都稳定贡献自己的几何。
///
/// 几何用 `to_bits()` 比较：`f64` 不是 `Ord`，而 `-0.0 == 0.0` 却 `to_bits` 不同 ——
/// 这里要的是"宿主重发了同一份几何"，位相等比数值相等更严格、也更贴合意图。
type RemapInput = (
    usize,
    u64,
    Option<String>,
    String,
    String,
    u64,
    u64,
    u64,
    u64,
    Option<usize>,
    bool,
);

/// 一帧里一个活 region 的 ARA 几何（宿主分割结构 diff 的输入）。
#[derive(Clone, PartialEq)]
struct RegionFrame {
    key: u64,
    item: Option<String>,
    audio_source: String,
    audio_modification: String,
    mod_start: f64,
    mod_dur: f64,
    play_start: f64,
    play_dur: f64,
}

impl RegionFrame {
    /// 跨会话稳定的 region 签名（见 `ParameterAtlas::ara_region_signature`）。
    ///
    /// 【为什么分割谱系需要它】`register_host_split` 此前要求三个半段都有 item GUID，
    /// 而 GUID 只在有原生扩展时才拿得到 —— Cubase 上恒为空，于是每次分割都退到几何
    /// 兜底，兄弟半段互相污染。签名只用 ARA 自己给的量，任何宿主都能算。
    fn signature(&self) -> String {
        crate::editor::parameter_atlas::ParameterAtlas::ara_region_signature(
            &self.audio_source,
            &self.audio_modification,
            self.play_start,
            self.play_dur,
        )
    }

    /// 身份判据：同一个 region 在两次映射之间几何不变时视为"同一帧的同一个对象"。
    ///
    /// 【为什么不用 region key】`allocate_region` 复用已销毁的最小槽位，分割后的左半段
    /// 常常拿到父段的 key ⇒ key 不是稳定身份。几何四元组才是。
    fn geometry(&self) -> (String, u64, u64, u64, u64) {
        (
            self.audio_source.clone(),
            self.mod_start.to_bits(),
            self.mod_dur.to_bits(),
            self.play_start.to_bits(),
            self.play_dur.to_bits(),
        )
    }

    /// 这条 region 的参数权威键（item GUID 优先，否则 modification）。
    fn authority_key(&self) -> Option<String> {
        crate::editor::parameter_atlas::authority_key(
            self.item.as_deref(),
            &self.audio_modification,
        )
    }
}

/// 一次**尚未收齐**的宿主分割：父段 + 已经见到的子段。
///
/// 【为什么需要它】结构 diff 只能比较"上一帧"与"这一帧"。宿主把一次分割拆成两轮交付
/// 时（先缩短左段、再创建右段），第一轮只看到 1 个子段（`split_children` 要求至少两段），
/// 第二轮又没有"消失的父段" —— 两条路都不登记，右侧半段从此没有任何参数权威。
///
/// 【为什么是"有界记忆"而不是放宽判据】判据仍然是`子段们**划分**了父段区间`（与单帧
/// 版本等价），只是把"同一帧"换成"最多 N 轮之内"。相邻不是父子，划分才是。
struct SplitCandidate {
    parent: RegionFrame,
    children: Vec<RegionFrame>,
    rounds_left: u8,
}

/// 一次分割最多等多少轮 remap（宿主通常 1~2 轮，8 轮足以覆盖节流与乱序）。
const SPLIT_CANDIDATE_ROUNDS: u8 = 8;
/// 同时跟踪多少条未收齐的分割（长会话里反复分割不会无界增长）。
const MAX_SPLIT_CANDIDATES: usize = 32;

impl SplitCandidate {
    /// 这个子段是否属于本候选（同源、播放区间与源窗口都落在父段内）。
    fn admits(&self, child: &RegionFrame) -> bool {
        if child.audio_source != self.parent.audio_source {
            return false;
        }
        let inside_play = child.play_start >= self.parent.play_start - 1e-6
            && child.play_start + child.play_dur
                <= self.parent.play_start + self.parent.play_dur + 1e-6;
        let inside_mod = child.mod_start >= self.parent.mod_start - 1e-6
            && child.mod_start + child.mod_dur
                <= self.parent.mod_start + self.parent.mod_dur + 1e-6;
        inside_play && inside_mod
    }

    /// 把本帧出现的、还没登记过的子段吸收进来（按 (play_start, play_dur) 去重）。
    fn absorb(&mut self, appeared: &[RegionFrame]) {
        for child in appeared {
            if !self.admits(child) {
                continue;
            }
            let duplicate = self.children.iter().any(|known| {
                known.play_start.to_bits() == child.play_start.to_bits()
                    && known.play_dur.to_bits() == child.play_dur.to_bits()
            });
            if !duplicate {
                self.children.push(child.clone());
            }
        }
        self.children
            .sort_by(|a, b| a.play_start.total_cmp(&b.play_start));
    }

    /// 子段集合是否**划分**了父段的播放区间（首尾相接、合起来覆盖父段）。
    fn partitions_parent(&self) -> bool {
        let close = crate::host::geometry::host_value_compatible;
        if self.children.len() < 2 {
            return false;
        }
        let mut cursor = self.parent.play_start;
        for child in &self.children {
            if !close(child.play_start, cursor) {
                return false;
            }
            cursor = child.play_start + child.play_dur;
        }
        close(cursor, self.parent.play_start + self.parent.play_dur)
    }
}

/// 宿主颜色（三个 0..1 分量）转成插件侧时间线用的 `#rrggbb`。
///
/// 【为什么要转】宿主给的是 float 分量（`ARAColor`），而内核的时间线用 `#rrggbb`
/// 字符串（与 `ClipLocalState::color` 同一格式）。任一分量非有限时返回 `None`：
/// 宁可让轨道留默认色，也不要写一个下游解析不了的字符串。
fn host_color_hex(color: Option<&ara2_bridge::core::Color>) -> Option<String> {
    let color = color?;
    let components = [color.red(), color.green(), color.blue()];
    if components.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(format!(
        "#{:02x}{:02x}{:02x}",
        (components[0].clamp(0., 1.) * 255.).round() as u8,
        (components[1].clamp(0., 1.) * 255.).round() as u8,
        (components[2].clamp(0., 1.) * 255.).round() as u8,
    ))
}

impl Default for ModelHandle {
    fn default() -> Self {
        static NEXT_DOCUMENT: AtomicU64 = AtomicU64::new(1);
        let document_id = NEXT_DOCUMENT.fetch_add(1, Ordering::Relaxed);
        Self {
            document_id,
            session: DocumentSession::new(document_id),
            region_keys: Vec::new(),
            sequence_persistent_ids: Vec::new(),
            destroyed_sequences: HashSet::new(),
            pending_sequence_deaths: Vec::new(),
            head_tail: {
                // 【为什么不用 expect】`Default for ModelHandle` 在**每次文档创建**时
                // 都会跑，而它在宿主进程里 —— 一次 panic 会带走整个 DAW 会话。容量常量
                // 8192 本身只可能因资源耗尽被拒，退到最小合法容量；连最小容量也建不出来
                // 时**降级为 None**（head/tail 快照只是实时优化，不是正确性依赖），
                // 而不是用 `expect` 在这里崩掉整个宿主。
                let make = |capacity| {
                    ara2_bridge::plugin::RealtimeHeadTailAdapter::new(capacity)
                        .map(Arc::new)
                        .ok()
                };
                make(8192).or_else(|| {
                    log::warn!("[ara] head/tail snapshot capacity 8192 refused; falling back to 1");
                    make(1).or_else(|| {
                        log::warn!("[ara] head/tail adapter unavailable at any capacity");
                        None
                    })
                })
            },
            document: AraDocument::default(),
            timeline: None,
            generation: None,
            editing: false,
            model_dirty: false,
            sequence_index_by_handle: HashMap::new(),
            source_index_by_handle: HashMap::new(),
            modification_index_by_handle: HashMap::new(),
            content_versions: Vec::new(),
            destroyed_regions: HashSet::new(),
            sequence_index_by_model_ref: HashMap::new(),
            lineage_frame: Vec::new(),
            split_candidates: Vec::new(),
            remap_stamp: None,
            ledger: std::sync::Arc::new(ara2_bridge::plugin::GraphLedger::default()),
        }
    }
}

impl ModelHandle {
    /// head/tail 快照适配器；资源耗尽导致创建失败时为 `None`（降级，不影响正确性）。
    pub(crate) fn head_tail(&self) -> Option<Arc<ara2_bridge::plugin::RealtimeHeadTailAdapter>> {
        self.head_tail.clone()
    }
    /// 本应用渐变位于clip边界内，没有额外head/tail；查询使用SDK的零分配快照接口。
    fn publish_head_tail(&self) -> Result<(), AraError> {
        let Some(head_tail) = self.head_tail.as_ref() else {
            // 适配器创建失败（内存耗尽）时降级：head/tail 快照是实时读取的**优化**，
            // 缺席只让宿主读边界包络时走非快照路径，不影响正确性。这里**不能**报错 ——
            // 调用方在 remap 主路径上，报错会把一次成功的重建整体判失败。
            log::warn!("[ara] head/tail adapter unavailable; skipping snapshot publish");
            return Ok(());
        };
        let entries = self
            .region_keys
            .iter()
            .enumerate()
            .filter(|(index, _)| !self.destroyed_regions.contains(index))
            .map(|(_, key)| ara2_bridge::core::HeadTailEntry::new(*key, 0., 0.))
            .collect::<Result<Vec<_>, _>>()?;
        head_tail.install(entries)
    }
    /// 借出本模型的文档生命周期，供工厂身份通知及 renderer 绑定使用。
    pub(crate) fn session(&self) -> Arc<DocumentSession> {
        self.session.clone()
    }

    /// 本地补丁：本模型的图账本（T8）。诊断路径读它，模型侧只负责交出句柄。
    pub fn ledger(&self) -> &std::sync::Arc<ara2_bridge::plugin::GraphLedger> {
        &self.ledger
    }

    /// 本地补丁：接管上游控制器已建好的账本（T8）。
    pub(crate) fn adopt_ledger(
        &mut self,
        ledger: std::sync::Arc<ara2_bridge::plugin::GraphLedger>,
    ) {
        // 两处都要：诊断路径从文档侧读（`stuck_report`），模型自己在收口时打读数。
        self.session
            .set_graph_ledger(std::sync::Arc::clone(&ledger));
        self.ledger = ledger;
    }
    /// 新建一份空模型。
    pub fn new() -> Self {
        Self::default()
    }

    /// 分配一个 region 槽位、登记身份并落盘这份 region 数据。
    ///
    /// 【为什么复用已销毁的槽位，而不是永远 `push`】`region_keys` /
    /// `destroyed_regions` / `document.playback_regions` 此前只增不减，长会话里随
    /// **编辑次数**无界增长（每次分割/移动都会 destroy + create）。取已销毁槽位中
    /// 最小的一个复用，就把三者都钉在"同时存活的 region 数"这个上界内。
    ///
    /// 【为什么复用是安全的】宿主把我们的 `usize` 状态当**不透明令牌**用；region
    /// 销毁后按 ARA 契约宿主不再引用它，所以把该槽位交给新 region 不会让任何存活对象
    /// 串号。存活 region 的槽位（进而 `ara-clip-N` 身份与 `renderer` 分配）一个都不动 ——
    /// 这正是"删除不压缩下标"这条既有不变式仍然成立的原因。
    fn allocate_region(&mut self, key: u64, region: AraPlaybackRegion) -> Result<usize, AraError> {
        let reused = self.destroyed_regions.iter().copied().min();
        let index = match reused {
            Some(slot) => slot,
            None => self.document.playback_regions.len(),
        };
        region_owners()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .register(key, self.document_id, index)
            .map_err(|error| {
                // 【为什么必须把真因接进状态链】失败对宿主只表现为一个 null region，
                // 而宿主随后**完全停止交付 region** —— 界面停在"等待宿主音频"，
                // 用户唯一能做的是删掉插件再添加，而他不知道原因。
                let reason = format!("region identity rejected: {error:?}");
                self.session.note_region_rejection(reason);
                AraError::InvalidState("region identity already owned or invalid")
            })?;
        match reused {
            Some(slot) => {
                self.region_keys[slot] = key;
                self.destroyed_regions.remove(&slot);
                self.document.playback_regions[slot] = region;
            }
            None => {
                self.region_keys.push(key);
                self.document.playback_regions.push(region);
            }
        }
        Ok(index)
    }

    /// 当前帧的活 region 几何快照（宿主分割结构 diff 的输入）。
    fn region_frames(&self) -> Vec<RegionFrame> {
        let items = self.session.region_items.lock_or_recover();
        self.document
            .playback_regions
            .iter()
            .enumerate()
            .filter(|(slot, _)| !self.destroyed_regions.contains(slot))
            .filter_map(|(slot, region)| {
                let key = *self.region_keys.get(slot)?;
                Some(RegionFrame {
                    key,
                    item: items.get(&key).cloned(),
                    audio_source: region.audio_source_persistent_id.clone(),
                    audio_modification: region.audio_modification_persistent_id.clone(),
                    mod_start: region.start_in_modification_time,
                    mod_dur: region.duration_in_modification_time,
                    play_start: region.start_in_playback_time,
                    play_dur: region.duration_in_playback_time,
                })
            })
            .collect()
    }

    /// 一次分割的子段集合（只用 ARA 自己给的量判定）。
    ///
    /// 【为什么不再要求"恰好两段且左右相邻"】旧判据（`is_host_split`）只认
    /// "1 个 region 消失 + 恰好 2 个出现、且左右源窗严格相邻"。一次多刀分割（N 段）、
    /// 或宿主在分割里顺带做了一点别的事，都不满足 —— 于是谱系一条都不登记，曲线退到
    /// 几何兜底。新判据只要求四件事，全部可判定：
    ///
    /// 1. 同一条 `audio_source`；
    /// 2. 每个子段的源窗落在父段源窗内；
    /// 3. **至少两段**（一段不是分割）；
    /// 4. 按源窗起点排序后**首尾相接且合起来覆盖父段源窗** —— 这一条把
    ///    "删掉一段 + 无关地新建了别的"排除掉，是本判据的安全边界。
    fn split_children<'a>(
        parent: &RegionFrame,
        appeared: &[&'a RegionFrame],
    ) -> Vec<&'a RegionFrame> {
        let close = crate::host::geometry::host_value_compatible;
        let mut children = appeared
            .iter()
            .copied()
            .filter(|child| {
                child.audio_source == parent.audio_source
                    && child.mod_start >= parent.mod_start - 1e-6
                    && child.mod_start + child.mod_dur <= parent.mod_start + parent.mod_dur + 1e-6
            })
            .collect::<Vec<_>>();
        if children.len() < 2 {
            return Vec::new();
        }
        children.sort_by(|a, b| {
            a.mod_start
                .partial_cmp(&b.mod_start)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cursor = parent.mod_start;
        for child in &children {
            if !close(child.mod_start, cursor) {
                return Vec::new();
            }
            cursor = child.mod_start + child.mod_dur;
        }
        if !close(cursor, parent.mod_start + parent.mod_dur) {
            return Vec::new();
        }
        children
    }

    /// 宿主分割的**结构 diff**：比较上一帧与当前帧的 region 集合，识别"父段消失、
    /// 若干子段出现"，并把谱系登记成与 GUI 分割相同的形状
    /// （`split_parents` + `split_seed` + `split_media_from`）。
    ///
    /// 【为什么必须有它】GUI 分割走 `host_split::execute`，显式 `register_split`；而宿主
    /// 分割（REAPER 里按 S）只表现为 region 的销毁与创建，**一个谱系都不写** ⇒ 右半段
    /// 落到几何兜底层，兄弟半段被当成合法祖先，曲线整体平移一个分割长度（用户报障：
    /// 宿主分割后 Clip 内产生偏移）。
    ///
    /// 【为什么没有 seed 就不登记】`register_split` 的 seed 是父段（分割前）的**完整**
    /// 窗口，右半段靠它做包含判定。父段没有曲线记录时（用户从没编辑过它）没有谱系可
    /// 继承，此时退回几何兜底，与本次改动之前完全一致 —— 宁可不登记，也不要造一条
    /// 解析不出祖先的假谱系。
    fn register_host_split_lineage(&mut self) {
        let current = self.region_frames();
        let previous = std::mem::take(&mut self.lineage_frame);
        let mut appeared: Vec<RegionFrame> = Vec::new();
        let mut fresh: Vec<SplitCandidate> = Vec::new();
        if !previous.is_empty() {
            let current_geometry: HashSet<_> = current.iter().map(RegionFrame::geometry).collect();
            let previous_geometry: HashSet<_> =
                previous.iter().map(RegionFrame::geometry).collect();
            let disappeared = previous
                .iter()
                .filter(|frame| !current_geometry.contains(&frame.geometry()))
                .cloned()
                .collect::<Vec<_>>();
            appeared = current
                .iter()
                .filter(|frame| !previous_geometry.contains(&frame.geometry()))
                .cloned()
                .collect();
            for parent in disappeared {
                let refs = appeared.iter().collect::<Vec<_>>();
                let children = Self::split_children(&parent, &refs)
                    .into_iter()
                    .cloned()
                    .collect::<Vec<_>>();
                fresh.push(SplitCandidate {
                    parent,
                    children,
                    rounds_left: SPLIT_CANDIDATE_ROUNDS,
                });
            }
        }
        // 【先吸收、后收口】老候选拿本帧的新子段（跨帧送达）；新候选可能本帧就齐。
        let mut kept: Vec<SplitCandidate> = Vec::new();
        let mut settled: Vec<SplitCandidate> = Vec::new();
        let mut expired = 0_usize;
        for mut candidate in std::mem::take(&mut self.split_candidates) {
            candidate.absorb(&appeared);
            if candidate.partitions_parent() {
                settled.push(candidate);
                continue;
            }
            candidate.rounds_left = candidate.rounds_left.saturating_sub(1);
            if candidate.rounds_left == 0 {
                expired += 1;
            } else {
                kept.push(candidate);
            }
        }
        let fresh_count = fresh.len();
        for mut candidate in fresh {
            candidate.absorb(&appeared);
            if candidate.partitions_parent() {
                settled.push(candidate);
            } else {
                kept.push(candidate);
            }
        }
        if kept.len() > MAX_SPLIT_CANDIDATES {
            let excess = kept.len() - MAX_SPLIT_CANDIDATES;
            kept.drain(..excess);
        }
        self.split_candidates = kept;
        for candidate in &settled {
            let refs = candidate.children.iter().collect::<Vec<_>>();
            self.register_host_split(&candidate.parent, &refs);
        }
        if !settled.is_empty() {
            // 【为什么打一行】"谱系登记成功"与"退回几何兜底"在用户侧是同一个画面，
            // 只有这行日志能把"这次分割到底有没有继承参数"变成可核对的事实。
            log::info!(
                "[ara] split lineage registered: {} candidate(s), parent=ara-region-{}, \
                 children={} (fresh={fresh_count})",
                settled.len(),
                settled[0].parent.key,
                settled.iter().map(|c| c.children.len()).sum::<usize>(),
            );
        }
        if expired > 0 {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.split.pending-expired",
                format_args!(
                    "[ara] {expired} host split candidate(s) never completed within \
                     {SPLIT_CANDIDATE_ROUNDS} remaps; their children keep the geometric fallback"
                ),
            );
        }
        self.lineage_frame = current;
    }

    /// 登记一次宿主分割（父段 + 全部子段）：曲线谱系尽力而为，媒体身份无条件登记。
    fn register_host_split(&mut self, parent: &RegionFrame, children: &[&RegionFrame]) {
        // 【P2：结构 diff 给出的父】宿主不调用 `cloneAudioModification` 时（或调用顺序
        // 不可预期时），分割只表现为"父段 region 消失、若干子段 region 出现"。这里用父
        // region 的 modification 播种**每个子段各自**的 modification。
        //
        // 【为什么每个子段都要播】此前只有右半段有种子（`register_ara_split` 的
        // `right_signature`），左半段靠"region key 槽位复用"的巧合继承 —— 而
        // `allocate_region` 复用已销毁槽位，那是巧合而非保证。
        //
        // 【为什么这一层对两种宿主都跑】它只用 ARA 自己给的量（region 的
        // `audio_modification` persistentID），不依赖 item GUID，也不依赖任何原生扩展。
        // 有 GUID 的宿主走下面两条既有谱系，无 GUID 的宿主此前只能落几何兜底 ——
        // 现在两条路都先把 modification 播好，兜底层因此不再被需要。
        {
            let mut edits = self.session.edits.lock_or_recover();
            let parent_key = parent.authority_key();
            for child in children {
                let child_key = child.authority_key();
                if let (Some(parent_key), Some(child_key)) = (parent_key.as_deref(), child_key) {
                    if edits.atlas.seed_authority(parent_key, &child_key) {
                        edits.revision = edits.revision.saturating_add(1);
                    }
                }
            }
        }
        // 【谱系边本身也要记】权威已经播好了（上面那块），但 `split_parents` 是诊断与
        // `find()` 旧路径的唯一读数：`register_split` 需要 per-clip 种子才建它，而
        // 新路径的种子是**权威**、不是 per-clip 记录。不补这一步，日志里会出现
        // "曲线继承成功但没有任何谱系边"的自相矛盾形状。
        if let Some(parent_item) = parent.item.as_deref() {
            let mut edits = self.session.edits.lock_or_recover();
            for child in children {
                if let Some(child_item) = child.item.as_deref() {
                    if child_item != parent_item {
                        edits.atlas.note_split_parent(parent_item, child_item);
                    }
                }
            }
        }
        // 【ARA-only 宿主的谱系】没有 item GUID 时（Cubase / Nuendo / Studio One 的常态）
        // 改按 region 签名登记。此前这里直接 return，于是那些宿主上每次分割都退到几何
        // 兜底，兄弟半段被当成合法祖先 —— 用户看到
        // `Conflict: ambiguous source parameter ancestry`，或曲线整体平移一个分割长度。
        //
        // 【为什么签名版也要求 seed】与 item 版同口径：seed 是父段**完整**窗口，
        // 子段靠包含判定继承。父段没有曲线记录时没有可继承的东西，退回几何兜底
        // 与改动前完全一致 —— 宁可不登记，也不要造一条解析不出祖先的假谱系。
        let has_guids = parent.item.is_some() && children.iter().all(|child| child.item.is_some());
        if !has_guids {
            let mut edits = self.session.edits.lock_or_recover();
            if let Some(seed) = edits.atlas.ara_split_seed(&parent.signature()) {
                let seed = seed.clone();
                for child in children {
                    match edits.atlas.register_ara_split(
                        &parent.signature(),
                        &child.signature(),
                        Some(seed.clone()),
                    ) {
                        Ok(()) => edits.revision = edits.revision.saturating_add(1),
                        Err(error) => {
                            log::warn!("[ara] ARA split lineage not registered: {error}")
                        }
                    }
                }
            }
            drop(edits);
            // 【为什么 ARA-only 上**不**登记 `split_media_from`】那条边按 clip id 索引，
            // 而它的消费方 `derive_split_media` 要先在时间线里找到**父段**才能继承 ——
            // 但父段在**同一帧**已被销毁（`remap_and_log` 只发布存活 region），
            // 于是这条边永远读不到，只是把表撑大（且键是会被复用的 region key，
            // 见 `allocate_region`，复用后会指向语义完全不同的"父段"）。
            //
            // 【为什么不需要它】映射层对**每一条**存活 region 都写了 `source_path` 与
            // 完整 take（`ara/mapping.rs`），PCM 又按 `audio_source.persistent_id` 共享
            // ⇒ 子段与父段同源，PCM 早已在位。也就是说 ARA-only 宿主上派生是**多余的**。
            log::info!(
                "[ara] ARA split detected: parent=ara-region-{} children={}",
                parent.key,
                children.len()
            );
            return;
        }
        let Some(parent_item) = parent.item.as_deref() else {
            // 有原生扩展但清单还没认领：此时两个系统都给不出稳定键，只能等下一次。
            return;
        };
        // 媒体派生谱系的锚点：沿用父 item GUID 的那一段（原生分割里通常是左半段），
        // 它才是"还活着的、有媒体授权的那一条"。找不到就退回第一个子段。
        let anchor_item = children
            .iter()
            .filter_map(|child| child.item.as_deref())
            .find(|item| *item == parent_item)
            .or_else(|| children.iter().find_map(|child| child.item.as_deref()));
        // 【登记子段窗口来源】左段（沿用父 item GUID 的那一条）的窗口是**可信的**：
        // 分割不改它的 D_STARTOFFS。子段按播放轴偏移从它切出来，宿主读数只作校验 ——
        // 右半段的 D_STARTOFFS 才是会被分割推过镜像中点的那个量。
        if let Some(left) = children
            .iter()
            .find(|child| child.item.as_deref() == Some(parent_item))
            .or_else(|| children.first())
        {
            if let Some(left_item) = left.item.clone() {
                for child in children {
                    let Some(child_item) = child.item.as_deref() else {
                        continue;
                    };
                    if child_item == left_item {
                        continue;
                    }
                    let offset = child.play_start - left.play_start;
                    if !(offset.is_finite() && offset > 1e-9) {
                        continue;
                    }
                    self.session.note_split_window(
                        child_item,
                        crate::render::document::SplitWindowSource {
                            left: left_item.clone(),
                            offset_sec: offset,
                            length_sec: child.play_dur,
                        },
                    );
                }
            }
        }
        let mut edits = self.session.edits.lock_or_recover();
        // 曲线谱系：只有父段有 live 曲线可继承时才登记（`register_split` 的 seed 是
        // 父段完整窗口，子段靠它做包含判定）。
        //
        // 【为什么不再 `return`】此前 seed 缺失会连**媒体派生谱系**一起跳过，子段
        // 因此既无曲线祖先也无媒体身份 —— 波形要等下一次授权才出现。媒体身份只依赖
        // "子段与父段同源"，与父段有没有曲线无关，两者必须解耦。
        let seed = edits.atlas.split_seed(parent_item);
        for child in children {
            let Some(child_item) = child.item.as_deref() else {
                continue;
            };
            // 沿用父 item GUID 的那一段自己就带着父段的权威，无需播种。
            if child_item == parent_item {
                continue;
            }
            if let Some(seed) = seed.clone() {
                match edits
                    .atlas
                    .register_split(parent_item, child_item, Some(seed))
                {
                    Ok(()) => edits.revision = edits.revision.saturating_add(1),
                    Err(error) => log::warn!("[ara] host split lineage not registered: {error}"),
                }
            }
        }
        drop(edits);
        // 媒体派生谱系：子段的源与父段是同一个文件（父段已授权 ⇒ 子段立刻有波形）。
        if let Some(anchor_item) = anchor_item {
            for child in children {
                if let Some(child_item) = child.item.as_deref() {
                    if child_item != anchor_item {
                        self.session
                            .split_media_from
                            .lock_or_recover()
                            .insert(child_item.to_owned(), anchor_item.to_owned());
                    }
                }
            }
        }
        log::info!(
            "[ara] host split detected: parent={parent_item} children={}",
            children.len()
        );
    }

    /// 当前累积到的 ARA 文档（供映射与测试）。
    pub fn document(&self) -> &AraDocument {
        &self.document
    }

    /// 最近一次 `end_editing` 映射出来的时间线。
    pub fn timeline(&self) -> Option<&TimelineState> {
        self.timeline.as_ref()
    }

    /// 协商到的 ARA 版本。
    pub fn generation(&self) -> Option<ApiGeneration> {
        self.generation
    }

    /// 把当前模型图同步进 session 的派生表（`regions` / `clip_ids` / `sequence_*`）。
    ///
    /// 【为什么与 `remap_and_log` 分开】这两张表是**查询用**的：渲染路径、工作区投影、
    /// 参数身份都从它们读"现在有哪些 region、每个 region 的 clip id 是什么"。
    /// `remap_and_log` 是重活（全文档映射 + 参数投影 + 宿主清单刷新），只在
    /// `end_editing` 跑；而宿主可能在收口之前就来读这些表 —— 那时它们必须是新的，
    /// 否则读到的是上一代图（用户看到"编辑操作不更新 ARA"）。
    ///
    /// 它很便宜（两次表重建 + 一次 clone），因此每个 mutation 回调都可以直接调，
    /// 不需要置脏、不需要合并。
    /// 把 `slots[index]` 设为 `value`；必要时先把向量补到 `index + 1`。
    ///
    /// 【为什么不是 `insert`】见 `create_region_sequence` 的说明：这里要的是**覆盖**
    /// 一个已存在的槽位。`insert` 在 `index < len` 时会右移，把身份表整体错位。
    /// 【G3】记下一条被销毁序列的身份，供紧接着的重建认领。
    ///
    /// 【为什么有上界】宿主可能整批重建序列（切屏重载 take）。留着最近 64 条足够覆盖
    /// 任何一次重建窗口，又不会在长会话里无界增长。
    fn remember_sequence_death(&mut self, order: i32, name: Option<String>, key: String) {
        const MAX_PENDING_SEQUENCE_DEATHS: usize = 64;
        self.pending_sequence_deaths
            .retain(|(_, _, known)| *known != key);
        self.pending_sequence_deaths.push((order, name, key));
        while self.pending_sequence_deaths.len() > MAX_PENDING_SEQUENCE_DEATHS {
            self.pending_sequence_deaths.remove(0);
        }
    }

    /// 【G3】新序列能否认领一条已销毁序列的身份。
    ///
    /// 【为什么要求唯一】两条待认领序列在 `(order_index, name)` 上并列时，挑一条等于
    /// 把参数挂到别人的轨上 —— 那比"这次没续上"严重得多。
    fn claim_sequence_death(&mut self, order: i32, name: Option<&str>) -> Option<String> {
        let matches = self
            .pending_sequence_deaths
            .iter()
            .enumerate()
            .filter(|(_, (known_order, known_name, _))| {
                *known_order == order && known_name.as_deref() == name
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let [index] = matches.as_slice() else {
            return None;
        };
        Some(self.pending_sequence_deaths.remove(*index).2)
    }
    fn set_sequence_slot<T: Default>(slots: &mut Vec<T>, index: usize, value: T) {
        if slots.len() <= index {
            slots.resize_with(index + 1, T::default);
        }
        slots[index] = value;
    }

    fn sync_session_derived_tables(&mut self) {
        {
            let mut regions = self.session.regions.lock_or_recover();
            regions.clear();
            for (slot, region) in self.document.playback_regions.iter().enumerate() {
                if !self.destroyed_regions.contains(&slot) {
                    if let Some(key) = self.region_keys.get(slot) {
                        regions.insert(*key, region.clone());
                    }
                }
            }
        }
        // 【T13：`clip_ids` 只能有**一个**写者，而那个写者是 `remap_and_log`】
        //
        // 此前这里会把整张表重写成 `ara-region-{key}`，而 `remap_and_log` 随后把它
        // 升级成 `ara-item-{guid}`。于是**任何**一次模型回调（尤其是 REAPER 在工程
        // 加载时发的 `updateMusicalContextContent`）都会把 id 打回 `ara-region-` 形态。
        // 而清单侧的"重贴标签"要拿 `clip_ids[key]` 去已发布的时间线里找同 id 的 clip
        // —— 形态对不上就永远匹配不到，宿主轨的 id 停在上一轮的值，`present_host_inventory`
        // 于是**凭一个过期 id 新建一条轨并把所有 clip 搬过去**（"分割后多出一条轨"）。
        //
        // 这里只补**缺席**的 key（新 region 在 `remap` 之前也要有身份），绝不覆盖已有值。
        let live_keys: Vec<u64> = self
            .region_keys
            .iter()
            .enumerate()
            .filter(|(slot, _)| !self.destroyed_regions.contains(slot))
            .map(|(_, key)| *key)
            .collect();
        let mut clip_ids = self.session.clip_ids.lock_or_recover();
        // 已销毁的槽位在这里清掉；存活槽位**保留已发布的形态**。
        clip_ids.retain(|key, _| live_keys.contains(key));
        for key in live_keys {
            clip_ids
                .entry(key)
                .or_insert_with(|| format!("ara-region-{key}"));
        }
    }

    /// 把当前文档映射成时间线并记一行摘要。
    ///
    /// 【为什么在 `end_editing` 调】ARA 约定：一次编辑的所有图变更都在
    /// `begin_editing` … `end_editing` 之间，收口点之后模型才是稳定的。
    /// 重算输入的完整指纹，按 region key 排序（见 [`RemapInput`]）。
    ///
    /// 【为什么排序】宿主交付同一份模型的顺序不保证一致；排序后"同一份模型"才有
    /// 同一个指纹，否则去重会因顺序抖动而失效。
    fn remap_fingerprint(&self) -> Vec<RemapInput> {
        let items = self.session.region_items.lock_or_recover();
        let mut rows = self
            .document
            .playback_regions
            .iter()
            .enumerate()
            .filter(|(slot, _)| !self.destroyed_regions.contains(slot))
            .map(|(slot, region)| {
                // 未分配身份时用 0（"未知"）而不是把这一行丢掉 —— 丢掉会让指纹退化。
                let key = self.region_keys.get(slot).copied().unwrap_or(0);
                let sample_access = self
                    .document
                    .audio_sources
                    .iter()
                    .find(|source| source.persistent_id == region.audio_source_persistent_id)
                    .is_some_and(|source| source.sample_access_enabled);
                (
                    slot,
                    key,
                    items.get(&key).cloned(),
                    region.audio_source_persistent_id.clone(),
                    region.audio_modification_persistent_id.clone(),
                    region.start_in_modification_time.to_bits(),
                    region.duration_in_modification_time.to_bits(),
                    region.start_in_playback_time.to_bits(),
                    region.duration_in_playback_time.to_bits(),
                    region.region_sequence_index,
                    sample_access,
                )
            })
            .collect::<Vec<_>>();
        rows.sort();
        rows
    }

    /// 这次 `end_editing` 是否需要重算 —— 即"已发布过时间线、且模型指纹未变"。
    ///
    /// 抽成独立函数是为了让判据本身可测：重算的**副作用**（版本推进、清单刷新）散布在
    /// 好几处，无法从外部观察"到底跑没跑"；而"该不该跑"是一个纯函数。
    fn remap_is_redundant(&self) -> bool {
        self.timeline.is_some() && self.remap_stamp.as_ref() == Some(&self.remap_fingerprint())
    }

    /// 记下"自上次成功发布以来模型变过"（见 [`Self::model_dirty`]）。
    fn mark_model_dirty(&mut self) {
        self.model_dirty = true;
        // 【T23】把"脏了多久"变成插件侧可读的事实：`model_dirty` 是本模型私有的布尔，
        // 而"宿主不再 flush"这条死路需要非 ARA 线程也能判断截止期。
        self.session.note_model_dirty();
    }

    fn remap_and_log(&mut self) {
        // 先更新 session 的派生表：它们必须**立刻**反映新图，否则渲染/工作区路径
        // 在收口前就读到旧的 region 集合（见 `sync_session_derived_tables`）。
        self.sync_session_derived_tables();
        // 【为什么孤儿回收排在这里】它在**文档级**、且每次 `end_editing` 都跑一次 ——
        // 这两点缺一不可：逐 renderer 的范围会把别的轨道的权威误删（见
        // `reclaim_orphans` 的说明），而放在下面那道去重闸门之后会让"同一份模型连着
        // 送 4 遍"时宽限期根本不前进。回收是**有宽限**的（`ORPHAN_GRACE_ROUNDS`），
        // 因此它不会碰到"刚销毁、马上撤销回来"的那条 modification。
        {
            let live = self
                .document
                .audio_modifications
                .iter()
                .filter(|modification| !modification.persistent_id.is_empty())
                .map(|modification| modification.persistent_id.clone())
                .collect::<std::collections::BTreeSet<_>>();
            let reclaimed = self
                .session
                .edits
                .lock_or_recover()
                .atlas
                .reclaim_orphans(&live);
            if reclaimed > 0 {
                log::info!(
                    "[ara] reclaimed {reclaimed} orphan parameter authority set(s) whose \
                     modification is no longer in the document graph"
                );
            }
        }
        // 【幂等：模型没变就不重算】宿主的编辑事务不是原子的，一次分割会送来多对
        // `beginEditing` / `endEditing`，每次 `end_editing` 都跑一遍全量重算 —— 实测
        // Cubase 15 是同**一份模型**连着算 4 遍（见 `ModelHandle::remap_stamp`）。
        // 已经发布过时间线、且这次交付的模型指纹与上次完全相同 ⇒ 这次重算不会产出
        // 任何新东西，只会重复失效渲染输入、拉大自动应用的冲突窗口。
        //
        // 【为什么不是按时间做去抖】去抖需要一个定时器/线程，而"模型是否相同"是一个
        // **可判定的事实**：相同就一定不用重算，与经过了多少毫秒无关。用指纹既确定
        // 又可测（见 `end_editing_skips_the_remap_when_the_model_did_not_change`）。
        if self.remap_is_redundant() {
            self.model_dirty = false;
            // 【T23】"模型没变"也是一次成功收口：截止期必须跟着清，否则下一轮
            // `request_settle_if_overdue` 会对一份已经收敛的模型反复请求 flush。
            self.session.note_model_clean();
            // 正常路径的跟踪信息：由日志级别控制，不套限流（见 `view.rs::EnterExit`）。
            log::debug!(
                "[ara] endEditing delivered an unchanged model ({} region(s)); \
                 the published timeline is kept without a re-map",
                self.remap_stamp.as_ref().map_or(0, Vec::len)
            );
            // 准备队列仍要保证在跑：宿主可能刚在这一轮授予了 PCM 访问，而那不是模型变化。
            self.session.prepare_renderers();
            return;
        }
        let fingerprint = self.remap_fingerprint();
        let session = self.session.clone();
        let transaction = session.transaction.lock_or_recover();
        let mut document = self.document.clone();
        document.playback_regions = document
            .playback_regions
            .into_iter()
            .enumerate()
            .filter(|(index, _)| !self.destroyed_regions.contains(index))
            .map(|(_, region)| region)
            .collect();
        // 【为什么要在这里注入工程速率】ARA 本体没有文档采样率（`ARADocumentProperties`
        // 只有 `structSize` 与 `name`），而映射层必须有它才能把宿主的"工程速率秒"折算成
        // 源自己的秒 —— 否则 48 kHz 素材进 44.1 kHz 工程时 `playback_rate` 恒为 1，
        // 素材按自身速率播完、剩下的工程时间留白（用户听感：ARA 前后不一致）。
        // 唯一权威来源是 VST3 `setupProcessing` 的 `ProcessSetup::sampleRate`，
        // 它由各 renderer 实例持有，取**第一个非零**值（同一文档的所有 renderer
        // 共享同一份 companion API 速率，不一致本身就是宿主错误）。
        document.project_sample_rate = session
            .renderer_owners()
            .into_iter()
            .map(|owner| owner.setup_sample_rate())
            .find(|rate| *rate > 0.0);
        // 【身份：region key + sequence persistentID】位置派生的 id 会在宿主删除/重排之后
        // 整体前移，而参数表按 id 索引 —— 于是编辑挂错 clip 或查不到身份。key 与
        // persistentID 都是宿主保证稳定的令牌，由它们派生身份（见 `MappingIdentities`）。
        let live_keys: Vec<u64> = self
            .region_keys
            .iter()
            .enumerate()
            .filter(|(slot, _)| !self.destroyed_regions.contains(slot))
            .map(|(_, key)| *key)
            .collect();
        let identities = crate::ara::mapping::MappingIdentities {
            region_keys: live_keys,
            // 【为什么不再过滤已销毁槽位】过滤会让后续身份整体前移一位：新序列
            // 拿到别人的 id、真正的新轨退回位置身份（"分割后多出一条轨"的成因）。
            // 已销毁槽位在这里保留空串，由 `live_sequences` 表达"不生成轨道"。
            sequence_ids: self.sequence_persistent_ids.clone(),
            live_sequences: (0..self.sequence_persistent_ids.len())
                .map(|slot| !self.destroyed_sequences.contains(&slot))
                .collect(),
        };
        match crate::ara::mapping::ara_document_to_timeline_with_identities(&document, &identities)
        {
            Ok(outcome) => {
                let crate::ara::mapping::MappingOutcome {
                    mut timeline,
                    clip_regions,
                    skipped,
                } = outcome;
                if !skipped.is_empty() {
                    // 逐 region 跳过是**局部**问题：记一行明细，但会话继续可用。
                    // 这正是"一个零长度/MIDI item 打死整个插件"的修复点。
                    log::warn!(
                        "[ara] skipped {} of {} playback region(s): {}",
                        skipped.len(),
                        document.playback_regions.len(),
                        skipped
                            .iter()
                            .map(|s| format!("{}:{}", s.index, s.reason.as_str()))
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                }
                // 让界面也知道"有 N 段没显示"：只 log 会让 MIDI / 零长度 item 在插件里
                // 静默消失，用户以为自己的内容丢了（见 `DocumentSession::skipped_regions`）。
                *self.session.skipped_regions.lock_or_recover() = skipped
                    .iter()
                    .map(|s| (s.index, s.reason.as_str().to_owned()))
                    .collect();
                // 【默认合成开关：只对**没有持久化编辑记录**的轨道播种，不再每次覆写】
                //
                // 此前这里每次重映射都无条件把每条轨道改成 `compose_enabled = true` +
                // `NsfHifiganOnnx`。那等于用"默认值"盖掉用户的选择：用户关掉合成、
                // 或把算法换成 WORLD 之后，只要宿主发生一次编辑就被抹回默认。
                // 而且它把**每一条 clip** 都推进处理器 —— 在 F0 推导出错时，
                // 这就是"大部分 clip 都中招"的放大器（见 `mixdown::apply_host_pitch_edit`
                // 的 F0 来源修复）。
                //
                // 现在：有持久化记录的轨道完全交给 `EditState::apply`（它在
                // `workspace_timeline_locked` 里随后叠加用户选择）；只有**新出现**的
                // 宿主轨道才播种插件默认值，保证开箱即用。
                //
                // 【锁序】此处持有 `transaction`（见上方），而文档声明的顺序是
                // `transaction → timeline → edits → host facts`，先取 `edits` 合法。
                let seeded: std::collections::BTreeSet<String> = self
                    .session
                    .edits
                    .lock_or_recover()
                    .tracks
                    .iter()
                    .map(|track| track.id.clone())
                    .collect();
                for track in &mut timeline.tracks {
                    if seeded.contains(&track.id) {
                        continue;
                    }
                    track.compose_enabled = true;
                    track.pitch_analysis_algo =
                        hifishifter_kernel::state::PitchAnalysisAlgo::NsfHifiganOnnx;
                }
                // 宿主本会话slot不随删除压缩；避免活着的clip因旧region销毁换身份。
                //
                // 【为什么不再在这里重贴 clip.id】身份已在映射层按 region key 派生
                // （`ara-region-{key}`），此处重贴成 `ara-clip-{N}` 会把稳定性又抹掉。
                // 这里只负责把 key → clip id 的映射登记进 `clip_ids` 供反查。
                //
                // 【为什么按 `clip_regions` 对齐而不是顺序 zip】映射层会逐 region 跳过
                // 坏条目；顺序 zip 会在有跳过时整体错位，把身份挂到错误的 clip 上。
                // `clip_regions[i]` 是该 clip 在**过滤后**文档里的下标，与 `region_keys` 同序。
                let mut clip_ids = self.session.clip_ids.lock_or_recover();
                clip_ids.clear();
                for (clip, region_index) in timeline.clips.iter().zip(&clip_regions) {
                    if let Some(key) = self.region_keys.get(*region_index).copied() {
                        clip_ids.insert(key, clip.id.clone());
                    }
                }
                drop(clip_ids);
                // 身份仅由真实region→sequence边与宿主persistentID构建，不从名字/序号推断。
                let mut bindings: crate::state_channel::TrackBindings = timeline
                    .tracks
                    .iter()
                    .map(|t| (t.id.clone(), vec![]))
                    .collect();
                for region in &document.playback_regions {
                    if let Some(index) = region.region_sequence_index {
                        if let Some(track) = timeline.tracks.get(index) {
                            // 【为什么不用 unwrap】`bindings` 由上面的 `timeline.tracks`
                            // 直接构建，当前必然命中；但这条路径跑在 **ARA 模型线程**，
                            // 未来任何一处 divergence 都会变成宿主进程里的硬 panic。
                            if let Some(members) = bindings.get_mut(&track.id) {
                                members.push((
                                    region.audio_modification_persistent_id.clone(),
                                    region.audio_source_persistent_id.clone(),
                                ));
                            }
                        }
                    }
                }
                for members in bindings.values_mut() {
                    members.sort();
                    members.dedup();
                }
                *self.session.track_bindings.lock_or_recover() = bindings.clone();
                // getter可能重入ARA：先发布真实基础图、释放事务，再采集稳定item GUID。
                *self.session.timeline.lock_or_recover() = Some(timeline.clone());
                drop(transaction);
                for owner in self.session.renderer_owners() {
                    owner.refresh_host_state_for_model();
                }
                for owner in self.session.renderer_owners() {
                    owner.refresh_ui_inventory();
                }
                let transaction = session.transaction.lock_or_recover();
                {
                    let items = self.session.region_items.lock_or_recover();
                    let mut ids = self.session.clip_ids.lock_or_recover();
                    let mut claimed = 0usize;
                    for clip in &mut timeline.clips {
                        if let Some((key, _)) = ids.iter().find(|(_, id)| **id == clip.id) {
                            if let Some(item) = items.get(key) {
                                let key = *key;
                                clip.id = format!("ara-item-{item}");
                                ids.insert(key, clip.id.clone());
                                claimed += 1;
                            }
                        }
                    }
                    // 用户报障时最需要的一行：多少 clip 认领到了宿主 item GUID。
                    // 认领不到的会停在 `ara-clip-N`，随后被清单侧 retain 剔除 ——
                    // 这就是"子轨道 Item 没有变成真 Clip"的直接读数。
                    if claimed < timeline.clips.len() {
                        log::warn!(
                            "[ara] {} of {} clip(s) claimed a host item GUID; the rest keep \
                             ara-clip ids and will not survive host inventory presentation",
                            claimed,
                            timeline.clips.len()
                        );
                    }
                }
                // 记下"授权这一刻"的 active take GUID。
                //
                // 【为什么在这里、而不是每次清单刷新都重记】ARA 的 playback region 指向
                // item 的 active take（`probe/ara/MULTI-TAKE-FINDINGS.md` 的 F-3），所以
                // 授权 PCM 属于**那一刻**的 active take。`sync_host_takes` 重建 take 列表
                // 时只有 GUID 相等的那一个能带上授权媒体；用户随后在 REAPER 里切换 active
                // take 时这条记录保持旧值，于是新 take 不会被挂上旧采样 —— 直到 ARA 模型
                // 重新认领（那时 region 的 source 也换了，这里跟着更新）。
                {
                    let items = self.session.region_items.lock_or_recover().clone();
                    let take_by_item = {
                        let tracks = self.session.ui_tracks.lock_or_recover();
                        tracks
                            .values()
                            .flat_map(|track| &track.items)
                            .map(|item| {
                                (item.geometry.item_id.clone(), item.geometry.take_id.clone())
                            })
                            .collect::<std::collections::HashMap<_, _>>()
                    };
                    // 同一时刻记下 active take 的**源文件路径**（可读时）：GUID 是主判据，
                    // 文件路径是"换成同一文件的另一个 take"时的回退身份（见
                    // `DocumentSession::authorized_sources`）。
                    let source_by_item = {
                        let tracks = self.session.ui_tracks.lock_or_recover();
                        tracks
                            .values()
                            .flat_map(|track| &track.items)
                            .filter_map(|item| {
                                item.geometry
                                    .source_file_name
                                    .clone()
                                    .map(|name| (item.geometry.item_id.clone(), name))
                            })
                            .collect::<std::collections::HashMap<_, _>>()
                    };
                    let mut takes = self.session.authorized_takes.lock_or_recover();
                    let mut sources = self.session.authorized_sources.lock_or_recover();
                    // 【为什么不再 `clear()`】这两张表按 region key 索引，key 的终点
                    // 只有两条：`destroy_playback_region` 逐 key 删除、
                    // `DocumentSession::close` 整表清空。一次 remap 看不到某个 key，
                    // 只说明"这一轮清单/几何还没来"（后台恢复、节流、宿主适配器暂时
                    // 不可用），不等于授权失效。整体清空会让 `sync_host_takes` 在
                    // 下一次清单刷新时把所有 Clip 的 `source_path` 清掉 —— 用户看到
                    // 的就是"正在等待宿主完成音频分配"。真正的 take 切换由
                    // `sync_host_takes` 的 GUID 比对处理，不需要在这里清授权。
                    for (key, item) in items.iter() {
                        if let Some(take) = take_by_item.get(item) {
                            takes.insert(*key, take.clone());
                        }
                        if let Some(source) = source_by_item.get(item) {
                            sources.insert(*key, source.clone());
                        }
                    }
                }
                // 宿主分割的结构 diff 必须在**参数投影之前**跑：投影会 `follow_geometry`
                // 并把父段记录置为非活，届时 `split_seed` 就取不到父段窗口了（见
                // `register_host_split_lineage`）。
                self.register_host_split_lineage();
                {
                    let mut edits = self.session.edits.lock_or_recover();
                    if !edits.atlas.is_empty() && !edits.needs_rebind {
                        let projection = (|| {
                            let identities = self.session.parameter_identities_locked(&timeline)?;
                            // 【先换几何，再重建视图】`follow_geometry` 只做查表与派生缓存，
                            // 真正的"视图"（`edits.params`）由 `resync_parameter_view_locked`
                            // 重建 —— 两件事收口在同一个入口，避免"模型重算时重建、清单刷新
                            // 时不重建"的分叉（那正是"分割后参数线停在旧几何"的根）。
                            let followed = edits.atlas.follow_geometry(&timeline, &identities)?;
                            edits.atlas = followed;
                            self.session
                                .resync_parameter_view_locked(&mut edits, &timeline)?;
                            // 无活动region的轨道可能只是mute；保留其曲线，不能当删除清空。
                            edits
                                .tracks
                                .retain(|track| bindings.contains_key(&track.id));
                            Ok::<_, String>(())
                        })();
                        match projection {
                            Ok(()) => {}
                            Err(error) => {
                                // 【为什么不整份中止】参数投影失败只说明"曲线身份这次没能
                                // 对齐"，几何与 PCM 都是好的。此前这里 `return`，于是
                                // 连 `prepare_renderers()` 都不跑 —— `ready` 停在 `false`，
                                // 而它只能由 `prepare_renderers` 置真，宿主若不重发
                                // `end_editing` 就再也回不来（实例永久砖化）。
                                // 现在只记一行、保留旧图集，继续把这份时间线发布出去。
                                log::error!("[ara] source parameter projection conflict: {error}");
                            }
                        }
                    } else if edits.needs_rebind {
                        // 【为什么必须留痕】`needs_rebind` 是**冷恢复/存档还原**的标记，
                        // 唯一的清零点是 `EditState::reconcile`（`state_channel.rs:183`），
                        // 而 `reconcile` 只被恢复路径调用。一旦某次撤销/重载的恢复身份
                        // 对不上，这个标记就停在 `true`：此后用户在宿主里做的**任何**编辑
                        // 都只会发布几何与 PCM，参数投影再也不跑 —— 用户报的症状正是
                        // "无论宿主里怎么编辑 ARA 都不同步，只有删掉插件再添加才好"。
                        //
                        // 【为什么不顺手把它清掉】清掉等于把一次冷恢复降级成普通合并：
                        // `reconcile` 的 `needs_rebind` 分支正是用 `Err` 拒绝"按旧序号猜
                        // 归属"的（想象一下：曲线被挂到另一条轨道上，然后写回宿主）。
                        // 正确的下一步是**先诊断** —— 这一行 + A4 的收口读数正是为此。
                        hifishifter_kernel::log_limiter::emit_limited_key(
                            log::Level::Warn,
                            "ara.rebind.stuck",
                            format_args!(
                                "[ara] parameter projection skipped: the loaded edit state still \
                                 asks for a rebind ({} authority set(s)). Host edits will publish \
                                 geometry and audio only until the pending restore is reconciled.",
                                edits.atlas.authorities.len()
                            ),
                        );
                    }
                }
                if let Err(error) = self.session.edits.lock_or_recover().reconcile(&bindings) {
                    log::error!("[ara] edit identity unresolved: {error}");
                }
                log::info!("{}", crate::ara::summary_line(&document, &timeline));
                log::info!("{}", crate::ara::clip_starts_line(&timeline));
                // 【T8：每次收口把图读数打一遍】这是宿主交付模型之后唯一稳定的时间点；
                // 它给出"五个 create 各发生几次、被拒几次"，因而能回答
                // "region 是宿主没发、还是我们拒了"。
                log::info!("[ara] {}", self.ledger.summary());
                self.log_published_revisions(&timeline);
                self.log_identity_readings(&timeline);
                *self.session.timeline.lock_or_recover() = Some(timeline.clone());
                self.timeline = Some(timeline);
                // 只有**成功发布**才记指纹：失败时留着旧指纹，下一次照常重试。
                self.remap_stamp = Some(fingerprint);
                self.model_dirty = false;
                self.session.note_model_clean();
                self.publish_readiness();
                drop(transaction);
            }
            Err(err) => {
                // 映射失败必须显式记录：它是"DAW 里看到的时间线不对"这类问题的唯一线索。
                //
                // 【为什么不再无条件把实例打成"未就绪"】能走到这里只剩文档级结构错误
                // （JSON 无法解析 / region 数量与声明对不上）。此前这里直接 `return`，
                // 跳过了 `prepare_renderers()` —— 而 `ready` 只能由它置真，宿主若不重发
                // `end_editing` 就永远回不来。现在：
                //   * 已经发布过一份可用时间线时，**保留它**并继续服务（渲染用上一版
                //     几何，比整个实例变砖强得多）；
                //   * 从未成功映射过（首次就失败）才置 `ready=false`。
                // 下一次 `end_editing` 会自然重试；`refresh_host` 的版本比较也会再触发。
                log::error!("[ara] mapping failed: {err:?}");
                if self.timeline.is_none() {
                    self.session.ready.store(false, Ordering::Release);
                    return;
                }
                log::warn!(
                    "[ara] keeping the previously published timeline; the host model will be \
                     re-mapped on the next edit"
                );
                // 【为什么必须显式释放】下面 `prepare_renderers()` 要取同一把
                // `transaction`；`std::sync::Mutex` 不可重入，持着它调用会当场自锁
                // （重开工程时映射失败即触发）。`Ok` 分支已在上面 `drop`，这里必须对齐。
                drop(transaction);
            }
        }
        self.session.prepare_renderers();
    }

    /// 收口之后发一行"读者们拿的是哪一版"的读数。
    ///
    /// 【为什么必须打在收口之后、`prepare_renderers` 之前】用户报的症状是
    /// "宿主里改了，ARA 里不动 / 只有删插件才好"。这条链有四步
    /// （模型回调 → `endEditing` 收口 → 编辑器会话读到 → 渲染输入失效），
    /// **任何一步断了症状都一样**。此前只有第 1、2 步有日志，于是"收口了但
    /// 读者没醒"与"读者醒了但内容不对"无法区分 —— 修 bug 只能靠猜。
    ///
    /// 这三个数正是 `refresh_host`（`editor/session.rs::refresh_host`）用来
    /// 判断"该不该重新加载"的**同一个**三元组（`document.editor_versions()`）。
    /// 所以"日志里 model_rev 变了而界面没变"从此是一眼看得出的事实，
    /// 而不必再在源码里反推哪一条路径该推进它。
    ///
    /// 【为什么单独一个方法】`remap_and_log` 已经很长，且这行日志的**取值**
    /// 要在测试里断言 —— 抽出来才好读，也才好复用（幂等分支也想知道读数）。
    /// 【T-A0-5】把"身份链现在长什么样"打成一行。
    ///
    /// 【为什么与 `[ara] stuck` 分开】那一行回答"渲染链断在哪一环"，这一行回答
    /// "参数权威还认不认得这些 clip"。两者同时出问题时会互相掩盖：`stuck` 说
    /// `regions=11/11 sources=1` 看起来完全健康，而 `authorities=0` 才是用户
    /// "参数全回退"的直接原因。
    fn log_identity_readings(&self, timeline: &TimelineState) {
        let live: Vec<u64> = self
            .region_keys
            .iter()
            .enumerate()
            .filter(|(slot, _)| !self.destroyed_regions.contains(slot))
            .map(|(_, key)| *key)
            .collect();
        let bound = {
            let items = self.session.region_items.lock_or_recover();
            live.iter().filter(|key| items.contains_key(key)).count()
        };
        let (authorities, derived, unbound) = {
            let edits = self.session.edits.lock_or_recover();
            (
                edits.atlas.authorities.len(),
                edits.atlas.regions.len(),
                edits.unbound.len(),
            )
        };
        log::info!(
            "[ara] identity: regions={} bound={} unmatched={} clips={} authorities={} \
             derived={} unbound={}",
            live.len(),
            bound,
            live.len().saturating_sub(bound),
            timeline.clips.len(),
            authorities,
            derived,
            unbound,
        );
    }

    fn log_published_revisions(&self, timeline: &TimelineState) {
        // 【为什么直接读原子量而不调 `editor_versions()`】后者自己取文档
        // `transaction` 锁。`remap_and_log` 的调用点持着同一把锁（它在
        // `drop(transaction)` 之后调用本函数，因此这里也是**无锁**状态）——
        // 直接读三个计数既等价又不引入"锁序里有隐藏依赖"的风险。
        let edits_revision = self.session.edits.lock_or_recover().revision;
        let model_revision = self.session.revision.load(Ordering::Acquire);
        let scope_revision = self.session.scope_revision.load(Ordering::Acquire);
        log::info!(
            "[ara] endEditing published: regions={} model_rev={} edit_rev={} scope_rev={} \
             ready={} editing={}",
            timeline.clips.len(),
            model_revision,
            edits_revision,
            scope_revision,
            self.session.ready.load(Ordering::Acquire),
            self.session.editing_session.load(Ordering::Acquire),
        );
    }
}

impl ModelHandle {
    /// **本次调用来自 ARA model 线程** —— 顺手把宿主线程队列排空。
    ///
    /// 【为什么泵必须在这里】`host::marshal` 的泵此前只装在**宿主 UI 线程**上
    /// （`IPlugView::getSize` / `on_size` / `can_resize`，以及 REAPER 的
    /// `plugin_register("timer")`）。而 ARA 的播放请求被钉在 **model 线程**上，
    /// 于是投给它的任务永远取不到 —— 现场日志是
    /// `host thread did not pick the request up within 250 ms`，且括号里的
    /// 待办数一路增长。
    ///
    /// 【为什么是这四个入口】它们覆盖了宿主每一次模型交互的入口：
    /// `take_forced_edit_close` 由 vendored 的 `ControllerAdapter::service_forced_settle`
    /// 在**每一次** `runtime_mut()` 里调用（创建/更新/删除片段、编辑会话、
    /// `notifyModelUpdates` 都会经过），另外三个是编辑会话与模型刷新的直接入口。
    /// 四个互为冗余：任何一个先到，队列都会被排空。
    ///
    /// 【为什么这里只允许排播放请求】此刻我们正被宿主调用，栈上持着
    /// `&mut ControllerAdapter`。往队列里放任意闭包，等于允许在那一刻重入模型 ——
    /// 那是未定义行为。类型上的约束见 `editor::host_threads::PlaybackRequest`。
    fn service_model_thread(&self) {
        crate::host::marshal::pump();
    }
}

impl DocumentLifecycle for ModelHandle {
    type Document = ();

    fn create_document(
        &mut self,
        context: &CreateContext,
        properties: DocumentProperties,
    ) -> Result<Self::Document, AraError> {
        self.generation = Some(context.generation());
        *self.session.generation.lock_or_recover() = self.generation;
        self.document.document_name = properties.name().unwrap_or("").to_string();
        log::info!(
            "[ara] document controller created: apiGeneration={:?}",
            context.generation()
        );
        Ok(())
    }

    /// 记录宿主编辑会话，区分宿主未通知与委托回调未处理。
    ///
    /// 【为什么这里**不再** `clear_renderers()`】`clear_renderers()` 会把 `ready` 按回
    /// `false`，而 `ready` 参与 `ensure_loaded` 的"宿主是否冲突"判据。一次编辑会话里
    /// 宿主会连续回调，会话**一开始**就无条件失效意味着整段会话都处在"冲突"读数里
    /// （用户报障：每次编辑都闪"尚未应用：Conflict: host changed"）。
    /// 真正需要失效的是**内容变化**：`update_playback_region`（几何真变时）、
    /// `destroy/create_playback_region`、`update_audio_source_content` 各自负责。
    /// 会话本身只需要被**记录**下来，供会话侧把"会话中的未就绪窗口"识别为暂缓。
    fn begin_editing(&mut self, _document: &mut Self::Document) -> Result<(), AraError> {
        self.service_model_thread();
        self.editing = true;
        // 【G5】把"编辑会话中"这个事实推给文档：存档载荷到达时据此判断它是
        // "一次撤销/重做"（必须覆盖）还是"宿主交回一份旧载荷"（必须补缺）。
        self.session.set_in_edit_session(true);
        // 【T23】连同**开始时刻**一起记：`editing_session` 只回答"在不在会话里"，
        // 而"在会话里多久了"才是插件侧判定"宿主不会再来收口"的唯一依据。
        self.session.note_editing_started();
        log::info!("[ara] begin_editing");
        Ok(())
    }

    fn end_editing(
        &mut self,
        _document: &mut Self::Document,
        _host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        self.service_model_thread();
        self.editing = false;
        self.session.set_in_edit_session(false);
        self.session.note_editing_finished();
        self.remap_and_log();
        Ok(())
    }

    fn destroy_document(&mut self, _document: Self::Document) {
        self.session.close();
        region_owners()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove_document(self.document_id);
        log::info!("[ara] document destroyed");
    }

    fn flush_pending_model_updates(&mut self) -> bool {
        self.service_model_thread();
        if !self.model_dirty {
            return false;
        }
        log::info!("[ara] flushing pending model changes without endEditing");
        self.remap_and_log();
        true
    }

    /// 【T23】取走插件侧置位的"强制收口编辑会话"请求。
    ///
    /// 【为什么读的是文档原子量】置位者可能是视图定时器或音频线程，而本方法在 ARA
    /// 模型线程上跑 —— 两者之间唯一的合法通道就是一个原子量。`swap` 保证同一次请求
    /// 只被消费一次（否则每次回调都会重跑一次全量重映射）。
    fn take_forced_edit_close(&mut self) -> bool {
        self.service_model_thread();
        self.session
            .forced_edit_close
            .swap(false, std::sync::atomic::Ordering::AcqRel)
    }

    /// 【T23】取走"把积压模型变化 flush 一次"的请求。
    fn take_forced_model_flush(&mut self) -> bool {
        self.session
            .forced_model_flush
            .swap(false, std::sync::atomic::Ordering::AcqRel)
    }
}

impl MusicalContexts for ModelHandle {
    type MusicalContext = ();

    fn create_musical_context(
        &mut self,
        _context: &CreateContext,
        properties: MusicalContextProperties,
        host: &HostContentScope<'_, '_>,
    ) -> Result<Self::MusicalContext, AraError> {
        self.mark_model_dirty();
        self.document.musical_contexts.push(AraMusicalContext {
            name: properties.name().map(str::to_owned),
            region_sequences: Vec::new(),
        });
        // 【为什么要在这里真读】宿主经 ARA 给出的 tempo map / 拍号是音乐上下文的权威
        // 内容，而 `createMusicalContext` 是它唯一可靠的到达点（有些宿主不会为内容的
        // 变化单独发 `updateMusicalContextContent`）。读不到时保持 `None`，
        // 投影退回 VST3 的单点 tempo。
        self.session
            .set_host_tempo_map(read_host_musical_content(host));
        Ok(())
    }

    fn update_musical_context(
        &mut self,
        _state: &mut Self::MusicalContext,
        properties: MusicalContextProperties,
        _host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        // 【为什么必须实现】此前落 trait 默认空实现。`MusicalContext = ()`，没有 per-context
        // 状态可改，但**名字**是文档级的展示信息，宿主改名时插件应当跟上。
        // 名字落在第一个 context 上（与 `create_region_sequence` 的归属口径一致）。
        self.mark_model_dirty();
        if let Some(context) = self.document.musical_contexts.first_mut() {
            context.name = properties.name().map(str::to_owned);
        }
        Ok(())
    }

    fn update_musical_context_content(
        &mut self,
        _state: &mut Self::MusicalContext,
        range: Option<ContentTimeRange>,
        flags: ContentUpdateScopes,
        host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        // 【为什么必须实现】宿主**改工程速度**走的就是这个回调。此前落默认空实现，
        // 于是变速后插件既不知道 tempo map 变了、也不会重算已发布 region 的拉伸比
        // （`playback_rate` 只在宿主重推 region 几何时才会更新）。
        //
        // 这里重读 tempo map：投影缓存以它为键，重读即失效，下一次投影就会用新速度
        // 重建 tempo 点与拍号网格。同时 `playback_rate` 的更正依赖宿主重推 region
        // 几何 —— 那一条由 `update_playback_region` 负责，与这里互补。
        log::info!(
            "[ara] musical_context content updated: range={:?} scopes={:?}",
            range,
            flags
        );
        self.mark_model_dirty();
        self.session
            .set_host_tempo_map(read_host_musical_content(host));
        self.sync_session_derived_tables();
        Ok(())
    }

    fn destroy_musical_context(&mut self, _state: Self::MusicalContext) {
        self.mark_model_dirty();
        // 【为什么必须实现】此前落 trait 默认空实现。宿主删除 musical context 时，
        // 它下面的 region sequence 一并消失；不清的话这些序列的身份与参数根边
        // 会一直留到文档关闭。
        //
        // 只处理"最后一个 context"：`create_region_sequence` 也把新序列归到最后一个
        // context，两者对称。多个 context 并存时宿主极少删除中间的，那种情形由
        // 上面的序列级销毁各自收口。
        if self.document.musical_contexts.len() > 1 {
            self.document.musical_contexts.pop();
        }
        log::info!("[ara] musical_context destroyed");
        self.sync_session_derived_tables();
    }
}

/// F-2 取证探针：宿主到底提不提供速度 / 拍号 / 调号内容？
///
/// 【为什么要先取证再动手】HiFiShifter 的渲染锚定在音乐上下文上，而 ARA 的
/// `Tempo` / `BarSignatures` / `KeySignatures` 内容**不是**"规范里有就一定有"：
/// REAPER 是否实现、实现到什么程度，只能实测。猜错的方向会一路渗进渲染缓存键。
///
/// 【为什么只记"有没有、几条"】`HostContentScope` 是 `!Send`，只在本次回调内有效，
/// 而且读事件要按类型逐个解包 —— 先拿到"宿主提供 / 不提供"这一条事实就够决定
/// 方案形状；真正消费（微分求 BPM、合并拍号）等这一条事实出来再做。
/// 宿主经 ARA 音乐上下文给出的**真实** tempo map 与拍号。
///
/// 【为什么必须真读】此前 `probe_host_musical_content` 只把条数记进日志，于是插件只用
/// VST3 `ProcessContext` 的**单点** tempo —— 带 tempo map 的工程里，小节/拍网格与
/// musical-time 投影只有 playhead 处是对的。而 `Tempo` / `BarSignatures` 是 ARA 2.0 的
/// 一等内容类型（`ARAContentTempoEntry` / `ARAContentBarSignature`），拿得到就该用。
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct HostTempoMap {
    /// `(位置秒, BPM)`，按位置升序。
    pub tempo: Vec<(f64, f64)>,
    /// `(位置秒, 分子, 分母)`，按位置升序。
    pub meter: Vec<(f64, u32, u32)>,
}

impl HostTempoMap {
    /// 指定时刻的 BPM（取不晚于该时刻的最后一个点；早于第一个点则取第一个）。
    pub(crate) fn bpm_at(&self, position_sec: f64) -> Option<f64> {
        self.tempo
            .iter()
            .rev()
            .find(|(at, _)| *at <= position_sec + 1e-9)
            .or_else(|| self.tempo.first())
            .map(|(_, bpm)| *bpm)
    }

    /// 指定时刻的拍号（同上口径）。
    pub(crate) fn meter_at(&self, position_sec: f64) -> Option<(u32, u32)> {
        self.meter
            .iter()
            .rev()
            .find(|(at, _, _)| *at <= position_sec + 1e-9)
            .or_else(|| self.meter.first())
            .map(|(_, numerator, denominator)| (*numerator, *denominator))
    }

    /// 是否含任何可用读数。
    pub(crate) fn is_empty(&self) -> bool {
        self.tempo.is_empty() && self.meter.is_empty()
    }
}

/// 把"四分音符位置"换算成秒。
///
/// 【为什么要换算】`ARAContentTempoEntry` 同时给秒与四分音符位置，而
/// `ARAContentBarSignature` **只给四分音符位置**。拍号点要落到时间线上，就必须先用
/// tempo 分段把它折成秒。
///
/// 【为什么分段线性】tempo map 的语义就是"每两个相邻 tempo 点之间速度恒定"，
/// 所以段内 `秒 = 段起点秒 + (q − 段起点q) × 60 / bpm`。
fn quarters_to_seconds(tempo: &[(f64, f64, f64)], quarter: f64) -> Option<f64> {
    // `tempo` 是 `(秒, 四分音符, bpm)`，按四分音符升序。
    let first = tempo.first()?;
    if quarter <= first.1 {
        return Some(first.0);
    }
    for pair in tempo.windows(2) {
        let (sec0, q0, bpm) = pair[0];
        let (sec1, q1, _) = pair[1];
        if quarter <= q1 {
            if bpm <= 0.0 {
                return Some(sec0);
            }
            return Some(sec0 + (quarter - q0) * 60.0 / bpm);
        }
        let _ = sec1;
    }
    // 落在最后一个点之后：按最后一段的速度外推。
    let (sec, q, bpm) = *tempo.last()?;
    if bpm <= 0.0 {
        return Some(sec);
    }
    Some(sec + (quarter - q) * 60.0 / bpm)
}

/// 读取宿主的 tempo map 与拍号；读不到就返回 `None`（调用方退回单点 tempo）。
///
/// 【为什么保留 grade 日志】"宿主提不提供内容"是宿主差异的第一手证据，而读取本身
/// 可能失败。两者分开记，下一次排障时一眼能看出是"宿主不给"还是"我们没读"。
fn read_host_musical_content(host: &HostContentScope<'_, '_>) -> Option<HostTempoMap> {
    let Some(context) = host.current_musical_context() else {
        log::info!("[ara] musical content: no current context in this scope");
        return None;
    };
    for (name, grade) in [
        ("tempo", host.musical_context_grade::<Tempo>(context)),
        (
            "bar_signatures",
            host.musical_context_grade::<BarSignatures>(context),
        ),
        (
            "key_signatures",
            host.musical_context_grade::<KeySignatures>(context),
        ),
    ] {
        match grade {
            Ok(grade) => log::info!("[ara] musical content {name}: grade={grade:?}"),
            Err(error) => log::info!("[ara] musical content {name}: unavailable ({error})"),
        }
    }

    // tempo 点：`(秒, 四分音符, bpm)`。BPM 由相邻两点推出（段内速度恒定）。
    let mut raw_tempo: Vec<(f64, f64, f64)> = Vec::new();
    match host.musical_context::<Tempo>(context, None) {
        Ok(mut reader) => {
            for index in 0..reader.len() {
                let Ok(event) = reader.event(index) else {
                    continue;
                };
                raw_tempo.push((event.time_position(), event.quarter_position(), 0.0));
            }
        }
        Err(error) => log::info!("[ara] tempo entries unavailable: {error}"),
    }
    raw_tempo.sort_by(|a, b| a.1.total_cmp(&b.1));
    raw_tempo.dedup_by(|a, b| (a.1 - b.1).abs() < 1e-9);
    for index in 0..raw_tempo.len() {
        let bpm = if index + 1 < raw_tempo.len() {
            let (sec0, q0, _) = raw_tempo[index];
            let (sec1, q1, _) = raw_tempo[index + 1];
            let dq = q1 - q0;
            let dt = sec1 - sec0;
            if dq > 1e-9 && dt > 1e-9 {
                60.0 * dq / dt
            } else {
                0.0
            }
        } else {
            0.0
        };
        raw_tempo[index].2 = bpm;
    }
    // 最后一段没有后继点：沿用前一段的速度（tempo map 的常规外推）。
    if raw_tempo.last().is_some_and(|(_, _, bpm)| *bpm <= 0.0) {
        // 先取前一段的速度，再写最后一个点 —— 同时借可变与不可变会借用冲突。
        let previous = raw_tempo
            .iter()
            .rev()
            .find(|(_, _, bpm)| *bpm > 0.0)
            .map(|(_, _, bpm)| *bpm)
            .unwrap_or(0.0);
        if let Some(last) = raw_tempo.last_mut() {
            last.2 = previous;
        }
    }
    let tempo: Vec<(f64, f64)> = raw_tempo
        .iter()
        .filter(|(_, _, bpm)| *bpm > 0.0)
        .map(|(sec, _, bpm)| (*sec, *bpm))
        .collect();

    // 拍号点只给四分音符位置：先折成秒再落到时间线上。
    let mut meter: Vec<(f64, u32, u32)> = Vec::new();
    match host.musical_context::<BarSignatures>(context, None) {
        Ok(mut reader) => {
            for index in 0..reader.len() {
                let Ok(event) = reader.event(index) else {
                    continue;
                };
                let Some(sec) = quarters_to_seconds(&raw_tempo, event.position()) else {
                    continue;
                };
                let numerator = u32::try_from(event.numerator()).unwrap_or(0);
                let denominator = u32::try_from(event.denominator()).unwrap_or(0);
                if numerator == 0 || denominator == 0 {
                    continue;
                }
                meter.push((sec, numerator, denominator));
            }
        }
        Err(error) => log::info!("[ara] bar signatures unavailable: {error}"),
    }
    meter.sort_by(|a, b| a.0.total_cmp(&b.0));

    let map = HostTempoMap { tempo, meter };
    if map.is_empty() {
        log::info!("[ara] musical content: the host supplied no tempo or bar-signature entries");
        None
    } else {
        log::info!(
            "[ara] musical content: {} tempo point(s), {} bar signature(s)",
            map.tempo.len(),
            map.meter.len()
        );
        Some(map)
    }
}

impl RegionSequences for ModelHandle {
    type RegionSequence = usize;

    fn create_region_sequence(
        &mut self,
        context: &CreateContext,
        properties: RegionSequenceProperties,
    ) -> Result<Self::RegionSequence, AraError> {
        self.mark_model_dirty();
        // 归到第一个（通常也是唯一一个）musical context 下；没有就现建一个。
        if self.document.musical_contexts.is_empty() {
            self.document.musical_contexts.push(AraMusicalContext {
                name: None,
                region_sequences: Vec::new(),
            });
        }
        // 【T7：优先复用已销毁的槽位】`destroy_region_sequence` 只抹掉槽位的身份，
        // 不删除占位条目（删除会让后续序列前移、身份整体错位）。如果这里继续
        // "永远追加"，宿主每做一次"销毁 + 重建"就会让扁平向量增长一格，而
        // `sequence_refs` / `sequence_persistent_ids` / `live_sequences` 全部跟着长。
        // 复用一个已销毁槽位与 `allocate_region` 复用已销毁 region 槽位是同一手法：
        // 下标稳定 ⇒ `region_sequence_index` 与 `sequence_model_ref` 的映射天然稳定。
        let reused = self
            .document
            .musical_contexts
            .iter()
            .flat_map(|context| context.region_sequences.iter())
            .enumerate()
            .find(|(slot, _)| self.destroyed_sequences.contains(slot))
            .map(|(slot, _)| slot);
        let index = match reused {
            Some(index) => {
                // 就地覆盖占位条目：`destroy_region_sequence` 保留了它的位置。
                if let Some(slot) = self
                    .document
                    .musical_contexts
                    .iter_mut()
                    .flat_map(|context| context.region_sequences.iter_mut())
                    .nth(index)
                {
                    slot.name = properties.name().map(str::to_owned);
                    slot.order_index = properties.order_index();
                    slot.color = host_color_hex(properties.color());
                    slot.playback_region_count = 0;
                }
                self.destroyed_sequences.remove(&index);
                index
            }
            None => {
                let index = self
                    .document
                    .musical_contexts
                    .iter()
                    .map(|context| context.region_sequences.len())
                    .sum();
                let context_index = self.document.musical_contexts.len() - 1;
                self.document.musical_contexts[context_index]
                    .region_sequences
                    .push(AraRegionSequence {
                        name: properties.name().map(str::to_owned),
                        order_index: properties.order_index(),
                        color: host_color_hex(properties.color()),
                        playback_region_count: 0,
                    });
                index
            }
        };
        // 【序列身份取自 realtime key，不是 persistentID】`ARARegionSequenceProperties`
        // 只有 name / orderIndex / musicalContextRef / color（`ARAInterface.h:775-793`），
        // **没有** persistentID（audio source / modification / processing algorithm 才有）。
        // 能拿到的稳定令牌就是宿主的 model-ref key —— 同一文档内它不随重排而变。
        // 拿不到 key 时记空串，映射层随即退回位置身份。
        let sequence_key = context.realtime_key().unwrap_or(0);
        // 【T10：这里是**覆盖**一个槽位，不是插入一个新槽位】
        //
        // `reused` 分支保证 `index < len`（被销毁的槽位仍占着位置），而 `Vec::insert`
        // 在 `index < len` 时是"插入并右移"：两张表各长一格，`index` 之后的所有序列
        // 身份整体错位一位。后果按严重度排列：
        //   * 第二条真实序列的 id 变成空串 ⇒ 轨道 id 退化成 `ara-track-{slot}` ⇒
        //     用户看到一条"新轨"，参数/名字/颜色/顺序全丢；
        //   * `region_sequence_of` 按 ref 解析时命中被右移过的槽位 ⇒ 同一批 clip
        //     分裂到两条轨（"分割后多出一条轨"）；
        //   * 每 split 一次两张表各长一格，长会话里是无界增长。
        //
        // 覆盖语义只能用 `assign`。三张表（`sequence_persistent_ids` /
        // `sequence_refs` / `region_sequences`）必须**严格同长、同下标**，
        // 否则 `MappingIdentities::live_sequences`（按第一张表的长度生成）会错位。
        // 【G3】优先认领一条刚被销毁的序列的身份：宿主重建序列时 `realtime_key`
        // 必然改变，而轨道 id 是从它派生的 —— 不认领就等于"整条轨换了一个人"，
        // 它的参数会被判成"查不到宿主身份"。
        let identity = self
            .claim_sequence_death(properties.order_index(), properties.name())
            .unwrap_or_else(|| sequence_key.to_string());
        Self::set_sequence_slot(&mut self.sequence_persistent_ids, index, identity);
        // 与扁平序列槽位一一对应：region 的 `sequence_model_ref` 靠它翻译回槽位。
        Self::set_sequence_slot(&mut self.document.sequence_refs, index, sequence_key);
        if let Some(handle) = context.object_handle() {
            self.sequence_index_by_handle.insert(handle, index);
        }
        if let Some(key) = context.realtime_key() {
            self.sequence_index_by_model_ref.insert(key as usize, index);
            self.session
                .sequence_regions
                .lock_or_recover()
                .insert(key, HashSet::new());
            // 记录 sequence → 轨道下标：空 sequence（无 region）也要能被认成参数根。
            self.session
                .sequence_track_index
                .lock_or_recover()
                .insert(key, index);
        }
        Ok(index)
    }

    fn update_region_sequence(
        &mut self,
        state: &mut Self::RegionSequence,
        properties: RegionSequenceProperties,
    ) -> Result<(), AraError> {
        // 【为什么必须实现】此前落 trait 默认空实现：宿主重命名轨道、或改变序列顺序时
        // 插件毫无反应，而 `order_index` 被读入后**从未使用**（映射层按枚举顺序排 track），
        // 于是"宿主把轨往下挪"在插件里看起来什么都没发生。
        self.mark_model_dirty();
        let sequences = self
            .document
            .musical_contexts
            .iter_mut()
            .flat_map(|context| context.region_sequences.iter_mut())
            .nth(*state);
        let Some(sequence) = sequences else {
            return Err(AraError::InvalidState(
                "region sequence state is not in the document",
            ));
        };
        sequence.name = properties.name().map(str::to_owned);
        sequence.order_index = properties.order_index();
        sequence.color = host_color_hex(properties.color());
        log::info!(
            "[ara] region_sequence #{state} updated: name={:?} orderIndex={}",
            sequence.name,
            sequence.order_index
        );
        self.sync_session_derived_tables();
        Ok(())
    }

    fn destroy_region_sequence(&mut self, state: Self::RegionSequence) {
        self.mark_model_dirty();
        // 【为什么必须实现】此前落 trait 默认空实现，而 `region_sequences` 是 push-only 的：
        // 宿主删除一条轨道后，它的 `ara-track-N` 身份与 `params_by_root_track` 条目
        // 一直留到文档关闭 —— 用户看到插件里有一条已经不存在的轨，且它的参数还在参与渲染。
        //
        // 同 `destroyed_sequences` 口径：不 compact 下标，只把该槽的宿主身份抹掉。
        // 映射层看到空身份就退回位置身份，而 `sequence_track_index` 由下方清除，
        // 于是这条序列不再被认成参数根。
        self.destroyed_sequences.insert(state);
        // 【G3】先把它记成"待认领"，再抹空。顺序不能反：抹空之后就没有身份可认领了。
        {
            let key = self
                .sequence_persistent_ids
                .get(state)
                .cloned()
                .unwrap_or_default();
            if !key.is_empty() {
                let (order, name) = self
                    .document
                    .musical_contexts
                    .iter()
                    .flat_map(|context| context.region_sequences.iter())
                    .nth(state)
                    .map(|sequence| (sequence.order_index, sequence.name.clone()))
                    .unwrap_or((0, None));
                self.remember_sequence_death(order, name, key);
            }
        }
        if let Some(id) = self.sequence_persistent_ids.get_mut(state) {
            id.clear();
        }
        // 【T7：把宿主身份从 ref 表与反向表里一起抹掉】只清 `sequence_persistent_ids`
        // 不够 —— `region_sequence_of` **首选**按 `sequence_model_ref` 解析
        // （`mapping.rs`），旧 ref 若还留在表里，存活 region 会继续解析到这个语义
        // 已变的槽位，于是同一批 clip 被分到两条轨（用户报的"分割后多出一条轨"）。
        // 清零而不是删除：`sequence_refs` 的下标必须与 `sequence_persistent_ids` /
        // `live_sequences` 对齐，删除会让后续序列前移、身份整体错位。
        if let Some(reference) = self.document.sequence_refs.get_mut(state) {
            let old = *reference;
            *reference = 0;
            if old != 0 {
                self.sequence_index_by_model_ref.remove(&(old as usize));
            }
        }
        self.sequence_index_by_handle
            .retain(|_, index| *index != state);
        if let Some(sequences) = self
            .document
            .musical_contexts
            .iter_mut()
            .flat_map(|context| context.region_sequences.iter_mut())
            .nth(state)
        {
            sequences.name = None;
            sequences.playback_region_count = 0;
        }
        // 清掉"被分配了 sequence 却没有 clip"的参数根边，否则已删除的轨仍会留下。
        self.session
            .sequence_regions
            .lock_or_recover()
            .retain(|key, _| {
                self.session
                    .sequence_track_index
                    .lock_or_recover()
                    .get(key)
                    .copied()
                    != Some(state)
            });
        self.session
            .sequence_track_index
            .lock_or_recover()
            .retain(|_, index| *index != state);
        log::info!("[ara] region_sequence destroyed #{state}");
        self.sync_session_derived_tables();
    }
}

impl AudioSources for ModelHandle {
    type AudioSource = usize;

    /// 源几何变化必须撤销旧 PCM；不把新的采样率/长度继续与旧内容混用。
    fn update_audio_source(
        &mut self,
        state: &mut Self::AudioSource,
        properties: AudioSourceProperties,
        host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        self.mark_model_dirty();
        let source = self
            .document
            .audio_sources
            .get_mut(*state)
            .ok_or(AraError::InvalidArgument("unknown source"))?;
        self.session.clear_renderers();
        self.session
            .sources
            .lock_or_recover()
            .remove(&source.persistent_id);
        self.session
            .edit_sources
            .lock_or_recover()
            .remove(&source.persistent_id);
        source.persistent_id = properties.persistent_id().to_string();
        source.name = properties.name().map(str::to_owned);
        source.sample_rate = properties.sample_rate();
        source.sample_count = properties.sample_count();
        source.channel_count = properties.channel_count();
        source.duration_seconds = source.sample_count as f64 / source.sample_rate;
        self.source_content_version_bump(*state);
        self.refresh_source_pcm(*state, host);
        Ok(())
    }

    /// undo history 中停用源时立即撤销准备好的 PCM；恢复后重新读取。
    fn deactivate_audio_source(
        &mut self,
        state: &mut Self::AudioSource,
        deactivate: bool,
        host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        self.mark_model_dirty();
        if let Some(source) = self.document.audio_sources.get_mut(*state) {
            source.deactivated = deactivate;
        }
        if deactivate {
            self.session.clear_renderers();
            if let Some(source) = self.document.audio_sources.get(*state) {
                self.session
                    .edit_sources
                    .lock_or_recover()
                    .remove(&source.persistent_id);
            }
        }
        // 【为什么两个方向都要刷新】旧实现只在 `deactivate == true` 时刷新，重新
        // 激活（`false`）什么都不做：源永远停在"已停用 + 无 PCM"的状态，界面显示
        // "正在等待宿主完成音频分配"。重新激活必须重读 PCM 并重排快照。
        self.refresh_source_pcm(*state, host);
        Ok(())
    }

    /// 源销毁时撤销访问和快照，不能继续播已销毁源的缓存。
    fn destroy_audio_source(&mut self, state: Self::AudioSource, host: &HostContentScope<'_, '_>) {
        self.mark_model_dirty();
        self.session.clear_renderers();
        if let Some(source) = self.document.audio_sources.get(state) {
            self.session
                .edit_sources
                .lock_or_recover()
                .remove(&source.persistent_id);
        }
        // 【T7】源被**销毁**是"已发布 PCM 必须消失"的三种情形之一：这里连同状态一起清掉。
        if let Some(source) = self.document.audio_sources.get(state) {
            self.session
                .source_states
                .lock_or_recover()
                .remove(&source.persistent_id);
        }
        if let Some(source) = self.document.audio_sources.get_mut(state) {
            source.sample_access_enabled = false;
        }
        self.refresh_source_pcm(state, host);
    }

    fn retry_failed_source_read(
        &mut self,
        state: &mut Self::AudioSource,
        host: &HostContentScope<'_, '_>,
    ) -> bool {
        let Some(source) = self.document.audio_sources.get(*state) else {
            return false;
        };
        let id = source.persistent_id.clone();
        let readable = source.sample_access_enabled && !source.deactivated;
        if !readable {
            return false;
        }
        // 【T9：重试不再只服务"读失败"】宿主收回一次样本访问之后，源会停在
        // `Revoked`（PCM 仍然保留），而恢复**不能**只押在"宿主记得再发一次 enable"
        // 上 —— 那条回调丢了就再也不会来。这里把"可读但没有 PCM / 上一次读失败"
        // 都纳入重试，由桥在 `notifyModelUpdates` 与样本访问变更时驱动（限流 1 s）。
        let needs_read = {
            let states = self.session.source_states.lock_or_recover();
            states.get(&id) != Some(&crate::render::document::SourceState::Published)
        };
        if !needs_read {
            return false;
        }
        self.refresh_source_pcm(*state, host);
        !self
            .session
            .source_failures
            .lock_or_recover()
            .contains_key(&id)
    }

    fn create_audio_source(
        &mut self,
        context: &CreateContext,
        properties: AudioSourceProperties,
        _host: &HostContentScope<'_, '_>,
    ) -> Result<Self::AudioSource, AraError> {
        self.mark_model_dirty();
        let persistent_id = properties.persistent_id().to_string();
        let sample_rate = properties.sample_rate();
        let sample_count = properties.sample_count();
        let index = self.document.audio_sources.len();
        self.document.audio_sources.push(AraAudioSource {
            persistent_id: persistent_id.clone(),
            name: properties.name().map(str::to_owned),
            sample_rate,
            sample_count,
            // 源时长由采样率与样本数推出：ARA 对象模型不带独立的时长字段。
            duration_seconds: if sample_rate > 0.0 {
                sample_count as f64 / sample_rate
            } else {
                0.0
            },
            channel_count: properties.channel_count(),
            sample_access_enabled: false,
            deactivated: false,
        });
        if let Some(handle) = context.object_handle() {
            self.source_index_by_handle.insert(handle, index);
        }
        log::info!(
            "[ara] audio_source #{index}: persistentID={persistent_id} sampleRate={sample_rate} sampleCount={sample_count}"
        );
        Ok(index)
    }

    fn update_audio_source_content(
        &mut self,
        state: &mut Self::AudioSource,
        _range: Option<ContentTimeRange>,
        _flags: ContentUpdateScopes,
        host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        // 内容变了 → 版本号前进。渲染缓存据此失效（设计 §4.5）。
        self.mark_model_dirty();
        // 【T-A1-4：播放中不撤渲染许可】播放中撤掉快照等于在用户的耳朵里挖一个洞；
        // 旧快照对应的是上一版内容（与 `StaleProcessed` 同一条纪律），比一次静音好。
        // 停播后第一次 prepare 会照常清，所以"内容过期"不会被永久忽略。
        if !self.session.realtime_playing() {
            self.session.clear_renderers();
        }
        // 【T12：读到新版本之后再替换】旧实现在这里**先删**已发布的编辑侧副本，再尝试重读。
        // 那一刻若样本访问还没开（后台、undo 历史、宿主整理缓存），新版本读不出来，
        // 而旧版本已经删了 ⇒ 编辑侧 PCM 永久消失，界面只能显示占位（用户看到的
        // "回前台后 Clip 一直等音频"）。现在只推进版本号，替换交给
        // `refresh_source_pcm` 在**读到新版本之后**做。
        self.source_content_version_bump(*state);
        self.refresh_source_pcm(*state, host);
        Ok(())
    }

    fn enable_audio_source_samples_access(
        &mut self,
        state: &mut Self::AudioSource,
        enable: bool,
        host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        if let Some(source) = self.document.audio_sources.get_mut(*state) {
            source.sample_access_enabled = enable;
        }
        // 【T-B0-1】账本里的 `access=0/0` 此前没有写者：它恒为 0/0，读起来像
        // "宿主一次都没授权过"。补上之后，"授权了几次、收回了几次"第一次可读。
        self.ledger.note_enable_access(enable);
        if enable {
            self.session
                .samples_access_granted
                .store(true, std::sync::atomic::Ordering::Release);
        }
        log::info!("[ara] audio_source samples_access enable={enable}");
        self.refresh_source_pcm(*state, host);
        if enable && std::env::var_os("HIFISHIFTER_PLUGIN_PCM_PROBE").is_some() {
            // 仅用于一次性宿主观测：比较原始源样本与 ARA 返回的源样本，判断倒放是否改源。
            let result = (|| {
                let source = self
                    .document
                    .audio_sources
                    .get(*state)
                    .ok_or(AraError::InvalidArgument("unknown source"))?;
                let source_ref = host
                    .current_audio_source()
                    .ok_or(AraError::InvalidState("missing source scope"))?;
                let channels = source.channel_count.max(1) as usize;
                let mut reader = host.audio_reader::<f32>(source_ref, channels)?;
                let mut samples = vec![vec![0.0_f32; 16]; channels];
                let mut planes = samples
                    .iter_mut()
                    .map(Vec::as_mut_slice)
                    .collect::<Vec<_>>();
                reader.read(0, &mut planes)?;
                log::info!("[ara] source PCM #{}: first16={:?}", state, samples[0]);
                Ok::<(), AraError>(())
            })();
            if let Err(error) = result {
                log::warn!("[ara] source PCM probe failed: {error:?}");
            }
        }
        Ok(())
    }
}

impl ModelHandle {
    /// 授权刷新撤销旧快照；内容/几何回调另行推进模型版本，仅从授权scope读取。
    ///
    /// 【T7：已发布的 PCM 只在三种情况下消失】源被销毁、内容版本换了且新版本就绪、
    /// 文档关闭。其余一切"暂时不可读"（样本访问被收回、源被放进 undo 历史、后台、
    /// 读取失败）都**保留**上一份，只改状态。
    ///
    /// 【为什么这条是"后台切回来之后 Clip 一直等宿主音频"的根因】旧实现在最开头
    /// 无条件 `sources.remove(...)`：宿主收回一次访问就等于永久丢掉音频，而恢复
    /// 只依赖"宿主记得再发一次 enable"——那条回调丢了就再也不会来。用户唯一的
    /// 出路是删掉插件再加。
    ///
    /// 【为什么不会让旧音频盖住新内容】内容变化时 `source_content_version` 会前进，
    /// 而 `SourcePcm` 带 `version`，渲染缓存键含它。保留旧 PCM 的那一窗口里渲染读到
    /// 的是**上一版内容**——与 Capture 的 `StaleProcessed` 是同一条既有纪律。
    fn refresh_source_pcm(&mut self, index: usize, host: &HostContentScope<'_, '_>) {
        // 【T-A1-4：这里**不能**无条件撤渲染权威】本函数被宿主每一次内容/访问变化调用，
        // 也被我们自己的 1 s 重试调用。旧实现在最开头 `revoke_renderers()`，于是
        // "一次重试"就硬清全部快照 ⇒ 每个重试周期一次静音。真正需要撤的只有一件事：
        // **我们已经有的那份 PCM 现在读不到**（那条分支里才撤）。
        let Some(source) = self.document.audio_sources.get(index) else {
            return;
        };
        let id = source.persistent_id.clone();
        let readable = source.sample_access_enabled && !source.deactivated;
        if !readable {
            // 【T7】渲染权威表必须撤掉（宿主收回了访问，不能继续播旧采样），但
            // **编辑侧的物化副本保留** —— 那是界面波形的来源，丢了就等于"整条轨
            // 突然空了"。这一对区分正是"暂时不可读"与"从来没有过"的落点。
            self.session.sources.lock_or_recover().remove(&id);
            // 【B5：从未授权 ≠ 被收回】
            //
            // 旧实现在这里**无条件**写 `Revoked`，于是"宿主还没给过访问权"与"宿主
            // 刚刚收回了访问权"变成同一件事。而 `decorate_host_media_locked` 把
            // `Revoked` 报成 `unavailable:source_revoked` —— 一句**终态**断言。
            // 重开工程时宿主通常先建源、后授权：那一小段时间里**每一个** clip 都会被
            // 写成"源被收回"，而正确的说法是"还没有访问权"。
            //
            // 判据取"我们手里有没有这份 PCM"：有 ⇒ 确实是被收回（保留 `Revoked`，
            // 界面说"暂时收回"）；没有 ⇒ 什么都不写（缺席即"从未授权"）。
            let had_pcm = {
                // 【锁序】`edit_sources` 先于 `source_states`（与 `publish_readiness` 同向）。
                self.session
                    .edit_sources
                    .lock_or_recover()
                    .contains_key(&id)
            };
            if had_pcm {
                // 宿主收回了访问：继续播旧采样等于播一份已经失效的结果，这一刻必须撤。
                // 与"一次重试"不同 —— 这是**真的**失去了输入。
                self.session.revoke_renderers();
            }
            let mut states = self.session.source_states.lock_or_recover();
            if had_pcm {
                states.insert(id, crate::render::document::SourceState::Revoked);
            } else {
                states.remove(&id);
            }
            drop(states);
            self.publish_readiness();
            if !self.editing {
                self.session.prepare_renderers();
            }
            return;
        }
        let result = usize::try_from(source.sample_count)
            .ok()
            .filter(|_| source.sample_rate == source.sample_rate.round())
            .ok_or(AraError::InvalidArgument("unsupported sample geometry"))
            .and_then(|frames| {
                crate::render::source::read_source_pcm(
                    host,
                    frames,
                    // 【为什么 `.max(1)`】宿主可能报 0 声道（畸形或未初始化）。
                    // `read_source_pcm` 会拒它，但拒的措辞是"通道数不在 1..=2" ——
                    // 那是**素材几何**问题，而 0 声道的真实含义是"未知"，按单声道
                    // 尝试读更接近宿主意图。对照 `create_audio_source` 里同一处
                    // 已经这么做了（`source.channel_count.max(1)`）。
                    source.channel_count.max(1) as usize,
                    source.sample_rate as u32,
                    self.source_content_version(index),
                )
            });
        match result {
            Ok(pcm) => {
                log::info!(
                    "[ara] host PCM ready source={index} frames={} version={}",
                    source.sample_count,
                    pcm.version
                );
                let pcm = Arc::new(pcm);
                self.session.source_failures.lock_or_recover().remove(&id);
                self.session
                    .source_states
                    .lock_or_recover()
                    .insert(id.clone(), crate::render::document::SourceState::Published);
                self.session.publish_source_pcm(id, pcm);
            }
            Err(error) => {
                // 【为什么单记一份真因】此前只 `log::warn`，于是"授权了却读不出来"
                // 在界面上与"还没认领"长得一模一样（都报 `awaiting_region`）——
                // 用户对着一个永远转不完的占位，无从判断该做什么。记下真因后
                // `decorate_host_media_locked` 把它报成 `source_read_failed`。
                log::warn!("[ara] host PCM unavailable: {error:?}");
                // 【T7】读取失败同样**保留**上一份：宿主样本侧的一次抖动不该让整条轨静音。
                self.session
                    .source_states
                    .lock_or_recover()
                    .insert(id.clone(), crate::render::document::SourceState::Failed);
                self.session
                    .source_failures
                    .lock_or_recover()
                    .insert(id, format!("{error:?}"));
            }
        }
        self.publish_readiness();
        if !self.editing {
            self.session.prepare_renderers();
        }
    }

    /// 把"输入到底可用到什么程度"算成一份读数并发给文档（T2）。
    ///
    /// 【为什么由模型侧算】`audio_sources` / `source_states` / `edit_sources` 三者
    /// 只有模型这一侧能同时看到；文档侧要判这件事就只能反向引用模型。
    ///
    /// 【为什么"有 region 但 0 个源"要单独成一档】那正是重开工程最典型的形状：
    /// 宿主给了 region、却没交付任何音频源。把它与"源有了但还没读出来"分开之后，
    /// 界面第一次能说出"宿主没有交付音频"，而不是永远一句"等待中"。
    fn publish_readiness(&self) {
        let regions = self.session.regions.lock_or_recover().len();
        let with_source = self
            .session
            .regions
            .lock_or_recover()
            .values()
            .filter(|region| {
                self.session
                    .edit_sources
                    .lock_or_recover()
                    .contains_key(&region.audio_source_persistent_id)
            })
            .count();
        // 【为什么先数「能出声的 PCM」】见 `InputVerdict::decide` 的说明：
        // `sample_access_enabled` 回答的是"宿主此刻允不允许读 ARA 样本"，
        // 而不是"我们手里有没有能出声的素材"。
        let usable = self.session.edit_sources.lock_or_recover().len();
        let states = self.session.source_states.lock_or_recover();
        let sources = self.document.audio_sources.len();
        let authorized = self
            .document
            .audio_sources
            .iter()
            .filter(|source| source.sample_access_enabled && !source.deactivated)
            .count();
        let published = states
            .values()
            .filter(|state| **state == crate::render::document::SourceState::Published)
            .count();
        let failed = states
            .values()
            .filter(|state| **state == crate::render::document::SourceState::Failed)
            .count();
        drop(states);
        let model_published = self.session.timeline.lock_or_recover().is_some();
        // 【判据收敛到一处】判定在 `InputVerdict::decide`（纯函数、带单测）。
        // `authorized` 仍然照常上报，但**不再单独决定** verdict —— 否则"宿主不授权
        // ARA 样本"会把整个编辑器永久钉在 `awaiting_samples`，而渲染侧其实正拿着
        // `edit_sources` 里的 PCM 出声。
        let verdict =
            crate::render::document::InputVerdict::decide(sources, published, failed, usable);
        self.session
            .note_readiness(crate::render::document::InputReadiness {
                regions,
                regions_with_source: with_source,
                sources,
                sources_authorized: authorized,
                sources_published: published,
                sources_failed: failed,
                model_published,
                verdict,
            });
    }

    /// 内容版本自增。
    fn source_content_version_bump(&mut self, index: usize) {
        // 【为什么单独存一个向量而不是写进 `AraAudioSource`】`AraAudioSource` 是探针期
        // 定下的样本形状，没有"内容版本"字段；而 `sample_count` / `sample_rate` 是映射层
        // 算时长要用的，不能借来记版本（借了就会读出错误的时长）。
        // 版本号最终由映射层折进 `Clip.source_file_fingerprint`（渲染键的既有输入）。
        if self.content_versions.len() <= index {
            self.content_versions.resize(index + 1, 0);
        }
        self.content_versions[index] += 1;
        log::info!(
            "[ara] audio_source content_version #{}: {}",
            index,
            self.content_versions[index]
        );
    }
}

impl ModelHandle {
    /// 某个源当前的宿主内容版本（0 = 从未变更）。
    pub fn source_content_version(&self, index: usize) -> u64 {
        self.content_versions.get(index).copied().unwrap_or(0)
    }
}

impl AudioModifications for ModelHandle {
    type AudioModification = usize;
    /// 与 `AudioSources::AudioSource` 同一个 Rust 类型（本地补丁要求两个 trait 各自声明一次）。
    type AudioSource = usize;

    fn create_audio_modification(
        &mut self,
        context: &CreateContext,
        source: &Self::AudioSource,
        properties: AudioModificationProperties,
    ) -> Result<Self::AudioModification, AraError> {
        self.mark_model_dirty();
        let persistent_id = properties.persistent_id().to_string();
        // 这条边（modification → source）来自本地补丁的 trait 参数：上游把它丢了。
        let audio_source_persistent_id = self
            .document
            .audio_sources
            .get(*source)
            .map(|s| s.persistent_id.clone())
            .unwrap_or_default();
        let index = self.document.audio_modifications.len();
        self.document
            .audio_modifications
            .push(AraAudioModification {
                persistent_id: persistent_id.clone(),
                audio_source_persistent_id,
            });
        // 【为什么留痕】分割时宿主到底走 `createAudioModification` 还是
        // `cloneAudioModification`，决定新半段有没有可继承的父参数。此前两者都无入口
        // 日志，真机上无法回答这个问题 —— 而它是"分割后参数归零"的第一分叉点。
        log::info!(
            "[ara] createAudioModification #{index} persistentID={persistent_id} \
             (no parent: parameters start empty)"
        );
        if let Some(handle) = context.object_handle() {
            self.modification_index_by_handle.insert(handle, index);
            // 通知要按宿主句柄投递，而插件侧状态按 persistentID 索引 —— 这里是唯一
            // 拿得到两者对应关系的时刻（见 `DocumentSession::notify_parameters_changed`）。
            self.session
                .register_modification_handle(&persistent_id, handle);
        }
        Ok(index)
    }

    fn clone_audio_modification(
        &mut self,
        context: &CreateContext,
        source: &Self::AudioModification,
        audio_source: &Self::AudioSource,
        properties: AudioModificationProperties,
    ) -> Result<Self::AudioModification, AraError> {
        self.mark_model_dirty();
        // 【为什么留痕】与 `create_audio_modification` 成对：两者都写一行，真机日志里
        // 才能直接读出"分割时宿主用了哪条原语"，而不是靠推断。
        log::info!(
            "[ara] cloneAudioModification from #{source} -> persistentID={}",
            properties.persistent_id()
        );
        // 克隆出的 modification 基于同一个 source —— 这正是"同一素材被复制成多份"
        // 在 ARA 里的形状，映射层会把它变成同一个 `Clip.source_path` 下的多个 clip。
        let mut cloned = self
            .document
            .audio_modifications
            .get(*source)
            .cloned()
            .unwrap_or_default();
        let parent_persistent_id = cloned.persistent_id.clone();
        cloned.persistent_id = properties.persistent_id().to_string();
        cloned.audio_source_persistent_id = self
            .document
            .audio_sources
            .get(*audio_source)
            .map(|s| s.persistent_id.clone())
            .unwrap_or_else(|| cloned.audio_source_persistent_id.clone());
        let child_persistent_id = cloned.persistent_id.clone();
        let index = self.document.audio_modifications.len();
        self.document.audio_modifications.push(cloned);
        if let Some(handle) = context.object_handle() {
            self.modification_index_by_handle.insert(handle, index);
            self.session
                .register_modification_handle(&child_persistent_id, handle);
        }
        // 【P1：宿主显式给出父】`cloneAudioModification` 是 ARA 对"创建独立变体"的
        // 专用原语（`ARAInterface.h:2653-2656`），因此它**就是**复制/分割的回执 ——
        // 与原生路径的 `SplitMediaItem` 回执同形，且不依赖任何原生扩展。把父的源域
        // 参数整份播给子；此后两者各自独立演化（投影按各自的源窗取子段）。
        //
        // 【为什么键是 modification 级】克隆回调只给出 modification persistentID，
        // 此时还不知道新 region 的 item GUID。因此播到 `mod:` 级；`authority` 的查表
        // 会在 item 级缺席时回退到它，而原生宿主上 item 级一旦被捕获就抢先命中。
        //
        // 【为什么在这里加锁是安全的】本回调由宿主在 `begin_editing`…`end_editing`
        // 之间调用，而 `remap_and_log` 持 `edits` 时只跑纯投影、不回调宿主 ——
        // 不存在"持锁重入宿主"的环路。锁序纪律见 `render/document.rs` 头部注释。
        if let (Some(parent_key), Some(child_key)) = (
            crate::editor::parameter_atlas::authority_key(None, &parent_persistent_id),
            crate::editor::parameter_atlas::authority_key(None, &child_persistent_id),
        ) {
            if self
                .session
                .edits
                .lock_or_recover()
                .atlas
                .seed_authority(&parent_key, &child_key)
            {
                log::info!(
                    "[ara] clone: parameters copied {parent_persistent_id} -> {child_persistent_id}"
                );
            }
        }
        Ok(index)
    }

    fn update_audio_modification(
        &mut self,
        state: &mut Self::AudioModification,
        properties: AudioModificationProperties,
    ) -> Result<(), AraError> {
        // 【为什么必须实现】此前落 trait 默认空实现，于是宿主换掉 modification 的
        // persistentID 时我们毫无反应，而 region 正是按 persistentID 引用它的 ——
        // 下一次映射会把所有引用它的 region 判成 `UnknownModification` 整条跳过。
        self.mark_model_dirty();
        let Some(modification) = self.document.audio_modifications.get_mut(*state) else {
            return Err(AraError::InvalidState(
                "audio modification state is not in the document",
            ));
        };
        let previous = std::mem::replace(
            &mut modification.persistent_id,
            properties.persistent_id().to_string(),
        );
        if previous != modification.persistent_id {
            log::info!(
                "[ara] audio_modification #{state} persistentID: {previous} -> {}",
                modification.persistent_id
            );
            // 通知句柄按 persistentID 索引：换了 ID 就要把登记跟着搬过去，
            // 否则后续的参数变更通知会投给一个已经不存在的键。
            let handle = self
                .modification_index_by_handle
                .iter()
                .find(|(_, index)| **index == *state)
                .map(|(handle, _)| *handle);
            self.session.forget_modification_handle(&previous);
            if let Some(handle) = handle {
                let current = modification.persistent_id.clone();
                self.session.register_modification_handle(&current, handle);
            }
            // 【为什么权威表也必须跟着搬】参数权威按 modification persistentID 索引
            // （`authority_key`），而这一句把 ID 换掉了。不搬就等于把这条 modification
            // 的曲线孤儿化 —— 下一次投影查 `mod:{新ID}` 必然落空，用户看到参数被重置。
            // 通知句柄与参数权威是**同一个键**的两种登记，必须一起搬。
            let remap = [(previous.clone(), modification.persistent_id.clone())];
            self.session
                .edits
                .lock_or_recover()
                .atlas
                .remap_authorities(&remap);
        }
        // persistentID 是 region 的引用键：它变了，session 的派生表必须立刻跟上，
        // 否则收口前的读取会拿旧键去查，把所有引用它的 region 判成 `UnknownModification`。
        self.sync_session_derived_tables();
        Ok(())
    }

    fn destroy_audio_modification(&mut self, state: Self::AudioModification) {
        self.mark_model_dirty();
        // 【为什么必须实现】此前落 trait 默认空实现，而 `document.audio_modifications`
        // 是 push-only 的：宿主销毁的 modification 永远留在表里，文档随编辑会话无界增长，
        // 且引用它的 region 仍能映射成功（拿着一个已不存在的 modification）。
        //
        // 这里不 compact 下标（与 `destroyed_regions` 同口径）—— 其余 modification 的
        // 下标变了会让 region 的引用边整体错位。只把这一槽标成"已消失"，
        // 映射层看到空 persistentID 就不再认它。
        if let Some(modification) = self.document.audio_modifications.get_mut(state) {
            self.session
                .forget_modification_handle(&modification.persistent_id);
            modification.persistent_id.clear();
            modification.audio_source_persistent_id.clear();
        }
        log::info!("[ara] audio_modification destroyed #{state}");
        // 立刻同步派生表：收口前的渲染/投影不能再看到这个 modification。
        self.sync_session_derived_tables();
    }
}

/// 宿主带来本插件不认识的 `transformationFlags` 位时限流记录一条。
///
/// 【为什么必须留痕】解码只认低四位（1 = Timestretch、2 = TimestretchReflectingTempo、
/// 4 = 内容淡出尾、8 = 内容淡出头）—— 其中 **2 只作留痕、不作语义**：拉伸由"时长差"
/// 表达，映射层用的就是 `durationInModificationTime / durationInPlaybackTime` 这个比值，
/// 而 tempo-reflecting region 的这个比值**正是**宿主算出的速度关系，所以按它渲染就是
/// 这一位所要求的结果（见 `runtime.rs` 的能力声明）。
///
/// 其余位（含未来 SDK 新增位）此前被静默丢弃，违背本插件"逐 region 显式降级"的纪律：
/// 能力协商没谈下来的东西，宿主却可能照给。窗口内每类未知位组合最多一条。
///
/// 【注意】工厂**不广告**的位不会走到这里 —— 库在 `create_playback_region` 里就按
/// `contains` 把整条 region 静默丢成 null 了。这里只处理"已广告、但语义上不参与渲染"的位。
fn log_unknown_transformation_flags(flags: i32) {
    const DECODED_TRANSFORMATION_FLAGS: i32 = 0b1111;
    let unknown = flags & !DECODED_TRANSFORMATION_FLAGS;
    if unknown != 0 {
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            &format!("ara:unknown-transformation-flags:{unknown:#x}"),
            format_args!(
                "[ara] playback region carries transformationFlags 0x{flags:X}; unknown bits {unknown:#x} ignored"
            ),
        );
    }
}

impl PlaybackRegions for ModelHandle {
    type PlaybackRegion = usize;
    /// 与 `AudioModifications::AudioModification` 同一个类型（本地补丁）。
    type AudioModification = usize;
    /// 与 `RegionSequences::RegionSequence` 同一个类型（本地补丁）。
    type RegionSequence = usize;

    fn create_playback_region(
        &mut self,
        context: &CreateContext,
        modification: &Self::AudioModification,
        sequence: &Self::RegionSequence,
        properties: PlaybackRegionProperties,
    ) -> Result<Self::PlaybackRegion, AraError> {
        // 这两条边（region → modification / region → sequence）来自本地补丁的 trait 参数。
        // 上游的委托层把它们**丢掉**了，而这正是建时间线缺的那两块拼图。
        self.mark_model_dirty();
        let sequence_index = Some(*sequence);
        let modification_persistent_id = self
            .document
            .audio_modifications
            .get(*modification)
            .map(|m| m.persistent_id.clone())
            .unwrap_or_default();
        let source_persistent_id = self
            .document
            .audio_modifications
            .get(*modification)
            .map(|m| m.audio_source_persistent_id.clone())
            .unwrap_or_default();

        let flags = properties.transformation_flags();
        log_unknown_transformation_flags(flags);
        // ARA 的内容淡化旗标（kARAPlaybackTransformationContentBasedFadeAtHead = 8，Tail = 4）。
        // 用数值是为了不依赖 sys 包的常量导出路径 —— 那两个值来自 ARAInterface.h，稳定。
        let has_content_based_fade_at_head = (flags & 8) != 0;
        let has_content_based_fade_at_tail = (flags & 4) != 0;
        let key = context
            .realtime_key()
            .ok_or(AraError::InvalidState("missing region key"))?;
        let index = self.allocate_region(
            key,
            AraPlaybackRegion {
                name: properties.name().map(str::to_owned),
                audio_source_persistent_id: source_persistent_id.clone(),
                audio_modification_persistent_id: modification_persistent_id,
                region_sequence_index: sequence_index,
                // 宿主句柄：按它解析归属，不受"序列重建换 slot"影响（见
                // `AraPlaybackRegion::sequence_model_ref`）。
                sequence_model_ref: properties
                    .region_sequence()
                    .map(|reference| reference.as_raw() as u64),
                start_in_modification_time: properties.start_in_modification_time(),
                duration_in_modification_time: properties.duration_in_modification_time(),
                start_in_playback_time: properties.start_in_playback_time(),
                duration_in_playback_time: properties.duration_in_playback_time(),
                // ARA 的拉伸由"时长差"表达，这个旗标只是声明；映射层用的是时长比值。
                is_timestretch_enabled: (flags & 1) != 0,
                has_content_based_fade_at_head,
                has_content_based_fade_at_tail,
            },
        )?;
        self.publish_head_tail()?;
        if let Some(sequence) = properties.region_sequence() {
            self.session
                .sequence_regions
                .lock_or_recover()
                .entry(sequence.as_raw() as usize as u64)
                .or_default()
                .insert(key);
        }
        log::info!(
            "[ara] playback_region #{}: source={} startMod={:.6} durationMod={:.6} startPlay={:.6} durationPlay={:.6} flags=0x{:X}",
            index,
            source_persistent_id,
            properties.start_in_modification_time(),
            properties.duration_in_modification_time(),
            properties.start_in_playback_time(),
            properties.duration_in_playback_time(),
            flags as u32,
        );
        // 【为什么必须在这里失效】分割在 ARA 侧是 `destroy(父) + create(左) + create(右)`。
        // `destroy_playback_region` 会失效，但那次失效发生在**两条新 region 还不存在**的
        // 时刻；而渲染缓存键 `PreparedVersion` 的五个分量（model / edit / epoch / scope /
        // keys）只要全等，`prepare_job` 就早退。此前 `create` 一条都不失效，于是：
        //   * `revision` 停在 destroy 那一次的值（`clear_renderers` 是它唯一的推进点）；
        //   * `scope` 只在宿主调用 `addPlaybackRegion` 时才变 —— 那一步并不保证发生，
        //     也不保证时序在失效之后。
        // 结果就是**声音与界面永远停在旧版本**：用户报的"分割后毫无反应"。
        //
        // 【为什么不能只依赖 `scope`】`scope_revision` 依赖宿主真的重发
        // `addPlaybackRegion`；Cubase 的 ARA 角色分配与 REAPER 不同（见 `extension.rs`
        // 的角色说明），这条路径不保证。`clear_renderers()` 是唯一同时推进 `revision`
        // 与 `render_epoch` 的原语，用它才是与 `destroy` / `update` 同口径的处置。
        //
        // 【顺带修好的第二条】`revision` 也是 `refresh_host_geometry` 的早退判据，而
        // `region_items`（宿主 item GUID）的唯一写者就在它里面。此前"只新增 region"的
        // 编辑会让新 region 永远拿不到 item GUID —— 那正是"同一个片段在界面上出现两份、
        // 一份有波形一份空占位"的成因。
        self.session.clear_renderers();
        // 立刻同步派生表：新建的 region 要能马上被渲染/工作区路径看到 ——
        // 此前它只在 `end_editing` 才进 `clip_ids`，那之前的每次读取都查不到身份。
        self.sync_session_derived_tables();
        Ok(index)
    }

    /// 宿主移动、裁切或拉伸已有 item 时，替换几何属性并保留所属源与修改。
    fn update_playback_region(
        &mut self,
        state: &mut Self::PlaybackRegion,
        properties: PlaybackRegionProperties,
    ) -> Result<(), AraError> {
        // 【为什么只在几何真变时才失效】宿主在一次编辑会话里会对同一条 region 反复回调
        // （拖动、吸附、逐帧对齐），其中大量回调的几何与上一次完全相同。无条件
        // `clear_renderers()` 会把 `ready` 反复按回 `false`，而 `ready` 又参与
        // `ensure_loaded` 的"宿主是否冲突"判据 ⇒ 用户每做一次编辑都撞一串
        // "Conflict: host changed"（见会话侧同一处注释）。这里逐项比较，值没动就不失效。
        // 【为什么用容差比较而不是 `!=`】宿主折算工程速率后会把**同一份几何**重复交付，
        // 而两个数值可能只差最后几位（float/double 中间量、宿主自己的一次往返）。
        // `!=` 会把这种"数值等价、位不同"的回调判成变化 ⇒ 每次回调都 `clear_renderers()`
        // ⇒ `revision` / `render_epoch` 反复推进 ⇒ 用户每次拖动都撞一串
        // "Conflict: host changed"。判据换成 `host_value_compatible`（与分割检测同一套，
        // 已实测对 ARA 的 double 时间类型足够），只挡"宿主重复交付同一份几何"，
        // 不挡真实编辑 —— 真实编辑的位移远大于 1e-6 的相对容差。
        let flags = properties.transformation_flags();
        log_unknown_transformation_flags(flags);
        self.mark_model_dirty();
        let close = crate::host::geometry::host_value_compatible;
        let changed = self
            .document
            .playback_regions
            .get(*state)
            .is_some_and(|region| {
                !close(
                    region.start_in_modification_time,
                    properties.start_in_modification_time(),
                ) || !close(
                    region.duration_in_modification_time,
                    properties.duration_in_modification_time(),
                ) || !close(
                    region.start_in_playback_time,
                    properties.start_in_playback_time(),
                ) || !close(
                    region.duration_in_playback_time,
                    properties.duration_in_playback_time(),
                ) || region.is_timestretch_enabled != ((flags & 1) != 0)
                    || region.has_content_based_fade_at_head != ((flags & 8) != 0)
                    || region.has_content_based_fade_at_tail != ((flags & 4) != 0)
            });
        if changed {
            self.session.clear_renderers();
        }
        if let Some(sequence) = properties.region_sequence() {
            let sequence_key = sequence.as_raw() as usize;
            if let Some(index) = self.sequence_index_by_model_ref.get(&sequence_key).copied() {
                if let Some(key) = self.region_keys.get(*state) {
                    let mut members = self.session.sequence_regions.lock_or_recover();
                    for regions in members.values_mut() {
                        regions.remove(key);
                    }
                    members.entry(sequence_key as u64).or_default().insert(*key);
                }
                if let Some(region) = self.document.playback_regions.get_mut(*state) {
                    region.region_sequence_index = Some(index);
                }
            }
        }
        let region = self
            .document
            .playback_regions
            .get_mut(*state)
            .ok_or(AraError::InvalidArgument("unknown playback region"))?;
        region.name = properties.name().map(str::to_owned);
        region.start_in_modification_time = properties.start_in_modification_time();
        region.duration_in_modification_time = properties.duration_in_modification_time();
        region.start_in_playback_time = properties.start_in_playback_time();
        region.duration_in_playback_time = properties.duration_in_playback_time();
        region.is_timestretch_enabled = (flags & 1) != 0;
        region.has_content_based_fade_at_head = (flags & 8) != 0;
        region.has_content_based_fade_at_tail = (flags & 4) != 0;
        // 句柄跟随宿主的最新交付：序列重建后 region 会被重新挂到新序列上。
        region.sequence_model_ref = properties
            .region_sequence()
            .map(|reference| reference.as_raw() as u64);
        log::info!(
            "[ara] playback_region updated #{}: startPlay={:.6}",
            state,
            region.start_in_playback_time
        );
        // 立刻同步派生表：收口前的渲染/投影必须读到新几何，否则宿主"移动了 item"
        // 在插件里看起来什么都没发生。
        self.sync_session_derived_tables();
        Ok(())
    }

    /// 标记已销毁区间，编辑结束时只映射存活的区域。
    fn destroy_playback_region(&mut self, state: Self::PlaybackRegion) {
        self.mark_model_dirty();
        self.session.clear_renderers();
        if let Some(key) = self.region_keys.get(state) {
            let key = *key;
            region_owners()
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .remove(key);
            for regions in self.session.sequence_regions.lock_or_recover().values_mut() {
                regions.remove(&key);
            }
            // 【为什么在这里清文档侧映射】下面这些表都按 `region_key` 索引，而 region
            // 销毁是"这个 key 再也不会被用到"的唯一时刻（槽位还会被 `destroyed_regions`
            // 复用，见 `allocate_region_slot`）。此前它们只在**文档销毁**时清 —— 长会话里
            // 每次分割/移动/粘贴都会留下永不释放的条目，是真正的无界增长。
            self.session.region_items.lock_or_recover().remove(&key);
            self.session.authorized_takes.lock_or_recover().remove(&key);
            self.session
                .authorized_sources
                .lock_or_recover()
                .remove(&key);
            self.session.clip_ids.lock_or_recover().remove(&key);
            // 【为什么媒体派生边也要在这里清】`split_media_from` 按 clip id 索引，
            // 此前只在**文档销毁**时清 —— 与上面几张表同一类无界增长。而且它的键是
            // region key 派生的 clip id，而 `allocate_region` 会**复用**已销毁槽位的
            // key：一条永不回收的 `ara-region-{key}` 边会在 key 被新 region 拿到之后
            // 指向一个语义完全不同的"父段"，把别的素材的媒体身份借过来。
            //
            // 两个方向都要清：本 region 作为**子**的条目（键），以及它作为**父**的条目
            // （值）—— 后者留着会让后来的片段从一条已不存在的边继承。
            let clip_id = format!("ara-region-{key}");
            let mut lineage = self.session.split_media_from.lock_or_recover();
            lineage.remove(&clip_id);
            lineage.retain(|_, parent| *parent != clip_id);
        }
        self.destroyed_regions.insert(state);
        if let Err(error) = self.publish_head_tail() {
            log::error!("head/tail snapshot failed: {error}");
        }
        log::info!("[ara] playback_region destroyed #{}", state);
        // 立刻同步派生表：上面的逐键清理只动了这一个 key，而 `regions` 表里它还留着。
        self.sync_session_derived_tables();
    }
}

impl Drop for ModelHandle {
    fn drop(&mut self) {
        self.session.close();
        // 初始化失败也可能直接释放模型而没有 destroy_document 回调。
        region_owners()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove_document(self.document_id);
    }
}

/// 【为什么单独测它】分量越界或非有限时宁可返回 `None`（轨道留默认色），
/// 也不要写一个下游解析不了的字符串。
#[test]
fn host_colors_clamp_into_a_parseable_hex_string() {
    let color = ara2_bridge::core::Color::new(0.2, 0.4, 0.6).unwrap();
    assert_eq!(host_color_hex(Some(&color)).as_deref(), Some("#336699"));
    assert_eq!(host_color_hex(None), None);
}

#[cfg(test)]
mod tests {
    /// 真实模型稳定收口必须重新投影编辑，不能只有数学helper通过而GUI/authority仍在旧位置。
    #[test]
    fn source_parameter_atlas_moves_and_stretches_in_real_document_transactions() {
        use ara2_bridge::plugin::ExtensionRoles;
        let mut model = ModelHandle::new();
        model.document = identity_document(&["A"]);
        let identity = Box::new(0_u8);
        let key = (&*identity as *const u8) as u64;
        model.region_keys.push(key);
        region_owners()
            .lock_or_recover()
            .register(key, model.document_id, 0)
            .unwrap();
        model.remap_and_log();
        let document = model.session();
        let owner = Arc::new(crate::render::extension::ExtensionOwner::default());
        let raw = owner
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::PLAYBACK_RENDERER,
                None,
            )
            .unwrap();
        unsafe {
            let ext = &*raw;
            ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                ext.playbackRendererRef,
                key as *mut _,
            );
        }
        let host = document.workspace_timeline().unwrap();
        let mut client = host.clone();
        let root = client.tracks[0].id.clone();
        client.params_by_root_track.insert(
            root.clone(),
            hifishifter_kernel::state::TrackParamsState {
                frame_period_ms: 100.,
                pitch_edit_user_modified: true,
                pitch_orig: vec![57.; 6],
                pitch_edit: vec![60., 61., 62., 63., 64., 65.],
                ..Default::default()
            },
        );
        let baseline = document.workspace_view().unwrap();
        document
            .accept_workspace_edits(
                0,
                document.revision.load(Ordering::Acquire),
                &client,
                &baseline,
            )
            .unwrap();
        assert!(!document.edits.lock_or_recover().atlas.is_empty());
        model.document.playback_regions[0].start_in_playback_time = 1.;
        model.document.playback_regions[0].duration_in_playback_time = 1.;
        model.document.playback_regions[0].is_timestretch_enabled = true;
        model.remap_and_log();
        let pitch = document.edits.lock_or_recover().params[&root]
            .pitch_edit
            .clone();
        document.close();
        assert!(pitch.len() >= 21, "authority仍停在旧项目数组: {pitch:?}");
        assert_eq!(
            &pitch[10..21],
            &[60., 60.5, 61., 61.5, 62., 62.5, 63., 63.5, 64., 64.5, 65.]
        );
        assert_eq!(pitch[0], 0., "旧位置不能继续遗留编辑");
    }
    use super::*;
    use ara2_bridge::core::{RegionSequenceKind, Registry};

    fn identity_document(order: &[&str]) -> AraDocument {
        let mut doc = crate::ara::ara_document_from_json(include_str!(
            "../../tests/fixtures/ara-model.reaper.json"
        ))
        .unwrap();
        doc.musical_contexts[0].region_sequences.clear();
        doc.playback_regions.clear();
        doc.audio_modifications.clear();
        for (index, name) in order.iter().enumerate() {
            doc.musical_contexts[0]
                .region_sequences
                .push(AraRegionSequence {
                    name: Some(name.to_string()),
                    order_index: index as i32,
                    color: None,
                    playback_region_count: 1,
                });
            let modification = format!("host-mod-{name}");
            let source = doc.audio_sources[0].persistent_id.clone();
            doc.audio_modifications.push(AraAudioModification {
                persistent_id: modification.clone(),
                audio_source_persistent_id: source.clone(),
            });
            doc.playback_regions.push(AraPlaybackRegion {
                audio_source_persistent_id: source,
                audio_modification_persistent_id: modification,
                region_sequence_index: Some(index),
                duration_in_modification_time: 0.5,
                duration_in_playback_time: 0.5,
                ..Default::default()
            });
        }
        doc
    }

    fn frame(
        item: &str,
        audio_source: &str,
        mod_start: f64,
        mod_dur: f64,
        play_start: f64,
        play_dur: f64,
    ) -> RegionFrame {
        RegionFrame {
            key: 0,
            item: Some(item.to_string()),
            audio_source: audio_source.to_string(),
            audio_modification: "mod".to_string(),
            mod_start,
            mod_dur,
            play_start,
            play_dur,
        }
    }

    /// **新建 region 必须推进渲染版本。**
    ///
    /// 【为什么这条是 P0 回归】分割在 ARA 侧是 `destroy(父) + create(左) + create(右)`。
    /// `destroy_playback_region` 会失效，但那次失效发生在**两条新 region 还不存在**的
    /// 时刻；若 `create` 不失效，`PreparedVersion` 的五个分量（model / edit / epoch /
    /// scope / keys）就可能全等 ⇒ `prepare_job` 早退 ⇒ **声音与界面永远停在旧版本**。
    /// 用户在 Cubase 上看到的就是"分割后毫无反应"。
    ///
    /// 断言两个原子值：`revision`（模型代次，也是 `refresh_host_geometry` 的早退判据，
    /// 关系到新 region 能否拿到宿主 item GUID）与 `render_epoch`（快照代次）。
    #[test]
    fn creating_a_playback_region_advances_the_render_version() {
        let mut model = ModelHandle::new();
        model.document = identity_document(&["A"]);
        let before_revision = model.session.revision.load(Ordering::Acquire);
        let before_epoch = model.session.render_epoch.load(Ordering::Acquire);

        let mut sequence_identity = 0_u8;
        // SAFETY: 只作为不透明 model-ref 地址；下面 `create_playback_region` 只把它当
        // 键比较与登记（`as_raw() as usize`），从不解引用。地址在调用期间保持有效。
        let sequence_ref = unsafe {
            ara2_bridge::core::ModelRef::from_raw(
                (&raw mut sequence_identity).cast::<std::ffi::c_void>(),
            )
        }
        .unwrap();
        let properties =
            PlaybackRegionProperties::for_ara2(0, 0.0, 0.5, 0.0, 0.5, sequence_ref, None, None)
                .unwrap();
        let context = CreateContext::for_test(ApiGeneration::V2Final, 0xABCD);
        let modification = 0_usize;
        let sequence = 0_usize;
        PlaybackRegions::create_playback_region(
            &mut model,
            &context,
            &modification,
            &sequence,
            properties,
        )
        .expect("合法属性必须建出 region");

        assert!(
            model.session.revision.load(Ordering::Acquire) > before_revision,
            "新建 region 必须推进模型代次，否则宿主几何清单不会刷新、\
             新 region 拿不到 item GUID"
        );
        assert!(
            model.session.render_epoch.load(Ordering::Acquire) > before_epoch,
            "新建 region 必须推进快照代次，否则 prepare_job 早退、声音停在旧版本"
        );
    }

    /// 建一个只有一条 region 的模型，走**真实的**创建回调（不是直接改 `document`）。
    ///
    /// 【为什么不直接改 `document`】直接改会绕开 `allocate_region` 的身份登记、
    /// `clear_renderers()` 的版本推进与 `sync_session_derived_tables` 的派生表同步 ——
    /// 而那三件事正是本轮要钉住的。既有 `a_host_split_*` 测试正是因为这个绕行，
    /// 才没能发现"重算被跑 4 遍"。
    fn model_with_one_region() -> (ModelHandle, u64) {
        let mut model = ModelHandle::new();
        model.document = identity_document(&["A"]);
        // 宿主是"先建源与轨、再建 region"，夹具里预置的那条 region 不属于本场景。
        model.document.playback_regions.clear();
        let mut sequence_identity = 0_u8;
        // SAFETY: 只作不透明 model-ref 地址；`create_playback_region` 只把它当键比较与
        // 登记（`as_raw() as usize`），从不解引用。地址在调用期间保持有效。
        let sequence_ref = unsafe {
            ara2_bridge::core::ModelRef::from_raw(
                (&raw mut sequence_identity).cast::<std::ffi::c_void>(),
            )
        }
        .unwrap();
        let properties =
            PlaybackRegionProperties::for_ara2(0, 0.0, 0.5, 0.0, 0.5, sequence_ref, None, None)
                .unwrap();
        let context = CreateContext::for_test(ApiGeneration::V2Final, 0xA1B2C3D4);
        let modification = 0_usize;
        let sequence = 0_usize;
        let slot = PlaybackRegions::create_playback_region(
            &mut model,
            &context,
            &modification,
            &sequence,
            properties,
        )
        .expect("合法属性必须建出 region");
        let key = model.region_keys[slot];
        (model, key)
    }

    /// **被推迟的编辑收口之后，时间线必须仍然被发布。**
    ///
    /// 【为什么这是问题二 2c 的回归】`endEditing` 在 content reader 存活时被推迟
    /// （见 bridge 的 `pending_end_editing`），而它是**唯一**调用 `remap_and_log()`
    /// 的地方。宿主不会为重发的同一个编辑会话再发送一次 `endEditing`，所以推迟
    /// 之后必须由 `destroy_content_reader` / `enable_audio_source_samples_access` /
    /// `notify_model_updates` 三个补跑点重新收口。任何一条没接上 ⇒ `remap_and_log`
    /// 永不执行 ⇒ `document.timeline` 停在 `None` ⇒ `claimed_items` 保持 0 ⇒
    /// 界面停在"等待宿主音频"。
    ///
    /// 【为什么在模型层测】补跑点在 bridge 层，但它保护的不变量是模型层的：
    /// "收口一次 ⇒ 时间线被发布一次"。这里钉住那个不变量，bridge 侧的补跑调用
    /// 线由 `ara_entry` / controller 的日志与 T4 的重开测试覆盖。
    #[test]
    fn a_deferred_edit_close_still_publishes_the_timeline() {
        let (mut model, _key) = model_with_one_region();
        // 编辑会话开始：宿主在这一刻交付 region，但 reader 还活着。
        model.editing = true;
        model.session.editing_session.store(true, Ordering::Release);
        // 模型侧等到收口才更新时间线。
        assert!(
            model.timeline.is_none(),
            "收口前不得发布时间线（读者可能还在读旧图）"
        );

        // 收口（无论它由哪个补跑点触发，最终都走到这里）。
        // 直接用 `remap_and_log`（`end_editing` 的全部实质）加编辑标志复位，
        // 因为 `HostContentScope::unavailable()` 在测试里不可构造。
        model.editing = false;
        model
            .session
            .editing_session
            .store(false, Ordering::Release);
        model.remap_and_log();
        assert!(
            model.timeline.is_some(),
            "收口必须发布时间线，否则界面永远停在等待宿主音频"
        );
        assert!(
            !model.session.editing_session.load(Ordering::Acquire),
            "收口必须解除编辑会话，否则后续通知会被 is_editing 拒绝"
        );
    }

    /// 同一份模型重复交付时**不重算**。
    ///
    /// 【为什么必须钉住】实测 Cubase 15 的一次分割里同一份模型被算了 4 遍（日志
    /// 22:20:54：同一条 `playback_region #0` 的映射读数连着出现 4 次）。每次重算都会
    /// 重跑参数投影与宿主清单刷新、并可能失效渲染输入 —— 那是自动应用
    /// "冻结 → 发布"窗口被自己打断、报 `Conflict: host model changed during automatic
    /// apply` 的直接来源。
    #[test]
    fn end_editing_skips_the_remap_when_the_model_did_not_change() {
        let (mut model, _key) = model_with_one_region();
        // 还没发布过时间线 ⇒ 必须算。
        assert!(!model.remap_is_redundant());

        model.remap_and_log();
        assert!(model.timeline.is_some(), "首次重算必须发布时间线");
        assert!(
            model.remap_is_redundant(),
            "同一份模型再交付一次时，这次重算必须是冗余的"
        );

        // 真实编辑（几何变了）⇒ 必须重算。
        let mut state = 0_usize;
        let mut sequence_identity = 0_u8;
        // SAFETY: 同上，只作不透明 model-ref 地址。
        let sequence_ref = unsafe {
            ara2_bridge::core::ModelRef::from_raw(
                (&raw mut sequence_identity).cast::<std::ffi::c_void>(),
            )
        }
        .unwrap();
        let moved =
            PlaybackRegionProperties::for_ara2(0, 0.0, 0.5, 0.25, 0.5, sequence_ref, None, None)
                .unwrap();
        PlaybackRegions::update_playback_region(&mut model, &mut state, moved)
            .expect("移动一条已知 region 必须成功");
        assert!(
            !model.remap_is_redundant(),
            "几何真的变了就必须重算，去重不得吞掉真实变化"
        );

        model.remap_and_log();
        assert!(model.remap_is_redundant());
    }

    /// **`needs_rebind` 卡住时，投影必须被跳过且必须留痕（A6）。**
    ///
    /// 【为什么这是用户报障的根因之一】`needs_rebind` 是冷恢复/存档还原的标记，唯一
    /// 清零点是 `EditState::reconcile`（`state_channel.rs:183`），而 `reconcile` 只被
    /// 恢复路径调用。一次撤销/重载的恢复身份对不上，这个标记就停在 `true` —— 此后宿主里
    /// **任何**编辑都只发布几何与 PCM，参数投影再也不跑。用户的原话是"无论宿主内怎么
    /// 编辑，ARA 中都不能产生同步，只有把 ARA 插件完全删除再添加，才生效"。
    ///
    /// 本用例钉住两件事：
    ///   1. 卡住时投影**确实没跑**（参数表没被投影结果改写）—— 这是现状，也是为什么
    ///      它需要日志；
    ///   2. 这种情况**留下了一条 warn**。没有第 2 条，第 1 条在真机上与"其它原因导致
    ///      的不同步"完全无法区分（修 bug 只能靠猜）。
    ///
    /// 【为什么不断言日志内容】本测试床不捕获日志；断言靠的是可观察状态（曲线有没有
    /// 被改写）。日志的存在由 `ara.rebind.stuck` 这个**限流键**本身保证 —— 它每窗口
    /// 只发一条，因此即便在 Cubase 4 连送的收口里也不会刷屏。
    #[test]
    fn a_stuck_rebind_request_skips_projection_and_says_so() {
        let (mut model, _key) = model_with_one_region();
        // 先正常收口一次，让时间线与 `remap_stamp` 就位。
        model.remap_and_log();
        assert!(model.timeline.is_some(), "夹具必须先能正常收口");
        // 用户给这条 modification 画了一条曲线。
        model
            .session
            .edits
            .lock_or_recover()
            .atlas
            .seed_test_pitch_curve("mod:host-mod-A", 250., vec![60., 62., 64., 66.]);
        // 模拟一次"恢复身份对不上"：`reconcile` 失败 ⇒ 标记留在 true。
        model.session.edits.lock_or_recover().needs_rebind = true;

        // 宿主做了一次真实编辑（几何变了）⇒ 收口必须跑，指纹必须推进。
        let mut state = 0_usize;
        let mut sequence_identity = 0_u8;
        // SAFETY: 只作不透明 model-ref 地址（同 `model_with_one_region`）。
        let sequence_ref = unsafe {
            ara2_bridge::core::ModelRef::from_raw(
                (&raw mut sequence_identity).cast::<std::ffi::c_void>(),
            )
        }
        .unwrap();
        let moved =
            PlaybackRegionProperties::for_ara2(0, 0.0, 0.5, 0.1, 0.5, sequence_ref, None, None)
                .unwrap();
        PlaybackRegions::update_playback_region(&mut model, &mut state, moved)
            .expect("移动 region");
        model.remap_and_log();

        // 1) 时间线照常发布：几何与 PCM 不受参数投影影响（这正是"界面有反应但曲线不动"）。
        assert!(
            model.timeline.is_some(),
            "卡住时也必须继续发布时间线，否则用户连几何都看不到"
        );
        // 2) 新几何确实生效了 —— 证明"不是整份收口失败"，只是投影这一支被跳过。
        let start = model
            .session
            .timeline
            .lock_or_recover()
            .as_ref()
            .and_then(|timeline| timeline.clips.first().map(|clip| clip.start_sec))
            .expect("发布时间线必须带 clip");
        assert!(
            (start - 0.1).abs() < 1e-6,
            "宿主的新几何必须已经发布（否则本用例测的就不是\"投影被跳过\"）：{start}"
        );
        // 3) 投影没跑：权威仍在，但 `params` 里没有被投影出来的新曲线。
        let edits = model.session.edits.lock_or_recover();
        assert!(
            edits.atlas.has_authority_parameters(None, "host-mod-A"),
            "权威表本身必须保留（跳过投影不等于丢掉曲线）"
        );
        assert!(
            edits.params.is_empty(),
            "投影被跳过 ⇒ params 不得被写入：这正是用户看到的\"参数线不动\""
        );
    }

    /// **幂等闸门跳过的是"重算"，不是"准备"（A5）。**
    ///
    /// 【为什么必须单独钉住】Cubase 一次分割把同一份模型连送 4 遍，
    /// `remap_is_redundant` 跳过后 3 遍的**重算**是对的（不要重复投影、不要
    /// 重复失效渲染输入）。但跳过的如果顺带把 `prepare_renderers()` 也跳掉，
    /// 就断了 `ready` 的唯一置真路径 —— `ready` 假会让 `ensure_loaded` 返回
    /// `Conflict: host changed`，用户看到的就是"宿主编了但插件不同步"。
    ///
    /// 这条用例把"幂等只作用于重算"变成可观察的事实：4 次收口，重算写的
    /// `remap_stamp` 只有 1 次变化，而 `ready` 始终被刷新。
    ///
    /// 【为什么不能像 `host_audio_state_tests` 那样持事务】`prepare_renderers`
    /// **自己取文档事务**（`render/document.rs:1283`）。持着它调 `remap_and_log`
    /// 就是当场自锁。那条测试持锁调的是 `invalidate_render_input` —— 它刻意
    /// 不取事务，所以不受影响。这里改为**不持锁**跑完断言，只在可能残留后台
    /// 任务时单独等一下。
    #[test]
    fn the_idempotence_gate_skips_the_remap_but_still_prepares_renderers() {
        let (mut model, _key) = model_with_one_region();
        // 带一个播放实例，`prepare_renderers` 才有对象可通知。
        let owner = Arc::new(crate::render::extension::ExtensionOwner::default());
        let _bound = owner
            .bind_to_document(
                model.session.clone(),
                ApiGeneration::V2Final,
                ara2_bridge::plugin::ExtensionRoles::all(),
                ara2_bridge::plugin::ExtensionRoles::all(),
                None,
            )
            .expect("夹具实例必须能绑定文档");

        model.remap_and_log();
        assert!(model.timeline.is_some());
        let published = model.session.revision.load(Ordering::Acquire);
        let prepares = owner.prepare_request_count();
        assert!(
            model.session.ready.load(Ordering::Acquire),
            "收口必须把 ready 置真：它是 ensure_loaded 判断\"宿主是否交付了输入\"的依据"
        );

        // 同一份模型再交付三次：模型没变 ⇒ 每次都该被判为冗余。
        for _ in 0..3 {
            assert!(model.remap_is_redundant(), "同一份模型必须判为冗余");
            model.remap_and_log();
        }

        assert_eq!(
            owner.prepare_request_count(),
            prepares + 3,
            "三次冗余收口也必须各自准备一次渲染输入（否则 ready 的置真路径被跳过）"
        );
        assert!(
            model.session.ready.load(Ordering::Acquire),
            "冗余收口之后 ready 必须仍为真"
        );
        assert_eq!(
            model.session.revision.load(Ordering::Acquire),
            published,
            "同一份模型不得反复推进 revision，否则每次都会触发一次全量重载"
        );
        // 后台可能已排了准备任务；等它自行退出，避免与用例结尾的析构抢生命周期。
        owner.await_preparation_idle(std::time::Duration::from_secs(3));
    }

    /// **指纹不得因"region 没有分配到的身份 key"而退化。**
    ///
    /// 【为什么必须钉住】`remap_fingerprint` 曾以 region key 起头并用 `?` 过滤 ——
    /// 而直接构造 `playback_regions`（恢复路径、夹具）不经过 `allocate_region`，
    /// `region_keys` 是空的。那些行会被整个丢掉，指纹退化成"空 == 空"，于是**真实变化
    /// 被当成没变**、重算被跳过（`persisted_b_edits_follow_host_identity_…` 因此失败）。
    /// 一个"会漏掉输入"的幂等判据比不去重更危险。
    #[test]
    fn the_remap_fingerprint_covers_regions_without_an_allocated_key() {
        let mut model = ModelHandle::new();
        model.document = identity_document(&["A", "B"]);
        assert!(
            model.region_keys.is_empty(),
            "本夹具直接构造 document，不经过 allocate_region"
        );
        assert_eq!(
            model.remap_fingerprint().len(),
            2,
            "没有身份 key 的 region 也必须留下自己的一行"
        );
        model.remap_and_log();
        assert!(model.remap_is_redundant());
        // 丢掉一条 region：指纹必须变，重算不得被跳过。
        model.destroyed_regions.insert(0);
        assert!(!model.remap_is_redundant(), "真实变化不得被去重吞掉");
    }

    /// **宿主分割的端到端形状：`destroy(父) + create(左) + create(右)`。**
    ///
    /// 【为什么这是本轮最关键的测试缺口】此前的分割测试（`a_host_split_registers_…`）
    /// **手工构造 `lineage_frame` + 直接改 `document.playback_regions` + 直调
    /// `register_host_split_lineage()`**，绕过了三个回调本身 —— 也就绕过了每次
    /// `clear_renderers()` 的版本推进、`allocate_region` 的身份登记与
    /// `sync_session_derived_tables` 的派生表同步。用户报的"分割后报错 / 片段消失"
    /// 恰恰发生在这些被绕过的部分里。
    ///
    /// 本用例驱动**真实的回调序列**，并断言：
    ///   1. 两个半段都出现在映射出的时间线里（"片段消失"）；
    ///   2. 父段身份已从派生表消失；
    ///   3. 每个 mutation 都失效了渲染版本（否则声音停在旧版本）；
    ///   4. 收口后的第二次交付同一模型被识别为冗余（"重算被跑 4 遍"）。
    #[test]
    fn a_host_split_keeps_both_halves_and_remaps_once() {
        let (mut model, parent_key) = model_with_one_region();
        model.remap_and_log();
        assert!(model.timeline().is_some(), "首次重算必须发布时间线");
        assert!(
            model
                .session
                .regions
                .lock_or_recover()
                .contains_key(&parent_key),
            "父段此刻应当是活的"
        );
        let after_setup = model.session.revision.load(Ordering::Acquire);
        let epoch_after_setup = model.session.render_epoch.load(Ordering::Acquire);

        let mut sequence_identity = 0_u8;
        // SAFETY: 只作不透明 model-ref 地址（同 `model_with_one_region`）。
        let sequence_ref = unsafe {
            ara2_bridge::core::ModelRef::from_raw(
                (&raw mut sequence_identity).cast::<std::ffi::c_void>(),
            )
        }
        .unwrap();

        // 1) 宿主销毁父段。
        PlaybackRegions::destroy_playback_region(&mut model, 0_usize);
        // 2) 宿主新建两个相邻半段（源窗 [0,0.25) 与 [0.25,0.5)，播放位置相邻）。
        let modification = 0_usize;
        let sequence = 0_usize;
        for (key, lo, hi) in [(0x1111_u64, 0.0_f64, 0.25_f64), (0x2222, 0.25, 0.5)] {
            let properties = PlaybackRegionProperties::for_ara2(
                0,
                lo,
                hi - lo,
                lo,
                hi - lo,
                sequence_ref,
                None,
                None,
            )
            .unwrap();
            let context = CreateContext::for_test(ApiGeneration::V2Final, key);
            PlaybackRegions::create_playback_region(
                &mut model,
                &context,
                &modification,
                &sequence,
                properties,
            )
            .expect("半段必须建得出来");
        }

        assert!(
            model.session.revision.load(Ordering::Acquire) >= after_setup + 3,
            "destroy + 两次 create 必须各失效一次渲染版本（否则宿主几何清单与快照都不刷新）"
        );
        assert!(
            model.session.render_epoch.load(Ordering::Acquire) > epoch_after_setup,
            "同一批 mutation 也必须推进快照代次"
        );

        // 3) 模型真的变了 ⇒ 这一次收口不许被去重跳过。
        assert!(!model.remap_is_redundant());
        model.remap_and_log();
        assert!(model.remap_is_redundant());

        // 4) 两个半段都必须在；父段必须已经不在派生表里。
        let workspace_keys = model.session.regions.lock_or_recover().clone();
        assert!(
            !workspace_keys.contains_key(&parent_key),
            "父段销毁后不应还留在派生表里"
        );
        let timeline = model.timeline().unwrap();
        assert_eq!(
            timeline.clips.len(),
            2,
            "两个半段都必须出现在时间线里（否则用户看到的是\"分割后片段消失\"）: {:?}",
            timeline
                .clips
                .iter()
                .map(|clip| clip.id.as_str())
                .collect::<Vec<_>>()
        );
    }

    /// **宿主分割后两个半段都保留父段参数**（用户报障的模型级回归）。
    ///
    /// 【为什么这是核心回归】旧设计按 `ara_region_signature`（含播放几何）认领曲线，
    /// 分割一发生签名就变、谱系蒸发，两个半段都拿不到父段记录 —— 用户在 Cubase 里看到的
    /// 就是"分割后参数被重置"。新设计把参数按稳定宿主身份持有：父段的源域曲线被整份播给
    /// 两个半段（P1 `cloneAudioModification` 或 P2 结构 diff），投影各取所需。
    #[test]
    fn a_host_split_keeps_the_parent_parameters_on_both_halves() {
        let (mut model, _parent_key) = model_with_one_region();
        model.remap_and_log();
        // 用户给父段画了一条音高曲线。
        model
            .session
            .edits
            .lock_or_recover()
            .atlas
            .seed_test_pitch_curve("mod:host-mod-A", 250., vec![60., 62., 64., 66.]);

        // 宿主克隆 modification：`cloneAudioModification` 是 ARA 对"创建独立变体"的
        // 专用原语（`ARAInterface.h:2653-2656`），因此它**就是**复制/分割回执。
        let cloned = AudioModificationProperties::new(None, "host-mod-A-split").unwrap();
        let source = 0_usize;
        let clone_index = AudioModifications::clone_audio_modification(
            &mut model,
            &CreateContext::for_test(ApiGeneration::V2Final, 0x55),
            &0_usize,
            &source,
            cloned,
        )
        .expect("克隆必须成功");
        assert!(
            model
                .session
                .edits
                .lock_or_recover()
                .atlas
                .has_authority_parameters(None, "host-mod-A-split"),
            "cloneAudioModification 是 ARA 给出的复制回执：父段参数必须播给新 modification"
        );

        // 宿主销毁父段、建两个相邻半段，两个半段都引用**克隆出来的** modification。
        PlaybackRegions::destroy_playback_region(&mut model, 0_usize);
        let mut sequence_identity = 0_u8;
        // SAFETY: 只作不透明 model-ref 地址（同 `model_with_one_region`）。
        let sequence_ref = unsafe {
            ara2_bridge::core::ModelRef::from_raw(
                (&raw mut sequence_identity).cast::<std::ffi::c_void>(),
            )
        }
        .unwrap();
        let sequence = 0_usize;
        for (key, lo, hi) in [(0x11_u64, 0.0_f64, 0.25_f64), (0x22, 0.25, 0.5)] {
            let properties = PlaybackRegionProperties::for_ara2(
                0,
                lo,
                hi - lo,
                lo,
                hi - lo,
                sequence_ref,
                None,
                None,
            )
            .unwrap();
            let context = CreateContext::for_test(ApiGeneration::V2Final, key);
            PlaybackRegions::create_playback_region(
                &mut model,
                &context,
                &clone_index,
                &sequence,
                properties,
            )
            .expect("半段必须建得出来");
        }
        model.remap_and_log();

        let timeline = model.timeline().unwrap().clone();
        assert_eq!(timeline.clips.len(), 2, "两个半段都必须出现在时间线里");
        let identities = model
            .session
            .parameter_identities_locked(&timeline)
            .unwrap();
        let atlas = model.session.edits.lock_or_recover().atlas.clone();
        // 渲染路径用的就是 `project_local`：数组以片段自己的零点对齐。
        let projected = atlas.project_local(&timeline, &identities).unwrap();
        let mut heads = Vec::new();
        for clip in &timeline.clips {
            let params = projected
                .get(&clip.id)
                .unwrap_or_else(|| panic!("半段 {} 必须拿到参数", clip.id));
            heads.push(params.pitch_edit[0]);
        }
        heads.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // 父曲线在源域是 60,62,64,66（每 250ms 一格）。左半段消费源窗 [0,0.25) ⇒ 60；
        // 右半段消费 [0.25,0.5) ⇒ 62。两个半段各自拿到**自己那段**，而不是被重置。
        assert_eq!(
            heads,
            vec![60., 62.],
            "分割后两个半段必须各自继承父段曲线的对应子段（用户报障的直接形状）"
        );
    }

    /// **宿主重复交付同一份几何**时不得失效渲染版本。
    ///
    /// 【为什么用容差而不是 `!=`】宿主折算工程速率后会把同一份几何重复回调，两个数可能
    /// 只差最后几位。`!=` 会把这种回调判成变化 ⇒ 每次回调都 `clear_renderers()` ⇒
    /// `revision` / `render_epoch` 反复推进 ⇒ 用户每次拖动都撞一串 "Conflict: host changed"。
    #[test]
    fn redelivering_the_same_geometry_does_not_invalidate_the_render_version() {
        let (mut model, _key) = model_with_one_region();
        let mut state = 0_usize;
        let mut sequence_identity = 0_u8;
        // SAFETY: 只作不透明 model-ref 地址（同 `model_with_one_region`）。
        let sequence_ref = unsafe {
            ara2_bridge::core::ModelRef::from_raw(
                (&raw mut sequence_identity).cast::<std::ffi::c_void>(),
            )
        }
        .unwrap();

        let before_revision = model.session.revision.load(Ordering::Acquire);
        let before_epoch = model.session.render_epoch.load(Ordering::Acquire);

        // 完全相同的几何。
        let same =
            PlaybackRegionProperties::for_ara2(0, 0.0, 0.5, 0.0, 0.5, sequence_ref, None, None)
                .unwrap();
        PlaybackRegions::update_playback_region(&mut model, &mut state, same).unwrap();
        assert_eq!(
            model.session.revision.load(Ordering::Acquire),
            before_revision,
            "重复交付同一份几何不得推进模型代次"
        );
        assert_eq!(
            model.session.render_epoch.load(Ordering::Acquire),
            before_epoch,
            "重复交付同一份几何不得推进快照代次"
        );

        // 只差最后几位的几何（宿主折算工程速率的典型表现）。
        let jittered = PlaybackRegionProperties::for_ara2(
            0,
            0.0,
            0.5 + 1e-12,
            0.0,
            0.5 + 1e-12,
            sequence_ref,
            None,
            None,
        )
        .unwrap();
        PlaybackRegions::update_playback_region(&mut model, &mut state, jittered).unwrap();
        assert_eq!(
            model.session.revision.load(Ordering::Acquire),
            before_revision,
            "容差内的抖动不得被当成一次真实编辑"
        );

        // 真实移动：必须失效。
        let moved =
            PlaybackRegionProperties::for_ara2(0, 0.0, 0.5, 0.25, 0.5, sequence_ref, None, None)
                .unwrap();
        PlaybackRegions::update_playback_region(&mut model, &mut state, moved).unwrap();
        assert!(
            model.session.revision.load(Ordering::Acquire) > before_revision,
            "真实编辑必须失效渲染版本，否则声音停在旧位置"
        );
    }

    /// 分割判据只认"同源 + 子段源窗都在父段内 + 首尾相接覆盖父段整段"，且至少两段。
    ///
    /// 【为什么放宽了旧判据】旧版要求"恰好两段且左右源窗严格相邻"，一次多刀分割
    /// （N 段）就不满足，谱系一条都不登记。新判据只保留**可判定**的覆盖关系。
    #[test]
    fn host_split_detection_requires_contiguous_windows_on_one_source() {
        let parent = frame("{P}", "src", 0., 1., 0., 1.);
        let left = frame("{P}", "src", 0., 0.5, 0., 0.5);
        let right = frame("{R}", "src", 0.5, 0.5, 0.5, 0.5);
        assert_eq!(
            ModelHandle::split_children(&parent, &[&left, &right]).len(),
            2,
            "两段相邻且覆盖父段 ⇒ 是一次分割"
        );

        // 一次多刀分割（N 段）也认 —— 旧判据在这里会一条谱系都不登记。
        let third = frame("{S}", "src", 0.75, 0.25, 0.75, 0.25);
        let half = frame("{R}", "src", 0.5, 0.25, 0.5, 0.25);
        assert_eq!(
            ModelHandle::split_children(&parent, &[&left, &half, &third]).len(),
            3,
            "三段首尾相接同样是一次分割"
        );

        // 源窗不连续（中间缺一段）⇒ 不是一次分割。
        let gapped = frame("{R}", "src", 0.6, 0.4, 0.5, 0.5);
        assert!(ModelHandle::split_children(&parent, &[&left, &gapped]).is_empty());
        // 不同源 ⇒ 不是同一次分割。
        let other_source = frame("{R}", "other", 0.5, 0.5, 0.5, 0.5);
        assert!(ModelHandle::split_children(&parent, &[&left, &other_source]).is_empty());
        // 半段之和与父窗口不符 ⇒ 不是分割（例如只是一次裁切）。
        let wrong_total = frame("{R}", "src", 0.5, 0.2, 0.5, 0.5);
        assert!(ModelHandle::split_children(&parent, &[&left, &wrong_total]).is_empty());
        // 只有一段 ⇒ 不是分割。
        assert!(ModelHandle::split_children(&parent, &[&left]).is_empty());
        // 子段源窗超出父段 ⇒ 不是分割。
        let overflow = frame("{R}", "src", 0.5, 0.8, 0.5, 0.8);
        assert!(ModelHandle::split_children(&parent, &[&left, &overflow]).is_empty());
    }

    /// 结构 diff 必须把宿主分割登记成与 GUI 分割相同的形状。
    ///
    /// 【为什么这条最要紧】宿主分割只表现为"1 个 region 消失 + 2 个出现"，没有任何显式
    /// 谱系。不登记的话右半段落到几何兜底层，兄弟半段被当成合法祖先，曲线整体平移一个
    /// 分割长度 —— 正是用户报障的"宿主分割后 Clip 内产生偏移"。
    #[test]
    fn a_host_split_registers_the_same_lineage_shape_as_a_gui_split() {
        use crate::editor::parameter_atlas::{ParameterAtlas, RegionIdentity};
        const PARENT: &str = "{11111111-1111-1111-1111-111111111111}";
        const RIGHT: &str = "{22222222-2222-2222-2222-222222222222}";

        let mut model = ModelHandle::new();
        model.document = identity_document(&["A"]);
        // 父段有一条 live 的曲线记录（`split_seed` 只认 live 记录）。
        let mut timeline = TimelineState::default();
        timeline.tracks.push(
            serde_json::from_value(serde_json::json!({"id":"ara-track-0","name":"A","order":0}))
                .unwrap(),
        );
        timeline.clips.push(
            serde_json::from_value(serde_json::json!({
                "id":"ara-clip-1","track_id":"ara-track-0","name":"C","start_sec":0.,"length_sec":1.,
                "takes":[{"id":"t","source_path":"src","source_end_sec":1.}]
            }))
            .unwrap(),
        );
        timeline.bpm = 120.;
        timeline.project_sec = 1.;
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }
        timeline.params_by_root_track.insert(
            "ara-track-0".into(),
            hifishifter_kernel::state::TrackParamsState {
                frame_period_ms: 100.,
                pitch_orig: vec![57.; 6],
                pitch_edit: vec![60., 61., 62., 63., 64., 65.],
                ..Default::default()
            },
        );
        let identities = std::collections::BTreeMap::from([(
            "ara-clip-1".to_string(),
            RegionIdentity {
                key: 1,
                item: Some(PARENT.to_string()),
                source: "src".into(),
                modification: "mod".into(),
                signature: String::new(),
            },
        )]);
        model.session.edits.lock_or_recover().atlas = ParameterAtlas::default()
            .capture(&timeline, &identities)
            .unwrap();

        // 上一帧：一个父 region。
        model.lineage_frame = vec![frame(PARENT, "src", 0., 1., 0., 1.)];
        // 当前帧：父 region 消失，左半段（沿用父段 item GUID）+ 右半段（新 GUID）出现。
        model.document.playback_regions = vec![
            AraPlaybackRegion {
                audio_source_persistent_id: "src".into(),
                audio_modification_persistent_id: "mod-left".into(),
                start_in_modification_time: 0.,
                duration_in_modification_time: 0.5,
                start_in_playback_time: 0.,
                duration_in_playback_time: 0.5,
                ..Default::default()
            },
            AraPlaybackRegion {
                audio_source_persistent_id: "src".into(),
                audio_modification_persistent_id: "mod-right".into(),
                start_in_modification_time: 0.5,
                duration_in_modification_time: 0.5,
                start_in_playback_time: 0.5,
                duration_in_playback_time: 0.5,
                ..Default::default()
            },
        ];
        model.region_keys = vec![7, 8];
        {
            let mut items = model.session.region_items.lock_or_recover();
            items.insert(7, PARENT.to_string());
            items.insert(8, RIGHT.to_string());
        }

        model.register_host_split_lineage();

        let edits = model.session.edits.lock_or_recover();
        assert_eq!(
            edits.atlas.split_parent_of(RIGHT),
            Some(PARENT),
            "右半段必须显式继承左半段（= 父段）"
        );
        assert!(
            edits.atlas.has_split_basis(RIGHT),
            "父段的完整窗口必须以 seed 形式保留，右半段靠它做包含判定"
        );
        drop(edits);
        assert_eq!(
            model
                .session
                .split_media_from
                .lock_or_recover()
                .get(RIGHT)
                .map(String::as_str),
            Some(PARENT),
            "媒体派生谱系也要登记（右半段借父段媒体）"
        );
        // 本帧成为下一帧的基线。
        assert_eq!(model.lineage_frame.len(), 2);
    }

    /// **一次分割分两轮送达：右半段仍然要拿到父段的参数权威。**
    ///
    /// 【为什么这条是回归】结构 diff 只看"上一帧 vs 这一帧"。宿主先把父段缩短成左段、
    /// 下一轮才创建右段时，第一轮只有一个子段（`split_children` 要求至少两段），
    /// 第二轮又没有"消失的父段" —— 旧实现两条路都不登记，右半段从此没有权威。
    /// 现在候选会跨轮记住父段窗口，等右段出现再收口。
    #[test]
    fn a_split_delivered_in_two_rounds_still_seeds_the_right_half() {
        const PARENT: &str = "{AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA}";
        const RIGHT: &str = "{BBBBBBBB-BBBB-BBBB-BBBB-BBBBBBBBBBBB}";

        let mut model = ModelHandle::new();
        model.document = identity_document(&["A"]);
        model
            .session
            .edits
            .lock_or_recover()
            .atlas
            .seed_test_pitch_curve(&format!("item:{PARENT}"), 100., vec![61.; 8]);

        let region = |mod_start: f64, mod_dur: f64, play_start: f64| AraPlaybackRegion {
            audio_source_persistent_id: "src".into(),
            audio_modification_persistent_id: "mod".into(),
            start_in_modification_time: mod_start,
            duration_in_modification_time: mod_dur,
            start_in_playback_time: play_start,
            duration_in_playback_time: mod_dur,
            ..Default::default()
        };

        // 第一轮：父段消失，只有左段（沿用父 item GUID）。
        model.lineage_frame = vec![frame(PARENT, "src", 0., 1., 0., 1.)];
        model.document.playback_regions = vec![region(0., 0.5, 0.)];
        model.region_keys = vec![7];
        {
            let mut items = model.session.region_items.lock_or_recover();
            items.insert(7, PARENT.to_string());
        }
        model.register_host_split_lineage();
        assert_eq!(
            model.split_candidates.len(),
            1,
            "一轮没齐的分割必须留在候选里，而不是被丢掉"
        );

        // 第二轮：右段出现（新 GUID）。父段仍在候选里，跨帧收口。
        model.document.playback_regions.push(region(0.5, 0.5, 0.5));
        model.region_keys.push(8);
        model
            .session
            .region_items
            .lock_or_recover()
            .insert(8, RIGHT.to_string());
        model.register_host_split_lineage();

        let edits = model.session.edits.lock_or_recover();
        assert_eq!(
            edits.atlas.split_parent_of(RIGHT),
            Some(PARENT),
            "右半段必须显式继承父段"
        );
        assert!(
            edits.atlas.has_authority_parameters(Some(RIGHT), "mod"),
            "跨帧收口后必须把父段的权威播给右半段"
        );
        assert!(
            model.split_candidates.is_empty(),
            "收口后候选必须清掉，不能每帧重复登记"
        );
        drop(edits);
        // 右段的窗口来源必须登记：它是"从父段切"这条推导的唯一输入。
        assert_eq!(model.session.split_window_count(), 1);
        let windows = model.session.split_windows.lock_or_recover();
        let source = windows
            .get(RIGHT)
            .expect("right half must have a window source");
        assert_eq!(source.left, PARENT);
        assert!((source.offset_sec - 0.5).abs() < 1e-9);
        assert!((source.length_sec - 0.5).abs() < 1e-9);
    }

    /// 父段**没有** live 曲线时：曲线谱系不登记（没有可继承的 seed），但**媒体身份必须
    /// 照常登记**。
    ///
    /// 【为什么单独钉住】此前 `split_seed` 取不到就 `return`，连 `split_media_from` 一起
    /// 跳过 ⇒ 右半段既无曲线祖先也无媒体身份，波形要等下一次授权才出现。媒体身份只依赖
    /// "父段与右半段同源"，与父段有没有曲线无关。
    #[test]
    fn a_host_split_without_a_curve_still_registers_media_identity() {
        const PARENT: &str = "{33333333-3333-3333-3333-333333333333}";
        const RIGHT: &str = "{44444444-4444-4444-4444-444444444444}";

        let mut model = ModelHandle::new();
        model.document = identity_document(&["A"]);
        // 刻意不 `capture` 任何曲线：父段没有 live 记录 ⇒ `split_seed` 返回 None。

        model.lineage_frame = vec![frame(PARENT, "src", 0., 1., 0., 1.)];
        model.document.playback_regions = vec![
            AraPlaybackRegion {
                audio_source_persistent_id: "src".into(),
                audio_modification_persistent_id: "mod-left".into(),
                start_in_modification_time: 0.,
                duration_in_modification_time: 0.5,
                start_in_playback_time: 0.,
                duration_in_playback_time: 0.5,
                ..Default::default()
            },
            AraPlaybackRegion {
                audio_source_persistent_id: "src".into(),
                audio_modification_persistent_id: "mod-right".into(),
                start_in_modification_time: 0.5,
                duration_in_modification_time: 0.5,
                start_in_playback_time: 0.5,
                duration_in_playback_time: 0.5,
                ..Default::default()
            },
        ];
        model.region_keys = vec![7, 8];
        {
            let mut items = model.session.region_items.lock_or_recover();
            items.insert(7, PARENT.to_string());
            items.insert(8, RIGHT.to_string());
        }

        model.register_host_split_lineage();

        let edits = model.session.edits.lock_or_recover();
        assert!(
            !edits.atlas.has_authority_parameters(Some(RIGHT), "mod"),
            "父段没有曲线时不得凭空造一份权威"
        );
        assert_eq!(
            edits.atlas.split_parent_of(RIGHT),
            Some(PARENT),
            "谱系边仍然要记：它是诊断与旧路径的唯一读数"
        );
        drop(edits);
        assert_eq!(
            model
                .session
                .split_media_from
                .lock_or_recover()
                .get(RIGHT)
                .map(String::as_str),
            Some(PARENT),
            "媒体身份与曲线 seed 解耦，必须照常登记"
        );
    }

    /// 保存后新建图的序号不可决定旧编辑归属；共同source由真正modification身份区分。
    #[test]
    fn persisted_b_edits_follow_host_identity_after_deleting_a_and_reordering_creation() {
        let mut model = ModelHandle::new();
        model.document = identity_document(&["A", "B"]);
        model.remap_and_log();
        let host = model.timeline().unwrap().clone();
        let mut client = host.clone();
        client.tracks[1].volume = 0.25;
        client.params_by_root_track.insert(
            "ara-track-1".into(),
            hifishifter_kernel::state::TrackParamsState {
                frame_period_ms: 5.0,
                pitch_edit: vec![62.0, 64.0],
                ..Default::default()
            },
        );
        // 这里只保留已编辑B，模拟B renderer的局部视图。
        client.tracks.remove(0);
        let mut view = host.clone();
        view.tracks.remove(0);
        *model.session.edits.lock_or_recover() = crate::state_channel::EditState::default()
            .merge(&view, &client, 0)
            .unwrap();
        // 同live图中删A区间，但B的会话slot不压缩。
        model.destroyed_regions.insert(0);
        model.remap_and_log();
        let bytes = model.session.edits.lock_or_recover().encode().unwrap();
        for order in [vec!["B"], vec!["B", "A"]] {
            let mut reopened = ModelHandle::new();
            reopened.document = identity_document(&order);
            reopened
                .session
                .edits
                .lock_or_recover()
                .restore(&bytes)
                .unwrap();
            reopened.remap_and_log();
            let mut timeline = reopened.timeline().unwrap().clone();
            reopened
                .session
                .edits
                .lock_or_recover()
                .apply(&mut timeline);
            assert_eq!(timeline.tracks[0].name, "B");
            assert_eq!(
                timeline.tracks[0].volume, 0.25,
                "B-first必须保留B音量而不依赖旧序号"
            );
            assert_eq!(
                timeline.params_by_root_track["ara-track-0"].pitch_edit,
                [62.0, 64.0]
            );
            if timeline.tracks.len() > 1 {
                assert_eq!(timeline.tracks[1].volume, 1.0, "不得串到A");
            }
            timeline.clips.retain(|clip| clip.track_id == "ara-track-0");
            let pcm = std::collections::HashMap::from([(
                reopened.document.audio_sources[0].persistent_id.clone(),
                hifishifter_kernel::mixdown::MixdownPcm {
                    sample_rate: 44100,
                    channels: 1,
                    samples: Arc::new(vec![0.2; 22050]),
                },
            )]);
            let options = hifishifter_kernel::mixdown::MixdownOptions {
                sample_rate: 44100,
                start_sec: 0.0,
                end_sec: Some(0.5),
                stretch: hifishifter_kernel::time_stretch::StretchAlgorithm::LinearResample,
                apply_pitch_edit: false,
                output: hifishifter_kernel::encode::OutputSpec::wav_32f(),
                quality_preset: hifishifter_kernel::mixdown::QualityPreset::Export,
                cancel_flag: None,
                progress: None,
                cache_stats: None,
            };
            let (_, _, _, samples) =
                hifishifter_kernel::mixdown::render_mixdown_with_pcm(&timeline, options, &pcm)
                    .unwrap();
            assert!(
                samples.iter().all(|sample| (*sample - 0.05).abs() < 1e-6),
                "重建后B输出保留0.25音量"
            );
        }
    }

    /// move 到别的 sequence 时几何与成员表必须一起迁移，旧 editor 不能继续播放。
    #[test]
    fn region_sequence_update_moves_the_editor_membership() {
        let mut model = ModelHandle::new();
        model.document = crate::ara::ara_document_from_json(include_str!(
            "../../tests/fixtures/ara-model.reaper.json"
        ))
        .unwrap();
        let mut registry = Registry::<RegionSequenceKind, ()>::new(2);
        let old_handle = registry.insert(()).unwrap();
        let new_handle = registry.insert(()).unwrap();
        let old = registry.model_ref(old_handle).unwrap();
        let new = registry.model_ref(new_handle).unwrap();
        let old_key = old.as_raw() as usize as u64;
        let new_key = new.as_raw() as usize as u64;
        let region = Box::new(0_u8);
        let region_key = (&*region as *const u8) as u64;
        model.region_keys.push(region_key);
        model
            .sequence_index_by_model_ref
            .insert(old_key as usize, 0);
        model
            .sequence_index_by_model_ref
            .insert(new_key as usize, 1);
        model
            .session
            .sequence_regions
            .lock_or_recover()
            .insert(old_key, [region_key].into_iter().collect());
        model
            .session
            .sequence_regions
            .lock_or_recover()
            .insert(new_key, HashSet::new());
        let properties =
            PlaybackRegionProperties::for_ara2(1, 0.0, 1.0, 2.0, 1.0, new, None, None).unwrap();
        PlaybackRegions::update_playback_region(&mut model, &mut 0, properties).unwrap();
        assert_eq!(
            model.document.playback_regions[0].region_sequence_index,
            Some(1)
        );
        assert!(model.session.sequence_regions.lock_or_recover()[&old_key].is_empty());
        assert_eq!(
            model.session.sequence_regions.lock_or_recover()[&new_key],
            [region_key].into_iter().collect()
        );
    }

    /// 实际授权/撤权回调必须更新源表，内容版本刷新后不能继续持旧 PCM。
    #[test]
    fn source_grant_version_change_and_revocation_refresh_only_host_pcm() {
        use ara2_bridge::plugin::{HostAudioSourceRef, HostClients};
        let mut model = ModelHandle::new();
        model.document = crate::ara::ara_document_from_json(include_str!(
            "../../tests/fixtures/ara-model.reaper.json"
        ))
        .unwrap();
        model.document.audio_sources[0].sample_count = 4;
        let mut fixture = crate::test_host::HostFixture::new(vec![vec![0.1, 0.2, 0.3, 0.4]]);
        let host = fixture.instance();
        // SAFETY: 稳定 fixture 保留到 clients 与模型授权 scope 释放。
        let clients = unsafe { HostClients::from_raw(&host, ApiGeneration::V2Final) }.unwrap();
        let mut identity = 0_u8;
        // SAFETY: 仅作为不透明宿主源身份，存活到全部回调结束。
        let source = unsafe { HostAudioSourceRef::from_raw((&raw mut identity).cast()) }.unwrap();
        let mut slot = 0;
        let original_revision = model.session.revision.load(Ordering::Acquire);
        clients
            .with_audio_source_management(source, |scope| {
                AudioSources::enable_audio_source_samples_access(
                    &mut model, &mut slot, true, &scope,
                )
            })
            .unwrap();
        assert_eq!(
            model.session.revision.load(Ordering::Acquire),
            original_revision,
            "授权不改变宿主模型版本"
        );
        let id = model.document.audio_sources[0].persistent_id.clone();
        assert_eq!(
            model.session.sources.lock_or_recover()[&id].planes[0],
            [0.1, 0.2, 0.3, 0.4]
        );
        fixture.planes[0] = vec![0.5, 0.6, 0.7, 0.8];
        let before_content = model.session.revision.load(Ordering::Acquire);
        clients
            .with_audio_source_management(source, |scope| {
                AudioSources::update_audio_source_content(
                    &mut model,
                    &mut slot,
                    None,
                    ContentUpdateScopes::empty(),
                    &scope,
                )
            })
            .unwrap();
        assert!(
            model.session.revision.load(Ordering::Acquire) > before_content,
            "真实内容通知推进模型版本"
        );
        assert_eq!(
            model.session.sources.lock_or_recover()[&id].planes[0],
            [0.5, 0.6, 0.7, 0.8]
        );
        assert_eq!(model.session.sources.lock_or_recover()[&id].version, 1);
        let before_revoke = model.session.revision.load(Ordering::Acquire);
        clients
            .with_audio_source_management(source, |scope| {
                AudioSources::enable_audio_source_samples_access(
                    &mut model, &mut slot, false, &scope,
                )
            })
            .unwrap();
        assert_eq!(
            model.session.revision.load(Ordering::Acquire),
            before_revoke,
            "撤权不使GUI快照冲突"
        );
        assert_eq!(
            fixture.created.load(Ordering::Acquire),
            fixture.destroyed.load(Ordering::Acquire),
            "reader同步释放"
        );
        assert!(model.session.sources.lock_or_recover().is_empty());
        assert_eq!(
            model.session.edit_sources.lock_or_recover()[&id].planes[0],
            [0.5, 0.6, 0.7, 0.8]
        );
        clients
            .with_audio_source_management(source, |scope| {
                AudioSources::deactivate_audio_source(&mut model, &mut slot, true, &scope)
            })
            .unwrap();
        assert!(model.session.edit_sources.lock_or_recover().is_empty());
        assert!(
            model.session.revision.load(Ordering::Acquire) > before_revoke,
            "停用确实改变模型"
        );
    }

    /// **源从 undo 历史恢复时必须重读 PCM。**
    ///
    /// 【为什么这是问题三的回归】旧实现只在 `deactivate == true` 时刷新；重新激活
    /// （`false`）什么都不做，源永远停在"已停用 + 无 PCM"，界面显示"正在等待宿主
    /// 完成音频分配"。后台/前台切换若走 undo 停用路径，就是这个形状。
    #[test]
    fn a_source_deactivated_for_undo_is_restored_on_reactivation() {
        use ara2_bridge::plugin::{HostAudioSourceRef, HostClients};
        let mut model = ModelHandle::new();
        model.document = identity_document(&["B"]);
        model.document.audio_sources[0].sample_count = 4;
        let mut fixture = crate::test_host::HostFixture::new(vec![vec![0.1, 0.2, 0.3, 0.4]]);
        let host = fixture.instance();
        // SAFETY: fixture 与不透明源身份保留到所有 scope 被释放。
        let clients = unsafe { HostClients::from_raw(&host, ApiGeneration::V2Final) }.unwrap();
        let mut identity = 0_u8;
        let source = unsafe { HostAudioSourceRef::from_raw((&raw mut identity).cast()) }.unwrap();
        let mut slot = 0;
        clients
            .with_audio_source_management(source, |scope| {
                AudioSources::enable_audio_source_samples_access(
                    &mut model, &mut slot, true, &scope,
                )
            })
            .unwrap();
        let id = model.document.audio_sources[0].persistent_id.clone();
        assert!(
            model
                .session
                .edit_sources
                .lock_or_recover()
                .contains_key(&id),
            "前提：授权后必须已有 PCM"
        );
        clients
            .with_audio_source_management(source, |scope| {
                AudioSources::deactivate_audio_source(&mut model, &mut slot, true, &scope)
            })
            .unwrap();
        assert!(
            !model
                .session
                .edit_sources
                .lock_or_recover()
                .contains_key(&id),
            "停用后必须撤销 PCM"
        );
        clients
            .with_audio_source_management(source, |scope| {
                AudioSources::deactivate_audio_source(&mut model, &mut slot, false, &scope)
            })
            .unwrap();
        assert!(
            model
                .session
                .edit_sources
                .lock_or_recover()
                .contains_key(&id),
            "重新激活必须重读 PCM，而不是停在无源状态"
        );
    }

    /// 同一GUI快照经历真实授权开关可提交；实际源几何/内容通知仍使旧提交失效。
    #[test]
    fn same_snapshot_commit_survives_access_toggle_but_not_real_source_changes() {
        use ara2_bridge::plugin::{ExtensionRoles, HostAudioSourceRef, HostClients};
        use hifishifter_host_link as link;
        let mut model = ModelHandle::new();
        model.document = identity_document(&["B"]);
        model.document.audio_sources[0].sample_count = 4;
        model.document.playback_regions[0].duration_in_modification_time = 4.0 / 44100.0;
        model.document.playback_regions[0].duration_in_playback_time = 4.0 / 44100.0;
        let region = Box::new(0_u8);
        let key = (&*region as *const u8) as u64;
        model.region_keys.push(key);
        region_owners()
            .lock_or_recover()
            .register(key, model.document_id, 0)
            .unwrap();
        model.remap_and_log();
        let mut fixture = crate::test_host::HostFixture::new(vec![vec![0.1, 0.2, 0.3, 0.4]]);
        let host = fixture.instance();
        // SAFETY: fixture与不透明源身份保留到所有scope/renderer被释放。
        let clients = unsafe { HostClients::from_raw(&host, ApiGeneration::V2Final) }.unwrap();
        let mut identity = 0_u8;
        let source = unsafe { HostAudioSourceRef::from_raw((&raw mut identity).cast()) }.unwrap();
        let mut slot = 0;
        clients
            .with_audio_source_management(source, |scope| {
                AudioSources::enable_audio_source_samples_access(
                    &mut model, &mut slot, true, &scope,
                )
            })
            .unwrap();
        let owner = Arc::new(crate::render::extension::ExtensionOwner::default());
        let raw = owner
            .bind_to_document(
                model.session.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        // SAFETY: native扩展及region身份都仍存活。
        unsafe {
            let ext = &*raw;
            ((*ext.editorRendererInterface).addPlaybackRegion.unwrap())(
                ext.editorRendererRef,
                key as *mut _,
            );
        }
        // 管道时代的 `Request::Snapshot` 换成了对**工作区**的导出：清单的 `revisions`
        // 与 `host_geometry` 正是消费者会原样送回来的那份读数。每次导出都返回一份带
        // 当前代次的载荷，供「旧读数被实际源变更作废」的断言使用。
        let root = std::env::temp_dir().join(format!(
            "hfs-link-model-{}-{}",
            std::process::id(),
            link::now_ms()
        ));
        let export_edit = |owner: &crate::render::extension::ExtensionOwner,
                           root: &std::path::Path| {
            let (manifest, _workspace) = owner.export_link(root).unwrap();
            let mut timeline: hifishifter_kernel::state::TimelineState =
                serde_json::from_value(manifest.timeline.clone()).unwrap();
            timeline.tracks[0].volume = 0.5;
            let edit = link::EditManifest {
                format: link::EDIT_FORMAT.into(),
                format_version: link::FORMAT_VERSION,
                link_id: manifest.link_id.clone(),
                base_revisions: manifest.revisions,
                host_geometry: hifishifter_kernel::editor::host_timeline::host_geometry_projection(
                    &timeline,
                ),
                params: serde_json::to_value(&timeline.params_by_root_track).unwrap(),
                tracks: serde_json::to_value(&timeline.tracks).unwrap(),
            };
            (manifest.link_id.clone(), edit)
        };
        let (link_id, initial) = export_edit(&owner, &root);
        clients
            .with_audio_source_management(source, |scope| {
                AudioSources::enable_audio_source_samples_access(
                    &mut model, &mut slot, false, &scope,
                )
            })
            .unwrap();
        let mut left = [9.0_f32; 4];
        let mut right = [9.0_f32; 4];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = crate::audio_abi::AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        assert!(
            !unsafe { owner.snapshots[0].copy_block(0, 44100, &mut bus, 4) },
            "撤权首先撤销实时旧快照"
        );
        assert_eq!(left, [0.0; 4]);
        assert_eq!(
            fixture.created.load(Ordering::Acquire),
            fixture.destroyed.load(Ordering::Acquire)
        );
        let committed = owner.apply_link_edit(&link_id, &initial);
        assert!(committed.ok, "{:?}", committed.error);
        for change in 0..3 {
            let (old_link_id, old) = export_edit(&owner, &root);
            assert_eq!(old_link_id, link_id, "同一份文档只对应一个工作区身份");
            if change == 0 {
                let id = model.document.audio_sources[0].persistent_id.clone();
                let properties =
                    AudioSourceProperties::new(None, &id, 4, 44100.0, 1, false.into()).unwrap();
                clients
                    .with_audio_source_management(source, |scope| {
                        AudioSources::update_audio_source(&mut model, &mut slot, properties, &scope)
                    })
                    .unwrap();
            } else if change == 1 {
                clients
                    .with_audio_source_management(source, |scope| {
                        AudioSources::update_audio_source_content(
                            &mut model,
                            &mut slot,
                            None,
                            ContentUpdateScopes::empty(),
                            &scope,
                        )
                    })
                    .unwrap();
            } else {
                let mut registry = Registry::<RegionSequenceKind, ()>::new(1);
                let handle = registry.insert(()).unwrap();
                let properties = PlaybackRegionProperties::for_ara2(
                    0,
                    0.0,
                    4.0 / 44100.0,
                    1.0,
                    4.0 / 44100.0,
                    registry.model_ref(handle).unwrap(),
                    None,
                    None,
                )
                .unwrap();
                PlaybackRegions::update_playback_region(&mut model, &mut 0, properties).unwrap();
                model.remap_and_log();
            }
            let rejected = owner.apply_link_edit(&link_id, &old);
            assert!(!rejected.ok);
            // 【为什么第三种改法报几何而不是代次】它改的正是 playback region 的时长 ——
            // 那就是宿主拥有的几何。判据先报几何并**指名字段**，比一句"代次对不上"
            // 更可行动（两者都以"在宿主里改完再重新导出"收尾）。
            // 前两种改的是素材属性/内容，几何不变，于是由代次判据拒绝。
            if change == 2 {
                assert_eq!(rejected.code, Some(link::error_code::GEOMETRY_CHANGED));
                assert!(
                    rejected.error.unwrap().contains("ara_host_fields:"),
                    "几何变了必须指名字段"
                );
            } else {
                assert_eq!(rejected.code, Some(link::error_code::CONFLICT));
                assert!(
                    rejected.error.unwrap().contains("host model is"),
                    "宿主模型代次变了必须被指名为冲突"
                );
            }
            clients
                .with_audio_source_management(source, |scope| {
                    AudioSources::enable_audio_source_samples_access(
                        &mut model, &mut slot, true, &scope,
                    )
                })
                .unwrap();
        }
        DocumentLifecycle::begin_editing(&mut model, &mut ()).unwrap();
        clients
            .with_audio_source_management(source, |scope| {
                AudioSources::enable_audio_source_samples_access(
                    &mut model, &mut slot, false, &scope,
                )
            })
            .unwrap();
        clients
            .with_audio_source_management(source, |scope| {
                AudioSources::enable_audio_source_samples_access(
                    &mut model, &mut slot, true, &scope,
                )
            })
            .unwrap();
        assert!(
            !model.session.ready.load(Ordering::Acquire),
            "逐对象创建/授权中不能提前认为完整图ready"
        );
        clients
            .with_audio_source_management(source, |scope| {
                DocumentLifecycle::end_editing(&mut model, &mut (), &scope)
            })
            .unwrap();
        assert!(model.session.ready.load(Ordering::Acquire));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// 宿主**不**按 leaf-first 销毁时，整个文档的身份仍然必须被撤销。
    ///
    /// 【为什么这是问题二的兜底回归】`PluginRuntime::destroy` 旧实现在"图非空"时
    /// **整份放弃**：一个 region 都没销毁 ⇒ `destroy_document` 不执行 ⇒
    /// `remove_document` 不执行 ⇒ 进程级表里留下死文档的 key。下一次文档在同一批
    /// 宿主管道域地址上 `create_playback_region` 直接撞 `DuplicateRegion`，
    /// 宿主从此不再交付任何 region，界面停在"等待宿主音频"。
    ///
    /// 兜底路径（T5 的强制销毁）保证"尽力销毁后一定调 destroy_document"；
    /// 而 `Drop`（模型直接释放、连 `destroy_document` 都没走）也必须撤销。
    #[test]
    fn dropping_a_model_with_live_regions_revokes_its_identities() {
        let identity = Box::new(0_u8);
        let key = (&*identity as *const u8) as u64;
        let document_id;
        {
            let model = ModelHandle::new();
            document_id = model.document_id;
            region_owners()
                .lock_or_recover()
                .register(key, model.document_id, 0)
                .unwrap();
            // 故意**不**调 `destroy_playback_region`：region 一直活着。
        }
        assert!(
            region_owners().lock_or_recover().owner_of(key).is_none(),
            "模型释放必须撤销文档身份，否则下一次文档会被遗留 key 卡住"
        );
        assert_ne!(document_id, 0, "文档身份必须非零（0 是无效身份）");
        // 被撤销的 key 可以被新文档认领。
        let replacement = ModelHandle::new();
        assert_eq!(
            region_owners()
                .lock_or_recover()
                .register(key, replacement.document_id, 0),
            Ok(())
        );
    }

    /// 实际模型销毁回调必须撤销索引，slot 表测试不能替代这个接线检查。
    #[test]
    fn destroying_a_model_region_revokes_its_global_identity() {
        let identity = Box::new(0_u8);
        let key = (&*identity as *const u8) as u64;
        let mut model = ModelHandle::new();
        model.region_keys.push(key);
        region_owners()
            .lock_or_recover()
            .register(key, model.document_id, 0)
            .unwrap();
        PlaybackRegions::destroy_playback_region(&mut model, 0);
        assert!(region_owners().lock_or_recover().resolve(&[key]).is_err());
    }

    /// 旧文档 Drop 的兜底不能撤销已被新文档复用地址的身份。
    #[test]
    fn document_teardown_and_fallback_drop_do_not_revoke_another_document() {
        let identity = Box::new(0_u8);
        let key = (&*identity as *const u8) as u64;
        let mut old = ModelHandle::new();
        region_owners()
            .lock_or_recover()
            .register(key, old.document_id, 0)
            .unwrap();
        DocumentLifecycle::destroy_document(&mut old, ());
        assert!(region_owners().lock_or_recover().resolve(&[key]).is_err());
        let replacement = ModelHandle::new();
        region_owners()
            .lock_or_recover()
            .register(key, replacement.document_id, 0)
            .unwrap();
        drop(old);
        assert_eq!(
            region_owners().lock_or_recover().resolve(&[key]).unwrap(),
            (replacement.document_id, vec![0])
        );
        drop(replacement);
        assert!(region_owners().lock_or_recover().resolve(&[key]).is_err());
    }

    /// 重开工程时宿主会复用同一批 region 地址，而上一份文档可能没有走完销毁。
    ///
    /// 【为什么必须是这条语义】这正是用户报的"重开只显示等待宿主音频、删掉插件再
    /// 添加就好"：key 是宿主持有的指针值，不按 leaf-first 销毁时遗留 key 一直留在
    /// 进程级表里；判成 `DuplicateRegion` 会让 `create_playback_region` 回 null，
    /// 宿主从此不再交付任何 region。删除插件让宿主重建 FX 槽、拿到不同指针值，
    /// 所以"删了再加就好"——这条测试把那个可Observable的事实钉住。
    #[test]
    fn a_reopened_document_reclaims_the_previous_documents_region_key() {
        let identity = Box::new(0_u8);
        let key = (&*identity as *const u8) as u64;
        let old = ModelHandle::new();
        region_owners()
            .lock_or_recover()
            .register(key, old.document_id, 7)
            .unwrap();
        // `old` 必须一直活着，否则它的 Drop 会把 key 撤销掉，这条语义就测不到了。
        let old_document = old.document_id;
        let reopened = ModelHandle::new();
        assert_ne!(old_document, reopened.document_id);
        assert_eq!(
            region_owners()
                .lock_or_recover()
                .register(key, reopened.document_id, 0),
            Ok(())
        );
        assert_eq!(
            region_owners().lock_or_recover().resolve(&[key]).unwrap(),
            (reopened.document_id, vec![0])
        );
        // 同一份文档内的重复仍然是真错误。
        assert_eq!(
            region_owners()
                .lock_or_recover()
                .register(key, reopened.document_id, 1),
            Err(crate::render::ownership::OwnershipError::DuplicateRegion)
        );
    }

    /// 宿主移动与拉伸更新必须替换现存区间，不能新增重复 clip 或忽略更新。
    #[test]
    fn host_region_update_changes_the_mapped_clip() {
        let mut model = ModelHandle::new();
        model.document = crate::ara::ara_document_from_json(include_str!(
            "../../tests/fixtures/ara-model.reaper.json"
        ))
        .unwrap();
        let mut index = 0;
        let mut sequences = Registry::<RegionSequenceKind, ()>::new(1);
        let handle = sequences.insert(()).unwrap();
        let sequence = sequences.model_ref(handle).unwrap();
        let properties = PlaybackRegionProperties::for_ara2(
            1,
            0.25,
            1.5,
            5.0,
            3.0,
            sequence,
            Some("moved"),
            None,
        )
        .unwrap();
        PlaybackRegions::update_playback_region(&mut model, &mut index, properties).unwrap();
        model.remap_and_log();
        let timeline = model.timeline().unwrap();
        assert_eq!(timeline.clips.len(), 1);
        let clip = &timeline.clips[0];
        assert_eq!(clip.start_sec, 5.0);
        assert_eq!(clip.length_sec, 3.0);
        assert_eq!(clip.source_start_sec, 0.25);
        assert_eq!(clip.playback_rate, 0.5);
        assert_eq!(clip.name, "moved");
        assert!(
            timeline.tracks.iter().all(|track| track.pitch_analysis_algo
                == hifishifter_kernel::state::PitchAnalysisAlgo::NsfHifiganOnnx),
            "插件新宿主轨道默认HiFiGAN"
        );
    }

    /// 删除区域后摘要和时间线只能包含存活区域，避免切片或撤销留下幽灵 clip。
    #[test]
    fn destroying_a_region_removes_it_from_the_timeline() {
        let mut model = ModelHandle::new();
        model.document = crate::ara::ara_document_from_json(include_str!(
            "../../tests/fixtures/ara-model.reaper.json"
        ))
        .unwrap();
        PlaybackRegions::destroy_playback_region(&mut model, 0);
        model.remap_and_log();
        assert!(model.timeline().unwrap().clips.is_empty());
    }

    /// **宿主 tempo map 的位置查询与四分音符→秒换算必须正确。**
    ///
    /// 【为什么这是回归测试】此前 `probe_host_musical_content` 只把条数记进日志，插件
    /// 只用 VST3 的单点 tempo —— 带 tempo map 的工程里，小节/拍网格只有 playhead 处对。
    /// 现在真读：tempo 点给出 `(秒, 四分音符)`，BPM 由相邻两点推出；而拍号点**只给
    /// 四分音符位置**，必须用 tempo 分段折成秒才能落到时间线上。
    #[test]
    fn host_tempo_and_bar_signatures_resolve_by_position() {
        // 120 BPM 一段：每四分音符 0.5 秒。四分音符 0 → 0.0 s，4 → 2.0 s。
        let raw = vec![(0.0, 0.0, 120.0), (2.0, 4.0, 120.0)];
        assert_eq!(quarters_to_seconds(&raw, 0.0), Some(0.0));
        assert_eq!(quarters_to_seconds(&raw, 2.0), Some(1.0));
        assert_eq!(quarters_to_seconds(&raw, 4.0), Some(2.0));
        // 末点之后按最后一段速度外推：四分音符 6 → 3.0 s。
        assert_eq!(quarters_to_seconds(&raw, 6.0), Some(3.0));
        // 首点之前夹到首点。
        assert_eq!(quarters_to_seconds(&raw, -1.0), Some(0.0));
        // 没有 tempo 点时无从换算。
        assert_eq!(quarters_to_seconds(&[], 1.0), None);

        let map = HostTempoMap {
            tempo: vec![(0.0, 120.0), (2.0, 60.0)],
            meter: vec![(0.0, 4, 4), (2.0, 3, 4)],
        };
        // 取"不晚于该时刻的最后一个点"。
        assert_eq!(map.bpm_at(-1.0), Some(120.0));
        assert_eq!(map.bpm_at(1.0), Some(120.0));
        assert_eq!(map.bpm_at(2.0), Some(60.0));
        assert_eq!(map.bpm_at(99.0), Some(60.0));
        assert_eq!(map.meter_at(0.5), Some((4, 4)));
        assert_eq!(map.meter_at(2.5), Some((3, 4)));
        assert!(!map.is_empty());
        assert!(HostTempoMap::default().is_empty());
        assert_eq!(HostTempoMap::default().bpm_at(0.0), None);
    }

    /// **mutation 回调必须立刻更新 session 的派生表，不等 `endEditing`。**
    ///
    /// 【为什么这是回归测试】`regions` / `clip_ids` 此前**只**由 `remap_and_log` 重写，
    /// 而它只跑在 `end_editing`。于是宿主在收口之前来读这两张表时，拿到的是上一代图 ——
    /// 渲染路径报 `missing assigned clip identity`、工作区投影漏掉新片段，用户看到
    /// "编辑操作不更新 ARA"。
    ///
    /// 桥接层的 `end_editing` 在 `content_readers` 非空时还会直接 return（长时间播放时
    /// 几乎总是如此），所以"等收口"并不总会发生。
    #[test]
    fn a_destroyed_region_leaves_the_session_tables_before_end_editing() {
        let mut model = ModelHandle::new();
        model.document = crate::ara::ara_document_from_json(include_str!(
            "../../tests/fixtures/ara-model.reaper.json"
        ))
        .unwrap();
        // 手动登记一个 key（`allocate_region` 在 create 时做，这里直接构造同一形状）。
        model.region_keys.push(4242);
        region_owners()
            .lock_or_recover()
            .register(4242, model.document_id, 0)
            .unwrap();
        model.remap_and_log();
        let session = model.session();
        assert!(
            session.clip_ids.lock_or_recover().contains_key(&4242),
            "映射之后 clip_ids 必须带上这个 region"
        );
        assert!(session.regions.lock_or_recover().contains_key(&4242));

        // 销毁它 —— **不**调 `remap_and_log`，模拟"宿主还没收口就来读"。
        PlaybackRegions::destroy_playback_region(&mut model, 0);
        assert!(
            !session.regions.lock_or_recover().contains_key(&4242),
            "派生表必须在销毁那一刻就反映新图，不能等 endEditing"
        );
        assert!(
            !session.clip_ids.lock_or_recover().contains_key(&4242),
            "clip_ids 同理：它已按 key 逐条清理，派生表不能把它加回来"
        );

        // 而身份仍然可由 key 派生（B1 的降级路径），不会变成致命错误。
        assert_eq!(
            session.clip_id_for(4242).as_deref(),
            Some("ara-region-4242")
        );
    }

    /// **被销毁的 region sequence 必须从派生表里消失。**
    ///
    /// 【为什么这是回归测试】`destroy_region_sequence` 此前落 trait 默认空实现，
    /// 而 `region_sequences` 是 push-only 的：宿主删除一条轨道后，它的身份与
    /// `params_by_root_track` 条目一直留到文档关闭 —— 用户看到插件里有一条已经
    /// 不存在的轨，且它的参数还在参与渲染。
    ///
    /// 【为什么不走 `create_region_sequence`】`CreateContext` 由桥接层在真实回调时构造，
    /// 测试里无法合成（它持有运行时的借用）。这里直接摆出"创建之后"的状态，只验销毁。
    #[test]
    fn a_destroyed_region_sequence_leaves_the_session_tables() {
        let mut model = ModelHandle::new();
        let session = model.session();
        // 摆出"宿主创建了两条序列"之后的状态：`sequence_track_index` 有两条键。
        session
            .sequence_track_index
            .lock_or_recover()
            .insert(0xAAAA, 0);
        session
            .sequence_track_index
            .lock_or_recover()
            .insert(0xBBBB, 1);
        model.sequence_persistent_ids = vec![(0xAAAA).to_string(), (0xBBBB).to_string()];
        // T7 的原料：`region_sequence_of` **首选**按 `sequence_model_ref` 解析，
        // 所以"销毁"必须同时从 ref 表与反向表里抹掉，否则存活 region 会继续
        // 解析到这个语义已变的槽位（"分割后多出一条轨"）。
        model.document.sequence_refs = vec![0xAAAA, 0xBBBB];
        model.sequence_index_by_model_ref.insert(0xAAAA, 0);
        model.sequence_index_by_model_ref.insert(0xBBBB, 1);

        // 销毁下标 0 的那条：键 0xAAAA 与下标 0 都必须消失，下标 1 的必须留下。
        RegionSequences::destroy_region_sequence(&mut model, 0);

        assert_eq!(
            model.document.sequence_refs,
            vec![0, 0xBBBB],
            "已销毁槽位的 ref 必须清零（下标保持对齐），否则旧 ref 仍可被解析"
        );
        assert!(
            !model.sequence_index_by_model_ref.contains_key(&0xAAAA),
            "反向表必须同时清理旧 key"
        );
        assert_eq!(
            model.sequence_index_by_model_ref.get(&0xBBBB),
            Some(&1),
            "另一条序列的映射不能受影响"
        );

        let index = session.sequence_track_index.lock_or_recover();
        assert!(
            !index.contains_key(&0xAAAA),
            "销毁序列必须同时清掉参数根边，否则已删除的轨仍会被认成参数根"
        );
        assert!(
            !index.values().any(|value| *value == 0),
            "指向该下标的边也要清掉"
        );
        assert!(
            index.contains_key(&0xBBBB) && index.values().any(|value| *value == 1),
            "另一条序列不能受影响"
        );
        assert_eq!(
            model.sequence_persistent_ids[0], "",
            "该槽的宿主身份必须抹掉，映射层随即退回位置身份"
        );
    }

    /// 首次映射就失败后，`ready` 必须能在**下一次**成功编辑上自愈，而不是砖化。
    ///
    /// 【为什么必须走生产路径】此前已有的用例手动 `ready.store(true)` 绕过
    /// `prepare_renderers` —— 那只证明"标志能被置真"，没证明"失败之后还能自己回来"。
    ///
    /// 【为什么不再用"声明数量对不上"当致命触发】该输入现在按顺序归属**成功映射**
    /// （见 `region_sequence_of`），不再整份失败 —— 因此这里改为钉住新的、更重要的性质：
    /// 归属信息缺失时**不砖化实例**，片段照常发布且 `ready` 为真。
    #[test]
    fn missing_sequence_attribution_publishes_clips_without_bricking_the_instance() {
        use std::sync::atomic::Ordering;

        let mut model = ModelHandle::new();
        // 声明 1 条 region，扁平列表却有 2 条，且都缺 `regionSequenceIndex` ⇒
        // 多出来的 region 归到首条序列，映射成功（宿主不填 count 也要能显示片段）。
        model.document = crate::ara::ara_document_from_json(
            r#"{
                "audioSources": [ { "persistentID": "s", "sampleRate": 44100.0 } ],
                "audioModifications": [
                    { "persistentID": "m", "audioSourcePersistentID": "s" }
                ],
                "musicalContexts": [ { "regionSequences": [ { "playbackRegionCount": 1 } ] } ],
                "playbackRegions": [
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "durationInPlaybackTime": 1.0 },
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "durationInPlaybackTime": 1.0 }
                ]
            }"#,
        )
        .unwrap();

        model.remap_and_log();
        let timeline = model.timeline().expect("归属缺失不得整份失败");
        assert_eq!(timeline.clips.len(), 2);
        assert!(
            model.session.ready.load(Ordering::Acquire),
            "成功映射必须经 `prepare_renderers` 把 `ready` 置真"
        );
    }

    /// 已销毁的槽位必须被**复用**：长会话里三份结构不得随"编辑次数"无界增长。
    #[test]
    fn destroyed_region_slots_are_reused_instead_of_growing_the_vectors() {
        // 用**存活**的堆地址作键，避免与并行运行的其他用例在进程级 `region_owners`
        // 上撞车（那里按 key 唯一登记）。
        let owners = [
            Box::new(0_u8),
            Box::new(0_u8),
            Box::new(0_u8),
            Box::new(0_u8),
        ];
        let key = |index: usize| (&*owners[index] as *const u8) as u64;

        let mut model = ModelHandle::new();
        for slot in 0..3 {
            model
                .allocate_region(key(slot), AraPlaybackRegion::default())
                .unwrap();
        }
        assert_eq!(model.region_keys.len(), 3);
        assert_eq!(model.document.playback_regions.len(), 3);

        // 销毁中间一个：下标不压缩（存活 region 的身份不变），槽位 1 进入待复用集合。
        PlaybackRegions::destroy_playback_region(&mut model, 1);
        assert_eq!(model.destroyed_regions, HashSet::from([1_usize]));
        assert_eq!(model.region_keys.len(), 3, "销毁不得压缩下标");

        // 再建一个：必须复用槽位 1，而不是把向量撑到 4。
        let index = model
            .allocate_region(key(3), AraPlaybackRegion::default())
            .unwrap();
        assert_eq!(index, 1);
        assert_eq!(model.region_keys.len(), 3, "槽位复用不得增长向量");
        assert_eq!(model.document.playback_regions.len(), 3);
        assert!(model.destroyed_regions.is_empty());
        // 复用的是**销毁过的**槽位；存活 region 的键一个都没动。
        assert_eq!(model.region_keys[0], key(0));
        assert_eq!(model.region_keys[1], key(3));
        assert_eq!(model.region_keys[2], key(2));
    }

    /// **没有 `endEditing` 时，`notifyModelUpdates` 也必须能发布模型。**
    ///
    /// 【为什么这是问题二的回归】宿主在工程加载/后台恢复时可能直接重建图却不发
    /// begin/end。`model_dirty` + `flush_pending_model_updates` 是那条通路的收口点；
    /// 没有它，`document.timeline` 永远是 None，region→item 认领与参数恢复全部停住。
    #[test]
    fn a_model_update_without_end_editing_still_publishes_the_timeline() {
        let mut model = ModelHandle::new();
        model.document = identity_document(&["B"]);
        let key = 5150_u64;
        model.region_keys.push(key);
        region_owners()
            .lock_or_recover()
            .register(key, model.document_id, 0)
            .unwrap();
        assert!(model.timeline.is_none(), "前提：还没有发布过时间线");
        model.mark_model_dirty();
        assert!(
            DocumentLifecycle::flush_pending_model_updates(&mut model),
            "dirty 模型必须被收口"
        );
        assert!(model.timeline.is_some(), "没有 endEditing 也必须发布时间线");
        assert!(
            !DocumentLifecycle::flush_pending_model_updates(&mut model),
            "没有新变化时不得重复重算"
        );
    }
}
