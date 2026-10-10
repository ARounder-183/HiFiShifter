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
    head_tail: Arc<ara2_bridge::plugin::RealtimeHeadTailAdapter>,
    /// 正在累积的 ARA 文档。
    document: AraDocument,
    /// 映射好的时间线（`end_editing` 之后可用）。
    timeline: Option<TimelineState>,
    /// 协商到的 ARA 版本。
    generation: Option<ApiGeneration>,
    /// 源授权可能在宿主事务中回调；完整图收口前不能准备/关联恢复编辑。
    editing: bool,
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
}

impl Default for ModelHandle {
    fn default() -> Self {
        static NEXT_DOCUMENT: AtomicU64 = AtomicU64::new(1);
        let document_id = NEXT_DOCUMENT.fetch_add(1, Ordering::Relaxed);
        Self {
            document_id,
            session: DocumentSession::new(document_id),
            region_keys: Vec::new(),
            head_tail: Arc::new(
                ara2_bridge::plugin::RealtimeHeadTailAdapter::new(8192)
                    .expect("constant head/tail capacity"),
            ),
            document: AraDocument::default(),
            timeline: None,
            generation: None,
            editing: false,
            sequence_index_by_handle: HashMap::new(),
            source_index_by_handle: HashMap::new(),
            modification_index_by_handle: HashMap::new(),
            content_versions: Vec::new(),
            destroyed_regions: HashSet::new(),
            sequence_index_by_model_ref: HashMap::new(),
        }
    }
}

impl ModelHandle {
    pub(crate) fn head_tail(&self) -> Arc<ara2_bridge::plugin::RealtimeHeadTailAdapter> {
        self.head_tail.clone()
    }
    /// 本应用渐变位于clip边界内，没有额外head/tail；查询使用SDK的零分配快照接口。
    fn publish_head_tail(&self) -> Result<(), AraError> {
        let entries = self
            .region_keys
            .iter()
            .enumerate()
            .filter(|(index, _)| !self.destroyed_regions.contains(index))
            .map(|(_, key)| ara2_bridge::core::HeadTailEntry::new(*key, 0., 0.))
            .collect::<Result<Vec<_>, _>>()?;
        self.head_tail.install(entries)
    }
    /// 借出本模型的文档生命周期，供工厂身份通知及 renderer 绑定使用。
    pub(crate) fn session(&self) -> Arc<DocumentSession> {
        self.session.clone()
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
            .map_err(|_| AraError::InvalidState("region identity already owned or invalid"))?;
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

    /// 把当前文档映射成时间线并记一行摘要。
    ///
    /// 【为什么在 `end_editing` 调】ARA 约定：一次编辑的所有图变更都在
    /// `begin_editing` … `end_editing` 之间，收口点之后模型才是稳定的。
    fn remap_and_log(&mut self) {
        let session = self.session.clone();
        let transaction = session.transaction.lock().unwrap();
        {
            let mut regions = self.session.regions.lock().unwrap();
            regions.clear();
            for (slot, region) in self.document.playback_regions.iter().enumerate() {
                if !self.destroyed_regions.contains(&slot) {
                    if let Some(key) = self.region_keys.get(slot) {
                        regions.insert(*key, region.clone());
                    }
                }
            }
        }
        let mut document = self.document.clone();
        document.playback_regions = document
            .playback_regions
            .into_iter()
            .enumerate()
            .filter(|(index, _)| !self.destroyed_regions.contains(index))
            .map(|(_, region)| region)
            .collect();
        match crate::ara::mapping::ara_document_to_timeline_reporting(&document) {
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
                for track in &mut timeline.tracks {
                    track.compose_enabled = true;
                    track.pitch_analysis_algo =
                        hifishifter_kernel::state::PitchAnalysisAlgo::NsfHifiganOnnx;
                }
                // 宿主本会话slot不随删除压缩；避免活着的clip因旧region销毁换身份。
                //
                // 【为什么按 `clip_regions` 对齐而不是顺序 zip】`ara_document_to_timeline`
                // 现在会逐 region 跳过坏条目；顺序 zip 会在有跳过时整体错位，把身份挂到
                // 错误的 clip 上。`clip_regions[i]` 是该 clip 在**过滤后**文档里的下标，
                // 与 `live_keys` 同序。
                let live_keys: Vec<u64> = self
                    .region_keys
                    .iter()
                    .enumerate()
                    .filter(|(slot, _)| !self.destroyed_regions.contains(slot))
                    .map(|(_, key)| *key)
                    .collect();
                let mut identities = self.session.clip_ids.lock().unwrap();
                identities.clear();
                for (clip, region_index) in timeline.clips.iter_mut().zip(&clip_regions) {
                    let Some(key) = live_keys.get(*region_index) else {
                        continue;
                    };
                    clip.id = format!("ara-clip-{}", region_index + 1);
                    identities.insert(*key, clip.id.clone());
                }
                drop(identities);
                // 身份仅由真实region→sequence边与宿主persistentID构建，不从名字/序号推断。
                let mut bindings: crate::state_channel::TrackBindings = timeline
                    .tracks
                    .iter()
                    .map(|t| (t.id.clone(), vec![]))
                    .collect();
                for region in &document.playback_regions {
                    if let Some(index) = region.region_sequence_index {
                        if let Some(track) = timeline.tracks.get(index) {
                            bindings.get_mut(&track.id).unwrap().push((
                                region.audio_modification_persistent_id.clone(),
                                region.audio_source_persistent_id.clone(),
                            ));
                        }
                    }
                }
                for members in bindings.values_mut() {
                    members.sort();
                    members.dedup();
                }
                *self.session.track_bindings.lock().unwrap() = bindings.clone();
                // getter可能重入ARA：先发布真实基础图、释放事务，再采集稳定item GUID。
                *self.session.timeline.lock().unwrap() = Some(timeline.clone());
                drop(transaction);
                for owner in self.session.renderer_owners() {
                    owner.refresh_reaper_state_for_model();
                }
                for owner in self.session.renderer_owners() {
                    owner.refresh_ui_inventory();
                }
                let transaction = session.transaction.lock().unwrap();
                {
                    let items = self.session.region_items.lock().unwrap();
                    let mut ids = self.session.clip_ids.lock().unwrap();
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
                    let items = self.session.region_items.lock().unwrap().clone();
                    let take_by_item = {
                        let tracks = self.session.ui_tracks.lock().unwrap();
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
                        let tracks = self.session.ui_tracks.lock().unwrap();
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
                    let mut takes = self.session.authorized_takes.lock().unwrap();
                    let mut sources = self.session.authorized_sources.lock().unwrap();
                    takes.clear();
                    sources.clear();
                    for (key, item) in items.iter() {
                        if let Some(take) = take_by_item.get(item) {
                            takes.insert(*key, take.clone());
                        }
                        if let Some(source) = source_by_item.get(item) {
                            sources.insert(*key, source.clone());
                        }
                    }
                }
                {
                    let mut edits = self.session.edits.lock().unwrap();
                    if !edits.atlas.is_empty() && !edits.needs_rebind {
                        let projection = (|| {
                            let identities = self.session.parameter_identities_locked(&timeline)?;
                            let roots = edits.atlas.project_roots(&timeline, &identities)?;
                            let followed = edits.atlas.follow_geometry(&timeline, &identities)?;
                            Ok::<_, String>((roots, followed))
                        })();
                        match projection {
                            Ok((roots, followed)) => {
                                // 无活动region的轨道可能只是mute；保留其曲线，不能当删除清空。
                                edits.params.extend(roots);
                                edits.atlas = followed;
                                edits
                                    .tracks
                                    .retain(|track| bindings.contains_key(&track.id));
                            }
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
                    }
                }
                if let Err(error) = self.session.edits.lock().unwrap().reconcile(&bindings) {
                    log::error!("[ara] edit identity unresolved: {error}");
                }
                log::info!("{}", crate::ara::summary_line(&document, &timeline));
                log::info!("{}", crate::ara::clip_starts_line(&timeline));
                *self.session.timeline.lock().unwrap() = Some(timeline.clone());
                self.timeline = Some(timeline);
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
            }
        }
        self.session.prepare_renderers();
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
        *self.session.generation.lock().unwrap() = self.generation;
        self.document.document_name = properties.name().unwrap_or("").to_string();
        log::info!(
            "[ara] document controller created: apiGeneration={:?}",
            context.generation()
        );
        Ok(())
    }

    /// 记录宿主编辑会话，区分宿主未通知与委托回调未处理。
    fn begin_editing(&mut self, _document: &mut Self::Document) -> Result<(), AraError> {
        self.editing = true;
        self.session.clear_renderers();
        log::info!("[ara] begin_editing");
        Ok(())
    }

    fn end_editing(
        &mut self,
        _document: &mut Self::Document,
        _host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        self.editing = false;
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
}

impl MusicalContexts for ModelHandle {
    type MusicalContext = ();

    fn create_musical_context(
        &mut self,
        _context: &CreateContext,
        properties: MusicalContextProperties,
        host: &HostContentScope<'_, '_>,
    ) -> Result<Self::MusicalContext, AraError> {
        self.document.musical_contexts.push(AraMusicalContext {
            name: properties.name().map(str::to_owned),
            region_sequences: Vec::new(),
        });
        probe_host_musical_content(host);
        Ok(())
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
fn probe_host_musical_content(host: &HostContentScope<'_, '_>) {
    let Some(context) = host.current_musical_context() else {
        log::info!("[ara][probe] musical content: no current context in this scope");
        return;
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
            Ok(grade) => log::info!("[ara][probe] musical content {name}: grade={grade:?}"),
            Err(error) => {
                log::info!("[ara][probe] musical content {name}: unavailable ({error})")
            }
        }
    }
    match host.musical_context::<Tempo>(context, None) {
        Ok(reader) => log::info!("[ara][probe] tempo entries: {}", reader.len()),
        Err(error) => log::info!("[ara][probe] tempo entries unavailable: {error}"),
    }
    match host.musical_context::<BarSignatures>(context, None) {
        Ok(reader) => log::info!("[ara][probe] bar signatures: {}", reader.len()),
        Err(error) => log::info!("[ara][probe] bar signatures unavailable: {error}"),
    }
}

impl RegionSequences for ModelHandle {
    type RegionSequence = usize;

    fn create_region_sequence(
        &mut self,
        context: &CreateContext,
        properties: RegionSequenceProperties,
    ) -> Result<Self::RegionSequence, AraError> {
        // 归到第一个（通常也是唯一一个）musical context 下；没有就现建一个。
        if self.document.musical_contexts.is_empty() {
            self.document.musical_contexts.push(AraMusicalContext {
                name: None,
                region_sequences: Vec::new(),
            });
        }
        let index = self
            .document
            .musical_contexts
            .iter()
            .map(|context| context.region_sequences.len())
            .sum();
        let context_index = self.document.musical_contexts.len() - 1;
        let sequences = &mut self.document.musical_contexts[context_index].region_sequences;
        sequences.push(AraRegionSequence {
            name: properties.name().map(str::to_owned),
            order_index: properties.order_index(),
            playback_region_count: 0,
        });
        if let Some(handle) = context.object_handle() {
            self.sequence_index_by_handle.insert(handle, index);
        }
        if let Some(key) = context.realtime_key() {
            self.sequence_index_by_model_ref.insert(key as usize, index);
            self.session
                .sequence_regions
                .lock()
                .unwrap()
                .insert(key, HashSet::new());
            // 记录 sequence → 轨道下标：空 sequence（无 region）也要能被认成参数根。
            self.session
                .sequence_track_index
                .lock()
                .unwrap()
                .insert(key, index);
        }
        Ok(index)
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
        let source = self
            .document
            .audio_sources
            .get_mut(*state)
            .ok_or(AraError::InvalidArgument("unknown source"))?;
        self.session.clear_renderers();
        self.session
            .sources
            .lock()
            .unwrap()
            .remove(&source.persistent_id);
        self.session
            .edit_sources
            .lock()
            .unwrap()
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

    /// undo history 中停用源时立即撤销准备好的 PCM；恢复后等待宿主重新授予访问。
    fn deactivate_audio_source(
        &mut self,
        state: &mut Self::AudioSource,
        deactivate: bool,
        host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        if deactivate {
            self.session.clear_renderers();
            if let Some(source) = self.document.audio_sources.get(*state) {
                self.session
                    .edit_sources
                    .lock()
                    .unwrap()
                    .remove(&source.persistent_id);
            }
            if let Some(source) = self.document.audio_sources.get_mut(*state) {
                source.sample_access_enabled = false;
            }
            self.refresh_source_pcm(*state, host);
        }
        Ok(())
    }

    /// 源销毁时撤销访问和快照，不能继续播已销毁源的缓存。
    fn destroy_audio_source(&mut self, state: Self::AudioSource, host: &HostContentScope<'_, '_>) {
        self.session.clear_renderers();
        if let Some(source) = self.document.audio_sources.get(state) {
            self.session
                .edit_sources
                .lock()
                .unwrap()
                .remove(&source.persistent_id);
        }
        if let Some(source) = self.document.audio_sources.get_mut(state) {
            source.sample_access_enabled = false;
        }
        self.refresh_source_pcm(state, host);
    }

    fn create_audio_source(
        &mut self,
        context: &CreateContext,
        properties: AudioSourceProperties,
        _host: &HostContentScope<'_, '_>,
    ) -> Result<Self::AudioSource, AraError> {
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
        self.session.clear_renderers();
        if let Some(source) = self.document.audio_sources.get(*state) {
            self.session
                .edit_sources
                .lock()
                .unwrap()
                .remove(&source.persistent_id);
        }
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
        log::info!("[ara] audio_source samples_access enable={enable}");
        self.refresh_source_pcm(*state, host);
        if enable && std::env::var_os("HIFISHIFTER_ARA_PCM_PROBE").is_some() {
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
    fn refresh_source_pcm(&mut self, index: usize, host: &HostContentScope<'_, '_>) {
        self.session.revoke_renderers();
        let Some(source) = self.document.audio_sources.get(index) else {
            return;
        };
        self.session
            .sources
            .lock()
            .unwrap()
            .remove(&source.persistent_id);
        if source.sample_access_enabled {
            let result = usize::try_from(source.sample_count)
                .ok()
                .filter(|_| source.sample_rate == source.sample_rate.round())
                .ok_or(AraError::InvalidArgument("unsupported sample geometry"))
                .and_then(|frames| {
                    crate::render::source::read_source_pcm(
                        host,
                        frames,
                        source.channel_count as usize,
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
                    self.session
                        .publish_source_pcm(source.persistent_id.clone(), pcm);
                }
                Err(error) => log::warn!("[ara] host PCM unavailable: {error:?}"),
            }
        }
        if !self.editing {
            self.session.prepare_renderers();
        }
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
                persistent_id,
                audio_source_persistent_id,
            });
        if let Some(handle) = context.object_handle() {
            self.modification_index_by_handle.insert(handle, index);
        }
        Ok(index)
    }

    fn clone_audio_modification(
        &mut self,
        _context: &CreateContext,
        source: &Self::AudioModification,
        audio_source: &Self::AudioSource,
        properties: AudioModificationProperties,
    ) -> Result<Self::AudioModification, AraError> {
        // 克隆出的 modification 基于同一个 source —— 这正是"同一素材被复制成多份"
        // 在 ARA 里的形状，映射层会把它变成同一个 `Clip.source_path` 下的多个 clip。
        let mut cloned = self
            .document
            .audio_modifications
            .get(*source)
            .cloned()
            .unwrap_or_default();
        cloned.persistent_id = properties.persistent_id().to_string();
        cloned.audio_source_persistent_id = self
            .document
            .audio_sources
            .get(*audio_source)
            .map(|s| s.persistent_id.clone())
            .unwrap_or_else(|| cloned.audio_source_persistent_id.clone());
        let index = self.document.audio_modifications.len();
        self.document.audio_modifications.push(cloned);
        Ok(index)
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
                .lock()
                .unwrap()
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
        Ok(index)
    }

    /// 宿主移动、裁切或拉伸已有 item 时，替换几何属性并保留所属源与修改。
    fn update_playback_region(
        &mut self,
        state: &mut Self::PlaybackRegion,
        properties: PlaybackRegionProperties,
    ) -> Result<(), AraError> {
        self.session.clear_renderers();
        if let Some(sequence) = properties.region_sequence() {
            let sequence_key = sequence.as_raw() as usize;
            if let Some(index) = self.sequence_index_by_model_ref.get(&sequence_key).copied() {
                if let Some(key) = self.region_keys.get(*state) {
                    let mut members = self.session.sequence_regions.lock().unwrap();
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
        let flags = properties.transformation_flags();
        region.name = properties.name().map(str::to_owned);
        region.start_in_modification_time = properties.start_in_modification_time();
        region.duration_in_modification_time = properties.duration_in_modification_time();
        region.start_in_playback_time = properties.start_in_playback_time();
        region.duration_in_playback_time = properties.duration_in_playback_time();
        region.is_timestretch_enabled = (flags & 1) != 0;
        region.has_content_based_fade_at_head = (flags & 8) != 0;
        region.has_content_based_fade_at_tail = (flags & 4) != 0;
        log::info!(
            "[ara] playback_region updated #{}: startPlay={:.6}",
            state,
            region.start_in_playback_time
        );
        Ok(())
    }

    /// 标记已销毁区间，编辑结束时只映射存活的区域。
    fn destroy_playback_region(&mut self, state: Self::PlaybackRegion) {
        self.session.clear_renderers();
        if let Some(key) = self.region_keys.get(state) {
            region_owners()
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .remove(*key);
            for regions in self.session.sequence_regions.lock().unwrap().values_mut() {
                regions.remove(key);
            }
        }
        self.destroyed_regions.insert(state);
        if let Err(error) = self.publish_head_tail() {
            log::error!("head/tail snapshot failed: {error}");
        }
        log::info!("[ara] playback_region destroyed #{}", state);
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
            .lock()
            .unwrap()
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
        let projection = document.workspace_projection().unwrap();
        document
            .accept_workspace_edits(
                0,
                document.revision.load(Ordering::Acquire),
                &client,
                &projection,
            )
            .unwrap();
        assert!(!document.edits.lock().unwrap().atlas.is_empty());
        model.document.playback_regions[0].start_in_playback_time = 1.;
        model.document.playback_regions[0].duration_in_playback_time = 1.;
        model.document.playback_regions[0].is_timestretch_enabled = true;
        model.remap_and_log();
        let pitch = document.edits.lock().unwrap().params[&root]
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
        *model.session.edits.lock().unwrap() = crate::state_channel::EditState::default()
            .merge(&view, &client, 0)
            .unwrap();
        // 同live图中删A区间，但B的会话slot不压缩。
        model.destroyed_regions.insert(0);
        model.remap_and_log();
        let bytes = model.session.edits.lock().unwrap().encode().unwrap();
        for order in [vec!["B"], vec!["B", "A"]] {
            let mut reopened = ModelHandle::new();
            reopened.document = identity_document(&order);
            reopened
                .session
                .edits
                .lock()
                .unwrap()
                .restore(&bytes)
                .unwrap();
            reopened.remap_and_log();
            let mut timeline = reopened.timeline().unwrap().clone();
            reopened.session.edits.lock().unwrap().apply(&mut timeline);
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
            .lock()
            .unwrap()
            .insert(old_key, [region_key].into_iter().collect());
        model
            .session
            .sequence_regions
            .lock()
            .unwrap()
            .insert(new_key, HashSet::new());
        let properties =
            PlaybackRegionProperties::for_ara2(1, 0.0, 1.0, 2.0, 1.0, new, None, None).unwrap();
        PlaybackRegions::update_playback_region(&mut model, &mut 0, properties).unwrap();
        assert_eq!(
            model.document.playback_regions[0].region_sequence_index,
            Some(1)
        );
        assert!(model.session.sequence_regions.lock().unwrap()[&old_key].is_empty());
        assert_eq!(
            model.session.sequence_regions.lock().unwrap()[&new_key],
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
            model.session.sources.lock().unwrap()[&id].planes[0],
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
            model.session.sources.lock().unwrap()[&id].planes[0],
            [0.5, 0.6, 0.7, 0.8]
        );
        assert_eq!(model.session.sources.lock().unwrap()[&id].version, 1);
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
        assert!(model.session.sources.lock().unwrap().is_empty());
        assert_eq!(
            model.session.edit_sources.lock().unwrap()[&id].planes[0],
            [0.5, 0.6, 0.7, 0.8]
        );
        clients
            .with_audio_source_management(source, |scope| {
                AudioSources::deactivate_audio_source(&mut model, &mut slot, true, &scope)
            })
            .unwrap();
        assert!(model.session.edit_sources.lock().unwrap().is_empty());
        assert!(
            model.session.revision.load(Ordering::Acquire) > before_revoke,
            "停用确实改变模型"
        );
    }

    /// 同一GUI快照经历真实授权开关可提交；实际源几何/内容通知仍使旧提交失效。
    #[test]
    fn same_snapshot_commit_survives_access_toggle_but_not_real_source_changes() {
        use ara2_bridge::plugin::{ExtensionRoles, HostAudioSourceRef, HostClients};
        use hifishifter_ara_ipc::{Request, Response};
        let mut model = ModelHandle::new();
        model.document = identity_document(&["B"]);
        model.document.audio_sources[0].sample_count = 4;
        model.document.playback_regions[0].duration_in_modification_time = 4.0 / 44100.0;
        model.document.playback_regions[0].duration_in_playback_time = 4.0 / 44100.0;
        let region = Box::new(0_u8);
        let key = (&*region as *const u8) as u64;
        model.region_keys.push(key);
        region_owners()
            .lock()
            .unwrap()
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
        let snapshot = owner.handle_request(Request::Snapshot);
        assert!(snapshot.ok, "{:?}", snapshot.error);
        let request = |snapshot: Response| {
            let mut client = snapshot.timeline.unwrap();
            client["tracks"][0]["volume"] = serde_json::json!(0.5);
            Request::Commit {
                base_revision: snapshot.revision,
                model_revision: snapshot.model_revision,
                timeline: client,
            }
        };
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
        let committed = owner.handle_request(request(snapshot));
        assert!(committed.ok, "{:?}", committed.error);
        for change in 0..3 {
            let old = owner.handle_request(Request::Snapshot);
            assert!(old.ok, "{:?}", old.error);
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
            let rejected = owner.handle_request(request(old));
            assert!(!rejected.ok);
            assert!(rejected.error.unwrap().contains("host model changed"));
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
    }

    /// 实际模型销毁回调必须撤销索引，slot 表测试不能替代这个接线检查。
    #[test]
    fn destroying_a_model_region_revokes_its_global_identity() {
        let identity = Box::new(0_u8);
        let key = (&*identity as *const u8) as u64;
        let mut model = ModelHandle::new();
        model.region_keys.push(key);
        region_owners()
            .lock()
            .unwrap()
            .register(key, model.document_id, 0)
            .unwrap();
        PlaybackRegions::destroy_playback_region(&mut model, 0);
        assert!(region_owners().lock().unwrap().resolve(&[key]).is_err());
    }

    /// 旧文档 Drop 的兜底不能撤销已被新文档复用地址的身份。
    #[test]
    fn document_teardown_and_fallback_drop_do_not_revoke_another_document() {
        let identity = Box::new(0_u8);
        let key = (&*identity as *const u8) as u64;
        let mut old = ModelHandle::new();
        region_owners()
            .lock()
            .unwrap()
            .register(key, old.document_id, 0)
            .unwrap();
        DocumentLifecycle::destroy_document(&mut old, ());
        assert!(region_owners().lock().unwrap().resolve(&[key]).is_err());
        let replacement = ModelHandle::new();
        region_owners()
            .lock()
            .unwrap()
            .register(key, replacement.document_id, 0)
            .unwrap();
        drop(old);
        assert_eq!(
            region_owners().lock().unwrap().resolve(&[key]).unwrap(),
            (replacement.document_id, vec![0])
        );
        drop(replacement);
        assert!(region_owners().lock().unwrap().resolve(&[key]).is_err());
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

    /// 首次映射就失败后，`ready` 必须能在**下一次**成功编辑上自愈，而不是砖化。
    ///
    /// 【为什么必须走生产路径】此前已有的用例手动 `ready.store(true)` 绕过
    /// `prepare_renderers` —— 那只证明"标志能被置真"，没证明"失败之后还能自己回来"。
    /// 这里让第一次映射真的以**文档级致命错误**失败（region 数量与序列声明对不上），
    /// 从未发布过时间线 ⇒ `ready=false`；随后喂一份可映射的文档，`ready` 必须由
    /// `prepare_renderers` 自己从 `false` 回到 `true`，否则宿主不重发 `end_editing`
    /// 时实例会永久失效。
    #[test]
    fn a_failed_first_mapping_recovers_on_the_next_successful_edit() {
        use std::sync::atomic::Ordering;

        let mut model = ModelHandle::new();
        // 声明 1 条 region，扁平列表却有 2 条，且都缺 `regionSequenceIndex` ⇒
        // `SequenceCountMismatch`（文档级致命错误，逐 region 跳过救不了）。
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
        assert!(model.timeline().is_none(), "致命错误不得发布半份时间线");
        assert!(
            !model.session.ready.load(Ordering::Acquire),
            "从未成功映射过 ⇒ 未就绪"
        );

        // 宿主随后给出可映射的文档：下一次收口必须把实例自己拉回可用状态。
        model.document = identity_document(&["A"]);
        model.remap_and_log();
        assert_eq!(model.timeline().unwrap().clips.len(), 1);
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
}
