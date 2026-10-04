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
use ara2_bridge::core::{
    ApiGeneration, AraError, AudioModificationProperties, AudioSourceProperties,
    ContentTimeRange, ContentUpdateScopes, DocumentProperties, MusicalContextProperties,
    PlaybackRegionProperties, RegionSequenceProperties,
};
use ara2_bridge::plugin::{
    AudioModifications, AudioSources, CreateContext, DocumentLifecycle, HostContentScope,
    MusicalContexts, PlaybackRegions, RegionSequences,
};
use hifishifter_kernel::state::TimelineState;
use std::collections::{HashMap, HashSet};

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
    /// 正在累积的 ARA 文档。
    document: AraDocument,
    /// 映射好的时间线（`end_editing` 之后可用）。
    timeline: Option<TimelineState>,
    /// 协商到的 ARA 版本。
    generation: Option<ApiGeneration>,
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
        Self {
            document: AraDocument::default(),
            timeline: None,
            generation: None,
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
    /// 新建一份空模型。
    pub fn new() -> Self {
        Self::default()
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
        let mut document = self.document.clone();
        document.playback_regions = document.playback_regions.into_iter()
            .enumerate()
            .filter(|(index, _)| !self.destroyed_regions.contains(index))
            .map(|(_, region)| region)
            .collect();
        match crate::ara::mapping::ara_document_to_timeline(&document) {
            Ok(timeline) => {
                log::info!("{}", crate::ara::summary_line(&document, &timeline));
                log::info!("{}", crate::ara::clip_starts_line(&timeline));
                self.timeline = Some(timeline);
            }
            Err(err) => {
                // 映射失败必须显式记录：它是"DAW 里看到的时间线不对"这类问题的唯一线索。
                log::error!("[ara] mapping failed: {err:?}");
            }
        }
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
        self.document.document_name = properties.name().unwrap_or("").to_string();
        log::info!(
            "[ara] document controller created: apiGeneration={:?}",
            context.generation()
        );
        Ok(())
    }

    /// 记录宿主编辑会话，区分宿主未通知与委托回调未处理。
    fn begin_editing(&mut self, _document: &mut Self::Document) -> Result<(), AraError> {
        log::info!("[ara] begin_editing");
        Ok(())
    }

    fn end_editing(
        &mut self,
        _document: &mut Self::Document,
        _host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        self.remap_and_log();
        Ok(())
    }

    fn destroy_document(&mut self, _document: Self::Document) {
        log::info!("[ara] document destroyed");
    }
}

impl MusicalContexts for ModelHandle {
    type MusicalContext = ();

    fn create_musical_context(
        &mut self,
        _context: &CreateContext,
        properties: MusicalContextProperties,
        _host: &HostContentScope<'_, '_>,
    ) -> Result<Self::MusicalContext, AraError> {
        self.document.musical_contexts.push(AraMusicalContext {
            name: properties.name().map(str::to_owned),
            region_sequences: Vec::new(),
        });
        Ok(())
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
        let context_index = self.document.musical_contexts.len() - 1;
        let sequences = &mut self.document.musical_contexts[context_index].region_sequences;
        let index = sequences.len();
        sequences.push(AraRegionSequence {
            name: properties.name().map(str::to_owned),
            order_index: properties.order_index(),
            playback_region_count: 0,
        });
        if let Some(handle) = context.object_handle() {
            self.sequence_index_by_handle.insert(handle, index);
        }
        Ok(index)
    }
}

impl AudioSources for ModelHandle {
    type AudioSource = usize;

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
        _host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        // 内容变了 → 版本号前进。渲染缓存据此失效（设计 §4.5）。
        self.source_content_version_bump(*state);
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
        if enable && std::env::var_os("HIFISHIFTER_ARA_PCM_PROBE").is_some() {
            // 仅用于一次性宿主观测：比较原始源样本与 ARA 返回的源样本，判断倒放是否改源。
            let result = (|| {
                let source = self.document.audio_sources.get(*state)
                    .ok_or(AraError::InvalidArgument("unknown source"))?;
                let source_ref = host.current_audio_source()
                    .ok_or(AraError::InvalidState("missing source scope"))?;
                let channels = source.channel_count.max(1) as usize;
                let mut reader = host.audio_reader::<f32>(source_ref, channels)?;
                let mut samples = vec![vec![0.0_f32; 16]; channels];
                let mut planes = samples.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>();
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
        log::info!("[ara] audio_source content_version #{}: {}", index, self.content_versions[index]);
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
        self.document.audio_modifications.push(AraAudioModification {
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
        _context: &CreateContext,
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
        let index = self.document.playback_regions.len();
        self.document.playback_regions.push(AraPlaybackRegion {
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
        });
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
        let region = self.document.playback_regions.get_mut(*state)
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
        log::info!("[ara] playback_region updated #{}: startPlay={:.6}", state, region.start_in_playback_time);
        Ok(())
    }

    /// 标记已销毁区间，编辑结束时只映射存活的区域。
    fn destroy_playback_region(&mut self, state: Self::PlaybackRegion) {
        self.destroyed_regions.insert(state);
        log::info!("[ara] playback_region destroyed #{}", state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ara2_bridge::core::{Registry, RegionSequenceKind};

    /// 宿主移动与拉伸更新必须替换现存区间，不能新增重复 clip 或忽略更新。
    #[test]
    fn host_region_update_changes_the_mapped_clip() {
        let mut model = ModelHandle::new();
        model.document = crate::ara::ara_document_from_json(include_str!(
            "../../../../probe/ara/captures/ara-model.reaper.json"
        )).unwrap();
        let mut index = 0;
        let mut sequences = Registry::<RegionSequenceKind, ()>::new(1);
        let handle = sequences.insert(()).unwrap();
        let sequence = sequences.model_ref(handle).unwrap();
        let properties = PlaybackRegionProperties::for_ara2(
            1, 0.25, 1.5, 5.0, 3.0, sequence, Some("moved"), None,
        ).unwrap();
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
    }

    /// 删除区域后摘要和时间线只能包含存活区域，避免切片或撤销留下幽灵 clip。
    #[test]
    fn destroying_a_region_removes_it_from_the_timeline() {
        let mut model = ModelHandle::new();
        model.document = crate::ara::ara_document_from_json(include_str!(
            "../../../../probe/ara/captures/ara-model.reaper.json"
        )).unwrap();
        PlaybackRegions::destroy_playback_region(&mut model, 0);
        model.remap_and_log();
        assert!(model.timeline().unwrap().clips.is_empty());
    }
}
