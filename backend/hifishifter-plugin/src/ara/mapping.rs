//! ARA 文档模型 → `TimelineState` 映射。
//!
//! 这是探针 Task 3 验证过的映射的产品化版本：落点是本体真实的
//! [`TimelineState`]（经 `hifishifter_kernel` 暴露），不是影子结构。
//!
//! 与探针版本的唯一功能差别：`AraPlaybackRegion` 多了 `region_sequence_index`，
//! 对应真实 ARA 里 `createPlaybackRegion` 自带的 sequence 参数。有它就按它归属，
//! 没有才退回"按各序列声明的 region 数量顺序切分"（Task 1 的 dump 缺这条边，
//! 探针只能那样做）。
//!
//! 已知丢失字段见 [`LOST_FIELDS`]：ARA 表达不到的输入一律**显式降级**，
//! 不允许悄悄填默认值。

use hifishifter_kernel::state::TimelineState;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;

/// ARA 文档（对应探针 Task 1 采集的 JSON 形状）。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AraDocument {
    /// 宿主给出的文档名。
    #[serde(default, rename = "documentName")]
    pub document_name: String,
    /// 宿主托管的音频源。
    #[serde(default, rename = "audioSources")]
    pub audio_sources: Vec<AraAudioSource>,
    /// 音乐上下文及其 region sequence（≈ 轨道）。
    #[serde(default, rename = "musicalContexts")]
    pub musical_contexts: Vec<AraMusicalContext>,
    /// 音频修改（≈ Take）。
    #[serde(default, rename = "audioModifications")]
    pub audio_modifications: Vec<AraAudioModification>,
    /// 播放区间（≈ Clip）。
    #[serde(default, rename = "playbackRegions")]
    pub playback_regions: Vec<AraPlaybackRegion>,
}

/// 一个 `ARAAudioSource`。
#[derive(Debug, Clone, Deserialize)]
pub struct AraAudioSource {
    /// 稳定 id —— REAPER 实测就是素材绝对路径。
    #[serde(rename = "persistentID")]
    pub persistent_id: String,
    /// 显示名。
    #[serde(default)]
    pub name: Option<String>,
    /// 源采样率。
    #[serde(default, rename = "sampleRate")]
    pub sample_rate: f64,
    /// 源样本数。
    #[serde(default, rename = "sampleCount")]
    pub sample_count: i64,
    /// 源时长（秒）。
    #[serde(default, rename = "durationSeconds")]
    pub duration_seconds: f64,
    /// 声道数。
    #[serde(default, rename = "channelCount")]
    pub channel_count: i32,
    /// 宿主是否已授予样本访问权。
    #[serde(default, rename = "sampleAccessEnabled")]
    pub sample_access_enabled: bool,
}

/// 一个 `ARAMusicalContext`。
#[derive(Debug, Clone, Deserialize)]
pub struct AraMusicalContext {
    /// 上下文名。
    #[serde(default)]
    pub name: Option<String>,
    /// 上下文内的 region sequence。
    #[serde(default, rename = "regionSequences")]
    pub region_sequences: Vec<AraRegionSequence>,
}

/// 一个 `ARARegionSequence`（≈ 轨道）。
#[derive(Debug, Clone, Deserialize)]
pub struct AraRegionSequence {
    /// 序列名 —— REAPER 实测就是轨道名。
    #[serde(default)]
    pub name: Option<String>,
    /// 序号。
    #[serde(default, rename = "orderIndex")]
    pub order_index: i32,
    /// 该序列下的 region 数量；`region_sequence_index` 缺席时用它切分。
    #[serde(default, rename = "playbackRegionCount")]
    pub playback_region_count: i64,
}

/// 一个 `ARAAudioModification`（≈ Take）。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AraAudioModification {
    /// 稳定 id。
    #[serde(rename = "persistentID")]
    pub persistent_id: String,
    /// 所属源的稳定 id。
    #[serde(rename = "audioSourcePersistentID")]
    pub audio_source_persistent_id: String,
}

/// 一个 `ARAPlaybackRegion`（≈ Clip）。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AraPlaybackRegion {
    /// 区间名。
    #[serde(default)]
    pub name: Option<String>,
    /// 所属源的稳定 id。
    #[serde(rename = "audioSourcePersistentID")]
    pub audio_source_persistent_id: String,
    /// 所属修改的稳定 id。
    #[serde(rename = "audioModificationPersistentID")]
    pub audio_modification_persistent_id: String,
    /// **真实 ARA 自带的归属边**：该 region 属于第几个 region sequence。
    /// 探针的 dump 没有这一项，因此允许缺席。
    #[serde(default, rename = "regionSequenceIndex")]
    pub region_sequence_index: Option<usize>,
    /// 在修改时间轴上的起点（秒）。
    #[serde(default, rename = "startInModificationTime")]
    pub start_in_modification_time: f64,
    /// 在修改时间轴上的长度（秒）。
    #[serde(default, rename = "durationInModificationTime")]
    pub duration_in_modification_time: f64,
    /// 在播放时间轴上的起点（秒）。
    #[serde(default, rename = "startInPlaybackTime")]
    pub start_in_playback_time: f64,
    /// 在播放时间轴上的长度（秒）。
    #[serde(default, rename = "durationInPlaybackTime")]
    pub duration_in_playback_time: f64,
    /// 宿主是否真的启用了时间拉伸。
    #[serde(default, rename = "isTimestretchEnabled")]
    pub is_timestretch_enabled: bool,
    /// 区间头部是否有基于内容的淡化。
    #[serde(default, rename = "hasContentBasedFadeAtHead")]
    pub has_content_based_fade_at_head: bool,
    /// 区间尾部是否有基于内容的淡化。
    #[serde(default, rename = "hasContentBasedFadeAtTail")]
    pub has_content_based_fade_at_tail: bool,
}

/// ARA 不提供、但渲染需要的输入。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LostField {
    /// 倒放：ARA 的 playback transformation 里**没有**反向位。
    Reversed,
    /// 淡化形状与曲率：只有"头/尾是否有基于内容的淡化"两个布尔。
    FadeShapeAndCurvature,
    /// 淡化长度。
    FadeLength,
    /// Loop source。
    LoopSource,
    /// Tempo Map（需经 content reader 另取）。
    TempoMap,
    /// Item / Take 增益。
    ItemGain,
    /// 源文件内容指纹（ARA 只给路径）。
    SourceFileFingerprint,
}

/// 无法从 ARA 取得的渲染输入清单。
pub const LOST_FIELDS: &[LostField] = &[
    LostField::Reversed,
    LostField::FadeShapeAndCurvature,
    LostField::FadeLength,
    LostField::LoopSource,
    LostField::TempoMap,
    LostField::ItemGain,
    LostField::SourceFileFingerprint,
];

/// 映射失败的原因。
#[derive(Debug)]
pub enum MappingError {
    /// JSON 形状不符。
    Json(serde_json::Error),
    /// region 引用了不存在的源。
    UnknownSource(String),
    /// region 引用了不存在的修改。
    UnknownModification(String),
    /// region 归属越界。
    UnknownRegionSequence(usize),
    /// 序列声明数量与扁平列表对不上。
    SequenceCountMismatch {
        /// 声明之和。
        declared: i64,
        /// 实际数量。
        actual: usize,
    },
    /// 播放时长为 0 或非有限值。
    NonPositivePlaybackDuration,
}

impl std::fmt::Display for MappingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(error) => write!(f, "ara document json: {error}"),
            Self::UnknownSource(id) => write!(f, "playback region references unknown source {id}"),
            Self::UnknownModification(id) => {
                write!(f, "playback region references unknown modification {id}")
            }
            Self::UnknownRegionSequence(index) => {
                write!(f, "playback region references region sequence {index}")
            }
            Self::SequenceCountMismatch { declared, actual } => write!(
                f,
                "region sequences declare {declared} regions but the flat list has {actual}"
            ),
            Self::NonPositivePlaybackDuration => {
                write!(f, "playback region has a non-positive playback duration")
            }
        }
    }
}

impl std::error::Error for MappingError {}

impl From<serde_json::Error> for MappingError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

/// 反序列化一份 ARA 文档。
pub fn ara_document_from_json(text: &str) -> Result<AraDocument, MappingError> {
    Ok(serde_json::from_str(text)?)
}

/// 先导映射尚未读取ARA Tempo Map；插件层再叠加有效VST3宿主tempo，初始用本体默认值。
pub const fn default_bpm() -> f64 {
    120.0
}

/// 样本里出现的所有源采样率。
pub fn source_sample_rates(doc: &AraDocument) -> Vec<u32> {
    doc.audio_sources
        .iter()
        .map(|source| source.sample_rate as u32)
        .collect()
}

/// 样本里是否存在"两个时长不等"的 region —— 即 ARA 是否真的表达了拉伸。
pub fn has_observed_time_stretch(doc: &AraDocument) -> bool {
    doc.playback_regions.iter().any(|region| {
        (region.duration_in_modification_time - region.duration_in_playback_time).abs() > 1e-9
    })
}

/// 每条 region 归属的 region sequence 下标。
fn region_sequence_of(doc: &AraDocument) -> Result<Vec<usize>, MappingError> {
    if doc
        .playback_regions
        .iter()
        .all(|region| region.region_sequence_index.is_some())
    {
        return Ok(doc
            .playback_regions
            .iter()
            .map(|region| region.region_sequence_index.unwrap_or(0))
            .collect());
    }

    let mut declared_counts = Vec::new();
    for context in &doc.musical_contexts {
        for sequence in &context.region_sequences {
            declared_counts.push(sequence.playback_region_count.max(0) as usize);
        }
    }
    let declared_total: i64 = declared_counts.iter().map(|count| *count as i64).sum();
    if !doc.playback_regions.is_empty() && declared_total != doc.playback_regions.len() as i64 {
        return Err(MappingError::SequenceCountMismatch {
            declared: declared_total,
            actual: doc.playback_regions.len(),
        });
    }

    let mut owned = Vec::with_capacity(doc.playback_regions.len());
    for (index, count) in declared_counts.iter().enumerate() {
        for _ in 0..*count {
            owned.push(index);
        }
    }
    Ok(owned)
}

/// ARA 文档 → `TimelineState`。
///
/// 映射口径见模块文档；拉伸按 `durationInModificationTime / durationInPlaybackTime`
/// 落到 take 的 `playback_rate`，而不是读 `isTimestretchEnabled` 标志位。
pub fn ara_document_to_timeline(doc: &AraDocument) -> Result<TimelineState, MappingError> {
    let sources: HashMap<&str, &AraAudioSource> = doc
        .audio_sources
        .iter()
        .map(|source| (source.persistent_id.as_str(), source))
        .collect();
    let modifications: HashMap<&str, &str> = doc
        .audio_modifications
        .iter()
        .map(|modification| {
            (
                modification.persistent_id.as_str(),
                modification.audio_source_persistent_id.as_str(),
            )
        })
        .collect();

    let mut tracks: Vec<Value> = Vec::new();
    let mut track_ids: Vec<String> = Vec::new();
    let mut order = 0;
    for context in &doc.musical_contexts {
        for sequence in &context.region_sequences {
            let id = format!("ara-track-{}", track_ids.len());
            tracks.push(json!({
                "id": id,
                "name": sequence.name.clone().unwrap_or_default(),
                "order": order,
            }));
            track_ids.push(id);
            order += 1;
        }
    }

    let ownership = region_sequence_of(doc)?;
    let mut clips: Vec<Value> = Vec::new();
    for (index, region) in doc.playback_regions.iter().enumerate() {
        let sequence_index = ownership.get(index).copied().unwrap_or(0);
        let track_id = track_ids
            .get(sequence_index)
            .ok_or(MappingError::UnknownRegionSequence(sequence_index))?;

        let source = sources
            .get(region.audio_source_persistent_id.as_str())
            .copied()
            .ok_or_else(|| MappingError::UnknownSource(region.audio_source_persistent_id.clone()))?;
        if !modifications.contains_key(region.audio_modification_persistent_id.as_str()) {
            return Err(MappingError::UnknownModification(
                region.audio_modification_persistent_id.clone(),
            ));
        }
        if !(region.duration_in_playback_time.is_finite()
            && region.duration_in_playback_time > 0.0)
        {
            return Err(MappingError::NonPositivePlaybackDuration);
        }

        let playback_rate =
            region.duration_in_modification_time / region.duration_in_playback_time;
        let clip_id = format!("ara-clip-{}", index + 1);
        let take_id = format!("{clip_id}-take-1");
        let clip_name = region
            .name
            .clone()
            .or_else(|| source.name.clone())
            .unwrap_or_else(|| clip_id.clone());

        clips.push(json!({
            "id": clip_id,
            "track_id": track_id,
            "name": clip_name,
            "start_sec": region.start_in_playback_time,
            "length_sec": region.duration_in_playback_time,
            "active_take_id": take_id,
            "clip_playback_rate": 1.0,
            "takes": [{
                "id": take_id,
                "name": clip_name,
                "source_path": source.persistent_id,
                "source_start_sec": region.start_in_modification_time,
                "source_end_sec":
                    region.start_in_modification_time + region.duration_in_modification_time,
                "playback_rate": playback_rate,
                "source_sample_rate": source.sample_rate as u32,
                "duration_sec": source.duration_seconds,
                "source_channels": source.channel_count.max(0) as u16,
            }],
        }));
    }

    let project_sec = clips
        .iter()
        .map(|clip| {
            let start = clip.get("start_sec").and_then(Value::as_f64).unwrap_or(0.0);
            let length = clip.get("length_sec").and_then(Value::as_f64).unwrap_or(0.0);
            start + length
        })
        .fold(0.0_f64, f64::max);

    let mut timeline: TimelineState = serde_json::from_value(json!({
        "tracks": tracks,
        "clips": clips,
        "bpm": default_bpm(),
        "project_sec": project_sec,
    }))?;

    // 把 take 物化到 Clip 的扁平投影 —— 与产品加载工程时走同一步。
    for clip in &mut timeline.clips {
        clip.normalize_takes();
    }
    Ok(timeline)
}

/// 一行可核对的摘要。
///
/// 【为什么把格式钉死】采集日志要能被 diff：判据 A1（"插件看到的数与工程实际一致"）
/// 就是靠比对这一行得出的。字段顺序与拼写变了，历史日志就不可比。
pub fn summary_line(document: &AraDocument, timeline: &TimelineState) -> String {
    let region_sequences: usize = document
        .musical_contexts
        .iter()
        .map(|context| context.region_sequences.len())
        .sum();
    format!(
        "ara: sources={} modifications={} regionSequences={} playbackRegions={} clips={}",
        document.audio_sources.len(),
        document.audio_modifications.len(),
        region_sequences,
        document.playback_regions.len(),
        timeline.clips.len(),
    )
}

/// 记录每个 clip 的播放起点，供宿主移动 item 后核对 ARA 更新。
pub fn clip_starts_line(timeline: &TimelineState) -> String {
    let starts = timeline
        .clips
        .iter()
        .map(|clip| format!("{:.6}", clip.start_sec))
        .collect::<Vec<_>>();
    format!("ara: clipStartsSec=[{}]", starts.join(","))
}
