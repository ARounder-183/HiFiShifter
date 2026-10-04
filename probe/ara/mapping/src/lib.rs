//! HiFiShifter ARA 探针 / Task 3 —— `ARA region → TimelineState` 映射原型。
//!
//! 目的：回答探针计划里 R2 的前半句 —— **ARA 给出的 region 模型，能不能无损地落到
//! 产品渲染真正消费的 `TimelineState`**。
//!
//! 落点是**真实类型**：`backend_lib::__test_internals::{Clip, TimelineState}`。
//! 之所以能这样，是因为 backend 已经有一个 `#[doc(hidden)] pub mod __test_internals`
//! 供 `tests/` 集成目标使用；探针复用它，**不需要给 backend 加任何 `pub`**。
//!
//! 构造方式用 serde：把映射结果写成 JSON 再反序列化成真实结构。理由：
//! `Clip` / `Track` / `ClipTake` 都没有 `Default`，但它们的 serde 默认值就是产品
//! 加载工程文件时走的同一条路径 —— 用同一条路径构造，既是省事，也是"与产品口径一致"。
//!
//! 一次性产物：不进入 `backend/`、`frontend/`。

use backend_lib::__test_internals::TimelineState;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;

/// ARA 文档样本（对应 Task 1 落盘的 `captures/ara-model.*.json`）。
#[derive(Debug, Clone, Deserialize)]
pub struct AraDocument {
    /// 宿主给出的文档名；REAPER 实测为空串。
    #[serde(default, rename = "documentName")]
    pub document_name: String,
    /// 宿主托管的音频源。
    #[serde(default, rename = "audioSources")]
    pub audio_sources: Vec<AraAudioSource>,
    /// 音乐上下文及其下的 region sequence（≈ 轨道）。
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
    /// 宿主是否已授予样本访问权（Task 1 的未决项）。
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
    /// 该序列下的 region 数量。用于把扁平的 `playbackRegions` 切回各自序列。
    #[serde(default, rename = "playbackRegionCount")]
    pub playback_region_count: i64,
}

/// 一个 `ARAAudioModification`（≈ Take）。
#[derive(Debug, Clone, Deserialize)]
pub struct AraAudioModification {
    /// 稳定 id。
    #[serde(rename = "persistentID")]
    pub persistent_id: String,
    /// 所属源的稳定 id。
    #[serde(rename = "audioSourcePersistentID")]
    pub audio_source_persistent_id: String,
}

/// 一个 `ARAPlaybackRegion`（≈ Clip）。
#[derive(Debug, Clone, Deserialize)]
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
    /// 在**修改时间轴**上的起点（秒）。
    #[serde(default, rename = "startInModificationTime")]
    pub start_in_modification_time: f64,
    /// 在修改时间轴上的长度（秒）。
    #[serde(default, rename = "durationInModificationTime")]
    pub duration_in_modification_time: f64,
    /// 在**播放时间轴**上的起点（秒）。
    #[serde(default, rename = "startInPlaybackTime")]
    pub start_in_playback_time: f64,
    /// 在播放时间轴上的长度（秒）。
    #[serde(default, rename = "durationInPlaybackTime")]
    pub duration_in_playback_time: f64,
    /// 宿主是否真的启用了时间拉伸（插件不声明支持时恒 false）。
    #[serde(default, rename = "isTimestretchEnabled")]
    pub is_timestretch_enabled: bool,
    /// 区间头部是否有基于内容的淡化（插件不声明支持时恒 false）。
    #[serde(default, rename = "hasContentBasedFadeAtHead")]
    pub has_content_based_fade_at_head: bool,
    /// 区间尾部是否有基于内容的淡化。
    #[serde(default, rename = "hasContentBasedFadeAtTail")]
    pub has_content_based_fade_at_tail: bool,
}

/// ARA 不提供、但渲染需要的输入。
///
/// 这份清单是 Task 3 最有价值的产出：它把"映射不全"从一句担心变成可核对的条目，
/// 每条都在测试里被显式钉住降级行为。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LostField {
    /// 倒放：ARA 的 region 模型没有方向位，Task 1 的倒放尝试也没有产生任何标志。
    Reversed,
    /// 淡化形状与曲率：ARA 只有"是否有基于内容的淡化"两个布尔，没有形状/曲率。
    FadeShapeAndCurvature,
    /// 淡化长度：ARA 不给出淡化的时长。
    FadeLength,
    /// Loop source：ARA 的 region 模型没有 REAPER / VEGAS 的 loop 语义。
    LoopSource,
    /// Tempo Map：ARA 对象模型不带 tempo（需经 content reader 另行读取）。
    TempoMap,
    /// Item / Take 增益：ARA 对象模型不提供 item gain。
    ItemGain,
    /// 源文件内容指纹：ARA 只给 persistentID（路径），不给内容哈希。
    SourceFileFingerprint,
}

/// 本映射原型无法从 ARA 取得的渲染输入清单。
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
    /// 样本文件不是合法 JSON，或字段形状与预期不符。
    Json(serde_json::Error),
    /// region 引用了一个样本里不存在的源。
    UnknownSource(String),
    /// region 引用了不存在的修改。
    UnknownModification(String),
    /// region sequence 声明的 region 总数与扁平 region 列表对不上。
    SequenceCountMismatch {
        /// 各序列声明的数量之和。
        declared: i64,
        /// 扁平列表里的实际数量。
        actual: usize,
    },
    /// region 的播放时长为 0 或非有限值，无法推出播放速率。
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

/// 反序列化一份 Task 1 采集的 ARA 文档样本。
pub fn ara_document_from_json(text: &str) -> Result<AraDocument, MappingError> {
    Ok(serde_json::from_str(text)?)
}

/// ARA 文档 → `TimelineState`。
///
/// 映射口径（与 spec §5.2 一致）：
/// - 每个 `regionSequence` → 一条 `Track`（REAPER 实测：序列名 = 轨道名）；
/// - 每个 `playbackRegion` → 一个 `Clip`；
/// - 每个 `audioSource` → `Clip` 的 take 源（`source_path` = persistentID）；
/// - 播放位置 / 长度 = `startInPlaybackTime` / `durationInPlaybackTime`（秒，无需换算）；
/// - **拉伸 = 时长比** `playback_rate = durationInModificationTime / durationInPlaybackTime`；
/// - 源内偏移 = `startInModificationTime`；
/// - 源采样率 / 声道数 / 时长取自该源，逐源保真。
///
/// 已知近似（因为 Task 1 的 dump 没记录 region→sequence 这条边）：
/// 用各 sequence 声明的 `playbackRegionCount` 按序切分扁平的 region 列表。
/// 真实 ARA 里这条边是直接存在的（`createPlaybackRegion` 带 sequence 参数），
/// 产品实现必须用它，不要沿用这里的切分。
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

    // ── 轨道：每个 region sequence 一条 ──
    let mut tracks: Vec<Value> = Vec::new();
    let mut track_ids: Vec<String> = Vec::new();
    let mut declared_counts: Vec<usize> = Vec::new();
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
            declared_counts.push(sequence.playback_region_count.max(0) as usize);
            order += 1;
        }
    }

    // ── 把 flat 的 region 列表按声明数量切回各自序列 ──
    let declared_total: i64 = declared_counts.iter().map(|count| *count as i64).sum();
    if !doc.playback_regions.is_empty() && declared_total != doc.playback_regions.len() as i64 {
        return Err(MappingError::SequenceCountMismatch {
            declared: declared_total,
            actual: doc.playback_regions.len(),
        });
    }

    let mut clips: Vec<Value> = Vec::new();
    let mut region_index = 0usize;
    for (sequence_index, count) in declared_counts.iter().enumerate() {
        for _ in 0..*count {
            let Some(region) = doc.playback_regions.get(region_index) else {
                break;
            };
            region_index += 1;

            let source = sources
                .get(region.audio_source_persistent_id.as_str())
                .copied()
                .ok_or_else(|| {
                    MappingError::UnknownSource(region.audio_source_persistent_id.clone())
                })?;
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

            // 拉伸：ARA 用两个时长的比值表达，而不是 isTimestretchEnabled 标志位。
            let playback_rate =
                region.duration_in_modification_time / region.duration_in_playback_time;

            let clip_id = format!("ara-clip-{region_index}");
            let take_id = format!("{clip_id}-take-1");
            let clip_name = region
                .name
                .clone()
                .or_else(|| source.name.clone())
                .unwrap_or_else(|| clip_id.clone());
            let track_id = &track_ids[sequence_index];

            clips.push(json!({
                "id": clip_id,
                "track_id": track_id,
                "name": clip_name,
                "start_sec": region.start_in_playback_time,
                "length_sec": region.duration_in_playback_time,
                "active_take_id": take_id,
                // ARA 的拉伸落在 take 上；Clip 级倍率保持 1.0。
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
    }

    let project_sec = clips
        .iter()
        .map(|clip| {
            let start = clip.get("start_sec").and_then(Value::as_f64).unwrap_or(0.0);
            let length = clip.get("length_sec").and_then(Value::as_f64).unwrap_or(0.0);
            start + length
        })
        .fold(0.0_f64, f64::max);

    let timeline: TimelineState = serde_json::from_value(json!({
        "tracks": tracks,
        "clips": clips,
        "bpm": default_bpm(),
        "project_sec": project_sec,
    }))?;

    // 把 take 物化到 Clip 的扁平投影（产品加载工程时走的就是这一步）。
    let mut timeline = timeline;
    for clip in &mut timeline.clips {
        clip.normalize_takes();
    }
    Ok(timeline)
}

/// ARA 对象模型不给 tempo，映射原型用产品默认 BPM。
pub const fn default_bpm() -> f64 {
    120.0
}

/// 收集样本里出现的所有源采样率（逐源保真的回归用）。
pub fn source_sample_rates(doc: &AraDocument) -> Vec<u32> {
    doc.audio_sources
        .iter()
        .map(|source| source.sample_rate as u32)
        .collect()
}

/// 该样本里是否存在"两个时长不等"的 region —— 也就是 ARA 是否真的表达了拉伸。
pub fn has_observed_time_stretch(doc: &AraDocument) -> bool {
    doc.playback_regions.iter().any(|region| {
        (region.duration_in_modification_time - region.duration_in_playback_time).abs() > 1e-9
    })
}
