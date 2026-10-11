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

/// ARA 的 playback transformation / region 结构**表达不到**、但渲染需要的输入。
///
/// 【这一层现在是什么】每一项都在**存在原生宿主扩展时**由宿主只读查询补回来：
///
/// | 字段 | 补回它的地方 |
/// |---|---|
/// | [`LostField::Reversed`] | `project_host_take_facts_locked`（`PCM_Source_GetSectionInfo` 的方向位） |
/// | [`LostField::FadeShapeAndCurvature`] | `project_ui_fades_locked`（`D_FADE*DIR` / `_NEW`） |
/// | [`LostField::FadeLength`] | `project_ui_fades_locked`（`D_FADE*LEN`） |
/// | [`LostField::LoopSource`] | `project_host_take_facts_locked`（item 级 `B_LOOPSRC`） |
/// | [`LostField::ItemGain`] | `display_item_gain`（`D_VOL`） |
/// | [`LostField::TempoMap`] | 仅**单点**：VST3 `ProcessContext` 播放位置处的 tempo（见下） |
/// | [`LostField::SourceFileFingerprint`] | `Clip::source_file_fingerprint`（宿主文件名 + 内容版本） |
///
/// 【TempoMap 的现状与上表不符的历史】`editor/session.rs` 的补回链只取
/// `ProcessContext` 里**播放位置处的单点 tempo** 写进 `timeline.bpm`；宿主经 ARA
/// musical context 给出的多点 Tempo/BarSignature 内容目前零消费
/// （`model.rs::probe_host_musical_content` 只记条数日志），多点 tempo map 由插件
/// 设置 + 单点 bpm 合成。含 tempo map 的工程里，小节/拍网格与 musical-time 投影
/// 因此是近似的（只有 playhead 处是对的）；秒域换算不受影响 —— 曲线 basis 存源
/// 坐标，与 tempo 无关。真消费多点 tempo map 时应把它折进渲染缓存键
/// （`PreparedVersion`）。
///
/// 【关键限定：这张表的补回点全是**原生扩展**】ARA 2.x 本身**没有**方向位、循环源、
/// 声道模式、item 增益这些概念（`ARAPlaybackRegion` 只给四段几何、`isTimestretchEnabled`
/// 与两个 content-based fade 布尔）。因此在**只提供 ARA 的宿主**（Studio One / Cubase /
/// Nuendo / Logic …）上，上表每一行都**补不回来** —— 插件会以安全默认渲染（正放 /
/// 不循环 / 声道模式 0 / 无 item 增益），而不是拿到错误的值。这不是"还没实现"，是
/// 这些概念在该宿主里**根本不存在**。此前这里的注释写成"都已经补回来、不再是丢失"，
/// 对 ARA-only 后端是**不准确的**。
///
/// 【为什么保留这份清单】它是**可审计的边界声明**：任何"ARA 给不了、只能问宿主"的输入
/// 都应先在这里登记，再决定它是否要进渲染缓存键（`PreparedVersion`）。同时它也是
/// "哪些功能天生依赖原生扩展"的唯一出处 —— 前端据此把对应控件在 ARA-only 上标成只读。
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

/// 无法从 ARA 本体取得的渲染输入清单；每项的宿主补回点见 [`LostField`]。
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
    // 【为什么不再整份失败】声明数量与扁平列表对不上时，此前直接返回
    // `SequenceCountMismatch` ⇒ **整份文档**映射失败 ⇒ 插件里一片空白。宿主（尤其
    // 非 REAPER）可能既不填 `playbackRegionCount`（恒 0）也不给 region→sequence 边 ——
    // 那不是"文档损坏"，而是"归属信息缺失"。此时退回**顺序归属**：按声明数量切分，
    // 多出来的 region 归到第一条序列（下标 0，调用点的 `unwrap_or(0)` 已实现）。
    // 宁可把片段放在首轨，也不要什么都不显示；真实布局缺失只记一行限流告警。
    if !doc.playback_regions.is_empty() && declared_total != doc.playback_regions.len() as i64 {
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            "ara.region-sequence-count",
            format_args!(
                "[ara] region sequences declare {declared_total} region(s) but the flat list has {}; \
                 attributing the surplus to the first sequence",
                doc.playback_regions.len()
            ),
        );
    }

    let mut owned = Vec::with_capacity(doc.playback_regions.len());
    for (index, count) in declared_counts.iter().enumerate() {
        for _ in 0..*count {
            owned.push(index);
        }
    }
    Ok(owned)
}

/// 一条被跳过的 region 及其原因。
///
/// 【为什么跳过而不是失败】一个零长度 item、一个 MIDI item（ARA 不给音频源）、
/// 一条引用了已删除源的 region —— 都是**局部**问题。此前任何一条都会让整份映射
/// `Err`，进而把 `DocumentSession::ready` 打成 `false`、整个实例永久砖化。
/// 现在逐 region 隔离：坏的那条不产生 clip，其余照常。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkippedRegion {
    /// 在 `doc.playback_regions` 里的下标。
    pub index: usize,
    /// 跳过原因（语言无关的分类名，文案由上层本地化）。
    pub reason: SkipReason,
}

/// 跳过原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// region 指向了不存在的 region sequence。
    UnknownRegionSequence,
    /// region 引用了不存在的音频源。
    UnknownSource,
    /// region 引用了不存在的修改。
    UnknownModification,
    /// 播放时长为 0 或非有限值（零长度 item）。
    NonPositivePlaybackDuration,
}

impl SkipReason {
    /// 语言无关的分类名，供日志与逐 clip 状态复用。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnknownRegionSequence => "unknown_region_sequence",
            Self::UnknownSource => "unknown_source",
            Self::UnknownModification => "unknown_modification",
            Self::NonPositivePlaybackDuration => "non_positive_duration",
        }
    }
}

/// 逐 region 映射的结果。
pub struct MappingOutcome {
    /// 映射出的时间线（只含成功映射的 region）。
    pub timeline: TimelineState,
    /// 与 `timeline.clips` **一一对应**：该 clip 来自 `doc.playback_regions` 的哪个下标。
    ///
    /// 【为什么必须带回这份对齐】调用方（`ModelHandle::remap_and_log`）要把 clip 身份
    /// 挂回 region key。此前它靠"文档顺序 zip 存活 region"来对齐 —— 一旦有 region
    /// 被跳过，这个 zip 就整体错位，把身份挂到错误的 clip 上。带回来就不会错。
    pub clip_regions: Vec<usize>,
    /// 被跳过的 region。
    pub skipped: Vec<SkippedRegion>,
}

/// ARA 文档 → `TimelineState`（整份成功或整份失败）。
///
/// 供只需要时间线、不关心跳过明细的调用方使用；坏 region 会被静默丢弃。
/// 需要逐 region 明细时用 [`ara_document_to_timeline_reporting`]。
pub fn ara_document_to_timeline(doc: &AraDocument) -> Result<TimelineState, MappingError> {
    Ok(ara_document_to_timeline_reporting(doc)?.timeline)
}

/// ARA 文档 → `TimelineState`，并回报逐 region 的跳过明细。
///
/// 映射口径见模块文档；拉伸按 `durationInModificationTime / durationInPlaybackTime`
/// 落到 take 的 `playback_rate`，而不是读 `isTimestretchEnabled` 标志位。
///
/// 【只有文档级结构不一致才整份失败】`Json`（无法解析）保留为致命错误 —— 连形状都
/// 读不出来时没有可映射的东西；单条 region 的问题一律降级为跳过，region→sequence
/// 归属缺失则退回顺序归属（见 [`region_sequence_of`]），不再整份失败。
pub fn ara_document_to_timeline_reporting(
    doc: &AraDocument,
) -> Result<MappingOutcome, MappingError> {
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
    let mut clip_regions: Vec<usize> = Vec::new();
    let mut skipped: Vec<SkippedRegion> = Vec::new();
    for (index, region) in doc.playback_regions.iter().enumerate() {
        let sequence_index = ownership.get(index).copied().unwrap_or(0);
        let Some(track_id) = track_ids.get(sequence_index) else {
            skipped.push(SkippedRegion {
                index,
                reason: SkipReason::UnknownRegionSequence,
            });
            continue;
        };

        let Some(source) = sources
            .get(region.audio_source_persistent_id.as_str())
            .copied()
        else {
            skipped.push(SkippedRegion {
                index,
                reason: SkipReason::UnknownSource,
            });
            continue;
        };
        if !modifications.contains_key(region.audio_modification_persistent_id.as_str()) {
            skipped.push(SkippedRegion {
                index,
                reason: SkipReason::UnknownModification,
            });
            continue;
        }
        if !(region.duration_in_playback_time.is_finite() && region.duration_in_playback_time > 0.0)
        {
            skipped.push(SkippedRegion {
                index,
                reason: SkipReason::NonPositivePlaybackDuration,
            });
            continue;
        }

        let playback_rate = region.duration_in_modification_time / region.duration_in_playback_time;
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
        // 下标是**原文档下标**（`ara-clip-{index+1}` 与它一致），所以身份在
        // "中间有 region 被跳过"时也保持稳定 —— 不会因为跳过而整体前移。
        clip_regions.push(index);
    }

    let project_sec = clips
        .iter()
        .map(|clip| {
            let start = clip.get("start_sec").and_then(Value::as_f64).unwrap_or(0.0);
            let length = clip
                .get("length_sec")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
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
    Ok(MappingOutcome {
        timeline,
        clip_regions,
        skipped,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 一份与探针 Task 1 采集形状一致的文档：一条源、一条修改、一条序列、两条 region。
    ///
    /// region 0 表达拉伸（`durationMod=1.0` / `durationPlay=2.0` ⇒ rate 0.5），
    /// region 1 不拉伸（rate 1.0）。
    fn probe_shaped_document() -> AraDocument {
        ara_document_from_json(
            r#"{
                "documentName": "session",
                "audioSources": [
                    { "persistentID": "C:/audio/one.wav", "name": "one", "sampleRate": 48000.0,
                      "sampleCount": 144000, "durationSeconds": 3.0, "channelCount": 2,
                      "sampleAccessEnabled": true }
                ],
                "musicalContexts": [
                    { "name": "ctx", "regionSequences": [
                        { "name": "Track A", "orderIndex": 0, "playbackRegionCount": 2 }
                    ] }
                ],
                "audioModifications": [
                    { "persistentID": "mod-1", "audioSourcePersistentID": "C:/audio/one.wav" }
                ],
                "playbackRegions": [
                    { "name": "clip a", "audioSourcePersistentID": "C:/audio/one.wav",
                      "audioModificationPersistentID": "mod-1", "regionSequenceIndex": 0,
                      "startInModificationTime": 0.5, "durationInModificationTime": 1.0,
                      "startInPlaybackTime": 4.0, "durationInPlaybackTime": 2.0,
                      "isTimestretchEnabled": false },
                    { "audioSourcePersistentID": "C:/audio/one.wav",
                      "audioModificationPersistentID": "mod-1", "regionSequenceIndex": 0,
                      "startInModificationTime": 1.5, "durationInModificationTime": 0.5,
                      "startInPlaybackTime": 6.0, "durationInPlaybackTime": 0.5 }
                ]
            }"#,
        )
        .expect("probe-shaped document parses")
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn json_round_trip_reads_the_probe_shape() {
        let doc = probe_shaped_document();
        assert_eq!(doc.document_name, "session");
        assert_eq!(doc.audio_sources.len(), 1);
        assert_eq!(doc.audio_sources[0].persistent_id, "C:/audio/one.wav");
        assert_eq!(doc.audio_sources[0].name.as_deref(), Some("one"));
        assert!(doc.audio_sources[0].sample_access_enabled);
        assert_eq!(doc.audio_modifications[0].persistent_id, "mod-1");
        assert_eq!(doc.playback_regions[0].region_sequence_index, Some(0));
        assert_eq!(doc.playback_regions[1].name, None);
    }

    #[test]
    fn source_rates_and_stretch_are_read_from_the_document() {
        let doc = probe_shaped_document();
        assert_eq!(source_sample_rates(&doc), vec![48000]);
        // region 0 的 mod/play 时长不等 ⇒ 观察到拉伸。
        assert!(has_observed_time_stretch(&doc));

        let untimed = ara_document_from_json(
            r#"{
                "audioSources": [
                    { "persistentID": "s", "sampleRate": 44100.0 }
                ],
                "audioModifications": [
                    { "persistentID": "m", "audioSourcePersistentID": "s" }
                ],
                "musicalContexts": [
                    { "regionSequences": [{ "playbackRegionCount": 1 }] }
                ],
                "playbackRegions": [
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "durationInModificationTime": 1.0, "durationInPlaybackTime": 1.0 }
                ]
            }"#,
        )
        .expect("untimed document parses");
        assert_eq!(source_sample_rates(&untimed), vec![44100]);
        assert!(!has_observed_time_stretch(&untimed));
    }

    #[test]
    fn mapping_projects_track_and_clip_geometry_and_take_media() {
        let doc = probe_shaped_document();
        let outcome = ara_document_to_timeline_reporting(&doc).expect("document maps");

        assert!(outcome.skipped.is_empty());
        assert_eq!(outcome.clip_regions, vec![0, 1]);

        assert_eq!(outcome.timeline.tracks.len(), 1);
        let track = &outcome.timeline.tracks[0];
        assert_eq!(track.id, "ara-track-0");
        assert_eq!(track.name, "Track A");
        assert_eq!(track.order, 0);

        assert_eq!(outcome.timeline.clips.len(), 2);
        let clip = &outcome.timeline.clips[0];
        assert_eq!(clip.id, "ara-clip-1");
        assert_eq!(clip.track_id, "ara-track-0");
        assert_eq!(clip.name, "clip a");
        assert_close(clip.start_sec, 4.0);
        assert_close(clip.length_sec, 2.0);

        // take 物化到扁平投影：源窗口是**正放口径**（startMod .. startMod+durMod）。
        assert_eq!(clip.source_path.as_deref(), Some("C:/audio/one.wav"));
        assert_close(clip.source_start_sec, 0.5);
        assert_close(clip.source_end_sec, 1.5);
        assert_eq!(clip.source_sample_rate, Some(48000));
        assert_eq!(clip.source_channels, Some(2));
        assert_eq!(clip.duration_sec, Some(3.0));
        // 拉伸由时长比值表达，而不是 isTimestretchEnabled 旗标。
        assert!((clip.playback_rate - 0.5).abs() < 1e-6);

        // 第二条 region 名字缺席时回退到源名，且不拉伸。
        let second = &outcome.timeline.clips[1];
        assert_eq!(second.id, "ara-clip-2");
        assert_eq!(second.name, "one");
        assert_close(second.start_sec, 6.0);
        assert!((second.playback_rate - 1.0).abs() < 1e-6);
    }

    #[test]
    fn playback_rate_comes_from_the_duration_ratio_not_the_stretch_flag() {
        // 旗标为 false，但时长确实不等：映射仍必须按比值给出速率。
        let doc = probe_shaped_document();
        assert!(!doc.playback_regions[0].is_timestretch_enabled);
        let clip = &ara_document_to_timeline(&doc).unwrap().clips[0];
        assert!((clip.playback_rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn zero_length_region_is_skipped_without_failing_the_document() {
        let doc = ara_document_from_json(
            r#"{
                "audioSources": [
                    { "persistentID": "s", "sampleRate": 44100.0 }
                ],
                "audioModifications": [
                    { "persistentID": "m", "audioSourcePersistentID": "s" }
                ],
                "musicalContexts": [
                    { "regionSequences": [{ "playbackRegionCount": 1 }] }
                ],
                "playbackRegions": [
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "durationInPlaybackTime": 0.0 }
                ]
            }"#,
        )
        .expect("document parses");

        // 整份映射**不**失败（这正是"一个零长度 item 打死整个插件"的修复点）。
        let outcome = ara_document_to_timeline_reporting(&doc).expect("document maps");
        assert!(outcome.timeline.clips.is_empty());
        assert_eq!(
            outcome.skipped,
            vec![SkippedRegion {
                index: 0,
                reason: SkipReason::NonPositivePlaybackDuration,
            }]
        );
        assert_eq!(
            SkipReason::NonPositivePlaybackDuration.as_str(),
            "non_positive_duration"
        );
    }

    #[test]
    fn unknown_source_and_modification_are_skipped_locally_with_reasons() {
        let doc = ara_document_from_json(
            r#"{
                "audioSources": [
                    { "persistentID": "s", "sampleRate": 44100.0 }
                ],
                "audioModifications": [
                    { "persistentID": "m", "audioSourcePersistentID": "s" }
                ],
                "musicalContexts": [
                    { "regionSequences": [{ "playbackRegionCount": 2 }] }
                ],
                "playbackRegions": [
                    { "audioSourcePersistentID": "missing", "audioModificationPersistentID": "m",
                      "durationInPlaybackTime": 1.0 },
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "missing",
                      "durationInPlaybackTime": 1.0 }
                ]
            }"#,
        )
        .expect("document parses");

        let outcome = ara_document_to_timeline_reporting(&doc).expect("document maps");
        assert!(outcome.timeline.clips.is_empty());
        assert_eq!(
            outcome.skipped,
            vec![
                SkippedRegion {
                    index: 0,
                    reason: SkipReason::UnknownSource,
                },
                SkippedRegion {
                    index: 1,
                    reason: SkipReason::UnknownModification,
                },
            ]
        );
        assert_eq!(SkipReason::UnknownSource.as_str(), "unknown_source");
        assert_eq!(
            SkipReason::UnknownModification.as_str(),
            "unknown_modification"
        );
    }

    #[test]
    fn clip_regions_keeps_original_indices_when_a_middle_region_is_skipped() {
        // 【为什么钉这条】调用方靠 `clip_regions` 把身份挂回 region key；一旦有 region
        // 被跳过，按顺序 zip 就会整体错位，把身份挂到错误的 clip 上。
        let doc = ara_document_from_json(
            r#"{
                "audioSources": [
                    { "persistentID": "s", "sampleRate": 44100.0 }
                ],
                "audioModifications": [
                    { "persistentID": "m", "audioSourcePersistentID": "s" }
                ],
                "musicalContexts": [
                    { "regionSequences": [{ "playbackRegionCount": 3 }] }
                ],
                "playbackRegions": [
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "startInPlaybackTime": 0.0, "durationInPlaybackTime": 1.0 },
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "startInPlaybackTime": 1.0, "durationInPlaybackTime": 0.0 },
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "startInPlaybackTime": 2.0, "durationInPlaybackTime": 1.0 }
                ]
            }"#,
        )
        .expect("document parses");

        let outcome = ara_document_to_timeline_reporting(&doc).expect("document maps");
        assert_eq!(outcome.clip_regions, vec![0, 2]);
        assert_eq!(
            outcome.skipped,
            vec![SkippedRegion {
                index: 1,
                reason: SkipReason::NonPositivePlaybackDuration,
            }]
        );
        // clip id 也按**原文档下标**生成，跳过后不前移。
        let ids = outcome
            .timeline
            .clips
            .iter()
            .map(|clip| clip.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["ara-clip-1", "ara-clip-3"]);
    }

    #[test]
    fn fallback_ownership_splits_by_declared_region_counts() {
        // 没有 regionSequenceIndex 的探针 dump：按各序列声明的数量顺序切分。
        let doc = ara_document_from_json(
            r#"{
                "audioSources": [
                    { "persistentID": "s", "sampleRate": 44100.0 }
                ],
                "audioModifications": [
                    { "persistentID": "m", "audioSourcePersistentID": "s" }
                ],
                "musicalContexts": [
                    { "regionSequences": [
                        { "name": "A", "playbackRegionCount": 1 },
                        { "name": "B", "playbackRegionCount": 1 }
                    ] }
                ],
                "playbackRegions": [
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "startInPlaybackTime": 0.0, "durationInPlaybackTime": 1.0 },
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "startInPlaybackTime": 1.0, "durationInPlaybackTime": 1.0 }
                ]
            }"#,
        )
        .expect("document parses");

        let timeline = ara_document_to_timeline(&doc).expect("document maps");
        assert_eq!(timeline.tracks.len(), 2);
        assert_eq!(timeline.tracks[0].id, "ara-track-0");
        assert_eq!(timeline.tracks[1].id, "ara-track-1");
        assert_eq!(timeline.clips[0].track_id, "ara-track-0");
        assert_eq!(timeline.clips[1].track_id, "ara-track-1");
    }

    /// 声明数量与扁平列表对不上**不再整份失败**：按声明数量切分，多出来的 region 归到
    /// 第一条序列。宿主不填 `playbackRegionCount` 时插件仍要能显示片段，而不是一片空白。
    #[test]
    fn declared_count_mismatch_falls_back_to_the_first_sequence() {
        let doc = ara_document_from_json(
            r#"{
                "audioSources": [
                    { "persistentID": "s", "sampleRate": 44100.0 }
                ],
                "audioModifications": [
                    { "persistentID": "m", "audioSourcePersistentID": "s" }
                ],
                "musicalContexts": [
                    { "regionSequences": [{ "playbackRegionCount": 1 }] }
                ],
                "playbackRegions": [
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "durationInPlaybackTime": 1.0 },
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "durationInPlaybackTime": 1.0 }
                ]
            }"#,
        )
        .expect("document parses");

        // 声明 1 条、实际 2 条 ⇒ 两条都落在首条序列上，映射成功而不是整份失败。
        let timeline = ara_document_to_timeline(&doc).expect("mismatch must not be fatal");
        assert_eq!(timeline.clips.len(), 2);
        assert!(timeline
            .clips
            .iter()
            .all(|clip| clip.track_id == "ara-track-0"));
    }

    /// 完全没有 region sequence 时，region 无轨可归 —— 如实逐条跳过，而不是整份失败。
    #[test]
    fn regions_without_any_sequence_are_skipped_individually() {
        let doc = ara_document_from_json(
            r#"{
                "audioSources": [
                    { "persistentID": "s", "sampleRate": 44100.0 }
                ],
                "audioModifications": [
                    { "persistentID": "m", "audioSourcePersistentID": "s" }
                ],
                "musicalContexts": [],
                "playbackRegions": [
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "durationInPlaybackTime": 1.0 }
                ]
            }"#,
        )
        .expect("document parses");
        let outcome = ara_document_to_timeline_reporting(&doc).expect("must not be fatal");
        assert!(outcome.timeline.clips.is_empty());
        assert_eq!(outcome.skipped.len(), 1);
        assert_eq!(outcome.skipped[0].reason, SkipReason::UnknownRegionSequence);
    }

    #[test]
    fn explicit_region_sequence_index_wins_and_out_of_range_is_skipped() {
        // 显式边存在时**不**做声明数量一致性检查。
        let explicit = ara_document_from_json(
            r#"{
                "audioSources": [
                    { "persistentID": "s", "sampleRate": 44100.0 }
                ],
                "audioModifications": [
                    { "persistentID": "m", "audioSourcePersistentID": "s" }
                ],
                "musicalContexts": [
                    { "regionSequences": [
                        { "playbackRegionCount": 0 },
                        { "playbackRegionCount": 0 }
                    ] }
                ],
                "playbackRegions": [
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "regionSequenceIndex": 1, "durationInPlaybackTime": 1.0 }
                ]
            }"#,
        )
        .expect("document parses");
        let outcome =
            ara_document_to_timeline_reporting(&explicit).expect("explicit ownership maps");
        assert!(outcome.skipped.is_empty());
        assert_eq!(outcome.timeline.clips[0].track_id, "ara-track-1");

        // 越界的显式下标 ⇒ 该 region 被跳过，其余照常。
        let out_of_range = ara_document_from_json(
            r#"{
                "audioSources": [
                    { "persistentID": "s", "sampleRate": 44100.0 }
                ],
                "audioModifications": [
                    { "persistentID": "m", "audioSourcePersistentID": "s" }
                ],
                "musicalContexts": [
                    { "regionSequences": [{ "playbackRegionCount": 0 }] }
                ],
                "playbackRegions": [
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "regionSequenceIndex": 5, "durationInPlaybackTime": 1.0 }
                ]
            }"#,
        )
        .expect("document parses");
        let outcome = ara_document_to_timeline_reporting(&out_of_range).expect("document maps");
        assert!(outcome.timeline.clips.is_empty());
        assert_eq!(
            outcome.skipped,
            vec![SkippedRegion {
                index: 0,
                reason: SkipReason::UnknownRegionSequence,
            }]
        );
        assert_eq!(
            SkipReason::UnknownRegionSequence.as_str(),
            "unknown_region_sequence"
        );
    }

    #[test]
    fn summary_and_clip_start_lines_are_stable() {
        let doc = probe_shaped_document();
        let timeline = ara_document_to_timeline(&doc).expect("document maps");
        assert_eq!(
            summary_line(&doc, &timeline),
            "ara: sources=1 modifications=1 regionSequences=1 playbackRegions=2 clips=2"
        );
        assert_eq!(
            clip_starts_line(&timeline),
            "ara: clipStartsSec=[4.000000,6.000000]"
        );
    }

    #[test]
    fn default_bpm_is_pinned() {
        assert_eq!(default_bpm(), 120.0);
    }
}
