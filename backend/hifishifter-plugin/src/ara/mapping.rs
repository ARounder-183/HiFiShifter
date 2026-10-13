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
    /// **工程（文档）采样率**，由宿主经 ARA 的 companion API（VST3 `ProcessSetup`）
    /// 通报。ARA 本体的 `ARADocumentProperties` 只有 `structSize` 与 `name`，没有采样率，
    /// 因此这一项是插件从 VST3 侧补进来的，**不是** ARA 结构体字段。
    ///
    /// `None` / `0` / 非有限值都当作"未知"，映射层随即退回推导或夹窗口。
    #[serde(default, skip)]
    pub project_sample_rate: Option<f64>,
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
///
/// 【为什么没有 persistentID 字段】`ARARegionSequenceProperties` 只有 name / orderIndex /
/// musicalContextRef / color（`ARAInterface.h:775-793`）—— **没有** persistentID。
/// 序列身份取自宿主的 model-ref key，由 `ModelHandle` 经 `MappingIdentities` 传入。
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
/// | [`LostField::TempoMap`] | 宿主 ARA 音乐上下文的 `Tempo` / `BarSignatures` 内容（`read_host_musical_content`）；读不到时退回 VST3 `ProcessContext` 的单点 tempo |
/// | [`LostField::SourceFileFingerprint`] | `Clip::source_file_fingerprint`（宿主文件名 + 内容版本） |
///
/// 【TempoMap 现在怎么补】`create_musical_context` 与 `update_musical_context_content`
/// 调 `read_host_musical_content` 读入宿主的 `Tempo` / `BarSignatures`，存进
/// `DocumentSession::host_tempo_map`。投影（`project_plugin_musical_context_locked`）
/// 按每个音阶点的**位置**取该处的宿主 BPM 与拍号，而不是全曲一个值；该 map 同时是
/// 投影缓存键的一部分，因此宿主变速会立刻重建网格。读不到内容时退回
/// `ProcessContext` 的单点 tempo（`host_meter`），行为与改动前一致。
///
/// 秒域换算不受影响 —— 曲线 basis 存源坐标，与 tempo 无关。
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
    /// Tempo Map。
    ///
    /// 【为什么仍留在表里】它**不是**"ARA 表达不到"的字段 —— `Tempo` / `BarSignatures`
    /// 是 ARA 2.0 的一等内容类型，插件现在真的读它（`read_host_musical_content`）。
    /// 留在表里的理由是这张表的另一个用途：它是"哪些输入依赖宿主扩展"的出处。
    /// Tempo map 依赖的是**内容读取**（另一种宿主扩展），与方向位/循环源那几项同属
    /// "宿主不给就只能降级"，因此一并登记，只是补回点不同。
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

/// 可接受的工程采样率区间。与 `render/source.rs` 的素材读取门保持一致，
/// 避免一边拒绝、一边又把它当成权威速率。
const PLAUSIBLE_SAMPLE_RATE_RANGE: std::ops::RangeInclusive<f64> = 8000.0..=384_000.0;

/// 解析出的工程采样率，连同它是怎么来的。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectRate {
    /// 工程采样率（Hz）。
    pub rate: f64,
    /// 来源，供日志与取证。
    pub origin: ProjectRateOrigin,
}

/// 工程采样率的来源。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectRateOrigin {
    /// 宿主经 ARA companion API（VST3 `ProcessSetup`）通报 —— 唯一权威来源。
    Host,
    /// 拿不到任何信号。
    Unknown,
}

/// 解析工程采样率。
///
/// 【为什么需要它】素材采样率与工程采样率不同时，宿主把
/// `durationInModificationTime` 按**工程**速率折算（实测 Cubase 15：48 kHz / 1526332 帧
/// 素材放进 44.1 kHz 工程，给出 `durationMod = durationPlay = 34.610703`，而
/// 34.610703 × 44100 恰好等于源帧数）。映射层若把修改时长当**源**速率秒，就会多要
/// 8.2% 的素材，`playback_rate` 也恒等于 1 —— 用户听感是"ARA 前后不一致"。
///
/// ARA 本体给不出这个速率：`ARADocumentProperties` 只有 `structSize` 与 `name`
/// （`ARAInterface.h:670-682`），全 SDK 唯一的 `sampleRate` 在
/// `ARAAudioSourceProperties` 上（`:856`），那是**源**速率。唯一权威来源是 ARA 的
/// companion API —— 插件侧就是 VST3 `setupProcessing` 的 `ProcessSetup::sampleRate`。
///
/// 【为什么没有"推导"这一级】曾经有一级 `sampleCount / durationMod` 的推导，已删除。
/// 它在数学上**不可识别**：
///
/// ```text
/// 源跨度（源秒） = durPlay × playback_rate
///              = durPlay × (durMod / durPlay) × (D / sr)
///              = durMod × D / sr = sampleCount / sr      ← 与 D 无关，恒等式
/// ```
///
/// 也就是说，"素材率 ≠ 工程率的整条 region"与"被裁剪过的 region"给出的数据**完全一样**，
/// 任何判据都无法区分。真实反例：REAPER 夹具 44100 Hz / 88200 帧，region 取 0.5 s，
/// 推导给出 176400 Hz 并据此把源窗口放大 4 倍，参数曲线锚点随之错位。
/// 宁可没有校正（退回夹窗口、保音高），也不要一个会把未裁剪素材放大四倍的假信号。
pub fn resolve_project_rate(doc: &AraDocument) -> ProjectRate {
    if let Some(rate) = doc
        .project_sample_rate
        .filter(|rate| rate.is_finite() && PLAUSIBLE_SAMPLE_RATE_RANGE.contains(rate))
    {
        return ProjectRate {
            rate,
            origin: ProjectRateOrigin::Host,
        };
    }
    ProjectRate {
        rate: 0.0,
        origin: ProjectRateOrigin::Unknown,
    }
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

/// 一条 region 的播放速率。
///
/// 【为什么不能只用时长比值】`durationInModificationTime` 与 `durationInPlaybackTime`
/// 同在**工程**速率域，它们的比值只表达"宿主自己算出的拉伸"，**不含**素材率与工程率之差。
/// 素材 48 kHz、工程 44.1 kHz 且宿主没做拉伸时比值是 1，真实速率却是 44100/48000：
/// 31.7986 s 的源内容要铺满 34.6107 s 的工程时间。缺了 `project_rate / source_rate` 这一项，
/// 素材被按自身速率播完、剩下的工程时间留白，用户听感即"ARA 前后不一致"。
///
/// 拿不到工程速率时（`project_rate` 为 0 / 非有限）**不猜**，按宿主比值原样返回，
/// 由调用方走夹窗口降级。
fn region_playback_rate(region: &AraPlaybackRegion, source_rate: f64, project_rate: f64) -> f64 {
    if !(region.duration_in_playback_time.is_finite() && region.duration_in_playback_time > 0.0) {
        return 1.0;
    }
    let host_ratio = region.duration_in_modification_time / region.duration_in_playback_time;
    if !host_ratio.is_finite() {
        return 1.0;
    }
    if !(project_rate.is_finite() && project_rate > 0.0)
        || !(source_rate.is_finite() && source_rate > 0.0)
    {
        return host_ratio;
    }
    let rate = host_ratio * (project_rate / source_rate);
    if rate.is_finite() {
        rate
    } else {
        host_ratio
    }
}

/// 样本里是否存在真正的播放速率偏离 1 —— 即 ARA 是否真的表达了拉伸。
///
/// 【为什么不能直接比较两个时长】宿主按工程速率折算修改时长时（见 [`resolve_project_rate`]），
/// 两个时长会**相等**而真实速率不是 1。那条路径上时长比较给出 false，调用方据此走"直通"
/// 绕过内核拉伸，音频就是错的。所以这里用映射层同一条速率算式。
pub fn has_observed_time_stretch(doc: &AraDocument) -> bool {
    let project_rate = resolve_project_rate(doc).rate;
    doc.playback_regions.iter().any(|region| {
        let source_rate = doc
            .audio_sources
            .iter()
            .find(|source| source.persistent_id == region.audio_source_persistent_id)
            .map_or(0.0, |source| source.sample_rate);
        (region_playback_rate(region, source_rate, project_rate) - 1.0).abs() > 1e-9
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
    /// region 的源窗口整条落在源之外，没有任何可读素材。
    ///
    /// 【为什么与"源不存在"分开】成因不同、下一步也不同：源不存在是宿主模型有问题；
    /// 这一条是宿主给的修改时长时间整体越界（素材与工程采样率不同时宿主会按工程速率
    /// 折算修改时长）。合并会让日志指错方向。
    SourceWindowOutsideSource,
}

impl SkipReason {
    /// 语言无关的分类名，供日志与逐 clip 状态复用。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnknownRegionSequence => "unknown_region_sequence",
            Self::UnknownSource => "unknown_source",
            Self::UnknownModification => "unknown_modification",
            Self::NonPositivePlaybackDuration => "non_positive_duration",
            Self::SourceWindowOutsideSource => "source_window_outside_source",
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

/// 调用方提供的**稳定身份**，按 `doc.playback_regions` 的下标对齐。
///
/// 【为什么需要它】位置派生的身份（`ara-clip-{index+1}` / `ara-track-{N}`）在宿主
/// 删除、移动或重排 region / track 之后会**整体前移**，而参数表（`params_by_root_track`、
/// parameter atlas、`track_bindings`）全部按身份索引 —— 于是一次删除就让其余片段的编辑
/// 挂到错误的 clip 上，或干脆查不到身份。`region_key` 是 ARA 在同一文档内保证稳定的
/// 令牌，用它派生身份即可消除整类错位。
///
/// 【为什么 region 没有 persistentID】`ARAPlaybackRegionProperties` 只有
/// `transformationFlags`、四段时间、`regionSequenceRef`、`name`、`color`（`ARAInterface.h:1012-1073`）
/// —— **没有** persistentID。所以会话内身份只能取自 region key（model-ref 键）；
/// 跨会话身份另见 `TrackBindings` 的 (modification, sequence) 签名。
#[derive(Debug, Clone, Default)]
pub struct MappingIdentities {
    /// 每条 region 的 ARA region key（0 = 未知，退回位置身份）。
    pub region_keys: Vec<u64>,
    /// 每条 region sequence 的宿主 **model-ref key**（空串 = 未知，退回位置身份）。
    ///
    /// 【为什么不是 persistentID】`ARARegionSequenceProperties` 没有这个字段（见上方
    /// `AraRegionSequence` 的说明），宿主给的稳定令牌只有 model-ref key。
    pub sequence_ids: Vec<String>,
}

/// ARA 文档 → `TimelineState`（整份成功或整份失败）。
///
/// 供只需要时间线、不关心跳过明细的调用方使用；坏 region 会被静默丢弃。
/// 需要逐 region 明细时用 [`ara_document_to_timeline_reporting`]。
pub fn ara_document_to_timeline(doc: &AraDocument) -> Result<TimelineState, MappingError> {
    Ok(ara_document_to_timeline_reporting(doc)?.timeline)
}

/// ARA 文档 → `TimelineState`，带**稳定身份**（推荐入口）。
///
/// 与 [`ara_document_to_timeline_reporting`] 同口径，差别只在 clip / track 的身份来源：
/// 传了 [`MappingIdentities`] 就按 region key 与 sequence model-ref key 派生，
/// 否则退回位置派生。生产路径（`ModelHandle::remap_and_log`）必须用这一个。
pub fn ara_document_to_timeline_with_identities(
    doc: &AraDocument,
    identities: &MappingIdentities,
) -> Result<MappingOutcome, MappingError> {
    ara_document_to_timeline_reporting_with_identities(doc, Some(identities))
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
    ara_document_to_timeline_reporting_with_identities(doc, None)
}

/// [`ara_document_to_timeline_reporting`] 的实现；`identities` 为 `None` 时用位置身份。
fn ara_document_to_timeline_reporting_with_identities(
    doc: &AraDocument,
    identities: Option<&MappingIdentities>,
) -> Result<MappingOutcome, MappingError> {
    let project = resolve_project_rate(doc);
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
    for (position, sequence) in doc
        .musical_contexts
        .iter()
        .flat_map(|context| context.region_sequences.iter())
        .enumerate()
    {
        // 【身份用宿主的 model-ref key，不是 persistentID】`ARARegionSequenceProperties`
        // 只有 name / orderIndex / musicalContextRef / color（`ARAInterface.h:775-793`）
        // —— **没有** persistentID（audio source / modification / processing algorithm
        // 才有）。能拿到的稳定令牌是宿主的 model-ref key，由 `ModelHandle` 经
        // `MappingIdentities` 传入；拿不到时才退回位置，后者会在宿主删掉一条序列之后
        // 让其余 track 的身份整体前移。
        let id = identities
            .and_then(|ids| ids.sequence_ids.get(position))
            .filter(|id| !id.is_empty())
            .map(|id| format!("ara-sequence-{id}"))
            .unwrap_or_else(|| format!("ara-track-{position}"));
        // 【为什么 order 取宿主的 orderIndex 而不是枚举序号】枚举序号是"宿主交给我们的
        // 顺序"，而 `orderIndex` 是"宿主认为的轨道次序"。两者在宿主重排后不一致，而
        // 内核按 `order` 排序（`state/model.rs` 的 `siblings.sort_by_key`）—— 用枚举
        // 序号会让插件里的轨道顺序停在宿主重排之前。这里**只改排序值**：`track_ids`
        // 仍按扁平位置排列，因为 `region_sequence_of` 返回的就是扁平下标，
        // 重排数组会让 region→track 归属整体错位。
        tracks.push(json!({
            "id": id,
            "name": sequence.name.clone().unwrap_or_default(),
            "order": sequence.order_index,
        }));
        track_ids.push(id);
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

        // 【速率】宿主声明的时长比值，再乘"工程速率 / 源速率"的校正。后者修正的是
        // "宿主按工程速率折算修改时长"这一事实（见 `region_playback_rate`）：
        // 48 kHz 素材进 44.1 kHz 工程且宿主不拉伸时，比值 1 × 44100/48000 = 0.91875。
        let playback_rate = region_playback_rate(region, source.sample_rate, project.rate);
        // 【身份优先用 region key】位置派生的 id 会在宿主删掉靠前一条 region 之后整体前移，
        // 而所有参数表都按 clip id 索引 —— 那正是"编辑操作不更新 ARA"的直接机理。
        // key 是 ARA 在同一文档内保证稳定的令牌（宿主把它当作不透明句柄），
        // 销毁只摘除不复用已活 key，所以它不会因删除而重排。
        let clip_id = identities
            .and_then(|ids| ids.region_keys.get(index).copied())
            .filter(|key| *key != 0)
            .map(|key| format!("ara-region-{key}"))
            .unwrap_or_else(|| format!("ara-clip-{}", index + 1));
        let take_id = format!("{clip_id}-take-1");
        let clip_name = region
            .name
            .clone()
            .or_else(|| source.name.clone())
            .unwrap_or_else(|| clip_id.clone());

        // 【源窗口】宿主给出的修改时长时间可以**超出源本身**，而这不是它的错：素材采样率
        // 与工程采样率不同时，宿主把 `durationInModificationTime` 按**工程**速率折算 ——
        // 实测 Cubase 15：48 kHz / 1526332 帧素材放进 44.1 kHz 工程，现场读数是
        // `durationMod = durationPlay = 34.610703`，而 34.610703 × 44100 恰好等于源帧数；
        // 乘**源**速率 48000 则得 1661314，凭空多出 8.2%。
        //
        // 【有工程速率时按速率折算，而不是夹掉】`playback_rate` 已经把 44100/48000 带上，
        // 内核消费的源跨度是 `durationPlay × playback_rate` = 34.610703 × 0.91875 =
        // 31.7986 s，恰好等于整条源 —— **不多要一帧，也不少要一帧**。此时再夹到
        // `duration_seconds` 是恒等的，所以只在拿不到速率时才需要夹。
        //
        // 【为什么夹而不是报错】拿不到速率时（宿主没通报、也推导不出），夹是**保音高**的：
        // 源按自身速率播放，缺的部分留白（ARA 对"读到源尾之后"的定义就是静音）。此前这段
        // 越界一路带到渲染校验，被 `SnapshotError::InvalidGeometry` 拒绝，而那是**致命**的：
        // 整条时间线一条片段都发布不出来，用户看到"尚未应用：unsupported host region:
        // InvalidGeometry"。一个越界的源窗口绝不该赔上整条时间线。
        //
        // 【为什么只在时长已知时才夹】`duration_seconds` 由 `sampleCount / sampleRate` 推出，
        // 宿主没给样本数时它是 0（"未知"，不是"零长"）。把未知当零长会把每条 region 都夹成
        // 空、进而全部跳过 —— 那比原来的致命拒绝还糟。
        //
        // 【有工程速率时先折算再判断】宿主给的 `durationInModificationTime` 是**工程**速率秒，
        // 内核消费的源跨度是 `durationPlay × playback_rate`。速率已知时这个乘积恰好落在源内
        // （Cubase 现场：34.610703 × 0.91875 = 31.7986 = 整条源），**不该夹** ——
        // 夹了反而丢掉真实内容。所以判据从"声明窗口 vs 源长度"换成"折算后的窗口 vs 源长度"。
        let declared_end = region.start_in_modification_time + region.duration_in_modification_time;
        let rate_known = project.rate > 0.0 && playback_rate.is_finite() && playback_rate > 0.0;
        let consumed_end = if rate_known {
            region.start_in_modification_time + region.duration_in_playback_time * playback_rate
        } else {
            declared_end
        };
        let known_duration = source.duration_seconds > 0.0;
        let source_span_end = if known_duration {
            consumed_end.min(source.duration_seconds)
        } else {
            consumed_end
        };
        let source_span_start = if known_duration {
            region.start_in_modification_time.min(source_span_end)
        } else {
            region.start_in_modification_time
        };
        if known_duration && source_span_end <= source_span_start {
            // 整条 region 都落在源之外：没有可读素材。这与"源不存在"同级 —— 记因、跳过，
            // 不发布一条空 take 让下游去猜。
            skipped.push(SkippedRegion {
                index,
                reason: SkipReason::SourceWindowOutsideSource,
            });
            continue;
        }
        if source_span_end < consumed_end {
            log::warn!(
                "[ara] playback_region #{index}: consumed source window [{:.6}, {:.6}] exceeds the \
                 source ({:.6}s at {} Hz); clamping to the source so the tail renders as silence \
                 instead of failing the whole timeline",
                region.start_in_modification_time,
                consumed_end,
                source.duration_seconds,
                source.sample_rate,
            );
        }

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
                "source_start_sec": source_span_start,
                "source_end_sec": source_span_end,
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

    /// take 的 `playback_rate` 在 JSON 里是 **f32**（内核模型的列类型），
    /// 所以按 f32 精度比较，不能用 f64 的 1e-9。
    fn assert_rate(actual: f32, expected: f64) {
        assert!(
            (f64::from(actual) - expected).abs() < 1e-6,
            "expected {expected}, got {}",
            actual
        );
    }

    /// **Cubase 15 的现场读数**：48 kHz / 1526332 帧素材放进 44.1 kHz 工程。
    ///
    /// `durationMod = durationPlay = 34.610703`，而 34.610703 × 44100 恰好等于源帧数 ——
    /// 宿主把修改时长按**工程**速率折算。真实播放速率是 44100/48000 = 0.91875：
    /// 31.7986 s 的源内容要铺满 34.6107 s 的工程时间。
    fn cubase_rate_mismatched_document() -> AraDocument {
        let mut doc = ara_document_from_json(
            r#"{
                "documentName": "session",
                "audioSources": [
                    { "persistentID": "C:/media/Vocal-5-1.wav", "name": "vocal",
                      "sampleRate": 48000.0, "sampleCount": 1526332,
                      "durationSeconds": 31.798583333333334, "channelCount": 2,
                      "sampleAccessEnabled": true }
                ],
                "musicalContexts": [
                    { "name": "ctx", "regionSequences": [
                        { "name": "Vocal", "orderIndex": 0, "playbackRegionCount": 1 }
                    ] }
                ],
                "audioModifications": [
                    { "persistentID": "mod-1", "audioSourcePersistentID": "C:/media/Vocal-5-1.wav" }
                ],
                "playbackRegions": [
                    { "name": "vocal", "audioSourcePersistentID": "C:/media/Vocal-5-1.wav",
                      "audioModificationPersistentID": "mod-1", "regionSequenceIndex": 0,
                      "startInModificationTime": 0.0, "durationInModificationTime": 34.610703,
                      "startInPlaybackTime": 0.0, "durationInPlaybackTime": 34.610703 }
                ]
            }"#,
        )
        .expect("rate-mismatched document parses");
        doc.project_sample_rate = Some(44100.0);
        doc
    }

    /// 宿主通报工程速率时，48 kHz 素材进 44.1 kHz 工程必须得到 0.91875 的播放速率，
    /// 且**不能**再夹源窗口 —— 31.7986 s 的源内容要完整消费。
    ///
    /// 【为什么这是回归测试】此前 `playback_rate = durationMod / durationPlay`，两个数同在
    /// 工程速率域，比值恒为 1；真实速率被抹平后，素材按自身速率播完、剩下的工程时间留白，
    /// 用户听感是"ARA 前后不一致"。而夹窗口那个缓解恰好相反：它保音高、丢内容。
    #[test]
    fn a_rate_mismatched_source_is_stretched_not_clamped_when_the_host_reports_the_project_rate() {
        let doc = cubase_rate_mismatched_document();
        assert_eq!(
            resolve_project_rate(&doc).origin,
            ProjectRateOrigin::Host,
            "宿主通报的速率必须优先于推导"
        );
        assert!(
            has_observed_time_stretch(&doc),
            "两个时长相等但速率不是 1：这条 region 确实在拉伸"
        );

        let outcome = ara_document_to_timeline_reporting(&doc).expect("document maps");
        assert!(outcome.skipped.is_empty(), "{:?}", outcome.skipped);
        let clip = &outcome.timeline.clips[0];
        let take = &clip.takes[0];
        assert_rate(take.playback_rate, 44100.0 / 48000.0);
        assert_close(take.source_start_sec, 0.0);
        // 源窗口是**折算后**的消费跨度：34.610703 × 0.91875 ≈ 31.7986 s，正好整条源。
        // 用 1e-5 而不是 1e-9：`playback_rate` 经 JSON 落到 f32，回读时已损失一位。
        assert!(
            (take.source_end_sec - 31.798583333333334).abs() < 1e-5,
            "got {}",
            take.source_end_sec
        );
        assert_close(clip.length_sec, 34.610703);
    }

    /// 拿不到工程速率时**不猜**：退回夹窗口 + 保音高的既有降级，并记下现场数字。
    #[test]
    fn a_rate_mismatched_source_falls_back_to_clamping_when_no_project_rate_is_known() {
        let mut doc = cubase_rate_mismatched_document();
        doc.project_sample_rate = None;
        // 推导路径被 `startInModificationTime > 0` 之外的判据挡住前先确认：
        // 这条 region 覆盖整条源，所以推导**会**成功 —— 要测"未知"必须连源帧数一并抹掉。
        doc.audio_sources[0].sample_count = 0;
        assert_eq!(
            resolve_project_rate(&doc).origin,
            ProjectRateOrigin::Unknown
        );

        let outcome = ara_document_to_timeline_reporting(&doc).expect("document maps");
        assert!(outcome.skipped.is_empty(), "{:?}", outcome.skipped);
        let take = &outcome.timeline.clips[0].takes[0];
        // 速率退回宿主比值（1.0），源窗口夹到源的真实长度。
        assert_rate(take.playback_rate, 1.0);
        assert_close(take.source_end_sec, 31.798583333333334);
    }

    /// 宿主没通报工程速率时**不猜** —— 这条契约曾经被违反过，必须钉住。
    ///
    /// 【为什么不能推导】`sampleCount / durationMod` 看似给出工程速率，但它在数学上
    /// **不可识别**：源跨度折算后恒等于 `sampleCount / sr`，与推导值无关（见
    /// [`resolve_project_rate`] 的推导）。也就是说"素材率 ≠ 工程率的整条 region"与
    /// "被裁剪过的 region"给出的数据**完全一样**。真实反例是 REAPER 夹具：44100 Hz /
    /// 88200 帧、region 取 0.5 s，推导给出 176400 Hz 并据此把源窗口放大 4 倍，
    /// 参数曲线锚点随之整体错位（`source_parameter_atlas_moves_and_stretches_…`
    /// 因此由 0.5 步进变成 0.125 步进）。
    ///
    /// 所以这里断言：没有宿主通报就没有速率，映射退回夹窗口 + 保音高。
    #[test]
    fn no_project_rate_means_no_correction_rather_than_a_guess() {
        let mut doc = cubase_rate_mismatched_document();
        doc.project_sample_rate = None;
        let resolved = resolve_project_rate(&doc);
        assert_eq!(resolved.origin, ProjectRateOrigin::Unknown);
        assert_eq!(resolved.rate, 0.0);

        let take = &ara_document_to_timeline_reporting(&doc)
            .expect("document maps")
            .timeline
            .clips[0]
            .takes[0];
        // 速率退回宿主比值，源窗口夹到源的真实长度 —— 音高保住了，尾部留白。
        assert_rate(take.playback_rate, 1.0);
        assert_close(take.source_end_sec, 31.798583333333334);
    }

    /// 宿主自己声明的拉伸必须**保留**，不能被速率校正抹掉。
    #[test]
    fn a_host_declared_stretch_survives_the_rate_correction() {
        let doc = probe_shaped_document();
        let clip = &ara_document_to_timeline(&doc).unwrap().clips[0];
        // 源率 48000、工程率未知 ⇒ 校正因子缺席，比值 0.5 原样保留。
        assert_rate(clip.takes[0].playback_rate, 0.5);
    }

    /// 同源同工程率下，宿主的拉伸比值与速率校正**相乘**。
    #[test]
    fn the_rate_correction_multiplies_the_host_ratio() {
        let mut doc = probe_shaped_document();
        doc.project_sample_rate = Some(44100.0);
        // 48 kHz 源、44.1 kHz 工程、宿主声明 0.5 ⇒ 0.5 × 44100/48000。
        let clip = &ara_document_to_timeline(&doc).unwrap().clips[0];
        assert_rate(clip.takes[0].playback_rate, 0.5 * 44100.0 / 48000.0);
    }

    /// 明显荒谬的工程速率必须被拒，不能拿去折算任何东西。
    #[test]
    fn an_implausible_reported_project_rate_is_ignored() {
        let mut doc = cubase_rate_mismatched_document();
        doc.project_sample_rate = Some(1.0);
        doc.audio_sources[0].sample_count = 0;
        assert_eq!(
            resolve_project_rate(&doc).origin,
            ProjectRateOrigin::Unknown,
            "1 Hz 不是可信的工程速率"
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

    /// 宿主给的修改时长可以超出源本身，而这不是它的错：素材采样率与工程采样率不同时，
    /// 宿主按**工程**速率折算修改时长。
    ///
    /// 实测 Cubase 15：48kHz / 1526332 帧素材进 44.1kHz 工程，宿主给
    /// `durationMod = durationPlay = 34.610703`，而 34.610703 × 44100 = 1526332，恰好是
    /// 源帧数；乘源速率 48000 则得 1661314，凭空多出 8.2%。
    ///
    /// 旧行为把这个越界一路带到渲染校验，被 `SnapshotError::InvalidGeometry` **致命**
    /// 拒绝 —— 整条时间线一条片段都发布不出来，用户看到
    /// "尚未应用：unsupported host region: InvalidGeometry"。这里钉住：越界绝不赔上整条时间线。
    ///
    /// 【本用例为什么退回夹窗口】这个夹具没有宿主通报的工程速率（`project_sample_rate`
    /// 缺席），而推导路径已被证明不可识别（见 [`resolve_project_rate`]）。所以速率保持
    /// 宿主比值 1.0、源窗口夹到源长度 —— 这是**已知会丢尾部内容**的降级，但它保音高、
    /// 不发明拉伸。真正带速率的那条路径由
    /// `a_rate_mismatched_source_is_stretched_not_clamped_…` 覆盖。
    #[test]
    fn a_modification_window_beyond_the_source_is_not_fatal() {
        // 源：48000 Hz / 144000 帧 = 3.0 s。region 声明的修改窗口 3.2635 s = 144000/44100，
        // 正是"宿主按工程速率折算"的形状。
        let doc = ara_document_from_json(
            r#"{
                "audioSources": [
                    { "persistentID": "s", "sampleRate": 48000.0, "sampleCount": 144000,
                      "durationSeconds": 3.0, "channelCount": 2, "sampleAccessEnabled": true }
                ],
                "audioModifications": [
                    { "persistentID": "m", "audioSourcePersistentID": "s" }
                ],
                "musicalContexts": [
                    { "regionSequences": [{ "playbackRegionCount": 1 }] }
                ],
                "playbackRegions": [
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "startInModificationTime": 0.0, "durationInModificationTime": 3.2635,
                      "startInPlaybackTime": 0.0, "durationInPlaybackTime": 3.2635 }
                ]
            }"#,
        )
        .expect("document parses");

        let outcome = ara_document_to_timeline_reporting(&doc).expect("document maps");
        // 关键断言：**不跳过、不失败** —— clip 必须发布出来。
        assert!(
            outcome.skipped.is_empty(),
            "a window past the source must not drop the clip: {:?}",
            outcome.skipped.iter().map(|s| s.reason).collect::<Vec<_>>()
        );
        assert_eq!(outcome.timeline.clips.len(), 1);

        let clip = &outcome.timeline.clips[0];
        // 播放几何保持宿主原值：clip 仍是 3.2635 s。
        assert_close(clip.length_sec, 3.2635);
        // 没有工程速率 ⇒ 不猜，速率保持宿主比值。
        assert_rate(clip.takes[0].playback_rate, 1.0);
        assert_close(clip.source_start_sec, 0.0);
        // 源窗口夹到源的真实长度：越界部分由渲染读成静音，而不是去读不存在的样本。
        assert_close(clip.source_end_sec, 3.0);
    }

    /// **身份必须来自 region key，而不是 region 在文档里的位置。**
    ///
    /// 【为什么这是回归测试】`ara-clip-{index+1}` 是位置派生的：宿主删掉靠前一条 region
    /// 之后，其后所有 clip 的 id 整体前移，而参数表（`params_by_root_track`、
    /// parameter atlas、`track_bindings`）全部按 id 索引 —— 于是编辑挂到错误的 clip 上，
    /// 或干脆查不到身份。这正是 Cubase 下"编辑操作不更新 ARA"的直接机理。
    ///
    /// 这条测试构造"删掉第一条 region"前后的两份文档，断言**存活** clip 的身份不变。
    #[test]
    fn clip_identity_comes_from_the_region_key_not_the_position() {
        // 三条 region，key 分别是 11 / 22 / 33（同一条源、同一条修改）。
        let mut doc = ara_document_from_json(
            r#"{
                "audioSources": [
                    { "persistentID": "s", "sampleRate": 44100.0, "sampleCount": 88200,
                      "durationSeconds": 2.0, "channelCount": 2, "sampleAccessEnabled": true }
                ],
                "audioModifications": [
                    { "persistentID": "m", "audioSourcePersistentID": "s" }
                ],
                "musicalContexts": [
                    { "regionSequences": [{ "name": "T", "orderIndex": 0, "playbackRegionCount": 3 }] }
                ],
                "playbackRegions": [
                    { "name": "a", "audioSourcePersistentID": "s",
                      "audioModificationPersistentID": "m", "regionSequenceIndex": 0,
                      "startInModificationTime": 0.0, "durationInModificationTime": 0.5,
                      "startInPlaybackTime": 0.0, "durationInPlaybackTime": 0.5 },
                    { "name": "b", "audioSourcePersistentID": "s",
                      "audioModificationPersistentID": "m", "regionSequenceIndex": 0,
                      "startInModificationTime": 0.5, "durationInModificationTime": 0.5,
                      "startInPlaybackTime": 0.5, "durationInPlaybackTime": 0.5 },
                    { "name": "c", "audioSourcePersistentID": "s",
                      "audioModificationPersistentID": "m", "regionSequenceIndex": 0,
                      "startInModificationTime": 1.0, "durationInModificationTime": 0.5,
                      "startInPlaybackTime": 1.0, "durationInPlaybackTime": 0.5 }
                ]
            }"#,
        )
        .expect("document parses");
        let identities = MappingIdentities {
            region_keys: vec![11, 22, 33],
            sequence_ids: vec![String::new()],
        };
        let before =
            ara_document_to_timeline_with_identities(&doc, &identities).expect("document maps");
        assert_eq!(
            before
                .timeline
                .clips
                .iter()
                .map(|c| c.id.as_str())
                .collect::<Vec<_>>(),
            vec!["ara-region-11", "ara-region-22", "ara-region-33"]
        );

        // 宿主销毁了第一条 region：文档里只剩 b 与 c，而**它们的 key 不变**。
        doc.playback_regions.remove(0);
        let after = ara_document_to_timeline_with_identities(
            &doc,
            &MappingIdentities {
                region_keys: vec![22, 33],
                sequence_ids: vec![String::new()],
            },
        )
        .expect("document maps");
        assert_eq!(
            after
                .timeline
                .clips
                .iter()
                .map(|c| c.id.as_str())
                .collect::<Vec<_>>(),
            vec!["ara-region-22", "ara-region-33"],
            "删除靠前的 region 不能改变其余 clip 的身份"
        );
        // 几何也跟着走：b 仍在 0.5 s 起。
        assert_close(after.timeline.clips[0].start_sec, 0.5);
    }

    /// **track 的排序值取宿主的 `orderIndex`，而 region→track 归属仍按扁平位置。**
    ///
    /// 【为什么两件事必须分开】内核按 `order` 排序显示轨道，所以 `orderIndex` 是"宿主
    /// 认为的次序"；而 `region_sequence_of` 返回的是**扁平位置**下标，`track_ids` 必须
    /// 按同一顺序排列，否则 region 会挂到错误的轨道上。把数组按 orderIndex 重排会同时
    /// 破坏归属 —— 所以这里只改排序值。
    #[test]
    fn track_order_follows_the_host_order_index_while_attribution_stays_positional() {
        let doc = ara_document_from_json(
            r#"{
                "audioSources": [
                    { "persistentID": "s", "sampleRate": 44100.0, "sampleCount": 88200,
                      "durationSeconds": 2.0, "channelCount": 2, "sampleAccessEnabled": true }
                ],
                "audioModifications": [
                    { "persistentID": "m", "audioSourcePersistentID": "s" }
                ],
                "musicalContexts": [
                    { "regionSequences": [
                        { "name": "second", "orderIndex": 5, "playbackRegionCount": 1 },
                        { "name": "first", "orderIndex": 1, "playbackRegionCount": 1 }
                    ] }
                ],
                "playbackRegions": [
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "regionSequenceIndex": 0, "startInModificationTime": 0.0,
                      "durationInModificationTime": 0.5, "startInPlaybackTime": 0.0,
                      "durationInPlaybackTime": 0.5 },
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "regionSequenceIndex": 1, "startInModificationTime": 0.5,
                      "durationInModificationTime": 0.5, "startInPlaybackTime": 0.5,
                      "durationInPlaybackTime": 0.5 }
                ]
            }"#,
        )
        .expect("document parses");
        let timeline = ara_document_to_timeline(&doc).expect("document maps");
        // 排序值来自 orderIndex（5 与 1），不是枚举序号（0 与 1）。
        assert_eq!(timeline.tracks[0].order, 5);
        assert_eq!(timeline.tracks[1].order, 1);
        // 而归属仍按位置：第 0 条 region 挂第 0 条 track，第 1 条挂第 1 条。
        assert_eq!(timeline.clips[0].track_id, timeline.tracks[0].id);
        assert_eq!(timeline.clips[1].track_id, timeline.tracks[1].id);
    }

    /// 没有身份信息时退回位置派生 —— 探针/测试夹具仍可工作。
    #[test]
    fn identity_falls_back_to_the_position_when_no_key_is_supplied() {
        let doc = probe_shaped_document();
        let timeline = ara_document_to_timeline(&doc).expect("document maps");
        assert_eq!(timeline.clips[0].id, "ara-clip-1");
        assert_eq!(timeline.tracks[0].id, "ara-track-0");
    }

    /// 整条 region 都落在源之外时跳过该 clip —— 与"源不存在"分开记因，但同样不失败。
    #[test]
    fn a_region_entirely_outside_the_source_is_skipped_with_its_own_reason() {
        let doc = ara_document_from_json(
            r#"{
                "audioSources": [
                    { "persistentID": "s", "sampleRate": 48000.0, "sampleCount": 144000,
                      "durationSeconds": 3.0, "channelCount": 2, "sampleAccessEnabled": true }
                ],
                "audioModifications": [
                    { "persistentID": "m", "audioSourcePersistentID": "s" }
                ],
                "musicalContexts": [
                    { "regionSequences": [{ "playbackRegionCount": 1 }] }
                ],
                "playbackRegions": [
                    { "audioSourcePersistentID": "s", "audioModificationPersistentID": "m",
                      "startInModificationTime": 5.0, "durationInModificationTime": 1.0,
                      "startInPlaybackTime": 0.0, "durationInPlaybackTime": 1.0 }
                ]
            }"#,
        )
        .expect("document parses");

        let outcome = ara_document_to_timeline_reporting(&doc).expect("document maps");
        assert!(outcome.timeline.clips.is_empty());
        assert_eq!(outcome.skipped.len(), 1);
        assert_eq!(
            outcome.skipped[0].reason,
            SkipReason::SourceWindowOutsideSource
        );
        assert_eq!(
            SkipReason::SourceWindowOutsideSource.as_str(),
            "source_window_outside_source"
        );
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
