use crate::midi_import::MidiNoteEvent;
use crate::project::CustomScale;
use crate::state::ClipFormantMorph;
use crate::time_stretch::UserStretchAlgorithm;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PitchRange {
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ProjectMetaPayload {
    pub name: String,
    pub path: Option<String>,
    pub dirty: bool,
    pub recent: Vec<String>,
    pub notes_markdown: String,
    pub base_scale: String,
    pub use_custom_scale: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_scale: Option<CustomScale>,
    pub beats_per_bar: u32,
    pub time_signature_denominator: u32,
    pub grid_size: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stretch_algorithm_override: Option<UserStretchAlgorithm>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hifigan_mel_stretch_override: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct TimelineTrack {
    pub id: String,
    pub name: String,
    pub parent_id: Option<String>,
    pub depth: Option<u32>,
    pub child_track_ids: Option<Vec<String>>,
    pub muted: bool,
    pub solo: bool,
    pub volume: f32,

    pub compose_enabled: bool,
    pub pitch_analysis_algo: String,

    /// 轨道主题色，hex 字符串，如 "#4f8ef7"
    #[serde(default)]
    pub color: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct TimelineClipTake {
    pub id: String,
    #[serde(default)]
    pub name: String,
    pub gain: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_path_relative: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_frames: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_sample_rate: Option<u32>,
    pub source_start_sec: f64,
    pub source_end_sec: f64,
    pub playback_rate: f32,
    pub reversed: bool,
    pub loop_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub midi_note_data: Option<Vec<MidiNoteEvent>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub midi_fill_gaps: Option<bool>,
}

impl TimelineClipTake {
    /// 完整转换（含 MIDI 音符数据）—— 用于全量 payload。
    pub fn from_take(take: &crate::state::ClipTake, include_midi: bool) -> Self {
        Self {
            id: take.id.clone(),
            name: take.name.clone(),
            gain: take.gain,
            source_path: take.source_path.clone(),
            source_path_relative: take.source_path_relative.clone(),
            duration_sec: take.duration_sec,
            duration_frames: take.duration_frames,
            source_sample_rate: take.source_sample_rate,
            source_start_sec: take.source_start_sec,
            source_end_sec: take.source_end_sec,
            playback_rate: take.playback_rate,
            reversed: take.reversed,
            loop_enabled: take.loop_enabled,
            midi_note_data: if include_midi {
                take.midi_note_data.clone()
            } else {
                None
            },
            midi_fill_gaps: if take.midi_note_data.is_some() {
                Some(take.midi_fill_gaps)
            } else {
                None
            },
        }
    }
}

impl From<&crate::state::ClipTake> for TimelineClipTake {
    fn from(take: &crate::state::ClipTake) -> Self {
        Self::from_take(take, true)
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct TimelineClip {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_id: Option<String>,
    pub track_id: String,
    pub name: String,
    pub start_sec: f64,
    pub length_sec: f64,
    pub color: String,

    /// 全部 take（active take 的缓存字段不含在此，避免重复传输波形等大数据）。
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub takes: Vec<TimelineClipTake>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_take_id: Option<String>,

    pub source_path: Option<String>,
    pub source_path_relative: Option<String>,
    pub duration_sec: Option<f64>,
    pub duration_frames: Option<u64>,
    pub source_sample_rate: Option<u32>,
    pub waveform_preview: Option<Vec<f32>>,
    pub pitch_range: Option<PitchRange>,

    pub gain: Option<f32>,
    pub muted: Option<bool>,
    pub source_start_sec: Option<f64>,
    pub source_end_sec: Option<f64>,
    pub playback_rate: Option<f32>,
    /// Clip 级播放倍率；实际速率 = clip_playback_rate × active take playback_rate。
    pub clip_playback_rate: Option<f32>,
    pub reversed: Option<bool>,
    /// Loop（循环源）属性：超出源媒体区间时按周期回绕产生循环内容。
    #[serde(default)]
    pub loop_enabled: bool,
    /// 吸附偏移（秒）：相对 Clip 起点的偏移，默认 0；旧工程缺失补齐为 0。
    pub snap_offset_sec: Option<f64>,
    pub fade_in_sec: Option<f64>,
    pub fade_out_sec: Option<f64>,
    /// REAPER 浮点形状 id（整数 0..6 为标准七预设；小数变体原样透传）。
    pub fade_in_shape: Option<f64>,
    pub fade_out_shape: Option<f64>,
    /// 曲率（REAPER D_FADEINDIR），范围 [-1, 1]。
    pub fade_in_dir: Option<f64>,
    pub fade_out_dir: Option<f64>,
    /// 自动交叉淡化长度（秒），与手动 fade（fade_in_sec/fade_out_sec）分离存储。
    pub auto_fade_in_sec: Option<f64>,
    pub auto_fade_out_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formant_morph: Option<ClipFormantMorphPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub midi_note_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub midi_note_data: Option<Vec<MidiNoteEvent>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub midi_fill_gaps: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ClipFormantMorphPayload {
    pub enabled: bool,
    pub target_f1_hz: f64,
    pub target_f2_hz: f64,
    pub strength: f64,
}

impl From<&ClipFormantMorph> for ClipFormantMorphPayload {
    fn from(value: &ClipFormantMorph) -> Self {
        Self {
            enabled: value.enabled,
            target_f1_hz: value.target_f1_hz,
            target_f2_hz: value.target_f2_hz,
            strength: value.strength,
        }
    }
}

/// 源文件变更检测结果：当用户切换回窗口时，检测导入的媒体文件是否被外部替换或删除。
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct SourceFileChangePayload {
    pub clip_id: String,
    pub clip_name: String,
    pub source_path: String,
    /// "deleted" | "modified"
    pub change: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct CheckSourceFilesChangedPayload {
    pub changed: Vec<SourceFileChangePayload>,
}

/// “搜索文件夹”的匹配模式。
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SearchSourceFileMode {
    /// 精准文件名匹配：按源文件的完整文件名（含扩展名）搜索。
    #[serde(rename = "file_name")]
    ByFileName,
    /// 文件扩展名 + 哈希匹配：扫描所有相同扩展名的文件，按内容指纹匹配。
    #[serde(rename = "extension_hash")]
    ByExtensionHash,
}

/// 按文件名称搜索到的候选源文件。
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct SourceFileMatchCandidatePayload {
    /// 候选文件的绝对路径。
    pub path: String,
    /// 内容指纹是否与工程记录的源文件完全一致。
    pub exact_hash: bool,
}

/// 批量搜索候选源文件的返回载荷，key 为 clip_id。
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct SearchSourceFileMatchesPayload {
    pub matches: std::collections::HashMap<String, Vec<SourceFileMatchCandidatePayload>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TempoScalePayload {
    pub key: Option<String>,
    pub name: Option<String>,
    pub notes: Option<Vec<u8>>,
}

/// Tempo Map 变化点（前端载荷，camelCase；同时用于 `set_timeline_tempo_map` 命令参数）。
/// 拍号 numerator/denominator 为 null 表示“跟随之前的拍号”。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TempoPointPayload {
    pub id: String,
    pub position_sec: f64,
    pub bpm: f64,
    pub numerator: Option<u32>,
    pub denominator: Option<u32>,
    pub scale: Option<TempoScalePayload>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct TimelineStatePayload {
    pub ok: bool,
    pub tracks: Vec<TimelineTrack>,
    pub clips: Vec<TimelineClip>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_clip_ids: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_track_ids: Option<Vec<String>>,
    pub selected_track_id: Option<String>,
    pub selected_clip_id: Option<String>,
    pub bpm: f64,
    pub playhead_sec: f64,
    pub project_sec: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<ProjectMetaPayload>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub missing_files: Option<Vec<String>>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub disabled_group_ids: Vec<String>,

    /// Tempo Map（None = 无 Tempo Map）。始终序列化该字段，保证前端能区分“无 Tempo Map”。
    #[serde(default)]
    pub tempo_map: Option<Vec<TempoPointPayload>>,
}

/// `open_project` 的返回载荷。
///
/// 除正常时间轴状态外，还可能携带“工程文件版本高于当前程序”的确认信息。
/// 该确认信息出现时后端尚未加载工程，前端应展示警告并在用户确认后以
/// `force = true` 再次调用 `open_project`。
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct OpenProjectPayload {
    #[serde(flatten)]
    pub timeline: TimelineStatePayload,
    /// 打开失败的具体原因（文件不存在/无权限/解析失败等），前端用于展示。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_version_too_new: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_file_version: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_project_file_version: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct RuntimeInfoPayload {
    pub ok: bool,
    pub device: String,
    pub model_loaded: bool,
    pub audio_loaded: bool,
    pub has_synthesized: bool,
    pub is_playing: Option<bool>,
    pub playback_target: Option<String>,
    pub timeline: Option<TimelineStatePayload>,
    /// Display name of the execution provider the live vocoder session is
    /// actually running on, e.g. "CoreML", "WebGPU", "DirectML" or "CPU".
    /// Empty until the first session has been built (or after an EP switch
    /// that has not been picked up by a render yet).
    pub gpu_backend: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelConfigPayload {
    pub ok: bool,
    pub config: ModelConfig,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelConfig {
    pub audio_sample_rate: u32,
    pub audio_num_mel_bins: u32,
    pub hop_size: u32,
    pub fmin: f32,
    pub fmax: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct PlaybackStatePayload {
    pub ok: bool,
    pub is_playing: bool,
    pub target: Option<String>,
    pub base_sec: f64,
    pub position_sec: f64,
    pub duration_sec: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct DebugRealtimeRenderStatsPayload {
    pub ok: bool,
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<RealtimeRenderStatsPayload>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct RealtimeRenderStatsPayload {
    pub callbacks_total: u64,
    pub callbacks_silenced_not_playing: u64,

    pub pitch_callbacks_total: u64,
    pub pitch_callbacks_silenced_waiting: u64,
    pub pitch_callbacks_prime_waiting: u64,
    pub pitch_callbacks_fallback_mixed: u64,

    pub base_callbacks_total: u64,
    pub base_callbacks_covered: u64,
    pub base_callbacks_fallback_mixed: u64,

    pub legacy_callbacks_mixed: u64,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamReferenceKind {
    SourceCurve,
    DefaultValue,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ParamFramesPayload {
    pub ok: bool,
    pub root_track_id: String,
    pub param: String,
    pub frame_period_ms: f64,
    pub start_frame: u32,
    pub orig: Vec<f32>,
    pub edit: Vec<f32>,
    /// 二进制编码的曲线数据（请求带 `binary=true` 时返回）。
    ///
    /// 协议见 `commands/params.rs::encode_param_frames_binary`，与前端
    /// `paramFramesBinaryCodec.ts` 配套。启用时 `orig`/`edit` 为空。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary: Option<String>,
    pub reference_kind: ParamReferenceKind,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub analysis_pending: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub analysis_progress: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub pitch_edit_user_modified: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub pitch_edit_backend_available: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct StaticParamValuePayload {
    pub ok: bool,
    pub root_track_id: String,
    pub param: String,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ProcessAudioPayload {
    pub ok: bool,
    pub audio: Option<ProcessedAudio>,
    pub feature: Option<AudioFeature>,
    pub timeline: Option<TimelineStatePayload>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ProcessedAudio {
    pub path: String,
    pub sample_rate: u32,
    pub duration_sec: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct AudioFeature {
    pub mel_shape: Option<Vec<u32>>,
    pub f0_frames: Option<u32>,
    pub segment_count: Option<u32>,
    pub segments_preview: Option<Vec<Vec<f32>>>,
    pub waveform_preview: Option<Vec<f32>>,
    pub pitch_range: Option<PitchRange>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct SynthesizePayload {
    pub ok: bool,
    pub sample_rate: u32,
    pub num_samples: u32,
    pub duration_sec: f64,
}
