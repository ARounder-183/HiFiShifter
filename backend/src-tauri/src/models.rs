use crate::project::CustomScale;
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
    pub base_scale: String,
    pub use_custom_scale: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_scale: Option<CustomScale>,
    pub beats_per_bar: u32,
    pub grid_size: String,
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
pub struct TimelineClip {
    pub id: String,
    pub track_id: String,
    pub name: String,
    pub start_sec: f64,
    pub length_sec: f64,
    pub color: String,

    pub source_path: Option<String>,
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
    pub fade_in_sec: Option<f64>,
    pub fade_out_sec: Option<f64>,
    pub fade_in_curve: Option<String>,
    pub fade_out_curve: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct TimelineStatePayload {
    pub ok: bool,
    pub tracks: Vec<TimelineTrack>,
    pub clips: Vec<TimelineClip>,
    pub selected_track_id: Option<String>,
    pub selected_clip_id: Option<String>,
    pub bpm: f64,
    pub playhead_sec: f64,
    pub project_sec: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<ProjectMetaPayload>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub missing_files: Option<Vec<String>>,
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
