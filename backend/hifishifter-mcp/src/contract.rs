use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// All positions are absolute project seconds; source positions are seconds from the file start.
/// Track IDs may refer to a client_id created earlier in the same atomic edit.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EditRequest {
    /// Opaque version returned by hfs_get_context/hfs_query. Never guess this value.
    pub expected_revision: String,
    /// Retry the identical request with the same key after a lost response.
    pub idempotency_key: String,
    pub label: Option<String>,
    pub operations: Vec<Operation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    CreateTrack {
        client_id: String,
        name: String,
        parent_id: Option<String>,
        index: Option<usize>,
    },
    PatchTrack {
        track_id: String,
        patch: TrackPatch,
    },
    MoveTrack {
        track_id: String,
        parent_id: Option<String>,
        index: usize,
    },
    DuplicateTrack {
        track_id: String,
    },
    RemoveTrack {
        track_id: String,
    },
    ImportClip {
        client_id: String,
        track_id: String,
        path: String,
        start_sec: f64,
        source_start_sec: Option<f64>,
        length_sec: Option<f64>,
        playback_rate: Option<f32>,
    },
    PatchClip {
        clip_id: String,
        patch: ClipPatch,
    },
    MoveClip {
        clip_id: String,
        track_id: Option<String>,
        start_sec: f64,
    },
    SplitClip {
        clip_id: String,
        at_sec: f64,
    },
    RemoveClip {
        clip_id: String,
    },
    SetActiveTake {
        clip_id: String,
        take_id: String,
    },
    RenameTake {
        clip_id: String,
        take_id: String,
        name: String,
    },
    DuplicateTake {
        clip_id: String,
        take_id: String,
    },
    RemoveTake {
        clip_id: String,
        take_id: String,
    },
    PackTakes {
        clip_ids: Vec<String>,
    },
    ExplodeTakes {
        clip_id: String,
    },
    DuplicateClips {
        clip_ids: Vec<String>,
        delta_sec: f64,
    },
    GroupClips {
        clip_ids: Vec<String>,
        grouped: bool,
    },
    WriteCurve {
        track_id: String,
        param: String,
        start_frame: u32,
        values: Vec<f32>,
    },
    RestoreCurve {
        track_id: String,
        param: String,
        start_frame: u32,
        frame_count: u32,
    },
    SetStatic {
        track_id: String,
        param: String,
        value: f64,
    },
    TransformCurve {
        track_id: String,
        param: String,
        start_sec: f64,
        end_sec: f64,
        transform: CurveTransform,
    },
    SetTempo {
        points: Vec<TempoPoint>,
    },
    SetProjectLength {
        seconds: f64,
    },
    SetParamBlocks {
        track_id: String,
        layout: serde_json::Value,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TrackPatch {
    pub name: Option<String>,
    pub color: Option<String>,
    pub muted: Option<bool>,
    pub solo: Option<bool>,
    pub volume: Option<f32>,
    pub compose_enabled: Option<bool>,
    /// Use an algorithm ID returned by hfs_get_edit_capabilities. Omit to keep App settings.
    pub algorithm: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ClipPatch {
    pub name: Option<String>,
    pub start_sec: Option<f64>,
    pub length_sec: Option<f64>,
    pub source_start_sec: Option<f64>,
    pub playback_rate: Option<f32>,
    pub gain: Option<f32>,
    pub item_gain: Option<f32>,
    pub muted: Option<bool>,
    pub reversed: Option<bool>,
    pub loop_enabled: Option<bool>,
    pub color: Option<String>,
    pub fade_in_sec: Option<f64>,
    pub fade_out_sec: Option<f64>,
    pub fade_in_shape: Option<f64>,
    pub fade_out_shape: Option<f64>,
    pub fade_in_dir: Option<f64>,
    pub fade_out_dir: Option<f64>,
    pub auto_fade_in_sec: Option<f64>,
    pub auto_fade_out_sec: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum CurveTransform {
    Set {
        value: f32,
    },
    Ramp {
        from: f32,
        to: f32,
    },
    Shift {
        amount: f32,
    },
    /// Pitch only: shift the source contour's voiced median onto target_midi.
    ShiftToNote {
        target_midi: f32,
    },
    /// Pitch only: flatten reliable voiced samples; unvoiced samples remain unedited.
    FlattenToNote {
        target_midi: f32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TempoPoint {
    pub position_sec: f64,
    pub bpm: f64,
    pub numerator: Option<u32>,
    pub denominator: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum QueryKind {
    Timeline,
    Tracks,
    Clips,
    Curve,
    Static,
    ParamBlocks,
    Settings,
    Playback,
    History,
    Render,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QueryRequest {
    pub kind: QueryKind,
    pub track_id: Option<String>,
    pub clip_id: Option<String>,
    pub param: Option<String>,
    pub start_frame: Option<u32>,
    pub frame_count: Option<u32>,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TargetRequest {
    pub track_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HistoryRequest {
    pub expected_revision: String,
    pub action: HistoryAction,
}
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HistoryAction {
    Undo,
    Redo,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TransportRequest {
    pub action: TransportAction,
    pub position_sec: Option<f64>,
}
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TransportAction {
    Play,
    Stop,
    Seek,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SaveRequest {
    pub expected_revision: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SettingsRequest {
    pub change: SettingsChange,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum SettingsChange {
    Read,
    /// Both override fields are the complete desired project override; null means inherit App defaults.
    SetProjectStretch {
        expected_revision: String,
        algorithm_override: Option<StretchAlgorithm>,
        hifigan_mel_stretch_override: Option<bool>,
    },
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StretchAlgorithm {
    Signalsmith,
    Soundtouch,
    Linear,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ViewRequest {
    /// Omit all fields to read the active editor view. This changes selection/view only, never audio.
    pub selected_track_id: Option<String>,
    pub selected_clip_id: Option<String>,
    pub active_param: Option<String>,
    pub reference_root_track_ids: Option<Vec<String>>,
    pub tool_mode: Option<ViewTool>,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ViewTool {
    Select,
    Draw,
    Line,
    Vibrato,
}

#[derive(Debug, Clone)]
pub enum EditorRequest {
    Context,
    Capabilities(TargetRequest),
    Query(QueryRequest),
    Edit(EditRequest),
    History(HistoryRequest),
    Transport(TransportRequest),
    Save(SaveRequest),
    Settings(SettingsRequest),
    View(ViewRequest),
}
