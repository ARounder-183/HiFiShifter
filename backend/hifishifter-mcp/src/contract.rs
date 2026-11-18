//! Strongly typed MCP tool contracts for HiFiShifter.
//!
//! Every request struct enforces `#[serde(deny_unknown_fields)]` and derives
//! `schemars::JsonSchema` so invalid fields or types are rejected at the SDK
//! boundary before touching live editor state.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Canonical domain error codes returned when `CallToolResult.is_error == Some(true)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum McpErrorCode {
    RevisionConflict,
    IdempotencyConflict,
    InvalidArgument,
    InvalidHierarchy,
    TrackNotFound,
    ClipNotFound,
    ParamNotSupported,
    ParamDisabled,
    ParamBlockInvalid,
    AnalysisPending,
    PitchUnavailable,
    HistoryEmpty,
    EditorBusy,
    ViewBusy,
    ViewUnavailable,
    ViewUpdateFailed,
    SaveFailed,
    PreviewRenderFailed,
    WaitTimeout,
    SerializationFailed,
    EditorTaskFailed,
}

impl McpErrorCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RevisionConflict => "REVISION_CONFLICT",
            Self::IdempotencyConflict => "IDEMPOTENCY_CONFLICT",
            Self::InvalidArgument => "INVALID_ARGUMENT",
            Self::InvalidHierarchy => "INVALID_HIERARCHY",
            Self::TrackNotFound => "TRACK_NOT_FOUND",
            Self::ClipNotFound => "CLIP_NOT_FOUND",
            Self::ParamNotSupported => "PARAM_NOT_SUPPORTED",
            Self::ParamDisabled => "PARAM_DISABLED",
            Self::ParamBlockInvalid => "PARAM_BLOCK_INVALID",
            Self::AnalysisPending => "ANALYSIS_PENDING",
            Self::PitchUnavailable => "PITCH_UNAVAILABLE",
            Self::HistoryEmpty => "HISTORY_EMPTY",
            Self::EditorBusy => "EDITOR_BUSY",
            Self::ViewBusy => "VIEW_BUSY",
            Self::ViewUnavailable => "VIEW_UNAVAILABLE",
            Self::ViewUpdateFailed => "VIEW_UPDATE_FAILED",
            Self::SaveFailed => "SAVE_FAILED",
            Self::PreviewRenderFailed => "PREVIEW_RENDER_FAILED",
            Self::WaitTimeout => "WAIT_TIMEOUT",
            Self::SerializationFailed => "SERIALIZATION_FAILED",
            Self::EditorTaskFailed => "EDITOR_TASK_FAILED",
        }
    }
}

/// All positions are absolute project seconds; source positions are seconds from the file start.
/// Track IDs may refer to a `client_id` created earlier in the same atomic edit.
/// Provide either `expected_revision` (whole-project CAS) or `expected_scope`
/// (root-track scoped CAS for concurrent human-agent co-editing).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EditRequest {
    /// Opaque project-wide version returned by `hfs_get_context` / `hfs_query`.
    #[serde(default)]
    pub expected_revision: Option<String>,
    /// Optional fine-grained concurrency token allowing concurrent human edits on other root tracks.
    #[serde(default)]
    pub expected_scope: Option<EditScope>,
    /// Retry the identical request with the same key after a lost response.
    pub idempotency_key: String,
    #[serde(default)]
    pub label: Option<String>,
    pub operations: Vec<Operation>,
}

/// Scoped optimistic concurrency token for real-time human-agent co-editing.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum EditScope {
    /// Batch only creates brand-new root tracks (plus child tracks, clips, and curves under them).
    /// Succeeds even if a human concurrently edits existing tracks, provided `tempo_revision` matches.
    IsolatedNewTracks { tempo_revision: String },
    /// Batch only mutates a single root track's subtree (tracks, clips, takes, curves, param blocks).
    /// Succeeds even if a human concurrently edits other root tracks, provided `track_revision` and
    /// `tempo_revision` both match.
    RootTrack {
        root_track_id: String,
        track_revision: String,
        tempo_revision: String,
    },
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
    /// Import Top-K ranked vocal candidates into a single Clip's Takes in one step.
    /// Candidate 0 becomes the active take; remaining candidates become alternative takes
    /// that the human can cycle through in-place with `T` / `Shift+T`.
    ImportCandidateTakes {
        client_id: String,
        track_id: String,
        start_sec: f64,
        candidates: Vec<CandidateTakeSpec>,
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

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CandidateTakeSpec {
    pub name: Option<String>,
    pub path: String,
    pub source_start_sec: Option<f64>,
    pub length_sec: Option<f64>,
    pub playback_rate: Option<f32>,
    pub gain: Option<f32>,
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
    /// Use an algorithm ID returned by `hfs_get_edit_capabilities`. Omit to keep App settings.
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
    /// Smooth zero-slope cubic S-curve (`3t^2 - 2t^3`) transition for natural portamento/envelopes.
    HermiteTransition {
        from: f32,
        to: f32,
    },
    Shift {
        amount: f32,
    },
    /// Pitch only: shift the source contour's voiced median onto `target_midi`.
    ShiftToNote {
        target_midi: f32,
    },
    /// Pitch only: flatten reliable voiced samples; unvoiced samples remain unedited.
    FlattenToNote {
        target_midi: f32,
    },
    /// Add sinusoidal vibrato modulation with optional attack/release fade envelope onto existing contour.
    Vibrato {
        rate_hz: f32,
        depth: f32,
        phase_deg: Option<f32>,
        attack_sec: Option<f32>,
        release_sec: Option<f32>,
    },
    /// Symmetric moving-average de-jitter smoothing (`window_frames` in `1..=201`).
    Smooth {
        window_frames: u32,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TempoScaleSpec {
    pub key: Option<String>,
    pub name: Option<String>,
    /// Scale pitch classes (`0..=11`).
    pub notes: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TempoPoint {
    pub position_sec: f64,
    pub bpm: f64,
    pub numerator: Option<u32>,
    pub denominator: Option<u32>,
    /// Musical scale override at this tempo point. When omitted (and `clear_scale != Some(true)`),
    /// an existing scale override at the same `position_sec` is preserved.
    #[serde(default)]
    pub scale: Option<TempoScaleSpec>,
    /// Set to `true` to explicitly clear an existing scale override at this tempo point.
    #[serde(default)]
    pub clear_scale: Option<bool>,
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
    Notebook,
    Ledger,
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
    /// Both override fields are the complete desired project override; `null` means inherit App defaults.
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
    /// Optional time-range focus (`[focus_start_sec, focus_end_sec]`) to pan/zoom the GUI viewport onto a phrase.
    #[serde(default)]
    pub focus_start_sec: Option<f64>,
    #[serde(default)]
    pub focus_end_sec: Option<f64>,
    /// When `true` alongside `focus_start_sec`, also moves the transport playhead to `focus_start_sec`.
    #[serde(default)]
    pub seek_playhead: Option<bool>,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ViewTool {
    Select,
    Draw,
    Line,
    Vibrato,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WaitTarget {
    PitchAnalysis,
    ClipRender,
    GuiView,
}

/// Server-side readiness barrier replacing blind client sleep/polling loops.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WaitRequest {
    pub targets: Vec<WaitTarget>,
    #[serde(default)]
    pub track_ids: Option<Vec<String>>,
    #[serde(default)]
    pub clip_ids: Option<Vec<String>>,
    /// When `true` (default), proactively schedules pitch analysis on requested root tracks if missing.
    #[serde(default)]
    pub trigger_analysis: Option<bool>,
    /// Bounded wait timeout in milliseconds (`0..=30000`, default `5000`).
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

/// Bounded offline mixdown & deterministic acoustic QA self-check for Agents.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RenderPreviewRequest {
    pub start_sec: f64,
    pub end_sec: f64,
    /// Optional subset of tracks (root or child) to solo-isolate in an offscreen snapshot.
    #[serde(default)]
    pub track_ids: Option<Vec<String>>,
    /// Output sample rate (`8000..=96000`, default `44100`).
    #[serde(default)]
    pub sample_rate: Option<u32>,
    /// Whether to apply vocal synthesis / pitch curves (`true` by default; `false` renders dry arrangement).
    #[serde(default)]
    pub apply_pitch_edit: Option<bool>,
    /// Optional absolute `.wav` path to write the rendered preview for multimodal audio listening.
    #[serde(default)]
    pub output_wav_path: Option<String>,
    /// Must be `true` to replace an existing `.wav` file at `output_wav_path`.
    #[serde(default)]
    pub overwrite: Option<bool>,
}

/// Human-in-the-loop collaboration via the project's built-in Markdown Notebook.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NotebookRequest {
    pub action: NotebookAction,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NotebookAction {
    Read,
    /// Atomically insert or replace a fenced `<!-- hfs-agent:start:<section_id> -->` section
    /// inside the project notebook without clobbering human notes outside that section.
    UpsertAgentSection {
        #[serde(default)]
        expected_revision: Option<String>,
        section_id: String,
        title: String,
        markdown_body: String,
    },
    RemoveAgentSection {
        #[serde(default)]
        expected_revision: Option<String>,
        section_id: String,
    },
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
    Wait(WaitRequest),
    RenderPreview(RenderPreviewRequest),
    Notebook(NotebookRequest),
}
