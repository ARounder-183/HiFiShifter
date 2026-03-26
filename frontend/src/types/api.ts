export type ApiResult<T> =
    | ({ ok: true } & T)
    | {
          ok: false;
          error: { code?: string; message: string; traceback?: string };
      };

export interface RuntimeInfo {
    ok: true;
    device: string;
    model_loaded: boolean;
    audio_loaded: boolean;
    has_synthesized: boolean;
    is_playing?: boolean;
    playback_target?: string | null;
    timeline?: TimelineState;
}

export interface TimelineTrack {
    id: string;
    name: string;
    parent_id?: string | null;
    depth?: number;
    child_track_ids?: string[];
    muted: boolean;
    solo: boolean;
    volume: number;

    compose_enabled: boolean;
    pitch_analysis_algo: string;
    color: string;
}

export interface TimelineClip {
    id: string;
    track_id: string;
    name: string;
    start_sec: number;
    length_sec: number;
    color: string;
    source_path?: string;
    duration_sec?: number;
    duration_frames?: number; // 精确frame总数
    source_sample_rate?: number; // 源文件采样率
    waveform_preview?:
        | number[]
        | { l: number[]; r: number[] }
        | { min: number[]; max: number[] };
    pitch_range?: {
        min: number;
        max: number;
    };
    gain?: number;
    muted?: boolean;
    source_start_sec?: number;
    source_end_sec?: number;
    playback_rate?: number;
    fade_in_sec?: number;
    fade_out_sec?: number;
    fade_in_curve?: string;
    fade_out_curve?: string;
}

export interface ProjectMeta {
    name: string;
    path?: string | null;
    dirty: boolean;
    recent: string[];
    base_scale?: string;
    use_custom_scale?: boolean;
    custom_scale?: {
        id: string;
        name: string;
        notes: number[];
    } | null;
    beats_per_bar?: number;
    grid_size?: string;
}

export interface TimelineState {
    tracks: TimelineTrack[];
    clips: TimelineClip[];
    selected_track_id: string | null;
    selected_clip_id: string | null;
    bpm: number;
    playhead_sec: number;
    project_sec?: number;
    project?: ProjectMeta;
    missing_files?: string[];
    skipped_files?: string[];
}

export interface TimelineResult {
    ok: true;
    tracks: TimelineTrack[];
    clips: TimelineClip[];
    selected_track_id: string | null;
    selected_clip_id: string | null;
    bpm: number;
    playhead_sec: number;
    project_sec?: number;
    project?: ProjectMeta;
    missing_files?: string[];
    skipped_files?: string[];
}

export interface TrackSummaryResult {
    ok: true;
    track_id: string;
    clip_count: number;
    waveform_preview: number[];
    pitch_range: {
        min: number;
        max: number;
    };
}

export interface ModelConfigResult {
    ok: true;
    config: {
        audio_sample_rate: number;
        audio_num_mel_bins: number;
        hop_size: number;
        fmin: number;
        fmax: number;
    };
}

export interface ProcessAudioResult {
    ok: true;
    audio: {
        path: string;
        sample_rate: number;
        duration_sec: number;
    };
    feature: {
        mel_shape: number[];
        f0_frames: number;
        segment_count: number;
        segments_preview: number[][];
        waveform_preview: number[];
        pitch_range: {
            min: number;
            max: number;
        };
    };
    timeline?: TimelineState;
}

export interface SynthesizeResult {
    ok: true;
    sample_rate: number;
    num_samples: number;
    duration_sec: number;
}

export interface PlaybackStateResult {
    ok: true;
    is_playing: boolean;
    target: string | null;
    base_sec: number;
    position_sec: number;
    duration_sec: number;
}

export interface WaveformPeaksSegmentPayload {
    ok: boolean;
    min: number[];
    max: number[];
}

/** HFSPeaks v2 mipmap 级别（L0=div64, L1=div512, L2=div4096） */
export type MipmapLevel = 0 | 1 | 2;

/** v2 波形峰值响应 */
export interface WaveformPeaksV2Payload {
    ok: boolean;
    min: number[];
    max: number[];
    sample_rate: number;
    mipmap_level: number;
    division_factor: number;
    /** 返回数据实际覆盖的起始时间（秒），由后端 floor/ceil 取整后的峰值索引决定 */
    actual_start_sec: number;
    /** 返回数据实际覆盖的持续时间（秒），由后端 floor/ceil 取整后的峰值索引决定 */
    actual_duration_sec: number;
}

/** v2 波形元数据响应 */
export interface WaveformPeaksV2MetaPayload {
    ok: boolean;
    sample_rate: number;
    channels: number;
    total_frames: number;
    mipmap_levels: Array<{
        level: number;
        division_factor: number;
        peak_count: number;
    }>;
    cached: boolean;
}

export type ParamReferenceKind = "source_curve" | "default_value";

export interface ParamFramesPayload {
    ok: boolean;
    root_track_id: string;
    param: string;
    frame_period_ms: number;
    start_frame: number;
    orig: number[];
    edit: number[];
    reference_kind: ParamReferenceKind;

    analysis_pending?: boolean;
    analysis_progress?: number;

    pitch_edit_user_modified?: boolean;
    pitch_edit_backend_available?: boolean;
}

export interface PitchProgressPayload {
    rootTrackId: string;
    progress: number;
    etaSeconds?: number;
    /** 当前正在分析�?clip 名称 */
    currentClipName?: string | null;
    /** 已完成的 clip 数量 */
    completedClips?: number;
    /** 需要分析的 clip 总数 */
    totalClips?: number;
}

export interface OnnxStatusResult {
    compiled: boolean;
    available: boolean;
    error: string | null;
    ep_choice: string;
}

export interface OnnxDiagnosticResult {
    compiled: boolean;
    available: boolean;
    error: string | null;
    ep_choice: string;
    onnx_version?: string;
    providers?: string[];
}

export interface PitchTaskStatusPayload {
    status: "running" | "completed" | "failed" | "cancelled";
    progress: number;
    error?: string | null;
    result_key?: string | null;
}

// ─── Processor param descriptors ────────────────────────────────────────────

export type ParamKindDto =
    | {
          type: "automation_curve";
          unit: string;
          default_value: number;
          min_value: number;
          max_value: number;
      }
    | {
          type: "static_enum";
          options: [string, number][];
          default_value: number;
      };

export interface ProcessorParamDescriptor {
    id: string;
    display_name: string;
    group: string;
    kind: ParamKindDto;
}

export interface StaticParamValuePayload {
    ok: boolean;
    root_track_id: string;
    param: string;
    value: number;
}
