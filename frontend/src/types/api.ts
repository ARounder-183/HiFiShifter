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
    /**
     * Backend the live vocoder session actually runs on: "CoreML", "WebGPU",
     * "DirectML", "CPU", or "" while no session has been built yet.
     * This reflects runtime reality, not the compiled-in default.
     */
    gpuBackend: string;
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

export interface TimelineClipTake {
    id: string;
    name: string;
    gain: number;
    source_path?: string;
    source_path_relative?: string;
    duration_sec?: number;
    duration_frames?: number;
    source_sample_rate?: number;
    source_start_sec: number;
    source_end_sec: number;
    playback_rate: number;
    reversed: boolean;
    loop_enabled: boolean;
    midi_note_data?: Array<{
        start_sec: number;
        end_sec: number;
        note: number;
        velocity: number;
        channel?: number;
    }>;
    midi_fill_gaps?: boolean;
}

export interface TimelineClip {
    id: string;
    group_id?: string;
    track_id: string;
    name: string;
    start_sec: number;
    length_sec: number;
    color: string;
    takes?: TimelineClipTake[];
    active_take_id?: string;
    source_path?: string;
    source_path_relative?: string;
    duration_sec?: number;
    duration_frames?: number; // 精确frame总数
    source_sample_rate?: number; // 源文件采样率
    waveform_preview?: number[] | { l: number[]; r: number[] } | { min: number[]; max: number[] };
    pitch_range?: {
        min: number;
        max: number;
    };
    gain?: number;
    muted?: boolean;
    source_start_sec?: number;
    source_end_sec?: number;
    playback_rate?: number;
    /** Clip 级播放倍率；实际速率 = clip_playback_rate × active take playback_rate。 */
    clip_playback_rate?: number;
    reversed?: boolean;
    /** Loop（循环源）：超出源媒体区间时按周期回绕产生循环内容。 */
    loop_enabled?: boolean;
    /** 吸附偏移（秒）：相对 Clip 起点的偏移，默认 0；旧工程缺失时补齐为 0。 */
    snap_offset_sec?: number;
    fade_in_sec?: number;
    fade_out_sec?: number;
    /** REAPER 浮点形状 id（整数 0..6 七预设；小数变体透传保存）。 */
    fade_in_shape?: number;
    fade_out_shape?: number;
    /** 曲率（REAPER D_FADEINDIR/OUTDIR），范围 [-1, 1]。 */
    fade_in_dir?: number;
    fade_out_dir?: number;
    /** 自动交叉淡化长度（秒），与手动 fade 分离存储。 */
    auto_fade_in_sec?: number;
    auto_fade_out_sec?: number;
    formant_morph?: {
        enabled: boolean;
        target_f1_hz: number;
        target_f2_hz: number;
        strength: number;
    };
    midi_note_count?: number;
    midi_note_data?: Array<{
        start_sec: number;
        end_sec: number;
        note: number;
        velocity: number;
        channel?: number;
    }>;
    midi_fill_gaps?: boolean;
}

export interface ProjectMeta {
    name: string;
    path?: string | null;
    dirty: boolean;
    recent: string[];
    notes_markdown?: string;
    base_scale?: string;
    use_custom_scale?: boolean;
    custom_scale?: {
        id: string;
        name: string;
        notes: number[];
    } | null;
    beats_per_bar?: number;
    /** 工程基准拍号分母（1/2/4/8/16/32）。 */
    time_signature_denominator?: number;
    grid_size?: string;
    stretch_algorithm_override?: "linear" | "signalsmith" | "soundtouch" | null;
    hifigan_mel_stretch_override?: boolean | null;
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
    disabled_group_ids?: string[];
    /** Tempo Map 数据（null = 无 Tempo Map）。 */
    tempo_map?: TempoMapPayload;
}

/** Tempo Map 变化点（后端 camelCase 载荷，与 `TempoPointPayload` 对应）。 */
export interface TempoPointPayload {
    id: string;
    positionSec: number;
    bpm: number;
    /** 拍号分子；null 表示“跟随之前的拍号”（0 位置初始点必须显式）。 */
    numerator: number | null;
    /** 拍号分母；null 表示“跟随之前的拍号”。 */
    denominator: number | null;
    scale: {
        key?: string | null;
        name?: string | null;
        notes?: number[] | null;
    } | null;
}

/**
 * 后端 Tempo Map 载荷：变化点的“裸数组”（null = 无 Tempo Map）。
 *
 * 与后端 `TimelineStatePayload.tempo_map: Option<Vec<TempoPointPayload>>` 及
 * `set_timeline_tempo_map` 命令参数一一对应 —— 注意是数组本身，
 * 不是 `{ points: [...] }` 包装对象。
 */
export type TempoMapPayload = TempoPointPayload[] | null;

export interface TimelineResult {
    ok: true;
    tracks: TimelineTrack[];
    clips: TimelineClip[];
    created_clip_ids?: string[];
    created_track_ids?: string[];
    selected_track_id: string | null;
    selected_clip_id: string | null;
    bpm: number;
    playhead_sec: number;
    project_sec?: number;
    project?: ProjectMeta;
    missing_files?: string[];
    skipped_files?: string[];
    disabled_group_ids?: string[];
    tempo_map?: TempoMapPayload;
    /** `open_project` 专用：工程文件版本高于当前程序，等待用户确认。 */
    project_version_too_new?: boolean;
    project_file_version?: number;
    current_project_file_version?: number;
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

/** HFSPeaks v2 mipmap 级别（L0=div16, L1=div512, L2=div4096；默认切换阈值 512/1024 spp） */
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
    /**
     * 二进制编码的曲线数据（Base64）。`paramsApi.getParamFrames` 默认请求二进制
     * 并在 API 层解码成 `orig`/`edit`，正常情况下调用方不会见到该字段有值。
     * 协议见 `pianoRoll/paramFramesBinaryCodec.ts`。
     */
    binary?: string | null;
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
    active_ep?: string;
    onnx_version?: string;
    providers?: string[];
    gpuDiagnostic?: GpuDiagnostic | null;
}

export interface GpuDiagnostic {
    availableProviders: string[];
    selectedEp: string;
    gpuDeviceId: number;
    gpuSmokeTestPassed: boolean;
    gpuSmokeTestError?: string | null;
    ortBuildInfo: string;
    gpuDllStatus: [string, boolean][];
}

/** Info about a single GPU device. */
export interface GpuDeviceInfo {
    deviceId: number;
    name: string;
    memoryMb: number;
    computeMajor: number;
    computeMinor: number;
}

export interface GpuEnumerationResult {
    devices: GpuDeviceInfo[];
    note?: string | null;
}

/** DirectML-compatible GPU adapter info from DXGI enumeration. */
export interface DmlAdapterInfo {
    deviceId: number;
    name: string;
    dedicatedVideoMemoryMb: number;
    sharedSystemMemoryMb: number;
    vendorId: number;
    deviceIdPci: number;
}

export interface DmlAdapterList {
    adapters: DmlAdapterInfo[];
    note?: string | null;
}

export interface BenchmarkResult {
    cpuMedianMs: number;
    cpuRtFactor: number;
    /** WebGPU inference time (ms), null if unavailable or failed. */
    gpuMedianMs?: number | null;
    /** WebGPU real-time factor, null if unavailable or failed. */
    gpuRtFactor?: number | null;
    /** Display name of the GPU backend ("CoreML", "WebGPU", ...). */
    gpuBackendName?: string | null;
    /** Detailed error message when the GPU benchmark could not complete. */
    gpuError?: string | null;
    /** DirectML GPU inference time (ms), null if unavailable or failed. */
    dmlMedianMs?: number | null;
    /** DirectML GPU real-time factor, null if unavailable or failed. */
    dmlRtFactor?: number | null;
    benchmarkSamples: number;
    /** True when WebGPU EP was available for the GPU benchmark. */
    gpuAvailable: boolean;
    /** True when DirectML EP was available for the benchmark. */
    dmlAvailable: boolean;
    /** GPU device ID that was used. */
    gpuDeviceId: number;
    /** All execution providers available in the ONNX Runtime DLL. */
    availableProviders: string[];
    /** ORT build info string for debugging. */
    ortBuildInfo: string;
    /** All GPUs discovered (name, memory, device ID). */
    gpuDevices: GpuDeviceInfo[];
}

export interface BenchmarkResult_simple {
    cpuMedianMs: number;
    cpuRtFactor: number;
    gpuMedianMs?: number | null;
    gpuRtFactor?: number | null;
    dmlMedianMs?: number | null;
    dmlRtFactor?: number | null;
    benchmarkSamples: number;
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
