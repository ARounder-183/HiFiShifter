import type {
    ModelConfigResult,
    OnnxDiagnosticResult,
    OnnxStatusResult,
    PitchProgressPayload,
    PitchTaskStatusPayload,
    PlaybackStateResult,
    ProcessAudioResult,
    RuntimeInfo,
    SynthesizeResult,
} from "../../types/api";

import { invoke } from "../invoke";

export interface AdvancedSeparatedTarget {
    kind: "root" | "sub";
    trackId: string;
}

export interface AdvancedExportRequest {
    mode: "project" | "separated";
    range: {
        kind: "all" | "custom";
        startSec?: number;
        endSec?: number;
    };
    projectOutputDir?: string;
    projectFileName?: string;
    projectOutputPath?: string;
    separatedOutputDir?: string;
    separatedNamePattern?: string;
    separatedTargets?: AdvancedSeparatedTarget[];
    overwriteExistingPaths?: string[];
    skipExistingPaths?: string[];
    sampleRate?: number;
    bitDepth?: 16 | 24 | 32;
}

export interface ExportAudioPlanItem {
    trackId?: string | null;
    path: string;
}

export interface ExportAudioPlan {
    ok: boolean;
    mode: "project" | "separated";
    targets: ExportAudioPlanItem[];
    existingPaths: string[];
}

export interface ExportAudioDefaults {
    ok: boolean;
    projectName: string;
    documentsDir: string;
    projectOutputDir: string;
    projectFileName: string;
    separatedOutputDir: string;
    separatedFileName: string;
    sampleRate: number;
    bitDepth: 16 | 24 | 32;
}

export const coreApi = {
    ping: () => invoke<{ ok: boolean; message: string }>("ping"),
    getRuntimeInfo: () => invoke<RuntimeInfo>("get_runtime_info"),
    getPlaybackState: () => invoke<PlaybackStateResult>("get_playback_state"),

    setUiLocale: (locale: string) =>
        invoke<{ ok: boolean; locale?: string }>("set_ui_locale", locale),

    openAudioDialog: () =>
        invoke<{ ok: boolean; canceled?: boolean; path?: string }>("open_audio_dialog"),

    openAudioDialogMultiple: () =>
        invoke<{ ok: boolean; canceled?: boolean; paths?: string[] }>("open_audio_dialog_multi"),

    pickOutputPath: () =>
        invoke<{ ok: boolean; canceled?: boolean; path?: string }>("pick_output_path"),

    closeWindow: () => invoke<{ ok: boolean }>("close_window"),

    openMidiDialog: () =>
        invoke<{ ok: boolean; canceled?: boolean; path?: string }>("open_midi_dialog"),

    clearWaveformCache: () =>
        invoke<{
            ok: boolean;
            removed_files: number;
            removed_bytes: number;
            dir: string;
        }>("clear_waveform_cache"),

    // Model / processing
    loadDefaultModel: () => invoke<ModelConfigResult>("load_default_model"),
    loadModel: (modelDir: string) => invoke<ModelConfigResult>("load_model", modelDir),
    processAudio: (audioPath: string) => invoke<ProcessAudioResult>("process_audio", audioPath),

    setPitchShift: (semitones: number) =>
        invoke<{ ok: boolean; pitch_shift?: number; frames?: number }>(
            "set_pitch_shift",
            semitones,
        ),

    synthesize: () => invoke<SynthesizeResult>("synthesize"),

    saveSynthesized: (outputPath: string) =>
        invoke<{
            ok: boolean;
            path?: string;
            sample_rate?: number;
            num_samples?: number;
        }>("save_synthesized", outputPath),

    saveSeparated: (outputDir: string) =>
        invoke<{
            ok: boolean;
            count?: number;
            output_dir?: string;
            tracks?: Array<{
                track_id: string;
                name: string;
                path?: string;
                ok: boolean;
                error?: string;
            }>;
        }>("save_separated", outputDir),

    exportAudioAdvanced: (request: AdvancedExportRequest) =>
        invoke<{
            ok: boolean;
            mode?: "project" | "separated";
            path?: string;
            output_dir?: string;
            count?: number;
            cancelled?: boolean;
            error?: string;
        }>("export_audio_advanced", request),

    cancelExportAudio: () => invoke<{ ok: boolean; active?: boolean }>("cancel_export_audio"),

    getExportAudioDefaults: () => invoke<ExportAudioDefaults>("get_export_audio_defaults"),

    previewExportAudioPlan: (request: AdvancedExportRequest) =>
        invoke<ExportAudioPlan>("preview_export_audio_plan", request),

    playOriginal: (startSec = 0) =>
        invoke<{ ok: boolean; playing?: string; start_sec?: number }>("play_original", startSec),

    stopAudio: () => invoke<{ ok: boolean }>("stop_audio"),

    // Pitch analysis progress
    getPitchAnalysisProgress: () =>
        invoke<PitchProgressPayload | null>("get_pitch_analysis_progress"),

    // ONNX status and diagnostics
    getOnnxStatus: () => invoke<OnnxStatusResult>("get_onnx_status"),
    getOnnxDiagnostic: () => invoke<OnnxDiagnosticResult>("get_onnx_diagnostic"),

    // Async pitch refresh task system
    startPitchRefreshTask: (rootTrackId: string) =>
        invoke<string>("start_pitch_refresh_task", rootTrackId),
    getPitchRefreshStatus: (taskId: string) =>
        invoke<PitchTaskStatusPayload | null>("get_pitch_refresh_status", taskId),
    cancelPitchTask: (taskId: string) => invoke<{ ok: boolean }>("cancel_pitch_task", taskId),
};
