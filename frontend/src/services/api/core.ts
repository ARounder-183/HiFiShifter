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

export const coreApi = {
    ping: () => invoke<{ ok: boolean; message: string }>("ping"),
    getRuntimeInfo: () => invoke<RuntimeInfo>("get_runtime_info"),
    getPlaybackState: () => invoke<PlaybackStateResult>("get_playback_state"),

    setUiLocale: (locale: string) =>
        invoke<{ ok: boolean; locale?: string }>("set_ui_locale", locale),

    openAudioDialog: () =>
        invoke<{ ok: boolean; canceled?: boolean; path?: string }>(
            "open_audio_dialog",
        ),

    openAudioDialogMultiple: () =>
        invoke<{ ok: boolean; canceled?: boolean; paths?: string[] }>(
            "open_audio_dialog_multi",
        ),

    pickOutputPath: () =>
        invoke<{ ok: boolean; canceled?: boolean; path?: string }>(
            "pick_output_path",
        ),

    closeWindow: () => invoke<{ ok: boolean }>("close_window"),

    openMidiDialog: () =>
        invoke<{ ok: boolean; canceled?: boolean; path?: string }>(
            "open_midi_dialog",
        ),

    clearWaveformCache: () =>
        invoke<{
            ok: boolean;
            removed_files: number;
            removed_bytes: number;
            dir: string;
        }>("clear_waveform_cache"),

    // Model / processing
    loadDefaultModel: () => invoke<ModelConfigResult>("load_default_model"),
    loadModel: (modelDir: string) =>
        invoke<ModelConfigResult>("load_model", modelDir),
    processAudio: (audioPath: string) =>
        invoke<ProcessAudioResult>("process_audio", audioPath),

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

    playOriginal: (startSec = 0) =>
        invoke<{ ok: boolean; playing?: string; start_sec?: number }>(
            "play_original",
            startSec,
        ),

    stopAudio: () => invoke<{ ok: boolean }>("stop_audio"),

    // Pitch analysis progress
    getPitchAnalysisProgress: () =>
        invoke<PitchProgressPayload | null>("get_pitch_analysis_progress"),

    // ONNX status and diagnostics
    getOnnxStatus: () => invoke<OnnxStatusResult>("get_onnx_status"),
    getOnnxDiagnostic: () =>
        invoke<OnnxDiagnosticResult>("get_onnx_diagnostic"),

    // Async pitch refresh task system
    startPitchRefreshTask: (rootTrackId: string) =>
        invoke<string>("start_pitch_refresh_task", rootTrackId),
    getPitchRefreshStatus: (taskId: string) =>
        invoke<PitchTaskStatusPayload | null>(
            "get_pitch_refresh_status",
            taskId,
        ),
    cancelPitchTask: (taskId: string) =>
        invoke<{ ok: boolean }>("cancel_pitch_task", taskId),
};
