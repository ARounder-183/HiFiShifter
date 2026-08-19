// 统一封装 Tauri / pywebview 调用
// - Tauri: window.__TAURI__.core.invoke / window.__TAURI__.invoke (named args)
// - pywebview: window.pywebview.api[method] (positional args)

/* eslint-disable @typescript-eslint/no-explicit-any */

declare global {
    interface Window {
        pywebview?: {
            api?: Record<string, (...args: any[]) => Promise<any>>;
        };
        __TAURI__?: {
            core?: {
                invoke?: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
            };
            invoke?: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
        };
    }
}

type PyWebviewApi = Record<string, (...args: any[]) => Promise<any>>;

type InvokeMode = "tauri" | "pywebview";

export class BackendInvokeError extends Error {
    public readonly mode: InvokeMode;
    public readonly method: string;
    public readonly args?: unknown;

    constructor(params: { mode: InvokeMode; method: string; args?: unknown; cause?: unknown }) {
        super(`Backend invoke failed: ${params.mode}:${params.method}`, {
            cause: params.cause,
        });
        this.name = "BackendInvokeError";
        this.mode = params.mode;
        this.method = params.method;
        this.args = params.args;
    }
}

let pywebviewAvailability: "unknown" | "available" | "unavailable" = "unknown";

async function waitForPyWebviewApi(timeoutMs: number): Promise<PyWebviewApi | null> {
    const already = window.pywebview?.api;
    if (already) {
        pywebviewAvailability = "available";
        return already as PyWebviewApi;
    }

    if (pywebviewAvailability === "unavailable") {
        return null;
    }

    const startedAt = performance.now();
    await new Promise<void>((resolve) => {
        let done = false;

        function finish() {
            if (done) return;
            done = true;
            window.removeEventListener("pywebviewready", onReady as any);
            document.removeEventListener("pywebviewready", onReady as any);
            clearInterval(pollId);
            clearTimeout(timeoutId);
            resolve();
        }

        function onReady() {
            finish();
        }

        const pollId = window.setInterval(() => {
            if (window.pywebview?.api) finish();
        }, 25);

        const timeoutId = window.setTimeout(
            () => {
                finish();
            },
            Math.max(0, timeoutMs),
        );

        window.addEventListener("pywebviewready", onReady as any, {
            once: true,
        });
        document.addEventListener("pywebviewready", onReady as any, {
            once: true,
        });
    });

    const api = window.pywebview?.api as PyWebviewApi | undefined;
    if (api) {
        pywebviewAvailability = "available";
        return api;
    }

    // If pywebview didn't appear after a meaningful wait, treat it as unavailable
    // to avoid stalling every call in browser/dev mode.
    if (performance.now() - startedAt >= 750) {
        pywebviewAvailability = "unavailable";
    }

    return null;
}

function getTauriInvoke(): (<T>(cmd: string, args?: Record<string, unknown>) => Promise<T>) | null {
    const tauriInvoke = window.__TAURI__?.core?.invoke ?? window.__TAURI__?.invoke;
    if (typeof tauriInvoke !== "function") return null;
    return tauriInvoke;
}

type BuildArgsResult = Record<string, unknown> | undefined | { __unwired: true };

function buildTauriArgs(method: string, args: unknown[]): BuildArgsResult {
    // 注意：Tauri invoke uses a named-argument object; pywebview uses positional args.
    switch (method) {
        case "set_transport": {
            const o: Record<string, unknown> = {};
            if (args[0] !== undefined) o.playheadSec = args[0];
            if (args[1] !== undefined) o.bpm = args[1];
            return o;
        }

        case "set_ui_locale":
            return { locale: args[0] };

        case "import_audio_item":
            return {
                audioPath: args[0],
                ...(args[1] !== undefined ? { trackId: args[1] } : {}),
                ...(args[2] !== undefined ? { startSec: args[2] } : {}),
                ...(args[3] !== undefined ? { mediaAudioStreamIndex: args[3] } : {}),
            };

        case "import_audio_bytes":
            return {
                fileName: args[0],
                base64Data: args[1],
                ...(args[2] !== undefined ? { trackId: args[2] } : {}),
                ...(args[3] !== undefined ? { startSec: args[3] } : {}),
            };

        case "add_track":
            return {
                name: args[0],
                parentTrackId: (args[1] ?? null) as unknown,
                index: args[2],
            };

        case "remove_track":
            return { trackId: args[0] };

        case "duplicate_track":
            return { trackId: args[0] };

        case "move_track":
            return {
                trackId: args[0],
                targetIndex: args[1],
                parentTrackId: (args[2] ?? null) as unknown,
            };

        case "set_track_state":
            return {
                trackId: args[0],
                muted: args[1],
                solo: args[2],
                volume: args[3],
                composeEnabled: args[4],
                pitchAnalysisAlgo: args[5],
                color: args[6],
                name: args[7],
            };

        case "select_track":
            return { trackId: args[0] };

        case "set_project_length":
            return { projectSec: args[0] };

        case "get_track_summary":
            return args[0] === undefined ? undefined : { trackId: args[0] };

        case "add_clip":
            return {
                trackId: (args[0] ?? null) as unknown,
                name: args[1],
                startSec: args[2],
                lengthSec: args[3],
                sourcePath: args[4],
            };

        case "create_clips_bulk":
            return { payload: args[0] };

        case "remove_clip":
            return { clipId: args[0] };

        case "remove_clips":
            return { clipIds: args[0] };

        case "move_clip":
            return {
                clipId: args[0],
                startSec: args[1],
                trackId: (args[2] ?? null) as unknown,
                moveLinkedParams: args[3],
            };

        case "move_clips":
            return {
                moves: args[0],
                moveLinkedParams: args[1],
            };

        case "get_clip_linked_params":
            return { clipId: args[0] };

        case "apply_clip_linked_params":
            return {
                clipId: args[0],
                linkedParams: args[1],
            };

        case "set_clip_state":
            return {
                clipId: args[0],
                name: args[1],
                startSec: args[2],
                lengthSec: args[3],
                gain: args[4],
                muted: args[5],
                sourceStartSec: args[6],
                sourceEndSec: args[7],
                playbackRate: args[8],
                reversed: args[9],
                fadeInSec: args[10],
                fadeOutSec: args[11],
                fadeInCurve: args[12],
                fadeOutCurve: args[13],
                color: args[14],
                formantMorph: args[15],
                checkpoint: args[16],
            };

        case "set_clips_state_bulk":
            return {
                updates: args[0],
                checkpoint: args[1],
            };

        case "duplicate_clips_bulk":
            return { payload: args[0] };

        case "replace_clip_source":
            return {
                clipIds: args[0],
                newSourcePath: args[1],
                replaceSameSource: args[2],
            };

        case "search_source_file_replacements":
            return {
                folderPath: args[0],
                clipIds: args[1],
                searchMode: args[2],
            };

        case "split_clip":
            return { clipId: args[0], splitSec: args[1] };

        case "split_clips_at":
            return { clipIds: args[0], splitSec: args[1] };

        case "glue_clips":
            return { clipIds: args[0] };

        case "group_clips":
            return { clipIds: args[0] };

        case "ungroup_clips":
            return { clipIds: args[0] };

        case "toggle_group_disabled":
            return { groupId: args[0] };

        case "convert_clips_to_pitch_reference":
            return { clipIds: args[0] };

        case "update_pitch_reference":
            return { clipIds: args[0] };

        case "select_clip":
            return { clipId: args[0] };

        case "load_model":
            return { modelDir: args[0] };

        case "process_audio":
            return { audioPath: args[0] };

        case "set_pitch_shift":
            return { semitones: args[0] };

        case "save_synthesized":
            return { outputPath: args[0] };

        case "save_separated":
            return { outputDir: args[0] };

        case "export_audio_advanced":
            return { request: args[0] };

        case "preview_export_audio_plan":
            return { request: args[0] };

        case "quick_export_selected_clips":
            return { request: args[0] };

        case "play_original":
            return { startSec: args[0] };

        case "open_project":
            return {
                projectPath: args[0],
                ...(args[1] !== undefined ? { force: args[1] } : {}),
            };

        case "run_timed_auto_backup":
            return { pathTemplate: args[0] };

        case "set_project_base_scale":
            return { baseScale: args[0] };

        case "set_project_custom_scale":
            return { customScale: args[0] };

        case "set_project_timeline_settings":
            return {
                beatsPerBar: args[0],
                timeSignatureDenominator: args[1],
                gridSize: args[2],
            };

        case "set_project_stretch_settings":
            return {
                stretchAlgorithmOverride: args[0],
                hifiganMelStretchOverride: args[1],
            };

        case "save_project":
            return args[0] === undefined ? undefined : { notesMarkdown: args[0] };

        case "import_project":
            return {
                projectPath: args[0],
                ...(args[1] !== undefined ? { placeAtPlayhead: args[1] } : {}),
                ...(args[2] !== undefined ? { importTempoMap: args[2] } : {}),
            };

        case "copy_timeline_clips":
            return { clipIds: args[0] };

        case "copy_timeline_tracks":
            return { trackIds: args[0] };

        case "paste_timeline_clipboard":
            return args[0] === undefined ? undefined : { mode: args[0] };

        case "has_timeline_clipboard":
        case "has_reaper_clipboard":
        case "read_system_clipboard_object":
            return {};

        case "write_system_clipboard_object":
            return {
                payload: args[0],
                ...(args[1] !== undefined ? { textSummary: args[1] } : {}),
            };

        case "save_project_as":
            return args[0] === undefined ? undefined : { notesMarkdown: args[0] };

        case "import_vocalshifter_project":
            return { vspPath: args[0] };

        case "import_reaper_project":
            return { rppPath: args[0] };

        case "paste_vocalshifter_clipboard":
            return {
                ...(args[0] !== undefined ? { selectionStartFrame: args[0] } : {}),
                ...(args[1] !== undefined ? { selectionMaxFrames: args[1] } : {}),
                ...(args[2] !== undefined ? { activeParam: args[2] } : {}),
            };

        case "paste_reaper_clipboard":
            return {
                ...(args[0] !== undefined ? { selectionStartFrame: args[0] } : {}),
                ...(args[1] !== undefined ? { selectionMaxFrames: args[1] } : {}),
            };

        case "open_audio_dialog_for_source":
            return { sourcePath: args[0], dialogTitle: args[1] };

        case "open_midi_dialog":
            return {};

        case "pick_midi_output_path":
            return {};

        case "export_pitch_to_midi":
            return { request: args[0] };

        case "get_waveform_mipmap_binary":
            return { sourcePath: args[0], level: args[1] };

        case "preload_waveform_mipmap":
            return { sourcePath: args[0] };

        case "batch_get_waveform_mipmap":
            return { sourcePaths: args[0] };

        case "get_root_mix_waveform_peaks_segment":
        case "get_track_mix_waveform_peaks_segment":
            return {
                trackId: args[0],
                startSec: args[1],
                durationSec: args[2],
                columns: args[3],
            };

        case "get_param_frames":
            return {
                trackId: args[0],
                param: args[1],
                startFrame: args[2],
                frameCount: args[3],
                stride: args[4],
            };

        case "set_param_frames":
            return {
                trackId: args[0],
                param: args[1],
                startFrame: args[2],
                values: args[3],
                checkpoint: args[4],
            };

        case "restore_param_frames":
            return {
                trackId: args[0],
                param: args[1],
                startFrame: args[2],
                frameCount: args[3],
                checkpoint: args[4],
            };

        case "get_static_param":
            return {
                trackId: args[0],
                param: args[1],
            };

        case "set_static_param":
            return {
                trackId: args[0],
                param: args[1],
                value: args[2],
                checkpoint: args[3],
            };

        case "list_directory":
            return { dirPath: args[0] };

        case "get_audio_file_info":
            return { filePath: args[0] };

        case "get_media_audio_streams":
            return { filePath: args[0] };

        case "read_audio_preview":
            return {
                filePath: args[0],
                ...(args[1] !== undefined ? { maxFrames: args[1] } : {}),
            };

        case "search_files_recursive":
            return { dirPath: args[0], query: args[1] };

        case "get_processor_params":
            return { algo: args[0] };

        case "get_midi_tracks":
            return {
                midiPath: args[0],
                ...(args[1] != null ? { clipboardGuid: args[1] } : {}),
            };

        case "import_midi_to_pitch":
            return {
                midiPath: args[0],
                ...(args[1] !== undefined ? { trackIndices: args[1] } : {}),
                ...(args[2] !== undefined ? { selectionStartFrame: args[2] } : {}),
                ...(args[3] !== undefined ? { selectionMaxFrames: args[3] } : {}),
                ...(args[4] !== undefined ? { fillGaps: args[4] } : {}),
                ...(args[5] !== undefined ? { noteBpmMode: args[5] } : {}),
                ...(args[6] !== undefined ? { specifiedBpm: args[6] } : {}),
                ...(args[7] !== undefined ? { importMidiBpmAsProject: args[7] } : {}),
                ...(args[8] != null ? { clipboardGuid: args[8] } : {}),
                ...(args[9] !== undefined ? { closeLeadingGap: args[9] } : {}),
            };

        case "import_midi_as_clip":
            return {
                midiPath: args[0],
                ...(args[1] !== undefined ? { trackIndices: args[1] } : {}),
                ...(args[2] !== undefined ? { trackId: args[2] } : {}),
                ...(args[3] !== undefined ? { startSec: args[3] } : {}),
                ...(args[4] !== undefined ? { fillGaps: args[4] } : {}),
                ...(args[5] !== undefined ? { multiTrackMerge: args[5] } : {}),
                ...(args[6] !== undefined ? { noteBpmMode: args[6] } : {}),
                ...(args[7] !== undefined ? { specifiedBpm: args[7] } : {}),
                ...(args[8] !== undefined ? { importMidiBpmAsProject: args[8] } : {}),
                ...(args[9] != null ? { clipboardGuid: args[9] } : {}),
                ...(args[10] !== undefined ? { closeLeadingGap: args[10] } : {}),
                ...(args[11] !== undefined ? { importMidiAsTempoMap: args[11] } : {}),
                ...(args[12] !== undefined ? { importMidiTempo: args[12] } : {}),
                ...(args[13] !== undefined ? { importMidiTimeSignature: args[13] } : {}),
                ...(args[14] !== undefined ? { importMidiKeySignature: args[14] } : {}),
            };

        case "set_timeline_tempo_map":
            return { tempoMap: args[0] };

        case "replace_midi_clip_data":
            return {
                clipId: args[0],
                midiPath: args[1],
                ...(args[2] !== undefined ? { trackIndices: args[2] } : {}),
                ...(args[3] !== undefined ? { fillGaps: args[3] } : {}),
                ...(args[4] !== undefined ? { noteBpmMode: args[4] } : {}),
                ...(args[5] !== undefined ? { specifiedBpm: args[5] } : {}),
                ...(args[6] !== undefined ? { importMidiBpmAsProject: args[6] } : {}),
                ...(args[7] != null ? { clipboardGuid: args[7] } : {}),
                ...(args[8] !== undefined ? { closeLeadingGap: args[8] } : {}),
            };

        case "save_ui_settings":
            return args[0] as Record<string, unknown>;

        case "save_auto_backup_settings":
            return { settings: args[0] };

        case "begin_undo_group":
            return undefined;

        case "end_undo_group":
            return undefined;

        default:
            return { __unwired: true };
    }
}

export async function invoke<T>(method: string, ...args: unknown[]): Promise<T> {
    const tauriInvoke = getTauriInvoke();
    if (tauriInvoke) {
        const invokeArgs = buildTauriArgs(method, args);
        if (invokeArgs && "__unwired" in invokeArgs) {
            if (args.length > 0) {
                throw new Error(
                    `Tauri backend: method not wired yet: ${method} (args: ${args.length})`,
                );
            }
            try {
                return await tauriInvoke<T>(method);
            } catch (err) {
                console.error("Tauri invoke failed", { method, err });
                throw new BackendInvokeError({
                    mode: "tauri",
                    method,
                    cause: err,
                });
            }
        }

        try {
            return await tauriInvoke<T>(method, invokeArgs);
        } catch (err) {
            console.error("Tauri invoke failed", { method, invokeArgs, err });
            throw new BackendInvokeError({
                mode: "tauri",
                method,
                args: invokeArgs,
                cause: err,
            });
        }
    }

    const api = (await waitForPyWebviewApi(1500)) ?? null;
    if (!api || typeof api[method] !== "function") {
        throw new BackendInvokeError({
            mode: "pywebview",
            method,
            args,
            cause: new Error("Python API not available"),
        });
    }

    try {
        return (await api[method](...args)) as T;
    } catch (err) {
        console.error("pywebview api call failed", { method, args, err });
        throw new BackendInvokeError({
            mode: "pywebview",
            method,
            args,
            cause: err,
        });
    }
}
