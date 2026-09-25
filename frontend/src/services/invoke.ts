// 统一封装 Tauri / pywebview 调用
// - Tauri: window.__TAURI__.core.invoke / window.__TAURI__.invoke (named args)
// - pywebview: window.pywebview.api[method] (positional args)

/* eslint-disable @typescript-eslint/no-explicit-any */

import { reportFrontendError } from "./frontendErrorLog";

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

/** 位置参数 → Tauri 命名参数（按各命令的手写映射表）。导出供回归测试锁定。 */
export function buildTauriArgs(method: string, args: unknown[]): BuildArgsResult {
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
            return {
                trackId: args[0],
                ...(args[1] !== undefined ? { parentTrackId: args[1] } : {}),
                ...(args[2] !== undefined ? { targetIndex: args[2] } : {}),
            };

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

        case "analyze_clip_formants":
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
                clipPlaybackRate: args[9],
                reversed: args[10],
                loopEnabled: args[11],
                snapOffsetSec: args[12],
                fadeInSec: args[13],
                fadeOutSec: args[14],
                fadeInShape: args[15],
                fadeOutShape: args[16],
                fadeInDir: args[17],
                fadeOutDir: args[18],
                autoFadeInSec: args[19],
                autoFadeOutSec: args[20],
                color: args[21],
                formantMorph: args[22],
                checkpoint: args[23],
            };

        case "set_clips_state_bulk":
            return {
                updates: args[0],
                checkpoint: args[1],
            };

        case "set_clip_active_take":
            return {
                clipId: args[0],
                takeId: args[1],
                checkpoint: args[2],
            };

        case "cycle_clip_takes":
            return {
                clipIds: args[0],
                direction: args[1],
                checkpoint: args[2],
            };

        case "pack_clips_into_takes":
            return {
                clipIds: args[0],
                checkpoint: args[1],
            };

        case "explode_clip_takes":
            return {
                clipId: args[0],
                checkpoint: args[1],
            };

        case "duplicate_clip_take":
            return {
                clipId: args[0],
                takeId: args[1],
                checkpoint: args[2],
            };

        case "remove_clip_take":
            return {
                clipId: args[0],
                takeId: args[1],
                checkpoint: args[2],
            };

        case "rename_clip_take":
            return {
                clipId: args[0],
                takeId: args[1],
                name: args[2],
                checkpoint: args[3],
            };

        case "set_clip_take_reversed":
            return {
                clipId: args[0],
                takeId: args[1],
                reversed: args[2],
                checkpoint: args[3],
            };

        case "set_clip_take_channel_mode":
            return {
                clipId: args[0],
                takeId: args[1],
                channelMode: args[2],
                checkpoint: args[3],
            };

        case "scan_and_convert_fake_stereo":
            // 两个参数都可选：缺省 = 整个工程 + 实扫。用条件展开而不是直接传
            // undefined，与同文件的 import_audio_item 保持一致的显式风格。
            return {
                ...(args[0] !== undefined ? { clipIds: args[0] } : {}),
                ...(args[1] !== undefined ? { dryRun: args[1] } : {}),
            };

        case "add_clip_take_from_media":
            return {
                clipId: args[0],
                sourcePath: args[1],
                name: args[2],
                checkpoint: args[3],
            };

        case "import_media_files_as_takes":
            return {
                paths: args[0],
                trackId: args[1],
                startSec: args[2],
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

        case "close_track_gaps":
            return { trackId: args[0], fromSec: args[1] };

        case "analyze_clip_silence":
            return { clipIds: args[0], options: args[1] };

        case "remove_clip_silence":
            return { clipIds: args[0], options: args[1] };

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

        case "set_metronome":
            return {
                enabled: args[0],
                gain: args[1],
                mode: args[2],
                accent: args[3],
                sound: args[4],
            };

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

        case "set_project_notes":
            return { notesMarkdown: args[0] };

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
        case "clipboard_kind":
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

        case "save_project_to_path":
            return {
                projectPath: args[0],
                ...(args[1] !== undefined ? { notesMarkdown: args[1] } : {}),
                ...(args[2] !== undefined ? { force: args[2] } : {}),
            };

        case "import_vocalshifter_project":
            return { vspPath: args[0] };

        case "import_reaper_project":
            return { rppPath: args[0] };

        case "paste_vocalshifter_clipboard":
            return {
                ...(args[0] !== undefined ? { selectionRanges: args[0] } : {}),
                ...(args[1] !== undefined ? { activeParam: args[1] } : {}),
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

        case "convert_mix_param":
            return {
                trackId: args[0],
                from: args[1],
                ranges: args[2],
            };

        case "get_param_frames":
            return {
                trackId: args[0],
                param: args[1],
                startFrame: args[2],
                frameCount: args[3],
                stride: args[4],
                binary: args[5],
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

        case "stretch_track_linked_params":
            return {
                trackId: args[0],
                mappings: args[1],
                checkpoint: args[2],
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
                ...(args[2] !== undefined ? { selectionRanges: args[2] } : {}),
                ...(args[3] !== undefined ? { fillGaps: args[3] } : {}),
                ...(args[4] !== undefined ? { noteBpmMode: args[4] } : {}),
                ...(args[5] !== undefined ? { specifiedBpm: args[5] } : {}),
                ...(args[6] !== undefined ? { importMidiBpmAsProject: args[6] } : {}),
                ...(args[7] != null ? { clipboardGuid: args[7] } : {}),
                ...(args[8] !== undefined ? { closeLeadingGap: args[8] } : {}),
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

        case "save_recording_settings":
            return { settings: args[0] };

        case "start_recording":
            return { startSec: args[0] };

        case "begin_undo_group":
            // 可选 label（语言无关的操作 key）：批量导入等场景给出更准确的名字。
            return args[0] === undefined ? undefined : { label: args[0] };

        case "set_history_position":
            return { position: args[0] };

        case "record_param_selection_step":
            // 「边缘拉伸」手势的选区快照：登记到当前那一步历史记录上
            // （撤销/重做时由后端随载荷带回，见 state.rs 的说明）。
            return { before: args[0], after: args[1] };

        case "set_project_save_undo_history":
            return { enabled: args[0] };

        case "end_undo_group":
            return undefined;

        case "clear_render_cache":
            return {
                scope: args[0],
                ...(args[1] !== undefined ? { days: args[1] } : {}),
            };

        case "get_render_cache_stats":
        case "open_render_cache_dir":
            return {};

        // ── 记事本（附件 / 剪贴板暂存 / 导出）──
        case "notebook_put_asset":
            return {
                assetId: args[0],
                kind: args[1],
                ext: args[2],
                mime: args[3] ?? null,
                dataBase64: args[4],
                meta: args[5] ?? null,
            };

        case "notebook_read_asset":
        case "notebook_remove_asset":
            return { assetId: args[0] };

        case "notebook_read_file_base64":
            return {
                path: args[0],
                ...(args[1] !== undefined ? { maxBytes: args[1] } : {}),
            };

        case "notebook_write_clipboard_payload":
            return {
                payloadBase64: args[0],
                textSummary: args[1] ?? null,
            };

        case "notebook_export_document":
            return {
                suggestedName: args[0],
                extension: args[1],
                content: args[2],
            };

        case "notebook_save_asset_as":
            return {
                assetId: args[0],
                suggestedName: args[1] ?? null,
            };

        case "export_diagnostics":
            return { outputPath: args[0] };

        case "log_frontend_error":
            return { message: args[0], detail: args[1] ?? null };

        default:
            // 无参命令白名单：不携带参数的命令在此统一登记（invoke 时以空
            // args 对象调用，与无 args 调用在 Tauri 侧等价）。新命令若携带
            // 参数，必须在上方 switch 显式登记位置参数 → 命名参数的映射，
            // 否则 buildTauriArgs 返回 __unwired、invoke 直接 throw ——
            // 这曾三次造成"前端乐观更新生效、后端调用从未到达"的静默分叉
            // （take 命令族、set_clip_take_reversed、set_clip_take_channel_mode）。
            // invoke.wiring.test.ts 会扫描全部调用点做穷举防回归。
            if (NO_ARG_COMMANDS.has(method)) return {};
            return { __unwired: true };
    }
}

/**
 * 无参命令白名单（不携带任何位置参数的 Tauri 命令）。
 * 新增无参命令时在此登记；新增带参命令必须在 switch 中登记映射。
 */
const NO_ARG_COMMANDS: ReadonlySet<string> = new Set([
    "cancel_background_render",
    "cancel_export_audio",
    "check_source_files_changed",
    "clear_waveform_cache",
    "clipboard_kind",
    "close_window",
    "consume_startup_project_path",
    "get_about_info",
    "get_project_meta",
    "get_auto_backup_settings",
    "get_dml_adapters",
    "get_export_audio_defaults",
    "get_gpu_devices",
    "get_history_state",
    "get_onnx_diagnostic",
    "get_onnx_status",
    "get_pitch_analysis_progress",
    "get_playback_state",
    "get_recording_apps",
    "get_recording_devices",
    "get_recording_settings",
    "get_recording_state",
    "get_runtime_info",
    "get_timeline_state",
    "get_ui_settings",
    "has_reaper_clipboard",
    "has_timeline_clipboard",
    "import_project_dialog",
    "load_default_model",
    "new_project",
    "notebook_list_assets",
    "notebook_prune_assets",
    "notebook_read_clipboard_image",
    "notebook_read_clipboard_payload",
    "seal_project_notes_history",
    "open_audio_dialog",
    "open_audio_dialog_multi",
    "open_log_folder",
    "open_midi_dialog",
    "pick_output_path",
    "open_project_dialog",
    "open_reaper_dialog",
    "open_vocalshifter_dialog",
    "pick_diagnostics_output_path",
    "pick_directory",
    "pick_midi_output_path",
    "ping",
    "read_system_clipboard_object",
    "redo_timeline",
    "run_vocoder_benchmark",
    "start_background_render",
    "stop_audio",
    "stop_recording",
    "synthesize",
    "undo_timeline",
]);

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
                reportFrontendError(`Invoke failed: ${method}`, err);
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
            reportFrontendError(`Invoke failed: ${method}`, err);
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
        // 与 Tauri 分支同口径：失败也上报前端诊断日志（pywebview 模式下
        // BackendInvokeError 只剩 message，原始 cause 不落盘就丢了）。
        reportFrontendError(`Invoke failed: ${method}`, err);
        throw new BackendInvokeError({
            mode: "pywebview",
            method,
            args,
            cause: err,
        });
    }
}
