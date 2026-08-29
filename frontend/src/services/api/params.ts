import type {
    ParamFramesPayload,
    ProcessorParamDescriptor,
    StaticParamValuePayload,
    TimelineResult,
} from "../../types/api";

import { invoke } from "../invoke";
import {
    decodeParamFramesFromBase64,
    paramFramesBinaryToArrays,
} from "../../components/layout/pianoRoll/paramFramesBinaryCodec";

export const paramsApi = {
    /**
     * 取参数曲线段。
     *
     * `binary=true` 时后端把 orig/edit 编码成 Base64 二进制（见
     * `pianoRoll/paramFramesBinaryCodec.ts`），返回体里 `orig`/`edit` 为空数组、
     * `binary` 为编码串。相比 JSON number[] 体积约缩小 4 倍，解析不再阻塞主线程。
     *
     * 默认开启二进制：调用方拿到的 payload 已带解码后的 `orig`/`edit`。
     */
    getParamFrames: (
        trackId: string,
        param: string,
        startFrame: number,
        frameCount: number,
        stride?: number,
        binary = true,
    ) =>
        invoke<ParamFramesPayload>(
            "get_param_frames",
            trackId,
            param,
            startFrame,
            frameCount,
            stride,
            binary,
        ).then((res) => {
            // 在 API 层统一解码：调用方拿到的 payload 与二进制模式开启前结构一致，
            // 六处取数点无需感知传输格式。
            const encoded = res?.binary;
            if (!res || !encoded) return res;
            const decoded = decodeParamFramesFromBase64(encoded);
            if (!decoded) return res; // 解码失败 → 回退空数组，调用方按 not-ok 处理
            const { orig, edit } = paramFramesBinaryToArrays(decoded);
            return { ...res, orig, edit, binary: undefined };
        }),

    setParamFrames: (
        trackId: string,
        param: string,
        startFrame: number,
        values: number[],
        checkpoint?: boolean,
    ) =>
        invoke<{ ok: boolean }>("set_param_frames", trackId, param, startFrame, values, checkpoint),

    restoreParamFrames: (
        trackId: string,
        param: string,
        startFrame: number,
        frameCount: number,
        checkpoint?: boolean,
    ) =>
        invoke<{ ok: boolean }>(
            "restore_param_frames",
            trackId,
            param,
            startFrame,
            frameCount,
            checkpoint,
        ),

    getStaticParam: (trackId: string, param: string) =>
        invoke<StaticParamValuePayload>("get_static_param", trackId, param),

    setStaticParam: (trackId: string, param: string, value: number, checkpoint?: boolean) =>
        invoke<{ ok: boolean }>("set_static_param", trackId, param, value, checkpoint),

    pasteVocalShifterClipboard: (
        selectionStartFrame?: number,
        selectionMaxFrames?: number,
        activeParam?: string,
    ) =>
        invoke<{ ok: boolean; error?: string; updated?: number }>(
            "paste_vocalshifter_clipboard",
            selectionStartFrame,
            selectionMaxFrames,
            activeParam,
        ),

    pasteReaperClipboard: (selectionStartFrame?: number, selectionMaxFrames?: number) =>
        invoke<
            TimelineResult & {
                ok: boolean;
                error?: string;
                skipped_files?: string[];
            }
        >("paste_reaper_clipboard", selectionStartFrame, selectionMaxFrames),

    getProcessorParams: (algo: string) =>
        invoke<ProcessorParamDescriptor[]>("get_processor_params", algo),

    getMidiTracks: (midiPath: string, clipboardGuid?: string) =>
        invoke<{
            ok: boolean;
            error?: string;
            tracks?: Array<{
                index: number;
                name: string;
                note_count: number;
                min_note: number;
                max_note: number;
            }>;
            initial_bpm?: number;
            has_bpm?: boolean;
            has_time_signature?: boolean;
            has_key_signature?: boolean;
            tempo_point_count?: number;
            time_signature_count?: number;
            key_signature_count?: number;
        }>("get_midi_tracks", midiPath, clipboardGuid ?? null),

    readMidiClipboardToMemory: () =>
        invoke<{
            ok: boolean;
            error?: string;
            guid?: string;
            tracks?: Array<{
                index: number;
                name: string;
                note_count: number;
                min_note: number;
                max_note: number;
            }>;
            initial_bpm?: number;
            has_bpm?: boolean;
            has_time_signature?: boolean;
            has_key_signature?: boolean;
            tempo_point_count?: number;
            time_signature_count?: number;
            key_signature_count?: number;
        }>("read_midi_clipboard_to_memory"),

    importMidiToPitch: (
        midiPath: string,
        trackIndices: number[],
        selectionStartFrame?: number,
        selectionMaxFrames?: number,
        fillGaps?: boolean,
        noteBpmMode?: string,
        specifiedBpm?: number,
        importMidiBpmAsProject?: boolean,
        clipboardGuid?: string,
        closeLeadingGap?: boolean,
    ) =>
        invoke<{
            ok: boolean;
            error?: string;
            notes_imported?: number;
            frames_touched?: number;
        }>(
            "import_midi_to_pitch",
            midiPath,
            trackIndices,
            selectionStartFrame,
            selectionMaxFrames,
            fillGaps,
            noteBpmMode,
            specifiedBpm,
            importMidiBpmAsProject,
            clipboardGuid ?? null,
            closeLeadingGap,
        ),

    importMidiAsClip: (
        midiPath: string,
        trackIndices: number[],
        trackId?: string,
        startSec?: number,
        fillGaps?: boolean,
        multiTrackMerge?: boolean,
        noteBpmMode?: string,
        specifiedBpm?: number,
        importMidiBpmAsProject?: boolean,
        clipboardGuid?: string,
        closeLeadingGap?: boolean,
        importMidiAsTempoMap?: boolean,
        importMidiTempo?: boolean,
        importMidiTimeSignature?: boolean,
        importMidiKeySignature?: boolean,
    ) =>
        invoke<TimelineResult & { ok: boolean; error?: string }>(
            "import_midi_as_clip",
            midiPath,
            trackIndices,
            trackId,
            startSec,
            fillGaps,
            multiTrackMerge,
            noteBpmMode,
            specifiedBpm,
            importMidiBpmAsProject,
            clipboardGuid ?? null,
            closeLeadingGap,
            importMidiAsTempoMap,
            importMidiTempo,
            importMidiTimeSignature,
            importMidiKeySignature,
        ),

    replaceMidiClipData: (
        clipId: string,
        midiPath: string,
        trackIndices: number[],
        fillGaps?: boolean,
        noteBpmMode?: string,
        specifiedBpm?: number,
        importMidiBpmAsProject?: boolean,
        clipboardGuid?: string,
        closeLeadingGap?: boolean,
    ) =>
        invoke<TimelineResult & { ok: boolean; error?: string }>(
            "replace_midi_clip_data",
            clipId,
            midiPath,
            trackIndices,
            fillGaps,
            noteBpmMode,
            specifiedBpm,
            importMidiBpmAsProject,
            clipboardGuid ?? null,
            closeLeadingGap,
        ),

    exportPitchToMidi: (request: {
        outputPath: string;
        tracks: Array<{
            trackId: string;
            rootTrackId: string;
            name: string;
            startSec: number;
            endSec: number;
            clipId?: string;
        }>;
        bpm: number;
        beatsPerBar: number;
        baseScale: string;
        projectScaleNotes: number[];
    }) => invoke<{ ok: boolean; error?: string }>("export_pitch_to_midi", request),
};
