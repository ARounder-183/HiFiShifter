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
        withSentinel = false,
    ) =>
        invoke<ParamFramesPayload>(
            "get_param_frames",
            trackId,
            param,
            startFrame,
            frameCount,
            stride,
            binary,
            withSentinel,
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

    /**
     * 把「参数编辑器边缘拉伸」带来的选区变化登记到**当前那一步**历史记录上。
     *
     * 必须在曲线回写（`set_param_frames`，首块打检查点）成功之后调用：后端据此
     * 把快照挂在刚产生的那一步上，撤销/重做该步时随载荷带回
     * （`param_selection_restore`）—— 前端因此不需要按撤销深度推断"我这一步是
     * 第几步"（那条路会因镜像滞后 / 分支裁剪而错位，见 state.rs 的说明）。
     *
     * 每一对是 `[startFrame, frameCount]`（**帧**单位，与选区的内部单位一致）。
     *
     * 后端只接受「参数曲线」步；写入被抑制等情况下返回 `ok = false`（忽略即可）。
     */
    recordParamSelectionStep: (before: [number, number][], after: [number, number][]) =>
        invoke<{ ok: boolean; reason?: string }>("record_param_selection_step", before, after),

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

    /**
     * 音量 ↔ 动态 曲线互转（后端单事务）。
     *
     * 换算（基线补偿）与源参数归位都在后端完成：逐帧基线与曲线存在性只有
     * 后端权威，前端不做任何算术。`ranges` 为参数编辑器的多选区。
     * 原声基线分析未就绪时返回 `{ ok: false, reason: "analysis_pending" }`。
     */
    convertMixParam: (
        trackId: string,
        from: "volume" | "dyn",
        ranges: Array<{ startFrame: number; frameCount: number }>,
    ) =>
        invoke<{
            ok: boolean;
            reason?: string;
            convertedFrames?: number;
            skippedFrames?: number;
        }>("convert_mix_param", trackId, from, ranges),

    /**
     * "锁定参数线"：剪辑拉伸后把旧范围内的参数曲线时域映射到新范围。
     *
     * 由后端一次性完成 pitch（用户编辑过时）/ tension / 所有已存在的自动化
     * 曲线的批量映射与旧范围恢复——曲线清单只能由后端枚举，前端旧实现只
     * 覆盖 pitch+tension，导致其余参数线在拉伸后遗留在旧位置。
     */
    stretchTrackLinkedParams: (
        trackId: string,
        mappings: Array<{
            oldStartSec: number;
            oldLengthSec: number;
            newStartSec: number;
            newLengthSec: number;
        }>,
        checkpoint?: boolean,
    ) => invoke<{ ok: boolean }>("stretch_track_linked_params", trackId, mappings, checkpoint),

    getStaticParam: (trackId: string, param: string) =>
        invoke<StaticParamValuePayload>("get_static_param", trackId, param),

    setStaticParam: (trackId: string, param: string, value: number, checkpoint?: boolean) =>
        invoke<{ ok: boolean }>("set_static_param", trackId, param, value, checkpoint),

    /**
     * 粘贴 VocalShifter 剪贴板。
     * `selectionRanges` 为参数编辑器的多选区（每段 startFrame/frameCount）：
     * 后端把数据对齐到**首段起点**，且只写入落在任一段内的帧（断层不填充）。
     */
    pasteVocalShifterClipboard: (
        selectionRanges?: Array<{ startFrame: number; frameCount: number }>,
        activeParam?: string,
    ) =>
        invoke<{ ok: boolean; error?: string; updated?: number }>(
            "paste_vocalshifter_clipboard",
            selectionRanges,
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

    /**
     * 导入 MIDI 到参数线（pitch）。
     * `selectionRanges` 为参数编辑器多选区（每段 startFrame/frameCount）：
     * 音符对齐首段起点，且只写入落在任一段内的帧（断层保持原值）。
     */
    importMidiToPitch: (
        midiPath: string,
        trackIndices: number[],
        selectionRanges?: Array<{ startFrame: number; frameCount: number }>,
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
            selectionRanges,
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
