import type {
    ParamFramesPayload,
    ProcessorParamDescriptor,
    StaticParamValuePayload,
    TimelineResult,
} from "../../types/api";

import { invoke } from "../invoke";

export const paramsApi = {
    getParamFrames: (
        trackId: string,
        param: string,
        startFrame: number,
        frameCount: number,
        stride?: number,
    ) =>
        invoke<ParamFramesPayload>(
            "get_param_frames",
            trackId,
            param,
            startFrame,
            frameCount,
            stride,
        ),

    setParamFrames: (
        trackId: string,
        param: string,
        startFrame: number,
        values: number[],
        checkpoint?: boolean,
    ) =>
        invoke<{ ok: boolean }>(
            "set_param_frames",
            trackId,
            param,
            startFrame,
            values,
            checkpoint,
        ),

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

    setStaticParam: (
        trackId: string,
        param: string,
        value: number,
        checkpoint?: boolean,
    ) =>
        invoke<{ ok: boolean }>(
            "set_static_param",
            trackId,
            param,
            value,
            checkpoint,
        ),

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

    pasteReaperClipboard: (
        selectionStartFrame?: number,
        selectionMaxFrames?: number,
    ) =>
        invoke<
            TimelineResult & {
                ok: boolean;
                error?: string;
                skipped_files?: string[];
            }
        >("paste_reaper_clipboard", selectionStartFrame, selectionMaxFrames),

    getProcessorParams: (algo: string) =>
        invoke<ProcessorParamDescriptor[]>("get_processor_params", algo),

    getMidiTracks: (midiPath: string) =>
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
        }>("get_midi_tracks", midiPath),

    importMidiToPitch: (
        midiPath: string,
        trackIndex?: number,
        selectionStartFrame?: number,
        selectionMaxFrames?: number,
    ) =>
        invoke<{
            ok: boolean;
            error?: string;
            notes_imported?: number;
            frames_touched?: number;
        }>("import_midi_to_pitch", midiPath, trackIndex, selectionStartFrame, selectionMaxFrames),
};
