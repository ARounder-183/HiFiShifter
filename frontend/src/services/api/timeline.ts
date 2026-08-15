import type { TimelineResult, TrackSummaryResult, TempoMapPayload } from "../../types/api";
import type { LinkedParamCurves } from "../../features/session/sessionTypes";

import { invoke } from "../invoke";
import type { ClipTemplate } from "../../features/session/sessionTypes";

export interface SourceFileChange {
    clip_id: string;
    clip_name: string;
    source_path: string;
    /** "deleted" | "modified" */
    change: string;
}

export interface CheckSourceFilesChangedResult {
    changed: SourceFileChange[];
}

export const timelineApi = {
    // Undo/Redo (backend-authoritative)
    undoTimeline: () => invoke<TimelineResult>("undo_timeline"),
    redoTimeline: () => invoke<TimelineResult>("redo_timeline"),

    // Undo grouping: all commands between begin/end share a single undo entry
    beginUndoGroup: () => invoke<TimelineResult>("begin_undo_group"),
    endUndoGroup: () => invoke<{ ok: boolean }>("end_undo_group"),

    getTimelineState: () => invoke<TimelineResult>("get_timeline_state"),

    // Transport
    setTransport: (payload: { playheadSec?: number; bpm?: number }) =>
        invoke<{ ok: boolean; playhead_sec?: number; bpm?: number }>(
            "set_transport",
            payload.playheadSec,
            payload.bpm,
        ),

    setProjectLength: (projectSec: number) =>
        invoke<TimelineResult>("set_project_length", projectSec),

    // Tempo Map
    setTimelineTempoMap: (tempoMap: TempoMapPayload | null) =>
        invoke<TimelineResult>("set_timeline_tempo_map", tempoMap),

    // Import
    importAudioItem: (audioPath: string, trackId?: string | null, startSec?: number) =>
        invoke<TimelineResult>("import_audio_item", audioPath, trackId, startSec),

    importAudioBytes: (
        fileName: string,
        base64Data: string,
        trackId?: string | null,
        startSec?: number,
    ) => invoke<TimelineResult>("import_audio_bytes", fileName, base64Data, trackId, startSec),

    // Tracks
    addTrack: (name?: string) => invoke<TimelineResult>("add_track", name),

    addTrackNested: (payload: { name?: string; parentTrackId?: string | null; index?: number }) =>
        invoke<TimelineResult>(
            "add_track",
            payload.name,
            payload.parentTrackId ?? null,
            payload.index,
        ),

    removeTrack: (trackId: string) => invoke<TimelineResult>("remove_track", trackId),

    duplicateTrack: (trackId: string) => invoke<TimelineResult>("duplicate_track", trackId),

    moveTrack: (payload: { trackId: string; targetIndex: number; parentTrackId?: string | null }) =>
        invoke<TimelineResult>(
            "move_track",
            payload.trackId,
            payload.targetIndex,
            payload.parentTrackId ?? null,
        ),

    setTrackState: (payload: {
        trackId: string;
        muted?: boolean;
        solo?: boolean;
        volume?: number;
        composeEnabled?: boolean;
        pitchAnalysisAlgo?: string;
        color?: string;
        name?: string;
    }) =>
        invoke<TimelineResult>(
            "set_track_state",
            payload.trackId,
            payload.muted,
            payload.solo,
            payload.volume,
            payload.composeEnabled,
            payload.pitchAnalysisAlgo,
            payload.color,
            payload.name,
        ),

    selectTrack: (trackId: string) => invoke<TimelineResult>("select_track", trackId),

    getTrackSummary: (trackId?: string) => invoke<TrackSummaryResult>("get_track_summary", trackId),

    // Clips
    addClip: (payload: {
        trackId?: string;
        name?: string;
        startSec?: number;
        lengthSec?: number;
        sourcePath?: string;
    }) =>
        invoke<TimelineResult>(
            "add_clip",
            payload.trackId,
            payload.name,
            payload.startSec,
            payload.lengthSec,
            payload.sourcePath,
        ),

    createClipsBulk: (payload: { templates: ClipTemplate[]; selectCreatedClips?: boolean }) =>
        invoke<TimelineResult>("create_clips_bulk", payload),

    removeClip: (clipId: string) => invoke<TimelineResult>("remove_clip", clipId),

    removeClips: (clipIds: string[]) => invoke<TimelineResult>("remove_clips", clipIds),

    moveClip: (payload: {
        clipId: string;
        startSec: number;
        trackId?: string;
        moveLinkedParams?: boolean;
    }) =>
        invoke<TimelineResult>(
            "move_clip",
            payload.clipId,
            payload.startSec,
            payload.trackId,
            payload.moveLinkedParams,
        ),

    moveClips: (payload: {
        moves: Array<{
            clipId: string;
            startSec: number;
            trackId?: string;
        }>;
        moveLinkedParams?: boolean;
    }) => invoke<TimelineResult>("move_clips", payload.moves, payload.moveLinkedParams),

    getClipLinkedParams: (clipId: string) =>
        invoke<{ ok: boolean; linkedParams?: LinkedParamCurves }>("get_clip_linked_params", clipId),

    applyClipLinkedParams: (payload: { clipId: string; linkedParams: LinkedParamCurves }) =>
        invoke<TimelineResult>("apply_clip_linked_params", payload.clipId, payload.linkedParams),

    setClipState: (payload: {
        clipId: string;
        name?: string;
        startSec?: number;
        lengthSec?: number;
        gain?: number;
        muted?: boolean;
        sourceStartSec?: number;
        sourceEndSec?: number;
        playbackRate?: number;
        reversed?: boolean;
        fadeInSec?: number;
        fadeOutSec?: number;
        fadeInCurve?: string;
        fadeOutCurve?: string;
        color?: string;
        formantMorph?: {
            enabled: boolean;
            targetF1Hz: number;
            targetF2Hz: number;
            strength: number;
        };
        /** 是否创建 undo checkpoint，默认为 true */
        checkpoint?: boolean;
    }) =>
        invoke<TimelineResult>(
            "set_clip_state",
            payload.clipId,
            payload.name,
            payload.startSec,
            payload.lengthSec,
            payload.gain,
            payload.muted,
            payload.sourceStartSec,
            payload.sourceEndSec,
            payload.playbackRate,
            payload.reversed,
            payload.fadeInSec,
            payload.fadeOutSec,
            payload.fadeInCurve,
            payload.fadeOutCurve,
            payload.color,
            payload.formantMorph,
            payload.checkpoint,
        ),

    setClipsStateBulk: (payload: {
        updates: Array<{
            clipId: string;
            gain?: number;
            muted?: boolean;
            fadeInSec?: number;
            fadeOutSec?: number;
        }>;
        checkpoint?: boolean;
    }) => invoke<TimelineResult>("set_clips_state_bulk", payload.updates, payload.checkpoint),

    duplicateClipsBulk: (payload: {
        sourceClipIds: string[];
        deltaSec: number;
        trackMode: Record<string, unknown>;
        copyLinkedParams?: boolean;
        selectCreatedClips?: boolean;
        applyAutoCrossfade?: boolean;
        placeOnSelectedTrack?: boolean;
        renameCopies?: boolean;
    }) => invoke<TimelineResult>("duplicate_clips_bulk", payload),

    replaceClipSource: (payload: {
        clipIds: string[];
        newSourcePath: string;
        replaceSameSource?: boolean;
    }) =>
        invoke<TimelineResult>(
            "replace_clip_source",
            payload.clipIds,
            payload.newSourcePath,
            payload.replaceSameSource,
        ),

    splitClip: (clipId: string, splitSec: number) =>
        invoke<TimelineResult>("split_clip", clipId, splitSec),

    splitClipsAt: (clipIds: string[], splitSec: number) =>
        invoke<TimelineResult>("split_clips_at", clipIds, splitSec),

    glueClips: (clipIds: string[]) => invoke<TimelineResult>("glue_clips", clipIds),

    groupClips: (clipIds: string[]) => invoke<TimelineResult>("group_clips", clipIds),

    ungroupClips: (clipIds: string[]) => invoke<TimelineResult>("ungroup_clips", clipIds),

    toggleGroupDisabled: (groupId: string) =>
        invoke<TimelineResult>("toggle_group_disabled", groupId),

    convertClipsToPitchReference: (clipIds: string[]) =>
        invoke<TimelineResult>("convert_clips_to_pitch_reference", clipIds),

    updatePitchReference: (clipIds: string[]) =>
        invoke<TimelineResult>("update_pitch_reference", clipIds),

    selectClip: (clipId: string | null) => invoke<TimelineResult>("select_clip", clipId),

    // Native cross-process timeline clipboard (backend system clipboard)
    copyTimelineClips: (clipIds: string[]) =>
        invoke<{ ok: boolean; error?: string; kind?: "clips" | "tracks" }>(
            "copy_timeline_clips",
            clipIds,
        ),

    copyTimelineTracks: (trackIds: string[]) =>
        invoke<{ ok: boolean; error?: string; kind?: "clips" | "tracks" }>(
            "copy_timeline_tracks",
            trackIds,
        ),

    pasteTimelineClipboard: () =>
        invoke<
            TimelineResult & {
                error?: string;
                sourceProject?: string;
                importedTrackCount?: number;
                importedClipCount?: number;
            }
        >("paste_timeline_clipboard"),

    hasTimelineClipboard: () =>
        invoke<{
            ok: boolean;
            available?: boolean;
            kind?: "clips" | "tracks" | "project";
            clipCount?: number;
            trackCount?: number;
            sourceProject?: string;
        }>("has_timeline_clipboard"),

    /// 检查所有已导入的音频源文件是否被外部修改或删除。
    /// 前端在窗口重新获得焦点时调用此方法。
    checkSourceFilesChanged: () =>
        invoke<CheckSourceFilesChangedResult>("check_source_files_changed"),
};
