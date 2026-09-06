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

export interface SourceFileMatchCandidate {
    path: string;
    exact_hash: boolean;
}

export interface SearchSourceFileMatchesResult {
    matches: Record<string, SourceFileMatchCandidate[]>;
}

/**
 * Clip 源共振峰分析结果（analyze_clip_formants）。
 * 与后端 commands/formant.rs 的 ClipFormantAnalysisPayload 一一对应。
 */
export interface ClipFormantAnalysisResult {
    ok: boolean;
    /** 统计源 F1（检出帧中位数，Hz；无检出为 0） */
    sourceF1Hz: number;
    /** 统计源 F2（Hz；无检出为 0） */
    sourceF2Hz: number;
    /** 稀疏轨迹 [t_norm, f1_hz, f2_hz]，按时间升序，≤64 点 */
    track: Array<[number, number, number]>;
    /** 检出候选的分析帧占比 [0,1] */
    voicedRatio: number;
    /** 诊断消息："source_too_short" / "no_voiced_frames" */
    message: string | null;
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
    importAudioItem: (
        audioPath: string,
        trackId?: string | null,
        startSec?: number,
        mediaAudioStreamIndex?: number,
    ) =>
        invoke<TimelineResult>(
            "import_audio_item",
            audioPath,
            trackId,
            startSec,
            mediaAudioStreamIndex,
        ),

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

    duplicateTrack: (
        trackId: string,
        placement?: { parentTrackId?: string | null; targetIndex?: number },
    ) =>
        invoke<TimelineResult>(
            "duplicate_track",
            trackId,
            placement?.parentTrackId ?? null,
            placement?.targetIndex,
        ),

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
        clipPlaybackRate?: number;
        reversed?: boolean;
        loopEnabled?: boolean;
        snapOffsetSec?: number;
        fadeInSec?: number;
        fadeOutSec?: number;
        fadeInShape?: number;
        fadeOutShape?: number;
        fadeInDir?: number;
        fadeOutDir?: number;
        autoFadeInSec?: number;
        autoFadeOutSec?: number;
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
            payload.clipPlaybackRate,
            payload.reversed,
            payload.loopEnabled,
            payload.snapOffsetSec,
            payload.fadeInSec,
            payload.fadeOutSec,
            payload.fadeInShape,
            payload.fadeOutShape,
            payload.fadeInDir,
            payload.fadeOutDir,
            payload.autoFadeInSec,
            payload.autoFadeOutSec,
            payload.color,
            payload.formantMorph,
            payload.checkpoint,
        ),

    /**
     * 分析 clip 源音频的共振峰（统计 F1/F2 + 稀疏轨迹）。
     * 用于共振峰工具窗口的"源点 → 目标点"可视化。
     */
    analyzeClipFormants: (clipId: string) =>
        invoke<ClipFormantAnalysisResult>("analyze_clip_formants", clipId),

    setClipsStateBulk: (payload: {
        updates: Array<{
            clipId: string;
            gain?: number;
            muted?: boolean;
            startSec?: number;
            lengthSec?: number;
            sourceStartSec?: number;
            sourceEndSec?: number;
            snapOffsetSec?: number;
            clipPlaybackRate?: number;
            fadeInSec?: number;
            fadeOutSec?: number;
            fadeInShape?: number;
            fadeInDir?: number;
            fadeOutShape?: number;
            fadeOutDir?: number;
            autoFadeInSec?: number;
            autoFadeOutSec?: number;
            /** 倒放开关（后端 ClipStatePatch 支持，必须与乐观更新字段一致）。 */
            reversed?: boolean;
            /** Loop（循环源）开关。 */
            loopEnabled?: boolean;
        }>;
        checkpoint?: boolean;
    }) => invoke<TimelineResult>("set_clips_state_bulk", payload.updates, payload.checkpoint),

    setClipActiveTake: (payload: { clipId: string; takeId: string; checkpoint?: boolean }) =>
        invoke<TimelineResult>(
            "set_clip_active_take",
            payload.clipId,
            payload.takeId,
            payload.checkpoint,
        ),

    cycleClipTakes: (payload: { clipIds: string[]; direction?: number; checkpoint?: boolean }) =>
        invoke<TimelineResult>(
            "cycle_clip_takes",
            payload.clipIds,
            payload.direction,
            payload.checkpoint,
        ),

    packClipsIntoTakes: (payload: { clipIds: string[]; checkpoint?: boolean }) =>
        invoke<TimelineResult>("pack_clips_into_takes", payload.clipIds, payload.checkpoint),

    explodeClipTakes: (payload: { clipId: string; checkpoint?: boolean }) =>
        invoke<TimelineResult>("explode_clip_takes", payload.clipId, payload.checkpoint),

    duplicateClipTake: (payload: { clipId: string; takeId: string; checkpoint?: boolean }) =>
        invoke<TimelineResult>(
            "duplicate_clip_take",
            payload.clipId,
            payload.takeId,
            payload.checkpoint,
        ),

    removeClipTake: (payload: { clipId: string; takeId: string; checkpoint?: boolean }) =>
        invoke<TimelineResult>(
            "remove_clip_take",
            payload.clipId,
            payload.takeId,
            payload.checkpoint,
        ),

    renameClipTake: (payload: {
        clipId: string;
        takeId: string;
        name: string;
        checkpoint?: boolean;
    }) =>
        invoke<TimelineResult>(
            "rename_clip_take",
            payload.clipId,
            payload.takeId,
            payload.name,
            payload.checkpoint,
        ),

    setClipTakeReversed: (payload: {
        clipId: string;
        takeId: string;
        reversed: boolean;
        checkpoint?: boolean;
    }) =>
        invoke<TimelineResult>(
            "set_clip_take_reversed",
            payload.clipId,
            payload.takeId,
            payload.reversed,
            payload.checkpoint,
        ),

    addClipTakeFromMedia: (payload: {
        clipId: string;
        sourcePath: string;
        name?: string;
        checkpoint?: boolean;
    }) =>
        invoke<TimelineResult>(
            "add_clip_take_from_media",
            payload.clipId,
            payload.sourcePath,
            payload.name,
            payload.checkpoint,
        ),

    importMediaFilesAsTakes: (payload: {
        paths: string[];
        trackId?: string | null;
        startSec?: number;
    }) =>
        invoke<TimelineResult>(
            "import_media_files_as_takes",
            payload.paths,
            payload.trackId,
            payload.startSec,
        ),

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

    pasteTimelineClipboard: (mode?: "selected" | "new_tracks") =>
        invoke<
            TimelineResult & {
                error?: string;
                sourceProject?: string;
                importedTrackCount?: number;
                importedClipCount?: number;
            }
        >("paste_timeline_clipboard", mode),

    hasTimelineClipboard: () =>
        invoke<{
            ok: boolean;
            available?: boolean;
            kind?: "clips" | "tracks" | "project" | "reaper";
            clipCount?: number;
            trackCount?: number;
            sourceProject?: string;
        }>("has_timeline_clipboard"),

    // 剪贴板载荷类型探测（粘贴内容路由依据）：后端缓存优先，未变化时
    // 不打开系统剪贴板。kind："clips" | "tracks" | "project" | "param" | null。
    clipboardKind: () => invoke<{ ok: boolean; kind: string | null }>("clipboard_kind"),

    hasReaperClipboard: () => invoke<{ ok: boolean; available?: boolean }>("has_reaper_clipboard"),

    /// 在指定文件夹及其子文件夹中搜索候选源文件。
    /// searchMode 为 "file_name"（精准文件名）或 "extension_hash"（文件扩展名 + 哈希）。
    searchSourceFileReplacements: (
        folderPath: string,
        clipIds: string[],
        searchMode: "file_name" | "extension_hash",
    ) =>
        invoke<SearchSourceFileMatchesResult>(
            "search_source_file_replacements",
            folderPath,
            clipIds,
            searchMode,
        ),

    /// 检查所有已导入的媒体源文件是否被外部修改或删除。
    /// 前端在窗口重新获得焦点时调用此方法。
    checkSourceFilesChanged: () =>
        invoke<CheckSourceFilesChangedResult>("check_source_files_changed"),
};
