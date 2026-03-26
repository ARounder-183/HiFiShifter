import { createSlice, current, type PayloadAction } from "@reduxjs/toolkit";
import type {
    TimelineClip,
    TimelineState,
    TrackSummaryResult,
} from "../../types/api";
import type {
    AutomationPoint,
    ClipInfo,
    ClipTemplate,
    DrawToolMode,
    DragDirection,
    DrawDragDirection,
    EditParam,
    FadeCurveType,
    GridSize,
    PitchSnapUnit,
    ToolMode,
    ToolModeGroup,
    TrackInfo,
} from "./sessionTypes";

import {
    addClipOnTrack,
    addTrackRemote,
    createClipsRemote,
    duplicateTrackRemote,
    fetchSelectedTrackSummary,
    glueClipsRemote,
    moveClipRemote,
    moveClipsRemote,
    moveTrackRemote,
    removeClipRemote,
    removeClipsRemote,
    removeTrackRemote,
    replaceClipSourceRemote,
    selectClipRemote,
    selectTrackRemote,
    setClipStateRemote,
    setProjectLengthRemote,
    splitClipRemote,
} from "./thunks/timelineThunks";

import {
    newProjectRemote,
    openProjectFromDialog,
    openProjectFromPath,
    openVocalShifterFromDialog,
    openVocalShifterFromPath,
    openReaperFromDialog,
    openReaperFromPath,
    redoRemote,
    saveProjectAsRemote,
    saveProjectRemote,
    setProjectBaseScaleRemote,
    setProjectCustomScaleRemote,
    setProjectTimelineSettingsRemote,
    undoRemote,
} from "./thunks/projectThunks";

import {
    fetchTimeline,
    playOriginal,
    seekPlayhead,
    stopAudioPlayback,
    syncPlaybackState,
    updateTransportBpm,
} from "./thunks/transportThunks";

import {
    clearWaveformCacheRemote,
    loadUiSettings,
    refreshRuntime,
} from "./thunks/runtimeThunks";

import { loadDefaultModel, loadModel } from "./thunks/modelThunks";

import {
    applyPitchShift,
    exportAudio,
    exportSeparated,
    pasteVocalShifterClipboard,
    pasteReaperClipboard,
    pickOutputPath,
    processAudio,
    synthesizeAudio,
} from "./thunks/audioThunks";

import { SCALE_KEYS } from "../../utils/musicalScales";
import type { CustomScalePreset } from "../../utils/customScales";
import { sanitizeCustomScalePreset } from "../../utils/customScales";
import {
    importAudioAtPosition,
    importAudioFileAtPosition,
    importAudioFromDialog,
    importAudioFromPath,
    importMultipleAudioAtPosition,
    importMultipleAudioFilesAtPosition,
} from "./thunks/importThunks";

import {
    removeSelectedClipRemote,
    setTrackStateRemote,
} from "./thunks/trackThunks";
import { markProjectDirty } from "./sessionDirtyState";
import { resolveTrackIdForClipSelection } from "./selectionFocus";

export type {
    AutomationPoint,
    ClipInfo,
    ClipTemplate,
    DrawToolMode,
    DragDirection,
    DrawDragDirection,
    EditParam,
    FadeCurveType,
    GridSize,
    ToolMode,
    ToolModeGroup,
    TrackInfo,
};

type ClipColor = ClipInfo["color"];
type WaveformPreview = number[] | { l: number[]; r: number[] };

export interface SessionState {
    toolMode: ToolMode;
    toolModeGroup: ToolModeGroup;
    drawToolMode: DrawToolMode;
    editParam: EditParam;
    bpm: number;
    beats: number;
    projectSec: number;
    grid: GridSize;

    /** 自动交叉淡入淡出 */
    autoCrossfadeEnabled: boolean;
    /** 网格吸附 */
    gridSnapEnabled: boolean;
    /** 音高吸附 */
    pitchSnapEnabled: boolean;
    pitchSnapUnit: PitchSnapUnit;
    /** 音高吸附容差（分）用于微调吸附强度 */
    pitchSnapToleranceCents: number;
    /** 基准音阶键名，如 "C" "Db" 等 */
    pitchSnapScale: import("../../utils/musicalScales").ScaleKey;
    /** 音阶高亮模式：始终 / 关闭 */
    scaleHighlightMode: "always" | "off";
    /** 播放头缩放 */
    playheadZoomEnabled: boolean;
    /** 自动滚动（播放时跟随播放头） */
    autoScrollEnabled: boolean;
    /** 允许参数编辑器点击时调整播放头 */
    paramEditorSeekPlayheadEnabled: boolean;
    /** 剪贴板预览（在参数编辑器选区内显示剪贴板曲线预览） */
    showClipboardPreview: boolean;
    /** 参数线附近显示参数值浮窗 */
    showParamValuePopup: boolean;
    /** 参数编辑器（选择工具）拖动方向限制 */
    selectDragDirection: DragDirection;
    /** 参数编辑器（绘制工具）拖动方向限制 */
    drawDragDirection: DrawDragDirection;
    /** 参数编辑器（直线/颤音工具）拖动方向限制 */
    lineVibratoDragDirection: DrawDragDirection;

    /** 参数编辑器选区拖拽时的边缘平滑度（0-100%） */
    edgeSmoothnessPercent: number;

    /** 在粘贴/创建时是否锁定参数线以应用 linked params */
    lockParamLinesEnabled: boolean;

    // Monotonic bump token for invalidating parameter curve caches.
    // - Not included in undo/redo snapshots.
    // - Should be bumped on any timeline/undo/redo operation that may affect param rendering.
    paramsEpoch: number;

    playheadSec: number;
    tracks: TrackInfo[];
    clips: ClipInfo[];
    selectedTrackId: string | null;
    selectedClipId: string | null;
    /** 多选 clip 的 id 列表（框选 / Ctrl+点击） */
    multiSelectedClipIds: string[];
    clipAutomation: Record<string, Record<string, AutomationPoint[]>>;
    selectedPointId: string | null;
    clipWaveforms: Record<string, WaveformPreview>;
    clipPitchRanges: Record<string, { min: number; max: number }>;

    /**
     * 后端推送的 per-clip 音高检测结果（MIDI 曲线）。
     * key: clip_id
     * value: { curveStartSec, midiCurve, framePeriodMs }
     */
    clipPitchCurves: Record<
        string,
        {
            /** MIDI 曲线第 0 帧对应的 timeline 绝对时间（秒） */
            curveStartSec: number;
            midiCurve: number[];
            framePeriodMs: number;
        }
    >;
    modelDir: string;
    audioPath: string;
    outputPath: string;
    pitchShift: number;
    playbackClipId: string | null;
    playbackAnchorSec: number;

    runtime: {
        device: string;
        modelLoaded: boolean;
        audioLoaded: boolean;
        hasSynthesized: boolean;
        isPlaying: boolean;
        playbackTarget: string | null;
        playbackPositionSec: number;
        playbackDurationSec: number;
    };

    selectedTrackSummary: {
        trackId: string | null;
        clipCount: number;
        waveformPreview: number[];
        pitchRange: { min: number; max: number };
    };

    historyPast: StateSnapshot[];
    historyFuture: StateSnapshot[];
    customScalePresets: CustomScalePreset[];
    project: {
        name: string;
        path: string | null;
        dirty: boolean;
        recent: string[];
        baseScale: import("../../utils/musicalScales").ScaleKey;
        useCustomScale: boolean;
        customScale: CustomScalePreset | null;
        beatsPerBar: number;
        gridSize: GridSize;
    };

    busy: boolean;
    status: string;
    error?: string;
    lastResult?: unknown;
    vocalShifterSkippedFilesDialog: string[] | null;
    reaperSkippedFilesDialog: string[] | null;

    /**
     * 交互锁计数器：当用户正在进行连续操作（拖动、滑动等）时 > 0。
     * 在锁定期间，连续操作类 thunk 的 fulfilled handler 将跳过 applyTimelineState()，
     * 避免后端返回的过期快照覆盖前端乐观更新导致的闪烁。
     * 不包含在 undo/redo 快照中。
     */
    _interactionLockCount: number;
}

interface StateSnapshot {
    clips: ClipInfo[];
    clipAutomation: SessionState["clipAutomation"];
    selectedTrackId: string | null;
    selectedClipId: string | null;
    selectedPointId: string | null;
    playheadSec: number;
    clipWaveforms: Record<string, WaveformPreview>;
    clipPitchRanges: Record<string, { min: number; max: number }>;
}

function clamp(value: number, minValue: number, maxValue: number): number {
    return Math.min(maxValue, Math.max(minValue, value));
}

function createId(prefix: string): string {
    return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

function createDefaultAutomation() {
    return {
        pitch: [
            { id: createId("pt_p"), beat: 0, value: 0 },
            { id: createId("pt_p"), beat: 3, value: 1.5 },
            { id: createId("pt_p"), beat: 7, value: -0.8 },
            { id: createId("pt_p"), beat: 12, value: 0.3 },
        ],
        tension: [
            { id: createId("pt_t"), beat: 0, value: 0.2 },
            { id: createId("pt_t"), beat: 4, value: 0.72 },
            { id: createId("pt_t"), beat: 8, value: 0.42 },
            { id: createId("pt_t"), beat: 12, value: 0.6 },
        ],
    };
}

function basenameFromPath(path: string): string {
    return path.split(/[\\/]/).filter(Boolean).pop() ?? "Audio.wav";
}

function ensureClipAutomation(state: SessionState, clipId: string) {
    if (!state.clipAutomation[clipId]) {
        state.clipAutomation[clipId] = createDefaultAutomation();
    }
}

function createSnapshot(state: SessionState): StateSnapshot {
    return {
        clips: state.clips.map((clip) => ({ ...clip })),
        clipAutomation: structuredClone(current(state.clipAutomation)),
        selectedTrackId: state.selectedTrackId,
        selectedClipId: state.selectedClipId,
        selectedPointId: state.selectedPointId,
        playheadSec: state.playheadSec,
        clipWaveforms: { ...state.clipWaveforms },
        clipPitchRanges: { ...state.clipPitchRanges },
    };
}

function applySnapshot(state: SessionState, snapshot: StateSnapshot) {
    state.clips = snapshot.clips;
    state.clipAutomation = snapshot.clipAutomation;
    state.selectedTrackId = snapshot.selectedTrackId;
    state.selectedClipId = snapshot.selectedClipId;
    state.selectedPointId = snapshot.selectedPointId;
    state.playheadSec = snapshot.playheadSec;
    state.clipWaveforms = snapshot.clipWaveforms;
    state.clipPitchRanges = snapshot.clipPitchRanges;
}

function pushHistory(state: SessionState) {
    state.historyPast.push(createSnapshot(state));
    if (state.historyPast.length > 40) {
        state.historyPast.shift();
    }
    state.historyFuture = [];
    markProjectDirty(state.project);
}

function normalizeClipColor(color: string | undefined): ClipColor {
    if (color === "blue") return "blue";
    if (color === "violet") return "violet";
    if (color === "amber") return "amber";
    return "emerald";
}

/**
 * Auto-crossfade logic applied directly in a reducer (no dispatch needed).
 * For each clip in `movedIds`, detect overlaps with same-track clips and set
 * fade in/out to the overlap duration.
 */
function applyAutoCrossfadeInReducer(
    state: SessionState,
    movedIds: string[],
) {
    if (!state.autoCrossfadeEnabled || movedIds.length === 0) return;

    const trackClipsMap: Record<string, ClipInfo[]> = {};
    const clipMap: Record<string, ClipInfo> = {}; 
    
    // 一次性建立轨道分组和全局 ID 索引，时间复杂度 O(N)
    for (const clip of state.clips) {
        clipMap[clip.id] = clip;
        if (!trackClipsMap[clip.trackId]) trackClipsMap[clip.trackId] = [];
        trackClipsMap[clip.trackId].push(clip);
    }

    const fadeInOverlaps = new Map<string, number>();
    const fadeOutOverlaps = new Map<string, number>();

    for (const id of movedIds) {
        // O(1) 直接获取，消除多余的 find 遍历
        const clip = clipMap[id]; 
        if (!clip) continue;
        const clipStart = Number(clip.startSec);
        const clipEnd = clipStart + Number(clip.lengthSec);

        const sameTrack = (trackClipsMap[clip.trackId] || []).filter(
            (c) => c.id !== id,
        );

        for (const other of sameTrack) {
            const otherStart = Number(other.startSec);
            const otherEnd = otherStart + Number(other.lengthSec);
            const overlapStart = Math.max(clipStart, otherStart);
            const overlapEnd = Math.min(clipEnd, otherEnd);
            const overlap = overlapEnd - overlapStart;
            if (overlap <= 0.001) continue;

            if (clipStart <= otherStart) {
                fadeOutOverlaps.set(id, Math.max(fadeOutOverlaps.get(id) ?? 0, overlap));
                fadeInOverlaps.set(other.id, Math.max(fadeInOverlaps.get(other.id) ?? 0, overlap));
            } else {
                fadeInOverlaps.set(id, Math.max(fadeInOverlaps.get(id) ?? 0, overlap));
                fadeOutOverlaps.set(other.id, Math.max(fadeOutOverlaps.get(other.id) ?? 0, overlap));
            }
        }
    }

    const allClipIds = new Set([
        ...fadeInOverlaps.keys(),
        ...fadeOutOverlaps.keys(),
        ...movedIds,
    ]);
    
    for (const clipId of allClipIds) {
        // O(1) 直接获取，消除多余的 find 遍历
        const clip = clipMap[clipId]; 
        if (!clip) continue;

        const hasOverlapIn = fadeInOverlaps.has(clipId);
        const hasOverlapOut = fadeOutOverlaps.has(clipId);

        if (hasOverlapIn) {
            clip.fadeInSec = Math.max(0, fadeInOverlaps.get(clipId) ?? 0);
        }
        if (hasOverlapOut) {
            clip.fadeOutSec = Math.max(0, fadeOutOverlaps.get(clipId) ?? 0);
        }
    }
}

/**
 * 将后端返回的 TimelineState 全量覆写到前端 Redux state。
 *
 * @param force  默认 false。当 `_interactionLockCount > 0`（用户正在拖动/滑动等连续交互）
 *               且 force 为 false 时，函数直接 return，跳过全量覆写，避免过期后端快照覆盖
 *               前端乐观更新导致的 UI 闪烁。
 *               对于"权威操作"（open/new/import/undo/redo/save/fetchTimeline 等），
 *               应传 `force: true` 以确保状态一定被应用。
 */
function applyTimelineState(
    state: SessionState,
    timeline: TimelineState,
    opts?: { force?: boolean },
) {
    // 交互锁守卫：拖动/滑动期间跳过非强制的全量覆写
    if (state._interactionLockCount > 0 && !opts?.force) {
        return;
    }

    state.tracks = timeline.tracks.map((track) => ({
        id: track.id,
        name: track.name,
        parentId: track.parent_id ?? null,
        depth: track.depth ?? 0,
        childTrackIds: track.child_track_ids ?? [],
        muted: Boolean(track.muted),
        solo: Boolean(track.solo),
        volume: clamp(Number(track.volume ?? 0.9), 0, 1),

        composeEnabled: Boolean(track.compose_enabled),
        pitchAnalysisAlgo: String(
            track.pitch_analysis_algo ?? "nsf_hifigan_onnx",
        ),
        color: track.color || undefined,
    }));

    state.clips = timeline.clips.map((clip: TimelineClip) => {
        const parsed = {
            id: clip.id,
            trackId: clip.track_id,
            name: clip.name,
            startSec: Number(clip.start_sec ?? 0),
            lengthSec: Math.max(0.0, Number(clip.length_sec ?? 1)),
            color: normalizeClipColor(clip.color),
            sourcePath: clip.source_path,
            durationSec: Number(clip.duration_sec ?? 0) || undefined,
            durationFrames: clip.duration_frames,
            sourceSampleRate: clip.source_sample_rate,
            gain: clamp(Number(clip.gain ?? 1), 0, 4),
            muted: Boolean(clip.muted),
            // Allow negative sourceStartSec to represent leading silence (slip-edit past source start).
            sourceStartSec: Number(clip.source_start_sec ?? 0) || 0,
            sourceEndSec: (() => {
                const raw = Math.max(0, Number(clip.source_end_sec ?? 0));
                // 旧项目兼容：source_end_sec == 0 曾表示"到源文件末尾"，修正为实际时长
                if (raw === 0) {
                    return (
                        Number(clip.duration_sec ?? 0) ||
                        Math.max(0, Number(clip.length_sec ?? 1))
                    );
                }
                return raw;
            })(),
            playbackRate: clamp(Number(clip.playback_rate ?? 1), 0.1, 10),
            fadeInSec: Math.max(0, Number(clip.fade_in_sec ?? 0)),
            fadeOutSec: Math.max(0, Number(clip.fade_out_sec ?? 0)),
            fadeInCurve: (clip.fade_in_curve ?? "sine") as FadeCurveType,
            fadeOutCurve: (clip.fade_out_curve ?? "sine") as FadeCurveType,
        };

        return parsed;
    });

    state.selectedTrackId = timeline.selected_track_id;
    state.selectedClipId = timeline.selected_clip_id;
    state.bpm = clamp(Number(timeline.bpm ?? state.bpm), 10, 300);
    state.playheadSec = Math.max(0, Number(timeline.playhead_sec ?? 0));
    state.projectSec = Math.max(
        4,
        Number(timeline.project_sec ?? state.projectSec),
    );

    const project = (timeline as any).project as
        | {
              name?: string;
              path?: string | null;
              dirty?: boolean;
              recent?: string[];
              base_scale?: string;
              use_custom_scale?: boolean;
              custom_scale?: {
                  id?: string;
                  name?: string;
                  notes?: number[];
              } | null;
              beats_per_bar?: number;
              grid_size?: string;
          }
        | undefined;
    if (project) {
        const nextBaseScaleRaw = String(
            project.base_scale ?? state.project.baseScale,
        );
        const nextBaseScale = (SCALE_KEYS as readonly string[]).includes(
            nextBaseScaleRaw,
        )
            ? (nextBaseScaleRaw as typeof state.project.baseScale)
            : "C";
        const nextBeatsPerBar = clamp(
            Number(project.beats_per_bar ?? state.project.beatsPerBar),
            1,
            32,
        );
        const nextGridSizeRaw = String(
            project.grid_size ?? state.project.gridSize,
        );
        const nextGridSize = (
            [
                "1/1", "1/2", "1/4", "1/8", "1/16", "1/32", "1/64",
                "1/1d", "1/2d", "1/4d", "1/8d", "1/16d", "1/32d", "1/64d",
                "1/1t", "1/2t", "1/4t", "1/8t", "1/16t", "1/32t", "1/64t",
            ] as const
        ).includes(nextGridSizeRaw as any)
            ? (nextGridSizeRaw as GridSize)
            : "1/4";

        state.project = {
            name: String(project.name ?? state.project.name ?? "Untitled"),
            path:
                project.path === undefined
                    ? state.project.path
                    : ((project.path as any) ?? null),
            dirty: Boolean(project.dirty),
            recent: Array.isArray(project.recent)
                ? project.recent
                : state.project.recent,
            baseScale: nextBaseScale,
            useCustomScale: Boolean(project.use_custom_scale),
            customScale: project.custom_scale
                ? sanitizeCustomScalePreset(project.custom_scale)
                : null,
            beatsPerBar: nextBeatsPerBar,
            gridSize: nextGridSize,
        };
        state.beats = nextBeatsPerBar;
        state.grid = nextGridSize;
    }

    const availableClipIds = new Set(state.clips.map((clip) => clip.id));
    for (const clipId of Object.keys(state.clipAutomation)) {
        if (!availableClipIds.has(clipId)) {
            delete state.clipAutomation[clipId];
        }
    }
    // 清理已删除 clip 的音高曲线数据，避免 PianoRoll 残留已删除 clip 的 detectedPitchCurve
    for (const clipId of Object.keys(state.clipPitchCurves)) {
        if (!availableClipIds.has(clipId)) {
            delete state.clipPitchCurves[clipId];
        }
    }

    const nextWaveforms: Record<string, WaveformPreview> = {};
    const nextPitchRanges: Record<string, { min: number; max: number }> = {};
    for (const clip of timeline.clips) {
        const clipId = clip.id;
        nextWaveforms[clipId] = (clip.waveform_preview ??
            []) as WaveformPreview;
        nextPitchRanges[clipId] = clip.pitch_range ?? { min: -24, max: 24 };
        ensureClipAutomation(state, clipId);
    }
    state.clipWaveforms = nextWaveforms;
    state.clipPitchRanges = nextPitchRanges;

    // Any timeline refresh may change pitch analysis inputs and therefore param curves.
    state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
}

function upsertImportedClip(
    state: SessionState,
    audioPath: string,
    meta?: {
        durationSec?: number;
        waveform?: number[];
        pitchRange?: { min: number; max: number };
    },
) {
    const existing = state.clips.find((clip) => clip.sourcePath === audioPath);
    if (existing) {
        state.selectedClipId = existing.id;
        ensureClipAutomation(state, existing.id);
        if (meta?.waveform) {
            state.clipWaveforms[existing.id] = meta.waveform;
        }
        if (meta?.pitchRange) {
            state.clipPitchRanges[existing.id] = meta.pitchRange;
        }
        return;
    }

    const targetTrackId = state.tracks[0]?.id ?? "track_imported";
    if (!state.tracks[0]) {
        state.tracks.push({
            id: targetTrackId,
            name: "Imported",
            muted: false,
            solo: false,
            volume: 0.9,

            composeEnabled: false,
            pitchAnalysisAlgo: "nsf_hifigan_onnx",
        });
    }

    const maxEndSec = state.clips.reduce(
        (maxSec, clip) => Math.max(maxSec, clip.startSec + clip.lengthSec),
        0,
    );
    const startSec = Math.max(0, Math.ceil(maxEndSec));
    const newClipId = createId("clip");
    const lengthSec = Math.max(1, meta?.durationSec ?? 4);
    state.clips.push({
        id: newClipId,
        trackId: targetTrackId,
        name: basenameFromPath(audioPath),
        startSec,
        lengthSec,
        color: "emerald",
        sourcePath: audioPath,
        durationSec: meta?.durationSec,
        gain: 1,
        muted: false,
        sourceStartSec: 0,
        sourceEndSec: meta?.durationSec ?? lengthSec,
        playbackRate: 1,
        fadeInSec: 0,
        fadeOutSec: 0,
        fadeInCurve: "sine" as FadeCurveType,
        fadeOutCurve: "sine" as FadeCurveType,
    });
    state.selectedClipId = newClipId;
    state.playheadSec = startSec;
    state.selectedPointId = null;
    ensureClipAutomation(state, newClipId);
    state.clipWaveforms[newClipId] = meta?.waveform ?? [];
    state.clipPitchRanges[newClipId] = meta?.pitchRange ?? {
        min: -24,
        max: 24,
    };
    // 导入后自动扩展工程边界
    const clipEnd = startSec + lengthSec;
    if (clipEnd > state.projectSec) {
        state.projectSec = Math.ceil(clipEnd);
    }
}

const initialState: SessionState = {
    toolMode: "draw",
    toolModeGroup: "draw",
    drawToolMode: "draw",
    editParam: "pitch",
    bpm: 120,
    beats: 4,
    projectSec: 30, // 默认 30 秒工程边界
    grid: "1/4",

    autoCrossfadeEnabled: true,
    gridSnapEnabled: true,
    pitchSnapEnabled: false,
    pitchSnapUnit: "semitone",
    pitchSnapScale: "C",
    pitchSnapToleranceCents: 0,
    scaleHighlightMode: "always",
    playheadZoomEnabled: false,
    autoScrollEnabled: false,
    paramEditorSeekPlayheadEnabled: true,
    showClipboardPreview: true,
    showParamValuePopup: true,
    selectDragDirection: "y-only" as DragDirection,
    drawDragDirection: "free" as DrawDragDirection,
    lineVibratoDragDirection: "free" as DrawDragDirection,
    edgeSmoothnessPercent: 0,
    lockParamLinesEnabled: false,

    paramsEpoch: 0,

    playheadSec: 0,
    tracks: [
        {
            id: "track_main",
            name: "Main",
            muted: false,
            solo: false,
            volume: 0.9,

            composeEnabled: false,
            pitchAnalysisAlgo: "nsf_hifigan_onnx",
        },
    ],
    clips: [],
    selectedTrackId: "track_main",
    selectedClipId: null,
    multiSelectedClipIds: [],
    clipAutomation: {},
    selectedPointId: null,
    clipWaveforms: {},
    clipPitchRanges: {},
    clipPitchCurves: {},

    modelDir: "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02",
    audioPath: "",
    outputPath: "outputs/webview_synth.wav",
    pitchShift: 0,
    playbackClipId: null,
    playbackAnchorSec: 0,

    runtime: {
        device: "unknown",
        modelLoaded: false,
        audioLoaded: false,
        hasSynthesized: false,
        isPlaying: false,
        playbackTarget: null,
        playbackPositionSec: 0,
        playbackDurationSec: 0,
    },

    selectedTrackSummary: {
        trackId: null,
        clipCount: 0,
        waveformPreview: [],
        pitchRange: { min: -24, max: 24 },
    },

    historyPast: [],
    historyFuture: [],
    customScalePresets: [],
    project: {
        name: "Untitled",
        path: null,
        dirty: false,
        recent: [],
        baseScale: "C",
        useCustomScale: false,
        customScale: null,
        beatsPerBar: 4,
        gridSize: "1/4",
    },

    busy: false,
    status: "Ready",
    vocalShifterSkippedFilesDialog: null,
    reaperSkippedFilesDialog: null,
    _interactionLockCount: 0,
};

export {
    undoRemote,
    redoRemote,
    newProjectRemote,
    openProjectFromDialog,
    openProjectFromPath,
    openVocalShifterFromDialog,
    openVocalShifterFromPath,
    openReaperFromDialog,
    openReaperFromPath,
    saveProjectRemote,
    saveProjectAsRemote,
    setProjectBaseScaleRemote,
    setProjectCustomScaleRemote,
    setProjectTimelineSettingsRemote,
} from "./thunks/projectThunks";

export {
    fetchTimeline,
    seekPlayhead,
    updateTransportBpm,
    syncPlaybackState,
    playOriginal,
    stopAudioPlayback,
} from "./thunks/transportThunks";

export {
    addTrackRemote,
    removeTrackRemote,
    moveTrackRemote,
    selectTrackRemote,
    setProjectLengthRemote,
    fetchSelectedTrackSummary,
    addClipOnTrack,
    createClipsRemote,
    removeClipRemote,
    removeClipsRemote,
    moveClipRemote,
    moveClipsRemote,
    duplicateTrackRemote,
    setClipStateRemote,
    replaceClipSourceRemote,
    splitClipRemote,
    glueClipsRemote,
    selectClipRemote,
} from "./thunks/timelineThunks";

export {
    setTrackStateRemote,
    removeSelectedClipRemote,
} from "./thunks/trackThunks";

export {
    refreshRuntime,
    clearWaveformCacheRemote,
    loadUiSettings,
    persistUiSettings,
} from "./thunks/runtimeThunks";

export { loadModel, loadDefaultModel } from "./thunks/modelThunks";

export {
    processAudio,
    pickOutputPath,
    applyPitchShift,
    synthesizeAudio,
    exportAudio,
    exportSeparated,
    pasteVocalShifterClipboard,
    pasteReaperClipboard,
} from "./thunks/audioThunks";

export {
    importAudioFromDialog,
    importAudioFromPath,
    importAudioAtPosition,
    importAudioFileAtPosition,
    importMultipleAudioAtPosition,
    importMultipleAudioFilesAtPosition,
} from "./thunks/importThunks";

const sessionSlice = createSlice({
    name: "session",
    initialState,
    reducers: {
        /**
         * 标记连续交互开始（拖动/滑动等）。
         * 在交互期间，连续操作类 thunk 的 fulfilled handler 会跳过 applyTimelineState()。
         */
        beginInteraction(state) {
            state._interactionLockCount = Math.max(0, state._interactionLockCount) + 1;
        },
        /**
         * 标记连续交互结束。计数器归零后恢复正常的后端状态同步。
         */
        endInteraction(state) {
            state._interactionLockCount = Math.max(0, state._interactionLockCount - 1);
        },
        /** 乐观更新轨道名称（立即反映到 UI，不等后端响应） */
        setTrackName(
            state,
            action: PayloadAction<{ trackId: string; name: string }>,
        ) {
            const track = state.tracks.find(
                (entry) => entry.id === action.payload.trackId,
            );
            if (track) {
                track.name = action.payload.name;
            }
        },
        checkpointHistory(state) {
            pushHistory(state);
            state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
        },
        setToolMode(state, action: PayloadAction<ToolMode>) {
            if (state.toolMode !== action.payload) {
                pushHistory(state);
            }
            state.toolMode = action.payload;
            if (action.payload === "select") {
                state.toolModeGroup = "select";
            } else {
                state.toolModeGroup = "draw";
                state.drawToolMode = action.payload;
            }
        },
        setEditParam(state, action: PayloadAction<EditParam>) {
            state.editParam = action.payload;
            state.selectedPointId = null;
        },
        setBpm(state, action: PayloadAction<number>) {
            state.bpm = clamp(action.payload, 10, 300);
        },
        setBeats(state, action: PayloadAction<number>) {
            state.beats = clamp(action.payload, 1, 32);
        },
        setGrid(state, action: PayloadAction<GridSize>) {
            state.grid = action.payload;
        },
        toggleAutoCrossfade(state) {
            state.autoCrossfadeEnabled = !state.autoCrossfadeEnabled;
        },
        toggleGridSnap(state) {
            state.gridSnapEnabled = !state.gridSnapEnabled;
        },
        togglePitchSnap(state) {
            state.pitchSnapEnabled = !state.pitchSnapEnabled;
        },
        setPitchSnapUnit(state, action: PayloadAction<PitchSnapUnit>) {
            state.pitchSnapUnit = action.payload;
        },
        setPitchSnapScale(
            state,
            action: PayloadAction<import("../../utils/musicalScales").ScaleKey>,
        ) {
            state.pitchSnapScale = action.payload;
        },
        setPitchSnapToleranceCents(state, action: PayloadAction<number>) {
            state.pitchSnapToleranceCents = clamp(action.payload, 0, 1000);
        },
        setScaleHighlightMode(
            state,
            action: PayloadAction<"always" | "off">,
        ) {
            state.scaleHighlightMode = action.payload;
        },
        upsertCustomScalePreset(
            state,
            action: PayloadAction<CustomScalePreset>,
        ) {
            const incoming = sanitizeCustomScalePreset(action.payload);
            const idx = state.customScalePresets.findIndex(
                (preset) => preset.id === incoming.id,
            );
            if (idx >= 0) {
                state.customScalePresets[idx] = incoming;
            } else {
                state.customScalePresets.push(incoming);
            }
        },
        togglePlayheadZoom(state) {
            state.playheadZoomEnabled = !state.playheadZoomEnabled;
        },
        toggleLockParamLines(state) {
            state.lockParamLinesEnabled = !state.lockParamLinesEnabled;
        },
        toggleAutoScroll(state) {
            state.autoScrollEnabled = !state.autoScrollEnabled;
        },
        toggleParamEditorSeekPlayhead(state) {
            state.paramEditorSeekPlayheadEnabled =
                !state.paramEditorSeekPlayheadEnabled;
        },
        toggleClipboardPreview(state) {
            state.showClipboardPreview = !state.showClipboardPreview;
        },
        toggleParamValuePopup(state) {
            state.showParamValuePopup = !state.showParamValuePopup;
        },
        cycleDragDirection(
            state,
            action: PayloadAction<"select" | "draw" | "vibrato">,
        ) {
            if (action.payload === "select") {
                const order: DragDirection[] = ["free", "x-only", "y-only"];
                const idx = order.indexOf(state.selectDragDirection);
                state.selectDragDirection = order[(idx + 1) % order.length];
                return;
            }
            const order: DrawDragDirection[] = ["free", "x-only"];
            if (action.payload === "draw") {
                const idx = order.indexOf(state.drawDragDirection);
                state.drawDragDirection = order[(idx + 1) % order.length];
                return;
            }
            const idx = order.indexOf(state.lineVibratoDragDirection);
            state.lineVibratoDragDirection = order[(idx + 1) % order.length];
        },
        setDragDirection(
            state,
            action: PayloadAction<{
                tool: "select" | "draw" | "vibrato";
                direction: DragDirection | DrawDragDirection;
            }>,
        ) {
            const { tool, direction } = action.payload;
            if (tool === "select") {
                if (["free", "x-only", "y-only"].includes(direction)) {
                    state.selectDragDirection = direction as DragDirection;
                }
                return;
            }
            if (tool === "draw") {
                if (["free", "x-only"].includes(direction)) {
                    state.drawDragDirection = direction as DrawDragDirection;
                }
                return;
            }
            if (["free", "x-only"].includes(direction)) {
                state.lineVibratoDragDirection = direction as DrawDragDirection;
            }
        },
        setEdgeSmoothnessPercent(state, action: PayloadAction<number>) {
            state.edgeSmoothnessPercent = clamp(Number(action.payload) || 0, 0, 100);
        },
        setplayheadSec(state, action: PayloadAction<number>) {
            state.playheadSec = Math.max(0, action.payload);
        },
        setModelDir(state, action: PayloadAction<string>) {
            state.modelDir = action.payload;
        },
        setAudioPath(state, action: PayloadAction<string>) {
            state.audioPath = action.payload;
        },
        setOutputPath(state, action: PayloadAction<string>) {
            state.outputPath = action.payload;
        },
        setPitchShift(state, action: PayloadAction<number>) {
            state.pitchShift = action.payload;
        },
        closeVocalShifterSkippedFilesDialog(state) {
            state.vocalShifterSkippedFilesDialog = null;
        },
        closeReaperSkippedFilesDialog(state) {
            state.reaperSkippedFilesDialog = null;
        },
        setSelectedClip(state, action: PayloadAction<string | null>) {
            state.selectedClipId = action.payload;
            state.selectedPointId = null;
            if (action.payload) {
                state.selectedTrackId = resolveTrackIdForClipSelection({
                    currentTrackId: state.selectedTrackId,
                    clips: state.clips,
                    clipId: action.payload,
                });
                ensureClipAutomation(state, action.payload);
            }
        },
        setSelectedClipPreservingTrack(state, action: PayloadAction<string | null>) {
            state.selectedClipId = action.payload;
            state.selectedPointId = null;
            if (action.payload) {
                ensureClipAutomation(state, action.payload);
            }
        },
        setMultiSelectedClipIds(state, action: PayloadAction<string[]>) {
            state.multiSelectedClipIds = action.payload;
        },
        moveClipStart(
            state,
            action: PayloadAction<{ clipId: string; startSec: number }>,
        ) {
            const clip = state.clips.find(
                (entry) => entry.id === action.payload.clipId,
            );
            if (clip) {
                clip.startSec = Math.max(0, action.payload.startSec);
                // 拖动超出边界时自动扩展工程时长
                const clipEnd = clip.startSec + clip.lengthSec;
                if (clipEnd > state.projectSec) {
                    state.projectSec = Math.ceil(clipEnd);
                }
            }
        },
        moveClipTrack(
            state,
            action: PayloadAction<{ clipId: string; trackId: string }>,
        ) {
            const clip = state.clips.find(
                (entry) => entry.id === action.payload.clipId,
            );
            if (clip) {
                clip.trackId = action.payload.trackId;
            }
        },
        setClipLength(
            state,
            action: PayloadAction<{ clipId: string; lengthSec: number }>,
        ) {
            const clip = state.clips.find(
                (entry) => entry.id === action.payload.clipId,
            );
            if (clip) {
                clip.lengthSec = Math.max(0.0, action.payload.lengthSec);
            }
        },
        setClipPlaybackRate(
            state,
            action: PayloadAction<{ clipId: string; playbackRate: number }>,
        ) {
            const clip = state.clips.find(
                (entry) => entry.id === action.payload.clipId,
            );
            if (!clip) return;
            clip.playbackRate = clamp(action.payload.playbackRate, 0.1, 10);
        },
        setClipSourceRange(
            state,
            action: PayloadAction<{
                clipId: string;
                sourceStartSec?: number;
                sourceEndSec?: number;
            }>,
        ) {
            const clip = state.clips.find(
                (entry) => entry.id === action.payload.clipId,
            );
            if (!clip) return;
            if (action.payload.sourceStartSec !== undefined) {
                clip.sourceStartSec =
                    Number(action.payload.sourceStartSec) || 0;
            }
            if (action.payload.sourceEndSec !== undefined) {
                clip.sourceEndSec = Math.max(0, action.payload.sourceEndSec);
            }
        },
        setClipFades(
            state,
            action: PayloadAction<{
                clipId: string;
                fadeInSec?: number;
                fadeOutSec?: number;
                fadeInCurve?: FadeCurveType;
                fadeOutCurve?: FadeCurveType;
            }>,
        ) {
            const clip = state.clips.find(
                (entry) => entry.id === action.payload.clipId,
            );
            if (!clip) return;
            if (action.payload.fadeInSec !== undefined) {
                clip.fadeInSec = Math.max(0, action.payload.fadeInSec);
            }
            if (action.payload.fadeOutSec !== undefined) {
                clip.fadeOutSec = Math.max(0, action.payload.fadeOutSec);
            }
            if (action.payload.fadeInCurve !== undefined) {
                clip.fadeInCurve = action.payload.fadeInCurve;
            }
            if (action.payload.fadeOutCurve !== undefined) {
                clip.fadeOutCurve = action.payload.fadeOutCurve;
            }
        },
        setClipGain(
            state,
            action: PayloadAction<{ clipId: string; gain: number }>,
        ) {
            const clip = state.clips.find(
                (entry) => entry.id === action.payload.clipId,
            );
            if (!clip) return;
            clip.gain = clamp(Number(action.payload.gain), 0, 4);
        },
        setClipMuted(
            state,
            action: PayloadAction<{ clipId: string; muted: boolean }>,
        ) {
            const clip = state.clips.find(
                (entry) => entry.id === action.payload.clipId,
            );
            if (!clip) return;
            clip.muted = Boolean(action.payload.muted);
        },
        /** 乐观更新 clip 颜色（立即反映到 UI，后端确认前先行生效�?*/
        optimisticUpdateClipColor(
            state,
            action: PayloadAction<{ clipId: string; color: ClipColor }>,
        ) {
            const clip = state.clips.find(
                (entry) => entry.id === action.payload.clipId,
            );
            if (!clip) return;
            clip.color = normalizeClipColor(action.payload.color);
        },
        /** 回滚 clip 颜色（后端失败时恢复到旧值） */
        rollbackClipColor(
            state,
            action: PayloadAction<{ clipId: string; color: ClipColor }>,
        ) {
            const clip = state.clips.find(
                (entry) => entry.id === action.payload.clipId,
            );
            if (!clip) return;
            clip.color = normalizeClipColor(action.payload.color);
        },
        addClip(state, action: PayloadAction<{ trackId: string }>) {
            pushHistory(state);
            const newClipId = createId("clip");
            state.clips.push({
                id: newClipId,
                trackId: action.payload.trackId,
                name: "New Clip.wav",
                startSec: Math.max(0, state.playheadSec),
                lengthSec: 2,
                color: "emerald",
                gain: 1,
                muted: false,
                sourceStartSec: 0,
                sourceEndSec: 2,
                playbackRate: 1,
                fadeInSec: 0,
                fadeOutSec: 0,
                fadeInCurve: "sine" as FadeCurveType,
                fadeOutCurve: "sine" as FadeCurveType,
            });
            state.selectedClipId = newClipId;
            state.selectedTrackId = action.payload.trackId;
            ensureClipAutomation(state, newClipId);
            state.clipWaveforms[newClipId] = [];
            state.clipPitchRanges[newClipId] = { min: -24, max: 24 };
        },
        removeSelectedClip(state) {
            const selectedId = state.selectedClipId;
            if (!selectedId) {
                return;
            }
            pushHistory(state);
            state.clips = state.clips.filter((clip) => clip.id !== selectedId);
            delete state.clipAutomation[selectedId];
            delete state.clipWaveforms[selectedId];
            delete state.clipPitchRanges[selectedId];
            delete state.clipPitchCurves[selectedId];
            state.selectedPointId = null;
            state.selectedClipId = state.clips[0]?.id ?? null;
            if (state.selectedClipId) {
                ensureClipAutomation(state, state.selectedClipId);
            }
        },
        toggleTrackMute(state, action: PayloadAction<string>) {
            const track = state.tracks.find(
                (entry) => entry.id === action.payload,
            );
            if (track) {
                track.muted = !track.muted;
            }
        },
        toggleTrackSolo(state, action: PayloadAction<string>) {
            const track = state.tracks.find(
                (entry) => entry.id === action.payload,
            );
            if (track) {
                track.solo = !track.solo;
            }
        },
        setTrackVolume(
            state,
            action: PayloadAction<{ trackId: string; volume: number }>,
        ) {
            const track = state.tracks.find(
                (entry) => entry.id === action.payload.trackId,
            );
            if (track) {
                track.volume = clamp(action.payload.volume, 0, 1);
            }
        },
        addAutomationPoint(
            state,
            action: PayloadAction<{
                param: EditParam;
                beat: number;
                value: number;
            }>,
        ) {
            const clipId = state.selectedClipId;
            if (!clipId) {
                return;
            }
            pushHistory(state);
            ensureClipAutomation(state, clipId);
            const target = state.clipAutomation[clipId][action.payload.param];
            target.push({
                id: createId("pt"),
                beat: Math.max(0, action.payload.beat),
                value: action.payload.value,
            });
            target.sort((left, right) => left.beat - right.beat);
        },
        moveAutomationPoint(
            state,
            action: PayloadAction<{
                param: EditParam;
                pointId: string;
                beat: number;
                value: number;
            }>,
        ) {
            const clipId = state.selectedClipId;
            if (!clipId) {
                return;
            }
            pushHistory(state);
            ensureClipAutomation(state, clipId);
            const target = state.clipAutomation[clipId][action.payload.param];
            const point = target.find(
                (entry) => entry.id === action.payload.pointId,
            );
            if (point) {
                point.beat = Math.max(0, action.payload.beat);
                point.value = action.payload.value;
                target.sort((left, right) => left.beat - right.beat);
            }
        },
        setSelectedPoint(state, action: PayloadAction<string | null>) {
            state.selectedPointId = action.payload;
        },
        removeAutomationPoint(
            state,
            action: PayloadAction<{ param: EditParam; pointId: string }>,
        ) {
            const clipId = state.selectedClipId;
            if (!clipId) {
                return;
            }
            pushHistory(state);
            ensureClipAutomation(state, clipId);
            const target = state.clipAutomation[clipId][action.payload.param];
            state.clipAutomation[clipId][action.payload.param] = target.filter(
                (entry) => entry.id !== action.payload.pointId,
            );
            if (state.selectedPointId === action.payload.pointId) {
                state.selectedPointId = null;
            }
        },
        /** 更新某个 clip 的音高曲线（来自后端 clip_pitch_data 事件�?*/
        setClipPitchData(
            state,
            action: PayloadAction<{
                clipId: string;
                curveStartSec: number;
                midiCurve: number[];
                framePeriodMs: number;
            }>,
        ) {
            const { clipId, curveStartSec, midiCurve, framePeriodMs } =
                action.payload;
            state.clipPitchCurves[clipId] = {
                curveStartSec,
                midiCurve,
                framePeriodMs,
            };
            // 同步触发轨道总体音高线刷新
            state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
        },
        /** 移除某个 clip 的音高曲线（clip 被删除时清理�?*/
        removeClipPitchData(state, action: PayloadAction<string>) {
            delete state.clipPitchCurves[action.payload];
        },
        undo(state) {
            const snapshot = state.historyPast.pop();
            if (!snapshot) {
                return;
            }
            state.historyFuture.push(createSnapshot(state));
            applySnapshot(state, snapshot);
            state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
        },
        redo(state) {
            const snapshot = state.historyFuture.pop();
            if (!snapshot) {
                return;
            }
            state.historyPast.push(createSnapshot(state));
            applySnapshot(state, snapshot);
            state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
        },
    },
    extraReducers: (builder) => {
        const setPending = (state: SessionState, label: string) => {
            state.busy = true;
            state.status = label;
            state.error = undefined;
        };
        const setRejected = (
            state: SessionState,
            action: { error?: { message?: string } },
        ) => {
            state.busy = false;
            state.error = action.error?.message ?? "Request failed";
            state.status = "Failed";
        };

        builder
            .addCase(refreshRuntime.pending, (state) =>
                setPending(state, "Refreshing runtime..."),
            )
            .addCase(refreshRuntime.fulfilled, (state, action) => {
                state.busy = false;
                if ((action.payload as { ok?: boolean }).ok) {
                    const payload = action.payload as {
                        device: string;
                        model_loaded: boolean;
                        audio_loaded: boolean;
                        has_synthesized: boolean;
                        is_playing?: boolean;
                        playback_target?: string | null;
                        timeline?: TimelineState;
                    };
                    state.runtime = {
                        device: payload.device,
                        modelLoaded: payload.model_loaded,
                        audioLoaded: payload.audio_loaded,
                        hasSynthesized: payload.has_synthesized,
                        isPlaying: payload.is_playing ?? false,
                        playbackTarget: payload.playback_target ?? null,
                        playbackPositionSec: state.runtime.playbackPositionSec,
                        playbackDurationSec: state.runtime.playbackDurationSec,
                    };
                    if (payload.timeline) {
                        applyTimelineState(state, payload.timeline, { force: true });
                    }
                    state.status = "Runtime updated";
                } else {
                    state.status = "Runtime update failed";
                }
                state.lastResult = action.payload;
            })
            .addCase(refreshRuntime.rejected, setRejected)

            .addCase(clearWaveformCacheRemote.pending, (state) =>
                setPending(state, "Clearing waveform cache..."),
            )
            .addCase(clearWaveformCacheRemote.fulfilled, (state, action) => {
                state.busy = false;
                const payload = action.payload as {
                    ok?: boolean;
                    removed_files?: number;
                    removed_bytes?: number;
                    dir?: string;
                };
                if (payload.ok) {
                    const n = Number(payload.removed_files ?? 0) || 0;
                    state.status = `Waveform cache cleared (${n} files)`;
                } else {
                    state.status = "Clear waveform cache failed";
                }
            })
            .addCase(clearWaveformCacheRemote.rejected, setRejected)

            .addCase(loadUiSettings.fulfilled, (state, action) => {
                const s = action.payload;
                state.autoCrossfadeEnabled = s.autoCrossfade;
                state.gridSnapEnabled = s.gridSnap;
                state.pitchSnapEnabled = s.pitchSnap;
                // Validate pitchSnapUnit
                const validUnits: PitchSnapUnit[] = ["semitone", "scale"];
                state.pitchSnapUnit = validUnits.includes(
                    s.pitchSnapUnit as PitchSnapUnit,
                )
                    ? (s.pitchSnapUnit as PitchSnapUnit)
                    : "semitone";
                // Validate pitchSnapScale
                state.pitchSnapScale = (
                    SCALE_KEYS as readonly string[]
                ).includes((s as any).pitchSnapScale)
                    ? ((s as any).pitchSnapScale as typeof state.pitchSnapScale)
                    : "C";
                // Load pitch snap tolerance (cents) if present in saved settings
                if ((s as any).pitchSnapToleranceCents != null) {
                    state.pitchSnapToleranceCents = clamp(
                        Number((s as any).pitchSnapToleranceCents) || 0,
                        0,
                        1000,
                    );
                }
                state.playheadZoomEnabled = s.playheadZoom;
                if (s.autoScroll != null)
                    state.autoScrollEnabled = s.autoScroll;
                if ((s as any).paramEditorSeekPlayhead != null)
                    state.paramEditorSeekPlayheadEnabled = Boolean(
                        (s as any).paramEditorSeekPlayhead,
                    );
                if (s.showClipboardPreview != null)
                    state.showClipboardPreview = s.showClipboardPreview;
                if ((s as any).showParamValuePopup != null)
                    state.showParamValuePopup = Boolean(
                        (s as any).showParamValuePopup,
                    );
                if (s.scaleHighlightMode != null)
                    state.scaleHighlightMode = s.scaleHighlightMode === "always" ? "always" : "off";
                if (s.lockParamLines != null)
                    state.lockParamLinesEnabled = Boolean(
                        s.lockParamLines,
                    );
                const selectDir = (s as any).selectDragDirection;
                if (selectDir != null && ["free", "x-only", "y-only"].includes(selectDir)) {
                    state.selectDragDirection = selectDir as DragDirection;
                }
                const drawDir = (s as any).drawDragDirection;
                if (drawDir != null && ["free", "x-only"].includes(drawDir)) {
                    state.drawDragDirection = drawDir as DrawDragDirection;
                }
                const lineVibratoDir = (s as any).lineVibratoDragDirection;
                if (lineVibratoDir != null && ["free", "x-only"].includes(lineVibratoDir)) {
                    state.lineVibratoDragDirection = lineVibratoDir as DrawDragDirection;
                }
                const smoothness =
                    (s as any).smoothnessPercent ??
                    (s as any).edgeSmoothnessPercent;
                if (smoothness != null) {
                    state.edgeSmoothnessPercent = clamp(
                        Number(smoothness) || 0,
                        0,
                        100,
                    );
                }
                if (Array.isArray((s as any).customScalePresets)) {
                    state.customScalePresets = (s as any).customScalePresets
                        .map((preset: unknown) =>
                            sanitizeCustomScalePreset(preset as Partial<CustomScalePreset>),
                        );
                }
            })

            .addCase(loadDefaultModel.pending, (state) =>
                setPending(state, "Loading default model..."),
            )
            .addCase(loadDefaultModel.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                state.status = (action.payload as { ok?: boolean }).ok
                    ? "Default model loaded"
                    : "Load default model failed";
            })
            .addCase(loadDefaultModel.rejected, setRejected)

            .addCase(loadModel.pending, (state) =>
                setPending(state, "Loading model..."),
            )
            .addCase(loadModel.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                state.status = (action.payload as { ok?: boolean }).ok
                    ? "Model loaded"
                    : "Load model failed";
            })
            .addCase(loadModel.rejected, setRejected)

            .addCase(processAudio.pending, (state) =>
                setPending(state, "Processing audio..."),
            )
            .addCase(processAudio.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                const payload = action.payload as {
                    ok?: boolean;
                    audio?: { path?: string; duration_sec?: number };
                    feature?: {
                        waveform_preview?: number[];
                        pitch_range?: { min: number; max: number };
                    };
                    timeline?: TimelineState;
                };
                if (payload.ok && payload.audio?.path) {
                    state.audioPath = payload.audio.path;
                    if (payload.timeline) {
                        applyTimelineState(state, payload.timeline, { force: true });
                    } else {
                        upsertImportedClip(state, payload.audio.path, {
                            durationSec: payload.audio.duration_sec,
                            waveform: payload.feature?.waveform_preview,
                            pitchRange: payload.feature?.pitch_range,
                        });
                    }
                }
                state.status = payload.ok
                    ? "Audio processed"
                    : "Process audio failed";
            })
            .addCase(processAudio.rejected, setRejected)

            .addCase(importAudioFromDialog.pending, (state) =>
                setPending(state, "Importing audio..."),
            )
            .addCase(importAudioFromDialog.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                const payload = action.payload as {
                    canceled?: boolean;
                    path?: string;
                    imported?: { ok?: boolean } & TimelineState;
                };
                if (payload.canceled) {
                    state.status = "Import canceled";
                    return;
                }
                if (payload.path) {
                    state.audioPath = payload.path;
                    if (payload.imported?.ok) {
                        applyTimelineState(state, payload.imported, { force: true });
                    }
                }
                state.status = payload.imported?.ok
                    ? "Audio imported"
                    : "Import audio failed";
            })
            .addCase(importAudioFromDialog.rejected, setRejected)

            .addCase(importAudioFromPath.pending, (state) =>
                setPending(state, "Importing dropped audio..."),
            )
            .addCase(importAudioFromPath.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                const payload = action.payload as {
                    path?: string;
                    imported?: { ok?: boolean } & TimelineState;
                };
                if (payload.path) {
                    state.audioPath = payload.path;
                    if (payload.imported?.ok) {
                        applyTimelineState(state, payload.imported, { force: true });
                    }
                }
                state.status = payload.imported?.ok
                    ? "Dropped audio imported"
                    : "Import audio failed";
            })
            .addCase(importAudioFromPath.rejected, setRejected)

            .addCase(importAudioAtPosition.pending, (state) =>
                setPending(state, "Importing audio..."),
            )
            .addCase(importAudioAtPosition.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                const payload = action.payload as {
                    ok?: boolean;
                    imported?: TimelineState;
                    newClipIds?: string[];
                };
                const ok = Boolean(payload.ok);
                state.status = ok ? "Import done" : "Import failed";
                if (
                    ok &&
                    payload.imported &&
                    (payload.imported as any).tracks
                ) {
                    applyTimelineState(state, payload.imported as any, { force: true });
                    // Apply auto-crossfade for newly imported clips
                    if (payload.newClipIds && payload.newClipIds.length > 0) {
                        applyAutoCrossfadeInReducer(state, payload.newClipIds);
                    }
                }
            })
            .addCase(importAudioAtPosition.rejected, setRejected)

            .addCase(importAudioFileAtPosition.pending, (state) =>
                setPending(state, "Importing audio..."),
            )
            .addCase(importAudioFileAtPosition.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                const payload = action.payload as {
                    ok?: boolean;
                    imported?: TimelineState;
                    newClipIds?: string[];
                };
                const ok = Boolean(payload.ok);
                state.status = ok ? "Import done" : "Import failed";
                if (
                    ok &&
                    payload.imported &&
                    (payload.imported as any).tracks
                ) {
                    applyTimelineState(state, payload.imported as any, { force: true });
                    if (payload.newClipIds && payload.newClipIds.length > 0) {
                        applyAutoCrossfadeInReducer(state, payload.newClipIds);
                    }
                }
            })
            .addCase(importAudioFileAtPosition.rejected, setRejected)

            .addCase(importMultipleAudioAtPosition.pending, (state) =>
                setPending(state, "Importing multiple audio files..."),
            )
            .addCase(importMultipleAudioAtPosition.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                const payload = action.payload as {
                    ok?: boolean;
                    imported?: TimelineState;
                    newClipIds?: string[];
                };
                const ok = Boolean(payload.ok);
                state.status = ok ? "Import done" : "Import failed";
                if (ok && payload.imported && (payload.imported as any).tracks) {
                    applyTimelineState(state, payload.imported as any, { force: true });
                    if (payload.newClipIds && payload.newClipIds.length > 0) {
                        applyAutoCrossfadeInReducer(state, payload.newClipIds);
                    }
                }
            })
            .addCase(importMultipleAudioAtPosition.rejected, setRejected)

            .addCase(importMultipleAudioFilesAtPosition.pending, (state) =>
                setPending(state, "Importing multiple audio files..."),
            )
            .addCase(importMultipleAudioFilesAtPosition.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                const payload = action.payload as {
                    ok?: boolean;
                    imported?: TimelineState;
                    newClipIds?: string[];
                };
                const ok = Boolean(payload.ok);
                state.status = ok ? "Import done" : "Import failed";
                if (ok && payload.imported && (payload.imported as any).tracks) {
                    applyTimelineState(state, payload.imported as any, { force: true });
                    if (payload.newClipIds && payload.newClipIds.length > 0) {
                        applyAutoCrossfadeInReducer(state, payload.newClipIds);
                    }
                    // select all imported clips
                    if (payload.newClipIds && payload.newClipIds.length > 0) {
                        state.multiSelectedClipIds = payload.newClipIds;
                        state.selectedClipId = payload.newClipIds[0] ?? null;
                    }
                }
            })
            .addCase(importMultipleAudioFilesAtPosition.rejected, setRejected)

            .addCase(pickOutputPath.pending, (state) =>
                setPending(state, "Selecting output path..."),
            )
            .addCase(pickOutputPath.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                const payload = action.payload as {
                    canceled?: boolean;
                    path?: string;
                };
                if (payload.canceled) {
                    state.status = "Pick output canceled";
                    return;
                }
                if (payload.path) {
                    state.outputPath = payload.path;
                    state.status = "Output path selected";
                }
            })
            .addCase(pickOutputPath.rejected, setRejected)

            .addCase(applyPitchShift.pending, (state) =>
                setPending(state, "Applying pitch shift..."),
            )
            .addCase(applyPitchShift.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                state.status = (action.payload as { ok?: boolean }).ok
                    ? "Pitch shift applied"
                    : "Pitch shift failed";
            })
            .addCase(applyPitchShift.rejected, setRejected)

            .addCase(synthesizeAudio.pending, (state) =>
                setPending(state, "Synthesizing..."),
            )
            .addCase(synthesizeAudio.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                state.status = (action.payload as { ok?: boolean }).ok
                    ? "Synthesis done"
                    : "Synthesis failed";
            })
            .addCase(synthesizeAudio.rejected, setRejected)

            .addCase(exportAudio.pending, (state) =>
                setPending(state, "Exporting WAV..."),
            )
            .addCase(exportAudio.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                const payload = action.payload as {
                    ok?: boolean;
                    path?: string;
                };
                if (payload.ok) {
                    // 状态栏文本会先匹配 statusKey 前缀 "Export done"，
                    // 再把路径附加在后面，用户能看到 "导出完成 — D:\xxx.wav"。
                    const suffix = payload.path ? ` — ${payload.path}` : "";
                    state.status = `Export done${suffix}`;
                } else {
                    state.status = "Export failed";
                }
            })
            .addCase(exportAudio.rejected, setRejected)

            .addCase(exportSeparated.pending, (state) =>
                setPending(state, "Exporting separated tracks..."),
            )
            .addCase(exportSeparated.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                const payload = action.payload as {
                    ok?: boolean;
                    count?: number;
                    output_dir?: string;
                };
                if (payload.ok) {
                    const suffix = payload.output_dir
                        ? ` — ${payload.output_dir} (${payload.count ?? 0} tracks)`
                        : "";
                    state.status = `Export separated done${suffix}`;
                } else {
                    state.status = "Export separated failed";
                }
            })
            .addCase(exportSeparated.rejected, setRejected)

            .addCase(pasteVocalShifterClipboard.pending, (state) =>
                setPending(state, "Pasting VocalShifter clipboard data..."),
            )
            .addCase(pasteVocalShifterClipboard.fulfilled, (state, action) => {
                state.busy = false;
                const payload = action.payload as any;
                if (payload?.tracks) {
                    applyTimelineState(state, payload, { force: true });
                }
                state.lastResult = payload;
                state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
                state.status = "Pasted VocalShifter clipboard data";
            })
            .addCase(pasteVocalShifterClipboard.rejected, (state, action) => {
                state.busy = false;
                state.error =
                    (action.payload as string) ??
                    action.error?.message ??
                    "Request failed";
                state.status = "Failed";
            })

            .addCase(pasteReaperClipboard.pending, (state) =>
                setPending(state, "Pasting Reaper clipboard data..."),
            )
            .addCase(pasteReaperClipboard.fulfilled, (state, action) => {
                state.busy = false;
                const payload = action.payload as any;
                if (payload?.timeline) {
                    applyTimelineState(state, payload.timeline, { force: true });
                }
                const skippedFiles = payload?.skippedFiles;
                state.reaperSkippedFilesDialog =
                    Array.isArray(skippedFiles) && skippedFiles.length > 0
                        ? skippedFiles
                        : null;
                state.status = "Pasted Reaper clipboard data";
            })
            .addCase(pasteReaperClipboard.rejected, (state, action) => {
                state.busy = false;
                state.error =
                    (action.payload as string) ??
                    action.error?.message ??
                    "Paste Reaper clipboard failed";
                state.status = "Failed";
            })

            .addCase(playOriginal.pending, (state) =>
                setPending(state, "Playing original..."),
            )
            .addCase(playOriginal.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                const payload = action.payload as {
                    ok?: boolean;
                    clipId?: string | null;
                    anchorSec?: number;
                };
                const ok = Boolean(payload.ok);
                state.runtime.isPlaying = ok;
                state.runtime.playbackTarget = ok ? "original" : null;
                state.playbackClipId = ok ? (payload.clipId ?? null) : null;
                // Store the playhead position at which playback started,
                // so Play/Stop can restore it.
                state.playbackAnchorSec = ok ? (payload.anchorSec ?? 0) : 0;
                state.status = ok ? "Playing original" : "Play original failed";
            })
            .addCase(playOriginal.rejected, setRejected)

            .addCase(stopAudioPlayback.pending, (state) =>
                setPending(state, "Stopping audio..."),
            )
            .addCase(stopAudioPlayback.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                const payload = action.payload as {
                    ok?: boolean;
                    restoreAnchor?: boolean;
                    wasPlaying?: boolean;
                    anchorSec?: number | null;
                };
                // If restoreAnchor is set (Play/Stop action), restore playhead to anchor position
                if (
                    payload.restoreAnchor &&
                    payload.wasPlaying &&
                    typeof payload.anchorSec === "number" &&
                    Number.isFinite(payload.anchorSec)
                ) {
                    state.playheadSec = Math.max(0, payload.anchorSec);
                }
                state.runtime.isPlaying = false;
                state.runtime.playbackTarget = null;
                state.runtime.playbackPositionSec = 0;
                state.runtime.playbackDurationSec = 0;
                state.playbackClipId = null;
                state.playbackAnchorSec = 0;
                state.status = payload.ok
                    ? "Audio stopped"
                    : "Stop audio failed";
            })
            .addCase(stopAudioPlayback.rejected, setRejected)

            .addCase(syncPlaybackState.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                    is_playing?: boolean;
                    target?: string | null;
                    base_sec?: number;
                    position_sec?: number;
                    duration_sec?: number;
                };
                if (!payload.ok) {
                    return;
                }

                const nextIsPlaying = Boolean(payload.is_playing);
                const nextTarget = payload.target ?? null;
                const nextPositionSec = payload.position_sec ?? 0;
                const nextDurationSec = payload.duration_sec ?? 0;

                // 0.5ms 阈值：避免轮询带来的浮点抖动导致无意义的 Redux 更新
                const EPS_SEC = 0.0005;

                let nextplayheadSec = state.playheadSec;
                if (nextIsPlaying) {
                    const absSec = (payload.base_sec ?? 0) + nextPositionSec;
                    nextplayheadSec = Math.max(0, absSec);
                }

                const shouldUpdatePlaybackFields =
                    nextIsPlaying !== state.runtime.isPlaying ||
                    nextTarget !== state.runtime.playbackTarget ||
                    Math.abs(
                        nextPositionSec - state.runtime.playbackPositionSec,
                    ) > EPS_SEC ||
                    Math.abs(
                        nextDurationSec - state.runtime.playbackDurationSec,
                    ) > EPS_SEC ||
                    (nextIsPlaying &&
                        Math.abs(nextplayheadSec - state.playheadSec) >
                            EPS_SEC);

                if (!shouldUpdatePlaybackFields) {
                    // No state change needed.
                    return;
                }

                state.runtime.isPlaying = nextIsPlaying;
                state.runtime.playbackTarget = nextTarget;
                state.runtime.playbackPositionSec = nextPositionSec;
                state.runtime.playbackDurationSec = nextDurationSec;

                if (nextIsPlaying) {
                    state.playheadSec = nextplayheadSec;
                } else {
                    state.playbackClipId = null;
                }
            })

            .addCase(fetchTimeline.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(undoRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) return;
                // 保留播放头位置，避免 UI 跳变
                const currentPlayheadSec = state.playheadSec;
                applyTimelineState(state, payload, { force: true });
                state.playheadSec = currentPlayheadSec;
            })

            .addCase(redoRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) return;
                // 保留播放头位置，避免 UI 跳变
                const currentPlayheadSec = state.playheadSec;
                applyTimelineState(state, payload, { force: true });
                state.playheadSec = currentPlayheadSec;
            })

            .addCase(newProjectRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) return;
                applyTimelineState(state, payload, { force: true });
                state.status = "New project";
            })

            .addCase(openProjectFromDialog.pending, (state) =>
                setPending(state, "Opening project..."),
            )
            .addCase(openProjectFromDialog.fulfilled, (state, action) => {
                state.busy = false;
                const payload = action.payload as
                    | { ok: true; canceled: true }
                    | { ok: true; canceled: false; timeline: TimelineState };
                if (!payload || (payload as any).canceled) {
                    state.status = "Open canceled";
                    return;
                }
                applyTimelineState(state, (payload as any).timeline, { force: true });
                state.status = "Project opened";
            })
            .addCase(openProjectFromDialog.rejected, (state, action) => {
                state.busy = false;
                state.error = action.error?.message ?? "Open project failed";
                state.status = "Open failed";
            })

            .addCase(openProjectFromPath.pending, (state) =>
                setPending(state, "Opening project..."),
            )
            .addCase(openProjectFromPath.fulfilled, (state, action) => {
                state.busy = false;
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) return;
                applyTimelineState(state, payload, { force: true });
                state.status = "Project opened";
            })
            .addCase(openProjectFromPath.rejected, (state, action) => {
                state.busy = false;
                state.error = action.error?.message ?? "Open project failed";
                state.status = "Open failed";
            })

            .addCase(openVocalShifterFromDialog.pending, (state) =>
                setPending(state, "Importing VocalShifter project..."),
            )
            .addCase(openVocalShifterFromDialog.fulfilled, (state, action) => {
                state.busy = false;
                const payload = action.payload as
                    | { ok: true; canceled: true }
                    | {
                          ok: true;
                          canceled: false;
                          timeline: TimelineState;
                          skippedFiles?: string[];
                      };
                if (!payload || (payload as any).canceled) {
                    state.status = "Import canceled";
                    return;
                }
                applyTimelineState(state, (payload as any).timeline, { force: true });
                const skippedFiles = (payload as any).skippedFiles;
                state.vocalShifterSkippedFilesDialog =
                    Array.isArray(skippedFiles) && skippedFiles.length > 0
                        ? skippedFiles
                        : null;
                state.status = "VocalShifter project imported";
            })
            .addCase(openVocalShifterFromDialog.rejected, (state, action) => {
                state.busy = false;
                state.error =
                    (action.payload as string) ??
                    action.error?.message ??
                    "Import VocalShifter failed";
                state.status = "Import failed";
            })

            .addCase(openVocalShifterFromPath.pending, (state) =>
                setPending(state, "Importing VocalShifter project..."),
            )
            .addCase(openVocalShifterFromPath.fulfilled, (state, action) => {
                state.busy = false;
                const payload = action.payload as
                    | { ok: true; canceled: true }
                    | {
                          ok: true;
                          canceled: false;
                          timeline: TimelineState;
                          skippedFiles?: string[];
                      };
                if (!payload || (payload as any).canceled) {
                    state.status = "Import canceled";
                    return;
                }
                applyTimelineState(state, (payload as any).timeline, { force: true });
                const skippedFiles = (payload as any).skippedFiles;
                state.vocalShifterSkippedFilesDialog =
                    Array.isArray(skippedFiles) && skippedFiles.length > 0
                        ? skippedFiles
                        : null;
                state.status = "VocalShifter project imported";
            })
            .addCase(openVocalShifterFromPath.rejected, (state, action) => {
                state.busy = false;
                state.error =
                    (action.payload as string) ??
                    action.error?.message ??
                    "Import VocalShifter failed";
                state.status = "Import failed";
            })

            .addCase(openReaperFromDialog.pending, (state) =>
                setPending(state, "Importing Reaper project..."),
            )
            .addCase(openReaperFromDialog.fulfilled, (state, action) => {
                state.busy = false;
                const payload = action.payload as
                    | { ok: true; canceled: true }
                    | {
                          ok: true;
                          canceled: false;
                          timeline: TimelineState;
                          skippedFiles?: string[];
                      };
                if (!payload || (payload as any).canceled) {
                    state.status = "Import canceled";
                    return;
                }
                applyTimelineState(state, (payload as any).timeline, { force: true });
                const skippedFiles = (payload as any).skippedFiles;
                state.reaperSkippedFilesDialog =
                    Array.isArray(skippedFiles) && skippedFiles.length > 0
                        ? skippedFiles
                        : null;
                state.status = "Reaper project imported";
            })
            .addCase(openReaperFromDialog.rejected, (state, action) => {
                state.busy = false;
                state.error =
                    (action.payload as string) ??
                    action.error?.message ??
                    "Import Reaper failed";
                state.status = "Import failed";
            })

            .addCase(openReaperFromPath.pending, (state) =>
                setPending(state, "Importing Reaper project..."),
            )
            .addCase(openReaperFromPath.fulfilled, (state, action) => {
                state.busy = false;
                const payload = action.payload as
                    | { ok: true; canceled: true }
                    | {
                          ok: true;
                          canceled: false;
                          timeline: TimelineState;
                          skippedFiles?: string[];
                      };
                if (!payload || (payload as any).canceled) {
                    state.status = "Import canceled";
                    return;
                }
                applyTimelineState(state, (payload as any).timeline, { force: true });
                const skippedFiles = (payload as any).skippedFiles;
                state.reaperSkippedFilesDialog =
                    Array.isArray(skippedFiles) && skippedFiles.length > 0
                        ? skippedFiles
                        : null;
                state.status = "Reaper project imported";
            })
            .addCase(openReaperFromPath.rejected, (state, action) => {
                state.busy = false;
                state.error =
                    (action.payload as string) ??
                    action.error?.message ??
                    "Import Reaper failed";
                state.status = "Import failed";
            })

            .addCase(saveProjectRemote.fulfilled, (state, action) => {
                const payload = action.payload as any;
                if (payload?.ok && payload?.canceled) {
                    state.status = "Save canceled";
                    return;
                }

                // 保留播放头位置和音高曲线，避免 UI 跳变
                const currentPlayheadSec = state.playheadSec;
                const currentParamsEpoch = state.paramsEpoch;
                const currentClipPitchCurves = state.clipPitchCurves;

                if (payload?.ok && payload?.timeline?.ok) {
                    applyTimelineState(
                        state,
                        payload.timeline as TimelineState,
                        { force: true },
                    );
                    state.playheadSec = currentPlayheadSec;
                    state.paramsEpoch = currentParamsEpoch;
                    state.clipPitchCurves = currentClipPitchCurves;
                    state.status = "Project saved";
                    return;
                }

                if (payload?.ok && payload?.tracks && payload?.clips) {
                    applyTimelineState(state, payload as TimelineState, { force: true });
                    state.playheadSec = currentPlayheadSec;
                    state.paramsEpoch = currentParamsEpoch;
                    state.clipPitchCurves = currentClipPitchCurves;
                    state.status = "Project saved";
                    return;
                }

                if (payload?.ok) {
                    state.project.dirty = false;
                    state.status = "Project saved";
                    return;
                }

                state.status = "Save failed";
            })

            .addCase(saveProjectAsRemote.fulfilled, (state, action) => {
                const payload = action.payload as any;
                if (payload?.ok && payload?.canceled) {
                    state.status = "Save As canceled";
                    return;
                }

                // 保留播放头位置和音高曲线，避免 UI 跳变
                const currentPlayheadSec = state.playheadSec;
                const currentParamsEpoch = state.paramsEpoch;
                const currentClipPitchCurves = state.clipPitchCurves;

                if (payload?.ok && payload?.timeline?.ok) {
                    applyTimelineState(
                        state,
                        payload.timeline as TimelineState,
                        { force: true },
                    );
                    state.playheadSec = currentPlayheadSec;
                    state.paramsEpoch = currentParamsEpoch;
                    state.clipPitchCurves = currentClipPitchCurves;
                    state.status = "Project saved";
                    return;
                }

                if (payload?.ok && payload?.tracks && payload?.clips) {
                    applyTimelineState(state, payload as TimelineState, { force: true });
                    state.playheadSec = currentPlayheadSec;
                    state.paramsEpoch = currentParamsEpoch;
                    state.clipPitchCurves = currentClipPitchCurves;
                    state.status = "Project saved";
                    return;
                }

                if (payload?.ok) {
                    state.project.dirty = false;
                    state.status = "Project saved";
                    return;
                }

                state.status = "Save As failed";
            })

            .addCase(setProjectBaseScaleRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                    project?: {
                        base_scale?: string;
                        use_custom_scale?: boolean;
                        custom_scale?: {
                            id?: string;
                            name?: string;
                            notes?: number[];
                        } | null;
                        dirty?: boolean;
                    };
                };
                if (!payload.ok) {
                    return;
                }
                const next = payload.project?.base_scale;
                if (next && (SCALE_KEYS as readonly string[]).includes(next)) {
                    state.project.baseScale = next as typeof state.project.baseScale;
                }
                if (typeof payload.project?.use_custom_scale === "boolean") {
                    state.project.useCustomScale = payload.project.use_custom_scale;
                }
                if (payload.project?.custom_scale != null) {
                    state.project.customScale = payload.project.custom_scale
                        ? sanitizeCustomScalePreset(payload.project.custom_scale)
                        : null;
                }
                if (typeof payload.project?.dirty === "boolean") {
                    state.project.dirty = payload.project.dirty;
                }
            })

            .addCase(setProjectCustomScaleRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                    project?: {
                        use_custom_scale?: boolean;
                        custom_scale?: {
                            id?: string;
                            name?: string;
                            notes?: number[];
                        } | null;
                        dirty?: boolean;
                    };
                };
                if (!payload.ok) return;
                if (typeof payload.project?.use_custom_scale === "boolean") {
                    state.project.useCustomScale = payload.project.use_custom_scale;
                }
                if (payload.project?.custom_scale != null) {
                    state.project.customScale = payload.project.custom_scale
                        ? sanitizeCustomScalePreset(payload.project.custom_scale)
                        : null;
                }
                if (typeof payload.project?.dirty === "boolean") {
                    state.project.dirty = payload.project.dirty;
                }
            })

            .addCase(
                setProjectTimelineSettingsRemote.fulfilled,
                (state, action) => {
                    const payload = action.payload as {
                        ok?: boolean;
                        project?: {
                            beats_per_bar?: number;
                            grid_size?: string;
                            dirty?: boolean;
                        };
                    };
                    if (!payload.ok) {
                        return;
                    }
                    const beats = clamp(
                        Number(payload.project?.beats_per_bar ?? state.beats),
                        1,
                        32,
                    );
                    const gridRaw = String(
                        payload.project?.grid_size ?? state.grid,
                    );
                    const valid = (
                        [
                            "1/1", "1/2", "1/4", "1/8", "1/16", "1/32", "1/64",
                            "1/1d", "1/2d", "1/4d", "1/8d", "1/16d", "1/32d", "1/64d",
                            "1/1t", "1/2t", "1/4t", "1/8t", "1/16t", "1/32t", "1/64t",
                        ] as const
                    ).includes(gridRaw as any);
                    const grid = (valid ? gridRaw : "1/4") as GridSize;

                    state.beats = beats;
                    state.grid = grid;
                    state.project.beatsPerBar = beats;
                    state.project.gridSize = grid;
                    if (typeof payload.project?.dirty === "boolean") {
                        state.project.dirty = payload.project.dirty;
                    }
                },
            )

            .addCase(addClipOnTrack.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(duplicateTrackRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                // 克隆轨道属于离散操作，必须立即同步后端快照，避免 UI 延迟刷新。
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(createClipsRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
                state.status = "Clips created";
            })

            .addCase(removeClipRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(removeClipsRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(removeSelectedClipRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(moveClipRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload);
            })

            .addCase(moveClipsRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload);
            })

            .addCase(splitClipRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(glueClipsRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
                state.status = "Glue done";
            })

            .addCase(setClipStateRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload);
            })

            .addCase(replaceClipSourceRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(selectClipRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload);
            })

            .addCase(setTrackStateRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                // 保留当前播放头位置，避免 mute/solo 操作时播放头跳变
                const currentPlayheadSec = state.playheadSec;
                // 保留 paramsEpoch 和 clipPitchCurves，避免触发钢琴窗音高曲线重新渲染
                const currentParamsEpoch = state.paramsEpoch;
                const currentClipPitchCurves = state.clipPitchCurves;
                applyTimelineState(state, payload);
                state.playheadSec = currentPlayheadSec;
                state.paramsEpoch = currentParamsEpoch;
                state.clipPitchCurves = currentClipPitchCurves;
            })

            .addCase(updateTransportBpm.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                    bpm?: number;
                } & Partial<TimelineState>;
                if (!payload.ok) {
                    return;
                }
                state.bpm = clamp(Number(payload.bpm ?? state.bpm), 10, 300);
                if (payload.tracks && payload.clips) {
                    applyTimelineState(state, payload as TimelineState, { force: true });
                }
            })

            .addCase(seekPlayhead.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                    playhead_sec?: number;
                } & Partial<TimelineState>;
                if (!payload.ok) {
                    return;
                }
                // 使用请求参数（action.meta.arg）作为可信值，而非后端返回的
                // playhead_sec，因为并发请求时旧响应可能晚于新请求到达，
                // 用旧值覆盖前端已更新的 playheadSec 会导致光标闪烁。
                const requestedSec = action.meta.arg as number;
                const backendSec = Number(
                    payload.playhead_sec ?? requestedSec,
                );
                // 仅在前端 playheadSec 与请求参数一致（未被更新的请求覆盖）
                // 或后端返回值与请求参数不同时才采纳后端值。
                const EPS = 0.001;
                if (Math.abs(backendSec - requestedSec) > EPS) {
                    // 后端对位置做了修正（如 clamp），采纳后端值
                    state.playheadSec = Math.max(0, backendSec);
                }
                // 否则保持前端已同步设好的 state.playheadSec，不再覆盖
                if (state.runtime.isPlaying) {
                    state.playbackAnchorSec = state.playheadSec;
                }
                if (payload.tracks && payload.clips) {
                    applyTimelineState(state, payload as TimelineState, { force: true });
                }
            })

            .addCase(addTrackRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(removeTrackRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(moveTrackRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(selectTrackRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload);
            })

            .addCase(setProjectLengthRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(fetchSelectedTrackSummary.fulfilled, (state, action) => {
                const payload = action.payload as
                    | TrackSummaryResult
                    | { ok?: false };
                if (!payload.ok) {
                    return;
                }
                state.selectedTrackSummary = {
                    trackId: payload.track_id,
                    clipCount: payload.clip_count,
                    waveformPreview: payload.waveform_preview,
                    pitchRange: payload.pitch_range,
                };
            });
    },
});

export const {
    beginInteraction,
    endInteraction,
    setTrackName,
    checkpointHistory,
    setToolMode,
    setEditParam,
    setBpm,
    setBeats,
    setGrid,
    toggleAutoCrossfade,
    toggleGridSnap,
    togglePitchSnap,
    setPitchSnapUnit,
    setPitchSnapScale,
    togglePlayheadZoom,
    toggleAutoScroll,
    toggleParamEditorSeekPlayhead,
    toggleClipboardPreview,
    toggleParamValuePopup,
    cycleDragDirection,
    setDragDirection,
    setEdgeSmoothnessPercent,
    setplayheadSec,
    setModelDir,
    setAudioPath,
    setOutputPath,
    setPitchShift,
    closeVocalShifterSkippedFilesDialog,
    closeReaperSkippedFilesDialog,
    setPitchSnapToleranceCents,
    setScaleHighlightMode,
    upsertCustomScalePreset,
    toggleLockParamLines,
    setSelectedClip,
    setSelectedClipPreservingTrack,
    setMultiSelectedClipIds,
    moveClipStart,
    moveClipTrack,
    setClipLength,
    setClipPlaybackRate,
    setClipSourceRange,
    setClipFades,
    setClipGain,
    setClipMuted,
    optimisticUpdateClipColor,
    rollbackClipColor,
    addClip,
    removeSelectedClip,
    toggleTrackMute,
    toggleTrackSolo,
    setTrackVolume,
    addAutomationPoint,
    moveAutomationPoint,
    setSelectedPoint,
    removeAutomationPoint,
    setClipPitchData,
    removeClipPitchData,
    undo,
    redo,
} = sessionSlice.actions;

export default sessionSlice.reducer;
