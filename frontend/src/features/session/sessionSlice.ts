import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import type {
    TimelineClip,
    TimelineClipTake,
    TimelineState,
    TrackSummaryResult,
} from "../../types/api";
import type {
    AutomationPoint,
    ClipInfo,
    ClipFormantAnalysisState,
    ClipFormantMorph,
    ClipTakeInfo,
    ClipTemplate,
    DrawToolMode,
    DragDirection,
    DrawDragDirection,
    EditParam,
    GridSize,
    PitchSnapUnit,
    SplitTransitionCurveType,
    TimelineSnapSettings,
    TimeUnit,
    TimeUnitChoice,
    TrackMeterInfo,
    ToolMode,
    ToolModeGroup,
    TrackInfo,
} from "./sessionTypes";
import { normalizeSplitTransitionCurve } from "./sessionTypes";
import { modEuclid, resolveLoopMediaDurationSec } from "../../utils/loopRender";

import {
    addClipOnTrack,
    addTrackRemote,
    createClipsRemote,
    pasteTimelineClipboardRemote,
    duplicateClipsBulkRemote,
    duplicateTrackRemote,
    fetchSelectedTrackSummary,
    convertClipsToPitchReferenceRemote,
    updatePitchReferenceRemote,
    glueClipsRemote,
    groupClipsRemote,
    setClipActiveTakeRemote,
    cycleClipTakesRemote,
    packClipsIntoTakesRemote,
    explodeClipTakesRemote,
    duplicateClipTakeRemote,
    removeClipTakeRemote,
    renameClipTakeRemote,
    setClipTakeReversedRemote,
    addClipTakeFromMediaRemote,
    ungroupClipsRemote,
    toggleGroupDisabledRemote,
    moveClipRemote,
    moveClipsRemote,
    moveTrackRemote,
    removeClipRemote,
    removeClipsRemote,
    removeTrackRemote,
    replaceClipSourceRemote,
    replaceMidiClipDataRemote,
    selectClipRemote,
    selectTrackRemote,
    setClipStateRemote,
    setClipsStateBulkRemote,
    setProjectLengthRemote,
    splitClipRemote,
    splitClipsAtRemote,
} from "./thunks/timelineThunks";

import {
    newProjectRemote,
    openProjectFromDialog,
    openProjectFromPath,
    openProjectFromPathForced,
    pickProjectToImport,
    importProjectFromPath,
    openVocalShifterFromDialog,
    openVocalShifterFromPath,
    openReaperFromDialog,
    openReaperFromPath,
    redoRemote,
    saveProjectAsRemote,
    saveProjectRemote,
    saveProjectToPathRemote,
    setProjectBaseScaleRemote,
    setProjectCustomScaleRemote,
    setProjectStretchSettingsRemote,
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

import { clearWaveformCacheRemote, loadUiSettings, refreshRuntime } from "./thunks/runtimeThunks";

import { loadDefaultModel, loadModel } from "./thunks/modelThunks";

import {
    applyPitchShift,
    exportAudio,
    exportAudioAdvanced,
    exportSeparated,
    pasteVocalShifterClipboard,
    pasteReaperClipboard,
    pickOutputPath,
    processAudio,
    synthesizeAudio,
} from "./thunks/audioThunks";

import { SCALE_KEYS } from "../../utils/musicalScales";
import type { ScaleLike } from "../../utils/musicalScales";
import type { CustomScalePreset } from "../../utils/customScales";
import { sanitizeCustomScalePreset } from "../../utils/customScales";
import type { TempoMap } from "../../utils/tempoMap";
import {
    clampDenominator,
    fromBackendTempoMap,
    normalizeTempoMap,
    scaleLikeToScaleData,
    TEMPO_DENOMINATORS,
} from "../../utils/tempoMap";
import { setTempoMapRemote } from "./thunks/tempoMapThunks";
import {
    importAudioAtPosition,
    importAudioFileAtPosition,
    importAudioFromDialog,
    importAudioFromPath,
    importMidiAsClip,
    importMultipleAudioAtPosition,
    importMultipleAudioFilesAtPosition,
} from "./thunks/importThunks";

import { removeSelectedClipRemote, setTrackStateRemote } from "./thunks/trackThunks";
import { markProjectDirty } from "./sessionDirtyState";
import { resolveTrackIdForClipSelection } from "./selectionFocus";

const MAX_TRACK_VOLUME = 4;
/** 播放光标"真实挪动"判定阈值：低于该值的差异视为浮点噪声，
 * 不触发撤销/重做后的"聚焦播放光标"（视图滚动）登记。 */
const PLAYHEAD_MOVE_EPS_SEC = 1e-6;
const VALID_GRID_SIZES = new Set<GridSize>([
    "1/1",
    "1/2",
    "1/4",
    "1/8",
    "1/16",
    "1/32",
    "1/64",
    "1/1d",
    "1/2d",
    "1/4d",
    "1/8d",
    "1/16d",
    "1/32d",
    "1/64d",
    "1/1t",
    "1/2t",
    "1/4t",
    "1/8t",
    "1/16t",
    "1/32t",
    "1/64t",
]);

const VALID_TIME_UNITS = new Set<TimeUnit>(["barBeats", "barDivisions", "seconds", "clock"]);
const DEFAULT_PRIMARY_TIME_UNIT: TimeUnit = "barBeats";
const DEFAULT_SECONDARY_TIME_UNIT: TimeUnitChoice = "clock";
const DEFAULT_RULER_LABEL_SPACING_PX = 110;

export function createDefaultTimelineSnapSettings(): TimelineSnapSettings {
    return {
        gridVisible: true,
        gridMinSpacingPx: 8,
        swingEnabled: false,
        swingPercent: 0,
        adjustClipsOnSwingChange: true,
        enabled: true,
        snapDistancePx: 4,
        snapRelativeToGrid: false,
        snapHighlightEnabled: true,
        snapClipsToSelectionMarkersCursor: true,
        snapClipsToGrid: true,
        snapSelectionToSelectionMarkersCursor: true,
        snapSelectionToGrid: true,
        snapCursorToSelectionMarkersCursor: true,
        snapCursorToGrid: true,
        snapFollowsGridVisibility: true,
        snapToGridAnyDistance: false,
        useIndependentSnapSpacing: false,
        snapSpacing: "1/4",
        snapSpacingMinPx: 8,
        snapClipEdges: true,
        snapClipSnapOffset: true,
        snapAcrossTracks: true,
        snapTrackDistance: 0,
        snapRazorEdits: true,
        snapToProjectSampleRate: false,
        snapClipsToSourceMedia: true,
        forceSelectionsToMultiples: false,
        selectionMultiple: "1/4",
        syncArrangeAndMidiGrid: true,
    };
}

function normalizeTimelineSnapSettings(
    base: TimelineSnapSettings,
    patch: Partial<TimelineSnapSettings>,
): TimelineSnapSettings {
    const grid = (value: unknown): GridSize =>
        VALID_GRID_SIZES.has(value as GridSize) ? (value as GridSize) : "1/4";
    const clampedPx = (value: unknown, fallback: number, min: number, max: number) => {
        const n = Number(value);
        return Number.isFinite(n) ? Math.min(max, Math.max(min, Math.round(n))) : fallback;
    };
    const clampedNum = (value: unknown, fallback: number, min: number, max: number) => {
        const n = Number(value);
        return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
    };
    const bool = (value: unknown, fallback: boolean) =>
        typeof value === "boolean" ? value : fallback;
    return {
        gridVisible: bool(patch.gridVisible, base.gridVisible),
        gridMinSpacingPx: clampedPx(patch.gridMinSpacingPx, base.gridMinSpacingPx, 2, 200),
        swingEnabled: bool(patch.swingEnabled, base.swingEnabled),
        swingPercent: clampedNum(patch.swingPercent, base.swingPercent, 0, 100),
        adjustClipsOnSwingChange: bool(
            patch.adjustClipsOnSwingChange,
            base.adjustClipsOnSwingChange,
        ),
        enabled: bool(patch.enabled, base.enabled),
        snapDistancePx: clampedPx(patch.snapDistancePx, base.snapDistancePx, 0, 200),
        snapRelativeToGrid: bool(patch.snapRelativeToGrid, base.snapRelativeToGrid),
        snapHighlightEnabled: bool(patch.snapHighlightEnabled, base.snapHighlightEnabled),
        snapClipsToSelectionMarkersCursor: bool(
            patch.snapClipsToSelectionMarkersCursor,
            base.snapClipsToSelectionMarkersCursor,
        ),
        snapClipsToGrid: bool(patch.snapClipsToGrid, base.snapClipsToGrid),
        snapSelectionToSelectionMarkersCursor: bool(
            patch.snapSelectionToSelectionMarkersCursor,
            base.snapSelectionToSelectionMarkersCursor,
        ),
        snapSelectionToGrid: bool(patch.snapSelectionToGrid, base.snapSelectionToGrid),
        snapCursorToSelectionMarkersCursor: bool(
            patch.snapCursorToSelectionMarkersCursor,
            base.snapCursorToSelectionMarkersCursor,
        ),
        snapCursorToGrid: bool(patch.snapCursorToGrid, base.snapCursorToGrid),
        snapFollowsGridVisibility: bool(
            patch.snapFollowsGridVisibility,
            base.snapFollowsGridVisibility,
        ),
        snapToGridAnyDistance: bool(patch.snapToGridAnyDistance, base.snapToGridAnyDistance),
        useIndependentSnapSpacing: bool(
            patch.useIndependentSnapSpacing,
            base.useIndependentSnapSpacing,
        ),
        snapSpacing: grid(patch.snapSpacing ?? base.snapSpacing),
        snapSpacingMinPx: clampedPx(patch.snapSpacingMinPx, base.snapSpacingMinPx, 2, 200),
        snapClipEdges: bool(patch.snapClipEdges, base.snapClipEdges),
        snapClipSnapOffset: bool(patch.snapClipSnapOffset, base.snapClipSnapOffset),
        snapAcrossTracks: bool(patch.snapAcrossTracks, base.snapAcrossTracks),
        snapTrackDistance: clampedNum(patch.snapTrackDistance, base.snapTrackDistance, 0, 32),
        snapRazorEdits: bool(patch.snapRazorEdits, base.snapRazorEdits),
        snapToProjectSampleRate: bool(patch.snapToProjectSampleRate, base.snapToProjectSampleRate),
        snapClipsToSourceMedia: bool(patch.snapClipsToSourceMedia, base.snapClipsToSourceMedia),
        forceSelectionsToMultiples: bool(
            patch.forceSelectionsToMultiples,
            base.forceSelectionsToMultiples,
        ),
        selectionMultiple: grid(patch.selectionMultiple ?? base.selectionMultiple),
        syncArrangeAndMidiGrid: bool(patch.syncArrangeAndMidiGrid, base.syncArrangeAndMidiGrid),
    };
}

export type { FadeCurveType } from "./sessionTypes";
export type {
    AutomationPoint,
    ClipInfo,
    ClipTemplate,
    DrawToolMode,
    DragDirection,
    DrawDragDirection,
    EditParam,
    GridSize,
    ToolMode,
    ToolModeGroup,
    TrackInfo,
};

type ClipColor = ClipInfo["color"];
type WaveformPreview = number[] | { l: number[]; r: number[] };
type StretchAlgorithmOption = "linear" | "signalsmith" | "soundtouch";
type ClipFormantToolWindowState = {
    open: boolean;
    clipId: string | null;
    x: number;
    y: number;
    hasMoved: boolean;
};

export interface SessionState {
    toolMode: ToolMode;
    toolModeGroup: ToolModeGroup;
    drawToolMode: DrawToolMode;
    editParam: EditParam;
    bpm: number;
    beats: number;
    projectSec: number;
    grid: GridSize;
    primaryTimeUnit: TimeUnit;
    secondaryTimeUnit: TimeUnitChoice;
    rulerLabelSpacingPx: number;
    showPlayheadTimeInTrackHeader: boolean;
    paramEditorSyncTimeline: boolean;

    autoCrossfadeEnabled: boolean;
    /** 空间足够时显示 Clip 内全部 Take 波形。 */
    showAllTakes: boolean;
    /** 同步编辑所有 Take：内容级编辑同步到同一 Clip 的全部 Take。 */
    syncEditsAcrossTakes: boolean;
    /** 为新的音频块启用循环（Loop / 循环源，默认开启；仅影响新建 Clip）。 */
    loopNewClipsEnabled: boolean;
    /** 分割过渡 */
    splitTransitionEnabled: boolean;
    splitTransitionMode: "fade" | "overlap";
    splitTransitionDurationUnit: "seconds" | "percent";
    splitTransitionDurationSec: number;
    splitTransitionDurationPercent: number;
    /** 淡化曲线（新版 REAPER 预设；"keep" = 分割后保留原 Clip 曲线，默认）。 */
    splitTransitionCurve: SplitTransitionCurveType;
    splitTransitionOverlapCrossfade: "auto" | "always";
    /** 吸附总开关（兼容旧字段 gridSnapEnabled）。 */
    snapEnabled: boolean;
    /** 完整的时间轴吸附/网格设置（snapEnabled 仅作为旧字段镜像 enabled）。 */
    timelineSnap: TimelineSnapSettings;
    /**
     * Tempo Map 数据（null = 无 Tempo Map，使用工程全局 BPM/拍号/音阶）。
     * 变化点按秒锚定；0 位置点始终存在。
     */
    tempoMap: import("../../utils/tempoMap").TempoMap | null;
    /** Tempo Map 标尺行可见性（视图菜单开关，默认开启）。 */
    tempoMapVisible: boolean;
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
    /** 忽略编组（启用后编组同步编辑操作不生效） */
    ignoreGrouping: boolean;
    /**
     * 波纹编辑（自动跟进）模式：
     * - `"off"`：关闭（默认）。
     * - `"track"`：仅被编辑的轨道上的后续剪辑一起跟进。
     * - `"all"`：所有轨道上位于编辑点之后的剪辑一起跟进。
     */
    rippleMode: "off" | "track" | "all";
    /** 被禁用的编组 ID 列表（禁用后该编组内同步编辑不生效） */
    disabledGroupIds: string[];
    /** 允许参数编辑器点击时调整播放头 */
    paramEditorSeekPlayheadEnabled: boolean;
    /** 允许时间轴上的点击操作自动切换当前轨道 */
    paramEditorTimelineClickSelectTrackEnabled: boolean;
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
    /** 快速搜索放置音频时自动规格化 */
    quickSearchAutoNormalizeEnabled: boolean;
    /** PianoRoll 中显示的其他 root track 参考线 */
    visibleReferenceRootTrackIds: string[];
    /** 全局默认外部拉伸算法 */
    defaultStretchAlgorithm: StretchAlgorithmOption;
    /** 全局默认 HiFiGAN mel stretch 开关 */
    defaultHifiganMelStretch: boolean;
    ortEp: string;
    gpuDeviceId: number;
    ortDeviceId: number | null;
    /** 后台预渲染：编辑后立即在后台渲染，无需等待播放触发 */
    autoBackgroundRender: boolean;

    // Monotonic bump token for invalidating parameter curve caches.
    // - Not included in undo/redo snapshots.
    // - Should be bumped on any timeline/undo/redo operation that may affect param rendering.
    paramsEpoch: number;
    /** 单调递增，任何 clip 的 playbackRate 变更时 +1，强制 UI 刷新 */
    playbackRateVersion: number;

    playheadSec: number;
    /**
     * 待执行的“聚焦播放光标”请求（粘贴后跳转光标时设置）。
     *
     * 粘贴可能显著扩充工程全长（水平可滚动范围随之扩大）。若在 fulfilled
     * reducer 提交之前就尝试滚动，dynamicProjectSec / DOM 内容宽度都还是
     * 旧值，滚动会被旧上限钳制，导致光标无法进入画面。因此这里只记录
     * 目标位置，由 TimelinePanel 的 useLayoutEffect 在新状态与 DOM 均提交
     * 后再执行滚动并回写 null。
     */
    pendingPlayheadRevealSec: number | null;
    tracks: TrackInfo[];
    trackMeters: Record<string, TrackMeterInfo>;
    clips: ClipInfo[];
    selectedTrackId: string | null;
    selectedClipId: string | null;
    /** 多选 clip 的 id 列表（框选 / 主修饰键 + 点击） */
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
    clipFormantStatus: Record<string, "ready" | "rebuilding" | "failed">;
    /** Clip 源共振峰分析结果（共振峰工具窗口可视化用） */
    clipFormantAnalysis: Record<string, ClipFormantAnalysisState>;
    clipFormantToolWindow: ClipFormantToolWindowState;
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
        gpuBackend: string;
    };

    selectedTrackSummary: {
        trackId: string | null;
        clipCount: number;
        waveformPreview: number[];
        pitchRange: { min: number; max: number };
    };

    customScalePresets: CustomScalePreset[];
    project: {
        name: string;
        path: string | null;
        dirty: boolean;
        recent: string[];
        notesMarkdown: string;
        baseScale: import("../../utils/musicalScales").ScaleKey;
        useCustomScale: boolean;
        customScale: CustomScalePreset | null;
        beatsPerBar: number;
        /** 工程基准拍号分母（1/2/4/8/16/32）。 */
        timeSignatureDenominator: number;
        gridSize: GridSize;
        stretchAlgorithmOverride: StretchAlgorithmOption | null;
        hifiganMelStretchOverride: boolean | null;
    };

    busy: boolean;
    status: string;
    error?: string;
    lastResult?: unknown;
    vocalShifterSkippedFilesDialog: string[] | null;
    reaperSkippedFilesDialog: string[] | null;

    /**
     * 保存/另存为目标位置已存在版本不一致的工程文件时，弹出的覆盖确认对话框。
     * 为 null 表示没有待确认的覆盖。
     */
    saveVersionConflictDialog: {
        path: string;
        existingVersion: number;
        currentVersion: number;
        existingIsNewer: boolean;
    } | null;

    /**
     * 交互锁计数器：当用户正在进行连续操作（拖动、滑动等）时 > 0。
     * 在锁定期间，连续操作类 thunk 的 fulfilled handler 将跳过 applyTimelineState()，
     * 避免后端返回的过期快照覆盖前端乐观更新导致的闪烁。
     */
    _interactionLockCount: number;

    /**
     * 最近一次 undo/redo 请求的 requestId。快速连续撤销/重做会产生多个
     * in-flight thunk；fulfilled/rejected 到达时与该字段比对，丢弃过期
     * 响应，防止旧快照以 force 覆盖新状态（与 seekPlayhead 的乱序防护同理）。
     */
    _latestHistoryOpRequestId: string | null;

    /**
     * 最近一次**编辑类**请求的 requestId（setClipState / bulk / move 族）。
     * 编辑 thunk 的 fulfilled 会 force-apply 后端全量快照；多选批量等场景
     * 若旧请求的响应迟到（或后端返回中间状态），会覆盖更新的乐观状态、
     * 造成"闪回原状 / 部分 Clip 还原"。pending 记录、fulfilled 比对，
     * 过期响应直接丢弃 —— 最新一次编辑的响应（含此前所有已提交变更）
     * 总是最后一个生效，撤/重做等权威操作不受影响。
     */
    _latestEditRequestId: string | null;
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

function normalizeClipColor(color: string | undefined): ClipColor {
    if (color === "blue") return "blue";
    if (color === "violet") return "violet";
    if (color === "amber") return "amber";
    if (color === "cyan") return "cyan";
    return "emerald";
}

/**
 * Auto-crossfade logic applied directly in a reducer (no dispatch needed).
 * For each clip in `movedIds`, detect overlaps with same-track clips and set
 * auto fade in/out to the crossfade-eligible overlap duration.
 *
 * A clip fully contained inside another (start/end coincident or inside) is
 * never a valid crossfade relationship, so its auto fade is cleared instead.
 */
function applyAutoCrossfadeInReducer(state: SessionState, movedIds: string[]) {
    if (!state.autoCrossfadeEnabled || movedIds.length === 0) return;

    const movedSet = new Set(movedIds);
    const trackClipsMap: Record<string, ClipInfo[]> = {};
    const clipMap: Record<string, ClipInfo> = {};

    // 一次性建立轨道分组和全局 ID 索引，时间复杂度 O(N)
    for (const clip of state.clips) {
        clipMap[clip.id] = clip;
        if (!trackClipsMap[clip.trackId]) trackClipsMap[clip.trackId] = [];
        trackClipsMap[clip.trackId].push(clip);
    }

    // raw* 用于判断哪一侧被触碰；fade*Overlaps 只保存可产生自动交叉淡化的值。
    const rawFadeInOverlaps = new Map<string, number>();
    const rawFadeOutOverlaps = new Map<string, number>();
    const fadeInOverlaps = new Map<string, number>();
    const fadeOutOverlaps = new Map<string, number>();

    const isContained = (aStart: number, aEnd: number, bStart: number, bEnd: number): boolean => {
        const eps = 1e-9;
        return aStart >= bStart - eps && aEnd <= bEnd + eps;
    };

    for (const id of movedIds) {
        // O(1) 直接获取，消除多余的 find 遍历
        const clip = clipMap[id];
        if (!clip) continue;
        const clipStart = Number(clip.startSec);
        const clipEnd = clipStart + Number(clip.lengthSec);

        const sameTrack = (trackClipsMap[clip.trackId] || []).filter((c) => c.id !== id);

        for (const other of sameTrack) {
            const otherStart = Number(other.startSec);
            const otherEnd = otherStart + Number(other.lengthSec);
            const overlapStart = Math.max(clipStart, otherStart);
            const overlapEnd = Math.min(clipEnd, otherEnd);
            const overlap = overlapEnd - overlapStart;
            if (overlap <= 0.001) continue;

            const eligible =
                isContained(clipStart, clipEnd, otherStart, otherEnd) ||
                isContained(otherStart, otherEnd, clipStart, clipEnd)
                    ? 0
                    : overlap;

            if (clipStart <= otherStart) {
                rawFadeOutOverlaps.set(id, Math.max(rawFadeOutOverlaps.get(id) ?? 0, overlap));
                fadeOutOverlaps.set(id, Math.max(fadeOutOverlaps.get(id) ?? 0, eligible));
                rawFadeInOverlaps.set(
                    other.id,
                    Math.max(rawFadeInOverlaps.get(other.id) ?? 0, overlap),
                );
                fadeInOverlaps.set(other.id, Math.max(fadeInOverlaps.get(other.id) ?? 0, eligible));
            } else {
                rawFadeInOverlaps.set(id, Math.max(rawFadeInOverlaps.get(id) ?? 0, overlap));
                fadeInOverlaps.set(id, Math.max(fadeInOverlaps.get(id) ?? 0, eligible));
                rawFadeOutOverlaps.set(
                    other.id,
                    Math.max(rawFadeOutOverlaps.get(other.id) ?? 0, overlap),
                );
                fadeOutOverlaps.set(
                    other.id,
                    Math.max(fadeOutOverlaps.get(other.id) ?? 0, eligible),
                );
            }
        }
    }

    const allClipIds = new Set([
        ...rawFadeInOverlaps.keys(),
        ...rawFadeOutOverlaps.keys(),
        ...movedIds,
    ]);

    for (const clipId of allClipIds) {
        // O(1) 直接获取，消除多余的 find 遍历
        const clip = clipMap[clipId];
        if (!clip) continue;

        const rawIn = rawFadeInOverlaps.get(clipId);
        const rawOut = rawFadeOutOverlaps.get(clipId);

        if (rawIn !== undefined) {
            clip.autoFadeInSec = Math.max(0, fadeInOverlaps.get(clipId) ?? 0);
        } else if (movedSet.has(clipId)) {
            // 新导入/新建 clip 没有合法交叉淡化重叠时，自动 fade 应为 0。
            clip.autoFadeInSec = 0;
        }
        if (rawOut !== undefined) {
            clip.autoFadeOutSec = Math.max(0, fadeOutOverlaps.get(clipId) ?? 0);
        } else if (movedSet.has(clipId)) {
            clip.autoFadeOutSec = 0;
        }
    }
}

function mapTimelineTracks(tracks: TimelineState["tracks"]): TrackInfo[] {
    return tracks.map((track) => ({
        id: track.id,
        name: track.name,
        parentId: track.parent_id ?? null,
        depth: track.depth ?? 0,
        childTrackIds: track.child_track_ids ?? [],
        muted: Boolean(track.muted),
        solo: Boolean(track.solo),
        volume: clamp(Number(track.volume ?? 1), 0, MAX_TRACK_VOLUME),

        composeEnabled: Boolean(track.compose_enabled),
        pitchAnalysisAlgo: String(track.pitch_analysis_algo ?? "nsf_hifigan_onnx"),
        color: track.color || undefined,
    }));
}

function applyTimelineTracksOnly(state: SessionState, timeline: TimelineState) {
    state.tracks = mapTimelineTracks(timeline.tracks);
    state.selectedTrackId = timeline.selected_track_id;
}

function applyTimelineStatePreservingPitchVisuals(state: SessionState, timeline: TimelineState) {
    const currentParamsEpoch = state.paramsEpoch;
    // 必须捕获浅拷贝快照：在 Immer producer 内直接持有 state.clipPitchCurves
    // 得到的是 draft 代理，applyTimelineState 随后的 prune（delete 缺失 clip）
    // 会透过代理可见，"恢复"赋值等于把已删空的 draft 原样写回。
    // 注意：此处**整包恢复**是有意契约（见 clipCreation.test）——部分载荷
    // 可能不含未变化的 clip，按新 clip 集过滤会误删仍然有效的曲线。
    const currentClipPitchCurves = { ...state.clipPitchCurves };
    applyTimelineState(state, timeline, { force: true });
    state.paramsEpoch = currentParamsEpoch;
    state.clipPitchCurves = currentClipPitchCurves;
}

function applyOptimisticTrackState(
    state: SessionState,
    payload: {
        trackId: string;
        muted?: boolean;
        solo?: boolean;
        volume?: number;
        composeEnabled?: boolean;
        pitchAnalysisAlgo?: string;
        color?: string;
        name?: string;
    },
) {
    const track = state.tracks.find((entry) => entry.id === payload.trackId);
    if (!track) return;
    if (payload.muted !== undefined) {
        track.muted = Boolean(payload.muted);
    }
    if (payload.solo !== undefined) {
        track.solo = Boolean(payload.solo);
    }
    if (payload.volume !== undefined) {
        track.volume = clamp(Number(payload.volume), 0, MAX_TRACK_VOLUME);
    }
    if (payload.composeEnabled !== undefined) {
        track.composeEnabled = Boolean(payload.composeEnabled);
    }
    if (payload.pitchAnalysisAlgo !== undefined) {
        track.pitchAnalysisAlgo = String(payload.pitchAnalysisAlgo || track.pitchAnalysisAlgo);
    }
    if (payload.color !== undefined) {
        track.color = payload.color || undefined;
    }
    if (payload.name !== undefined) {
        track.name = String(payload.name || track.name);
    }
}

function applyOptimisticClipState(
    state: SessionState,
    payload: {
        clipId: string;
        name?: string;
        color?: string;
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
        formantMorph?: ClipFormantMorph;
    },
) {
    const clip = state.clips.find((entry) => entry.id === payload.clipId);
    if (!clip) return;
    if (payload.name !== undefined) {
        clip.name = String(payload.name || clip.name);
    }
    if (payload.color !== undefined) {
        clip.color = normalizeClipColor(payload.color);
    }
    if (payload.startSec !== undefined) {
        clip.startSec = Math.max(0, Number(payload.startSec) || 0);
    }
    if (payload.lengthSec !== undefined) {
        clip.lengthSec = Math.max(0, Number(payload.lengthSec) || 0);
        // trim 改写长度而未携带 snapOffset 时同步下钳（与后端 patch_clip_state
        // 口径一致），避免残留 offset > length 的"幻影吸附目标"。
        if (payload.snapOffsetSec === undefined) {
            clip.snapOffsetSec = Math.min(Math.max(0, clip.snapOffsetSec), clip.lengthSec);
        }
    }
    if (payload.gain !== undefined) {
        clip.gain = clamp(Number(payload.gain), 0, 4);
    }
    if (payload.muted !== undefined) {
        clip.muted = Boolean(payload.muted);
    }
    if (payload.sourceStartSec !== undefined) {
        clip.sourceStartSec = Number(payload.sourceStartSec) || 0;
    }
    if (payload.sourceEndSec !== undefined) {
        // 不得钳制到 ≥0：倒放 Clip 的消费窗口锚定 se，se<0（整窗在媒体
        // 下方的静音段）与 se>D（前导静音）都是合法状态。
        const value = Number(payload.sourceEndSec);
        clip.sourceEndSec = Number.isFinite(value) ? value : clip.sourceEndSec;
    }
    if (payload.playbackRate !== undefined) {
        clip.playbackRate = clamp(Number(payload.playbackRate), 0.1, 10);
    }
    if (payload.clipPlaybackRate !== undefined) {
        const nextClipRate = clamp(Number(payload.clipPlaybackRate) || 1, 0.1, 10);
        const takes = clip.takes ?? [];
        const activeTake = takes.find((entry) => entry.id === clip.activeTakeId) ?? takes[0];
        const previousTakeRate = activeTake
            ? activeTake.playbackRate
            : clamp(clip.playbackRate / getClipRateMultiplier(clip), 0.1, 10);
        clip.clipPlaybackRate = nextClipRate;
        clip.playbackRate = clamp(nextClipRate * previousTakeRate, 0.1, 10);
    }
    if (payload.reversed !== undefined) {
        // 方向翻转（且本请求未显式指定源窗口）时镜像后端 flip_direction_source_window
        // 的换算，保持消费窗口不变 —— 权威载荷到达前波形不跳变。
        if (
            payload.reversed !== clip.reversed &&
            payload.sourceStartSec === undefined &&
            payload.sourceEndSec === undefined
        ) {
            const rate =
                Number.isFinite(clip.playbackRate) && clip.playbackRate > 1e-6
                    ? clip.playbackRate
                    : 1;
            const mediaTotal = resolveLoopMediaDurationSec({
                durationFrames: clip.durationFrames,
                sourceSampleRate: clip.sourceSampleRate,
                durationSec: clip.durationSec,
            });
            flipSourceWindowForDirection(
                clip,
                Math.max(0, clip.lengthSec) * rate,
                mediaTotal > 0 ? mediaTotal : null,
            );
        }
        clip.reversed = Boolean(payload.reversed);
    }
    if (payload.loopEnabled !== undefined) {
        clip.loopEnabled = Boolean(payload.loopEnabled);
    }
    if (payload.snapOffsetSec !== undefined) {
        const offset = Math.max(0, Number(payload.snapOffsetSec) || 0);
        // 与后端一致：偏移点必须落在 [0, length] 内才可见、可吸附。
        clip.snapOffsetSec = Math.min(offset, clip.lengthSec);
    }
    if (payload.fadeInSec !== undefined) {
        clip.fadeInSec = Math.max(0, Number(payload.fadeInSec) || 0);
    }
    if (payload.fadeOutSec !== undefined) {
        clip.fadeOutSec = Math.max(0, Number(payload.fadeOutSec) || 0);
    }
    if (payload.fadeInShape !== undefined) {
        clip.fadeInShape = payload.fadeInShape;
    }
    if (payload.fadeOutShape !== undefined) {
        clip.fadeOutShape = payload.fadeOutShape;
    }
    if (payload.fadeInDir !== undefined) {
        clip.fadeInDir = Math.min(1, Math.max(-1, Number(payload.fadeInDir) || 0));
    }
    if (payload.fadeOutDir !== undefined) {
        clip.fadeOutDir = Math.min(1, Math.max(-1, Number(payload.fadeOutDir) || 0));
    }
    if (payload.autoFadeInSec !== undefined) {
        clip.autoFadeInSec = Math.max(0, Number(payload.autoFadeInSec) || 0);
    }
    if (payload.autoFadeOutSec !== undefined) {
        clip.autoFadeOutSec = Math.max(0, Number(payload.autoFadeOutSec) || 0);
    }
    if (payload.formantMorph !== undefined) {
        clip.formantMorph = payload.formantMorph ? { ...payload.formantMorph } : undefined;
    }
    updateTakesFromFlatWithSync(state, clip);

    const clipEnd = clip.startSec + clip.lengthSec;
    if (clipEnd > state.projectSec) {
        state.projectSec = Math.ceil(clipEnd);
    }
}

function applyOptimisticBulkClipState(
    state: SessionState,
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
        reversed?: boolean;
        loopEnabled?: boolean;
    }>,
) {
    for (const update of updates) {
        const clip = state.clips.find((entry) => entry.id === update.clipId);
        if (!clip) continue;
        if (update.gain !== undefined) {
            clip.gain = clamp(Number(update.gain), 0, 4);
        }
        if (update.muted !== undefined) {
            clip.muted = Boolean(update.muted);
        }
        if (update.startSec !== undefined) {
            clip.startSec = Math.max(0, Number(update.startSec) || 0);
        }
        if (update.lengthSec !== undefined) {
            clip.lengthSec = Math.max(0, Number(update.lengthSec) || 0);
            // trim 改写长度而未携带 snapOffset 时同步下钳（与后端 patch_clip_state
            // 口径一致），避免残留 offset > length 的"幻影吸附目标"。
            if (update.snapOffsetSec === undefined) {
                clip.snapOffsetSec = Math.min(Math.max(0, clip.snapOffsetSec), clip.lengthSec);
            }
        }
        if (update.snapOffsetSec !== undefined) {
            clip.snapOffsetSec = Math.min(
                Math.max(0, Number(update.snapOffsetSec) || 0),
                clip.lengthSec,
            );
        }
        if (update.clipPlaybackRate !== undefined) {
            // 与 applyOptimisticClipState 同口径：写 Clip 级倍率，组合有效
            // 速率 = Clip 级 × active take 速率。
            const nextClipRate = clamp(Number(update.clipPlaybackRate) || 1, 0.1, 10);
            const takes = clip.takes ?? [];
            const activeTake = takes.find((entry) => entry.id === clip.activeTakeId) ?? takes[0];
            const previousTakeRate = activeTake
                ? activeTake.playbackRate
                : clamp(clip.playbackRate / getClipRateMultiplier(clip), 0.1, 10);
            clip.clipPlaybackRate = nextClipRate;
            clip.playbackRate = clamp(nextClipRate * previousTakeRate, 0.1, 10);
        }
        if (update.sourceStartSec !== undefined) {
            clip.sourceStartSec = Number(update.sourceStartSec) || 0;
        }
        if (update.sourceEndSec !== undefined) {
            // 同 setClipSourceRange：不得钳制到 ≥0（倒放窗口合法含负值）。
            const value = Number(update.sourceEndSec);
            clip.sourceEndSec = Number.isFinite(value) ? value : clip.sourceEndSec;
        }
        if (update.fadeInSec !== undefined) {
            clip.fadeInSec = Math.max(0, Number(update.fadeInSec) || 0);
        }
        if (update.fadeOutSec !== undefined) {
            clip.fadeOutSec = Math.max(0, Number(update.fadeOutSec) || 0);
        }
        if (update.fadeInShape !== undefined) {
            clip.fadeInShape = update.fadeInShape;
        }
        if (update.fadeInDir !== undefined) {
            clip.fadeInDir = Math.min(1, Math.max(-1, Number(update.fadeInDir) || 0));
        }
        if (update.fadeOutShape !== undefined) {
            clip.fadeOutShape = update.fadeOutShape;
        }
        if (update.fadeOutDir !== undefined) {
            clip.fadeOutDir = Math.min(1, Math.max(-1, Number(update.fadeOutDir) || 0));
        }
        if (update.autoFadeInSec !== undefined) {
            clip.autoFadeInSec = Math.max(0, Number(update.autoFadeInSec) || 0);
        }
        if (update.autoFadeOutSec !== undefined) {
            clip.autoFadeOutSec = Math.max(0, Number(update.autoFadeOutSec) || 0);
        }
        if (update.reversed !== undefined) {
            // 方向翻转（且本请求未显式指定源窗口）时镜像后端换算，保持
            // 消费窗口不变 —— 权威载荷到达前波形不跳变。
            if (
                update.reversed !== clip.reversed &&
                update.sourceStartSec === undefined &&
                update.sourceEndSec === undefined
            ) {
                const rate =
                    Number.isFinite(clip.playbackRate) && clip.playbackRate > 1e-6
                        ? clip.playbackRate
                        : 1;
                const mediaTotal = resolveLoopMediaDurationSec({
                    durationFrames: clip.durationFrames,
                    sourceSampleRate: clip.sourceSampleRate,
                    durationSec: clip.durationSec,
                });
                flipSourceWindowForDirection(
                    clip,
                    Math.max(0, clip.lengthSec) * rate,
                    mediaTotal > 0 ? mediaTotal : null,
                );
            }
            clip.reversed = Boolean(update.reversed);
        }
        if (update.loopEnabled !== undefined) {
            clip.loopEnabled = Boolean(update.loopEnabled);
        }
        updateTakesFromFlatWithSync(state, clip);
    }
}

function parseSelectClipRemoteArg(
    arg:
        | string
        | null
        | {
              clipId: string | null;
              preserveTrackFocus?: boolean;
          },
): { clipId: string | null; preserveTrackFocus: boolean } {
    if (typeof arg === "object" && arg !== null && "clipId" in arg) {
        return {
            clipId: arg.clipId,
            preserveTrackFocus: Boolean(arg.preserveTrackFocus),
        };
    }
    return {
        clipId: arg,
        preserveTrackFocus: false,
    };
}

function parseTimelineClipTake(take: TimelineClipTake): ClipTakeInfo {
    return {
        id: take.id,
        name: take.name ?? "",
        gain: clamp(Number(take.gain ?? 1), 0, 4),
        sourcePath: take.source_path,
        sourcePathRelative: take.source_path_relative,
        durationSec: Number(take.duration_sec ?? 0) || undefined,
        durationFrames: take.duration_frames,
        sourceSampleRate: take.source_sample_rate,
        sourceStartSec: Number(take.source_start_sec ?? 0) || 0,
        sourceEndSec: Number.isFinite(Number(take.source_end_sec))
            ? Number(take.source_end_sec)
            : 0,
        playbackRate: take.playback_rate != null ? clamp(Number(take.playback_rate), 0.1, 10) : 1,
        reversed: Boolean(take.reversed),
        loopEnabled: Boolean(take.loop_enabled),
        midiNoteData: take.midi_note_data?.map((n) => ({
            startSec: n.start_sec,
            endSec: n.end_sec,
            note: n.note,
            velocity: n.velocity,
            channel: n.channel ?? 0,
        })),
        midiFillGaps: take.midi_fill_gaps ?? false,
    };
}

function parseClipTakes(clip: TimelineClip, flat: ClipInfo): ClipTakeInfo[] {
    if (Array.isArray(clip.takes) && clip.takes.length > 0) {
        return clip.takes.map(parseTimelineClipTake);
    }
    // 旧后端/过渡载荷：用 flat 投影合成一个 take。
    const clipRate = clamp(Number(clip.clip_playback_rate ?? 1) || 1, 0.1, 10);
    return [
        {
            id: clip.active_take_id ?? `${clip.id}_take_1`,
            name: flat.name,
            gain: flat.gain,
            sourcePath: flat.sourcePath,
            sourcePathRelative: undefined,
            durationSec: flat.durationSec,
            durationFrames: flat.durationFrames,
            sourceSampleRate: flat.sourceSampleRate,
            sourceStartSec: flat.sourceStartSec,
            sourceEndSec: flat.sourceEndSec,
            playbackRate: clamp(flat.playbackRate / clipRate, 0.1, 10),
            reversed: flat.reversed,
            loopEnabled: flat.loopEnabled,
            midiNoteData: flat.midiNoteData,
            midiFillGaps: flat.midiFillGaps ?? false,
        },
    ];
}

function getClipRateMultiplier(clip: ClipInfo): number {
    const rate = Number(clip.clipPlaybackRate ?? 1);
    return Number.isFinite(rate) && rate > 0 ? rate : 1;
}

/**
 * 方向翻转（正放 ↔ 倒放）时保持**消费内容**不变的源窗口/锚点换算。
 *
 * 与后端 `flip_direction_source_window` 逐字段同口径，必须在改写
 * `reversed` **之前**调用（内部按"翻转前方向"读取消费窗口）：
 *   - 非 Loop 正放消费窗口 [ss, ss+span)、倒放 [se−span, se)。派生窗口
 *     模型下非锚定方向的存储字段可能陈旧，直接翻布尔会让消费内容跳变
 *     （如裁剪过的 Clip 倒放后播到陈旧 se 所指的文件末段）；
 *   - Loop 的字段承载回绕锚点（引擎正放自 mod(ss, D) 升、倒放自
 *     mod(se, D) 降），换算以原方向消费区间为准：翻为倒放锚定消费
 *     终点 mod(ss + span, D)（自该点下降覆盖原区间的镜像），翻为正放
 *     锚定消费起点 mod(se − span, D)。
 */
function flipSourceWindowForDirection(
    fields: {
        reversed: boolean;
        loopEnabled: boolean;
        sourceStartSec: number;
        sourceEndSec: number;
    },
    spanSec: number,
    mediaTotalSec: number | null,
): void {
    if (fields.loopEnabled) {
        if (mediaTotalSec != null && mediaTotalSec > 0) {
            if (fields.reversed) {
                // 翻为正放：正放自锚点上升 → 新锚 = 原倒放消费起点。
                fields.sourceStartSec = modEuclid(fields.sourceEndSec - spanSec, mediaTotalSec);
            } else {
                // 翻为倒放：倒放自锚点下降 → 新锚 = 原正放消费终点。
                fields.sourceEndSec = modEuclid(fields.sourceStartSec + spanSec, mediaTotalSec);
            }
        } else if (fields.reversed) {
            // 媒体时长未知：退化为原始字段直算（引擎侧回绕兜底）。
            fields.sourceStartSec = fields.sourceEndSec - spanSec;
        } else {
            fields.sourceEndSec = fields.sourceStartSec + spanSec;
        }
        return;
    }
    if (fields.reversed) {
        // 翻为正放：ss := se − span（原倒放消费窗口的起点）。
        fields.sourceStartSec = fields.sourceEndSec - spanSec;
    } else {
        // 翻为倒放：se := ss + span（原正放消费窗口的终点）。
        fields.sourceEndSec = fields.sourceStartSec + spanSec;
    }
}

/** [`flipSourceWindowForDirection`] 的 Take 封装：span 按该 Take 的组合消费
 * 速率（Clip 倍率 × Take 速率）计算。 */
function flipTakeSourceWindowForDirection(
    take: ClipTakeInfo,
    lengthSec: number,
    clipRate: number,
): void {
    const clipRateNum = Number.isFinite(clipRate) && clipRate > 1e-6 ? clipRate : 1;
    const takeRate =
        Number.isFinite(take.playbackRate) && take.playbackRate > 1e-6 ? take.playbackRate : 1;
    const span = Math.max(0, Number(lengthSec) || 0) * clipRateNum * takeRate;
    const mediaTotal = resolveLoopMediaDurationSec({
        durationFrames: take.durationFrames,
        sourceSampleRate: take.sourceSampleRate,
        durationSec: take.durationSec,
    });
    flipSourceWindowForDirection(take, span, mediaTotal > 0 ? mediaTotal : null);
}

function updateActiveTakeFromFlat(clip: ClipInfo): void {
    const takes = clip.takes ?? [];
    const take = takes.find((entry) => entry.id === clip.activeTakeId) ?? takes[0];
    if (!take) return;
    take.gain = clip.gain;
    take.sourcePath = clip.sourcePath;
    take.sourcePathRelative = clip.sourcePathRelative;
    take.durationSec = clip.durationSec;
    take.durationFrames = clip.durationFrames;
    take.sourceSampleRate = clip.sourceSampleRate;
    take.sourceStartSec = clip.sourceStartSec;
    take.sourceEndSec = clip.sourceEndSec;
    take.playbackRate = clamp(clip.playbackRate / getClipRateMultiplier(clip), 0.1, 10);
    take.reversed = clip.reversed;
    take.loopEnabled = clip.loopEnabled;
    take.midiNoteData = clip.midiNoteData;
    take.midiFillGaps = clip.midiFillGaps;
}

/**
 * 与后端 `patch_clip_state` 的“同步编辑所有 Take”语义对齐：开启该设置时，
 * 内容级乐观编辑（增益/源窗口/速率/倒放/Loop）需要镜像到全部 Take，
 * 否则多 Take 展示下 inactive lane 的拖拽预览与后端口径短暂分叉
 * （fulfilled 快照才会收敛）。`playbackRate` 平铺值是组合有效速率，
 * 写入各 Take 前按倍率反推 —— 与后端 from_clip 口径一致。
 */
function updateTakesFromFlatWithSync(state: SessionState, clip: ClipInfo): void {
    updateActiveTakeFromFlat(clip);
    if (!state.syncEditsAcrossTakes) return;
    const takes = clip.takes ?? [];
    if (takes.length <= 1) return;
    const rateMultiplier = getClipRateMultiplier(clip);
    for (const take of takes) {
        take.gain = clip.gain;
        take.sourceStartSec = clip.sourceStartSec;
        take.sourceEndSec = clip.sourceEndSec;
        take.playbackRate = clamp(clip.playbackRate / rateMultiplier, 0.1, 10);
        take.reversed = clip.reversed;
        take.loopEnabled = clip.loopEnabled;
    }
}

function applyActiveTakeToFlat(clip: ClipInfo, take: ClipTakeInfo): void {
    clip.gain = take.gain;
    clip.sourcePath = take.sourcePath;
    clip.sourcePathRelative = take.sourcePathRelative;
    clip.durationSec = take.durationSec;
    clip.durationFrames = take.durationFrames;
    clip.sourceSampleRate = take.sourceSampleRate;
    clip.sourceStartSec = take.sourceStartSec;
    clip.sourceEndSec = take.sourceEndSec;
    clip.playbackRate = getClipRateMultiplier(clip) * take.playbackRate;
    clip.reversed = take.reversed;
    clip.loopEnabled = take.loopEnabled;
    clip.midiNoteData = take.midiNoteData;
    clip.midiNoteCount = take.midiNoteData?.length;
    clip.midiFillGaps = take.midiFillGaps ?? false;
}

/**
 * 应用权威时间轴，但保留前端当前播放光标。
 *
 * 后端 `TimelineState.playhead_sec` 只在显式 seek / transport 操作时同步；
 * 普通 take 管理命令返回的全量快照可能携带旧值。切换/管理 take 不应该
 * 改变播放位置，因此这里在覆写后恢复本地光标。
 */
/**
 * DAW 惯例：分割后选中右段、取消左段。
 *
 * 左段继承被分割 clip 的原 id（选择因此"残留"在左段上），右段为后端新建
 * clip，经 payload.created_clip_ids 按输入顺序返回。这里把单选/多选迁移到
 * 右段：
 * - 多选中未被分割的成员保持选中；
 * - 原单选若属于被分割的 clip，则映射到其对应右段（输入与右段按顺序
 *   一一对应；数量不一致时回退到第一个右段）。
 */
function applySplitSelection(
    state: SessionState,
    action: { meta: { arg: { clipId?: string; clipIds?: string[] } } },
    payload: { created_clip_ids?: string[] | null } & TimelineState,
) {
    const prevSelectedClipId = state.selectedClipId;
    const prevMultiSelectedClipIds = state.multiSelectedClipIds;
    applyTimelineState(state, payload, { force: true });

    const arg = action.meta.arg ?? {};
    const argIds = arg.clipIds ?? (arg.clipId ? [arg.clipId] : []);
    const splitOriginals = new Set(argIds);
    const rightIds = (payload.created_clip_ids ?? []).filter((id) =>
        state.clips.some((clip) => clip.id === id),
    );
    if (rightIds.length === 0) {
        return;
    }
    state.multiSelectedClipIds = [
        ...prevMultiSelectedClipIds.filter((id) => !splitOriginals.has(id)),
        ...rightIds,
    ];
    if (prevSelectedClipId && splitOriginals.has(prevSelectedClipId)) {
        const selectedIndex = argIds.indexOf(prevSelectedClipId);
        state.selectedClipId =
            selectedIndex >= 0 && argIds.length === rightIds.length
                ? rightIds[selectedIndex]
                : rightIds[0];
    }
}

function applyTimelineStatePreservingPlayhead(state: SessionState, timeline: TimelineState): void {
    const playheadSec = state.playheadSec;
    applyTimelineState(state, timeline, { force: true });
    state.playheadSec = Math.max(0, Number(playheadSec ?? 0) || 0);
}

// ── Take 乐观切换的回滚支持 ─────────────────────────────────────────────
// pending 阶段会本地切换 activeTakeId 并物化平铺投影；后端失败（rejected 或
// fulfilled ok:false）时若无回滚，UI 显示与音频引擎播放的内容将持续分裂，
// 直到下一次全量 fetchTimeline。这里按 clipId 记录切换前快照供恢复。

type TakeOptimisticFlat = Pick<
    ClipInfo,
    | "gain"
    | "sourcePath"
    | "sourcePathRelative"
    | "durationSec"
    | "durationFrames"
    | "sourceSampleRate"
    | "sourceStartSec"
    | "sourceEndSec"
    | "playbackRate"
    | "clipPlaybackRate"
    | "reversed"
    | "loopEnabled"
    | "midiNoteData"
    | "midiNoteCount"
    | "midiFillGaps"
>;

const pendingTakeRollbacks = new Map<
    string,
    { activeTakeId: string | undefined; flat: TakeOptimisticFlat }
>();

const TAKE_FLAT_KEYS = [
    "gain",
    "sourcePath",
    "sourcePathRelative",
    "durationSec",
    "durationFrames",
    "sourceSampleRate",
    "sourceStartSec",
    "sourceEndSec",
    "playbackRate",
    "clipPlaybackRate",
    "reversed",
    "loopEnabled",
    "midiNoteData",
    "midiNoteCount",
    "midiFillGaps",
] as const;

function snapshotTakeValue(value: unknown): unknown {
    // captureTakeRollback 在 Immer producer 内运行：对象/数组字段（如
    // midiNoteData）读到的是 draft 代理，存进模块级 Map 后会在 producer
    // 结束时被吊销；回滚时把吊销代理写回 state 会让 Immer finalization
    // 抛 "Cannot perform 'get' on a proxy that has been revoked"。
    // 这里做一次脱壳拷贝（take 快照字段均为 JSON 数据）。
    if (value === null || typeof value !== "object") return value;
    return JSON.parse(JSON.stringify(value)) as unknown;
}

function captureTakeRollback(state: SessionState, clipIds: readonly string[]): void {
    for (const clipId of clipIds) {
        const clip = state.clips.find((entry) => entry.id === clipId);
        if (!clip) continue;
        // 同一 clip 已有待回滚快照时保留**最早**的那份：快速连续切换两次且
        // 都失败时，第二次失败后仍能回到最初的 take，而不是中间态。
        if (pendingTakeRollbacks.has(clipId)) continue;
        const flat = {} as TakeOptimisticFlat;
        for (const key of TAKE_FLAT_KEYS) {
            // @ts-expect-error -- 按同构键组浅拷贝，字段集合由类型约束保证一致
            flat[key] = snapshotTakeValue(clip[key]);
        }
        pendingTakeRollbacks.set(clipId, { activeTakeId: clip.activeTakeId, flat });
    }
}

function restoreTakeRollback(state: SessionState, clipIds: readonly string[]): void {
    for (const clipId of clipIds) {
        const snapshot = pendingTakeRollbacks.get(clipId);
        if (!snapshot) continue;
        pendingTakeRollbacks.delete(clipId);
        const clip = state.clips.find((entry) => entry.id === clipId);
        if (!clip) continue;
        clip.activeTakeId = snapshot.activeTakeId;
        Object.assign(clip, snapshot.flat);
    }
}

function clearTakeRollback(clipIds: readonly string[]): void {
    for (const clipId of clipIds) {
        pendingTakeRollbacks.delete(clipId);
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
 *               `adoptPlayhead: true` 时采纳载荷的 playhead_sec —— 仅限工程生命
 *               周期载入（打开/新建/导入）与撤销/重做（检查点快照即回退/
 *               恢复后的后端实际光标，视觉编辑点必须跟随）等播放头权威流程。
 *               编辑类命令不得传入：它们携带的 playhead_sec 是编辑未触及的
 *               旧值（播放期间停留在本次播放的起始位置），采纳会让光标跳变。
 */
function applyTimelineState(
    state: SessionState,
    timeline: TimelineState,
    opts?: { force?: boolean; preserveProjectNotes?: boolean; adoptPlayhead?: boolean },
) {
    // 交互锁守卫：拖动/滑动期间跳过非强制的全量覆写
    if (state._interactionLockCount > 0 && !opts?.force) {
        return;
    }

    applyTimelineTracksOnly(state, timeline);

    state.clips = timeline.clips.map((clip: TimelineClip) => {
        const parsed = {
            id: clip.id,
            trackId: clip.track_id,
            name: clip.name,
            startSec: Number(clip.start_sec ?? 0),
            lengthSec: Math.max(0.0, Number(clip.length_sec ?? 1)),
            color: normalizeClipColor(clip.color),
            sourcePath: clip.source_path,
            sourcePathRelative: clip.source_path_relative,
            durationSec: Number(clip.duration_sec ?? 0) || undefined,
            durationFrames: clip.duration_frames,
            sourceSampleRate: clip.source_sample_rate,
            gain: clamp(Number(clip.gain ?? 1), 0, 4),
            muted: Boolean(clip.muted),
            // Allow negative sourceStartSec to represent leading silence (slip-edit past source start).
            sourceStartSec: Number(clip.source_start_sec ?? 0) || 0,
            sourceEndSec: (() => {
                const raw = Number(clip.source_end_sec);
                // 仅当字段**缺失/非法**时才回退默认值。
                // 注意两点：
                // 1. 不得 Math.max(0, …)：倒放 Clip 的消费窗口锚定 se，
                //    se<0（整窗在媒体下方的静音段）是合法状态 —— 钳零会把
                //    分割出的静音段凭空拉回媒体域；
                // 2. 不得把 se==0 当作"到媒体末尾"的旧哨兵改写为 duration：
                //    该哨兵语义已在后端 open_project 迁移中展开为真实时长，
                //    此处的二次改写会摧毁合法的 0/负值窗口。
                if (!Number.isFinite(raw)) {
                    return (
                        Number(clip.duration_sec ?? 0) || Math.max(0, Number(clip.length_sec ?? 1))
                    );
                }
                return raw;
            })(),
            playbackRate:
                clip.playback_rate != null
                    ? clamp(Number(clip.playback_rate), 0.1, 10)
                    : (state.clips.find((c) => c.id === clip.id)?.playbackRate ?? 1),
            clipPlaybackRate: clamp(Number(clip.clip_playback_rate ?? 1) || 1, 0.1, 10),
            reversed: Boolean(clip.reversed),
            loopEnabled: Boolean(clip.loop_enabled),
            // SnapOffset（吸附偏移）：旧工程缺失时自动补齐为 0。
            snapOffsetSec: Math.max(0, Number(clip.snap_offset_sec ?? 0) || 0),
            fadeInSec: Math.max(0, Number(clip.fade_in_sec ?? 0)),
            fadeOutSec: Math.max(0, Number(clip.fade_out_sec ?? 0)),
            // 后端未提供形状字段时（极旧开发版载荷），与新建默认一致取快起。
            fadeInShape: Number.isFinite(Number(clip.fade_in_shape))
                ? Number(clip.fade_in_shape)
                : 1,
            fadeOutShape: Number.isFinite(Number(clip.fade_out_shape))
                ? Number(clip.fade_out_shape)
                : 1,
            fadeInDir: Math.min(1, Math.max(-1, Number(clip.fade_in_dir ?? 0) || 0)),
            fadeOutDir: Math.min(1, Math.max(-1, Number(clip.fade_out_dir ?? 0) || 0)),
            autoFadeInSec: Math.max(0, Number(clip.auto_fade_in_sec ?? 0) || 0),
            autoFadeOutSec: Math.max(0, Number(clip.auto_fade_out_sec ?? 0) || 0),
            formantMorph: clip.formant_morph
                ? {
                      enabled: Boolean(clip.formant_morph.enabled),
                      targetF1Hz: Number(clip.formant_morph.target_f1_hz ?? 800),
                      targetF2Hz: Number(clip.formant_morph.target_f2_hz ?? 1400),
                      strength: Number(clip.formant_morph.strength ?? 0.5),
                  }
                : undefined,
            midiNoteCount: clip.midi_note_count,
            midiNoteData: clip.midi_note_data?.map(
                (n: {
                    start_sec: number;
                    end_sec: number;
                    note: number;
                    velocity: number;
                    channel?: number;
                }) => ({
                    startSec: n.start_sec,
                    endSec: n.end_sec,
                    note: n.note,
                    velocity: n.velocity,
                    channel: n.channel ?? 0,
                }),
            ),
            midiFillGaps: clip.midi_fill_gaps ?? false,
            groupId: clip.group_id ?? undefined,
        };

        const takes = parseClipTakes(clip, parsed);
        return {
            ...parsed,
            takes,
            activeTakeId:
                clip.active_take_id && takes.some((take) => take.id === clip.active_take_id)
                    ? clip.active_take_id
                    : (takes[0]?.id ?? `${clip.id}_take_1`),
        };
    });
    state.clipFormantStatus = Object.fromEntries(
        Object.entries(state.clipFormantStatus).filter(([clipId]) =>
            state.clips.some((clip) => clip.id === clipId),
        ),
    ) as Record<string, "ready" | "rebuilding" | "failed">;
    state.clipFormantAnalysis = Object.fromEntries(
        Object.entries(state.clipFormantAnalysis).filter(([clipId]) =>
            state.clips.some((clip) => clip.id === clipId),
        ),
    );
    if (
        state.clipFormantToolWindow.clipId &&
        !state.clips.some((clip) => clip.id === state.clipFormantToolWindow.clipId)
    ) {
        state.clipFormantToolWindow.open = false;
        state.clipFormantToolWindow.clipId = null;
    }

    state.selectedTrackId = timeline.selected_track_id;
    state.selectedClipId = timeline.selected_clip_id;
    // 与 Tempo Map 变化点一致的 BPM 范围（10-960）。
    state.bpm = clamp(Number(timeline.bpm ?? state.bpm), 10, 960);
    // 播放头归传输层（轮询 / seek / stop_audio / 显式跳转）所有。编辑命令
    // 返回的全量快照携带的 playhead_sec 是编辑未触及的旧值（播放期间停留
    // 在本次播放的起始位置），无条件采纳会让光标跳回起点——尤其在播放中
    // 编辑、或引擎等待重渲染短暂报告未播放时。只有显式 opt-in 的
    // 工程生命周期载入流程才采纳。需要移动光标的操作（如粘贴跳转
    // pasteEndSec）在本函数之后显式设置，不受影响。
    if (opts?.adoptPlayhead) {
        state.playheadSec = Math.max(0, Number(timeline.playhead_sec ?? 0));
    }
    state.projectSec = Math.max(4, Number(timeline.project_sec ?? state.projectSec));
    state.disabledGroupIds = Array.isArray(timeline.disabled_group_ids)
        ? [...timeline.disabled_group_ids]
        : [];

    const project = timeline.project as
        | {
              name?: string;
              path?: string | null;
              dirty?: boolean;
              recent?: string[];
              notes_markdown?: string;
              base_scale?: string;
              use_custom_scale?: boolean;
              custom_scale?: {
                  id?: string;
                  name?: string;
                  notes?: number[];
              } | null;
              beats_per_bar?: number;
              time_signature_denominator?: number;
              grid_size?: string;
              stretch_algorithm_override?: StretchAlgorithmOption | null;
              hifigan_mel_stretch_override?: boolean | null;
          }
        | undefined;
    if (project) {
        const nextBaseScaleRaw = String(project.base_scale ?? state.project.baseScale);
        const nextBaseScale = (SCALE_KEYS as readonly string[]).includes(nextBaseScaleRaw)
            ? (nextBaseScaleRaw as typeof state.project.baseScale)
            : "C";
        const nextBeatsPerBar = clamp(
            Number(project.beats_per_bar ?? state.project.beatsPerBar),
            1,
            32,
        );
        const nextTimeSignatureDenominator = (TEMPO_DENOMINATORS as readonly number[]).includes(
            Number(project.time_signature_denominator),
        )
            ? Number(project.time_signature_denominator)
            : clampDenominator(state.project.timeSignatureDenominator);
        const nextGridSizeRaw = String(project.grid_size ?? state.project.gridSize);
        const nextGridSize = VALID_GRID_SIZES.has(nextGridSizeRaw as GridSize)
            ? (nextGridSizeRaw as GridSize)
            : "1/4";

        state.project = {
            name: String(project.name ?? state.project.name ?? "Untitled"),
            path: project.path === undefined ? state.project.path : (project.path ?? null),
            dirty: Boolean(project.dirty),
            recent: Array.isArray(project.recent) ? project.recent : state.project.recent,
            notesMarkdown:
                opts?.preserveProjectNotes === false
                    ? String(project.notes_markdown ?? "")
                    : state.project.notesMarkdown,
            baseScale: nextBaseScale,
            useCustomScale: Boolean(project.use_custom_scale),
            customScale: project.custom_scale
                ? sanitizeCustomScalePreset(project.custom_scale)
                : null,
            beatsPerBar: nextBeatsPerBar,
            timeSignatureDenominator: nextTimeSignatureDenominator,
            gridSize: nextGridSize,
            stretchAlgorithmOverride:
                project.stretch_algorithm_override === undefined
                    ? state.project.stretchAlgorithmOverride
                    : (project.stretch_algorithm_override ?? null),
            hifiganMelStretchOverride:
                project.hifigan_mel_stretch_override === undefined
                    ? state.project.hifiganMelStretchOverride
                    : (project.hifigan_mel_stretch_override ?? null),
        };
        state.beats = nextBeatsPerBar;
        state.grid = nextGridSize;
    }

    // Tempo Map（后端载荷始终带 tempo_map 字段；旧后端无此字段时保持现值）。
    // 在工程元数据之后解析：初始点缺失时用工程音阶物化（初始点即工程基准记录）。
    const rawTempoMap = (timeline as unknown as { tempo_map?: unknown }).tempo_map;
    if (rawTempoMap !== undefined) {
        const projectScale: ScaleLike | null =
            state.project.useCustomScale && state.project.customScale
                ? state.project.customScale.notes
                : state.project.baseScale;
        state.tempoMap = fromBackendTempoMap(rawTempoMap, state.bpm, state.beats || 4, {
            projectScale: projectScale ?? undefined,
            projectScaleName: state.project.useCustomScale
                ? (state.project.customScale?.name ?? undefined)
                : undefined,
            projectDenominator: state.project.timeSignatureDenominator,
        });
        if (state.tempoMap) {
            // 0 位置点与工程基准 BPM/拍号保持一致（后端同步保证）。
            const first = state.tempoMap.points[0];
            state.bpm = clamp(first.bpm, 10, 960);
            const firstSig = first.timeSignature ?? { numerator: 4, denominator: 4 };
            state.beats = Math.min(32, Math.max(1, Math.round(firstSig.numerator)));
            state.project.timeSignatureDenominator = clampDenominator(firstSig.denominator);
        }
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
    // 清理已删除 clip 的多选 ID，避免删除轨道组后残留无效的 clip 引用
    if (state.multiSelectedClipIds.length > 0) {
        state.multiSelectedClipIds = state.multiSelectedClipIds.filter((id) =>
            availableClipIds.has(id),
        );
    }

    const nextWaveforms: Record<string, WaveformPreview> = {};
    const nextPitchRanges: Record<string, { min: number; max: number }> = {};
    for (const clip of timeline.clips) {
        const clipId = clip.id;
        nextWaveforms[clipId] = (clip.waveform_preview ?? []) as WaveformPreview;
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
            volume: 1,

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
        reversed: false,
        // 乐观创建的导入 Clip：Loop 跟随"为新的音频块启用循环"设置
        //（默认开启；后端权威载荷返回后会覆盖该值）。
        loopEnabled: state.loopNewClipsEnabled !== false,
        snapOffsetSec: 0,
        fadeInSec: 0,
        fadeOutSec: 0,
        // 新建默认曲线 = 快起（REAPER Fast Start；dir 锚点 0）。
        fadeInShape: 1,
        fadeOutShape: 1,
        fadeInDir: 0,
        fadeOutDir: 0,
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
    primaryTimeUnit: DEFAULT_PRIMARY_TIME_UNIT,
    secondaryTimeUnit: DEFAULT_SECONDARY_TIME_UNIT,
    rulerLabelSpacingPx: DEFAULT_RULER_LABEL_SPACING_PX,
    showPlayheadTimeInTrackHeader: true,
    paramEditorSyncTimeline: true,

    autoCrossfadeEnabled: true,
    showAllTakes: true,
    syncEditsAcrossTakes: true,
    loopNewClipsEnabled: true,
    splitTransitionEnabled: true,
    splitTransitionMode: "overlap",
    splitTransitionDurationUnit: "seconds",
    splitTransitionDurationSec: 0.01,
    splitTransitionDurationPercent: 1,
    splitTransitionCurve: "keep",
    splitTransitionOverlapCrossfade: "auto",
    snapEnabled: true,
    timelineSnap: createDefaultTimelineSnapSettings(),
    tempoMap: null,
    tempoMapVisible: true,
    pitchSnapEnabled: false,
    pitchSnapUnit: "semitone",
    pitchSnapScale: "C",
    pitchSnapToleranceCents: 0,
    scaleHighlightMode: "off",
    playheadZoomEnabled: false,
    autoScrollEnabled: false,
    ignoreGrouping: false,
    rippleMode: "off",
    disabledGroupIds: [],
    paramEditorSeekPlayheadEnabled: true,
    paramEditorTimelineClickSelectTrackEnabled: true,
    showClipboardPreview: true,
    showParamValuePopup: true,
    selectDragDirection: "y-only" as DragDirection,
    drawDragDirection: "free" as DrawDragDirection,
    lineVibratoDragDirection: "free" as DrawDragDirection,
    edgeSmoothnessPercent: 0,
    lockParamLinesEnabled: true,
    quickSearchAutoNormalizeEnabled: false,
    visibleReferenceRootTrackIds: [],
    defaultStretchAlgorithm: "signalsmith",
    defaultHifiganMelStretch: true,
    ortEp: "auto",
    gpuDeviceId: 0,
    ortDeviceId: null,
    autoBackgroundRender: true,

    paramsEpoch: 0,
    playbackRateVersion: 0,

    playheadSec: 0,
    pendingPlayheadRevealSec: null,
    tracks: [
        {
            id: "track_main",
            name: "Main",
            muted: false,
            solo: false,
            volume: 1,
            // 与后端 TRACK_COLOR_PALETTE[0]（TimelineState::default 的初始
            // Main 轨道色）保持一致：灰色。前端最初始状态（未加载/未新建
            // 工程）必须与"新建工程"看到同一个颜色，否则轨道头缺色回退
            // 会显示成旧版高亮蓝。
            color: "#74787e",

            composeEnabled: false,
            pitchAnalysisAlgo: "nsf_hifigan_onnx",
        },
    ],
    trackMeters: {},
    clips: [],
    selectedTrackId: "track_main",
    selectedClipId: null,
    multiSelectedClipIds: [],
    clipAutomation: {},
    selectedPointId: null,
    clipWaveforms: {},
    clipPitchRanges: {},
    clipPitchCurves: {},
    clipFormantStatus: {},
    clipFormantAnalysis: {},
    clipFormantToolWindow: {
        open: false,
        clipId: null,
        x: 160,
        y: 120,
        hasMoved: false,
    },

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
        gpuBackend: "",
    },

    selectedTrackSummary: {
        trackId: null,
        clipCount: 0,
        waveformPreview: [],
        pitchRange: { min: -24, max: 24 },
    },

    customScalePresets: [],
    project: {
        name: "Untitled",
        path: null,
        dirty: false,
        recent: [],
        notesMarkdown: "",
        baseScale: "C",
        useCustomScale: false,
        customScale: null,
        beatsPerBar: 4,
        timeSignatureDenominator: 4,
        gridSize: "1/4",
        stretchAlgorithmOverride: null,
        hifiganMelStretchOverride: null,
    },

    busy: false,
    status: "Ready",
    vocalShifterSkippedFilesDialog: null,
    reaperSkippedFilesDialog: null,
    saveVersionConflictDialog: null,
    _interactionLockCount: 0,
    _latestHistoryOpRequestId: null,
    _latestEditRequestId: null,
};

export {
    undoRemote,
    redoRemote,
    newProjectRemote,
    openProjectFromDialog,
    openProjectFromPath,
    openProjectFromPathForced,
    pickProjectToImport,
    importProjectFromPath,
    openVocalShifterFromDialog,
    openVocalShifterFromPath,
    openReaperFromDialog,
    openReaperFromPath,
    saveProjectRemote,
    saveProjectAsRemote,
    saveProjectToPathRemote,
    setProjectBaseScaleRemote,
    setProjectCustomScaleRemote,
    setProjectStretchSettingsRemote,
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
    pasteTimelineClipboardRemote,
    removeClipRemote,
    removeClipsRemote,
    moveClipRemote,
    moveClipsRemote,
    duplicateClipsBulkRemote,
    duplicateTrackRemote,
    setClipStateRemote,
    setClipsStateBulkRemote,
    setClipActiveTakeRemote,
    cycleClipTakesRemote,
    packClipsIntoTakesRemote,
    explodeClipTakesRemote,
    duplicateClipTakeRemote,
    removeClipTakeRemote,
    renameClipTakeRemote,
    setClipTakeReversedRemote,
    addClipTakeFromMediaRemote,
    replaceClipSourceRemote,
    replaceMidiClipDataRemote,
    splitClipRemote,
    splitClipsAtRemote,
    convertClipsToPitchReferenceRemote,
    updatePitchReferenceRemote,
    glueClipsRemote,
    selectClipRemote,
} from "./thunks/timelineThunks";

export { setTrackStateRemote, removeSelectedClipRemote } from "./thunks/trackThunks";

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
    exportAudioAdvanced,
    exportSeparated,
    pasteVocalShifterClipboard,
    pasteReaperClipboard,
} from "./thunks/audioThunks";

export {
    importAudioFromDialog,
    importAudioFromPath,
    importAudioAtPosition,
    importAudioFileAtPosition,
    importMidiAsClip,
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
        setTrackName(state, action: PayloadAction<{ trackId: string; name: string }>) {
            const track = state.tracks.find((entry) => entry.id === action.payload.trackId);
            if (track) {
                track.name = action.payload.name;
            }
        },
        setTrackMeters(state, action: PayloadAction<Record<string, TrackMeterInfo>>) {
            state.trackMeters = action.payload;
        },
        clearTrackMeters(state) {
            state.trackMeters = {};
        },
        checkpointHistory(state) {
            // 撤销/重做由后端权威管理（undo_timeline / redo_timeline 返回完整
            // 时间线快照）。前端不再维护平行的快照栈——旧实现的本检查点与
            // 后端检查点在深度与内容上都可能错位，撤销时会先乐观渲染出任意
            // 旧状态、再被后端快照纠正，造成轨道视图闪屏。此 action 现在
            // 仅负责：1) 标记工程已修改；2) 递增 paramsEpoch 让参数编辑器
            // 重新取数（如 shiftParam 系列直接写曲线后的刷新）。
            markProjectDirty(state.project);
            state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
        },
        /** 供录音等后端直接导入时间轴的命令同步完整快照。 */
        applyTimelinePayload(state, action: PayloadAction<TimelineState>) {
            // 后端直接导入的权威快照（如录音导入）：采纳其后端播放头。
            applyTimelineState(state, action.payload, { force: true, adoptPlayhead: true });
        },
        bumpParamsEpoch(state) {
            state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
        },
        setToolMode(state, action: PayloadAction<ToolMode>) {
            // 工具切换是纯视图状态：不修改工程内容。
            // 不入 undo 历史（否则 Ctrl+Z 会先回退一次"不存在的内容变更"），
            // 也不标记 project.dirty（否则仅切工具就会触发"未保存"退出确认）。
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
            // 与 Tempo Map 变化点一致的 BPM 范围（10-960）。
            state.bpm = clamp(action.payload, 10, 960);
        },
        setBeats(state, action: PayloadAction<number>) {
            state.beats = clamp(action.payload, 1, 32);
        },
        setGrid(state, action: PayloadAction<GridSize>) {
            state.grid = action.payload;
        },
        setPrimaryTimeUnit(state, action: PayloadAction<TimeUnit>) {
            if (VALID_TIME_UNITS.has(action.payload)) {
                state.primaryTimeUnit = action.payload;
            }
        },
        setSecondaryTimeUnit(state, action: PayloadAction<TimeUnitChoice>) {
            if (action.payload === "none" || VALID_TIME_UNITS.has(action.payload)) {
                state.secondaryTimeUnit = action.payload;
            }
        },
        setRulerLabelSpacingPx(state, action: PayloadAction<number>) {
            state.rulerLabelSpacingPx = Math.max(
                40,
                Math.min(320, Math.round(Number(action.payload) || 110)),
            );
        },
        setShowPlayheadTimeInTrackHeader(state, action: PayloadAction<boolean>) {
            state.showPlayheadTimeInTrackHeader = Boolean(action.payload);
        },
        setParamEditorSyncTimeline(state, action: PayloadAction<boolean>) {
            state.paramEditorSyncTimeline = Boolean(action.payload);
        },
        toggleAutoCrossfade(state) {
            state.autoCrossfadeEnabled = !state.autoCrossfadeEnabled;
        },
        toggleShowAllTakes(state) {
            state.showAllTakes = !state.showAllTakes;
        },
        toggleSyncEditsAcrossTakes(state) {
            state.syncEditsAcrossTakes = !state.syncEditsAcrossTakes;
        },
        toggleSplitTransition(state) {
            state.splitTransitionEnabled = !state.splitTransitionEnabled;
        },
        setSplitTransitionMode(state, action: PayloadAction<"fade" | "overlap">) {
            state.splitTransitionMode = action.payload;
        },
        setSplitTransitionDurationUnit(state, action: PayloadAction<"seconds" | "percent">) {
            state.splitTransitionDurationUnit = action.payload;
        },
        setSplitTransitionDurationSec(state, action: PayloadAction<number>) {
            state.splitTransitionDurationSec = clamp(Number(action.payload) || 0.01, 0.001, 10);
        },
        setSplitTransitionDurationPercent(state, action: PayloadAction<number>) {
            state.splitTransitionDurationPercent = clamp(Number(action.payload) || 1, 0.01, 100);
        },
        setSplitTransitionCurve(state, action: PayloadAction<SplitTransitionCurveType>) {
            state.splitTransitionCurve = action.payload;
        },
        setSplitTransitionOverlapCrossfade(state, action: PayloadAction<"auto" | "always">) {
            state.splitTransitionOverlapCrossfade = action.payload;
        },
        toggleSnap(state) {
            const next = !state.timelineSnap.enabled;
            state.timelineSnap.enabled = next;
            state.snapEnabled = next;
        },
        setTimelineSnapSettings(state, action: PayloadAction<Partial<TimelineSnapSettings>>) {
            state.timelineSnap = normalizeTimelineSnapSettings(state.timelineSnap, action.payload);
            state.snapEnabled = state.timelineSnap.enabled;
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
        setScaleHighlightMode(state, action: PayloadAction<"always" | "off">) {
            state.scaleHighlightMode = action.payload;
        },
        setTempoMap(state, action: PayloadAction<TempoMap | null>) {
            const projectScale: ScaleLike | null =
                state.project.useCustomScale && state.project.customScale
                    ? state.project.customScale.notes
                    : state.project.baseScale;
            state.tempoMap = normalizeTempoMap(action.payload, state.bpm, state.beats || 4, {
                projectScale: projectScale ?? undefined,
                projectScaleName: state.project.useCustomScale
                    ? (state.project.customScale?.name ?? undefined)
                    : undefined,
                projectDenominator: state.project.timeSignatureDenominator,
            });
            // 保持工程基准值（bpm / 拍号）与 0 位置点一致，删除 Tempo Map 后回退一致。
            if (state.tempoMap) {
                const first = state.tempoMap.points[0];
                state.bpm = clamp(first.bpm, 10, 960);
                const firstSig = first.timeSignature ?? { numerator: 4, denominator: 4 };
                state.beats = Math.min(32, Math.max(1, Math.round(firstSig.numerator)));
                state.project.timeSignatureDenominator = clampDenominator(firstSig.denominator);
            }
        },
        setTempoMapVisible(state, action: PayloadAction<boolean>) {
            state.tempoMapVisible = action.payload;
        },
        toggleTempoMapVisible(state) {
            state.tempoMapVisible = !state.tempoMapVisible;
        },
        upsertCustomScalePreset(state, action: PayloadAction<CustomScalePreset>) {
            const incoming = sanitizeCustomScalePreset(action.payload);
            const idx = state.customScalePresets.findIndex((preset) => preset.id === incoming.id);
            if (idx >= 0) {
                state.customScalePresets[idx] = incoming;
            } else {
                state.customScalePresets.push(incoming);
            }
        },
        removeCustomScalePreset(state, action: PayloadAction<string>) {
            const presetId = action.payload;
            state.customScalePresets = state.customScalePresets.filter(
                (preset) => preset.id !== presetId,
            );
        },
        togglePlayheadZoom(state) {
            state.playheadZoomEnabled = !state.playheadZoomEnabled;
        },
        toggleLockParamLines(state) {
            state.lockParamLinesEnabled = !state.lockParamLinesEnabled;
        },
        toggleQuickSearchAutoNormalize(state) {
            state.quickSearchAutoNormalizeEnabled = !state.quickSearchAutoNormalizeEnabled;
        },
        setDefaultStretchAlgorithm(state, action: PayloadAction<StretchAlgorithmOption>) {
            state.defaultStretchAlgorithm = action.payload;
        },
        setDefaultHifiganMelStretch(state, action: PayloadAction<boolean>) {
            state.defaultHifiganMelStretch = action.payload;
        },
        setOrtEp(state, action: PayloadAction<string>) {
            state.ortEp = action.payload;
        },
        setGpuDeviceId(state, action: PayloadAction<number>) {
            state.gpuDeviceId = action.payload;
        },
        setOrtDeviceId(state, action: PayloadAction<number | null>) {
            state.ortDeviceId = action.payload;
        },
        toggleAutoBackgroundRender(state) {
            state.autoBackgroundRender = !state.autoBackgroundRender;
        },
        setVisibleReferenceRootTrackIds(state, action: PayloadAction<string[]>) {
            state.visibleReferenceRootTrackIds = Array.from(
                new Set(
                    action.payload.filter(
                        (id): id is string => typeof id === "string" && id.trim().length > 0,
                    ),
                ),
            );
        },
        toggleVisibleReferenceRootTrackId(state, action: PayloadAction<string>) {
            const trackId = action.payload;
            if (state.visibleReferenceRootTrackIds.includes(trackId)) {
                state.visibleReferenceRootTrackIds = state.visibleReferenceRootTrackIds.filter(
                    (id) => id !== trackId,
                );
                return;
            }
            const track = state.tracks.find((t) => t.id === trackId);
            if (!track || track.parentId) return;
            let currentRootId: string | null = state.selectedTrackId;
            if (currentRootId) {
                let cursor: { id: string; parentId?: string | null } | undefined =
                    state.tracks.find((t) => t.id === currentRootId);
                while (cursor?.parentId) {
                    cursor = state.tracks.find((t) => t.id === cursor!.parentId);
                }
                currentRootId = cursor?.id ?? null;
            }
            if (trackId === currentRootId) return;
            state.visibleReferenceRootTrackIds = [...state.visibleReferenceRootTrackIds, trackId];
        },
        toggleAutoScroll(state) {
            state.autoScrollEnabled = !state.autoScrollEnabled;
        },
        toggleIgnoreGrouping(state) {
            state.ignoreGrouping = !state.ignoreGrouping;
        },
        /** 循环波纹编辑模式：off → track → all → off（对应 REAPER 波纹编辑按钮的三态切换）。 */
        cycleRippleMode(state) {
            state.rippleMode =
                state.rippleMode === "off" ? "track" : state.rippleMode === "track" ? "all" : "off";
        },
        setRippleMode(state, action: PayloadAction<"off" | "track" | "all">) {
            if (
                action.payload === "off" ||
                action.payload === "track" ||
                action.payload === "all"
            ) {
                state.rippleMode = action.payload;
            }
        },
        toggleGroupDisabledLocal(state, action: PayloadAction<string>) {
            const idx = state.disabledGroupIds.indexOf(action.payload);
            if (idx >= 0) {
                state.disabledGroupIds.splice(idx, 1);
            } else {
                state.disabledGroupIds.push(action.payload);
            }
        },
        toggleParamEditorSeekPlayhead(state) {
            state.paramEditorSeekPlayheadEnabled = !state.paramEditorSeekPlayheadEnabled;
        },
        toggleParamEditorTimelineClickSelectTrack(state) {
            state.paramEditorTimelineClickSelectTrackEnabled =
                !state.paramEditorTimelineClickSelectTrackEnabled;
        },
        toggleClipboardPreview(state) {
            state.showClipboardPreview = !state.showClipboardPreview;
        },
        toggleParamValuePopup(state) {
            state.showParamValuePopup = !state.showParamValuePopup;
        },
        cycleDragDirection(state, action: PayloadAction<"select" | "draw" | "vibrato">) {
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
        /** 设置/清除待执行的“聚焦播放光标”请求（见 pendingPlayheadRevealSec 注释）。 */
        setPendingPlayheadReveal(state, action: PayloadAction<number | null>) {
            state.pendingPlayheadRevealSec = action.payload;
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
        setProjectNotesMarkdown(state, action: PayloadAction<string>) {
            const next = action.payload;
            if (state.project.notesMarkdown === next) {
                return;
            }
            state.project.notesMarkdown = next;
            state.project.dirty = true;
        },
        closeVocalShifterSkippedFilesDialog(state) {
            state.vocalShifterSkippedFilesDialog = null;
        },
        closeReaperSkippedFilesDialog(state) {
            state.reaperSkippedFilesDialog = null;
        },
        closeSaveVersionConflictDialog(state) {
            state.saveVersionConflictDialog = null;
        },
        setSelectedClip(state, action: PayloadAction<string | null>) {
            state.selectedClipId = action.payload;
            state.selectedPointId = null;
            if (action.payload) {
                const nextTrackId = resolveTrackIdForClipSelection({
                    currentTrackId: state.selectedTrackId,
                    clips: state.clips,
                    clipId: action.payload,
                });
                if (nextTrackId !== state.selectedTrackId) {
                    state.selectedTrackId = nextTrackId;
                }
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
        /** 复制/剪切失败（系统剪贴板被占用等）时在状态栏给出可见反馈。 */
        setClipboardOperationFailed(state, action: PayloadAction<{ op: "copy" | "cut" }>) {
            state.status =
                action.payload.op === "cut" ? "Clipboard cut failed" : "Clipboard copy failed";
        },
        moveClipStart(state, action: PayloadAction<{ clipId: string; startSec: number }>) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (clip) {
                clip.startSec = Math.max(0, action.payload.startSec);
                // 拖动超出边界时自动扩展工程时长
                const clipEnd = clip.startSec + clip.lengthSec;
                if (clipEnd > state.projectSec) {
                    state.projectSec = Math.ceil(clipEnd);
                }
            }
        },
        moveClipTrack(state, action: PayloadAction<{ clipId: string; trackId: string }>) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (clip) {
                clip.trackId = action.payload.trackId;
            }
        },
        setClipLength(state, action: PayloadAction<{ clipId: string; lengthSec: number }>) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (clip) {
                clip.lengthSec = Math.max(0.0, action.payload.lengthSec);
                // 与后端及 applyOptimisticClipState 一致：偏移必须落在
                // [0, length] 内，否则留下越界的"幻影吸附目标"。
                if (clip.snapOffsetSec > clip.lengthSec) {
                    clip.snapOffsetSec = clip.lengthSec;
                }
            }
        },
        /** SnapOffset（吸附偏移）乐观更新：拖拽三角手柄实时预览。 */
        setClipSnapOffset(state, action: PayloadAction<{ clipId: string; snapOffsetSec: number }>) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            // 与后端一致：偏移必须落在 [0, length] 内。
            const offset = Math.max(0, Number(action.payload.snapOffsetSec) || 0);
            clip.snapOffsetSec = Math.min(offset, clip.lengthSec);
        },
        setClipPlaybackRate(
            state,
            action: PayloadAction<{ clipId: string; clipPlaybackRate: number }>,
        ) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            const takes = clip.takes ?? [];
            const activeTake = takes.find((entry) => entry.id === clip.activeTakeId) ?? takes[0];
            const takeRate = activeTake
                ? activeTake.playbackRate
                : clamp(clip.playbackRate / getClipRateMultiplier(clip), 0.1, 10);
            clip.clipPlaybackRate = clamp(action.payload.clipPlaybackRate, 0.1, 10);
            clip.playbackRate = clamp(clip.clipPlaybackRate * takeRate, 0.1, 10);
            // "同步编辑所有 Take"开启时，后端会把全部 take 的速率统一为
            // 有效速率 ÷ 倍率；乐观阶段同步镜像，否则拖拽期间 inactive lane
            // 按旧速率渲染、persist+fulfilled 才收敛（视觉瞬态失真）。
            if (state.syncEditsAcrossTakes && takes.length > 1) {
                const mirrored = clamp(clip.playbackRate / getClipRateMultiplier(clip), 0.1, 10);
                for (const take of takes) {
                    take.playbackRate = mirrored;
                }
            }
            state.playbackRateVersion = (Number(state.playbackRateVersion) || 0) + 1;
        },
        setClipSourceRange(
            state,
            action: PayloadAction<{
                clipId: string;
                sourceStartSec?: number;
                sourceEndSec?: number;
            }>,
        ) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            if (action.payload.sourceStartSec !== undefined) {
                clip.sourceStartSec = Number(action.payload.sourceStartSec) || 0;
            }
            if (action.payload.sourceEndSec !== undefined) {
                // 注意：**不得钳制到 ≥0**。倒放 Clip 的消费窗口锚定 se，
                // 合法状态包含 se<0（整窗在媒体下方的纯静音段）与 se>D
                // （前导静音）；此处一旦拍扁，渲染/音频/后续编辑全部基于
                // 被摧毁的窗口工作。
                const value = Number(action.payload.sourceEndSec);
                clip.sourceEndSec = Number.isFinite(value) ? value : clip.sourceEndSec;
            }

            // Slip 的实时预览写在 active-take 投影上；展开全部 Take 时，
            // 必须同步对应 lane，否则拖拽期间波形不会移动。
            updateActiveTakeFromFlat(clip);
            if (state.syncEditsAcrossTakes) {
                for (const take of clip.takes ?? []) {
                    if (action.payload.sourceStartSec !== undefined) {
                        take.sourceStartSec = clip.sourceStartSec;
                    }
                    if (action.payload.sourceEndSec !== undefined) {
                        take.sourceEndSec = clip.sourceEndSec;
                    }
                }
            }
        },
        setClipFades(
            state,
            action: PayloadAction<{
                clipId: string;
                fadeInSec?: number;
                fadeOutSec?: number;
                fadeInShape?: number;
                fadeOutShape?: number;
                fadeInDir?: number;
                fadeOutDir?: number;
            }>,
        ) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            if (action.payload.fadeInSec !== undefined) {
                clip.fadeInSec = Math.max(0, action.payload.fadeInSec);
            }
            if (action.payload.fadeOutSec !== undefined) {
                clip.fadeOutSec = Math.max(0, action.payload.fadeOutSec);
            }
            // 形状原样透传（REAPER 允许小数变体）；曲率夹紧到 [-1, 1]。
            if (action.payload.fadeInShape !== undefined) {
                clip.fadeInShape = action.payload.fadeInShape;
            }
            if (action.payload.fadeOutShape !== undefined) {
                clip.fadeOutShape = action.payload.fadeOutShape;
            }
            if (action.payload.fadeInDir !== undefined) {
                clip.fadeInDir = Math.min(1, Math.max(-1, action.payload.fadeInDir));
            }
            if (action.payload.fadeOutDir !== undefined) {
                clip.fadeOutDir = Math.min(1, Math.max(-1, action.payload.fadeOutDir));
            }
        },
        /** 自动交叉淡化长度（与手动 fade 分离，见 autoCrossfade.ts 的模型说明）。 */
        setClipAutoFades(
            state,
            action: PayloadAction<{
                clipId: string;
                autoFadeInSec?: number;
                autoFadeOutSec?: number;
            }>,
        ) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            if (action.payload.autoFadeInSec !== undefined) {
                clip.autoFadeInSec = Math.max(0, action.payload.autoFadeInSec);
            }
            if (action.payload.autoFadeOutSec !== undefined) {
                clip.autoFadeOutSec = Math.max(0, action.payload.autoFadeOutSec);
            }
        },
        setClipGain(state, action: PayloadAction<{ clipId: string; gain: number }>) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            clip.gain = clamp(Number(action.payload.gain), 0, 4);
        },
        setClipMuted(state, action: PayloadAction<{ clipId: string; muted: boolean }>) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            clip.muted = Boolean(action.payload.muted);
        },
        setClipsGroupId(
            state,
            action: PayloadAction<{ clipIds: string[]; groupId: string | null }>,
        ) {
            const ids = new Set(action.payload.clipIds);
            for (const clip of state.clips) {
                if (ids.has(clip.id)) {
                    clip.groupId = action.payload.groupId ?? undefined;
                }
            }
        },
        /** 乐观更新 clip 颜色（立即反映到 UI，后端确认前先行生效�?*/
        optimisticUpdateClipColor(
            state,
            action: PayloadAction<{ clipId: string; color: ClipColor }>,
        ) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            clip.color = normalizeClipColor(action.payload.color);
        },
        /** 回滚 clip 颜色（后端失败时恢复到旧值） */
        rollbackClipColor(state, action: PayloadAction<{ clipId: string; color: ClipColor }>) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            clip.color = normalizeClipColor(action.payload.color);
        },
        addClip(state, action: PayloadAction<{ trackId: string }>) {
            markProjectDirty(state.project);
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
                reversed: false,
                loopEnabled: state.loopNewClipsEnabled !== false,
                snapOffsetSec: 0,
                fadeInSec: 0,
                fadeOutSec: 0,
                fadeInShape: 1,
                fadeOutShape: 1,
                fadeInDir: 0,
                fadeOutDir: 0,
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
            markProjectDirty(state.project);
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
            const track = state.tracks.find((entry) => entry.id === action.payload);
            if (track) {
                track.muted = !track.muted;
            }
        },
        toggleTrackSolo(state, action: PayloadAction<string>) {
            const track = state.tracks.find((entry) => entry.id === action.payload);
            if (track) {
                track.solo = !track.solo;
            }
        },
        setTrackVolume(state, action: PayloadAction<{ trackId: string; volume: number }>) {
            const track = state.tracks.find((entry) => entry.id === action.payload.trackId);
            if (track) {
                track.volume = clamp(action.payload.volume, 0, MAX_TRACK_VOLUME);
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
            markProjectDirty(state.project);
            ensureClipAutomation(state, clipId);
            // ensureClipAutomation 只播种 pitch/tension；其他 param（如
            // vocoder 参数）直接索引会拿到 undefined 并在 push 时抛错。
            const target = (state.clipAutomation[clipId][action.payload.param] ??= []);
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
            markProjectDirty(state.project);
            ensureClipAutomation(state, clipId);
            const target = (state.clipAutomation[clipId][action.payload.param] ??= []);
            const point = target.find((entry) => entry.id === action.payload.pointId);
            if (point) {
                point.beat = Math.max(0, action.payload.beat);
                point.value = action.payload.value;
                target.sort((left, right) => left.beat - right.beat);
            }
        },
        setSelectedPoint(state, action: PayloadAction<string | null>) {
            state.selectedPointId = action.payload;
        },
        removeAutomationPoint(state, action: PayloadAction<{ param: EditParam; pointId: string }>) {
            const clipId = state.selectedClipId;
            if (!clipId) {
                return;
            }
            markProjectDirty(state.project);
            ensureClipAutomation(state, clipId);
            const target = (state.clipAutomation[clipId][action.payload.param] ??= []);
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
            const { clipId, curveStartSec, midiCurve, framePeriodMs } = action.payload;
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
        setClipFormantStatus(
            state,
            action: PayloadAction<{
                clipId: string;
                status: "ready" | "rebuilding" | "failed";
            }>,
        ) {
            state.clipFormantStatus[action.payload.clipId] = action.payload.status;
        },
        /** 写入某个 clip 的源共振峰分析结果（loading/ready/failed 三态）。 */
        setClipFormantAnalysis(
            state,
            action: PayloadAction<{ clipId: string; analysis: ClipFormantAnalysisState }>,
        ) {
            state.clipFormantAnalysis[action.payload.clipId] = action.payload.analysis;
        },
        openClipFormantToolWindow(
            state,
            action: PayloadAction<{
                clipId: string;
                anchor: { x: number; y: number };
            }>,
        ) {
            const { clipId, anchor } = action.payload;
            state.clipFormantToolWindow.open = true;
            state.clipFormantToolWindow.clipId = clipId;
            if (!state.clipFormantToolWindow.hasMoved) {
                state.clipFormantToolWindow.x = Math.max(12, Math.round(anchor.x));
                state.clipFormantToolWindow.y = Math.max(12, Math.round(anchor.y));
            }
        },
        setClipFormantToolWindowPosition(
            state,
            action: PayloadAction<{
                x: number;
                y: number;
            }>,
        ) {
            state.clipFormantToolWindow.x = Math.max(0, Math.round(action.payload.x));
            state.clipFormantToolWindow.y = Math.max(0, Math.round(action.payload.y));
            state.clipFormantToolWindow.hasMoved = true;
        },
        closeClipFormantToolWindow(state) {
            state.clipFormantToolWindow.open = false;
            state.clipFormantToolWindow.clipId = null;
        },
    },
    extraReducers: (builder) => {
        const setPending = (state: SessionState, label: string) => {
            state.busy = true;
            state.status = label;
            state.error = undefined;
        };
        const setRejected = (state: SessionState, action: { error?: { message?: string } }) => {
            state.busy = false;
            state.error = action.error?.message ?? "Request failed";
            state.status = "Failed";
        };

        builder
            .addCase(refreshRuntime.pending, (state) => setPending(state, "Refreshing runtime..."))
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
                        gpu_backend?: string;
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
                        gpuBackend: payload.gpu_backend ?? "",
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
                state.showAllTakes = Boolean(s.showAllTakes ?? true);
                state.syncEditsAcrossTakes = Boolean(s.syncEditsAcrossTakes ?? true);
                state.loopNewClipsEnabled = s.loopNewClips ?? true;
                state.splitTransitionEnabled = Boolean(s.splitTransitionEnabled ?? true);
                state.splitTransitionMode = s.splitTransitionMode === "fade" ? "fade" : "overlap";
                state.splitTransitionDurationUnit =
                    s.splitTransitionDurationUnit === "percent" ? "percent" : "seconds";
                const splitDuration = Number(s.splitTransitionDurationSec);
                if (Number.isFinite(splitDuration)) {
                    state.splitTransitionDurationSec = clamp(splitDuration, 0.001, 10);
                }
                const splitPercent = Number(s.splitTransitionDurationPercent);
                if (Number.isFinite(splitPercent)) {
                    state.splitTransitionDurationPercent = clamp(splitPercent, 0.01, 100);
                }
                const splitCurve = s.splitTransitionCurve as string;
                // 旧命名曲线在加载期迁移为新版预设 id；未知值回退 "keep"
                //（保留原 Clip 曲线，不再改写）。
                state.splitTransitionCurve = normalizeSplitTransitionCurve(splitCurve);
                state.splitTransitionOverlapCrossfade =
                    s.splitTransitionOverlapCrossfade === "always" ? "always" : "auto";
                const loadedSnapEnabled = s.snapEnabled ?? s.gridSnap ?? true;
                state.snapEnabled = loadedSnapEnabled;
                if (s.timelineSnap && typeof s.timelineSnap === "object") {
                    state.timelineSnap = normalizeTimelineSnapSettings(
                        createDefaultTimelineSnapSettings(),
                        s.timelineSnap as Partial<TimelineSnapSettings>,
                    );
                    state.timelineSnap.enabled = Boolean(loadedSnapEnabled);
                    state.snapEnabled = state.timelineSnap.enabled;
                }
                if (s.tempoMapVisible != null) {
                    state.tempoMapVisible = Boolean(s.tempoMapVisible);
                }
                const loadedPrimaryUnit = s.primaryTimeUnit as unknown;
                if (VALID_TIME_UNITS.has(loadedPrimaryUnit as TimeUnit)) {
                    state.primaryTimeUnit = loadedPrimaryUnit as TimeUnit;
                }
                const loadedSecondaryUnit = s.secondaryTimeUnit as unknown;
                if (
                    loadedSecondaryUnit === "none" ||
                    VALID_TIME_UNITS.has(loadedSecondaryUnit as TimeUnit)
                ) {
                    state.secondaryTimeUnit = loadedSecondaryUnit as TimeUnitChoice;
                }
                const loadedSpacing = Number(s.rulerLabelSpacingPx);
                if (Number.isFinite(loadedSpacing)) {
                    state.rulerLabelSpacingPx = Math.max(
                        40,
                        Math.min(320, Math.round(loadedSpacing)),
                    );
                }
                if (s.showPlayheadTimeInTrackHeader != null) {
                    state.showPlayheadTimeInTrackHeader = Boolean(s.showPlayheadTimeInTrackHeader);
                }
                if (s.paramEditorSyncTimeline != null) {
                    state.paramEditorSyncTimeline = Boolean(s.paramEditorSyncTimeline);
                }
                state.pitchSnapEnabled = s.pitchSnap;
                // Validate pitchSnapUnit
                const validUnits: PitchSnapUnit[] = ["semitone", "scale"];
                state.pitchSnapUnit = validUnits.includes(s.pitchSnapUnit as PitchSnapUnit)
                    ? (s.pitchSnapUnit as PitchSnapUnit)
                    : "semitone";
                // Validate pitchSnapScale
                state.pitchSnapScale = (SCALE_KEYS as readonly string[]).includes(
                    s.pitchSnapScale ?? "C",
                )
                    ? ((s.pitchSnapScale ?? "C") as typeof state.pitchSnapScale)
                    : "C";
                // Load pitch snap tolerance (cents) if present in saved settings
                if (s.pitchSnapToleranceCents != null) {
                    state.pitchSnapToleranceCents = clamp(
                        Number(s.pitchSnapToleranceCents) || 0,
                        0,
                        1000,
                    );
                }
                state.playheadZoomEnabled = s.playheadZoom;
                if (s.autoScroll != null) state.autoScrollEnabled = s.autoScroll;
                if (s.paramEditorSeekPlayhead != null)
                    state.paramEditorSeekPlayheadEnabled = Boolean(s.paramEditorSeekPlayhead);
                if (s.paramEditorTimelineClickSelectTrack != null) {
                    state.paramEditorTimelineClickSelectTrackEnabled = Boolean(
                        s.paramEditorTimelineClickSelectTrack,
                    );
                }
                if (s.showClipboardPreview != null)
                    state.showClipboardPreview = s.showClipboardPreview;
                if (s.showParamValuePopup != null)
                    state.showParamValuePopup = Boolean(s.showParamValuePopup);
                if (s.scaleHighlightMode != null)
                    state.scaleHighlightMode = s.scaleHighlightMode === "always" ? "always" : "off";
                if (s.ignoreGrouping != null) state.ignoreGrouping = Boolean(s.ignoreGrouping);
                const loadedRippleMode = s.rippleMode;
                if (loadedRippleMode === "track" || loadedRippleMode === "all") {
                    state.rippleMode = loadedRippleMode;
                } else {
                    state.rippleMode = "off";
                }
                if (s.lockParamLines != null)
                    state.lockParamLinesEnabled = Boolean(s.lockParamLines);
                if (s.quickSearchAutoNormalize != null)
                    state.quickSearchAutoNormalizeEnabled = Boolean(s.quickSearchAutoNormalize);
                if (Array.isArray(s.visibleReferenceRootTrackIds)) {
                    state.visibleReferenceRootTrackIds = s.visibleReferenceRootTrackIds
                        .filter((id: unknown): id is string => typeof id === "string")
                        .map((id: string) => id.trim())
                        .filter((id: string) => id.length > 0);
                }
                const defaultStretchAlgorithm = s.defaultStretchAlgorithm;
                if (
                    defaultStretchAlgorithm != null &&
                    ["linear", "signalsmith", "soundtouch"].includes(defaultStretchAlgorithm)
                ) {
                    state.defaultStretchAlgorithm =
                        defaultStretchAlgorithm as StretchAlgorithmOption;
                }
                if (s.defaultHifiganMelStretch != null) {
                    state.defaultHifiganMelStretch = Boolean(s.defaultHifiganMelStretch);
                }
                if (s.ortEp != null) {
                    const normalized = String(s.ortEp).toLowerCase();
                    state.ortEp = ["auto", "cpu", "gpu"].includes(normalized) ? normalized : "auto";
                }
                if (s.gpuDeviceId != null) {
                    state.gpuDeviceId = Number(s.gpuDeviceId);
                }
                if (s.ortDeviceId !== undefined) {
                    const val = s.ortDeviceId;
                    state.ortDeviceId = val == null ? null : Number(val);
                }
                if (s.autoBackgroundRender != null) {
                    state.autoBackgroundRender = Boolean(s.autoBackgroundRender);
                }
                const selectDir = s.selectDragDirection;
                if (selectDir != null && ["free", "x-only", "y-only"].includes(selectDir)) {
                    state.selectDragDirection = selectDir as DragDirection;
                }
                const drawDir = s.drawDragDirection;
                if (drawDir != null && ["free", "x-only"].includes(drawDir)) {
                    state.drawDragDirection = drawDir as DrawDragDirection;
                }
                const lineVibratoDir = s.lineVibratoDragDirection;
                if (lineVibratoDir != null && ["free", "x-only"].includes(lineVibratoDir)) {
                    state.lineVibratoDragDirection = lineVibratoDir as DrawDragDirection;
                }
                const smoothness = s.smoothnessPercent ?? s.edgeSmoothnessPercent;
                if (smoothness != null) {
                    state.edgeSmoothnessPercent = clamp(Number(smoothness) || 0, 0, 100);
                }
                if (Array.isArray(s.customScalePresets)) {
                    state.customScalePresets = s.customScalePresets.map((preset: unknown) =>
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

            .addCase(loadModel.pending, (state) => setPending(state, "Loading model..."))
            .addCase(loadModel.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                state.status = (action.payload as { ok?: boolean }).ok
                    ? "Model loaded"
                    : "Load model failed";
            })
            .addCase(loadModel.rejected, setRejected)

            .addCase(processAudio.pending, (state) => setPending(state, "Processing audio..."))
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
                state.status = payload.ok ? "Audio processed" : "Process audio failed";
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
                    newClipIds?: string[];
                    requiresModeChoice?: boolean;
                    requiresStreamChoice?: boolean;
                };
                if (payload.canceled) {
                    state.status = "Import canceled";
                    return;
                }
                // 多文件/多音轨流程在这里只是"等待用户下一步选择"，并没有
                // imported 载荷 —— 不能落入下面的 else 分支误报导入失败。
                if (payload.requiresModeChoice || payload.requiresStreamChoice) {
                    state.status = "Waiting for import options";
                    return;
                }
                if (payload.path) {
                    state.audioPath = payload.path;
                    if (payload.imported?.ok) {
                        applyTimelineState(state, payload.imported, { force: true });
                        if (payload.newClipIds && payload.newClipIds.length > 0) {
                            state.multiSelectedClipIds = payload.newClipIds;
                            state.selectedClipId = payload.newClipIds[0] ?? null;
                        }
                    }
                }
                if (payload.imported?.ok) {
                    state.status = "Audio imported";
                } else {
                    state.status = "Import audio failed";
                    state.error = "import_audio_failed";
                }
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
                    newClipIds?: string[];
                };
                if (payload.path) {
                    state.audioPath = payload.path;
                    if (payload.imported?.ok) {
                        applyTimelineState(state, payload.imported, { force: true });
                        if (payload.newClipIds && payload.newClipIds.length > 0) {
                            state.multiSelectedClipIds = payload.newClipIds;
                            state.selectedClipId = payload.newClipIds[0] ?? null;
                        }
                    }
                }
                if (payload.imported?.ok) {
                    state.status = "Dropped audio imported";
                } else {
                    state.status = "Import audio failed";
                    state.error = "import_audio_failed";
                }
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
                if (ok) {
                    state.status = "Import done";
                } else {
                    state.status = "Import failed";
                    // 合成错误码：让状态栏走红色 error 通道（原版失败只写灰色 status）
                    state.error = "import_audio_failed";
                }
                if (ok && payload.imported && payload.imported.tracks) {
                    applyTimelineStatePreservingPitchVisuals(state, payload.imported);
                    if (payload.newClipIds && payload.newClipIds.length > 0) {
                        applyAutoCrossfadeInReducer(state, payload.newClipIds);
                        state.multiSelectedClipIds = payload.newClipIds;
                        state.selectedClipId = payload.newClipIds[0] ?? null;
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
                if (ok) {
                    state.status = "Import done";
                } else {
                    state.status = "Import failed";
                    // 合成错误码：让状态栏走红色 error 通道（原版失败只写灰色 status）
                    state.error = "import_audio_failed";
                }
                if (ok && payload.imported && payload.imported.tracks) {
                    applyTimelineStatePreservingPitchVisuals(state, payload.imported);
                    if (payload.newClipIds && payload.newClipIds.length > 0) {
                        applyAutoCrossfadeInReducer(state, payload.newClipIds);
                        state.multiSelectedClipIds = payload.newClipIds;
                        state.selectedClipId = payload.newClipIds[0] ?? null;
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
                if (ok) {
                    state.status = "Import done";
                } else {
                    state.status = "Import failed";
                    // 合成错误码：让状态栏走红色 error 通道（原版失败只写灰色 status）
                    state.error = "import_audio_failed";
                }
                if (ok && payload.imported && payload.imported.tracks) {
                    applyTimelineStatePreservingPitchVisuals(state, payload.imported);
                    if (payload.newClipIds && payload.newClipIds.length > 0) {
                        applyAutoCrossfadeInReducer(state, payload.newClipIds);
                        state.multiSelectedClipIds = payload.newClipIds;
                        state.selectedClipId = payload.newClipIds[0] ?? null;
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
                if (ok) {
                    state.status = "Import done";
                } else {
                    state.status = "Import failed";
                    // 合成错误码：让状态栏走红色 error 通道（原版失败只写灰色 status）
                    state.error = "import_audio_failed";
                }
                if (ok && payload.imported && payload.imported.tracks) {
                    applyTimelineStatePreservingPitchVisuals(state, payload.imported);
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

            .addCase(importMidiAsClip.pending, (state) =>
                setPending(state, "Importing MIDI clip..."),
            )
            .addCase(importMidiAsClip.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                const payload = action.payload as {
                    ok?: boolean;
                    imported?: TimelineState;
                    newClipIds?: string[];
                };
                const ok = Boolean(payload.ok);
                state.status = ok ? "MIDI clip created" : "MIDI import failed";
                if (ok && payload.imported && payload.imported.tracks) {
                    applyTimelineStatePreservingPitchVisuals(state, payload.imported);
                    if (payload.newClipIds && payload.newClipIds.length > 0) {
                        state.multiSelectedClipIds = payload.newClipIds;
                        state.selectedClipId = payload.newClipIds[0] ?? null;
                    }
                }
            })
            .addCase(importMidiAsClip.rejected, setRejected)

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

            .addCase(synthesizeAudio.pending, (state) => setPending(state, "Synthesizing..."))
            .addCase(synthesizeAudio.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                state.status = (action.payload as { ok?: boolean }).ok
                    ? "Synthesis done"
                    : "Synthesis failed";
            })
            .addCase(synthesizeAudio.rejected, setRejected)

            .addCase(exportAudio.pending, (state) => setPending(state, "Exporting WAV..."))
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

            .addCase(exportAudioAdvanced.pending, (state) =>
                setPending(state, "Exporting audio..."),
            )
            .addCase(exportAudioAdvanced.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                const payload = action.payload as {
                    ok?: boolean;
                    mode?: "project" | "separated";
                    path?: string;
                    output_dir?: string;
                    count?: number;
                };
                if (!payload.ok) {
                    state.status =
                        payload.mode === "separated" ? "Export separated failed" : "Export failed";
                    return;
                }
                if (payload.mode === "separated") {
                    const suffix = payload.output_dir
                        ? ` — ${payload.output_dir} (${payload.count ?? 0} tracks)`
                        : "";
                    state.status = `Export separated done${suffix}`;
                    return;
                }
                const suffix = payload.path ? ` — ${payload.path}` : "";
                state.status = `Export done${suffix}`;
            })
            .addCase(exportAudioAdvanced.rejected, setRejected)

            .addCase(pasteVocalShifterClipboard.pending, (state) =>
                setPending(state, "Pasting VocalShifter clipboard data..."),
            )
            .addCase(pasteVocalShifterClipboard.fulfilled, (state, action) => {
                state.busy = false;
                const payload = action.payload as {
                    tracks?: unknown;
                    newClipIds?: string[];
                    pasteEndSec?: number | null;
                };
                if (payload?.tracks) {
                    applyTimelineState(state, payload as TimelineState, { force: true });
                    if (payload.newClipIds && payload.newClipIds.length > 0) {
                        state.multiSelectedClipIds = payload.newClipIds;
                        state.selectedClipId = payload.newClipIds[0] ?? null;
                    }
                    // 粘贴后光标跳到所有新 Clip 中最靠右的结束位置
                    // （transport 已由 thunk 同步，这里对齐本地状态）。
                    // 视图聚焦延迟到提交后由 TimelinePanel 的
                    // useLayoutEffect 依据 pendingPlayheadRevealSec 执行。
                    if (
                        typeof payload.pasteEndSec === "number" &&
                        Number.isFinite(payload.pasteEndSec)
                    ) {
                        state.playheadSec = Math.max(0, payload.pasteEndSec);
                        state.pendingPlayheadRevealSec = Math.max(0, payload.pasteEndSec);
                    }
                }
                state.lastResult = payload;
                state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
                state.status = "Pasted VocalShifter clipboard data";
            })
            .addCase(pasteVocalShifterClipboard.rejected, (state, action) => {
                state.busy = false;
                state.error =
                    (action.payload as string) ?? action.error?.message ?? "Request failed";
                state.status = "Failed";
            })

            .addCase(pasteReaperClipboard.pending, (state) =>
                setPending(state, "Pasting Reaper clipboard data..."),
            )
            .addCase(pasteReaperClipboard.fulfilled, (state, action) => {
                state.busy = false;
                const payload = action.payload as {
                    timeline?: TimelineState;
                    newClipIds?: string[];
                    pasteEndSec?: number | null;
                    skippedFiles?: string[];
                };
                if (payload?.timeline) {
                    applyTimelineState(state, payload.timeline, { force: true });
                    if (payload.newClipIds && payload.newClipIds.length > 0) {
                        state.multiSelectedClipIds = payload.newClipIds;
                        state.selectedClipId = payload.newClipIds[0] ?? null;
                    }
                    // 粘贴后光标跳到所有新 Clip 中最靠右的结束位置
                    // （transport 已由 thunk 同步，这里对齐本地状态）。
                    // 视图聚焦延迟到提交后由 TimelinePanel 的
                    // useLayoutEffect 依据 pendingPlayheadRevealSec 执行。
                    if (
                        typeof payload.pasteEndSec === "number" &&
                        Number.isFinite(payload.pasteEndSec)
                    ) {
                        state.playheadSec = Math.max(0, payload.pasteEndSec);
                        state.pendingPlayheadRevealSec = Math.max(0, payload.pasteEndSec);
                    }
                }
                const skippedFiles = payload?.skippedFiles;
                state.reaperSkippedFilesDialog =
                    Array.isArray(skippedFiles) && skippedFiles.length > 0 ? skippedFiles : null;
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

            .addCase(playOriginal.pending, (state) => setPending(state, "Playing original..."))
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

            .addCase(stopAudioPlayback.pending, (state) => {
                setPending(state, "Stopping audio...");
                // Pause UI immediately on user stop/pause command; backend sync will
                // confirm the final transport state shortly after.
                state.runtime.isPlaying = false;
                state.runtime.playbackTarget = null;
            })
            .addCase(stopAudioPlayback.fulfilled, (state, action) => {
                state.busy = false;
                state.lastResult = action.payload;
                const payload = action.payload as {
                    ok?: boolean;
                    restoreAnchor?: boolean;
                    wasPlaying?: boolean;
                    anchorSec?: number | null;
                    stopped_at_sec?: number | null;
                };
                // If restoreAnchor is set (Play/Stop action), restore playhead to anchor position
                if (
                    payload.restoreAnchor &&
                    payload.wasPlaying &&
                    typeof payload.anchorSec === "number" &&
                    Number.isFinite(payload.anchorSec)
                ) {
                    state.playheadSec = Math.max(0, payload.anchorSec);
                } else if (
                    typeof payload.stopped_at_sec === "number" &&
                    Number.isFinite(payload.stopped_at_sec)
                ) {
                    // 暂停：把视觉光标对齐到引擎的实际停止位置。轮询存在至多
                    // 一个周期（~33ms）+ 往返的滞后，最后一次采样后音频仍在
                    // 前进；后端记录的暂停点是这个精确位置。若不同步，视觉
                    // 位置会落后于真实位置，后续任何编辑回灌快照都会让光标
                    // 再次右跳。
                    state.playheadSec = Math.max(0, payload.stopped_at_sec);
                }
                state.runtime.isPlaying = false;
                state.runtime.playbackTarget = null;
                state.runtime.playbackPositionSec = 0;
                state.runtime.playbackDurationSec = 0;
                state.playbackClipId = null;
                state.playbackAnchorSec = 0;
                state.status = payload.ok ? "Audio stopped" : "Stop audio failed";
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
                } else if (state.runtime.isPlaying) {
                    // 播放→停止跃迁（音频自然结束 / 引擎等待重渲染时自动暂停）：
                    // 引擎的 position 冻结在真实停止点，而本地光标还停留在最后
                    // 一次轮询采样（略落后）。对齐一次，保证视觉位置 = 真实
                    // 位置；此后的轮询（is_playing=false 分支）不再改写光标。
                    nextplayheadSec = Math.max(0, (payload.base_sec ?? 0) + nextPositionSec);
                }

                const shouldUpdatePlaybackFields =
                    nextIsPlaying !== state.runtime.isPlaying ||
                    nextTarget !== state.runtime.playbackTarget ||
                    Math.abs(nextPositionSec - state.runtime.playbackPositionSec) > EPS_SEC ||
                    Math.abs(nextDurationSec - state.runtime.playbackDurationSec) > EPS_SEC ||
                    Math.abs(nextplayheadSec - state.playheadSec) > EPS_SEC;

                if (!shouldUpdatePlaybackFields) {
                    // No state change needed.
                    return;
                }

                state.runtime.isPlaying = nextIsPlaying;
                state.runtime.playbackTarget = nextTarget;
                state.runtime.playbackPositionSec = nextPositionSec;
                state.runtime.playbackDurationSec = nextDurationSec;

                if (nextIsPlaying || nextplayheadSec !== state.playheadSec) {
                    state.playheadSec = nextplayheadSec;
                }
                if (!nextIsPlaying) {
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
                applyTimelineState(state, payload, {
                    force: true,
                    preserveProjectNotes: false,
                    // 启动同步采纳后端播放头；播放中的回滚式重取（如 Tempo Map
                    // 提交失败）保留本地播放头，避免跳回旧的起始位置。
                    adoptPlayhead: !state.runtime.isPlaying,
                });
            })

            .addCase(undoRemote.pending, (state, action) => {
                // 撤销/重做以后端为唯一权威：pending 阶段不做任何本地快照
                // 回放（旧实现会先乐观渲染前端快照——它与其后端检查点在深度
                // 与内容上都可能错位，造成轨道视图闪现任意旧状态后再被
                // fulfilled 的后端快照纠正）。这里仅记录最新请求，乱序到达
                // 的旧响应在 fulfilled/rejected 中被丢弃。
                state._latestHistoryOpRequestId = action.meta.requestId;
                // 作废在途编辑响应：迟到的编辑快照不得覆盖撤销结果。
                state._latestEditRequestId = null;
            })

            .addCase(undoRemote.fulfilled, (state, action) => {
                if (state._latestHistoryOpRequestId !== action.meta.requestId) return;
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) return;
                // 撤销把时间线（含 playhead_sec）整体回退到上一个检查点：快照里
                // 的 playhead_sec 就是该状态形成时的播放光标位置，也是回退之后
                // 一切以光标为锚点的编辑操作（粘贴/分割/录音起点等）在后端的
                // 实际操作点。必须采纳它让视觉光标同步归位 —— 若沿用撤销前的
                // 本地光标（可能是上一步操作挪过去的位置，如粘贴的
                // pasteEndSec），视觉停在 B、后端实际停在 A，下一次操作就会落
                // 在视觉之外的位置（视觉编辑点 ≠ 实际编辑点）。
                // 播放中例外：光标归传输层（音频时钟）所有，检查点值停留在
                // 本次播放的起始位置已过期，且轮询会持续覆写，不做本地改写。
                const prevPlayheadSec = state.playheadSec;
                applyTimelineState(state, payload, {
                    force: true,
                    preserveProjectNotes: false,
                    adoptPlayhead: !state.runtime.isPlaying,
                });
                // 光标随回退真实挪动时登记"聚焦播放光标"：离屏时由
                // TimelinePanel 的 useLayoutEffect 滚动到可见（画面内不扰动），
                // 保证用户看得到回退后的实际操作点。
                if (
                    !state.runtime.isPlaying &&
                    Math.abs(state.playheadSec - prevPlayheadSec) > PLAYHEAD_MOVE_EPS_SEC
                ) {
                    state.pendingPlayheadRevealSec = state.playheadSec;
                }
            })

            .addCase(undoRemote.rejected, (state, action) => {
                // 无乐观本地变更需要回滚：后端 undo 失败时时间线未变，
                // 前端保持现状即可（仅丢弃过期响应）。
                if (state._latestHistoryOpRequestId !== action.meta.requestId) return;
            })

            .addCase(redoRemote.pending, (state, action) => {
                // 同 undoRemote.pending：以后端为唯一权威，不做本地快照回放。
                state._latestHistoryOpRequestId = action.meta.requestId;
                // 作废在途编辑响应：迟到的编辑快照不得覆盖重做结果。
                state._latestEditRequestId = null;
            })

            .addCase(redoRemote.fulfilled, (state, action) => {
                if (state._latestHistoryOpRequestId !== action.meta.requestId) return;
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) return;
                // 与 undoRemote.fulfilled 对称：重做恢复的时间线快照携带的
                // playhead_sec 是该状态（被撤销暂存时）的光标位置，即重做后
                // 后端的实际操作点 —— 采纳它，视觉光标与实际编辑点保持一致。
                // 播放中例外同 undoRemote.fulfilled：光标归传输层所有。
                const prevPlayheadSec = state.playheadSec;
                applyTimelineState(state, payload, {
                    force: true,
                    preserveProjectNotes: false,
                    adoptPlayhead: !state.runtime.isPlaying,
                });
                if (
                    !state.runtime.isPlaying &&
                    Math.abs(state.playheadSec - prevPlayheadSec) > PLAYHEAD_MOVE_EPS_SEC
                ) {
                    state.pendingPlayheadRevealSec = state.playheadSec;
                }
            })

            .addCase(redoRemote.rejected, (state, action) => {
                // 同 undoRemote.rejected：无乐观本地变更需要回滚。
                if (state._latestHistoryOpRequestId !== action.meta.requestId) return;
            })

            .addCase(newProjectRemote.pending, (state) => {
                // 新建工程为权威替换：作废一切在途编辑响应。
                state._latestEditRequestId = null;
            })
            .addCase(newProjectRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) return;
                applyTimelineState(state, payload, {
                    force: true,
                    preserveProjectNotes: false,
                    adoptPlayhead: true,
                });
                state.status = "New project";
            })

            .addCase(openProjectFromDialog.pending, (state) => {
                setPending(state, "Opening project...");
                // 工程载入为权威替换：作废一切在途编辑响应。
                state._latestEditRequestId = null;
            })
            .addCase(openProjectFromDialog.fulfilled, (state, action) => {
                state.busy = false;
                const payload = action.payload as
                    | { ok: true; canceled: true }
                    | {
                          ok: boolean;
                          canceled: false;
                          timeline?: TimelineState;
                          error?: string;
                          projectVersionTooNew?: boolean;
                      };
                if (!payload || payload.canceled) {
                    state.status = "Open canceled";
                    return;
                }
                if ((payload as { projectVersionTooNew?: boolean }).projectVersionTooNew) {
                    state.status = "Project version confirmation required";
                    return;
                }
                if (payload.ok === false) {
                    // 后端返回了明确的失败原因（文件不存在/解析失败等）。
                    state.error = payload.error ?? "Open project failed";
                    state.status = "Open failed";
                    return;
                }
                applyTimelineState(state, payload.timeline as TimelineState, {
                    force: true,
                    preserveProjectNotes: false,
                    adoptPlayhead: true,
                });
                state.status = "Project opened";
            })
            .addCase(openProjectFromDialog.rejected, (state, action) => {
                state.busy = false;
                state.error = action.error?.message ?? "Open project failed";
                state.status = "Open failed";
            })

            .addCase(openProjectFromPath.pending, (state) => {
                setPending(state, "Opening project...");
                // 工程载入为权威替换：作废一切在途编辑响应。
                state._latestEditRequestId = null;
            })
            .addCase(openProjectFromPath.fulfilled, (state, action) => {
                state.busy = false;
                const payload = action.payload as {
                    ok?: boolean;
                    error?: string;
                    projectVersionTooNew?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    state.error = payload.error ?? "Open project failed";
                    state.status = "Open failed";
                    return;
                }
                if (payload.projectVersionTooNew) {
                    state.status = "Project version confirmation required";
                    return;
                }
                applyTimelineState(state, payload, {
                    force: true,
                    preserveProjectNotes: false,
                    adoptPlayhead: true,
                });
                state.status = "Project opened";
            })
            .addCase(openProjectFromPath.rejected, (state, action) => {
                state.busy = false;
                state.error = action.error?.message ?? "Open project failed";
                state.status = "Open failed";
            })

            .addCase(openProjectFromPathForced.pending, (state) =>
                setPending(state, "Opening project..."),
            )
            .addCase(openProjectFromPathForced.fulfilled, (state, action) => {
                state.busy = false;
                const payload = action.payload as {
                    ok?: boolean;
                    error?: string;
                    projectVersionTooNew?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    state.error = payload.error ?? "Open project failed";
                    state.status = "Open failed";
                    return;
                }
                if (payload.projectVersionTooNew) {
                    state.status = "Project version confirmation required";
                    return;
                }
                applyTimelineState(state, payload, {
                    force: true,
                    preserveProjectNotes: false,
                    adoptPlayhead: true,
                });
                state.status = "Project opened";
            })
            .addCase(openProjectFromPathForced.rejected, (state, action) => {
                state.busy = false;
                state.error = action.error?.message ?? "Open project failed";
                state.status = "Open failed";
            })

            .addCase(pickProjectToImport.pending, (state) =>
                setPending(state, "Picking project to import..."),
            )
            .addCase(pickProjectToImport.fulfilled, (state, action) => {
                state.busy = false;
                const payload = action.payload as
                    | { ok: true; canceled: true }
                    | { ok: true; canceled: false; path: string };
                if (!payload || (payload as { canceled?: boolean }).canceled) {
                    state.status = "Import canceled";
                }
            })
            .addCase(pickProjectToImport.rejected, (state, action) => {
                state.busy = false;
                state.error = action.error?.message ?? "Import project failed";
                state.status = "Import failed";
            })

            .addCase(importProjectFromPath.pending, (state) =>
                setPending(state, "Importing project..."),
            )
            .addCase(importProjectFromPath.fulfilled, (state, action) => {
                state.busy = false;
                const payload = action.payload as
                    | { ok: true; canceled: true }
                    | {
                          ok: true;
                          canceled: false;
                          timeline: TimelineState;
                          newClipIds?: string[];
                          sourceProject?: string;
                      };
                if (!payload || payload.canceled) {
                    state.status = "Import canceled";
                    return;
                }
                applyTimelineState(state, payload.timeline, {
                    force: true,
                    preserveProjectNotes: false,
                    adoptPlayhead: true,
                });
                const newClipIds = payload.newClipIds;
                if (newClipIds && newClipIds.length > 0) {
                    state.multiSelectedClipIds = newClipIds;
                    state.selectedClipId = newClipIds[0] ?? null;
                }
                state.status = "Project imported";
            })
            .addCase(importProjectFromPath.rejected, (state, action) => {
                state.busy = false;
                state.error =
                    (action.payload as string) ?? action.error?.message ?? "Import project failed";
                state.status = "Import failed";
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
                          newClipIds?: string[];
                          skippedFiles?: string[];
                      };
                if (!payload || payload.canceled) {
                    state.status = "Import canceled";
                    return;
                }
                applyTimelineState(state, payload.timeline, {
                    force: true,
                    preserveProjectNotes: false,
                    adoptPlayhead: true,
                });
                const newClipIds = payload.newClipIds;
                if (newClipIds && newClipIds.length > 0) {
                    state.multiSelectedClipIds = newClipIds;
                    state.selectedClipId = newClipIds[0] ?? null;
                }
                const skippedFiles = payload.skippedFiles;
                state.vocalShifterSkippedFilesDialog =
                    Array.isArray(skippedFiles) && skippedFiles.length > 0 ? skippedFiles : null;
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
                          newClipIds?: string[];
                          skippedFiles?: string[];
                      };
                if (!payload || payload.canceled) {
                    state.status = "Import canceled";
                    return;
                }
                applyTimelineState(state, payload.timeline, {
                    force: true,
                    preserveProjectNotes: false,
                    adoptPlayhead: true,
                });
                const newClipIds = payload.newClipIds;
                if (newClipIds && newClipIds.length > 0) {
                    state.multiSelectedClipIds = newClipIds;
                    state.selectedClipId = newClipIds[0] ?? null;
                }
                const skippedFiles = payload.skippedFiles;
                state.vocalShifterSkippedFilesDialog =
                    Array.isArray(skippedFiles) && skippedFiles.length > 0 ? skippedFiles : null;
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
                          newClipIds?: string[];
                          skippedFiles?: string[];
                      };
                if (!payload || payload.canceled) {
                    state.status = "Import canceled";
                    return;
                }
                applyTimelineState(state, payload.timeline, {
                    force: true,
                    preserveProjectNotes: false,
                    adoptPlayhead: true,
                });
                const newClipIds = payload.newClipIds;
                if (newClipIds && newClipIds.length > 0) {
                    state.multiSelectedClipIds = newClipIds;
                    state.selectedClipId = newClipIds[0] ?? null;
                }
                const skippedFiles = payload.skippedFiles;
                state.reaperSkippedFilesDialog =
                    Array.isArray(skippedFiles) && skippedFiles.length > 0 ? skippedFiles : null;
                state.status = "Reaper project imported";
            })
            .addCase(openReaperFromDialog.rejected, (state, action) => {
                state.busy = false;
                state.error =
                    (action.payload as string) ?? action.error?.message ?? "Import Reaper failed";
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
                          newClipIds?: string[];
                          skippedFiles?: string[];
                      };
                if (!payload || payload.canceled) {
                    state.status = "Import canceled";
                    return;
                }
                applyTimelineState(state, payload.timeline, {
                    force: true,
                    preserveProjectNotes: false,
                    adoptPlayhead: true,
                });
                const newClipIds = payload.newClipIds;
                if (newClipIds && newClipIds.length > 0) {
                    state.multiSelectedClipIds = newClipIds;
                    state.selectedClipId = newClipIds[0] ?? null;
                }
                const skippedFiles = payload.skippedFiles;
                state.reaperSkippedFilesDialog =
                    Array.isArray(skippedFiles) && skippedFiles.length > 0 ? skippedFiles : null;
                state.status = "Reaper project imported";
            })
            .addCase(openReaperFromPath.rejected, (state, action) => {
                state.busy = false;
                state.error =
                    (action.payload as string) ?? action.error?.message ?? "Import Reaper failed";
                state.status = "Import failed";
            })

            .addCase(saveProjectRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                    canceled?: boolean;
                    versionConflict?: boolean;
                    path?: string;
                    existingVersion?: number;
                    currentVersion?: number;
                    existingIsNewer?: boolean;
                    timeline?: TimelineState & { ok?: boolean };
                    tracks?: unknown;
                    clips?: unknown;
                };
                if (payload?.versionConflict) {
                    state.saveVersionConflictDialog = {
                        path: payload.path as string,
                        existingVersion: Number(payload.existingVersion ?? 0),
                        currentVersion: Number(payload.currentVersion ?? 0),
                        existingIsNewer: Boolean(payload.existingIsNewer),
                    };
                    state.status = "Save version confirmation required";
                    return;
                }
                if (payload?.ok && payload?.canceled) {
                    state.status = "Save canceled";
                    return;
                }

                // 保留播放头位置和音高曲线，避免 UI 跳变
                const currentPlayheadSec = state.playheadSec;
                const currentParamsEpoch = state.paramsEpoch;
                // 浅拷贝快照：draft 代理会被 applyTimelineState 的 prune 透过修改（见 applyTimelineStatePreservingPitchVisuals）
                const currentClipPitchCurves = { ...state.clipPitchCurves };

                if (payload?.ok && payload?.timeline?.ok) {
                    applyTimelineState(state, payload.timeline as TimelineState, { force: true });
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
                const payload = action.payload as {
                    ok?: boolean;
                    canceled?: boolean;
                    versionConflict?: boolean;
                    path?: string;
                    existingVersion?: number;
                    currentVersion?: number;
                    existingIsNewer?: boolean;
                    timeline?: TimelineState & { ok?: boolean };
                    tracks?: unknown;
                    clips?: unknown;
                };
                if (payload?.versionConflict) {
                    state.saveVersionConflictDialog = {
                        path: payload.path as string,
                        existingVersion: Number(payload.existingVersion ?? 0),
                        currentVersion: Number(payload.currentVersion ?? 0),
                        existingIsNewer: Boolean(payload.existingIsNewer),
                    };
                    state.status = "Save version confirmation required";
                    return;
                }
                if (payload?.ok && payload?.canceled) {
                    state.status = "Save As canceled";
                    return;
                }

                // 保留播放头位置和音高曲线，避免 UI 跳变
                const currentPlayheadSec = state.playheadSec;
                const currentParamsEpoch = state.paramsEpoch;
                // 浅拷贝快照：draft 代理会被 applyTimelineState 的 prune 透过修改（见 applyTimelineStatePreservingPitchVisuals）
                const currentClipPitchCurves = { ...state.clipPitchCurves };

                if (payload?.ok && payload?.timeline?.ok) {
                    applyTimelineState(state, payload.timeline as TimelineState, { force: true });
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

            .addCase(saveProjectToPathRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                    canceled?: boolean;
                    versionConflict?: boolean;
                    path?: string;
                    existingVersion?: number;
                    currentVersion?: number;
                    existingIsNewer?: boolean;
                    timeline?: TimelineState & { ok?: boolean };
                    tracks?: unknown;
                    clips?: unknown;
                };
                if (payload?.versionConflict) {
                    // 理论上 force 保存不会再冲突；兜底再次弹出确认框。
                    state.saveVersionConflictDialog = {
                        path: payload.path as string,
                        existingVersion: Number(payload.existingVersion ?? 0),
                        currentVersion: Number(payload.currentVersion ?? 0),
                        existingIsNewer: Boolean(payload.existingIsNewer),
                    };
                    state.status = "Save version confirmation required";
                    return;
                }
                if (payload?.ok && payload?.canceled) {
                    state.status = "Save canceled";
                    return;
                }

                // 保留播放头位置和音高曲线，避免 UI 跳变
                const currentPlayheadSec = state.playheadSec;
                const currentParamsEpoch = state.paramsEpoch;
                // 浅拷贝快照：draft 代理会被 applyTimelineState 的 prune 透过修改（见 applyTimelineStatePreservingPitchVisuals）
                const currentClipPitchCurves = { ...state.clipPitchCurves };

                if (payload?.ok && payload?.timeline?.ok) {
                    applyTimelineState(state, payload.timeline as TimelineState, { force: true });
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
            .addCase(saveProjectToPathRemote.rejected, (state) => {
                state.status = "Save failed";
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
                // 后端会把工程音阶同步到 Tempo Map 初始点（初始点即工程基准记录）：
                // 前端镜像该同步，避免 effectiveScaleAtSec 等 UI 计算使用过期音阶
                // （钢琴卷帘高亮、音高吸附会与音频渲染不一致）。
                if (state.tempoMap && state.tempoMap.points.length > 0) {
                    const projectScale: ScaleLike | null =
                        state.project.useCustomScale && state.project.customScale
                            ? state.project.customScale.notes
                            : state.project.baseScale;
                    state.tempoMap = {
                        ...state.tempoMap,
                        points: [
                            {
                                ...state.tempoMap.points[0],
                                scale: scaleLikeToScaleData(
                                    projectScale ?? undefined,
                                    state.project.useCustomScale
                                        ? (state.project.customScale?.name ?? undefined)
                                        : undefined,
                                ),
                            },
                            ...state.tempoMap.points.slice(1),
                        ],
                    };
                }
                // 工程音阶变化会影响子轨道“度数差”等依赖音阶的渲染，触发参数曲线/渲染缓存失效。
                state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
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
                // 与 setProjectBaseScaleRemote 一致：镜像后端对 Tempo Map 初始点
                // 音阶的同步（初始点即工程基准记录）。
                if (state.tempoMap && state.tempoMap.points.length > 0) {
                    const projectScale: ScaleLike | null =
                        state.project.useCustomScale && state.project.customScale
                            ? state.project.customScale.notes
                            : state.project.baseScale;
                    state.tempoMap = {
                        ...state.tempoMap,
                        points: [
                            {
                                ...state.tempoMap.points[0],
                                scale: scaleLikeToScaleData(
                                    projectScale ?? undefined,
                                    state.project.useCustomScale
                                        ? (state.project.customScale?.name ?? undefined)
                                        : undefined,
                                ),
                            },
                            ...state.tempoMap.points.slice(1),
                        ],
                    };
                }
                // 工程音阶变化会影响子轨道“度数差”等依赖音阶的渲染，触发参数曲线/渲染缓存失效。
                state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
            })

            .addCase(setProjectTimelineSettingsRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                    project?: {
                        beats_per_bar?: number;
                        time_signature_denominator?: number;
                        grid_size?: string;
                        dirty?: boolean;
                    };
                };
                if (!payload.ok) {
                    return;
                }
                const beats = clamp(Number(payload.project?.beats_per_bar ?? state.beats), 1, 32);
                const denominator = (TEMPO_DENOMINATORS as readonly number[]).includes(
                    Number(payload.project?.time_signature_denominator),
                )
                    ? Number(payload.project?.time_signature_denominator)
                    : clampDenominator(state.project.timeSignatureDenominator);
                const gridRaw = String(payload.project?.grid_size ?? state.grid);
                const valid = VALID_GRID_SIZES.has(gridRaw as GridSize);
                const grid = (valid ? gridRaw : "1/4") as GridSize;

                state.beats = beats;
                state.grid = grid;
                state.project.beatsPerBar = beats;
                state.project.timeSignatureDenominator = denominator;
                state.project.gridSize = grid;
                if (typeof payload.project?.dirty === "boolean") {
                    state.project.dirty = payload.project.dirty;
                }
            })

            .addCase(setTempoMapRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok || !payload.tracks || !payload.clips) {
                    return;
                }
                // 后端为权威来源：应用完整快照（含 tempo_map 与工程基准值）。
                applyTimelineState(state, payload, { force: true });
                // 显式触发后台预渲染：Tempo Map 音阶变化会影响子轨道“度数差”等
                // 依赖音阶的渲染。applyTimelineState 已使 paramsEpoch 递增，
                // App 层据此调用 startBackgroundRender（与工程音阶变更路径
                // setProjectBaseScaleRemote.fulfilled 保持一致）；此处再显式递增，
                // 确保该触发不依赖 applyTimelineState 的内部实现细节。
                state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
                state.status = "Tempo map updated";
            })
            .addCase(setTempoMapRemote.rejected, setRejected)

            .addCase(setProjectStretchSettingsRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                    project?: {
                        stretch_algorithm_override?: StretchAlgorithmOption | null;
                        hifigan_mel_stretch_override?: boolean | null;
                        dirty?: boolean;
                    };
                };
                if (!payload.ok) {
                    return;
                }
                if (payload.project?.stretch_algorithm_override !== undefined) {
                    state.project.stretchAlgorithmOverride =
                        payload.project.stretch_algorithm_override ?? null;
                }
                if (payload.project?.hifigan_mel_stretch_override !== undefined) {
                    state.project.hifiganMelStretchOverride =
                        payload.project.hifigan_mel_stretch_override ?? null;
                }
                if (typeof payload.project?.dirty === "boolean") {
                    state.project.dirty = payload.project.dirty;
                }
            })

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

            .addCase(pasteTimelineClipboardRemote.pending, (state) =>
                setPending(state, "Pasting timeline clipboard..."),
            )
            .addCase(pasteTimelineClipboardRemote.fulfilled, (state, action) => {
                state.busy = false;
                const payload = action.payload as {
                    ok?: boolean;
                    timeline?: TimelineState;
                    newClipIds?: string[];
                    pasteEndSec?: number | null;
                };
                if (!payload?.ok || !payload.timeline?.tracks) {
                    state.status = "Paste timeline clipboard failed";
                    return;
                }
                applyTimelineState(state, payload.timeline, { force: true });
                if (payload.newClipIds && payload.newClipIds.length > 0) {
                    state.multiSelectedClipIds = payload.newClipIds;
                    state.selectedClipId = payload.newClipIds[0] ?? null;
                }
                // 粘贴后光标跳到所有新 Clip 中最靠右的结束位置
                // （transport 已由 thunk 同步，这里对齐本地状态）。
                // 视图聚焦延迟到提交后由 TimelinePanel 的
                // useLayoutEffect 依据 pendingPlayheadRevealSec 执行，
                // 避免在工程全长扩充前被旧滚动上限钳制。
                if (
                    typeof payload.pasteEndSec === "number" &&
                    Number.isFinite(payload.pasteEndSec)
                ) {
                    state.playheadSec = Math.max(0, payload.pasteEndSec);
                    state.pendingPlayheadRevealSec = Math.max(0, payload.pasteEndSec);
                }
                state.status = "Timeline clipboard pasted";
            })
            .addCase(pasteTimelineClipboardRemote.rejected, (state, action) => {
                state.busy = false;
                state.error =
                    (action.payload as string) ??
                    action.error?.message ??
                    "Paste timeline clipboard failed";
                state.status = "Paste timeline clipboard failed";
            })

            .addCase(duplicateClipsBulkRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
                state.status = "Clips duplicated";
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

            .addCase(moveClipRemote.pending, (state, action) => {
                state._latestEditRequestId = action.meta.requestId;
            })
            .addCase(moveClipRemote.fulfilled, (state, action) => {
                // 乱序守卫：只采纳最近一次编辑请求的响应（见 _latestEditRequestId）。
                if (state._latestEditRequestId !== action.meta.requestId) {
                    return;
                }
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                // force：移动是本次交互的权威结果（含波纹编辑对后续剪辑的平移）。
                // 交互锁期间若不强制应用，后端返回的波纹位移会被丢弃，导致波纹“无效”。
                applyTimelineState(state, payload, { force: true });
            })
            .addCase(moveClipsRemote.pending, (state, action) => {
                state._latestEditRequestId = action.meta.requestId;
            })
            .addCase(moveClipsRemote.fulfilled, (state, action) => {
                // 乱序守卫：只采纳最近一次编辑请求的响应（见 _latestEditRequestId）。
                if (state._latestEditRequestId !== action.meta.requestId) {
                    return;
                }
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                // force：见 moveClipRemote.fulfilled 注释（交互锁期间也必须应用波纹结果）。
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(splitClipRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                    created_clip_ids?: string[] | null;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applySplitSelection(state, action, payload);
            })

            .addCase(splitClipsAtRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                    created_clip_ids?: string[] | null;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applySplitSelection(state, action, payload);
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

            .addCase(groupClipsRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(ungroupClipsRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(toggleGroupDisabledRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(convertClipsToPitchReferenceRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(updatePitchReferenceRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(setClipStateRemote.fulfilled, (state, action) => {
                // 乱序守卫：只采纳最近一次编辑请求的响应（见 _latestEditRequestId）。
                // 多选批量松手时若旧请求的中间状态快照迟到，会被这里丢弃，
                // 避免"闪回原状/部分 Clip 还原"；最新响应（含此前全部已提交
                // 变更）总是最后一个生效。
                if (state._latestEditRequestId !== action.meta.requestId) {
                    return;
                }
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                // force：重设尺寸（右缘位移）的权威结果含波纹平移，交互锁期间不能丢弃。
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(setClipStateRemote.pending, (state, action) => {
                state._latestEditRequestId = action.meta.requestId;
                applyOptimisticClipState(state, action.meta.arg);
                const clip = state.clips.find((entry) => entry.id === action.meta.arg.clipId);
                const formantEnabled =
                    action.meta.arg.formantMorph?.enabled ?? clip?.formantMorph?.enabled ?? false;
                const formantRelevantChange =
                    action.meta.arg.formantMorph !== undefined ||
                    action.meta.arg.sourceStartSec !== undefined ||
                    action.meta.arg.sourceEndSec !== undefined ||
                    action.meta.arg.reversed !== undefined;
                if (formantEnabled && formantRelevantChange) {
                    state.clipFormantStatus[action.meta.arg.clipId] = "rebuilding";
                }
            })
            .addCase(setClipStateRemote.rejected, (state, action) => {
                // 后端拒绝：保留乐观值（用户意图可见），给出非致命反馈。
                // 此后任何权威快照会以真实状态纠正；不静默回滚。
                applyOptimisticClipState(state, action.meta.arg);
                state.status = "Clip edit rejected";
            })

            .addCase(setClipsStateBulkRemote.fulfilled, (state, action) => {
                // 乱序守卫：见 setClipStateRemote.fulfilled。
                if (state._latestEditRequestId !== action.meta.requestId) {
                    return;
                }
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                // force：批量状态（含未来尺寸波纹路径）的权威结果，交互锁期间不能丢弃。
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(setClipsStateBulkRemote.pending, (state, action) => {
                state._latestEditRequestId = action.meta.requestId;
                applyOptimisticBulkClipState(state, action.meta.arg.updates);
            })
            .addCase(setClipsStateBulkRemote.rejected, (state, action) => {
                // 后端拒绝：保留乐观值（用户意图可见），给出非致命反馈。
                applyOptimisticBulkClipState(state, action.meta.arg.updates);
                state.status = "Clip edit rejected";
            })

            .addCase(setClipActiveTakeRemote.pending, (state, action) => {
                const clip = state.clips.find((entry) => entry.id === action.meta.arg.clipId);
                if (!clip) return;
                const takes = clip.takes ?? [];
                const take = takes.find((entry) => entry.id === action.meta.arg.takeId);
                if (!take) return;
                captureTakeRollback(state, [action.meta.arg.clipId]);
                clip.activeTakeId = take.id;
                applyActiveTakeToFlat(clip, take);
            })
            .addCase(setClipActiveTakeRemote.fulfilled, (state, action) => {
                const payload = action.payload as { ok?: boolean } & TimelineState;
                if (!payload.ok) {
                    // 后端拒绝：回滚乐观切换并给出可见反馈。
                    restoreTakeRollback(state, [action.meta.arg.clipId]);
                    state.error = "Take switch rejected";
                    state.status = "Failed";
                    return;
                }
                clearTakeRollback([action.meta.arg.clipId]);
                applyTimelineStatePreservingPlayhead(state, payload);
            })
            .addCase(setClipActiveTakeRemote.rejected, (state, action) => {
                restoreTakeRollback(state, [action.meta.arg.clipId]);
                setRejected(state, action);
            })

            .addCase(cycleClipTakesRemote.pending, (state, action) => {
                for (const clipId of action.meta.arg.clipIds) {
                    const clip = state.clips.find((entry) => entry.id === clipId);
                    const takes = clip?.takes ?? [];
                    if (!clip || takes.length <= 1) continue;
                    const direction = action.meta.arg.direction ?? 1;
                    const currentIndex = Math.max(
                        0,
                        takes.findIndex((take) => take.id === clip.activeTakeId),
                    );
                    const nextIndex =
                        direction >= 0
                            ? (currentIndex + 1) % takes.length
                            : (currentIndex + takes.length - 1) % takes.length;
                    const next = takes[nextIndex];
                    captureTakeRollback(state, [clipId]);
                    clip.activeTakeId = next.id;
                    applyActiveTakeToFlat(clip, next);
                }
            })
            .addCase(cycleClipTakesRemote.fulfilled, (state, action) => {
                const payload = action.payload as { ok?: boolean } & TimelineState;
                if (!payload.ok) {
                    restoreTakeRollback(state, action.meta.arg.clipIds);
                    state.error = "Take cycle rejected";
                    state.status = "Failed";
                    return;
                }
                clearTakeRollback(action.meta.arg.clipIds);
                applyTimelineStatePreservingPlayhead(state, payload);
            })
            .addCase(cycleClipTakesRemote.rejected, (state, action) => {
                restoreTakeRollback(state, action.meta.arg.clipIds);
                setRejected(state, action);
            })

            .addCase(setClipTakeReversedRemote.pending, (state, action) => {
                // 乐观翻转单个 Take：与后端 flip_take_playback_direction 同口径
                // 换算该 Take 的源窗口/锚点（保持消费内容不变）。active take
                // 需物化到 flat 投影；inactive take 只动自身条目。
                const clip = state.clips.find((entry) => entry.id === action.meta.arg.clipId);
                if (!clip) return;
                const takes = clip.takes ?? [];
                const take = takes.find((entry) => entry.id === action.meta.arg.takeId);
                if (!take) return;
                if (take.reversed !== action.meta.arg.reversed) {
                    flipTakeSourceWindowForDirection(
                        take,
                        clip.lengthSec,
                        clip.clipPlaybackRate ?? 1,
                    );
                }
                take.reversed = action.meta.arg.reversed;
                if (take.id === clip.activeTakeId) {
                    applyActiveTakeToFlat(clip, take);
                }
            })
            .addCase(setClipTakeReversedRemote.fulfilled, (state, action) => {
                const payload = action.payload as { ok?: boolean } & TimelineState;
                if (!payload.ok) {
                    state.error = "Take reverse rejected";
                    state.status = "Failed";
                    return;
                }
                applyTimelineStatePreservingPlayhead(state, payload);
            })
            .addCase(setClipTakeReversedRemote.rejected, setRejected)

            .addCase(packClipsIntoTakesRemote.rejected, setRejected)

            .addCase(explodeClipTakesRemote.rejected, setRejected)

            .addCase(duplicateClipTakeRemote.rejected, setRejected)
            .addCase(removeClipTakeRemote.rejected, setRejected)
            .addCase(renameClipTakeRemote.rejected, setRejected)
            .addCase(addClipTakeFromMediaRemote.rejected, setRejected)

            .addCase(packClipsIntoTakesRemote.fulfilled, (state, action) => {
                const payload = action.payload as { ok?: boolean } & TimelineState;
                if (!payload.ok) {
                    state.error = "Pack into takes failed";
                    state.status = "Failed";
                    return;
                }
                applyTimelineStatePreservingPlayhead(state, payload);
            })

            .addCase(explodeClipTakesRemote.fulfilled, (state, action) => {
                const payload = action.payload as { ok?: boolean } & TimelineState;
                if (!payload.ok) {
                    state.error = "Explode takes failed";
                    state.status = "Failed";
                    return;
                }
                applyTimelineStatePreservingPlayhead(state, payload);
            })

            .addCase(duplicateClipTakeRemote.fulfilled, (state, action) => {
                const payload = action.payload as { ok?: boolean } & TimelineState;
                if (!payload.ok) {
                    state.error = "Duplicate take failed";
                    state.status = "Failed";
                    return;
                }
                applyTimelineStatePreservingPlayhead(state, payload);
            })
            .addCase(removeClipTakeRemote.fulfilled, (state, action) => {
                const payload = action.payload as { ok?: boolean } & TimelineState;
                if (!payload.ok) {
                    state.error = "Remove take failed";
                    state.status = "Failed";
                    return;
                }
                applyTimelineStatePreservingPlayhead(state, payload);
            })
            .addCase(renameClipTakeRemote.fulfilled, (state, action) => {
                const payload = action.payload as { ok?: boolean } & TimelineState;
                if (!payload.ok) {
                    state.error = "Rename take failed";
                    state.status = "Failed";
                    return;
                }
                applyTimelineStatePreservingPlayhead(state, payload);
            })
            .addCase(addClipTakeFromMediaRemote.fulfilled, (state, action) => {
                const payload = action.payload as { ok?: boolean } & TimelineState;
                if (!payload.ok) {
                    state.error = "Add take from media failed";
                    state.status = "Failed";
                    return;
                }
                applyTimelineStatePreservingPlayhead(state, payload);
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

            .addCase(replaceMidiClipDataRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                applyTimelineState(state, payload, { force: true });
            })

            .addCase(selectClipRemote.pending, (state, action) => {
                const { clipId, preserveTrackFocus } = parseSelectClipRemoteArg(action.meta.arg);
                state.selectedClipId = clipId;
                state.selectedPointId = null;
                if (clipId) {
                    const nextTrackId = resolveTrackIdForClipSelection({
                        currentTrackId: state.selectedTrackId,
                        clips: state.clips,
                        clipId,
                        preserveTrackFocus,
                    });
                    if (nextTrackId !== state.selectedTrackId) {
                        state.selectedTrackId = nextTrackId;
                    }
                    ensureClipAutomation(state, clipId);
                }
            })

            .addCase(selectClipRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                    __preserveTrackFocus?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                const currentSelectedTrackId = state.selectedTrackId;
                state.selectedClipId = payload.selected_clip_id;
                state.selectedPointId = null;
                if (payload.selected_clip_id) {
                    ensureClipAutomation(state, payload.selected_clip_id);
                }
                if (payload.__preserveTrackFocus) {
                    state.selectedTrackId = currentSelectedTrackId;
                } else if (payload.selected_track_id !== undefined) {
                    state.selectedTrackId = payload.selected_track_id;
                }
            })

            .addCase(setTrackStateRemote.pending, (state, action) => {
                applyOptimisticTrackState(state, action.meta.arg);
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
                // 浅拷贝快照，防止 draft 代理被后续 prune 透过修改
                const currentClipPitchCurves = { ...state.clipPitchCurves };
                applyTimelineTracksOnly(state, payload);
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
                // 与 Tempo Map 变化点一致的 BPM 范围（10-960）。
                state.bpm = clamp(Number(payload.bpm ?? state.bpm), 10, 960);
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
                const backendSec = Number(payload.playhead_sec ?? requestedSec);
                // 仅当【前端 playheadSec 仍等于本次请求值】（未被更新的请求
                // 覆盖）且后端返回值与请求参数不同（发生了 clamp 之类修正）
                // 时才采纳后端值 —— 缺少前一半条件时，先发请求的迟到响应
                // 会把光标从新位置拖回旧位置。
                const EPS = 0.001;
                if (
                    Math.abs(state.playheadSec - requestedSec) <= EPS &&
                    Math.abs(backendSec - requestedSec) > EPS
                ) {
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
                // 交互锁期间（如拖拽中）仅同步轨道列表，
                // 避免 add_track 的后端快照覆盖前端 clip 乐观位置并产生闪烁。
                if (state._interactionLockCount > 0) {
                    applyTimelineTracksOnly(state, payload);
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

            .addCase(selectTrackRemote.pending, (state, action) => {
                state.selectedTrackId =
                    typeof action.meta.arg === "string" ? action.meta.arg : action.meta.arg.trackId;
            })

            .addCase(selectTrackRemote.fulfilled, (state, action) => {
                const payload = action.payload as {
                    ok?: boolean;
                } & TimelineState;
                if (!payload.ok) {
                    return;
                }
                const currentPlayheadSec = state.playheadSec;
                applyTimelineTracksOnly(state, payload);
                // applySelectedClip: false（点击轨道空白切轨）时不得恢复后端
                // 记住的 selected_clip_id —— 否则异步 fulfilled 会把刚完成的
                // "点击空白取消选中"覆盖回去。
                const applySelectedClip =
                    typeof action.meta.arg !== "object" ||
                    action.meta.arg.applySelectedClip !== false;
                if (applySelectedClip && payload.selected_clip_id !== undefined) {
                    state.selectedClipId = payload.selected_clip_id;
                }
                state.playheadSec = currentPlayheadSec;
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
                const payload = action.payload as TrackSummaryResult | { ok?: false };
                if (!payload.ok) {
                    return;
                }
                // 过期响应丢弃：快速连续切轨时，旧轨道的迟到摘要不得覆盖
                // 当前选中轨道。
                if (payload.track_id !== state.selectedTrackId) {
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
    setTrackMeters,
    clearTrackMeters,
    checkpointHistory,
    applyTimelinePayload,
    setToolMode,
    setEditParam,
    setBpm,
    setBeats,
    setGrid,
    setPrimaryTimeUnit,
    setSecondaryTimeUnit,
    setRulerLabelSpacingPx,
    setShowPlayheadTimeInTrackHeader,
    setParamEditorSyncTimeline,
    toggleAutoCrossfade,
    toggleShowAllTakes,
    toggleSyncEditsAcrossTakes,
    toggleSplitTransition,
    setSplitTransitionMode,
    setSplitTransitionDurationUnit,
    setSplitTransitionDurationSec,
    setSplitTransitionDurationPercent,
    setSplitTransitionCurve,
    setSplitTransitionOverlapCrossfade,
    toggleSnap,
    setTimelineSnapSettings,
    setTempoMap,
    setTempoMapVisible,
    toggleTempoMapVisible,
    togglePitchSnap,
    setPitchSnapUnit,
    setPitchSnapScale,
    togglePlayheadZoom,
    toggleAutoScroll,
    toggleIgnoreGrouping,
    cycleRippleMode,
    setRippleMode,
    toggleParamEditorSeekPlayhead,
    toggleParamEditorTimelineClickSelectTrack,
    toggleClipboardPreview,
    toggleParamValuePopup,
    cycleDragDirection,
    setDragDirection,
    setEdgeSmoothnessPercent,
    setplayheadSec,
    setPendingPlayheadReveal,
    setModelDir,
    setAudioPath,
    setOutputPath,
    setPitchShift,
    setProjectNotesMarkdown,
    closeVocalShifterSkippedFilesDialog,
    closeReaperSkippedFilesDialog,
    closeSaveVersionConflictDialog,
    setPitchSnapToleranceCents,
    setScaleHighlightMode,
    upsertCustomScalePreset,
    removeCustomScalePreset,
    toggleLockParamLines,
    toggleQuickSearchAutoNormalize,
    setDefaultStretchAlgorithm,
    setDefaultHifiganMelStretch,
    setOrtEp,
    setGpuDeviceId,
    setOrtDeviceId,
    toggleAutoBackgroundRender,
    setVisibleReferenceRootTrackIds,
    toggleVisibleReferenceRootTrackId,
    setSelectedClip,
    setSelectedClipPreservingTrack,
    setMultiSelectedClipIds,
    setClipboardOperationFailed,
    moveClipStart,
    moveClipTrack,
    setClipLength,
    setClipSnapOffset,
    setClipPlaybackRate,
    setClipSourceRange,
    setClipFades,
    setClipAutoFades,
    setClipGain,
    setClipMuted,
    setClipsGroupId,
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
    setClipFormantStatus,
    setClipFormantAnalysis,
    openClipFormantToolWindow,
    setClipFormantToolWindowPosition,
    closeClipFormantToolWindow,
    removeClipPitchData,
    bumpParamsEpoch,
} = sessionSlice.actions;

export default sessionSlice.reducer;
