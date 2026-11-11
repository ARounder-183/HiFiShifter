// Session 的轻量设置 helper 与工具模式派生。拆自 sessionSlice。
import type {
    GridSize,
    SelectToolMode,
    TimeUnit,
    TimeUnitChoice,
    TimelineSnapSettings,
    ToolMode,
} from "../sessionTypes";
import type { SessionState } from "./types";
import { isBlockSelectionActive } from "../toolMode";

export const MAX_TRACK_VOLUME = 4;
/** 播放光标"真实挪动"判定阈值：低于该值的差异视为浮点噪声，
 * 不触发撤销/重做后的"聚焦播放光标"（视图滚动）登记。 */
export const PLAYHEAD_MOVE_EPS_SEC = 1e-6;
export const VALID_GRID_SIZES = new Set<GridSize>([
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

export const VALID_TIME_UNITS = new Set<TimeUnit>(["barBeats", "barDivisions", "seconds", "clock"]);
export const DEFAULT_PRIMARY_TIME_UNIT: TimeUnit = "barBeats";
export const DEFAULT_SECONDARY_TIME_UNIT: TimeUnitChoice = "clock";
export const DEFAULT_RULER_LABEL_SPACING_PX = 110;

/**
 * 旧的"仅显示媒体文件"存储位置。
 *
 * 该键由 `fileBrowserSlice` 写入（`hifishifter.fileBrowser.audioOnly`）。视图选项
 * 迁到 `UiSettings.fileBrowser` 之后，老用户的偏好在旧键里 —— 读取一次用于迁移，
 * 之后以配置为准（见 `migrateLegacyMediaOnly`）。
 */
const LEGACY_MEDIA_ONLY_KEY = "hifishifter.fileBrowser.audioOnly";

export function readLegacyMediaOnly(): string | null {
    try {
        return localStorage.getItem(LEGACY_MEDIA_ONLY_KEY);
    } catch {
        // 无 localStorage 的环境（node 环境的单测）→ 没有可迁移的旧值。
        return null;
    }
}

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

export function normalizeTimelineSnapSettings(
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

/**
 * 把一次工具族切换写进状态。
 *
 * 【只写工具身份】当前工具族与绘制族记忆。
 *
 * 【不写设置】旧实现会在进入 block 子模式时把块图层强制置为可见，因为那时
 * 「图层可见」是一个状态位，而「在块模式里却看不见块」被当成不可达状态。
 * 可见性改成派生之后（见 isParamBlockLayerVisible）这个补丁失去理由：块模式
 * 自己就让图层可见，不需要任何人代写设置。
 *
 * `selectToolMode` 是选择族的**当前 / 上次**子模式：切到绘制族时保留它，Tab
 * 回到选择族时才能恢复上次的 range / block。块是否**正在**激活只看
 * `isBlockSelectionActive(toolMode, selectToolMode)`，不由这里另存布尔值。
 */
export function applyToolMode(state: SessionState, mode: ToolMode): void {
    state.toolMode = mode;
    if (mode !== "select") state.drawToolMode = mode;
}

/** 写入选择族子模式（连带把工具族切到选择族）；不碰任何设置。 */
export function applySelectToolMode(state: SessionState, mode: SelectToolMode): void {
    state.toolMode = "select";
    state.selectToolMode = mode;
}

/**
 * 块层**是否应当绘制** —— 全仓唯一判据。
 *
 * 【两根轴】常显设置（用户偏好）与块选择子模式（此刻在做什么）。两者是**或**关系：
 * 块编辑永远需要看见块；常显设置只负责「不在块编辑时也看见」。
 *
 * 【为什么不是状态】派生结果没有写入者，也就不可能出现「工具已经退出块模式、图层
 * 却还开着」或「图层关着、工具还在块模式」这两种需要额外规则去消灭的状态。
 */
export function isParamBlockLayerVisible(args: {
    readonly alwaysShowParamBlocks: boolean;
    readonly toolMode: ToolMode;
    readonly selectToolMode: SelectToolMode;
}): boolean {
    return args.alwaysShowParamBlocks || isBlockSelectionActive(args.toolMode, args.selectToolMode);
}

/**
 * 写入「永远显示参数块」设置。
 *
 * 【没有副作用】它不改工具、不改选择子模式。旧实现（applyParamBlocksVisible）在
 * 关闭时把 selectToolMode 收回 range，那是「图层被工具强制打开」这一设计的补丁；
 * 该设计已删除，补丁一并删除。
 */
export function applyAlwaysShowParamBlocks(state: SessionState, value: boolean): void {
    state.alwaysShowParamBlocks = value;
}

/** 外部设置输入的白名单归一：只认 block，其余一律 range。 */
export function normalizeSelectToolMode(value: unknown): SelectToolMode {
    return value === "block" ? "block" : "range";
}
