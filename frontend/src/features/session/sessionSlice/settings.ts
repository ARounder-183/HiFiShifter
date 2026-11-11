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
 * 【唯一实现】用户切换（`setToolMode`）与启动恢复（`loadUiSettings.fulfilled`）
 * 必须落到同一组字段上：当前工具族、绘制族记忆，以及选择族记忆所要求的图层
 * 可见性。两处各写一遍，迟早会漏掉一条不变量。
 *
 * `selectToolMode` 是选择族的**当前 / 上次**子模式：切到绘制族时保留它，Tab
 * 回到选择族时才能恢复上次的 range / block。块是否**正在**激活只看
 * `toolMode === "select" && selectToolMode === "block"`，不由这里另存布尔值。
 */
export function applyToolMode(state: SessionState, mode: ToolMode): void {
    state.toolMode = mode;
    if (mode !== "select") state.drawToolMode = mode;
    // block 子模式必须能看见块层；这是状态不变量，不靠组件 effect 补。
    if (mode === "select" && state.selectToolMode === "block") {
        state.paramBlocksVisible = true;
    }
}

/** 写入选择族子模式；进入 block 时同步打开块层。 */
export function applySelectToolMode(state: SessionState, mode: SelectToolMode): void {
    state.toolMode = "select";
    state.selectToolMode = mode;
    if (mode === "block") state.paramBlocksVisible = true;
}

/**
 * 写入参数块图层可见性。
 *
 * 图层关闭时不可能继续做块选择，因此在 reducer 收口为 range，避免出现
 * “工具仍想选块、图层却已隐藏”的死状态。重新打开图层不会自动回到 block ——
 * 可见性与工具子模式是两根轴，用户需要显式再选块。
 */
export function applyParamBlocksVisible(state: SessionState, visible: boolean): void {
    state.paramBlocksVisible = visible;
    if (!visible) state.selectToolMode = "range";
}

/** 外部设置输入的白名单归一：只认 block，其余一律 range。 */
export function normalizeSelectToolMode(value: unknown): SelectToolMode {
    return value === "block" ? "block" : "range";
}
