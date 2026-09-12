/**
 * stretchGroup.ts - 多选 Clip 统一拉伸几何计算。
 *
 * 作用：
 * - 判定当前边缘拖拽是否应触发“多选整体拉伸”。
 * - 计算固定一侧（左/右）时，所有 Clip 的新 start/length/clipPlaybackRate。
 * - 计算「被编辑区域右缘的净位移」（波纹跟随预览的驱动量，见
 *   `computeRegionRightEdgeDelta`）——旧实现与渲染内核共用同一份，避免两套波纹语义。
 */
import type { ClipInfo } from "../../../../features/session/sessionTypes";
import { clamp } from "../math";

export type StretchEdge = "stretch_left" | "stretch_right";

export type StretchGroupClipInitial = {
    clipId: string;
    startSec: number;
    endSec: number;
    lengthSec: number;
    clipPlaybackRate: number;
    fadeInSec: number;
    fadeOutSec: number;
    trackId: string;
};

export type StretchGroupState = {
    clipIds: string[];
    minStartSec: number;
    maxEndSec: number;
    spanSec: number;
    initialById: Record<string, StretchGroupClipInitial>;
};

export type StretchGroupClipNext = {
    startSec: number;
    lengthSec: number;
    clipPlaybackRate: number;
    fadeInSec: number;
    fadeOutSec: number;
};

export type StretchGroupUpdate = {
    scale: number;
    groupStartSec: number;
    groupEndSec: number;
    byId: Record<string, StretchGroupClipNext>;
};

const EDGE_EPSILON_SEC = 1e-6;
const MIN_SPAN_SEC = 1e-6;
const MAX_TIMELINE_SEC = 10_000;

function normalizePlaybackRate(rate: number): number {
    if (!Number.isFinite(rate) || rate <= 0) return 1;
    return rate;
}

function normalizeLength(lengthSec: number): number {
    if (!Number.isFinite(lengthSec) || lengthSec <= 0) return MIN_SPAN_SEC;
    return Math.max(MIN_SPAN_SEC, lengthSec);
}

function normalizeFade(fadeSec: number, lengthSec: number): number {
    if (!Number.isFinite(fadeSec) || fadeSec <= 0) return 0;
    return clamp(fadeSec, 0, Math.max(0, lengthSec));
}

export function scaleClipFadesForStretch(params: {
    baseFadeInSec: number;
    baseFadeOutSec: number;
    baseLengthSec: number;
    nextLengthSec: number;
}): { fadeInSec: number; fadeOutSec: number } {
    const baseLengthSec = normalizeLength(params.baseLengthSec);
    const nextLengthSec = Math.max(0, Number(params.nextLengthSec) || 0);
    const ratio = nextLengthSec / baseLengthSec;

    const baseFadeInSec = normalizeFade(params.baseFadeInSec, baseLengthSec);
    const baseFadeOutSec = normalizeFade(params.baseFadeOutSec, baseLengthSec);

    return {
        fadeInSec: normalizeFade(baseFadeInSec * ratio, nextLengthSec),
        fadeOutSec: normalizeFade(baseFadeOutSec * ratio, nextLengthSec),
    };
}

/**
 * SnapOffset 随拉伸同步缩放。
 *
 * 语义：偏移点标记的是 Clip **内容**中的位置，因此随长度按**总比例**线性缩放，
 * 并钳制到新长度内（越界会让三角跑到 Clip 之外）。基准偏移 <= 0 时恒为 0。
 *
 * 特殊说明：这是旧实现（`useEditDrag`）与渲染内核共用的唯一实现——两处各写一份
 * 会让「拉伸后三角位置」在两个渲染模式下不一致。
 *
 * @param baseOffsetSec 拖拽开始时的偏移（秒）。
 * @param totalRatio 新长度 / 基准长度。
 * @param nextLengthSec 新长度（秒）。
 * @returns 缩放后的偏移（秒）。
 */
export function scaleSnapOffsetForStretch(
    baseOffsetSec: number | undefined,
    totalRatio: number,
    nextLengthSec: number,
): number {
    const base = Number(baseOffsetSec) || 0;
    if (!(base > 0)) return 0;
    const safeRatio = Number.isFinite(totalRatio) && totalRatio > 0 ? totalRatio : 1;
    return clamp(base * safeRatio, 0, Math.max(0, nextLengthSec));
}

/** 单 clip 拉伸的输入（全部为**按下时**的基准值）。 */
export interface ClipStretchArgs {
    /** 拉伸哪一侧（决定固定哪条边）。 */
    readonly edge: StretchEdge;
    /** 指针所在的工程时间（秒，已含吸附结果）。 */
    readonly pointerSec: number;
    /** 按下时的起点（秒）。 */
    readonly baseStartSec: number;
    /** 按下时的长度（秒）。 */
    readonly baseLengthSec: number;
    /** 按下时的播放速率（`clipPlaybackRate`）。 */
    readonly basePlaybackRate: number;
    readonly baseFadeInSec: number;
    readonly baseFadeOutSec: number;
    readonly baseSnapOffsetSec?: number;
    /** 最小长度（秒）；缺省 0。 */
    readonly minLengthSec?: number;
}

/** 单 clip 拉伸的结果（绝对值，供乐观写入与提交共用）。 */
export interface ClipStretchResult {
    readonly startSec: number;
    readonly lengthSec: number;
    readonly clipPlaybackRate: number;
    readonly fadeInSec: number;
    readonly fadeOutSec: number;
    readonly snapOffsetSec: number;
    /** 新长度 / 基准长度（调用方可能还要按比例缩放其它随长度变化的值）。 */
    readonly scale: number;
}

/** 拉伸时的长度上限（秒）：与旧实现一致的防御性上界。 */
const STRETCH_MAX_LENGTH_SEC = 10_000;
/** 播放速率上下限（与旧实现一致）。 */
const STRETCH_MIN_RATE = 0.1;
const STRETCH_MAX_RATE = 10;

/**
 * 单 clip 拉伸（`Alt` + 拖左右边缘）：保持**对侧边缘固定**，按新长度反算播放速率。
 *
 * 流程（与旧实现 `useEditDrag` 的 `stretch_left` / `stretch_right` 逐行等价）：
 * 1. 由指针位置求「期望长度」（左拉伸：`右缘 − 指针`；右拉伸：`指针 − 左缘`），
 *    钳制到 `[minLength, 上限]`；
 * 2. 反算速率 `rate = baseRate × baseLength / 期望长度`，钳制到 `[0.1, 10]`；
 * 3. 用**钳制后的速率**回算实际长度（`baseRate × baseLength / rate`）——否则速率被
 *    钳制时长度会与实际速率不匹配（画面长度与音频时长对不上）；
 * 4. 淡变按长度比例缩放、SnapOffset 按总比例缩放。
 *
 * 特殊说明：`startSec` 对左拉伸是「固定右缘 − 实际长度」，对右拉伸**保持基准起点**。
 *
 * @param args 见 `ClipStretchArgs`。
 * @returns 见 `ClipStretchResult`。
 */
export function computeClipStretch(args: ClipStretchArgs): ClipStretchResult {
    const minLen = Math.max(MIN_SPAN_SEC, Number(args.minLengthSec) || 0);
    const baseLen = Math.max(MIN_SPAN_SEC, Number(args.baseLengthSec) || 0);
    const baseRate =
        Number(args.basePlaybackRate) > 0 && Number.isFinite(args.basePlaybackRate)
            ? Number(args.basePlaybackRate)
            : 1;
    const fixedEdgeSec =
        args.edge === "stretch_left" ? args.baseStartSec + baseLen : args.baseStartSec;

    // 期望长度：左拉伸固定右缘（指针不能越过 `右缘 − 最小长度`），右拉伸固定左缘。
    const desiredLength =
        args.edge === "stretch_left"
            ? clamp(
                  fixedEdgeSec - clamp(args.pointerSec, 0, fixedEdgeSec - minLen),
                  minLen,
                  STRETCH_MAX_LENGTH_SEC,
              )
            : clamp(
                  clamp(args.pointerSec, args.baseStartSec + minLen, STRETCH_MAX_LENGTH_SEC) -
                      args.baseStartSec,
                  minLen,
                  STRETCH_MAX_LENGTH_SEC,
              );

    const nextRate = clamp(
        (baseRate * baseLen) / Math.max(MIN_SPAN_SEC, desiredLength),
        STRETCH_MIN_RATE,
        STRETCH_MAX_RATE,
    );
    const correctedLength = (baseRate * baseLen) / nextRate;
    const startSec =
        args.edge === "stretch_left" ? fixedEdgeSec - correctedLength : args.baseStartSec;
    const scale = correctedLength / baseLen;
    const scaledFades = scaleClipFadesForStretch({
        baseFadeInSec: args.baseFadeInSec,
        baseFadeOutSec: args.baseFadeOutSec,
        baseLengthSec: baseLen,
        nextLengthSec: correctedLength,
    });

    return {
        startSec,
        lengthSec: correctedLength,
        clipPlaybackRate: nextRate,
        fadeInSec: scaledFades.fadeInSec,
        fadeOutSec: scaledFades.fadeOutSec,
        snapOffsetSec: scaleSnapOffsetForStretch(args.baseSnapOffsetSec, scale, correctedLength),
        scale,
    };
}

export function buildStretchGroupState(params: {
    clips: ClipInfo[];
    selectedClipIds: string[];
    anchorClipId: string;
    edge: StretchEdge;
}): StretchGroupState | null {
    const { clips, selectedClipIds, anchorClipId, edge } = params;
    if (selectedClipIds.length < 2) {
        return null;
    }

    const clipMap = new Map(clips.map((clip) => [clip.id, clip]));
    const dedupedIds = Array.from(new Set(selectedClipIds));

    const initialById: Record<string, StretchGroupClipInitial> = {};
    let minStartSec = Number.POSITIVE_INFINITY;
    let maxEndSec = Number.NEGATIVE_INFINITY;

    for (const clipId of dedupedIds) {
        const clip = clipMap.get(clipId);
        if (!clip) continue;
        const startSec = Number(clip.startSec) || 0;
        const lengthSec = Math.max(0, Number(clip.lengthSec) || 0);
        const endSec = startSec + lengthSec;
        initialById[clipId] = {
            clipId,
            startSec,
            endSec,
            lengthSec,
            clipPlaybackRate: normalizePlaybackRate(Number(clip.clipPlaybackRate ?? 1) || 1),
            fadeInSec: normalizeFade(Number(clip.fadeInSec) || 0, lengthSec),
            fadeOutSec: normalizeFade(Number(clip.fadeOutSec) || 0, lengthSec),
            trackId: String(clip.trackId),
        };
        minStartSec = Math.min(minStartSec, startSec);
        maxEndSec = Math.max(maxEndSec, endSec);
    }

    const clipIds = Object.keys(initialById);
    if (clipIds.length < 2) {
        return null;
    }

    const anchor = initialById[anchorClipId];
    if (!anchor) {
        return null;
    }

    const isBoundaryAnchor =
        edge === "stretch_left"
            ? Math.abs(anchor.startSec - minStartSec) <= EDGE_EPSILON_SEC
            : Math.abs(anchor.endSec - maxEndSec) <= EDGE_EPSILON_SEC;

    if (!isBoundaryAnchor) {
        return null;
    }

    return {
        clipIds,
        minStartSec,
        maxEndSec,
        spanSec: Math.max(MIN_SPAN_SEC, maxEndSec - minStartSec),
        initialById,
    };
}

export function computeStretchGroupUpdate(params: {
    group: StretchGroupState;
    edge: StretchEdge;
    pointerSec: number;
}): StretchGroupUpdate {
    const { group, edge, pointerSec } = params;

    const nextGroupStart =
        edge === "stretch_left"
            ? clamp(pointerSec, 0, group.maxEndSec - MIN_SPAN_SEC)
            : group.minStartSec;
    const nextGroupEnd =
        edge === "stretch_left"
            ? group.maxEndSec
            : clamp(pointerSec, group.minStartSec + MIN_SPAN_SEC, MAX_TIMELINE_SEC);

    const nextSpanSec = Math.max(MIN_SPAN_SEC, nextGroupEnd - nextGroupStart);
    const scale = nextSpanSec / Math.max(MIN_SPAN_SEC, group.spanSec);

    const byId: Record<string, StretchGroupClipNext> = {};
    for (const clipId of group.clipIds) {
        const initial = group.initialById[clipId];
        if (!initial) continue;

        const relStart = initial.startSec - group.minStartSec;
        const relEnd = initial.endSec - group.minStartSec;

        const startSec = nextGroupStart + relStart * scale;
        const endSec = nextGroupStart + relEnd * scale;
        const lengthSec = Math.max(MIN_SPAN_SEC, endSec - startSec);
        const clipPlaybackRate = clamp(
            (initial.clipPlaybackRate * Math.max(MIN_SPAN_SEC, initial.lengthSec)) /
                Math.max(MIN_SPAN_SEC, lengthSec),
            0.1,
            10,
        );
        const scaledFades = scaleClipFadesForStretch({
            baseFadeInSec: initial.fadeInSec,
            baseFadeOutSec: initial.fadeOutSec,
            baseLengthSec: initial.lengthSec,
            nextLengthSec: lengthSec,
        });

        byId[clipId] = {
            startSec,
            lengthSec,
            clipPlaybackRate,
            fadeInSec: scaledFades.fadeInSec,
            fadeOutSec: scaledFades.fadeOutSec,
        };
    }

    return {
        scale,
        groupStartSec: nextGroupStart,
        groupEndSec: nextGroupEnd,
        byId,
    };
}

/**
 * 「被编辑区域右缘」的净位移（**带符号**，秒）。
 *
 * 【用途】波纹跟随预览的驱动量：把跟随集按本位移平移，用户就能在拖拽过程中看到
 * 后续 clip 自动跟进（提交后的权威结果仍由后端计算）。
 *
 * 【为什么取"右缘净位移"而不是"拖拽位移"】与后端区域化波纹一致——平移量 = 区域
 * 右缘的**实际**位移（已含吸附、素材长度限制等约束后的真值），这样"预览 → 提交"
 * 不会跳变。取拖拽原始位移时，约束把边缘卡住后跟随集仍继续移动，松手瞬间回跳。
 *
 * ⚠️ 必须是**带符号**：拖右缘向左（缩短 / 截短）时位移为负，跟随剪辑要向左收拢。
 * 不能用"对 0 取 max"或"对各成员取最大正位移"的方式，否则负位移会被吞掉、
 * 表现为"向右正常而向左无实时波纹"（曾为此引入 bug）。
 *
 * @param clipIds 本次编辑的参与者（锚点 + 多选 + 编组展开后的全部成员）。
 * @param baseById 各参与者按下时的几何（`startSec` / `lengthSec`）。
 * @param clips 当前（乐观更新后）的 session clips。
 * @returns 右缘净位移（秒）；无有效数据时为 0。
 */
export function computeRegionRightEdgeDelta(args: {
    readonly clipIds: readonly string[];
    readonly baseById: Readonly<Record<string, { startSec: number; lengthSec: number }>>;
    readonly clips:
        | readonly ClipInfo[]
        | readonly { id: string; startSec: number; lengthSec: number }[];
}): number {
    let maxOldRight = Number.NEGATIVE_INFINITY;
    let maxNewRight = Number.NEGATIVE_INFINITY;
    for (const id of args.clipIds) {
        const base = args.baseById[id];
        const now = args.clips.find((clip) => clip.id === id);
        if (!base || !now) continue;
        maxOldRight = Math.max(maxOldRight, base.startSec + base.lengthSec);
        maxNewRight = Math.max(maxNewRight, Number(now.startSec) + Number(now.lengthSec));
    }
    if (!Number.isFinite(maxOldRight) || !Number.isFinite(maxNewRight)) {
        return 0;
    }
    return maxNewRight - maxOldRight;
}
