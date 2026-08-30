/**
 * stretchGroup.ts - 多选 Clip 统一拉伸几何计算。
 *
 * 作用：
 * - 判定当前边缘拖拽是否应触发“多选整体拉伸”。
 * - 计算固定一侧（左/右）时，所有 Clip 的新 start/length/clipPlaybackRate。
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
