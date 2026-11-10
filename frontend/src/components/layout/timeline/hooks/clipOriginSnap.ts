/**
 * Clip drag-origin snap.
 *
 * The old "vertical lock" zeroed the horizontal delta whenever the pointer
 * crossed a track. The intended behavior is a horizontal magnetic snap back to
 * the clip's pre-drag start, not a track-row snap: the user can still drag
 * diagonally, but small horizontal drift while changing tracks is removed.
 */

export const CLIP_ORIGIN_SNAP_MIN_PX = 24;
export const CLIP_ORIGIN_SNAP_MAX_PX = 32;
export const CLIP_ORIGIN_SNAP_SCALE = 0.06;

export function resolveClipOriginSnapThresholdPx(pxPerSec: number): number {
    if (!Number.isFinite(pxPerSec) || pxPerSec <= 0) return CLIP_ORIGIN_SNAP_MIN_PX;
    return Math.round(
        Math.min(
            CLIP_ORIGIN_SNAP_MAX_PX,
            Math.max(CLIP_ORIGIN_SNAP_MIN_PX, pxPerSec * CLIP_ORIGIN_SNAP_SCALE),
        ),
    );
}

export interface ClipOriginSnapInput {
    readonly rawStartSec: number;
    readonly originStartSec: number;
    readonly pxPerSec: number;
    /** True when the target track differs from the drag's origin track. */
    readonly crossTrack: boolean;
    /** False when snapping is globally disabled or the no-snap modifier is held. */
    readonly enabled: boolean;
}

export interface ClipOriginSnapResult {
    readonly active: boolean;
    readonly startSec: number;
    readonly thresholdPx: number;
    readonly distancePx: number;
}

export function resolveClipOriginSnap(input: ClipOriginSnapInput): ClipOriginSnapResult {
    const thresholdPx = resolveClipOriginSnapThresholdPx(input.pxPerSec);
    const distancePx =
        Number.isFinite(input.rawStartSec) &&
        Number.isFinite(input.originStartSec) &&
        Number.isFinite(input.pxPerSec)
            ? Math.abs(input.rawStartSec - input.originStartSec) * input.pxPerSec
            : Number.POSITIVE_INFINITY;

    if (
        !input.enabled ||
        !input.crossTrack ||
        !Number.isFinite(input.rawStartSec) ||
        !Number.isFinite(input.originStartSec) ||
        !Number.isFinite(input.pxPerSec) ||
        input.pxPerSec <= 0 ||
        distancePx > thresholdPx + 1e-9
    ) {
        return {
            active: false,
            startSec: input.rawStartSec,
            thresholdPx,
            distancePx,
        };
    }

    return {
        active: true,
        startSec: input.originStartSec,
        thresholdPx,
        distancePx,
    };
}
