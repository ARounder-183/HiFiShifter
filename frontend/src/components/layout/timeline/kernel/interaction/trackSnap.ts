/**
 * Timeline track drop snapping.
 *
 * Replaces the old "vertical lock" (which zeroed horizontal movement whenever a
 * clip crossed a row). This helper only adds a small hysteresis around row
 * boundaries so a vertical drag does not flicker between adjacent tracks; the
 * horizontal delta is never touched.
 */

export interface TrackSnapInput {
    readonly contentY: number;
    readonly rowHeight: number;
    readonly trackCount: number;
    /** Last accepted target index; null before the first frame. */
    readonly previousIndex: number | null;
    readonly thresholdPx?: number;
}

export interface TrackSnapResult {
    /** 0..trackCount; trackCount is the "new track" sentinel row. */
    readonly index: number;
    readonly snapped: boolean;
    /** Row boundary to highlight, or null when the target did not change. */
    readonly boundaryIndex: number | null;
}

const DEFAULT_MIN_THRESHOLD_PX = 4;
const DEFAULT_MAX_THRESHOLD_PX = 10;

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

function finiteIndex(value: number | null, trackCount: number): number | null {
    if (value === null || !Number.isFinite(value)) return null;
    const index = Math.trunc(value);
    return index >= 0 && index <= trackCount ? index : null;
}

/**
 * Resolve the target row with a small "must be inside the new row" threshold.
 *
 * The sentinel row (below all tracks) gets the same treatment: the pointer must
 * be at least threshold px past the last row's bottom before the target becomes
 * "new track".
 */
export function resolveTrackSnap(input: TrackSnapInput): TrackSnapResult {
    const trackCount = Number.isFinite(input.trackCount)
        ? Math.max(0, Math.floor(input.trackCount))
        : 0;
    const rowHeight = Number.isFinite(input.rowHeight) && input.rowHeight > 0 ? input.rowHeight : 0;
    if (trackCount <= 0 || rowHeight <= 0) {
        return { index: -1, snapped: false, boundaryIndex: null };
    }

    const contentY = Number.isFinite(input.contentY) ? Math.max(0, input.contentY) : 0;
    const rawIndex = clamp(Math.floor(contentY / rowHeight), 0, trackCount);
    const previous = finiteIndex(input.previousIndex, trackCount);
    const threshold = clamp(
        input.thresholdPx ?? rowHeight * 0.18,
        DEFAULT_MIN_THRESHOLD_PX,
        DEFAULT_MAX_THRESHOLD_PX,
    );

    if (previous === null || rawIndex === previous) {
        return { index: rawIndex, snapped: false, boundaryIndex: null };
    }

    let index = rawIndex;
    if (rawIndex > previous) {
        const intoNewRow =
            rawIndex >= trackCount
                ? contentY - trackCount * rowHeight
                : contentY - rawIndex * rowHeight;
        if (intoNewRow < threshold) index = previous;
    } else {
        // Moving up: the pointer must leave the old row past its top boundary.
        const pastOldTop = previous * rowHeight - contentY;
        if (pastOldTop < threshold) index = previous;
    }

    if (index === previous) {
        return { index, snapped: false, boundaryIndex: null };
    }
    return {
        index,
        snapped: true,
        boundaryIndex: index > previous ? index : previous,
    };
}
