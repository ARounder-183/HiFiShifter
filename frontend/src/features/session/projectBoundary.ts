import type { ClipInfo } from "./sessionTypes";

const EMPTY_PROJECT_BOUNDARY_SEC = 30;

export function getDynamicProjectSec(clips: ClipInfo[]): number {
    if (!Array.isArray(clips) || clips.length === 0) {
        return EMPTY_PROJECT_BOUNDARY_SEC;
    }

    let maxEndSec = 0;
    for (const clip of clips) {
        const startSec = Math.max(0, Number(clip.startSec) || 0);
        const lengthSec = Math.max(0, Number(clip.lengthSec) || 0);
        const endSec = startSec + lengthSec;
        if (endSec > maxEndSec) maxEndSec = endSec;
    }

    return Math.max(1, maxEndSec);
}
