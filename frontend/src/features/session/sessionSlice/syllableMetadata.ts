import type { PayloadAction } from "@reduxjs/toolkit";
import type { SessionState } from "./types";

/** GAME changes block metadata, never the parameter/audio revision. */
export function applyClipSyllableAnalysisReducer(
    state: SessionState,
    action: PayloadAction<{
        clipId: string;
        takeId: string;
        sourcePath?: string;
        analyzed: boolean;
        sourceSpansSec: Array<{ startSec: number; endSec: number }>;
    }>,
) {
    const result = action.payload;
    if (
        !result ||
        !result.analyzed ||
        !Array.isArray(result.sourceSpansSec) ||
        result.sourceSpansSec.length === 0 ||
        result.sourceSpansSec.length > 20000
    )
        return;
    if (
        result.sourceSpansSec.some(
            (span, index, spans) =>
                !Number.isFinite(span.startSec) ||
                !Number.isFinite(span.endSec) ||
                span.endSec <= span.startSec ||
                (index > 0 && span.startSec < spans[index - 1].endSec - 1e-9),
        )
    )
        return;
    const clip = state.clips.find((clip) => clip.id === result.clipId);
    const take = clip?.takes?.find((take) => take.id === result.takeId);
    if (
        !clip ||
        !take ||
        (result.sourcePath !== undefined &&
            (take.sourcePath ?? clip.sourcePath) !== result.sourcePath)
    )
        return;
    const previous = take.syllableSpansSec;
    if (
        previous?.length === result.sourceSpansSec.length &&
        previous.every(
            (span, index) =>
                span.startSec === result.sourceSpansSec[index].startSec &&
                span.endSec === result.sourceSpansSec[index].endSec,
        )
    )
        return;
    take.syllableSpansSec = result.sourceSpansSec;
    if ((clip.activeTakeId ?? clip.takes?.[0]?.id) === take.id)
        clip.syllableSpansSec = result.sourceSpansSec;
    // A timeline request issued before this receipt may still contain
    // the old Take cache. Reject it without resetting any curve cache.
    state._pluginTimelineEpoch += 1;
}
