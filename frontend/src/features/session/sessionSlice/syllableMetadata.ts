import type { PayloadAction } from "@reduxjs/toolkit";
import type { SessionState } from "./types";
import { reportParamBlockDiagnostic } from "../../../components/layout/pianoRoll/paramBlockDiagnostics";

/** GAME changes block metadata, never the parameter/audio revision. */
export function applyClipSyllableAnalysisReducer(
    state: SessionState,
    action: PayloadAction<{
        clipId: string;
        takeId: string;
        sourcePath?: string;
        analyzed: boolean;
        sourceSpansSec: Array<{ startSec: number; endSec: number }>;
        /** Parallel source notes; one entry per span. */
        sourceNotes: Array<number | null>;
        /** Source-domain window covered by this analysis. */
        analyzedWindowSec: [number, number];
    }>,
) {
    const result = action.payload;
    const reject = (code: string) => {
        reportParamBlockDiagnostic({
            code,
            gesture: "layout",
            rootTrackId: null,
            blockId: null,
            frame: null,
            detail: JSON.stringify({ clipId: result?.clipId, takeId: result?.takeId }),
        });
    };
    if (
        !result ||
        !result.analyzed ||
        !Array.isArray(result.sourceSpansSec) ||
        result.sourceSpansSec.length === 0 ||
        result.sourceSpansSec.length > 20000 ||
        !Array.isArray(result.sourceNotes) ||
        result.sourceNotes.length !== result.sourceSpansSec.length ||
        !Array.isArray(result.analyzedWindowSec) ||
        result.analyzedWindowSec.length !== 2 ||
        !result.analyzedWindowSec.every((value) => Number.isFinite(value)) ||
        result.analyzedWindowSec[1] < result.analyzedWindowSec[0]
    ) {
        reject("analysis_payload_invalid");
        return;
    }
    if (
        result.sourceSpansSec.some(
            (span, index, spans) =>
                !Number.isFinite(span.startSec) ||
                !Number.isFinite(span.endSec) ||
                span.endSec <= span.startSec ||
                span.startSec < result.analyzedWindowSec[0] - 1e-6 ||
                span.endSec > result.analyzedWindowSec[1] + 1e-6 ||
                (index > 0 && span.startSec < spans[index - 1].endSec - 1e-9),
        ) ||
        result.sourceNotes.some(
            (note) => note !== null && (typeof note !== "number" || !Number.isFinite(note)),
        )
    ) {
        reject("analysis_window_stale");
        return;
    }
    const clip = state.clips.find((clip) => clip.id === result.clipId);
    const take = clip?.takes?.find((take) => take.id === result.takeId);
    if (
        !clip ||
        !take ||
        (result.sourcePath !== undefined &&
            (take.sourcePath ?? clip.sourcePath) !== result.sourcePath)
    ) {
        reject("analysis_payload_stale");
        return;
    }
    const nextSpans = result.sourceSpansSec.map((span, index) => ({
        startSec: span.startSec,
        endSec: span.endSec,
        note: result.sourceNotes[index] ?? null,
    }));
    const window: [number, number] = [result.analyzedWindowSec[0], result.analyzedWindowSec[1]];
    const previous = take.syllableSpansSec;
    if (
        previous?.length === nextSpans.length &&
        previous.every(
            (span, index) =>
                span.startSec === nextSpans[index].startSec &&
                span.endSec === nextSpans[index].endSec &&
                (span.note ?? null) === (nextSpans[index].note ?? null),
        ) &&
        take.syllableAnalysisWindowSec?.[0] === window[0] &&
        take.syllableAnalysisWindowSec?.[1] === window[1]
    )
        return;
    take.syllableSpansSec = nextSpans;
    take.syllableNotes = [...result.sourceNotes];
    take.syllableAnalysisWindowSec = window;
    if ((clip.activeTakeId ?? clip.takes?.[0]?.id) === take.id) {
        clip.syllableSpansSec = nextSpans;
        clip.syllableNotes = [...result.sourceNotes];
        clip.syllableAnalysisWindowSec = window;
    }
    // A timeline request issued before this receipt may still contain
    // the old Take cache. Reject it without resetting any curve cache.
    state._pluginTimelineEpoch += 1;
}
