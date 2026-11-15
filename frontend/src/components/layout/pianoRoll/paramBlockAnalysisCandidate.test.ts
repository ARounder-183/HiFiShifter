import { describe, expect, it } from "vitest";
import { paramBlockStaleAnalysisCandidates } from "./paramBlockAnalysisCandidate";
import { paramBlockAnalysisRequestKey } from "./paramBlockProjection";
import type { ClipInfo } from "../../../features/session/sessionTypes";

function clip(overrides: Partial<ClipInfo> = {}): ClipInfo {
    return {
        id: "clip",
        trackId: "root",
        name: "Clip",
        startSec: 0,
        lengthSec: 2,
        color: "blue",
        gain: 1,
        muted: false,
        sourceStartSec: 0,
        sourceEndSec: 2,
        sourcePath: "a.wav",
        playbackRate: 1,
        reversed: false,
        channelMode: 0,
        loopEnabled: false,
        snapOffsetSec: 0,
        fadeInSec: 0,
        fadeOutSec: 0,
        fadeInShape: 1,
        fadeOutShape: 1,
        fadeInDir: 0,
        fadeOutDir: 0,
        syllableSpansSec: [{ startSec: 0.046, endSec: 0.252, eventId: "event-1" }],
        syllableNotes: [64],
        syllableEventIds: ["event-1"],
        syllableAnalysisWindowSec: [0, 2],
        ...overrides,
    } as ClipInfo;
}

describe("paramBlockStaleAnalysisCandidates", () => {
    it("does not queue a current window", () => {
        const result = paramBlockStaleAnalysisCandidates({
            clips: [clip()],
            rootTrackId: "root",
            framePeriodMs: 5,
            attemptedKeys: new Set(),
        });
        expect(result.clips).toHaveLength(0);
    });

    it("queues a stale window once per request key and keeps old spans in the view", () => {
        const stale = clip({ syllableAnalysisWindowSec: [0.2, 1] });
        const first = paramBlockStaleAnalysisCandidates({
            clips: [stale],
            rootTrackId: "root",
            framePeriodMs: 5,
            attemptedKeys: new Set(),
        });
        expect(first.clips).toHaveLength(1);
        const key = first.keysByClipId.get("clip");
        expect(key).toBeDefined();
        const second = paramBlockStaleAnalysisCandidates({
            clips: [stale],
            rootTrackId: "root",
            framePeriodMs: 5,
            attemptedKeys: new Set([key as string]),
        });
        expect(second.clips).toHaveLength(0);
    });

    it("changes the request key when the audible window changes", () => {
        const base = {
            rootTrackId: "root",
            clipId: "clip",
            activeTakeId: "take",
            sourceDigest: "a.wav",
            framePeriodMs: 5,
        };
        const before = paramBlockAnalysisRequestKey({
            ...base,
            audible: { startSec: 0, endSec: 2 },
        });
        const after = paramBlockAnalysisRequestKey({
            ...base,
            audible: { startSec: 0.054, endSec: 2 },
        });
        expect(after).not.toBe(before);
    });
});
