import { describe, expect, it } from "vitest";
import { createDefaultTimelineSnapSettings } from "../../../features/session/sessionSlice";
import {
    clearParamBlockSnapHighlight,
    publishParamBlockSnapHighlight,
    resolveParamBlockPitchSnapDelta,
    resolveParamBlockTimeSnap,
} from "./paramBlockSnap";
import { getSnapHighlightSnapshot } from "../../../utils/snapHighlight";
import { resolveParamBlockSplitSec } from "./paramBlocks";

describe("paramBlockSnap", () => {
    it("pitch snap follows settings and XOR modifier", () => {
        const base = {
            startValue: 60.4,
            rawDelta: -0.2,
            pitchSnapEnabled: true,
            pitchSnapUnit: "semitone" as const,
            pitchSnapToleranceCents: 0,
            projectScale: null,
            anchorSec: 0,
        };
        expect(
            resolveParamBlockPitchSnapDelta({ ...base, toggleModifierActive: false }),
        ).toBeCloseTo(-0.4);
        expect(
            resolveParamBlockPitchSnapDelta({ ...base, toggleModifierActive: true }),
        ).toBeCloseTo(-0.2);
        expect(
            resolveParamBlockPitchSnapDelta({
                ...base,
                pitchSnapEnabled: false,
                toggleModifierActive: true,
            }),
        ).toBeCloseTo(-0.4);
    });

    it("timeline snap follows enabled and XOR modifier", () => {
        const settings = createDefaultTimelineSnapSettings();
        const input = {
            settings,
            grid: "1/4" as const,
            bpm: 120,
            beatsPerBar: 4,
            tempoMap: null,
            pxPerSec: 100,
            clips: [],
            tracks: [],
            selectedClipIds: [],
            playheadSec: 0,
            toggleModifierActive: false,
        };
        expect(resolveParamBlockTimeSnap({ ...input, rawSec: 0.51 })).toBeCloseTo(0.5);
        expect(
            resolveParamBlockTimeSnap({ ...input, rawSec: 0.51, toggleModifierActive: true }),
        ).toBeCloseTo(0.51);
        expect(
            resolveParamBlockTimeSnap({
                ...input,
                rawSec: 0.51,
                settings: { ...settings, enabled: false },
                toggleModifierActive: true,
            }),
        ).toBeCloseTo(0.5);
    });

    it("split snap only applies within the pixel threshold", () => {
        expect(
            resolveParamBlockSplitSec({
                sec: 0.51,
                snapSec: () => 0.5,
                enabled: true,
                pxPerSec: 100,
                thresholdPx: 5,
            }),
        ).toBeCloseTo(0.5);
        expect(
            resolveParamBlockSplitSec({
                sec: 0.8,
                snapSec: () => 0.5,
                enabled: true,
                pxPerSec: 100,
                thresholdPx: 5,
            }),
        ).toBeCloseTo(0.8);
        expect(
            resolveParamBlockSplitSec({
                sec: 0.51,
                snapSec: () => 0.5,
                enabled: false,
                pxPerSec: 100,
                thresholdPx: 5,
            }),
        ).toBeCloseTo(0.51);
    });
});

it("publishes only the snap target, never a second line at the raw pointer", () => {
    clearParamBlockSnapHighlight();
    publishParamBlockSnapHighlight({
        sec: 1,
        snapped: true,
        distancePx: 12,
        candidate: { sec: 1, kind: "grid", priority: 0 },
    });
    const entries = getSnapHighlightSnapshot().entries.filter((entry) =>
        entry.id.startsWith("paramBlock\u0000"),
    );
    expect(entries).toHaveLength(1);
    expect(entries[0]?.markers).toHaveLength(1);
    expect(entries[0]?.markers[0]).toMatchObject({ role: "target", sec: 1 });
    clearParamBlockSnapHighlight();
});
