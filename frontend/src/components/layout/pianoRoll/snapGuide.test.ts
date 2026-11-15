import { afterEach, describe, expect, it } from "vitest";
import {
    buildCandidateHighlightEntry,
    clearSnapHighlights,
    getSnapGuideEntriesForSurface,
    getSnapHighlightSnapshot,
    publishSnapHighlights,
} from "../../../utils/snapHighlight";

afterEach(() => clearSnapHighlights());

describe("snap guide surface model", () => {
    it("routes piano-roll entries away from the timeline", () => {
        publishSnapHighlights(
            "paramBlock",
            [buildCandidateHighlightEntry({ id: "pitch", kind: "grid", sec: 1 })],
            "pianoRoll",
        );
        expect(getSnapGuideEntriesForSurface("pianoRoll")).toHaveLength(1);
        expect(getSnapGuideEntriesForSurface("timeline")).toHaveLength(0);
        expect(getSnapHighlightSnapshot().entries[0]?.surface).toBe("pianoRoll");
        expect(getSnapHighlightSnapshot().entries[0]?.markers[0]).toMatchObject({
            axis: "time",
            position: 1,
        });
    });

    it("shows all-surface guides on both surfaces", () => {
        publishSnapHighlights(
            "ruler",
            [
                {
                    id: "tempo",
                    kind: "grid",
                    surface: "timeline",
                    markers: [
                        {
                            role: "target",
                            axis: "time",
                            position: 1,
                            trackId: null,
                            clipId: null,
                        },
                    ],
                },
            ],
            "all",
        );
        expect(getSnapGuideEntriesForSurface("timeline")).toHaveLength(1);
        expect(getSnapGuideEntriesForSurface("pianoRoll")).toHaveLength(1);
    });
});
