import { test } from "vitest";

import { buildTimelinePerfScenario } from "./timelinePerfScenario.js";

test("components/layout/timeline/runtime/timelinePerfScenario.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    const scenario = buildTimelinePerfScenario({
        trackCount: 80,
        clipsPerTrack: 62,
    });

    assertEqual(scenario.tracks.length, 80, "track count");
    assertEqual(scenario.clips.length, 4960, "clip count");
    assertEqual(scenario.clips[0]?.trackId, "track-0", "first clip track");

    console.log("timelinePerfScenario checks passed");
});
