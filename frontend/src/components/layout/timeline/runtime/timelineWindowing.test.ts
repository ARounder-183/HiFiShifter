import { test } from "vitest";

import { computeVisibleTrackWindow, sliceVisibleClipIds } from "./timelineWindowing.js";

test("components/layout/timeline/runtime/timelineWindowing.test.ts scripted checks", async () => {
    function assertDeepEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    const windowed = computeVisibleTrackWindow({
        totalTracks: 80,
        rowHeight: 48,
        scrollTopPx: 96,
        viewportHeightPx: 240,
        overscanRows: 2,
    });

    assertDeepEqual(windowed, { startIndex: 0, endIndex: 9 }, "visible track window");

    const clipIds = sliceVisibleClipIds({
        viewportStartSec: 10,
        viewportEndSec: 20,
        bufferSec: 2,
        clips: [
            { id: "a", startSec: 1, lengthSec: 3 },
            { id: "b", startSec: 8, lengthSec: 5 },
            { id: "c", startSec: 15, lengthSec: 2 },
            { id: "d", startSec: 25, lengthSec: 2 },
        ],
    });

    assertDeepEqual(clipIds, ["b", "c"], "visible clip ids");

});
