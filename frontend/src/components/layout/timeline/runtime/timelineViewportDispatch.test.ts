import { test } from "vitest";

import { shouldDispatchTimelineViewport } from "./timelineViewportDispatch.js";

test("components/layout/timeline/runtime/timelineViewportDispatch.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    assertEqual(
        shouldDispatchTimelineViewport({
            previous: {
                scrollLeft: 0,
                pxPerSec: 12,
                viewportWidth: 1400,
            },
            next: {
                scrollLeft: 0,
                pxPerSec: 8,
                viewportWidth: 1400,
            },
        }),
        true,
        "zoom changes must dispatch even when scrollLeft is unchanged",
    );

    assertEqual(
        shouldDispatchTimelineViewport({
            previous: {
                scrollLeft: 120,
                pxPerSec: 24,
                viewportWidth: 1400,
            },
            next: {
                scrollLeft: 120,
                pxPerSec: 24,
                viewportWidth: 1400,
            },
        }),
        false,
        "identical viewport snapshots do not dispatch",
    );

    assertEqual(
        shouldDispatchTimelineViewport({
            previous: null,
            next: {
                scrollLeft: 0,
                pxPerSec: 24,
                viewportWidth: 1400,
            },
        }),
        true,
        "first viewport snapshot always dispatches",
    );

    console.log("timelineViewportDispatch checks passed");
});
