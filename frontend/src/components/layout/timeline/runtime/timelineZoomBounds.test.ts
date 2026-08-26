import { test } from "vitest";

import { resolveTimelineMinPxPerSec } from "./timelineZoomBounds.js";

test("components/layout/timeline/runtime/timelineZoomBounds.test.ts scripted checks", async () => {
    function assertNear(actual: number, expected: number, label: string): void {
        if (Math.abs(actual - expected) > 1e-6) {
            throw new Error(`${label}: expected ${expected}, received ${actual}`);
        }
    }

    assertNear(
        resolveTimelineMinPxPerSec({
            baseMinPxPerSec: 4,
            projectSec: 12,
            viewportWidthPx: 1440,
        }),
        0.5,
        "project can shrink below legacy minimum once boundary is fully visible",
    );

    assertNear(
        resolveTimelineMinPxPerSec({
            baseMinPxPerSec: 4,
            projectSec: 400,
            viewportWidthPx: 1440,
        }),
        4,
        "long projects keep the legacy minimum",
    );

    console.log("timelineZoomBounds checks passed");
});
