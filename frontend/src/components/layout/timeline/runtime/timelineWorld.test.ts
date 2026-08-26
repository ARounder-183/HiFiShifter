import { test } from "vitest";

import {
    createTimelineWorld,
    screenXToWorldSec,
    worldSecToScreenX,
    computeAnchoredScrollLeftPx,
    computeAnchoredScrollTopPx,
    computeWorldDragDelta,
} from "./timelineWorld.js";

test("components/layout/timeline/runtime/timelineWorld.test.ts scripted checks", async () => {
    function assertNear(actual: number, expected: number, label: string): void {
        if (Math.abs(actual - expected) > 1e-6) {
            throw new Error(`${label}: expected ${expected}, received ${actual}`);
        }
    }

    const world = createTimelineWorld({
        pxPerSec: 100,
        rowHeight: 48,
        scrollLeftPx: 250,
        scrollTopPx: 96,
    });

    assertNear(screenXToWorldSec(50, world), 3, "screen to world sec");
    assertNear(worldSecToScreenX(3, world), 50, "world sec to screen");
    assertNear(
        computeAnchoredScrollLeftPx(
            {
                anchorSec: 3,
                anchorScreenX: 50,
                nextPxPerSec: 200,
            },
            world,
        ),
        550,
        "anchored horizontal zoom",
    );
    assertNear(
        computeAnchoredScrollTopPx(
            {
                anchorTrackUnit: 3.5,
                anchorScreenY: 24,
                nextRowHeight: 60,
            },
            world,
        ),
        186,
        "anchored vertical zoom",
    );

    const drag = computeWorldDragDelta(
        {
            startScreenX: 50,
            startScreenY: 24,
            currentScreenX: 130,
            currentScreenY: 120,
        },
        world,
    );

    assertNear(drag.deltaSec, 0.8, "drag delta sec");
    assertNear(drag.deltaTrackUnits, 2, "drag delta tracks");

    console.log("timelineWorld checks passed");
});
