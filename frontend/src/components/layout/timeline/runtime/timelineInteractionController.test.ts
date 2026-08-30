import { test } from "vitest";

import {
    beginClipDrag,
    resolveWheelZoom,
    updateClipDrag,
} from "./timelineInteractionController.js";

test("components/layout/timeline/runtime/timelineInteractionController.test.ts scripted checks", async () => {
    function assertNear(actual: number, expected: number, label: string): void {
        if (Math.abs(actual - expected) > 1e-6) {
            throw new Error(`${label}: expected ${expected}, received ${actual}`);
        }
    }

    const zoom = resolveWheelZoom({
        anchorScreenX: 80,
        anchorSec: 4,
        nextPxPerSec: 240,
    });
    assertNear(zoom.nextScrollLeftPx, 880, "anchored zoom scrollLeft");

    const drag = beginClipDrag({
        clipId: "clip-a",
        trackId: "track-a",
        startWorldSec: 4,
        startTrackIndex: 2,
    });

    const moved = updateClipDrag(drag, {
        currentWorldSec: 5.25,
        currentTrackIndex: 4,
    });

    assertNear(moved.deltaSec, 1.25, "drag delta sec");
    assertNear(moved.deltaTrackIndex, 2, "drag delta track index");
});
