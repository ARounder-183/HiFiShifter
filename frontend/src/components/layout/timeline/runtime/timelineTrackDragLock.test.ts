import { test } from "vitest";

import {
    computeTimelineTrackDragLock,
    computeTimelineTrackDragLockThresholdPx,
} from "./timelineTrackDragLock.js";

test("components/layout/timeline/runtime/timelineTrackDragLock.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    assertEqual(
        computeTimelineTrackDragLockThresholdPx(40),
        24,
        "threshold clamps to stable lower bound",
    );

    assertEqual(
        computeTimelineTrackDragLockThresholdPx(1000),
        32,
        "threshold clamps to stable upper bound",
    );

    assertEqual(
        computeTimelineTrackDragLock({
            initialTrackId: "track-a",
            hoveredTrackId: "track-a",
            horizontalDeltaPx: 8,
            thresholdPx: 24,
        }),
        {
            locked: false,
            lockedTrackId: null,
        },
        "same-track drags never vertical-lock",
    );

    assertEqual(
        computeTimelineTrackDragLock({
            initialTrackId: "track-a",
            hoveredTrackId: "track-b",
            horizontalDeltaPx: 12,
            thresholdPx: 24,
        }),
        {
            locked: true,
            lockedTrackId: "track-b",
        },
        "small horizontal motion across tracks locks to vertical track move",
    );

    assertEqual(
        computeTimelineTrackDragLock({
            initialTrackId: "track-a",
            hoveredTrackId: "track-b",
            horizontalDeltaPx: 30,
            thresholdPx: 24,
        }),
        {
            locked: false,
            lockedTrackId: null,
        },
        "large horizontal motion exits vertical track lock",
    );
});
