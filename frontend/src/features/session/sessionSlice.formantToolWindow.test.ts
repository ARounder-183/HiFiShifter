import { test } from "vitest";

import reducer from "./sessionSlice.ts";

test("features/session/sessionSlice.formantToolWindow.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (JSON.stringify(actual) !== JSON.stringify(expected)) {
            throw new Error(
                `${label}: expected ${JSON.stringify(expected)}, received ${JSON.stringify(actual)}`,
            );
        }
    }

    const baseState = reducer(undefined, { type: "@@INIT" });

    const opened = reducer(baseState, {
        type: "session/openClipFormantToolWindow",
        payload: {
            clipId: "clip-a",
            anchor: { x: 180, y: 96 },
        },
    });

    assertEqual(
        opened.clipFormantToolWindow,
        {
            open: true,
            clipId: "clip-a",
            x: 180,
            y: 96,
            hasMoved: false,
        },
        "opens formant tool window at anchor position",
    );

    const moved = reducer(opened, {
        type: "session/setClipFormantToolWindowPosition",
        payload: { x: 420, y: 260 },
    });

    assertEqual(
        moved.clipFormantToolWindow,
        {
            open: true,
            clipId: "clip-a",
            x: 420,
            y: 260,
            hasMoved: true,
        },
        "moving tool window updates shared position",
    );

    const switched = reducer(moved, {
        type: "session/openClipFormantToolWindow",
        payload: {
            clipId: "clip-b",
            anchor: { x: 20, y: 30 },
        },
    });

    assertEqual(switched.clipFormantToolWindow.clipId, "clip-b", "switches the active clip");
    assertEqual(switched.clipFormantToolWindow.x, 420, "keeps shared x after moving");
    assertEqual(switched.clipFormantToolWindow.y, 260, "keeps shared y after moving");

    const closed = reducer(switched, {
        type: "session/closeClipFormantToolWindow",
    });

    assertEqual(closed.clipFormantToolWindow.open, false, "close hides tool window");
    assertEqual(closed.clipFormantToolWindow.x, 420, "close preserves shared x");
    assertEqual(closed.clipFormantToolWindow.y, 260, "close preserves shared y");

    console.log("sessionSlice formant tool window checks passed");
});
