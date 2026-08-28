import { test } from "vitest";

import { shouldRouteClipPasteToParamEditor } from "./clipboardFocusRouting.ts";

test("components/layout/timeline/clipboardFocusRouting.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    assertEqual(
        shouldRouteClipPasteToParamEditor({
            inPianoRoll: true,
            inTrackHeader: false,
        }),
        true,
        "piano roll keeps param paste priority",
    );

    assertEqual(
        shouldRouteClipPasteToParamEditor({
            inPianoRoll: false,
            inTrackHeader: true,
        }),
        false,
        "track header should not swallow clip paste",
    );

});
