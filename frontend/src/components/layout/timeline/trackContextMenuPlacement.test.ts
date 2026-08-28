import { test } from "vitest";

import { getInsertBelowTargetIndex } from "./trackContextMenuPlacement.js";

test("components/layout/timeline/trackContextMenuPlacement.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    assertEqual(
        getInsertBelowTargetIndex(
            [{ id: "track-1" }, { id: "track-2" }, { id: "track-3" }],
            "track-2",
        ),
        2,
        "insert below selected track",
    );

    assertEqual(
        getInsertBelowTargetIndex([{ id: "track-1" }], "missing-track"),
        1,
        "fallback appends when anchor missing",
    );
});
