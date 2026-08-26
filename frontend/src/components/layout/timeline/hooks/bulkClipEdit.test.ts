import { test } from "vitest";

import {
    applyBulkFadeValue,
    applyBulkGainDeltaDb,
    getBulkEditableClipIds,
} from "./bulkClipEdit.js";

test("components/layout/timeline/hooks/bulkClipEdit.test.ts scripted checks", async () => {
    function assertDeepEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    const selectedIds = getBulkEditableClipIds({
        activeClipId: "b",
        multiSelectedClipIds: ["a", "b", "c"],
        multiSelectedSet: new Set(["a", "b", "c"]),
    });

    assertDeepEqual(selectedIds, ["a", "b", "c"], "bulk-selected ids");

    const singleIds = getBulkEditableClipIds({
        activeClipId: "x",
        multiSelectedClipIds: ["a", "b", "c"],
        multiSelectedSet: new Set(["a", "b", "c"]),
    });

    assertDeepEqual(singleIds, ["x"], "single fallback ids");

    const fadeUpdates = applyBulkFadeValue({
        clipIds: ["a", "b"],
        clipsById: new Map([
            ["a", { lengthSec: 3 }],
            ["b", { lengthSec: 1.25 }],
        ]),
        target: "fadeOutSec",
        nextValue: 2,
    });

    assertDeepEqual(
        fadeUpdates,
        [
            { clipId: "a", fadeOutSec: 2 },
            { clipId: "b", fadeOutSec: 1.25 },
        ],
        "fade updates clamp per clip",
    );

    const fadeInUpdates = applyBulkFadeValue({
        clipIds: ["a", "b"],
        clipsById: new Map([
            ["a", { lengthSec: 0.4 }],
            ["b", { lengthSec: 2.5 }],
        ]),
        target: "fadeInSec",
        nextValue: -1,
    });

    assertDeepEqual(
        fadeInUpdates,
        [
            { clipId: "a", fadeInSec: 0 },
            { clipId: "b", fadeInSec: 0 },
        ],
        "fade values clamp to zero",
    );

    const gainUpdates = applyBulkGainDeltaDb({
        clipIds: ["a", "b"],
        clipsById: new Map([
            ["a", { gain: 1 }],
            ["b", { gain: 0.5 }],
        ]),
        deltaDb: 6,
        minDb: -12,
        maxDb: 12,
    });

    assertDeepEqual(
        gainUpdates.map((entry) => ({
            clipId: entry.clipId,
            gain: Number(entry.gain.toFixed(4)),
        })),
        [
            { clipId: "a", gain: 1.9953 },
            { clipId: "b", gain: 0.9976 },
        ],
        "gain updates preserve per-clip relative delta",
    );

    console.log("bulk clip edit helpers checks passed");
});
