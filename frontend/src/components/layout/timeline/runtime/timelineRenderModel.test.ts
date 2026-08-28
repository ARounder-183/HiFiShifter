import { test } from "vitest";

import { buildTimelineRenderModel } from "./timelineRenderModel.js";

test("components/layout/timeline/runtime/timelineRenderModel.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    const model = buildTimelineRenderModel({
        tracks: [{ id: "t1" }, { id: "t2" }, { id: "t3" }],
        clips: [
            { id: "c1", trackId: "t1", startSec: 2, lengthSec: 1 },
            { id: "c2", trackId: "t2", startSec: 12, lengthSec: 5 },
        ],
        viewportStartSec: 10,
        viewportEndSec: 20,
        rowHeight: 48,
        scrollTopPx: 48,
        viewportHeightPx: 96,
    });

    assertEqual(model.visibleTrackIds, ["t1", "t2", "t3"], "visible track ids");
    assertEqual(model.visibleClipIdsByTrackId.t2, ["c2"], "visible clip mapping");

    const virtualizedModel = buildTimelineRenderModel({
        tracks: Array.from({ length: 12 }, (_, index) => ({ id: `track-${index}` })),
        clips: [],
        viewportStartSec: 0,
        viewportEndSec: 10,
        rowHeight: 48,
        scrollTopPx: 96,
        viewportHeightPx: 192,
    });

    assertEqual(virtualizedModel.startIndex, 0, "virtual window start");
    assertEqual(virtualizedModel.endIndex, 10, "virtual window end");

});
