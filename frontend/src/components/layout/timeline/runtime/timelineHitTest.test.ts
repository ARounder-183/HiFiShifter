import { test } from "vitest";

import { buildTimelineHitTestIndex, hitTestTimeline } from "./timelineHitTest.js";

test("components/layout/timeline/runtime/timelineHitTest.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    const index = buildTimelineHitTestIndex({
        rowHeight: 48,
        pxPerSec: 100,
        visibleTracks: [{ id: "track-a", topPx: 0 }],
        visibleClips: [{ id: "clip-a", trackId: "track-a", startSec: 1, lengthSec: 2 }],
    });

    assertEqual(
        hitTestTimeline({ screenX: 102, screenY: 20, scrollLeftPx: 0, scrollTopPx: 0 }, index),
        { trackId: "track-a", clipId: "clip-a", zone: "trim_left" },
        "left trim hit",
    );

    assertEqual(
        hitTestTimeline({ screenX: 240, screenY: 20, scrollLeftPx: 0, scrollTopPx: 0 }, index),
        { trackId: "track-a", clipId: "clip-a", zone: "body" },
        "body hit",
    );

    assertEqual(
        hitTestTimeline({ screenX: 298, screenY: 20, scrollLeftPx: 0, scrollTopPx: 0 }, index),
        { trackId: "track-a", clipId: "clip-a", zone: "trim_right" },
        "right trim hit",
    );

    assertEqual(
        hitTestTimeline({ screenX: 20, screenY: 20, scrollLeftPx: 0, scrollTopPx: 0 }, index),
        { trackId: "track-a", clipId: null, zone: "empty" },
        "empty lane hit",
    );

    // ── SnapOffset 角部区（行底部条带 + Clip 左缘 10px）────────────────
    // rowHeight=48 → 底部条带 localY ≥ 37；pxPerSec=100 → 10px = 0.1s。
    assertEqual(
        hitTestTimeline({ screenX: 105, screenY: 44, scrollLeftPx: 0, scrollTopPx: 0 }, index),
        { trackId: "track-a", clipId: "clip-a", zone: "snap_offset" },
        "bottom-left corner hits snap offset",
    );
    // 角部区优先于 trim_left（同一 x 在条带外才是 trim_left）。
    assertEqual(
        hitTestTimeline({ screenX: 102, screenY: 44, scrollLeftPx: 0, scrollTopPx: 0 }, index),
        { trackId: "track-a", clipId: "clip-a", zone: "snap_offset" },
        "corner takes precedence over trim_left",
    );
    // 底部条带内但越过角宽（>10px）：回落到常规分区。
    assertEqual(
        hitTestTimeline({ screenX: 115, screenY: 44, scrollLeftPx: 0, scrollTopPx: 0 }, index),
        { trackId: "track-a", clipId: "clip-a", zone: "body" },
        "bottom strip beyond corner falls back to body",
    );
    // 条带外同一 x：仍是 trim_left。
    assertEqual(
        hitTestTimeline({ screenX: 105, screenY: 20, scrollLeftPx: 0, scrollTopPx: 0 }, index),
        { trackId: "track-a", clipId: "clip-a", zone: "trim_left" },
        "same x above strip stays trim_left",
    );
    // 空车道底部条带：empty。
    assertEqual(
        hitTestTimeline({ screenX: 20, screenY: 44, scrollLeftPx: 0, scrollTopPx: 0 }, index),
        { trackId: "track-a", clipId: null, zone: "empty" },
        "empty lane bottom strip",
    );

    // ── 命中区跟随 ◣ 三角位置（offset>0 时不再固定在左下角）──────────
    const offsetIndex = buildTimelineHitTestIndex({
        rowHeight: 48,
        pxPerSec: 100,
        visibleTracks: [{ id: "track-a", topPx: 0 }],
        visibleClips: [
            { id: "clip-a", trackId: "track-a", startSec: 1, lengthSec: 2, snapOffsetSec: 0.5 },
        ],
    });
    // 三角在 50px 处：x∈[46..60] 命中。
    assertEqual(
        hitTestTimeline(
            { screenX: 148, screenY: 44, scrollLeftPx: 0, scrollTopPx: 0 },
            offsetIndex,
        ),
        { trackId: "track-a", clipId: "clip-a", zone: "snap_offset" },
        "hit zone follows triangle at offset",
    );
    // 原左下角（三角已离开）：回落常规分区（0.05s 在 trim_left 阈值内）。
    assertEqual(
        hitTestTimeline(
            { screenX: 105, screenY: 44, scrollLeftPx: 0, scrollTopPx: 0 },
            offsetIndex,
        ),
        { trackId: "track-a", clipId: "clip-a", zone: "trim_left" },
        "old corner falls back to trim_left once triangle moved",
    );
    // 三角贴近 Clip 末端时钳制到 width−9：len=2 → triX=min(195,191)=191，
    // 命中区 [187..201]，取 x=190 命中。
    const nearEndClip = { ...offsetIndex.clipsByTrackId.get("track-a")![0], snapOffsetSec: 1.95 };
    const nearEndIndex = {
        ...offsetIndex,
        clipsByTrackId: new Map([["track-a", [nearEndClip]]]),
    };
    assertEqual(
        hitTestTimeline(
            { screenX: 295, screenY: 44, scrollLeftPx: 0, scrollTopPx: 0 },
            nearEndIndex,
        ),
        { trackId: "track-a", clipId: "clip-a", zone: "snap_offset" },
        "triangle clamped before clip end stays hittable",
    );

    const overlappingIndex = buildTimelineHitTestIndex({
        rowHeight: 48,
        pxPerSec: 100,
        visibleTracks: [{ id: "track-a", topPx: 0 }],
        visibleClips: [
            { id: "clip-back", trackId: "track-a", startSec: 1, lengthSec: 4 },
            { id: "clip-front", trackId: "track-a", startSec: 2, lengthSec: 3 },
        ],
    });

    assertEqual(
        hitTestTimeline(
            { screenX: 250, screenY: 16, scrollLeftPx: 0, scrollTopPx: 0 },
            overlappingIndex,
        ),
        { trackId: "track-a", clipId: "clip-front", zone: "body" },
        "top-most overlapping clip wins",
    );

    console.log("timelineHitTest checks passed");
});
