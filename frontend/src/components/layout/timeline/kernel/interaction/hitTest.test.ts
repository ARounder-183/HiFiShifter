import { describe, expect, it } from "vitest";

import { hitTest, type HitTestArgs, type HitTestClip } from "./hitTest";

/** 构造命中参数：2 条轨道，轨道 A 上三个 clip（含首尾相接）。 */
function makeArgs(overrides: Partial<HitTestArgs> = {}): HitTestArgs {
    const clips: HitTestClip[] = [
        { id: "a1", trackId: "A", startSec: 1, lengthSec: 2 },
        { id: "a2", trackId: "A", startSec: 3, lengthSec: 2 },
        { id: "a3", trackId: "A", startSec: 10, lengthSec: 1 },
    ];
    return {
        contentX: 0,
        contentY: 0,
        pxPerSec: 100,
        rowHeight: 80,
        tracks: [{ id: "A" }, { id: "B" }],
        clipsByTrack: new Map([
            ["A", clips],
            ["B", []],
        ]),
        headerHeightPx: 18,
        ...overrides,
    };
}

describe("hitTest", () => {
    it("命中 clip 的 body 分区", () => {
        // 2.0s → contentX = 200；行 0 内 y = 40（> header 18）
        const result = hitTest(makeArgs({ contentX: 200, contentY: 40 }));
        expect(result.kind).toBe("clip");
        if (result.kind !== "clip") return;
        expect(result.clip.id).toBe("a1");
        expect(result.region).toBe("body");
        expect(result.trackIndex).toBe(0);
    });

    it("命中 clip 的 header 分区", () => {
        const result = hitTest(makeArgs({ contentX: 200, contentY: 5 }));
        expect(result.kind).toBe("clip");
        if (result.kind !== "clip") return;
        expect(result.region).toBe("header");
    });

    it("首尾相接处命中右侧 clip（半开区间）", () => {
        // a1 = [1,3)，a2 = [3,5)：3.0s 应命中 a2
        const result = hitTest(makeArgs({ contentX: 300, contentY: 40 }));
        expect(result.kind).toBe("clip");
        if (result.kind !== "clip") return;
        expect(result.clip.id).toBe("a2");
    });

    it("clip 右端之后落在空白", () => {
        // 5.0s 已越过 a2 的 [3,5)
        const result = hitTest(makeArgs({ contentX: 500, contentY: 40 }));
        expect(result.kind).toBe("empty");
        if (result.kind !== "empty") return;
        expect(result.trackId).toBe("A");
    });

    it("空轨道命中 empty 且带回轨道 id", () => {
        const result = hitTest(makeArgs({ contentX: 200, contentY: 120 }));
        expect(result.kind).toBe("empty");
        if (result.kind !== "empty") return;
        expect(result.trackId).toBe("B");
        expect(result.trackIndex).toBe(1);
    });

    it("轨道区之外返回 empty 且 trackId 为 null", () => {
        const result = hitTest(makeArgs({ contentX: 200, contentY: 999 }));
        expect(result.kind).toBe("empty");
        if (result.kind !== "empty") return;
        expect(result.trackId).toBeNull();
        expect(result.trackIndex).toBe(-1);
    });

    it("时间换算按 pxPerSec，且负数钳制到 0", () => {
        const result = hitTest(makeArgs({ contentX: -50, contentY: 40 }));
        expect(result.sec).toBe(0);
    });

    it("缩放变化后命中仍正确", () => {
        // pxPerSec = 50：2.5s → contentX = 125，命中 a1（[1,3)）
        const result = hitTest(makeArgs({ contentX: 125, contentY: 40, pxPerSec: 50 }));
        expect(result.kind).toBe("clip");
        if (result.kind !== "clip") return;
        expect(result.clip.id).toBe("a1");
    });
});
