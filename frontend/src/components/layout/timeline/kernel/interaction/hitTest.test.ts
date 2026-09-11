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

    it("命中左边缘（trim 手柄）", () => {
        // a1 = [1,3) → 内容坐标 100..300；x=102 落在左边缘（默认 6px 内）
        const result = hitTest(makeArgs({ contentX: 102, contentY: 40 }));
        expect(result.kind).toBe("clip");
        if (result.kind !== "clip") return;
        expect(result.region).toBe("left-edge");
    });

    it("命中右边缘（trim 手柄）", () => {
        const result = hitTest(makeArgs({ contentX: 298, contentY: 40 }));
        expect(result.kind).toBe("clip");
        if (result.kind !== "clip") return;
        expect(result.region).toBe("right-edge");
    });

    it("边缘优先于 header 分区（顶部的 trim 手柄不能被 header 抢走）", () => {
        const result = hitTest(makeArgs({ contentX: 102, contentY: 5 }));
        expect(result.kind).toBe("clip");
        if (result.kind !== "clip") return;
        expect(result.region).toBe("left-edge");
    });

    it("极短 clip 的边缘宽度收敛到 1/3，body 仍可命中", () => {
        // 0.06s × 100px/s = 6px 宽 → 边缘收敛到 2px
        const args = makeArgs({
            contentY: 40,
            clipsByTrack: new Map([
                ["A", [{ id: "tiny", trackId: "A", startSec: 0, lengthSec: 0.06 }]],
                ["B", []],
            ]),
        });
        const result = hitTest({ ...args, contentX: 3 });
        expect(result.kind).toBe("clip");
        if (result.kind !== "clip") return;
        expect(result.region).toBe("body");
    });

    it("命中淡变角（body 顶部、贴左边缘）", () => {
        // a1 = [1,3) → 内容 100..300；行高 96、header 18 → body 高 76，
        // 保留区 = max(14, round(76/3)) = 25。y=23 → body 顶部 5px（角部带内）。
        const result = hitTest(makeArgs({ contentX: 105, contentY: 23 }));
        expect(result.kind).toBe("clip");
        if (result.kind !== "clip") return;
        expect(result.region).toBe("fade-in-corner");
    });

    it("命中淡变角（body 顶部、贴右边缘）", () => {
        const result = hitTest(makeArgs({ contentX: 295, contentY: 23 }));
        expect(result.kind).toBe("clip");
        if (result.kind !== "clip") return;
        expect(result.region).toBe("fade-out-corner");
    });

    it("body 深处贴左边缘 → trim 而非淡变角（按竖直方向切分）", () => {
        // 注意 makeArgs 的 rowHeight = 80（不是 96）：y 必须留在第 0 轨内。
        // body 高 = 80 − 2 − 18 = 60 → 保留区 = max(14, 20) = 20；y=60 → body 内 42px。
        const result = hitTest(makeArgs({ contentX: 105, contentY: 60 }));
        expect(result.kind).toBe("clip");
        if (result.kind !== "clip") return;
        expect(result.region).toBe("left-edge");
    });

    it("header 内且不贴边缘 → header（角部只在 body 顶部）", () => {
        // x=120 距左边缘 20px：不触发 trim 边缘（6px），且 y 在 header 内（角部带之外）。
        const result = hitTest(makeArgs({ contentX: 120, contentY: 5 }));
        expect(result.kind).toBe("clip");
        if (result.kind !== "clip") return;
        expect(result.region).toBe("header");
    });

    it("header 内贴边缘 → 仍判边缘（边缘优先于 header）", () => {
        const result = hitTest(makeArgs({ contentX: 105, contentY: 5 }));
        expect(result.kind).toBe("clip");
        if (result.kind !== "clip") return;
        expect(result.region).toBe("left-edge");
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

describe("hitTest · clip 内局部坐标", () => {
    it("返回以 clip 左上角为原点的局部坐标（供 header 控件级命中复用）", () => {
        // a1 起点 1s、pxPerSec 100 → clipLeft = 100px；contentX = 200 → localX = 100
        const result = hitTest(makeArgs({ contentX: 200, contentY: 40 }));
        expect(result.kind).toBe("clip");
        if (result.kind !== "clip") return;
        expect(result.localX).toBeCloseTo(100, 5);
        expect(result.localY).toBeCloseTo(40, 5);
    });

    it("跨行命中时局部 y 以该行顶边为原点", () => {
        // 行 1（B 轨）：contentY = 95 → localY = 95 − 80 = 15
        const result = hitTest(
            makeArgs({
                contentX: 200,
                contentY: 95,
                clipsByTrack: new Map([
                    ["A", []],
                    ["B", [{ id: "b1", trackId: "B", startSec: 1, lengthSec: 2 }]],
                ]),
            }),
        );
        expect(result.kind).toBe("clip");
        if (result.kind !== "clip") return;
        expect(result.localX).toBeCloseTo(100, 5);
        expect(result.localY).toBeCloseTo(15, 5);
    });
});
