import { describe, expect, it } from "vitest";
import { hitTestParamBlockTransitionHandles } from "../../paramBlocks";

import { createTimelineAxis } from "../../../renderKernel/timelineAxis";
import { buildParamBlockInstances, buildParamBlockScene } from "./blockInstances";

const rgba = [0.2, 0.2, 0.2, 1] as const;
const selectedRgba = [0.1, 0.5, 1, 1] as const;
const groupedRgba = [1, 0.5, 0.2, 1] as const;

describe("buildParamBlockInstances", () => {
    const axis = createTimelineAxis({ pxPerSec: 100, scrollLeftPx: 0, viewportWidthPx: 800 });

    it("每个块产出四条边框，且不画填充", () => {
        const items = buildParamBlockInstances({
            axis,
            viewportHeightPx: 400,
            blocks: [
                {
                    id: "a",
                    startSec: 1,
                    endSec: 3,
                    topPx: 100,
                    heightPx: 40,
                    selected: false,
                    grouped: false,
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            groupedRgba,
        });
        expect(items).toHaveLength(4);
        expect(items.every((item) => item.rgba === rgba)).toBe(true);
        expect(items.map((item) => [item.x, item.y, item.w, item.h])).toEqual([
            [100, 100, 1, 40],
            [299, 100, 1, 40],
            [100, 100, 200, 1],
            [100, 139, 200, 1],
        ]);
    });

    it("选中块只换描边色，不改变几何", () => {
        const items = buildParamBlockInstances({
            axis,
            viewportHeightPx: 400,
            blocks: [
                {
                    id: "a",
                    startSec: 1,
                    endSec: 3,
                    topPx: 100,
                    heightPx: 40,
                    selected: true,
                    grouped: false,
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            groupedRgba,
        });
        expect(items.every((item) => item.rgba === selectedRgba)).toBe(true);
    });

    it("亚像素短块仍保留至少 1 CSS px 的可见填充", () => {
        const scene = buildParamBlockScene({
            axis,
            viewportHeightPx: 400,
            blocks: [
                {
                    id: "tiny",
                    startSec: 1,
                    endSec: 1.001,
                    topPx: 100,
                    heightPx: 40,
                    selected: false,
                    grouped: false,
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            groupedRgba,
            fillRgba: rgba,
        });
        expect(scene.fills).toHaveLength(1);
        expect(scene.fills[0]?.w).toBeGreaterThanOrEqual(1);
        expect(scene.items.length).toBeGreaterThan(0);
    });

    it("每个块都保留左右边界，即使它与相邻块时间相接", () => {
        const items = buildParamBlockInstances({
            axis,
            viewportHeightPx: 400,
            blocks: [
                {
                    id: "a",
                    startSec: 1,
                    endSec: 2,
                    topPx: 100,
                    heightPx: 40,
                    selected: false,
                    grouped: false,
                    contiguousEnd: true,
                },
                {
                    id: "b",
                    startSec: 2,
                    endSec: 3,
                    topPx: 100,
                    heightPx: 40,
                    selected: false,
                    grouped: false,
                    contiguousStart: true,
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            groupedRgba,
        });
        // 2 块 × 4 边；左右边界没有因连续关系而省略。
        expect(items).toHaveLength(8);
        expect(items.filter((item) => item.w === 1 && item.h === 40)).toHaveLength(4);
    });

    it("合并块额外有一条归属线", () => {
        const items = buildParamBlockInstances({
            axis,
            viewportHeightPx: 400,
            blocks: [
                {
                    id: "m",
                    startSec: 1,
                    endSec: 5,
                    topPx: 80,
                    heightPx: 60,
                    selected: false,
                    grouped: true,
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            groupedRgba,
        });
        expect(items).toHaveLength(5);
        expect(items[4].rgba).toBe(groupedRgba);
    });

    it("视口外块与零宽块不产生几何", () => {
        const items = buildParamBlockInstances({
            axis,
            viewportHeightPx: 400,
            blocks: [
                {
                    id: "off",
                    startSec: 20,
                    endSec: 25,
                    topPx: 0,
                    heightPx: 40,
                    selected: false,
                    grouped: false,
                },
                {
                    id: "zero",
                    startSec: 3,
                    endSec: 3,
                    topPx: 0,
                    heightPx: 40,
                    selected: false,
                    grouped: false,
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            groupedRgba,
        });
        expect(items).toEqual([]);
    });

    it("高度与位置收敛在视口内，不产生 NaN", () => {
        const items = buildParamBlockInstances({
            axis,
            viewportHeightPx: 400,
            blocks: [
                {
                    id: "h",
                    startSec: 1,
                    endSec: 2,
                    topPx: 390,
                    heightPx: 80,
                    selected: false,
                    grouped: false,
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            groupedRgba,
        });
        expect(items.every((item) => Number.isFinite(item.y) && Number.isFinite(item.h))).toBe(
            true,
        );
        expect(items.every((item) => item.y + item.h <= 400 + 1e-6)).toBe(true);
    });

    it("块足够宽高时生成标签；窄块省略标签", () => {
        const scene = buildParamBlockScene({
            axis,
            viewportHeightPx: 400,
            blocks: [
                {
                    id: "wide",
                    startSec: 1,
                    endSec: 3,
                    topPx: 100,
                    heightPx: 40,
                    selected: false,
                    grouped: false,
                    label: "64.2",
                },
                {
                    id: "narrow",
                    startSec: 4,
                    endSec: 4.03,
                    topPx: 100,
                    heightPx: 40,
                    selected: false,
                    grouped: false,
                    label: "72.0",
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            groupedRgba,
            labelRgba: rgba,
        });
        expect(scene.labels).toHaveLength(1);
        expect(scene.labels[0]).toMatchObject({ text: "64.2", align: "left", baseline: "middle" });
    });

    it("选中块标签用选中色；没有标签色时不产生文字", () => {
        const block = {
            id: "a",
            startSec: 1,
            endSec: 3,
            topPx: 100,
            heightPx: 40,
            selected: true,
            grouped: false,
            label: "60.0",
        };
        expect(
            buildParamBlockScene({
                axis,
                viewportHeightPx: 400,
                blocks: [block],
                borderRgba: rgba,
                selectedRgba,
                groupedRgba,
            }).labels,
        ).toHaveLength(0);
        expect(
            buildParamBlockScene({
                axis,
                viewportHeightPx: 400,
                blocks: [block],
                borderRgba: rgba,
                selectedRgba,
                groupedRgba,
                labelRgba: rgba,
            }).labels[0].rgba,
        ).toBe(rgba);
        expect(
            buildParamBlockScene({
                axis,
                viewportHeightPx: 400,
                blocks: [block],
                borderRgba: rgba,
                selectedRgba,
                groupedRgba,
                labelRgba: rgba,
                selectedLabelRgba: selectedRgba,
            }).labels[0].rgba,
        ).toBe(selectedRgba);
    });

    it("实心块：填充、顶部高光、边框按顺序分层", () => {
        const scene = buildParamBlockScene({
            axis,
            viewportHeightPx: 400,
            blocks: [
                {
                    id: "a",
                    startSec: 1,
                    endSec: 3,
                    topPx: 100,
                    heightPx: 40,
                    selected: false,
                    grouped: false,
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            groupedRgba,
            fillRgba: [0.5, 0.6, 0.7, 1],
            topHighlightRgba: [1, 1, 1, 0.2],
            topHighlightPx: 2,
            cornerRadiusPx: 2,
        });
        expect(scene.fills).toHaveLength(1);
        expect(scene.fills[0].radius).toBe(2);
        expect(scene.highlights).toHaveLength(1);
        expect(scene.items).toHaveLength(4);
    });

    it("选中块的过渡手柄位于右边界外侧上方，由圆点 + 横线组成", () => {
        const scene = buildParamBlockScene({
            axis,
            viewportHeightPx: 400,
            blocks: [
                {
                    id: "a",
                    startSec: 1,
                    endSec: 2,
                    topPx: 100,
                    heightPx: 24,
                    selected: true,
                    grouped: false,
                    transitionHandle: { sec: 2, leftId: "a", rightId: "b", widthSec: 0.1 },
                },
                {
                    id: "b",
                    startSec: 2,
                    endSec: 3,
                    topPx: 130,
                    heightPx: 24,
                    selected: false,
                    grouped: false,
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            groupedRgba,
            transitionRgba: selectedRgba,
        });
        // 横线 + 柔光 + 圆点 + 相接点圆点：共 4 个实例。
        expect(scene.handles).toHaveLength(4);
        // 圆点中心与右边界对齐，只向上抬到块外：右边界 x=200、块顶 y=100。
        expect(scene.handles[0]).toMatchObject({ x: 192, y: 93.5, w: 8, h: 1 });
        expect(scene.handles[1]).toMatchObject({ x: 194.5, y: 88.5, w: 11, h: 11 });
        expect(scene.handles[2]).toMatchObject({ x: 196.5, y: 90.5, w: 7, h: 7 });
        expect(scene.handles[2].rgba).toBe(selectedRgba);
        expect(scene.transitions).toHaveLength(1);
        // 过渡以相接点为中心向两侧展开：0.1s × 100px/s = 每侧 10px。
        expect(scene.transitions[0]?.[0]).toMatchObject({ x: 190, y: 112 });
        expect(scene.transitions[0]?.at(-1)).toMatchObject({ x: 210, y: 142 });
    });

    it("未选中的相接块不画手柄，避免整层出现铆钉", () => {
        const scene = buildParamBlockScene({
            axis,
            viewportHeightPx: 400,
            blocks: [
                {
                    id: "a",
                    startSec: 1,
                    endSec: 2,
                    topPx: 100,
                    heightPx: 24,
                    selected: false,
                    grouped: false,
                    transitionHandle: { sec: 2, leftId: "a", rightId: "b", widthSec: 0.1 },
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            groupedRgba,
        });
        expect(scene.handles).toEqual([]);
        expect(scene.transitions).toEqual([]);
    });
});

it("at a small zoom the grip stays above an ascending transition and hit testing follows it", () => {
    const axis = createTimelineAxis({ pxPerSec: 10, scrollLeftPx: 0, viewportWidthPx: 800 });
    const blocks = [
        {
            id: "left",
            startSec: 0,
            endSec: 1,
            topPx: 100,
            heightPx: 24,
            selected: true,
            grouped: false,
            transitionHandle: { sec: 1, leftId: "left", rightId: "right", widthSec: 0 },
        },
        {
            id: "right",
            startSec: 1,
            endSec: 2,
            topPx: 60,
            heightPx: 24,
            selected: false,
            grouped: false,
        },
    ];
    const scene = buildParamBlockScene({
        axis,
        viewportHeightPx: 400,
        blocks,
        borderRgba: rgba,
        selectedRgba,
        groupedRgba,
        transitionRgba: selectedRgba,
    });
    const grip = scene.handles[2];
    expect(grip.y + grip.h).toBeLessThan(60);
    expect(scene.transitions[0].every((point) => point.y > grip.y + grip.h)).toBe(true);
    expect(scene.transitions[0][0].x).toBe(5);
    expect(scene.transitions[0].at(-1)?.x).toBe(15);
    expect(
        hitTestParamBlockTransitionHandles({
            blocks,
            clientX: 10,
            localY: 54,
            secToX: (sec) => sec * 10,
        })?.leftId,
    ).toBe("left");
    expect(
        hitTestParamBlockTransitionHandles({
            blocks,
            clientX: 10,
            localY: 94,
            secToX: (sec) => sec * 10,
        }),
    ).toBeNull();
});
