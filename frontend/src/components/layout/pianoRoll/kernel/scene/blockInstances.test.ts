import { describe, expect, it } from "vitest";

import { createTimelineAxis } from "../../../renderKernel/timelineAxis";
import { buildParamBlockInstances, buildParamBlockScene } from "./blockInstances";

const rgba = [0.2, 0.2, 0.2, 1] as const;
const selectedRgba = [0.1, 0.5, 1, 1] as const;
const mergedRgba = [1, 0.5, 0.2, 1] as const;

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
                    merged: false,
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            mergedRgba,
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
                    merged: false,
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            mergedRgba,
        });
        expect(items.every((item) => item.rgba === selectedRgba)).toBe(true);
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
                    merged: false,
                    contiguousEnd: true,
                },
                {
                    id: "b",
                    startSec: 2,
                    endSec: 3,
                    topPx: 100,
                    heightPx: 40,
                    selected: false,
                    merged: false,
                    contiguousStart: true,
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            mergedRgba,
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
                    merged: true,
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            mergedRgba,
        });
        expect(items).toHaveLength(5);
        expect(items[4].rgba).toBe(mergedRgba);
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
                    merged: false,
                },
                {
                    id: "zero",
                    startSec: 3,
                    endSec: 3,
                    topPx: 0,
                    heightPx: 40,
                    selected: false,
                    merged: false,
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            mergedRgba,
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
                    merged: false,
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            mergedRgba,
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
                    merged: false,
                    label: "64.2",
                },
                {
                    id: "narrow",
                    startSec: 4,
                    endSec: 4.03,
                    topPx: 100,
                    heightPx: 40,
                    selected: false,
                    merged: false,
                    label: "72.0",
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            mergedRgba,
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
            merged: false,
            label: "60.0",
        };
        expect(
            buildParamBlockScene({
                axis,
                viewportHeightPx: 400,
                blocks: [block],
                borderRgba: rgba,
                selectedRgba,
                mergedRgba,
            }).labels,
        ).toHaveLength(0);
        expect(
            buildParamBlockScene({
                axis,
                viewportHeightPx: 400,
                blocks: [block],
                borderRgba: rgba,
                selectedRgba,
                mergedRgba,
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
                mergedRgba,
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
                    merged: false,
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            mergedRgba,
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
                    merged: false,
                    transitionHandle: { sec: 2, leftId: "a", rightId: "b", widthSec: 0.1 },
                },
                {
                    id: "b",
                    startSec: 2,
                    endSec: 3,
                    topPx: 130,
                    heightPx: 24,
                    selected: false,
                    merged: false,
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            mergedRgba,
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
                    merged: false,
                    transitionHandle: { sec: 2, leftId: "a", rightId: "b", widthSec: 0.1 },
                },
            ],
            borderRgba: rgba,
            selectedRgba,
            mergedRgba,
        });
        expect(scene.handles).toEqual([]);
        expect(scene.transitions).toEqual([]);
    });
});
