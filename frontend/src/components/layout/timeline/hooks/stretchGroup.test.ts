/**
 * 拉伸几何（./stretchGroup 的 computeClipStretch / scaleSnapOffsetForStretch /
 * computeRegionRightEdgeDelta）行为自检。
 *
 * 【主要内容】
 * 1. 左右两个方向的「固定对侧边缘」语义；
 * 2. 速率反算与钳制（含"速率被钳制时长度用钳制后的速率回算"）；
 * 3. 淡变按长度比例缩放、SnapOffset 按总比例缩放并钳到新长度内；
 * 4. 最小长度与非法输入（NaN / 0 / 负值）的退化行为；
 * 5. 波纹跟随的驱动量（区域右缘净位移）带符号、取各成员右缘的最大值。
 *
 * 【作用】这是渲染内核「Alt + 拖边缘 = 拉伸」与旧实现 `useEditDrag` 共用的唯一
 * 几何实现：一旦方向搞反（拉伸变成平移）或漏掉速率回算，画面长度会与音频时长
 * 对不上——这类缺陷在截图上不明显，必须靠断言守住。波纹驱动量同理：符号弄反
 * 会让跟随集在"缩短"时朝反方向跑。
 *
 * 【与其他模块的关系】覆盖 `stretchGroup.ts` 的新增部分；不依赖 React / DOM。
 */

import { describe, expect, it } from "vitest";

import {
    computeClipStretch,
    computeRegionRightEdgeDelta,
    scaleSnapOffsetForStretch,
} from "./stretchGroup";

function base(overrides: Partial<Parameters<typeof computeClipStretch>[0]> = {}) {
    return computeClipStretch({
        edge: "stretch_right",
        pointerSec: 6,
        baseStartSec: 2,
        baseLengthSec: 4,
        basePlaybackRate: 1,
        baseFadeInSec: 0.4,
        baseFadeOutSec: 0.8,
        baseSnapOffsetSec: 1,
        minLengthSec: 0.05,
        ...overrides,
    });
}

describe("computeClipStretch", () => {
    it("右拉伸：左缘固定，长度随指针变化，速率按比例反算", () => {
        const result = base({ pointerSec: 8 });
        expect(result.startSec).toBe(2);
        expect(result.lengthSec).toBeCloseTo(6, 6);
        expect(result.clipPlaybackRate).toBeCloseTo(4 / 6, 6);
        expect(result.scale).toBeCloseTo(1.5, 6);
    });

    it("左拉伸：右缘固定（右缘 = 基准起点 + 基准长度）", () => {
        const result = base({ edge: "stretch_left", pointerSec: 3 });
        // 固定右缘 = 2 + 4 = 6；指针 3 → 期望长度 3 → 速率 4/3 → 长度 3、起点 3。
        expect(result.lengthSec).toBeCloseTo(3, 6);
        expect(result.startSec).toBeCloseTo(3, 6);
        expect(result.clipPlaybackRate).toBeCloseTo(4 / 3, 6);
    });

    it("缩短到极小长度时速率被上限钳制，长度用钳制后的速率回算（不得与速率脱钩）", () => {
        const result = base({ pointerSec: 2.01 });
        // 期望长度 0.01 → 速率 400 被钳到 10 → 长度必须回算为 4/10 = 0.4。
        expect(result.clipPlaybackRate).toBe(10);
        expect(result.lengthSec).toBeCloseTo(0.4, 6);
        expect(result.startSec).toBe(2);
    });

    it("拉长到极大长度时速率被下限钳制并回算长度", () => {
        const result = base({ pointerSec: 400 });
        // 期望长度 398 → 速率 4/398 ≈ 0.0101 被钳到 0.1 → 长度回算为 40。
        expect(result.clipPlaybackRate).toBe(0.1);
        expect(result.lengthSec).toBeCloseTo(40, 6);
    });

    it("淡变按长度比例缩放并钳到新长度内", () => {
        const result = base({ pointerSec: 8 });
        expect(result.fadeInSec).toBeCloseTo(0.6, 6);
        expect(result.fadeOutSec).toBeCloseTo(1.2, 6);
    });

    it("SnapOffset 按总比例缩放并钳到新长度内", () => {
        const scaled = base({ pointerSec: 8 });
        expect(scaled.snapOffsetSec).toBeCloseTo(1.5, 6);
        // 缩到 0.4s（比例 0.1）→ 偏移 1 × 0.1 = 0.1。
        const shrunk = base({ pointerSec: 2.01 });
        expect(shrunk.snapOffsetSec).toBeCloseTo(0.1, 6);
        // 偏移大于基准长度（历史数据 / 外部写入）时按新长度封顶，三角不得越界。
        const clamped = base({ pointerSec: 2.01, baseSnapOffsetSec: 6 });
        expect(clamped.snapOffsetSec).toBeCloseTo(0.4, 6);
    });

    it("基准偏移 <= 0 时恒为 0", () => {
        expect(base({ pointerSec: 8, baseSnapOffsetSec: 0 }).snapOffsetSec).toBe(0);
        expect(base({ pointerSec: 8, baseSnapOffsetSec: -3 }).snapOffsetSec).toBe(0);
    });

    it("指针越过固定边时不产生负长度（钳到最小长度）", () => {
        const left = base({ edge: "stretch_left", pointerSec: 100 });
        expect(left.lengthSec).toBeGreaterThan(0);
        expect(left.startSec).toBeGreaterThanOrEqual(0);
        const right = base({ pointerSec: -50 });
        expect(right.lengthSec).toBeGreaterThan(0);
        expect(right.startSec).toBe(2);
    });

    it("非法基准值退化为安全默认（速率 1、长度下限）", () => {
        const result = base({
            basePlaybackRate: Number.NaN,
            baseLengthSec: 0,
            baseFadeInSec: Number.NaN,
            baseFadeOutSec: Number.NaN,
        });
        expect(result.clipPlaybackRate).toBeGreaterThan(0);
        expect(result.lengthSec).toBeGreaterThan(0);
        expect(result.fadeInSec).toBe(0);
        expect(result.fadeOutSec).toBe(0);
    });

    it("非 1 的基准速率参与反算（拉伸不改变内容时长语义）", () => {
        const result = base({ basePlaybackRate: 2, pointerSec: 6 });
        // 内容时长 = 4 × 2 = 8；新长度 4 → 速率 = 8/4 = 2（长度未变则速率不变）。
        expect(result.clipPlaybackRate).toBeCloseTo(2, 6);
        expect(result.lengthSec).toBeCloseTo(4, 6);
    });
});

describe("scaleSnapOffsetForStretch", () => {
    it("按比例缩放并钳到新长度内", () => {
        expect(scaleSnapOffsetForStretch(2, 1.5, 10)).toBeCloseTo(3, 6);
        expect(scaleSnapOffsetForStretch(2, 1.5, 2)).toBe(2);
    });

    it("非法比例退化为不缩放", () => {
        expect(scaleSnapOffsetForStretch(2, Number.NaN, 10)).toBe(2);
        expect(scaleSnapOffsetForStretch(2, 0, 10)).toBe(2);
    });
});

describe("computeRegionRightEdgeDelta（波纹跟随的驱动量）", () => {
    const baseById = {
        a: { startSec: 2, lengthSec: 4 }, // 旧右缘 6
        b: { startSec: 10, lengthSec: 4 }, // 旧右缘 14
    };

    it("取各成员右缘的最大值之差（区域右缘，不是单个 clip 的右缘）", () => {
        // a 拉长到 8（右缘 10）、b 不动（右缘 14）→ 区域右缘仍是 14 → 位移 0。
        // 若错用"各成员位移之和"或"锚点位移"，这里会得到 +4。
        const delta = computeRegionRightEdgeDelta({
            clipIds: ["a", "b"],
            baseById,
            clips: [
                { id: "a", startSec: 2, lengthSec: 8 },
                { id: "b", startSec: 10, lengthSec: 4 },
            ],
        });
        expect(delta).toBeCloseTo(0, 9);
    });

    it("向右延伸为正、向左收拢为负（符号必须保留）", () => {
        const grow = computeRegionRightEdgeDelta({
            clipIds: ["a"],
            baseById,
            clips: [{ id: "a", startSec: 2, lengthSec: 6 }],
        });
        expect(grow).toBeCloseTo(2, 9);

        const shrink = computeRegionRightEdgeDelta({
            clipIds: ["a"],
            baseById,
            clips: [{ id: "a", startSec: 2, lengthSec: 2 }],
        });
        // 负位移绝不能被吞成 0（否则"向右有波纹、向左没有"）。
        expect(shrink).toBeCloseTo(-2, 9);
    });

    it("只看参与者：无关 clip 的移动不影响结果", () => {
        const delta = computeRegionRightEdgeDelta({
            clipIds: ["a"],
            baseById,
            clips: [
                { id: "a", startSec: 2, lengthSec: 4 },
                { id: "unrelated", startSec: 10, lengthSec: 99 },
            ],
        });
        expect(delta).toBeCloseTo(0, 9);
    });

    it("成员缺失 / 几何非法时退化为 0，不产生 NaN", () => {
        expect(computeRegionRightEdgeDelta({ clipIds: ["missing"], baseById, clips: [] })).toBe(0);
        expect(
            computeRegionRightEdgeDelta({
                clipIds: ["a"],
                baseById,
                clips: [{ id: "a", startSec: Number.NaN, lengthSec: Number.NaN }],
            }),
        ).toBe(0);
    });

    it("空参与者集合返回 0", () => {
        expect(computeRegionRightEdgeDelta({ clipIds: [], baseById, clips: [] })).toBe(0);
    });
});
