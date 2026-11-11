import { describe, expect, it } from "vitest";

import { centerFromVerticalScrollTop, verticalScrollTopFromCenter } from "./verticalScrollMapping";

/**
 * 滚动条映射的自检。
 *
 * 【本测试守护什么】`verticalScrollTopFromCenter` 与 `centerFromVerticalScrollTop`
 * 必须是同一条直线的两个方向 —— 滚动条 thumb 位置与值域中心不允许有两套算术。
 * 边缘滚动的速度 / 带宽 / 单帧上限**不在**本模块（见
 * `kernel/valueAxisEdgeScroller.test.ts`）。
 */
const PITCH = { min: 0, max: 127, span: 24, scrollRangePx: 1600 } as const;

describe("verticalScrollTopFromCenter", () => {
    it("puts the highest centre at scrollTop 0 and the lowest at the range end", () => {
        expect(verticalScrollTopFromCenter({ ...PITCH, center: PITCH.max - PITCH.span / 2 })).toBe(
            0,
        );
        expect(verticalScrollTopFromCenter({ ...PITCH, center: PITCH.min + PITCH.span / 2 })).toBe(
            PITCH.scrollRangePx,
        );
    });

    it("is linear in the centre", () => {
        const mid = (PITCH.min + PITCH.max) / 2;
        expect(verticalScrollTopFromCenter({ ...PITCH, center: mid })).toBeCloseTo(800, 6);
    });

    it("clamps a centre outside the movable range", () => {
        expect(verticalScrollTopFromCenter({ ...PITCH, center: 1e9 })).toBe(0);
        expect(verticalScrollTopFromCenter({ ...PITCH, center: -1e9 })).toBe(PITCH.scrollRangePx);
    });

    it("returns 0 when the value axis cannot move", () => {
        expect(verticalScrollTopFromCenter({ ...PITCH, span: 127, center: 63 })).toBe(0);
        expect(verticalScrollTopFromCenter({ ...PITCH, span: 200, center: 63 })).toBe(0);
        expect(verticalScrollTopFromCenter({ ...PITCH, scrollRangePx: 0, center: 63 })).toBe(0);
    });
});

describe("centerFromVerticalScrollTop", () => {
    it("round-trips with verticalScrollTopFromCenter", () => {
        for (const center of [15, 40, 63.5, 100, 112]) {
            const scrollTop = verticalScrollTopFromCenter({ ...PITCH, center });
            expect(centerFromVerticalScrollTop({ ...PITCH, scrollTop })).toBeCloseTo(center, 6);
        }
    });

    it("clamps scrollTop to the range", () => {
        expect(centerFromVerticalScrollTop({ ...PITCH, scrollTop: -100 })).toBeCloseTo(115, 6);
        expect(centerFromVerticalScrollTop({ ...PITCH, scrollTop: 9999 })).toBeCloseTo(12, 6);
    });

    it("falls back to the middle when the axis cannot move", () => {
        expect(centerFromVerticalScrollTop({ ...PITCH, span: 127, scrollTop: 800 })).toBeCloseTo(
            63.5,
            6,
        );
    });
});
