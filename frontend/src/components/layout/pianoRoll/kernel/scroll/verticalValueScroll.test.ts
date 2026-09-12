/**
 * 参数编辑器内核 · 竖向「值域 ↔ 像素滚动」适配单测。
 *
 * 【特殊说明】这些断言是**手感契约**：内核接管滚动后，滚轮步进、拖拽比例与旧实现
 * 必须完全一致。因此这里不验证「公式好看」，而是钉住三件事——范围常量、方向、
 * 以及往返可逆性。方向写反或常量漂移都会让用户立刻感到不对劲。
 */
import { describe, expect, it } from "vitest";

import {
    PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX,
    centerFromKernelScrollTop,
    kernelScrollTopFromCenter,
} from "./verticalValueScroll";

describe("verticalValueScroll（值域 ↔ 内核像素滚动）", () => {
    const bounds = { min: 0, max: 100 };

    it("滚动范围常量与旧实现一致（1600px）", () => {
        expect(PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX).toBe(1600);
    });

    it("中心值居中时滚动位置位于范围中点", () => {
        const top = kernelScrollTopFromCenter({ ...bounds, span: 50, center: 50 });
        expect(top).toBeCloseTo(800, 6);
    });

    it("中心越靠上 → 滚动位置越小（与旧 verticalScrollTopFromCenter 同向）", () => {
        const high = kernelScrollTopFromCenter({ ...bounds, span: 20, center: 80 });
        const low = kernelScrollTopFromCenter({ ...bounds, span: 20, center: 20 });
        expect(high).toBeLessThan(low);
    });

    // 【修正记录】原计划此用例含 center=88，但 span=30 / [0,100] 的可动中心域是
    // [15,85]，88 属域外会被钳制，往返必然有损。已实测旧 verticalScrollTopFromCenter
    // 同样钳制（88 与 85 得到相同 scrollTop），故实现无误、原断言的前提有误。
    // 现把用例拆成「域内无损」与「域外钳制」两条，行为被显式钉住而非被掩盖。
    it("可动中心域内往返转换是无损的（可逆）", () => {
        for (const center of [15, 37.5, 50, 85]) {
            const top = kernelScrollTopFromCenter({ ...bounds, span: 30, center });
            const back = centerFromKernelScrollTop({ ...bounds, span: 30, scrollTop: top });
            expect(back).toBeCloseTo(center, 6);
        }
    });

    it("域外 center 被钳制到可动中心域（与旧实现一致，不静默漂移）", () => {
        // span=30 → minCenter=15、maxCenter=85；88 应为 85，而非被"修正"成原值。
        const top = kernelScrollTopFromCenter({ ...bounds, span: 30, center: 88 });
        expect(top).toBeCloseTo(kernelScrollTopFromCenter({ ...bounds, span: 30, center: 85 }), 6);
        expect(centerFromKernelScrollTop({ ...bounds, span: 30, scrollTop: top })).toBeCloseTo(
            85,
            6,
        );
    });

    it("span 覆盖整个范围时不可动（映射退化为中点）", () => {
        const top = kernelScrollTopFromCenter({ ...bounds, span: 100, center: 50 });
        expect(top).toBe(0);
        expect(centerFromKernelScrollTop({ ...bounds, span: 100, scrollTop: 999 })).toBeCloseTo(
            50,
            6,
        );
    });

    it("越界输入被钳制，不产生 NaN", () => {
        const top = kernelScrollTopFromCenter({ ...bounds, span: 20, center: 9999 });
        expect(Number.isFinite(top)).toBe(true);
        expect(top).toBeGreaterThanOrEqual(0);
        expect(top).toBeLessThanOrEqual(PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX);
    });
});
