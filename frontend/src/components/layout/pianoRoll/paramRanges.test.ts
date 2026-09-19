import { describe, expect, it } from "vitest";

import {
    DYN_DEFAULT_VIEW,
    DYN_FOLLOW_ORIG,
    DYN_VALUE_MAX,
    dynMultiplicativeFactor,
    isDynParam,
    restoreDynSentinels,
    VOLUME_DEFAULT_VIEW,
} from "./paramRanges";

describe("isDynParam", () => {
    it("识别 dyn 与历史别名 dyn_edit", () => {
        expect(isDynParam("dyn")).toBe(true);
        expect(isDynParam("dyn_edit")).toBe(true);
        expect(isDynParam("volume")).toBe(false);
        expect(isDynParam("pitch")).toBe(false);
        expect(isDynParam(null)).toBe(false);
        expect(isDynParam(undefined)).toBe(false);
    });
});

describe("restoreDynSentinels", () => {
    it("把位图标记的未画帧恢复成哨兵（写回时不得物化基线）", () => {
        // 帧 0/2 是用户画的，帧 1/3 未画（载荷里已解析成基线值 0.7）。
        const values = [0.8, 0.7, 0.5, 0.7];
        restoreDynSentinels(values, [false, true, false, true]);
        expect(values[0]).toBe(0.8);
        expect(values[1]).toBe(DYN_FOLLOW_ORIG);
        expect(values[2]).toBe(0.5);
        expect(values[3]).toBe(DYN_FOLLOW_ORIG);
    });

    it("无位图时原样返回（非 dyn / 旧载荷兼容）", () => {
        const values = [0.8, 0.7];
        expect(restoreDynSentinels(values, undefined)).toBe(values);
        expect(restoreDynSentinels(values, [])).toBe(values);
    });

    it("位图长度短于值数组时只处理重叠区间", () => {
        const values = [0.8, 0.7, 0.6];
        restoreDynSentinels(values, [true]);
        expect(values[0]).toBe(DYN_FOLLOW_ORIG);
        expect(values[1]).toBe(0.7);
        expect(values[2]).toBe(0.6);
    });
});

describe("dyn 值域与默认视口", () => {
    it("值域 0..2（+6 dB 余量）；默认视口 0..1.25（0 dB 在 80% 高度）", () => {
        expect(DYN_VALUE_MAX).toBe(2);
        expect(DYN_DEFAULT_VIEW).toEqual({ center: 0.625, span: 1.25 });
        // 0 dB（值 1.0）位于视口上部：下缘到 0 dB 的距离占视口的 80%。
        const low = DYN_DEFAULT_VIEW.center - DYN_DEFAULT_VIEW.span / 2;
        const high = DYN_DEFAULT_VIEW.center + DYN_DEFAULT_VIEW.span / 2;
        expect(low).toBe(0);
        expect((1.0 - low) / DYN_DEFAULT_VIEW.span).toBeCloseTo(0.8, 9);
        expect(high).toBeCloseTo(1.25, 9);
        // volume 保持 0..2 视口（>1 的提升是常态）。
        expect(VOLUME_DEFAULT_VIEW).toEqual({ center: 1.0, span: 2.0 });
    });
});

describe("dynMultiplicativeFactor（乘性拖拽核心）", () => {
    it("0.5 个值单位 = ×2（+6 dB）；负向 = ×0.5", () => {
        expect(dynMultiplicativeFactor(0.5)).toBeCloseTo(2, 9);
        expect(dynMultiplicativeFactor(-0.5)).toBeCloseTo(0.5, 9);
        expect(dynMultiplicativeFactor(0)).toBe(1);
    });

    it("★ 乘性语义：静音（0）× 任何系数仍为 0，0.2 → ×2 = 0.4", () => {
        // 用户核心诉求：拖拽不能把无声拖出响度。
        const factor = dynMultiplicativeFactor(0.5); // ×2
        const curve = [0, 0.2, 0.5, 1.0];
        const dragged = curve.map((v) => v * factor);
        expect(dragged[0]).toBe(0);
        expect(dragged[1]).toBeCloseTo(0.4, 9);
        expect(dragged[2]).toBeCloseTo(1.0, 9);
        expect(dragged[3]).toBeCloseTo(2.0, 9);
    });

    it("非有限输入回退恒等", () => {
        expect(dynMultiplicativeFactor(Number.NaN)).toBe(1);
        expect(dynMultiplicativeFactor(Number.POSITIVE_INFINITY)).toBe(1);
    });
});
