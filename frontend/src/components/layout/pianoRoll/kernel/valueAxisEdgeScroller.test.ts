/**
 * 值轴边缘滚动积分器（./valueAxisEdgeScroller）自检。
 *
 * 【本测试守护什么】用户报告的「上下滚动步进太大、抓不住位置」有三条根因，
 * 全部在这里被钉住：
 * 1. **缩放不变**：任何 span 下满带速度都是 0.10 个视口/秒（旧实现按 scrollTop
 *    代理域换算，缩放后屏幕速度会跟着变）；
 * 2. **单帧上限**：任何帧时长下位移都不超过 0.0025 个视口（旧实现没有上限，
 *    一次卡顿帧能跨过好几个像素对应的参数值）；
 * 3. **微调区**：二次软启动让「距边缘 2px」只有满带速度的 1%（旧实现是 1.6 次
 *    幂，2px 处仍有 2.2px/秒，用户根本停不住）。
 *
 * 另外锁住符号：驱动传进来的 ratio 正 = 指针偏「下」，值域中心必须**下移**
 * （看到更低的值）。这是唯一容易写反的地方。
 */
import { describe, expect, it } from "vitest";

import {
    createValueAxisEdgeScrollAxis,
    resolveValueAxisEdgeStep,
    VALUE_EDGE_SCROLL_BAND_PX,
    VALUE_EDGE_SCROLL_MAX_FRAME_MS,
    VALUE_EDGE_SCROLL_MAX_RATIO,
    VALUE_EDGE_SCROLL_MAX_STEP_VIEWPORTS,
    VALUE_EDGE_SCROLL_RAMP_EXPONENT,
    VALUE_EDGE_SCROLL_VIEWPORTS_PER_SEC,
    valueAxisScrollable,
} from "./valueAxisEdgeScroller";

const FRAME_60HZ = 1000 / 60;

describe("constants", () => {
    it("pins the calibrated feel", () => {
        expect(VALUE_EDGE_SCROLL_BAND_PX).toBe(20);
        expect(VALUE_EDGE_SCROLL_VIEWPORTS_PER_SEC).toBe(0.1);
        expect(VALUE_EDGE_SCROLL_MAX_RATIO).toBe(1);
        expect(VALUE_EDGE_SCROLL_RAMP_EXPONENT).toBe(2);
        expect(VALUE_EDGE_SCROLL_MAX_STEP_VIEWPORTS).toBe(0.0025);
        expect(VALUE_EDGE_SCROLL_MAX_FRAME_MS).toBe(32);
    });

    it("keeps the per-frame cap out of the way at 60Hz (frame-rate independence)", () => {
        // 满带 60Hz 单帧 = 0.10 / 60 = 0.001667 视口 < cap 0.0025。
        const perFrameViewports = (VALUE_EDGE_SCROLL_VIEWPORTS_PER_SEC * FRAME_60HZ) / 1000;
        expect(perFrameViewports).toBeLessThan(VALUE_EDGE_SCROLL_MAX_STEP_VIEWPORTS);
    });
});

describe("resolveValueAxisEdgeStep", () => {
    const at = (ratio: number, frameMs: number, span: number): number =>
        resolveValueAxisEdgeStep({ ratio, frameMs, span });

    it("moves 0.04 values per frame at full band (span 24, 60Hz)", () => {
        expect(at(1, FRAME_60HZ, 24)).toBeCloseTo(0.04, 6);
    });

    it("is linear in the ratio", () => {
        expect(at(0.5, FRAME_60HZ, 24)).toBeCloseTo(0.02, 6);
        expect(at(0.1, FRAME_60HZ, 24)).toBeCloseTo(0.004, 6);
    });

    it("keeps the same screen speed at any zoom (proportional to span)", () => {
        expect(at(1, FRAME_60HZ, 6)).toBeCloseTo(0.01, 6);
        expect(at(1, FRAME_60HZ, 24)).toBeCloseTo(0.04, 6);
        expect(at(1, FRAME_60HZ, 120)).toBeCloseTo(0.2, 6);
    });

    it("caps a single frame at 0.0025 viewports", () => {
        expect(at(1, 1000, 24)).toBeCloseTo(VALUE_EDGE_SCROLL_MAX_STEP_VIEWPORTS * 24, 9);
        expect(at(1, 32, 24)).toBeLessThanOrEqual(VALUE_EDGE_SCROLL_MAX_STEP_VIEWPORTS * 24);
    });

    it("carries the sign of the ratio", () => {
        expect(at(-1, FRAME_60HZ, 24)).toBeCloseTo(-0.04, 6);
    });

    it("returns zero for degenerate input instead of producing NaN", () => {
        expect(at(1, 16.67, 0)).toBe(0);
        expect(at(1, Number.NaN, 24)).toBe(0);
        expect(at(1, FRAME_60HZ, Number.NaN)).toBe(0);
        expect(at(Number.NaN, FRAME_60HZ, 24)).toBe(0);
        expect(at(1, 0, 24)).toBe(0);
        expect(at(0, FRAME_60HZ, 24)).toBe(0);
    });
});

describe("valueAxisScrollable", () => {
    it("is true only when the visible span leaves movable center range", () => {
        expect(valueAxisScrollable({ min: 0, max: 127, span: 24 })).toBe(true);
        expect(valueAxisScrollable({ min: 0, max: 127, span: 127 })).toBe(false);
        expect(valueAxisScrollable({ min: 0, max: 127, span: 200 })).toBe(false);
        expect(valueAxisScrollable({ min: 0, max: 0, span: 0 })).toBe(false);
    });

    it("rejects non-finite input instead of producing NaN scroll positions", () => {
        expect(valueAxisScrollable({ min: Number.NaN, max: 1, span: 0 })).toBe(false);
        expect(valueAxisScrollable({ min: 0, max: Number.POSITIVE_INFINITY, span: 1 })).toBe(false);
    });
});

/** 受控值域宿主：中心会被 `clampViewport` 一样夹在可移动范围内。 */
function createValueAxis(args?: { min?: number; max?: number; span?: number }) {
    const min = args?.min ?? 0;
    const max = args?.max ?? 127;
    let span = args?.span ?? 24;
    let center = (min + max) / 2;
    let writable = true;
    const axis = createValueAxisEdgeScrollAxis({
        getSpan: () => span,
        getCenter: () => center,
        setCenter: (next) => {
            if (!writable) return;
            const half = span / 2;
            center = Math.min(max - half, Math.max(min + half, next));
        },
        canScroll: () => valueAxisScrollable({ min, max, span }),
    });
    return {
        axis,
        center: () => center,
        setSpan: (next: number) => {
            span = next;
        },
        setWritable: (next: boolean) => {
            writable = next;
        },
    };
}

describe("createValueAxisEdgeScrollAxis", () => {
    it("is a y axis with the value-axis policy", () => {
        const h = createValueAxis();
        expect(h.axis.id).toBe("y");
        expect(h.axis.policy).toEqual({
            bandPx: VALUE_EDGE_SCROLL_BAND_PX,
            maxRatio: VALUE_EDGE_SCROLL_MAX_RATIO,
            rampExponent: VALUE_EDGE_SCROLL_RAMP_EXPONENT,
        });
        expect(h.axis.maxFrameMs).toBe(VALUE_EDGE_SCROLL_MAX_FRAME_MS);
    });

    it("moves the centre DOWN when the pointer is near the bottom edge", () => {
        const h = createValueAxis();
        const before = h.center();
        expect(h.axis.apply(1, FRAME_60HZ)).toBe(true);
        expect(h.center()).toBeLessThan(before);
    });

    it("moves the centre UP when the pointer is near the top edge", () => {
        const h = createValueAxis();
        const before = h.center();
        expect(h.axis.apply(-1, FRAME_60HZ)).toBe(true);
        expect(h.center()).toBeGreaterThan(before);
    });

    it("moves 0.04 values at full band for span 24", () => {
        const h = createValueAxis();
        const before = h.center();
        h.axis.apply(1, FRAME_60HZ);
        expect(before - h.center()).toBeCloseTo(0.04, 6);
    });

    it("stops at the movable edge and reports no movement there", () => {
        const h = createValueAxis();
        for (let i = 0; i < 5000; i += 1) h.axis.apply(1, FRAME_60HZ);
        expect(h.center()).toBeCloseTo(12, 6);
        expect(h.axis.apply(1, FRAME_60HZ)).toBe(false);
    });

    it("reports no movement when the setter refuses the write (read-back check)", () => {
        const h = createValueAxis();
        h.setWritable(false);
        const before = h.center();
        expect(h.axis.apply(1, FRAME_60HZ)).toBe(false);
        expect(h.center()).toBe(before);
    });

    it("does nothing when the value range cannot move", () => {
        const h = createValueAxis({ span: 127 });
        expect(h.axis.canScroll()).toBe(false);
        expect(h.axis.apply(1, FRAME_60HZ)).toBe(false);
    });

    it("does nothing on a zero-length frame (clock just reset)", () => {
        const h = createValueAxis();
        const before = h.center();
        expect(h.axis.apply(1, 0)).toBe(false);
        expect(h.center()).toBe(before);
    });
});
