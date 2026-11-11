/**
 * 轴抽象（./edgeScrollAxis）自检。
 *
 * 【本测试守护什么】
 * 1. 比例几何：带外 0、带内线性、上限饱和、软启动指数、退化视口；
 * 2. 水平轴的**像素**语义：步长与帧时长成正比、上下界钳制（含负下界）、
 *    被钳住时返回 false（驱动据此停止排帧）。
 *
 * 【为什么单独测水平轴】它是从旧 `edgeScrollDriver` 里搬出来的唯一一段会写
 * `scrollLeft` 的算术。搬移过程中最容易出的事故是「下界被写死成 0」——
 * 开启同步时间轴后左侧留白是合法负区间，写死 0 会让视图在两处往复（闪现）。
 */
import { describe, expect, it } from "vitest";

import {
    createHorizontalEdgeScrollAxis,
    DEFAULT_EDGE_SCROLL_AXIS_POLICY,
    resolveEdgeScrollRatio,
    type EdgeScrollAxisPolicy,
} from "./edgeScrollAxis";

const POLICY: EdgeScrollAxisPolicy = { bandPx: 32, maxRatio: 1.5, rampExponent: 1 };
const VIEW = { minPx: 100, maxPx: 1100 } as const;
const FRAME_60HZ = 1000 / 60;

const ratioAt = (pointerPx: number, policy: EdgeScrollAxisPolicy = POLICY): number =>
    resolveEdgeScrollRatio({ ...VIEW, pointerPx, policy });

describe("resolveEdgeScrollRatio", () => {
    it("is zero outside the band and exactly at its inner edge", () => {
        expect(ratioAt(600)).toBe(0);
        expect(ratioAt(100 + 32)).toBe(0);
        expect(ratioAt(1100 - 32)).toBe(0);
    });

    it("is linear inside the band, negative near the near edge", () => {
        expect(ratioAt(100 + 16)).toBeCloseTo(-0.5, 12);
        expect(ratioAt(1100 - 16)).toBeCloseTo(0.5, 12);
        expect(ratioAt(100 + 8)).toBeCloseTo(-0.75, 12);
    });

    it("saturates at maxRatio once the pointer leaves the viewport", () => {
        expect(ratioAt(100)).toBe(-1);
        expect(ratioAt(-500)).toBe(-1.5);
        expect(ratioAt(5000)).toBe(1.5);
    });

    it("applies the ramp exponent to the magnitude only", () => {
        const ramp = { ...POLICY, rampExponent: 2 };
        expect(ratioAt(100 + 16, ramp)).toBeCloseTo(-0.25, 12);
        expect(ratioAt(1100 - 16, ramp)).toBeCloseTo(0.25, 12);
        expect(ratioAt(5000, ramp)).toBeCloseTo(1.5 * 1.5, 12);
    });

    it("falls back to the documented defaults for missing / invalid policy fields", () => {
        const loose = { bandPx: Number.NaN, maxRatio: -1, rampExponent: 0 };
        // bandPx → 32、maxRatio → 1.5、rampExponent → 1
        expect(ratioAt(1100 - 16, loose as EdgeScrollAxisPolicy)).toBeCloseTo(0.5, 12);
        expect(ratioAt(5000, loose as EdgeScrollAxisPolicy)).toBeCloseTo(1.5, 12);
    });

    it("returns zero for degenerate geometry or non-finite input", () => {
        expect(resolveEdgeScrollRatio({ ...VIEW, pointerPx: Number.NaN, policy: POLICY })).toBe(0);
        expect(
            resolveEdgeScrollRatio({ minPx: 100, maxPx: 100, pointerPx: 100, policy: POLICY }),
        ).toBe(0);
        expect(
            resolveEdgeScrollRatio({ minPx: 200, maxPx: 100, pointerPx: 200, policy: POLICY }),
        ).toBe(0);
    });
});

/** 受控的水平轴宿主：位置、上下界可变。 */
function createHorizontal(initial = 0, max = 10_000, min = 0) {
    let position = initial;
    const axis = createHorizontalEdgeScrollAxis({
        getScrollLeft: () => position,
        setScrollLeft: (next) => {
            position = next;
        },
        getMaxScrollLeft: () => max,
        getMinScrollLeft: () => min,
        maxSpeedPxPerSec: 420,
    });
    return {
        axis,
        position: () => position,
        setMax: (next: number) => {
            max = next;
        },
    };
}

describe("createHorizontalEdgeScrollAxis", () => {
    it("uses the documented default policy (32px / 1.5 / linear)", () => {
        expect(DEFAULT_EDGE_SCROLL_AXIS_POLICY).toEqual({
            bandPx: 32,
            maxRatio: 1.5,
            rampExponent: 1,
        });
    });

    it("moves at 420 px/s at full band and scales linearly with the frame time", () => {
        const full = createHorizontal(500);
        expect(full.axis.apply(1, FRAME_60HZ)).toBe(true);
        expect(full.position() - 500).toBeCloseTo((420 * FRAME_60HZ) / 1000, 9);

        const half = createHorizontal(500);
        half.axis.apply(0.5, FRAME_60HZ);
        expect(half.position() - 500).toBeCloseTo((210 * FRAME_60HZ) / 1000, 9);
    });

    it("does not move when the frame time is zero (clock just reset)", () => {
        const h = createHorizontal(500);
        expect(h.axis.apply(1, 0)).toBe(false);
        expect(h.position()).toBe(500);
    });

    it("clamps to the upper bound and reports no movement when pinned", () => {
        const h = createHorizontal(0, 5);
        expect(h.axis.apply(1, FRAME_60HZ)).toBe(true);
        expect(h.position()).toBe(5);
        expect(h.axis.apply(1, FRAME_60HZ)).toBe(false);
    });

    it("accepts a negative lower bound (synchronised timeline gutter)", () => {
        const h = createHorizontal(0, 10_000, -200);
        expect(h.axis.apply(-1, FRAME_60HZ)).toBe(true);
        expect(h.position()).toBeLessThan(0);
    });

    it("reports cannot-scroll when the bounds collapse", () => {
        const h = createHorizontal(0, 0, 0);
        expect(h.axis.canScroll()).toBe(false);
        expect(h.axis.apply(1, FRAME_60HZ)).toBe(false);
    });

    it("ignores a non-finite frame time instead of writing NaN", () => {
        const h = createHorizontal(500);
        expect(h.axis.apply(1, Number.NaN)).toBe(false);
        expect(h.position()).toBe(500);
    });
});
