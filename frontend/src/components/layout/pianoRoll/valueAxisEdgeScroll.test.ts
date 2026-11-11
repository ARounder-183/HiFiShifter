import { describe, expect, it } from "vitest";

import {
    VALUE_EDGE_SCROLL_BAND_PX,
    VALUE_EDGE_SCROLL_MAX_RATIO,
    VALUE_EDGE_SCROLL_RAMP_EXPONENT,
    VALUE_EDGE_SCROLL_VIEWPORTS_PER_SEC,
    valueAxisScrollable,
    valueEdgeScrollDeltaPx,
    valueEdgeScrollMaxSpeedPxPerSec,
} from "./valueAxisEdgeScroll";

const PITCH = { min: 0, max: 127, span: 24, scrollRangePx: 1600 } as const;
const FRAME_60HZ = 1000 / 60;

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

describe("valueEdgeScrollMaxSpeedPxPerSec", () => {
    it("moves exactly 0.12 visible value viewports per second", () => {
        const speed = valueEdgeScrollMaxSpeedPxPerSec(PITCH);
        expect(speed).toBeCloseTo(44.7379, 4);
        const centerPerSecond = speed * ((127 - 24) / 1600);
        expect(centerPerSecond / 24).toBeCloseTo(VALUE_EDGE_SCROLL_VIEWPORTS_PER_SEC, 12);
    });

    it("normalizes by visible span so zoom keeps the same screen speed", () => {
        for (const span of [6, 12, 24, 48]) {
            const speed = valueEdgeScrollMaxSpeedPxPerSec({ ...PITCH, span });
            const centerPerSecond = speed * ((127 - span) / 1600);
            expect(centerPerSecond / span).toBeCloseTo(VALUE_EDGE_SCROLL_VIEWPORTS_PER_SEC, 12);
        }
        expect(valueEdgeScrollMaxSpeedPxPerSec({ ...PITCH, span: 6 })).toBeCloseTo(9.5207, 4);
        expect(valueEdgeScrollMaxSpeedPxPerSec({ ...PITCH, span: 48 })).toBeCloseTo(116.6582, 4);
    });

    it("returns zero for a full or over-full value range", () => {
        expect(valueEdgeScrollMaxSpeedPxPerSec({ ...PITCH, span: 127 })).toBe(0);
        expect(valueEdgeScrollMaxSpeedPxPerSec({ ...PITCH, span: 200 })).toBe(0);
    });

    it("supports an injected viewports-per-second value", () => {
        const base = valueEdgeScrollMaxSpeedPxPerSec(PITCH);
        const doubled = valueEdgeScrollMaxSpeedPxPerSec({
            ...PITCH,
            viewportsPerSec: VALUE_EDGE_SCROLL_VIEWPORTS_PER_SEC * 2,
        });
        expect(doubled).toBeCloseTo(base * 2, 12);
    });

    it("returns zero for non-finite or non-positive inputs", () => {
        for (const bad of [Number.NaN, Number.POSITIVE_INFINITY]) {
            expect(valueEdgeScrollMaxSpeedPxPerSec({ ...PITCH, min: bad })).toBe(0);
            expect(valueEdgeScrollMaxSpeedPxPerSec({ ...PITCH, span: bad })).toBe(0);
            expect(valueEdgeScrollMaxSpeedPxPerSec({ ...PITCH, scrollRangePx: bad })).toBe(0);
        }
        expect(valueEdgeScrollMaxSpeedPxPerSec({ ...PITCH, viewportsPerSec: -1 })).toBe(0);
    });
});

describe("valueEdgeScrollDeltaPx", () => {
    const view = { topPx: 0, bottomPx: 600, frameMs: FRAME_60HZ } as const;

    it("uses an 18px band and is zero exactly at the band boundary", () => {
        expect(VALUE_EDGE_SCROLL_BAND_PX).toBe(18);
        expect(valueEdgeScrollDeltaPx({ ...view, ...PITCH, pointerPx: 582 })).toBe(0);
        expect(valueEdgeScrollDeltaPx({ ...view, ...PITCH, pointerPx: 18 })).toBe(0);
        expect(valueEdgeScrollDeltaPx({ ...view, ...PITCH, pointerPx: 600 })).toBeGreaterThan(0);
        expect(valueEdgeScrollDeltaPx({ ...view, ...PITCH, pointerPx: 0 })).toBeLessThan(0);
    });

    it("reaches the full-band scrollTop speed at the viewport edge", () => {
        const maxSpeed = valueEdgeScrollMaxSpeedPxPerSec(PITCH);
        const expected = (maxSpeed * FRAME_60HZ) / 1000;
        expect(valueEdgeScrollDeltaPx({ ...view, ...PITCH, pointerPx: 600 })).toBeCloseTo(
            expected,
            9,
        );
        expect(valueEdgeScrollDeltaPx({ ...view, ...PITCH, pointerPx: 0 })).toBeCloseTo(
            -expected,
            9,
        );
    });

    it("soft-starts with ratio^1.6 instead of jumping to half speed at half band", () => {
        expect(VALUE_EDGE_SCROLL_RAMP_EXPONENT).toBe(1.6);
        const maxSpeed = valueEdgeScrollMaxSpeedPxPerSec(PITCH);
        const full = (maxSpeed * FRAME_60HZ) / 1000;
        const atHalfBand = valueEdgeScrollDeltaPx({ ...view, ...PITCH, pointerPx: 591 });
        expect(atHalfBand).toBeCloseTo(full * Math.pow(0.5, 1.6), 9);
        expect(atHalfBand).toBeLessThan(full / 2);
    });

    it("does not accelerate after the pointer leaves the viewport (max ratio 1)", () => {
        expect(VALUE_EDGE_SCROLL_MAX_RATIO).toBe(1);
        const edge = valueEdgeScrollDeltaPx({ ...view, ...PITCH, pointerPx: 600 });
        expect(valueEdgeScrollDeltaPx({ ...view, ...PITCH, pointerPx: 5000 })).toBeCloseTo(
            edge,
            9,
        );
    });

    it("returns zero when the value axis cannot move", () => {
        expect(valueEdgeScrollDeltaPx({ ...view, ...PITCH, span: 127, pointerPx: 600 })).toBe(0);
        expect(valueEdgeScrollDeltaPx({ ...view, ...PITCH, span: 200, pointerPx: 600 })).toBe(0);
    });

    it("returns zero for degenerate or non-finite geometry", () => {
        expect(
            valueEdgeScrollDeltaPx({ ...view, ...PITCH, pointerPx: Number.NaN }),
        ).toBe(0);
        expect(
            valueEdgeScrollDeltaPx({ ...view, ...PITCH, pointerPx: 600, bottomPx: 0 }),
        ).toBe(0);
        expect(
            valueEdgeScrollDeltaPx({ ...view, ...PITCH, pointerPx: 600, topPx: Number.NaN }),
        ).toBe(0);
    });
});
