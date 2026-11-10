import { describe, expect, it } from "vitest";
import {
    VALUE_EDGE_SCROLL_VIEWPORTS_PER_SEC,
    valueAxisScrollable,
    valueEdgeScrollMaxSpeedPxPerSec,
} from "./verticalScrollMapping";

describe("valueAxisScrollable", () => {
    it("reports a movable center only when the span is smaller than the range", () => {
        expect(valueAxisScrollable({ min: 0, max: 127, span: 24 })).toBe(true);
        expect(valueAxisScrollable({ min: 0, max: 127, span: 127 })).toBe(false);
        expect(valueAxisScrollable({ min: 0, max: 127, span: 200 })).toBe(false);
        expect(valueAxisScrollable({ min: 0, max: 0, span: 0 })).toBe(false);
    });

    it("is safe for non-finite input", () => {
        expect(valueAxisScrollable({ min: Number.NaN, max: 1, span: 0 })).toBe(false);
        expect(valueAxisScrollable({ min: 0, max: Number.POSITIVE_INFINITY, span: 1 })).toBe(true);
    });
});

describe("valueEdgeScrollMaxSpeedPxPerSec", () => {
    it("moves about 0.30 visible value viewports per second", () => {
        const speed = valueEdgeScrollMaxSpeedPxPerSec({
            min: 0,
            max: 127,
            span: 24,
            scrollRangePx: 1600,
        });
        // 0.30 * 24 * 1600 / (127 - 24)
        expect(speed).toBeCloseTo(111.8447, 3);

        const centerPerSecond = speed * ((127 - 24) / 1600);
        expect(centerPerSecond / 24).toBeCloseTo(VALUE_EDGE_SCROLL_VIEWPORTS_PER_SEC, 9);
    });

    it("scales with the visible span so zoom does not change the feel", () => {
        const at12 = valueEdgeScrollMaxSpeedPxPerSec({
            min: 0,
            max: 127,
            span: 12,
            scrollRangePx: 1600,
        });
        const at48 = valueEdgeScrollMaxSpeedPxPerSec({
            min: 0,
            max: 127,
            span: 48,
            scrollRangePx: 1600,
        });
        expect(at12).toBeCloseTo(50.087, 3);
        expect(at48).toBeCloseTo(291.646, 3);
        expect(at48).toBeGreaterThan(at12);
    });

    it("returns zero when the value axis cannot move", () => {
        expect(
            valueEdgeScrollMaxSpeedPxPerSec({ min: 0, max: 127, span: 127, scrollRangePx: 1600 }),
        ).toBe(0);
        expect(
            valueEdgeScrollMaxSpeedPxPerSec({ min: 0, max: 127, span: 200, scrollRangePx: 1600 }),
        ).toBe(0);
    });

    it("accepts an injected viewports-per-second value", () => {
        const base = valueEdgeScrollMaxSpeedPxPerSec({
            min: 0,
            max: 127,
            span: 24,
            scrollRangePx: 1600,
        });
        const doubled = valueEdgeScrollMaxSpeedPxPerSec({
            min: 0,
            max: 127,
            span: 24,
            scrollRangePx: 1600,
            viewportsPerSec: VALUE_EDGE_SCROLL_VIEWPORTS_PER_SEC * 2,
        });
        expect(doubled).toBeCloseTo(base * 2, 9);
    });

    it("returns zero for non-finite or non-positive inputs", () => {
        expect(
            valueEdgeScrollMaxSpeedPxPerSec({
                min: Number.NaN,
                max: 127,
                span: 24,
                scrollRangePx: 1600,
            }),
        ).toBe(0);
        expect(
            valueEdgeScrollMaxSpeedPxPerSec({
                min: 0,
                max: 127,
                span: Number.NaN,
                scrollRangePx: 1600,
            }),
        ).toBe(0);
        expect(
            valueEdgeScrollMaxSpeedPxPerSec({
                min: 0,
                max: 127,
                span: 24,
                scrollRangePx: Number.NaN,
            }),
        ).toBe(0);
        expect(
            valueEdgeScrollMaxSpeedPxPerSec({
                min: 0,
                max: 127,
                span: 24,
                scrollRangePx: 1600,
                viewportsPerSec: -1,
            }),
        ).toBe(0);
    });
});
