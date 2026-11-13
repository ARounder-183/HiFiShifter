import { describe, expect, it } from "vitest";

import {
    centerForPanDelta,
    centerForZoomAnchor,
    isAxisInverted,
    valueAtViewportT,
} from "./paramAxisProjection";

describe("paramAxisProjection", () => {
    it("only inverts pan when direction is left-top", () => {
        expect(isAxisInverted("pan", { pan: "left-top" })).toBe(true);
        expect(isAxisInverted("pan", {})).toBe(false);
        expect(isAxisInverted("volume", { pan: "left-top" })).toBe(false);
    });

    it("maps t to values in both directions", () => {
        expect(valueAtViewportT({ param: "pan", directions: {}, t: 0, min: -1, span: 2 })).toBe(1);
        expect(
            valueAtViewportT({
                param: "pan",
                directions: { pan: "left-top" },
                t: 0,
                min: -1,
                span: 2,
            }),
        ).toBe(-1);
        expect(
            valueAtViewportT({
                param: "pan",
                directions: { pan: "left-top" },
                t: 1,
                min: -1,
                span: 2,
            }),
        ).toBe(1);
    });

    it("keeps the pointer value fixed while zooming", () => {
        const normal = centerForZoomAnchor({
            param: "volume",
            directions: {},
            valueAtPointer: 0.5,
            t: 0.25,
            nextSpan: 1,
        });
        const inverted = centerForZoomAnchor({
            param: "pan",
            directions: { pan: "left-top" },
            valueAtPointer: 0.5,
            t: 0.25,
            nextSpan: 1,
        });
        expect(normal).toBe(0.25);
        expect(inverted).toBe(0.75);
    });

    it("negates pan delta only for inverted pan", () => {
        expect(
            centerForPanDelta({ param: "pan", directions: {}, center: 0, deltaCenter: 0.2 }),
        ).toBe(0.2);
        expect(
            centerForPanDelta({
                param: "pan",
                directions: { pan: "left-top" },
                center: 0,
                deltaCenter: 0.2,
            }),
        ).toBe(-0.2);
        expect(
            centerForPanDelta({ param: "volume", directions: {}, center: 0, deltaCenter: 0.2 }),
        ).toBe(0.2);
    });
});
