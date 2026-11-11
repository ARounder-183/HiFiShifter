import { describe, expect, it } from "vitest";

import {
    DEFAULT_PAN_AXIS_DIRECTION,
    formatPanAxisLabel,
    formatPanValue,
    nextPanAxisDirection,
    normalizePanAxisDirections,
    panPercentOf,
    panSideOf,
    resolvePanAxisDirection,
} from "./paramPan";
import { formatAxisMarkLabel } from "./kernel/scene/axisMarkInstances";

const labels = { left: "左", center: "中", right: "右" };

describe("paramPan display", () => {
    it("formats axis labels without signed numbers", () => {
        expect(formatPanAxisLabel(-1, labels)).toBe("左100");
        expect(formatPanAxisLabel(-0.5, labels)).toBe("左50");
        expect(formatPanAxisLabel(0, labels)).toBe("中");
        expect(formatPanAxisLabel(0.5, labels)).toBe("右50");
        expect(formatPanAxisLabel(1, labels)).toBe("右100");
    });

    it("formats readouts with an explicit percent", () => {
        expect(formatPanValue(-0.35, labels)).toBe("左 35%");
        expect(formatPanValue(0.35, labels)).toBe("右 35%");
        expect(formatPanValue(0, labels)).toBe("中");
    });

    it("treats floating-point dust around zero as center", () => {
        expect(panSideOf(-1e-9)).toBe("center");
        expect(panPercentOf(Number.NaN)).toBe(0);
    });

    it("normalizes only valid pan entries", () => {
        expect(normalizePanAxisDirections({ pan: "left-top", volume: "left-top" })).toEqual({
            pan: "left-top",
        });
        expect(normalizePanAxisDirections({ pan: "up" })).toEqual({});
        expect(normalizePanAxisDirections(null)).toEqual({});
    });

    it("defaults to left-bottom and toggles", () => {
        expect(DEFAULT_PAN_AXIS_DIRECTION).toBe("left-bottom");
        expect(resolvePanAxisDirection(undefined, "pan")).toBe("left-bottom");
        expect(resolvePanAxisDirection({ pan: "left-top" }, "pan")).toBe("left-top");
        expect(nextPanAxisDirection("left-bottom")).toBe("left-top");
        expect(nextPanAxisDirection("left-top")).toBe("left-bottom");
    });
});

describe("axis mark labels for pan", () => {
    it("uses pan labels before dB and child-offset formatting", () => {
        expect(formatAxisMarkLabel(-1, "pan", "db", labels)).toBe("左100");
        expect(formatAxisMarkLabel(0, "pan", undefined, labels)).toBe("中");
        expect(formatAxisMarkLabel(0.5, "pan", undefined, labels)).toBe("右50");
    });
});
