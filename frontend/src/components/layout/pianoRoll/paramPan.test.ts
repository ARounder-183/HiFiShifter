import { describe, expect, it } from "vitest";

import {
    DEFAULT_PAN_AXIS_DIRECTION,
    PAN_AXIS_CENTER,
    PAN_AXIS_LEFT,
    PAN_AXIS_RIGHT,
    formatPanAxisCompactLabel,
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
    it("formats axis labels with a language-independent single letter", () => {
        // 端点/中心只给字母：轴列窄，`Left100` 这类完整词会挤占相邻刻度。
        expect(formatPanAxisCompactLabel(-1)).toBe("L");
        expect(formatPanAxisCompactLabel(-0.5)).toBe("L50");
        expect(formatPanAxisCompactLabel(0)).toBe("C");
        expect(formatPanAxisCompactLabel(0.5)).toBe("R50");
        expect(formatPanAxisCompactLabel(1)).toBe("R");
        expect(formatPanAxisCompactLabel(0.25)).toBe("R25");
    });

    it("keeps the axis letters short and locale-free", () => {
        // 三个字母是导出常量，其它模块（角标）复用同一份口径。
        expect([PAN_AXIS_LEFT, PAN_AXIS_CENTER, PAN_AXIS_RIGHT]).toEqual(["L", "C", "R"]);
        for (const v of [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]) {
            expect(formatPanAxisCompactLabel(v).length).toBeLessThanOrEqual(4);
        }
    });

    it("treats floating-point dust around zero as the center letter", () => {
        expect(formatPanAxisCompactLabel(-1e-9)).toBe("C");
        expect(formatPanAxisCompactLabel(1e-12)).toBe("C");
    });

    it("formats readouts with the full localized word and an explicit percent", () => {
        // 读数通道保持完整词：ToolTip / 拖拽读数 / Set Value 空间充足，语义要明确。
        expect(formatPanValue(-0.35, labels)).toBe("左 35%");
        expect(formatPanValue(0.35, labels)).toBe("右 35%");
        expect(formatPanValue(0, labels)).toBe("中");
        expect(formatPanValue(-0.35, { left: "Left", center: "Center", right: "Right" })).toBe(
            "Left 35%",
        );
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
    it("uses the compact letter table before dB and child-offset formatting", () => {
        // pan 分支必须最先命中：声像不是 dB 量、也不走子音高偏移的显示换算。
        expect(formatAxisMarkLabel(-1, "pan", "db")).toBe("L");
        expect(formatAxisMarkLabel(0, "pan")).toBe("C");
        expect(formatAxisMarkLabel(0.5, "pan")).toBe("R50");
    });
});
