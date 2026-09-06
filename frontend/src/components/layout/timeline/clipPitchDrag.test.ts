import { describe, expect, it } from "vitest";
import {
    PITCH_DRAG_CENTS_PER_PX,
    computePitchDragCents,
    formatPitchDragCents,
    shiftPitchFrames,
    shiftPitchValue,
} from "./clipPitchDrag";

describe("computePitchDragCents", () => {
    it("向上拖拽（clientY 减小）为正音分", () => {
        expect(computePitchDragCents(-50)).toBe(50 * PITCH_DRAG_CENTS_PER_PX);
    });

    it("向下拖拽为负音分；0 位移为 0", () => {
        expect(computePitchDragCents(25)).toBe(-25 * PITCH_DRAG_CENTS_PER_PX);
        expect(computePitchDragCents(0)).toBe(0);
    });
});

describe("shiftPitchFrames", () => {
    it("整体平移所有非零帧（半音 = 音分/100）", () => {
        expect(shiftPitchFrames([60, 60.5, 61], 1.2)).toEqual([61.2, 61.7, 62.2]);
    });

    it("无声帧（0）保持为 0，与移调对话框语义一致", () => {
        expect(shiftPitchFrames([0, 60, 0], 2)).toEqual([0, 62, 0]);
    });

    it("偏移为 0 时返回内容相同的副本，不改写输入", () => {
        const base = [60, 0, 61.5];
        const out = shiftPitchFrames(base, 0);
        expect(out).not.toBe(base);
        expect(out).toEqual(base);
    });
});

describe("shiftPitchValue", () => {
    it("非零帧加半音偏移；无声帧（0）保持为 0", () => {
        expect(shiftPitchValue(60, 1.2)).toBe(61.2);
        expect(shiftPitchValue(60, -2)).toBe(58);
        expect(shiftPitchValue(0, 3)).toBe(0);
    });
});

describe("formatPitchDragCents", () => {
    it("带符号；整数值不出现小数点", () => {
        expect(formatPitchDragCents(123.4)).toBe("+123.4");
        expect(formatPitchDragCents(-100)).toBe("-100");
        expect(formatPitchDragCents(2)).toBe("+2");
    });

    it("近零显示为 0", () => {
        expect(formatPitchDragCents(0.04)).toBe("0");
        expect(formatPitchDragCents(0)).toBe("0");
    });
});
