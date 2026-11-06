import { describe, expect, it } from "vitest";
import { normalizePitchToNearestSemitone, scalePitchAroundCenter } from "./paramBlockPitch";

describe("normalizePitchToNearestSemitone", () => {
    it("整段平移到最近半音，保留颤音轮廓", () => {
        const out = normalizePitchToNearestSemitone({
            values: [59.8, 60.2, 60.1, 0],
            modeCenter: 60.0,
        });
        expect(out[0]).toBeCloseTo(59.8, 6);
        expect(out[1]).toBeCloseTo(60.2, 6);
        expect(out[2]).toBeCloseTo(60.1, 6);
        expect(out[3]).toBe(0);
    });

    it("落在黑键上的块也吸附到最近的半音", () => {
        const out = normalizePitchToNearestSemitone({
            values: [60.8, 61.2, 61.0],
            modeCenter: 60.8,
        });
        expect(out[0]).toBeCloseTo(61, 6);
        expect(out[1]).toBeCloseTo(61.4, 6);
        expect(out[2]).toBeCloseTo(61.2, 6);
    });
});

describe("scalePitchAroundCenter", () => {
    it("放大与压平都围绕中心，0 哨兵不变", () => {
        const values = [59, 60, 61, 0];
        expect(scalePitchAroundCenter({ values, center: 60, factor: 2 })).toEqual([
            58, 60, 62, 0,
        ]);
        expect(scalePitchAroundCenter({ values, center: 60, factor: 0.5 })).toEqual([
            59.5, 60, 60.5, 0,
        ]);
    });

    it("缩放结果钳制在可编辑 MIDI 范围", () => {
        const out = scalePitchAroundCenter({
            values: [1, 127],
            center: 64,
            factor: 10,
            min: 1,
            max: 127,
        });
        expect(out).toEqual([1, 127]);
    });
});
