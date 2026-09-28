import { describe, expect, it } from "vitest";

import { canConvertParam, planParamConversion } from "./paramConversion";

describe("planParamConversion", () => {
    it("maps volume → dyn（方向与切换目标）", () => {
        const plan = planParamConversion("volume");
        expect(plan).not.toBeNull();
        expect(plan!.direction).toBe("volume_to_dyn");
        expect(plan!.targetParam).toBe("dyn");
        expect(plan!.sourceParam).toBe("volume");
    });

    it("maps dyn → volume（方向与切换目标）", () => {
        const plan = planParamConversion("dyn");
        expect(plan).not.toBeNull();
        expect(plan!.direction).toBe("dyn_to_volume");
        expect(plan!.targetParam).toBe("volume");
        expect(plan!.sourceParam).toBe("dyn");
    });

    it("returns null for unrelated params", () => {
        for (const p of ["pitch", "pan", "formant_shift_cents", "breath_gain", "synth_mode"]) {
            expect(planParamConversion(p)).toBeNull();
        }
    });
});

describe("canConvertParam", () => {
    it("requires a non-empty selection", () => {
        expect(canConvertParam({ editParam: "volume", selectionFrameCount: 0 })).toBe(false);
        expect(canConvertParam({ editParam: "volume", selectionFrameCount: 1 })).toBe(true);
        expect(canConvertParam({ editParam: "dyn", selectionFrameCount: 500 })).toBe(true);
    });

    it("is unavailable for params without a counterpart", () => {
        expect(canConvertParam({ editParam: "pitch", selectionFrameCount: 100 })).toBe(false);
        expect(canConvertParam({ editParam: "pan", selectionFrameCount: 100 })).toBe(false);
    });
});
