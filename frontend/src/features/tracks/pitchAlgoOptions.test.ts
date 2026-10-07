import { describe, expect, it } from "vitest";

import {
    buildPitchAlgoOptions,
    PITCH_ALGO_IDS,
    resolveEffectivePitchAlgo,
    resolvePitchAlgoSelectValue,
} from "./pitchAlgoOptions";

/** 造一份选项列表（默认 noneLabel / "不可用"标注固定，便于断言）。 */
function options(args: { vslibAvailable: boolean | null; currentValue?: string }) {
    return buildPitchAlgoOptions({
        noneLabel: "None",
        formatUnavailable: (label) => `${label} (unavailable)`,
        vslibAvailable: args.vslibAvailable,
        currentValue: args.currentValue,
    });
}

function values(list: ReturnType<typeof options>): string[] {
    return list.map((option) => option.value);
}

describe("buildPitchAlgoOptions", () => {
    it("offers vslib when it is available", () => {
        expect(values(options({ vslibAvailable: true }))).toEqual([
            "nsf_hifigan_onnx",
            "world_dll",
            "vslib",
            "none",
        ]);
    });

    it("hides vslib when it is unavailable", () => {
        expect(values(options({ vslibAvailable: false }))).toEqual([
            "nsf_hifigan_onnx",
            "world_dll",
            "none",
        ]);
    });

    it("hides vslib while availability is still unknown", () => {
        // 未知态按不可用处理：宁可短暂不显示，也不显示一个点下去会静默
        // 回退到别的算法的选项。
        expect(values(options({ vslibAvailable: null }))).not.toContain("vslib");
    });

    it("keeps an unavailable vslib that is the current value, marked as such", () => {
        const list = options({ vslibAvailable: false, currentValue: "vslib" });
        expect(values(list)).toContain("vslib");
        const vslib = list.find((option) => option.value === "vslib");
        expect(vslib?.label).toBe("vslib (unavailable)");
    });

    it("does not mark vslib when it is available", () => {
        const list = options({ vslibAvailable: true, currentValue: "vslib" });
        expect(list.find((option) => option.value === "vslib")?.label).toBe("vslib");
    });

    it("always keeps none, using the supplied label", () => {
        for (const available of [true, false, null]) {
            const list = options({ vslibAvailable: available });
            expect(list.find((option) => option.value === "none")?.label).toBe("None");
        }
    });
});

describe("resolvePitchAlgoSelectValue", () => {
    it("passes through a value that is in the options", () => {
        const list = options({ vslibAvailable: true });
        expect(resolvePitchAlgoSelectValue("world_dll", list)).toBe("world_dll");
    });

    it("passes through a kept-but-unavailable vslib", () => {
        const list = options({ vslibAvailable: false, currentValue: "vslib" });
        expect(resolvePitchAlgoSelectValue("vslib", list)).toBe("vslib");
    });

    it("falls back to nsf-hifigan for an unknown value", () => {
        const list = options({ vslibAvailable: true });
        expect(resolvePitchAlgoSelectValue("something_else", list)).toBe("nsf_hifigan_onnx");
        expect(resolvePitchAlgoSelectValue(undefined, list)).toBe("nsf_hifigan_onnx");
    });
});

describe("resolveEffectivePitchAlgo", () => {
    it("passes every known id through", () => {
        for (const id of PITCH_ALGO_IDS) {
            expect(resolveEffectivePitchAlgo(id)).toBe(id);
        }
    });

    it("falls back to the project default for unrecognized values", () => {
        // 后端 `PitchAnalysisAlgo::effective()` 对未知算法同样回退到默认算法；
        // 前端任何按算法分支的地方都必须给出同一个答案，否则界面与声音对不上。
        for (const raw of ["unknown", "world_onnx_2027", "", undefined, null]) {
            expect(resolveEffectivePitchAlgo(raw)).toBe("nsf_hifigan_onnx");
        }
    });
});
