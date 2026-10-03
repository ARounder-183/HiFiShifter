import { describe, expect, test } from "vitest";

import {
    clampDepthCentsForParam,
    depthDisplayFactor,
    depthMappingFor,
    depthStepCentsFor,
    depthStepUnitFor,
    depthToDisplay,
    depthToParamUnit,
    displayToDepth,
    fullSwingCentsFor,
    FULL_SWING_DIVISIONS,
    NON_CENTS_FULL_SWING_CENTS,
    paramUnitToDepth,
} from "./vibratoDepth";
import { VIBRATO_LIMITS } from "./vibratoPresets";

/** 真实参数描述符里的量程（与后端 `renderer` 一致）。 */
const RANGES = {
    pitch: { min: 24, max: 108 },
    formantHifigan: { min: -500, max: 500 },
    formantVslib: { min: -2400, max: 2400 },
    childCents: { min: -2400, max: 2400 },
    childDegrees: { min: -14, max: 14 },
    volume: { min: 0, max: 2 },
    dyn: { min: 0, max: 1 },
    tension: { min: -100, max: 100 },
    breathiness: { min: -10000, max: 10000 },
    pan: { min: -1, max: 1 },
} as const;

describe("fullSwingCentsFor", () => {
    test("音高：深度上限就是满摆幅（音高轴的范围是音域，不是调制范围）", () => {
        // 音高轴 24..108 是"音域"，换算成 4200 分，远超工具深度上限，
        // 因此满摆幅取上限 1200 —— 与历史手感一致。
        expect(fullSwingCentsFor("pitch", RANGES.pitch)).toBe(VIBRATO_LIMITS.depthCents.max);
        expect(fullSwingCentsFor("pitch")).toBe(VIBRATO_LIMITS.depthCents.max);
    });

    test("cents 类窄量程参数：满摆幅被参数自己的量程压住", () => {
        // 共振峰 ±500：再大的深度也只会被写入口钳平，满摆幅就是 500。
        expect(fullSwingCentsFor("formant_shift_cents", RANGES.formantHifigan)).toBe(500);
        expect(fullSwingCentsFor("formant_shift_cents", RANGES.formantVslib)).toBe(
            VIBRATO_LIMITS.depthCents.max,
        );
    });

    test("音级类参数：量程按 100 分/音级换算后再与上限取小", () => {
        // ±14 音级 = ±1400 分 → 仍被 1200 的上限压住。
        expect(fullSwingCentsFor("child_pitch_offset_degrees@t1", RANGES.childDegrees)).toBe(
            VIBRATO_LIMITS.depthCents.max,
        );
    });

    test("子轨音分偏移：量程 ±2400 分，满摆幅仍是上限 1200", () => {
        expect(fullSwingCentsFor("child_pitch_offset_cents@t1", RANGES.childCents)).toBe(
            VIBRATO_LIMITS.depthCents.max,
        );
    });

    test("增益倍率与原始值域：满摆幅恒为 100 分（= ±1.0 倍率 / ±半量程）", () => {
        for (const [param, range] of [
            ["dyn", RANGES.dyn],
            ["volume", RANGES.volume],
            ["breath_gain", RANGES.volume],
            ["hifigan_tension", RANGES.tension],
            ["breathiness", RANGES.breathiness],
            ["pan", RANGES.pan],
        ] as const) {
            expect(fullSwingCentsFor(param, range)).toBe(NON_CENTS_FULL_SWING_CENTS);
        }
    });

    test("原始值域参数缺描述符时靠兜底量程拿到同一结果（张力 id 必须是真实 id）", () => {
        // 真实 id 是 `hifigan_tension`；漏掉它会让因子退回 {0,2} 的兜底量程，
        // 深度换算差 100 倍。满摆幅本身对原始值域是常量，但换算因子会错。
        expect(fullSwingCentsFor("hifigan_tension")).toBe(NON_CENTS_FULL_SWING_CENTS);
        expect(fullSwingCentsFor("tension")).toBe(NON_CENTS_FULL_SWING_CENTS);
    });
});

describe("depthStepCentsFor", () => {
    test("一律是满摆幅的 1/50", () => {
        expect(FULL_SWING_DIVISIONS).toBe(50);
        for (const [param, range] of [
            ["pitch", RANGES.pitch],
            ["child_pitch_offset_cents@t1", RANGES.childCents],
            ["formant_shift_cents", RANGES.formantHifigan],
            ["dyn", RANGES.dyn],
            ["hifigan_tension", RANGES.tension],
            ["pan", RANGES.pan],
            ["breathiness", RANGES.breathiness],
        ] as const) {
            expect(depthStepCentsFor(param, range)).toBeCloseTo(
                fullSwingCentsFor(param, range) / FULL_SWING_DIVISIONS,
                9,
            );
        }
    });

    test("音高与 cents 类：24 分/格（与历史手感一致）", () => {
        expect(depthStepCentsFor("pitch", RANGES.pitch)).toBe(24);
        expect(depthStepCentsFor("child_pitch_offset_cents@t1", RANGES.childCents)).toBe(24);
        expect(depthStepCentsFor("child_pitch_offset_degrees@t1", RANGES.childDegrees)).toBe(24);
    });

    test("窄量程的 cents 参数按自己的满摆幅走（共振峰 ±500 → 10 分/格）", () => {
        expect(depthStepCentsFor("formant_shift_cents", RANGES.formantHifigan)).toBe(10);
    });

    test("增益倍率与原始值域：2 分/格（2%），各族不再各写各的", () => {
        // 旧实现：增益 1 分、原始值域 0.5 分、cents 类按原生量程/200 ——
        // 同一个"一格"在不同参数上差几十倍，声像上从零扫到满幅要 400 格。
        expect(depthStepCentsFor("dyn", RANGES.dyn)).toBe(2);
        expect(depthStepCentsFor("volume", RANGES.volume)).toBe(2);
        expect(depthStepCentsFor("hifigan_tension", RANGES.tension)).toBe(2);
        expect(depthStepCentsFor("breathiness", RANGES.breathiness)).toBe(2);
        expect(depthStepCentsFor("pan", RANGES.pan)).toBe(2);
    });
});

describe("clampDepthCentsForParam", () => {
    test("音高：钳在工具上限", () => {
        expect(clampDepthCentsForParam(5000, "pitch", RANGES.pitch)).toBe(
            VIBRATO_LIMITS.depthCents.max,
        );
        expect(clampDepthCentsForParam(-5000, "pitch", RANGES.pitch)).toBe(
            VIBRATO_LIMITS.depthCents.min,
        );
    });

    test("窄量程参数：钳在参数自己的量程", () => {
        expect(clampDepthCentsForParam(1200, "formant_shift_cents", RANGES.formantHifigan)).toBe(
            500,
        );
        expect(clampDepthCentsForParam(1200, "pan", RANGES.pan)).toBe(100);
        expect(clampDepthCentsForParam(-1200, "pan", RANGES.pan)).toBe(-100);
    });

    test("量程内的值原样通过", () => {
        expect(clampDepthCentsForParam(37, "pan", RANGES.pan)).toBe(37);
        expect(clampDepthCentsForParam(-37, "dyn", RANGES.dyn)).toBe(-37);
    });

    test("非有限值归零而不是传播 NaN", () => {
        expect(clampDepthCentsForParam(Number.NaN, "pitch")).toBe(0);
        expect(clampDepthCentsForParam(Number.POSITIVE_INFINITY, "pan", RANGES.pan)).toBe(100);
    });
});

describe("depthMappingFor：按参数类型换算振幅", () => {
    test("音级类参数按 1 音级 = 100 分换算（曾漏判成原始值域，深度大 14 倍）", () => {
        expect(depthMappingFor("child_pitch_offset_degrees@t1", RANGES.childDegrees)).toEqual({
            family: "cents",
            mode: "additive",
            factor: 0.01,
        });
        // 50 分的揉弦 = 0.5 音级；漏判时会算成 7 音级。
        expect(
            depthToParamUnit(50, "child_pitch_offset_degrees@t1", RANGES.childDegrees),
        ).toBeCloseTo(0.5, 9);
        expect(
            paramUnitToDepth(0.5, "child_pitch_offset_degrees@t1", RANGES.childDegrees),
        ).toBeCloseTo(50, 9);
    });

    test("原始值域参数的兜底量程按真实 id 命中（张力曾漏成 {0,2}，深度小 100 倍）", () => {
        // 张力 ±100：半量程 100 → 因子 1，30 分 = 30 个原生单位。
        expect(depthMappingFor("hifigan_tension", RANGES.tension).factor).toBeCloseTo(1, 9);
        expect(depthToParamUnit(30, "hifigan_tension", RANGES.tension)).toBeCloseTo(30, 9);
        // 缺描述符时靠兜底表；表里漏掉真实 id 会退到 {0, 2} → 因子 0.01。
        expect(depthMappingFor("hifigan_tension").factor).toBeCloseTo(1, 9);
        expect(depthMappingFor("pan").factor).toBeCloseTo(0.01, 9);
        expect(depthMappingFor("breathiness").factor).toBeCloseTo(100, 9);
    });

    test("增益倍率恒按 cents/100 换算，静音段不被抬起（乘性）", () => {
        expect(depthMappingFor("volume", RANGES.volume)).toEqual({
            family: "ratio",
            mode: "multiplicative",
            factor: 0.01,
        });
        expect(depthMappingFor("dyn", RANGES.dyn).factor).toBeCloseTo(0.01, 9);
    });
});

describe("显示换算与步长单位", () => {
    test("cents 族按分显示，因子 1", () => {
        expect(depthDisplayFactor("pitch", RANGES.pitch)).toBe(1);
        expect(depthToDisplay(30, "pitch", RANGES.pitch)).toBe(30);
        expect(displayToDepth(30, "pitch", RANGES.pitch)).toBe(30);
        expect(depthStepUnitFor("pitch")).toBe("cents");
    });

    test("音级类按音级显示", () => {
        expect(depthToDisplay(50, "child_pitch_offset_degrees@t1", RANGES.childDegrees)).toBe(0.5);
        expect(displayToDepth(0.5, "child_pitch_offset_degrees@t1", RANGES.childDegrees)).toBe(50);
        expect(depthStepUnitFor("child_pitch_offset_degrees@t1")).toBe("scaleDegree");
    });

    test("增益倍率与原始值域按满摆幅百分比显示（不再是裸的原生数字）", () => {
        // 声像原生量程 ±1：旧实现把 30 分显示成 `0.3`，气声（±10000）显示成 `3000`。
        expect(depthToDisplay(30, "pan", RANGES.pan)).toBe(30);
        expect(depthToDisplay(30, "breathiness", RANGES.breathiness)).toBe(30);
        expect(depthToDisplay(30, "hifigan_tension", RANGES.tension)).toBe(30);
        expect(depthStepUnitFor("pan")).toBe("percent");
        expect(depthStepUnitFor("breathiness")).toBe("percent");
        expect(depthStepUnitFor("dyn")).toBe("percent");
    });

    test("显示换算在各族上都可逆", () => {
        for (const [param, range] of [
            ["pitch", RANGES.pitch],
            ["dyn", RANGES.dyn],
            ["hifigan_tension", RANGES.tension],
            ["breathiness", RANGES.breathiness],
            ["pan", RANGES.pan],
            ["child_pitch_offset_degrees@t1", RANGES.childDegrees],
        ] as const) {
            expect(displayToDepth(depthToDisplay(37, param, range), param, range)).toBeCloseTo(
                37,
                6,
            );
        }
    });
});
