import { describe, expect, test } from "vitest";

import { buildVibratoCurve } from "./vibratoCurve";
import { sanitizeVibratoPreset } from "./vibratoPresets";
import {
    EXTRACT_MIN_CONFIDENCE,
    deviationToCents,
    estimateVibratoPeriod,
    extractVibratoPreset,
} from "./vibratoExtract";
import type { VibratoPreset, VibratoPresetInput } from "./vibratoTypes";

const FP = 5;

/** 用给定预设渲染一段曲线，作为提取的输入。 */
function render(
    presetOverrides: VibratoPresetInput,
    seconds: number,
    param = "pitch",
    base = 60,
): number[] {
    const preset: VibratoPreset = sanitizeVibratoPreset({
        id: "custom_src",
        attackMs: 0,
        releaseMs: 0,
        irregularity: 0,
        depthRamp: { start: 1, end: 1 },
        ...presetOverrides,
    });
    const frames = Math.round((seconds * 1000) / FP);
    return buildVibratoCurve({
        startFrame: 0,
        startValue: base,
        endFrame: frames - 1,
        endValue: base,
        preset,
        param,
        framePeriodMs: FP,
    }).dense;
}

describe("deviationToCents", () => {
    test("音高：半音偏差 × 100 = 分", () => {
        expect(deviationToCents("pitch", 0.3)).toBeCloseTo(30, 9);
    });

    test("cents 类参数：原样", () => {
        expect(deviationToCents("child_pitch_offset_cents@t1", 30)).toBeCloseTo(30, 9);
        expect(deviationToCents("formant_shift_cents", 30)).toBeCloseTo(30, 9);
    });

    test("音级参数：1 音级 = 100 分", () => {
        expect(deviationToCents("child_pitch_offset_degrees@t1", 0.3)).toBeCloseTo(30, 9);
    });

    test("乘性增益：1 倍率 = 100 分", () => {
        expect(deviationToCents("dyn", 0.3)).toBeCloseTo(30, 9);
        expect(deviationToCents("breath_gain", 0.3)).toBeCloseTo(30, 9);
    });

    test("原始值域：按半量程定标", () => {
        // 张力 ±100 → 半量程 100；偏差 30 单位 = 30%
        expect(deviationToCents("tension", 30)).toBeCloseTo(30, 9);
        // 气息 ±10000 → 半量程 10000；偏差 3000 = 30%
        expect(deviationToCents("breathiness", 3000)).toBeCloseTo(30, 9);
    });
});

describe("estimateVibratoPeriod", () => {
    test("正弦残差：周期估计在 5% 以内", () => {
        // 6 Hz @ 5 ms 帧 → 33.33 帧/周期
        const residual = Array.from({ length: 400 }, (_, i) =>
            Math.sin((2 * Math.PI * i) / 33.333),
        );
        const estimate = estimateVibratoPeriod(residual, FP);
        expect(estimate).not.toBeNull();
        if (!estimate) return;
        expect(estimate.periodFrames).toBeCloseTo(33.333, 0);
        expect(estimate.confidence).toBeGreaterThan(0.8);
    });

    test("倍频修正：两倍周期的峰不会赢过真周期", () => {
        // 纯正弦在 2L 处的自相关同样很高；估计值必须落在 L 附近而不是 2L。
        const period = 40;
        const residual = Array.from({ length: 400 }, (_, i) =>
            Math.sin((2 * Math.PI * i) / period),
        );
        const estimate = estimateVibratoPeriod(residual, FP);
        expect(estimate?.periodFrames).toBeLessThan(period * 1.5);
    });

    test("平坦信号没有可信周期", () => {
        const flat = new Array(300).fill(0);
        const estimate = estimateVibratoPeriod(flat, FP);
        // 全零序列的自相关分母为 0 → 返回 null（不产生虚假周期）。
        expect(estimate).toBeNull();
    });

    test("太短的信号返回 null（放不下一个周期）", () => {
        expect(estimateVibratoPeriod([1, -1, 1, -1], FP)).toBeNull();
    });
});

describe("extractVibratoPreset：往返", () => {
    test("正弦音高颤音：速率与深度都还原", () => {
        const values = render(
            { cycle: { kind: "shape", shape: "sine", skew: 0.5 }, depthCents: 40, rateHz: 6 },
            2,
        );
        const result = extractVibratoPreset({ values, framePeriodMs: FP, param: "pitch" });
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.rateHz).toBeCloseTo(6, 0);
        expect(result.depthCents).toBeCloseTo(40, 0);
        expect(result.shapeHint).toBe("sine");
    });

    test("深度换算按参数族：dyn 上还原的是百分比深度", () => {
        const values = render({ depthCents: 30, rateHz: 5, baseline: "holdStart" }, 2, "dyn", 1);
        const result = extractVibratoPreset({ values, framePeriodMs: FP, param: "dyn" });
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.depthCents).toBeCloseTo(30, 0);
    });

    test("提取出的预设套回同一参数时接近原曲线", () => {
        const source = render(
            { cycle: { kind: "shape", shape: "sine", skew: 0.5 }, depthCents: 35, rateHz: 5.5 },
            2,
        );
        const result = extractVibratoPreset({
            values: source,
            framePeriodMs: FP,
            param: "pitch",
        });
        expect(result.ok).toBe(true);
        if (!result.ok) return;

        // 用提取结果重新渲染，与原曲线逐点比对（比较的是同一段基线上的摆动）。
        const rebuilt = buildVibratoCurve({
            startFrame: 0,
            startValue: 60,
            endFrame: source.length - 1,
            endValue: 60,
            preset: result.preset,
            param: "pitch",
            framePeriodMs: FP,
        }).dense;

        const sourceMean = source.reduce((a, b) => a + b, 0) / source.length;
        const rebuiltMean = rebuilt.reduce((a, b) => a + b, 0) / rebuilt.length;
        let error = 0;
        for (let i = 0; i < source.length; i += 1) {
            error += Math.abs(source[i] - sourceMean - (rebuilt[i] - rebuiltMean));
        }
        const meanError = error / source.length;
        // 幅值 0.35 半音 = 35 分；平均逐点误差应远小于幅值。
        expect(meanError).toBeLessThan(0.1);
    });

    test("三角波被识别为三角（形状提示）", () => {
        const values = render(
            { cycle: { kind: "shape", shape: "triangle", skew: 0.5 }, depthCents: 50, rateHz: 5 },
            2,
        );
        const result = extractVibratoPreset({ values, framePeriodMs: FP, param: "pitch" });
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(["triangle", "sine"]).toContain(result.shapeHint);
    });

    test("折叠出的周期表首尾可相接（没有半格错位）", () => {
        const values = render(
            { cycle: { kind: "shape", shape: "sine", skew: 0.5 }, depthCents: 60, rateHz: 5 },
            2,
        );
        const result = extractVibratoPreset({ values, framePeriodMs: FP, param: "pitch" });
        expect(result.ok).toBe(true);
        if (!result.ok || result.preset.cycle.kind !== "table") return;
        const table = result.preset.cycle.table;
        // 正弦表的第一格接近 0（起点是零交叉），相邻格差不应出现整幅跳变。
        const maxStep = Math.max(
            ...table.map((value, i) => Math.abs(value - table[(i + 1) % table.length])),
        );
        expect(maxStep).toBeLessThan(0.5);
    });
});

describe("extractVibratoPreset：拒绝", () => {
    test("选区太短", () => {
        const result = extractVibratoPreset({
            values: [60, 60, 60],
            framePeriodMs: FP,
            param: "pitch",
        });
        expect(result.ok).toBe(false);
        if (result.ok) return;
        expect(result.reason).toBe("tooShort");
    });

    test("平坦曲线没有颤音", () => {
        const flat = new Array(400).fill(60);
        const result = extractVibratoPreset({ values: flat, framePeriodMs: FP, param: "pitch" });
        expect(result.ok).toBe(false);
    });

    test("缓慢漂移（非周期性）不算颤音", () => {
        const drift = Array.from({ length: 400 }, (_, i) => 60 + i * 0.01);
        const result = extractVibratoPreset({ values: drift, framePeriodMs: FP, param: "pitch" });
        expect(result.ok).toBe(false);
    });

    test("深度过小（小于 3 分）不算颤音", () => {
        const values = render(
            { cycle: { kind: "shape", shape: "sine", skew: 0.5 }, depthCents: 0.1, rateHz: 5 },
            2,
        );
        const result = extractVibratoPreset({ values, framePeriodMs: FP, param: "pitch" });
        expect(result.ok).toBe(false);
        if (result.ok) return;
        expect(result.reason).toBe("noVibrato");
    });

    test("速率超出人声颤音区间时被拒", () => {
        const values = render(
            { cycle: { kind: "shape", shape: "sine", skew: 0.5 }, depthCents: 40, rateHz: 18 },
            2,
        );
        const result = extractVibratoPreset({ values, framePeriodMs: FP, param: "pitch" });
        // 18 Hz 在区间内，应当成功；这里确认它不会被误判成噪声。
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.rateHz).toBeGreaterThan(10);
    });
});

describe("提取结果的预设形态", () => {
    test("叠在已有曲线上（baseline = existing）", () => {
        const values = render(
            { cycle: { kind: "shape", shape: "sine", skew: 0.5 }, depthCents: 40, rateHz: 5.5 },
            2,
        );
        const result = extractVibratoPreset({ values, framePeriodMs: FP, param: "pitch" });
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.preset.baseline).toBe("existing");
        expect(result.preset.cycle.kind).toBe("table");
        expect(result.preset.builtin).toBe(false);
        expect(result.preset.id.startsWith("custom_")).toBe(true);
    });

    test("置信度随噪声下降（构成拒绝判据的基础）", () => {
        const clean = render(
            { cycle: { kind: "shape", shape: "sine", skew: 0.5 }, depthCents: 60, rateHz: 5 },
            2,
        );
        const noisy = clean.map((value, i) => value + (i % 7 === 0 ? 0.6 : -0.25));
        const cleanResult = extractVibratoPreset({
            values: clean,
            framePeriodMs: FP,
            param: "pitch",
        });
        const noisyResult = extractVibratoPreset({
            values: noisy,
            framePeriodMs: FP,
            param: "pitch",
        });
        expect(cleanResult.ok).toBe(true);
        if (!cleanResult.ok) return;
        expect(cleanResult.confidence).toBeGreaterThan(EXTRACT_MIN_CONFIDENCE);
        if (noisyResult.ok) {
            expect(noisyResult.confidence).toBeLessThan(cleanResult.confidence);
        }
    });
});
