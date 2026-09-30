import { describe, expect, test } from "vitest";

import type { MessageKey } from "../../../i18n/messages";
import { enUS } from "../../../i18n/en-US";
import {
    builtinVibratoPresetId,
    SYSTEM_VIBRATO_PRESETS,
} from "../../../features/vibrato/systemPresets";
import { sanitizeVibratoPreset } from "../../../features/vibrato/vibratoPresets";
import {
    BASELINE_MODE_KEYS,
    BASELINE_MODE_ORDER,
    ENVELOPE_CURVE_KEYS,
    ENVELOPE_CURVE_ORDER,
    RATE_MODE_KEYS,
    PREVIEW_RANGE_LADDER,
    WAVE_SHAPE_KEYS,
    WAVE_SHAPE_ORDER,
    buildAppliedPreview,
    buildVibratoPreview,
    builtinIdOf,
    cycleShapeLabelKey,
    depthForParam,
    depthToCents,
    depthUnitLabelKey,
    fitPreviewRangeCents,
    formatNumber,
    previewScaleCents,
    vibratoPresetDescription,
    vibratoPresetLabel,
    vibratoPresetSummary,
} from "./vibratoDialogLogic";

/** 用参考语系的真实词条当翻译函数：这样"键写错了"会当场暴露成缺键。 */
const t = (key: MessageKey): string => enUS[key];

describe("标签映射", () => {
    test("波形形状的词条键齐全且真实存在", () => {
        for (const shape of WAVE_SHAPE_ORDER) {
            expect(t(WAVE_SHAPE_KEYS[shape])).toBeTruthy();
        }
        expect(new Set(WAVE_SHAPE_ORDER).size).toBe(Object.keys(WAVE_SHAPE_KEYS).length);
    });

    test("包络曲线 / 基线模式 / 速率模式的词条键齐全", () => {
        for (const curve of ENVELOPE_CURVE_ORDER)
            expect(t(ENVELOPE_CURVE_KEYS[curve])).toBeTruthy();
        for (const mode of BASELINE_MODE_ORDER) expect(t(BASELINE_MODE_KEYS[mode])).toBeTruthy();
        for (const mode of ["hz", "cycles"] as const) expect(t(RATE_MODE_KEYS[mode])).toBeTruthy();
    });

    test("枚举顺序覆盖了全部取值（漏一个就选不到）", () => {
        expect(new Set(ENVELOPE_CURVE_ORDER)).toEqual(new Set(Object.keys(ENVELOPE_CURVE_KEYS)));
        expect(new Set(BASELINE_MODE_ORDER)).toEqual(new Set(Object.keys(BASELINE_MODE_KEYS)));
        expect(new Set(WAVE_SHAPE_ORDER)).toEqual(new Set(Object.keys(WAVE_SHAPE_KEYS)));
    });
});

describe("builtinIdOf", () => {
    test("解析系统预设 id", () => {
        expect(builtinIdOf("builtin.natural")).toBe("natural");
        expect(builtinIdOf("builtin.straight")).toBe("straight");
    });

    test("用户预设与未知键返回 undefined", () => {
        expect(builtinIdOf("custom_abc")).toBeUndefined();
        expect(builtinIdOf("builtin.does_not_exist")).toBeUndefined();
    });
});

describe("vibratoPresetLabel / Description", () => {
    test("系统预设走词条", () => {
        const natural = SYSTEM_VIBRATO_PRESETS.find(
            (preset) => preset.id === builtinVibratoPresetId("natural"),
        );
        expect(natural).toBeDefined();
        if (!natural) return;
        expect(vibratoPresetLabel(natural, t)).toBe(enUS.vibrato_preset_natural);
        expect(vibratoPresetDescription(natural, t)).toBe(enUS.vibrato_preset_natural_desc);
    });

    test("用户预设用用户输入的名字，且没有说明", () => {
        const custom = sanitizeVibratoPreset({ id: "custom_a", name: "  我的颤音  " });
        expect(vibratoPresetLabel(custom, t)).toBe("我的颤音");
        expect(vibratoPresetDescription(custom, t)).toBeUndefined();
    });
});

describe("vibratoPresetSummary", () => {
    test("频率模式：形状 · 深度 · Hz", () => {
        const preset = sanitizeVibratoPreset({
            id: "custom_a",
            cycle: { kind: "shape", shape: "sine", skew: 0.5 },
            depthCents: 30,
            rateMode: "hz",
            rateHz: 5.5,
        });
        expect(vibratoPresetSummary(preset, t)).toBe("Sine · 30 cents · 5.5 Hz");
    });

    test("周期数模式：换成周期数显示", () => {
        const preset = sanitizeVibratoPreset({
            id: "custom_a",
            cycle: { kind: "shape", shape: "triangle", skew: 0.5 },
            depthCents: 12,
            rateMode: "cycles",
            cycles: 6,
        });
        expect(vibratoPresetSummary(preset, t)).toContain("6 cycles");
        expect(vibratoPresetSummary(preset, t)).toContain("Triangle");
    });

    test("采样表像某个形状时，报那个形状名", () => {
        // 这 8 个点恰好是三角波一个周期。
        const preset = sanitizeVibratoPreset({
            id: "custom_a",
            cycle: { kind: "table", table: [0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5] },
        });
        expect(vibratoPresetSummary(preset, t)).toContain(enUS.vibrato_shape_triangle);
    });

    /*
     * 【为什么值得测】采样表有两个来源（从选区提取、手绘），把两者一律说成"来自选区"
     * 是错的 —— 手绘才是编辑器里的主要操作。这里钉住新契约：标签描述**形状**，都不像
     * 才承认是手绘。
     */
    test("采样表捏成自定义波形时报「手绘」，而不是「来自选区」", () => {
        const preset = sanitizeVibratoPreset({
            id: "custom_a",
            // 一个周期里塞两个正弦 —— 参数式形状里没有这一款。
            cycle: {
                kind: "table",
                table: [0, 0.7, 1, 0.7, 0, -0.7, -1, -0.7, 0, 0.7, 1, 0.7, 0, -0.7, -1, -0.7],
            },
        });
        const summary = vibratoPresetSummary(preset, t);
        expect(summary).toContain(enUS.vibrato_cycle_drawn);
        expect(summary).not.toContain(enUS.vibrato_from_selection);
    });
});

describe("cycleShapeLabelKey", () => {
    test("参数式形状直接给形状名", () => {
        expect(cycleShapeLabelKey({ kind: "shape", shape: "sine", skew: 0.5 })).toBe(
            "vibrato_shape_sine",
        );
        expect(cycleShapeLabelKey({ kind: "shape", shape: "trill", skew: 0.5 })).toBe(
            "vibrato_shape_trill",
        );
    });

    test("采样表按最接近的形状取名", () => {
        expect(
            cycleShapeLabelKey({ kind: "table", table: [0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5] }),
        ).toBe("vibrato_shape_triangle");
    });

    test("都不像时归为手绘（不会硬套一个形状名）", () => {
        expect(
            cycleShapeLabelKey({
                kind: "table",
                table: [0, 0.7, 1, 0.7, 0, -0.7, -1, -0.7, 0, 0.7, 1, 0.7, 0, -0.7, -1, -0.7],
            }),
        ).toBe("vibrato_cycle_drawn");
    });
});

describe("formatNumber", () => {
    test("去掉无意义的小数尾巴", () => {
        expect(formatNumber(30)).toBe("30");
        expect(formatNumber(5.5)).toBe("5.5");
        expect(formatNumber(5.4999)).toBe("5.5");
    });

    test("非有限值显示 0 而不是 NaN", () => {
        expect(formatNumber(Number.NaN)).toBe("0");
        expect(formatNumber(Number.POSITIVE_INFINITY)).toBe("0");
    });
});

describe("深度单位换算", () => {
    test("音高：cents → cents（编辑器里就是分）", () => {
        expect(depthForParam(30, "pitch")).toBeCloseTo(30, 9);
        expect(depthToCents(30, "pitch")).toBeCloseTo(30, 9);
    });

    test("动态：cents ↔ 百分比", () => {
        expect(depthForParam(30, "dyn")).toBeCloseTo(30, 9);
        expect(depthToCents(30, "dyn")).toBeCloseTo(30, 9);
    });

    test("原始值域：cents ↔ 满摆幅百分比（满摆幅 100 分 → 1:1）", () => {
        // 声像的原生量程是 ±1，若按原生单位显示会得到 `0.3` 这种没有量纲感的
        // 裸数字；改成"占满摆幅的百分比"之后，声像、张力、气声一律说成 30%。
        expect(depthForParam(30, "tension")).toBeCloseTo(30, 9);
        expect(depthToCents(30, "tension")).toBeCloseTo(30, 9);
        expect(depthForParam(30, "pan", { min: -1, max: 1 })).toBeCloseTo(30, 9);
        expect(depthToCents(30, "pan", { min: -1, max: 1 })).toBeCloseTo(30, 9);
        expect(depthForParam(30, "breathiness", { min: -10000, max: 10000 })).toBeCloseTo(30, 9);
    });

    test("音级参数：cents ↔ 音级（100 分 = 1 音级）", () => {
        expect(depthForParam(50, "child_pitch_offset_degrees@t1")).toBeCloseTo(0.5, 9);
        expect(depthToCents(0.5, "child_pitch_offset_degrees@t1")).toBeCloseTo(50, 9);
    });

    test("换算可逆（各参数族往返一致）", () => {
        for (const param of ["pitch", "dyn", "volume", "tension", "breathiness", "pan"]) {
            expect(depthToCents(depthForParam(37, param), param)).toBeCloseTo(37, 6);
        }
    });
});

describe("depthUnitLabelKey", () => {
    /*
     * 深度显示值已经按参数换算成原生单位，单位标签必须跟着变 ——
     * 在 dyn 上标 "cents" 就是把百分比说成了音分。
     */
    test("音高与 cents 类参数用分", () => {
        expect(depthUnitLabelKey("pitch")).toBe("vibrato_unit_cents");
        expect(depthUnitLabelKey("child_pitch_offset_cents@t1")).toBe("vibrato_unit_cents");
    });

    test("乘性增益用百分比", () => {
        expect(depthUnitLabelKey("dyn")).toBe("vibrato_unit_percent");
        expect(depthUnitLabelKey("volume")).toBe("vibrato_unit_percent");
        expect(depthUnitLabelKey("breath_gain")).toBe("vibrato_unit_percent");
    });

    test("音级参数用音级", () => {
        expect(depthUnitLabelKey("child_pitch_offset_degrees@t1")).toBe("vibrato_unit_degree");
    });

    test("原始值域参数同样用百分比（深度说的是占满摆幅的几成，不是原生数字）", () => {
        expect(depthUnitLabelKey("hifigan_tension")).toBe("vibrato_unit_percent");
        expect(depthUnitLabelKey("tension")).toBe("vibrato_unit_percent");
        expect(depthUnitLabelKey("breathiness")).toBe("vibrato_unit_percent");
        expect(depthUnitLabelKey("pan")).toBe("vibrato_unit_percent");
    });

    test("返回的词条键都真实存在", () => {
        for (const param of [
            "pitch",
            "dyn",
            "child_pitch_offset_degrees@t1",
            "hifigan_tension",
            "pan",
        ]) {
            const key = depthUnitLabelKey(param);
            expect(key).not.toBeNull();
            if (key) expect(t(key)).toBeTruthy();
        }
    });
});

describe("buildVibratoPreview", () => {
    test("采样点数与请求一致，且都是有限值", () => {
        const preset = sanitizeVibratoPreset({ depthCents: 40 });
        const samples = buildVibratoPreview(preset, { frameCount: 100, framePeriodMs: 5 });
        expect(samples.wave).toHaveLength(100);
        expect(samples.envelope).toHaveLength(100);
        for (const value of samples.wave) expect(Number.isFinite(value)).toBe(true);
    });

    test("单位是 cents：深度 40 的波形峰值接近 ±40", () => {
        const preset = sanitizeVibratoPreset({ depthCents: 40, attackMs: 0, releaseMs: 0 });
        const samples = buildVibratoPreview(preset, { frameCount: 400, framePeriodMs: 5 });
        const peak = Math.max(...samples.wave.map((value) => Math.abs(value)));
        expect(peak).toBeGreaterThan(35);
        expect(peak).toBeLessThanOrEqual(40.001);
    });

    test("包络恒非负，且渐入时开头接近 0", () => {
        const preset = sanitizeVibratoPreset({ depthCents: 40, attackMs: 300 });
        const samples = buildVibratoPreview(preset, { frameCount: 200, framePeriodMs: 5 });
        for (const value of samples.envelope) expect(value).toBeGreaterThanOrEqual(0);
        expect(samples.envelope[0]).toBeLessThan(1);
    });

    test("深度 0 的预设波形全程为 0（预览不会假装有颤音）", () => {
        const preset = sanitizeVibratoPreset({ depthCents: 0 });
        const samples = buildVibratoPreview(preset, { frameCount: 64, framePeriodMs: 5 });
        for (const value of samples.wave) expect(value).toBe(0);
    });

    /*
     * ★ 回归：管理器预览必须跟着预设的 `seed` 字段走。
     *
     * 骰子按钮改的正是这个字段；预览若在渲染时另取种子（曾经兜底为 0），按钮
     * 就"完全不起作用" —— 字段变了，画出来的东西一模一样。
     */
    test("预览跟着预设的 seed 字段变（骰子按钮换的就是它）", () => {
        const base = { depthCents: 40, irregularity: 60, attackMs: 0, releaseMs: 0 };
        const one = buildVibratoPreview(sanitizeVibratoPreset({ ...base, seed: 1 }), {
            frameCount: 200,
            framePeriodMs: 5,
        });
        const two = buildVibratoPreview(sanitizeVibratoPreset({ ...base, seed: 2 }), {
            frameCount: 200,
            framePeriodMs: 5,
        });
        expect(one.wave).not.toEqual(two.wave);
        // 同一个预设反复预览必须逐帧一致（否则编辑时画面会闪）。
        const again = buildVibratoPreview(sanitizeVibratoPreset({ ...base, seed: 1 }), {
            frameCount: 200,
            framePeriodMs: 5,
        });
        expect(again.wave).toEqual(one.wave);
    });

    test("baseline 为 existing 也被强制按 line 预览（无原曲线时才有确定形状）", () => {
        const preset = sanitizeVibratoPreset({ depthCents: 40, baseline: "existing" });
        const samples = buildVibratoPreview(preset, { frameCount: 64, framePeriodMs: 5 });
        const peak = Math.max(...samples.wave.map((value) => Math.abs(value)));
        expect(peak).toBeGreaterThan(1);
    });

    test("极短请求不会崩（至少两点）", () => {
        const samples = buildVibratoPreview(sanitizeVibratoPreset({ depthCents: 20 }), {
            frameCount: 1,
            framePeriodMs: 5,
        });
        expect(samples.wave.length).toBeGreaterThanOrEqual(2);
    });

    test("峰值随深度增大而增大（定标依赖这一单调性）", () => {
        const shallow = buildVibratoPreview(sanitizeVibratoPreset({ depthCents: 10 }));
        const deep = buildVibratoPreview(sanitizeVibratoPreset({ depthCents: 90 }));
        expect(deep.peakCents).toBeGreaterThan(shallow.peakCents);
    });
});

describe("previewScaleCents", () => {
    test("留出 15% 余量并向上取整", () => {
        expect(previewScaleCents(40)).toBe(46);
        expect(previewScaleCents(100)).toBe(115);
    });

    test("下限为 1，避免纵轴除零", () => {
        expect(previewScaleCents(0)).toBe(2);
        expect(previewScaleCents(Number.NaN)).toBe(2);
    });
});

describe("buildAppliedPreview（套用到选区的预览）", () => {
    // 一段音高曲线（半音）：0 → 2 的上行，叠加 30 分正弦颤音。
    const ramp = Array.from({ length: 200 }, (_, i) => (i / 199) * 2);
    const preset = sanitizeVibratoPreset({
        id: "custom_a",
        depthCents: 30,
        rateHz: 5.5,
        attackMs: 0,
        releaseMs: 0,
        baseline: "existing",
    });

    test("原曲线与结果同长，且都围绕中心（均值≈0）", () => {
        const preview = buildAppliedPreview({
            preset,
            original: ramp,
            param: "pitch",
            framePeriodMs: 5,
        });
        expect(preview).not.toBeNull();
        expect(preview!.original.length).toBe(ramp.length);
        expect(preview!.wave.length).toBe(ramp.length);
        const mean = preview!.original.reduce((sum, value) => sum + value, 0) / ramp.length;
        expect(Math.abs(mean)).toBeLessThan(1e-6);
    });

    test("baseline existing：结果 = 原曲线 + 颤音，因此偏离原曲线的幅度约等于深度", () => {
        const preview = buildAppliedPreview({
            preset,
            original: ramp,
            param: "pitch",
            framePeriodMs: 5,
        })!;
        const deviation = preview.wave.map((value, i) => value - preview.original[i]);
        const peak = Math.max(...deviation.map((value) => Math.abs(value)));
        // 30 分深度：音高按分换算，偏离峰值应当在 30 附近（允许不规则度为 0 的解析值）。
        expect(peak).toBeGreaterThan(28);
        expect(peak).toBeLessThan(32);
    });

    test("深度为 0：结果与原曲线重合（直线预设不改动选区）", () => {
        const flat = sanitizeVibratoPreset({ id: "custom_b", depthCents: 0, baseline: "existing" });
        const preview = buildAppliedPreview({
            preset: flat,
            original: ramp,
            param: "pitch",
            framePeriodMs: 5,
        })!;
        for (let i = 0; i < ramp.length; i += 1) {
            expect(preview.wave[i]).toBeCloseTo(preview.original[i], 6);
        }
    });

    test("数据不足两点时返回 null（由调用方显示占位提示）", () => {
        expect(
            buildAppliedPreview({ preset, original: [0], param: "pitch", framePeriodMs: 5 }),
        ).toBeNull();
        expect(
            buildAppliedPreview({ preset, original: [], param: "pitch", framePeriodMs: 5 }),
        ).toBeNull();
    });

    test("非有限的帧值被当作 0，不产生 NaN 曲线", () => {
        const preview = buildAppliedPreview({
            preset,
            original: [0, Number.NaN, 1, 2],
            param: "pitch",
            framePeriodMs: 5,
        })!;
        for (const value of [...preview.wave, ...preview.original]) {
            expect(Number.isFinite(value)).toBe(true);
        }
    });

    test("乘性参数（dyn）也给出可辨的偏离（换算到分后仍围绕中心）", () => {
        const dyn = Array.from({ length: 120 }, () => 80);
        const preview = buildAppliedPreview({
            preset: sanitizeVibratoPreset({ id: "custom_c", depthCents: 30, baseline: "existing" }),
            original: dyn,
            param: "dyn",
            framePeriodMs: 5,
        })!;
        const peak = Math.max(...preview.wave.map((value) => Math.abs(value)));
        expect(peak).toBeGreaterThan(1);
    });
});

describe("fitPreviewRangeCents（一次性拟合纵轴）", () => {
    test("落在阶梯档位上，且比峰值大出余量", () => {
        for (const peak of [0, 3, 5, 12, 30, 40, 70, 100, 150, 300, 700, 1200]) {
            const range = fitPreviewRangeCents(peak);
            expect(PREVIEW_RANGE_LADDER).toContain(range);
            expect(range).toBeGreaterThanOrEqual(peak * 1.15 - 1e-9);
        }
    });

    test("常见深度：波形约占画布六成（不顶格也不趴平）", () => {
        for (const peak of [5, 12, 30, 55, 100]) {
            const range = fitPreviewRangeCents(peak);
            const fill = peak / range;
            expect(fill).toBeGreaterThan(0.4);
            expect(fill).toBeLessThanOrEqual(1);
        }
    });

    test("单调不减（峰值更大绝不会得到更小的量程）", () => {
        let previous = 0;
        for (let peak = 0; peak <= 1200; peak += 7) {
            const range = fitPreviewRangeCents(peak);
            expect(range).toBeGreaterThanOrEqual(previous);
            previous = range;
        }
    });

    test("深度 0 也给一个可见量程（不会退化成除零）", () => {
        expect(fitPreviewRangeCents(0)).toBeGreaterThan(0);
    });

    test("非有限输入回落到最小档位", () => {
        expect(fitPreviewRangeCents(Number.NaN)).toBe(PREVIEW_RANGE_LADDER[0]);
        expect(fitPreviewRangeCents(Number.POSITIVE_INFINITY)).toBe(PREVIEW_RANGE_LADDER[0]);
    });

    test("包络可以远大于 depthCents（渐强 × 不规则度），量程仍跟得上", () => {
        // depthCents 上限 1200，渐强 2× 与不规则度抖动叠加后峰值可到数千。
        const range = fitPreviewRangeCents(3240);
        expect(range).toBeGreaterThanOrEqual(3240 * 1.15 - 1e-9);
        expect(PREVIEW_RANGE_LADDER).toContain(range);
    });
});

describe("直线预设的读数（peakCents）", () => {
    /*
     * 【为什么值得测】`peakCents` 是读数（"±N 分"）的来源，曾经在这里保底为 1，
     * 于是完全平直的颤音线一直显示"±1 分"。需要非零尺度的地方各自兜底，读数应当如实。
     */
    test("深度为 0 的预设峰值为 0", () => {
        const samples = buildVibratoPreview(
            sanitizeVibratoPreset({ id: "custom_a", depthCents: 0 }),
        );
        expect(samples.peakCents).toBe(0);
    });

    test("纵轴定标仍然非零（尺度由定标函数兜底，不靠读数）", () => {
        const samples = buildVibratoPreview(
            sanitizeVibratoPreset({ id: "custom_a", depthCents: 0 }),
        );
        expect(previewScaleCents(samples.peakCents)).toBeGreaterThan(0);
        expect(fitPreviewRangeCents(samples.peakCents)).toBeGreaterThan(0);
    });

    test("有深度的预设峰值等于深度（按参数换算成 cents）", () => {
        const samples = buildVibratoPreview(
            sanitizeVibratoPreset({
                id: "custom_a",
                depthCents: 40,
                attackMs: 0,
                releaseMs: 0,
                irregularity: 0,
            }),
        );
        expect(samples.peakCents).toBeCloseTo(40, 6);
    });
});
