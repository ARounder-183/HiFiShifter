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
    // 一段音高曲线（半音）：C4 → D4 的上行，叠加 30 分正弦颤音。
    //
    // 【为什么从 60 起而不是从 0 起】音高参数里 **0 = 未检测到音高**，不是"音高 0"
    // （见 `vibratoPitch.ts`）。拿 0 当数据写进夹具，测的就是一个不存在的情形。
    const ramp = Array.from({ length: 200 }, (_, i) => 60 + (i / 199) * 2);
    const preset = sanitizeVibratoPreset({
        id: "custom_a",
        depthCents: 30,
        rateHz: 5.5,
        attackMs: 0,
        releaseMs: 0,
        baseline: "existing",
    });

    /*
     * 两条参数线共用一套纵轴。
     *
     * 【为什么必须共用】用户要对比的是"新参数线相对原参数线差多少"。两条线相差两个
     * 数量级（轮廓可以跨十几个半音、颤音只有几十分），各自减掉**同一个**中心之后，
     * 它们才落在同一根标尺上 —— 按「适应」换比例尺时两条一起缩放，而不是只有一条动
     * （用户报过："原参数线不也应该跟着动吗"）。
     */
    test("原线相对共同中心取值（形状保留），结果线与它之差就是颤音", () => {
        const preview = buildAppliedPreview({
            preset,
            original: ramp,
            param: "pitch",
            framePeriodMs: 5,
        })!;
        expect(preview.contour.length).toBe(ramp.length);
        expect(preview.wave.length).toBe(ramp.length);
        expect(preview.envelope.length).toBe(ramp.length);
        // 素材是 60 → 62 的上行，中心 = 均值 61 分音 → 原线围绕 0 摆动 ±100 分。
        const contour = preview.contour.filter(Number.isFinite);
        expect(Math.min(...contour)).toBeCloseTo(-100, 6);
        expect(Math.max(...contour)).toBeCloseTo(100, 6);
        // 结果线 − 原线 = 颤音：两条线在图上只差一点点，这正是要看的对比。
        const diff = preview.wave
            .map((value, index) => value - preview.contour[index])
            .filter(Number.isFinite);
        expect(Math.max(...diff.map(Math.abs))).toBeGreaterThan(28);
        expect(Math.max(...diff.map(Math.abs))).toBeLessThan(32);
    });

    test("读数用的颤音幅度与素材轮廓宽度无关", () => {
        const preview = buildAppliedPreview({
            preset,
            original: ramp,
            param: "pitch",
            framePeriodMs: 5,
        })!;
        // 30 分深度：读数只跟包络有关，不该被素材自身的音高起伏撑大。
        expect(preview.vibratoPeakCents).toBeGreaterThan(28);
        expect(preview.vibratoPeakCents).toBeLessThan(32);
    });

    test("深度为 0：结果线与原线重合（不改动音高）", () => {
        const flat = sanitizeVibratoPreset({ id: "custom_b", depthCents: 0, baseline: "existing" });
        const preview = buildAppliedPreview({
            preset: flat,
            original: ramp,
            param: "pitch",
            framePeriodMs: 5,
        })!;
        for (let i = 0; i < ramp.length; i += 1) {
            expect(preview.wave[i]).toBeCloseTo(preview.contour[i], 6);
        }
        expect(preview.vibratoPeakCents).toBe(0);
    });

    test("数据不足两点时返回 null（由调用方显示占位提示）", () => {
        expect(
            buildAppliedPreview({ preset, original: [0], param: "pitch", framePeriodMs: 5 }),
        ).toBeNull();
        expect(
            buildAppliedPreview({ preset, original: [], param: "pitch", framePeriodMs: 5 }),
        ).toBeNull();
    });

    /*
     * 本组最重要的一条：选区首尾落在气口上时，锚点绝不能取哨兵 0，也不能取到
     * 浊清边界上"低而非零"的过渡帧。
     *
     * 判据不是逐帧的（`> 0` 只挡住第一类），而是**按音符段**：一段连续有声帧要够长
     * （`MIN_NOTE_MS`）且落在音高值域内才算一个音符。见 `vibratoPitch.ts`。
     */
    describe("音高哨兵与音符段", () => {
        /*
         * 报告场景：两头气口（0），紧挨气口还有 5 帧**低而非零**的过渡帧
         * （音高跟踪器在浊清边界给出的 20~40 Hz 低估 → MIDI 3~22），中间是一句真实
         * 音高的唱段（C4 → E4，300ms）。
         *
         * 旧实现取首末帧当锚点（锚到 0），再往前的实现取首个非零帧（锚到 3）——
         * 两者都会把中间真实唱出来的音高拖向那个错值。
         */
        const GAP = 20;
        const BLIP = [3, 8, 15, 20, 22];
        const NOTE_LEN = 60;
        const noteStart = GAP + BLIP.length;
        const noteEnd = noteStart + NOTE_LEN;
        const withGaps = [
            ...new Array<number>(GAP).fill(0),
            ...BLIP,
            ...Array.from({ length: NOTE_LEN }, (_, i) => 60 + (i / (NOTE_LEN - 1)) * 4),
            ...new Array<number>(GAP).fill(0),
        ];

        const line = sanitizeVibratoPreset({
            id: "custom_line",
            depthCents: 0,
            rateHz: 5.5,
            attackMs: 0,
            releaseMs: 0,
            baseline: "line",
        });

        test("边界过渡帧不再把整段拉平：锚点是音符本身的音高", () => {
            const preview = buildAppliedPreview({
                preset: line,
                original: withGaps,
                param: "pitch",
                framePeriodMs: 5,
            })!;
            /*
             * 直线预设把选区拉直成一条线，而这条线锚在**音符两端的音高**上
             * （这一段素材本身就是 60 → 64 的四度上行，见夹具注释）。
             */
            const values: number[] = [];
            for (let i = noteStart; i < noteEnd; i += 1) values.push(preview.wave[i] as number);
            // 一条直线：相邻差恒定。
            const steps = values.slice(1).map((v, i) => v - values[i]);
            expect(Math.max(...steps) - Math.min(...steps), "结果线应当是一条直线").toBeLessThan(
                1e-6,
            );
            // 且它就在音符音高附近、且是**上行**的。
            expect(Math.abs(values[0]), "锚点不得被那 5 帧 3..22 拽走").toBeLessThan(250);
            expect(values[values.length - 1]).toBeGreaterThan(values[0]);
        });

        /*
         * 段内的音高走向必须体现在锚点上 —— 否则四种「摆放方式」里三种会撞在一起。
         *
         * 【报告场景】"只有全程均值和保持现有曲线正常工作，其他和全程均值一模一样"。
         * 根因：锚点原本取**首 / 末音符段的整段中位数**，而一段连奏乐句只有一个音符段
         * → 两个锚点必然相等 → 「起点 → 终点」「起点水平」「终点水平」全部退化成
         * 同一条水平线（= 全程均值）。
         *
         * 这条用"一段 60 → 64 的音符"钉住：四种模式的**终点值**必须互不相同。
         */
        test("段内的走向进得了锚点：四种摆放方式互不相同", () => {
            const note = [
                ...new Array<number>(20).fill(0),
                ...Array.from({ length: 60 }, (_, i) => 60 + (i / 59) * 4),
                ...new Array<number>(20).fill(0),
            ];
            const at = (baseline: "line" | "holdStart" | "holdEnd" | "average") => {
                const preview = buildAppliedPreview({
                    preset: sanitizeVibratoPreset({
                        id: `custom_${baseline}`,
                        depthCents: 0,
                        attackMs: 0,
                        releaseMs: 0,
                        irregularity: 0,
                        baseline,
                    }),
                    original: note,
                    param: "pitch",
                    framePeriodMs: 5,
                })!;
                // 取音符中段的几帧（避开两端的过渡），代表这条线摆在哪。
                return preview.wave[55] as number;
            };
            const line = at("line");
            const holdStart = at("holdStart");
            const holdEnd = at("holdEnd");
            const average = at("average");
            const values = [line, holdStart, holdEnd, average];
            for (let i = 0; i < values.length; i += 1) {
                for (let j = i + 1; j < values.length; j += 1) {
                    expect(
                        Math.abs(values[i] - values[j]),
                        `两种摆放方式不该给出同一个结果（${i} vs ${j}）`,
                    ).toBeGreaterThan(10);
                }
            }
            // 起点水平应当比终点水平低（素材是上行的）。
            expect(holdStart).toBeLessThan(holdEnd);
        });

        /*
         * 跟踪器在音符**内部**给出的异常段（八度跳、快滑）不得在轮廓上打洞。
         *
         * 【报告场景】"对原参数线的识别有问题，在不应该断的地方断了"。根因：画图用的
         * 掩码是 `refineNoteFrames` 的结果 —— 那是"值不值得拿去拟合"的统计判断，
         * 一段残差很大的帧（八度跳 40 帧）会被整段剔掉，轮廓于是断开。
         *
         * 现在画图认**音符帧**（`plan.modulatable`，也就是落盘掩码），断口只留给
         * 真正的气口；异常段照画，但**不参与纵轴定标**（否则 40 分的颤音会被压成
         * 一条直线）。
         */
        test("音符内部的异常段照画（不断开），但不参与纵轴定标", () => {
            const glitch = [
                ...new Array<number>(50).fill(0),
                ...Array.from({ length: 300 }, (_, i) => 60 + Math.sin(i / 6) * 0.3),
                ...Array.from({ length: 40 }, (_, i) => 72 + Math.sin(i / 6) * 0.3),
                ...Array.from({ length: 260 }, (_, i) => 60 + Math.sin(i / 6) * 0.3),
            ];
            const preview = buildAppliedPreview({
                preset: sanitizeVibratoPreset({
                    id: "custom_glitch",
                    depthCents: 40,
                    rateHz: 5.5,
                    attackMs: 0,
                    releaseMs: 0,
                    irregularity: 0,
                }),
                original: glitch,
                param: "pitch",
                framePeriodMs: 5,
            })!;

            // 轮廓从第一个音符帧到最后一个音符帧**连续**（异常段不断开）。
            for (let i = 50; i < 650; i += 1) {
                expect(Number.isFinite(preview.contour[i]), `第 ${i} 帧不该断`).toBe(true);
                expect(Number.isFinite(preview.wave[i]), `第 ${i} 帧不该断`).toBe(true);
            }
            // 但纵轴不认那个八度跳：量程仍按"稳的那部分"来。
            expect(preview.peakCents, "异常段不得把纵轴撑到上千分").toBeLessThan(300);
            // 读数报的是颤音自身幅度，与纵轴的取舍无关。
            expect(preview.vibratoPeakCents).toBeGreaterThan(20);
        });

        test("不受颤音影响的帧画成断口（NaN），且不参与纵轴", () => {
            const preview = buildAppliedPreview({
                preset: line,
                original: withGaps,
                param: "pitch",
                framePeriodMs: 5,
            })!;
            // 气口与边界过渡帧都要断开 —— 后者是"低而非零"，只判 == 0 会漏掉。
            for (let i = 0; i < noteStart; i += 1) {
                expect(Number.isNaN(preview.wave[i])).toBe(true);
                expect(Number.isNaN(preview.envelope[i])).toBe(true);
            }
            for (let i = noteEnd; i < withGaps.length; i += 1) {
                expect(Number.isNaN(preview.wave[i])).toBe(true);
            }
            // 音符帧有值：直线预设（深度 0）下结果就是那条线本身 —— 一条连接音符两端
            // 音高的**直线**（不是"全段一个常数"：那样只有把锚点取成整段中位数才会出现，
            // 而那正好抹掉了段内本来有的音高走向）。
            const lineValues = preview.wave.slice(noteStart, noteEnd).filter(Number.isFinite);
            expect(lineValues.length, "音符帧应当都画出来").toBe(noteEnd - noteStart);
            const lineSteps = lineValues.slice(1).map((v, i) => v - lineValues[i]);
            expect(Math.max(...lineSteps) - Math.min(...lineSteps)).toBeLessThan(1e-6);
            // 纵轴只按画出来的值拟合：气口与过渡帧是断口，不参与。
            expect(preview.peakCents).toBeGreaterThan(0);
            expect(preview.peakCents).toBeLessThan(500);
        });

        test("不受颤音影响的帧，包络也是断口（不在气口上画出一条颤音带）", () => {
            const deep = sanitizeVibratoPreset({
                id: "custom_deep",
                depthCents: 40,
                rateHz: 5.5,
                attackMs: 0,
                releaseMs: 0,
                baseline: "existing",
            });
            const preview = buildAppliedPreview({
                preset: deep,
                original: withGaps,
                param: "pitch",
                framePeriodMs: 5,
            })!;
            expect(Number.isNaN(preview.envelope[0])).toBe(true);
            expect(Number.isNaN(preview.envelope[GAP])).toBe(true);
            expect(Number.isFinite(preview.envelope[noteStart])).toBe(true);
        });

        test("没有够长的音符段：没有可调制的对象，返回 null", () => {
            // 整段未检测（含非有限值混排）。
            expect(
                buildAppliedPreview({
                    preset: line,
                    original: [0, 0, 0, 0],
                    param: "pitch",
                    framePeriodMs: 5,
                }),
            ).toBeNull();
            expect(
                buildAppliedPreview({
                    preset: line,
                    original: [Number.NaN, 0, Number.NaN],
                    param: "pitch",
                    framePeriodMs: 5,
                }),
            ).toBeNull();
            // 有音高但只有 3 帧（15ms）—— 够不上一个音符。
            expect(
                buildAppliedPreview({
                    preset: line,
                    original: [
                        ...new Array<number>(10).fill(0),
                        60,
                        61,
                        62,
                        ...new Array<number>(10).fill(0),
                    ],
                    param: "pitch",
                    framePeriodMs: 5,
                }),
            ).toBeNull();
        });

        test("非哨兵参数不受影响：0 仍是合法值，也不产生断口", () => {
            const values = [0, Number.NaN, 1, 2, 3];
            const preview = buildAppliedPreview({
                preset,
                original: values,
                param: "volume",
                framePeriodMs: 5,
            })!;
            for (const value of preview.wave) {
                expect(Number.isFinite(value)).toBe(true);
            }
        });
    });

    /*
     * 纵轴定标：只能跟着**颤音自己的幅度**走，不能跟着音高轮廓的宽度走。
     *
     * 【这条锁的是报告过两次的缺陷】纵轴按峰值自适应，所以**凡是参与峰值的东西都会
     * 决定纵轴**。两次都栽在这上面：
     * 1. 早先画的是"相对选区均值的绝对音高"，纵轴得容下整条轮廓 —— 一段 40→60 的
     *    过渡缓升（跟踪器在浊清边界的爬升）就把峰值撑到 1840 分；
     * 2. 改成"相对基线的偏移"后，**参考线**（原曲线相对基线的偏移）仍在峰值里 ——
     *    对 `line` / `hold*` / `average` 基线，素材可以离基线几千分（默认预设「直线」
     *    正是 `line`），45 分的颤音只剩 1% 的画布高度。
     *
     * 现在参与峰值的只有画出来的颤音偏移与包络，两者上界都是深度 —— 于是纵轴恒等于
     * 颤音幅度，与素材轮廓多宽、基线模式是哪种都无关。
     */
    describe("纵轴按颤音幅度定标", () => {
        const deep = sanitizeVibratoPreset({
            id: "custom_axis",
            depthCents: 40,
            rateHz: 5.5,
            attackMs: 0,
            releaseMs: 0,
            baseline: "existing",
        });
        const finite = (values: number[]) => values.filter(Number.isFinite);

        /** 稳定长音：没有轮廓可言。 */
        const steadyNote = [
            ...new Array<number>(100).fill(0),
            ...new Array<number>(300).fill(60),
            ...new Array<number>(100).fill(0),
        ];
        /** 同样的长音，但前面挂一段 40→60 的缓升（帧数与 `steadyNote` 一致）。 */
        const rampedNote = [
            ...new Array<number>(100).fill(0),
            ...Array.from({ length: 60 }, (_, i) => 40 + (i / 59) * 20),
            ...new Array<number>(240).fill(60),
            ...new Array<number>(100).fill(0),
        ];

        /*
         * 纵轴按**画出来的全部**拟合（两条线 + 包络带），因此素材轮廓越宽、纵轴越大；
         * 而"这个预设会摆多少"单独由 `vibratoPeakCents` 表达 —— 它与轮廓宽度无关。
         * 这正是"读得出颤音大小"的落点（读数用的就是它）。
         */
        test("读数用的颤音幅度与素材轮廓宽度无关", () => {
            const a = buildAppliedPreview({
                preset: deep,
                original: steadyNote,
                param: "pitch",
                framePeriodMs: 5,
            })!;
            const b = buildAppliedPreview({
                preset: deep,
                original: rampedNote,
                param: "pitch",
                framePeriodMs: 5,
            })!;
            for (const preview of [a, b]) {
                expect(preview.vibratoPeakCents).toBeGreaterThan(38);
                expect(preview.vibratoPeakCents).toBeLessThan(42);
            }
            // 两种素材的颤音幅度完全一致 —— 轮廓的有无不影响它。
            expect(b.vibratoPeakCents).toBeCloseTo(a.vibratoPeakCents, 9);
        });

        /*
         * 默认预设就是「直线」（列表首位），它的基线是 `line`：一条从首音符到末音符
         * 的直线。纵轴要同时容得下那条直线与素材原线，而读数只报颤音自己的幅度。
         */
        test("line 基线（默认「直线」预设）下，纵轴容得下两条线，颤音幅度仍等于深度", () => {
            const lineWithDepth = sanitizeVibratoPreset({
                id: "custom_line_depth",
                depthCents: 45,
                rateHz: 5.5,
                attackMs: 0,
                releaseMs: 0,
                baseline: "line",
            });
            const preview = buildAppliedPreview({
                preset: lineWithDepth,
                original: rampedNote,
                param: "pitch",
                framePeriodMs: 5,
            })!;
            // 结果线与原线都在纵轴之内（否则会被画布裁掉）。
            const drawn = [preview.wave, preview.contour]
                .flatMap((series) => finite(series))
                .map(Math.abs);
            expect(preview.peakCents).toBeGreaterThanOrEqual(Math.max(...drawn));
            // 读数只跟颤音有关。
            expect(preview.vibratoPeakCents).toBeGreaterThan(43);
            expect(preview.vibratoPeakCents).toBeLessThan(47);
        });

        /*
         * 两条线共用一套标尺：原线保留素材的形状，结果线 = 基线 + 颤音，两者之差就是
         * "这个预设改动了多少"。按「适应」换比例尺时两条一起缩放 —— 这是"新参数线相对
         * 原参数线差多少"读得出来的前提。
         */
        describe("两条线共用一套标尺", () => {
            test("原线保留素材的音高走向；结果线与它之差就是颤音", () => {
                const preview = buildAppliedPreview({
                    preset: deep,
                    original: rampedNote,
                    param: "pitch",
                    framePeriodMs: 5,
                })!;
                // 素材是 40→60 的缓升 + 稳定音：原线把这段走向完整带过来（千分级）。
                // 【注】300ms 的缓升属于**真实音高运动**，不是要剔掉的过渡段 ——
                // 只有比局部趋势快一个数量级的滑音才算过渡帧（见 `refineNoteFrames`）。
                const contour = finite(preview.contour);
                expect(Math.max(...contour) - Math.min(...contour)).toBeGreaterThan(1500);
                // 结果线 − 原线 = 颤音（`existing` 基线下基线就是原线）。
                const diff = preview.wave
                    .map((value, index) => value - preview.contour[index])
                    .filter(Number.isFinite);
                expect(Math.max(...diff.map(Math.abs))).toBeCloseTo(preview.vibratoPeakCents, 6);
            });

            test("断口两条线一起断（气口与过渡帧都不画）", () => {
                const preview = buildAppliedPreview({
                    preset: deep,
                    original: rampedNote,
                    param: "pitch",
                    framePeriodMs: 5,
                })!;
                for (let i = 0; i < 100; i += 1) {
                    expect(Number.isNaN(preview.contour[i])).toBe(true);
                    expect(Number.isNaN(preview.wave[i])).toBe(true);
                }
                expect(Number.isFinite(preview.contour[150])).toBe(true);
                expect(Number.isFinite(preview.wave[150])).toBe(true);
            });

            /*
             * "两条线共用标尺"最直观的证据：`line` 基线把结果线搬成一条直线，而原线
             * 保留素材的起伏 —— 同一个坐标系里一眼就能看出预设做了什么。
             *
             * 【素材为什么是拱形】`line` 的含义是"起点音高 → 终点音高"，所以**单调
             * 上行**的素材被它拉直后与素材本身重合（那是对的行为，但演示不出对比）。
             * 拱形（起止同高、中间隆起）才看得出"预设把起伏抹掉了"。
             */
            test("直线基线：结果线被搬成直线，原线保留起伏", () => {
                const straight = sanitizeVibratoPreset({
                    id: "custom_straight",
                    depthCents: 0,
                    rateHz: 5.5,
                    attackMs: 0,
                    releaseMs: 0,
                    irregularity: 0,
                    baseline: "line",
                });
                // 一段**拱形**素材：起止同高（都是 60）、中间隆起 2 个半音。
                // 它必须是一段"音符"（快滑音会被判成过渡段），所以用缓慢的正弦拱。
                const phrase = Array.from(
                    { length: 300 },
                    (_, i) => 60 + Math.sin((Math.PI * i) / 299) * 2,
                );
                const preview = buildAppliedPreview({
                    preset: straight,
                    original: phrase,
                    param: "pitch",
                    framePeriodMs: 5,
                })!;
                const result = finite(preview.wave);
                const contour = finite(preview.contour);
                // 结果线：直线预设把它搬平（两端锚点同高 → 一条水平线；留几分容差，
                // 两端锚点是各自 100ms 窗口的中位数，不会逐位相等）。
                expect(Math.max(...result) - Math.min(...result)).toBeLessThan(5);
                // 原线：素材的拱形完整保留（相对共同中心是 0..+200 分）。
                // 拱顶落在两帧之间，峰值取到 199.998 分 —— 容差给到整分。
                expect(Math.max(...contour) - Math.min(...contour)).toBeCloseTo(200, 0);
            });
        });
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
