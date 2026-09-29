import { describe, expect, test } from "vitest";

import { CYCLE_TABLE_DEFAULT_LEN, CYCLE_TABLE_MIN_LEN } from "./vibratoCycle";
import {
    createVibratoPresetId,
    dedupeVibratoPresets,
    DEFAULT_VIBRATO_PRESET,
    duplicateVibratoPreset,
    isBuiltinVibratoPresetId,
    sanitizeCycleSource,
    sanitizeVibratoPreset,
    VIBRATO_LIMITS,
} from "./vibratoPresets";

describe("createVibratoPresetId", () => {
    test("带 custom_ 前缀且每次都不同", () => {
        const a = createVibratoPresetId();
        const b = createVibratoPresetId();
        expect(a.startsWith("custom_")).toBe(true);
        expect(a).not.toBe(b);
    });
});

describe("isBuiltinVibratoPresetId", () => {
    test("按前缀判定，而不是按 builtin 字段", () => {
        expect(isBuiltinVibratoPresetId("builtin.natural")).toBe(true);
        expect(isBuiltinVibratoPresetId("custom_abc")).toBe(false);
        // 字段可以被手改，前缀不能 —— 手改的 builtin:true 不会伪造出系统预设。
        expect(isBuiltinVibratoPresetId("custom_builtin")).toBe(false);
    });
});

describe("sanitizeVibratoPreset", () => {
    test("空输入回退到出厂默认值并生成 id", () => {
        const preset = sanitizeVibratoPreset(null);
        expect(preset.id.startsWith("custom_")).toBe(true);
        expect(preset.builtin).toBe(false);
        expect(preset.depthCents).toBe(DEFAULT_VIBRATO_PRESET.depthCents);
        expect(preset.rateHz).toBe(DEFAULT_VIBRATO_PRESET.rateHz);
        expect(preset.baseline).toBe(DEFAULT_VIBRATO_PRESET.baseline);
    });

    test("builtin 前缀的 id 会被认成系统预设", () => {
        const preset = sanitizeVibratoPreset({ id: "builtin.natural", builtin: true });
        expect(preset.builtin).toBe(true);
        // 即便传了 builtin:false，前缀说了算。
        expect(sanitizeVibratoPreset({ id: "builtin.deep", builtin: false }).builtin).toBe(true);
    });

    test("名字去空白；空名字保持空（由 UI 决定兜底显示）", () => {
        expect(sanitizeVibratoPreset({ name: "  我的颤音  " }).name).toBe("我的颤音");
        expect(sanitizeVibratoPreset({ name: "   " }).name).toBe("");
    });

    test("数值越界被钳进合法区间", () => {
        const preset = sanitizeVibratoPreset({
            depthCents: 99_999,
            rateHz: 0,
            cycles: -5,
            attackMs: -100,
            releaseMs: 1e9,
            irregularity: 500,
            biasCents: -9999,
            blend: 500,
        });
        expect(preset.depthCents).toBe(VIBRATO_LIMITS.depthCents.max);
        expect(preset.rateHz).toBe(VIBRATO_LIMITS.rateHz.min);
        expect(preset.cycles).toBe(VIBRATO_LIMITS.cycles.min);
        expect(preset.attackMs).toBe(VIBRATO_LIMITS.attackMs.min);
        expect(preset.releaseMs).toBe(VIBRATO_LIMITS.releaseMs.max);
        expect(preset.irregularity).toBe(VIBRATO_LIMITS.irregularity.max);
        expect(preset.biasCents).toBe(VIBRATO_LIMITS.biasCents.min);
        expect(preset.blend).toBe(VIBRATO_LIMITS.blend.max);
    });

    test("非有限数值回退到默认而不是变成 NaN", () => {
        const preset = sanitizeVibratoPreset({
            depthCents: Number.NaN,
            rateHz: Number.POSITIVE_INFINITY,
            attackMs: Number.NaN,
        });
        expect(preset.depthCents).toBe(DEFAULT_VIBRATO_PRESET.depthCents);
        expect(preset.rateHz).toBe(DEFAULT_VIBRATO_PRESET.rateHz);
        expect(preset.attackMs).toBe(DEFAULT_VIBRATO_PRESET.attackMs);
    });

    test("非法枚举值回退到默认", () => {
        const preset = sanitizeVibratoPreset({
            rateMode: "wobble" as never,
            baseline: "diagonal" as never,
            attackCurve: "bounce" as never,
            releaseCurve: "bounce" as never,
        });
        expect(preset.rateMode).toBe(DEFAULT_VIBRATO_PRESET.rateMode);
        expect(preset.baseline).toBe(DEFAULT_VIBRATO_PRESET.baseline);
        expect(preset.attackCurve).toBe(DEFAULT_VIBRATO_PRESET.attackCurve);
    });

    test("相位按圈取模，360 与 0 等价", () => {
        expect(sanitizeVibratoPreset({ startPhaseDeg: 360 }).startPhaseDeg).toBeCloseTo(0, 9);
        expect(sanitizeVibratoPreset({ startPhaseDeg: 450 }).startPhaseDeg).toBeCloseTo(90, 9);
        expect(sanitizeVibratoPreset({ startPhaseDeg: -90 }).startPhaseDeg).toBeCloseTo(270, 9);
    });

    test("渐强倍率各自独立钳制", () => {
        const preset = sanitizeVibratoPreset({ depthRamp: { start: -1, end: 99 } as never });
        expect(preset.depthRamp.start).toBe(VIBRATO_LIMITS.depthRamp.min);
        expect(preset.depthRamp.end).toBe(VIBRATO_LIMITS.depthRamp.max);
    });

    test("缺失的 depthRamp 用默认而不是抛异常", () => {
        expect(sanitizeVibratoPreset({ depthRamp: undefined }).depthRamp).toEqual(
            DEFAULT_VIBRATO_PRESET.depthRamp,
        );
    });

    test("布尔字段缺失时是 false", () => {
        expect(sanitizeVibratoPreset({}).alignCycles).toBe(false);
        expect(sanitizeVibratoPreset({ alignCycles: true }).alignCycles).toBe(true);
    });

    test("规整是幂等的", () => {
        const once = sanitizeVibratoPreset({ depthCents: 44.44, rateHz: 6.66, name: "x" });
        expect(sanitizeVibratoPreset(once)).toEqual(once);
    });
});

describe("sanitizeCycleSource", () => {
    test("合法 shape 源原样保留", () => {
        expect(sanitizeCycleSource({ kind: "shape", shape: "triangle", skew: 0.3 })).toEqual({
            kind: "shape",
            shape: "triangle",
            skew: 0.3,
        });
    });

    test("未知形状回退到正弦", () => {
        expect(sanitizeCycleSource({ kind: "shape", shape: "wobble" })).toEqual({
            kind: "shape",
            shape: "sine",
            skew: 0.5,
        });
    });

    test("偏斜被钳进 (0,1) 开区间，避免退化成方波或纯平", () => {
        expect(
            (sanitizeCycleSource({ kind: "shape", shape: "triangle", skew: 0 }) as { skew: number })
                .skew,
        ).toBe(0.02);
        expect(
            (sanitizeCycleSource({ kind: "shape", shape: "triangle", skew: 1 }) as { skew: number })
                .skew,
        ).toBe(0.98);
    });

    test("非法输入回退到默认正弦源", () => {
        expect(sanitizeCycleSource(null)).toEqual(DEFAULT_VIBRATO_PRESET.cycle);
        expect(sanitizeCycleSource({ kind: "bogus" })).toEqual(DEFAULT_VIBRATO_PRESET.cycle);
    });

    test("table 源：过短的表回退成完整正弦表", () => {
        const source = sanitizeCycleSource({ kind: "table", table: [0, 1] });
        expect(source.kind).toBe("table");
        if (source.kind === "table") expect(source.table).toHaveLength(CYCLE_TABLE_DEFAULT_LEN);
    });

    test("table 源：合法的表被保留并钳制", () => {
        const table = [0, 5, -5, 0, 0, 0, 0, 0];
        const source = sanitizeCycleSource({ kind: "table", table });
        expect(source.kind).toBe("table");
        if (source.kind === "table") {
            expect(source.table).toHaveLength(CYCLE_TABLE_MIN_LEN);
            expect(Math.max(...source.table)).toBe(1);
            expect(Math.min(...source.table)).toBe(-1);
        }
    });
});

describe("duplicateVibratoPreset", () => {
    test("派生出新的用户预设，不复用系统 id", () => {
        const source = sanitizeVibratoPreset({
            id: "builtin.natural",
            builtin: true,
            name: "Natural",
        });
        const copy = duplicateVibratoPreset(source);
        expect(copy.id).not.toBe(source.id);
        expect(isBuiltinVibratoPresetId(copy.id)).toBe(false);
        expect(copy.builtin).toBe(false);
        expect(copy.name).toBe("Natural 2");
    });

    test("可指定名称", () => {
        const copy = duplicateVibratoPreset(sanitizeVibratoPreset({ name: "A" }), "我的 A");
        expect(copy.name).toBe("我的 A");
    });

    test("采样式波形被深拷贝：改副本不影响原型", () => {
        const source = sanitizeVibratoPreset({
            cycle: { kind: "table", table: [0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5] },
        });
        const copy = duplicateVibratoPreset(source);
        if (source.cycle.kind === "table" && copy.cycle.kind === "table") {
            copy.cycle.table[0] = 0.9;
            expect(source.cycle.table[0]).toBe(0);
        } else {
            throw new Error("周期来源应为 table");
        }
    });

    test("渐强倍率被深拷贝：改副本不影响原型", () => {
        const source = sanitizeVibratoPreset({ depthRamp: { start: 0.3, end: 0.9 } });
        const copy = duplicateVibratoPreset(source);
        copy.depthRamp.start = 1;
        expect(source.depthRamp.start).toBeCloseTo(0.3, 9);
    });

    test("参数逐字段相等（除 id / builtin / name）", () => {
        const source = sanitizeVibratoPreset({ depthCents: 42, rateHz: 3.3, irregularity: 12 });
        const copy = duplicateVibratoPreset(source);
        expect(copy.depthCents).toBe(source.depthCents);
        expect(copy.rateHz).toBe(source.rateHz);
        expect(copy.irregularity).toBe(source.irregularity);
        expect(copy.cycle).toEqual(source.cycle);
    });
});

describe("dedupeVibratoPresets", () => {
    test("同 id 保留后者（后写覆盖先写）", () => {
        const list = dedupeVibratoPresets([
            sanitizeVibratoPreset({ id: "custom_a", name: "旧" }),
            sanitizeVibratoPreset({ id: "custom_b", name: "B" }),
            sanitizeVibratoPreset({ id: "custom_a", name: "新" }),
        ]);
        expect(list).toHaveLength(2);
        expect(list.find((preset) => preset.id === "custom_a")?.name).toBe("新");
    });

    test("保持首次出现的顺序", () => {
        const list = dedupeVibratoPresets([
            sanitizeVibratoPreset({ id: "custom_b" }),
            sanitizeVibratoPreset({ id: "custom_a" }),
        ]);
        expect(list.map((preset) => preset.id)).toEqual(["custom_b", "custom_a"]);
    });
});
