import { describe, expect, test } from "vitest";

import type { Keybinding } from "../../../features/keybindings/types";
import { sanitizeVibratoPreset } from "../../../features/vibrato/vibratoPresets";
import {
    buildDragVibratoCurve,
    computeVibratoDragAdjustment,
    createDragWorking,
    depthStepCentsFor,
    resolveVibratoDragKeyboardAdjustment,
    resolveVibratoPresetSwitch,
    switchDragPreset,
} from "./vibratoDragAdjust";

const preset = (overrides: Parameters<typeof sanitizeVibratoPreset>[0] = {}) =>
    sanitizeVibratoPreset({ id: "custom_test", ...overrides });

const kb = (overrides: Partial<Keybinding> = {}): Keybinding => ({ key: "a", ...overrides });

describe("createDragWorking", () => {
    test("没有 lastVibrato* 时用预设自带值", () => {
        const p = preset({ depthCents: 42, rateHz: 6.5 });
        const working = createDragWorking(p, null, null);
        expect(working.depthCents).toBe(42);
        expect(working.rateHz).toBe(6.5);
        expect(working.preset.id).toBe(p.id);
        expect(working.adjusted).toBe(false);
    });

    test("有 lastVibrato* 时优先续上次停下的位置", () => {
        const working = createDragWorking(preset({ depthCents: 42, rateHz: 6.5 }), 18, 3.2);
        expect(working.depthCents).toBe(18);
        expect(working.rateHz).toBe(3.2);
    });

    test("非有限的 lastVibrato* 不污染工作副本", () => {
        const working = createDragWorking(preset({ depthCents: 30 }), Number.NaN, Number.NaN);
        expect(working.depthCents).toBe(30);
    });
});

describe("switchDragPreset", () => {
    test("深度与速率整体换成新预设的值（不是叠加）", () => {
        const before = { ...createDragWorking(preset({ depthCents: 40 }), 90, 9), adjusted: true };
        const after = switchDragPreset(preset({ id: "builtin.deep", depthCents: 55, rateHz: 4.5 }));
        expect(after.depthCents).toBe(55);
        expect(after.rateHz).toBe(4.5);
        expect(after.adjusted).toBe(false);
        // 旧的调整不该泄漏过来
        expect(after.depthCents).not.toBe(before.depthCents);
    });
});

describe("depthStepCentsFor", () => {
    test("音高：24 分/格（与历史手感一致）", () => {
        expect(depthStepCentsFor("pitch")).toBe(24);
    });

    test("乘性增益：1 分 = 1% / 格", () => {
        expect(depthStepCentsFor("dyn")).toBe(1);
        expect(depthStepCentsFor("volume")).toBe(1);
        expect(depthStepCentsFor("breath_gain")).toBe(1);
    });

    test("cents 类参数：按值域的 1/200，下限 1 分", () => {
        // 子轨音分偏移值域 ±2400 → 4800 分 / 200 = 24 分
        expect(depthStepCentsFor("child_pitch_offset_cents@t1")).toBeCloseTo(24, 9);
        // 极度受限的值域不会退化成 0
        expect(depthStepCentsFor("formant_shift_cents")).toBeGreaterThanOrEqual(1);
    });

    test("原始值域参数：半量程的 1/200，恒为 0.5 分", () => {
        expect(depthStepCentsFor("tension")).toBe(0.5);
        expect(depthStepCentsFor("breathiness")).toBe(0.5);
    });
});

describe("computeVibratoDragAdjustment", () => {
    const base = { editParam: "pitch", depthCents: 30, rateHz: 5.5 };

    test("深度是加性的", () => {
        const up = computeVibratoDragAdjustment({
            ...base,
            target: "depth",
            direction: 1,
            steps: 1,
            fineScale: 1,
        });
        expect(up.depthCents).toBeCloseTo(54, 9);
        const down = computeVibratoDragAdjustment({
            ...base,
            target: "depth",
            direction: -1,
            steps: 1,
            fineScale: 1,
        });
        expect(down.depthCents).toBeCloseTo(6, 9);
    });

    test("深度下限为 0（可用它调成直线）", () => {
        const next = computeVibratoDragAdjustment({
            ...base,
            depthCents: 5,
            target: "depth",
            direction: -1,
            steps: 10,
            fineScale: 1,
        });
        expect(next.depthCents).toBe(0);
    });

    test("精细修饰键缩放深度步长", () => {
        const coarse = computeVibratoDragAdjustment({
            ...base,
            target: "depth",
            direction: 1,
            steps: 1,
            fineScale: 1,
        });
        const fine = computeVibratoDragAdjustment({
            ...base,
            target: "depth",
            direction: 1,
            steps: 1,
            fineScale: 0.2,
        });
        expect(fine.depthCents).toBeLessThan(coarse.depthCents);
        expect(fine.depthCents).toBeCloseTo(30 + 24 * 0.2, 9);
    });

    test("速率是几何的（等比），保证高低速端手感一致", () => {
        const up = computeVibratoDragAdjustment({
            ...base,
            target: "rate",
            direction: 1,
            steps: 1,
            fineScale: 1,
        });
        expect(up.rateHz).toBeCloseTo(5.5 * 1.1, 9);
        const down = computeVibratoDragAdjustment({
            ...base,
            target: "rate",
            direction: -1,
            steps: 1,
            fineScale: 1,
        });
        expect(down.rateHz).toBeCloseTo(5.5 / 1.1, 9);
    });

    test("速率被钳在预设的合法区间内", () => {
        const high = computeVibratoDragAdjustment({
            ...base,
            rateHz: 19,
            target: "rate",
            direction: 1,
            steps: 50,
            fineScale: 1,
        });
        expect(high.rateHz).toBeLessThanOrEqual(20);
        const low = computeVibratoDragAdjustment({
            ...base,
            rateHz: 0.2,
            target: "rate",
            direction: -1,
            steps: 50,
            fineScale: 1,
        });
        expect(low.rateHz).toBeGreaterThanOrEqual(0.1);
    });

    test("非有限的 fineScale 回退到 1", () => {
        const next = computeVibratoDragAdjustment({
            ...base,
            target: "depth",
            direction: 1,
            steps: 1,
            fineScale: Number.NaN,
        });
        expect(next.depthCents).toBeCloseTo(54, 9);
    });
});

describe("resolveVibratoDragKeyboardAdjustment", () => {
    const bindings = {
        amplitudeIncrease: kb({ key: "arrowup" }),
        amplitudeDecrease: kb({ key: "arrowdown" }),
        frequencyIncrease: kb({ key: "arrowleft" }),
        frequencyDecrease: kb({ key: "arrowright" }),
    };
    // `matchesKeybinding` 会逐项比较修饰键，因此夹具必须给出全部修饰键字段
    // （缺字段时 `undefined !== false`，会把所有绑定都判成不匹配）。
    const event = (key: string) =>
        ({
            key,
            code: key,
            ctrlKey: false,
            shiftKey: false,
            altKey: false,
            metaKey: false,
        }) as KeyboardEvent;

    test("四个方向各自映射到深度 / 速率", () => {
        expect(resolveVibratoDragKeyboardAdjustment(event("arrowup"), bindings)).toEqual({
            target: "depth",
            direction: 1,
        });
        expect(resolveVibratoDragKeyboardAdjustment(event("arrowdown"), bindings)).toEqual({
            target: "depth",
            direction: -1,
        });
        expect(resolveVibratoDragKeyboardAdjustment(event("arrowleft"), bindings)).toEqual({
            target: "rate",
            direction: 1,
        });
        expect(resolveVibratoDragKeyboardAdjustment(event("arrowright"), bindings)).toEqual({
            target: "rate",
            direction: -1,
        });
    });

    test("未绑定的按键返回 null", () => {
        expect(resolveVibratoDragKeyboardAdjustment(event("q"), bindings)).toBeNull();
    });

    test("modifierOnly 绑定一律不参与（避免把纯修饰键当成调参）", () => {
        const modifierOnly = {
            ...bindings,
            amplitudeIncrease: kb({ key: "alt", modifierOnly: true, alt: true }),
        };
        expect(resolveVibratoDragKeyboardAdjustment(event("alt"), modifierOnly)).toBeNull();
    });
});

describe("resolveVibratoPresetSwitch", () => {
    const event = (key: string, extra: Partial<KeyboardEvent> = {}) =>
        ({
            key,
            code: key,
            ctrlKey: false,
            shiftKey: false,
            altKey: false,
            ...extra,
        }) as KeyboardEvent;

    test("默认绑定：`,` 上一个、`.` 下一个", () => {
        const prev = kb({ key: "," });
        const next = kb({ key: "." });
        expect(resolveVibratoPresetSwitch(event(","), prev, next)).toBe(-1);
        expect(resolveVibratoPresetSwitch(event("."), prev, next)).toBe(1);
    });

    test("未命中返回 null", () => {
        expect(
            resolveVibratoPresetSwitch(event("q"), kb({ key: "," }), kb({ key: "." })),
        ).toBeNull();
    });

    test("两边都未绑定时不消费按键", () => {
        const none = kb({ key: "__none__" });
        expect(resolveVibratoPresetSwitch(event(","), none, none)).toBeNull();
    });

    test("只有一侧绑定也能工作", () => {
        const none = kb({ key: "__none__" });
        expect(resolveVibratoPresetSwitch(event("."), none, kb({ key: "." }))).toBe(1);
        expect(resolveVibratoPresetSwitch(event(","), kb({ key: "," }), none)).toBe(-1);
    });

    test("带修饰键的绑定要求修饰键同时按下", () => {
        const next = kb({ key: ".", shift: true });
        expect(resolveVibratoPresetSwitch(event("."), kb({ key: "x" }), next)).toBeNull();
        expect(
            resolveVibratoPresetSwitch(event(".", { shiftKey: true }), kb({ key: "x" }), next),
        ).toBe(1);
    });
});

describe("buildDragVibratoCurve", () => {
    test("工作副本的深度 / 速率覆盖预设自身的值", () => {
        const p = preset({ depthCents: 100, rateHz: 5, attackMs: 0, releaseMs: 0 });
        const working = { preset: p, depthCents: 10, rateHz: 5, adjusted: true };
        const shallow = buildDragVibratoCurve({
            working,
            startFrame: 0,
            startValue: 60,
            endFrame: 200,
            endValue: 60,
            param: "pitch",
            framePeriodMs: 5,
        });
        const deepest = Math.max(...shallow.dense.map((v) => Math.abs(v - 60)));
        // 深度被覆盖成 10 分：幅值应当是 ±0.1 半音，而不是预设的 ±1。
        expect(deepest).toBeLessThan(0.11);
        expect(deepest).toBeGreaterThan(0.05);
    });

    test("拖拽期间一律按 Hz（周期数模式不参与）", () => {
        // 预设是"整段 2 个周期"，但拖拽只给了 0.2 秒；若按周期数均分，
        // 会得到 10 Hz。按 Hz 则应当是设定值附近。
        const p = preset({ rateMode: "cycles", cycles: 2, rateHz: 5, attackMs: 0, releaseMs: 0 });
        const working = { preset: p, depthCents: 50, rateHz: 5, adjusted: false };
        const built = buildDragVibratoCurve({
            working,
            startFrame: 0,
            startValue: 60,
            endFrame: 40,
            endValue: 60,
            param: "pitch",
            framePeriodMs: 5,
        });
        let crossings = 0;
        for (let i = 1; i < built.dense.length; i += 1) {
            if (built.dense[i - 1] < 60 && built.dense[i] >= 60) crossings += 1;
        }
        // 0.2 秒 × 5 Hz ≈ 1 个周期（而不是 2 个）。
        expect(crossings).toBeLessThanOrEqual(2);
    });

    test("吸附回调作用于合成后的值（保留量化画线）", () => {
        const working = {
            preset: preset({ depthCents: 37, attackMs: 0, releaseMs: 0 }),
            depthCents: 37,
            rateHz: 5,
            adjusted: false,
        };
        const built = buildDragVibratoCurve({
            working,
            startFrame: 0,
            startValue: 60,
            endFrame: 100,
            endValue: 60,
            param: "pitch",
            framePeriodMs: 5,
            snapFinalValue: (value) => Math.round(value),
        });
        for (const value of built.dense) expect(Number.isInteger(value)).toBe(true);
    });
});
