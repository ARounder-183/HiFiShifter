import { describe, expect, test } from "vitest";

import type { Keybinding } from "../../../features/keybindings/types";
import { VIBRATO_LIMITS, sanitizeVibratoPreset } from "../../../features/vibrato/vibratoPresets";
import {
    buildDragVibratoCurve,
    computeVibratoDragAdjustment,
    createDragWorking,
    depthStepCentsFor,
    resolveVibratoDragKeyboardAdjustment,
    resolveVibratoPresetSwitch,
    resolveVibratoSideButton,
    SIDE_BUTTON_BACK,
    SIDE_BUTTON_BACK_MASK,
    SIDE_BUTTON_FORWARD,
    SIDE_BUTTON_FORWARD_MASK,
    switchDragPreset,
} from "./vibratoDragAdjust";

const preset = (overrides: Parameters<typeof sanitizeVibratoPreset>[0] = {}) =>
    sanitizeVibratoPreset({ id: "custom_test", ...overrides });

const kb = (overrides: Partial<Keybinding> = {}): Keybinding => ({ key: "a", ...overrides });

describe("createDragWorking", () => {
    test("深度与速率都取预设自带值", () => {
        const p = preset({ depthCents: 42, rateHz: 6.5 });
        const working = createDragWorking(p);
        expect(working.depthCents).toBe(42);
        expect(working.rateHz).toBe(6.5);
        expect(working.preset.id).toBe(p.id);
        expect(working.adjusted).toBe(false);
    });

    test("起手不携带任何跨手势的调整", () => {
        const working = createDragWorking(preset({ depthCents: 42, rateHz: 6.5 }));
        expect(working.depthCents).toBe(42);
        expect(working.rateHz).toBe(6.5);
    });

    test("调整标记初始为 false（未调整）", () => {
        const working = createDragWorking(preset({ depthCents: 30 }));
        expect(working.adjusted).toBe(false);
    });
});

describe("switchDragPreset", () => {
    test("未调整过：深度与速率取新预设自带的值", () => {
        const before = createDragWorking(preset({ depthCents: 40 }));
        const after = switchDragPreset(
            before,
            preset({ id: "builtin.deep", depthCents: 55, rateHz: 4.5 }),
        );
        expect(after.depthCents).toBe(55);
        expect(after.rateHz).toBe(4.5);
        expect(after.adjusted).toBe(false);
        expect(after.preset.id).toBe("builtin.deep");
    });

    test("调整过：深度与速率沿用本次手势的值（换音色不改幅度）", () => {
        const before = {
            ...createDragWorking(preset({ depthCents: 40, rateHz: 5 })),
            depthCents: 90,
            rateHz: 9,
            adjusted: true,
        };
        const after = switchDragPreset(
            before,
            preset({ id: "builtin.deep", depthCents: 55, rateHz: 4.5 }),
        );
        // 两个量都继承 —— 用户只调了幅度时，速率也应保持这一笔一直在用的值。
        expect(after.depthCents).toBe(90);
        expect(after.rateHz).toBe(9);
        expect(after.adjusted).toBe(true);
        // 但波形等仍来自新预设。
        expect(after.preset.id).toBe("builtin.deep");
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

    test("深度可为负（波形反相），并钳在合法下界", () => {
        const next = computeVibratoDragAdjustment({
            ...base,
            depthCents: 5,
            target: "depth",
            direction: -1,
            steps: 10,
            fineScale: 1,
        });
        // 5 - 24*10 = -235：负值是合法结果（反相），不再停在 0。
        expect(next.depthCents).toBeCloseTo(-235, 9);

        // 越过下界才钳住。
        const floored = computeVibratoDragAdjustment({
            ...base,
            depthCents: 5,
            target: "depth",
            direction: -1,
            steps: 1000,
            fineScale: 1,
        });
        expect(floored.depthCents).toBe(VIBRATO_LIMITS.depthCents.min);
    });

    test("深度为负时继续向上调可以回到正值", () => {
        const next = computeVibratoDragAdjustment({
            ...base,
            depthCents: -30,
            target: "depth",
            direction: 1,
            steps: 1,
            fineScale: 1,
        });
        expect(next.depthCents).toBeCloseTo(-30 + 24, 9);
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

describe("resolveVibratoSideButton", () => {
    /*
     * 侧键的 `button`（索引 3 / 4）与 `buttons`（位掩码 8 / 16）是两套编号 ——
     * 与 `penInput.ts` 里"橡皮端是位 32 不是位 2"同一类陷阱。这里把两者都钉住，
     * 因为"按了没反应"和"按了乱跳"都不会自己报错。
     */
    test("前进键 = 下一个，后退键 = 上一个（与键盘同一约定）", () => {
        expect(resolveVibratoSideButton(SIDE_BUTTON_FORWARD)).toBe(1);
        expect(resolveVibratoSideButton(SIDE_BUTTON_BACK)).toBe(-1);
    });

    test("左右中键与其他按键都不参与", () => {
        for (const button of [0, 1, 2, 5, -1, 99]) {
            expect(resolveVibratoSideButton(button)).toBeNull();
        }
    });

    test("位掩码常量与 button 索引一致（左键 1 之外互不重叠）", () => {
        // 侧键索引 3 / 4 对应位 3 / 4，即 8 / 16；两者不能相等，
        // 否则"按住左键 + 按侧键"的 `buttons` 校验会失效。
        expect(SIDE_BUTTON_BACK_MASK).toBe(1 << SIDE_BUTTON_BACK);
        expect(SIDE_BUTTON_FORWARD_MASK).toBe(1 << SIDE_BUTTON_FORWARD);
        expect(SIDE_BUTTON_BACK_MASK & 1).toBe(0);
        expect(SIDE_BUTTON_FORWARD_MASK & 1).toBe(0);
    });

    test("左键 + 侧键同时按下时，左键位仍然成立（拖拽不会被误判为松手）", () => {
        const bothBack = 1 | SIDE_BUTTON_BACK_MASK;
        const bothForward = 1 | SIDE_BUTTON_FORWARD_MASK;
        expect(bothBack & 1).toBe(1);
        expect(bothForward & 1).toBe(1);
    });
});
