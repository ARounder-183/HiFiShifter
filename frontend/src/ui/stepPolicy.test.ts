/*
 * 步长策略测试。
 *
 * 锁的是"同一个单位语义在任何地方都得到同一个步长"这条契约 —— 这正是
 * 审查发现的缺口（BPM 在三个地方一致、在第四个地方没有精细调整）。
 */
import { describe, expect, test } from "vitest";

import { resolveStep, stepFor, stepValue, type StepUnit } from "./stepPolicy";

describe("stepFor", () => {
    test("精细步长严格小于粗调步长（除非该单位本身是离散的）", () => {
        const discrete: StepUnit[] = ["semitone", "integer", "pixels"];
        for (const unit of Object.keys(stepFor("bpm")) as StepUnit[]) {
            void unit;
        }
        for (const unit of [
            "bpm",
            "cents",
            "percent",
            "percentFine",
            "gainDb",
            "levelDb",
            "rate",
            "milliseconds",
            "seconds",
        ] as StepUnit[]) {
            const spec = stepFor(unit);
            expect(spec.fine, `${unit} 的精调步长应小于粗调`).toBeLessThan(spec.coarse);
        }
        for (const unit of discrete) {
            const spec = stepFor(unit);
            expect(spec.fine, `${unit} 是离散量，精调应与粗调相同`).toBe(spec.coarse);
        }
    });

    test("取整百分比与小数百分比是两个单位（避免把 0.35% 截成 1%）", () => {
        expect(stepFor("percent").decimals).toBe(0);
        expect(stepFor("percentFine").decimals).toBe(2);
        expect(stepFor("percentFine").fine).toBeLessThan(stepFor("percent").fine);
    });

    test("小数位与步长匹配（不会把 0.1 步长写成 0 位小数）", () => {
        expect(stepFor("bpm").decimals).toBe(1);
        expect(stepFor("rate").decimals).toBe(2);
        expect(stepFor("seconds").decimals).toBe(3);
        expect(stepFor("integer").decimals).toBe(0);
    });

    test("每个单位都有定义（新增单位必须显式给步长）", () => {
        const units: StepUnit[] = [
            "bpm",
            "cents",
            "semitone",
            "percent",
            "percentFine",
            "gainDb",
            "levelDb",
            "rate",
            "milliseconds",
            "seconds",
            "pixels",
            "integer",
        ];
        for (const unit of units) {
            const spec = stepFor(unit);
            expect(spec.coarse, `${unit} 缺粗调步长`).toBeGreaterThan(0);
            expect(spec.fine, `${unit} 缺精调步长`).toBeGreaterThan(0);
        }
    });
});

describe("resolveStep", () => {
    test("按修饰键在粗/精之间切换", () => {
        expect(resolveStep("bpm", false)).toBe(1);
        expect(resolveStep("bpm", true)).toBe(0.1);
        expect(resolveStep("percent", false)).toBe(5);
        expect(resolveStep("percent", true)).toBe(1);
    });
});

describe("stepValue", () => {
    test("向上/向下各走一格", () => {
        const base = { unit: "bpm" as StepUnit, fine: false, min: 0, max: 999 };
        expect(stepValue({ ...base, value: 120, direction: 1 })).toBe(121);
        expect(stepValue({ ...base, value: 120, direction: -1 })).toBe(119);
    });

    test("夹紧到上下界", () => {
        const base = { unit: "percent" as StepUnit, fine: false, min: 0, max: 100 };
        expect(stepValue({ ...base, value: 100, direction: 1 })).toBe(100);
        expect(stepValue({ ...base, value: 0, direction: -1 })).toBe(0);
    });

    test("浮点噪声被消掉（不产生 0.30000000000000004）", () => {
        const out = stepValue({
            value: 0.2,
            direction: 1,
            unit: "rate",
            fine: true, // 0.01
            min: 0,
            max: 10,
        });
        expect(out).toBe(0.21);
        expect(String(out)).not.toContain("000000");
    });

    test("精调一次只走精调步长", () => {
        const args = { value: 120, direction: 1 as const, unit: "bpm" as StepUnit, min: 0, max: 999 };
        expect(stepValue({ ...args, fine: false })).toBe(121);
        expect(stepValue({ ...args, fine: true })).toBe(120.1);
    });
});
