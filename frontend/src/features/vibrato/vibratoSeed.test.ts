import { describe, expect, test } from "vitest";

import { sanitizeVibratoPreset, VIBRATO_LIMITS } from "./vibratoPresets";
import { randomVibratoSeed, VIBRATO_SEED_MAX, vibratoSeedForPreset } from "./vibratoSeed";

describe("vibratoSeedForPreset", () => {
    test("有 seed 字段时直接读字段", () => {
        expect(vibratoSeedForPreset({ id: "custom_a", seed: 12345 })).toBe(12345);
        // 0 是合法种子，不能被当成"缺失"。
        expect(vibratoSeedForPreset({ id: "custom_a", seed: 0 })).toBe(0);
    });

    test("seed 越界时钳回合法区间", () => {
        expect(vibratoSeedForPreset({ id: "x", seed: -5 })).toBe(0);
        expect(vibratoSeedForPreset({ id: "x", seed: VIBRATO_SEED_MAX + 100 })).toBe(
            VIBRATO_SEED_MAX,
        );
        expect(vibratoSeedForPreset({ id: "x", seed: 12.7 })).toBe(13);
    });

    test("非有限的 seed 回落到按 id 派生（旧数据兼容）", () => {
        const derived = vibratoSeedForPreset({ id: "builtin.natural" });
        expect(derived).toBeGreaterThanOrEqual(0);
        expect(derived).toBeLessThanOrEqual(VIBRATO_SEED_MAX);
        // 同一 id 每次一致。
        expect(vibratoSeedForPreset({ id: "builtin.natural" })).toBe(derived);
        // 显式给 NaN / Infinity 与缺席等价。
        expect(vibratoSeedForPreset({ id: "builtin.natural", seed: Number.NaN })).toBe(derived);
    });

    test("不同 id 的派生种子一般不同（预设各有各的抖动图案）", () => {
        expect(vibratoSeedForPreset({ id: "builtin.natural" })).not.toBe(
            vibratoSeedForPreset({ id: "builtin.drift" }),
        );
    });
});

describe("randomVibratoSeed", () => {
    test("落在 0..VIBRATO_SEED_MAX 且为整数", () => {
        for (let i = 0; i < 200; i += 1) {
            const value = randomVibratoSeed();
            expect(Number.isInteger(value)).toBe(true);
            expect(value).toBeGreaterThanOrEqual(0);
            expect(value).toBeLessThanOrEqual(VIBRATO_SEED_MAX);
        }
    });
});

describe("sanitizeVibratoPreset 的 seed 处理", () => {
    test("缺失时按 id 派生（与移除字段前的图案一致）", () => {
        const preset = sanitizeVibratoPreset({ id: "builtin.natural" });
        expect(preset.seed).toBe(vibratoSeedForPreset({ id: "builtin.natural" }));
    });

    test("显式 seed 原样保留", () => {
        expect(sanitizeVibratoPreset({ id: "custom_a", seed: 777 }).seed).toBe(777);
    });

    test("越界 seed 被钳进合法区间", () => {
        expect(sanitizeVibratoPreset({ id: "custom_a", seed: -1 }).seed).toBe(
            VIBRATO_LIMITS.seed.min,
        );
        expect(sanitizeVibratoPreset({ id: "custom_a", seed: 1e9 }).seed).toBe(
            VIBRATO_LIMITS.seed.max,
        );
    });

    test("掷出的骰子值能被净化器原样收下", () => {
        for (let i = 0; i < 50; i += 1) {
            const rolled = randomVibratoSeed();
            expect(sanitizeVibratoPreset({ id: "custom_a", seed: rolled }).seed).toBe(rolled);
        }
    });
});
