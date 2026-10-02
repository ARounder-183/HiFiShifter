import { describe, expect, it } from "vitest";

import {
    createPressureCalibration,
    createPressureSpread,
    observePressureSpread,
    spreadLooksConstant,
    DEFAULT_PRESSURE_CONFIG,
    observePressure,
    pressureCeiling,
    pressureLooksConstant,
    pressureToGain,
} from "./pressureCurve";

const CFG = DEFAULT_PRESSURE_CONFIG;

describe("pressureToGain", () => {
    it("returns the minimum gain at or below the dead zone", () => {
        expect(pressureToGain(0, CFG)).toBeCloseTo(CFG.minGain, 6);
        expect(pressureToGain(CFG.deadZone, CFG)).toBeCloseTo(CFG.minGain, 6);
        expect(pressureToGain(0.01, CFG)).toBeCloseTo(CFG.minGain, 6);
    });

    it("returns the maximum gain at the configured ceiling", () => {
        expect(pressureToGain(CFG.ceiling, CFG)).toBeCloseTo(CFG.maxGain, 6);
    });

    it("clamps beyond the ceiling", () => {
        expect(pressureToGain(1, CFG)).toBeCloseTo(CFG.maxGain, 6);
        expect(pressureToGain(4, CFG)).toBeCloseTo(CFG.maxGain, 6);
    });

    it("is monotonically increasing across the whole range", () => {
        let previous = Number.NEGATIVE_INFINITY;
        for (let raw = 0; raw <= 1.0001; raw += 0.01) {
            const gain = pressureToGain(raw, CFG);
            expect(gain).toBeGreaterThanOrEqual(previous - 1e-9);
            previous = gain;
        }
    });

    it("puts a firm press near 1.0 so the default drag keeps mouse-like speed", () => {
        // "正常用力"应当接近 1:1，否则用户会觉得"一换笔就变慢"。
        const firm = pressureToGain(0.64, CFG);
        expect(firm).toBeGreaterThan(0.9);
        expect(firm).toBeLessThan(1.12);
    });

    it("keeps a light press clearly finer and a hard press clearly faster", () => {
        expect(pressureToGain(0.15, CFG)).toBeLessThan(0.5);
        expect(pressureToGain(0.9, CFG)).toBeGreaterThan(1.3);
    });

    it("never leaves the configured output range", () => {
        for (let raw = 0; raw <= 1; raw += 0.05) {
            const gain = pressureToGain(raw, CFG);
            expect(gain).toBeGreaterThanOrEqual(CFG.minGain - 1e-9);
            expect(gain).toBeLessThanOrEqual(CFG.maxGain + 1e-9);
        }
    });

    it("falls back to 1 for non-finite input", () => {
        // 驱动在按下瞬间会上报 NaN；此时按"不加速也不减速"处理最安全。
        expect(pressureToGain(Number.NaN, CFG)).toBe(1);
        expect(pressureToGain(Number.POSITIVE_INFINITY, CFG)).toBe(1);
    });

    it("survives a degenerate config instead of producing NaN", () => {
        const broken = { ...CFG, ceiling: 0, deadZone: 0 };
        const gain = pressureToGain(0.5, broken);
        expect(Number.isFinite(gain)).toBe(true);
        expect(gain).toBeGreaterThan(0);
    });

    it("ignores a non-positive gamma instead of inverting the curve", () => {
        const broken = { ...CFG, gamma: 0 };
        const gain = pressureToGain(0.64, broken);
        expect(Number.isFinite(gain)).toBe(true);
        // 回退到默认 gamma，仍是"重压更快"。
        expect(gain).toBeGreaterThan(CFG.minGain);
    });
});

describe("pressure calibration", () => {
    it("lifts the ceiling when the driver reports above the configured one", () => {
        // 少数笔能报到 1.0 以上；不抬上界的话"再用力也不会更快"。
        const calibration = createPressureCalibration();
        const before = pressureCeiling(calibration, CFG);
        observePressure(calibration, 0.98);
        expect(pressureCeiling(calibration, CFG)).toBeGreaterThan(before);
        expect(pressureCeiling(calibration, CFG)).toBeCloseTo(0.98 * 1.05, 6);
    });

    it("never lowers the ceiling below the configured value", () => {
        // 上界只向上抬：向下压会把"轻描淡写的一笔"判成"用尽全力"，
        // 使同一段手势里的值反而走得更快 —— 与"轻 = 精细"正好相反。
        const calibration = createPressureCalibration();
        observePressure(calibration, 0.2);
        expect(pressureCeiling(calibration, CFG)).toBe(CFG.ceiling);
        observePressure(calibration, 0.5);
        expect(pressureCeiling(calibration, CFG)).toBe(CFG.ceiling);
    });

    it("leaves headroom above the observed maximum", () => {
        const calibration = createPressureCalibration();
        observePressure(calibration, 0.95);
        // 留 5% 余量，避免"一碰就顶格"。
        expect(pressureCeiling(calibration, CFG)).toBeCloseTo(0.9975, 6);
    });

    it("keeps a saturated driver able to go faster", () => {
        // 该驱动能到 1.0：不标定时 0.9 就已顶格；标定后 1.0 仍高于 0.9 的倍率。
        const calibration = createPressureCalibration();
        observePressure(calibration, 1);
        const atCeiling = pressureToGain(0.9, CFG, calibration);
        const atMax = pressureToGain(1, CFG, calibration);
        expect(atMax).toBeGreaterThan(atCeiling);
        expect(atCeiling).toBeLessThan(CFG.maxGain);
    });

    it("ignores non-finite and non-positive samples", () => {
        const calibration = createPressureCalibration();
        observePressure(calibration, Number.NaN);
        observePressure(calibration, -1);
        observePressure(calibration, 0);
        expect(calibration.observedMax).toBe(0);
    });

    it("tracks the maximum, not the latest sample", () => {
        const calibration = createPressureCalibration();
        observePressure(calibration, 0.7);
        observePressure(calibration, 0.3);
        expect(calibration.observedMax).toBe(0.7);
    });

    it("stays monotonic with a calibration in play", () => {
        const calibration = createPressureCalibration();
        observePressure(calibration, 1);
        let previous = Number.NEGATIVE_INFINITY;
        for (let raw = 0; raw <= 1.0001; raw += 0.01) {
            const gain = pressureToGain(raw, CFG, calibration);
            expect(gain).toBeGreaterThanOrEqual(previous - 1e-9);
            previous = gain;
        }
    });
});

describe("pressureLooksConstant", () => {
    it("detects drivers that report a fixed value", () => {
        expect(pressureLooksConstant([0.5, 0.5, 0.5])).toBe(true);
        expect(pressureLooksConstant([1, 1, 1, 1])).toBe(true);
        expect(pressureLooksConstant([0, 0, 0])).toBe(true);
    });

    it("treats a single sample as constant", () => {
        expect(pressureLooksConstant([0.7])).toBe(true);
        expect(pressureLooksConstant([])).toBe(true);
    });

    it("accepts a real varying stroke", () => {
        expect(pressureLooksConstant([0.1, 0.4, 0.9, 0.3])).toBe(false);
    });

    it("tolerates quantisation noise", () => {
        expect(pressureLooksConstant([0.5, 0.5005, 0.5], 1e-3)).toBe(true);
    });

    it("ignores non-finite samples when judging", () => {
        expect(pressureLooksConstant([0.5, Number.NaN, 0.5])).toBe(true);
        expect(pressureLooksConstant([Number.NaN, Number.NaN])).toBe(true);
    });
});

/*
 * 增量极值分布：与一次性的数组版本必须给出同一个判定。
 *
 * 【为什么要有两套】手势可能持续十几秒、采样 250Hz（几千个样本），每次都扫整段
 * 序列是 O(n²)。hook 里用增量版，数组版留给一次性查询与测试 —— 两者若判定不一致，
 * "这台设备到底算不算在报压感"就会在两条路径上给出不同答案。
 */
describe("PressureSpread", () => {
    it("agrees with the array version on every case", () => {
        const cases: number[][] = [
            [],
            [0.5],
            [0.5, 0.5, 0.5],
            [0, 0, 0],
            [1, 1, 1, 1],
            [0.1, 0.4, 0.9, 0.3],
            [0.5, Number.NaN, 0.5],
            [Number.NaN, Number.NaN],
        ];
        for (const samples of cases) {
            const spread = createPressureSpread();
            for (const sample of samples) observePressureSpread(spread, sample);
            expect(spreadLooksConstant(spread)).toBe(pressureLooksConstant(samples));
        }
    });

    it("tracks the extremes and the count", () => {
        const spread = createPressureSpread();
        observePressureSpread(spread, 0.7);
        observePressureSpread(spread, 0.2);
        observePressureSpread(spread, 0.5);
        expect(spread.count).toBe(3);
        expect(spread.min).toBe(0.2);
        expect(spread.max).toBe(0.7);
    });

    it("ignores non-finite samples without counting them", () => {
        const spread = createPressureSpread();
        observePressureSpread(spread, Number.NaN);
        observePressureSpread(spread, Number.POSITIVE_INFINITY);
        expect(spread.count).toBe(0);
        expect(spreadLooksConstant(spread)).toBe(true);
    });
});
