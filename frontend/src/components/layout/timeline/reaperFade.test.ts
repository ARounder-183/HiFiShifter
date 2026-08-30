import { describe, expect, it } from "vitest";

import {
    DEFAULT_FADE_DIR_BY_SHAPE,
    FADE_LANDING_FRAC,
    FADE_PRESETS,
    defaultFadeDirFor,
    fadeGainSigned,
    resolveCurvatureEditBase,
    solveDirAt,
} from "./reaperFade";

describe("fadeGainSigned", () => {
    it("endpoints are exact for all shapes and dirs in both directions", () => {
        const shapes = [...FADE_PRESETS.map((p) => p.shape), 1.1, 5.1];
        for (const shape of shapes) {
            for (const dir of [-1, -0.35, 0, 0.35, 1]) {
                expect(fadeGainSigned(shape, dir, "in", 0)).toBe(0);
                expect(fadeGainSigned(shape, dir, "in", 1)).toBe(1);
                expect(fadeGainSigned(shape, dir, "out", 0)).toBe(1);
                expect(fadeGainSigned(shape, dir, "out", 1)).toBe(0);
            }
        }
    });

    it("gain is monotonic in t (in rising, out falling)", () => {
        for (const shape of [0, 1, 2, 3, 4, 5, 6, 1.1, 5.1]) {
            for (const dir of [-1, -0.35, 0, 0.35, 1]) {
                let prevIn = fadeGainSigned(shape, dir, "in", 0);
                let prevOut = fadeGainSigned(shape, dir, "out", 0);
                for (let step = 1; step <= 200; step += 1) {
                    const t = step / 200;
                    const gIn = fadeGainSigned(shape, dir, "in", t);
                    const gOut = fadeGainSigned(shape, dir, "out", t);
                    // 极小 t 处 x^e 的 ULP 舍入允许 <1e-9 回退。
                    expect(gIn).toBeGreaterThanOrEqual(prevIn - 1e-9);
                    expect(gOut).toBeLessThanOrEqual(prevOut + 1e-9);
                    prevIn = gIn;
                    prevOut = gOut;
                }
            }
        }
    });

    /**
     * 黄金锚点：必须与 Rust `fade_curves.rs` 的
     * `golden_values_match_typescript_anchor_set` 完全一致。
     * 常数定标依据（用户实测）：
     * - fast_start 在 dir≈±0.33 视觉线性；
     * - 快收/陡峭族在各自默认 dir 上为锚点形态、dir=0 处线性；
     * - 线性预设 ±1 端呈"类似快起"走向；陡峭慢起慢收中点斜率远大于慢起慢收。
     */
    it("golden values match rust anchor set", () => {
        expect(fadeGainSigned(1, 0, "in", 0.25)).toBeCloseTo(0.5358867312681466, 9);
        expect(fadeGainSigned(1, 0.33, "in", 0.25)).toBeCloseTo(0.25, 8);
        expect(fadeGainSigned(1, -0.33, "out", 0.75)).toBeCloseTo(0.25, 8); // σ 镜像同 u
        expect(fadeGainSigned(2, 1, "in", 0.25)).toBeCloseTo(0.011841535675862483, 9);
        expect(fadeGainSigned(2, 0, "in", 0.25)).toBeCloseTo(0.25, 8); // 默认锚即线性点
        expect(fadeGainSigned(3, -1, "in", 0.25)).toBeCloseTo(0.8705505632961241, 9);
        expect(fadeGainSigned(4, 1, "in", 0.25)).toBeCloseTo(1.52587890625e-5, 12);
        expect(fadeGainSigned(0, -1, "in", 0.25)).toBeCloseTo(0.7690081847607293, 9); // 快起向
        expect(fadeGainSigned(0, 0, "in", 0.37)).toBeCloseTo(0.37, 12); // 直线
        expect(fadeGainSigned(5, 0, "in", 0.25)).toBeCloseTo(0.08188949557659113, 9);
        expect(fadeGainSigned(6, 0, "in", 0.5)).toBeCloseTo(0.5, 12);
        expect(fadeGainSigned(1.1, 0, "in", 0.25)).toBeCloseTo(0.14644660940672624, 9);
    });

    it("steep S-curve is visibly steeper than slight at midpoint", () => {
        const slopeOf = (shape: number) =>
            (fadeGainSigned(shape, 0, "in", 0.501) - fadeGainSigned(shape, 0, "in", 0.499)) / 0.002;
        expect(slopeOf(6)).toBeGreaterThan(slopeOf(5) * 2);
    });

    it("sigma mirror: same world-u yields identical ascending geometry", () => {
        // 方向镜像恒等式：淡入 dir=+d 与淡出 dir=−d 归一到同一世界参数
        // u，因此以"上升视角"取值（out 模式的几何自变量就是其返回值本身
        // 所在的下降轨迹）在相同 x 处相等 —— 这正是默认曲率表跨方向一致
        // 的数学根据。
        for (const shape of [1, 2, 3, 4]) {
            for (const d of [0.6, 0.2, 1]) {
                for (const x of [0.2, 0.5, 0.8]) {
                    const gIn = fadeGainSigned(shape, d, "in", x);
                    // out 模式入口带时间镜像（内部 x=1-t），故取补角调用。
                    const gOut = fadeGainSigned(shape, -d, "out", 1 - x);
                    expect(gIn).toBeCloseTo(gOut, 12);
                }
            }
        }
    });

    it("landing window is continuous at its junctions", () => {
        const eps = 1e-9;
        const tau = FADE_LANDING_FRAC;
        const shapes = [...FADE_PRESETS.map((p) => p.shape), 1.1, 5.1];
        for (const shape of shapes) {
            for (const dir of [-1, 0, 1]) {
                const lo = 1 - tau;
                expect(
                    Math.abs(
                        fadeGainSigned(shape, dir, "out", lo - eps) -
                            fadeGainSigned(shape, dir, "out", lo + eps),
                    ),
                ).toBeLessThan(1e-6);
                expect(
                    Math.abs(
                        fadeGainSigned(shape, dir, "in", tau - eps) -
                            fadeGainSigned(shape, dir, "in", tau + eps),
                    ),
                ).toBeLessThan(1e-6);
            }
        }
    });

    it("per-frame gain step is click-safe for slow-start fade-outs", () => {
        // “先慢后快”最恶劣组合：淡出 u=-dir<0 → e<1（末端陡峭、旧实现
        // 在末帧留下 (1/N)^e 级增益阶跃 → Click）。端点锁定约定：
        // 第 k 帧进度 (k+1)/N，最后一帧恰为 1 → 增益精确 0。
        const worstCases: Array<[number, number]> = [
            [1, 1],
            [1, 0.6],
            [3, 1],
            [3, 0.35],
        ];
        for (const [shape, dir] of worstCases) {
            for (const frames of [480, 2048, 48000]) {
                let prev = fadeGainSigned(shape, dir, "out", 0);
                let maxStep = 0;
                for (let frame = 0; frame < frames; frame += 1) {
                    const consumed = (frame + 1) / frames;
                    const g = fadeGainSigned(shape, dir, "out", consumed);
                    expect(g).toBeLessThanOrEqual(prev + 1e-6);
                    maxStep = Math.max(maxStep, prev - g);
                    prev = g;
                }
                expect(prev).toBe(0);
                expect(maxStep).toBeLessThan(0.025);
            }
        }
    });

    it("per-frame gain step is click-safe for fast-start fade-ins", () => {
        // 对称问题：淡入 u=dir<0 → e<1（起点陡峭），首帧不再携带
        // (1/N)^e 级增益突跳。
        const worstCases: Array<[number, number]> = [
            [1, -1],
            [1, -0.6],
            [3, -1],
            [3, -0.35],
        ];
        for (const [shape, dir] of worstCases) {
            for (const frames of [480, 2048, 48000]) {
                let prev = fadeGainSigned(shape, dir, "in", 0);
                let maxStep = 0;
                for (let frame = 0; frame < frames; frame += 1) {
                    const consumed = (frame + 1) / frames;
                    const g = fadeGainSigned(shape, dir, "in", consumed);
                    expect(g).toBeGreaterThanOrEqual(prev - 1e-6);
                    maxStep = Math.max(maxStep, g - prev);
                    prev = g;
                }
                expect(maxStep).toBeLessThan(0.025);
            }
        }
    });
});

describe("default curvature table", () => {
    it("matches the user-measured REAPER defaults ({fade_in, fade_out})", () => {
        expect(DEFAULT_FADE_DIR_BY_SHAPE.linear).toEqual({ in_: 0, out: 0 });
        expect(DEFAULT_FADE_DIR_BY_SHAPE.convexSlight).toEqual({ in_: 0, out: 0 });
        expect(DEFAULT_FADE_DIR_BY_SHAPE.lateSlight).toEqual({ in_: 1, out: -1 });
        expect(DEFAULT_FADE_DIR_BY_SHAPE.convexSharp).toEqual({ in_: -1, out: 1 });
        expect(DEFAULT_FADE_DIR_BY_SHAPE.lateSharp).toEqual({ in_: 1, out: -1 });
        expect(DEFAULT_FADE_DIR_BY_SHAPE.sSlight).toEqual({ in_: 0, out: 0 });
        expect(DEFAULT_FADE_DIR_BY_SHAPE.sSharp).toEqual({ in_: 0, out: 0 });
    });

    it("resolves per-side reset values by shape id", () => {
        expect(defaultFadeDirFor(2, false)).toBe(1);
        expect(defaultFadeDirFor(2, true)).toBe(-1);
        expect(defaultFadeDirFor(3, false)).toBe(-1);
        expect(defaultFadeDirFor(5.1, true)).toBe(0); // 分数变体取整到锐利 S
    });
});

describe("resolveCurvatureEditBase", () => {
    it("no longer promotes presets — curvature applies to every shape including linear", () => {
        // 模型修正后：线性是一条真实可弯的预设轴（dir 驱动 e 全量程），
        // 不再需要首次编辑时切换形状。非法值仍归一到线性而非悄悄换形状。
        for (const raw of [0, 1, 2, 3, 4, 5, 6, 1.1, 5.1]) {
            const resolved = resolveCurvatureEditBase(raw);
            expect(resolved).toEqual({ shape: raw, promotedFromLinear: false });
        }
        expect(resolveCurvatureEditBase(Number.NaN)).toEqual({
            shape: 0,
            promotedFromLinear: false,
        });
        expect(resolveCurvatureEditBase(7.3)).toEqual({ shape: 0, promotedFromLinear: false });
    });

    it("linear actually bends with curvature (the original dead-end regression)", () => {
        const straight = fadeGainSigned(0, 0, "in", 0.35);
        const bentFast = fadeGainSigned(0, -1, "in", 0.35); // 快起向
        const bentLate = fadeGainSigned(0, 1, "in", 0.35); // 快收向
        expect(Math.abs(bentFast - straight)).toBeGreaterThan(0.01);
        expect(Math.abs(bentLate - straight)).toBeGreaterThan(0.01);
    });
});

describe("solveDirAt", () => {
    it("recovers the direction that produced a target gain (all shapes)", () => {
        for (const shape of [0, 1, 2, 3, 4]) {
            for (const dir of [-0.75, -0.33, 0, 0.45, 1]) {
                for (const t of [0.2, 0.5, 0.8]) {
                    for (const mode of ["in", "out"] as const) {
                        const target = fadeGainSigned(shape, dir, mode, t);
                        const solved = solveDirAt(shape, mode, t, target, dir);
                        expect(Math.abs(solved - dir)).toBeLessThan(0.02);
                    }
                }
            }
        }
    });

    it("returns some in-range solution for S families and equal-power variant", () => {
        for (const shape of [5, 6, 1.1]) {
            for (const t of [0.25, 0.5, 0.7]) {
                const target = fadeGainSigned(shape, 0.5, "in", t);
                const solved = solveDirAt(shape, "in", t, target, 0.5);
                expect(solved).toBeGreaterThanOrEqual(-1);
                expect(solved).toBeLessThanOrEqual(1);
                expect(Math.abs(fadeGainSigned(shape, solved, "in", t) - target)).toBeLessThan(
                    1e-3,
                );
            }
        }
    });

    it("clamps unreachable targets to a finite boundary", () => {
        const solved = solveDirAt(3, "in", 0.98, 0.02, 0);
        expect(Number.isFinite(solved)).toBe(true);
        expect(Math.abs(solved)).toBeLessThanOrEqual(1);
    });
});
