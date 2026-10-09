/**
 * 曲率拖拽的**二维求解**（REAPER ≥7.81 的 `(curvature, S)` 两轴宿主）。
 *
 * 【为什么值得单测】一维求解器 `solveNearestCurveDir` 的签名是 `(t, dir) => gain`，
 * 结构上只能解一维。把它用在两轴宿主上时，它会**把 S 钉死成常数**、只动 curvature
 * —— 于是用户在 S 族曲线上拖动时，解出的点根本不在曲线族上，表现为"拖了不跟手 /
 * 跳变"。本文件钉住"二维解确实落在族上、且不比一维差"。
 */
import { describe, expect, it } from "vitest";

import { hostFadeGainForAxes } from "./hostFadeDisplay";
import { solveNearestCurveAxes, solveNearestCurveDir } from "./reaperFade";

/** 拖拽里指针的 y 权重（绘制高度 / 宽度）。取一个非 1 的值以覆盖换算路径。 */
const ASPECT = 0.5;

function distance(mode: "in" | "out", c: number, s: number, t: number, px: number, py: number) {
    const g = hostFadeGainForAxes(mode, c, s, t);
    return Math.hypot(t - px, (g - py) * ASPECT);
}

describe("solveNearestCurveAxes（两轴宿主）", () => {
    it("recovers a point that lies exactly on the two-axis family", () => {
        // 覆盖预设坐标（0/±0.5/±1）与表外的连续坐标 —— 后者才是用户拖出来的常态。
        const axes: Array<[number, number]> = [
            [0, 0],
            [0.5, 0],
            [-0.5, 0],
            [1, 0],
            [0, 0.5],
            [0, 1],
            [0.25, -0.4],
        ];
        for (const [c0, s0] of axes) {
            for (const t0 of [0.2, 0.35, 0.6, 0.8]) {
                const g0 = hostFadeGainForAxes("in", c0, s0, t0);
                const solved = solveNearestCurveAxes({
                    mode: "in",
                    curvature: c0,
                    s: s0,
                    pointerX01: t0,
                    pointerY01: g0,
                    aspectYOverX: ASPECT,
                    gainAt: (t, c, s) => hostFadeGainForAxes("in", c, s, t),
                });
                // 解出的曲线在指针 x 处的增益必须≈指针 y（"画的=抓的=解的"）。
                expect(Math.abs(solved.gain - g0)).toBeLessThan(0.03);
                expect(Math.abs(solved.t - t0)).toBeLessThan(0.06);
                expect(solved.curvature).toBeGreaterThanOrEqual(-1);
                expect(solved.curvature).toBeLessThanOrEqual(1);
                expect(solved.s).toBeGreaterThanOrEqual(-1);
                expect(solved.s).toBeLessThanOrEqual(1);
            }
        }
    });

    it("never lands farther from the pointer than the 1-D solver on the same pointer", () => {
        for (const s0 of [0.5, 1, -0.75]) {
            for (const t0 of [0.25, 0.5, 0.75]) {
                for (const py of [0.2, 0.5, 0.8]) {
                    const two = solveNearestCurveAxes({
                        mode: "in",
                        curvature: 0,
                        s: s0,
                        pointerX01: t0,
                        pointerY01: py,
                        aspectYOverX: ASPECT,
                        gainAt: (t, c, s) => hostFadeGainForAxes("in", c, s, t),
                    });
                    const d2 = distance("in", two.curvature, two.s, two.t, t0, py);
                    const one = solveNearestCurveDir({
                        shape: 0,
                        dir: 0,
                        mode: "in",
                        pointerX01: t0,
                        pointerY01: py,
                        aspectYOverX: ASPECT,
                        gainAt: (t, c) => hostFadeGainForAxes("in", c, s0, t),
                    });
                    const d1 = distance("in", one.dir, s0, one.t, t0, py);
                    // 二维族是一维族（S 固定切片）的**超集**，解不可能更差。
                    // 容差吸收两者粗扫网格步长不同带来的舍入差异。
                    expect(d2).toBeLessThanOrEqual(d1 + 0.02);
                }
            }
        }
    });

    it("keeps S near zero when the family is pure curvature", () => {
        // 纯曲率族（S=0）上的点：S 分量不应被凭空引入。
        const t0 = 0.3;
        const g0 = hostFadeGainForAxes("in", 0.5, 0, t0);
        const solved = solveNearestCurveAxes({
            mode: "in",
            curvature: 0.5,
            s: 0,
            pointerX01: t0,
            pointerY01: g0,
            aspectYOverX: ASPECT,
            gainAt: (t, c, s) => hostFadeGainForAxes("in", c, s, t),
        });
        expect(Math.abs(solved.s)).toBeLessThan(0.05);
    });

    it("reaches a curvature-family target that the S-fixed 1-D solver cannot", () => {
        // 目标点落在**纯曲率族**上，而一维求解器把 S 钉死在 1（S 族）——
        // 它只能沿 S 族滑动，够不到目标；二维解则能同时调整两个分量。
        const t0 = 0.3;
        const g0 = hostFadeGainForAxes("in", 0.5, 0, t0);
        const two = solveNearestCurveAxes({
            mode: "in",
            curvature: 0,
            s: 1,
            pointerX01: t0,
            pointerY01: g0,
            aspectYOverX: ASPECT,
            gainAt: (t, c, s) => hostFadeGainForAxes("in", c, s, t),
        });
        const one = solveNearestCurveDir({
            shape: 0,
            dir: 0,
            mode: "in",
            pointerX01: t0,
            pointerY01: g0,
            aspectYOverX: ASPECT,
            gainAt: (t, c) => hostFadeGainForAxes("in", c, 1, t),
        });
        const d2 = distance("in", two.curvature, two.s, two.t, t0, g0);
        const d1 = distance("in", one.dir, 1, one.t, t0, g0);
        expect(d2).toBeLessThan(0.03);
        expect(d2).toBeLessThan(d1);
    });
});
