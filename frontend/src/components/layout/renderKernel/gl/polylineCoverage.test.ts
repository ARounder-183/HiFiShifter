/**
 * 折线覆盖率（./polylineCoverage）行为自检。
 *
 * 【主要内容】
 * 1. 横向覆盖率：中心满覆盖、边缘半覆盖、外侧零覆盖，过渡宽度等于 `aaWidth`；
 * 2. 虚线覆盖率：墨心满覆盖、隙心零覆盖、边界半覆盖、周期性；
 * 3. 边界情形：`aaWidth` 非法、`dash < 0`（实线）、负弧长、超大弧长。
 *
 * 【作用】
 * 这两个函数是"曲线看起来对不对"的核心数学，而它们的 GLSL 副本无法在 node 环境
 * 执行。本测试锁住 TS 参考实现的行为；`polylineProgram.ts` 的片元着色器保留逐行
 * 等价副本（见该文件与 `polylineCoverage.ts` 的同步义务说明）。
 *
 * 【为什么这些断言值有意义】半覆盖（0.5）出现在**几何边界**上——这正是抗锯齿的
 * 定义：边界处的像素应当拿到一半的墨。若实现把过渡区间整体偏移半个宽度，边界处
 * 会变成 1 或 0，曲线的视觉粗细就会随落点相位跳变（Phase 2 的网格层踩过同类问题）。
 */
import { describe, expect, it } from "vitest";

import { dashCoverage, lateralCoverage } from "./polylineCoverage";

describe("lateralCoverage", () => {
    it("中心满覆盖、边界半覆盖、外侧零覆盖", () => {
        const halfWidth = 0.9; // lineWidth 1.8
        const aa = 1 / 2; // dpr = 2
        expect(lateralCoverage(0, halfWidth, aa)).toBe(1);
        expect(lateralCoverage(0.5, halfWidth, aa)).toBe(1);
        // 几何边界处恰好半覆盖（抗锯齿的定义）
        expect(lateralCoverage(halfWidth, halfWidth, aa)).toBeCloseTo(0.5, 9);
        expect(lateralCoverage(-halfWidth, halfWidth, aa)).toBeCloseTo(0.5, 9);
        // 过渡区外侧为零
        expect(lateralCoverage(halfWidth + aa, halfWidth, aa)).toBe(0);
        expect(lateralCoverage(-(halfWidth + aa), halfWidth, aa)).toBe(0);
    });

    it("过渡是单调的（从内到外不反弹）", () => {
        const halfWidth = 1.3; // lineWidth 2.6
        const aa = 0.5;
        let prev = 1;
        for (let d = 0; d <= halfWidth + aa + 0.1; d += 0.05) {
            const c = lateralCoverage(d, halfWidth, aa);
            expect(c).toBeLessThanOrEqual(prev + 1e-9);
            prev = c;
        }
    });

    it("对距离取绝对值（左右对称）", () => {
        const halfWidth = 1.8; // lineWidth 3.6
        const aa = 0.5;
        for (const d of [0.3, 1.0, 1.75, 1.95]) {
            expect(lateralCoverage(d, halfWidth, aa)).toBeCloseTo(
                lateralCoverage(-d, halfWidth, aa),
                12,
            );
        }
    });

    it("过渡宽度等于 aaWidth（外侧恰好到达时归零）", () => {
        const halfWidth = 2;
        const aa = 0.8;
        // 过渡区 = [halfWidth - aa/2, halfWidth + aa/2]
        expect(lateralCoverage(halfWidth - aa / 2, halfWidth, aa)).toBe(1);
        expect(lateralCoverage(halfWidth + aa / 2, halfWidth, aa)).toBe(0);
        // 中点 = 半覆盖
        expect(lateralCoverage(halfWidth, halfWidth, aa)).toBeCloseTo(0.5, 9);
    });

    it("aaWidth 非法时回退到极小正值（不产生 NaN / 除零）", () => {
        for (const bad of [0, -1, Number.NaN]) {
            const c = lateralCoverage(0, 1, bad);
            expect(Number.isFinite(c)).toBe(true);
            expect(c).toBeGreaterThanOrEqual(0);
            expect(c).toBeLessThanOrEqual(1);
        }
    });
});

describe("dashCoverage", () => {
    const dash = 6;
    const gap = 6;
    const aa = 0.5;

    it("墨心满覆盖、隙心零覆盖、边界半覆盖", () => {
        expect(dashCoverage(0, dash, gap, aa)).toBeCloseTo(0.5, 9); // 起点即边界
        expect(dashCoverage(3, dash, gap, aa)).toBe(1); // 墨心
        expect(dashCoverage(6, dash, gap, aa)).toBeCloseTo(0.5, 9); // 墨/隙边界
        expect(dashCoverage(9, dash, gap, aa)).toBe(0); // 隙心
        expect(dashCoverage(12, dash, gap, aa)).toBeCloseTo(0.5, 9); // 周期边界
    });

    it("按周期重复", () => {
        const period = dash + gap;
        for (const along of [1.5, 4, 7, 10.5]) {
            expect(dashCoverage(along, dash, gap, aa)).toBeCloseTo(
                dashCoverage(along + period * 3, dash, gap, aa),
                9,
            );
        }
    });

    it("沿一个周期遍历：覆盖率不出现负值或超过 1", () => {
        for (let a = 0; a <= dash + gap + 1; a += 0.1) {
            const c = dashCoverage(a, dash, gap, aa);
            expect(c).toBeGreaterThanOrEqual(0);
            expect(c).toBeLessThanOrEqual(1);
        }
    });

    it("多个周期内墨段总长趋近 dash（不含 AA 过渡）", () => {
        // 用极小的 aa 逼近"硬边"虚线，统计墨段长度应接近 dash × 周期数
        const tiny = 1e-6;
        const period = dash + gap;
        const cycles = 10;
        const steps = 20000;
        let ink = 0;
        for (let i = 0; i < steps; i += 1) {
            const along = (i / steps) * period * cycles;
            ink += dashCoverage(along, dash, gap, tiny);
        }
        const totalLength = period * cycles;
        const measured = (ink / steps) * totalLength;
        expect(measured).toBeCloseTo(dash * cycles, 1);
    });

    it("dash < 0 表示实线（恒满覆盖）", () => {
        for (const along of [0, 3, 7.25, 100]) {
            expect(dashCoverage(along, -1, 6, 0.5)).toBe(1);
        }
    });

    it("负弧长与超大弧长都安全（取模归一后仍在 [0,1]）", () => {
        for (const along of [-3, -1000, 1e9, 1e9 + 0.25]) {
            const c = dashCoverage(along, dash, gap, aa);
            expect(Number.isFinite(c)).toBe(true);
            expect(c).toBeGreaterThanOrEqual(0);
            expect(c).toBeLessThanOrEqual(1);
        }
        // 负弧长与对应正弧长等价（相位按周期归一）
        expect(dashCoverage(-3, dash, gap, aa)).toBeCloseTo(dashCoverage(9, dash, gap, aa), 9);
    });

    it("dash=0（全隙）恒为 0，dash+gap=0 不产生 NaN", () => {
        expect(dashCoverage(1, 0, 6, aa)).toBe(0);
        expect(Number.isFinite(dashCoverage(1, 0, 0, aa))).toBe(true);
    });
});
