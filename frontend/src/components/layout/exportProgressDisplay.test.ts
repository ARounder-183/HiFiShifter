/**
 * 导出进度显示推进（./exportProgressDisplay）行为自检。
 *
 * 【主要内容】
 * 1. 有真实目标时单调逼近、不越过、不回退；
 * 2. 无真实目标时缓慢爬升且不超过封顶；
 * 3. 换算与快照相等判定的边界（非法值、null）。
 *
 * 【作用】钉住"显示值只升不降、永不偏离真实进度"这两条保证 —— 它们的回归会直接
 * 表现为用户可见的进度条冻结或跳变（2026-09 报告的"进度条不更新"）。
 *
 * 【与其他模块的关系】覆盖 `exportProgressDisplay.ts`；不依赖 DOM / React。
 */

import { describe, expect, it } from "vitest";

import {
    CONVERGE_PERCENT_PER_SEC,
    DISPLAY_TICK_MS,
    FAKE_PROGRESS_CAP,
    isSameExportProgress,
    nextDisplayProgress,
    realProgressPercent,
} from "./exportProgressDisplay";

const STEP = (CONVERGE_PERCENT_PER_SEC * DISPLAY_TICK_MS) / 1000;

describe("nextDisplayProgress", () => {
    it("有真实目标：按收敛速度前进，且不越过目标", () => {
        const next = nextDisplayProgress({ current: 0, target: 100, tickMs: DISPLAY_TICK_MS });
        expect(next).toBeGreaterThan(0);
        expect(next).toBeLessThanOrEqual(STEP + 1e-9);

        // 距目标不足一步 → 恰好落在目标上（不越过）。
        const near = nextDisplayProgress({
            current: 99,
            target: 100,
            tickMs: DISPLAY_TICK_MS,
        });
        expect(near).toBe(100);
    });

    it("显示值已达到或超过目标：保持不变（不回退）", () => {
        expect(nextDisplayProgress({ current: 60, target: 60, tickMs: DISPLAY_TICK_MS })).toBe(60);
        // 理论上不应发生（后端进度单调），但绝不回退。
        expect(nextDisplayProgress({ current: 80, target: 40, tickMs: DISPLAY_TICK_MS })).toBe(80);
    });

    it("无真实目标：缓慢爬升且不超过封顶", () => {
        expect(
            nextDisplayProgress({ current: 0, target: null, tickMs: DISPLAY_TICK_MS }),
        ).toBeGreaterThan(0);
        expect(
            nextDisplayProgress({
                current: FAKE_PROGRESS_CAP,
                target: null,
                tickMs: DISPLAY_TICK_MS,
            }),
        ).toBe(FAKE_PROGRESS_CAP);
        // 单帧增幅很小（明显慢于真实收敛速度）。
        const creep = nextDisplayProgress({ current: 0, target: null, tickMs: DISPLAY_TICK_MS });
        expect(creep).toBeLessThan(STEP);
    });

    it("真实目标接管后不再走假进度分支", () => {
        // 假进度停在 5% 时真实进度只有 1%：目标存在即不去爬 90，而是保持不回退。
        const next = nextDisplayProgress({ current: 5, target: 1, tickMs: DISPLAY_TICK_MS });
        expect(next).toBe(5);
    });

    it("恒在 [0, 100]，且不小于当前值", () => {
        for (const current of [-10, 0, 42, 100, 150, Number.NaN]) {
            for (const target of [null, -5, 0, 33.3, 100, 200]) {
                const next = nextDisplayProgress({ current, target, tickMs: DISPLAY_TICK_MS });
                expect(next).toBeGreaterThanOrEqual(
                    Math.max(0, Math.min(100, Number.isFinite(current) ? current : 0)),
                );
                expect(next).toBeLessThanOrEqual(100);
            }
        }
    });

    it("tickMs 非法（0 / NaN / 负数）时保持当前值", () => {
        expect(nextDisplayProgress({ current: 30, target: 90, tickMs: 0 })).toBe(30);
        expect(nextDisplayProgress({ current: 30, target: 90, tickMs: Number.NaN })).toBe(30);
        expect(nextDisplayProgress({ current: 30, target: null, tickMs: -5 })).toBe(30);
    });
});

describe("realProgressPercent", () => {
    it("0..1 换算为 0..100 并夹取", () => {
        expect(realProgressPercent(0)).toBe(0);
        expect(realProgressPercent(0.5)).toBe(50);
        expect(realProgressPercent(1)).toBe(100);
        expect(realProgressPercent(1.5)).toBe(100);
        expect(realProgressPercent(-0.25)).toBe(0);
    });

    it("null / 非有限值返回 null（= 尚无真实进度）", () => {
        expect(realProgressPercent(null)).toBeNull();
        expect(realProgressPercent(Number.NaN)).toBeNull();
        expect(realProgressPercent(Number.POSITIVE_INFINITY)).toBeNull();
    });
});

describe("isSameExportProgress", () => {
    const base = { active: true, mode: "project", progress: 0.5, current: 1, total: 1 };

    it("字段逐一相等时为 true（用于跳过无变化的重渲染）", () => {
        expect(isSameExportProgress(base, { ...base })).toBe(true);
    });

    it("任一字段变化即为 false", () => {
        expect(isSameExportProgress(base, { ...base, progress: 0.6 })).toBe(false);
        expect(isSameExportProgress(base, { ...base, active: false })).toBe(false);
        expect(isSameExportProgress(base, { ...base, current: 2 })).toBe(false);
        expect(isSameExportProgress(base, { ...base, mode: "separated" })).toBe(false);
    });
});
