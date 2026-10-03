import { describe, expect, it } from "vitest";

import {
    TOOLBAR_MAX_TIER,
    TOOLBAR_OVERFLOW_HYSTERESIS_PX,
    nextToolbarTier,
    type ToolbarRowMeasurement,
} from "./toolbarOverflow";

/** 便捷构造：`available` 固定、`needed` 可调。 */
function row(available: number, needed: number): ToolbarRowMeasurement {
    return { available, needed };
}

describe("nextToolbarTier", () => {
    it("没有测量结果时保持不动（首帧/未挂载）", () => {
        expect(nextToolbarTier({ currentTier: 0, maxTier: TOOLBAR_MAX_TIER, rows: [] })).toBe(0);
        expect(nextToolbarTier({ currentTier: 3, maxTier: TOOLBAR_MAX_TIER, rows: [] })).toBe(3);
    });

    it("内容放得下且未达回落余量时保持不动", () => {
        // available 1000、needed 990：没有溢出，但余量只有 10px（< 24px 滞后带）→ 停在原级。
        expect(
            nextToolbarTier({ currentTier: 2, maxTier: TOOLBAR_MAX_TIER, rows: [row(1000, 990)] }),
        ).toBe(2);
    });

    it("任一行溢出即升级一级", () => {
        expect(
            nextToolbarTier({ currentTier: 0, maxTier: TOOLBAR_MAX_TIER, rows: [row(500, 620)] }),
        ).toBe(1);
        // 两行里只要有一行溢出就升级（另一行宽裕不影响判据）。
        expect(
            nextToolbarTier({
                currentTier: 1,
                maxTier: TOOLBAR_MAX_TIER,
                rows: [row(900, 300), row(500, 620)],
            }),
        ).toBe(2);
    });

    it("每次最多升一级（不跳级），保证收敛而不震荡", () => {
        // 极端溢出也只 +1：隐藏一级后内容需求会下降，下一轮再决定。
        expect(
            nextToolbarTier({ currentTier: 0, maxTier: TOOLBAR_MAX_TIER, rows: [row(100, 9000)] }),
        ).toBe(1);
    });

    it("升到最大层级后不再升级", () => {
        expect(
            nextToolbarTier({
                currentTier: TOOLBAR_MAX_TIER,
                maxTier: TOOLBAR_MAX_TIER,
                rows: [row(100, 9000)],
            }),
        ).toBe(TOOLBAR_MAX_TIER);
    });

    it("所有行都留出余量才降一级", () => {
        expect(
            nextToolbarTier({ currentTier: 3, maxTier: TOOLBAR_MAX_TIER, rows: [row(1000, 400)] }),
        ).toBe(2);
    });

    it("只要有一行不够余量就不降级（避免来回切换）", () => {
        expect(
            nextToolbarTier({
                currentTier: 3,
                maxTier: TOOLBAR_MAX_TIER,
                rows: [row(1000, 400), row(1000, 990)],
            }),
        ).toBe(3);
    });

    it("已是最外层时不会降到负数", () => {
        expect(
            nextToolbarTier({ currentTier: 0, maxTier: TOOLBAR_MAX_TIER, rows: [row(1000, 400)] }),
        ).toBe(0);
    });

    it("滞后带内不动作：既不溢出、也不够余量 → 稳定", () => {
        // 滞后带 = (available - 24, available]；needed 落在这里时两个条件都不成立。
        const available = 1000;
        for (const needed of [available - 20, available - 1, available]) {
            expect(
                nextToolbarTier({
                    currentTier: 2,
                    maxTier: TOOLBAR_MAX_TIER,
                    rows: [row(available, needed)],
                }),
                `needed=${needed} 应停在原级`,
            ).toBe(2);
        }
    });

    it("刚好越过滞后带边界才降级", () => {
        const available = 1000;
        // needed = available - hysteresis 恰好命中 `<=` → 降级。
        expect(
            nextToolbarTier({
                currentTier: 2,
                maxTier: TOOLBAR_MAX_TIER,
                rows: [row(available, available - TOOLBAR_OVERFLOW_HYSTERESIS_PX)],
            }),
        ).toBe(1);
    });

    it("可自定义滞后余量", () => {
        expect(
            nextToolbarTier({
                currentTier: 2,
                maxTier: TOOLBAR_MAX_TIER,
                rows: [row(1000, 990)],
                hysteresisPx: 0,
            }),
        ).toBe(1);
    });

    it("模拟一次收窄—放宽往返：层级单调进出且回到原点", () => {
        const maxTier = TOOLBAR_MAX_TIER;
        let tier = 0;
        // 收窄：内容需求逐步超过可见宽度。
        for (const needed of [700, 900, 1200, 1600]) {
            tier = nextToolbarTier({ currentTier: tier, maxTier, rows: [row(800, needed)] });
        }
        expect(tier).toBeGreaterThan(0);

        // 放宽：内容需求逐步回落，最终必须回到 0（不会卡在某一级）。
        for (let i = 0; i < 40; i += 1) {
            tier = nextToolbarTier({ currentTier: tier, maxTier, rows: [row(2000, 400)] });
        }
        expect(tier).toBe(0);
    });
});
