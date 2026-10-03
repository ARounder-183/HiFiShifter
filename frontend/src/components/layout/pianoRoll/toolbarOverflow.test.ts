import { describe, expect, it } from "vitest";

import { TOOLBAR_MAX_TIER, nextToolbarTier } from "./toolbarOverflow";

const MAX = TOOLBAR_MAX_TIER;

/** 便捷调用：只关心 available/needed。 */
function tier(params: {
    currentTier: number;
    available: number;
    needed: number;
    neededAtLowerTier?: number;
}): number {
    return nextToolbarTier({ maxTier: MAX, ...params });
}

describe("nextToolbarTier", () => {
    it("放得下且上一级放不下时保持不动（不动点）", () => {
        expect(
            tier({ currentTier: 2, available: 1000, needed: 900, neededAtLowerTier: 1100 }),
        ).toBe(2);
    });

    it("放不下就多隐藏一级", () => {
        expect(tier({ currentTier: 0, available: 900, needed: 1000 })).toBe(1);
    });

    it("每次最多升一级（不跳级），保证收敛而不震荡", () => {
        expect(tier({ currentTier: 0, available: 100, needed: 9000 })).toBe(1);
    });

    it("到最大级后不再升级", () => {
        expect(tier({ currentTier: MAX, available: 100, needed: 9000 })).toBe(MAX);
    });

    it("少隐藏一级也放得下就恢复一级", () => {
        expect(tier({ currentTier: 3, available: 1000, needed: 900, neededAtLowerTier: 950 })).toBe(
            2,
        );
    });

    it("上一级放不下时不恢复", () => {
        expect(
            tier({ currentTier: 3, available: 1000, needed: 900, neededAtLowerTier: 1001 }),
        ).toBe(3);
    });

    it("上一级恰好等于可见宽度时恢复（判据是 `<=`）", () => {
        expect(
            tier({ currentTier: 3, available: 1000, needed: 900, neededAtLowerTier: 1000 }),
        ).toBe(2);
    });

    it("未测过上一级需求宽度时不恢复（保守）", () => {
        expect(tier({ currentTier: 3, available: 2000, needed: 900 })).toBe(3);
    });

    it("已在最外层时不会降到负数", () => {
        expect(
            tier({ currentTier: 0, available: 2000, needed: 900, neededAtLowerTier: 1000 }),
        ).toBe(0);
    });

    it("同一宽度下压缩与恢复收敛到同一层级（对称）", () => {
        // 各层级的内容需求宽度（隐藏越多越窄），与"宽度"无关地固定。
        const neededByTier = [1200, 1140, 1110, 1004, 954, 834, 794, 724, 664];
        const available = 1000;
        // 该宽度下"刚好放得下"的最小层级。
        const expected = neededByTier.findIndex((n) => n <= available);
        expect(expected).toBeGreaterThan(0);

        // 压缩：从 0 出发，反复"放不下就藏"。
        let down = 0;
        for (let i = 0; i < 20; i += 1) {
            down = tier({
                currentTier: down,
                available,
                needed: neededByTier[down],
                neededAtLowerTier: down > 0 ? neededByTier[down - 1] : undefined,
            });
        }
        expect(down).toBe(expected);

        // 恢复：从最深层出发，反复"上一级放得下就恢复"。
        let up = MAX;
        for (let i = 0; i < 20; i += 1) {
            up = tier({
                currentTier: up,
                available,
                needed: neededByTier[up],
                neededAtLowerTier: up > 0 ? neededByTier[up - 1] : undefined,
            });
        }
        expect(up).toBe(expected);
    });
});
