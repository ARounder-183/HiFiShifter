import { describe, expect, it } from "vitest";

import { TOOLBAR_MAX_TIER } from "./toolbarOverflow";
import { chooseTier } from "./toolbarGauge";

/**
 * 各层级**需要**的宽度：藏得越多越窄。与"可用宽度"无关地固定。
 *
 * 数值来自真实工具栏的实测（tier 9 的下界 772px 可直接从 DOM 量到），但本用例不关心
 * 具体数字 —— 它钉的是"怎么从一张表里选层级"这条规则。
 */
const NEEDED: readonly number[] = [1200, 1140, 1110, 1004, 954, 834, 794, 724, 664, 620];

/** 把"需要多宽"翻译成"这个宽度下放不放得下"—— 与 `measureTierFits` 的输出同形状。 */
function fitsAt(available: number): boolean[] {
    return NEEDED.map((needed) => needed <= available);
}

describe("chooseTier", () => {
    it("放得下就用第 0 级（全部显示）", () => {
        expect(chooseTier(fitsAt(5000))).toBe(0);
    });

    it("取最小的、放得下的那一级（不逐级收敛，一次到位）", () => {
        // 1000 放不下 0/1/2/3 级（1200/1140/1110/1004），第 4 级 954 可以。
        expect(chooseTier(fitsAt(1000))).toBe(4);
    });

    it("恰好等于时可放行（判据是 <= ）", () => {
        expect(chooseTier(fitsAt(1110))).toBe(2);
    });

    it("全都放不下时停在最大级（由 overflow 裁掉，而不是永远收缩）", () => {
        expect(chooseTier(fitsAt(100))).toBe(TOOLBAR_MAX_TIER);
    });

    it("空表（量不出来）时退回第 0 级", () => {
        expect(chooseTier([])).toBe(0);
    });

    it("压缩与恢复收敛到同一层级（同一宽度只有一个答案）", () => {
        const available = 1000;
        const expected = NEEDED.findIndex((needed) => needed <= available);
        expect(expected).toBeGreaterThan(0);
        // 无论从"全部显示"出发还是从"全藏"出发，答案只由这张表决定。
        expect(chooseTier(fitsAt(available))).toBe(expected);
        expect(chooseTier(fitsAt(available))).toBe(expected);
        // 略宽一点点就应当回到更少的隐藏层级（没有滞后余量）。
        const wider = NEEDED[expected - 1] ?? Number.POSITIVE_INFINITY;
        expect(chooseTier(fitsAt(wider))).toBe(expected - 1);
    });

    it("表单调不降时，答案随可用宽度单调不增（宽度变大不会藏更多）", () => {
        let previous = Number.POSITIVE_INFINITY;
        for (let available = 600; available <= 1300; available += 5) {
            const tier = chooseTier(fitsAt(available));
            expect(tier).toBeLessThanOrEqual(previous);
            previous = tier;
        }
    });
});
