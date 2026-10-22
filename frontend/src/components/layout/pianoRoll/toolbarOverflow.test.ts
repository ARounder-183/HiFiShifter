import { describe, expect, it } from "vitest";

import { TOOLBAR_MAX_TIER } from "./toolbarOverflow";
import { chooseTier } from "./toolbarGauge";

/** 各层级的内容宽度：藏得越多越窄。与"可用宽度"无关地固定。 */
const NEEDED: readonly number[] = [1200, 1140, 1110, 1004, 954, 834, 794, 724, 664, 620];

describe("chooseTier", () => {
    it("放得下就用第 0 级（全部显示）", () => {
        expect(chooseTier(NEEDED, 5000)).toBe(0);
    });

    it("取最小的、放得下的那一级（不逐级收敛，一次到位）", () => {
        // 1000 放不下 0/1/2/3 级（1200/1140/1110/1004），第 4 级 954 可以。
        expect(chooseTier(NEEDED, 1000)).toBe(4);
    });

    it("恰好等于时可放行（判据是 <= ）", () => {
        expect(chooseTier(NEEDED, 1110)).toBe(2);
    });

    it("全都放不下时停在最大级（由 overflow 裁掉，而不是永远收缩）", () => {
        expect(chooseTier(NEEDED, 100)).toBe(TOOLBAR_MAX_TIER);
    });

    it("宽度为 0 / 负数时按最大级处理（面板当前不可见时的保守值）", () => {
        expect(chooseTier(NEEDED, 0)).toBe(TOOLBAR_MAX_TIER);
        expect(chooseTier(NEEDED, -10)).toBe(TOOLBAR_MAX_TIER);
    });

    it("空表（量不出来）时退回第 0 级", () => {
        expect(chooseTier([], 800)).toBe(0);
    });

    it("压缩与恢复收敛到同一层级（同一宽度只有一个答案）", () => {
        const available = 1000;
        const expected = NEEDED.findIndex((needed) => needed <= available);
        expect(expected).toBeGreaterThan(0);
        // 无论从"全部显示"出发还是从"全藏"出发，答案只由宽度表与可用宽度决定。
        expect(chooseTier(NEEDED, available)).toBe(expected);
        expect(chooseTier(NEEDED, available)).toBe(expected);
        // 略宽一点点就应当回到更少的隐藏层级（没有滞后余量）。
        const wider = NEEDED[expected - 1] ?? Number.POSITIVE_INFINITY;
        expect(chooseTier(NEEDED, wider)).toBe(expected - 1);
    });
});
