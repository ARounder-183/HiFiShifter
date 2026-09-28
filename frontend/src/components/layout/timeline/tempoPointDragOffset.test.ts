/**
 * 变化点拖拽阈值自检（"双击后输入框偏移"的回归）。
 *
 * 关键性质：阈值内的位移一律折算为 0（点击不改变变化点）；越过阈值后从阈值处
 * 起算，因此不会出现"刚越过就跳一格"。
 */
import { describe, expect, it } from "vitest";

import { resolveTempoDragOffsetPx, TEMPO_DRAG_THRESHOLD_PX } from "./tempoPointDragOffset";

describe("resolveTempoDragOffsetPx（拖拽启动阈值）", () => {
    it("★ 阈值内的抖动折算为 0（双击的第一下不会挪动变化点）", () => {
        for (const dx of [0, 1, -1, 2, -2, TEMPO_DRAG_THRESHOLD_PX, -TEMPO_DRAG_THRESHOLD_PX]) {
            expect(resolveTempoDragOffsetPx(dx)).toBe(0);
        }
    });

    it("越过阈值后从阈值处起算（不跳一格）", () => {
        expect(resolveTempoDragOffsetPx(TEMPO_DRAG_THRESHOLD_PX + 1)).toBe(1);
        expect(resolveTempoDragOffsetPx(TEMPO_DRAG_THRESHOLD_PX + 10)).toBe(10);
        expect(resolveTempoDragOffsetPx(-TEMPO_DRAG_THRESHOLD_PX - 7)).toBe(-7);
    });

    it("位移连续（阈值两侧不出现跳变）", () => {
        const below = resolveTempoDragOffsetPx(TEMPO_DRAG_THRESHOLD_PX);
        const above = resolveTempoDragOffsetPx(TEMPO_DRAG_THRESHOLD_PX + 0.001);
        expect(Math.abs(above - below)).toBeLessThan(0.01);
    });

    it("非有限位移折算为 0（不把 NaN 带进位置计算）", () => {
        expect(resolveTempoDragOffsetPx(Number.NaN)).toBe(0);
        expect(resolveTempoDragOffsetPx(Number.POSITIVE_INFINITY)).toBe(0);
    });
});
