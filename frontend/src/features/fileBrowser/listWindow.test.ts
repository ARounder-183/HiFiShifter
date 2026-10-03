import { describe, expect, it } from "vitest";

import { FALLBACK_VIEWPORT_PX, LIST_OVERSCAN, computeListWindow } from "./listWindow";

const ROW = 22;
const VIEWPORT = 440; // 20 行

describe("computeListWindow", () => {
    it("空列表返回空窗口", () => {
        expect(
            computeListWindow({ total: 0, rowHeight: ROW, scrollTop: 0, viewportHeight: VIEWPORT }),
        ).toEqual({ first: 0, last: 0, totalHeight: 0, offsetTop: 0 });
    });

    it("顶部：从第 0 行开始，只渲染视口 + 缓冲", () => {
        const w = computeListWindow({
            total: 20000,
            rowHeight: ROW,
            scrollTop: 0,
            viewportHeight: VIEWPORT,
        });
        expect(w.first).toBe(0);
        // 20 行可见 + 8 行下缓冲。
        expect(w.last).toBe(20 + LIST_OVERSCAN);
        expect(w.offsetTop).toBe(0);
        expect(w.totalHeight).toBe(20000 * ROW);
    });

    it("渲染行数与列表总量无关（这是窗口化的全部意义）", () => {
        for (const total of [50, 500, 20000, 200000]) {
            const w = computeListWindow({
                total,
                rowHeight: ROW,
                scrollTop: ROW * 1000,
                viewportHeight: VIEWPORT,
            });
            expect(w.last - w.first).toBeLessThanOrEqual(20 + LIST_OVERSCAN * 2);
        }
    });

    it("中部：窗口跟随滚动位置，上下都有缓冲", () => {
        const w = computeListWindow({
            total: 20000,
            rowHeight: ROW,
            scrollTop: ROW * 1000,
            viewportHeight: VIEWPORT,
        });
        expect(w.first).toBe(1000 - LIST_OVERSCAN);
        expect(w.last).toBe(1000 + 20 + LIST_OVERSCAN);
        expect(w.offsetTop).toBe(w.first * ROW);
    });

    it("底部：末行不越界，且最后一行一定被包含", () => {
        const total = 100;
        const maxScroll = total * ROW - VIEWPORT;
        const w = computeListWindow({
            total,
            rowHeight: ROW,
            scrollTop: maxScroll,
            viewportHeight: VIEWPORT,
        });
        expect(w.last).toBe(total);
        expect(w.first).toBeLessThan(total);
        // 滚动到最底时最后一行必须在窗口内。
        expect(w.last - w.first).toBeGreaterThanOrEqual(total - w.first);
    });

    it("负 scrollTop（回弹）按 0 处理", () => {
        const w = computeListWindow({
            total: 100,
            rowHeight: ROW,
            scrollTop: -300,
            viewportHeight: VIEWPORT,
        });
        expect(w.first).toBe(0);
    });

    it("视口尚未测量（高度 0）时用估值，而不是渲染空窗", () => {
        const w = computeListWindow({
            total: 20000,
            rowHeight: ROW,
            scrollTop: 0,
            viewportHeight: 0,
        });
        expect(w.last - w.first).toBeGreaterThanOrEqual(Math.ceil(FALLBACK_VIEWPORT_PX / ROW));
    });

    it("行高为 0 的首帧只渲染一小段，绝不铺开全量", () => {
        const w = computeListWindow({
            total: 20000,
            rowHeight: 0,
            scrollTop: 0,
            viewportHeight: VIEWPORT,
        });
        expect(w.last).toBeLessThanOrEqual(LIST_OVERSCAN * 2);
        expect(w.last).toBeGreaterThan(0);
    });

    it("窗口永不为空（否则列表整屏空白）", () => {
        for (const scrollTop of [0, 1, 21, 22, 23, 999999]) {
            const w = computeListWindow({
                total: 3,
                rowHeight: ROW,
                scrollTop,
                viewportHeight: VIEWPORT,
            });
            expect(w.last).toBeGreaterThan(w.first);
        }
    });
});
