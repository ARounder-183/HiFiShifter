/**
 * 参数编辑器内核 · 自绘滚动条几何装配单测。
 *
 * 【特殊说明】几何本身复用时间轴内核的 `computeScrollbar`（已在那边被单测覆盖），
 * 这里只验证**参数编辑器特有的装配语义**：水平轴用内容像素、竖向轴用值域范围。
 * 尤其是竖向——若误把竖向内容当成像素内容，thumb 长度会算错，用户拖拽时
 * 会发现「拖到底却没到底」，而这在类型上完全看不出来。
 */
import { describe, expect, it } from "vitest";

import { resolvePianoRollScrollbarGeometries } from "./scrollbarSpec";

describe("resolvePianoRollScrollbarGeometries", () => {
    it("两条轴都按视口尺寸与内容尺寸算出几何", () => {
        const { horizontal, vertical } = resolvePianoRollScrollbarGeometries({
            viewportWidthPx: 1000,
            viewportHeightPx: 600,
            contentWidthPx: 5000,
            scrollLeftPx: 0,
            scrollTopPx: 0,
            maxScrollLeftPx: 5000,
            maxScrollTopPx: 1600,
        });
        expect(horizontal.scrollable).toBe(true);
        expect(vertical.scrollable).toBe(true);
        // 水平：视口 1000 / 内容 5000 → thumb = 1000 * 200 = 200
        expect(horizontal.thumbLengthPx).toBeCloseTo(200, 6);
    });

    it("内容不足一屏时该轴不可滚动", () => {
        const { horizontal } = resolvePianoRollScrollbarGeometries({
            viewportWidthPx: 1000,
            viewportHeightPx: 600,
            contentWidthPx: 400,
            scrollLeftPx: 0,
            scrollTopPx: 0,
            maxScrollLeftPx: 0,
            maxScrollTopPx: 1600,
        });
        expect(horizontal.scrollable).toBe(false);
    });

    it("竖向用 1600px 值域范围而不是内容像素", () => {
        const { vertical } = resolvePianoRollScrollbarGeometries({
            viewportWidthPx: 1000,
            viewportHeightPx: 600,
            contentWidthPx: 5000,
            scrollLeftPx: 0,
            scrollTopPx: 800,
            maxScrollLeftPx: 5000,
            maxScrollTopPx: 1600,
            verticalContentSizePx: 1600 + 600,
        });
        // 内容 2200 / 视口 600 → thumb = 600 * (600/2200)
        expect(vertical.thumbLengthPx).toBeCloseTo(600 * (600 / 2200), 6);
    });

    it("竖向内容尺寸缺省时由「滚动上限 + 视口高度」推出（与显式传入等价）", () => {
        const args = {
            viewportWidthPx: 1000,
            viewportHeightPx: 600,
            contentWidthPx: 5000,
            scrollLeftPx: 0,
            scrollTopPx: 800,
            maxScrollLeftPx: 5000,
            maxScrollTopPx: 1600,
        };
        const implicit = resolvePianoRollScrollbarGeometries(args).vertical;
        const explicit = resolvePianoRollScrollbarGeometries({
            ...args,
            verticalContentSizePx: 1600 + 600,
        }).vertical;
        expect(implicit.thumbLengthPx).toBeCloseTo(explicit.thumbLengthPx, 6);
        expect(implicit.thumbStartPx).toBeCloseTo(explicit.thumbStartPx, 6);
    });

    it("竖向 thumb 位置随 scrollTop 单调右移（下移），不反向", () => {
        const at = (scrollTopPx: number) =>
            resolvePianoRollScrollbarGeometries({
                viewportWidthPx: 1000,
                viewportHeightPx: 600,
                contentWidthPx: 5000,
                scrollLeftPx: 0,
                scrollTopPx,
                maxScrollLeftPx: 5000,
                maxScrollTopPx: 1600,
            }).vertical.thumbStartPx;
        expect(at(0)).toBeLessThan(at(800));
        expect(at(800)).toBeLessThan(at(1600));
    });
});
