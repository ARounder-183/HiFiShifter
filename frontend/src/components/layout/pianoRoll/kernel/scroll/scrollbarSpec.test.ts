/**
 * 参数编辑器内核 · 自绘滚动条几何装配单测。
 *
 * 【特殊说明】几何本身复用时间轴内核的 `computeScrollbar`（已在那边被单测覆盖），
 * 这里只验证**参数编辑器特有的装配语义**：两条轴的内容尺寸口径，以及位置用绘制坐标。
 *
 * 【为什么重点测内容尺寸】原生的 thumb 长度是 `视口² / scrollWidth`，而
 * `scrollWidth = 内容宽 + 视口宽`。若误把「内容宽」当内容尺寸，thumb 会偏长——
 * 这在类型上完全看不出来，只有和旧实现的原生滚动条比长度才会暴露。下面的
 * 「实测值」用例直接钉住 macOS 上量到的两个 thumb 长度。
 */
import { describe, expect, it } from "vitest";

import { resolvePianoRollScrollbarGeometries } from "./scrollbarSpec";

describe("resolvePianoRollScrollbarGeometries", () => {
    it("水平 thumb 用原生 scrollWidth 口径（实测 316.18，而非内容宽算出的 380.77）", () => {
        const { horizontal } = resolvePianoRollScrollbarGeometries({
            viewportWidthPx: 1864,
            viewportHeightPx: 823,
            scrollLeftPx: 0,
            scrollTopPx: 0,
            // 原生域：内容宽 8925 + 同步偏移 200 = 9125
            maxScrollLeftPx: 9125,
            maxScrollTopPx: 1600,
        });
        // scrollWidth = 9125 + 1864 = 10989 → thumb = 1864² / 10989
        expect(horizontal.thumbLengthPx).toBeCloseTo((1864 * 1864) / 10989, 6);
        // 反例：若误用内容宽 8925 当内容尺寸，会得到明显更长的 thumb。
        expect(horizontal.thumbLengthPx).not.toBeCloseTo((1864 * 1864) / 8925, 3);
    });

    it("竖直 thumb 同为「上限 + 视口」口径（实测 279.54）", () => {
        const { vertical } = resolvePianoRollScrollbarGeometries({
            viewportWidthPx: 1864,
            viewportHeightPx: 823,
            scrollLeftPx: 0,
            scrollTopPx: 0,
            maxScrollLeftPx: 9125,
            maxScrollTopPx: 1600,
        });
        // 内容 1600 + 823 = 2423 → thumb = 823² / 2423
        expect(vertical.thumbLengthPx).toBeCloseTo((823 * 823) / 2423, 6);
    });

    it("两轴可滚动性按各自上限判定", () => {
        const both = resolvePianoRollScrollbarGeometries({
            viewportWidthPx: 1000,
            viewportHeightPx: 600,
            scrollLeftPx: 0,
            scrollTopPx: 0,
            maxScrollLeftPx: 5000,
            maxScrollTopPx: 1600,
        });
        expect(both.horizontal.scrollable).toBe(true);
        expect(both.vertical.scrollable).toBe(true);

        const noHorizontal = resolvePianoRollScrollbarGeometries({
            viewportWidthPx: 1000,
            viewportHeightPx: 600,
            scrollLeftPx: 0,
            scrollTopPx: 0,
            maxScrollLeftPx: 0,
            maxScrollTopPx: 1600,
        });
        expect(noHorizontal.horizontal.scrollable).toBe(false);
    });

    it("竖向 thumb 位置随 scrollTop 单调下移，不反向", () => {
        const at = (scrollTopPx: number) =>
            resolvePianoRollScrollbarGeometries({
                viewportWidthPx: 1000,
                viewportHeightPx: 600,
                scrollLeftPx: 0,
                scrollTopPx,
                maxScrollLeftPx: 5000,
                maxScrollTopPx: 1600,
            }).vertical.thumbStartPx;
        expect(at(0)).toBeLessThan(at(800));
        expect(at(800)).toBeLessThan(at(1600));
    });

    it("显式传入的内容尺寸优先于缺省推导", () => {
        const explicit = resolvePianoRollScrollbarGeometries({
            viewportWidthPx: 1000,
            viewportHeightPx: 600,
            scrollLeftPx: 0,
            scrollTopPx: 0,
            maxScrollLeftPx: 5000,
            maxScrollTopPx: 1600,
            horizontalContentSizePx: 2000,
        });
        // 2000 < 视口 1000*2？不：内容 2000 > 视口 1000 → thumb = 1000²/2000 = 500
        expect(explicit.horizontal.thumbLengthPx).toBeCloseTo(500, 6);
    });

    /**
     * 竖向值域口径（用户报告"竖直 thumb 长度不随缩放变化、缩到最小不隐藏"）：
     * 传入 valueDomain 后 thumb 长度 = 轨道 × span / range，与原生滚动条
     * "可见占比决定 thumb 长度"同构；缩到最小（span ≥ range）时不可滚动。
     */
    describe("竖向值域口径（verticalValueDomain）", () => {
        const base = {
            viewportWidthPx: 1000,
            viewportHeightPx: 600,
            scrollLeftPx: 0,
            maxScrollLeftPx: 5000,
            maxScrollTopPx: 1600,
        };

        it("thumb 长度按 span/range 占比伸缩（zoom in → 变短，zoom out → 变长）", () => {
            const range = 60;
            const at = (span: number) =>
                resolvePianoRollScrollbarGeometries({
                    ...base,
                    scrollTopPx: 0,
                    verticalValueDomain: { min: 0, max: range, span },
                }).vertical;
            // 音高默认视口：span 24 / 全长 60 → thumb = 600 × 24/60 = 240。
            expect(at(24).thumbLengthPx).toBeCloseTo(240, 6);
            // 放大一倍（span 12）→ thumb 减半。
            expect(at(12).thumbLengthPx).toBeCloseTo(120, 6);
            // 缩小一倍（span 48）→ thumb 翻倍。
            expect(at(48).thumbLengthPx).toBeCloseTo(480, 6);
            expect(at(12).scrollable).toBe(true);
        });

        it("缩放到最小（span ≥ range）时不可滚动（宿主据此隐藏 thumb）", () => {
            const { vertical } = resolvePianoRollScrollbarGeometries({
                ...base,
                scrollTopPx: 0,
                verticalValueDomain: { min: 0, max: 60, span: 60 },
            });
            expect(vertical.scrollable).toBe(false);
        });

        it("thumb 起点仍按 scrollTop/maxScrollTop 在行程内线性映射（手感不变）", () => {
            const range = 60;
            const at = (scrollTopPx: number) =>
                resolvePianoRollScrollbarGeometries({
                    ...base,
                    viewportHeightPx: 300,
                    scrollTopPx,
                    verticalValueDomain: { min: 0, max: range, span: 30 },
                }).vertical;
            // span 30/60 → thumb 150、行程 150；scrollTop=800/1600 → 起点 75。
            expect(at(800).thumbLengthPx).toBeCloseTo(150, 6);
            expect(at(800).thumbStartPx).toBeCloseTo(75, 6);
            expect(at(0).thumbStartPx).toBeCloseTo(0, 6);
            expect(at(1600).thumbStartPx).toBeCloseTo(150, 6);
        });

        it("非法值域（range ≤ 0 / span ≤ 0 / 非有限）回退旧口径，不产生 NaN", () => {
            for (const domain of [
                { min: 5, max: 5, span: 1 },
                { min: 0, max: 60, span: 0 },
                { min: Number.NaN, max: 60, span: 10 },
            ]) {
                const { vertical } = resolvePianoRollScrollbarGeometries({
                    ...base,
                    viewportHeightPx: 823,
                    scrollTopPx: 0,
                    verticalValueDomain: domain,
                });
                // 旧口径：内容 1600 + 823 = 2423 → thumb = 823² / 2423。
                expect(vertical.thumbLengthPx).toBeCloseTo((823 * 823) / 2423, 6);
            }
        });
    });
});
