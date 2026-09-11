/**
 * 自绘滚动条几何（./scrollbars）行为自检。
 *
 * 【主要内容】
 * 1. 可滚动时 thumb 长度按视口 / 内容比例、起点按滚动比例映射；
 * 2. thumb 长度不低于最小值、不超出轨道；起点不越出可用行程；
 * 3. 内容不足一屏（maxScroll = 0）时不可滚动、thumb 占满轨道；
 * 4. 非法输入不产生 NaN 几何；
 * 5. thumb 命中测试与拖拽位移换算。
 *
 * 【作用】滚动条几何是自绘滚动里最容易出 NaN / 除零的地方（内容不足一屏、
 * 视口为 0、上限为 0），本文件把这些边界锁死。
 *
 * 【与其他模块的关系】覆盖 `scrollbars.ts`；不依赖 DOM。
 */

import { describe, expect, it } from "vitest";

import { computeScrollbar, hitTestScrollbarThumb, scrollDeltaFromThumbDrag } from "./scrollbars";

describe("computeScrollbar", () => {
    it("可滚动时按比例计算 thumb 长度与起点", () => {
        // 内容 2000、视口 500、轨道 500 → thumb 长 125；滚到一半 → 起点 (500-125)/2
        const geometry = computeScrollbar({
            contentSizePx: 2000,
            viewportSizePx: 500,
            scrollPx: 750,
            maxScrollPx: 1500,
        });
        expect(geometry.scrollable).toBe(true);
        expect(geometry.thumbLengthPx).toBe(125);
        expect(geometry.thumbStartPx).toBeCloseTo(187.5, 6);
    });

    it("thumb 长度不低于最小值", () => {
        const geometry = computeScrollbar({
            contentSizePx: 100000,
            viewportSizePx: 500,
            scrollPx: 0,
            maxScrollPx: 99500,
            minThumbLengthPx: 24,
        });
        expect(geometry.thumbLengthPx).toBe(24);
    });

    it("内容不足一屏时不可滚动且 thumb 占满轨道", () => {
        const geometry = computeScrollbar({
            contentSizePx: 300,
            viewportSizePx: 500,
            scrollPx: 0,
            maxScrollPx: 0,
        });
        expect(geometry.scrollable).toBe(false);
        expect(geometry.thumbStartPx).toBe(0);
        expect(geometry.thumbLengthPx).toBe(500);
    });

    it("滚动位置越界时 thumb 起点被夹取在可用行程内", () => {
        const geometry = computeScrollbar({
            contentSizePx: 2000,
            viewportSizePx: 500,
            scrollPx: 999999,
            maxScrollPx: 1500,
        });
        expect(geometry.thumbStartPx).toBeCloseTo(375, 6);
    });

    it("非法输入不产生 NaN 几何", () => {
        const geometry = computeScrollbar({
            contentSizePx: Number.NaN,
            viewportSizePx: Number.NaN,
            scrollPx: Number.NaN,
            maxScrollPx: Number.NaN,
        });
        expect(geometry.scrollable).toBe(false);
        expect(Number.isFinite(geometry.thumbStartPx)).toBe(true);
        expect(Number.isFinite(geometry.thumbLengthPx)).toBe(true);
    });
});

describe("hitTestScrollbarThumb", () => {
    it("命中 thumb 区间（含边界）", () => {
        const geometry = computeScrollbar({
            contentSizePx: 2000,
            viewportSizePx: 500,
            scrollPx: 0,
            maxScrollPx: 1500,
        });
        expect(hitTestScrollbarThumb(0, geometry)).toBe(true);
        expect(hitTestScrollbarThumb(125, geometry)).toBe(true);
        expect(hitTestScrollbarThumb(126, geometry)).toBe(false);
    });

    it("不可滚动时恒不命中", () => {
        const geometry = computeScrollbar({
            contentSizePx: 100,
            viewportSizePx: 500,
            scrollPx: 0,
            maxScrollPx: 0,
        });
        expect(hitTestScrollbarThumb(0, geometry)).toBe(false);
    });
});

describe("scrollDeltaFromThumbDrag", () => {
    it("按可用行程比例换算滚动增量", () => {
        const geometry = computeScrollbar({
            contentSizePx: 2000,
            viewportSizePx: 500,
            scrollPx: 0,
            maxScrollPx: 1500,
        });
        // 轨道可用行程 = 500 - 125 = 375；拖动 37.5px → 10% × 1500 = 150
        expect(scrollDeltaFromThumbDrag(37.5, geometry, 1500)).toBeCloseTo(150, 6);
    });

    it("不可滚动时返回 0", () => {
        const geometry = computeScrollbar({
            contentSizePx: 100,
            viewportSizePx: 500,
            scrollPx: 0,
            maxScrollPx: 0,
        });
        expect(scrollDeltaFromThumbDrag(50, geometry, 0)).toBe(0);
    });
});
