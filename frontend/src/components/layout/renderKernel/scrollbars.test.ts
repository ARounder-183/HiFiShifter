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

import {
    computeScrollbar,
    hitTestScrollbarThumb,
    scrollDeltaFromThumbDrag,
    scrollTargetFromTrackClick,
} from "./scrollbars";

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

describe("scrollTargetFromTrackClick", () => {
    // 内容 2000 / 视口 500 / 轨道 500 → thumb 长 125；滚到一半 → 起点 187.5，终点 312.5。
    const geometry = computeScrollbar({
        contentSizePx: 2000,
        viewportSizePx: 500,
        scrollPx: 750,
        maxScrollPx: 1500,
    });

    it("点在 thumb 之后 → 向后翻一页", () => {
        expect(scrollTargetFromTrackClick(400, geometry, 750, 500)).toBe(1250);
    });

    it("点在 thumb 之前 → 向前翻一页", () => {
        expect(scrollTargetFromTrackClick(50, geometry, 750, 500)).toBe(250);
    });

    it("点在 thumb 上（含两端边界）→ 不跳转", () => {
        // 那是拖拽起点，必须返回 null，否则一次 thumb 拖拽会被叠加一次翻页。
        expect(scrollTargetFromTrackClick(187.5, geometry, 750, 500)).toBe(null);
        expect(scrollTargetFromTrackClick(250, geometry, 750, 500)).toBe(null);
        expect(scrollTargetFromTrackClick(312.5, geometry, 750, 500)).toBe(null);
        // 紧贴两端之外的一像素即视为「轨道」：边界是闭区间，只有真正落在外侧才翻页。
        expect(scrollTargetFromTrackClick(186, geometry, 750, 500)).toBe(250);
        expect(scrollTargetFromTrackClick(314, geometry, 750, 500)).toBe(1250);
    });

    it("不可滚动时恒为 null", () => {
        const flat = computeScrollbar({
            contentSizePx: 100,
            viewportSizePx: 500,
            scrollPx: 0,
            maxScrollPx: 0,
        });
        expect(scrollTargetFromTrackClick(400, flat, 0, 500)).toBe(null);
    });

    it("页长非法（0 / NaN / 负数）时不跳转", () => {
        expect(scrollTargetFromTrackClick(400, geometry, 750, 0)).toBe(null);
        expect(scrollTargetFromTrackClick(400, geometry, 750, Number.NaN)).toBe(null);
        expect(scrollTargetFromTrackClick(400, geometry, 750, -100)).toBe(null);
    });

    it("非法指针 / 当前滚动值按 0 处理，不产生 NaN", () => {
        expect(scrollTargetFromTrackClick(Number.NaN, geometry, 750, 500)).toBe(250);
        const result = scrollTargetFromTrackClick(400, geometry, Number.NaN, 500);
        expect(Number.isFinite(result as number)).toBe(true);
    });

    it("结果可能越界，由调用方（ScrollKernel）钳制", () => {
        // 滚动已在顶部时点上方：结果 -500 —— 本函数只做算术，钳制是单一上限来源。
        expect(scrollTargetFromTrackClick(50, geometry, 0, 500)).toBe(-500);
    });
});
