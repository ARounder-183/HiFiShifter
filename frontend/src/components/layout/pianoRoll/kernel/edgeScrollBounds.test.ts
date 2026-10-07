/**
 * 边缘自动滚屏的**上下界必须与内核真值一致**（回归锁）。
 *
 * 【为什么单列一个跨模块的测试】"闪现而不是滚动"这类缺陷的成因不是公式算错，而是
 * **两套边界不一致**：驱动按一个上界写入、内核按自己的上界回写，视图每帧在两处
 * 往复。只测公式（见 `shared/edgeAutoScroll.test.ts`）抓不到它 —— 必须拿驱动的边界
 * 与**内核实际接受**的边界对照。
 *
 * 【本测试钉住的两条】
 * 1. 上界 = 内容宽（不减视口宽），且按「绘制 = 原生 − 偏移」投影；
 * 2. 下界在同步模式下是 **−偏移**（左侧预留的对齐留白是合法可滚区间），不是 0。
 */
import { describe, expect, it } from "vitest";

import { createEdgeScrollDriver } from "./edgeScrollDriver";
import { edgeScrollMaxLeftPx } from "../../shared/edgeAutoScroll";

/**
 * 内核在给定偏移下的**绘制坐标系**边界（实测值，取自宿主行为）。
 *
 * 宿主对外一律绘制坐标：`scrollLeft` = 原生 − 偏移。原生上界 = 内容宽，因此
 * 绘制上界 = 内容宽 − 偏移、绘制下界 = −偏移。
 */
function kernelDrawingBounds(offsetPx: number, contentWidthPx: number): [number, number] {
    return [-offsetPx, contentWidthPx - offsetPx];
}

/** 驱动侧边界：与 `usePianoRollInteractions` 里的接法一致。 */
function driverBounds(args: {
    offsetPx: number;
    contentWidthPx: number;
    viewportWidthPx: number;
}): [number, number] {
    const { offsetPx, contentWidthPx, viewportWidthPx } = args;
    // 内容宽 → 末帧（帧周期 5ms、缩放 100px/s ⇒ 每帧 0.5px）。
    const maxFrame = Math.round(contentWidthPx / 0.5);
    return [
        -offsetPx,
        edgeScrollMaxLeftPx({
            pxPerSec: 100,
            framePeriodMs: 5,
            maxFrame,
            viewportWidthPx,
            nativeOffsetPx: offsetPx,
        }),
    ];
}

describe("边缘自动滚屏的上下界 vs 内核真值", () => {
    const CONTENT_W = 5000;
    const VIEWPORT_W = 1864;

    it("未同步（偏移 0）：上下界与内核一致", () => {
        const kernel = kernelDrawingBounds(0, CONTENT_W);
        const driver = driverBounds({
            offsetPx: 0,
            contentWidthPx: CONTENT_W,
            viewportWidthPx: VIEWPORT_W,
        });
        expect(driver[0]).toBeCloseTo(kernel[0], 6);
        expect(driver[1]).toBeCloseTo(kernel[1], 6);
    });

    it("★ 同步（偏移 200）：下界是负的预留留白，上界随之左移", () => {
        const kernel = kernelDrawingBounds(200, CONTENT_W);
        const driver = driverBounds({
            offsetPx: 200,
            contentWidthPx: CONTENT_W,
            viewportWidthPx: VIEWPORT_W,
        });
        expect(driver[0]).toBeCloseTo(kernel[0], 6);
        expect(driver[1]).toBeCloseTo(kernel[1], 6);
        // 留白可滚：下界必须严格为负，否则"向左滚"会被夹回 0 而与内核打架。
        expect(driver[0]).toBeLessThan(0);
    });

    it("★ 驱动在负下界区间内能逐帧推进（闪现的直接回归）", () => {
        const MIN = -200;
        let scrollLeft = 0;
        const seen: number[] = [];
        const driver = createEdgeScrollDriver({
            getBounds: () => ({ left: 100, right: 1100 }),
            getScrollLeft: () => scrollLeft,
            setScrollLeft: (next) => {
                scrollLeft = next;
            },
            getMaxScrollLeft: () => CONTENT_W,
            getMinScrollLeft: () => MIN,
            maxSpeedPxPerSec: 420,
            now: () => seen.length * (1000 / 60),
            requestFrame: () => 1,
            cancelFrame: () => {},
        });
        // 指针停在左缘，手动推进若干帧。
        for (let i = 0; i < 6; i += 1) {
            driver.step(100 + 4);
            seen.push(scrollLeft);
        }
        // 严格递减 ⇒ 每帧都在推进；若下界被错当成 0，则全部为 0。
        for (let i = 1; i < seen.length; i += 1) {
            expect(seen[i]).toBeLessThan(seen[i - 1]);
        }
        expect(seen.at(-1)).toBeLessThan(0);
    });
});
