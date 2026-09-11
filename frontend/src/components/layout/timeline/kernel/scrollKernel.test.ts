/**
 * 滚动内核（./scrollKernel）行为自检。
 *
 * 【主要内容】
 * 1. 写入即钳制：scrollLeft 落在 [0, 内容宽 − 视口宽]，内容不足一屏时上限为 0；
 * 2. 锚点缩放：缩放前后锚点像素位置对应的工程时间不变，且缩放后的 scrollLeft 仍被钳制；
 * 3. 通知契约：仅状态真正变化才通知订阅者。
 *
 * 【作用】这三组断言是下游模块（渲染循环 / 几何构建 / 输入层）依赖的语义边界：
 * 一旦钳制被挪到读侧、或通知退化成「无条件派发」，渲染会开始空转或出现视口漂移。
 *
 * 【与其他模块的关系】覆盖 `scrollKernel.ts`；不依赖 `runtime/` 与 React。
 */

import { describe, expect, it, vi } from "vitest";

import { createScrollKernel, type ScrollKernelOptions } from "./scrollKernel";

function makeKernel(overrides: Partial<ScrollKernelOptions> = {}) {
    return createScrollKernel({
        pxPerSec: 100,
        rowHeight: 80,
        projectSec: () => 1000,
        trackCount: () => 10,
        viewportWidthPx: () => 800,
        viewportHeightPx: () => 400,
        ...overrides,
    });
}

describe("scrollKernel", () => {
    it("钳制 scrollLeft 到 [0, contentWidth - viewportWidth]", () => {
        const k = makeKernel();
        k.setScrollLeft(-50);
        expect(k.get().scrollLeft).toBe(0);
        k.setScrollLeft(999999);
        expect(k.get().scrollLeft).toBe(1000 * 100 - 800);
    });

    it("内容不足一屏时 maxScroll 为 0", () => {
        const k = makeKernel({ projectSec: () => 1 });
        k.setScrollLeft(500);
        expect(k.get().scrollLeft).toBe(0);
    });

    it("缩放保持指针锚点下的时间不变", () => {
        const k = makeKernel();
        k.setScrollLeft(300);
        const anchorScreenX = 200;
        const anchorSecBefore = (300 + anchorScreenX) / 100;
        k.setZoom(200, anchorScreenX);
        expect(k.get().pxPerSec).toBe(200);
        const anchorSecAfter = (k.get().scrollLeft + anchorScreenX) / 200;
        expect(anchorSecAfter).toBeCloseTo(anchorSecBefore, 6);
    });

    it("缩放后 scrollLeft 仍被钳制（下界）", () => {
        const k = makeKernel();
        // 锚点在视口右侧、缩放变小 → 反算值 = 5 × 10 − 200 = −150，必须钳到 0。
        // 若实现漏掉钳制，此断言会读到 −150 而失败（保证断言有判别力）。
        k.setScrollLeft(300);
        k.setZoom(10, 200);
        expect(k.get().scrollLeft).toBe(0);
    });

    it("缩放后 scrollLeft 仍被钳制（上界）", () => {
        const k = makeKernel();
        // 内容末端、缩放变小 → 反算值 = 992 × 1 = 992，超过新上限 200，必须钳到 200。
        k.setScrollLeft(1000 * 100 - 800);
        k.setZoom(1, 0);
        expect(k.get().scrollLeft).toBe(1000 * 1 - 800);
    });

    it("状态变化通知订阅者，未变化不通知", () => {
        const k = makeKernel();
        const spy = vi.fn();
        k.subscribe(spy);
        k.setScrollLeft(100);
        expect(spy).toHaveBeenCalledTimes(1);
        k.setScrollLeft(100);
        expect(spy).toHaveBeenCalledTimes(1);
    });
});
