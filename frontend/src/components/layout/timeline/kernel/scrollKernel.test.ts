/**
 * 滚动内核（./scrollKernel）行为自检。
 *
 * 【主要内容】
 * 1. 钳制语义：水平上限 = 工程宽度（允许把工程右端滚到视口左缘）；竖直上限 =
 *    内容高 − 视口高；非法写入被忽略；
 * 2. 缩放语义：锚点像素位置对应的工程时间不变；缩放后 scrollLeft 仍被钳制（下界）；
 *    pxPerSec 被夹取到 [minPxPerSec, maxPxPerSec]；
 * 3. 状态契约：仅真实变化才通知；get() 在字段未变化时返回同一冻结引用
 *    （下游据此判脏）；退订后不再收到通知；
 * 4. reclamp：外部边界收窄后把越界位置钳回，未越界时不通知。
 *
 * 【作用】这些断言是下游模块（渲染循环 / 几何构建 / 输入层）依赖的语义边界：
 * 一旦水平上限退化成「工程宽 − 视口宽」、或通知/引用契约被破坏，渲染会开始空转，
 * 缩放锚点会在工程边缘漂移。
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
        viewportHeightPx: () => 400,
        ...overrides,
    });
}

describe("scrollKernel", () => {
    describe("钳制", () => {
        it("水平上限 = 工程宽度（允许把工程右端滚到视口左缘）", () => {
            const k = makeKernel();
            k.setScrollLeft(-50);
            expect(k.get().scrollLeft).toBe(0);
            k.setScrollLeft(999999);
            // 1000s × 100px/s = 100000；不是「工程宽 − 视口宽」。
            expect(k.get().scrollLeft).toBe(1000 * 100);
        });

        it("内容不足一屏时水平仍可滚到工程右端", () => {
            const k = makeKernel({ projectSec: () => 1 });
            k.setScrollLeft(500);
            expect(k.get().scrollLeft).toBe(1 * 100);
        });

        it("竖直上限 = 内容高 − 视口高", () => {
            const k = makeKernel();
            k.setScrollTop(-10);
            expect(k.get().scrollTop).toBe(0);
            k.setScrollTop(999999);
            // 10 轨 × 80px = 800，视口 400 → 上限 400。
            expect(k.get().scrollTop).toBe(10 * 80 - 400);
        });

        it("轨道不足一屏时竖直上限为 0", () => {
            const k = makeKernel({ trackCount: () => 2 });
            k.setScrollTop(500);
            expect(k.get().scrollTop).toBe(0);
        });

        it("非法写入（NaN / Infinity）被忽略", () => {
            const k = makeKernel();
            k.setScrollLeft(100);
            k.setScrollLeft(Number.NaN);
            k.setScrollTop(Number.POSITIVE_INFINITY);
            expect(k.get().scrollLeft).toBe(100);
            expect(k.get().scrollTop).toBe(0);
        });
    });

    describe("缩放", () => {
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
            // 缩小且锚点秒 > 0 → 反算值 = 5 × 10 − 200 = −150，必须钳到 0。
            // 若实现漏掉钳制，此断言会读到 −150 而失败（保证断言有判别力）。
            k.setScrollLeft(300);
            k.setZoom(10, 200);
            expect(k.get().scrollLeft).toBe(0);
        });

        it("pxPerSec 夹取到 [minPxPerSec, maxPxPerSec]", () => {
            const k = makeKernel({ minPxPerSec: 4, maxPxPerSec: 8000 });
            k.setZoom(0.001, 0);
            expect(k.get().pxPerSec).toBe(4);
            k.setZoom(1e9, 0);
            expect(k.get().pxPerSec).toBe(8000);
        });
    });

    describe("状态与通知契约", () => {
        it("下限支持函数形式：钳制随返回值实时变化", () => {
        // 下限随工程长度 / 视口宽度变化（见 resolveTimelineMinPxPerSec）：
        // 冻结成常量会让内核钳制与滚轮解析不一致。
        let min = 10;
        const k = makeKernel({ minPxPerSec: () => min });
        expect(k.setZoom(1, 0)).toBe(10);
        min = 0.5;
        expect(k.setZoom(1, 0)).toBe(1);
    });

    it("setZoom 返回实际生效值（被上下限钳制时）", () => {
        const k = makeKernel({ minPxPerSec: 5, maxPxPerSec: 100 });
        expect(k.setZoom(1, 0)).toBe(5);
        expect(k.setZoom(1000, 0)).toBe(100);
        expect(k.setZoom(50, 0)).toBe(50);
    });

    it("非法 pxPerSec 时 setZoom 返回当前值且不改变状态", () => {
        const k = makeKernel({ minPxPerSec: 5 });
        k.setZoom(50, 0);
        expect(k.setZoom(Number.NaN, 0)).toBe(50);
        expect(k.get().pxPerSec).toBe(50);
    });

    it("设置行高后内容高度与竖直上限同步更新", () => {
        const k = makeKernel();
        // 10 轨 × 80 = 800；视口高 400 → 上限 400
        expect(k.contentHeightPx()).toBe(800);
        expect(k.maxScrollTop()).toBe(400);
        k.setRowHeight(120);
        expect(k.get().rowHeight).toBe(120);
        expect(k.contentHeightPx()).toBe(1200);
        expect(k.maxScrollTop()).toBe(800);
    });

    it("行高变矮时竖直位置被重新钳制", () => {
        const k = makeKernel();
        k.setScrollTop(400);
        // 行高减半：内容高 400，视口高 400 → 上限 0，位置必须回到 0。
        k.setRowHeight(40);
        expect(k.get().scrollTop).toBe(0);
    });

    it("行高同值写入不通知订阅者", () => {
        const k = makeKernel();
        const spy = vi.fn();
        k.subscribe(spy);
        k.setRowHeight(80);
        expect(spy).not.toHaveBeenCalled();
        k.setRowHeight(96);
        expect(spy).toHaveBeenCalledTimes(1);
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

        it("退订后不再收到通知", () => {
            const k = makeKernel();
            const spy = vi.fn();
            const unsubscribe = k.subscribe(spy);
            unsubscribe();
            k.setScrollLeft(100);
            expect(spy).not.toHaveBeenCalled();
        });

        it("字段未变化时 get() 返回同一引用（下游可据此判脏）", () => {
            const k = makeKernel();
            const before = k.get();
            expect(k.get()).toBe(before);
            k.setScrollLeft(100);
            const afterMove = k.get();
            expect(afterMove).not.toBe(before);
            // 写入同值不换引用：滚动帧判定「视口未变化」的依据。
            k.setScrollLeft(100);
            expect(k.get()).toBe(afterMove);
        });

        it("状态对象被冻结", () => {
            const k = makeKernel();
            expect(Object.isFrozen(k.get())).toBe(true);
        });
    });

    describe("reclamp", () => {
        it("外部边界收窄后把越界位置钳回上限", () => {
            let projectSec = 1000;
            const k = makeKernel({ projectSec: () => projectSec });
            k.setScrollLeft(1000 * 100);
            expect(k.get().scrollLeft).toBe(100000);
            // 工程变短 → 水平上限随之收窄，reclamp 把位置钳回。
            projectSec = 100;
            k.reclamp();
            expect(k.get().scrollLeft).toBe(100 * 100);
        });

        it("位置未越界时不产生通知", () => {
            const k = makeKernel();
            k.setScrollLeft(100);
            const spy = vi.fn();
            k.subscribe(spy);
            k.reclamp();
            expect(spy).not.toHaveBeenCalled();
        });
    });
});
