/**
 * 时间轴视口访问器（TimelineViewportAccess）
 *
 * 【主要内容】
 * 把「读视口矩形 / 读滚动位置 / 读视口宽度 / 写横向滚动位置 / 原子写缩放+滚动」
 * 这几件与**滚动载体**强相关的操作收敛成一个模式无关的接口，内部给出两种实现：
 * - 旧实现：原生滚动容器（`scrollRef`，浏览器维护 scrollLeft）；
 * - 渲染内核：`ScrollKernel` 自绘滚动（`kernelHostRef`，内核是视口真值源）。
 *
 * 【作用】
 * 内核模式下旧滚动容器不挂载（`scrollRef.current === null`），而面板里大量逻辑
 * 仍按「原生 scroller 存在」编写：素材拖入落点换算、自动滚屏、聚焦播放光标、
 * 参数编辑器视图同步。这些逻辑本身与渲染方式无关，缺的只是一个可注入的视口来源。
 * 本模块让「判断当前模式」只发生一次，而不是散落在每个调用点。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 用 `scrollRef` + `kernelHostRef` 构造一个**稳定**实例
 *   （随渲染重建会让下游 hook 的 effect 依赖抖动、反复重注册监听）。
 * - 下游：`useTimelineDragDrop`（拖入落点）、`useTimelineEventHandlers`
 *   （键盘缩放 / 聚焦播放光标）、`TimelinePanel` 的自动滚屏帧回调。
 * - 依赖：`runtime/nativeScrollApply`（旧实现的写后回读）与内核宿主的公开句柄类型。
 *
 * 【约定（与既有实现一致）】
 * 1. `setScrollLeft` 返回**实际生效值**：旧实现经浏览器钳制/量化后回读；内核经
 *    `ScrollKernel` 钳制后回读。调用方不得把请求值当作已生效值使用——否则
 *    跟随视口的图层会与真实视口错位。
 * 2. 本模块**不做任何钳制**：旧实现的钳制归浏览器，内核归 `ScrollKernel`
 *    （见 scrollKernel 文件头的「钳制只做一次」约束）。
 * 3. `getViewportWidth` 在内核模式下取宿主缓存的量测值（O(1)、不触发布局），
 *    因此可以安全地在每帧路径（自动滚屏）里调用。
 */
import type React from "react";

import type { TimelineKernelHost } from "../kernel/host/timelineKernelHost";
import { applyNativeScrollLeft } from "../runtime/nativeScrollApply";

/** 模式无关的时间轴视口访问器。 */
export interface TimelineViewportAccess {
    /**
     * 当前是否由渲染内核承载视口。
     *
     * 特殊说明：只有「缩放落地」这一件事需要区分模式——旧模式的缩放必须先提交
     * React state（内容按新 pxPerSec 重排后，浏览器才会接受新的 scrollLeft），
     * 内核模式则可以一次原子提交。其余操作两种模式语义相同。
     */
    isKernel(): boolean;
    /**
     * 视口矩形（client 坐标），用于 clientX/clientY → 工程坐标换算。
     *
     * 特殊说明：会触发布局读取，只允许在低频事件（拖放 / 右键）中调用。
     *
     * @returns 容器矩形；容器不可用时为 null。
     */
    getRect(): DOMRect | null;
    /** 当前横向滚动位置（CSS px）。 */
    getScrollLeft(): number;
    /** 当前纵向滚动位置（CSS px）。 */
    getScrollTop(): number;
    /** 视口宽度（CSS px）；内核模式取缓存量测值，不触发布局。 */
    getViewportWidth(): number;
    /**
     * 写入横向滚动位置。
     *
     * @param px 目标位置（可越界，由载体自行钳制）。
     * @returns 实际生效值（回读）。
     */
    setScrollLeft(px: number): number;
    /**
     * 原子地设置缩放与横向滚动位置（**仅内核模式**）。
     *
     * 特殊说明：旧模式不支持——它的缩放必须经 React state 重排内容宽度后再落
     * scrollLeft（见 `useTimelineEventHandlers` 的旧分支）。调用方必须先判断
     * `isKernel()`。
     *
     * @param pxPerSec 目标缩放。
     * @param scrollLeft 目标横向滚动位置。
     * @returns 实际生效的真值。
     */
    setZoomAndScroll(
        pxPerSec: number,
        scrollLeft: number,
    ): { pxPerSec: number; scrollLeft: number };
}

/**
 * 创建视口访问器。
 *
 * 流程：两个 ref 都只被**延迟读取**（每次方法调用现读），因此实例可以在内核宿主
 * 创建之前构造，无需随挂载状态重建——调用方只需保证引用稳定。
 *
 * @param args.scrollRef 旧实现的原生滚动容器引用。
 * @param args.kernelHostRef 渲染内核宿主句柄引用。
 * @returns 模式无关的视口访问器。
 */
export function createTimelineViewportAccess(args: {
    scrollRef: React.MutableRefObject<HTMLDivElement | null>;
    kernelHostRef: React.MutableRefObject<TimelineKernelHost | null>;
}): TimelineViewportAccess {
    const { scrollRef, kernelHostRef } = args;

    return {
        isKernel() {
            return kernelHostRef.current !== null;
        },

        getRect() {
            const host = kernelHostRef.current;
            if (host !== null) return host.getContainerRect();
            return scrollRef.current?.getBoundingClientRect() ?? null;
        },

        getScrollLeft() {
            const host = kernelHostRef.current;
            if (host !== null) return host.getViewport().scrollLeft;
            return scrollRef.current?.scrollLeft ?? 0;
        },

        getScrollTop() {
            const host = kernelHostRef.current;
            if (host !== null) return host.getViewport().scrollTop;
            return scrollRef.current?.scrollTop ?? 0;
        },

        getViewportWidth() {
            const host = kernelHostRef.current;
            if (host !== null) return host.getViewport().viewportWidth;
            return scrollRef.current?.clientWidth ?? 0;
        },

        setScrollLeft(px: number) {
            const host = kernelHostRef.current;
            if (host !== null) {
                host.setScrollLeft(px);
                return host.getViewport().scrollLeft;
            }
            const scroller = scrollRef.current;
            if (scroller === null) return px;
            return applyNativeScrollLeft(scroller, px);
        },

        setZoomAndScroll(pxPerSec: number, scrollLeft: number) {
            const host = kernelHostRef.current;
            if (host === null) {
                // 旧模式不支持：返回当前真值，调用方必须先判断 isKernel()。
                const scroller = scrollRef.current;
                return {
                    pxPerSec,
                    scrollLeft: scroller?.scrollLeft ?? scrollLeft,
                };
            }
            return host.setViewport({ pxPerSec, scrollLeft });
        },
    };
}
