/**
 * 参数编辑器内核 · 宿主（阶段 1：滚动 / 视口所有权）
 *
 * 【主要内容】
 * 命令式宿主：持有滚动容器、两条自绘滚动条的 thumb、尺寸量测、`ScrollKernel`
 * 与帧循环，并把这些拼成唯一的视口真值源。面板不再从 `scroller.scrollLeft /
 * scrollTop` 读滚动状态，改为调用本宿主的 API。
 *
 * 【作用】
 * 旧实现把**原生 scroller 当唯一事实源**：滚动量由浏览器维护，面板经
 * `syncScrollLeft` → `applyScrollLayers` → 总线 → 每帧 `reconcile` 把它推给各图层。
 * 阶段 1 反转这个关系：`ScrollKernel` 持有真值，DOM scroller 退化为兼容镜像。
 * 这样后续阶段（GL 网格、曲线三角化）才能做到「滚动帧只改一个 uniform」。
 *
 * 【阶段 1 的范围（刻意很小）】
 * - **不引入 WebGL**：绘制仍由面板的 Canvas2D 完成，宿主经 `onFrame(axis)` 交回。
 *   因此本模块可在无 jsdom 的 node 环境单测（见同目录 `.test.ts`）。
 * - **不接管输入**：滚轮 / 键盘 / 中键拖拽 / 滚动条 thumb 拖拽与轨道翻页均属
 *   Task 7。宿主当前**不注册任何事件监听**，输入语义留在面板，避免本任务的验收
 *   被手势细节污染；`dispose()` 因此只释放订阅、帧循环与尺寸观察。
 * - **不迁移缩放**：`pxPerSec` 由面板解析后经 `setPxPerSec` 写入；内核只据此算
 *   内容宽度与横向上限。
 *
 * 【竖向为什么用「额外高度」凑出 1600】
 * 参数编辑器的竖向不是像素行而是**值域**。内核只认像素，因此把值域范围表达为
 * 内容高度：`extraContentHeightPx = 1600 + 视口高`，配合 `trackCount = 0` 得到
 * `maxScrollTop = (1600 + vh) − vh = 1600`——与旧实现 1600px spacer 的实测上限
 * **逐值相等**（浏览器实测：spacer 1600px、原生 maxScrollTop 1600）。换算仍全部走
 * `verticalValueScroll`，不在本文件重写比值公式。
 *
 * 【与其他模块的关系】
 * - 上游：`PianoRollPanel` 在挂载 effect 里创建并持有，卸载时 `dispose()`。
 * - 复用：`timeline/kernel/scrollKernel`（视口真值）、`renderLoop`（帧调度）、
 *   `timeline/kernel/input/scrollbars`（滚动条几何）、`timelineAxis`（投影）。
 * - 下游：`onFrame` 把投影交回面板绘制（标尺 / 网格 / 画布 / 波形 / 播放头）。
 */

import { readDevicePixelRatio } from "../../../../../utils/devicePixelLine";
import { invokeGridRedrawHandler } from "../../../timeline/gridRedrawBridge";
import { createRenderLoop } from "../../../timeline/kernel/renderLoop";
import {
    createScrollKernel,
    type TimelineViewportState,
} from "../../../timeline/kernel/scrollKernel";
import { createTimelineAxis, type TimelineAxis } from "../../../timeline/runtime/timelineAxis";
import {
    resolvePianoRollScrollbarGeometries,
    type PianoRollScrollbarGeometries,
} from "../scroll/scrollbarSpec";
import {
    centerFromKernelScrollTop,
    kernelScrollTopFromCenter,
    PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX,
} from "../scroll/verticalValueScroll";
import type { PianoRollKernelData } from "./pianoRollKernelData";

/**
 * 水平滚动向 React 量化提交的步长（CSS px）。
 *
 * 与时间轴内核取同一量级：标尺的**刻度范围**由 React 按当前视口计算，只写内容层
 * transform 会让刻度停留在初始视口。量化提交保证 React 不进滚动热路径。
 */
const SCROLL_COMMIT_STEP_PX = 256;

/** 宿主需要跟随视口的 DOM（阶段 1：标尺内容层与背景网格）。 */
export interface PianoRollKernelDomSync {
    /** 标尺内容层：按绘制坐标反向平移。 */
    readonly rulerContent?: HTMLElement | null;
    /** 背景网格层：经 `gridRedrawBridge` 重绘（自行判定是否真的重画）。 */
    readonly gridLayer?: HTMLElement | null;
}

/** 宿主构造参数。 */
export interface PianoRollKernelHostArgs {
    /** 宿主容器（尺寸来源；`ResizeObserver` 目标）。 */
    readonly container: HTMLElement;
    /** 水平滚动条 thumb。 */
    readonly hScrollbarThumb: HTMLElement;
    /** 竖直滚动条 thumb。 */
    readonly vScrollbarThumb: HTMLElement;
    /** 数据镜像读取（每帧调用；面板每次 render 更新字段）。 */
    readonly data: () => PianoRollKernelData;
    /** 初始水平缩放（每秒像素数）。缩放真值仍在面板，见文件头。 */
    readonly initialPxPerSec: number;
    /** 需要跟随视口的外部 DOM。 */
    readonly sync?: PianoRollKernelDomSync;
    /**
     * 帧提交回调：宿主每帧把当前投影交回面板。
     *
     * 阶段 1 的绘制（画布 / 波形总线 / 播放头 DOM）仍全部由面板实现，因此这里传
     * `axis` 而不是让宿主自己画——保持绘制逻辑单一来源，Phase 2 再逐步收进内核。
     */
    readonly onFrame?: (axis: TimelineAxis) => void;
    /** 水平滚动位置的量化提交（跨步长才回调）。 */
    readonly onScrollLeftCommit?: (scrollLeftPx: number) => void;
    /** 帧调度注入（默认 rAF；测试注入手动实现）。 */
    readonly requestFrame?: (callback: FrameRequestCallback) => number;
    /** 取消帧注入。 */
    readonly cancelFrame?: (handle: number) => void;
}

/** 宿主句柄。 */
export interface PianoRollKernelHost {
    /** 请求水平滚动（自动滚屏 / 时间轴同步）。钳制由内核完成。 */
    setScrollLeft(px: number): void;
    /** 请求竖向滚动（像素域，0..1600）。钳制由内核完成。 */
    setScrollTop(px: number): void;
    /**
     * 一次性提交水平滚动与缩放。
     *
     * 特殊说明：两个字段必须一次提交，否则会出现「用旧上限钳制新缩放」的中间态。
     */
    setViewport(next: { pxPerSec?: number; scrollLeft?: number }): void;
    /** 更新水平缩放（内容宽度随之变化，内部按新边界钳制滚动）。 */
    setPxPerSec(pxPerSec: number): void;
    /** 读取当前视口真值。 */
    getViewport(): {
        scrollLeft: number;
        scrollTop: number;
        pxPerSec: number;
        viewportWidth: number;
        viewportHeight: number;
    };
    /** 读取当前投影（供面板绘制与波形图层复用）。 */
    getAxis(): TimelineAxis;
    /** 读取当前值域中心（由内核像素位置换算）。 */
    getValueCenter(): number;
    /** 写入值域中心（换算为内核像素位置后提交）。 */
    setValueCenter(center: number): void;
    /** 读取两条滚动条的几何（供面板绘制 / 调试）。 */
    getScrollbarGeometries(): PianoRollScrollbarGeometries;
    /** 标脏：请求下一帧提交。 */
    invalidate(): void;
    /** 释放全部资源（帧循环、滚动订阅、尺寸观察）。重复调用安全。 */
    dispose(): void;
}

/**
 * 判断数值是否值得写入 DOM（跨过容差）。
 *
 * 特殊说明：用**取反**写法是刻意的——`previous` 为 NaN（从未写入）时
 * `Math.abs(next - NaN) <= eps` 恒为 false，若正着写会得到「无需写入」，
 * 表现为标尺在第一次交互前完全不动。取反让 NaN 稳定走「需要写入」分支
 * （与时间轴宿主同一写法）。
 *
 * @param next 本次要写入的值。
 * @param previous 上一次写入的值（NaN = 从未写入）。
 * @param epsilon 视为「无变化」的最大差值。
 * @returns 需要写入时为 true。
 */
function shouldWrite(next: number, previous: number, epsilon = 0.01): boolean {
    return !(Math.abs(next - previous) <= epsilon);
}

/**
 * 创建参数编辑器内核宿主。
 *
 * 流程：
 * 1. 量测容器尺寸（初值取 `clientWidth/Height`，有 `ResizeObserver` 时持续更新）；
 * 2. 建 `ScrollKernel`（竖向经「额外高度」表达 1600px 值域范围，见文件头）；
 * 3. 建帧循环并订阅滚动变更（变化即标脏）；
 * 4. 标脏首帧，使首屏与旧实现同样立即画出。
 *
 * @param args 宿主参数。
 * @returns 宿主句柄；调用方须在卸载时 `dispose()`。
 */
export function createPianoRollKernelHost(args: PianoRollKernelHostArgs): PianoRollKernelHost {
    const { container, hScrollbarThumb, vScrollbarThumb } = args;
    const { data, sync, onFrame, onScrollLeftCommit } = args;

    /** 待释放的资源（倒序执行；幂等由 `disposed` 保证）。 */
    const teardown: Array<() => void> = [];
    let disposed = false;

    // ── 尺寸量测 ────────────────────────────────────────────────────
    // 量测值必须 O(1) 读取且**不触发布局**：内核的写入路径每次都会读视口高，
    // 直接读 clientHeight 会强制样式重算（见 ScrollKernelOptions 的注入契约）。
    let viewportWidthPx = Math.max(1, container.clientWidth || 1);
    let viewportHeightPx = Math.max(1, container.clientHeight || 1);

    /**
     * 竖向「内容高度」的额外量：把值域滚动范围表达为像素内容。
     *
     * 取 `1600 + 视口高` 使 `maxScrollTop` 恰好等于 1600（见文件头推导），与旧实现
     * 的 1600px spacer 逐值一致；随视口高变化可保证该等式在任意窗口尺寸下成立。
     *
     * @returns 额外内容高度（CSS px）。
     */
    function verticalExtraHeightPx(): number {
        return PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX + viewportHeightPx;
    }

    // ── 滚动内核 ────────────────────────────────────────────────────
    const scroll = createScrollKernel({
        pxPerSec: Number.isFinite(args.initialPxPerSec) ? Math.max(1e-9, args.initialPxPerSec) : 1,
        // 参数编辑器没有「行」：行高与轨道数恒为 0，竖向范围全部经额外高度表达。
        rowHeight: 0,
        projectSec: () => data().projectSec,
        trackCount: () => 0,
        extraContentHeightPx: verticalExtraHeightPx,
        viewportHeightPx: () => viewportHeightPx,
    });

    const resizeObserverCtor = (globalThis as { ResizeObserver?: typeof ResizeObserver })
        .ResizeObserver;
    if (typeof resizeObserverCtor === "function") {
        const observer = new resizeObserverCtor(() => {
            viewportWidthPx = Math.max(1, container.clientWidth || 1);
            viewportHeightPx = Math.max(1, container.clientHeight || 1);
            // 尺寸变化会改变横向上限（视口宽影响滚动条几何）与竖向上限的组成项，
            // 必须按新边界重新钳制，否则已提交位置可能越界。
            scroll.reclamp();
            loop.invalidate();
        });
        observer.observe(container);
        teardown.push(() => observer.disconnect());
    }

    // ── 帧循环 ──────────────────────────────────────────────────────
    /** thumb 几何去重 key（只有极少数组合需要写 DOM）。 */
    let lastHThumbKey = "";
    let lastVThumbKey = "";
    /** 上一次写入标尺内容层的平移量（NaN = 从未写入）。 */
    let lastRulerTranslateX = Number.NaN;
    /** 上一次量化提交给 React 的水平滚动位置（NaN = 从未提交）。 */
    let lastCommittedScrollLeft = Number.NaN;

    /**
     * 更新两条滚动条 thumb 的几何（值变化才写 DOM）。
     *
     * @param view 当前视口真值。
     */
    function updateScrollbars(view: TimelineViewportState): void {
        const geometries = resolvePianoRollScrollbarGeometries({
            viewportWidthPx,
            viewportHeightPx,
            contentWidthPx: view.pxPerSec * Math.max(0, data().projectSec),
            scrollLeftPx: view.scrollLeft,
            scrollTopPx: view.scrollTop,
            maxScrollLeftPx: scroll.maxScrollLeft(),
            maxScrollTopPx: scroll.maxScrollTop(),
        });

        const horizontal = geometries.horizontal;
        const horizontalKey = `${horizontal.scrollable ? 1 : 0}|${Math.round(
            horizontal.thumbLengthPx,
        )}|${Math.round(horizontal.thumbStartPx)}`;
        if (horizontalKey !== lastHThumbKey) {
            lastHThumbKey = horizontalKey;
            hScrollbarThumb.style.width = `${Math.max(0, horizontal.thumbLengthPx)}px`;
            hScrollbarThumb.style.transform = `translateX(${Math.max(
                0,
                horizontal.thumbStartPx,
            )}px)`;
            hScrollbarThumb.style.display = horizontal.scrollable ? "block" : "none";
        }

        const vertical = geometries.vertical;
        const verticalKey = `${vertical.scrollable ? 1 : 0}|${Math.round(
            vertical.thumbLengthPx,
        )}|${Math.round(vertical.thumbStartPx)}`;
        if (verticalKey !== lastVThumbKey) {
            lastVThumbKey = verticalKey;
            vScrollbarThumb.style.height = `${Math.max(0, vertical.thumbLengthPx)}px`;
            vScrollbarThumb.style.transform = `translateY(${Math.max(
                0,
                vertical.thumbStartPx,
            )}px)`;
            vScrollbarThumb.style.display = vertical.scrollable ? "block" : "none";
        }
    }

    /**
     * 把视口写到需要跟随的 DOM（标尺内容层 / 背景网格）。
     *
     * @param view 当前视口真值。
     */
    function syncDom(view: TimelineViewportState): void {
        const ruler = sync?.rulerContent;
        if (ruler != null && shouldWrite(view.scrollLeft, lastRulerTranslateX)) {
            lastRulerTranslateX = view.scrollLeft;
            ruler.style.transform = `translateX(${-view.scrollLeft}px)`;
        }
        // 网格层自带重绘节流（`BackgroundGrid` 内部判定），这里只需转交绘制坐标。
        if (sync?.gridLayer != null) {
            invokeGridRedrawHandler(sync.gridLayer, view.scrollLeft);
        }
    }

    /**
     * 构造当前投影（内容坐标 → 视口坐标的唯一来源）。
     *
     * @returns 冻结的 axis。
     */
    function currentAxis(): TimelineAxis {
        const view = scroll.get();
        return createTimelineAxis({
            pxPerSec: view.pxPerSec,
            scrollLeftPx: view.scrollLeft,
            scrollTopPx: view.scrollTop,
            viewportWidthPx,
            dpr: readDevicePixelRatio(),
        });
    }

    /** 帧提交：滚动条几何 → DOM 同步 → 面板绘制 → 量化提交。 */
    function draw(): void {
        if (disposed) return;
        const view = scroll.get();
        updateScrollbars(view);
        syncDom(view);
        onFrame?.(currentAxis());

        // 量化提交：标尺的刻度范围由 React 按视口计算，不同步就会出现「滚动后
        // 刻度消失」。按步长提交保证 React 不进滚动热路径。
        if (onScrollLeftCommit !== undefined) {
            if (shouldWrite(view.scrollLeft, lastCommittedScrollLeft, SCROLL_COMMIT_STEP_PX)) {
                lastCommittedScrollLeft = view.scrollLeft;
                onScrollLeftCommit(view.scrollLeft);
            }
        }
    }

    const loop = createRenderLoop({
        draw,
        requestFrame: args.requestFrame,
        cancelFrame: args.cancelFrame,
    });
    loop.start();
    const unsubscribeScroll = scroll.subscribe(() => loop.invalidate());
    // 倒序释放：先停循环，再断订阅（顺序对结果无影响，但先停循环可确保
    // 释放过程中不会再有帧回调写 DOM）。
    teardown.push(() => unsubscribeScroll());
    teardown.push(() => loop.stop());

    // 首帧：与旧实现一样在挂载后立即画出内容（不等用户交互）。
    loop.invalidate();

    return {
        setScrollLeft(px) {
            if (disposed) return;
            scroll.setScrollLeft(px);
        },

        setScrollTop(px) {
            if (disposed) return;
            scroll.setScrollTop(px);
        },

        setViewport(next) {
            if (disposed) return;
            scroll.setViewport(next);
        },

        setPxPerSec(pxPerSec) {
            if (disposed) return;
            // 经 setViewport 原子提交：用**新** pxPerSec 算出的上限钳制滚动位置，
            // 避免「新缩放 + 旧上限」的中间态（与 setZoom 的注释同因）。阶段 1 的
            // 缩放锚点解析仍在面板（Task 7 迁移），故这里不传 anchorScreenX。
            scroll.setViewport({ pxPerSec });
        },

        getViewport() {
            const view = scroll.get();
            return {
                scrollLeft: view.scrollLeft,
                scrollTop: view.scrollTop,
                pxPerSec: view.pxPerSec,
                viewportWidth: viewportWidthPx,
                viewportHeight: viewportHeightPx,
            };
        },

        getAxis() {
            return currentAxis();
        },

        getValueCenter() {
            const view = scroll.get();
            const domain = data().valueDomain;
            return centerFromKernelScrollTop({
                min: domain.min,
                max: domain.max,
                span: domain.span,
                scrollTop: view.scrollTop,
            });
        },

        setValueCenter(center) {
            if (disposed) return;
            const domain = data().valueDomain;
            scroll.setScrollTop(
                kernelScrollTopFromCenter({
                    min: domain.min,
                    max: domain.max,
                    span: domain.span,
                    center,
                }),
            );
        },

        getScrollbarGeometries() {
            const view = scroll.get();
            return resolvePianoRollScrollbarGeometries({
                viewportWidthPx,
                viewportHeightPx,
                contentWidthPx: view.pxPerSec * Math.max(0, data().projectSec),
                scrollLeftPx: view.scrollLeft,
                scrollTopPx: view.scrollTop,
                maxScrollLeftPx: scroll.maxScrollLeft(),
                maxScrollTopPx: scroll.maxScrollTop(),
            });
        },

        invalidate() {
            if (disposed) return;
            loop.invalidate();
        },

        dispose() {
            if (disposed) return;
            disposed = true;
            for (let i = teardown.length - 1; i >= 0; i -= 1) {
                teardown[i]();
            }
            teardown.length = 0;
        },
    };
}
