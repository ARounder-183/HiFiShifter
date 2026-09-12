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
import {
    scrollDeltaFromThumbDrag,
    scrollTargetFromTrackClick,
} from "../../../timeline/kernel/input/scrollbars";
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
    /**
     * 水平滚动条**轨道**（thumb 的父元素）。
     *
     * 【为什么需要】自绘滚动条隐藏了原生滚动条，拖动 thumb 之外还必须补上
     * 「点击轨道空白翻页」——旧实现由浏览器按平台默认提供，自绘后必须自己实现，
     * 否则用户失去除拖 thumb 与滚轮之外的第三种定位手段。
     */
    readonly hScrollbarTrack?: HTMLElement;
    /** 竖直滚动条轨道。 */
    readonly vScrollbarTrack?: HTMLElement;
    /** 数据镜像读取（每帧调用；面板每次 render 更新字段）。 */
    readonly data: () => PianoRollKernelData;
    /** 初始水平缩放（每秒像素数）。缩放真值仍在面板，见文件头。 */
    readonly initialPxPerSec: number;
    /**
     * 水平同步偏移（CSS px）：内容层相对绘制区原点的右移量，缺省 0。
     *
     * 【为什么必须注入】「同步时间轴视图」开启时，参数编辑器的绘制区比轨道区窄，
     * 内容必须整体右移一个偏移量，两边网格线才能在屏幕上对齐。旧实现把它做进
     * `paddedContentWidth`，于是**原生**水平域是 `[0, 内容宽 + 偏移]`，**绘制**域是
     * `[−偏移, 内容宽]`——**含负值**。
     *
     * 内核的位置字段恒被钳到 `[0, max]`（无法表示负值），因此内核持有的是**原生
     * 坐标**（域 `[0, 内容宽 + 偏移]`），偏移在本文件的两个边界各换算一次：
     * - 产出投影 / 读视口时：`drawing = native − offset`；
     * - 外部按绘制坐标写入时：`native = drawing + offset`。
     *
     * 偏移为 0（未开启同步）时两个坐标系重合，行为与既有调用方完全一致。
     */
    readonly horizontalOffsetPx?: () => number;
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
    /**
     * 按当前外部边界重新钳制两轴位置。
     *
     * 特殊说明：钳制只在写入时发生，因此**同步偏移变化**（开启 / 关闭同步、布局位移）
     * 会改变水平上限，宿主必须被显式告知才能把已提交位置收回新边界内。尺寸变化由
     * 宿主自己的 `ResizeObserver` 处理，无需调用方关心。
     */
    reclamp(): void;
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
    const { container, hScrollbarThumb, vScrollbarThumb, hScrollbarTrack, vScrollbarTrack } = args;
    const { data, sync, onFrame, onScrollLeftCommit } = args;

    /** 待释放的资源（倒序执行；幂等由 `disposed` 保证）。 */
    const teardown: Array<() => void> = [];
    let disposed = false;

    /**
     * 登记一个事件监听并在 `dispose()` 时摘除。
     *
     * 【为什么要统一登记】宿主最典型的泄漏是「加了 window 监听却忘了摘」，卸载后
     * 仍持有回调并继续写 DOM。把 add 与 remove 配成一条记录，使配平由结构保证，
     * 而不是靠每处记得写 cleanup。
     *
     * @param target 事件目标。
     * @param type 事件类型。
     * @param handler 事件处理器。
     */
    function registerListener(target: EventTarget, type: string, handler: EventListener): void {
        target.addEventListener(type, handler);
        teardown.push(() => target.removeEventListener(type, handler));
    }

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

    /**
     * 读取当前水平同步偏移（CSS px）。
     *
     * @returns 偏移量（非有限值按 0 处理）。
     */
    function horizontalOffsetPx(): number {
        const raw = args.horizontalOffsetPx?.() ?? 0;
        return Number.isFinite(raw) ? raw : 0;
    }

    // ── 滚动内核 ────────────────────────────────────────────────────
    const scroll = createScrollKernel({
        pxPerSec: Number.isFinite(args.initialPxPerSec) ? Math.max(1e-9, args.initialPxPerSec) : 1,
        // 参数编辑器没有「行」：行高与轨道数恒为 0，竖向范围全部经额外高度表达。
        rowHeight: 0,
        projectSec: () => data().projectSec,
        trackCount: () => 0,
        extraContentHeightPx: verticalExtraHeightPx,
        // 水平域 = 内容宽 + 同步偏移，与旧实现 `paddedContentWidth` 撑出的原生域一致；
        // 内核因此持有**原生坐标**，绘制坐标在边界处减去偏移（见 args 的说明）。
        extraContentWidthPx: horizontalOffsetPx,
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

    // ── 初始位置：采纳镜像的当前值 ───────────────────────────────────
    //
    // 【为什么必须采纳】创建宿主的是 effect，而「值域 → 竖向滚动条」与「时间轴同步」
    // 两个 layout effect 都排在它**之前**执行——它们已经往原生 scroller 写过目标
    // 位置，但那时的宿主还不存在（`hostRef.current` 仍为 null，走了旧实现分支）。
    // 若内核仍从 0 起步，它的首帧镜像回写会把这些位置**覆盖掉**，表现为
    // 「钢琴键盘整体偏移一个八度（竖向回顶端）/ 横向跳回工程起点」——两者都曾在
    // 浏览器像素比对中实际出现。
    //
    // 因此这里把容器当前的两轴原生位置都收进内核。此刻值的来源有三，都应当采纳：
    // 上述 layout effect 写入的目标值、浏览器恢复的历史滚动位置（旧实现里同样是
    // 事实源）、以及重挂载前内核自己镜像回写的值。
    {
        const nativeLeft = container.scrollLeft;
        if (Number.isFinite(nativeLeft) && nativeLeft > 0) {
            scroll.setScrollLeft(nativeLeft);
        }
        const nativeTop = container.scrollTop;
        if (Number.isFinite(nativeTop) && nativeTop > 0) {
            scroll.setScrollTop(nativeTop);
        }
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
     * 计算两条滚动条的几何（**绘制与命中的唯一几何来源**）。
     *
     * 【为什么必须是唯一来源】宿主有三处需要几何：每帧写 thumb 样式、thumb 拖拽的
     * 位移换算、轨道点击的翻页判定。若各自算一遍，口径（内容尺寸是否含偏移、位置用
     * 原生还是绘制坐标）迟早分叉，表现就是「画出来的 thumb 和能拖的范围不一致」。
     *
     * 口径（与旧实现的原生滚动条逐值对齐，见 `scrollbarSpec` 的入参说明）：
     * - 内容尺寸 = 滚动上限 + 视口尺寸（水平即原生 `scrollWidth`，含同步偏移）；
     * - 位置用**绘制坐标**（原生坐标减去同步偏移），因为滚动条画在绘制区上。
     *
     * @returns 两条轴的几何；不可滚动时 `scrollable` 为 false。
     */
    function scrollbarGeometries(): PianoRollScrollbarGeometries {
        const view = scroll.get();
        return resolvePianoRollScrollbarGeometries({
            viewportWidthPx,
            viewportHeightPx,
            scrollLeftPx: Math.max(0, view.scrollLeft - horizontalOffsetPx()),
            scrollTopPx: view.scrollTop,
            maxScrollLeftPx: scroll.maxScrollLeft(),
            maxScrollTopPx: scroll.maxScrollTop(),
        });
    }

    /**
     * 更新两条滚动条 thumb 的几何（值变化才写 DOM）。
     *
     * @param view 当前视口真值（仅用于去重判定，几何经 `scrollbarGeometries` 统一计算）。
     */
    function updateScrollbars(view: TimelineViewportState): void {
        void view;
        const geometries = scrollbarGeometries();

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
            vScrollbarThumb.style.transform = `translateY(${Math.max(0, vertical.thumbStartPx)}px)`;
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
     * 特殊说明：`scrollLeftPx` 必须是**绘制坐标**（原生值减去同步偏移）——投影的
     * 消费者（标尺 / 网格 / 画布 / 波形）画的都是绘制区。用原生值会让同步模式下
     * 整个参数编辑器内容偏一个偏移量，与时间轴网格错位（曾实际发生）。
     *
     * @returns 冻结的 axis。
     */
    function currentAxis(): TimelineAxis {
        const view = scroll.get();
        return createTimelineAxis({
            pxPerSec: view.pxPerSec,
            scrollLeftPx: view.scrollLeft - horizontalOffsetPx(),
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

    // ── 输入：自绘滚动条（拖 thumb / 点轨道翻页）──────────────────────
    //
    // 【为什么必须在阶段 1 就做】内核模式隐藏了原生滚动条（否则与自绘条重复显示）。
    // 隐藏后「拖滚动条」这个动作就没有载体了——不补上就是**功能退化**，而不是
    // 视觉差异。轨道点击同理（旧实现由浏览器翻页）。滚轮 / 键盘 / 中键在阶段 1
    // 仍走面板既有路径（Task 7 迁移），故此处只做这两种指针交互。
    let dragAxis: "x" | "y" | null = null;
    let dragStartPointer = 0;
    let dragStartScroll = 0;

    /**
     * 造一个 thumb 拖拽的按下处理器。
     *
     * 流程：记录起点（指针位置 + 当时的滚动位置）→ 指针移动时按位移比例换算增量。
     *
     * 特殊说明：位移换算走 `scrollDeltaFromThumbDrag`（thumb 行程 → 内容滚动空间的
     * 比例放大），与时间轴内核同一函数，保证两个面板拖拽手感一致。起点记录滚动值
     * 而不是每帧累加，避免换算误差累积导致「拖远了松开还在漂」。
     *
     * @param axis 轴别。
     * @returns 事件处理器。
     */
    function makeThumbPointerDown(axis: "x" | "y") {
        return (event: PointerEvent) => {
            event.preventDefault();
            // 阻止冒泡：轨道处理器据此判定「到达轨道的按下必然不在 thumb 上」，
            // 无需二次命中判定（与时间轴内核同一约定）。
            event.stopPropagation();
            dragAxis = axis;
            dragStartPointer = axis === "x" ? event.clientX : event.clientY;
            dragStartScroll = axis === "x" ? scroll.get().scrollLeft : scroll.get().scrollTop;
            (event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId);
        };
    }

    /**
     * 把 thumb 拖拽位移换算为滚动位置并提交。
     *
     * 特殊说明：水平轴的起点是**原生坐标**，因此拖拽增量直接加在原生位置上——
     * 偏移在两侧同时存在、相减抵消，无需在此换算。
     *
     * @param pointerDeltaPx 指针相对按下点的位移（CSS px）。
     */
    function applyThumbDrag(pointerDeltaPx: number): void {
        const geometries = scrollbarGeometries();
        if (dragAxis === "x") {
            scroll.setScrollLeft(
                dragStartScroll +
                    scrollDeltaFromThumbDrag(
                        pointerDeltaPx,
                        geometries.horizontal,
                        scroll.maxScrollLeft(),
                    ),
            );
            return;
        }
        scroll.setScrollTop(
            dragStartScroll +
                scrollDeltaFromThumbDrag(
                    pointerDeltaPx,
                    geometries.vertical,
                    scroll.maxScrollTop(),
                ),
        );
    }

    const onThumbPointerMove = (event: PointerEvent): void => {
        if (dragAxis === null) return;
        applyThumbDrag((dragAxis === "x" ? event.clientX : event.clientY) - dragStartPointer);
    };
    const onThumbPointerUp = (): void => {
        dragAxis = null;
    };

    registerListener(hScrollbarThumb, "pointerdown", makeThumbPointerDown("x") as EventListener);
    registerListener(vScrollbarThumb, "pointerdown", makeThumbPointerDown("y") as EventListener);
    // 拖拽的移动 / 抬起挂在 **window** 上：指针移出 thumb（甚至移出窗口）后
    // 仍要跟随，只挂 thumb 会在移出瞬间丢掉拖拽。
    //
    // 特殊说明：`window` 经 `globalThis` 安全解析而不是裸引用——本模块要在无 DOM 的
    // node 环境单测（见同目录 `.test.ts`），裸引用会让导入期就抛 ReferenceError。
    const windowTarget = (globalThis as { window?: Window }).window;
    if (windowTarget !== undefined) {
        registerListener(windowTarget, "pointermove", onThumbPointerMove as EventListener);
        registerListener(windowTarget, "pointerup", onThumbPointerUp as EventListener);
        registerListener(windowTarget, "pointercancel", onThumbPointerUp as EventListener);
    }

    /**
     * 造一个「点击轨道翻页」的按下处理器。
     *
     * 语义与原生滚动条一致：点在 thumb **之前** 向该方向翻一页，**之后** 向后翻一页。
     * 翻页量 = 该轴视口尺寸（浏览器的轨道翻页按整屏走）。目标值可能越界，交给
     * `ScrollKernel` 钳制（与所有写入路径同一约定）。
     *
     * 特殊说明：几何与当前位置都必须是**绘制坐标**口径（与 `scrollbarGeometries`
     * 同源），否则同步模式下判定用的 thumb 区间会整体偏移一个 offset，表现为
     * 「点轨道有时候没反应」。写回时再换回原生坐标（水平轴加偏移）。
     *
     * @param axis 轴别。
     * @returns 事件处理器。
     */
    function makeTrackPointerDown(axis: "x" | "y") {
        return (event: PointerEvent) => {
            event.preventDefault();
            event.stopPropagation();
            const track = event.currentTarget as HTMLElement;
            const trackRect = track.getBoundingClientRect();
            const view = scroll.get();
            const geometries = scrollbarGeometries();
            if (axis === "x") {
                const drawingScrollLeft = Math.max(0, view.scrollLeft - horizontalOffsetPx());
                const target = scrollTargetFromTrackClick(
                    event.clientX - trackRect.left,
                    geometries.horizontal,
                    drawingScrollLeft,
                    viewportWidthPx,
                );
                if (target !== null) scroll.setScrollLeft(target + horizontalOffsetPx());
                return;
            }
            const target = scrollTargetFromTrackClick(
                event.clientY - trackRect.top,
                geometries.vertical,
                view.scrollTop,
                viewportHeightPx,
            );
            if (target !== null) scroll.setScrollTop(target);
        };
    }

    if (hScrollbarTrack !== undefined) {
        registerListener(
            hScrollbarTrack,
            "pointerdown",
            makeTrackPointerDown("x") as EventListener,
        );
    }
    if (vScrollbarTrack !== undefined) {
        registerListener(
            vScrollbarTrack,
            "pointerdown",
            makeTrackPointerDown("y") as EventListener,
        );
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
            // 外部按**绘制坐标**写入；内核持有原生坐标，故加回同步偏移。
            scroll.setScrollLeft(px + horizontalOffsetPx());
        },

        setScrollTop(px) {
            if (disposed) return;
            scroll.setScrollTop(px);
        },

        setViewport(next) {
            if (disposed) return;
            scroll.setViewport(
                next.scrollLeft === undefined
                    ? { pxPerSec: next.pxPerSec }
                    : {
                          pxPerSec: next.pxPerSec,
                          scrollLeft: next.scrollLeft + horizontalOffsetPx(),
                      },
            );
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
                // 对外一律**绘制坐标**（与 `getAxis` 同一口径，也与旧实现
                // `timelineViewportNativeToState` 之后的值一致）。
                scrollLeft: view.scrollLeft - horizontalOffsetPx(),
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
            return scrollbarGeometries();
        },

        reclamp() {
            if (disposed) return;
            scroll.reclamp();
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
