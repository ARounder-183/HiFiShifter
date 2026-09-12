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
 * - **输入按轴分配所有权**：滚动条 thumb 拖拽与轨道翻页、以及**竖向键盘翻页**
 *   （PageUp / PageDown / Home / End，见 `scroll/keyboardScroll`）由宿主注册监听；
 *   其余手势（滚轮、中键平移、绘制）仍在面板与 `usePianoRollInteractions`。
 *   竖向键盘之所以必须收进来：内核模式下原生 scroller 只是被动镜像，其 `scroll`
 *   事件会被面板当作回声忽略，原生滚动的那次位移就没人采纳了——四个键会彻底失效。
 *   `dispose()` 释放全部监听、订阅、帧循环与尺寸观察。
 * - **不迁移缩放**：`pxPerSec` 由面板解析后经 `setPxPerSec` 写入；内核只据此算
 *   内容宽度与横向上限。竖向 `span` 同理，仍由面板决定。
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
 * - 复用：`renderKernel/scrollKernel`（视口真值）、`renderLoop`（帧调度）、
 *   `renderKernel/scrollbars`（滚动条几何）、`timelineAxis`（投影）、
 *   `scene/gridView`（实时视口快照与几何签名）、`scroll/keyboardScroll`（键盘目标）。
 * - 下游：`onFrame` 把投影交回面板绘制（标尺 / 网格 / 画布 / 波形 / 播放头）；
 *   `onScrollTopFrame` 把竖向真值逐帧交回面板刷新其值域视口 ref。
 */

import { readDevicePixelRatio } from "../../../../../utils/devicePixelLine";
import { invokeGridRedrawHandler } from "../../../timeline/gridRedrawBridge";
import {
    scrollDeltaFromThumbDrag,
    scrollTargetFromTrackClick,
} from "../../../renderKernel/scrollbars";
import { createGlCanvas, type GlCanvasHandle } from "../../../renderKernel/gl/glContext";
import {
    createPolylineProgram,
    type PolylineProgram,
} from "../../../renderKernel/gl/polylineProgram";
import { buildPolylineVertices } from "../../../renderKernel/gl/polylineGeometry";
import { decimatePolylinePoints } from "../../../renderKernel/gl/polylineDecimation";
import {
    projectClipboardPreviewPoints,
    projectCurvePoints,
    projectDetectedCurvePoints,
} from "../scene/curvePoints";
import { CLIP_INSTANCE_FLOATS, writeFlatInstance } from "../../../renderKernel/gl/instanceLayout";
import { createSdfBoxProgram, type SdfBoxProgram } from "../../../renderKernel/gl/sdfBoxProgram";
import { createRenderLoop } from "../../../renderKernel/renderLoop";
import { createScrollKernel, type TimelineViewportState } from "../../../renderKernel/scrollKernel";
import { isBlackKey, midiToLabel } from "../../utils";
import {
    createPianoRollGlyphs,
    type PianoRollGlyphs,
    type TextRequest,
} from "../glyph/pianoRollGlyphs";
import { buildAxisMarkInstances, resolveAxisKind } from "../scene/axisMarkInstances";
import { buildKeyboardInstances } from "../scene/keyboardInstances";
import { createGlyphProgram, type GlyphProgram } from "../../../renderKernel/gl/glyphProgram";
import type { GlyphQuad } from "../../../renderKernel/gl/glyphQuads";
import {
    createTimelineAxis,
    secToViewportPx,
    strokePx,
    type TimelineAxis,
} from "../../../renderKernel/timelineAxis";
import type { FlatInstance } from "../../../renderKernel/instanceTypes";
import { wholeDevicePxLength } from "../../../../../utils/devicePixelLine";
import {
    buildPitchGridInstances,
    buildValueGridInstances,
    type GridInstance,
} from "../scene/gridInstances";
import {
    gridGeometrySignature,
    keyboardGeometrySignature,
    resolveLiveGridView,
    type LiveGridView,
} from "../scene/gridView";
import {
    resolvePianoRollScrollbarGeometries,
    type PianoRollScrollbarGeometries,
} from "../scroll/scrollbarSpec";
import {
    centerFromKernelScrollTop,
    kernelScrollTopFromCenter,
    PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX,
} from "../scroll/verticalValueScroll";
import { resolveKeyboardScrollTarget } from "../scroll/keyboardScroll";
import type { PianoRollKernelData, PianoRollGridSpec } from "./pianoRollKernelData";

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
     * GL 场景层画布（阶段 2）。缺省 / 传 null 时只用 Canvas2D 路径。
     *
     * 特殊说明：宿主**不**创建画布，只接收面板已经渲染好的 `<canvas>`。画布的
     * 可见性、层级（z-index）与是否挂载由 React 侧决定——宿主只负责在里面画。
     */
    readonly glCanvas?: HTMLCanvasElement | null;
    /**
     * 键盘轴 GL 画布（阶段 2，Task 4）。缺省 / null 时不画键盘。
     *
     * 【为什么需要**第二块**画布】键盘位于左侧轴列，DOM 上是主画布的**兄弟**
     * 元素（固定 56px 宽、不随横向滚动移动），而主 GL 画布在 sticky 视口层内、
     * 宽度等于视口宽。把键盘画在主画布上会跟着 sticky 层一起被裁切，位置也对不上。
     * 因此键盘单独一块画布，几何坐标即"轴列视口坐标"，与 Canvas2D 的布局一一对应。
     */
    readonly glAxisCanvas?: HTMLCanvasElement | null;
    /**
     * 动态叠加层 GL 画布（阶段 2，Task 6）：播放头与选区。
     *
     * 【为什么必须是**独立的、位于曲线之上的**画布】播放帧只改播放头位置。曲线留在
     * Canvas2D 细节层（阶段 2 不迁移曲线），若播放头也画在 Curve 画布上，每帧都得
     * 重绘整张画布（含曲线）——那正是阶段 2 要消除的开销。把播放头放到**曲线上方的
     * 独立 GL 画布**后：曲线画布只在内容/滚动/缩放变化时重绘，播放帧只清一块空画布
     * + 提交 1 个实例。
     *
     * 层序：GL 静态层（网格/键盘/轴文字）→ Canvas2D 曲线层 → 本层（播放头/选区）。
     */
    readonly glOverlayCanvas?: HTMLCanvasElement | null;
    /**
     * 键盘轴宽度（CSS px）。缺省用 `AXIS_W` 的值（56）。
     *
     * 特殊说明：由调用方注入而不是在此硬编码——轴宽是**布局**常量，属于面板；
     * 宿主硬编码会在轴宽调整时悄悄画错宽度。
     */
    readonly axisWidthPx?: number;
    /**
     * 是否启用 GL 场景层。缺省 `false`。
     *
     * 特殊说明：这是**运行时开关**而不是模块常量，便于测试两种路径而不必重新
     * 导入模块（模块级常量在 dev 下还会被 HMR 缓存）。
     */
    readonly glSceneEnabled?: boolean;
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
    /**
     * **用户手势**改变了水平滚动位置（绘制坐标）。
     *
     * 【为什么必须与 `onScrollLeftCommit` 分开】宿主每帧把真值写回原生 scroller
     * （镜像），这会触发 `scroll` 事件。面板若在该事件里把位置推给共享视口，就会把
     * 宿主的**自身回写**误当成用户滚动；若不推送，则**滚动条拖拽 / 轨道翻页**这类
     * 真实手势又会漏掉同步（旧实现里拖原生滚动条是会被同步的）。
     *
     * 两者无法靠比较数值区分（拖拽后的镜像回写与用户原生滚动值相同），因此由
     * **来源**区分：只有宿主自己解析出的用户手势（拖 thumb、点轨道翻页）走本回调。
     * 面板据此推送共享视口，而不必依赖 `scroll` 事件。
     *
     * @param drawingScrollLeft 新的水平位置（绘制坐标）。
     */
    readonly onUserScrollLeft?: (drawingScrollLeft: number) => void;
    /**
     * **竖向**滚动由内核改变时通知面板（含用户手势与键盘翻页）。
     *
     * 【为什么竖向要单独开一条通道】横向有 `onFrame` 逐帧回报（面板据此更新
     * `scrollLeftRef` 并重绘），竖向却**没有**：`axis.scrollTopPx` 只喂给
     * `applyScrollLayers` 之外的地方。于是面板持有的值域视口（`pitchViewRef` /
     * `paramViewsRef`）在竖向滚动期间只能靠原生 `scroll` 事件 + 反向回写来"猜"，
     * 而那条路正是镜像回声的来源（内核刚写下的位置被滞后读回并覆盖回去）。
     *
     * 本回调把内核的竖向真值直接交给面板，面板据此**正向**刷新自己的视口 ref。
     * 回声因此不再承担任何职责，可以在面板侧被安全忽略。
     *
     * 特殊说明：每帧都可能触发；实现必须只做赋值（不进 React），与
     * `onScrollLeftFrame` 同一约定。
     *
     * @param scrollTopPx 内核竖向真值（CSS px，0..1600）。
     */
    readonly onScrollTopFrame?: (scrollTopPx: number) => void;
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
    /**
     * 读取 GL 场景层的运行状态（供面板提示与浏览器验证使用）。
     *
     * 【为什么放进公开句柄】GL 走的是"失败软着陆"策略：不可用时静默退回 Canvas2D，
     * 面板外观完全正常。于是"GL 到底有没有生效"无法从画面上判断——必须能读到
     * 明确状态，否则验证会误把"退回 Canvas2D"当成"GL 正常"。
     *
     * @returns `active` 表示 GL 已建好并在绘制；
     *          `failureReason` 在尝试启用但失败时给出原因（未启用时为 null）；
     *          `gridInstanceCount` 是当前上传的几何实例数（0 = 没有网格）。
     */
    getGlStatus(): {
        readonly active: boolean;
        readonly failureReason: string | null;
        readonly gridInstanceCount: number;
        /** 键盘层已上传的实例数（仅 pitch 参数非零）。 */
        readonly keyboardInstanceCount: number;
        /**
         * 当前用于 GL 几何的网格输入快照（诊断用）。
         *
         * 【为什么暴露它】两条渲染路径必须几何一致，而"不一致"只体现在像素上、
         * 极难归因。有了这份快照，浏览器验证可以直接复算构建器的输出并与像素
         * 对照，而不必从像素反推 view（那条路会引入大量猜测）。
         */
        readonly gridSpec: PianoRollGridSpec | null;
    };
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
    const { data, sync, onFrame, onScrollLeftCommit, onUserScrollLeft, onScrollTopFrame } = args;

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

    // ── GL 场景层（阶段 2）──────────────────────────────────────────
    //
    // 【职责】把**静态**图层（网格等）画到独立画布上：几何按内容坐标构建并常驻
    // GPU，滚动帧只更新 `u_viewOrigin`、播放帧不重绘几何。这正是阶段 2 的收益点。
    //
    // 【失败必须是软着陆】WebGL2 不可用（老驱动 / 远程桌面 / 上下文数超限）是
    // 预期内的环境差异。任一步失败都只记原因并**退回 Canvas2D 路径**——绝不能让
    // 参数编辑器变空白。因此这里不抛错，只把 `gl` 置空。
    let glHandle: GlCanvasHandle | null = null;
    let glProgram: SdfBoxProgram | null = null;
    let glFailureReason: string | null = null;
    if (args.glSceneEnabled === true && args.glCanvas != null) {
        try {
            glHandle = createGlCanvas(args.glCanvas);
            if (glHandle === null) {
                glFailureReason = "WebGL2 不可用";
            } else {
                glProgram = createSdfBoxProgram(glHandle.gl);
            }
        } catch (error) {
            glHandle = null;
            glProgram = null;
            glFailureReason = error instanceof Error ? error.message : String(error);
        }
    }

    // 键盘轴 GL 层（Task 4）：独立画布 + 独立 program / 实例缓冲。
    // 与主画布各自持有 program，是因为两者尺寸与绘制时机不同（轴列不随横向滚动），
    // 共用会引入"谁负责 resize"的耦合。
    let glAxisHandle: GlCanvasHandle | null = null;
    let glAxisProgram: SdfBoxProgram | null = null;
    if (args.glSceneEnabled === true && args.glAxisCanvas != null) {
        try {
            glAxisHandle = createGlCanvas(args.glAxisCanvas);
            if (glAxisHandle === null) {
                // 主画布可能已经成功；键盘单独失败时只跳过键盘，不影响网格。
                glAxisHandle = null;
            } else {
                glAxisProgram = createSdfBoxProgram(glAxisHandle.gl);
            }
        } catch {
            glAxisHandle = null;
            glAxisProgram = null;
        }
    }
    /** 键盘轴宽度（CSS px）。 */
    const axisWidthPx =
        Number.isFinite(args.axisWidthPx) && (args.axisWidthPx as number) > 0
            ? (args.axisWidthPx as number)
            : 56;
    // 字形层（Task 5）：与键盘共用同一块轴画布（文字画在键体/刻度线之上），
    // 因此共用 `glAxisHandle` 的上下文，program 独立。
    let glGlyphProgram: GlyphProgram | null = null;
    let glGlyphs: PianoRollGlyphs | null = null;
    if (glAxisHandle !== null) {
        try {
            glGlyphProgram = createGlyphProgram(glAxisHandle.gl);
            glGlyphs = createPianoRollGlyphs({ dpr: readDevicePixelRatio() });
            if (glGlyphs === null) {
                glGlyphProgram = null;
            }
        } catch {
            glGlyphProgram = null;
            glGlyphs = null;
        }
    }
    /** 上一帧上传过的图集页（按页增量上传，避免每帧重传整张图集）。 */
    const uploadedAtlasPages = new Set<number>();
    /** 本帧待绘制的文字四边形。 */
    let glTextQuads: GlyphQuad[] = [];

    // 曲线层（阶段 3）：复用**主 GL 画布**（网格/键盘之上），program 独立。
    // 与网格共用一个上下文是有意的：曲线与网格是同一张画布的先后绘制，
    // 共用上下文避免再多申请一个 WebGL context（上下文数量有上限）。
    let glCurveProgram: PolylineProgram | null = null;
    if (glHandle !== null) {
        try {
            glCurveProgram = createPolylineProgram(glHandle.gl);
        } catch {
            glCurveProgram = null;
        }
    }

    // 动态叠加层（Task 6）：独立画布 + program，位于曲线层之上。
    let glOverlayHandle: GlCanvasHandle | null = null;
    let glOverlayProgram: SdfBoxProgram | null = null;
    if (args.glSceneEnabled === true && args.glOverlayCanvas != null) {
        try {
            glOverlayHandle = createGlCanvas(args.glOverlayCanvas);
            if (glOverlayHandle === null) {
                glOverlayHandle = null;
            } else {
                glOverlayProgram = createSdfBoxProgram(glOverlayHandle.gl);
            }
        } catch {
            glOverlayHandle = null;
            glOverlayProgram = null;
        }
    }
    /** 叠加层实例缓冲与上传状态。 */
    let glOverlayInstances = new Float32Array(0);
    let glOverlayUploadedCount = 0;

    /** 键盘轴实例缓冲与上传状态。 */
    let glAxisInstances = new Float32Array(0);
    let glAxisUploadedCount = 0;
    let glAxisGeometryUploaded = false;
    let lastKeyboardSignature = "";

    /** GL 实例缓冲（跨帧复用，容量按需增长）。 */
    let glInstances = new Float32Array(0);
    /** 当前已上传到 GPU 的实例数。 */
    let glUploadedCount = 0;
    /** 几何是否已上传（false 时下一帧走 `render` 全量上传）。 */
    let glGeometryUploaded = false;
    /** 上一次构建几何用的**内容签名**（变化才重建，见 `gridGeometrySignature`）。 */
    let lastGridSignature = "";

    /**
     * 解析**本帧的实时视口**（竖向中心取自内核真值）。
     *
     * 【为什么不能直接用 `spec.view`】那是面板在 React render 期写下的快照，而
     * 竖向滚动的真值在内核（`scrollTop`）且**没有逐帧回写面板的通道**——拖竖向
     * 滚动条 / 中键竖向平移 / 键盘翻页都不进 React。用快照会让中心在竖向滚动期间
     * 恒定，几何因此永不重建（实测：内核中心 72 → 79.29，GL 画布像素逐点相同）。
     * 详见 `scene/gridView` 的文件头。
     *
     * 特殊说明：`span` 仍取自快照——缩放由面板决定，内核不参与。
     *
     * @param spec 当前网格输入；空时返回 null。
     * @returns 实时视口；无网格时为 null。
     */
    function liveGridView(spec: PianoRollGridSpec | null | undefined): LiveGridView | null {
        if (spec == null) return null;
        return resolveLiveGridView({
            absMin: spec.absMin,
            absMax: spec.absMax,
            span: spec.view.span,
            scrollTop: scroll.get().scrollTop,
        });
    }

    /**
     * 计算网格几何的**内容签名**（含实时竖向中心）。
     *
     * 【作用】几何只在这些输入变化时才需要重建。把输入拼成字符串做比较，比逐字段
     * 深比较便宜，也比"每帧重建"省下大量分配——网格逐行构建在 formant 参数下
     * 可达近百个实例，每帧重建纯属浪费。
     *
     * 【必须含实时中心】签名是"要不要重建"的唯一判据。含快照中心等于用一个在竖向
     * 滚动期间恒定的量，重建条件退化成"只有 React 重渲染才重建"——这就是"拖竖向
     * 滚动条网格不动"的直接原因。
     *
     * 特殊说明：**不含横向滚动位置**——横线横跨整个视口、竖线的 y 也已在视口坐标
     * 里算好，因此横向滚动确实无需重建（"横向滚动零重建"成立）。视口宽高要含
     * （分别决定横线长度与可见行数）；dpr 要含（决定半像素取向与线厚）。
     *
     * @param spec 当前网格输入。
     * @returns 内容签名；无网格时为空串。
     */
    function gridSignature(spec: PianoRollGridSpec | null | undefined): string {
        const view = liveGridView(spec);
        if (spec == null || view === null) return "";
        return gridGeometrySignature({
            kind: spec.kind,
            view,
            absMin: spec.absMin,
            absMax: spec.absMax,
            viewportWidthPx,
            viewportHeightPx,
            dpr: readDevicePixelRatio(),
            strongRgba: spec.strongRgba,
            weakRgba: spec.weakRgba,
        });
    }

    /**
     * 按当前镜像重建网格几何并标记需要重新上传。
     *
     * 流程：取镜像的 grid spec → 分派到对应构建器（pitch / 值域步进）→ 写入实例缓冲。
     *
     * 特殊说明：构建器返回的 `GridInstance` 里 `w` 是**横向范围**、`h` 是**线厚**，
     * 而 `writeFlatInstance` 的矩形语义就是 `(x, y, w, h)`，两者一致、无需换序。
     *
     * @param spec 当前网格输入；null 时清空几何。
     */
    function rebuildGlGeometry(spec: PianoRollGridSpec | null | undefined): void {
        const view = liveGridView(spec);
        const items: GridInstance[] =
            spec == null || view === null
                ? []
                : spec.kind === "pitch"
                  ? buildPitchGridInstances({
                        view,
                        absMin: spec.absMin,
                        absMax: spec.absMax,
                        heightPx: viewportHeightPx,
                        viewportWidthPx,
                        dpr: readDevicePixelRatio(),
                        valueToY: spec.valueToY,
                        colorC: spec.strongRgba,
                        colorOther: spec.weakRgba,
                    })
                  : buildValueGridInstances({
                        kind: spec.kind,
                        view,
                        heightPx: viewportHeightPx,
                        viewportWidthPx,
                        dpr: readDevicePixelRatio(),
                        valueToY: spec.valueToY,
                        strongRgba: spec.strongRgba,
                        weakRgba: spec.weakRgba,
                    });

        const needed = items.length * CLIP_INSTANCE_FLOATS;
        if (glInstances.length < needed) glInstances = new Float32Array(needed);
        for (let index = 0; index < items.length; index += 1) {
            writeFlatInstance(glInstances, index, items[index]);
        }
        glUploadedCount = items.length;
        // 标记需要重新上传：实例数据变了，不能用 `repaint`（它只改 uniform）。
        glGeometryUploaded = false;
    }

    /**
     * 绘制 GL 场景层。
     *
     * 流程：尺寸同步 → 清屏 → 几何变化时 `render` 全量上传，否则 `repaint` 只更新
     * 视口原点。
     *
     * 【为什么视口原点恒为 (0, 0)】横向网格线**横跨整个视口**、与滚动位置无关
     * （Canvas2D 路径就是 `moveTo(0, y) → lineTo(w, y)`）；竖向网格线的 y 同样是
     * 在视口坐标里算好的（见 `rebuildGlGeometry` 传 `heightPx = viewportHeightPx`）。
     * 因此网格几何**本就是视口坐标**，不能再减视口原点——减了会把整组线按滚动量
     * 平移，表现为"参数编辑器左侧有一条与滚动量同宽的空白带"（实测静置时左侧
     * 200px 没有网格线，恰好等于同步偏移量）。
     *
     * 特殊说明：这**不**违反"滚动零重建"——恰恰相反：几何是视口坐标，滚动时
     * 既不需要重建几何、也不需要改 uniform，是真正的零成本滚动。后续把曲线等
     * **内容坐标**图层搬上 GL 时，才需要用到视口原点平移。
     */
    function drawGlScene(): void {
        if (glHandle === null || glProgram === null) return;
        const target = glHandle.resize(viewportWidthPx, viewportHeightPx, readDevicePixelRatio());
        glHandle.clear();
        if (glUploadedCount === 0) return;
        if (glGeometryUploaded) {
            glProgram.repaint(target, 0, 0);
        } else {
            glProgram.render(glInstances, glUploadedCount, target, 0, 0);
            glGeometryUploaded = true;
        }
    }

    /**
     * 计算键盘几何的**内容签名**（变化才重建）。
     *
     * 特殊说明：含轴宽与视口高——两者都直接决定键体几何；**不含**横向滚动位置
     * （键盘不随横向滚动变化），因此横向滚动是零重建的。
     *
     * @param spec 当前网格输入（键盘复用它的值域信息）。
     * @returns 内容签名；非音高参数（无键盘）时为空串。
     */
    function keyboardSignature(spec: PianoRollGridSpec | null | undefined): string {
        const view = liveGridView(spec);
        if (spec == null || view === null) return "";
        return keyboardGeometrySignature({
            kind: spec.kind,
            view,
            absMin: spec.absMin,
            absMax: spec.absMax,
            viewportHeightPx,
            axisWidthPx,
            dpr: readDevicePixelRatio(),
            strongRgba: spec.strongRgba,
            weakRgba: spec.weakRgba,
            whiteKeyRgba: spec.whiteKeyRgba,
            blackKeyRgba: spec.blackKeyRgba,
            blackKeyGradientRgba: spec.blackKeyGradientRgba,
            cSeparatorRgba: spec.cSeparatorRgba,
            keySeparatorRgba: spec.keySeparatorRgba,
            axisBorderRgba: spec.axisBorderRgba,
        });
    }

    /**
     * 重建键盘轴几何并标记需要重新上传。
     *
     * 特殊说明 1：只有**音高**参数才有键盘（其余参数画的是刻度标签），因此非 pitch
     * 时清空几何——否则切到别的参数后键盘会残留在 GL 画布上。
     *
     * 特殊说明 2：视口走 `liveGridView`（竖向中心取自内核真值），与签名同源。
     * 两边取不同来源会让"签名说该重建、几何却按旧视口建"，线条依旧不动。
     *
     * @param spec 当前网格输入。
     */
    function rebuildKeyboardGeometry(spec: PianoRollGridSpec | null | undefined): void {
        const view = liveGridView(spec);
        if (
            spec == null ||
            view === null ||
            spec.kind !== "pitch" ||
            spec.whiteKeyRgba === undefined
        ) {
            glAxisUploadedCount = 0;
            glAxisGeometryUploaded = false;
            return;
        }
        const items = buildKeyboardInstances({
            view,
            absMin: spec.absMin,
            absMax: spec.absMax,
            heightPx: viewportHeightPx,
            axisWidthPx,
            dpr: readDevicePixelRatio(),
            valueToY: spec.valueToY,
            isBlackKey,
            whiteKeyRgba: spec.whiteKeyRgba,
            blackKeyRgba: spec.blackKeyRgba ?? spec.whiteKeyRgba,
            blackKeyGradientRgba: spec.blackKeyGradientRgba ?? [0, 0, 0, 0],
            cSeparatorRgba: spec.cSeparatorRgba ?? [0, 0, 0, 0],
            keySeparatorRgba: spec.keySeparatorRgba ?? [0, 0, 0, 0],
            axisBorderRgba: spec.axisBorderRgba ?? [0, 0, 0, 0],
        });
        const needed = items.length * CLIP_INSTANCE_FLOATS;
        if (glAxisInstances.length < needed) glAxisInstances = new Float32Array(needed);
        for (let index = 0; index < items.length; index += 1) {
            writeFlatInstance(glAxisInstances, index, items[index]);
        }
        glAxisUploadedCount = items.length;
        glAxisGeometryUploaded = false;
    }

    /**
     * 绘制曲线图层（阶段 3）。
     *
     * 流程：逐层投影采样值 → **按设备像素列抽稀** → 构建折线顶点 → 一次 upload + draw call。
     *
     * 【为什么每帧重建】滚动/缩放会改变**每个点**的视口 x，因此曲线层没有像网格层
     * 那样的"零重建"路径——这是它与静态图层的本质区别。
     *
     * 【为什么必须抽稀（实测依据）】取数侧 `stride = 1`（200 点/秒）不可改——编辑路径
     * 依赖全分辨率（`selectionEditData` 的零 IPC 快路径要求 `stride === 1`）。于是最小
     * 缩放（4 px/s，视口覆盖 466 秒）下视口内有 ~93,200 个采样点，却只有 3,728 个设备
     * 像素列，**每列 25 个点渲染在同一个像素上**。这些点全额进入几何构建：
     *
     * | | 点数 | 顶点数 | 每帧成本 |
     * |---|---|---|---|
     * | 抽稀前 | 213,595 | 1,922,315（30 MB） | 32 ms |
     * | 抽稀后 | ~22,000 | ~200,000（3 MB） | ~3 ms |
     *
     * 抽稀按列保留 y 的 min/max，因此**不丢包络**（曲线折返的极值仍在）；弧长取自
     * 原始序列，虚线相位不会漂移。详见 `polylineDecimation` 的精度契约。
     *
     * 特殊说明 1：绘制顺序即数组顺序（后画的在上），因此**面板必须按期望的层序**
     * 提供图层：参考线/检测/副参数在下，原始/编辑曲线在中，选区高亮/剪贴板在上。
     * 这与 Canvas2D 路径的调用顺序一一对应。
     *
     * 特殊说明 2：`u_viewOrigin` 传 (0, 0)——投影函数已用 `axis.secToViewportPx`
     * 减去了 scrollLeft，所以几何本身就是**视口坐标**，与网格/键盘层同一约定。
     *
     * 特殊说明 3：抗锯齿宽度取 `1/dpr`（1 个设备像素），与 Canvas2D 的边缘过渡
     * 尺度一致；`u_halfWidth` 取线宽的一半（**不含**几何的 AA 余量）。
     */
    function drawGlCurves(): void {
        if (glCurveProgram === null || glHandle === null) return;
        const layers = data().curves;
        if (layers === null || layers === undefined || layers.length === 0) return;

        const target = glHandle.resize(viewportWidthPx, viewportHeightPx, readDevicePixelRatio());
        const dpr = readDevicePixelRatio();
        const axis = currentAxis();

        for (const layer of layers) {
            // 每条图层用自己的投影（副参数值域与主参数不同，见 spec 说明）。
            const valueToY = layer.valueToY;
            // 三种投影的时间基准不同（见 `PianoRollCurveLayer.projection` 说明）：
            // 检测曲线自带绝对起始秒且要跳无声帧，剪贴板预览锚定选区起点，
            // 其余按 `startFrame + i × stride`。混用会造成平移或尖刺。
            const points =
                layer.projection === "detected"
                    ? projectDetectedCurvePoints({
                          midiCurve: layer.values,
                          curveStartSec: layer.curveStartSec ?? 0,
                          framePeriodMs: layer.framePeriodMs,
                          axis,
                          valueToY,
                      })
                    : layer.projection === "clipboard"
                      ? projectClipboardPreviewPoints({
                            values: layer.values,
                            param: layer.param,
                            framePeriodMs: layer.framePeriodMs,
                            selStartSec: layer.clipStartSec ?? 0,
                            selEndSec: layer.clipEndSec ?? 0,
                            axis,
                            valueToY,
                        })
                      : projectCurvePoints({
                            values: layer.values,
                            param: layer.param,
                            startFrame: layer.startFrame,
                            stride: layer.stride,
                            framePeriodMs: layer.framePeriodMs,
                            axis,
                            valueToY,
                        });
            if (points.length < 2) continue;

            // 按设备像素列抽稀（见上方"为什么必须抽稀"）。
            const reduced = decimatePolylinePoints({ points, dpr, viewportWidthPx });
            if (reduced.points.length < 2) continue;

            const vertices = buildPolylineVertices({
                points: reduced.points,
                lineWidth: layer.lineWidthPx,
                // 与 Canvas2D 的默认 lineJoin="miter" + 默认 miterLimit=10 一致
                // （本工程从未设置过 lineJoin，见 Phase 3 计划的 R2）。
                miterLimit: 10,
                // 抽稀后必须用**原始**弧长推进虚线相位，否则图案会随缩放漂移。
                alongOverride: reduced.along ?? undefined,
            });
            if (vertices.length === 0) continue;

            glCurveProgram.draw({
                vertices,
                vertexCount: vertices.length / 4,
                target,
                viewOriginX: 0,
                viewOriginY: 0,
                halfWidthPx: layer.lineWidthPx / 2,
                color: layer.rgba,
                aaWidthPx: 1 / dpr,
                dash: layer.dash ?? null,
                clipRect: layer.clipRect ?? null,
            });
        }
    }

    /**
     * 重建动态叠加层的几何（选区块 + 播放头）。
     *
     * 流程：选区块（半透明填充 + 边框）→ 播放头竖线（单实例）→ 写入实例缓冲。
     *
     * 【为什么播放头是"一个实例"就够】播放帧只改播放头的 x：实例缓冲的其余部分
     * （选区块）不变。因此这里按"每帧重建、但只有 2~3 个实例"来做——重建成本可以
     * 忽略，而**画布清屏范围**从"整张曲线画布"降到"一块空画布"，这才是收益所在。
     *
     * 【为什么选区也放这一层】选区块是时间区间，滚动/缩放会改变它的像素位置，
     * 但它与曲线层的内容变化时机**不同**（拖动选区时曲线不变）。放在叠加层后，
     * 拖动选区只重绘叠加层，曲线层保持缓存。
     *
     * 特殊说明：坐标是**绘制坐标**（视口坐标），因此 `u_viewOrigin` 传 (0, 0)——
     * 与网格/键盘层同一约定；`axis.secToViewportPx` 内部已减去 scrollLeft。
     */
    function rebuildOverlayGeometry(): void {
        const overlay = data().overlay;
        const axis = currentAxis();
        const h = viewportHeightPx;
        const items: FlatInstance[] = [];

        // 【为什么这里只有播放头，没有选区块】选区块属于**曲线之下**的图层：
        // Canvas2D 路径先画选区、再画各条曲线。而本叠加层位于曲线**之上**，
        // 若把选区放这里，它的半透明填充与边框会盖住曲线——与迁移前的观感相反。
        // 阶段 2 曾把两者一起搬进来，阶段 3 修正为：选区块回主画布（`render.ts` 的
        // `selection` 段），叠加层只保留播放头（它在 Canvas2D 里本就是最上层）。

        if (overlay?.playheadSec !== null && overlay?.playheadSec !== undefined) {
            // 与 render.ts:1148-1154 同一对齐：线宽取整物理像素，奇数宽度补半个设备像素。
            const phWidthPx = wholeDevicePxLength(1, axis.dpr);
            const phx = strokePx(axis, secToViewportPx(axis, overlay.playheadSec), phWidthPx);
            items.push({
                x: phx - phWidthPx / 2,
                y: 0,
                w: phWidthPx,
                h,
                rgba: overlay.playheadRgba ?? [0, 0, 0, 0.2],
            });
        }

        const needed = items.length * CLIP_INSTANCE_FLOATS;
        if (glOverlayInstances.length < needed) glOverlayInstances = new Float32Array(needed);
        for (let index = 0; index < items.length; index += 1) {
            writeFlatInstance(glOverlayInstances, index, items[index]);
        }
        glOverlayUploadedCount = items.length;
    }

    /**
     * 绘制动态叠加层。
     *
     * 特殊说明：视口原点恒为 (0, 0)——几何本就是视口坐标（见
     * `rebuildOverlayGeometry`）。每帧都重建几何：实例数只有 1~3 个，重建成本
     * 远低于"判断是否需要重建"的复杂度；而清屏范围是**空画布**，与曲线层无关。
     */
    function drawGlOverlay(): void {
        if (glOverlayHandle === null || glOverlayProgram === null) return;
        const target = glOverlayHandle.resize(
            viewportWidthPx,
            viewportHeightPx,
            readDevicePixelRatio(),
        );
        glOverlayHandle.clear();
        if (glOverlayUploadedCount === 0) return;
        glOverlayProgram.render(glOverlayInstances, glOverlayUploadedCount, target, 0, 0);
    }

    /**
     * 收集本帧要绘制的文字请求（键盘音名 + 数值轴刻度标签）。
     *
     * 流程：按网格种类分派 → pitch 走键盘音名（仅键高足够时）→ 其余走刻度标签与
     * 刻度线 → 组装为 `TextRequest[]`（视口坐标，中间对齐）。
     *
     * 【为什么键高不足 6px 时不画音名】`render.ts:464` 有 `keyH >= 6` 的门槛：
     * 键太矮时音名会互相重叠成一片。这是既有的可读性保护，必须保留。
     *
     * 【坐标约定】全部是**轴列视口坐标**（x 从轴列左缘 0 起、y 为视口 y），与键盘
     * 几何同一套坐标，因此 GL 侧 `u_viewOrigin` 传 (0, 0)。
     *
     * @param spec 当前网格输入。
     * @returns 文字请求序列；无字形渲染器时为空数组。
     */
    function collectAxisTextRequests(spec: PianoRollGridSpec | null | undefined): TextRequest[] {
        if (spec === null || spec === undefined) return [];
        // 实时视口（竖向中心取自内核真值）：与几何同源，否则文字与线条会错位。
        const view = liveGridView(spec);
        if (view === null) return [];
        const family = spec.fontFamily ?? "sans-serif";
        const requests: TextRequest[] = [];

        if (spec.kind === "pitch") {
            const whiteRgba = spec.whiteKeyLabelRgba;
            const blackRgba = spec.blackKeyLabelRgba;
            const cRgba = spec.cLabelRgba;
            if (whiteRgba === undefined || blackRgba === undefined || cRgba === undefined) {
                return [];
            }
            const range = spec.absMax - spec.absMin;
            if (!Number.isFinite(range) || range <= 0) return [];
            const span = Math.min(Math.max(view.span, 1e-6), range);
            const min = Math.min(Math.max(view.center - span / 2, spec.absMin), spec.absMax - span);
            const startMidi = Math.min(Math.max(Math.floor(min), spec.absMin), spec.absMax);
            const endMidi = Math.min(Math.max(Math.ceil(min + span), spec.absMin), spec.absMax);

            for (let midi = startMidi; midi < endMidi; midi += 1) {
                const y0 = spec.valueToY(midi, viewportHeightPx);
                const y1 = spec.valueToY(midi + 1, viewportHeightPx);
                const top = Math.min(y0, y1);
                const keyH = Math.max(1, Math.max(y0, y1) - top);
                // 与 render.ts:464 一致：键高不足 6px 时省略音名
                if (keyH < 6) continue;
                const black = isBlackKey(midi);
                const pc = ((midi % 12) + 12) % 12;
                requests.push({
                    text: midiToLabel(midi),
                    fontKey: black
                        ? `8px ${family}`
                        : pc === 0
                          ? `bold 9px ${family}`
                          : `9px ${family}`,
                    // 黑键标签 x=3、白键 x=4（render.ts:488/497）
                    x: black ? 3 : 4,
                    y: top + keyH / 2,
                    align: "left",
                    baseline: "middle",
                    rgba: black ? blackRgba : pc === 0 ? cRgba : whiteRgba,
                });
            }
            return requests;
        }

        // 非音高参数：数值轴刻度标签
        const labelRgba = spec.tensionLabelRgba;
        if (labelRgba === undefined || spec.paramName === undefined) return [];
        const marks = buildAxisMarkInstances({
            kind: resolveAxisKind(spec.paramName),
            view,
            heightPx: viewportHeightPx,
            axisWidthPx,
            dpr: readDevicePixelRatio(),
            valueToY: spec.valueToY,
            paramName: spec.paramName,
        });
        for (const mark of marks) {
            requests.push({
                text: mark.label,
                fontKey: `10px ${family}`,
                // render.ts:555 的标签锚点 x=6
                x: 6,
                y: mark.line.y + mark.line.h / 2 - 0.5,
                align: "left",
                baseline: "middle",
                rgba: labelRgba,
            });
        }
        return requests;
    }

    /**
     * 重建数值轴刻度线几何（非音高参数）。
     *
     * 特殊说明 1：pitch 参数的轴列是键盘（由 `rebuildKeyboardGeometry` 负责），
     * 此函数只处理数值轴；两者互斥，因此可以共用同一块轴画布与实例缓冲。
     *
     * 特殊说明 2：视口走 `liveGridView`（竖向中心取自内核真值），与签名同源。
     *
     * @param spec 当前网格输入。
     * @returns 刻度线实例数（供调用方决定是否上传）。
     */
    function rebuildAxisMarkGeometry(spec: PianoRollGridSpec | null | undefined): number {
        const view = liveGridView(spec);
        if (
            spec === null ||
            spec === undefined ||
            view === null ||
            spec.kind === "pitch" ||
            spec.paramName === undefined ||
            spec.tensionLineRgba === undefined
        ) {
            return 0;
        }
        const marks = buildAxisMarkInstances({
            kind: resolveAxisKind(spec.paramName),
            view,
            heightPx: viewportHeightPx,
            axisWidthPx,
            dpr: readDevicePixelRatio(),
            valueToY: spec.valueToY,
            paramName: spec.paramName,
        });
        // `lineOnly` 的项只画标签、没有配对的分隔线（见 axisMarkInstances 的说明）
        const lines = marks.filter((m) => !m.lineOnly);
        const needed = lines.length * CLIP_INSTANCE_FLOATS;
        if (glAxisInstances.length < needed) glAxisInstances = new Float32Array(needed);
        for (let index = 0; index < lines.length; index += 1) {
            const line = lines[index].line;
            writeFlatInstance(glAxisInstances, index, {
                x: line.x,
                y: line.y,
                w: line.w,
                h: line.h,
                rgba: spec.tensionLineRgba,
            });
        }
        glAxisUploadedCount = lines.length;
        glAxisGeometryUploaded = false;
        return lines.length;
    }

    /**
     * 绘制键盘轴 GL 层。
     *
     * 流程：尺寸同步 → 清屏 → 先画几何（键体 / 刻度线，`sdfBox` program）→
     * 再画文字（字形 program，覆盖在几何之上）。
     *
     * 【已知的亚像素差异】文字位置与 Canvas2D 相差至多 **0.5 个设备像素**（纵向）。
     * 原因是图集槽位是整数像素对齐的：字形在槽内按整数设备像素光栅化，而
     * `middle` 基准给出的槽位顶部可能是分数设备像素，采样时无法落在半像素上。
     * 实测 9px / bold 9px（白键与 C 音名）**完全一致**，8px（黑键音名）与 10px
     * （刻度标签）各差 0.5 设备像素，肉眼不可见。这是图集方案的固有量化误差，
     * 不通过"按字号查表偏移"去修——那会把像素级的栅格巧合固化成规则。
     *
     * 特殊说明 1：视口原点恒为 (0, 0)——键盘与刻度的几何、文字都是轴列视口坐标
     * （`valueToY` 直接给出视口 y，x 从轴列左缘起算），与网格层同一约定。
     *
     * 特殊说明 2：两个 program 交替使用时必须各自重新设置绘制状态（混合 / VAO），
     * 因为 program 的 `render` 内部会绑定自己的 VAO。此处按"几何 → 文字"固定顺序，
     * 不交错，避免状态互相污染。
     */
    function drawGlKeyboard(): void {
        if (glAxisHandle === null) return;
        const target = glAxisHandle.resize(axisWidthPx, viewportHeightPx, readDevicePixelRatio());
        glAxisHandle.clear();

        // ① 几何层（键体 / 刻度线）
        if (glAxisProgram !== null && glAxisUploadedCount > 0) {
            if (glAxisGeometryUploaded) {
                glAxisProgram.repaint(target, 0, 0);
            } else {
                glAxisProgram.render(glAxisInstances, glAxisUploadedCount, target, 0, 0);
                glAxisGeometryUploaded = true;
            }
        }

        // ② 文字层
        if (glGlyphProgram !== null && glGlyphs !== null && glTextQuads.length > 0) {
            // 按页增量上传图集：只传本帧有变更的页，避免每帧重传整张图集。
            // 特殊说明：`uploadAtlas` 只接受单页数据（program 内部持有单张纹理），
            // 因此多页图集时以**最后一页**为准。本面板的字符集（约 150 个字形）
            // 远小于单页容量，实际只会有第 0 页。
            const dirty = glGlyphs.consumeDirtyPages();
            for (const page of dirty) {
                const data = glGlyphs.readPage(page);
                if (data !== null) {
                    glGlyphProgram.uploadAtlas(data, glGlyphs.atlasPageSizePx());
                    uploadedAtlasPages.add(page);
                }
            }
            if (uploadedAtlasPages.size > 0) {
                glGlyphProgram.render(glTextQuads, target, 0, 0);
            }
        }
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
    /** 上一次逐帧上报给面板的竖向位置（NaN = 从未上报）。 */
    let lastFrameScrollTop = Number.NaN;

    /**
     * 通知面板：水平位置由**用户手势**改变（绘制坐标）。
     *
     * 特殊说明：只在拖 thumb / 点轨道翻页这类宿主亲自解析的手势后调用，绝不在
     * 镜像回写或面板命令式写入后调用——详见 `onUserScrollLeft` 的说明。
     */
    function notifyUserScrollLeft(): void {
        onUserScrollLeft?.(scroll.get().scrollLeft - horizontalOffsetPx());
    }

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

    /**
     * 帧提交：面板绘制 → GL 场景层 → 滚动条几何 → DOM 同步 → 量化提交。
     *
     * 特殊说明 1（GL 与 Canvas2D 的画序无关观感）：两个层是**兄弟画布**，靠 DOM
     * 层级叠放（GL 在下、Canvas2D 在上），因此先画谁不影响最终像素。
     *
     * 特殊说明 2（**为什么面板绘制必须排在曲线 GL 之前**）：曲线层的数据是面板在
     * `onFrame` → `applyScrollLayers` → `drawRef.current()` 里**就地写入**
     * `data().curves` 的（见 `buildCurveLayers`）。若先执行 `drawGlCurves()`，
     * 它读到的永远是**上一帧**的图层列表——首帧更是空数组，于是「刚打开时曲线
     * 完全不显示，直到发生一次滚动才出现」。这个顺序是有数据依赖的，不能调换。
     */
    function draw(): void {
        if (disposed) return;
        const view = scroll.get();

        // 竖向逐帧上报（见 `onScrollTopFrame` 说明）：面板据此**正向**刷新值域
        // 视口 ref，从而不再依赖原生 `scroll` 事件的反向回写（那是镜像回声的来源）。
        //
        // 【必须在 onFrame 之前】面板的绘制（曲线投影 / 命中测试 / `valueToY`）读的
        // 就是这些 ref。放在 `onFrame` 之后会让画面慢一帧，竖向拖动时表现为
        // 「内容追着指针走但永远差一点」——与我们要修的手感问题同源。
        // 与横向同一约定：只做赋值，不进 React。
        if (onScrollTopFrame !== undefined && view.scrollTop !== lastFrameScrollTop) {
            lastFrameScrollTop = view.scrollTop;
            onScrollTopFrame(view.scrollTop);
        }

        // 面板帧提交：写 DOM / Canvas2D，并**填充本帧的曲线图层描述符**。
        // 必须在曲线 GL 之前（见上方特殊说明 2）。
        onFrame?.(currentAxis());

        // GL 几何重建：只在内容签名变化时做（滚动 / 播放不触发重建）。
        if (glProgram !== null) {
            const spec = data().grid;
            const signature = gridSignature(spec);
            if (signature !== lastGridSignature) {
                lastGridSignature = signature;
                rebuildGlGeometry(spec);
            }
            drawGlScene();
        }

        // 键盘 / 数值轴 GL 层：与网格共用同一份 spec，但几何与画布独立。
        if (glAxisProgram !== null) {
            const spec = data().grid;
            const signature = keyboardSignature(spec);
            if (signature !== lastKeyboardSignature) {
                lastKeyboardSignature = signature;
                // pitch 走键盘几何，其余走数值轴刻度线；两者互斥。
                if (spec !== null && spec !== undefined && spec.kind === "pitch") {
                    rebuildKeyboardGeometry(spec);
                } else {
                    rebuildAxisMarkGeometry(spec);
                }
                // 文字只随几何一起重建（位置由几何决定，几何没变则文字也没变）。
                glTextQuads =
                    glGlyphs !== null ? glGlyphs.build(collectAxisTextRequests(spec)) : [];
            }
            drawGlKeyboard();
        }

        // 曲线层（阶段 3）：在网格之上、播放头之下。每帧重建几何——滚动/缩放会
        // 改变每个点的视口位置，没有零重建路径（见 drawGlCurves 说明）。
        if (glCurveProgram !== null) {
            drawGlCurves();
        }

        // 动态叠加层（选区块 + 播放头）：每帧重建 + 绘制，但它清的是**空画布**，
        // 与曲线层无关——这正是播放帧不再重绘曲线的关键。
        if (glOverlayProgram !== null) {
            rebuildOverlayGeometry();
            drawGlOverlay();
        }

        updateScrollbars(view);
        syncDom(view);

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
            // 用户手势：拖 horizontal thumb 是真实意图，需通知面板同步共享视口。
            notifyUserScrollLeft();
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
                if (target !== null) {
                    scroll.setScrollLeft(target + horizontalOffsetPx());
                    // 用户手势：点轨道翻页同样需要同步共享视口。
                    notifyUserScrollLeft();
                }
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

    // ── 输入：竖向键盘滚动 ───────────────────────────────────────────
    //
    // 【为什么必须收进内核】旧实现里 PageUp / PageDown / Home / End 是靠**浏览器
    // 的原生滚动**生效的。内核模式下原生 scroller 只是被动镜像，其 `scroll` 事件
    // 会被当作回声忽略（原生 scroller 只是镜像）——原生那次变化就没人采纳了，四个
    // 键会**彻底失效**。时间轴内核早已把键盘滚动收进宿主（见
    // `timelineKernelHost.onKeyDown`），本面板此前漏了这一条。
    //
    // 【为什么只处理竖向】横向的四个键在既有实现里本来就不生效（旧实现实测：
    // PageDown/End/Home 均不改变 `scrollLeft`，只有竖向变化），迁移不得顺手改变
    // 行为。横向键盘滚动是独立议题，不在本次范围。
    //
    // 特殊说明：监听挂在**容器**上而不是 window。容器带 `tabIndex`，是键盘焦点
    // 的宿主；挂 window 会把按键从其它面板抢过来。
    function onKeyDown(event: KeyboardEvent): void {
        const target = resolveKeyboardScrollTarget({
            key: event.key,
            scrollTopPx: scroll.get().scrollTop,
            viewportHeightPx,
            maxScrollTopPx: scroll.maxScrollTop(),
        });
        if (target === null) return;
        // 只有确实归本面板处理时才阻止默认：否则会连带吞掉浏览器的其它行为。
        event.preventDefault();
        scroll.setScrollTop(target);
    }
    registerListener(container, "keydown", onKeyDown as EventListener);

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

        getGlStatus() {
            return {
                active: glProgram !== null,
                failureReason: glFailureReason,
                keyboardInstanceCount: glAxisUploadedCount,
                gridSpec: data().grid ?? null,
                gridInstanceCount: glUploadedCount,
            };
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
            // GL 资源最后释放：program 持有 VAO / buffer / shader，漏掉会随
            // 反复挂载（StrictMode 双挂载 / HMR）累积并最终耗尽上下文。
            glProgram?.dispose();
            glProgram = null;
            glHandle = null;
            glAxisProgram?.dispose();
            glAxisProgram = null;
            glGlyphProgram?.dispose();
            glGlyphProgram = null;
            glGlyphs?.dispose();
            glGlyphs = null;
            glAxisHandle = null;
            glOverlayProgram?.dispose();
            glOverlayProgram = null;
            glOverlayHandle = null;
            glCurveProgram?.dispose();
            glCurveProgram = null;
        },
    };
}
