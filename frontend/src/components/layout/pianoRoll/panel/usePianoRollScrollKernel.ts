/**
 * 参数编辑器的**滚动提交与内核宿主**：各图层的同帧落地 + 原生位置的采纳 + 宿主生命周期。
 *
 * 【为什么是独立的一个 hook】这一组原本散落在面板模型的 1438-1906 行，被视口状态
 * （`usePianoRollViewport`）与绘制装配夹在中间。但它们其实是**同一台机器的两端**：
 * 一端是「把一次位置变化同时落到标尺 / 网格 / 画布 / 波形」(`applyScrollLayers`)，
 * 另一端是「创建那个每帧驱动这些落地的内核宿主」。中间那些 `applyEdgeScrollLeft` /
 * `syncScrollLeft` 全都是这两端之间的接线——任何一处改动都要同时照顾另外几处。
 *
 * 【所有权不变】宿主引用、滚动位置、镜像 ref 仍然属于**面板**（面板重挂载即归零，
 * 与拆分前一致），这里只是把声明与装配搬进了子 hook：调用点、hook 顺序、依赖数组
 * 全部保持原位，因此渲染行为与拆分前逐字节相同。
 *
 * 【坐标约定】**绘制坐标**是参数编辑器画布里的 x 对应的位置，**原生坐标**是原生
 * scroller 的 `scrollLeft`；两者相差一个 `timelineOffset`。跨面板的写入必须先换算回
 * 原生坐标（见 `usePianoRollViewport` 文件头）。
 *
 * 【本文件不负责什么】`paintNow` 与 `commitViewportNow` 留在面板模型里：前者被
 * `setPitchView` / `setParamViewport` 在**本 hook 调用之前**的闭包里引用，后者以
 * 函数声明形式提前交给 `usePianoRollViewport`（见模型里的说明）。它们都只读 ref，
 * 因此留在原地不需要任何额外接线。
 */
import { useCallback, useEffect } from "react";
import type { RootState } from "../../../../app/store";
import { readDevicePixelRatio } from "../../../../utils/devicePixelLine";
import {
    timelineViewportNativeToState,
    timelineViewportStateToNative,
    timelineViewportSync,
} from "../../../../utils/timelineViewportSync";
import {
    createTimelineAxis,
    playheadLineLeftPx,
    rulerLayerTranslatePx,
} from "../../renderKernel/timelineAxis.js";
import { invokeGridRedrawHandler } from "../../timeline/gridRedrawBridge";
import { AXIS_W } from "../constants";
import {
    createPianoRollKernelHost,
    type PianoRollKernelHost,
} from "../kernel/host/pianoRollKernelHost";
import type { MutablePianoRollKernelData } from "../kernel/host/pianoRollKernelData";
import { resolvePanelRenderViewport } from "../kernel/viewportSource";
import { pianoRollViewportBus } from "../pianoRollViewportBus";
import type { ParamName, ValueViewport } from "../types";
import { centerFromVerticalScrollTop } from "../verticalScrollMapping";
import {
    PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX,
    PIANO_ROLL_SYNC_ORIGIN,
    shouldWriteNumber,
} from "./panelHelpers";

/** 本 hook 读取的会话切片（只声明真正用到的字段）。 */
type ScrollSession = Pick<RootState["session"], "paramEditorSyncTimeline">;

type PianoRollScrollKernelInputs = {
    /** 会话切片（同上）。 */
    s: ScrollSession;
    /** 内核宿主的稳定引用；宿主尚未挂载时为 null。 */
    hostRef: { current: PianoRollKernelHost | null };
    /** 原生滚动容器。 */
    scrollerRef: { current: HTMLDivElement | null };
    /** 绘制坐标的当前滚动位置（渲染期 ref 镜像）。 */
    scrollLeftRef: { current: number };
    /** 上一次已采纳的绘制坐标；`null` 表示尚未采纳过。 */
    lastScrollLeftRef: { current: number | null };
    /** 上一次镜像写入原生 scroller 的偏移（判定滚动回声的基准）。 */
    lastMirroredScrollLeftRef: { current: number };
    lastMirroredScrollTopRef: { current: number };
    /** 当前缩放（px/s）。 */
    pxPerSecRef: { current: number };
    /** 可视区尺寸。 */
    viewSizeRef: { current: { w: number; h: number } };
    /** 与时间轴同步开关的 ref 镜像。 */
    paramEditorSyncTimelineRef: { current: boolean };
    /** 绘制坐标 ↔ 原生坐标的偏移。 */
    timelineOffsetRef: { current: number };
    /** 正在应用共享视口（避免回推）。 */
    timelineSyncApplyingRef: { current: boolean };
    /** 标尺刻度范围的量化提交（rAF 合并）。 */
    scrollStateRafRef: { current: number | null };
    setScrollLeft: (next: number) => void;
    /** 视觉插值播放头（无宿主时兜底写标尺播放头线）。 */
    visualPlayheadSecRef: { current: number };
    rulerPlayheadLineRef: { current: HTMLDivElement | null };
    rulerPlayheadHeadRef: { current: HTMLDivElement | null };
    captureBandsRef: { current: HTMLDivElement | null };
    rulerContentRef: { current: HTMLDivElement | null };
    gridLayerRef: { current: HTMLDivElement | null };
    /** 面板自己的帧绘制入口（Canvas2D 细节层）。 */
    drawRef: { current: () => void };
    /** 内核数据镜像（宿主每帧现读）。 */
    kernelDataRef: { current: MutablePianoRollKernelData };
    /** 当前编辑的参数（宿主的竖向回调按它换算视口）。 */
    editParamRef: { current: ParamName };
    pitchViewRef: { current: ValueViewport };
    paramViewsRef: { current: Record<string, ValueViewport> };
    hScrollbarThumbRef: { current: HTMLDivElement | null };
    vScrollbarThumbRef: { current: HTMLDivElement | null };
    hScrollbarTrackRef: { current: HTMLDivElement | null };
    vScrollbarTrackRef: { current: HTMLDivElement | null };
    glCanvasRef: { current: HTMLCanvasElement | null };
    glAxisCanvasRef: { current: HTMLCanvasElement | null };
    glOverlayCanvasRef: { current: HTMLCanvasElement | null };
    getCurrentViewportForScrollbar: (param: ParamName) => ValueViewport;
    getParamValueBoundsForScrollbar: (param: ParamName) => { min: number; max: number };
};

/**
 * 滚动提交 + 内核宿主。
 *
 * 调用点必须固定在面板模型里原来的位置：`applyScrollLayers` / `syncScrollLeft` 的
 * 声明位置就是它们的**使用顺序**（`useVisualPlayhead` 的 onFrame 在模型里早于本
 * hook 的调用点，靠闭包延迟读取；与 `commitViewportNow` 提前交出去是同一模式）。
 */
export function usePianoRollScrollKernel({
    s,
    hostRef,
    scrollerRef,
    scrollLeftRef,
    lastScrollLeftRef,
    lastMirroredScrollLeftRef,
    lastMirroredScrollTopRef,
    pxPerSecRef,
    viewSizeRef,
    paramEditorSyncTimelineRef,
    timelineOffsetRef,
    timelineSyncApplyingRef,
    scrollStateRafRef,
    setScrollLeft,
    visualPlayheadSecRef,
    rulerPlayheadLineRef,
    rulerPlayheadHeadRef,
    captureBandsRef,
    rulerContentRef,
    gridLayerRef,
    drawRef,
    kernelDataRef,
    editParamRef,
    pitchViewRef,
    paramViewsRef,
    hScrollbarThumbRef,
    vScrollbarThumbRef,
    hScrollbarTrackRef,
    vScrollbarTrackRef,
    glCanvasRef,
    glAxisCanvasRef,
    glOverlayCanvasRef,
    getCurrentViewportForScrollbar,
    getParamValueBoundsForScrollbar,
}: PianoRollScrollKernelInputs) {
    // 以 useCallback 稳定引用（依赖仅含 ref 与模块级工具）：供 syncScrollLeft 等
    // 需要长期持有本函数的路径使用，避免随渲染重建引发上游回调引用抖动。
    const applyScrollLayers = useCallback(function applyScrollLayers(next: number) {
        if (rulerContentRef.current) {
            // 平移量吸附到设备像素（见 `rulerLayerTranslatePx`）：与内核同一约定，
            // 否则层内标尺竖线在系统缩放率 > 1 时粗细不一。
            rulerContentRef.current.style.transform = `translateX(${-rulerLayerTranslatePx(next, readDevicePixelRatio())}px)`;
        }

        if (gridLayerRef.current) {
            invokeGridRedrawHandler(gridLayerRef.current, next);
        }

        // 同步绘制：滚动事件在绘制前触发，画布必须与标尺/网格（上方已同步
        // 落地）在同一帧内提交，否则滚动中会出现画布与网格的分层漂移。
        drawRef.current();
        // 波形面走同一条同步链：不能在 React state（rAF）提交后再画。
        //
        // 缩放一并取内核真值（与 `drawRef` 同一理由，见 `resolvePanelRenderViewport`）：
        // 渲染期 ref 可能提前于内核变化，波形与其余图层就会差一个缩放。
        const emitPxPerSec = resolvePanelRenderViewport({
            kernelView: hostRef.current?.getViewport() ?? null,
            refPxPerSec: pxPerSecRef.current,
            refScrollLeftPx: scrollLeftRef.current,
        }).pxPerSec;
        pianoRollViewportBus.emit(next, emitPxPerSec, viewSizeRef.current.w);
        // 标尺播放头线：**宿主模式下由内核写**（同一次帧提交、同一份内核视口，
        // 与 GL 主体播放头逐设备像素一致）。这里只在无宿主时兜底 —— 两个写者写同一
        // 个 `style.left` 是"最后一次写入者获胜"，一旦口径不同就会分离。
        if (hostRef.current === null) {
            const playheadLeftPx = playheadLineLeftPx(
                createTimelineAxis({
                    pxPerSec: emitPxPerSec,
                    scrollLeftPx: next,
                    viewportWidthPx: viewSizeRef.current.w,
                    dpr: readDevicePixelRatio(),
                }),
                visualPlayheadSecRef.current,
            );
            if (rulerPlayheadLineRef.current) {
                rulerPlayheadLineRef.current.style.left = `${playheadLeftPx}px`;
            }
            if (rulerPlayheadHeadRef.current) {
                rulerPlayheadHeadRef.current.style.left = `${playheadLeftPx}px`;
            }
        }
    }, []);

    /**
     * 把「绘制坐标」的水平位置提交到当前滚动载体。
     *
     * 【为什么要单独一个函数】面板里有若干**权威写入**点（时间轴同步、键盘缩放、
     * 缩放事务落地、关闭同步时还原位置）：它们算出目标位置后直接落到滚动载体
     * （宿主，内部换算原生坐标并镜像回写）。把换算收在一处，避免每调用点各写一份。
     *
     * 特殊说明：滚轮 / 自动滚屏等**增量**路径不走这里——它们仍写原生 scroller，
     * 再由 `syncScrollLeft` 采纳进内核（见该函数注释）。宿主尚未创建（挂载期）时
     * 退回直接写原生 scroller，与调用方原本的语义一致。
     *
     * @param drawingScrollLeft 目标水平位置（绘制坐标）。
     */
    /**
     * 把绘制坐标的水平滚动落到 DOM（无宿主时的回退路径）。
     *
     * 【为什么必须是 `useCallback`】它原先是一个函数**声明** —— 组件每次渲染都会
     * 产生一个新的函数对象，于是把 `applyEdgeScrollLeft` 的依赖数组每帧都改掉，
     * `react-hooks/exhaustive-deps` 报的正是这一点（而 `applyEdgeScrollLeft` 又
     * 挂在拖拽的 rAF 循环上，每帧重建回调会连带重建整条调用链）。
     * 函数体只读 ref，依赖为空数组即可，身份因此恒定。
     */
    const applyHorizontalScrollPosition = useCallback(function applyHorizontalScrollPosition(
        drawingScrollLeft: number,
    ): void {
        const host = hostRef.current;
        if (host) {
            host.setScrollLeft(drawingScrollLeft);
            return;
        }
        const scroller = scrollerRef.current;
        if (!scroller) return;
        const offset = paramEditorSyncTimelineRef.current ? timelineOffsetRef.current : 0;
        const native = timelineViewportStateToNative(drawingScrollLeft, offset);
        scroller.scrollLeft = native;
        lastMirroredScrollLeftRef.current = native;
    }, []);

    /**
     * **拖拽边缘自动滚屏**的落点：写内核 + 推共享视口，**不自己重绘**。
     *
     * 【为什么不复用 `syncScrollLeft`】那条路径为"原生滚动事件驱动"设计，必须同任务
     * 重绘（`paintNow`），否则画面会比原生滚动慢一帧。而自动滚屏发生在**笔画自己的
     * rAF 合帧循环**里：那一帧本来就要绘制一次，再同步重绘一次等于**每帧两次全帧
     * 提交**（重建 GL 场景 + 重绘所有图层）—— 即用户报告的"滚动时非常卡顿"。
     *
     * 这里只写内核（钳制 + 标脏）并广播，绘制交给那一帧本就要跑的绘制路径。
     * 帧内合并且幂等，因此"标脏之后一定有人画"由现有帧循环保证。
     *
     * @param drawingScrollLeft 目标水平位置（绘制坐标）。
     */
    const applyEdgeScrollLeft = useCallback(
        function applyEdgeScrollLeft(drawingScrollLeft: number) {
            scrollLeftRef.current = drawingScrollLeft;
            // `syncScrollLeft` 早退的判据：留着旧值时，镜像回写触发的 `scroll`
            // 事件会被误判成"用户滚动"而再推一次共享视口。
            lastScrollLeftRef.current = drawingScrollLeft;

            const host = hostRef.current;
            if (host) {
                host.setScrollLeft(drawingScrollLeft);
                host.invalidate(); // ★ 不 paintNow —— 见上方说明
            } else {
                applyHorizontalScrollPosition(drawingScrollLeft);
                applyScrollLayers(drawingScrollLeft);
            }

            if (paramEditorSyncTimelineRef.current && !timelineSyncApplyingRef.current) {
                timelineViewportSync.setViewport(
                    {
                        scrollLeft: timelineViewportStateToNative(
                            drawingScrollLeft,
                            timelineOffsetRef.current,
                        ),
                        pxPerSec: pxPerSecRef.current,
                    },
                    PIANO_ROLL_SYNC_ORIGIN,
                );
            }
        },
        [applyHorizontalScrollPosition, applyScrollLayers],
    );

    /**
     * 采纳一次水平滚动位置变化（原生 scroller → 真值）。
     *
     * 流程：原生坐标 → 绘制坐标 → 同步开关时推送共享视口 → 通知各图层 → 量化提交 state。
     *
     * 【本函数的语义（重要）】内核是唯一渲染路径，它是**采纳点**：尚未迁移的输入
     * 代码（滚轮、中键平移、自动滚屏）仍然直接写 `scroller.scrollLeft`，这里把该值
     * 收进内核（`host.setScrollLeft`），由内核完成钳制、镜像回写与帧提交。这样：
     * - 所有既有手势**逐条保持可用**，无需在本任务里重写输入路径（Task 7 才迁移）；
     * - 钳制与渲染真值收敛到内核一处，不会出现两套边界；
     * - 收敛性：内核回写的值与内核当前值相同 → 不产生状态变化 → 不再触发帧，
     *   因此「写原生 → 采纳 → 镜像回写 → 再触发 onScroll」不会形成循环。
     *
     * 特殊说明：原生 scroller 的 `scroll` 事件是**纯镜像回声**，不再经本函数
     * （见 `onScrollerScroll`）；本函数只由显式写入原生的输入路径调用。
     *
     * @param scroller 原生滚动容器。
     */
    // 以 useCallback 稳定引用（依赖仅含同步开关设置态与 ref）：下游
    // `onScrollerScroll` 等回调的依赖数组直接持有它，引用抖动会连带放大。
    const syncScrollLeft = useCallback(
        function syncScrollLeft(scroller: HTMLDivElement) {
            const syncEnabled = s.paramEditorSyncTimeline;
            const offset = syncEnabled ? timelineOffsetRef.current : 0;
            const next = timelineViewportNativeToState(scroller.scrollLeft, offset);
            if (lastScrollLeftRef.current != null && lastScrollLeftRef.current === next) {
                return;
            }
            lastScrollLeftRef.current = next;
            scrollLeftRef.current = next;
            if (syncEnabled && !timelineSyncApplyingRef.current) {
                // 【必须用 next（采纳值）而不是 scroller.scrollLeft（原生 DOM）】
                //
                // 原生 scroller 只是**镜像**：真值在 ScrollKernel，由宿主在帧
                // 提交时回写。直接读 DOM 会拿到**尚未回写**的旧值，于是把旧位置当成
                // "用户滚动"推回共享视口——时间轴收到后应用旧值，位置出现回退。
                //
                // 实测（拖时间轴带动参数编辑器时）：共享视口序列 `10 → 20 → pianoRoll
                // 推回 10`，时间轴内核随之从 20 退回 10。连续拖拽时每三帧回退一次
                // （增量呈 `+30, +10, -10` 循环），即用户报告的"阶梯感/被吸附感"。
                //
                // `next` 是刚由原生位置换算出的绘制坐标，再换算回原生即得权威值；
                // 与 `onUserScrollLeft` 的口径一致（后者用的是内核的绘制坐标）。
                timelineViewportSync.setViewport(
                    {
                        scrollLeft: timelineViewportStateToNative(next, offset),
                        pxPerSec: pxPerSecRef.current,
                    },
                    PIANO_ROLL_SYNC_ORIGIN,
                );
            }
            // 交给内核（它会按新边界钳制、镜像回写并提交各图层）。
            //
            // 【为什么在内核写入后立刻 `paintNow`】本函数由**原生滚动事件**驱动（触摸拖拽、
            // 触控板惯性、中键平移、框选自动滚屏）。原生滚动发生在浏览器的渲染步骤里，而
            // 滚轮/拖拽任务里排队的 rAF 要等**下一帧**才跑——这会让可见内容（标尺、网格、
            // 主画布、波形、曲线、播放头）比原生滚动慢一帧。旧实现是在滚动事件里同步重绘，
            // 因此没有这一帧差；这里用 `paintNow()`（同任务提交）恢复到同一时序。
            const host = hostRef.current;
            if (host) {
                host.setScrollLeft(next);
                host.paintNow();
                return;
            }
            applyScrollLayers(next);
            if (scrollStateRafRef.current == null) {
                scrollStateRafRef.current = requestAnimationFrame(() => {
                    scrollStateRafRef.current = null;
                    setScrollLeft(scrollLeftRef.current);
                });
            }
        },
        [applyScrollLayers, s.paramEditorSyncTimeline, setScrollLeft],
    );

    // ── 内核宿主：创建 / 销毁 ────────────────────────────────────────
    //
    // 【为什么只在挂载时创建】宿主持有滚动位置与手势状态，是长生命周期运行时
    // 对象；随渲染重建会让滚动位置静默归零（与 `ScrollKernel` 的生命周期约束
    // 一致）。数据经 `kernelDataRef` 流入，尺寸经宿主自己的 ResizeObserver 更新。
    //
    // 【两套坐标系（本任务最容易出错的地方）】
    // 「同步时间轴视图」开启时，参数编辑器的内容层要整体右移一个偏移量（`offset`，
    // 实测 200px），两边的网格线才能在屏幕上对齐。于是：
    // - **原生坐标**：`[0, 内容宽]`——与轨道视图**同一格式、同一范围**（同步模式下
    //   它就是两个面板共享的那个位置，见 `horizontalOffsetPx` 的「关键不变量」）；
    // - **绘制坐标**：`[−offset, 内容宽 − offset]`——含负值，各图层（标尺/网格/画布/
    //   波形）用。偏移只是投影：它把内容整体右移，不扩大可滚范围。
    //
    // 内核的位置字段恒被钳到 `[0, max]`（无法表示负值），所以内核持有**原生坐标**，
    // 本面板消费的 `axis.scrollLeftPx` 是**绘制坐标**。换算只在宿主边界发生
    // （见宿主 `horizontalOffsetPx`），面板这一侧不再自行换算，避免出现第三份口径。
    useEffect(() => {
        const container = scrollerRef.current;
        if (!container) return;
        const hThumb = hScrollbarThumbRef.current;
        const vThumb = vScrollbarThumbRef.current;
        if (!hThumb || !vThumb) return;

        const host = createPianoRollKernelHost({
            container,
            hScrollbarThumb: hThumb,
            vScrollbarThumb: vThumb,
            hScrollbarTrack: hScrollbarTrackRef.current ?? undefined,
            vScrollbarTrack: vScrollbarTrackRef.current ?? undefined,
            data: () => kernelDataRef.current,
            initialPxPerSec: pxPerSecRef.current,
            glCanvas: glCanvasRef.current,
            glAxisCanvas: glAxisCanvasRef.current,
            glOverlayCanvas: glOverlayCanvasRef.current,
            axisWidthPx: AXIS_W,
            // GL 场景层恒开启：它是唯一渲染路径（见下方 `skip*` 的「已知并接受的
            // 限制」说明与设计文档 §2.4）。
            glSceneEnabled: true,
            // 偏移经 ref 读取（同步开关与布局偏移都在运行时变化，闭包捕获会读到挂载时的旧值）。
            horizontalOffsetPx: () =>
                paramEditorSyncTimelineRef.current ? timelineOffsetRef.current : 0,
            // 全部传 **getter**（见 `PianoRollKernelDomSync` 的说明）：这些元素可能
            // 在宿主创建之后才挂载（标尺随视图出现、停靠重排后 DOM 重建）。按值捕获
            // 会让宿主永久持有 null 或已脱离文档的节点，对应写入从此静默失效。
            sync: {
                rulerContent: () => rulerContentRef.current,
                gridLayer: () => gridLayerRef.current,
                // 标尺播放头线与三角由内核在同一次帧提交里写（与 GL 播放头同源同帧），
                // 避免"面板按自己的缩放写、GL 按内核缩放画"造成的水平分离。
                rulerPlayheadLine: () => rulerPlayheadLineRef.current,
                rulerPlayheadHead: () => rulerPlayheadHeadRef.current,
                // 捕获带：与标尺同源同帧（见 host 里 captureBands 的说明）。
                captureBands: () => captureBandsRef.current,
            },
            // 帧提交：宿主已完成滚动条几何与标尺 / 网格的 DOM 写入，这里只做
            // 「画布 + 波形 + 播放头」三项的提交。复用 `applyScrollLayers`，
            // 保证各图层与宿主同帧刷新。
            onFrame: (axis) => {
                const drawing = axis.scrollLeftPx;
                scrollLeftRef.current = drawing;
                applyScrollLayers(drawing);
                // 【为什么必须同步 lastScrollLeftRef】镜像回写会让原生 scroller 触发
                // `scroll` 事件。把"上一次已知位置"同步成内核真值后，即便有代码经
                // `syncScrollLeft` 采纳，`next` 也与之相等 → 立即早退，**不会**把值
                // 推回共享视口（否则镜像回写会被误当成用户滚动，把时间轴也推着走）。
                lastScrollLeftRef.current = drawing;
                // 镜像回写：让尚未迁移的输入代码（中键拖拽 / 框选自动滚动）读到的
                // 原生位置始终等于内核真值。仅在真正不一致时写，避免每帧触发样式重算。
                const native = timelineViewportStateToNative(
                    drawing,
                    paramEditorSyncTimelineRef.current ? timelineOffsetRef.current : 0,
                );
                if (shouldWriteNumber(container.scrollLeft, native, 0.5)) {
                    container.scrollLeft = native;
                    // 记下"我写下的值"：它是判定原生 `scroll` 事件是否为回声的唯一基准
                    // （见 `isMirrorEcho` 与镜像 ref 的说明）。
                    lastMirroredScrollLeftRef.current = native;
                }
                if (shouldWriteNumber(container.scrollTop, axis.scrollTopPx, 0.5)) {
                    container.scrollTop = axis.scrollTopPx;
                    lastMirroredScrollTopRef.current = axis.scrollTopPx;
                }
            },
            onScrollTopFrame: (scrollTopPx) => {
                // 竖向逐帧上报：把内核真值**正向**写回面板的值域视口 ref。
                //
                // 【为什么必须有这条通道】`pitchViewRef` / `paramViewsRef` 是竖向滚动
                // 的使用者（曲线投影、命中测试、`valueToY` 都读它们），但竖向真值在
                // 内核且此前**没有回写通道**——面板只能靠原生 `scroll` 事件反向回写
                // 来"猜"，而那条路正是镜像回声的来源（实测每帧把内核回退 ~7.5px，
                // 用户报告为"上下拖经常拖不动 / 有吸附感"）。
                //
                // 改用正向通道后回声不再承担任何职责，可以在 `onScrollerScroll` 里
                // 安全忽略。宿主保证本回调排在 `onFrame`（绘制）之前，因此画面与数据
                // 同帧对齐、不会慢一拍。
                //
                // 只做赋值，不进 React（与横向 `onScrollLeftFrame` 同一约定）。
                const param = editParamRef.current;
                const current = getCurrentViewportForScrollbar(param);
                const bounds = getParamValueBoundsForScrollbar(param);
                const center = centerFromVerticalScrollTop({
                    min: bounds.min,
                    max: bounds.max,
                    span: current.span,
                    scrollTop: scrollTopPx,
                    scrollRangePx: PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX,
                });
                if (param === "pitch") {
                    pitchViewRef.current = { center, span: current.span };
                } else {
                    paramViewsRef.current = {
                        ...paramViewsRef.current,
                        [param]: { center, span: current.span },
                    };
                }
            },
            onScrollLeftFrame: (drawingScrollLeft) => {
                // 逐帧把真值推给共享视口（时间轴据此跟随）。
                //
                // 【为什么必须与 onScrollLeftCommit 分开做】后者按 256px 量化，只为把
                // 标尺刻度范围提交给 React；用它的精度推同步会让时间轴"先不动、然后
                // 跳一大段"。这条只做赋值 + 广播，不进 React —— 与时间轴的
                // `syncScrollLeftFrame` 同一约定。
                //
                // 【为什么 lastScrollLeftRef 也要同步】它是 `syncScrollLeft` 早退的
                // 判据；留着旧值时，镜像回写触发的 `scroll` 事件会被误判成"用户滚动"
                // 而再推一次共享视口（`onFrame` 里已有同一处理）。
                scrollLeftRef.current = drawingScrollLeft;
                lastScrollLeftRef.current = drawingScrollLeft;
                if (paramEditorSyncTimelineRef.current && !timelineSyncApplyingRef.current) {
                    timelineViewportSync.setViewport(
                        {
                            scrollLeft: timelineViewportStateToNative(
                                drawingScrollLeft,
                                timelineOffsetRef.current,
                            ),
                            // 与位置**同源**：取内核真值。用渲染期的 `pxPerSecRef` 会配出
                            // "新缩放 + 旧位置"这一对自相矛盾的视口（时间轴侧的
                            // `livePxPerSec` 就是为此存在）。
                            pxPerSec:
                                hostRef.current?.getViewport().pxPerSec ?? pxPerSecRef.current,
                        },
                        PIANO_ROLL_SYNC_ORIGIN,
                    );
                }
            },
            onScrollLeftCommit: () => {
                // 量化提交：标尺的刻度范围由 React 按视口计算，不同步就会出现
                // 「滚动后刻度消失」。注意 `px` 是**绘制坐标**（宿主对外统一口径）。
                if (scrollStateRafRef.current == null) {
                    scrollStateRafRef.current = requestAnimationFrame(() => {
                        scrollStateRafRef.current = null;
                        setScrollLeft(scrollLeftRef.current);
                    });
                }
            },
            onUserScrollLeft: (drawingScrollLeft) => {
                // 用户手势（拖 thumb / 点轨道翻页）：需要把新位置推给共享视口，
                // 否则同步模式下滚动时间轴不会跟随。
                //
                // 【为什么不走 syncScrollLeft】那条路径依赖 `scroll` 事件，而内核的
                // 镜像是同帧程序化写入的，事件到达时无法再区分"用户滚动"与"自身回写"。
                // 由宿主主动上报手势来源是唯一可靠判据（见 `onUserScrollLeft` 说明）。
                scrollLeftRef.current = drawingScrollLeft;
                lastScrollLeftRef.current = drawingScrollLeft;
                const offset = paramEditorSyncTimelineRef.current ? timelineOffsetRef.current : 0;
                if (paramEditorSyncTimelineRef.current && !timelineSyncApplyingRef.current) {
                    timelineViewportSync.setViewport(
                        {
                            scrollLeft: timelineViewportStateToNative(drawingScrollLeft, offset),
                            pxPerSec: pxPerSecRef.current,
                        },
                        PIANO_ROLL_SYNC_ORIGIN,
                    );
                }
            },
        });
        hostRef.current = host;
        // dev-only 调试出口：浏览器里读取内核视口真值（滚动 / 值域中心 / 滚动条
        // 几何），用于核对「滚动条 thumb 与实际可滚范围是否一致」这类问题。
        if (import.meta.env.DEV) {
            (
                window as unknown as { __hsPianoRollKernel?: PianoRollKernelHost }
            ).__hsPianoRollKernel = host;
        }
        return () => {
            hostRef.current = null;
            if (import.meta.env.DEV) {
                delete (window as unknown as { __hsPianoRollKernel?: PianoRollKernelHost })
                    .__hsPianoRollKernel;
            }
            host.dispose();
        };
        // 挂载时创建一次；数据经 kernelDataRef 流入（见上方注释）。
        //
        // 依赖项为空是刻意的：宿主一旦重建，滚动位置与手势状态会静默归零（见
        // `ScrollKernel` 的生命周期约束）。回调里的 `editParamRef` 与
        // `getCurrentViewportForScrollbar` / `getParamValueBoundsForScrollbar`
        // 都只读 ref（不读渲染期 state），因此闭包捕获的旧引用仍能取到最新值。
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    return {
        applyScrollLayers,
        applyHorizontalScrollPosition,
        applyEdgeScrollLeft,
        syncScrollLeft,
    };
}
