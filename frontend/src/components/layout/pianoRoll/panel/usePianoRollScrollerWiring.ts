/**
 * 参数编辑器的**滚动容器与轴列接线**：把原生 DOM 事件接到模型上。
 *
 * 【为什么是独立的一个 hook】这一段原本夹在悬停浮窗重算与编辑命令分发之间（模型的
 * 模型后段），内容是同一件事的各个面：原生 `scroll` 事件的采纳（其中绝大部分是
 * 内核镜像回写造成的**回声**）、自绘滚动条的滚轮/翻页/拖拽接线、标尺滚轮、轴列的两个
 * 交互 hook，以及工具模式切换时光标与右键菜单的复位。
 *
 * 【所有权不变】所有状态仍属于面板：`scrollerRef` / 各滚动条 ref 来自
 * `usePianoRollViewport`，`setCanvasCursor` / `setCtxMenu` / `setAxisValuePreview`
 * 是面板自己的 state setter。本 hook 只是把语句搬进子 hook，因此 hook 顺序、
 * effect 注册顺序与依赖数组与拆分前逐条相同 —— 调用点必须固定在模型原处。
 *
 * 【坐标约定】原生 `scroll` 事件报的是**原生坐标**，内核与各图层用**绘制坐标**；
 * 换算在 `syncScrollLeft` 与宿主的 `horizontalOffsetPx` 边界完成（见
 * `usePianoRollViewport` 文件头）。
 *
 * 【对 lint 的已知影响】ref 与 state setter 从"本函数里的 `useRef` / `useState` 结果"
 * 变成"另一个 hook 经参数传进来的值"之后，`react-hooks/exhaustive-deps` 无法再证明
 * 它们的引用稳定性，于是这些 effect 会各多一条 missing-dependency 警告。那是**误报**
 * —— 它们的身份与拆分前完全相同（同一次渲染内恒等）。为了不让真实的依赖问题淹没在
 * 噪声里，这几处各带一条写明理由的豁免注释（与 `usePianoRollViewport` 的说明同源）。
 */
import {
    useCallback,
    useEffect,
    useLayoutEffect,
    useRef,
    type CSSProperties,
    type Dispatch,
    type SetStateAction,
} from "react";
import type * as React from "react";
import type { Keybinding } from "../../../../features/keybindings/types";
import { computeAutoFollowScrollLeft } from "../../../../utils/autoFollowScroll";
import { useNonPassiveWheel } from "../../../../utils/useNonPassiveWheel";
import { isMirrorEcho } from "../../timeline/scrollEcho";
import type { PianoRollKernelHost } from "../kernel/host/pianoRollKernelHost";
import type { ProcessorParamDescriptor } from "../../../../types/api";
import type { ValueViewport } from "../types";
import { usePianoRollAxisInteractions } from "./usePianoRollAxisInteractions";
import { usePianoRollAxisWheel } from "./usePianoRollAxisWheel";
import type { usePianoRollInteractions } from "../usePianoRollInteractions";
import type { ModelDeps } from "./modelDeps";

type PianoRollScrollerWiringInputs = {
    /** 会话切片。 */
    s: ModelDeps["s"];
    dispatch: ModelDeps["dispatch"];
    editParam: ModelDeps["editParam"];
    /** 渲染期缩放与内容宽（滚动上限与自动跟随都要用）。 */
    pxPerSec: number;
    contentWidth: number;
    /** 原生滚动容器与各层元素。 */
    scrollerRef: { current: HTMLDivElement | null };
    axisWrapRef: { current: HTMLDivElement | null };
    hScrollbarTrackRef: { current: HTMLDivElement | null };
    vScrollbarTrackRef: { current: HTMLDivElement | null };
    viewSizeRef: { current: { w: number; h: number } };
    /** 内核宿主；未挂载时为 null（滚轮/滚轮平移据此分流）。 */
    hostRef: { current: PianoRollKernelHost | null };
    /** 镜像回写基准与同步应用标志。 */
    lastMirroredScrollLeftRef: { current: number };
    lastMirroredScrollTopRef: { current: number };
    timelineSyncApplyingRef: { current: boolean };
    /** 渲染期刷新的采纳函数引用（供只注册一次的监听器调用）。 */
    syncScrollLeftRef: { current: (scroller: HTMLDivElement) => void };
    /** 视觉播放头（自动跟随的锚点）。 */
    visualPlayheadSecRef: { current: number };
    /** 值域视口与处理器参数。 */
    pitchViewRef: ModelDeps["pitchViewRef"];
    paramViewsRef: ModelDeps["paramViewsRef"];
    processorParamsRef: ModelDeps["processorParamsRef"];
    processorParams: ProcessorParamDescriptor[];
    /** 标脏入口。 */
    invalidate: ModelDeps["invalidate"];
    /** 视口换算（竖向滚动条 / 轴列滚轮共用）。 */
    clampViewport: ModelDeps["clampViewport"];
    getCurrentViewportForScrollbar: ModelDeps["getCurrentViewportForScrollbar"];
    yToValue: (param: string, y: number, h: number) => number;
    setPitchView: (next: ValueViewport) => void;
    setParamViewport: (param: string, next: ValueViewport) => void;
    syncVerticalScrollbarForViewport: (param: string, view: ValueViewport) => void;
    /** 采纳一次原生滚动位置（定义在 `usePianoRollScrollKernel` 里）。 */
    syncScrollLeft: (scroller: HTMLDivElement) => void;
    /** 轴列浮窗的数值格式化。 */
    formatParamValuePreview: (value: number) => string;
    /** 交互回调集合（滚轮入口来自这里）。 */
    interactions: ReturnType<typeof usePianoRollInteractions>;
    /** 滚轮/缩放/平移键位绑定。 */
    horizontalZoomKb: Keybinding;
    scrollHorizontalKb: Keybinding;
    pianoKeysVerticalScrollKb: Keybinding;
    pianoKeysVerticalZoomKb: Keybinding;
    /** 工具模式切换时的光标与右键菜单。 */
    setCanvasCursor: Dispatch<SetStateAction<CSSProperties["cursor"]>>;
    setCtxMenu: Dispatch<SetStateAction<{ x: number; y: number } | null>>;
    /** 轴列浮窗。 */
    setAxisValuePreview: Dispatch<
        SetStateAction<{ clientX: number; clientY: number; text: string } | null>
    >;
};

/**
 * 滚动容器与轴列接线。
 *
 * 调用点必须固定在模型里原来的位置：`usePianoRollAxisWheel` 与
 * `usePianoRollAxisInteractions` 在这里被调用，它们的状态槽与 effect 顺序取决于
 * 本 hook 的调用位置；`scrollerWheelHandlerRef` 的 layout effect 也在这里注册。
 */
export function usePianoRollScrollerWiring({
    s,
    dispatch,
    editParam,
    pxPerSec,
    contentWidth,
    scrollerRef,
    axisWrapRef,
    hScrollbarTrackRef,
    vScrollbarTrackRef,
    viewSizeRef,
    hostRef,
    lastMirroredScrollLeftRef,
    lastMirroredScrollTopRef,
    timelineSyncApplyingRef,
    syncScrollLeftRef,
    visualPlayheadSecRef,
    pitchViewRef,
    paramViewsRef,
    processorParamsRef,
    processorParams,
    invalidate,
    clampViewport,
    getCurrentViewportForScrollbar,
    yToValue,
    setPitchView,
    setParamViewport,
    syncVerticalScrollbarForViewport,
    syncScrollLeft,
    formatParamValuePreview,
    interactions,
    horizontalZoomKb,
    scrollHorizontalKb,
    pianoKeysVerticalScrollKb,
    pianoKeysVerticalZoomKb,
    setCanvasCursor,
    setCtxMenu,
    setAxisValuePreview,
}: PianoRollScrollerWiringInputs) {
    const onScrollerWheelNative = interactions.onScrollerWheelNative;
    /**
     * 原生滚动容器的 `scroll` 事件（面板侧的唯一入口）。
     *
     * 【绝大部分是镜像回声，但不是全部】
     * 内核是唯一渲染路径，原生 scroller 只是**镜像**：`onFrame` 每帧把内核真值写回
     * 它，该写入会触发原生 `scroll` 事件。旧代码在这里**无条件**读回位置再写回内核，
     * 而那一刻读到的是上一帧镜像的旧值（实测内核已到 548.054、事件里读到 540.5），
     * 使内核被回退 7.554px，逐帧往复即用户报告的「上下拖经常拖不动 / 阶梯感」。
     *
     * 但"无条件忽略"同样不对：原生容器仍是 `overflow: scroll`，**触摸拖拽 /
     * 触控板惯性 / 焦点滚入视口**这三类输入只会产生原生 `scroll`，没有任何显式
     * 入口——忽略它等于这些输入完全失效（旧实现在这里读回位置正是为了它们）。
     *
     * 【判据：与「上次镜像写入值」比，**不是**与「内核当前值」比】
     * 本处理函数一度拿事件值与 `host.getViewport()`（内核**当前**值）比较。那在连续
     * 滚动 / 缩放时必然失效：事件报的是**上一帧**写下的值，内核此刻已前进一整帧，
     * 两者必然不等 → 回声被稳定地误收成"容器自己动了" → 采纳并（同步开启时）把它
     * 推回共享视口。推回的还带着浏览器的设备像素量化误差（≤0.5px），于是两个面板
     * 进入亚像素往复——即用户报告的「启用同步后滚轮缩放仍然抽动」（同步关闭时不会
     * 推回共享视口，所以看不到）。
     *
     * 基准必须是**我上次写进容器的值**（`lastMirroredScroll*Ref`）：回声报的正是它
     * （误差仅来自浏览器量化，有界），而真实输入会把容器带到别的值上。这正是
     * `timeline/scrollEcho` 为轨道头写下的同一条结论（判来源，不判与真值的距离），
     * 现在两个容器共用同一个纯函数。
     *
     * @param event 原生 `scroll` 事件。
     * @returns 无返回值。
     */
    const onScrollerScroll = useCallback(
        (event: React.UIEvent<HTMLDivElement>) => {
            // 应用共享视口写入期间不采纳：那一刻原生镜像还停在上一帧的旧值
            //（与 `timelineSyncApplyingRef` 的既有用法同一理由）。
            if (timelineSyncApplyingRef.current) return;
            const scroller = event.currentTarget;
            const host = hostRef.current;
            if (host === null) return;
            if (
                !isMirrorEcho({
                    mirroredPx: lastMirroredScrollLeftRef.current,
                    nativePx: scroller.scrollLeft,
                })
            ) {
                syncScrollLeft(scroller);
            }
            if (
                !isMirrorEcho({
                    mirroredPx: lastMirroredScrollTopRef.current,
                    nativePx: scroller.scrollTop,
                })
            ) {
                host.setScrollTop(scroller.scrollTop);
                // 原生滚动事件驱动的竖向滚动（触摸 / 触控板 / 焦点滚入）：
                // 与横向同因——内核写完后必须**同任务**提交各图层，否则竖向内容
                // （音高键 / 数值轴 / 曲线 / 播放头）比原生滚动慢一帧（见 `paintNow`）。
                host.paintNow();
            }
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps -- hostRef / 两个镜像 ref / timelineSyncApplyingRef 都是面板的 useRef 结果（按构造稳定），本 hook 只是经参数拿到它们
        [syncScrollLeft],
    );
    const scrollerWheelHandlerRef = useRef(onScrollerWheelNative);

    useLayoutEffect(() => {
        scrollerWheelHandlerRef.current = onScrollerWheelNative;
    });

    useEffect(() => {
        const el = scrollerRef.current;
        if (!el) return;

        const handler: EventListener = (evt) => {
            scrollerWheelHandlerRef.current(evt as globalThis.WheelEvent);
        };

        el.addEventListener("wheel", handler, {
            passive: false,
        } as globalThis.AddEventListenerOptions);
        return () => {
            el.removeEventListener("wheel", handler);
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps -- scrollerRef 是面板的 useRef 结果（按构造稳定）
    }, []); // 空依赖

    // ── 自绘滚动条 / 标尺上的滚轮 ──────────────────────────────────────
    // 自绘滚动条是滚动容器的**兄弟节点**（见下方轨道注释），标尺则在滚动容器
    // **之上**，两者都不在 `scroller` 的事件路径里，原生 `wheel` 不会冒泡到它。
    // 于是它们各自把「轴」显式托付给同一个入口：命中判定不同，语义同源
    //（无修饰键 = 该轴滚动，`modifier.scrollbarZoom` = 该轴缩放），不存在第二套口径。
    //
    // 用回调 ref（而非 `useEffect(..., [])` + `ref.current`）挂监听：轨道与标尺
    // 会随停靠重排重建 DOM，回调 ref 把「元素出现」本身当作挂载时机。
    const attachVerticalScrollbarWheel = useNonPassiveWheel<HTMLDivElement>((event) => {
        scrollerWheelHandlerRef.current(event, "vertical");
    });
    const attachHorizontalScrollbarWheel = useNonPassiveWheel<HTMLDivElement>((event) => {
        scrollerWheelHandlerRef.current(event, "horizontal");
    });
    const attachVerticalScrollbarTrack = useCallback(
        (element: HTMLDivElement | null) => {
            vScrollbarTrackRef.current = element;
            attachVerticalScrollbarWheel(element);
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps -- vScrollbarTrackRef 是面板的 useRef 结果（按构造稳定）
        [attachVerticalScrollbarWheel],
    );
    const attachHorizontalScrollbarTrack = useCallback(
        (element: HTMLDivElement | null) => {
            hScrollbarTrackRef.current = element;
            attachHorizontalScrollbarWheel(element);
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps -- hScrollbarTrackRef 是面板的 useRef 结果（按构造稳定）
        [attachHorizontalScrollbarWheel],
    );
    /**
     * 标尺上的滚轮：**与画布内的滚轮完全同义**。
     *
     * 标尺在滚动容器**之上**，不在 `scroller` 的事件路径里；这里只做"转交"，不指定
     * 滚动条轴 —— 交给画布滚轮总入口后，命中判定由 `nativeScrollbarZoneAt` 按指针
     * 位置自行得出（标尺在容器矩形之外 ⇒ 返回 null），keybinding 判定、锚点换算、
     * 上下限全部与画布同一套，标尺不定义第二种语义。
     */
    const handleRulerWheel = useCallback((event: WheelEvent) => {
        scrollerWheelHandlerRef.current(event);
    }, []);

    // 参数切换或参数描述符变化后，刷新竖向滚动条位置，保证滚动条与当前视口保持一致。
    useLayoutEffect(() => {
        syncVerticalScrollbarForViewport(editParam, getCurrentViewportForScrollbar(editParam));
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [editParam, processorParams]);

    // Auto-scroll: keep playhead visible in parameter editor during playback
    useEffect(() => {
        if (s.paramEditorSyncTimeline) return;
        if (!s.autoScrollEnabled || !s.runtime.isPlaying) return;
        const scroller = scrollerRef.current;
        if (!scroller) return;
        const next = computeAutoFollowScrollLeft({
            playheadSec: visualPlayheadSecRef.current,
            pxPerSec,
            viewportWidth: scroller.clientWidth,
            contentWidth,
        });
        if (Math.abs(scroller.scrollLeft - next) > 0.5) {
            scroller.scrollLeft = next;
            syncScrollLeft(scroller);
        }
        // syncScrollLeft reads the latest scroll state through refs; see onFrame above.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [
        s.paramEditorSyncTimeline,
        s.autoScrollEnabled,
        s.runtime.isPlaying,
        s.playheadSec,
        pxPerSec,
        contentWidth,
    ]);

    // 左轴列的滚轮路由（与主画布同一套 keybinding / 投影 / 上下限）。
    usePianoRollAxisWheel({
        axisWrapRef,
        viewSizeRef,
        editParam,
        pitchViewRef,
        paramViewsRef,
        processorParamsRef,
        setPitchView,
        setParamViewport,
        invalidate,
        clampViewport,
        scrollerRef,
        syncScrollLeftRef,
        scrollHorizontalKb,
        pianoKeysVerticalScrollKb,
        pianoKeysVerticalZoomKb,
        horizontalZoomKb,
    });

    // 左轴列（琴键 / 数值刻度）的三个交互：试听、单位切换、悬浮读数。
    usePianoRollAxisInteractions({
        s,
        editParam,
        axisWrapRef,
        viewSizeRef,
        pitchViewRef,
        dispatch,
        yToValue,
        formatParamValuePreview,
        setAxisValuePreview,
    });

    // 选区在所有工具模式下保留并持续渲染（选区带 / 剪贴板预览在 render.ts
    // 不按工具门控）：绘制 / 直线颤音工具下不再自动清空，用户可以换轨后
    // 直接对保留的选区做复制/剪切（路由见 focusRouting.resolveCopyCutRoute，
    // 依据 session.paramSelectionActive / selectionContext）。与选区的交互
    // （拖拽 / 右键菜单 / 变形）仍只在选择工具下生效（各交互路径已有
    // toolMode 门控）。

    useEffect(() => {
        setCanvasCursor(s.toolMode === "select" ? "default" : "crosshair");
        // eslint-disable-next-line react-hooks/exhaustive-deps -- setCanvasCursor 是面板 useState 的 setter（按构造稳定）
    }, [s.toolMode]);

    useEffect(() => {
        setCtxMenu(null);
        // eslint-disable-next-line react-hooks/exhaustive-deps -- setCtxMenu 是面板 useState 的 setter（按构造稳定）
    }, [s.toolMode]);
    return {
        onScrollerScroll,
        attachVerticalScrollbarTrack,
        attachHorizontalScrollbarTrack,
        handleRulerWheel,
    };
}
