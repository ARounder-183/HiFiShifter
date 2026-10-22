import { seekPlayhead, setplayheadSec } from "../../../../features/session/sessionSlice";
import {
    isEraserButton,
    isStylusLike,
    shouldRejectConcurrentPointer,
} from "../../../../utils/penInput";
import { edgeScrollMaxLeftPx } from "../../shared/edgeAutoScroll";
import {
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
} from "../childPitchOffsetParams";
import { createEdgeScrollDriver } from "../kernel/edgeScrollDriver";
import { getDrawPreviewValue } from "../paramValuePreviewLogic";
import { timeToFrame } from "../utils";
import type { StrokeMode } from "../types";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { PointerDeps } from "./pointerDeps";
import { handleVibratoPointerDown } from "./vibratoPointerDown";
import { handleSelectPointerDown } from "./selectPointerDown";

/** 指针按下所需的上下文。 */
type CanvasPointerDownDeps = Pick<
    PointerDeps,
    | "activePointerGestureEndRef"
    | "applyDenseToLiveEdit"
    | "applyEdgeScrollLeft"
    | "applyMorphOverlayPreview"
    | "applyPostStrokeSmoothing"
    | "applyVibratoChoice"
    | "applyVibratoReset"
    | "axisFromRefs"
    | "blendDenseEdges"
    | "buildMorphDense"
    | "buildUploadSegments"
    | "buildVibratoDense"
    | "bumpRefreshToken"
    | "canvasRef"
    | "clampViewport"
    | "clearActivePointerGestureEnd"
    | "commitStroke"
    | "createFineAdjustedPointerState"
    | "currentParamRange"
    | "dispatch"
    | "disposeFineAdjustedPointerState"
    | "dragDirection"
    | "dynamicProjectSec"
    | "edgeHalfSpanForIndices"
    | "edgeSmoothnessPercent"
    | "editParam"
    | "ensureLiveEditBase"
    | "fetchCommitBaseSource"
    | "findStretchSelectionEdge"
    | "finishVibratoDrag"
    | "framePeriodMs"
    | "frameToViewportPx"
    | "getDefaultCanvasCursor"
    | "getFineAdjustedPointerPosition"
    | "getViewportTruth"
    | "installDragDirectionKeyCycler"
    | "invalidate"
    | "isEffectivePitchSnapActive"
    | "isPointerNearDraggableSelection"
    | "isSnapToggleModifierHeld"
    | "liveEditActiveRef"
    | "liveEditOverrideRef"
    | "morphDragRef"
    | "morphModifierDownRef"
    | "morphOverlayRef"
    | "onContextMenu"
    | "onCycleDragDirection"
    | "onParamCommitSucceeded"
    | "onParamValuePreviewChange"
    | "onPitchSnapGestureActiveChange"
    | "panRef"
    | "paramEditorSeekPlayheadEnabled"
    | "paramMultiSelectKb"
    | "paramStretchKb"
    | "paramValuePopupEnabled"
    | "paramView"
    | "paramViewRef"
    | "paramViewsRef"
    | "pitchSnapToleranceCents"
    | "pitchSnapUnit"
    | "pitchViewRef"
    | "pointerFrame"
    | "pointerSec"
    | "pointerValue"
    | "projectScale"
    | "reportVibratoDragState"
    | "requestWaveformRepaint"
    | "resetLiveEditPreview"
    | "rootTrackId"
    | "scaleAtSec"
    | "scrollLeftRef"
    | "scrollerRef"
    | "selectionRef"
    | "setActivePointerGestureEnd"
    | "setCanvasCursor"
    | "setMorphOverlay"
    | "setParamView"
    | "setParamViewport"
    | "setPitchView"
    | "setVibratoDragCaptureActive"
    | "snapDrawValue"
    | "straightVibratoPreset"
    | "strokeRef"
    | "syncScrollLeft"
    | "syncTimelineEnabled"
    | "timelineOffsetRef"
    | "toolMode"
    | "updateSelectionUi"
    | "valueToY"
    | "vibratoFrequencyAdjustKb"
    | "vibratoHudAnchorRef"
    | "vibratoPreset"
    | "vibratoPresetCycleList"
    | "vibratoStateRef"
    | "viewSizeRef"
>;

/**
 * 处理画布上的指针按下。
 *
 * @param deps 回调作用域里的上下文（见 `PointerDeps`）。
 * @param e 触发本次手势的指针事件。
 *
 * 【后续拆分的切点】`if (toolMode === "select")` 与 `if (usesVibratoPath)` 两个分支
 * 分别是两千多行与六百多行。前者需要先把"放弃整个回调"的 `return` 改写成显式的
 * "已处理"返回值；后者只需把 `onEdgeScrolled` 换成一个可变持有者（它被
 * `edgeScroll` 的闭包读、被分支写）。两件事都是改语义，必须单独验证。
 */
export function handleCanvasPointerDown(
    deps: CanvasPointerDownDeps,
    e: ReactPointerEvent<HTMLCanvasElement>,
): void {
    const {
        activePointerGestureEndRef,
        applyDenseToLiveEdit,
        applyEdgeScrollLeft,
        applyMorphOverlayPreview,
        applyPostStrokeSmoothing,
        applyVibratoChoice,
        applyVibratoReset,
        axisFromRefs,
        blendDenseEdges,
        buildMorphDense,
        buildUploadSegments,
        buildVibratoDense,
        bumpRefreshToken,
        canvasRef,
        clampViewport,
        clearActivePointerGestureEnd,
        commitStroke,
        createFineAdjustedPointerState,
        currentParamRange,
        dispatch,
        disposeFineAdjustedPointerState,
        dragDirection,
        dynamicProjectSec,
        edgeHalfSpanForIndices,
        edgeSmoothnessPercent,
        editParam,
        ensureLiveEditBase,
        fetchCommitBaseSource,
        findStretchSelectionEdge,
        finishVibratoDrag,
        framePeriodMs,
        frameToViewportPx,
        getDefaultCanvasCursor,
        getFineAdjustedPointerPosition,
        getViewportTruth,
        installDragDirectionKeyCycler,
        invalidate,
        isEffectivePitchSnapActive,
        isPointerNearDraggableSelection,
        isSnapToggleModifierHeld,
        liveEditActiveRef,
        liveEditOverrideRef,
        morphDragRef,
        morphModifierDownRef,
        morphOverlayRef,
        onContextMenu,
        onCycleDragDirection,
        onParamCommitSucceeded,
        onParamValuePreviewChange,
        onPitchSnapGestureActiveChange,
        panRef,
        paramEditorSeekPlayheadEnabled,
        paramMultiSelectKb,
        paramStretchKb,
        paramValuePopupEnabled,
        paramView,
        paramViewRef,
        paramViewsRef,
        pitchSnapToleranceCents,
        pitchSnapUnit,
        pitchViewRef,
        pointerFrame,
        pointerSec,
        pointerValue,
        projectScale,
        reportVibratoDragState,
        requestWaveformRepaint,
        resetLiveEditPreview,
        rootTrackId,
        scaleAtSec,
        scrollLeftRef,
        scrollerRef,
        selectionRef,
        setActivePointerGestureEnd,
        setCanvasCursor,
        setMorphOverlay,
        setParamView,
        setParamViewport,
        setPitchView,
        setVibratoDragCaptureActive,
        snapDrawValue,
        straightVibratoPreset,
        strokeRef,
        syncScrollLeft,
        syncTimelineEnabled,
        timelineOffsetRef,
        toolMode,
        updateSelectionUi,
        valueToY,
        vibratoFrequencyAdjustKb,
        vibratoHudAnchorRef,
        vibratoPreset,
        vibratoPresetCycleList,
        vibratoStateRef,
        viewSizeRef,
    } = deps;

    // 掌压拒识：笔 / 鼠标手势进行中时，触摸（书写时的手掌）按下不中止、
    // 不接管当前手势。pen / mouse 的第二指针维持原语义（切笔尖 / 橡皮
    // 是单手用户的正常路径）。
    if (shouldRejectConcurrentPointer(e.nativeEvent, Boolean(activePointerGestureEndRef.current))) {
        return;
    }
    // 注意：鼠标侧键（X1 / X2）切预设**不在这里**处理。
    //
    // 【为什么】对鼠标而言 `pointerdown` 只在"从没有任何键按下"变为"有键
    // 按下"时派发一次。左键已按住时再按侧键**不会再派发 `pointerdown`**，
    // 只会有 `pointermove`（带着更新后的 `buttons`）—— 所以这个处理函数对
    // 侧键根本不会被调用，挂在这里是死代码。这也正是"键盘有效、侧键无效"
    // 的由来：键盘走的是独立的 window keydown，不依赖按键事件模型。
    //
    // 侧键改在 line/vibrato 分支已安装的 `window` + 捕获阶段的 `mousedown`
    // 里处理：`mousedown` 对**每一次**按键按下都会派发，与"拖拽中右键切换
    // 拖动方向"共用同一个已验证可用的入口。
    if (activePointerGestureEndRef.current) {
        activePointerGestureEndRef.current();
    }

    // 数位笔不触发「按下即 seek」：笔尖接触（button 0）与鼠标左键同值，
    // 但画参数线的每一笔起笔都会被误当成播放头定位 —— pen 的编辑意图
    // 全部走下方各手势分支，seek 保留给鼠标。
    const stylusDown = isStylusLike(e.nativeEvent);
    if (e.button === 0 && !stylusDown && paramEditorSeekPlayheadEnabled !== false) {
        const sec = pointerSec(e.clientX);
        dispatch(setplayheadSec(sec));
        void dispatch(seekPlayhead(sec));
    }

    if (
        e.button === 0 &&
        !stylusDown &&
        paramValuePopupEnabled &&
        // 只有选择与自由绘制弹这个气泡。直线 / 颤音工具另有一套合并气泡
        // （参数值 + 预设 / 深度 / 速率，见 `reportVibratoDragState`），
        // 两者同时冒出来会互相遮挡。
        (toolMode === "select" || toolMode === "draw")
    ) {
        const rawPreviewValue = pointerValue(e.clientY);
        const downPreviewValue =
            toolMode === "draw"
                ? getDrawPreviewValue({
                      editParam,
                      rawValue: rawPreviewValue,
                      effectiveSnap: isEffectivePitchSnapActive(e.nativeEvent),
                      pitchSnapUnit,
                      projectScale: scaleAtSec?.(pointerSec(e.clientX)) ?? projectScale,
                      pitchSnapToleranceCents,
                  })
                : rawPreviewValue;
        onParamValuePreviewChange?.({
            clientX: e.clientX,
            clientY: e.clientY,
            value: downPreviewValue,
        });
    }

    if (!rootTrackId) return;

    // ── 拖拽边缘自动滚屏（手势级驱动）──────────────────────────────────
    //
    // 【为什么每个手势新建一个】驱动的时钟在构造时归零，于是"新手势第一步"
    // 必然按 1/60 秒起步；跨手势复用会让两次拖拽之间的空闲被算进"本帧时长"，
    // 新手势第一下就跳一大步（纯函数虽把时长夹到 100ms，那仍是上百像素）。
    // 每个手势一个实例，这条约束由构造方式保证，不依赖调用方记得复位。
    //
    // 【为什么用 getter 而不是先算好】边界、缩放、滚动位置、可滚上界都在
    // 拖拽期间变化（缩放可被滚轮改、上界随 paramView 窗口变），逐次求值才能
    // 用到当帧的真值。
    //
    // 【为什么 onScrolled 经一个可变槽转发】选择工具与绘制类工具对"滚动之后
    // 该做什么"的需求不同：选择工具按新的轴重算选区（它自己就会做，无需回调），
    // 绘制类必须按**同一个指针位置**重跑一次采样 —— 视图滚了、指针屏幕坐标
    // 没变，但它对应的帧号变了；不重算，笔画就会在滚动处跳变。槽在分支里赋值，
    // 避免驱动去理解"当前是哪个工具"。
    /**
     * 边缘自动滚屏的回调持有者（**可变对象**而不是 `let`）：滚屏驱动在构造时闭包读它，
     * 颤音分支在拖动中改写它。用持有者而不是直接传值，是为了让分支能原样搬进独立函数
     * —— 传值会复制一份，写入就到不了驱动的闭包。
     */
    const onEdgeScrolledRef: { current: ((clientX: number) => void) | null } = { current: null };

    /**
     * 滚动上界：工程末端与**已取数窗口末端**取小。
     *
     * 【为什么绘制类要额外受窗口约束】`usePianoRollData` 的取数窗口是
     * "可见区 ± 一个可见时长"，而笔画期间取数被 `liveEditDeferral` 推迟到
     * pointer-up（换窗口会让正在画的轨迹消失，见
     * docs/plans/2026-09-26-volume-dyn-drag-trail-fix.md）。滚出窗口后
     * 提交仍会写进后端，但**画面上什么都不会出现** —— 用户对着空白拖一条
     * 看不见的线。窗口自带一整个视口宽的余量，因此这个上限足够一次边缘
     * 滚屏走完一屏，用户感知不到限制。
     *
     * 选择工具不受此限：选区是纯视觉范围，不依赖已取数的曲线。
     */
    const maxScrollLeftFor = (limitToLoadedWindow: boolean): number => {
        const scroller = scrollerRef.current;
        if (!scroller) return 0;
        const axis = axisFromRefs();
        const nativeOffsetPx = syncTimelineEnabled ? timelineOffsetRef.current : 0;
        const projectEndFrame = Math.max(0, timeToFrame(dynamicProjectSec, framePeriodMs));
        const projectMax = edgeScrollMaxLeftPx({
            pxPerSec: axis.pxPerSec,
            framePeriodMs,
            maxFrame: projectEndFrame,
            viewportWidthPx: scroller.clientWidth,
            nativeOffsetPx,
        });
        if (!limitToLoadedWindow) return projectMax;

        const pv = paramViewRef.current;
        if (!pv || pv.edit.length === 0) return 0;
        const stride = Math.max(1, Math.floor(pv.stride));
        const windowEndFrame = pv.startFrame + (pv.edit.length - 1) * stride;
        const windowMax = edgeScrollMaxLeftPx({
            pxPerSec: axis.pxPerSec,
            framePeriodMs,
            maxFrame: windowEndFrame,
            viewportWidthPx: scroller.clientWidth,
            nativeOffsetPx,
        });
        return Math.min(projectMax, windowMax);
    };

    const edgeScroll = createEdgeScrollDriver({
        getBounds: () => {
            const scroller = scrollerRef.current;
            if (!scroller) return null;
            const rect = scroller.getBoundingClientRect();
            return { left: rect.left, right: rect.right };
        },
        // ★ 读**内核真值**（绘制坐标），不读原生 `scrollLeft`。
        //
        // 【为什么不能读原生】写入走 `applyEdgeScrollLeft` → `host.setScrollLeft`
        //（内核），而原生 scroller 只是**镜像**，要到下一次帧提交才被回写。
        // 若驱动读原生值，第二次 step 读到的仍是上一帧的旧位置，于是每一帧
        // 都从同一个旧位置重新加一个增量 —— 滚动会"原地踏步"或严重偏慢。
        // 这与面板其它地方"真值在内核"的约定一致（见 `getViewportTruth`）。
        getScrollLeft: () => getViewportTruth()?.scrollLeft ?? scrollLeftRef.current,
        setScrollLeft: (nextDrawing) => {
            // 走自动滚屏专用落点：写内核 + 推共享视口 + **标脏**，
            // 不自己重绘（绘制由笔画的合帧循环负责，避免每帧两次全帧提交）。
            applyEdgeScrollLeft(nextDrawing);
        },
        // 上界同为绘制坐标（`edgeScrollMaxLeftPx` 已含 nativeOffset 平移）。
        getMaxScrollLeft: () => maxScrollLeftFor(toolMode !== "select"),
        // ★ 下界必须也是绘制坐标且可为负：开启同步时绘制域是
        // `[−偏移, 内容宽 − 偏移]`，左侧那片预留的对齐留白是**合法的负区间**。
        // 传 0 会让"向左滚"每帧都被夹回 0，与内核自己的下界（−偏移）打架 ——
        // 视图在两处往复即用户看到的"闪现"。
        getMinScrollLeft: () => (syncTimelineEnabled ? -timelineOffsetRef.current : 0),
        onScrolled: (clientX) => onEdgeScrolledRef.current?.(clientX),
    });

    // Middle mouse: pan (time axis)
    if (e.button === 1) {
        e.preventDefault();
        setCanvasCursor("grabbing");
        const scroller = scrollerRef.current;
        if (!scroller) return;
        const pid = e.pointerId;
        panRef.current = {
            pointerId: pid,
            startClientX: e.clientX,
            startClientY: e.clientY,
            startScrollLeft: scroller.scrollLeft,
            startView:
                editParam === "pitch"
                    ? pitchViewRef.current
                    : (paramViewsRef.current[editParam] ?? {
                          center: 0.5,
                          span: 1,
                      }),
            startRectH:
                (canvasRef.current?.getBoundingClientRect().height ?? viewSizeRef.current.h) || 1,
        };
        (e.currentTarget as HTMLCanvasElement).setPointerCapture(pid);
        const onMove = (ev: globalThis.PointerEvent) => {
            // 掌侧误触防御（与同文件其余手势一致）：第二指针的 move 若
            // 先过 buttons 检查，touch 的 buttons 恒不满足 &4，会把
            // 进行中的平移整个终止掉；先校验 pointerId 直接忽略。
            if (ev.pointerId !== pid) return;
            if ((ev.buttons & 4) !== 4) {
                onUp();
                return;
            }
            const pan = panRef.current;
            if (!pan) return;
            const dx = ev.clientX - pan.startClientX;
            const dy = ev.clientY - pan.startClientY;
            scroller.scrollLeft = Math.max(0, pan.startScrollLeft - dx);
            syncScrollLeft(scroller);

            const hPx = Math.max(1, pan.startRectH);
            const deltaCenter = (dy / hPx) * pan.startView.span;
            if (editParam === "pitch") {
                setPitchView(
                    clampViewport("pitch", {
                        span: pan.startView.span,
                        center: pan.startView.center + deltaCenter,
                    }),
                );
            } else {
                setParamViewport(
                    editParam,
                    clampViewport(editParam, {
                        span: pan.startView.span,
                        center: pan.startView.center + deltaCenter,
                    }),
                );
            }
            invalidate();
        };
        const onUp = () => {
            panRef.current = null;
            setCanvasCursor(getDefaultCanvasCursor());
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            window.removeEventListener("pointercancel", onUp);
            clearActivePointerGestureEnd(onUp);
        };
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
        window.addEventListener("pointercancel", onUp);
        setActivePointerGestureEnd(onUp);
        return;
    }

    if (toolMode === "select") {
        if (
            handleSelectPointerDown(
                {
                    applyDenseToLiveEdit,
                    applyMorphOverlayPreview,
                    axisFromRefs,
                    blendDenseEdges,
                    buildMorphDense,
                    buildUploadSegments,
                    bumpRefreshToken,
                    canvasRef,
                    clearActivePointerGestureEnd,
                    createFineAdjustedPointerState,
                    disposeFineAdjustedPointerState,
                    dragDirection,
                    dynamicProjectSec,
                    edgeHalfSpanForIndices,
                    edgeScroll,
                    edgeSmoothnessPercent,
                    editParam,
                    ensureLiveEditBase,
                    fetchCommitBaseSource,
                    findStretchSelectionEdge,
                    framePeriodMs,
                    frameToViewportPx,
                    getFineAdjustedPointerPosition,
                    installDragDirectionKeyCycler,
                    invalidate,
                    isEffectivePitchSnapActive,
                    isPointerNearDraggableSelection,
                    liveEditActiveRef,
                    liveEditOverrideRef,
                    morphDragRef,
                    morphModifierDownRef,
                    morphOverlayRef,
                    onContextMenu,
                    onCycleDragDirection,
                    onParamCommitSucceeded,
                    onParamValuePreviewChange,
                    onPitchSnapGestureActiveChange,
                    paramMultiSelectKb,
                    paramStretchKb,
                    paramValuePopupEnabled,
                    paramViewRef,
                    pitchSnapUnit,
                    pointerSec,
                    pointerValue,
                    projectScale,
                    requestWaveformRepaint,
                    resetLiveEditPreview,
                    rootTrackId,
                    scaleAtSec,
                    scrollerRef,
                    selectionRef,
                    setActivePointerGestureEnd,
                    setCanvasCursor,
                    setMorphOverlay,
                    setParamView,
                    updateSelectionUi,
                    valueToY,
                    viewSizeRef,
                    pointerFrame,
                },
                e,
            )
        ) {
            return;
        }
    }

    // 笔尖（button 0）= 画线；鼠标右键 / pen 笔杆键（button 2）与
    // pen 橡皮端（button 5）= 恢复（擦除回原始曲线）。橡皮端此前
    // 不在白名单里，落笔会被静默丢弃。
    const penEraserDown = isEraserButton(e.button, e.nativeEvent.pointerType);
    const secondaryDown = e.button === 2 || penEraserDown;
    const mode: StrokeMode = secondaryDown ? "restore" : "draw";
    if (e.button !== 0 && !secondaryDown) return;
    setCanvasCursor(getDefaultCanvasCursor());
    if (
        editParam === "pitch" ||
        isChildPitchOffsetCentsParam(editParam) ||
        isChildPitchOffsetDegreesParam(editParam)
    ) {
        onPitchSnapGestureActiveChange?.(true);
    }
    const pv = paramViewRef.current;
    if (pv) ensureLiveEditBase(pv);
    const fp = paramView?.framePeriodMs ?? 5;
    // 落笔帧 = 指针所在的那一帧（`timeToFrame` 向下取整，与曲线取值同一口径）。
    const frame = timeToFrame(pointerSec(e.clientX), fp);
    const rawValue = pointerValue(e.clientY);
    const isDrawMode = mode === "draw";
    const snapToggleHeld = isSnapToggleModifierHeld(e.nativeEvent);
    const value = isDrawMode ? snapDrawValue(rawValue, snapToggleHeld, frame) : rawValue;

    // 直线工具就是"颤音工具 + 直线预设"，两者走**同一条**拖拽与调参路径
    // —— 差别只有起手预设。把它们合成一个谓词，是为了让这条等价关系由
    // 结构保证，而不是靠两处分支各自记得同步。
    const isLineTool = toolMode === "line";
    const usesVibratoPath = isLineTool || toolMode === "vibrato";
    const dragPreset = isLineTool ? straightVibratoPreset : vibratoPreset;

    strokeRef.current = {
        mode,
        pointerId: e.pointerId,
        param: editParam,
        points: [{ frame, value }],
    };
    if (!usesVibratoPath) {
        vibratoStateRef.current = null;
        setVibratoDragCaptureActive(false);
    }
    // 标记 live 编辑开始，阻止 pitch_orig_updated 事件立即刷新曲线
    if (liveEditActiveRef) liveEditActiveRef.current = true;

    // For line tool, only show the start point initially
    const pv0 = paramViewRef.current;
    if (pv0) {
        applyDenseToLiveEdit(pv0, frame, mode === "restore" ? null : [value], frame, frame, mode);
    }
    (e.currentTarget as HTMLCanvasElement).setPointerCapture(e.pointerId);
    invalidate();
    // 与 move 路径同一约定：live 覆盖不进 React，波形需显式重绘。
    requestWaveformRepaint();

    if (usesVibratoPath) {
        handleVibratoPointerDown(
            {
                applyDenseToLiveEdit,
                applyPostStrokeSmoothing,
                applyVibratoChoice,
                applyVibratoReset,
                buildVibratoDense,
                clearActivePointerGestureEnd,
                commitStroke,
                createFineAdjustedPointerState,
                currentParamRange,
                disposeFineAdjustedPointerState,
                dragDirection,
                dragPreset,
                edgeScroll,
                editParam,
                finishVibratoDrag,
                fp,
                frame,
                getFineAdjustedPointerPosition,
                installDragDirectionKeyCycler,
                invalidate,
                isDrawMode,
                isLineTool,
                isSnapToggleModifierHeld,
                liveEditActiveRef,
                liveEditOverrideRef,
                mode,
                onCycleDragDirection,
                onEdgeScrolledRef,
                onPitchSnapGestureActiveChange,
                paramViewRef,
                penEraserDown,
                pointerSec,
                pointerValue,
                reportVibratoDragState,
                requestWaveformRepaint,
                resetLiveEditPreview,
                setActivePointerGestureEnd,
                setVibratoDragCaptureActive,
                snapDrawValue,
                snapToggleHeld,
                strokeRef,
                usesVibratoPath,
                value,
                vibratoFrequencyAdjustKb,
                vibratoHudAnchorRef,
                vibratoPresetCycleList,
                vibratoStateRef,
            },
            e,
        );
    }
}
