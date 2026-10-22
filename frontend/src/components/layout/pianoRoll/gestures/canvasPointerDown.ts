/**
 * 参数编辑器主画布的**指针按下**：整个"按下之后会发生什么"的分派。
 *
 * 【为什么单独成文件】它是 `onCanvasPointerDown` 的函数体 —— 三千多行，占了
 * `usePianoRollInteractions` 的一半以上。里面按工具与按键分成几条互斥通路：播放头
 * 定位、标尺拖动、中键平移、**选择工具的拖动 / 框选 / 拉伸 / 形变**、以及绘制与颤音
 * 起手。留在 hook 里时，读这个 hook 的任何一处都要先翻过它。
 *
 * 【为什么整段搬而不是按分支拆】选择工具那一段（两千多行）有十几处**提前 return**，
 * 语义是"放弃整个回调"。把它拆成子函数会让这些 return 变成"只返回子函数"，需要先
 * 改写成"已处理"信号 —— 那是改语义，不是搬移。因此先整段搬走，分支拆分留待单独
 * 一轮（见下面函数注释里的切点说明）。
 *
 * 【依赖怎么拿】上下文由调用方打包传入（`Pick<PointerDeps, …>`，字段清单由 AST
 * 扫描得出），函数体原样搬过来，开头解构回局部变量。**没有一行逻辑被改写。**
 */
import { isModifierActive } from "../../../../features/keybindings/keybindingsSlice";
import { seekPlayhead, setplayheadSec } from "../../../../features/session/sessionSlice";
import {
    cycleVibratoChoice,
    vibratoCycleAnchorId,
} from "../../../../features/vibrato/vibratoPresetList";
import { vibratoSeedForPreset } from "../../../../features/vibrato/vibratoSeed";
import { profileFor, scaledHitRadius } from "../../../../utils/inputProfile";
import { scaleStepDeltaBetween, transposePitchByScaleSteps } from "../../../../utils/musicalScales";
import {
    PEN_ERASER_BUTTONS_MASK,
    coalescedEventsOf,
    isEraserButton,
    isStylusLike,
    shouldRejectConcurrentPointer,
} from "../../../../utils/penInput";
import { armRightDragContextMenuGuard } from "../../../../utils/rightDragContextMenuGuard";
import { secToViewportPx } from "../../renderKernel/timelineAxis.js";
import { edgeScrollMaxLeftPx } from "../../shared/edgeAutoScroll";
import { clamp } from "../../timeline";
import {
    isChildFormantOffsetCentsParam,
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
} from "../childPitchOffsetParams";
import { createEdgeScrollDriver } from "../kernel/edgeScrollDriver";
import {
    CURVE_HIT_RADIUS_PX,
    SELECTION_EDGE_MIN_WIDTH_PX,
    curvePointAtPointer,
    isPointerNearCurve,
} from "../kernel/gestureHitTest";
import { isDynParam, shiftDynValueForDrag, shiftValueForDrag } from "../paramRanges";
import {
    addPointerRange,
    clampSelectionShift,
    frameBoundLeftFromPointer,
    frameBoundRightFromPointer,
    frameRangeEnd,
    frameRangeEndCut,
    frameRangeFromFrames,
    frameRangeStartCut,
    normalizeSelection,
    rangeIndexAtFrame,
    removeRangeAtFrame,
    resizeFrameRangeEdge,
    selectionFromPointers,
    selectionToFrameSpans,
    shiftSelectionRanges,
    snapFrame,
} from "../paramSelection";
import { editablePitchValue } from "../paramSmoothing";
import {
    formatRightDragMorphPercent,
    getDrawPreviewValue,
    getSelectDragPreviewValue,
} from "../paramValuePreviewLogic";
import { frameFromViewportClientX } from "../seekPlayheadMapping";
import {
    buildMultiRangeEditPlan,
    expandStrideSampledDense,
    fetchFullResCurve,
    readPvRange,
    restoreDynSentinelsFromParamView,
    uploadFullResCurve,
    uploadFullResCurveSegments,
} from "../selectionEditData";
import {
    createSelectionAmplifier,
    rightDragUpScale,
    transformSelectionByRightDrag,
} from "../selectionTransforms";
import { framesToTime, timeToFrame } from "../utils";
import {
    SIDE_BUTTON_BACK_MASK,
    SIDE_BUTTON_FORWARD_MASK,
    createDragWorking,
    resolveVibratoMiddleClickReset,
    resolveVibratoSideButton,
} from "../vibratoDragAdjust";
import {
    PARAM_DRAG_DEAD_ZONE_PX,
    makePvValueSource,
    recordGestureParamSelectionStep,
} from "./pointerSupport";
import type { FrameSpan } from "../paramSelection";
import type { ParamMorphOverlay, ParamViewSegment, StrokeMode } from "../types";
import type { PointerEvent as ReactPointerEvent } from "react";

import type { PointerDeps } from "./pointerDeps";

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
    let onEdgeScrolled: ((clientX: number) => void) | null = null;

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
        onScrolled: (clientX) => onEdgeScrolled?.(clientX),
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
        if (e.button !== 0 && e.button !== 2) return;

        const existingMorph = morphOverlayRef.current;
        const pvForMorph = paramViewRef.current;
        const canvas = canvasRef.current;
        if (existingMorph && existingMorph.length > 0 && pvForMorph && canvas) {
            const rect = canvas.getBoundingClientRect();
            const h = Math.max(1, rect.height || viewSizeRef.current.h || 1);
            const fp = Math.max(1e-6, pvForMorph.framePeriodMs);
            const stride = Math.max(1, pvForMorph.stride);
            // 多选区：控制线每段一条，命中取距离最近的一个控制点
            let hit: {
                overlayIndex: number;
                point: ParamMorphOverlay["points"][number];
            } | null = null;
            let hitDistance = Number.POSITIVE_INFINITY;
            for (let i = 0; i < existingMorph.length; i += 1) {
                for (const point of existingMorph[i].points) {
                    const sec = (point.frame * fp) / 1000;
                    const x = secToViewportPx(axisFromRefs(), sec);
                    const mapped = editParam === "pitch" ? point.value + 0.5 : point.value;
                    const y = valueToY(editParam, mapped, h);
                    const dx = Math.abs(e.clientX - rect.left - x);
                    const dy = Math.abs(e.clientY - rect.top - y);
                    if (dx > 8 || dy > 8) continue;
                    const distance = dx + dy;
                    if (distance < hitDistance) {
                        hitDistance = distance;
                        hit = { overlayIndex: i, point };
                    }
                }
            }

            if (hit && e.button === 0) {
                e.preventDefault();
                morphDragRef.current = {
                    pointerId: e.pointerId,
                    overlayIndex: hit.overlayIndex,
                    pointKind: hit.point.kind,
                };
                setCanvasCursor("grabbing");
                ensureLiveEditBase(pvForMorph);
                if (liveEditActiveRef) liveEditActiveRef.current = true;
                (e.currentTarget as HTMLCanvasElement).setPointerCapture(e.pointerId);
                const finePointerState = createFineAdjustedPointerState(
                    e.nativeEvent,
                    e.currentTarget as HTMLCanvasElement,
                );

                if (paramValuePopupEnabled) {
                    onParamValuePreviewChange?.({
                        clientX: e.clientX,
                        clientY: e.clientY,
                        value: hit.point.value,
                    });
                }

                // 预览重算合帧到下一渲染帧（与拉伸 / 选区拖拽的
                // schedulePreview 同一模式）：pointermove 在高刷鼠标 / pen
                // 上可达数百 Hz，而 setMorphOverlay（整面板重渲）与
                // applyMorphOverlayPreview（每条 overlay 一次 buildMorphDense）
                // 逐事件执行纯属浪费。ref 同步保持逐事件（下一次 move 与
                // onUp 都要读到最新控制点），只有 React 状态 / live 预览 /
                // 浮窗延后到 rAF。
                let previewRafId: number | null = null;
                let pendingOverlays: ParamMorphOverlay[] | null = null;
                let pendingPopup: {
                    clientX: number;
                    clientY: number;
                    value: number;
                } | null = null;
                const flushPreview = () => {
                    previewRafId = null;
                    if (pendingOverlays != null) {
                        const overlays = pendingOverlays;
                        pendingOverlays = null;
                        setMorphOverlay(overlays);
                        applyMorphOverlayPreview(overlays);
                    }
                    if (pendingPopup != null) {
                        onParamValuePreviewChange?.(pendingPopup);
                        pendingPopup = null;
                    }
                };

                const onMove = (ev: globalThis.PointerEvent) => {
                    if ((ev.buttons & 1) !== 1) {
                        onUp();
                        return;
                    }
                    // 只接受本指针的 move：第二指针（掌压触摸等）的
                    // buttons 恒为 1，不校验 pointerId 会驱动本次手势，
                    // 松手还会把掌压驱动的曲线提交到后端。
                    if (ev.pointerId !== e.pointerId) return;
                    const drag = morphDragRef.current;
                    const overlayNow = morphOverlayRef.current;
                    const pvNow = paramViewRef.current;
                    if (!drag || drag.pointerId !== e.pointerId || !overlayNow || !pvNow) {
                        return;
                    }
                    const target = overlayNow[drag.overlayIndex];
                    if (!target) return;
                    const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);

                    const nextPoints = target.points.map((pt) => {
                        if (pt.kind !== drag.pointKind) return pt;
                        const newValue = pointerValue(adjusted.clientY);
                        if (pt.kind === "left" || pt.kind === "right") {
                            return { ...pt, value: newValue };
                        }
                        const rawFrame = Math.max(0, snapFrame(pointerFrame(adjusted.clientX)));
                        const clampedFrame = clamp(
                            rawFrame,
                            target.selectionStartFrame,
                            target.selectionEndFrame,
                        );
                        return {
                            ...pt,
                            frame: clampedFrame,
                            value: newValue,
                        };
                    });

                    // 只替换被拖动的那条控制线（其余段的控制线保持不动）
                    const nextOverlays: ParamMorphOverlay[] = overlayNow.map((overlay, index) =>
                        index === drag.overlayIndex ? { ...overlay, points: nextPoints } : overlay,
                    );
                    morphOverlayRef.current = nextOverlays;
                    pendingOverlays = nextOverlays;

                    if (paramValuePopupEnabled) {
                        const movedPoint = nextPoints.find((pt) => pt.kind === drag.pointKind);
                        pendingPopup = {
                            clientX: ev.clientX,
                            clientY: ev.clientY,
                            value: movedPoint?.value ?? pointerValue(adjusted.clientY),
                        };
                    }

                    if (previewRafId == null) {
                        previewRafId = requestAnimationFrame(flushPreview);
                    }
                };

                const onUp = () => {
                    const drag = morphDragRef.current;
                    const overlayNow = morphOverlayRef.current;
                    const pvNow = paramViewRef.current;
                    morphDragRef.current = null;
                    window.removeEventListener("pointermove", onMove);
                    window.removeEventListener("pointerup", onUp);
                    window.removeEventListener("pointercancel", onUp);
                    disposeFineAdjustedPointerState(finePointerState);
                    clearActivePointerGestureEnd(onUp);
                    // 最后一次 move 的预览若还在 rAF 队列里，必须在提交前
                    // 同步冲刷：提交读 overlayNow / live 层，漏冲刷会把
                    // 末段位移丢掉（与选区拖拽"松手即取消"不同——这里
                    // 提交数据就来自挂起的预览状态）。
                    if (previewRafId != null) {
                        cancelAnimationFrame(previewRafId);
                        flushPreview();
                    }

                    if (!drag || !overlayNow || !pvNow || !rootTrackId) {
                        // 早退同样必须复位 liveEdit（否则曲线刷新被永久搁置，
                        // 与右键拖拽路径的处理一致）。
                        if (liveEditActiveRef) liveEditActiveRef.current = false;
                        setCanvasCursor("default");
                        return;
                    }

                    // 每段各自的控制线 → 逐段 dense；整次提交一个撤销点。
                    // 回写必须逐帧：buildMorphDense 产出的是 pv 步距采样
                    // （dense[k] ↔ startFrame + k×stride），把 stride 间隔
                    // 采样当连续帧写回会造成时间压缩 + 覆盖未选帧
                    // （stride=1 时原样返回，零开销）。
                    const packedPerOverlay = overlayNow.map((overlay) => {
                        const packed = buildMorphDense(overlay, stride);
                        const values = expandStrideSampledDense(packed.dense, stride);
                        return {
                            startFrame: packed.startFrame,
                            // dyn 哨兵保留（与拉伸 / 选区拖拽同一契约）：
                            // 未画帧写回哨兵，不物化成显式基线。
                            values: isDynParam(editParam)
                                ? restoreDynSentinelsFromParamView(
                                      values.slice(),
                                      packed.startFrame,
                                      pvNow,
                                  )
                                : values,
                        };
                    });

                    const nextEdit = pvNow.edit.slice();
                    const pvStepUp = Math.max(1, Math.floor(pvNow.stride));
                    for (const packed of packedPerOverlay) {
                        for (let i = 0; i < packed.values.length; i += 1) {
                            const globalIdx = Math.round(
                                (packed.startFrame + i - pvNow.startFrame) / pvStepUp,
                            );
                            if (globalIdx >= 0 && globalIdx < nextEdit.length) {
                                nextEdit[globalIdx] = packed.values[i];
                            }
                        }
                    }
                    setParamView({ ...pvNow, edit: nextEdit });

                    void (async () => {
                        try {
                            await uploadFullResCurveSegments({
                                trackId: rootTrackId,
                                param: editParam,
                                segments: packedPerOverlay,
                            });
                            // 提交成功：保留覆盖层让波形继续显示提交值，等
                            // 提交之后取的响度快照到达再撤下（消除松手闪屏）。
                            onParamCommitSucceeded();
                        } catch (err) {
                            console.error("[pianoRoll] morph upload failed", err);
                            // IPC 失败必须撤下覆盖层，否则冻结的预览值会一直
                            // 盖在真实曲线上。
                            liveEditOverrideRef.current = null;
                        } finally {
                            if (liveEditActiveRef) liveEditActiveRef.current = false;
                            bumpRefreshToken();
                        }
                    })();

                    if (morphModifierDownRef.current) {
                        // 保持调整点位置不变；baselineValues 是进入形变模式时一次性捕获的，
                        // 每次拖拽提交都在同一基准线上施加"总偏移"，不重置。
                        // 只有松开修饰键再重新按下时，才会调用 buildMorphOverlaysFromSelection 重置。
                    } else {
                        setMorphOverlay(null);
                    }
                    setCanvasCursor("default");
                };

                window.addEventListener("pointermove", onMove);
                window.addEventListener("pointerup", onUp);
                window.addEventListener("pointercancel", onUp);
                setActivePointerGestureEnd(onUp);
                return;
            }
        }

        const b = pointerFrame(e.clientX);
        const sel = selectionRef.current;

        // 选区构建的指针换算（普通框选、多选追加、边缘拖拽共用）。
        // 允许自动滚动时有 32px 边缘触发区。
        //
        // 【这里只产出**连续帧坐标**】"指针位置 → 选区边界"的两条规则（左 =
        // 最近的那一帧、右 = 所在帧之后）与"工程起点之前不可选"的钳制都在
        // `paramSelection` 里，由 `frameRangeFromPointers` /
        // `resizeFrameRangeEdge` 按边界选用；本函数只负责把指针换成帧坐标。
        //
        // 【只夹右端，**不夹左端**】右端夹到 `maxSelectableFrame - 1`：右边界是
        // "指针所在帧之后"（`floor(f) + 1`），指针落在最后一帧上时它正好等于工程
        // 末端的排他边界，再往右会把选区拖出工程。
        // 左端**照实保留负值**：同步到时间轴时左侧那段与轨道头等宽的额外空间是
        // 负时间，`frameRangeFromPointers` 要靠它判断"这次拖拽是否整段都在工程
        // 起点之前"（是则根本不产生选区）。若在这里夹成 0，那个判断就永远为假。
        const maxSelectableFrame = Math.max(0, timeToFrame(dynamicProjectSec, framePeriodMs));
        const clampSelectionFrame = (frame: number) =>
            Math.min(frame, Math.max(0, maxSelectableFrame - 1));
        const selectionFrameFromClientX = (clientX: number, allowAutoScroll: boolean) =>
            clampSelectionFrame(rawFrameFromClientX(clientX, allowAutoScroll));

        /**
         * 视口 x → **连续帧坐标**（不夹取）。自动滚动与投影都在这里。
         */
        function rawFrameFromClientX(clientX: number, allowAutoScroll: boolean): number {
            const scroller = scrollerRef.current;
            if (!scroller) {
                return pointerFrame(clientX);
            }

            // 本帧的交互投影：**与渲染侧同源**（内核真值，见 `axisFromRefs`）。
            const axis = axisFromRefs();
            const bounds = scroller.getBoundingClientRect();

            if (allowAutoScroll) {
                // 边缘自动滚动收口在 `edgeScroll` 驱动里（几何是
                // `shared/edgeAutoScroll` 的纯函数，驱动本身有单测）。
                // 这里只需"按当前指针位置滚一帧"：滚动量、可达上界、方向
                // 符号、帧时长归一都由它负责，本函数不再自己算一遍。
                //
                // 【为什么选择工具不启用持续滚动】它的语义是"指针到哪选到哪"，
                // 松手即定型 —— 停在边缘不动没有意义。持续滚动只服务绘制类
                // 工具（笔画必须跟着视图继续延伸）。
                edgeScroll.step(clientX);
            }

            const clampedClientX = clamp(clientX, bounds.left, bounds.right);
            // 视口 x → 帧：一律走统一投影的逆函数，**不再**用
            // `(scrollLeftRef + dx) / pxPerBeatRef` 这条独立算式。
            //
            // 【为什么这条算式是缺陷根因】它读的 `scrollLeftRef` 会被渲染期的
            // state→ref 同步覆盖回去，而 state 是 256px 步长的量化提交，滚动后
            // 最多滞后内核 255px——于是框选产生的选区整体偏移同一距离，而画面上
            // 选区块按内核绘制，两者对不上（用户报告"实际产生的选区与鼠标划定的
            // 区域不一致"，且先缩放后滚动才出现）。
            return frameFromViewportClientX({
                clientX: clampedClientX,
                viewportLeft: bounds.left,
                axis,
                framePeriodMs,
            });
        }

        // ── 选区上的手势优先级（高 → 低，唯一定义处）──────────────────
        //   0. Alt 形变控制点（上方已 return；控制点是画面上显式的小圆点，
        //      命中框 8px，比参数线还窄，且只在按住修饰键时出现 → 排最高）；
        //   1. **已选中的参数线**（落在某段内且靠近曲线）；
        //   2. Alt + 选区边缘 → 拉伸选区内的曲线（改数据、进撤销栈）；
        //   3. 多选区追加：⌘/Ctrl + 左键，**或右键落在未选中区域**
        //      （后者与前者等同；原地点击已有段仍是"取消该段"）；
        //   4. 无修饰键 + 边缘 → 只调整选区边界；
        //   5. 右键 + 已选中区域内部 → 左右平移**指针所在的那一段**（只动
        //      选区，不动数据；多段选区下其余段不动）；
        //   6. 普通框选（替换整个选区）。
        //
        // 【右键为什么分两种】右键落在**已选中区域**里 = 平移那一段（选区调整）；
        // 落在**未选中区域**里 = 追加一段（多选区操作）—— 平移对"空处"没有
        // 意义，而"再选一段"正是用户在那里起手时想做的事。
        //
        // 【为什么参数线永远第一】参数线的命中范围最窄（10px 纵向带 + 必须
        // 落在段内），而它在画面里可能只占很短一截，很难瞄准；选区边缘则是两条
        // 贯穿整个高度的明显竖线，用户想抓边缘时会自己去找那两条线。因此**所有
        // 选区自身的交互（边缘 / 多选 / 平移）都必须给参数线让路** —— 否则
        // "抓选区边界处那截曲线"会被选区交互永远吃掉。
        //
        // 下面每一处选区交互都显式让位（`!onSelectedCurve` / 各自的修饰键
        // 取反），因为这里的分支顺序本身就编码了优先级。
        const onSelectedCurve = isPointerNearDraggableSelection(
            e.clientX,
            e.clientY,
            e.nativeEvent.pointerType,
        );

        // ── 多选区追加：⌘/Ctrl + 左键，**或右键落在未选中区域** ─────────
        // 拖动 = 在已有选区上**追加**一段（并集；重叠/相接自动合并）；
        // 原地点击已有段 = 取消该段（与时间轴 ⌘+点击多选切换同源语义）。
        //
        // 【为什么右键也算多选】右键在**未选中区域**起手时，用户想做的通常
        // 就是"再选一段" —— 选区平移只对已选中区域有意义（见下一个分支），
        // 因此这里的右键与按住多选区修饰键**完全等同**。右键在已选中区域内
        // 起手仍是平移、压在参数线上仍是曲线交互，两者都不变。
        //
        // Alt 拉伸修饰键按下时本分支让位（见条件里的取反），否则在选区内侧
        // 边缘处无法拉伸；指针压住**已选中的参数线**时也让位，否则按住修饰键
        // 就再也拖不动选区内的曲线（而这正是该修饰键最不该挡住的交互）。
        const rightDragInUnselectedArea =
            e.button === 2 &&
            !isModifierActive(paramStretchKb, e.nativeEvent) &&
            rangeIndexAtFrame(sel, b) === -1;
        if (
            rightDragInUnselectedArea ||
            (e.button === 0 &&
                !onSelectedCurve &&
                !isModifierActive(paramStretchKb, e.nativeEvent) &&
                isModifierActive(paramMultiSelectKb, e.nativeEvent))
        ) {
            // 起手的指针位置：既用于"点在不在已有段里"，也作为新段的起点。
            const startFrame = selectionFrameFromClientX(e.clientX, false);
            const baseSelection = selectionRef.current ?? [];
            const hitIndex = rangeIndexAtFrame(baseSelection, startFrame);
            const hitExistingRange = hitIndex >= 0;
            const startClientX = e.clientX;
            let moved = false;
            // 起手用的是哪个键：拖动期间按这个掩码判断"键还按着"（左 1 / 右 2）。
            const pressMask = e.button === 2 ? 2 : 1;
            const isRightDrag = e.button === 2;

            (e.currentTarget as HTMLCanvasElement).setPointerCapture(e.pointerId);
            // 右键手势的收尾菜单：拖拽期间一律吞掉，**没拖动**时按"右键点击"
            // 处理（手工打开菜单）—— 与选区平移 / 曲线形变两处右键分支同一套。
            const suppressContextMenu = (ev: Event) => {
                ev.preventDefault();
                ev.stopImmediatePropagation();
            };
            if (isRightDrag) {
                window.addEventListener("contextmenu", suppressContextMenu, true);
            }
            const onMove = (ev: globalThis.PointerEvent) => {
                if ((ev.buttons & pressMask) !== pressMask) {
                    onUp(ev);
                    return;
                }
                // 只接受本指针的 move：第二指针（掌压触摸等）的
                // buttons 恒为 1，不校验 pointerId 会驱动本次手势。
                if (ev.pointerId !== e.pointerId) return;
                // 3px 死区：区分「点击切换」与「拖动追加」，避免手抖误删段
                if (!moved && Math.abs(ev.clientX - startClientX) > 3) {
                    moved = true;
                    // 确认构成拖拽：武装跨表面守卫，松手若落在别的表面上
                    // （标尺 / 轨道头）也不会弹出那个表面的菜单。
                    if (isRightDrag) armRightDragContextMenuGuard();
                }
                if (!moved) return;
                const bbFrame = selectionFrameFromClientX(ev.clientX, true);
                selectionRef.current = addPointerRange(baseSelection, startFrame, bbFrame);
                updateSelectionUi(selectionRef.current);
                invalidate();
            };
            const onUp = (ev?: globalThis.PointerEvent) => {
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onUp);
                window.removeEventListener("pointercancel", onUp);
                if (isRightDrag) {
                    window.removeEventListener("contextmenu", suppressContextMenu, true);
                }
                clearActivePointerGestureEnd(onUp);
                // 「原地点击已有段 = 取消该段」只属于**修饰键**路径：右键起手的
                // 分支只在未选中区域触发，右键点击在那里是"交回菜单"（见下），
                // 不应借帧坐标的边界夹取误差误删某一段。
                if (!moved && hitExistingRange && !isRightDrag) {
                    // 切换取消：移除被点击的那一段
                    // （removeRangeAtFrame 已覆盖"按点击帧定位该段"的语义）
                    selectionRef.current = removeRangeAtFrame(selectionRef.current, startFrame);
                    updateSelectionUi(selectionRef.current);
                }
                if (isRightDrag && !moved) {
                    // 右键点击（没有拖动）：未选中区域起手时既没有可取消的段，
                    // 也没有要追加的范围 —— 原样交回上下文菜单。
                    if (onContextMenu && document.hasFocus() && ev) {
                        onContextMenu(ev.clientX, ev.clientY);
                    }
                }
                invalidate();
            };
            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp);
            window.addEventListener("pointercancel", onUp);
            setActivePointerGestureEnd(onUp);
            return;
        }

        // 如果已有选区，且鼠标在选区范围内且靠近曲线，则进入拖拽曲线模式
        if (sel) {
            // 本块内四种选区交互的取舍（总表见上方"手势优先级"）：
            // - Alt + 边缘 = **拉伸选区内的曲线**（改参数值、进撤销栈）；
            // - 无修饰键 + 边缘 = **只调整选区边界**（不碰曲线、不进撤销栈）；
            // - 右键 + 选区内部 = **左右平移整个选区**（同样只动选区）；
            // - 已选中的参数线 = 拖动曲线（优先级最高，见 `onSelectedCurve`）。
            //
            // 【"拉伸"与"调整选区"是两件事】用户选定一段后想把范围改大改小，
            // 最直觉的做法就是抓住边缘拖 —— 那时他做的是**选区调整**，不是
            // 拉伸内容：所以无修饰键的边缘拖拽只改边界、不重采样选区内的曲线
            // 值，也不写撤销历史（否则用户按一次撤销，看到的却是"什么都没变"）。
            // Alt 才是"拉伸"这个显式意图。
            //
            // 边缘手势都只作用于**被抓住的那一段**（其余段不动）；命中判定取
            // 所有段中最近的边缘。
            const stretchHit = onSelectedCurve ? null : findStretchSelectionEdge(e);
            if (stretchHit) {
                // 被拉伸那一段的两条**数据边界**（整数帧）：左边界 = 首帧，
                // 右边界 = 末帧 + 1（半开区间右端）。拉伸就是把其中一条跟着手
                // 移走，另一条不动。指针位置按抓住的是哪条边界套对应规则
                // （见 `frameBoundLeftFromPointer` / `frameBoundRightFromPointer`）。
                const aBound = sel[stretchHit.rangeIndex].startFrame;
                const bBound = frameRangeEnd(sel[stretchHit.rangeIndex]);
                const edgeKind = stretchHit.edge;
                // pointerdown 时的选区快照：拉伸期间的选区改写都以它为基底
                // 重建（只替换被拉伸的那一段），因此归一化合并不会造成
                // 「越拉越偏」的下标漂移。
                const stretchBaseSelection = sel;
                // 本手势同时改曲线与选区：收尾时把这对选区快照登记到该手势
                // 打出的那个历史步骤上（后端持有，撤销恢复 `before`、重做恢复
                // `after`，见 recordGestureParamSelectionStep）。
                const selectionBeforeStretch = stretchBaseSelection.map((range) => ({
                    ...range,
                }));
                let stretchStartBound = aBound;
                let stretchEndBound = bBound;

                const pv = paramViewRef.current;
                if (!pv || pv.edit.length === 0) return;
                const stride = Math.max(1, pv.stride);
                // 源窗口 = 选区内**恰好那些帧**（`[aBound, bBound)`），逐帧取下标；
                // 不再有"拍 → 帧两端取整不对称"带来的多一帧。
                const oldStartBound = aBound;
                const oldEndBound = bBound;
                const oldStartIdx = clamp(
                    Math.round((oldStartBound - pv.startFrame) / stride),
                    0,
                    pv.edit.length - 1,
                );
                const oldLastIdx = clamp(
                    Math.round((oldEndBound - 1 - pv.startFrame) / stride),
                    oldStartIdx,
                    pv.edit.length - 1,
                );
                const oldValues = pv.edit.slice(oldStartIdx, oldLastIdx + 1);
                if (oldValues.length <= 0) return;

                const pid = e.pointerId;
                (e.currentTarget as HTMLCanvasElement).setPointerCapture(pid);
                setCanvasCursor("ew-resize");
                ensureLiveEditBase(pv);
                if (liveEditActiveRef) liveEditActiveRef.current = true;
                // 点击死区（见 PARAM_DRAG_DEAD_ZONE_PX）：按住拉伸修饰键在边缘
                // 点一下同样不该回写等值数据、不该多打一个撤销点。
                const startClientX = e.clientX;
                const startClientY = e.clientY;
                let didDrag = false;
                const finePointerState = createFineAdjustedPointerState(
                    e.nativeEvent,
                    e.currentTarget as HTMLCanvasElement,
                );

                const buildDense = (
                    pvNow: ParamViewSegment,
                    nextStartBoundRaw: number,
                    nextEndBoundRaw: number,
                ) => {
                    // 数据边界归一为整数帧，并把归一后的值返回给调用方复用 ——
                    // 于是"预览画出来的选区"与"提交写回去的选区"必然是同一组
                    // 数字，不会差半帧。
                    const nextStartBound = Math.max(0, snapFrame(nextStartBoundRaw));
                    const nextEndBound = Math.max(nextStartBound, snapFrame(nextEndBoundRaw));
                    const nextStartIdx = clamp(
                        Math.round((nextStartBound - pvNow.startFrame) / stride),
                        0,
                        pvNow.edit.length - 1,
                    );
                    const nextLastIdx = clamp(
                        Math.round((nextEndBound - 1 - pvNow.startFrame) / stride),
                        nextStartIdx,
                        pvNow.edit.length - 1,
                    );
                    const nextLen = nextLastIdx - nextStartIdx + 1;
                    if (nextLen <= 0) return null;

                    // 平滑度 → 毫秒定标的过渡带半宽（dense 索引）
                    const edgeHalfSpanIdx = edgeHalfSpanForIndices(nextLen, stride);
                    const extraEdgeFrames = Math.ceil(edgeHalfSpanIdx) * stride;
                    const overallMinFrame = Math.max(
                        0,
                        Math.min(oldStartBound, nextStartBound) - extraEdgeFrames,
                    );
                    const overallMaxFrame = Math.max(oldEndBound, nextEndBound) + extraEdgeFrames;
                    const overallLen = Math.floor((overallMaxFrame - overallMinFrame) / stride) + 1;
                    const dense = new Array<number>(overallLen);
                    for (let i = 0; i < overallLen; i += 1) {
                        const frame = overallMinFrame + i * stride;
                        const idx = Math.round((frame - pvNow.startFrame) / stride);
                        dense[i] = idx >= 0 && idx < pvNow.edit.length ? pvNow.edit[idx] : 0;
                    }
                    const denseBefore = dense.slice();

                    const newValues = new Array<number>(nextLen);
                    for (let i = 0; i < nextLen; i += 1) {
                        const t = nextLen > 1 ? i / (nextLen - 1) : 0;
                        const srcF = t * (oldValues.length - 1);
                        const lo = Math.floor(srcF);
                        const hi = Math.min(lo + 1, oldValues.length - 1);
                        const frac = srcF - lo;
                        const loVal = Number(oldValues[lo] ?? 0);
                        const hiVal = Number(oldValues[hi] ?? 0);
                        if (editParam === "pitch" && loVal === 0 && hiVal === 0) {
                            newValues[i] = 0;
                        } else {
                            newValues[i] = loVal + (hiVal - loVal) * frac;
                        }
                    }

                    for (let i = 0; i < nextLen; i += 1) {
                        const frame = nextStartBound + i * stride;
                        const dIdx = Math.round((frame - overallMinFrame) / stride);
                        if (dIdx >= 0 && dIdx < dense.length) {
                            dense[dIdx] = newValues[i];
                        }
                    }

                    const sampleOutsideValue = (srcFrame: number, fallback: number) => {
                        const srcIdx = Math.round((srcFrame - pvNow.startFrame) / stride);
                        if (srcIdx >= 0 && srcIdx < pvNow.edit.length) {
                            return pvNow.edit[srcIdx];
                        }
                        return fallback;
                    };

                    const maxOutsideWindow = Math.max(1, Math.round(oldValues.length * 0.2));
                    const smoothRatio = clamp(Number(edgeSmoothnessPercent) || 0, 0, 100) / 100;
                    const outsideWindowLen = Math.max(
                        1,
                        Math.round(maxOutsideWindow * smoothRatio),
                    );

                    // 缩短时，用原选区内侧一小段值回填被腾空区域。
                    // smoothness=0 时 outsideWindowLen=1，相当于边缘值沿边界内侧延展。
                    if (nextStartBound > oldStartBound) {
                        const fillLen = Math.floor((nextStartBound - oldStartBound) / stride);
                        for (let i = 0; i < fillLen; i += 1) {
                            const targetFrame = oldStartBound + i * stride;
                            const targetIdx = Math.round((targetFrame - overallMinFrame) / stride);
                            const srcWindowPos =
                                fillLen > 1
                                    ? Math.round((i / (fillLen - 1)) * (outsideWindowLen - 1))
                                    : 0;
                            const srcFrame = oldStartBound + srcWindowPos * stride;
                            if (targetIdx >= 0 && targetIdx < dense.length) {
                                dense[targetIdx] = sampleOutsideValue(srcFrame, dense[targetIdx]);
                            }
                        }
                    }
                    if (nextEndBound < oldEndBound) {
                        const fillLen = Math.floor((oldEndBound - nextEndBound) / stride);
                        for (let i = 0; i < fillLen; i += 1) {
                            const targetFrame = nextEndBound + (i + 1) * stride;
                            const targetIdx = Math.round((targetFrame - overallMinFrame) / stride);
                            const srcWindowPos =
                                fillLen > 1
                                    ? Math.round((i / (fillLen - 1)) * (outsideWindowLen - 1))
                                    : 0;
                            // 源取**末帧往内**（`oldEndBound - 1`）：源窗口就是选区内
                            // 那些帧，右切点本身不属于选区。
                            const srcFrame = oldEndBound - 1 - srcWindowPos * stride;
                            if (targetIdx >= 0 && targetIdx < dense.length) {
                                dense[targetIdx] = sampleOutsideValue(srcFrame, dense[targetIdx]);
                            }
                        }
                    }

                    const movedStartDenseIdx = Math.round(
                        (nextStartBound - overallMinFrame) / stride,
                    );
                    blendDenseEdges(
                        dense,
                        denseBefore,
                        movedStartDenseIdx,
                        nextLen,
                        edgeHalfSpanIdx,
                    );

                    return {
                        dense,
                        overallMinFrame,
                        overallMaxFrame,
                        nextStartBound,
                        nextEndBound,
                    };
                };

                /** 拉伸后的选区至少保留一个采样点（`stride` 帧）。 */
                const minFrameSpan = Math.max(1, stride);

                // 预览重算 rAF 合帧：dense 重建 + 整份 live 拷贝 +
                // React setState 都是 O(n)，pointermove 在高刷鼠标上
                // 可达数百 Hz，逐事件执行长选区会卡顿；与左键选区
                // 拖动路径同模式，一帧至多重算一次。
                let previewRafId: number | null = null;
                let queuedCursorFrame: number | null = null;
                const runPreviewStep = () => {
                    const cursorFrame = queuedCursorFrame;
                    if (cursorFrame == null) return;
                    queuedCursorFrame = null;
                    const pvNow = paramViewRef.current;
                    if (!pvNow) return;
                    // 指针 → **数据边界**：按抓住的是哪条边界选规则（与框选同一套，
                    // 见 `paramSelection` 的两条规则）。于是把左边界一路拖到工程
                    // 最左端，起点就能落到**第 0 帧**。
                    const cursorBound =
                        edgeKind === "left"
                            ? frameBoundLeftFromPointer(cursorFrame)
                            : frameBoundRightFromPointer(cursorFrame);
                    // 被抓住的那条边界跟着手走，另一条不动；两条边界之间至少留
                    // 一个采样点（`minFrameSpan`）。
                    //
                    // 【起点为什么夹在 >= 0（而框选不夹）】拉伸的窗口**就是**数据
                    // 窗口：重采样出来的值必须铺满它。工程起点之前没有数据，把起点
                    // 放到负帧会让"选区说 [-3, 5)、实际只写了 [0, 5)"两下对不上。
                    // 框选的选区是**视觉范围**，允许越过工程起点（数据侧再由
                    // `selectionToFrameRanges` 截断到 0），所以那边不夹。
                    const nextStartBound =
                        edgeKind === "left" ? clamp(cursorBound, 0, bBound - minFrameSpan) : aBound;
                    const nextEndBound =
                        edgeKind === "right"
                            ? Math.max(aBound + minFrameSpan, cursorBound)
                            : bBound;
                    const built = buildDense(pvNow, nextStartBound, nextEndBound);
                    if (!built) return;
                    // 用 buildDense **归一后**的边界：选区与曲线因此逐帧一致。
                    stretchStartBound = built.nextStartBound;
                    stretchEndBound = built.nextEndBound;
                    applyDenseToLiveEdit(
                        pvNow,
                        built.overallMinFrame,
                        built.dense,
                        built.overallMinFrame,
                        built.overallMaxFrame,
                        "draw",
                    );
                    selectionRef.current = normalizeSelection(
                        stretchBaseSelection.map((range, index) =>
                            index === stretchHit.rangeIndex
                                ? frameRangeFromFrames(
                                      built.nextStartBound,
                                      built.nextEndBound - built.nextStartBound,
                                  )
                                : range,
                        ),
                    );
                    updateSelectionUi(selectionRef.current);
                    invalidate();
                    // live 覆盖只改 ref：音量/动态拖拽时波形面需显式重绘。
                    requestWaveformRepaint();
                };

                const onMove = (ev: globalThis.PointerEvent) => {
                    if ((ev.buttons & 1) !== 1) {
                        onUp();
                        return;
                    }
                    // 只接受本指针的 move（掌压拒识的 move 侧防线）。
                    if (ev.pointerId !== pid) return;
                    // 点击死区：未越过阈值前不预览、不更新（画面上没有任何变化），
                    // 松手时按"点击"处理 —— 不回写数据、不打撤销点。拉伸路径的
                    // 收尾同样会整段回写并打检查点，没有位移时那是纯粹的噪音。
                    if (!didDrag) {
                        if (
                            Math.abs(ev.clientX - startClientX) <= PARAM_DRAG_DEAD_ZONE_PX &&
                            Math.abs(ev.clientY - startClientY) <= PARAM_DRAG_DEAD_ZONE_PX
                        ) {
                            return;
                        }
                        didDrag = true;
                    }
                    const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);
                    queuedCursorFrame = pointerFrame(adjusted.clientX);
                    if (previewRafId == null) {
                        previewRafId = requestAnimationFrame(() => {
                            previewRafId = null;
                            runPreviewStep();
                        });
                    }
                };

                const onUp = () => {
                    window.removeEventListener("pointermove", onMove);
                    window.removeEventListener("pointerup", onUp);
                    window.removeEventListener("pointercancel", onUp);
                    disposeFineAdjustedPointerState(finePointerState);
                    clearActivePointerGestureEnd(onUp);
                    // 取消挂起的预览帧：松手后的提交路径负责重建。
                    if (previewRafId != null) {
                        cancelAnimationFrame(previewRafId);
                        previewRafId = null;
                    }
                    // 提交前先同步消费挂起的游标：边界标量（stretchStartBound /
                    // stretchEndBound）只在 runPreviewStep 里更新，若松手与越过
                    // 死区的第一次 move 落在同一帧，取消 rAF 后标量仍是初始值，
                    // buildDense 会原样复现输入数据还多打一个"毫无变化"的撤销点
                    // （与自由绘制路径提交前先 flush 队列同一处理）。
                    runPreviewStep();
                    queuedCursorFrame = null;

                    // 单击（未越过点击死区）：这次手势没有改动任何值 —— 丢弃
                    // live 覆盖层并复位 active 标记后直接收尾。走提交路径会把
                    // 与原始数据逐帧相同的变换结果整段写回，并多打一个"按下毫无
                    // 变化"的撤销点。
                    if (!didDrag) {
                        liveEditOverrideRef.current = null;
                        if (liveEditActiveRef) {
                            liveEditActiveRef.current = false;
                        }
                        setCanvasCursor("default");
                        invalidate();
                        return;
                    }

                    const pvNow = paramViewRef.current;
                    if (!pvNow || !rootTrackId) {
                        // 早退同样必须复位 liveEdit（否则曲线刷新被永久搁置，
                        // 与 !didDrag 路径的处理一致）。
                        if (liveEditActiveRef) liveEditActiveRef.current = false;
                        setCanvasCursor("default");
                        return;
                    }
                    // 被拉伸那一段的最终边界（多选区只改这一段；帧口径）
                    const nextStartBound = stretchStartBound;
                    const nextEndBound = stretchEndBound;
                    const built = buildDense(pvNow, nextStartBound, nextEndBound);
                    if (!built) {
                        setCanvasCursor("default");
                        return;
                    }

                    // 提交前把选区钉到**本次提交用的那组边界**上：预览的最后一帧
                    // 可能被 rAF 取消，若沿用 `selectionRef.current`，撤销恢复的
                    // 选区就可能与真正写回去的数据差一帧。
                    selectionRef.current = normalizeSelection(
                        stretchBaseSelection.map((range, index) =>
                            index === stretchHit.rangeIndex
                                ? frameRangeFromFrames(
                                      built.nextStartBound,
                                      built.nextEndBound - built.nextStartBound,
                                  )
                                : range,
                        ),
                    );
                    updateSelectionUi(selectionRef.current);

                    const nextEdit = pvNow.edit.slice();
                    for (let i = 0; i < built.dense.length; i += 1) {
                        const frame = built.overallMinFrame + i * stride;
                        const idx = Math.round((frame - pvNow.startFrame) / stride);
                        if (idx >= 0 && idx < nextEdit.length) {
                            nextEdit[idx] = built.dense[i];
                        }
                    }
                    setParamView({ ...pvNow, edit: nextEdit });
                    // 同上：不立刻撤下覆盖层（见 onParamCommitSucceeded）。
                    onParamCommitSucceeded();

                    // 全分辨率无损提交（与右拖 / 选区拖拽同一范式）：
                    // buildDense 产生的是 pv 步距采样（dense[k] ↔
                    // overallMinFrame + k×stride）；按渲染同款线性插值
                    // 展开为逐帧值后分块回写 —— 绝不把 stride 间隔采样
                    // 当连续帧写入（旧实现在 stride>1 时时间压缩 +
                    // 覆盖未选帧，见 selectionEditData.expandStrideSampledDense）。
                    const expanded = expandStrideSampledDense(built.dense, stride);
                    // dyn 哨兵保留：拉伸是"读-变换-写回"，未画帧必须以哨兵
                    // 写回（数据源 = pv 自带位图，见
                    // restoreDynSentinelsFromParamView），否则被物化成显式
                    // 基线，日后基线重分析时响度静默漂移。
                    const expandedForUpload = isDynParam(editParam)
                        ? restoreDynSentinelsFromParamView(
                              expanded.slice(),
                              built.overallMinFrame,
                              pvNow,
                          )
                        : expanded;
                    // 拉伸后的选区（本轮手势的最终形态）：与拉伸前的快照
                    // 一起登记为该历史步骤的选区。
                    const selectionAfterStretch = selectionRef.current
                        ? selectionRef.current.map((range) => ({ ...range }))
                        : null;
                    void (async () => {
                        let committed = false;
                        try {
                            await uploadFullResCurve({
                                trackId: rootTrackId,
                                param: editParam,
                                startFrame: built.overallMinFrame,
                                values: expandedForUpload,
                            });
                            committed = true;
                        } catch (err) {
                            console.error("[pianoRoll] stretch-edge commit failed", err);
                            // IPC 失败必须撤下覆盖层，否则冻结的预览值会一直
                            // 盖在真实曲线上（与右拖 / 选区拖拽 / 形变同规）。
                            liveEditOverrideRef.current = null;
                        } finally {
                            if (liveEditActiveRef) {
                                liveEditActiveRef.current = false;
                            }
                            bumpRefreshToken();
                        }
                        if (!committed) return;

                        // 回写成功（首块已打检查点、位置已指向新步）之后，把本次
                        // 拉伸的选区变化登记到**那一步**上：撤销恢复拉伸前、重做
                        // 恢复拉伸后的选区。快照由后端持有，前端不做任何"我这是
                        // 第几步"的推断（见 recordGestureParamSelectionStep）。
                        await recordGestureParamSelectionStep({
                            before: selectionBeforeStretch,
                            after: selectionAfterStretch,
                        });
                    })();
                    setCanvasCursor("default");
                    invalidate();
                };

                window.addEventListener("pointermove", onMove);
                window.addEventListener("pointerup", onUp);
                window.addEventListener("pointercancel", onUp);
                setActivePointerGestureEnd(onUp);
                return;
            }

            // ── 边缘拖拽（无修饰键，左键）：只调整选区边界 ───────────────
            //
            // 与上面 Alt 拉伸的区别是**语义**：这里只挪动被抓住的那条边界，
            // 选区内已有的曲线值原样保留（不重采样、不写回后端），因此它既
            // 不需要 live 覆盖层、也不打撤销检查点 —— 撤销栈里应当只有"对参数
            // 的编辑"，把单纯的选区调整写进去会让"撤销"变成一次无变化的空操作。
            //
            // 让位规则：指针同时压住"已选中的参数线"时优先拖曲线（`onSelectedCurve`
            // 已在上方算好）；右键不参与本支（右键是"平移整个选区"，见下一段）。
            const selectionEdgeHit =
                onSelectedCurve || e.button !== 0
                    ? null
                    : findStretchSelectionEdge(e, { requireModifier: false });
            if (selectionEdgeHit) {
                const rangeIndex = selectionEdgeHit.rangeIndex;
                const edge = selectionEdgeHit.edge;
                // pointerdown 时的快照：拖拽期间每帧都以它重建（只替换被拖动的
                // 那一段）。以快照为基底而不是"当前选区"，是为了让
                // `normalizeSelection` 的合并/排序不会造成下标漂移 —— 交换左右
                // 也靠这一点保持稳定（固定端始终是"按下时的对侧边界"）。
                const baseSelection = sel;
                const pid = e.pointerId;
                (e.currentTarget as HTMLCanvasElement).setPointerCapture(pid);
                setCanvasCursor("ew-resize");

                const applyEdgePointer = (clientX: number, allowAutoScroll: boolean) => {
                    const pointerFrame = selectionFrameFromClientX(clientX, allowAutoScroll);
                    selectionRef.current = normalizeSelection(
                        baseSelection.map((range, index) =>
                            index === rangeIndex
                                ? resizeFrameRangeEdge(range, edge, pointerFrame)
                                : range,
                        ),
                    );
                    updateSelectionUi(selectionRef.current);
                    invalidate();
                };

                const onMove = (ev: globalThis.PointerEvent) => {
                    if ((ev.buttons & 1) !== 1) {
                        onUp();
                        return;
                    }
                    // 只接受本指针的 move（掌压拒识的 move 侧防线）。
                    if (ev.pointerId !== pid) return;
                    applyEdgePointer(ev.clientX, true);
                };
                const onUp = () => {
                    window.removeEventListener("pointermove", onMove);
                    window.removeEventListener("pointerup", onUp);
                    window.removeEventListener("pointercancel", onUp);
                    clearActivePointerGestureEnd(onUp);
                    // 收尾：把被拖到**不可见宽度**的段丢掉（与"单击不留零宽
                    // 选区"同一条规则）。判据与边缘命中同源 —— 像素宽度不足
                    // `SELECTION_EDGE_MIN_WIDTH_PX` 的段既看不到区域，也再也
                    // 抓不到它的边缘；留着只会留下"有光标提示、没东西可拖"
                    // 的状态。交换左右的过程中会瞬时经过零宽，那是正常的中途
                    // 形态，只有松手时才算数。
                    const current = selectionRef.current;
                    if (current) {
                        const visible = current.filter(
                            (range) =>
                                Math.abs(
                                    frameToViewportPx(frameRangeEndCut(range)) -
                                        frameToViewportPx(frameRangeStartCut(range)),
                                ) >= SELECTION_EDGE_MIN_WIDTH_PX,
                        );
                        if (visible.length !== current.length) {
                            selectionRef.current = normalizeSelection(visible);
                            updateSelectionUi(selectionRef.current);
                        }
                    }
                    invalidate();
                };

                window.addEventListener("pointermove", onMove);
                window.addEventListener("pointerup", onUp);
                window.addEventListener("pointercancel", onUp);
                setActivePointerGestureEnd(onUp);
                return;
            }

            // ── 右键拖拽选区内部：左右平移指针所在的那一段 ───────────────
            //
            // 与"边缘调整"（上一段）同属**选区自身的调整**：只平移边界，选区
            // 内的曲线值原样保留，不写后端、不进撤销栈。区别只是这里两条边界
            // **同时**跟着走（整段搬移），而不是只动被抓住的那一条。
            //
            // 【多段选区：只搬被抓住的那一段】平移的作用范围就是指针按下的那
            // 一段（`rangeIndexAtFrame` 定位），其余段纹丝不动 —— 与边缘调整
            // "只作用于被抓住的那一段"同一条规则。若整段一起搬，用户想只挪
            // 其中一段时会连带改掉全部，无法只调整局部。
            //
            // 让位规则（与全局优先级一致）：
            // - 指针压住已选中的参数线 → 由下面的曲线分支处理（右键在曲线上是
            //   形变/幅度变换，属"参数线交互"，优先级更高）；
            // - 形变修饰键按下 → 让位给 Alt 边缘拉伸（那段已在上方 return）。
            const shiftRangeIndex = rangeIndexAtFrame(sel, b);
            if (
                e.button === 2 &&
                !onSelectedCurve &&
                !isModifierActive(paramStretchKb, e.nativeEvent) &&
                shiftRangeIndex !== -1
            ) {
                // 平移量以「按下时指针所在帧」为基准，避免逐帧累加造成漂移。
                const grabFrame = selectionFrameFromClientX(e.clientX, false);
                // 快照：每帧都以它重建（只替换被搬动的那一段），归一化
                // （合并/排序）不会造成下标漂移。
                const baseSelection = sel;
                // 夹取只看**被搬动的那一段**的包围区间：其余段不参与，因此
                // 它们不会限制本次平移的可用范围。
                const startRange = sel[shiftRangeIndex];
                const pid = e.pointerId;
                (e.currentTarget as HTMLCanvasElement).setPointerCapture(pid);

                // 右键手势的收尾菜单：拖拽期间一律吞掉（否则松手会弹出上下文
                // 菜单）；**没拖动**时按"右键点击"处理，手工打开菜单 —— 与
                // 曲线上右键拖拽（形变）那一段同一套处理，见那里的注释。
                let didDrag = false;
                const startClientX = e.clientX;
                const suppressContextMenu = (ev: Event) => {
                    ev.preventDefault();
                    ev.stopImmediatePropagation();
                };
                const onMove = (ev: globalThis.PointerEvent) => {
                    if ((ev.buttons & 2) !== 2) {
                        onUp(ev);
                        return;
                    }
                    // 只接受本指针的 move（掌压拒识的 move 侧防线）。
                    if (ev.pointerId !== pid) return;
                    // 3px 死区：区分「右键点击」与「右键拖拽」，手抖不该被判成
                    // 拖拽（否则正常右键点击的菜单会被吞掉）。
                    if (!didDrag && Math.abs(ev.clientX - startClientX) > 3) {
                        didDrag = true;
                        // 光标只在确认拖拽后才变（右键点击不该留下拖拽光标）。
                        setCanvasCursor("grabbing");
                        // 确认构成拖拽：显式武装全局守卫，松手若落在**另一个**
                        // 表面（标尺 / 轨道头）上也能吞掉那个表面的菜单。
                        armRightDragContextMenuGuard();
                    }
                    // 死区内不平移：右键**点击**（含手抖）不得挪动选区，只有
                    // 越过死区才算"拖拽"（与多选追加 / 框选同一套语义）。
                    if (!didDrag) return;
                    const frame = selectionFrameFromClientX(ev.clientX, true);
                    // 位移夹到"平移后被搬动那一段仍在 [0, 工程末端] 内"——
                    // 段的长度不变，只让它停在两端（见 clampSelectionShift）。
                    // 基准点永远是**按下时**的帧：死区内不累计偏移，越过死区后
                    // 一次就对齐到真实位移（与多选追加 / 框选同一套死区语义）。
                    // 位移先取整到整数帧，夹取与落位才是同一组数字。
                    const delta = clampSelectionShift(
                        [startRange],
                        snapFrame(frame - grabFrame),
                        0,
                        maxSelectableFrame,
                    );
                    // 只替换被抓住的那一段，其余段原样保留（快照为基底，
                    // 因此归一化的合并/排序不会造成下标漂移）。
                    selectionRef.current = normalizeSelection(
                        baseSelection.map((range, index) =>
                            index === shiftRangeIndex
                                ? {
                                      startFrame: range.startFrame + delta,
                                      frameCount: range.frameCount,
                                  }
                                : range,
                        ),
                    );
                    updateSelectionUi(selectionRef.current);
                    invalidate();
                };
                const onUp = (ev?: globalThis.PointerEvent) => {
                    window.removeEventListener("pointermove", onMove);
                    window.removeEventListener("pointerup", onUp);
                    window.removeEventListener("pointercancel", onUp);
                    window.removeEventListener("contextmenu", suppressContextMenu, true);
                    clearActivePointerGestureEnd(onUp);
                    if (!didDrag) {
                        // 右键点击（没有平移）：选区原样保留，交回菜单。
                        if (onContextMenu && document.hasFocus() && ev) {
                            onContextMenu(ev.clientX, ev.clientY);
                        }
                    }
                    // 只重绘：选区平移已经实时生效，**不写撤销历史、不写后端**。
                    invalidate();
                };

                window.addEventListener("contextmenu", suppressContextMenu, true);
                window.addEventListener("pointermove", onMove);
                window.addEventListener("pointerup", onUp);
                window.addEventListener("pointercancel", onUp);
                setActivePointerGestureEnd(onUp);
                return;
            }

            if (rangeIndexAtFrame(sel, b) !== -1) {
                // 判断鼠标是否在**参数线**附近（沿折线的纵向距离 < 10px）
                const pv = paramViewRef.current;
                if (pv && pv.edit.length > 0) {
                    // 插值 + 投影 + pitch 的 +0.5 偏移 + 邻域判定全部抽到
                    // `kernel/gestureHitTest`（纯函数，有单测），与绘制折线同源。
                    //
                    // 【为什么必须沿折线插值】参数线是连续的：采样点之间由直线
                    // 相连，指针落在两点中间时，取"最近采样点的值"量到的是端点 y
                    // —— 陡峭段上可以远超 10px，于是"光标明明压在线上，却拖不动
                    // 参数线"（只有正好落在采样点上才抓得住）。
                    const canvas = canvasRef.current;
                    const rectH = canvas
                        ? canvas.getBoundingClientRect().height
                        : viewSizeRef.current.h || 1;
                    const curvePoint = curvePointAtPointer({
                        sec: framesToTime(b, pv.framePeriodMs),
                        startFrame: pv.startFrame,
                        stride: pv.stride,
                        framePeriodMs: pv.framePeriodMs,
                        values: pv.edit,
                        param: editParam,
                        valueToY: (v) => valueToY(editParam, v, rectH),
                    });
                    // 下面拖拽分支仍要用这两个量（选区帧范围换算 / 起点值），
                    // 因此保留声明，只是命中判定改走纯函数。
                    const fp = pv.framePeriodMs;
                    const mouseVal = pointerValue(e.clientY);
                    const near = isPointerNearCurve({
                        pointerY: canvas ? e.clientY - canvas.getBoundingClientRect().top : 0,
                        curveY: curvePoint?.y ?? Number.NaN,
                        hitRadiusPx: scaledHitRadius(
                            CURVE_HIT_RADIUS_PX,
                            profileFor(e.nativeEvent),
                        ),
                    });

                    if (near) {
                        if (e.button === 2) {
                            e.preventDefault();
                            const startClientY = e.clientY;
                            const pid = e.pointerId;
                            (e.currentTarget as HTMLCanvasElement).setPointerCapture(pid);
                            const finePointerState = createFineAdjustedPointerState(
                                e.nativeEvent,
                                e.currentTarget as HTMLCanvasElement,
                            );

                            setCanvasCursor("grabbing");
                            if (paramValuePopupEnabled) {
                                onParamValuePreviewChange?.({
                                    clientX: e.clientX,
                                    clientY: e.clientY,
                                    value: 0,
                                    displayText: formatRightDragMorphPercent(0),
                                });
                            }

                            // 多选区：每段独立取数、独立变换（断层两侧互不影响）
                            const rightDragSpans: FrameSpan[] = selectionToFrameSpans(sel);
                            if (rightDragSpans.length === 0) return;
                            // 拖拽数据与选区移动拖拽同一模式：先用 pv 近似值立即
                            // 预览，全分辨率到位后自动替换；提交必须等全分辨率
                            // 数据 —— 绝不把降采样值当连续帧写回后端（旧实现在
                            // stride>1 时会把 stride 间隔采样当连续帧写入：
                            // 时间压缩 + 覆盖未选帧，已修复）。
                            let origValuesPerRange: number[][] = rightDragSpans.map((span) =>
                                readPvRange(pv, span.startFrame, span.endFrame),
                            );
                            let lastDy = 0;
                            let didDrag = false;
                            // 下拖预览的 rAF 合帧状态（见 onMove）。
                            let previewRafId: number | null = null;
                            let previewQueuedDy: number | null = null;

                            ensureLiveEditBase(pv);
                            if (liveEditActiveRef) liveEditActiveRef.current = true;

                            const origCurvePromises = rightDragSpans.map((span) =>
                                fetchFullResCurve({
                                    trackId: rootTrackId ?? "",
                                    param: editParam,
                                    startFrame: span.startFrame,
                                    endFrame: span.endFrame,
                                    paramView: pv,
                                }),
                            );
                            let dragSettled = false;
                            void Promise.all(origCurvePromises)
                                .then((curves) => {
                                    if (dragSettled) return;
                                    origValuesPerRange = curves.map((curve) => curve.values);
                                })
                                .catch((err) => {
                                    // 取数失败只影响预览保真度：保持 pv 近似值，
                                    // 提交路径会再次尝试并有自己的失败处理。
                                    console.error("[pianoRoll] full-res curve fetch failed", err);
                                });

                            // 每段自己的残差放大器：趋势按该段值数组引用缓存，
                            // 每次 move 只做逐帧缩放，不重复高斯卷积。
                            const amplifierCache = new Map<
                                number,
                                { src: number[]; apply: (scale: number) => number[] }
                            >();
                            const transformRightDragRange = (
                                index: number,
                                dy: number,
                            ): number[] => {
                                const values = origValuesPerRange[index];
                                if (!values || values.length === 0) return [];
                                if (dy >= 0) {
                                    let cached = amplifierCache.get(index);
                                    if (!cached || cached.src !== values) {
                                        cached = {
                                            src: values,
                                            apply: createSelectionAmplifier(values, editParam, {
                                                framePeriodMs: fp,
                                            }).apply,
                                        };
                                        amplifierCache.set(index, cached);
                                    }
                                    return cached.apply(rightDragUpScale(dy));
                                }
                                return transformSelectionByRightDrag(values, editParam, dy, {
                                    framePeriodMs: fp,
                                });
                            };

                            const edgeHalfSpanForRange = (index: number) =>
                                edgeHalfSpanForIndices(origValuesPerRange[index]?.length ?? 0, 1);

                            /**
                             * 帧索引 dense 片段（values[k] ↔ startFrame + k）：
                             * 每段值已含变换，边缘淡化在计划内完成。复用与选区移动
                             * 完全相同的纯函数，任意 stride 下预览与提交一致。
                             */
                            const buildRightDragPlan = (pvNow: ParamViewSegment, dy: number) =>
                                buildMultiRangeEditPlan({
                                    ranges: rightDragSpans,
                                    valuesAt: (index) => transformRightDragRange(index, dy),
                                    sourceAt: makePvValueSource(pvNow),
                                    edgeHalfSpanAt: edgeHalfSpanForRange,
                                    isEditable:
                                        editParam === "pitch" ? editablePitchValue : undefined,
                                });

                            const suppressContextMenu = (ev: Event) => {
                                ev.preventDefault();
                                ev.stopImmediatePropagation();
                            };

                            const onMove = (ev: globalThis.PointerEvent) => {
                                if ((ev.buttons & 2) !== 2) {
                                    onUp(ev);
                                    return;
                                }
                                // 只接受本指针的 move（掌压拒识的 move 侧防线）。
                                if (ev.pointerId !== pid) return;
                                const adjusted = getFineAdjustedPointerPosition(
                                    finePointerState,
                                    ev,
                                );
                                const dy = startClientY - adjusted.clientY;
                                if (Math.abs(dy) >= 2) {
                                    didDrag = true;
                                    // 本次手势在自己的阈值处确认构成拖拽：
                                    // 显式武装收尾守卫。松手若发生在**另一个**
                                    // 表面（标尺 / 轨道头）上，那个表面的菜单
                                    // 会被吞掉（见守卫模块头注释）——本手势
                                    // 自己的 window 级抑制只覆盖画布路径。
                                    armRightDragContextMenuGuard();
                                }
                                lastDy = dy;

                                // 悬停弹窗跟随原始事件坐标（廉价，不合帧）。
                                if (paramValuePopupEnabled) {
                                    onParamValuePreviewChange?.({
                                        clientX: ev.clientX,
                                        clientY: ev.clientY,
                                        value: dy,
                                        displayText: formatRightDragMorphPercent(dy),
                                    });
                                }

                                // 下拖是 O(n·r) 高斯重卷积（r 随拖距增长）：
                                // rAF 合帧，一帧至多重算一次 —— 与上拖的
                                // "趋势缓存"路径保持同量级的每帧成本，长选区
                                // 全选拖动不再随拖距变卡。
                                previewQueuedDy = dy;
                                if (previewRafId == null) {
                                    previewRafId = requestAnimationFrame(() => {
                                        previewRafId = null;
                                        const queued = previewQueuedDy;
                                        previewQueuedDy = null;
                                        if (queued === null) return;
                                        const pvNow = paramViewRef.current;
                                        if (!pvNow) return;
                                        // 上拖：残差放大（各段趋势已缓存，仅逐帧
                                        // 缩放）；下拖：高斯平滑。两者都基于各段
                                        // 原始值无状态重算，段间互不影响。
                                        const pieces = buildRightDragPlan(pvNow, queued);
                                        for (const piece of pieces) {
                                            applyDenseToLiveEdit(
                                                pvNow,
                                                piece.startFrame,
                                                piece.values,
                                                piece.startFrame,
                                                piece.endFrame,
                                                "draw",
                                            );
                                        }
                                        invalidate();
                                        // live 覆盖只改 ref：音量/动态拖拽时波形面需显式重绘。
                                        requestWaveformRepaint();
                                    });
                                }
                            };

                            const onUp = async (ev?: globalThis.PointerEvent) => {
                                window.removeEventListener("pointermove", onMove);
                                window.removeEventListener("pointerup", onUp);
                                window.removeEventListener("pointercancel", onUp);
                                window.removeEventListener(
                                    "contextmenu",
                                    suppressContextMenu,
                                    true,
                                );
                                disposeFineAdjustedPointerState(finePointerState);
                                clearActivePointerGestureEnd(onUp);
                                if (previewRafId != null) {
                                    cancelAnimationFrame(previewRafId);
                                    previewRafId = null;
                                    previewQueuedDy = null;
                                }

                                if (!didDrag) {
                                    liveEditOverrideRef.current = null;
                                    if (liveEditActiveRef) {
                                        liveEditActiveRef.current = false;
                                    }
                                    setCanvasCursor("grab");
                                    if (onContextMenu && document.hasFocus() && ev) {
                                        onContextMenu(ev.clientX, ev.clientY);
                                    }
                                    invalidate();
                                    return;
                                }

                                // 释放后按下的 contextmenu 一律吞掉（右键拖拽
                                // 的收尾手势）；先注册再走后续早退分支，避免
                                // 异常退出时弹出原生菜单。
                                const suppressOnce = (evt: globalThis.MouseEvent) => {
                                    evt.preventDefault();
                                    evt.stopPropagation();
                                    window.removeEventListener("contextmenu", suppressOnce, true);
                                };
                                window.addEventListener("contextmenu", suppressOnce, true);
                                setTimeout(() => {
                                    window.removeEventListener("contextmenu", suppressOnce, true);
                                }, 0);

                                const pvNow = paramViewRef.current;
                                if (!pvNow || !rootTrackId) {
                                    if (liveEditActiveRef) {
                                        liveEditActiveRef.current = false;
                                    }
                                    setCanvasCursor("grab");
                                    invalidate();
                                    return;
                                }

                                // 全分辨率无损提交（与选区移动拖拽同一范式）：
                                // 等待拖动开始时发起的全分辨率取数，用最终 dy 在
                                // 真实曲线上重算变换，帧索引重建（含边缘淡化）后
                                // 分块回写 —— 预览用的 pv 近似值绝不写回后端。
                                // await 链路上的任何 IPC 失败都必须复位
                                // liveEdit 状态（否则预览覆盖层永久冻结、曲线
                                // 刷新被永久搁置）。
                                dragSettled = true;
                                try {
                                    // 先刷新各段的全分辨率原始值（拖动开始时已发起）
                                    const fullResCurves = await Promise.all(origCurvePromises);
                                    origValuesPerRange = fullResCurves.map((curve) => curve.values);
                                    // 提交基准 = 计划写入窗口（含边缘淡化扩展）的
                                    // 全分辨率数据；变换与预览同源（transformRightDragRange），
                                    // 因此写回结果与用户看到的预览逐帧一致。
                                    const commit = await fetchCommitBaseSource({
                                        trackId: rootTrackId,
                                        param: editParam,
                                        ranges: rightDragSpans,
                                        frameDelta: 0,
                                        edgeHalfSpanAt: edgeHalfSpanForRange,
                                        paramView: pvNow,
                                        withSentinel: isDynParam(editParam),
                                    });
                                    const pieces = buildMultiRangeEditPlan({
                                        ranges: rightDragSpans,
                                        valuesAt: (index) => transformRightDragRange(index, lastDy),
                                        sourceAt: commit.sourceAt,
                                        edgeHalfSpanAt: edgeHalfSpanForRange,
                                        isEditable:
                                            editParam === "pitch" ? editablePitchValue : undefined,
                                    });

                                    // 立即同步本地 paramView state（逐帧映射到 pv 的
                                    // 采样栅格上；pv 只是显示，随后会被后端数据刷新）
                                    const nextEdit = pvNow.edit.slice();
                                    const pvStepUp = Math.max(1, Math.floor(pvNow.stride));
                                    for (const piece of pieces) {
                                        for (let i = 0; i < piece.values.length; i += 1) {
                                            const globalIdx = Math.round(
                                                (piece.startFrame + i - pvNow.startFrame) /
                                                    pvStepUp,
                                            );
                                            if (globalIdx >= 0 && globalIdx < nextEdit.length) {
                                                nextEdit[globalIdx] = piece.values[i];
                                            }
                                        }
                                    }
                                    setParamView({ ...pvNow, edit: nextEdit });
                                    // 同上：不立刻撤下覆盖层（见 onParamCommitSucceeded）。
                                    onParamCommitSucceeded();

                                    // 分块回写：整次编辑（含所有段）只打一个撤销点，
                                    // 块之间让出事件循环，超长工程也不会卡死界面。
                                    // 上传失败同样复位 liveEdit（finally），不让覆盖层冻结。
                                    void (async () => {
                                        try {
                                            await uploadFullResCurveSegments({
                                                trackId: rootTrackId,
                                                param: editParam,
                                                segments: buildUploadSegments(
                                                    editParam,
                                                    pieces,
                                                    commit.sentinelAt,
                                                ),
                                            });
                                        } catch (err) {
                                            console.error(
                                                "[pianoRoll] right-drag upload failed",
                                                err,
                                            );
                                        } finally {
                                            if (liveEditActiveRef) {
                                                liveEditActiveRef.current = false;
                                            }
                                            bumpRefreshToken();
                                        }
                                    })();
                                } catch (err) {
                                    console.error("[pianoRoll] right-drag commit failed", err);
                                    liveEditOverrideRef.current = null;
                                    if (liveEditActiveRef) {
                                        liveEditActiveRef.current = false;
                                    }
                                }

                                setCanvasCursor("grab");
                                invalidate();
                            };

                            window.addEventListener("contextmenu", suppressContextMenu, true);
                            window.addEventListener("pointermove", onMove);
                            window.addEventListener("pointerup", onUp);
                            window.addEventListener("pointercancel", onUp);
                            setActivePointerGestureEnd(onUp);
                            return;
                        }

                        // 进入拖拽选中曲线模式（支持 X+Y 双向拖拽）
                        // 多选区语义：所有段同步移动同一 Δx/Δy，断层保持不变。
                        setCanvasCursor("grabbing");
                        const startMouseVal = mouseVal;
                        const startFrame = pointerFrame(e.clientX);
                        // 点击死区：未越过阈值前不预览、不更新，收尾时按"点击"
                        // 处理（不回写数据、不打撤销点，见 PARAM_DRAG_DEAD_ZONE_PX）。
                        const startClientX = e.clientX;
                        const startClientY = e.clientY;
                        let didDrag = false;
                        const pid = e.pointerId;
                        (e.currentTarget as HTMLCanvasElement).setPointerCapture(pid);
                        const finePointerState = createFineAdjustedPointerState(
                            e.nativeEvent,
                            e.currentTarget as HTMLCanvasElement,
                        );

                        // 选区帧闭区间（逐段，升序互斥）—— 注意这里**不**夹到 pv 的
                        // 已加载窗口内。「全选」时选区覆盖整个工程，若按 pv 裁剪，
                        // 后续只会改到显示范围内的参数，工程其余部分纹丝不动。
                        const dragSpans: FrameSpan[] = selectionToFrameSpans(sel);
                        if (dragSpans.length === 0) return;

                        // 取各段的全分辨率原始值。
                        // pv 在低缩放下按画布宽度做了降采样，直接拿去变换再回写
                        // 会把全分辨率曲线覆盖掉，所以变换的输入必须是 stride=1
                        // 的数据。pv 已以 stride=1 覆盖时零开销直接切片，行为与
                        // 改动前一致；否则分块向后端拉取（不阻塞拖动）。
                        const origCurvePromises = dragSpans.map((span) =>
                            fetchFullResCurve({
                                trackId: rootTrackId ?? "",
                                param: editParam,
                                startFrame: span.startFrame,
                                endFrame: span.endFrame,
                                paramView: pv,
                            }),
                        );
                        // 先用 pv 覆盖到的部分做即时预览；全分辨率数据到位后自动
                        // 替换，之后的每一帧预览都基于无损数据。
                        let origValuesPerRange: number[][] = dragSpans.map((span) =>
                            readPvRange(pv, span.startFrame, span.endFrame),
                        );
                        let dragSettled = false;
                        void Promise.all(origCurvePromises)
                            .then((curves) => {
                                if (dragSettled) return;
                                origValuesPerRange = curves.map((curve) => curve.values);
                            })
                            .catch(() => {
                                // 取数失败只降级预览保真度（继续用 pv 数据预览）；
                                // 提交路径（onUp）有自己的 try/catch 兜底。
                            });

                        // 逐帧变换。拖动过程中 `lastValueDelta` / `lastScaleStepDelta`
                        // 会被 onMove 更新，这里读到的是最新值。
                        const transformDragValue = (orig: number, frame: number) => {
                            if (
                                useScaleDegreeTranspose &&
                                editParam === "pitch" &&
                                dragAnchorScale
                            ) {
                                // 每个帧用其落地时刻的生效音阶做度数移调。
                                const frameScale =
                                    scaleAtSec?.((frame * fp) / 1000) ?? dragAnchorScale;
                                return orig === 0
                                    ? 0
                                    : transposePitchByScaleSteps(
                                          orig,
                                          lastScaleStepDelta,
                                          frameScale,
                                      );
                            }
                            // 动态（比值域）走**锚点缩放**：系数由"被抓住
                            // 那条线的值"导出，使锚点位移恰好等于指针位移
                            // （跟手），其余点按比例缩放（倍率域相对关系保留），
                            // 静音 0 × k 仍为 0。锚点贴地时该函数内部退回线性
                            // 偏移。完整法则收在 paramRanges.shiftDynValueForDrag，
                            // 预览与提交共用它。
                            if (isDynParam(editParam)) {
                                return shiftDynValueForDrag(
                                    orig,
                                    dynDragAnchor ?? Number.NaN,
                                    lastValueDelta,
                                );
                            }
                            // 其余参数：值域内**线性偏移** —— 纵轴是线性刻度，
                            // `原值 + Δ` 才让被抓取的那一点停在光标下
                            //（见 paramRanges.shiftValueForDrag）。
                            return shiftValueForDrag(editParam, orig, lastValueDelta);
                        };

                        // Tempo Map 感知：度数差以拖动锚点帧（选区整体起点）的
                        // 生效音阶计算 —— 多选区下仍以第一段起点为锚点，
                        // 保证一次拖动只有一个锚定音阶。
                        const dragAnchorFrame = dragSpans[0].startFrame;
                        const dragAnchorScale =
                            scaleAtSec?.((dragAnchorFrame * fp) / 1000) ?? projectScale;
                        ensureLiveEditBase(pv);
                        if (liveEditActiveRef) liveEditActiveRef.current = true;
                        if (
                            editParam === "pitch" ||
                            isChildPitchOffsetCentsParam(editParam) ||
                            isChildPitchOffsetDegreesParam(editParam) ||
                            isChildFormantOffsetCentsParam(editParam)
                        ) {
                            onPitchSnapGestureActiveChange?.(true);
                        }

                        // 用闭包变量记录最新 X/Y 偏移量
                        let lastValueDelta = 0;
                        /**
                         * 拖拽的**锚点值**：按下时指针所在帧上的曲线值
                         * （即"被抓住的那条线"的位置）。动态的拖拽幅度系数
                         * 由它导出（见 `paramRanges.dynDragScaleFactor`），
                         * 因此取**按下那一刻**的值、拖动过程中不变 ——
                         * 若跟着实时值走，系数会随拖动自身变化而自激。
                         *
                         * 命中判定已经算过这个值（`curvePoint`，用于 10px 邻域
                         * 判定）；此处只是把它固定下来。取不到时为 null，
                         * 动态将退回线性偏移。
                         */
                        const dynDragAnchor: number | null =
                            isDynParam(editParam) && curvePoint !== null ? curvePoint.value : null;
                        let lastScaleStepDelta = 0;
                        let useScaleDegreeTranspose = false;
                        let lastFrameDelta = 0; // 帧偏移（整数）
                        // 使用闭包变量跟踪当前拖动方向（可通过右键切换）
                        let currentDragDir = dragDirection ?? "y-only";

                        // 预览重算 rAF 合帧：与右键下拖路径同模式。pointermove
                        // 在高刷新率鼠标上可达 125–1000Hz，而每次预览是
                        // O(选区帧数) 的 dense 构建 + 整份 live 拷贝 + React
                        // setState（全面板重渲），长选区逐事件执行会明显卡顿；
                        // 一帧至多重算一次，onMove 只更新标量偏移。
                        let previewRafId: number | null = null;
                        let previewQueued = false;
                        const runPreviewStep = () => {
                            const pvNow = paramViewRef.current;
                            if (!pvNow) return;

                            // X 向平移会把早先帧写在计划窗口之外的值留在 live
                            // 覆盖里。不能整份重建（= null + ensure，每帧 O(窗口)
                            // 拷贝，见 resetLiveEditPreview 文档）：按"本手势写过的
                            // 区间并集"精确还原后再写本帧计划窗口。写入点
                            // applyDenseToLiveEdit 内部会按需建基准。
                            resetLiveEditPreview(pvNow, { keepWrittenRange: true });

                            // 多段计划：覆盖「各段原位 ∪ 落地位 ± 边缘淡化」的
                            // 合并窗口（逐帧索引）。预览从 pv 取上下文 —— pv 只是
                            // 显示数据，不会回写；提交走同一计划的另一份 sourceAt。
                            const pieces = buildMultiRangeEditPlan({
                                ranges: dragSpans,
                                frameDelta: lastFrameDelta,
                                valuesAt: (index) => origValuesPerRange[index],
                                sourceAt: makePvValueSource(pvNow),
                                transformAt: (_index, _i, sourceValue, targetFrame) =>
                                    transformDragValue(sourceValue, targetFrame),
                                edgeHalfSpanAt: (index) =>
                                    edgeHalfSpanForIndices(
                                        origValuesPerRange[index]?.length ?? 0,
                                        1,
                                    ),
                                isEditable: editParam === "pitch" ? editablePitchValue : undefined,
                            });

                            for (const piece of pieces) {
                                applyDenseToLiveEdit(
                                    pvNow,
                                    piece.startFrame,
                                    piece.values,
                                    piece.startFrame,
                                    piece.endFrame,
                                    "draw",
                                );
                            }

                            // 实时更新选区位置显示（所有段同步平移；纯平移不改变
                            // 段间距，故不会产生新的重叠段）。选区的单位就是帧，
                            // 因此**直接**用同一个帧位移，不再换算回拍。
                            selectionRef.current = shiftSelectionRanges(sel, lastFrameDelta);
                            updateSelectionUi(selectionRef.current);

                            invalidate();
                            // live 覆盖只改 ref：拖拽音量/动态时波形面需
                            // 显式重绘（波形 = 可听结果，须实时跟随）。
                            requestWaveformRepaint();
                        };
                        const schedulePreview = () => {
                            previewQueued = true;
                            if (previewRafId == null) {
                                previewRafId = requestAnimationFrame(() => {
                                    previewRafId = null;
                                    if (!previewQueued) return;
                                    previewQueued = false;
                                    runPreviewStep();
                                });
                            }
                        };

                        const onMove = (ev: globalThis.PointerEvent) => {
                            // 掌侧误触防御（与同文件其余手势一致）：触控笔/手指的
                            // pointermove 在 touch 场景同样满足 buttons&1，不加
                            // pointerId 卫兵会让第二指改写本次拖拽的位移状态。
                            if (ev.pointerId !== pid) return;
                            if ((ev.buttons & 1) !== 1) {
                                onUp();
                                return;
                            }
                            // 点击死区：越过阈值之前一律按"还没开始拖"处理 ——
                            // 不更新位移、不预览、不弹浮窗（因此画面上不会有任何
                            // 变化），松手时也不会回写。
                            if (!didDrag) {
                                if (
                                    Math.abs(ev.clientX - startClientX) <=
                                        PARAM_DRAG_DEAD_ZONE_PX &&
                                    Math.abs(ev.clientY - startClientY) <= PARAM_DRAG_DEAD_ZONE_PX
                                ) {
                                    return;
                                }
                                didDrag = true;
                            }
                            const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);
                            const currentVal = pointerValue(adjusted.clientY);
                            let rawValueDelta = currentVal - startMouseVal;

                            // 音高吸附：Toggle snap modifier (XOR with pitchSnapEnabled)
                            const effectiveSnap = isEffectivePitchSnapActive(ev);
                            const yDragEnabled = currentDragDir !== "x-only";
                            if (effectiveSnap && editParam === "pitch" && yDragEnabled) {
                                if (pitchSnapUnit === "scale" && dragAnchorScale) {
                                    useScaleDegreeTranspose = true;
                                    lastScaleStepDelta = scaleStepDeltaBetween(
                                        startMouseVal,
                                        currentVal,
                                        dragAnchorScale,
                                    );
                                    rawValueDelta = 0;
                                } else {
                                    useScaleDegreeTranspose = false;
                                    rawValueDelta = Math.round(rawValueDelta);
                                }
                            } else if (
                                effectiveSnap &&
                                isChildPitchOffsetCentsParam(editParam) &&
                                yDragEnabled
                            ) {
                                useScaleDegreeTranspose = false;
                                rawValueDelta = Math.round(rawValueDelta / 100) * 100;
                            } else if (
                                effectiveSnap &&
                                isChildPitchOffsetDegreesParam(editParam) &&
                                yDragEnabled
                            ) {
                                useScaleDegreeTranspose = false;
                                rawValueDelta = Math.round(rawValueDelta);
                            } else if (
                                effectiveSnap &&
                                isChildFormantOffsetCentsParam(editParam) &&
                                yDragEnabled
                            ) {
                                useScaleDegreeTranspose = false;
                                rawValueDelta = Math.round(rawValueDelta / 50) * 50;
                            } else {
                                useScaleDegreeTranspose = false;
                                if (!yDragEnabled) {
                                    lastScaleStepDelta = 0;
                                    rawValueDelta = 0;
                                }
                            }

                            // 计算 X 方向帧偏移：指针帧坐标之差，取整到整数帧
                            // （选区与数据都按整数帧落位，无需任何单位换算）。
                            const currentFrame = pointerFrame(adjusted.clientX);
                            const rawFrameDelta = snapFrame(currentFrame - startFrame);

                            // 应用拖动方向限制
                            lastValueDelta = yDragEnabled ? rawValueDelta : 0;
                            lastFrameDelta = currentDragDir === "y-only" ? 0 : rawFrameDelta;

                            // 悬停弹窗跟随原始事件坐标（廉价，不合帧）。
                            if (paramValuePopupEnabled) {
                                const previewCurrentVal = yDragEnabled ? currentVal : startMouseVal;
                                onParamValuePreviewChange?.({
                                    clientX: ev.clientX,
                                    clientY: ev.clientY,
                                    value: getSelectDragPreviewValue({
                                        editParam,
                                        startValue: startMouseVal,
                                        currentValue: previewCurrentVal,
                                        fineScale: 1,
                                        effectiveSnap,
                                        pitchSnapUnit,
                                        projectScale:
                                            scaleAtSec?.(pointerSec(ev.clientX)) ?? projectScale,
                                    }),
                                });
                            }

                            // O(n) 预览重算合帧到下一渲染帧（见上方注释）。
                            schedulePreview();
                        };

                        const onUp = async () => {
                            window.removeEventListener("pointermove", onMove);
                            window.removeEventListener("pointerup", onUp);
                            window.removeEventListener("pointercancel", onUp);
                            disposeDragDirKey();
                            disposeFineAdjustedPointerState(finePointerState);
                            clearActivePointerGestureEnd(onUp);
                            // 取消挂起的预览帧：松手后由提交路径以全分辨率
                            // 数据重建，预览层不得再覆盖提交结果。
                            if (previewRafId != null) {
                                cancelAnimationFrame(previewRafId);
                                previewRafId = null;
                            }
                            previewQueued = false;
                            // 标记手势结束，避免已发出的取数请求再覆写 origValues
                            dragSettled = true;

                            // 捕获阶段的 contextmenu/mousedown 抑制与临时 snap
                            // 高亮是纯清理：必须放在所有早退路径（包括下面的
                            // 单击死区 return）之前，否则一次普通点击就会把
                            // window 级监听器永久留在窗口上——全局右键菜单从此
                            // 失效，且残留的 mousedown 监听还会让下一次拖拽的
                            // 右键换向被执行两次。
                            window.removeEventListener(
                                "contextmenu",
                                onContextMenuDuringDrag,
                                true,
                            );
                            window.removeEventListener("mousedown", onMouseDownDuringDrag, true);
                            if (
                                editParam === "pitch" ||
                                isChildPitchOffsetCentsParam(editParam) ||
                                isChildPitchOffsetDegreesParam(editParam)
                            ) {
                                onPitchSnapGestureActiveChange?.(false);
                            }

                            // 单击（未越过点击死区）：这次手势没有改动任何值 ——
                            // 丢弃 live 覆盖层并复位 active 标记后直接收尾。
                            // 走提交路径会把变换结果整段写回后端（此处与原始数据
                            // 逐帧相同）并多打一个"按下毫无变化"的撤销点。
                            if (!didDrag) {
                                liveEditOverrideRef.current = null;
                                if (liveEditActiveRef) {
                                    liveEditActiveRef.current = false;
                                }
                                if (paramValuePopupEnabled) {
                                    onParamValuePreviewChange?.(null);
                                }
                                setCanvasCursor("grab");
                                invalidate();
                                return;
                            }

                            // 提交拖拽结果到后端
                            const pvNow = paramViewRef.current;
                            if (pvNow && rootTrackId) {
                                // await 链路上的任何 IPC 失败都必须复位
                                // liveEdit 状态（否则预览覆盖层永久冻结、
                                // 曲线刷新被永久搁置）。
                                try {
                                    // 等待拖动开始时发起的全分辨率取数完成 —— 提交必须
                                    // 基于无损数据，不能拿降采样的 pv 值去覆盖后端。
                                    const fullResCurves = await Promise.all(origCurvePromises);
                                    origValuesPerRange = fullResCurves.map((curve) => curve.values);

                                    // 边缘平滑度：毫秒定标的过渡带半宽（帧），
                                    // 扩展写入窗口以包含各段边界外侧上下文。
                                    // 提交 dense 恒为 stride=1。
                                    const edgeHalfSpanForDragRange = (index: number) =>
                                        edgeHalfSpanForIndices(
                                            origValuesPerRange[index]?.length ?? 0,
                                            1,
                                        );

                                    // 提交前把计划写入窗口（「各段原位 ∪ 落地位 ±
                                    // 边缘淡化」合并后）的全分辨率数据拉下来作为基底。
                                    // 这是「回写不失真」的关键：预览用的是可能降采样的
                                    // pv，提交必须用 stride=1 的真实曲线，否则会把
                                    // 降采样后的值写回后端、覆盖掉原始分辨率。
                                    const commit = await fetchCommitBaseSource({
                                        trackId: rootTrackId,
                                        param: editParam,
                                        ranges: dragSpans,
                                        frameDelta: lastFrameDelta,
                                        edgeHalfSpanAt: edgeHalfSpanForDragRange,
                                        paramView: pvNow,
                                        withSentinel: isDynParam(editParam),
                                    });

                                    const pieces = buildMultiRangeEditPlan({
                                        ranges: dragSpans,
                                        frameDelta: lastFrameDelta,
                                        valuesAt: (index) => origValuesPerRange[index],
                                        sourceAt: commit.sourceAt,
                                        transformAt: (_index, _i, sourceValue, targetFrame) =>
                                            transformDragValue(sourceValue, targetFrame),
                                        edgeHalfSpanAt: edgeHalfSpanForDragRange,
                                        isEditable:
                                            editParam === "pitch" ? editablePitchValue : undefined,
                                    });

                                    if (pieces.length > 0) {
                                        // 立即同步更新本地 paramView state（逐帧映射到 pv 的
                                        // 采样栅格上；pv 只是显示，随后会被后端数据刷新）
                                        const nextEdit = pvNow.edit.slice();
                                        const pvStepUp = Math.max(1, Math.floor(pvNow.stride));
                                        for (const piece of pieces) {
                                            for (let i = 0; i < piece.values.length; i += 1) {
                                                const globalIdx = Math.round(
                                                    (piece.startFrame + i - pvNow.startFrame) /
                                                        pvStepUp,
                                                );
                                                if (globalIdx >= 0 && globalIdx < nextEdit.length) {
                                                    nextEdit[globalIdx] = piece.values[i];
                                                }
                                            }
                                        }
                                        setParamView({
                                            ...pvNow,
                                            edit: nextEdit,
                                        });
                                        // 选区拖拽不走 commitStroke，这里显式收尾：
                                        // 保留覆盖层直到"提交之后取的快照"到达。
                                        onParamCommitSucceeded();

                                        // 确保选区位置最终正确（多段同步平移；帧位移直接复用）
                                        selectionRef.current = shiftSelectionRanges(
                                            sel,
                                            lastFrameDelta,
                                        );
                                        updateSelectionUi(selectionRef.current);

                                        // 分块回写：整次编辑（含所有段）只打一个撤销点，
                                        // 块之间让出事件循环，超长工程也不会卡死界面。
                                        // 上传失败同样复位 liveEdit（finally），不让
                                        // 覆盖层冻结。
                                        // 拖动前的选区快照：本手势把选区与数据
                                        // 一起平移，因此撤销时选区也必须一起回来
                                        // （与拉伸同一套登记，见其注释）。
                                        const selectionBeforeDrag = sel
                                            ? sel.map((range) => ({ ...range }))
                                            : null;
                                        const selectionAfterDrag = selectionRef.current
                                            ? selectionRef.current.map((range) => ({
                                                  ...range,
                                              }))
                                            : null;
                                        void (async () => {
                                            let uploaded = false;
                                            try {
                                                await uploadFullResCurveSegments({
                                                    trackId: rootTrackId,
                                                    param: editParam,
                                                    segments: buildUploadSegments(
                                                        editParam,
                                                        pieces,
                                                        commit.sentinelAt,
                                                    ),
                                                });
                                                uploaded = true;
                                            } catch (err) {
                                                console.error(
                                                    "[pianoRoll] select-drag upload failed",
                                                    err,
                                                );
                                            } finally {
                                                if (liveEditActiveRef)
                                                    liveEditActiveRef.current = false;
                                                bumpRefreshToken();
                                            }
                                            if (!uploaded) return;
                                            await recordGestureParamSelectionStep({
                                                before: selectionBeforeDrag,
                                                after: selectionAfterDrag,
                                            });
                                        })();
                                    }
                                } catch (err) {
                                    console.error("[pianoRoll] select-drag commit failed", err);
                                    liveEditOverrideRef.current = null;
                                    if (liveEditActiveRef) liveEditActiveRef.current = false;
                                }
                            } else {
                                if (liveEditActiveRef) liveEditActiveRef.current = false;
                            }
                            if (paramValuePopupEnabled) {
                                onParamValuePreviewChange?.(null);
                            }

                            setCanvasCursor("grab");
                            invalidate();
                        };

                        // 拖拽过程中右键点击切换拖动方向
                        const onContextMenuDuringDrag = (ev: Event) => {
                            ev.preventDefault();
                            ev.stopImmediatePropagation();
                        };
                        const cycleSelectDragDir = () => {
                            const order: Array<"free" | "x-only" | "y-only"> = [
                                "free",
                                "x-only",
                                "y-only",
                            ];
                            const idx = order.indexOf(currentDragDir);
                            currentDragDir = order[(idx + 1) % order.length];
                            // Also cycle the global setting
                            if (onCycleDragDirection) onCycleDragDirection("select");
                        };
                        const onMouseDownDuringDrag = (ev: globalThis.MouseEvent) => {
                            if (ev.button !== 2) return;
                            // 仅在左键拖拽进行中时，右键才切换拖拽方向。
                            if ((ev.buttons & 1) !== 1) return;
                            ev.preventDefault();
                            ev.stopPropagation();
                            cycleSelectDragDir();
                        };
                        // 触控板替代：拖拽中按下快捷键与右键同义。
                        const disposeDragDirKey = installDragDirectionKeyCycler(cycleSelectDragDir);

                        window.addEventListener("pointermove", onMove);
                        window.addEventListener("pointerup", onUp);
                        window.addEventListener("pointercancel", onUp);
                        window.addEventListener("contextmenu", onContextMenuDuringDrag, true);
                        window.addEventListener("mousedown", onMouseDownDuringDrag, true);
                        setActivePointerGestureEnd(onUp);
                        return;
                    }
                }
            }
        }

        // 默认行为：仅左键创建新选区（**替换**整个选区为单段）；
        // 右键不应在 pointerdown 时清除选区。
        // 指针 → 帧坐标的换算 selectionFrameFromClientX 已在上方定义
        // （与多选追加共用）；"指针 → 边界"的两条规则在 `paramSelection`。
        if (e.button === 0) {
            const startFrame = selectionFrameFromClientX(e.clientX, false);
            // 【按下时**什么都不改**】按下只是"武装"一次可能的框选：用户可能
            // 只是长按（还没决定要拖），此时既不该亮出新选区，也不该动原选区。
            // 选区在**第一次真正拖动**时（越过下面的死区）才建出来。
            //
            // 【为什么不能"先写一段再等拖动覆盖"】那样长按时就会先亮出一段
            // 选区（哪怕只有一帧），而手还没动 —— 反馈与意图不符。
            //
            // 【为什么不需要在这里标脏】没有任何状态被改动：原选区框保持不动
            // 正是"按下还没发生什么"的正确画面。第一次拖动时 onMove 会赋**新
            // 对象**（主画布内容签名走引用比较，见 `mainCanvasSignature.ts`）
            // 并 invalidate，缓存随之失效。
            const entrySelection = selectionRef.current;
            const pid = e.pointerId;
            (e.currentTarget as HTMLCanvasElement).setPointerCapture(pid);
            const finePointerState = createFineAdjustedPointerState(
                e.nativeEvent,
                e.currentTarget as HTMLCanvasElement,
            );
            // 区分"单击"与"框选"的死区（与多选追加分支同一阈值）。
            //
            // 【为什么必须区分】长按与手抖期间也会来 pointermove（1px 抖动、
            // 笔的压力漂移），不做死区判定的话这些抖动就会被当成"拖拽"而建出
            // 选区；松手时的收尾也靠它决定"这一次到底是单击还是框选"。
            const startClientX = e.clientX;
            let moved = false;
            const onMove = (ev: globalThis.PointerEvent) => {
                if ((ev.buttons & 1) !== 1) {
                    onUp();
                    return;
                }
                // 只接受本指针的 move（掌压拒识的 move 侧防线）。
                if (ev.pointerId !== pid) return;
                if (!moved && Math.abs(ev.clientX - startClientX) > 3) moved = true;
                // 拖拽途中选区被外部清除（如 BackSpace / 取消选择）时不再续建。
                // 判据必须带上"按下时本来**有**选区"：本次拖拽正要创建第一个
                // 选区时 `selectionRef.current` 本来就是 null，只判 null 会把它
                // 误当成"被清除"，框选永远建不起来。
                if (entrySelection != null && selectionRef.current == null) return;
                if (!moved) return;
                const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);
                const bbFrame = selectionFrameFromClientX(adjusted.clientX, true);
                // 整段拖拽都落在工程起点之前（左侧那段额外空间）→
                // `frameRangeFromPointers` 给出"无选区"：既不建新选区，也**不动**
                // 原选区（否则在空白处拖一下就会把用户的选区清掉）。
                const next = selectionFromPointers(startFrame, bbFrame);
                if (next == null) return;
                selectionRef.current = next;
                updateSelectionUi(selectionRef.current);
                invalidate(); // 实时重绘选区
            };
            const onUp = () => {
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onUp);
                window.removeEventListener("pointercancel", onUp);
                disposeFineAdjustedPointerState(finePointerState);
                clearActivePointerGestureEnd(onUp);
                if (!moved) {
                    // 单击（含笔/触摸的抖动）：没有拖动就没有新选区，同时把
                    // 按下前的那一段清掉 —— 与"单击空白处 = 取消选择"一致。
                    selectionRef.current = null;
                    updateSelectionUi(null);
                }
                invalidate();
            };
            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp);
            window.addEventListener("pointercancel", onUp);
            setActivePointerGestureEnd(onUp);
            return;
        }
        // 右键：不在 pointerdown 时清除或重建选区，交由 contextmenu 或右键拖拽逻辑处理
        if (e.button === 2) {
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
        // Line tool: draw a straight line from start to current pointer
        const startFrame = frame;
        const startValue = value;
        // 橡皮端（button 5）在 buttons 位掩码里是位 32，不是右键的位 2。
        const requiredButtonMask =
            mode === "restore" ? (penEraserDown ? PEN_ERASER_BUTTONS_MASK : 2) : 1;
        let currentDragDir: "free" | "x-only" = dragDirection === "x-only" ? "x-only" : "free";
        const canCycleDragDirection = e.button === 0;

        {
            // 起手自起手预设：颤音工具取活动预设，直线工具固定取直线预设。
            // 深度 / 速率都是预设自带值 —— 预设切换已持久化，不存在
            // "上一次的调整"要续。
            const working = createDragWorking(dragPreset, editParam, currentParamRange);
            // 先记下锚点：用户可能一起手就滚轮 / 按方向键调参，那时还没有
            // 任何 pointermove，气泡得有个位置可放。
            vibratoHudAnchorRef.current = { clientX: e.clientX, clientY: e.clientY };
            vibratoStateRef.current = {
                pointerId: e.pointerId,
                startFrame,
                startValue,
                currentFrame: startFrame,
                currentValue: startValue,
                mode,
                // 手势从起手时的工具开始；此后由本手势自己的切换改写
                // （见 `VibratoDragState.tool` 的说明）。
                tool: isLineTool ? "line" : "vibrato",
                working,
                shiftHeld: snapToggleHeld,
                seed: vibratoSeedForPreset(working.preset),
            };
            setVibratoDragCaptureActive(true);
        }
        const finePointerState = createFineAdjustedPointerState(
            e.nativeEvent,
            e.currentTarget as HTMLCanvasElement,
        );

        // 直线 / 颤音工具是「起点 → 当前点」的端点式预览：每个事件都重算
        // 全线 dense，O(线长)。pen 高采样率下逐事件执行 = 同帧多次
        // 全线重建；与自由绘制分支同一 rAF 合帧模式（事件入队，一帧
        // 一消费，getCoalescedEvents 展开同帧采样）。端点式预览只需
        // 队列里**最后一个**采样点即可得到正确结果。
        let lineRafId: number | null = null;
        let pendingLineEvent: globalThis.PointerEvent | null = null;
        // 最近一次处理的采样：边缘自动滚屏之后按**同一个**指针位置重跑一次
        // 端点式预览（视图滚了，指针屏幕坐标没变，对应的帧号却变了）。
        let lastLineSample: {
            clientX: number;
            clientY: number;
            ctrlKey: boolean;
            shiftKey: boolean;
            altKey: boolean;
        } | null = null;
        // 直线 / 颤音同样启用**持续**边缘滚屏：端点必须跟着视图继续延伸。
        // 端点式预览每帧整段重算，因此这里重跑一次天然正确 ——
        // 没有自由绘制那种"补一笔"的语义问题。
        onEdgeScrolled = (clientX) => {
            const base = lastLineSample;
            if (base === null) return;
            processLineEvent({ ...base, clientX });
        };

        const processLineEvent = (ev: {
            clientX: number;
            clientY: number;
            ctrlKey: boolean;
            shiftKey: boolean;
            altKey: boolean;
        }) => {
            const st = strokeRef.current;
            if (!st || st.pointerId !== e.pointerId) return;
            lastLineSample = {
                clientX: ev.clientX,
                clientY: ev.clientY,
                ctrlKey: ev.ctrlKey,
                shiftKey: ev.shiftKey,
                altKey: ev.altKey,
            };
            const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);
            const f2 = timeToFrame(pointerSec(adjusted.clientX), fp);
            const yDragEnabled = currentDragDir !== "x-only";
            const rawV2 = yDragEnabled ? pointerValue(adjusted.clientY) : value;
            const moveSnapToggleHeld = isSnapToggleModifierHeld(ev);
            const v2 = isDrawMode ? snapDrawValue(rawV2, moveSnapToggleHeld, f2) : rawV2;

            // Update stroke to only have start and current end
            st.points = [
                { frame: startFrame, value: startValue },
                { frame: f2, value: v2 },
            ];

            const pv2 = paramViewRef.current;
            if (pv2) {
                // 擦掉上一帧的直线预览，避免端点往回拖时留下残影。
                // 只还原上一帧写过的区间（旧实现是丢弃整份覆盖再整份拷贝，
                // 每帧一次 O(窗口) 拷贝）。
                resetLiveEditPreview(pv2);
                const minF = Math.min(startFrame, f2);
                const maxF = Math.max(startFrame, f2);
                if (mode === "restore") {
                    applyDenseToLiveEdit(pv2, minF, null, minF, maxF, mode);
                } else {
                    // 只有颤音内核一条路径：直线工具喂的是直线预设（深度 0），
                    // 内核输出的就是两端之间的那条直线 —— 与"另写一条插值"相比
                    // 少了第二份会与内核漂移的实现。
                    const vib = vibratoStateRef.current;
                    if (vib) {
                        vib.currentFrame = f2;
                        vib.currentValue = v2;
                        vib.shiftHeld = moveSnapToggleHeld;
                        const built = buildVibratoDense(
                            startFrame,
                            startValue,
                            f2,
                            v2,
                            vib.working,
                            moveSnapToggleHeld,
                            vib.seed,
                        );
                        applyDenseToLiveEdit(
                            pv2,
                            built.minF,
                            built.dense,
                            built.minF,
                            built.maxF,
                            mode,
                        );
                        reportVibratoDragState(adjusted.clientX, adjusted.clientY);
                    }
                }
            }
            invalidate();
            // 绘制中：live 覆盖只改 ref，波形面需显式重绘才能按动态值变化。
            requestWaveformRepaint();
        };
        const flushPendingLine = () => {
            lineRafId = null;
            const ev = pendingLineEvent;
            pendingLineEvent = null;
            if (ev != null) processLineEvent(ev);
        };

        const onMove = (ev: globalThis.PointerEvent) => {
            if ((ev.buttons & requiredButtonMask) !== requiredButtonMask) {
                onUp();
                return;
            }
            const st = strokeRef.current;
            if (!st || st.pointerId !== e.pointerId) return;
            // 边缘自动滚屏：立刻滚一帧并维持 rAF 循环（指针停在边缘不动时
            // 也继续滚）。滚动引发的端点重算由 `onEdgeScrolled` 负责。
            edgeScroll.track(ev.clientX);
            // 端点式预览：只保留最新采样（含同帧 coalesced 末点）。
            const coalesced = coalescedEventsOf(ev);
            pendingLineEvent = coalesced[coalesced.length - 1] ?? ev;
            if (lineRafId == null) {
                lineRafId = requestAnimationFrame(flushPendingLine);
            }
        };

        const cancelPendingLineFrame = () => {
            if (lineRafId != null) {
                cancelAnimationFrame(lineRafId);
                lineRafId = null;
            }
            pendingLineEvent = null;
        };

        const onUp = () => {
            const st = strokeRef.current;
            const isOwnStroke = Boolean(st && st.pointerId === e.pointerId);
            const vib = isOwnStroke ? vibratoStateRef.current : null;
            // 提交前先同步消费挂起的端点采样：processLineEvent 依赖
            // strokeRef 在位（按 pointerId 自守卫），先 flush 再清引用，
            // 保证 st.points 收到最后一个采样对应的端点。
            if (lineRafId != null) {
                cancelAnimationFrame(lineRafId);
                lineRafId = null;
            }
            const pending = pendingLineEvent;
            pendingLineEvent = null;
            if (pending != null) processLineEvent(pending);
            if (isOwnStroke) {
                strokeRef.current = null;
            }
            disposeDragDirKey();
            disposeFineAdjustedPointerState(finePointerState);
            // 停掉边缘自动滚屏：松手后视图绝不能继续滚（见 driver 的 `stop`）。
            onEdgeScrolled = null;
            edgeScroll.stop();
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            window.removeEventListener("pointercancel", onCancel);
            window.removeEventListener("contextmenu", onContextMenuDuringDraw, true);
            window.removeEventListener("mousedown", onMouseDownDuringDraw, true);
            clearActivePointerGestureEnd(onUp);
            invalidate();
            if (!isOwnStroke || !st) return;
            if (
                editParam === "pitch" ||
                isChildPitchOffsetCentsParam(editParam) ||
                isChildPitchOffsetDegreesParam(editParam)
            ) {
                onPitchSnapGestureActiveChange?.(false);
            }
            void (async () => {
                if (usesVibratoPath && vib && st.mode === "draw") {
                    const built = buildVibratoDense(
                        vib.startFrame,
                        vib.startValue,
                        vib.currentFrame,
                        vib.currentValue,
                        vib.working,
                        vib.shiftHeld,
                        vib.seed,
                    );
                    const densePoints = built.dense.map((valueAtFrame, idx) => ({
                        frame: built.minF + idx,
                        value: valueAtFrame,
                    }));
                    await commitStroke(densePoints, st.mode);
                    await applyPostStrokeSmoothing(densePoints, st.mode);
                } else {
                    await commitStroke(st.points, st.mode);
                    await applyPostStrokeSmoothing(st.points, st.mode);
                }
            })();
            finishVibratoDrag();
            vibratoStateRef.current = null;
            setVibratoDragCaptureActive(false);
        };

        /**
         * pointercancel 收尾：**回滚**而不是提交。OS 主动取消（触摸
         * 接管 / 掌拒 / 笔离开感应区）意味着手势不可信，半截笔画
         * 不应写进后端 —— 后端从未收到数据，丢弃 live 预览层即可。
         */
        const onCancel = () => {
            const st = strokeRef.current;
            const isOwnStroke = Boolean(st && st.pointerId === e.pointerId);
            if (isOwnStroke) {
                strokeRef.current = null;
            }
            cancelPendingLineFrame();
            vibratoStateRef.current = null;
            setVibratoDragCaptureActive(false);
            disposeDragDirKey();
            disposeFineAdjustedPointerState(finePointerState);
            // 取消路径同样必须停掉自动滚屏（否则视图会一直滚下去）。
            onEdgeScrolled = null;
            edgeScroll.stop();
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            window.removeEventListener("pointercancel", onCancel);
            window.removeEventListener("contextmenu", onContextMenuDuringDraw, true);
            window.removeEventListener("mousedown", onMouseDownDuringDraw, true);
            clearActivePointerGestureEnd(onUp);
            // 与 `onUp` 同款收尾：HUD 气泡（深度 / 速率 / 预设）挂在面板状态上，
            // 只有 `onVibratoDragEnd` 会把它撤下。取消路径漏掉这一步，气泡会
            // 永久停在画面上 —— 取消本就"什么都没发生"，却留下一个读数框。
            //（`blur` 路径不受影响：它经 `activePointerGestureEnd` 走 `onUp`。）
            finishVibratoDrag();
            liveEditOverrideRef.current = null;
            if (liveEditActiveRef) liveEditActiveRef.current = false;
            invalidate();
        };

        const onContextMenuDuringDraw = (ev: Event) => {
            ev.preventDefault();
            ev.stopImmediatePropagation();
        };
        const cycleLineDragDir = () => {
            if (!canCycleDragDirection) return;
            currentDragDir = currentDragDir === "free" ? "x-only" : "free";
            if (onCycleDragDirection) {
                // 直线与颤音共用同一份拖动方向，因此两者都归到 `"vibrato"`。
                onCycleDragDirection(usesVibratoPath ? "vibrato" : "draw");
            }
        };
        const onMouseDownDuringDraw = (ev: globalThis.MouseEvent) => {
            // 鼠标侧键（X1 / X2）在颤音拖拽期间切换预设。
            //
            // 【为什么在这里而不是画布的 pointerdown】见
            // `onCanvasPointerDown` 顶部那段注释：指针捕获不重定向
            // `pointerdown`，命中测试按真实坐标走，画布收不到。这条
            // `window` + 捕获阶段的 `mousedown` 不依赖命中测试，与下面
            // "右键切拖动方向"用的是同一个入口。
            //
            // 【为什么判 `buttons` 位掩码而不是 `button`】`button` 是
            // 事件里"哪个键变了"（侧键是 3 / 4），此时左键仍在按住，因此
            // 用位掩码确认"这次确实是拖拽中按下"（左键位 1 仍在）。
            if (usesVibratoPath) {
                // 两个侧键**同时**按下 = 切到直线工具（并清掉两条
                // 调整记录）。`buttons` 里两位都在，说明两次按下有重叠。
                const bothSideButtons =
                    (ev.buttons & SIDE_BUTTON_BACK_MASK) !== 0 &&
                    (ev.buttons & SIDE_BUTTON_FORWARD_MASK) !== 0;
                const direction = resolveVibratoSideButton(ev.button);
                if (direction !== null || bothSideButtons) {
                    if ((ev.buttons & 1) !== 1) return;
                    ev.preventDefault();
                    ev.stopPropagation();
                    if (bothSideButtons) {
                        applyVibratoReset("line", ev.clientX, ev.clientY);
                        return;
                    }
                    // 走到这里 `direction` 必非 null（仅 bothSideButtons 分支
                    // 才可能两者皆空，而那已在上面返回）。
                    if (direction === null) return;
                    const choice = cycleVibratoChoice(
                        vibratoPresetCycleList,
                        vibratoCycleAnchorId({
                            lineTool: vibratoStateRef.current?.tool === "line",
                            currentPresetId: vibratoStateRef.current?.working.preset.id ?? null,
                        }),
                        direction,
                    );
                    if (choice) {
                        applyVibratoChoice(choice, ev.clientX, ev.clientY);
                    }
                    return;
                }

                // 中键：按**当前滚轮所处的那一路**重置（默认是振幅；按住
                // 频率修饰键时是频率）。与滚轮共用同一套修饰键判定。
                if (ev.button === 1) {
                    if ((ev.buttons & 1) !== 1) return;
                    ev.preventDefault();
                    ev.stopPropagation();
                    applyVibratoReset(
                        resolveVibratoMiddleClickReset(ev, vibratoFrequencyAdjustKb),
                        ev.clientX,
                        ev.clientY,
                    );
                    return;
                }
            }

            if (ev.button !== 2) return;
            if (!canCycleDragDirection) return;
            // 仅在左键拖拽进行中时，右键才切换拖拽方向。
            if ((ev.buttons & 1) !== 1) return;
            ev.preventDefault();
            ev.stopPropagation();
            cycleLineDragDir();
        };
        // 触控板替代：拖拽中按下快捷键与右键同义。
        const disposeDragDirKey = installDragDirectionKeyCycler(cycleLineDragDir);

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
        window.addEventListener("pointercancel", onCancel);
        window.addEventListener("contextmenu", onContextMenuDuringDraw, true);
        window.addEventListener("mousedown", onMouseDownDuringDraw, true);
        setActivePointerGestureEnd(onUp);
    } else {
        // Draw tool: freehand drawing with interpolation between points
        // 橡皮端（button 5）在 buttons 位掩码里是位 32，不是右键的位 2。
        const requiredButtonMask =
            mode === "restore" ? (penEraserDown ? PEN_ERASER_BUTTONS_MASK : 2) : 1;
        let currentDragDir: "free" | "x-only" = dragDirection === "x-only" ? "x-only" : "free";
        const canCycleDragDirection = e.button === 0;
        const finePointerState = createFineAdjustedPointerState(
            e.nativeEvent,
            e.currentTarget as HTMLCanvasElement,
        );
        // 最近一次处理的采样（含修饰键）。边缘自动滚屏之后要按**同一个**
        // 指针位置重跑一次：视图滚了、指针屏幕坐标没变，但它对应的帧号变了。
        // 不重跑，笔画就会在滚动处跳变（视图过去了、线没跟上）。
        let lastStrokeSample: {
            clientX: number;
            clientY: number;
            ctrlKey: boolean;
            shiftKey: boolean;
            altKey: boolean;
        } | null = null;
        // 绘制类工具启用**持续**边缘滚屏（指针停在边缘不动也继续滚），
        // 见 `kernel/edgeScrollDriver` 的模块头说明。
        onEdgeScrolled = (clientX) => {
            const base = lastStrokeSample;
            if (base === null) return;
            // 用"当前指针 x + 最后一次采样的其余字段"重跑：修饰键 / 拖动方向
            // 等手势状态必须与用户最后一次真实输入一致，而 x 取回调带过来的
            // 最新值（比队列里那份可能还新一个事件）。
            processStrokeEvent({ ...base, clientX });
            // 自由绘制的 `processStrokeEvent` 只调 `invalidate()`（主画布）——
            // 波形面的重绘一向由外层的合帧循环统一发出。这里绕过了那层循环，
            // 必须自己补一次，否则滚屏期间波形面按旧动态值显示。
            requestWaveformRepaint();
        };

        // 逐事件处理的主体从 onMove 拆出：onMove 只把事件压进待处理
        // 队列并请求 rAF，真正的 dense 构建 + live 写入 + 重绘合帧到
        // 每渲染帧一次。pen 采样率（133–266Hz+）高于渲染帧率，逐事件
        // 执行 O(n) dense 分配会让同一帧内多次重算；与选区拖拽的
        // schedulePreview 同一模式。
        const processStrokeEvent = (ev: {
            clientX: number;
            clientY: number;
            ctrlKey: boolean;
            shiftKey: boolean;
            altKey: boolean;
        }) => {
            const st = strokeRef.current;
            if (!st || st.pointerId !== e.pointerId) return;
            lastStrokeSample = {
                clientX: ev.clientX,
                clientY: ev.clientY,
                ctrlKey: ev.ctrlKey,
                shiftKey: ev.shiftKey,
                altKey: ev.altKey,
            };
            const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);
            const last = st.points[st.points.length - 1];
            const f2 = timeToFrame(pointerSec(adjusted.clientX), fp);
            const yDragEnabled = currentDragDir !== "x-only";
            const rawV2Abs = yDragEnabled ? pointerValue(adjusted.clientY) : (last?.value ?? value);
            const rawV2 = rawV2Abs;
            const moveSnapToggleHeld = isSnapToggleModifierHeld(ev);
            const v2 = isDrawMode ? snapDrawValue(rawV2, moveSnapToggleHeld, f2) : rawV2;

            const pv2 = paramViewRef.current;
            if (last && last.frame === f2) {
                last.value = v2;
                if (pv2) {
                    applyDenseToLiveEdit(pv2, f2, mode === "restore" ? null : [v2], f2, f2, mode);
                }
            } else if (last) {
                const a = { frame: last.frame, value: last.value };
                const b = { frame: f2, value: v2 };
                st.points.push(b);

                const minF = Math.min(a.frame, b.frame);
                const maxF = Math.max(a.frame, b.frame);

                let dense: number[] | null = null;
                if (mode !== "restore") {
                    const len = maxF - minF + 1;
                    dense = new Array<number>(len);
                    const denom = b.frame - a.frame;
                    for (let f = minF; f <= maxF; f += 1) {
                        const t = denom === 0 ? 1 : (f - a.frame) / denom;
                        dense[f - minF] = a.value + (b.value - a.value) * t;
                    }
                }

                if (pv2) {
                    applyDenseToLiveEdit(pv2, minF, dense, minF, maxF, mode);
                }
            }
            invalidate();
            // live 覆盖只改 ref，波形面收不到通知 —— 重绘请求由本帧的
            // 处理循环末尾统一发出（见 flushPendingMoves / onUp）：
            // 同一帧内的全部 coalesced 采样只重绘一次。
        };

        // 待处理事件队列 + rAF 合帧状态。flush 后队列即清空；onUp/
        // onCancel 里取消挂起帧（挂起的帧只能看到"队列已消费"的状态，
        // 不取消也不会重复处理，但显式取消避免收尾后还跑一次回调）。
        let pendingMoveEvents: globalThis.PointerEvent[] = [];
        let strokeRafId: number | null = null;
        const flushPendingMoves = () => {
            strokeRafId = null;
            const queue = pendingMoveEvents;
            pendingMoveEvents = [];
            let processed = false;
            for (const ev of queue) {
                // 同帧多点：getCoalescedEvents 展开高速运笔时一帧内
                // 积累的全部采样，轨迹逐点保留（不支持时回退单事件）。
                for (const sample of coalescedEventsOf(ev)) {
                    processStrokeEvent(sample);
                    processed = true;
                }
            }
            // 波形重绘放在采样循环**之外**：一帧内的全部采样合起来只请求
            // 一次重绘（`requestWaveformRepaint` 自身也做帧内合并，这里
            // 少的是每次采样的闸门判定与修订号自增）。轨迹点仍逐采样
            // 累积到 `st.points`，提交保真度不变。
            if (processed) requestWaveformRepaint();
        };

        const onMove = (ev: globalThis.PointerEvent) => {
            if ((ev.buttons & requiredButtonMask) !== requiredButtonMask) {
                onUp();
                return;
            }
            const st = strokeRef.current;
            if (!st || st.pointerId !== e.pointerId) return;
            // 边缘自动滚屏：`track` 会立刻滚一帧并维持 rAF 循环（指针停在
            // 边缘不动时也继续滚）。**不在这里读坐标做处理** —— 事件仍走
            // 下面的合帧队列，滚动引发的重算由 `onEdgeScrolled` 负责。
            edgeScroll.track(ev.clientX);
            pendingMoveEvents.push(ev);
            if (strokeRafId == null) {
                strokeRafId = requestAnimationFrame(flushPendingMoves);
            }
        };

        const cancelPendingStrokeFrame = () => {
            if (strokeRafId != null) {
                cancelAnimationFrame(strokeRafId);
                strokeRafId = null;
            }
            pendingMoveEvents = [];
        };

        const onUp = () => {
            const st = strokeRef.current;
            const isOwnStroke = Boolean(st && st.pointerId === e.pointerId);
            // 提交前先同步消费挂起的采样队列：processStrokeEvent 依赖
            // strokeRef 仍在位（其内部按 pointerId 自守卫），必须先
            // flush 再清空引用，否则最后一帧内的轨迹点会丢失
            // （快速运笔时笔尾缺一段 ≤16ms 的点）。
            if (strokeRafId != null) {
                cancelAnimationFrame(strokeRafId);
                strokeRafId = null;
            }
            const queued = pendingMoveEvents;
            pendingMoveEvents = [];
            let processedQueued = false;
            for (const ev of queued) {
                for (const sample of coalescedEventsOf(ev)) {
                    processStrokeEvent(sample);
                    processedQueued = true;
                }
            }
            // 末帧同样只请求一次（见 flushPendingMoves）。
            if (processedQueued) requestWaveformRepaint();
            if (isOwnStroke) {
                strokeRef.current = null;
                vibratoStateRef.current = null;
                setVibratoDragCaptureActive(false);
            }
            disposeDragDirKey();
            disposeFineAdjustedPointerState(finePointerState);
            // 停掉边缘自动滚屏：松手后视图绝不能继续滚（见 driver 的 `stop`）。
            onEdgeScrolled = null;
            edgeScroll.stop();
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            window.removeEventListener("pointercancel", onCancel);
            window.removeEventListener("contextmenu", onContextMenuDuringDraw, true);
            window.removeEventListener("mousedown", onMouseDownDuringDraw, true);
            clearActivePointerGestureEnd(onUp);
            invalidate();
            if (!isOwnStroke || !st) return;
            if (
                editParam === "pitch" ||
                isChildPitchOffsetCentsParam(editParam) ||
                isChildPitchOffsetDegreesParam(editParam)
            ) {
                onPitchSnapGestureActiveChange?.(false);
            }
            void (async () => {
                await commitStroke(st.points, st.mode);
                await applyPostStrokeSmoothing(st.points, st.mode);
            })();
        };

        /**
         * pointercancel 收尾：**回滚**而不是提交（与直线 / 颤音工具分支
         * 同一语义）。OS 主动取消意味着手势不可信，半截笔画不写后端。
         */
        const onCancel = () => {
            const st = strokeRef.current;
            const isOwnStroke = Boolean(st && st.pointerId === e.pointerId);
            if (isOwnStroke) {
                strokeRef.current = null;
            }
            cancelPendingStrokeFrame();
            vibratoStateRef.current = null;
            setVibratoDragCaptureActive(false);
            disposeDragDirKey();
            disposeFineAdjustedPointerState(finePointerState);
            // 取消路径同样必须停掉自动滚屏（否则视图会一直滚下去）。
            onEdgeScrolled = null;
            edgeScroll.stop();
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            window.removeEventListener("pointercancel", onCancel);
            window.removeEventListener("contextmenu", onContextMenuDuringDraw, true);
            window.removeEventListener("mousedown", onMouseDownDuringDraw, true);
            clearActivePointerGestureEnd(onUp);
            liveEditOverrideRef.current = null;
            if (liveEditActiveRef) liveEditActiveRef.current = false;
            invalidate();
        };

        const onContextMenuDuringDraw = (ev: Event) => {
            ev.preventDefault();
            ev.stopImmediatePropagation();
        };
        const cycleFreehandDragDir = () => {
            if (!canCycleDragDirection) return;
            currentDragDir = currentDragDir === "free" ? "x-only" : "free";
            if (onCycleDragDirection) {
                onCycleDragDirection("draw");
            }
        };
        const onMouseDownDuringDraw = (ev: globalThis.MouseEvent) => {
            if (ev.button !== 2) return;
            if (!canCycleDragDirection) return;
            // 仅在左键拖拽进行中时，右键才切换拖拽方向。
            if ((ev.buttons & 1) !== 1) return;
            ev.preventDefault();
            ev.stopPropagation();
            cycleFreehandDragDir();
        };
        // 触控板替代：拖拽中按下快捷键与右键同义。
        const disposeDragDirKey = installDragDirectionKeyCycler(cycleFreehandDragDir);

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
        window.addEventListener("pointercancel", onCancel);
        window.addEventListener("contextmenu", onContextMenuDuringDraw, true);
        window.addEventListener("mousedown", onMouseDownDuringDraw, true);
        setActivePointerGestureEnd(onUp);
    }
}
