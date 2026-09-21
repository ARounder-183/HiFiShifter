import type {
    KeyboardEvent,
    MouseEvent as ReactMouseEvent,
    MutableRefObject,
    PointerEvent as ReactPointerEvent,
    UIEvent,
} from "react";
import { useCallback, useEffect, useRef } from "react";

import type { ParamFramesPayload } from "../../../types/api";
import type { AppDispatch } from "../../../app/store";
import { paramsApi } from "../../../services/api";
import {
    recordParamSelectionStretchStep,
    seekPlayhead,
    setplayheadSec,
} from "../../../features/session/sessionSlice";
import { clamp, MAX_PX_PER_SEC, MIN_PX_PER_SEC } from "../timeline";
import type { LiveEditOverride } from "./useLiveParamEditing";
import type {
    ParamMorphOverlay,
    ParamName,
    ParamViewSegment,
    StrokeMode,
    StrokePoint,
    ValueViewport,
} from "./types";
import { computeDynGeometricMean } from "./selectionTransforms";
import { isDynParam, shiftDynValueForDrag, shiftValueForDrag } from "./paramRanges";
import {
    curveValueAtPointerFrame,
    hitTestSelectionBody,
    hitTestSelectionEdge,
    isPointerNearCurve,
} from "./kernel/gestureHitTest";
import {
    beatToFrameDelta,
    edgeAutoScrollDeltaPx,
    frameDeltaToBeat,
    selectionIndexRange,
} from "./kernel/dragArithmetic";
import type { MutableRefObject as MutRef } from "react";
import { isModifierActive, isNoneBinding } from "../../../features/keybindings/keybindingsSlice";
import {
    matchesKeybinding,
    matchesKeybindingAllowingFineModifier,
} from "../../../features/keybindings/useKeybindings";
import { ACTION_META } from "../../../features/keybindings/defaultKeybindings";
import type { Keybinding } from "../../../features/keybindings/types";
import type { KeybindingMap, ActionId } from "../../../features/keybindings/types";
import {
    scaleStepDeltaBetween,
    snapToScale,
    snapToSemitone,
    transposePitchByScaleSteps,
} from "../../../utils/musicalScales";
import type { ScaleLike } from "../../../utils/musicalScales";
import {
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
    isChildFormantOffsetCentsParam,
    snapChildPitchOffsetValue,
} from "./childPitchOffsetParams";
import { resolveHorizontalWheelZoom } from "../timeline/runtime/timelineScrollRange";
import { resolveTimelineMinPxPerSec } from "../timeline/runtime/timelineZoomBounds";
import { nativeScrollbarZoneAt } from "../../../utils/nativeScrollbar";
import {
    coalescedEventsOf,
    isEraserButton,
    isLegacyMouseEventFromStylus,
    isStylusLike,
    PEN_ERASER_BUTTONS_MASK,
    shouldRejectConcurrentPointer,
} from "../../../utils/penInput";
import { getParamEditorWheelAction, getVibratoDragWheelTarget } from "./wheelGesture";
import { armRightDragContextMenuGuard } from "../../../utils/rightDragContextMenuGuard";
import {
    createSelectionAmplifier,
    rightDragUpScale,
    transformSelectionByRightDrag,
} from "./selectionTransforms";
import {
    applyEdgeBlend,
    drawSmoothSigmaMsFromStrength,
    edgeHalfSpanFramesForSelection,
    editablePitchValue,
    smoothCurveGaussian,
} from "./paramSmoothing";
import {
    buildMultiRangeEditPlan,
    expandStrideSampledDense,
    fetchFullResCurve,
    planSelectionEditWindows,
    readPvRange,
    uploadFullResCurve,
    uploadFullResCurveSegments,
} from "./selectionEditData";
import {
    addBeatRange,
    beatRangesToInclusiveSpans,
    normalizeSelection,
    rangeIndexAtBeat,
    removeRangeAtBeat,
    selectionFromBeatRange,
    shiftSelectionRanges,
    type FrameSpan,
    type ParamSelection,
} from "./paramSelection";
import {
    computeVibratoDragAdjustment,
    resolveVibratoDragKeyboardAdjustment,
} from "./vibratoDragAdjust";
import {
    formatRightDragMorphPercent,
    getDrawPreviewValue,
    getSelectDragPreviewValue,
} from "./paramValuePreviewLogic";
import { secFromViewportClientX } from "./seekPlayheadMapping";
import { resolvePanelRenderViewport } from "./kernel/viewportSource";
import {
    createTimelineAxis,
    secToViewportPx,
    viewportPxToSec,
} from "../renderKernel/timelineAxis.js";

type CanvasCursor = "default" | "crosshair" | "grab" | "grabbing" | "ew-resize";

/**
 * pv（可能被降采样）的逐帧读取器。
 *
 * 只用于**即时预览**：pv 是显示数据（低缩放下按画布宽度降采样），拿去回写会
 * 覆盖全分辨率曲线，因此提交路径必须另取全分辨率数据。预览与提交共用同一套
 * 变换函数（见 selectionEditData），所以两者结果一致。
 */
function makePvValueSource(pv: ParamViewSegment): (frame: number) => number {
    const step = Math.max(1, Math.floor(pv.stride));
    return (frame: number) => {
        const idx = Math.round((frame - pv.startFrame) / step);
        return idx >= 0 && idx < pv.edit.length ? pv.edit[idx] : 0;
    };
}

export function usePianoRollInteractions(args: {
    dispatch: AppDispatch;
    rootTrackId: string | null;
    editParam: ParamName;
    pitchEnabled: boolean;
    toolMode: string;
    secPerBeat: number;
    scrollLeftRef: MutableRefObject<number>;
    pxPerBeatRef: MutableRefObject<number>;
    pxPerSecRef: MutableRefObject<number>;
    /**
     * **交互用视口的权威来源**（内核真值；宿主未挂载时返回 null）。
     *
     * 【为什么必须有它——这是一个真实缺陷的根因】面板的三个视口 ref 会在**渲染期**
     * 被同步成 React state 的值，而横向滚动位置的真值在内核、state 只是**量化提交**
     * （256px 步长，见宿主 `SCROLL_COMMIT_STEP_PX`）：滚轮 / 拖 thumb / 触摸只改内核
     * 与镜像，最后一次不足 256px 的位移**永远不会**提交进 state。
     *
     * 于是同一次框选手势里出现了两套视口：
     * - **绘制**（选区块、曲线）走内核真值（`resolvePanelRenderViewport`）；
     * - **拍坐标换算**（起点 / 终点 / 边缘命中）走 refs，而 refs 会被那个滞后的
     *   state 在渲染期覆盖回去。
     *
     * 结果：划定的选区与鼠标划过的区域相差 `内核位置 − 滞后 state`（可达 256px），
     * 且差值随量化残差变化 → 用户报告"有可能造成偏移、随机出现"。
     *
     * 复现步骤也完全对得上：**先做一次水平缩放**会经 `flushSync` 原子提交 state
     * （state == 内核，故正常）；**之后任何一次滚动**都只动内核、state 逐渐滞后，
     * 于是开始出现偏移；**再做一次缩放**又原子对齐一次，偏移"恢复正常"。
     *
     * 因此交互侧的坐标换算必须与渲染侧同源：一律取内核真值，宿主不在时才退回 refs。
     * 这与 `resolvePanelRenderViewport`（渲染侧）和时间轴的 `livePxPerSec` 是同一条
     * 既有约定，本字段只是把它接到交互路径上。
     */
    getViewportTruth: () => { pxPerSec: number; scrollLeft: number } | null;
    /** 连续滚轮缩放期间暂存最新结果，下一 tick 以此为基础继续锚定。 */
    horizontalZoomChainRef: MutableRefObject<{
        nextPxPerSec: number;
        nextScrollLeft: number;
    } | null>;
    /** 水平缩放结果回调（与轨道视图共用同一套计算逻辑）。 */
    onHorizontalZoom: (nextPxPerSec: number, nextScrollLeft: number) => void;
    /** 是否启用“同步时间轴视图”。 */
    syncTimelineEnabled: boolean;
    /** 轨道头与钢琴卷帘之间的水平偏移（ref，始终与缩放应用侧一致）。 */
    timelineOffsetRef: MutableRefObject<number>;
    /** 项目时长（秒），用于计算缩放时的 maxScroll */
    dynamicProjectSec: number;
    setPitchView: (next: ValueViewport) => void;
    setParamViewport: (param: string, next: ValueViewport) => void;
    pitchViewRef: MutableRefObject<ValueViewport>;
    paramViewsRef: MutableRefObject<Record<string, ValueViewport>>;
    scrollerRef: MutableRefObject<HTMLDivElement | null>;
    canvasRef: MutableRefObject<HTMLCanvasElement | null>;
    viewSizeRef: MutableRefObject<{ w: number; h: number }>;

    /** 多选区（升序、互不相交、相邻已合并；null = 无选区） */
    selectionRef: MutableRefObject<ParamSelection | null>;
    selectionUi?: ParamSelection | null;
    setSelectionUi: (next: ParamSelection | null) => void;
    setCanvasCursor: (next: CanvasCursor) => void;

    strokeRef: MutableRefObject<{
        mode: StrokeMode;
        pointerId: number;
        param: ParamName;
        points: StrokePoint[];
    } | null>;
    panRef: MutableRefObject<{
        pointerId: number;
        startClientX: number;
        startClientY: number;
        startScrollLeft: number;
        startView: ValueViewport;
        startRectH: number;
    } | null>;

    paramView: ParamViewSegment | null;
    paramViewRef: MutableRefObject<ParamViewSegment | null>;

    bumpRefreshToken: () => void;
    syncScrollLeft: (scroller: HTMLDivElement) => void;
    invalidate: () => void;

    yToViewportT: (y: number, h: number) => number;
    yToValue: (param: ParamName, y: number, h: number) => number;
    valueToY: (param: ParamName, v: number, h: number) => number;
    clampViewport: (param: ParamName, v: ValueViewport) => ValueViewport;

    ensureLiveEditBase: (pv: ParamViewSegment) => void;
    applyDenseToLiveEdit: (
        pv: ParamViewSegment,
        denseStartFrame: number,
        dense: number[] | null,
        minF: number,
        maxF: number,
        mode: StrokeMode,
    ) => void;

    /**
     * 参数**写入成功**后的收尾（面板注入）。
     *
     * 语义：保留 live 覆盖层（其值就是刚提交的曲线），让波形在"提交之后取的响度
     * 快照"到达之前继续显示提交值 —— 否则波形的幅度因子会退回旧快照，出现
     * "松手闪回旧波形、再恢复新波形"。同时记录取数序号水位以排除提交前发出的
     * 在飞请求。失败路径仍由各 catch 硬清除覆盖层。
     */
    onParamCommitSucceeded: () => void;

    /**
     * 擦掉上一帧的 live 预览（回到已提交曲线的状态）。
     *
     * 直线 / 颤音工具的预览每帧重算整段，必须先擦掉上一帧写过的点；本入口
     * 只还原**上一帧写过的区间**，不需要重建整份覆盖（见 useLiveParamEditing）。
     */
    resetLiveEditPreview: (pv: ParamViewSegment) => void;

    /**
     * 请求波形面重绘。
     *
     * 与 `invalidate` 的区别：波形是 memo 组件 + 几何缓存，只认投影变化；
     * 而绘制中的 live 覆盖写在 ref 上，不触发 React 渲染，投影也没变。
     * 因此动态曲线的实时预览必须经此入口才能反映到波形上。
     */
    requestWaveformRepaint: () => void;

    commitStroke: (points: StrokePoint[], mode: StrokeMode) => Promise<void>;

    /** 用于选区拖拽 onUp 时同步更新本地 paramView state（与 commitStroke 行为一致） */
    setParamView: (next: ParamViewSegment | null) => void;
    /** 用于选区拖拽 onUp 时清除 live edit overlay */
    liveEditOverrideRef: MutRef<LiveEditOverride | null>;

    /** pointer down 期间设为 true，pointer up 后由 commitStroke 包装层重置为 false。
     *  用于保护 pitch_orig_updated 事件触发的曲线刷新不覆盖正在绘制的内容。 */
    liveEditActiveRef?: MutableRefObject<boolean>;
    /** modifier.pianoRollVerticalZoom 绑定 */
    prVerticalZoomKb: Keybinding;
    /** modifier.horizontalZoom 绑定 */
    horizontalZoomKb: Keybinding;
    /** modifier.scrollHorizontal 绑定 */
    scrollHorizontalKb: Keybinding;
    /** modifier.scrollVertical 绑定 */
    scrollVerticalKb: Keybinding;
    /** modifier.scrollbarZoom 绑定（悬停滚动条 + 滚轮 = 该轴缩放） */
    scrollbarZoomKb: Keybinding;
    /** modifier.paramMorph 绑定 */
    paramMorphKb: Keybinding;
    /** modifier.paramMultiSelect 绑定（按住拖动追加选区段 / 点击取消该段） */
    paramMultiSelectKb: Keybinding;
    /** modifier.paramFineAdjust 绑定 */
    paramFineAdjustKb: Keybinding;
    /** `modifier.paramStretch` 绑定（选择工具下参数选区边缘拉伸） */
    paramStretchKb: Keybinding;
    /** modifier.vibratoAmplitudeAdjust 绑定 */
    vibratoAmplitudeAdjustKb: Keybinding;
    /** modifier.vibratoFrequencyAdjust 绑定 */
    vibratoFrequencyAdjustKb: Keybinding;
    /** 直线/颤音拖拽时增大振幅 */
    vibratoDragAmplitudeIncreaseKb: Keybinding;
    /** 直线/颤音拖拽时减小振幅 */
    vibratoDragAmplitudeDecreaseKb: Keybinding;
    /** 直线/颤音拖拽时增大频率 */
    vibratoDragFrequencyIncreaseKb: Keybinding;
    /** 直线/颤音拖拽时减小频率 */
    vibratoDragFrequencyDecreaseKb: Keybinding;
    /** 右键菜单回调 */
    onContextMenu?: (x: number, y: number) => void;
    /** 播放头位置（秒）读取器，用于以播放头为中心缩放。播放中必须返回与
     *  绘制同源的视觉插值值（rAF 逐帧更新），不能用轮询的 store 滞后值——
     *  以滞后值锚定会让缩放提交时播放头跳变 δ·Δpx。 */
    getPlayheadSec?: () => number;
    /** 是否以播放头为中心缩放 */
    playheadZoomEnabled?: boolean;
    /** 参数编辑器左键按下时是否同步调整播放头 */
    paramEditorSeekPlayheadEnabled?: boolean;
    /** 参数值浮窗是否启用 */
    paramValuePopupEnabled?: boolean;
    /** 参数值浮窗预览回调 */
    onParamValuePreviewChange?: (
        next: {
            clientX: number;
            clientY: number;
            value: number;
            displayText?: string;
        } | null,
    ) => void;
    /** 是否启用绘制时音高吸附 */
    pitchSnapEnabled?: boolean;
    /** 音高吸附方式 */
    pitchSnapUnit?: "semitone" | "scale";
    /** 音高吸附调式（支持内置与自定义） */
    projectScale?: ScaleLike;
    /**
     * Tempo Map 感知的音阶解析：给定绝对秒返回生效音阶。
     * 未提供时退化为全局 projectScale。
     */
    scaleAtSec?: (sec: number) => ScaleLike | undefined;
    /** 音高吸附容差（音分） */
    pitchSnapToleranceCents?: number;
    /** 快捷键映射表 */
    keybindingMap?: KeybindingMap;
    /** 参数编辑操作回调 (op: selectAll, deselect, initialize, ...) */
    onEditAction?: (op: string) => void;
    /** 拖动方向限制 */
    dragDirection?: "free" | "x-only" | "y-only";
    /** 切换拖动方向的回调 */
    onCycleDragDirection?: (tool: "select" | "draw" | "vibrato") => void;
    /** 拖拽期间切换拖动方向的快捷键（触控板用户替代「拖拽中右键」） */
    cycleDragDirectionKb?: Keybinding;
    /**
     * 撤销栈深度读取器（后端权威镜像）。仅「边缘拉伸」手势在提交成功后
     * 用它登记选区步骤（撤销/重做恢复对应选区），其余操作不消费。
     */
    getHistoryPosition?: () => number;
    /** 选区拖拽时边缘平滑度（0-100%） */
    edgeSmoothnessPercent?: number;
    /** 选择拖拽/绘制进行中时，用于临时切换吸附按钮视觉 */
    onPitchSnapGestureActiveChange?: (active: boolean) => void;
    /** 形变控制线变化回调（null 表示隐藏；多选区时每段一条） */
    onMorphOverlayChange?: (next: ParamMorphOverlay[] | null) => void;
    /** 当前参数值域（用于振幅滚轮步进自适应） */
    currentParamRange?: { min: number; max: number };
}) {
    const {
        dispatch,
        rootTrackId,
        editParam,
        pitchEnabled,
        toolMode,
        secPerBeat,
        scrollLeftRef,
        pxPerBeatRef,
        pxPerSecRef,
        getViewportTruth,
        horizontalZoomChainRef,
        onHorizontalZoom,
        syncTimelineEnabled,
        timelineOffsetRef,
        dynamicProjectSec,
        setPitchView,
        setParamViewport,
        pitchViewRef,
        paramViewsRef,
        scrollerRef,
        canvasRef,
        viewSizeRef,
        selectionRef,
        selectionUi,
        setSelectionUi,
        setCanvasCursor,
        strokeRef,
        panRef,
        paramView,
        paramViewRef,
        bumpRefreshToken,
        syncScrollLeft,
        invalidate,
        yToViewportT,
        yToValue,
        valueToY,
        clampViewport,
        ensureLiveEditBase,
        applyDenseToLiveEdit,
        onParamCommitSucceeded,
        resetLiveEditPreview,
        /**
         * 波形面重绘入口（与 `invalidate` 不同：波形是 memo 组件 + 几何缓存，
         * 只认投影变化，而 live 覆盖写在 ref 上不触发 React 渲染）。
         * 绘制中的动态曲线要让它跟着动，必须经此入口。
         */
        requestWaveformRepaint,
        commitStroke,
        setParamView,
        liveEditOverrideRef,
        liveEditActiveRef,
        prVerticalZoomKb,
        horizontalZoomKb,
        scrollHorizontalKb,
        scrollVerticalKb,
        scrollbarZoomKb,
        paramMorphKb,
        paramMultiSelectKb,
        paramFineAdjustKb,
        paramStretchKb,
        vibratoAmplitudeAdjustKb,
        vibratoFrequencyAdjustKb,
        vibratoDragAmplitudeIncreaseKb,
        vibratoDragAmplitudeDecreaseKb,
        vibratoDragFrequencyIncreaseKb,
        vibratoDragFrequencyDecreaseKb,
        getPlayheadSec,
        playheadZoomEnabled,
        paramEditorSeekPlayheadEnabled,
        paramValuePopupEnabled,
        onParamValuePreviewChange,
        onContextMenu,
    } = args;

    const {
        pitchSnapEnabled,
        pitchSnapUnit,
        projectScale,
        scaleAtSec,
        pitchSnapToleranceCents,
        keybindingMap,
        onEditAction,
        dragDirection,
        onCycleDragDirection,
        cycleDragDirectionKb,
        getHistoryPosition,
        edgeSmoothnessPercent,
        onPitchSnapGestureActiveChange,
        onMorphOverlayChange,
        currentParamRange,
    } = args;

    /**
     * 构造当前**交互用**投影（与渲染侧共用同一个视口来源解析器）。
     *
     * 【为什么不能只用 refs（这是一个真实缺陷的根因）】refs 在渲染期被同步成 React
     * state，而横向位置的真值在内核、state 只是 256px 步长的**量化提交**——滚动后
     * refs 会滞后内核最多 255px。命中测试与框选换算若用 refs，就会与（按内核绘制的）
     * 画面相差同一距离，表现为"划定的选区与鼠标划过的区域不一致"。
     *
     * 因此这里直接复用渲染侧那个**已被单测覆盖**的解析器（`resolvePanelRenderViewport`，
     * 内核优先、回落 refs、NaN 安全）。复用而不是再写一遍"取内核否则取 refs"，是为了
     * 让两条路径的取值语义在结构上不可能分叉——这正是本缺陷的教训。
     */
    const axisFromRefs = useCallback(() => {
        const viewport = resolvePanelRenderViewport({
            kernelView: getViewportTruth(),
            refPxPerSec: pxPerSecRef.current,
            refScrollLeftPx: scrollLeftRef.current,
        });
        return createTimelineAxis({
            pxPerSec: viewport.pxPerSec,
            scrollLeftPx: viewport.scrollLeftPx,
        });
    }, [getViewportTruth, pxPerSecRef, scrollLeftRef]);

    /**
     * beat → 视口 x。
     *
     * 此前写作 `beat * pxPerBeat - scrollLeft`（pxPerBeat = pxPerSec * secPerBeat），
     * 与主画布的曲线投影分属两条算式。现在统一为「beat 先转 sec，再走
     * `secToViewportPx`」，与时间线侧和 `render.ts` 同源。
     */
    const beatToViewportPx = useCallback(
        (beat: number) => secToViewportPx(axisFromRefs(), beat * secPerBeat),
        [axisFromRefs, secPerBeat],
    );

    const PARAM_FINE_WHEEL_SCALE = 0.1;

    type FineAdjustedPointerInput = {
        clientX: number;
        clientY: number;
        ctrlKey: boolean;
        shiftKey: boolean;
        altKey: boolean;
        metaKey?: boolean;
        movementX?: number;
        movementY?: number;
    };

    type FineAdjustedPointerState = {
        adjustedClientX: number;
        adjustedClientY: number;
    };

    const disposeFineAdjustedPointerState = useCallback(
        (_state: FineAdjustedPointerState | null | undefined) => {
            // 精细调整不再参与参数编辑器拖拽逻辑；此处保留空实现以复用既有拖拽收尾流程。
            void _state;
        },
        [],
    );

    const pointerFineWheelScale = useCallback(
        (ev: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey?: boolean }) =>
            isModifierActive(paramFineAdjustKb, ev) ? PARAM_FINE_WHEEL_SCALE : 1,
        [paramFineAdjustKb],
    );

    const createFineAdjustedPointerState = useCallback(
        (ev: FineAdjustedPointerInput, _dragTarget: HTMLCanvasElement | null = null) => {
            void _dragTarget;
            return {
                adjustedClientX: ev.clientX,
                adjustedClientY: ev.clientY,
            };
        },
        [],
    );

    const getFineAdjustedPointerPosition = useCallback(
        (state: FineAdjustedPointerState, ev: FineAdjustedPointerInput) => {
            state.adjustedClientX = ev.clientX;
            state.adjustedClientY = ev.clientY;

            return {
                clientX: ev.clientX,
                clientY: ev.clientY,
                fineActive: false,
            };
        },
        [],
    );

    const isSnapToggleModifierHeld = useCallback(
        (ev: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey?: boolean }) => {
            const noSnapKb = keybindingMap?.["modifier.clipNoSnap" as ActionId];
            if (noSnapKb) {
                return Boolean(isModifierActive(noSnapKb, ev));
            }
            return Boolean(ev.shiftKey);
        },
        [keybindingMap],
    );

    const isEffectivePitchSnapActive = useCallback(
        (ev: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey?: boolean }) => {
            const snapToggled = isSnapToggleModifierHeld(ev);
            return Boolean(snapToggled ? !pitchSnapEnabled : pitchSnapEnabled);
        },
        [isSnapToggleModifierHeld, pitchSnapEnabled],
    );

    /**
     * 边缘淡化（delta 空间交叉淡化）的原地包装。
     * halfSpan 由调用方经 edgeHalfSpanForIndices 预先算好（毫秒定标）。
     */
    const blendDenseEdges = useCallback(
        (
            dense: number[],
            base: number[],
            editedStartIdx: number,
            editedLen: number,
            halfSpanFrames: number,
        ) => {
            if (!(halfSpanFrames > 0) || editedLen <= 0) return;
            applyEdgeBlend({
                dense,
                base,
                editedStartIdx,
                editedLen,
                halfSpanFrames,
                isEditable: editParam === "pitch" ? editablePitchValue : undefined,
            });
        },
        [editParam],
    );

    /**
     * 平滑度 → 每侧过渡带半宽（dense 索引）。
     * indicesLen 为 dense 索引数；strideValue 为 dense 索引的采样间距
     * （提交路径恒为 1，预览路径与 pv 的 stride 一致）。
     */
    const edgeHalfSpanForIndices = useCallback(
        (indicesLen: number, strideValue: number) => {
            const pv = paramViewRef.current;
            const fpMs = (pv?.framePeriodMs ?? 5) * Math.max(1, strideValue);
            return edgeHalfSpanFramesForSelection({
                strengthPercent: edgeSmoothnessPercent ?? 0,
                framePeriodMs: fpMs,
                editedLen: indicesLen,
            });
        },
        [edgeSmoothnessPercent, paramViewRef],
    );

    /**
     * 多段提交前的统一取数：按计划的写入窗口逐窗取**全分辨率**基准曲线，
     * 返回可直接喂给 buildMultiRangeEditPlan 的 `sourceAt`。
     *
     * 窗口内未被任何选区段覆盖的帧由基准值填充 —— 与旧单段提交路径
     * （selectionDragRange + 全分辨率取数）语义逐帧一致。
     */
    const fetchCommitBaseSource = useCallback(
        async (input: {
            trackId: string;
            param: ParamName;
            ranges: readonly FrameSpan[];
            frameDelta: number;
            edgeHalfSpanAt: (rangeIndex: number) => number;
            paramView: ParamViewSegment | null;
        }): Promise<(frame: number) => number> => {
            const windows = planSelectionEditWindows({
                ranges: input.ranges,
                frameDelta: input.frameDelta,
                edgeHalfSpanAt: input.edgeHalfSpanAt,
            });
            const curves = await Promise.all(
                windows.map((window) =>
                    fetchFullResCurve({
                        trackId: input.trackId,
                        param: input.param,
                        startFrame: window.startFrame,
                        endFrame: window.endFrame,
                        paramView: input.paramView,
                    }),
                ),
            );
            return (frame: number) => {
                for (let i = 0; i < windows.length; i += 1) {
                    const window = windows[i];
                    if (frame < window.startFrame || frame > window.endFrame) continue;
                    return Number(curves[i]?.values[frame - window.startFrame]) || 0;
                }
                return 0;
            };
        },
        [],
    );

    const morphOverlayRef = useRef<ParamMorphOverlay[] | null>(null);
    const morphDragRef = useRef<{
        pointerId: number;
        /** 命中的控制线在 overlay 数组中的下标（多选区每段一条） */
        overlayIndex: number;
        pointKind: "left" | "mid1" | "mid2" | "right";
    } | null>(null);
    const morphModifierDownRef = useRef(false);
    const vibratoStateRef = useRef<{
        pointerId: number;
        startFrame: number;
        startValue: number;
        currentFrame: number;
        currentValue: number;
        mode: StrokeMode;
        amplitude: number;
        frequency: number;
        shiftHeld: boolean;
    } | null>(null);
    // Track last pointer position so we can synthesize pointermove when modifiers change
    const lastPointerPosRef = useRef<{
        clientX: number;
        clientY: number;
        pointerId?: number;
        buttons?: number;
    }>({
        clientX: 0,
        clientY: 0,
        pointerId: 0,
        buttons: 0,
    });
    const activePointerGestureEndRef = useRef<(() => void) | null>(null);
    /** 最近一次真实 pointermove 的设备类型：合成事件回放时复用。 */
    const lastPointerTypeRef = useRef<string | null>(null);
    const VIBRATO_DRAG_CAPTURE_ATTR = "data-piano-roll-vibrato-drag-active";
    /** 参数线拖拽进行中标记：全局快捷键为「拖动方向切换」键放行（见安装器）。 */
    const PARAM_DRAG_ATTR = "data-piano-roll-param-drag-active";

    const setVibratoDragCaptureActive = useCallback(
        (active: boolean) => {
            if (active) {
                document.body.setAttribute(VIBRATO_DRAG_CAPTURE_ATTR, "true");
            } else {
                document.body.removeAttribute(VIBRATO_DRAG_CAPTURE_ATTR);
            }
        },
        [VIBRATO_DRAG_CAPTURE_ATTR],
    );

    useEffect(() => {
        return () => {
            setVibratoDragCaptureActive(false);
        };
    }, [setVibratoDragCaptureActive]);

    const setActivePointerGestureEnd = useCallback((endGesture: () => void) => {
        activePointerGestureEndRef.current = endGesture;
    }, []);

    const clearActivePointerGestureEnd = useCallback((endGesture?: () => void) => {
        if (!endGesture || activePointerGestureEndRef.current === endGesture) {
            activePointerGestureEndRef.current = null;
        }
    }, []);

    /**
     * 拖拽期间按下「拖动方向」快捷键 → 切换本次拖拽方向（触控板替代右键）。
     *
     * 安装时置位 `PARAM_DRAG_ATTR`：全局 useKeybindings 检测到该属性后对
     * 本键放行（不消费、不派发），由这里唯一处理 —— 与拖拽中右键完全同义
     * （切换本次拖拽方向 + 循环持久化设置）。若不做这层放行，同一按键会先
     * 被全局派发一次（重复步进），并且叠按「精细调整」修饰键时还会撞上
     * Ctrl+D 的克隆轨道兜底。
     *
     * 允许叠按「精细调整」修饰键：拖拽中按下 Ctrl 表示微调，不应屏蔽切换。
     */
    const installDragDirectionKeyCycler = useCallback(
        (cycleLocalDragDir: () => void) => {
            const kb = cycleDragDirectionKb;
            if (!kb || isNoneBinding(kb)) {
                return () => {};
            }
            document.body.setAttribute(PARAM_DRAG_ATTR, "true");
            const onKeyDown = (e: globalThis.KeyboardEvent) => {
                if (e.repeat) return;
                if (!matchesKeybindingAllowingFineModifier(e, kb, paramFineAdjustKb)) return;
                e.preventDefault();
                e.stopPropagation();
                cycleLocalDragDir();
            };
            window.addEventListener("keydown", onKeyDown, true);
            return () => {
                window.removeEventListener("keydown", onKeyDown, true);
                document.body.removeAttribute(PARAM_DRAG_ATTR);
            };
        },
        [PARAM_DRAG_ATTR, cycleDragDirectionKb, paramFineAdjustKb],
    );

    useEffect(() => {
        const endActiveGesture = () => {
            activePointerGestureEndRef.current?.();
        };

        const onVisibilityChange = () => {
            if (document.visibilityState !== "visible") {
                endActiveGesture();
            }
        };

        window.addEventListener("blur", endActiveGesture);
        document.addEventListener("visibilitychange", onVisibilityChange);

        return () => {
            window.removeEventListener("blur", endActiveGesture);
            document.removeEventListener("visibilitychange", onVisibilityChange);
            endActiveGesture();
        };
    }, []);

    const setMorphOverlay = useCallback(
        (next: ParamMorphOverlay[] | null) => {
            morphOverlayRef.current = next;
            onMorphOverlayChange?.(next);
            invalidate();
        },
        [invalidate, onMorphOverlayChange],
    );

    /**
     * 由当前选区构建形变控制线：**每段一条**（每段各自计算基准线与四个控制点），
     * 断层不参与形变。无有效段时返回 null（null 而非空数组，调用方以
     * `!morphOverlayRef.current` 判断"尚未构建"）。
     */
    const buildMorphOverlaysFromSelection = useCallback((): ParamMorphOverlay[] | null => {
        const sel = selectionRef.current;
        const pv = paramViewRef.current;
        if (!sel || sel.length === 0 || !pv || pv.edit.length === 0) return null;

        const stride = Math.max(1, pv.stride);
        const overlays: ParamMorphOverlay[] = [];

        for (const range of sel) {
            const aBeat = range.startBeat;
            const bBeat = range.endBeat;
            if (!Number.isFinite(aBeat) || !Number.isFinite(bBeat) || bBeat <= aBeat) continue;

            // 选区 → 帧 → 采样下标：抽到 `kernel/dragArithmetic`（纯函数，有单测）。
            // 这段算术在 hook 里原有 3 份副本，任一处漂移都会让拉伸预览与提交错位。
            const selRange = selectionIndexRange({
                aBeat,
                bBeat,
                secPerBeat,
                framePeriodMs: pv.framePeriodMs,
                paramView: pv,
            });
            if (selRange === null) continue;
            const { startIdx: selStartIdx, endIdx: selEndIdx } = selRange;
            const baselineValues = pv.edit.slice(selStartIdx, selEndIdx + 1);
            if (baselineValues.length === 0) continue;

            const valid =
                editParam === "pitch"
                    ? baselineValues.filter((v) => Number(v) !== 0)
                    : baselineValues;
            // dyn：控制线的 pivot 用**几何均值**（倍率域的中心趋势），应用时
            // 按"控制值 / pivot"的比值缩放基线 —— 算术均值会把静音帧抬离 0。
            const meanValue = isDynParam(editParam)
                ? computeDynGeometricMean(baselineValues)
                : valid.length > 0
                  ? valid.reduce((sum, v) => sum + (Number(v) || 0), 0) / valid.length
                  : 0;

            const selectionStartFrame = pv.startFrame + selStartIdx * stride;
            const selectionEndFrame = pv.startFrame + selEndIdx * stride;
            const span = Math.max(0, selectionEndFrame - selectionStartFrame);
            const p1 = Math.round(selectionStartFrame + span / 3);
            const p2 = Math.round(selectionStartFrame + (span * 2) / 3);

            overlays.push({
                selectionStartFrame,
                selectionEndFrame,
                meanValue,
                baselineValues,
                points: [
                    { kind: "left", frame: selectionStartFrame, value: meanValue },
                    { kind: "mid1", frame: p1, value: meanValue },
                    { kind: "mid2", frame: p2, value: meanValue },
                    { kind: "right", frame: selectionEndFrame, value: meanValue },
                ],
            });
        }
        return overlays.length > 0 ? overlays : null;
    }, [editParam, paramViewRef, secPerBeat, selectionRef]);

    const buildMorphDense = useCallback(
        (overlay: ParamMorphOverlay, stride: number) => {
            const step = Math.max(1, stride);
            const startFrame = overlay.selectionStartFrame;
            const endFrame = overlay.selectionEndFrame;
            const len = Math.max(1, Math.floor((endFrame - startFrame) / step) + 1);
            const dense = new Array<number>(len);
            const ordered = overlay.points.slice().sort((a, b) => a.frame - b.frame);

            // 使用反距离加权（IDW, p=2）插值：四个控制点对选区内每个参数点均有
            // 基于 X 轴距离的加权影响，越近权重越大，所有点都有贡献。
            const curveValueAt = (frame: number): number => {
                let totalWeight = 0;
                let weightedValue = 0;
                for (const p of ordered) {
                    const dist = Math.max(1, Math.abs(frame - p.frame));
                    const w = 1 / (dist * dist); // inverse square distance
                    totalWeight += w;
                    weightedValue += w * p.value;
                }
                return totalWeight > 1e-12 ? weightedValue / totalWeight : overlay.meanValue;
            };

            for (let i = 0; i < len; i += 1) {
                const frame = startFrame + i * step;
                const base = Number(overlay.baselineValues[i] ?? 0);
                if (editParam === "pitch" && base === 0) {
                    dense[i] = 0;
                    continue;
                }
                // dyn：乘性应用（base × 控制比值）—— 静音帧（base = 0）保持 0；
                // 控制线在 pivot（几何均值）处 = 不改变。pivot 无效时退恒等。
                if (isDynParam(editParam)) {
                    dense[i] =
                        overlay.meanValue > 0
                            ? base * (curveValueAt(frame) / overlay.meanValue)
                            : base;
                    continue;
                }
                const delta = curveValueAt(frame) - overlay.meanValue;
                dense[i] = base + delta;
            }
            return { startFrame, endFrame, dense };
        },
        [editParam],
    );

    const applyMorphOverlayPreview = useCallback(
        (overlays: ParamMorphOverlay[]) => {
            const pv = paramViewRef.current;
            if (!pv) return;
            ensureLiveEditBase(pv);
            for (const overlay of overlays) {
                const packed = buildMorphDense(overlay, pv.stride);
                applyDenseToLiveEdit(
                    pv,
                    packed.startFrame,
                    packed.dense,
                    packed.startFrame,
                    packed.endFrame,
                    "draw",
                );
            }
            // live 覆盖只改 ref：波形面不会自行重绘，显式请求一次。
            requestWaveformRepaint();
        },
        [
            applyDenseToLiveEdit,
            buildMorphDense,
            ensureLiveEditBase,
            paramViewRef,
            requestWaveformRepaint,
        ],
    );

    /**
     * 手绘结束后的自动平滑：单次高斯核（σ = 强度% × 40ms，毫秒定标）。
     * 只作用于绘制范围本身（旧实现会外溢 1% 并把绘制曲线整体替换为
     * 101 帧窗 ×3 遍的 box 均值 —— 手势细节被抹平）；trend 延拓消除端点
     * 内拉；pitch=0 哨兵帧不参与也不被改写；长未浊缺口不跨段桥接。
     */
    const applyPostStrokeSmoothing = useCallback(
        async (points: StrokePoint[], mode: StrokeMode) => {
            if (mode !== "draw") return;
            const trackId = rootTrackId;
            if (!trackId || points.length === 0) return;

            const strengthPercent = clamp(Number(edgeSmoothnessPercent) || 0, 0, 100);
            const sigmaMs = drawSmoothSigmaMsFromStrength(strengthPercent);
            if (!(sigmaMs > 0)) return;

            let minF = Number.POSITIVE_INFINITY;
            let maxF = 0;
            for (const p of points) {
                const f = Math.max(0, Math.floor(Number(p.frame) || 0));
                minF = Math.min(minF, f);
                maxF = Math.max(maxF, f);
            }
            if (!Number.isFinite(minF) || maxF < minF) return;

            const smoothCount = maxF - minF + 1;
            const res = await paramsApi.getParamFrames(trackId, editParam, minF, smoothCount, 1);
            if (!res?.ok) return;
            const payload = res as ParamFramesPayload;
            const vals = (payload.edit ?? []).map((v) => Number(v) || 0);
            if (vals.length === 0) return;
            const fpMs = Number(payload.frame_period_ms) || 5;

            const smoothed = smoothCurveGaussian(vals, {
                sigmaMs,
                framePeriodMs: fpMs,
                valueFilter: editParam === "pitch" ? editablePitchValue : undefined,
            });

            try {
                await paramsApi.setParamFrames(trackId, editParam, minF, smoothed, false);
            } catch (err) {
                // 绘制后的自动平滑是增强步骤：失败只跳过平滑与视图同步，
                // 不产生 unhandledrejection（主提交 commitStroke 已自兜底）。
                console.error("[applyPostStrokeSmoothing] failed:", err);
                return;
            }

            const pvNow = paramViewRef.current;
            if (pvNow) {
                const nextEdit = pvNow.edit.slice();
                const stride = Math.max(1, pvNow.stride);
                for (let i = 0; i < smoothed.length; i += 1) {
                    const frame = minF + i;
                    const idx = Math.round((frame - pvNow.startFrame) / stride);
                    if (idx >= 0 && idx < nextEdit.length) {
                        nextEdit[idx] = smoothed[i];
                    }
                }
                setParamView({ ...pvNow, edit: nextEdit });
            }
            bumpRefreshToken();
        },
        [
            bumpRefreshToken,
            edgeSmoothnessPercent,
            editParam,
            paramViewRef,
            rootTrackId,
            setParamView,
        ],
    );

    /** Apply pitch snap to a drawn value when editParam is "pitch" and snap is enabled.
     *  When snapToggleHeld=true, the snap state is toggled (XOR with pitchSnapEnabled). */
    const snapDrawValue = useCallback(
        (v: number, snapToggleHeld = false, frame?: number): number => {
            const effective = snapToggleHeld ? !pitchSnapEnabled : pitchSnapEnabled;
            if (!effective) return v;

            if (
                isChildPitchOffsetCentsParam(editParam) ||
                isChildPitchOffsetDegreesParam(editParam)
            ) {
                return snapChildPitchOffsetValue(editParam, v);
            }

            if (editParam !== "pitch") return v;
            // Tempo Map 感知：优先按帧时刻解析生效音阶。
            const framePeriodMs = paramViewRef.current?.framePeriodMs;
            const effectiveScale =
                pitchSnapUnit === "scale"
                    ? frame != null && framePeriodMs != null
                        ? (scaleAtSec?.((frame * framePeriodMs) / 1000) ?? projectScale)
                        : projectScale
                    : undefined;
            const snapped =
                pitchSnapUnit === "scale" && effectiveScale
                    ? snapToScale(v, effectiveScale)
                    : snapToSemitone(v);
            const toleranceSemitone = Math.max(0, Number(pitchSnapToleranceCents ?? 0) / 100);
            if (Math.abs(v - snapped) <= toleranceSemitone) {
                return v;
            }
            return snapped + (v - snapped > 0 ? 1 : -1) * toleranceSemitone;
        },
        [
            pitchSnapEnabled,
            pitchSnapUnit,
            projectScale,
            scaleAtSec,
            pitchSnapToleranceCents,
            editParam,
            paramViewRef,
        ],
    );

    const updateSelectionUi = useCallback(
        (next: ParamSelection | null) => {
            setSelectionUi(next);
            if (morphModifierDownRef.current && !morphDragRef.current) {
                setMorphOverlay(buildMorphOverlaysFromSelection());
            }
        },
        [buildMorphOverlaysFromSelection, setMorphOverlay, setSelectionUi],
    );

    const buildVibratoDense = useCallback(
        (
            startFrame: number,
            startValue: number,
            endFrame: number,
            endValue: number,
            amplitude: number,
            frequency: number,
            shiftHeld: boolean,
        ) => {
            const minF = Math.min(startFrame, endFrame);
            const maxF = Math.max(startFrame, endFrame);
            const len = maxF - minF + 1;
            const dense = new Array<number>(len);
            const denom = endFrame - startFrame;
            const safeFreq = Math.max(1e-4, Number.isFinite(frequency) ? frequency : 1);
            for (let f = minF; f <= maxF; f += 1) {
                const t = denom === 0 ? 1 : (f - startFrame) / denom;
                const base = startValue + (endValue - startValue) * t;
                const wave = amplitude * Math.sin(2 * Math.PI * safeFreq * t);
                dense[f - minF] = snapDrawValue(base + wave, shiftHeld, f);
            }
            return { minF, maxF, dense };
        },
        [snapDrawValue],
    );

    const applyVibratoDragAdjustment = useCallback(
        (input: {
            target: "amplitude" | "frequency";
            direction: 1 | -1;
            steps: number;
            shiftHeld: boolean;
            fineEvent: {
                ctrlKey: boolean;
                shiftKey: boolean;
                altKey: boolean;
                metaKey?: boolean;
            };
        }): boolean => {
            const vib = vibratoStateRef.current;
            if (!vib) return false;

            const fineScale = pointerFineWheelScale(input.fineEvent);
            const next = computeVibratoDragAdjustment({
                editParam,
                currentParamRange,
                amplitude: vib.amplitude,
                frequency: vib.frequency,
                target: input.target,
                direction: input.direction,
                steps: input.steps,
                fineScale,
            });
            vib.amplitude = next.amplitude;
            vib.frequency = next.frequency;

            const st = strokeRef.current;
            const pvNow = paramViewRef.current;
            if (st && pvNow && st.pointerId === vib.pointerId) {
                // 参数变更后重画整段预览：先擦掉**上一帧写过的区间**（而不是丢弃
                // 整份覆盖层再整份拷贝 —— 那是每帧一次 O(窗口) 的拷贝），再按新
                // 参数写入。`applyDenseToLiveEdit` 内部会按需重建基准。
                resetLiveEditPreview(pvNow);
                const built = buildVibratoDense(
                    vib.startFrame,
                    vib.startValue,
                    vib.currentFrame,
                    vib.currentValue,
                    vib.amplitude,
                    vib.frequency,
                    input.shiftHeld,
                );
                vib.shiftHeld = input.shiftHeld;
                st.points = [
                    { frame: vib.startFrame, value: vib.startValue },
                    { frame: vib.currentFrame, value: vib.currentValue },
                ];
                applyDenseToLiveEdit(
                    pvNow,
                    built.minF,
                    st.mode === "restore" ? null : built.dense,
                    built.minF,
                    built.maxF,
                    st.mode,
                );
                invalidate();
                // 同上：live 覆盖只改 ref，波形面需显式重绘。
                requestWaveformRepaint();
            }

            return true;
        },
        [
            pointerFineWheelScale,
            editParam,
            currentParamRange,
            strokeRef,
            paramViewRef,
            buildVibratoDense,
            applyDenseToLiveEdit,
            resetLiveEditPreview,
            invalidate,
            requestWaveformRepaint,
        ],
    );

    useEffect(() => {
        if (toolMode !== "select") {
            morphModifierDownRef.current = false;
            if (!morphDragRef.current) {
                setMorphOverlay(null);
            }
        }

        const updateMorphActivation = (
            e:
                | globalThis.KeyboardEvent
                | { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey?: boolean },
        ) => {
            const active =
                toolMode === "select" &&
                !panRef.current &&
                !strokeRef.current &&
                !morphDragRef.current &&
                isModifierActive(paramMorphKb, e);
            morphModifierDownRef.current = active;

            if (!active) {
                if (!morphDragRef.current) {
                    setMorphOverlay(null);
                    if (!strokeRef.current && !panRef.current && !liveEditActiveRef?.current) {
                        liveEditOverrideRef.current = null;
                        if (liveEditActiveRef) liveEditActiveRef.current = false;
                    }
                }
                return;
            }

            if (!morphOverlayRef.current) {
                setMorphOverlay(buildMorphOverlaysFromSelection());
            }
        };

        const onKey = (e: globalThis.KeyboardEvent) => {
            updateMorphActivation(e);
        };
        const onBlur = () => {
            morphModifierDownRef.current = false;
            if (!morphDragRef.current) {
                setMorphOverlay(null);
                if (!strokeRef.current && !panRef.current && !liveEditActiveRef?.current) {
                    liveEditOverrideRef.current = null;
                    if (liveEditActiveRef) liveEditActiveRef.current = false;
                }
            }
        };

        window.addEventListener("keydown", onKey);
        window.addEventListener("keyup", onKey);
        window.addEventListener("blur", onBlur);

        return () => {
            window.removeEventListener("keydown", onKey);
            window.removeEventListener("keyup", onKey);
            window.removeEventListener("blur", onBlur);
        };
    }, [
        buildMorphOverlaysFromSelection,
        liveEditActiveRef,
        liveEditOverrideRef,
        panRef,
        paramMorphKb,
        setMorphOverlay,
        strokeRef,
        toolMode,
    ]);

    useEffect(() => {
        if (!selectionUi) return;
        if (toolMode !== "select") return;
        if (!morphModifierDownRef.current || morphDragRef.current) return;
        setMorphOverlay(buildMorphOverlaysFromSelection());
    }, [buildMorphOverlaysFromSelection, selectionUi, setMorphOverlay, toolMode]);

    // Track last pointer position and synthesize pointermove on key changes
    useEffect(() => {
        const updatePos = (ev: globalThis.PointerEvent) => {
            lastPointerPosRef.current = {
                clientX: ev.clientX,
                clientY: ev.clientY,
                pointerId: ev.pointerId,
                buttons: ev.buttons,
            };
            // 合成事件必须携带与真实设备一致的 pointerType：否则 pen 拖拽中
            // 改修饰键时，重放的 pointermove 会被 pen 判定逻辑当成 mouse。
            lastPointerTypeRef.current = ev.pointerType || null;
        };

        const onKeyMod = (e: globalThis.KeyboardEvent) => {
            const st = strokeRef.current;
            const hasActiveStroke = Boolean(st);
            const hasActiveLiveDrag = Boolean(liveEditActiveRef?.current);
            if (!hasActiveStroke && !hasActiveLiveDrag) return;

            const last = lastPointerPosRef.current;
            if (!last) {
                invalidate();
                return;
            }

            try {
                const pe = new PointerEvent("pointermove", {
                    clientX: last.clientX,
                    clientY: last.clientY,
                    pointerId: st?.pointerId ?? last.pointerId ?? 1,
                    pointerType: lastPointerTypeRef.current ?? "mouse",
                    buttons: last.buttons ?? 1,
                    bubbles: true,
                    cancelable: true,
                    composed: true,
                    ctrlKey: e.ctrlKey,
                    shiftKey: e.shiftKey,
                    altKey: e.altKey,
                    metaKey: e.metaKey,
                } as PointerEventInit);
                window.dispatchEvent(pe);
            } catch {
                // Fallback: force redraw
                invalidate();
            }
        };

        window.addEventListener("pointermove", updatePos, { passive: true });
        window.addEventListener("keydown", onKeyMod);
        window.addEventListener("keyup", onKeyMod);

        return () => {
            window.removeEventListener("pointermove", updatePos);
            window.removeEventListener("keydown", onKeyMod);
            window.removeEventListener("keyup", onKeyMod);
        };
    }, [strokeRef, liveEditActiveRef, invalidate]);

    const pointerBeat = useCallback(
        (clientX: number): number => {
            const canvas = canvasRef.current;
            if (!canvas) return 0;
            const rect = canvas.getBoundingClientRect();
            // 视口 x → sec → beat：逆投影走 axis，与 pointerSec 同源。
            // 此前写作 `(scrollLeft + x) / pxPerBeat`，是又一套独立算式。
            return viewportPxToSec(axisFromRefs(), clientX - rect.left) / secPerBeat;
        },
        [canvasRef, axisFromRefs, secPerBeat],
    );

    const pointerSec = useCallback(
        (clientX: number): number => {
            const canvas = canvasRef.current;
            if (!canvas) return 0;
            const rect = canvas.getBoundingClientRect();
            return secFromViewportClientX({
                clientX,
                viewportLeft: rect.left,
                // 与 `pointerBeat` 同源：一律走 `axisFromRefs`（内核真值的投影），
                // 不再就地用可能滞后的 refs 另建一份轴——那会让"同一指针位置"在两条
                // 路径上得到不同的秒数（量化残差），是框选偏移的同类根因。
                axis: axisFromRefs(),
            });
        },
        [canvasRef, axisFromRefs],
    );

    const pointerValue = useCallback(
        (clientY: number): number => {
            const canvas = canvasRef.current;
            if (!canvas) return 0;
            const rect = canvas.getBoundingClientRect();
            const y = clientY - rect.top;
            const raw = yToValue(editParam, y, rect.height);
            // render.ts 绘制 pitch 曲线时对值加了 +0.5（使曲线居于琴键中心），
            // 此处减去相同偏移，确保编辑点与显示位置对齐。
            return editParam === "pitch" ? raw - 0.5 : raw;
        },
        [canvasRef, editParam, yToValue],
    );

    const onRulerMouseDown = useCallback(
        (e: ReactMouseEvent<HTMLDivElement>) => {
            // 数位笔 / 触摸不触发标尺 seek：pen 的兼容 mouse 事件（无
            // pointerType）会让"悬停画线起笔"误定位播放头；悬停时间气泡
            // （只读）不受影响。按"最近一次真实 pointer 事件"的设备类型判定。
            if (isLegacyMouseEventFromStylus()) return;
            if (e.button !== 0) return;
            const ruler = e.currentTarget as HTMLDivElement;
            let moved = false;

            const updateAt = (clientX: number, commit: boolean): number => {
                const bounds = ruler.getBoundingClientRect();
                const sec = clamp(
                    secFromViewportClientX({
                        clientX,
                        viewportLeft: bounds.left,
                        // 同上：seek 的投影也必须取内核真值，否则滚动后点标尺会落到
                        // 与画面播放头不同的位置（量化残差）。
                        axis: axisFromRefs(),
                    }),
                    0,
                    1e12,
                );

                if (commit) {
                    dispatch(setplayheadSec(sec));
                    void dispatch(seekPlayhead(sec));
                }
                return sec;
            };

            // 标尺没有其他编辑操作需要区分，按下时立即提交一次 seek。
            let lastSec = updateAt(e.clientX, true);

            // 拖动中的 setplayheadSec 逐事件派发会让所有订阅 session 的组件
            // 随 mousemove 频率重渲；rAF 合帧后与画布重绘节奏一致。
            let rafId: number | null = null;
            let queuedSec: number | null = null;

            const onMove = (ev: MouseEvent) => {
                moved = true;
                queuedSec = updateAt(ev.clientX, false);
                if (rafId == null) {
                    rafId = requestAnimationFrame(() => {
                        rafId = null;
                        if (queuedSec == null) return;
                        dispatch(setplayheadSec(queuedSec));
                        queuedSec = null;
                    });
                }
            };

            const onEnd = (ev: MouseEvent) => {
                window.removeEventListener("mousemove", onMove, true);
                window.removeEventListener("mouseup", onEnd, true);
                window.removeEventListener("mouseleave", onEnd, true);
                if (rafId != null) {
                    cancelAnimationFrame(rafId);
                    rafId = null;
                }
                if (!moved) return;
                lastSec = updateAt(ev.clientX, true);
                void dispatch(seekPlayhead(lastSec));
            };

            window.addEventListener("mousemove", onMove, true);
            window.addEventListener("mouseup", onEnd, true);
            window.addEventListener("mouseleave", onEnd, true);
        },
        [dispatch, scrollLeftRef, pxPerSecRef],
    );

    const onScrollerAuxClick = useCallback((e: ReactMouseEvent) => {
        if (e.button === 1) e.preventDefault();
    }, []);

    /**
     * 原生滚动容器的 `scroll` 事件。
     *
     * 【已移出本 hook】本处原先无条件忽略该事件（当时的结论是"每个输入都有显式
     * 入口，事件永远只是镜像回声"）。该结论对**显式入口**成立，但原生容器仍是
     * `overflow: scroll`：触摸拖拽 / 触控板惯性 / 焦点滚入视口只会产生原生
     * `scroll`，在无条件忽略下这些输入完全失效。
     *
     * 现在由面板侧 `PianoRollPanel.onScrollerScroll` 处理，判据是"原生位置与内核
     * 当前持有的位置是否一致"（一致 = 回声，忽略；不一致 = 容器自己动了，采纳），
     * 既消掉"阶梯感"的回退，又不丢那三类输入。该处保留了完整的根因说明。
     */
    const onScrollerScroll = useCallback((e: UIEvent<HTMLDivElement>) => {
        void e;
    }, []);

    const onScrollerContextMenu = useCallback(
        (e: ReactMouseEvent) => {
            e.preventDefault();
            if (onContextMenu) {
                onContextMenu(e.clientX, e.clientY);
            }
        },
        [onContextMenu],
    );

    const onScrollerKeyDown = useCallback(
        (e: KeyboardEvent<HTMLDivElement>) => {
            const key = e.key.toLowerCase();
            if (
                key === "arrowup" ||
                key === "arrowdown" ||
                key === "arrowleft" ||
                key === "arrowright"
            ) {
                e.preventDefault();
            }

            if (!rootTrackId) return;
            if (editParam === "pitch" && !pitchEnabled) return;

            // pianoRoll.shiftParamUp / shiftParamDown 已移至全局 handleKeybindingAction 处理

            // Handle edit.* keybindings passed through from useKeybindings global handler
            // (must be before the selectionRef guard since selectAll/deselect work without selection)
            if (keybindingMap && onEditAction) {
                const editActionEntries = (
                    Object.entries(keybindingMap) as [ActionId, Keybinding][]
                ).filter(([id]) => id.startsWith("edit."));
                // 需要弹出对话框的操作列表
                const dialogOps = new Set([
                    "transposeCents",
                    "transposeDegrees",
                    "setPitch",
                    "average",
                    "smooth",
                    "addVibrato",
                    "quantize",
                    "meanQuantize",
                ]);
                for (const [actionId, kb] of editActionEntries) {
                    if (kb.modifierOnly) continue;
                    if (matchesKeybinding(e.nativeEvent, kb)) {
                        const meta = ACTION_META[actionId];
                        // paramEditorSelect-scoped actions only work with "select" tool
                        if (meta?.scopedContext === "paramEditorSelect" && toolMode !== "select") {
                            continue;
                        }
                        const op = actionId.replace("edit.", "");
                        // undo/redo handled globally, skip here
                        if (op === "undo" || op === "redo") continue;
                        e.preventDefault();
                        // 需要弹窗的操作 → 派发 openEditDialog 事件打开对话框
                        if (dialogOps.has(op)) {
                            window.dispatchEvent(
                                new CustomEvent("hifi:openEditDialog", { detail: { dialog: op } }),
                            );
                        } else {
                            onEditAction(op);
                        }
                        return;
                    }
                }
            }

            // 复制/粘贴（pianoRoll.copy / pianoRoll.paste）已上移：由全局路由
            // （useKeybindings → focusRouting.resolveEditOpRoute）按活动编辑
            // 表面定向派发到 PianoRollPanel.handleEditOp 唯一执行，本地不再
            // 维护平行的键位匹配分支（旧分支与 handleEditOp 重复且行为已
            // 分叉，是复制/粘贴冲突的放大器）。
        },
        [rootTrackId, editParam, pitchEnabled, keybindingMap, onEditAction, toolMode],
    );

    const onScrollerWheelNative = useCallback(
        (e: globalThis.WheelEvent) => {
            const el = scrollerRef.current;
            if (!el) return;

            const vib = vibratoStateRef.current;
            if (vib) {
                const ampRequested =
                    isNoneBinding(vibratoAmplitudeAdjustKb) ||
                    isModifierActive(vibratoAmplitudeAdjustKb, e);
                const freqRequested =
                    isNoneBinding(vibratoFrequencyAdjustKb) ||
                    isModifierActive(vibratoFrequencyAdjustKb, e);
                const wheelTarget = getVibratoDragWheelTarget({
                    deltaX: e.deltaX,
                    deltaY: e.deltaY,
                    deltaMode: e.deltaMode,
                    amplitudeRequested: ampRequested,
                    frequencyRequested: freqRequested,
                });

                // During vibrato drag, wheel always adjusts vibrato (amplitude by default,
                // frequency via modifier or horizontal scroll). If no modifier is held and
                // neither binding is "None", fall back to amplitude adjustment so the wheel
                // never gets blocked or interpreted as zoom/scroll.
                const resolvedTarget = wheelTarget !== "none" ? wheelTarget : "amplitude";

                e.preventDefault();
                const controlDelta =
                    resolvedTarget === "frequency" && Math.abs(e.deltaX) > Math.abs(e.deltaY)
                        ? e.deltaX
                        : e.deltaY;
                const steps = Math.max(1, Math.round(Math.abs(controlDelta) / 100));
                const direction =
                    resolvedTarget === "amplitude"
                        ? controlDelta < 0
                            ? 1
                            : -1
                        : controlDelta < 0
                          ? -1
                          : 1;
                applyVibratoDragAdjustment({
                    target: resolvedTarget,
                    direction,
                    steps,
                    shiftHeld: e.shiftKey,
                    fineEvent: e,
                });
                return;
            }

            const noModifierPressed = !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey;
            const isWheelBindingRequested = (kb: Keybinding) => {
                if (isNoneBinding(kb)) return noModifierPressed;
                return isModifierActive(kb, e);
            };
            // ── 悬停原生滚动条：滚轮语义只归属该滚动条的轴 ──────────────
            // 无修饰键 = 该轴滚动（竖直条 → 视口纵向平移；水平条 → 横向滚动）；
            // 按住 modifier.scrollbarZoom（默认 Alt）= 该轴缩放。优先于全局绑定。
            const scrollbarZone = nativeScrollbarZoneAt(el, e.clientX, e.clientY);
            const scrollbarZoomRequested =
                scrollbarZone != null &&
                !isNoneBinding(scrollbarZoomKb) &&
                isModifierActive(scrollbarZoomKb, e);
            const horizontalScrollModifierActive = isWheelBindingRequested(scrollHorizontalKb);
            const wheelAction = getParamEditorWheelAction({
                deltaX: e.deltaX,
                deltaY: e.deltaY,
                horizontalScrollRequested: horizontalScrollModifierActive,
                verticalPanRequested: isWheelBindingRequested(scrollVerticalKb),
                verticalZoomRequested: isWheelBindingRequested(prVerticalZoomKb),
                horizontalZoomRequested: isWheelBindingRequested(horizontalZoomKb),
                scrollbarZone,
                scrollbarZoomRequested,
            });

            const applyVerticalPanDelta = (deltaY: number) => {
                const h = Math.max(1, el.clientHeight);
                const delta = (-deltaY / h) * 0.5;
                if (editParam === "pitch") {
                    const cur = pitchViewRef.current;
                    const next = clampViewport("pitch", {
                        span: cur.span,
                        center: cur.center + delta * cur.span,
                    });
                    setPitchView(next);
                } else {
                    const fallbackRange = currentParamRange ?? { min: 0, max: 1 };
                    const cur = paramViewsRef.current[editParam] ?? {
                        center: (fallbackRange.min + fallbackRange.max) / 2,
                        span: Math.max(1e-6, fallbackRange.max - fallbackRange.min),
                    };
                    const next = clampViewport(editParam, {
                        span: cur.span,
                        center: cur.center + delta * cur.span,
                    });
                    setParamViewport(editParam, next);
                }
                invalidate();
            };

            const horizontalDelta = Math.abs(e.deltaX) > 0.5 ? e.deltaX : e.deltaY;

            if (wheelAction === "free-scroll") {
                e.preventDefault();
                el.scrollLeft += e.deltaX;
                syncScrollLeft(el);
                applyVerticalPanDelta(e.deltaY);
                return;
            }

            // Scroll modifier: convert wheel to horizontal scroll
            if (wheelAction === "horizontal-scroll") {
                e.preventDefault();
                el.scrollLeft += horizontalDelta;
                syncScrollLeft(el);
                return;
            }

            // Scroll modifier: convert wheel to vertical scroll
            if (wheelAction === "vertical-pan") {
                e.preventDefault();
                applyVerticalPanDelta(e.deltaY);
                return;
            }

            // Anchor zoom to the actual drawable viewport (canvas), not the scroller.
            // The scroller may include rulers/padding, which makes zoom feel off-center.
            const canvas = canvasRef.current;
            const bounds = (canvas ?? el).getBoundingClientRect();

            const pointerXRaw = e.clientX - bounds.left;
            const pointerYRaw = e.clientY - bounds.top;

            // We rely on preventDefault to stop native scrolling while zooming.
            e.preventDefault();

            if (wheelAction === "vertical-zoom") {
                if (
                    pointerXRaw < 0 ||
                    pointerYRaw < 0 ||
                    pointerXRaw > bounds.width ||
                    pointerYRaw > bounds.height
                ) {
                    return;
                }
                const h = Math.max(1, bounds.height);
                const y = clamp(pointerYRaw, 0, h);
                const t = yToViewportT(y, h);
                const valueAtPointer = yToValue(editParam, y, h);

                const factor = e.deltaY < 0 ? 0.9 : 1.1;
                if (editParam === "pitch") {
                    const cur = pitchViewRef.current;
                    const nextSpan = cur.span * factor;
                    const next = clampViewport("pitch", {
                        span: nextSpan,
                        center: valueAtPointer - (0.5 - t) * nextSpan,
                    });
                    setPitchView(next);
                } else {
                    const fallbackRange = currentParamRange ?? { min: 0, max: 1 };
                    const cur = paramViewsRef.current[editParam] ?? {
                        center: (fallbackRange.min + fallbackRange.max) / 2,
                        span: Math.max(1e-6, fallbackRange.max - fallbackRange.min),
                    };
                    const nextSpan = cur.span * factor;
                    const next = clampViewport(editParam, {
                        span: nextSpan,
                        center: valueAtPointer - (0.5 - t) * nextSpan,
                    });
                    setParamViewport(editParam, next);
                }
                invalidate();
                return;
            }

            // Wheel: horizontal zoom (time axis)
            if (wheelAction !== "horizontal-zoom") {
                return;
            }
            const dir = e.deltaY < 0 ? 1 : -1;
            const factor = dir > 0 ? 1.1 : 0.9;
            const totalSec = Math.max(0, dynamicProjectSec);
            const minPxPerSec = resolveTimelineMinPxPerSec({
                baseMinPxPerSec: MIN_PX_PER_SEC,
                projectSec: totalSec,
                viewportWidthPx: el.clientWidth,
            });
            // 与轨道视图共用同一套水平缩放逻辑（秒为单位）：
            // 鼠标锚点 / 播放光标锚点 / 平滑右延滚动范围全部一致。
            const pendingZoom = horizontalZoomChainRef.current;
            const zoomResult = resolveHorizontalWheelZoom({
                factor,
                basePxPerSec: pendingZoom?.nextPxPerSec ?? pxPerSecRef.current,
                baseScrollLeft: pendingZoom?.nextScrollLeft ?? scrollLeftRef.current,
                totalSec,
                viewportWidth: el.clientWidth,
                playheadZoomEnabled: Boolean(playheadZoomEnabled),
                playheadSec: getPlayheadSec?.() ?? null,
                anchorScreenX: pointerXRaw,
                minPxPerSec,
                maxPxPerSec: MAX_PX_PER_SEC,
                minScrollLeft: syncTimelineEnabled ? -timelineOffsetRef.current : 0,
                anchorOffsetPx: syncTimelineEnabled ? timelineOffsetRef.current : 0,
            });
            if (!zoomResult) return;

            horizontalZoomChainRef.current = {
                nextPxPerSec: zoomResult.nextPxPerSec,
                nextScrollLeft: zoomResult.nextScrollLeft,
            };
            onHorizontalZoom(zoomResult.nextPxPerSec, zoomResult.nextScrollLeft);
        },
        [
            scrollerRef,
            canvasRef,
            editParam,
            yToViewportT,
            yToValue,
            pitchViewRef,
            paramViewsRef,
            clampViewport,
            setPitchView,
            setParamViewport,
            invalidate,
            syncScrollLeft,
            horizontalZoomChainRef,
            onHorizontalZoom,
            syncTimelineEnabled,
            timelineOffsetRef,
            prVerticalZoomKb,
            scrollHorizontalKb,
            scrollVerticalKb,
            scrollbarZoomKb,
            horizontalZoomKb,
            vibratoAmplitudeAdjustKb,
            vibratoFrequencyAdjustKb,
            applyVibratoDragAdjustment,
            dynamicProjectSec,
            getPlayheadSec,
            playheadZoomEnabled,
            currentParamRange,
            pxPerSecRef,
            scrollLeftRef,
        ],
    );

    useEffect(() => {
        const onKeyDown = (e: globalThis.KeyboardEvent) => {
            if (!vibratoStateRef.current) return;

            const adjustment = resolveVibratoDragKeyboardAdjustment(
                e,
                {
                    amplitudeIncrease: vibratoDragAmplitudeIncreaseKb,
                    amplitudeDecrease: vibratoDragAmplitudeDecreaseKb,
                    frequencyIncrease: vibratoDragFrequencyIncreaseKb,
                    frequencyDecrease: vibratoDragFrequencyDecreaseKb,
                },
                paramFineAdjustKb,
            );
            if (!adjustment) return;

            e.preventDefault();
            e.stopPropagation();

            applyVibratoDragAdjustment({
                target: adjustment.target,
                direction: adjustment.direction,
                steps: 1,
                shiftHeld: e.shiftKey,
                fineEvent: e,
            });
        };

        window.addEventListener("keydown", onKeyDown, true);
        return () => {
            window.removeEventListener("keydown", onKeyDown, true);
        };
    }, [
        applyVibratoDragAdjustment,
        vibratoDragAmplitudeIncreaseKb,
        vibratoDragAmplitudeDecreaseKb,
        vibratoDragFrequencyIncreaseKb,
        vibratoDragFrequencyDecreaseKb,
        paramFineAdjustKb,
    ]);

    const getDefaultCanvasCursor = useCallback((): CanvasCursor => {
        return toolMode === "select" ? "default" : "crosshair";
    }, [toolMode]);

    /** 指针横坐标所在帧的曲线值（不含「靠近参数线」的邻域判定）。 */
    const getCurveValueAtPointerFrame = useCallback(
        (clientX: number): number | null => {
            const pv = paramViewRef.current;
            const canvas = canvasRef.current;
            if (!pv || pv.edit.length === 0 || !canvas) return null;

            // 秒 → 帧 → 采样下标：抽到 `kernel/gestureHitTest`（纯函数，有单测）。
            // `pointerBeat` 返回 beat，先乘 `secPerBeat` 还原为秒，与本函数入参口径一致。
            return curveValueAtPointerFrame({
                sec: pointerBeat(clientX) * secPerBeat,
                startFrame: pv.startFrame,
                stride: pv.stride,
                framePeriodMs: pv.framePeriodMs,
                values: pv.edit,
            });
        },
        [paramViewRef, canvasRef, pointerBeat, secPerBeat],
    );

    const getCurveValueNearPointer = useCallback(
        (clientX: number, clientY: number): number | null => {
            const curveVal = getCurveValueAtPointerFrame(clientX);
            if (curveVal == null) return null;

            const canvas = canvasRef.current;
            if (!canvas) return null;
            const rect = canvas.getBoundingClientRect();
            const rectH = rect.height || viewSizeRef.current.h || 1;
            // 邻域判定抽到 `kernel/gestureHitTest`（纯函数，有单测）：它显式施加
            // pitch 的 +0.5 偏移，与 render.ts 的绘制位置同源。
            return isPointerNearCurve({
                pointerY: clientY - rect.top,
                param: editParam,
                valueToY: (v) => valueToY(editParam, v, rectH),
                curveValue: curveVal,
            })
                ? curveVal
                : null;
        },
        [getCurveValueAtPointerFrame, canvasRef, viewSizeRef, editParam, valueToY],
    );

    // ── 悬停浮窗的数据变化跟随 ─────────────────────────────────
    // 浮窗值是 pointermove 时的快照；键盘平移参数线 / 撤销重做 / 远端写入
    // 只会更新 paramView 数据，不动鼠标 —— 没有下面这两个 ref，浮窗会一直
    // 显示旧值，直到用户再次移动鼠标。
    // - lastPointerClientRef：最后一次画布内的指针位置（pointerleave 清空）；
    // - hoverPreviewNearCurveRef：悬停预览是否已激活（指针此前落在参数线
    //   10px 邻域内）。激活后即使曲线因大幅平移（如 Shift+= 的一个八度）
    //   远离指针，数据刷新仍按指针所在帧的新值续显 —— 否则第一拍就会把
    //   浮窗甩没；指针再次移动时恢复常规邻域判定。
    const lastPointerClientRef = useRef<{ x: number; y: number } | null>(null);
    const hoverPreviewNearCurveRef = useRef(false);

    /**
     * 用最新参数数据重算当前悬停浮窗的值（无需移动鼠标）。
     *
     * 仅当悬停预览已激活且无拖拽/平移手势进行时生效；拖拽路径的预览由
     * 拖拽自身实时驱动，不在此重算。曲线数据被清空等待重取时（如音频块
     * 范围平移经 checkpointHistory 触发的清空+强制重取）续显旧值，避免
     * 长按期间浮窗反复闪烁；新数据落地后按指针所在帧重算。
     */
    const refreshParamValuePreview = useCallback(() => {
        if (!paramValuePopupEnabled) return;
        if (!hoverPreviewNearCurveRef.current) return;
        if (strokeRef.current || panRef.current) return;
        const last = lastPointerClientRef.current;
        if (!last) return;
        if (!paramViewRef.current) return;
        const value = getCurveValueAtPointerFrame(last.x);
        if (value == null || !Number.isFinite(value)) {
            onParamValuePreviewChange?.(null);
            return;
        }
        onParamValuePreviewChange?.({
            clientX: last.x,
            clientY: last.y,
            value,
        });
    }, [
        paramValuePopupEnabled,
        onParamValuePreviewChange,
        getCurveValueAtPointerFrame,
        strokeRef,
        panRef,
        paramViewRef,
    ]);

    const isPointerNearDraggableSelection = useCallback(
        (clientX: number, clientY: number): boolean => {
            if (toolMode !== "select") return false;
            const sel = selectionRef.current;
            const canvas = canvasRef.current;
            if (!sel || sel.length === 0 || !canvas) return false;

            // 判定抽到 `kernel/gestureHitTest`（纯函数，有单测）。
            //
            // 【为什么把 beat 区间比较改成像素区间比较是等价的】原实现是
            // `beat < aBeat || beat > bBeat`，而 `beatToViewportPx` 是
            // `pxPerSec > 0` 下的单调线性投影，且 `pointerBeat` 正是它的逆
            // （同一 `axisFromRefs`、同一 `rect.left`）。因此「beat 落在区间内」
            // 与「像素落在区间内」同真同假。改用像素后，与边缘命中判定共用
            // 同一套坐标口径，不再有两份换算。
            //
            // 多选区：落在**任一段**内且靠近曲线即可拖动（拖动会带起所有段）。
            const nearCurve = getCurveValueNearPointer(clientX, clientY) != null;
            if (!nearCurve) return false;
            const rect = canvas.getBoundingClientRect();
            const localXPx = clientX - rect.left;
            return sel.some((range) =>
                hitTestSelectionBody({
                    leftXPx: beatToViewportPx(range.startBeat),
                    rightXPx: beatToViewportPx(range.endBeat),
                    localXPx,
                    nearCurve,
                }),
            );
        },
        [toolMode, selectionRef, canvasRef, beatToViewportPx, getCurveValueNearPointer],
    );

    /**
     * 边缘拉伸命中：在 `modifier.paramStretch`（默认 Alt）按下时，找**最近**的
     * 选区段边缘。多选区下返回被命中的段号 + 哪一侧 —— 拉伸只作用于那一段，
     * 其余段不动。
     */
    const findStretchSelectionEdge = useCallback(
        (
            e: ReactPointerEvent<HTMLCanvasElement>,
        ): { rangeIndex: number; edge: "left" | "right" } | null => {
            if (toolMode !== "select") return null;
            if (!isModifierActive(paramStretchKb, e.nativeEvent)) return null;
            const sel = selectionRef.current;
            const canvas = canvasRef.current;
            if (!sel || sel.length === 0 || !canvas) return null;
            const rect = canvas.getBoundingClientRect();
            // 命中判定抽到 `kernel/gestureHitTest`（纯函数，有单测）：逐段调
            // `hitTestSelectionEdge`（阈值 `SELECTION_EDGE_HIT_PX`，与原内联的
            // 8px 一致），在所有段中取**最近**的边缘——拉伸只作用于被抓住的
            // 那一段，其余段不动。同段左缘优先、跨段取先出现者，与原实现一致。
            const localXPx = e.clientX - rect.left;
            let best: { rangeIndex: number; edge: "left" | "right" } | null = null;
            let bestDistance = Number.POSITIVE_INFINITY;
            for (let i = 0; i < sel.length; i += 1) {
                const leftXPx = beatToViewportPx(sel[i].startBeat);
                const rightXPx = beatToViewportPx(sel[i].endBeat);
                const edge = hitTestSelectionEdge({ leftXPx, rightXPx, localXPx });
                if (edge === null) continue;
                const edgeXPx =
                    edge === "left" ? Math.min(leftXPx, rightXPx) : Math.max(leftXPx, rightXPx);
                const distance = Math.abs(localXPx - edgeXPx);
                if (distance < bestDistance) {
                    best = { rangeIndex: i, edge };
                    bestDistance = distance;
                }
            }
            return best;
        },
        [toolMode, paramStretchKb, selectionRef, canvasRef, beatToViewportPx],
    );

    const isPointerNearStretchSelectionEdge = useCallback(
        (e: ReactPointerEvent<HTMLCanvasElement>): boolean => findStretchSelectionEdge(e) !== null,
        [findStretchSelectionEdge],
    );

    const onCanvasPointerMove = useCallback(
        (e: ReactPointerEvent<HTMLCanvasElement>) => {
            lastPointerClientRef.current = { x: e.clientX, y: e.clientY };
            if (paramValuePopupEnabled) {
                const draggingLeft = Boolean(strokeRef.current) && (e.buttons & 1) === 1;
                if (draggingLeft) {
                    const rawPreviewValue = pointerValue(e.clientY);
                    const dragPreviewValue =
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
                        value: dragPreviewValue,
                    });
                } else {
                    const nearCurveValue = getCurveValueNearPointer(e.clientX, e.clientY);
                    // 记录悬停激活状态：数据刷新（refreshParamValuePreview）
                    // 只续显已激活的浮窗。
                    hoverPreviewNearCurveRef.current = nearCurveValue != null;
                    if (nearCurveValue == null) {
                        onParamValuePreviewChange?.(null);
                    } else {
                        onParamValuePreviewChange?.({
                            clientX: e.clientX,
                            clientY: e.clientY,
                            value: nearCurveValue,
                        });
                    }
                }
            }

            if (panRef.current || strokeRef.current) return;
            if (isPointerNearStretchSelectionEdge(e)) {
                setCanvasCursor("ew-resize");
                return;
            }
            // 多选修饰键按下时提示"可框选/可切换段"（Alt 拉伸更具体，优先）
            if (
                toolMode === "select" &&
                !isModifierActive(paramStretchKb, e.nativeEvent) &&
                isModifierActive(paramMultiSelectKb, e.nativeEvent)
            ) {
                setCanvasCursor("crosshair");
                return;
            }
            if (isPointerNearDraggableSelection(e.clientX, e.clientY)) {
                setCanvasCursor("grab");
                return;
            }
            setCanvasCursor(getDefaultCanvasCursor());
        },
        [
            paramValuePopupEnabled,
            onParamValuePreviewChange,
            pointerValue,
            toolMode,
            editParam,
            isEffectivePitchSnapActive,
            pitchSnapUnit,
            projectScale,
            scaleAtSec,
            pitchSnapToleranceCents,
            getCurveValueNearPointer,
            panRef,
            strokeRef,
            isPointerNearStretchSelectionEdge,
            isPointerNearDraggableSelection,
            paramStretchKb,
            paramMultiSelectKb,
            setCanvasCursor,
            getDefaultCanvasCursor,
            pointerSec,
        ],
    );

    const onCanvasPointerLeave = useCallback(() => {
        if (panRef.current || strokeRef.current) return;
        // 指针离开画布：悬停激活状态与最后位置一并失效，避免离画布后的
        // 数据刷新用过期坐标续显浮窗。
        hoverPreviewNearCurveRef.current = false;
        lastPointerClientRef.current = null;
        onParamValuePreviewChange?.(null);
        setCanvasCursor(getDefaultCanvasCursor());
    }, [panRef, strokeRef, onParamValuePreviewChange, setCanvasCursor, getDefaultCanvasCursor]);

    const onCanvasPointerDown = useCallback(
        (e: ReactPointerEvent<HTMLCanvasElement>) => {
            // 掌压拒识：笔 / 鼠标手势进行中时，触摸（书写时的手掌）按下不中止、
            // 不接管当前手势。pen / mouse 的第二指针维持原语义（切笔尖 / 橡皮
            // 是单手用户的正常路径）。
            if (
                shouldRejectConcurrentPointer(
                    e.nativeEvent,
                    Boolean(activePointerGestureEndRef.current),
                )
            ) {
                return;
            }
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
                (toolMode === "select" || toolMode === "draw" || toolMode === "line")
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
                        (canvasRef.current?.getBoundingClientRect().height ??
                            viewSizeRef.current.h) ||
                        1,
                };
                (e.currentTarget as HTMLCanvasElement).setPointerCapture(pid);
                const onMove = (ev: globalThis.PointerEvent) => {
                    if ((ev.buttons & 4) !== 4) {
                        onUp();
                        return;
                    }
                    const pan = panRef.current;
                    if (!pan || pan.pointerId !== pid) return;
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

                        const onMove = (ev: globalThis.PointerEvent) => {
                            if ((ev.buttons & 1) !== 1) {
                                onUp();
                                return;
                            }
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
                                const beat = pointerBeat(adjusted.clientX);
                                const sec = beat * secPerBeat;
                                const rawFrame = Math.max(
                                    0,
                                    Math.floor((sec * 1000) / Math.max(1e-6, pvNow.framePeriodMs)),
                                );
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
                            const nextOverlays: ParamMorphOverlay[] = overlayNow.map(
                                (overlay, index) =>
                                    index === drag.overlayIndex
                                        ? { ...overlay, points: nextPoints }
                                        : overlay,
                            );
                            setMorphOverlay(nextOverlays);
                            applyMorphOverlayPreview(nextOverlays);

                            if (paramValuePopupEnabled) {
                                const movedPoint = nextPoints.find(
                                    (pt) => pt.kind === drag.pointKind,
                                );
                                onParamValuePreviewChange?.({
                                    clientX: ev.clientX,
                                    clientY: ev.clientY,
                                    value: movedPoint?.value ?? pointerValue(adjusted.clientY),
                                });
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

                            if (!drag || !overlayNow || !pvNow || !rootTrackId) {
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
                                return {
                                    startFrame: packed.startFrame,
                                    values: expandStrideSampledDense(packed.dense, stride),
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
                                await uploadFullResCurveSegments({
                                    trackId: rootTrackId,
                                    param: editParam,
                                    segments: packedPerOverlay,
                                });
                                // 提交成功：保留覆盖层让波形继续显示提交值，等
                                // 提交之后取的响度快照到达再撤下（消除松手闪屏）。
                                onParamCommitSucceeded();
                                if (liveEditActiveRef) liveEditActiveRef.current = false;
                                bumpRefreshToken();
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

                const b = pointerBeat(e.clientX);
                const sel = selectionRef.current;

                // 选区构建的拍坐标换算（普通框选与多选追加共用）。
                // 允许自动滚动时有 32px 边缘触发区，且夹在 [0, 工程时长] 内。
                const maxSelectableBeat = Math.max(
                    0,
                    dynamicProjectSec / Math.max(1e-9, secPerBeat),
                );
                const clampSelectionBeat = (beat: number) => clamp(beat, 0, maxSelectableBeat);
                const selectionBeatFromClientX = (clientX: number, allowAutoScroll: boolean) => {
                    const scroller = scrollerRef.current;
                    if (!scroller) {
                        return clampSelectionBeat(pointerBeat(clientX));
                    }

                    // 本帧的交互投影：**与渲染侧同源**（内核真值，见 `axisFromRefs`）。
                    const axis = axisFromRefs();
                    const bounds = scroller.getBoundingClientRect();

                    if (allowAutoScroll) {
                        // 边缘自动滚动的映射抽到 `kernel/dragArithmetic`
                        // （纯函数，有单测）：含边缘带宽、单帧步长与比例上限
                        // 三个魔法数，原来的内联版本无法单测。
                        const deltaPx = edgeAutoScrollDeltaPx({
                            clientX,
                            leftPx: bounds.left,
                            rightPx: bounds.right,
                        });

                        if (Math.abs(deltaPx) > 0.01) {
                            // 上限必须用**同一个投影**的 pxPerSec 算（不能用可能滞后的
                            // `pxPerBeatRef`）：否则自动滚动的可达右界与实际内容宽不一致。
                            const pxPerBeatNow = axis.pxPerSec * Math.max(1e-9, secPerBeat);
                            const drawingMaxScrollLeft = Math.max(
                                0,
                                maxSelectableBeat * Math.max(1e-9, pxPerBeatNow) -
                                    scroller.clientWidth,
                            );
                            const nativeOffset = syncTimelineEnabled
                                ? timelineOffsetRef.current
                                : 0;
                            const nativeMaxScrollLeft = drawingMaxScrollLeft + nativeOffset;
                            const nextScrollLeft = clamp(
                                scroller.scrollLeft + deltaPx,
                                0,
                                nativeMaxScrollLeft,
                            );
                            if (Math.abs(nextScrollLeft - scroller.scrollLeft) > 0.01) {
                                scroller.scrollLeft = nextScrollLeft;
                                syncScrollLeft(scroller);
                            }
                        }
                    }

                    const clampedClientX = clamp(clientX, bounds.left, bounds.right);
                    // 视口 x → 秒 → 拍：一律走统一投影的逆函数，**不再**用
                    // `(scrollLeftRef + dx) / pxPerBeatRef` 这条独立算式。
                    //
                    // 【为什么这条算式是缺陷根因】它读的 `scrollLeftRef` 会被渲染期的
                    // state→ref 同步覆盖回去，而 state 是 256px 步长的量化提交，滚动后
                    // 最多滞后内核 255px——于是框选产生的选区整体偏移同一距离，而画面上
                    // 选区块按内核绘制，两者对不上（用户报告"实际产生的选区与鼠标划定的
                    // 区域不一致"，且先缩放后滚动才出现）。
                    const beat =
                        viewportPxToSec(axis, clampedClientX - bounds.left) /
                        Math.max(1e-9, secPerBeat);
                    return clampSelectionBeat(beat);
                };

                // ── 多选修饰键（默认 ⌘/Ctrl）─────────────────────────────────
                // 拖动 = 在已有选区上**追加**一段（并集；重叠/相接自动合并）；
                // 原地点击已有段 = 取消该段（与时间轴 ⌘+点击多选切换同源语义）。
                //
                // 手势优先级（高 → 低）：Alt 形变控制点（上方已 return）→
                // Alt 边缘拉伸（下面的拉伸分支）→ **本分支** → 段内近曲线拖动
                // （移动所有段）→ 普通框选（替换整个选区）。Alt 拉伸修饰键按下时
                // 本分支让位，否则在选区内侧边缘处无法拉伸。
                if (
                    e.button === 0 &&
                    !isModifierActive(paramStretchKb, e.nativeEvent) &&
                    isModifierActive(paramMultiSelectKb, e.nativeEvent)
                ) {
                    const startBeat = selectionBeatFromClientX(e.clientX, false);
                    const baseSelection = selectionRef.current ?? [];
                    const hitIndex = rangeIndexAtBeat(baseSelection, startBeat);
                    const hitExistingRange = hitIndex >= 0;
                    const startClientX = e.clientX;
                    let moved = false;

                    (e.currentTarget as HTMLCanvasElement).setPointerCapture(e.pointerId);
                    const onMove = (ev: globalThis.PointerEvent) => {
                        if ((ev.buttons & 1) !== 1) {
                            onUp();
                            return;
                        }
                        // 只接受本指针的 move：第二指针（掌压触摸等）的
                        // buttons 恒为 1，不校验 pointerId 会驱动本次手势。
                        if (ev.pointerId !== e.pointerId) return;
                        // 3px 死区：区分「点击切换」与「拖动追加」，避免手抖误删段
                        if (!moved && Math.abs(ev.clientX - startClientX) > 3) moved = true;
                        if (!moved) return;
                        const bb = selectionBeatFromClientX(ev.clientX, true);
                        selectionRef.current = addBeatRange(baseSelection, startBeat, bb);
                        updateSelectionUi(selectionRef.current);
                        invalidate();
                    };
                    const onUp = () => {
                        window.removeEventListener("pointermove", onMove);
                        window.removeEventListener("pointerup", onUp);
                        window.removeEventListener("pointercancel", onUp);
                        clearActivePointerGestureEnd(onUp);
                        if (!moved && hitExistingRange) {
                            // 切换取消：移除被点击的那一段
                            // （removeRangeAtBeat 已覆盖"按点击拍定位该段"的语义）
                            selectionRef.current = removeRangeAtBeat(
                                selectionRef.current,
                                startBeat,
                            );
                            updateSelectionUi(selectionRef.current);
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
                    // 边缘拉伸只作用于**被抓住的那一段**（其余段不动）；命中判定
                    // 取所有段中最近的边缘。
                    const stretchHit = findStretchSelectionEdge(e);
                    if (stretchHit) {
                        const aBeat = sel[stretchHit.rangeIndex].startBeat;
                        const bBeat = sel[stretchHit.rangeIndex].endBeat;
                        const edgeKind = stretchHit.edge;
                        // pointerdown 时的选区快照：拉伸期间的选区改写都以它为基底
                        // 重建（只替换被拉伸的那一段），因此归一化合并不会造成
                        // 「越拉越偏」的下标漂移。
                        const stretchBaseSelection = sel;
                        // 撤销历史：这是唯一会把「选区变化」写进历史的操作 ——
                        // 撤销恢复拉伸前的选区，重做恢复拉伸后的选区。位置取自
                        // 后端深度镜像，该手势的回写固定只打一个检查点。
                        const selectionBeforeStretch = stretchBaseSelection.map((range) => ({
                            ...range,
                        }));
                        const historyPositionBeforeStretch = getHistoryPosition?.() ?? 0;
                        let stretchABeat = aBeat;
                        let stretchBBeat = bBeat;

                        const pv = paramViewRef.current;
                        if (!pv || pv.edit.length === 0) return;
                        const fp = Math.max(1e-6, pv.framePeriodMs);
                        const stride = Math.max(1, pv.stride);
                        const oldStartFrame = Math.max(
                            0,
                            Math.floor((aBeat * secPerBeat * 1000) / fp),
                        );
                        const oldEndFrame = Math.max(
                            oldStartFrame,
                            Math.ceil((bBeat * secPerBeat * 1000) / fp),
                        );
                        const oldStartIdx = clamp(
                            Math.round((oldStartFrame - pv.startFrame) / stride),
                            0,
                            pv.edit.length - 1,
                        );
                        const oldEndIdx = clamp(
                            Math.round((oldEndFrame - pv.startFrame) / stride),
                            oldStartIdx,
                            pv.edit.length - 1,
                        );
                        const oldValues = pv.edit.slice(oldStartIdx, oldEndIdx + 1);
                        if (oldValues.length <= 0) return;

                        const pid = e.pointerId;
                        (e.currentTarget as HTMLCanvasElement).setPointerCapture(pid);
                        setCanvasCursor("ew-resize");
                        ensureLiveEditBase(pv);
                        if (liveEditActiveRef) liveEditActiveRef.current = true;
                        const finePointerState = createFineAdjustedPointerState(
                            e.nativeEvent,
                            e.currentTarget as HTMLCanvasElement,
                        );

                        const buildDense = (
                            pvNow: ParamViewSegment,
                            nextABeat: number,
                            nextBBeat: number,
                        ) => {
                            const nextStartFrame = Math.max(
                                0,
                                Math.floor((nextABeat * secPerBeat * 1000) / fp),
                            );
                            const nextEndFrame = Math.max(
                                nextStartFrame,
                                Math.ceil((nextBBeat * secPerBeat * 1000) / fp),
                            );
                            const nextStartIdx = clamp(
                                Math.round((nextStartFrame - pvNow.startFrame) / stride),
                                0,
                                pvNow.edit.length - 1,
                            );
                            const nextEndIdx = clamp(
                                Math.round((nextEndFrame - pvNow.startFrame) / stride),
                                nextStartIdx,
                                pvNow.edit.length - 1,
                            );
                            const nextLen = nextEndIdx - nextStartIdx + 1;
                            if (nextLen <= 0) return null;

                            // 平滑度 → 毫秒定标的过渡带半宽（dense 索引）
                            const edgeHalfSpanIdx = edgeHalfSpanForIndices(nextLen, stride);
                            const extraEdgeFrames = Math.ceil(edgeHalfSpanIdx) * stride;
                            const overallMinFrame = Math.max(
                                0,
                                Math.min(oldStartFrame, nextStartFrame) - extraEdgeFrames,
                            );
                            const overallMaxFrame =
                                Math.max(oldEndFrame, nextEndFrame) + extraEdgeFrames;
                            const overallLen =
                                Math.floor((overallMaxFrame - overallMinFrame) / stride) + 1;
                            const dense = new Array<number>(overallLen);
                            for (let i = 0; i < overallLen; i += 1) {
                                const frame = overallMinFrame + i * stride;
                                const idx = Math.round((frame - pvNow.startFrame) / stride);
                                dense[i] =
                                    idx >= 0 && idx < pvNow.edit.length ? pvNow.edit[idx] : 0;
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
                                const frame = nextStartFrame + i * stride;
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

                            const maxOutsideWindow = Math.max(
                                1,
                                Math.round(oldValues.length * 0.2),
                            );
                            const smoothRatio =
                                clamp(Number(edgeSmoothnessPercent) || 0, 0, 100) / 100;
                            const outsideWindowLen = Math.max(
                                1,
                                Math.round(maxOutsideWindow * smoothRatio),
                            );

                            // 缩短时，用原选区内侧一小段值回填被腾空区域。
                            // smoothness=0 时 outsideWindowLen=1，相当于边缘值沿边界内侧延展。
                            if (nextStartFrame > oldStartFrame) {
                                const fillLen = Math.floor(
                                    (nextStartFrame - oldStartFrame) / stride,
                                );
                                for (let i = 0; i < fillLen; i += 1) {
                                    const targetFrame = oldStartFrame + i * stride;
                                    const targetIdx = Math.round(
                                        (targetFrame - overallMinFrame) / stride,
                                    );
                                    const srcWindowPos =
                                        fillLen > 1
                                            ? Math.round(
                                                  (i / (fillLen - 1)) * (outsideWindowLen - 1),
                                              )
                                            : 0;
                                    const srcFrame = oldStartFrame + srcWindowPos * stride;
                                    if (targetIdx >= 0 && targetIdx < dense.length) {
                                        dense[targetIdx] = sampleOutsideValue(
                                            srcFrame,
                                            dense[targetIdx],
                                        );
                                    }
                                }
                            }
                            if (nextEndFrame < oldEndFrame) {
                                const fillLen = Math.floor((oldEndFrame - nextEndFrame) / stride);
                                for (let i = 0; i < fillLen; i += 1) {
                                    const targetFrame = nextEndFrame + (i + 1) * stride;
                                    const targetIdx = Math.round(
                                        (targetFrame - overallMinFrame) / stride,
                                    );
                                    const srcWindowPos =
                                        fillLen > 1
                                            ? Math.round(
                                                  (i / (fillLen - 1)) * (outsideWindowLen - 1),
                                              )
                                            : 0;
                                    const srcFrame = oldEndFrame - srcWindowPos * stride;
                                    if (targetIdx >= 0 && targetIdx < dense.length) {
                                        dense[targetIdx] = sampleOutsideValue(
                                            srcFrame,
                                            dense[targetIdx],
                                        );
                                    }
                                }
                            }

                            const movedStartDenseIdx = Math.round(
                                (nextStartFrame - overallMinFrame) / stride,
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
                                nextABeat,
                                nextBBeat,
                            };
                        };

                        const minBeatSpan = Math.max(1e-6, (stride * fp) / 1000 / secPerBeat);

                        // 预览重算 rAF 合帧：dense 重建 + 整份 live 拷贝 +
                        // React setState 都是 O(n)，pointermove 在高刷鼠标上
                        // 可达数百 Hz，逐事件执行长选区会卡顿；与左键选区
                        // 拖动路径同模式，一帧至多重算一次。
                        let previewRafId: number | null = null;
                        let queuedCursorBeat: number | null = null;
                        const runPreviewStep = () => {
                            const cursorBeat = queuedCursorBeat;
                            if (cursorBeat == null) return;
                            queuedCursorBeat = null;
                            const pvNow = paramViewRef.current;
                            if (!pvNow) return;
                            const nextABeat =
                                edgeKind === "left"
                                    ? clamp(cursorBeat, 0, bBeat - minBeatSpan)
                                    : aBeat;
                            const nextBBeat =
                                edgeKind === "right"
                                    ? Math.max(aBeat + minBeatSpan, cursorBeat)
                                    : bBeat;
                            const built = buildDense(pvNow, nextABeat, nextBBeat);
                            if (!built) return;
                            stretchABeat = nextABeat;
                            stretchBBeat = nextBBeat;
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
                                        ? {
                                              startBeat: built.nextABeat,
                                              endBeat: built.nextBBeat,
                                          }
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
                            const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);
                            queuedCursorBeat = pointerBeat(adjusted.clientX);
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
                            queuedCursorBeat = null;

                            const pvNow = paramViewRef.current;
                            if (!pvNow || !rootTrackId) {
                                setCanvasCursor("default");
                                return;
                            }
                            // 被拉伸那一段的最终边界（多选区只改这一段）
                            const nextABeat = stretchABeat;
                            const nextBBeat = stretchBBeat;
                            const built = buildDense(pvNow, nextABeat, nextBBeat);
                            if (!built) {
                                setCanvasCursor("default");
                                return;
                            }

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
                            // 拉伸后的选区（本轮手势的最终形态）：与拉伸前的快照
                            // 一起登记为该历史步骤的选区。
                            const selectionAfterStretch = selectionRef.current
                                ? selectionRef.current.map((range) => ({ ...range }))
                                : null;
                            void (async () => {
                                try {
                                    await uploadFullResCurve({
                                        trackId: rootTrackId,
                                        param: editParam,
                                        startFrame: built.overallMinFrame,
                                        values: expanded,
                                    });
                                    // 回写成功（已打检查点）后才登记：撤销这一步会
                                    // 恢复拉伸前的选区，重做恢复拉伸后的选区；
                                    // 其它任何选区变化都不进历史。
                                    dispatch(
                                        recordParamSelectionStretchStep({
                                            positionBefore: historyPositionBeforeStretch,
                                            before: selectionBeforeStretch,
                                            after: selectionAfterStretch,
                                        }),
                                    );
                                } catch (err) {
                                    console.error("[pianoRoll] stretch-edge commit failed", err);
                                } finally {
                                    if (liveEditActiveRef) {
                                        liveEditActiveRef.current = false;
                                    }
                                    bumpRefreshToken();
                                }
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

                    if (rangeIndexAtBeat(sel, b) !== -1) {
                        // 判断鼠标是否在曲线附近（像素距离 < 10px）
                        const pv = paramViewRef.current;
                        if (pv && pv.edit.length > 0) {
                            // 帧 → 采样下标、pitch +0.5 偏移、10px 邻域判定全部抽到
                            // `kernel/gestureHitTest`（纯函数，有单测）。此前这里是
                            // 第三份内联副本——三份各自维护 "+0.5 到底加没加" 很容易分叉。
                            const curveVal = curveValueAtPointerFrame({
                                sec: b * secPerBeat,
                                startFrame: pv.startFrame,
                                stride: pv.stride,
                                framePeriodMs: pv.framePeriodMs,
                                values: pv.edit,
                            });
                            // 下面拖拽分支仍要用这两个量（选区帧范围换算 / 起点值），
                            // 因此保留声明，只是命中判定改走纯函数。
                            const fp = pv.framePeriodMs;
                            const mouseVal = pointerValue(e.clientY);
                            const canvas = canvasRef.current;
                            const rectH = canvas
                                ? canvas.getBoundingClientRect().height
                                : viewSizeRef.current.h || 1;
                            const near = isPointerNearCurve({
                                pointerY: canvas
                                    ? e.clientY - canvas.getBoundingClientRect().top
                                    : 0,
                                param: editParam,
                                valueToY: (v) => valueToY(editParam, v, rectH),
                                curveValue: curveVal ?? Number.NaN,
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
                                    const rightDragSpans: FrameSpan[] = beatRangesToInclusiveSpans(
                                        sel,
                                        secPerBeat,
                                        fp,
                                    );
                                    if (rightDragSpans.length === 0) return;
                                    // 拖拽数据与选区移动拖拽同一模式：先用 pv 近似值立即
                                    // 预览，全分辨率到位后自动替换；提交必须等全分辨率
                                    // 数据 —— 绝不把降采样值当连续帧写回后端（旧实现在
                                    // stride>1 时会把 stride 间隔采样当连续帧写入：
                                    // 时间压缩 + 覆盖未选帧，已修复）。
                                    let origValuesPerRange: number[][] = rightDragSpans.map(
                                        (span) => readPvRange(pv, span.startFrame, span.endFrame),
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
                                            origValuesPerRange = curves.map(
                                                (curve) => curve.values,
                                            );
                                        })
                                        .catch((err) => {
                                            // 取数失败只影响预览保真度：保持 pv 近似值，
                                            // 提交路径会再次尝试并有自己的失败处理。
                                            console.error(
                                                "[pianoRoll] full-res curve fetch failed",
                                                err,
                                            );
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
                                                    apply: createSelectionAmplifier(
                                                        values,
                                                        editParam,
                                                        {
                                                            framePeriodMs: fp,
                                                        },
                                                    ).apply,
                                                };
                                                amplifierCache.set(index, cached);
                                            }
                                            return cached.apply(rightDragUpScale(dy));
                                        }
                                        return transformSelectionByRightDrag(
                                            values,
                                            editParam,
                                            dy,
                                            { framePeriodMs: fp },
                                        );
                                    };

                                    const edgeHalfSpanForRange = (index: number) =>
                                        edgeHalfSpanForIndices(
                                            origValuesPerRange[index]?.length ?? 0,
                                            1,
                                        );

                                    /**
                                     * 帧索引 dense 片段（values[k] ↔ startFrame + k）：
                                     * 每段值已含变换，边缘淡化在计划内完成。复用与选区移动
                                     * 完全相同的纯函数，任意 stride 下预览与提交一致。
                                     */
                                    const buildRightDragPlan = (
                                        pvNow: ParamViewSegment,
                                        dy: number,
                                    ) =>
                                        buildMultiRangeEditPlan({
                                            ranges: rightDragSpans,
                                            valuesAt: (index) => transformRightDragRange(index, dy),
                                            sourceAt: makePvValueSource(pvNow),
                                            edgeHalfSpanAt: edgeHalfSpanForRange,
                                            isEditable:
                                                editParam === "pitch"
                                                    ? editablePitchValue
                                                    : undefined,
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
                                            window.removeEventListener(
                                                "contextmenu",
                                                suppressOnce,
                                                true,
                                            );
                                        };
                                        window.addEventListener("contextmenu", suppressOnce, true);
                                        setTimeout(() => {
                                            window.removeEventListener(
                                                "contextmenu",
                                                suppressOnce,
                                                true,
                                            );
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
                                            const fullResCurves =
                                                await Promise.all(origCurvePromises);
                                            origValuesPerRange = fullResCurves.map(
                                                (curve) => curve.values,
                                            );
                                            // 提交基准 = 计划写入窗口（含边缘淡化扩展）的
                                            // 全分辨率数据；变换与预览同源（transformRightDragRange），
                                            // 因此写回结果与用户看到的预览逐帧一致。
                                            const commitSourceAt = await fetchCommitBaseSource({
                                                trackId: rootTrackId,
                                                param: editParam,
                                                ranges: rightDragSpans,
                                                frameDelta: 0,
                                                edgeHalfSpanAt: edgeHalfSpanForRange,
                                                paramView: pvNow,
                                            });
                                            const pieces = buildMultiRangeEditPlan({
                                                ranges: rightDragSpans,
                                                valuesAt: (index) =>
                                                    transformRightDragRange(index, lastDy),
                                                sourceAt: commitSourceAt,
                                                edgeHalfSpanAt: edgeHalfSpanForRange,
                                                isEditable:
                                                    editParam === "pitch"
                                                        ? editablePitchValue
                                                        : undefined,
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
                                                    if (
                                                        globalIdx >= 0 &&
                                                        globalIdx < nextEdit.length
                                                    ) {
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
                                                        segments: pieces.map((piece) => ({
                                                            startFrame: piece.startFrame,
                                                            values: piece.values,
                                                        })),
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
                                            console.error(
                                                "[pianoRoll] right-drag commit failed",
                                                err,
                                            );
                                            liveEditOverrideRef.current = null;
                                            if (liveEditActiveRef) {
                                                liveEditActiveRef.current = false;
                                            }
                                        }

                                        setCanvasCursor("grab");
                                        invalidate();
                                    };

                                    window.addEventListener(
                                        "contextmenu",
                                        suppressContextMenu,
                                        true,
                                    );
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
                                const startBeat = pointerBeat(e.clientX);
                                const pid = e.pointerId;
                                (e.currentTarget as HTMLCanvasElement).setPointerCapture(pid);
                                const finePointerState = createFineAdjustedPointerState(
                                    e.nativeEvent,
                                    e.currentTarget as HTMLCanvasElement,
                                );

                                // 选区帧闭区间（逐段，升序互斥）—— 注意这里**不**夹到 pv 的
                                // 已加载窗口内。「全选」时选区覆盖整个工程，若按 pv 裁剪，
                                // 后续只会改到显示范围内的参数，工程其余部分纹丝不动。
                                const dragSpans: FrameSpan[] = beatRangesToInclusiveSpans(
                                    sel,
                                    secPerBeat,
                                    fp,
                                );
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
                                 * 命中判定已经算过这个值（`curveVal`，用于 10px 邻域
                                 * 判定）；此处只是把它固定下来。取不到时为 null，
                                 * 动态将退回线性偏移。
                                 */
                                const dynDragAnchor: number | null =
                                    isDynParam(editParam) && curveVal !== null ? curveVal : null;
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

                                    // Reset live overlay before each preview step to prevent
                                    // stale values from the previous drag position lingering
                                    // outside the current range
                                    liveEditOverrideRef.current = null;
                                    ensureLiveEditBase(pvNow);

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
                                        isEditable:
                                            editParam === "pitch" ? editablePitchValue : undefined,
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
                                    // 段间距，故不会产生新的重叠段）
                                    const beatDeltaForSel = frameDeltaToBeat({
                                        frameDelta: lastFrameDelta,
                                        framePeriodMs: fp,
                                        secPerBeat,
                                    });
                                    selectionRef.current = shiftSelectionRanges(
                                        sel,
                                        beatDeltaForSel,
                                    );
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
                                    if ((ev.buttons & 1) !== 1) {
                                        onUp();
                                        return;
                                    }
                                    const adjusted = getFineAdjustedPointerPosition(
                                        finePointerState,
                                        ev,
                                    );
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

                                    // 计算 X 方向帧偏移
                                    const currentBeat = pointerBeat(adjusted.clientX);
                                    const beatDelta = currentBeat - startBeat;
                                    // beat → 帧位移：抽到 `kernel/dragArithmetic`。
                                    const rawFrameDelta = beatToFrameDelta({
                                        beatDelta,
                                        secPerBeat,
                                        framePeriodMs: fp,
                                    });

                                    // 应用拖动方向限制
                                    lastValueDelta = yDragEnabled ? rawValueDelta : 0;
                                    lastFrameDelta =
                                        currentDragDir === "y-only" ? 0 : rawFrameDelta;

                                    // 悬停弹窗跟随原始事件坐标（廉价，不合帧）。
                                    if (paramValuePopupEnabled) {
                                        const previewCurrentVal = yDragEnabled
                                            ? currentVal
                                            : startMouseVal;
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
                                                    scaleAtSec?.(pointerSec(ev.clientX)) ??
                                                    projectScale,
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

                                    // 提交拖拽结果到后端
                                    const pvNow = paramViewRef.current;
                                    if (pvNow && rootTrackId) {
                                        // await 链路上的任何 IPC 失败都必须复位
                                        // liveEdit 状态（否则预览覆盖层永久冻结、
                                        // 曲线刷新被永久搁置）。
                                        try {
                                            // 等待拖动开始时发起的全分辨率取数完成 —— 提交必须
                                            // 基于无损数据，不能拿降采样的 pv 值去覆盖后端。
                                            const fullResCurves =
                                                await Promise.all(origCurvePromises);
                                            origValuesPerRange = fullResCurves.map(
                                                (curve) => curve.values,
                                            );

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
                                            const commitSourceAt = await fetchCommitBaseSource({
                                                trackId: rootTrackId,
                                                param: editParam,
                                                ranges: dragSpans,
                                                frameDelta: lastFrameDelta,
                                                edgeHalfSpanAt: edgeHalfSpanForDragRange,
                                                paramView: pvNow,
                                            });

                                            const pieces = buildMultiRangeEditPlan({
                                                ranges: dragSpans,
                                                frameDelta: lastFrameDelta,
                                                valuesAt: (index) => origValuesPerRange[index],
                                                sourceAt: commitSourceAt,
                                                transformAt: (
                                                    _index,
                                                    _i,
                                                    sourceValue,
                                                    targetFrame,
                                                ) => transformDragValue(sourceValue, targetFrame),
                                                edgeHalfSpanAt: edgeHalfSpanForDragRange,
                                                isEditable:
                                                    editParam === "pitch"
                                                        ? editablePitchValue
                                                        : undefined,
                                            });

                                            if (pieces.length > 0) {
                                                // 立即同步更新本地 paramView state（逐帧映射到 pv 的
                                                // 采样栅格上；pv 只是显示，随后会被后端数据刷新）
                                                const nextEdit = pvNow.edit.slice();
                                                const pvStepUp = Math.max(
                                                    1,
                                                    Math.floor(pvNow.stride),
                                                );
                                                for (const piece of pieces) {
                                                    for (
                                                        let i = 0;
                                                        i < piece.values.length;
                                                        i += 1
                                                    ) {
                                                        const globalIdx = Math.round(
                                                            (piece.startFrame +
                                                                i -
                                                                pvNow.startFrame) /
                                                                pvStepUp,
                                                        );
                                                        if (
                                                            globalIdx >= 0 &&
                                                            globalIdx < nextEdit.length
                                                        ) {
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

                                                // 确保选区位置最终正确（多段同步平移）
                                                const beatDeltaForSel = frameDeltaToBeat({
                                                    frameDelta: lastFrameDelta,
                                                    framePeriodMs: fp,
                                                    secPerBeat,
                                                });
                                                selectionRef.current = shiftSelectionRanges(
                                                    sel,
                                                    beatDeltaForSel,
                                                );
                                                updateSelectionUi(selectionRef.current);

                                                // 分块回写：整次编辑（含所有段）只打一个撤销点，
                                                // 块之间让出事件循环，超长工程也不会卡死界面。
                                                // 上传失败同样复位 liveEdit（finally），不让
                                                // 覆盖层冻结。
                                                void (async () => {
                                                    try {
                                                        await uploadFullResCurveSegments({
                                                            trackId: rootTrackId,
                                                            param: editParam,
                                                            segments: pieces.map((piece) => ({
                                                                startFrame: piece.startFrame,
                                                                values: piece.values,
                                                            })),
                                                        });
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
                                                })();
                                            }
                                        } catch (err) {
                                            console.error(
                                                "[pianoRoll] select-drag commit failed",
                                                err,
                                            );
                                            liveEditOverrideRef.current = null;
                                            if (liveEditActiveRef)
                                                liveEditActiveRef.current = false;
                                        }
                                    } else {
                                        if (liveEditActiveRef) liveEditActiveRef.current = false;
                                    }
                                    // 清除参数浮窗预览（如果启用）
                                    if (paramValuePopupEnabled) {
                                        onParamValuePreviewChange?.(null);
                                    }

                                    if (
                                        editParam === "pitch" ||
                                        isChildPitchOffsetCentsParam(editParam) ||
                                        isChildPitchOffsetDegreesParam(editParam)
                                    ) {
                                        onPitchSnapGestureActiveChange?.(false);
                                    }
                                    setCanvasCursor("grab");
                                    invalidate();
                                    window.removeEventListener(
                                        "contextmenu",
                                        onContextMenuDuringDrag,
                                        true,
                                    );
                                    window.removeEventListener(
                                        "mousedown",
                                        onMouseDownDuringDrag,
                                        true,
                                    );
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
                                const disposeDragDirKey =
                                    installDragDirectionKeyCycler(cycleSelectDragDir);

                                window.addEventListener("pointermove", onMove);
                                window.addEventListener("pointerup", onUp);
                                window.addEventListener("pointercancel", onUp);
                                window.addEventListener(
                                    "contextmenu",
                                    onContextMenuDuringDrag,
                                    true,
                                );
                                window.addEventListener("mousedown", onMouseDownDuringDrag, true);
                                setActivePointerGestureEnd(onUp);
                                return;
                            }
                        }
                    }
                }

                // 默认行为：仅左键创建新选区（**替换**整个选区为单段）；
                // 右键不应在 pointerdown 时清除选区。
                // 拍坐标换算 selectionBeatFromClientX 已在上方定义（与多选追加共用）。
                if (e.button === 0) {
                    const startBeat = selectionBeatFromClientX(e.clientX, false);
                    selectionRef.current = selectionFromBeatRange(startBeat, startBeat);
                    updateSelectionUi(selectionRef.current);
                    // 立即重绘：此刻新选区是**零宽**的，画布上还显示着**上一个**
                    // 选区框（以及新位置的一条边线）。不在这里标脏，用户就会看到
                    // "按下后旧框仍在原处不动"——直到指针移动触发下面的 onMove 才更新，
                    // 这正是报告症状的一部分。
                    //
                    // 与 onMove（实时重绘选区）和 onUp（收尾重绘）保持一致：一次
                    // 手势的三个阶段都各自标脏，任何一段缺失都会留下陈旧画面。
                    // 【为什么光有它还不够】主画布走内容签名缓存，签名必须是
                    // 引用比较（见 `mainCanvasSignature.ts`）——此处每次赋**新对象**，
                    // 正是让签名变化、缓存失效的前提。
                    invalidate();
                    const pid = e.pointerId;
                    (e.currentTarget as HTMLCanvasElement).setPointerCapture(pid);
                    const finePointerState = createFineAdjustedPointerState(
                        e.nativeEvent,
                        e.currentTarget as HTMLCanvasElement,
                    );
                    const onMove = (ev: globalThis.PointerEvent) => {
                        if ((ev.buttons & 1) !== 1) {
                            onUp();
                            return;
                        }
                        // 只接受本指针的 move（掌压拒识的 move 侧防线）。
                        if (ev.pointerId !== pid) return;
                        // 选区在拖拽途中被外部清除（如 BackSpace / 取消选择）时
                        // 不再续建，与改造前的守卫语义一致。
                        if (selectionRef.current == null) return;
                        const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);
                        const bb = selectionBeatFromClientX(adjusted.clientX, true);
                        selectionRef.current = selectionFromBeatRange(startBeat, bb);
                        updateSelectionUi(selectionRef.current);
                        invalidate(); // 实时重绘选区
                    };
                    const onUp = () => {
                        window.removeEventListener("pointermove", onMove);
                        window.removeEventListener("pointerup", onUp);
                        window.removeEventListener("pointercancel", onUp);
                        disposeFineAdjustedPointerState(finePointerState);
                        clearActivePointerGestureEnd(onUp);
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
            const beat = pointerBeat(e.clientX);
            const sec = beat * secPerBeat;
            const frame = Math.max(0, Math.floor((sec * 1000) / fp));
            const rawValue = pointerValue(e.clientY);
            const isDrawMode = mode === "draw";
            const snapToggleHeld = isSnapToggleModifierHeld(e.nativeEvent);
            const value = isDrawMode ? snapDrawValue(rawValue, snapToggleHeld, frame) : rawValue;

            const isLineTool = toolMode === "line";
            const isVibratoTool = toolMode === "vibrato";

            strokeRef.current = {
                mode,
                pointerId: e.pointerId,
                param: editParam,
                points: [{ frame, value }],
            };
            if (!isVibratoTool) {
                vibratoStateRef.current = null;
                setVibratoDragCaptureActive(false);
            }
            // 标记 live 编辑开始，阻止 pitch_orig_updated 事件立即刷新曲线
            if (liveEditActiveRef) liveEditActiveRef.current = true;

            // For line tool, only show the start point initially
            const pv0 = paramViewRef.current;
            if (pv0) {
                applyDenseToLiveEdit(
                    pv0,
                    frame,
                    mode === "restore" ? null : [value],
                    frame,
                    frame,
                    mode,
                );
            }
            (e.currentTarget as HTMLCanvasElement).setPointerCapture(e.pointerId);
            invalidate();
            // 与 move 路径同一约定：live 覆盖不进 React，波形需显式重绘。
            requestWaveformRepaint();

            if (isLineTool || isVibratoTool) {
                // Line tool: draw a straight line from start to current pointer
                const startFrame = frame;
                const startValue = value;
                // 橡皮端（button 5）在 buttons 位掩码里是位 32，不是右键的位 2。
                const requiredButtonMask =
                    mode === "restore" ? (penEraserDown ? PEN_ERASER_BUTTONS_MASK : 2) : 1;
                let currentDragDir: "free" | "x-only" =
                    dragDirection === "x-only" ? "x-only" : "free";
                const canCycleDragDirection = e.button === 0;

                if (isVibratoTool) {
                    vibratoStateRef.current = {
                        pointerId: e.pointerId,
                        startFrame,
                        startValue,
                        currentFrame: startFrame,
                        currentValue: startValue,
                        mode,
                        amplitude: 0,
                        frequency: 3,
                        shiftHeld: snapToggleHeld,
                    };
                    setVibratoDragCaptureActive(true);
                }
                const finePointerState = createFineAdjustedPointerState(
                    e.nativeEvent,
                    e.currentTarget as HTMLCanvasElement,
                );

                // 直线 / 颤音是「起点 → 当前点」的端点式预览：每个事件都重算
                // 全线 dense，O(线长)。pen 高采样率下逐事件执行 = 同帧多次
                // 全线重建；与自由绘制分支同一 rAF 合帧模式（事件入队，一帧
                // 一消费，getCoalescedEvents 展开同帧采样）。端点式预览只需
                // 队列里**最后一个**采样点即可得到正确结果。
                let lineRafId: number | null = null;
                let pendingLineEvent: globalThis.PointerEvent | null = null;
                const processLineEvent = (ev: {
                    clientX: number;
                    clientY: number;
                    ctrlKey: boolean;
                    shiftKey: boolean;
                    altKey: boolean;
                }) => {
                    const st = strokeRef.current;
                    if (!st || st.pointerId !== e.pointerId) return;
                    const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);
                    const b2 = pointerBeat(adjusted.clientX);
                    const sec2 = b2 * secPerBeat;
                    const f2 = Math.max(0, Math.floor((sec2 * 1000) / fp));
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
                            if (isVibratoTool) {
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
                                        vib.amplitude,
                                        vib.frequency,
                                        moveSnapToggleHeld,
                                    );
                                    applyDenseToLiveEdit(
                                        pv2,
                                        built.minF,
                                        built.dense,
                                        built.minF,
                                        built.maxF,
                                        mode,
                                    );
                                }
                            } else {
                                const len = maxF - minF + 1;
                                const dense = new Array<number>(len);
                                const denom = f2 - startFrame;
                                for (let f = minF; f <= maxF; f++) {
                                    const t = denom === 0 ? 1 : (f - startFrame) / denom;
                                    const raw = startValue + (v2 - startValue) * t;
                                    dense[f - minF] = isDrawMode
                                        ? snapDrawValue(raw, moveSnapToggleHeld, f)
                                        : raw;
                                }
                                applyDenseToLiveEdit(pv2, minF, dense, minF, maxF, mode);
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
                    window.removeEventListener("pointermove", onMove);
                    window.removeEventListener("pointerup", onUp);
                    window.removeEventListener("pointercancel", onUp);
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
                        if (isVibratoTool && vib && st.mode === "draw") {
                            const built = buildVibratoDense(
                                vib.startFrame,
                                vib.startValue,
                                vib.currentFrame,
                                vib.currentValue,
                                vib.amplitude,
                                vib.frequency,
                                vib.shiftHeld,
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
                const cycleLineDragDir = () => {
                    if (!canCycleDragDirection) return;
                    currentDragDir = currentDragDir === "free" ? "x-only" : "free";
                    if (onCycleDragDirection) {
                        onCycleDragDirection(isVibratoTool ? "vibrato" : "draw");
                    }
                };
                const onMouseDownDuringDraw = (ev: globalThis.MouseEvent) => {
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
                let currentDragDir: "free" | "x-only" =
                    dragDirection === "x-only" ? "x-only" : "free";
                const canCycleDragDirection = e.button === 0;
                const finePointerState = createFineAdjustedPointerState(
                    e.nativeEvent,
                    e.currentTarget as HTMLCanvasElement,
                );
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
                    const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);
                    const b2Raw = pointerBeat(adjusted.clientX);
                    const last = st.points[st.points.length - 1];
                    const b2 = b2Raw;
                    const sec2 = b2 * secPerBeat;
                    const f2 = Math.max(0, Math.floor((sec2 * 1000) / fp));
                    const yDragEnabled = currentDragDir !== "x-only";
                    const rawV2Abs = yDragEnabled
                        ? pointerValue(adjusted.clientY)
                        : (last?.value ?? value);
                    const rawV2 = rawV2Abs;
                    const moveSnapToggleHeld = isSnapToggleModifierHeld(ev);
                    const v2 = isDrawMode ? snapDrawValue(rawV2, moveSnapToggleHeld, f2) : rawV2;

                    const pv2 = paramViewRef.current;
                    if (last && last.frame === f2) {
                        last.value = v2;
                        if (pv2) {
                            applyDenseToLiveEdit(
                                pv2,
                                f2,
                                mode === "restore" ? null : [v2],
                                f2,
                                f2,
                                mode,
                            );
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
                 * pointercancel 收尾：**回滚**而不是提交（与直线/颤音分支
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
        },
        [
            rootTrackId,
            editParam,
            toolMode,
            scrollerRef,
            canvasRef,
            viewSizeRef,
            panRef,
            pitchViewRef,
            paramViewsRef,
            syncScrollLeft,
            clampViewport,
            setPitchView,
            setParamViewport,
            invalidate,
            pointerBeat,
            pointerSec,
            selectionRef,
            updateSelectionUi,
            paramMultiSelectKb,
            findStretchSelectionEdge,
            fetchCommitBaseSource,
            paramViewRef,
            ensureLiveEditBase,
            paramView?.framePeriodMs,
            secPerBeat,
            axisFromRefs,
            pointerValue,
            strokeRef,
            applyDenseToLiveEdit,
            resetLiveEditPreview,
            onParamCommitSucceeded,
            blendDenseEdges,
            edgeHalfSpanForIndices,
            applyMorphOverlayPreview,
            applyPostStrokeSmoothing,
            buildMorphDense,
            commitStroke,
            bumpRefreshToken,
            liveEditOverrideRef,
            setMorphOverlay,
            setParamView,
            setCanvasCursor,
            onPitchSnapGestureActiveChange,
            pitchSnapUnit,
            projectScale,
            scaleAtSec,
            pitchSnapToleranceCents,
            isSnapToggleModifierHeld,
            isEffectivePitchSnapActive,
            createFineAdjustedPointerState,
            getFineAdjustedPointerPosition,
            disposeFineAdjustedPointerState,
            pxPerBeatRef,
            scrollLeftRef,
            syncTimelineEnabled,
            timelineOffsetRef,
            valueToY,
            buildVibratoDense,
            paramEditorSeekPlayheadEnabled,
            paramValuePopupEnabled,
            onParamValuePreviewChange,
            setActivePointerGestureEnd,
            clearActivePointerGestureEnd,
            setVibratoDragCaptureActive,
            dynamicProjectSec,
            dispatch,
            dragDirection,
            edgeSmoothnessPercent,
            getDefaultCanvasCursor,
            liveEditActiveRef,
            onContextMenu,
            onCycleDragDirection,
            getHistoryPosition,
            installDragDirectionKeyCycler,
            paramStretchKb,
            snapDrawValue,
        ],
    );

    useEffect(() => {
        if (!paramValuePopupEnabled) {
            onParamValuePreviewChange?.(null);
            return;
        }
        // pointerup/pointercancel 清除浮窗的同时复位悬停激活状态：拖拽/平移/
        // 点击结束后，激活与否必须由下一次 pointermove 重新判定，否则随后
        // 的数据刷新会在旧激活标记下用过期坐标续显浮窗。
        const clearPreview = () => {
            hoverPreviewNearCurveRef.current = false;
            onParamValuePreviewChange?.(null);
        };
        window.addEventListener("pointerup", clearPreview);
        window.addEventListener("pointercancel", clearPreview);
        return () => {
            window.removeEventListener("pointerup", clearPreview);
            window.removeEventListener("pointercancel", clearPreview);
        };
    }, [paramValuePopupEnabled, onParamValuePreviewChange]);

    return {
        onRulerMouseDown,
        onScrollerAuxClick,
        onScrollerScroll,
        onScrollerContextMenu,
        onScrollerKeyDown,
        onScrollerWheelNative,
        onCanvasPointerMove,
        onCanvasPointerLeave,
        onCanvasPointerDown,
        refreshParamValuePreview,
    };
}
