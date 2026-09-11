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
import { seekPlayhead, setplayheadSec } from "../../../features/session/sessionSlice";
import { clamp, MAX_PX_PER_SEC, MIN_PX_PER_SEC } from "../timeline";
import type {
    ParamMorphOverlay,
    ParamName,
    ParamViewSegment,
    StrokeMode,
    StrokePoint,
    ValueViewport,
} from "./types";
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
import { getParamEditorWheelAction, getVibratoDragWheelTarget } from "./wheelGesture";
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
    selectionContainsBeat,
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
import {
    createTimelineAxis,
    secToViewportPx,
    viewportPxToSec,
} from "../timeline/runtime/timelineAxis.js";

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

    commitStroke: (points: StrokePoint[], mode: StrokeMode) => Promise<void>;

    /** 用于选区拖拽 onUp 时同步更新本地 paramView state（与 commitStroke 行为一致） */
    setParamView: (next: ParamViewSegment | null) => void;
    /** 用于选区拖拽 onUp 时清除 live edit overlay */
    liveEditOverrideRef: MutRef<{ key: string; edit: number[] } | null>;

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
    /** modifier.clipStretch 绑定（选择工具参数拉伸） */
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
        edgeSmoothnessPercent,
        onPitchSnapGestureActiveChange,
        onMorphOverlayChange,
        currentParamRange,
    } = args;

    /**
     * 由滚动 ref 构造当前投影。
     *
     * 用 ref 而非 React state：滚动事件中 ref 同步更新，state 要到下一帧才落地，
     * 命中测试若用 state 会比画面慢一帧。
     */
    const axisFromRefs = useCallback(
        () =>
            createTimelineAxis({
                pxPerSec: pxPerSecRef.current,
                scrollLeftPx: scrollLeftRef.current,
            }),
        [pxPerSecRef, scrollLeftRef],
    );

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

        const fp = Math.max(1e-6, pv.framePeriodMs);
        const stride = Math.max(1, pv.stride);
        const overlays: ParamMorphOverlay[] = [];

        for (const range of sel) {
            const aBeat = range.startBeat;
            const bBeat = range.endBeat;
            if (!Number.isFinite(aBeat) || !Number.isFinite(bBeat) || bBeat <= aBeat) continue;

            const selStartFrameRaw = Math.max(0, Math.floor((aBeat * secPerBeat * 1000) / fp));
            const selEndFrameRaw = Math.max(
                selStartFrameRaw,
                Math.ceil((bBeat * secPerBeat * 1000) / fp),
            );
            const selStartIdx = clamp(
                Math.round((selStartFrameRaw - pv.startFrame) / stride),
                0,
                pv.edit.length - 1,
            );
            const selEndIdx = clamp(
                Math.round((selEndFrameRaw - pv.startFrame) / stride),
                selStartIdx,
                pv.edit.length - 1,
            );
            const baselineValues = pv.edit.slice(selStartIdx, selEndIdx + 1);
            if (baselineValues.length === 0) continue;

            const valid =
                editParam === "pitch"
                    ? baselineValues.filter((v) => Number(v) !== 0)
                    : baselineValues;
            const meanValue =
                valid.length > 0
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
        },
        [applyDenseToLiveEdit, buildMorphDense, ensureLiveEditBase, paramViewRef],
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
                liveEditOverrideRef.current = null;
                ensureLiveEditBase(pvNow);
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
            }

            return true;
        },
        [
            pointerFineWheelScale,
            editParam,
            currentParamRange,
            strokeRef,
            paramViewRef,
            liveEditOverrideRef,
            ensureLiveEditBase,
            buildVibratoDense,
            applyDenseToLiveEdit,
            invalidate,
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
                axis: createTimelineAxis({
                    pxPerSec: pxPerSecRef.current,
                    scrollLeftPx: scrollLeftRef.current,
                }),
            });
        },
        [canvasRef, scrollLeftRef, pxPerSecRef],
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
            if (e.button !== 0) return;
            const ruler = e.currentTarget as HTMLDivElement;
            let moved = false;

            const updateAt = (clientX: number, commit: boolean): number => {
                const bounds = ruler.getBoundingClientRect();
                const sec = clamp(
                    secFromViewportClientX({
                        clientX,
                        viewportLeft: bounds.left,
                        axis: createTimelineAxis({
                            pxPerSec: pxPerSecRef.current,
                            scrollLeftPx: scrollLeftRef.current,
                        }),
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

    const onScrollerScroll = useCallback(
        (e: UIEvent<HTMLDivElement>) => {
            syncScrollLeft(e.currentTarget as HTMLDivElement);
        },
        [syncScrollLeft],
    );

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

            const beat = pointerBeat(clientX);
            const fp = pv.framePeriodMs;
            const sec = beat * secPerBeat;
            const frame = Math.max(0, Math.floor((sec * 1000) / fp));
            const idx = Math.round((frame - pv.startFrame) / Math.max(1, pv.stride));
            const curveVal = idx >= 0 && idx < pv.edit.length ? Number(pv.edit[idx]) : null;
            if (curveVal == null || !Number.isFinite(curveVal)) return null;
            return curveVal;
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
            const mouseY = clientY - rect.top;
            const mappedCurveVal = editParam === "pitch" ? curveVal + 0.5 : curveVal;
            const curveY = valueToY(editParam, mappedCurveVal, rectH);
            return Math.abs(mouseY - curveY) < 10 ? curveVal : null;
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
            if (!sel || sel.length === 0) return false;

            // 多选区：落在**任一段**内且靠近曲线即可拖动（拖动会带起所有段）
            if (!selectionContainsBeat(sel, pointerBeat(clientX))) return false;

            return getCurveValueNearPointer(clientX, clientY) != null;
        },
        [toolMode, selectionRef, pointerBeat, getCurveValueNearPointer],
    );

    /**
     * 边缘拉伸命中：在 Alt（modifier.clipStretch）按下时，找**最近**的选区段边缘。
     * 多选区下返回被命中的段号 + 哪一侧 —— 拉伸只作用于那一段，其余段不动。
     */
    const findStretchSelectionEdge = useCallback(
        (e: ReactPointerEvent<HTMLCanvasElement>): { rangeIndex: number; edge: "left" | "right" } | null => {
            if (toolMode !== "select") return null;
            if (!isModifierActive(paramStretchKb, e.nativeEvent)) return null;
            const sel = selectionRef.current;
            const canvas = canvasRef.current;
            if (!sel || sel.length === 0 || !canvas) return null;
            const rect = canvas.getBoundingClientRect();
            const localX = e.clientX - rect.left;
            const edgeHitPx = 8;
            let best: { rangeIndex: number; edge: "left" | "right" } | null = null;
            let bestDistance = Number.POSITIVE_INFINITY;
            for (let i = 0; i < sel.length; i += 1) {
                const leftX = beatToViewportPx(sel[i].startBeat);
                const rightX = beatToViewportPx(sel[i].endBeat);
                const leftDistance = Math.abs(localX - leftX);
                const rightDistance = Math.abs(localX - rightX);
                if (leftDistance <= edgeHitPx && leftDistance < bestDistance) {
                    best = { rangeIndex: i, edge: "left" };
                    bestDistance = leftDistance;
                }
                if (rightDistance <= edgeHitPx && rightDistance < bestDistance) {
                    best = { rangeIndex: i, edge: "right" };
                    bestDistance = rightDistance;
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
            if (activePointerGestureEndRef.current) {
                activePointerGestureEndRef.current();
            }

            if (e.button === 0 && paramEditorSeekPlayheadEnabled !== false) {
                const sec = pointerSec(e.clientX);
                dispatch(setplayheadSec(sec));
                void dispatch(seekPlayhead(sec));
            }

            if (
                e.button === 0 &&
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
                    let hit: { overlayIndex: number; point: ParamMorphOverlay["points"][number] } | null =
                        null;
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
                                liveEditOverrideRef.current = null;
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

                    const bounds = scroller.getBoundingClientRect();
                    const edgePx = 32;
                    const maxStepPx = 18;

                    if (allowAutoScroll) {
                        let deltaPx = 0;
                        if (clientX < bounds.left + edgePx) {
                            const ratio = (bounds.left + edgePx - clientX) / edgePx;
                            deltaPx = -clamp(ratio, 0, 1.5) * maxStepPx;
                        } else if (clientX > bounds.right - edgePx) {
                            const ratio = (clientX - (bounds.right - edgePx)) / edgePx;
                            deltaPx = clamp(ratio, 0, 1.5) * maxStepPx;
                        }

                        if (Math.abs(deltaPx) > 0.01) {
                            const drawingMaxScrollLeft = Math.max(
                                0,
                                maxSelectableBeat * Math.max(1e-9, pxPerBeatRef.current) -
                                    scroller.clientWidth,
                            );
                            const nativeOffset = syncTimelineEnabled ? timelineOffsetRef.current : 0;
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
                    const beat =
                        (scrollLeftRef.current + (clampedClientX - bounds.left)) /
                        Math.max(1e-9, pxPerBeatRef.current);
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
                                Math.floor((overallMaxFrame - overallMinFrame) / stride) +
                                1;
                            const dense = new Array<number>(overallLen);
                            for (let i = 0; i < overallLen; i += 1) {
                                const frame = overallMinFrame + i * stride;
                                const idx = Math.round((frame - pvNow.startFrame) / stride);
                                dense[i] =
                                    idx >= 0 && idx < pvNow.edit.length
                                        ? pvNow.edit[idx]
                                        : 0;
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

                            const sampleOutsideValue = (
                                srcFrame: number,
                                fallback: number,
                            ) => {
                                const srcIdx = Math.round(
                                    (srcFrame - pvNow.startFrame) / stride,
                                );
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
                                                  (i / (fillLen - 1)) *
                                                      (outsideWindowLen - 1),
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
                                const fillLen = Math.floor(
                                    (oldEndFrame - nextEndFrame) / stride,
                                );
                                for (let i = 0; i < fillLen; i += 1) {
                                    const targetFrame = nextEndFrame + (i + 1) * stride;
                                    const targetIdx = Math.round(
                                        (targetFrame - overallMinFrame) / stride,
                                    );
                                    const srcWindowPos =
                                        fillLen > 1
                                            ? Math.round(
                                                  (i / (fillLen - 1)) *
                                                      (outsideWindowLen - 1),
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

                        const minBeatSpan = Math.max(
                            1e-6,
                            (stride * fp) / 1000 / secPerBeat,
                        );

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
                            liveEditOverrideRef.current = null;

                            // 全分辨率无损提交（与右拖 / 选区拖拽同一范式）：
                            // buildDense 产生的是 pv 步距采样（dense[k] ↔
                            // overallMinFrame + k×stride）；按渲染同款线性插值
                            // 展开为逐帧值后分块回写 —— 绝不把 stride 间隔采样
                            // 当连续帧写入（旧实现在 stride>1 时时间压缩 +
                            // 覆盖未选帧，见 selectionEditData.expandStrideSampledDense）。
                            const expanded = expandStrideSampledDense(built.dense, stride);
                            void (async () => {
                                try {
                                    await uploadFullResCurve({
                                        trackId: rootTrackId,
                                        param: editParam,
                                        startFrame: built.overallMinFrame,
                                        values: expanded,
                                    });
                                } catch (err) {
                                    console.error(
                                        "[pianoRoll] stretch-edge commit failed",
                                        err,
                                    );
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
                            const fp = pv.framePeriodMs;
                            const sec = b * secPerBeat;
                            const frame = Math.max(0, Math.floor((sec * 1000) / fp));
                            const idx = Math.round(
                                (frame - pv.startFrame) / Math.max(1, pv.stride),
                            );
                            const curveVal = idx >= 0 && idx < pv.edit.length ? pv.edit[idx] : null;
                            const mouseVal = pointerValue(e.clientY);

                            // 使用像素距离判断是否靠近曲线，避免不同参数值域差异的影响
                            const canvas = canvasRef.current;
                            const rectH = canvas
                                ? canvas.getBoundingClientRect().height
                                : viewSizeRef.current.h || 1;
                            const mouseY = canvas
                                ? e.clientY - canvas.getBoundingClientRect().top
                                : 0;
                            // pitch 绘制时有 +0.5 偏移（画在琴键中心），命中检测需保持一致
                            const mappedCurveVal =
                                curveVal !== null
                                    ? editParam === "pitch"
                                        ? curveVal + 0.5
                                        : curveVal
                                    : null;
                            const curveY =
                                mappedCurveVal !== null
                                    ? valueToY(editParam, mappedCurveVal, rectH)
                                    : null;
                            const HIT_THRESHOLD_PX = 10;

                            if (curveY !== null && Math.abs(mouseY - curveY) < HIT_THRESHOLD_PX) {
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
                                                    apply: createSelectionAmplifier(values, editParam, {
                                                        framePeriodMs: fp,
                                                    }).apply,
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
                                        const adjusted = getFineAdjustedPointerPosition(
                                            finePointerState,
                                            ev,
                                        );
                                        const dy = startClientY - adjusted.clientY;
                                        if (Math.abs(dy) >= 2) {
                                            didDrag = true;
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
                                            const fullResCurves = await Promise.all(
                                                origCurvePromises,
                                            );
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
                                            liveEditOverrideRef.current = null;

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
                                    return orig + lastValueDelta;
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
                                    const beatDeltaForSel =
                                        (lastFrameDelta * fp) / 1000 / secPerBeat;
                                    selectionRef.current = shiftSelectionRanges(
                                        sel,
                                        beatDeltaForSel,
                                    );
                                    updateSelectionUi(selectionRef.current);

                                    invalidate();
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
                                    const secDelta = beatDelta * secPerBeat;
                                    const rawFrameDelta = Math.round((secDelta * 1000) / fp);

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
                                            const fullResCurves = await Promise.all(
                                                origCurvePromises,
                                            );
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
                                                transformAt: (_index, _i, sourceValue, targetFrame) =>
                                                    transformDragValue(sourceValue, targetFrame),
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
                                                const pvStepUp = Math.max(1, Math.floor(pvNow.stride));
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
                                                liveEditOverrideRef.current = null;

                                                // 确保选区位置最终正确（多段同步平移）
                                                const beatDeltaForSel =
                                                    (lastFrameDelta * fp) / 1000 / secPerBeat;
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

            const mode: StrokeMode = e.button === 2 ? "restore" : "draw";
            if (e.button !== 0 && e.button !== 2) return;
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

            if (isLineTool || isVibratoTool) {
                // Line tool: draw a straight line from start to current pointer
                const startFrame = frame;
                const startValue = value;
                const requiredButtonMask = mode === "restore" ? 2 : 1;
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

                const onMove = (ev: globalThis.PointerEvent) => {
                    if ((ev.buttons & requiredButtonMask) !== requiredButtonMask) {
                        onUp();
                        return;
                    }
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
                        // Reset live overlay so the previous line preview doesn't leave artifacts
                        liveEditOverrideRef.current = null;
                        ensureLiveEditBase(pv2);
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
                };

                const onUp = () => {
                    const st = strokeRef.current;
                    const isOwnStroke = Boolean(st && st.pointerId === e.pointerId);
                    const vib = isOwnStroke ? vibratoStateRef.current : null;
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
                window.addEventListener("pointercancel", onUp);
                window.addEventListener("contextmenu", onContextMenuDuringDraw, true);
                window.addEventListener("mousedown", onMouseDownDuringDraw, true);
                setActivePointerGestureEnd(onUp);
            } else {
                // Draw tool: freehand drawing with interpolation between points
                const requiredButtonMask = mode === "restore" ? 2 : 1;
                let currentDragDir: "free" | "x-only" =
                    dragDirection === "x-only" ? "x-only" : "free";
                const canCycleDragDirection = e.button === 0;
                const finePointerState = createFineAdjustedPointerState(
                    e.nativeEvent,
                    e.currentTarget as HTMLCanvasElement,
                );
                const onMove = (ev: globalThis.PointerEvent) => {
                    if ((ev.buttons & requiredButtonMask) !== requiredButtonMask) {
                        onUp();
                        return;
                    }
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
                };

                const onUp = () => {
                    const st = strokeRef.current;
                    const isOwnStroke = Boolean(st && st.pointerId === e.pointerId);
                    if (isOwnStroke) {
                        strokeRef.current = null;
                        vibratoStateRef.current = null;
                        setVibratoDragCaptureActive(false);
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
                        await commitStroke(st.points, st.mode);
                        await applyPostStrokeSmoothing(st.points, st.mode);
                    })();
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
                window.addEventListener("pointercancel", onUp);
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
