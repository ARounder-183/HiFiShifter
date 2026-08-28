import type {
    KeyboardEvent,
    MouseEvent as ReactMouseEvent,
    MutableRefObject,
    PointerEvent as ReactPointerEvent,
    UIEvent,
    WheelEvent,
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
import { matchesKeybinding } from "../../../features/keybindings/useKeybindings";
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
    CHILD_PITCH_OFFSET_DEGREES_RANGE,
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
    isChildFormantOffsetCentsParam,
    snapChildPitchOffsetValue,
} from "./childPitchOffsetParams";
import { buildChildOffsetPasteValues as buildChildOffsetPasteValuesHelper } from "./childPitchOffsetPaste";
import { resolveHorizontalWheelZoom } from "../timeline/runtime/timelineScrollRange";
import { resolveTimelineMinPxPerSec } from "../timeline/runtime/timelineZoomBounds";
import { getParamEditorWheelAction, getVibratoDragWheelTarget } from "./wheelGesture";
import { transformSelectionByRightDrag } from "./selectionTransforms";
import {
    buildSelectionDragDense,
    fetchFullResCurve,
    readPvRange,
    selectionDragRange,
    uploadFullResCurve,
} from "./selectionEditData";
import {
    computeVibratoDragAdjustment,
    resolveVibratoDragKeyboardAdjustment,
} from "./vibratoDragAdjust";
import {
    formatRightDragMorphPercent,
    getDrawPreviewValue,
    getSelectDragPreviewValue,
} from "./paramValuePreviewLogic";
import {
    readSystemClipboardObject,
    writeSystemClipboardObject,
} from "../../../utils/systemClipboard";
import { secFromViewportClientX } from "./seekPlayheadMapping";

type CanvasCursor = "default" | "crosshair" | "grab" | "grabbing" | "ew-resize";

export function usePianoRollInteractions(args: {
    dispatch: AppDispatch;
    rootTrackId: string | null;
    selectedTrackId: string | null;
    tracks: Array<{ id: string; parentId?: string | null }>;
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

    selectionRef: MutableRefObject<{ aBeat: number; bBeat: number } | null>;
    selectionUi?: { aBeat: number; bBeat: number } | null;
    setSelectionUi: (next: { aBeat: number; bBeat: number } | null) => void;
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

    clipboardRef: MutableRefObject<{
        param: ParamName;
        framePeriodMs: number;
        values: number[];
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
    liveEditActiveRef?: MutableRefObject<boolean>; /** pianoRoll.copy 绑定 */
    pianoRollCopyKb: Keybinding;
    /** pianoRoll.paste 绑定 */
    pianoRollPasteKb: Keybinding;
    /** modifier.pianoRollVerticalZoom 绑定 */
    prVerticalZoomKb: Keybinding;
    /** modifier.horizontalZoom 绑定 */
    horizontalZoomKb: Keybinding;
    /** modifier.scrollHorizontal 绑定 */
    scrollHorizontalKb: Keybinding;
    /** modifier.scrollVertical 绑定 */
    scrollVerticalKb: Keybinding;
    /** modifier.paramMorph 绑定 */
    paramMorphKb: Keybinding;
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
    /** 播放头位置（秒）用于以播放头为中心缩放 */
    playheadSec?: number;
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
    /** 选区拖拽时边缘平滑度（0-100%） */
    edgeSmoothnessPercent?: number;
    /** 选择拖拽/绘制进行中时，用于临时切换吸附按钮视觉 */
    onPitchSnapGestureActiveChange?: (active: boolean) => void;
    /** 形变控制线变化回调（null 表示隐藏） */
    onMorphOverlayChange?: (next: ParamMorphOverlay | null) => void;
    /** 当前参数值域（用于振幅滚轮步进自适应） */
    currentParamRange?: { min: number; max: number };
}) {
    const {
        dispatch,
        rootTrackId,
        selectedTrackId,
        tracks,
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
        clipboardRef,
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
        pianoRollCopyKb,
        pianoRollPasteKb,
        prVerticalZoomKb,
        horizontalZoomKb,
        scrollHorizontalKb,
        scrollVerticalKb,
        paramMorphKb,
        paramFineAdjustKb,
        paramStretchKb,
        vibratoAmplitudeAdjustKb,
        vibratoFrequencyAdjustKb,
        vibratoDragAmplitudeIncreaseKb,
        vibratoDragAmplitudeDecreaseKb,
        vibratoDragFrequencyIncreaseKb,
        vibratoDragFrequencyDecreaseKb,
        playheadSec,
        playheadZoomEnabled,
        paramEditorSeekPlayheadEnabled,
        paramValuePopupEnabled,
        onParamValuePreviewChange,
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
        edgeSmoothnessPercent,
        onPitchSnapGestureActiveChange,
        onMorphOverlayChange,
        currentParamRange,
    } = args;

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
            // 参数微调不再参与参数编辑器拖拽逻辑；此处保留空实现以复用既有拖拽收尾流程。
        },
        [],
    );

    const pointerFineWheelScale = useCallback(
        (ev: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey?: boolean }) =>
            isModifierActive(paramFineAdjustKb, ev as any) ? PARAM_FINE_WHEEL_SCALE : 1,
        [paramFineAdjustKb],
    );

    const createFineAdjustedPointerState = useCallback(
        (ev: FineAdjustedPointerInput, _dragTarget: HTMLCanvasElement | null = null) => {
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
                return Boolean(isModifierActive(noSnapKb, ev as any));
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

    const computeSelectionChangeFactor = useCallback(
        (
            beforeDense: number[],
            afterDense: number[],
            editedStartIdx: number,
            editedLen: number,
        ) => {
            if (editedLen <= 0) return 0;
            const maxIdx = Math.min(beforeDense.length, afterDense.length) - 1;
            if (maxIdx < 0) return 0;

            const startIdx = clamp(editedStartIdx, 0, maxIdx);
            const endIdx = clamp(editedStartIdx + editedLen - 1, startIdx, maxIdx);

            const calcMean = (arr: number[]) => {
                let sum = 0;
                let count = 0;
                for (let idx = startIdx; idx <= endIdx; idx += 1) {
                    const v = Number(arr[idx] ?? 0);
                    if (!Number.isFinite(v)) continue;
                    if (editParam === "pitch" && v === 0) continue;
                    sum += v;
                    count += 1;
                }
                return { sum, count };
            };

            const before = calcMean(beforeDense);
            const after = calcMean(afterDense);
            if (before.count <= 0 || after.count <= 0) return 0;

            const meanDelta = Math.abs(after.sum / after.count - before.sum / before.count);
            if (meanDelta <= 1e-9) return 0;

            let boundaryDelta = 0;
            let boundaryCount = 0;
            if (startIdx > 0) {
                boundaryDelta += Math.abs(
                    Number(beforeDense[startIdx] ?? 0) - Number(beforeDense[startIdx - 1] ?? 0),
                );
                boundaryCount += 1;
            }
            if (endIdx < maxIdx) {
                boundaryDelta += Math.abs(
                    Number(beforeDense[endIdx] ?? 0) - Number(beforeDense[endIdx + 1] ?? 0),
                );
                boundaryCount += 1;
            }
            const boundaryMean = boundaryCount > 0 ? boundaryDelta / boundaryCount : 0;
            return clamp(meanDelta / (meanDelta + boundaryMean + 1e-6), 0, 1);
        },
        [editParam],
    );

    const applyEdgeSmoothingToDense = useCallback(
        (editedDense: number[], editedStartIdx: number, editedLen: number, changeFactor = 1) => {
            const effectiveChange = clamp(Number(changeFactor) || 0, 0, 1);
            if (effectiveChange <= 0) return;
            const strength = clamp(Number(edgeSmoothnessPercent) || 0, 0, 100);
            if (strength <= 0 || editedLen <= 1) return;

            const maxTransitionFrames = Math.floor(editedLen / 2);
            if (maxTransitionFrames <= 0) return;

            const transitionFrames = Math.round((strength / 100) * maxTransitionFrames);
            if (transitionFrames <= 0) return;

            const snapshot = editedDense.slice();
            const maxIdx = snapshot.length - 1;
            if (maxIdx < 1) return;

            const startIdx = clamp(editedStartIdx, 0, maxIdx);
            const endIdx = clamp(editedStartIdx + editedLen - 1, 0, maxIdx);
            const halfSpan = transitionFrames / 2;
            if (halfSpan <= 0) return;

            // 左边界：以边界为中心，在 [P-half, P+half] 内从选区外过渡到选区内。
            if (startIdx > 0) {
                const left = Math.max(0, Math.floor(startIdx - halfSpan));
                const right = Math.min(maxIdx, Math.ceil(startIdx + halfSpan));
                const span = Math.max(1e-9, 2 * halfSpan);
                for (let idx = left; idx <= right; idx++) {
                    const t = clamp((idx - (startIdx - halfSpan)) / span, 0, 1);
                    const outsideIdx = Math.min(startIdx - 1, idx);
                    const insideIdx = Math.max(startIdx, idx);
                    const outsideVal = snapshot[outsideIdx] ?? editedDense[idx];
                    const insideVal = snapshot[insideIdx] ?? editedDense[idx];
                    const smoothed = outsideVal + (insideVal - outsideVal) * t;
                    editedDense[idx] = snapshot[idx] + (smoothed - snapshot[idx]) * effectiveChange;
                }
            }

            // 右边界：以边界为中心，在 [Q-half, Q+half] 内从选区内过渡到选区外。
            if (endIdx < maxIdx) {
                const left = Math.max(0, Math.floor(endIdx - halfSpan));
                const right = Math.min(maxIdx, Math.ceil(endIdx + halfSpan));
                const span = Math.max(1e-9, 2 * halfSpan);
                for (let idx = left; idx <= right; idx++) {
                    const t = clamp((idx - (endIdx - halfSpan)) / span, 0, 1);
                    const insideIdx = Math.min(endIdx, idx);
                    const outsideIdx = Math.max(endIdx + 1, idx);
                    const insideVal = snapshot[insideIdx] ?? editedDense[idx];
                    const outsideVal = snapshot[outsideIdx] ?? editedDense[idx];
                    const smoothed = insideVal + (outsideVal - insideVal) * t;
                    editedDense[idx] = snapshot[idx] + (smoothed - snapshot[idx]) * effectiveChange;
                }
            }
        },
        [edgeSmoothnessPercent],
    );

    const morphOverlayRef = useRef<ParamMorphOverlay | null>(null);
    const morphDragRef = useRef<{
        pointerId: number;
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
        (next: ParamMorphOverlay | null) => {
            morphOverlayRef.current = next;
            onMorphOverlayChange?.(next);
            invalidate();
        },
        [invalidate, onMorphOverlayChange],
    );

    const buildMorphOverlayFromSelection = useCallback((): ParamMorphOverlay | null => {
        const sel = selectionRef.current;
        const pv = paramViewRef.current;
        if (!sel || !pv || pv.edit.length === 0) return null;

        const aBeat = Math.min(sel.aBeat, sel.bBeat);
        const bBeat = Math.max(sel.aBeat, sel.bBeat);
        if (!Number.isFinite(aBeat) || !Number.isFinite(bBeat) || bBeat <= aBeat) return null;

        const fp = Math.max(1e-6, pv.framePeriodMs);
        const stride = Math.max(1, pv.stride);
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
        if (baselineValues.length === 0) return null;

        const valid =
            editParam === "pitch" ? baselineValues.filter((v) => Number(v) !== 0) : baselineValues;
        const meanValue =
            valid.length > 0
                ? valid.reduce((sum, v) => sum + (Number(v) || 0), 0) / valid.length
                : 0;

        const selectionStartFrame = pv.startFrame + selStartIdx * stride;
        const selectionEndFrame = pv.startFrame + selEndIdx * stride;
        const span = Math.max(0, selectionEndFrame - selectionStartFrame);
        const p1 = Math.round(selectionStartFrame + span / 3);
        const p2 = Math.round(selectionStartFrame + (span * 2) / 3);

        return {
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
        };
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
        (overlay: ParamMorphOverlay) => {
            const pv = paramViewRef.current;
            if (!pv) return;
            ensureLiveEditBase(pv);
            const packed = buildMorphDense(overlay, pv.stride);
            applyDenseToLiveEdit(
                pv,
                packed.startFrame,
                packed.dense,
                packed.startFrame,
                packed.endFrame,
                "draw",
            );
        },
        [applyDenseToLiveEdit, buildMorphDense, ensureLiveEditBase, paramViewRef],
    );

    const applyPostStrokeSmoothing = useCallback(
        async (points: StrokePoint[], mode: StrokeMode) => {
            if (mode !== "draw") return;
            const trackId = rootTrackId;
            if (!trackId || points.length === 0) return;

            const strengthPercent = clamp(Number(edgeSmoothnessPercent) || 0, 0, 100);
            if (strengthPercent <= 0) return;

            let minF = Number.POSITIVE_INFINITY;
            let maxF = 0;
            for (const p of points) {
                const f = Math.max(0, Math.floor(Number(p.frame) || 0));
                minF = Math.min(minF, f);
                maxF = Math.max(maxF, f);
            }
            if (!Number.isFinite(minF) || maxF < minF) return;

            const editedLen = maxF - minF + 1;
            const extend = Math.max(1, Math.ceil(editedLen * 0.01));
            const smoothStart = Math.max(0, minF - extend);
            const smoothEnd = maxF + extend;
            const smoothCount = smoothEnd - smoothStart + 1;

            const res = await paramsApi.getParamFrames(
                trackId,
                editParam,
                smoothStart,
                smoothCount,
                1,
            );
            if (!res?.ok) return;
            const payload = res as ParamFramesPayload;
            const vals = (payload.edit ?? []).map((v) => Number(v) || 0);
            if (vals.length === 0) return;

            const strength = strengthPercent / 100;
            const radius = Math.max(1, Math.round(strength * 50));
            const passes = Math.max(1, Math.round(strength * 3));
            let buf = vals.slice();
            for (let p = 0; p < passes; p += 1) {
                const next = new Array<number>(buf.length);
                for (let i = 0; i < buf.length; i += 1) {
                    const lo = Math.max(0, i - radius);
                    const hi = Math.min(buf.length - 1, i + radius);
                    let sum = 0;
                    let count = 0;
                    for (let j = lo; j <= hi; j += 1) {
                        if (editParam === "pitch" && vals[j] === 0) continue;
                        sum += buf[j];
                        count += 1;
                    }
                    next[i] =
                        editParam === "pitch" && vals[i] === 0
                            ? 0
                            : count > 0
                              ? sum / count
                              : buf[i];
                }
                buf = next;
            }

            const smoothed = vals.map((v, i) =>
                editParam === "pitch" && v === 0 ? 0 : v + (buf[i] - v) * strength,
            );

            await paramsApi.setParamFrames(trackId, editParam, smoothStart, smoothed, false);

            const pvNow = paramViewRef.current;
            if (pvNow) {
                const nextEdit = pvNow.edit.slice();
                const stride = Math.max(1, pvNow.stride);
                for (let i = 0; i < smoothed.length; i += 1) {
                    const frame = smoothStart + i;
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

    const pitchDeltaToDegreeSteps = useCallback(
        (basePitch: number, targetPitch: number, scale: ScaleLike): number => {
            if (!Number.isFinite(basePitch) || !Number.isFinite(targetPitch)) {
                return 0;
            }
            if (Math.abs(targetPitch - basePitch) <= 1e-9) return 0;

            const minStep = Number(CHILD_PITCH_OFFSET_DEGREES_RANGE.min);
            const maxStep = Number(CHILD_PITCH_OFFSET_DEGREES_RANGE.max);
            const minPitch = transposePitchByScaleSteps(basePitch, minStep, scale);
            const maxPitch = transposePitchByScaleSteps(basePitch, maxStep, scale);
            const lowPitch = Math.min(minPitch, maxPitch);
            const highPitch = Math.max(minPitch, maxPitch);
            if (targetPitch <= lowPitch) {
                return minPitch <= maxPitch ? minStep : maxStep;
            }
            if (targetPitch >= highPitch) {
                return minPitch <= maxPitch ? maxStep : minStep;
            }

            let left = minStep;
            let right = maxStep;
            const ascending = minPitch <= maxPitch;
            for (let i = 0; i < 24; i += 1) {
                const mid = (left + right) / 2;
                const midPitch = transposePitchByScaleSteps(basePitch, mid, scale);
                if (midPitch < targetPitch === ascending) {
                    left = mid;
                } else {
                    right = mid;
                }
            }
            return (left + right) / 2;
        },
        [],
    );

    const buildChildOffsetPasteValues = useCallback(
        async (
            targetTrackId: string,
            startFrame: number,
            frameCount: number,
            clipboardPitch: number[],
            mode: "cents" | "degrees",
        ): Promise<number[] | null> => {
            const fp = paramViewRef.current?.framePeriodMs;
            return buildChildOffsetPasteValuesHelper({
                tracks,
                rootTrackId,
                targetTrackId,
                startFrame,
                frameCount,
                clipboardPitch,
                mode,
                paramsApi,
                pitchDeltaToDegreeSteps,
                projectScale,
                scaleAtFrame:
                    fp != null
                        ? (frame: number) => scaleAtSec?.((frame * fp) / 1000) ?? projectScale
                        : undefined,
            });
        },
        // paramViewRef is a ref; it is intentionally not a dependency.
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [tracks, pitchDeltaToDegreeSteps, projectScale, scaleAtSec, rootTrackId],
    );

    const updateSelectionUi = useCallback(
        (next: { aBeat: number; bBeat: number } | null) => {
            setSelectionUi(next);
            if (morphModifierDownRef.current && !morphDragRef.current) {
                setMorphOverlay(buildMorphOverlayFromSelection());
            }
        },
        [buildMorphOverlayFromSelection, setMorphOverlay, setSelectionUi],
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
                isModifierActive(paramMorphKb, e as any);
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
                setMorphOverlay(buildMorphOverlayFromSelection());
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
        buildMorphOverlayFromSelection,
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
        setMorphOverlay(buildMorphOverlayFromSelection());
    }, [buildMorphOverlayFromSelection, selectionUi, setMorphOverlay, toolMode]);

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
            const x = clientX - rect.left;
            const sl = scrollLeftRef.current;
            const ppb = pxPerBeatRef.current;
            return (sl + x) / Math.max(1e-9, ppb);
        },
        [canvasRef, scrollLeftRef, pxPerBeatRef],
    );

    const pointerSec = useCallback(
        (clientX: number): number => {
            const canvas = canvasRef.current;
            if (!canvas) return 0;
            const rect = canvas.getBoundingClientRect();
            return secFromViewportClientX({
                clientX,
                viewportLeft: rect.left,
                scrollLeft: scrollLeftRef.current,
                pxPerSec: pxPerSecRef.current,
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
            let lastSec = 0;

            const updateAt = (clientX: number, commit: boolean): number => {
                const bounds = ruler.getBoundingClientRect();
                const sec = clamp(
                    secFromViewportClientX({
                        clientX,
                        viewportLeft: bounds.left,
                        scrollLeft: scrollLeftRef.current,
                        pxPerSec: pxPerSecRef.current,
                    }),
                    0,
                    1e12,
                );

                dispatch(setplayheadSec(sec));
                if (commit) {
                    void dispatch(seekPlayhead(sec));
                }
                return sec;
            };

            // 标尺没有其他编辑操作需要区分，按下时立即提交一次 seek。
            lastSec = updateAt(e.clientX, true);

            const onMove = (ev: MouseEvent) => {
                moved = true;
                lastSec = updateAt(ev.clientX, false);
            };

            const onEnd = (ev: MouseEvent) => {
                window.removeEventListener("mousemove", onMove, true);
                window.removeEventListener("mouseup", onEnd, true);
                window.removeEventListener("mouseleave", onEnd, true);
                if (!moved) return;
                lastSec = updateAt(ev.clientX, false);
                void dispatch(seekPlayhead(lastSec));
            };

            window.addEventListener("mousemove", onMove, true);
            window.addEventListener("mouseup", onEnd, true);
            window.addEventListener("mouseleave", onEnd, true);
        },
        [dispatch, scrollLeftRef, pxPerSecRef],
    );

    const onScrollerMouseDownCapture = useCallback((e: ReactMouseEvent) => {
        if (e.button === 1) e.preventDefault();
    }, []);

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
            if (args.onContextMenu) {
                args.onContextMenu(e.clientX, e.clientY);
            }
        },
        [args.onContextMenu],
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

            if (!selectionRef.current) return;

            const sel = selectionRef.current;
            const aBeat = Math.min(sel.aBeat, sel.bBeat);
            const bBeat = Math.max(sel.aBeat, sel.bBeat);
            const startSec = aBeat * secPerBeat;
            const durSec = Math.max(0, (bBeat - aBeat) * secPerBeat);
            const fp = paramView?.framePeriodMs ?? 5;
            const startFrame = Math.max(0, Math.floor((startSec * 1000) / fp));
            const frameCount = clamp(Math.ceil((durSec * 1000) / fp), 1, 200_000);

            // 检测 pianoRoll.copy 绑定
            {
                const kb = pianoRollCopyKb;
                let keyMatch = false;
                if (kb.modifierOnly) {
                    keyMatch = isModifierActive(kb, e.nativeEvent);
                } else {
                    let pressedKey = e.key === " " ? "space" : e.key.toLowerCase();
                    if (pressedKey !== kb.key) keyMatch = false;
                    else {
                        const isMac = navigator.platform.toLowerCase().includes("mac");
                        const modKey = isMac ? e.metaKey : e.ctrlKey;
                        keyMatch =
                            modKey === Boolean(kb.ctrl) &&
                            e.shiftKey === Boolean(kb.shift) &&
                            e.altKey === Boolean(kb.alt);
                    }
                }
                if (keyMatch) {
                    e.preventDefault();
                    void (async () => {
                        const res = await paramsApi.getParamFrames(
                            rootTrackId,
                            editParam,
                            startFrame,
                            frameCount,
                            1,
                        );
                        if (!res?.ok) return;
                        const payload = res as ParamFramesPayload;
                        clipboardRef.current = {
                            param: editParam,
                            framePeriodMs: Number(payload.frame_period_ms ?? fp) || fp,
                            values: (payload.edit ?? []).map((v) => Number(v) || 0),
                        };
                        try {
                            await writeSystemClipboardObject({
                                version: 1,
                                kind: "param",
                                param: editParam,
                                framePeriodMs: Number(payload.frame_period_ms ?? fp) || fp,
                                values: (payload.edit ?? []).map((v) => Number(v) || 0),
                            });
                        } catch {
                            // ignore clipboard write failures
                        }
                        // 刷新剪贴板预览
                        invalidate();
                    })();
                    return;
                }
            }

            // 检测 pianoRoll.paste 绑定
            {
                const kb = pianoRollPasteKb;
                let keyMatch = false;
                if (kb.modifierOnly) {
                    keyMatch = isModifierActive(kb, e.nativeEvent);
                } else {
                    let pressedKey = e.key === " " ? "space" : e.key.toLowerCase();
                    if (pressedKey !== kb.key) keyMatch = false;
                    else {
                        const isMac = navigator.platform.toLowerCase().includes("mac");
                        const modKey = isMac ? e.metaKey : e.ctrlKey;
                        keyMatch =
                            modKey === Boolean(kb.ctrl) &&
                            e.shiftKey === Boolean(kb.shift) &&
                            e.altKey === Boolean(kb.alt);
                    }
                }
                if (keyMatch) {
                    e.preventDefault();
                    void (async () => {
                        let clip = clipboardRef.current;
                        try {
                            const fromSystem = await readSystemClipboardObject("param");
                            if (fromSystem?.kind === "param") {
                                clip = {
                                    param: fromSystem.param,
                                    framePeriodMs: Number(fromSystem.framePeriodMs) || fp,
                                    values: Array.isArray(fromSystem.values)
                                        ? fromSystem.values.map((v) => Number(v) || 0)
                                        : [],
                                };
                                clipboardRef.current = clip;
                            }
                        } catch {
                            // ignore and fallback to internal clipboard
                        }
                        if (!clip) return;
                        const targetIsChildCents = isChildPitchOffsetCentsParam(editParam);
                        const targetIsChildDegrees = isChildPitchOffsetDegreesParam(editParam);
                        const canConvertPitchToChildOffset =
                            (targetIsChildCents || targetIsChildDegrees) &&
                            clip.param === "pitch" &&
                            selectedTrackId != null;

                        let pasteValues: number[];
                        if (clip.param === editParam) {
                            pasteValues =
                                clip.values.length > frameCount
                                    ? clip.values.slice(0, frameCount)
                                    : clip.values;
                        } else if (canConvertPitchToChildOffset) {
                            const converted = await buildChildOffsetPasteValues(
                                selectedTrackId!,
                                startFrame,
                                frameCount,
                                clip.values.length > frameCount
                                    ? clip.values.slice(0, frameCount)
                                    : clip.values,
                                targetIsChildCents ? "cents" : "degrees",
                            );
                            if (!converted) return;
                            pasteValues = converted;
                        } else {
                            return;
                        }

                        await paramsApi.setParamFrames(
                            rootTrackId,
                            editParam,
                            startFrame,
                            pasteValues,
                            true,
                        );
                        bumpRefreshToken();
                    })();
                }
            }
        },
        [
            rootTrackId,
            selectionRef,
            secPerBeat,
            paramView?.framePeriodMs,
            editParam,
            pitchEnabled,
            clipboardRef,
            bumpRefreshToken,
            pianoRollCopyKb,
            pianoRollPasteKb,
            keybindingMap,
            onEditAction,
            toolMode,
            selectedTrackId,
            buildChildOffsetPasteValues,
        ],
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
            const horizontalScrollModifierActive = isWheelBindingRequested(scrollHorizontalKb);
            const wheelAction = getParamEditorWheelAction({
                deltaX: e.deltaX,
                deltaY: e.deltaY,
                horizontalScrollRequested: horizontalScrollModifierActive,
                verticalPanRequested: isWheelBindingRequested(scrollVerticalKb),
                verticalZoomRequested: isWheelBindingRequested(prVerticalZoomKb),
                horizontalZoomRequested: isWheelBindingRequested(horizontalZoomKb),
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
                playheadSec: playheadSec ?? null,
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
            horizontalZoomKb,
            vibratoAmplitudeAdjustKb,
            vibratoFrequencyAdjustKb,
            applyVibratoDragAdjustment,
            dynamicProjectSec,
            playheadSec,
            playheadZoomEnabled,
            currentParamRange,
        ],
    );

    // React's onWheel handler may run in a passive listener in modern React.
    // Keep this for compatibility, but do not call preventDefault here.
    const onScrollerWheel = useCallback((_e: WheelEvent<HTMLDivElement>) => {
        // no-op; wheel is handled via native listener with passive:false
    }, []);

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

    const getCurveValueNearPointer = useCallback(
        (clientX: number, clientY: number): number | null => {
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

            const rect = canvas.getBoundingClientRect();
            const rectH = rect.height || viewSizeRef.current.h || 1;
            const mouseY = clientY - rect.top;
            const mappedCurveVal = editParam === "pitch" ? curveVal + 0.5 : curveVal;
            const curveY = valueToY(editParam, mappedCurveVal, rectH);
            return Math.abs(mouseY - curveY) < 10 ? curveVal : null;
        },
        [paramViewRef, canvasRef, pointerBeat, secPerBeat, viewSizeRef, editParam, valueToY],
    );

    const isPointerNearDraggableSelection = useCallback(
        (clientX: number, clientY: number): boolean => {
            if (toolMode !== "select") return false;
            const sel = selectionRef.current;
            if (!sel) return false;

            const beat = pointerBeat(clientX);
            const aBeat = Math.min(sel.aBeat, sel.bBeat);
            const bBeat = Math.max(sel.aBeat, sel.bBeat);
            if (beat < aBeat || beat > bBeat) return false;

            return getCurveValueNearPointer(clientX, clientY) != null;
        },
        [toolMode, selectionRef, pointerBeat, getCurveValueNearPointer],
    );

    const isPointerNearStretchSelectionEdge = useCallback(
        (e: ReactPointerEvent<HTMLCanvasElement>): boolean => {
            if (toolMode !== "select") return false;
            if (!isModifierActive(paramStretchKb, e.nativeEvent as any)) return false;
            const sel = selectionRef.current;
            const canvas = canvasRef.current;
            if (!sel || !canvas) return false;
            const aBeat = Math.min(sel.aBeat, sel.bBeat);
            const bBeat = Math.max(sel.aBeat, sel.bBeat);
            const rect = canvas.getBoundingClientRect();
            const leftX = aBeat * pxPerBeatRef.current - scrollLeftRef.current;
            const rightX = bBeat * pxPerBeatRef.current - scrollLeftRef.current;
            const localX = e.clientX - rect.left;
            const edgeHitPx = 8;
            return Math.abs(localX - leftX) <= edgeHitPx || Math.abs(localX - rightX) <= edgeHitPx;
        },
        [toolMode, paramStretchKb, selectionRef, canvasRef, pxPerBeatRef, scrollLeftRef],
    );

    const onCanvasPointerMove = useCallback(
        (e: ReactPointerEvent<HTMLCanvasElement>) => {
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
            setCanvasCursor,
            getDefaultCanvasCursor,
        ],
    );

    const onCanvasPointerLeave = useCallback(() => {
        if (panRef.current || strokeRef.current) return;
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
                if (existingMorph && pvForMorph && canvas) {
                    const rect = canvas.getBoundingClientRect();
                    const h = Math.max(1, rect.height || viewSizeRef.current.h || 1);
                    const fp = Math.max(1e-6, pvForMorph.framePeriodMs);
                    const stride = Math.max(1, pvForMorph.stride);
                    const hit = existingMorph.points.find((p) => {
                        const sec = (p.frame * fp) / 1000;
                        const beat = sec / secPerBeat;
                        const x = beat * pxPerBeatRef.current - scrollLeftRef.current;
                        const mapped = editParam === "pitch" ? p.value + 0.5 : p.value;
                        const y = valueToY(editParam, mapped, h);
                        return (
                            Math.abs(e.clientX - rect.left - x) <= 8 &&
                            Math.abs(e.clientY - rect.top - y) <= 8
                        );
                    });

                    if (hit && e.button === 0) {
                        e.preventDefault();
                        morphDragRef.current = {
                            pointerId: e.pointerId,
                            pointKind: hit.kind,
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
                                value: hit.value,
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
                            const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);

                            const nextPoints = overlayNow.points.map((pt) => {
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
                                    overlayNow.selectionStartFrame,
                                    overlayNow.selectionEndFrame,
                                );
                                return {
                                    ...pt,
                                    frame: clampedFrame,
                                    value: newValue,
                                };
                            });

                            const nextOverlay: ParamMorphOverlay = {
                                ...overlayNow,
                                points: nextPoints,
                            };
                            setMorphOverlay(nextOverlay);
                            applyMorphOverlayPreview(nextOverlay);

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

                            const packed = buildMorphDense(overlayNow, stride);

                            const nextEdit = pvNow.edit.slice();
                            for (let i = 0; i < packed.dense.length; i += 1) {
                                const frame = packed.startFrame + i * stride;
                                const idx = Math.round((frame - pvNow.startFrame) / stride);
                                if (idx >= 0 && idx < nextEdit.length) {
                                    nextEdit[idx] = packed.dense[i];
                                }
                            }
                            setParamView({ ...pvNow, edit: nextEdit });

                            void (async () => {
                                await paramsApi.setParamFrames(
                                    rootTrackId,
                                    editParam,
                                    packed.startFrame,
                                    packed.dense,
                                    true,
                                );
                                liveEditOverrideRef.current = null;
                                if (liveEditActiveRef) liveEditActiveRef.current = false;
                                bumpRefreshToken();
                            })();

                            if (morphModifierDownRef.current) {
                                // 保持调整点位置不变；baselineValues 是进入形变模式时一次性捕获的，
                                // 每次拖拽提交都在同一基准线上施加"总偏移"，不重置。
                                // 只有松开修饰键再重新按下时，才会调用 buildMorphOverlayFromSelection 重置。
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

                // 如果已有选区，且鼠标在选区范围内且靠近曲线，则进入拖拽曲线模式
                if (sel) {
                    const aBeat = Math.min(sel.aBeat, sel.bBeat);
                    const bBeat = Math.max(sel.aBeat, sel.bBeat);

                    if (isModifierActive(paramStretchKb, e.nativeEvent as any)) {
                        const canvas = canvasRef.current;
                        if (canvas) {
                            const rect = canvas.getBoundingClientRect();
                            const leftX = aBeat * pxPerBeatRef.current - scrollLeftRef.current;
                            const rightX = bBeat * pxPerBeatRef.current - scrollLeftRef.current;
                            const localX = e.clientX - rect.left;
                            const EDGE_HIT_PX = 8;
                            const hitLeft = Math.abs(localX - leftX) <= EDGE_HIT_PX;
                            const hitRight = Math.abs(localX - rightX) <= EDGE_HIT_PX;
                            const edgeKind: "left" | "right" | null = hitLeft
                                ? "left"
                                : hitRight
                                  ? "right"
                                  : null;

                            if (edgeKind) {
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

                                    const edgeSmoothStr = clamp(
                                        Number(edgeSmoothnessPercent) || 0,
                                        0,
                                        100,
                                    );
                                    const edgeHalfSpanIdx = Math.ceil(
                                        Math.round(
                                            (edgeSmoothStr / 100) * Math.floor(nextLen / 2),
                                        ) / 2,
                                    );
                                    const extraEdgeFrames = edgeHalfSpanIdx * stride;
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
                                    const changeFactor = computeSelectionChangeFactor(
                                        denseBefore,
                                        dense,
                                        movedStartDenseIdx,
                                        nextLen,
                                    );
                                    applyEdgeSmoothingToDense(
                                        dense,
                                        movedStartDenseIdx,
                                        nextLen,
                                        changeFactor,
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

                                const onMove = (ev: globalThis.PointerEvent) => {
                                    if ((ev.buttons & 1) !== 1) {
                                        onUp();
                                        return;
                                    }
                                    const pvNow = paramViewRef.current;
                                    if (!pvNow) return;
                                    const adjusted = getFineAdjustedPointerPosition(
                                        finePointerState,
                                        ev,
                                    );
                                    const cursorBeat = pointerBeat(adjusted.clientX);
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
                                    applyDenseToLiveEdit(
                                        pvNow,
                                        built.overallMinFrame,
                                        built.dense,
                                        built.overallMinFrame,
                                        built.overallMaxFrame,
                                        "draw",
                                    );
                                    selectionRef.current = {
                                        aBeat: built.nextABeat,
                                        bBeat: built.nextBBeat,
                                    };
                                    updateSelectionUi(selectionRef.current);
                                    invalidate();
                                };

                                const onUp = () => {
                                    window.removeEventListener("pointermove", onMove);
                                    window.removeEventListener("pointerup", onUp);
                                    window.removeEventListener("pointercancel", onUp);
                                    disposeFineAdjustedPointerState(finePointerState);
                                    clearActivePointerGestureEnd(onUp);

                                    const pvNow = paramViewRef.current;
                                    const selNow = selectionRef.current;
                                    if (!pvNow || !selNow || !rootTrackId) {
                                        setCanvasCursor("default");
                                        return;
                                    }
                                    const nextABeat = Math.min(selNow.aBeat, selNow.bBeat);
                                    const nextBBeat = Math.max(selNow.aBeat, selNow.bBeat);
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

                                    void (async () => {
                                        await paramsApi.setParamFrames(
                                            rootTrackId,
                                            editParam,
                                            built.overallMinFrame,
                                            built.dense,
                                            true,
                                        );
                                        if (liveEditActiveRef) liveEditActiveRef.current = false;
                                        bumpRefreshToken();
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
                        }
                    }

                    if (b >= aBeat && b <= bBeat) {
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

                                    const selStartSec = aBeat * secPerBeat;
                                    const selEndSec = bBeat * secPerBeat;
                                    const selStartFrame = Math.max(
                                        0,
                                        Math.floor((selStartSec * 1000) / fp),
                                    );
                                    const selEndFrame = Math.max(
                                        0,
                                        Math.ceil((selEndSec * 1000) / fp),
                                    );
                                    const stride = Math.max(1, pv.stride);
                                    const selStartIdx = Math.max(
                                        0,
                                        Math.round((selStartFrame - pv.startFrame) / stride),
                                    );
                                    const selEndIdx = Math.min(
                                        pv.edit.length - 1,
                                        Math.round((selEndFrame - pv.startFrame) / stride),
                                    );
                                    const origValues = pv.edit.slice(selStartIdx, selEndIdx + 1);

                                    let didDrag = false;
                                    let latestDense = origValues.slice();
                                    let latestAppliedStartFrame = selStartFrame;
                                    let latestAppliedDense = origValues.slice();

                                    ensureLiveEditBase(pv);
                                    if (liveEditActiveRef) liveEditActiveRef.current = true;

                                    const buildRightDragDense = (
                                        pvNow: ParamViewSegment,
                                        nextSelectionValues: number[],
                                    ) => {
                                        const selLen = nextSelectionValues.length;
                                        const maxTransitionFrames = Math.floor(selLen / 2);
                                        const transitionFrames =
                                            Number(edgeSmoothnessPercent) > 0 &&
                                            maxTransitionFrames > 0
                                                ? Math.round(
                                                      (clamp(
                                                          Number(edgeSmoothnessPercent) || 0,
                                                          0,
                                                          100,
                                                      ) /
                                                          100) *
                                                          maxTransitionFrames,
                                                  )
                                                : 0;
                                        const extraEdgeFrames =
                                            Math.max(0, Math.ceil(transitionFrames / 2)) * stride;
                                        const denseStartFrame = Math.max(
                                            0,
                                            selStartFrame - extraEdgeFrames,
                                        );
                                        const denseEndFrame = selEndFrame + extraEdgeFrames;
                                        const denseLength =
                                            Math.floor((denseEndFrame - denseStartFrame) / stride) +
                                            1;
                                        const dense = new Array<number>(denseLength);

                                        for (let index = 0; index < denseLength; index += 1) {
                                            const globalIdx = Math.round(
                                                (denseStartFrame +
                                                    index * stride -
                                                    pvNow.startFrame) /
                                                    stride,
                                            );
                                            dense[index] =
                                                globalIdx >= 0 && globalIdx < pvNow.edit.length
                                                    ? pvNow.edit[globalIdx]
                                                    : 0;
                                        }

                                        const denseBefore = dense.slice();
                                        const selectionStartDenseIdx = Math.round(
                                            (selStartFrame - denseStartFrame) / stride,
                                        );

                                        for (
                                            let index = 0;
                                            index < nextSelectionValues.length;
                                            index += 1
                                        ) {
                                            const denseIdx = selectionStartDenseIdx + index;
                                            if (denseIdx >= 0 && denseIdx < dense.length) {
                                                dense[denseIdx] = nextSelectionValues[index] ?? 0;
                                            }
                                        }

                                        const changeFactor = computeSelectionChangeFactor(
                                            denseBefore,
                                            dense,
                                            selectionStartDenseIdx,
                                            selLen,
                                        );
                                        applyEdgeSmoothingToDense(
                                            dense,
                                            selectionStartDenseIdx,
                                            selLen,
                                            changeFactor,
                                        );

                                        return {
                                            denseStartFrame,
                                            denseEndFrame,
                                            dense,
                                        };
                                    };

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

                                        latestDense = transformSelectionByRightDrag(
                                            origValues,
                                            editParam,
                                            dy,
                                        );

                                        const pvNow = paramViewRef.current;
                                        if (!pvNow) return;
                                        const nextApplied = buildRightDragDense(pvNow, latestDense);
                                        latestAppliedStartFrame = nextApplied.denseStartFrame;
                                        latestAppliedDense = nextApplied.dense;
                                        applyDenseToLiveEdit(
                                            pvNow,
                                            nextApplied.denseStartFrame,
                                            nextApplied.dense,
                                            nextApplied.denseStartFrame,
                                            nextApplied.denseEndFrame,
                                            "draw",
                                        );
                                        if (paramValuePopupEnabled) {
                                            onParamValuePreviewChange?.({
                                                clientX: ev.clientX,
                                                clientY: ev.clientY,
                                                value: dy,
                                                displayText: formatRightDragMorphPercent(dy),
                                            });
                                        }
                                        invalidate();
                                    };

                                    const onUp = (ev?: globalThis.PointerEvent) => {
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

                                        if (!didDrag) {
                                            liveEditOverrideRef.current = null;
                                            if (liveEditActiveRef) {
                                                liveEditActiveRef.current = false;
                                            }
                                            setCanvasCursor("grab");
                                            if (args.onContextMenu && document.hasFocus() && ev) {
                                                args.onContextMenu(ev.clientX, ev.clientY);
                                            }
                                            invalidate();
                                            return;
                                        }

                                        const pvNow = paramViewRef.current;
                                        if (!pvNow || !rootTrackId) {
                                            if (liveEditActiveRef) {
                                                liveEditActiveRef.current = false;
                                            }
                                            setCanvasCursor("grab");
                                            invalidate();
                                            return;
                                        }

                                        const nextEdit = pvNow.edit.slice();
                                        for (let i = 0; i < latestAppliedDense.length; i += 1) {
                                            const frame = latestAppliedStartFrame + i * stride;
                                            const idx = Math.round(
                                                (frame - pvNow.startFrame) / stride,
                                            );
                                            if (idx >= 0 && idx < nextEdit.length) {
                                                nextEdit[idx] = latestAppliedDense[i] ?? 0;
                                            }
                                        }
                                        setParamView({ ...pvNow, edit: nextEdit });
                                        liveEditOverrideRef.current = null;

                                        void (async () => {
                                            await paramsApi.setParamFrames(
                                                rootTrackId,
                                                editParam,
                                                latestAppliedStartFrame,
                                                latestAppliedDense,
                                                true,
                                            );
                                            if (liveEditActiveRef) {
                                                liveEditActiveRef.current = false;
                                            }
                                            bumpRefreshToken();
                                        })();

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
                                setCanvasCursor("grabbing");
                                const startMouseVal = mouseVal;
                                const startBeat = pointerBeat(e.clientX);
                                const pid = e.pointerId;
                                (e.currentTarget as HTMLCanvasElement).setPointerCapture(pid);
                                const finePointerState = createFineAdjustedPointerState(
                                    e.nativeEvent,
                                    e.currentTarget as HTMLCanvasElement,
                                );

                                // 选区帧范围 —— 注意这里**不**夹到 pv 的已加载窗口内。
                                // 「全选」时选区覆盖整个工程，若按 pv 裁剪，后续只会改到
                                // 显示范围内的参数，工程其余部分纹丝不动（本次修复的 bug）。
                                const selStartSec = aBeat * secPerBeat;
                                const selEndSec = bBeat * secPerBeat;
                                const selStartFrame = Math.max(
                                    0,
                                    Math.floor((selStartSec * 1000) / fp),
                                );
                                const selEndFrame = Math.max(0, Math.ceil((selEndSec * 1000) / fp));

                                // 取选区的全分辨率原始值。
                                // pv 在低缩放下按画布宽度做了降采样，直接拿去变换再回写
                                // 会把全分辨率曲线覆盖掉，所以变换的输入必须是 stride=1
                                // 的数据。pv 已以 stride=1 覆盖时零开销直接切片，行为与
                                // 改动前一致；否则分块向后端拉取（不阻塞拖动）。
                                const origCurvePromise = fetchFullResCurve({
                                    trackId: rootTrackId ?? "",
                                    param: editParam,
                                    startFrame: selStartFrame,
                                    endFrame: selEndFrame,
                                    paramView: pv,
                                });
                                // 先用 pv 覆盖到的部分做即时预览；全分辨率数据到位后自动
                                // 替换，之后的每一帧预览都基于无损数据。
                                let origValues = readPvRange(pv, selStartFrame, selEndFrame);
                                let dragSettled = false;
                                void origCurvePromise.then((curve) => {
                                    if (dragSettled) return;
                                    origValues = curve.values;
                                });

                                // 预览取数：从 pv 读当前值（pv 可能降采样，仅用于即时反馈）。
                                const makePvSourceAt = (src: typeof pv) => {
                                    const step = Math.max(1, Math.floor(src.stride));
                                    return (frame: number) => {
                                        const idx = Math.round((frame - src.startFrame) / step);
                                        return idx >= 0 && idx < src.edit.length
                                            ? src.edit[idx]
                                            : 0;
                                    };
                                };

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

                                // Tempo Map 感知：度数差以拖动锚点帧的生效音阶计算。
                                const dragAnchorFrame = selStartFrame;
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

                                    const pvNow = paramViewRef.current;
                                    if (!pvNow) return;

                                    // Reset live overlay before each move to prevent stale values
                                    // from the previous drag position lingering outside the current range
                                    liveEditOverrideRef.current = null;
                                    ensureLiveEditBase(pvNow);

                                    // 构造覆盖原选区 + 新位置的 dense 数组（逐帧索引）。
                                    // 预览阶段从 pv 取上下文 —— pv 只是显示数据，不会回写。
                                    const selLen = origValues.length;
                                    if (selLen === 0) return;

                                    // 边缘平滑度：扩展范围以包含选区边界外侧上下文
                                    // halfSpan 与 applyEdgeSmoothingToDense 内的计算保持一致
                                    const edgeSmoothStr = clamp(
                                        Number(edgeSmoothnessPercent) || 0,
                                        0,
                                        100,
                                    );
                                    const extraEdgeFrames = Math.ceil(
                                        Math.round((edgeSmoothStr / 100) * Math.floor(selLen / 2)) /
                                            2,
                                    );

                                    const built = buildSelectionDragDense({
                                        sourceAt: makePvSourceAt(pvNow),
                                        origValues,
                                        origStartFrame: selStartFrame,
                                        frameDelta: lastFrameDelta,
                                        extraEdgeFrames,
                                        transform: (orig, frame) => transformDragValue(orig, frame),
                                        computeChangeFactor: computeSelectionChangeFactor,
                                        applyEdgeSmoothing: applyEdgeSmoothingToDense,
                                    });

                                    applyDenseToLiveEdit(
                                        pvNow,
                                        built.startFrame,
                                        built.values,
                                        built.startFrame,
                                        built.endFrame,
                                        "draw",
                                    );

                                    // 实时更新选区位置显示
                                    const beatDeltaForSel =
                                        (lastFrameDelta * fp) / 1000 / secPerBeat;
                                    selectionRef.current = {
                                        aBeat: aBeat + beatDeltaForSel,
                                        bBeat: bBeat + beatDeltaForSel,
                                    };
                                    updateSelectionUi(selectionRef.current);

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

                                    invalidate();
                                };

                                const onUp = async () => {
                                    window.removeEventListener("pointermove", onMove);
                                    window.removeEventListener("pointerup", onUp);
                                    window.removeEventListener("pointercancel", onUp);
                                    disposeFineAdjustedPointerState(finePointerState);
                                    clearActivePointerGestureEnd(onUp);
                                    // 标记手势结束，避免已发出的取数请求再覆写 origValues
                                    dragSettled = true;

                                    // 提交拖拽结果到后端
                                    const pvNow = paramViewRef.current;
                                    if (pvNow && rootTrackId) {
                                        // 等待拖动开始时发起的全分辨率取数完成 —— 提交必须
                                        // 基于无损数据，不能拿降采样的 pv 值去覆盖后端。
                                        origValues = (await origCurvePromise).values;
                                        const selLen = origValues.length;
                                        if (selLen > 0) {
                                            // 边缘平滑度：扩展 dense 范围以包含选区边界外侧上下文
                                            const edgeSmoothStrUp = clamp(
                                                Number(edgeSmoothnessPercent) || 0,
                                                0,
                                                100,
                                            );
                                            const extraEdgeFramesUp = Math.ceil(
                                                Math.round(
                                                    (edgeSmoothStrUp / 100) *
                                                        Math.floor(selLen / 2),
                                                ) / 2,
                                            );

                                            // 提交前把整段范围的全分辨率数据拉下来作为基底。
                                            // 这是「回写不失真」的关键：预览用的是可能降采样的
                                            // pv，提交必须用 stride=1 的真实曲线，否则会把
                                            // 降采样后的值写回后端、覆盖掉原始分辨率。
                                            const upRange = selectionDragRange({
                                                origStartFrame: selStartFrame,
                                                origValuesLength: selLen,
                                                frameDelta: lastFrameDelta,
                                                extraEdgeFrames: extraEdgeFramesUp,
                                            });
                                            // pv 若已以 stride=1 覆盖该范围则直接切片（零 IPC）；
                                            // 否则分块拉取。pvNow.edit 是拖动前的值，正是回写
                                            // 所需的「当前后端值」。
                                            const commitBase = await fetchFullResCurve({
                                                trackId: rootTrackId,
                                                param: editParam,
                                                startFrame: upRange.startFrame,
                                                endFrame: upRange.endFrame,
                                                paramView: pvNow,
                                            });

                                            const built = buildSelectionDragDense({
                                                sourceAt: (frame) =>
                                                    commitBase.values[
                                                        frame - commitBase.startFrame
                                                    ] ?? 0,
                                                origValues,
                                                origStartFrame: selStartFrame,
                                                frameDelta: lastFrameDelta,
                                                extraEdgeFrames: extraEdgeFramesUp,
                                                transform: (orig, frame) =>
                                                    transformDragValue(orig, frame),
                                                computeChangeFactor: computeSelectionChangeFactor,
                                                applyEdgeSmoothing: applyEdgeSmoothingToDense,
                                            });
                                            const finalDense = built.values;
                                            const overallMinFrameExt = built.startFrame;

                                            // 立即同步更新本地 paramView state（逐帧映射到 pv 的
                                            // 采样栅格上；pv 只是显示，随后会被后端数据刷新）
                                            const nextEdit = pvNow.edit.slice();
                                            const pvStepUp = Math.max(1, Math.floor(pvNow.stride));
                                            for (let i = 0; i < finalDense.length; i++) {
                                                const globalIdx = Math.round(
                                                    (overallMinFrameExt + i - pvNow.startFrame) /
                                                        pvStepUp,
                                                );
                                                if (globalIdx >= 0 && globalIdx < nextEdit.length) {
                                                    nextEdit[globalIdx] = finalDense[i];
                                                }
                                            }
                                            setParamView({
                                                ...pvNow,
                                                edit: nextEdit,
                                            });
                                            liveEditOverrideRef.current = null;

                                            // 确保选区位置最终正确
                                            const beatDeltaForSel =
                                                (lastFrameDelta * fp) / 1000 / secPerBeat;
                                            selectionRef.current = {
                                                aBeat: aBeat + beatDeltaForSel,
                                                bBeat: bBeat + beatDeltaForSel,
                                            };
                                            updateSelectionUi(selectionRef.current);

                                            // 分块回写：整段编辑只打一个撤销点，块之间让出
                                            // 事件循环，超长工程也不会卡死界面。
                                            void (async () => {
                                                await uploadFullResCurve({
                                                    trackId: rootTrackId,
                                                    param: editParam,
                                                    startFrame: overallMinFrameExt,
                                                    values: finalDense,
                                                });
                                                if (liveEditActiveRef)
                                                    liveEditActiveRef.current = false;
                                                bumpRefreshToken();
                                            })();
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
                                const onMouseDownDuringDrag = (ev: globalThis.MouseEvent) => {
                                    if (ev.button !== 2) return;
                                    // 仅在左键拖拽进行中时，右键才切换拖拽方向。
                                    if ((ev.buttons & 1) !== 1) return;
                                    ev.preventDefault();
                                    ev.stopPropagation();
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

                // 默认行为：仅左键创建新选区；右键不应在 pointerdown 时清除选区
                if (e.button === 0) {
                    const maxSelectableBeat = Math.max(
                        0,
                        dynamicProjectSec / Math.max(1e-9, secPerBeat),
                    );
                    const clampSelectionBeat = (beat: number) => clamp(beat, 0, maxSelectableBeat);

                    const selectionBeatFromClientX = (
                        clientX: number,
                        allowAutoScroll: boolean,
                    ) => {
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
                        const beat =
                            (scrollLeftRef.current + (clampedClientX - bounds.left)) /
                            Math.max(1e-9, pxPerBeatRef.current);
                        return clampSelectionBeat(beat);
                    };

                    const startBeat = selectionBeatFromClientX(e.clientX, false);
                    selectionRef.current = { aBeat: startBeat, bBeat: startBeat };
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
                        if (selectionRef.current == null) return;
                        const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);
                        const bb = selectionBeatFromClientX(adjusted.clientX, true);
                        selectionRef.current = {
                            aBeat: selectionRef.current.aBeat,
                            bBeat: bb,
                        };
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
                const onMouseDownDuringDraw = (ev: globalThis.MouseEvent) => {
                    if (ev.button !== 2) return;
                    if (!canCycleDragDirection) return;
                    // 仅在左键拖拽进行中时，右键才切换拖拽方向。
                    if ((ev.buttons & 1) !== 1) return;
                    ev.preventDefault();
                    ev.stopPropagation();
                    currentDragDir = currentDragDir === "free" ? "x-only" : "free";
                    if (onCycleDragDirection) {
                        onCycleDragDirection(isVibratoTool ? "vibrato" : "draw");
                    }
                };

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
                const onMouseDownDuringDraw = (ev: globalThis.MouseEvent) => {
                    if (ev.button !== 2) return;
                    if (!canCycleDragDirection) return;
                    // 仅在左键拖拽进行中时，右键才切换拖拽方向。
                    if ((ev.buttons & 1) !== 1) return;
                    ev.preventDefault();
                    ev.stopPropagation();
                    currentDragDir = currentDragDir === "free" ? "x-only" : "free";
                    if (onCycleDragDirection) {
                        onCycleDragDirection("draw");
                    }
                };

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
            paramViewRef,
            ensureLiveEditBase,
            paramView?.framePeriodMs,
            secPerBeat,
            pointerValue,
            strokeRef,
            applyDenseToLiveEdit,
            applyEdgeSmoothingToDense,
            computeSelectionChangeFactor,
            applyMorphOverlayPreview,
            applyPostStrokeSmoothing,
            buildMorphDense,
            buildMorphOverlayFromSelection,
            commitStroke,
            bumpRefreshToken,
            liveEditOverrideRef,
            setMorphOverlay,
            setParamView,
            setCanvasCursor,
            onPitchSnapGestureActiveChange,
            pitchSnapEnabled,
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
        ],
    );

    useEffect(() => {
        if (!paramValuePopupEnabled) {
            onParamValuePreviewChange?.(null);
            return;
        }
        const clearPreview = () => onParamValuePreviewChange?.(null);
        window.addEventListener("pointerup", clearPreview);
        window.addEventListener("pointercancel", clearPreview);
        return () => {
            window.removeEventListener("pointerup", clearPreview);
            window.removeEventListener("pointercancel", clearPreview);
        };
    }, [paramValuePopupEnabled, onParamValuePreviewChange]);

    return {
        onRulerMouseDown,
        onScrollerMouseDownCapture,
        onScrollerAuxClick,
        onScrollerScroll,
        onScrollerContextMenu,
        onScrollerKeyDown,
        onScrollerWheel,
        onScrollerWheelNative,
        onCanvasPointerMove,
        onCanvasPointerLeave,
        onCanvasPointerDown,
    };
}
