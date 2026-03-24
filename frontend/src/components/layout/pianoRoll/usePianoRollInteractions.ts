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
import {
    seekPlayhead,
    setplayheadSec,
} from "../../../features/session/sessionSlice";
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
import {
    isModifierActive,
    isNoneBinding,
} from "../../../features/keybindings/keybindingsSlice";
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
import { getParamEditorWheelAction } from "./wheelGesture";
import { transformSelectionByRightDrag } from "./selectionTransforms";

type CanvasCursor =
    | "default"
    | "crosshair"
    | "grab"
    | "grabbing"
    | "ew-resize";

export function usePianoRollInteractions(args: {
    dispatch: AppDispatch;
    rootTrackId: string | null;
    editParam: ParamName;
    pitchEnabled: boolean;
    toolMode: string;
    secPerBeat: number;
    scrollLeftRef: MutableRefObject<number>;
    pxPerBeatRef: MutableRefObject<number>;
    setPxPerBeat: (next: number) => void;
    /** 当前 BPM，用于动态计算 pxPerBeat 的合法范围 */
    bpm: number;
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
    onParamValuePreviewChange?: (next: {
        clientX: number;
        clientY: number;
        value: number;
    } | null) => void;
    /** 是否启用绘制时音高吸附 */
    pitchSnapEnabled?: boolean;
    /** 音高吸附方式 */
    pitchSnapUnit?: "semitone" | "scale";
    /** 音高吸附调式（支持内置与自定义） */
    projectScale?: ScaleLike;
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
        editParam,
        pitchEnabled,
        toolMode,
        secPerBeat,
        scrollLeftRef,
        pxPerBeatRef,
        setPxPerBeat,
        bpm,
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

    const pointerFineScale = useCallback(
        (
            ev: {
                ctrlKey: boolean;
                shiftKey: boolean;
                altKey: boolean;
                metaKey?: boolean;
            },
        ) => (isModifierActive(paramFineAdjustKb, ev as any) ? 0.1 : 1),
        [paramFineAdjustKb],
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
        (
            editedDense: number[],
            editedStartIdx: number,
            editedLen: number,
            changeFactor = 1,
        ) => {
            const effectiveChange = clamp(Number(changeFactor) || 0, 0, 1);
            if (effectiveChange <= 0) return;
            const strength = clamp(Number(edgeSmoothnessPercent) || 0, 0, 100);
            if (strength <= 0 || editedLen <= 1) return;

            const maxTransitionFrames = Math.floor(editedLen / 2);
            if (maxTransitionFrames <= 0) return;

            const transitionFrames = Math.round(
                (strength / 100) * maxTransitionFrames,
            );
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
                    editedDense[idx] =
                        snapshot[idx] + (smoothed - snapshot[idx]) * effectiveChange;
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
                    editedDense[idx] =
                        snapshot[idx] + (smoothed - snapshot[idx]) * effectiveChange;
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
    const lastPointerPosRef = useRef<{ clientX: number; clientY: number; pointerId?: number; buttons?: number }>({
        clientX: 0,
        clientY: 0,
        pointerId: 0,
        buttons: 0,
    });

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
        if (!Number.isFinite(aBeat) || !Number.isFinite(bBeat) || bBeat <= aBeat)
            return null;

        const fp = Math.max(1e-6, pv.framePeriodMs);
        const stride = Math.max(1, pv.stride);
        const selStartFrameRaw = Math.max(
            0,
            Math.floor((aBeat * secPerBeat * 1000) / fp),
        );
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
            editParam === "pitch"
                ? baselineValues.filter((v) => Number(v) !== 0)
                : baselineValues;
        const meanValue =
            valid.length > 0
                ? valid.reduce((sum, v) => sum + (Number(v) || 0), 0) /
                  valid.length
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
     *  When shiftHeld=true, the snap state is toggled (XOR with pitchSnapEnabled). */
    const snapDrawValue = useCallback(
        (v: number, shiftHeld = false): number => {
            const effective = shiftHeld ? !pitchSnapEnabled : pitchSnapEnabled;
            if (!effective || editParam !== "pitch") return v;
            const snapped =
                pitchSnapUnit === "scale" && projectScale
                    ? snapToScale(v, projectScale)
                    : snapToSemitone(v);
            const toleranceSemitone = Math.max(
                0,
                Number(pitchSnapToleranceCents ?? 0) / 100,
            );
            if (Math.abs(v - snapped) <= toleranceSemitone) {
                return v;
            }
            return snapped + (((v - snapped) > 0 ? 1 : -1)) * toleranceSemitone;
        },
        [
            pitchSnapEnabled,
            pitchSnapUnit,
            projectScale,
            pitchSnapToleranceCents,
            editParam,
        ],
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
                dense[f - minF] = snapDrawValue(base + wave, shiftHeld);
            }
            return { minF, maxF, dense };
        },
        [snapDrawValue],
    );

    useEffect(() => {
        if (toolMode !== "select") {
            morphModifierDownRef.current = false;
            if (!morphDragRef.current) {
                setMorphOverlay(null);
            }
        }

        const updateMorphActivation = (
            e: globalThis.KeyboardEvent | { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey?: boolean },
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
                    if (!strokeRef.current && !panRef.current) {
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
                if (!strokeRef.current && !panRef.current) {
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
            // If no active stroke, nothing to refresh here
            const st = strokeRef.current;
            if (!st) return;

            const last = lastPointerPosRef.current;
            if (!last) {
                invalidate();
                return;
            }

            try {
                const pe = new PointerEvent("pointermove", {
                    clientX: last.clientX,
                    clientY: last.clientY,
                    pointerId: st.pointerId ?? last.pointerId ?? 1,
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
            } catch (err) {
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
    }, [strokeRef, invalidate]);

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
            const bounds = (
                e.currentTarget as HTMLDivElement
            ).getBoundingClientRect();
            const sl = scrollLeftRef.current;
            const ppb = pxPerBeatRef.current;
            const beat = clamp(
                (e.clientX - bounds.left + sl) / Math.max(1e-9, ppb),
                0,
                1e12,
            );
            // beat → sec：playheadSec 存储的是秒，必须转换后再 dispatch
            const sec = beat * secPerBeat;
            dispatch(setplayheadSec(sec));
            void dispatch(seekPlayhead(sec));
        },
        [dispatch, scrollLeftRef, pxPerBeatRef, secPerBeat],
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
            if (!rootTrackId) return;
            if (editParam === "pitch" && !pitchEnabled) return;

            // pianoRoll.shiftParamUp / shiftParamDown 已移至全局 handleKeybindingAction 处理

            // Handle edit.* keybindings passed through from useKeybindings global handler
            // (must be before the selectionRef guard since selectAll/deselect work without selection)
            if (keybindingMap && onEditAction) {
                const editActionEntries = (Object.entries(keybindingMap) as [ActionId, Keybinding][])
                    .filter(([id]) => id.startsWith("edit."));
                // 需要弹出对话框的操作列表
                const dialogOps = new Set([
                    "transposeCents", "transposeDegrees", "setPitch",
                    "average", "smooth", "addVibrato", "quantize", "meanQuantize",
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
            const frameCount = clamp(
                Math.ceil((durSec * 1000) / fp),
                1,
                200_000,
            );

            // 检测 pianoRoll.copy 绑定
            {
                const kb = pianoRollCopyKb;
                let keyMatch = false;
                if (kb.modifierOnly) {
                    keyMatch = isModifierActive(kb, e.nativeEvent);
                } else {
                    let pressedKey =
                        e.key === " " ? "space" : e.key.toLowerCase();
                    if (pressedKey !== kb.key) keyMatch = false;
                    else {
                        const isMac = navigator.platform
                            .toLowerCase()
                            .includes("mac");
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
                            framePeriodMs:
                                Number(payload.frame_period_ms ?? fp) || fp,
                            values: (payload.edit ?? []).map(
                                (v) => Number(v) || 0,
                            ),
                        };
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
                    let pressedKey =
                        e.key === " " ? "space" : e.key.toLowerCase();
                    if (pressedKey !== kb.key) keyMatch = false;
                    else {
                        const isMac = navigator.platform
                            .toLowerCase()
                            .includes("mac");
                        const modKey = isMac ? e.metaKey : e.ctrlKey;
                        keyMatch =
                            modKey === Boolean(kb.ctrl) &&
                            e.shiftKey === Boolean(kb.shift) &&
                            e.altKey === Boolean(kb.alt);
                    }
                }
                if (keyMatch) {
                    e.preventDefault();
                    const clip = clipboardRef.current;
                    if (!clip) return;
                    if (clip.param !== editParam) return;
                    // 将剪贴板数据截断到选区帧数范围内
                    const pasteValues = clip.values.length > frameCount
                        ? clip.values.slice(0, frameCount)
                        : clip.values;
                    void (async () => {
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
                // 频率调整优先级更高：当两者同时命中时，只调整频率。
                const freqActive = freqRequested;
                const ampActive = ampRequested && !freqActive;

                if (ampActive || freqActive) {
                    e.preventDefault();
                    const steps = Math.max(1, Math.round(Math.abs(e.deltaY) / 100));
                    const fineScale = pointerFineScale(e);
                    if (ampActive) {
                        const rangeSpan =
                            editParam === "pitch"
                                ? 48
                                : Math.max(
                                      1e-6,
                                      Number(currentParamRange?.max ?? 1) -
                                          Number(currentParamRange?.min ?? 0),
                                  );
                        const baseAmpStep = Math.max(rangeSpan / 200, 0.01);
                        // 上滚增大振幅，下滚减小；允许 0 和负值。
                        const dir = e.deltaY < 0 ? 1 : -1;
                        vib.amplitude += dir * baseAmpStep * steps * fineScale;
                    }
                    if (freqActive) {
                        // 上滚减小频率，下滚增大频率；按倍率缩放且始终为正。
                        const ratio = Math.pow(1 + 0.1 * fineScale, steps);
                        vib.frequency = e.deltaY < 0 ? vib.frequency / ratio : vib.frequency * ratio;
                        vib.frequency = Math.max(1e-4, vib.frequency);
                    }

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
                            e.shiftKey,
                        );
                                    vib.shiftHeld = e.shiftKey;
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
                    return;
                }

                // 颤音拖拽期间默认屏蔽滚轮，防止画布缩放/滚动。
                e.preventDefault();
                return;
            }

            const noModifierPressed =
                !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey;
            const isWheelBindingRequested = (kb: Keybinding) => {
                if (isNoneBinding(kb)) return noModifierPressed;
                return isModifierActive(kb, e);
            };
            const horizontalScrollModifierActive = isWheelBindingRequested(
                scrollHorizontalKb,
            );
            const wheelAction = getParamEditorWheelAction({
                deltaX: e.deltaX,
                deltaY: e.deltaY,
                horizontalScrollRequested: horizontalScrollModifierActive,
                verticalPanRequested: isWheelBindingRequested(scrollVerticalKb),
                verticalZoomRequested: isWheelBindingRequested(prVerticalZoomKb),
                horizontalZoomRequested: isWheelBindingRequested(horizontalZoomKb),
            });

            // Scroll modifier: convert wheel to horizontal scroll
            if (wheelAction === "horizontal-scroll") {
                e.preventDefault();
                el.scrollLeft += horizontalScrollModifierActive
                    ? e.deltaY
                    : e.deltaX;
                syncScrollLeft(el);
                return;
            }

            // Scroll modifier: convert wheel to vertical scroll
            if (wheelAction === "vertical-pan") {
                e.preventDefault();
                // 实现参数值轴的平移（center 上下移动）
                const h = Math.max(1, el.clientHeight);
                const delta = (-e.deltaY / h) * 0.5;
                if (editParam === "pitch") {
                    const cur = pitchViewRef.current;
                    const next = clampViewport("pitch", {
                        span: cur.span,
                        center: cur.center + delta * cur.span,
                    });
                    setPitchView(next);
                } else {
                    const cur = paramViewsRef.current[editParam] ?? { center: 0.5, span: 1 };
                    const next = clampViewport(editParam, {
                        span: cur.span,
                        center: cur.center + delta * cur.span,
                    });
                    setParamViewport(editParam, next);
                }
                invalidate();
                return;
            }

            // Anchor zoom to the actual drawable viewport (canvas), not the scroller.
            // The scroller may include rulers/padding, which makes zoom feel off-center.
            const canvas = canvasRef.current;
            const bounds = (canvas ?? el).getBoundingClientRect();

            const pointerXRaw = e.clientX - bounds.left;
            const pointerYRaw = e.clientY - bounds.top;
            if (
                pointerXRaw < 0 ||
                pointerYRaw < 0 ||
                pointerXRaw > bounds.width ||
                pointerYRaw > bounds.height
            ) {
                return;
            }

            // We rely on preventDefault to stop native scrolling while zooming.
            e.preventDefault();

            if (wheelAction === "vertical-zoom") {
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
                    const cur = paramViewsRef.current[editParam] ?? {
                        center: 0.5,
                        span: 1,
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
            const curPxPerBeat = pxPerBeatRef.current;

            // Playhead-based zoom: use playhead position as anchor instead of pointer
            const secPerBeatLocal = 60 / Math.max(1, bpm);
            let anchorX: number;
            let anchorBeat: number;
            if (playheadZoomEnabled && playheadSec != null) {
                anchorBeat = playheadSec / secPerBeatLocal;
                anchorX = anchorBeat * curPxPerBeat - el.scrollLeft;
                // 如果 playhead 在可视区域外，先将其居中，再以其为锚点缩放
                if (anchorX < 0 || anchorX > bounds.width) {
                    const centeredScrollLeft = anchorBeat * curPxPerBeat - bounds.width / 2;
                    el.scrollLeft = Math.max(0, centeredScrollLeft);
                    syncScrollLeft(el);
                    anchorX = anchorBeat * curPxPerBeat - el.scrollLeft;
                }
                anchorX = clamp(anchorX, 0, Math.max(1, bounds.width));
            } else {
                anchorX = clamp(pointerXRaw, 0, Math.max(1, bounds.width));
                anchorBeat =
                    (anchorX + el.scrollLeft) / Math.max(1e-9, curPxPerBeat);
            }

            const minPxPerBeat = MIN_PX_PER_SEC * secPerBeatLocal;
            const maxPxPerBeat = MAX_PX_PER_SEC * secPerBeatLocal;

            const next = clamp(
                curPxPerBeat * factor,
                minPxPerBeat,
                maxPxPerBeat,
            );
            if (Math.abs(next - curPxPerBeat) < 1e-9) return;

            setPxPerBeat(next);
            const nextScrollLeft = anchorBeat * next - anchorX;
            // 使用 dynamicProjectSec 计算正确的 maxScroll，避免缩放时波形偏移
            const totalBeats = dynamicProjectSec / secPerBeatLocal;
            const maxScroll = Math.max(0, totalBeats * next - el.clientWidth);
            el.scrollLeft = clamp(nextScrollLeft, 0, maxScroll);
            syncScrollLeft(el);
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
            pxPerBeatRef,
            setPxPerBeat,
            syncScrollLeft,
            prVerticalZoomKb,
            scrollHorizontalKb,
            scrollVerticalKb,
            horizontalZoomKb,
            vibratoAmplitudeAdjustKb,
            vibratoFrequencyAdjustKb,
            paramFineAdjustKb,
            bpm,
            dynamicProjectSec,
            playheadSec,
            playheadZoomEnabled,
            strokeRef,
            paramViewRef,
            liveEditOverrideRef,
            ensureLiveEditBase,
            buildVibratoDense,
        ],
    );

    // React's onWheel handler may run in a passive listener in modern React.
    // Keep this for compatibility, but do not call preventDefault here.
    const onScrollerWheel = useCallback((_e: WheelEvent<HTMLDivElement>) => {
        // no-op; wheel is handled via native listener with passive:false
    }, []);

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
            const idx = Math.round(
                (frame - pv.startFrame) / Math.max(1, pv.stride),
            );
            const curveVal =
                idx >= 0 && idx < pv.edit.length ? Number(pv.edit[idx]) : null;
            if (curveVal == null || !Number.isFinite(curveVal)) return null;

            const rect = canvas.getBoundingClientRect();
            const rectH = rect.height || viewSizeRef.current.h || 1;
            const mouseY = clientY - rect.top;
            const mappedCurveVal =
                editParam === "pitch" ? curveVal + 0.5 : curveVal;
            const curveY = valueToY(editParam, mappedCurveVal, rectH);
            return Math.abs(mouseY - curveY) < 10 ? curveVal : null;
        },
        [
            paramViewRef,
            canvasRef,
            pointerBeat,
            secPerBeat,
            viewSizeRef,
            editParam,
            valueToY,
        ],
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
        [
            toolMode,
            selectionRef,
            pointerBeat,
            getCurveValueNearPointer,
        ],
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
            return (
                Math.abs(localX - leftX) <= edgeHitPx ||
                Math.abs(localX - rightX) <= edgeHitPx
            );
        },
        [
            toolMode,
            paramStretchKb,
            selectionRef,
            canvasRef,
            pxPerBeatRef,
            scrollLeftRef,
        ],
    );

    const onCanvasPointerMove = useCallback(
        (e: ReactPointerEvent<HTMLCanvasElement>) => {
            if (paramValuePopupEnabled) {
                const draggingLeft =
                    Boolean(strokeRef.current) && (e.buttons & 1) === 1;
                if (draggingLeft) {
                    onParamValuePreviewChange?.({
                        clientX: e.clientX,
                        clientY: e.clientY,
                        value: pointerValue(e.clientY),
                    });
                } else {
                    const nearCurveValue = getCurveValueNearPointer(
                        e.clientX,
                        e.clientY,
                    );
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
    }, [
        panRef,
        strokeRef,
        onParamValuePreviewChange,
        setCanvasCursor,
        getDefaultCanvasCursor,
    ]);

    const onCanvasPointerDown = useCallback(
        (e: ReactPointerEvent<HTMLCanvasElement>) => {
            if (e.button === 0 && paramEditorSeekPlayheadEnabled !== false) {
                const beat = pointerBeat(e.clientX);
                const sec = Math.max(0, beat * secPerBeat);
                dispatch(setplayheadSec(sec));
                void dispatch(seekPlayhead(sec));
            }

            if (
                e.button === 0 &&
                paramValuePopupEnabled &&
                (toolMode === "select" || toolMode === "draw" || toolMode === "line")
            ) {
                onParamValuePreviewChange?.({
                    clientX: e.clientX,
                    clientY: e.clientY,
                    value: pointerValue(e.clientY),
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
                };
                window.addEventListener("pointermove", onMove);
                window.addEventListener("pointerup", onUp);
                window.addEventListener("pointercancel", onUp);
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
                            Math.abs((e.clientX - rect.left) - x) <= 8 &&
                            Math.abs((e.clientY - rect.top) - y) <= 8
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

                        const onMove = (ev: globalThis.PointerEvent) => {
                            const drag = morphDragRef.current;
                            const overlayNow = morphOverlayRef.current;
                            const pvNow = paramViewRef.current;
                            if (
                                !drag ||
                                drag.pointerId !== e.pointerId ||
                                !overlayNow ||
                                !pvNow
                            ) {
                                return;
                            }

                            const nextPoints = overlayNow.points.map((pt) => {
                                if (pt.kind !== drag.pointKind) return pt;
                                const newValue = pointerValue(ev.clientY);
                                if (pt.kind === "left" || pt.kind === "right") {
                                    return { ...pt, value: newValue };
                                }
                                const beat = pointerBeat(ev.clientX);
                                const sec = beat * secPerBeat;
                                const rawFrame = Math.max(0, Math.floor((sec * 1000) / Math.max(1e-6, pvNow.framePeriodMs)));
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
                        };

                        const onUp = () => {
                            const drag = morphDragRef.current;
                            const overlayNow = morphOverlayRef.current;
                            const pvNow = paramViewRef.current;
                            morphDragRef.current = null;
                            window.removeEventListener("pointermove", onMove);
                            window.removeEventListener("pointerup", onUp);
                            window.removeEventListener("pointercancel", onUp);

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
                            const leftX =
                                aBeat * pxPerBeatRef.current - scrollLeftRef.current;
                            const rightX =
                                bBeat * pxPerBeatRef.current - scrollLeftRef.current;
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
                                            (edgeSmoothStr / 100) *
                                                Math.floor(nextLen / 2),
                                        ) / 2,
                                    );
                                    const extraEdgeFrames = edgeHalfSpanIdx * stride;
                                    const overallMinFrame = Math.max(
                                        0,
                                        Math.min(oldStartFrame, nextStartFrame) -
                                            extraEdgeFrames,
                                    );
                                    const overallMaxFrame =
                                        Math.max(oldEndFrame, nextEndFrame) +
                                        extraEdgeFrames;
                                    const overallLen =
                                        Math.floor(
                                            (overallMaxFrame - overallMinFrame) / stride,
                                        ) + 1;
                                    const dense = new Array<number>(overallLen);
                                    for (let i = 0; i < overallLen; i += 1) {
                                        const frame = overallMinFrame + i * stride;
                                        const idx = Math.round(
                                            (frame - pvNow.startFrame) / stride,
                                        );
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
                                    const smoothRatio = clamp(
                                        Number(edgeSmoothnessPercent) || 0,
                                        0,
                                        100,
                                    ) / 100;
                                    const outsideWindowLen = Math.max(
                                        1,
                                        Math.round(maxOutsideWindow * smoothRatio),
                                    );

                                    // 缩短时，用原选区内侧一小段值回填被腾空区域。
                                    // smoothness=0 时 outsideWindowLen=1，相当于边缘值沿边界内侧延展。
                                    if (nextStartFrame > oldStartFrame) {
                                        const fillLen = Math.floor((nextStartFrame - oldStartFrame) / stride);
                                        for (let i = 0; i < fillLen; i += 1) {
                                            const targetFrame = oldStartFrame + i * stride;
                                            const targetIdx = Math.round((targetFrame - overallMinFrame) / stride);
                                            const srcWindowPos = fillLen > 1
                                                ? Math.round((i / (fillLen - 1)) * (outsideWindowLen - 1))
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
                                            const targetIdx = Math.round((targetFrame - overallMinFrame) / stride);
                                            const srcWindowPos = fillLen > 1
                                                ? Math.round((i / (fillLen - 1)) * (outsideWindowLen - 1))
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
                                    const pvNow = paramViewRef.current;
                                    if (!pvNow) return;
                                    const cursorBeat = pointerBeat(ev.clientX);
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
                                        if (liveEditActiveRef)
                                            liveEditActiveRef.current = false;
                                        bumpRefreshToken();
                                    })();
                                    setCanvasCursor("default");
                                    invalidate();
                                };

                                window.addEventListener("pointermove", onMove);
                                window.addEventListener("pointerup", onUp);
                                window.addEventListener("pointercancel", onUp);
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
                            const frame = Math.max(
                                0,
                                Math.floor((sec * 1000) / fp),
                            );
                            const idx = Math.round(
                                (frame - pv.startFrame) /
                                    Math.max(1, pv.stride),
                            );
                            const curveVal =
                                idx >= 0 && idx < pv.edit.length
                                    ? pv.edit[idx]
                                    : null;
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

                            if (
                                curveY !== null &&
                                Math.abs(mouseY - curveY) < HIT_THRESHOLD_PX
                            ) {
                                if (e.button === 2) {
                                    e.preventDefault();
                                    const startClientY = e.clientY;
                                    const pid = e.pointerId;
                                    (
                                        e.currentTarget as HTMLCanvasElement
                                    ).setPointerCapture(pid);

                                    setCanvasCursor("grabbing");

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
                                        Math.round(
                                            (selStartFrame - pv.startFrame) /
                                                stride,
                                        ),
                                    );
                                    const selEndIdx = Math.min(
                                        pv.edit.length - 1,
                                        Math.round(
                                            (selEndFrame - pv.startFrame) /
                                                stride,
                                        ),
                                    );
                                    const origValues = pv.edit.slice(
                                        selStartIdx,
                                        selEndIdx + 1,
                                    );

                                    let didDrag = false;
                                    let latestDense = origValues.slice();
                                    let latestAppliedStartFrame = selStartFrame;
                                    let latestAppliedDense = origValues.slice();

                                    ensureLiveEditBase(pv);
                                    if (liveEditActiveRef)
                                        liveEditActiveRef.current = true;

                                    const buildRightDragDense = (
                                        pvNow: ParamViewSegment,
                                        nextSelectionValues: number[],
                                    ) => {
                                        const selLen = nextSelectionValues.length;
                                        const maxTransitionFrames = Math.floor(selLen / 2);
                                        const transitionFrames =
                                            Number(edgeSmoothnessPercent) > 0 && maxTransitionFrames > 0
                                                ? Math.round(
                                                    (clamp(Number(edgeSmoothnessPercent) || 0, 0, 100) / 100) *
                                                        maxTransitionFrames,
                                                )
                                                : 0;
                                        const extraEdgeFrames =
                                            Math.max(0, Math.ceil(transitionFrames / 2)) * stride;
                                        const denseStartFrame = Math.max(0, selStartFrame - extraEdgeFrames);
                                        const denseEndFrame = selEndFrame + extraEdgeFrames;
                                        const denseLength =
                                            Math.floor((denseEndFrame - denseStartFrame) / stride) + 1;
                                        const dense = new Array<number>(denseLength);

                                        for (let index = 0; index < denseLength; index += 1) {
                                            const globalIdx = Math.round(
                                                (denseStartFrame + index * stride - pvNow.startFrame) / stride,
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

                                        for (let index = 0; index < nextSelectionValues.length; index += 1) {
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
                                        const fineScale = pointerFineScale(ev);
                                        const dy =
                                            (startClientY - ev.clientY) * fineScale;
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
                                        const nextApplied = buildRightDragDense(
                                            pvNow,
                                            latestDense,
                                        );
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
                                        invalidate();
                                    };

                                    const onUp = (ev: globalThis.PointerEvent) => {
                                        window.removeEventListener("pointermove", onMove);
                                        window.removeEventListener("pointerup", onUp);
                                        window.removeEventListener("pointercancel", onUp);
                                        window.removeEventListener(
                                            "contextmenu",
                                            suppressContextMenu,
                                            true,
                                        );

                                        if (!didDrag) {
                                            liveEditOverrideRef.current = null;
                                            if (liveEditActiveRef) {
                                                liveEditActiveRef.current = false;
                                            }
                                            setCanvasCursor("grab");
                                            if (args.onContextMenu) {
                                                args.onContextMenu(
                                                    ev.clientX,
                                                    ev.clientY,
                                                );
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

                                        const suppressOnce = (
                                            evt: globalThis.MouseEvent,
                                        ) => {
                                            evt.preventDefault();
                                            evt.stopPropagation();
                                            window.removeEventListener(
                                                "contextmenu",
                                                suppressOnce,
                                                true,
                                            );
                                        };
                                        window.addEventListener(
                                            "contextmenu",
                                            suppressOnce,
                                            true,
                                        );
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
                                    return;
                                }

                                // 进入拖拽选中曲线模式（支持 X+Y 双向拖拽）
                                setCanvasCursor("grabbing");
                                const startMouseVal = mouseVal;
                                const startBeat = pointerBeat(e.clientX);
                                const pid = e.pointerId;
                                (
                                    e.currentTarget as HTMLCanvasElement
                                ).setPointerCapture(pid);

                                // 保存选区内曲线原始值
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
                                    Math.round(
                                        (selStartFrame - pv.startFrame) /
                                            stride,
                                    ),
                                );
                                const selEndIdx = Math.min(
                                    pv.edit.length - 1,
                                    Math.round(
                                        (selEndFrame - pv.startFrame) / stride,
                                    ),
                                );
                                const origValues = pv.edit.slice(
                                    selStartIdx,
                                    selEndIdx + 1,
                                );
                                ensureLiveEditBase(pv);
                                if (liveEditActiveRef)
                                    liveEditActiveRef.current = true;
                                if (editParam === "pitch") {
                                    onPitchSnapGestureActiveChange?.(true);
                                }

                                // 用闭包变量记录最新 X/Y 偏移量
                                let lastValueDelta = 0;
                                let lastScaleStepDelta = 0;
                                let useScaleDegreeTranspose = false;
                                let lastFrameDelta = 0; // 帧偏移（整数）
                                // 使用闭包变量跟踪当前拖动方向（可通过右键切换）
                                let currentDragDir = dragDirection ?? "y-only";

                                const onMove = (
                                    ev: globalThis.PointerEvent,
                                ) => {
                                    const fineScale = pointerFineScale(ev);
                                    const currentVal = pointerValue(ev.clientY);
                                    let rawValueDelta =
                                        (currentVal - startMouseVal) * fineScale;

                                    // 音高吸附：Toggle snap modifier (XOR with pitchSnapEnabled)
                                    const noSnapKb = keybindingMap?.["modifier.clipNoSnap" as ActionId];
                                    const snapToggled = noSnapKb ? isModifierActive(noSnapKb, ev) : false;
                                    const effectiveSnap = snapToggled ? !pitchSnapEnabled : pitchSnapEnabled;
                                    const yDragEnabled = currentDragDir !== "x-only";
                                    if (effectiveSnap && editParam === "pitch" && yDragEnabled) {
                                        if (pitchSnapUnit === "scale" && projectScale) {
                                            useScaleDegreeTranspose = true;
                                            lastScaleStepDelta = scaleStepDeltaBetween(
                                                startMouseVal,
                                                currentVal,
                                                projectScale,
                                            );
                                            rawValueDelta = 0;
                                        } else {
                                            useScaleDegreeTranspose = false;
                                            rawValueDelta = Math.round(rawValueDelta);
                                        }
                                    } else {
                                        useScaleDegreeTranspose = false;
                                        if (!yDragEnabled) {
                                            lastScaleStepDelta = 0;
                                            rawValueDelta = 0;
                                        }
                                    }

                                    // 计算 X 方向帧偏移
                                    const currentBeat = pointerBeat(ev.clientX);
                                    const beatDelta =
                                        (currentBeat - startBeat) * fineScale;
                                    const secDelta = beatDelta * secPerBeat;
                                    const rawFrameDelta = Math.round(
                                        (secDelta * 1000) / fp,
                                    );

                                    // 应用拖动方向限制
                                    lastValueDelta = yDragEnabled ? rawValueDelta : 0;
                                    lastFrameDelta = currentDragDir === "y-only" ? 0 : rawFrameDelta;

                                    const pvNow = paramViewRef.current;
                                    if (!pvNow) return;

                                    // Reset live overlay before each move to prevent stale values
                                    // from the previous drag position lingering outside the current range
                                    liveEditOverrideRef.current = null;
                                    ensureLiveEditBase(pvNow);

                                    // 构造覆盖原选区 + 新位置的完整 dense 数组
                                    const selLen = selEndIdx - selStartIdx + 1;
                                    const origDenseStart =
                                        pv.startFrame + selStartIdx * stride;

                                    // 计算需要覆盖的帧范围：原选区 ∪ 新位置选区
                                    const newDenseStart =
                                        origDenseStart + lastFrameDelta;
                                    const overallMinFrame = Math.max(
                                        0,
                                        Math.min(origDenseStart, newDenseStart),
                                    );
                                    const origDenseEnd =
                                        origDenseStart + (selLen - 1) * stride;
                                    const newDenseEnd =
                                        newDenseStart + (selLen - 1) * stride;
                                    const overallMaxFrame = Math.max(
                                        origDenseEnd,
                                        newDenseEnd,
                                    );

                                    // 边缘平滑度：扩展 dense 范围以包含选区边界外侧上下文
                                    // halfSpan 与 applyEdgeSmoothingToDense 内的计算保持一致
                                    const edgeSmoothStr = clamp(Number(edgeSmoothnessPercent) || 0, 0, 100);
                                    const edgeHalfSpanIdx = Math.ceil(
                                        Math.round((edgeSmoothStr / 100) * Math.floor(selLen / 2)) / 2,
                                    );
                                    const extraEdgeFrames = edgeHalfSpanIdx * stride;
                                    const overallMinFrameExt = Math.max(0, overallMinFrame - extraEdgeFrames);
                                    const overallMaxFrameExt = overallMaxFrame + extraEdgeFrames;

                                    const overallLen =
                                        Math.floor(
                                            (overallMaxFrameExt -
                                                overallMinFrameExt) /
                                                stride,
                                        ) + 1;
                                    const dense = new Array<number>(overallLen);

                                    // 先用当前 edit 曲线填充整个范围（含扩展部分；选区外锚点应基于当前新值）
                                    for (let i = 0; i < overallLen; i++) {
                                        const globalIdx = Math.round(
                                            (overallMinFrameExt +
                                                i * stride -
                                                pv.startFrame) /
                                                stride,
                                        );
                                        dense[i] =
                                            globalIdx >= 0 &&
                                            globalIdx < pvNow.edit.length
                                                ? pvNow.edit[globalIdx]
                                                : 0;
                                    }
                                    const denseBefore = dense.slice();
                                    // 再将选区值写入新位置（覆盖 orig）
                                    for (let i = 0; i < selLen; i++) {
                                        const targetFrame =
                                            newDenseStart + i * stride;
                                        const denseIdx = Math.round(
                                            (targetFrame - overallMinFrameExt) /
                                                stride,
                                        );
                                        if (
                                            denseIdx >= 0 &&
                                            denseIdx < overallLen
                                        ) {
                                            const orig = origValues[i] ?? 0;
                                            if (
                                                useScaleDegreeTranspose &&
                                                editParam === "pitch" &&
                                                projectScale
                                            ) {
                                                dense[denseIdx] =
                                                    orig === 0
                                                        ? 0
                                                        : transposePitchByScaleSteps(
                                                              orig,
                                                              lastScaleStepDelta,
                                                              projectScale,
                                                          );
                                            } else {
                                                dense[denseIdx] =
                                                    orig + lastValueDelta;
                                            }
                                        }
                                    }

                                    const movedStartDenseIdx = Math.round(
                                        (newDenseStart - overallMinFrameExt) /
                                            stride,
                                    );
                                    const changeFactor =
                                        computeSelectionChangeFactor(
                                            denseBefore,
                                            dense,
                                            movedStartDenseIdx,
                                            selLen,
                                        );
                                    applyEdgeSmoothingToDense(
                                        dense,
                                        movedStartDenseIdx,
                                        selLen,
                                        changeFactor,
                                    );

                                    applyDenseToLiveEdit(
                                        pvNow,
                                        overallMinFrameExt,
                                        dense,
                                        overallMinFrameExt,
                                        overallMaxFrameExt,
                                        "draw",
                                    );

                                    // 实时更新选区位置显示
                                    const beatDeltaForSel =
                                        (lastFrameDelta * fp) /
                                        1000 /
                                        secPerBeat;
                                    selectionRef.current = {
                                        aBeat: aBeat + beatDeltaForSel,
                                        bBeat: bBeat + beatDeltaForSel,
                                    };
                                    updateSelectionUi(selectionRef.current);

                                    if (paramValuePopupEnabled) {
                                        onParamValuePreviewChange?.({
                                            clientX: ev.clientX,
                                            clientY: ev.clientY,
                                            value: pointerValue(ev.clientY),
                                        });
                                    }

                                    invalidate();
                                };

                                const onUp = () => {
                                    window.removeEventListener(
                                        "pointermove",
                                        onMove,
                                    );
                                    window.removeEventListener(
                                        "pointerup",
                                        onUp,
                                    );
                                    window.removeEventListener(
                                        "pointercancel",
                                        onUp,
                                    );

                                    // 提交拖拽结果到后端
                                    const pvNow = paramViewRef.current;
                                    if (pvNow && rootTrackId) {
                                        const selLen =
                                            selEndIdx - selStartIdx + 1;
                                        const origDenseStart =
                                            pv.startFrame +
                                            selStartIdx * stride;
                                        const newDenseStart =
                                            origDenseStart + lastFrameDelta;

                                        const overallMinFrame = Math.max(
                                            0,
                                            Math.min(
                                                origDenseStart,
                                                newDenseStart,
                                            ),
                                        );
                                        const origDenseEnd =
                                            origDenseStart +
                                            (selLen - 1) * stride;
                                        const newDenseEnd =
                                            newDenseStart +
                                            (selLen - 1) * stride;
                                        const overallMaxFrame = Math.max(
                                            origDenseEnd,
                                            newDenseEnd,
                                        );

                                        // 边缘平滑度：扩展 dense 范围以包含选区边界外侧上下文
                                        const edgeSmoothStrUp = clamp(Number(edgeSmoothnessPercent) || 0, 0, 100);
                                        const edgeHalfSpanIdxUp = Math.ceil(
                                            Math.round((edgeSmoothStrUp / 100) * Math.floor(selLen / 2)) / 2,
                                        );
                                        const extraEdgeFramesUp = edgeHalfSpanIdxUp * stride;
                                        const overallMinFrameExt = Math.max(0, overallMinFrame - extraEdgeFramesUp);
                                        const overallMaxFrameExt = overallMaxFrame + extraEdgeFramesUp;

                                        const overallLen =
                                            Math.floor(
                                                (overallMaxFrameExt -
                                                    overallMinFrameExt) /
                                                    stride,
                                            ) + 1;

                                        // 构造最终提交的 dense 数组
                                        const finalDense = new Array<number>(
                                            overallLen,
                                        );

                                        // 先用当前 edit 填充整个范围（含扩展部分；选区外锚点应基于当前新值）
                                        for (let i = 0; i < overallLen; i++) {
                                            const globalIdx = Math.round(
                                                (overallMinFrameExt +
                                                    i * stride -
                                                    pvNow.startFrame) /
                                                    stride,
                                            );
                                            finalDense[i] =
                                                globalIdx >= 0 &&
                                                globalIdx < pvNow.edit.length
                                                    ? pvNow.edit[globalIdx]
                                                    : 0;
                                        }
                                        const finalDenseBefore =
                                            finalDense.slice();
                                        // 再将偏移后的选区值写入新位置
                                        for (let i = 0; i < selLen; i++) {
                                            const targetFrame =
                                                newDenseStart + i * stride;
                                            const denseIdx = Math.round(
                                                (targetFrame -
                                                    overallMinFrameExt) /
                                                    stride,
                                            );
                                            if (
                                                denseIdx >= 0 &&
                                                denseIdx < overallLen
                                            ) {
                                                const orig = origValues[i] ?? 0;
                                                if (
                                                    useScaleDegreeTranspose &&
                                                    editParam === "pitch" &&
                                                    projectScale
                                                ) {
                                                    finalDense[denseIdx] =
                                                        orig === 0
                                                            ? 0
                                                            : transposePitchByScaleSteps(
                                                                  orig,
                                                                  lastScaleStepDelta,
                                                                  projectScale,
                                                              );
                                                } else {
                                                    finalDense[denseIdx] =
                                                        orig + lastValueDelta;
                                                }
                                            }
                                        }

                                        const movedStartDenseIdx = Math.round(
                                            (newDenseStart - overallMinFrameExt) /
                                                stride,
                                        );
                                        const changeFactor =
                                            computeSelectionChangeFactor(
                                                finalDenseBefore,
                                                finalDense,
                                                movedStartDenseIdx,
                                                selLen,
                                            );
                                        applyEdgeSmoothingToDense(
                                            finalDense,
                                            movedStartDenseIdx,
                                            selLen,
                                            changeFactor,
                                        );

                                        // 立即同步更新本地 paramView state
                                        const nextEdit = pvNow.edit.slice();
                                        for (let i = 0; i < overallLen; i++) {
                                            const globalIdx = Math.round(
                                                (overallMinFrameExt +
                                                    i * stride -
                                                    pvNow.startFrame) /
                                                    stride,
                                            );
                                            if (
                                                globalIdx >= 0 &&
                                                globalIdx < nextEdit.length
                                            ) {
                                                nextEdit[globalIdx] =
                                                    finalDense[i];
                                            }
                                        }
                                        setParamView({
                                            ...pvNow,
                                            edit: nextEdit,
                                        });
                                        liveEditOverrideRef.current = null;

                                        // 确保选区位置最终正确
                                        const beatDeltaForSel =
                                            (lastFrameDelta * fp) /
                                            1000 /
                                            secPerBeat;
                                        selectionRef.current = {
                                            aBeat: aBeat + beatDeltaForSel,
                                            bBeat: bBeat + beatDeltaForSel,
                                        };
                                        updateSelectionUi(selectionRef.current);

                                        void (async () => {
                                            await paramsApi.setParamFrames(
                                                rootTrackId,
                                                editParam,
                                                overallMinFrameExt,
                                                finalDense,
                                                true,
                                            );
                                            if (liveEditActiveRef)
                                                liveEditActiveRef.current = false;
                                            bumpRefreshToken();
                                        })();
                                    } else {
                                        if (liveEditActiveRef)
                                            liveEditActiveRef.current = false;
                                    }
                                    // 清除参数浮窗预览（如果启用）
                                    if (paramValuePopupEnabled) {
                                        onParamValuePreviewChange?.(null);
                                    }

                                    if (editParam === "pitch") {
                                        onPitchSnapGestureActiveChange?.(false);
                                    }
                                    setCanvasCursor("grab");
                                    invalidate();
                                    window.removeEventListener("contextmenu", onContextMenuDuringDrag, true);
                                    window.removeEventListener("mousedown", onMouseDownDuringDrag, true);
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
                                    const order: Array<"free" | "x-only" | "y-only"> = ["free", "x-only", "y-only"];
                                    const idx = order.indexOf(currentDragDir);
                                    currentDragDir = order[(idx + 1) % order.length];
                                    // Also cycle the global setting
                                    if (onCycleDragDirection) onCycleDragDirection("select");
                                };

                                window.addEventListener("pointermove", onMove);
                                window.addEventListener("pointerup", onUp);
                                window.addEventListener("pointercancel", onUp);
                                window.addEventListener("contextmenu", onContextMenuDuringDrag, true);
                                window.addEventListener("mousedown", onMouseDownDuringDrag, true);
                                return;
                            }
                        }
                    }
                }

                // 默认行为：仅左键创建新选区；右键不应在 pointerdown 时清除选区
                if (e.button === 0) {
                    selectionRef.current = { aBeat: b, bBeat: b };
                    updateSelectionUi(selectionRef.current);
                    const pid = e.pointerId;
                    (e.currentTarget as HTMLCanvasElement).setPointerCapture(pid);
                    const onMove = (ev: globalThis.PointerEvent) => {
                        if (selectionRef.current == null) return;
                        const bb = pointerBeat(ev.clientX);
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
                        invalidate();
                    };
                    window.addEventListener("pointermove", onMove);
                    window.addEventListener("pointerup", onUp);
                    window.addEventListener("pointercancel", onUp);
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
            if (editParam === "pitch") {
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
            const value = isDrawMode ? snapDrawValue(rawValue, e.shiftKey) : rawValue;

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
            (e.currentTarget as HTMLCanvasElement).setPointerCapture(
                e.pointerId,
            );
            invalidate();

            if (isLineTool || isVibratoTool) {
                // Line tool: draw a straight line from start to current pointer
                const startFrame = frame;
                const startValue = value;
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
                        shiftHeld: e.shiftKey,
                    };
                }

                const onMove = (ev: globalThis.PointerEvent) => {
                    const st = strokeRef.current;
                    if (!st || st.pointerId !== e.pointerId) return;
                    const fineScale = pointerFineScale(ev);
                    const b2Raw = pointerBeat(ev.clientX);
                    const b2 = beat + (b2Raw - beat) * fineScale;
                    const sec2 = b2 * secPerBeat;
                    const f2 = Math.max(0, Math.floor((sec2 * 1000) / fp));
                    const yDragEnabled = currentDragDir !== "x-only";
                    const rawV2Abs = yDragEnabled ? pointerValue(ev.clientY) : value;
                    const rawV2 = value + (rawV2Abs - value) * fineScale;
                    const v2 = isDrawMode ? snapDrawValue(rawV2, ev.shiftKey) : rawV2;

                    // Update stroke to only have start and current end
                    st.points = [{ frame: startFrame, value: startValue }, { frame: f2, value: v2 }];

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
                                    vib.shiftHeld = ev.shiftKey;
                                    const built = buildVibratoDense(
                                        startFrame,
                                        startValue,
                                        f2,
                                        v2,
                                        vib.amplitude,
                                        vib.frequency,
                                        ev.shiftKey,
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
                                    dense[f - minF] = isDrawMode ? snapDrawValue(raw, ev.shiftKey) : raw;
                                }
                                applyDenseToLiveEdit(pv2, minF, dense, minF, maxF, mode);
                            }
                        }
                    }
                    invalidate();
                };

                const onUp = () => {
                    const st = strokeRef.current;
                    if (!st || st.pointerId !== e.pointerId) return;
                    const vib = vibratoStateRef.current;
                    strokeRef.current = null;
                    window.removeEventListener("pointermove", onMove);
                    window.removeEventListener("pointerup", onUp);
                    window.removeEventListener("pointercancel", onUp);
                    window.removeEventListener("contextmenu", onContextMenuDuringDraw, true);
                    window.removeEventListener("mousedown", onMouseDownDuringDraw, true);
                    invalidate();
                    if (editParam === "pitch") {
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
            } else {
                // Draw tool: freehand drawing with interpolation between points
                let currentDragDir: "free" | "x-only" =
                    dragDirection === "x-only" ? "x-only" : "free";
                const canCycleDragDirection = e.button === 0;
                const onMove = (ev: globalThis.PointerEvent) => {
                    const st = strokeRef.current;
                    if (!st || st.pointerId !== e.pointerId) return;
                    const fineScale = pointerFineScale(ev);
                    const b2Raw = pointerBeat(ev.clientX);
                    const last = st.points[st.points.length - 1];
                    const lastBeat = last
                        ? (last.frame * fp) / 1000 / secPerBeat
                        : b2Raw;
                    const b2 = lastBeat + (b2Raw - lastBeat) * fineScale;
                    const sec2 = b2 * secPerBeat;
                    const f2 = Math.max(0, Math.floor((sec2 * 1000) / fp));
                    const yDragEnabled = currentDragDir !== "x-only";
                    const rawV2Abs = yDragEnabled ? pointerValue(ev.clientY) : (last?.value ?? value);
                    const baseV = last?.value ?? rawV2Abs;
                    const rawV2 = baseV + (rawV2Abs - baseV) * fineScale;
                    const v2 = isDrawMode ? snapDrawValue(rawV2, ev.shiftKey) : rawV2;

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
                            applyDenseToLiveEdit(
                                pv2,
                                minF,
                                dense,
                                minF,
                                maxF,
                                mode,
                            );
                        }
                    }
                    invalidate();
                };

                const onUp = () => {
                    const st = strokeRef.current;
                    if (!st || st.pointerId !== e.pointerId) return;
                    strokeRef.current = null;
                    vibratoStateRef.current = null;
                    window.removeEventListener("pointermove", onMove);
                    window.removeEventListener("pointerup", onUp);
                    window.removeEventListener("pointercancel", onUp);
                    window.removeEventListener("contextmenu", onContextMenuDuringDraw, true);
                    window.removeEventListener("mousedown", onMouseDownDuringDraw, true);
                    invalidate();
                    if (editParam === "pitch") {
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
            paramFineAdjustKb,
            pxPerBeatRef,
            scrollLeftRef,
            valueToY,
            buildVibratoDense,
            paramEditorSeekPlayheadEnabled,
            paramValuePopupEnabled,
            onParamValuePreviewChange,
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
