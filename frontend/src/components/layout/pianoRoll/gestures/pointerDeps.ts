/**
 * 指针回调可以取用的**上下文**（`usePianoRollInteractions` 的回调作用域）。
 *
 * 【为什么需要这个文件】`onCanvasPointerDown` 是三千多行的指针按下分派，拆出来时要把
 * 回调作用域里能看到的九十个值传进去。若拆出去的模块直接引用
 * `usePianoRollInteractions` 的返回类型，类型检查会绕成环（见 `../panel/modelDeps.ts`
 * 的说明）；这里只引用各值的**声明类型**，环就断了。
 *
 * 【怎么维护】生成物：用 TypeScript 编译器 API 读取调用点上那个依赖对象的**类型**
 * （`getTypeAtLocation`），逐字段打印并解析所需类型 import。
 * 新增字段 = 在回调里多用一个名字，然后重新生成。
 */
import type { AppDispatch } from "../../../../app/store";
import type { Keybinding } from "../../../../features/keybindings/types";
import type { DrawToolMode } from "../../../../features/session/sessionTypes";
import type { VibratoChoice } from "../../../../features/vibrato/vibratoPresetList";
import type { VibratoPreset } from "../../../../features/vibrato/vibratoTypes";
import type { ScaleLike } from "../../../../utils/musicalScales";
import type { TimelineAxis } from "../../renderKernel/timelineAxis";
import type { FrameSpan, ParamSelection } from "../paramSelection";
import type { MultiRangeEditPiece } from "../selectionEditData";
import type {
    ParamMorphOverlay,
    ParamName,
    ParamViewSegment,
    StrokeMode,
    StrokePoint,
    ValueViewport,
} from "../types";
import type { LiveEditOverride } from "../useLiveParamEditing";
import type { VibratoDragResetIntent, VibratoDragWorking } from "../vibratoDragAdjust";
import type { VibratoHudAnchor } from "../vibratoHudState";
import type { CanvasCursor } from "./pointerSupport";
import type { MutableRefObject, PointerEvent as ReactPointerEvent, RefObject } from "react";

/** 指针回调需要的上下文（各自按需 `Pick`）。 */
export type PointerDeps = {
    activePointerGestureEndRef: RefObject<(() => void) | null>;
    applyDenseToLiveEdit: (
        pv: ParamViewSegment,
        denseStartFrame: number,
        dense: number[] | null,
        minF: number,
        maxF: number,
        mode: StrokeMode,
    ) => void;
    applyEdgeScrollLeft: (drawingScrollLeft: number) => void;
    applyMorphOverlayPreview: (overlays: ParamMorphOverlay[]) => void;
    applyPostStrokeSmoothing: (points: StrokePoint[], mode: StrokeMode) => Promise<void>;
    applyVibratoChoice: (choice: VibratoChoice, clientX?: number, clientY?: number) => boolean;
    applyVibratoReset: (
        intent: VibratoDragResetIntent,
        clientX?: number,
        clientY?: number,
    ) => boolean;
    axisFromRefs: () => TimelineAxis;
    blendDenseEdges: (
        dense: number[],
        base: number[],
        editedStartIdx: number,
        editedLen: number,
        halfSpanFrames: number,
    ) => void;
    buildMorphDense: (
        overlay: ParamMorphOverlay,
        stride: number,
    ) => { startFrame: number; endFrame: number; dense: number[] };
    buildUploadSegments: (
        param: ParamName,
        pieces: readonly MultiRangeEditPiece[],
        sentinelAt?: (frame: number) => boolean | undefined,
    ) => Array<{ startFrame: number; values: number[] }>;
    buildVibratoDense: (
        startFrame: number,
        startValue: number,
        endFrame: number,
        endValue: number,
        working: VibratoDragWorking,
        shiftHeld: boolean,
        seed: number,
    ) => { minF: number; maxF: number; dense: number[] };
    bumpRefreshToken: () => void;
    canvasRef: MutableRefObject<HTMLCanvasElement | null>;
    clampViewport: (param: ParamName, v: ValueViewport) => ValueViewport;
    clearActivePointerGestureEnd: (endGesture?: () => void) => void;
    commitStroke: (points: StrokePoint[], mode: StrokeMode) => Promise<void>;
    createFineAdjustedPointerState: (
        ev: {
            clientX: number;
            clientY: number;
            ctrlKey: boolean;
            shiftKey: boolean;
            altKey: boolean;
            metaKey?: boolean;
            movementX?: number;
            movementY?: number;
        },
        _dragTarget?: HTMLCanvasElement | null,
    ) => { adjustedClientX: number; adjustedClientY: number };
    currentParamRange: { min: number; max: number } | undefined;
    dispatch: AppDispatch;
    disposeFineAdjustedPointerState: (
        _state: { adjustedClientX: number; adjustedClientY: number } | null | undefined,
    ) => void;
    dragDirection: "free" | "x-only" | "y-only" | undefined;
    dynamicProjectSec: number;
    edgeHalfSpanForIndices: (indicesLen: number, strideValue: number) => number;
    edgeSmoothnessPercent: number | undefined;
    editParam: string;
    ensureLiveEditBase: (pv: ParamViewSegment) => void;
    fetchCommitBaseSource: (input: {
        trackId: string;
        param: ParamName;
        ranges: readonly FrameSpan[];
        frameDelta: number;
        edgeHalfSpanAt: (rangeIndex: number) => number;
        paramView: ParamViewSegment | null;
        withSentinel?: boolean;
    }) => Promise<{
        sourceAt: (frame: number) => number;
        sentinelAt: (frame: number) => boolean | undefined;
    }>;
    findStretchSelectionEdge: (
        e: ReactPointerEvent<HTMLCanvasElement>,
        options?: { readonly requireModifier?: boolean },
    ) => { rangeIndex: number; edge: "left" | "right" } | null;
    finishVibratoDrag: () => void;
    framePeriodMs: number;
    frameToViewportPx: (cut: number) => number;
    getDefaultCanvasCursor: () => CanvasCursor;
    getFineAdjustedPointerPosition: (
        state: { adjustedClientX: number; adjustedClientY: number },
        ev: {
            clientX: number;
            clientY: number;
            ctrlKey: boolean;
            shiftKey: boolean;
            altKey: boolean;
            metaKey?: boolean;
            movementX?: number;
            movementY?: number;
        },
    ) => { clientX: number; clientY: number; fineActive: boolean };
    getViewportTruth: () => { pxPerSec: number; scrollLeft: number } | null;
    installDragDirectionKeyCycler: (cycleLocalDragDir: () => void) => () => void;
    invalidate: () => void;
    isEffectivePitchSnapActive: (ev: {
        ctrlKey: boolean;
        shiftKey: boolean;
        altKey: boolean;
        metaKey?: boolean;
    }) => boolean;
    isPointerNearDraggableSelection: (
        clientX: number,
        clientY: number,
        pointerType?: string | null,
    ) => boolean;
    isSnapToggleModifierHeld: (ev: {
        ctrlKey: boolean;
        shiftKey: boolean;
        altKey: boolean;
        metaKey?: boolean;
    }) => boolean;
    liveEditActiveRef: MutableRefObject<boolean> | undefined;
    liveEditOverrideRef: MutableRefObject<LiveEditOverride | null>;
    morphDragRef: RefObject<{
        pointerId: number;
        overlayIndex: number;
        pointKind: "left" | "mid1" | "mid2" | "right";
    } | null>;
    morphModifierDownRef: RefObject<boolean>;
    morphOverlayRef: RefObject<ParamMorphOverlay[] | null>;
    onContextMenu: ((x: number, y: number) => void) | undefined;
    onCycleDragDirection: ((tool: "select" | "draw" | "vibrato") => void) | undefined;
    onParamCommitSucceeded: () => void;
    onParamValuePreviewChange:
        | ((
              next: {
                  clientX: number;
                  clientY: number;
                  value: number;
                  displayText?: string;
              } | null,
          ) => void)
        | undefined;
    onPitchSnapGestureActiveChange: ((active: boolean) => void) | undefined;
    panRef: MutableRefObject<{
        pointerId: number;
        startClientX: number;
        startClientY: number;
        startScrollLeft: number;
        startView: ValueViewport;
        startRectH: number;
    } | null>;
    paramEditorSeekPlayheadEnabled: boolean | undefined;
    paramMultiSelectKb: Keybinding;
    paramStretchKb: Keybinding;
    paramValuePopupEnabled: boolean | undefined;
    paramView: ParamViewSegment | null;
    paramViewRef: MutableRefObject<ParamViewSegment | null>;
    paramViewsRef: MutableRefObject<Record<string, ValueViewport>>;
    pitchSnapToleranceCents: number | undefined;
    pitchSnapUnit: "semitone" | "scale" | undefined;
    pitchViewRef: MutableRefObject<ValueViewport>;
    pointerFrame: (clientX: number) => number;
    pointerSec: (clientX: number) => number;
    pointerValue: (clientY: number) => number;
    projectScale: ScaleLike | undefined;
    reportVibratoDragState: (clientX?: number, clientY?: number) => void;
    requestWaveformRepaint: () => void;
    resetLiveEditPreview: (pv: ParamViewSegment, opts?: { keepWrittenRange?: boolean }) => void;
    rootTrackId: string | null;
    scaleAtSec: ((sec: number) => ScaleLike | undefined) | undefined;
    scrollLeftRef: MutableRefObject<number>;
    scrollerRef: MutableRefObject<HTMLDivElement | null>;
    selectionRef: MutableRefObject<ParamSelection | null>;
    setActivePointerGestureEnd: (endGesture: () => void) => void;
    setCanvasCursor: (next: CanvasCursor) => void;
    setMorphOverlay: (next: ParamMorphOverlay[] | null) => void;
    setParamView: (next: ParamViewSegment | null) => void;
    setParamViewport: (param: string, next: ValueViewport) => void;
    setPitchView: (next: ValueViewport) => void;
    setVibratoDragCaptureActive: (active: boolean) => void;
    snapDrawValue: (v: number, snapToggleHeld?: boolean, frame?: number) => number;
    straightVibratoPreset: VibratoPreset;
    strokeRef: MutableRefObject<{
        mode: StrokeMode;
        pointerId: number;
        param: ParamName;
        points: StrokePoint[];
    } | null>;
    syncScrollLeft: (scroller: HTMLDivElement) => void;
    syncTimelineEnabled: boolean;
    timelineOffsetRef: MutableRefObject<number>;
    toolMode: string;
    updateSelectionUi: (next: ParamSelection | null) => void;
    valueToY: (param: ParamName, v: number, h: number) => number;
    vibratoFrequencyAdjustKb: Keybinding;
    vibratoHudAnchorRef: RefObject<VibratoHudAnchor | null>;
    vibratoPreset: VibratoPreset;
    vibratoPresetCycleList: readonly VibratoPreset[];
    vibratoStateRef: RefObject<{
        pointerId: number;
        startFrame: number;
        startValue: number;
        currentFrame: number;
        currentValue: number;
        mode: StrokeMode;
        tool: DrawToolMode;
        working: VibratoDragWorking;
        shiftHeld: boolean;
        seed: number;
    } | null>;
    viewSizeRef: MutableRefObject<{ w: number; h: number }>;
};
