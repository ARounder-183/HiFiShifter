/**
 * 手势回调可以取用的**上下文**（`usePianoRollInteractions` 的回调作用域）。
 *
 * 【为什么需要这个文件】指针 / 滚轮 / 键盘手势已经搬到 `gestures/` 下，每个回调都要
 * 声明"我需要 hook 作用域里的哪几个值"。若它们直接引用 `usePianoRollInteractions` 的
 * 返回类型，类型检查会绕成环（见 `../panel/modelDeps.ts` 的说明）；这里只引用各值的
 * **声明类型**，环就断了。
 *
 * 【怎么维护】生成物：用 TypeScript 编译器 API 读取每个手势调用点上那个依赖对象的
 * **类型**（`getTypeAtLocation`），取并集逐字段打印。新增字段 = 回调里多用一个名字，
 * 然后重新生成。
 */
import type { AppDispatch } from "../../../../app/store";
import type { Keybinding, KeybindingMap } from "../../../../features/keybindings/types";
import type { DrawToolMode } from "../../../../features/session/sessionTypes";
import type { VibratoChoice } from "../../../../features/vibrato/vibratoPresetList";
import type { VibratoPreset } from "../../../../features/vibrato/vibratoTypes";
import type { ScaleLike } from "../../../../utils/musicalScales";
import type { PinchStepState } from "../../../../utils/pinchGesture";
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
import type {
    VibratoAdjustTarget,
    VibratoDragResetIntent,
    VibratoDragWorking,
} from "../vibratoDragAdjust";
import type { VibratoHudAnchor } from "../vibratoHudState";
import type { CanvasCursor } from "./pointerSupport";
import type { MutableRefObject, PointerEvent as ReactPointerEvent, RefObject } from "react";

/** 块工具 pointerdown 的结构化结果：是否消费，以及单击是否应移动播放头。 */
export type ParamBlockPointerDownResult =
    { readonly handled: true; readonly seekOnClick: boolean } | { readonly handled: false };

/** 手势回调需要的上下文（各自按需 `Pick`）。 */
export type PointerDeps = {
    /**
     * 块编辑工具命中。
     *
     * handled=true 表示这次按下已由块层消费，参数线手势不再启动。
     */
    onParamBlockPointerDown: (args: {
        sec: number;
        localY: number;
        clientY: number;
        clientX: number;
        add: boolean;
        toggle: boolean;
        pointerType: string;
        pointerId: number;
        captureTarget: HTMLCanvasElement;
        setActivePointerGestureEnd: (endGesture: () => void) => void;
        clearActivePointerGestureEnd: (endGesture?: () => void) => void;
        /** 画布高度（CSS px）：把指针 y 换算到值域。 */
        canvasHeightPx: number;
        /** 点击值域的纵向映射。 */
        yToValue: (param: string, y: number, h: number) => number;
        /** 当前吸附开关与步长。 */
        blockSnapEnabled: boolean;
        blockSnapStep: number;
        blockSnapScale?: ScaleLike | null;
        /** 实时的指针值（用于折算纵向拖动偏移）。 */
        pointerValue: (clientY: number) => number;
        /** 按下时的 Alt 状态；块工具用 Alt+上下拖缩放块内音高。 */
        altKey?: boolean;
        /** 按下时的 Shift 状态；块编辑临时取消吸附（按 settings 绑定）。 */
        shiftKey?: boolean;
        ctrlKey?: boolean;
        metaKey?: boolean;
        /** 屏幕 clientX → 内容秒（用于判断块时间范围）。 */
        secAtClientX: (clientX: number) => number;
        /** 吸附开关（拖动中可随修饰键变化）。 */
        isSnapToggleModifierHeld: (event: {
            ctrlKey: boolean;
            shiftKey: boolean;
            altKey: boolean;
            metaKey?: boolean;
        }) => boolean;
        /** 拖动参数偏移的网格步长。 */
        /** 当前参数名（决定 dyn 等乘性语义）。 */
        editParam: string;
        /** 参数网格步长（吸附拖动用）。 */
        gridStep: number;
        /** 是否为右键：命中块时打开块菜单，不启动拖动。 */
        contextMenu: boolean;
        /** 双击命中的块内切分点；由面板负责执行切分。 */
        doubleClick?: boolean;
    }) => ParamBlockPointerDownResult;
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
    applyVibratoDragAdjustment: (input: {
        target: VibratoAdjustTarget;
        direction: 1 | -1;
        steps: number;
        shiftHeld: boolean;
        fineEvent: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey?: boolean };
        clientX?: number;
        clientY?: number;
    }) => boolean;
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
    currentParamQuantizeUnit: number;
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
    getCurveValueNearPointer: (
        clientX: number,
        clientY: number,
        pointerType?: string | null,
    ) => number | null;
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
    getPlayheadSec: (() => number) | undefined;
    getViewportTruth: () => { pxPerSec: number; scrollLeft: number } | null;
    horizontalZoomChainRef: MutableRefObject<{
        nextPxPerSec: number;
        nextScrollLeft: number;
    } | null>;
    horizontalZoomKb: Keybinding;
    hoverPreviewNearCurveRef: RefObject<boolean>;
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
    isPointerNearSelectionEdge: (e: ReactPointerEvent<HTMLCanvasElement>) => boolean;
    isPointerNearStretchSelectionEdge: (e: ReactPointerEvent<HTMLCanvasElement>) => boolean;
    isSnapToggleModifierHeld: (ev: {
        ctrlKey: boolean;
        shiftKey: boolean;
        altKey: boolean;
        metaKey?: boolean;
    }) => boolean;
    keybindingMap: KeybindingMap | undefined;
    lastPointerClientRef: RefObject<{ x: number; y: number } | null>;
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
    onEditAction: ((op: string) => void) | undefined;
    onHorizontalZoom: (nextPxPerSec: number, nextScrollLeft: number) => void;
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
    pitchEnabled: boolean;
    pitchSnapToleranceCents: number | undefined;
    pitchSnapUnit: "semitone" | "scale" | undefined;
    pitchViewRef: MutableRefObject<ValueViewport>;
    playheadZoomEnabled: boolean | undefined;
    pointerFrame: (clientX: number) => number;
    pointerSec: (clientX: number) => number;
    pointerValue: (clientY: number) => number;
    projectScale: ScaleLike | undefined;
    prVerticalZoomKb: Keybinding;
    pxPerSecRef: MutableRefObject<number>;
    reportVibratoDragState: (clientX?: number, clientY?: number) => void;
    requestWaveformRepaint: () => void;
    resetLiveEditPreview: (pv: ParamViewSegment, opts?: { keepWrittenRange?: boolean }) => void;
    rootTrackId: string | null;
    scaleAtSec: ((sec: number) => ScaleLike | undefined) | undefined;
    scrollbarZoomKb: Keybinding;
    scrollerRef: MutableRefObject<HTMLDivElement | null>;
    scrollHorizontalKb: Keybinding;
    scrollLeftRef: MutableRefObject<number>;
    scrollVerticalKb: Keybinding;
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
    vibratoAmplitudeAdjustKb: Keybinding;
    vibratoFrequencyAdjustKb: Keybinding;
    vibratoHudAnchorRef: RefObject<VibratoHudAnchor | null>;
    vibratoPinchRef: RefObject<PinchStepState>;
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
    yToValue: (param: ParamName, y: number, h: number) => number;
    yToViewportT: (y: number, h: number) => number;
};
