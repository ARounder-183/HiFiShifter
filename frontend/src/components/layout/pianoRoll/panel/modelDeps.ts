/**
 * 面板模型对外暴露的**捕获值类型**。
 *
 * 【为什么需要这个文件】模型的类型是 `ReturnType<typeof usePianoRollPanelModel>`，
 * 而拆出去的执行体（`editOperations` / `sceneBuilders` / `drawFrame` /
 * `pitchReferenceOps`）都要声明"我需要模型里的哪几个字段"。若它们直接写
 * `Pick<PianoRollPanelModel, …>`，类型检查会绕成一个环：模型返回类型 ← 这些函数的
 * 类型 ← `Pick<模型返回类型>` ← 模型返回类型。**一两个模块时 TS 还能兜住，再多一个
 * 就整条链退化成 `any`** —— 症状是下游文件一片 implicit any（本轮实测踩到过）。
 *
 * 所以把这份字段类型单独放在这里：它只引用各字段的**声明类型**，不引用模型本身，
 * 环就断了。拆出去的模块写 `Pick<ModelDeps, …>`，仍然只声明自己真正用到的字段。
 *
 * 【怎么维护】本文件是**生成物**（用 TypeScript 编译器 API 从模型里读每个绑定的声明
 * 类型）。要新增字段：在模型里暴露它，然后重新生成，最后在模块的 `Pick<>` 里加上。
 * 手改会和模型漂移 —— 字段类型一错，环又会以"implicit any"的形式回来。
 */
import type { AppDispatch, RootState } from "../../../../app/store";
import type { SessionState } from "../../../../features/session/sessionSlice";
import type { BaselineMode, VibratoPreset } from "../../../../features/vibrato/vibratoTypes";
import type { MessageKey } from "../../../../i18n/messages";
import type { ThemeMode } from "../../../../theme/AppThemeProvider";
import type { ProcessorParamDescriptor } from "../../../../types/api";
import type { ScaleLike } from "../../../../utils/musicalScales";
import type { TimelineAxis } from "../../renderKernel/timelineAxis";
import type {
    MutablePianoRollKernelData,
    PianoRollCurveLayer,
    PianoRollSelectionBandSpec,
} from "../kernel/host/pianoRollKernelData";
import type { PianoRollKernelHost } from "../kernel/host/pianoRollKernelHost";
import type { ParamAxisUnit } from "../paramAxisUnits";
import type { ParamClipboardData } from "../paramClipboardMapping";
import type { FrameRange, ParamSelection } from "../paramSelection";
import type { DetectedPitchCurve, ReferencePitchOverlay } from "../render";
import type { ParamMorphOverlay, ParamName, ParamViewSegment, ValueViewport } from "../types";
import type { LiveEditOverride } from "../useLiveParamEditing";
import type { Dispatch, RefObject, SetStateAction } from "react";
import type { Store } from "redux";

/** 拆出去的执行体可以取用的模型字段（各自按需 `Pick`）。 */
export type ModelDeps = {
    dispatch: AppDispatch;
    store: Store<RootState>;
    setParamView: Dispatch<SetStateAction<ParamViewSegment | null>>;
    liveEditOverrideRef: RefObject<LiveEditOverride | null>;
    secondaryParamViews: Partial<Record<string, ParamViewSegment>>;
    paramViewRef: RefObject<ParamViewSegment | null>;
    secSpanClipRect: (
        axis: TimelineAxis,
        startSec: number,
        endSec: number,
    ) => { x: number; y: number; w: number; h: number } | null;
    paramView: ParamViewSegment | null;
    editParamAxisUnit: ParamAxisUnit;
    selectionClipRect: (
        axis: TimelineAxis,
        range: FrameRange,
    ) => { x: number; y: number; w: number; h: number } | null;
    s: SessionState;
    vibratoBaseline: BaselineMode;
    t: (key: MessageKey) => string;
    themeMode: ThemeMode;
    buildSelectionBandSpec: (selection: ParamSelection | null) => PianoRollSelectionBandSpec | null;
    handleEditOp: (op: string, data?: Record<string, unknown>) => Promise<void>;
    setMidiDialogSelection: Dispatch<SetStateAction<ParamSelection | null>>;
    setSelectionUi: Dispatch<SetStateAction<ParamSelection | null>>;
    midiDialogOpenParamsRef: RefObject<{ editParam: string; toolMode: string }>;
    clampViewport: (param: ParamName, v: ValueViewport) => ValueViewport;
    getParamValueBoundsForScrollbar: (param: ParamName) => { min: number; max: number };
    setClipboardGuid: Dispatch<SetStateAction<string | null>>;
    setMidiPath: Dispatch<SetStateAction<string | null>>;
    midiDialogSourceRef: RefObject<"reaperClipboard" | "paramEditor">;
    valueToY: (param: ParamName, v: number, h: number) => number;
    resolveScaleFromToken: (scaleToken: string, atSec?: number) => ScaleLike;
    paramViewsRef: RefObject<Record<string, ValueViewport>>;
    buildCurveLayers: (axis: TimelineAxis) => PianoRollCurveLayer[];
    setMidiDialogOpen: Dispatch<SetStateAction<boolean>>;
    hostRef: RefObject<PianoRollKernelHost | null>;
    kernelDataRef: RefObject<MutablePianoRollKernelData>;
    processorParamsRef: RefObject<ProcessorParamDescriptor[]>;
    clipboardRef: RefObject<ParamClipboardData | null>;
    viewSizeRef: RefObject<{ w: number; h: number }>;
    axisCanvasRef: RefObject<HTMLCanvasElement | null>;
    canvasRef: RefObject<HTMLCanvasElement | null>;
    currentParamRange: { min: number; max: number } | undefined;
    projectScaleAtSec: (sec: number) => ScaleLike | undefined;
    scrollerRef: RefObject<HTMLDivElement | null>;
    selectionRef: RefObject<ParamSelection | null>;
    getCurrentViewportForScrollbar: (param: ParamName) => ValueViewport;
    pitchViewRef: RefObject<ValueViewport>;
    paramMorphOverlays: ParamMorphOverlay[] | null;
    viewSize: { w: number; h: number };
    lastMirroredScrollTopRef: RefObject<number>;
    pxPerSecRef: RefObject<number>;
    referencePitchOverlays: ReferencePitchOverlay[];
    scrollLeftRef: RefObject<number>;
    visualPlayheadSecRef: RefObject<number>;
    detectedPitchCurves: DetectedPitchCurve[];
    resolvedVibratoPresets: VibratoPreset[];
    childPitchHardDisableReason: string | null;
    pitchHardDisableReason: string | null;
    rootTrackId: string | null;
    selectAllParamRange: () => boolean;
    prAxis: TimelineAxis;
    bumpRefreshToken: () => void;
    invalidate: () => void;
    effectiveProjectScale: ScaleLike;
    visibleSecondaryParamIds: string[];
    loudnessAnalysisPending: boolean;
    pitchEnabled: boolean;
    activeVibratoPresetId: string;
    currentParamDefaultValue: number;
    currentParamQuantizeUnit: number;
    editParam: string;
    fontFamily: string;
    pitchDeltaToDegreeSteps: (basePitch: number, targetPitch: number, scale: ScaleLike) => number;
};
