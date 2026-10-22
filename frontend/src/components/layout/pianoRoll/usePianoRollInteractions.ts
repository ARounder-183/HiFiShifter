import type {
    KeyboardEvent,
    MouseEvent as ReactMouseEvent,
    MutableRefObject,
    PointerEvent as ReactPointerEvent,
    UIEvent,
} from "react";
import { useCallback, useEffect, useRef } from "react";
import { pianoRollViewportBus } from "./pianoRollViewportBus";
import {
    resolveViewportReplay,
    warnViewportReplayButtonsLost,
    type ViewportProjection,
} from "./viewportReplay";
import type { ParamFramesPayload } from "../../../types/api";
import type { AppDispatch } from "../../../app/store";
import { paramsApi } from "../../../services/api";
import {
    persistUiSettings,
    setActiveVibratoPreset,
    setToolModePersistent,
} from "../../../features/session/sessionSlice";
import { clamp } from "../timeline";
import type { LiveEditOverride } from "./useLiveParamEditing";
import type { DrawToolMode } from "../../../features/session/sessionTypes";
import type {
    ParamMorphOverlay,
    ParamName,
    ParamViewSegment,
    StrokeMode,
    StrokePoint,
    ValueViewport,
} from "./types";
import { computeDynGeometricMean } from "./selectionTransforms";
import { framesToTime } from "./utils";
import { isDynParam, restoreDynSentinels } from "./paramRanges";
import {
    CURVE_HIT_RADIUS_PX,
    curvePointAtPointer,
    hitTestSelectionBody,
    hitTestSelectionEdge,
    isPointerNearCurve,
    SELECTION_EDGE_HIT_PX,
} from "./kernel/gestureHitTest";
import { selectionIndexRange } from "./kernel/dragArithmetic";
import type { MutableRefObject as MutRef } from "react";
import {
    firstBinding,
    isModifierActive,
    isNoneBinding,
    isNoneBindingList,
} from "../../../features/keybindings/keybindingsSlice";
import { matchKeybindingAllowingFineModifier } from "../../../features/keybindings/useKeybindings";
import type { Keybinding } from "../../../features/keybindings/types";
import type { KeybindingMap, ActionId } from "../../../features/keybindings/types";
import { snapToScale, snapToSemitone } from "../../../utils/musicalScales";
import type { ScaleLike } from "../../../utils/musicalScales";
import {
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
    snapChildPitchOffsetValue,
} from "./childPitchOffsetParams";
import { profileFor, scaledHitRadius } from "../../../utils/inputProfile";
import { createPinchStepState, type PinchStepState } from "../../../utils/pinchGesture";
import { type ScrollbarZone } from "./wheelGesture";
import {
    applyEdgeBlend,
    drawSmoothSigmaMsFromStrength,
    edgeHalfSpanFramesForSelection,
    editablePitchValue,
    smoothCurveGaussian,
} from "./paramSmoothing";
import {
    fetchFullResCurve,
    planSelectionEditWindows,
    type MultiRangeEditPiece,
} from "./selectionEditData";
import {
    frameRangeEndCut,
    frameRangeStartCut,
    type FrameSpan,
    type ParamSelection,
} from "./paramSelection";
import {
    buildDragVibratoCurve,
    computeVibratoDragAdjustment,
    createDragWorking,
    matchedVibratoResetSlots,
    resetVibratoDragDepth,
    resetVibratoDragRate,
    resolveVibratoDragKeyboardAdjustment,
    resolveVibratoPairReset,
    resolveVibratoPresetSwitch,
    switchDragPreset,
    type VibratoAdjustTarget,
    type VibratoDragResetIntent,
    type VibratoDragResetBindings,
    type VibratoDragResetSlot,
    type VibratoDragWorking,
} from "./vibratoDragAdjust";
import {
    cycleVibratoChoice,
    findVibratoPreset,
    vibratoCycleAnchorId,
    type VibratoChoice,
} from "../../../features/vibrato/vibratoPresetList";
import { vibratoSeedForPreset } from "../../../features/vibrato/vibratoSeed";
import { nextVibratoHudAnchor, vibratoHudState, type VibratoHudAnchor } from "./vibratoHudState";
import type { VibratoPreset } from "../../../features/vibrato/vibratoTypes";
import { frameFromViewportClientX, secFromViewportClientX } from "./seekPlayheadMapping";
import { resolvePanelRenderViewport } from "./kernel/viewportSource";
import { createTimelineAxis, secToViewportPx } from "../renderKernel/timelineAxis.js";
import { handleCanvasPointerDown } from "./gestures/canvasPointerDown";
import type { CanvasCursor } from "./gestures/pointerSupport";
import { handleScrollerWheel } from "./gestures/scrollerWheel";
import { handleCanvasPointerMove } from "./gestures/canvasPointerMove";
import { handleRulerPointerDown } from "./gestures/rulerPointerDown";
import { handleScrollerKeyDown } from "./gestures/scrollerKeyDown";

export function usePianoRollInteractions(args: {
    dispatch: AppDispatch;
    rootTrackId: string | null;
    editParam: ParamName;
    pitchEnabled: boolean;
    toolMode: string;
    /** 每帧时长（毫秒）。选区的单位是**帧**（见 `paramSelection.ts`），因此本 hook
     *  只需要帧周期，不再需要每拍秒数 —— 选区的所有换算都与 BPM 无关。 */
    framePeriodMs: number;
    scrollLeftRef: MutableRefObject<number>;
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
    /**
     * 拖拽边缘自动滚屏的落点（绘制坐标）。
     *
     * 与 `syncScrollLeft` 的区别：那条为"原生滚动事件驱动"设计，必须同任务重绘；
     * 本条只写内核并标脏 —— 绘制由笔画自己的 rAF 合帧循环负责，避免每帧两次全帧
     * 提交。见 `PianoRollPanel` 里 `applyEdgeScrollLeft` 的说明。
     */
    applyEdgeScrollLeft: (drawingScrollLeft: number) => void;
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
    resetLiveEditPreview: (pv: ParamViewSegment, opts?: { keepWrittenRange?: boolean }) => void;

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
    /** 颤音拖拽时增大振幅（直线工具同样可用） */
    vibratoDragAmplitudeIncreaseKb: Keybinding;
    /** 颤音拖拽时减小振幅（直线工具同样可用） */
    vibratoDragAmplitudeDecreaseKb: Keybinding;
    /** 颤音拖拽时增大频率（直线工具同样可用） */
    vibratoDragFrequencyIncreaseKb: Keybinding;
    /** 颤音拖拽时减小频率（直线工具同样可用） */
    vibratoDragFrequencyDecreaseKb: Keybinding;
    /** 拖拽颤音时切换到上一个预设 */
    vibratoPresetPrevKb: Keybinding;
    /** 拖拽颤音时切换到下一个预设 */
    vibratoPresetNextKb: Keybinding;
    /**
     * 当前活动的颤音预设。
     *
     * 拖拽起手时从它取波形、包络、深度与速率。系统预设与用户预设的合成见
     * `vibratoPresetList.resolveVibratoPresets`。
     */
    vibratoPreset: VibratoPreset;
    /**
     * 「直线」预设 —— 直线工具的固定起手预设。
     *
     * 直线工具就是"颤音工具 + 这个预设"（见 `STRAIGHT_VIBRATO_PRESET_ID`），
     * 因此它必须始终可用：即使活动预设被停用或列表被改得面目全非，直线工具
     * 也照画直线。由面板用 `systemVibratoPreset("straight")` 兜底解析后传入。
     */
    straightVibratoPreset: VibratoPreset;
    /**
     * 拖拽中环绕切换时**实际参与**的列表（已剔除被停用的预设）。
     *
     * 与 `vibratoPresetList` 分开：停用只影响"挑预设"，不影响"显示当前预设"。
     */
    vibratoPresetCycleList: readonly VibratoPreset[];
    /**
     * 拖拽中调参 / 切换后的上报（面板据此刷新 HUD）。
     *
     * 深度 / 速率只是本次手势的工作副本；预设切换的持久化由本钩子直接
     * dispatch，不经这里。
     */
    onVibratoDragStateChange?: (next: {
        presetId: string;
        depthCents: number;
        rateHz: number;
        adjusted: boolean;
        /** 指针的视口坐标，供 HUD 定位。 */
        clientX: number;
        clientY: number;
    }) => void;
    /** 拖拽结束：面板据此清掉 HUD。 */
    onVibratoDragEnd?: () => void;
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
    cycleDragDirectionKb?: readonly Keybinding[];
    /**
     * 撤销栈深度读取器（后端权威镜像）。仅「边缘拉伸」手势在提交成功后
     * 用它登记选区步骤（撤销/重做恢复对应选区），其余操作不消费。
     */
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
        framePeriodMs,
        scrollLeftRef,
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
        applyEdgeScrollLeft,
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
        vibratoPresetPrevKb,
        vibratoPresetNextKb,
        vibratoPreset,
        straightVibratoPreset,
        vibratoPresetCycleList,
        onVibratoDragStateChange,
        onVibratoDragEnd,
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
     * 帧切点 → 视口 x。
     *
     * 此前写作 `beat * pxPerBeat - scrollLeft`（pxPerBeat = pxPerSec * secPerBeat），
     * 与主画布的曲线投影分属两条算式。现在统一为「帧先转秒，再走
     * `secToViewportPx`」，与时间线侧和 `render.ts` 同源。
     *
     * 传进来的是**切点**（连续帧坐标）：选区的左右边界就是两个切点，右边界是
     * `startFrame + frameCount`（最后一帧的右缘），因此画出来的选区带与真正会被
     * 改动的帧集合逐帧重合。
     */
    const frameToViewportPx = useCallback(
        (cut: number) => secToViewportPx(axisFromRefs(), framesToTime(cut, framePeriodMs)),
        [axisFromRefs, framePeriodMs],
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
            // 修饰键手势只有一个绑定，取主绑定（见 types.ts 的 KeybindingMap）。
            const noSnapKb = firstBinding(keybindingMap?.["modifier.clipNoSnap" as ActionId]);
            if (noSnapKb && !isNoneBinding(noSnapKb)) {
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
     * 返回可直接喂给 buildMultiRangeEditPlan 的 `sourceAt`；`withSentinel` 时还返回
     * 逐帧「未画」位图查询 `sentinelAt`（dyn 提交在写回前还原哨兵帧，
     * 见 `paramRanges::restoreDynSentinels`）。
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
            withSentinel?: boolean;
        }): Promise<{
            sourceAt: (frame: number) => number;
            sentinelAt: (frame: number) => boolean | undefined;
        }> => {
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
                        withSentinel: input.withSentinel,
                    }),
                ),
            );
            return {
                sourceAt: (frame: number) => {
                    for (let i = 0; i < windows.length; i += 1) {
                        const window = windows[i];
                        if (frame < window.startFrame || frame > window.endFrame) continue;
                        return Number(curves[i]?.values[frame - window.startFrame]) || 0;
                    }
                    return 0;
                },
                sentinelAt: (frame: number) => {
                    for (let i = 0; i < windows.length; i += 1) {
                        const window = windows[i];
                        if (frame < window.startFrame || frame > window.endFrame) continue;
                        return curves[i]?.sentinel?.[frame - window.startFrame];
                    }
                    return undefined;
                },
            };
        },
        [],
    );

    /**
     * 组装上传片段：dyn 的「读-变换-写回」提交在写回前把「未画」帧还原成哨兵
     * （否则拖拽会把未画帧物化成显式基线，日后基线重分析时响度静默漂移）。
     * 本地 pv/覆盖层仍用未还原的 pieces（显示解析后的基线值，与拖拽预览一致）；
     * 哨兵只影响写回后端的那份。非 dyn 或位图缺失时与原实现逐字节相同。
     */
    const buildUploadSegments = useCallback(
        (
            param: ParamName,
            pieces: readonly MultiRangeEditPiece[],
            sentinelAt?: (frame: number) => boolean | undefined,
        ): Array<{ startFrame: number; values: number[] }> =>
            pieces.map((piece) => {
                if (!isDynParam(param) || !sentinelAt) {
                    return { startFrame: piece.startFrame, values: piece.values };
                }
                const sentinels = new Array<boolean>(piece.endFrame - piece.startFrame + 1);
                for (let f = piece.startFrame; f <= piece.endFrame; f += 1) {
                    sentinels[f - piece.startFrame] = sentinelAt(f) === true;
                }
                return {
                    startFrame: piece.startFrame,
                    values: restoreDynSentinels(piece.values.slice(), sentinels),
                };
            }),
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
        /**
         * 本次手势当前所用的工具。
         *
         * 【为什么记在拖拽状态里，而不是读 `toolMode` 属性】拖拽期间工具会变：
         * 轮转落在直线预设上会切到直线工具，再轮转回来又切回颤音工具。而
         * 手势的键盘 / 侧键监听是在 `pointerdown` 时装上的，闭包捕获的是**那一刻**
         * 的 `toolMode` —— 用它做判断，第二次切换就会读到过期值，出现"画的是颤音、
         * 工具却停在直线"这种既非此也非彼的状态。手势自己记，就不会过期。
         */
        tool: DrawToolMode;
        /** 拖拽工作副本：预设 + 本次拖拽的深度 / 速率覆盖。 */
        working: VibratoDragWorking;
        shiftHeld: boolean;
        /** 不规则度的确定性种子：同一次拖拽内保持恒定，预览与提交才逐帧一致。 */
        seed: number;
    } | null>(null);
    /**
     * 拖拽期间**当前按住**的颤音绑定槽（用于识别"双键同时按下"）。
     *
     * 按绑定槽而不是键位记录：用户可以把任一动作改绑到任意键。keydown 加入、
     * keyup 移除；拖拽结束与窗口失焦时整体清空，避免某次 keyup 被漏掉之后残留，
     * 让后续的单击被误判成"双键"。
     */
    const vibratoResetSlotsRef = useRef<Set<VibratoDragResetSlot>>(new Set());
    /**
     * 颤音拖拽期间捏合的整步累积器。
     *
     * 【为什么必须累积】触控板捏合吐出的是一串连续的小 delta，而 `steps` 的算法是
     * `Math.max(1, Math.round(|delta| / 100))` —— 那是给"一格一个事件"的机械滚轮
     * 写的。若不累积，每个小 delta 都会 `max(1, …)` 算作一整步，深度会以事件频率
     * 飞走（这正是一个既有缺陷）。累积到满一格才走一步，与滚轮同速。
     */
    const vibratoPinchRef = useRef<PinchStepState>(createPinchStepState());
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
            const bindings = cycleDragDirectionKb;
            if (!bindings || isNoneBindingList(bindings)) {
                return () => {};
            }
            document.body.setAttribute(PARAM_DRAG_ATTR, "true");
            const onKeyDown = (e: globalThis.KeyboardEvent) => {
                if (e.repeat) return;
                if (!matchKeybindingAllowingFineModifier(e, bindings, paramFineAdjustKb)) return;
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
            if (!Number.isFinite(range.startFrame) || !Number.isFinite(range.frameCount)) continue;
            // 零长选区（单击留下）没有可形变的区间，与旧实现跳过 `bBeat <= aBeat` 一致。
            if (range.frameCount <= 0) continue;

            // 选区帧区间 → 采样下标：抽到 `kernel/dragArithmetic`（纯函数，有单测）。
            // 这段算术在 hook 里原有 3 份副本，任一处漂移都会让拉伸预览与提交错位。
            const selRange = selectionIndexRange({
                startFrame: range.startFrame,
                frameCount: range.frameCount,
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
    }, [editParam, paramViewRef, selectionRef]);

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

    // 选区 UI 状态合帧：拖拽路径逐事件调用 updateSelectionUi，而 setSelectionUi
    // 会让整个面板重渲。画布侧读 selectionRef（逐事件、无延迟），React 状态只喂
    // 非画布消费方（菜单 / 选区激活标记 / 总线发布），滞后一帧不可感知。
    const selectionUiRafRef = useRef<number | null>(null);
    const selectionUiPendingRef = useRef<{ value: ParamSelection | null } | null>(null);
    useEffect(() => {
        return () => {
            if (selectionUiRafRef.current != null) {
                cancelAnimationFrame(selectionUiRafRef.current);
                selectionUiRafRef.current = null;
            }
            selectionUiPendingRef.current = null;
        };
    }, []);

    const updateSelectionUi = useCallback(
        (next: ParamSelection | null) => {
            if (next === null) {
                // 清除必须立即生效并取消挂起设置：手势收尾后，过期选区不得在
                // 下一帧"复活"。
                if (selectionUiRafRef.current != null) {
                    cancelAnimationFrame(selectionUiRafRef.current);
                    selectionUiRafRef.current = null;
                }
                selectionUiPendingRef.current = null;
                setSelectionUi(null);
                if (morphModifierDownRef.current && !morphDragRef.current) {
                    setMorphOverlay(buildMorphOverlaysFromSelection());
                }
                return;
            }
            selectionUiPendingRef.current = { value: next };
            if (selectionUiRafRef.current == null) {
                selectionUiRafRef.current = requestAnimationFrame(() => {
                    selectionUiRafRef.current = null;
                    const pending = selectionUiPendingRef.current;
                    selectionUiPendingRef.current = null;
                    if (!pending) return;
                    setSelectionUi(pending.value);
                    if (morphModifierDownRef.current && !morphDragRef.current) {
                        setMorphOverlay(buildMorphOverlaysFromSelection());
                    }
                });
            }
        },
        [buildMorphOverlaysFromSelection, setMorphOverlay, setSelectionUi],
    );

    /**
     * 按拖拽工作副本渲染整段颤音。
     *
     * 走 `buildDragVibratoCurve`（→ `buildVibratoCurve`）：波形、包络（渐入 /
     * 渐强 / 渐出）、速率渐变、不规则度、基线模式全部来自预设，只有深度与速率
     * 是本次拖拽覆盖的。
     *
     * 吸附仍按**合成后的值**逐帧作用 —— 这是刻意保留的行为：开着吸附时用户
     * 可以快速画出一条量化的参数线。
     */
    const buildVibratoDense = useCallback(
        (
            startFrame: number,
            startValue: number,
            endFrame: number,
            endValue: number,
            working: VibratoDragWorking,
            shiftHeld: boolean,
            seed: number,
        ) => {
            return buildDragVibratoCurve({
                working,
                startFrame,
                startValue,
                endFrame,
                endValue,
                param: editParam,
                framePeriodMs: paramViewRef.current?.framePeriodMs ?? 5,
                range: currentParamRange,
                snapFinalValue: (value, frame) => snapDrawValue(value, shiftHeld, frame),
                seed,
            });
        },
        [editParam, currentParamRange, paramViewRef, snapDrawValue],
    );

    /**
     * 重画拖拽预览（参数或预设变更后调用）。
     *
     * 先擦掉**上一帧写过的区间**（而不是丢弃整份覆盖层再整份拷贝 —— 那是每帧
     * 一次 O(窗口) 的拷贝），再按当前工作副本写入。`applyDenseToLiveEdit`
     * 内部会按需重建基准。
     */
    const repaintVibratoDragPreview = useCallback(
        (shiftHeld: boolean): boolean => {
            const vib = vibratoStateRef.current;
            if (!vib) return false;

            const st = strokeRef.current;
            const pvNow = paramViewRef.current;
            if (!st || !pvNow || st.pointerId !== vib.pointerId) return false;

            resetLiveEditPreview(pvNow);
            const built = buildVibratoDense(
                vib.startFrame,
                vib.startValue,
                vib.currentFrame,
                vib.currentValue,
                vib.working,
                shiftHeld,
                vib.seed,
            );
            vib.shiftHeld = shiftHeld;
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
            return true;
        },
        [
            strokeRef,
            paramViewRef,
            buildVibratoDense,
            applyDenseToLiveEdit,
            resetLiveEditPreview,
            invalidate,
            requestWaveformRepaint,
        ],
    );

    /**
     * HUD 气泡的锚点（指针坐标）。
     *
     * 【为什么记下来】深度 / 速率也能由滚轮与方向键改，而**键盘事件没有指针坐标** ——
     * 若上报必须带上坐标，键盘调参就只能等下一次指针移动才刷新读数，用户看到的是
     * "气泡里的数字比曲线慢一拍"。这里存住最近一次指针位置，让所有调参路径都能
     * 立即上报（见 `nextVibratoHudAnchor`）。
     */
    const vibratoHudAnchorRef = useRef<VibratoHudAnchor | null>(null);

    /**
     * 把当前工作副本上报给面板（仅 HUD —— 切换的持久化走 dispatch）。
     *
     * @param clientX 新的指针横坐标；省略则沿用上一次的位置（滚轮 / 方向键调参）。
     * @param clientY 同上，纵坐标。
     */
    const reportVibratoDragState = useCallback(
        (clientX?: number, clientY?: number) => {
            const vib = vibratoStateRef.current;
            if (!vib || !onVibratoDragStateChange) return;
            vibratoHudAnchorRef.current = nextVibratoHudAnchor(
                vibratoHudAnchorRef.current,
                clientX,
                clientY,
            );
            const next = vibratoHudState(
                {
                    presetId: vib.working.preset.id,
                    depthCents: vib.working.depthCents,
                    rateHz: vib.working.rateHz,
                    depthAdjusted: vib.working.depthAdjusted,
                    rateAdjusted: vib.working.rateAdjusted,
                },
                vibratoHudAnchorRef.current,
            );
            if (next) onVibratoDragStateChange(next);
        },
        [onVibratoDragStateChange],
    );

    const applyVibratoDragAdjustment = useCallback(
        (input: {
            target: VibratoAdjustTarget;
            direction: 1 | -1;
            steps: number;
            shiftHeld: boolean;
            fineEvent: {
                ctrlKey: boolean;
                shiftKey: boolean;
                altKey: boolean;
                metaKey?: boolean;
            };
            clientX?: number;
            clientY?: number;
        }): boolean => {
            const vib = vibratoStateRef.current;
            if (!vib) return false;

            const fineScale = pointerFineWheelScale(input.fineEvent);
            const next = computeVibratoDragAdjustment({
                editParam,
                currentParamRange,
                depthCents: vib.working.depthCents,
                rateHz: vib.working.rateHz,
                target: input.target,
                direction: input.direction,
                steps: input.steps,
                fineScale,
            });
            vib.working = {
                ...vib.working,
                depthCents: next.depthCents,
                rateHz: next.rateHz,
                // 只标记被调的那一个：另一个在切换预设时仍取新预设自带的值。
                depthAdjusted: vib.working.depthAdjusted || input.target === "depth",
                rateAdjusted: vib.working.rateAdjusted || input.target === "rate",
            };

            repaintVibratoDragPreview(input.shiftHeld);
            // 曲线是命令式重画的（立即生效），读数走 React 状态。这里**无条件**上报：
            // 滚轮 / 方向键改的也是深度与速率，若因为"没有指针坐标"就跳过，气泡里的
            // 数字要等下一次指针移动才追上 —— 那正是"读数比曲线慢一拍"。
            reportVibratoDragState(input.clientX, input.clientY);
            return true;
        },
        [
            pointerFineWheelScale,
            editParam,
            currentParamRange,
            repaintVibratoDragPreview,
            reportVibratoDragState,
        ],
    );

    /**
     * 切换拖拽中的颤音预设。
     *
     * 波形 / 包络 / 摆放换成新预设（换的是"音色"）；本次手势若调过深度或速率，
     * 则这两个量跟着走（见 `switchDragPreset`）。切换后立刻按新预设重画 ——
     * 不重画的话用户要等下一次指针移动才看到变化。
     */
    const switchVibratoDragPreset = useCallback(
        (next: VibratoPreset, clientX?: number, clientY?: number): boolean => {
            const vib = vibratoStateRef.current;
            if (!vib) return false;
            vib.working = switchDragPreset(vib.working, next, editParam, currentParamRange);
            // 不规则度的种子跟着预设换：不同预设的"随机味"不一样，沿用旧种子
            // 会让两个预设的抖动图案完全相同。
            vib.seed = vibratoSeedForPreset(next);
            repaintVibratoDragPreview(vib.shiftHeld);
            // 与滚轮 / 方向键同理：换预设也要立刻刷新读数（预设名就在气泡里）。
            reportVibratoDragState(clientX, clientY);
            return true;
        },
        [editParam, currentParamRange, repaintVibratoDragPreview, reportVibratoDragState],
    );

    /**
     * 把本次手势切到直线工具：工具、工作副本与种子一起换。
     *
     * 【为什么不写活动预设】直线工具是**工具**，不是音色。顺手把用户记住的活动
     * 预设改成直线预设，等于"我只是画条直线，你把我选的音色忘了" —— 下次切回
     * 颤音工具时，他会发现自己挑的预设没了。工具与预设是两条独立的记忆。
     */
    const switchDragToLineTool = useCallback(
        (clientX?: number, clientY?: number): boolean => {
            const vib = vibratoStateRef.current;
            if (!vib) return false;
            // 无条件断言：手势画的是什么，工具就必须是什么。重复断言由
            // `setToolModePersistent` 判重后跳过写盘，因此这里是免费的。
            vib.tool = "line";
            void dispatch(setToolModePersistent("line"));
            vib.working = createDragWorking(straightVibratoPreset, editParam, currentParamRange);
            vib.seed = vibratoSeedForPreset(straightVibratoPreset);
            repaintVibratoDragPreview(vib.shiftHeld);
            reportVibratoDragState(clientX, clientY);
            return true;
        },
        [
            dispatch,
            straightVibratoPreset,
            editParam,
            currentParamRange,
            repaintVibratoDragPreview,
            reportVibratoDragState,
        ],
    );

    /**
     * 应用一次预设选择：要么切到直线工具，要么切到"颤音工具 + 某个预设"。
     *
     * 【语义】侧键 / `,` `.` / 预设子菜单与管理器是同一个动作 —— "我换了预设"。
     * 落在直线预设上时按 `VibratoChoice` 的规则变成"切到直线工具"（直线工具 ≡
     * 颤音工具 + 直线预设），因此**不必**在各调用点各写一份判断。
     *
     * 工作副本按 `switchDragPreset` 的规则更新：调过深度 / 速率就沿用本次手势的
     * 值（换音色不改幅度），没调过则取新预设自带的值。
     */
    const applyVibratoChoice = useCallback(
        (choice: VibratoChoice, clientX?: number, clientY?: number): boolean => {
            const vib = vibratoStateRef.current;
            if (!vib) return false;
            if (choice.tool === "line") return switchDragToLineTool(clientX, clientY);

            const next = findVibratoPreset(vibratoPresetCycleList, choice.presetId);
            if (!next) return false;
            // 从直线工具轮转出去 = 换回颤音工具：工具与预设一起变，否则会停在
            // "直线工具 + 一个颤音预设"这种既非此也非彼的状态上。
            //
            // 【为什么是断言而不是"仅当需要时切换"】手势的按键监听是 pointerdown
            // 时装上的，闭包里的 `toolMode` 属性停留在那一刻 —— 拿它做判断，从直线
            // 工具切回颤音预设时就会漏掉回切，出现"画的是颤音、工具却停在直线"。
            // 干脆每次都断言，重复的那次由 `setToolModePersistent` 判重跳过。
            vib.tool = "vibrato";
            void dispatch(setToolModePersistent("vibrato"));
            dispatch(setActiveVibratoPreset(next.id));
            void dispatch(persistUiSettings());
            return switchVibratoDragPreset(next, clientX, clientY);
        },
        [dispatch, vibratoPresetCycleList, switchDragToLineTool, switchVibratoDragPreset],
    );

    /**
     * 双键重置：把本次手势的某个量（或整个工具）拉回已知状态。
     *
     * - `line`：切到直线工具并清掉两条调整记录 —— 直接画一条直线，不必先切走再切回。
     * - `depth` / `rate`：撤销本次手势对该量的微调（回到预设自带值）并清掉记录，
     *   于是下一次切换预设时该量会取新预设自己的值。
     */
    const applyVibratoReset = useCallback(
        (intent: VibratoDragResetIntent, clientX?: number, clientY?: number): boolean => {
            const vib = vibratoStateRef.current;
            if (!vib) return false;

            if (intent === "line") {
                return switchDragToLineTool(clientX, clientY);
            }
            if (intent === "depth") {
                vib.working = resetVibratoDragDepth(vib.working, editParam, currentParamRange);
            } else {
                vib.working = resetVibratoDragRate(vib.working);
            }

            repaintVibratoDragPreview(vib.shiftHeld);
            // 重置同样立刻刷新读数：深度 / 速率回到预设值，气泡不能停在旧数字上。
            reportVibratoDragState(clientX, clientY);
            return true;
        },
        [
            editParam,
            currentParamRange,
            switchDragToLineTool,
            repaintVibratoDragPreview,
            reportVibratoDragState,
        ],
    );

    /** 拖拽结束时的收尾：清掉双键记录与气泡锚点并上报（面板据此清掉 HUD）。 */
    const finishVibratoDrag = useCallback(() => {
        vibratoResetSlotsRef.current.clear();
        // 锚点属于本次手势：留着会让下一次起手在"还没动过指针"时先用旧位置闪一下。
        vibratoHudAnchorRef.current = null;
        // 步进累加器同样属于本次手势：残留的不足一步的余量会带进下一次
        // 滚轮手势，让第一档提前（或跳过）一步。
        vibratoPinchRef.current = createPinchStepState();
        if (!onVibratoDragEnd) return;
        onVibratoDragEnd();
    }, [onVibratoDragEnd]);

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

    /**
     * 视口 x → **连续**帧坐标（未取整）。
     *
     * 帧栅格是工程级常量（`frame_period_ms` 恒为 5.0），因此这个映射与 BPM 无关 ——
     * 选区的所有几何都由它决定，改 BPM 不会让选区移动。
     *
     * 【为什么不在这里量化】量化只发生在 `paramSelection` 的指针规则里
     * （`frameBoundLeftFromPointer` / `frameBoundRightFromPointer`）。曲线取值路径
     * （`frameToIndex`）自己会四舍五入；在这里量化会让"边界在哪"出现第二处定义。
     */
    const pointerFrame = useCallback(
        (clientX: number): number => {
            const canvas = canvasRef.current;
            if (!canvas) return 0;
            const rect = canvas.getBoundingClientRect();
            // 逆投影走 axis，与 pointerSec 同源（此前写作 `(scrollLeft + x) / pxPerBeat`，
            // 是又一套独立算式）。
            return frameFromViewportClientX({
                clientX,
                viewportLeft: rect.left,
                axis: axisFromRefs(),
                framePeriodMs,
            });
        },
        [canvasRef, axisFromRefs, framePeriodMs],
    );

    const pointerSec = useCallback(
        (clientX: number): number => {
            const canvas = canvasRef.current;
            if (!canvas) return 0;
            const rect = canvas.getBoundingClientRect();
            return secFromViewportClientX({
                clientX,
                viewportLeft: rect.left,
                // 与 `pointerFrame` 同源：一律走 `axisFromRefs`（内核真值的投影），
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
        (e: ReactMouseEvent<HTMLDivElement>) =>
            handleRulerPointerDown(
                {
                    axisFromRefs,
                    dispatch,
                },
                e,
            ),
        [axisFromRefs, dispatch],
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
        (e: KeyboardEvent<HTMLDivElement>) =>
            handleScrollerKeyDown(
                {
                    editParam,
                    keybindingMap,
                    onEditAction,
                    pitchEnabled,
                    rootTrackId,
                    toolMode,
                },
                e,
            ),
        [rootTrackId, editParam, pitchEnabled, keybindingMap, onEditAction, toolMode],
    );

    /**
     * 最近一次指针移动事件与"指针是否按下"。
     *
     * 【用途】视口在拖拽期间被改变（滚轮滚动/缩放、跨面板同步）后，按最近指针位置
     * **重放**一次移动 —— 各手势的落点换算读的是实时内核视口，因此重放后对象立刻
     * 回到光标处，不必等用户真的动鼠标。
     *
     * 【为什么按键状态要单独记】重放载荷里的 `buttons` **不能**沿用最近一次
     * `pointermove` 的：按下之后、第一次真实移动之前，那个"最近一次"是按下**之前**
     * 的悬停事件（`buttons === 0`），而各手势把 `buttons` 当作"键还按着吗"的判据
     * （`(ev.buttons & mask) !== mask → onUp()`），会把重放读成"用户松手了"并当场
     * 收尾手势 —— 用绘制工具画 `音量` / `动态` 时"按下那一点在、拖拽轨迹完全不画"
     * 正是这样发生的（详见 `viewportReplay.ts` 的模块头说明）。
     */
    const lastPointerMoveRef = useRef<globalThis.PointerEvent | null>(null);
    const pointerDownRef = useRef(false);
    /**
     * **此刻**按下的按键位掩码（0 = 没有任何键按下）。
     *
     * 由同一组捕获监听维护：`pointerdown` 记下起始掩码、`pointermove` 同步、
     * `pointerup` / `pointercancel` 清零。这是重放载荷唯一的按键来源。
     */
    const pressedButtonsRef = useRef(0);

    useEffect(() => {
        const onMove = (event: globalThis.PointerEvent) => {
            lastPointerMoveRef.current = event;
            // 只在**有键按下**时同步：一次"没有键按下"的移动不可能发生在拖拽中
            // （那需要先来一次 pointerup），照单全收会把掩码清零、让重放失效并误报
            // 一条"按键丢失"告警。
            if (event.buttons !== 0) pressedButtonsRef.current = event.buttons;
        };
        const onDown = (event: globalThis.PointerEvent) => {
            pressedButtonsRef.current = event.buttons;
            if (event.button === 0) pointerDownRef.current = true;
        };
        const onUp = () => {
            pointerDownRef.current = false;
            pressedButtonsRef.current = 0;
        };
        // 捕获阶段：即使某个手势 stopPropagation 也要记到。
        window.addEventListener("pointermove", onMove, true);
        window.addEventListener("pointerdown", onDown, true);
        window.addEventListener("pointerup", onUp, true);
        window.addEventListener("pointercancel", onUp, true);
        return () => {
            window.removeEventListener("pointermove", onMove, true);
            window.removeEventListener("pointerdown", onDown, true);
            window.removeEventListener("pointerup", onUp, true);
            window.removeEventListener("pointercancel", onUp, true);
        };
    }, []);

    /**
     * 视口变化 → 若正在拖拽则重放一次指针移动。
     *
     * 【为什么需要】落点由「实时视口 + 指针位置」决定。拖拽期间用滚轮滚动或缩放时，
     * 视口变了而指针没动；不重放的话被拖对象会停在旧落点上、与光标脱开，直到用户再
     * 动一下鼠标才归位（用户报告的"拖拽中滚动/缩放导致偏移"）。
     *
     * 【为什么必须自己判投影】重放挂在总线的 `subscribe()` 上，而
     * `pianoRollViewportBus.invalidate()`（"内容变了，按同一投影重画一次"）同样会
     * paint 全部订阅者 —— 绘制 `音量` / `动态` 时**每帧**一次的波形重绘就会走到这里。
     * 不区分"视口变了"与"只是重绘"，重放就会被无谓注入，并构成
     * `invalidate → 重放 → 移动 → invalidate` 的自激环。
     *
     * 【全部判定规则与理由收口在 `viewportReplay`】：包括载荷的按键状态必须取
     * "此刻"而非最近一次事件 —— 那是"按下那一点在、拖拽轨迹完全不画"的成因。
     */
    useEffect(() => {
        let lastProjection: ViewportProjection | null = null;
        return pianoRollViewportBus.subscribe((scrollLeftPx, pxPerSec, viewportWidthPx) => {
            const projection: ViewportProjection = { scrollLeftPx, pxPerSec, viewportWidthPx };
            const decision = resolveViewportReplay({
                projection,
                previousProjection: lastProjection,
                dragging: pointerDownRef.current,
                pressedButtons: pressedButtonsRef.current,
                lastMove: lastPointerMoveRef.current,
            });
            // 无论是否重放都要推进基准：漏推一次会让下一次比较错位，真实滚动反而被
            // 当成"没变"而漏掉重放。
            lastProjection = projection;
            if (!decision.replay) {
                if (decision.reason === "pressed-buttons-lost") {
                    warnViewportReplayButtonsLost(lastPointerMoveRef.current?.buttons ?? null);
                }
                return;
            }
            window.dispatchEvent(new PointerEvent("pointermove", decision.payload));
        });
    }, []);

    /**
     * 滚轮总入口（滚动 / 平移 / 缩放）。画布、标尺与自绘滚动条共用它。
     *
     * @param e 原生 wheel 事件。
     * @param forcedScrollbarZone 调用方**已判定**的滚动条轴，只有**自绘滚动条轨道**
     *   会传。轨道位于滚动容器之外（见 PianoRollPanel 的轨道注释），其上的滚轮不会
     *   冒泡到 `scroller`，因此由轨道自己把轴「托付」进来 —— 后续语义（该轴滚动 /
     *   `modifier.scrollbarZoom` 该轴缩放）与画布内的 `nativeScrollbarZoneAt` 同源，
     *   不存在第二套口径。
     *   省略时按指针位置自行判定：画布内命中原生滚动条区，而**标尺**在容器矩形
     *   之外 ⇒ 判定为 null ⇒ 走的就是画布内的那一套 keybinding 语义。
     */
    const onScrollerWheelNative = useCallback(
        (e: globalThis.WheelEvent, forcedScrollbarZone?: ScrollbarZone) =>
            handleScrollerWheel(
                {
                    applyVibratoDragAdjustment,
                    canvasRef,
                    clampViewport,
                    currentParamRange,
                    dynamicProjectSec,
                    editParam,
                    getPlayheadSec,
                    horizontalZoomChainRef,
                    horizontalZoomKb,
                    invalidate,
                    onHorizontalZoom,
                    paramViewsRef,
                    pitchViewRef,
                    playheadZoomEnabled,
                    prVerticalZoomKb,
                    pxPerSecRef,
                    scrollHorizontalKb,
                    scrollLeftRef,
                    scrollVerticalKb,
                    scrollbarZoomKb,
                    scrollerRef,
                    setParamViewport,
                    setPitchView,
                    syncScrollLeft,
                    syncTimelineEnabled,
                    timelineOffsetRef,
                    vibratoAmplitudeAdjustKb,
                    vibratoFrequencyAdjustKb,
                    vibratoPinchRef,
                    vibratoStateRef,
                    yToValue,
                    yToViewportT,
                },
                e,
                forcedScrollbarZone,
            ),
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
        const resetBindings: VibratoDragResetBindings = {
            presetPrev: vibratoPresetPrevKb,
            presetNext: vibratoPresetNextKb,
            amplitudeIncrease: vibratoDragAmplitudeIncreaseKb,
            amplitudeDecrease: vibratoDragAmplitudeDecreaseKb,
            frequencyIncrease: vibratoDragFrequencyIncreaseKb,
            frequencyDecrease: vibratoDragFrequencyDecreaseKb,
        };

        const onKeyDown = (e: globalThis.KeyboardEvent) => {
            if (!vibratoStateRef.current) return;

            // ── 双键重置 ──────────────────────────────────────────────
            // 先登记这次按住了哪些绑定槽，再看是否凑成一对。放在最前是因为凑成
            // 一对时本次按键就是"重置"，不该再落进下面的单键切换 / 调参分支 ——
            // 否则会先切一次预设、再重置，两个动作叠在一起。
            const matched = matchedVibratoResetSlots(e, resetBindings, paramFineAdjustKb);
            for (const slot of matched) vibratoResetSlotsRef.current.add(slot);
            if (matched.length > 0 && !e.repeat) {
                const intent = resolveVibratoPairReset(vibratoResetSlotsRef.current);
                if (intent) {
                    e.preventDefault();
                    e.stopPropagation();
                    applyVibratoReset(intent);
                    return;
                }
            }

            // 预设切换先判：`,` / `.` 的命中与深度 / 速率的绑定互斥（默认值
            // 不重叠，但用户可能把它们绑到同一个键上），先把离散跳转判掉。
            const switchDirection = resolveVibratoPresetSwitch(
                e,
                vibratoPresetPrevKb,
                vibratoPresetNextKb,
                paramFineAdjustKb,
            );
            if (switchDirection) {
                e.preventDefault();
                e.stopPropagation();
                // 起点按**工具**推导：直线工具在序列里占着直线预设的位置，
                // 拿活动预设当起点会让"从直线工具轮转一步"跳到别处。
                const anchorId = vibratoCycleAnchorId({
                    lineTool: vibratoStateRef.current.tool === "line",
                    currentPresetId: vibratoStateRef.current.working.preset.id,
                });
                const choice = cycleVibratoChoice(
                    vibratoPresetCycleList,
                    anchorId,
                    switchDirection,
                );
                if (choice) applyVibratoChoice(choice);
                return;
            }

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

        // 松键即注销该槽。不依赖拖拽状态：拖拽可能在两次按键之间结束，
        // 那时仍要把槽清掉（拖拽结束本身也会整体清空，这里是兜底）。
        const onKeyUp = (e: globalThis.KeyboardEvent) => {
            const matched = matchedVibratoResetSlots(e, resetBindings, paramFineAdjustKb);
            for (const slot of matched) vibratoResetSlotsRef.current.delete(slot);
        };

        // 失焦（切到别的窗口）时按键状态会丢失：整体清空，避免残留把后续的
        // 单击误判成"双键"。
        const onBlur = () => vibratoResetSlotsRef.current.clear();

        window.addEventListener("keydown", onKeyDown, true);
        window.addEventListener("keyup", onKeyUp, true);
        window.addEventListener("blur", onBlur);
        return () => {
            window.removeEventListener("keydown", onKeyDown, true);
            window.removeEventListener("keyup", onKeyUp, true);
            window.removeEventListener("blur", onBlur);
        };
    }, [
        applyVibratoDragAdjustment,
        applyVibratoReset,
        applyVibratoChoice,
        toolMode,
        vibratoDragAmplitudeIncreaseKb,
        vibratoDragAmplitudeDecreaseKb,
        vibratoDragFrequencyIncreaseKb,
        vibratoDragFrequencyDecreaseKb,
        vibratoPresetPrevKb,
        vibratoPresetNextKb,
        vibratoPresetCycleList,
        paramFineAdjustKb,
    ]);

    const getDefaultCanvasCursor = useCallback((): CanvasCursor => {
        return toolMode === "select" ? "default" : "crosshair";
    }, [toolMode]);

    /**
     * 指针 x 处**参数线上的点**（沿渲染折线插值，见 `curvePointAtPointer`）：
     * 像素 y + 参数值。无数据 / 指针在曲线时间范围外时为 null。
     */
    const getCurvePointAtPointer = useCallback(
        (clientX: number): { y: number; value: number } | null => {
            const pv = paramViewRef.current;
            const canvas = canvasRef.current;
            if (!pv || pv.edit.length === 0 || !canvas) return null;

            const rect = canvas.getBoundingClientRect();
            const rectH = rect.height || viewSizeRef.current.h || 1;
            // 插值 + 投影 + pitch 的 +0.5 偏移都在纯函数里一次算完（有单测），
            // 与 `projectCurvePoints` 的折线逐像素同源。
            return curvePointAtPointer({
                sec: pointerSec(clientX),
                startFrame: pv.startFrame,
                stride: pv.stride,
                framePeriodMs: pv.framePeriodMs,
                values: pv.edit,
                param: editParam,
                valueToY: (v) => valueToY(editParam, v, rectH),
            });
        },
        [paramViewRef, canvasRef, pointerSec, editParam, valueToY, viewSizeRef],
    );

    const getCurveValueNearPointer = useCallback(
        (clientX: number, clientY: number, pointerType?: string | null): number | null => {
            const point = getCurvePointAtPointer(clientX);
            if (point == null) return null;

            const canvas = canvasRef.current;
            if (!canvas) return null;
            const rect = canvas.getBoundingClientRect();
            // 邻域判定抽到 `kernel/gestureHitTest`（纯函数，有单测）：只比纵向距离，
            // 曲线的位置由 `getCurvePointAtPointer` 一次算定。
            // 命中半径按设备剖面缩放：手指接触面约 9mm，10px 的半径抓不住曲线。
            return isPointerNearCurve({
                pointerY: clientY - rect.top,
                curveY: point.y,
                hitRadiusPx: scaledHitRadius(CURVE_HIT_RADIUS_PX, profileFor({ pointerType })),
            })
                ? point.value
                : null;
        },
        [getCurvePointAtPointer, canvasRef],
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
        const point = getCurvePointAtPointer(last.x);
        if (point == null || !Number.isFinite(point.value)) {
            onParamValuePreviewChange?.(null);
            return;
        }
        onParamValuePreviewChange?.({
            clientX: last.x,
            clientY: last.y,
            value: point.value,
        });
    }, [
        paramValuePopupEnabled,
        onParamValuePreviewChange,
        getCurvePointAtPointer,
        strokeRef,
        panRef,
        paramViewRef,
    ]);

    const isPointerNearDraggableSelection = useCallback(
        (clientX: number, clientY: number, pointerType?: string | null): boolean => {
            if (toolMode !== "select") return false;
            const sel = selectionRef.current;
            const canvas = canvasRef.current;
            if (!sel || sel.length === 0 || !canvas) return false;

            // 判定抽到 `kernel/gestureHitTest`（纯函数，有单测）。
            //
            // 【为什么把区间比较改成像素区间比较是等价的】原实现是
            // `beat < aBeat || beat > bBeat`，而 `frameToViewportPx` 是
            // `pxPerSec > 0` 下的单调线性投影，且 `pointerFrame` 正是它的逆
            // （同一 `axisFromRefs`、同一 `rect.left`）。因此「帧落在区间内」
            // 与「像素落在区间内」同真同假。改用像素后，与边缘命中判定共用
            // 同一套坐标口径，不再有两份换算。
            //
            // 多选区：落在**任一段**内且靠近曲线即可拖动（拖动会带起所有段）。
            const nearCurve = getCurveValueNearPointer(clientX, clientY, pointerType) != null;
            if (!nearCurve) return false;
            const rect = canvas.getBoundingClientRect();
            const localXPx = clientX - rect.left;
            return sel.some((range) =>
                hitTestSelectionBody({
                    // 选区主体画在**切点**之间（两帧中间），命中测试必须同一口径。
                    leftXPx: frameToViewportPx(frameRangeStartCut(range)),
                    rightXPx: frameToViewportPx(frameRangeEndCut(range)),
                    localXPx,
                    nearCurve,
                }),
            );
        },
        [toolMode, selectionRef, canvasRef, frameToViewportPx, getCurveValueNearPointer],
    );

    /**
     * 边缘拉伸命中：在 `modifier.paramStretch`（默认 Alt）按下时，找**最近**的
     * 选区段边缘。多选区下返回被命中的段号 + 哪一侧 —— 拉伸只作用于那一段，
     * 其余段不动。
     *
     * 【`requireModifier` 的两种用法】缺省（`true`）＝既有的"按住 Alt 拉伸"，也是
     * 光标提示（`ew-resize`）所依据的判定；传 `false` ＝**不按任何修饰键**也能抓
     * 边缘拖拽（用户报告的期望行为）。后者在 pointerdown 里要让位给"已选中参数线"
     * 的拖拽，见那里的优先级说明。
     */
    const findStretchSelectionEdge = useCallback(
        (
            e: ReactPointerEvent<HTMLCanvasElement>,
            options?: { readonly requireModifier?: boolean },
        ): { rangeIndex: number; edge: "left" | "right" } | null => {
            if (toolMode !== "select") return null;
            if (
                (options?.requireModifier ?? true) &&
                !isModifierActive(paramStretchKb, e.nativeEvent)
            ) {
                return null;
            }
            const sel = selectionRef.current;
            const canvas = canvasRef.current;
            if (!sel || sel.length === 0 || !canvas) return null;
            const rect = canvas.getBoundingClientRect();
            // 命中判定抽到 `kernel/gestureHitTest`（纯函数，有单测）：逐段调
            // `hitTestSelectionEdge`（阈值 `SELECTION_EDGE_HIT_PX`，与原内联的
            // 8px 一致），在所有段中取**最近**的边缘——拉伸只作用于被抓住的
            // 那一段，其余段不动。同段左缘优先、跨段取先出现者，与原实现一致。
            // 带宽按设备剖面缩放：手指够不到 8px 的边缘。
            const localXPx = e.clientX - rect.left;
            const edgeHitWidth = scaledHitRadius(SELECTION_EDGE_HIT_PX, profileFor(e.nativeEvent));
            let best: { rangeIndex: number; edge: "left" | "right" } | null = null;
            let bestDistance = Number.POSITIVE_INFINITY;
            for (let i = 0; i < sel.length; i += 1) {
                // 两条边界的像素位置取自**切点**（与绘制同一口径）。
                const leftXPx = frameToViewportPx(frameRangeStartCut(sel[i]));
                const rightXPx = frameToViewportPx(frameRangeEndCut(sel[i]));
                const edge = hitTestSelectionEdge({
                    leftXPx,
                    rightXPx,
                    localXPx,
                    hitWidthPx: edgeHitWidth,
                });
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
        [toolMode, paramStretchKb, selectionRef, canvasRef, frameToViewportPx],
    );

    const isPointerNearStretchSelectionEdge = useCallback(
        (e: ReactPointerEvent<HTMLCanvasElement>): boolean => findStretchSelectionEdge(e) !== null,
        [findStretchSelectionEdge],
    );

    /**
     * 无修饰键的边缘命中（不按任何修饰键、直接左键拖拽边缘即可拉伸）。
     *
     * 与 {@link isPointerNearStretchSelectionEdge} 分开命名：前者是"Alt 拉伸"的
     * 提示与路径，后者是新增的直接拖拽路径，两者在 pointerdown 里的**优先级不同**
     * （已选中的参数线要压过后者），混用一个判定会让优先级失去表达。
     */
    const isPointerNearSelectionEdge = useCallback(
        (e: ReactPointerEvent<HTMLCanvasElement>): boolean =>
            findStretchSelectionEdge(e, { requireModifier: false }) !== null,
        [findStretchSelectionEdge],
    );

    const onCanvasPointerMove = useCallback(
        (e: ReactPointerEvent<HTMLCanvasElement>) =>
            handleCanvasPointerMove(
                {
                    editParam,
                    getCurveValueNearPointer,
                    getDefaultCanvasCursor,
                    hoverPreviewNearCurveRef,
                    isEffectivePitchSnapActive,
                    isPointerNearDraggableSelection,
                    isPointerNearSelectionEdge,
                    isPointerNearStretchSelectionEdge,
                    lastPointerClientRef,
                    morphDragRef,
                    onParamValuePreviewChange,
                    panRef,
                    paramMultiSelectKb,
                    paramStretchKb,
                    paramValuePopupEnabled,
                    pitchSnapToleranceCents,
                    pitchSnapUnit,
                    pointerSec,
                    pointerValue,
                    projectScale,
                    scaleAtSec,
                    setCanvasCursor,
                    strokeRef,
                    toolMode,
                },
                e,
            ),
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
            isPointerNearSelectionEdge,
            paramStretchKb,
            paramMultiSelectKb,
            setCanvasCursor,
            getDefaultCanvasCursor,
            pointerSec,
        ],
    );

    const onCanvasPointerLeave = useCallback(() => {
        if (panRef.current || strokeRef.current || morphDragRef.current) return;
        // 指针离开画布：悬停激活状态与最后位置一并失效，避免离画布后的
        // 数据刷新用过期坐标续显浮窗。
        hoverPreviewNearCurveRef.current = false;
        lastPointerClientRef.current = null;
        onParamValuePreviewChange?.(null);
        setCanvasCursor(getDefaultCanvasCursor());
    }, [panRef, strokeRef, onParamValuePreviewChange, setCanvasCursor, getDefaultCanvasCursor]);

    const onCanvasPointerDown = useCallback(
        (e: ReactPointerEvent<HTMLCanvasElement>) =>
            handleCanvasPointerDown(
                {
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
                },
                e,
            ),
        [
            rootTrackId,
            editParam,
            toolMode,
            straightVibratoPreset,
            scrollerRef,
            // 边缘自动滚屏与视口真值：`applyEdgeScrollLeft` / `getViewportTruth`
            // 在调用方都是稳定引用（前者是 `useCallback`，后者的依赖已在本数组
            // 里），`scrollLeftRef` 是 ref —— 列入依赖不会让本回调每帧重建，
            // 只是把"确实读到的值"如实登记。
            applyEdgeScrollLeft,
            getViewportTruth,
            scrollLeftRef,
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
            requestWaveformRepaint,
            pointerFrame,
            pointerSec,
            selectionRef,
            updateSelectionUi,
            paramMultiSelectKb,
            findStretchSelectionEdge,
            isPointerNearDraggableSelection,
            frameToViewportPx,
            fetchCommitBaseSource,
            buildUploadSegments,
            paramViewRef,
            ensureLiveEditBase,
            paramView,
            framePeriodMs,
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
            // 颤音拖拽：预设起手、拖拽中切换预设、双键 / 中键重置、HUD 上报与收尾清屏。
            // 全量列表 `vibratoPresetList` 只在 `applyVibratoReset` 内部用到，故由它
            // 自己带着依赖，这里只列循环切换用的 `vibratoPresetCycleList`。
            vibratoPreset,
            vibratoPresetCycleList,
            // 深度按当前参数的满摆幅换算与钳制（`createDragWorking`），因此起手
            // 也依赖参数与它的量程。
            currentParamRange,
            applyVibratoChoice,
            applyVibratoReset,
            vibratoFrequencyAdjustKb,
            reportVibratoDragState,
            finishVibratoDrag,
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
