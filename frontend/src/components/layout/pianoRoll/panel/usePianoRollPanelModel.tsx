import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useAppDispatch, useAppSelector, useAppStore } from "../../../../app/hooks";
import type { RootState } from "../../../../app/store";
import { useI18n } from "../../../../i18n/I18nProvider";
import {
    setEditParam,
    setEdgeSmoothnessPercent,
    setTrackStateRemote,
    cycleDragDirection,
    persistUiSettings,
    applyClipSyllableAnalysis,
    markProjectEdited,
    setParamBlocksVisible,
    toggleParamBlocksVisible as toggleParamBlocksVisibleAction,
} from "../../../../features/session/sessionSlice";
import { resolveRootTrackId } from "../../../../features/session/trackUtils";
import {
    buildPitchAlgoOptions,
    resolveEffectivePitchAlgo,
} from "../../../../features/tracks/pitchAlgoOptions";
import { useAppTheme } from "../../../../theme/AppThemeProvider";
import { timelineApi } from "../../../../services/api";
import { getWaveformColors } from "../../../../theme/waveformColors";
import { isScaleKey } from "../../../../utils/musicalScales";
import {
    firstBinding,
    isModifierActive,
    isNoneBinding,
} from "../../../../features/keybindings/keybindingsSlice";
import {
    SEPARATION_PARAM_ID,
    firstGatedParamId,
    isSeparationEnabled,
} from "../../../../features/pitch/separationGate";
import { useNonPassiveWheel } from "../../../../utils/useNonPassiveWheel";
import type { ScaleLike } from "../../../../utils/musicalScales";
import { clamp } from "../../timeline";
import { effectiveScaleAtSec } from "../../../../utils/tempoMap";
import { publishPianoRollSelection } from "../../../../utils/pianoRollSelectionBus";
import { resolveTimelineScrollRange } from "../../timeline/runtime/timelineScrollRange";
import { TOOLBAR_MAX_TIER } from "../toolbarOverflow";
import { useToolbarOverflowTier } from "../useToolbarOverflowTier";
import {
    selectionBoundingSpan,
    selectionFromPointers,
    selectionFromFrames,
    rangeIndexAtFrame,
} from "../paramSelection";
import { paramBlockIdsInSelection, paramBlockRestoreRanges } from "../paramBlockSelection";
import {
    captureParamBlockCreateCurve,
    enqueueParamBlockCreateCommit,
} from "../paramBlockCreationTransaction";
import { usePianoRollData } from "../usePianoRollData";
import { useParamBlockDragHandoff } from "../useParamBlockDragHandoff";
import { createSyllableAnalysisQueue } from "../syllableAnalysisQueue";
import { useParamBlockPersistence } from "../useParamBlockPersistence";
import { composeParamBlockView } from "../paramBlockComposition";
import { toParamBlockSourceClip } from "../paramBlockProjection";
import type { ParamBlockView } from "../paramBlockTypes";
import { fetchFullResCurve, readPvRange, uploadFullResCurveSegments } from "../selectionEditData";
import { startPointerGesture } from "../gestures/pointerGesture";
import { armRightDragContextMenuGuard } from "../../../../utils/rightDragContextMenuGuard";
import { normalizeCssColor } from "../colors";
import { parseRgbaColor } from "../../timeline/runtime/timelineClipGlRenderer";
import {
    EMPTY_PARAM_BLOCK_STATE,
    averageParamValueInSpan,
    canGroupParamBlocks,
    deleteParamBlocks,
    formatParamBlockLabel,
    paramBlockCursorForHover,
    hitTestParamBlockEdges,
    hitTestParamBlockTransitionHandles,
    hitTestParamBlocks,
    hitTestParamBlocksAtPoint,
    materializeParamBlocks,
    blocksFromViewForPanel,
    groupParamBlocks,
    paramBlockResizeCandidates,
    projectParamBlockResizeDisplay,
    applyFullResSegmentsToParamView,
    projectParamBlockCreatePreviewFromBase,
    shiftParamBlockCurveValuesToTarget,
    shouldUseParamBlockDragSnapshotForRender,
    nearestParamBlockByTime,
    isStrictlyInsideParamBlock,
    resolveParamBlockSplitSec,
    resolveSplittableParamBlockAt,
    resolveParamBlockCreateSpan,
    resolveParamBlockResizePlan,
    resolveParamBlockResizeTarget,
    ungroupParamBlocks,
    resizeParamBlock,
    removeParamBlockAtMinWidth,
    restoreDeletedParamBlocks,
    splitParamBlock,
    setParamBlockTransition,
    buildParamBlockTransitionSegments,
    buildParamBlockTransitionPreview,
    PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX,
    PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX,
    sortedParamBlockTransitionHandles,
    sortParamBlocksByTime,
    createParamBlock,
    clipAtParamBlockCreateSpan,
    resolveCreatedParamBlockValue,
    paramViewContainsCommittedSegments,
    newParamBlockId,
    type ParamBlock,
    type ParamBlockState,
} from "../paramBlocks";
import { resolveParamBlockFrameView, resolveParamBlockPanelValues } from "../paramBlockFrame";
import { isParamAuthorityConfirmedAfterCommit } from "../paramBlockTransaction";
import {
    beginParamBlockDrag,
    type ParamBlockDragStart,
    type ParamBlockDragSnapshot,
    type ParamBlockDragCommit,
    type ParamBlockDragActivity,
} from "../paramBlockDrag";
import { publishBlockAnalysisProgress } from "../blockAnalysisProgress";
import { normalizePitchToNearestSemitone } from "../paramBlockPitch";
import { getDrawPreviewValue } from "../paramValuePreviewLogic";
import {
    clearParamBlockSnapHighlight,
    publishParamBlockSnapHighlight,
    resolveParamBlockPitchSnapDelta,
    resolveParamBlockTimeSnapResult,
} from "../paramBlockSnap";
import { publishParamBlockFeedback } from "../paramBlockFeedback";
import { warnDev } from "../devWarn";
import { backendDetail } from "../../../../services/invoke";
import { makeLoudnessAmplitudeMap } from "../PianoRollWaveformSurface";
import { formatDbReadout, resolveParamAxisUnit, supportsParamAxisUnit } from "../paramAxisUnits";
import { useLoudnessCurves, type LoudnessSnapshot } from "../useLoudnessCurves";
import { createLiveOverrideReader, type LiveOverrideReader } from "../liveLoudnessOverride";
import { pianoRollViewportBus } from "../pianoRollViewportBus";
import { createLoudnessGeometryWarp, type LoudnessGeometryWarp } from "../loudnessGeometryWarp";
import {
    getClipGeometryPreviewOrigin,
    subscribeClipGeometryPreview,
    type ClipGeometrySnapshot,
} from "../../timeline/clipGeometryPreviewBus";
import { createRenderLoop, type RenderLoop } from "../../renderKernel/renderLoop.js";
import {
    createTimelineAxis,
    playheadLineLeftPx,
    viewportPxToSec,
} from "../../renderKernel/timelineAxis.js";
import { usePianoRollInteractions } from "../usePianoRollInteractions";
import { resolveMenuMaxHeight } from "../../menuPlacement";
import {
    enabledVibratoPresets,
    resolveActiveVibratoPreset,
    resolveVibratoPresets,
    findVibratoPreset,
    systemVibratoPreset,
} from "../../../../features/vibrato/vibratoPresetList";
import { STRAIGHT_VIBRATO_PRESET_ID } from "../../../../features/vibrato/systemPresets";
import { vibratoPresetLabel } from "../../vibrato/vibratoDialogLogic";
import {
    buildChildPitchOffsetCentsParam,
    buildChildPitchOffsetDegreesParam,
    buildChildFormantOffsetCentsParam,
    childPitchOffsetValueToDisplay,
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
    isChildFormantOffsetCentsParam,
    isChildPitchOffsetParam,
} from "../childPitchOffsetParams";
import { readDevicePixelRatio } from "../../../../utils/devicePixelLine";
import {
    subscribeDevicePixelRatio,
    useDevicePixelRatio,
} from "../../../../hooks/useDevicePixelRatio";
import { toggleSecondaryParamVisibility } from "../secondaryOverlaySelection";
import type { ParamName, ParamViewSegment, ValueViewport } from "../types";
import {
    selectKeybinding,
    selectKeybindings,
    selectMergedKeybindings,
} from "../../../../features/keybindings/keybindingsSlice";
import { clearPianoRollLoading, setPianoRollLoading } from "../../../../utils/pianoRollStatusBus";
import { parseCustomScaleToken } from "../../../../utils/scaleSelection";
import { type PianoRollKernelHost } from "../kernel/host/pianoRollKernelHost";
import type {
    MutablePianoRollKernelData,
    PianoRollGridSpec,
    PianoRollParamBlockSpec,
} from "../kernel/host/pianoRollKernelData";
import { resolvePanelRenderViewport } from "../kernel/viewportSource";
import { NOTE_NAMES_SHARP } from "./panelHelpers";
import { usePianoRollParamGates } from "./usePianoRollParamGates";
import { usePianoRollScrollerWiring } from "./usePianoRollScrollerWiring";
import { usePianoRollSelectionActions } from "./usePianoRollSelectionActions";
import { useTimelineDisplayControls } from "./useTimelineDisplayControls";
import { buildToolIcons } from "./toolIcons";
import { usePianoRollEditCommands } from "./usePianoRollEditCommands";
import { usePianoRollEditSession } from "./usePianoRollEditSession";
import { usePianoRollFrameAssembly } from "./usePianoRollFrameAssembly";
import { usePianoRollScrollKernel } from "./usePianoRollScrollKernel";
import { useMidiImport } from "./useMidiImport";
import {
    pitchDeltaToDegreeSteps,
    buildGridSpec as buildGridSpecImpl,
    valueToY as valueToYImpl,
    yToViewportT as yToViewportTImpl,
    yToValue as yToValueImpl,
    clampViewport as clampViewportImpl,
    getParamValueBoundsForScrollbar as getParamValueBoundsForScrollbarImpl,
    getCurrentViewportForScrollbar as getCurrentViewportForScrollbarImpl,
    syncVerticalScrollbarForViewport as syncVerticalScrollbarForViewportImpl,
} from "./sceneBuilders";
import { usePianoRollViewport } from "./usePianoRollViewport";
import { useProcessorParams } from "./useProcessorParams";

/**
 * 高频会话字段：它们由 `PianoRollPlayheadBridge` 单独订阅，面板主体不因它们重渲染。
 *
 * 【为什么必须从面板选择器里剔除】`playheadSec` 与 `runtime` 在播放时约 30Hz 更新；
 * 面板有数百个 hook / 子组件，若随每次轮询重渲染，60Hz 的光标与自动滚屏会被 React
 * 提交拖慢。这里只忽略它们，其余会话字段仍按引用浅比较。
 */
const PIANO_ROLL_VOLATILE_SESSION_KEYS = new Set<keyof RootState["session"]>([
    "playheadSec",
    "playheadSampledAtMs",
    "runtime",
]);

function pianoRollSessionEqual(left: RootState["session"], right: RootState["session"]): boolean {
    if (left === right) return true;
    for (const key of Object.keys(left) as Array<keyof RootState["session"]>) {
        if (PIANO_ROLL_VOLATILE_SESSION_KEYS.has(key)) continue;
        if (left[key] !== right[key]) return false;
    }
    return true;
}

/**
 * 装配参数编辑器面板的模型。
 *
 * @param dockFormId 本窗体在停靠布局里的 id（判断是否与时间轴上下堆叠）。
 * @returns 面板视图层需要的全部状态、派生值与回调。
 */
export function usePianoRollPanelModel(dockFormId: string | undefined) {
    const dispatch = useAppDispatch();
    // 事件监听器内同步读取 session（如 selectClipParamRange 的 Clip 查找），
    // 避免闭包快照滞后。
    const store = useAppStore();
    const rafRef = useRef<number | null>(null);
    const visualPlayheadSecRef = useRef(store.getState().session.playheadSec);
    // 视觉插值播放头读取器（与绘制同源）：缩放锚点必须使用它，不能用 33Hz
    // 轮询的 store 滞后值——播放中缩放以滞后值锚定会让播放头跳变 δ·Δpx。
    const getVisualPlayheadSec = useCallback(() => visualPlayheadSecRef.current, []);
    const rulerPlayheadLineRef = useRef<HTMLDivElement | null>(null);
    /** 标尺上的捕获区间带**容器**（几何由内核按段写，见宿主里的 captureBands）。 */
    const captureBandsRef = useRef<HTMLDivElement | null>(null);
    const rulerPlayheadHeadRef = useRef<HTMLDivElement | null>(null);
    /**
     * 内核宿主的稳定引用。
     *
     * 【为什么声明在这里】`invalidate`（紧随其后）要把标脏转交宿主，
     * 因此必须在它之前声明。放在下方原来的位置会让 `invalidate` 的闭包引用一个
     * 尚未初始化的 `const`（TDZ）——首次调用即抛 ReferenceError。
     */
    const hostRef = useRef<PianoRollKernelHost | null>(null);
    const drawRef = useRef<() => void>(() => {});
    /**
     * 请求重绘（所有会改变画面的数据/状态变更都经此入口）。
     *
     * 【为什么要转交宿主】曲线层在 GL 上，画面由**两个**渲染循环驱动：
     * 面板自己的 rAF（Canvas2D 细节层）与内核宿主的 rAF（GL 层）。只调度前者会让
     * GL 层永远停留在旧内容上——典型症状是"曲线数据到了但屏幕上不出现，直到滚动
     * 一下才显示"。因此标脏一律交给宿主：宿主的帧提交里会回调 `onFrame`
     * → `applyScrollLayers` → `drawRef.current()`，Canvas2D 与 GL 同帧一起刷新。
     *
     * 特殊说明：`drawRef.current()` 自身不调用 `invalidate()`，所以这条链不会自激。
     * 宿主尚未创建（挂载前 / 卸载后）时退回面板自己的 rAF，避免丢失首次绘制。
     */
    const invalidate = useCallback(() => {
        const host = hostRef.current;
        if (host != null) {
            host.invalidate();
            return;
        }
        if (rafRef.current != null) return;
        rafRef.current = requestAnimationFrame(() => {
            rafRef.current = null;
            drawRef.current();
        });
    }, []);

    /**
     * DPR 变化必须重绘主画布。
     *
     * `rasterize` 会按新 dpr 重设 backing store 与 CSS 尺寸，但**没有任何东西**
     * 会在纯 dpr 变化时触发重绘：`ResizeObserver` 观察的是 CSS 布局盒（dpr 变了
     * 尺寸也没变），而本组件的 `viewSize` 也保持不变。缺这条订阅时，换显示器或改
     * 系统缩放后主画布会一直用旧 dpr 的光栅化结果（发虚），直到某次滚动才恢复。
     * 标脏转交宿主，与 GL 层同帧刷新（见 `invalidate` 的说明）。
     */
    React.useEffect(() => subscribeDevicePixelRatio(() => invalidate()), [invalidate]);

    /**
     * 标尺播放头元素（竖线 / 倒三角）的挂载回调。
     *
     * 【为什么挂载时要请求一帧】位置只由内核在帧提交里写，而元素可能在**播放头
     * 静止**时被重建（面板重挂载 / 视图切换）：新节点没有任何 `left`，等价于
     * `left: auto` ⇒ 落在静态位置（左缘 = 工程起始处）。此时若没有任何东西弄脏帧
     * （暂停且不滚动），内核不会提交，元素就**停在工程起始处不动**。元素一出现就
     * 请求一帧，写入器随即按当前视口定位它（去重键含元素身份，见
     * `createPlayheadElementWriter`）。
     */
    const attachRulerPlayheadLine = useCallback((element: HTMLDivElement | null) => {
        rulerPlayheadLineRef.current = element;
        if (element !== null) hostRef.current?.invalidate();
    }, []);
    const attachRulerPlayheadHead = useCallback((element: HTMLDivElement | null) => {
        rulerPlayheadHeadRef.current = element;
        if (element !== null) hostRef.current?.invalidate();
    }, []);
    const { t, tf, tVars } = useI18n();
    const s = useAppSelector((state: RootState) => state.session, pianoRollSessionEqual);

    // 工程会话切换：强制视口总线按当前工程内容重绘一次（跨工程投影保留契约
    // 的强制点，见 utils/timelineViewportBus.invalidate 的说明）。
    const projectSessionPath = s.project.path;
    React.useEffect(() => {
        pianoRollViewportBus.invalidate();
    }, [projectSessionPath]);
    const effectiveProjectScale = useMemo<ScaleLike>(
        () =>
            s.project.useCustomScale && s.project.customScale
                ? s.project.customScale.notes
                : s.project.baseScale,
        [s.project.baseScale, s.project.customScale, s.project.useCustomScale],
    );
    /**
     * 某秒位置生效的“工程音阶”（受 Tempo Map 音阶变化点影响）。
     * 无 Tempo Map 音阶覆盖时即为工程音阶。
     */
    const projectScaleAtSec = useCallback(
        (sec: number): ScaleLike | undefined =>
            effectiveScaleAtSec(s.tempoMap, sec, effectiveProjectScale),
        [s.tempoMap, effectiveProjectScale],
    );
    /**
     * 将音阶 token 解析为 ScaleLike。
     * `__project__` 在提供 `atSec` 时按该时刻的 Tempo Map 生效音阶解析，
     * 否则使用工程音阶（用于全局场景）。
     */
    const resolveScaleFromToken = useCallback(
        (scaleToken: string, atSec?: number): ScaleLike => {
            if (scaleToken === "__project__") {
                return atSec != null ? (projectScaleAtSec(atSec) ?? "C") : effectiveProjectScale;
            }

            const customScaleId = parseCustomScaleToken(scaleToken);
            if (customScaleId) {
                const preset = s.customScalePresets.find((item) => item.id === customScaleId);
                if (preset) {
                    return preset.notes;
                }
            }

            return isScaleKey(scaleToken) ? scaleToken : "C";
        },
        [effectiveProjectScale, projectScaleAtSec, s.customScalePresets],
    );
    const editParam = s.editParam as ParamName;
    /**
     * `editParam` 的 ref 镜像。
     *
     * 【为什么需要】宿主在**挂载时创建一次**（长生命周期），其回调（`onFrame` /
     * `onScrollTopFrame`）里无法直接读到最新的 `editParam`——闭包捕获的是挂载时的
     * 值，切换参数后回调仍按旧参数换算视口，会出现"切了参数但竖向位置按上一个参数
     * 映射"的错位。ref 每次渲染刷新，回调现读即最新。
     */
    const editParamRef = useRef<ParamName>(editParam);
    editParamRef.current = editParam;
    // pitchSnapOpen 已在顶部工具栏 JSX 内声明和使用，无需重复声明
    // pianoRoll.copy/cut/paste 的复制/剪切/粘贴已由全局路由统一派发到
    // handleEditOp，本地不再需要键位匹配（见 useKeybindings/focusRouting）。
    const prVerticalZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.pianoRollVerticalZoom"),
    );
    const horizontalZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.horizontalZoom"),
    );
    const scrollHorizontalKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollHorizontal"),
    );
    const scrollVerticalKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollVertical"),
    );
    const scrollbarZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollbarZoom"),
    );
    const pianoKeysVerticalScrollKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.pianoKeysVerticalScroll"),
    );
    const pianoKeysVerticalZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.pianoKeysVerticalZoom"),
    );
    const paramMorphKb = useAppSelector((state) => selectKeybinding(state, "modifier.paramMorph"));
    // 多选区修饰键（默认 ⌘/Ctrl）：按住拖动追加一段选区，按住点击取消该段
    const paramMultiSelectKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramMultiSelect"),
    );
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );
    // 边缘平滑度滑块的滚轮步进：React 17+ 的根容器 wheel 监听是 passive，
    // JSX onWheel 里的 preventDefault 无效（伴随干预警告），必须走原生
    // 非 passive 监听（与主画布滚轮路径同模式）。
    //
    // 【修饰键必须读 `e` 本身】这里拿到的是**原生** WheelEvent（见
    // `useNonPassiveWheel` 的说明），`ctrlKey/shiftKey/altKey` 就在它自己身上。
    // 曾经写的是 `e.nativeEvent`（那是 React 合成事件的取法），原生事件没有这个
    // 属性，取到 undefined 之后 `isModifierActive` 读 `ctrlKey` 直接抛异常 ——
    // 处理器在第一行就炸，后面的 dispatch 从未执行，表现为**滚轮完全无效**。
    const attachEdgeSmoothnessWheel = useNonPassiveWheel<HTMLInputElement>((e) => {
        e.preventDefault();
        const fine = isModifierActive(paramFineAdjustKb, e);
        const step = fine ? 1 : 5;
        const dir = e.deltaY < 0 ? 1 : -1;
        const next = clamp(Math.round(s.edgeSmoothnessPercent) + dir * step, 0, 100);
        dispatch(setEdgeSmoothnessPercent(next));
        void dispatch(persistUiSettings());
    });
    // 参数选区边缘拉伸的修饰键。与时间轴 clip 边缘的 `modifier.clipStretch`
    // **分离**（两个表面各自独立改绑），只用于选择工具下的选区边缘拉伸。
    const stretchKb = useAppSelector((state) => selectKeybinding(state, "modifier.paramStretch"));
    const vibratoAmplitudeAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.vibratoAmplitudeAdjust"),
    );
    const vibratoFrequencyAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.vibratoFrequencyAdjust"),
    );
    const vibratoDragAmplitudeIncreaseKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.vibratoDragAmplitudeIncrease"),
    );
    const vibratoDragAmplitudeDecreaseKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.vibratoDragAmplitudeDecrease"),
    );
    const vibratoDragFrequencyIncreaseKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.vibratoDragFrequencyIncrease"),
    );
    const vibratoDragFrequencyDecreaseKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.vibratoDragFrequencyDecrease"),
    );
    // 拖拽颤音时切换预设（默认 `,` / `.`）。只在拖拽期间生效 —— 这两个动作的
    // `scopedContext` 是 `pianoRollVibratoDrag`，全局派发器不会执行它们。
    const vibratoPresetPrevKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.vibratoPresetPrev"),
    );
    const vibratoPresetNextKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.vibratoPresetNext"),
    );
    const vibratoPresetUserList = useAppSelector((state) => state.session.vibratoPresets);
    const activeVibratoPresetId = useAppSelector((state) => state.session.activeVibratoPresetId);
    /** 「摆放方式」：添加颤音的参数（存在设置里，与预设无关）。 */
    const vibratoBaseline = useAppSelector((state) => state.session.vibratoBaseline);
    const disabledVibratoPresetIds = useAppSelector(
        (state) => state.session.disabledVibratoPresetIds,
    );
    const builtinVibratoPresetOrder = useAppSelector(
        (state) => state.session.builtinVibratoPresetOrder,
    );
    const resolvedVibratoPresets = useMemo(
        () => resolveVibratoPresets(vibratoPresetUserList, builtinVibratoPresetOrder).all,
        [vibratoPresetUserList, builtinVibratoPresetOrder],
    );
    /**
     * 可选（未被停用）的预设：工具栏下拉只列这些，拖拽中的循环切换也只在这些里绕。
     *
     * 停用名单之外的用途仍走全量 `resolvedVibratoPresets` —— HUD 要按 id 找到
     * 正在用的预设（它可能已被停用），「重置到直线」也要能找到直线预设。
     */
    const enabledVibratoPresetList = useMemo(
        () => enabledVibratoPresets(resolvedVibratoPresets, disabledVibratoPresetIds),
        [resolvedVibratoPresets, disabledVibratoPresetIds],
    );
    const activeVibratoPreset = useMemo(
        () => resolveActiveVibratoPreset(resolvedVibratoPresets, activeVibratoPresetId),
        [resolvedVibratoPresets, activeVibratoPresetId],
    );
    /**
     * 直线工具的固定起手预设。
     *
     * 从全量列表里找，**找不到就用出厂那份兜底**（`systemVibratoPreset` 自带构造）：
     * 直线工具不依赖用户是否停用了直线预设 —— 停用只影响"怎么挑预设"，不该让
     * 一个工具失去它唯一的工作方式。
     */
    const straightVibratoPreset = useMemo(
        () =>
            findVibratoPreset(resolvedVibratoPresets, STRAIGHT_VIBRATO_PRESET_ID) ??
            systemVibratoPreset("straight"),
        [resolvedVibratoPresets],
    );
    /**
     * 拖拽 HUD 的内容。`null` = 没在拖。
     *
     * 用一个本地 state 而不是 ref：HUD 要跟着切换预设 / 滚轮调参实时刷新。
     */
    const [vibratoDragHud, setVibratoDragHudState] = useState<{
        presetId: string;
        depthCents: number;
        rateHz: number;
        adjusted: boolean;
        clientX: number;
        clientY: number;
        /** 画布容器 rect 快照（随 HUD 同帧采集，与 paramValuePreview 同一口径）。 */
        rectLeft: number;
        rectTop: number;
    } | null>(null);
    /** 落地时顺手快照画布 rect：渲染期（EditorArea）不再量 getBoundingClientRect。 */
    const setVibratoDragHud = useCallback(
        (
            hud: {
                presetId: string;
                depthCents: number;
                rateHz: number;
                adjusted: boolean;
                clientX: number;
                clientY: number;
            } | null,
        ) => {
            if (!hud) {
                setVibratoDragHudState(null);
                return;
            }
            const rect = canvasRef.current?.getBoundingClientRect();
            setVibratoDragHudState({
                ...hud,
                rectLeft: rect?.left ?? 0,
                rectTop: rect?.top ?? 0,
            });
        },
        [],
    );
    /**
     * 颤音窗口是否打开。
     *
     * 参数编辑器自己持有这个状态（不走 `hifi:openEditDialog`）：那个通道是给
     * 「一次性参数对话框」用的，而颤音窗口是常驻的预设库，菜单栏的「管理预设…」
     * 与这里的右键 / 工具栏入口共用同一个对话框组件。
     */
    const [vibratoDialogOpen, setVibratoDialogOpen] = useState(false);
    /** 打开时要编辑哪一条；`null` = 按当前活动预设播种。 */
    const [vibratoInitialPresetId, setVibratoInitialPresetId] = useState<string | null>(null);
    /**
     * 这次打开是哪一面：`"apply"` = 「添加颤音」（带套用预览 / 提取 / 应用），
     * `"manage"` = 「管理预设…」（纯预设库）。
     *
     * 【为什么必须有它】窗口有两副面孔，而"从哪个入口进来的"只有宿主知道。窗口自己
     * 不该去猜"我是被打开发动机是什么"—— 于是宿主把结论（有没有选区宿主）直接告诉它：
     * `apply` 才传 `applyTarget`，`manage` 不传，窗口据此收起全部套用专属能力。
     */
    const [vibratoDialogMode, setVibratoDialogMode] = useState<"apply" | "manage">("manage");
    /**
     * 窗口的"会话号"：每次打开自增并作为组件的 `key`。
     *
     * 窗口借此**重新挂载** —— 草稿回到播种值、预览页签与两把纵轴复位、选区帧值重取。
     * 用 key 而不是 effect 内重置，是为了不在 effect 里同步 setState（那会级联渲染，
     * 也被 lint 禁止）。这一条替代了合并前"应用弹窗的会话号 + 管理器的播种闸"两套机制。
     */
    const [vibratoDialogSession, setVibratoDialogSession] = useState(0);

    /**
     * 打开颤音窗口。
     *
     * @param mode 哪一面（决定要不要给选区宿主，见 `vibratoDialogMode`）。
     * @param initialPresetId 打开时编辑哪一条；省略 = 当前活动预设。
     */
    const openVibratoDialog = useCallback(
        (mode: "apply" | "manage" = "manage", initialPresetId?: string) => {
            setVibratoDialogMode(mode);
            setVibratoInitialPresetId(initialPresetId ?? null);
            setVibratoDialogSession((session) => session + 1);
            setVibratoDialogOpen(true);
        },
        [],
    );

    /**
     * 工具菜单里「颤音预设」二级面板中**预设列表**的最大高度。
     *
     * 预设列表从独立的下拉搬进子菜单后，高度仍需按锚点在**本面板内**的位置算
     * （见 `resolveMenuMaxHeight`）：面板可停靠在窗口任意高度，按视口取上限会让
     * 列表伸出面板之外。锚点用绘制类型工具按钮（预设入口就在它的菜单里）。
     */
    const [presetFlyoutMaxHeight, setPresetFlyoutMaxHeight] = useState(320);
    // 拖动方向循环切换键：拖拽进行中按下可即时切换本次拖拽方向（触控板替代右键）。
    const cycleDragDirectionKb = useAppSelector((state) =>
        selectKeybindings(state, "pianoRoll.cycleDragDirection"),
    );
    // 工具菜单里三项各自的快捷键：把绑定显示在菜单上，用户不必去设置里查。
    const drawToolKb = useAppSelector((state) => selectKeybindings(state, "mode.drawTool"));
    const lineToolKb = useAppSelector((state) => selectKeybindings(state, "mode.lineTool"));
    const vibratoToolKb = useAppSelector((state) => selectKeybindings(state, "mode.vibratoTool"));
    const mergedKeybindings = useAppSelector(selectMergedKeybindings);
    // 是否按住切换吸附的修饰键（临时切换吸附时用于高亮显示）
    const [snapToggleHeld, setSnapToggleHeld] = useState(false);
    // 仅在参数编辑实际操作期间（选择拖拽/绘制）参与临时吸附视觉切换
    const [snapGestureActive, setSnapGestureActive] = useState(false);
    const [hoveredReferenceRootTrackId, setHoveredReferenceRootTrackId] = useState<string | null>(
        null,
    );

    useEffect(() => {
        // 修饰键手势只有一个绑定，取主绑定（见 types.ts 的 KeybindingMap）。
        const kb = firstBinding(mergedKeybindings["modifier.clipNoSnap"]);
        if (!kb || isNoneBinding(kb)) return;
        const onKey = (e: KeyboardEvent) => {
            const active = isModifierActive(kb, e);
            setSnapToggleHeld(active);
        };
        window.addEventListener("keydown", onKey as EventListener);
        window.addEventListener("keyup", onKey as EventListener);
        // also track blur to clear state
        const onBlur = () => setSnapToggleHeld(false);
        window.addEventListener("blur", onBlur);
        return () => {
            window.removeEventListener("keydown", onKey as EventListener);
            window.removeEventListener("keyup", onKey as EventListener);
            window.removeEventListener("blur", onBlur);
        };
    }, [mergedKeybindings]);
    const { mode: themeMode, fontFamily } = useAppTheme();
    const waveformColors = useMemo(() => getWaveformColors(themeMode, "piano-roll"), [themeMode]);

    const effectivePitchSnapVisual =
        snapGestureActive && snapToggleHeld ? !s.pitchSnapEnabled : s.pitchSnapEnabled;

    // 右键编辑菜单状态
    const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number } | null>(null);
    const [drawToolMenuOpen, setDrawToolMenuOpen] = useState(false);
    /**
     * 工具栏下拉的**锚点**与**面板** ref 必须分开。
     *
     * 【为什么】面板经 `AppAnchoredMenu` 挂到 `document.body` 下，不再是锚点的
     * DOM 后代 —— 用锚点的 ref 做 `contains` 判定会永远为假，于是"点菜单里的项"
     * 会被当成"点了外面"，菜单在选中前就被关掉。
     */
    const drawToolMenuAnchorRef = useRef<HTMLDivElement | null>(null);
    const drawToolMenuRef = useRef<HTMLDivElement | null>(null);
    const [pitchSnapMenuOpen, setPitchSnapMenuOpen] = useState(false);
    const pitchSnapMenuAnchorRef = useRef<HTMLDivElement | null>(null);
    const pitchSnapMenuRef = useRef<HTMLDivElement | null>(null);
    const [paramValuePreview, setParamValuePreview] = useState<{
        clientX: number;
        clientY: number;
        value: number;
        displayText?: string;
        /** 画布容器 rect 快照（随读数同帧采集，见下方合帧注释）。 */
        rectLeft: number;
        rectTop: number;
    } | null>(null);
    // 浮窗更新合帧（见 onParamValuePreviewChange 处的说明）：pointermove 在高刷
    // 设备上可达数百 Hz，逐事件 setState 会让整个面板以事件频率重渲。
    const paramValuePreviewRafRef = useRef<number | null>(null);
    const paramValuePreviewPendingRef = useRef<{
        payload: { clientX: number; clientY: number; value: number; displayText?: string } | null;
    } | null>(null);
    useEffect(() => {
        return () => {
            if (paramValuePreviewRafRef.current != null) {
                cancelAnimationFrame(paramValuePreviewRafRef.current);
                paramValuePreviewRafRef.current = null;
            }
            paramValuePreviewPendingRef.current = null;
        };
    }, []);
    /**
     * 纵轴标尺的悬浮读数（`弹出展示参数` 在左轴上的形态）。
     *
     * 【为什么与 `paramValuePreview` 分开】那个浮窗挂**曲线画布**的坐标系
     * （`canvasRef` 的 rect），本浮窗挂轴列（`axisWrapRef` 的 rect）。两者坐标系
     * 不同，混用会让浮窗横向偏出 56px 宽的轴列。
     */
    const [axisValuePreview, setAxisValuePreview] = useState<{
        clientX: number;
        clientY: number;
        text: string;
        /** 轴列容器 rect 快照（随读数同帧采集，见 usePianoRollAxisInteractions）。 */
        rectLeft: number;
        rectTop: number;
    } | null>(null);

    /**
     * 当前参数的纵轴展示单位（音量 / 动态支持倍率 ↔ dB 切换）。
     *
     * 一个值同时喂给三处：GL 刻度标签（经 `buildGridSpec`）、参数线悬浮浮窗
     * （`formatParamValuePreview`）、纵轴标尺悬浮浮窗。三者必须同源，否则同一位
     * 置会出现"刻度写 −6、浮窗写 0.501"的自相矛盾读数。
     */
    const editParamAxisUnit = resolveParamAxisUnit(s.paramAxisUnits, editParam);
    /** 当前参数的左轴是否可切换展示单位（决定光标与角标）。 */
    const axisUnitToggleAvailable = supportsParamAxisUnit(editParam);

    const formatParamValuePreview = useCallback(
        (value: number): string => {
            if (!Number.isFinite(value)) return "";
            if (editParam === "pitch") {
                const rounded = Math.round(value);
                const pitchClass = ((rounded % 12) + 12) % 12;
                const octave = Math.floor(rounded / 12) - 1;
                const noteName = `${NOTE_NAMES_SHARP[pitchClass]}${octave}`;
                const cents = Math.round((value - rounded) * 100);
                const signedCents = cents >= 0 ? `+${cents}` : `${cents}`;
                return `${noteName}${signedCents}`;
            }
            if (isChildPitchOffsetDegreesParam(editParam)) {
                const display = childPitchOffsetValueToDisplay(editParam, value);
                if (Math.abs(display) >= 100) return display.toFixed(1);
                if (Math.abs(display) >= 10) return display.toFixed(2);
                return display.toFixed(3);
            }
            // 音量 / 动态切到 dB 读法：1× = 0 dB（见 paramAxisUnits）。带 `dB` 后缀
            // 与倍率读数区分——浮窗有空间写清楚单位，用户才不会把 −6 误读成倍率。
            if (editParamAxisUnit === "db" && supportsParamAxisUnit(editParam)) {
                return formatDbReadout(value);
            }
            if (Math.abs(value) >= 100) return value.toFixed(1);
            if (Math.abs(value) >= 10) return value.toFixed(2);
            return value.toFixed(3);
        },
        [editParam, editParamAxisUnit],
    );

    const currentDrawTool = s.drawToolMode;
    const drawToolButtonTitle =
        currentDrawTool === "vibrato"
            ? // 颤音工具下把当前预设一并报出来：预设按钮已经并进右键菜单，
              // 工具按钮的 tooltip 是"接下来会画出什么"的最后一块可扫读信息。
              `${tf("vibrato_draw_tool")}: ${vibratoPresetLabel(activeVibratoPreset, tf)}`
            : currentDrawTool === "line"
              ? tf("line_draw_tool")
              : tf("draw_tool");
    const activeDragDirection =
        s.toolMode === "select" || s.toolMode === "block"
            ? s.selectDragDirection
            : currentDrawTool === "draw"
              ? s.drawDragDirection
              : s.lineVibratoDragDirection;
    // 直线与颤音共用同一份拖动方向（`lineVibratoDragDirection`）—— 它们是同一个
    // "起点 → 终点"手势，拆成两份只会让"在直线下调过、切到颤音又不算"成为新的困惑。
    const activeDragDirectionTool =
        s.toolMode === "select" || s.toolMode === "block"
            ? ("select" as const)
            : currentDrawTool === "draw"
              ? ("draw" as const)
              : ("vibrato" as const);

    useEffect(() => {
        if (!drawToolMenuOpen && !pitchSnapMenuOpen) return;
        const onPointerDown = (e: PointerEvent) => {
            const target = e.target as Node | null;
            if (drawToolMenuRef.current?.contains(target)) return;
            if (pitchSnapMenuRef.current?.contains(target)) return;
            setDrawToolMenuOpen(false);
            setPitchSnapMenuOpen(false);
        };
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                setDrawToolMenuOpen(false);
                setPitchSnapMenuOpen(false);
            }
        };
        window.addEventListener("pointerdown", onPointerDown, true);
        window.addEventListener("keydown", onKeyDown, true);
        return () => {
            window.removeEventListener("pointerdown", onPointerDown, true);
            window.removeEventListener("keydown", onKeyDown, true);
        };
    }, [drawToolMenuOpen, pitchSnapMenuOpen]);

    /** 量一次预设列表的最大高度（锚点 / 面板缺失时返回 null）。 */
    const measurePresetFlyoutMaxHeight = useCallback((): number | null => {
        const anchor = drawToolMenuAnchorRef.current;
        const container = paramEditorRef.current;
        if (!anchor || !container) return null;
        const anchorRect = anchor.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();
        return resolveMenuMaxHeight({
            anchorBottom: anchorRect.bottom,
            containerTop: containerRect.top,
            containerBottom: containerRect.bottom,
        });
    }, []);

    // 工具菜单开着时窗口缩放、或面板被拖动分隔条改变大小，都要重算预设列表的
    // 可用高度。setState 放在回调里（而不是 effect 体内同步调用），避免级联渲染。
    useEffect(() => {
        if (!drawToolMenuOpen) return;
        const update = () => {
            const maxHeight = measurePresetFlyoutMaxHeight();
            if (maxHeight != null) setPresetFlyoutMaxHeight(maxHeight);
        };
        update();
        window.addEventListener("resize", update);
        const observer = new ResizeObserver(update);
        if (paramEditorRef.current) observer.observe(paramEditorRef.current);
        return () => {
            window.removeEventListener("resize", update);
            observer.disconnect();
        };
    }, [drawToolMenuOpen, measurePresetFlyoutMaxHeight]);

    // ── MIDI 拖放到参数编辑器（文件浏览器拖拽 + Tauri 系统文件拖放）────────
    // 与“导入 MIDI”按钮同属参数编辑器场景：导入目标默认 Pitch Param，并持久化
    // 到 midiImportTargetParamEditor（与按钮共用同一设置项）。
    const paramEditorRef = useRef<HTMLDivElement | null>(null);

    const effectiveSelectedTrackId = useMemo(() => {
        if (s.selectedTrackId) return s.selectedTrackId;
        const clipId = s.selectedClipId;
        if (!clipId) return null;
        const clip = s.clips.find((c) => c.id === clipId);
        return clip?.trackId ?? null;
    }, [s.selectedTrackId, s.selectedClipId, s.clips]);

    const selectedTrack = useMemo(() => {
        if (!effectiveSelectedTrackId) return null;
        return s.tracks.find((track) => track.id === effectiveSelectedTrackId) ?? null;
    }, [effectiveSelectedTrackId, s.tracks]);

    const selectedIsChildTrack = Boolean(selectedTrack?.parentId);

    const childPitchOffsetCentsParam = useMemo(() => {
        if (!effectiveSelectedTrackId || !selectedIsChildTrack) return null;
        return buildChildPitchOffsetCentsParam(effectiveSelectedTrackId);
    }, [effectiveSelectedTrackId, selectedIsChildTrack]);

    const childPitchOffsetDegreesParam = useMemo(() => {
        if (!effectiveSelectedTrackId || !selectedIsChildTrack) return null;
        return buildChildPitchOffsetDegreesParam(effectiveSelectedTrackId);
    }, [effectiveSelectedTrackId, selectedIsChildTrack]);
    // 视口（滚动 / 缩放 / 与时间轴同步）整块在 ``usePianoRollViewport`` 里。调用点必须留
    // 在原位：hook 顺序、以及 `useLayoutEffect` 的提交时机（"缩放先提交 state，再由
    // layout effect 落地"）都是行为的一部分。
    //
    // `commitViewportNow` 在下方以**函数声明**定义（hoisted），所以这里能提前把它交出去；
    // 它只在事件期被调用，不存在渲染期读到未初始化绑定的问题。
    const {
        scrollLeft,
        setScrollLeft,
        pxPerSec,
        pxPerBeat,
        dynamicProjectSec,
        scrollLeftRef,
        pxPerSecRef,
        zoomRafRef,
        paramEditorSyncTimelineRef,
        timelineOffsetRef,
        timelineOffsetPx,
        timelineSyncApplyingRef,
        horizontalZoomChainRef,
        lastScrollLeftRef,
        scrollerRef,
        viewSizeRef,
        getScrollLeftLive,
        contentWidthAt,
        handleHorizontalZoom,
    } = usePianoRollViewport({
        s,
        hostRef,
        dockFormId,
        getVisualPlayheadSec,
        commitViewportNow,
    });
    // 副参数独立显示开关，默认全部关闭
    const [secondaryParamVisible, setSecondaryParamVisible] = useState<
        Partial<Record<ParamName, boolean>>
    >({});

    const toggleSecondaryParam = useCallback((param: ParamName) => {
        setSecondaryParamVisible((prev) => toggleSecondaryParamVisibility(prev, param));
    }, []);

    const pitchViewRef = useRef<ValueViewport>({
        center: 72,
        span: 24,
    });
    const setPitchView = useCallback(
        (next: ValueViewport) => {
            pitchViewRef.current = next;
            syncVerticalScrollbarForViewport("pitch", next);
            // 值域视口是**竖轴的手势提交点**：滚动条 thumb 已同步写下 DOM，曲线 /
            // 音高键 / 数值轴（GL）必须在**同一任务**里提交，否则竖轴平移/缩放时
            // 这些线比滚动条慢一帧到几帧（见 `paintNow`）。
            paintNow();
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [invalidate],
    );

    const paramViewsRef = useRef<Record<string, ValueViewport>>({});
    const setParamViewport = useCallback(
        (param: string, next: ValueViewport) => {
            paramViewsRef.current = { ...paramViewsRef.current, [param]: next };
            syncVerticalScrollbarForViewport(param as ParamName, next);
            // 同 `setPitchView`：竖轴的手势提交点，DOM 与 GL 必须同任务。
            paintNow();
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [invalidate],
    );

    const rootTrackId = useMemo(() => {
        return resolveRootTrackId(s.tracks, effectiveSelectedTrackId);
    }, [effectiveSelectedTrackId, s.tracks]);

    const rootTrack = useMemo(() => {
        if (!rootTrackId) return null;
        return s.tracks.find((tr) => tr.id === rootTrackId) ?? null;
    }, [s.tracks, rootTrackId]);

    // 算法下拉的选项与取值：唯一来源（见 pitchAlgoOptions.ts）。不可用时
    // vslib 从列表消失；当前值已是 vslib 时保留并标注"不可用"，避免下拉框
    // 回退显示成 nsf-hifigan 而谎报轨道真实算法。
    const pitchAlgoOptions = useMemo(
        () =>
            buildPitchAlgoOptions({
                noneLabel: t("common_none"),
                vslibAvailable: s.vslibAvailable,
                formatUnavailable: (label) => tVars("algo_unavailable_label", { name: label }),
                currentValue: rootTrack?.pitchAnalysisAlgo,
            }),
        [t, tVars, s.vslibAvailable, rootTrack?.pitchAnalysisAlgo],
    );

    const childFormantOffsetParam = useMemo(() => {
        if (!effectiveSelectedTrackId || !selectedIsChildTrack) return null;
        // 归一后再判：未识别的算法在后端按默认算法（nsf-hifigan）渲染，而
        // nsf-hifigan 链路是支持子轨道共振峰偏移的（见
        // `does_clip_need_processor_render` 的 `has_child_formant_offset`）——
        // 用原始字符串判断会把这个控件藏掉，功能在后端生效、界面却给不了入口。
        const algo = resolveEffectivePitchAlgo(rootTrack?.pitchAnalysisAlgo);
        if (algo !== "nsf_hifigan_onnx" && algo !== "vslib") return null;
        return buildChildFormantOffsetCentsParam(effectiveSelectedTrackId);
    }, [effectiveSelectedTrackId, selectedIsChildTrack, rootTrack?.pitchAnalysisAlgo]);

    const pitchGroupActive =
        editParam === "pitch" ||
        editParam === childPitchOffsetCentsParam ||
        editParam === childPitchOffsetDegreesParam;
    // 工具栏上的音高组按钮使用简写（非中文语系为 PIT），完整名称放入 ToolTip。
    const pitchGroupLabel =
        editParam === childPitchOffsetCentsParam
            ? t("child_pitch_mode_cents")
            : editParam === childPitchOffsetDegreesParam
              ? t("child_pitch_mode_degrees")
              : t("param_btn_pitch");
    const pitchGroupTooltip =
        editParam === childPitchOffsetCentsParam
            ? t("child_pitch_offset_cents_label")
            : editParam === childPitchOffsetDegreesParam
              ? t("child_pitch_offset_degrees_label")
              : t("common_pitch");

    // 声码器参数描述符 + 当前参数元数据 + 三个标签映射。
    const {
        processorParams,
        processorParamsRef,
        processorStaticParams,
        processorStaticValues,
        orderedProcessorParams,
        currentParamRange,
        currentParamDefaultValue,
        currentParamQuantizeUnit,
        handleStaticParamChange,
        getProcessorParamLabel,
        getProcessorParamShortLabel,
        getStaticOptionLabel,
    } = useProcessorParams({
        rootTrack,
        rootTrackId,
        editParam,
        paramViewsRef,
        invalidate,
        t,
    });

    // 当 processorParams 变化时，若 editParam 不在可用集合内，自动回退到 pitch
    useEffect(() => {
        const available = new Set([
            "pitch",
            ...processorParams.map((p) => p.id),
            ...(childPitchOffsetCentsParam ? [childPitchOffsetCentsParam] : []),
            ...(childPitchOffsetDegreesParam ? [childPitchOffsetDegreesParam] : []),
            ...(childFormantOffsetParam ? [childFormantOffsetParam] : []),
        ]);
        if (isChildPitchOffsetParam(editParam)) {
            if (!selectedIsChildTrack || !effectiveSelectedTrackId) {
                dispatch(setEditParam("pitch"));
                return;
            }
            if (isChildPitchOffsetCentsParam(editParam)) {
                const expected = buildChildPitchOffsetCentsParam(effectiveSelectedTrackId);
                if (editParam !== expected) {
                    dispatch(setEditParam(expected));
                    return;
                }
            }
            if (isChildPitchOffsetDegreesParam(editParam)) {
                const expected = buildChildPitchOffsetDegreesParam(effectiveSelectedTrackId);
                if (editParam !== expected) {
                    dispatch(setEditParam(expected));
                    return;
                }
            }
            if (isChildFormantOffsetCentsParam(editParam)) {
                const expected = buildChildFormantOffsetCentsParam(effectiveSelectedTrackId);
                if (editParam !== expected) {
                    dispatch(setEditParam(expected));
                    return;
                }
            }
        }

        if (!available.has(editParam)) {
            dispatch(setEditParam("pitch"));
        }
    }, [
        processorParams,
        editParam,
        dispatch,
        childPitchOffsetCentsParam,
        childPitchOffsetDegreesParam,
        childFormantOffsetParam,
        effectiveSelectedTrackId,
        selectedIsChildTrack,
    ]);

    // ── 气声分离开关状态 ────────────────────────────────────────────────────
    //
    // 开关值为 1 表示"做谐波/噪声分离"。它是气声音量与张力共同的前提：
    // 关闭时两者置灰、不可编辑，且不参与合成（后端在 `ClipProcessContext`
    // 构造点剥离这两条曲线，因此也不会走 HNSEP）。
    // 阈值语义见 `separationGate.ts`（与后端 `extra_param_enabled` 同口径）。
    //
    // `separationDesc` 同时决定开关**是否存在**：该描述符只挂在 nsf-hifigan 的
    // 描述符表里（见 `renderer/chain.rs` 的 `HIFIGAN_PARAM_DESCRIPTORS`），
    // 因此 world / vslib 下开关不渲染 —— 那两种算法本就没有气声 / 张力参数。
    const separationDesc = useMemo(
        () => processorStaticParams.find((sp) => sp.id === SEPARATION_PARAM_ID) ?? null,
        [processorStaticParams],
    );
    const separationEnabled = useMemo(() => {
        const defaultValue =
            separationDesc && separationDesc.kind.type === "static_enum"
                ? separationDesc.kind.default_value
                : 0;
        return isSeparationEnabled(processorStaticValues[SEPARATION_PARAM_ID], defaultValue);
    }, [separationDesc, processorStaticValues]);

    // 分离开关在工具栏上的落点：被门禁组（气声 + 张力）的**组首**。
    //
    // 【为什么是组首而不是"算法下拉之前"】开关只随 nsf-hifigan 出现；若把它放在
    // 工具栏最右端（算法下拉左侧），切换算法时它整块消失 / 出现，会把音量、声像、
    // 算法选项一起推来推去。放在参数组内部后，它右侧紧邻的就是固定的
    // 「音量 → 动态 → 声像」，消失时右侧纹丝不动。
    //
    // 位置由 `firstGatedParamId` 从门禁列表推导（见该函数说明），不写死具体 id。
    const separationSwitchAnchorParamId = useMemo(
        () => (separationDesc ? firstGatedParamId(orderedProcessorParams.map((p) => p.id)) : null),
        [separationDesc, orderedProcessorParams],
    );

    // ── 工具栏横向不足时的分级隐藏 ──────────────────────────────────────────
    //
    // 参数编辑器工具栏是**一整行**（标题 / 工具 / 平滑度在左，参数药丸 / 算法在右，
    // 同属 `h-qt-bar-main` 这一个容器），因此只观察这一个容器：任一处放不下即升级。
    // 完整层级见 `toolbarOverflow.ts` 的 `TOOLBAR_MAX_TIER`：
    //   1 标题 · 2 算法文本 · 3 张力药丸 · 4 气声药丸（开关保留）· 5 平滑度文本 ·
    //   6 平滑度数值 · 7 平滑度滑块 · 8 导入 MIDI · 9 参考轨道组文本 ·
    //   10 静态参数 + 算法下拉 → 「⋯」· 11 参数药丸整排 → 「⋯」
    // 原则是**只砍冗余、不砍入口**：被隐藏项要么是纯装饰文字（旁边控件已表达同一信息），
    // 要么功能另有入口；唯一入口（如参考轨道组、分离开关）只瘦身、不隐藏。
    // 平滑度滑块是**控件**、数值只是读数，故数值（6）先于滑块（7）让位。
    //
    // 【层级只能改 `display`】任何"到了某一级就不渲染"的写法都会让判据自反馈并振荡 ——
    // 铁律见 `toolbarOverflow` 模块头。视图侧只允许用 `<ToolbarSlot hideAt/showAt>`。
    const toolbarRowRef = useRef<HTMLDivElement | null>(null);
    const toolbarTier = useToolbarOverflowTier(toolbarRowRef, TOOLBAR_MAX_TIER);

    // 门禁与可见性（Compose / 分离两道门禁、药丸让位、参考轨道与副参数的眼睛）
    // 整块在 `usePianoRollParamGates` 里（见该文件头）。调用点留在原位：它内部注册的
    // 两个 effect 的相对顺序是行为的一部分。
    const {
        gatedPillHideOrder,
        composeEnabled,
        groupTrackIds,
        referenceRootTrackOptions,
        visibleReferenceRootTrackIds,
        pitchHardDisableReason,
        childPitchHardDisableReason,
        pitchEnabled,
        visibleSecondaryParamIds,
        updateVisibleReferenceRootTrackIds,
    } = usePianoRollParamGates({
        dispatch,
        editParam,
        s,
        rootTrackId,
        rootTrack,
        orderedProcessorParams,
        processorParams,
        separationEnabled,
        secondaryParamVisible,
        setSecondaryParamVisible,
        hoveredReferenceRootTrackId,
        setHoveredReferenceRootTrackId,
        t,
    });

    // 【已删除 `secPerBeat = 60 / bpm`】参数编辑器的选区、剪贴板映射、交互换算、
    // 取数窗口全部改为**帧制**（工程级常量栅格，见 `paramSelection.ts`），BPM 只
    // 剩下一个消费者：网格与标尺（`buildTimelineTicks` 直接读 `s.bpm` + Tempo Map）。
    // 因此这里不再需要"每拍秒数"，改 BPM 也不会再牵动选区或触发曲线重取。
    const contentWidth = contentWidthAt(pxPerSec);

    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const axisCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const axisWrapRef = useRef<HTMLDivElement | null>(null);
    const scrollStateRafRef = useRef<number | null>(null);
    /**
     * **上一次镜像写入 scroller 的原生偏移**（两轴各自记录）。
     *
     * 用途只有一个：判定原生 `scroll` 事件是不是镜像回写造成的**回声**
     * （见 `timeline/scrollEcho` 的 `isMirrorEcho`）。判据必须拿"我上次写下的值"做
     * 基准，而不是拿内核的当前值——连续滚动 / 缩放时事件报的是**上一帧**镜像的值，
     * 与内核此刻已差一整帧位移，用它判会稳定地把回声误收成用户输入，进而把量化误差
     * 推回共享视口，让时间轴跟着抽动。
     *
     * 初值 `NaN` = 从未写过 → 一律判"不是回声"（宁可多采纳一次也不吞掉首帧前的输入）。
     */
    const lastMirroredScrollLeftRef = useRef(Number.NaN);
    const lastMirroredScrollTopRef = useRef(Number.NaN);

    const rulerContentRef = useRef<HTMLDivElement | null>(null);
    const gridLayerRef = useRef<HTMLDivElement | null>(null);

    // ── 渲染内核（滚动 / 视口所有权）────────────────────────────────────
    //
    // 【所有权在谁手里】内核是唯一渲染路径，宿主持有滚动真值，**原生 scroller 只是
    // 被动镜像**——宿主每帧把真值写回它，使尚未迁移的输入代码（中键拖拽、框选自动
    // 滚动等直接读写 `scroller.scrollLeft` 的地方）读到的仍是同一份位置，行为不变。
    // 各图层由宿主的帧回调 `applyScrollLayers` 驱动（取代此前的 `scroll` 事件）。
    //
    /**
     * 内核数据镜像（每次 render 更新字段，宿主每帧现读）。
     *
     * 特殊说明：显式标注为 `PianoRollKernelData` 而不是让它靠初始值推断——推断出的
     * 窄类型会让后续写入 `grid`（可选字段）报错，也会漏掉字段名拼写错误。
     */
    const kernelDataRef = useRef<MutablePianoRollKernelData>({
        projectSec: 1,
        valueDomain: { min: 0, max: 1, span: 1 },
        grid: null,
        overlay: null,
        selectionBand: null,
        paramBlocks: null,
        paramBlocksVisible: false,
        paramBlockSamples: null,
    });
    /**
     * 块手势事务的曲线帧。存在时，曲线层和块样本都只读这一份冻结窗口；
     * React 的 paramView / live 层即使异步刷新也不能参与当前帧。
     */
    const paramBlockInteractionCurveRef = useRef<ParamViewSegment | null>(null);
    /** 自绘滚动条的 thumb（恒挂载）。 */
    const hScrollbarThumbRef = useRef<HTMLDivElement | null>(null);
    const vScrollbarThumbRef = useRef<HTMLDivElement | null>(null);
    /** 自绘滚动条的**轨道**（承接「点空白翻页」，恒挂载）。 */
    const hScrollbarTrackRef = useRef<HTMLDivElement | null>(null);
    const vScrollbarTrackRef = useRef<HTMLDivElement | null>(null);
    /** GL 静态层画布：网格 / 键盘几何，恒挂载（它是这些图层的唯一绘制者）。 */
    const glCanvasRef = useRef<HTMLCanvasElement | null>(null);
    /** 键盘轴 GL 画布（阶段 2/3）：键盘几何 + 音名标签。独立画布：轴列不随横向滚动移动。 */
    const glAxisCanvasRef = useRef<HTMLCanvasElement | null>(null);
    /** 动态叠加层 GL 画布（阶段 2/3）：播放头，位于曲线层之上，恒挂载。 */
    const glOverlayCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const [viewSize, setViewSize] = useState({ w: 1, h: 1 });
    /**
     * 当前 dpr（React 形态）：仅供**渲染期内联样式**里的物理像素线宽使用。
     * 命令式绘制路径每帧现读 `readDevicePixelRatio()`，不经此值。
     */
    const devicePixelRatio = useDevicePixelRatio();
    const [timeDisplaySettingsOpen, setTimeDisplaySettingsOpen] = useState(false);
    // 参数编辑器的内容绘制在 sticky 视口层中，滚动范围由后面的 spacer 提供。
    // 两个子元素按垂直方向堆叠，因此 scrollWidth 取二者宽度最大值；
    // 想让原生最大滚动位置为「工程宽」，spacer 需再加一个视口宽。
    //
    // 【不变量：与轨道视图同一份可滚范围】这里复用轨道视图的同一个函数
    // （`resolveTimelineScrollRange`），两者都传自己的视口宽——于是两个面板的
    // 最大滚动位置**逐值相等**（都 = 内容宽）。
    //
    // 【为什么同步偏移不进这里（用户报告的「工程结尾右侧空白长度不同」）】spacer 的
    // 宽度决定原生最大滚动位置。此前同步开启时会再加一个 `timelineOffsetPx`，参数
    // 编辑器因此比轨道视图多出一段「只有自己能滚」的空白：滚到最右时轨道视图停在
    // 内容宽处（工程结尾落在轨道区左缘），参数编辑器却能再滚一个偏移量（工程结尾落到
    // 轨道区左缘以左），两个面板不再对齐。偏移是**纯投影量**，只改内容画在屏幕上的
    // 位置（见内核宿主的 `horizontalOffsetPx`），不参与可滚范围。
    const timelineScrollRange = useMemo(
        () => resolveTimelineScrollRange({ contentWidth, viewportWidth: viewSize.w }),
        [contentWidth, viewSize.w],
    );
    const paddedContentWidth = timelineScrollRange.paddedContentWidth;

    useLayoutEffect(() => {
        const el = scrollerRef.current;
        if (!el) return;
        const ro = new ResizeObserver(() => {
            // 【为什么用 clientWidth（整数）而不是 getBoundingClientRect（分数）】
            // 这个值同时是下方 sticky 包裹层的 CSS `width`（`style={{ width: viewSize.w }}`）
            // 与各画布 `rasterize` 的 CSS 尺寸入参。包裹层是 `overflow: hidden`，
            // 画布被 `rasterize` 吸附到设备像素后最多比它宽 `0.5 / dpr` CSS px ——
            // 这点亚像素差被裁掉、不产生滚动条；反过来若取分数宽度，包裹层与画布
            // 会各自落在不同的亚像素相位上，投影锚点（left: 0）之外的边缘更容易露缝。
            // 用整数是"最不易出错"的一侧；`Math.floor` 在此是恒等操作
            //（clientWidth 已是整数），保留只为显式表达"这里要的是整数 CSS 宽"。
            const w = Math.max(1, Math.floor(el.clientWidth));
            const h = Math.max(1, Math.floor(el.clientHeight));
            viewSizeRef.current = { w, h };
            setViewSize({ w, h });
            // 视口尺寸（含首次测量）必须立即同步到参数编辑器总线：总线
            // 初始窗口是 1px 占位，PianoRollWaveformSurface 按总线 axis
            // 裁剪（窗口外 clip 直接跳过、不发起取数）——fresh 应用在
            // 首次滚动/缩放之前总线永远不会被 emit，波形因此无法显示。
            pianoRollViewportBus.emit(scrollLeftRef.current, pxPerSecRef.current, w);
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, [pxPerSecRef, scrollLeftRef, scrollerRef, viewSizeRef]);

    /**
     * 标尺播放头线的**首帧与缩放后**定位。
     *
     * 【为什么位置不再交给 React 渲染】见 `TimeRulerPlayhead.positionFromProps`：
     * React 只持有 30Hz 的已提交播放头，而画布/GL 上的播放头用的是 60Hz 插值
     * 位置，两者同时写 `style.left` 会让标尺上的线落在旧位置 —— 用户看到的就是
     * "标尺的播放头与参数编辑器里的播放头分离"。
     *
     * 这里用 layout effect（绘制前执行）补上首帧与缩放后的位置，播放/滚动期间的
     * 逐帧位置由 `PianoRollPlayheadBridge` 的共享视觉时钟与 `applyScrollLayers` 负责。
     */
    useLayoutEffect(() => {
        // 内核宿主每帧都在写这两个元素（单写者原则，见 4b0bbb40）：此 effect 只在
        // 无宿主的回退路径补首帧/缩放后定位，否则一次一帧过期的 React 提交会把
        // 播放头拉回旧位置（"标尺播放头与主体分离"回归）。
        if (hostRef.current !== null) {
            return;
        }
        // 缩放取**内核真值**（与 GL 播放头、标尺平移同一口径）：面板的 React
        // `pxPerSec` 是请求值，内核可能因视口宽度变化而钳制它，用请求值定位会与
        // 主体播放头差一个 `播放头秒数 × 缩放差`。此分支只在无宿主时执行，
        // 因此没有内核视口可取，直接退回面板值。
        const view = resolvePanelRenderViewport({
            kernelView: null,
            refPxPerSec: pxPerSecRef.current,
            refScrollLeftPx: scrollLeftRef.current,
        });
        const leftPx = playheadLineLeftPx(
            createTimelineAxis({
                pxPerSec: view.pxPerSec,
                scrollLeftPx: view.scrollLeftPx,
                viewportWidthPx: viewSizeRef.current.w,
                dpr: readDevicePixelRatio(),
            }),
            visualPlayheadSecRef.current,
        );
        if (rulerPlayheadLineRef.current) {
            rulerPlayheadLineRef.current.style.left = `${leftPx}px`;
        }
        if (rulerPlayheadHeadRef.current) {
            rulerPlayheadHeadRef.current.style.left = `${leftPx}px`;
        }
    }, [pxPerSec, pxPerSecRef, scrollLeftRef, viewSizeRef]);

    useEffect(() => {
        return () => {
            if (scrollStateRafRef.current != null) {
                cancelAnimationFrame(scrollStateRafRef.current);
                scrollStateRafRef.current = null;
            }
            if (zoomRafRef.current != null) {
                cancelAnimationFrame(zoomRafRef.current);
                zoomRafRef.current = null;
            }
            if (rafRef.current != null) {
                cancelAnimationFrame(rafRef.current);
                rafRef.current = null;
            }
        };
    }, [zoomRafRef]);

    // 内核宿主 + 各图层的同帧落地整块在 `usePianoRollScrollKernel` 里（见该文件头）。
    // 调用点留在原位：这几个函数的声明位置就是它们的**使用顺序**——`syncScrollLeft` 在
    // 各交互与滚动路径里被延迟读取（与 `commitViewportNow` 提前交给
    // `usePianoRollViewport` 是同一模式），其余调用者都在本行之后。
    const {
        applyScrollLayers,
        applyHorizontalScrollPosition,
        applyEdgeScrollLeft,
        syncScrollLeft,
    } = usePianoRollScrollKernel({
        s,
        hostRef,
        scrollerRef,
        scrollLeftRef,
        lastScrollLeftRef,
        lastMirroredScrollLeftRef,
        lastMirroredScrollTopRef,
        pxPerSecRef,
        viewSizeRef,
        paramEditorSyncTimelineRef,
        timelineOffsetRef,
        timelineSyncApplyingRef,
        scrollStateRafRef,
        setScrollLeft,
        visualPlayheadSecRef,
        rulerPlayheadLineRef,
        rulerPlayheadHeadRef,
        captureBandsRef,
        rulerContentRef,
        gridLayerRef,
        drawRef,
        kernelDataRef,
        editParamRef,
        pitchViewRef,
        paramViewsRef,
        hScrollbarThumbRef,
        vScrollbarThumbRef,
        hScrollbarTrackRef,
        vScrollbarTrackRef,
        glCanvasRef,
        glAxisCanvasRef,
        glOverlayCanvasRef,
        getCurrentViewportForScrollbar,
        getParamValueBoundsForScrollbar,
        /*
         * 块由宿主按实时轴 + 实时值域直接绘制与命中，视口变化不需要 React
         * 重新投影；宿主自己的滚动订阅已经会请求完整帧提交。
         */
    });

    /**
     * **立即**提交一帧（有宿主时同任务提交，否则退回标脏 rAF）。
     *
     * 【为什么需要它：DOM / Canvas2D 与 GL 必须落在同一个任务里】
     * 面板的权威写入点（缩放落地、共享视口应用、值域平移/缩放）会同步写 DOM /
     * Canvas2D。若 GL 侧（参数线 / 原始音高线 / 播放头）等下一帧才跟上，同一份视口
     * 就被两套图层分帧呈现——用户看到的就是"这些线迟缓几帧渲染"。
     *
     * 实测（Chrome，参数编辑器内滚轮缩小，按帧记录两个绘制路径的 scrollLeft）：
     * 面板在 t=6384 写下 DOM/Canvas2D，GL 到 t=6446 才用同一个值重绘（React 提交 +
     * rAF 重新排队，慢一帧到数帧）。改用本方法后两者同任务提交，手感与旧实现
     * （原生滚动 + 在事件里同步重绘）一致。
     *
     * 与 `invalidate()` 的分工：数据变更（编辑结果、主题、音阶）仍走 rAF 合并；
     * **用户手势的视口提交**走本方法。
     */
    function paintNow(): void {
        const host = hostRef.current;
        if (host) {
            host.paintNow();
            return;
        }
        invalidate();
    }

    /**
     * 视口提交的统一入口：内核写入 + 各图层**同任务**提交。
     *
     * 流程：内核写入（含钳制与标脏）→ `paintNow()`（同一任务的 GL + DOM + Canvas2D 提交）。
     *
     * 【缩放必须与位置**一起**写入内核（否则画布层会抽搐一帧）】
     * 只写位置时，内核仍是**旧缩放**：本函数紧接着的同帧绘制（波形 / 参数线 / 播放头 /
     * 选区都在 GL 或 Canvas2D 上）就会用旧缩放画出新位置，而 DOM 侧的标尺 / 网格（由
     * React 的 `pxPerSec` 驱动）已经是新缩放——一屏之内两套缩放，下一帧才被纠正。
     * 实测（Chrome，参数编辑器内滚轮缩放，日志打在绘制前）：`kernelPps=150 targetPps=165`
     * ——画布按 150 画、DOM 按 165 画。传 `pxPerSecAtCommit` 后内核先原子换到新缩放，
     * 同帧绘制即与 DOM 一致（`kernelPps=165 targetPps=165`）。
     *
     * 特殊说明：宿主尚未创建（挂载期）时退回 `applyScrollLayers` 直接写 DOM，与各调用点
     * 原本的兜底语义一致。
     *
     * @param drawingScrollLeft 目标水平位置（绘制坐标）。
     * @param pxPerSecAtCommit 本次提交应生效的缩放；缺省表示缩放不变（纯滚动）。
     */
    function commitViewportNow(drawingScrollLeft: number, pxPerSecAtCommit?: number): void {
        const host = hostRef.current;
        if (host) {
            if (pxPerSecAtCommit !== undefined && Number.isFinite(pxPerSecAtCommit)) {
                // 缩放 + 位置**一次原子写入**：内核用新缩放算上限并钳制，随后同帧绘制
                // 用的就是新缩放（见上方说明）。
                host.setViewport({ pxPerSec: pxPerSecAtCommit, scrollLeft: drawingScrollLeft });
            } else {
                applyHorizontalScrollPosition(drawingScrollLeft);
            }
            host.paintNow();
            return;
        }
        applyHorizontalScrollPosition(drawingScrollLeft);
        applyScrollLayers(drawingScrollLeft);
    }

    // 内核数据镜像：每次 render 更新字段（宿主每帧现读，不触发重建）。
    //
    // 特殊说明 1：`projectSec` 必须是**秒**而不是 `contentWidth`（像素）——宿主内部
    // 会自行乘以 pxPerSec 得到内容宽，传像素会让内容宽度被放大 pxPerSec 倍，
    // 横向滚动条 thumb 缩成一条线。
    //
    // 特殊说明 2：值域（`min/max/span`）必须与面板当前视口同源。宿主用它把像素位置
    // 换算成值域中心；若停留在占位值，竖向换算的比例就错了——表现为钢琴键盘整体
    // 偏移一个八度、竖向滚动条 thumb 位置也不对（曾实际发生）。
    //
    // 特殊说明 3：`span` 用**当前视口**的跨度（而非值域全长）。竖向像素位置 ↔ 值域
    // 中心的映射依赖 span，用错会让滚到同一位置对应的中心值不同。

    /**
     * 构建 GL 场景层的网格输入（阶段 2）。
     *
     * 流程：由 `editParam` 判定网格种类 → 取值域边界与当前视口 → 解析该主题的两条
     * 网格线颜色（以及钢琴背景 / 音阶高亮颜色）为数值 RGBA → 绑定 `valueToY` →
     * pitch 下再补键盘颜色与音阶音级。
     *
     * 特殊说明 1：`valueToY` 传的是**面板自己的**那个函数（只绑定 `editParam`），
     * 与 Canvas2D 路径共用同一份投影——这正是两种渲染模式网格不会错位的原因。
     *
     * 特殊说明 2：颜色经 `parseRgbaColor` 解析为 0..1 浮点，因为 GL 上传统一用浮点；
     * 在渲染热路径上反复解析 CSS 字符串是纯浪费，故在镜像更新（低频）时做。
     * 解析失败（拿到 NaN）时退回**不透明黑**——宁可颜色不对也不要上传 NaN，
     * NaN 会让整个实例属性失效、整层消失。
     *
     * 特殊说明 3（音阶高亮）：条件与旧 Canvas2D 分支**逐字对齐**——`always` 模式且
     * 存在工程音阶时才给音级集合（`off` / 无音阶时给空数组，等价于不高亮）。
     *
     * 【已知限制：Tempo Map 分段音阶未迁移】旧 Canvas2D 路径还支持 Tempo Map 的
     * **分段音阶**（`buildScaleSegments`：不同时间段用不同音阶，按段画不同 x 范围）。
     * GL 网格几何是**视口坐标且不含时间轴**——它不知道自己在哪个时间段上，无法表达
     * 分段。要迁移需要把网格层做成时间相关（或把高亮拆成独立的时间感知图层），
     * 超出本次改动范围。因此这里只实现**单一工程音阶**的均匀路径：有 Tempo Map
     * 音阶变化的工程，高亮会按工程音阶（而非各段音阶）绘制。
     *
     * 特殊说明 4（钢琴背景）：黑键行背景带**恒开**、无开关、无持久化设置；它只是
     * 纹理，与音阶高亮（语义）互不影响，后者在 GL 实例顺序上压在它之上。
     *
     * @returns 网格输入；GL 关闭或无法解析时返回 null（宿主按"没有网格"处理）。
     */
    function buildGridSpec(): PianoRollGridSpec | null {
        return buildGridSpecImpl({
            clampViewport,
            editParam,
            editParamAxisUnit,
            effectiveProjectScale,
            fontFamily,
            getCurrentViewportForScrollbar,
            getParamValueBoundsForScrollbar,
            prAxis,
            processorParamsRef,
            s,
            themeMode,
            valueToY,
        });
    }

    /**
     * 内核数据镜像：项目时长 / 值域 / 网格输入。
     *
     * 特殊说明 1：网格输入必须每次构建——它是 GL 静态层的唯一来源（Canvas2D 侧的
     * 音高网格分支已整体删除，网格只有 GL 一个绘制者）。颜色解析会创建 DOM 探针
     * （`normalizeCssColor`），但这是唯一路径，没有可省的余地。
     *
     * 特殊说明 2（**依赖项必须覆盖 buildGridSpec 读取的全部输入**）：本 effect 是
     * 镜像里 `grid` 的**唯一**写入点，因此 `buildGridSpec` 读到的每一个输入都必须
     * 在依赖里出现，否则该输入变化时镜像不刷新、签名不变、GL 几何不重建——症状
     * 正是"点了按钮但画面纹丝不动"。因此除视口 / 参数外，还必须含：
     * - `themeMode`：决定两套配色（切主题后背景带与网格线都要换色）；
     * - `s.scaleHighlightMode`：决定音阶高亮是否产生音级集合（缺陷 #6）；
     * - `effectiveProjectScale`：换调 / 换音阶后高亮的音级集合变化。
     *
     * 实测（补依赖之前）：点击「音阶高亮」按钮后 Redux 状态确实翻转（按钮 variant
     * `ghost → solid`），但 GL 画布的像素**完全不变**——本 effect 没重跑，镜像里的
     * `grid` 还是旧对象（连带旧签名），几何自然不重建。补上依赖后同样操作能画出
     * 琥珀色强调线（像素级实测：变化行数 = 音阶音级数 × 每线设备像素行数，且再点
     * 一次回到逐像素相同）。
     */
    useLayoutEffect(() => {
        const bounds = getParamValueBoundsForScrollbar(editParam);
        const view = clampViewport(editParam, getCurrentViewportForScrollbar(editParam));
        kernelDataRef.current.projectSec = dynamicProjectSec;
        kernelDataRef.current.valueDomain = {
            min: bounds.min,
            max: bounds.max,
            span: view.span,
        };
        kernelDataRef.current.grid = buildGridSpec();
        // pxPerSec 由面板解析后写入内核（阶段 1 不迁移缩放，见宿主文件头）。
        hostRef.current?.setPxPerSec(pxPerSec);
        // 同步偏移会改变**水平上限**（原生域 = 内容宽 + 偏移），必须按新边界重钳。
        hostRef.current?.reclamp();
        hostRef.current?.invalidate();
        // `pitchViewRef` / `paramViewsRef` 是 ref（变更不触发渲染），故依赖项里无法列出；
        // 值域跨度变化由下面的显式同步点负责（`syncVerticalScrollbarForViewport`）。
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [
        dynamicProjectSec,
        pxPerSec,
        editParam,
        processorParams,
        s.paramEditorSyncTimeline,
        timelineOffsetPx,
        // 见特殊说明 2：buildGridSpec 读到的主题 / 高亮 / 音阶都必须在此列出。
        themeMode,
        s.scaleHighlightMode,
        effectiveProjectScale,
        // 纵轴展示单位（倍率 / dB）：同样是 buildGridSpec 的输入，漏了就是
        // "点了切换但刻度纹丝不动"（签名不变 → GL 几何与文字都不重建）。
        editParamAxisUnit,
        // Tempo Map 分段音阶（缺陷 #6）：换 Tempo Map 会换分段。
        s.tempoMap,
    ]);

    // 渲染期刷新 syncScrollLeft 引用（useCallback 化后引用仅在同步开关变化时更新）：
    // 供只注册一次的原生 `wheel` 监听器调用，避免闭包捕获陈旧实现。
    const syncScrollLeftRef = useRef(syncScrollLeft);
    syncScrollLeftRef.current = syncScrollLeft;

    // 【已删除：每帧对账自愈循环】
    // 该循环的存在前提是「原生 scroller 是滚动/缩放的唯一事实源」。内核是唯一渲染
    // 路径后真值源反转（内核持有、原生是被动镜像），用原生值反向对账会变成两个方向
    // 的自愈互相打架：内核刚写镜像，循环又把镜像读回来当作"权威"重发一次。内核自己
    // 每帧提交、且镜像由内核回写，故该职责整体消失。

    const valueToY = useCallback(
        (param: ParamName, v: number, h: number) =>
            valueToYImpl({ paramViewsRef, pitchViewRef, processorParamsRef }, param, v, h),
        [processorParamsRef],
    );

    const yToViewportT = yToViewportTImpl;

    const yToValue = useCallback(
        (param: ParamName, y: number, h: number) =>
            yToValueImpl({ paramViewsRef, pitchViewRef, processorParamsRef }, param, y, h),
        [processorParamsRef],
    );

    function clampViewport(param: ParamName, v: ValueViewport): ValueViewport {
        return clampViewportImpl({ processorParamsRef }, param, v);
    }

    function getParamValueBoundsForScrollbar(param: ParamName): { min: number; max: number } {
        return getParamValueBoundsForScrollbarImpl({ processorParamsRef }, param);
    }

    function getCurrentViewportForScrollbar(param: ParamName): ValueViewport {
        return getCurrentViewportForScrollbarImpl(
            { getParamValueBoundsForScrollbar, paramViewsRef, pitchViewRef },
            param,
        );
    }

    /**
     * 把值域视口同步到竖向滚动位置（值域 → 像素）。
     *
     * 流程：钳制视口 → 更新内核数据镜像的值域 → 把 `center` 交给滚动载体。
     *
     * 特殊说明 1：滚动载体是内核宿主。像素域恒为 0..1600，映射由
     * `verticalScrollTopFromCenter` 单一来源保证。宿主内部自带去重，故无需在这里
     * 再判一次差值。宿主尚未创建时（挂载期）退回原生 scroller，语义一致。
     *
     * 特殊说明 2（**必须先刷新镜像的 span**）：竖向缩放（钢琴键区 alt/ctrl+滚轮）
     * 只改 `span`，而视口存在 **ref** 里——ref 变更不触发 React 渲染，于是「内核数据
     * 镜像」那个 effect 不会重跑，内核读到的 `valueDomain.span` 仍是旧值。此时
     * `setValueCenter` 会按**旧 span** 反算像素位置，结果完全错位（实测：缩放后
     * nativeTop 直接掉到 0 且再也动不了）。因此这里在提交前就地刷新镜像。
     *
     * @param param 参数名（决定值域边界）。
     * @param view 目标值域视口。
     */
    function syncVerticalScrollbarForViewport(param: ParamName, view: ValueViewport): void {
        return syncVerticalScrollbarForViewportImpl(
            {
                clampViewport,
                getParamValueBoundsForScrollbar,
                hostRef,
                kernelDataRef,
                lastMirroredScrollTopRef,
                scrollerRef,
            },
            param,
            view,
        );
    }

    /**
     * 【已删除：`applyViewportFromVerticalScrollbar`（像素 → 值域）】
     *
     * 它的唯一调用者是原生 `scroll` 事件里的竖向对账分支，而那段的成立前提是
     * 「原生 scroller 是竖向事实源」。内核是唯一渲染路径后竖向由内核拥有，值域视口
     * 由宿主的 `onScrollTopFrame` **正向**回写（见上方内核宿主创建处的说明），
     * 反向采纳路径失去存在理由，故随分支一并删除。值域 → 像素方向仍由
     * `syncVerticalScrollbarForViewport` 提供。
     */

    // 编辑会话状态（选区 / 变形叠加 / 光标 / 笔画与平移手势 / 剪贴板槽位 /
    // "笔画进行中"标志）整块在 `usePianoRollEditSession` 里（见该文件头）。
    const {
        selectionRef,
        selectionUi,
        setSelectionUi,
        paramMorphOverlays,
        setParamMorphOverlays,
        canvasCursor,
        setCanvasCursor,
        strokeRef,
        panRef,
        clipboardRef,
        liveEditEndHandlerRef,
        liveEditActiveRef,
    } = usePianoRollEditSession({ dispatch, invalidate, s });

    const {
        paramView,
        authorityRevision: paramAuthorityRevision,
        getAuthorityWatermark,
        setParamView,
        secondaryParamViews,
        referencePitchViews,
        bumpRefreshToken,
        refreshToken,
        refreshNow,
        refreshSecondaryNow,
        notifyLiveEditEnded,
        isLoading,
    } = usePianoRollData({
        editParam,
        secondaryParamIds: visibleSecondaryParamIds,
        referenceRootTrackIds: visibleReferenceRootTrackIds,
        pitchEnabled,
        // SessionState 已正式声明 paramsEpoch；此前经 `as unknown as` 鸭子类型
        // 读取，字段一旦改名就静默退化成 0、曲线不再跟随时间线重取。
        paramsEpoch: s.paramsEpoch,
        rootTrackId,
        selectedTrackId: effectiveSelectedTrackId,
        scrollLeft,
        // 视口窗口只需要「秒 ↔ 像素」这一个系数。此前传的是 `pxPerBeat` +
        // `secPerBeat`（两者相乘才等于它），于是**改 BPM 会让取数 effect 重跑**、
        // 整条曲线白取一次；换成 pxPerSec 后 BPM 与取数彻底解耦。
        pxPerSec,
        viewWidth: viewSize.w,
        viewSizeRef,
        scrollLeftRef,
        pxPerSecRef,
        invalidate,
        liveEditActiveRef,
    });

    /**
     * 波形「可听结果」映射所需的响度自动化快照（整条工程的 volume 曲线 +
     * 动态目标/原声基线）。
     *
     * 【为什么独立于 paramView】波形的形变与"当前编辑哪个参数"无关 —— 用户在
     * **任何**参数面板都要实时看到音频波形（画一笔音量/动态曲线波形立刻跟着
     * 动）。paramView 只覆盖当前参数且随视口窗口化，撑不起这个语义；快照按
     * 整工程自适应 stride 拉取，滚动/缩放零重取（见 useLoudnessCurves）。
     */
    const loudnessFpMs = paramView?.framePeriodMs ?? 5;
    const loudnessProjectFrames = Math.max(1, Math.ceil((dynamicProjectSec * 1000) / loudnessFpMs));
    const {
        snapshot: loudnessSnapshot,
        analysisPending: loudnessAnalysisPending,
        snapshotFetchSeq: loudnessSnapshotFetchSeq,
        getLatestFetchSeq: getLatestLoudnessFetchSeq,
        flushPending: flushLoudnessPending,
    } = useLoudnessCurves({
        rootTrackId,
        projectFrames: loudnessProjectFrames,
        framePeriodMs: loudnessFpMs,
        liveEditActiveRef,
        paramsEpoch: s.paramsEpoch,
        refreshToken,
    });

    /**
     * live 覆盖读取器（解析按覆盖对象身份缓存，见 `createLiveOverrideReader`）。
     *
     * 【为什么必须缓存】`readLiveOverrideFor` 在波形几何重建的热路径上被反复
     * 调用 —— 几何层按列内增益切片调用幅度映射，一次重建可达数万次（实测
     * 2112 列 × 16 切片 × 2 = 6.8 万次）。若每次询问都 `split` key 再新建视图
     * 对象，仅字符串切分就要吃掉 ~26ms/帧 —— 用户报告的「编辑音量/动态时
     * 卡顿」。读取器把稳态压缩成一次身份比较，零分配。
     */
    const liveOverrideReaderRef = useRef<LiveOverrideReader | null>(null);
    if (liveOverrideReaderRef.current === null) {
        liveOverrideReaderRef.current = createLiveOverrideReader();
    }

    /**
     * 把 live 覆盖读成 LoudnessLiveCurve（按 key 中的参数 id 与窗口对齐）。
     *
     * 键解析与视图对象由 `createLiveOverrideReader` 按覆盖对象身份缓存。
     */
    const readLiveOverrideFor = useCallback((param: "volume" | "dyn") => {
        return liveOverrideReaderRef.current?.read(param, liveEditOverrideRef.current) ?? null;
        // eslint-disable-next-line react-hooks/exhaustive-deps -- 两个 ref 都是稳定引用（reader 在下方惰性创建、live 覆盖由 useLiveParamEditing 持有并就地更新）；加入依赖不会让回调更正确，反而会在每次渲染换掉引用、使幅度映射的几何缓存键失效
    }, []);

    /**
     * 波形响度映射的修订号（ref，**不触发 React 渲染**）。
     *
     * 绘制中的 live 覆盖写在 ref 上，幅度映射内部用延迟取值读它，因此函数引用
     * 不变也能产出不同结果 —— 几何缓存若只按引用比较会误判"没变"，
     * 绘制中的曲线就画不上去。这里用 ref 累加计数、映射通过 `revision()`
     * 惰性读取：若改成 state，拖动时每次 pointermove 都会重渲染整块面板。
     */
    const loudnessWaveformRevisionRef = useRef(0);

    /**
     * 时间轴几何手势的「按下时几何」快照（见 `clipGeometryPreviewBus`）。
     *
     * 【为什么用 state 而不是 ref】手势起止是**离散**事件（每只手势各一次），
     * 而它决定波形的乘数从哪儿来 —— 必须触发一次重渲染，让幅度映射挂上/撤下
     * 几何映射。手势内部的逐帧变化不走 state（走 `loudnessWarpRef` + LUT 的
     * 映射身份比较），否则整块面板会以指针频率重渲染。
     */
    const [clipGeometryPreviewOrigin, setClipGeometryPreviewOrigin] = useState<
        readonly ClipGeometrySnapshot[] | null
    >(() => getClipGeometryPreviewOrigin());

    /**
     * 收尾期的「基线溯源基准」：手势结束**之前**最后落地的那份快照的溯源键。
     *
     * 【为什么不能用取数序号】提交会先 `checkpointHistory()` 递增 `paramsEpoch`
     * （早于后端写入），而取数序号在**发出**时（`useLoudnessCurves` 的
     * `fetchReqIdRef`）就推进 —— 水位与"这份数据反映哪份几何"必然错拍，
     * 于是映射时而提前撤下（闪一下）、时而过晚撤下（新基线被再搬一次）。
     *
     * 溯源键是**事实**：拖拽期间后端几何被冻结 ⇒ 键恒定；提交写回后端后键必变。
     * 于是"键变了"当且仅当"这份基线反映新几何"。判定在**渲染期**完成（见
     * `warpSettled`），撤下与快照落地因此落在同一个 React 提交里，
     * 结构上不存在"新快照 × 旧映射"或"旧快照 × 无映射"的中间帧。
     */
    const warpBasisKeyRef = useRef<string | null>(null);
    /** 键不可用时的退化基准（快照对象身份），见 `warpSettled`。 */
    const warpBasisSnapshotRef = useRef<LoudnessSnapshot | null>(null);
    /** 手势是否已结束、正在等权威快照追上。 */
    const warpSettlingRef = useRef(false);

    useEffect(() => {
        setClipGeometryPreviewOrigin(getClipGeometryPreviewOrigin());
        warpSettlingRef.current = false;
        return subscribeClipGeometryPreview(() => {
            const origin = getClipGeometryPreviewOrigin();
            if (origin !== null) {
                warpSettlingRef.current = false;
                setClipGeometryPreviewOrigin(origin);
                return;
            }
            // 手势结束：**不在这里**撤下映射。提交之后、权威快照回来之前，映射仍是
            // 把基线搬到新位置的唯一正确来源（后端此刻正在算的正是同一件事）。
            // 撤下交给 `warpSettled`（渲染期判定，与快照落地同批）。
            //
            // 【取消手势为什么同样安全】取消会把 Redux 几何还原成按下时的值，
            // 差分随之退化为空 ⇒ 映射为 `null`，撤不撤在画面上没有区别。
            warpSettlingRef.current = true;
        });
    }, []);

    /**
     * 当前生效的几何映射（渲染期同步的 ref 镜像，见文件内既有的同款模式）。
     *
     * 幅度映射以 provider 惰性读取它，因此映射对象的**引用**保持不变（几何缓存与
     * 组件 memo 不会因它而重建），而变化由查表键里的映射身份比较捕捉
     * （见 `makeLoudnessAmplitudeMap` 的 `warp` 说明）。
     */
    const loudnessWarpRef = useRef<LoudnessGeometryWarp | null>(null);
    const readLoudnessWarp = useCallback(() => loudnessWarpRef.current, []);

    /**
     * 波形重绘的**帧内合并**调度（与曲线层同用 `renderKernel/renderLoop` 语义）。
     *
     * 【为什么需要】一次全量重建要重算两千余列 × 16 切片的包络并上传数百 KB
     * 顶点（实测典型窗口 ~2.7ms）。而绘制中的 live 覆盖在**同一帧内可能被更新
     * 多次**：手绘工具会把一个 `pointermove` 的 coalesced 采样全部展开处理
     * （`usePianoRollInteractions` 的 `flushPendingMoves`），高刷鼠标 / 笔一帧
     * 可积 2~8 个采样。此前每个采样都**同步**强制提交一次全量重建 —— 单帧
     * 8~30ms 的几何工作，正是用户报告的"编辑音量/动态时卡顿"。
     *
     * 改为标脏 + rAF 合并后一帧至多重建一次，且与曲线层（面板自己的
     * `invalidate()` → 宿主 renderLoop）落在**同一帧**，两层不再错帧。
     *
     * 【为什么不用同帧 `flush()`】滚动路径要求同帧提交，是因为要与原生滚动的
     * DOM 内容层对齐（见 PianoRollWaveformSurface / viewportBus 的契约）。绘制
     * 参数曲线的路径没有随滚动移动的 DOM 层，与曲线层同帧即可。
     */
    const waveformRepaintLoopRef = useRef<RenderLoop | null>(null);
    if (waveformRepaintLoopRef.current === null) {
        waveformRepaintLoopRef.current = createRenderLoop({
            draw: () => pianoRollViewportBus.invalidate(),
        });
    }
    useEffect(() => {
        const loop = waveformRepaintLoopRef.current;
        loop?.start();
        return () => loop?.stop();
    }, []);

    /**
     * 强制重绘参数面板波形（**仅当绘制中的 live 覆盖会改变波形时**）。
     *
     * 必要性：绘制中的响度曲线只写在 `liveEditOverrideRef`（ref 变更不触发
     * React 渲染），而波形面是 memo 组件 + 几何缓存，既收不到 ref 变更、
     * 也不会因 props 未变而重绘。故显式强制重绘一次：总线以同一份投影 force
     * commit，波形图层命中重绘并因修订号变化而重建几何（其余图层内容未变，
     * 开销可忽略）。
     *
     * 【为什么必须按参数过滤】波形画的是「可听结果」
     * （`源峰值 × clip增益×淡化 × volume(t) × dyn增益(t)`，见
     * `makeLoudnessAmplitudeMap`）—— 它**不依赖音高 / 共振峰 / 齿度**等参数。
     * 而波形面的几何重建在总线驱动下是**每次全量**（`WaveformSurface.draw`：
     * 总线驱动的 `canReuse` 恒为 false，见该函数的说明），一次重建要重算
     * 两千余列 × 16 切片的包络。绘制音高时逐帧触发它纯属浪费 —— 一次重建
     * 在参数面板的典型窗口下约 2.7ms（响度映射），叠加曲线自身重绘后会
     * 明显抬高指针帧成本。因此这里用当前 live 覆盖的参数 id 做闸门：
     * 只有编辑 volume / dyn 时才推进修订号并请求重绘。
     *
     * 【与曲线绘制的关系】曲线（选区、绘制中的参数线）由面板自己的
     * `invalidate()` → 宿主帧提交驱动，**不经过本函数**；因此本闸门只影响波形
     * 面，绘制音高时曲线仍然逐帧更新。
     */
    const requestWaveformRepaint = useCallback(() => {
        // 闸门：只有编辑 volume / dyn 时波形才可能变（波形画的是可听结果，
        // 与音高 / 共振峰等参数无关）。判定复用读取器的解析缓存，零分配。
        if (!liveOverrideReaderRef.current?.affectsWaveform(liveEditOverrideRef.current)) {
            return;
        }
        // 修订号在**请求时**推进（不是绘制时）：同一帧内其它路径触发的绘制也要
        // 看到"上次重建之后 live 覆盖变过"，否则几何缓存会误判为可复用。
        loudnessWaveformRevisionRef.current += 1;
        // 绘制请求帧内合并：同一帧内多次请求只重绘一次（见 waveformRepaintLoop）。
        waveformRepaintLoopRef.current?.invalidate();
        // eslint-disable-next-line react-hooks/exhaustive-deps -- liveOverrideReaderRef / waveformRepaintLoopRef 是稳定引用（惰性创建）；live 覆盖经 ref 读取，故本回调必须保持引用稳定（它进入 usePianoRollInteractions 的依赖链）
    }, []);

    /**
     * 参数面板波形的幅度映射（**所有参数统一**）。
     *
     * 把源波形画成**应用响度自动化之后的可听结果**
     * （`源峰值 × clip增益×淡化 × volume(t) × dyn增益(t)`）：
     * - 编辑任何参数时画一笔音量/动态曲线，波形立刻跟着起伏；
     * - 切换参数时波形保持稳定（映射不依赖 editParam），只有曲线叠加层变化；
     * - 音量↔动态互转前后波形逐像素不变 —— 等效性最直观的佐证。
     *
     * 【引用稳定性】该值参与 `WaveformSurface` 的几何缓存键，因此必须 memo：
     * 只在快照对象真正变化时换引用。绘制中的 live 覆盖不改变本对象
     * （延迟取值 + 修订号），由 `requestWaveformRepaint` 驱动重建。
     *
     * 【恒等快化】volume 恒 1 且动态无基线（未使用响度自动化的工程）时
     * `identity = true` → 不挂映射，与时间线及既有行为逐像素一致。
     */
    const pianoRollAmplitudeMap = useMemo(() => {
        if (!loudnessSnapshot || loudnessSnapshot.identity) return undefined;
        if (loudnessSnapshot.volume.length === 0) return undefined;
        return makeLoudnessAmplitudeMap(
            loudnessSnapshot,
            {
                volume: () => readLiveOverrideFor("volume"),
                dyn: () => readLiveOverrideFor("dyn"),
            },
            () => loudnessWaveformRevisionRef.current,
            // 拖拽期间的几何映射：把基线/曲线搬到 clip 的当前位置，消除
            // 「旧位置的基线 × 新位置的峰值」造成的整段错位（见该模块的文件头）。
            readLoudnessWarp,
        );
    }, [loudnessSnapshot, readLiveOverrideFor, readLoudnessWarp]);

    const refreshSecondaryNowRef = useRef(refreshSecondaryNow);
    useEffect(() => {
        refreshSecondaryNowRef.current = refreshSecondaryNow;
    }, [refreshSecondaryNow]);

    useEffect(() => {
        if (!rootTrackId) {
            invalidate();
            return;
        }
        if (visibleSecondaryParamIds.length > 0 || visibleReferenceRootTrackIds.length > 0) {
            void refreshSecondaryNowRef.current();
            return;
        }
        invalidate();
    }, [invalidate, rootTrackId, visibleReferenceRootTrackIds, visibleSecondaryParamIds]);

    const handleRequestEnableCompose = useCallback(() => {
        const rtId = rootTrackId;
        if (!rtId) return;
        dispatch(
            setTrackStateRemote({
                trackId: rtId,
                composeEnabled: true,
            }),
        );
    }, [dispatch, rootTrackId]);

    /*
     * MIDI 导入：对话框状态、导入参数、拖放接入与「导入为 Clip」都归这一个 hook
     * （见 `useMidiImport.ts`）。它自带 15 个 useState 与一条独立通路，与画布、滚动、
     * 绘制没有交集 —— 是面板里少数几个边界清楚、可以整块搬走的功能。
     */
    const {
        clipboardGuid,
        closeLeadingGap,
        fillGaps,
        handleCloseLeadingGapChange,
        handleFillGapsChange,
        handleImportAsClip,
        handleImportBpmAsProjectChange,
        handleImportPositionChange,
        handleImportTargetChange,
        handleImportTempoMapEnabledChange,
        handleImportTempoMapKeySignatureChange,
        handleImportTempoMapTempoChange,
        handleImportTempoMapTimeSignatureChange,
        handleMidiImported,
        handleMultiTrackMergeChange,
        handleNoteBpmModeChange,
        handleOpenMidiDialog,
        handleSpecifiedBpmChange,
        importBpmAsProject,
        importPosition,
        importTargetParamEditor,
        importTargetReaperClipboard,
        importTempoMapEnabled,
        importTempoMapKeySignature,
        importTempoMapTempo,
        importTempoMapTimeSignature,
        midiDialogOpen,
        midiDialogOpenParamsRef,
        midiDialogSourceRef,
        midiPath,
        midiSelRanges,
        midiSelectionAvailable,
        multiTrackMerge,
        noteBpmMode,
        paramEditorMidiDragOver,
        setClipboardGuid,
        setMidiDialogOpen,
        setMidiDialogSelection,
        setMidiPath,
        specifiedBpm,
    } = useMidiImport({ dispatch, paramEditorRef, refreshNow, s, selectionRef, store });

    // 将当前选区发布到总线，供 MenuBar 等判断“工程音阶”是否受 Tempo Map 影响。
    // 多选区发布**包围区间**（首段起点 → 末段终点）：该提示只关心"选区覆盖的
    // 时间跨度里有没有音阶变化点"，包围区间是它的保守上界。
    useEffect(() => {
        const sel = selectionUi;
        const bounding = selectionBoundingSpan(sel);
        if (!bounding) {
            publishPianoRollSelection(null);
            return;
        }
        // 选区已经是帧制：直接发布，不再有任何换算（此前要经 `secPerBeat` 转一道）。
        publishPianoRollSelection({
            startFrame: Math.max(0, bounding.startFrame),
            frameCount: Math.max(1, bounding.frameCount),
            framePeriodMs: paramView?.framePeriodMs ?? 5,
        });
        return () => {
            publishPianoRollSelection(null);
        };
    }, [selectionUi, paramView?.framePeriodMs]);

    // 获取当前 track 下的所 ?clips，用 ?per-clip 波形叠加绘制
    // 获取轨道组内所有 clips（包含 root 轨道及所有子轨道的 clip）
    const trackClips = useMemo(() => {
        const inGroup = s.clips.filter((c) => groupTrackIds.has(c.trackId));
        if (inGroup.length > 0) return inGroup;
        // 旧工程 / 浏览器 mock 可能没有显式 childTrackIds；此时根轨道本身就是
        // 唯一成员，不能把整组当成空组，否则参数波形与块都会消失。
        const rootId = rootTrackId;
        if (rootId === null) return [];
        return s.clips.filter((c) => c.trackId === rootId);
    }, [s.clips, groupTrackIds, rootTrackId]);
    /**
     * 参数块的 Clip 来源。
     *
     * 参数编辑器画的是“根轨道组”的一条参数线，而参考音频 Clip 在
     * `trackClips` 里；`s.clips` 是**整个工程**的全部 Clip。把二者兜底混用会
     * 在根轨道没有子组时把别的轨道也派生进来；若当前选中轨道暂时没有可解析
     * 的 Clip，又会出现“曲线有数据但块规格为空”。这里只按根轨道组取。
     */
    const paramBlockClips = useMemo(() => {
        if (trackClips.length > 0) return trackClips;
        if (rootTrackId === null) return [];
        return s.clips.filter((clip) => clip.trackId === rootTrackId);
    }, [rootTrackId, s.clips, trackClips]);
    /**
     * 块来源的统一音节区间。
     *
     * 时间线刷新可能先更新 active Take 上的缓存，而顶层 `syllableSpansSec` 还是
     * 旧值；块派生若只读顶层，就会在分析成功后反而得到“尚未分析”并全部隐藏。
     */
    const paramBlockSyllableSpans = useMemo(() => {
        const byClip = new Map<string, { startSec: number; endSec: number }[]>();
        for (const clip of paramBlockClips) {
            const activeTake =
                clip.takes?.find((take) => take.id === clip.activeTakeId) ?? clip.takes?.[0];
            const spans = activeTake?.syllableSpansSec ?? clip.syllableSpansSec;
            if (spans !== undefined && spans.length > 0) {
                byClip.set(clip.id, spans);
            }
        }
        return byClip;
    }, [paramBlockClips]);

    /**
     * 拖拽期间的响度时域映射：把「快照所依据的几何」与当前几何的差分，应用到基线 /
     * 曲线（见 `loudnessGeometryWarp` 的文件头）。
     *
     * 【为什么只取本轨道组的 clip】原声基线是**按根轨道组**组装的；别的轨道组的
     * clip 与本组基线无关，把它们的映射算进来会错误地搬移本组曲线。`trackClips`
     * 已按 `groupTrackIds` 过滤，正好是基线的参与集合。
     *
     * 【为什么在渲染期判定收尾】`warpSettled` 只依赖"当前快照的溯源键"这一个事实，
     * 因此它翻转的那次渲染**就是**新快照生效的那次渲染：映射在同一次提交里变成
     * `null`，`pianoRollAmplitudeMap` 也在同一次提交里换新 —— 不存在中间帧。
     */
    const loudnessSnapshotKey = loudnessSnapshot?.baselineKey ?? null;
    // 非收尾期：基准跟随最新快照。拖拽期间后端几何被冻结，期间落地的快照同样是
    // "按下之前"的几何，对它可以安全地继续套用映射（不 re-base 会让映射平白失效）。
    if (!warpSettlingRef.current) {
        warpBasisKeyRef.current = loudnessSnapshotKey;
        warpBasisSnapshotRef.current = loudnessSnapshot;
    }
    /**
     * 收尾是否已完成：手势结束**之后**才落地的一份快照带回了不同的溯源键
     * （= 它反映的是新几何）。
     *
     * 键不可用（后端从未组装过基线）时退化为「快照换了对象即算追上」—— 该路径下
     * 后端没有可比对的几何信息，且基线本身也多半为空（映射在画面上无作用）。
     */
    const warpSettled =
        warpSettlingRef.current &&
        (warpBasisKeyRef.current !== null && loudnessSnapshotKey !== null
            ? loudnessSnapshotKey !== warpBasisKeyRef.current
            : loudnessSnapshot !== warpBasisSnapshotRef.current);

    const loudnessGeometryWarp = useMemo(() => {
        if (clipGeometryPreviewOrigin === null || warpSettled) return null;
        return createLoudnessGeometryWarp({
            origin: clipGeometryPreviewOrigin,
            clips: trackClips,
            framePeriodMs: loudnessFpMs,
            lockParamLines: s.lockParamLinesEnabled,
        });
    }, [
        clipGeometryPreviewOrigin,
        trackClips,
        s.lockParamLinesEnabled,
        loudnessFpMs,
        // 快照对象本身**不是**依赖：映射只由"按下时几何 + 当前几何"决定，快照换了
        // 对象而键没变（拖拽期间后端几何被冻结时的重取）不该让映射重算。
        // 需要跟着快照变化的那一件事 —— 收尾判据 `warpSettled` —— 已在依赖里。
        warpSettled,
    ]);

    // 收尾完成 ⇒ 可以安全丢掉基准几何：此时映射已是 `null`，且下一次手势会发布新的。
    useEffect(() => {
        if (!warpSettled || clipGeometryPreviewOrigin === null) return;
        warpSettlingRef.current = false;
        setClipGeometryPreviewOrigin(null);
    }, [warpSettled, clipGeometryPreviewOrigin]);

    /**
     * 映射变化 ⇒ **渲染期**推进修订号。
     *
     * 【为什么必须在渲染期】映射引用在渲染期写入 `loudnessWarpRef`（读取方在绘制期
     * 惰性读取），而修订号此前在被动 effect 里推进 —— 两者不同步：`WaveformSurface`
     * 的 layout effect 会在同一次提交里同步全量重建，它读到的将是「新映射 / 旧修订号」，
     * 几何缓存据此误判"可复用"从而画出上一帧的顶点。这里把修订号与引用一起在渲染期
     * 更新，任何一次绘制看到的 `(amplitudeMap, revision)` 都与当前映射一致。
     * 重绘**请求**仍交给下面的 effect（那只影响"什么时候画"，不影响"画得对不对"）。
     */
    const warpRevisionRef = useRef<LoudnessGeometryWarp | null>(null);
    if (warpRevisionRef.current !== loudnessGeometryWarp) {
        warpRevisionRef.current = loudnessGeometryWarp;
        loudnessWaveformRevisionRef.current += 1;
    }
    // 渲染期同步镜像（本文件既有的取值新鲜度模式：读取方在绘制期惰性读取）。
    loudnessWarpRef.current = loudnessGeometryWarp;

    /**
     * 映射变化 ⇒ 请求一次重绘。
     *
     * 【为什么不能只靠 rows 变化】拖拽期间 `rows` 确实每帧都换（几何跟着走），
     * 但映射的变化源不止几何：手势结束、锁定参数线开关、以及"等权威快照追上"的
     * 收尾，都可能在 `rows` 不变的情况下换掉映射。少了这一步，那些帧会停在旧映射
     * 上（画面与数据不一致）。
     */
    const lastLoudnessWarpRef = useRef<LoudnessGeometryWarp | null>(null);
    useEffect(() => {
        if (lastLoudnessWarpRef.current === loudnessGeometryWarp) return;
        lastLoudnessWarpRef.current = loudnessGeometryWarp;
        waveformRepaintLoopRef.current?.invalidate();
    }, [loudnessGeometryWarp]);

    // 帧装配（轴投影 → 峰值 → live 覆盖 → 曲线/选区图层 → `drawRef.current`）整块在
    // `usePianoRollFrameAssembly` 里（见该文件头）。调用点必须留在原位：`useClipsPeaksForPianoRoll`
    // 与 `useLiveParamEditing` 由它调用，状态槽与 effect 顺序取决于这个位置。
    const {
        prAxis,
        prTickAxis,
        clipPeaks,
        paramViewRef,
        liveEditOverrideRef,
        ensureLiveEditBase,
        applyDenseToLiveEdit,
        resetLiveEditPreview,
        clearLiveEditOverride,
        commitStroke,
        onParamCommitSucceeded,
        detectedPitchCurves,
        referencePitchOverlays,
        buildCurveLayers,
        buildSelectionBandSpec,
        secSpanClipRect,
        selectionClipRect,
    } = usePianoRollFrameAssembly({
        s,
        t,
        themeMode,
        fontFamily,
        pxPerSec,
        pxPerBeat,
        scrollLeft,
        viewSize,
        editParam,
        pitchEnabled,
        pitchHardDisableReason,
        childPitchHardDisableReason,
        effectiveProjectScale,
        trackClips,
        groupTrackIds,
        rootTrackId,
        rootTrack,
        visibleReferenceRootTrackIds,
        visibleSecondaryParamIds,
        hoveredReferenceRootTrackId,
        snapToggleHeld,
        paramView,
        paramBlockInteractionCurveRef,
        setParamView,
        secondaryParamViews,
        referencePitchViews,
        bumpRefreshToken,
        notifyLiveEditEnded,
        loudnessAnalysisPending,
        loudnessSnapshotFetchSeq,
        getLatestLoudnessFetchSeq,
        flushLoudnessPending,
        selectionRef,
        clipboardRef,
        paramViewsRef,
        pitchViewRef,
        paramMorphOverlays,
        pxPerSecRef,
        scrollLeftRef,
        viewSizeRef,
        visualPlayheadSecRef,
        canvasRef,
        axisCanvasRef,
        hostRef,
        kernelDataRef,
        drawRef,
        liveEditActiveRef,
        liveEditEndHandlerRef,
        valueToY,
        invalidate,
    });

    const handleEditActionRef = useRef<(op: string) => void>(() => {});
    const paramBlockSpecForDragSnapshotRef = useRef<
        ((snapshot: ParamBlockDragSnapshot) => PianoRollParamBlockSpec | null) | null
    >(null);
    /**
     * 块间圆滑手柄写入：**只改写右块**相接点之后的 `width` 帧。
     *
     * 过渡不是覆盖原曲线，而是从右块原曲线出发、按 smoothstep 权重拉向
     * 右块目标值；左块及其左侧帧始终不写。
     */
    /** 本次手势是否是拖拽块边缘（纵向拖动不启动）。 */
    const paramBlockResizeRef = useRef<{
        readonly blockId: string;
        readonly edge: "left" | "right";
        readonly pointerClientX: number;
        readonly startSec: number;
        readonly endSec: number;
        readonly neighborStartSec: number | null;
        readonly neighborEndSec: number | null;
        readonly resizeCandidates: readonly ParamBlock[];
        /** 手势起点的完整块几何；每帧从它重新投影，已删除块不会永久丢失。 */
        readonly baseBlockSpans: readonly {
            readonly id: string;
            readonly startSec: number;
            readonly endSec: number;
            readonly selected?: boolean;
        }[];
        /** 新建块预览的冻结块集；提交后 React 派生变化不能污染预览。 */
        readonly createBaseBlocks?: PianoRollParamBlockSpec["blocks"];
        readonly createdBlockId?: string;
        /** 手势起点的帧周期；提交与 minBlockSec 都按它折算，不受回包影响。 */
        readonly framePeriodMsBasis?: number;
        targetSec: number;
        /** 本帧完整伸缩计划；预览 / 宿主刷新 / 提交共用。 */
        resizePlan: ReturnType<typeof resolveParamBlockResizePlan> | null;
        /** 手势起点左右块的代表值/高度；仅用于锁定预览，不提交。 */
        readonly displayCenterValue?: number | null;
        readonly displayHeightPx?: number | null;
        /** 新建块的实时纵向值（只用于 `__create__` 预览）。 */
        readonly value?: number | null;
    } | null>(null);
    /**
     * 新建块松手后的交接。预览保持到 React 块记录与本轮权威曲线刷新都落地，
     * 然后跨越一帧再撤下，避免“预览 -> 空一帧 -> 正式块”闪烁。
     */
    const paramBlockCreateCommitRef = useRef<{
        readonly blockId: string;
        readonly authorityRevisionAtStart: number;
        readonly authorityRequestSeqAtCommit: number;
        readonly requiresCurveAuthority: boolean;
        committedSegments?: readonly { startFrame: number; values: readonly number[] }[];
        committed: boolean;
    } | null>(null);
    const paramBlockCreateReleaseRafRef = useRef<number | null>(null);
    /**
     * 边缘伸缩刚提交后的短前向快照。
     *
     * `applyParamBlockEdit` 通过 React state 提交，宿主却可能在同一次 pointerup
     * 后先绘一帧旧派生几何；用户会看到“松手闪回旧位置，再跳到新位置”。
     * 这里冻结刚提交的两个块区间，直到下一次派生真正包含它们。
     */
    const paramBlockCommittedSpanRef = useRef<{
        readonly blockId: string;
        readonly blockSpan: { readonly startSec: number; readonly endSec: number };
        readonly neighbors: readonly {
            readonly id: string;
            readonly startSec: number;
            readonly endSec: number;
            readonly remove: boolean;
        }[];
        readonly expiresAt: number;
    } | null>(null);
    /**
     * 纵向块拖动提交后的交接。
     *
     * 松手提交期间，曲线的 live 值已经到位，但 `paramView` 快照可能晚一帧。
     * 若此刻清拖动快照，宿主会按旧主簇画一次，块就闪回原位。这里保留快照与
     * 松手时的显示中心，直到提交后的权威曲线窗口同步。
     */
    const paramBlockDragCommitRef = useRef<ParamBlockDragCommit | null>(null);
    const paramBlockDragCommitReleaseRafRef = useRef<number | null>(null);
    /** Pointer activity and queued writes independently hold the refresh guard. */
    const paramBlockDragActivityRef = useRef<ParamBlockDragActivity>({
        pointerGestureToken: null,
        pendingCommits: 0,
    });
    const paramBlockCommitQueueRef = useRef<Promise<void>>(Promise.resolve());
    const paramBlockGestureSerialRef = useRef(0);
    const paramBlockActiveGestureTokenRef = useRef<number | null>(null);
    const paramBlockClickCandidateRef = useRef<{
        readonly blockId: string;
        readonly token: string;
        readonly at: number;
    } | null>(null);
    const hoveredParamBlockIdRef = useRef<string | null>(null);
    const hoveredTransitionIdRef = useRef<string | null>(null);
    const paramBlockDoubleClickTokenRef = useRef(0);
    // Stable callback that delegates to the latest handleEditOp via ref
    const stableEditAction = useCallback((op: string) => {
        handleEditActionRef.current(op);
    }, []);
    /** 块编辑动作在指针回调之后才定义；用 ref 打断声明顺序依赖。 */
    const applyParamBlockEditRef = useRef<
        ((result: ReturnType<typeof splitParamBlock>) => void) | null
    >(null);
    const paramBlockStateRef = useRef<ParamBlockState>(EMPTY_PARAM_BLOCK_STATE);
    /** 让宿主下一次绘制读到刚更新的拖动快照，而不是等某个 React 状态变化。 */
    const latestParamBlockSpecRef = useRef<PianoRollParamBlockSpec | null>(null);
    const invalidateParamBlockSpec = useCallback(() => {
        const snapshot = paramBlockDragSnapshotRef.current;
        const pv = paramViewRef.current;
        if (
            snapshot !== null &&
            shouldUseParamBlockDragSnapshotForRender({
                snapshotKey: snapshot.key,
                paramViewKey: pv?.key ?? null,
                gestureActive: paramBlockActiveGestureTokenRef.current !== null,
                commitHeld: paramBlockDragCommitRef.current !== null,
            })
        ) {
            kernelDataRef.current.paramBlocks =
                paramBlockSpecForDragSnapshotRef.current?.(snapshot) ?? null;
            const live = liveEditOverrideRef.current;
            const interactionCurve = paramBlockInteractionCurveRef.current;
            kernelDataRef.current.paramBlockSamples = {
                /*
                 * 代表值必须来自**手势起点**的曲线窗口。提交期间后端可能先把
                 * `paramView` 换成新快照；若这里改读新窗口，冻结块的区间与
                 * 数组下标会短暂错配，块本体就会在两帧位置之间跳。
                 */
                values:
                    interactionCurve?.edit ??
                    (live !== null && live.key === snapshot.key ? live.edit : snapshot.baseValues),
                startFrame: interactionCurve?.startFrame ?? snapshot.startFrame,
                stride: interactionCurve?.stride ?? snapshot.stride,
                framePeriodMs: interactionCurve?.framePeriodMs ?? snapshot.framePeriodMs,
                isPitch: editParam === "pitch",
                param: editParam,
            };
        } else {
            kernelDataRef.current.paramBlocks = latestParamBlockSpecRef.current;
        }
        invalidate();
    }, [
        editParam,
        invalidate,
        kernelDataRef,
        liveEditOverrideRef,
        paramBlockInteractionCurveRef,
        paramViewRef,
    ]);
    const paramBlockPointerRef = useRef<
        (args: {
            sec: number;
            localY: number;
            clientY: number;
            clientX: number;
            contextMenu: boolean;
            add: boolean;
            toggle: boolean;
            pointerType: string;
            pointerId: number;
            captureTarget: HTMLCanvasElement;
            setActivePointerGestureEnd: (endGesture: () => void) => void;
            clearActivePointerGestureEnd: (endGesture?: () => void) => void;
            pointerValue: (clientY: number) => number;
            altKey?: boolean;
            shiftKey?: boolean;
            ctrlKey?: boolean;
            metaKey?: boolean;
        }) => import("../gestures/pointerDeps").ParamBlockPointerDownResult
    >(() => ({ handled: false }));
    const splitParamBlockAtPlayheadRef = useRef<((snapToggleHeld?: boolean) => void) | null>(null);
    const deleteSelectedParamBlocksRef = useRef<(() => void) | null>(null);
    const groupSelectedParamBlocksRef = useRef<(() => void) | null>(null);
    const ungroupSelectedParamBlocksRef = useRef<(() => void) | null>(null);
    const restoreParamBlocksRef = useRef<((blockIds?: readonly string[]) => void) | null>(null);
    const restoreGameParamBlocksRef = useRef<(() => void) | null>(null);
    const normalizeSelectedParamBlockPitchRef = useRef<(() => void) | null>(null);
    /**
     * 正在拖动的块几何快照。
     *
     * 手势开始时冻结成员区间与起点曲线；拖动过程中宿主只读这份快照，不再
     * 每帧从“当前派生块集合”重算。否则合并记录、自动重叠归并与提交后的
     * paramView 刷新会在同一帧给出两套几何，正是块在两个位置快速闪烁的来源。
     */
    const paramBlockDragSnapshotRef = useRef<ParamBlockDragSnapshot | null>(null);
    /** 块命中容差（CSS px）：触摸 / 笔比鼠标更容易漂，分别给不同容差。 */
    const paramBlockHitSlopPx = 4;

    const interactions = usePianoRollInteractions({
        dispatch,
        rootTrackId,
        editParam,
        pitchEnabled,
        toolMode: s.toolMode,
        framePeriodMs: paramView?.framePeriodMs ?? 5,
        dynamicProjectSec,
        scrollLeftRef,
        pxPerSecRef,
        // 交互侧坐标换算的视口真值：与渲染侧（`resolvePanelRenderViewport`）同源，
        // 避免框选 / 命中测试读到量化提交滞后的 `scrollLeftRef`。见该字段的说明。
        getViewportTruth: useCallback(() => hostRef.current?.getViewport() ?? null, []),
        horizontalZoomChainRef,
        onHorizontalZoom: handleHorizontalZoom,
        syncTimelineEnabled: s.paramEditorSyncTimeline,
        timelineOffsetRef,
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
        resetLiveEditPreview,
        onParamCommitSucceeded,
        onParamBlockPointerDown: (args) => paramBlockPointerRef.current(args),
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
        paramStretchKb: stretchKb,
        vibratoAmplitudeAdjustKb,
        vibratoFrequencyAdjustKb,
        vibratoDragAmplitudeIncreaseKb,
        vibratoDragAmplitudeDecreaseKb,
        vibratoDragFrequencyIncreaseKb,
        vibratoDragFrequencyDecreaseKb,
        vibratoPresetPrevKb,
        vibratoPresetNextKb,
        vibratoPreset: activeVibratoPreset,
        straightVibratoPreset,
        vibratoPresetCycleList: enabledVibratoPresetList,
        onVibratoDragStateChange: setVibratoDragHud,
        onVibratoDragEnd: useCallback(() => {
            setVibratoDragHud(null);
        }, []),
        cycleDragDirectionKb,
        paramFineAdjustKb,
        onContextMenu: useCallback((x: number, y: number) => {
            setCtxMenu({ x, y });
        }, []),
        getPlayheadSec: getVisualPlayheadSec,
        playheadZoomEnabled: s.playheadZoomEnabled,
        paramEditorSeekPlayheadEnabled: s.paramEditorSeekPlayheadEnabled,
        pitchSnapEnabled: s.pitchSnapEnabled,
        pitchSnapUnit: s.pitchSnapUnit,
        projectScale: effectiveProjectScale,
        /** Tempo Map 感知：按帧时刻解析生效音阶。 */
        scaleAtSec: projectScaleAtSec,
        pitchSnapToleranceCents: s.pitchSnapToleranceCents,
        keybindingMap: mergedKeybindings,
        onEditAction: stableEditAction,
        dragDirection: activeDragDirection,
        onCycleDragDirection: useCallback(
            (tool: "select" | "draw" | "vibrato") => {
                dispatch(cycleDragDirection(tool));
                void dispatch(persistUiSettings());
            },
            [dispatch],
        ),
        edgeSmoothnessPercent: s.edgeSmoothnessPercent,
        onMorphOverlayChange: setParamMorphOverlays,
        currentParamRange,
        onPitchSnapGestureActiveChange: useCallback((active: boolean) => {
            setSnapGestureActive(active);
        }, []),
        paramValuePopupEnabled: s.showParamValuePopup,
        onParamValuePreviewChange: useCallback(
            (
                next: {
                    clientX: number;
                    clientY: number;
                    value: number;
                    displayText?: string;
                } | null,
            ) => {
                // 合帧提交：只记最新载荷、每渲染帧至多 setState 一次（与交互层
                // 各手势的 schedulePreview 同一动机）。浮窗是纯读数 UI，滞后一帧
                // 不可感知；不合帧时每次 move 的新对象都无法让 React 跳过重渲。
                paramValuePreviewPendingRef.current = { payload: next };
                if (paramValuePreviewRafRef.current != null) return;
                paramValuePreviewRafRef.current = requestAnimationFrame(() => {
                    paramValuePreviewRafRef.current = null;
                    const pending = paramValuePreviewPendingRef.current;
                    paramValuePreviewPendingRef.current = null;
                    if (!pending) return;
                    // 容器 rect 在这里（rAF 内、事件后）随读数同帧快照：
                    // EditorArea 渲染期只消费数值，不再 getBoundingClientRect
                    // （渲染期强制同步布局，hover 时每帧都付一次）。
                    const rect = canvasRef.current?.getBoundingClientRect();
                    setParamValuePreview(
                        pending.payload && {
                            ...pending.payload,
                            rectLeft: rect?.left ?? 0,
                            rectTop: rect?.top ?? 0,
                        },
                    );
                });
            },
            [],
        ),
    });

    // 参数数据更新后重算当前悬停浮窗：键盘平移参数线（"=" / "-" / "]" / "["
    // 及其 Shift/Ctrl 变体）、撤销/重做、远端写入都只更新 paramView 数据，
    // 不会触发 pointermove —— 悬停值是 pointermove 时的快照，若不在此重算，
    // 浮窗会一直显示旧值直到用户再次移动鼠标。声明顺序在 paramViewRef 同步
    // effect（上方）之后，重算读到的是本次渲染的最新数据。
    // refreshParamValuePreview 先解构出稳定引用：直接依赖 interactions 对象
    // 会让 effect 每次渲染都触发（setState 新对象 → 无限重渲染）。
    const { refreshParamValuePreview } = interactions;
    useEffect(() => {
        refreshParamValuePreview();
    }, [paramView, refreshParamValuePreview]);

    // 滚动容器与轴列的输入接线（原生 scroll 采纳、滚动条滚轮/翻页、标尺滚轮、
    // 轴列两个交互 hook、工具模式的光标与右键菜单复位）整块在
    // `usePianoRollScrollerWiring` 里（见该文件头）。调用点留在原位：轴列两个 hook
    // 的状态槽与 effect 顺序取决于这个位置。
    const {
        onScrollerScroll,
        attachVerticalScrollbarTrack,
        attachHorizontalScrollbarTrack,
        handleRulerWheel,
    } = usePianoRollScrollerWiring({
        s,
        dispatch,
        editParam,
        scrollerRef,
        axisWrapRef,
        hScrollbarTrackRef,
        vScrollbarTrackRef,
        viewSizeRef,
        hostRef,
        lastMirroredScrollLeftRef,
        lastMirroredScrollTopRef,
        timelineSyncApplyingRef,
        syncScrollLeftRef,
        pitchViewRef,
        paramViewsRef,
        processorParamsRef,
        processorParams,
        invalidate,
        clampViewport,
        getCurrentViewportForScrollbar,
        yToValue,
        setPitchView,
        setParamViewport,
        syncVerticalScrollbarForViewport,
        syncScrollLeft,
        formatParamValuePreview,
        interactions,
        horizontalZoomKb,
        scrollHorizontalKb,
        pianoKeysVerticalScrollKb,
        pianoKeysVerticalZoomKb,
        setCanvasCursor,
        setCtxMenu,
        setAxisValuePreview,
    });

    // 同步数据加载状态到状态栏。
    //
    // 走**外部 store** 而非 Context：这个 boolean 只服务状态栏一处显示，而它随每次
    // 取数（水平缩放 / 滚动 / 编辑提交）翻转 —— 曾经它住在包裹整个 AppInner 的
    // Context 里，于是每次翻转都重渲染整棵应用树，把正在进行的绘制笔画打断。
    // 现在只有状态栏那个小组件会重渲染（见 pianoRollStatusBus）。
    //
    // 按**窗体 id** 记账：参数编辑器可以多开，共用一个 boolean 会让一个实例卸载时
    // 抹掉另一个实例的状态。
    const loadingPublisherId = dockFormId ?? "param-editor";
    useEffect(() => {
        setPianoRollLoading(loadingPublisherId, isLoading);
    }, [isLoading, loadingPublisherId]);

    // 卸载时注销（否则本实例的最后状态会永远粘住，状态栏再也不会收起）。
    useEffect(() => () => clearPianoRollLoading(loadingPublisherId), [loadingPublisherId]);

    // 编辑命令（全选、命令分发、剪贴板键盘与外部同步接线）整块在
    // `usePianoRollEditCommands` 里（见该文件头）。
    const { selectAllParamRange, handleEditOp } = usePianoRollEditCommands({
        activeVibratoPresetId,
        bumpRefreshToken,
        clipboardRef,
        currentParamDefaultValue,
        currentParamQuantizeUnit,
        currentParamRange,
        dispatch,
        dynamicProjectSec,
        editParam,
        effectiveProjectScale,
        handleEditActionRef,
        invalidate,
        midiDialogOpenParamsRef,
        midiDialogSourceRef,
        openVibratoDialog,
        paramView,
        paramViewRef,
        pitchEnabled,
        processorParamsRef,
        projectScaleAtSec,
        resolvedVibratoPresets,
        resolveScaleFromToken,
        rootTrackId,
        s,
        scrollerRef,
        selectionRef,
        setClipboardGuid,
        setMidiDialogOpen,
        setMidiDialogSelection,
        setMidiPath,
        setParamView,
        setSelectionUi,
        store,
        vibratoBaseline,
        curveEditsSuppressed: s.toolMode === "block",
    });

    // Dispatch helper: context menu dialog ops → open MenuBar dialogs
    const openEditDialog = useCallback((dialog: string) => {
        window.dispatchEvent(
            new CustomEvent("hifi:openEditDialog", {
                detail: { dialog },
            }),
        );
    }, []);

    // 以选区为作用域的动作（提取/套用颤音、另存为音高参考、导出 MIDI）整块在
    // `usePianoRollSelectionActions` 里（见该文件头）。调用点留在原位：这几个回调的
    // 引用身份决定 `vibratoApplyTarget` 是否稳定。
    const { vibratoApplyTarget, handleSaveAsPitchRef, handleExportMidiFromEditor } =
        usePianoRollSelectionActions({
            currentParamRange,
            dispatch,
            editParam,
            handleEditOp,
            paramView,
            rootTrackId,
            s,
            selectAllParamRange,
            selectionRef,
            t,
        });

    // Pitch Snap 设置弹窗状态
    const [pitchSnapOpen, setPitchSnapOpen] = useState(false);

    /** 块工具来自现有工具状态；切换离开时清掉块选择，避免“看不见但还选中”。 */
    const paramBlockToolActive = s.toolMode === "block";
    const paramBlocksVisible = s.paramBlocksVisible;
    useLayoutEffect(() => {
        kernelDataRef.current.paramBlocksVisible = paramBlocksVisible;
        hostRef.current?.invalidate();
    }, [paramBlocksVisible]);
    /**
     * 显式打破 `paramBlockSpec` 的 memo 缓存。
     *
     * 拖拽快照是 ref：开始拖动写入、松手清空，二者都不会让 memo 重算。
     * 若某次渲染恰好缓存了“快照存在”的无手柄几何，清空 ref 后必须换一个
     * 依赖值才能回到正常派生。
     */
    const [paramBlockStructureTick, setParamBlockStructureTick] = useState(0);
    const bumpParamBlockStructure = useCallback(() => {
        setParamBlockStructureTick((value) => value + 1);
    }, []);
    const [paramBlockState, setParamBlockState] =
        useState<ParamBlockState>(EMPTY_PARAM_BLOCK_STATE);
    /*
     * 【几何只有一个权威】块视图是 `(clips, editSet, framePeriodMs)` 的纯函数。
     *
     * 它**不落状态**：落下来就有两个写入者（本地合成与后端回读），而回读在手势
     * 期间拿到的是旧几何 —— 屏幕上的块于是逐帧来回跳，切分 / 伸缩也读到过期边界
     * （"切分点离边缘太近"）。派生值只该被现算。
     */
    const paramBlockSourceClips = useMemo(
        () => paramBlockClips.map(toParamBlockSourceClip),
        [paramBlockClips],
    );
    const paramBlockView = useMemo<ParamBlockView | null>(
        () =>
            rootTrackId === null
                ? null
                : composeParamBlockView(
                      rootTrackId,
                      paramBlockSourceClips,
                      paramBlockState.editSet,
                      paramBlockState.framePeriodMs,
                  ),
        [
            rootTrackId,
            paramBlockSourceClips,
            paramBlockState.editSet,
            paramBlockState.framePeriodMs,
        ],
    );
    const paramBlockViewRef = useRef<ParamBlockView | null>(paramBlockView);
    useLayoutEffect(() => {
        paramBlockViewRef.current = paramBlockView;
    }, [paramBlockView]);
    /** 与派生视图同源的 clip 快照；编辑函数按它把源域区间拆开 / 合并。 */
    const paramBlockSourceClipsRef = useRef(paramBlockSourceClips);
    useLayoutEffect(() => {
        paramBlockSourceClipsRef.current = paramBlockSourceClips;
    }, [paramBlockSourceClips]);
    const [selectedParamBlockIds, setSelectedParamBlockIds] = useState<string[]>([]);
    /*
     * 块的纵向几何与拖动位移都在宿主里按实时值域计算；这里不再保留
     * “视口版本号 / 拖动位移”两份 React 状态，避免出现视觉与命中各自刷新。
     */
    /*
     * 【没有 warning 通道】块动作的前置条件（选了几个、是不是编组块、有没有
     * 可还原的区间）都是**瞬时状态**：菜单已经置灰，用户再点一下就变。为它们
     * 弹一条要读三秒的提示，等于把"当前选择还不满足条件"说成了一次错误。
     * 只有真正的失败 —— 后端拒绝、IPC 异常 —— 才进状态栏，且必须是
     * `tone: "danger"` + 后端原文 tooltip（见下方的 `publishParamBlockFeedback`）。
     */
    const [paramBlockContextMenu, setParamBlockContextMenu] = useState<{
        x: number;
        y: number;
        /**
         * 打开菜单那一刻"播放头是否落在一个可切的块内"。
         *
         * 在打开时算一次是"所见即所点"：面板不随播放头重渲染（`playheadSec`
         * 被刻意剔出选择器），渲染期算出来的禁用态会和用户点下去那一刻不一致。
         * 菜单开着时播放头再动（播放中）属于边角情形 —— 点击时会重新解析，
         * 解析不到就静默，最多差一次点击，不会误切。
         */
        canSplit: boolean;
    } | null>(null);
    const toggleParamBlocksVisible = useCallback(() => {
        dispatch(toggleParamBlocksVisibleAction());
        void dispatch(persistUiSettings());
    }, [dispatch]);
    useEffect(() => {
        // 进入块工具只需把块层从隐藏变为显示（单向），不存在级联回写。
        if (paramBlockToolActive && !paramBlocksVisible) {
            dispatch(setParamBlocksVisible(true));
        }
    }, [dispatch, paramBlockToolActive, paramBlocksVisible]);
    /** 每个 Clip 在本面板会话内的自动分析尝试（只拦自动路径，手动重分析不受影响）。 */
    const autoAnalysisAttemptsRef = useRef<Set<string>>(new Set());
    /** 上一轮"有音节区间"的 Clip 集合；用于发现分析结果被意外清空。 */
    const analyzedClipIdsRef = useRef<Set<string>>(new Set());
    const autoAnalysisQueueRef = useRef(createSyllableAnalysisQueue());
    const autoAnalysisRootRef = useRef<string | null>(null);
    useEffect(() => {
        autoAnalysisRootRef.current = paramBlocksVisible && pitchEnabled ? rootTrackId : null;
        return () => {
            if (autoAnalysisRootRef.current !== null) publishBlockAnalysisProgress(null);
            autoAnalysisRootRef.current = null;
        };
    }, [paramBlocksVisible, pitchEnabled, rootTrackId]);
    /*
     * 【安全网】某条 Clip "曾经有音节区间、现在没有" = 分析结果被某条路径清掉了
     * （撤销跨过分析完成点曾经是主要来源）。把自动分析的"已尝试"记录删掉，让它
     * 重跑一次 —— 否则块会整片消失而且永远不会自己回来。
     *
     * 后端已修好（撤销时按素材身份搬运分析），所以正常路径下这里一次都不触发；
     * 代价只在真的丢了的时候才付（一次模型推理）。
     *
     * 【为什么必须排在分析 effect 之前】effect 按声明顺序执行：先删记录、再判
     * "要不要分析"，同一次提交里就能补上。反过来的话 ref 变更不会触发渲染，
     * 分析要等到下一次无关的重渲染才会重试。
     */
    useEffect(() => {
        const current = new Set(paramBlockSyllableSpans.keys());
        for (const clipId of analyzedClipIdsRef.current) {
            if (!current.has(clipId)) autoAnalysisAttemptsRef.current.delete(clipId);
        }
        analyzedClipIdsRef.current = current;
    }, [paramBlockSyllableSpans]);
    /**
     * 时间轴内核手势锁的镜像。
     *
     * 不能在回调里直接读 `s` —— 那是渲染期快照，拖动期间不会更新；而
     * `deferReadback` 恰恰是在拖动期间被调用的。`_interactionLockCount` 覆盖
     * 移动 / 裁剪 / Slip / 淡化 / 增益 / 交叉淡化全部内核手势。
     */
    const interactionLockRef = useRef(s._interactionLockCount);
    useLayoutEffect(() => {
        interactionLockRef.current = s._interactionLockCount;
    }, [s._interactionLockCount]);
    /**
     * 历史位置镜像。块编辑的 persist 写入会打后端撤销点，但 React 的
     * `paramBlockState` 不会自动回读；只在历史深度变化时 reload，避免普通
     * 重渲染或拖动中的每次进度提交都发一轮 IPC。
     */
    const blockPersistence = useParamBlockPersistence({
        rootTrackId,
        clips: paramBlockClips,
        enabled: paramBlocksVisible || paramBlockToolActive,
        onCommitted: () => dispatch(markProjectEdited()),
        /*
         * 只跟历史位置走。clip 几何不进这个键：块视图是派生的，clip 一变就
         * 自动跟上；把它拼进来会让一次 Clip 拖动每帧发一轮回读，而那时的回包
         * 带的是旧几何 —— 闪烁就是这么来的。
         */
        historyKey: `${s.historyUndoDepth}:${s.historyRedoDepth}`,
        state: paramBlockState,
        setState: setParamBlockState,
        onError: (detail) =>
            publishParamBlockFeedback({
                text: tf("status_param_block_edit_rejected"),
                tone: "danger",
                tooltip: detail,
                holdMs: 15_000,
            }),
        /*
         * 手势期间一律不回读：参数块手势（拖动 / 过渡）与**时间轴内核手势**
         * （移动 / 裁剪 / Slip / 淡化 / 增益）都算。`_interactionLockCount` 是
         * 时间轴那条既有契约，缺了它，一次 Clip 拖动就会把旧几何的回包放进来。
         */
        deferReadback: () =>
            paramBlockDragActivityRef.current.pointerGestureToken !== null ||
            paramBlockDragActivityRef.current.pendingCommits > 0 ||
            interactionLockRef.current > 0,
    });
    /** 只提交块过渡布局；所有参数都持久化 transitionOutSec。 */
    const persistBlockTransition = useCallback(
        async (
            left: ParamBlock,
            right: ParamBlock,
            widthSec: number,
            curvePatches: readonly {
                param: string;
                startFrame: number;
                values: readonly number[];
            }[] = [],
        ): Promise<void> => {
            if (!rootTrackId) return;
            const state = paramBlockStateRef.current;
            if (paramBlockViewRef.current === null) return;
            const edit = setParamBlockTransition({
                state,
                left,
                right,
                transitionSec: widthSec,
                clips: paramBlockSourceClipsRef.current,
            });
            if (!edit.ok) throw new Error(edit.reason);
            setParamBlockState(edit.state);
            await blockPersistence.commit(edit.state, true, curvePatches);
            bumpRefreshToken();
            invalidate();
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps -- blockPersistence 是稳定 facade；只读其 commit 方法。
        [blockPersistence.commit, bumpRefreshToken, invalidate, rootTrackId],
    );

    /** Pitch 额外写曲线；非 Pitch 只走上方的布局提交。 */
    const persistPitchTransitionCurve = useCallback(
        async (
            left: ParamBlock,
            right: ParamBlock,
            widthSec: number,
            options?: { readonly restoreOriginal?: boolean },
        ): Promise<void> => {
            if (!rootTrackId || editParam !== "pitch") return;
            if (widthSec <= 1e-6 && options?.restoreOriginal !== true) return;
            const framePeriodMs = paramViewRef.current?.framePeriodMs ?? 5;
            if (options?.restoreOriginal === true) {
                await persistBlockTransition(left, right, 0);
                const boundaryFrame = Math.round((right.startSec * 1000) / framePeriodMs);
                const widthFrames = Math.max(1, Math.round((widthSec * 1000) / framePeriodMs));
                const startFrame = Math.max(0, Math.floor((left.startSec * 1000) / framePeriodMs));
                const endFrame = Math.max(
                    boundaryFrame,
                    Math.round((right.endSec * 1000) / framePeriodMs) - 1,
                );
                const restoreStart = Math.max(startFrame, boundaryFrame - widthFrames);
                const restoreEnd = Math.min(endFrame, boundaryFrame + widthFrames - 1);
                const pv = paramViewRef.current;
                const original = pv === null ? null : readPvRange(pv, restoreStart, restoreEnd);
                if (original !== null && original.length > 0) {
                    const restoreSegments = [{ startFrame: restoreStart, values: original }];
                    await uploadFullResCurveSegments({
                        trackId: rootTrackId,
                        param: "pitch",
                        segments: restoreSegments,
                    });
                    const pvNow = paramViewRef.current;
                    if (pvNow !== null) {
                        setParamView(
                            applyFullResSegmentsToParamView({
                                paramView: pvNow,
                                segments: restoreSegments,
                            }),
                        );
                    }
                    onParamCommitSucceeded();
                    bumpRefreshToken();
                    invalidate();
                }
                return;
            }

            const boundaryFrame = Math.round((right.startSec * 1000) / framePeriodMs);
            const widthFrames = Math.max(1, Math.round((widthSec * 1000) / framePeriodMs));
            const startFrame = Math.max(0, Math.floor((left.startSec * 1000) / framePeriodMs));
            const endFrame = Math.max(
                boundaryFrame,
                Math.round((right.endSec * 1000) / framePeriodMs) - 1,
            );
            const source = await fetchFullResCurve({
                trackId: rootTrackId,
                param: "pitch",
                startFrame: Math.max(startFrame, boundaryFrame - widthFrames),
                endFrame: Math.min(endFrame, boundaryFrame + widthFrames - 1),
                paramView: paramBlockViewRef.current !== null ? null : paramViewRef.current,
            });
            const leftValue = left.averageValue ?? 60;
            const rightValue = right.averageValue ?? 60;
            const committedAt = (frame: number): number =>
                Number(source.values[frame - source.startFrame]) || rightValue;
            const segments = buildParamBlockTransitionSegments({
                boundaryFrame,
                widthFrames,
                targetValue: rightValue,
                leftStartFrame: startFrame,
                rightEndFrame: endFrame,
                leftValue,
                rightValue,
                sourceAt: committedAt,
            });
            await persistBlockTransition(
                left,
                right,
                widthSec,
                segments.map((segment) => ({ ...segment, param: "pitch" })),
            );
        },
        [
            bumpRefreshToken,
            editParam,
            invalidate,
            onParamCommitSucceeded,
            paramViewRef,
            persistBlockTransition,
            rootTrackId,
            setParamView,
        ],
    );

    useEffect(() => {
        // 撤销/重做会整批替换块记录；刚提交的伸缩交接快照不能跨历史版本生效。
        paramBlockCommittedSpanRef.current = null;
    }, [rootTrackId, s.historyRedoDepth, s.historyUndoDepth]);
    /*
     * 持久化合并不在组件状态里：从工程参数读取，写入时走带撤销点的命令。
     * 只在根轨道切换 / 历史推进时刷新；拖动块本身不触发网络请求。
     */
    useEffect(() => {
        if (!rootTrackId) return;
        /*
         * 块谱面可见性只决定“是否显示 / 分配 block 编辑 UI”；音节分析本身
         * 需要“块工具打开”或“眼睛打开”任一即可。进入块工具时眼睛会同步打开，
         * 但显隐 effect 与这个分析 effect 的提交顺序不应决定是否分析。
         */
        const blocksEnabled = paramBlocksVisible && blockPersistence.ready;
        /*
         * 这里不再维护 `cancelled`：分析请求一旦发出，后端会落盘结果。
         * 若在依赖变化 / 卸载时直接丢弃 Promise，下一次挂载读到的旧快照会
         * 永远把这条 Clip 判为“未分析”。结果始终参与刷新；只以本地编辑代次
         * 防止覆盖用户刚提交的块编辑。
         */
        /*
         * 只分析**当前根轨道组**内的音频 Clip。`s.clips` 是整个工程；
         * 用它兜底会让未开启 Compose 的其它轨道提前跑 GAME。根轨道也不
         * 携带 Compose 时直接不分析：块是音高编辑的一部分，不能为一个
         * 不参与合成的轨道凭空消耗模型推理。
         */
        const audioClips =
            blocksEnabled && pitchEnabled
                ? paramBlockClips.filter(
                      (clip) =>
                          clip.trackId.length > 0 &&
                          (clip.midiNoteData?.length ?? 0) === 0 &&
                          !paramBlockSyllableSpans.has(clip.id) &&
                          Boolean(clip.sourcePath) &&
                          !autoAnalysisAttemptsRef.current.has(clip.id),
                  )
                : [];
        for (const clip of audioClips) autoAnalysisAttemptsRef.current.add(clip.id);
        if (audioClips.length === 0) return;
        void autoAnalysisQueueRef.current
            .runBatch({
                clips: audioClips,
                shouldStart: () => autoAnalysisRootRef.current === rootTrackId,
                onSkipped: (clip) => autoAnalysisAttemptsRef.current.delete(clip.id),
                onStart: (clip) =>
                    publishBlockAnalysisProgress({
                        active: true,
                        clipId: clip.id,
                        clipName: clip.name,
                        progress: 0,
                        stage: "queued",
                        algorithm: "game",
                        message: null,
                    }),
                analyze: async (clip) => {
                    const result = await timelineApi.analyzeClipSyllables(clip.id);
                    dispatch(applyClipSyllableAnalysis({ ...result, sourcePath: clip.sourcePath }));
                    // 几何是派生的：分析回填只要重读一次编辑集，块就会自己长出来。
                    if (result.rootTrackId) blockPersistence.receive(result.rootTrackId);
                    return result.analyzed && result.sourceSpansSec.length > 0;
                },
            })
            .then((results) => {
                if (autoAnalysisRootRef.current === rootTrackId) publishBlockAnalysisProgress(null);
                for (const result of results)
                    if (!result.analyzed) autoAnalysisAttemptsRef.current.delete(result.clipId);
            });
        // eslint-disable-next-line react-hooks/exhaustive-deps -- blockPersistence 是稳定 facade；依赖其具体方法。
    }, [
        bumpRefreshToken,
        dispatch,
        paramBlockClips,
        paramBlockSyllableSpans,
        paramBlocksVisible,
        pitchEnabled,
        rootTrackId,
        s.paramsEpoch,
        blockPersistence.ready,
        blockPersistence.receive,
    ]);
    const persistParamBlockEdits = useCallback(
        (state: ParamBlockState) => {
            void blockPersistence.persist(state);
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps -- blockPersistence 是稳定 facade；只读其 persist 方法。
        [blockPersistence.persist],
    );
    /*
     * 面板块 = 派生视图 + 当前参数曲线。视图随 clip 变化自动重算，所以 Clip
     * 拖动期间块会平滑跟随，不需要任何"本地重投影"或回读补丁。
     */
    const paramBlocksForPanel = useMemo(
        () => materializeParamBlocks({ rootTrackId: rootTrackId ?? "", view: paramBlockView }),
        [rootTrackId, paramBlockView],
    );
    const paramBlocksForPanelRef =
        useRef<ReturnType<typeof materializeParamBlocks>>(paramBlocksForPanel);
    useLayoutEffect(() => {
        paramBlocksForPanelRef.current = paramBlocksForPanel;
    }, [paramBlocksForPanel]);
    const paramBlocksWithValues = useMemo(() => {
        const valueBounds = getParamValueBoundsForScrollbarImpl({ processorParamsRef }, editParam);
        /*
         * 没有曲线（例如尚未取到参数样本）时，块仍然要能以中性高度显示。
         * `layoutParamBlockRects` 会跳过 `averageValue === null` 的块；旧逻辑
         * 直接返回这些块，结果“分析完成但整层不画”。取参数值域中点作为显示
         * 锚点只影响视觉位置，不写回任何曲线。
         */
        const fallbackValue =
            Number.isFinite(valueBounds.min) && Number.isFinite(valueBounds.max)
                ? (valueBounds.min + valueBounds.max) / 2
                : editParam === "pitch"
                  ? 60
                  : 0;
        return resolveParamBlockPanelValues({
            blocks: paramBlocksForPanel,
            param: editParam,
            paramView,
            fallbackValue,
        });
        // eslint-disable-next-line react-hooks/exhaustive-deps -- processorParamsRef 是稳定 ref，其 current 变化不应让整批块重建。
    }, [editParam, paramBlocksForPanel, paramView]);
    const paramBlocksWithValuesRef = useRef(paramBlocksWithValues);
    useLayoutEffect(() => {
        paramBlocksWithValuesRef.current = paramBlocksWithValues;
    }, [paramBlocksWithValues]);
    /**
     * 编组块不能做水平伸缩。
     *
     * 编组块的几何是"若干源域区间"的联合，横向改边界没有唯一解，内核一律以
     * `grouped_locked` 拒绝。既然结果是静默失败，就不该先给出可拖的手柄与
     * `ew-resize` 光标 —— 那会让用户以为拖不动是卡住了。垂直移动、过渡手柄、
     * 删除、取消编组都不受影响。
     */
    const groupedParamBlockIds = useMemo(
        () =>
            new Set(
                paramBlocksWithValues
                    .filter((block) => block.kind === "grouped")
                    .map((block) => block.id),
            ),
        [paramBlocksWithValues],
    );
    // 工具切换时清空选中；不用 effect，避免“先画一帧旧选择再清”的闪烁。
    /** 视图变化（重分析 / 切 Take / 删除）后修剪已经不存在块的选择。 */
    useEffect(() => {
        const valid = new Set(paramBlocksWithValues.map((block) => block.id));
        // eslint-disable-next-line react-hooks/set-state-in-effect -- 选择修剪是视图变化后的必要同步；仅在集合变化时返回新数组。
        setSelectedParamBlockIds((previous) => {
            const next = previous.filter((id) => valid.has(id));
            return next.length === previous.length ? previous : next;
        });
    }, [paramBlocksWithValues]);
    const activeSelectedParamBlockIds = useMemo(
        () => (paramBlockToolActive ? selectedParamBlockIds : []),
        [paramBlockToolActive, selectedParamBlockIds],
    );
    const bounds = getParamValueBoundsForScrollbar(editParam);
    const selectedParamBlockIdSet = useMemo(
        () => new Set(activeSelectedParamBlockIds),
        [activeSelectedParamBlockIds],
    );
    const paramBlockSpecForDragSnapshot = useCallback(
        (snapshot: ParamBlockDragSnapshot): PianoRollParamBlockSpec | null => {
            if (!paramBlocksVisible) return null;
            if (snapshot.blocks.length === 0) return null;
            return {
                curveOwned: paramBlockViewRef.current !== null,
                param: editParam,
                valueMin: bounds.min,
                valueMax: bounds.max,
                valueToY: (value, heightPx) => valueToY(editParam, value, heightPx),
                blocks: [
                    ...snapshot.blocks.map((block) => ({
                        id: block.id,
                        startSec: block.startSec,
                        endSec: block.endSec,
                        averageValue: block.anchorValue,
                        frameCenterValue: snapshot.centerValues.get(block.id) ?? null,
                        displayCenterSnapToSemitone: false,
                        selected: true,
                        grouped: block.grouped,
                    })),
                    ...paramBlocksWithValues
                        .filter((block) => !snapshot.ids.has(block.id))
                        .map((block) => ({
                            id: block.id,
                            startSec: block.startSec,
                            endSec: block.endSec,
                            averageValue: block.averageValue,
                            logicalCenterValue: block.manualValue,
                            displayCenterSnapToSemitone: block.manualValue == null,
                            selected: false,
                            grouped: block.kind === "grouped",
                        })),
                ],
                borderRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(124,198,236,0.95)" : "rgba(31,103,160,0.95)",
                    ),
                ),
                // 过渡曲线与手柄同色系，和 OpenUtau 的音高轨迹一致。
                transitionRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(219,246,255,0.92)" : "rgba(0,94,156,0.90)",
                    ),
                ),
                transitionPointRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(255,255,255,1)" : "rgba(0,66,112,1)",
                    ),
                ),
                transitionPointRadiusPx: 2.5,
                selectedRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(210,242,255,1)" : "rgba(0,72,128,1)",
                    ),
                ),
                groupedRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(124,198,236,0.95)" : "rgba(31,103,160,0.95)",
                    ),
                ),
                labelRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(244,251,255,0.92)" : "rgba(235,247,255,0.92)",
                    ),
                ),
                selectedLabelRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(255,255,255,1)" : "rgba(255,255,255,1)",
                    ),
                ),
                fillRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(57,111,145,0.90)" : "rgba(107,158,192,0.84)",
                    ),
                ),
                selectedFillRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(49,132,180,0.98)" : "rgba(54,132,184,0.96)",
                    ),
                ),
                groupedFillRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(57,111,145,0.90)" : "rgba(107,158,192,0.84)",
                    ),
                ),
                topHighlightRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(255,255,255,0.11)" : "rgba(255,255,255,0.20)",
                    ),
                ),
                cornerRadiusPx: 2,
                topHighlightPx: 1,
            };
        },
        [
            bounds.max,
            bounds.min,
            editParam,
            paramBlocksVisible,
            paramBlocksWithValues,
            themeMode,
            valueToY,
        ],
    );
    paramBlockSpecForDragSnapshotRef.current = paramBlockSpecForDragSnapshot;
    const beginBlockDrag = useCallback(
        (args: ParamBlockDragStart) => {
            const anchorBlock = paramBlocksWithValues.find(
                (block) => block.id === args.anchorBlockId,
            );
            if (paramBlockResizeRef.current?.blockId === "__create__") {
                paramBlockResizeRef.current = null;
                paramBlockCreateCommitRef.current = null;
                if (paramBlockCreateReleaseRafRef.current !== null)
                    cancelAnimationFrame(paramBlockCreateReleaseRafRef.current);
                paramBlockCreateReleaseRafRef.current = null;
            }
            return beginParamBlockDrag(args, {
                paramView,
                rootTrackId,
                editParam,
                paramBlocksWithValues,
                paramBlockDragActivityRef,
                paramBlockGestureSerialRef,
                paramBlockActiveGestureTokenRef,
                paramBlockDragCommitRef,
                paramBlockDragCommitReleaseRafRef,
                paramBlockInteractionCurveRef,
                paramBlockDragSnapshotRef,
                paramBlockStateRef,
                curveOwned: paramBlockViewRef.current !== null,
                paramBlockCommitQueueRef,
                paramViewRef,
                liveEditActiveRef,
                applyDenseToLiveEdit,
                resetLiveEditPreview,
                invalidateParamBlockSpec,
                requestWaveformRepaint,
                bumpParamBlockStructure,
                setParamBlockState,
                /*
                 * 曲线补丁与块布局走**同一个事务**：后端每次 `set` 都打一个撤销点，
                 * 分成两次写就会让一次拖拽需要按两次撤销。`state` 是本次手势算好的
                 * 最终编辑集（含 Pitch 块的逻辑中心）。
                 */
                commitLineCurve: async (state, segments, revision, geometry, signature) => {
                    await blockPersistence.commit(
                        state,
                        true,
                        segments.map((segment) => ({ ...segment, param: editParam })),
                        revision,
                        geometry,
                        signature,
                    );
                },
                getAuthorityWatermark,
                paramAuthorityRevision,
                onParamCommitSucceeded,
                bumpRefreshToken,
                paintNow: () => hostRef.current?.paintNow(),
                pitchSnapEnabled: s.pitchSnapEnabled,
                onError: (error) => {
                    publishParamBlockFeedback({
                        text: tf("status_param_block_action_failed"),
                        tone: "danger",
                        tooltip: backendDetail(error) ?? String(error),
                        holdMs: 15_000,
                    });
                },
                snapToggleHeldAt: (event) => {
                    const binding = firstBinding(mergedKeybindings["modifier.clipNoSnap"]);
                    return (
                        binding !== undefined &&
                        !isNoneBinding(binding) &&
                        isModifierActive(binding, event)
                    );
                },
                resolveValueDelta: (startValue, rawDelta, snapToggleActive) => {
                    if (editParam === "pitch") {
                        return resolveParamBlockPitchSnapDelta({
                            startValue,
                            rawDelta,
                            pitchSnapEnabled: s.pitchSnapEnabled,
                            pitchSnapUnit: s.pitchSnapUnit,
                            pitchSnapToleranceCents: s.pitchSnapToleranceCents,
                            projectScale: effectiveProjectScale,
                            scaleAtSec: (sec) => projectScaleAtSec(sec) ?? effectiveProjectScale,
                            anchorSec: anchorBlock?.startSec ?? 0,
                            toggleModifierActive: snapToggleActive,
                        });
                    }
                    return rawDelta;
                },
            });
        },
        [
            applyDenseToLiveEdit,
            bumpRefreshToken,
            bumpParamBlockStructure,
            editParam,
            blockPersistence.commit,
            getAuthorityWatermark,
            invalidateParamBlockSpec,
            liveEditActiveRef,
            mergedKeybindings,
            onParamCommitSucceeded,
            paramBlocksWithValues,
            paramView,
            paramAuthorityRevision,
            requestWaveformRepaint,
            resetLiveEditPreview,
            rootTrackId,
            setParamBlockState,
            paramViewRef,
        ],
    );

    const onParamBlockPointerDown = useCallback(
        (hit: {
            sec: number;
            localY: number;
            clientX: number;
            clientY: number;
            contextMenu: boolean;
            add: boolean;
            toggle: boolean;
            pointerType: string;
            pointerId: number;
            captureTarget: HTMLCanvasElement;
            setActivePointerGestureEnd: (endGesture: () => void) => void;
            clearActivePointerGestureEnd: (endGesture?: () => void) => void;
            pointerValue: (clientY: number) => number;
            altKey?: boolean;
            shiftKey?: boolean;
            ctrlKey?: boolean;
            metaKey?: boolean;
        }): import("../gestures/pointerDeps").ParamBlockPointerDownResult => {
            if (!paramBlockToolActive) return { handled: false };
            /*
             * 块层未就绪 = 此刻做不了任何块操作。返回 `handled: false`，让指针
             * 落进块工具的空白分支（按设置移动播放头）—— 用户看到的是"点了一下、
             * 播放头动了"，而不是一条要读三秒的提示。
             */
            if (!blockPersistence.ready || !paramViewRef.current) return { handled: false };
            if (hit.contextMenu) {
                const layout = paramBlockViewRef.current;
                if (!layout) return { handled: true, seekOnClick: false };
                paramBlockClickCandidateRef.current = null;
                setParamBlockContextMenu(null);
                const fp = layout.framePeriodMs;
                const originalSelection = selectionRef.current;
                const originalIds = selectedParamBlockIds;
                let dragging = false;
                let lastX = hit.clientX;
                let raf = 0;
                const setRegion = (region: ReturnType<typeof selectionFromPointers>) => {
                    selectionRef.current = region;
                    setSelectionUi(region);
                    setSelectedParamBlockIds(paramBlockIdsInSelection(layout.blocks, region));
                    invalidate();
                };
                const update = (clientX: number) => {
                    lastX = clientX;
                    if (!dragging && Math.abs(clientX - hit.clientX) < 4) return;
                    dragging = true;
                    const rect = hit.captureTarget.getBoundingClientRect();
                    const axis = hostRef.current?.getAxis();
                    const sec = axis
                        ? (clientX - rect.left + axis.scrollLeftPx) / axis.pxPerSec
                        : hit.sec;
                    setRegion(selectionFromPointers((hit.sec * 1000) / fp, (sec * 1000) / fp));
                };
                const edgeScroll = () => {
                    if (dragging) {
                        const rect = hit.captureTarget.getBoundingClientRect();
                        const axis = hostRef.current?.getAxis();
                        const dx = lastX < rect.left + 24 ? -12 : lastX > rect.right - 24 ? 12 : 0;
                        if (dx && axis) {
                            applyEdgeScrollLeft(axis.scrollLeftPx + dx);
                            update(lastX);
                        }
                    }
                    raf = requestAnimationFrame(edgeScroll);
                };
                raf = requestAnimationFrame(edgeScroll);
                startPointerGesture({
                    pointerId: hit.pointerId,
                    captureTarget: hit.captureTarget,
                    onMove: (event) => {
                        if ((event.buttons & 2) === 2) update(event.clientX);
                    },
                    onEnd: (reason, event) => {
                        cancelAnimationFrame(raf);
                        armRightDragContextMenuGuard();
                        if (reason !== "up") {
                            selectionRef.current = originalSelection;
                            setSelectionUi(originalSelection);
                            setSelectedParamBlockIds(originalIds);
                            return;
                        }
                        if (event) update(event.clientX);
                        if (dragging) return;
                        const frame = (hit.sec * 1000) / fp;
                        if (rangeIndexAtFrame(selectionRef.current, frame) < 0) {
                            const block = layout.blocks.find(
                                (block) =>
                                    frame >= block.startFrame && frame < block.endFrameExclusive,
                            );
                            if (block) {
                                if (!originalIds.includes(block.id)) {
                                    setRegion(
                                        selectionFromFrames(
                                            block.startFrame,
                                            block.endFrameExclusive - block.startFrame,
                                        ),
                                    );
                                }
                            } else setRegion(null);
                        }
                        setParamBlockContextMenu({
                            x: event?.clientX ?? hit.clientX,
                            y: event?.clientY ?? hit.clientY,
                            /*
                             * 禁用态在**打开时**算一次：面板不随播放头重渲染，
                             * 渲染期算出来的会和用户点下去那一刻不一致。
                             */
                            canSplit:
                                rootTrackId !== null &&
                                resolveSplittableParamBlockAt(
                                    blocksFromViewForPanel(rootTrackId, layout),
                                    store.getState().session.playheadSec,
                                    layout.framePeriodMs,
                                ) !== null,
                        });
                    },
                    registerEnd: hit.setActivePointerGestureEnd,
                    clearEnd: hit.clearActivePointerGestureEnd,
                });
                return { handled: true, seekOnClick: false };
            }

            const framedHits = sortParamBlocksByTime(
                hostRef.current?.getParamBlockHitRects() ?? [],
            );
            const noSnapKeyHeld = (event: {
                ctrlKey: boolean;
                shiftKey: boolean;
                altKey: boolean;
                metaKey?: boolean;
            }): boolean => {
                const noSnapKb = firstBinding(mergedKeybindings["modifier.clipNoSnap"]);
                return (
                    noSnapKb !== undefined &&
                    !isNoneBinding(noSnapKb) &&
                    isModifierActive(noSnapKb, event)
                );
            };
            const hitRect = hitTestParamBlocksAtPoint({
                blocks: framedHits,
                sec: hit.sec,
                localY: hit.localY,
                // 触控笔 / 触摸的指针更容易漂 2~3px；鼠标保持精确命中，避免点空白误选。
                slopPx:
                    hit.pointerType === "mouse"
                        ? Math.min(1, paramBlockHitSlopPx)
                        : paramBlockHitSlopPx,
            });
            /*
             * 手柄优先于块体：它画在右边界外侧，已经离开块矩形。若先做块体
             * 命中，手柄区域会被当成空白，后面的手柄手势永远进不去。
             * 手柄可见时即可拖，不要求块已经选中。
             */
            const axisForHandle = hostRef.current?.getAxis();
            const touchedHandle =
                axisForHandle === undefined
                    ? null
                    : hitTestParamBlockTransitionHandles({
                          blocks: framedHits,
                          clientX: hit.clientX - hit.captureTarget.getBoundingClientRect().left,
                          localY: hit.localY,
                          secToX: (sec) =>
                              sec * axisForHandle.pxPerSec - axisForHandle.scrollLeftPx,
                          outsetPx: PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX,
                          outsetYPx: PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX,
                      });
            const handleEdgeHit =
                axisForHandle === undefined
                    ? null
                    : hitTestParamBlockEdges({
                          // 编组块不提供水平伸缩（见 `groupedParamBlockIds`）。
                          blocks: framedHits.filter((rect) => !groupedParamBlockIds.has(rect.id)),
                          clientX: hit.clientX - hit.captureTarget.getBoundingClientRect().left,
                          localY: hit.localY,
                          secToX: (sec) =>
                              sec * axisForHandle.pxPerSec - axisForHandle.scrollLeftPx,
                          hitPx: 8,
                      });
            /*
             * 手柄圆点与左侧块的右边界共线，横向命中区必然覆盖那条伸缩热区。
             * 因此不能因为“命中了左块右边界”就放弃手柄；只有指针落在**另一块**
             * （通常是右邻居）的左边界附近，才应优先让边缘伸缩接管。
             */
            const handleStealsEdge =
                handleEdgeHit !== null &&
                touchedHandle !== null &&
                handleEdgeHit.block.id !== touchedHandle.leftId;
            /*
             * 手柄在右边界正上方；当它离右块的左边界很近时，用户真正想抓的
             * 可能是左边界。边缘（左/右边界）在这些位置优先：手柄只在
             * 不落在任何边缘热区内时才接管。
             */
            if (touchedHandle !== null && !handleStealsEdge && !hit.contextMenu) {
                const leftBlock = paramBlocksWithValues.find(
                    (block) => block.id === touchedHandle.leftId,
                );
                const rightBlock = paramBlocksWithValues.find(
                    (block) => block.id === touchedHandle.rightId,
                );
                if (leftBlock !== undefined && rightBlock !== undefined) {
                    hit.captureTarget.setPointerCapture?.(hit.pointerId);
                    const startX = hit.clientX;
                    const startWidth = Math.max(0, touchedHandle.widthSec);
                    let targetWidth = startWidth;
                    let activated = false;
                    let lastClientX = startX;
                    const transitionFramePeriodMs = paramViewRef.current?.framePeriodMs ?? 5;
                    /*
                     * 手柄拖动只改过渡曲线，不改两个块的高度。
                     * 手势开始时冻结两块代表值与行高，防止 live 主簇变化
                     * 让块跟随新重心上下漂移；松手后由正式派生值接替。
                     */
                    const lockedLeftCenter = leftBlock.averageValue;
                    const lockedRightCenter = rightBlock.averageValue;
                    const lockedHeightPx =
                        hostRef.current
                            ?.getParamBlockHitRects()
                            .find((rect) => rect.id === leftBlock.id)?.heightPx ?? null;
                    const transitionBoundaryFrame = Math.round(
                        (rightBlock.startSec * 1000) / transitionFramePeriodMs,
                    );
                    const transitionStartBase = (() => {
                        const pv = paramViewRef.current;
                        if (pv === null) return null;
                        ensureLiveEditBase(pv);
                        // 过渡以相接点为对称中心：冻结两侧最大宽度内的原曲线。
                        const leftMaxFrames = Math.max(
                            1,
                            Math.ceil(
                                ((leftBlock.endSec - leftBlock.startSec) * 1000) /
                                    transitionFramePeriodMs,
                            ),
                        );
                        const rightMaxFrames = Math.max(
                            1,
                            Math.ceil(
                                ((rightBlock.endSec - rightBlock.startSec) * 1000) /
                                    transitionFramePeriodMs,
                            ),
                        );
                        const readStart = Math.max(0, transitionBoundaryFrame - leftMaxFrames);
                        const readEnd = Math.min(
                            Math.round((rightBlock.endSec * 1000) / transitionFramePeriodMs) - 1,
                            transitionBoundaryFrame + rightMaxFrames,
                        );
                        return {
                            pv,
                            base: readPvRange(pv, readStart, readEnd),
                            readStart,
                        };
                    })();
                    const applyTransitionPreview = (widthSec: number) => {
                        const frozen = transitionStartBase;
                        const pv = paramViewRef.current;
                        if (frozen === null || pv === null) return;
                        const source = frozen.pv;
                        const widthFrames = Math.max(
                            1,
                            Math.round((widthSec * 1000) / transitionFramePeriodMs),
                        );
                        const segments = buildParamBlockTransitionPreview({
                            boundaryFrame: transitionBoundaryFrame,
                            widthFrames,
                            targetValue: rightBlock.averageValue ?? leftBlock.averageValue ?? 60,
                            leftStartFrame: Math.max(
                                0,
                                Math.floor((leftBlock.startSec * 1000) / transitionFramePeriodMs),
                            ),
                            rightEndFrame:
                                Math.round((rightBlock.endSec * 1000) / transitionFramePeriodMs) -
                                1,
                            leftValue: leftBlock.averageValue ?? 60,
                            rightValue: rightBlock.averageValue ?? 60,
                            paramView: {
                                startFrame: frozen.readStart,
                                stride: source.stride,
                                edit: frozen.base,
                            },
                        });
                        for (const segment of segments) {
                            applyDenseToLiveEdit(
                                pv,
                                segment.startFrame,
                                segment.values,
                                pv.startFrame,
                                segment.startFrame + segment.values.length - 1,
                                "draw",
                            );
                        }
                        requestWaveformRepaint();
                    };
                    const transitionGesture = startPointerGesture({
                        pointerId: hit.pointerId,
                        captureTarget: hit.captureTarget,
                        onMove: (event) => {
                            if ((event.buttons & 1) !== 1) {
                                transitionGesture.end();
                                return;
                            }
                            const dx = event.clientX - startX;
                            lastClientX = event.clientX;
                            if (!activated) {
                                if (Math.abs(dx) < 3) return;
                                activated = true;
                            }
                            const perSec = pxPerSecRef.current;
                            targetWidth = Math.max(
                                0,
                                Math.min(
                                    (rightBlock.endSec - rightBlock.startSec) / 2,
                                    startWidth - dx / Math.max(1, perSec),
                                ),
                            );
                            const widthSpec = kernelDataRef.current.paramBlocks ?? paramBlockSpec;
                            if (widthSpec !== null) {
                                kernelDataRef.current.paramBlocks = {
                                    ...widthSpec,
                                    blocks: widthSpec.blocks.map((block) =>
                                        block.id === touchedHandle.leftId ||
                                        block.id === touchedHandle.rightId
                                            ? {
                                                  ...block,
                                                  displayCenterValue:
                                                      block.id === touchedHandle.leftId
                                                          ? lockedLeftCenter
                                                          : lockedRightCenter,
                                                  displayHeightPx: lockedHeightPx,
                                              }
                                            : block,
                                    ),
                                };
                            }
                            applyTransitionPreview(targetWidth);
                            /*
                             * 每次先写冻结块高 / 中心，再把新手柄宽度叠到同一份
                             * 规格上；从 ref 重读会在 React 重绘的间隙拿到旧手柄，
                             * 拉动时就会在两个宽度之间闪。
                             */
                            const liveSpec = kernelDataRef.current.paramBlocks ?? widthSpec;
                            if (liveSpec !== null) {
                                kernelDataRef.current.paramBlocks = {
                                    ...liveSpec,
                                    blocks: liveSpec.blocks.map((block) =>
                                        block.id === touchedHandle.leftId
                                            ? {
                                                  ...block,
                                                  transitionHandle: {
                                                      ...touchedHandle,
                                                      widthSec: targetWidth,
                                                  },
                                              }
                                            : block,
                                    ),
                                };
                            }
                            invalidate();
                        },
                        onEnd: (reason) => {
                            if (reason === "up" && !activated) {
                                const dx = lastClientX - startX;
                                if (Math.abs(dx) >= 3) {
                                    activated = true;
                                    const perSec = pxPerSecRef.current;
                                    targetWidth = Math.max(
                                        0,
                                        Math.min(
                                            (rightBlock.endSec - rightBlock.startSec) / 2,
                                            startWidth - dx / Math.max(1, perSec),
                                        ),
                                    );
                                }
                            }
                            if (reason === "up" && activated) {
                                paramBlockDragSnapshotRef.current = null;
                                invalidateParamBlockSpec();
                                const transitionCommit =
                                    editParam === "pitch"
                                        ? targetWidth <= 1e-6
                                            ? persistPitchTransitionCurve(
                                                  leftBlock,
                                                  rightBlock,
                                                  0,
                                                  { restoreOriginal: true },
                                              )
                                            : persistPitchTransitionCurve(
                                                  leftBlock,
                                                  rightBlock,
                                                  targetWidth,
                                              )
                                        : persistBlockTransition(
                                              leftBlock,
                                              rightBlock,
                                              targetWidth,
                                          );
                                void transitionCommit.catch((error) => {
                                    console.error("[pianoRoll] transition commit failed", error);
                                    publishParamBlockFeedback({
                                        text: tf("status_param_block_action_failed"),
                                        tone: "danger",
                                        tooltip: backendDetail(error) ?? String(error),
                                        holdMs: 15_000,
                                    });
                                    const pvNow = paramViewRef.current;
                                    if (pvNow !== null) resetLiveEditPreview(pvNow);
                                    bumpRefreshToken();
                                    invalidate();
                                });
                            } else {
                                const pvNow = paramViewRef.current;
                                if (pvNow !== null) resetLiveEditPreview(pvNow);
                            }
                            invalidate();
                            clearParamBlockSnapHighlight();
                        },
                        registerEnd: hit.setActivePointerGestureEnd,
                        clearEnd: hit.clearActivePointerGestureEnd,
                    });
                    return { handled: true, seekOnClick: false };
                }
            }
            /*
             * 宿主首帧尚未产出块几何时（工具刚切换 / 数据刚到），不能用空矩形
             * 集合当成“点在空白”。改为按时间就近命中，保证第一次点击就能选中，
             * 而不是要点两次。
             */
            const fallbackBlock =
                hitRect === null && framedHits.length === 0
                    ? nearestParamBlockByTime(
                          paramBlocksWithValues.map((block) => ({
                              id: block.id,
                              startSec: block.startSec,
                              endSec: block.endSec,
                          })),
                          hit.sec,
                          paramBlockHitSlopPx / Math.max(1, pxPerSecRef.current),
                      )
                    : null;
            const hitId = hitRect?.id ?? fallbackBlock?.id ?? null;
            if (hitId === null) {
                selectionRef.current = null;
                setSelectionUi(null);
                setSelectedParamBlockIds([]);
                /*
                 * 在空白处按住并横向拖 = 新建块。它必须发生在块工具内，
                 * 且达到最小宽度后才提交；短点击仍然只是清空选择。
                 */
                if (paramBlockToolActive && hit.add) {
                    const flattenCreatedPitch = editParam === "pitch" && hit.altKey === true;
                    const createStartSec = hit.sec;
                    const createStartClientX = hit.clientX;
                    const createStartClientY = hit.clientY;
                    const createStartValue = hit.pointerValue(createStartClientY);
                    const snapCreateResult = (sec: number, toggle: boolean) =>
                        resolveParamBlockTimeSnapResult({
                            rawSec: sec,
                            settings: s.timelineSnap,
                            grid: s.project.gridSize,
                            bpm: s.bpm,
                            beatsPerBar: Math.max(1, Math.round(s.beats || 4)),
                            tempoMap: s.tempoMap,
                            pxPerSec: hostRef.current?.getAxis()?.pxPerSec ?? pxPerSecRef.current,
                            clips: paramBlockClips,
                            tracks: s.tracks,
                            selectedClipIds: s.multiSelectedClipIds,
                            playheadSec: s.playheadSec,
                            originSec: createStartSec,
                            toggleModifierActive: toggle,
                        });
                    const snapCreateAt = (sec: number, toggle: boolean): number => {
                        const result = snapCreateResult(sec, toggle);
                        publishParamBlockSnapHighlight(result, sec);
                        return result.sec;
                    };
                    const noSnapKb = firstBinding(mergedKeybindings["modifier.clipNoSnap"]);
                    const createNoSnap = (event: {
                        shiftKey: boolean;
                        altKey: boolean;
                        ctrlKey: boolean;
                        metaKey: boolean;
                    }) =>
                        noSnapKb !== undefined &&
                        !isNoneBinding(noSnapKb) &&
                        isModifierActive(noSnapKb, event);
                    const createSnapToggle = createNoSnap({
                        shiftKey: hit.shiftKey === true,
                        altKey: hit.altKey === true,
                        ctrlKey: hit.ctrlKey === true,
                        metaKey: hit.metaKey === true,
                    });
                    const createEffectiveSnap = s.timelineSnap.enabled !== createSnapToggle;
                    let createSpan = resolveParamBlockCreateSpan({
                        framePeriodMs: paramBlockStateRef.current.framePeriodMs,
                        anchorSec: createStartSec,
                        pointerSec: createStartSec,
                        snapToGrid: (sec) => snapCreateAt(sec, false),
                        snapEnabled: createEffectiveSnap,
                        minBlockSec: paramBlockStateRef.current.framePeriodMs / 1000,
                    });
                    let createCommitted = false;
                    let createActivated = false;
                    let preserveCreatePreview = false;
                    const createId = newParamBlockId();
                    const createToken = ++paramBlockGestureSerialRef.current;
                    const previousCreateHandoff = paramBlockCreateCommitRef.current;
                    const previousCreatePreview = paramBlockResizeRef.current;
                    const previousInteractionCurve = paramBlockInteractionCurveRef.current;
                    if (paramBlockCreateReleaseRafRef.current !== null) {
                        cancelAnimationFrame(paramBlockCreateReleaseRafRef.current);
                        paramBlockCreateReleaseRafRef.current = null;
                    }
                    paramBlockCreateCommitRef.current = null;
                    paramBlockDragActivityRef.current.pointerGestureToken = createToken;
                    // The new gesture takes over the previous visible curve.
                    paramBlockActiveGestureTokenRef.current = null;
                    paramBlockDragSnapshotRef.current = null;
                    paramBlockDragCommitRef.current = null;
                    const ownsCreatePreview = () =>
                        paramBlockResizeRef.current?.createdBlockId === createId;
                    let createValue = createStartValue;
                    const resolveCreateBlockValue = (value: number): number =>
                        s.paramBlockFollowPitch && editParam === "pitch"
                            ? getDrawPreviewValue({
                                  editParam: "pitch",
                                  rawValue: value,
                                  effectiveSnap: true,
                                  pitchSnapUnit: s.pitchSnapUnit,
                                  projectScale:
                                      s.pitchSnapUnit === "scale"
                                          ? (projectScaleAtSec(createSpan.startSec) ??
                                            effectiveProjectScale)
                                          : undefined,
                                  pitchSnapToleranceCents: s.pitchSnapToleranceCents,
                              })
                            : value;
                    const createLayoutBasis = paramBlockViewRef.current;
                    const createBaseSpec = kernelDataRef.current.paramBlocks ?? paramBlockSpec;
                    const createCurveBase = captureParamBlockCreateCurve(
                        paramViewRef.current,
                        kernelDataRef.current.paramEditorFrame?.view?.key ===
                            paramViewRef.current?.key
                            ? (kernelDataRef.current.paramEditorFrame?.view ??
                                  paramBlockInteractionCurveRef.current)
                            : paramBlockInteractionCurveRef.current,
                    );
                    const createBaseBlocks =
                        createLayoutBasis && rootTrackId
                            ? resolveParamBlockPanelValues({
                                  blocks: blocksFromViewForPanel(rootTrackId, createLayoutBasis),
                                  param: editParam,
                                  paramView: createCurveBase,
                                  fallbackValue: editParam === "pitch" ? 60 : 0,
                              }).map((block) => ({ ...block, selected: false, grouped: false }))
                            : (createBaseSpec?.blocks ?? []);
                    const createStableValues = new Map(
                        createBaseBlocks
                            .filter(
                                (block) =>
                                    block.averageValue != null &&
                                    Number.isFinite(block.averageValue),
                            )
                            .map((block) => [
                                block.id,
                                {
                                    startSec: block.startSec,
                                    endSec: block.endSec,
                                    value: block.averageValue!,
                                },
                            ]),
                    );
                    for (const block of hostRef.current?.getParamBlockHitRects() ?? []) {
                        createStableValues.set(block.id, {
                            startSec: block.startSec,
                            endSec: block.endSec,
                            value: block.averageValue,
                        });
                    }
                    const createAuthorityRevisionAtStart = paramAuthorityRevision;
                    const createAuthorityRequestSeqAtCommit =
                        getAuthorityWatermark().issuedRequestSeq;
                    const updateCreatePreview = (event: {
                        clientX: number;
                        clientY: number;
                        ctrlKey: boolean;
                        shiftKey: boolean;
                        altKey: boolean;
                        metaKey: boolean;
                    }) => {
                        const noSnap = createNoSnap(event);
                        const axis = hostRef.current?.getAxis();
                        if (axis === undefined) return;
                        const canvasRect = hit.captureTarget.getBoundingClientRect();
                        const pointerSec = viewportPxToSec(axis, event.clientX - canvasRect.left);
                        createSpan = resolveParamBlockCreateSpan({
                            framePeriodMs: paramBlockStateRef.current.framePeriodMs,
                            anchorSec: createStartSec,
                            pointerSec,
                            snapToGrid: (sec) => snapCreateAt(sec, false),
                            snapEnabled: s.timelineSnap.enabled !== noSnap,
                            minBlockSec: paramBlockStateRef.current.framePeriodMs / 1000,
                        });
                        const axisValue = yToValue(
                            editParam,
                            event.clientY - canvasRect.top,
                            canvasRect.height,
                        );
                        createValue = resolveCreatedParamBlockValue({
                            pointerValue: editParam === "pitch" ? axisValue - 0.5 : axisValue,
                            param: editParam,
                            pitchSnapEnabled: s.pitchSnapEnabled,
                            pitchSnapUnit: s.pitchSnapUnit,
                            pitchSnapToleranceCents: s.pitchSnapToleranceCents,
                            projectScale:
                                projectScaleAtSec(createSpan.startSec) ?? effectiveProjectScale,
                            toggleModifierActive: noSnap,
                        });
                        const createBlockValue = resolveCreateBlockValue(createValue);
                        paramBlockResizeRef.current = {
                            blockId: "__create__",
                            edge: "right",
                            pointerClientX: createStartClientX,
                            startSec: createSpan.startSec,
                            endSec: createSpan.endSec,
                            neighborStartSec: null,
                            neighborEndSec: null,
                            resizeCandidates: [],
                            baseBlockSpans: [],
                            createBaseBlocks,
                            createdBlockId: createId,
                            resizePlan: null,
                            targetSec: createSpan.endSec,
                            value: createBlockValue,
                        };
                        if (createBaseSpec !== null) {
                            const preview = projectParamBlockCreatePreviewFromBase({
                                framePeriodMs: paramBlockStateRef.current.framePeriodMs,
                                baseBlocks: createBaseBlocks,
                                startSec: createSpan.startSec,
                                endSec: createSpan.endSec,
                                createId,
                                value: createBlockValue,
                            });
                            kernelDataRef.current.paramBlocks = {
                                ...createBaseSpec,
                                blocks: preview.map((block) => ({
                                    ...block,
                                    grouped: false,
                                    selected: block.id === createId,
                                    frameCenterValue:
                                        block.id === createId ? createBlockValue : undefined,
                                    displayCenterValue:
                                        block.id === createId
                                            ? createBlockValue
                                            : (block.displayCenterValue ?? null),
                                })),
                            };
                        }
                        if (createCurveBase !== null) {
                            const fp = Math.max(1e-6, createCurveBase.framePeriodMs);
                            const stride = Math.max(1, createCurveBase.stride);
                            const startFrame = Math.max(
                                0,
                                (createLayoutBasis ? Math.round : Math.floor)(
                                    (createSpan.startSec * 1000) / fp,
                                ),
                            );
                            const endFrame = Math.max(
                                startFrame,
                                (createLayoutBasis ? Math.round : Math.ceil)(
                                    (createSpan.endSec * 1000) / fp,
                                ) - 1,
                            );
                            const source: number[] = [];
                            for (let frame = startFrame; frame <= endFrame; frame += 1) {
                                const index = Math.round(
                                    (frame - createCurveBase.startFrame) / stride,
                                );
                                const value = createCurveBase.edit[index];
                                source.push(
                                    value !== undefined && Number.isFinite(value) ? value : 0,
                                );
                            }
                            resetLiveEditPreview(createCurveBase);
                            const shiftedValues = shiftParamBlockCurveValuesToTarget({
                                values: source,
                                target: resolveCreateBlockValue(createValue),
                                param: editParam,
                                flatten: flattenCreatedPitch,
                                curveOwned: paramBlockViewRef.current !== null,
                            });
                            applyDenseToLiveEdit(
                                createCurveBase,
                                startFrame,
                                shiftedValues,
                                createCurveBase.startFrame,
                                endFrame,
                                "draw",
                            );
                            paramBlockInteractionCurveRef.current = {
                                ...createCurveBase,
                                preview: {
                                    base: createCurveBase,
                                    patches: [{ startFrame, values: shiftedValues }],
                                    affectedRanges: [{ startFrame, frameCount: source.length }],
                                    blockValues: createStableValues,
                                },
                                edit: applyFullResSegmentsToParamView({
                                    paramView: createCurveBase,
                                    segments: [{ startFrame, values: shiftedValues }],
                                }).edit,
                            };
                            requestWaveformRepaint();
                        }
                        hostRef.current?.paintNow();
                        invalidate();
                    };
                    if (liveEditActiveRef) liveEditActiveRef.current = true;
                    startPointerGesture({
                        pointerId: hit.pointerId,
                        captureTarget: hit.captureTarget,
                        onMove: (event) => {
                            if ((event.buttons & 1) !== 1) return;
                            const dx = event.clientX - createStartClientX;
                            const dy = event.clientY - createStartClientY;
                            if (!createActivated) {
                                if (Math.abs(dx) < 4 && Math.abs(dy) < 4) return;
                                createActivated = true;
                            }
                            updateCreatePreview(event);
                        },
                        onEnd: (reason, event) => {
                            const activity = paramBlockDragActivityRef.current;
                            if (activity.pointerGestureToken === createToken)
                                activity.pointerGestureToken = null;
                            if (reason === "up" && event !== null) {
                                const dx = event.clientX - createStartClientX;
                                const dy = event.clientY - createStartClientY;
                                if (!createActivated && Math.hypot(dx, dy) >= 4) {
                                    createActivated = true;
                                }
                                if (createActivated) updateCreatePreview(event);
                            }
                            if (reason === "up" && createActivated && !createCommitted) {
                                createCommitted = true;
                                const clip =
                                    clipAtParamBlockCreateSpan({
                                        clips: paramBlockSourceClipsRef.current,
                                        startSec: createSpan.startSec,
                                        endSec: createSpan.endSec,
                                    }) ?? undefined;
                                if (clip !== undefined || paramBlockViewRef.current !== null) {
                                    const created = createParamBlock({
                                        state: paramBlockStateRef.current,
                                        clip,
                                        clips: paramBlockSourceClipsRef.current,
                                        blockId: createId,
                                        startSec: createSpan.startSec,
                                        endSec: createSpan.endSec,
                                        value: resolveCreateBlockValue(createValue),
                                        existingBlocks: paramBlocksWithValues,
                                    });
                                    if (created.ok) {
                                        setParamBlockState(created.state);
                                        paramBlockStateRef.current = created.state;
                                    } else applyParamBlockEditRef.current?.(created);
                                    preserveCreatePreview = created.ok;
                                    const createHandoff = created.ok
                                        ? {
                                              blockId: createId,
                                              authorityRevisionAtStart:
                                                  createAuthorityRevisionAtStart,
                                              authorityRequestSeqAtCommit:
                                                  createAuthorityRequestSeqAtCommit,
                                              requiresCurveAuthority: rootTrackId !== null,
                                              committed: rootTrackId === null,
                                              committedSegments: undefined as
                                                  | readonly {
                                                        startFrame: number;
                                                        values: readonly number[];
                                                    }[]
                                                  | undefined,
                                          }
                                        : null;
                                    if (createHandoff !== null) {
                                        paramBlockCreateCommitRef.current = createHandoff;
                                    }
                                    if (created.ok && rootTrackId !== null) {
                                        const curveBase = createCurveBase ?? paramViewRef.current;
                                        void enqueueParamBlockCreateCommit({
                                            queue: paramBlockCommitQueueRef,
                                            activity: paramBlockDragActivityRef,
                                            liveActive: liveEditActiveRef,
                                            onSettled: () => {
                                                blockPersistence.flushReadback();
                                                bumpRefreshToken();
                                                bumpParamBlockStructure();
                                                invalidate();
                                            },
                                            work: async () => {
                                                if (createHandoff === null) return;
                                                try {
                                                    if (curveBase === null) {
                                                        createHandoff.committed = true;
                                                        createHandoff.requiresCurveAuthority = false;
                                                        if (ownsCreatePreview()) {
                                                            paramBlockCreateCommitRef.current =
                                                                null;
                                                            paramBlockResizeRef.current = null;
                                                            paramBlockInteractionCurveRef.current =
                                                                null;
                                                        }
                                                        return;
                                                    }
                                                    const fp = Math.max(
                                                        1e-6,
                                                        curveBase.framePeriodMs,
                                                    );
                                                    const startFrame = Math.max(
                                                        0,
                                                        Math.floor(
                                                            (createSpan.startSec * 1000) / fp,
                                                        ),
                                                    );
                                                    const endFrame = Math.max(
                                                        startFrame,
                                                        (createLayoutBasis
                                                            ? Math.round
                                                            : Math.ceil)(
                                                            (createSpan.endSec * 1000) / fp,
                                                        ) - 1,
                                                    );
                                                    const source = await fetchFullResCurve({
                                                        trackId: rootTrackId,
                                                        param: editParam,
                                                        startFrame,
                                                        endFrame,
                                                        paramView:
                                                            createLayoutBasis !== null
                                                                ? null
                                                                : curveBase,
                                                    });
                                                    const segments = [
                                                        {
                                                            startFrame: source.startFrame,
                                                            values: shiftParamBlockCurveValuesToTarget(
                                                                {
                                                                    values: source.values,
                                                                    target: resolveCreateBlockValue(
                                                                        createValue,
                                                                    ),
                                                                    param: editParam,
                                                                    flatten: flattenCreatedPitch,
                                                                    curveOwned:
                                                                        createLayoutBasis !== null,
                                                                },
                                                            ),
                                                        },
                                                    ];
                                                    if (createLayoutBasis !== null) {
                                                        await blockPersistence.commit(
                                                            created.state,
                                                            true,
                                                            segments.map((segment) => ({
                                                                ...segment,
                                                                param: editParam,
                                                            })),
                                                            source.authorityRevision,
                                                            source.geometryRevision,
                                                            source.geometrySignature,
                                                        );
                                                    } else {
                                                        await uploadFullResCurveSegments({
                                                            trackId: rootTrackId,
                                                            param: editParam,
                                                            segments,
                                                        });
                                                    }
                                                    createHandoff.committedSegments = segments;
                                                    createHandoff.authorityRequestSeqAtCommit =
                                                        getAuthorityWatermark().issuedRequestSeq;
                                                    createHandoff.committed = true;
                                                    onParamCommitSucceeded();
                                                } catch (error) {
                                                    console.error(
                                                        "[pianoRoll] create block curve commit failed",
                                                        error,
                                                    );
                                                    publishParamBlockFeedback({
                                                        text: tf(
                                                            "status_param_block_action_failed",
                                                        ),
                                                        tone: "danger",
                                                        tooltip:
                                                            backendDetail(error) ?? String(error),
                                                        holdMs: 15_000,
                                                    });
                                                    createHandoff.committed = true;
                                                    createHandoff.requiresCurveAuthority = false;
                                                    if (ownsCreatePreview()) {
                                                        paramBlockCreateCommitRef.current = null;
                                                        paramBlockResizeRef.current = null;
                                                        paramBlockInteractionCurveRef.current =
                                                            null;
                                                        const pvNow = paramViewRef.current;
                                                        if (pvNow !== null)
                                                            resetLiveEditPreview(pvNow);
                                                    }
                                                }
                                            },
                                        }).catch((error) => {
                                            console.error("[pianoRoll] create queue failed", error);
                                            publishParamBlockFeedback({
                                                text: tf("status_param_block_action_failed"),
                                                tone: "danger",
                                                tooltip: backendDetail(error) ?? String(error),
                                                holdMs: 15_000,
                                            });
                                        });
                                    }
                                    setSelectedParamBlockIds([createId]);
                                    if (!created.ok || rootTrackId === null) {
                                        if (!created.ok) {
                                            const pvNow = paramViewRef.current;
                                            if (pvNow !== null) resetLiveEditPreview(pvNow);
                                        }
                                        liveEditActiveRef.current =
                                            activity.pendingCommits > 0 ||
                                            activity.pointerGestureToken !== null;
                                    }
                                } else {
                                    const pvNow = paramViewRef.current;
                                    if (pvNow !== null) resetLiveEditPreview(pvNow);
                                    if (liveEditActiveRef)
                                        liveEditActiveRef.current =
                                            activity.pendingCommits > 0 ||
                                            activity.pointerGestureToken !== null;
                                }
                            } else {
                                const pvNow = paramViewRef.current;
                                if (pvNow !== null) resetLiveEditPreview(pvNow);
                                if (liveEditActiveRef)
                                    liveEditActiveRef.current =
                                        activity.pendingCommits > 0 ||
                                        activity.pointerGestureToken !== null;
                            }
                            if (!preserveCreatePreview) {
                                paramBlockCreateCommitRef.current = previousCreateHandoff;
                                paramBlockResizeRef.current = previousCreatePreview;
                                paramBlockInteractionCurveRef.current = previousInteractionCurve;
                                const pvNow = paramViewRef.current;
                                if (pvNow) resetLiveEditPreview(pvNow);
                                blockPersistence.flushReadback();
                            }
                            bumpParamBlockStructure();
                            invalidate();
                            clearParamBlockSnapHighlight();
                        },
                        registerEnd: hit.setActivePointerGestureEnd,
                        clearEnd: hit.clearActivePointerGestureEnd,
                    });
                    return { handled: true, seekOnClick: false };
                }
                selectionRef.current = null;
                setSelectionUi(null);
                setSelectedParamBlockIds([]);
                setParamBlockContextMenu(null);
                invalidate();
                return { handled: false };
            }
            const fullBlock = paramBlocksWithValues.find((block) => block.id === hitId);
            if (fullBlock === undefined) return { handled: false };
            if (hit.contextMenu) {
                paramBlockClickCandidateRef.current = null;
            }
            /* 命中优先级：手柄已在上方入口处理；这里只剩边缘缩放 > 块体。 */
            const axis = hostRef.current?.getAxis();
            const edgeHit =
                axis === undefined || groupedParamBlockIds.has(fullBlock.id)
                    ? null
                    : hitTestParamBlockEdges({
                          blocks: framedHits.filter((rect) => rect.id === fullBlock.id),
                          clientX: hit.clientX - hit.captureTarget.getBoundingClientRect().left,
                          localY: hit.localY,
                          secToX: (sec) => sec * axis.pxPerSec - axis.scrollLeftPx,
                          hitPx: 8,
                      });
            if (edgeHit !== null) {
                const resizeEdge = edgeHit.edge;
                /*
                 * 一次伸缩可能跨过空隙、连续覆盖多个块。候选集按拖动方向
                 * 由近到远排列，最终哪些块缩短 / 删除由共享计划决定。
                 */
                const resizeCandidates = paramBlockResizeCandidates(
                    paramBlocksWithValues,
                    fullBlock,
                    resizeEdge,
                );
                const resizePlan = resolveParamBlockResizePlan({
                    blockId: fullBlock.id,
                    edge: resizeEdge,
                    blockStartSec: fullBlock.startSec,
                    blockEndSec: fullBlock.endSec,
                    targetSec: resizeEdge === "left" ? fullBlock.startSec : fullBlock.endSec,
                    neighbors: resizeCandidates,
                    minBlockSec: paramBlockStateRef.current.framePeriodMs / 1000,
                });
                const gestureRef = {
                    blockId: fullBlock.id,
                    edge: resizeEdge,
                    pointerClientX: hit.clientX,
                    startSec: fullBlock.startSec,
                    endSec: fullBlock.endSec,
                    neighborStartSec: resizeCandidates[0]?.startSec ?? null,
                    neighborEndSec: resizeCandidates[0]?.endSec ?? null,
                    resizeCandidates,
                    baseBlockSpans: hostRef.current?.getParamBlockHitRects().map((rect) => ({
                        id: rect.id,
                        startSec: rect.startSec,
                        endSec: rect.endSec,
                    })) ?? [
                        {
                            id: fullBlock.id,
                            startSec: fullBlock.startSec,
                            endSec: fullBlock.endSec,
                        },
                        ...resizeCandidates.map((block) => ({
                            id: block.id,
                            startSec: block.startSec,
                            endSec: block.endSec,
                        })),
                    ],
                    resizePlan,
                    framePeriodMsBasis: paramBlockStateRef.current.framePeriodMs,
                    displayCenterValue: fullBlock.averageValue,
                    // 块高按当前行高冻结（1.2 × row），A snapshot 的 heightPx 在
                    // spec 里是视口坐标，可直接沿用；缺省时由 kernel 按本帧行高重算。
                    displayHeightPx:
                        hostRef.current
                            ?.getParamBlockHitRects()
                            .find((rect) => rect.id === fullBlock.id)?.heightPx ?? null,
                    targetSec:
                        resizeEdge === "left" ? resizePlan.block.startSec : resizePlan.block.endSec,
                } as NonNullable<typeof paramBlockResizeRef.current>;
                paramBlockResizeRef.current = gestureRef;
                let activateResize = false;
                const resizeGesture = startPointerGesture({
                    pointerId: hit.pointerId,
                    captureTarget: hit.captureTarget,
                    onMove: (event) => {
                        if ((event.buttons & 1) !== 1) {
                            resizeGesture.end();
                            return;
                        }
                        const dx = event.clientX - gestureRef.pointerClientX;
                        if (!activateResize) {
                            if (Math.abs(dx) < 3 && event.pointerType === "mouse") {
                                return;
                            }
                            activateResize = true;
                        }
                        /*
                         * 边缘必须严格跟随指针的**像素位移**。用「指针工程秒 −
                         * 按下工程秒」在横向缩放或视口刚滚动时会引入锚点误差；
                         * px / pxPerSec 只依赖 pointerdown 时冻结的 pxPerSec。
                         */
                        const targetSec =
                            (gestureRef.edge === "left" ? gestureRef.startSec : gestureRef.endSec) +
                            dx / Math.max(1e-9, pxPerSecRef.current);
                        const minBlockSec = paramBlockStateRef.current.framePeriodMs / 1000;
                        const snappedTarget = resolveParamBlockResizeTarget({
                            edge: gestureRef.edge,
                            startSec: gestureRef.startSec,
                            endSec: gestureRef.endSec,
                            rawTargetSec: targetSec,
                            unclampedTargetSec: targetSec,
                            snapToGrid: (sec) => {
                                const result = resolveParamBlockTimeSnapResult({
                                    rawSec: sec,
                                    settings: s.timelineSnap,
                                    grid: s.project.gridSize,
                                    bpm: s.bpm,
                                    beatsPerBar: Math.max(1, Math.round(s.beats || 4)),
                                    tempoMap: s.tempoMap,
                                    pxPerSec:
                                        hostRef.current?.getAxis()?.pxPerSec ?? pxPerSecRef.current,
                                    clips: paramBlockClips,
                                    tracks: s.tracks,
                                    selectedClipIds: s.multiSelectedClipIds,
                                    playheadSec: s.playheadSec,
                                    originSec: gestureRef.startSec,
                                    toggleModifierActive: noSnapKeyHeld(event),
                                });
                                publishParamBlockSnapHighlight(result, sec);
                                return result.sec;
                            },
                            minBlockSec,
                            neighborStartSec: gestureRef.neighborStartSec,
                            neighborEndSec: gestureRef.neighborEndSec,
                        });
                        gestureRef.resizePlan = resolveParamBlockResizePlan({
                            blockId: gestureRef.blockId,
                            edge: gestureRef.edge,
                            blockStartSec: gestureRef.startSec,
                            blockEndSec: gestureRef.endSec,
                            targetSec: snappedTarget,
                            neighbors: gestureRef.resizeCandidates,
                            minBlockSec,
                        });
                        gestureRef.targetSec =
                            gestureRef.edge === "left"
                                ? gestureRef.resizePlan.block.startSec
                                : gestureRef.resizePlan.block.endSec;
                        /*
                         * 拖动中直接刷新宿主块规格。`invalidate()` 只重绘现有
                         * GL 实例，不会重算 React memo；边缘必须在 onMove 里
                         * 立刻写新几何，否则要等松手后的状态提交才动。
                         */
                        const liveSpec = kernelDataRef.current.paramBlocks ?? paramBlockSpec;
                        if (liveSpec !== null) {
                            const plan = gestureRef.resizePlan;
                            if (plan === null) return;
                            /*
                             * 每帧都从**手势起点**重新投影；不能从上一帧 live
                             * 列表继续，否则块一旦被判定删除，后续再拉回来时
                             * 已经没有源块，预览会永远缺失到松手。
                             */
                            const displayBlocks = projectParamBlockResizeDisplay({
                                baseBlocks: gestureRef.baseBlockSpans,
                                blockId: gestureRef.blockId,
                                plan,
                            });
                            const displayById = new Map(
                                displayBlocks.map((block) => [block.id, block]),
                            );
                            kernelDataRef.current.paramBlocks = {
                                ...liveSpec,
                                blocks: gestureRef.baseBlockSpans.flatMap((base) => {
                                    const shown = displayById.get(base.id);
                                    if (shown === undefined) return [];
                                    const original =
                                        liveSpec.blocks.find((block) => block.id === base.id) ??
                                        latestParamBlockSpecRef.current?.blocks.find(
                                            (block) => block.id === base.id,
                                        );
                                    return [
                                        {
                                            ...(original ?? {
                                                id: base.id,
                                                averageValue: null,
                                                selected: false,
                                                grouped: false,
                                            }),
                                            startSec: shown.startSec,
                                            endSec: shown.endSec,
                                            selected:
                                                shown.selected ||
                                                (original?.selected === true &&
                                                    base.id !== gestureRef.blockId),
                                        },
                                    ];
                                }),
                            };
                        }
                        invalidate();
                    },
                    onEnd: (reason) => {
                        const latest = paramBlocksWithValues.find(
                            (block) => block.id === gestureRef.blockId,
                        );
                        if (reason === "up" && activateResize && latest !== undefined) {
                            const minBlockSec =
                                gestureRef.framePeriodMsBasis !== undefined
                                    ? gestureRef.framePeriodMsBasis / 1000
                                    : Math.max(0.04, 2 / Math.max(1, pxPerSecRef.current));
                            const plan = gestureRef.resizePlan;
                            const removeSelf = plan?.removeBlock === true;
                            if (!removeSelf && plan !== null) {
                                paramBlockCommittedSpanRef.current = {
                                    blockId: latest.id,
                                    blockSpan: plan.block,
                                    neighbors: plan.neighbors,
                                    expiresAt: performance.now() + 800,
                                };
                            }
                            applyParamBlockEditRef.current?.(
                                removeSelf
                                    ? removeParamBlockAtMinWidth({
                                          state: paramBlockStateRef.current,
                                          blocks: paramBlocksWithValues,
                                          blockId: latest.id,
                                          clips: paramBlockSourceClipsRef.current,
                                      })
                                    : resizeParamBlock({
                                          state: paramBlockStateRef.current,
                                          block: latest,
                                          edge: gestureRef.edge,
                                          targetSec: gestureRef.targetSec,
                                          clips: paramBlockSourceClipsRef.current,
                                          targetSpan: plan?.block ?? null,
                                          neighborSpans: plan?.neighbors,
                                          neighborBlocks: gestureRef.resizeCandidates,
                                          minBlockSec,
                                      }),
                            );
                            if (removeSelf) {
                                setSelectedParamBlockIds((ids) =>
                                    ids.filter((id) => id !== latest.id),
                                );
                            }
                        }
                        paramBlockResizeRef.current = null;
                        bumpParamBlockStructure();
                        invalidate();
                        clearParamBlockSnapHighlight();
                    },
                    registerEnd: hit.setActivePointerGestureEnd,
                    clearEnd: hit.clearActivePointerGestureEnd,
                });
                return { handled: true, seekOnClick: false };
            }
            /*
             * 拖动开始前先同步“本次参与拖动”的选中集合。点击空白外时
             * React 选中态会在拖动手势已经建立之后才提交；把锚点加入
             * 局部集合，保证第一次按住就能拖动，且不牵动别的块。
             */
            const dragSelection = hit.toggle
                ? selectedParamBlockIds.includes(fullBlock.id)
                    ? selectedParamBlockIds.filter((id) => id !== fullBlock.id)
                    : [...selectedParamBlockIds, fullBlock.id]
                : hit.add
                  ? selectedParamBlockIds.includes(fullBlock.id)
                      ? selectedParamBlockIds
                      : [...selectedParamBlockIds, fullBlock.id]
                  : [fullBlock.id];
            setParamBlockContextMenu(null);
            setSelectedParamBlockIds(dragSelection);
            bumpParamBlockStructure();
            if (!hit.contextMenu) {
                paramBlockDoubleClickTokenRef.current += 1;
                paramBlockClickCandidateRef.current = {
                    blockId: fullBlock.id,
                    token: `block:${paramBlockDoubleClickTokenRef.current}`,
                    at: performance.now(),
                };
            }
            /*
             * Alt + 上下拖 = 块内音高缩放。Alt 是参数编辑器已有的参数
             * 修饰键（拉伸/精细调整都是它），但块工具下它只作用在块身上：
             * 不会进入块平移，也不改变块的起止时间。
             */
            const pitchScaleGesture = editParam === "pitch" && hit.altKey === true;
            /*
             * 用**这一次按下已经命中的块**开始拖动，而不是等 React 把选中状态
             * 提交后再回头找。否则首次按住块时 selected 还是上一帧的空集合，
             * 手势会直接返回，用户必须“先点一下再拖”才能移动。
             */
            beginBlockDrag({
                anchorBlockId: fullBlock.id,
                clientX: hit.clientX,
                clientY: hit.clientY,
                contextMenu: hit.contextMenu,
                pointerId: hit.pointerId,
                captureTarget: hit.captureTarget,
                setActivePointerGestureEnd: hit.setActivePointerGestureEnd,
                clearActivePointerGestureEnd: hit.clearActivePointerGestureEnd,
                pointerValue: hit.pointerValue,
                selection: dragSelection,
                anchorValue: framedHits.find((block) => block.id === hitId)?.averageValue ?? null,
                centerValues: new Map(
                    framedHits.flatMap((block) =>
                        block.averageValue == null ? [] : [[block.id, block.averageValue] as const],
                    ),
                ),
                pitchScale: pitchScaleGesture,
                shiftKey: hit.shiftKey,
                altKey: hit.altKey,
            });
            return { handled: true, seekOnClick: true };
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps -- paramBlockSpec 作为事件期几何读取；加入依赖会让手势在预览帧重建。
        [
            editParam,
            beginBlockDrag,
            bumpParamBlockStructure,
            groupedParamBlockIds,
            invalidate,
            paramBlockClips,
            paramBlockToolActive,
            blockPersistence.ready,
            paramBlocksWithValues,
            rootTrackId,
            pxPerSecRef,
            selectedParamBlockIds,
            store,
            s.bpm,
            s.paramBlockFollowPitch,
            s.project.gridSize,
            s.project.timeSignatureDenominator,
            s.timelineSnap.enabled,
            blockPersistence,
            effectiveProjectScale,
            projectScaleAtSec,
            s.pitchSnapEnabled,
            s.pitchSnapToleranceCents,
            s.pitchSnapUnit,
            tf,
        ],
    );
    const groupSelectedParamBlocks = useCallback(() => {
        const result = groupParamBlocks({
            state: paramBlockState,
            blocks: paramBlocksWithValues,
            selectedBlockIds: selectedParamBlockIds,
            clips: paramBlockSourceClips,
        });
        // 前置条件不满足 = 静默 no-op（与时间轴的 `groupClips` 同一约定）。
        // 菜单项已经置灰；走到这里只可能是"菜单打开后状态变了"。
        if (!result.ok) return;
        setParamBlockState(result.state);
        persistParamBlockEdits(result.state);
        setSelectedParamBlockIds([]);
        setParamBlockContextMenu(null);
    }, [
        paramBlockSourceClips,
        paramBlockState,
        paramBlocksWithValues,
        persistParamBlockEdits,
        selectedParamBlockIds,
    ]);
    groupSelectedParamBlocksRef.current = groupSelectedParamBlocks;
    const ungroupSelectedParamBlocks = useCallback(() => {
        const selectedIds = [...selectedParamBlockIdSet].filter((id) =>
            paramBlocksWithValues.some((block) => block.id === id && block.kind === "grouped"),
        );
        // 没有编组块可拆 = 静默；菜单项已经置灰。
        if (selectedIds.length === 0) return;
        try {
            const next = ungroupParamBlocks(paramBlockState, selectedIds);
            setParamBlockState(next);
            persistParamBlockEdits(next);
            setSelectedParamBlockIds([]);
            setParamBlockContextMenu(null);
        } catch (error) {
            warnDev("[pianoRoll] ungroup param blocks failed: " + String(error));
        }
    }, [paramBlockState, paramBlocksWithValues, persistParamBlockEdits, selectedParamBlockIdSet]);
    ungroupSelectedParamBlocksRef.current = ungroupSelectedParamBlocks;
    const deleteSelectedParamBlocks = useCallback(() => {
        const result = deleteParamBlocks({
            state: paramBlockState,
            blocks: paramBlocksWithValues,
            selectedBlockIds: selectedParamBlockIds,
            clips: paramBlockSourceClips,
        });
        // 没选中块 / 块不可删 = 静默；菜单项已经置灰。
        if (!result.ok) return;
        setParamBlockState(result.state);
        persistParamBlockEdits(result.state);
        setSelectedParamBlockIds([]);
        setParamBlockContextMenu(null);
    }, [
        paramBlockSourceClips,
        paramBlockState,
        paramBlocksWithValues,
        persistParamBlockEdits,
        selectedParamBlockIds,
    ]);
    deleteSelectedParamBlocksRef.current = deleteSelectedParamBlocks;
    const restoreParamBlocks = useCallback(
        (blockIds?: readonly string[]) => {
            const result = restoreDeletedParamBlocks({
                state: paramBlockState,
                selectedBlockIds: blockIds,
            });
            // 没有可恢复的块 = 静默；菜单项已经置灰。
            if (!result.ok) return;
            setParamBlockState(result.state);
            persistParamBlockEdits(result.state);
            setSelectedParamBlockIds([]);
            setParamBlockContextMenu(null);
        },
        [paramBlockState, persistParamBlockEdits],
    );
    restoreParamBlocksRef.current = restoreParamBlocks;
    const applyParamBlockEdit = useCallback(
        (result: ReturnType<typeof splitParamBlock>) => {
            /*
             * 能走到这里说明前置条件已经通过（目标块由调用方按位置解析），
             * 失败属于实现层问题：开发构建打印，生产静默。
             */
            if (!result.ok) {
                warnDev("[pianoRoll] param block edit failed: " + result.reason);
                return;
            }
            setParamBlockState(result.state);
            persistParamBlockEdits(result.state);
            setParamBlockContextMenu(null);
        },
        [persistParamBlockEdits],
    );
    applyParamBlockEditRef.current = applyParamBlockEdit;
    paramBlockStateRef.current = paramBlockState;
    /**
     * 在播放头处切开。
     *
     * 【操作数由位置决定】块在时间上互不重叠，播放头最多落在一个块里 —— "切哪个块"
     * 天然唯一。要求先选中一个块是多余的一步，还会在 Slip 之后拿旧几何当判据
     * （"切分点离边缘太近"正是这么来的）。与时间轴的 `resolveSplitTargetsAtSec`
     * 同一条思路：位置解析优先，解析不到就静默。
     *
     * 【前置条件不满足 = 静默】没有块 / 编组块 / 切点贴边：没有提示、没有 IPC、
     * 不改选择。只有后端真的拒绝才进状态栏。
     */
    const splitParamBlockAtPlayhead = useCallback(
        (snapToggleHeld = false) => {
            if (rootTrackId === null) return;
            const view = paramBlockViewRef.current;
            if (view === null) return;
            /*
             * 播放头必须读 store：面板选择器刻意剔除了 `playheadSec`
             * （见 `PIANO_ROLL_VOLATILE_SESSION_KEYS`），`s.playheadSec` 是滞后快照。
             */
            const rawSec = store.getState().session.playheadSec;
            const state = paramBlockStateRef.current;
            const framePeriodMs = state.framePeriodMs;
            const clips = paramBlockSourceClipsRef.current;
            const blocks = blocksFromViewForPanel(rootTrackId, view);
            // 播放头不在任何可切的块内（含编组块）：静默。
            const target = resolveSplittableParamBlockAt(blocks, rawSec, framePeriodMs);
            if (target === null) return;

            const pxPerSec = hostRef.current?.getAxis()?.pxPerSec ?? pxPerSecRef.current;
            const snap = resolveParamBlockTimeSnapResult({
                rawSec,
                settings: s.timelineSnap,
                grid: s.project.gridSize,
                bpm: s.bpm,
                beatsPerBar: Math.max(1, Math.round(s.beats || 4)),
                tempoMap: s.tempoMap,
                pxPerSec,
                clips: paramBlockClips,
                tracks: s.tracks,
                selectedClipIds: s.multiSelectedClipIds,
                playheadSec: rawSec,
                originSec: target.startSec,
                toggleModifierActive: snapToggleHeld,
            });
            const snapped = resolveParamBlockSplitSec({
                sec: rawSec,
                snapSec: () => snap.sec,
                enabled: s.timelineSnap.enabled !== snapToggleHeld,
                pxPerSec,
                thresholdPx: 12,
            });
            /*
             * 磁吸候选里有相邻块的边界：吸上去会正好落在边界（切出 0 长块），
             * 再往外一点甚至会吸进别的块。刀口一旦离开目标块就退回播放头原值 ——
             * 播放头在块内就一定能切，不会出现"按了没反应"。
             */
            const splitSec = isStrictlyInsideParamBlock(target, snapped, framePeriodMs)
                ? snapped
                : rawSec;
            applyParamBlockEdit(splitParamBlock({ state, block: target, sec: splitSec, clips }));
        },
        [
            applyParamBlockEdit,
            paramBlockClips,
            rootTrackId,
            s.bpm,
            s.beats,
            s.multiSelectedClipIds,
            s.project.gridSize,
            s.timelineSnap,
            s.tempoMap,
            s.tracks,
            store,
        ],
    );
    splitParamBlockAtPlayheadRef.current = splitParamBlockAtPlayhead;
    /** 打开菜单时算一次：播放头处是否有可切的块。 */
    const canSplitParamBlockAtPlayhead = paramBlockContextMenu?.canSplit ?? false;
    const gameRestoreRanges = paramBlockRestoreRanges(
        paramBlockView?.blocks ?? [],
        selectedParamBlockIds,
        selectionUi,
    );
    const canRestoreGameParamBlocks = gameRestoreRanges.length > 0;
    const restoreGameParamBlocks = useCallback(() => {
        const state = paramBlockStateRef.current;
        const view = paramBlockViewRef.current;
        if (view === null || !rootTrackId) return;
        const ranges = paramBlockRestoreRanges(
            view.blocks,
            selectedParamBlockIds,
            selectionRef.current,
        );
        if (ranges.length === 0) return;
        const first = ranges[0];
        const last = ranges[ranges.length - 1];
        void timelineApi
            .restoreGameParamBlocks({
                rootTrackId,
                startFrame: first.startFrame,
                endFrameExclusive: last.startFrame + last.frameCount,
                checkpoint: true,
            })
            .then((reply) => {
                const next: ParamBlockState = {
                    ...state,
                    rootTrackId,
                    editSet: reply.edits,
                    framePeriodMs: reply.view.framePeriodMs,
                };
                setParamBlockState(next);
                setSelectedParamBlockIds(paramBlockIdsInSelection(reply.view.blocks, first));
                dispatch(markProjectEdited());
            })
            .catch((error) => {
                console.error("[pianoRoll] restore blocks failed", error);
                publishParamBlockFeedback({
                    text: tf("status_param_block_action_failed"),
                    tone: "danger",
                    tooltip: backendDetail(error) ?? String(error),
                    holdMs: 15_000,
                });
            });
        setParamBlockContextMenu(null);
    }, [dispatch, rootTrackId, selectedParamBlockIds, selectionRef, tf]);
    restoreGameParamBlocksRef.current = restoreGameParamBlocks;

    /**
     * 把一段音高整体平移到最近钢琴键（半音），保留颤音 / 滑音轮廓。
     * 写回全分辨率帧，撤销点由 `uploadFullResCurveSegments` 的首块提交建立。
     */
    const normalizeParamBlockPitch = useCallback(
        async (block: ParamBlock): Promise<void> => {
            if (!rootTrackId || editParam !== "pitch") return;
            try {
                const framePeriodMs = paramView?.framePeriodMs ?? 5;
                const startFrame = Math.max(0, Math.ceil((block.startSec * 1000) / framePeriodMs));
                const endFrame = Math.max(
                    startFrame,
                    Math.ceil((block.endSec * 1000) / framePeriodMs) - 1,
                );
                const curve = await fetchFullResCurve({
                    trackId: rootTrackId,
                    param: "pitch",
                    startFrame,
                    endFrame,
                    paramView,
                });
                const normalized = normalizePitchToNearestSemitone({
                    values: curve.values,
                    modeCenter: block.averageValue,
                });
                if (normalized.every((value, index) => value === curve.values[index])) return;
                await uploadFullResCurveSegments({
                    trackId: rootTrackId,
                    param: "pitch",
                    segments: [{ startFrame: curve.startFrame, values: normalized }],
                });
                bumpRefreshToken();
            } catch (error) {
                console.error("[pianoRoll] normalize block pitch failed", error);
                publishParamBlockFeedback({
                    text: tf("status_param_block_action_failed"),
                    tone: "danger",
                    tooltip: backendDetail(error) ?? String(error),
                    holdMs: 15_000,
                });
            }
        },
        [bumpRefreshToken, editParam, paramView, rootTrackId, tf],
    );
    const normalizeParamBlockPitchRef = useRef(normalizeParamBlockPitch);
    normalizeParamBlockPitchRef.current = normalizeParamBlockPitch;
    normalizeSelectedParamBlockPitchRef.current = () => {
        if (selectedParamBlockIds.length !== 1) return;
        const block = paramBlocksWithValues.find((item) => item.id === selectedParamBlockIds[0]);
        if (block !== undefined) void normalizeParamBlockPitch(block);
    };
    /**
     * 双击块 = 把整块音高归一到最近半音。切分保留右键菜单 / S 键。
     *
     * 块工具下 pointerdown 会接管指针，但左键不能调用 preventDefault，
     * 否则兼容 mouse 事件被抑制、dblclick 永远不派发。
     */
    const onCanvasDoubleClick = useCallback(
        (event: React.MouseEvent<HTMLCanvasElement>) => {
            if (!paramBlockToolActive) return;
            const rect = event.currentTarget.getBoundingClientRect();
            const axis = hostRef.current?.getAxis();
            if (axis === undefined) return;
            const localX = event.clientX - rect.left;
            const localY = event.clientY - rect.top;
            /*
             * 双击手柄 = 恢复原始状态：
             * 1. 删除这条相接关系的 transitionSec 覆盖，块回到默认硬切换；
             * 2. 从当前相接点向两侧写回**提交态原曲线**，撤掉手柄扭曲。
             *
             * 必须先判手柄、再判块，否则手柄位于块顶上方，会被块体双击
             * 分支漏掉（或误当成“音高归一化”）。
             */
            const transitionHandle = hitTestParamBlockTransitionHandles({
                blocks: sortParamBlocksByTime(hostRef.current?.getParamBlockHitRects() ?? []),
                clientX: localX,
                localY,
                secToX: (value) => value * axis.pxPerSec - axis.scrollLeftPx,
                outsetPx: PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX,
                outsetYPx: PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX,
            });
            if (transitionHandle !== null) {
                const leftBlock = paramBlocksWithValues.find(
                    (item) => item.id === transitionHandle.leftId,
                );
                const rightBlock = paramBlocksWithValues.find(
                    (item) => item.id === transitionHandle.rightId,
                );
                if (leftBlock !== undefined && rightBlock !== undefined) {
                    event.preventDefault();
                    paramBlockClickCandidateRef.current = null;
                    // 块层没有根轨道时没有可持久化的编辑集：只改本地状态。
                    if (paramBlockViewRef.current === null)
                        applyParamBlockEditRef.current?.(
                            setParamBlockTransition({
                                state: paramBlockStateRef.current,
                                left: leftBlock,
                                right: rightBlock,
                                transitionSec: 0,
                                clips: paramBlockSourceClipsRef.current,
                            }),
                        );
                    /*
                     * 过渡曲线不是音频数据，撤销它需要写回提交态原值；
                     * 宽度 0 的提交函数会同时清空 live 覆盖并按帧恢复
                     * 相接点两侧的曲线。
                     */
                    if (editParam === "pitch") {
                        void persistPitchTransitionCurve(leftBlock, rightBlock, 0, {
                            restoreOriginal: true,
                        });
                    } else {
                        void persistBlockTransition(leftBlock, rightBlock, 0);
                    }
                    invalidateParamBlockSpec();
                    return;
                }
            }
            const sec = viewportPxToSec(axis, localX);
            /*
             * 双击事件在块工具里可能被 pointerdown 的“选中 + 捕获”抢先处理；
             * 这里同时接受“本次指针下的实时矩形命中”和“刚刚按下的块候选”，
             * 后者只按稳定 block id 找到当前派生块，绝不使用过期的几何。
             */
            const framedHits = sortParamBlocksByTime(
                hostRef.current?.getParamBlockHitRects() ?? [],
            );
            const hit = hitTestParamBlocks({
                blocks: framedHits,
                sec,
                localY,
                slopPx: paramBlockHitSlopPx,
            });
            const candidate = paramBlockClickCandidateRef.current;
            const candidateBlock =
                candidate === null || performance.now() - candidate.at > 700
                    ? null
                    : (paramBlocksWithValues.find((item) => item.id === candidate.blockId) ?? null);
            const candidateHit =
                candidateBlock !== null &&
                sec >= candidateBlock.startSec &&
                sec <= candidateBlock.endSec
                    ? candidateBlock
                    : null;
            const blockId = hit?.id ?? candidateHit?.id ?? null;
            if (blockId === null) return;
            const block = paramBlocksWithValues.find((item) => item.id === blockId);
            if (block === undefined) return;
            paramBlockClickCandidateRef.current = null;
            void normalizeParamBlockPitchRef.current(block);
            event.preventDefault();
        },
        [
            hostRef,
            invalidateParamBlockSpec,
            paramBlockHitSlopPx,
            paramBlockToolActive,
            paramBlocksWithValues,
            persistBlockTransition,
            persistPitchTransitionCurve,
        ],
    );
    const onCanvasPointerMoveForBlocks = useCallback(
        (event: React.PointerEvent<HTMLCanvasElement>) => {
            if (!paramBlockToolActive) return false;
            if (event.buttons !== 0) {
                // 手势期间保持拖动光标；由 pointerup 的默认光标接替。
                setCanvasCursor(
                    paramBlockResizeRef.current?.blockId === "__create__"
                        ? "crosshair"
                        : "grabbing",
                );
                return true;
            }
            const rect = event.currentTarget.getBoundingClientRect();
            const localX = event.clientX - rect.left;
            const localY = event.clientY - rect.top;
            const axis = hostRef.current?.getAxis();
            if (axis === undefined) return false;
            const sec = viewportPxToSec(axis, localX);
            const framedHits = hostRef.current?.getParamBlockHitRects() ?? [];
            const secToX = (value: number) => value * axis.pxPerSec - axis.scrollLeftPx;
            const transition = hitTestParamBlockTransitionHandles({
                // 手柄可见时即可抓住；首次 hover 不依赖块已选中。
                blocks: framedHits,
                clientX: localX,
                localY,
                secToX,
                outsetPx: PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX,
                outsetYPx: PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX,
            });
            if (transition !== null) {
                setCanvasCursor(paramBlockCursorForHover("transition"));
                hoveredParamBlockIdRef.current = transition.leftId;
                hoveredTransitionIdRef.current = transition.leftId;
                return true;
            }
            const hit = hitTestParamBlocks({
                blocks: framedHits,
                sec,
                localY,
                slopPx: paramBlockHitSlopPx,
            });
            if (hit === null) return false;
            const block = paramBlocksWithValues.find((item) => item.id === hit.id);
            if (block === undefined) return false;
            // 编组块只有块体光标（`grab`），没有左右伸缩光标。
            const edge = groupedParamBlockIds.has(hit.id)
                ? null
                : hitTestParamBlockEdges({
                      blocks: framedHits,
                      clientX: localX,
                      localY,
                      secToX,
                      hitPx: 8,
                  });
            /* 手柄在右边缘外侧上方，已在上面优先处理；这里只剩边缘与块体。 */
            setCanvasCursor(
                paramBlockCursorForHover(
                    edge !== null ? (edge.edge === "left" ? "left-edge" : "right-edge") : "body",
                ),
            );
            /*
             * 悬停状态只用于光标与局部强调；它不重建块数据，避免在一次普通
             * hover 里改变手柄的可见性结构。
             */
            const nextHoverBlockId = null;
            const nextTransitionId = null;
            if (
                hoveredParamBlockIdRef.current !== nextHoverBlockId ||
                hoveredTransitionIdRef.current !== nextTransitionId
            ) {
                /*
                 * Hover is computed from the **host's frame geometry**, while
                 * `invalidateParamBlockSpec()` rebuilds from React state. The two
                 * can disagree for a frame (selection commit is async), and the
                 * rebuilt spec may drop every transition handle. Keep the refs for
                 * visual hover only; do not invalidate the structural spec here.
                 */
                hoveredParamBlockIdRef.current = nextHoverBlockId;
                hoveredTransitionIdRef.current = nextTransitionId;
            }
            return true;
        },
        [
            groupedParamBlockIds,
            paramBlockHitSlopPx,
            paramBlockToolActive,
            paramBlocksWithValues,
            setCanvasCursor,
        ],
    );
    paramBlockPointerRef.current = onParamBlockPointerDown;
    useEffect(() => {
        const handler = (event: Event) => {
            const detail = (
                event as CustomEvent<{
                    op?: string;
                    stroke?: {
                        ctrl: boolean;
                        shift: boolean;
                        alt: boolean;
                        meta: boolean;
                    };
                }>
            ).detail;
            if (detail?.op === undefined) return;
            switch (detail.op) {
                case "split": {
                    /*
                     * 切分不是拖拽手势，但"临时反转吸附"对磁吸同样有意义，
                     * 而且与拖动块边缘 / 新建块的 XOR 语义一致。
                     */
                    const noSnapKb = firstBinding(mergedKeybindings["modifier.clipNoSnap"]);
                    const stroke = detail.stroke;
                    const toggle =
                        noSnapKb !== undefined &&
                        !isNoneBinding(noSnapKb) &&
                        stroke !== undefined &&
                        isModifierActive(noSnapKb, {
                            ctrlKey: stroke.ctrl,
                            shiftKey: stroke.shift,
                            altKey: stroke.alt,
                            metaKey: stroke.meta,
                        });
                    splitParamBlockAtPlayheadRef.current?.(toggle);
                    break;
                }
                case "delete":
                    deleteSelectedParamBlocksRef.current?.();
                    break;
                case "group":
                    groupSelectedParamBlocksRef.current?.();
                    break;
                case "ungroup":
                    ungroupSelectedParamBlocksRef.current?.();
                    break;
                case "selectAll":
                    setSelectedParamBlockIds(
                        paramBlocksWithValuesRef.current.map((block) => block.id),
                    );
                    break;
                case "deselect":
                    setSelectedParamBlockIds([]);
                    break;
                case "restoreGame":
                    restoreGameParamBlocksRef.current?.();
                    break;
                case "restoreDeleted":
                    restoreParamBlocksRef.current?.();
                    break;
                case "normalizePitch":
                    normalizeSelectedParamBlockPitchRef.current?.();
                    break;
            }
        };
        window.addEventListener("hifi:paramBlockEditOp", handler);
        return () => window.removeEventListener("hifi:paramBlockEditOp", handler);
    }, [mergedKeybindings, store]);
    /**
     * 块层数据是命令式写入宿主的；用受控 effect 在选中集合变化时立即冲刷，
     * 避免 React 选择态已经更新、宿主仍持有上一个 spec 的情况。
     */
    useEffect(() => {
        hostRef.current?.invalidate();
    }, [hostRef, selectedParamBlockIds]);
    const paramBlockSpec = useMemo<PianoRollParamBlockSpec | null>(() => {
        // Visibility applies before every gesture/commit preview branch too.
        if (!paramBlocksVisible) return null;
        // 拖动 ref 由命令式路径更新；tick 是让下一次 React memo 读取新 ref 的显式闸门。
        void paramBlockStructureTick;
        /*
         * 边缘伸缩 / 新建块的预览必须优先于拖拽快照。以前这里先检查
         * `paramBlockDragSnapshotRef`，而它的 key 与 `paramView.key` 相同，
         * 于是伸缩的 targetSec 记录在 ref 里却一直用旧块几何 —— 表现为
         * “松手才跳到位，不跟手”。同一帧里，行为 ref 比旧快照更权威。
         */
        const resizing = paramBlockResizeRef.current;
        if (resizing !== null && paramView !== null) {
            const resizeStart = resizing.resizePlan?.block.startSec ?? resizing.startSec;
            const resizeEnd = resizing.resizePlan?.block.endSec ?? resizing.endSec;
            if (resizing.blockId === "__create__") {
                const value =
                    resizing.value ??
                    averageParamValueInSpan({
                        startSec: Math.min(resizeStart, resizeEnd),
                        endSec: Math.max(resizeStart, resizeEnd),
                        values: paramView.edit,
                        startFrame: paramView.startFrame,
                        stride: paramView.stride,
                        framePeriodMs: paramView.framePeriodMs,
                        isPitch: editParam === "pitch",
                        param: editParam,
                    }) ??
                    (Number.isFinite(bounds.min) && Number.isFinite(bounds.max)
                        ? (bounds.min + bounds.max) / 2
                        : null);
                const previewBlocks = projectParamBlockCreatePreviewFromBase({
                    framePeriodMs: paramBlockStateRef.current.framePeriodMs,
                    baseBlocks:
                        resizing.createBaseBlocks ??
                        paramBlocksWithValues.map((block) => ({
                            id: block.id,
                            startSec: block.startSec,
                            endSec: block.endSec,
                            averageValue: block.averageValue,
                            logicalCenterValue: block.manualValue,
                            displayCenterSnapToSemitone: block.manualValue == null,
                            selected: false,
                            grouped: block.kind === "grouped",
                        })),
                    startSec: resizeStart,
                    endSec: resizeEnd,
                    createId: resizing.createdBlockId ?? "__create__",
                    value,
                });
                return {
                    curveOwned: paramBlockViewRef.current !== null,
                    param: editParam,
                    valueMin: bounds.min,
                    valueMax: bounds.max,
                    valueToY: (value, heightPx) => valueToY(editParam, value, heightPx),
                    blocks: previewBlocks.map((block) => ({
                        ...block,
                        selected: block.id === (resizing.createdBlockId ?? "__create__"),
                        frameCenterValue:
                            block.id === (resizing.createdBlockId ?? "__create__")
                                ? value
                                : undefined,
                        grouped:
                            "grouped" in block && typeof block.grouped === "boolean"
                                ? block.grouped
                                : false,
                    })),
                    borderRgba: parseRgbaColor(
                        normalizeCssColor(
                            themeMode === "dark"
                                ? "rgba(124,198,236,0.95)"
                                : "rgba(31,103,160,0.95)",
                        ),
                    ),
                    selectedRgba: parseRgbaColor(
                        normalizeCssColor(
                            themeMode === "dark" ? "rgba(210,242,255,1)" : "rgba(0,72,128,1)",
                        ),
                    ),
                    groupedRgba: parseRgbaColor(
                        normalizeCssColor(
                            themeMode === "dark"
                                ? "rgba(124,198,236,0.95)"
                                : "rgba(31,103,160,0.95)",
                        ),
                    ),
                    labelRgba: parseRgbaColor(
                        normalizeCssColor(
                            themeMode === "dark"
                                ? "rgba(244,251,255,0.92)"
                                : "rgba(235,247,255,0.92)",
                        ),
                    ),
                    selectedLabelRgba: parseRgbaColor(normalizeCssColor("rgba(255,255,255,1)")),
                    fillRgba: parseRgbaColor(
                        normalizeCssColor(
                            themeMode === "dark"
                                ? "rgba(57,111,145,0.90)"
                                : "rgba(107,158,192,0.84)",
                        ),
                    ),
                    selectedFillRgba: parseRgbaColor(
                        normalizeCssColor(
                            themeMode === "dark"
                                ? "rgba(49,132,180,0.98)"
                                : "rgba(54,132,184,0.96)",
                        ),
                    ),
                    groupedFillRgba: parseRgbaColor(
                        normalizeCssColor(
                            themeMode === "dark"
                                ? "rgba(57,111,145,0.90)"
                                : "rgba(107,158,192,0.84)",
                        ),
                    ),
                    topHighlightRgba: parseRgbaColor(
                        normalizeCssColor(
                            themeMode === "dark"
                                ? "rgba(255,255,255,0.11)"
                                : "rgba(255,255,255,0.20)",
                        ),
                    ),
                    cornerRadiusPx: 2,
                    topHighlightPx: 1,
                };
            }
            return {
                curveOwned: paramBlockViewRef.current !== null,
                param: editParam,
                valueMin: bounds.min,
                valueMax: bounds.max,
                valueToY: (value, heightPx) => valueToY(editParam, value, heightPx),
                /*
                 * 预览每帧从手势起点重新投影，而不是从上一帧的可见块继续。
                 * 否则块被判定删除后会从源集合里消失，指针再拉回来也无法恢复。
                 */
                blocks: (() => {
                    const plan = resizing.resizePlan;
                    if (plan === null) {
                        return paramBlocksWithValues.map((block) => ({
                            id: block.id,
                            startSec: block.startSec,
                            endSec: block.endSec,
                            averageValue: block.averageValue,
                            logicalCenterValue: block.manualValue,
                            displayCenterSnapToSemitone: block.manualValue == null,
                            selected: false,
                            grouped: block.kind === "grouped",
                        }));
                    }
                    const display = projectParamBlockResizeDisplay({
                        baseBlocks: resizing.baseBlockSpans,
                        blockId: resizing.blockId,
                        plan,
                    });
                    return display.map((shown) => {
                        const block = paramBlocksWithValues.find(
                            (candidate) => candidate.id === shown.id,
                        );
                        return {
                            id: shown.id,
                            startSec: shown.startSec,
                            endSec: shown.endSec,
                            averageValue: block?.averageValue ?? null,
                            logicalCenterValue: block?.manualValue,
                            displayCenterValue:
                                shown.id === resizing.blockId
                                    ? (resizing.displayCenterValue ?? null)
                                    : null,
                            /* 时间伸缩不改音高，显示中心保持吸附行心。 */
                            displayCenterSnapToSemitone: block?.manualValue == null,
                            displayHeightPx:
                                shown.id === resizing.blockId
                                    ? (resizing.displayHeightPx ?? null)
                                    : null,
                            selected: shown.selected,
                            grouped: block?.kind === "grouped",
                        };
                    });
                })(),
                borderRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(124,198,236,0.95)" : "rgba(31,103,160,0.95)",
                    ),
                ),
                transitionRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(219,246,255,0.92)" : "rgba(0,94,156,0.90)",
                    ),
                ),
                transitionPointRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(255,255,255,1)" : "rgba(0,66,112,1)",
                    ),
                ),
                transitionPointRadiusPx: 2.5,
                selectedRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(210,242,255,1)" : "rgba(0,72,128,1)",
                    ),
                ),
                groupedRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(124,198,236,0.95)" : "rgba(31,103,160,0.95)",
                    ),
                ),
                labelRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(244,251,255,0.92)" : "rgba(235,247,255,0.92)",
                    ),
                ),
                selectedLabelRgba: parseRgbaColor(normalizeCssColor("rgba(255,255,255,1)")),
                fillRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(57,111,145,0.90)" : "rgba(107,158,192,0.84)",
                    ),
                ),
                selectedFillRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(49,132,180,0.98)" : "rgba(54,132,184,0.96)",
                    ),
                ),
                groupedFillRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(57,111,145,0.90)" : "rgba(107,158,192,0.84)",
                    ),
                ),
                topHighlightRgba: parseRgbaColor(
                    normalizeCssColor(
                        themeMode === "dark" ? "rgba(255,255,255,0.11)" : "rgba(255,255,255,0.20)",
                    ),
                ),
                cornerRadiusPx: 2,
                topHighlightPx: 1,
            };
        }
        const dragSnapshot = paramBlockDragSnapshotRef.current;
        if (
            dragSnapshot !== null &&
            shouldUseParamBlockDragSnapshotForRender({
                snapshotKey: dragSnapshot.key,
                paramViewKey: paramView?.key ?? null,
                gestureActive: paramBlockActiveGestureTokenRef.current !== null,
                commitHeld: paramBlockDragCommitRef.current !== null,
            })
        ) {
            return paramBlockSpecForDragSnapshot(dragSnapshot);
        }
        /*
         * 提交后的短暂交接：只有当前派生还没有反映刚提交的区间时才用快照
         * 覆盖。一旦 state 更新到位，块会自然走正常派生，不会永久钉住。
         */
        const committed = paramBlockCommittedSpanRef.current;
        if (committed !== null) {
            const now = performance.now();
            const current = paramBlocksWithValues.find((block) => block.id === committed.blockId);
            const settled =
                current !== undefined &&
                Math.abs(current.startSec - committed.blockSpan.startSec) <= 1e-6 &&
                Math.abs(current.endSec - committed.blockSpan.endSec) <= 1e-6;
            if (now > committed.expiresAt || settled) {
                paramBlockCommittedSpanRef.current = null;
            }
        }
        const committedSpans = paramBlockCommittedSpanRef.current;
        if (paramBlocksWithValues.length === 0) return null;
        if (!paramView) return null;
        const sortedBlocks = paramBlocksWithValues
            .slice()
            .sort((a, b) => a.startSec - b.startSec || a.endSec - b.endSec);
        const transitionByLeftId = new Map(
            sortedParamBlockTransitionHandles(paramBlocksWithValues).map((handle) => [
                handle.leftId,
                handle,
            ]),
        );
        return {
            curveOwned: paramBlockViewRef.current !== null,
            param: editParam,
            valueMin: bounds.min,
            valueMax: bounds.max,
            valueToY: (value, heightPx) => valueToY(editParam, value, heightPx),
            blocks: sortedBlocks
                .map<PianoRollParamBlockSpec["blocks"][number] | null>(
                    (block, blockIndex, allBlocks) => {
                        const previous = allBlocks[blockIndex - 1];
                        const next = allBlocks[blockIndex + 1];
                        const contiguousStart =
                            previous !== undefined &&
                            Math.abs(previous.endSec - block.startSec) < 1e-6;
                        const contiguousEnd =
                            next !== undefined && Math.abs(next.startSec - block.endSec) < 1e-6;
                        const connected = transitionByLeftId.get(block.id);
                        const transitionHandle =
                            connected !== undefined
                                ? {
                                      sec: connected.sec,
                                      leftId: connected.leftId,
                                      rightId: connected.rightId,
                                      // 该相接关系只挂在这个左块上；不要按右块 id 回查，
                                      // 否则同一宽度会串到右块的另一条边界。
                                      widthSec: connected.transitionSec,
                                  }
                                : null;
                        const ordinal = blockIndex + 1;
                        const committedSpan =
                            committedSpans?.blockId === block.id
                                ? committedSpans.blockSpan
                                : (committedSpans?.neighbors.find((item) => item.id === block.id) ??
                                  null);
                        if (
                            committedSpan !== null &&
                            "remove" in committedSpan &&
                            committedSpan.remove
                        ) {
                            return null;
                        }
                        return {
                            id: block.id,
                            startSec: committedSpan?.startSec ?? block.startSec,
                            endSec: committedSpan?.endSec ?? block.endSec,
                            topPx: 0,
                            heightPx: 0,
                            averageValue: block.averageValue,
                            logicalCenterValue: block.manualValue,
                            displayCenterSnapToSemitone: block.manualValue == null,
                            contiguousStart,
                            contiguousEnd,
                            transitionHandle,
                            selected: selectedParamBlockIdSet.has(block.id),
                            grouped: block.kind === "grouped",
                            label: formatParamBlockLabel(block.averageValue, ordinal, editParam),
                        };
                    },
                )
                .filter(
                    (block): block is PianoRollParamBlockSpec["blocks"][number] => block !== null,
                ),
            borderRgba: parseRgbaColor(
                normalizeCssColor(
                    themeMode === "dark" ? "rgba(124,198,236,0.95)" : "rgba(31,103,160,0.95)",
                ),
            ),
            transitionRgba: parseRgbaColor(
                normalizeCssColor(
                    themeMode === "dark" ? "rgba(219,246,255,0.92)" : "rgba(0,94,156,0.90)",
                ),
            ),
            transitionPointRgba: parseRgbaColor(
                normalizeCssColor(
                    themeMode === "dark" ? "rgba(255,255,255,1)" : "rgba(0,66,112,1)",
                ),
            ),
            transitionPointRadiusPx: 2.5,
            selectedRgba: parseRgbaColor(
                normalizeCssColor(
                    themeMode === "dark" ? "rgba(210,242,255,1)" : "rgba(0,72,128,1)",
                ),
            ),
            groupedRgba: parseRgbaColor(
                normalizeCssColor(
                    themeMode === "dark" ? "rgba(124,198,236,0.95)" : "rgba(31,103,160,0.95)",
                ),
            ),
            labelRgba: parseRgbaColor(
                normalizeCssColor(
                    themeMode === "dark" ? "rgba(244,251,255,0.92)" : "rgba(235,247,255,0.92)",
                ),
            ),
            selectedLabelRgba: parseRgbaColor(
                normalizeCssColor(
                    themeMode === "dark" ? "rgba(255,255,255,1)" : "rgba(255,255,255,1)",
                ),
            ),
            fillRgba: parseRgbaColor(
                normalizeCssColor(
                    themeMode === "dark" ? "rgba(57,111,145,0.90)" : "rgba(107,158,192,0.84)",
                ),
            ),
            selectedFillRgba: parseRgbaColor(
                normalizeCssColor(
                    themeMode === "dark" ? "rgba(49,132,180,0.98)" : "rgba(54,132,184,0.96)",
                ),
            ),
            groupedFillRgba: parseRgbaColor(
                normalizeCssColor(
                    themeMode === "dark" ? "rgba(57,111,145,0.90)" : "rgba(107,158,192,0.84)",
                ),
            ),
            topHighlightRgba: parseRgbaColor(
                normalizeCssColor(
                    themeMode === "dark" ? "rgba(255,255,255,0.11)" : "rgba(255,255,255,0.20)",
                ),
            ),
            /*
             * 视觉统一：未选中块都是一种实心蓝灰底，只有“被选中 / 合并”
             * 才换强调色。之前按 merged 选色会让用户看到几种色块混在一起。
             */
            cornerRadiusPx: 2,
            topHighlightPx: 1,
        };
    }, [
        paramBlocksVisible,
        paramBlocksWithValues,
        paramBlockStructureTick,
        selectedParamBlockIdSet,
        valueToY,
        paramView,
        themeMode,
        bounds.min,
        bounds.max,
        editParam,
        paramBlockSpecForDragSnapshot,
    ]);
    latestParamBlockSpecRef.current = paramBlockSpec;
    /*
     * 新建块预览 -> 正式块的交接。只有同时满足以下条件才撤预览：
     * 1. React 块记录已包含新块；
     * 2. pitch 曲线提交后至少一份权威 paramView 已落地；
     * 3. 当前正式块已绘制过一帧。
     */
    useEffect(() => {
        const handoff = paramBlockCreateCommitRef.current;
        if (
            handoff === null ||
            !handoff.committed ||
            paramBlockDragActivityRef.current.pointerGestureToken !== null ||
            paramBlockDragActivityRef.current.pendingCommits > 0
        )
            return;
        if (
            handoff.requiresCurveAuthority &&
            (paramAuthorityRevision <= handoff.authorityRevisionAtStart ||
                !paramView ||
                !handoff.committedSegments ||
                !paramViewContainsCommittedSegments({
                    paramView,
                    segments: handoff.committedSegments,
                }) ||
                !isParamAuthorityConfirmedAfterCommit({
                    watermark: getAuthorityWatermark(),
                    issuedRequestSeqAtCommit: handoff.authorityRequestSeqAtCommit,
                }))
        ) {
            return;
        }
        if (!paramBlocksWithValues.some((block) => block.id === handoff.blockId)) return;
        if (paramBlockResizeRef.current?.blockId !== "__create__") return;
        if (paramBlockCreateReleaseRafRef.current !== null) return;
        paramBlockCreateReleaseRafRef.current = requestAnimationFrame(() => {
            paramBlockCreateReleaseRafRef.current = null;
            if (
                paramBlockCreateCommitRef.current !== handoff ||
                paramBlockDragActivityRef.current.pointerGestureToken !== null ||
                paramBlockDragActivityRef.current.pendingCommits > 0
            )
                return;
            paramBlockCreateCommitRef.current = null;
            if (paramBlockResizeRef.current?.blockId === "__create__") {
                paramBlockResizeRef.current = null;
            }
            paramBlockInteractionCurveRef.current = null;
            if (paramView) resetLiveEditPreview(paramView);
            bumpParamBlockStructure();
            invalidateParamBlockSpec();
        });
        return () => {
            if (paramBlockCreateReleaseRafRef.current !== null) {
                cancelAnimationFrame(paramBlockCreateReleaseRafRef.current);
                paramBlockCreateReleaseRafRef.current = null;
            }
        };
    }, [
        bumpParamBlockStructure,
        getAuthorityWatermark,
        invalidateParamBlockSpec,
        paramAuthorityRevision,
        paramView,
        resetLiveEditPreview,
        paramBlockSpec,
        paramBlocksWithValues,
    ]);
    useParamBlockDragHandoff({
        commitRef: paramBlockDragCommitRef,
        snapshotRef: paramBlockDragSnapshotRef,
        interactionCurveRef: paramBlockInteractionCurveRef,
        activeGestureTokenRef: paramBlockActiveGestureTokenRef,
        releaseRafRef: paramBlockDragCommitReleaseRafRef,
        paramView,
        authorityRevision: paramAuthorityRevision,
        getAuthorityWatermark,
        structureVersion: paramBlockStructureTick,
        clearLiveEditOverride,
        onReleased: bumpParamBlockStructure,
    });
    const drawParamBlocks = useCallback(
        (spec: PianoRollParamBlockSpec | null) => {
            const hoveredBlockId = hoveredParamBlockIdRef.current;
            const hoveredTransitionId = hoveredTransitionIdRef.current;
            kernelDataRef.current.paramBlocks =
                spec === null
                    ? spec
                    : {
                          ...spec,
                          curveOwned: paramBlockViewRef.current !== null,
                          blocks:
                              hoveredBlockId === null && hoveredTransitionId === null
                                  ? spec.blocks
                                  : spec.blocks.map((block) => ({
                                        ...block,
                                        hovered: block.id === hoveredBlockId,
                                        transitionHovered:
                                            hoveredTransitionId !== null &&
                                            block.id === hoveredTransitionId,
                                    })),
                      };
            /*
             * The blue block is a handle for this exact curve view. The shared
             * resolver also feeds buildCurveLayers, so no persisted center or
             * gesture snapshot can move the block independently from the line.
             */
            const frameView = resolveParamBlockFrameView({
                interaction: paramBlockInteractionCurveRef.current,
                live: liveEditOverrideRef.current,
                paramView: paramViewRef.current,
                param: editParam,
            });
            kernelDataRef.current.paramBlockSamples =
                frameView === null
                    ? null
                    : {
                          values: frameView.edit,
                          startFrame: frameView.startFrame,
                          stride: frameView.stride,
                          framePeriodMs: frameView.framePeriodMs,
                          isPitch: editParam === "pitch",
                          param: editParam,
                      };
            invalidate();
        },
        [editParam, invalidate, liveEditOverrideRef, paramBlockInteractionCurveRef, paramViewRef],
    );
    useLayoutEffect(() => {
        drawParamBlocks(paramBlockSpec);
    }, [drawParamBlocks, paramBlockSpec]);
    useLayoutEffect(() => {
        /*
         * 曲线 / 块拖动写在 live override 的原地数组里，不换 React state。
         * 每次数据窗口、参数或可见性变化重建一次采样引用；拖动时宿主读到的
         * 仍是同一数组的最新内容。
         */
        drawParamBlocks(paramBlockSpec);
    }, [
        drawParamBlocks,
        paramBlockSpec,
        paramView,
        liveEditOverrideRef,
        paramBlockDragSnapshotRef,
    ]);
    useLayoutEffect(() => {
        // 块层与曲线同属“内容坐标 → 视口投影”的图层；初次挂载时 host 可能
        // 晚于本 effect 创建，补一帧请求避免块停在首帧之前。
        invalidate();
    }, [invalidate, paramBlockSpec]);
    useLayoutEffect(() => {
        hostRef.current?.invalidate();
    }, [paramBlockToolActive, paramBlocksVisible]);
    useLayoutEffect(() => {
        // 类型/参数切换时，参数窗口要等下一次取数才回来。块高依赖这窗口；
        // 先清掉，避免上一参数的块轮廓停在新标尺上。
        if (paramView !== null) return;
        drawParamBlocks(null);
    }, [drawParamBlocks, paramView]);

    // 工具图标是纯 JSX 常量（只依赖当前工具），见 `toolIcons.tsx`。
    const {
        vibratoToolIcon,
        lineToolIcon,
        pitchSnapSemitoneIcon,
        pitchSnapScaleIcon,
        currentDrawToolIcon,
    } = buildToolIcons(currentDrawTool);

    // 标尺刻度 / 时间格式 / 单位切换 / 复制播放头时间整块在
    // `useTimelineDisplayControls` 里（见该文件头）。它们只依赖会话切片与刻度轴。
    const {
        timelineTicks,
        timeContext,
        handlePrimaryUnitChange,
        handleSecondaryUnitChange,
        handleTempoMapChange,
        handleTempoMapCommit,
        handleCopyPlayheadTime,
    } = useTimelineDisplayControls({ dispatch, s, prTickAxis });

    /*
     * 【这个返回对象就是视图层的全部输入】字段清单是三个视图文件 AST 扫描结果的并集
     * （另加外框自己用的 `paramEditorRef`）。新增字段 = 视图里用一个新名字，然后在这里补上。
     */
    return {
        activeDragDirection,
        activeDragDirectionTool,
        activeVibratoPreset,
        activeVibratoPresetId,
        attachEdgeSmoothnessWheel,
        childFormantOffsetParam,
        childPitchOffsetCentsParam,
        childPitchOffsetDegreesParam,
        composeEnabled,
        currentDrawTool,
        currentDrawToolIcon,
        cycleDragDirectionKb,
        devicePixelRatio,
        dispatch,
        drawToolButtonTitle,
        drawToolKb,
        drawToolMenuAnchorRef,
        drawToolMenuOpen,
        drawToolMenuRef,
        editParam,
        effectivePitchSnapVisual,
        enabledVibratoPresetList,
        gatedPillHideOrder,
        getProcessorParamLabel,
        getProcessorParamShortLabel,
        getStaticOptionLabel,
        handleOpenMidiDialog,
        handleStaticParamChange,
        lineToolIcon,
        lineToolKb,
        openVibratoDialog,
        orderedProcessorParams,
        pitchAlgoOptions,
        pitchEnabled,
        pitchGroupActive,
        pitchGroupLabel,
        pitchGroupTooltip,
        pitchHardDisableReason,
        pitchSnapMenuAnchorRef,
        pitchSnapMenuOpen,
        pitchSnapMenuRef,
        pitchSnapOpen,
        paramBlockToolActive,
        paramBlocksVisible,
        paramBlockContextMenu,
        canRestoreGameParamBlocks,
        restoreGameParamBlocks,
        setParamBlockContextMenu,
        selectedParamBlockCount: selectedParamBlockIds.length,
        selectedParamBlockIds,
        canGroupSelectedParamBlocks: canGroupParamBlocks(
            paramBlocksWithValues,
            selectedParamBlockIds,
        ),
        canUngroupSelectedParamBlocks: paramBlocksWithValues.some(
            (block) => block.kind === "grouped" && selectedParamBlockIdSet.has(block.id),
        ),
        groupSelectedParamBlocks,
        ungroupSelectedParamBlocks,
        deleteSelectedParamBlocks,
        restoreParamBlocks,
        canDeleteSelectedParamBlocks: selectedParamBlockIds.length > 0,
        canRestoreDeletedParamBlocks: (paramBlockState.lastDeletedBlocks ?? []).length > 0,
        canNormalizeSelectedParamBlock: editParam === "pitch" && selectedParamBlockIds.length === 1,
        normalizeSelectedParamBlockPitch: () => {
            if (selectedParamBlockIds.length !== 1) return;
            const block = paramBlocksWithValues.find(
                (item) => item.id === selectedParamBlockIds[0],
            );
            if (block !== undefined) void normalizeParamBlockPitch(block);
        },
        paramBlockResizeRef,
        onCanvasDoubleClick,
        onCanvasPointerMoveForBlocks,
        canSplitParamBlockAtPlayhead,
        splitParamBlockAtPlayhead,
        pitchSnapScaleIcon,
        pitchSnapSemitoneIcon,
        presetFlyoutMaxHeight,
        processorParamsRef,
        processorStaticParams,
        processorStaticValues,
        referenceRootTrackOptions,
        rootTrack,
        rootTrackId,
        s,
        secondaryParamVisible,
        selectedIsChildTrack,
        separationEnabled,
        separationSwitchAnchorParamId,
        setDrawToolMenuOpen,
        setHoveredReferenceRootTrackId,
        setPitchSnapMenuOpen,
        setPitchSnapOpen,
        setTimeDisplaySettingsOpen,
        t,
        tf,
        timeDisplaySettingsOpen,
        toggleSecondaryParam,
        toggleParamBlocksVisible,
        toolbarRowRef,
        toolbarTier,
        updateVisibleReferenceRootTrackIds,
        vibratoToolIcon,
        vibratoToolKb,
        visibleReferenceRootTrackIds,
        attachHorizontalScrollbarTrack,
        attachRulerPlayheadHead,
        attachRulerPlayheadLine,
        attachVerticalScrollbarTrack,
        axisCanvasRef,
        axisUnitToggleAvailable,
        axisValuePreview,
        axisWrapRef,
        canvasCursor,
        canvasRef,
        captureBandsRef,
        clipPeaks,
        contentWidth,
        currentParamRange,
        dynamicProjectSec,
        editParamAxisUnit,
        effectiveProjectScale,
        formatParamValuePreview,
        getScrollLeftLive,
        glAxisCanvasRef,
        glCanvasRef,
        glOverlayCanvasRef,
        gridLayerRef,
        hScrollbarThumbRef,
        handleCopyPlayheadTime,
        handlePrimaryUnitChange,
        handleRulerWheel,
        handleSecondaryUnitChange,
        handleTempoMapChange,
        handleTempoMapCommit,
        interactions,
        onParamBlockPointerDown,
        onScrollerScroll,
        paddedContentWidth,
        paramBlocksWithValues,
        paramValuePreview,
        pianoRollAmplitudeMap,
        prTickAxis,
        pxPerBeat,
        pxPerSec,
        resolvedVibratoPresets,
        rulerContentRef,
        scrollLeft,
        scrollerRef,
        selectedParamBlockIdSet,
        timeContext,
        timelineTicks,
        vScrollbarThumbRef,
        vibratoDragHud,
        viewSize,
        waveformColors,
        clipboardGuid,
        closeLeadingGap,
        ctxMenu,
        fillGaps,
        handleCloseLeadingGapChange,
        handleEditOp,
        handleExportMidiFromEditor,
        handleFillGapsChange,
        handleImportAsClip,
        handleImportBpmAsProjectChange,
        handleImportPositionChange,
        handleImportTargetChange,
        handleImportTempoMapEnabledChange,
        handleImportTempoMapKeySignatureChange,
        handleImportTempoMapTempoChange,
        handleImportTempoMapTimeSignatureChange,
        handleMidiImported,
        handleMultiTrackMergeChange,
        handleNoteBpmModeChange,
        handleRequestEnableCompose,
        handleSaveAsPitchRef,
        handleSpecifiedBpmChange,
        importBpmAsProject,
        importPosition,
        importTargetParamEditor,
        importTargetReaperClipboard,
        importTempoMapEnabled,
        importTempoMapKeySignature,
        importTempoMapTempo,
        importTempoMapTimeSignature,
        midiDialogOpen,
        midiDialogSourceRef,
        midiPath,
        midiSelRanges,
        midiSelectionAvailable,
        multiTrackMerge,
        noteBpmMode,
        openEditDialog,
        paramEditorMidiDragOver,
        setCtxMenu,
        setMidiDialogOpen,
        setVibratoDialogOpen,
        specifiedBpm,
        // —— 内部依赖（不经视图，供 `editOperations.ts` 的命令分发取用）——
        bumpRefreshToken,
        clipboardRef,
        currentParamDefaultValue,
        currentParamQuantizeUnit,
        invalidate,
        midiDialogOpenParamsRef,
        paramView,
        paramViewRef,
        pitchDeltaToDegreeSteps,
        projectScaleAtSec,
        resolveScaleFromToken,
        selectAllParamRange,
        selectionRef,
        setClipboardGuid,
        setMidiDialogSelection,
        setMidiPath,
        setParamView,
        setSelectionUi,
        store,
        vibratoBaseline,
        // —— 内部依赖（不经视图，供 `sceneBuilders.ts` / `drawFrame.ts` 取用）——
        buildCurveLayers,
        buildSelectionBandSpec,
        childPitchHardDisableReason,
        clampViewport,
        detectedPitchCurves,
        fontFamily,
        getCurrentViewportForScrollbar,
        getParamValueBoundsForScrollbar,
        hostRef,
        kernelDataRef,
        lastMirroredScrollTopRef,
        liveEditOverrideRef,
        loudnessAnalysisPending,
        paramMorphOverlays,
        paramViewsRef,
        pitchViewRef,
        prAxis,
        pxPerSecRef,
        referencePitchOverlays,
        scrollLeftRef,
        secSpanClipRect,
        secondaryParamViews,
        selectionClipRect,
        themeMode,
        valueToY,
        viewSizeRef,
        visibleSecondaryParamIds,
        visualPlayheadSecRef,
        vibratoApplyTarget,
        vibratoDialogMode,
        vibratoDialogOpen,
        vibratoDialogSession,
        vibratoInitialPresetId,
        paramEditorRef,
    };
}

/** 模型对象的类型：由实现推出，视图层据此声明 `Pick<>`。 */
export type PianoRollPanelModel = ReturnType<typeof usePianoRollPanelModel>;
