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
    toggleAlwaysShowParamBlocks as toggleAlwaysShowParamBlocksAction,
    isParamBlockLayerVisible,
    seekPlayhead,
    setPlayheadSec,
    setParamSelectionActive,
} from "../../../../features/session/sessionSlice";
import { resolveRootTrackId } from "../../../../features/session/trackUtils";
import type { ClipInfo } from "../../../../features/session/sessionTypes";
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
    normalizeSelection,
    removeRangeAtFrame,
    selectionBoundingSpan,
    selectionFromFrames,
    selectionToFrameRanges,
    type ParamSelection,
} from "../paramSelection";
import {
    expandParamBlockSelection,
    logicalFrameRangesFromView,
    paramBlockIdsInSelection,
    paramBlockLogicalFrameRanges,
    paramBlockRestoreRanges,
    selectAllParamBlockRanges,
} from "../paramBlockSelection";
import {
    resolveBlockAnchor,
    resolveBlockPressPlan,
    type BlockClickMode,
    type ParamBlockSelectionAnchor,
} from "../paramBlockSelectionGesture";
import { selectionForRangeDrag } from "../paramSelectionDrag";
import type { ParamBlockCommandContext } from "../paramBlockCommandMenu";
import { isBlockSelectionActive, isRangeToolActive } from "../../../../features/session/toolMode";
import {
    captureParamBlockCreateCurve,
    enqueueParamBlockCreateCommit,
} from "../paramBlockCreationTransaction";
import { usePianoRollData } from "../usePianoRollData";
import { useParamBlockDragHandoff } from "../useParamBlockDragHandoff";
import { resolveParamBlockSplitSelection } from "../paramBlockSplitSelection";
import { HISTORY_JUMP_EVENT } from "../../../../features/session/historyJump";
import { createSyllableAnalysisQueue } from "../syllableAnalysisQueue";
import { useParamBlockPersistence } from "../useParamBlockPersistence";
import { useParamBlockState } from "../useParamBlockState";
import { useLatestRef, useLatestRefValue } from "../useLatestRef";
import { trackParamBlockCommit } from "../paramBlockCommitCoordinator";
import { composeParamBlockView } from "../paramBlockComposition";
import {
    analysisWindowStatus,
    projectionForClip,
    audibleSourceWindow,
} from "../paramBlockProjection";
import { paramBlockStaleAnalysisCandidates } from "../paramBlockAnalysisCandidate";
import { planParamBlockGesture, type ParamBlockGesturePlan } from "../paramBlockGesture";
import { reportParamBlockDiagnostic } from "../paramBlockDiagnostics";
import { publishParamBlockWriteFailure } from "../paramBlockFeedback";
import { createParamBlockWritePlan } from "./paramBlockCreateWritePlan";
import { paramBlockStructureToken } from "../paramBlockStructureToken";
import {
    reportParamBlockLocalRejection,
    reportParamBlockWriteFailure,
} from "./paramBlockFailureReporting";
import type { ParamBlockWritePreparation } from "../paramBlockWritePlan";
import {
    toFrame,
    toParamBlockSourceClip,
    type ParamBlockSourceClip,
} from "../paramBlockProjection";
import { logicalBlockId, type ParamBlockView } from "../paramBlockTypes";
import { resolveParamBlockGeometryClips } from "../paramBlockGeometryFreeze";
import { fetchFullResCurve, readPvRange } from "../selectionEditData";
import { useParamBlockTransitionPersistence } from "./useParamBlockTransitionPersistence";
import { startPointerGesture } from "../gestures/pointerGesture";
import { armRightDragContextMenuGuard } from "../../../../utils/rightDragContextMenuGuard";
import { normalizeCssColor } from "../colors";
import { parseRgbaColor } from "../../timeline/runtime/timelineClipGlRenderer";
import {
    EMPTY_PARAM_BLOCK_STATE,
    canGroupParamBlocks,
    clearParamBlockValues,
    deleteParamBlocks,
    paramBlockEdgeCandidates,
    hitTestParamBlockEdges,
    hitTestParamBlockTransitionHandles,
    hitTestParamBlocks,
    hitTestParamBlocksAtPoint,
    materializeParamBlocks,
    logicalBlockKey,
    blocksFromViewForPanel,
    groupParamBlocks,
    applyFullResSegmentsToParamView,
    projectParamBlockCreatePreviewFromBase,
    shiftParamBlockCurveValuesToTarget,
    shouldUseParamBlockDragSnapshotForRender,
    nearestParamBlockByTime,
    isStrictlyInsideParamBlock,
    resolveParamBlockSplitSec,
    resolveParamBlockSplitTarget,
    resolveSplittableParamBlockAt,
    resolveParamBlockCreateSpan,
    resolveParamBlockResizeTarget,
    ungroupParamBlocks,
    restoreDeletedParamBlocks,
    splitParamBlock,
    setParamBlockTransition,
    buildParamBlockTransitionPreview,
    PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX,
    PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX,
    sortParamBlocksByTime,
    createParamBlock,
    clipAtParamBlockCreateSpan,
    resolveCreatedParamBlockValue,
    newParamBlockId,
    type ParamBlock,
    type ParamBlockState,
} from "../paramBlocks";
import { resolveParamBlockFrameView, resolveParamBlockPanelValues } from "../paramBlockFrame";
import {
    buildParamBlockRenderSpec,
    type ParamBlockRenderBlock,
    type ParamBlockSpecColors,
} from "../paramBlockRenderSpec";
import {
    adoptParamBlockVisualStructureAfterCreate,
    adoptParamBlockVisualStructureAfterEdits,
    isParamBlockVisualEpochCurrent,
    sameParamBlockEpochLayout,
    type ParamBlockEpoch,
    type ParamBlockVisualTransaction,
    type ParamBlockVisualTransactionKind,
} from "../paramBlockVisualTransaction";
import {
    isParamBlockPreviewGeometrySettled,
    PARAM_BLOCK_PREVIEW_RELEASE_TIMEOUT_MS,
    resolveParamBlockRelease,
    shouldReleaseParamBlockPreview,
    stampParamBlockCurveIntent,
} from "../paramBlockPreviewRelease";
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
import { isDynParam } from "../paramRanges";
import {
    resolveParamBlockCreateShape,
    resolveParamBlockModifierSnapshot,
    type ParamBlockCreateShape,
} from "../paramBlockModifierPolicy";
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
import { resolveParamAxisUnit, supportsParamAxisUnit } from "../paramAxisUnits";
import { isPanParam, resolvePanAxisDirection, type PanAxisDirections } from "../paramPan";
import { supportsParamFlip } from "../paramFlip";
import { useLoudnessCurves, type LoudnessSnapshot } from "../useLoudnessCurves";
import { createLiveOverrideReader, type LiveOverrideReader } from "../liveLoudnessOverride";
import { pianoRollViewportBus } from "../pianoRollViewportBus";
import { pianoRollInteractionViewportBus } from "../pianoRollInteractionViewportBus";
import { createLoudnessGeometryWarp, type LoudnessGeometryWarp } from "../loudnessGeometryWarp";
import {
    getClipGeometryPreviewClips,
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
import type { EdgeScrollDriver } from "../kernel/edgeScrollDriver";
import type {
    MutablePianoRollKernelData,
    PianoRollGridSpec,
    PianoRollParamBlockSpec,
} from "../kernel/host/pianoRollKernelData";
import { resolvePanelRenderViewport } from "../kernel/viewportSource";
import { usePianoRollParamGates } from "./usePianoRollParamGates";
import { usePianoRollScrollerWiring } from "./usePianoRollScrollerWiring";
import { usePianoRollSelectionActions } from "./usePianoRollSelectionActions";
import { useTimelineDisplayControls } from "./useTimelineDisplayControls";
import { buildToolIcons } from "./toolIcons";
import { handleParamBlockHover } from "./paramBlockHover";
import { useParamBlockEditOpChannel } from "./useParamBlockEditOpChannel";
import { useParamBlockRenderSpec, type ParamBlockResizePreview } from "./useParamBlockRenderSpec";
import { useParamValuePreviewPublisher } from "./useParamValuePreviewPublisher";
import {
    isParamBlockMenuOp,
    runParamBlockMenuOp,
    type ParamBlockMenuOp,
    type ParamBlockOpHandlers,
} from "./paramBlockEditOps";
import { usePianoRollEditCommands } from "./usePianoRollEditCommands";
import { usePianoRollEditSession } from "./usePianoRollEditSession";
import { usePianoRollFrameAssembly } from "./usePianoRollFrameAssembly";
import { usePianoRollScrollKernel } from "./usePianoRollScrollKernel";
import { useMidiImport } from "./useMidiImport";
import {
    pitchDeltaToDegreeSteps,
    buildGridSpec as buildGridSpecImpl,
    yToViewportT as yToViewportTImpl,
    clampViewport as clampViewportImpl,
    getParamValueBoundsForScrollbar as getParamValueBoundsForScrollbarImpl,
    getCurrentViewportForScrollbar as getCurrentViewportForScrollbarImpl,
    syncVerticalScrollbarForViewport as syncVerticalScrollbarForViewportImpl,
} from "./sceneBuilders";
import { usePianoRollViewport } from "./usePianoRollViewport";
import { usePianoRollProjection } from "./usePianoRollProjection";
import { useParamValueFormatter } from "./useParamValueFormatter";
import { useProcessorParams } from "./useProcessorParams";

/**
 * 空选中集合的**共享实例**：非块模式下用它替代真实选区，避免每次渲染换引用
 * 让 `paramBlockSpec` 的 memo 白跑一次。集合从不被写入，因此可以安全共享。
 */
const EMPTY_BLOCK_ID_SET: ReadonlySet<string> = new Set<string>();
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
    /** Block selection is a select-family submode, not a third ToolMode. */
    const paramBlockToolActive = isBlockSelectionActive(s.toolMode, s.selectToolMode);
    /** 选择族 range 子模式；四点编辑 / 选区光标 / 边缘拉伸只属于它（见 isRangeToolActive）。 */
    const rangeToolActive = isRangeToolActive(s.toolMode, s.selectToolMode);

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
    const editParamRef = useLatestRef(editParam);
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
    const paramRangeSelectKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramRangeSelect"),
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
    const selectToolKb = useAppSelector((state) => selectKeybindings(state, "mode.selectTool"));
    const blockSelectionKb = useAppSelector((state) =>
        selectKeybindings(state, "mode.toggleBlockSelection"),
    );
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
    const [selectToolMenuOpen, setSelectToolMenuOpen] = useState(false);
    /**
     * 工具栏下拉的**锚点**与**面板** ref 必须分开。
     *
     * 【为什么】面板经 `AppAnchoredMenu` 挂到 `document.body` 下，不再是锚点的
     * DOM 后代 —— 用锚点的 ref 做 `contains` 判定会永远为假，于是"点菜单里的项"
     * 会被当成"点了外面"，菜单在选中前就被关掉。
     */
    const drawToolMenuAnchorRef = useRef<HTMLDivElement | null>(null);
    const drawToolMenuRef = useRef<HTMLDivElement | null>(null);
    const selectToolMenuAnchorRef = useRef<HTMLDivElement | null>(null);
    const selectToolMenuRef = useRef<HTMLDivElement | null>(null);
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
    const editParamAxisDirection = resolvePanAxisDirection(s.paramAxisDirections, editParam);
    const axisPanDirectionToggleAvailable = isPanParam(editParam);
    const axisToggleAvailable = axisUnitToggleAvailable || axisPanDirectionToggleAvailable;

    const formatParamValuePreview = useParamValueFormatter({
        editParam,
        axisUnit: editParamAxisUnit,
        tf,
    });

    const currentDrawTool = s.drawToolMode;
    const selectToolButtonTitle =
        s.selectToolMode === "block" ? tf("select_tool_block") : tf("select_tool_range");
    const drawToolButtonTitle =
        currentDrawTool === "vibrato"
            ? // 颤音工具下把当前预设一并报出来：预设按钮已经并进右键菜单，
              // 工具按钮的 tooltip 是"接下来会画出什么"的最后一块可扫读信息。
              `${tf("vibrato_draw_tool")}: ${vibratoPresetLabel(activeVibratoPreset, tf)}`
            : currentDrawTool === "line"
              ? tf("line_draw_tool")
              : tf("draw_tool");
    const activeDragDirection =
        s.toolMode === "select"
            ? s.selectDragDirection
            : currentDrawTool === "draw"
              ? s.drawDragDirection
              : s.lineVibratoDragDirection;
    // 直线与颤音共用同一份拖动方向（`lineVibratoDragDirection`）—— 它们是同一个
    // "起点 → 终点"手势，拆成两份只会让"在直线下调过、切到颤音又不算"成为新的困惑。
    const activeDragDirectionTool =
        s.toolMode === "select"
            ? ("select" as const)
            : currentDrawTool === "draw"
              ? ("draw" as const)
              : ("vibrato" as const);

    useEffect(() => {
        if (!drawToolMenuOpen && !selectToolMenuOpen && !pitchSnapMenuOpen) return;
        const onPointerDown = (e: PointerEvent) => {
            const target = e.target as Node | null;
            if (drawToolMenuRef.current?.contains(target)) return;
            if (selectToolMenuRef.current?.contains(target)) return;
            if (pitchSnapMenuRef.current?.contains(target)) return;
            setDrawToolMenuOpen(false);
            setSelectToolMenuOpen(false);
            setPitchSnapMenuOpen(false);
        };
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                setDrawToolMenuOpen(false);
                setSelectToolMenuOpen(false);
                setPitchSnapMenuOpen(false);
            }
        };
        window.addEventListener("pointerdown", onPointerDown, true);
        window.addEventListener("keydown", onKeyDown, true);
        return () => {
            window.removeEventListener("pointerdown", onPointerDown, true);
            window.removeEventListener("keydown", onKeyDown, true);
        };
    }, [drawToolMenuOpen, selectToolMenuOpen, pitchSnapMenuOpen]);

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
    type ValueViewportCommitMode = "immediate" | "coalesced";
    /**
     * The single value-viewport commit entry.
     *
     * "immediate" is for wheel / middle-pan: DOM and GL must land in the same
     * task. "coalesced" is for edge auto-scroll: the gesture's own rAF already
     * draws this frame, so another synchronous full-frame paint would double
     * the work (the reported vertical jank).
     */
    function commitValueViewport(
        param: ParamName,
        next: ValueViewport,
        mode: ValueViewportCommitMode,
    ): void {
        const clamped = clampViewport(param, next);
        if (param === "pitch") pitchViewRef.current = clamped;
        else paramViewsRef.current = { ...paramViewsRef.current, [param]: clamped };
        syncVerticalScrollbarForViewport(param, clamped);
        publishInteractionViewport(param, clamped);
        if (mode === "immediate") {
            paintNow();
            return;
        }
        const host = hostRef.current;
        if (host) host.invalidate();
        else invalidate();
    }

    const setPitchView = useCallback(
        (next: ValueViewport) => {
            // 值域视口是**竖轴的手势提交点**：滚动条 thumb 已同步写下 DOM，曲线 /
            // 音高键 / 数值轴（GL）必须在**同一任务**里提交，否则竖轴平移/缩放时
            // 这些线比滚动条慢一帧到几帧（见 `paintNow`）。
            commitValueViewport("pitch", next, "immediate");
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [invalidate],
    );

    const paramViewsRef = useRef<Record<string, ValueViewport>>({});
    const paramAxisDirectionsRef = useRef<PanAxisDirections>(s.paramAxisDirections);
    paramAxisDirectionsRef.current = s.paramAxisDirections;
    const setParamViewport = useCallback(
        (param: string, next: ValueViewport) => {
            // 同 `setPitchView`：竖轴的手势提交点，DOM 与 GL 必须同任务。
            commitValueViewport(param as ParamName, next, "immediate");
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [invalidate],
    );

    function publishInteractionViewport(param: string, view: ValueViewport): void {
        const size = viewSizeRef.current;
        pianoRollInteractionViewportBus.publish({
            scrollLeftPx: scrollLeftRef.current,
            pxPerSec: pxPerSecRef.current,
            viewportWidthPx: size.w,
            viewportHeightPx: size.h,
            param,
            valueCenter: view.center,
            valueSpan: view.span,
            dpr: readDevicePixelRatio(),
        });
    }

    /**
     * Edge auto-scroll lands here for the value axis.
     *
     * Only refs / kernel writes; the current gesture's rAF (or the host frame
     * loop) owns the paint. The old path called setPitchView() -> paintNow()
     * on every driver tick, which is exactly the vertical jank.
     *
     * 【为什么只收「中心」而不是 scrollTop】值轴的积分器直接产出 Δ值（见
     * kernel/valueAxisEdgeScroller）。把中心再翻译成 1600px 的 scrollTop 代理域
     * 会让单帧步长在值域上失去上限 —— 那正是"上下滚动手感极差"的根因。
     */
    const applyValueScrollCenter = useCallback(
        (center: number) => {
            const param = editParamRef.current;
            const current = getCurrentViewportForScrollbar(param);
            commitValueViewport(param, { center, span: current.span }, "coalesced");
        },
        // All reads are refs / hoisted functions; a stable identity keeps the
        // edge driver and the interaction callback from rebuilding every frame.
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [],
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
            // ResizeObserver 只在挂载时注册，回调必须读**当前**参数轴（切换参数后
            // 再触发 resize 时，用 mount 时的 editParam 会把旧参数的 valueCenter /
            // valueSpan 广播到交互视口总线）。同文件 applyValueScrollCenter 的惯例。
            const param = editParamRef.current;
            publishInteractionViewport(param, getCurrentViewportForScrollbar(param));
        });
        ro.observe(el);
        return () => ro.disconnect();
        // 刻意只挂载一次：ResizeObserver 的回调只读 ref / 稳定总线，
        // publishInteractionViewport 与 getCurrentViewportForScrollbar 是每次
        // render 重建的普通函数，进 deps 会让缩放/参数切换时反复重注册 observer。
        // eslint-disable-next-line react-hooks/exhaustive-deps
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
            paramAxisDirectionsRef,
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
        // pan 方向：只改刻度文字/投影，不改内部值；必须触发网格镜像重建。
        editParamAxisDirection,
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

    const { valueToY, yToValue } = usePianoRollProjection({
        paramAxisDirectionsRef,
        paramViewsRef,
        pitchViewRef,
        processorParamsRef,
    });
    const yToViewportT = yToViewportTImpl;

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

    /** 值域视口 → 竖向滚动位置；提交前先刷新镜像 span（竖向缩放只改 ref，不触发镜像 effect）。 */
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
        authoritySnapshot: paramAuthoritySnapshot,
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
    /** Block spans whose analyzed window still contains the audible window. */
    const paramBlockSyllableSpans = useMemo(() => {
        const byClip = new Map<
            string,
            Array<{ startSec: number; endSec: number; note?: number | null }>
        >();
        for (const clip of paramBlockClips) {
            const activeTake =
                clip.takes?.find((take) => take.id === clip.activeTakeId) ?? clip.takes?.[0];
            const spans = activeTake?.syllableSpansSec ?? clip.syllableSpansSec;
            // Stale windows keep visible spans and only trigger reanalysis.
            // Stale/missing analysis still draws intersecting full-source events.
            if (spans?.length) byClip.set(clip.id, spans);
        }
        return byClip;
    }, [paramBlockClips]);
    useEffect(() => {
        for (const clip of paramBlockClips) {
            const take =
                clip.takes?.find((item) => item.id === clip.activeTakeId) ?? clip.takes?.[0];
            const analyzed = take?.syllableAnalysisWindowSec ?? clip.syllableAnalysisWindowSec;
            const spans = take?.syllableSpansSec ?? clip.syllableSpansSec;
            const audible = audibleSourceWindow(projectionForClip(toParamBlockSourceClip(clip)));
            if (analysisWindowStatus(audible, analyzed, spans) !== "current") {
                reportParamBlockDiagnostic({
                    code: "analysis_window_stale",
                    gesture: "layout",
                    rootTrackId,
                    blockId: null,
                    frame: null,
                    detail: JSON.stringify({
                        clipId: clip.id,
                        window: [audible.startSec, audible.endSec],
                        analyzed,
                    }),
                });
            }
        }
    }, [paramBlockClips, rootTrackId]);

    /** Drag-time loudness warp from the frozen geometry to the current one. */
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

    // Clip 源音高参考线：纯 UI 开关（默认开，不写入工程）。
    const [clipPitchOverlayVisible, setClipPitchOverlayVisible] = useState(true);
    const toggleClipPitchOverlay = useCallback(() => setClipPitchOverlayVisible((v) => !v), []);

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
        clipPitchOverlays,
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
        clipPitchOverlayVisible,
        currentParamDefaultValue,
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
    const paramBlockResizeRef = useRef<ParamBlockResizePreview | null>(null);
    /**
     * 新建块松手后的交接。预览保持到 React 块记录与本轮权威曲线刷新都落地，
     * 然后跨越一帧再撤下，避免“预览 -> 空一帧 -> 正式块”闪烁。
     */
    type ParamBlockCreateHandoff = {
        readonly blockId: string;
        /** 松手提交的时刻；`resolveParamBlockRelease` 用它算超时。 */
        readonly startedAt: number;
        readonly curveRevisionAtStart: number;
        authorityRequestSeqAtCommit: number;
        requiresCurveAuthority: boolean;
        /** 提交成功回包里的新后端曲线代次（预览释放的唯一证据）。 */
        curveRevisionCommitted?: number;
        committedSegments?: readonly { startFrame: number; values: readonly number[] }[];
        committed: boolean;
    };
    const paramBlockCreateCommitRef = useRef<ParamBlockCreateHandoff | null>(null);
    const paramBlockCreateReleaseRafRef = useRef<number | null>(null);
    /**
     * 边缘伸缩刚提交后的短前向快照。
     *
     * `applyParamBlockEdit` 通过 React state 提交，宿主却可能在同一次 pointerup
     * 后先绘一帧旧派生几何；用户会看到“松手闪回旧位置，再跳到新位置”。
     * 这里冻结刚提交的两个块区间，直到下一次派生真正包含它们。
     */
    const paramBlockVisualTransactionRef = useRef<ParamBlockVisualTransaction | null>(null);
    /**
     * 纵向块拖动提交后的交接。
     *
     * 松手提交期间，曲线的 live 值已经到位，但 `paramView` 快照可能晚一帧。
     * 若此刻清拖动快照，宿主会按旧主簇画一次，块就闪回原位。这里保留快照与
     * 松手时的显示中心，直到提交后的权威曲线窗口同步。
     */
    const paramBlockDragCommitRef = useRef<ParamBlockDragCommit | null>(null);
    const paramBlockDragCommitReleaseRafRef = useRef<number | null>(null);
    /** 块手势是否进行中；只用于保护曲线悬停读数的所有权。 */
    const isParamBlockGestureActive = useCallback(
        () =>
            paramBlockDragActivityRef.current.pointerGestureToken !== null ||
            paramBlockDragActivityRef.current.pendingCommits > 0 ||
            paramBlockResizeRef.current !== null,
        [],
    );
    /** Pointer activity and queued writes independently hold the refresh guard. */
    const paramBlockDragActivityRef = useRef<ParamBlockDragActivity>({
        pointerGestureToken: null,
        pendingCommits: 0,
    });
    const paramBlockGestureSerialRef = useRef(0);
    const paramBlockActiveGestureTokenRef = useRef<number | null>(null);
    /**
     * 刚刚按下的块（供双击的第二次 pointerdown 复用稳定 id）。
     *
     * 【为什么只需要 blockId + at】旧实现还存了一个自增 token 字符串，但没有任何
     * 读者 —— 双击判定只比较 id 与时间窗。
     */
    const paramBlockClickCandidateRef = useRef<{
        readonly blockId: string;
        readonly at: number;
    } | null>(null);
    /** Shift 范围选择的锚点；不进入 Redux / undo / 持久化。 */
    const paramBlockAnchorRef = useRef<ParamBlockSelectionAnchor | null>(null);
    const hoveredParamBlockIdRef = useRef<string | null>(null);
    const hoveredTransitionLeftIdRef = useRef<string | null>(null);
    // Stable callback that delegates to the latest handleEditOp via ref
    const stableEditAction = useCallback((op: string) => {
        handleEditActionRef.current(op);
    }, []);
    /** 块编辑动作在指针回调之后才定义；用 ref 打断声明顺序依赖。 */
    const applyParamBlockEditRef = useRef<
        ((result: ReturnType<typeof splitParamBlock>) => void) | null
    >(null);
    /**
     * 每个 run 最后一次有效的显示值。
     *
     * 参数窗口刷新是异步的：新窗口还没到时，canonical 块不能因为采样失败就掉回
     * 值域中点（pitch 下那正好是 F#4，所有块会叠成一条线）。缓存只用于显示，
     * 不写回后端；作用域切换时清空，结构变化时按现存 run 裁剪。
     */
    type ParamBlockCachedValue = {
        readonly value: number;
        readonly startSec: number;
        readonly endSec: number;
    };
    const paramBlockValueCacheRef = useRef<Map<string, ParamBlockCachedValue>>(new Map());
    const paramBlockValueCacheKeyRef = useRef<string | null>(null);
    /** 让宿主下一次绘制读到刚更新的拖动快照，而不是等某个 React 状态变化。 */
    const latestParamBlockSpecRef = useRef<PianoRollParamBlockSpec | null>(null);
    const invalidateParamBlockSpec = useCallback(() => {
        const transaction = paramBlockVisualTransactionRef.current;
        if (transaction !== null) {
            kernelDataRef.current.paramBlocks = transaction.spec;
            invalidate();
            return;
        }
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
            multiSelect: boolean;
            rangeSelect: boolean;
            pointerFrame?: (clientX: number) => number;
            clampSelectionFrame?: (frame: number) => number;
            edgeScroll?: EdgeScrollDriver;
            pointerType: string;
            nativeEvent: globalThis.PointerEvent;
            pointerId: number;
            captureTarget: HTMLCanvasElement;
            setActivePointerGestureEnd: (endGesture: () => void) => void;
            clearActivePointerGestureEnd: (endGesture?: () => void) => void;
            pointerValue: (clientY: number) => number;
            secAtClientX: (clientX: number) => number;
            altKey?: boolean;
            shiftKey?: boolean;
            ctrlKey?: boolean;
            metaKey?: boolean;
        }) => import("../gestures/pointerDeps").ParamBlockPointerDownResult
    >(() => ({ handled: false }));
    const splitParamBlockAtPlayheadRef = useRef<((snapToggleHeld?: boolean) => void) | null>(null);
    const splitParamBlockAtSecRef = useRef<
        | ((args: {
              readonly rawSec: number;
              readonly targetBlockId?: string;
              readonly snapToggleHeld?: boolean;
              readonly movePlayhead?: boolean;
              readonly selectResult?: boolean;
              readonly snapMode?: "none" | "grid";
          }) => boolean)
        | null
    >(null);
    const suppressBlockDoubleClickUntilRef = useRef(0);
    const deleteSelectedParamBlocksRef = useRef<(() => void) | null>(null);
    const groupSelectedParamBlocksRef = useRef<(() => void) | null>(null);
    const ungroupSelectedParamBlocksRef = useRef<(() => void) | null>(null);
    const restoreParamBlocksRef = useRef<(() => void) | null>(null);
    const restoreGameParamBlocksRef = useRef<(() => void) | null>(null);
    const normalizeSelectedParamBlockPitchRef = useRef<(() => void) | null>(null);
    const createParamBlocksFromSelectionRef = useRef<(() => void) | null>(null);
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

    /**
     * 参数值浮窗的唯一入口（合帧后 setState）。
     *
     * 【为什么提成具名常量】它同时喂给 usePianoRollInteractions（参数线 / 颤动 / 四点
     * 编辑）与块手势（拖块 / 新建）。内联在 interactions 的入参对象里，块手势就只能
     * 另写一份 setState —— 那会绕过合帧，并且出现两个写入者。
     */
    const publishParamValuePreview = useParamValuePreviewPublisher({
        canvasRef,
        setPreview: setParamValuePreview,
    });
    const interactions = usePianoRollInteractions({
        dispatch,
        rootTrackId,
        editParam,
        pitchEnabled,
        toolMode: s.toolMode,
        paramBlockToolActive,
        isParamBlockGestureActive,
        rangeToolActive,
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
        paramAxisDirectionsRef,
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
        applyValueScrollCenter,
        invalidate,
        yToViewportT,
        yToValue,
        valueToY,
        clampViewport,
        getCurrentViewportForScrollbar,
        getParamValueBoundsForScrollbar,
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
        paramRangeSelectKb,
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
        }, [setVibratoDragHud]),
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
        onParamValuePreviewChange: publishParamValuePreview,
    });
    const beginParamBlockMorphRef = useRef(interactions.beginParamBlockMorph);
    beginParamBlockMorphRef.current = interactions.beginParamBlockMorph;

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
        paramAxisDirectionsRef,
        editParamAxisDirection,
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
    /**
     * 曲线被参数命令改写之后，pitch 块的持久化中心已经过期。
     *
     * 只清 `value` 并按**布局变更**提交（checkpoint=false）：用户撤销一次就回到
     * 曲线命令之前，不会被"清中心"多插一个撤销步。
     *
     * 【名字说的是「清中心」，不是「刷新显示值」】旧名 `refreshParamBlockValuesAfterParamEdit`
     * 让人以为它只重算显示，于是很容易被当成纯刷新调用 —— 它其实会写一次编辑集。
     */
    const clearStaleBlockCentersAfterParamEdit = useCallback(
        (_op: string) => {
            if (editParam !== "pitch") return;
            const state = paramBlockStateRef.current;
            if (state.editSet.blocks.length === 0) return;
            const next = clearParamBlockValues(state);
            if (next === state) return;
            setParamBlockState(next);
            void blockPersistence.commit(next, false).catch(() => undefined);
            bumpRefreshToken();
            invalidate();
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps -- blockPersistence 是稳定 facade；只读其 commit。
        [bumpRefreshToken, editParam, invalidate],
    );
    const { selectAllParamRange, handleEditOp } = usePianoRollEditCommands({
        activeVibratoPresetId,
        bumpRefreshToken,
        clipboardRef,
        currentParamDefaultValue,
        currentParamQuantizeUnit,
        currentParamRange,
        dispatch,
        dynamicProjectSec,
        paramBlockToolActive,
        paramBlockSelectAllRanges: () => {
            const view = paramBlockViewRef.current;
            return view === null ? null : selectAllParamBlockRanges(view.blocks);
        },
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
        onParamEditApplied: clearStaleBlockCentersAfterParamEdit,
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

    /** 块层是否绘制：唯一派生点是 isParamBlockLayerVisible；面板要的是「现在画不画」。 */
    const alwaysShowParamBlocks = s.alwaysShowParamBlocks;
    const paramBlocksLayerVisible = isParamBlockLayerVisible({
        alwaysShowParamBlocks,
        toolMode: s.toolMode,
        selectToolMode: s.selectToolMode,
    });
    useLayoutEffect(() => {
        kernelDataRef.current.paramBlocksVisible = paramBlocksLayerVisible;
        hostRef.current?.invalidate();
    }, [paramBlocksLayerVisible]);
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
    const {
        state: paramBlockState,
        stateRef: paramBlockStateRef,
        setState: setParamBlockState,
    } = useParamBlockState(EMPTY_PARAM_BLOCK_STATE);
    /** Explicit history jumps only; self commits advance history depth too. */
    const [paramBlockExternalHistoryToken, setParamBlockExternalHistoryToken] = useState(0);
    /** Incremented after every successful parameter-block commit. */
    useEffect(() => {
        const onHistoryJump = () => setParamBlockExternalHistoryToken((value) => value + 1);
        window.addEventListener(HISTORY_JUMP_EVENT, onHistoryJump);
        return () => window.removeEventListener(HISTORY_JUMP_EVENT, onHistoryJump);
    }, []);
    /*
     * 【几何只有一个权威】块视图是 `(clips, editSet, framePeriodMs)` 的纯函数。
     *
     * 它**不落状态**：落下来就有两个写入者（本地合成与后端回读），而回读在手势
     * 期间拿到的是旧几何 —— 屏幕上的块于是逐帧来回跳，切分 / 伸缩也读到过期边界
     * （"切分点离边缘太近"）。派生值只该被现算。
     */
    const paramBlockSourceClipsLive = useMemo(
        () => paramBlockClips.map(toParamBlockSourceClip),
        [paramBlockClips],
    );
    /*
     * 【手势期间冻结块几何】块是参数线的分段，两者必须同源。而手势期间 clip 只做
     * Redux 乐观更新、参数线要等落库才重取 —— 不冻结就会出现"块跟着 Clip 走、
     * 线原地不动"（块从曲线上滑走）。
     *
     * 快照来自 `clipGeometryPreviewBus`：它在**任何乐观写入之前**建立（交互锁却
     * 要等到首个位移帧才 +1，读 `s.clips` 会漏掉第一帧的位移）。松手后发布方
     * 清空快照，块层在下一帧就跟上权威几何 —— 与 Clip 图层换新同步，只有一次跳变。
     */
    const [frozenGestureClips, setFrozenGestureClips] = useState<
        readonly ParamBlockSourceClip[] | null
    >(null);
    useEffect(() => {
        const sync = () => {
            const clips = getClipGeometryPreviewClips<ClipInfo>();
            setFrozenGestureClips(clips === null ? null : clips.map(toParamBlockSourceClip));
            // 块层换了一份几何 ⇒ 必须重绘（宿主不会因为 state 变化自己提交帧）。
            invalidate();
        };
        // 订阅不补发当前状态（见总线的说明），所以先同步一次再订阅。
        sync();
        return subscribeClipGeometryPreview(sync);
    }, [invalidate]);
    const paramBlockSourceClips = useMemo(
        () => resolveParamBlockGeometryClips(paramBlockSourceClipsLive, frozenGestureClips),
        [paramBlockSourceClipsLive, frozenGestureClips],
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
    const paramBlockEpoch = useMemo<ParamBlockEpoch>(
        () => ({
            rootTrackId,
            param: editParam,
            layoutKey: paramBlockView?.layoutKey ?? null,
            /*
             * 结构令牌取代 revision：revision 前端是内容代次、后端是 CAS 代次，
             * 不可比；而预览只关心「块的形状有没有被第三方改过」。
             */
            structureToken: paramBlockStructureToken(paramBlockState.editSet),
            externalHistoryToken: `${rootTrackId ?? ""}:${paramBlockExternalHistoryToken}`,
            framePeriodMs: paramBlockState.framePeriodMs,
        }),
        [
            rootTrackId,
            editParam,
            paramBlockView?.layoutKey,
            paramBlockState.editSet,
            paramBlockState.framePeriodMs,
            paramBlockExternalHistoryToken,
        ],
    );
    const paramBlockEpochRef = useLatestRef(paramBlockEpoch);
    /** Clip snapshot used by editing functions to split/merge source spans. */
    const paramBlockSourceClipsRef = useRef(paramBlockSourceClips);
    useLayoutEffect(() => {
        paramBlockSourceClipsRef.current = paramBlockSourceClips;
    }, [paramBlockSourceClips]);
    /** Recompose from the live edit set, never a stale view ref. */
    const composeParamBlockViewNow = useCallback((): ParamBlockView | null => {
        if (rootTrackId === null) return null;
        const state = paramBlockStateRef.current;
        return composeParamBlockView(
            rootTrackId,
            paramBlockSourceClipsRef.current,
            state.editSet,
            state.framePeriodMs,
        );
    }, [paramBlockSourceClipsRef, paramBlockStateRef, rootTrackId]);

    /** Logical ids fully covered by the current selection, from live geometry. */
    const selectedParamBlockIdsNow = useCallback((): string[] => {
        const view = composeParamBlockViewNow();
        if (view === null) return [];
        return paramBlockIdsInSelection(
            view.blocks.map((block) => ({
                id: block.id,
                ...(block.userBlockId === undefined ? {} : { userBlockId: block.userBlockId }),
                startFrame: block.startFrame,
                endFrameExclusive: block.endFrameExclusive,
            })),
            selectionRef.current,
        );
    }, [composeParamBlockViewNow, selectionRef]);

    /*
     * 块选择不是第二份状态：selectedParamBlockIds 由 selectionUi 在块层投影派生。
     */
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
    } | null>(null);
    /*
     * 【一个根只有一个权威】根切换后，旧根的 editSet / 选择 / 删除快照绝不能
     * 进入新根。持久化钩子的 ready 判据是 rootTrackId + state.rootTrackId，
     * 因此这里必须同步清空 state；否则第一次按 S 会把 A 根的编辑集提交到 B 根。
     *
     * projectEpoch 目前没有独立字段：打开工程时根轨道几乎总会变化；若未来加入
     * 工程代次，把它拼进 authorityKey 即可，不影响其余逻辑。
     */
    const rootTrackIdRef = useLatestRef(rootTrackId);
    const paramBlockAuthorityKey = rootTrackId ?? "";
    const lastParamBlockAuthorityRef = useRef(paramBlockAuthorityKey);
    useEffect(() => {
        // 外部布局 / 历史变化后，右键菜单里的旧块快照已不再可信。
        setParamBlockContextMenu(null);
    }, [
        paramBlockEpoch.layoutKey,
        paramBlockEpoch.externalHistoryToken,
        paramBlockEpoch.rootTrackId,
    ]);
    useEffect(() => {
        // Any external epoch change must not leave a guide from the old view.
        clearParamBlockSnapHighlight();
    }, [
        paramBlockEpoch.layoutKey,
        paramBlockEpoch.externalHistoryToken,
        paramBlockEpoch.rootTrackId,
    ]);
    useEffect(() => {
        // Undo/redo can invalidate a live preview even when no drag snapshot
        // exists anymore. Clear the override as well as the interaction curve.
        clearLiveEditOverride();
        paramBlockInteractionCurveRef.current = null;
        if (
            liveEditActiveRef.current &&
            paramBlockDragActivityRef.current.pointerGestureToken === null &&
            paramBlockDragActivityRef.current.pendingCommits === 0 &&
            paramBlockDragCommitRef.current === null
        ) {
            liveEditActiveRef.current = false;
        }
    }, [
        paramBlockExternalHistoryToken,
        clearLiveEditOverride,
        liveEditActiveRef,
        paramBlockDragCommitRef,
        paramBlockInteractionCurveRef,
    ]);
    useEffect(() => {
        const clear = () => clearParamBlockSnapHighlight();
        const onVisibility = () => {
            if (document.visibilityState === "hidden") clear();
        };
        window.addEventListener("blur", clear);
        document.addEventListener("visibilitychange", onVisibility);
        return () => {
            window.removeEventListener("blur", clear);
            document.removeEventListener("visibilitychange", onVisibility);
        };
    }, []);
    /**
     * The only writer for parameter-block selection.
     *
     * Keeping selectionRef and selectionUi in one function removes the window
     * where a command reads the ref while the UI still shows the old range (or
     * vice versa). It also keeps session.paramSelectionActive in sync for
     * immediate Ctrl+C / Ctrl+X after a block click.
     */
    const setParamBlockSelection = useCallback(
        (next: ParamSelection | null) => {
            selectionRef.current = next;
            setSelectionUi(next);
            dispatch(setParamSelectionActive(next !== null));
            invalidate();
        },
        [dispatch, invalidate, selectionRef, setSelectionUi],
    );
    useEffect(() => {
        if (lastParamBlockAuthorityRef.current === paramBlockAuthorityKey) return;
        lastParamBlockAuthorityRef.current = paramBlockAuthorityKey;
        setParamBlockState(EMPTY_PARAM_BLOCK_STATE);
        paramBlockVisualTransactionRef.current = null;
        paramBlockResizeRef.current = null;
        paramBlockAnchorRef.current = null;
        setParamBlockSelection(null);
        setParamBlockContextMenu(null);
    }, [paramBlockAuthorityKey, setParamBlockSelection, setParamBlockState]);
    const clearParamBlockSelection = useCallback(() => {
        paramBlockAnchorRef.current = null;
        setParamBlockSelection(null);
        setParamBlockContextMenu(null);
    }, [setParamBlockSelection]);
    const previousBlockToolActiveRef = useRef(paramBlockToolActive);
    useEffect(() => {
        if (previousBlockToolActiveRef.current === paramBlockToolActive) return;
        previousBlockToolActiveRef.current = paramBlockToolActive;
        setParamBlockContextMenu(null);
        /*
         * 选区属于参数编辑器，不随工具切换消失。进入参数块选择模式时把自由选区
         * 规整到被碰到的逻辑块；离开只关闭块菜单，块高亮由工具门控隐藏。
         */
        if (paramBlockToolActive) {
            const next = expandParamBlockSelection(
                paramBlockViewRef.current?.blocks ?? [],
                selectionRef.current,
            );
            if (next.length > 0) {
                setParamBlockSelection(next);
            }
        }
        invalidate();
    }, [invalidate, paramBlockToolActive, selectionRef, setParamBlockSelection]);
    const toggleAlwaysShowParamBlocks = useCallback(() => {
        dispatch(toggleAlwaysShowParamBlocksAction());
        void dispatch(persistUiSettings());
    }, [dispatch]);
    /** Per-session auto-analysis attempts; manual refresh is never blocked. */
    const autoAnalysisAttemptsRef = useRef<Set<string>>(new Set());
    /** 上一轮"有音节区间"的 Clip 集合；用于发现分析结果被意外清空。 */
    const analyzedClipIdsRef = useRef<Set<string>>(new Set());
    const autoAnalysisQueueRef = useRef(createSyllableAnalysisQueue());
    const autoAnalysisRootRef = useRef<string | null>(null);
    useEffect(() => {
        autoAnalysisRootRef.current = paramBlocksLayerVisible && pitchEnabled ? rootTrackId : null;
        return () => {
            if (autoAnalysisRootRef.current !== null) publishBlockAnalysisProgress(null);
            autoAnalysisRootRef.current = null;
        };
    }, [paramBlocksLayerVisible, pitchEnabled, rootTrackId]);
    /** Geometry/window changes create a fresh request key; stale results stay visible. */
    useEffect(() => {
        const current = new Set(paramBlockSyllableSpans.keys());
        for (const clipId of analyzedClipIdsRef.current) {
            if (!current.has(clipId)) autoAnalysisAttemptsRef.current.delete(clipId);
        }
        analyzedClipIdsRef.current = current;
    }, [paramBlockSyllableSpans]);
    /**
     * 历史位置镜像。块编辑的 persist 写入会打后端撤销点，但 React 的
     * `paramBlockState` 不会自动回读；只在历史深度变化时 reload，避免普通
     * 重渲染或拖动中的每次进度提交都发一轮 IPC。
     */
    const blockPersistence = useParamBlockPersistence({
        rootTrackId,
        // 块层在画 = 块选择模式或常显设置开着；两者都由 isParamBlockLayerVisible 派生。
        enabled: paramBlocksLayerVisible,
        onCommitted: () => {
            dispatch(markProjectEdited());
        },
        /*
         * 只跟历史位置走。clip 几何不进这个键：块视图是派生的，clip 一变就
         * 自动跟上；把它拼进来会让一次 Clip 拖动每帧发一轮回读，而那时的回包
         * 带的是旧几何 —— 闪烁就是这么来的。
         */
        historyKey: `${s.historyUndoDepth}:${s.historyRedoDepth}`,
        state: paramBlockState,
        setState: setParamBlockState,
        /* 写入失败按**类别**反馈：rejected / conflict / transport / silent。 */
        onWriteError: (failure, error) => {
            publishParamBlockWriteFailure({ failure, error, tf });
        },
        /*
         * 【只挡参数块手势】deferral 的唯一职责是保护**正在进行中的本地编辑集**
         * ——块拖动 / 过渡手柄逐帧改它，回包此刻落地会把预览打断。
         *
         * 时间轴内核手势（移动 / 裁剪 / Slip / 淡化 / 增益）**不算**：它不改编辑集，
         * 而几何早已是派生值（回包不携带几何，也就无从"带回旧几何"）。反过来把
         * 它挡掉是有害的 —— 撤销之后、手势仍在进行时权威编辑集回不来，镜像留在
         * 旧代次，下一次块编辑必定被 CAS 拒绝（"点了没反应"）。
         */
        deferReadback: () =>
            paramBlockDragActivityRef.current.pointerGestureToken !== null ||
            paramBlockDragActivityRef.current.pendingCommits > 0,
    });
    /*
     * 过渡提交拆在专用 hook 里：布局、曲线补丁、权威 watermark 与视觉预览
     * 归属必须一起处理，集中在一处避免再次漏传 authority。
     */
    const { persistBlockTransition, persistPitchTransitionCurve } =
        useParamBlockTransitionPersistence({
            rootTrackId,
            editParam,
            blockPersistence,
            paramBlockStateRef,
            paramBlockViewRef,
            paramBlockSourceClipsRef,
            paramViewRef,
            setParamBlockState,
            setParamView,
            onParamCommitSucceeded,
            bumpRefreshToken,
            invalidate,
            stampCurveCommit: ({ segments, authority }) => {
                const tx = paramBlockVisualTransactionRef.current;
                if (tx === null) return;
                tx.curveRevisionAtStart = authority?.expectedCurveRevision;
                // 回包里的新代次由 stampCurveCommitted 在提交成功后补上。
                tx.curveRevisionCommitted = undefined;
                tx.authorityRequestSeqAtCommit = getAuthorityWatermark().issuedRequestSeq;
                tx.committedSegments = segments;
            },
            stampCurveCommitted: (curveRevision) => {
                const tx = paramBlockVisualTransactionRef.current;
                if (tx === null) return;
                tx.curveRevisionCommitted = curveRevision;
            },
            onLocalStructureCommitted: (edits) => {
                paramBlockVisualTransactionRef.current = adoptParamBlockVisualStructureAfterEdits(
                    paramBlockVisualTransactionRef.current,
                    edits,
                );
            },
        });

    /**
     * The one exit door for every visual preview.
     *
     * Structural commands, history jumps, layout changes and handoff timeouts
     * all come through here. Clearing all refs together is deliberate: a
     * surviving interaction curve outranks `paramView` forever after this point.
     */
    const invalidateParamBlockVisualPreviews = useCallback(
        (reason: "structure" | "history" | "layout" | "timeout" | "error") => {
            const activity = paramBlockDragActivityRef.current;
            const hadVisual =
                paramBlockVisualTransactionRef.current !== null ||
                paramBlockResizeRef.current !== null ||
                paramBlockCreateCommitRef.current !== null ||
                paramBlockDragSnapshotRef.current !== null ||
                paramBlockDragCommitRef.current !== null ||
                paramBlockInteractionCurveRef.current !== null ||
                paramBlockActiveGestureTokenRef.current !== null;
            paramBlockVisualTransactionRef.current = null;
            paramBlockResizeRef.current = null;
            paramBlockCreateCommitRef.current = null;
            paramBlockDragSnapshotRef.current = null;
            paramBlockDragCommitRef.current = null;
            paramBlockInteractionCurveRef.current = null;
            paramBlockActiveGestureTokenRef.current = null;
            activity.pointerGestureToken = null;
            if (paramBlockCreateReleaseRafRef.current !== null) {
                cancelAnimationFrame(paramBlockCreateReleaseRafRef.current);
                paramBlockCreateReleaseRafRef.current = null;
            }
            if (paramBlockDragCommitReleaseRafRef.current !== null) {
                cancelAnimationFrame(paramBlockDragCommitReleaseRafRef.current);
                paramBlockDragCommitReleaseRafRef.current = null;
            }
            if (liveEditActiveRef) {
                liveEditActiveRef.current = activity.pendingCommits > 0;
            }
            clearLiveEditOverride();
            const pvNow = paramViewRef.current;
            if (pvNow !== null) resetLiveEditPreview(pvNow);
            clearParamBlockSnapHighlight();
            if (hadVisual) bumpParamBlockStructure();
            invalidate();
            void reason;
        },
        [
            bumpParamBlockStructure,
            clearLiveEditOverride,
            invalidate,
            liveEditActiveRef,
            paramViewRef,
            resetLiveEditPreview,
        ],
    );

    /*
     * 外部 epoch 变化（Clip 编辑 / 撤销 / 根切换 / 参数窗口变化）必须在绘制前
     * 撤下所有视觉事务。清理放在 layout effect（早于 paint），所以不会先闪一帧。
     */
    useLayoutEffect(() => {
        const tx = paramBlockVisualTransactionRef.current;
        const txMismatch =
            tx !== null &&
            !isParamBlockVisualEpochCurrent({
                epoch: tx.epoch,
                current: paramBlockEpoch,
            });
        const dragSnapshot = paramBlockDragSnapshotRef.current;
        const dragMismatch =
            dragSnapshot !== null &&
            !isParamBlockVisualEpochCurrent({
                epoch: dragSnapshot.epoch,
                current: paramBlockEpoch,
            });
        const resizing = paramBlockResizeRef.current;
        const resizeMismatch =
            resizing?.epoch !== undefined &&
            !isParamBlockVisualEpochCurrent({
                epoch: resizing.epoch,
                current: paramBlockEpoch,
            });
        if (!txMismatch && !dragMismatch && !resizeMismatch) return;
        invalidateParamBlockVisualPreviews("layout");
    }, [invalidateParamBlockVisualPreviews, paramBlockEpoch]);
    /*
     * 持久化合并不在组件状态里：从工程参数读取，写入时走带撤销点的命令。
     * 只在根轨道切换 / 历史推进时刷新；拖动块本身不触发网络请求。
     */
    useEffect(() => {
        if (!rootTrackId) return;
        /*
         * 自动音节分析只在「块层真的在画」时跑：块是参数线上按音节切出来的段，
         * 没有分析结果就没有块。图层可见 = 常显设置开着，或当前就是块选择模式。
         * 分析会跑模型推理，所以这是「常开图层」唯一的代价 —— 也是它默认关闭的理由。
         */
        const blocksEnabled = paramBlocksLayerVisible && blockPersistence.ready;
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
        const { clips: audioClips, keysByClipId: staleKeys } = paramBlockStaleAnalysisCandidates({
            clips: blocksEnabled && pitchEnabled ? paramBlockClips : [],
            rootTrackId,
            framePeriodMs: paramBlockView?.framePeriodMs ?? 5,
            attemptedKeys: autoAnalysisAttemptsRef.current,
        });
        for (const key of staleKeys.values()) autoAnalysisAttemptsRef.current.add(key);
        if (audioClips.length === 0) return;
        void autoAnalysisQueueRef.current
            .runBatch({
                clips: audioClips,
                shouldStart: () => autoAnalysisRootRef.current === rootTrackId,
                onSkipped: (clip) => {
                    const key = staleKeys.get(clip.id);
                    if (key !== undefined) autoAnalysisAttemptsRef.current.delete(key);
                },
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
                for (const result of results) {
                    if (!result.analyzed) {
                        const key = staleKeys.get(result.clipId);
                        if (key !== undefined) autoAnalysisAttemptsRef.current.delete(key);
                    }
                }
            });
        // eslint-disable-next-line react-hooks/exhaustive-deps -- blockPersistence 是稳定 facade；依赖其具体方法。
    }, [
        bumpRefreshToken,
        dispatch,
        paramBlockClips,
        paramBlockSyllableSpans,
        paramBlocksLayerVisible,
        pitchEnabled,
        rootTrackId,
        s.paramsEpoch,
        blockPersistence.ready,
        blockPersistence.receive,
    ]);
    const requireBlockPersistenceReady = useCallback((): boolean => {
        if (blockPersistence.ready) return true;
        publishParamBlockFeedback({
            text: tf("status_param_block_not_ready"),
            tone: "warning",
            holdMs: 5000,
        });
        return false;
    }, [blockPersistence.ready, tf]);
    const persistParamBlockEdits = useCallback(
        (state: ParamBlockState) => {
            void blockPersistence.persist(state);
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps -- blockPersistence 是稳定 facade；只读其 persist 方法。
        [blockPersistence.persist],
    );
    /** Panel blocks = derived view + current parameter curve. */
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
        const cacheKey = `${rootTrackId ?? ""}|${editParam}`;
        const previousValues =
            paramBlockValueCacheKeyRef.current === cacheKey
                ? paramBlockValueCacheRef.current
                : undefined;
        // Fallbacks are display-only; never borrow the viewport midpoint.
        return resolveParamBlockPanelValues({
            blocks: paramBlocksForPanel,
            param: editParam,
            paramView,
            fallbackValue: 0,
            previousValues,
            allowFallback: false,
            currentParamDefaultValue,
        });
    }, [currentParamDefaultValue, editParam, paramBlocksForPanel, paramView, rootTrackId]);
    useLayoutEffect(() => {
        const cacheKey = `${rootTrackId ?? ""}|${editParam}`;
        const currentIds = new Set(paramBlocksForPanel.map((block) => logicalBlockKey(block)));
        const next = new Map<string, ParamBlockCachedValue>();
        if (paramBlockValueCacheKeyRef.current === cacheKey) {
            for (const [id, cached] of paramBlockValueCacheRef.current) {
                if (currentIds.has(id) && Number.isFinite(cached.value)) {
                    next.set(id, cached);
                }
            }
        }
        for (const block of paramBlocksWithValues) {
            if (
                currentIds.has(block.id) &&
                block.averageValue !== null &&
                Number.isFinite(block.averageValue)
            ) {
                next.set(logicalBlockKey(block), {
                    value: block.averageValue,
                    startSec: block.startSec,
                    endSec: block.endSec,
                });
            }
        }
        paramBlockValueCacheKeyRef.current = cacheKey;
        paramBlockValueCacheRef.current = next;
    }, [editParam, paramBlocksForPanel, paramBlocksWithValues, rootTrackId]);
    const paramBlocksWithValuesRef = useRef(paramBlocksWithValues);
    useLayoutEffect(() => {
        paramBlocksWithValuesRef.current = paramBlocksWithValues;
    }, [paramBlocksWithValues]);
    /** 块选区 = 参数选区在块层的投影；多 run 逻辑块的全部片段都参与。 */
    const selectionFrameBlocks = useMemo(() => {
        /*
         * 视图的帧周期才是当前渲染 / 命中使用的周期；用 state 的周期会在
         * 视图切换后的一个提交间隙里把选区投影偏一帧。回退到 state 只覆盖
         * 首次尚未合成视图的启动帧。
         */
        const framePeriodMs = paramBlockView?.framePeriodMs ?? paramBlockState.framePeriodMs;
        return paramBlocksWithValues.map((block) => ({
            id: block.id,
            ...(block.userBlockId === undefined ? {} : { userBlockId: block.userBlockId }),
            startFrame: Math.round((block.startSec * 1000) / Math.max(0.1, framePeriodMs)),
            endFrameExclusive: Math.round((block.endSec * 1000) / Math.max(0.1, framePeriodMs)),
        }));
    }, [paramBlocksWithValues, paramBlockState.framePeriodMs, paramBlockView?.framePeriodMs]);
    const selectedParamBlockIds = useMemo(
        () =>
            paramBlockToolActive ? paramBlockIdsInSelection(selectionFrameBlocks, selectionUi) : [],
        [paramBlockToolActive, selectionFrameBlocks, selectionUi],
    );
    useEffect(() => {
        if (!paramBlockToolActive || selectionUi === null) return;
        if (paramBlockAnchorRef.current !== null) return;
        const ids = paramBlockIdsInSelection(selectionFrameBlocks, selectionUi);
        const first = selectionFrameBlocks.find((block) => ids.includes(logicalBlockId(block)));
        if (first !== undefined) {
            paramBlockAnchorRef.current = {
                blockId: logicalBlockId(first),
                frame: first.startFrame,
            };
        }
    }, [paramBlockToolActive, selectionFrameBlocks, selectionUi]);
    const activeSelectedParamBlockIds = selectedParamBlockIds;
    const bounds = getParamValueBoundsForScrollbar(editParam);
    const paramBlockSpecColors = useMemo<ParamBlockSpecColors>(
        () => ({
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
        }),
        [themeMode],
    );
    const selectedParamBlockIdSet = useMemo(
        () => new Set(activeSelectedParamBlockIds),
        [activeSelectedParamBlockIds],
    );
    /**
     * 渲染用的块选中集合。
     *
     * 选区本身跨工具保留（换工具不清空选择，这是既有设计）；但「选中」在画面上是一组
     * **交互 affordance**（高亮填充、过渡手柄、可拖光标）。非块模式下块层只负责显示，
     * 因此这些 affordance 必须关掉 —— 否则用户看到的是「能拖的样子、拖不动的结果」。
     */
    const paramBlockRenderSelectedIds = useMemo(
        () => (paramBlockToolActive ? selectedParamBlockIdSet : EMPTY_BLOCK_ID_SET),
        [paramBlockToolActive, selectedParamBlockIdSet],
    );
    const paramBlockSpecForDragSnapshot = useCallback(
        (snapshot: ParamBlockDragSnapshot): PianoRollParamBlockSpec | null => {
            if (!paramBlocksLayerVisible) return null;
            if (snapshot.blocks.length === 0) return null;
            const draggedById = new Map<string, (typeof snapshot.blocks)[number]>();
            for (const draggedBlock of snapshot.blocks) {
                draggedById.set(draggedBlock.id, draggedBlock);
                if (draggedBlock.userBlockId !== undefined) {
                    draggedById.set(draggedBlock.userBlockId, draggedBlock);
                }
            }
            const blocks: ParamBlockRenderBlock[] = paramBlocksWithValues.map((block) => {
                const dragged = snapshot.ids.has(logicalBlockId(block))
                    ? (draggedById.get(block.id) ??
                      (block.userBlockId === undefined
                          ? undefined
                          : draggedById.get(block.userBlockId)))
                    : undefined;
                return {
                    id: block.id,
                    userBlockId: block.userBlockId,
                    startSec: dragged?.startSec ?? block.startSec,
                    endSec: dragged?.endSec ?? block.endSec,
                    averageValue: dragged?.anchorValue ?? block.averageValue,
                    logicalCenterValue: block.manualValue,
                    frameCenterValue:
                        dragged === undefined
                            ? undefined
                            : (snapshot.frameCenters.get(block.id) ?? null),
                    displayCenterSnapToSemitone:
                        dragged === undefined ? block.manualValue == null : false,
                    selected: dragged !== undefined,
                    grouped: block.kind === "grouped",
                    transitionSec: block.transitionSec,
                    centerPolicy: dragged === undefined ? "live" : "locked",
                };
            });
            return buildParamBlockRenderSpec({
                blocks,
                param: editParam,
                valueMin: bounds.min,
                valueMax: bounds.max,
                valueToY: (value, heightPx) => valueToY(editParam, value, heightPx),
                colors: paramBlockSpecColors,
                curveOwned: paramBlockViewRef.current !== null,
            });
        },
        [
            bounds.max,
            bounds.min,
            editParam,
            paramBlockSpecColors,
            paramBlocksLayerVisible,
            paramBlocksWithValues,
            valueToY,
        ],
    );
    useLayoutEffect(() => {
        paramBlockSpecForDragSnapshotRef.current = paramBlockSpecForDragSnapshot;
    }, [paramBlockSpecForDragSnapshot]);
    const publishParamBlockVisualTransaction = useCallback(
        (kind: ParamBlockVisualTransactionKind, spec: PianoRollParamBlockSpec) => {
            const previous = paramBlockVisualTransactionRef.current;
            const sameGesture =
                previous !== null &&
                previous.kind === kind &&
                sameParamBlockEpochLayout(previous.epoch, paramBlockEpochRef.current);
            paramBlockVisualTransactionRef.current = {
                id: sameGesture ? previous.id : ++paramBlockGestureSerialRef.current,
                epoch: paramBlockEpochRef.current,
                kind,
                spec,
                startedAt: sameGesture ? previous.startedAt : performance.now(),
            };
        },
        [paramBlockEpochRef],
    );
    const beginBlockDrag = useCallback(
        (args: ParamBlockDragStart) => {
            const anchorBlock = paramBlocksWithValues.find(
                (block) =>
                    block.id === args.anchorBlockId || logicalBlockId(block) === args.anchorBlockId,
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
                paramBlockEpoch,
                getCurrentParamBlockEpoch: () => paramBlockEpochRef.current,
                curveOwned: paramBlockViewRef.current !== null,
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
                 * 分成两次写就会让一次拖拽需要按两次撤销。plan 的 prepare 里既算
                 * 最终编辑集（含 Pitch 块的逻辑中心）也算曲线补丁。
                 *
                 * 【为什么交给 commitPlan 而不是 commit】prepare 会**读基准**
                 * （全分辨率曲线 + 权威代次），它必须和写入在同一个队列槽里；
                 * 冲突时写队列会再调一次 prepare 完成重基。
                 */
                commitBlockWrite: (plan) => blockPersistence.commitPlan(plan),
                getAuthorityWatermark,
                authorityCurveRevision: paramAuthoritySnapshot.curveRevision,
                onParamCommitSucceeded,
                bumpRefreshToken,
                paramValuePopupEnabled: s.showParamValuePopup,
                onParamValuePreviewChange: publishParamValuePreview,
                paintNow: () => hostRef.current?.paintNow(),
                /*
                 * 写入失败已经由持久化层按类别上报过（rejected / conflict /
                 * transport / silent）。这里再发一条只会把分类结果覆盖成一句通用
                 * 红条 —— 只留结构化诊断。
                 */
                onError: (error) => {
                    reportParamBlockWriteFailure({ error, gesture: "drag", rootTrackId });
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
            blockPersistence,
            paramBlockEpochRef,
            paramBlockStateRef,
            effectiveProjectScale,
            getAuthorityWatermark,
            invalidateParamBlockSpec,
            liveEditActiveRef,
            mergedKeybindings,
            onParamCommitSucceeded,
            paramBlockEpoch,
            paramBlocksWithValues,
            paramView,
            paramAuthoritySnapshot,
            publishParamValuePreview,
            projectScaleAtSec,
            s.showParamValuePopup,
            requestWaveformRepaint,
            resetLiveEditPreview,
            rootTrackId,
            s.pitchSnapEnabled,
            s.pitchSnapToleranceCents,
            s.pitchSnapUnit,
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
            multiSelect: boolean;
            rangeSelect: boolean;
            pointerFrame?: (clientX: number) => number;
            clampSelectionFrame?: (frame: number) => number;
            edgeScroll?: EdgeScrollDriver;
            pointerType: string;
            nativeEvent: globalThis.PointerEvent;
            pointerId: number;
            captureTarget: HTMLCanvasElement;
            setActivePointerGestureEnd: (endGesture: () => void) => void;
            clearActivePointerGestureEnd: (endGesture?: () => void) => void;
            pointerValue: (clientY: number) => number;
            secAtClientX: (clientX: number) => number;
            altKey?: boolean;
            shiftKey?: boolean;
            ctrlKey?: boolean;
            metaKey?: boolean;
        }): import("../gestures/pointerDeps").ParamBlockPointerDownResult => {
            if (!paramBlockToolActive) return { handled: false };
            // 新手势优先：上一笔等待回读的视觉事务不能盖住本次按下。
            paramBlockVisualTransactionRef.current = null;
            /*
             * 块层未就绪 = 此刻做不了任何块操作。返回 `handled: false`，让指针
             * 落进参数块选择模式的空白分支（按设置移动播放头）—— 用户看到的是"点了一下、
             * 播放头动了"，而不是一条要读三秒的提示。
             */
            if (!blockPersistence.ready || !paramViewRef.current) return { handled: false };
            const blockModifierActive = (
                action:
                    | "modifier.paramBlockSplitAtClick"
                    | "modifier.paramBlockCreate"
                    | "modifier.paramBlockCreateFlat",
            ): boolean => {
                const binding = firstBinding(mergedKeybindings[action]);
                return (
                    binding !== undefined &&
                    !isNoneBinding(binding) &&
                    isModifierActive(binding, {
                        ctrlKey: hit.ctrlKey === true,
                        shiftKey: hit.shiftKey === true,
                        altKey: hit.altKey === true,
                        metaKey: hit.metaKey === true,
                    })
                );
            };
            const splitAtClick = blockModifierActive("modifier.paramBlockSplitAtClick");
            const createBlock = blockModifierActive("modifier.paramBlockCreate");
            const createFlat = blockModifierActive("modifier.paramBlockCreateFlat");
            if (hit.contextMenu) {
                const layout = paramBlockViewRef.current;
                if (!layout) return { handled: true, seekOnClick: false };
                paramBlockClickCandidateRef.current = null;
                setParamBlockContextMenu(null);
                const fp = layout.framePeriodMs;
                const framedHits = sortParamBlocksByTime(
                    hostRef.current?.getParamBlockHitRects() ?? [],
                );
                const rightHitRect = hitTestParamBlocksAtPoint({
                    blocks: framedHits,
                    sec: hit.sec,
                    localY: hit.localY,
                    slopPx:
                        hit.pointerType === "mouse"
                            ? Math.min(1, paramBlockHitSlopPx)
                            : paramBlockHitSlopPx,
                });
                const axisForRight = hostRef.current?.getAxis();
                const rightEdgeHit =
                    axisForRight === undefined
                        ? null
                        : hitTestParamBlockEdges({
                              blocks: framedHits,
                              clientX: hit.clientX - hit.captureTarget.getBoundingClientRect().left,
                              localY: hit.localY,
                              secToX: (sec) =>
                                  sec * axisForRight.pxPerSec - axisForRight.scrollLeftPx,
                              hitPx: 8,
                          });
                const rightTransition =
                    axisForRight === undefined
                        ? null
                        : hitTestParamBlockTransitionHandles({
                              blocks: framedHits,
                              clientX: hit.clientX - hit.captureTarget.getBoundingClientRect().left,
                              localY: hit.localY,
                              secToX: (sec) =>
                                  sec * axisForRight.pxPerSec - axisForRight.scrollLeftPx,
                              outsetPx: PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX,
                              outsetYPx: PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX,
                          });
                const rightBlockId =
                    rightHitRect?.id ?? rightEdgeHit?.block.id ?? rightTransition?.leftId ?? null;
                const rightBlock =
                    rightBlockId === null
                        ? null
                        : (paramBlocksWithValues.find((block) => block.id === rightBlockId) ??
                          null);
                if (rightBlock !== null) {
                    const logicalId = logicalBlockId(rightBlock);
                    const frame = toFrame(hit.sec, fp);
                    if (!selectedParamBlockIdSet.has(logicalId)) {
                        setParamBlockSelection(
                            normalizeSelection(
                                paramBlockLogicalFrameRanges(selectionFrameBlocks, rightBlock.id),
                            ),
                        );
                    }
                    paramBlockAnchorRef.current = { blockId: logicalId, frame };
                    const rawSelection = selectionRef.current;
                    const expanded =
                        rawSelection === null
                            ? []
                            : expandParamBlockSelection(layout.blocks, rawSelection);
                    const morphSelection =
                        expanded.length > 0 ? normalizeSelection(expanded) : rawSelection;
                    if (morphSelection !== rawSelection) setParamBlockSelection(morphSelection);
                    beginParamBlockMorphRef.current({
                        selection: morphSelection ?? [],
                        nativeEvent: hit.nativeEvent,
                        clientX: hit.clientX,
                        clientY: hit.clientY,
                        pointerId: hit.pointerId,
                        captureTarget: hit.captureTarget,
                        onContextMenu: (x, y) => setParamBlockContextMenu({ x, y }),
                        commitAmplitude: async (segments, authority) => {
                            /*
                             * One backend transaction for both the transformed
                             * curve and the cleared pitch centers. Splitting this
                             * into upload + checkpoint=false used to create a
                             * second write that could land after undo.
                             */
                            const state = clearParamBlockValues(paramBlockStateRef.current, [
                                ...selectedParamBlockIdSet,
                            ]);
                            if (state !== paramBlockStateRef.current) {
                                setParamBlockState(state);
                            }
                            await blockPersistence.commit(
                                state,
                                true,
                                segments.map((segment) => ({ ...segment, param: editParam })),
                                authority,
                                "amplitude",
                            );
                        },
                    });
                    return { handled: true, seekOnClick: false };
                }
                const originalSelection = selectionRef.current;
                const anchorFrame = toFrame(hit.sec, fp);
                let dragging = false;
                let lastX = hit.clientX;
                const setRegion = (region: ParamSelection | null) => {
                    const expanded =
                        region === null ? null : expandParamBlockSelection(layout.blocks, region);
                    const next = expanded !== null && expanded.length > 0 ? expanded : region;
                    setParamBlockSelection(next);
                };
                const seekFollowingPointer = (clientX: number) => {
                    if (!s.paramEditorSeekPlayheadEnabled) return;
                    const sec = hit.secAtClientX(clientX);
                    dispatch(setPlayheadSec(sec));
                    void dispatch(seekPlayhead(sec));
                };
                const update = (clientX: number, seek: boolean) => {
                    lastX = clientX;
                    if (!dragging && Math.abs(clientX - hit.clientX) < 3) return;
                    dragging = true;
                    const pointerFrame = toFrame(hit.secAtClientX(clientX), fp);
                    setRegion(
                        selectionForRangeDrag({
                            base: originalSelection,
                            anchorFrame,
                            pointerFrame,
                            mode: "add",
                        }),
                    );
                    if (seek) seekFollowingPointer(clientX);
                };
                startPointerGesture({
                    pointerId: hit.pointerId,
                    captureTarget: hit.captureTarget,
                    onMove: (event) => {
                        if ((event.buttons & 2) !== 2) return;
                        hit.edgeScroll?.track(event.clientX, event.clientY, () => {
                            update(lastX, true);
                        });
                        update(event.clientX, true);
                    },
                    onEnd: (reason, event) => {
                        hit.edgeScroll?.stop();
                        armRightDragContextMenuGuard();
                        if (reason !== "up") {
                            setParamBlockSelection(originalSelection);
                            return;
                        }
                        if (event && dragging) update(event.clientX, true);
                        if (dragging) {
                            paramBlockAnchorRef.current = resolveBlockAnchor({
                                blocks: layout.blocks,
                                anchor: paramBlockAnchorRef.current,
                                selection: selectionRef.current,
                                playheadFrame: anchorFrame,
                            });
                            if (event) seekFollowingPointer(event.clientX);
                            return;
                        }
                        setParamBlockContextMenu({
                            x: event?.clientX ?? hit.clientX,
                            y: event?.clientY ?? hit.clientY,
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
            const edgeCandidateIds = new Set(
                paramBlockEdgeCandidates(paramBlocksWithValues).map((block) => block.id),
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
                          blocks: framedHits.filter((rect) => edgeCandidateIds.has(rect.id)),
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
            /*
             * Alt + 左键点击块内 = 在点击位置分割。它必须优先于边缘伸缩、
             * 过渡手柄与普通块拖动；拖动越过 3px 死区后由 beginBlockDrag
             * 自动取消“点击”回调，回到普通块拖动。
             */
            const splitBlock =
                hitRect !== null
                    ? (paramBlocksWithValues.find((block) => block.id === hitRect.id) ?? null)
                    : handleEdgeHit !== null
                      ? (paramBlocksWithValues.find(
                            (block) => block.id === handleEdgeHit.block.id,
                        ) ?? null)
                      : null;
            if (splitAtClick && !createFlat && splitBlock !== null) {
                const logicalId = logicalBlockId(splitBlock);
                const splitFrame = toFrame(hit.sec, paramBlockStateRef.current.framePeriodMs);
                const splitOriginalSelection = selectionRef.current;
                const splitOriginalAnchor = paramBlockAnchorRef.current;
                if (!selectedParamBlockIdSet.has(logicalId)) {
                    setParamBlockSelection(
                        normalizeSelection(
                            paramBlockLogicalFrameRanges(selectionFrameBlocks, splitBlock.id),
                        ),
                    );
                }
                paramBlockAnchorRef.current = { blockId: logicalId, frame: splitFrame };
                paramBlockClickCandidateRef.current = null;
                setParamBlockContextMenu(null);
                const dragSelection = paramBlockIdsInSelection(
                    selectionFrameBlocks,
                    selectionRef.current,
                );
                const splitDragStarted = beginBlockDrag({
                    anchorBlockId: splitBlock.id,
                    clientX: hit.clientX,
                    clientY: hit.clientY,
                    contextMenu: false,
                    pointerId: hit.pointerId,
                    captureTarget: hit.captureTarget,
                    setActivePointerGestureEnd: hit.setActivePointerGestureEnd,
                    clearActivePointerGestureEnd: hit.clearActivePointerGestureEnd,
                    pointerValue: hit.pointerValue,
                    edgeScroll: hit.edgeScroll,
                    selection: dragSelection,
                    anchorValue:
                        framedHits.find((block) => block.id === splitBlock.id)?.averageValue ??
                        null,
                    centerValues: new Map(
                        framedHits.flatMap((block) =>
                            block.averageValue == null
                                ? []
                                : [[block.id, block.averageValue] as const],
                        ),
                    ),
                    shiftKey: hit.shiftKey,
                    altKey: hit.altKey,
                    ctrlKey: hit.ctrlKey,
                    metaKey: hit.metaKey,
                    onClickWithoutDrag: (event) => {
                        const modifierEvent = {
                            ctrlKey: event?.ctrlKey ?? hit.ctrlKey === true,
                            shiftKey: event?.shiftKey ?? hit.shiftKey === true,
                            altKey: event?.altKey ?? hit.altKey === true,
                            metaKey: event?.metaKey ?? hit.metaKey === true,
                        };
                        const snapshot = resolveParamBlockModifierSnapshot({
                            event: modifierEvent,
                            keybindings: mergedKeybindings,
                        });
                        // 分割是“点击意图”：松开分割键后 pointerup 不再分割。
                        if (!snapshot.splitAtClick) return;
                        const ok = splitParamBlockAtSecRef.current?.({
                            rawSec: hit.sec,
                            targetBlockId: splitBlock.id,
                            snapToggleHeld: snapshot.snapToggle,
                            movePlayhead: false,
                            selectResult: true,
                            snapMode: "grid",
                        });
                        if (ok) {
                            suppressBlockDoubleClickUntilRef.current = performance.now() + 500;
                        } else {
                            setParamBlockSelection(splitOriginalSelection);
                            paramBlockAnchorRef.current = splitOriginalAnchor;
                        }
                    },
                });
                if (!splitDragStarted) return { handled: true, seekOnClick: false };
                return { handled: true, seekOnClick: false };
            }

            if (touchedHandle !== null && !handleStealsEdge && !hit.contextMenu) {
                const leftBlock = paramBlocksWithValues.find(
                    (block) => block.id === touchedHandle.leftId,
                );
                const rightBlock = paramBlocksWithValues.find(
                    (block) => block.id === touchedHandle.rightId,
                );
                if (leftBlock !== undefined && rightBlock !== undefined) {
                    const transitionPress = resolveBlockPressPlan({
                        blocks: selectionFrameBlocks,
                        selection: selectionRef.current,
                        anchor: paramBlockAnchorRef.current,
                        targetBlockId: leftBlock.id,
                        targetFrame: toFrame(hit.sec, paramBlockStateRef.current.framePeriodMs),
                        mode: "replace",
                    });
                    if (transitionPress.downSelection !== selectionRef.current) {
                        setParamBlockSelection(transitionPress.downSelection);
                    }
                    if (transitionPress.downAnchor !== null) {
                        paramBlockAnchorRef.current = transitionPress.downAnchor;
                    }
                    hit.captureTarget.setPointerCapture?.(hit.pointerId);
                    const startX = hit.clientX;
                    const pointerStartSec = hit.secAtClientX(hit.clientX);
                    const startWidth = Math.max(0, touchedHandle.widthSec);
                    let targetWidth = startWidth;
                    let activated = false;
                    let lastClientX = startX;
                    let lastTransitionEvent: globalThis.PointerEvent | null = null;
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
                            lastTransitionEvent = event;
                            const applyTransitionAt = (clientX: number) => {
                                const pointerNowSec = hit.secAtClientX(clientX);
                                targetWidth = Math.max(
                                    0,
                                    Math.min(
                                        (rightBlock.endSec - rightBlock.startSec) / 2,
                                        startWidth + (pointerStartSec - pointerNowSec),
                                    ),
                                );
                                const widthSpec =
                                    paramBlockSpec ?? kernelDataRef.current.paramBlocks ?? null;
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
                                                      centerPolicy: "locked",
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
                                const liveSpec =
                                    kernelDataRef.current.paramBlocks ?? widthSpec ?? null;
                                if (liveSpec !== null) {
                                    const nextSpec = {
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
                                    kernelDataRef.current.paramBlocks = nextSpec;
                                    publishParamBlockVisualTransaction("transition", nextSpec);
                                }
                                invalidate();
                            };
                            hit.edgeScroll?.track(event.clientX, event.clientY, (clientX) => {
                                applyTransitionAt(clientX);
                            });
                            applyTransitionAt(event.clientX);
                        },
                        onEnd: (reason, event) => {
                            hit.edgeScroll?.stop();
                            if (reason === "up" && !activated) {
                                const dx = lastClientX - startX;
                                if (Math.abs(dx) >= 3) {
                                    activated = true;
                                    const releaseEvent = event ?? lastTransitionEvent;
                                    const pointerNowSec =
                                        releaseEvent === null
                                            ? pointerStartSec
                                            : hit.secAtClientX(releaseEvent.clientX);
                                    targetWidth = Math.max(
                                        0,
                                        Math.min(
                                            (rightBlock.endSec - rightBlock.startSec) / 2,
                                            startWidth + (pointerStartSec - pointerNowSec),
                                        ),
                                    );
                                }
                            }
                            if (reason === "up" && activated) {
                                const finalSpec = kernelDataRef.current.paramBlocks ?? null;
                                if (finalSpec !== null) {
                                    publishParamBlockVisualTransaction("transition", finalSpec);
                                    // 过渡几何不变：不在此刻宣布"要写曲线"就会被立刻释放。
                                    stampParamBlockCurveIntent(
                                        paramBlockVisualTransactionRef.current,
                                        editParam === "pitch" && targetWidth > 1e-6,
                                    );
                                }
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
                                    reportParamBlockWriteFailure({
                                        error,
                                        gesture: "transition",
                                        rootTrackId,
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
                /*
                 * Alt + 左键在空白处拖动 = 快速新建块。普通左键 / Ctrl+左键
                 * 属于选区手势，不能与创建抢占。
                 */
                if (paramBlockToolActive && (createBlock || createFlat) && !hit.contextMenu) {
                    let createShape: ParamBlockCreateShape = createFlat ? "flat" : "normal";
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
                        publishParamBlockSnapHighlight(result);
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
                    let lastCreateEvent: globalThis.PointerEvent | null = null;
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
                    const createBaseSpec = paramBlockSpec ?? kernelDataRef.current.paramBlocks;
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
                                  fallbackValue: 0,
                                  previousValues: paramBlockValueCacheRef.current,
                                  allowFallback: false,
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
                    const createAuthorityCurveRevision = paramAuthoritySnapshot.curveRevision;
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
                        const modifierSnapshot = resolveParamBlockModifierSnapshot({
                            event,
                            keybindings: mergedKeybindings,
                        });
                        createShape = resolveParamBlockCreateShape({
                            previous: createShape,
                            snapshot: modifierSnapshot,
                        });
                        const flatten = createShape === "flat";
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
                            epoch: paramBlockEpochRef.current,
                            resizePlan: null,
                            targetSec: createSpan.endSec,
                            targetFrame: toFrame(
                                createSpan.endSec,
                                paramBlockStateRef.current.framePeriodMs,
                            ),
                            value: createBlockValue,
                        };
                        /*
                         * 新建块也要报读数：这一笔把块定在什么值上，与画笔落笔同义。
                         * 读数入口与拖块共用（同一个合帧 setState），不另写一份。
                         */
                        if (s.showParamValuePopup) {
                            publishParamValuePreview({
                                clientX: event.clientX,
                                clientY: event.clientY,
                                value: createBlockValue,
                            });
                        }
                        if (createBaseSpec !== null) {
                            const preview = projectParamBlockCreatePreviewFromBase({
                                framePeriodMs: paramBlockStateRef.current.framePeriodMs,
                                baseBlocks: createBaseBlocks,
                                startSec: createSpan.startSec,
                                endSec: createSpan.endSec,
                                createId,
                                value: createBlockValue,
                            });
                            const blocks: ParamBlockRenderBlock[] = preview.map((raw) => {
                                const block = raw as ParamBlockRenderBlock & {
                                    readonly grouped?: boolean;
                                    readonly transitionSec?: number;
                                };
                                const existing =
                                    paramBlocksWithValues.find((item) => item.id === block.id) ??
                                    paramBlocksWithValues.find(
                                        (item) => item.userBlockId === block.id,
                                    );
                                const created = block.id === createId;
                                return {
                                    id: block.id,
                                    userBlockId: block.userBlockId ?? existing?.userBlockId,
                                    startSec: block.startSec,
                                    endSec: block.endSec,
                                    averageValue:
                                        block.averageValue ??
                                        existing?.averageValue ??
                                        createBlockValue,
                                    logicalCenterValue: existing?.manualValue,
                                    frameCenterValue: created ? createBlockValue : undefined,
                                    displayCenterValue: created
                                        ? createBlockValue
                                        : (block.displayCenterValue ?? null),
                                    displayCenterSnapToSemitone: created
                                        ? false
                                        : existing?.manualValue == null,
                                    selected: created,
                                    grouped: block.grouped === true,
                                    transitionSec: block.transitionSec ?? existing?.transitionSec,
                                    centerPolicy: created ? "locked" : "live",
                                };
                            });
                            const previewSpec = buildParamBlockRenderSpec({
                                blocks,
                                param: editParam,
                                valueMin: bounds.min,
                                valueMax: bounds.max,
                                valueToY: (value, heightPx) => valueToY(editParam, value, heightPx),
                                colors: paramBlockSpecColors,
                                curveOwned: paramBlockViewRef.current !== null,
                            });
                            kernelDataRef.current.paramBlocks = previewSpec;
                            publishParamBlockVisualTransaction("create", previewSpec);
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
                                flatten,
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
                            lastCreateEvent = event;
                            hit.edgeScroll?.track(
                                event.clientX,
                                event.clientY,
                                (clientX, clientY) => {
                                    const source = lastCreateEvent;
                                    if (source === null) return;
                                    /*
                                     * 直接复用同一份预览逻辑，只替换坐标；
                                     * 修饰键从最后一次真实事件快照读取。
                                     */
                                    updateCreatePreview({
                                        clientX,
                                        clientY: clientY ?? source.clientY,
                                        ctrlKey: source.ctrlKey,
                                        shiftKey: source.shiftKey,
                                        altKey: source.altKey,
                                        metaKey: source.metaKey,
                                    });
                                },
                            );
                            updateCreatePreview(event);
                        },
                        onEnd: (reason, event) => {
                            hit.edgeScroll?.stop();
                            if (s.showParamValuePopup) publishParamValuePreview(null);
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
                                const finalPreviewSpec = kernelDataRef.current.paramBlocks ?? null;
                                if (finalPreviewSpec !== null) {
                                    publishParamBlockVisualTransaction("create", finalPreviewSpec);
                                    // 盖上「权威水位 + 曲线意图」；否则释放判据只能退回几何相等。
                                    const createTx = paramBlockVisualTransactionRef.current;
                                    if (createTx !== null) {
                                        createTx.curveRevisionAtStart =
                                            createAuthorityCurveRevision;
                                        createTx.authorityRequestSeqAtCommit =
                                            createAuthorityRequestSeqAtCommit;
                                        stampParamBlockCurveIntent(
                                            createTx,
                                            (createCurveBase ?? paramViewRef.current) !== null,
                                        );
                                    }
                                }
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
                                    });
                                    if (created.ok) {
                                        setParamBlockState(created.state);
                                        paramBlockVisualTransactionRef.current =
                                            adoptParamBlockVisualStructureAfterCreate(
                                                paramBlockVisualTransactionRef.current,
                                                created.state.editSet,
                                                createId,
                                                paramBlockResizeRef,
                                            );
                                    } else applyParamBlockEditRef.current?.(created);
                                    preserveCreatePreview = created.ok;
                                    /*
                                     * 显式标注成 ref 的元素类型：字面量推断会漏掉
                                     * `curveRevisionCommitted`（它只在提交成功后写入）。
                                     */
                                    const createHandoff: ParamBlockCreateHandoff | null = created.ok
                                        ? {
                                              blockId: createId,
                                              startedAt: performance.now(),
                                              curveRevisionAtStart: createAuthorityCurveRevision,
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
                                        /**
                                         * 本槽算出的逐帧补丁（成功后抄进 handoff / 事务）。
                                         * prepare 在冲突重基时会被再调一次，所以要 mutable。
                                         */
                                        let preparedCreateSegments:
                                            | readonly {
                                                  readonly startFrame: number;
                                                  readonly values: readonly number[];
                                              }[]
                                            | undefined;
                                        const prepareCreate = createParamBlockWritePlan({
                                            state: created.state,
                                            rootTrackId,
                                            editParam,
                                            curveBase,
                                            layoutBasis: createLayoutBasis,
                                            span: createSpan,
                                            value: resolveCreateBlockValue(createValue),
                                            shape: createShape,
                                            onSegments: (segments) => {
                                                preparedCreateSegments = segments;
                                            },
                                        });
                                        void enqueueParamBlockCreateCommit({
                                            activity: paramBlockDragActivityRef,
                                            liveActive: liveEditActiveRef,
                                            run: () =>
                                                blockPersistence.commitPlan({
                                                    kind: "create",
                                                    rootTrackId,
                                                    prepare: prepareCreate,
                                                }),
                                            onSettled: () => {
                                                blockPersistence.flushReadback();
                                                bumpRefreshToken();
                                                bumpParamBlockStructure();
                                                invalidate();
                                            },
                                        })
                                            .then((result) => {
                                                if (createHandoff === null) return;
                                                const createTxNow =
                                                    paramBlockVisualTransactionRef.current;
                                                if (preparedCreateSegments !== undefined) {
                                                    createHandoff.committedSegments =
                                                        preparedCreateSegments;
                                                    if (createTxNow !== null) {
                                                        createTxNow.committedSegments =
                                                            preparedCreateSegments;
                                                    }
                                                }
                                                /*
                                                 * 回包里的新曲线代次是释放判据的唯一证据；
                                                 * 没有它（纯布局提交 / 曲线上传路径）就退回
                                                 * 「几何已到位」。
                                                 */
                                                const committedCurveRevision =
                                                    result?.curveRevision;
                                                createHandoff.curveRevisionCommitted =
                                                    committedCurveRevision;
                                                if (createTxNow !== null) {
                                                    createTxNow.curveRevisionCommitted =
                                                        committedCurveRevision;
                                                }
                                                createHandoff.authorityRequestSeqAtCommit =
                                                    getAuthorityWatermark().issuedRequestSeq;
                                                createHandoff.committed = true;
                                                /*
                                                 * 只有**没有曲线片段**的提交才允许按几何释放。
                                                 * 无条件置 false 会让新建块也按几何释放 ——
                                                 * 那就是「先回弹」。
                                                 */
                                                createHandoff.requiresCurveAuthority =
                                                    preparedCreateSegments !== undefined &&
                                                    createLayoutBasis !== null;
                                                onParamCommitSucceeded();
                                            })
                                            .catch((error) => {
                                                console.error(
                                                    "[pianoRoll] create block commit failed",
                                                    error,
                                                );
                                                reportParamBlockWriteFailure(error);
                                                if (createHandoff !== null) {
                                                    createHandoff.committed = true;
                                                    createHandoff.requiresCurveAuthority = false;
                                                }
                                                if (ownsCreatePreview()) {
                                                    paramBlockCreateCommitRef.current = null;
                                                    paramBlockResizeRef.current = null;
                                                    paramBlockVisualTransactionRef.current = null;
                                                    paramBlockInteractionCurveRef.current = null;
                                                    const pvNow = paramViewRef.current;
                                                    if (pvNow !== null) resetLiveEditPreview(pvNow);
                                                }
                                            });
                                    }
                                    {
                                        const finalFramePeriod =
                                            paramBlockStateRef.current.framePeriodMs;
                                        const finalStart = toFrame(
                                            createSpan.startSec,
                                            finalFramePeriod,
                                        );
                                        const finalEnd = Math.max(
                                            finalStart + 1,
                                            toFrame(createSpan.endSec, finalFramePeriod),
                                        );
                                        setParamBlockSelection(
                                            selectionFromFrames(finalStart, finalEnd - finalStart),
                                        );
                                    }
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
                                paramBlockVisualTransactionRef.current = null;
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
                const framePeriodMs = paramBlockStateRef.current.framePeriodMs;
                const anchorFrame = toFrame(hit.sec, framePeriodMs);
                const startClientX = hit.clientX;
                const originalSelection = selectionRef.current;
                const baseSelection = hit.multiSelect ? originalSelection : null;
                const mode = hit.multiSelect ? "add" : "replace";
                let moved = false;
                let lastX = startClientX;
                const seekFollowingPointer = (clientX: number) => {
                    if (!s.paramEditorSeekPlayheadEnabled) return;
                    const sec = hit.secAtClientX(clientX);
                    dispatch(setPlayheadSec(sec));
                    void dispatch(seekPlayhead(sec));
                };
                const update = (clientX: number, seek: boolean) => {
                    lastX = clientX;
                    if (!moved && Math.abs(clientX - startClientX) < 3) return;
                    moved = true;
                    const pointerFrame = toFrame(hit.secAtClientX(clientX), framePeriodMs);
                    const raw = selectionForRangeDrag({
                        base: baseSelection,
                        anchorFrame,
                        pointerFrame,
                        mode,
                    });
                    const expanded =
                        raw === null
                            ? null
                            : expandParamBlockSelection(
                                  paramBlockViewRef.current?.blocks ?? [],
                                  raw,
                              );
                    setParamBlockSelection(
                        expanded !== null && expanded.length > 0 ? expanded : raw,
                    );
                    if (seek) seekFollowingPointer(clientX);
                };
                startPointerGesture({
                    pointerId: hit.pointerId,
                    captureTarget: hit.captureTarget,
                    onMove: (event) => {
                        if ((event.buttons & 1) !== 1) return;
                        hit.edgeScroll?.track(event.clientX, event.clientY, () => {
                            update(lastX, true);
                        });
                        update(event.clientX, true);
                    },
                    onEnd: (reason, event) => {
                        hit.edgeScroll?.stop();
                        if (reason !== "up") {
                            setParamBlockSelection(originalSelection);
                            return;
                        }
                        if (event && moved) update(event.clientX, true);
                        if (!moved) {
                            if (hit.multiSelect) {
                                setParamBlockSelection(
                                    removeRangeAtFrame(originalSelection, anchorFrame),
                                );
                            } else {
                                setParamBlockSelection(null);
                            }
                            paramBlockAnchorRef.current = null;
                        } else {
                            if (event) seekFollowingPointer(event.clientX);
                            const startBlock = paramBlockViewRef.current?.blocks.find(
                                (block) =>
                                    anchorFrame >= block.startFrame &&
                                    anchorFrame < block.endFrameExclusive,
                            );
                            paramBlockAnchorRef.current =
                                startBlock !== undefined
                                    ? {
                                          blockId: startBlock.userBlockId ?? startBlock.id,
                                          frame: anchorFrame,
                                      }
                                    : resolveBlockAnchor({
                                          blocks: paramBlockViewRef.current?.blocks ?? [],
                                          anchor: null,
                                          selection: selectionRef.current,
                                          playheadFrame: anchorFrame,
                                      });
                        }
                        setParamBlockContextMenu(null);
                    },
                    registerEnd: hit.setActivePointerGestureEnd,
                    clearEnd: hit.clearActivePointerGestureEnd,
                });
                return { handled: true, seekOnClick: true };
            }
            const fullBlock = paramBlocksWithValues.find((block) => block.id === hitId);
            if (fullBlock === undefined) return { handled: false };
            /* 命中优先级：手柄已在上方入口处理；这里只剩边缘缩放 > 块体。 */
            const axis = hostRef.current?.getAxis();
            const edgeHit =
                axis === undefined
                    ? null
                    : hitTestParamBlockEdges({
                          blocks: framedHits.filter(
                              (rect) => rect.id === fullBlock.id && edgeCandidateIds.has(rect.id),
                          ),
                          clientX: hit.clientX - hit.captureTarget.getBoundingClientRect().left,
                          localY: hit.localY,
                          secToX: (sec) => sec * axis.pxPerSec - axis.scrollLeftPx,
                          hitPx: 8,
                      });
            if (edgeHit !== null) {
                const resizeEdge = edgeHit.edge;
                const resizePress = resolveBlockPressPlan({
                    blocks: selectionFrameBlocks,
                    selection: selectionRef.current,
                    anchor: paramBlockAnchorRef.current,
                    targetBlockId: fullBlock.id,
                    targetFrame: toFrame(hit.sec, paramBlockStateRef.current.framePeriodMs),
                    mode: "replace",
                });
                if (resizePress.downSelection !== selectionRef.current) {
                    setParamBlockSelection(resizePress.downSelection);
                }
                if (resizePress.downAnchor !== null) {
                    paramBlockAnchorRef.current = resizePress.downAnchor;
                }
                const resizeSelectedIds = new Set(resizePress.dragIds);
                const framePeriodMs =
                    paramBlockViewRef.current?.framePeriodMs ??
                    paramBlockStateRef.current.framePeriodMs;
                const baseBlockSpans = paramBlocksWithValues.map((block) => ({
                    id: block.id,
                    startSec: block.startSec,
                    endSec: block.endSec,
                    selected: resizeSelectedIds.has(logicalBlockId(block)),
                }));
                const valueForBlock = (
                    blockId: string,
                    startSec: number,
                    endSec: number,
                ): number | null => {
                    const exact = paramBlocksWithValues.find((block) => block.id === blockId);
                    if (exact?.averageValue != null) return exact.averageValue;
                    const cached = paramBlockValueCacheRef.current.get(blockId);
                    if (
                        cached !== undefined &&
                        Math.abs(cached.startSec - startSec) <= 1e-6 &&
                        Math.abs(cached.endSec - endSec) <= 1e-6 &&
                        Number.isFinite(cached.value)
                    ) {
                        return cached.value;
                    }
                    const overlapping = paramBlocksWithValues.find(
                        (block) => block.startSec < endSec && block.endSec > startSec,
                    );
                    return overlapping?.averageValue ?? null;
                };
                const previewSpecBlocks = (
                    plan: ParamBlockGesturePlan,
                ): ParamBlockRenderBlock[] => {
                    const fp = plan.viewAfter.framePeriodMs;
                    return plan.viewAfter.blocks.map((block) => {
                        const startSec = (block.startFrame * fp) / 1000;
                        const endSec = (block.endFrameExclusive * fp) / 1000;
                        const existing =
                            paramBlocksWithValues.find((item) => item.id === block.id) ??
                            paramBlocksWithValues.find(
                                (item) => item.userBlockId === block.userBlockId,
                            );
                        return {
                            id: block.id,
                            userBlockId: block.userBlockId,
                            startSec,
                            endSec,
                            averageValue:
                                existing?.averageValue ?? valueForBlock(block.id, startSec, endSec),
                            logicalCenterValue: existing?.manualValue,
                            displayCenterValue:
                                block.id === gestureRef.blockId
                                    ? (gestureRef.displayCenterValue ?? null)
                                    : null,
                            displayCenterSnapToSemitone:
                                block.id === gestureRef.blockId
                                    ? false
                                    : existing?.manualValue == null,
                            displayHeightPx:
                                block.id === gestureRef.blockId
                                    ? (gestureRef.displayHeightPx ?? null)
                                    : null,
                            selected:
                                block.id === gestureRef.blockId ||
                                (existing !== undefined &&
                                    resizeSelectedIds.has(logicalBlockId(existing)) &&
                                    block.id !== gestureRef.blockId),
                            grouped: block.kind === "grouped",
                            transitionSec: existing?.transitionSec,
                            centerPolicy: block.id === gestureRef.blockId ? "locked" : "live",
                        };
                    });
                };
                const initialTargetFrame =
                    resizeEdge === "left"
                        ? toFrame(fullBlock.startSec, framePeriodMs)
                        : toFrame(fullBlock.endSec, framePeriodMs);
                const gestureRef = {
                    blockId: fullBlock.id,
                    edge: resizeEdge,
                    epoch: paramBlockEpochRef.current,
                    pointerClientX: hit.clientX,
                    pointerStartSec: hit.secAtClientX(hit.clientX),
                    startSec: fullBlock.startSec,
                    endSec: fullBlock.endSec,
                    neighborStartSec: null,
                    neighborEndSec: null,
                    resizeCandidates: [],
                    baseBlockSpans,
                    framePeriodMsBasis: framePeriodMs,
                    displayCenterValue: fullBlock.averageValue,
                    displayHeightPx:
                        hostRef.current
                            ?.getParamBlockHitRects()
                            .find((rect) => rect.id === fullBlock.id)?.heightPx ?? null,
                    targetSec: resizeEdge === "left" ? fullBlock.startSec : fullBlock.endSec,
                    targetFrame: initialTargetFrame,
                    resizePlan: null as ParamBlockGesturePlan | null,
                };
                const runResizePlan = (targetFrame: number) =>
                    planParamBlockGesture({
                        clips: paramBlockSourceClipsRef.current,
                        edits: paramBlockStateRef.current.editSet,
                        framePeriodMs,
                        gesture: {
                            kind: "resize",
                            blockId: gestureRef.blockId,
                            edge: gestureRef.edge,
                            targetFrame,
                        },
                    });
                let activateResize = false;
                let lastResizeEvent: globalThis.PointerEvent | null = null;
                const applyResizeAt = (event: {
                    clientX: number;
                    ctrlKey: boolean;
                    shiftKey: boolean;
                    altKey: boolean;
                    metaKey: boolean;
                }) => {
                    /*
                     * 用“当前指针内容坐标 - 按下时内容坐标”的位移，而不是
                     * dx / pxPerSec：滚动或缩放后 pxPerSec 与 scrollLeft 都
                     * 会变，纯像素差会让目标边缘漂移。
                     */
                    const pointerNowSec = hit.secAtClientX(event.clientX);
                    const targetSec =
                        (gestureRef.edge === "left" ? gestureRef.startSec : gestureRef.endSec) +
                        (pointerNowSec - gestureRef.pointerStartSec);
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
                            publishParamBlockSnapHighlight(result);
                            return result.sec;
                        },
                        minBlockSec: framePeriodMs / 1000,
                    });
                    const targetFrame = toFrame(snappedTarget, framePeriodMs);
                    gestureRef.targetSec = snappedTarget;
                    gestureRef.targetFrame = targetFrame;
                    const result = runResizePlan(targetFrame);
                    if (!result.ok) return;
                    gestureRef.resizePlan = result.plan;
                    const previewSpec = buildParamBlockRenderSpec({
                        blocks: previewSpecBlocks(result.plan),
                        param: editParam,
                        valueMin: bounds.min,
                        valueMax: bounds.max,
                        valueToY: (value, heightPx) => valueToY(editParam, value, heightPx),
                        colors: paramBlockSpecColors,
                        curveOwned: paramBlockViewRef.current !== null,
                    });
                    kernelDataRef.current.paramBlocks = previewSpec;
                    publishParamBlockVisualTransaction("resize", previewSpec);
                    invalidate();
                };
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
                        lastResizeEvent = event;
                        hit.edgeScroll?.track(event.clientX, event.clientY, (clientX) => {
                            const source = lastResizeEvent;
                            if (source === null) return;
                            applyResizeAt({
                                clientX,
                                ctrlKey: source.ctrlKey,
                                shiftKey: source.shiftKey,
                                altKey: source.altKey,
                                metaKey: source.metaKey,
                            });
                        });
                        applyResizeAt(event);
                    },
                    onEnd: (reason) => {
                        const plan = gestureRef.resizePlan;
                        if (reason === "up" && activateResize && plan !== null) {
                            publishParamBlockVisualTransaction(
                                "resize",
                                buildParamBlockRenderSpec({
                                    blocks: previewSpecBlocks(plan),
                                    param: editParam,
                                    valueMin: bounds.min,
                                    valueMax: bounds.max,
                                    valueToY: (value, heightPx) =>
                                        valueToY(editParam, value, heightPx),
                                    colors: paramBlockSpecColors,
                                    curveOwned: paramBlockViewRef.current !== null,
                                }),
                            );
                            const state = paramBlockStateRef.current;
                            const next: ParamBlockState = { ...state, editSet: plan.edits };
                            setParamBlockState(next);
                            paramBlockVisualTransactionRef.current =
                                adoptParamBlockVisualStructureAfterEdits(
                                    paramBlockVisualTransactionRef.current,
                                    next.editSet,
                                );
                            persistParamBlockEdits(next);
                            if (plan.removedIds.includes(gestureRef.blockId)) {
                                paramBlockAnchorRef.current = null;
                            }
                            const remainingIds = new Set(
                                plan.viewAfter.blocks.map((block) => logicalBlockId(block)),
                            );
                            if ([...resizeSelectedIds].some((id) => !remainingIds.has(id))) {
                                const ranges = plan.viewAfter.blocks
                                    .filter((block) => resizeSelectedIds.has(logicalBlockId(block)))
                                    .flatMap((block) =>
                                        logicalFrameRangesFromView(plan.viewAfter.blocks, block.id),
                                    );
                                setParamBlockSelection(normalizeSelection(ranges));
                            }
                            for (const warning of plan.warnings) {
                                reportParamBlockDiagnostic({
                                    code: warning,
                                    gesture: "resize",
                                    rootTrackId,
                                    blockId: gestureRef.blockId,
                                    frame: gestureRef.targetFrame,
                                    edge: gestureRef.edge,
                                });
                            }
                        }
                        hit.edgeScroll?.stop();
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
             * 按下计划：无修饰键按下已选块时保留整个选区，只有 pointerup
             * 且没有拖动时才把选区收敛成目标块。这样多选拖动不会在
             * pointerdown 阶段被 resolveBlockClickSelection 压成单选。
             */
            const framePeriodMsForClick =
                paramBlockViewRef.current?.framePeriodMs ??
                paramBlockStateRef.current.framePeriodMs;
            const targetFrame = toFrame(hit.sec, framePeriodMsForClick);
            const mode: BlockClickMode = hit.rangeSelect
                ? "range"
                : hit.multiSelect
                  ? "toggle"
                  : "replace";
            const press = resolveBlockPressPlan({
                blocks: selectionFrameBlocks,
                selection: selectionRef.current,
                anchor: paramBlockAnchorRef.current,
                targetBlockId: fullBlock.id,
                targetFrame,
                mode,
            });
            if (press.downSelection !== selectionRef.current) {
                setParamBlockSelection(press.downSelection);
            }
            if (press.downAnchor !== null) paramBlockAnchorRef.current = press.downAnchor;
            setParamBlockContextMenu(null);
            bumpParamBlockStructure();
            if (!hit.contextMenu) {
                paramBlockClickCandidateRef.current = {
                    blockId: fullBlock.id,
                    at: performance.now(),
                };
            }
            if (!press.startsDrag) {
                /*
                 * Ctrl 移除在按下时已经生效，不需要再启动拖动；Shift 在空选区上
                 * 点击则只等待 pointerup 应用范围选择。这里用一个轻量点击手势
                 * 保证“延迟到松手”的语义，同时不产生任何持久化写入。
                 */
                if (press.clickSelection === null) return { handled: true, seekOnClick: false };
                let clickGesture: ReturnType<typeof startPointerGesture> | null = null;
                let handedOff = false;
                const startX = hit.clientX;
                const startY = hit.clientY;
                clickGesture = startPointerGesture({
                    pointerId: hit.pointerId,
                    captureTarget: hit.captureTarget,
                    onMove: (event) => {
                        if (handedOff) return;
                        if (Math.hypot(event.clientX - startX, event.clientY - startY) < 3) return;
                        handedOff = true;
                        clickGesture?.end("cancel");
                        void beginBlockDrag({
                            anchorBlockId: fullBlock.id,
                            clientX: event.clientX,
                            clientY: event.clientY,
                            contextMenu: hit.contextMenu,
                            pointerId: hit.pointerId,
                            captureTarget: hit.captureTarget,
                            setActivePointerGestureEnd: hit.setActivePointerGestureEnd,
                            clearActivePointerGestureEnd: hit.clearActivePointerGestureEnd,
                            pointerValue: hit.pointerValue,
                            edgeScroll: hit.edgeScroll,
                            selection: press.dragIds.length > 0 ? press.dragIds : [fullBlock.id],
                            anchorValue:
                                framedHits.find((block) => block.id === hitId)?.averageValue ??
                                null,
                            centerValues: new Map(
                                framedHits.flatMap((block) =>
                                    block.averageValue == null
                                        ? []
                                        : [[block.id, block.averageValue] as const],
                                ),
                            ),
                            shiftKey: hit.shiftKey,
                            altKey: hit.altKey,
                            ctrlKey: hit.ctrlKey,
                            metaKey: hit.metaKey,
                        });
                    },
                    onEnd: (reason) => {
                        if (handedOff) return;
                        if (reason === "up") setParamBlockSelection(press.clickSelection);
                    },
                    registerEnd: hit.setActivePointerGestureEnd,
                    clearEnd: hit.clearActivePointerGestureEnd,
                });
                return { handled: true, seekOnClick: false };
            }
            const dragStarted = beginBlockDrag({
                anchorBlockId: fullBlock.id,
                clientX: hit.clientX,
                clientY: hit.clientY,
                contextMenu: hit.contextMenu,
                pointerId: hit.pointerId,
                captureTarget: hit.captureTarget,
                setActivePointerGestureEnd: hit.setActivePointerGestureEnd,
                clearActivePointerGestureEnd: hit.clearActivePointerGestureEnd,
                pointerValue: hit.pointerValue,
                edgeScroll: hit.edgeScroll,
                selection: press.dragIds.length > 0 ? press.dragIds : [fullBlock.id],
                anchorValue: framedHits.find((block) => block.id === hitId)?.averageValue ?? null,
                centerValues: new Map(
                    framedHits.flatMap((block) =>
                        block.averageValue == null ? [] : [[block.id, block.averageValue] as const],
                    ),
                ),
                shiftKey: hit.shiftKey,
                altKey: hit.altKey,
                ctrlKey: hit.ctrlKey,
                metaKey: hit.metaKey,
                onClickWithoutDrag:
                    press.clickSelection === null
                        ? undefined
                        : () => setParamBlockSelection(press.clickSelection),
            });
            if (!dragStarted && press.clickSelection !== null) {
                setParamBlockSelection(press.clickSelection);
            }
            return { handled: true, seekOnClick: true };
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps -- paramBlockSpec 作为事件期几何读取；加入依赖会让手势在预览帧重建。
        [
            editParam,
            beginBlockDrag,
            clearStaleBlockCentersAfterParamEdit,
            mergedKeybindings,
            selectedParamBlockIdSet,
            bumpParamBlockStructure,
            invalidate,
            paramBlockClips,
            paramBlockToolActive,
            blockPersistence.ready,
            paramBlocksWithValues,
            rootTrackId,
            pxPerSecRef,
            selectedParamBlockIds,
            selectionFrameBlocks,
            setParamBlockSelection,
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
        if (!requireBlockPersistenceReady()) return;
        // Read the live command operands, not the render closure: two quick
        // commands must see the edit set produced by the first one.
        const state = paramBlockStateRef.current;
        const blocks = paramBlocksWithValuesRef.current;
        const clips = paramBlockSourceClipsRef.current;
        const result = groupParamBlocks({
            state,
            blocks,
            selectedBlockIds: selectedParamBlockIdsNow(),
            clips,
        });
        // 前置条件不满足 = 静默 no-op（与时间轴的 `groupClips` 同一约定）。
        // 菜单项已经置灰；走到这里只可能是"菜单打开后状态变了"。
        if (!result.ok) return;
        invalidateParamBlockVisualPreviews("structure");
        setParamBlockState(result.state);
        persistParamBlockEdits(result.state);
        const previousIds = new Set(state.editSet.blocks.map((block) => block.id));
        const createdGroup = result.state.editSet.blocks.find(
            (block) => block.geometry.kind === "group" && !previousIds.has(block.id),
        );
        if (createdGroup !== undefined && rootTrackId !== null) {
            const view = composeParamBlockView(
                rootTrackId,
                clips,
                result.state.editSet,
                result.state.framePeriodMs,
            );
            const ranges = logicalFrameRangesFromView(view.blocks, createdGroup.id);
            setParamBlockSelection(normalizeSelection(ranges));
            paramBlockAnchorRef.current =
                ranges.length > 0
                    ? { blockId: createdGroup.id, frame: ranges[0]!.startFrame }
                    : null;
        } else {
            clearParamBlockSelection();
        }
        setParamBlockContextMenu(null);
    }, [
        clearParamBlockSelection,
        invalidateParamBlockVisualPreviews,
        persistParamBlockEdits,
        paramBlockStateRef,
        rootTrackId,
        selectedParamBlockIdsNow,
        setParamBlockSelection,
        setParamBlockState,
        requireBlockPersistenceReady,
    ]);
    useLatestRefValue(groupSelectedParamBlocksRef, groupSelectedParamBlocks);
    const ungroupSelectedParamBlocks = useCallback(() => {
        if (!requireBlockPersistenceReady()) return;
        const state = paramBlockStateRef.current;
        const clips = paramBlockSourceClipsRef.current;
        const selectedIds = selectedParamBlockIdsNow().filter((id) =>
            paramBlocksWithValuesRef.current.some(
                (block) => logicalBlockId(block) === id && block.kind === "grouped",
            ),
        );
        // 没有编组块可拆 = 静默；菜单项已经置灰。
        if (selectedIds.length === 0) return;
        try {
            const next = ungroupParamBlocks(state, selectedIds);
            invalidateParamBlockVisualPreviews("structure");
            setParamBlockState(next);
            persistParamBlockEdits(next);
            if (rootTrackId !== null) {
                const view = composeParamBlockView(
                    rootTrackId,
                    clips,
                    next.editSet,
                    next.framePeriodMs,
                );
                const childIds = new Set(
                    selectedIds.flatMap((id) => {
                        const group = state.editSet.blocks.find(
                            (block) => block.id === id && block.geometry.kind === "group",
                        );
                        return group?.geometry.kind === "group"
                            ? group.geometry.children.map((child) => child.id)
                            : [];
                    }),
                );
                const remainingIds = new Set(
                    next.editSet.blocks
                        .filter((block) => block.deleted !== true && childIds.has(block.id))
                        .map((block) => block.id),
                );
                const remainingBlocks = view.blocks.filter((block) =>
                    remainingIds.has(logicalBlockId(block)),
                );
                const ranges = remainingBlocks.flatMap((block) =>
                    logicalFrameRangesFromView(view.blocks, block.id),
                );
                setParamBlockSelection(normalizeSelection(ranges));
                const firstRemaining = remainingBlocks[0];
                paramBlockAnchorRef.current =
                    firstRemaining === undefined
                        ? null
                        : {
                              blockId: logicalBlockId(firstRemaining),
                              frame: firstRemaining.startFrame,
                          };
            } else {
                clearParamBlockSelection();
            }
            setParamBlockContextMenu(null);
        } catch (error) {
            warnDev("[pianoRoll] ungroup param blocks failed: " + String(error));
        }
    }, [
        clearParamBlockSelection,
        invalidateParamBlockVisualPreviews,
        persistParamBlockEdits,
        paramBlockStateRef,
        rootTrackId,
        selectedParamBlockIdsNow,
        setParamBlockSelection,
        setParamBlockState,
        requireBlockPersistenceReady,
    ]);
    useLatestRefValue(ungroupSelectedParamBlocksRef, ungroupSelectedParamBlocks);
    const createParamBlocksFromSelection = useCallback(() => {
        if (!rootTrackId || !requireBlockPersistenceReady()) return;
        const ranges = selectionToFrameRanges(selectionRef.current);
        if (ranges.length === 0) return;
        const framePeriodMs = paramBlockStateRef.current.framePeriodMs;
        let working = paramBlockStateRef.current;
        let changed = false;
        let lastCreatedId: string | null = null;
        let lastCreatedFrame = 0;
        const failures: string[] = [];
        for (const range of ranges) {
            const result = createParamBlock({
                state: working,
                clips: paramBlockSourceClipsRef.current,
                startSec: (range.startFrame * framePeriodMs) / 1000,
                endSec: ((range.startFrame + range.frameCount) * framePeriodMs) / 1000,
                value: editParam === "pitch" ? 60 : null,
            });
            if (!result.ok) {
                failures.push(result.reason);
                continue;
            }
            working = result.state;
            changed = true;
            lastCreatedId = result.blockId;
            lastCreatedFrame = range.startFrame;
        }
        if (!changed) {
            if (failures.length > 0) {
                reportParamBlockLocalRejection({
                    code: "create_block_failed",
                    detail: failures.join(", "),
                    gesture: "create",
                    rootTrackId,
                    tf,
                });
            }
            return;
        }
        if (failures.length > 0) {
            reportParamBlockLocalRejection({
                code: "create_block_partially_failed",
                detail: failures.join(", "),
                gesture: "create",
                rootTrackId,
                tf,
            });
        }
        invalidateParamBlockVisualPreviews("structure");
        setParamBlockState(working);
        if (lastCreatedId !== null) {
            paramBlockAnchorRef.current = {
                blockId: lastCreatedId,
                frame: lastCreatedFrame,
            };
        }
        persistParamBlockEdits(working);
        bumpParamBlockStructure();
        setParamBlockContextMenu(null);
    }, [
        bumpParamBlockStructure,
        editParam,
        invalidateParamBlockVisualPreviews,
        persistParamBlockEdits,
        paramBlockStateRef,
        requireBlockPersistenceReady,
        rootTrackId,
        selectionRef,
        setParamBlockState,
        tf,
    ]);
    useLatestRefValue(createParamBlocksFromSelectionRef, createParamBlocksFromSelection);

    const deleteSelectedParamBlocks = useCallback(() => {
        if (!requireBlockPersistenceReady()) return;
        const result = deleteParamBlocks({
            state: paramBlockStateRef.current,
            blocks: paramBlocksWithValuesRef.current,
            selectedBlockIds: selectedParamBlockIdsNow(),
            clips: paramBlockSourceClipsRef.current,
        });
        // 没选中块 / 块不可删 = 静默；菜单项已经置灰。
        if (!result.ok) return;
        invalidateParamBlockVisualPreviews("structure");
        setParamBlockState(result.state);
        persistParamBlockEdits(result.state);
        clearParamBlockSelection();
        setParamBlockContextMenu(null);
    }, [
        clearParamBlockSelection,
        invalidateParamBlockVisualPreviews,
        paramBlockStateRef,
        persistParamBlockEdits,
        selectedParamBlockIdsNow,
        requireBlockPersistenceReady,
        setParamBlockState,
    ]);
    useLatestRefValue(deleteSelectedParamBlocksRef, deleteSelectedParamBlocks);
    const restoreParamBlocks = useCallback(() => {
        if (!requireBlockPersistenceReady()) return;
        const state = paramBlockStateRef.current;
        const clips = paramBlockSourceClipsRef.current;
        const deletedSnapshots = state.lastDeletedBlocks ?? [];
        const result = restoreDeletedParamBlocks({
            state,
        });
        // 没有可恢复的块 = 静默；菜单项已经置灰。
        if (!result.ok) return;
        invalidateParamBlockVisualPreviews("structure");
        setParamBlockState(result.state);
        persistParamBlockEdits(result.state);
        if (rootTrackId !== null) {
            const view = composeParamBlockView(
                rootTrackId,
                clips,
                result.state.editSet,
                result.state.framePeriodMs,
            );
            const restoredIds = new Set(deletedSnapshots.map((snapshot) => snapshot.block.id));
            const ranges = view.blocks
                .filter((block) => restoredIds.has(logicalBlockId(block)))
                .flatMap((block) => logicalFrameRangesFromView(view.blocks, block.id));
            setParamBlockSelection(normalizeSelection(ranges));
            paramBlockAnchorRef.current =
                ranges.length > 0
                    ? { blockId: result.blockId, frame: ranges[0]!.startFrame }
                    : null;
        } else {
            clearParamBlockSelection();
        }
        setParamBlockContextMenu(null);
    }, [
        clearParamBlockSelection,
        invalidateParamBlockVisualPreviews,
        paramBlockStateRef,
        persistParamBlockEdits,
        requireBlockPersistenceReady,
        rootTrackId,
        setParamBlockSelection,
        setParamBlockState,
    ]);
    useLatestRefValue(restoreParamBlocksRef, restoreParamBlocks);
    const applyParamBlockEdit = useCallback(
        (result: ReturnType<typeof splitParamBlock>, gesture: "split" | "create" = "split") => {
            if (!requireBlockPersistenceReady()) return;
            /*
             * 能走到这里说明前置条件已经通过（目标块由调用方按位置解析），
             * 失败属于实现层问题：开发构建打印，生产静默。
             */
            if (!result.ok) {
                reportParamBlockDiagnostic({
                    code: result.reason,
                    gesture,
                    rootTrackId,
                    blockId: null,
                    frame: null,
                });
                reportParamBlockLocalRejection({
                    code: result.reason,
                    gesture,
                    rootTrackId,
                    tf,
                });
                return;
            }
            invalidateParamBlockVisualPreviews("structure");
            setParamBlockState(result.state);
            persistParamBlockEdits(result.state);
            setParamBlockContextMenu(null);
        },
        [
            invalidateParamBlockVisualPreviews,
            persistParamBlockEdits,
            requireBlockPersistenceReady,
            rootTrackId,
            setParamBlockState,
            tf,
        ],
    );
    useLatestRefValue(applyParamBlockEditRef, applyParamBlockEdit);
    /**
     * 在播放头处分割。
     *
     * 【操作数由位置决定】块在时间上互不重叠，播放头最多落在一个块里 —— "切哪个块"
     * 天然唯一。要求先选中一个块是多余的一步，还会在 Slip 之后拿旧几何当判据
     * （"切分点离边缘太近"正是这么来的）。与时间轴的 `resolveSplitTargetsAtSec`
     * 同一条思路：位置解析优先，解析不到就静默。
     *
     * 【前置条件不满足 = 静默】没有块 / 编组块 / 切点贴边：没有提示、没有 IPC、
     * 不改选择。只有后端真的拒绝才进状态栏。
     */
    const splitParamBlockAtSec = useCallback(
        (args: {
            readonly rawSec: number;
            readonly targetBlockId?: string;
            readonly snapToggleHeld?: boolean;
            readonly movePlayhead?: boolean;
            readonly selectResult?: boolean;
            /**
             * "none" = exact playhead split; "grid" = click split with
             * grid / clip-edge snapping, but never the playhead itself.
             */
            readonly snapMode?: "none" | "grid";
        }): boolean => {
            if (rootTrackId === null) return false;
            const state = paramBlockStateRef.current;
            const clips = paramBlockSourceClipsRef.current;
            /*
             * Compose from the live command operands. React may not have
             * committed the previous split yet; reading paramBlockViewRef here
             * would split the pre-previous geometry.
             */
            const view = composeParamBlockView(
                rootTrackId,
                clips,
                state.editSet,
                state.framePeriodMs,
            );
            const rawSec = args.rawSec;
            const framePeriodMs = view.framePeriodMs;
            const blocks = blocksFromViewForPanel(rootTrackId, view);
            /*
             * A target id is only a hint. If it no longer strictly contains the
             * split position, the gesture must fail rather than silently split
             * a different block that happens to share an id.
             */
            const target =
                args.targetBlockId === undefined
                    ? resolveParamBlockSplitTarget(blocks, rawSec, framePeriodMs)
                    : (blocks.find(
                          (block) =>
                              (block.id === args.targetBlockId ||
                                  logicalBlockId(block) === args.targetBlockId) &&
                              /*
                               * 与播放头分支同一口径（整数帧）。旧实现在这里用浮点秒判
                               * 包含，边界上（rawSec 与块起点相差不到一帧）两个分支会给出
                               * 不同结论 —— 表现为「位置明显在块内，却提示切分点太靠边」。
                               */
                              isStrictlyInsideParamBlock(block, rawSec, framePeriodMs),
                      ) ?? null);
            if (target === null) return false;
            const selectionBefore = selectionRef.current;
            const snapToggleHeld = args.snapToggleHeld === true;
            const pxPerSec = hostRef.current?.getAxis()?.pxPerSec ?? pxPerSecRef.current;
            const snapMode = args.snapMode ?? "grid";
            let splitSec = rawSec;
            if (snapMode === "grid") {
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
                    includePlayhead: false,
                });
                const snapped = resolveParamBlockSplitSec({
                    sec: rawSec,
                    snapSec: () => snap.sec,
                    enabled: s.timelineSnap.enabled !== snapToggleHeld,
                    pxPerSec,
                    thresholdPx: 12,
                });
                if (isStrictlyInsideParamBlock(target, snapped, framePeriodMs)) splitSec = snapped;
            }
            // A split is a click, not a drag: never publish a snap guide here.
            // The playhead split is exact; the click split may move to a grid /
            // clip edge, but it never publishes the playhead line.
            const plan = planParamBlockGesture({
                clips,
                edits: state.editSet,
                framePeriodMs,
                gesture: {
                    kind: "split",
                    blockId: target.id,
                    frame: toFrame(splitSec, framePeriodMs),
                },
            });
            if (!plan.ok) {
                reportParamBlockDiagnostic({
                    code: plan.error.code,
                    gesture: "split",
                    rootTrackId,
                    blockId: target.id,
                    frame: toFrame(splitSec, framePeriodMs),
                    ...(plan.error.detail === undefined ? {} : { detail: plan.error.detail }),
                });
                if (plan.error.code === "split_too_close_to_edge") {
                    publishParamBlockFeedback({
                        text: tf("status_param_block_split_too_close"),
                        tone: "warning",
                        holdMs: 3000,
                    });
                } else {
                    reportParamBlockLocalRejection({
                        code: plan.error.code,
                        ...(plan.error.detail === undefined ? {} : { detail: plan.error.detail }),
                        gesture: "split",
                        rootTrackId,
                        blockId: target.id,
                        tf,
                    });
                }
                return false;
            }
            const next: ParamBlockState = { ...state, editSet: plan.plan.edits };
            invalidateParamBlockVisualPreviews("structure");
            setParamBlockState(next);
            persistParamBlockEdits(next);
            if (args.movePlayhead === true) {
                dispatch(setPlayheadSec(splitSec));
                void dispatch(seekPlayhead(splitSec));
            }
            const leftBlockId = plan.plan.leftBlockId;
            const rightBlockId = plan.plan.rightBlockId;
            if (args.selectResult !== false) {
                const nextSelection = resolveParamBlockSplitSelection({
                    beforeBlocks: view.blocks,
                    beforeSelection: selectionBefore,
                    splitFrame: toFrame(splitSec, framePeriodMs),
                    targetLogicalId: logicalBlockId(target),
                    leftBlockId,
                    rightBlockId,
                    viewAfter: plan.plan.viewAfter,
                    rightSpan: plan.plan.rightSpan,
                });
                setParamBlockSelection(nextSelection.selection);
                paramBlockAnchorRef.current = nextSelection.anchor;
            }
            bumpParamBlockStructure();
            setParamBlockContextMenu(null);
            return true;
        },
        [
            bumpParamBlockStructure,
            dispatch,
            invalidateParamBlockVisualPreviews,
            paramBlockClips,
            paramBlockStateRef,
            persistParamBlockEdits,
            pxPerSecRef,
            setParamBlockState,
            rootTrackId,
            s.bpm,
            s.beats,
            s.multiSelectedClipIds,
            selectionRef,
            setParamBlockSelection,
            tf,
            s.project.gridSize,
            s.timelineSnap,
            s.tempoMap,
            s.tracks,
        ],
    );
    useLatestRefValue(splitParamBlockAtSecRef, splitParamBlockAtSec);
    const splitParamBlockAtPlayhead = useCallback(
        (snapToggleHeld = false) => {
            const rawSec = store.getState().session.playheadSec;
            splitParamBlockAtSec({
                rawSec,
                snapToggleHeld,
                movePlayhead: false,
                snapMode: "none",
            });
        },
        [splitParamBlockAtSec, store],
    );
    useLatestRefValue(splitParamBlockAtPlayheadRef, splitParamBlockAtPlayhead);
    /** 实时求值：播放头移动 / 块结构变化后，菜单打开即得到正确可用态。 */
    const canSplitParamBlockAtPlayhead = useCallback((): boolean => {
        const view = composeParamBlockViewNow();
        if (view === null || rootTrackId === null) return false;
        return (
            resolveSplittableParamBlockAt(
                blocksFromViewForPanel(rootTrackId, view),
                store.getState().session.playheadSec,
                view.framePeriodMs,
                {
                    clips: paramBlockSourceClipsRef.current,
                    edits: paramBlockStateRef.current.editSet,
                },
            ) !== null
        );
    }, [composeParamBlockViewNow, paramBlockStateRef, rootTrackId, store]);
    const gameRestoreRanges = paramBlockRestoreRanges(
        paramBlockView?.blocks ?? [],
        selectedParamBlockIds,
        selectionUi,
    );
    const canRestoreGameParamBlocks = gameRestoreRanges.length > 0;
    const restoreGameParamBlocks = useCallback(() => {
        if (!requireBlockPersistenceReady()) return;
        const state = paramBlockStateRef.current;
        const view = composeParamBlockViewNow();
        if (view === null || !rootTrackId) return;
        const ranges = paramBlockRestoreRanges(
            view.blocks,
            selectedParamBlockIdsNow(),
            selectionRef.current,
        );
        if (ranges.length === 0) return;
        const requestRoot = rootTrackId;
        invalidateParamBlockVisualPreviews("structure");
        const request = timelineApi.restoreGameParamBlocks({
            rootTrackId,
            ranges,
            checkpoint: true,
        });
        void trackParamBlockCommit("restore", requestRoot, request)
            .then((reply) => {
                if (rootTrackIdRef.current !== requestRoot) return;
                /*
                 * 【必须把回包交回持久化钩子】这条写入绕过了 `commit`，服务端代次
                 * 已经前进而钩子里的权威镜像还停在旧值 —— 下一次提交就会以过期基准
                 * 被 CAS 拒绝（表现为"还原之后编辑点了没反应"）。`adopt` 走的是
                 * 同一条落地路径（含单调闸与手势守卫），不需要多一次 IPC。
                 */
                blockPersistence.adopt(rootTrackId, reply);
                const next: ParamBlockState = {
                    ...state,
                    rootTrackId,
                    editSet: reply.edits,
                    framePeriodMs: reply.view.framePeriodMs,
                };
                setParamBlockState(next);
                setParamBlockSelection(expandParamBlockSelection(reply.view.blocks, ranges));
                const restored = reply.view.blocks.find((block) =>
                    ranges.some(
                        (range) =>
                            block.startFrame < range.startFrame + range.frameCount &&
                            block.endFrameExclusive > range.startFrame,
                    ),
                );
                paramBlockAnchorRef.current =
                    restored === undefined
                        ? null
                        : { blockId: logicalBlockId(restored), frame: restored.startFrame };
                dispatch(markProjectEdited());
            })
            .catch((error) => {
                console.error("[pianoRoll] restore blocks failed", error);
                /* 这条命令不走块写入队列（整段还原由后端直接落盘），所以由这里上报。 */
                publishParamBlockWriteFailure({
                    failure: "transport",
                    error: {
                        code: "restore_failed",
                        detail: backendDetail(error) ?? String(error),
                    },
                    tf,
                });
            });
        setParamBlockContextMenu(null);
    }, [
        blockPersistence,
        composeParamBlockViewNow,
        dispatch,
        invalidateParamBlockVisualPreviews,
        paramBlockStateRef,
        rootTrackId,
        rootTrackIdRef,
        selectedParamBlockIdsNow,
        selectionRef,
        setParamBlockSelection,
        setParamBlockState,
        requireBlockPersistenceReady,
        tf,
    ]);
    useLatestRefValue(restoreGameParamBlocksRef, restoreGameParamBlocks);

    /**
     * 把一段音高整体平移到最近钢琴键（半音），保留颤音 / 滑音轮廓。
     * 写回全分辨率帧，撤销点由 `uploadFullResCurveSegments` 的首块提交建立。
     */
    const normalizeParamBlockPitch = useCallback(
        async (block: ParamBlock): Promise<void> => {
            if (!rootTrackId || editParam !== "pitch" || !requireBlockPersistenceReady()) return;
            try {
                /**
                 * 队列槽内的准备：逐窗口读全分辨率曲线 → 归一到最近半音。
                 *
                 * 【为什么必须进队列】这条写入走 `set_param_frames`（无 CAS），但它
                 * 同样推进 `param_curve_revision`。绕过队列就会暗算任何同时在飞的
                 * CAS 提交 —— 旧实现正是这样把「归一化音高」变成概率性失败的。
                 */
                const prepareNormalize = async (): Promise<ParamBlockWritePreparation> => {
                    const ranges = paramBlockLogicalFrameRanges(selectionFrameBlocks, block.id);
                    const segments: Array<{ startFrame: number; values: number[] }> = [];
                    let modeCenter = block.averageValue;
                    for (const range of ranges) {
                        const startFrame = Math.max(0, range.startFrame);
                        const endFrame = Math.max(
                            startFrame,
                            range.startFrame + range.frameCount - 1,
                        );
                        const curve = await fetchFullResCurve({
                            trackId: rootTrackId,
                            param: "pitch",
                            startFrame,
                            endFrame,
                            paramView,
                        });
                        if (modeCenter === null && curve.values.length > 0) {
                            modeCenter = curve.values.find((value) => value > 0) ?? null;
                        }
                        const normalized = normalizePitchToNearestSemitone({
                            values: curve.values,
                            modeCenter,
                        });
                        if (normalized.every((value, index) => value === curve.values[index]))
                            continue;
                        segments.push({ startFrame: curve.startFrame, values: normalized });
                    }
                    return {
                        state: paramBlockStateRef.current,
                        checkpoint: true,
                        curveOnly: { param: "pitch", segments },
                    };
                };
                await blockPersistence.commitPlan({
                    kind: "normalize",
                    rootTrackId,
                    prepare: prepareNormalize,
                });
                bumpRefreshToken();
                clearStaleBlockCentersAfterParamEdit("normalizePitch");
            } catch (error) {
                console.error("[pianoRoll] normalize block pitch failed", error);
                reportParamBlockWriteFailure({ error, gesture: "normalize", rootTrackId });
            }
        },
        [
            blockPersistence,
            bumpRefreshToken,
            editParam,
            paramBlockStateRef,
            paramView,
            clearStaleBlockCentersAfterParamEdit,
            requireBlockPersistenceReady,
            rootTrackId,
            selectionFrameBlocks,
        ],
    );
    useLatestRefValue(normalizeSelectedParamBlockPitchRef, () => {
        if (selectedParamBlockIds.length !== 1) return;
        const block = paramBlocksWithValues.find(
            (item) => logicalBlockId(item) === selectedParamBlockIds[0],
        );
        if (block !== undefined) void normalizeParamBlockPitch(block);
    });
    /**
     * 双击块 = 把整块音高归一到最近半音。切分保留右键菜单 / S 键。
     *
     * 参数块选择模式下 pointerdown 会接管指针，但左键不能调用 preventDefault，
     * 否则兼容 mouse 事件被抑制、dblclick 永远不派发。
     */
    const onCanvasDoubleClick = useCallback(
        (event: React.MouseEvent<HTMLCanvasElement>) => {
            if (!paramBlockToolActive) return;
            if (performance.now() < suppressBlockDoubleClickUntilRef.current) {
                event.preventDefault();
                return;
            }
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
             * 双击事件在参数块选择模式里可能被 pointerdown 的“选中 + 捕获”抢先处理；
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
            if (hit === null && candidateHit === null) return;
            /*
             * 双击块 = **无操作**。旧实现双击就做音高归一化 —— 一条不可逆的曲线
             * 重写：块内每个音高事件被拉到块均值，没有确认、没有预览，只能撤销
             * 找回。归一化仍保留在右键菜单与快捷键里（显式意图）。
             */
            paramBlockClickCandidateRef.current = null;
        },
        [
            editParam,
            hostRef,
            invalidateParamBlockSpec,
            paramBlockHitSlopPx,
            paramBlockStateRef,
            paramBlockToolActive,
            paramBlocksWithValues,
            persistBlockTransition,
            persistPitchTransitionCurve,
        ],
    );
    const onCanvasPointerMoveForBlocks = useCallback(
        (event: React.PointerEvent<HTMLCanvasElement>) =>
            handleParamBlockHover(
                {
                    paramBlockToolActive,
                    paramBlockHitSlopPx,
                    paramBlocksWithValues,
                    hostRef,
                    hoveredParamBlockIdRef,
                    hoveredTransitionLeftIdRef,
                    isCreateGesture: () => paramBlockResizeRef.current?.blockId === "__create__",
                    getDefaultCanvasCursor: interactions.getDefaultCanvasCursor,
                    setCanvasCursor,
                    invalidate,
                },
                event,
            ),
        [
            interactions.getDefaultCanvasCursor,
            invalidate,
            paramBlockHitSlopPx,
            paramBlockToolActive,
            paramBlocksWithValues,
            setCanvasCursor,
        ],
    );
    useLatestRefValue(paramBlockPointerRef, onParamBlockPointerDown);
    /**
     * 块命令的唯一执行入口。键盘通道与右键菜单共用（见 paramBlockEditOps）。
     *
     * 【为什么 handlers 用 ref 包一层】这些回调里有一半是在
     * `onParamBlockPointerDown` 之后才定义的（`groupSelectedParamBlocks` /
     * `splitParamBlockAtPlayhead` …）。用 ref 转发可以避免「为了声明顺序把 1900 行
     * 回调往上搬」。
     */
    const paramBlockOpHandlersRef = useRef<ParamBlockOpHandlers | null>(null);
    paramBlockOpHandlersRef.current = {
        split: ({ snapToggleHeld }) => splitParamBlockAtPlayheadRef.current?.(snapToggleHeld),
        createFromSelection: () => createParamBlocksFromSelectionRef.current?.(),
        group: () => groupSelectedParamBlocksRef.current?.(),
        ungroup: () => ungroupSelectedParamBlocksRef.current?.(),
        delete: () => deleteSelectedParamBlocksRef.current?.(),
        selectAll: () => selectAllParamRange(),
        deselect: () => clearParamBlockSelection(),
        restoreGame: () => restoreGameParamBlocksRef.current?.(),
        restoreDeleted: () => restoreParamBlocksRef.current?.(),
        normalizePitch: () => normalizeSelectedParamBlockPitchRef.current?.(),
    };
    const runParamBlockOp = useCallback(
        (op: ParamBlockMenuOp, context: { readonly snapToggleHeld: boolean }) => {
            const handlers = paramBlockOpHandlersRef.current;
            if (handlers !== null) runParamBlockMenuOp(op, handlers, context);
        },
        [],
    );
    useParamBlockEditOpChannel({ keybindings: mergedKeybindings, runOp: runParamBlockOp });
    /**
     * 块层数据是命令式写入宿主的；用受控 effect 在选中集合变化时立即冲刷，
     * 避免 React 选择态已经更新、宿主仍持有上一个 spec 的情况。
     */
    useEffect(() => {
        hostRef.current?.invalidate();
    }, [hostRef, selectedParamBlockIds]);
    const paramBlockSpec = useParamBlockRenderSpec({
        layerVisible: paramBlocksLayerVisible,
        structureTick: paramBlockStructureTick,
        epoch: paramBlockEpoch,
        visualTransactionRef: paramBlockVisualTransactionRef,
        resizeRef: paramBlockResizeRef,
        dragSnapshotRef: paramBlockDragSnapshotRef,
        activeGestureTokenRef: paramBlockActiveGestureTokenRef,
        dragCommitRef: paramBlockDragCommitRef,
        valueCacheRef: paramBlockValueCacheRef,
        latestSpecRef: latestParamBlockSpecRef,
        canonicalBlocks: paramBlocksWithValues,
        paramView,
        selectedIds: paramBlockRenderSelectedIds,
        editParam,
        colors: paramBlockSpecColors,
        bounds,
        valueToY,
        framePeriodMs: paramBlockStateRef.current.framePeriodMs,
        curveOwned: paramBlockViewRef.current !== null,
        buildDragSnapshotSpec: paramBlockSpecForDragSnapshot,
    });
    /*
     * 视觉事务收尾：**唯一**判据是「这次提交产生的权威已经可见」。
     *
     * 【为什么不能用「本地派生几何已经等于预览几何」】`setParamBlockState` 是同步
     * 提交，那个条件在 pointerup 之后**一次渲染**就成立。于是预览被撤下、曲线回到
     * 旧值、块按旧曲线重新采样 —— 用户看到「先回弹旧参数、再生效新参数」。过渡
     * 手柄更糟：它的几何根本不变化，同一次渲染就满足旧判据。
     *
     * 判据搬到 `shouldReleaseParamBlockPreview` 之后，带曲线补丁的提交必须等
     * 「曲线 revision 前进 ∧ 权威水位确认 ∧ 当前窗口包含提交片段」三者同时成立。
     *
     * 【超时】超时前必须先把权威读回来（`flushReadback` + `clearLiveEditOverride`），
     * 否则撤下预览的第一帧仍然是旧曲线 —— 那正是要消灭的现象。超时同时记一条
     * 诊断，便于定位「为什么这条路径没等到权威」。
     */
    useEffect(() => {
        const tx = paramBlockVisualTransactionRef.current;
        if (tx === null) return;
        const decision = shouldReleaseParamBlockPreview({
            transaction: tx,
            now: performance.now(),
            geometrySettled: isParamBlockPreviewGeometrySettled({
                transaction: tx,
                blocks: paramBlocksWithValues,
            }),
            gestureActive: paramBlockDragActivityRef.current.pointerGestureToken !== null,
            pendingCommits: paramBlockDragActivityRef.current.pendingCommits,
            authority: paramAuthoritySnapshot,
            watermark: getAuthorityWatermark(),
            paramView,
        });
        if (decision === "wait") {
            const remaining = Math.max(
                0,
                PARAM_BLOCK_PREVIEW_RELEASE_TIMEOUT_MS - (performance.now() - tx.startedAt),
            );
            const timer = window.setTimeout(() => {
                if (paramBlockVisualTransactionRef.current !== tx) return;
                invalidateParamBlockVisualPreviews("timeout");
            }, remaining);
            return () => window.clearTimeout(timer);
        }
        if (decision === "timeout") {
            reportParamBlockDiagnostic({
                code: "preview_release_timeout",
                gesture: tx.kind === "create" ? "create" : tx.kind === "drag" ? "drag" : "resize",
                rootTrackId,
                blockId: null,
                frame: null,
                detail: `kind=${tx.kind}`,
            });
            // 对账：把权威编辑集写回状态（进而触发曲线窗口重取），再撤预览。
            blockPersistence.flushReadback();
            clearLiveEditOverride();
        }
        paramBlockVisualTransactionRef.current = null;
        paramBlockInteractionCurveRef.current = null;
        const pvNow = paramViewRef.current;
        if (pvNow !== null) resetLiveEditPreview(pvNow);
        bumpParamBlockStructure();
        invalidate();
    }, [
        blockPersistence,
        bumpParamBlockStructure,
        clearLiveEditOverride,
        getAuthorityWatermark,
        invalidate,
        invalidateParamBlockVisualPreviews,
        paramAuthoritySnapshot,
        paramBlockStructureTick,
        paramBlocksWithValues,
        paramView,
        paramViewRef,
        resetLiveEditPreview,
        rootTrackId,
    ]);
    /*
     * 新建块预览 -> 正式块的交接。
     *
     * 【判据与超时都归 resolveParamBlockRelease】上一版这里内联了同一组三元组，
     * 但**忘了超时**：权威永远不包含提交片段时（例如提交后立刻被另一笔编辑覆盖），
     * 预览会永久停留 —— 屏幕上留一个只能切参数才消失的幽灵块。
     */
    useEffect(() => {
        const handoff = paramBlockCreateCommitRef.current;
        if (handoff === null || !handoff.committed) return;
        if (paramBlockResizeRef.current?.blockId !== "__create__") return;
        const decision = resolveParamBlockRelease({
            wait: {
                curveRevisionAtStart: handoff.curveRevisionAtStart,
                curveRevisionCommitted: handoff.curveRevisionCommitted,
                authorityRequestSeqAtCommit: handoff.authorityRequestSeqAtCommit,
                committedSegments: handoff.requiresCurveAuthority
                    ? (handoff.committedSegments ?? null)
                    : undefined,
            },
            authority: paramAuthoritySnapshot,
            watermark: getAuthorityWatermark(),
            paramView,
            geometrySettled: paramBlocksWithValues.some((block) => block.id === handoff.blockId),
            gestureActive: paramBlockDragActivityRef.current.pointerGestureToken !== null,
            pendingCommits: paramBlockDragActivityRef.current.pendingCommits,
            startedAt: handoff.startedAt,
            now: performance.now(),
        });
        if (decision === "wait") {
            /* 超时前必须让 effect 再跑一次，否则没人来判这一次超时。 */
            const remaining = Math.max(
                0,
                PARAM_BLOCK_PREVIEW_RELEASE_TIMEOUT_MS - (performance.now() - handoff.startedAt),
            );
            const timer = window.setTimeout(() => bumpParamBlockStructure(), remaining);
            return () => window.clearTimeout(timer);
        }
        if (decision === "timeout") {
            reportParamBlockDiagnostic({
                code: "preview_release_timeout",
                gesture: "create",
                rootTrackId,
                blockId: handoff.blockId,
                frame: null,
                detail: "kind=create-handoff",
            });
            /* 对账：把权威编辑集写回状态（进而触发曲线窗口重取），再撤预览。 */
            blockPersistence.flushReadback();
            clearLiveEditOverride();
        }
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
            paramBlockVisualTransactionRef.current = null;
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
        blockPersistence,
        bumpParamBlockStructure,
        clearLiveEditOverride,
        getAuthorityWatermark,
        invalidateParamBlockSpec,
        paramAuthoritySnapshot,
        paramView,
        resetLiveEditPreview,
        rootTrackId,
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
        authority: paramAuthoritySnapshot,
        getAuthorityWatermark,
        structureVersion: paramBlockStructureTick,
        clearLiveEditOverride,
        onReleased: bumpParamBlockStructure,
    });
    const drawParamBlocks = useCallback(
        (spec: PianoRollParamBlockSpec | null) => {
            const hoveredBlockId = hoveredParamBlockIdRef.current;
            const hoveredTransitionId = hoveredTransitionLeftIdRef.current;
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
    }, [paramBlockToolActive, paramBlocksLayerVisible]);
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
        currentSelectToolIcon,
        paramBlockToolIcon,
    } = buildToolIcons(currentDrawTool, s.selectToolMode);

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

    const paramBlockMenuContext = useMemo<
        ParamBlockCommandContext & { readonly isVolume: boolean; readonly isDyn: boolean }
    >(
        () => ({
            canSplit: canSplitParamBlockAtPlayhead,
            canGroup: canGroupParamBlocks(paramBlocksWithValues, selectedParamBlockIds),
            canUngroup: paramBlocksWithValues.some(
                (block) =>
                    block.kind === "grouped" && selectedParamBlockIdSet.has(logicalBlockId(block)),
            ),
            canDelete: selectedParamBlockIds.length > 0,
            canRestoreGame: canRestoreGameParamBlocks,
            canRestoreDeleted: (paramBlockState.lastDeletedBlocks ?? []).length > 0,
            canNormalizePitch: editParam === "pitch" && selectedParamBlockIds.length === 1,
            hasSelection: selectionUi !== null && selectionUi.length > 0,
            isPitch: editParam === "pitch",
            isVolume: editParam === "volume",
            isDyn: isDynParam(editParam),
        }),
        [
            canRestoreGameParamBlocks,
            canSplitParamBlockAtPlayhead,
            editParam,
            paramBlockState.lastDeletedBlocks,
            paramBlocksWithValues,
            selectedParamBlockIdSet,
            selectedParamBlockIds,
            selectionUi,
        ],
    );
    const handleParamBlockMenuAction = useCallback(
        (channel: "paramBlock" | "edit", op: string) => {
            if (channel === "edit") {
                switch (op) {
                    case "transposeCents":
                        openEditDialog("transposeCents");
                        return;
                    case "transposeDegrees":
                        openEditDialog("transposeDegrees");
                        return;
                    case "setPitch":
                        openEditDialog("setPitch");
                        return;
                    case "average":
                        openEditDialog("average");
                        return;
                    case "smooth":
                        openEditDialog("smooth");
                        return;
                    case "quantize":
                        openEditDialog("quantize");
                        return;
                    case "meanQuantize":
                        openEditDialog("meanQuantize");
                        return;
                    case "addVibrato":
                        openVibratoDialog("apply");
                        return;
                    case "saveAsPitchRef":
                        void handleSaveAsPitchRef();
                        return;
                    case "exportMidi":
                        void handleExportMidiFromEditor();
                        return;
                    default:
                        void handleEditOp(op);
                        return;
                }
            }
            /*
             * 与键盘通道共用同一张 op 表（paramBlockEditOps）。这里**不再**写第二份
             * switch —— 上一版两边各写一份，selectAll / deselect 只在键盘侧存在，
             * 而键盘侧因为通道判据写错整条失效，于是「菜单能用、快捷键不行」。
             */
            if (isParamBlockMenuOp(op)) runParamBlockOp(op, { snapToggleHeld: false });
            setParamBlockContextMenu(null);
        },
        [
            handleEditOp,
            handleExportMidiFromEditor,
            handleSaveAsPitchRef,
            openEditDialog,
            openVibratoDialog,
            runParamBlockOp,
        ],
    );

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
        currentSelectToolIcon,
        cycleDragDirectionKb,
        devicePixelRatio,
        dispatch,
        drawToolButtonTitle,
        drawToolKb,
        drawToolMenuAnchorRef,
        drawToolMenuOpen,
        drawToolMenuRef,
        editParam,
        selectToolMenuAnchorRef,
        selectToolMenuOpen,
        selectToolMenuRef,
        setSelectToolMenuOpen,
        selectToolKb,
        blockSelectionKb,
        selectToolButtonTitle,
        paramBlockToolIcon,
        paramBlockToolActive,
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
        paramBlocksLayerVisible,
        paramBlockContextMenu,
        paramBlockMenuContext,
        handleParamBlockMenuAction,
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
            (block) =>
                block.kind === "grouped" && selectedParamBlockIdSet.has(logicalBlockId(block)),
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
                (item) => logicalBlockId(item) === selectedParamBlockIds[0],
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
        toggleClipPitchOverlay,
        clipPitchOverlayVisible,
        toggleAlwaysShowParamBlocks,
        alwaysShowParamBlocks,
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
        axisPanDirectionToggleAvailable,
        axisToggleAvailable,
        axisUnitToggleAvailable,
        axisValuePreview,
        axisWrapRef,
        canvasCursor,
        canvasRef,
        captureBandsRef,
        clipPeaks,
        contentWidth,
        canFlipParam: supportsParamFlip(currentParamRange, currentParamDefaultValue),
        currentParamRange,
        dynamicProjectSec,
        editParamAxisDirection,
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
        clipPitchOverlays,
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
