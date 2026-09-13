/**
 * TimelinePanel — Timeline 面板 UI 组件（精简后）
 *
 * 所有业务逻辑已拆分至 4 个 hook：
 * - useTimelineState        → state / ref / viewport / scroll / 坐标转换
 * - useTimelineDragDrop     → Tauri 原生拖放 + 文件浏览器面板自定义拖拽
 * - useTimelineClipActions  → Clip 多选 + 操作回调
 * - useTimelineEventHandlers→ 全局事件监听
 *
 * 此文件只保留：JSX 渲染 + 胶水 + 拖拽 hooks 桥接
 *
 * 【播放头桥接】`TimelineTransportBridge` 逐帧写标尺播放头（React 声明式元素），
 * 并调用 `kernelHostRef.invalidatePlayhead()` 请求内核重绘**轨道区**播放头——
 * 后者只在宿主 `draw()` 内被写，而内核渲染循环是纯脏标记驱动的，镜像变化不会
 * 自动标脏（不请求就永不重绘）。
 *
 * 【渲染路径】轨道区**恒由** `TimelineKernelView`（WebGL2 渲染内核）承载。
 * 旧实现（原生滚动 + Canvas2D 的 `TimelineScrollArea` / `TimelineSurface` /
 * `TrackLane` / `ClipItem` 等）已随"唯一路径"改造删除，此处**没有运行期二选一、
 * 也没有可回退的第二条路径**。
 *
 * 【失败处理】WebGL2 不可用时内核视图回报失败（`handleKernelUnavailable`），本面板改
 * 渲染 `KernelUnavailableNotice` —— 一个可自助排障的界面（原因 + 排查清单 + 可复制的
 * 诊断信息）。实测 WebGL2 在纯 CPU 环境（含无 GPU）可用（ANGLE + SwiftShader），只有
 * 显式禁止软件光栅一类配置才会失败。
 *
 * 【历史背景（勿删）】内核曾是 opt-in 且默认值跟随 `import.meta.env.DEV`，导致打包后
 * **静默退回旧渲染器**（Phase 3 计划 R8）。结论是"绝不静默降级"：失败必须显式告知。
 *
 * @see docs/superpowers/specs/2026-09-13-timeline-single-path-design.md
 */
import React, { useMemo, Profiler } from "react";
import { Flex, Dialog, Button, Text } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppTheme } from "../../theme/AppThemeProvider";
import { useAppSelector } from "../../app/hooks";
import { shallowEqual } from "react-redux";
import { isModifierActive } from "../../features/keybindings/keybindingsSlice";
import { resolveClipDragCopyMode } from "./timeline/hooks/clipDragCopyMode";
import { copyClipsFromDrag } from "./timeline/hooks/copyClipsFromDrag";
import {
    createNewTrackForKernelDrop,
    createTrackIdsForDrop,
} from "./timeline/hooks/createNewTrackForDrop";
import { resolveKernelDropTarget } from "./timeline/hooks/kernelDropCommit";
import { normalizedTrackColorCss } from "./timeline/runtime/timelineCanvasStyle";
import {
    defaultFadeDirFor,
    FADE_PRESETS,
    resolveCurvatureEditBase,
    resolveCurvePointer,
    solveNearestCurveDir,
} from "./timeline/reaperFade";
import {
    buildCrossfadeGripInfoContent,
    buildSingleFadeInfoContent,
    publishFadeRichTooltip,
    type FadeLabelLookup,
    type FadeLengthFormatContext,
} from "./timeline/fadeTooltipText";
import { effectiveFadeSec } from "./timeline/kernel/interaction/fadeTargets";
import type { ClipHitRegion } from "./timeline/kernel/interaction/hitTest";
import type { ClipHeaderControl } from "./timeline/kernel/interaction/clipHeaderControls";
import { FadeContextMenuHost } from "./timeline/FadeContextMenuHost";
import {
    requestOpenFadeContextMenu,
    type FadeContextMenuRequest,
} from "./timeline/fadeContextMenuBus";
import { createPortal } from "react-dom";
import {
    addTrackRemote,
    closeClipFormantToolWindow,
    openClipFormantToolWindow,
    duplicateTrackRemote,
    removeTrackRemote,
    selectTrackRemote,
    setClipFormantToolWindowPosition,
    setTrackStateRemote,
    seekPlayhead,
    moveTrackRemote,
    setClipMuted,
    importAudioAtPosition,
    importAudioFileAtPosition,
    importMidiAsClip,
    replaceMidiClipDataRemote,
    importMultipleAudioAtPosition,
    setClipStateRemote,
    setClipsStateBulkRemote,
    setClipFades,
    setClipGain,
    setClipPlaybackRate,
    setClipActiveTakeRemote,
    glueClipsRemote,
    convertClipsToPitchReferenceRemote,
    updatePitchReferenceRemote,
    removeClipsRemote,
    closeTrackGapsRemote,
    persistUiSettings,
    setPrimaryTimeUnit,
    setSecondaryTimeUnit,
    setTempoMap,
    setTrackName,
    setTrackVolume,
    setPendingPlayheadReveal,
    setplayheadSec,
    moveClipStart,
    moveClipTrack,
    checkpointHistory,
    bumpParamsEpoch,
    setClipLength,
    setClipSourceRange,
    setClipSnapOffset,
    setClipAutoFades,
    selectClipRemote,
    beginInteraction,
    endInteraction,
} from "../../features/session/sessionSlice";
import { beginSnapGesture, endSnapGesture } from "../../utils/timelineSnapping";
import { batch } from "react-redux";
import { moveClipRemote, moveClipsRemote } from "../../features/session/thunks/timelineThunks";
import { computeTimelineRectSelection } from "./timeline/useTimelineSelectionRect";
import { setTempoMapRemote } from "../../features/session/thunks/tempoMapThunks";

import { NEW_TRACK_SENTINEL } from "./timeline/hooks/useClipDrag";
import { getBulkEditableClipIds } from "./timeline/hooks/bulkClipEdit";
import { registerDragAbort } from "./timeline/gestureFocusGuard";
import { getInsertBelowTargetIndex } from "./timeline/trackContextMenuPlacement";
import { collectFadeContextClips } from "./timeline/clipFadeContext";
import { emitExternalFileAction } from "../../features/session/projectOpenEvents";
import { webApi } from "../../services/webviewApi";
import { useClipPitchDrag } from "./timeline/hooks/useClipPitchDrag";
import { AppTooltipBubble } from "../AppTooltip";
import { formatPitchDragCents } from "./timeline/clipPitchDrag";
import { coreApi } from "../../services/api/core";
import { paramsApi } from "../../services/api/params";
import { resolveRootTrackId } from "../../features/session/trackUtils";
import { SCALE_NOTES } from "../../utils/musicalScales";
import { QuickClipExportDialog } from "./QuickClipExportDialog";
import { SilenceDetectionDialog } from "./timeline/SilenceDetectionDialog";
import { MidiTrackSelectDialog } from "./MidiTrackSelectDialog";

import {
    ClipContextMenu,
    TrackAreaContextMenu,
    TimeRuler,
    TrackList,
    detectExternalPathAction,
    extractLocalFilePath,
    formatCursorTime,
    hasFileDrag,
} from "./timeline";
import { timeRulerHeightPx } from "./timeline/rulerHeight";
import type { TimeFormatContext, TimeUnit, TimeUnitChoice } from "./timeline";
import { formatEditNumber, formatGainDbValue, gainToDb } from "./timeline/math";
import { requestResetFadeCurvature } from "./timeline/fadeContextMenuBus";
import { parsePlaybackRateInput } from "./timeline/runtime/timelineCanvasStyle";
import {
    SNAP_HIGHLIGHT_GROUP,
    buildLoopBoundaryHighlightEntry,
    clearSnapHighlights,
    publishSnapHighlights,
} from "../../utils/snapHighlight";
import {
    loopSnapThresholdSec,
    nearestBoundarySnapOffsetSec,
    slipBoundaryAlignedSides,
} from "../../utils/loopSnap";
import type { TempoMap } from "../../utils/tempoMap";
import { TimelineKernelView } from "./timeline/kernel/TimelineKernelView";
import { isKernelAvailable } from "./timeline/kernel/kernelAvailability";
import { KernelUnavailableNotice } from "./timeline/kernel/KernelUnavailableNotice";
import type { TimelineKernelHost } from "./timeline/kernel/host/timelineKernelHost";

import type { ScaleLike } from "../../utils/musicalScales";
import { TimelineDisplaySettingsDialog } from "./TimelineDisplaySettingsDialog";
import { resolveTimelineScrollRange } from "./timeline/runtime/timelineScrollRange";

// ── 拆分出的 hooks ──────────────────────────────────────────
import { useTimelineState } from "./timeline/hooks/useTimelineState";
import { useTimelineDragDrop } from "./timeline/hooks/useTimelineDragDrop";
import {
    createTimelineViewportAccess,
    type TimelineViewportAccess,
} from "./timeline/hooks/timelineViewportAccess";
import {
    applyKernelEditDelta,
    resolveKernelEditParticipants,
    type KernelEditParticipant,
} from "./timeline/hooks/kernelEditSet";
import {
    applyRippleFollowerShift,
    buildRippleFollowers,
    type RippleFollowerMap,
} from "../../features/session/ripplePreview";
import {
    applyAutoCrossfade,
    applyDetachedAutoCrossfadeClears,
    computeInitialCrossfadeSides,
    previewAutoCrossfade,
} from "./timeline/hooks/autoCrossfade";
import { computeEffectiveSnap } from "../../utils/timelineSnapping";
import { store } from "../../app/store";
import { applyBulkFadeValue, applyBulkGainDeltaDb } from "./timeline/hooks/bulkClipEdit";
import { advanceFineAxisDrag, type FineAxisDragState } from "./timeline/fineAxisDrag";
import { CLIP_GAIN_DRAG_DB_PER_PX } from "./timeline/constants";
import {
    buildStretchGroupState,
    computeClipStretch,
    computeRegionRightEdgeDelta,
    computeStretchGroupUpdate,
    scaleSnapOffsetForStretch,
    type StretchGroupState,
} from "./timeline/hooks/stretchGroup";
import {
    stretchLinkedParams,
    stretchTrackLinkedParams,
    type StretchRangeMapping,
} from "./timeline/hooks/stretchParams";
import { computeSlipWindow, readSlipClip, toBoundarySnapClip } from "./timeline/hooks/slipWindow";
import {
    computeCrossfadeGrip,
    type CrossfadeGripClipBase,
} from "./timeline/hooks/crossfadeGripWindow";
import { useTimelineClipActions } from "./timeline/hooks/useTimelineClipActions";
import { useTimelineEventHandlers } from "./timeline/hooks/useTimelineEventHandlers";
import { expandClipIdsWithGroups } from "./timeline/hooks/useGroupExpansion";
import { useVisualPlayhead } from "../../hooks/useVisualPlayhead";
import { useDebouncedPersist } from "../../hooks/useDebouncedPersist";
import { ClipRateEditorDialog } from "./timeline/ClipRateEditorDialog";
import {
    computeAutoFollowScrollLeft,
    computeFocusCursorScrollLeft,
} from "../../utils/autoFollowScroll";
import { readDevicePixelRatio, snapToDevicePx } from "../../utils/devicePixelLine";
import { resolveQuickExportClipIds } from "./timeline/quickExportSelection";
import { isTrackListMirrorEcho } from "./timeline/scrollEcho";
import {
    activeClipTakeName,
    clipDisplayName,
    type ClipFormantMorph,
} from "../../features/session/sessionTypes";
import { ClipFormantToolWindow } from "./timeline/clip/ClipFormantToolWindow";

const TimelineTransportBridge = React.memo(function TimelineTransportBridge(props: {
    pxPerSecRef: React.MutableRefObject<number>;
    rulerPlayheadLineRef: React.MutableRefObject<HTMLDivElement | null>;
    rulerPlayheadHeadRef: React.MutableRefObject<HTMLDivElement | null>;
    /**
     * 模式无关的视口访问器。
     *
     * 自动滚屏与播放头定位都需要「视口宽度 + 当前横向滚动量」，内核模式下这些
     * 量由 `ScrollKernel` 持有（DOM 容器的 scrollLeft 恒为 0），因此不能再直接读
     * 原生 scroller。
     */
    viewport: TimelineViewportAccess;
    /** 接收每帧视觉插值播放头（秒），供缩放锚点等命令式读取（与绘制同源）。 */
    visualPlayheadRef: React.MutableRefObject<number>;
    syncScrollLeft: (next: number) => void;
    autoScrollEnabled: boolean;
    projectSec: number;
    /** 请求时间轴内核重绘播放头（见 kernelHost.invalidatePlayhead 的说明）。 */
    requestPlayheadRepaint: () => void;
}) {
    const {
        pxPerSecRef,
        rulerPlayheadLineRef,
        rulerPlayheadHeadRef,
        viewport,
        visualPlayheadRef,
        syncScrollLeft,
        autoScrollEnabled,
        projectSec,
        requestPlayheadRepaint,
    } = props;
    const transport = useAppSelector(
        (state) => ({
            playheadSec: state.session.playheadSec,
            playheadSampledAtMs: state.session.playheadSampledAtMs,
            isPlaying: state.session.runtime.isPlaying,
            playbackWaitingForRender: state.session.runtime.playbackWaitingForRender,
            playbackPositionSec: state.session.runtime.playbackPositionSec,
        }),
        // 无 shallowEqual 时每次 dispatch 都产生新对象引用，
        // 该桥接组件会在任意 store 更新（含 33Hz 播放轮询）时重渲染。
        shallowEqual,
    );

    // 原地等待渲染（位置冻结）期间不得推进视觉插值：否则 RAF 会以 1x 从
    // 冻结前锚点持续外推整个等待时长，恢复采样到达时光标大幅回跳
    // （往复跳动的根源）。positionSec 条件是等待标志缺省时的兼容兜底。
    const isTransportAdvancing =
        transport.isPlaying &&
        !transport.playbackWaitingForRender &&
        transport.playbackPositionSec > 1e-4;

    useVisualPlayhead({
        syncedPlayheadSec: transport.playheadSec,
        syncedAtMs: transport.playheadSampledAtMs,
        isTransportAdvancing,
        onFrame: React.useCallback(
            (visualPlayheadSec: number) => {
                // 同步共享 ref：缩放锚点（以播放头为锚时的锚点解析）读取的必须是
                // 与绘制同源的插值播放头，否则播放中缩放会以滞后的 store 值锚定
                // 造成跳变。内核宿主也经 `playheadSec` getter 读同一份真值
                // （见 `timelineKernelHost.readPlayheadSec`）。
                visualPlayheadRef.current = visualPlayheadSec;
                const playheadLeftPx = visualPlayheadSec * pxPerSecRef.current;

                // 自动滚动先行：syncScrollLeft 内部会用 Redux 同步播放头（滞后于
                // 视觉插值）重写播放头位置 —— 若先定位播放头再滚动，播放头每帧
                // 会在"视觉位置"与"同步位置"之间跳动（自动滚屏抽搐的根因）。
                // 滚动先行、播放头定位收尾，最终写入获胜。
                if (autoScrollEnabled && transport.isPlaying) {
                    const next = computeAutoFollowScrollLeft({
                        playheadSec: visualPlayheadSec,
                        pxPerSec: pxPerSecRef.current,
                        viewportWidth: viewport.getViewportWidth(),
                        contentWidth: projectSec * pxPerSecRef.current,
                    });
                    if (Math.abs(viewport.getScrollLeft() - next) > 0.5) {
                        // 写后回读实际生效值再广播：跟随滚动接近工程右端时请求值
                        // 可能被载体钳制，跟随视口的图层必须与真实视口同源。
                        syncScrollLeft(viewport.setScrollLeft(next));
                    }
                }

                // 标尺播放头定位（在自动滚动之后，用最新的视觉插值位置）。
                // 写入前吸附到设备像素边界（readDevicePixelRatio 每帧现读，
                // 浏览器缩放/跨屏后下一帧自愈）：分数 DPR 下不吸附的落点相位
                // 随播放连续变化，1/2 物理像素交替 —— 即"播放时粗细不一"。
                // 轨道区播放头不在这里写：它由内核在 draw() 内自绘（见下方重绘请求）。
                const dpr = readDevicePixelRatio();
                if (rulerPlayheadLineRef.current) {
                    rulerPlayheadLineRef.current.style.left = `${snapToDevicePx(playheadLeftPx, dpr)}px`;
                }
                if (rulerPlayheadHeadRef.current) {
                    rulerPlayheadHeadRef.current.style.left = `${snapToDevicePx(playheadLeftPx, dpr)}px`;
                }

                // 请求内核重绘轨道区播放头。
                //
                // 【为什么必须显式请求】内核的渲染循环是纯脏标记驱动的
                // （`renderKernel/renderLoop`：`start()` 不绘制、无常驻 rAF），而
                // 轨道区播放头元素**只在 `draw()` 内被写**。播放头位置经数据镜像
                // 流入内核，镜像变化本身不会标脏——不请求就永远不重绘：点标尺
                // seek 后标尺播放头（React 声明式）动了，网格上的播放头却冻结在
                // 旧位置（缺陷 #2 的根因）。自动滚屏开着时之所以看不到，是因为它
                // 每帧写 scrollLeft 触发滚动订阅顺带标脏；而它默认关闭。
                requestPlayheadRepaint();
            },
            [
                autoScrollEnabled,
                pxPerSecRef,
                rulerPlayheadHeadRef,
                rulerPlayheadLineRef,
                viewport,
                syncScrollLeft,
                transport.isPlaying,
                projectSec,
                visualPlayheadRef,
                requestPlayheadRepaint,
            ],
        ),
    });

    return null;
});

interface TimelinePanelProps {
    midiClipDialogOpen: boolean;
    midiClipPath: string | null;
    midiClipStartSec: number;
    midiClipTrackId: string | null;
    fillGaps: boolean;
    multiTrackMerge: boolean;
    importBpmAsProject: boolean;
    noteBpmMode: string;
    specifiedBpm: number;
    onMidiClipDialogOpenChange: (open: boolean) => void;
    onMidiClipPathChange: (path: string | null) => void;
    onMidiClipStartSecChange: (sec: number) => void;
    onMidiClipTrackIdChange: (trackId: string | null) => void;
    onFillGapsChange: (v: boolean) => void;
    onMultiTrackMergeChange: (v: boolean) => void;
    onImportBpmAsProjectChange: (v: boolean) => void;
    onNoteBpmModeChange: (v: string) => void;
    onSpecifiedBpmChange: (v: number) => void;
    midiClipClipboardGuid?: string | null;
    importPosition: string;
    onImportPositionChange: (position: string) => void;
    closeLeadingGap: boolean;
    onCloseLeadingGapChange: (v: boolean) => void;
    midiDialogSource: "menu" | "dragDrop";
    onMidiDialogSourceChange: (v: "menu" | "dragDrop") => void;
    importTargetMenu?: string;
    onImportTargetMenuChange?: (v: string) => void;
    importTargetDragDrop?: string;
    onImportTargetDragDropChange?: (v: string) => void;
    importTempoMapEnabled?: boolean;
    onImportTempoMapEnabledChange?: (v: boolean) => void;
    importTempoMapTempo?: boolean;
    onImportTempoMapTempoChange?: (v: boolean) => void;
    importTempoMapTimeSignature?: boolean;
    onImportTempoMapTimeSignatureChange?: (v: boolean) => void;
    importTempoMapKeySignature?: boolean;
    onImportTempoMapKeySignatureChange?: (v: boolean) => void;
}

export const TimelinePanel: React.FC<TimelinePanelProps> = ({
    midiClipDialogOpen,
    midiClipPath,
    midiClipStartSec,
    midiClipTrackId,
    fillGaps,
    multiTrackMerge,
    importBpmAsProject,
    noteBpmMode,
    specifiedBpm,
    onMidiClipDialogOpenChange,
    onMidiClipPathChange,
    onMidiClipStartSecChange,
    onMidiClipTrackIdChange,
    onFillGapsChange,
    onMultiTrackMergeChange,
    onImportBpmAsProjectChange,
    onNoteBpmModeChange,
    onSpecifiedBpmChange,
    midiClipClipboardGuid,
    importPosition,
    onImportPositionChange,
    closeLeadingGap,
    onCloseLeadingGapChange,
    midiDialogSource,
    onMidiDialogSourceChange,
    importTargetMenu,
    onImportTargetMenuChange,
    importTargetDragDrop,
    onImportTargetDragDropChange,
    importTempoMapEnabled,
    onImportTempoMapEnabledChange,
    importTempoMapTempo,
    onImportTempoMapTempoChange,
    importTempoMapTimeSignature,
    onImportTempoMapTimeSignatureChange,
    importTempoMapKeySignature,
    onImportTempoMapKeySignatureChange,
}) => {
    const importTarget = midiDialogSource === "dragDrop" ? importTargetDragDrop : importTargetMenu;
    const onImportTargetChange =
        midiDialogSource === "dragDrop" ? onImportTargetDragDropChange : onImportTargetMenuChange;
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const ignoreGrouping = useAppSelector((state) => state.session.ignoreGrouping);
    const disabledGroupIds = useAppSelector((state) => state.session.disabledGroupIds);
    /**
     * 静音检测预览区段（内核模式用）。
     *
     * 由静音检测对话框实时写入；内核把它画成半透明红色覆盖层。`TimelineSessionSlice`
     * 不含该字段，因此在这里单独选择（与旧实现 `ClipItem` 的取值方式一致）。
     */
    const silencePreviewSegments = useAppSelector((state) => state.session.silencePreviewSegments);
    // 时间轴 scroller 水平滚动条的占用高度（offsetHeight - clientHeight）。
    // 轨道头底部按此留出同高占位（bottomGutterHeightPx），保证轨道头与
    // 时间轴区域的竖直滚动范围严格一致。
    const [horizontalScrollbarGutterPx, setHorizontalScrollbarGutterPx] = React.useState(0);
    // 【已删除：timelineViewportHeightPx】它只为旧实现的 `TimelineSurface` 的
    // `playheadHeightPx` 供数（把播放光标延伸到时间轴可视区底部）。内核路径的
    // 播放头是内核容器（视口元素）的直接子节点、恒 `top-0 bottom-0`，高度天然
    // 等于可视区高度，无需测量——留着这个 state 只会是永不读取的死状态。
    const [quickExportDialog, setQuickExportDialog] = React.useState<{
        open: boolean;
        clipIds: string[];
    }>({ open: false, clipIds: [] });
    const [silenceDialogIds, setSilenceDialogIds] = React.useState<string[] | null>(null);

    const [replaceMidiDialog, setReplaceMidiDialog] = React.useState<{
        open: boolean;
        clipId: string | null;
        midiPath: string | null;
    }>({ open: false, clipId: null, midiPath: null });
    const [timeDisplaySettingsOpen, setTimeDisplaySettingsOpen] = React.useState(false);

    // 文件浏览器拖入 HiFiShifter 工程（hshp/hsp）时的「打开工程 / 导入工程」菜单。
    const [projectActionMenu, setProjectActionMenu] = React.useState<{
        x: number;
        y: number;
        path: string;
    } | null>(null);

    // ── 1. State / refs / viewport / scroll / 坐标转换 ──────
    /**
     * 内核宿主句柄（仅内核模式下非空）。
     *
     * 用途：把「轨道头滚动」这类外部意图转发给内核（`setScrollTop`），以及让
     * 标尺点击 seek / 拖入落点 / 参数编辑器同步等需要「当前视口」的逻辑在两种
     * 模式下共用一份取值。
     *
     * 特殊说明：必须在 `useTimelineState()` **之前**声明——该 hook 需要它来
     * 落地「参数编辑器视图同步」（内核模式没有原生 scroller 可写）。
     */
    const kernelHostRef = React.useRef<TimelineKernelHost | null>(null);

    /**
     * 内核运行期失败的原因（`null` = 未失败）。
     *
     * 【为什么需要】WebGL2 不可用（老驱动 / 远程桌面 / GPU 黑名单 / 上下文数超限）
     * 是预期内的环境差异。内核是**唯一**渲染路径，没有回退可走，因此必须让用户能
     * 自助排障——失败界面（`KernelUnavailableNotice`）要展示具体原因。
     *
     * 【为什么用 state 而不是 ref】它决定渲染内核还是失败界面，必须触发重渲染。
     *
     * 【为什么置位后不再尝试】失败原因（无 GL）在会话内不会自愈，而每次重渲染都重挂
     * 一次内核会反复失败并刷日志。要恢复只需刷新页面。
     */
    const [kernelUnavailableReason, setKernelUnavailableReason] = React.useState<string | null>(
        null,
    );

    /**
     * `TimelineKernelView` 回报内核不可用：记录原因并告警。
     *
     * 流程：记录原因（触发重渲染，切到失败界面）→ 控制台告警（带原因，供诊断）。
     *
     * 特殊说明 1：**没有回退**。旧实现（原生滚动 + Canvas2D）已随阶段 2/3 删除，
     * 因此这里不会、也无法切回任何第二套渲染实现；唯一出路是失败界面 + 重启。
     *
     * 特殊说明 2：用 `useCallback` 保持引用稳定——它经 props 传给内核视图，而视图把
     * 它放进「回调镜像」ref；引用抖动虽不会重建宿主，但稳定引用更省心。
     *
     * @param reason 失败原因（来自内核创建的 catch）。
     */
    const handleKernelUnavailable = React.useCallback((reason: string) => {
        console.warn(`[TimelinePanel] 时间轴内核不可用（${reason}），当前没有回退渲染路径`);
        setKernelUnavailableReason(reason);
    }, []);

    /**
     * 「clip 左键按下拦截」句柄（内核在启动自己的手势之前调用）。
     *
     * 【为什么用 ref 中转】拦截需要复用旧实现的 `useClipPitchDrag`（它在面板里
     * 于渲染流程的**后段**才实例化），而内核交互对象在**前段**构建。用 ref 中转
     * 既避免把 hook 调用提前（改动面大），也不产生 TDZ。
     *
     * @returns true = 面板已接管，内核不得再启动自己的手势。
     */
    const kernelClipInterceptRef = React.useRef<
        | ((args: {
              clipId: string;
              clientX: number;
              clientY: number;
              pointerId: number;
              modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
              container: HTMLElement;
          }) => boolean)
        | null
    >(null);
    const state = useTimelineState({ kernelHostRef });
    // 视觉插值播放头的共享读取点：bridge 的 onFrame 每帧写入（与绘制同源），
    // 缩放锚点与内核宿主（经 `playheadSec` getter）读取同一值——播放中缩放不得
    // 以 33Hz 轮询的 store 滞后值锚定，否则播放头会跳变 δ·Δpx（δ = 轮询间隔内的
    // 插值领先量）。
    const visualPlayheadSecRef = React.useRef(0);
    const getVisualPlayheadSec = React.useCallback(() => visualPlayheadSecRef.current, []);
    /**
     * 请求时间轴内核重绘播放头。
     *
     * 【为什么走 hostRef 而不是 React state】播放头每帧都可能移动，走 React 会
     * 把 60fps 的更新灌进渲染；这里只做一次命令式标脏。
     *
     * 特殊说明：**不是** `invalidateScene()`——那个会置 `sceneDirty` 并在下一帧
     * 重建全部 GPU 几何，播放中每帧重建等于自毁性能（见宿主接口说明）。
     */
    const handleRequestPlayheadRepaint = React.useCallback(() => {
        kernelHostRef.current?.invalidatePlayhead();
    }, []);
    // 面板卸载时清空吸附竖线高亮（拖拽手势异常中断的兜底）。
    React.useEffect(() => {
        return () => {
            clearSnapHighlights();
        };
    }, []);

    // 播放开始（键盘快捷键 / 播放按钮 / 远端传输皆可触发）＝拖拽视觉语境终止：
    // 按住鼠标拖拽期间开始播放时，吸附竖线高亮必须立即消失，而不是冻结在原地
    // 直到松手才被手势结束逻辑清理。
    const timelineRuntimeIsPlaying = useAppSelector((state) => state.session.runtime.isPlaying);
    React.useEffect(() => {
        if (timelineRuntimeIsPlaying) {
            clearSnapHighlights();
        }
    }, [timelineRuntimeIsPlaying]);
    const {
        dispatch,
        s,
        sessionRef,
        scrollRef,
        trackListScrollRef,
        rulerContentRef,
        rulerPlayheadLineRef,
        rulerPlayheadHeadRef,
        dropPreviewRef,
        lastClickedClipIdRef,
        pxPerSecRef,
        viewportWidthRef,
        rowHeightRef,
        pxPerSec,
        setPxPerSec,
        viewportWidth,
        setViewportWidth,
        rowHeight,
        setRowHeight,
        trackVolumeUi,

        setTrackVolumeUi,
        sameSourceConfirmOpen,
        setSameSourceConfirmOpen,
        sameSourceConfirmResolverRef,
        pxPerBeat,
        contentWidth,
        contentHeight,
        dynamicProjectSec,
        timelineTicks,
        rulerScrollLeft,
        verticalZoomKb,
        paramFineAdjustKb,
        slipEditKb,
        crossfadeGripKb,
        fadeCurvatureKb,
        stretchKbRef,
        pitchDragKb,
        noSnapKb,
        copyDragKb,
        dropPreview,
        setDropPreview,
        pendingDropDurationPathRef,
        syncScrollLeft,
        setScrollLeftState,
        beatFromClientX,
        trackIdFromClientY,
        rowTopForTrackId,
        ensureDropPreviewDuration,
        getDropPreviewWidthPx,
        snapTimeline,
        snapTimelineDetailed,
        setPlayheadFromClientX,
        keyboardZoomPendingRef,
    } = state;

    /**
     * 模式无关的视口访问器。
     *
     * 内核模式下旧滚动容器不挂载（`scrollRef.current === null`），凡是「读/写视口」
     * 的逻辑（拖入落点、自动滚屏、键盘缩放、聚焦光标、参数编辑器同步）都必须经它，
     * 而不是各自判断模式。
     *
     * 特殊说明：引用必须**稳定**——下游 hook 把它作为 effect 依赖，随渲染重建会
     * 导致事件监听反复重注册。因此惰性创建一次并缓存。
     */
    const viewportAccessRef = React.useRef<TimelineViewportAccess | null>(null);
    if (viewportAccessRef.current === null) {
        viewportAccessRef.current = createTimelineViewportAccess({ scrollRef, kernelHostRef });
    }
    const viewportAccess = viewportAccessRef.current;

    /**
     * clientY → 轨道 id（模式无关）。
     *
     * 旧实现读原生 scroller 的 `scrollTop`；内核模式下容器不再滚动（真值在宿主里），
     * 必须改用「容器矩形 + 宿主缓存的 scrollTop」。
     *
     * 特殊说明：宿主存在时走内核分支；宿主为 null（挂载前 / 测试环境）时委托给
     * `trackIdFromClientY` 兜底。旧的原生 scroller 实现已随"渲染内核唯一路径"改造删除，
     * `scrollRef` 已无 JSX 挂载点，因此这里**不是**在两种渲染模式之间做选择。
     */
    const resolveTrackIdAtClientY = React.useCallback(
        (clientY: number): string | null => {
            const host = kernelHostRef.current;
            if (host === null) return trackIdFromClientY(clientY);
            const rect = host.getContainerRect();
            if (rect === null) return null;
            const y = clientY - rect.top + host.getViewport().scrollTop;
            const idx = Math.floor(y / rowHeight);
            const tracks = sessionRef.current.tracks;
            if (idx < 0 || idx >= tracks.length) return null;
            return tracks[idx]?.id ?? null;
        },
        [rowHeight, sessionRef, trackIdFromClientY],
    );

    /**
     * 内核水平滚动提交（每跨过 256px 一次）。
     *
     * 走旧实现的同一条通知链（`syncScrollLeft`）：更新 React state（标尺刻度
     * 窗口）**并**把视口写入 `timelineViewportSync`（参数编辑器同步的写侧）。
     * 若直接传 `setScrollLeftState`，同步写侧会被漏掉——表现为「内核滚动时
     * 参数编辑器不跟随」。
     */
    const handleKernelScrollLeftCommit = React.useCallback(
        (next: number) => {
            syncScrollLeft(next);
        },
        [syncScrollLeft],
    );

    // ── 轨道头与时间轴区域的竖直滚动对齐 ─────────────────
    // 右侧时间轴 scroller 常驻水平滚动条（占高 h），其竖直滚动范围因此比
    // 轨道头少 h 像素：内容同为「轨道数 × rowHeight」时，轨道头滚到底会
    // 比时间轴多滚 h 像素，行无法对齐。这里实测 h（offsetHeight 减去
    // clientHeight），平台/样式自适应（overlay 滚动条占位为 0）。
    React.useLayoutEffect(() => {
        const scroller = scrollRef.current;
        if (!scroller) return;
        const measure = () => {
            setHorizontalScrollbarGutterPx(scroller.offsetHeight - scroller.clientHeight);
        };
        measure();
        if (typeof ResizeObserver !== "undefined") {
            const observer = new ResizeObserver(measure);
            observer.observe(scroller);
            return () => observer.disconnect();
        }
        window.addEventListener("resize", measure);
        return () => window.removeEventListener("resize", measure);
    }, [scrollRef]);

    // ── 粘贴后“聚焦播放光标”（提交后执行）────────────────────
    // 粘贴可能大幅扩充工程全长（dynamicProjectSec / 水平可滚动范围随之
    // 扩大）。滚动必须在本状态与对应 DOM（paddedContentWidth）都提交后
    // 执行，否则会被旧的滚动上限钳制，导致光标无法进入画面。
    // pendingPlayheadRevealSec 由粘贴 fulfilled reducer 记录，这里在
    // useLayoutEffect 中消费并立即清除。
    const pendingPlayheadRevealSec = s.pendingPlayheadRevealSec;
    React.useLayoutEffect(() => {
        if (pendingPlayheadRevealSec == null) return;
        // 仅当新光标位置不在可视范围内时才滚动（需求语义：画面内不扰动视图）。
        // 视口来源必须模式无关：内核模式下旧滚动容器不挂载，若在这里早退会
        // **静默丢弃**聚焦请求（表现为「粘贴后视图不跟随」）。
        const x = Math.max(0, pendingPlayheadRevealSec) * pxPerSec;
        const left = viewportAccess.getScrollLeft();
        const right = left + viewportAccess.getViewportWidth();
        if (x >= left && x <= right) {
            dispatch(setPendingPlayheadReveal(null));
            return;
        }
        const next = computeFocusCursorScrollLeft({
            playheadSec: pendingPlayheadRevealSec,
            pxPerSec,
            contentWidth: dynamicProjectSec * pxPerSec,
        });
        if (Math.abs(viewportAccess.getScrollLeft() - next) > 0.5) {
            syncScrollLeft(viewportAccess.setScrollLeft(next));
        }
        dispatch(setPendingPlayheadReveal(null));
    }, [
        pendingPlayheadRevealSec,
        pxPerSec,
        dynamicProjectSec,
        viewportAccess,
        syncScrollLeft,
        dispatch,
    ]);

    const timeContext = React.useMemo<TimeFormatContext>(
        () => ({
            bpm: s.bpm,
            beatsPerBar: Math.max(1, Math.round(s.beats || 4)),
            grid: s.grid,
            tempoMap: s.tempoMap,
        }),
        [s.bpm, s.beats, s.grid, s.tempoMap],
    );

    // 淡化长度 ToolTips 的相对时长上下文：主/副时间单位 + 工程计时参数。
    const fadeLengthFormatCtx = React.useMemo<FadeLengthFormatContext>(
        () => ({
            primaryTimeUnit: s.primaryTimeUnit,
            secondaryTimeUnit: s.secondaryTimeUnit,
            bpm: s.bpm,
            beatsPerBar: Math.max(1, Math.round(s.beats || 4)),
            grid: s.grid,
        }),
        [s.primaryTimeUnit, s.secondaryTimeUnit, s.bpm, s.beats, s.grid],
    );

    const projectScale = React.useMemo<ScaleLike | null>(
        () =>
            s.project.useCustomScale && s.project.customScale
                ? s.project.customScale.notes
                : s.project.baseScale,
        [s.project.baseScale, s.project.customScale, s.project.useCustomScale],
    );

    const handleTempoMapChange = React.useCallback(
        (next: TempoMap | null) => {
            dispatch(setTempoMap(next));
        },
        [dispatch],
    );
    const handleTempoMapCommit = React.useCallback(
        (next: TempoMap | null) => {
            dispatch(setTempoMap(next));
            void dispatch(setTempoMapRemote(next));
        },
        [dispatch],
    );
    const handlePrimaryUnitChange = React.useCallback(
        (unit: TimeUnit) => {
            dispatch(setPrimaryTimeUnit(unit));
            void dispatch(persistUiSettings());
        },
        [dispatch],
    );
    const handleSecondaryUnitChange = React.useCallback(
        (unit: TimeUnitChoice) => {
            dispatch(setSecondaryTimeUnit(unit));
            void dispatch(persistUiSettings());
        },
        [dispatch],
    );
    const handleCopyPlayheadTime = React.useCallback(async () => {
        const text = formatCursorTime(
            s.primaryTimeUnit,
            s.secondaryTimeUnit,
            Number(sessionRef.current.playheadSec ?? 0) || 0,
            timeContext,
        ).combined;
        try {
            await navigator.clipboard.writeText(text);
        } catch {
            try {
                const textarea = document.createElement("textarea");
                textarea.value = text;
                textarea.style.position = "fixed";
                textarea.style.opacity = "0";
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand("copy");
                textarea.remove();
            } catch {
                // 忽略复制失败
            }
        }
    }, [s.primaryTimeUnit, s.secondaryTimeUnit, sessionRef, timeContext]);

    // ── 记录最近点击的 clientX，用于 Shift 范围选择的锚点位置
    const lastClickedClientXRef = React.useRef<number | null>(null);

    // ── 时间轴缩放与行高的持久化 ─────────────────────────────
    //
    // 旧实现由 `TimelineScrollArea`（原生滚动容器）负责写这两个键；"内核唯一路径"
    // 改造删掉该组件时**没有把写入点接回来**，于是读取侧（`useTimelineState` 初始化
    // 读 `hifishifter.pxPerSec` / `hifishifter.rowHeight`）一直在，但再没有人写
    // ——表现为每次重启都回到默认缩放与行高。这里接回同一对键名与取值语义。
    //
    // 用防抖 hook 而不是直接 `localStorage.setItem`：滚轮缩放是高频事件，同步落盘
    // 会与同帧的渲染 / 重绘挤在一起（见 `useDebouncedPersist` 文件头）。
    useDebouncedPersist("hifishifter.pxPerSec", pxPerSec);
    useDebouncedPersist("hifishifter.rowHeight", rowHeight);

    // ── 2. Clip 多选 + 操作回调 ─────────────────────────────
    const clipActions = useTimelineClipActions({
        sessionRef,
        scrollRef,
        // 模式无关视口：范围选择的「点击位置 → 工程秒」换算需要它（内核模式下
        // 原生 scroller 不存在，缺了它会退化成用 clip 起点近似）。
        viewport: viewportAccess,
        lastClickedClipIdRef,
        lastClickedClientXRef,
        pxPerSec,
        pxPerBeat,
        rowHeight,
        ignoreGrouping,
        disabledGroupIds,
        dispatch,
        sameSourceConfirmResolverRef,
        setSameSourceConfirmOpen,
        setPlayheadFromClientX,
    });
    const {
        multiSelectedClipIds,
        multiSelectedSet,
        setMultiSelectedClipIds,
        contextMenu,
        setContextMenu,
        trackAreaMenu,
        setTrackAreaMenu,
        importModeMenu,
        setImportModeMenu,
        clipboardAvailable,
        copyClips,
        cutClips,
        groupClips,
        ungroupClips,
        toggleGroupDisabled,
        normalizeClips,
        replaceClipSources,
        splitClipIdsAtPlayhead,
        splitSelectedAtPlayhead,
        selectClipRangeByRect,
        recordLastClickPosition,
        pasteClipsAtPlayhead,
        clearContextMenu,
        ensureTrackLaneSelected,
        selectTrackLaneClipRemote,
        deselectAllTrackLaneClips,
        toggleTrackLaneClipMuted,
        toggleTrackLaneCtrlSelection,
        commitTrackLaneRename,
        commitTrackLaneGain,
        commitTrackLaneRate,
    } = clipActions;
    // 右键播放速率角标 → 高级编辑浮层（BPM 换算）的目标 Clip 与锚点。
    const [rateEditorClipId, setRateEditorClipId] = React.useState<string | null>(null);
    const [rateEditorPosition, setRateEditorPosition] = React.useState<{
        x: number;
        y: number;
    } | null>(null);
    const openRateBadgeMenu = React.useCallback(
        (clipId: string, screenX: number, screenY: number) => {
            setRateEditorClipId(clipId);
            setRateEditorPosition({ x: screenX, y: screenY });
        },
        [],
    );

    // 角标行内编辑提交：按字段路由到速率（自动调整时长）/增益提交。
    // 传给 React.memo 化的 TrackList / TrackLane 的回调必须引用稳定，
    // 否则每次 TimelinePanel 渲染（播放头提交值、滚动、修饰键）都会击穿 memo。
    const handleToggleGroupDisabled = React.useCallback(
        (groupId: string) => {
            toggleGroupDisabled(groupId);
        },
        [toggleGroupDisabled],
    );
    const commitTrackLaneFormantMorph = React.useCallback(
        (clipId: string, value: ClipFormantMorph, checkpoint: boolean) => {
            void dispatch(
                setClipStateRemote({
                    clipId,
                    formantMorph: value,
                    checkpoint,
                }),
            );
        },
        [dispatch],
    );
    const activateTrackLaneTake = React.useCallback(
        (clipId: string, takeId: string) => {
            void dispatch(setClipActiveTakeRemote({ clipId, takeId }));
        },
        [dispatch],
    );
    /**
     * 单侧循环到下一个形状并重置默认曲率。
     *
     * 多选：重点 clip 属于选区时应用到全部选中（与淡变菜单同一判定），
     * 每个 clip 从**自身**当前形状循环前进；一次 bulk 提交 = 单撤销步。
     *
     * @param checkpoint 单独调用默认 true（一笔后端写入 = 一个撤销步）；
     *                   交叉点双列循环时传 false 并由调用方开 undo group
     *                   合并为单步。
     */
    const cycleOneFade = React.useCallback(
        (clipId: string, side: "in" | "out", checkpoint = true) => {
            const targets = getBulkEditableClipIds({
                activeClipId: clipId,
                multiSelectedClipIds,
                multiSelectedSet,
            });
            const updates: Array<{
                clipId: string;
                fadeInShape?: number;
                fadeInDir?: number;
                fadeOutShape?: number;
                fadeOutDir?: number;
            }> = [];
            for (const targetId of targets) {
                const clip = sessionRef.current.clips.find((entry) => entry.id === targetId);
                if (!clip) continue;
                const rawShape = side === "in" ? clip.fadeInShape : clip.fadeOutShape;
                const currentShape = Number.isFinite(rawShape) ? Math.trunc(rawShape) : 0;
                const index = FADE_PRESETS.findIndex((preset) => preset.shape === currentShape);
                const nextPreset =
                    FADE_PRESETS[(index + 1 + FADE_PRESETS.length) % FADE_PRESETS.length];
                const nextDir = defaultFadeDirFor(nextPreset.shape, side === "out");
                dispatch(
                    setClipFades({
                        clipId: targetId,
                        ...(side === "in"
                            ? { fadeInShape: nextPreset.shape, fadeInDir: nextDir }
                            : { fadeOutShape: nextPreset.shape, fadeOutDir: nextDir }),
                    }),
                );
                updates.push({
                    clipId: targetId,
                    ...(side === "in"
                        ? { fadeInShape: nextPreset.shape, fadeInDir: nextDir }
                        : { fadeOutShape: nextPreset.shape, fadeOutDir: nextDir }),
                });
            }
            if (updates.length > 0) {
                void dispatch(setClipsStateBulkRemote({ updates, checkpoint }));
            }
        },
        [dispatch, sessionRef, multiSelectedClipIds, multiSelectedSet],
    );

    // Ctrl+点击循环切换：普通包络线只切该线；交叉点抓手同时切换两侧
    // （前者淡出 + 后者淡入）。
    const handleFadeShapeCycleClick = React.useCallback(
        (clipId: string, side: "in" | "out") => {
            cycleOneFade(clipId, side);
        },
        [cycleOneFade],
    );
    const handleCrossfadeCycleClick = React.useCallback(
        (sides: Array<{ clipId: string; isOut: boolean }>) => {
            // 交叉点双列循环 = 一次手势：开 undo group 把两侧循环合并为
            // 单个撤销步（否则两笔 checkpoint:true 会变成两个撤销步）。
            void (async () => {
                await webApi.beginUndoGroup("edit_clip");
                try {
                    for (const side of sides) {
                        cycleOneFade(side.clipId, side.isOut ? "out" : "in", false);
                    }
                } finally {
                    await webApi.endUndoGroup();
                }
            })().catch(() => undefined);
        },
        [cycleOneFade],
    );
    const activeFormantToolClip = React.useMemo(
        () =>
            s.clipFormantToolWindow.clipId
                ? (s.clips.find((clip) => clip.id === s.clipFormantToolWindow.clipId) ?? null)
                : null,
        [s.clipFormantToolWindow.clipId, s.clips],
    );

    // ── MIDI clip drag-drop handler ──────────────────────
    const handleMidiClipImport = React.useCallback(
        (result: {
            trackIndices: number[];
            notesCount: number;
            midiPath: string;
            fillGaps: boolean;
            multiTrackMerge?: boolean;
            noteBpmMode?: string;
            specifiedBpm?: number;
            importBpmAsProject?: boolean;
            clipboardGuid?: string;
            closeLeadingGap?: boolean;
            importAsTempoMap?: boolean;
            importTempo?: boolean;
            importTimeSignature?: boolean;
            importKeySignature?: boolean;
        }) => {
            void dispatch(
                importMidiAsClip({
                    midiPath: result.midiPath,
                    trackIndices: result.trackIndices,
                    trackId: midiClipTrackId,
                    startSec: midiClipStartSec,
                    fillGaps: result.fillGaps || undefined,
                    multiTrackMerge: result.multiTrackMerge,
                    noteBpmMode: result.noteBpmMode,
                    specifiedBpm: result.specifiedBpm,
                    importBpmAsProject: result.importBpmAsProject,
                    clipboardGuid: result.clipboardGuid,
                    closeLeadingGap: result.closeLeadingGap,
                    importAsTempoMap: result.importAsTempoMap,
                    importTempo: result.importTempo,
                    importTimeSignature: result.importTimeSignature,
                    importKeySignature: result.importKeySignature,
                }),
            );
        },
        [dispatch, midiClipTrackId, midiClipStartSec],
    );

    // ── Replace MIDI ──
    const handleReplaceMidiImport = React.useCallback(
        (result: {
            trackIndices: number[];
            notesCount: number;
            midiPath: string;
            fillGaps: boolean;
            multiTrackMerge?: boolean;
            noteBpmMode?: string;
            specifiedBpm?: number;
            importBpmAsProject?: boolean;
            closeLeadingGap?: boolean;
        }) => {
            const clipId = replaceMidiDialog.clipId;
            if (!clipId) return;
            void dispatch(
                replaceMidiClipDataRemote({
                    clipId,
                    midiPath: result.midiPath,
                    trackIndices: result.trackIndices,
                    fillGaps: result.fillGaps || undefined,
                    noteBpmMode: result.noteBpmMode,
                    specifiedBpm: result.specifiedBpm,
                    importMidiBpmAsProject: result.importBpmAsProject,
                    closeLeadingGap: result.closeLeadingGap,
                }),
            );
            setReplaceMidiDialog({ open: false, clipId: null, midiPath: null });
        },
        [dispatch, replaceMidiDialog.clipId],
    );

    const openReplaceMidiForClip = React.useCallback(async (clipId: string) => {
        const picked = await webApi.openMidiDialog();
        if (!picked.ok || picked.canceled || !picked.path) return;
        setReplaceMidiDialog({ open: true, clipId, midiPath: picked.path });
    }, []);

    const midiClipRootTrackComposeEnabled = React.useMemo(() => {
        if (!midiClipTrackId) return true;
        const rootId = resolveRootTrackId(s.tracks, midiClipTrackId);
        if (!rootId) return true;
        const rootTrack = s.tracks.find((t) => t.id === rootId);
        return rootTrack?.composeEnabled ?? true;
    }, [midiClipTrackId, s.tracks]);

    const handleRequestEnableCompose = React.useCallback(() => {
        if (!midiClipTrackId) return;
        const rootId = resolveRootTrackId(s.tracks, midiClipTrackId);
        if (!rootId) return;
        dispatch(
            setTrackStateRemote({
                trackId: rootId,
                composeEnabled: true,
            }),
        );
    }, [dispatch, midiClipTrackId, s.tracks]);

    const handleExportMidi = React.useCallback(
        async (clipIds: string[]) => {
            const saveResult = await coreApi.pickMidiOutputPath();
            if (!saveResult.ok || saveResult.canceled || !saveResult.path) return;

            const s = sessionRef.current;
            const clipsMap = new Map(s.clips.map((c) => [c.id, c]));
            const trackMap = new Map(s.tracks.map((t) => [t.id, t]));

            const entries: Array<{
                trackId: string;
                rootTrackId: string;
                name: string;
                startSec: number;
                endSec: number;
                clipId?: string;
            }> = [];
            const seenComposeRoots = new Set<string>();

            for (const id of clipIds) {
                const clip = clipsMap.get(id);
                if (!clip) continue;
                const rootId = resolveRootTrackId(s.tracks, clip.trackId);
                if (!rootId) continue;
                const rootTrack = trackMap.get(rootId);
                const isComposeEnabled = rootTrack?.composeEnabled ?? false;

                if (isComposeEnabled) {
                    // Compose 轨道：按 rootTrackId 去重（共享 track 级音高数据）
                    if (seenComposeRoots.has(rootId)) continue;
                    seenComposeRoots.add(rootId);
                }

                const track = trackMap.get(clip.trackId);
                entries.push({
                    trackId: clip.trackId,
                    rootTrackId: rootId,
                    name: track?.name ?? clip.name,
                    startSec: clip.startSec,
                    endSec: clip.startSec + clip.lengthSec,
                    ...(isComposeEnabled ? {} : { clipId: clip.id }),
                });
            }

            if (entries.length === 0) return;

            const scaleNotes =
                SCALE_NOTES[(s.project?.baseScale as keyof typeof SCALE_NOTES) ?? "C"] ??
                SCALE_NOTES.C;

            await paramsApi.exportPitchToMidi({
                outputPath: saveResult.path,
                tracks: entries,
                bpm: s.bpm,
                beatsPerBar: s.project?.beatsPerBar ?? 4,
                baseScale: s.project?.baseScale ?? "C",
                projectScaleNotes: scaleNotes,
            });
        },
        [sessionRef],
    );

    // ── 3. DragDrop (Tauri + 文件浏览器) ─────────────────────
    const { tauriDraggedPathRef, tauriLastDropPathRef, tauriDropHandledAtRef } =
        useTimelineDragDrop({
            dispatch,
            viewport: viewportAccess,
            sessionRef,
            pxPerSecRef,
            rowHeightRef,
            dropPreviewRef,
            pendingDropDurationPathRef,
            beatFromClientX,
            snapTimeline,
            trackIdFromClientY: resolveTrackIdAtClientY,
            rowTopForTrackId,
            setDropPreview,
            ensureDropPreviewDuration,
            getDropPreviewWidthPx,
            setImportModeMenu,
            setProjectActionMenu,
            pxPerSec,
            rowHeight,
            onMidiDrop: (payload) => {
                onMidiDialogSourceChange("dragDrop");
                onMidiClipPathChange(payload.midiPath);
                onMidiClipStartSecChange(payload.startSec);
                onMidiClipTrackIdChange(payload.trackId);
                onMidiClipDialogOpenChange(true);
            },
        });

    /**
     * 拖入几何用的水平滚动量（模式无关）。
     *
     * 内核模式下时间轴是**自绘滚动**，DOM 容器的 scrollLeft 恒为 0——真值在内核
     * 宿主里。旧模式仍取容器自身的 scrollLeft。
     *
     * @param el 事件目标容器（旧模式下即滚动容器）。
     * @returns 当前水平滚动量（CSS px）。
     */
    function dragScrollLeftOf(el: HTMLElement): number {
        const host = kernelHostRef.current;
        if (host !== null) return host.getViewport().scrollLeft;
        return el.scrollLeft;
    }

    /** 内核 seek 的待提交位置与 rAF 句柄（拖拽期间按帧节流，松手立即提交）。 */
    const kernelSeekPendingRef = React.useRef<number | null>(null);
    const kernelSeekRafRef = React.useRef<number | null>(null);

    /**
     * 内核 seek 回调：单击或拖拽空白处跳转播放头。
     *
     * 拖拽帧以 rAF 节流提交（内核按 rAF 频率回调），避免每个指针事件都打一次
     * 后端 seek；松手（commit）时取消待提交帧并立即提交最终位置，保证落点精确。
     *
     * 特殊说明（**必须与 seek 一起乐观写 playheadSec**）：`seekPlayhead.fulfilled`
     * 只在「后端返回值与请求值不同」（如被 clamp 修正）时才采纳后端值，其余情况
     * 依赖调用方**已经**写好 `state.playheadSec`；真实后端 `set_transport` 对
     * 非负请求原样回显（`playhead_sec = v.max(0.0)`），于是后端值恒等于请求值、
     * 采纳分支永不命中——只派发 `seekPlayhead` 时引擎动了而 store 的
     * `playheadSec` 纹丝不动（"点空白处播放头不跟随"的根因；播放中 30Hz 轮询
     * 重新锚定才让它"偶尔"生效）。此处与标尺路径
     * （`useTimelineState.setPlayheadFromClientX`：`setplayheadSec` + `seekPlayhead`
     * 成对派发）保持同一契约，两个分支都不得只派发 seek。
     */
    /**
     * 把请求的秒数按旧实现的「光标吸附」规则规范化（**不改状态**）。
     *
     * 所有播放头落点路径（标尺 / 轨道空白 / clip 点击）在旧实现里都汇入
     * `setPlayheadFromClientX`：先 `snapTimelineDetailed(…, "cursor", …)` 吸附，
     * 再写 `setplayheadSec` + `seekPlayhead`；`commit=true` 表示手势语境结束，
     * 顺带清除瞬态吸附高亮。内核原先直接写**裸**秒数——标尺路径吸附、轨道空白
     * 路径不吸附，同一个界面里两条 seek 行为不一致。
     *
     * @param sec 目标秒（未吸附）。
     * @param commit true = 提交式落点（单击 / 拖拽松手），false = 拖拽中间帧。
     * @returns 吸附后的秒。
     */
    const resolveKernelSeekSec = React.useCallback(
        (sec: number, commit: boolean): number =>
            snapTimelineDetailed(
                sec,
                "cursor",
                // 高亮只在拖拽期间发布（单击跳转完全不走高亮通道），与旧实现一致。
                commit ? undefined : { highlight: { sources: [] } },
            ).sec,
        [snapTimelineDetailed],
    );

    /**
     * 纯粹的播放头落点（吸附 + 乐观写 + 高亮收口）——**不含**空白点击的
     * 「清空选中 / 切换轨道」语义。
     *
     * 旧实现的 clip 单击 seek（`ClipItem` / `ClipEdgeHandles` / `FadeHitLayer`
     * 的 `seekFromClientX`）只移动播放头；把"清空选中"混进来会让「点一下 clip
     * 把播放头带过去」顺带取消选中，与旧实现完全相反。
     *
     * @param sec 目标秒。
     * @returns 无返回值。
     */
    const handleKernelSeekTo = React.useCallback(
        (sec: number): void => {
            const target = resolveKernelSeekSec(sec, true);
            clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            dispatch(setplayheadSec(target));
            void dispatch(seekPlayhead(target));
        },
        [dispatch, resolveKernelSeekSec],
    );

    const handleKernelSeek = React.useCallback(
        (sec: number, commit: boolean, trackId?: string | null) => {
            const target = resolveKernelSeekSec(sec, commit);
            if (commit) {
                // 空白点击的选中语义（与旧实现 pointerdown 捕获分支同源）：
                // 1) 清空 clip 选中——但**保留轨道焦点**（空白点击是"取消 clip
                //    目标"，不是"切换轨道目标"）；
                // 2) 按「允许时间轴点击切换轨道」把当前轨道切到点击所在轨道。
                deselectAllTrackLaneClips();
                if (
                    trackId != null &&
                    sessionRef.current.paramEditorTimelineClickSelectTrackEnabled &&
                    trackId !== sessionRef.current.selectedTrackId
                ) {
                    // `applySelectedClip: false` 是**契约**，不是可选优化：
                    //
                    // 1) 上面的 `deselectAllTrackLaneClips()` 是**纯本地** reducer
                    //    （`setSelectedClipPreservingTrack(null)`），从不通知后端，
                    //    因此后端一直记着旧的 `selected_clip_id`；
                    // 2) `selectTrackRemote.fulfilled` 默认拿后端快照覆盖前端选中
                    //    （`sessionSlice.ts` 的 `applySelectedClip` 闸门）；
                    // 3) 纯字符串让 `typeof arg !== "object"` 判为"要恢复"，于是刚
                    //    清掉的选中被**异步复活** —— 表现为"点空白切轨时选中没被
                    //    取消"。复活后的 `selectedClipId` 可能**不属于**当前轨道
                    //    （`select_track` 刻意不清它，见 `state.rs`），于是它成为一个
                    //    跨轨的陈旧目标，被按 `selectedClipId` 取用命令的路径直接
                    //    读到（例如 `removeSelectedClipRemote`）。
                    //
                    // 该契约由 `019e93ed`（"blank-click deselect survives track
                    // switching"）建立，当时两处 DOM 调用点都传了
                    // `applySelectedClip: false`；内核补全提交 `464a78bb` 在新调用点
                    // 写成纯字符串，**静默解除**了该修复。修改此处前请先读这两个提交，
                    // 不要把对象形式"简化"回纯字符串。
                    //
                    // 特殊说明（残留风险，本闸门覆盖不到）：取消选中仍是纯本地的，
                    // 因此任何返回**完整快照**的 `*Remote.fulfilled` 仍可能经
                    // `applyTimelineState` 无条件写回 `selected_clip_id` 而复活它。
                    // 那条通道是既有行为、与本闸门无关；要彻底修需另开一轮（让
                    // `applyTimelineState` 也尊重"本地已清空"的意图）。
                    void dispatch(selectTrackRemote({ trackId, applySelectedClip: false }));
                }
            }
            kernelSeekPendingRef.current = target;
            if (commit) {
                if (kernelSeekRafRef.current != null) {
                    cancelAnimationFrame(kernelSeekRafRef.current);
                    kernelSeekRafRef.current = null;
                }
                kernelSeekPendingRef.current = null;
                clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                // 乐观写 store：见函数头注释（缺了它 playheadSec 不会动）。
                dispatch(setplayheadSec(target));
                void dispatch(seekPlayhead(target));
                return;
            }
            if (kernelSeekRafRef.current != null) return;
            kernelSeekRafRef.current = requestAnimationFrame(() => {
                kernelSeekRafRef.current = null;
                const pending = kernelSeekPendingRef.current;
                if (pending == null) return;
                kernelSeekPendingRef.current = null;
                // 拖拽中间帧**只写乐观位置，不打后端**：旧实现
                // `startDeferredPlayheadSeek` 的移动分支走 `commit = false`，整段拖拽
                // 只在松手时发一次 `seekPlayhead`（内核原先逐帧成对派发，等于按住拖动
                // 时以 rAF 频率持续刷后端）。松手的收尾由内核补发 `commit = true`。
                dispatch(setplayheadSec(pending));
            });
        },
        [deselectAllTrackLaneClips, dispatch, resolveKernelSeekSec, sessionRef],
    );

    /**
     * 内核选中回调：点击 clip 选中。
     *
     * 多选修饰键（Ctrl / ⌘）切换集合成员；否则单选并同步焦点 clip
     * （焦点 clip 驱动参数编辑器的编辑目标，必须与多选集合一起更新）。
     *
     * 特殊说明：两个分支都**复用旧实现的选中入口**而不是自己拼状态——
     * `selectTrackLaneClipRemote` / `toggleTrackLaneCtrlSelection` 还负责
     * 「点击 clip 切轨」（手册：「点击时间轴中的音频块或空白区域会自动切换当前
     * 轨道」，受 `允许时间轴点击切换轨道` 控制）与后端 `selected_clip` 落库。
     * 内核只识别手势，这些语义不在内核侧重写。
     */
    const handleKernelSelectClip = React.useCallback(
        (clipId: string, additive: boolean, rangeSelect?: boolean, clientX?: number) => {
            // 范围选择（默认 Shift）：复用旧实现的范围选择语义——它自带锚点状态
            // 与「按点击位置取时间范围」的规则，重写一份必然分叉。
            if (rangeSelect === true) {
                selectClipRangeByRect(clipId, undefined, clientX);
                return;
            }
            if (additive) {
                toggleTrackLaneCtrlSelection(clipId);
                return;
            }
            // 普通单击同样要维护范围选择的锚点与点击位置：下一次 Shift 点击要靠它们
            // 决定范围起点。不记录时「先点 A、再 Shift 点 B」会退化成只选中 B
            // （旧实现在 `TrackLane` 的点击收尾里做同一件事）。
            ensureTrackLaneSelected(clipId);
            recordLastClickPosition(clientX ?? 0);
            selectTrackLaneClipRemote(clipId);
        },
        [
            ensureTrackLaneSelected,
            recordLastClickPosition,
            selectClipRangeByRect,
            selectTrackLaneClipRemote,
            toggleTrackLaneCtrlSelection,
        ],
    );

    /** 内核拖拽：按下时的原始位置（把相对位移换算为绝对位置，并支持回滚）。 */
    /**
     * copy 拖拽的 ghost 预览（内容坐标）。
     *
     * 移动语义下内核靠"乐观位置"显示，copy 语义下原 clip 不动，必须有这一层——
     * 否则 ⌘+拖拽期间画面毫无反馈。纵向位置由内核视图按 `rowHeight` 换算
     * （面板没有行高），这里只给内容坐标的左缘与宽度。
     */
    const [kernelGhost, setKernelGhost] = React.useState<Array<{
        key: string;
        leftPx: number;
        widthPx: number;
        trackId: string;
    }> | null>(null);

    /**
     * 内核拖拽是否正落在「新建轨道」哨兵上（拖到全部轨道之下）。
     *
     * React state 驱动幽灵行的渲染，ref 做去重（预览每帧都会给出同一个判定，
     * 逐帧 setState 会白白重渲染整个面板）。
     */
    const [kernelDropToNewTrack, setKernelDropToNewTrack] = React.useState(false);
    const kernelDropToNewTrackRef = React.useRef(false);

    /**
     * 内核手势的**交互锁**（`beginInteraction` / `endInteraction` 的成对封装）。
     *
     * 【为什么每个手势都必须持有】`sessionSlice` 在 `_interactionLockCount > 0`
     * 时丢弃后端回包的过期快照。手势期间若有一条 `*Remote.fulfilled` 抵达（例如
     * 上一次点击的 `selectClipRemote`、或后端的周期性状态推送），没有这把锁它的
     * 旧快照会**覆盖掉正在预览的乐观值**——表现为拖拽中途画面突然弹回原位。
     * 旧实现的每个连续手势都成对调用（`useClipDrag` / `useEditDrag` /
     * `useSlipDrag` / `useSnapOffsetDrag` 各一处），内核路径此前只有 snap offset 接了。
     *
     * 【为什么用 ref 去重】预览回调逐帧触发，而 begin / end 每只手势只能各一次。
     *
     * 【释放时机】与旧实现一致：**落库请求完成后**才释放。若在提交瞬间就释放，
     * `endInteraction()` 到 `fulfilled` 之间会留出一个窗口，窗口内其他 in-flight
     * thunk 的旧快照仍能把乐观值打回去（旧实现 `useClipDrag` 对此有逐字注释）。
     * 纯取消 / 零位移的路径没有任何远程写入，直接释放。
     */
    const kernelGestureInteractionActiveRef = React.useRef(false);
    const beginKernelGestureInteraction = React.useCallback((): void => {
        if (kernelGestureInteractionActiveRef.current) return;
        kernelGestureInteractionActiveRef.current = true;
        dispatch(beginInteraction());
    }, [dispatch]);
    const endKernelGestureInteraction = React.useCallback((): void => {
        if (!kernelGestureInteractionActiveRef.current) return;
        kernelGestureInteractionActiveRef.current = false;
        dispatch(endInteraction());
    }, [dispatch]);

    const kernelDragOriginRef = React.useRef<{
        clipId: string;
        startSec: number;
        lengthSec: number;
        trackId: string;
        /**
         * clip 自身的吸附偏移点（秒）。
         *
         * 多源吸附把「起点 / 终点 / 自身吸附偏移点」同时作为被吸附对象（旧实现
         * `useClipDrag` 同源）。必须随 origin 一起记下：预览期间 Redux 里的
         * `startSec` 已被改写，但 `snapOffsetSec` 是 clip 的固有属性，拖拽中不变。
         */
        snapOffsetSec: number;
        /**
         * 本次拖拽是否已进入 copy 模式。
         *
         * **单向**：false → true 允许（拖拽中途按下复制键），true → false 不允许
         * （与旧实现 `resolveClipDragCopyMode` 的既有语义一致——中途松开复制键
         * 不会把已经"复制"的意图退回成"移动"，避免松手瞬间语义反转）。
         */
        copyMode: boolean;
        /**
         * 本次拖拽实际作用于的全部 clip（含锚点）。
         *
         * 由「多选集合 + 编组展开」在按下时解析一次（见 `kernelEditSet`）：
         * 内核只回调锚点位移，不展开就会出现「选中多个只动一个」「同组不联动」。
         */
        participants: KernelEditParticipant[];
        /** 锚点初始轨道序号：把内核给的 `targetTrackId` 换成轨道偏移量。 */
        anchorTrackIndex: number;
        /** 最近一次预览生效的**共享位移**（已含吸附与左边界钳制）——提交时用它。 */
        lastDeltaStartSec: number;
        /** 波纹跟随集（乐观预览；提交后的权威结果由后端计算）。 */
        rippleFollowers: RippleFollowerMap;
        /** 自动交叉淡化：受本次编辑影响的 clip（参与者 + 波纹跟随）。 */
        editedXfadeClipIds: string[];
        /** 自动交叉淡化：编辑前每侧的既有重叠关系（拖开时只清自动、保留手动 fade）。 */
        initialCrossfadeSides: ReturnType<typeof computeInitialCrossfadeSides>;
        /**
         * 本次拖拽是否为 **slip**（`Alt` + 拖 clip 中部 = 调整内部偏移）。
         *
         * 语义与移动 / 复制完全不同：时间轴位置与长度都不变，只平移源窗口。
         * 在**按下时**定死（与旧实现 `startSlipDrag` 的独立手势一致）。
         */
        slipMode: boolean;
        /** slip：已应用的累计位移（换算增量用，避免逐帧叠加导致越拖越快）。 */
        appliedSlipSec: number;
        /** slip：每个参与者按下时的源窗口（取消时回滚）。 */
        baseSourceById: Map<string, { sourceStartSec: number; sourceEndSec: number }>;
        /**
         * slip：锚 clip 按下时的「媒体边界吸附」视图。
         *
         * 候选族只依赖初始几何（平移不变），因此按下时快照一次即可（与旧实现
         * `useSlipDrag` 的 `anchorSnapshot` 同源）；内容时长 D 的解析规则由
         * `toBoundarySnapClip` 与旧实现共用。
         */
        slipAnchor: ReturnType<typeof toBoundarySnapClip>;
        /**
         * slip：最近一次分发的源窗口。
         *
         * 提交用它而**不回读 Redux**——与旧实现 `lastById` 同源（防并发更新 /
         * 历史归一化把交互数学结果污染）。
         */
        lastSourceById: Map<string, { sourceStartSec: number; sourceEndSec: number }>;
    } | null>(null);

    /**
     * 内核拖拽预览：写乐观位置（拖拽期间每帧，内核已按值去重）。
     *
     * 特殊说明：内核回调的是**相对按下位置的位移**，而 Redux 里该 clip 的位置
     * 在上一次预览时已被改写——必须用按下时记下的原始位置换算，否则位移会逐帧叠加
     * （表现为 clip 越拖越快）。
     */
    const handleKernelDragPreview = React.useCallback(
        (args: {
            clipId: string;
            deltaSec: number;
            targetTrackId: string;
            modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
        }) => {
            if (kernelDragOriginRef.current?.clipId !== args.clipId) {
                const session = sessionRef.current;
                const clip = session.clips.find((item) => item.id === args.clipId);
                if (clip === undefined) return;
                const trackIds = session.tracks.map((track) => track.id);
                // 参与集合：多选集合 + 编组展开（与旧实现 `useClipDrag` 同源）。
                // 不展开时「选中多个只移动一个」「同组不联动」——属数据语义错误。
                const participants = resolveKernelEditParticipants({
                    anchorClipId: clip.id,
                    multiSelectedClipIds,
                    clips: session.clips,
                    trackIds,
                    ignoreGrouping: session.ignoreGrouping,
                    disabledGroupIds: session.disabledGroupIds,
                    expandGroups: true,
                });
                if (participants.length === 0) return;
                const rippleFollowers = buildRippleFollowers(
                    session.clips,
                    new Set(participants.map((item) => item.clipId)),
                    Math.min(...participants.map((item) => item.startSec)),
                    session.rippleMode,
                    new Set(participants.map((item) => item.trackId)),
                );
                const editedXfadeClipIds = Array.from(
                    new Set<string>([
                        ...participants.map((item) => item.clipId),
                        ...Object.keys(rippleFollowers),
                    ]),
                );
                kernelDragOriginRef.current = {
                    clipId: clip.id,
                    startSec: clip.startSec,
                    lengthSec: clip.lengthSec,
                    trackId: clip.trackId,
                    snapOffsetSec: Math.max(0, Number(clip.snapOffsetSec) || 0),
                    copyMode: false,
                    slipMode: isModifierActive(slipEditKb, args.modifiers),
                    slipAnchor: toBoundarySnapClip(clip),
                    appliedSlipSec: 0,
                    baseSourceById: new Map(
                        participants.map((participant) => {
                            const item = session.clips.find(
                                (candidate) => candidate.id === participant.clipId,
                            );
                            return [
                                participant.clipId,
                                {
                                    sourceStartSec: Number(item?.sourceStartSec ?? 0) || 0,
                                    sourceEndSec: Number(item?.sourceEndSec ?? 0) || 0,
                                },
                            ] as const;
                        }),
                    ),
                    lastSourceById: new Map(),
                    participants,
                    anchorTrackIndex: trackIds.indexOf(clip.trackId),
                    lastDeltaStartSec: 0,
                    rippleFollowers,
                    editedXfadeClipIds,
                    initialCrossfadeSides: computeInitialCrossfadeSides(
                        session.clips,
                        editedXfadeClipIds,
                    ),
                };
                // 首个真实位移帧（内核只在越过拖拽阈值后才回调预览）= 手势真正开始：
                // 上交互锁，让手势期间抵达的后端旧快照不再覆盖乐观值。
                beginKernelGestureInteraction();
            }
            const origin = kernelDragOriginRef.current;
            if (origin === null) return;
            // copy 模式判定：复用旧实现的函数（含"已配置绑定为准 + 非 macOS 的 Ctrl
            // 回退"）。**单向**——一旦进入 copy 就不再退回移动，避免松手瞬间语义反转。
            if (origin.slipMode) {
                // Alt + 拖 clip 中部 = **slip**（调整内部偏移）：时间轴位置与长度
                // 都不变，只平移源窗口。逐帧按**增量**应用（用当前 Redux 值 + 增量
                // 换算），避免累计位移被重复施加导致越拖越快。
                //
                // ── 循环节 / 内容边界吸附（拖拽全程生效）──────────────────
                // 属于常规吸附体系：受吸附总开关与「拖动时切换吸附」修饰键（XOR）
                // 控制，且需在吸附设置中启用「Clip 边缘吸附到源素材首尾」。命中时
                // 把**累计位移**替换为吸附值，再以「目标累计 − 已应用累计」驱动增量
                // （与旧实现 `useSlipDrag` 同一算法，只是位移正负号约定相反）。
                const timelineSnap = sessionRef.current.timelineSnap;
                let desiredTotal = args.deltaSec;
                {
                    const anchor = origin.slipAnchor;
                    const snapActive = computeEffectiveSnap(
                        s.snapEnabled,
                        isModifierActive(noSnapKb, args.modifiers),
                    );
                    if (
                        (anchor.isContentBearing || anchor.loopEnabled) &&
                        timelineSnap.snapClipsToSourceMedia &&
                        snapActive &&
                        timelineSnap.snapDistancePx > 0
                    ) {
                        // 内核约定正 = 向右拖；旧实现约定正 = 向左拖（起点指针 − 当前
                        // 指针），故取反号换算到同一 X 域。
                        const dir = anchor.reversed ? -1 : 1;
                        const rawWindowShift = -desiredTotal * dir;
                        const snappedW = nearestBoundarySnapOffsetSec(
                            anchor,
                            "slip",
                            rawWindowShift,
                        );
                        if (
                            snappedW != null &&
                            Math.abs(snappedW - rawWindowShift) <=
                                loopSnapThresholdSec(timelineSnap.snapDistancePx, pxPerSec) + 1e-12
                        ) {
                            desiredTotal = -snappedW * dir;
                            // 循环节命中：只高亮**真正对齐**的那一侧（媒体边界恰好
                            // 落在 Clip 起点 → 亮起点；落在终点 → 亮终点；len·r 恰为
                            // 整周期等两侧同时对齐才两缘同亮）。
                            const anchorClip = sessionRef.current.clips.find(
                                (item) => item.id === origin.clipId,
                            );
                            if (anchorClip !== undefined) {
                                const aligned = slipBoundaryAlignedSides(anchor, snappedW);
                                const clipStartSec = Math.max(0, Number(anchorClip.startSec) || 0);
                                const clipLen = Math.max(0, Number(anchorClip.lengthSec) || 0);
                                const secs: number[] = [];
                                if (aligned.start) secs.push(clipStartSec);
                                if (aligned.end) secs.push(clipStartSec + clipLen);
                                if (secs.length > 0) {
                                    publishSnapHighlights(SNAP_HIGHLIGHT_GROUP, [
                                        buildLoopBoundaryHighlightEntry({
                                            secs,
                                            trackId: anchorClip.trackId,
                                            clipId: origin.clipId,
                                        }),
                                    ]);
                                } else {
                                    clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                                }
                            }
                        } else {
                            clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                        }
                    } else {
                        clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                    }
                }
                const dApplied = desiredTotal - origin.appliedSlipSec;
                if (Math.abs(dApplied) < 1e-12) return;
                origin.appliedSlipSec = desiredTotal;
                const updates: Array<{
                    clipId: string;
                    sourceStartSec: number;
                    sourceEndSec: number;
                }> = [];
                for (const participant of origin.participants) {
                    const clip = sessionRef.current.clips.find(
                        (item) => item.id === participant.clipId,
                    );
                    if (clip === undefined) continue;
                    // ⚠️ **取反号**：`dApplied` 在「屏幕位移」域（内核约定正 = 向右拖），
                    // 而 `computeSlipWindow` 要的是「窗口平移量」域（正 = 源窗口向素材
                    // **后段**平移）。向右拖 = 把内容往右推 = 露出**更早**的素材 = 窗口
                    // 向前段平移，故屏幕正位移对应窗口负平移。
                    //
                    // 旧实现 `useSlipDrag` 的 `desiredTotal = 起点指针 − 当前指针`
                    // （向右拖为负）恰好已经是窗口域，所以它直接传即可；内核给的是
                    // 屏幕域，漏掉这个负号会让方向整体反过来（拖右显示更晚的素材）。
                    const next = computeSlipWindow(clip, -dApplied);
                    if (next === null) continue;
                    origin.lastSourceById.set(participant.clipId, next);
                    updates.push({ clipId: participant.clipId, ...next });
                }
                batch(() => {
                    for (const update of updates) {
                        dispatch(
                            setClipSourceRange({
                                clipId: update.clipId,
                                sourceStartSec: update.sourceStartSec,
                                sourceEndSec: update.sourceEndSec,
                            }),
                        );
                    }
                });
                return;
            }
            // copy 模式判定：复用旧实现的函数（含"已配置绑定为准 + 非 macOS 的 Ctrl
            // 回退"）。**单向**——一旦进入 copy 就不再退回移动，避免松手瞬间语义反转。
            origin.copyMode = resolveClipDragCopyMode({
                existingCopyMode: origin.copyMode,
                ctrlKey: args.modifiers.ctrlKey,
                modifierActive: isModifierActive(copyDragKb, args.modifiers),
            });
            // 免吸附修饰键（默认 Shift）把吸附总开关临时取反——与旧实现同一函数，
            // 拖拽中途按下/松开即时生效。
            const snapActive = computeEffectiveSnap(
                s.snapEnabled,
                isModifierActive(noSnapKb, args.modifiers),
            );
            const rawStart = Math.max(0, origin.startSec + args.deltaSec);
            // 吸附：复用旧实现的 snapTimelineDetailed（多源候选 + 取更近者），
            // 内核只给几何位移，吸附规则不在内核里重写。
            //
            // `highlight` 必须传：吸附高亮的**发布与清除**都由该函数按此选项统一
            // 处理（见 useTimelineState.snapTimelineDetailed）。不传时吸附本身仍
            // 生效、位置也对，但完全没有视觉反馈——表现为"吸附没生效"。
            // `excludeClipIds` 必须覆盖**全部参与者**：整组一起移动时，组内其他
            // clip 不应成为自己的吸附目标。
            const nextStart = snapActive
                ? snapTimelineDetailed(rawStart, "clip", {
                      originSec: origin.startSec,
                      anchorTrackId: args.targetTrackId,
                      excludeClipIds: new Set(origin.participants.map((item) => item.clipId)),
                      moveLengthSec: origin.lengthSec,
                      moveSnapOffsetSec: origin.snapOffsetSec,
                      highlight: {
                          sources: [{ trackId: args.targetTrackId, clipId: args.clipId }],
                      },
                  }).sec
                : rawStart;
            // 吸附被关闭（拖拽中切开关 / 按住临时取反键）：高亮必须清掉，
            // 否则会残留上一次的吸附提示（旧实现同样在 else 分支清除）。
            if (!snapActive) clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            const trackIds = sessionRef.current.tracks.map((track) => track.id);
            const targetTrackIndex = trackIds.indexOf(args.targetTrackId);
            const deltaTrack =
                targetTrackIndex >= 0 && origin.anchorTrackIndex >= 0
                    ? targetTrackIndex - origin.anchorTrackIndex
                    : 0;
            // 拖到全部轨道之下 = 新建轨道哨兵：乐观位置写哨兵轨（与旧实现
            // `moveClipTrack(NEW_TRACK_SENTINEL)` 同源），面板据此渲染幽灵行。
            const dropToNewTrack = args.targetTrackId === NEW_TRACK_SENTINEL;
            if (dropToNewTrack !== kernelDropToNewTrackRef.current) {
                kernelDropToNewTrackRef.current = dropToNewTrack;
                setKernelDropToNewTrack(dropToNewTrack);
            }
            const { moves, deltaStartSec } = applyKernelEditDelta({
                participants: origin.participants,
                deltaStartSec: nextStart - origin.startSec,
                deltaTrack,
                trackIds,
                dropToNewTrack,
            });
            origin.lastDeltaStartSec = deltaStartSec;
            if (origin.copyMode) {
                // copy：**原 clip 不动**，只更新 ghost（整组，内容坐标）。
                setKernelGhost(
                    origin.participants.map((participant, index) => ({
                        key: participant.clipId,
                        leftPx: moves[index].startSec * pxPerSec,
                        widthPx: Math.max(1, participant.lengthSec * pxPerSec),
                        trackId: moves[index].trackId,
                    })),
                );
                // 复制不移动原片：波纹跟随集恢复原位（覆盖「拖拽中途切到复制」的残留预览）。
                applyRippleFollowerShift(dispatch, origin.rippleFollowers, 0);
                return;
            }
            batch(() => {
                for (const move of moves) {
                    dispatch(moveClipStart({ clipId: move.clipId, startSec: move.startSec }));
                    dispatch(moveClipTrack({ clipId: move.clipId, trackId: move.trackId }));
                }
                // 波纹（自动跟进）实时预览：后续 clip 随拖拽同步平移，位移取
                // **钳制后的共享位移**（否则左边界处会与参与者错位）。
                applyRippleFollowerShift(dispatch, origin.rippleFollowers, deltaStartSec);
            });
            // 自动交叉淡化实时预览：按当前（乐观）位置重算重叠并更新自动 fade。
            // 必须用 `store.getState().session`（同步新鲜）而不是 sessionRef——
            // react-redux 的 batch 会延迟订阅回调，batch 内 sessionRef 仍是上一帧
            // 位置，会让"拖开瞬间"的预览滞留最后一帧自动淡化长度。
            if (s.autoCrossfadeEnabled) {
                previewAutoCrossfade(
                    store.getState().session,
                    origin.editedXfadeClipIds,
                    dispatch,
                    origin.initialCrossfadeSides,
                );
            }
        },
        [
            beginKernelGestureInteraction,
            copyDragKb,
            dispatch,
            multiSelectedClipIds,
            noSnapKb,
            pxPerSec,
            s.autoCrossfadeEnabled,
            s.snapEnabled,
            sessionRef,
            snapTimelineDetailed,
        ],
    );

    /**
     * 内核拖拽收尾：提交或回滚。
     *
     * 取消路径（Esc / pointercancel）必须把乐观位置还原——否则 Redux 会停在半途
     * 位置，与后端分叉（旧实现同样在取消分支显式回滚）。
     */
    const handleKernelDragCommit = React.useCallback(
        (args: {
            clipId: string;
            deltaSec: number;
            targetTrackId: string;
            cancelled: boolean;
            modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
        }) => {
            const origin = kernelDragOriginRef.current;
            kernelDragOriginRef.current = null;
            if (origin === null) return;
            // 手势结束：清掉吸附高亮（旧实现同样在收尾清除，否则最后一次的
            // 吸附提示会一直挂在画面上）。
            clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);

            if (origin.slipMode) {
                if (args.cancelled) {
                    batch(() => {
                        for (const [clipId, base] of origin.baseSourceById) {
                            dispatch(
                                setClipSourceRange({
                                    clipId,
                                    sourceStartSec: base.sourceStartSec,
                                    sourceEndSec: base.sourceEndSec,
                                }),
                            );
                        }
                    });
                    endKernelGestureInteraction();
                    return;
                }
                // 零位移（未越过阈值 / 拖回原位）：不写后端（旧实现同样不置 dirty）。
                if (origin.lastSourceById.size === 0) {
                    endKernelGestureInteraction();
                    return;
                }
                dispatch(checkpointHistory());
                // 用**交互数学结果**提交（不回读 Redux——防并发更新 / 历史归一化
                // 把窗口值污染，与旧实现 `lastById` 同源）。
                void dispatch(
                    setClipsStateBulkRemote({
                        updates: [...origin.lastSourceById].map(([clipId, window]) => ({
                            clipId,
                            sourceStartSec: window.sourceStartSec,
                            sourceEndSec: window.sourceEndSec,
                        })),
                    }),
                )
                    .unwrap()
                    .catch(() => undefined)
                    .finally(endKernelGestureInteraction);
                return;
            }

            if (origin.copyMode) {
                setKernelGhost(null);
                // 落点解析必须早于取消早退：取消路径同样要清掉「落到新轨」标记，
                // 否则虚线新轨行会一直挂在画面上，且之后每一次普通移动都会被误读成
                // 哨兵落点（`kernelDropToNewTrackRef` 的语义是"本次手势的落点"，
                // 手势结束就必须复位——与下方 move 分支同一清理方式）。
                const dropTarget = resolveKernelDropTarget({
                    targetTrackId: args.targetTrackId,
                    trackIds: sessionRef.current.tracks.map((track) => track.id),
                    anchorTrackIndex: origin.anchorTrackIndex,
                    newTrackSentinel: NEW_TRACK_SENTINEL,
                });
                if (kernelDropToNewTrackRef.current) {
                    kernelDropToNewTrackRef.current = false;
                    setKernelDropToNewTrack(false);
                }
                // copy 模式下原 clip 从未被移动：既不需要回滚，也不走 move 提交。
                if (args.cancelled) {
                    endKernelGestureInteraction();
                    return;
                }
                // 落库复用抽出的共享函数（与旧实现**同一份**复制语义）：
                // 参与集合整体复制，目标轨按各自初始序号 + 同一偏移量解析。
                const trackIds = sessionRef.current.tracks.map((track) => track.id);
                void copyClipsFromDrag({
                    sourceClipIds: origin.participants.map((item) => item.clipId),
                    initialById: Object.fromEntries(
                        origin.participants.map((item) => [
                            item.clipId,
                            { startSec: item.startSec, trackId: item.trackId },
                        ]),
                    ),
                    initialTrackIndexById: Object.fromEntries(
                        origin.participants.map((item) => [item.clipId, item.trackIndex]),
                    ),
                    // 用**吸附后**的共享位移，与 ghost 预览的位置一致
                    // （用内核原始位移会绕开吸附，表现为"预览吸附、落库不吸附"）。
                    deltaSec: origin.lastDeltaStartSec,
                    // 与预览分支同一套落点语义（预览分支内联算 `deltaTrack` /
                    // `dropToNewTrack`）：哨兵 → 新建轨道；已有轨道 → 相对锚点的轨道
                    // 偏移量。写死 false / 0 会让"幽灵预览能到新轨道和其他轨道、落库
                    // 却留在原轨"，两处解析结果必须一致。
                    dropToNewTrack: dropTarget.dropToNewTrack,
                    trackOffset: dropTarget.trackOffset,
                    allowTrackMove: true,
                    // 选区是否跨多条轨道（旧实现 `useClipDrag` 的
                    // `hasMixedTrackSelection`）。只有跨轨时才按**来源轨道跨度**建同样
                    // 多的新轨、让成员各自落位；写死 false 会让"跨轨多选拖到轨道区下方"
                    // 把整组塌到同一条新轨上，成员之间的相对轨道布局丢失。
                    hasMixedTrackSelection:
                        new Set(origin.participants.map((item) => item.trackId)).size > 1,
                    autoCrossfadeEnabled: s.autoCrossfadeEnabled,
                    dispatch,
                    sessionRef,
                    setMultiSelectedClipIds,
                    // 每个参与者按各自初始轨道序号 + 同一偏移量解析目标轨。
                    //
                    // 特殊说明 1：这里的 id 查询**必须带上 `trackOffset`**。只写
                    // `trackIds[participant.trackIndex]` 会永远返回原轨（偏移量为 0 时
                    // 才恰好正确），使整个跨轨修复变成静默无效——调用方
                    // （`copyClipsFromDrag`）此时已经按 `trackOffset !== 0` 判定过
                    // "应该跨轨"，拿到原轨后不会报错，只会默默同轨复制。
                    //
                    // 特殊说明 2：越界必须**夹取**到 [0, lastIndex]，与预览分支
                    // （`applyKernelEditDelta` 的 `Math.min(lastIndex, Math.max(0, …))`）
                    // 逐字一致。若越界时回落**原轨**，"幽灵显示到了末轨、落库却留在
                    // 原轨"会重新出现：跨轨多选 + 偏移越界时实测预览把两个 participant
                    // 都放到末轨，而落库把越界的那个留在原轨。
                    resolveTrackIdByOffset: (clipId) => {
                        const participant = origin.participants.find(
                            (item) => item.clipId === clipId,
                        );
                        if (participant === undefined || participant.trackIndex < 0) return null;
                        const lastIndex = trackIds.length - 1;
                        if (lastIndex < 0) return null;
                        const targetIndex = Math.min(
                            lastIndex,
                            Math.max(0, participant.trackIndex + dropTarget.trackOffset),
                        );
                        return trackIds[targetIndex] ?? null;
                    },
                    maybeSelectTargetTrack: (trackId) => {
                        // 跨轨复制后把**目标轨道**设为当前轨道（旧实现
                        // `useClipDrag` 的 `maybeSelectTargetTrack` 同源：目标是锚点的
                        // 原轨道 / 已经是当前轨道时跳过）。
                        //
                        // `applySelectedClip: false`：这里只切"当前轨道"，不得让后端
                        // 记忆的 `selected_clip_id` 异步复活（契约与
                        // `handleKernelSeek` 的空白点击同源，见 019e93ed）。
                        if (trackId === null) return;
                        if (trackId === origin.trackId) return;
                        if (sessionRef.current.selectedTrackId === trackId) return;
                        void dispatch(selectTrackRemote({ trackId, applySelectedClip: false }));
                    },
                    createNewTracksForDrop: (span: number) =>
                        createTrackIdsForDrop({ dispatch, sessionRef }, span),
                    createNewTrackForDrop: async () => {
                        const created = await createTrackIdsForDrop({ dispatch, sessionRef }, 1);
                        return created[0] ?? null;
                    },
                })
                    .catch(() => undefined)
                    .finally(endKernelGestureInteraction);
                return;
            }
            if (args.cancelled) {
                batch(() => {
                    for (const participant of origin.participants) {
                        dispatch(
                            moveClipStart({
                                clipId: participant.clipId,
                                startSec: participant.startSec,
                            }),
                        );
                        dispatch(
                            moveClipTrack({
                                clipId: participant.clipId,
                                trackId: participant.trackId,
                            }),
                        );
                    }
                });
                // 波纹跟随集恢复原位（预览期间已被平移）。
                applyRippleFollowerShift(dispatch, origin.rippleFollowers, 0);
                if (kernelDropToNewTrackRef.current) {
                    kernelDropToNewTrackRef.current = false;
                    setKernelDropToNewTrack(false);
                }
                endKernelGestureInteraction();
                return;
            }
            // 哨兵轨：先清掉落点标记（幽灵行随手势一起消失），再建轨并落库。
            if (kernelDropToNewTrackRef.current) {
                kernelDropToNewTrackRef.current = false;
                setKernelDropToNewTrack(false);
                dispatch(checkpointHistory());
                void createNewTrackForKernelDrop({
                    clipIds: origin.participants.map((item) => item.clipId),
                    // 目标起点取**当前**（乐观、含吸附）值，与幽灵行显示的位置一致。
                    startSecById: Object.fromEntries(
                        origin.participants.map((item) => {
                            const clip = sessionRef.current.clips.find(
                                (candidate) => candidate.id === item.clipId,
                            );
                            return [item.clipId, clip?.startSec ?? item.startSec] as const;
                        }),
                    ),
                    originTrackIdById: Object.fromEntries(
                        origin.participants.map((item) => [item.clipId, item.trackId] as const),
                    ),
                    originStartSecById: Object.fromEntries(
                        origin.participants.map((item) => [item.clipId, item.startSec] as const),
                    ),
                    // 各成员的初始轨道序号：跨轨选区据此按跨度建多条新轨并各自落位
                    // （与 copy 路径的 `hasMixedTrackSelection` / span 建轨同源）。
                    trackIndexById: Object.fromEntries(
                        origin.participants.map((item) => [item.clipId, item.trackIndex] as const),
                    ),
                    dispatch,
                    sessionRef,
                    moveLinkedParams: sessionRef.current.lockParamLinesEnabled,
                })
                    .catch(() => undefined)
                    .finally(endKernelGestureInteraction);
                return;
            }
            // 提交值取 Redux 里的当前值（预览已写入**吸附后**的结果）——
            // 用 origin + 内核原始位移会绕开吸附，导致"预览吸附、提交不吸附"。
            const session = sessionRef.current;
            const moves = origin.participants
                .map((participant) => {
                    const clip = session.clips.find((item) => item.id === participant.clipId);
                    const startSec = clip?.startSec ?? participant.startSec;
                    const trackId = clip?.trackId ?? participant.trackId;
                    // 零位移（拖回原位 / 未越过阈值）的成员不入提交集：旧实现
                    // `useClipDrag` 同样以 `|Δstart| > 1e-6 || 换轨` 过滤，全员
                    // 无变化时**整段提交都不发生**（不写后端、不置 dirty）。
                    const changedStart = Math.abs(startSec - participant.startSec) > 1e-6;
                    const changedTrack = trackId !== participant.trackId;
                    if (!changedStart && !changedTrack) return null;
                    return { clipId: participant.clipId, startSec, trackId };
                })
                .filter(
                    (move): move is { clipId: string; startSec: number; trackId: string } =>
                        move !== null,
                );
            if (moves.length === 0) {
                // 全员零位移：没有任何远程写入，交互锁直接释放。
                endKernelGestureInteraction();
                return;
            }
            dispatch(checkpointHistory());
            // 跨轨移动后把**目标轨道**设为当前轨道（旧实现 `useClipDrag` 的
            // `maybeSelectTargetTrack(drag.lastTrackId)` 同源：目标是锚点原轨道或已是
            // 当前轨道时跳过）。`applySelectedClip: false` 见 copy 分支的说明。
            const anchorMove = moves.find((move) => move.clipId === origin.clipId);
            if (
                anchorMove !== undefined &&
                anchorMove.trackId !== origin.trackId &&
                session.selectedTrackId !== anchorMove.trackId
            ) {
                void dispatch(
                    selectTrackRemote({ trackId: anchorMove.trackId, applySelectedClip: false }),
                );
            }
            // 「锁定参数线」必须随移动一起透出（旧实现 `useClipDrag` 同源）：
            // 漏传会让参数线在该开关打开时**不跟随 clip 移动**，且没有任何提示。
            const moveLinkedParams = session.lockParamLinesEnabled;
            const movePromise =
                moves.length > 1
                    ? dispatch(moveClipsRemote({ moves, moveLinkedParams })).unwrap()
                    : dispatch(
                          moveClipRemote({
                              clipId: moves[0].clipId,
                              startSec: moves[0].startSec,
                              trackId: moves[0].trackId,
                              moveLinkedParams,
                          }),
                      ).unwrap();
            void (async () => {
                try {
                    await movePromise;
                } finally {
                    // 自动交叉淡化：按移动后的新重叠关系写回自动 fade；开关关闭时
                    // 只清理「已脱离重叠」的自动值（保证分离后手动 fade 能恢复显示）。
                    const latest = sessionRef.current;
                    if (s.autoCrossfadeEnabled) {
                        await applyAutoCrossfade(latest, origin.editedXfadeClipIds, dispatch, {
                            affectedSides: origin.initialCrossfadeSides,
                        });
                    } else {
                        await applyDetachedAutoCrossfadeClears(
                            latest,
                            origin.editedXfadeClipIds,
                            dispatch,
                            origin.initialCrossfadeSides,
                        );
                    }
                }
            })()
                .catch(() => undefined)
                // 交互锁在落库（含自动交叉淡化写回）完成后才释放——旧实现同一约定。
                .finally(endKernelGestureInteraction);
        },
        [
            dispatch,
            endKernelGestureInteraction,
            s.autoCrossfadeEnabled,
            sessionRef,
            setMultiSelectedClipIds,
        ],
    );

    /** 内核 trim / stretch：按下时的原始几何（把相对位移换算为绝对值，并支持回滚）。 */
    const kernelTrimOriginRef = React.useRef<{
        clipId: string;
        startSec: number;
        lengthSec: number;
        trackId: string;
        sourceStartSec: number;
        sourceEndSec: number;
        /**
         * 本次边缘手势的模式。
         *
         * `Alt`（`modifier.clipStretch`）按住 = **拉伸**（改播放速率、内容不被裁掉），
         * 否则 = **裁切**（改源区间）。模式在**按下时**定死，拖拽中途按/松 Alt 不切换
         * ——与旧实现一致（旧实现由 `altPressed` 在 pointerdown 时快照）。
         */
        mode: "trim" | "stretch";
        /** 拉伸所需的按下时基准（裁切模式不使用）。 */
        basePlaybackRate: number;
        baseFadeInSec: number;
        baseFadeOutSec: number;
        baseSnapOffsetSec: number;
        /**
         * 本次边缘手势作用于的全部 clip（多选集合 + 编组展开）。
         *
         * 旧实现 `useEditDrag` 的 `supportsGroupExpansion` 对 trim 与 stretch 都
         * 展开编组（只排除 fade / gain）；裁切按**锚点位移**逐 clip 换算，拉伸则
         * 走下方的 `stretchGroup` 整组等比缩放。
         */
        participants: KernelEditParticipant[];
        /**
         * 组拉伸：多选 + 编组展开后的整组缩放状态（不满足条件时为 null）。
         *
         * 旧实现用 `buildStretchGroupState` 判定「锚点是否位于选区边界」：只有锚点
         * 是选区最左（拖左缘）/ 最右（拖右缘）的成员时才做整组等比缩放，否则退化
         * 为单 clip 拉伸——判定与几何都在纯函数里，内核不重复实现。
         */
        stretchGroup: StretchGroupState | null;
        /** 组拉伸：各成员按下时的吸附偏移（随长度比例缩放；提交与回滚共用）。 */
        stretchBaseSnapOffsetById: Map<string, number>;
        /** 各参与者的按下时几何（裁切按**锚点位移**逐 clip 换算）。 */
        baseById: Map<
            string,
            {
                startSec: number;
                lengthSec: number;
                sourceStartSec: number;
                sourceEndSec: number;
                /** 该 clip 的播放速率：源位移 = 时间轴位移 × 速率。 */
                playbackRate: number;
            }
        >;
        /**
         * 自动交叉淡化：受影响 clip（= 参与者）与编辑前每侧重叠关系、可调整侧。
         *
         * 与旧实现 `useEditDrag` 同源：`initialCrossfadeSides` 记录编辑前的重叠关系
         * （用于「拖开时只清自动、保留手动 fade」），`editSides` 限定本次编辑**允许
         * 自动调整**的侧（裁切左缘只动 fadeIn，右缘只动 fadeOut）——否则裁切左缘会
         * 顺带改掉右缘与邻居的交叉淡化。
         */
        xfadeClipIds: string[];
        initialCrossfadeSides: ReturnType<typeof computeInitialCrossfadeSides>;
        editSides: Record<string, { fadeIn: boolean; fadeOut: boolean }>;
        /**
         * 波纹跟随集（按下时快照）。
         *
         * 旧实现的实时波纹覆盖 trim 与 stretch（由区域右缘净位移驱动），内核原先只在
         * **拖拽移动**时做波纹——裁切 / 拉伸期间后续 clip 不跟随，松手才跳过去。
         * 快照规则与旧实现同源：原点 = 参与者最早起点、轨道集 = 参与者所在轨道。
         */
        rippleFollowers: RippleFollowerMap;
        /**
         * 循环节 / 内容边界吸附的按下时快照（`toBoundarySnapClip`）。
         *
         * 裁切会逐帧改写源窗口，而同余式必须基于**按下时**的源窗口 / 循环状态
         * ——与旧实现 `useEditDrag` 用 `drag.baseByClipId` 同一约定。
         */
        boundaryAnchor: ReturnType<typeof toBoundarySnapClip>;
    } | null>(null);

    /**
     * 内核裁切 / 拉伸的波纹实时预览。
     *
     * 流程：按当前（乐观）几何算出**编辑区域右缘净位移** → 把跟随集平移到
     * 「初始位置 + 位移」。
     *
     * 特殊说明：
     * - 位移取区域右缘而不是锚点右缘：与后端区域化波纹一致，多成员编辑时以整个
     *   区域的最右缘为准（`computeRegionRightEdgeDelta` 是共享纯函数）；
     * - 位移带符号，向左收拢时为负——跟随集必须跟着向左（对 0 取 max 会吞掉负位移）；
     * - 用 `store.getState().session`（同步新鲜）：react-redux 的 batch 会延迟订阅
     *   回调，batch 内 `sessionRef` 仍是上一帧位置。
     *
     * @param origin 本次边缘手势的按下时快照（含 `rippleFollowers` 与参与者几何）。
     * @returns 无返回值。
     */
    const applyKernelTrimRipplePreview = React.useCallback(
        (origin: NonNullable<typeof kernelTrimOriginRef.current>) => {
            // `origin.baseById` 是 Map（按下时按参与者建），纯函数要 Record——
            // 这里就地转换，避免为"一次读取"再存一份平行结构。
            const baseById: Record<string, { startSec: number; lengthSec: number }> = {};
            for (const [clipId, base] of origin.baseById) {
                baseById[clipId] = { startSec: base.startSec, lengthSec: base.lengthSec };
            }
            const rippleRightDelta = computeRegionRightEdgeDelta({
                clipIds: origin.participants.map((item) => item.clipId),
                baseById,
                clips: store.getState().session.clips,
            });
            // 位移为 0 时也要执行：它等价于"把跟随集恢复到初始位置"，正是取消 /
            // 拖回原位所需要的（旧实现同样无条件调用）。
            applyRippleFollowerShift(dispatch, origin.rippleFollowers, rippleRightDelta);
        },
        [dispatch, store],
    );

    /**
     * 内核 trim 预览：写乐观几何。
     *
     * 源区间必须同步改：左边缘 trim 改 `sourceStartSec`、右边缘改 `sourceEndSec`——
     * 只改 `lengthSec` 会让音频内容被拉伸（波形与音频对不上），而 trim 的语义是
     * **裁切**（内容不滑动）。
     */
    const handleKernelTrimPreview = React.useCallback(
        (args: {
            clipId: string;
            edge: "left" | "right";
            startSec: number;
            lengthSec: number;
            deltaSec: number;
            modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
        }) => {
            if (kernelTrimOriginRef.current?.clipId !== args.clipId) {
                const session = sessionRef.current;
                const clip = session.clips.find((item) => item.id === args.clipId);
                if (clip === undefined) return;
                // 参与集合：裁切与拉伸都是「多选 + 编组展开」（旧实现
                // supportsGroupExpansion 覆盖 trim / stretch，只排除 fade / gain）。
                const stretchMode = isModifierActive(stretchKbRef.current, args.modifiers);
                const participants = resolveKernelEditParticipants({
                    anchorClipId: clip.id,
                    multiSelectedClipIds,
                    clips: session.clips,
                    trackIds: session.tracks.map((track) => track.id),
                    ignoreGrouping: session.ignoreGrouping,
                    disabledGroupIds: session.disabledGroupIds,
                    expandGroups: true,
                });
                // 组拉伸：锚点位于选区边界时才成立（纯函数判定，含「选区不足两个
                // 成员 → null」）；不成立时退化为单 clip 拉伸。
                const stretchGroup = stretchMode
                    ? buildStretchGroupState({
                          clips: session.clips,
                          selectedClipIds: participants.map((item) => item.clipId),
                          anchorClipId: clip.id,
                          edge: args.edge === "left" ? "stretch_left" : "stretch_right",
                      })
                    : null;
                const stretchBaseSnapOffsetById = new Map<string, number>();
                if (stretchGroup !== null) {
                    for (const clipId of stretchGroup.clipIds) {
                        const item = session.clips.find((candidate) => candidate.id === clipId);
                        stretchBaseSnapOffsetById.set(
                            clipId,
                            Math.max(0, Number(item?.snapOffsetSec) || 0),
                        );
                    }
                }
                const baseById = new Map<
                    string,
                    {
                        startSec: number;
                        lengthSec: number;
                        sourceStartSec: number;
                        sourceEndSec: number;
                        playbackRate: number;
                    }
                >();
                for (const participant of participants) {
                    const item = session.clips.find(
                        (candidate) => candidate.id === participant.clipId,
                    );
                    if (item === undefined) continue;
                    baseById.set(participant.clipId, {
                        startSec: Number(item.startSec) || 0,
                        lengthSec: Math.max(0, Number(item.lengthSec) || 0),
                        sourceStartSec: Number(item.sourceStartSec ?? 0) || 0,
                        sourceEndSec: Number(item.sourceEndSec ?? 0) || 0,
                        playbackRate: Number(item.playbackRate ?? 1) || 1,
                    });
                }
                // 自动交叉淡化：受影响集合 = 参与者；可调整侧按拖拽的边缘决定。
                const xfadeClipIds = participants.map((participant) => participant.clipId);
                const editSides: Record<string, { fadeIn: boolean; fadeOut: boolean }> = {};
                for (const id of xfadeClipIds) {
                    editSides[id] =
                        args.edge === "left"
                            ? { fadeIn: true, fadeOut: false }
                            : { fadeIn: false, fadeOut: true };
                }
                kernelTrimOriginRef.current = {
                    clipId: clip.id,
                    startSec: clip.startSec,
                    lengthSec: clip.lengthSec,
                    trackId: clip.trackId,
                    sourceStartSec: clip.sourceStartSec,
                    sourceEndSec: clip.sourceEndSec,
                    xfadeClipIds,
                    initialCrossfadeSides: computeInitialCrossfadeSides(
                        session.clips,
                        xfadeClipIds,
                    ),
                    editSides,
                    // 波纹跟随集：与旧实现 `useEditDrag` 同一套快照规则（原点取
                    // 参与者最早起点、作用轨道取参与者所在轨道）。
                    rippleFollowers: buildRippleFollowers(
                        session.clips,
                        new Set(xfadeClipIds),
                        Math.min(...participants.map((item) => item.startSec)),
                        session.rippleMode,
                        new Set(participants.map((item) => item.trackId)),
                    ),
                    // Alt 按住 = 拉伸（与旧实现 `modifier.clipStretch` 同源）。
                    mode: stretchMode ? "stretch" : "trim",
                    basePlaybackRate: Number(clip.clipPlaybackRate ?? 1) || 1,
                    baseFadeInSec: Number(clip.fadeInSec) || 0,
                    baseFadeOutSec: Number(clip.fadeOutSec) || 0,
                    baseSnapOffsetSec: Math.max(0, Number(clip.snapOffsetSec) || 0),
                    participants,
                    stretchGroup,
                    stretchBaseSnapOffsetById,
                    baseById,
                    // 循环节 / 内容边界吸附的按下时快照（与 slip 手势同一套）：
                    // 源窗口在裁切过程中逐帧变化，用当前值算同余式会漂移。
                    boundaryAnchor: toBoundarySnapClip(clip),
                };
                // 首个真实位移帧 = 手势开始：上交互锁（见 helper 说明）。
                beginKernelGestureInteraction();
            }
            const origin = kernelTrimOriginRef.current;
            if (origin === null) return;

            // 免吸附修饰键（默认 Shift）临时取反吸附总开关（与旧实现同一函数）。
            const snapActive = computeEffectiveSnap(
                s.snapEnabled,
                isModifierActive(noSnapKb, args.modifiers),
            );

            if (origin.mode === "stretch") {
                // Alt + 拖边缘 = **拉伸**：对侧边缘固定、改播放速率（内容不被裁掉）。
                // 被吸附对象是**正在拖的那条边**（左缘 / 右缘），与裁切一致；
                // 几何换算复用与旧实现共用的纯函数（单 clip 用 `computeClipStretch`、
                // 组拉伸用 `computeStretchGroupUpdate`，都含速率钳制与比例缩放）。
                const rawEdgeSec =
                    args.edge === "left" ? args.startSec : args.startSec + args.lengthSec;
                const edgeSec = snapActive
                    ? snapTimelineDetailed(rawEdgeSec, "clip", {
                          originSec:
                              args.edge === "left"
                                  ? origin.startSec
                                  : origin.startSec + origin.lengthSec,
                          anchorTrackId: origin.trackId,
                          // 排除**整组**（旧实现排除 `selectedClipIds`）：组内其他
                          // 成员随本次拉伸一起移动，不应成为自己的吸附目标。
                          excludeClipIds: new Set(origin.stretchGroup?.clipIds ?? [args.clipId]),
                          highlight: {
                              sources: [{ trackId: origin.trackId, clipId: args.clipId }],
                          },
                      }).sec
                    : rawEdgeSec;
                if (!snapActive) clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                if (origin.stretchGroup !== null) {
                    // 组拉伸：整组等比缩放（对侧整组边界固定，成员按相对位置缩放，
                    // 各自速率反算并钳制、淡变与 SnapOffset 按比例缩放）。
                    const group = origin.stretchGroup;
                    const update = computeStretchGroupUpdate({
                        group,
                        edge: args.edge === "left" ? "stretch_left" : "stretch_right",
                        pointerSec: edgeSec,
                    });
                    batch(() => {
                        for (const clipId of group.clipIds) {
                            const next = update.byId[clipId];
                            if (next === undefined) continue;
                            dispatch(moveClipStart({ clipId, startSec: next.startSec }));
                            dispatch(setClipLength({ clipId, lengthSec: next.lengthSec }));
                            dispatch(
                                setClipPlaybackRate({
                                    clipId,
                                    clipPlaybackRate: next.clipPlaybackRate,
                                }),
                            );
                            dispatch(
                                setClipFades({
                                    clipId,
                                    fadeInSec: next.fadeInSec,
                                    fadeOutSec: next.fadeOutSec,
                                }),
                            );
                            const initial = group.initialById[clipId];
                            if (initial !== undefined) {
                                // SnapOffset 随成员长度按「总比例 × 基准偏移」缩放
                                //（基准 = 按下时快照，禁止逐帧复合）。
                                const ratio = next.lengthSec / Math.max(1e-6, initial.lengthSec);
                                dispatch(
                                    setClipSnapOffset({
                                        clipId,
                                        snapOffsetSec: scaleSnapOffsetForStretch(
                                            origin.stretchBaseSnapOffsetById.get(clipId),
                                            ratio,
                                            next.lengthSec,
                                        ),
                                    }),
                                );
                            }
                        }
                    });
                    // 波纹实时预览（组拉伸改变区域右缘 → 跟随集同步平移）。
                    applyKernelTrimRipplePreview(origin);
                    if (s.autoCrossfadeEnabled) {
                        previewAutoCrossfade(
                            store.getState().session,
                            origin.xfadeClipIds,
                            dispatch,
                            origin.initialCrossfadeSides,
                            origin.editSides,
                        );
                    }
                    return;
                }
                const result = computeClipStretch({
                    edge: args.edge === "left" ? "stretch_left" : "stretch_right",
                    pointerSec: edgeSec,
                    baseStartSec: origin.startSec,
                    baseLengthSec: origin.lengthSec,
                    basePlaybackRate: origin.basePlaybackRate,
                    baseFadeInSec: origin.baseFadeInSec,
                    baseFadeOutSec: origin.baseFadeOutSec,
                    baseSnapOffsetSec: origin.baseSnapOffsetSec,
                    // 最小长度交由 `computeClipStretch` 的内部极小值护栏（`MIN_SPAN_SEC`）
                    // 兜底——旧实现 `useEditDrag` 的拉伸同样用 `minLen = 0.0`。
                });
                batch(() => {
                    dispatch(moveClipStart({ clipId: args.clipId, startSec: result.startSec }));
                    dispatch(setClipLength({ clipId: args.clipId, lengthSec: result.lengthSec }));
                    dispatch(
                        setClipPlaybackRate({
                            clipId: args.clipId,
                            clipPlaybackRate: result.clipPlaybackRate,
                        }),
                    );
                    dispatch(
                        setClipFades({
                            clipId: args.clipId,
                            fadeInSec: result.fadeInSec,
                            fadeOutSec: result.fadeOutSec,
                        }),
                    );
                    dispatch(
                        setClipSnapOffset({
                            clipId: args.clipId,
                            snapOffsetSec: result.snapOffsetSec,
                        }),
                    );
                });
                // 波纹实时预览（单 clip 拉伸改变区域右缘 → 跟随集同步平移）。
                applyKernelTrimRipplePreview(origin);
                // 自动交叉淡化实时预览（拉伸改变重叠 → 自动 fade 随之变化）。
                if (s.autoCrossfadeEnabled) {
                    previewAutoCrossfade(
                        store.getState().session,
                        origin.xfadeClipIds,
                        dispatch,
                        origin.initialCrossfadeSides,
                        origin.editSides,
                    );
                }
                return;
            }

            // 吸附：左边缘吸**起点**、右边缘吸**右端**——两者吸附的对象不同，
            // 统一吸起点会让右边缘 trim 落在错误的位置。
            let nextStart = args.startSec;
            let nextLength = args.lengthSec;
            let deltaSec = args.deltaSec;
            if (snapActive) {
                const snapArgs = {
                    // 被吸附对象**只有正在拖的那一条边**。旧实现 `useEditDrag` 的
                    // trim 分支不传 `moveLengthSec` / `moveSnapOffsetSec`：多源候选
                    // 会把「对侧边缘 / clip 自身的吸附偏移点」也当成吸附目标，
                    // 落点因此与旧实现不同（右缘还会多出 `右缘 + 长度` 这种伪候选）。
                    originSec:
                        args.edge === "left" ? origin.startSec : origin.startSec + origin.lengthSec,
                    anchorTrackId: origin.trackId,
                    // 排除**全部参与者**（旧实现排除 `selectedClipIds`）：整组一起
                    // 裁切时，组内其他成员不应成为自己的吸附目标。
                    excludeClipIds: new Set(origin.participants.map((item) => item.clipId)),
                    // `highlight` 必须传：吸附高亮的发布 / 清除由
                    // snapTimelineDetailed 按此选项统一处理，不传则 trim 有吸附
                    // 但没有任何视觉反馈。
                    highlight: {
                        sources: [{ trackId: origin.trackId, clipId: args.clipId }],
                    },
                };
                if (args.edge === "left") {
                    nextStart = snapTimelineDetailed(args.startSec, "clip", snapArgs).sec;
                    nextLength = origin.startSec + origin.lengthSec - nextStart;
                    deltaSec = nextStart - origin.startSec;
                } else {
                    const snappedRight = snapTimelineDetailed(
                        args.startSec + args.lengthSec,
                        "clip",
                        snapArgs,
                    ).sec;
                    nextLength = Math.max(0, snappedRight - args.startSec);
                    deltaSec = nextLength - origin.lengthSec;
                }

                // ── 循环节 / 内容边界吸附（命中时**覆盖**常规网格吸附）──────
                // 与旧实现 `useEditDrag` 同一套：属于常规吸附体系（受总开关与
                // 「拖动时切换吸附」修饰键的 XOR 控制），并且需要在吸附设置里启用
                // 「Clip 边缘吸附到源素材首尾」。被吸附对象是**移动边缘相对 clip
                // 基准起点的时间线偏移**——该同余式对左右两条边同样成立，故统一处理。
                const timelineSnap = sessionRef.current.timelineSnap;
                if (timelineSnap.snapClipsToSourceMedia && timelineSnap.snapDistancePx > 0) {
                    const edgeSec = args.edge === "left" ? nextStart : args.startSec + nextLength;
                    const rawOffset = edgeSec - origin.startSec;
                    const snappedOffset = nearestBoundarySnapOffsetSec(
                        origin.boundaryAnchor,
                        "edge",
                        rawOffset,
                    );
                    if (
                        snappedOffset != null &&
                        Math.abs(snappedOffset - rawOffset) <=
                            loopSnapThresholdSec(timelineSnap.snapDistancePx, pxPerSec) + 1e-12
                    ) {
                        const snappedEdgeSec = origin.startSec + snappedOffset;
                        if (args.edge === "left") {
                            nextStart = snappedEdgeSec;
                            nextLength = origin.startSec + origin.lengthSec - nextStart;
                            deltaSec = nextStart - origin.startSec;
                        } else {
                            nextLength = Math.max(0, snappedEdgeSec - args.startSec);
                            deltaSec = nextLength - origin.lengthSec;
                        }
                        // 循环节命中：以「循环节」专用高亮**覆盖**常规吸附高亮
                        // （同组发布即整组替换）。目标（源媒体边界的投影）与被吸附
                        // 边重合于同一 x，行内双亮条强调。
                        publishSnapHighlights(SNAP_HIGHLIGHT_GROUP, [
                            buildLoopBoundaryHighlightEntry({
                                secs: [snappedEdgeSec],
                                trackId: origin.trackId,
                                clipId: args.clipId,
                            }),
                        ]);
                    }
                }
            } else {
                clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            }

            batch(() => {
                dispatch(moveClipStart({ clipId: args.clipId, startSec: nextStart }));
                dispatch(setClipLength({ clipId: args.clipId, lengthSec: nextLength }));
                // 源位移 = 时间轴位移 × 该 clip 的播放速率（rate ≠ 1 时源域与时间轴
                // 不同步；旧实现同样按 rate 折算）。
                const anchorSourceDelta = deltaSec * origin.basePlaybackRate;
                if (args.edge === "left") {
                    dispatch(
                        setClipSourceRange({
                            clipId: args.clipId,
                            sourceStartSec: origin.sourceStartSec + anchorSourceDelta,
                        }),
                    );
                } else {
                    dispatch(
                        setClipSourceRange({
                            clipId: args.clipId,
                            sourceEndSec: origin.sourceEndSec + anchorSourceDelta,
                        }),
                    );
                }
                // 多选批量：其余参与者按**锚点位移**（deltaSec）逐 clip 换算——与旧实现
                // `useEditDrag` 同源（以锚点的实际位移为基准，不是各自重新吸附）。
                for (const participant of origin.participants) {
                    if (participant.clipId === args.clipId) continue;
                    const base = origin.baseById.get(participant.clipId);
                    if (base === undefined) continue;
                    const sourceDelta = deltaSec * base.playbackRate;
                    if (args.edge === "left") {
                        dispatch(
                            moveClipStart({
                                clipId: participant.clipId,
                                startSec: base.startSec + deltaSec,
                            }),
                        );
                        dispatch(
                            setClipLength({
                                clipId: participant.clipId,
                                lengthSec: Math.max(0, base.lengthSec - deltaSec),
                            }),
                        );
                        dispatch(
                            setClipSourceRange({
                                clipId: participant.clipId,
                                sourceStartSec: base.sourceStartSec + sourceDelta,
                            }),
                        );
                    } else {
                        dispatch(
                            setClipLength({
                                clipId: participant.clipId,
                                lengthSec: Math.max(0, base.lengthSec + deltaSec),
                            }),
                        );
                        dispatch(
                            setClipSourceRange({
                                clipId: participant.clipId,
                                sourceEndSec: base.sourceEndSec + sourceDelta,
                            }),
                        );
                    }
                }
            });
            // 波纹（自动跟进）实时预览：以编辑区域**右缘净位移**为准驱动跟随集，
            // 与旧实现 `useEditDrag` 同源（同一个纯函数）。必须在三种模式（拉伸 /
            // 组拉伸 / 裁切）之后统一执行——只在某一个分支里做会让其余分支漏掉波纹。
            // 用 `store.getState().session`（同步新鲜）而不是 `sessionRef`：batch 内
            // ref 仍停在上一帧位置，会让位移滞后一帧。
            applyKernelTrimRipplePreview(origin);
            // 自动交叉淡化实时预览：裁切改变重叠 → 按当前乐观位置重算自动 fade。
            // `editSides` 限定只动本次拖拽的那一侧（裁切左缘不得改右缘的交叉淡化）。
            if (s.autoCrossfadeEnabled) {
                previewAutoCrossfade(
                    store.getState().session,
                    origin.xfadeClipIds,
                    dispatch,
                    origin.initialCrossfadeSides,
                    origin.editSides,
                );
            }
        },
        [
            beginKernelGestureInteraction,
            dispatch,
            sessionRef,
            s.autoCrossfadeEnabled,
            s.snapEnabled,
            noSnapKb,
            snapTimelineDetailed,
            stretchKbRef,
            // 参与集合在按下时解析，必须读到**最新**的多选集合：漏这个依赖会让
            // 回调闭包停在挂载时的空选择上（表现为"框选多个后仍只裁切一个"）。
            multiSelectedClipIds,
            setClipFades,
            setClipPlaybackRate,
            setClipSnapOffset,
        ],
    );

    /** 内核 trim / stretch 收尾：提交或回滚。 */
    /**
     * 内核吸附偏移手势是否已真正开始。
     *
     * 预览回调每帧触发，但 `beginInteraction` / `checkpointHistory` /
     * `beginSnapGesture` 只能做一次——用这个 ref 去重，同时给收尾一个
     * 「零位移单击」的判据（旧实现用 `drag.checkpointed`）。
     */
    const kernelSnapOffsetActiveRef = React.useRef(false);

    /**
     * 内核吸附偏移手势按下时的偏移（秒），供取消路径回滚。
     *
     * 预览逐帧改写 `snapOffsetSec` 乐观值，取消（Esc / pointercancel）必须还原到
     * 按下时的值——否则 Redux 停在中途位置，与后端分叉（与其它内核手势的取消
     * 路径同一约定）。在首次真实位移时快照：那一刻的 Redux 值就是按下时的值。
     */
    const kernelSnapOffsetBaseRef = React.useRef<{ clipId: string; snapOffsetSec: number } | null>(
        null,
    );

    /**
     * 内核吸附偏移预览：与旧实现 `useSnapOffsetDrag` 同源。
     *
     * 被吸附对象是**手柄的绝对时间线位置**（`clipStart + offset`）——不是 clip 起点；
     * 高亮发布为该 clip 所在行的亮条。落库前把偏移钳制到 `[0, clip 长度]`。
     */
    const handleKernelSnapOffsetPreview = React.useCallback(
        (args: {
            clipId: string;
            rawOffsetSec: number;
            modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
        }) => {
            const clip = sessionRef.current.clips.find((item) => item.id === args.clipId);
            if (clip === undefined) return;
            const clipStart = Number(clip.startSec) || 0;
            const clipLen = Math.max(0, Number(clip.lengthSec) || 0);
            if (!kernelSnapOffsetActiveRef.current) {
                // 首次真实位移：交互锁 + undo 检查点 + 吸附手势一起开。
                kernelSnapOffsetActiveRef.current = true;
                // 按下时的偏移快照（此刻尚未被本次手势改写）——取消路径回滚用。
                kernelSnapOffsetBaseRef.current = {
                    clipId: args.clipId,
                    snapOffsetSec: Math.max(0, Number(clip.snapOffsetSec) || 0),
                };
                beginKernelGestureInteraction();
                dispatch(checkpointHistory());
                beginSnapGesture();
            }
            const rawAbs = clipStart + args.rawOffsetSec;
            // 免吸附修饰键（默认 Shift）临时取反吸附总开关（与旧实现同一函数）。
            const snapActive = computeEffectiveSnap(
                s.snapEnabled,
                isModifierActive(noSnapKb, args.modifiers),
            );
            const nextAbs = snapActive
                ? snapTimelineDetailed(rawAbs, "clip", {
                      originSec: clipStart + (Number(clip.snapOffsetSec) || 0),
                      anchorTrackId: clip.trackId,
                      excludeClipIds: new Set([args.clipId]),
                      highlight: {
                          sources: [{ trackId: clip.trackId, clipId: args.clipId }],
                      },
                  }).sec
                : rawAbs;
            if (!snapActive) clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            dispatch(
                setClipSnapOffset({
                    clipId: args.clipId,
                    snapOffsetSec: Math.min(Math.max(nextAbs - clipStart, 0), clipLen),
                }),
            );
        },
        [dispatch, noSnapKb, s.snapEnabled, snapTimelineDetailed],
    );

    /**
     * 内核吸附偏移收尾。
     *
     * 零位移单击**不写后端**（与旧实现一致：单击不产生 undo 步）；真实拖拽则
     * 一次性持久化当前乐观值（`checkpoint: true` → 整次拖拽恰好一个撤销步）。
     */
    const handleKernelSnapOffsetCommit = React.useCallback(
        (args: { clipId: string; cancelled: boolean; changed: boolean }) => {
            const wasActive = kernelSnapOffsetActiveRef.current;
            const base = kernelSnapOffsetBaseRef.current;
            kernelSnapOffsetActiveRef.current = false;
            kernelSnapOffsetBaseRef.current = null;
            clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            // 手势从未真正开始（零位移单击）：begin* 都没调用过，不能 end*。
            if (!wasActive) return;
            endSnapGesture();
            if (args.cancelled || !args.changed) {
                // 取消：乐观值必须还原到按下时的偏移，否则 Redux 与后端分叉
                // （与其它内核手势的取消路径同一约定）。
                if (args.cancelled && base !== null && base.clipId === args.clipId) {
                    dispatch(
                        setClipSnapOffset({
                            clipId: args.clipId,
                            snapOffsetSec: base.snapOffsetSec,
                        }),
                    );
                }
                endKernelGestureInteraction();
                return;
            }
            const clip = sessionRef.current.clips.find((item) => item.id === args.clipId);
            const clipLen = Math.max(0, Number(clip?.lengthSec) || 0);
            void dispatch(
                setClipStateRemote({
                    clipId: args.clipId,
                    snapOffsetSec: Math.min(Math.max(Number(clip?.snapOffsetSec) || 0, 0), clipLen),
                    checkpoint: true,
                }),
            )
                .unwrap()
                .catch(() => {
                    // 失败不产生 unhandled rejection；交互锁仍需释放。
                })
                .finally(endKernelGestureInteraction);
        },
        [dispatch, endKernelGestureInteraction],
    );

    const handleKernelTrimCommit = React.useCallback(
        (args: {
            clipId: string;
            edge: "left" | "right";
            startSec: number;
            lengthSec: number;
            cancelled: boolean;
        }) => {
            const origin = kernelTrimOriginRef.current;
            kernelTrimOriginRef.current = null;
            if (origin === null) return;
            // 手势结束：清掉吸附高亮（与拖拽收尾同源）。
            clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            // 波纹跟随集：取消时**必须还原原位**（预览期间已被平移），否则 Redux 会
            // 停在中途位置、与后端分叉。提交时不还原——后端按权威结果写回，
            // 前端保留的乐观位置正好避免"松手回跳再跳过去"。
            if (args.cancelled) {
                applyRippleFollowerShift(dispatch, origin.rippleFollowers, 0);
            }

            if (origin.mode === "stretch") {
                // 拉伸收尾：回滚 / 提交都作用于**五个字段**（起点、长度、速率、
                // 两侧淡变、吸附偏移）——只还原其中一部分会让预览与落库分叉。
                if (origin.stretchGroup !== null) {
                    // ── 组拉伸：整组成员一起回滚 / 提交 ─────────────────────
                    const group = origin.stretchGroup;
                    if (args.cancelled) {
                        batch(() => {
                            for (const clipId of group.clipIds) {
                                const initial = group.initialById[clipId];
                                if (initial === undefined) continue;
                                dispatch(moveClipStart({ clipId, startSec: initial.startSec }));
                                dispatch(setClipLength({ clipId, lengthSec: initial.lengthSec }));
                                dispatch(
                                    setClipPlaybackRate({
                                        clipId,
                                        clipPlaybackRate: initial.clipPlaybackRate,
                                    }),
                                );
                                dispatch(
                                    setClipFades({
                                        clipId,
                                        fadeInSec: initial.fadeInSec,
                                        fadeOutSec: initial.fadeOutSec,
                                    }),
                                );
                                dispatch(
                                    setClipSnapOffset({
                                        clipId,
                                        snapOffsetSec:
                                            origin.stretchBaseSnapOffsetById.get(clipId) ?? 0,
                                    }),
                                );
                            }
                        });
                        // 自动交叉淡化回到按下时的重叠关系（按已还原的几何重算）。
                        if (s.autoCrossfadeEnabled) {
                            previewAutoCrossfade(
                                store.getState().session,
                                origin.xfadeClipIds,
                                dispatch,
                                origin.initialCrossfadeSides,
                                origin.editSides,
                            );
                        }
                        return;
                    }
                    dispatch(checkpointHistory());
                    // 提交值取 Redux 当前值（预览已写入钳制后的结果）。
                    const groupSession = sessionRef.current;
                    const updates = group.clipIds.flatMap((clipId) => {
                        const clip = groupSession.clips.find((item) => item.id === clipId);
                        if (clip === undefined) return [];
                        return [
                            {
                                clipId,
                                startSec: clip.startSec,
                                lengthSec: clip.lengthSec,
                                clipPlaybackRate: Number(clip.clipPlaybackRate ?? 1) || 1,
                                fadeInSec: Number(clip.fadeInSec) || 0,
                                fadeOutSec: Number(clip.fadeOutSec) || 0,
                                snapOffsetSec: Math.max(0, Number(clip.snapOffsetSec) || 0),
                            },
                        ];
                    });
                    if (updates.length === 0) {
                        endKernelGestureInteraction();
                        return;
                    }
                    const groupPersist = dispatch(setClipsStateBulkRemote({ updates })).unwrap();
                    void (async () => {
                        try {
                            await groupPersist;
                        } finally {
                            // 速率二次写回：后端可能按自身计算覆盖速率，前端计算值
                            // 才是权威（与旧实现 `reapplyRates` 同源，只回写非 1 的成员）。
                            for (const update of updates) {
                                if (update.clipPlaybackRate === 1) continue;
                                dispatch(
                                    setClipPlaybackRate({
                                        clipId: update.clipId,
                                        clipPlaybackRate: update.clipPlaybackRate,
                                    }),
                                );
                            }
                            // 锁定参数线：按**根轨道**聚合成员的时域映射，一次请求
                            // 完成该轨道的曲线映射（与旧实现同源）。
                            if (sessionRef.current.lockParamLinesEnabled) {
                                const mappingsByRootTrack = new Map<
                                    string,
                                    StretchRangeMapping[]
                                >();
                                for (const clipId of group.clipIds) {
                                    const initial = group.initialById[clipId];
                                    const now = sessionRef.current.clips.find(
                                        (item) => item.id === clipId,
                                    );
                                    if (initial === undefined || now === undefined) continue;
                                    const rootTrackId = resolveRootTrackId(
                                        sessionRef.current.tracks,
                                        now.trackId,
                                    );
                                    if (rootTrackId === null || rootTrackId === undefined) {
                                        continue;
                                    }
                                    const mappings = mappingsByRootTrack.get(rootTrackId) ?? [];
                                    mappings.push({
                                        oldStartSec: initial.startSec,
                                        oldLengthSec: initial.lengthSec,
                                        newStartSec: now.startSec,
                                        newLengthSec: now.lengthSec,
                                    });
                                    mappingsByRootTrack.set(rootTrackId, mappings);
                                }
                                await Promise.allSettled(
                                    Array.from(mappingsByRootTrack, ([trackId, mappings]) =>
                                        stretchTrackLinkedParams(trackId, mappings),
                                    ),
                                );
                                dispatch(bumpParamsEpoch());
                            }
                            // 自动交叉淡化：按落库后的重叠关系写回自动 fade（开关
                            // 关闭时只清理「已脱离重叠」的自动值）。
                            const latest = sessionRef.current;
                            if (s.autoCrossfadeEnabled) {
                                await applyAutoCrossfade(latest, origin.xfadeClipIds, dispatch, {
                                    affectedSides: origin.initialCrossfadeSides,
                                    editSides: origin.editSides,
                                });
                            } else {
                                await applyDetachedAutoCrossfadeClears(
                                    latest,
                                    origin.xfadeClipIds,
                                    dispatch,
                                    origin.initialCrossfadeSides,
                                    origin.editSides,
                                );
                            }
                        }
                    })()
                        .catch(() => undefined)
                        .finally(endKernelGestureInteraction);
                    return;
                }
                if (args.cancelled) {
                    batch(() => {
                        dispatch(
                            moveClipStart({ clipId: origin.clipId, startSec: origin.startSec }),
                        );
                        dispatch(
                            setClipLength({ clipId: origin.clipId, lengthSec: origin.lengthSec }),
                        );
                        dispatch(
                            setClipPlaybackRate({
                                clipId: origin.clipId,
                                clipPlaybackRate: origin.basePlaybackRate,
                            }),
                        );
                        dispatch(
                            setClipFades({
                                clipId: origin.clipId,
                                fadeInSec: origin.baseFadeInSec,
                                fadeOutSec: origin.baseFadeOutSec,
                            }),
                        );
                        dispatch(
                            setClipSnapOffset({
                                clipId: origin.clipId,
                                snapOffsetSec: origin.baseSnapOffsetSec,
                            }),
                        );
                    });
                    endKernelGestureInteraction();
                    return;
                }
                dispatch(checkpointHistory());
                // 提交值取 Redux 当前值（预览已写入钳制后的结果）——用基准 + 位移
                // 重算会绕开速率钳制与长度回算，表现为"预览到上限、落库却超出"。
                const clip = sessionRef.current.clips.find((item) => item.id === origin.clipId);
                const next = {
                    clipId: origin.clipId,
                    startSec: clip?.startSec ?? origin.startSec,
                    lengthSec: clip?.lengthSec ?? origin.lengthSec,
                    clipPlaybackRate:
                        Number(clip?.clipPlaybackRate ?? origin.basePlaybackRate) || 1,
                    fadeInSec: Number(clip?.fadeInSec ?? origin.baseFadeInSec) || 0,
                    fadeOutSec: Number(clip?.fadeOutSec ?? origin.baseFadeOutSec) || 0,
                    snapOffsetSec: Math.max(
                        0,
                        Number(clip?.snapOffsetSec ?? origin.baseSnapOffsetSec) || 0,
                    ),
                };
                const persist = dispatch(setClipsStateBulkRemote({ updates: [next] })).unwrap();
                void (async () => {
                    try {
                        await persist;
                    } finally {
                        // 速率二次写回：后端可能按自身计算覆盖速率，前端计算值才是
                        // 权威（与旧实现 `reapplyRates` 同源）。
                        dispatch(
                            setClipPlaybackRate({
                                clipId: origin.clipId,
                                clipPlaybackRate: next.clipPlaybackRate,
                            }),
                        );
                        // 锁定参数线：把该轨道的曲线从旧时间范围映射到新范围
                        // （与旧实现同源，复用抽出的 `stretchLinkedParams`）。
                        if (sessionRef.current.lockParamLinesEnabled) {
                            await stretchLinkedParams(
                                origin.trackId,
                                origin.startSec,
                                origin.lengthSec,
                                next.startSec,
                                next.lengthSec,
                            );
                        }
                        // 自动交叉淡化：拉伸改变重叠 → 按落库后的关系写回自动 fade
                        //（旧实现 `shouldApplyAutoCrossfade` 覆盖 stretch，开关关闭时
                        // 只清理「已脱离重叠」的自动值）。
                        const latest = sessionRef.current;
                        if (s.autoCrossfadeEnabled) {
                            await applyAutoCrossfade(latest, origin.xfadeClipIds, dispatch, {
                                affectedSides: origin.initialCrossfadeSides,
                                editSides: origin.editSides,
                            });
                        } else {
                            await applyDetachedAutoCrossfadeClears(
                                latest,
                                origin.xfadeClipIds,
                                dispatch,
                                origin.initialCrossfadeSides,
                                origin.editSides,
                            );
                        }
                    }
                })()
                    .catch(() => undefined)
                    .finally(endKernelGestureInteraction);
                return;
            }

            if (args.cancelled) {
                // 取消：**全部参与者**一起还原（多选裁切只还原锚点会让其余 clip
                // 停在半途位置，与后端分叉）。
                batch(() => {
                    for (const [clipId, base] of origin.baseById) {
                        dispatch(moveClipStart({ clipId, startSec: base.startSec }));
                        dispatch(setClipLength({ clipId, lengthSec: base.lengthSec }));
                        dispatch(
                            setClipSourceRange({
                                clipId,
                                sourceStartSec: base.sourceStartSec,
                                sourceEndSec: base.sourceEndSec,
                            }),
                        );
                    }
                });
                // 自动交叉淡化也要回到按下时的重叠关系（按已还原的几何重算即可）。
                if (s.autoCrossfadeEnabled) {
                    previewAutoCrossfade(
                        store.getState().session,
                        origin.xfadeClipIds,
                        dispatch,
                        origin.initialCrossfadeSides,
                        origin.editSides,
                    );
                }
                endKernelGestureInteraction();
                return;
            }
            dispatch(checkpointHistory());
            // 同拖拽：提交值取 Redux 当前值（预览已写入吸附后的结果）。
            const session = sessionRef.current;
            const updates = origin.participants.flatMap((participant) => {
                const clip = session.clips.find((item) => item.id === participant.clipId);
                if (clip === undefined) return [];
                return [
                    {
                        clipId: participant.clipId,
                        startSec: clip.startSec,
                        lengthSec: clip.lengthSec,
                        // 裁切必须同时提交**源区间**：只改长度会让后端按旧源区间
                        // 重新解释内容（波形与音频都会对不上）。预览阶段已同步改过
                        // 源区间，这里取 Redux 当前值即可。
                        sourceStartSec: clip.sourceStartSec,
                        sourceEndSec: clip.sourceEndSec,
                    },
                ];
            });
            if (updates.length === 0) {
                endKernelGestureInteraction();
                return;
            }
            const persist = dispatch(setClipsStateBulkRemote({ updates })).unwrap();
            void (async () => {
                try {
                    await persist;
                } finally {
                    // 自动交叉淡化：按落库后的重叠关系写回自动 fade（开关关闭时只清理
                    // 「已脱离重叠」的自动值，保证分离后手动 fade 能恢复显示）。
                    const latest = sessionRef.current;
                    if (s.autoCrossfadeEnabled) {
                        await applyAutoCrossfade(latest, origin.xfadeClipIds, dispatch, {
                            affectedSides: origin.initialCrossfadeSides,
                            editSides: origin.editSides,
                        });
                    } else {
                        await applyDetachedAutoCrossfadeClears(
                            latest,
                            origin.xfadeClipIds,
                            dispatch,
                            origin.initialCrossfadeSides,
                            origin.editSides,
                        );
                    }
                }
            })()
                .catch(() => undefined)
                .finally(endKernelGestureInteraction);
        },
        [dispatch, endKernelGestureInteraction, sessionRef, s.autoCrossfadeEnabled],
    );

    /** 内核淡变角：按下时的原始值（用于回滚）。 */
    const kernelFadeOriginRef = React.useRef<{
        clipId: string;
        fadeInSec: number;
        fadeOutSec: number;
        /**
         * 本次淡变作用于的全部 clip（多选集合）。
         *
         * 旧实现 `useEditDrag` 的 `supportsGroupExpansion` **排除 fade**——
         * 淡变拖拽不展开编组，只作用于选中的 clip。
         */
        participants: KernelEditParticipant[];
        /** 各参与者的按下时基准（回滚与批量换算用）。 */
        baseById: Map<
            string,
            {
                fadeInSec: number;
                fadeOutSec: number;
                lengthSec: number;
                /** 曲率与形状：曲率拖拽会改它们，取消必须能还原。 */
                fadeInDir: number;
                fadeOutDir: number;
                fadeInShape: number;
                fadeOutShape: number;
            }
        >;
        /**
         * 各参与者按下时的**自动交叉淡化**长度（取消时一并还原）。
         *
         * 预览期会把拖拽侧的自动值清零（手动 fade 必须赢过自动值，见预览处的说明），
         * 取消就必须把它放回去——否则用户按 Esc 之后，那条自动交叉淡化会永久消失。
         */
        autoBaseById: Map<string, { autoFadeInSec: number; autoFadeOutSec: number }>;
        /** 自动交叉淡化：受影响集合 / 编辑前重叠关系 / 可调整侧（与裁切同源）。 */
        xfadeClipIds: string[];
        initialCrossfadeSides: ReturnType<typeof computeInitialCrossfadeSides>;
        editSides: Record<string, { fadeIn: boolean; fadeOut: boolean }>;
    } | null>(null);

    /**
     * 内核 clip 悬停 → 发布浮标内容（旧实现各 `data-tooltip` 的等价物）。
     *
     * 与淡变浮标共用**同一个锚点**（内核容器）：旧实现把这些文案挂在各自的 DOM
     * 元素上，内核自绘后那些元素不存在，因此必须由命中结果反推"指针下是什么控件"。
     * 两条通道互斥（宿主保证）：淡变控件命中时走富内容通道，其余走这里。
     *
     * 文案与取法逐条对齐旧实现：
     * - 名称区 → MIDI 前辍 + 显示名，或源文件路径（`clip.sourcePath`）；
     * - 静音 → 按当前状态给「静音 / 取消静音」；
     * - 锁链 → 按该组当前是否被禁用给「启用 / 禁用编组」；
     * - 速率 / 增益徽标 → 各自的静态提示（数值由双击后的输入框给出）；
     * - 共振峰 → 面板标题；
     * - 增益旋钮 → 带**实时数值**的提示（拖动中带增量，见增益预览）；
     * - SnapOffset 手柄 → 拖动调整提示。
     *
     * @param args 命中信息；不在任何 clip 上时为 null。
     * @returns 无返回值。
     */
    const handleKernelClipHover = React.useCallback(
        (
            args: {
                clipId: string;
                region: ClipHitRegion;
                headerControl: ClipHeaderControl | null;
            } | null,
        ) => {
            const anchor =
                typeof document === "undefined"
                    ? null
                    : (document.querySelector(
                          "[data-hs-fade-tooltip-anchor]",
                      ) as HTMLElement | null);
            if (anchor === null) return;
            if (args === null) {
                publishFadeRichTooltip(anchor, null);
                return;
            }
            const session = sessionRef.current;
            const clip = session.clips.find((item) => item.id === args.clipId);
            if (clip === undefined) {
                publishFadeRichTooltip(anchor, null);
                return;
            }
            const displayName = clipDisplayName(clip);
            let text: string | null = null;
            switch (args.headerControl) {
                case "name":
                    text =
                        clip.midiNoteCount != null
                            ? `${t("clip_type_midi_prefix")} ${displayName}`
                            : (clip.sourcePath ?? displayName);
                    break;
                case "mute":
                    text = clip.muted ? t("clip_unmute") : t("clip_mute");
                    break;
                case "chain": {
                    const groupId = clip.groupId;
                    if (groupId != null && groupId !== "") {
                        const disabled = session.disabledGroupIds.includes(groupId);
                        text = disabled ? t("enable_group") : t("disable_group");
                    }
                    break;
                }
                case "rate-label":
                    text = t("clip_badge_rate_tip");
                    break;
                case "gain-label":
                    text = t("clip_badge_gain_tip");
                    break;
                case "formant":
                    text = t("clip_formant_title");
                    break;
                case "gain-knob":
                    text = t("gain_value_tooltip").replace(
                        "{gain}",
                        formatGainDbValue(Math.min(12, Math.max(-12, gainToDb(clip.gain)))),
                    );
                    break;
                default:
                    // SnapOffset 手柄没有 header 控件，用分区识别。
                    if (args.region === "snap-offset-handle") {
                        text = t("clip_snap_offset");
                    }
                    break;
            }
            publishFadeRichTooltip(anchor, text);
        },
        [sessionRef, t],
    );

    /** 内核淡变角预览：只改对应一侧的淡变长度（另一侧保持不变）。 */
    const handleKernelFadePreview = React.useCallback(
        (args: {
            clipId: string;
            side: "in" | "out";
            fadeSec: number;
            deltaSec: number;
            modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
            curveEnv: {
                clientY: number;
                envTopClientY: number;
                bodyHeightPx: number;
                pointerSec: number;
            };
        }) => {
            if (kernelFadeOriginRef.current?.clipId !== args.clipId) {
                const session = sessionRef.current;
                const clip = session.clips.find((item) => item.id === args.clipId);
                if (clip === undefined) return;
                // 不展开编组（旧实现 supportsGroupExpansion 排除 fade）。
                const participants = resolveKernelEditParticipants({
                    anchorClipId: clip.id,
                    multiSelectedClipIds,
                    clips: session.clips,
                    trackIds: session.tracks.map((track) => track.id),
                    ignoreGrouping: session.ignoreGrouping,
                    disabledGroupIds: session.disabledGroupIds,
                    expandGroups: false,
                });
                const baseById = new Map<
                    string,
                    {
                        fadeInSec: number;
                        fadeOutSec: number;
                        lengthSec: number;
                        fadeInDir: number;
                        fadeOutDir: number;
                        fadeInShape: number;
                        fadeOutShape: number;
                    }
                >();
                const autoBaseById = new Map<
                    string,
                    { autoFadeInSec: number; autoFadeOutSec: number }
                >();
                for (const participant of participants) {
                    const item = session.clips.find(
                        (candidate) => candidate.id === participant.clipId,
                    );
                    if (item === undefined) continue;
                    autoBaseById.set(participant.clipId, {
                        autoFadeInSec: Number(item.autoFadeInSec) || 0,
                        autoFadeOutSec: Number(item.autoFadeOutSec) || 0,
                    });
                    baseById.set(participant.clipId, {
                        fadeInSec: Number(item.fadeInSec) || 0,
                        fadeOutSec: Number(item.fadeOutSec) || 0,
                        lengthSec: Math.max(0, Number(item.lengthSec) || 0),
                        fadeInDir: Number(item.fadeInDir) || 0,
                        fadeOutDir: Number(item.fadeOutDir) || 0,
                        fadeInShape: Number(item.fadeInShape) || 0,
                        fadeOutShape: Number(item.fadeOutShape) || 0,
                    });
                }
                // 自动交叉淡化：受影响集合 = 参与者；可调整侧 = 本次拖拽的那一侧
                // （拖淡入只允许自动调整 fadeIn）。
                const xfadeClipIds = participants.map((participant) => participant.clipId);
                const editSides: Record<string, { fadeIn: boolean; fadeOut: boolean }> = {};
                for (const id of xfadeClipIds) {
                    editSides[id] =
                        args.side === "in"
                            ? { fadeIn: true, fadeOut: false }
                            : { fadeIn: false, fadeOut: true };
                }
                kernelFadeOriginRef.current = {
                    clipId: clip.id,
                    fadeInSec: clip.fadeInSec,
                    fadeOutSec: clip.fadeOutSec,
                    participants,
                    baseById,
                    autoBaseById,
                    xfadeClipIds,
                    initialCrossfadeSides: computeInitialCrossfadeSides(
                        session.clips,
                        xfadeClipIds,
                    ),
                    editSides,
                };
                // 首个真实位移帧 = 手势开始：上交互锁（见 helper 说明）。
                beginKernelGestureInteraction();
            }
            const origin = kernelFadeOriginRef.current;
            if (origin === null) return;

            // ── 曲率拖拽：`modifier.fadeCurvatureDrag`（默认 Alt）按住时改**曲率** ──
            // 与旧实现 `useEditDrag` 的 fade 分支同一套：把指针投影到曲线族上取最近
            // 点，解出该侧的新 `dir`。按帧判定修饰键 → 长度/曲率可以无缝互切。
            //
            // 曲率只作用于**锚点 clip 的该侧**（旧实现明确："曲率只作用于当前 clip
            // 的该侧：指针 Y 必须映射到该 clip 自己的 gain=1 基线"——各行 body 几何
            // 不同，无法跨 clip 共用同一指针 Y）。
            if (isModifierActive(fadeCurvatureKb, args.modifiers)) {
                const clip = sessionRef.current.clips.find((item) => item.id === args.clipId);
                if (clip === undefined) return;
                const clipStart = Number(clip.startSec) || 0;
                const clipLen = Math.max(0, Number(clip.lengthSec) || 0);
                const widthSec =
                    args.side === "in"
                        ? effectiveFadeSec(clip.fadeInSec, clip.autoFadeInSec)
                        : effectiveFadeSec(clip.fadeOutSec, clip.autoFadeOutSec);
                const leftSec = args.side === "in" ? clipStart : clipStart + clipLen - widthSec;
                const pt = resolveCurvePointer(
                    args.curveEnv,
                    { leftSec, widthSec },
                    args.curveEnv.pointerSec,
                    args.curveEnv.clientY,
                );
                if (pt === null) return;
                const shape = resolveCurvatureEditBase(
                    (args.side === "in" ? clip.fadeInShape : clip.fadeOutShape) ?? 0,
                ).shape;
                const baseDir = (args.side === "in" ? clip.fadeInDir : clip.fadeOutDir) ?? 0;
                const nextDir = solveNearestCurveDir({
                    shape,
                    dir: baseDir,
                    mode: args.side,
                    pointerX01: pt.t,
                    pointerY01: pt.gain,
                    aspectYOverX: args.curveEnv.bodyHeightPx / Math.max(1, widthSec * pxPerSec),
                }).dir;
                dispatch(
                    args.side === "in"
                        ? setClipFades({ clipId: args.clipId, fadeInDir: nextDir })
                        : setClipFades({ clipId: args.clipId, fadeOutDir: nextDir }),
                );
                return;
            }

            // 多选批量：**同一个淡变长度值**应用到全部参与者（各自按自身长度钳制），
            // 与旧实现 `applyBulkFadeValue` 同源（不是"同一增量"）。
            const target = args.side === "in" ? "fadeInSec" : "fadeOutSec";
            const updates = applyBulkFadeValue({
                clipIds: origin.participants.map((participant) => participant.clipId),
                clipsById: new Map(
                    [...origin.baseById].map(([clipId, base]) => [
                        clipId,
                        { lengthSec: base.lengthSec },
                    ]),
                ),
                target,
                nextValue: args.fadeSec,
            });
            batch(() => {
                for (const update of updates) dispatch(setClipFades(update));
            });
            // 手动拖拽淡变 = 用户手动 fade：**该侧的自动交叉淡化必须清零**。
            //
            // 旧实现 `useEditDrag` 的 fade 分支正是这样做的（拖拽期
            // `setClipAutoFades({autoFade*: 0})` + 节流 remote 写 0），而**不是**
            // 去跑 `previewAutoCrossfade`。原因是绘制端按「自动 > 0 时自动赢」取值：
            // 若让自动值在拖拽中继续按重叠量重算，用户刚拖出来的手动长度会被立即
            // 覆盖回自动值——表现为「拖这条包络线完全不动」。
            batch(() => {
                for (const participant of origin.participants) {
                    dispatch(
                        args.side === "in"
                            ? setClipAutoFades({ clipId: participant.clipId, autoFadeInSec: 0 })
                            : setClipAutoFades({ clipId: participant.clipId, autoFadeOutSec: 0 }),
                    );
                }
            });
        },
        [beginKernelGestureInteraction, dispatch, multiSelectedClipIds, sessionRef],
    );

    /** 内核淡变角收尾：提交或回滚（取消时两侧一起还原）。 */
    const handleKernelFadeCommit = React.useCallback(
        (args: { clipId: string; side: "in" | "out"; fadeSec: number; cancelled: boolean }) => {
            const origin = kernelFadeOriginRef.current;
            kernelFadeOriginRef.current = null;
            if (origin === null) return;
            if (args.cancelled) {
                // 取消：全部参与者一起还原（多选淡变只还原锚点会让其余 clip 留在半途）。
                // **包括自动交叉淡化**：预览期把它清零了（手动值必须赢），取消就必须
                // 放回去，否则按一次 Esc 会让那条自动交叉淡化永久消失。
                batch(() => {
                    for (const [clipId, base] of origin.baseById) {
                        dispatch(
                            setClipFades({
                                clipId,
                                fadeInSec: base.fadeInSec,
                                fadeOutSec: base.fadeOutSec,
                                fadeInDir: base.fadeInDir,
                                fadeOutDir: base.fadeOutDir,
                                fadeInShape: base.fadeInShape,
                                fadeOutShape: base.fadeOutShape,
                            }),
                        );
                    }
                    for (const [clipId, auto] of origin.autoBaseById) {
                        dispatch(
                            setClipAutoFades({
                                clipId,
                                autoFadeInSec: auto.autoFadeInSec,
                                autoFadeOutSec: auto.autoFadeOutSec,
                            }),
                        );
                    }
                });
                endKernelGestureInteraction();
                return;
            }
            dispatch(checkpointHistory());
            // 提交值取 Redux 当前值（预览已写入按各自长度钳制后的结果）。
            //
            // 只提交**手动长度确实被本手势改动**的参与者，并把该侧自动交叉淡化
            // 一起写 0（自动 → 手动的转换）——与旧实现 `useEditDrag` 的 fade 分支
            // 同一规则（它按 `lengthEdited` 过滤后逐 clip 清 auto）。
            const session = sessionRef.current;
            const updates: Array<{
                clipId: string;
                fadeInSec?: number;
                fadeOutSec?: number;
                autoFadeInSec?: number;
                autoFadeOutSec?: number;
                fadeInDir?: number;
                fadeOutDir?: number;
            }> = [];
            for (const participant of origin.participants) {
                const clip = session.clips.find((item) => item.id === participant.clipId);
                if (clip === undefined) continue;
                const base = origin.baseById.get(participant.clipId);
                if (args.side === "in") {
                    const manual = Number(clip.fadeInSec) || 0;
                    const lengthEdited = Math.abs(manual - (base?.fadeInSec ?? 0)) > 1e-9;
                    const nextDir = Number(clip.fadeInDir) || 0;
                    const dirEdited = Math.abs(nextDir - (base?.fadeInDir ?? 0)) > 1e-9;
                    // 长度没改、曲率也没改 → 本次手势对该 clip 无写入。
                    if (!lengthEdited && !dirEdited) continue;
                    updates.push({
                        clipId: participant.clipId,
                        // 长度只在真的被改动时写（纯曲率拖拽必须保留原有长度与自动值）。
                        ...(lengthEdited ? { fadeInSec: manual, autoFadeInSec: 0 } : {}),
                        // 曲率/形状的落盘：缺了它们，bulk 回灌会把拖拽期的修改丢掉
                        //（旧实现同一处理——它总是带上 dir / shape）。
                        fadeInDir: nextDir,
                    });
                } else {
                    const manual = Number(clip.fadeOutSec) || 0;
                    const lengthEdited = Math.abs(manual - (base?.fadeOutSec ?? 0)) > 1e-9;
                    const nextDir = Number(clip.fadeOutDir) || 0;
                    const dirEdited = Math.abs(nextDir - (base?.fadeOutDir ?? 0)) > 1e-9;
                    if (!lengthEdited && !dirEdited) continue;
                    updates.push({
                        clipId: participant.clipId,
                        ...(lengthEdited ? { fadeOutSec: manual, autoFadeOutSec: 0 } : {}),
                        fadeOutDir: nextDir,
                    });
                }
            }
            if (updates.length === 0) {
                endKernelGestureInteraction();
                return;
            }
            // 淡变手势**不做** `applyAutoCrossfade` 收尾：手动 fade 拖拽的结果就是
            // 手动值 + 该侧自动清零（旧实现同样不在这里重算自动交叉淡化——重算会
            // 立刻把手动值覆盖回自动值，等于拖拽无效）。
            void dispatch(setClipsStateBulkRemote({ updates }))
                .unwrap()
                .catch(() => undefined)
                .finally(endKernelGestureInteraction);
        },
        [dispatch, endKernelGestureInteraction, sessionRef],
    );

    /** 内核框选：拖动前的选择快照（合并与回滚的基准）。 */
    const kernelBoxSelectOriginRef = React.useRef<string[] | null>(null);
    /** 框选期间「已同步为焦点 clip」的去重（避免每帧重复一次同样的后端请求）。 */
    const kernelBoxSelectSingleRef = React.useRef<string | null>(null);
    const multiSelectedIdsRef = React.useRef<string[]>([]);
    // eslint-disable-next-line react-hooks/refs -- 选择镜像：框选手势需在回调里读最新值
    multiSelectedIdsRef.current = multiSelectedClipIds;

    /**
     * 内核框选预览：复用既有合并语义。
     *
     * 合并规则（是否保留原有选择、主修饰键切换）由
     * `computeTimelineRectSelection` 决定——两处各写一份会让「按修饰键框选」
     * 的行为在两种渲染模式下分叉。
     */
    const handleKernelBoxSelectPreview = React.useCallback(
        (args: { clipIds: readonly string[]; additive: boolean }) => {
            if (kernelBoxSelectOriginRef.current === null) {
                // 拖动前的基线 = 多选集合（非空时），否则退化为**单选**目标
                // ——旧实现 `useTimelineSelectionRect` 同一口径。只看多选集合
                // 会让"单选了 A，再框选一片空白"时 A 的取消/保留判定出错。
                const multi = multiSelectedIdsRef.current;
                const single = sessionRef.current.selectedClipId;
                kernelBoxSelectOriginRef.current =
                    multi.length > 0 ? [...multi] : single ? [single] : [];
            }
            const selected = computeTimelineRectSelection({
                selectionBeforeDrag: kernelBoxSelectOriginRef.current,
                selectedInRect: [...args.clipIds],
                primaryModifierPressedAtStart: args.additive,
            });
            setMultiSelectedClipIds(selected);
            // 框内只剩一个 clip 时，顺手把它同步为**焦点 clip**（后端 `selected_clip`
            // 落库）——旧实现 `onSingleSelect` 同源。焦点 clip 决定参数编辑器的编辑
            // 目标，缺了这一步框选出单个 clip 后参数编辑器还停在旧目标上。
            //
            // 去重（同一目标不重复请求）是纯优化：旧实现每帧都发，行为不可见。
            if (selected.length === 1) {
                if (kernelBoxSelectSingleRef.current !== selected[0]) {
                    kernelBoxSelectSingleRef.current = selected[0];
                    void dispatch(selectClipRemote(selected[0]));
                }
            } else {
                kernelBoxSelectSingleRef.current = null;
            }
        },
        [dispatch, setMultiSelectedClipIds],
    );

    /** 内核框选收尾：取消时恢复拖动前的选择。 */
    const handleKernelBoxSelectCommit = React.useCallback(
        (args: { clipIds: readonly string[]; additive: boolean; cancelled: boolean }) => {
            const origin = kernelBoxSelectOriginRef.current;
            kernelBoxSelectOriginRef.current = null;
            if (args.cancelled) {
                if (origin !== null) setMultiSelectedClipIds(origin);
                return;
            }
            kernelBoxSelectSingleRef.current = null;
            // 预览已写入最终选择；这里只补「框内为空且非叠加」应清空选择的语义。
            if (args.clipIds.length === 0 && !args.additive) {
                setMultiSelectedClipIds([]);
            }
        },
        [setMultiSelectedClipIds],
    );

    /**
     * 内核右键菜单：复用既有分支（clip 菜单 / 轨道区菜单）。
     *
     * 与旧实现的差别只在**命中来源**：旧实现用 `trackIdFromClientY` +
     * `beatFromClientX`（依赖原生 scroller 的 scrollLeft），内核直接给命中的
     * 轨道与 clip 列表，菜单分支本身不变。
     */
    const handleKernelContextMenu = React.useCallback(
        (args: {
            clientX: number;
            clientY: number;
            clipIds: readonly string[];
            trackId: string | null;
            sec: number;
        }) => {
            setContextMenu(null);
            setTrackAreaMenu(null);
            if (args.trackId === null) return;
            if (args.clipIds.length > 0) {
                // 最上面那个（startSec 最大）作为主目标，其余作为"重叠选择"入口——
                // 与旧实现取 `clipsHere[clipsHere.length - 1]` 一致。
                setContextMenu({
                    x: args.clientX,
                    y: args.clientY,
                    clipId: args.clipIds[args.clipIds.length - 1],
                    overlappingClipIds: args.clipIds.length > 1 ? [...args.clipIds] : undefined,
                });
                return;
            }
            if (sessionRef.current.selectedTrackId !== args.trackId) {
                // 轨道区空白右键 = "把这条轨道设为当前轨道"（随后弹粘贴/建轨菜单）：
                // 只切焦点，不得让后端把全局记住的 `selected_clip_id` 恢复回来。
                // 后端的选中记忆是**全工程唯一**的（不是每轨一份，见
                // `state.rs::select_track`），恢复出来的 clip 完全可能属于另一条
                // 轨道，从而在"本地取消选中"之后被异步复活。契约与
                // `handleKernelSeek` 的空白点击同源（019e93ed）。
                void dispatch(
                    selectTrackRemote({ trackId: args.trackId, applySelectedClip: false }),
                );
            }
            setTrackAreaMenu({
                x: args.clientX,
                y: args.clientY,
                trackId: args.trackId,
                timeSec: args.sec,
            });
        },
        [dispatch, sessionRef, setContextMenu, setTrackAreaMenu],
    );

    /**
     * 内核淡变专属右键菜单：转发到全局总线（菜单宿主挂在面板上）。
     *
     * 复用旧实现的 `requestOpenFadeContextMenu`——载荷类型与语义完全一致，因此
     * 曲率滑块 / 形状 / 长度 / 重置等菜单项在内核模式下与旧实现同源。
     * 通用 clip 菜单要显式关掉（旧实现里两者互斥：右键落点只属于其中一方）。
     */
    const handleKernelFadeContextMenu = React.useCallback(
        (request: FadeContextMenuRequest) => {
            setContextMenu(null);
            setTrackAreaMenu(null);
            requestOpenFadeContextMenu(request);
        },
        [setContextMenu, setTrackAreaMenu],
    );

    /**
     * 内核淡变悬停浮标：发布到 AppTooltip 的富内容注册表。
     *
     * 【为什么在面板拼装内容】提示文案（形状名 i18n、长度按时间轴显示设置格式化、
     * 内联曲线图标）是渲染无关的领域知识，且与旧实现 `FadeHitLayer` /
     * `OverlapEditLayer` 共用同一套 `buildSingleFadeInfoContent` /
     * `buildCrossfadeGripInfoContent`——内核侧重写会产生第二份文案规则。
     *
     * 【锚点元素】内核的淡变控件是自绘的、没有 DOM，宿主因此把**内核容器本身**
     * 标记为浮标载体（`data-hs-fade-tooltip-anchor`）：容器本就是指针事件的目标，
     * 复用它就不需要插入任何会吞事件的浮层。面板把内容发布到它上面即可整体复用
     * AppTooltip 的显示 / 钉住 / 随指针跟随 / 菜单抑制语义。
     *
     * 长度取**生效值**（自动交叉淡化优先于手动），与绘制端和右键菜单同一规则。
     */
    const handleKernelFadeHover = React.useCallback(
        (
            args: {
                clipId: string;
                side: "in" | "out";
                isLine: boolean;
                partnerClipId?: string;
            } | null,
        ) => {
            const anchor =
                typeof document === "undefined"
                    ? null
                    : (document.querySelector(
                          "[data-hs-fade-tooltip-anchor]",
                      ) as HTMLElement | null);
            if (anchor === null) return;
            if (args === null) {
                // 收起：content 传 null → Provider 移除该元素的内容（浮标消失）。
                publishFadeRichTooltip(anchor, null);
                return;
            }
            const clips = sessionRef.current.clips;
            const sideOf = (clipId: string, isOut: boolean) => {
                const clip = clips.find((item) => item.id === clipId);
                if (clip === undefined) return null;
                return {
                    shape: (isOut ? clip.fadeOutShape : clip.fadeInShape) ?? 0,
                    dir: (isOut ? clip.fadeOutDir : clip.fadeInDir) ?? 0,
                    lengthSec: effectiveFadeSec(
                        isOut ? clip.fadeOutSec : clip.fadeInSec,
                        isOut ? clip.autoFadeOutSec : clip.autoFadeInSec,
                    ),
                };
            };
            // 交叉点抓手 = 双列（前块淡出在前、后块淡入在后，与右键菜单列序一致）。
            if (args.partnerClipId !== undefined) {
                const earlier = sideOf(args.partnerClipId, true);
                const later = sideOf(args.clipId, false);
                if (earlier === null || later === null) {
                    publishFadeRichTooltip(anchor, null);
                    return;
                }
                publishFadeRichTooltip(
                    anchor,
                    buildCrossfadeGripInfoContent({
                        earlier,
                        later,
                        formatCtx: fadeLengthFormatCtx,
                        t: t as unknown as FadeLabelLookup,
                    }),
                );
                return;
            }
            const isOut = args.side === "out";
            const side = sideOf(args.clipId, isOut);
            if (side === null) {
                publishFadeRichTooltip(anchor, null);
                return;
            }
            publishFadeRichTooltip(
                anchor,
                buildSingleFadeInfoContent({
                    isOut,
                    ...side,
                    formatCtx: fadeLengthFormatCtx,
                    t: t as unknown as FadeLabelLookup,
                }),
            );
        },
        [fadeLengthFormatCtx, sessionRef, t],
    );

    /**
     * 内核单击 inactive take lane：切换活跃 Take（复用旧实现的提交入口）。
     *
     * 与旧实现同源：暂停 / 停止时还会把播放光标带到点击位置（播放中不打断当前
     * 播放位置），切换本身走 `setClipActiveTakeRemote`（含后端落库与撤销步）。
     *
     * 特殊说明：本回调**确实要移动播放头**，因此同样遵循「乐观写 + seek 成对派发」
     * 契约（`setplayheadSec` 必须先于 `seekPlayhead`）——只派发 seek 时后端原样
     * 回显请求值，`seekPlayhead.fulfilled` 的采纳分支不命中，store 的
     * `playheadSec` 不会变（同 `handleKernelSeek` 的说明）。
     */
    const handleKernelActivateTake = React.useCallback(
        (clipId: string, takeId: string, sec: number) => {
            if (!timelineRuntimeIsPlaying) {
                dispatch(setplayheadSec(sec));
                void dispatch(seekPlayhead(sec));
            }
            activateTrackLaneTake(clipId, takeId);
        },
        [activateTrackLaneTake, dispatch, timelineRuntimeIsPlaying],
    );

    /**
     * 内核双击 clip：请求参数编辑器按 clip 起止范围创建选区。
     *
     * 与旧实现（`ClipItem` 的双击分支）同源：关闭右键菜单 → 派发
     * `hifi:editOp/selectClipParamRange`，交互焦点随之切到参数编辑器侧。
     * 用 window 事件而不是直接调 store：参数编辑器在另一棵子树里监听它，
     * 这条契约与渲染模式无关（内核 / 旧实现共用同一入口）。
     */
    const handleKernelDoubleClickClip = React.useCallback(
        (clipId: string, mode: "replace" | "toggle" = "replace") => {
            clearContextMenu();
            window.dispatchEvent(
                new CustomEvent("hifi:editOp", {
                    // mode 缺省 replace（不传即旧行为）；按住
                    // `modifier.clipRangeToParamSelection`（默认 Alt）双击时内核
                    // 传 "toggle"，由参数编辑器并入 / 挖掉该块范围。
                    detail: { op: "selectClipParamRange", clipId, mode },
                }),
            );
        },
        [clearContextMenu],
    );

    /**
     * 内核静音切换：复用旧实现的处理器（内部含乐观更新、分组联动与远端提交）。
     */
    const handleKernelToggleClipMute = React.useCallback(
        (clipId: string, nextMuted: boolean) => {
            toggleTrackLaneClipMuted(clipId, nextMuted);
        },
        [toggleTrackLaneClipMuted],
    );

    /**
     * 内核打开共振峰工具窗口：锚点用内核给的指针屏幕坐标。
     *
     * 旧实现取按钮右缘 +12 / 上缘（见 `ClipFormantButton`）；内核模式下指针就在
     * 徽标上，直接用指针位置等价且更简单（浮窗自身会钳制到视口内）。
     */
    const handleKernelOpenClipFormant = React.useCallback(
        (clipId: string, screenX: number, screenY: number) => {
            dispatch(
                openClipFormantToolWindow({
                    clipId,
                    anchor: { x: Math.round(screenX + 12), y: Math.round(screenY) },
                }),
            );
        },
        [dispatch],
    );

    /**
     * 内核速率高级编辑（右键速率标签）：复用既有 `ClipRateEditorDialog`。
     *
     * 该对话框渲染在内核开关之外（见文件末尾），两种渲染模式共用，此前只是缺入口。
     */
    const handleKernelRateBadgeMenu = React.useCallback(
        (clipId: string, screenX: number, screenY: number) => {
            setRateEditorClipId(clipId);
            setRateEditorPosition({ x: screenX, y: screenY });
        },
        [],
    );

    /**
     * 内核音量旋钮拖拽：按下时的基准（参与集合 + 各自增益 + 精细调整轴向状态）。
     *
     * 特殊说明：参与集合是**多选集合**，不展开编组——旧实现 `useEditDrag` 的
     * `supportsGroupExpansion` 明确排除 gain（增益拖拽只作用于选中的 clip）。
     */
    const kernelGainOriginRef = React.useRef<{
        clipId: string;
        clipIds: string[];
        baseGainById: Map<string, number>;
        fineAxis: FineAxisDragState;
    } | null>(null);

    /**
     * 内核音量旋钮预览：竖直位移 → dB → 逐 clip 钳制 ±12dB 写乐观值。
     *
     * 与旧实现同源：`deltaDb = 位移 × CLIP_GAIN_DRAG_DB_PER_PX`（向上拖 = 增益变大），
     * 精细调整修饰键经 `advanceFineAxisDrag` 减速，钳制与换算复用 `applyBulkGainDeltaDb`。
     */
    const handleKernelGainDragPreview = React.useCallback(
        (args: {
            clipId: string;
            deltaYPx: number;
            modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
        }) => {
            if (kernelGainOriginRef.current?.clipId !== args.clipId) {
                const session = sessionRef.current;
                const participants = resolveKernelEditParticipants({
                    anchorClipId: args.clipId,
                    multiSelectedClipIds,
                    clips: session.clips,
                    trackIds: session.tracks.map((track) => track.id),
                    ignoreGrouping: session.ignoreGrouping,
                    disabledGroupIds: session.disabledGroupIds,
                    // 增益拖拽不展开编组（旧实现 supportsGroupExpansion 排除 gain）。
                    expandGroups: false,
                });
                if (participants.length === 0) return;
                kernelGainOriginRef.current = {
                    clipId: args.clipId,
                    clipIds: participants.map((participant) => participant.clipId),
                    baseGainById: new Map(
                        participants.map((participant) => {
                            const clip = session.clips.find(
                                (item) => item.id === participant.clipId,
                            );
                            return [participant.clipId, Number(clip?.gain ?? 1) || 1] as const;
                        }),
                    ),
                    // 轴向状态以「相对按下点的位移」为 raw：状态内部只比较增量，
                    // 因此不需要绝对 clientY。
                    fineAxis: { raw: 0, adjusted: 0, fineActive: false },
                };
                // 首个真实位移帧 = 手势开始：上交互锁（见 helper 说明）。
                beginKernelGestureInteraction();
            }
            const origin = kernelGainOriginRef.current;
            if (origin === null) return;
            const adjustedY = advanceFineAxisDrag(
                origin.fineAxis,
                args.deltaYPx,
                isModifierActive(paramFineAdjustKb, args.modifiers),
            );
            const updates = applyBulkGainDeltaDb({
                clipIds: origin.clipIds,
                clipsById: new Map(
                    [...origin.baseGainById].map(([clipId, gain]) => [clipId, { gain }]),
                ),
                deltaDb: -adjustedY * CLIP_GAIN_DRAG_DB_PER_PX,
                minDb: -12,
                maxDb: 12,
            });
            batch(() => {
                for (const update of updates) dispatch(setClipGain(update));
            });
            // 拖动中的实时浮标（旧实现 `ClipHeader` 的 `gainTooltip` 拖动变体）：
            // 普通悬停只显示当前值，拖动时还要显示**本次拖动的增量**。内容逐帧
            // 更新——浮标的锚点仍是内核容器，AppTooltip 自己跟随指针。
            const anchor =
                typeof document === "undefined"
                    ? null
                    : (document.querySelector(
                          "[data-hs-fade-tooltip-anchor]",
                      ) as HTMLElement | null);
            if (anchor !== null) {
                const baseGain = origin.baseGainById.get(args.clipId);
                const currentGain =
                    sessionRef.current.clips.find((item) => item.id === args.clipId)?.gain ??
                    baseGain ??
                    1;
                const clamped = Math.min(12, Math.max(-12, gainToDb(currentGain)));
                publishFadeRichTooltip(
                    anchor,
                    baseGain === undefined
                        ? t("gain_value_tooltip").replace("{gain}", formatGainDbValue(clamped))
                        : t("gain_value_tooltip_drag")
                              .replace("{gain}", formatGainDbValue(clamped))
                              .replace("{delta}", formatGainDbValue(clamped - gainToDb(baseGain))),
                );
            }
        },
        [
            beginKernelGestureInteraction,
            dispatch,
            multiSelectedClipIds,
            paramFineAdjustKb,
            sessionRef,
            t,
        ],
    );

    /**
     * 内核音量旋钮收尾：提交或回滚。
     *
     * 特殊说明：未越起手阈值的单击（`changed = false`）**不写后端**——旧实现同样
     * 只在真正拖动后才落库（单击不产生 undo 步）。
     */
    const handleKernelGainDragCommit = React.useCallback(
        (args: { clipId: string; changed: boolean; cancelled: boolean }) => {
            const origin = kernelGainOriginRef.current;
            kernelGainOriginRef.current = null;
            // 未越起手阈值的单击（`changed = false`）没有远程写入：交互锁直接释放。
            if (origin === null || !args.changed) {
                endKernelGestureInteraction();
                return;
            }
            if (args.cancelled) {
                batch(() => {
                    for (const [clipId, gain] of origin.baseGainById) {
                        dispatch(setClipGain({ clipId, gain }));
                    }
                });
                endKernelGestureInteraction();
                return;
            }
            dispatch(checkpointHistory());
            // 提交值取 Redux 当前值（预览已写入钳制后的结果）——用基准 + 位移重算
            // 会绕开钳制，表现为"预览到 ±12dB 上限、落库却超出"。
            const session = sessionRef.current;
            void dispatch(
                setClipsStateBulkRemote({
                    updates: origin.clipIds.map((clipId) => ({
                        clipId,
                        gain:
                            Number(session.clips.find((item) => item.id === clipId)?.gain ?? 1) ||
                            1,
                    })),
                }),
            )
                .unwrap()
                .catch(() => undefined)
                .finally(endKernelGestureInteraction);
        },
        [dispatch, endKernelGestureInteraction, sessionRef],
    );

    /** 内核双击旋钮 → 恢复 0 dB（复用旧实现的增益提交入口，含乐观更新与落库）。 */
    const handleKernelGainReset = React.useCallback(
        (clipId: string) => {
            commitTrackLaneGain(clipId, 0);
        },
        [commitTrackLaneGain],
    );

    /**
     * 内核态行内编辑状态（重命名 / 增益 / 速率）。
     *
     * 值的格式化与解析都留在面板（领域知识：增益是 dB、速率有 `x` / `%` 前缀）；
     * 内核视图只负责把输入框定位到 clip header 上。
     */
    const [kernelInlineEdit, setKernelInlineEdit] = React.useState<{
        clipId: string;
        field: "name" | "gain" | "rate";
        initialValue: string;
        inputMode: "text" | "decimal";
    } | null>(null);

    /** 内核双击名称区 → 进入重命名。 */
    const handleKernelRenameClipStart = React.useCallback((clipId: string) => {
        const clip = sessionRef.current.clips.find((item) => item.id === clipId);
        if (clip === undefined) return;
        setKernelInlineEdit({
            clipId,
            field: "name",
            // 初值必须是**活动 Take 的名字**（`activeClipTakeName`），不是容器
            // clip 的 `name`。
            //
            // 提交方（`commitTrackLaneRename` → `renameClipTakeRemote`）在多 Take
            // clip 上写的是**当前 Take 的名字**：预填容器名会让"双击 + 直接回车"
            // 把 Take 名**改成容器名**，而用户什么都没输入。旧实现 `ClipHeader`
            // 的 `editTakeName` 同源。
            initialValue: activeClipTakeName(clip),
            inputMode: "text",
        });
    }, []);

    /**
     * 内核单击增益 / 速率标签 → 进入行内编辑。
     *
     * 初值取法与旧实现 `ClipHeader` 完全一致：速率用 `formatEditNumber(playbackRate)`、
     * 增益用钳制到 ±12dB 后的 `formatEditNumber`（编辑态必须保留精度，不能用展示级取整）。
     */
    const handleKernelBadgeEditStart = React.useCallback(
        (clipId: string, field: "gain" | "rate") => {
            const clip = sessionRef.current.clips.find((item) => item.id === clipId);
            if (clip === undefined) return;
            const initialValue =
                field === "rate"
                    ? formatEditNumber(clip.playbackRate)
                    : formatEditNumber(Math.min(12, Math.max(-12, gainToDb(clip.gain))));
            setKernelInlineEdit({ clipId, field, initialValue, inputMode: "decimal" });
        },
        [],
    );

    /**
     * 行内编辑的提交 / 取消（引用随编辑目标变化）。
     *
     * 解析失败一律按「取消」处理：旧实现同样在解析失败时静默放弃，而不是写入
     * 一个兜底值——后者会让用户的一次误输入变成一次真实的状态变更。
     */
    const kernelInlineEditProp = React.useMemo(() => {
        if (kernelInlineEdit === null) return null;
        const { clipId, field, initialValue, inputMode } = kernelInlineEdit;
        return {
            clipId,
            field,
            initialValue,
            inputMode,
            onCommit: (raw: string): void => {
                setKernelInlineEdit(null);
                if (field === "name") {
                    const trimmed = raw.trim();
                    // 空输入 = 保持原名（旧实现 `ClipHeader` 的
                    // `finalName = trimmed.length > 0 ? trimmed : editTakeName`）。
                    // 直接 return 也等价——但必须先确认**预填值就是提交目标**
                    // （多 Take clip 的预填是活动 Take 名，见
                    // `handleKernelRenameClipStart`），否则"回车不变名"会变成
                    // "回车把 Take 名改成容器名"。
                    if (trimmed.length === 0 || trimmed === initialValue) return;
                    commitTrackLaneRename(clipId, trimmed);
                    return;
                }
                if (field === "rate") {
                    const parsed = parsePlaybackRateInput(raw);
                    if (parsed == null) return;
                    commitTrackLaneRate(clipId, { rate: parsed });
                    return;
                }
                const parsed = Number.parseFloat(raw);
                if (!Number.isFinite(parsed)) return;
                commitTrackLaneGain(clipId, parsed);
            },
            onCancel: (): void => setKernelInlineEdit(null),
        };
    }, [kernelInlineEdit, commitTrackLaneRename, commitTrackLaneRate, commitTrackLaneGain]);

    /** 内核交叉点抓手：拖拽起点的两侧几何（换算位移与取消回滚）。 */
    const kernelCrossfadeOriginRef = React.useRef<{
        /** 两侧按下时的完整几何快照（含源窗口 / 循环 / 倒放 / 媒体时长）。 */
        earlier: CrossfadeGripClipBase;
        later: CrossfadeGripClipBase;
        /** 按下时的重叠长度（反向模式的淡变缩放基准）与两侧生效淡变。 */
        baseOverlapSec: number;
        earlierFadeOutSec: number;
        laterFadeInSec: number;
        earlierFadeOutAuto: boolean;
        laterFadeInAuto: boolean;
        /** 各 clip 按下时的全部可回滚字段（取消路径用）。 */
        baseById: Map<
            string,
            {
                startSec: number;
                lengthSec: number;
                sourceStartSec: number;
                sourceEndSec: number;
                fadeInSec: number;
                fadeOutSec: number;
                autoFadeInSec: number;
                autoFadeOutSec: number;
            }
        >;
    } | null>(null);

    /**
     * 内核交叉点抓手预览。
     *
     * 几何换算整体交给共享纯函数 `computeCrossfadeGrip`（与旧实现
     * `useEditDrag` 的 `crossfade_edges` 分支同一套语义）：
     * - **同向**（默认）：两侧边缘同向平移，重叠不变；
     * - **反向**（按住 `modifier.crossfadeGrip`，默认 Ctrl/⌘）：两侧相向移动，
     *   重叠变化，两侧淡变按「新重叠 / 原重叠」比例缩放。
     *
     * 每一侧都要按「循环 / 倒放 / 非 Loop 派生窗口」改写**源窗口**——只改长度会让
     * 波形与音频对不上（旧实现的抓手拖拽同样维护源窗口）。
     */
    const handleKernelCrossfadeGripPreview = React.useCallback(
        (args: {
            earlierClipId: string;
            laterClipId: string;
            deltaSec: number;
            modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
        }) => {
            const clips = sessionRef.current.clips;
            const earlier = clips.find((item) => item.id === args.earlierClipId);
            const later = clips.find((item) => item.id === args.laterClipId);
            if (earlier === undefined || later === undefined) return;
            if (kernelCrossfadeOriginRef.current?.earlier.id !== args.earlierClipId) {
                const toBase = (clip: typeof earlier): CrossfadeGripClipBase => {
                    const view = readSlipClip(clip);
                    return {
                        id: clip.id,
                        startSec: Number(clip.startSec) || 0,
                        lengthSec: view.lengthSec,
                        sourceStartSec: view.sourceStartSec,
                        sourceEndSec: view.sourceEndSec,
                        playbackRate: view.playbackRate,
                        loopEnabled: view.loopEnabled,
                        reversed: view.reversed,
                        mediaDurationSec: view.contentDurSec ?? 0,
                    };
                };
                const baseById = new Map<
                    string,
                    {
                        startSec: number;
                        lengthSec: number;
                        sourceStartSec: number;
                        sourceEndSec: number;
                        fadeInSec: number;
                        fadeOutSec: number;
                        autoFadeInSec: number;
                        autoFadeOutSec: number;
                    }
                >();
                for (const clip of [earlier, later]) {
                    baseById.set(clip.id, {
                        startSec: Number(clip.startSec) || 0,
                        lengthSec: Math.max(0, Number(clip.lengthSec) || 0),
                        sourceStartSec: Number(clip.sourceStartSec ?? 0) || 0,
                        sourceEndSec: Number(clip.sourceEndSec ?? 0) || 0,
                        fadeInSec: Number(clip.fadeInSec) || 0,
                        fadeOutSec: Number(clip.fadeOutSec) || 0,
                        autoFadeInSec: Number(clip.autoFadeInSec) || 0,
                        autoFadeOutSec: Number(clip.autoFadeOutSec) || 0,
                    });
                }
                kernelCrossfadeOriginRef.current = {
                    earlier: toBase(earlier),
                    later: toBase(later),
                    baseOverlapSec: Math.max(
                        0,
                        (Number(earlier.startSec) || 0) +
                            (Number(earlier.lengthSec) || 0) -
                            (Number(later.startSec) || 0),
                    ),
                    // 缩放基准取**生效**淡变（自动交叉淡化 > 0 时它赢）——用户看到的
                    // 是那条包络线，按它缩放才与画面一致。
                    earlierFadeOutSec: effectiveFadeSec(earlier.fadeOutSec, earlier.autoFadeOutSec),
                    laterFadeInSec: effectiveFadeSec(later.fadeInSec, later.autoFadeInSec),
                    earlierFadeOutAuto: Number(earlier.autoFadeOutSec ?? 0) > 0,
                    laterFadeInAuto: Number(later.autoFadeInSec ?? 0) > 0,
                    baseById,
                };
                // 首个真实位移帧 = 手势开始：上交互锁（见 helper 说明）。
                beginKernelGestureInteraction();
            }
            const origin = kernelCrossfadeOriginRef.current;
            if (origin === null) return;
            const result = computeCrossfadeGrip({
                earlier: origin.earlier,
                later: origin.later,
                deltaSec: args.deltaSec,
                // 反向模式：与旧实现同源（`modifier.crossfadeGrip`）。
                opposite: isModifierActive(crossfadeGripKb, args.modifiers),
                baseOverlapSec: origin.baseOverlapSec,
                earlierFadeOutSec: origin.earlierFadeOutSec,
                laterFadeInSec: origin.laterFadeInSec,
                earlierFadeOutAuto: origin.earlierFadeOutAuto,
                laterFadeInAuto: origin.laterFadeInAuto,
            });
            if (result === null) return;
            batch(() => {
                dispatch(
                    setClipLength({
                        clipId: result.earlier.clipId,
                        lengthSec: result.earlier.lengthSec,
                    }),
                );
                if (result.earlier.sourceStartSec !== undefined) {
                    dispatch(
                        setClipSourceRange({
                            clipId: result.earlier.clipId,
                            sourceStartSec: result.earlier.sourceStartSec,
                        }),
                    );
                }
                if (result.earlier.sourceEndSec !== undefined) {
                    dispatch(
                        setClipSourceRange({
                            clipId: result.earlier.clipId,
                            sourceEndSec: result.earlier.sourceEndSec,
                        }),
                    );
                }
                dispatch(
                    moveClipStart({ clipId: result.later.clipId, startSec: result.later.startSec }),
                );
                dispatch(
                    setClipLength({
                        clipId: result.later.clipId,
                        lengthSec: result.later.lengthSec,
                    }),
                );
                if (result.later.sourceStartSec !== undefined) {
                    dispatch(
                        setClipSourceRange({
                            clipId: result.later.clipId,
                            sourceStartSec: result.later.sourceStartSec,
                        }),
                    );
                }
                if (result.later.sourceEndSec !== undefined) {
                    dispatch(
                        setClipSourceRange({
                            clipId: result.later.clipId,
                            sourceEndSec: result.later.sourceEndSec,
                        }),
                    );
                }
                // 反向模式：两侧淡变按新重叠比例缩放（自动值写 auto、手动值写手动）。
                for (const fade of result.fades) dispatch(setClipFades(fade));
            });
        },
        [beginKernelGestureInteraction, crossfadeGripKb, dispatch, sessionRef],
    );

    /**
     * 内核交叉点抓手收尾：取消则回滚两侧，否则提交。
     *
     * 与拖拽 / trim 同源——**提交值取 Redux 当前值**（预览已写入钳制后的结果），
     * 用 origin + 位移重算会绕开钳制，表现为「松手后跳回越界位置」。
     */
    const handleKernelCrossfadeGripCommit = React.useCallback(
        (args: {
            earlierClipId: string;
            laterClipId: string;
            deltaSec: number;
            cancelled: boolean;
        }) => {
            const origin = kernelCrossfadeOriginRef.current;
            kernelCrossfadeOriginRef.current = null;
            if (origin === null) return;
            if (args.cancelled) {
                // 取消：两侧**全部字段**一起还原——预览改过源窗口与淡变（反向模式），
                // 只还原起点 / 长度会让 Redux 停在半途、与后端分叉。
                batch(() => {
                    for (const [clipId, base] of origin.baseById) {
                        dispatch(moveClipStart({ clipId, startSec: base.startSec }));
                        dispatch(setClipLength({ clipId, lengthSec: base.lengthSec }));
                        dispatch(
                            setClipSourceRange({
                                clipId,
                                sourceStartSec: base.sourceStartSec,
                                sourceEndSec: base.sourceEndSec,
                            }),
                        );
                        dispatch(
                            setClipFades({
                                clipId,
                                fadeInSec: base.fadeInSec,
                                fadeOutSec: base.fadeOutSec,
                            }),
                        );
                        dispatch(
                            setClipAutoFades({
                                clipId,
                                autoFadeInSec: base.autoFadeInSec,
                                autoFadeOutSec: base.autoFadeOutSec,
                            }),
                        );
                    }
                });
                endKernelGestureInteraction();
                return;
            }
            const clips = sessionRef.current.clips;
            const earlier = clips.find((item) => item.id === origin.earlier.id);
            const later = clips.find((item) => item.id === origin.later.id);
            if (earlier === undefined || later === undefined) {
                endKernelGestureInteraction();
                return;
            }
            dispatch(checkpointHistory());
            // 源窗口必须一起提交：只写长度会让后端按旧源区间重新解释内容
            // （波形与音频都对不上）——与裁切提交同一约束。
            void dispatch(
                setClipsStateBulkRemote({
                    updates: [
                        {
                            clipId: earlier.id,
                            lengthSec: earlier.lengthSec,
                            sourceStartSec: earlier.sourceStartSec,
                            sourceEndSec: earlier.sourceEndSec,
                        },
                        {
                            clipId: later.id,
                            startSec: later.startSec,
                            lengthSec: later.lengthSec,
                            sourceStartSec: later.sourceStartSec,
                            sourceEndSec: later.sourceEndSec,
                        },
                    ],
                }),
            )
                .unwrap()
                .catch(() => undefined)
                .finally(endKernelGestureInteraction);
        },
        [dispatch, endKernelGestureInteraction, sessionRef],
    );

    /** 内核交互回调集合（引用稳定：内核创建时取一次）。 */
    const kernelInteractions = React.useMemo(
        () => ({
            onSeek: handleKernelSeek,
            onSeekTo: handleKernelSeekTo,
            onSelectClip: handleKernelSelectClip,
            onDoubleClickClip: handleKernelDoubleClickClip,
            onToggleClipMute: handleKernelToggleClipMute,
            onToggleGroupDisabled: handleToggleGroupDisabled,
            onOpenClipFormant: handleKernelOpenClipFormant,
            onRateBadgeMenu: handleKernelRateBadgeMenu,
            onRenameClipStart: handleKernelRenameClipStart,
            onBadgeEditStart: handleKernelBadgeEditStart,
            // 拦截经 ref 中转：面板的接管实现（音高拖拽）在渲染后段才实例化。
            onClipPointerDownIntercept: (args: {
                clipId: string;
                clientX: number;
                clientY: number;
                pointerId: number;
                modifiers: {
                    ctrlKey: boolean;
                    shiftKey: boolean;
                    altKey: boolean;
                    metaKey: boolean;
                };
                container: HTMLElement;
            }) => kernelClipInterceptRef.current?.(args) ?? false,
            onGainDragPreview: handleKernelGainDragPreview,
            onGainDragCommit: handleKernelGainDragCommit,
            onGainReset: handleKernelGainReset,
            onCrossfadeGripPreview: handleKernelCrossfadeGripPreview,
            onCrossfadeGripCommit: handleKernelCrossfadeGripCommit,
            onFadeShapeCycle: handleFadeShapeCycleClick,
            onCrossfadeCycle: handleCrossfadeCycleClick,
            // 重置曲率走既有总线（旧实现同样经它派发）：消费者在淡变相关的 hook 里，
            // 这条契约与渲染模式无关。内核只给「哪些侧」，请求包络由这里组装。
            onResetFadeCurvature: (sides: Array<{ clipId: string; isOut: boolean }>) =>
                requestResetFadeCurvature({ sides }),
            onDragPreview: handleKernelDragPreview,
            onDragCommit: handleKernelDragCommit,
            onTrimPreview: handleKernelTrimPreview,
            onTrimCommit: handleKernelTrimCommit,
            onFadePreview: handleKernelFadePreview,
            onFadeCommit: handleKernelFadeCommit,
            onSnapOffsetPreview: handleKernelSnapOffsetPreview,
            onSnapOffsetCommit: handleKernelSnapOffsetCommit,
            onBoxSelectPreview: handleKernelBoxSelectPreview,
            onBoxSelectCommit: handleKernelBoxSelectCommit,
            onContextMenu: handleKernelContextMenu,
            onFadeContextMenu: handleKernelFadeContextMenu,
            onFadeHover: handleKernelFadeHover,
            onClipHover: handleKernelClipHover,
            onActivateTake: handleKernelActivateTake,
        }),
        [
            handleKernelSeek,
            handleKernelSeekTo,
            handleKernelSelectClip,
            handleKernelDragPreview,
            handleKernelDragCommit,
            handleKernelTrimPreview,
            handleKernelTrimCommit,
            handleKernelFadePreview,
            handleKernelFadeCommit,
            handleKernelSnapOffsetPreview,
            handleKernelSnapOffsetCommit,
            handleKernelBoxSelectPreview,
            handleKernelBoxSelectCommit,
            handleKernelContextMenu,
            handleKernelFadeContextMenu,
            handleKernelFadeHover,
            handleKernelClipHover,
            handleKernelActivateTake,
            handleKernelRateBadgeMenu,
            // 下面这些回调内联在 `kernelInteractions` 里（或经局部函数转发），
            // 原先同样漏在依赖数组外：`useMemo` 会继续持有**首帧闭包**，表现为
            // 「改了设置 / 选择后，内核手势仍按旧值执行」。依赖数组必须与
            // useMemo 体内引用的回调一一对应。
            handleKernelDoubleClickClip,
            handleKernelToggleClipMute,
            handleToggleGroupDisabled,
            handleKernelOpenClipFormant,
            handleKernelRenameClipStart,
            handleKernelBadgeEditStart,
            handleKernelGainDragPreview,
            handleKernelGainDragCommit,
            handleKernelGainReset,
            handleKernelCrossfadeGripPreview,
            handleKernelCrossfadeGripCommit,
            handleFadeShapeCycleClick,
            handleCrossfadeCycleClick,
        ],
    );

    // ── 4. 全局事件监听 ─────────────────────────────────────
    useTimelineEventHandlers({
        dispatch,
        sessionRef,
        getPlayheadSec: getVisualPlayheadSec,
        scrollRef,
        viewport: viewportAccess,
        trackListScrollRef,
        pxPerSecRef,
        viewportWidthRef,
        keyboardZoomPendingRef,
        pxPerSec,
        setPxPerSec,
        commitScrollLeftState: setScrollLeftState,
        rowHeight,
        setMultiSelectedClipIds,
        copyClips,
        cutClips,
        pasteClipsAtPlayhead,
        splitSelectedAtPlayhead,
        normalizeClips,
        groupClips,
        ungroupClips,
        contextMenu,
        trackAreaMenu,
        setContextMenu,
        setTrackAreaMenu,
        syncScrollLeft,
    });

    // ── 5. 拖拽 hooks 桥接 ──────────────────────────────────
    const formatClipPitchDragTooltip = React.useCallback(
        (cents: number) =>
            t("clip_pitch_drag_tooltip").replace("{delta}", formatPitchDragCents(cents)),
        [t],
    );
    const { startClipPitchDrag, pitchDragTooltip } = useClipPitchDrag({
        sessionRef,
        dispatch,
        fineAdjustKb: paramFineAdjustKb,
        formatDragTooltip: formatClipPitchDragTooltip,
    });

    /**
     * 内核 clip 左键按下拦截：`Alt + Shift`（`modifier.clipPitchDrag`）按住时
     * 把这次手势**整体交给旧实现的 `useClipPitchDrag`**。
     *
     * 【为什么委托而不是重写】该手势是一台带异步状态机的完整实现：先取基准参数帧、
     * 逐帧节流写后端预览、undo group 惰性开启、收尾提交 / 回滚、tooltip 发布。
     * 内核侧重写会产生第二份音高语义；委托则天然与旧实现一致（tooltip 浮层也已
     * 由面板渲染，与渲染模式无关）。
     *
     * 特殊说明：旧 hook 需要「React 合成事件」形态的入参（只用 button / pointerId /
     * clientX / clientY / preventDefault / stopPropagation / currentTarget），因此这里
     * 构造一个最小鸭子类型对象（与旧 `ClipHeader` 调用 `startEditDrag` 的手法一致）。
     */
    const handleKernelClipPointerDownIntercept = React.useCallback(
        (args: {
            clipId: string;
            clientX: number;
            clientY: number;
            pointerId: number;
            modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
            container: HTMLElement;
        }): boolean => {
            if (!isModifierActive(pitchDragKb, args.modifiers)) return false;
            const clip = sessionRef.current.clips.find((item) => item.id === args.clipId);
            if (clip === undefined) return false;
            startClipPitchDrag(
                {
                    button: 0,
                    pointerId: args.pointerId,
                    clientX: args.clientX,
                    clientY: args.clientY,
                    preventDefault: () => undefined,
                    stopPropagation: () => undefined,
                    currentTarget: args.container,
                } as unknown as React.PointerEvent<HTMLDivElement>,
                args.clipId,
            );
            return true;
        },
        [pitchDragKb, sessionRef, startClipPitchDrag],
    );
    kernelClipInterceptRef.current = handleKernelClipPointerDownIntercept;

    const clipById = useMemo(
        () => new Map(s.clips.map((clip) => [clip.id, clip] as const)),
        [s.clips],
    );

    /**
     * 轨道头左键按下：把该轨道设为当前轨道（面板 → `TrackList` 的 `onSelectTrack`）。
     *
     * 特殊说明：`applySelectedClip: false` —— 轨道头点击的**全部**意图就是换当前
     * 轨道（手册：「点击左侧轨道头切换当前轨道」），不包含"恢复某条 clip 的选中"。
     * 后端的选中记忆是**全工程唯一**的（`state.rs::select_track` 只改
     * `selected_track_id`），恢复出来的 clip 可能属于另一条轨道，会把用户刚做完的
     * "点空白取消选中"异步复活。实测（忠实 mock）：纯字符串形式下点轨道头会让
     * `selectedClipId` 从 `null` 变回 `track-1-clip-1`。
     */
    const handleSelectTrack = React.useCallback(
        (trackId: string) => {
            if (sessionRef.current.selectedTrackId === trackId) {
                return;
            }
            void dispatch(selectTrackRemote({ trackId, applySelectedClip: false }));
        },
        [dispatch, sessionRef],
    );
    const handleRemoveTrack = React.useCallback(
        (trackId: string) => {
            dispatch(removeTrackRemote(trackId));
        },
        [dispatch],
    );
    const handleMoveTrack = React.useCallback(
        (payload: { trackId: string; targetIndex: number; parentTrackId: string | null }) => {
            dispatch(
                moveTrackRemote({
                    trackId: payload.trackId,
                    targetIndex: payload.targetIndex,
                    parentTrackId: payload.parentTrackId,
                }),
            );
        },
        [dispatch],
    );
    const handleToggleTrackMute = React.useCallback(
        (trackId: string, nextMuted: boolean) => {
            dispatch(
                setTrackStateRemote({
                    trackId,
                    muted: nextMuted,
                }),
            );
        },
        [dispatch],
    );
    const handleToggleTrackSolo = React.useCallback(
        (trackId: string, nextSolo: boolean) => {
            dispatch(
                setTrackStateRemote({
                    trackId,
                    solo: nextSolo,
                }),
            );
        },
        [dispatch],
    );
    const handleToggleTrackCompose = React.useCallback(
        (trackId: string, nextComposeEnabled: boolean) => {
            dispatch(
                setTrackStateRemote({
                    trackId,
                    composeEnabled: nextComposeEnabled,
                }),
            );
        },
        [dispatch],
    );
    const handleTrackVolumeUiChange = React.useCallback(
        (trackId: string, nextVolume: number) => {
            setTrackVolumeUi((prev) => ({
                ...prev,
                [trackId]: nextVolume,
            }));
        },
        [setTrackVolumeUi],
    );
    const handleTrackVolumeCommit = React.useCallback(
        (trackId: string, nextVolume: number) => {
            dispatch(setTrackVolume({ trackId, volume: nextVolume }));
            setTrackVolumeUi((prev) => {
                const copy = { ...prev };
                delete copy[trackId];
                return copy;
            });
            dispatch(
                setTrackStateRemote({
                    trackId,
                    volume: nextVolume,
                }),
            );
        },
        [dispatch, setTrackVolumeUi],
    );
    const handleAddTrack = React.useCallback(() => {
        dispatch(addTrackRemote({}));
    }, [dispatch]);
    const handleTrackColorChange = React.useCallback(
        (trackId: string, color: string) => {
            dispatch(
                setTrackStateRemote({
                    trackId,
                    color,
                }),
            );
        },
        [dispatch],
    );
    const handleTrackAlgoChange = React.useCallback(
        (trackId: string, algo: string) => {
            dispatch(
                setTrackStateRemote({
                    trackId,
                    pitchAnalysisAlgo: algo,
                }),
            );
        },
        [dispatch],
    );
    const handleTrackNameChange = React.useCallback(
        (trackId: string, name: string) => {
            dispatch(setTrackName({ trackId, name }));
            dispatch(
                setTrackStateRemote({
                    trackId,
                    name,
                }),
            );
        },
        [dispatch],
    );
    const handleDuplicateTrack = React.useCallback(
        (trackId: string) => {
            dispatch(duplicateTrackRemote(trackId));
        },
        [dispatch],
    );
    // “复制拖动”修饰键 + 轨道头拖拽：在拖放位置克隆轨道（含子树）。
    const handleDuplicateTrackTo = React.useCallback(
        (payload: { trackId: string; targetIndex: number; parentTrackId: string | null }) => {
            dispatch(
                duplicateTrackRemote({
                    trackId: payload.trackId,
                    parentTrackId: payload.parentTrackId,
                    targetIndex: payload.targetIndex,
                }),
            );
        },
        [dispatch],
    );
    const handleCreateTrackBelow = React.useCallback(
        (trackId: string) => {
            void (async () => {
                const existingTracks = [...sessionRef.current.tracks];
                const beforeIds = new Set(existingTracks.map((track) => track.id));
                const added = (await dispatch(
                    addTrackRemote({ name: undefined, parentTrackId: null }),
                ).unwrap()) as {
                    tracks?: Array<{ id?: string }>;
                    selected_track_id?: string | null;
                };
                const nextTracks = Array.isArray(added.tracks) ? added.tracks : [];
                const createdTrackId =
                    nextTracks.find((track) => !beforeIds.has(String(track?.id)))?.id ??
                    added.selected_track_id ??
                    null;
                if (!createdTrackId) return;
                await dispatch(
                    moveTrackRemote({
                        trackId: String(createdTrackId),
                        targetIndex: getInsertBelowTargetIndex(existingTracks, trackId),
                        parentTrackId: null,
                    }),
                );
            })();
        },
        [dispatch, sessionRef],
    );
    const handleTrackListScrollTopChange = React.useCallback((scrollTop: number) => {
        // 内核模式：轨道头只是**被动镜像**，它报来的 `scrollTop` 有两种来源，
        // 且数值上无法区分，必须靠"宿主刚写过的值"来判（见 `timeline/scrollEcho`）：
        //
        // 1. **镜像回声**（绝大多数）：宿主每帧写 `trackList.scrollTop`，这次写入
        //    触发的原生 `scroll` 事件会把**滞后一帧**的值报回来。若照单全收，内核
        //    每帧被拉回一次——实测 40 步拖拽里 19 次调用**全部**是回声、每次
        //    `delta ≈ −9px`，即用户报告的"纵向拖起来卡卡的、像被吸附"。
        // 2. **真实输入**：焦点在轨道头控件上时，浏览器原生 scroll-into-view
        //    （Tab / PageDown / End）会真正改变容器位置（实测 0 → 308 / 132 / 361）。
        //    这类必须继续回灌，否则焦点导航时轨道头会与时间轴脱节。
        //
        // 【为什么旧判据失效】旧实现拿事件值与**内核当前值**比（< 0.5px 即忽略）。
        // 拖拽时内核每帧前进，而事件报的是上一帧镜像值，两者相差约 9px，因此回声
        // 被误当成用户输入收下，形成「内核 → DOM → 内核」的回退循环。
        const host = kernelHostRef.current;
        if (host != null) {
            if (
                isTrackListMirrorEcho({
                    mirroredScrollTop: host.getMirroredTrackListScrollTop(),
                    nativeScrollTop: scrollTop,
                })
            ) {
                return;
            }
            host.setScrollTop(scrollTop);
            return;
        }
        // 宿主不可用（挂载前）时不回灌：旧的原生 scroller 分支已删除——
        // `scrollRef` 无 JSX 挂载点，`scrollRef.current` 恒为 null。
    }, []);

    const activeGroupIds = useMemo(() => {
        const ids = new Set<string>();
        for (const cid of multiSelectedClipIds) {
            const gid = clipById.get(cid)?.groupId;
            if (gid && !disabledGroupIds.includes(gid)) ids.add(gid);
        }
        if (s.selectedClipId) {
            const gid = clipById.get(s.selectedClipId)?.groupId;
            if (gid && !disabledGroupIds.includes(gid)) ids.add(gid);
        }
        return ids.size > 0 ? ids : undefined;
    }, [multiSelectedClipIds, clipById, s.selectedClipId, disabledGroupIds]);
    /**
     * 内核用分组数组。
     *
     * `activeGroupIds` 是派生的 `Set`，内核侧要数组；这里转一次并保持引用稳定，
     * 避免每次渲染都产生新数组（数据镜像引用抖动会让宿主每帧判定「内容已变」）。
     */
    const kernelActiveGroupIds = React.useMemo(
        () => (activeGroupIds === undefined ? [] : Array.from(activeGroupIds)),
        [activeGroupIds],
    );
    // 主题切换 → darkMode prop 变化 → clip 体画布同帧按新主题重绘配色。
    const { mode: themeMode } = useAppTheme();
    const darkMode = themeMode === "dark";
    const timelineScrollRange = useMemo(
        () => resolveTimelineScrollRange({ contentWidth, viewportWidth }),
        [contentWidth, viewportWidth],
    );
    // ═════════════════════════════════════════════════════════
    // JSX 渲染
    // ═════════════════════════════════════════════════════════

    // scrollLeft 按 REACT_SCROLL_STEP_PX 量化提交，React 渲染期用 `timelineAxis`
    // 算出的位置最多滞后一个步长。轨道区播放头由内核自绘，其重绘请求由
    // TimelineTransportBridge 的 onFrame 逐帧发起（那里读的是视觉插值真值）；
    // 这里不需要再纠正任何 DOM。

    /**
     * 标尺节点（内核模式与旧模式共用同一实例）。
     *
     * 标尺是重交互、低频变化的 DOM 子树（刻度标签 / Tempo Map 旗帜拖拽 / 内联编辑 /
     * 右键菜单），搬进 canvas 等于整体重写且收益极低。让两种渲染模式共用它，视觉与
     * 交互天然与旧实现一致；内核只把「跟随水平滚动」的部分收敛为 rAF 内一次
     * transform 写入（见 kernel host 的 syncDom）。
     */
    const timeRulerNode = (
        // playheadSec 传提交值（而非渲染期读 ref）：视觉插值由
        // `rulerPlayheadLineRef` / `rulerPlayheadHeadRef` 命令式驱动；React 仅在
        // 该值真正变化时重写 style.left，写入的是最新提交位置而非陈旧值。
        //
        // 标尺不消费实时滚动位置：刻度与可见范围都按量化的 `rulerScrollLeft` 生成
        // （缓冲已保证覆盖视口），这样滚动期间 `TimeRulerMarks` 的 memo 不会失效，
        // 整棵刻度子树不必每帧重渲染。内核模式下水平滚动由内核在 rAF 内直接写
        // 内容层 transform（不经 React），这条量化约定依然成立。
        <TimeRuler
            scrollLeft={rulerScrollLeft}
            ticks={timelineTicks}
            pxPerSec={pxPerSec}
            viewportWidth={viewportWidth}
            playheadSec={s.playheadSec}
            playheadLineRef={rulerPlayheadLineRef}
            playheadHeadRef={rulerPlayheadHeadRef}
            contentRef={rulerContentRef}
            timeContext={timeContext}
            primaryUnit={s.primaryTimeUnit}
            secondaryUnit={s.secondaryTimeUnit}
            onPrimaryUnitChange={handlePrimaryUnitChange}
            onSecondaryUnitChange={handleSecondaryUnitChange}
            onOpenSettings={() => setTimeDisplaySettingsOpen(true)}
            onCopyPlayheadTime={() => void handleCopyPlayheadTime()}
            t={t as (key: string) => string}
            tempoMap={s.tempoMap}
            tempoMapVisible={s.tempoMapVisible}
            projectSec={dynamicProjectSec}
            grid={s.grid}
            snapEnabled={s.snapEnabled}
            timelineSnap={s.timelineSnap}
            projectScale={projectScale}
            projectScaleName={
                s.project.useCustomScale ? (s.project.customScale?.name ?? undefined) : undefined
            }
            fallbackDenominator={s.project.timeSignatureDenominator}
            customScalePresets={s.customScalePresets}
            onTempoMapChange={handleTempoMapChange}
            onTempoMapCommit={handleTempoMapCommit}
            onMouseDown={(e) => {
                if (e.button !== 0) return;
                // 水平滚动位置取自内核视口。取值的「实时性」是硬要求——拖拽期间滚动
                // 位置可能被自动滚动改变，因此每次换算都重新读，不缓存。
                // 读滚动真值：内核宿主是唯一来源（旧的原生 scroller 分支已删除：
                // `scrollRef` 无 JSX 挂载点，恒为 null）。
                const readScrollLeft = (): number | null => {
                    const host = kernelHostRef.current;
                    return host != null ? host.getViewport().scrollLeft : null;
                };
                if (readScrollLeft() === null) return;
                const ruler = e.currentTarget as HTMLDivElement;
                let moved = false;
                let lastClientX = e.clientX;
                let lastSec = 0;

                const updateAt = (clientX: number, commit: boolean): number =>
                    setPlayheadFromClientX(
                        clientX,
                        ruler.getBoundingClientRect(),
                        readScrollLeft() ?? 0,
                        commit,
                    );

                // 标尺没有其他编辑操作需要区分，按下时立即提交一次 seek。
                lastSec = updateAt(e.clientX, true);

                const onMove = (ev: MouseEvent) => {
                    moved = true;
                    lastClientX = ev.clientX;
                    lastSec = updateAt(ev.clientX, false);
                };

                // 失焦取消：切屏期间 mouseup 不送达本窗口，blur 时以
                // 最后一次已知位置收尾（提交 seek + 清吸附高亮），
                // 防止监听器泄漏：否则下次点击会被旧的 onEnd 消费。
                const finish = () => {
                    unregisterAbort();
                    window.removeEventListener("mousemove", onMove, true);
                    window.removeEventListener("mouseup", onEnd, true);
                    window.removeEventListener("mouseleave", onEnd, true);
                    if (!moved) {
                        // 未拖动的单击不会发布高亮；仍兜底清除一次，
                        // 防止此前异常中断手势的残留。
                        clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                        return;
                    }
                    lastSec = updateAt(lastClientX, false);
                    void dispatch(seekPlayhead(lastSec));
                    // 最后一步 update 仍会发布一次吸附高亮，必须在其后
                    // 清除：否则拖拽标尺后网格吸附的竖线会冻结在画面上，
                    // 且任何单击跳转都不再清理它。
                    clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                };
                const onEnd = (ev: MouseEvent) => {
                    lastClientX = ev.clientX;
                    finish();
                };
                const unregisterAbort = registerDragAbort(finish);

                window.addEventListener("mousemove", onMove, true);
                window.addEventListener("mouseup", onEnd, true);
                window.addEventListener("mouseleave", onEnd, true);
            }}
        />
    );

    const handleTimelineDragOver = (e: React.DragEvent<HTMLDivElement>) => {
        const dt = e.dataTransfer;
        const tauriPath = tauriDraggedPathRef.current;
        const hasDomFile = Boolean(dt?.files && dt.files.length > 0);
        const isTauri = Boolean((window as unknown as { __TAURI__?: unknown }).__TAURI__);
        if (!isTauri && !hasFileDrag(dt) && !hasDomFile && !tauriPath) return;
        e.preventDefault();
        const info = extractLocalFilePath(dt);
        const el = e.currentTarget as HTMLDivElement;
        const bounds = el.getBoundingClientRect();
        const beat = beatFromClientX(e.clientX, bounds, dragScrollLeftOf(el));
        const trackId = resolveTrackIdAtClientY(e.clientY);
        const path = info?.path || tauriPath || "";
        const fileName =
            info?.name ||
            (tauriPath
                ? String(tauriPath.split(/[\\/]/).pop() ?? tauriPath)
                : hasDomFile
                  ? String(dt?.files?.[0]?.name ?? "Audio")
                  : "Audio");
        const dragAction = detectExternalPathAction(path);
        if (path && dragAction !== "importAudio" && dragAction !== "importMidi") {
            setDropPreview(null);
            return;
        }
        if (dragAction === "importMidi") {
            // MIDI 文件使用默认时长显示 drop preview
            setDropPreview({
                path,
                fileName,
                trackId,
                startSec: beat,
                durationSec: 2,
            });
        } else {
            if (path) {
                ensureDropPreviewDuration(path);
            }
            setDropPreview({
                path,
                fileName,
                trackId,
                startSec: beat,
                durationSec: 0,
            });
        }
    };

    const handleTimelineDrop = (e: React.DragEvent<HTMLDivElement>) => {
        const dt = e.dataTransfer;
        const tauriPath = tauriDraggedPathRef.current;
        const lastTauriDropPath = tauriLastDropPathRef.current;
        const hasDomFile = Boolean(dt?.files && dt.files.length > 0);
        const isTauri = Boolean((window as unknown as { __TAURI__?: unknown }).__TAURI__);
        if (!isTauri && !hasFileDrag(dt) && !hasDomFile && !tauriPath) return;
        e.preventDefault();

        if (isTauri && Date.now() - (tauriDropHandledAtRef.current || 0) < 500) {
            setDropPreview(null);
            return;
        }

        const info = extractLocalFilePath(dt);
        const el = e.currentTarget as HTMLDivElement;
        const bounds = el.getBoundingClientRect();
        const beat = beatFromClientX(e.clientX, bounds, dragScrollLeftOf(el));
        const trackId = resolveTrackIdAtClientY(e.clientY);
        setDropPreview(null);
        const resolvedPath = info?.path || lastTauriDropPath || tauriPath;
        if (resolvedPath) {
            tauriDraggedPathRef.current = null;
            tauriLastDropPathRef.current = null;
            const actionKind = detectExternalPathAction(resolvedPath);
            if (actionKind === "importMidi") {
                onMidiClipPathChange(resolvedPath);
                onMidiClipStartSecChange(beat);
                onMidiClipTrackIdChange(trackId);
                onMidiClipDialogOpenChange(true);
                return;
            }
            if (actionKind && actionKind !== "importAudio") {
                emitExternalFileAction(actionKind, resolvedPath);
                return;
            }
            void dispatch(
                importAudioAtPosition({
                    audioPath: resolvedPath,
                    trackId,
                    startSec: beat,
                }),
            );
            return;
        }

        if (isTauri) {
            window.setTimeout(() => {
                const p = tauriLastDropPathRef.current || tauriDraggedPathRef.current;
                if (!p) return;
                tauriDraggedPathRef.current = null;
                tauriLastDropPathRef.current = null;
                const actionKind = detectExternalPathAction(p);
                if (actionKind === "importMidi") {
                    onMidiClipPathChange(p);
                    onMidiClipStartSecChange(beat);
                    onMidiClipTrackIdChange(trackId);
                    onMidiClipDialogOpenChange(true);
                    return;
                }
                if (actionKind && actionKind !== "importAudio") {
                    emitExternalFileAction(actionKind, p);
                    return;
                }
                void dispatch(
                    importAudioAtPosition({
                        audioPath: p,
                        trackId,
                        startSec: beat,
                    }),
                );
            }, 0);
        }

        const fallbackFile = dt.files?.[0] ?? null;
        if (fallbackFile) {
            void dispatch(
                importAudioFileAtPosition({
                    file: fallbackFile,
                    trackId,
                    startSec: beat,
                }),
            );
        }
    };

    return (
        <Profiler
            id="TimelinePanel"
            onRender={(_id, _phase, actualDuration) => {
                // dev 帧率探针：经 globalThis 挂钩上报 React 提交耗时。
                // 未启用探针时只有一次属性查找，零成本。
                (
                    globalThis as unknown as {
                        __hfsFrameProfiler?: { recordReact(ms: number): void };
                    }
                ).__hfsFrameProfiler?.recordReact(actualDuration);
            }}
        >
            <Flex
                className="h-full w-full bg-qt-graph-bg overflow-hidden"
                // 编辑表面声明：文档级 pointerdown/focusin 捕获据此把本面板
                // 解析为「timeline」表面（轨道列在其内部以 trackHeader 就近
                // 覆盖，见 focusSurface.ts / data-hs-surface）。
                data-hs-surface="timeline"
                onPointerDownCapture={(e) => {
                    // 点击轨道背景等非输入区域时，主动失焦当前聚焦的输入框，
                    // 让名称/速率/增益/轨道增益等行内编辑器走各自的 onBlur
                    // 提交路径（背景 pointerdown 会 preventDefault，焦点不
                    // 转移时 blur 不会自然触发，输入框将无法退出）。
                    //
                    // ★ 失焦延迟到 rAF：若点击落在编辑中的 Clip 内部，ClipItem
                    //   捕获阶段的 controller.commit() 必须先获胜——同步 blur
                    //   会经 onBlur 抢先提交一次，controller 再提交一次，造成
                    //   双重提交。rAF 时编辑器已卸载/失焦，重复提交自然消失。
                    //   浮层面板（data-hs-floating-menu）不参与失焦逻辑。
                    const target = e.target as HTMLElement | null;
                    if (
                        target?.closest?.(
                            "input,textarea,select,[contenteditable='true'],[data-hs-floating-menu]",
                        )
                    ) {
                        return;
                    }
                    const active = document.activeElement as HTMLElement | null;
                    if (active && active.tagName === "INPUT") {
                        requestAnimationFrame(() => {
                            if (document.activeElement === active) {
                                active.blur();
                            }
                        });
                    }
                }}
            >
                <TrackList
                    t={t}
                    tracks={s.tracks}
                    trackMeters={s.trackMeters}
                    selectedTrackId={s.selectedTrackId}
                    rowHeight={rowHeight}
                    setRowHeight={setRowHeight}
                    verticalZoomKb={verticalZoomKb}
                    paramFineAdjustKb={paramFineAdjustKb}
                    trackVolumeUi={trackVolumeUi}
                    listScrollRef={trackListScrollRef}
                    onSelectTrack={handleSelectTrack}
                    onRemoveTrack={handleRemoveTrack}
                    onMoveTrack={handleMoveTrack}
                    copyDragKb={copyDragKb}
                    onDuplicateTrackTo={handleDuplicateTrackTo}
                    onToggleMute={handleToggleTrackMute}
                    onToggleSolo={handleToggleTrackSolo}
                    onToggleCompose={handleToggleTrackCompose}
                    onVolumeUiChange={handleTrackVolumeUiChange}
                    onVolumeCommit={handleTrackVolumeCommit}
                    onAddTrack={handleAddTrack}
                    onTrackColorChange={handleTrackColorChange}
                    onAlgoChange={handleTrackAlgoChange}
                    onTrackNameChange={handleTrackNameChange}
                    onDuplicateTrack={handleDuplicateTrack}
                    onCreateTrackBelow={handleCreateTrackBelow}
                    onScrollTopChange={handleTrackListScrollTopChange}
                    headerHeight={timeRulerHeightPx(
                        Boolean(s.tempoMap && s.tempoMap.points.length > 0 && s.tempoMapVisible),
                    )}
                    bottomGutterHeightPx={horizontalScrollbarGutterPx}
                />

                {/* Timeline View (Right) */}
                <Flex direction="column" className="flex-1 relative overflow-hidden bg-qt-graph-bg">
                    {/* 新渲染内核（自绘滚动 + 单 WebGL2）渲染**轨道区**。
                        标尺（`timeRulerNode`）与左侧轨道头保留 DOM：它们重交互、低频
                        变化，搬进 canvas 等于重写 Tempo 旗帜拖拽与电平表，收益极低；
                        内核只把「跟随视口」的部分收敛为 rAF 内一次 transform / scrollTop
                        写入。轨道区（网格 / clip / 波形 / 交互）才是滚动瓶颈，由内核自绘。

                        内核是**唯一**渲染路径：旧实现（原生滚动 + Canvas2D 的
                        `TimelineScrollArea` / `TimelineSurface` / `TrackLane` 等）已随
                        阶段 2/3 删除，此处没有运行期二选一，也没有可回退的第二条路径。
                        WebGL2 不可用时内核视图会回报失败，此处改渲染可自助排障的
                        失败界面（见 `KernelUnavailableNotice` 与
                        `timeline/kernel/kernelAvailability`）——没有回退渲染路径。 */}
                    {isKernelAvailable(kernelUnavailableReason !== null) ? (
                        <>
                            {timeRulerNode}
                            <TimelineKernelView
                                rowHeight={rowHeight}
                                onRowHeightChange={setRowHeight}
                                initialPxPerSec={pxPerSec}
                                onPxPerSecChange={setPxPerSec}
                                onScrollLeftCommit={handleKernelScrollLeftCommit}
                                onScrollLeftFrame={state.syncScrollLeftFrame}
                                onViewportWidthChange={setViewportWidth}
                                onUnavailable={handleKernelUnavailable}
                                getPlayheadSec={getVisualPlayheadSec}
                                rulerContentRef={rulerContentRef}
                                trackListScrollerRef={trackListScrollRef}
                                rulerPlayheadLineRef={rulerPlayheadLineRef}
                                hostRef={kernelHostRef}
                                interactions={kernelInteractions}
                                activeGroupIds={kernelActiveGroupIds}
                                disabledGroupIds={disabledGroupIds}
                                silenceSegmentsByClipId={silencePreviewSegments ?? undefined}
                                showAllTakes={s.showAllTakes}
                                inlineEdit={kernelInlineEditProp}
                                snapHighlight={{
                                    pxPerSec,
                                    contentWidth: timelineScrollRange.paddedContentWidth,
                                    contentHeight,
                                }}
                                ghost={
                                    kernelGhost === null
                                        ? undefined
                                        : {
                                              // ghost 的轨道着色与真实 clip 同源（旧实现
                                              // 用 color-mix + 归一化轨道色）。
                                              items: kernelGhost.map((item) => {
                                                  const trackColor =
                                                      s.tracks.find((t) => t.id === item.trackId)
                                                          ?.color || undefined;
                                                  const tint =
                                                      trackColor === undefined
                                                          ? null
                                                          : normalizedTrackColorCss(
                                                                trackColor,
                                                                darkMode,
                                                            );
                                                  return {
                                                      ...item,
                                                      headerBackground:
                                                          tint === null
                                                              ? "var(--qt-clip-bg)"
                                                              : `color-mix(in oklab, var(--qt-clip-bg) 40%, ${tint} 60%)`,
                                                      bodyBackground:
                                                          tint === null
                                                              ? "var(--qt-clip-bg)"
                                                              : `color-mix(in oklab, var(--qt-clip-bg) 45%, ${tint} 55%)`,
                                                  };
                                              }),
                                              contentWidth: timelineScrollRange.paddedContentWidth,
                                              contentHeight,
                                          }
                                }
                                dropPreview={
                                    dropPreview === null || dropPreview.trackId === null
                                        ? undefined
                                        : {
                                              leftPx: Math.max(0, dropPreview.startSec * pxPerSec),
                                              widthPx: Math.max(
                                                  1,
                                                  pxPerSec *
                                                      (dropPreview.durationSec > 0
                                                          ? dropPreview.durationSec
                                                          : 1),
                                              ),
                                              trackId: dropPreview.trackId,
                                              fileName: dropPreview.fileName,
                                              contentWidth: timelineScrollRange.paddedContentWidth,
                                              contentHeight,
                                          }
                                }
                                onDragOver={handleTimelineDragOver}
                                onDrop={handleTimelineDrop}
                                // 拖出容器即清掉落点预览。旧实现（`TimelineScrollArea`
                                // 的 `onDragLeave`）同源：只看 `relatedTarget` 是否仍在
                                // 容器内——子元素之间移动也会触发 `dragleave`，不加这层
                                // 判断会把预览抖掉。缺了它，把文件拖出时间轴后预览会一直
                                // 挂着（没有任何后续事件会清它）。
                                onDragLeave={(event) => {
                                    const related = event.relatedTarget as Node | null;
                                    if (related !== null && event.currentTarget.contains(related)) {
                                        return;
                                    }
                                    setDropPreview(null);
                                }}
                                // 预览内层元素的 ref：`useTimelineDragDrop` 拖动期间
                                // 直接写它的 style 移动预览（不 setState）。不接上时
                                // 同一条轨道内移动指针预览不动（见该 prop 的说明）。
                                dropPreviewItemRef={dropPreviewRef}
                                newTrackDrop={
                                    kernelDropToNewTrack
                                        ? {
                                              // 幽灵行里的 clip 取 Redux 里的乐观位置
                                              // （预览已把参与者写到哨兵轨上，含吸附结果）。
                                              items: s.clips
                                                  .filter(
                                                      (clip) => clip.trackId === NEW_TRACK_SENTINEL,
                                                  )
                                                  .map((clip) => ({
                                                      key: clip.id,
                                                      leftPx: Math.max(0, clip.startSec * pxPerSec),
                                                      widthPx: Math.max(
                                                          1,
                                                          clip.lengthSec * pxPerSec,
                                                      ),
                                                  })),
                                              contentWidth: timelineScrollRange.paddedContentWidth,
                                          }
                                        : undefined
                                }
                            />
                        </>
                    ) : (
                        <KernelUnavailableNotice reason={kernelUnavailableReason ?? "未知原因"} />
                    )}

                    {/* 共振峰工具窗口：`fixed` 定位（视口坐标），与渲染模式无关。
                        原先它随旧子树一起被包在内核开关内，内核模式下右键
                        「共振峰变形」后窗口不出现——这里移到开关之外，两种模式共用。 */}
                    {s.clipFormantToolWindow.open && activeFormantToolClip ? (
                        <ClipFormantToolWindow
                            clip={activeFormantToolClip}
                            status={s.clipFormantStatus[activeFormantToolClip.id] ?? "ready"}
                            x={s.clipFormantToolWindow.x}
                            y={s.clipFormantToolWindow.y}
                            onCommit={commitTrackLaneFormantMorph}
                            onMove={(x, y) => dispatch(setClipFormantToolWindowPosition({ x, y }))}
                            onClose={() => dispatch(closeClipFormantToolWindow())}
                        />
                    ) : null}

                    {/* 导入模式选择菜单 */}
                    {importModeMenu && (
                        <div
                            className="fixed inset-0 z-[9999]"
                            onClick={() => setImportModeMenu(null)}
                            onContextMenu={(e) => {
                                e.preventDefault();
                                setImportModeMenu(null);
                            }}
                        >
                            <div
                                className="absolute bg-qt-panel border border-qt-border rounded shadow-lg py-1 min-w-[180px]"
                                style={{
                                    left: importModeMenu.x,
                                    top: importModeMenu.y,
                                }}
                                onClick={(e) => e.stopPropagation()}
                            >
                                <button
                                    className="w-full text-left px-3 py-1.5 text-sm text-qt-text hover:bg-qt-hover"
                                    onClick={() => {
                                        const m = importModeMenu;
                                        setImportModeMenu(null);
                                        if (m.audioPaths.length === 1) {
                                            void dispatch(
                                                importAudioAtPosition({
                                                    audioPath: m.audioPaths[0],
                                                    trackId: m.trackId,
                                                    startSec: m.startSec,
                                                }),
                                            );
                                        } else {
                                            void dispatch(
                                                importMultipleAudioAtPosition({
                                                    audioPaths: m.audioPaths,
                                                    mode: "across-time",
                                                    trackId: m.trackId,
                                                    startSec: m.startSec,
                                                }),
                                            );
                                        }
                                    }}
                                >
                                    {t("import_across_time") || "Import across time (same track)"}
                                </button>
                                <button
                                    className="w-full text-left px-3 py-1.5 text-sm text-qt-text hover:bg-qt-hover"
                                    onClick={() => {
                                        const m = importModeMenu;
                                        setImportModeMenu(null);
                                        if (m.audioPaths.length === 1) {
                                            void dispatch(
                                                importAudioAtPosition({
                                                    audioPath: m.audioPaths[0],
                                                    trackId: null,
                                                    startSec: m.startSec,
                                                }),
                                            );
                                        } else {
                                            void dispatch(
                                                importMultipleAudioAtPosition({
                                                    audioPaths: m.audioPaths,
                                                    mode: "across-tracks",
                                                    trackId: m.trackId,
                                                    startSec: m.startSec,
                                                }),
                                            );
                                        }
                                    }}
                                >
                                    {t("import_across_tracks")}
                                </button>
                                <button
                                    className="w-full text-left px-3 py-1.5 text-sm text-qt-text hover:bg-qt-hover"
                                    onClick={() => {
                                        const m = importModeMenu;
                                        setImportModeMenu(null);
                                        void dispatch(
                                            importMultipleAudioAtPosition({
                                                audioPaths: m.audioPaths,
                                                mode: "as-takes",
                                                trackId: m.trackId,
                                                startSec: m.startSec,
                                            }),
                                        );
                                    }}
                                >
                                    {t("import_as_takes")}
                                </button>
                            </div>
                        </div>
                    )}

                    {/* 工程文件（hshp/hsp）拖放操作菜单：打开工程 / 导入工程 */}
                    {projectActionMenu && (
                        <div
                            className="fixed inset-0 z-[9999]"
                            onClick={() => setProjectActionMenu(null)}
                            onContextMenu={(e) => {
                                e.preventDefault();
                                setProjectActionMenu(null);
                            }}
                        >
                            <div
                                className="absolute bg-qt-panel border border-qt-border rounded shadow-lg py-1 min-w-[180px]"
                                style={{
                                    left: projectActionMenu.x,
                                    top: projectActionMenu.y,
                                }}
                                onClick={(e) => e.stopPropagation()}
                            >
                                <button
                                    className="w-full text-left px-3 py-1.5 text-sm text-qt-text hover:bg-qt-hover"
                                    onClick={() => {
                                        const m = projectActionMenu;
                                        setProjectActionMenu(null);
                                        emitExternalFileAction("openProject", m.path);
                                    }}
                                >
                                    {t("menu_open_project")}
                                </button>
                                <button
                                    className="w-full text-left px-3 py-1.5 text-sm text-qt-text hover:bg-qt-hover"
                                    onClick={() => {
                                        const m = projectActionMenu;
                                        setProjectActionMenu(null);
                                        window.dispatchEvent(
                                            new CustomEvent("hifi:importProjectPick", {
                                                detail: { path: m.path },
                                            }),
                                        );
                                    }}
                                >
                                    {tAny("import_project_dialog_title")}
                                </button>
                            </div>
                        </div>
                    )}

                    <FadeContextMenuHost />
                    {contextMenu
                        ? (() => {
                              const ctxClip = sessionRef.current.clips.find(
                                  (c) => c.id === contextMenu.clipId,
                              );
                              if (!ctxClip) return null;

                              const selectedIds = resolveQuickExportClipIds({
                                  contextClipId: contextMenu.clipId,
                                  multiSelectedClipIds,
                              });
                              const selectedClips = sessionRef.current.clips.filter((c) =>
                                  selectedIds.includes(c.id),
                              );

                              // 右键位置的工程时间：**必须**用与渲染模式无关的视口
                              // 访问器。旧实现读原生 scroller 的 bounds + scrollLeft；
                              // 内核模式下 `scrollRef.current` 为 null，原写法会静默退化
                              // 成 `ctxClip.startSec`，于是"指针下有哪些重叠的淡变 clip"
                              // 这份候选集会与旧实现不同（菜单里少/多出条目）。
                              const _ctxBounds = viewportAccess.getRect();
                              const contextTimeSec =
                                  _ctxBounds !== null
                                      ? beatFromClientX(
                                            contextMenu.x,
                                            _ctxBounds,
                                            viewportAccess.getScrollLeft(),
                                        )
                                      : ctxClip.startSec;

                              const overlappingFadeClips = collectFadeContextClips({
                                  allClips: sessionRef.current.clips,
                                  contextClip: ctxClip,
                                  contextTimeSec,
                                  explicitOverlappingClipIds: contextMenu.overlappingClipIds,
                              });

                              const currentPlayheadSec = sessionRef.current.playheadSec;
                              const playheadInClip =
                                  currentPlayheadSec >= ctxClip.startSec &&
                                  currentPlayheadSec <= ctxClip.startSec + ctxClip.lengthSec;

                              return createPortal(
                                  <ClipContextMenu
                                      x={contextMenu.x}
                                      y={contextMenu.y}
                                      clip={ctxClip}
                                      selectedClips={selectedClips}
                                      overlappingClips={overlappingFadeClips}
                                      playheadInClip={playheadInClip}
                                      canSplitSelected={selectedClips.some((c) => {
                                          const splitSec = Math.max(
                                              0,
                                              Number(sessionRef.current.playheadSec ?? 0) || 0,
                                          );
                                          return (
                                              splitSec >= c.startSec &&
                                              splitSec <= c.startSec + c.lengthSec
                                          );
                                      })}
                                      onClose={() => setContextMenu(null)}
                                      onDelete={(ids) => {
                                          setContextMenu(null);
                                          setMultiSelectedClipIds([]);
                                          void dispatch(removeClipsRemote(ids));
                                      }}
                                      onMute={(ids, muted) => {
                                          // 批量走 bulk 通道：单次 IPC + 单个撤销步
                                          //（逐个 setClipStateRemote 会产生 N 次
                                          // IPC/N 步撤销）。乐观更新先行。
                                          for (const id of ids) {
                                              dispatch(
                                                  setClipMuted({
                                                      clipId: id,
                                                      muted,
                                                  }),
                                              );
                                          }
                                          void dispatch(
                                              setClipsStateBulkRemote({
                                                  updates: ids.map((id) => ({
                                                      clipId: id,
                                                      muted,
                                                  })),
                                                  checkpoint: true,
                                              }),
                                          );
                                      }}
                                      onRename={(clipId) => {
                                          setContextMenu(null);
                                          clipActions.setRenamingClipId(clipId);
                                      }}
                                      onCopy={(ids) => {
                                          const s = sessionRef.current;
                                          const expandedIds = expandClipIdsWithGroups(
                                              ids,
                                              s.clips,
                                              s.ignoreGrouping,
                                              s.disabledGroupIds,
                                          );
                                          void copyClips(expandedIds);
                                      }}
                                      onCut={(ids) => {
                                          const s = sessionRef.current;
                                          const expandedIds = expandClipIdsWithGroups(
                                              ids,
                                              s.clips,
                                              s.ignoreGrouping,
                                              s.disabledGroupIds,
                                          );
                                          setContextMenu(null);
                                          cutClips(expandedIds);
                                      }}
                                      onReplace={(ids) => {
                                          void replaceClipSources(ids);
                                      }}
                                      onReplaceMidi={(ids) => {
                                          if (ids.length > 0) {
                                              void openReplaceMidiForClip(ids[0]);
                                          }
                                      }}
                                      onQuickExport={(ids) => {
                                          setQuickExportDialog({
                                              open: true,
                                              clipIds: ids,
                                          });
                                      }}
                                      onSplit={(clipIds) => {
                                          setContextMenu(null);
                                          splitClipIdsAtPlayhead(clipIds);
                                      }}
                                      onGroup={(ids) => {
                                          setContextMenu(null);
                                          groupClips(ids);
                                      }}
                                      onUngroup={(ids) => {
                                          setContextMenu(null);
                                          ungroupClips(ids);
                                      }}
                                      onGlue={(ids) => {
                                          setContextMenu(null);
                                          if (ids.length >= 2) {
                                              void dispatch(glueClipsRemote(ids));
                                              setMultiSelectedClipIds([]);
                                          }
                                      }}
                                      onConvertToPitchRef={(ids) => {
                                          setContextMenu(null);
                                          void dispatch(convertClipsToPitchReferenceRemote(ids));
                                          setMultiSelectedClipIds([]);
                                      }}
                                      onUpdatePitchRef={(ids) => {
                                          setContextMenu(null);
                                          void dispatch(updatePitchReferenceRemote(ids));
                                          setMultiSelectedClipIds([]);
                                      }}
                                      onExportMidi={(ids) => {
                                          setContextMenu(null);
                                          void handleExportMidi(ids);
                                      }}
                                      onAddToParamSelection={(ids) => {
                                          // 批量入口：把所选音频块的时间范围并入
                                          // 参数编辑器选区（隐藏菜单由 PianoRollPanel
                                          // 消费，按当前根轨道组过滤，见 handleEditOp）。
                                          setContextMenu(null);
                                          window.dispatchEvent(
                                              new CustomEvent("hifi:editOp", {
                                                  detail: {
                                                      op: "addClipsToParamSelection",
                                                      clipIds: ids,
                                                  },
                                              }),
                                          );
                                      }}
                                      onRemoveFromParamSelection={(ids) => {
                                          setContextMenu(null);
                                          window.dispatchEvent(
                                              new CustomEvent("hifi:editOp", {
                                                  detail: {
                                                      op: "removeClipsFromParamSelection",
                                                      clipIds: ids,
                                                  },
                                              }),
                                          );
                                      }}
                                      onFadeShapeChange={(clipId, target, shape) => {
                                          // 切换形状必须重置曲率（REAPER 语义：各形状的
                                          // 默认曲率由形状自身定义，见 reaperFade 的
                                          // DEFAULT_FADE_DIR_BY_SHAPE / defaultFadeDirFor）。
                                          const dir = defaultFadeDirFor(shape, target === "out");
                                          dispatch(
                                              setClipFades({
                                                  clipId,
                                                  ...(target === "in"
                                                      ? {
                                                            fadeInShape: shape,
                                                            fadeInDir: dir,
                                                        }
                                                      : {
                                                            fadeOutShape: shape,
                                                            fadeOutDir: dir,
                                                        }),
                                              }),
                                          );
                                          void dispatch(
                                              setClipStateRemote({
                                                  clipId,
                                                  ...(target === "in"
                                                      ? {
                                                            fadeInShape: shape,
                                                            fadeInDir: dir,
                                                        }
                                                      : {
                                                            fadeOutShape: shape,
                                                            fadeOutDir: dir,
                                                        }),
                                              }),
                                          );
                                      }}
                                      onSilenceDetection={(ids) => setSilenceDialogIds(ids)}
                                      onNormalize={normalizeClips}
                                      onEditRate={openRateBadgeMenu}
                                      onToggleReverse={(ids, reversed) => {
                                          // 批量走 bulk 通道：单次 IPC + 单个撤销步
                                          //（逐个 setClipStateRemote 会产生 N 次 IPC/N 步撤销）。
                                          void dispatch(
                                              setClipsStateBulkRemote({
                                                  updates: ids.map((id) => ({
                                                      clipId: id,
                                                      reversed,
                                                  })),
                                                  checkpoint: true,
                                              }),
                                          );
                                      }}
                                      onToggleLoop={(ids, loopEnabled) => {
                                          const session = sessionRef.current;
                                          const updates = ids.map((id) => {
                                              const clip = session.clips.find(
                                                  (entry) => entry.id === id,
                                              );
                                              const update: {
                                                  clipId: string;
                                                  loopEnabled: boolean;
                                                  sourceEndSec?: number;
                                              } = { clipId: id, loopEnabled };
                                              // 关闭循环的瞬间：非 Loop 正放 Clip 按
                                              // 派生窗口模型归一 source_end
                                              //（= 起点+长度×速率）。循环期间锚点被
                                              // 回绕/窗口被保持，直接关掉会把陈旧
                                              // 窗口带入非 Loop 状态 —— 静音区冻结、
                                              // 音频错位都源于此。
                                              // 与后端 clip_effective_source_end_sec
                                              // 一致：不按 midiNoteData 排除 —— 音高
                                              // 参考块等无源媒体 Clip 的音高曲线
                                              //（trim_and_resample_midi）同样使用派生
                                              // 窗口，存储值也必须一并归一。
                                              if (!loopEnabled && clip && !clip.reversed) {
                                                  const rate =
                                                      Number(clip.playbackRate) > 0
                                                          ? Number(clip.playbackRate)
                                                          : 1;
                                                  update.sourceEndSec =
                                                      (Number(clip.sourceStartSec) || 0) +
                                                      Math.max(0, clip.lengthSec) * rate;
                                              }
                                              return update;
                                          });
                                          void dispatch(
                                              setClipsStateBulkRemote({
                                                  updates,
                                                  checkpoint: true,
                                              }),
                                          );
                                      }}
                                  />,
                                  document.body,
                              );
                          })()
                        : null}

                    {trackAreaMenu
                        ? createPortal(
                              <TrackAreaContextMenu
                                  x={trackAreaMenu.x}
                                  y={trackAreaMenu.y}
                                  canPaste={clipboardAvailable}
                                  canSplit={(multiSelectedClipIds.length > 0
                                      ? multiSelectedClipIds
                                      : sessionRef.current.selectedClipId
                                        ? [sessionRef.current.selectedClipId]
                                        : []
                                  ).some((id) => {
                                      const clip = sessionRef.current.clips.find(
                                          (c) => c.id === id,
                                      );
                                      if (!clip) return false;
                                      const splitSec = Math.max(
                                          0,
                                          Number(sessionRef.current.playheadSec ?? 0) || 0,
                                      );
                                      return (
                                          splitSec >= clip.startSec &&
                                          splitSec <= clip.startSec + clip.lengthSec
                                      );
                                  })}
                                  canCloseGaps={sessionRef.current.clips.some(
                                      (c) =>
                                          c.trackId === trackAreaMenu.trackId &&
                                          c.startSec > trackAreaMenu.timeSec + 1e-9,
                                  )}
                                  onCloseGaps={() => {
                                      void dispatch(
                                          closeTrackGapsRemote({
                                              trackId: trackAreaMenu.trackId,
                                              fromSec: trackAreaMenu.timeSec,
                                          }),
                                      );
                                  }}
                                  onPaste={pasteClipsAtPlayhead}
                                  onSplit={splitSelectedAtPlayhead}
                                  onClose={() => setTrackAreaMenu(null)}
                              />,
                              document.body,
                          )
                        : null}

                    <SilenceDetectionDialog
                        open={silenceDialogIds != null}
                        clipIds={silenceDialogIds ?? []}
                        onOpenChange={(open) => {
                            if (!open) setSilenceDialogIds(null);
                        }}
                    />
                    <QuickClipExportDialog
                        open={quickExportDialog.open}
                        clipIds={quickExportDialog.clipIds}
                        onOpenChange={(open) =>
                            setQuickExportDialog((prev) =>
                                open ? prev : { open: false, clipIds: [] },
                            )
                        }
                    />

                    <MidiTrackSelectDialog
                        open={midiClipDialogOpen}
                        onOpenChange={onMidiClipDialogOpenChange}
                        midiPath={midiClipPath}
                        importTarget={importTarget}
                        onImportTargetChange={onImportTargetChange}
                        clipboardGuid={midiClipClipboardGuid ?? null}
                        rootTrackComposeEnabled={midiClipRootTrackComposeEnabled}
                        onRequestEnableCompose={handleRequestEnableCompose}
                        onImportAsClip={handleMidiClipImport}
                        importPosition={importPosition}
                        onImportPositionChange={onImportPositionChange}
                        fillGaps={fillGaps}
                        onFillGapsChange={onFillGapsChange}
                        multiTrackMerge={multiTrackMerge}
                        onMultiTrackMergeChange={onMultiTrackMergeChange}
                        projectBpm={s.bpm}
                        importBpmAsProject={importBpmAsProject}
                        onImportBpmAsProjectChange={onImportBpmAsProjectChange}
                        noteBpmMode={noteBpmMode}
                        onNoteBpmModeChange={onNoteBpmModeChange}
                        specifiedBpm={specifiedBpm}
                        onSpecifiedBpmChange={onSpecifiedBpmChange}
                        closeLeadingGap={closeLeadingGap}
                        onCloseLeadingGapChange={onCloseLeadingGapChange}
                        importTempoMapEnabled={importTempoMapEnabled}
                        onImportTempoMapEnabledChange={onImportTempoMapEnabledChange}
                        importTempoMapTempo={importTempoMapTempo}
                        onImportTempoMapTempoChange={onImportTempoMapTempoChange}
                        importTempoMapTimeSignature={importTempoMapTimeSignature}
                        onImportTempoMapTimeSignatureChange={onImportTempoMapTimeSignatureChange}
                        importTempoMapKeySignature={importTempoMapKeySignature}
                        onImportTempoMapKeySignatureChange={onImportTempoMapKeySignatureChange}
                    />

                    <MidiTrackSelectDialog
                        open={replaceMidiDialog.open}
                        onOpenChange={(open) => {
                            if (!open)
                                setReplaceMidiDialog({
                                    open: false,
                                    clipId: null,
                                    midiPath: null,
                                });
                        }}
                        midiPath={replaceMidiDialog.midiPath}
                        mode="replaceMidi"
                        onImportAsClip={handleReplaceMidiImport}
                        fillGaps={fillGaps}
                        onFillGapsChange={onFillGapsChange}
                        projectBpm={s.bpm}
                        importBpmAsProject={importBpmAsProject}
                        onImportBpmAsProjectChange={onImportBpmAsProjectChange}
                        noteBpmMode={noteBpmMode}
                        onNoteBpmModeChange={onNoteBpmModeChange}
                        specifiedBpm={specifiedBpm}
                        onSpecifiedBpmChange={onSpecifiedBpmChange}
                        closeLeadingGap={closeLeadingGap}
                        onCloseLeadingGapChange={onCloseLeadingGapChange}
                    />

                    <Dialog.Root
                        open={sameSourceConfirmOpen}
                        onOpenChange={(open) => {
                            setSameSourceConfirmOpen(open);
                            if (!open && sameSourceConfirmResolverRef.current) {
                                sameSourceConfirmResolverRef.current(false);
                                sameSourceConfirmResolverRef.current = null;
                            }
                        }}
                    >
                        <Dialog.Content maxWidth="480px">
                            <Dialog.Title>{t("ctx_replace")}</Dialog.Title>
                            <Dialog.Description>
                                <Text size="2">{t("clip_replace_same_source_confirm")}</Text>
                            </Dialog.Description>
                            <Flex justify="end" gap="2" mt="4">
                                <Button
                                    variant="soft"
                                    color="gray"
                                    onClick={() => {
                                        setSameSourceConfirmOpen(false);
                                        if (sameSourceConfirmResolverRef.current) {
                                            sameSourceConfirmResolverRef.current(false);
                                            sameSourceConfirmResolverRef.current = null;
                                        }
                                    }}
                                >
                                    {t("cancel")}
                                </Button>
                                <Button
                                    onClick={() => {
                                        setSameSourceConfirmOpen(false);
                                        if (sameSourceConfirmResolverRef.current) {
                                            sameSourceConfirmResolverRef.current(true);
                                            sameSourceConfirmResolverRef.current = null;
                                        }
                                    }}
                                >
                                    {t("ok")}
                                </Button>
                            </Flex>
                        </Dialog.Content>
                    </Dialog.Root>

                    <TimelineTransportBridge
                        pxPerSecRef={pxPerSecRef}
                        rulerPlayheadLineRef={rulerPlayheadLineRef}
                        rulerPlayheadHeadRef={rulerPlayheadHeadRef}
                        viewport={viewportAccess}
                        visualPlayheadRef={visualPlayheadSecRef}
                        syncScrollLeft={syncScrollLeft}
                        autoScrollEnabled={s.autoScrollEnabled}
                        projectSec={dynamicProjectSec}
                        requestPlayheadRepaint={handleRequestPlayheadRepaint}
                    />

                    {/* 右键播放速率角标 → 高级编辑（倍率 + BPM 换算，批量应用） */}
                    <ClipRateEditorDialog
                        open={rateEditorClipId != null && rateEditorPosition != null}
                        clip={
                            rateEditorClipId
                                ? (s.clips.find((entry) => entry.id === rateEditorClipId) ?? null)
                                : null
                        }
                        tempoMap={s.tempoMap}
                        position={rateEditorPosition}
                        projectBpm={s.bpm}
                        targetCount={
                            rateEditorClipId != null &&
                            multiSelectedClipIds.length > 0 &&
                            multiSelectedSet.has(rateEditorClipId)
                                ? multiSelectedClipIds.length
                                : 1
                        }
                        formatCtx={fadeLengthFormatCtx}
                        onApply={(rate, adjustLength, durationSec) => {
                            if (rateEditorClipId != null) {
                                commitTrackLaneRate(rateEditorClipId, {
                                    rate,
                                    durationSec: durationSec ?? undefined,
                                    autoLength: adjustLength,
                                });
                            }
                        }}
                        onOpenChange={(o) => {
                            if (!o) setRateEditorClipId(null);
                        }}
                    />

                    <TimelineDisplaySettingsDialog
                        open={timeDisplaySettingsOpen}
                        onOpenChange={setTimeDisplaySettingsOpen}
                    />

                    {/* 音高拖拽悬浮 ToolTips：跟随指针展示 Clip 范围内音高变化量 */}
                    <AppTooltipBubble
                        text={pitchDragTooltip?.text ?? null}
                        position={pitchDragTooltip?.position ?? null}
                    />
                </Flex>
            </Flex>
        </Profiler>
    );
};
