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
 * 【Spike 开关】`TIMELINE_KERNEL_ENABLED`（localStorage `hifishifter.timelineKernel`）
 * 开启时，在时间轴区域额外渲染 `TimelineKernelSpikeView`（新渲染内核，覆盖式）；
 * 默认关闭，既有实现完全不受影响。
 */
import React, { useMemo, Profiler } from "react";
import { Flex, Dialog, Button, Text } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppTheme } from "../../theme/AppThemeProvider";
import { useAppSelector } from "../../app/hooks";
import { shallowEqual } from "react-redux";
import { isModifierActive, selectKeybinding } from "../../features/keybindings/keybindingsSlice";
import { resolveClipDragCopyMode } from "./timeline/hooks/clipDragCopyMode";
import { copyClipsFromDrag } from "./timeline/hooks/copyClipsFromDrag";
import { defaultFadeDirFor, FADE_PRESETS } from "./timeline/reaperFade";
import type { FadeLengthFormatContext } from "./timeline/fadeTooltipText";
import { FadeContextMenuHost } from "./timeline/FadeContextMenuHost";
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
    setSelectedClip,
    moveClipStart,
    moveClipTrack,
    checkpointHistory,
    setClipLength,
    setClipSourceRange,
    setClipSnapOffset,
    beginInteraction,
    endInteraction,
} from "../../features/session/sessionSlice";
import { beginSnapGesture, endSnapGesture } from "../../utils/timelineSnapping";
import { batch } from "react-redux";
import { moveClipsRemote } from "../../features/session/thunks/timelineThunks";
import { computeTimelineRectSelection } from "./timeline/useTimelineSelectionRect";
import { setTempoMapRemote } from "../../features/session/thunks/tempoMapThunks";

import { NEW_TRACK_SENTINEL, useClipDrag } from "./timeline/hooks/useClipDrag";
import { useEditDrag } from "./timeline/hooks/useEditDrag";
import { useSlipDrag } from "./timeline/hooks/useSlipDrag";
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
    TRACK_ADD_ROW_HEIGHT,
    TrackAreaContextMenu,
    TimelineScrollArea,
    TimelineSurface,
    TimeRuler,
    TrackLane,
    TrackList,
    detectExternalPathAction,
    extractLocalFilePath,
    formatCursorTime,
    hasFileDrag,
} from "./timeline";
import { timeRulerHeightPx } from "./timeline/rulerHeight";
import type { TimeFormatContext, TimeUnit, TimeUnitChoice } from "./timeline";
import { SnapHighlightLayer } from "./timeline/SnapHighlightLayer";
import { formatEditNumber, gainToDb } from "./timeline/math";
import { requestResetFadeCurvature } from "./timeline/fadeContextMenuBus";
import { parsePlaybackRateInput } from "./timeline/runtime/timelineCanvasStyle";
import { SNAP_HIGHLIGHT_GROUP, clearSnapHighlights } from "../../utils/snapHighlight";
import type { TempoMap } from "../../utils/tempoMap";
import { isTimelineKernelEnabled } from "./timeline/kernel/featureFlag";
import { TimelineKernelView } from "./timeline/kernel/TimelineKernelView";
import type { TimelineKernelHost } from "./timeline/kernel/host/timelineKernelHost";

/**
 * 时间轴渲染内核（Spike）开关：模块加载时读一次。
 *
 * 开启后时间轴区域由新内核（自绘滚动 + 单 WebGL2）**替换**渲染；关闭时完全走既有实现。
 * 默认值：dev 环境开启、生产关闭（见 `timeline/kernel/featureFlag`）；切换需刷新页面
 * （dev 环境的 PERF 悬浮面板有切换按钮，切换后自动刷新）。
 */
const TIMELINE_KERNEL_ENABLED = isTimelineKernelEnabled();
import type { ScaleLike } from "../../utils/musicalScales";
import { TimelineDisplaySettingsDialog } from "./TimelineDisplaySettingsDialog";
import { resolveTimelineScrollRange } from "./timeline/runtime/timelineScrollRange";
import { applyNativeScrollLeft } from "./timeline/runtime/nativeScrollApply";

// ── 拆分出的 hooks ──────────────────────────────────────────
import { useTimelineState } from "./timeline/hooks/useTimelineState";
import { useTimelineDragDrop } from "./timeline/hooks/useTimelineDragDrop";
import { useTimelineClipActions } from "./timeline/hooks/useTimelineClipActions";
import { useTimelineEventHandlers } from "./timeline/hooks/useTimelineEventHandlers";
import { useSnapOffsetDrag } from "./timeline/hooks/useSnapOffsetDrag";
import { expandClipIdsWithGroups } from "./timeline/hooks/useGroupExpansion";
import { useVisualPlayhead } from "../../hooks/useVisualPlayhead";
import { ClipRateEditorDialog } from "./timeline/ClipRateEditorDialog";
import {
    computeAutoFollowScrollLeft,
    computeFocusCursorScrollLeft,
} from "../../utils/autoFollowScroll";
import { readDevicePixelRatio, snapToDevicePx } from "../../utils/devicePixelLine";
import { buildSparseClipRenderModel } from "./timeline/runtime/timelineCanvasModel";
import { buildTimelineRenderModel } from "./timeline/runtime/timelineRenderModel";
import { computeLeadingOverlapSecByClipId } from "./timeline/TrackLane";
import { createTimelineAxis } from "./timeline/runtime/timelineAxis";
import { resolveQuickExportClipIds } from "./timeline/quickExportSelection";
import type { ClipFormantMorph } from "../../features/session/sessionTypes";
import { ClipFormantToolWindow } from "./timeline/clip/ClipFormantToolWindow";
import type { ClipRenameClickCandidate } from "./timeline/clip/ClipHeader";

const TimelineTransportBridge = React.memo(function TimelineTransportBridge(props: {
    pxPerSecRef: React.MutableRefObject<number>;
    playheadRef: React.MutableRefObject<HTMLDivElement | null>;
    rulerPlayheadLineRef: React.MutableRefObject<HTMLDivElement | null>;
    rulerPlayheadHeadRef: React.MutableRefObject<HTMLDivElement | null>;
    scrollRef: React.MutableRefObject<HTMLDivElement | null>;
    /** 接收每帧视觉插值播放头（秒），供缩放锚点等命令式读取（与绘制同源）。 */
    visualPlayheadRef: React.MutableRefObject<number>;
    syncScrollLeft: (next: number) => void;
    autoScrollEnabled: boolean;
    projectSec: number;
}) {
    const {
        pxPerSecRef,
        playheadRef,
        rulerPlayheadLineRef,
        rulerPlayheadHeadRef,
        scrollRef,
        visualPlayheadRef,
        syncScrollLeft,
        autoScrollEnabled,
        projectSec,
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
                // 同步共享 ref：缩放锚点与提交后纠正读取的必须是与绘制同源的
                // 插值播放头，否则播放中缩放会以滞后的 store 值锚定造成跳变。
                visualPlayheadRef.current = visualPlayheadSec;
                const playheadLeftPx = visualPlayheadSec * pxPerSecRef.current;

                // 自动滚动先行：syncScrollLeft 内部会用 Redux 同步播放头（滞后于
                // 视觉插值）重写播放头位置 —— 若先定位播放头再滚动，播放头每帧
                // 会在"视觉位置"与"同步位置"之间跳动（自动滚屏抽搐的根因）。
                // 滚动先行、播放头定位收尾，最终写入获胜。
                if (autoScrollEnabled && transport.isPlaying) {
                    const scroller = scrollRef.current;
                    if (scroller) {
                        const next = computeAutoFollowScrollLeft({
                            playheadSec: visualPlayheadSec,
                            pxPerSec: pxPerSecRef.current,
                            viewportWidth: scroller.clientWidth,
                            contentWidth: projectSec * pxPerSecRef.current,
                        });
                        if (Math.abs(scroller.scrollLeft - next) > 0.5) {
                            // 写后回读浏览器实际接受的偏移再广播：跟随滚动接近
                            // 工程右端时请求值可能被钳制，画布层必须与原生 DOM
                            // 层使用同一偏移。
                            const applied = applyNativeScrollLeft(scroller, next);
                            syncScrollLeft(applied);
                        }
                    }
                }

                // 播放头定位（在自动滚动之后，用最新 scrollLeft + 视觉插值）。
                // 写入前吸附到设备像素边界（readDevicePixelRatio 每帧现读，
                // 浏览器缩放/跨屏后下一帧自愈）：分数 DPR 下不吸附的落点相位
                // 随播放连续变化，1/2 物理像素交替 —— 即"播放时粗细不一"。
                // 与 React 渲染侧（TimelineSurface / TimeRulerPlayhead）同一
                // 吸附函数，命令式与声明式两条路径逐设备像素一致。
                const dpr = readDevicePixelRatio();
                const scroller = scrollRef.current;
                const screenLeft = playheadLeftPx - (scroller?.scrollLeft ?? 0);
                if (playheadRef.current) {
                    playheadRef.current.style.left = `${snapToDevicePx(screenLeft, dpr)}px`;
                }
                if (rulerPlayheadLineRef.current) {
                    rulerPlayheadLineRef.current.style.left = `${snapToDevicePx(playheadLeftPx, dpr)}px`;
                }
                if (rulerPlayheadHeadRef.current) {
                    rulerPlayheadHeadRef.current.style.left = `${snapToDevicePx(playheadLeftPx, dpr)}px`;
                }
            },
            [
                autoScrollEnabled,
                pxPerSecRef,
                playheadRef,
                rulerPlayheadHeadRef,
                rulerPlayheadLineRef,
                scrollRef,
                syncScrollLeft,
                transport.isPlaying,
                projectSec,
                visualPlayheadRef,
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
    // 双击名称的第一次点击会把播放头移动到点击位置，第二次点击可能落在播放头线上。
    // 记录名称区域的第一次点击，让播放头在短时间内收到同位置点击时转而进入重命名。
    const renameClickCandidateRef = React.useRef<ClipRenameClickCandidate | null>(null);
    const registerRenameClickCandidate = React.useCallback(
        (candidate: ClipRenameClickCandidate | null) => {
            renameClickCandidateRef.current = candidate;
        },
        [],
    );
    const [timelineScrollTop, setTimelineScrollTop] = React.useState(0);
    // ── 竖直滚动的 React 提交量化（与水平 scrollLeft 同一套思路）──────
    // onScroll 每帧直写 state 会让 TimelinePanel 整树重渲染（探针实测
    // react p50 20ms，是竖直拖拽掉帧的根因）。React 里的 scrollTop 只服务
    // 窗口化与裁剪模型，竖直 overscan 有 4 行缓冲，滞后半个缓冲以内绝对安全；
    // sticky 画布层走视口总线命令式更新，不受滞后影响。
    const scrollTopRafRef = React.useRef<number | null>(null);
    const reactCommittedScrollTopRef = React.useRef(0);
    const lastScrollTopRef = React.useRef(0);
    React.useEffect(
        () => () => {
            if (scrollTopRafRef.current != null) {
                cancelAnimationFrame(scrollTopRafRef.current);
            }
        },
        [],
    );
    // 时间轴 scroller 水平滚动条的占用高度（offsetHeight - clientHeight）。
    // 轨道头底部按此留出同高占位（bottomGutterHeightPx），保证轨道头与
    // 时间轴区域的竖直滚动范围严格一致。
    const [horizontalScrollbarGutterPx, setHorizontalScrollbarGutterPx] = React.useState(0);
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
    const state = useTimelineState();
    // 视觉插值播放头的共享读取点：bridge 的 onFrame 每帧写入（与绘制同源），
    // 缩放锚点与提交后纠正读取同一值——播放中缩放不得以 33Hz 轮询的 store
    // 滞后值锚定，否则播放头会跳变 δ·Δpx（δ = 轮询间隔内的插值领先量）。
    const visualPlayheadSecRef = React.useRef(0);
    const getVisualPlayheadSec = React.useCallback(() => visualPlayheadSecRef.current, []);
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
        trackGridOverlayLayerRef,
        rulerContentRef,
        rulerPlayheadLineRef,
        rulerPlayheadHeadRef,
        playheadRef,
        dropPreviewRef,
        lastClickedClipIdRef,
        syncScrollTop,
        pxPerSecRef,
        viewportWidthRef,
        rowHeightRef,
        scrollLeft,
        pxPerSec,
        setPxPerSec,
        viewportWidth,
        setViewportWidth,
        rowHeight,
        setRowHeight,
        altPressed,
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
        viewportStartSec,
        viewportEndSec,
        scrollHorizontalKb,
        scrollVerticalKb,
        scrollbarZoomKb,
        horizontalZoomKb,
        verticalZoomKb,
        paramFineAdjustKb,
        slipEditKb,
        pitchDragKb,
        noSnapKb,
        copyDragKb,
        crossfadeGripKb,
        fadeCurvatureKb,
        dropPreview,
        setDropPreview,
        clipDropNewTrack,
        setClipDropNewTrack,
        pendingDropDurationPathRef,
        syncScrollLeft,
        setScrollLeftAction,
        setScrollLeftState,
        beatFromClientX,
        trackIdFromClientY,
        rowTopForTrackId,
        ensureDropPreviewDuration,
        getDropPreviewWidthPx,
        snapTimeline,
        snapTimelineDetailed,
        isEditableTarget,
        isPointerOnNativeScrollbar,
        startPanPointer,
        setPlayheadFromClientX,
        startDeferredPlayheadSeek,
        keyboardZoomPendingRef,
    } = state;

    /** 竖直量化步长：overscan(4 行) 缓冲的一半，滞后永不越出 overscan 窗口。 */
    const scrollTopStepPx = Math.max(1, Math.round(rowHeight * 2));
    const commitTimelineScrollTop = React.useCallback(
        (next: number) => {
            lastScrollTopRef.current = next;
            if (scrollTopRafRef.current != null) return;
            scrollTopRafRef.current = requestAnimationFrame(() => {
                scrollTopRafRef.current = null;
                const latest = lastScrollTopRef.current;
                if (Math.abs(latest - reactCommittedScrollTopRef.current) < scrollTopStepPx) {
                    return;
                }
                reactCommittedScrollTopRef.current = latest;
                setTimelineScrollTop(latest);
            });
        },
        [scrollTopStepPx],
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
        const scroller = scrollRef.current;
        if (!scroller) {
            dispatch(setPendingPlayheadReveal(null));
            return;
        }
        // 仅当新光标位置不在可视范围内时才滚动（需求语义：画面内不扰动视图）。
        const x = Math.max(0, pendingPlayheadRevealSec) * pxPerSec;
        const left = scroller.scrollLeft;
        const right = left + scroller.clientWidth;
        if (x >= left && x <= right) {
            dispatch(setPendingPlayheadReveal(null));
            return;
        }
        const next = computeFocusCursorScrollLeft({
            playheadSec: pendingPlayheadRevealSec,
            pxPerSec,
            contentWidth: dynamicProjectSec * pxPerSec,
        });
        if (Math.abs(scroller.scrollLeft - next) > 0.5) {
            const applied = applyNativeScrollLeft(scroller, next);
            syncScrollLeft(applied);
        }
        dispatch(setPendingPlayheadReveal(null));
    }, [
        pendingPlayheadRevealSec,
        pxPerSec,
        dynamicProjectSec,
        scrollRef,
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

    // ── 2. Clip 多选 + 操作回调 ─────────────────────────────
    const clipActions = useTimelineClipActions({
        sessionRef,
        scrollRef,
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
        renamingClipId,
        selectionRect,
        onSelectionRectPointerDown,
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
        rangeSelectAnchorClipId,
        recordLastClickPosition,
        pasteClipsAtPlayhead,
        clearContextMenu,
        ensureTrackLaneSelected,
        selectTrackLaneClipRemote,
        deselectAllTrackLaneClips,
        openTrackLaneContextMenu,
        seekFromTrackLaneClientX,
        toggleTrackLaneClipMuted,
        toggleTrackLaneCtrlSelection,
        toggleTrackLaneMultiSelect,
        commitTrackLaneRename,
        handleTrackLaneRenameDone,
        commitTrackLaneGain,
        commitTrackLaneRate,
        editingBadge,
        setEditingBadge,
        handleBadgeEditDone,
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

    // 角标行内编辑开始：镜像 renamingClipId（onRenameStart）的两参适配器。
    const startTrackLaneBadgeEdit = React.useCallback(
        (clipId: string, field: "rate" | "gain") => {
            setEditingBadge({ clipId, field });
        },
        [setEditingBadge],
    );

    // 角标行内编辑提交：按字段路由到速率（自动调整时长）/增益提交。
    const commitTrackLaneBadgeEdit = React.useCallback(
        (clipId: string, field: "rate" | "gain", value: number) => {
            if (field === "rate") {
                commitTrackLaneRate(clipId, { rate: value, autoLength: true });
            } else {
                commitTrackLaneGain(clipId, value);
            }
        },
        [commitTrackLaneGain, commitTrackLaneRate],
    );
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
    // 淡化曲线循环点击：Ctrl（modifier.fadeShapeCycleClick）+左键点包络线
    // → 顺序切换到下一个预设形状，并把该侧曲率重置为新形状的默认值。
    const fadeShapeCycleKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.fadeShapeCycleClick"),
    );
    const clipMultiSelectToggleKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.clipMultiSelectToggle"),
    );
    const clipRangeSelectKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.clipRangeSelect"),
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
                await webApi.beginUndoGroup();
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
            scrollRef,
            sessionRef,
            pxPerSecRef,
            rowHeightRef,
            dropPreviewRef,
            pendingDropDurationPathRef,
            beatFromClientX,
            snapTimeline,
            trackIdFromClientY,
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
     * 内核宿主句柄（仅内核模式下非空）。
     *
     * 用途：把「轨道头滚动」这类外部意图转发给内核（`setScrollTop`），以及让
     * 标尺点击 seek 等需要「当前水平滚动位置」的逻辑在两种模式下共用一份取值。
     */
    const kernelHostRef = React.useRef<TimelineKernelHost | null>(null);

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
     */
    const handleKernelSeek = React.useCallback(
        (sec: number, commit: boolean) => {
            kernelSeekPendingRef.current = sec;
            if (commit) {
                if (kernelSeekRafRef.current != null) {
                    cancelAnimationFrame(kernelSeekRafRef.current);
                    kernelSeekRafRef.current = null;
                }
                kernelSeekPendingRef.current = null;
                void dispatch(seekPlayhead(sec));
                return;
            }
            if (kernelSeekRafRef.current != null) return;
            kernelSeekRafRef.current = requestAnimationFrame(() => {
                kernelSeekRafRef.current = null;
                const target = kernelSeekPendingRef.current;
                if (target == null) return;
                kernelSeekPendingRef.current = null;
                void dispatch(seekPlayhead(target));
            });
        },
        [dispatch],
    );

    /**
     * 内核选中回调：点击 clip 选中。
     *
     * 多选修饰键（Ctrl / ⌘）切换集合成员；否则单选并同步焦点 clip
     * （焦点 clip 驱动参数编辑器的编辑目标，必须与多选集合一起更新）。
     */
    const handleKernelSelectClip = React.useCallback(
        (clipId: string, additive: boolean) => {
            if (additive) {
                setMultiSelectedClipIds((prev) =>
                    prev.includes(clipId) ? prev.filter((id) => id !== clipId) : [...prev, clipId],
                );
                return;
            }
            setMultiSelectedClipIds([clipId]);
            dispatch(setSelectedClip(clipId));
        },
        [dispatch, setMultiSelectedClipIds],
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
                const clip = sessionRef.current.clips.find((item) => item.id === args.clipId);
                if (clip === undefined) return;
                kernelDragOriginRef.current = {
                    clipId: clip.id,
                    startSec: clip.startSec,
                    lengthSec: clip.lengthSec,
                    trackId: clip.trackId,
                    snapOffsetSec: Math.max(0, Number(clip.snapOffsetSec) || 0),
                    copyMode: false,
                };
            }
            const origin = kernelDragOriginRef.current;
            if (origin === null) return;
            // copy 模式判定：复用旧实现的函数（含"已配置绑定为准 + 非 macOS 的 Ctrl
            // 回退"）。**单向**——一旦进入 copy 就不再退回移动，避免松手瞬间语义反转。
            origin.copyMode = resolveClipDragCopyMode({
                existingCopyMode: origin.copyMode,
                ctrlKey: args.modifiers.ctrlKey,
                modifierActive: isModifierActive(copyDragKb, args.modifiers),
            });
            const rawStart = Math.max(0, origin.startSec + args.deltaSec);
            // 吸附：复用旧实现的 snapTimelineDetailed（多源候选 + 取更近者），
            // 内核只给几何位移，吸附规则不在内核里重写。
            //
            // `highlight` 必须传：吸附高亮的**发布与清除**都由该函数按此选项统一
            // 处理（见 useTimelineState.snapTimelineDetailed）。不传时吸附本身仍
            // 生效、位置也对，但完全没有视觉反馈——表现为"吸附没生效"。
            const nextStart = s.snapEnabled
                ? snapTimelineDetailed(rawStart, "clip", {
                      originSec: origin.startSec,
                      anchorTrackId: args.targetTrackId,
                      excludeClipIds: new Set([args.clipId]),
                      moveLengthSec: origin.lengthSec,
                      moveSnapOffsetSec: origin.snapOffsetSec,
                      highlight: {
                          sources: [{ trackId: args.targetTrackId, clipId: args.clipId }],
                      },
                  }).sec
                : rawStart;
            // 吸附被关闭（拖拽中切开关 / 按住临时取反键）：高亮必须清掉，
            // 否则会残留上一次的吸附提示（旧实现同样在 else 分支清除）。
            if (!s.snapEnabled) clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            if (origin.copyMode) {
                // copy：**原 clip 不动**，只更新 ghost（内容坐标）。
                setKernelGhost([
                    {
                        key: args.clipId,
                        leftPx: nextStart * pxPerSec,
                        widthPx: Math.max(1, origin.lengthSec * pxPerSec),
                        trackId: args.targetTrackId,
                    },
                ]);
                return;
            }
            batch(() => {
                dispatch(moveClipStart({ clipId: args.clipId, startSec: nextStart }));
                dispatch(moveClipTrack({ clipId: args.clipId, trackId: args.targetTrackId }));
            });
        },
        [copyDragKb, dispatch, pxPerSec, s.snapEnabled, sessionRef, snapTimelineDetailed],
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
            if (origin.copyMode) {
                setKernelGhost(null);
                // copy 模式下原 clip 从未被移动：既不需要回滚，也不走 move 提交。
                if (args.cancelled) return;
                // 落库复用抽出的共享函数（与旧实现**同一份**复制语义）。
                void copyClipsFromDrag({
                    sourceClipIds: [origin.clipId],
                    initialById: {
                        [origin.clipId]: {
                            startSec: origin.startSec,
                            trackId: origin.trackId,
                        },
                    },
                    initialTrackIndexById: {},
                    deltaSec: args.deltaSec,
                    // 内核拖拽的落点始终是已有轨道（`resolveTargetTrackIndex` 越界时
                    // 回落原轨），因此不涉及建新轨。
                    dropToNewTrack: false,
                    trackOffset: 0,
                    allowTrackMove: true,
                    hasMixedTrackSelection: false,
                    autoCrossfadeEnabled: s.autoCrossfadeEnabled,
                    dispatch,
                    sessionRef,
                    setMultiSelectedClipIds,
                    // 目标轨由内核给（它做了落点换算），不走偏移解析。
                    resolveTrackIdByOffset: () => args.targetTrackId,
                    maybeSelectTargetTrack: () => undefined,
                    createNewTracksForDrop: async () => [],
                    createNewTrackForDrop: async () => null,
                }).catch(() => undefined);
                return;
            }
            if (args.cancelled) {
                batch(() => {
                    dispatch(moveClipStart({ clipId: origin.clipId, startSec: origin.startSec }));
                    dispatch(moveClipTrack({ clipId: origin.clipId, trackId: origin.trackId }));
                });
                return;
            }
            dispatch(checkpointHistory());
            // 提交值取 Redux 里的当前值（预览已写入**吸附后**的结果）——
            // 用 origin + 内核原始位移会绕开吸附，导致"预览吸附、提交不吸附"。
            const clip = sessionRef.current.clips.find((item) => item.id === args.clipId);
            void dispatch(
                moveClipsRemote({
                    moves: [
                        {
                            clipId: args.clipId,
                            startSec:
                                clip?.startSec ?? Math.max(0, origin.startSec + args.deltaSec),
                            trackId: args.targetTrackId,
                        },
                    ],
                }),
            );
        },
        [dispatch, sessionRef],
    );

    /** 内核 trim：按下时的原始几何（把相对位移换算为绝对值，并支持回滚）。 */
    const kernelTrimOriginRef = React.useRef<{
        clipId: string;
        startSec: number;
        lengthSec: number;
        trackId: string;
        sourceStartSec: number;
        sourceEndSec: number;
    } | null>(null);

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
        }) => {
            if (kernelTrimOriginRef.current?.clipId !== args.clipId) {
                const clip = sessionRef.current.clips.find((item) => item.id === args.clipId);
                if (clip === undefined) return;
                kernelTrimOriginRef.current = {
                    clipId: clip.id,
                    startSec: clip.startSec,
                    lengthSec: clip.lengthSec,
                    trackId: clip.trackId,
                    sourceStartSec: clip.sourceStartSec,
                    sourceEndSec: clip.sourceEndSec,
                };
            }
            const origin = kernelTrimOriginRef.current;
            if (origin === null) return;

            // 吸附：左边缘吸**起点**、右边缘吸**右端**——两者吸附的对象不同，
            // 统一吸起点会让右边缘 trim 落在错误的位置。
            let nextStart = args.startSec;
            let nextLength = args.lengthSec;
            let deltaSec = args.deltaSec;
            if (s.snapEnabled) {
                const snapArgs = {
                    originSec: origin.startSec,
                    anchorTrackId: origin.trackId,
                    excludeClipIds: new Set([args.clipId]),
                    moveLengthSec: args.lengthSec,
                    moveSnapOffsetSec: 0,
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
            } else {
                clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            }

            batch(() => {
                dispatch(moveClipStart({ clipId: args.clipId, startSec: nextStart }));
                dispatch(setClipLength({ clipId: args.clipId, lengthSec: nextLength }));
                if (args.edge === "left") {
                    dispatch(
                        setClipSourceRange({
                            clipId: args.clipId,
                            sourceStartSec: origin.sourceStartSec + deltaSec,
                        }),
                    );
                } else {
                    dispatch(
                        setClipSourceRange({
                            clipId: args.clipId,
                            sourceEndSec: origin.sourceEndSec + deltaSec,
                        }),
                    );
                }
            });
        },
        [dispatch, sessionRef, s.snapEnabled, snapTimelineDetailed],
    );

    /** 内核 trim 收尾：提交或回滚（取消时三个字段一起还原）。 */
    /**
     * 内核吸附偏移手势是否已真正开始。
     *
     * 预览回调每帧触发，但 `beginInteraction` / `checkpointHistory` /
     * `beginSnapGesture` 只能做一次——用这个 ref 去重，同时给收尾一个
     * 「零位移单击」的判据（旧实现用 `drag.checkpointed`）。
     */
    const kernelSnapOffsetActiveRef = React.useRef(false);

    /**
     * 内核吸附偏移预览：与旧实现 `useSnapOffsetDrag` 同源。
     *
     * 被吸附对象是**手柄的绝对时间线位置**（`clipStart + offset`）——不是 clip 起点；
     * 高亮发布为该 clip 所在行的亮条。落库前把偏移钳制到 `[0, clip 长度]`。
     *
     * 说明：与内核既有的拖拽 / trim 预览一致，这里只读 `s.snapEnabled`，不处理
     * 「拖拽中按住免吸附修饰键」——那是内核所有手势共有的缺口，应统一补。
     */
    const handleKernelSnapOffsetPreview = React.useCallback(
        (args: { clipId: string; rawOffsetSec: number }) => {
            const clip = sessionRef.current.clips.find((item) => item.id === args.clipId);
            if (clip === undefined) return;
            const clipStart = Number(clip.startSec) || 0;
            const clipLen = Math.max(0, Number(clip.lengthSec) || 0);
            if (!kernelSnapOffsetActiveRef.current) {
                // 首次真实位移：交互锁 + undo 检查点 + 吸附手势一起开。
                kernelSnapOffsetActiveRef.current = true;
                dispatch(beginInteraction());
                dispatch(checkpointHistory());
                beginSnapGesture();
            }
            const rawAbs = clipStart + args.rawOffsetSec;
            const nextAbs = s.snapEnabled
                ? snapTimelineDetailed(rawAbs, "clip", {
                      originSec: clipStart + (Number(clip.snapOffsetSec) || 0),
                      anchorTrackId: clip.trackId,
                      excludeClipIds: new Set([args.clipId]),
                      highlight: {
                          sources: [{ trackId: clip.trackId, clipId: args.clipId }],
                      },
                  }).sec
                : rawAbs;
            if (!s.snapEnabled) clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            dispatch(
                setClipSnapOffset({
                    clipId: args.clipId,
                    snapOffsetSec: Math.min(Math.max(nextAbs - clipStart, 0), clipLen),
                }),
            );
        },
        [dispatch, s.snapEnabled, snapTimelineDetailed],
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
            kernelSnapOffsetActiveRef.current = false;
            clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            // 手势从未真正开始（零位移单击）：begin* 都没调用过，不能 end*。
            if (!wasActive) return;
            endSnapGesture();
            if (args.cancelled || !args.changed) {
                dispatch(endInteraction());
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
                .finally(() => {
                    dispatch(endInteraction());
                });
        },
        [dispatch],
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
            if (args.cancelled) {
                batch(() => {
                    dispatch(moveClipStart({ clipId: origin.clipId, startSec: origin.startSec }));
                    dispatch(setClipLength({ clipId: origin.clipId, lengthSec: origin.lengthSec }));
                    dispatch(
                        setClipSourceRange({
                            clipId: origin.clipId,
                            sourceStartSec: origin.sourceStartSec,
                            sourceEndSec: origin.sourceEndSec,
                        }),
                    );
                });
                return;
            }
            dispatch(checkpointHistory());
            // 同拖拽：提交值取 Redux 当前值（预览已写入吸附后的结果）。
            const clip = sessionRef.current.clips.find((item) => item.id === args.clipId);
            void dispatch(
                setClipsStateBulkRemote({
                    updates: [
                        {
                            clipId: args.clipId,
                            startSec: clip?.startSec ?? args.startSec,
                            lengthSec: clip?.lengthSec ?? args.lengthSec,
                        },
                    ],
                }),
            );
        },
        [dispatch, sessionRef],
    );

    /** 内核淡变角：按下时的原始值（用于回滚）。 */
    const kernelFadeOriginRef = React.useRef<{
        clipId: string;
        fadeInSec: number;
        fadeOutSec: number;
    } | null>(null);

    /** 内核淡变角预览：只改对应一侧的淡变长度（另一侧保持不变）。 */
    const handleKernelFadePreview = React.useCallback(
        (args: { clipId: string; side: "in" | "out"; fadeSec: number; deltaSec: number }) => {
            if (kernelFadeOriginRef.current?.clipId !== args.clipId) {
                const clip = sessionRef.current.clips.find((item) => item.id === args.clipId);
                if (clip === undefined) return;
                kernelFadeOriginRef.current = {
                    clipId: clip.id,
                    fadeInSec: clip.fadeInSec,
                    fadeOutSec: clip.fadeOutSec,
                };
            }
            dispatch(
                setClipFades(
                    args.side === "in"
                        ? { clipId: args.clipId, fadeInSec: args.fadeSec }
                        : { clipId: args.clipId, fadeOutSec: args.fadeSec },
                ),
            );
        },
        [dispatch, sessionRef],
    );

    /** 内核淡变角收尾：提交或回滚（取消时两侧一起还原）。 */
    const handleKernelFadeCommit = React.useCallback(
        (args: { clipId: string; side: "in" | "out"; fadeSec: number; cancelled: boolean }) => {
            const origin = kernelFadeOriginRef.current;
            kernelFadeOriginRef.current = null;
            if (origin === null) return;
            if (args.cancelled) {
                dispatch(
                    setClipFades({
                        clipId: origin.clipId,
                        fadeInSec: origin.fadeInSec,
                        fadeOutSec: origin.fadeOutSec,
                    }),
                );
                return;
            }
            dispatch(checkpointHistory());
            void dispatch(
                setClipsStateBulkRemote({
                    updates: [
                        args.side === "in"
                            ? { clipId: args.clipId, fadeInSec: args.fadeSec }
                            : { clipId: args.clipId, fadeOutSec: args.fadeSec },
                    ],
                }),
            );
        },
        [dispatch],
    );

    /** 内核框选：拖动前的选择快照（合并与回滚的基准）。 */
    const kernelBoxSelectOriginRef = React.useRef<string[] | null>(null);
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
                kernelBoxSelectOriginRef.current = [...multiSelectedIdsRef.current];
            }
            setMultiSelectedClipIds(
                computeTimelineRectSelection({
                    selectionBeforeDrag: kernelBoxSelectOriginRef.current,
                    selectedInRect: [...args.clipIds],
                    primaryModifierPressedAtStart: args.additive,
                }),
            );
        },
        [setMultiSelectedClipIds],
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
                void dispatch(selectTrackRemote(args.trackId));
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
     * 内核双击 clip：请求参数编辑器按 clip 起止范围创建选区。
     *
     * 与旧实现（`ClipItem` 的双击分支）同源：关闭右键菜单 → 派发
     * `hifi:editOp/selectClipParamRange`，交互焦点随之切到参数编辑器侧。
     * 用 window 事件而不是直接调 store：参数编辑器在另一棵子树里监听它，
     * 这条契约与渲染模式无关（内核 / 旧实现共用同一入口）。
     */
    const handleKernelDoubleClickClip = React.useCallback(
        (clipId: string) => {
            clearContextMenu();
            window.dispatchEvent(
                new CustomEvent("hifi:editOp", {
                    detail: { op: "selectClipParamRange", clipId },
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
            initialValue: clip.name,
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
                    if (trimmed.length === 0) return;
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
        earlierClipId: string;
        laterClipId: string;
        earlierStartSec: number;
        earlierLengthSec: number;
        laterStartSec: number;
        laterLengthSec: number;
    } | null>(null);

    /**
     * 内核交叉点抓手预览：同时移动双方边缘。
     *
     * 语义（与旧实现 `crossfade_edges` 一致）：前一个 clip 的**右缘**与后一个
     * clip 的**左缘**按同一位移移动 → 重叠长度不变（手动 / 自动淡变长度都不受影响）。
     * - earlier：起点不动，长度 `+= delta`（右缘随之移动）
     * - later：起点 `+= delta`，长度 `-= delta`（左缘移动、右缘不动）
     *
     * 位移钳制到 `[-earlierLength, laterLength]`：越界会让任一侧长度为负。
     */
    const handleKernelCrossfadeGripPreview = React.useCallback(
        (args: { earlierClipId: string; laterClipId: string; deltaSec: number }) => {
            const clips = sessionRef.current.clips;
            const earlier = clips.find((item) => item.id === args.earlierClipId);
            const later = clips.find((item) => item.id === args.laterClipId);
            if (earlier === undefined || later === undefined) return;
            if (kernelCrossfadeOriginRef.current?.earlierClipId !== args.earlierClipId) {
                kernelCrossfadeOriginRef.current = {
                    earlierClipId: earlier.id,
                    laterClipId: later.id,
                    earlierStartSec: earlier.startSec,
                    earlierLengthSec: earlier.lengthSec,
                    laterStartSec: later.startSec,
                    laterLengthSec: later.lengthSec,
                };
            }
            const origin = kernelCrossfadeOriginRef.current;
            if (origin === null) return;
            const delta = Math.min(
                origin.laterLengthSec,
                Math.max(-origin.earlierLengthSec, args.deltaSec),
            );
            batch(() => {
                dispatch(
                    setClipLength({
                        clipId: origin.earlierClipId,
                        lengthSec: Math.max(0, origin.earlierLengthSec + delta),
                    }),
                );
                dispatch(
                    moveClipStart({
                        clipId: origin.laterClipId,
                        startSec: Math.max(0, origin.laterStartSec + delta),
                    }),
                );
                dispatch(
                    setClipLength({
                        clipId: origin.laterClipId,
                        lengthSec: Math.max(0, origin.laterLengthSec - delta),
                    }),
                );
            });
        },
        [dispatch, sessionRef],
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
                batch(() => {
                    dispatch(
                        moveClipStart({
                            clipId: origin.earlierClipId,
                            startSec: origin.earlierStartSec,
                        }),
                    );
                    dispatch(
                        setClipLength({
                            clipId: origin.earlierClipId,
                            lengthSec: origin.earlierLengthSec,
                        }),
                    );
                    dispatch(
                        moveClipStart({
                            clipId: origin.laterClipId,
                            startSec: origin.laterStartSec,
                        }),
                    );
                    dispatch(
                        setClipLength({
                            clipId: origin.laterClipId,
                            lengthSec: origin.laterLengthSec,
                        }),
                    );
                });
                return;
            }
            const clips = sessionRef.current.clips;
            const earlier = clips.find((item) => item.id === origin.earlierClipId);
            const later = clips.find((item) => item.id === origin.laterClipId);
            if (earlier === undefined || later === undefined) return;
            dispatch(checkpointHistory());
            void dispatch(
                setClipsStateBulkRemote({
                    updates: [
                        { clipId: earlier.id, lengthSec: earlier.lengthSec },
                        {
                            clipId: later.id,
                            startSec: later.startSec,
                            lengthSec: later.lengthSec,
                        },
                    ],
                }),
            );
        },
        [dispatch, sessionRef],
    );

    /** 内核交互回调集合（引用稳定：内核创建时取一次）。 */
    const kernelInteractions = React.useMemo(
        () => ({
            onSeek: handleKernelSeek,
            onSelectClip: handleKernelSelectClip,
            onDoubleClickClip: handleKernelDoubleClickClip,
            onToggleClipMute: handleKernelToggleClipMute,
            onOpenClipFormant: handleKernelOpenClipFormant,
            onRateBadgeMenu: handleKernelRateBadgeMenu,
            onRenameClipStart: handleKernelRenameClipStart,
            onBadgeEditStart: handleKernelBadgeEditStart,
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
        }),
        [
            handleKernelSeek,
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
        ],
    );

    // ── 4. 全局事件监听 ─────────────────────────────────────
    useTimelineEventHandlers({
        dispatch,
        sessionRef,
        getPlayheadSec: getVisualPlayheadSec,
        scrollRef,
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
    const { startEditDrag } = useEditDrag({
        scrollRef,
        sessionRef,
        dispatch,
        multiSelectedClipIds,
        multiSelectedSet,
        snapTimelineDetailed: state.snapTimelineDetailed,
        beatFromClientX,
        noSnapKb,
        snapEnabled: s.timelineSnap.enabled,
        timelineSnap: s.timelineSnap,
        pxPerSec,
        ignoreGrouping,
        paramFineAdjustKb,
        crossfadeGripKb,
        fadeCurvatureKb,
    });

    const startSlipDrag = useSlipDrag({
        scrollRef,
        sessionRef,
        dispatch,
        multiSelectedClipIds,
        multiSelectedSet,
        beatFromClientX,
        ignoreGrouping,
        timelineSnap: s.timelineSnap,
        pxPerSec,
        noSnapKb,
    });

    // SnapOffset 三角手柄拖拽（走完整吸附引擎与竖线高亮）。
    const startSnapOffsetDrag = useSnapOffsetDrag({
        scrollRef,
        sessionRef,
        dispatch,
        snapTimelineDetailed: state.snapTimelineDetailed,
        beatFromClientX,
        noSnapKb,
        snapEnabled: s.timelineSnap.enabled,
    });

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

    const {
        startClipDrag: _startClipDragInner,
        ghostDrag,
        verticalTrackLockTrackId,
    } = useClipDrag({
        scrollRef,
        sessionRef,
        rowHeight,
        pxPerSec,
        multiSelectedClipIds,
        multiSelectedSet,
        dispatch,
        snapTimelineDetailed: state.snapTimelineDetailed,
        beatFromClientX,
        trackIdFromClientY,
        setClipDropNewTrack,
        setMultiSelectedClipIds,
        slipEditKb,
        noSnapKb,
        snapEnabled: s.timelineSnap.enabled,
        copyDragKb,
        multiSelectToggleKb: clipMultiSelectToggleKb,
        rangeSelectKb: clipRangeSelectKb,
        autoCrossfadeEnabled: s.autoCrossfadeEnabled,
        ignoreGrouping,
        onCtrlClick: toggleTrackLaneCtrlSelection,
    });

    const clipById = useMemo(
        () => new Map(s.clips.map((clip) => [clip.id, clip] as const)),
        [s.clips],
    );

    const newTrackGhostClips = useMemo(() => {
        if (clipDropNewTrack) {
            const moved = s.clips.filter((clip) => clip.trackId === NEW_TRACK_SENTINEL);
            if (moved.length > 0) return moved;
        }
        if (!ghostDrag || ghostDrag.targetTrackId != null) {
            return [];
        }
        return ghostDrag.clipIds
            .map((clipId) => {
                const initial = ghostDrag.initialById[clipId];
                const clip = clipById.get(clipId);
                if (!initial || !clip) return null;
                return {
                    ...clip,
                    startSec: Math.max(0, initial.startSec + ghostDrag.deltaSec),
                };
            })
            .filter((clip): clip is (typeof s.clips)[number] => clip != null);
        // eslint-disable-next-line react-hooks/exhaustive-deps -- 仅依赖 s.clips 粒度；加入整个 s 会在任何会话字段变化时重算（既有粒度模式）
    }, [clipById, clipDropNewTrack, ghostDrag, s.clips]);

    const startClipDrag = React.useCallback(
        (
            e: React.PointerEvent<HTMLDivElement>,
            clipId: string,
            clipstartSec: number,
            altPressedHint?: boolean,
        ) => {
            _startClipDragInner(e, clipId, clipstartSec, altPressedHint, startSlipDrag);
        },
        [_startClipDragInner, startSlipDrag],
    );
    const handleSelectTrack = React.useCallback(
        (trackId: string) => {
            if (sessionRef.current.selectedTrackId === trackId) {
                return;
            }
            void dispatch(selectTrackRemote(trackId));
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
    const handleTrackListScrollTopChange = React.useCallback(
        (scrollTop: number) => {
            // 内核模式：把轨道头的滚动意图转发给内核（由内核统一钳制与标脏；
            // 内核回写轨道头时带 0.5px 容差，因此不会形成来回循环）。
            const host = kernelHostRef.current;
            if (host != null) {
                if (Math.abs(host.getViewport().scrollTop - scrollTop) < 0.5) return;
                host.setScrollTop(scrollTop);
                return;
            }
            const timelineScroller = scrollRef.current;
            if (!timelineScroller) return;
            if (Math.abs(timelineScroller.scrollTop - scrollTop) < 0.5) return;
            timelineScroller.scrollTop = scrollTop;
        },
        [scrollRef],
    );

    const trackGridHeight = Math.max(0, contentHeight - TRACK_ADD_ROW_HEIGHT);
    const timelineRenderModel = useMemo(
        () =>
            buildTimelineRenderModel({
                tracks: s.tracks,
                clips: s.clips,
                viewportStartSec,
                viewportEndSec,
                pxPerSec,
                rowHeight,
                scrollTopPx: timelineScrollTop,
                viewportHeightPx: scrollRef.current?.clientHeight ?? 0,
            }),
        [
            pxPerSec,
            rowHeight,
            s.clips,
            s.tracks,
            scrollRef,
            timelineScrollTop,
            viewportEndSec,
            viewportStartSec,
        ],
    );
    // slice 每次渲染都会产生新引用；不缓存的话下游所有 useMemo 与
    // 两块画布的 memo 会在每次无关更新（播放头/滚动/修饰键）时全量重算重绘。
    const visibleTracks = React.useMemo(
        () => s.tracks.slice(timelineRenderModel.startIndex, timelineRenderModel.endIndex + 1),
        [s.tracks, timelineRenderModel.startIndex, timelineRenderModel.endIndex],
    );
    const visibleTrackClipCacheRef = React.useRef<
        Record<
            string,
            {
                clipIds: string[];
                clips: typeof s.clips;
            }
        >
    >({});
    /** 上一次返回的 `Record<trackId, clips>`；用于在外层做引用复用。 */
    const visibleTrackClipsByIdRef = React.useRef<Record<string, typeof s.clips>>({});
    const visibleTrackClipsById = useMemo(() => {
        const nextCache: typeof visibleTrackClipCacheRef.current = {};
        const nextByTrackId = {} as Record<string, typeof s.clips>;

        for (const track of visibleTracks) {
            const clipIds = timelineRenderModel.visibleClipIdsByTrackId[track.id] ?? [];
            const prev = visibleTrackClipCacheRef.current[track.id];
            const canReusePrev =
                prev != null &&
                prev.clipIds.length === clipIds.length &&
                clipIds.every(
                    (clipId, index) =>
                        prev.clipIds[index] === clipId &&
                        prev.clips[index] === clipById.get(clipId),
                );

            const clips = canReusePrev
                ? prev.clips
                : (clipIds
                      .map((clipId) => clipById.get(clipId) ?? null)
                      .filter(
                          (clip): clip is (typeof s.clips)[number] => clip != null,
                      ) as typeof s.clips);

            nextCache[track.id] = {
                clipIds,
                clips,
            };
            nextByTrackId[track.id] = clips;
        }

        visibleTrackClipCacheRef.current = nextCache;

        // 连外层对象一起复用：各轨道的 `clips` 数组本身已是稳定引用，但若
        // 每次都新建外层对象，下游 `TimelineWaveformSurface.rows` 与
        // `buildSparseClipRenderModel` 的 memo 会在**每个滚动帧**失效——
        // 视口秒窗每帧都变，导致 `visibleClipIdsByTrackId` 每次都是新数组。
        // 那会让两块画布在总线 paint 之外又被 React 提交重绘一次（P1 要消除的
        // 重复绘制）。轨道集合与各自的 clips 引用都没变时，直接返回旧对象。
        const prevByTrackId = visibleTrackClipsByIdRef.current;
        const prevKeys = Object.keys(prevByTrackId);
        const nextKeys = Object.keys(nextByTrackId);
        const sameShape =
            prevKeys.length === nextKeys.length &&
            nextKeys.every((key) => prevByTrackId[key] === nextByTrackId[key]);
        if (sameShape) return prevByTrackId;
        visibleTrackClipsByIdRef.current = nextByTrackId;
        return nextByTrackId;
        // eslint-disable-next-line react-hooks/exhaustive-deps -- 仅类型位置引用 s.clips（缓存已按需要稳定化）；加入整个 s 会让缓存扫描随任何会话变化失效（既有模式）
    }, [clipById, timelineRenderModel.visibleClipIdsByTrackId, visibleTracks]);
    const selectedClipTrackId = s.selectedClipId
        ? (clipById.get(s.selectedClipId)?.trackId ?? null)
        : null;
    const visibleTrackCanvasHeight = Math.max(1, visibleTracks.length * rowHeight);
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
    // 全图层共享的统一坐标投影：网格 / 标尺 / clip 体 / 波形 / 播放头都从这里
    // 取位置与缩放，任何图层都不许再自行执行 `sec * pxPerSec`（历史错位根因）。
    // 用 useMemo 缓存引用，否则下游 React.memo 会因新对象引用而每帧失效。
    const timelineAxis = useMemo(
        () =>
            createTimelineAxis({
                pxPerSec,
                scrollLeftPx: scrollLeft,
                scrollTopPx: timelineScrollTop,
                viewportWidthPx: Math.max(1, Math.ceil(viewportWidth)),
                dpr: window.devicePixelRatio || 1,
            }),
        [pxPerSec, scrollLeft, timelineScrollTop, viewportWidth],
    );
    /**
     * 「内容轴」：只含 `pxPerSec`，不含滚动。
     *
     * clip 体渲染模型（`buildSparseClipRenderModel`）的全部投影都落在**内容
     * 坐标系**上——`secToContentPx` / `durationToWidthPx` / `secToSpanPx`
     * 只消费 `pxPerSec`，`topPx` 由 `(startTrackIndex + i) * rowHeight` 得出。
     * 因此滚动帧里模型内容**逐像素不变**，却因为 `timelineAxis` 每帧都是新
     * 对象而被整体重建一次，进而让 `drawClips` 也是新数组、让 clip 体画布在
     * 总线 paint 之外又被 React 提交重绘一遍。
     *
     * 把滚动从依赖里剥掉后，`drawClips` 的**引用**在纯滚动帧保持稳定，
     * 这是 P1 消除重复绘制的前提。
     */
    const contentAxis = useMemo(
        () => createTimelineAxis({ pxPerSec, dpr: window.devicePixelRatio || 1 }),
        [pxPerSec],
    );
    const sparseClipRenderModel = useMemo(() => {
        // 前导重叠秒数：每个 clip 的"被同轨前一个 clip 压住"部分，
        // canvas 在该区画半透色块，让下 clip 的色块/波形透出——避免两层
        // 不透明色块叠加成脏色。
        const leadingOverlapSecByClipId: Record<string, number> = {};
        for (const track of visibleTracks) {
            const clips = visibleTrackClipsById[track.id] ?? [];
            Object.assign(leadingOverlapSecByClipId, computeLeadingOverlapSecByClipId(clips));
        }
        return buildSparseClipRenderModel({
            visibleTracks,
            startTrackIndex: timelineRenderModel.startIndex,
            visibleTrackClipsById,
            axis: contentAxis,
            rowHeight,
            selectedClipId: s.selectedClipId,
            multiSelectedClipIds,
            renamingClipId,
            disabledGroupIds,
            leadingOverlapSecByClipId,
        });
    }, [
        multiSelectedClipIds,
        contentAxis,
        renamingClipId,
        rowHeight,
        s.selectedClipId,
        timelineRenderModel.startIndex,
        visibleTrackClipsById,
        visibleTracks,
        disabledGroupIds,
    ]);
    const timelineCanvasModel = useMemo(
        () => ({
            drawClips: sparseClipRenderModel.drawClips,
            activeGroupIds,
            disabledGroupIds,
        }),
        [sparseClipRenderModel.drawClips, activeGroupIds, disabledGroupIds],
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

    // scrollLeft 现在按 REACT_SCROLL_STEP_PX 量化提交，React 渲染期用
    // `timelineAxis` 算出的播放头 left 最多滞后一个步长；而播放头的真实位置
    // 由 useVisualPlayhead / syncScrollLeft 用**实时** scrollLeft 命令式写入。
    // 这里在每次提交后立即用视觉插值值纠正一次，避免 React 的滞后写入把播放头
    // 推回旧位置。仅在提交时运行（滚动中约每 256px 一次），成本可忽略。
    // ★ 必须读视觉插值 ref 而非 s.playheadSec：后者是 33Hz 轮询的滞后值，
    //   缩放提交帧用它写播放头，下一帧 rAF 又写视觉值——表现为缩放瞬间跳变。
    React.useLayoutEffect(() => {
        const scroller = scrollRef.current;
        if (!scroller || !playheadRef.current) return;
        const playheadLeftPx = visualPlayheadSecRef.current * pxPerSec;
        // 与播放头其余写入点同一设备像素吸附（见 TimelineTransportBridge onFrame）。
        playheadRef.current.style.left = `${snapToDevicePx(
            playheadLeftPx - scroller.scrollLeft,
            readDevicePixelRatio(),
        )}px`;
    }, [pxPerSec, s.playheadSec, scrollLeft, playheadRef, scrollRef, visualPlayheadSecRef]);

    /**
     * 标尺节点（内核模式与旧模式共用同一实例）。
     *
     * 标尺是重交互、低频变化的 DOM 子树（刻度标签 / Tempo Map 旗帜拖拽 / 内联编辑 /
     * 右键菜单），搬进 canvas 等于整体重写且收益极低。让两种渲染模式共用它，视觉与
     * 交互天然与旧实现一致；内核只把「跟随水平滚动」的部分收敛为 rAF 内一次
     * transform 写入（见 kernel host 的 syncDom）。
     */
    const timeRulerNode = (
        // playheadSec 传提交值（而非渲染期读 ref）：视觉插值由 playheadLineRef /
        // playheadHeadRef 命令式驱动；React 仅在该值真正变化时重写 style.left，
        // 写入的是最新提交位置而非陈旧值。
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
                // 水平滚动位置：旧模式取原生 scroller，内核模式取内核视口
                // （两种模式互斥挂载，但取值的「实时性」必须一致——拖拽期间
                // 滚动位置可能被自动滚动改变，因此每次换算都重新读）。
                const readScrollLeft = (): number | null => {
                    const scroller = scrollRef.current;
                    if (scroller != null) return scroller.scrollLeft;
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
        const trackId = trackIdFromClientY(e.clientY);
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
        const trackId = trackIdFromClientY(e.clientY);
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
                    {/* 新渲染内核（自绘滚动 + 单 WebGL2）替换**轨道区**。
                        标尺（`timeRulerNode`）与左侧轨道头保留 DOM：它们重交互、低频
                        变化，搬进 canvas 等于重写 Tempo 旗帜拖拽与电平表，收益极低；
                        内核只把「跟随视口」的部分收敛为 rAF 内一次 transform / scrollTop
                        写入。轨道区（网格 / clip / 波形 / 交互）才是滚动瓶颈，由内核自绘。
                        开关默认关闭（见 featureFlag），两套渲染不会同时挂载。 */}
                    {TIMELINE_KERNEL_ENABLED ? (
                        <>
                            {timeRulerNode}
                            <TimelineKernelView
                                rowHeight={rowHeight}
                                onRowHeightChange={setRowHeight}
                                initialPxPerSec={pxPerSec}
                                onPxPerSecChange={setPxPerSec}
                                onScrollLeftCommit={setScrollLeftState}
                                onViewportWidthChange={setViewportWidth}
                                getPlayheadSec={getVisualPlayheadSec}
                                rulerContentRef={rulerContentRef}
                                trackListScrollerRef={trackListScrollRef}
                                rulerPlayheadLineRef={rulerPlayheadLineRef}
                                hostRef={kernelHostRef}
                                interactions={kernelInteractions}
                                activeGroupIds={kernelActiveGroupIds}
                                disabledGroupIds={disabledGroupIds}
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
                                              items: kernelGhost,
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
                            />
                        </>
                    ) : (
                        <>
                            {timeRulerNode}
                            {/* Tracks Area */}
                            <TimelineScrollArea
                                scrollRef={scrollRef}
                                projectSec={dynamicProjectSec}
                                pxPerSec={pxPerSec}
                                setPxPerSec={setPxPerSec}
                                rowHeight={rowHeight}
                                setRowHeight={setRowHeight}
                                setScrollLeft={setScrollLeftAction}
                                commitScrollLeftState={setScrollLeftState}
                                commitScrollTopState={commitTimelineScrollTop}
                                rulerContentRef={rulerContentRef}
                                scrollHorizontalKb={scrollHorizontalKb}
                                scrollVerticalKb={scrollVerticalKb}
                                scrollbarZoomKb={scrollbarZoomKb}
                                horizontalZoomKb={horizontalZoomKb}
                                verticalZoomKb={verticalZoomKb}
                                getPlayheadSec={getVisualPlayheadSec}
                                playheadZoomEnabled={s.playheadZoomEnabled}
                                className="flex-1 bg-qt-graph-bg overflow-auto relative custom-scrollbar"
                                data-timeline-scroller
                                onDoubleClickCapture={(e) => {
                                    // 时间轴非输入区域的双击只用于自定义交互，不应触发 WebView 文本选择；
                                    // 显式声明可选择（data-hs-selectable）的区域保留原生双击行为。
                                    if (isEditableTarget(e.target)) return;
                                    const target = e.target as HTMLElement | null;
                                    if (target?.closest?.("[data-hs-selectable='true']")) return;
                                    e.preventDefault();
                                }}
                                onScroll={(e) => {
                                    const el = e.currentTarget as HTMLDivElement;
                                    // 竖直轴同帧提交：sticky 画布层（clip 体/波形面）必须在
                                    // 绘制前拿到新 scrollTop（总线同步派发）；React state
                                    // 只驱动窗口化等非视觉更新。
                                    syncScrollTop(el.scrollTop);
                                    commitTimelineScrollTop(el.scrollTop);
                                    if (trackListScrollRef.current) {
                                        if (
                                            Math.abs(
                                                trackListScrollRef.current.scrollTop - el.scrollTop,
                                            ) >= 0.5
                                        ) {
                                            trackListScrollRef.current.scrollTop = el.scrollTop;
                                        }
                                    }
                                }}
                                onMouseDownCapture={(e) => {
                                    if (e.button === 1) {
                                        e.preventDefault();
                                    }
                                }}
                                onAuxClick={(e) => {
                                    if (e.button === 1) {
                                        e.preventDefault();
                                    }
                                }}
                                onContextMenu={(e) => {
                                    e.preventDefault();
                                    setContextMenu(null);

                                    const target = e.target as HTMLElement | null;
                                    if (target?.closest?.("[data-hs-context-menu='1']")) return;

                                    const trackId = trackIdFromClientY(e.clientY);
                                    if (!trackId) {
                                        setTrackAreaMenu(null);
                                        return;
                                    }

                                    const scroller = scrollRef.current;
                                    const bounds = scroller?.getBoundingClientRect() ?? null;
                                    const timeAtPointer =
                                        bounds && scroller
                                            ? beatFromClientX(
                                                  e.clientX,
                                                  bounds,
                                                  scroller.scrollLeft,
                                              )
                                            : null;

                                    if (timeAtPointer != null) {
                                        const clipsHere = sessionRef.current.clips
                                            .filter((c) => c.trackId === trackId)
                                            .filter((c) => {
                                                const start = Number(c.startSec ?? 0) || 0;
                                                const end = start + (Number(c.lengthSec ?? 0) || 0);
                                                return (
                                                    timeAtPointer >= start && timeAtPointer <= end
                                                );
                                            })
                                            .sort((a, b) => a.startSec - b.startSec);

                                        if (clipsHere.length > 0) {
                                            if (target?.closest?.("[data-hs-clip-item='1']"))
                                                return;

                                            const topClip = clipsHere[clipsHere.length - 1];
                                            setContextMenu({
                                                x: e.clientX,
                                                y: e.clientY,
                                                clipId: topClip.id,
                                                overlappingClipIds:
                                                    clipsHere.length > 1
                                                        ? clipsHere.map((c) => c.id)
                                                        : undefined,
                                            });
                                            return;
                                        }
                                    }

                                    if (sessionRef.current.selectedTrackId !== trackId) {
                                        void dispatch(selectTrackRemote(trackId));
                                    }
                                    setTrackAreaMenu({
                                        x: e.clientX,
                                        y: e.clientY,
                                        trackId,
                                        timeSec: timeAtPointer ?? 0,
                                    });
                                }}
                                onPointerDown={onSelectionRectPointerDown}
                                onDragOver={handleTimelineDragOver}
                                onDragLeave={(e) => {
                                    const related = e.relatedTarget as Node | null;
                                    if (
                                        related &&
                                        (e.currentTarget as HTMLDivElement).contains(related)
                                    )
                                        return;
                                    setDropPreview(null);
                                }}
                                onDrop={handleTimelineDrop}
                                onPointerDownCapture={(e) => {
                                    const scroller = scrollRef.current;
                                    if (
                                        scroller &&
                                        isPointerOnNativeScrollbar(scroller, e.clientX, e.clientY)
                                    ) {
                                        return;
                                    }
                                    if (e.button === 0) {
                                        const target = e.target as HTMLElement | null;
                                        // 任意空白处按下即取消 clip 选中：容器捕获先于所有
                                        // lane 处理器执行，保证"点击任意轨道的空白（含轨道区
                                        // 下方空白）都取消选中"。clip / overlap 层 / 标尺 /
                                        // 输入目标除外 —— 它们各自的路由决定选中的去向。
                                        if (
                                            !isEditableTarget(e.target) &&
                                            !target?.closest?.(
                                                "[data-hs-clip-item='1'],[data-hs-overlap-layer='1'],[data-hs-context-menu='1'],[data-hs-floating-menu='1']",
                                            )
                                        ) {
                                            deselectAllTrackLaneClips();
                                        }
                                        // 在 capture 阶段直接切换轨道：不依赖后续 mousedown，
                                        // 即使子元素在 pointerdown 里 preventDefault/停止冒泡，
                                        // “允许时间轴点击切换轨道”也能稳定触发。
                                        // applySelectedClip: false —— 点击切轨不得让后端把
                                        // 该轨道记住的 selected_clip_id 恢复回来，否则刚完成
                                        // 的空白取消选中会被异步覆盖（"点其他轨道空白不取消
                                        // 选中"的根因）。
                                        if (!isEditableTarget(e.target)) {
                                            const trackId = trackIdFromClientY(e.clientY);
                                            if (
                                                s.paramEditorTimelineClickSelectTrackEnabled &&
                                                trackId &&
                                                trackId !== sessionRef.current.selectedTrackId
                                            ) {
                                                void dispatch(
                                                    selectTrackRemote({
                                                        trackId,
                                                        applySelectedClip: false,
                                                    }),
                                                );
                                            }
                                        }
                                        return;
                                    }
                                    if (e.button !== 1) return;
                                    if (isEditableTarget(e.target)) return;
                                    e.preventDefault();
                                    startPanPointer(e);
                                }}
                                onMouseDown={(e) => {
                                    if (e.button !== 0) return;
                                    // 输入框/可编辑区域内的点击只负责文本光标，不应触发时间轴点击逻辑
                                    //（尤其不能在名称编辑框中点击时跳转播放头）。
                                    if (isEditableTarget(e.target)) return;
                                    // Guard scrollbar interactions first — avoid clearing
                                    // multi-selection when dragging the native scrollbar.
                                    const scroller = scrollRef.current;
                                    if (
                                        scroller &&
                                        isPointerOnNativeScrollbar(scroller, e.clientX, e.clientY)
                                    )
                                        return;
                                    setContextMenu(null);
                                    setTrackAreaMenu(null);
                                    setMultiSelectedClipIds([]);
                                    if (!scroller) return;
                                    const trackId = trackIdFromClientY(e.clientY);
                                    if (
                                        s.paramEditorTimelineClickSelectTrackEnabled &&
                                        trackId &&
                                        trackId !== sessionRef.current.selectedTrackId
                                    ) {
                                        // 同容器捕获路径：点击切轨不恢复后端记住的选中 clip。
                                        void dispatch(
                                            selectTrackRemote({
                                                trackId,
                                                applySelectedClip: false,
                                            }),
                                        );
                                    }
                                    startDeferredPlayheadSeek({
                                        startClientX: e.clientX,
                                        startClientY: e.clientY,
                                        getBounds: () => {
                                            const cur = scrollRef.current;
                                            return cur ? cur.getBoundingClientRect() : null;
                                        },
                                        getScrollLeft: () => {
                                            const cur = scrollRef.current;
                                            return cur ? cur.scrollLeft : scroller.scrollLeft;
                                        },
                                    });
                                }}
                            >
                                {/* Track Lanes（外层含右侧虚拟宽度，内容层覆盖工程宽 + 视口宽） */}
                                <div
                                    className="relative"
                                    style={{
                                        width: timelineScrollRange.paddedContentWidth,
                                        height: contentHeight,
                                    }}
                                >
                                    {/* 内容层宽度 = 工程宽 + 视口宽（= paddedContentWidth）：
                            拖拽预览 / 吸附竖线高亮 / 拖拽中的 clip 与 ghost / 选区框等
                            瞬态 UI 不再被“严格等于工程宽”的旧内容层按工程长度裁剪 ——
                            用户看到与操作到的轨道在可视与可操作范围内表现为无限延伸
                            （水平滚动的 maxScrollLeft 上限是有意保留的）。 */}
                                    <div
                                        className="absolute top-0 left-0 overflow-hidden"
                                        style={{
                                            width: timelineScrollRange.paddedContentWidth,
                                            height: contentHeight,
                                        }}
                                    >
                                        {selectionRect ? (
                                            <div
                                                className="absolute z-40 pointer-events-none"
                                                style={{
                                                    left: selectionRect.x1,
                                                    top: selectionRect.y1,
                                                    width: Math.max(
                                                        1,
                                                        selectionRect.x2 - selectionRect.x1,
                                                    ),
                                                    height: Math.max(
                                                        1,
                                                        selectionRect.y2 - selectionRect.y1,
                                                    ),
                                                    border: "1px dashed var(--qt-highlight)",
                                                    backgroundColor:
                                                        "color-mix(in oklab, var(--qt-highlight) 12%, transparent)",
                                                }}
                                            />
                                        ) : null}

                                        {clipDropNewTrack ? (
                                            <div
                                                className="absolute left-0 right-0 pointer-events-none z-20"
                                                style={{
                                                    top: s.tracks.length * rowHeight,
                                                    height: rowHeight,
                                                }}
                                            >
                                                <div
                                                    className="absolute inset-0"
                                                    style={{
                                                        border: "1px dashed var(--qt-highlight)",
                                                        backgroundColor:
                                                            "color-mix(in oklab, var(--qt-highlight) 10%, transparent)",
                                                    }}
                                                />
                                                {newTrackGhostClips.map((clip) => (
                                                    <div
                                                        key={`new-track-ghost-${clip.id}`}
                                                        className="absolute opacity-60"
                                                        style={{
                                                            left: Math.max(
                                                                0,
                                                                clip.startSec * pxPerSec,
                                                            ),
                                                            width: Math.max(
                                                                1,
                                                                clip.lengthSec * pxPerSec,
                                                            ),
                                                            top: 0,
                                                            height: rowHeight - 8,
                                                            paddingTop: 8,
                                                        }}
                                                    >
                                                        <div
                                                            className="absolute left-0 right-0 top-0 rounded-t-sm"
                                                            style={{
                                                                height: 18,
                                                                backgroundColor:
                                                                    "color-mix(in oklab, var(--qt-highlight) 55%, transparent)",
                                                            }}
                                                        />
                                                        <div
                                                            className="absolute left-0 right-0 bottom-0 rounded-sm border border-dashed border-white/70"
                                                            style={{
                                                                top: 18,
                                                                backgroundColor:
                                                                    "color-mix(in oklab, var(--qt-highlight) 20%, transparent)",
                                                            }}
                                                        />
                                                    </div>
                                                ))}
                                            </div>
                                        ) : null}

                                        <div
                                            className="absolute left-0 right-0"
                                            style={{
                                                top: timelineRenderModel.startIndex * rowHeight,
                                            }}
                                        >
                                            {visibleTracks.map((track) => {
                                                const trackClips =
                                                    visibleTrackClipsById[track.id] ??
                                                    ([] as typeof s.clips);

                                                return (
                                                    <TrackLane
                                                        key={track.id}
                                                        track={track}
                                                        allTracks={s.tracks}
                                                        trackClips={trackClips}
                                                        rowHeight={rowHeight}
                                                        pxPerSec={pxPerSec}
                                                        bpm={s.bpm}
                                                        viewportWidthPx={viewportWidth}
                                                        viewportStartSec={viewportStartSec}
                                                        viewportEndSec={viewportEndSec}
                                                        overlayClipIds={
                                                            sparseClipRenderModel
                                                                .overlayClipIdsByTrackId[
                                                                track.id
                                                            ] ?? []
                                                        }
                                                        altPressed={altPressed}
                                                        selectedClipId={
                                                            selectedClipTrackId === track.id
                                                                ? s.selectedClipId
                                                                : null
                                                        }
                                                        multiSelectedClipIds={multiSelectedClipIds}
                                                        multiSelectedSet={multiSelectedSet}
                                                        trackColor={track.color || undefined}
                                                        ensureSelected={ensureTrackLaneSelected}
                                                        selectClipRemote={selectTrackLaneClipRemote}
                                                        deselectAllClips={deselectAllTrackLaneClips}
                                                        onShiftRangeSelect={selectClipRangeByRect}
                                                        rangeSelectAnchorClipId={
                                                            rangeSelectAnchorClipId
                                                        }
                                                        recordLastClickPosition={
                                                            recordLastClickPosition
                                                        }
                                                        openContextMenu={openTrackLaneContextMenu}
                                                        seekFromClientX={seekFromTrackLaneClientX}
                                                        ghostDrag={ghostDrag}
                                                        verticalTrackLockTrackId={
                                                            verticalTrackLockTrackId
                                                        }
                                                        allClips={s.clips}
                                                        showAllTakes={s.showAllTakes}
                                                        onActivateTake={activateTrackLaneTake}
                                                        fadeShapeCycleKb={fadeShapeCycleKb}
                                                        multiSelectToggleKb={
                                                            clipMultiSelectToggleKb
                                                        }
                                                        rangeSelectKb={clipRangeSelectKb}
                                                        pitchDragKb={pitchDragKb}
                                                        onClipPitchDragStart={startClipPitchDrag}
                                                        fadeLengthFormatCtx={fadeLengthFormatCtx}
                                                        onFadeShapeCycleClick={
                                                            handleFadeShapeCycleClick
                                                        }
                                                        onCrossfadeCycleClick={
                                                            handleCrossfadeCycleClick
                                                        }
                                                        startClipDrag={startClipDrag}
                                                        startEditDrag={startEditDrag}
                                                        startSnapOffsetDrag={startSnapOffsetDrag}
                                                        toggleClipMuted={toggleTrackLaneClipMuted}
                                                        onCtrlToggleSelect={
                                                            toggleTrackLaneCtrlSelection
                                                        }
                                                        clearContextMenu={clearContextMenu}
                                                        toggleMultiSelect={
                                                            toggleTrackLaneMultiSelect
                                                        }
                                                        renamingClipId={renamingClipId}
                                                        onRenameStart={
                                                            clipActions.setRenamingClipId
                                                        }
                                                        onRenameClickCandidate={
                                                            registerRenameClickCandidate
                                                        }
                                                        onRenameCommit={commitTrackLaneRename}
                                                        onRenameDone={handleTrackLaneRenameDone}
                                                        onGainCommit={commitTrackLaneGain}
                                                        editingBadge={editingBadge}
                                                        onBadgeEditStart={startTrackLaneBadgeEdit}
                                                        onBadgeEditCommit={commitTrackLaneBadgeEdit}
                                                        onBadgeEditDone={handleBadgeEditDone}
                                                        onRateBadgeMenu={openRateBadgeMenu}
                                                        onFormantMorphCommit={
                                                            commitTrackLaneFormantMorph
                                                        }
                                                        activeGroupIds={activeGroupIds}
                                                        disabledGroupIds={disabledGroupIds}
                                                        onToggleGroupDisabled={
                                                            handleToggleGroupDisabled
                                                        }
                                                    />
                                                );
                                            })}
                                        </div>

                                        {/* 吸附竖线高亮层：拖拽手势中高亮吸附对象与被吸附对象 */}
                                        <SnapHighlightLayer
                                            pxPerSec={pxPerSec}
                                            rowHeight={rowHeight}
                                            tracks={s.tracks}
                                            contentHeight={contentHeight}
                                        />

                                        {/* Playhead 已移入 TimelineSurface sticky 层：与网格/Clip/
                                波形在同一滚动事件内更新，避免 DOM 原生层与 sticky 层错帧。 */}
                                    </div>

                                    {/* Drop preview (ghost item)。
                            渲染在外层 padded 容器内（同一坐标原点）：预览宽度超出
                            工程右缘时仍完整显示 —— 拖入比工程剩余更长或更靠右的
                            媒体时，预览与实际导入一样不受“工程长度”限制。 */}
                                    {dropPreview ? (
                                        <div
                                            ref={dropPreviewRef}
                                            className="absolute z-30 pointer-events-none"
                                            style={{
                                                left: Math.max(0, dropPreview.startSec * pxPerSec),
                                                top: rowTopForTrackId(dropPreview.trackId) + 8,
                                                width:
                                                    dropPreview.durationSec > 0
                                                        ? Math.max(
                                                              1,
                                                              pxPerSec * dropPreview.durationSec,
                                                          )
                                                        : 80,
                                                height: rowHeight - 16,
                                            }}
                                        >
                                            <div className="h-full w-full rounded-sm border border-dashed border-qt-highlight bg-[color-mix(in_oklab,var(--qt-highlight)_20%,transparent)]">
                                                <div className="px-2 pt-1 text-[10px] text-qt-text truncate">
                                                    {dropPreview.fileName}
                                                </div>
                                            </div>
                                        </div>
                                    ) : null}

                                    {viewportWidth > 0 ? (
                                        /* 背景网格 / Clip 体 / 波形面全部锚定在同一 sticky 视口层：
                               滚动时三者经同一条同步链（scroll 事件内）提交位移，任一
                               层都不允许再走 React state / rAF，否则会与其它层分裂。 */
                                        <TimelineSurface
                                            tracks={visibleTracks}
                                            startTrackIndex={timelineRenderModel.startIndex}
                                            clipsByTrackId={visibleTrackClipsById}
                                            rowHeight={rowHeight}
                                            widthPx={Math.max(1, Math.ceil(viewportWidth))}
                                            heightPx={visibleTrackCanvasHeight}
                                            topPx={0}
                                            axis={timelineAxis}
                                            playheadSec={s.playheadSec}
                                            clipModel={timelineCanvasModel}
                                            darkMode={darkMode}
                                            contentWidth={contentWidth}
                                            pxPerBeat={pxPerBeat}
                                            grid={s.grid}
                                            beatsPerBar={Math.max(1, Math.round(s.beats || 4))}
                                            gridVisible={s.timelineSnap.gridVisible}
                                            gridMinSpacingPx={s.timelineSnap.gridMinSpacingPx}
                                            gridSwingPercent={
                                                s.timelineSnap.swingEnabled
                                                    ? s.timelineSnap.swingPercent
                                                    : 0
                                            }
                                            ticks={timelineTicks}
                                            gridBottomPx={trackGridHeight}
                                            gridOverlayLayerRef={trackGridOverlayLayerRef}
                                            playheadLineRef={playheadRef}
                                        />
                                    ) : null}
                                </div>
                            </TimelineScrollArea>
                        </>
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

                              const _ctxScroller = scrollRef.current;
                              const _ctxBounds = _ctxScroller?.getBoundingClientRect();
                              const contextTimeSec =
                                  _ctxBounds && _ctxScroller
                                      ? beatFromClientX(
                                            contextMenu.x,
                                            _ctxBounds,
                                            _ctxScroller.scrollLeft,
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
                        playheadRef={playheadRef}
                        rulerPlayheadLineRef={rulerPlayheadLineRef}
                        rulerPlayheadHeadRef={rulerPlayheadHeadRef}
                        scrollRef={scrollRef}
                        visualPlayheadRef={visualPlayheadSecRef}
                        syncScrollLeft={syncScrollLeft}
                        autoScrollEnabled={s.autoScrollEnabled}
                        projectSec={dynamicProjectSec}
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
