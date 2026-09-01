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
 */
import React, { useMemo } from "react";
import { Flex, Dialog, Button, Text } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppTheme } from "../../theme/AppThemeProvider";
import { useAppSelector } from "../../app/hooks";
import { shallowEqual } from "react-redux";
import { selectKeybinding } from "../../features/keybindings/keybindingsSlice";
import { defaultFadeDirFor, FADE_PRESETS } from "./timeline/reaperFade";
import type { FadeLengthFormatContext } from "./timeline/fadeTooltipText";
import { FadeContextMenuHost } from "./timeline/FadeContextMenuHost";
import { createPortal } from "react-dom";
import {
    addTrackRemote,
    closeClipFormantToolWindow,
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
    persistUiSettings,
    setPrimaryTimeUnit,
    setSecondaryTimeUnit,
    setTempoMap,
    setTrackName,
    setTrackVolume,
    setPendingPlayheadReveal,
} from "../../features/session/sessionSlice";
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
import { SNAP_HIGHLIGHT_GROUP, clearSnapHighlights } from "../../utils/snapHighlight";
import type { TempoMap } from "../../utils/tempoMap";
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
import {
    computeAutoFollowScrollLeft,
    computeFocusCursorScrollLeft,
} from "../../utils/autoFollowScroll";
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
        syncScrollLeft,
        autoScrollEnabled,
        projectSec,
    } = props;
    const transport = useAppSelector(
        (state) => ({
            playheadSec: state.session.playheadSec,
            isPlaying: state.session.runtime.isPlaying,
            playbackPositionSec: state.session.runtime.playbackPositionSec,
        }),
        // 无 shallowEqual 时每次 dispatch 都产生新对象引用，
        // 该桥接组件会在任意 store 更新（含 33Hz 播放轮询）时重渲染。
        shallowEqual,
    );

    const isTransportAdvancing = transport.isPlaying && transport.playbackPositionSec > 1e-4;

    useVisualPlayhead({
        syncedPlayheadSec: transport.playheadSec,
        isTransportAdvancing,
        onFrame: React.useCallback(
            (visualPlayheadSec: number) => {
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
                const scroller = scrollRef.current;
                const screenLeft = playheadLeftPx - (scroller?.scrollLeft ?? 0);
                if (playheadRef.current) {
                    playheadRef.current.style.left = `${screenLeft}px`;
                }
                if (rulerPlayheadLineRef.current) {
                    rulerPlayheadLineRef.current.style.left = `${playheadLeftPx}px`;
                }
                if (rulerPlayheadHeadRef.current) {
                    rulerPlayheadHeadRef.current.style.left = `${playheadLeftPx}px`;
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
    // 时间轴 scroller 水平滚动条的占用高度（offsetHeight - clientHeight）。
    // 轨道头底部按此留出同高占位（bottomGutterHeightPx），保证轨道头与
    // 时间轴区域的竖直滚动范围严格一致。
    const [horizontalScrollbarGutterPx, setHorizontalScrollbarGutterPx] = React.useState(0);
    const [quickExportDialog, setQuickExportDialog] = React.useState<{
        open: boolean;
        clipIds: string[];
    }>({ open: false, clipIds: [] });

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
        viewportStartSec,
        viewportEndSec,
        scrollHorizontalKb,
        scrollVerticalKb,
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
        isEditableTarget,
        isPointerOnNativeScrollbar,
        startPanPointer,
        setPlayheadFromClientX,
        startDeferredPlayheadSeek,
        keyboardZoomPendingRef,
    } = state;

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
    } = clipActions;
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

    // ── 4. 全局事件监听 ─────────────────────────────────────
    useTimelineEventHandlers({
        dispatch,
        sessionRef,
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
        isEditableTarget,
        contextMenu,
        trackAreaMenu,
        setContextMenu,
        setTrackAreaMenu,
        syncScrollLeft,
        dynamicProjectSec,
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
                rowHeight,
                scrollTopPx: timelineScrollTop,
                viewportHeightPx: scrollRef.current?.clientHeight ?? 0,
            }),
        [
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

    return (
        <Flex className="h-full w-full bg-qt-graph-bg overflow-hidden">
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
                {/* playheadSec 传提交值（而非渲染期读 ref）：视觉插值由
                    playheadLineRef/playheadHeadRef 命令式驱动；React 仅在该值
                    真正变化时重写 style.left，写入的是最新提交位置而非陈旧值。 */}
                <TimeRuler
                    scrollLeft={scrollLeft}
                    ticks={timelineTicks}
                    pxPerBeat={pxPerBeat}
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
                        s.project.useCustomScale
                            ? (s.project.customScale?.name ?? undefined)
                            : undefined
                    }
                    fallbackDenominator={s.project.timeSignatureDenominator}
                    customScalePresets={s.customScalePresets}
                    onTempoMapChange={handleTempoMapChange}
                    onTempoMapCommit={handleTempoMapCommit}
                    onMouseDown={(e) => {
                        if (e.button !== 0) return;
                        document.body.setAttribute("data-hs-focus-window", "timeline");
                        const scroller = scrollRef.current;
                        if (!scroller) return;
                        const ruler = e.currentTarget as HTMLDivElement;
                        let moved = false;
                        let lastClientX = e.clientX;
                        let lastSec = 0;

                        const updateAt = (clientX: number, commit: boolean): number =>
                            setPlayheadFromClientX(
                                clientX,
                                ruler.getBoundingClientRect(),
                                scroller.scrollLeft,
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
                    commitScrollTopState={setTimelineScrollTop}
                    rulerContentRef={rulerContentRef}
                    scrollHorizontalKb={scrollHorizontalKb}
                    scrollVerticalKb={scrollVerticalKb}
                    horizontalZoomKb={horizontalZoomKb}
                    verticalZoomKb={verticalZoomKb}
                    getPlayheadSec={() => Number(sessionRef.current.playheadSec ?? 0) || 0}
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
                        setTimelineScrollTop(el.scrollTop);
                        if (trackListScrollRef.current) {
                            if (
                                Math.abs(trackListScrollRef.current.scrollTop - el.scrollTop) >= 0.5
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
                                ? beatFromClientX(e.clientX, bounds, scroller.scrollLeft)
                                : null;

                        if (timeAtPointer != null) {
                            const clipsHere = sessionRef.current.clips
                                .filter((c) => c.trackId === trackId)
                                .filter((c) => {
                                    const start = Number(c.startSec ?? 0) || 0;
                                    const end = start + (Number(c.lengthSec ?? 0) || 0);
                                    return timeAtPointer >= start && timeAtPointer <= end;
                                })
                                .sort((a, b) => a.startSec - b.startSec);

                            if (clipsHere.length > 0) {
                                if (target?.closest?.("[data-hs-clip-item='1']")) return;

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
                        });
                    }}
                    onPointerDown={onSelectionRectPointerDown}
                    onDragOver={(e) => {
                        const dt = e.dataTransfer;
                        const tauriPath = tauriDraggedPathRef.current;
                        const hasDomFile = Boolean(dt?.files && dt.files.length > 0);
                        const isTauri = Boolean(
                            (window as unknown as { __TAURI__?: unknown }).__TAURI__,
                        );
                        if (!isTauri && !hasFileDrag(dt) && !hasDomFile && !tauriPath) return;
                        e.preventDefault();
                        const info = extractLocalFilePath(dt);
                        const el = e.currentTarget as HTMLDivElement;
                        const bounds = el.getBoundingClientRect();
                        const beat = beatFromClientX(e.clientX, bounds, el.scrollLeft);
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
                    }}
                    onDragLeave={(e) => {
                        const related = e.relatedTarget as Node | null;
                        if (related && (e.currentTarget as HTMLDivElement).contains(related))
                            return;
                        setDropPreview(null);
                    }}
                    onDrop={(e) => {
                        const dt = e.dataTransfer;
                        const tauriPath = tauriDraggedPathRef.current;
                        const lastTauriDropPath = tauriLastDropPathRef.current;
                        const hasDomFile = Boolean(dt?.files && dt.files.length > 0);
                        const isTauri = Boolean(
                            (window as unknown as { __TAURI__?: unknown }).__TAURI__,
                        );
                        if (!isTauri && !hasFileDrag(dt) && !hasDomFile && !tauriPath) return;
                        e.preventDefault();

                        if (isTauri && Date.now() - (tauriDropHandledAtRef.current || 0) < 500) {
                            setDropPreview(null);
                            return;
                        }

                        const info = extractLocalFilePath(dt);
                        const el = e.currentTarget as HTMLDivElement;
                        const bounds = el.getBoundingClientRect();
                        const beat = beatFromClientX(e.clientX, bounds, el.scrollLeft);
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
                                const p =
                                    tauriLastDropPathRef.current || tauriDraggedPathRef.current;
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
                    }}
                    onPointerDownCapture={(e) => {
                        document.body.setAttribute("data-hs-focus-window", "timeline");
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
                        if (scroller && isPointerOnNativeScrollbar(scroller, e.clientX, e.clientY))
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
                            void dispatch(selectTrackRemote({ trackId, applySelectedClip: false }));
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
                                        width: Math.max(1, selectionRect.x2 - selectionRect.x1),
                                        height: Math.max(1, selectionRect.y2 - selectionRect.y1),
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
                                                left: Math.max(0, clip.startSec * pxPerSec),
                                                width: Math.max(1, clip.lengthSec * pxPerSec),
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
                                        visibleTrackClipsById[track.id] ?? ([] as typeof s.clips);

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
                                                sparseClipRenderModel.overlayClipIdsByTrackId[
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
                                            rangeSelectAnchorClipId={rangeSelectAnchorClipId}
                                            recordLastClickPosition={recordLastClickPosition}
                                            openContextMenu={openTrackLaneContextMenu}
                                            seekFromClientX={seekFromTrackLaneClientX}
                                            ghostDrag={ghostDrag}
                                            verticalTrackLockTrackId={verticalTrackLockTrackId}
                                            allClips={s.clips}
                                            showAllTakes={s.showAllTakes}
                                            onActivateTake={activateTrackLaneTake}
                                            fadeShapeCycleKb={fadeShapeCycleKb}
                                            multiSelectToggleKb={clipMultiSelectToggleKb}
                                            rangeSelectKb={clipRangeSelectKb}
                                            pitchDragKb={pitchDragKb}
                                            onClipPitchDragStart={startClipPitchDrag}
                                            fadeLengthFormatCtx={fadeLengthFormatCtx}
                                            onFadeShapeCycleClick={handleFadeShapeCycleClick}
                                            onCrossfadeCycleClick={handleCrossfadeCycleClick}
                                            startClipDrag={startClipDrag}
                                            startEditDrag={startEditDrag}
                                            startSnapOffsetDrag={startSnapOffsetDrag}
                                            toggleClipMuted={toggleTrackLaneClipMuted}
                                            onCtrlToggleSelect={toggleTrackLaneCtrlSelection}
                                            clearContextMenu={clearContextMenu}
                                            toggleMultiSelect={toggleTrackLaneMultiSelect}
                                            renamingClipId={renamingClipId}
                                            onRenameStart={clipActions.setRenamingClipId}
                                            onRenameClickCandidate={registerRenameClickCandidate}
                                            onRenameCommit={commitTrackLaneRename}
                                            onRenameDone={handleTrackLaneRenameDone}
                                            onGainCommit={commitTrackLaneGain}
                                            onFormantMorphCommit={commitTrackLaneFormantMorph}
                                            activeGroupIds={activeGroupIds}
                                            disabledGroupIds={disabledGroupIds}
                                            onToggleGroupDisabled={handleToggleGroupDisabled}
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

                            {s.clipFormantToolWindow.open && activeFormantToolClip ? (
                                <ClipFormantToolWindow
                                    clip={activeFormantToolClip}
                                    status={
                                        s.clipFormantStatus[activeFormantToolClip.id] ?? "ready"
                                    }
                                    x={s.clipFormantToolWindow.x}
                                    y={s.clipFormantToolWindow.y}
                                    onCommit={commitTrackLaneFormantMorph}
                                    onMove={(x, y) =>
                                        dispatch(setClipFormantToolWindowPosition({ x, y }))
                                    }
                                    onClose={() => dispatch(closeClipFormantToolWindow())}
                                />
                            ) : null}

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
                                            ? Math.max(1, pxPerSec * dropPreview.durationSec)
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
                                    s.timelineSnap.swingEnabled ? s.timelineSnap.swingPercent : 0
                                }
                                ticks={timelineTicks}
                                gridBottomPx={trackGridHeight}
                                gridOverlayLayerRef={trackGridOverlayLayerRef}
                                playheadLineRef={playheadRef}
                            />
                        ) : null}
                    </div>
                </TimelineScrollArea>

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
                            style={{ left: projectActionMenu.x, top: projectActionMenu.y }}
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
                                                  ? { fadeInShape: shape, fadeInDir: dir }
                                                  : { fadeOutShape: shape, fadeOutDir: dir }),
                                          }),
                                      );
                                      void dispatch(
                                          setClipStateRemote({
                                              clipId,
                                              ...(target === "in"
                                                  ? { fadeInShape: shape, fadeInDir: dir }
                                                  : { fadeOutShape: shape, fadeOutDir: dir }),
                                          }),
                                      );
                                  }}
                                  onNormalize={normalizeClips}
                                  onToggleReverse={(ids, reversed) => {
                                      // 批量走 bulk 通道：单次 IPC + 单个撤销步
                                      //（逐个 setClipStateRemote 会产生 N 次 IPC/N 步撤销）。
                                      void dispatch(
                                          setClipsStateBulkRemote({
                                              updates: ids.map((id) => ({ clipId: id, reversed })),
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
                                  const clip = sessionRef.current.clips.find((c) => c.id === id);
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
                              onPaste={pasteClipsAtPlayhead}
                              onSplit={splitSelectedAtPlayhead}
                              onClose={() => setTrackAreaMenu(null)}
                          />,
                          document.body,
                      )
                    : null}

                <QuickClipExportDialog
                    open={quickExportDialog.open}
                    clipIds={quickExportDialog.clipIds}
                    onOpenChange={(open) =>
                        setQuickExportDialog((prev) => (open ? prev : { open: false, clipIds: [] }))
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
                            setReplaceMidiDialog({ open: false, clipId: null, midiPath: null });
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
                    syncScrollLeft={syncScrollLeft}
                    autoScrollEnabled={s.autoScrollEnabled}
                    projectSec={dynamicProjectSec}
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
    );
};
