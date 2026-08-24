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
import { useAppSelector } from "../../app/hooks";
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
} from "../../features/session/sessionSlice";
import { setTempoMapRemote } from "../../features/session/thunks/tempoMapThunks";

import { NEW_TRACK_SENTINEL, useClipDrag } from "./timeline/hooks/useClipDrag";
import { useEditDrag } from "./timeline/hooks/useEditDrag";
import { useSlipDrag } from "./timeline/hooks/useSlipDrag";
import { getInsertBelowTargetIndex } from "./timeline/trackContextMenuPlacement";
import { collectFadeContextClips } from "./timeline/clipFadeContext";
import { emitExternalFileAction } from "../../features/session/projectOpenEvents";
import { webApi } from "../../services/webviewApi";
import { coreApi } from "../../services/api/core";
import { paramsApi } from "../../services/api/params";
import { resolveRootTrackId } from "../../features/session/trackUtils";
import { SCALE_NOTES } from "../../utils/musicalScales";
import { QuickClipExportDialog } from "./QuickClipExportDialog";
import { MidiTrackSelectDialog } from "./MidiTrackSelectDialog";

import {
    BackgroundGrid,
    ClipContextMenu,
    TRACK_ADD_ROW_HEIGHT,
    TrackAreaContextMenu,
    TimelineCanvasViewport,
    TimelineScrollArea,
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
import type { TempoMap } from "../../utils/tempoMap";
import type { ScaleLike } from "../../utils/musicalScales";
import { TimelineDisplaySettingsDialog } from "./TimelineDisplaySettingsDialog";
import { resolveTimelineScrollRange } from "./timeline/runtime/timelineScrollRange";

// ── 拆分出的 hooks ──────────────────────────────────────────
import { useTimelineState } from "./timeline/hooks/useTimelineState";
import { useTimelineDragDrop } from "./timeline/hooks/useTimelineDragDrop";
import { useTimelineClipActions } from "./timeline/hooks/useTimelineClipActions";
import { useTimelineEventHandlers } from "./timeline/hooks/useTimelineEventHandlers";
import { expandClipIdsWithGroups } from "./timeline/hooks/useGroupExpansion";
import { useVisualPlayhead } from "../../hooks/useVisualPlayhead";
import { computeAutoFollowScrollLeft } from "../../utils/autoFollowScroll";
import { buildSparseClipRenderModel } from "./timeline/runtime/timelineCanvasModel";
import { buildTimelineRenderModel } from "./timeline/runtime/timelineRenderModel";
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
    const transport = useAppSelector((state) => ({
        playheadSec: state.session.playheadSec,
        isPlaying: state.session.runtime.isPlaying,
        playbackPositionSec: state.session.runtime.playbackPositionSec,
    }));

    const isTransportAdvancing = transport.isPlaying && transport.playbackPositionSec > 1e-4;

    useVisualPlayhead({
        syncedPlayheadSec: transport.playheadSec,
        isTransportAdvancing,
        onFrame: React.useCallback(
            (visualPlayheadSec: number) => {
                const playheadLeftPx = visualPlayheadSec * pxPerSecRef.current;
                if (playheadRef.current) {
                    playheadRef.current.style.left = `${playheadLeftPx}px`;
                }
                if (rulerPlayheadLineRef.current) {
                    rulerPlayheadLineRef.current.style.left = `${playheadLeftPx}px`;
                }
                if (rulerPlayheadHeadRef.current) {
                    rulerPlayheadHeadRef.current.style.left = `${playheadLeftPx}px`;
                }
                if (!autoScrollEnabled || !transport.isPlaying) return;
                const scroller = scrollRef.current;
                if (!scroller) return;
                const next = computeAutoFollowScrollLeft({
                    playheadSec: visualPlayheadSec,
                    pxPerSec: pxPerSecRef.current,
                    viewportWidth: scroller.clientWidth,
                    contentWidth: projectSec * pxPerSecRef.current,
                });
                if (Math.abs(scroller.scrollLeft - next) <= 0.5) return;
                scroller.scrollLeft = next;
                syncScrollLeft(next);
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
    const ignoreGrouping = useAppSelector((state) => state.session.ignoreGrouping);
    const disabledGroupIds = useAppSelector((state) => state.session.disabledGroupIds);
    const rulerPlayheadLineRef = React.useRef<HTMLDivElement | null>(null);
    const rulerPlayheadHeadRef = React.useRef<HTMLDivElement | null>(null);
    // 双击名称的第一次点击会把播放头移动到点击位置，第二次点击可能落在播放头线上。
    // 记录名称区域的第一次点击，让播放头在短时间内收到同位置点击时转而进入重命名。
    const renameClickCandidateRef = React.useRef<ClipRenameClickCandidate | null>(null);
    const suppressPlayheadMouseDownRef = React.useRef(false);
    const registerRenameClickCandidate = React.useCallback(
        (candidate: ClipRenameClickCandidate | null) => {
            renameClickCandidateRef.current = candidate;
        },
        [],
    );
    const [timelineScrollTop, setTimelineScrollTop] = React.useState(0);
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

    // ── 1. State / refs / viewport / scroll / 坐标转换 ──────
    const state = useTimelineState();
    const {
        dispatch,
        s,
        sessionRef,
        scrollRef,
        trackListScrollRef,
        rulerContentRef,
        playheadRef,
        dropPreviewRef,
        playheadDragRef,
        lastClickedClipIdRef,
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
        secPerBeat,
        pxPerBeat,
        contentWidth,
        contentHeight,
        dynamicProjectSec,
        ticks,
        tempoGridLineXs,
        viewportStartSec,
        viewportEndSec,
        scrollHorizontalKb,
        scrollVerticalKb,
        horizontalZoomKb,
        verticalZoomKb,
        paramFineAdjustKb,
        slipEditKb,
        noSnapKb,
        copyDragKb,
        crossfadeGripKb,
        dropPreview,
        setDropPreview,
        clipDropNewTrack,
        setClipDropNewTrack,
        pendingDropDurationPathRef,
        syncScrollLeft,
        setScrollLeftAction,
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

    const timeContext = React.useMemo<TimeFormatContext>(
        () => ({
            bpm: s.bpm,
            beatsPerBar: Math.max(1, Math.round(s.beats || 4)),
            grid: s.grid,
            tempoMap: s.tempoMap,
        }),
        [s.bpm, s.beats, s.grid, s.tempoMap],
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
        copyTracks,
        cutTracks,
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
        openTrackLaneContextMenu,
        seekFromTrackLaneClientX,
        toggleTrackLaneClipMuted,
        toggleTrackLaneCtrlSelection,
        toggleTrackLaneMultiSelect,
        commitTrackLaneRename,
        handleTrackLaneRenameDone,
        commitTrackLaneGain,
    } = clipActions;
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
        rowHeight,
        multiSelectedClipIds,
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
    const { editDragRef: _editDragRef, startEditDrag } = useEditDrag({
        scrollRef,
        sessionRef,
        dispatch,
        multiSelectedClipIds,
        multiSelectedSet,
        snapTimeline,
        beatFromClientX,
        noSnapKb,
        snapEnabled: s.timelineSnap.enabled,
        timelineSnap: s.timelineSnap,
        pxPerSec,
        ignoreGrouping,
        paramFineAdjustKb,
        crossfadeGripKb,
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

    const {
        clipDragRef: _clipDragRef,
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
        snapTimeline,
        beatFromClientX,
        trackIdFromClientY,
        setClipDropNewTrack,
        setMultiSelectedClipIds,
        slipEditKb,
        noSnapKb,
        snapEnabled: s.timelineSnap.enabled,
        copyDragKb,
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
    const handleTrackVolumeUiChange = React.useCallback((trackId: string, nextVolume: number) => {
        setTrackVolumeUi((prev) => ({
            ...prev,
            [trackId]: nextVolume,
        }));
    }, []);
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
        [dispatch],
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
        const timelineScroller = scrollRef.current;
        if (!timelineScroller) return;
        if (Math.abs(timelineScroller.scrollTop - scrollTop) < 0.5) return;
        timelineScroller.scrollTop = scrollTop;
    }, []);

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
        [rowHeight, s.clips, s.tracks, timelineScrollTop, viewportEndSec, viewportStartSec],
    );
    const visibleTracks = s.tracks.slice(
        timelineRenderModel.startIndex,
        timelineRenderModel.endIndex + 1,
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
        return nextByTrackId;
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
    const sparseClipRenderModel = useMemo(
        () =>
            buildSparseClipRenderModel({
                visibleTracks,
                visibleTrackClipsById,
                pxPerSec,
                rowHeight,
                scrollLeft,
                selectedClipId: s.selectedClipId,
                multiSelectedClipIds,
                renamingClipId,
                disabledGroupIds,
            }),
        [
            multiSelectedClipIds,
            pxPerSec,
            renamingClipId,
            rowHeight,
            s.selectedClipId,
            scrollLeft,
            visibleTrackClipsById,
            visibleTracks,
        ],
    );
    const timelineCanvasModel = useMemo(
        () => ({
            drawClips: sparseClipRenderModel.drawClips,
            activeGroupIds,
            disabledGroupIds,
        }),
        [sparseClipRenderModel.drawClips, activeGroupIds, disabledGroupIds],
    );
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
                onCopyTrack={(trackId) => void copyTracks([trackId])}
                onCutTrack={(trackId) => cutTracks([trackId])}
                onCreateTrackBelow={handleCreateTrackBelow}
                onScrollTopChange={handleTrackListScrollTopChange}
                headerHeight={timeRulerHeightPx(
                    Boolean(s.tempoMap && s.tempoMap.points.length > 0 && s.tempoMapVisible),
                )}
            />

            {/* Timeline View (Right) */}
            <Flex direction="column" className="flex-1 relative overflow-hidden bg-qt-graph-bg">
                <TimeRuler
                    contentWidth={contentWidth}
                    scrollLeft={scrollLeft}
                    ticks={ticks}
                    pxPerBeat={pxPerBeat}
                    pxPerSec={pxPerSec}
                    secPerBeat={secPerBeat}
                    viewportWidth={viewportWidth}
                    playheadSec={Number(sessionRef.current.playheadSec ?? 0) || 0}
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
                        document.body.setAttribute("data-hs-focus-window", "timeline");
                        const scroller = scrollRef.current;
                        if (!scroller) return;
                        const ruler = e.currentTarget as HTMLDivElement;
                        let moved = false;
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
                            lastSec = updateAt(ev.clientX, false);
                        };

                        const onEnd = (ev: MouseEvent) => {
                            window.removeEventListener("mousemove", onMove, true);
                            window.removeEventListener("mouseup", onEnd, true);
                            window.removeEventListener("mouseleave", onEnd, true);
                            if (!moved) return;
                            lastSec = updateAt(ev.clientX, false);
                            void dispatch(seekPlayhead(lastSec));
                        };

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
                            void dispatch(selectTrackRemote(trackId));
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
                    {/* Track Lanes（外层含右侧虚拟宽度，内容层严格等于工程宽） */}
                    <div
                        className="relative"
                        style={{
                            width: timelineScrollRange.paddedContentWidth,
                            height: contentHeight,
                        }}
                    >
                        {/* 内容层宽度严格等于工程宽：右侧虚拟宽度只用于滚动，不渲染任何内容 */}
                        <div
                            className="absolute top-0 left-0 overflow-hidden"
                            style={{ width: contentWidth, height: contentHeight }}
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

                            <BackgroundGrid
                                contentWidth={contentWidth}
                                contentHeight={trackGridHeight}
                                pxPerBeat={pxPerBeat}
                                grid={s.grid}
                                beatsPerBar={Math.max(1, Math.round(s.beats || 4))}
                                viewportWidth={viewportWidth}
                                scrollLeft={scrollLeft}
                                visible={s.timelineSnap.gridVisible}
                                minSpacingPx={s.timelineSnap.gridMinSpacingPx}
                                swingPercent={
                                    s.timelineSnap.swingEnabled ? s.timelineSnap.swingPercent : 0
                                }
                                weakLineXs={tempoGridLineXs?.weak ?? null}
                                strongLineXs={tempoGridLineXs?.strong ?? null}
                            />

                            {viewportWidth > 0 ? (
                                <div
                                    className="absolute pointer-events-none"
                                    style={{
                                        top: timelineRenderModel.startIndex * rowHeight,
                                        left: scrollLeft,
                                        width: viewportWidth,
                                        height: visibleTrackCanvasHeight,
                                        zIndex: 1,
                                    }}
                                >
                                    <TimelineCanvasViewport
                                        width={Math.max(1, Math.ceil(viewportWidth))}
                                        height={visibleTrackCanvasHeight}
                                        model={timelineCanvasModel}
                                    />
                                </div>
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
                                            onShiftRangeSelect={selectClipRangeByRect}
                                            rangeSelectAnchorClipId={rangeSelectAnchorClipId}
                                            recordLastClickPosition={recordLastClickPosition}
                                            openContextMenu={openTrackLaneContextMenu}
                                            seekFromClientX={seekFromTrackLaneClientX}
                                            ghostDrag={ghostDrag}
                                            verticalTrackLockTrackId={verticalTrackLockTrackId}
                                            allClips={s.clips}
                                            startClipDrag={startClipDrag}
                                            startEditDrag={startEditDrag}
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
                                            onToggleGroupDisabled={(groupId) => {
                                                toggleGroupDisabled(groupId);
                                            }}
                                        />
                                    );
                                })}
                            </div>

                            <div className="absolute inset-0 pointer-events-none z-[12]">
                                <BackgroundGrid
                                    contentWidth={contentWidth}
                                    contentHeight={trackGridHeight}
                                    pxPerBeat={pxPerBeat}
                                    grid={s.grid}
                                    beatsPerBar={Math.max(1, Math.round(s.beats || 4))}
                                    viewportWidth={viewportWidth}
                                    scrollLeft={scrollLeft}
                                    visible={s.timelineSnap.gridVisible}
                                    minSpacingPx={s.timelineSnap.gridMinSpacingPx}
                                    swingPercent={
                                        s.timelineSnap.swingEnabled
                                            ? s.timelineSnap.swingPercent
                                            : 0
                                    }
                                    lineOpacity={0.38}
                                    showBoundary={false}
                                    weakLineXs={tempoGridLineXs?.weak ?? null}
                                    strongLineXs={tempoGridLineXs?.strong ?? null}
                                />
                            </div>

                            <div
                                className="absolute left-0 right-0 pointer-events-none z-10"
                                style={{
                                    top: contentHeight - TRACK_ADD_ROW_HEIGHT,
                                    height: TRACK_ADD_ROW_HEIGHT,
                                    backgroundColor: "var(--qt-graph-bg)",
                                }}
                            />

                            {/* Drop preview (ghost item) */}
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

                            {/* Playhead Cursor */}
                            <div
                                ref={playheadRef}
                                className="absolute top-0 bottom-0 w-px bg-qt-playhead z-20 cursor-ew-resize"
                                style={{
                                    left: (Number(s.playheadSec ?? 0) || 0) * pxPerSec,
                                }}
                                onMouseDown={(e) => {
                                    if (!suppressPlayheadMouseDownRef.current) return;
                                    suppressPlayheadMouseDownRef.current = false;
                                    e.preventDefault();
                                    e.stopPropagation();
                                }}
                                onPointerDown={(e) => {
                                    if (e.button !== 0) return;

                                    // 双击名称的第二次点击：第一次点击已把播放头移到这里，
                                    // 因此本次 pointerdown 会命中播放头而不是名称 div。
                                    const candidate = renameClickCandidateRef.current;
                                    const now = performance.now();
                                    if (
                                        candidate &&
                                        candidate.pointerId === e.pointerId &&
                                        Math.abs(candidate.clientX - e.clientX) <= 6 &&
                                        Math.abs(candidate.clientY - e.clientY) <= 6 &&
                                        now - candidate.time <= 500
                                    ) {
                                        renameClickCandidateRef.current = null;
                                        suppressPlayheadMouseDownRef.current = true;
                                        // mousedown 紧随 pointerdown；若本次事件未产生 mousedown，
                                        // 下一个任务里清除标记，避免影响后续播放头拖拽。
                                        setTimeout(() => {
                                            suppressPlayheadMouseDownRef.current = false;
                                        }, 0);
                                        e.preventDefault();
                                        e.stopPropagation();
                                        clipActions.setRenamingClipId(candidate.clipId);
                                        return;
                                    }

                                    e.stopPropagation();
                                    const scroller = scrollRef.current;
                                    if (!scroller) return;
                                    const startX = e.clientX;
                                    const startY = e.clientY;
                                    let moved = false;
                                    const bounds = scroller.getBoundingClientRect();
                                    const initialSec =
                                        Number(sessionRef.current.playheadSec ?? 0) || 0;
                                    playheadDragRef.current = {
                                        pointerId: e.pointerId,
                                        lastBeat: initialSec,
                                    };
                                    (e.currentTarget as HTMLDivElement).setPointerCapture(
                                        e.pointerId,
                                    );

                                    function onPointerMove(ev: PointerEvent) {
                                        const drag = playheadDragRef.current;
                                        const currentScroller = scrollRef.current;
                                        if (
                                            !drag ||
                                            drag.pointerId !== e.pointerId ||
                                            !currentScroller
                                        ) {
                                            return;
                                        }
                                        const dx = ev.clientX - startX;
                                        const dy = ev.clientY - startY;
                                        if (!moved && dx * dx + dy * dy >= 9) {
                                            moved = true;
                                        }
                                        if (!moved) return;
                                        const currentBounds =
                                            currentScroller.getBoundingClientRect();
                                        drag.lastBeat = setPlayheadFromClientX(
                                            ev.clientX,
                                            currentBounds,
                                            currentScroller.scrollLeft,
                                            false,
                                        );
                                    }

                                    function endDrag() {
                                        const drag = playheadDragRef.current;
                                        if (!drag || drag.pointerId !== e.pointerId) return;
                                        playheadDragRef.current = null;
                                        if (!moved) {
                                            drag.lastBeat = setPlayheadFromClientX(
                                                startX,
                                                bounds,
                                                scroller!.scrollLeft,
                                                false,
                                            );
                                        }
                                        void dispatch(seekPlayhead(drag.lastBeat));
                                        window.removeEventListener("pointermove", onPointerMove);
                                        window.removeEventListener("pointerup", endDrag);
                                        window.removeEventListener("pointercancel", endDrag);
                                    }

                                    window.addEventListener("pointermove", onPointerMove);
                                    window.addEventListener("pointerup", endDrag);
                                    window.addEventListener("pointercancel", endDrag);
                                }}
                            />
                        </div>
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
                                {t("import_across_time" as any) ||
                                    "Import across time (same track)"}
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
                                {t("import_across_tracks" as any) ||
                                    "Import across tracks (one per track)"}
                            </button>
                        </div>
                    </div>
                )}

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

                          return (
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
                                      for (const id of ids) {
                                          dispatch(
                                              setClipMuted({
                                                  clipId: id,
                                                  muted,
                                              }),
                                          );
                                          void dispatch(
                                              setClipStateRemote({
                                                  clipId: id,
                                                  muted,
                                              }),
                                          );
                                      }
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
                                  onFadeCurveChange={(clipId, target, curve) => {
                                      dispatch(
                                          setClipFades({
                                              clipId,
                                              ...(target === "in"
                                                  ? {
                                                        fadeInCurve: curve,
                                                    }
                                                  : {
                                                        fadeOutCurve: curve,
                                                    }),
                                          }),
                                      );
                                      void dispatch(
                                          setClipStateRemote({
                                              clipId,
                                              ...(target === "in"
                                                  ? {
                                                        fadeInCurve: curve,
                                                    }
                                                  : {
                                                        fadeOutCurve: curve,
                                                    }),
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
                              />
                          );
                      })()
                    : null}

                {trackAreaMenu ? (
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
                    />
                ) : null}

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
                            <Text size="2">{t("clip_replace_same_source_confirm" as any)}</Text>
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
            </Flex>
        </Flex>
    );
};
