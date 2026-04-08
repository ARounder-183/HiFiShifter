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
import {
    addTrackRemote,
    duplicateTrackRemote,
    removeTrackRemote,
    selectTrackRemote,
    setTrackStateRemote,
    seekPlayhead,
    moveTrackRemote,
    setClipMuted,
    importAudioAtPosition,
    importAudioFileAtPosition,
    importMultipleAudioAtPosition,
    setClipStateRemote,
    setClipFades,
    glueClipsRemote,
    removeClipsRemote,
    setTrackName,
    setTrackVolume,
} from "../../features/session/sessionSlice";

import { NEW_TRACK_SENTINEL, useClipDrag } from "./timeline/hooks/useClipDrag";
import { useEditDrag } from "./timeline/hooks/useEditDrag";
import { useSlipDrag } from "./timeline/hooks/useSlipDrag";
import { collectFadeContextClips } from "./timeline/clipFadeContext";
import { emitExternalFileAction } from "../../features/session/projectOpenEvents";

import {
    BackgroundGrid,
    ClipContextMenu,
    TRACK_ADD_ROW_HEIGHT,
    TrackAreaContextMenu,
    TimelineScrollArea,
    TimeRuler,
    TrackLane,
    TrackList,
    detectExternalPathAction,
    extractLocalFilePath,
    hasFileDrag,
} from "./timeline";

// ── 拆分出的 hooks ──────────────────────────────────────────
import { useTimelineState } from "./timeline/hooks/useTimelineState";
import { useTimelineDragDrop } from "./timeline/hooks/useTimelineDragDrop";
import { useTimelineClipActions } from "./timeline/hooks/useTimelineClipActions";
import { useTimelineEventHandlers } from "./timeline/hooks/useTimelineEventHandlers";
import { useVisualPlayhead } from "../../hooks/useVisualPlayhead";
import { computeAutoFollowScrollLeft } from "../../utils/autoFollowScroll";
import { writeSystemClipboardObject } from "../../utils/systemClipboard";

export const TimelinePanel: React.FC = () => {
    const { t } = useI18n();
    const rulerPlayheadLineRef = React.useRef<HTMLDivElement | null>(null);
    const rulerPlayheadHeadRef = React.useRef<HTMLDivElement | null>(null);

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
        bars,
        clipsByTrackId,
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
        snapBeat,
        isEditableTarget,
        isPointerOnNativeScrollbar,
        startPanPointer,
        setPlayheadFromClientX,
        startDeferredPlayheadSeek,
        keyboardZoomPendingRef,
    } = state;

    // ── 2. Clip 多选 + 操作回调 ─────────────────────────────
    const clipActions = useTimelineClipActions({
        sessionRef,
        scrollRef,
        lastClickedClipIdRef,
        pxPerSec,
        pxPerBeat,
        rowHeight,
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
        clipClipboardRef,
        buildClipClipboardTemplates,
        normalizeClips,
        replaceClipSources,
        splitClipIdsAtPlayhead,
        splitSelectedAtPlayhead,
        selectClipRangeByRect,
        rangeSelectAnchorClipId,
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
            trackIdFromClientY,
            rowTopForTrackId,
            setDropPreview,
            ensureDropPreviewDuration,
            getDropPreviewWidthPx,
            setImportModeMenu,
            pxPerSec,
            rowHeight,
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
        clipClipboardRef,
        buildClipClipboardTemplates,
        pasteClipsAtPlayhead,
        splitSelectedAtPlayhead,
        normalizeClips,
        isEditableTarget,
        contextMenu,
        trackAreaMenu,
        setContextMenu,
        setTrackAreaMenu,
        syncScrollLeft,
        autoScrollEnabled: s.autoScrollEnabled,
        isPlaying: s.runtime.isPlaying,
        playheadSec: s.playheadSec,
    });

    const isTransportAdvancing = s.runtime.isPlaying && s.runtime.playbackPositionSec > 1e-4;

    useVisualPlayhead({
        syncedPlayheadSec: s.playheadSec,
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
                if (!s.autoScrollEnabled || !s.runtime.isPlaying) return;
                const scroller = scrollRef.current;
                if (!scroller) return;
                const next = computeAutoFollowScrollLeft({
                    playheadSec: visualPlayheadSec,
                    pxPerSec: pxPerSecRef.current,
                    viewportWidth: scroller.clientWidth,
                    contentWidth: scroller.scrollWidth,
                });
                if (Math.abs(scroller.scrollLeft - next) <= 0.5) return;
                scroller.scrollLeft = next;
                syncScrollLeft(next);
            },
            [pxPerSecRef, s.autoScrollEnabled, s.runtime.isPlaying, scrollRef, syncScrollLeft],
        ),
    });

    // ── 5. 拖拽 hooks 桥接 ──────────────────────────────────
    const { editDragRef: _editDragRef, startEditDrag } = useEditDrag({
        scrollRef,
        sessionRef,
        dispatch,
        snapBeat,
        beatFromClientX,
        noSnapKb,
        gridSnapEnabled: s.gridSnapEnabled,
    });

    const { slipDragRef: _slipDragRef, startSlipDrag } = useSlipDrag({
        scrollRef,
        sessionRef,
        dispatch,
        multiSelectedClipIds,
        multiSelectedSet,
        beatFromClientX,
    });

    const {
        clipDragRef: _clipDragRef,
        startClipDrag: _startClipDragInner,
        ghostDrag,
    } = useClipDrag({
        scrollRef,
        sessionRef,
        rowHeight,
        multiSelectedClipIds,
        multiSelectedSet,
        dispatch,
        snapBeat,
        beatFromClientX,
        trackIdFromClientY,
        setClipDropNewTrack,
        setMultiSelectedClipIds,
        slipEditKb,
        noSnapKb,
        gridSnapEnabled: s.gridSnapEnabled,
        copyDragKb,
        autoCrossfadeEnabled: s.autoCrossfadeEnabled,
        onCtrlClick: toggleTrackLaneCtrlSelection,
    });

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
                const clip = s.clips.find((item) => item.id === clipId);
                if (!initial || !clip) return null;
                return {
                    ...clip,
                    startSec: Math.max(0, initial.startSec + ghostDrag.deltaSec),
                };
            })
            .filter((clip): clip is (typeof s.clips)[number] => clip != null);
    }, [clipDropNewTrack, ghostDrag, s.clips]);

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

    const trackGridHeight = Math.max(0, contentHeight - TRACK_ADD_ROW_HEIGHT);

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
                onSelectTrack={(trackId) => {
                    if (sessionRef.current.selectedTrackId === trackId) {
                        return;
                    }
                    void dispatch(selectTrackRemote(trackId));
                }}
                onRemoveTrack={(trackId) => {
                    dispatch(removeTrackRemote(trackId));
                }}
                onMoveTrack={(payload) => {
                    dispatch(
                        moveTrackRemote({
                            trackId: payload.trackId,
                            targetIndex: payload.targetIndex,
                            parentTrackId: payload.parentTrackId,
                        }),
                    );
                }}
                onToggleMute={(trackId, nextMuted) => {
                    dispatch(
                        setTrackStateRemote({
                            trackId,
                            muted: nextMuted,
                        }),
                    );
                }}
                onToggleSolo={(trackId, nextSolo) => {
                    dispatch(
                        setTrackStateRemote({
                            trackId,
                            solo: nextSolo,
                        }),
                    );
                }}
                onToggleCompose={(trackId, nextComposeEnabled) => {
                    dispatch(
                        setTrackStateRemote({
                            trackId,
                            composeEnabled: nextComposeEnabled,
                        }),
                    );
                }}
                onVolumeUiChange={(trackId, nextVolume) => {
                    setTrackVolumeUi((prev) => ({
                        ...prev,
                        [trackId]: nextVolume,
                    }));
                }}
                onVolumeCommit={(trackId, nextVolume) => {
                    // 先同步更新 Redux 中的 track.volume 为新值，
                    // 再清除 trackVolumeUi 覆盖，这样即使 setTrackStateRemote
                    // 尚未完成，TrackList 也能从 backendVolume 读到正确的值，避免回弹。
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
                }}
                onAddTrack={() => {
                    dispatch(addTrackRemote({}));
                }}
                onTrackColorChange={(trackId, color) => {
                    dispatch(
                        setTrackStateRemote({
                            trackId,
                            color,
                        }),
                    );
                }}
                onAlgoChange={(trackId, algo) => {
                    dispatch(
                        setTrackStateRemote({
                            trackId,
                            pitchAnalysisAlgo: algo,
                        }),
                    );
                }}
                onTrackNameChange={(trackId, name) => {
                    dispatch(setTrackName({ trackId, name }));
                    dispatch(
                        setTrackStateRemote({
                            trackId,
                            name,
                        }),
                    );
                }}
                onDuplicateTrack={(trackId) => {
                    dispatch(duplicateTrackRemote(trackId));
                }}
                onScrollTopChange={(scrollTop) => {
                    const timelineScroller = scrollRef.current;
                    if (!timelineScroller) return;
                    if (Math.abs(timelineScroller.scrollTop - scrollTop) < 0.5) return;
                    timelineScroller.scrollTop = scrollTop;
                }}
            />

            {/* Timeline View (Right) */}
            <Flex direction="column" className="flex-1 relative overflow-hidden bg-qt-graph-bg">
                <TimeRuler
                    contentWidth={contentWidth}
                    scrollLeft={scrollLeft}
                    bars={bars}
                    pxPerBeat={pxPerBeat}
                    pxPerSec={pxPerSec}
                    secPerBeat={secPerBeat}
                    viewportWidth={viewportWidth}
                    playheadSec={s.playheadSec}
                    playheadLineRef={rulerPlayheadLineRef}
                    playheadHeadRef={rulerPlayheadHeadRef}
                    contentRef={rulerContentRef}
                    onMouseDown={(e) => {
                        if (e.button !== 0) return;
                        document.body.setAttribute("data-hs-focus-window", "timeline");
                        const scroller = scrollRef.current;
                        if (!scroller) return;
                        const bounds = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
                        startDeferredPlayheadSeek({
                            startClientX: e.clientX,
                            startClientY: e.clientY,
                            getBounds: () => bounds,
                            getScrollLeft: () => scroller.scrollLeft,
                        });
                    }}
                />

                {/* Tracks Area */}
                <TimelineScrollArea
                    scrollRef={scrollRef}
                    projectSec={dynamicProjectSec}
                    bpm={s.bpm}
                    pxPerSec={pxPerSec}
                    setPxPerSec={setPxPerSec}
                    rowHeight={rowHeight}
                    setRowHeight={setRowHeight}
                    setScrollLeft={setScrollLeftAction}
                    scrollHorizontalKb={scrollHorizontalKb}
                    scrollVerticalKb={scrollVerticalKb}
                    horizontalZoomKb={horizontalZoomKb}
                    verticalZoomKb={verticalZoomKb}
                    playheadSec={s.playheadSec}
                    playheadZoomEnabled={s.playheadZoomEnabled}
                    className="flex-1 bg-qt-graph-bg overflow-auto relative custom-scrollbar"
                    data-timeline-scroller
                    onScroll={(e) => {
                        const el = e.currentTarget as HTMLDivElement;
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
                        if (path && detectExternalPathAction(path) !== "importAudio") {
                            setDropPreview(null);
                            return;
                        }
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
                        setContextMenu(null);
                        setTrackAreaMenu(null);
                        setMultiSelectedClipIds([]);
                        const scroller = scrollRef.current;
                        if (!scroller) return;
                        if (isPointerOnNativeScrollbar(scroller, e.clientX, e.clientY)) return;
                        const trackId = trackIdFromClientY(e.clientY);
                        if (trackId && trackId !== sessionRef.current.selectedTrackId) {
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
                    {/* Track Lanes */}
                    <div
                        className="relative"
                        style={{
                            width: contentWidth,
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

                        <BackgroundGrid
                            contentWidth={contentWidth}
                            contentHeight={trackGridHeight}
                            pxPerBeat={pxPerBeat}
                            grid={s.grid}
                            beatsPerBar={Math.max(1, Math.round(s.beats || 4))}
                        />

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

                        {s.tracks.map((track) => {
                            const trackClips =
                                clipsByTrackId.get(track.id) ?? ([] as typeof s.clips);

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
                                    altPressed={altPressed}
                                    selectedClipId={s.selectedClipId}
                                    multiSelectedClipIds={multiSelectedClipIds}
                                    multiSelectedSet={multiSelectedSet}
                                    trackColor={track.color || undefined}
                                    ensureSelected={ensureTrackLaneSelected}
                                    selectClipRemote={selectTrackLaneClipRemote}
                                    onShiftRangeSelect={selectClipRangeByRect}
                                    rangeSelectAnchorClipId={rangeSelectAnchorClipId}
                                    openContextMenu={openTrackLaneContextMenu}
                                    seekFromClientX={seekFromTrackLaneClientX}
                                    ghostDrag={ghostDrag}
                                    allClips={s.clips}
                                    startClipDrag={startClipDrag}
                                    startEditDrag={startEditDrag}
                                    toggleClipMuted={toggleTrackLaneClipMuted}
                                    onCtrlToggleSelect={toggleTrackLaneCtrlSelection}
                                    clearContextMenu={clearContextMenu}
                                    toggleMultiSelect={toggleTrackLaneMultiSelect}
                                    renamingClipId={renamingClipId}
                                    onRenameCommit={commitTrackLaneRename}
                                    onRenameDone={handleTrackLaneRenameDone}
                                    onGainCommit={commitTrackLaneGain}
                                />
                            );
                        })}

                        <div className="absolute inset-0 pointer-events-none z-[12]">
                            <BackgroundGrid
                                contentWidth={contentWidth}
                                contentHeight={trackGridHeight}
                                pxPerBeat={pxPerBeat}
                                grid={s.grid}
                                beatsPerBar={Math.max(1, Math.round(s.beats || 4))}
                                lineOpacity={0.38}
                                showBoundary={false}
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

                        {/* Playhead Cursor */}
                        <div
                            ref={playheadRef}
                            className="absolute top-0 bottom-0 w-px bg-qt-playhead z-20 cursor-ew-resize"
                            style={{
                                left: s.playheadSec * pxPerSec,
                            }}
                            onPointerDown={(e) => {
                                if (e.button !== 0) return;
                                e.stopPropagation();
                                const scroller = scrollRef.current;
                                if (!scroller) return;
                                const startX = e.clientX;
                                const startY = e.clientY;
                                let moved = false;
                                const bounds = scroller.getBoundingClientRect();
                                const initialSec = s.playheadSec;
                                playheadDragRef.current = {
                                    pointerId: e.pointerId,
                                    lastBeat: initialSec,
                                };
                                (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);

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
                                    const currentBounds = currentScroller.getBoundingClientRect();
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

                          const selectedIds =
                              multiSelectedClipIds.length >= 2
                                  ? multiSelectedClipIds
                                  : [contextMenu.clipId];
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
                                      void (async () => {
                                          const templates = await buildClipClipboardTemplates(ids);
                                          if (templates.length > 0) {
                                              clipClipboardRef.current = templates;
                                              try {
                                                  await writeSystemClipboardObject({
                                                      version: 1,
                                                      kind: "clip",
                                                      templates,
                                                  });
                                              } catch {
                                                  // ignore clipboard write errors
                                              }
                                          }
                                      })();
                                  }}
                                  onCut={(ids) => {
                                      void (async () => {
                                          const templates = await buildClipClipboardTemplates(ids);
                                          if (templates.length === 0) return;
                                          clipClipboardRef.current = templates;
                                          try {
                                              await writeSystemClipboardObject({
                                                  version: 1,
                                                  kind: "clip",
                                                  templates,
                                              });
                                          } catch {
                                              // ignore clipboard write errors
                                          }
                                          setContextMenu(null);
                                          setMultiSelectedClipIds([]);
                                          void dispatch(removeClipsRemote(ids));
                                      })();
                                  }}
                                  onReplace={(ids) => {
                                      void replaceClipSources(ids);
                                  }}
                                  onSplit={(clipIds) => {
                                      setContextMenu(null);
                                      splitClipIdsAtPlayhead(clipIds);
                                  }}
                                  onGlue={(ids) => {
                                      setContextMenu(null);
                                      if (ids.length >= 2) {
                                          void dispatch(glueClipsRemote(ids));
                                          setMultiSelectedClipIds([]);
                                      }
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
                                      for (const id of ids) {
                                          void dispatch(
                                              setClipStateRemote({
                                                  clipId: id,
                                                  reversed,
                                              }),
                                          );
                                      }
                                  }}
                              />
                          );
                      })()
                    : null}

                {trackAreaMenu ? (
                    <TrackAreaContextMenu
                        x={trackAreaMenu.x}
                        y={trackAreaMenu.y}
                        canPaste={
                            Boolean(clipClipboardRef.current) &&
                            (clipClipboardRef.current?.length ?? 0) > 0
                        }
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
            </Flex>
        </Flex>
    );
};
