/**
 * useTimelineEventHandlers — 全局自定义事件监听
 *
 * 从 TimelinePanel.tsx 拆分而来，负责：
 * - hifi:editOp（selectAll / deselect / paste / split）
 * - hifi:nudgePlayhead（播放头微移）
 * - hifi:zoomTimelineFocus（聚焦缩放）
 * - context menu dismiss（pointerdown 外部关闭）
 * - auto-scroll（播放时保持播放头可见）
 * - hifi:focusCursor（滚动到播放头中心）
 * - useKeyboardShortcuts 桥接
 */
import { useEffect } from "react";
import type { AppDispatch, RootState } from "../../../../app/store";
import {
    seekPlayhead,
    selectTrackRemote,
    setplayheadSec,
    setSelectedClip,
    setSelectedClipPreservingTrack,
} from "../../../../features/session/sessionSlice";
import { computeAnchoredHorizontalZoom } from "../../../../utils/horizontalZoom";
import { useKeyboardShortcuts } from "./useKeyboardShortcuts";
import { gridStepBeats, MIN_PX_PER_SEC, MAX_PX_PER_SEC } from "../";
import type { ClipTemplate } from "../../../../features/session/sessionTypes";
import { computeAutoFollowScrollLeft } from "../../../../utils/autoFollowScroll";

// ── Args 类型 ─────────────────────────────────────────────────
export interface UseTimelineEventHandlersArgs {
    dispatch: AppDispatch;
    sessionRef: React.MutableRefObject<RootState["session"]>;
    scrollRef: React.MutableRefObject<HTMLDivElement | null>;
    trackListScrollRef: React.MutableRefObject<HTMLDivElement | null>;
    pxPerSecRef: React.MutableRefObject<number>;
    viewportWidthRef: React.MutableRefObject<number>;
    keyboardZoomPendingRef: React.MutableRefObject<{
        nextScale: number;
        nextScrollLeft: number;
    } | null>;

    // state values
    pxPerSec: number;
    setPxPerSec: React.Dispatch<React.SetStateAction<number>>;
    rowHeight: number;

    // multi-select
    multiSelectedClipIds: string[];
    setMultiSelectedClipIds: (ids: string[] | ((prev: string[]) => string[])) => void;

    // clipboard
    clipClipboardRef: React.MutableRefObject<ClipTemplate[] | null>;
    buildClipClipboardTemplates: (ids: string[]) => Promise<ClipTemplate[]>;

    // clip actions
    pasteClipsAtPlayhead: () => void;
    splitSelectedAtPlayhead: () => void;
    normalizeClips: (ids: string[]) => void;
    isEditableTarget: (target: EventTarget | null) => boolean;

    // context menu
    contextMenu: {
        x: number;
        y: number;
        clipId: string;
        overlappingClipIds?: string[];
    } | null;
    trackAreaMenu: {
        x: number;
        y: number;
        trackId: string;
    } | null;
    setContextMenu: React.Dispatch<
        React.SetStateAction<{
            x: number;
            y: number;
            clipId: string;
            overlappingClipIds?: string[];
        } | null>
    >;
    setTrackAreaMenu: React.Dispatch<
        React.SetStateAction<{
            x: number;
            y: number;
            trackId: string;
        } | null>
    >;

    // auto-scroll
    syncScrollLeft: (next: number) => void;

    // session values (for auto-scroll / focusCursor)
    autoScrollEnabled: boolean;
    isPlaying: boolean;
    playheadSec: number;
}

// ── Hook 实现 ─────────────────────────────────────────────────
export function useTimelineEventHandlers(args: UseTimelineEventHandlersArgs): void {
    const {
        dispatch,
        sessionRef,
        scrollRef,
        trackListScrollRef,
        pxPerSecRef,
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
        autoScrollEnabled,
        isPlaying,
        playheadSec,
    } = args;

    // ── useKeyboardShortcuts 桥接 ────────────────────────────
    useKeyboardShortcuts({
        sessionRef,
        dispatch,
        multiSelectedClipIds,
        setMultiSelectedClipIds,
        clipClipboardRef,
        buildClipClipboardTemplates,
        isEditableTarget,
        onNormalize: normalizeClips,
        onPaste: pasteClipsAtPlayhead,
        onSplitSelected: splitSelectedAtPlayhead,
    });

    // ── hifi:editOp ──────────────────────────────────────────
    useEffect(() => {
        function onEditOp(e: Event) {
            const op = (e as CustomEvent<{ op?: string }>).detail?.op;
            const active = document.activeElement as HTMLElement | null;
            const inPianoRoll =
                active?.hasAttribute("data-piano-roll-scroller") ||
                active?.closest?.("[data-piano-roll-scroller]");
            const inTrackHeader =
                Boolean(active?.closest?.("[data-track-list-panel]")) ||
                document.body.getAttribute("data-hs-focus-window") === "trackHeader";
            const deferToPianoRollForSelection =
                inPianoRoll &&
                sessionRef.current.toolMode === "select" &&
                (op === "selectAll" || op === "deselect");
            if (deferToPianoRollForSelection) return;
            if (op === "paste" && (inPianoRoll || inTrackHeader)) {
                return;
            }
            if (inPianoRoll && op !== "selectAll" && op !== "deselect") {
                return;
            }

            if (op === "selectAll") {
                const allIds = sessionRef.current.clips.map((clip) => clip.id);
                setMultiSelectedClipIds(allIds);
                dispatch(setSelectedClipPreservingTrack(allIds[0] ?? null));
                return;
            }

            if (op === "deselect") {
                setMultiSelectedClipIds([]);
                dispatch(setSelectedClip(null));
                return;
            }

            if (op === "paste") {
                pasteClipsAtPlayhead();
            }
            if (op === "split") {
                splitSelectedAtPlayhead();
            }
        }
        window.addEventListener("hifi:editOp", onEditOp as EventListener);
        return () => window.removeEventListener("hifi:editOp", onEditOp as EventListener);
    }, [pasteClipsAtPlayhead, splitSelectedAtPlayhead]);

    // ── hifi:selectAdjacentTrack ────────────────────────────
    useEffect(() => {
        function onSelectAdjacentTrack(e: Event) {
            const direction = Math.sign(
                Number((e as CustomEvent<{ direction?: number }>).detail?.direction ?? 0),
            );
            if (!direction) return;

            const tracks = sessionRef.current.tracks;
            if (tracks.length === 0) return;

            const currentTrackId = sessionRef.current.selectedTrackId ?? tracks[0]?.id ?? null;
            if (!currentTrackId) return;

            let currentIndex = tracks.findIndex((track) => track.id === currentTrackId);
            if (currentIndex < 0) currentIndex = 0;

            const nextIndex = Math.max(0, Math.min(tracks.length - 1, currentIndex + direction));
            if (nextIndex === currentIndex) return;

            const nextTrackId = tracks[nextIndex]?.id;
            if (!nextTrackId) return;

            void dispatch(selectTrackRemote(nextTrackId));

            const ensureTrackVisible = (el: HTMLDivElement): number | null => {
                const trackTop = nextIndex * rowHeight;
                const trackBottom = trackTop + rowHeight;
                let nextScrollTop = el.scrollTop;

                if (trackTop < el.scrollTop) {
                    nextScrollTop = trackTop;
                } else if (trackBottom > el.scrollTop + el.clientHeight) {
                    nextScrollTop = trackBottom - el.clientHeight;
                }

                const maxScrollTop = Math.max(0, el.scrollHeight - el.clientHeight);
                nextScrollTop = Math.max(0, Math.min(maxScrollTop, nextScrollTop));
                if (Math.abs(nextScrollTop - el.scrollTop) <= 0.5) return null;
                el.scrollTop = nextScrollTop;
                return nextScrollTop;
            };

            const timelineScroller = scrollRef.current;
            const trackScroller = trackListScrollRef.current;

            const timelineNextScrollTop = timelineScroller
                ? ensureTrackVisible(timelineScroller)
                : null;

            if (!trackScroller) return;
            if (timelineNextScrollTop != null) {
                if (Math.abs(trackScroller.scrollTop - timelineNextScrollTop) > 0.5) {
                    trackScroller.scrollTop = timelineNextScrollTop;
                }
                return;
            }

            const trackNextScrollTop = ensureTrackVisible(trackScroller);
            if (
                trackNextScrollTop != null &&
                timelineScroller &&
                Math.abs(timelineScroller.scrollTop - trackNextScrollTop) > 0.5
            ) {
                timelineScroller.scrollTop = trackNextScrollTop;
            }
        }

        window.addEventListener("hifi:selectAdjacentTrack", onSelectAdjacentTrack as EventListener);
        return () =>
            window.removeEventListener(
                "hifi:selectAdjacentTrack",
                onSelectAdjacentTrack as EventListener,
            );
    }, [dispatch, rowHeight]);

    // ── hifi:nudgePlayhead ───────────────────────────────────
    useEffect(() => {
        function onNudge(e: Event) {
            const direction = Number(
                (e as CustomEvent<{ direction?: number }>).detail?.direction ?? 0,
            );
            if (!direction) return;
            const stepSec =
                gridStepBeats(sessionRef.current.grid) * (60 / Math.max(1, sessionRef.current.bpm));
            const current = Number(sessionRef.current.playheadSec ?? 0) || 0;
            const next = Math.max(0, current + Math.sign(direction) * stepSec);
            dispatch(setplayheadSec(next));
            void dispatch(seekPlayhead(next));
        }

        window.addEventListener("hifi:nudgePlayhead", onNudge as EventListener);
        return () => window.removeEventListener("hifi:nudgePlayhead", onNudge as EventListener);
    }, [dispatch]);

    // ── hifi:zoomTimelineFocus ───────────────────────────────
    useEffect(() => {
        function onZoomFocused(e: Event) {
            const active = document.activeElement as HTMLElement | null;
            const inTimeline =
                active?.hasAttribute("data-timeline-scroller") ||
                active?.closest?.("[data-timeline-scroller]") ||
                document.body.getAttribute("data-hs-focus-window") === "timeline";
            if (!inTimeline) return;

            const factor = Number((e as CustomEvent<{ factor?: number }>).detail?.factor ?? 1);
            if (!Number.isFinite(factor) || factor <= 0) return;

            const scroller = scrollRef.current;
            if (!scroller) return;

            const zoom = computeAnchoredHorizontalZoom({
                currentScale: pxPerSecRef.current,
                factor,
                minScale: MIN_PX_PER_SEC,
                maxScale: MAX_PX_PER_SEC,
                scrollLeft: scroller.scrollLeft,
                viewportWidth: scroller.clientWidth,
                anchorSec: Number(sessionRef.current.playheadSec ?? 0) || 0,
                contentSec: sessionRef.current.projectSec,
            });
            if (!zoom) return;

            keyboardZoomPendingRef.current = {
                nextScale: zoom.nextScale,
                nextScrollLeft: zoom.nextScrollLeft,
            };
            setPxPerSec(zoom.nextScale);
        }

        window.addEventListener("hifi:zoomTimelineFocus", onZoomFocused as EventListener);
        return () =>
            window.removeEventListener("hifi:zoomTimelineFocus", onZoomFocused as EventListener);
    }, []);

    // ── Context menu dismiss ─────────────────────────────────
    useEffect(() => {
        if (!contextMenu && !trackAreaMenu) return;
        function onAnyPointerDown(e: PointerEvent) {
            const target = e.target as HTMLElement | null;
            if (target?.closest?.("[data-hs-context-menu='1']")) return;
            setContextMenu(null);
            setTrackAreaMenu(null);
        }
        window.addEventListener("pointerdown", onAnyPointerDown, true);
        return () => window.removeEventListener("pointerdown", onAnyPointerDown, true);
    }, [contextMenu, trackAreaMenu]);

    // ── Auto-scroll: keep playhead visible during playback ───
    useEffect(() => {
        if (!autoScrollEnabled || !isPlaying) return;
        const scroller = scrollRef.current;
        if (!scroller) return;
        const next = computeAutoFollowScrollLeft({
            playheadSec,
            pxPerSec,
            viewportWidth: scroller.clientWidth,
            contentWidth: scroller.scrollWidth,
        });
        if (Math.abs(scroller.scrollLeft - next) > 0.5) {
            scroller.scrollLeft = next;
            syncScrollLeft(next);
        }
    }, [autoScrollEnabled, isPlaying, playheadSec, pxPerSec]);

    // ── hifi:focusCursor ─────────────────────────────────────
    useEffect(() => {
        function handler() {
            const scroller = scrollRef.current;
            if (!scroller) return;
            const next = computeAutoFollowScrollLeft({
                playheadSec,
                pxPerSec,
                viewportWidth: scroller.clientWidth,
                contentWidth: scroller.scrollWidth,
            });
            scroller.scrollLeft = next;
            syncScrollLeft(next);
        }
        window.addEventListener("hifi:focusCursor", handler);
        return () => window.removeEventListener("hifi:focusCursor", handler);
    }, [playheadSec, pxPerSec]);
}
