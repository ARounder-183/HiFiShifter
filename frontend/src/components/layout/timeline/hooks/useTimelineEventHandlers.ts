/**
 * useTimelineEventHandlers — 全局自定义事件监听
 *
 * 从 TimelinePanel.tsx 拆分而来，负责：
 * - hifi:editOp（selectAll / deselect / paste / split）
 * - hifi:nudgePlayhead（播放头微移）
 * - hifi:zoomTimelineFocus（聚焦缩放）
 * - context menu dismiss（pointerdown 外部关闭）
 * - auto-scroll（播放时保持播放头可见）
 * - hifi:focusCursor（滚动到播放头中心；粘贴后的聚焦由
 *   pendingPlayheadRevealSec + TimelinePanel useLayoutEffect 驱动）
 * - useKeyboardShortcuts 桥接
 */
import { useEffect } from "react";
import { flushSync } from "react-dom";
import type { AppDispatch, RootState } from "../../../../app/store";
import {
    seekPlayhead,
    selectTrackRemote,
    setplayheadSec,
    setSelectedClip,
    setSelectedClipPreservingTrack,
} from "../../../../features/session/sessionSlice";
import { resolveHorizontalWheelZoom } from "../runtime/timelineScrollRange";
import { applyNativeScrollLeft } from "../runtime/nativeScrollApply";
import { useKeyboardShortcuts } from "./useKeyboardShortcuts";
import { gridStepBeats, MIN_PX_PER_SEC, MAX_PX_PER_SEC } from "../";
import { computeFocusCursorScrollLeft } from "../../../../utils/autoFollowScroll";
import { resolveTimelineMinPxPerSec } from "../runtime/timelineZoomBounds";
import { shouldRouteClipPasteToParamEditor } from "../clipboardFocusRouting";
import { expandClipIdsWithGroups } from "./useGroupExpansion";

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
    copyClips: (ids: string[]) => Promise<boolean>;
    cutClips: (ids: string[]) => void;

    // clip actions
    pasteClipsAtPlayhead: (mode?: "selected" | "new_tracks") => void;
    splitSelectedAtPlayhead: () => void;
    normalizeClips: (ids: string[]) => void;
    groupClips: (ids: string[]) => void;
    ungroupClips: (ids: string[]) => void;
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

    // session values (for zoom / focusCursor)
    dynamicProjectSec: number;
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
    } = args;

    // ── useKeyboardShortcuts 桥接 ────────────────────────────
    useKeyboardShortcuts({
        sessionRef,
        dispatch,
        multiSelectedClipIds,
        setMultiSelectedClipIds,
        copyClips,
        cutClips,
        isEditableTarget,
        onNormalize: normalizeClips,
        onPaste: pasteClipsAtPlayhead,
        onSplitSelected: splitSelectedAtPlayhead,
        onGroup: groupClips,
        onUngroup: ungroupClips,
    });

    // ── hifi:editOp ──────────────────────────────────────────
    useEffect(() => {
        function onEditOp(e: Event) {
            const op = (e as CustomEvent<{ op?: string }>).detail?.op;
            const active = document.activeElement as HTMLElement | null;
            const inPianoRoll = Boolean(
                active?.hasAttribute("data-piano-roll-scroller") ||
                active?.closest?.("[data-piano-roll-scroller]"),
            );
            const inTrackHeader = Boolean(
                Boolean(active?.closest?.("[data-track-list-panel]")) ||
                document.body.getAttribute("data-hs-focus-window") === "trackHeader",
            );
            const deferToPianoRollForSelection =
                inPianoRoll &&
                sessionRef.current.toolMode === "select" &&
                (op === "selectAll" || op === "deselect");
            if (deferToPianoRollForSelection) return;
            if (
                op === "paste" &&
                shouldRouteClipPasteToParamEditor({
                    inPianoRoll,
                    inTrackHeader,
                })
            ) {
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
                return;
            }
            if (op === "pasteTracks") {
                pasteClipsAtPlayhead("new_tracks");
                return;
            }
            if (op === "split") {
                splitSelectedAtPlayhead();
            }
        }
        window.addEventListener("hifi:editOp", onEditOp as EventListener);
        return () => window.removeEventListener("hifi:editOp", onEditOp as EventListener);
    }, [multiSelectedClipIds, pasteClipsAtPlayhead, sessionRef, splitSelectedAtPlayhead]);

    // ── hifi:timelineEditOp (menu routing when timeline has focus) ─
    useEffect(() => {
        function onTimelineEditOp(e: Event) {
            const op = (e as CustomEvent<{ op?: string }>).detail?.op;
            const selectedIds =
                multiSelectedClipIds.length > 0
                    ? [...multiSelectedClipIds]
                    : sessionRef.current.selectedClipId
                      ? [sessionRef.current.selectedClipId]
                      : [];
            if (op === "copy" || op === "cut") {
                if (selectedIds.length === 0) return;
                const s = sessionRef.current;
                const expandedIds = expandClipIdsWithGroups(
                    selectedIds,
                    s.clips,
                    s.ignoreGrouping,
                    s.disabledGroupIds,
                );
                if (op === "copy") void copyClips(expandedIds);
                else cutClips(expandedIds);
                return;
            }
            if (op === "paste") {
                pasteClipsAtPlayhead();
            } else if (op === "pasteTracks") {
                pasteClipsAtPlayhead("new_tracks");
            }
        }
        window.addEventListener("hifi:timelineEditOp", onTimelineEditOp as EventListener);
        return () =>
            window.removeEventListener("hifi:timelineEditOp", onTimelineEditOp as EventListener);
    }, [copyClips, cutClips, multiSelectedClipIds, pasteClipsAtPlayhead, sessionRef]);

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

            const zoom = resolveHorizontalWheelZoom({
                factor,
                basePxPerSec: pxPerSecRef.current,
                baseScrollLeft: scroller.scrollLeft,
                totalSec: dynamicProjectSec,
                viewportWidth: scroller.clientWidth,
                playheadZoomEnabled: true,
                playheadSec: Number(sessionRef.current.playheadSec ?? 0) || 0,
                anchorScreenX: 0,
                minPxPerSec: resolveTimelineMinPxPerSec({
                    baseMinPxPerSec: MIN_PX_PER_SEC,
                    projectSec: dynamicProjectSec,
                    viewportWidthPx: scroller.clientWidth,
                }),
                maxPxPerSec: MAX_PX_PER_SEC,
            });
            if (!zoom) return;

            keyboardZoomPendingRef.current = {
                nextScale: zoom.nextPxPerSec,
                nextScrollLeft: zoom.nextScrollLeft,
            };
            pxPerSecRef.current = zoom.nextPxPerSec;
            // 原子缩放：flushSync 保证 DOM 按新缩放重排后，layout effect 在
            // 同一绘制帧内写原生 scrollLeft 并同步重绘标尺与画布（与滚轮
            // 缩放的 TimelineScrollArea 路径一致，避免画布先行的抽动）。
            flushSync(() => {
                setPxPerSec(zoom.nextPxPerSec);
            });
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

    // ── hifi:focusCursor（快捷键"聚焦播放光标"）──────────────
    // 粘贴后的视图聚焦不再走事件：由 reducer 记录的 pendingPlayheadRevealSec
    // 驱动，在 TimelinePanel 的 useLayoutEffect 中于状态与 DOM 均提交后执行。
    // （旧的事件方案会在工程全长扩充前触发，滚动被旧上限钳制导致聚焦失败。）
    useEffect(() => {
        // 无条件把当前播放光标滚到视口内固定偏移处。滚动上限使用“新滚动
        // 模型”（= 工程宽度）：光标接近工程末尾时也必须能正确进入画面，
        // 而不是被“工程宽 − 视口宽”的旧上限卡在画面右缘。
        function handler() {
            const scroller = scrollRef.current;
            if (!scroller) return;
            const next = computeFocusCursorScrollLeft({
                playheadSec: Number(sessionRef.current.playheadSec ?? 0) || 0,
                pxPerSec,
                contentWidth: dynamicProjectSec * pxPerSec,
            });
            // 写后回读浏览器实际接受的偏移再广播：请求值可能被钳制/量化/锚定
            // 修正，sticky 画布层必须与原生 DOM 层使用同一偏移。
            const applied = applyNativeScrollLeft(scroller, next);
            syncScrollLeft(applied);
        }
        window.addEventListener("hifi:focusCursor", handler);
        return () => window.removeEventListener("hifi:focusCursor", handler);
    }, [pxPerSec, sessionRef, syncScrollLeft, dynamicProjectSec, scrollRef]);
}
