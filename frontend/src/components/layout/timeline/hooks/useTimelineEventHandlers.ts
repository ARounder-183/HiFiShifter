/**
 * useTimelineEventHandlers — 全局自定义事件监听
 *
 * 从 TimelinePanel.tsx 拆分而来，负责：
 * - hifi:timelineEditOp（时间轴编辑操作唯一入口：copy/cut/paste/pasteTracks/
 *   split/delete/normalize/group/ungroup/cycleTake/selectAll/deselect ——
 *   由全局路由 focusRouting.resolveEditOpRoute 按活动编辑表面定向派发，
 *   事件名即契约，消费者不再自行判断焦点）
 * - hifi:nudgePlayhead（播放头微移）
 * - hifi:zoomTimelineFocus（聚焦缩放）
 * - context menu dismiss（pointerdown 外部关闭）
 * - auto-scroll（播放时保持播放头可见）
 * - hifi:focusCursor（滚动到播放头中心；粘贴后的聚焦由
 *   pendingPlayheadRevealSec + TimelinePanel useLayoutEffect 驱动）
 */
import { useEffect } from "react";
import { flushSync } from "react-dom";
import type { AppDispatch, RootState } from "../../../../app/store";
import { useAppStore } from "../../../../app/hooks";
import {
    cycleClipTakesRemote,
    removeClipsRemote,
    seekPlayhead,
    selectTrackRemote,
    setplayheadSec,
    setSelectedClip,
    setSelectedClipPreservingTrack,
} from "../../../../features/session/sessionSlice";
import { beginHoldRepeat, selectMergedKeybindings } from "../../../../features/keybindings";
import { getActiveSurface } from "../../../../features/uiFocus/focusSurface";
import { resolveHorizontalWheelZoom } from "../runtime/timelineScrollRange";
import { applyNativeScrollLeft } from "../runtime/nativeScrollApply";
import { gridStepBeats, MIN_PX_PER_SEC, MAX_PX_PER_SEC } from "../";
import { computeFocusCursorScrollLeft } from "../../../../utils/autoFollowScroll";
import { resolveTimelineMinPxPerSec } from "../runtime/timelineZoomBounds";
import { getDynamicProjectSec } from "../../../../features/session/projectBoundary";
import { expandClipIdsWithGroups } from "./useGroupExpansion";

// ── Args 类型 ─────────────────────────────────────────────────
export interface UseTimelineEventHandlersArgs {
    dispatch: AppDispatch;
    sessionRef: React.MutableRefObject<RootState["session"]>;
    /**
     * 读取当前播放头（秒）。播放中必须返回与绘制同源的视觉插值值
     * （rAF 逐帧更新），不能用 33Hz 轮询的 store 滞后值——以滞后值锚定
     * 会让缩放提交时播放头跳变 δ·Δpx。未提供时回退 sessionRef。
     */
    getPlayheadSec?: () => number;
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
    /** 缩放时与 pxPerSec 同帧提交的 scrollLeft state。 */
    commitScrollLeftState: React.Dispatch<React.SetStateAction<number>>;
    rowHeight: number;

    // multi-select
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
}

// ── Hook 实现 ─────────────────────────────────────────────────
export function useTimelineEventHandlers(args: UseTimelineEventHandlersArgs): void {
    const {
        dispatch,
        sessionRef,
        getPlayheadSec,
        scrollRef,
        trackListScrollRef,
        pxPerSecRef,
        keyboardZoomPendingRef,
        pxPerSec,
        setPxPerSec,
        commitScrollLeftState,
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
    } = args;

    // 实时 store（事件监听器内同步读取，避免闭包捕获过期选区/状态）
    const store = useAppStore();

    // ── hifi:timelineEditOp（时间轴编辑操作唯一入口）──────────
    // 全局路由（focusRouting.resolveEditOpRoute）按「活动编辑表面」把
    // clip.* 快捷键与菜单编辑操作定向派发到这里 —— 事件名即契约，本消费
    // 者信任事件、不再自行判断焦点（旧版在此重猜 activeElement / body
    // 属性，与参数编辑器侧的判断互相矛盾，正是复制/剪切/粘贴冲突的根因）。
    // 选区在处理器内实时读取 store（同步、权威）：菜单打开时机与鼠标点击
    // 选择之间可能隔着未提交的渲染，闭包里的选区是旧值，会让复制/剪切
    // 作用到过期（甚至已删除）的 Clip 上而静默失败。
    useEffect(() => {
        function onTimelineEditOp(e: Event) {
            const op = (e as CustomEvent<{ op?: string }>).detail?.op;
            if (!op) return;
            const session = store.getState().session;
            const rawSelectedIds =
                session.multiSelectedClipIds.length > 0
                    ? [...session.multiSelectedClipIds]
                    : session.selectedClipId
                      ? [session.selectedClipId]
                      : [];
            // 过滤已删除/已不存在的 Clip id：让失效选区（如删除、胶合、
            // 拆分替换 id 后的残留）不再把死 id 传给后端造成静默失败。
            const selectedIds = rawSelectedIds.filter((id) =>
                session.clips.some((clip) => clip.id === id),
            );
            const expandedIds = () =>
                expandClipIdsWithGroups(
                    selectedIds,
                    session.clips,
                    session.ignoreGrouping,
                    session.disabledGroupIds,
                );

            switch (op) {
                case "copy": {
                    if (selectedIds.length === 0) return;
                    void copyClips(expandedIds());
                    return;
                }
                case "cut": {
                    if (selectedIds.length === 0) return;
                    cutClips(expandedIds());
                    return;
                }
                case "paste": {
                    pasteClipsAtPlayhead();
                    // 长按重复：首次立即粘贴，持续按住后按统一节奏连续重复
                    // （holdRepeat 管理器，与「添加轨道」等共用同一套长按逻辑）。
                    // 仅作用于时间轴粘贴路径（参数编辑器粘贴不重复）。
                    const pasteKb = selectMergedKeybindings(store.getState())["clip.paste"];
                    if (pasteKb) beginHoldRepeat(pasteKb, pasteClipsAtPlayhead);
                    return;
                }
                case "pasteTracks": {
                    pasteClipsAtPlayhead("new_tracks");
                    return;
                }
                case "split": {
                    splitSelectedAtPlayhead();
                    return;
                }
                case "delete": {
                    if (selectedIds.length === 0) return;
                    setMultiSelectedClipIds([]);
                    void dispatch(removeClipsRemote(selectedIds));
                    return;
                }
                case "normalize": {
                    if (selectedIds.length === 0) return;
                    normalizeClips(selectedIds);
                    return;
                }
                case "group": {
                    if (selectedIds.length < 2) return;
                    groupClips(selectedIds);
                    return;
                }
                case "ungroup": {
                    if (selectedIds.length === 0) return;
                    ungroupClips(selectedIds);
                    return;
                }
                case "cycleTake": {
                    if (selectedIds.length === 0) return;
                    void dispatch(cycleClipTakesRemote({ clipIds: selectedIds, direction: 1 }));
                    return;
                }
                case "cycleTakePrev": {
                    if (selectedIds.length === 0) return;
                    void dispatch(cycleClipTakesRemote({ clipIds: selectedIds, direction: -1 }));
                    return;
                }
                case "selectAll": {
                    const allIds = session.clips.map((clip) => clip.id);
                    setMultiSelectedClipIds(allIds);
                    dispatch(setSelectedClipPreservingTrack(allIds[0] ?? null));
                    return;
                }
                case "deselect": {
                    setMultiSelectedClipIds([]);
                    dispatch(setSelectedClip(null));
                    return;
                }
            }
        }
        window.addEventListener("hifi:timelineEditOp", onTimelineEditOp as EventListener);
        return () =>
            window.removeEventListener("hifi:timelineEditOp", onTimelineEditOp as EventListener);
    }, [
        dispatch,
        store,
        copyClips,
        cutClips,
        pasteClipsAtPlayhead,
        splitSelectedAtPlayhead,
        normalizeClips,
        groupClips,
        ungroupClips,
        setMultiSelectedClipIds,
    ]);

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
    }, [dispatch, rowHeight, scrollRef, sessionRef, trackListScrollRef]);

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
    }, [dispatch, sessionRef]);

    // ── hifi:zoomTimelineFocus ───────────────────────────────
    useEffect(() => {
        function onZoomFocused(e: Event) {
            const inTimeline = getActiveSurface() === "timeline";
            if (!inTimeline) return;

            const factor = Number((e as CustomEvent<{ factor?: number }>).detail?.factor ?? 1);
            if (!Number.isFinite(factor) || factor <= 0) return;

            const scroller = scrollRef.current;
            if (!scroller) return;

            // 实时工程长度：本监听的依赖刻意不含 dynamicProjectSec（工程变化
            // 不重建监听），闭包值是挂载时的快照——工程变长后缩放上限仍钳在
            // 旧工程末端，视图无法到达当前允许的滚动范围。sessionRef 在
            // store 订阅内同步更新，事件触发时读取的必然是当前值。
            const totalSec = getDynamicProjectSec(sessionRef.current.clips);
            const zoom = resolveHorizontalWheelZoom({
                factor,
                basePxPerSec: pxPerSecRef.current,
                baseScrollLeft: scroller.scrollLeft,
                totalSec,
                viewportWidth: scroller.clientWidth,
                playheadZoomEnabled: true,
                playheadSec: getPlayheadSec
                    ? getPlayheadSec()
                    : Number(sessionRef.current.playheadSec ?? 0) || 0,
                anchorScreenX: 0,
                minPxPerSec: resolveTimelineMinPxPerSec({
                    baseMinPxPerSec: MIN_PX_PER_SEC,
                    projectSec: totalSec,
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
            // 同帧提交 scrollLeft state，防止窗口化把屏内 Clip 裁掉。
            flushSync(() => {
                setPxPerSec(zoom.nextPxPerSec);
                commitScrollLeftState(zoom.nextScrollLeft);
            });
        }

        window.addEventListener("hifi:zoomTimelineFocus", onZoomFocused as EventListener);
        return () =>
            window.removeEventListener("hifi:zoomTimelineFocus", onZoomFocused as EventListener);
        // 工程长度经 sessionRef 实时读取，监听无需随工程变化重建。
    }, [
        commitScrollLeftState,
        getPlayheadSec,
        keyboardZoomPendingRef,
        pxPerSecRef,
        scrollRef,
        sessionRef,
        setPxPerSec,
    ]);

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
    }, [contextMenu, setContextMenu, setTrackAreaMenu, trackAreaMenu]);

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
                playheadSec: getPlayheadSec
                    ? getPlayheadSec()
                    : Number(sessionRef.current.playheadSec ?? 0) || 0,
                pxPerSec,
                // 与 zoomTimelineFocus 同源：实时读取工程长度（sessionRef 在
                // store 订阅内同步更新），不依赖渲染期闭包。
                contentWidth: getDynamicProjectSec(sessionRef.current.clips) * pxPerSec,
            });
            // 写后回读浏览器实际接受的偏移再广播：请求值可能被钳制/量化/锚定
            // 修正，sticky 画布层必须与原生 DOM 层使用同一偏移。
            const applied = applyNativeScrollLeft(scroller, next);
            syncScrollLeft(applied);
        }
        window.addEventListener("hifi:focusCursor", handler);
        return () => window.removeEventListener("hifi:focusCursor", handler);
    }, [getPlayheadSec, pxPerSec, sessionRef, syncScrollLeft, scrollRef]);
}
