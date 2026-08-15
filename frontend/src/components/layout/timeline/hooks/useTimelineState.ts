/**
 * useTimelineState — Timeline 面板的所有 state / ref / viewport / scroll 逻辑
 *
 * 从 TimelinePanel.tsx 拆分而来，集中管理：
 * - useState / useRef 声明
 * - pxPerSec / rowHeight 持久化 & 缩放
 * - viewport 尺寸监测（ResizeObserver）
 * - syncScrollLeft → DOM 直通 + timelineViewportBus
 * - secFromClientX / trackIdFromClientY / rowTopForTrackId 坐标转换
 * - snapSec / isEditableTarget / isPointerOnNativeScrollbar 工具函数
 * - startPanPointer 中键平移
 * - setPlayheadFromClientX / startDeferredPlayheadSeek 播放头拖拽
 * - altPressed (stretch modifier) 键盘监听
 * - bars / clipsByTrackId / contentWidth/Height 派生计算
 * - Mipmap 预加载
 */
import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useAppDispatch, useAppSelector } from "../../../../app/hooks";
import { store, type RootState } from "../../../../app/store";
import { shallowEqual } from "react-redux";
import { timelineViewportBus } from "../../../../utils/timelineViewportBus";
import { timelineViewportSync } from "../../../../utils/timelineViewportSync";

import { waveformMipmapStore } from "../../../../utils/waveformMipmapStore";
import { seekPlayhead, setplayheadSec } from "../../../../features/session/sessionSlice";
import { selectKeybinding } from "../../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../../features/keybindings/types";
import { getDynamicProjectSec } from "../../../../features/session/projectBoundary";
import {
    DEFAULT_PX_PER_SEC,
    DEFAULT_ROW_HEIGHT,
    MAX_PX_PER_SEC,
    MAX_ROW_HEIGHT,
    MIN_PX_PER_SEC,
    MIN_ROW_HEIGHT,
    TRACK_ADD_ROW_HEIGHT,
    buildRulerTicks,
    gridStepBeats,
} from "../";
import type { RulerTick } from "../timeFormat.js";
import {
    buildTempoGridLineXsForViewport,
} from "../../../../utils/tempoMap.js";
import {
    snapTimelinePosition,
    type SnapObjectKind,
} from "../../../../utils/timelineSnapping";

// ── 返回类型 ─────────────────────────────────────────────────────
type TimelineSessionSlice = Pick<
    RootState["session"],
    | "autoCrossfadeEnabled"
    | "autoScrollEnabled"
    | "beats"
    | "bpm"
    | "clips"
    | "clipFormantStatus"
    | "clipFormantToolWindow"
    | "customScalePresets"
    | "grid"
    | "gridSnapEnabled"
    | "timelineSnap"
    | "paramEditorSyncTimeline"
    | "paramEditorTimelineClickSelectTrackEnabled"
    | "playheadSec"
    | "primaryTimeUnit"
    | "project"
    | "secondaryTimeUnit"
    | "rulerLabelSpacingPx"
    | "showPlayheadTimeInTrackHeader"
    | "playheadZoomEnabled"
    | "selectedClipId"
    | "selectedTrackId"
    | "tempoMap"
    | "tempoMapVisible"
    | "trackMeters"
    | "tracks"
>;

export interface TimelineStateResult {
    // Redux
    dispatch: ReturnType<typeof useAppDispatch>;
    s: TimelineSessionSlice;
    sessionRef: React.MutableRefObject<RootState["session"]>;

    // DOM refs
    scrollRef: React.MutableRefObject<HTMLDivElement | null>;
    trackListScrollRef: React.MutableRefObject<HTMLDivElement | null>;
    rulerContentRef: React.MutableRefObject<HTMLDivElement | null>;
    playheadRef: React.MutableRefObject<HTMLDivElement | null>;
    dropPreviewRef: React.MutableRefObject<HTMLDivElement | null>;
    playheadDragRef: React.MutableRefObject<{
        pointerId: number;
        lastBeat: number;
    } | null>;
    lastClickedClipIdRef: React.MutableRefObject<string | null>;
    scrollLeftRef: React.MutableRefObject<number>;
    pxPerSecRef: React.MutableRefObject<number>;
    viewportWidthRef: React.MutableRefObject<number>;
    rowHeightRef: React.MutableRefObject<number>;
    panRef: React.MutableRefObject<{
        pointerId: number | null;
        startX: number;
        startY: number;
        scrollLeft: number;
        scrollTop: number;
    } | null>;

    // State values
    scrollLeft: number;
    pxPerSec: number;
    setPxPerSec: React.Dispatch<React.SetStateAction<number>>;
    viewportWidth: number;
    rowHeight: number;
    setRowHeight: React.Dispatch<React.SetStateAction<number>>;
    altPressed: boolean;
    trackVolumeUi: Record<string, number>;
    setTrackVolumeUi: React.Dispatch<React.SetStateAction<Record<string, number>>>;
    sameSourceConfirmOpen: boolean;
    setSameSourceConfirmOpen: React.Dispatch<React.SetStateAction<boolean>>;
    sameSourceConfirmResolverRef: React.MutableRefObject<((confirmed: boolean) => void) | null>;

    // Derived
    secPerBeat: number;
    pxPerBeat: number;
    contentWidth: number;
    contentHeight: number;
    dynamicProjectSec: number;
    ticks: RulerTick[];
    /** Tempo Map 显式网格线（内容坐标 x）；无 Tempo Map 时为 null。 */
    tempoGridLineXs: { weak: number[]; strong: number[] } | null;
    clipsByTrackId: Map<string, RootState["session"]["clips"]>;
    viewportStartSec: number;
    viewportEndSec: number;

    // Keybinding refs / values
    stretchKbRef: React.MutableRefObject<Keybinding>;
    scrollHorizontalKb: Keybinding;
    scrollVerticalKb: Keybinding;
    horizontalZoomKb: Keybinding;
    verticalZoomKb: Keybinding;
    paramFineAdjustKb: Keybinding;
    slipEditKb: Keybinding;
    noSnapKb: Keybinding;
    copyDragKb: Keybinding;

    // Drop preview
    dropPreview: {
        path: string;
        fileName: string;
        trackId: string | null;
        startSec: number;
        durationSec: number;
    } | null;
    setDropPreview: React.Dispatch<
        React.SetStateAction<{
            path: string;
            fileName: string;
            trackId: string | null;
            startSec: number;
            durationSec: number;
        } | null>
    >;
    dropExtraRows: number;
    clipDropNewTrack: boolean;
    setClipDropNewTrack: React.Dispatch<React.SetStateAction<boolean>>;
    pendingDropDurationPathRef: React.MutableRefObject<string | null>;

    // Functions
    syncScrollLeft: (next: number) => void;
    setScrollLeftAction: React.Dispatch<React.SetStateAction<number>>;
    secFromClientX: (clientX: number, bounds: DOMRect, xScroll: number) => number;
    beatFromClientX: (clientX: number, bounds: DOMRect, xScroll: number) => number;
    trackIdFromClientY: (clientY: number) => string | null;
    rowTopForTrackId: (trackId: string | null) => number;
    ensureDropPreviewDuration: (path: string) => void;
    getDropPreviewWidthPx: (durationSec: number) => number;
    /** 完整吸附引擎入口。 */
    snapTimeline: (
        sec: number,
        object: SnapObjectKind,
        opts?: {
            originSec?: number;
            anchorTrackId?: string | null;
            excludeClipIds?: ReadonlySet<string>;
        },
    ) => number;
    isEditableTarget: (target: EventTarget | null) => boolean;
    isPointerOnNativeScrollbar: (
        scroller: HTMLDivElement,
        clientX: number,
        clientY: number,
    ) => boolean;
    startPanPointer: (e: React.PointerEvent) => void;
    setPlayheadFromClientX: (
        clientX: number,
        bounds: DOMRect,
        xScroll: number,
        commit: boolean,
    ) => number;
    startDeferredPlayheadSeek: (args: {
        startClientX: number;
        startClientY: number;
        getBounds: () => DOMRect | null;
        getScrollLeft: () => number;
    }) => void;

    // Keyboard zoom pending ref (needed in useLayoutEffect)
    keyboardZoomPendingRef: React.MutableRefObject<{
        nextScale: number;
        nextScrollLeft: number;
    } | null>;
}

// ── Hook 实现 ────────────────────────────────────────────────────
export function useTimelineState(): TimelineStateResult {
    const dispatch = useAppDispatch();
    const s = useAppSelector(
        (state: RootState) => ({
            autoCrossfadeEnabled: state.session.autoCrossfadeEnabled,
            autoScrollEnabled: state.session.autoScrollEnabled,
            beats: state.session.beats,
            bpm: state.session.bpm,
            clips: state.session.clips,
            clipFormantStatus: state.session.clipFormantStatus,
            clipFormantToolWindow: state.session.clipFormantToolWindow,
            customScalePresets: state.session.customScalePresets,
            grid: state.session.grid,
            gridSnapEnabled: state.session.gridSnapEnabled,
            timelineSnap: state.session.timelineSnap,
            playheadSec: state.session.playheadSec,
            playheadZoomEnabled: state.session.playheadZoomEnabled,
            paramEditorSyncTimeline: state.session.paramEditorSyncTimeline,
            paramEditorTimelineClickSelectTrackEnabled:
                state.session.paramEditorTimelineClickSelectTrackEnabled,
            primaryTimeUnit: state.session.primaryTimeUnit,
            playbackRateVersion: state.session.playbackRateVersion,
            project: state.session.project,
            rulerLabelSpacingPx: state.session.rulerLabelSpacingPx,
            secondaryTimeUnit: state.session.secondaryTimeUnit,
            selectedClipId: state.session.selectedClipId,
            selectedTrackId: state.session.selectedTrackId,
            showPlayheadTimeInTrackHeader: state.session.showPlayheadTimeInTrackHeader,
            tempoMap: state.session.tempoMap,
            tempoMapVisible: state.session.tempoMapVisible,
            trackMeters: state.session.trackMeters,
            tracks: state.session.tracks,
        }),
        shallowEqual,
    );
    const sessionRef = useRef(store.getState().session);
    useEffect(() => {
        sessionRef.current = store.getState().session;
        return store.subscribe(() => {
            sessionRef.current = store.getState().session;
        });
    }, []);

    // ── DOM refs ──────────────────────────────────────────────
    const scrollRef = useRef<HTMLDivElement | null>(null);
    const trackListScrollRef = useRef<HTMLDivElement | null>(null);
    const rulerContentRef = useRef<HTMLDivElement | null>(null);
    const scrollLeftRef = useRef(0);
    const scrollStateRafRef = useRef<number | null>(null);
    const paramEditorSyncTimelineRef = useRef(s.paramEditorSyncTimeline);
    paramEditorSyncTimelineRef.current = s.paramEditorSyncTimeline;
    const timelineSyncApplyingRef = useRef(false);
    const pendingTimelineSyncViewportRef = useRef<{
        scrollLeft: number;
        pxPerSec: number;
    } | null>(null);
    const playheadDragRef = useRef<{
        pointerId: number;
        lastBeat: number;
    } | null>(null);
    const lastClickedClipIdRef = useRef<string | null>(null);
    const playheadRef = useRef<HTMLDivElement | null>(null);
    const dropPreviewRef = useRef<HTMLDivElement | null>(null);
    const pendingDropDurationPathRef = useRef<string | null>(null);

    // ── State 声明 ────────────────────────────────────────────
    const [scrollLeft, setScrollLeft] = useState(0);
    const [pxPerSec, setPxPerSec] = useState(() => {
        const stored = Number(localStorage.getItem("hifishifter.pxPerSec"));
        return Number.isFinite(stored) && stored > 0
            ? Math.min(MAX_PX_PER_SEC, Math.max(MIN_PX_PER_SEC, stored))
            : DEFAULT_PX_PER_SEC;
    });
    const pxPerSecRef = useRef(pxPerSec);
    pxPerSecRef.current = pxPerSec; // 渲染期直接同步，确保 syncScrollLeft emit 时值最新

    const keyboardZoomPendingRef = useRef<{
        nextScale: number;
        nextScrollLeft: number;
    } | null>(null);

    const [viewportWidth, setViewportWidth] = useState(0);
    const viewportWidthRef = useRef(0);
    useEffect(() => {
        viewportWidthRef.current = viewportWidth;
    }, [viewportWidth]);

    const [sameSourceConfirmOpen, setSameSourceConfirmOpen] = useState(false);
    const sameSourceConfirmResolverRef = useRef<((confirmed: boolean) => void) | null>(null);

    useEffect(() => {
        scrollLeftRef.current = scrollLeft;
    }, [scrollLeft]);

    useEffect(() => {
        return () => {
            if (scrollStateRafRef.current != null) {
                cancelAnimationFrame(scrollStateRafRef.current);
                scrollStateRafRef.current = null;
            }
        };
    }, []);

    // ── ResizeObserver → viewportWidth ────────────────────────
    useEffect(() => {
        const scroller = scrollRef.current;
        if (!scroller) return;

        const updateViewportWidth = () => {
            setViewportWidth(scroller.clientWidth || 0);
        };

        updateViewportWidth();

        if (typeof ResizeObserver !== "undefined") {
            const observer = new ResizeObserver(() => {
                updateViewportWidth();
            });
            observer.observe(scroller);
            return () => {
                observer.disconnect();
            };
        }

        window.addEventListener("resize", updateViewportWidth);
        return () => {
            window.removeEventListener("resize", updateViewportWidth);
        };
    }, []);

    // ── syncScrollLeft → DOM 直通 + bus ───────────────────────
    function syncScrollLeft(next: number) {
        scrollLeftRef.current = next;
        if (paramEditorSyncTimelineRef.current && !timelineSyncApplyingRef.current) {
            timelineViewportSync.setViewport({
                scrollLeft: next,
                pxPerSec: pxPerSecRef.current,
            });
        }
        if (rulerContentRef.current) {
            rulerContentRef.current.style.transform = `translateX(${-next}px)`;
        }
        // ★ 立即广播视口变化 → WaveformTrackCanvas 直接 invalidate（绕过 React）
        timelineViewportBus.emit(next, pxPerSecRef.current, viewportWidthRef.current);
        // 用 rAF 合并状态更新，保证自动滚屏可达 60Hz 且避免同步抖动
        if (scrollStateRafRef.current == null) {
            scrollStateRafRef.current = requestAnimationFrame(() => {
                scrollStateRafRef.current = null;
                setScrollLeft(scrollLeftRef.current);
            });
        }
    }

    const setScrollLeftAction: React.Dispatch<React.SetStateAction<number>> = (action) => {
        const next =
            typeof action === "function"
                ? (action as (prev: number) => number)(scrollLeftRef.current)
                : action;
        syncScrollLeft(next);
    };

    // 同步开关（双向交互）：订阅共享视口，并把参数编辑器写入的值应用到轨道视图。
    useEffect(() => {
        if (!s.paramEditorSyncTimeline) return;
        const apply = () => {
            const store = timelineViewportSync.get();
            timelineSyncApplyingRef.current = true;
            pendingTimelineSyncViewportRef.current = {
                scrollLeft: store.scrollLeft,
                pxPerSec: store.pxPerSec,
            };
            setScrollLeft(store.scrollLeft);
            setPxPerSec(store.pxPerSec);
            timelineSyncApplyingRef.current = false;
        };
        const unsubscribe = timelineViewportSync.subscribe(apply);
        return () => {
            unsubscribe();
            pendingTimelineSyncViewportRef.current = null;
        };
    }, [s.paramEditorSyncTimeline]);

    // 启用同步时，立即把轨道视图当前的水平位置与缩放写入共享视口作为基准。
    useEffect(() => {
        if (s.paramEditorSyncTimeline) {
            timelineViewportSync.setViewport({
                scrollLeft: scrollLeftRef.current,
                pxPerSec: pxPerSecRef.current,
            });
        }
    }, [s.paramEditorSyncTimeline]);

    // 同步视口必须等内容宽度按新 pxPerSec 更新后再落到 DOM。
    // 否则设置 scroller.scrollLeft 时会被浏览器钳回旧的最大滚动位置，
    // 形成“缩放已变、滚动没变”的水平漂移。
    useLayoutEffect(() => {
        const pending = pendingTimelineSyncViewportRef.current;
        if (!pending || !s.paramEditorSyncTimeline) return;
        if (Math.abs(pxPerSec - pending.pxPerSec) > 1e-9) return;
        if (Math.abs(scrollLeft - pending.scrollLeft) > 0.5) return;

        pendingTimelineSyncViewportRef.current = null;
        const scroller = scrollRef.current;
        if (!scroller) return;

        timelineSyncApplyingRef.current = true;
        scroller.scrollLeft = pending.scrollLeft;
        syncScrollLeft(pending.scrollLeft);
        timelineSyncApplyingRef.current = false;
    }, [pxPerSec, scrollLeft, s.paramEditorSyncTimeline]);

    // ── keyboard zoom layout effect ──────────────────────────
    useLayoutEffect(() => {
        const pending = keyboardZoomPendingRef.current;
        if (!pending) return;
        if (Math.abs(pending.nextScale - pxPerSec) > 1e-9) return;
        const scroller = scrollRef.current;
        if (!scroller) return;

        keyboardZoomPendingRef.current = null;
        scroller.scrollLeft = pending.nextScrollLeft;
        syncScrollLeft(pending.nextScrollLeft);
    }, [pxPerSec]);

    // ── pxPerBeat / secPerBeat ───────────────────────────────
    const secPerBeat = 60 / Math.max(1, s.bpm);
    const pxPerBeat = pxPerSec * secPerBeat;

    // ── rowHeight ────────────────────────────────────────────
    const [rowHeight, setRowHeight] = useState(() => {
        const stored = Number(localStorage.getItem("hifishifter.rowHeight"));
        return Number.isFinite(stored)
            ? Math.min(MAX_ROW_HEIGHT, Math.max(MIN_ROW_HEIGHT, stored))
            : DEFAULT_ROW_HEIGHT;
    });
    const rowHeightRef = useRef(rowHeight);
    useEffect(() => {
        rowHeightRef.current = rowHeight;
    }, [rowHeight]);

    // ── pan ref ──────────────────────────────────────────────
    const panRef = useRef<{
        pointerId: number | null;
        startX: number;
        startY: number;
        scrollLeft: number;
        scrollTop: number;
    } | null>(null);

    // ── trackVolumeUi ────────────────────────────────────────
    const [trackVolumeUi, setTrackVolumeUi] = useState<Record<string, number>>({});

    // ── dropPreview ──────────────────────────────────────────
    const [dropPreview, setDropPreview] = useState<{
        path: string;
        fileName: string;
        trackId: string | null;
        startSec: number;
        durationSec: number;
    } | null>(null);
    const [clipDropNewTrack, setClipDropNewTrack] = useState(false);

    // ── altPressed (stretch modifier key) ────────────────────
    const [altPressed, setAltPressed] = useState(false);

    // ── Keybindings ──────────────────────────────────────────
    const stretchKb = useAppSelector((state) => selectKeybinding(state, "modifier.clipStretch"));
    const slipEditKb = useAppSelector((state) => selectKeybinding(state, "modifier.clipSlipEdit"));
    const noSnapKb = useAppSelector((state) => selectKeybinding(state, "modifier.clipNoSnap"));
    const copyDragKb = useAppSelector((state) => selectKeybinding(state, "modifier.clipCopyDrag"));
    const scrollHorizontalKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollHorizontal"),
    );
    const scrollVerticalKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollVertical"),
    );
    const horizontalZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.horizontalZoom"),
    );
    const verticalZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.pianoRollVerticalZoom"),
    );
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );
    const stretchKbRef = useRef<Keybinding>(stretchKb);
    useEffect(() => {
        stretchKbRef.current = stretchKb;
    }, [stretchKb]);

    // ── altPressed key listeners ─────────────────────────────
    useEffect(() => {
        function isStretchModifier(e: KeyboardEvent): boolean {
            const kb = stretchKbRef.current;
            if (kb.ctrl && (e.key === "Control" || e.ctrlKey || e.metaKey)) return true;
            if (kb.alt && (e.key === "Alt" || e.altKey)) return true;
            if (kb.shift && (e.key === "Shift" || e.shiftKey)) return true;
            return false;
        }
        function checkStretchState(e: KeyboardEvent): boolean {
            const kb = stretchKbRef.current;
            if (kb.ctrl) return e.ctrlKey || e.metaKey;
            if (kb.alt) return e.altKey;
            if (kb.shift) return e.shiftKey;
            return false;
        }
        function onKeyDown(e: KeyboardEvent) {
            if (isStretchModifier(e)) setAltPressed(true);
        }
        function onKeyUp(e: KeyboardEvent) {
            if (!checkStretchState(e)) setAltPressed(false);
        }
        function onBlur() {
            setAltPressed(false);
        }
        window.addEventListener("keydown", onKeyDown, true);
        window.addEventListener("keyup", onKeyUp, true);
        window.addEventListener("blur", onBlur);
        return () => {
            window.removeEventListener("keydown", onKeyDown, true);
            window.removeEventListener("keyup", onKeyUp, true);
            window.removeEventListener("blur", onBlur);
        };
    }, []);

    // ── dynamicProjectSec / contentWidth / contentHeight ─────
    const dynamicProjectSec = useMemo(() => getDynamicProjectSec(s.clips), [s.clips]);

    const contentWidth = useMemo(
        () => Math.max(1, Math.ceil(dynamicProjectSec * pxPerSec)),
        [dynamicProjectSec, pxPerSec],
    );

    const dropExtraRows =
        (dropPreview && !dropPreview.trackId ? 1 : 0) + (clipDropNewTrack ? 1 : 0);
    const contentHeight = (s.tracks.length + dropExtraRows) * rowHeight + TRACK_ADD_ROW_HEIGHT;

    // ── ticks（自适应标尺刻度）──────────────────────────────────
    const ticks = useMemo(() => {
        const beatsPerBar = Math.max(1, Math.round(s.beats || 4));
        return buildRulerTicks({
            pxPerSec,
            scrollLeft,
            viewportWidth: Number.isFinite(viewportWidth) ? viewportWidth : 0,
            projectSec: dynamicProjectSec,
            bpm: s.bpm,
            beatsPerBar,
            grid: s.grid,
            primaryUnit: s.primaryTimeUnit,
            secondaryUnit: s.secondaryTimeUnit,
            minLabelSpacingPx: s.rulerLabelSpacingPx,
            tempoMap: s.tempoMap,
        });
    }, [
        s.beats,
        s.bpm,
        s.grid,
        s.primaryTimeUnit,
        s.secondaryTimeUnit,
        s.rulerLabelSpacingPx,
        s.tempoMap,
        dynamicProjectSec,
        viewportWidth,
        pxPerSec,
        scrollLeft,
    ]);

    // ── Tempo Map 显式网格线（供 BackgroundGrid 使用）──────────
    const tempoGridLineXs = useMemo(() => {
        return buildTempoGridLineXsForViewport({
            tempoMap: s.tempoMap,
            scrollLeft,
            viewportWidth: Number.isFinite(viewportWidth) ? viewportWidth : 0,
            pxPerSec,
            projectSec: dynamicProjectSec,
            stepBeats: gridStepBeats(s.grid),
            fallbackBpm: s.bpm,
            fallbackBeatsPerBar: Math.max(1, Math.round(s.beats || 4)),
            swingPercent: s.timelineSnap.swingEnabled ? s.timelineSnap.swingPercent : 0,
            minSpacingPx: s.timelineSnap.gridMinSpacingPx,
        });
    }, [s.tempoMap, s.bpm, s.beats, s.grid, s.timelineSnap, scrollLeft, viewportWidth, pxPerSec, dynamicProjectSec]);

    // ── clipsByTrackId ───────────────────────────────────────
    const clipsByTrackId = useMemo(() => {
        const map = new Map<string, typeof s.clips>();
        for (const clip of s.clips) {
            const arr = map.get(clip.trackId);
            if (arr) {
                arr.push(clip);
            } else {
                map.set(clip.trackId, [clip]);
            }
        }

        for (const arr of map.values()) {
            arr.sort((a, b) => {
                const d = (a.startSec ?? 0) - (b.startSec ?? 0);
                if (Math.abs(d) > 1e-9) return d;
                return String(a.id).localeCompare(String(b.id));
            });
        }

        return map;
    }, [s.clips]);

    // ── Mipmap 预加载 ────────────────────────────────────────
    const preloadedPathsRef = useRef(new Set<string>());
    useEffect(() => {
        const newPaths: string[] = [];
        for (const clip of s.clips) {
            const sp = clip.sourcePath;
            if (sp && !preloadedPathsRef.current.has(sp)) {
                preloadedPathsRef.current.add(sp);
                newPaths.push(sp);
            }
        }
        if (newPaths.length > 0) {
            void waveformMipmapStore.batchPreload(newPaths);
        }
    }, [s.clips]);

    // ── 坐标转换函数 ─────────────────────────────────────────
    const secFromClientX = React.useCallback(
        (clientX: number, bounds: DOMRect, xScroll: number) => {
            const x = clientX - bounds.left + xScroll;
            return Math.max(0, x / pxPerSecRef.current);
        },
        [],
    );
    const beatFromClientX = secFromClientX;

    function trackIdFromClientY(clientY: number) {
        const scroller = scrollRef.current;
        if (!scroller) return null;
        const bounds = scroller.getBoundingClientRect();
        const y = clientY - bounds.top + scroller.scrollTop;
        const idx = Math.floor(y / rowHeightRef.current);
        const tracks = sessionRef.current.tracks;
        if (idx < 0 || idx >= tracks.length) return null;
        return tracks[idx]?.id ?? null;
    }

    function rowTopForTrackId(trackId: string | null) {
        const tracks = sessionRef.current.tracks;
        const rowHeightPx = rowHeightRef.current;
        if (!trackId) {
            return tracks.length * rowHeightPx;
        }
        const idx = tracks.findIndex((t) => t.id === trackId);
        if (idx < 0) {
            return tracks.length * rowHeightPx;
        }
        return idx * rowHeightPx;
    }

    // ── Drop preview helpers ─────────────────────────────────
    function ensureDropPreviewDuration(path: string) {
        if (!path || pendingDropDurationPathRef.current === path) return;
        pendingDropDurationPathRef.current = path;
        void import("../../../../services/api/fileBrowser")
            .then(({ fileBrowserApi }) => fileBrowserApi.getAudioFileInfo(path))
            .then((info) => {
                setDropPreview((prev) => {
                    if (!prev || prev.path !== path) return prev;
                    return {
                        ...prev,
                        durationSec: Math.max(
                            0,
                            Number(info?.durationSec ?? prev.durationSec) || 0,
                        ),
                    };
                });
            })
            .catch(() => undefined)
            .finally(() => {
                if (pendingDropDurationPathRef.current === path) {
                    pendingDropDurationPathRef.current = null;
                }
            });
    }

    function getDropPreviewWidthPx(durationSec: number) {
        return durationSec > 0 ? Math.max(1, pxPerSecRef.current * durationSec) : 80;
    }

    // ── snapTimeline ─────────────────────────────────────────
    const snapTimeline = React.useCallback(
        (
            sec: number,
            object: SnapObjectKind,
            opts?: {
                originSec?: number;
                anchorTrackId?: string | null;
                excludeClipIds?: ReadonlySet<string>;
            },
        ) => {
            const session = sessionRef.current;
            const result = snapTimelinePosition(
                {
                    settings: session.timelineSnap,
                    grid: session.grid,
                    bpm: session.bpm,
                    beatsPerBar: session.beats,
                    tempoMap: session.tempoMap,
                    pxPerSec: pxPerSecRef.current,
                    clips: session.clips,
                    tracks: session.tracks,
                    selectedClipIds:
                        session.multiSelectedClipIds.length > 0
                            ? session.multiSelectedClipIds
                            : session.selectedClipId
                              ? [session.selectedClipId]
                              : [],
                    playheadSec: session.playheadSec,
                    object,
                    originSec: opts?.originSec,
                    anchorTrackId: opts?.anchorTrackId ?? session.selectedTrackId,
                    excludeClipIds: opts?.excludeClipIds,
                },
                sec,
            );
            return result.sec;
        },
        [],
    );

    // ── Playhead helpers ─────────────────────────────────────
    const setPlayheadFromClientX = React.useCallback(
        (clientX: number, bounds: DOMRect, xScroll: number, commit: boolean) => {
            const rawBeat = beatFromClientX(clientX, bounds, xScroll);
            const beat = snapTimeline(rawBeat, "cursor");

            if (commit) {
                dispatch(setplayheadSec(beat));
                void dispatch(seekPlayhead(beat));
            } else {
                // 更新 Redux state 使三角形头部（TimeRulerPlayhead）与竖线同步
                dispatch(setplayheadSec(beat));
                // 同时直接操作 DOM 确保竖线无延迟跟随
                if (playheadRef.current) {
                    playheadRef.current.style.left = `${beat * pxPerSecRef.current}px`;
                }
            }
            return beat;
        },
        [beatFromClientX, dispatch, snapTimeline],
    );

    const startDeferredPlayheadSeek = React.useCallback(
        (args: {
            startClientX: number;
            startClientY: number;
            getBounds: () => DOMRect | null;
            getScrollLeft: () => number;
        }) => {
            const { startClientX, startClientY, getBounds, getScrollLeft } = args;
            let moved = false;
            let lastSec = 0;

            const updateAt = (clientX: number, commit: boolean) => {
                const bounds = getBounds();
                if (!bounds) return null;
                const sec = setPlayheadFromClientX(clientX, bounds, getScrollLeft(), commit);
                return sec;
            };

            const onMove = (ev: MouseEvent) => {
                const dx = ev.clientX - startClientX;
                const dy = ev.clientY - startClientY;
                if (!moved && dx * dx + dy * dy >= 9) {
                    moved = true;
                }
                if (!moved) return;
                const sec = updateAt(ev.clientX, false);
                if (sec != null) lastSec = sec;
            };

            const onEnd = (ev: MouseEvent) => {
                window.removeEventListener("mousemove", onMove, true);
                window.removeEventListener("mouseup", onEnd, true);
                window.removeEventListener("mouseleave", onEnd, true);

                if (!moved) {
                    updateAt(ev.clientX, true);
                    return;
                }

                const sec = updateAt(ev.clientX, false);
                const finalSec = sec == null ? lastSec : sec;
                void dispatch(seekPlayhead(finalSec));
            };

            window.addEventListener("mousemove", onMove, true);
            window.addEventListener("mouseup", onEnd, true);
            window.addEventListener("mouseleave", onEnd, true);
        },
        [dispatch, setPlayheadFromClientX],
    );

    // ── isEditableTarget ─────────────────────────────────────
    function isEditableTarget(target: EventTarget | null): boolean {
        const el = target as HTMLElement | null;
        if (!el) return false;
        const tag = (el.tagName ?? "").toLowerCase();
        if (tag === "input" || tag === "textarea" || tag === "select") {
            return true;
        }
        if (el.isContentEditable) return true;
        if (el.closest?.('input,textarea,select,[contenteditable="true"]')) {
            return true;
        }
        return false;
    }

    // ── isPointerOnNativeScrollbar ───────────────────────────
    function isPointerOnNativeScrollbar(
        scroller: HTMLDivElement,
        clientX: number,
        clientY: number,
    ): boolean {
        const bounds = scroller.getBoundingClientRect();
        const horizontalScrollbarHeight = scroller.offsetHeight - scroller.clientHeight;
        if (horizontalScrollbarHeight > 0 && clientY > bounds.bottom - horizontalScrollbarHeight) {
            return true;
        }
        const verticalScrollbarWidth = scroller.offsetWidth - scroller.clientWidth;
        if (verticalScrollbarWidth > 0 && clientX > bounds.right - verticalScrollbarWidth) {
            return true;
        }
        return false;
    }

    // ── startPanPointer (中键平移) ───────────────────────────
    function startPanPointer(e: React.PointerEvent) {
        const scroller = scrollRef.current;
        if (!scroller) return;
        if (e.pointerType !== "mouse") return;
        panRef.current = {
            pointerId: e.pointerId,
            startX: e.clientX,
            startY: e.clientY,
            scrollLeft: scroller.scrollLeft,
            scrollTop: scroller.scrollTop,
        };

        const prevCursor = document.body.style.cursor;
        const prevSelect = document.body.style.userSelect;
        document.body.style.cursor = "grabbing";
        document.body.style.userSelect = "none";

        try {
            (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        } catch {
            // ignore
        }

        function onMove(ev: PointerEvent) {
            const pan = panRef.current;
            const el = scrollRef.current;
            if (!pan || !el) return;
            if (pan.pointerId != null && ev.pointerId !== pan.pointerId) return;
            el.scrollLeft = pan.scrollLeft - (ev.clientX - pan.startX);
            el.scrollTop = pan.scrollTop - (ev.clientY - pan.startY);
            syncScrollLeft(el.scrollLeft);
        }

        function end(ev: PointerEvent) {
            const pan = panRef.current;
            if (!pan) return;
            if (pan.pointerId != null && ev.pointerId !== pan.pointerId) return;
            panRef.current = null;
            document.body.style.cursor = prevCursor;
            document.body.style.userSelect = prevSelect;
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", end);
            window.removeEventListener("pointercancel", end);
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", end);
        window.addEventListener("pointercancel", end);
    }

    // ── viewport start/end ───────────────────────────────────
    const viewportStartSec = scrollLeft / Math.max(1e-9, pxPerSec);
    const viewportEndSec = (scrollLeft + viewportWidth) / Math.max(1e-9, pxPerSec);

    // ── Return ───────────────────────────────────────────────
    return {
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
        scrollLeftRef,
        pxPerSecRef,
        viewportWidthRef,
        rowHeightRef,
        panRef,

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
        clipsByTrackId,
        viewportStartSec,
        viewportEndSec,

        stretchKbRef,
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
        dropExtraRows,
        clipDropNewTrack,
        setClipDropNewTrack,
        pendingDropDurationPathRef,

        syncScrollLeft,
        setScrollLeftAction,
        secFromClientX,
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
    };
}
