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
import React, {
    useEffect,
    useLayoutEffect,
    useMemo,
    useRef,
    useState,
} from "react";
import { useAppDispatch, useAppSelector } from "../../../../app/hooks";
import type { RootState } from "../../../../app/store";
import { timelineViewportBus } from "../../../../utils/timelineViewportBus";

import { waveformMipmapStore } from "../../../../utils/waveformMipmapStore";
import {
    seekPlayhead,
    setplayheadSec,
} from "../../../../features/session/sessionSlice";
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
    gridStepBeats,
} from "../";

// ── 返回类型 ─────────────────────────────────────────────────────
export interface TimelineStateResult {
    // Redux
    dispatch: ReturnType<typeof useAppDispatch>;
    s: RootState["session"];
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
    setTrackVolumeUi: React.Dispatch<
        React.SetStateAction<Record<string, number>>
    >;
    sameSourceConfirmOpen: boolean;
    setSameSourceConfirmOpen: React.Dispatch<React.SetStateAction<boolean>>;
    sameSourceConfirmResolverRef: React.MutableRefObject<
        ((confirmed: boolean) => void) | null
    >;

    // Derived
    secPerBeat: number;
    pxPerBeat: number;
    contentWidth: number;
    contentHeight: number;
    dynamicProjectSec: number;
    bars: Array<{ beat: number; label: string }>;
    clipsByTrackId: Map<string, RootState["session"]["clips"]>;
    viewportStartSec: number;
    viewportEndSec: number;

    // Keybinding refs / values
    stretchKbRef: React.MutableRefObject<Keybinding>;
    scrollHorizontalKb: Keybinding;
    scrollVerticalKb: Keybinding;
    horizontalZoomKb: Keybinding;
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
    secFromClientX: (
        clientX: number,
        bounds: DOMRect,
        xScroll: number,
    ) => number;
    beatFromClientX: (
        clientX: number,
        bounds: DOMRect,
        xScroll: number,
    ) => number;
    trackIdFromClientY: (clientY: number) => string | null;
    rowTopForTrackId: (trackId: string | null) => number;
    ensureDropPreviewDuration: (path: string) => void;
    getDropPreviewWidthPx: (durationSec: number) => number;
    snapSec: (sec: number) => number;
    snapBeat: (sec: number) => number;
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
    const s = useAppSelector((state: RootState) => state.session);
    const sessionRef = useRef(s);
    useEffect(() => {
        sessionRef.current = s;
    }, [s]);

    // ── DOM refs ──────────────────────────────────────────────
    const scrollRef = useRef<HTMLDivElement | null>(null);
    const trackListScrollRef = useRef<HTMLDivElement | null>(null);
    const rulerContentRef = useRef<HTMLDivElement | null>(null);
    const scrollLeftRef = useRef(0);
    const scrollStateRafRef = useRef<number | null>(null);
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
    const sameSourceConfirmResolverRef = useRef<
        ((confirmed: boolean) => void) | null
    >(null);

    useEffect(() => {
        scrollLeftRef.current = scrollLeft;
    }, [scrollLeft]);

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
        if (rulerContentRef.current) {
            rulerContentRef.current.style.transform = `translateX(${-next}px)`;
        }
        // ★ 立即广播视口变化 → WaveformTrackCanvas 直接 invalidate（绕过 React）
        timelineViewportBus.emit(next, pxPerSecRef.current, viewportWidthRef.current);
        // 50ms setTimeout 降频 React state 更新
        if (scrollStateRafRef.current == null) {
            scrollStateRafRef.current = setTimeout(() => {
                scrollStateRafRef.current = null;
                setScrollLeft(scrollLeftRef.current);
            }, 50) as unknown as number;
        }
    }

    const setScrollLeftAction: React.Dispatch<React.SetStateAction<number>> = (
        action,
    ) => {
        const next =
            typeof action === "function"
                ? (action as (prev: number) => number)(scrollLeftRef.current)
                : action;
        syncScrollLeft(next);
    };

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
    const [trackVolumeUi, setTrackVolumeUi] = useState<
        Record<string, number>
    >({});

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
    const stretchKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.clipStretch"),
    );
    const slipEditKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.clipSlipEdit"),
    );
    const noSnapKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.clipNoSnap"),
    );
    const copyDragKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.clipCopyDrag"),
    );
    const scrollHorizontalKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollHorizontal"),
    );
    const scrollVerticalKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollVertical"),
    );
    const horizontalZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.horizontalZoom"),
    );
    const stretchKbRef = useRef<Keybinding>(stretchKb);
    useEffect(() => {
        stretchKbRef.current = stretchKb;
    }, [stretchKb]);

    // ── altPressed key listeners ─────────────────────────────
    useEffect(() => {
        function isStretchModifier(e: KeyboardEvent): boolean {
            const kb = stretchKbRef.current;
            if (kb.ctrl && (e.key === "Control" || e.ctrlKey || e.metaKey))
                return true;
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
    const dynamicProjectSec = useMemo(
        () => getDynamicProjectSec(s.clips),
        [s.clips],
    );

    const contentWidth = useMemo(
        () => Math.max(1, Math.ceil(dynamicProjectSec * pxPerSec)),
        [dynamicProjectSec, pxPerSec],
    );

    const dropExtraRows =
        (dropPreview && !dropPreview.trackId ? 1 : 0) +
        (clipDropNewTrack ? 1 : 0);
    const contentHeight =
        (s.tracks.length + dropExtraRows) * rowHeight + TRACK_ADD_ROW_HEIGHT;

    // ── bars ─────────────────────────────────────────────────
    const bars = useMemo(() => {
        const beatsPerBar = Math.max(1, Math.round(s.beats || 4));
        const secPerBeatLocal = 60 / Math.max(1, s.bpm);
        const totalBeats = Math.max(
            1,
            Math.ceil(dynamicProjectSec / secPerBeatLocal),
        );
        const totalBars = Math.max(1, Math.ceil(totalBeats / beatsPerBar));

        let startBarIndex = 0;
        let endBarIndex = totalBars;

        if (Number.isFinite(viewportWidth) && viewportWidth > 0) {
            const beatPx = Math.max(1e-9, secPerBeatLocal * pxPerSec);
            const bufferPx = Math.max(240, viewportWidth * 0.5);
            const leftPx = Math.max(0, scrollLeft - bufferPx);
            const rightPx = scrollLeft + viewportWidth + bufferPx;

            const leftBeat = leftPx / beatPx;
            const rightBeat = rightPx / beatPx;

            startBarIndex = Math.max(
                0,
                Math.floor(leftBeat / beatsPerBar) - 1,
            );
            endBarIndex = Math.min(
                totalBars,
                Math.ceil(rightBeat / beatsPerBar) + 1,
            );
        }

        const result: Array<{ beat: number; label: string }> = [];
        for (
            let barIndex = startBarIndex;
            barIndex <= endBarIndex;
            barIndex += 1
        ) {
            const beat = barIndex * beatsPerBar;
            if (beat > totalBeats) break;
            result.push({ beat, label: `${barIndex + 1}.1` });
        }
        return result;
    }, [s.beats, dynamicProjectSec, s.bpm, viewportWidth, pxPerSec, scrollLeft]);

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
        if (!trackId) {
            return s.tracks.length * rowHeight;
        }
        const idx = s.tracks.findIndex((t) => t.id === trackId);
        if (idx < 0) {
            return s.tracks.length * rowHeight;
        }
        return idx * rowHeight;
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
        return durationSec > 0
            ? Math.max(1, pxPerSecRef.current * durationSec)
            : 80;
    }

    // ── Playhead helpers ─────────────────────────────────────
    const setPlayheadFromClientX = React.useCallback(
        (
            clientX: number,
            bounds: DOMRect,
            xScroll: number,
            commit: boolean,
        ) => {
            const beat = beatFromClientX(clientX, bounds, xScroll);

            if (commit) {
                dispatch(setplayheadSec(beat));
                void dispatch(seekPlayhead(beat));
            } else {
                if (playheadRef.current) {
                    playheadRef.current.style.left = `${beat * pxPerSecRef.current}px`;
                }
            }
            return beat;
        },
        [beatFromClientX, dispatch],
    );

    const startDeferredPlayheadSeek = React.useCallback(
        (args: {
            startClientX: number;
            startClientY: number;
            getBounds: () => DOMRect | null;
            getScrollLeft: () => number;
        }) => {
            const { startClientX, startClientY, getBounds, getScrollLeft } =
                args;
            let moved = false;
            let lastSec = 0;

            const updateAt = (clientX: number, commit: boolean) => {
                const bounds = getBounds();
                if (!bounds) return null;
                const sec = setPlayheadFromClientX(
                    clientX,
                    bounds,
                    getScrollLeft(),
                    commit,
                );
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

    // ── snapSec / snapBeat ───────────────────────────────────
    function snapSec(sec: number) {
        const stepBeats = gridStepBeats(s.grid);
        const stepSec = stepBeats * (60 / Math.max(1, s.bpm));
        return Math.round(sec / stepSec) * stepSec;
    }
    const snapBeat = snapSec;

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
        const horizontalScrollbarHeight =
            scroller.offsetHeight - scroller.clientHeight;
        if (
            horizontalScrollbarHeight > 0 &&
            clientY > bounds.bottom - horizontalScrollbarHeight
        ) {
            return true;
        }
        const verticalScrollbarWidth =
            scroller.offsetWidth - scroller.clientWidth;
        if (
            verticalScrollbarWidth > 0 &&
            clientX > bounds.right - verticalScrollbarWidth
        ) {
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
    const viewportEndSec =
        (scrollLeft + viewportWidth) / Math.max(1e-9, pxPerSec);

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
        bars,
        clipsByTrackId,
        viewportStartSec,
        viewportEndSec,

        stretchKbRef,
        scrollHorizontalKb,
        scrollVerticalKb,
        horizontalZoomKb,
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
        snapSec,
        snapBeat,
        isEditableTarget,
        isPointerOnNativeScrollbar,
        startPanPointer,
        setPlayheadFromClientX,
        startDeferredPlayheadSeek,

        keyboardZoomPendingRef,
    };
}
