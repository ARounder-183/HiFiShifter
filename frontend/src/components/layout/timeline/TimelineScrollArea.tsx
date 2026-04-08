import React, { useEffect, useLayoutEffect, useRef } from "react";

import { MAX_PX_PER_SEC, MAX_ROW_HEIGHT, MIN_PX_PER_SEC, MIN_ROW_HEIGHT } from "./constants";
import { clamp } from "./math";
import { isNoneBinding, isModifierActive } from "../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../features/keybindings/types";
import { getTimelineWheelAction } from "../wheelGesture";
import { computeAnchoredHorizontalZoom } from "../../../utils/horizontalZoom";

export const TimelineScrollArea: React.FC<
    Omit<React.HTMLAttributes<HTMLDivElement>, "ref"> & {
        scrollRef: React.MutableRefObject<HTMLDivElement | null>;
        projectSec: number;
        bpm: number;
        pxPerSec: number;
        setPxPerSec: React.Dispatch<React.SetStateAction<number>>;
        rowHeight: number;
        setRowHeight: React.Dispatch<React.SetStateAction<number>>;
        setScrollLeft: React.Dispatch<React.SetStateAction<number>>;
        scrollHorizontalKb?: Keybinding;
        scrollVerticalKb?: Keybinding;
        horizontalZoomKb?: Keybinding;
        verticalZoomKb?: Keybinding;
        playheadSec?: number;
        playheadZoomEnabled?: boolean;
    }
> = ({
    scrollRef,
    projectSec,
    bpm,
    pxPerSec,
    setPxPerSec,
    rowHeight,
    setRowHeight,
    setScrollLeft,
    onScroll,
    onWheel,
    scrollHorizontalKb,
    scrollVerticalKb,
    horizontalZoomKb,
    verticalZoomKb,
    playheadSec,
    playheadZoomEnabled,
    ...divProps
}) => {
    const lastScrollLeftRef = useRef<number | null>(null);
    const pxPerSecRef = useRef(pxPerSec);
    const zoomRafRef = useRef<number | null>(null);
    const zoomPendingRef = useRef<{
        nextPxPerSec: number;
        nextScrollLeft: number;
    } | null>(null);

    // zoom 中心点以秒为基准
    const pendingZoomRef = useRef<{
        nextPxPerSec: number;
        nextScrollLeft: number;
    } | null>(null);

    const rowHeightRef = useRef(rowHeight);
    const pendingVerticalZoomRef = useRef<{
        pointerY: number;
        rowUnitAtPointer: number;
        nextRowHeight: number;
        nextScrollTop: number;
    } | null>(null);

    useEffect(() => {
        pxPerSecRef.current = pxPerSec;
    }, [pxPerSec]);

    useEffect(() => {
        rowHeightRef.current = rowHeight;
    }, [rowHeight]);

    function syncScrollLeft(scroller: HTMLDivElement) {
        const next = scroller.scrollLeft;
        if (lastScrollLeftRef.current != null && lastScrollLeftRef.current === next) {
            return;
        }
        lastScrollLeftRef.current = next;
        setScrollLeft(next);
    }

    useEffect(() => {
        const scroller = scrollRef.current;
        if (!scroller) return;
        syncScrollLeft(scroller);
    }, [scrollRef, setScrollLeft]);

    useEffect(() => {
        return () => {};
    }, []);

    useEffect(() => {
        return () => {
            if (zoomRafRef.current != null) {
                cancelAnimationFrame(zoomRafRef.current);
                zoomRafRef.current = null;
            }
        };
    }, []);

    useLayoutEffect(() => {
        // Apply pending cursor-centered zoom scrollLeft after pxPerSec has updated
        const scroller = scrollRef.current;
        const pending = pendingZoomRef.current;
        if (!scroller || !pending) return;
        if (Math.abs(pending.nextPxPerSec - pxPerSec) > 1e-9) return;

        pendingZoomRef.current = null;
        scroller.scrollLeft = pending.nextScrollLeft;
        syncScrollLeft(scroller);
    }, [projectSec, bpm, pxPerSec, scrollRef]);

    useLayoutEffect(() => {
        const scroller = scrollRef.current;
        const pending = pendingVerticalZoomRef.current;
        if (!scroller || !pending) return;
        if (Math.abs(pending.nextRowHeight - rowHeight) > 1e-9) return;

        pendingVerticalZoomRef.current = null;

        const maxScrollTop = Math.max(0, scroller.scrollHeight - scroller.clientHeight);
        scroller.scrollTop = Math.min(Math.max(0, pending.nextScrollTop), maxScrollTop);
    }, [rowHeight, scrollRef]);

    useEffect(() => {
        localStorage.setItem("hifishifter.pxPerSec", String(pxPerSec));
    }, [pxPerSec]);

    useEffect(() => {
        localStorage.setItem("hifishifter.rowHeight", String(rowHeight));
    }, [rowHeight]);

    useEffect(() => {
        const scroller = scrollRef.current;
        if (!scroller) return;

        const handler: EventListener = (evt) => {
            const e = evt as globalThis.WheelEvent;
            const clipGainKnobEl = (e.target as HTMLElement | null)?.closest(
                "[data-clip-gain-knob]",
            );
            if (clipGainKnobEl) {
                return;
            }
            const noModifierPressed = !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey;
            const isWheelBindingRequested = (kb?: Keybinding) => {
                if (!kb) return false;
                if (isNoneBinding(kb)) return noModifierPressed;
                return isModifierActive(kb, e);
            };
            const horizontalScrollRequested = isWheelBindingRequested(scrollHorizontalKb);
            const verticalScrollRequested = isWheelBindingRequested(scrollVerticalKb);
            const horizontalZoomRequested = isWheelBindingRequested(horizontalZoomKb);
            const verticalZoomRequested = isWheelBindingRequested(verticalZoomKb);

            const wheelAction = getTimelineWheelAction({
                deltaX: e.deltaX,
                deltaY: e.deltaY,
                horizontalScrollRequested,
                verticalScrollRequested,
                verticalZoomRequested,
                horizontalZoomRequested,
            });

            if (wheelAction === "horizontal-scroll") {
                e.preventDefault();
                scroller.scrollLeft += horizontalScrollRequested ? e.deltaY : e.deltaX;
                syncScrollLeft(scroller);
                return;
            }

            if (wheelAction === "vertical-scroll") {
                e.preventDefault();
                scroller.scrollTop += e.deltaY;
                return;
            }

            if (wheelAction === "native") {
                return;
            }

            const bounds = scroller.getBoundingClientRect();

            if (wheelAction === "vertical-zoom") {
                e.preventDefault();
                const dir = e.deltaY < 0 ? 1 : -1;
                const factor = dir > 0 ? 1.1 : 0.9;
                const baseRowHeight =
                    pendingVerticalZoomRef.current?.nextRowHeight ?? rowHeightRef.current;
                const baseScrollTop =
                    pendingVerticalZoomRef.current?.nextScrollTop ?? scroller.scrollTop;
                const pointerY = clamp(e.clientY - bounds.top, 0, Math.max(1, bounds.height));
                const rowUnitAtPointer = (baseScrollTop + pointerY) / Math.max(1e-9, baseRowHeight);
                const nextRowHeight = Math.round(
                    clamp(baseRowHeight * factor, MIN_ROW_HEIGHT, MAX_ROW_HEIGHT),
                );
                if (Math.abs(nextRowHeight - baseRowHeight) < 1e-9) {
                    return;
                }
                pendingVerticalZoomRef.current = {
                    pointerY,
                    rowUnitAtPointer,
                    nextRowHeight,
                    nextScrollTop: Math.max(0, rowUnitAtPointer * nextRowHeight - pointerY),
                };
                setRowHeight(nextRowHeight);
                return;
            }

            if (wheelAction !== "horizontal-zoom") {
                return;
            }

            e.preventDefault();
            const dir = e.deltaY < 0 ? 1 : -1;
            const factor = dir > 0 ? 1.1 : 0.9;

            const basePxPerSec = zoomPendingRef.current?.nextPxPerSec ?? pxPerSecRef.current;
            const baseScrollLeft = zoomPendingRef.current?.nextScrollLeft ?? scroller.scrollLeft;

            const totalSec = Math.max(0, projectSec);
            let anchorSec: number;

            // Playhead-based zoom: use playhead as anchor instead of pointer
            if (playheadZoomEnabled && playheadSec != null) {
                anchorSec = clamp(playheadSec, 0, totalSec);
            } else {
                const anchorX = clamp(e.clientX - bounds.left, 0, Math.max(1, bounds.width));
                anchorSec = (anchorX + baseScrollLeft) / Math.max(1e-9, basePxPerSec);
            }

            const zoom = computeAnchoredHorizontalZoom({
                currentScale: basePxPerSec,
                factor,
                minScale: MIN_PX_PER_SEC,
                maxScale: MAX_PX_PER_SEC,
                scrollLeft: baseScrollLeft,
                viewportWidth: Math.max(1, bounds.width),
                anchorSec,
                contentSec: totalSec,
            });
            if (!zoom) return;

            zoomPendingRef.current = {
                nextPxPerSec: zoom.nextScale,
                nextScrollLeft: zoom.nextScrollLeft,
            };

            if (zoomRafRef.current == null) {
                zoomRafRef.current = requestAnimationFrame(() => {
                    zoomRafRef.current = null;
                    const pending = zoomPendingRef.current;
                    if (!pending) return;
                    zoomPendingRef.current = null;
                    pendingZoomRef.current = pending;
                    setPxPerSec(pending.nextPxPerSec);
                });
            }
        };

        scroller.addEventListener("wheel", handler, {
            passive: false,
        } as globalThis.AddEventListenerOptions);
        return () => {
            scroller.removeEventListener("wheel", handler);
        };
    }, [
        pxPerSec,
        scrollRef,
        setPxPerSec,
        setRowHeight,
        scrollHorizontalKb,
        scrollVerticalKb,
        horizontalZoomKb,
        verticalZoomKb,
        playheadSec,
        playheadZoomEnabled,
    ]);

    return (
        <div
            {...divProps}
            ref={scrollRef}
            onScroll={(e) => {
                syncScrollLeft(e.currentTarget as HTMLDivElement);
                onScroll?.(e);
            }}
        />
    );
};
