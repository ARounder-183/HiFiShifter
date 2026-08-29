import React, { useCallback, useEffect, useLayoutEffect, useRef } from "react";
import { flushSync } from "react-dom";

import { MAX_PX_PER_SEC, MAX_ROW_HEIGHT, MIN_PX_PER_SEC, MIN_ROW_HEIGHT } from "./constants";
import { clamp } from "./math";
import { isNoneBinding, isModifierActive } from "../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../features/keybindings/types";
import { getTimelineWheelAction } from "../wheelGesture";
import { shouldDispatchTimelineViewport } from "./runtime/timelineViewportDispatch";
import { resolveTimelineMinPxPerSec } from "./runtime/timelineZoomBounds";
import { applyNativeScrollLeft } from "./runtime/nativeScrollApply";
import { resolveHorizontalWheelZoom } from "./runtime/timelineScrollRange";

export const TimelineScrollArea: React.FC<
    Omit<React.HTMLAttributes<HTMLDivElement>, "ref"> & {
        scrollRef: React.MutableRefObject<HTMLDivElement | null>;
        projectSec: number;
        pxPerSec: number;
        setPxPerSec: React.Dispatch<React.SetStateAction<number>>;
        rowHeight: number;
        setRowHeight: React.Dispatch<React.SetStateAction<number>>;
        setScrollLeft: React.Dispatch<React.SetStateAction<number>>;
        /** 缩放的同一 flushSync 内提交 scrollLeft state（供窗口化立即使用）。 */
        commitScrollLeftState: React.Dispatch<React.SetStateAction<number>>;
        /** 竖直缩放提交 rowHeight 时同步提交 scrollTop state。 */
        commitScrollTopState: React.Dispatch<React.SetStateAction<number>>;
        rulerContentRef: React.MutableRefObject<HTMLDivElement | null>;
        scrollHorizontalKb?: Keybinding;
        scrollVerticalKb?: Keybinding;
        horizontalZoomKb?: Keybinding;
        verticalZoomKb?: Keybinding;
        getPlayheadSec?: () => number;
        playheadZoomEnabled?: boolean;
    }
> = ({
    scrollRef,
    projectSec,
    pxPerSec,
    setPxPerSec,
    rowHeight,
    setRowHeight,
    setScrollLeft,
    commitScrollLeftState,
    commitScrollTopState,
    rulerContentRef,
    onScroll,
    scrollHorizontalKb,
    scrollVerticalKb,
    horizontalZoomKb,
    verticalZoomKb,
    getPlayheadSec,
    playheadZoomEnabled,
    ...divProps
}) => {
    const lastViewportDispatchRef = useRef<{
        scrollLeft: number;
        pxPerSec: number;
        viewportWidth: number;
    } | null>(null);
    const pxPerSecRef = useRef(pxPerSec);
    // 滚轮缩放路径会在 flushSync 前手动刷新此 ref（见下方 wheel handler），
    // 其余路径由被动 effect 兜底，供 dispatch 去重快照使用。
    useEffect(() => {
        pxPerSecRef.current = pxPerSec;
    }, [pxPerSec]);
    const zoomRafRef = useRef<number | null>(null);
    const zoomPendingRef = useRef<{
        nextPxPerSec: number;
        nextScrollLeft: number;
    } | null>(null);

    // zoom 中心点以秒为基准：rAF 提交的待落地缩放（含目标滚动偏移），
    // 由 pxPerSec 提交后的 useLayoutEffect 消费。
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
        rowHeightRef.current = rowHeight;
    }, [rowHeight]);

    const syncScrollLeft = useCallback(
        function syncScrollLeft(scroller: HTMLDivElement) {
            const next = scroller.scrollLeft;
            const nextSnapshot = {
                scrollLeft: next,
                pxPerSec: pxPerSecRef.current,
                viewportWidth: scroller.clientWidth,
            };
            if (
                !shouldDispatchTimelineViewport({
                    previous: lastViewportDispatchRef.current,
                    next: nextSnapshot,
                })
            ) {
                return;
            }
            lastViewportDispatchRef.current = nextSnapshot;
            if (rulerContentRef.current) {
                rulerContentRef.current.style.transform = `translateX(${-next}px)`;
            }
            setScrollLeft(next);
        },
        [rulerContentRef, setScrollLeft],
    );

    useEffect(() => {
        const scroller = scrollRef.current;
        if (!scroller) return;
        syncScrollLeft(scroller);
    }, [scrollRef, syncScrollLeft]);

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

        // flushSync 已让 DOM（含 paddedContentWidth）按新缩放完成提交，这里
        // 写原生 scrollLeft 不会再被旧宽度钳制。事务在同一布局 effect 内闭合，
        // 绝不跨帧保持打开：写后回读浏览器实际接受的偏移（钳制/量化/锚定
        // 都可能修正请求值），并一律以回读值为准同步标尺、React state 与
        // 视口总线。此前“接受失败则保持事务打开、吞掉后续滚动事件”的设计
        // 一旦命中拒绝分支就无法恢复，画布层会冻结在旧偏移上，Clip 与其
        // 选中框从此错位。
        pendingZoomRef.current = null;
        applyNativeScrollLeft(scroller, pending.nextScrollLeft);
        syncScrollLeft(scroller);
    }, [projectSec, pxPerSec, scrollRef, syncScrollLeft]);

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

            const horizontalDelta = Math.abs(e.deltaX) > 0.5 ? e.deltaX : e.deltaY;

            if (wheelAction === "free-scroll") {
                e.preventDefault();
                scroller.scrollLeft += e.deltaX;
                scroller.scrollTop += e.deltaY;
                syncScrollLeft(scroller);
                return;
            }

            if (wheelAction === "horizontal-scroll") {
                e.preventDefault();
                scroller.scrollLeft += horizontalDelta;
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
                // 与 rowHeight 同批提交 scrollTop state：下一渲染立即使用新的
                // 竖直窗口，避免窗口化滞后把 Clip 整体裁掉。
                commitScrollTopState(pendingVerticalZoomRef.current.nextScrollTop);
                setRowHeight(nextRowHeight);
                return;
            }

            if (wheelAction !== "horizontal-zoom") {
                return;
            }

            e.preventDefault();
            const dir = e.deltaY < 0 ? 1 : -1;
            const factor = dir > 0 ? 1.1 : 0.9;

            // 连续滚动帧内的缩放链：以本帧已请求但未落地的偏移为基准；
            // 否则以原生 scrollLeft（浏览器实际接受的偏移）为基准。缩放
            // 事务在布局 effect 内同帧闭合，不存在“已请求但长期未落地”的
            // 状态，因此无需再回退读取待落地事务。
            const basePxPerSec = zoomPendingRef.current?.nextPxPerSec ?? pxPerSecRef.current;
            const baseScrollLeft = zoomPendingRef.current?.nextScrollLeft ?? scroller.scrollLeft;

            const totalSec = Math.max(0, projectSec);
            const minPxPerSec = resolveTimelineMinPxPerSec({
                baseMinPxPerSec: MIN_PX_PER_SEC,
                projectSec: totalSec,
                viewportWidthPx: scroller.clientWidth,
            });
            const zoom = resolveHorizontalWheelZoom({
                factor,
                basePxPerSec,
                baseScrollLeft,
                totalSec,
                viewportWidth: scroller.clientWidth,
                playheadZoomEnabled: Boolean(playheadZoomEnabled),
                playheadSec: getPlayheadSec?.() ?? null,
                anchorScreenX: e.clientX - bounds.left,
                minPxPerSec,
                maxPxPerSec: MAX_PX_PER_SEC,
            });
            if (!zoom) return;

            zoomPendingRef.current = {
                nextPxPerSec: zoom.nextPxPerSec,
                nextScrollLeft: zoom.nextScrollLeft,
            };

            if (zoomRafRef.current == null) {
                zoomRafRef.current = requestAnimationFrame(() => {
                    zoomRafRef.current = null;
                    const pending = zoomPendingRef.current;
                    if (!pending) return;
                    zoomPendingRef.current = null;
                    pendingZoomRef.current = pending;
                    pxPerSecRef.current = pending.nextPxPerSec;
                    // 原子缩放：flushSync 让 DOM（Clip/网格/contentWidth）在本帧内
                    // 按新缩放重排，layout effect 随即写原生 scrollLeft 并经
                    // syncScrollLeft 同步重绘标尺与画布——全部发生在绘制前。
                    // scrollLeft state 也必须在同一 flushSync 提交：窗口化/裁剪
                    // 会读取该 state，旧值会短暂把视口内的 Clip 判为屏外而消失。
                    flushSync(() => {
                        setPxPerSec(pending.nextPxPerSec);
                        commitScrollLeftState(pending.nextScrollLeft);
                    });
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
        scrollRef,
        setPxPerSec,
        setRowHeight,
        commitScrollLeftState,
        commitScrollTopState,
        rulerContentRef,
        scrollHorizontalKb,
        scrollVerticalKb,
        horizontalZoomKb,
        verticalZoomKb,
        getPlayheadSec,
        playheadZoomEnabled,
    ]);

    return (
        <div
            {...divProps}
            ref={scrollRef}
            style={{ overflowAnchor: "none", ...(divProps.style ?? {}) }}
            onScroll={(e) => {
                syncScrollLeft(e.currentTarget as HTMLDivElement);
                onScroll?.(e);
            }}
        />
    );
};
