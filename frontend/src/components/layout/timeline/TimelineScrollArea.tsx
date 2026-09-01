import React, { useCallback, useEffect, useLayoutEffect, useRef } from "react";

import { MAX_PX_PER_SEC, MAX_ROW_HEIGHT, MIN_PX_PER_SEC, MIN_ROW_HEIGHT } from "./constants";
import { clamp } from "./math";
import { isNoneBinding, isModifierActive } from "../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../features/keybindings/types";
import { useDebouncedPersist } from "../../../hooks/useDebouncedPersist";
import { getTimelineWheelAction } from "../wheelGesture";
import { shouldDispatchTimelineViewport } from "./runtime/timelineViewportDispatch";
import { resolveTimelineMinPxPerSec } from "./runtime/timelineZoomBounds";
import { applyNativeScrollLeft } from "./runtime/nativeScrollApply";
import {
    resolveHorizontalWheelZoom,
    resolveTimelineScrollRange,
} from "./runtime/timelineScrollRange";

export const TimelineScrollArea: React.FC<
    Omit<React.HTMLAttributes<HTMLDivElement>, "ref"> & {
        scrollRef: React.MutableRefObject<HTMLDivElement | null>;
        /**
         * 水平内容层（决定原生滚动上限的那个 div）。
         *
         * 缩放时它的宽度必须**先**按新的 pxPerSec 落地，否则写
         * `scroller.scrollLeft` 会被浏览器按旧的最大滚动位置钳回去。此前这
         * 一步靠 `flushSync` 强制整棵树同步提交来实现，现在改为直接命令式
         * 写入，React 就不必再被阻塞地同步渲染了。
         */
        contentSizerRef: React.RefObject<HTMLDivElement | null>;
        projectSec: number;
        pxPerSec: number;
        setPxPerSec: React.Dispatch<React.SetStateAction<number>>;
        rowHeight: number;
        setRowHeight: React.Dispatch<React.SetStateAction<number>>;
        setScrollLeft: React.Dispatch<React.SetStateAction<number>>;
        /** 缩放时与 pxPerSec 一起提交 scrollLeft state（供窗口化使用）。 */
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
    contentSizerRef,
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
    // 滚轮缩放路径会在提交 state 前手动刷新此 ref（见下方 wheel handler），
    // 其余路径由被动 effect 兜底，供 dispatch 去重快照使用。
    useEffect(() => {
        pxPerSecRef.current = pxPerSec;
    }, [pxPerSec]);
    const zoomRafRef = useRef<number | null>(null);
    const zoomPendingRef = useRef<{
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
        const scroller = scrollRef.current;
        const pending = pendingVerticalZoomRef.current;
        if (!scroller || !pending) return;
        if (Math.abs(pending.nextRowHeight - rowHeight) > 1e-9) return;

        pendingVerticalZoomRef.current = null;

        const maxScrollTop = Math.max(0, scroller.scrollHeight - scroller.clientHeight);
        scroller.scrollTop = Math.min(Math.max(0, pending.nextScrollTop), maxScrollTop);
    }, [rowHeight, scrollRef]);

    // ★ 缩放设置必须防抖落盘：`localStorage.setItem` 是同步磁盘 I/O，而
    // pxPerSec / rowHeight 在滚轮缩放手势中每帧都变，直接写会让每个滚轮
    // 事件都堵在一次落盘上，与同帧的 React 渲染和画布重绘挤在一起卡顿。
    // 防抖后一次手势只写一次；卸载时 hook 会补写未落盘的值。
    useDebouncedPersist("hifishifter.pxPerSec", pxPerSec);
    useDebouncedPersist("hifishifter.rowHeight", rowHeight);

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
                    pxPerSecRef.current = pending.nextPxPerSec;

                    // ── 原子缩放（无 flushSync 版）─────────────────────
                    // 旧实现用 flushSync 把整棵树同步提交一遍，但它的**唯一
                    // 硬约束**其实只有一条：内容层宽度必须先按新 pxPerSec
                    // 重排，否则写 scroller.scrollLeft 会被浏览器按旧的最大
                    // 滚动位置钳回去，表现为"缩放变了、滚动没变"的水平漂移。
                    //
                    // 既然卡点只在内容层宽度，就直接命令式写它（公式与
                    // TimelinePanel 里 contentWidth / timelineScrollRange 完全
                    // 一致，React 随后异步提交时会写入相同的字符串，无抖动），
                    // 于是不必再阻塞地同步渲染整棵树。
                    //
                    // 画布层由下面的 syncScrollLeft 在同一帧内同步重绘，与
                    // 原生 DOM 内容层同帧；React state 随后异步跟上，只服务
                    // 于窗口化 / 裁剪这类非视觉用途。
                    let appliedScrollLeft: number | null = null;
                    const scroller = scrollRef.current;
                    if (scroller) {
                        const sizer = contentSizerRef.current;
                        if (sizer) {
                            const range = resolveTimelineScrollRange({
                                contentWidth: Math.max(
                                    1,
                                    Math.ceil(projectSec * pending.nextPxPerSec),
                                ),
                                viewportWidth: scroller.clientWidth,
                            });
                            sizer.style.width = `${range.paddedContentWidth}px`;
                        }
                        // 写后回读浏览器实际接受的偏移再广播：钳制/量化/锚定
                        // 都可能修正请求值，画布层与原生 DOM 层不允许以
                        // "请求值"为准失步。
                        appliedScrollLeft = applyNativeScrollLeft(scroller, pending.nextScrollLeft);
                        syncScrollLeft(scroller);
                    }
                    setPxPerSec(pending.nextPxPerSec);
                    // 必须用**回读值**而非请求值：内容宽度此刻才刚落地，浏览器
                    // 很可能把请求值钳到新的最大滚动位置上。写入请求值会让
                    // state 与真实 DOM 差一截，窗口化随之判错可见区间。
                    commitScrollLeftState(appliedScrollLeft ?? pending.nextScrollLeft);
                });
            }
        };

        scroller.addEventListener("wheel", handler, {
            passive: false,
        } as globalThis.AddEventListenerOptions);
        return () => {
            scroller.removeEventListener("wheel", handler);
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps -- projectSec 随工程长度变化，加入依赖会让 wheel 监听在工程变化时重建（既有模式）
    }, [
        scrollRef,
        syncScrollLeft,
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
