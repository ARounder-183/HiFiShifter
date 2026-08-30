/**
 * 时间轴右键框选逻辑。
 *
 * 规则：
 * - 右键按下后先进入待判定状态；
 * - 仅当拖拽超过阈值时，才启动框选并在抬起时提交选区；
 * - 未达到拖拽阈值时，不改动现有多选，让右键菜单正常弹出。
 *
 * 注意：右键框选**刻意不做吸附、也不显示吸附高亮** —— 它是 Clip 框选
 * 手势，不与时间轴网格/候选直接交互（吸附仅服务于移动/编辑类拖拽）。
 */
import { useRef, useState } from "react";
import { registerDragAbort } from "./gestureFocusGuard";
import type * as React from "react";

import type { SessionState } from "../../../features/session/sessionSlice";
import { isPrimaryModifierDown } from "../../../utils/platform";

export function shouldStartTimelineSelectionRect(button: number): boolean {
    // Only start selection for right-click (button === 2).
    // Allow right-click drag anywhere on the timeline (including
    // clip elements) to initiate the selection rect.
    return button === 2;
}

export const TIMELINE_SELECTION_DRAG_THRESHOLD_PX = 5;

export function isTimelineSelectionDrag(
    startX: number,
    startY: number,
    curX: number,
    curY: number,
    thresholdPx = TIMELINE_SELECTION_DRAG_THRESHOLD_PX,
): boolean {
    const dx = curX - startX;
    const dy = curY - startY;
    return dx * dx + dy * dy >= thresholdPx * thresholdPx;
}

export function computeTimelineRectSelection(params: {
    selectionBeforeDrag: string[];
    selectedInRect: string[];
    primaryModifierPressedAtStart: boolean;
}): string[] {
    const { selectionBeforeDrag, selectedInRect, primaryModifierPressedAtStart } = params;
    if (!primaryModifierPressedAtStart) {
        return selectedInRect;
    }
    const beforeSet = new Set(selectionBeforeDrag);
    const inRectSet = new Set(selectedInRect);
    const kept = selectionBeforeDrag.filter((id) => !inRectSet.has(id));
    const appended = selectedInRect.filter((id) => !beforeSet.has(id));
    return [...kept, ...appended];
}

export function useTimelineSelectionRect(params: {
    scrollRef: React.RefObject<HTMLDivElement | null>;
    sessionRef: React.RefObject<SessionState>;
    /** 内容坐标系使用的像素密度：秒 → px（不是 beat → px）。 */
    pxPerSec: number;
    rowHeight: number;

    clearContextMenu: () => void;
    setMultiSelectedClipIds: (ids: string[] | ((prev: string[]) => string[])) => void;
    onSingleSelect: (clipId: string) => void;
}) {
    const {
        scrollRef,
        sessionRef,
        pxPerSec,
        rowHeight,
        clearContextMenu,
        setMultiSelectedClipIds,
        onSingleSelect,
    } = params;

    const selectionDragRef = useRef<{
        pointerId: number;
        /** 指针按下时的客户端坐标：仅用于位移阈值判定（纯指针位移，
         * 不受拖拽期间滚动/缩放影响 —— 否则按住右键后滚动容器也会
         * 误触发框选、吞掉右键菜单）。 */
        startClientX: number;
        startClientY: number;
        /** 世界（内容像素）坐标锚点：矩形与命中测试使用。
         * 已知限制：拖拽期间变焦会使锚点的像素刻度失效（极低频）。 */
        startX: number;
        startY: number;
        curX: number;
        curY: number;
        hasSelectionDrag: boolean;
        primaryModifierPressedAtStart: boolean;
        selectionBeforeDrag: string[];
        deferredContextMenu: {
            clientX: number;
            clientY: number;
            target: EventTarget | null;
        } | null;
    } | null>(null);

    const [selectionRect, setSelectionRect] = useState<{
        x1: number;
        y1: number;
        x2: number;
        y2: number;
    } | null>(null);

    function onPointerDown(e: React.PointerEvent<HTMLDivElement>) {
        if (!shouldStartTimelineSelectionRect(e.button)) return;
        e.preventDefault();

        const el = e.currentTarget as HTMLDivElement;
        const bounds = el.getBoundingClientRect();
        const x = e.clientX - bounds.left + el.scrollLeft;
        const y = e.clientY - bounds.top + el.scrollTop;
        const session = sessionRef.current;
        const currentSelectionIds =
            session.multiSelectedClipIds.length > 0
                ? [...session.multiSelectedClipIds]
                : session.selectedClipId
                  ? [session.selectedClipId]
                  : [];
        selectionDragRef.current = {
            pointerId: e.pointerId,
            startClientX: e.clientX,
            startClientY: e.clientY,
            startX: x,
            startY: y,
            curX: x,
            curY: y,
            hasSelectionDrag: false,
            primaryModifierPressedAtStart: isPrimaryModifierDown(e),
            selectionBeforeDrag: currentSelectionIds,
            deferredContextMenu: null,
        };
        // 失焦取消：切屏期间 pointerup/pointercancel 不送达本窗口，blur 时
        // 走与 end() 完全相同的收尾（提交框选结果 / 重放被抑制的右键菜单）。
        const unregisterAbort = registerDragAbort(end);

        // GTK/WebKit fires `contextmenu` on right-button *press* (unlike
        // WebView2, which fires it on release). Keep the native event
        // suppressed while a right-button press could still become a
        // selection drag, then re-dispatch it on release when no drag
        // happened so plain right-click menus behave the same everywhere.
        const suppressContextMenu = (ev: Event) => {
            const drag = selectionDragRef.current;
            if (!drag || drag.pointerId !== e.pointerId) return;
            ev.preventDefault();
            ev.stopPropagation();
            drag.deferredContextMenu = {
                clientX: (ev as MouseEvent).clientX,
                clientY: (ev as MouseEvent).clientY,
                target: ev.target,
            };
        };
        window.addEventListener("contextmenu", suppressContextMenu, true);

        function onMove(ev: PointerEvent) {
            const drag = selectionDragRef.current;
            const current = scrollRef.current;
            if (!drag || drag.pointerId !== e.pointerId || !current) return;
            const b = current.getBoundingClientRect();
            const cx = ev.clientX - b.left + current.scrollLeft;
            const cy = ev.clientY - b.top + current.scrollTop;
            drag.curX = cx;
            drag.curY = cy;

            if (
                !drag.hasSelectionDrag &&
                // 阈值按指针的客户端位移判定：与内容坐标系解耦，滚动/变焦
                // 本身不会把"普通右键"误升级为框选手势。
                isTimelineSelectionDrag(
                    drag.startClientX,
                    drag.startClientY,
                    ev.clientX,
                    ev.clientY,
                )
            ) {
                drag.hasSelectionDrag = true;
                clearContextMenu();
            }

            if (!drag.hasSelectionDrag) return;

            // 右键框选不做吸附：矩形边界即原始指针位置。
            setSelectionRect({
                x1: Math.min(drag.startX, cx),
                y1: Math.min(drag.startY, cy),
                x2: Math.max(drag.startX, cx),
                y2: Math.max(drag.startY, cy),
            });
        }

        function end() {
            const drag = selectionDragRef.current;
            if (!drag || drag.pointerId !== e.pointerId) return;
            selectionDragRef.current = null;
            unregisterAbort(); // 收尾第一步注销失焦守卫（幂等防双触发）

            window.removeEventListener("contextmenu", suppressContextMenu, true);
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", end);
            window.removeEventListener("pointercancel", end);

            const hasSelectionDrag = drag.hasSelectionDrag;

            // 右键框选不做吸附：提交矩形即原始指针位置。
            const rect = {
                x1: Math.min(drag.startX, drag.curX),
                y1: Math.min(drag.startY, drag.curY),
                x2: Math.max(drag.startX, drag.curX),
                y2: Math.max(drag.startY, drag.curY),
            };
            setSelectionRect(null);

            if (!hasSelectionDrag) {
                // No drag: re-emit the suppressed contextmenu so plain
                // right-click menus still open (at release time, matching
                // Windows).
                const deferred = drag.deferredContextMenu;
                if (deferred) {
                    const clientX = deferred.clientX;
                    const clientY = deferred.clientY;
                    let target: EventTarget | null = deferred.target;
                    if (!(target instanceof Element) || !document.contains(target)) {
                        target = document.elementFromPoint(clientX, clientY);
                    }
                    if (target) {
                        target.dispatchEvent(
                            new MouseEvent("contextmenu", {
                                bubbles: true,
                                cancelable: true,
                                clientX,
                                clientY,
                                button: 2,
                                buttons: 0,
                                view: window,
                            }),
                        );
                    }
                }
                return;
            }

            const session = sessionRef.current;
            // Clip 像素位置必须与矩形同一坐标系：两者都使用内容像素密度
            // pxPerSec。不要从 pxPerBeat 反推；调用方曾误传 pxPerSec，
            // 会在非 60 BPM 时造成水平命中偏移。
            const contentPxPerSec = Math.max(1e-9, pxPerSec);
            const selectedInRect: string[] = [];
            for (const clip of session.clips) {
                const trackIdx = session.tracks.findIndex((t) => t.id === clip.trackId);
                if (trackIdx < 0) continue;
                const cx1 = clip.startSec * contentPxPerSec;
                const cx2 = (clip.startSec + clip.lengthSec) * contentPxPerSec;
                const cy1 = trackIdx * rowHeight;
                const cy2 = cy1 + rowHeight;
                const hit = cx2 >= rect.x1 && cx1 <= rect.x2 && cy2 >= rect.y1 && cy1 <= rect.y2;
                if (hit) selectedInRect.push(clip.id);
            }

            const selected = computeTimelineRectSelection({
                selectionBeforeDrag: drag.selectionBeforeDrag,
                selectedInRect,
                primaryModifierPressedAtStart: drag.primaryModifierPressedAtStart,
            });

            setMultiSelectedClipIds(selected);
            if (selected.length === 1) {
                onSingleSelect(selected[0]);
            }

            // 真正发生右键拖拽框选时，抑制本次 contextmenu。
            const suppressContextMenuAfterDrag = (ev: Event) => {
                ev.preventDefault();
                ev.stopPropagation();
            };
            window.addEventListener("contextmenu", suppressContextMenuAfterDrag, {
                capture: true,
                once: true,
            });
            // 安全回退：200ms 后自动移除，防止意外吞掉后续正常右键
            setTimeout(() => {
                window.removeEventListener("contextmenu", suppressContextMenuAfterDrag, {
                    capture: true,
                });
            }, 200);
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", end);
        window.addEventListener("pointercancel", end);
    }

    return { selectionRect, onPointerDown };
}
