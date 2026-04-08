/**
 * 时间轴右键框选逻辑。
 *
 * 规则：
 * - 右键按下后先进入待判定状态；
 * - 仅当拖拽超过阈值时，才启动框选并在抬起时提交选区；
 * - 未达到拖拽阈值时，不改动现有多选，让右键菜单正常弹出。
 */
import { useRef, useState } from "react";
import type * as React from "react";

import type { SessionState } from "../../../features/session/sessionSlice";

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

export function useTimelineSelectionRect(params: {
    scrollRef: React.RefObject<HTMLDivElement | null>;
    sessionRef: React.RefObject<SessionState>;
    pxPerBeat: number;
    rowHeight: number;

    clearContextMenu: () => void;
    setMultiSelectedClipIds: (ids: string[] | ((prev: string[]) => string[])) => void;
    onSingleSelect: (clipId: string) => void;
}) {
    const {
        scrollRef,
        sessionRef,
        pxPerBeat,
        rowHeight,
        clearContextMenu,
        setMultiSelectedClipIds,
        onSingleSelect,
    } = params;

    const selectionDragRef = useRef<{
        pointerId: number;
        startX: number;
        startY: number;
        curX: number;
        curY: number;
        hasSelectionDrag: boolean;
    } | null>(null);

    const [selectionRect, setSelectionRect] = useState<{
        x1: number;
        y1: number;
        x2: number;
        y2: number;
    } | null>(null);

    function onPointerDown(e: React.PointerEvent<HTMLDivElement>) {
        if (!shouldStartTimelineSelectionRect(e.button)) return;
        const el = e.currentTarget as HTMLDivElement;
        const bounds = el.getBoundingClientRect();
        const x = e.clientX - bounds.left + el.scrollLeft;
        const y = e.clientY - bounds.top + el.scrollTop;
        selectionDragRef.current = {
            pointerId: e.pointerId,
            startX: x,
            startY: y,
            curX: x,
            curY: y,
            hasSelectionDrag: false,
        };

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
                isTimelineSelectionDrag(drag.startX, drag.startY, drag.curX, drag.curY)
            ) {
                drag.hasSelectionDrag = true;
                clearContextMenu();
            }

            if (!drag.hasSelectionDrag) return;

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

            const hasSelectionDrag = drag.hasSelectionDrag;

            const rect = {
                x1: Math.min(drag.startX, drag.curX),
                y1: Math.min(drag.startY, drag.curY),
                x2: Math.max(drag.startX, drag.curX),
                y2: Math.max(drag.startY, drag.curY),
            };
            setSelectionRect(null);

            if (!hasSelectionDrag) {
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", end);
                window.removeEventListener("pointercancel", end);
                return;
            }

            const session = sessionRef.current;
            const selected: string[] = [];
            for (const clip of session.clips) {
                const trackIdx = session.tracks.findIndex((t) => t.id === clip.trackId);
                if (trackIdx < 0) continue;
                const cx1 = clip.startSec * pxPerBeat;
                const cx2 = (clip.startSec + clip.lengthSec) * pxPerBeat;
                const cy1 = trackIdx * rowHeight;
                const cy2 = cy1 + rowHeight;
                const hit = cx2 >= rect.x1 && cx1 <= rect.x2 && cy2 >= rect.y1 && cy1 <= rect.y2;
                if (hit) selected.push(clip.id);
            }
            setMultiSelectedClipIds(selected);
            if (selected.length === 1) {
                onSingleSelect(selected[0]);
            }

            // 真正发生右键拖拽框选时，抑制本次 contextmenu。
            const suppressContextMenu = (ev: Event) => {
                ev.preventDefault();
                ev.stopPropagation();
            };
            window.addEventListener("contextmenu", suppressContextMenu, {
                capture: true,
                once: true,
            });
            // 安全回退：200ms 后自动移除，防止意外吞掉后续正常右键
            setTimeout(() => {
                window.removeEventListener("contextmenu", suppressContextMenu, { capture: true });
            }, 200);

            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", end);
            window.removeEventListener("pointercancel", end);
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", end);
        window.addEventListener("pointercancel", end);
    }

    return { selectionRect, onPointerDown };
}
