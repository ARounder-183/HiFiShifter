/**
 * SnapOffset（吸附偏移）三角手柄拖拽。
 *
 * SnapOffset 是 Clip 自身属性：相对 Clip 起点的偏移（秒，默认 0，与倒放
 * 无关）。拖动 Clip 左下角的三角手柄时：
 * - 手柄的**绝对时间线位置**（clip.start + offset）作为被吸附对象，
 *   参与完整吸附引擎 —— 网格线、其他 Clip 边缘/SnapOffset、播放光标等
 *   全部候选照常生效；
 * - 命中吸附时发布竖线高亮（目标 + 手柄所在行内亮条）；
 * - 拖拽全程乐观更新 Redux（波形竖线与手柄实时跟随），松手时一次性
 *   持久化到后端。
 */
import { useRef } from "react";
import type { AppDispatch } from "../../../../app/store";
import type { SessionState } from "../../../../features/session/sessionSlice";
import {
    beginInteraction,
    checkpointHistory,
    endInteraction,
    setClipSnapOffset,
} from "../../../../features/session/sessionSlice";
import { setClipStateRemote } from "../../../../features/session/thunks/timelineThunks";
import { isModifierActive } from "../../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../../features/keybindings/types";
import { computeEffectiveSnap, beginSnapGesture, endSnapGesture } from "../../../../utils/timelineSnapping";
import {
    SNAP_HIGHLIGHT_GROUP,
    clearSnapHighlights,
} from "../../../../utils/snapHighlight";
import type { SnapObjectKind, SnapResult } from "../../../../utils/timelineSnapping";
import type { SnapTimelineOpts } from "./useTimelineState";

function clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
}

export function useSnapOffsetDrag(deps: {
    scrollRef: React.RefObject<HTMLDivElement | null>;
    sessionRef: React.RefObject<SessionState>;
    dispatch: AppDispatch;
    /** 完整吸附结果入口（负责发布吸附竖线高亮）。 */
    snapTimelineDetailed: (
        sec: number,
        object: SnapObjectKind,
        opts?: SnapTimelineOpts,
    ) => SnapResult;
    beatFromClientX: (clientX: number, bounds: DOMRect, xScroll: number) => number;
    /** modifier.clipNoSnap 绑定 */
    noSnapKb: Keybinding;
    /** 吸附全局开关 */
    snapEnabled: boolean;
}) {
    const {
        scrollRef,
        sessionRef,
        dispatch,
        snapTimelineDetailed,
        beatFromClientX,
        noSnapKb,
        snapEnabled,
    } = deps;

    const dragRef = useRef<{ pointerId: number } | null>(null);

    function startSnapOffsetDrag(e: React.PointerEvent, clipId: string) {
        if (e.button !== 0) return;
        const clip = sessionRef.current.clips.find((c) => c.id === clipId);
        if (!clip) return;
        const scroller = scrollRef.current;
        if (!scroller) return;

        e.preventDefault();
        e.stopPropagation();
        dispatch(checkpointHistory());
        dispatch(beginInteraction());

        const bounds = scroller.getBoundingClientRect();
        const startPointerSec = beatFromClientX(e.clientX, bounds, scroller.scrollLeft);
        const baseOffset = clamp(Number(clip.snapOffsetSec) || 0, 0, Math.max(0, clip.lengthSec));
        const clipStart = Number(clip.startSec) || 0;
        const clipLen = Math.max(0, Number(clip.lengthSec) || 0);
        const anchorTrackId = clip.trackId;

        dragRef.current = { pointerId: e.pointerId };
        try {
            (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        } catch {
            // ignore
        }
        beginSnapGesture();

        function applyOffset(next: number): void {
            dispatch(
                setClipSnapOffset({
                    clipId,
                    snapOffsetSec: clamp(next, 0, clipLen),
                }),
            );
        }

        function onMove(ev: PointerEvent) {
            const drag = dragRef.current;
            const el = scrollRef.current;
            if (!drag || drag.pointerId !== ev.pointerId || !el) return;
            const b = el.getBoundingClientRect();
            const pointerSec = beatFromClientX(ev.clientX, b, el.scrollLeft);
            const rawAbs = clipStart + baseOffset + (pointerSec - startPointerSec);

            // "拖动时切换吸附"：修饰键把吸附总开关临时取反。
            const effectiveSnap = computeEffectiveSnap(
                snapEnabled,
                isModifierActive(noSnapKb, ev),
            );
            let absPos = rawAbs;
            if (effectiveSnap) {
                // 单点吸附：手柄绝对位置对齐目标；被吸附对象侧高亮 =
                // 手柄自身所在行内的亮条。
                absPos = snapTimelineDetailed(rawAbs, "clip", {
                    originSec: clipStart + baseOffset,
                    anchorTrackId,
                    excludeClipIds: new Set([clipId]),
                    highlight: {
                        sources: [{ trackId: anchorTrackId, clipId }],
                    },
                }).sec;
            } else {
                clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            }
            applyOffset(absPos - clipStart);
        }

        function onEnd(ev: PointerEvent) {
            const drag = dragRef.current;
            if (!drag || drag.pointerId !== ev.pointerId) return;
            dragRef.current = null;
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onEnd);
            window.removeEventListener("pointercancel", onEnd);
            endSnapGesture();
            clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            // 松手持久化最终值（读取当前乐观状态，拖拽期间已收敛）。
            const finalClip = sessionRef.current.clips.find((c) => c.id === clipId);
            void dispatch(
                setClipStateRemote({
                    clipId,
                    snapOffsetSec: clamp(
                        Number(finalClip?.snapOffsetSec ?? baseOffset) || 0,
                        0,
                        clipLen,
                    ),
                }),
            ).finally(() => {
                dispatch(endInteraction());
            });
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onEnd);
        window.addEventListener("pointercancel", onEnd);
    }

    return startSnapOffsetDrag;
}
