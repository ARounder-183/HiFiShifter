import { useRef } from "react";
import type { AppDispatch } from "../../../../app/store";
import type { SessionState } from "../../../../features/session/sessionSlice";
import {
    checkpointHistory,
    setClipStateRemote,
    setClipSourceRange,
    beginInteraction,
    endInteraction,
} from "../../../../features/session/sessionSlice";

export type SlipDragState = {
    pointerId: number;
    anchorClipId: string;
    clipIds: string[];
    initialPointerBeat: number;
    initialById: Record<
        string,
        {
            sourceStartSec: number;
            sourceEndSec: number;
            playbackRate: number;
            sourceDurationSec: number | null;
            maxSlipSec: number;
        }
    >;
};

export function useSlipDrag(deps: {
    scrollRef: React.RefObject<HTMLDivElement | null>;
    sessionRef: React.RefObject<SessionState>;
    dispatch: AppDispatch;
    multiSelectedClipIds: string[];
    multiSelectedSet: Set<string>;
    beatFromClientX: (clientX: number, bounds: DOMRect, xScroll: number) => number;
}) {
    const {
        scrollRef,
        sessionRef,
        dispatch,
        multiSelectedClipIds,
        multiSelectedSet,
        beatFromClientX,
    } = deps;

    const slipDragRef = useRef<SlipDragState | null>(null);

    function startSlipDrag(e: React.PointerEvent<HTMLDivElement>, clipId: string) {
        if (e.button !== 0) return;
        const anchor = sessionRef.current.clips.find((c) => c.id === clipId);
        if (!anchor) return;
        const scroller = scrollRef.current;
        if (!scroller) return;

        dispatch(checkpointHistory());
        dispatch(beginInteraction());

        const bounds = scroller.getBoundingClientRect();
        const beatAtPointer = beatFromClientX(e.clientX, bounds, scroller.scrollLeft);

        const clipIds =
            multiSelectedClipIds.length > 0 && multiSelectedSet.has(clipId)
                ? [...multiSelectedClipIds]
                : [clipId];

        const initialById: SlipDragState["initialById"] = {};
        for (const id of clipIds) {
            const c = sessionRef.current.clips.find((x) => x.id === id);
            if (!c) continue;
            // 纯秒域：source 文件总时长就是最大 slip 范围
            const sourceDurationSec = Number(c.durationSec ?? 0) || null;
            const sourceStartSec = Number(c.sourceStartSec ?? 0) || 0;
            const sourceEndSec = Math.max(0, Number(c.sourceEndSec ?? 0) || 0);
            const maxSlipSec =
                sourceDurationSec != null && Number.isFinite(sourceDurationSec)
                    ? Math.max(0, sourceDurationSec)
                    : Math.max(0, Number(c.lengthSec ?? 0) || 0);
            initialById[id] = {
                sourceStartSec,
                sourceEndSec,
                playbackRate: Number(c.playbackRate ?? 1) || 1,
                sourceDurationSec,
                maxSlipSec,
            };
        }

        slipDragRef.current = {
            pointerId: e.pointerId,
            anchorClipId: clipId,
            clipIds,
            initialPointerBeat: beatAtPointer,
            initialById,
        };

        (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);

        function onMove(ev: PointerEvent) {
            const drag = slipDragRef.current;
            const el = scrollRef.current;
            if (!drag || drag.pointerId !== e.pointerId || !el) return;
            const b = el.getBoundingClientRect();
            const beatNow = beatFromClientX(ev.clientX, b, el.scrollLeft);
            let deltaBeat = drag.initialPointerBeat - beatNow;

            for (const id of drag.clipIds) {
                const initial = drag.initialById[id];
                if (!initial) continue;
                const rate =
                    initial.playbackRate > 0 && Number.isFinite(initial.playbackRate)
                        ? initial.playbackRate
                        : 1;
                const deltaSrcSec = deltaBeat * rate;
                let nextSourceStart = initial.sourceStartSec + deltaSrcSec;
                let nextSourceEnd = initial.sourceEndSec + deltaSrcSec;

                // clamp: sourceStart 不能小于 0，sourceEnd 不能超过源文件时长
                if (Number.isFinite(initial.maxSlipSec) && initial.maxSlipSec > 1e-6) {
                    if (nextSourceStart < 0) {
                        nextSourceEnd -= nextSourceStart;
                        nextSourceStart = 0;
                    }
                    if (nextSourceEnd > initial.maxSlipSec) {
                        nextSourceStart -= nextSourceEnd - initial.maxSlipSec;
                        nextSourceEnd = initial.maxSlipSec;
                    }
                }
                dispatch(
                    setClipSourceRange({
                        clipId: id,
                        sourceStartSec: nextSourceStart,
                        sourceEndSec: nextSourceEnd,
                    }),
                );
            }
        }

        function end() {
            const drag = slipDragRef.current;
            if (!drag || drag.pointerId !== e.pointerId) return;
            slipDragRef.current = null;

            // 交互锁在最终持久化请求完成后才释放，
            // 避免 endInteraction() 到 fulfilled 之间的窗口内，
            // 其他 in-flight thunk 的旧快照覆盖前端乐观更新导致闪烁。

            const session = sessionRef.current;
            const persistPromises: Promise<unknown>[] = [];
            for (const id of drag.clipIds) {
                const now = session.clips.find((c) => c.id === id);
                if (!now) continue;
                persistPromises.push(
                    dispatch(
                        setClipStateRemote({
                            clipId: id,
                            sourceStartSec: Number(now.sourceStartSec ?? 0) || 0,
                            sourceEndSec: Number(now.sourceEndSec ?? 0) || 0,
                        }),
                    ).unwrap(),
                );
            }

            void Promise.allSettled(persistPromises).finally(() => {
                dispatch(endInteraction());
            });

            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", end);
            window.removeEventListener("pointercancel", end);
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", end);
        window.addEventListener("pointercancel", end);
    }

    return { slipDragRef, startSlipDrag };
}
