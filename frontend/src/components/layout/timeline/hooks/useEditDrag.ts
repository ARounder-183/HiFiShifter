import { useRef } from "react";
import type { AppDispatch } from "../../../../app/store";
import type { SessionState } from "../../../../features/session/sessionSlice";
import {
    checkpointHistory,
    moveClipStart,
    setClipFades,
    setClipGain,
    setClipLength,
    setClipPlaybackRate,
    setClipStateRemote,
    setClipSourceRange,
    beginInteraction,
    endInteraction,
} from "../../../../features/session/sessionSlice";
import { applyAutoCrossfade } from "./autoCrossfade";
import { clamp, gainToDb, dbToGain } from "../math";
import { isModifierActive } from "../../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../../features/keybindings/types";
import { paramsApi } from "../../../../services/api";

export function resolveStretchParamTypes(
    pitchEditUserModified: boolean | null | undefined,
): Array<"pitch" | "tension"> {
    // 未手动编辑的 pitch 曲线由后端根据 clip 几何自动重建，
    // 前端若再次映射会造成“二次拉伸”。
    if (pitchEditUserModified === false) {
        return ["tension"];
    }
    return ["pitch", "tension"];
}

/**
 * 拉伸后对参数线进行时域映射（拉伸或压缩）。
 * 将旧范围 [oldStartSec, oldStartSec+oldLengthSec] 内的参数值，
 * 线性重映射到新范围 [newStartSec, newStartSec+newLengthSec]，
 * 并将不再被音频块覆盖的旧帧恢复为原始值。
 */
async function stretchLinkedParams(
    trackId: string,
    oldStartSec: number,
    oldLengthSec: number,
    newStartSec: number,
    newLengthSec: number,
): Promise<void> {
    if (
        Math.abs(oldLengthSec - newLengthSec) < 1e-6 &&
        Math.abs(oldStartSec - newStartSec) < 1e-6
    ) {
        return;
    }

    // 获取帧周期（通过最小量探针请求）。
    // 同时读取 pitch_edit_user_modified 以决定是否应手动映射 pitch。
    const probe = await paramsApi.getParamFrames(trackId, "pitch", 0, 1, 1);
    if (!probe?.ok) return;
    const fp = Math.max(1, Number(probe.frame_period_ms) || 5);
    const stretchParams = resolveStretchParamTypes(
        probe.pitch_edit_user_modified,
    );

    const oldStartFrame = Math.round((oldStartSec * 1000) / fp);
    const oldEndFrame = Math.round(((oldStartSec + oldLengthSec) * 1000) / fp);
    const oldFrameCount = Math.max(1, oldEndFrame - oldStartFrame);

    const newStartFrame = Math.round((newStartSec * 1000) / fp);
    const newEndFrame = Math.round(((newStartSec + newLengthSec) * 1000) / fp);
    const newFrameCount = Math.max(1, newEndFrame - newStartFrame);

    for (const paramType of stretchParams) {
        const res = await paramsApi.getParamFrames(
            trackId,
            paramType,
            oldStartFrame,
            oldFrameCount,
            1,
        );
        if (!res?.ok) continue;
        const oldValues = (res.edit ?? []).map((v) => Number(v) || 0);
        if (oldValues.length === 0) continue;

        // 线性插值时域映射：用旧帧值填充新帧
        const newValues = new Array<number>(newFrameCount);
        for (let i = 0; i < newFrameCount; i++) {
            const t = newFrameCount > 1 ? i / (newFrameCount - 1) : 0;
            const oldIdxF = t * (oldValues.length - 1);
            const lo = Math.floor(oldIdxF);
            const hi = Math.min(lo + 1, oldValues.length - 1);
            const frac = oldIdxF - lo;
            const loVal = oldValues[lo] ?? 0;
            const hiVal = oldValues[hi] ?? 0;
            if (paramType === "pitch") {
                // pitch=0 表示无效（无声）帧，保留 0
                if (loVal === 0 && hiVal === 0) {
                    newValues[i] = 0;
                } else if (loVal === 0) {
                    newValues[i] = 0;
                } else if (hiVal === 0) {
                    newValues[i] = frac < 0.5 ? loVal : 0;
                } else {
                    newValues[i] = loVal + (hiVal - loVal) * frac;
                }
            } else {
                newValues[i] = loVal + (hiVal - loVal) * frac;
            }
        }

        // 将重映射后的值写入新范围
        await paramsApi.setParamFrames(
            trackId,
            paramType,
            newStartFrame,
            newValues,
            false,
        );

        // 恢复旧范围中不再被新音频块覆盖的帧（还原到原始值）
        const newRangeMax = newStartFrame + newFrameCount - 1;
        const oldRangeMax = oldStartFrame + oldFrameCount - 1;

        if (oldStartFrame < newStartFrame) {
            const clearLen = newStartFrame - oldStartFrame;
            void paramsApi.restoreParamFrames(
                trackId,
                paramType,
                oldStartFrame,
                clearLen,
                false,
            );
        }
        if (oldRangeMax > newRangeMax) {
            const clearFrom = newRangeMax + 1;
            const clearLen = oldRangeMax - newRangeMax;
            void paramsApi.restoreParamFrames(
                trackId,
                paramType,
                clearFrom,
                clearLen,
                false,
            );
        }
    }
}

export type EditDragType =
    | "trim_left"
    | "trim_right"
    | "stretch_left"
    | "stretch_right"
    | "fade_in"
    | "fade_out"
    | "gain";

export type EditDragState = {
    type: EditDragType;
    pointerId: number;
    clipId: string;
    basestartSec: number;
    baselengthSec: number;
    basePlaybackRate: number;
    baseSourceStartSec: number;
    baseSourceEndSec: number;
    basefadeInSec: number;
    basefadeOutSec: number;
    baseGain: number;
    sourceBeats: number | null;
    rightEdgeBeat: number;
};

export function useEditDrag(deps: {
    scrollRef: React.RefObject<HTMLDivElement | null>;
    sessionRef: React.RefObject<SessionState>;
    dispatch: AppDispatch;
    snapBeat: (beat: number) => number;
    beatFromClientX: (
        clientX: number,
        bounds: DOMRect,
        xScroll: number,
    ) => number;
    /** modifier.clipNoSnap 绑定 */
    noSnapKb: Keybinding;
    /** 网格吸附全局开关 */
    gridSnapEnabled: boolean;
}) {
    const {
        scrollRef,
        sessionRef,
        dispatch,
        snapBeat,
        beatFromClientX,
        noSnapKb,
        gridSnapEnabled,
    } = deps;

    const editDragRef = useRef<EditDragState | null>(null);
    // 用于节流向后端发送 clip 状态更新，避免拖动时频繁覆盖与后端同步引起闪烁
    const lastRemoteSentRef = useRef<Record<string, number>>({});

    function startEditDrag(
        e: React.PointerEvent,
        clipId: string,
        type: EditDragType,
    ) {
        if (e.button !== 0) return;
        const clip = sessionRef.current.clips.find((c) => c.id === clipId);
        if (!clip) return;
        const scroller = scrollRef.current;
        if (!scroller) return;
        const rightEdgeBeat = clip.startSec + clip.lengthSec;

        dispatch(checkpointHistory());
        dispatch(beginInteraction());

        editDragRef.current = {
            type,
            pointerId: e.pointerId,
            clipId,
            basestartSec: clip.startSec,
            baselengthSec: clip.lengthSec,
            basePlaybackRate: Number(clip.playbackRate ?? 1) || 1,
            baseSourceStartSec: clip.sourceStartSec,
            baseSourceEndSec: clip.sourceEndSec,
            basefadeInSec: clip.fadeInSec,
            basefadeOutSec: clip.fadeOutSec,
            baseGain: clip.gain,
            sourceBeats: null,
            rightEdgeBeat,
        };

        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);

        function onMove(ev: PointerEvent) {
            const drag = editDragRef.current;
            const el = scrollRef.current;
            if (!drag || drag.pointerId !== e.pointerId || !el) return;
            const b = el.getBoundingClientRect();
            let beat = beatFromClientX(ev.clientX, b, el.scrollLeft);
            const shouldSnap =
                drag.type === "trim_left" ||
                drag.type === "trim_right" ||
                drag.type === "stretch_left" ||
                drag.type === "stretch_right";
            const noSnapActive = isModifierActive(noSnapKb, ev);
            const effectiveSnap = gridSnapEnabled
                ? !noSnapActive
                : noSnapActive;
            if (shouldSnap && effectiveSnap) {
                beat = snapBeat(beat);
            }

            const clipNow = sessionRef.current.clips.find(
                (c) => c.id === drag.clipId,
            );
            if (!clipNow) return;

            const minLen = 0.0;
            if (drag.type === "fade_in") {
                const raw = beat - drag.basestartSec;
                const next = clamp(raw, 0, Math.max(0, drag.baselengthSec));
                dispatch(
                    setClipFades({ clipId: drag.clipId, fadeInSec: next }),
                );
                // Throttle remote updates to at most ~200ms
                try {
                    const now = Date.now();
                    const last = lastRemoteSentRef.current[drag.clipId] || 0;
                    if (now - last > 200) {
                        lastRemoteSentRef.current[drag.clipId] = now;
                        void dispatch(
                            setClipStateRemote({
                                clipId: drag.clipId,
                                fadeInSec: next,
                            }),
                        );
                    }
                } catch {}
                return;
            }
            if (drag.type === "fade_out") {
                const raw = drag.rightEdgeBeat - beat;
                const next = clamp(raw, 0, Math.max(0, drag.baselengthSec));
                dispatch(
                    setClipFades({ clipId: drag.clipId, fadeOutSec: next }),
                );
                try {
                    const now = Date.now();
                    const last = lastRemoteSentRef.current[drag.clipId] || 0;
                    if (now - last > 200) {
                        lastRemoteSentRef.current[drag.clipId] = now;
                        void dispatch(
                            setClipStateRemote({
                                clipId: drag.clipId,
                                fadeOutSec: next,
                            }),
                        );
                    }
                } catch {}
                return;
            }
            if (drag.type === "gain") {
                const movementY = (ev.movementY ?? 0) as number;
                const deltaDb = -movementY * 0.25;
                const nextDb = clamp(gainToDb(clipNow.gain) + deltaDb, -12, 12);
                const nextGain = clamp(
                    dbToGain(nextDb),
                    dbToGain(-12),
                    dbToGain(12),
                );
                dispatch(setClipGain({ clipId: drag.clipId, gain: nextGain }));
                try {
                    const now = Date.now();
                    const last = lastRemoteSentRef.current[drag.clipId] || 0;
                    if (now - last > 200) {
                        lastRemoteSentRef.current[drag.clipId] = now;
                        void dispatch(
                            setClipStateRemote({
                                clipId: drag.clipId,
                                gain: nextGain,
                            }),
                        );
                    }
                } catch {}
                return;
            }

            if (drag.type === "trim_left") {
                const desiredStart = clamp(
                    beat,
                    0,
                    drag.rightEdgeBeat - minLen,
                );
                const desiredDelta = desiredStart - drag.basestartSec;
                const rate =
                    Number(clipNow.playbackRate ?? 1) > 0
                        ? Number(clipNow.playbackRate ?? 1)
                        : 1;
                let nextTrimStart =
                    drag.baseSourceStartSec + desiredDelta * rate;
                nextTrimStart = Math.max(0, nextTrimStart);
                const actualDeltaTrim = nextTrimStart - drag.baseSourceStartSec;
                const actualDeltaTimeline = actualDeltaTrim / rate;
                const nextStart = drag.basestartSec + actualDeltaTimeline;
                const nextLen = clamp(
                    drag.baselengthSec - actualDeltaTimeline,
                    minLen,
                    10_000,
                );
                dispatch(
                    moveClipStart({ clipId: drag.clipId, startSec: nextStart }),
                );
                dispatch(
                    setClipLength({ clipId: drag.clipId, lengthSec: nextLen }),
                );
                dispatch(
                    setClipSourceRange({
                        clipId: drag.clipId,
                        sourceStartSec: nextTrimStart,
                    }),
                );
                return;
            }

            if (drag.type === "stretch_left") {
                const desiredStart = clamp(
                    beat,
                    0,
                    drag.rightEdgeBeat - minLen,
                );
                const rawLen = clamp(
                    drag.rightEdgeBeat - desiredStart,
                    minLen,
                    10_000,
                );
                const baseLen = Math.max(1e-6, Number(drag.baselengthSec) || 0);
                const baseRate =
                    drag.basePlaybackRate > 0 &&
                    Number.isFinite(drag.basePlaybackRate)
                        ? drag.basePlaybackRate
                        : 1;
                const nextRate = clamp(
                    (baseRate * baseLen) / Math.max(1e-6, rawLen),
                    0.1,
                    10,
                );
                // 用 clamp 后的 rate 反算真实长度，确保 lengthSec 和 playbackRate 一致
                const correctedLen = (baseRate * baseLen) / nextRate;
                const nextStart = drag.rightEdgeBeat - correctedLen;
                dispatch(
                    moveClipStart({ clipId: drag.clipId, startSec: nextStart }),
                );
                dispatch(
                    setClipLength({
                        clipId: drag.clipId,
                        lengthSec: correctedLen,
                    }),
                );
                dispatch(
                    setClipPlaybackRate({
                        clipId: drag.clipId,
                        playbackRate: nextRate,
                    }),
                );
                return;
            }

            if (drag.type === "trim_right") {
                const desiredRight = clamp(
                    beat,
                    drag.basestartSec + minLen,
                    10_000,
                );
                const rate =
                    Number(clipNow.playbackRate ?? 1) > 0
                        ? Number(clipNow.playbackRate ?? 1)
                        : 1;
                // 计算源文件总时长，用于 clamp sourceEndSec 的上限
                const sourceDuration = (() => {
                    if (
                        clipNow.durationFrames &&
                        clipNow.sourceSampleRate &&
                        clipNow.sourceSampleRate > 0
                    ) {
                        return (
                            clipNow.durationFrames / clipNow.sourceSampleRate
                        );
                    }
                    return Number(clipNow.durationSec ?? 0) || 0;
                })();
                const desiredLen = desiredRight - drag.basestartSec;
                const nextLen = clamp(desiredLen, minLen, 10_000);
                const usedDeltaTimeline = nextLen - drag.baselengthSec;
                let nextTrimEnd =
                    drag.baseSourceEndSec + usedDeltaTimeline * rate;
                nextTrimEnd = Math.max(0, nextTrimEnd);
                // 不允许超出源文件实际时长
                if (sourceDuration > 0) {
                    nextTrimEnd = Math.min(nextTrimEnd, sourceDuration);
                }
                // 反算实际可用的 timeline 长度（sourceEndSec 被 clamp 后，lengthSec 也要同步受限）
                const actualSourceLen =
                    nextTrimEnd - (clipNow.sourceStartSec ?? 0);
                const maxTimelineLen = actualSourceLen / rate;
                const finalLen =
                    maxTimelineLen > 0
                        ? Math.min(nextLen, maxTimelineLen)
                        : nextLen;
                dispatch(
                    setClipLength({ clipId: drag.clipId, lengthSec: finalLen }),
                );
                dispatch(
                    setClipSourceRange({
                        clipId: drag.clipId,
                        sourceEndSec: nextTrimEnd,
                    }),
                );
                return;
            }

            if (drag.type === "stretch_right") {
                const desiredRight = clamp(
                    beat,
                    drag.basestartSec + minLen,
                    10_000,
                );
                const rawLen = clamp(
                    desiredRight - drag.basestartSec,
                    minLen,
                    10_000,
                );
                const baseLen = Math.max(1e-6, Number(drag.baselengthSec) || 0);
                const baseRate =
                    drag.basePlaybackRate > 0 &&
                    Number.isFinite(drag.basePlaybackRate)
                        ? drag.basePlaybackRate
                        : 1;
                const nextRate = clamp(
                    (baseRate * baseLen) / Math.max(1e-6, rawLen),
                    0.1,
                    10,
                );
                // 用 clamp 后的 rate 反算真实长度，确保 lengthSec 和 playbackRate 一致
                const correctedLen = (baseRate * baseLen) / nextRate;
                dispatch(
                    setClipLength({
                        clipId: drag.clipId,
                        lengthSec: correctedLen,
                    }),
                );
                dispatch(
                    setClipPlaybackRate({
                        clipId: drag.clipId,
                        playbackRate: nextRate,
                    }),
                );
            }
        }

        function end() {
            const drag = editDragRef.current;
            if (!drag || drag.pointerId !== e.pointerId) return;
            editDragRef.current = null;

            const clipNow = sessionRef.current.clips.find(
                (c) => c.id === drag.clipId,
            );
            if (!clipNow) {
                dispatch(endInteraction());
                return;
            }

            // 交互锁在最终持久化请求完成后才释放，
            // 避免 endInteraction() 到 fulfilled 之间的窗口内，
            // 其他 in-flight thunk 的旧快照覆盖前端乐观更新导致闪烁。

            let persistPromise: Promise<unknown> | null = null;
            if (drag.type === "trim_left") {
                persistPromise = dispatch(
                    setClipStateRemote({
                        clipId: drag.clipId,
                        startSec: clipNow.startSec,
                        lengthSec: clipNow.lengthSec,
                        sourceStartSec: clipNow.sourceStartSec,
                    }),
                ).unwrap();
            } else if (drag.type === "trim_right") {
                persistPromise = dispatch(
                    setClipStateRemote({
                        clipId: drag.clipId,
                        lengthSec: clipNow.lengthSec,
                        sourceEndSec: clipNow.sourceEndSec,
                    }),
                ).unwrap();
            } else if (drag.type === "stretch_left") {
                persistPromise = dispatch(
                    setClipStateRemote({
                        clipId: drag.clipId,
                        startSec: clipNow.startSec,
                        lengthSec: clipNow.lengthSec,
                        playbackRate: clipNow.playbackRate,
                    }),
                ).unwrap();
            } else if (drag.type === "stretch_right") {
                persistPromise = dispatch(
                    setClipStateRemote({
                        clipId: drag.clipId,
                        lengthSec: clipNow.lengthSec,
                        playbackRate: clipNow.playbackRate,
                    }),
                ).unwrap();
            } else if (drag.type === "fade_in") {
                // 确保结束时把最终值持久化到后端
                persistPromise = dispatch(
                    setClipStateRemote({
                        clipId: drag.clipId,
                        fadeInSec: clipNow.fadeInSec,
                    }),
                ).unwrap();
            } else if (drag.type === "fade_out") {
                persistPromise = dispatch(
                    setClipStateRemote({
                        clipId: drag.clipId,
                        fadeOutSec: clipNow.fadeOutSec,
                    }),
                ).unwrap();
            } else if (drag.type === "gain") {
                persistPromise = dispatch(
                    setClipStateRemote({
                        clipId: drag.clipId,
                        gain: clipNow.gain,
                    }),
                ).unwrap();
            }

            const shouldApplyAutoCrossfade =
                sessionRef.current.autoCrossfadeEnabled &&
                (drag.type === "trim_left" ||
                    drag.type === "trim_right" ||
                    drag.type === "stretch_left" ||
                    drag.type === "stretch_right");
            if (shouldApplyAutoCrossfade) {
                void Promise.resolve(persistPromise).finally(() => {
                    void applyAutoCrossfade(
                        sessionRef.current,
                        [drag.clipId],
                        dispatch,
                    );
                });
            }

            // 拉伸后同步参数线：当"锁定参数线"启用时，将旧范围内的参数值时域映射到新范围
            const isStretch =
                drag.type === "stretch_left" || drag.type === "stretch_right";
            if (
                isStretch &&
                sessionRef.current.lockParamLinesEnabled &&
                clipNow.trackId
            ) {
                const stretchTrackId = clipNow.trackId;
                const oldStartSec = drag.basestartSec;
                const oldLengthSec = drag.baselengthSec;
                const newStartSec = clipNow.startSec;
                const newLengthSec = clipNow.lengthSec;
                void Promise.resolve(persistPromise).then(() =>
                    stretchLinkedParams(
                        stretchTrackId,
                        oldStartSec,
                        oldLengthSec,
                        newStartSec,
                        newLengthSec,
                    ),
                );
            }

            // 在所有持久化请求完成后释放交互锁
            void Promise.resolve(persistPromise).finally(() => {
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

    return { editDragRef, startEditDrag };
}
