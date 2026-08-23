import { useRef } from "react";
import type { AppDispatch } from "../../../../app/store";
import type { SessionState } from "../../../../features/session/sessionSlice";
import type { TimelineSnapSettings } from "../../../../features/session/sessionTypes";
import {
    checkpointHistory,
    setClipStateRemote,
    setClipSourceRange,
    beginInteraction,
    endInteraction,
} from "../../../../features/session/sessionSlice";
import { webApi } from "../../../../services/webviewApi";
import {
    clipMediaDurationSec,
    loopSnapThresholdSec,
    nearestBoundarySnapOffsetSec,
} from "../../../../utils/loopSnap";
import { expandClipIdsWithGroups } from "./useGroupExpansion";

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
            /** Loop（循环源）：内容按整个媒体时长回绕，Slip 可无限平移。 */
            loopEnabled: boolean;
            reversed: boolean;
            /** 有真实音频媒体（非纯 MIDI）：非 Loop 采用派生窗口模型自由平移。 */
            hasMedia: boolean;
            lengthSec: number;
            durationFrames: number | null;
            sourceSampleRate: number | null;
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
    ignoreGrouping: boolean;
    /** 完整吸附设置：循环节吸附距离从 snapDistancePx 读取（无论 enabled 与否都生效）。 */
    timelineSnap: TimelineSnapSettings;
    /** 当前缩放（像素/秒）：用于把吸附距离换算成秒。 */
    pxPerSec: number;
}) {
    const {
        scrollRef,
        sessionRef,
        dispatch,
        multiSelectedClipIds,
        multiSelectedSet,
        beatFromClientX,
        ignoreGrouping,
        timelineSnap,
        pxPerSec,
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

        // Expand to include selected clips and their group members
        const initialIds =
            multiSelectedClipIds.length > 0 && multiSelectedSet.has(clipId)
                ? [...multiSelectedClipIds]
                : [clipId];
        const clipIds = ignoreGrouping
            ? initialIds
            : expandClipIdsWithGroups(
                  initialIds,
                  sessionRef.current.clips,
                  false,
                  sessionRef.current.disabledGroupIds,
              );

        const initialById: SlipDragState["initialById"] = {};
        for (const id of clipIds) {
            const c = sessionRef.current.clips.find((x) => x.id === id);
            if (!c) continue;
            // MIDI clip：从 midiNoteData 计算源时长；音频 clip：使用 durationSec
            let sourceDurationSec: number | null;
            if (c.midiNoteData && c.midiNoteData.length > 0) {
                sourceDurationSec = c.midiNoteData.reduce((max, n) => Math.max(max, n.endSec), 0);
            } else {
                sourceDurationSec = Number(c.durationSec ?? 0) || null;
            }
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
                loopEnabled: !!c.loopEnabled,
                reversed: !!c.reversed,
                hasMedia: !!c.sourcePath && !(c.midiNoteData && c.midiNoteData.length > 0),
                lengthSec: Math.max(0, Number(c.lengthSec ?? 0) || 0),
                durationFrames: c.durationFrames ?? null,
                sourceSampleRate: c.sourceSampleRate ?? null,
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

            // ── 循环节/内容边界吸附（Slip 版）─────────────────────────
            // 无论"吸附"功能是否启用都生效；吸附距离读自吸附设置的 snapDistancePx。
            // 候选偏移族（相对 Clip 基准起点的 Slip 偏移 δ，δ·rate 为源域平移量）：
            // - Loop：媒体边界相位与 Clip 起点对齐的 mod-D 族（±len 第二族见工具注释）；
            // - 非 Loop：媒体边界（s=0 与 s=D）对齐到 Clip 起点/终点的有限候选
            //   —— 即让"原始媒体内容在 Clip 内的终止位置"精确落到边缘上。
            {
                const anchorInitial = drag.initialById[drag.anchorClipId];
                if (anchorInitial?.hasMedia || anchorInitial?.loopEnabled) {
                    const snappedOffset = nearestBoundarySnapOffsetSec(
                        {
                            loopEnabled: !!anchorInitial.loopEnabled,
                            reversed: !!anchorInitial.reversed,
                            sourceStartSec: anchorInitial.sourceStartSec,
                            sourceEndSec: anchorInitial.sourceEndSec,
                            playbackRate: anchorInitial.playbackRate,
                            lengthSec: anchorInitial.lengthSec,
                            durationFrames: anchorInitial.durationFrames,
                            sourceSampleRate: anchorInitial.sourceSampleRate,
                        },
                        "slip",
                        deltaBeat,
                    );
                    if (
                        snappedOffset != null &&
                        Math.abs(snappedOffset - deltaBeat) <=
                            loopSnapThresholdSec(timelineSnap.snapDistancePx, pxPerSec) + 1e-12
                    ) {
                        deltaBeat = snappedOffset;
                    }
                }
            }

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

                if (initial.loopEnabled) {
                    // Loop（循环源）：内容按整个媒体文件回绕，Slip 可无限向左/向右
                    // 平移 —— 源窗口两端对媒体时长取模环绕（floor_mod），与渲染/
                    // 引擎的回绕映射一致。窗口跨越边界时 end < start 为合法的
                    // "环绕窗口"状态（split 等操作已产生同类状态）。
                    const mediaDur = clipMediaDurationSec(initial);
                    if (mediaDur != null && mediaDur > 1e-9) {
                        nextSourceStart = ((nextSourceStart % mediaDur) + mediaDur) % mediaDur;
                        nextSourceEnd = ((nextSourceEnd % mediaDur) + mediaDur) % mediaDur;
                    } else {
                        // 媒体时长未知：退化为不钳制（保持平移），避免卡死。
                    }
                } else if (initial.hasMedia) {
                    // 非 Loop 音频 Clip：**派生窗口模型**（REAPER 语义）——
                    // Clip 消费源区间 [source_start, source_start + len·rate)，
                    // 区间落在 [0, D) 之外的部分渲染为静音。同时把 source_end
                    // 归一到派生值，自愈历史数据中 length 与窗口跨度不一致的
                    // 状态。
                    // 向左 / 向右延伸**完全对称、均不设限**（产品决策更新）：
                    // 左延伸产生前导静音（source_start < 0），右延伸产生尾部
                    // 静音（终点 > D），静音都随内容一起平移。Loop 开关切换
                    // 或历史数据留下的陈旧窗口由派生值自愈。
                    nextSourceEnd = nextSourceStart + initial.lengthSec * rate;
                } else {
                    // 纯 MIDI（无音频媒体）：维持既有音符窗口内钳制。
                    const maxSlipSec = initial.maxSlipSec;
                    if (Number.isFinite(maxSlipSec) && maxSlipSec > 1e-6) {
                        if (nextSourceStart < 0) {
                            nextSourceEnd -= nextSourceStart;
                            nextSourceStart = 0;
                        }
                        if (nextSourceEnd > maxSlipSec) {
                            nextSourceStart -= nextSourceEnd - maxSlipSec;
                            nextSourceEnd = maxSlipSec;
                        }
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
            const patches = drag.clipIds
                .map((id) => {
                    const now = session.clips.find((c) => c.id === id);
                    if (!now) return null;
                    return {
                        clipId: id,
                        sourceStartSec: Number(now.sourceStartSec ?? 0) || 0,
                        sourceEndSec: Number(now.sourceEndSec ?? 0) || 0,
                    };
                })
                .filter(
                    (
                        patch,
                    ): patch is {
                        clipId: string;
                        sourceStartSec: number;
                        sourceEndSec: number;
                    } => patch != null,
                );

            let persistPromise: Promise<unknown>;
            if (patches.length <= 1) {
                const patch = patches[0];
                persistPromise = patch
                    ? dispatch(
                          setClipStateRemote({
                              clipId: patch.clipId,
                              sourceStartSec: patch.sourceStartSec,
                              sourceEndSec: patch.sourceEndSec,
                          }),
                      ).unwrap()
                    : Promise.resolve();
            } else {
                persistPromise = (async () => {
                    await webApi.beginUndoGroup();
                    try {
                        const persistPromises = patches.map((patch) =>
                            dispatch(
                                setClipStateRemote({
                                    clipId: patch.clipId,
                                    sourceStartSec: patch.sourceStartSec,
                                    sourceEndSec: patch.sourceEndSec,
                                    checkpoint: false,
                                }),
                            ).unwrap(),
                        );
                        await Promise.allSettled(persistPromises);
                    } finally {
                        await webApi.endUndoGroup();
                    }
                })();
            }

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

    return { slipDragRef, startSlipDrag };
}
