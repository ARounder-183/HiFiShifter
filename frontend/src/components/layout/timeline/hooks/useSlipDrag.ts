import { useRef } from "react";
import type { AppDispatch } from "../../../../app/store";
import type { SessionState } from "../../../../features/session/sessionSlice";
import type { TimelineSnapSettings } from "../../../../features/session/sessionTypes";
import {
    checkpointHistory,
    setClipSourceRange,
    beginInteraction,
    endInteraction,
} from "../../../../features/session/sessionSlice";
import { setClipStateRemote } from "../../../../features/session/thunks/timelineThunks";
import { webApi } from "../../../../services/webviewApi";
import { resolveClipContentDurationSec } from "../../../../utils/loopRender";
import {
    loopSnapThresholdSec,
    nearestBoundarySnapOffsetSec,
    slipBoundaryAlignedSides,
} from "../../../../utils/loopSnap";
import {
    beginSnapGesture,
    computeEffectiveSnap,
    endSnapGesture,
} from "../../../../utils/timelineSnapping";
import {
    SNAP_HIGHLIGHT_GROUP,
    buildLoopBoundaryHighlightEntry,
    clearSnapHighlights,
    publishSnapHighlights,
} from "../../../../utils/snapHighlight";
import { isModifierActive } from "../../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../../features/keybindings/types";
import { expandClipIdsWithGroups } from "./useGroupExpansion";

/**
 * Slip（内容平移）拖拽 —— **增量式实时状态驱动**。
 *
 * 设计要点（吸取历史缺陷教训）：
 * - 不在拖拽开始时冻结任何几何基线。每个指针事件都从 Redux 的**当前**
 *   Clip 状态出发，只应用本事件的增量 —— 权威载荷（split/paste 等）
 *   无论何时落地，后续事件都自动基于最新真实值，不存在"过期基线"。
 * - 倒放方向：可闻事件位于 t=(se−p)/r（镜像时间轴）。源窗口若随指针
 *   同向平移会让事件反向移动（拖左反而更晚），因此倒放取 dir=−1，
 *   保证"内容跟随拖动方向"与正放手感一致。
 * - 持久化使用交互数学的最终值（lastById），不回读 Redux。
 */

export type SlipDragState = {
    pointerId: number;
    anchorClipId: string;
    clipIds: string[];
    /** 拖拽起点指针位置（秒）：累计吸附的原始位移基准。 */
    startPointerBeat: number;
    /**
     * 已应用于 clip 的**累计**指针位移（含吸附修正，指针空间秒）。
     * 实时吸附以"目标累计值 − 已应用累计值"的差值驱动增量分发。
     */
    appliedTotal: number;
    /** 锚 clip 启动时快照：循环节候选族只依赖初始几何（平移不变）。 */
    anchorSnapshot: {
        loopEnabled: boolean;
        reversed: boolean;
        sourceStartSec: number;
        sourceEndSec: number;
        playbackRate: number;
        lengthSec: number;
        durationFrames: number | null;
        sourceSampleRate: number | null;
        contentDurationSec: number | null;
        isContentBearing: boolean;
    };
    /**
     * 每个 clip 最近一次分发的源窗口值。持久化必须使用这里记录的
     * 交互数学结果，不回读 Redux（防并发更新/历史归一化污染）。
     */
    lastById: Record<string, { sourceStartSec: number; sourceEndSec: number }>;
};

/** 从 SessionState 的 ClipInfo 提取 Slip 所需字段。 */
function readClip(c: SessionState["clips"][number]) {
    const playbackRate = Number(c.playbackRate ?? 1) || 1;
    const sourceStartSec = Number(c.sourceStartSec ?? 0) || 0;
    const sourceEndSec = Number(c.sourceEndSec ?? 0) || 0;
    const isContentBearing = !!c.sourcePath || !!(c.midiNoteData && c.midiNoteData.length > 0);
    const contentDurSec = resolveClipContentDurationSec({
        sourcePath: c.sourcePath,
        midiNoteData: c.midiNoteData ?? null,
        durationFrames: c.durationFrames,
        sourceSampleRate: c.sourceSampleRate,
        durationSec: c.durationSec,
    });
    return {
        playbackRate: playbackRate > 0 && Number.isFinite(playbackRate) ? playbackRate : 1,
        sourceStartSec,
        sourceEndSec,
        lengthSec: Math.max(0, Number(c.lengthSec ?? 0) || 0),
        reversed: !!c.reversed,
        loopEnabled: !!c.loopEnabled,
        isContentBearing,
        contentDurSec,
    };
}

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
    /** "拖动时切换吸附"修饰键绑定（XOR 取反吸附总开关）。 */
    noSnapKb: Keybinding;
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
        noSnapKb,
    } = deps;

    const slipDragRef = useRef<SlipDragState | null>(null);

    /** 对单个 clip 应用窗口平移增量（读取当前真实状态，返回结果窗口）。 */
    function computeShiftedWindow(
        id: string,
        deltaBeat: number,
    ): { sourceStartSec: number; sourceEndSec: number } | null {
        const c = sessionRef.current.clips.find((x) => x.id === id);
        if (!c) return null;
        const v = readClip(c);
        // 方向语义：倒放的播放时间轴是镜像的，源窗口必须沿指针反方向平移。
        const dir = v.reversed ? -1 : 1;
        const deltaSrcSec = deltaBeat * v.playbackRate * dir;
        let nextSourceStart = v.sourceStartSec + deltaSrcSec;
        let nextSourceEnd = v.sourceEndSec + deltaSrcSec;

        if (v.loopEnabled) {
            // Loop（循环源）：窗口两端对内容时长取模环绕（floor_mod），与
            // 渲染/引擎的回绕映射一致；音高参考块的 D = 音符内容范围。
            if (v.contentDurSec != null && v.contentDurSec > 1e-9) {
                const mediaDur = v.contentDurSec;
                nextSourceStart = ((nextSourceStart % mediaDur) + mediaDur) % mediaDur;
                nextSourceEnd = ((nextSourceEnd % mediaDur) + mediaDur) % mediaDur;
            }
            // 内容时长未知：保持平移，避免卡死。
        } else if (v.isContentBearing && !v.reversed) {
            // 非 Loop 正放：派生窗口模型 —— 终点 = 起点 + len·rate；
            // 越出媒体的部分渲染静音（前导/尾部对称无界）。
            nextSourceEnd = nextSourceStart + v.lengthSec * v.playbackRate;
        } else {
            // 非 Loop 倒放：source_end 是反向锚点，跨度可合法大于 len·rate
            //（延伸产生的静音区）。只做整体平移，保持跨度不变。
        }
        return { sourceStartSec: nextSourceStart, sourceEndSec: nextSourceEnd };
    }

    function startSlipDrag(e: React.PointerEvent<HTMLDivElement>, clipId: string) {
        if (e.button !== 0) return;
        const anchor = sessionRef.current.clips.find((c) => c.id === clipId);
        if (!anchor) return;
        const scroller = scrollRef.current;
        if (!scroller) return;

        // 交互锁 / dirty 标记推迟到首次真实移动（B8）：纯点击（零位移）不
        // 置 dirty、不开锁 —— 与 useClipDrag/useSnapOffsetDrag 一致。
        let armed = false;

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

        const anchorRead = readClip(anchor);
        slipDragRef.current = {
            pointerId: e.pointerId,
            anchorClipId: clipId,
            clipIds,
            startPointerBeat: beatAtPointer,
            appliedTotal: 0,
            anchorSnapshot: {
                loopEnabled: anchorRead.loopEnabled,
                reversed: anchorRead.reversed,
                sourceStartSec: anchorRead.sourceStartSec,
                sourceEndSec: anchorRead.sourceEndSec,
                playbackRate: anchorRead.playbackRate,
                lengthSec: anchorRead.lengthSec,
                durationFrames: anchor.durationFrames ?? null,
                sourceSampleRate: anchor.sourceSampleRate ?? null,
                contentDurationSec: anchorRead.contentDurSec,
                isContentBearing: anchorRead.isContentBearing,
            },
            lastById: {},
        };

        beginSnapGesture();

        (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);

        function onMove(ev: PointerEvent) {
            const drag = slipDragRef.current;
            const el = scrollRef.current;
            if (!drag || drag.pointerId !== e.pointerId || !el) return;
            // 首次真实移动时武装交互（锁 + dirty 标记）。
            if (!armed) {
                if (Math.abs(ev.clientX - e.clientX) < 2 && Math.abs(ev.clientY - e.clientY) < 2) {
                    return;
                }
                armed = true;
                dispatch(checkpointHistory());
                dispatch(beginInteraction());
            }
            const b = el.getBoundingClientRect();
            const beatNow = beatFromClientX(ev.clientX, b, el.scrollLeft);

            // ── 实时循环节/内容边界吸附（拖拽全程生效）────────────────
            // 属于常规吸附体系：受"吸附"总开关与"拖动时切换吸附"修饰键
            // （XOR）控制，且需在吸附设置中启用"Clip 边缘吸附到源素材首尾"。
            // 吸附距离读自 snapDistancePx。候选族只依赖锚 clip 的**初始**
            // 几何（平移不变）：Loop 为媒体边界相位对齐 Clip 边缘的 mod-D
            // 等差族；非 Loop 为媒体边界对齐 Clip 边缘的有限候选。命中时把
            // 累计位移替换为吸附值，再以"目标累计 − 已应用累计"驱动增量。
            let desiredTotal = drag.startPointerBeat - beatNow;
            {
                const a = drag.anchorSnapshot;
                const noSnapActive = isModifierActive(noSnapKb, ev);
                const effectiveSnap = computeEffectiveSnap(timelineSnap.enabled, noSnapActive);
                if (
                    (a.isContentBearing || a.loopEnabled) &&
                    timelineSnap.snapClipsToSourceMedia &&
                    effectiveSnap &&
                    timelineSnap.snapDistancePx > 0
                ) {
                    const dir = a.reversed ? -1 : 1;
                    const rawWindowShift = desiredTotal * dir;
                    const snappedW = nearestBoundarySnapOffsetSec(
                        {
                            loopEnabled: a.loopEnabled,
                            reversed: a.reversed,
                            sourceStartSec: a.sourceStartSec,
                            sourceEndSec: a.sourceEndSec,
                            playbackRate: a.playbackRate,
                            lengthSec: a.lengthSec,
                            durationFrames: a.durationFrames,
                            sourceSampleRate: a.sourceSampleRate,
                            contentDurationSec: a.contentDurationSec,
                        },
                        "slip",
                        rawWindowShift,
                    );
                    if (
                        snappedW != null &&
                        Math.abs(snappedW - rawWindowShift) <=
                            loopSnapThresholdSec(timelineSnap.snapDistancePx, pxPerSec) + 1e-12
                    ) {
                        desiredTotal = snappedW * dir;
                        // 循环节命中：只高亮**真正对齐**的那一侧（媒体边界恰好
                        // 落在 Clip 起点 → 高亮起点；落在终点 → 高亮终点；
                        // len·r 恰为整周期等两侧同时对齐时才两缘同亮）。
                        const anchorClip = sessionRef.current.clips.find(
                            (c) => c.id === drag.anchorClipId,
                        );
                        if (anchorClip) {
                            const aligned = slipBoundaryAlignedSides(a, snappedW);
                            const clipStartSec = Math.max(0, Number(anchorClip.startSec) || 0);
                            const clipLen = Math.max(0, Number(anchorClip.lengthSec) || 0);
                            const secs: number[] = [];
                            if (aligned.start) secs.push(clipStartSec);
                            if (aligned.end) secs.push(clipStartSec + clipLen);
                            if (secs.length > 0) {
                                publishSnapHighlights(SNAP_HIGHLIGHT_GROUP, [
                                    buildLoopBoundaryHighlightEntry({
                                        secs,
                                        trackId: anchorClip.trackId,
                                        clipId: drag.anchorClipId,
                                    }),
                                ]);
                            } else {
                                clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                            }
                        }
                    } else {
                        clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                    }
                } else {
                    clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                }
            }

            const dApplied = desiredTotal - drag.appliedTotal;
            if (Math.abs(dApplied) < 1e-12) return;
            drag.appliedTotal = desiredTotal;

            for (const id of drag.clipIds) {
                const next = computeShiftedWindow(id, dApplied);
                if (!next) continue;
                drag.lastById[id] = next;
                dispatch(
                    setClipSourceRange({
                        clipId: id,
                        sourceStartSec: next.sourceStartSec,
                        sourceEndSec: next.sourceEndSec,
                    }),
                );
            }
        }

        function end(ev: PointerEvent) {
            const drag = slipDragRef.current;
            if (!drag || drag.pointerId !== ev.pointerId) return;
            slipDragRef.current = null;
            endSnapGesture();

            // 无操作守卫（B7）：零位移/拖回原点 → 不落盘、不开 undo group、
            // 不产生死撤销步。注意未武装（纯点击）时也不能解锁（锁未开）。
            const zeroNet = drag.appliedTotal === 0 || Math.abs(drag.appliedTotal) < 1e-12;
            if (zeroNet) {
                if (armed) {
                    dispatch(endInteraction());
                    armed = false;
                }
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", end);
                window.removeEventListener("pointercancel", end);
                return;
            }

            // 持久化交互数学的最终值（不回读 Redux）。实时吸附已在 move 中
            // 把累计位移收敛到循环节候选上，无需松手二次修正。
            const patches = drag.clipIds
                .map((id) => {
                    const last = drag.lastById[id];
                    if (!last) return null;
                    return {
                        clipId: id,
                        sourceStartSec: last.sourceStartSec,
                        sourceEndSec: last.sourceEndSec,
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

    return startSlipDrag;
}
