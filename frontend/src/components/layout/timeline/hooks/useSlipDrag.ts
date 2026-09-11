import { useRef } from "react";
import { registerDragAbort } from "../gestureFocusGuard";
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

/**
 * slip 的几何与字段提取统一放在 `slipWindow`（旧实现与渲染内核共用一份，
 * 避免倒放 / 循环 / 非内容承载这几条分支在两个渲染模式下分叉）。
 */
import { computeSlipWindow, toBoundarySnapClip } from "./slipWindow";

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

        // 媒体边界吸附视图与渲染内核共用 `toBoundarySnapClip`（内容时长 D 的解析
        // 规则单一来源，含吸附参与条件 `isContentBearing`）。
        const anchorBoundary = toBoundarySnapClip(anchor);
        slipDragRef.current = {
            pointerId: e.pointerId,
            anchorClipId: clipId,
            clipIds,
            startPointerBeat: beatAtPointer,
            appliedTotal: 0,
            anchorSnapshot: {
                loopEnabled: anchorBoundary.loopEnabled,
                reversed: anchorBoundary.reversed,
                sourceStartSec: anchorBoundary.sourceStartSec,
                sourceEndSec: anchorBoundary.sourceEndSec,
                playbackRate: anchorBoundary.playbackRate,
                lengthSec: anchorBoundary.lengthSec,
                durationFrames: anchorBoundary.durationFrames ?? null,
                sourceSampleRate: anchorBoundary.sourceSampleRate ?? null,
                contentDurationSec: anchorBoundary.contentDurationSec ?? null,
                isContentBearing: anchorBoundary.isContentBearing,
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
                const clip = sessionRef.current.clips.find((item) => item.id === id);
                if (clip === undefined) continue;
                const next = computeSlipWindow(clip, dApplied);
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

        function finish() {
            const drag = slipDragRef.current;
            if (!drag) return;
            slipDragRef.current = null;
            // 收尾第一步注销失焦守卫（幂等防双触发）。
            unregisterAbort();
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

            // 持久化失败（后端拒绝/网络错误）不得变成 unhandledrejection；
            // endInteraction 已在 finally 内保证执行。
            void Promise.resolve(persistPromise)
                .catch(() => {
                    // 失败已由 setClipStateRemote 的 rejected reducer 呈现给用户。
                })
                .finally(() => {
                    dispatch(endInteraction());
                });

            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", end);
            window.removeEventListener("pointercancel", end);
        }

        function end(ev: PointerEvent) {
            const drag = slipDragRef.current;
            if (!drag || drag.pointerId !== ev.pointerId) return;
            finish();
        }

        // 失焦取消：切屏期间 pointerup/pointercancel 不送达本窗口，注册
        // 事件无关的 finish()，由 gestureFocusGuard 在窗口 blur 时统一收尾。
        const unregisterAbort = registerDragAbort(finish);

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", end);
        window.addEventListener("pointercancel", end);
    }

    return startSlipDrag;
}
