// 时间轴内核裁剪 / 边缘拉伸 / 吸附偏移手势。拆自 TimelinePanel。
import React from "react";
import { isModifierActive } from "../../../../features/keybindings/keybindingsSlice";
import { publishFadeRichTooltip } from "../fadeTooltipText";
import {
    buildSnapOffsetInfoText,
    type TimeValueFormatContext,
    type TimeValueLabelLookup,
} from "../timeValueText";
import {
    setClipStateRemote,
    setClipsStateBulkRemote,
    setClipFades,
    setClipPlaybackRate,
    moveClipStart,
    checkpointHistory,
    bumpParamsEpoch,
    setClipLength,
    setClipSourceRange,
    setClipSnapOffset,
} from "../../../../features/session/sessionSlice";
import type { TimelineSessionSlice } from "./useTimelineState";
import { beginSnapGesture, endSnapGesture } from "../../../../utils/timelineSnapping";
import { batch } from "react-redux";
import { holdLoudnessFetch, releaseLoudnessFetch } from "../loudnessFetchGate";
import { resolveTrimSourceWindow, resolveTrimSnapOffset } from "../trimSourceWindow";
import { resolveClipContentDurationSec } from "../../../../utils/loopRender";
import { resolveRootTrackId } from "../../../../features/session/trackUtils";
import {
    SNAP_HIGHLIGHT_GROUP,
    buildLoopBoundaryHighlightEntry,
    clearSnapHighlights,
    publishSnapHighlights,
} from "../../../../utils/snapHighlight";
import { loopSnapThresholdSec, nearestBoundarySnapOffsetSec } from "../../../../utils/loopSnap";
import { resolveKernelEditParticipants, type KernelEditParticipant } from "../hooks/kernelEditSet";
import {
    applyRippleFollowerShift,
    buildRippleFollowers,
    type RippleFollowerMap,
} from "../../../../features/session/ripplePreview";
import {
    applyAutoCrossfade,
    applyDetachedAutoCrossfadeClears,
    computeInitialCrossfadeSides,
    previewAutoCrossfade,
} from "../hooks/autoCrossfade";
import { computeEffectiveSnap } from "../../../../utils/timelineSnapping";
import { store } from "../../../../app/store";
import {
    buildStretchGroupState,
    computeClipStretch,
    computeRegionRightEdgeDelta,
    computeStretchGroupUpdate,
    scaleSnapOffsetForStretch,
    type StretchGroupState,
} from "../hooks/stretchGroup";
import {
    stretchLinkedParams,
    stretchTrackLinkedParams,
    type StretchRangeMapping,
} from "../hooks/stretchParams";
import { toBoundarySnapClip } from "../hooks/slipWindow";
import type { useTimelineState } from "../hooks/useTimelineState";
import type { useI18n } from "../../../../i18n/I18nProvider";
import type { AppDispatch } from "../../../../app/store";

type Deps = {
    state: ReturnType<typeof useTimelineState>;
    s: TimelineSessionSlice;
    dispatch: AppDispatch;
    multiSelectedClipIds: string[];
    timeValueFormatCtx: TimeValueFormatContext;
    tVars: ReturnType<typeof useI18n>["tVars"];
    beginKernelGestureInteraction: () => void;
    endKernelGestureInteraction: () => void;
};

export function useTimelineKernelTrim({
    state,
    s,
    dispatch,
    multiSelectedClipIds,
    timeValueFormatCtx,
    tVars,
    beginKernelGestureInteraction,
    endKernelGestureInteraction,
}: Deps) {
    const { sessionRef, stretchKbRef, pxPerSecRef, snapTimelineDetailed, noSnapKb } = state;
    const kernelTrimOriginRef = React.useRef<{
        clipId: string;
        startSec: number;
        lengthSec: number;
        trackId: string;
        sourceStartSec: number;
        sourceEndSec: number;
        /**
         * 锚 clip 是否倒放 / 是否 Loop。
         *
         * 【为什么必须记下】裁切的源窗口换算对倒放是**镜像**的（时间轴右端对应
         * `sourceStartSec`）、对 Loop 则完全不该动源字段 —— 见 `trimSourceWindow`。
         * 此前 origin 只记 `sourceStartSec / sourceEndSec`，换算处无从判断方向，
         * 于是倒放 Clip 拖右缘改到了左端。
         */
        reversed: boolean;
        loopEnabled: boolean;
        /**
         * 本次边缘手势的模式。
         *
         * `Alt`（`modifier.clipStretch`）按住 = **拉伸**（改播放速率、内容不被裁掉），
         * 否则 = **裁切**（改源区间）。模式在**按下时**定死，拖拽中途按/松 Alt 不切换
         * ——与旧实现一致（旧实现由 `altPressed` 在 pointerdown 时快照）。
         */
        mode: "trim" | "stretch";
        /** 拉伸所需的按下时基准（裁切模式不使用）。 */
        basePlaybackRate: number;
        /**
         * 裁切源位移用的**组合**消费速率（`clipPlaybackRate × take.playbackRate`）。
         *
         * 【为什么不能复用 `basePlaybackRate`】后者是 **clip 级**倍率，Alt 拉伸会把它
         * 写回 `setClipPlaybackRate`（必须是 clip 级）。而裁切的源位移必须按后端消费
         * 数学的组合速率折算（`state.rs:121` 的 `span = length × clip.playback_rate`，
         * 后端 `playback_rate` 即组合值）—— 用 clip 级倍率会在 take 速率 ≠ 1 时算错。
         */
        baseConsumeRate: number;
        /**
         * Loop 回绕周期 D（秒）；`0` = 未知。仅用于把 Loop 的相位锚点环绕到
         * `[0, D)`（见 `trimSourceWindow`）。
         */
        mediaDurationSec: number;
        baseFadeInSec: number;
        baseFadeOutSec: number;
        baseSnapOffsetSec: number;
        /**
         * 本次边缘手势作用于的全部 clip（多选集合 + 编组展开）。
         *
         * 旧实现 `useEditDrag` 的 `supportsGroupExpansion` 对 trim 与 stretch 都
         * 展开编组（只排除 fade / gain）；裁切按**锚点位移**逐 clip 换算，拉伸则
         * 走下方的 `stretchGroup` 整组等比缩放。
         */
        participants: KernelEditParticipant[];
        /**
         * 组拉伸：多选 + 编组展开后的整组缩放状态（不满足条件时为 null）。
         *
         * 旧实现用 `buildStretchGroupState` 判定「锚点是否位于选区边界」：只有锚点
         * 是选区最左（拖左缘）/ 最右（拖右缘）的成员时才做整组等比缩放，否则退化
         * 为单 clip 拉伸——判定与几何都在纯函数里，内核不重复实现。
         */
        stretchGroup: StretchGroupState | null;
        /** 组拉伸：各成员按下时的吸附偏移（随长度比例缩放；提交与回滚共用）。 */
        stretchBaseSnapOffsetById: Map<string, number>;
        /** 各参与者的按下时几何（裁切按**锚点位移**逐 clip 换算）。 */
        baseById: Map<
            string,
            {
                startSec: number;
                lengthSec: number;
                sourceStartSec: number;
                sourceEndSec: number;
                /**
                 * 该 clip 的**组合**播放速率（`clipPlaybackRate × take.playbackRate`）：
                 * 源位移 = 时间轴位移 × 速率。与后端消费数学（`state.rs:121` 的
                 * `span = length × clip.playback_rate`）同一口径 —— 用 clip 级倍率会在
                 * take 速率 ≠ 1 时算错源位移。
                 */
                playbackRate: number;
                /** 倒放：时间轴左/右端与源字段的对应关系是镜像的（见 `trimSourceWindow`）。 */
                reversed: boolean;
                /** Loop：源字段是回绕锚点；右缘只改长度、左缘改锚点（见 `trimSourceWindow`）。 */
                loopEnabled: boolean;
                /** Loop 回绕周期 D（秒）；0 = 未知（见 `trimSourceWindow`）。 */
                mediaDurationSec: number;
                /** 按下时的吸附偏移（秒）：左缘拖拽要保持它的**绝对位置**不变。 */
                snapOffsetSec: number;
            }
        >;
        /**
         * 自动交叉淡化：受影响 clip（= 参与者）与编辑前每侧重叠关系、可调整侧。
         *
         * 与旧实现 `useEditDrag` 同源：`initialCrossfadeSides` 记录编辑前的重叠关系
         * （用于「拖开时只清自动、保留手动 fade」），`editSides` 限定本次编辑**允许
         * 自动调整**的侧（裁切左缘只动 fadeIn，右缘只动 fadeOut）——否则裁切左缘会
         * 顺带改掉右缘与邻居的交叉淡化。
         */
        xfadeClipIds: string[];
        initialCrossfadeSides: ReturnType<typeof computeInitialCrossfadeSides>;
        editSides: Record<string, { fadeIn: boolean; fadeOut: boolean }>;
        /**
         * 波纹跟随集（按下时快照）。
         *
         * 旧实现的实时波纹覆盖 trim 与 stretch（由区域右缘净位移驱动），内核原先只在
         * **拖拽移动**时做波纹——裁切 / 拉伸期间后续 clip 不跟随，松手才跳过去。
         * 快照规则与旧实现同源：原点 = 参与者最早起点、轨道集 = 参与者所在轨道。
         */
        rippleFollowers: RippleFollowerMap;
        /**
         * 循环节 / 内容边界吸附的按下时快照（`toBoundarySnapClip`）。
         *
         * 裁切会逐帧改写源窗口，而同余式必须基于**按下时**的源窗口 / 循环状态
         * ——与旧实现 `useEditDrag` 用 `drag.baseByClipId` 同一约定。
         */
        boundaryAnchor: ReturnType<typeof toBoundarySnapClip>;
    } | null>(null);

    /**
     * 内核裁切 / 拉伸的波纹实时预览。
     *
     * 流程：按当前（乐观）几何算出**编辑区域右缘净位移** → 把跟随集平移到
     * 「初始位置 + 位移」。
     *
     * 特殊说明：
     * - 位移取区域右缘而不是锚点右缘：与后端区域化波纹一致，多成员编辑时以整个
     *   区域的最右缘为准（`computeRegionRightEdgeDelta` 是共享纯函数）；
     * - 位移带符号，向左收拢时为负——跟随集必须跟着向左（对 0 取 max 会吞掉负位移）；
     * - 用 `store.getState().session`（同步新鲜）：react-redux 的 batch 会延迟订阅
     *   回调，batch 内 `sessionRef` 仍是上一帧位置。
     *
     * @param origin 本次边缘手势的按下时快照（含 `rippleFollowers` 与参与者几何）。
     * @returns 无返回值。
     */
    const applyKernelTrimRipplePreview = React.useCallback(
        (origin: NonNullable<typeof kernelTrimOriginRef.current>) => {
            // `origin.baseById` 是 Map（按下时按参与者建），纯函数要 Record——
            // 这里就地转换，避免为"一次读取"再存一份平行结构。
            const baseById: Record<string, { startSec: number; lengthSec: number }> = {};
            for (const [clipId, base] of origin.baseById) {
                baseById[clipId] = { startSec: base.startSec, lengthSec: base.lengthSec };
            }
            const rippleRightDelta = computeRegionRightEdgeDelta({
                clipIds: origin.participants.map((item) => item.clipId),
                baseById,
                clips: store.getState().session.clips,
            });
            // 位移为 0 时也要执行：它等价于"把跟随集恢复到初始位置"，正是取消 /
            // 拖回原位所需要的（旧实现同样无条件调用）。
            applyRippleFollowerShift(dispatch, origin.rippleFollowers, rippleRightDelta);
        },
        [dispatch],
    );

    /**
     * 内核 trim 预览：写乐观几何。
     *
     * 源区间必须同步改：左边缘 trim 改 `sourceStartSec`、右边缘改 `sourceEndSec`——
     * 只改 `lengthSec` 会让音频内容被拉伸（波形与音频对不上），而 trim 的语义是
     * **裁切**（内容不滑动）。
     */
    const handleKernelTrimPreview = React.useCallback(
        (args: {
            clipId: string;
            edge: "left" | "right";
            startSec: number;
            lengthSec: number;
            deltaSec: number;
            modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
        }) => {
            if (kernelTrimOriginRef.current?.clipId !== args.clipId) {
                const session = sessionRef.current;
                const clip = session.clips.find((item) => item.id === args.clipId);
                if (clip === undefined) return;
                // 参与集合：裁切与拉伸都是「多选 + 编组展开」（旧实现
                // supportsGroupExpansion 覆盖 trim / stretch，只排除 fade / gain）。
                const stretchMode = isModifierActive(stretchKbRef.current, args.modifiers);
                const participants = resolveKernelEditParticipants({
                    anchorClipId: clip.id,
                    multiSelectedClipIds,
                    clips: session.clips,
                    trackIds: session.tracks.map((track) => track.id),
                    ignoreGrouping: session.ignoreGrouping,
                    disabledGroupIds: session.disabledGroupIds,
                    expandGroups: true,
                });
                // 组拉伸：锚点位于选区边界时才成立（纯函数判定，含「选区不足两个
                // 成员 → null」）；不成立时退化为单 clip 拉伸。
                const stretchGroup = stretchMode
                    ? buildStretchGroupState({
                          clips: session.clips,
                          selectedClipIds: participants.map((item) => item.clipId),
                          anchorClipId: clip.id,
                          edge: args.edge === "left" ? "stretch_left" : "stretch_right",
                      })
                    : null;
                const stretchBaseSnapOffsetById = new Map<string, number>();
                if (stretchGroup !== null) {
                    for (const clipId of stretchGroup.clipIds) {
                        const item = session.clips.find((candidate) => candidate.id === clipId);
                        stretchBaseSnapOffsetById.set(
                            clipId,
                            Math.max(0, Number(item?.snapOffsetSec) || 0),
                        );
                    }
                }
                const baseById = new Map<
                    string,
                    {
                        startSec: number;
                        lengthSec: number;
                        sourceStartSec: number;
                        sourceEndSec: number;
                        playbackRate: number;
                        reversed: boolean;
                        loopEnabled: boolean;
                        mediaDurationSec: number;
                        snapOffsetSec: number;
                    }
                >();
                for (const participant of participants) {
                    const item = session.clips.find(
                        (candidate) => candidate.id === participant.clipId,
                    );
                    if (item === undefined) continue;
                    baseById.set(participant.clipId, {
                        startSec: Number(item.startSec) || 0,
                        lengthSec: Math.max(0, Number(item.lengthSec) || 0),
                        sourceStartSec: Number(item.sourceStartSec ?? 0) || 0,
                        sourceEndSec: Number(item.sourceEndSec ?? 0) || 0,
                        // 组合速率（clip × take）：与后端消费数学同一口径。
                        playbackRate: Number(item.playbackRate ?? 1) || 1,
                        reversed: item.reversed === true,
                        loopEnabled: item.loopEnabled === true,
                        mediaDurationSec: resolveClipContentDurationSec(item) ?? 0,
                        snapOffsetSec: Math.max(0, Number(item.snapOffsetSec) || 0),
                    });
                }
                // 自动交叉淡化：受影响集合 = 参与者；可调整侧按拖拽的边缘决定。
                const xfadeClipIds = participants.map((participant) => participant.clipId);
                const editSides: Record<string, { fadeIn: boolean; fadeOut: boolean }> = {};
                for (const id of xfadeClipIds) {
                    editSides[id] =
                        args.edge === "left"
                            ? { fadeIn: true, fadeOut: false }
                            : { fadeIn: false, fadeOut: true };
                }
                kernelTrimOriginRef.current = {
                    clipId: clip.id,
                    startSec: clip.startSec,
                    lengthSec: clip.lengthSec,
                    trackId: clip.trackId,
                    sourceStartSec: clip.sourceStartSec,
                    sourceEndSec: clip.sourceEndSec,
                    xfadeClipIds,
                    initialCrossfadeSides: computeInitialCrossfadeSides(
                        session.clips,
                        xfadeClipIds,
                    ),
                    editSides,
                    // 波纹跟随集：与旧实现 `useEditDrag` 同一套快照规则（原点取
                    // 参与者最早起点、作用轨道取参与者所在轨道）。
                    rippleFollowers: buildRippleFollowers(
                        session.clips,
                        new Set(xfadeClipIds),
                        Math.min(...participants.map((item) => item.startSec)),
                        session.rippleMode,
                        new Set(participants.map((item) => item.trackId)),
                    ),
                    // Alt 按住 = 拉伸（与旧实现 `modifier.clipStretch` 同源）。
                    mode: stretchMode ? "stretch" : "trim",
                    basePlaybackRate: Number(clip.clipPlaybackRate ?? 1) || 1,
                    baseConsumeRate: Number(clip.playbackRate ?? 1) || 1,
                    reversed: clip.reversed === true,
                    loopEnabled: clip.loopEnabled === true,
                    mediaDurationSec: resolveClipContentDurationSec(clip) ?? 0,
                    baseFadeInSec: Number(clip.fadeInSec) || 0,
                    baseFadeOutSec: Number(clip.fadeOutSec) || 0,
                    baseSnapOffsetSec: Math.max(0, Number(clip.snapOffsetSec) || 0),
                    participants,
                    stretchGroup,
                    stretchBaseSnapOffsetById,
                    baseById,
                    // 循环节 / 内容边界吸附的按下时快照（与 slip 手势同一套）：
                    // 源窗口在裁切过程中逐帧变化，用当前值算同余式会漂移。
                    boundaryAnchor: toBoundarySnapClip(clip),
                };
                // 首个真实位移帧 = 手势开始：上交互锁（见 helper 说明）。
                beginKernelGestureInteraction();
            }
            const origin = kernelTrimOriginRef.current;
            if (origin === null) return;

            // 免吸附修饰键（默认 Shift）临时取反吸附总开关（与旧实现同一函数）。
            const snapActive = computeEffectiveSnap(
                s.snapEnabled,
                isModifierActive(noSnapKb, args.modifiers),
            );

            if (origin.mode === "stretch") {
                // Alt + 拖边缘 = **拉伸**：对侧边缘固定、改播放速率（内容不被裁掉）。
                // 被吸附对象是**正在拖的那条边**（左缘 / 右缘），与裁切一致；
                // 几何换算复用与旧实现共用的纯函数（单 clip 用 `computeClipStretch`、
                // 组拉伸用 `computeStretchGroupUpdate`，都含速率钳制与比例缩放）。
                const rawEdgeSec =
                    args.edge === "left" ? args.startSec : args.startSec + args.lengthSec;
                const edgeSec = snapActive
                    ? snapTimelineDetailed(rawEdgeSec, "clip", {
                          originSec:
                              args.edge === "left"
                                  ? origin.startSec
                                  : origin.startSec + origin.lengthSec,
                          anchorTrackId: origin.trackId,
                          // 排除**整组**（旧实现排除 `selectedClipIds`）：组内其他
                          // 成员随本次拉伸一起移动，不应成为自己的吸附目标。
                          excludeClipIds: new Set(origin.stretchGroup?.clipIds ?? [args.clipId]),
                          highlight: {
                              sources: [{ trackId: origin.trackId, clipId: args.clipId }],
                          },
                      }).sec
                    : rawEdgeSec;
                if (!snapActive) clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                if (origin.stretchGroup !== null) {
                    // 组拉伸：整组等比缩放（对侧整组边界固定，成员按相对位置缩放，
                    // 各自速率反算并钳制、淡变与 SnapOffset 按比例缩放）。
                    const group = origin.stretchGroup;
                    const update = computeStretchGroupUpdate({
                        group,
                        edge: args.edge === "left" ? "stretch_left" : "stretch_right",
                        pointerSec: edgeSec,
                    });
                    batch(() => {
                        for (const clipId of group.clipIds) {
                            const next = update.byId[clipId];
                            if (next === undefined) continue;
                            dispatch(moveClipStart({ clipId, startSec: next.startSec }));
                            dispatch(setClipLength({ clipId, lengthSec: next.lengthSec }));
                            dispatch(
                                setClipPlaybackRate({
                                    clipId,
                                    clipPlaybackRate: next.clipPlaybackRate,
                                }),
                            );
                            dispatch(
                                setClipFades({
                                    clipId,
                                    fadeInSec: next.fadeInSec,
                                    fadeOutSec: next.fadeOutSec,
                                }),
                            );
                            const initial = group.initialById[clipId];
                            if (initial !== undefined) {
                                // SnapOffset 随成员长度按「总比例 × 基准偏移」缩放
                                //（基准 = 按下时快照，禁止逐帧复合）。
                                const ratio = next.lengthSec / Math.max(1e-6, initial.lengthSec);
                                dispatch(
                                    setClipSnapOffset({
                                        clipId,
                                        snapOffsetSec: scaleSnapOffsetForStretch(
                                            origin.stretchBaseSnapOffsetById.get(clipId),
                                            ratio,
                                            next.lengthSec,
                                        ),
                                    }),
                                );
                            }
                        }
                    });
                    // 波纹实时预览（组拉伸改变区域右缘 → 跟随集同步平移）。
                    applyKernelTrimRipplePreview(origin);
                    if (s.autoCrossfadeEnabled) {
                        previewAutoCrossfade(
                            store.getState().session,
                            origin.xfadeClipIds,
                            dispatch,
                            origin.initialCrossfadeSides,
                            origin.editSides,
                        );
                    }
                    return;
                }
                const result = computeClipStretch({
                    edge: args.edge === "left" ? "stretch_left" : "stretch_right",
                    pointerSec: edgeSec,
                    baseStartSec: origin.startSec,
                    baseLengthSec: origin.lengthSec,
                    basePlaybackRate: origin.basePlaybackRate,
                    baseFadeInSec: origin.baseFadeInSec,
                    baseFadeOutSec: origin.baseFadeOutSec,
                    baseSnapOffsetSec: origin.baseSnapOffsetSec,
                    // 最小长度交由 `computeClipStretch` 的内部极小值护栏（`MIN_SPAN_SEC`）
                    // 兜底——旧实现 `useEditDrag` 的拉伸同样用 `minLen = 0.0`。
                });
                batch(() => {
                    dispatch(moveClipStart({ clipId: args.clipId, startSec: result.startSec }));
                    dispatch(setClipLength({ clipId: args.clipId, lengthSec: result.lengthSec }));
                    dispatch(
                        setClipPlaybackRate({
                            clipId: args.clipId,
                            clipPlaybackRate: result.clipPlaybackRate,
                        }),
                    );
                    dispatch(
                        setClipFades({
                            clipId: args.clipId,
                            fadeInSec: result.fadeInSec,
                            fadeOutSec: result.fadeOutSec,
                        }),
                    );
                    dispatch(
                        setClipSnapOffset({
                            clipId: args.clipId,
                            snapOffsetSec: result.snapOffsetSec,
                        }),
                    );
                });
                // 波纹实时预览（单 clip 拉伸改变区域右缘 → 跟随集同步平移）。
                applyKernelTrimRipplePreview(origin);
                // 自动交叉淡化实时预览（拉伸改变重叠 → 自动 fade 随之变化）。
                if (s.autoCrossfadeEnabled) {
                    previewAutoCrossfade(
                        store.getState().session,
                        origin.xfadeClipIds,
                        dispatch,
                        origin.initialCrossfadeSides,
                        origin.editSides,
                    );
                }
                return;
            }

            // 吸附：左边缘吸**起点**、右边缘吸**右端**——两者吸附的对象不同，
            // 统一吸起点会让右边缘 trim 落在错误的位置。
            let nextStart = args.startSec;
            let nextLength = args.lengthSec;
            let deltaSec = args.deltaSec;
            if (snapActive) {
                const snapArgs = {
                    // 被吸附对象**只有正在拖的那一条边**。旧实现 `useEditDrag` 的
                    // trim 分支不传 `moveLengthSec` / `moveSnapOffsetSec`：多源候选
                    // 会把「对侧边缘 / clip 自身的吸附偏移点」也当成吸附目标，
                    // 落点因此与旧实现不同（右缘还会多出 `右缘 + 长度` 这种伪候选）。
                    originSec:
                        args.edge === "left" ? origin.startSec : origin.startSec + origin.lengthSec,
                    anchorTrackId: origin.trackId,
                    // 排除**全部参与者**（旧实现排除 `selectedClipIds`）：整组一起
                    // 裁切时，组内其他成员不应成为自己的吸附目标。
                    excludeClipIds: new Set(origin.participants.map((item) => item.clipId)),
                    // `highlight` 必须传：吸附高亮的发布 / 清除由
                    // snapTimelineDetailed 按此选项统一处理，不传则 trim 有吸附
                    // 但没有任何视觉反馈。
                    highlight: {
                        sources: [{ trackId: origin.trackId, clipId: args.clipId }],
                    },
                };
                if (args.edge === "left") {
                    nextStart = snapTimelineDetailed(args.startSec, "clip", snapArgs).sec;
                    nextLength = origin.startSec + origin.lengthSec - nextStart;
                    deltaSec = nextStart - origin.startSec;
                } else {
                    const snappedRight = snapTimelineDetailed(
                        args.startSec + args.lengthSec,
                        "clip",
                        snapArgs,
                    ).sec;
                    nextLength = Math.max(0, snappedRight - args.startSec);
                    deltaSec = nextLength - origin.lengthSec;
                }

                // ── 循环节 / 内容边界吸附（命中时**覆盖**常规网格吸附）──────
                // 与旧实现 `useEditDrag` 同一套：属于常规吸附体系（受总开关与
                // 「拖动时切换吸附」修饰键的 XOR 控制），并且需要在吸附设置里启用
                // 「Clip 边缘吸附到源素材首尾」。被吸附对象是**移动边缘相对 clip
                // 基准起点的时间线偏移**——该同余式对左右两条边同样成立，故统一处理。
                const timelineSnap = sessionRef.current.timelineSnap;
                if (timelineSnap.snapClipsToSourceMedia && timelineSnap.snapDistancePx > 0) {
                    const edgeSec = args.edge === "left" ? nextStart : args.startSec + nextLength;
                    const rawOffset = edgeSec - origin.startSec;
                    const snappedOffset = nearestBoundarySnapOffsetSec(
                        origin.boundaryAnchor,
                        "edge",
                        rawOffset,
                    );
                    if (
                        snappedOffset != null &&
                        Math.abs(snappedOffset - rawOffset) <=
                            loopSnapThresholdSec(timelineSnap.snapDistancePx, pxPerSecRef.current) +
                                1e-12
                    ) {
                        const snappedEdgeSec = origin.startSec + snappedOffset;
                        if (args.edge === "left") {
                            nextStart = snappedEdgeSec;
                            nextLength = origin.startSec + origin.lengthSec - nextStart;
                            deltaSec = nextStart - origin.startSec;
                        } else {
                            nextLength = Math.max(0, snappedEdgeSec - args.startSec);
                            deltaSec = nextLength - origin.lengthSec;
                        }
                        // 循环节命中：以「循环节」专用高亮**覆盖**常规吸附高亮
                        // （同组发布即整组替换）。目标（源媒体边界的投影）与被吸附
                        // 边重合于同一 x，行内双亮条强调。
                        publishSnapHighlights(SNAP_HIGHLIGHT_GROUP, [
                            buildLoopBoundaryHighlightEntry({
                                secs: [snappedEdgeSec],
                                trackId: origin.trackId,
                                clipId: args.clipId,
                            }),
                        ]);
                    }
                }
            } else {
                clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            }

            batch(() => {
                dispatch(moveClipStart({ clipId: args.clipId, startSec: nextStart }));
                dispatch(setClipLength({ clipId: args.clipId, lengthSec: nextLength }));
                // 源窗口换算收口到 `resolveTrimSourceWindow`：它按 `reversed` 镜像方向
                // （时间轴右端在倒放时对应 `sourceStartSec`），并对 Loop 返回 null
                // （源字段是回绕锚点，裁切只改长度）。此前这里手写了"右缘动
                // sourceEndSec"的分支且没有倒放/Loop 分支 —— 倒放 Clip 拖右缘改到
                // 左端、拖左缘改到右端，固定的一侧整个颠倒。
                const anchorWindow = resolveTrimSourceWindow({
                    edge: args.edge,
                    reversed: origin.reversed,
                    loopEnabled: origin.loopEnabled,
                    mediaDurationSec: origin.mediaDurationSec,
                    deltaSec,
                    rate: origin.baseConsumeRate,
                    sourceStartSec: origin.sourceStartSec,
                    sourceEndSec: origin.sourceEndSec,
                });
                if (anchorWindow !== null) {
                    dispatch(
                        setClipSourceRange({
                            clipId: args.clipId,
                            sourceStartSec: anchorWindow.sourceStartSec,
                            sourceEndSec: anchorWindow.sourceEndSec,
                        }),
                    );
                }
                // 吸附偏移：它的语义是"素材内的一个点"（手柄绝对位置 =
                // clipStart + offset），因此拖**左缘**改变 clipStart 时必须反向调整
                // 相对值，该点才会留在同一处素材上。此前裁切完全不动它，于是它作为
                // "相对起点的偏移"被保留、手柄跟着起点平移到了另一段素材。
                dispatch(
                    setClipSnapOffset({
                        clipId: args.clipId,
                        snapOffsetSec: resolveTrimSnapOffset({
                            edge: args.edge,
                            deltaSec,
                            snapOffsetSec: origin.baseSnapOffsetSec,
                            newLengthSec: nextLength,
                        }),
                    }),
                );
                // 多选批量：其余参与者按**锚点位移**（deltaSec）逐 clip 换算——与旧实现
                // `useEditDrag` 同源（以锚点的实际位移为基准，不是各自重新吸附）。
                for (const participant of origin.participants) {
                    if (participant.clipId === args.clipId) continue;
                    const base = origin.baseById.get(participant.clipId);
                    if (base === undefined) continue;
                    if (args.edge === "left") {
                        dispatch(
                            moveClipStart({
                                clipId: participant.clipId,
                                startSec: base.startSec + deltaSec,
                            }),
                        );
                        dispatch(
                            setClipLength({
                                clipId: participant.clipId,
                                lengthSec: Math.max(0, base.lengthSec - deltaSec),
                            }),
                        );
                    } else {
                        dispatch(
                            setClipLength({
                                clipId: participant.clipId,
                                lengthSec: Math.max(0, base.lengthSec + deltaSec),
                            }),
                        );
                    }
                    // 每个参与者按**自己**的倒放 / Loop / 组合速率换算源窗口
                    // （混合多选里正放与倒放可以同时存在）。
                    const window = resolveTrimSourceWindow({
                        edge: args.edge,
                        reversed: base.reversed,
                        loopEnabled: base.loopEnabled,
                        mediaDurationSec: base.mediaDurationSec,
                        deltaSec,
                        rate: base.playbackRate,
                        sourceStartSec: base.sourceStartSec,
                        sourceEndSec: base.sourceEndSec,
                    });
                    if (window !== null) {
                        dispatch(
                            setClipSourceRange({
                                clipId: participant.clipId,
                                sourceStartSec: window.sourceStartSec,
                                sourceEndSec: window.sourceEndSec,
                            }),
                        );
                    }
                    // 吸附偏移同锚点：按各自的按下时偏移保持"素材内的点"不动。
                    dispatch(
                        setClipSnapOffset({
                            clipId: participant.clipId,
                            snapOffsetSec: resolveTrimSnapOffset({
                                edge: args.edge,
                                deltaSec,
                                snapOffsetSec: base.snapOffsetSec,
                                newLengthSec:
                                    args.edge === "left"
                                        ? Math.max(0, base.lengthSec - deltaSec)
                                        : Math.max(0, base.lengthSec + deltaSec),
                            }),
                        }),
                    );
                }
            });
            // 波纹（自动跟进）实时预览：以编辑区域**右缘净位移**为准驱动跟随集，
            // 与旧实现 `useEditDrag` 同源（同一个纯函数）。必须在三种模式（拉伸 /
            // 组拉伸 / 裁切）之后统一执行——只在某一个分支里做会让其余分支漏掉波纹。
            // 用 `store.getState().session`（同步新鲜）而不是 `sessionRef`：batch 内
            // ref 仍停在上一帧位置，会让位移滞后一帧。
            applyKernelTrimRipplePreview(origin);
            // 自动交叉淡化实时预览：裁切改变重叠 → 按当前乐观位置重算自动 fade。
            // `editSides` 限定只动本次拖拽的那一侧（裁切左缘不得改右缘的交叉淡化）。
            if (s.autoCrossfadeEnabled) {
                previewAutoCrossfade(
                    store.getState().session,
                    origin.xfadeClipIds,
                    dispatch,
                    origin.initialCrossfadeSides,
                    origin.editSides,
                );
            }
        },
        [
            applyKernelTrimRipplePreview,
            beginKernelGestureInteraction,
            dispatch,
            // 阈值换算读实时值：创建时快照会在缩放后把吸附容差算错（px→sec 的
            // 比例尺过期），ref 读取既新鲜又不让回调随缩放重建。
            pxPerSecRef,
            sessionRef,
            s.autoCrossfadeEnabled,
            s.snapEnabled,
            noSnapKb,
            snapTimelineDetailed,
            stretchKbRef,
            // 参与集合在按下时解析，必须读到**最新**的多选集合：漏这个依赖会让
            // 回调闭包停在挂载时的空选择上（表现为"框选多个后仍只裁切一个"）。
            multiSelectedClipIds,
        ],
    );

    /** 内核 trim / stretch 收尾：提交或回滚。 */
    /**
     * 内核吸附偏移手势是否已真正开始。
     *
     * 预览回调每帧触发，但 `beginInteraction` / `checkpointHistory` /
     * `beginSnapGesture` 只能做一次——用这个 ref 去重，同时给收尾一个
     * 「零位移单击」的判据（旧实现用 `drag.checkpointed`）。
     */
    const kernelSnapOffsetActiveRef = React.useRef(false);

    /**
     * 内核吸附偏移手势按下时的偏移（秒），供取消路径回滚。
     *
     * 预览逐帧改写 `snapOffsetSec` 乐观值，取消（Esc / pointercancel）必须还原到
     * 按下时的值——否则 Redux 停在中途位置，与后端分叉（与其它内核手势的取消
     * 路径同一约定）。在首次真实位移时快照：那一刻的 Redux 值就是按下时的值。
     */
    const kernelSnapOffsetBaseRef = React.useRef<{ clipId: string; snapOffsetSec: number } | null>(
        null,
    );

    /**
     * 内核吸附偏移预览：与旧实现 `useSnapOffsetDrag` 同源。
     *
     * 被吸附对象是**手柄的绝对时间线位置**（`clipStart + offset`）——不是 clip 起点；
     * 高亮发布为该 clip 所在行的亮条。落库前把偏移钳制到 `[0, clip 长度]`。
     */
    const handleKernelSnapOffsetPreview = React.useCallback(
        (args: {
            clipId: string;
            rawOffsetSec: number;
            modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
        }) => {
            const clip = sessionRef.current.clips.find((item) => item.id === args.clipId);
            if (clip === undefined) return;
            const clipStart = Number(clip.startSec) || 0;
            const clipLen = Math.max(0, Number(clip.lengthSec) || 0);
            if (!kernelSnapOffsetActiveRef.current) {
                // 首次真实位移：交互锁 + undo 检查点 + 吸附手势一起开。
                kernelSnapOffsetActiveRef.current = true;
                // 按下时的偏移快照（此刻尚未被本次手势改写）——取消路径回滚用。
                kernelSnapOffsetBaseRef.current = {
                    clipId: args.clipId,
                    snapOffsetSec: Math.max(0, Number(clip.snapOffsetSec) || 0),
                };
                beginKernelGestureInteraction();
                dispatch(checkpointHistory());
                beginSnapGesture();
            }
            // 手柄必须留在 Clip 内：**先把查询位置钳进** [clipStart, clipStart+clipLen]，
            // 再把吸附候选限制在同一区间（`candidateRangeSec`）。
            // 【为什么两件都要做】只钳位置不滤候选 ⇒ 最近的候选可能落在 Clip 外，
            // 吸附把落点算到外面、调用方再钳回边界 —— 高亮线画在 Clip 外而手柄停在
            // 边界，看起来像"范围外也产生吸附"（用户报告的现象）。
            const rawAbs = clipStart + args.rawOffsetSec;
            const clampedAbs = Math.min(Math.max(rawAbs, clipStart), clipStart + clipLen);
            // 免吸附修饰键（默认 Shift）临时取反吸附总开关（与旧实现同一函数）。
            const snapActive = computeEffectiveSnap(
                s.snapEnabled,
                isModifierActive(noSnapKb, args.modifiers),
            );
            const nextAbs = snapActive
                ? snapTimelineDetailed(clampedAbs, "clip", {
                      // 拖动起点：必须用**按下时**的偏移（`kernelSnapOffsetBaseRef`），
                      // 不能用被本手势逐帧改写的 `clip.snapOffsetSec` —— 后者会让
                      // "相对网格吸附"每帧以自己上一步的落点为基准，逐步漂移。
                      originSec:
                          clipStart +
                          (kernelSnapOffsetBaseRef.current?.snapOffsetSec ??
                              (Number(clip.snapOffsetSec) || 0)),
                      anchorTrackId: clip.trackId,
                      excludeClipIds: new Set([args.clipId]),
                      candidateRangeSec: { lo: clipStart, hi: clipStart + clipLen },
                      highlight: {
                          sources: [{ trackId: clip.trackId, clipId: args.clipId }],
                      },
                  }).sec
                : clampedAbs;
            if (!snapActive) clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            const nextOffset = Math.min(Math.max(nextAbs - clipStart, 0), clipLen);
            dispatch(
                setClipSnapOffset({
                    clipId: args.clipId,
                    snapOffsetSec: nextOffset,
                }),
            );
            // 拖动中的实时浮标（与增益旋钮 `handleKernelGainDragPreview` 同构）：
            // 两行都追加**本次拖拽的位移**。增量基准取按下时的偏移快照，不能用
            // 逐帧改写的当前值（那会退化成"每帧归零"）。内容逐帧更新，浮标锚点
            // 仍是内核容器，AppTooltip 自己跟随指针。
            const anchor =
                typeof document === "undefined"
                    ? null
                    : (document.querySelector(
                          "[data-hs-fade-tooltip-anchor]",
                      ) as HTMLElement | null);
            if (anchor !== null) {
                const baseOffset = kernelSnapOffsetBaseRef.current?.snapOffsetSec ?? nextOffset;
                publishFadeRichTooltip(
                    anchor,
                    buildSnapOffsetInfoText({
                        offsetSec: nextOffset,
                        positionSec: clipStart + nextOffset,
                        deltaSec: nextOffset - baseOffset,
                        formatCtx: timeValueFormatCtx,
                        t: tVars as TimeValueLabelLookup,
                    }),
                );
            }
        },
        [
            beginKernelGestureInteraction,
            dispatch,
            noSnapKb,
            s.snapEnabled,
            sessionRef,
            snapTimelineDetailed,
            tVars,
            timeValueFormatCtx,
        ],
    );

    /**
     * 内核吸附偏移收尾。
     *
     * 零位移单击**不写后端**（与旧实现一致：单击不产生 undo 步）；真实拖拽则
     * 一次性持久化当前乐观值（`checkpoint: true` → 整次拖拽恰好一个撤销步）。
     */
    const handleKernelSnapOffsetCommit = React.useCallback(
        (args: { clipId: string; cancelled: boolean; changed: boolean }) => {
            const wasActive = kernelSnapOffsetActiveRef.current;
            const base = kernelSnapOffsetBaseRef.current;
            kernelSnapOffsetActiveRef.current = false;
            kernelSnapOffsetBaseRef.current = null;
            clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            // 手势从未真正开始（零位移单击）：begin* 都没调用过，不能 end*。
            if (!wasActive) return;
            endSnapGesture();
            if (args.cancelled || !args.changed) {
                // 取消：乐观值必须还原到按下时的偏移，否则 Redux 与后端分叉
                // （与其它内核手势的取消路径同一约定）。
                if (args.cancelled && base !== null && base.clipId === args.clipId) {
                    dispatch(
                        setClipSnapOffset({
                            clipId: args.clipId,
                            snapOffsetSec: base.snapOffsetSec,
                        }),
                    );
                }
                endKernelGestureInteraction();
                return;
            }
            const clip = sessionRef.current.clips.find((item) => item.id === args.clipId);
            const clipLen = Math.max(0, Number(clip?.lengthSec) || 0);
            void dispatch(
                setClipStateRemote({
                    clipId: args.clipId,
                    snapOffsetSec: Math.min(Math.max(Number(clip?.snapOffsetSec) || 0, 0), clipLen),
                    checkpoint: true,
                }),
            )
                .unwrap()
                .catch(() => {
                    // 失败不产生 unhandled rejection；交互锁仍需释放。
                })
                .finally(endKernelGestureInteraction);
        },
        [dispatch, endKernelGestureInteraction, sessionRef],
    );

    /**
     * 拉伸提交的公共收尾（单 clip 拉伸与组拉伸共用）。
     *
     * 【为什么必须共用】此前两条路径各写一遍，于是分叉：组路径在改写参数线后补了
     * `bumpParamsEpoch()`，单路径漏了 —— 单 clip 拉伸松手后波形停在「新几何 × 旧曲线」
     * 且只有再做一次别的操作才恢复（用户报告的现象）。收口到一处即从结构上杜绝再分叉。
     *
     * 【取数闸门】调用方在**落库派发之前** `holdLoudnessFetch()`（见
     * `loudnessFetchGate`）：几何落库的 fulfilled handler 会 `applyTimelineState`
     * 递增 paramsEpoch，触发一次「新几何 × 旧曲线」的中间态取数，必须压住。
     * 本函数只负责改写曲线；闸门的**释放与补取数**由调用方的 `finally` 统一执行
     * （保证持久化失败 / 改写抛错等任何路径都不会让闸门悬挂）。
     */
    const finishStretchWithParamLines = React.useCallback(
        async (opts: {
            /** 需要二次写回的速率（后端可能按自身计算覆盖速率，前端计算值才权威）。 */
            rateWritebacks: ReadonlyArray<{ clipId: string; clipPlaybackRate: number }>;
            /** 锁定参数线开启时改写曲线；`null` = 本次不搬曲线。 */
            rewriteParamLines: (() => Promise<unknown>) | null;
            xfadeClipIds: string[];
            initialCrossfadeSides: ReturnType<typeof computeInitialCrossfadeSides>;
            editSides: Record<string, { fadeIn: boolean; fadeOut: boolean }>;
        }): Promise<void> => {
            for (const writeback of opts.rateWritebacks) {
                dispatch(setClipPlaybackRate(writeback));
            }
            if (opts.rewriteParamLines !== null) {
                await opts.rewriteParamLines();
            }
            // 自动交叉淡化：按落库后的重叠关系写回自动 fade（开关关闭时只清理
            // 「已脱离重叠」的自动值）。
            const latest = sessionRef.current;
            if (s.autoCrossfadeEnabled) {
                await applyAutoCrossfade(latest, opts.xfadeClipIds, dispatch, {
                    affectedSides: opts.initialCrossfadeSides,
                    editSides: opts.editSides,
                });
            } else {
                await applyDetachedAutoCrossfadeClears(
                    latest,
                    opts.xfadeClipIds,
                    dispatch,
                    opts.initialCrossfadeSides,
                    opts.editSides,
                );
            }
        },
        [dispatch, sessionRef, s.autoCrossfadeEnabled],
    );

    /**
     * 拉伸提交链的收尾：打开取数闸门并补一次**权威取数**，然后释放交互锁。
     *
     * 闸门只有在"本次确实要搬曲线"（`lockParamLinesEnabled`）时才合上，因此释放
     * 也只在那种情形下执行；否则释放是无意义的多余取数。释放后立刻 `bumpParamsEpoch()`
     * —— 这是唯一一次带「新几何 + 改写后曲线」的取数（见 `loudnessFetchGate`）。
     */
    const finishStretchCommitChain = React.useCallback(
        (lockParamLines: boolean): void => {
            if (lockParamLines) {
                releaseLoudnessFetch();
                dispatch(bumpParamsEpoch());
            }
            endKernelGestureInteraction();
        },
        [dispatch, endKernelGestureInteraction],
    );

    const handleKernelTrimCommit = React.useCallback(
        (args: {
            clipId: string;
            edge: "left" | "right";
            startSec: number;
            lengthSec: number;
            cancelled: boolean;
        }) => {
            const origin = kernelTrimOriginRef.current;
            kernelTrimOriginRef.current = null;
            if (origin === null) return;
            // 手势结束：清掉吸附高亮（与拖拽收尾同源）。
            clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            // 波纹跟随集：取消时**必须还原原位**（预览期间已被平移），否则 Redux 会
            // 停在中途位置、与后端分叉。提交时不还原——后端按权威结果写回，
            // 前端保留的乐观位置正好避免"松手回跳再跳过去"。
            if (args.cancelled) {
                applyRippleFollowerShift(dispatch, origin.rippleFollowers, 0);
            }

            if (origin.mode === "stretch") {
                // 拉伸收尾：回滚 / 提交都作用于**五个字段**（起点、长度、速率、
                // 两侧淡变、吸附偏移）——只还原其中一部分会让预览与落库分叉。
                if (origin.stretchGroup !== null) {
                    // ── 组拉伸：整组成员一起回滚 / 提交 ─────────────────────
                    const group = origin.stretchGroup;
                    if (args.cancelled) {
                        batch(() => {
                            for (const clipId of group.clipIds) {
                                const initial = group.initialById[clipId];
                                if (initial === undefined) continue;
                                dispatch(moveClipStart({ clipId, startSec: initial.startSec }));
                                dispatch(setClipLength({ clipId, lengthSec: initial.lengthSec }));
                                dispatch(
                                    setClipPlaybackRate({
                                        clipId,
                                        clipPlaybackRate: initial.clipPlaybackRate,
                                    }),
                                );
                                dispatch(
                                    setClipFades({
                                        clipId,
                                        fadeInSec: initial.fadeInSec,
                                        fadeOutSec: initial.fadeOutSec,
                                    }),
                                );
                                dispatch(
                                    setClipSnapOffset({
                                        clipId,
                                        snapOffsetSec:
                                            origin.stretchBaseSnapOffsetById.get(clipId) ?? 0,
                                    }),
                                );
                            }
                        });
                        // 自动交叉淡化回到按下时的重叠关系（按已还原的几何重算）。
                        if (s.autoCrossfadeEnabled) {
                            previewAutoCrossfade(
                                store.getState().session,
                                origin.xfadeClipIds,
                                dispatch,
                                origin.initialCrossfadeSides,
                                origin.editSides,
                            );
                        }
                        // 取消同样要归还交互锁：手势起手已 beginKernelGestureInteraction()
                        // 增加锁计数，若不释放，锁计数长期 ≥1 会跳过非强制的
                        // applyTimelineState（后端周期快照 / 普通命令响应停止落地）。
                        endKernelGestureInteraction();
                        return;
                    }
                    dispatch(checkpointHistory());
                    // 提交值取 Redux 当前值（预览已写入钳制后的结果）。
                    const groupSession = sessionRef.current;
                    const updates = group.clipIds.flatMap((clipId) => {
                        const clip = groupSession.clips.find((item) => item.id === clipId);
                        if (clip === undefined) return [];
                        return [
                            {
                                clipId,
                                startSec: clip.startSec,
                                lengthSec: clip.lengthSec,
                                clipPlaybackRate: Number(clip.clipPlaybackRate ?? 1) || 1,
                                fadeInSec: Number(clip.fadeInSec) || 0,
                                fadeOutSec: Number(clip.fadeOutSec) || 0,
                                snapOffsetSec: Math.max(0, Number(clip.snapOffsetSec) || 0),
                            },
                        ];
                    });
                    if (updates.length === 0) {
                        endKernelGestureInteraction();
                        return;
                    }
                    // 锁定参数线：本次是否要搬曲线（决定取数闸门是否需要合上）。
                    const lockParamLines = sessionRef.current.lockParamLinesEnabled;
                    // 闸门必须在落库派发**之前**合上（见 `loudnessFetchGate`）。
                    if (lockParamLines) holdLoudnessFetch();
                    const groupPersist = dispatch(setClipsStateBulkRemote({ updates })).unwrap();
                    void (async () => {
                        try {
                            await groupPersist;
                            await finishStretchWithParamLines({
                                // 速率二次写回：后端可能按自身计算覆盖速率，前端计算值
                                // 才是权威（与旧实现 `reapplyRates` 同源，只回写非 1 的成员）。
                                rateWritebacks: updates
                                    .filter((update) => update.clipPlaybackRate !== 1)
                                    .map((update) => ({
                                        clipId: update.clipId,
                                        clipPlaybackRate: update.clipPlaybackRate,
                                    })),
                                // 按**根轨道**聚合成员的时域映射，一次请求完成该轨道的曲线映射
                                //（与旧实现同源）。
                                rewriteParamLines: lockParamLines
                                    ? () => {
                                          const mappingsByRootTrack = new Map<
                                              string,
                                              StretchRangeMapping[]
                                          >();
                                          for (const clipId of group.clipIds) {
                                              const initial = group.initialById[clipId];
                                              const now = sessionRef.current.clips.find(
                                                  (item) => item.id === clipId,
                                              );
                                              if (initial === undefined || now === undefined) {
                                                  continue;
                                              }
                                              const rootTrackId = resolveRootTrackId(
                                                  sessionRef.current.tracks,
                                                  now.trackId,
                                              );
                                              if (
                                                  rootTrackId === null ||
                                                  rootTrackId === undefined
                                              ) {
                                                  continue;
                                              }
                                              const mappings =
                                                  mappingsByRootTrack.get(rootTrackId) ?? [];
                                              mappings.push({
                                                  oldStartSec: initial.startSec,
                                                  oldLengthSec: initial.lengthSec,
                                                  newStartSec: now.startSec,
                                                  newLengthSec: now.lengthSec,
                                              });
                                              mappingsByRootTrack.set(rootTrackId, mappings);
                                          }
                                          return Promise.allSettled(
                                              Array.from(
                                                  mappingsByRootTrack,
                                                  ([trackId, mappings]) =>
                                                      stretchTrackLinkedParams(trackId, mappings),
                                              ),
                                          );
                                      }
                                    : null,
                                xfadeClipIds: origin.xfadeClipIds,
                                initialCrossfadeSides: origin.initialCrossfadeSides,
                                editSides: origin.editSides,
                            });
                        } catch {
                            // 失败不产生 unhandled rejection；交互锁与闸门仍需释放
                            //（闸门由 finishStretchCommitChain 统一释放）。
                        } finally {
                            finishStretchCommitChain(lockParamLines);
                        }
                    })();
                    return;
                }
                if (args.cancelled) {
                    batch(() => {
                        dispatch(
                            moveClipStart({ clipId: origin.clipId, startSec: origin.startSec }),
                        );
                        dispatch(
                            setClipLength({ clipId: origin.clipId, lengthSec: origin.lengthSec }),
                        );
                        dispatch(
                            setClipPlaybackRate({
                                clipId: origin.clipId,
                                clipPlaybackRate: origin.basePlaybackRate,
                            }),
                        );
                        dispatch(
                            setClipFades({
                                clipId: origin.clipId,
                                fadeInSec: origin.baseFadeInSec,
                                fadeOutSec: origin.baseFadeOutSec,
                            }),
                        );
                        dispatch(
                            setClipSnapOffset({
                                clipId: origin.clipId,
                                snapOffsetSec: origin.baseSnapOffsetSec,
                            }),
                        );
                    });
                    endKernelGestureInteraction();
                    return;
                }
                dispatch(checkpointHistory());
                // 提交值取 Redux 当前值（预览已写入钳制后的结果）——用基准 + 位移
                // 重算会绕开速率钳制与长度回算，表现为"预览到上限、落库却超出"。
                const clip = sessionRef.current.clips.find((item) => item.id === origin.clipId);
                const next = {
                    clipId: origin.clipId,
                    startSec: clip?.startSec ?? origin.startSec,
                    lengthSec: clip?.lengthSec ?? origin.lengthSec,
                    clipPlaybackRate:
                        Number(clip?.clipPlaybackRate ?? origin.basePlaybackRate) || 1,
                    fadeInSec: Number(clip?.fadeInSec ?? origin.baseFadeInSec) || 0,
                    fadeOutSec: Number(clip?.fadeOutSec ?? origin.baseFadeOutSec) || 0,
                    snapOffsetSec: Math.max(
                        0,
                        Number(clip?.snapOffsetSec ?? origin.baseSnapOffsetSec) || 0,
                    ),
                };
                // 锁定参数线：本次是否要搬曲线（决定取数闸门是否需要合上）。
                const lockParamLines = sessionRef.current.lockParamLinesEnabled;
                // 闸门必须在落库派发**之前**合上（见 `loudnessFetchGate`）。
                if (lockParamLines) holdLoudnessFetch();
                const persist = dispatch(setClipsStateBulkRemote({ updates: [next] })).unwrap();
                void (async () => {
                    try {
                        await persist;
                        await finishStretchWithParamLines({
                            rateWritebacks: [
                                {
                                    clipId: origin.clipId,
                                    clipPlaybackRate: next.clipPlaybackRate,
                                },
                            ],
                            // 把该轨道的曲线从旧时间范围映射到新范围
                            //（与旧实现同源，复用抽出的 `stretchLinkedParams`）。
                            rewriteParamLines: lockParamLines
                                ? () =>
                                      stretchLinkedParams(
                                          origin.trackId,
                                          origin.startSec,
                                          origin.lengthSec,
                                          next.startSec,
                                          next.lengthSec,
                                      )
                                : null,
                            xfadeClipIds: origin.xfadeClipIds,
                            initialCrossfadeSides: origin.initialCrossfadeSides,
                            editSides: origin.editSides,
                        });
                    } catch {
                        // 失败不产生 unhandled rejection；交互锁与闸门仍需释放。
                    } finally {
                        finishStretchCommitChain(lockParamLines);
                    }
                })();
                return;
            }

            if (args.cancelled) {
                // 取消：**全部参与者**一起还原（多选裁切只还原锚点会让其余 clip
                // 停在半途位置，与后端分叉）。
                batch(() => {
                    for (const [clipId, base] of origin.baseById) {
                        dispatch(moveClipStart({ clipId, startSec: base.startSec }));
                        dispatch(setClipLength({ clipId, lengthSec: base.lengthSec }));
                        dispatch(
                            setClipSourceRange({
                                clipId,
                                sourceStartSec: base.sourceStartSec,
                                sourceEndSec: base.sourceEndSec,
                            }),
                        );
                        // 吸附偏移也要还原：预览的左缘分支会按"素材内的点不动"
                        // 改写它（见 `resolveTrimSnapOffset`），漏还原会让 Redux
                        // 停在半途值、与后端分叉。
                        dispatch(
                            setClipSnapOffset({
                                clipId,
                                snapOffsetSec: base.snapOffsetSec,
                            }),
                        );
                    }
                });
                // 自动交叉淡化也要回到按下时的重叠关系（按已还原的几何重算即可）。
                if (s.autoCrossfadeEnabled) {
                    previewAutoCrossfade(
                        store.getState().session,
                        origin.xfadeClipIds,
                        dispatch,
                        origin.initialCrossfadeSides,
                        origin.editSides,
                    );
                }
                endKernelGestureInteraction();
                return;
            }
            dispatch(checkpointHistory());
            // 同拖拽：提交值取 Redux 当前值（预览已写入吸附后的结果）。
            const session = sessionRef.current;
            const updates = origin.participants.flatMap((participant) => {
                const clip = session.clips.find((item) => item.id === participant.clipId);
                if (clip === undefined) return [];
                return [
                    {
                        clipId: participant.clipId,
                        startSec: clip.startSec,
                        lengthSec: clip.lengthSec,
                        // 裁切必须同时提交**源区间**：只改长度会让后端按旧源区间
                        // 重新解释内容（波形与音频都会对不上）。预览阶段已同步改过
                        // 源区间，这里取 Redux 当前值即可。
                        sourceStartSec: clip.sourceStartSec,
                        sourceEndSec: clip.sourceEndSec,
                        // 吸附偏移同样必须提交：左缘裁切会按"素材内的点不动"改写它，
                        // 漏提交会让权威快照把预览值回滚成按下前的相对偏移。
                        snapOffsetSec: Math.max(0, Number(clip.snapOffsetSec) || 0),
                    },
                ];
            });
            if (updates.length === 0) {
                endKernelGestureInteraction();
                return;
            }
            const persist = dispatch(setClipsStateBulkRemote({ updates })).unwrap();
            void (async () => {
                try {
                    await persist;
                } finally {
                    // 自动交叉淡化：按落库后的重叠关系写回自动 fade（开关关闭时只清理
                    // 「已脱离重叠」的自动值，保证分离后手动 fade 能恢复显示）。
                    const latest = sessionRef.current;
                    if (s.autoCrossfadeEnabled) {
                        await applyAutoCrossfade(latest, origin.xfadeClipIds, dispatch, {
                            affectedSides: origin.initialCrossfadeSides,
                            editSides: origin.editSides,
                        });
                    } else {
                        await applyDetachedAutoCrossfadeClears(
                            latest,
                            origin.xfadeClipIds,
                            dispatch,
                            origin.initialCrossfadeSides,
                            origin.editSides,
                        );
                    }
                }
            })()
                .catch(() => undefined)
                .finally(endKernelGestureInteraction);
        },
        [
            dispatch,
            endKernelGestureInteraction,
            finishStretchCommitChain,
            finishStretchWithParamLines,
            sessionRef,
            s.autoCrossfadeEnabled,
        ],
    );

    /** 内核淡变角：按下时的原始值（用于回滚）。 */
    return {
        handleKernelTrimPreview,
        handleKernelTrimCommit,
        handleKernelSnapOffsetPreview,
        handleKernelSnapOffsetCommit,
    };
}
