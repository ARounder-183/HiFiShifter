// 时间轴内核 seek / 拖拽手势。拆自 TimelinePanel。
import React from "react";
import { isModifierActive } from "../../../../features/keybindings/keybindingsSlice";
import { resolveClipDragCopyMode } from "../hooks/clipDragCopyMode";
import { copyClipsFromDrag } from "../hooks/copyClipsFromDrag";
import { createNewTrackForKernelDrop, createTrackIdsForDrop } from "../hooks/createNewTrackForDrop";
import { resolveKernelDropTarget } from "../hooks/kernelDropCommit";
import { planSeekGesture, type SeekGesturePhase } from "../kernel/interaction/seekGesture";
import {
    selectTrackRemote,
    seekPlayhead,
    setClipsStateBulkRemote,
    setPlayheadSec,
    moveClipStart,
    moveClipTrack,
    checkpointHistory,
    setClipSourceRange,
    beginInteraction,
    endInteraction,
} from "../../../../features/session/sessionSlice";
import type { TimelineSessionSlice } from "./useTimelineState";
import { batch } from "react-redux";
import {
    moveClipRemote,
    moveClipsRemote,
} from "../../../../features/session/thunks/timelineThunks";
import { NEW_TRACK_SENTINEL } from "../constants";
import { beginClipGeometryPreview, endClipGeometryPreview } from "../clipGeometryPreviewBus";
import {
    SNAP_HIGHLIGHT_GROUP,
    buildLoopBoundaryHighlightEntry,
    clearSnapHighlights,
    publishSnapHighlights,
} from "../../../../utils/snapHighlight";
import {
    loopSnapThresholdSec,
    nearestBoundarySnapOffsetSec,
    slipBoundaryAlignedSides,
} from "../../../../utils/loopSnap";
import { useTimelineState } from "../hooks/useTimelineState";
import {
    applyKernelEditDelta,
    resolveKernelEditParticipants,
    type KernelEditParticipant,
} from "../hooks/kernelEditSet";
import { shouldCollapseStaleSelectionOnDrag } from "../kernel/interaction/primeSelection";
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
import { computeSlipWindow, toBoundarySnapClip } from "../hooks/slipWindow";
import type { TimelineKernelHost } from "../kernel/host/timelineKernelHost";
import type { useTimelineClipActions } from "../hooks/useTimelineClipActions";
import type { AppDispatch } from "../../../../app/store";

type Deps = {
    state: ReturnType<typeof useTimelineState>;
    clipActions: ReturnType<typeof useTimelineClipActions>;
    s: TimelineSessionSlice;
    dispatch: AppDispatch;
    kernelHostRef: React.MutableRefObject<TimelineKernelHost | null>;
};

export function useTimelineKernelSeekDrag({
    state,
    clipActions,
    s,
    dispatch,
    kernelHostRef,
}: Deps) {
    const {
        sessionRef,
        pxPerSec,
        snapTimelineDetailed,
        clipMultiSelectToggleKb,
        slipEditKb,
        noSnapKb,
        copyDragKb,
    } = state;
    const {
        deselectAllTrackLaneClips,
        selectClipRangeByRect,
        toggleTrackLaneCtrlSelection,
        ensureTrackLaneSelected,
        recordLastClickPosition,
        selectTrackLaneClipRemote,
        multiSelectedClipIds,
        setMultiSelectedClipIds,
        setMultiSelectedClipIdsFromAction,
    } = clipActions;
    function dragScrollLeftOf(el: HTMLElement): number {
        const host = kernelHostRef.current;
        if (host !== null) return host.getViewport().scrollLeft;
        return el.scrollLeft;
    }

    /** 内核 seek 的待提交位置与 rAF 句柄（拖拽期间按帧节流，松手立即提交）。 */
    const kernelSeekPendingRef = React.useRef<number | null>(null);
    const kernelSeekRafRef = React.useRef<number | null>(null);

    /**
     * 内核 seek 回调：单击或拖拽空白处跳转播放头。
     *
     * 拖拽帧以 rAF 节流提交（内核按 rAF 频率回调），避免每个指针事件都打一次
     * 后端 seek；松手（commit）时取消待提交帧并立即提交最终位置，保证落点精确。
     *
     * 特殊说明（**必须与 seek 一起乐观写 playheadSec**）：`seekPlayhead.fulfilled`
     * 只在「后端返回值与请求值不同」（如被 clamp 修正）时才采纳后端值，其余情况
     * 依赖调用方**已经**写好 `state.playheadSec`；真实后端 `set_transport` 对
     * 非负请求原样回显（`playhead_sec = v.max(0.0)`），于是后端值恒等于请求值、
     * 采纳分支永不命中——只派发 `seekPlayhead` 时引擎动了而 store 的
     * `playheadSec` 纹丝不动（"点空白处播放头不跟随"的根因；播放中 30Hz 轮询
     * 重新锚定才让它"偶尔"生效）。此处与标尺路径
     * （`useTimelineState.setPlayheadFromClientX`：`setPlayheadSec` + `seekPlayhead`
     * 成对派发）保持同一契约，两个分支都不得只派发 seek。
     */
    /**
     * 把请求的秒数按旧实现的「光标吸附」规则规范化（**不改状态**）。
     *
     * 所有播放头落点路径（标尺 / 轨道空白 / clip 点击）在旧实现里都汇入
     * `setPlayheadFromClientX`：先 `snapTimelineDetailed(…, "cursor", …)` 吸附，
     * 再写 `setPlayheadSec` + `seekPlayhead`；`commit=true` 表示手势语境结束，
     * 顺带清除瞬态吸附高亮。内核原先直接写**裸**秒数——标尺路径吸附、轨道空白
     * 路径不吸附，同一个界面里两条 seek 行为不一致。
     *
     * @param sec 目标秒（未吸附）。
     * @param commit true = 提交式落点（单击 / 拖拽松手），false = 拖拽中间帧。
     * @returns 吸附后的秒。
     */
    const resolveKernelSeekSec = React.useCallback(
        (sec: number, commit: boolean): number =>
            snapTimelineDetailed(
                sec,
                "cursor",
                // 高亮只在拖拽期间发布（单击跳转完全不走高亮通道），与旧实现一致。
                commit ? undefined : { highlight: { sources: [] } },
            ).sec,
        [snapTimelineDetailed],
    );

    /**
     * 纯粹的播放头落点（吸附 + 乐观写 + 高亮收口）——**不含**空白点击的
     * 「清空选中 / 切换轨道」语义。
     *
     * 旧实现的 clip 单击 seek（`ClipItem` / `ClipEdgeHandles` / `FadeHitLayer`
     * 的 `seekFromClientX`）只移动播放头；把"清空选中"混进来会让「点一下 clip
     * 把播放头带过去」顺带取消选中，与旧实现完全相反。
     *
     * @param sec 目标秒。
     * @returns 无返回值。
     */
    const handleKernelSeekTo = React.useCallback(
        (sec: number): void => {
            const target = resolveKernelSeekSec(sec, true);
            clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            dispatch(setPlayheadSec(target));
            void dispatch(seekPlayhead(target));
        },
        [dispatch, resolveKernelSeekSec],
    );

    const handleKernelSeek = React.useCallback(
        (sec: number, phase: SeekGesturePhase, trackId?: string | null) => {
            // 播放态实时判定：必须读 store（sessionRef 在 effect 提交后才刷新，
            // 连打播放/停止时是上一拍的值）。播放中按下与拖拽都不写播放头，
            // 否则乐观值与 30Hz 轮询（用引擎真实播放位置覆写 playheadSec）
            // 反复争夺同一字段，视觉插值在"硬复位"与"按采样外推"之间交替
            // ——表现就是播放光标闪回（按下跳到指针处、下一帧又被拽回引擎位置）。
            const isPlayingNow = Boolean(store.getState().session.runtime.isPlaying);
            const plan = planSeekGesture(phase, isPlayingNow);

            if (plan.blankClickSemantics) {
                // 空白点击的选中语义（与旧实现 pointerdown 捕获分支同源）：
                // 1) 清空 clip 选中——但**保留轨道焦点**（空白点击是"取消 clip
                //    目标"，不是"切换轨道目标"）；
                // 2) 按「允许时间轴点击切换轨道」把当前轨道切到点击所在轨道。
                // 只在按下的那一拍执行（松手不再重复）。
                deselectAllTrackLaneClips();
                if (
                    trackId != null &&
                    sessionRef.current.paramEditorTimelineClickSelectTrackEnabled &&
                    trackId !== sessionRef.current.selectedTrackId
                ) {
                    // `applySelectedClip: false` 是**契约**，不是可选优化：
                    //
                    // 1) 上面的 `deselectAllTrackLaneClips()` 是**纯本地** reducer
                    //    （`setSelectedClipPreservingTrack(null)`），从不通知后端，
                    //    因此后端一直记着旧的 `selected_clip_id`；
                    // 2) `selectTrackRemote.fulfilled` 默认拿后端快照覆盖前端选中
                    //    （`sessionSlice.ts` 的 `applySelectedClip` 闸门）；
                    // 3) 纯字符串让 `typeof arg !== "object"` 判为"要恢复"，于是刚
                    //    清掉的选中被**异步复活** —— 表现为"点空白切轨时选中没被
                    //    取消"。复活后的 `selectedClipId` 可能**不属于**当前轨道
                    //    （`select_track` 刻意不清它，见 `state.rs`），于是它成为一个
                    //    跨轨的陈旧目标，被按 `selectedClipId` 取用命令的路径直接
                    //    读到（例如 `removeSelectedClipRemote`）。
                    //
                    // 该契约由 `019e93ed`（"blank-click deselect survives track
                    // switching"）建立，当时两处 DOM 调用点都传了
                    // `applySelectedClip: false`；内核补全提交 `464a78bb` 在新调用点
                    // 写成纯字符串，**静默解除**了该修复。修改此处前请先读这两个提交，
                    // 不要把对象形式"简化"回纯字符串。
                    //
                    // 特殊说明（残留风险，本闸门覆盖不到）：取消选中仍是纯本地的，
                    // 因此任何返回**完整快照**的 `*Remote.fulfilled` 仍可能经
                    // `applyTimelineState` 无条件写回 `selected_clip_id` 而复活它。
                    // 那条通道是既有行为、与本闸门无关；要彻底修需另开一轮（让
                    // `applyTimelineState` 也尊重"本地已清空"的意图）。
                    void dispatch(selectTrackRemote({ trackId, applySelectedClip: false }));
                }
            }
            // 播放中（`plan.playhead === "none"`）：按下与拖拽都不写播放头、也不
            // 发布吸附高亮 —— 光标留在引擎位置，只有松手那一拍提交落点。
            // 未播放：按下即提交、拖拽逐帧预览（吸附高亮只在拖拽期间发布），
            // 与旧实现 `startDeferredPlayheadSeek` 完全一致。
            if (plan.playhead === "none") {
                if (kernelSeekRafRef.current != null) {
                    cancelAnimationFrame(kernelSeekRafRef.current);
                    kernelSeekRafRef.current = null;
                }
                kernelSeekPendingRef.current = null;
                // 按下仍清一次高亮：旧实现的按下走提交分支顺带清理，是"上一次
                // 手势异常中断留下高亮"的兜底。本分支不再发布高亮，但保留该清理
                // 以保证同一不变式；拖拽帧不重复调用（无高亮可清）。
                if (phase === "press") clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                return;
            }
            const target = resolveKernelSeekSec(sec, plan.playhead === "commit");
            kernelSeekPendingRef.current = target;
            if (plan.playhead === "commit") {
                if (kernelSeekRafRef.current != null) {
                    cancelAnimationFrame(kernelSeekRafRef.current);
                    kernelSeekRafRef.current = null;
                }
                kernelSeekPendingRef.current = null;
                clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                // 乐观写 store：见函数头注释（缺了它 playheadSec 不会动）。
                dispatch(setPlayheadSec(target));
                void dispatch(seekPlayhead(target));
                return;
            }
            if (kernelSeekRafRef.current != null) return;
            kernelSeekRafRef.current = requestAnimationFrame(() => {
                kernelSeekRafRef.current = null;
                const pending = kernelSeekPendingRef.current;
                if (pending == null) return;
                kernelSeekPendingRef.current = null;
                // 拖拽中间帧**只写乐观位置，不打后端**：旧实现
                // `startDeferredPlayheadSeek` 的移动分支走 `commit = false`，整段拖拽
                // 只在松手时发一次 `seekPlayhead`（内核原先逐帧成对派发，等于按住拖动
                // 时以 rAF 频率持续刷后端）。松手的收尾由内核补发提交式落点。
                dispatch(setPlayheadSec(pending));
            });
        },
        [deselectAllTrackLaneClips, dispatch, resolveKernelSeekSec, sessionRef],
    );

    // 面板卸载时取消挂起的 seek 预览帧：拖拽中（空白区 seek）卸载（视图切换 /
    // dock 关闭）会让 rAF 在卸载后仍向 store 派发一次过期落点。
    React.useEffect(() => {
        return () => {
            if (kernelSeekRafRef.current != null) {
                cancelAnimationFrame(kernelSeekRafRef.current);
                kernelSeekRafRef.current = null;
            }
        };
    }, []);

    /**
     * 内核选中回调：点击 clip 选中。
     *
     * 多选修饰键（Ctrl / ⌘）切换集合成员；否则单选并同步焦点 clip
     * （焦点 clip 驱动参数编辑器的编辑目标，必须与多选集合一起更新）。
     *
     * 特殊说明：两个分支都**复用旧实现的选中入口**而不是自己拼状态——
     * `selectTrackLaneClipRemote` / `toggleTrackLaneCtrlSelection` 还负责
     * 「点击 clip 切轨」（手册：「点击时间轴中的音频块或空白区域会自动切换当前
     * 轨道」，受 `允许时间轴点击切换轨道` 控制）与后端 `selected_clip` 落库。
     * 内核只识别手势，这些语义不在内核侧重写。
     */
    const handleKernelSelectClip = React.useCallback(
        (clipId: string, additive: boolean, rangeSelect?: boolean, clientX?: number) => {
            // 范围选择（默认 Shift）：复用旧实现的范围选择语义——它自带锚点状态
            // 与「按点击位置取时间范围」的规则，重写一份必然分叉。
            if (rangeSelect === true) {
                selectClipRangeByRect(clipId, undefined, clientX);
                return;
            }
            if (additive) {
                toggleTrackLaneCtrlSelection(clipId);
                return;
            }
            // 普通单击同样要维护范围选择的锚点与点击位置：下一次 Shift 点击要靠它们
            // 决定范围起点。不记录时「先点 A、再 Shift 点 B」会退化成只选中 B
            // （旧实现在 `TrackLane` 的点击收尾里做同一件事）。
            ensureTrackLaneSelected(clipId);
            recordLastClickPosition(clientX ?? 0);
            selectTrackLaneClipRemote(clipId);
        },
        [
            ensureTrackLaneSelected,
            recordLastClickPosition,
            selectClipRangeByRect,
            selectTrackLaneClipRemote,
            toggleTrackLaneCtrlSelection,
        ],
    );

    /** 内核拖拽：按下时的原始位置（把相对位移换算为绝对位置，并支持回滚）。 */
    /**
     * copy 拖拽的 ghost 预览（内容坐标）。
     *
     * 移动语义下内核靠"乐观位置"显示，copy 语义下原 clip 不动，必须有这一层——
     * 否则 ⌘+拖拽期间画面毫无反馈。纵向位置由内核视图按 `rowHeight` 换算
     * （面板没有行高），这里只给内容坐标的左缘与宽度。
     */
    const [kernelGhost, setKernelGhost] = React.useState<Array<{
        key: string;
        leftPx: number;
        widthPx: number;
        trackId: string;
    }> | null>(null);

    /**
     * 内核拖拽是否正落在「新建轨道」哨兵上（拖到全部轨道之下）。
     *
     * React state 驱动幽灵行的渲染，ref 做去重（预览每帧都会给出同一个判定，
     * 逐帧 setState 会白白重渲染整个面板）。
     */
    const [kernelDropToNewTrack, setKernelDropToNewTrack] = React.useState(false);
    const kernelDropToNewTrackRef = React.useRef(false);

    /**
     * 内核手势的**交互锁**（`beginInteraction` / `endInteraction` 的成对封装）。
     *
     * 【为什么每个手势都必须持有】`sessionSlice` 在 `_interactionLockCount > 0`
     * 时丢弃后端回包的过期快照。手势期间若有一条 `*Remote.fulfilled` 抵达（例如
     * 上一次点击的 `selectClipRemote`、或后端的周期性状态推送），没有这把锁它的
     * 旧快照会**覆盖掉正在预览的乐观值**——表现为拖拽中途画面突然弹回原位。
     * 旧实现的每个连续手势都成对调用（`useClipDrag` / `useEditDrag` /
     * `useSlipDrag` / `useSnapOffsetDrag` 各一处），内核路径此前只有 snap offset 接了。
     *
     * 【为什么用 ref 去重】预览回调逐帧触发，而 begin / end 每只手势只能各一次。
     *
     * 【释放时机】与旧实现一致：**落库请求完成后**才释放。若在提交瞬间就释放，
     * `endInteraction()` 到 `fulfilled` 之间会留出一个窗口，窗口内其他 in-flight
     * thunk 的旧快照仍能把乐观值打回去（旧实现 `useClipDrag` 对此有逐字注释）。
     * 纯取消 / 零位移的路径没有任何远程写入，直接释放。
     */
    const kernelGestureInteractionActiveRef = React.useRef(false);
    const beginKernelGestureInteraction = React.useCallback((): void => {
        if (kernelGestureInteractionActiveRef.current) return;
        kernelGestureInteractionActiveRef.current = true;
        // 记录**按下时**的全量 clip 几何，供参数编辑器把响度基线/曲线搬到正确位置
        //（见 `clipGeometryPreviewBus`）。时序要求：必须早于本次手势的任何乐观写入
        // —— 本函数在首个真实位移帧、且在任何预览派发之前调用，满足该要求。
        beginClipGeometryPreview(sessionRef.current.clips);
        dispatch(beginInteraction());
    }, [dispatch, sessionRef]);
    const endKernelGestureInteraction = React.useCallback((): void => {
        if (!kernelGestureInteractionActiveRef.current) return;
        kernelGestureInteractionActiveRef.current = false;
        // 只宣告手势结束；映射本身由消费方保留到"提交之后取的快照"落地（否则
        // 松手瞬间会退回旧位置的基线，表现为闪一下）。
        endClipGeometryPreview();
        dispatch(endInteraction());
    }, [dispatch]);

    const kernelDragOriginRef = React.useRef<{
        clipId: string;
        startSec: number;
        lengthSec: number;
        trackId: string;
        /**
         * clip 自身的吸附偏移点（秒）。
         *
         * 多源吸附把「起点 / 终点 / 自身吸附偏移点」同时作为被吸附对象（旧实现
         * `useClipDrag` 同源）。必须随 origin 一起记下：预览期间 Redux 里的
         * `startSec` 已被改写，但 `snapOffsetSec` 是 clip 的固有属性，拖拽中不变。
         */
        snapOffsetSec: number;
        /**
         * 本次拖拽是否已进入 copy 模式。
         *
         * **单向**：false → true 允许（拖拽中途按下复制键），true → false 不允许
         * （与旧实现 `resolveClipDragCopyMode` 的既有语义一致——中途松开复制键
         * 不会把已经"复制"的意图退回成"移动"，避免松手瞬间语义反转）。
         */
        copyMode: boolean;
        /**
         * 本次拖拽实际作用于的全部 clip（含锚点）。
         *
         * 由「多选集合 + 编组展开」在按下时解析一次（见 `kernelEditSet`）：
         * 内核只回调锚点位移，不展开就会出现「选中多个只动一个」「同组不联动」。
         */
        participants: KernelEditParticipant[];
        /** 锚点初始轨道序号：把内核给的 `targetTrackId` 换成轨道偏移量。 */
        anchorTrackIndex: number;
        /** 最近一次预览生效的**共享位移**（已含吸附与左边界钳制）——提交时用它。 */
        lastDeltaStartSec: number;
        /** 波纹跟随集（乐观预览；提交后的权威结果由后端计算）。 */
        rippleFollowers: RippleFollowerMap;
        /** 自动交叉淡化：受本次编辑影响的 clip（参与者 + 波纹跟随）。 */
        editedXfadeClipIds: string[];
        /** 自动交叉淡化：编辑前每侧的既有重叠关系（拖开时只清自动、保留手动 fade）。 */
        initialCrossfadeSides: ReturnType<typeof computeInitialCrossfadeSides>;
        /**
         * 本次拖拽是否为 **slip**（`Alt` + 拖 clip 中部 = 调整内部偏移）。
         *
         * 语义与移动 / 复制完全不同：时间轴位置与长度都不变，只平移源窗口。
         * 在**按下时**定死（与旧实现 `startSlipDrag` 的独立手势一致）。
         */
        slipMode: boolean;
        /** slip：已应用的累计位移（换算增量用，避免逐帧叠加导致越拖越快）。 */
        appliedSlipSec: number;
        /** slip：每个参与者按下时的源窗口（取消时回滚）。 */
        baseSourceById: Map<string, { sourceStartSec: number; sourceEndSec: number }>;
        /**
         * slip：锚 clip 按下时的「媒体边界吸附」视图。
         *
         * 候选族只依赖初始几何（平移不变），因此按下时快照一次即可（与旧实现
         * `useSlipDrag` 的 `anchorSnapshot` 同源）；内容时长 D 的解析规则由
         * `toBoundarySnapClip` 与旧实现共用。
         */
        slipAnchor: ReturnType<typeof toBoundarySnapClip>;
        /**
         * slip：最近一次分发的源窗口。
         *
         * 提交用它而**不回读 Redux**——与旧实现 `lastById` 同源（防并发更新 /
         * 历史归一化把交互数学结果污染）。
         */
        lastSourceById: Map<string, { sourceStartSec: number; sourceEndSec: number }>;
    } | null>(null);

    /**
     * 内核拖拽预览：写乐观位置（拖拽期间每帧，内核已按值去重）。
     *
     * 特殊说明：内核回调的是**相对按下位置的位移**，而 Redux 里该 clip 的位置
     * 在上一次预览时已被改写——必须用按下时记下的原始位置换算，否则位移会逐帧叠加
     * （表现为 clip 越拖越快）。
     */
    const handleKernelDragPreview = React.useCallback(
        (args: {
            clipId: string;
            deltaSec: number;
            /** 未钳制的指针位移（秒）。Slip 用它 —— 见下方 `desiredTotal` 的说明。 */
            rawDeltaSec: number;
            targetTrackId: string;
            modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
        }) => {
            if (kernelDragOriginRef.current?.clipId !== args.clipId) {
                const session = sessionRef.current;
                const clip = session.clips.find((item) => item.id === args.clipId);
                if (clip === undefined) return;
                const trackIds = session.tracks.map((track) => track.id);
                // ── 拖拽起手：陈旧选区收敛 ────────────────────────────────
                // 参与集合来自 `multiSelectedClipIds`：被拖的 clip 与邻居**都在**
                // 集合里时整组一起移动。而集合可能是**动作驱动**的批量填充
                // （多文件导入 / 打开工程 / 粘贴 / 分割 / 复制拖拽）——那不代表
                // 用户想整组拖动：导入后同轨相邻的多个 clip 都在集合里，随手抓一个
                // 拖，右邻就跟着走，而多选描边仅 2px 看不出来。这正是
                // 「拖一个 clip，右边的 clip 也跟着动」的根因。
                //
                // 判定放在这里（而非宿主按下时）有两个原因：
                // 1. 需要"选区来源"这一 Redux 事实（`multiSelectionIntentional`），
                //    宿主不持有；
                // 2. Shift 在这里只按 **drag** 语义（`clipNoSnap` 免吸附）参与判定，
                //    不再误用其 **click** 语义（`clipRangeSelect` 范围选择）——
                //    修复前正因按下时按 click 语义跳过收敛，Shift+拖拽会带回陈旧邻块。
                //
                // 必须早于 participants 计算：晚一步第一次预览就按旧集合算过了。
                const useSingleAnchor = shouldCollapseStaleSelectionOnDrag({
                    multiSelectToggleActive: isModifierActive(
                        clipMultiSelectToggleKb,
                        args.modifiers,
                    ),
                    selectionIntentional: s.multiSelectionIntentional,
                    multiSelectionSize: multiSelectedClipIds.length,
                    anchorInMultiSelection: multiSelectedClipIds.includes(clip.id),
                });
                const participantSeedIds = useSingleAnchor ? [clip.id] : multiSelectedClipIds;
                // 参与集合：多选集合 + 编组展开（与旧实现 `useClipDrag` 同源）。
                // 不展开时「选中多个只移动一个」「同组不联动」——属数据语义错误。
                const participants = resolveKernelEditParticipants({
                    anchorClipId: clip.id,
                    multiSelectedClipIds: participantSeedIds,
                    clips: session.clips,
                    trackIds,
                    ignoreGrouping: session.ignoreGrouping,
                    disabledGroupIds: session.disabledGroupIds,
                    expandGroups: true,
                });
                if (participants.length === 0) return;
                const rippleFollowers = buildRippleFollowers(
                    session.clips,
                    new Set(participants.map((item) => item.clipId)),
                    Math.min(...participants.map((item) => item.startSec)),
                    session.rippleMode,
                    new Set(participants.map((item) => item.trackId)),
                );
                const editedXfadeClipIds = Array.from(
                    new Set<string>([
                        ...participants.map((item) => item.clipId),
                        ...Object.keys(rippleFollowers),
                    ]),
                );
                kernelDragOriginRef.current = {
                    clipId: clip.id,
                    startSec: clip.startSec,
                    lengthSec: clip.lengthSec,
                    trackId: clip.trackId,
                    snapOffsetSec: Math.max(0, Number(clip.snapOffsetSec) || 0),
                    copyMode: false,
                    slipMode: isModifierActive(slipEditKb, args.modifiers),
                    slipAnchor: toBoundarySnapClip(clip),
                    appliedSlipSec: 0,
                    baseSourceById: new Map(
                        participants.map((participant) => {
                            const item = session.clips.find(
                                (candidate) => candidate.id === participant.clipId,
                            );
                            return [
                                participant.clipId,
                                {
                                    sourceStartSec: Number(item?.sourceStartSec ?? 0) || 0,
                                    sourceEndSec: Number(item?.sourceEndSec ?? 0) || 0,
                                },
                            ] as const;
                        }),
                    ),
                    lastSourceById: new Map(),
                    participants,
                    anchorTrackIndex: trackIds.indexOf(clip.trackId),
                    lastDeltaStartSec: 0,
                    rippleFollowers,
                    editedXfadeClipIds,
                    initialCrossfadeSides: computeInitialCrossfadeSides(
                        session.clips,
                        editedXfadeClipIds,
                    ),
                };
                // 首个真实位移帧（内核只在越过拖拽阈值后才回调预览）= 手势真正开始：
                // 上交互锁，让手势期间抵达的后端旧快照不再覆盖乐观值。
                beginKernelGestureInteraction();
            }
            const origin = kernelDragOriginRef.current;
            if (origin === null) return;
            // copy 模式判定：复用旧实现的函数（含"已配置绑定为准 + 非 macOS 的 Ctrl
            // 回退"）。**单向**——一旦进入 copy 就不再退回移动，避免松手瞬间语义反转。
            if (origin.slipMode) {
                // Alt + 拖 clip 中部 = **slip**（调整内部偏移）：时间轴位置与长度
                // 都不变，只平移源窗口。逐帧按**增量**应用（用当前 Redux 值 + 增量
                // 换算），避免累计位移被重复施加导致越拖越快。
                //
                // ── 循环节 / 内容边界吸附（拖拽全程生效）──────────────────
                // 属于常规吸附体系：受吸附总开关与「拖动时切换吸附」修饰键（XOR）
                // 控制，且需在吸附设置中启用「Clip 边缘吸附到源素材首尾」。命中时
                // 把**累计位移**替换为吸附值，再以「目标累计 − 已应用累计」驱动增量
                // （与旧实现 `useSlipDrag` 同一算法，只是位移正负号约定相反）。
                const timelineSnap = sessionRef.current.timelineSnap;
                // 【为什么 Slip 用未钳制位移】`args.deltaSec` 是**位置**位移（内核按
                // "clip 不能从负时间开始"钳到 ≥0）；而 Slip 改的是 clip 内部的源窗口
                // 偏移，合法域是任意符号任意大小（向左滑出媒体起点 = 前导静音）。
                // 用被钳的值会让 Slip 单向卡死：`startSec = 0` 的 clip 完全无法向左滑。
                let desiredTotal = args.rawDeltaSec;
                {
                    const anchor = origin.slipAnchor;
                    const snapActive = computeEffectiveSnap(
                        s.snapEnabled,
                        isModifierActive(noSnapKb, args.modifiers),
                    );
                    if (
                        (anchor.isContentBearing || anchor.loopEnabled) &&
                        timelineSnap.snapClipsToSourceMedia &&
                        snapActive &&
                        timelineSnap.snapDistancePx > 0
                    ) {
                        // 内核约定正 = 向右拖；旧实现约定正 = 向左拖（起点指针 − 当前
                        // 指针），故取反号换算到同一 X 域。
                        const dir = anchor.reversed ? -1 : 1;
                        const rawWindowShift = -desiredTotal * dir;
                        const snappedW = nearestBoundarySnapOffsetSec(
                            anchor,
                            "slip",
                            rawWindowShift,
                        );
                        if (
                            snappedW != null &&
                            Math.abs(snappedW - rawWindowShift) <=
                                loopSnapThresholdSec(timelineSnap.snapDistancePx, pxPerSec) + 1e-12
                        ) {
                            desiredTotal = -snappedW * dir;
                            // 循环节命中：只高亮**真正对齐**的那一侧（媒体边界恰好
                            // 落在 Clip 起点 → 亮起点；落在终点 → 亮终点；len·r 恰为
                            // 整周期等两侧同时对齐才两缘同亮）。
                            const anchorClip = sessionRef.current.clips.find(
                                (item) => item.id === origin.clipId,
                            );
                            if (anchorClip !== undefined) {
                                const aligned = slipBoundaryAlignedSides(anchor, snappedW);
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
                                            clipId: origin.clipId,
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
                const dApplied = desiredTotal - origin.appliedSlipSec;
                if (Math.abs(dApplied) < 1e-12) return;
                origin.appliedSlipSec = desiredTotal;
                const updates: Array<{
                    clipId: string;
                    sourceStartSec: number;
                    sourceEndSec: number;
                }> = [];
                for (const participant of origin.participants) {
                    const clip = sessionRef.current.clips.find(
                        (item) => item.id === participant.clipId,
                    );
                    if (clip === undefined) continue;
                    // ⚠️ **取反号**：`dApplied` 在「屏幕位移」域（内核约定正 = 向右拖），
                    // 而 `computeSlipWindow` 要的是「窗口平移量」域（正 = 源窗口向素材
                    // **后段**平移）。向右拖 = 把内容往右推 = 露出**更早**的素材 = 窗口
                    // 向前段平移，故屏幕正位移对应窗口负平移。
                    //
                    // 旧实现 `useSlipDrag` 的 `desiredTotal = 起点指针 − 当前指针`
                    // （向右拖为负）恰好已经是窗口域，所以它直接传即可；内核给的是
                    // 屏幕域，漏掉这个负号会让方向整体反过来（拖右显示更晚的素材）。
                    const next = computeSlipWindow(clip, -dApplied);
                    if (next === null) continue;
                    origin.lastSourceById.set(participant.clipId, next);
                    updates.push({ clipId: participant.clipId, ...next });
                }
                batch(() => {
                    for (const update of updates) {
                        dispatch(
                            setClipSourceRange({
                                clipId: update.clipId,
                                sourceStartSec: update.sourceStartSec,
                                sourceEndSec: update.sourceEndSec,
                            }),
                        );
                    }
                });
                return;
            }
            // copy 模式判定：复用旧实现的函数（含"已配置绑定为准 + 非 macOS 的 Ctrl
            // 回退"）。**单向**——一旦进入 copy 就不再退回移动，避免松手瞬间语义反转。
            origin.copyMode = resolveClipDragCopyMode({
                existingCopyMode: origin.copyMode,
                ctrlKey: args.modifiers.ctrlKey,
                modifierActive: isModifierActive(copyDragKb, args.modifiers),
            });
            // 免吸附修饰键（默认 Shift）把吸附总开关临时取反——与旧实现同一函数，
            // 拖拽中途按下/松开即时生效。
            const snapActive = computeEffectiveSnap(
                s.snapEnabled,
                isModifierActive(noSnapKb, args.modifiers),
            );
            const rawStart = Math.max(0, origin.startSec + args.deltaSec);
            // 吸附：复用旧实现的 snapTimelineDetailed（多源候选 + 取更近者），
            // 内核只给几何位移，吸附规则不在内核里重写。
            //
            // `highlight` 必须传：吸附高亮的**发布与清除**都由该函数按此选项统一
            // 处理（见 useTimelineState.snapTimelineDetailed）。不传时吸附本身仍
            // 生效、位置也对，但完全没有视觉反馈——表现为"吸附没生效"。
            // `excludeClipIds` 必须覆盖**全部参与者**：整组一起移动时，组内其他
            // clip 不应成为自己的吸附目标。
            const nextStart = snapActive
                ? snapTimelineDetailed(rawStart, "clip", {
                      originSec: origin.startSec,
                      anchorTrackId: args.targetTrackId,
                      excludeClipIds: new Set(origin.participants.map((item) => item.clipId)),
                      moveLengthSec: origin.lengthSec,
                      moveSnapOffsetSec: origin.snapOffsetSec,
                      highlight: {
                          sources: [{ trackId: args.targetTrackId, clipId: args.clipId }],
                      },
                  }).sec
                : rawStart;
            // 吸附被关闭（拖拽中切开关 / 按住临时取反键）：高亮必须清掉，
            // 否则会残留上一次的吸附提示（旧实现同样在 else 分支清除）。
            if (!snapActive) clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            const trackIds = sessionRef.current.tracks.map((track) => track.id);
            const targetTrackIndex = trackIds.indexOf(args.targetTrackId);
            const deltaTrack =
                targetTrackIndex >= 0 && origin.anchorTrackIndex >= 0
                    ? targetTrackIndex - origin.anchorTrackIndex
                    : 0;
            // 拖到全部轨道之下 = 新建轨道哨兵：乐观位置写哨兵轨（与旧实现
            // `moveClipTrack(NEW_TRACK_SENTINEL)` 同源），面板据此渲染幽灵行。
            const dropToNewTrack = args.targetTrackId === NEW_TRACK_SENTINEL;
            if (dropToNewTrack !== kernelDropToNewTrackRef.current) {
                kernelDropToNewTrackRef.current = dropToNewTrack;
                setKernelDropToNewTrack(dropToNewTrack);
            }
            const { moves, deltaStartSec } = applyKernelEditDelta({
                participants: origin.participants,
                deltaStartSec: nextStart - origin.startSec,
                deltaTrack,
                trackIds,
                dropToNewTrack,
            });
            origin.lastDeltaStartSec = deltaStartSec;
            if (origin.copyMode) {
                // copy：**原 clip 不动**，只更新 ghost（整组，内容坐标）。
                setKernelGhost(
                    origin.participants.map((participant, index) => ({
                        key: participant.clipId,
                        leftPx: moves[index].startSec * pxPerSec,
                        widthPx: Math.max(1, participant.lengthSec * pxPerSec),
                        trackId: moves[index].trackId,
                    })),
                );
                // 复制不移动原片：波纹跟随集恢复原位（覆盖「拖拽中途切到复制」的残留预览）。
                applyRippleFollowerShift(dispatch, origin.rippleFollowers, 0);
                return;
            }
            batch(() => {
                for (const move of moves) {
                    dispatch(moveClipStart({ clipId: move.clipId, startSec: move.startSec }));
                    dispatch(moveClipTrack({ clipId: move.clipId, trackId: move.trackId }));
                }
                // 波纹（自动跟进）实时预览：后续 clip 随拖拽同步平移，位移取
                // **钳制后的共享位移**（否则左边界处会与参与者错位）。
                applyRippleFollowerShift(dispatch, origin.rippleFollowers, deltaStartSec);
            });
            // 自动交叉淡化实时预览：按当前（乐观）位置重算重叠并更新自动 fade。
            // 必须用 `store.getState().session`（同步新鲜）而不是 sessionRef——
            // react-redux 的 batch 会延迟订阅回调，batch 内 sessionRef 仍是上一帧
            // 位置，会让"拖开瞬间"的预览滞留最后一帧自动淡化长度。
            if (s.autoCrossfadeEnabled) {
                previewAutoCrossfade(
                    store.getState().session,
                    origin.editedXfadeClipIds,
                    dispatch,
                    origin.initialCrossfadeSides,
                );
            }
        },
        [
            beginKernelGestureInteraction,
            clipMultiSelectToggleKb,
            copyDragKb,
            dispatch,
            multiSelectedClipIds,
            s.multiSelectionIntentional,
            noSnapKb,
            pxPerSec,
            s.autoCrossfadeEnabled,
            s.snapEnabled,
            sessionRef,
            slipEditKb,
            snapTimelineDetailed,
        ],
    );

    /**
     * 内核拖拽收尾：提交或回滚。
     *
     * 取消路径（Esc / pointercancel）必须把乐观位置还原——否则 Redux 会停在半途
     * 位置，与后端分叉（旧实现同样在取消分支显式回滚）。
     */
    const handleKernelDragCommit = React.useCallback(
        (args: {
            clipId: string;
            deltaSec: number;
            targetTrackId: string;
            cancelled: boolean;
            modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
        }) => {
            const origin = kernelDragOriginRef.current;
            kernelDragOriginRef.current = null;
            if (origin === null) return;
            // 手势结束：清掉吸附高亮（旧实现同样在收尾清除，否则最后一次的
            // 吸附提示会一直挂在画面上）。
            clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);

            if (origin.slipMode) {
                if (args.cancelled) {
                    batch(() => {
                        for (const [clipId, base] of origin.baseSourceById) {
                            dispatch(
                                setClipSourceRange({
                                    clipId,
                                    sourceStartSec: base.sourceStartSec,
                                    sourceEndSec: base.sourceEndSec,
                                }),
                            );
                        }
                    });
                    endKernelGestureInteraction();
                    return;
                }
                // 零位移（未越过阈值 / 拖回原位）：不写后端（旧实现同样不置 dirty）。
                if (origin.lastSourceById.size === 0) {
                    endKernelGestureInteraction();
                    return;
                }
                dispatch(checkpointHistory());
                // 用**交互数学结果**提交（不回读 Redux——防并发更新 / 历史归一化
                // 把窗口值污染，与旧实现 `lastById` 同源）。
                void dispatch(
                    setClipsStateBulkRemote({
                        updates: [...origin.lastSourceById].map(([clipId, window]) => ({
                            clipId,
                            sourceStartSec: window.sourceStartSec,
                            sourceEndSec: window.sourceEndSec,
                        })),
                    }),
                )
                    .unwrap()
                    .catch(() => undefined)
                    .finally(endKernelGestureInteraction);
                return;
            }

            if (origin.copyMode) {
                setKernelGhost(null);
                // 落点解析必须早于取消早退：取消路径同样要清掉「落到新轨」标记，
                // 否则虚线新轨行会一直挂在画面上，且之后每一次普通移动都会被误读成
                // 哨兵落点（`kernelDropToNewTrackRef` 的语义是"本次手势的落点"，
                // 手势结束就必须复位——与下方 move 分支同一清理方式）。
                const dropTarget = resolveKernelDropTarget({
                    targetTrackId: args.targetTrackId,
                    trackIds: sessionRef.current.tracks.map((track) => track.id),
                    anchorTrackIndex: origin.anchorTrackIndex,
                    newTrackSentinel: NEW_TRACK_SENTINEL,
                });
                if (kernelDropToNewTrackRef.current) {
                    kernelDropToNewTrackRef.current = false;
                    setKernelDropToNewTrack(false);
                }
                // copy 模式下原 clip 从未被移动：既不需要回滚，也不走 move 提交。
                if (args.cancelled) {
                    endKernelGestureInteraction();
                    return;
                }
                // 落库复用抽出的共享函数（与旧实现**同一份**复制语义）：
                // 参与集合整体复制，目标轨按各自初始序号 + 同一偏移量解析。
                const trackIds = sessionRef.current.tracks.map((track) => track.id);
                void copyClipsFromDrag({
                    sourceClipIds: origin.participants.map((item) => item.clipId),
                    initialById: Object.fromEntries(
                        origin.participants.map((item) => [
                            item.clipId,
                            { startSec: item.startSec, trackId: item.trackId },
                        ]),
                    ),
                    initialTrackIndexById: Object.fromEntries(
                        origin.participants.map((item) => [item.trackId, item.trackIndex]),
                    ),
                    // 用**吸附后**的共享位移，与 ghost 预览的位置一致
                    // （用内核原始位移会绕开吸附，表现为"预览吸附、落库不吸附"）。
                    deltaSec: origin.lastDeltaStartSec,
                    // 与预览分支同一套落点语义（预览分支内联算 `deltaTrack` /
                    // `dropToNewTrack`）：哨兵 → 新建轨道；已有轨道 → 相对锚点的轨道
                    // 偏移量。写死 false / 0 会让"幽灵预览能到新轨道和其他轨道、落库
                    // 却留在原轨"，两处解析结果必须一致。
                    dropToNewTrack: dropTarget.dropToNewTrack,
                    trackOffset: dropTarget.trackOffset,
                    allowTrackMove: true,
                    // 选区是否跨多条轨道（旧实现 `useClipDrag` 的
                    // `hasMixedTrackSelection`）。只有跨轨时才按**来源轨道跨度**建同样
                    // 多的新轨、让成员各自落位；写死 false 会让"跨轨多选拖到轨道区下方"
                    // 把整组塌到同一条新轨上，成员之间的相对轨道布局丢失。
                    hasMixedTrackSelection:
                        new Set(origin.participants.map((item) => item.trackId)).size > 1,
                    autoCrossfadeEnabled: s.autoCrossfadeEnabled,
                    dispatch,
                    sessionRef,
                    setMultiSelectedClipIds,
                    setMultiSelectedClipIdsFromAction,
                    // 每个参与者按各自初始轨道序号 + 同一偏移量解析目标轨。
                    //
                    // 特殊说明 1：这里的 id 查询**必须带上 `trackOffset`**。只写
                    // `trackIds[participant.trackIndex]` 会永远返回原轨（偏移量为 0 时
                    // 才恰好正确），使整个跨轨修复变成静默无效——调用方
                    // （`copyClipsFromDrag`）此时已经按 `trackOffset !== 0` 判定过
                    // "应该跨轨"，拿到原轨后不会报错，只会默默同轨复制。
                    //
                    // 特殊说明 2：越界必须**夹取**到 [0, lastIndex]，与预览分支
                    // （`applyKernelEditDelta` 的 `Math.min(lastIndex, Math.max(0, …))`）
                    // 逐字一致。若越界时回落**原轨**，"幽灵显示到了末轨、落库却留在
                    // 原轨"会重新出现：跨轨多选 + 偏移越界时实测预览把两个 participant
                    // 都放到末轨，而落库把越界的那个留在原轨。
                    resolveTrackIdByOffset: (clipId) => {
                        const participant = origin.participants.find(
                            (item) => item.clipId === clipId,
                        );
                        if (participant === undefined || participant.trackIndex < 0) return null;
                        const lastIndex = trackIds.length - 1;
                        if (lastIndex < 0) return null;
                        const targetIndex = Math.min(
                            lastIndex,
                            Math.max(0, participant.trackIndex + dropTarget.trackOffset),
                        );
                        return trackIds[targetIndex] ?? null;
                    },
                    maybeSelectTargetTrack: (trackId) => {
                        // 跨轨复制后把**目标轨道**设为当前轨道（旧实现
                        // `useClipDrag` 的 `maybeSelectTargetTrack` 同源：目标是锚点的
                        // 原轨道 / 已经是当前轨道时跳过）。
                        //
                        // `applySelectedClip: false`：这里只切"当前轨道"，不得让后端
                        // 记忆的 `selected_clip_id` 异步复活（契约与
                        // `handleKernelSeek` 的空白点击同源，见 019e93ed）。
                        if (trackId === null) return;
                        if (trackId === origin.trackId) return;
                        if (sessionRef.current.selectedTrackId === trackId) return;
                        void dispatch(selectTrackRemote({ trackId, applySelectedClip: false }));
                    },
                    createNewTracksForDrop: (span: number) =>
                        createTrackIdsForDrop({ dispatch, sessionRef }, span),
                    createNewTrackForDrop: async () => {
                        const created = await createTrackIdsForDrop({ dispatch, sessionRef }, 1);
                        return created[0] ?? null;
                    },
                })
                    .catch(() => undefined)
                    .finally(endKernelGestureInteraction);
                return;
            }
            if (args.cancelled) {
                batch(() => {
                    for (const participant of origin.participants) {
                        dispatch(
                            moveClipStart({
                                clipId: participant.clipId,
                                startSec: participant.startSec,
                            }),
                        );
                        dispatch(
                            moveClipTrack({
                                clipId: participant.clipId,
                                trackId: participant.trackId,
                            }),
                        );
                    }
                });
                // 波纹跟随集恢复原位（预览期间已被平移）。
                applyRippleFollowerShift(dispatch, origin.rippleFollowers, 0);
                if (kernelDropToNewTrackRef.current) {
                    kernelDropToNewTrackRef.current = false;
                    setKernelDropToNewTrack(false);
                }
                endKernelGestureInteraction();
                return;
            }
            // 哨兵轨：先清掉落点标记（幽灵行随手势一起消失），再建轨并落库。
            if (kernelDropToNewTrackRef.current) {
                kernelDropToNewTrackRef.current = false;
                setKernelDropToNewTrack(false);
                dispatch(checkpointHistory());
                void createNewTrackForKernelDrop({
                    clipIds: origin.participants.map((item) => item.clipId),
                    // 目标起点取**当前**（乐观、含吸附）值，与幽灵行显示的位置一致。
                    startSecById: Object.fromEntries(
                        origin.participants.map((item) => {
                            const clip = sessionRef.current.clips.find(
                                (candidate) => candidate.id === item.clipId,
                            );
                            return [item.clipId, clip?.startSec ?? item.startSec] as const;
                        }),
                    ),
                    originTrackIdById: Object.fromEntries(
                        origin.participants.map((item) => [item.clipId, item.trackId] as const),
                    ),
                    originStartSecById: Object.fromEntries(
                        origin.participants.map((item) => [item.clipId, item.startSec] as const),
                    ),
                    // 各成员的初始轨道序号：跨轨选区据此按跨度建多条新轨并各自落位
                    // （与 copy 路径的 `hasMixedTrackSelection` / span 建轨同源）。
                    trackIndexById: Object.fromEntries(
                        origin.participants.map((item) => [item.clipId, item.trackIndex] as const),
                    ),
                    dispatch,
                    sessionRef,
                    moveLinkedParams: sessionRef.current.lockParamLinesEnabled,
                })
                    .catch(() => undefined)
                    .finally(endKernelGestureInteraction);
                return;
            }
            // 提交值取 Redux 里的当前值（预览已写入**吸附后**的结果）——
            // 用 origin + 内核原始位移会绕开吸附，导致"预览吸附、提交不吸附"。
            const session = sessionRef.current;
            const moves = origin.participants
                .map((participant) => {
                    const clip = session.clips.find((item) => item.id === participant.clipId);
                    const startSec = clip?.startSec ?? participant.startSec;
                    const trackId = clip?.trackId ?? participant.trackId;
                    // 零位移（拖回原位 / 未越过阈值）的成员不入提交集：旧实现
                    // `useClipDrag` 同样以 `|Δstart| > 1e-6 || 换轨` 过滤，全员
                    // 无变化时**整段提交都不发生**（不写后端、不置 dirty）。
                    const changedStart = Math.abs(startSec - participant.startSec) > 1e-6;
                    const changedTrack = trackId !== participant.trackId;
                    if (!changedStart && !changedTrack) return null;
                    return { clipId: participant.clipId, startSec, trackId };
                })
                .filter(
                    (move): move is { clipId: string; startSec: number; trackId: string } =>
                        move !== null,
                );
            if (moves.length === 0) {
                // 全员零位移：没有任何远程写入，交互锁直接释放。
                endKernelGestureInteraction();
                return;
            }
            dispatch(checkpointHistory());
            // 跨轨移动后把**目标轨道**设为当前轨道（旧实现 `useClipDrag` 的
            // `maybeSelectTargetTrack(drag.lastTrackId)` 同源：目标是锚点原轨道或已是
            // 当前轨道时跳过）。`applySelectedClip: false` 见 copy 分支的说明。
            const anchorMove = moves.find((move) => move.clipId === origin.clipId);
            if (
                anchorMove !== undefined &&
                anchorMove.trackId !== origin.trackId &&
                session.selectedTrackId !== anchorMove.trackId
            ) {
                void dispatch(
                    selectTrackRemote({ trackId: anchorMove.trackId, applySelectedClip: false }),
                );
            }
            // 「锁定参数线」必须随移动一起透出（旧实现 `useClipDrag` 同源）：
            // 漏传会让参数线在该开关打开时**不跟随 clip 移动**，且没有任何提示。
            const moveLinkedParams = session.lockParamLinesEnabled;
            const movePromise =
                moves.length > 1
                    ? dispatch(moveClipsRemote({ moves, moveLinkedParams })).unwrap()
                    : dispatch(
                          moveClipRemote({
                              clipId: moves[0].clipId,
                              startSec: moves[0].startSec,
                              trackId: moves[0].trackId,
                              moveLinkedParams,
                          }),
                      ).unwrap();
            void (async () => {
                try {
                    await movePromise;
                } finally {
                    // 自动交叉淡化：按移动后的新重叠关系写回自动 fade；开关关闭时
                    // 只清理「已脱离重叠」的自动值（保证分离后手动 fade 能恢复显示）。
                    const latest = sessionRef.current;
                    if (s.autoCrossfadeEnabled) {
                        await applyAutoCrossfade(latest, origin.editedXfadeClipIds, dispatch, {
                            affectedSides: origin.initialCrossfadeSides,
                        });
                    } else {
                        await applyDetachedAutoCrossfadeClears(
                            latest,
                            origin.editedXfadeClipIds,
                            dispatch,
                            origin.initialCrossfadeSides,
                        );
                    }
                }
            })()
                .catch(() => undefined)
                // 交互锁在落库（含自动交叉淡化写回）完成后才释放——旧实现同一约定。
                .finally(endKernelGestureInteraction);
        },
        [
            dispatch,
            endKernelGestureInteraction,
            s.autoCrossfadeEnabled,
            sessionRef,
            setMultiSelectedClipIds,
            setMultiSelectedClipIdsFromAction,
        ],
    );

    return {
        dragScrollLeftOf,
        handleKernelSeek,
        handleKernelSeekTo,
        handleKernelSelectClip,
        handleKernelDragPreview,
        handleKernelDragCommit,
        kernelDropToNewTrack,
        setKernelDropToNewTrack,
        kernelGhost,
        setKernelGhost,
        kernelGestureInteractionActiveRef,
        beginKernelGestureInteraction,
        endKernelGestureInteraction,
        kernelDragOriginRef,
    };
}
