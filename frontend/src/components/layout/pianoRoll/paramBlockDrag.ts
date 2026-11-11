import type { useLiveParamEditing } from "./useLiveParamEditing";
import type { ParamViewSegment } from "./types";
import { fetchFullResCurve, uploadFullResCurveSegments } from "./selectionEditData";
import { startPointerGesture } from "./gestures/pointerGesture";
import {
    applyFullResSegmentsToParamView,
    buildParamBlockDragSegments,
    buildParamBlockPreviewSegments,
    paramBlockFrameSpan,
    projectParamBlockDragDisplayValue,
    resolveParamBlockDragRelease,
    setParamBlockLogicalCenters,
    type ParamBlock,
    type ParamBlockDragCommitReleaseState,
    type ParamBlockMember,
    type ParamBlockState,
} from "./paramBlocks";
import { resolveParamBlockDragCenters } from "./paramBlockDragCenters";
import { ownsParamBlockGesture, type ParamAuthorityWatermark } from "./paramBlockTransaction";
import { sameParamBlockExternalEpoch, type ParamBlockEpoch } from "./paramBlockVisualTransaction";
import type { EdgeScrollDriver } from "./kernel/edgeScrollDriver";
import { blockMatchesLogicalId, logicalBlockId } from "./paramBlockTypes";
import { trackParamBlockCommit } from "./paramBlockCommitCoordinator";

type Ref<T> = { current: T };
type LiveEditing = ReturnType<typeof useLiveParamEditing>;
export type SnapModifiers = {
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
    metaKey?: boolean;
};

export type ParamBlockDragSnapshot = {
    readonly key: string;
    readonly epoch: ParamBlockEpoch;
    /**
     * Edit revision produced by this gesture's own commit. Mutable so the
     * commit path can keep the visual alive while a third-party structural
     * edit invalidates it immediately.
     */
    ownedEditRevision: number | null;
    readonly startedAt: number;
    readonly ids: ReadonlySet<string>;
    readonly members: readonly ParamBlockMember[];
    readonly blocks: readonly {
        readonly id: string;
        readonly userBlockId?: string;
        readonly startSec: number;
        readonly endSec: number;
        readonly anchorValue: number | null;
        readonly grouped: boolean;
    }[];
    /** Per-run preview centers for this frame. */
    frameCenters: ReadonlyMap<string, number>;
    /** Per-logical-block preview centers, shared by every run. */
    logicalCenters: ReadonlyMap<string, number>;
    /** Immutable start center per logical block. */
    readonly logicalStartValues: ReadonlyMap<string, number>;
    readonly baseValues: readonly number[];
    readonly startFrame: number;
    readonly stride: number;
    readonly framePeriodMs: number;
};

export type ParamBlockDragCommit = {
    readonly blockId: string;
    readonly gestureToken: number;
    readonly key: string;
    readonly startedAt: number;
    /** 手势冻结窗口的起点 / 步长。 */
    readonly startFrame: number;
    readonly stride: number;
    /** 手势开始时已经落地的后端参数 revision。 */
    readonly authorityRevisionAtStart: number;
    /** 写入成功时已发出的最大请求序号，排除写入前及写入中的陈旧响应。 */
    authorityRequestSeqAtCommit: number;
    /** 后端提交的逐帧值；权威窗口必须实际包含这些值才能撤预览。 */
    committedSegments?: readonly {
        readonly startFrame: number;
        readonly values: readonly number[];
    }[];
    failed?: boolean;
    committed: boolean;
    releaseState: ParamBlockDragCommitReleaseState;
};

export type ParamBlockDragActivity = { pointerGestureToken: number | null; pendingCommits: number };

/**
 * A failed commit may only roll back the state it started from or the state it
 * applied itself. A third-party structural edit during the async read must win.
 */
export function canRollbackParamBlockDragCommit(args: {
    readonly currentRevision: number;
    readonly beforeRevision: number;
    readonly appliedRevision: number | null;
}): boolean {
    return (
        args.currentRevision === args.beforeRevision ||
        (args.appliedRevision !== null && args.currentRevision === args.appliedRevision)
    );
}

export interface ParamBlockDragContext {
    readonly paramBlockDragActivityRef: Ref<ParamBlockDragActivity>;
    readonly paramView: ParamViewSegment | null;
    readonly rootTrackId: string | null;
    readonly editParam: string;
    readonly paramBlocksWithValues: readonly ParamBlock[];
    readonly paramBlockGestureSerialRef: Ref<number>;
    readonly paramBlockActiveGestureTokenRef: Ref<number | null>;
    readonly paramBlockDragCommitRef: Ref<ParamBlockDragCommit | null>;
    readonly paramBlockDragCommitReleaseRafRef: Ref<number | null>;
    readonly paramBlockInteractionCurveRef: Ref<ParamViewSegment | null>;
    readonly paramBlockDragSnapshotRef: Ref<ParamBlockDragSnapshot | null>;
    readonly paramBlockStateRef: Ref<ParamBlockState>;
    readonly paramBlockEpoch: ParamBlockEpoch;
    readonly getCurrentParamBlockEpoch?: () => ParamBlockEpoch;
    /**
     * 块层是否在管理当前曲线。
     *
     * 旧实现读 `paramBlockStateRef.current.view` —— 那是"后端回过一次视图"的
     * 副产物；派生视图之后，判据就是"有根轨道，块层参与合成"。
     */
    readonly curveOwned: boolean;
    readonly paramBlockCommitQueueRef: Ref<Promise<void>>;
    readonly paramViewRef: Ref<ParamViewSegment | null>;
    readonly liveEditActiveRef: Ref<boolean> | undefined;
    readonly applyDenseToLiveEdit: LiveEditing["applyDenseToLiveEdit"];
    readonly resetLiveEditPreview: LiveEditing["resetLiveEditPreview"];
    readonly invalidateParamBlockSpec: () => void;
    readonly requestWaveformRepaint: () => void;
    readonly paintNow: () => void;
    readonly bumpParamBlockStructure: () => void;
    readonly setParamBlockState: (state: ParamBlockState) => void;
    /**
     * 一次提交同时写**曲线补丁**与**块布局**。
     *
     * 【为什么必须一次】后端每次 `set_param_block_edits` 都会打一个撤销点；
     * 曲线与块中心分成两次写，就等于一次手势产生两步历史 —— 用户要按两次
     * 撤销才能回到原样。这里把最终编辑集一并交给提交方，让它们走同一个事务。
     */
    readonly commitLineCurve?: (
        state: ParamBlockState,
        segments: readonly { startFrame: number; values: readonly number[] }[],
        authorityRevision?: number,
        geometryRevision?: number,
        geometrySignature?: string,
    ) => Promise<unknown>;
    readonly getAuthorityWatermark: () => ParamAuthorityWatermark;
    readonly paramAuthorityRevision: number;
    readonly onParamCommitSucceeded: () => void;
    readonly bumpRefreshToken: () => void;
    readonly snapToggleHeldAt: (event: SnapModifiers) => boolean;
    /** 当前音高吸附总开关；与 snapToggleHeldAt 做 XOR 得到本次手势的有效吸附。 */
    /** Publishes/clears the horizontal pitch snap guide while the gesture runs. */
    readonly onError?: (error: unknown) => void;
    readonly resolveValueDelta: (
        startValue: number,
        rawDelta: number,
        snapToggleHeld: boolean,
    ) => number;
}

export type ParamBlockDragStart = {
    anchorBlockId: string;
    clientX: number;
    clientY: number;
    contextMenu: boolean;
    pointerId: number;
    captureTarget: HTMLCanvasElement;
    setActivePointerGestureEnd: (endGesture: () => void) => void;
    clearActivePointerGestureEnd: (endGesture?: () => void) => void;
    pointerValue: (clientY: number) => number;
    /** 本次按下后的选中集合（已含锚点）；首次拖动不依赖 React 提交。 */
    selection: readonly string[];
    /** 按下时屏幕上锚点块的真实代表值（来自宿主实时几何）。 */
    anchorValue?: number | null;
    /** All selected centers from the same rendered hit geometry. */
    centerValues?: ReadonlyMap<string, number>;
    /** 按下瞬间的取消吸附键状态（Shift 等，按设置绑定）。 */
    shiftKey?: boolean;
    altKey?: boolean;
    ctrlKey?: boolean;
    metaKey?: boolean;
    /** 未拖动（点击）时执行；用于 Shift 延迟范围选择 / 分割键松开检查。 */
    onClickWithoutDrag?: (event?: globalThis.PointerEvent | null) => void;
    /** 块手势共享的二维边缘滚动驱动。 */
    edgeScroll?: EdgeScrollDriver;
};

/** Owns one pointer gesture and its queued full-resolution commit. */
export function beginParamBlockDrag(
    args: ParamBlockDragStart,
    context: ParamBlockDragContext,
): boolean {
    const {
        paramBlockDragActivityRef,
        paramView,
        rootTrackId,
        editParam,
        paramBlocksWithValues,
        paramBlockGestureSerialRef,
        paramBlockActiveGestureTokenRef,
        paramBlockDragCommitRef,
        paramBlockDragCommitReleaseRafRef,
        paramBlockInteractionCurveRef,
        paramBlockDragSnapshotRef,
        paramBlockStateRef,
        paramBlockEpoch,
        getCurrentParamBlockEpoch,
        curveOwned,
        paramBlockCommitQueueRef,
        paramViewRef,
        liveEditActiveRef,
        applyDenseToLiveEdit,
        resetLiveEditPreview,
        invalidateParamBlockSpec,
        requestWaveformRepaint,
        paintNow,
        bumpParamBlockStructure,
        setParamBlockState,
        getAuthorityWatermark,
        paramAuthorityRevision,
        onParamCommitSucceeded,
        bumpRefreshToken,
        snapToggleHeldAt,
        onError,
        resolveValueDelta,
    } = context;
    const pv = paramView;
    if (!pv || !rootTrackId) return false;
    /*
     * 预览基准必须是**一次性快照**。
     *
     * 若直接引用 live.edit，预览会写回同一个数组，下一帧再把它当基准
     * 叠加一次 —— 位移逐帧累加，正是“越拖越大”。
     */
    const startBase = (() => {
        /*
         * 连续手势优先接在上一手势冻结后的曲线上；否则接当前权威
         * paramView。不能优先读 live：提交后的 live 可能仍是上一个
         * 事务的值，会让块移动而曲线从旧音高起算。
         */
        const interaction = paramBlockInteractionCurveRef.current;
        return interaction !== null && interaction.key === pv.key
            ? interaction.edit.slice()
            : pv.edit.slice();
    })();
    /*
     * 只拖本次手势参与的块：按下的锚点块；若它已在多选集合里，
     * 才把其余选中块一起带走。未选中的块绝不跟着整体平移。
     */
    /*
     * 锚点已在选区里才拖整个选区；否则只拖锚点。Ctrl 点击"移除"后
     * targetSelected=false，调用方不会启动拖动；这里仍保留这条规则，
     * 防止 Shift / 右键等路径误把未选中的块并进拖动。
     */
    /*
     * The selection can contain run ids or logical ids (Ctrl/Shift and the
     * context-menu paths use both). Normalize it to logical identities before
     * deciding which blocks participate; otherwise a run id that is not also
     * the parent id silently drops every sibling run.
     */
    const selectionLogicalIds = new Set<string>();
    for (const id of args.selection.map(String)) {
        const matches = paramBlocksWithValues.filter((block) => blockMatchesLogicalId(block, id));
        if (matches.length === 0) selectionLogicalIds.add(id);
        else for (const block of matches) selectionLogicalIds.add(logicalBlockId(block));
    }
    const anchorBlock = paramBlocksWithValues.find((block) =>
        blockMatchesLogicalId(block, args.anchorBlockId),
    );
    const anchorLogicalId = logicalBlockId(anchorBlock ?? { id: args.anchorBlockId });
    const anchorInSelection =
        selectionLogicalIds.has(anchorLogicalId) || selectionLogicalIds.has(args.anchorBlockId);
    const dragIds = new Set<string>(anchorInSelection ? selectionLogicalIds : [anchorLogicalId]);
    dragIds.add(anchorLogicalId);
    const selectedBlocks = paramBlocksWithValues.filter((block) =>
        dragIds.has(logicalBlockId(block)),
    );
    if (selectedBlocks.length === 0) return false;
    const gestureToken = ++paramBlockGestureSerialRef.current;
    /*
     * 成员的来源区间可能彼此重叠（自动归并块 / 手动合并块）。
     * 预览与提交都会把成员区间合并后逐帧写入；这里先冻结一次快照，
     * 整个手势只读它，避免拖动期间块重新派生时引用漂移。
     */
    const dragMembers = selectedBlocks.flatMap((block) =>
        (block.members.length === 0
            ? [{ clipId: rootTrackId, startSec: block.startSec, endSec: block.endSec }]
            : block.members
        ).map((member) => ({ ...member })),
    );
    const dragBlocks = [{ members: dragMembers }];
    /*
     * 不可在这里清空 hover 状态：拖动一开始，`kernelDataRef` 会被
     * 拖拽快照覆盖，快照里通常没有手柄。若同时把 hover 归零，已经被
     * 选中的块会在下一帧失去所有手柄，用户松开后也要等重新 hover
     * 才恢复。拖动结束时统一清理即可。
     */
    /*
     * Resolve one start center per logical block, then fan it out to every
     * visible run. The old implementation only resolved the anchor group, so
     * every other selected block had anchorValue=null and stayed still.
     */
    const dragCenters = resolveParamBlockDragCenters({
        selectedBlocks,
        anchorBlockId: args.anchorBlockId,
        anchorValue: args.anchorValue,
        centerValues: args.centerValues,
        param: editParam,
    });
    const dragSnapshotBlocks = selectedBlocks.map((block) => ({
        id: block.id,
        ...(block.userBlockId === undefined ? {} : { userBlockId: block.userBlockId }),
        startSec: block.startSec,
        endSec: block.endSec,
        anchorValue: dragCenters.byRun.get(block.id) ?? null,
        grouped: block.kind === "grouped",
    }));
    const snapshot: ParamBlockDragSnapshot = {
        key: pv.key,
        epoch: paramBlockEpoch,
        ownedEditRevision: null,
        startedAt: performance.now(),
        ids: new Set(dragIds),
        members: dragMembers.map((member) => ({ ...member })),
        blocks: dragSnapshotBlocks,
        frameCenters: new Map(dragCenters.byRun),
        logicalCenters: new Map(dragCenters.byLogical),
        logicalStartValues: new Map(dragCenters.byLogical),
        baseValues: startBase,
        startFrame: pv.startFrame,
        stride: pv.stride,
        framePeriodMs: pv.framePeriodMs,
    };
    let rollback: {
        token: number | null;
        snapshot: ParamBlockDragSnapshot | null;
        curve: ParamViewSegment | null;
        commit: ParamBlockDragCommit | null;
    } | null = null;
    const activate = () => {
        if (rollback !== null) return;
        rollback = {
            token: paramBlockActiveGestureTokenRef.current,
            snapshot: paramBlockDragSnapshotRef.current,
            curve: paramBlockInteractionCurveRef.current,
            commit: paramBlockDragCommitRef.current,
        };
        paramBlockActiveGestureTokenRef.current = gestureToken;
        /*
         * A new drag supersedes the previous visual handoff. The previous
         * backend write may still run, but its completion must not retire
         * this gesture's snapshot.
         */
        paramBlockDragCommitRef.current = null;
        if (paramBlockDragCommitReleaseRafRef.current !== null) {
            cancelAnimationFrame(paramBlockDragCommitReleaseRafRef.current);
            paramBlockDragCommitReleaseRafRef.current = null;
        }
        paramBlockInteractionCurveRef.current = { ...pv, edit: startBase };
        paramBlockDragSnapshotRef.current = snapshot;
        paramBlockDragActivityRef.current.pointerGestureToken = gestureToken;
        if (liveEditActiveRef) liveEditActiveRef.current = true;
    };
    const startClientY = args.clientY;
    /*
     * 拖动起点直接取块的代表值（主簇中心），不用指针落在块内的任意
     * y。否则“抓住块中部”会把整块先跳到指针位置，松手后的落点与
     * 按下时的块中心不一致。
     */
    /*
     * 块的中心在屏幕上不一定正好位于 `averageValue` 对应的曲线上
     * （代表值来自主簇，不是逐帧值）。因此锚点必须取按下瞬间的
     * **像素中心**，再由它换算成值；否则按下就会先跳一圈。
     */
    const anchorPointerValue = args.pointerValue(args.clientY);
    const dragAnchorValue =
        args.anchorValue !== null &&
        args.anchorValue !== undefined &&
        Number.isFinite(args.anchorValue)
            ? args.anchorValue
            : dragCenters.byLogical.get(anchorLogicalId);
    // No guessed midpoint: a block without a known center must not start a
    // gesture that would later snap every selected block to the viewport middle.
    if (dragAnchorValue === undefined || !Number.isFinite(dragAnchorValue)) return false;
    let didDrag = false;
    let lastDelta = 0;
    let lastClientY = args.clientY;
    let lastClientX = args.clientX;
    let lastSnapToggleHeld = snapToggleHeldAt({
        ctrlKey: args.ctrlKey === true,
        shiftKey: args.shiftKey === true,
        altKey: args.altKey === true,
        metaKey: args.metaKey === true,
    });
    let rafId: number | null = null;
    let gestureHandle: ReturnType<typeof startPointerGesture> | null = null;
    const framePeriodMs = pv.framePeriodMs;
    /** 指针在**值域纵轴**上的相对位移（只表达手指移动量）。 */
    const valueDeltaAtClientY = (clientY: number): number =>
        args.pointerValue(clientY) - anchorPointerValue;
    const snapDeltaAt = (event: SnapModifiers, rawDelta: number) =>
        resolveValueDelta(dragAnchorValue, rawDelta, snapToggleHeldAt(event));
    const preview = () => {
        if (rafId !== null) {
            cancelAnimationFrame(rafId);
            rafId = null;
        }
        activate();
        const pvNow = paramViewRef.current;
        /*
         * The interaction frame is anchored to the window captured at
         * pointerdown. A background fetch may replace paramView while
         * the user is still dragging; that must not cancel the preview.
         * Only the optional live mirror is window-specific.
         */
        const livePv = pvNow?.key === pv.key ? pvNow : null;
        const segments = buildParamBlockPreviewSegments({
            fillEmptyPitch: curveOwned,
            blocks: dragBlocks,
            framePeriodMs,
            param: editParam,
            anchorValue: dragAnchorValue,
            valueDelta: lastDelta,
            /*
             * 预览始终以**按下时**的已提交曲线为基准；拖动中 `pvNow`
             * 可能被别的手势 / 取数替换，但拖动位移必须叠加在同一份
             * 起点上，否则松手后的落点会漂移。
             */
            pv: {
                startFrame: pv.startFrame,
                stride: pv.stride,
                edit: pv.edit,
                orig: startBase,
            },
        });
        if (livePv !== null) {
            for (const segment of segments) {
                applyDenseToLiveEdit(
                    {
                        ...livePv,
                        startFrame: pv.startFrame,
                        stride: pv.stride,
                        edit: livePv.edit,
                        orig: startBase,
                    },
                    segment.startFrame,
                    segment.values,
                    pv.startFrame,
                    segment.startFrame + segment.values.length - 1,
                    "draw",
                );
            }
        }
        paramBlockInteractionCurveRef.current = {
            ...pv,
            edit: applyFullResSegmentsToParamView({
                paramView: { ...pv, edit: startBase },
                segments,
            }).edit,
        };
        const dragSnapshot = paramBlockDragSnapshotRef.current;
        if (dragSnapshot !== null) {
            const frameCenters = new Map<string, number>();
            for (const block of dragSnapshot.blocks) {
                /*
                 * `lastDelta` is already the snapped delta for the anchor.
                 * Re-snapping each block would move blocks by different
                 * amounts while the curve moves by the anchor delta.
                 */
                const center = projectParamBlockDragDisplayValue({
                    startValue: block.anchorValue,
                    valueDelta: lastDelta,
                    param: editParam,
                    snapToSemitone: false,
                });
                if (center !== null && Number.isFinite(center)) {
                    frameCenters.set(block.id, center);
                }
            }
            const logicalCenters = new Map<string, number>();
            for (const [logicalId, startValue] of dragSnapshot.logicalStartValues) {
                const center = projectParamBlockDragDisplayValue({
                    startValue,
                    valueDelta: lastDelta,
                    param: editParam,
                    snapToSemitone: false,
                });
                if (center !== null && Number.isFinite(center)) {
                    logicalCenters.set(logicalId, center);
                }
            }
            dragSnapshot.frameCenters = frameCenters;
            dragSnapshot.logicalCenters = logicalCenters;
        }
        invalidateParamBlockSpec();
        requestWaveformRepaint();
    };
    const schedule = () => {
        if (rafId == null) rafId = requestAnimationFrame(preview);
    };
    /*
     * 首个越过死区的移动立即写 live 并同任务提交，避免“先等一帧
     * 才出现结果”的起步延迟；后续高频 move 仍按 rAF 合并。
     */
    let pendingImmediatePreview = true;
    const onMove = (event: globalThis.PointerEvent) => {
        if ((event.buttons & 1) !== 1) {
            gestureHandle?.end("up", event);
            return;
        }
        const dx = event.clientX - args.clientX;
        const dy = event.clientY - startClientY;
        if (!didDrag && Math.hypot(dx, dy) <= 3) return;
        didDrag = true;
        /*
         * 拖动位移只由**指针相对按下点的移动**推出；锚点块中心与
         * 指针之间的初始偏移被保留，所以曲线与块不会在按下瞬间
         * 跳到指针位置，也不会把上一轮提交位置当作新基准。
         */
        const rawDelta = valueDeltaAtClientY(event.clientY);
        lastDelta = snapDeltaAt(event, rawDelta);
        lastClientY = event.clientY;
        lastClientX = event.clientX;
        lastSnapToggleHeld = snapToggleHeldAt(event);
        args.edgeScroll?.track(event.clientX, event.clientY, (clientX, clientY) => {
            lastClientX = clientX;
            if (clientY !== undefined) lastClientY = clientY;
            lastDelta = resolveValueDelta(
                dragAnchorValue,
                valueDeltaAtClientY(lastClientY),
                lastSnapToggleHeld,
            );
            preview();
        });
        if (pendingImmediatePreview) {
            pendingImmediatePreview = false;
            preview();
            paintNow();
            return;
        }
        schedule();
    };
    const finish = (
        reason: "up" | "cancel",
        releaseEvent: globalThis.PointerEvent | null = null,
    ) => {
        if (rafId != null) {
            cancelAnimationFrame(rafId);
            rafId = null;
        }
        args.edgeScroll?.stop();
        if (reason === "up") {
            const release = resolveParamBlockDragRelease({
                didDrag,
                startClientY,
                lastMoveClientY: lastClientY,
                releaseClientY: releaseEvent?.clientY ?? lastClientY,
                startClientX: args.clientX,
                lastMoveClientX: lastClientX,
                releaseClientX: releaseEvent?.clientX ?? lastClientX,
            });
            didDrag = release.didDrag;
            lastClientY = release.clientY;
            if (releaseEvent !== null) lastSnapToggleHeld = snapToggleHeldAt(releaseEvent);
        }
        if (reason === "cancel" || !didDrag) {
            /*
             * 点击（未拖动）必须在 rollback 早退之前回调：手势可能从未
             * activate（rollback === null），但 Shift 延迟选择仍然要落地。
             */
            if (reason === "up" && !didDrag) args.onClickWithoutDrag?.(releaseEvent);
            // A click has not acquired the visual transaction or the refresh guard.
            if (rollback === null) return;
            const previous = rollback.commit?.failed === true ? null : rollback;
            paramBlockActiveGestureTokenRef.current = previous?.token ?? null;
            paramBlockDragCommitRef.current = previous?.commit ?? null;
            paramBlockDragSnapshotRef.current = previous?.snapshot ?? null;
            paramBlockInteractionCurveRef.current = previous?.curve ?? null;
            paramBlockDragActivityRef.current.pointerGestureToken = null;
            resetLiveEditPreview(previous?.curve ?? paramViewRef.current ?? pv);
            if (previous?.commit?.committed) onParamCommitSucceeded();
            if (liveEditActiveRef) {
                liveEditActiveRef.current = paramBlockDragActivityRef.current.pendingCommits > 0;
            }
            bumpParamBlockStructure();
            invalidateParamBlockSpec();
            requestWaveformRepaint();
            return;
        }
        /*
         * 松手时不能再用可能延迟的 rAF 预览覆盖 live 曲线：直接按
         * 最后一次指针位置同步落到 live，然后提交。
         */
        lastDelta = resolveValueDelta(
            dragAnchorValue,
            valueDeltaAtClientY(lastClientY),
            lastSnapToggleHeld,
        );
        /*
         * 横向拖动越过死区后，纵向位移可能回到 0。此时没有任何帧值变化：
         * 不能生成一个空撤销点，也不能把它当成点击去触发分割。
         */
        if (Math.abs(lastDelta) < 1e-9) {
            if (rollback === null) return;
            const previous = rollback.commit?.failed === true ? null : rollback;
            paramBlockActiveGestureTokenRef.current = previous?.token ?? null;
            paramBlockDragCommitRef.current = previous?.commit ?? null;
            paramBlockDragSnapshotRef.current = previous?.snapshot ?? null;
            paramBlockInteractionCurveRef.current = previous?.curve ?? null;
            paramBlockDragActivityRef.current.pointerGestureToken = null;
            resetLiveEditPreview(previous?.curve ?? paramViewRef.current ?? pv);
            if (previous?.commit?.committed) onParamCommitSucceeded();
            if (liveEditActiveRef) {
                liveEditActiveRef.current = paramBlockDragActivityRef.current.pendingCommits > 0;
            }
            bumpParamBlockStructure();
            invalidateParamBlockSpec();
            requestWaveformRepaint();
            return;
        }
        preview();
        const finalSnapshot = paramBlockDragSnapshotRef.current;
        const logicalCenters = new Map<string, number>(finalSnapshot?.logicalCenters ?? []);
        const commitBlocks = dragBlocks;
        const commitDelta = lastDelta;
        /*
         * 落点只由“按下时看到的块中心 + 本次指针位移”决定。
         *
         * 不能从降采样 live 曲线重新求主簇：用户看到的是按下时的
         * 块中心，而提交会读取全分辨率曲线；两边各自求主簇会在低
         * 缩放时得到不同结果，松手后块像被拉到远处。
         */
        const commitCenterBlockId = args.anchorBlockId;
        const stateBeforeCommit = paramBlockStateRef.current;
        let appliedCommitState: ParamBlockState | null = null;
        const commitStartedAt = performance.now();
        const handoff: ParamBlockDragCommit = {
            blockId: commitCenterBlockId,
            gestureToken,
            key: pv.key,
            startedAt: commitStartedAt,
            startFrame: pv.startFrame,
            stride: pv.stride,
            authorityRevisionAtStart: paramAuthorityRevision,
            authorityRequestSeqAtCommit: getAuthorityWatermark().issuedRequestSeq,
            committed: false,
            releaseState: "waiting-for-curve",
        };
        paramBlockDragCommitRef.current = handoff;
        paramBlockDragActivityRef.current.pointerGestureToken = null;
        paramBlockDragActivityRef.current.pendingCommits += 1;
        const commitEpoch = paramBlockEpoch;
        const commitRun = paramBlockCommitQueueRef.current
            .catch(() => undefined)
            .then(async () => {
                try {
                    /*
                     * 提交必须按**全分辨率**曲线计算，不能复用拖动时的 pv（低缩放下
                     * 它是降采样的）。逐块取真实帧范围，再按同一帧切分合并成分段；
                     * `sourceAt` 由取回的逐帧值提供，避免把台阶写进原始曲线。
                     */
                    const ranges = commitBlocks.flatMap((block) =>
                        block.members.map((member) => {
                            const span = paramBlockFrameSpan(
                                member.startSec,
                                member.endSec,
                                framePeriodMs,
                            );
                            return {
                                startFrame: span.startFrame,
                                endFrame: span.endFrameExclusive - 1,
                            };
                        }),
                    );
                    const curves = await Promise.all(
                        ranges.map((range) =>
                            fetchFullResCurve({
                                trackId: rootTrackId,
                                param: editParam,
                                startFrame: range.startFrame,
                                endFrame: range.endFrame,
                                paramView: null,
                            }),
                        ),
                    );
                    if (
                        getCurrentParamBlockEpoch !== undefined &&
                        !sameParamBlockExternalEpoch(commitEpoch, getCurrentParamBlockEpoch())
                    ) {
                        throw new Error("Block drag commit cancelled: external history changed");
                    }
                    /*
                     * Never turn a missing / short read into zeros. A backend
                     * error or a truncated window must cancel the commit, not
                     * write a silent gap into the curve.
                     */
                    for (let index = 0; index < ranges.length; index += 1) {
                        const range = ranges[index];
                        const curve = curves[index];
                        const expected = range.endFrame - range.startFrame + 1;
                        if (
                            curve === undefined ||
                            curve.startFrame !== range.startFrame ||
                            curve.values.length !== expected
                        ) {
                            throw new Error(
                                "Block drag commit cancelled: incomplete full-resolution curve",
                            );
                        }
                    }
                    const sourceAt = (frame: number): number => {
                        for (let index = 0; index < ranges.length; index += 1) {
                            const range = ranges[index];
                            if (frame < range.startFrame || frame > range.endFrame) {
                                continue;
                            }
                            const value = curves[index]?.values[frame - range.startFrame];
                            if (value === undefined || !Number.isFinite(value)) {
                                throw new Error(
                                    "Block drag commit cancelled: missing source frame",
                                );
                            }
                            return value;
                        }
                        throw new Error("Block drag commit cancelled: source frame outside ranges");
                    };
                    /*
                     * MIDI-like semantics: the curve moves by the same
                     * explicit delta as the block. Do not re-cluster the
                     * full-resolution source here.
                     */
                    const fullResDelta = commitDelta;
                    const windows = buildParamBlockDragSegments({
                        fillEmptyPitch: curveOwned,
                        blocks: commitBlocks,
                        framePeriodMs,
                        param: editParam,
                        anchorValue: dragAnchorValue,
                        valueDelta: fullResDelta,
                        sourceAt,
                    });
                    const commitSegments = windows;
                    if (commitSegments.length === 0)
                        throw new Error("Block drag has no editable frames");
                    if (liveEditActiveRef) liveEditActiveRef.current = true;
                    /*
                     * A structural edit may have landed while the full-res
                     * reads were in flight. Never overwrite it with the
                     * gesture's pre-commit snapshot.
                     */
                    if (
                        stateBeforeCommit.editSet.revision !==
                        paramBlockStateRef.current.editSet.revision
                    ) {
                        throw new Error(
                            "Block drag commit cancelled: parameter blocks changed during read",
                        );
                    }
                    /*
                     * 【一次手势 = 一个历史步】Pitch 块的显示中心来自 `value`，
                     * 曲线位移只改帧值；两者都要落盘，但必须落在**同一个事务**里。
                     * 先把最终编辑集算出来，再连同曲线分段一起提交。
                     */
                    const commitState =
                        editParam === "pitch"
                            ? setParamBlockLogicalCenters({
                                  state: paramBlockStateRef.current,
                                  blocks: selectedBlocks,
                                  centers: logicalCenters,
                              })
                            : paramBlockStateRef.current;
                    if (commitState !== paramBlockStateRef.current) {
                        // 本地先落位：块的显示中心立刻到位，不必等回包。
                        appliedCommitState = commitState;
                        paramBlockStateRef.current = commitState;
                        setParamBlockState(commitState);
                        const dragSnapshot = paramBlockDragSnapshotRef.current;
                        if (dragSnapshot !== null) {
                            dragSnapshot.ownedEditRevision = commitState.editSet.revision;
                        }
                    }
                    if (context.commitLineCurve) {
                        const revision = curves[0]?.authorityRevision;
                        const geometry = curves[0]?.geometryRevision;
                        const signature = curves[0]?.geometrySignature;
                        if (
                            curves.some(
                                (curve) =>
                                    curve.authorityRevision !== revision ||
                                    (signature
                                        ? curve.geometrySignature !== signature
                                        : curve.geometryRevision !== geometry),
                            )
                        )
                            throw new Error("Conflict: parameter curve changed during block read");
                        await context.commitLineCurve(
                            commitState,
                            commitSegments,
                            revision,
                            geometry,
                            signature,
                        );
                    } else
                        await uploadFullResCurveSegments({
                            trackId: rootTrackId,
                            param: editParam,
                            segments: commitSegments,
                        });
                    // Record completion even when a newer gesture owns the visual frame:
                    // cancelling that gesture can restore this pending handoff.
                    handoff.committedSegments = commitSegments.map((segment) => ({
                        startFrame: segment.startFrame,
                        values: segment.values.slice(),
                    }));
                    handoff.authorityRequestSeqAtCommit = getAuthorityWatermark().issuedRequestSeq;
                    handoff.committed = true;
                    if (
                        ownsParamBlockGesture({
                            activeGestureToken: paramBlockActiveGestureTokenRef.current,
                            gestureToken,
                        })
                    )
                        onParamCommitSucceeded();
                } catch (error) {
                    handoff.failed = true;
                    console.error("[pianoRoll] block drag commit failed", error);
                    onError?.(error);
                    if (
                        ownsParamBlockGesture({
                            activeGestureToken: paramBlockActiveGestureTokenRef.current,
                            gestureToken,
                        })
                    ) {
                        // 曲线写失败时回滚本地 logicalCenters，但不能覆盖提交期间
                        // 已经落地的第三方结构编辑。只允许回滚到本手势的起点或
                        // 本手势自己刚应用的状态。
                        const current = paramBlockStateRef.current;
                        const canRollback =
                            current === stateBeforeCommit ||
                            (appliedCommitState !== null && current === appliedCommitState) ||
                            canRollbackParamBlockDragCommit({
                                currentRevision: current.editSet.revision,
                                beforeRevision: stateBeforeCommit.editSet.revision,
                                appliedRevision: appliedCommitState?.editSet.revision ?? null,
                            });
                        if (canRollback) setParamBlockState(stateBeforeCommit);
                        const pvNow = paramViewRef.current;
                        if (pvNow !== null) resetLiveEditPreview(pvNow);
                        paramBlockDragCommitRef.current = null;
                        paramBlockDragSnapshotRef.current = null;
                        paramBlockInteractionCurveRef.current = null;
                        paramBlockActiveGestureTokenRef.current = null;
                        bumpParamBlockStructure();
                    }
                } finally {
                    const activity = paramBlockDragActivityRef.current;
                    activity.pendingCommits -= 1;
                    if (
                        activity.pendingCommits === 0 &&
                        activity.pointerGestureToken === null &&
                        liveEditActiveRef
                    ) {
                        liveEditActiveRef.current = false;
                    }
                    /*
                     * live-deferral 在 liveEditActive 为 true 时不会取数；
                     * 必须在撤销保护后触发权威刷新。块快照只等这份权威数据，
                     * 不再用本地回显骗自己“已经落盘”。
                     */
                    bumpRefreshToken();
                    /*
                     * 快照不能在这里立刻释放。留在 ref 中，由下一次绘制
                     * 根据“提交完成 + 权威曲线窗口同步”统一收尾。
                     */
                    if (
                        ownsParamBlockGesture({
                            activeGestureToken: paramBlockActiveGestureTokenRef.current,
                            gestureToken,
                        })
                    ) {
                        invalidateParamBlockSpec();
                    }
                }
            });
        paramBlockCommitQueueRef.current = trackParamBlockCommit(
            "drag",
            rootTrackId,
            commitRun,
        ).then(() => undefined);
    };
    gestureHandle = startPointerGesture({
        pointerId: args.pointerId,
        captureTarget: args.captureTarget,
        onMove,
        onEnd: (reason, event) => finish(reason, event),
        registerEnd: args.setActivePointerGestureEnd,
        clearEnd: args.clearActivePointerGestureEnd,
    });
    return true;
}
