import type { useLiveParamEditing } from "./useLiveParamEditing";
import type { ParamViewSegment } from "./types";
import { fetchFullResCurve, readPvRange, uploadFullResCurveSegments } from "./selectionEditData";
import { startPointerGesture } from "./gestures/pointerGesture";
import {
    applyFullResSegmentsToParamView,
    buildParamBlockDragSegments,
    buildParamBlockPreviewSegments,
    projectParamBlockDragDisplayValue,
    resolveParamBlockCommitDisplayCenter,
    resolveParamBlockDragRelease,
    setParamBlockLogicalCenters,
    type ParamBlock,
    type ParamBlockDragCommitReleaseState,
    type ParamBlockMember,
    type ParamBlockState,
} from "./paramBlocks";
import { buildParamBlockFrameCenters } from "./paramBlockFrame";
import { ownsParamBlockGesture, type ParamAuthorityWatermark } from "./paramBlockTransaction";
import { pitchScaleFactorFromDrag, scalePitchAroundCenter } from "./paramBlockPitch";

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
    readonly ids: ReadonlySet<string>;
    readonly members: readonly ParamBlockMember[];
    readonly blocks: readonly {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
        readonly anchorValue: number | null;
        readonly merged: boolean;
    }[];
    centerValues: ReadonlyMap<string, number>;
    readonly baseValues: readonly number[];
    readonly startFrame: number;
    readonly stride: number;
    readonly framePeriodMs: number;
};

export type ParamBlockDragCommit = {
    readonly blockId: string;
    readonly gestureToken: number;
    readonly center: number;
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
    readonly persistParamBlockEdits: (state: ParamBlockState) => void;
    readonly commitLineCurve?: (
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
    /** Alt+上下拖：缩放块内音高（中心不动），而不是平移块。 */
    pitchScale?: boolean;
    /** 按下瞬间的取消吸附键状态（Shift 等，按设置绑定）。 */
    shiftKey?: boolean;
    altKey?: boolean;
};

/** Owns one pointer gesture and its queued full-resolution commit. */
export function beginParamBlockDrag(
    args: ParamBlockDragStart,
    context: ParamBlockDragContext,
): void {
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
        persistParamBlockEdits,
        getAuthorityWatermark,
        paramAuthorityRevision,
        onParamCommitSucceeded,
        bumpRefreshToken,
        snapToggleHeldAt,
        resolveValueDelta,
    } = context;
    const pv = paramView;
    if (!pv || !rootTrackId) return;
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
    const dragIds = new Set<string>(args.selection);
    dragIds.add(args.anchorBlockId);
    const selectedBlocks = paramBlocksWithValues.filter((block) => dragIds.has(block.id));
    if (selectedBlocks.length === 0) return;
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
    const pitchScaleDrag = editParam === "pitch" && args.pitchScale === true;
    const pitchScaleCenter =
        selectedBlocks.length === 1
            ? resolveParamBlockCommitDisplayCenter({
                  rawCenter:
                      args.anchorValue !== undefined &&
                      args.anchorValue !== null &&
                      Number.isFinite(args.anchorValue)
                          ? args.anchorValue
                          : (selectedBlocks[0]?.averageValue ?? null),
                  param: editParam,
                  continuous: false,
              })
            : null;
    const pitchScaleValues = pitchScaleDrag ? startBase.slice() : null;
    /*
     * 不可在这里清空 hover 状态：拖动一开始，`kernelDataRef` 会被
     * 拖拽快照覆盖，快照里通常没有手柄。若同时把 hover 归零，已经被
     * 选中的块会在下一帧失去所有手柄，用户松开后也要等重新 hover
     * 才恢复。拖动结束时统一清理即可。
     */
    const dragSnapshotBlocks = selectedBlocks.map((block) => ({
        id: block.id,
        startSec: block.startSec,
        endSec: block.endSec,
        /*
         * 锚点块使用宿主本帧命中几何的代表值，和用户按下时看到的
         * 矩形完全同源；其余多选块沿用各自派生代表值。
         */
        anchorValue:
            block.id === args.anchorBlockId &&
            args.anchorValue !== undefined &&
            args.anchorValue !== null &&
            Number.isFinite(args.anchorValue)
                ? args.anchorValue
                : (args.centerValues?.get(block.id) ??
                  paramBlockDragSnapshotRef.current?.centerValues.get(block.id) ??
                  (editParam === "pitch" ? block.manualValue : null) ??
                  block.averageValue),
        merged: block.kind === "merged",
    }));
    const snapshot: ParamBlockDragSnapshot = {
        key: pv.key,
        ids: new Set(dragIds),
        members: dragMembers.map((member) => ({ ...member })),
        blocks: dragSnapshotBlocks,
        centerValues: buildParamBlockFrameCenters({
            blocks: dragSnapshotBlocks,
            valueDelta: 0,
            param: editParam,
            snapToSemitone: false,
        }),
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
    const anchorBlock = selectedBlocks.find((block) => block.id === args.anchorBlockId);
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
            : anchorBlock?.averageValue !== null &&
                anchorBlock?.averageValue !== undefined &&
                Number.isFinite(anchorBlock.averageValue)
              ? anchorBlock.averageValue
              : anchorPointerValue;
    let didDrag = false;
    let lastDelta = 0;
    let lastClientY = args.clientY;
    let lastSnapToggleHeld = snapToggleHeldAt({
        ctrlKey: false,
        shiftKey: args.shiftKey === true,
        altKey: args.altKey === true,
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
        rafId = null;
        activate();
        const pvNow = paramViewRef.current;
        /*
         * The interaction frame is anchored to the window captured at
         * pointerdown. A background fetch may replace paramView while
         * the user is still dragging; that must not cancel the preview.
         * Only the optional live mirror is window-specific.
         */
        const livePv = pvNow?.key === pv.key ? pvNow : null;
        const scaleValues =
            pitchScaleValues === null
                ? null
                : (() => {
                      const startFrame = Math.max(
                          0,
                          Math.round(((selectedBlocks[0]?.startSec ?? 0) * 1000) / framePeriodMs),
                      );
                      const endFrame = Math.max(
                          startFrame,
                          Math.round(
                              ((selectedBlocks.at(-1)?.endSec ?? 0) * 1000) / framePeriodMs,
                          ) - 1,
                      );
                      return scalePitchAroundCenter({
                          values: readPvRange({ ...pv, edit: startBase }, startFrame, endFrame),
                          center: pitchScaleCenter,
                          factor: pitchScaleFactorFromDrag(lastDelta * 24),
                      });
                  })();
        if (scaleValues !== null) {
            const startFrame = Math.max(
                0,
                Math.round(((selectedBlocks[0]?.startSec ?? 0) * 1000) / framePeriodMs),
            );
            /*
             * Alt 缩放只更新 live 草稿，不能每帧把预览写进权威
             * `paramView`。后者会触发取数、重新派生和另一条异步重绘，
             * 表现为块上下抖动。块中心由 displayCenterLock 冻结。
             */
            if (livePv !== null) {
                resetLiveEditPreview(livePv);
                applyDenseToLiveEdit(
                    livePv,
                    startFrame,
                    scaleValues,
                    pv.startFrame,
                    startFrame + scaleValues.length - 1,
                    "draw",
                );
            }
            paramBlockInteractionCurveRef.current = {
                ...pv,
                edit: applyFullResSegmentsToParamView({
                    paramView: { ...pv, edit: startBase },
                    segments: [{ startFrame, values: scaleValues }],
                }).edit,
            };
            const scaleSnapshot = paramBlockDragSnapshotRef.current;
            if (scaleSnapshot !== null) {
                scaleSnapshot.centerValues = buildParamBlockFrameCenters({
                    blocks: scaleSnapshot.blocks,
                    valueDelta: 0,
                    param: editParam,
                    snapToSemitone: false,
                });
            }
            /*
             * The interaction curve changed in place. Rebuild the block
             * frame from that exact curve before the next paint.
             */
            invalidateParamBlockSpec();
            requestWaveformRepaint();
            return;
        }
        const segments = buildParamBlockPreviewSegments({
            fillEmptyPitch: Boolean(paramBlockStateRef.current.view),
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
            dragSnapshot.centerValues = buildParamBlockFrameCenters({
                blocks: dragSnapshot.blocks,
                valueDelta: lastDelta,
                param: editParam,
                snapToSemitone: !lastSnapToggleHeld,
            });
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
        const dy = event.clientY - startClientY;
        if (!didDrag && Math.abs(dy) <= 3) return;
        didDrag = true;
        /*
         * 拖动位移只由**指针相对按下点的移动**推出；锚点块中心与
         * 指针之间的初始偏移被保留，所以曲线与块不会在按下瞬间
         * 跳到指针位置，也不会把上一轮提交位置当作新基准。
         */
        const rawDelta = valueDeltaAtClientY(event.clientY);
        /*
         * Alt 缩放必须连续。以前这里也套用音高半音吸附，缩放因子
         * 按指数放大量级后会在相邻半音门槛间大幅跳变，正是抖动来源。
         */
        lastDelta = pitchScaleDrag ? rawDelta : snapDeltaAt(event, rawDelta);
        lastClientY = event.clientY;
        lastSnapToggleHeld = snapToggleHeldAt(event);
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
        if (reason === "up") {
            const release = resolveParamBlockDragRelease({
                didDrag,
                startClientY,
                lastMoveClientY: lastClientY,
                releaseClientY: releaseEvent?.clientY ?? lastClientY,
            });
            didDrag = release.didDrag;
            lastClientY = release.clientY;
            if (releaseEvent !== null) lastSnapToggleHeld = snapToggleHeldAt(releaseEvent);
        }
        if (reason === "cancel" || !didDrag) {
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
        lastDelta = pitchScaleDrag
            ? valueDeltaAtClientY(lastClientY)
            : resolveValueDelta(
                  dragAnchorValue,
                  valueDeltaAtClientY(lastClientY),
                  lastSnapToggleHeld,
              );
        preview();
        const finalSnapshot = paramBlockDragSnapshotRef.current;
        const logicalCenters = new Map<string, number>();
        if (finalSnapshot !== null) {
            for (const block of finalSnapshot.blocks) {
                const center = pitchScaleDrag
                    ? block.anchorValue
                    : projectParamBlockDragDisplayValue({
                          startValue: block.anchorValue,
                          valueDelta: lastDelta,
                          param: editParam,
                          snapToSemitone: !lastSnapToggleHeld,
                      });
                if (center !== null && Number.isFinite(center)) {
                    logicalCenters.set(block.id, center);
                }
            }
        }
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
        const rawCommitCenter = pitchScaleDrag ? null : dragAnchorValue + commitDelta;
        /*
         * 交接锁存必须使用**最终显示中心**，而不是未吸附的曲线主簇。
         * 普通绘制会把 pitch 中心吸到最近半音行心；若这里锁住 60.4，
         * 松手后曲线到位的瞬间就会先显示 60.4、再跳到 60，形成闪烁。
         */
        const commitCenter = resolveParamBlockCommitDisplayCenter({
            rawCenter: rawCommitCenter,
            param: editParam,
            continuous: pitchScaleDrag || lastSnapToggleHeld,
        });
        const commitStartedAt = performance.now();
        const handoff: ParamBlockDragCommit = {
            blockId: commitCenterBlockId,
            gestureToken,
            center: commitCenter ?? 0,
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
                            const startFrame = Math.max(
                                0,
                                Math.floor((member.startSec * 1000) / framePeriodMs),
                            );
                            const endFrame = Math.max(
                                startFrame,
                                Math.ceil((member.endSec * 1000) / framePeriodMs) - 1,
                            );
                            return { startFrame, endFrame };
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
                    const sourceAt = (frame: number): number => {
                        for (let index = 0; index < ranges.length; index += 1) {
                            const range = ranges[index];
                            if (frame < range.startFrame || frame > range.endFrame) {
                                continue;
                            }
                            return Number(curves[index]?.values[frame - range.startFrame]) || 0;
                        }
                        return 0;
                    };
                    /*
                     * MIDI-like semantics: the curve moves by the same
                     * explicit delta as the block. Do not re-cluster the
                     * full-resolution source here.
                     */
                    const fullResDelta = commitDelta;
                    const windows = buildParamBlockDragSegments({
                        fillEmptyPitch: Boolean(paramBlockStateRef.current.view),
                        blocks: commitBlocks,
                        framePeriodMs,
                        param: editParam,
                        anchorValue: dragAnchorValue,
                        valueDelta: fullResDelta,
                        sourceAt,
                    });
                    let scaleWindows: Array<{ startFrame: number; values: number[] }> | null = null;
                    if (pitchScaleValues !== null) {
                        const block = selectedBlocks[0];
                        if (block !== undefined) {
                            const startFrame = Math.max(
                                0,
                                Math.floor((block.startSec * 1000) / framePeriodMs),
                            );
                            const endFrame = Math.max(
                                startFrame,
                                Math.ceil((block.endSec * 1000) / framePeriodMs) - 1,
                            );
                            const curve = await fetchFullResCurve({
                                trackId: rootTrackId,
                                param: "pitch",
                                startFrame,
                                endFrame,
                                paramView: null,
                            });
                            scaleWindows = [
                                {
                                    startFrame,
                                    values: scalePitchAroundCenter({
                                        values: curve.values,
                                        center: pitchScaleCenter,
                                        factor: pitchScaleFactorFromDrag(commitDelta * 24),
                                    }),
                                },
                            ];
                        } else {
                            scaleWindows = [];
                        }
                    }
                    const commitSegments = scaleWindows ?? windows;
                    if (commitSegments.length === 0)
                        throw new Error("Block drag has no editable frames");
                    if (liveEditActiveRef) liveEditActiveRef.current = true;
                    if (paramBlockStateRef.current.view && context.commitLineCurve) {
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
                    if (editParam === "pitch") {
                        const nextState = setParamBlockLogicalCenters({
                            state: paramBlockStateRef.current,
                            blocks: selectedBlocks,
                            centers: logicalCenters,
                        });
                        if (nextState !== paramBlockStateRef.current) {
                            paramBlockStateRef.current = nextState;
                            setParamBlockState(nextState);
                            persistParamBlockEdits(nextState);
                        }
                    }
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
                    if (
                        ownsParamBlockGesture({
                            activeGestureToken: paramBlockActiveGestureTokenRef.current,
                            gestureToken,
                        })
                    ) {
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
        paramBlockCommitQueueRef.current = commitRun;
    };
    gestureHandle = startPointerGesture({
        pointerId: args.pointerId,
        captureTarget: args.captureTarget,
        onMove,
        onEnd: (reason, event) => finish(reason, event),
        registerEnd: args.setActivePointerGestureEnd,
        clearEnd: args.clearActivePointerGestureEnd,
    });
}
