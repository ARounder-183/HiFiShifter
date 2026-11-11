/**
 * 参数块过渡持久化。
 *
 * 从 `usePianoRollPanelModel` 拆出：过渡手柄的提交同时涉及布局、曲线补丁、
 * 权威 watermark 与视觉预览归属，集中在一个 hook 里比继续膨胀面板模型更
 * 容易审计。对外只暴露两个稳定入口：布局过渡 / Pitch 曲线过渡。
 *
 * 【关键约束】带 curve patches 的提交必须携带 `expectedCurveRevision`；否则
 * 后端会以 `Conflict: parameter curve changed` 拒绝整笔事务。
 *
 * 【为什么读基准在队列槽里】曲线代次是进程全局计数，任何无关写入都会让它前进。
 * 旧实现先 `fetchFullResCurve` 拿 authority、再调 `commit` 排队，两者之间只要
 * 有一次别的写入，这次过渡就必然以 `curve` 冲突失败 —— 而旧的重试只会原样重放
 * 同一份 authority，重试也救不回来。现在整段（读基准 → 算补丁 → 写）都在
 * `commitPlan` 的槽内，冲突时队列会再调一次 `prepare` 完成重基。
 *
 * 【违反边界的症状】把读挪回槽外 ⇒ 拖完块立刻拖过渡手柄会偶发「参数块操作失败」。
 */
import { useCallback } from "react";
import type { Dispatch, SetStateAction } from "react";

import type { ParamBlockView } from "../paramBlockTypes";
import type { ParamBlockSourceClip } from "../paramBlockProjection";
import type { ParamViewSegment } from "../types";
import {
    applyFullResSegmentsToParamView,
    buildParamBlockTransitionSegments,
    setParamBlockTransition,
    type ParamBlock,
    type ParamBlockState,
} from "../paramBlocks";
import type { ParamCurveCommitAuthority } from "../paramBlockCommitPolicy";
import { ParamBlockCommitError } from "../paramBlockCommitError";
import { markParamBlockPreviewOwnedRevision } from "../paramBlockPreviewOwnership";
import { fetchFullResCurve } from "../selectionEditData";
import type { ParamBlockWritePreparation } from "../paramBlockWritePlan";
import type { useParamBlockPersistence } from "../useParamBlockPersistence";

type Ref<T> = { current: T };

export function useParamBlockTransitionPersistence(args: {
    readonly rootTrackId: string | null;
    readonly editParam: string;
    readonly blockPersistence: Pick<ReturnType<typeof useParamBlockPersistence>, "commitPlan">;
    readonly paramBlockStateRef: Ref<ParamBlockState>;
    readonly paramBlockViewRef: Ref<ParamBlockView | null>;
    readonly paramBlockSourceClipsRef: Ref<readonly ParamBlockSourceClip[]>;
    readonly paramViewRef: Ref<ParamViewSegment | null>;
    readonly setParamBlockState: Dispatch<SetStateAction<ParamBlockState>>;
    readonly setParamView: (next: ParamViewSegment) => void;
    readonly onParamCommitSucceeded: () => void;
    readonly bumpRefreshToken: () => void;
    readonly invalidate: () => void;
    /**
     * 把「这次提交依据的权威水位 + 提交的曲线片段」写到当前预览事务上。
     *
     * 【为什么必须做】过渡手柄的**几何根本不变化**，所以旧释放判据（几何相等）
     * 在 pointerup 的同一次渲染就成立 —— 预览被撤、曲线回旧值，用户看到
     * 「先回弹、再生效」。写上门槛之后，释放只能发生在权威曲线可见之后。
     */
    readonly stampCurveCommit: (args: {
        readonly segments: readonly { startFrame: number; values: readonly number[] }[];
        readonly authority: ParamCurveCommitAuthority | null | undefined;
    }) => void;
    readonly visualTransactionRef: Ref<{ ownedEditRevision?: number | null } | null>;
    readonly dragSnapshotRef: Ref<{ ownedEditRevision?: number | null } | null>;
    readonly resizeRef: Ref<{ ownedEditRevision?: number | null } | null>;
}): {
    persistBlockTransition: (
        left: ParamBlock,
        right: ParamBlock,
        widthSec: number,
    ) => Promise<void>;
    persistPitchTransitionCurve: (
        left: ParamBlock,
        right: ParamBlock,
        widthSec: number,
        options?: { readonly restoreOriginal?: boolean },
    ) => Promise<void>;
} {
    const {
        rootTrackId,
        editParam,
        blockPersistence,
        paramBlockStateRef,
        paramBlockViewRef,
        paramBlockSourceClipsRef,
        paramViewRef,
        setParamBlockState,
        setParamView,
        onParamCommitSucceeded,
        bumpRefreshToken,
        invalidate,
        stampCurveCommit,
        visualTransactionRef,
        dragSnapshotRef,
        resizeRef,
    } = args;

    const markOwned = useCallback(
        (revision: number) => {
            markParamBlockPreviewOwnedRevision(
                {
                    visualTransaction: visualTransactionRef,
                    dragSnapshot: dragSnapshotRef,
                    resize: resizeRef,
                },
                revision,
            );
        },
        [dragSnapshotRef, resizeRef, visualTransactionRef],
    );

    /**
     * 把「设置过渡宽度」这一步做成可重跑的闭包。
     *
     * 【为什么需要 base】重基时本地状态已经被上一次尝试改过；必须始终从**本次
     * 手势开始前**的编辑集重算，否则第二次会叠加上一次的结果。
     */
    const planTransition = useCallback(
        (
            left: ParamBlock,
            right: ParamBlock,
            widthSec: number,
            withCurve: boolean,
            restoreOriginal: boolean,
        ): {
            readonly prepare: () => Promise<ParamBlockWritePreparation>;
            /** 本槽最近一次准备算出的曲线片段（重基时会刷新）。 */
            readonly segments: { current: readonly { startFrame: number; values: number[] }[] };
        } => {
            let base: ParamBlockState | null = null;
            const computed: { current: readonly { startFrame: number; values: number[] }[] } = {
                current: [],
            };
            return {
                segments: computed,
                prepare: async () => {
                    if (base === null) base = paramBlockStateRef.current;
                    const edit = setParamBlockTransition({
                        state: base,
                        left,
                        right,
                        transitionSec: widthSec,
                        clips: paramBlockSourceClipsRef.current,
                    });
                    if (!edit.ok) {
                        throw new ParamBlockCommitError({
                            kind: "rejected",
                            code: edit.reason,
                            detail: edit.reason,
                        });
                    }
                    /*
                     * 本地先落位：过渡宽度立刻到位，不必等回包。
                     * 单一写入者：只通过 hook 的 setState 更新。
                     */
                    setParamBlockState(edit.state);
                    markOwned(edit.state.editSet.revision);
                    if (!withCurve) {
                        return { state: edit.state, checkpoint: true, patches: [] };
                    }
                    const framePeriodMs = paramViewRef.current?.framePeriodMs ?? 5;
                    const boundaryFrame = Math.round((right.startSec * 1000) / framePeriodMs);
                    const widthFrames = Math.max(1, Math.round((widthSec * 1000) / framePeriodMs));
                    const startFrame = Math.max(
                        0,
                        Math.floor((left.startSec * 1000) / framePeriodMs),
                    );
                    const endFrame = Math.max(
                        boundaryFrame,
                        Math.round((right.endSec * 1000) / framePeriodMs) - 1,
                    );
                    const readStart = Math.max(startFrame, boundaryFrame - widthFrames);
                    const readEnd = Math.min(endFrame, boundaryFrame + widthFrames - 1);
                    const source = await fetchFullResCurve({
                        trackId: rootTrackId as string,
                        param: "pitch",
                        startFrame: readStart,
                        endFrame: readEnd,
                        paramView: null,
                        forceAuthority: true,
                    });
                    if (source.authorityRevision === undefined) {
                        throw new ParamBlockCommitError({
                            kind: "rejected",
                            code: "missing_curve_authority",
                            detail: "param block transition requires expectedCurveRevision",
                        });
                    }
                    const authority: ParamCurveCommitAuthority = {
                        expectedCurveRevision: source.authorityRevision,
                        ...(source.geometryRevision === undefined
                            ? {}
                            : { expectedGeometryRevision: source.geometryRevision }),
                        ...(source.geometrySignature === undefined
                            ? {}
                            : { expectedGeometrySignature: source.geometrySignature }),
                    };
                    const segments = restoreOriginal
                        ? [{ startFrame: source.startFrame, values: source.values.slice() }]
                        : buildParamBlockTransitionSegments({
                              boundaryFrame,
                              widthFrames,
                              targetValue: right.averageValue ?? 60,
                              leftStartFrame: startFrame,
                              rightEndFrame: endFrame,
                              leftValue: left.averageValue ?? 60,
                              rightValue: right.averageValue ?? 60,
                              sourceAt: (frame: number): number =>
                                  Number(source.values[frame - source.startFrame]) ||
                                  (right.averageValue ?? 60),
                          });
                    computed.current = segments;
                    stampCurveCommit({ segments, authority });
                    return {
                        state: edit.state,
                        checkpoint: true,
                        patches: segments.map((segment) => ({ ...segment, param: "pitch" })),
                        authority,
                    };
                },
            };
        },
        [
            markOwned,
            paramBlockSourceClipsRef,
            paramBlockStateRef,
            paramViewRef,
            rootTrackId,
            setParamBlockState,
            stampCurveCommit,
        ],
    );

    /** 只提交块过渡布局；所有参数都持久化 transitionOutSec。 */
    const persistBlockTransition = useCallback(
        async (left: ParamBlock, right: ParamBlock, widthSec: number): Promise<void> => {
            if (!rootTrackId) return;
            if (paramBlockViewRef.current === null) return;
            await blockPersistence.commitPlan({
                kind: "transition",
                rootTrackId,
                ...planTransition(left, right, widthSec, false, false),
            });
            bumpRefreshToken();
            invalidate();
        },
        [
            blockPersistence,
            bumpRefreshToken,
            invalidate,
            paramBlockViewRef,
            planTransition,
            rootTrackId,
        ],
    );

    /** Pitch 额外写曲线；非 Pitch 只走上方的布局提交。 */
    const persistPitchTransitionCurve = useCallback(
        async (
            left: ParamBlock,
            right: ParamBlock,
            widthSec: number,
            options?: { readonly restoreOriginal?: boolean },
        ): Promise<void> => {
            if (!rootTrackId || editParam !== "pitch") return;
            if (widthSec <= 1e-6 && options?.restoreOriginal !== true) return;
            const restoreOriginal = options?.restoreOriginal === true;
            const plan = planTransition(left, right, widthSec, true, restoreOriginal);
            await blockPersistence.commitPlan({ kind: "transition", rootTrackId, ...plan });
            if (restoreOriginal) {
                /*
                 * 撤销过渡必须把**提交态**曲线同步到显示（而不是留在 live 预览上）。
                 * 片段就是刚才真正写进去的那一份 —— 不再多读一次 IPC。
                 */
                const pvNow = paramViewRef.current;
                if (pvNow !== null && plan.segments.current.length > 0) {
                    setParamView(
                        applyFullResSegmentsToParamView({
                            paramView: pvNow,
                            segments: plan.segments.current,
                        }),
                    );
                }
            }
            onParamCommitSucceeded();
            bumpRefreshToken();
            invalidate();
        },
        [
            blockPersistence,
            bumpRefreshToken,
            editParam,
            invalidate,
            onParamCommitSucceeded,
            paramViewRef,
            planTransition,
            rootTrackId,
            setParamView,
        ],
    );

    return { persistBlockTransition, persistPitchTransitionCurve };
}
