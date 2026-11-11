/**
 * 参数块过渡持久化。
 *
 * 从 `usePianoRollPanelModel` 拆出：过渡手柄的提交同时涉及布局、曲线补丁、
 * 权威 watermark 与视觉预览归属，集中在一个 hook 里比继续膨胀面板模型更
 * 容易审计。对外只暴露两个稳定入口：布局过渡 / Pitch 曲线过渡。
 *
 * 【关键约束】带 curve patches 的提交必须携带 `expectedCurveRevision`；否则
 * 后端会以 `Conflict: parameter curve changed` 拒绝整笔事务。
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
import { markParamBlockPreviewOwnedRevision } from "../paramBlockPreviewOwnership";
import { fetchFullResCurve } from "../selectionEditData";
import type { useParamBlockPersistence } from "../useParamBlockPersistence";

type Ref<T> = { current: T };
type CurvePatch = {
    readonly param: string;
    readonly startFrame: number;
    readonly values: readonly number[];
};

export function useParamBlockTransitionPersistence(args: {
    readonly rootTrackId: string | null;
    readonly editParam: string;
    readonly blockPersistence: Pick<ReturnType<typeof useParamBlockPersistence>, "commit">;
    readonly paramBlockStateRef: Ref<ParamBlockState>;
    readonly paramBlockViewRef: Ref<ParamBlockView | null>;
    readonly paramBlockSourceClipsRef: Ref<readonly ParamBlockSourceClip[]>;
    readonly paramViewRef: Ref<ParamViewSegment | null>;
    readonly setParamBlockState: Dispatch<SetStateAction<ParamBlockState>>;
    readonly setParamView: (next: ParamViewSegment) => void;
    readonly onParamCommitSucceeded: () => void;
    readonly bumpRefreshToken: () => void;
    readonly invalidate: () => void;
    readonly visualTransactionRef: Ref<{ ownedEditRevision?: number | null } | null>;
    readonly dragSnapshotRef: Ref<{ ownedEditRevision?: number | null } | null>;
    readonly resizeRef: Ref<{ ownedEditRevision?: number | null } | null>;
}): {
    persistBlockTransition: (
        left: ParamBlock,
        right: ParamBlock,
        widthSec: number,
        curvePatches?: readonly CurvePatch[],
        authority?: ParamCurveCommitAuthority | null,
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

    /** 只提交块过渡布局；所有参数都持久化 transitionOutSec。 */
    const persistBlockTransition = useCallback(
        async (
            left: ParamBlock,
            right: ParamBlock,
            widthSec: number,
            curvePatches: readonly CurvePatch[] = [],
            authority?: ParamCurveCommitAuthority | null,
        ): Promise<void> => {
            if (!rootTrackId) return;
            const state = paramBlockStateRef.current;
            if (paramBlockViewRef.current === null) return;
            const edit = setParamBlockTransition({
                state,
                left,
                right,
                transitionSec: widthSec,
                clips: paramBlockSourceClipsRef.current,
            });
            if (!edit.ok) throw new Error(edit.reason);
            setParamBlockState(edit.state);
            markOwned(edit.state.editSet.revision);
            await blockPersistence.commit(edit.state, true, curvePatches, authority, "transition");
            bumpRefreshToken();
            invalidate();
        },
        [
            blockPersistence,
            bumpRefreshToken,
            invalidate,
            markOwned,
            paramBlockSourceClipsRef,
            paramBlockStateRef,
            paramBlockViewRef,
            rootTrackId,
            setParamBlockState,
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
            const framePeriodMs = paramViewRef.current?.framePeriodMs ?? 5;
            const boundaryFrame = Math.round((right.startSec * 1000) / framePeriodMs);
            const widthFrames = Math.max(1, Math.round((widthSec * 1000) / framePeriodMs));
            const startFrame = Math.max(0, Math.floor((left.startSec * 1000) / framePeriodMs));
            const endFrame = Math.max(
                boundaryFrame,
                Math.round((right.endSec * 1000) / framePeriodMs) - 1,
            );
            const readStart = Math.max(startFrame, boundaryFrame - widthFrames);
            const readEnd = Math.min(endFrame, boundaryFrame + widthFrames - 1);
            const source = await fetchFullResCurve({
                trackId: rootTrackId,
                param: "pitch",
                startFrame: readStart,
                endFrame: readEnd,
                paramView: null,
                forceAuthority: true,
            });
            if (source.authorityRevision === undefined) {
                throw new Error("param block transition requires expectedCurveRevision");
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

            if (options?.restoreOriginal === true) {
                /*
                 * 撤销过渡必须写回**提交态**曲线，而不是当前 live 预览；
                 * 一次带 authority 的读 + 一次 width=0 的布局事务，保证
                 * 布局与曲线落在同一个撤销点。
                 */
                const restoreSegments = [{ startFrame: source.startFrame, values: source.values }];
                await persistBlockTransition(
                    left,
                    right,
                    0,
                    restoreSegments.map((segment) => ({ ...segment, param: "pitch" })),
                    authority,
                );
                const pvNow = paramViewRef.current;
                if (pvNow !== null) {
                    setParamView(
                        applyFullResSegmentsToParamView({
                            paramView: pvNow,
                            segments: restoreSegments,
                        }),
                    );
                }
                onParamCommitSucceeded();
                bumpRefreshToken();
                invalidate();
                return;
            }

            const leftValue = left.averageValue ?? 60;
            const rightValue = right.averageValue ?? 60;
            const committedAt = (frame: number): number =>
                Number(source.values[frame - source.startFrame]) || rightValue;
            const segments = buildParamBlockTransitionSegments({
                boundaryFrame,
                widthFrames,
                targetValue: rightValue,
                leftStartFrame: startFrame,
                rightEndFrame: endFrame,
                leftValue,
                rightValue,
                sourceAt: committedAt,
            });
            await persistBlockTransition(
                left,
                right,
                widthSec,
                segments.map((segment) => ({ ...segment, param: "pitch" })),
                authority,
            );
        },
        [
            bumpRefreshToken,
            editParam,
            invalidate,
            onParamCommitSucceeded,
            paramViewRef,
            persistBlockTransition,
            rootTrackId,
            setParamView,
        ],
    );

    return { persistBlockTransition, persistPitchTransitionCurve };
}
