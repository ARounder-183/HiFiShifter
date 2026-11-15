import { useEffect, type RefObject } from "react";
import type { ParamBlockDragCommit, ParamBlockDragSnapshot } from "./paramBlockDrag";
import {
    PARAM_BLOCK_PREVIEW_RELEASE_TIMEOUT_MS,
    isParamBlockAuthorityVisible,
} from "./paramBlockPreviewRelease";
import {
    ownsParamBlockGesture,
    type ParamAuthoritySnapshot,
    type ParamAuthorityWatermark,
} from "./paramBlockTransaction";
import type { ParamViewSegment } from "./types";

export interface ParamBlockDragHandoffInputs {
    readonly commitRef: RefObject<ParamBlockDragCommit | null>;
    readonly snapshotRef: RefObject<ParamBlockDragSnapshot | null>;
    readonly interactionCurveRef: RefObject<ParamViewSegment | null>;
    readonly activeGestureTokenRef: RefObject<number | null>;
    readonly releaseRafRef: RefObject<number | null>;
    readonly paramView: ParamViewSegment | null;
    /** 已落地的权威曲线代次（后端单位）。 */
    readonly authority: ParamAuthoritySnapshot;
    readonly getAuthorityWatermark: () => ParamAuthorityWatermark;
    readonly structureVersion: number;
    readonly clearLiveEditOverride: () => void;
    readonly onReleased: () => void;
    /** 提交等待硬超时；超时后回到 canonical，绝不永久锁住旧快照。 */
    readonly timeoutMs?: number;
}

/** Retire a block gesture only after a matching post-commit window can take over. */
export function useParamBlockDragHandoff({
    commitRef,
    snapshotRef,
    interactionCurveRef,
    activeGestureTokenRef,
    releaseRafRef,
    paramView,
    authority,
    getAuthorityWatermark,
    structureVersion,
    clearLiveEditOverride,
    onReleased,
    timeoutMs = PARAM_BLOCK_PREVIEW_RELEASE_TIMEOUT_MS,
}: ParamBlockDragHandoffInputs): void {
    useEffect(() => {
        const pending = commitRef.current;
        if (pending === null || pending.releaseState === "released") return;
        // The hard timeout only starts after the write has settled. Releasing a
        // still-in-flight commit would expose the old canonical geometry and
        // make the block jump back until the write lands.
        if (!pending.committed) return;
        const timer = setTimeout(
            () => {
                if (
                    commitRef.current !== pending ||
                    !ownsParamBlockGesture({
                        activeGestureToken: activeGestureTokenRef.current,
                        gestureToken: pending.gestureToken,
                    })
                ) {
                    return;
                }
                clearLiveEditOverride();
                pending.releaseState = "released";
                commitRef.current = null;
                snapshotRef.current = null;
                interactionCurveRef.current = null;
                activeGestureTokenRef.current = null;
                onReleased();
            },
            Math.max(0, timeoutMs),
        );
        return () => clearTimeout(timer);
    }, [
        activeGestureTokenRef,
        clearLiveEditOverride,
        commitRef,
        interactionCurveRef,
        onReleased,
        snapshotRef,
        timeoutMs,
    ]);

    useEffect(() => {
        const handoff = commitRef.current;
        if (
            handoff === null ||
            !handoff.committed ||
            !ownsParamBlockGesture({
                activeGestureToken: activeGestureTokenRef.current,
                gestureToken: handoff.gestureToken,
            })
        )
            return;
        /*
         * 判据只有一份（isParamBlockAuthorityVisible）：回包里的新代次已落地 ∧ 水位
         * 确认 ∧ 当前窗口包含提交片段。三处交接（块拖动 / 新建 / 视觉事务）共用它，
         * 上一版各写一遍，其中新建那条忘了超时。
         */
        const curveSettled = isParamBlockAuthorityVisible({
            wait: handoff,
            authority,
            watermark: getAuthorityWatermark(),
            paramView,
        });
        if (!curveSettled || releaseRafRef.current !== null) return;
        handoff.releaseState = "waiting-for-paint";
        releaseRafRef.current = requestAnimationFrame(() => {
            releaseRafRef.current = null;
            if (
                commitRef.current !== handoff ||
                !ownsParamBlockGesture({
                    activeGestureToken: activeGestureTokenRef.current,
                    gestureToken: handoff.gestureToken,
                })
            )
                return;
            // History readback can have reset the live mirror to the old curve.
            // It has higher rendering priority than paramView, so retire it in
            // this same handoff before exposing the confirmed authoritative view.
            clearLiveEditOverride();
            handoff.releaseState = "released";
            commitRef.current = null;
            snapshotRef.current = null;
            interactionCurveRef.current = null;
            activeGestureTokenRef.current = null;
            onReleased();
        });
        return () => {
            if (releaseRafRef.current !== null) {
                cancelAnimationFrame(releaseRafRef.current);
                releaseRafRef.current = null;
            }
        };
    }, [
        activeGestureTokenRef,
        authority,
        clearLiveEditOverride,
        commitRef,
        getAuthorityWatermark,
        interactionCurveRef,
        onReleased,
        paramView,
        releaseRafRef,
        snapshotRef,
        structureVersion,
    ]);
}
