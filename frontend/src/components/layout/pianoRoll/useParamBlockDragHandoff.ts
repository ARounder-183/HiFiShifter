import { useEffect, type RefObject } from "react";
import type { ParamBlockDragCommit, ParamBlockDragSnapshot } from "./paramBlockDrag";
import { paramViewContainsCommittedSegments } from "./paramBlocks";
import {
    isParamAuthorityConfirmedAfterCommit,
    ownsParamBlockGesture,
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
    readonly authorityRevision: number;
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
    authorityRevision,
    getAuthorityWatermark,
    structureVersion,
    clearLiveEditOverride,
    onReleased,
    timeoutMs = 1200,
}: ParamBlockDragHandoffInputs): void {
    useEffect(() => {
        const pending = commitRef.current;
        if (pending === null || pending.releaseState === "released") return;
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
        const curveSettled =
            paramView !== null &&
            authorityRevision > handoff.authorityRevisionAtStart &&
            isParamAuthorityConfirmedAfterCommit({
                watermark: getAuthorityWatermark(),
                issuedRequestSeqAtCommit: handoff.authorityRequestSeqAtCommit,
            }) &&
            handoff.committedSegments !== undefined &&
            paramViewContainsCommittedSegments({ paramView, segments: handoff.committedSegments });
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
        authorityRevision,
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
