/**
 * Parameter-block visual transaction state.
 *
 * A visual preview may not outlive an external layout/history/parameter
 * change. The epoch carries those identities; the transaction additionally
 * records the edit revision produced by its own commit so a self-commit does
 * not invalidate its own preview.
 */
import type { PianoRollParamBlockSpec } from "./kernel/host/pianoRollKernelData";

export interface ParamBlockEpoch {
    readonly rootTrackId: string | null;
    /** Parameter name; pitch and volume must never share a visual transaction. */
    readonly param: string;
    readonly layoutKey: string | null;
    readonly editRevision: number;
    /**
     * Only explicit history jumps (undo / redo / jump / project switch) change
     * this token. A block commit also advances the backend history depth, but
     * that must not be mistaken for an external jump while the gesture hands
     * its visual state over to the authoritative view.
     */
    readonly externalHistoryToken: string;
    /** Monotonic token for commits produced by this panel. */
    readonly commitToken: string;
    readonly framePeriodMs: number;
}

export type ParamBlockVisualTransactionKind = "drag" | "resize" | "create" | "transition";
export type ParamBlockVisualTransactionPhase =
    "gesture" | "committing" | "awaiting-authority" | "released";

export interface ParamBlockVisualTransaction {
    readonly id: number;
    readonly epoch: ParamBlockEpoch;
    readonly kind: ParamBlockVisualTransactionKind;
    spec: PianoRollParamBlockSpec;
    phase: ParamBlockVisualTransactionPhase;
    readonly startedAt: number;
    /**
     * Edit revision produced by this transaction's own commit. A transaction
     * may survive that revision, but never a third-party structural edit.
     */
    ownedEditRevision: number | null;
    expectedEditRevision?: number;
    expectedGeometrySignature?: string;
    committedSegments?: readonly {
        readonly startFrame: number;
        readonly values: readonly number[];
    }[];
}

/**
 * Drag visuals survive their own commit changing editRevision / commitToken,
 * but not an external history jump, root switch, parameter switch, clip layout
 * change, or frame period change.
 */
export function sameParamBlockEpochLayout(a: ParamBlockEpoch, b: ParamBlockEpoch): boolean {
    return (
        a.rootTrackId === b.rootTrackId &&
        a.param === b.param &&
        a.layoutKey === b.layoutKey &&
        a.externalHistoryToken === b.externalHistoryToken &&
        Math.abs(a.framePeriodMs - b.framePeriodMs) < 1e-9
    );
}

/** Cancellation predicate for an in-flight drag commit: only external changes matter. */
export function sameParamBlockExternalEpoch(a: ParamBlockEpoch, b: ParamBlockEpoch): boolean {
    return sameParamBlockEpochLayout(a, b);
}

/**
 * A visual may survive its own commit changing editRevision, but not a
 * third-party structural edit. External edits do not know the owned revision,
 * so they fail this check immediately.
 */
export function isParamBlockVisualEpochCurrent(args: {
    readonly epoch: ParamBlockEpoch;
    readonly current: ParamBlockEpoch;
    readonly ownedEditRevision: number | null;
}): boolean {
    if (!sameParamBlockEpochLayout(args.epoch, args.current)) return false;
    if (args.current.editRevision === args.epoch.editRevision) return true;
    return args.ownedEditRevision !== null && args.current.editRevision === args.ownedEditRevision;
}

export function shouldUseParamBlockVisualTransaction(
    tx: ParamBlockVisualTransaction | null,
    currentEpoch: ParamBlockEpoch,
): boolean {
    return (
        tx !== null && tx.phase !== "released" && sameParamBlockEpochLayout(tx.epoch, currentEpoch)
    );
}
