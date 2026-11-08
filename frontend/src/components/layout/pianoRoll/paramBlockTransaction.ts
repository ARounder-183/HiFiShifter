/** Request watermarks exposed by the authoritative parameter-window loader. */
export interface ParamAuthorityWatermark {
    readonly issuedRequestSeq: number;
    readonly appliedRequestSeq: number;
}

/**
 * A commit preview may only be retired by a response whose request was issued
 * after the commit began.
 *
 * Merely comparing a response counter with its value at commit time is unsafe:
 * a request issued before the commit can finish afterwards and look newer while
 * still carrying the previous curve.
 */
export function isParamAuthorityConfirmedAfterCommit(args: {
    readonly watermark: ParamAuthorityWatermark;
    readonly issuedRequestSeqAtCommit: number;
}): boolean {
    return (
        Number.isFinite(args.watermark.issuedRequestSeq) &&
        Number.isFinite(args.watermark.appliedRequestSeq) &&
        args.watermark.appliedRequestSeq > args.issuedRequestSeqAtCommit
    );
}

/** Only the newest block gesture may mutate the shared visual transaction. */
export function ownsParamBlockGesture(args: {
    readonly activeGestureToken: number | null;
    readonly gestureToken: number;
}): boolean {
    return args.activeGestureToken === args.gestureToken;
}
