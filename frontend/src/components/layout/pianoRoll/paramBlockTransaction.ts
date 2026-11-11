/** Request watermarks exposed by the authoritative parameter-window loader. */
export interface ParamAuthorityWatermark {
    readonly issuedRequestSeq: number;
    readonly appliedRequestSeq: number;
}

/**
 * 已落地的**权威**曲线代次（后端 `param_curve_revision`）。
 *
 * 【为什么必须是后端代次】它是 `set_param_block_edits` 的 CAS 单位，也是「这次写入
 * 有没有生效」的唯一可比证据。旧实现暴露的是一个**本地计数器**（每落地一次参数
 * 回包 +1），而事务上存的是后端代次 —— 两个数不可比，于是带曲线补丁的过渡事务
 * 每次都要等满 2 秒超时才释放预览。
 *
 * 【requestSeq】产生这份代次的那次读请求序号，配合 {@link ParamAuthorityWatermark}
 * 排除「提交前发出、提交后才到」的陈旧回包。
 */
export interface ParamAuthoritySnapshot {
    /** 后端曲线代次；0 = 还没落地过任何回包。 */
    readonly curveRevision: number;
    /** 产生这份代次的读请求序号。 */
    readonly requestSeq: number;
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
