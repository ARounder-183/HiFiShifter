/**
 * Parameter-block visual transaction state.
 *
 * A visual preview may not outlive an external layout/history/parameter
 * change. The epoch carries those identities. It deliberately does **not**
 * carry an edit revision: revisions are a content counter on the client and a
 * CAS counter on the server, so they are not comparable — see
 * `paramBlockStructureToken`.
 */
import type { PianoRollParamBlockSpec } from "./kernel/host/pianoRollKernelData";

export interface ParamBlockEpoch {
    readonly rootTrackId: string | null;
    /** Parameter name; pitch and volume must never share a visual transaction. */
    readonly param: string;
    readonly layoutKey: string | null;
    /**
     * 编辑集的**结构身份**（块 id / 几何 / 过渡宽度 / 删除标记的指纹）。
     *
     * 【为什么不是 revision】revision 在前端是内容代次、在后端是 CAS 代次，两者
     * 不可比；而视觉预览真正关心的是「块的形状有没有被第三方改过」。结构令牌只在
     * 形状变化时变化：一次拖动只改 value（中心）与曲线，令牌不变 ⇒ 自己的提交
     * 不会撤销自己的预览。
     */
    readonly structureToken: string;
    /**
     * Only explicit history jumps (undo / redo / jump / project switch) change
     * this token. A block commit also advances the backend history depth, but
     * that must not be mistaken for an external jump while the gesture hands
     * its visual state over to the authoritative view.
     */
    readonly externalHistoryToken: string;
    readonly framePeriodMs: number;
}

export type ParamBlockVisualTransactionKind = "drag" | "resize" | "create" | "transition";

export interface ParamBlockVisualTransaction {
    readonly id: number;
    readonly epoch: ParamBlockEpoch;
    readonly kind: ParamBlockVisualTransactionKind;
    spec: PianoRollParamBlockSpec;
    readonly startedAt: number;
    /**
     * 本次提交**依据**的曲线权威代次（= 提交时携带的 expectedCurveRevision）。
     *
     * 【单位】后端 `param_curve_revision`。全仓只有这一个单位 —— 旧实现里过渡手柄
     * 在这里存后端代次、却拿本地取数计数器去比，于是判据几乎永远为假，每次都要等满
     * 2s 超时才释放预览（用户看到「拖完过渡手柄卡两秒」）。
     *
     * 缺省（`undefined`）= 纯布局提交，没有曲线可等。
     */
    curveRevisionAtStart?: number;
    /**
     * 提交成功回包里的**新**后端曲线代次。缺省 = 还没拿到回包。
     *
     * 释放判据拿它和「已落地的权威曲线代次」比：`>=` 才说明这次写入的曲线已经
     * 进了当前窗口。用「回包里的新代次」而不是「起始代次 + 1」，是因为两者之间可能
     * 还有别的写入。
     */
    curveRevisionCommitted?: number;
    /** 提交时的权威请求序号；配合 watermark 排除"提交前发出的在飞回包"。 */
    authorityRequestSeqAtCommit?: number;
    /**
     * 本次提交写入的曲线片段；释放前必须能在当前窗口里看到它们。
     *
     * - `undefined`：纯布局提交，没有曲线可等；
     * - `null`：**已宣布**要写曲线，但片段还没算出来（异步读基准期间）。
     *   这段窗口必须继续等 —— 过渡手柄的几何不变化，若此时按几何相等释放，
     *   就会退回旧曲线（"先回弹"的根因）；
     * - 数组：片段已知，按权威判据等它们出现。
     */
    committedSegments?:
        | readonly {
              readonly startFrame: number;
              readonly values: readonly number[];
          }[]
        | null;
}

/**
 * Drag visuals survive their own commit changing the edit set, but not an
 * external history jump, root switch, parameter switch, clip layout change, or
 * frame period change.
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
 * 视觉预览是否仍然有效。
 *
 * 【判据只有两条】布局 / 历史 / 参数 / 帧周期都没变，且**结构令牌**没变。
 * 旧实现还比 revision（外加一个永远对不上的 `ownedEditRevision`），于是自己提交的
 * 回包一落地就撤掉自己的预览 —— 「先回弹、再生效」的成因。
 */
export function isParamBlockVisualEpochCurrent(args: {
    readonly epoch: ParamBlockEpoch;
    readonly current: ParamBlockEpoch;
}): boolean {
    return (
        sameParamBlockEpochLayout(args.epoch, args.current) &&
        args.epoch.structureToken === args.current.structureToken
    );
}

export function shouldUseParamBlockVisualTransaction(
    tx: ParamBlockVisualTransaction | null,
    currentEpoch: ParamBlockEpoch,
): boolean {
    return tx !== null && sameParamBlockEpochLayout(tx.epoch, currentEpoch);
}
