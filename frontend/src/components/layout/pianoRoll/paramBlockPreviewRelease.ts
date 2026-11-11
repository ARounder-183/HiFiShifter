/**
 * 参数块预览的**唯一**释放判据。
 *
 * 【它解决什么问题】
 * 旧实现按「本地派生几何是否已经等于预览几何」判断预览可以撤下了。这个条件在
 * pointerup 之后**一次渲染**就成立（`setParamBlockState` 是同步提交），于是：
 * 预览被撤下 → 曲线回到旧值 → 块按旧曲线重新采样 → 用户看到「先回弹旧参数」；
 * 等权威窗口到了才跳到新值 —— 即「再生效为新参数」。过渡手柄更糟：它的几何
 * 根本不变化，所以同一次渲染就满足判据。
 *
 * 【现在的判据】只有一条：**这次提交产生的权威已经可见**。
 * - 带曲线补丁的提交：曲线 revision 前进 ∧ 权威水位已确认 ∧ 当前窗口包含
 *   提交的片段 —— 三者必须同时成立（缺一条就说明我们看到的还是旧曲线）；
 * - 纯布局提交：几何已包含新值即可（没有曲线可回弹）。
 *
 * 【超时】超时不是「可以不等了」，而是「先把权威读回来再撤」：调用方必须先
 * `flushReadback()` + `clearLiveEditOverride()`，否则撤下预览的第一帧仍然是
 * 旧曲线 —— 那正是本模块要消灭的现象。
 *
 * 【边界】纯函数：不读 React、不发请求、不改任何 ref。
 */
import type { ParamBlockVisualTransaction } from "./paramBlockVisualTransaction";
import { paramViewContainsCommittedSegments } from "./paramBlocks";
import {
    isParamAuthorityConfirmedAfterCommit,
    type ParamAuthorityWatermark,
} from "./paramBlockTransaction";
import type { ParamViewSegment } from "./types";

/** 默认硬超时：超过它就必须对账后强制释放，绝不永久锁住预览。 */
export const PARAM_BLOCK_PREVIEW_RELEASE_TIMEOUT_MS = 2000;

export type ParamBlockPreviewRelease = "wait" | "release" | "timeout";

/**
 * 宣布「这次提交会写曲线」。
 *
 * 【为什么必须在 pointerup 那一刻宣布】过渡手柄的几何根本不变化，如果等到异步
 * 读基准之后才写 `committedSegments`，这中间的每一帧都会按"几何相等"判定可以
 * 释放 —— 预览被撤、曲线回旧值，用户看到"先回弹、再生效"。
 *
 * @param pending true = 会写曲线但片段还没算出来（`null`）；false = 纯布局。
 */
export function stampParamBlockCurveIntent(
    transaction: ParamBlockVisualTransaction | null,
    pending: boolean,
): void {
    if (transaction === null) return;
    transaction.committedSegments = pending ? null : undefined;
}

/**
 * 预览几何是否已经出现在派生块里（纯布局提交的判据）。
 *
 * 只比 `startSec` / `endSec`：块的值来自曲线采样，属于曲线判据的事；几何相等
 * 只说明"React 侧已经拿到这次布局"。
 */
export function isParamBlockPreviewGeometrySettled(args: {
    readonly transaction: ParamBlockVisualTransaction;
    readonly blocks: readonly {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
        readonly transitionSec?: number;
    }[];
}): boolean {
    const { transaction } = args;
    const matches = (id: string, startSec: number, endSec: number): boolean => {
        const current = args.blocks.find((block) => block.id === id);
        return (
            current !== undefined &&
            Math.abs(current.startSec - startSec) <= 1e-6 &&
            Math.abs(current.endSec - endSec) <= 1e-6
        );
    };
    const geometry =
        transaction.kind === "create"
            ? transaction.spec.blocks
                  .filter((block) => block.selected)
                  .every((block) => matches(block.id, block.startSec, block.endSec))
            : transaction.spec.blocks.every((block) =>
                  matches(block.id, block.startSec, block.endSec),
              );
    if (!geometry) return false;
    if (transaction.kind !== "transition") return true;
    return transaction.spec.blocks.every((block) => {
        const handle = block.transitionHandle;
        if (handle === undefined || handle === null) return true;
        const current = args.blocks.find((candidate) => candidate.id === block.id);
        return (
            current?.transitionSec !== undefined &&
            Math.abs(current.transitionSec - handle.widthSec) <= 1e-6
        );
    });
}

/**
 * 一次提交「权威是否已经可见」的输入 —— 全仓唯一判据的形状。
 *
 * 【三态 committedSegments】
 * - undefined：纯布局提交，没有曲线可回弹 ⇒ 立即可见；
 * - null：已宣布「这次会写曲线」但片段还没算出来 ⇒ 必须等（几何相等在这里没有意义）；
 * - 数组：曲线片段已知 ⇒ 三个条件必须同时成立。
 */
export interface ParamBlockAuthorityWait {
    readonly authorityRevisionAtStart?: number | null;
    readonly authorityRequestSeqAtCommit?: number | null;
    readonly committedSegments?:
        | readonly {
              readonly startFrame: number;
              readonly values: readonly number[];
          }[]
        | null;
}

/**
 * 这次提交写入的曲线，权威是否已经能在当前窗口里看到。
 *
 * 【三个条件缺一不可】revision 前进（后端确实接受了这次写）∧ 水位确认（这份回包不是
 * 提交前发出的在飞响应）∧ 当前窗口包含提交片段（看到的不是别的窗口）。
 */
export function isParamBlockAuthorityVisible(args: {
    readonly wait: ParamBlockAuthorityWait;
    readonly authorityRevision: number;
    readonly watermark: ParamAuthorityWatermark;
    readonly paramView: ParamViewSegment | null;
}): boolean {
    const committed = args.wait.committedSegments;
    if (committed === undefined || committed === null || committed.length === 0) return false;
    return (
        Number.isFinite(args.wait.authorityRevisionAtStart ?? Number.NaN) &&
        args.authorityRevision > (args.wait.authorityRevisionAtStart ?? Number.POSITIVE_INFINITY) &&
        isParamAuthorityConfirmedAfterCommit({
            watermark: args.watermark,
            issuedRequestSeqAtCommit: args.wait.authorityRequestSeqAtCommit ?? 0,
        }) &&
        args.paramView !== null &&
        paramViewContainsCommittedSegments({ paramView: args.paramView, segments: committed })
    );
}

/**
 * 预览 / 交接的**统一**释放决策。
 *
 * 【为什么合成一个函数】上一版三处各自写一遍三元组判据，其中一处（新建块的交接）
 * 忘了超时 —— 权威永远不落地时预览**永久停留**，屏幕上留一个只能切参数才消失的
 * 幽灵块。判据只有一份，超时只有一个常量。
 */
export function resolveParamBlockRelease(input: {
    readonly wait: ParamBlockAuthorityWait;
    readonly authorityRevision: number;
    readonly watermark: ParamAuthorityWatermark;
    readonly paramView: ParamViewSegment | null;
    /** 纯布局提交的补充判据（几何已到位 / 块已出现在派生视图）。 */
    readonly geometrySettled: boolean;
    readonly gestureActive: boolean;
    readonly pendingCommits: number;
    readonly startedAt: number;
    readonly now: number;
    readonly timeoutMs?: number;
}): ParamBlockPreviewRelease {
    if (input.gestureActive || input.pendingCommits > 0) return "wait";
    const timeoutMs = input.timeoutMs ?? PARAM_BLOCK_PREVIEW_RELEASE_TIMEOUT_MS;
    const timedOut = input.now - input.startedAt > timeoutMs;
    const committed = input.wait.committedSegments;
    // 已宣布要写曲线但片段还没算出来：必须等（几何相等在这里没有意义）。
    if (committed === null) return timedOut ? "timeout" : "wait";
    if (
        isParamBlockAuthorityVisible({
            wait: input.wait,
            authorityRevision: input.authorityRevision,
            watermark: input.watermark,
            paramView: input.paramView,
        })
    ) {
        return "release";
    }
    if (committed !== undefined && committed.length > 0) return timedOut ? "timeout" : "wait";
    if (input.geometrySettled) return "release";
    return timedOut ? "timeout" : "wait";
}

export interface ParamBlockPreviewReleaseInput {
    readonly transaction: ParamBlockVisualTransaction;
    readonly now: number;
    /** 几何是否已包含预览值；纯布局提交的唯一判据。 */
    readonly geometrySettled: boolean;
    /** 指针手势是否仍在进行（含其它块手势）。 */
    readonly gestureActive: boolean;
    /** 队列里还有未落地的提交。 */
    readonly pendingCommits: number;
    readonly authorityRevision: number;
    readonly watermark: ParamAuthorityWatermark;
    readonly paramView: ParamViewSegment | null;
    readonly timeoutMs?: number;
}

/**
 * 该不该撤下这份预览。
 *
 * - `"wait"`：继续等（预览必须继续画）；
 * - `"release"`：权威已可见，可以撤（调用方仍应跨一帧再撤，避免闪一帧）；
 * - `"timeout"`：超时，调用方**必须先对账**再按 release 处理。
 */
export function shouldReleaseParamBlockPreview(
    input: ParamBlockPreviewReleaseInput,
): ParamBlockPreviewRelease {
    const { transaction } = input;
    if (transaction.phase === "gesture") return "wait";
    /* 薄适配器：判据只有一份（resolveParamBlockRelease），这里只负责拆包。 */
    return resolveParamBlockRelease({
        wait: transaction,
        authorityRevision: input.authorityRevision,
        watermark: input.watermark,
        paramView: input.paramView,
        geometrySettled: input.geometrySettled,
        gestureActive: input.gestureActive,
        pendingCommits: input.pendingCommits,
        startedAt: transaction.startedAt,
        now: input.now,
        ...(input.timeoutMs === undefined ? {} : { timeoutMs: input.timeoutMs }),
    });
}
