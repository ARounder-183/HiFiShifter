/**
 * 参数块失败的**唯一**上报入口（调用点侧）。
 *
 * 【为什么需要这一层】写入层（`useParamBlockPersistence`）已经按类别上报过：
 * rejected / conflict / transport / silent 各有自己的文案、tone 与停留时长。
 * 而旧实现的每个 `await commit(...)` 调用点又各自 `catch` 并弹一条
 * 「参数块操作失败」—— `paramBlockFeedback` 是**单槽**，后发布的通用红条
 * **必然覆盖**先发布的分类消息。于是用户永远只看到那一句，真实原因被吞掉。
 *
 * 这里把调用点允许做的两件事分开：
 * 1. {@link reportParamBlockWriteFailure}：写入层已经报过，这里**只写诊断**，
 *    不再发任何用户可见提示；
 * 2. {@link reportParamBlockLocalRejection}：本地纯函数计划失败（预检 / 分割太近 /
 *    新建失败）—— 这类失败**根本没有发起写入**，所以必须由调用点报一条 warning。
 *
 * 【违反边界的症状】在写入失败的分支里再发一条反馈 ⇒ 分类消息被通用红条覆盖，
 * 用户看到「参数块操作失败」而编辑其实成功了。
 */
import { publishParamBlockFeedback } from "../paramBlockFeedback";
import { reportParamBlockDiagnostic } from "../paramBlockDiagnostics";
import { classifyParamBlockFailure, paramBlockFailureFeedback } from "../paramBlockCommitError";

type Gesture = Parameters<typeof reportParamBlockDiagnostic>[0]["gesture"];

/**
 * 写入失败：**只**写结构化诊断。
 *
 * 用户可见的反馈由写入层负责（它掌握分类结果）。这里保留诊断，是因为调用点
 * 知道「哪一次用户操作」触发了它，而写入层只知道「哪个根」。
 */
export function reportParamBlockWriteFailure(args: {
    readonly error: unknown;
    readonly gesture: Gesture;
    readonly rootTrackId: string | null;
    readonly blockId?: string | null;
    readonly frame?: number | null;
}): void {
    const failure = classifyParamBlockFailure(args.error);
    const feedback = paramBlockFailureFeedback(failure);
    reportParamBlockDiagnostic({
        code: feedback === "silent" ? "write_cancelled" : failure.code,
        gesture: args.gesture,
        rootTrackId: args.rootTrackId,
        blockId: args.blockId ?? null,
        frame: args.frame ?? null,
        ...(failure.detail === undefined ? {} : { detail: failure.detail }),
    });
}

/**
 * 本地拒绝：没有发起任何写入的失败。
 *
 * 文案与写入层共用同一张表（`status_param_block_edit_rejected`），所以这里
 * **不新增** i18n key，也不会与写入层的消息长得不一样。
 */
export function reportParamBlockLocalRejection(args: {
    readonly code: string;
    readonly detail?: string;
    readonly gesture: Gesture;
    readonly rootTrackId: string | null;
    readonly blockId?: string | null;
    readonly frame?: number | null;
    readonly tf: (key: string) => string;
}): void {
    reportParamBlockDiagnostic({
        code: args.code,
        gesture: args.gesture,
        rootTrackId: args.rootTrackId,
        blockId: args.blockId ?? null,
        frame: args.frame ?? null,
        ...(args.detail === undefined ? {} : { detail: args.detail }),
    });
    publishParamBlockFeedback({
        text: args.tf("status_param_block_edit_rejected"),
        tone: "warning",
        tooltip: args.detail ?? args.code,
        holdMs: 6_000,
    });
}
