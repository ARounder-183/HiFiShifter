import type { RefObject } from "react";
import type { ParamBlockDragActivity } from "./paramBlockDrag";
import type { ParamViewSegment } from "./types";

/** A new gesture continues the visible edits, including commits still in flight. */
export function captureParamBlockCreateCurve(
    authoritative: ParamViewSegment | null,
    interaction: ParamViewSegment | null,
): ParamViewSegment | null {
    const basis = interaction?.key === authoritative?.key ? interaction : authoritative;
    return basis
        ? { ...basis, preview: undefined, edit: basis.edit.slice(), orig: basis.orig.slice() }
        : null;
}

/**
 * 新建块提交的**手势侧**簿记。
 *
 * 【它不再排队】串行化已经收口到 `useParamBlockPersistence.commitPlan`（一个根一条
 * 队列，「读基准 → 算补丁 → 写」整体在槽内）。旧实现让创建手势拥有第二条队列，
 * 与持久化层的队列共用同一个后端 CAS —— 交错时后写者必然过期，那是「各种编辑
 * 场景仍有概率报参数块操作失败」的结构性根因。
 *
 * 这里只剩两件手势本地的事：
 * 1. `pendingCommits` / `liveEditActive` 的记账（它们决定回读要不要被推迟）；
 * 2. 无论成功失败都调一次 `onSettled`（刷新权威、重算派生视图）。
 *
 * 【违反边界的症状】在这里再排一次队 ⇒ 两层队列共用同一个 CAS，冲突重新出现；
 * 忘记 `onSettled` ⇒ 块停留在本地预览，直到下一次无关的刷新。
 */
export function enqueueParamBlockCreateCommit<T>(args: {
    activity: RefObject<ParamBlockDragActivity>;
    liveActive: RefObject<boolean>;
    /** 完整提交（含读基准）；由 `commitPlan` 串行化。 */
    run: () => Promise<T>;
    onSettled: () => void;
}): Promise<T> {
    args.activity.current.pendingCommits++;
    args.liveActive.current = true;
    return args.run().finally(() => {
        args.activity.current.pendingCommits--;
        if (
            !args.activity.current.pendingCommits &&
            args.activity.current.pointerGestureToken === null
        )
            args.liveActive.current = false;
        args.onSettled();
    });
}
