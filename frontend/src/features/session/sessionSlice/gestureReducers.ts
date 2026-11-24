// 手势会话的 slice reducer（`beginGesture` / `endGesture`）。
//
// 与 `beginInteraction` / `endInteraction` 的分工：
// - `beginInteraction` 是**时间线内核**的交互锁（原有语义不变），它同时入账手势会话；
// - `beginGesture` 是**任何画布**都能用的手势会话（参数编辑器用它，且不占交互锁）；
// - 两者共用同一个 `_gestureDepth`，因此嵌套/重叠时只有最后一个结束时才落地快照。
//
// 【为什么落地放在 reducer 里而不是 thunk 里】推迟的快照是 store 里的数据，
// 只有 reducer 能保证「取出 + 清空 + 应用」是一个不可分割的步骤。放到 thunk
// 里就要先读再写，中间夹一次 React 渲染就可能读到已被替换的旧值。

import type { PayloadAction } from "@reduxjs/toolkit";
import type { TimelineState } from "../../../types/api";
import {
    beginGestureSession,
    endGestureSession,
    readPayloadRevision,
    type DeferredSnapshot,
    type GestureKind,
} from "../gestureSession";
import { applyTimelineState } from "./helpersTimeline";
import type { SessionState } from "./types";

/** 手势会话开始。 */
export function beginGestureReducer(
    state: SessionState,
    action: PayloadAction<{ kind?: GestureKind } | undefined>,
) {
    beginGestureSession(state, action.payload ?? undefined, state.timelineRevision);
}

/**
 * 落地一份被推迟的快照（手势结束时的唯一出口）。
 *
 * 【为什么只落地「更新」的那一份】手势结束时后端必然还会回一份**提交后**的
 * 权威快照；被推迟的那份通常已经被它取代。无条件落地会让界面先退回手势中
 * 间态、再被提交回包纠正一次 —— 那正是用户描述的「回弹」。
 */
export function flushDeferredSnapshot(state: SessionState, deferred: DeferredSnapshot) {
    const revision = deferred.revision ?? readPayloadRevision(deferred.payload);
    if (revision != null && revision <= state.timelineRevision) return;
    applyTimelineState(state, deferred.payload as TimelineState, { force: true });
}

/** 手势会话结束：深度归零时把推迟的快照落地一次。 */
export function endGestureReducer(state: SessionState) {
    const deferred = endGestureSession(state);
    if (!deferred) return;
    flushDeferredSnapshot(state, deferred);
}
