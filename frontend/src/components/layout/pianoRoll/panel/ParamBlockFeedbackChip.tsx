/** 状态栏里的参数块反馈片；订阅轻量外部 store，不带动整棵树重渲染。 */
import { useSyncExternalStore } from "react";
import { AppStatusChip } from "../../../../ui";
import { paramBlockFeedbackStore } from "../paramBlockFeedback";

export function ParamBlockFeedbackChip() {
    const feedback = useSyncExternalStore(
        paramBlockFeedbackStore.subscribe,
        paramBlockFeedbackStore.getSnapshot,
        paramBlockFeedbackStore.getSnapshot,
    );
    if (feedback === null) return null;
    return (
        <AppStatusChip tone={feedback.tone} tooltip={feedback.tooltip}>
            {feedback.text}
        </AppStatusChip>
    );
}
