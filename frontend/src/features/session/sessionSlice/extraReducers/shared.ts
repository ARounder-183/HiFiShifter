// Session extraReducers 共用的 pending/rejected 状态写入。
import { rejectedMessage } from "../helpersCore";
import type { SessionState } from "../types";

export function setPending(state: SessionState, label: string) {
    state.busy = true;
    state.status = label;
    state.error = undefined;
}

export function setRejected(
    state: SessionState,
    action: { payload?: unknown; error?: { message?: string } },
) {
    state.busy = false;
    state.error = rejectedMessage(action);
    state.status = "Failed";
}
