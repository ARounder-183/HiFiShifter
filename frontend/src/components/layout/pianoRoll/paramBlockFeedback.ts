/**
 * 参数块操作反馈的全局状态栏通道。
 *
 * 画布内不再渲染块错误浮层；所有失败、拒绝与需要用户处置的提示都从这里进入
 * 左下角状态栏，与音高分析 / 波形分析共用同一处视线落点。
 */
import type { ParamBlockFailureFeedback } from "./paramBlockCommitError";

export interface ParamBlockFeedback {
    readonly text: string;
    readonly tone: "success" | "warning" | "danger" | "accent";
    readonly tooltip?: string;
    readonly holdMs?: number;
}

let current: ParamBlockFeedback | null = null;
let hideTimer: ReturnType<typeof setTimeout> | null = null;
const listeners = new Set<() => void>();

function emit(): void {
    for (const listener of listeners) listener();
}

export function publishParamBlockFeedback(next: ParamBlockFeedback | null): void {
    if (hideTimer !== null) clearTimeout(hideTimer);
    hideTimer = null;
    current = next;
    emit();
    if (next?.holdMs && next.holdMs > 0) {
        hideTimer = setTimeout(() => {
            current = null;
            hideTimer = null;
            emit();
        }, next.holdMs);
    }
}

/**
 * 把一次**写入失败**翻译成状态栏反馈。
 *
 * 【为什么收在这里】四类失败（rejected / conflict / transport / silent）各有自己
 * 的文案、tone 与停留时长；散在面板里会让"同一类失败两种措辞"再次出现。
 * 旧实现把四类压成一句 danger，用户无法判断"我的编辑到底有没有生效"。
 *
 * @param tf i18n 取词函数（面板注入，保持本模块与 i18n 解耦）。
 */
export function publishParamBlockWriteFailure(args: {
    readonly failure: ParamBlockFailureFeedback;
    readonly error: { readonly code: string; readonly detail?: string };
    readonly tf: (key: string) => string;
}): void {
    const { failure, error, tf } = args;
    if (failure === "silent") return;
    if (failure === "rejected") {
        publishParamBlockFeedback({
            text: tf("status_param_block_edit_rejected"),
            tone: "warning",
            tooltip: error.detail ?? error.code,
            holdMs: 6_000,
        });
        return;
    }
    if (failure === "conflict") {
        publishParamBlockFeedback({
            text: tf("status_param_block_conflict_resynced"),
            tone: "warning",
            tooltip: error.detail ?? error.code,
            holdMs: 6_000,
        });
        return;
    }
    publishParamBlockFeedback({
        text: tf("status_param_block_write_failed"),
        tone: "danger",
        tooltip: error.detail ?? error.code,
        holdMs: 15_000,
    });
}

export const paramBlockFeedbackStore = {
    subscribe(listener: () => void): () => void {
        listeners.add(listener);
        return () => listeners.delete(listener);
    },
    getSnapshot(): ParamBlockFeedback | null {
        return current;
    },
};
