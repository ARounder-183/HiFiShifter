/**
 * 参数块操作反馈的全局状态栏通道。
 *
 * 画布内不再渲染块错误浮层；所有失败、拒绝与需要用户处置的提示都从这里进入
 * 左下角状态栏，与音高分析 / 波形分析共用同一处视线落点。
 */
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

export const paramBlockFeedbackStore = {
    subscribe(listener: () => void): () => void {
        listeners.add(listener);
        return () => listeners.delete(listener);
    },
    getSnapshot(): ParamBlockFeedback | null {
        return current;
    },
};
