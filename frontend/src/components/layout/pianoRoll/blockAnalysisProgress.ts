/**
 * 块分析进度的前端即时镜像。
 *
 * 后端进度快照仍由 250ms 轮询读取；这里只负责“请求刚发出 / 刚结束”的
 * 即时状态，避免状态栏在第一个轮询周期内仍显示旧的“导入完成”。
 * 与 `appStatusProgressBus` 同样使用外部 store，让高频轮询不带动整棵树重渲染。
 */
import type { SyllableAnalysisProgressPayload } from "../../../types/api";

let progress: SyllableAnalysisProgressPayload | null = null;
const listeners = new Set<() => void>();

function emit(): void {
    for (const listener of listeners) listener();
}

export function publishBlockAnalysisProgress(
    next: SyllableAnalysisProgressPayload | null,
): void {
    if (
        progress === next ||
        (progress !== null &&
            next !== null &&
            progress.active === next.active &&
            progress.clipId === next.clipId &&
            progress.clipName === next.clipName &&
            progress.progress === next.progress &&
            progress.stage === next.stage &&
            progress.algorithm === next.algorithm &&
            progress.message === next.message)
    ) {
        return;
    }
    progress = next;
    emit();
}

export const blockAnalysisProgressStore = {
    subscribe(listener: () => void): () => void {
        listeners.add(listener);
        return () => listeners.delete(listener);
    },
    getSnapshot(): SyllableAnalysisProgressPayload | null {
        return progress;
    },
};
