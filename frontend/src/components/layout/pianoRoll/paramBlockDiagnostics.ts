/**
 * 参数块手势诊断。
 *
 * 【为什么需要】块手势的失败以前只走 `warnDev`：开发构建打印，生产构建静默。
 * 用户看到的是“播放光标在块内，按 S 没反应”，日志里什么也没有。这里把失败
 * 收敛成结构化事件（只含 id / 帧号 / 错误码，不含曲线或文件路径），并按
 * code 限频，避免拖动期间刷屏。
 */

import { reportFrontendError } from "../../../services/frontendErrorLog";

export interface ParamBlockDiagnostic {
    readonly code: string;
    readonly gesture: "split" | "resize" | "create" | "delete" | "group" | "restore";
    readonly rootTrackId: string | null;
    readonly blockId: string | null;
    readonly frame: number | null;
    readonly edge?: "left" | "right";
    readonly detail?: string;
    readonly viewBefore?: { readonly blocks: number; readonly revision: number };
    readonly viewAfter?: { readonly blocks: number; readonly revision: number };
}

const lastReported = new Map<string, number>();

/** 同一手势同一错误码最多 1 秒一条；真正的用户反馈不受限频影响。 */
export function reportParamBlockDiagnostic(diagnostic: ParamBlockDiagnostic): void {
    const now = typeof performance !== "undefined" ? performance.now() : Date.now();
    const key = `${diagnostic.gesture}:${diagnostic.code}`;
    const previous = lastReported.get(key);
    if (previous !== undefined && now - previous < 1000) return;
    lastReported.set(key, now);
    reportFrontendError(
        `param_block_${diagnostic.gesture}:${diagnostic.code}`,
        JSON.stringify(diagnostic),
    );
}

/** 仅测试使用：清空限频状态。 */
export function resetParamBlockDiagnosticsForTests(): void {
    lastReported.clear();
}
