/**
 * frontendErrorLog — 前端错误回传到后端统一日志。
 *
 * 目的：让前端异常（invoke 失败、未捕获错误、未处理的 Promise rejection）
 * 与后端日志落在同一文件、同一时间轴上，用户只需要提交一个日志文件。
 *
 * 注意：这里直接使用 window.__TAURI__ 而非 services/invoke，避免
 * 「上报失败 → 触发上报」的递归；pywebview 模式下没有 Tauri IPC，静默跳过。
 */

const MAX_DETAIL_CHARS = 4000;

function tauriInvokeRaw(cmd: string, args: Record<string, unknown>): unknown {
    const tauri = window as unknown as {
        __TAURI__?: {
            core?: { invoke?: (cmd: string, args?: Record<string, unknown>) => unknown };
            invoke?: (cmd: string, args?: Record<string, unknown>) => unknown;
        };
    };
    const invokeFn = tauri.__TAURI__?.core?.invoke ?? tauri.__TAURI__?.invoke;
    if (typeof invokeFn !== "function") return null;
    try {
        return invokeFn(cmd, args);
    } catch {
        return null;
    }
}

function toDetailText(detail: unknown): string {
    if (detail === undefined || detail === null) return "";
    let text: string;
    if (detail instanceof Error) {
        text = detail.stack || `${detail.name}: ${detail.message}`;
    } else if (typeof detail === "string") {
        text = detail;
    } else {
        try {
            text = JSON.stringify(detail) ?? String(detail);
        } catch {
            text = String(detail);
        }
    }
    if (text.length > MAX_DETAIL_CHARS) {
        text = `${text.slice(0, MAX_DETAIL_CHARS)}…(truncated)`;
    }
    return text;
}

/** 把一条前端错误写入后端日志（fire-and-forget，永不抛出）。 */
export function reportFrontendError(message: string, detail?: unknown): void {
    try {
        const detailText = toDetailText(detail);
        const result = tauriInvokeRaw("log_frontend_error", {
            message,
            detail: detailText || null,
        });
        if (result && typeof (result as Promise<unknown>).catch === "function") {
            (result as Promise<unknown>).catch(() => {});
        }
    } catch {
        // 日志回传自身绝不能抛错。
    }
}

/** 安装全局兜底上报：window error 与 unhandledrejection。在应用入口调用一次。 */
export function installGlobalErrorReporting(): void {
    if (typeof window === "undefined") return;

    window.addEventListener("error", (event) => {
        const err = event.error as Error | undefined;
        reportFrontendError(
            `Uncaught error: ${event.message || "(no message)"}`,
            err?.stack ?? `${event.filename}:${event.lineno}:${event.colno}`,
        );
    });

    window.addEventListener("unhandledrejection", (event) => {
        const reason = (event as PromiseRejectionEvent).reason;
        reportFrontendError(
            `Unhandled rejection: ${reason instanceof Error ? reason.message : String(reason)}`,
            reason,
        );
    });
}
