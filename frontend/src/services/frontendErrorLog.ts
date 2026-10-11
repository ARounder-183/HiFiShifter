/**
 * frontendErrorLog — 前端错误回传到后端统一日志。
 *
 * 目的：让前端异常（invoke 失败、未捕获错误、未处理的 Promise rejection）
 * 与后端日志落在同一文件、同一时间轴上，用户只需要提交一个日志文件。
 *
 * 注意：这里直接使用 window.__TAURI__ 而非 services/invoke，避免
 * 「上报失败 → 触发上报」的递归；pywebview 模式下没有 Tauri IPC，静默跳过。
 */

import { getPluginHost } from "./pluginHost";

const MAX_DETAIL_CHARS = 4000;

function tauriInvokeRaw(cmd: string, args: Record<string, unknown>): unknown {
    const plugin = getPluginHost();
    if (plugin) return plugin.invoke(cmd, args);
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

/**
 * 同一错误的抑制窗口：窗口内只上报一次，窗口过后再报一次并附上重复次数。
 *
 * 【为什么前端也要去重，而不是只靠后端限流】后端限流能保住日志文件，但挡不住
 * **每次失败都要跨一次 IPC**：一个每秒重试 4 次的终态错误，即使后端只写一行，
 * 前端仍会把整条错误对象序列化 4 次/秒（含最长 4000 字符的栈）并投递过去。
 * 两侧都做，才是"既不刷屏、也不白费 IPC"。
 */
const DEDUP_WINDOW_MS = 60_000;
/** 同时跟踪的键上限：超限时按最久未见淘汰，避免自身变成内存增长点。 */
const DEDUP_MAX_KEYS = 64;

type DedupState = { firstSeen: number; lastReported: number; count: number };
const dedupKeys = new Map<string, DedupState>();

/**
 * 判定这条错误是否应当**现在**上报；返回 null 表示抑制。
 *
 * 返回值为附加在消息末尾的重复说明（首次上报为空串）。
 */
function dedupDecision(key: string, now: number): string | null {
    const existing = dedupKeys.get(key);
    if (!existing) {
        if (dedupKeys.size >= DEDUP_MAX_KEYS) {
            // 淘汰最久未上报的键。Map 迭代顺序即插入顺序，但 lastReported 才是
            // 真正的"新近度"，所以按它找最小值。
            let stalestKey: string | null = null;
            let stalestAt = Number.POSITIVE_INFINITY;
            for (const [candidate, state] of dedupKeys) {
                if (state.lastReported < stalestAt) {
                    stalestAt = state.lastReported;
                    stalestKey = candidate;
                }
            }
            if (stalestKey !== null) dedupKeys.delete(stalestKey);
        }
        dedupKeys.set(key, { firstSeen: now, lastReported: now, count: 1 });
        return "";
    }
    existing.count += 1;
    if (now - existing.lastReported < DEDUP_WINDOW_MS) {
        return null;
    }
    const seconds = Math.max(1, Math.round((now - existing.firstSeen) / 1000));
    const repeats = existing.count - 1;
    existing.lastReported = now;
    existing.count = 1;
    existing.firstSeen = now;
    return ` (repeated ${repeats} time(s) over the last ${seconds}s)`;
}

/** 仅供测试重置模块级去重状态。 */
export function resetFrontendErrorDedupForTests(): void {
    dedupKeys.clear();
}

/** 把一条前端错误写入后端日志（fire-and-forget，永不抛出）。 */
export function reportFrontendError(message: string, detail?: unknown): void {
    try {
        const detailText = toDetailText(detail);
        // 键取"消息 + 详情前 200 字符"：足够区分不同错误，又不会被栈里的地址/行号
        // 打散成每次都不同的键（那会让去重完全失效）。
        const key = `${message}\u0000${detailText.slice(0, 200)}`;
        const suffix = dedupDecision(key, Date.now());
        if (suffix === null) return;
        const result = tauriInvokeRaw("log_frontend_error", {
            message: `${message}${suffix}`,
            detail: detailText || null,
        });
        if (result && typeof (result as Promise<unknown>).catch === "function") {
            (result as Promise<unknown>).catch(() => {});
        }
    } catch {
        // 日志回传自身绝不能抛错。
    }
}

/**
 * 已知的"良性"错误消息模式：上报它们只会制造噪音，不携带任何可行动的信息。
 *
 * - `ResizeObserver loop completed with undelivered notifications`：
 *   浏览器的布局收敛提示 —— 本帧来不及投递的尺寸回调会在下一帧自动投递，
 *   不是应用错误（Chrome/WebView 的既定行为，见 WHATWG / Chromium issue）。
 */
const BENIGN_ERROR_PATTERNS: RegExp[] = [/^ResizeObserver loop/i];

/** 安装全局兜底上报：window error 与 unhandledrejection。在应用入口调用一次。 */
export function installGlobalErrorReporting(): void {
    if (typeof window === "undefined") return;

    window.addEventListener("error", (event) => {
        const message = event.message || "";
        if (BENIGN_ERROR_PATTERNS.some((pattern) => pattern.test(message))) {
            return;
        }
        const err = event.error as Error | undefined;
        reportFrontendError(
            `Uncaught error: ${message || "(no message)"}`,
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
