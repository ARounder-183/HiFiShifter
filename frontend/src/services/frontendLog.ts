/**
 * frontendLog — 前端**唯一**的日志出口。
 *
 * 目的：把散落在 30+ 文件里的 `console.*`、三个 localStorage 调试开关，以及
 * `reportFrontendError` 收成一条通道，按用户在「帮助 → 日志详细程度」里选择的档位
 * 过滤、去重、单行化后回传后端文件日志（复用既有 `log_frontend_error` 命令，
 * 不新增 IPC 通道）。
 *
 * 【为什么 Off 必须本地短路】`无` 档要求严格零产出：0 次 IPC、0 行 console、
 * 连 4000 字符的栈都不许先序列化。策略副本由启动时的 `getLogSettings()` 与
 * `hs:log-settings-changed` 事件共同维护，主窗口与卫星窗口都生效。
 */
import { getPluginHost } from "./pluginHost";
import type { LogSettingsView } from "./api/logSettings";

export type FrontendLevel = "error" | "warn" | "info" | "debug";

export interface FrontendLogPolicy {
    verbosity: "off" | "low" | "medium" | "high";
    accept: { error: boolean; warn: boolean; info: boolean; debug: boolean };
    detailCap: number;
    overridden: boolean;
}

const DEFAULT_POLICY: FrontendLogPolicy = {
    verbosity: "low",
    accept: { error: true, warn: false, info: false, debug: false },
    detailCap: 1000,
    overridden: false,
};

let policy: FrontendLogPolicy = DEFAULT_POLICY;
const listeners = new Set<(next: FrontendLogPolicy) => void>();

/** 策略矩阵与后端 `log_policy::LogVerbosity::policy` 逐字段对应。 */
export function policyFromView(view: LogSettingsView): FrontendLogPolicy {
    const verbosity = view.verbosity;
    const accept = {
        error: verbosity !== "off",
        warn: verbosity === "medium" || verbosity === "high",
        info: verbosity === "medium" || verbosity === "high",
        debug: verbosity === "high",
    };
    return {
        verbosity,
        accept,
        detailCap: verbosity === "high" ? 4000 : verbosity === "medium" ? 2000 : 1000,
        overridden: view.overridden,
    };
}

export function setFrontendLogPolicy(next: FrontendLogPolicy): void {
    policy = next;
    for (const listener of Array.from(listeners)) listener(next);
}

export function getFrontendLogPolicy(): FrontendLogPolicy {
    return policy;
}

export function subscribeFrontendLogPolicy(
    listener: (next: FrontendLogPolicy) => void,
): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

/** 仅测试用。 */
export function resetFrontendLogPolicyForTests(): void {
    policy = DEFAULT_POLICY;
    listeners.clear();
    resetDedupForTests();
}

// ── 去重（原 frontendErrorLog 的逻辑迁入，四个级别共用）────────────────────
const DEDUP_WINDOW_MS = 60_000;
const DEDUP_MAX_KEYS = 64;
type DedupState = { firstSeen: number; lastReported: number; count: number };
const dedupKeys = new Map<string, DedupState>();

export function resetFrontendErrorDedupForTests(): void {
    dedupKeys.clear();
}
const resetDedupForTests = resetFrontendErrorDedupForTests;

export function dedupDecision(key: string, now: number): string | null {
    const existing = dedupKeys.get(key);
    if (!existing) {
        if (dedupKeys.size >= DEDUP_MAX_KEYS) {
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
    if (now - existing.lastReported < DEDUP_WINDOW_MS) return null;
    const seconds = Math.max(1, Math.round((now - existing.firstSeen) / 1000));
    const repeats = existing.count - 1;
    existing.lastReported = now;
    existing.count = 1;
    existing.firstSeen = now;
    return ` (repeated ${repeats} time(s) over the last ${seconds}s)`;
}

export function toDetailText(detail: unknown, maxChars: number): string {
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
    text = text.replace(/\r/g, "\\r").replace(/\n/g, "\\n").replace(/\t/g, "\\t");
    if (text.length > maxChars) text = `${text.slice(0, maxChars)}…(truncated)`;
    return text;
}

/** 已知的"良性"错误消息模式：上报它们只会制造噪音。 */
const BENIGN_ERROR_PATTERNS: RegExp[] = [/^ResizeObserver loop/i];

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

/** 统一回传：Off / 未接收级别本地短路，永不抛错。 */
export function frontendLog(level: FrontendLevel, message: string, detail?: unknown): void {
    try {
        if (!policy.accept[level]) return;
        if (level === "error" && BENIGN_ERROR_PATTERNS.some((p) => p.test(message))) return;
        const messageText = message
            .replace(/\r/g, "\\r")
            .replace(/\n/g, "\\n")
            .replace(/\t/g, "\\t");
        const detailText = toDetailText(detail, policy.detailCap);
        const key = `${level}:${message}\u0000${detailText.slice(0, 200)}`;
        const suffix = dedupDecision(key, Date.now());
        if (suffix === null) return;
        const result = tauriInvokeRaw("log_frontend_error", {
            message: `${messageText}${suffix}`,
            detail: detailText || null,
            level,
        });
        if (result && typeof (result as Promise<unknown>).catch === "function") {
            (result as Promise<unknown>).catch(() => {});
        }
    } catch {
        // 日志回传自身绝不能抛错。
    }
}

/** 兼容既有 26 个调用点：语义等同于 `frontendLog("error", ...)`。 */
export function reportFrontendError(message: string, detail?: unknown): void {
    frontendLog("error", message, detail);
}

/** 安装全局兜底上报。在应用入口调用一次；Off 档下监听器注册后也会短路。 */
export function installGlobalErrorReporting(): void {
    if (typeof window === "undefined") return;
    window.addEventListener("error", (event) => {
        const message = event.message || "";
        const err = event.error as Error | undefined;
        frontendLog(
            "error",
            `Uncaught error: ${message || "(no message)"}`,
            err?.stack ?? `${event.filename}:${event.lineno}:${event.colno}`,
        );
    });
    window.addEventListener("unhandledrejection", (event) => {
        const reason = (event as PromiseRejectionEvent).reason;
        frontendLog(
            "error",
            `Unhandled rejection: ${reason instanceof Error ? reason.message : String(reason)}`,
            reason,
        );
    });
}

/** 启动时取一次策略并订阅广播；两个入口（主窗口 / 卫星窗口）都调它。 */
export async function installFrontendLogPolicy(): Promise<void> {
    try {
        const { getLogSettings } = await import("./api/logSettings");
        const { listen } = await import("./hostEvents");
        const apply = (view: LogSettingsView) => setFrontendLogPolicy(policyFromView(view));
        // 先订阅再读，避免读取与广播之间的窗口里丢一次变更。
        void listen<LogSettingsView>("hs:log-settings-changed", (event) => {
            if (event.payload && typeof event.payload === "object") apply(event.payload);
        });
        try {
            apply(await getLogSettings());
        } catch {
            // 后端尚未就绪（纯浏览器 / mock）：保持默认「低」，只回传 error。
        }
    } catch {
        // 动态导入失败也不该拖垮应用启动。
    }
}
