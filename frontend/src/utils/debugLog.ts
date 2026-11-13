/**
 * 通用诊断日志通道（与 `waveformDebug.ts` 同一取舍）。
 *
 * 【为什么收口到前端统一通道】散落的 `console.log` 即使被 localStorage 开关守卫，
 * 仍是生产包里的活代码；而 localStorage 开关与用户选择的日志档位毫无关系 ——
 * 「无」档下它们照样输出，"高"档下的内容也进不了文件日志。现在全部经
 * `frontendLog("debug", ...)`：只有「高」档才回传，Off 档本地短路。
 */
import { frontendLog, getFrontendLogPolicy, type FrontendLevel } from "../services/frontendLog";

export interface DebugLogger {
    /** 当前档位是否接收 debug 级（由后端策略广播驱动）。 */
    readonly enabled: boolean;
    log(...args: unknown[]): void;
    info(...args: unknown[]): void;
    warn(...args: unknown[]): void;
}

/**
 * 创建一个按日志详细程度过滤的诊断 logger。
 *
 * `flag` 只作为消息前缀保留（既有调用点按开关名区分诊断族），**不再**读取
 * localStorage —— 那三个独立开关已收编到「高」档。
 */
export function createDebugLogger(flag: string): DebugLogger {
    const emit = (level: FrontendLevel, args: unknown[]): void => {
        const message = args
            .map((arg) => (typeof arg === "string" ? arg : safeStringify(arg)))
            .join(" ");
        frontendLog(level, `[${flag}] ${message}`);
    };
    return {
        get enabled() {
            return getFrontendLogPolicy().accept.debug;
        },
        log: (...args) => emit("debug", args),
        info: (...args) => emit("debug", args),
        warn: (...args) => emit("debug", args),
    };
}

function safeStringify(value: unknown): string {
    if (value instanceof Error) return value.stack || value.message;
    try {
        return JSON.stringify(value) ?? String(value);
    } catch {
        return String(value);
    }
}
