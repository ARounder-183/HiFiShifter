/**
 * 通用诊断日志通道（与 `waveformDebug.ts` 同一取舍）。
 *
 * 【为什么要收口】散落的 `console.log` 即使被 localStorage 开关守卫，仍是生产包
 * 里的活代码；每处各写一份开关读取与 `if` 判断，容易漏守卫或读错键名。这里把
 * "按开关名探测 + 节流缓存"做成唯一实现，调用方只声明自己用哪个开关。
 *
 * 【为什么开关探测要节流】诊断钩子可能挂在高频路径上，逐次同步读 localStorage
 * 会拖慢热路径（部分平台还会触发存储访问警告）。与 waveformDebug 一致：开关
 * 变化最多 2 秒后生效，足够用于排查。
 */
const FLAG_CHECK_INTERVAL_MS = 2000;

export interface DebugLogger {
    /** 当前开关是否开启（节流后的读数）。 */
    readonly enabled: boolean;
    log(...args: unknown[]): void;
    info(...args: unknown[]): void;
    warn(...args: unknown[]): void;
}

/**
 * 创建一个受 localStorage 开关控制的诊断 logger。
 *
 * @param flag localStorage 键名，例如 `"hifishifter.debugDnd"`；值为 `"1"` 时开启。
 */
export function createDebugLogger(flag: string): DebugLogger {
    let enabled = false;
    let lastCheck = 0;

    const isEnabled = (): boolean => {
        const now = Date.now();
        if (now - lastCheck < FLAG_CHECK_INTERVAL_MS) return enabled;
        lastCheck = now;
        enabled =
            typeof window !== "undefined" && window.localStorage?.getItem(flag) === "1";
        return enabled;
    };

    const emit = (level: "log" | "info" | "warn", args: unknown[]): void => {
        if (!isEnabled()) return;
        console[level](...args);
    };

    return {
        get enabled() {
            return isEnabled();
        },
        log: (...args) => emit("log", args),
        info: (...args) => emit("info", args),
        warn: (...args) => emit("warn", args),
    };
}
