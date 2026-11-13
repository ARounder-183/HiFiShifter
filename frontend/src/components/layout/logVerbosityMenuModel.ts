/**
 * 帮助菜单里「日志详细程度」的**纯模型**。
 *
 * 【为什么单独抽出来】MenuBar.tsx 有 2000+ 行且没有测试宿主（既有
 * `logFolderNotice.ts` 就是同一取舍）。把"四个单选项、选中态、覆盖态、状态行"
 * 做成纯函数后，这些判定才有可自动执行的回归门禁。
 */
import type { LogSettingsView, LogVerbosity } from "../../services/api/logSettings";

export type LogMenuStatusKey = "menu_log_current" | "menu_log_overridden" | "menu_log_unavailable";

export interface LogVerbosityMenuOption {
    value: LogVerbosity;
    labelKey:
        | "menu_log_level_off"
        | "menu_log_level_low"
        | "menu_log_level_medium"
        | "menu_log_level_high";
    disabled: boolean;
}

export interface LogVerbosityMenuModel {
    options: LogVerbosityMenuOption[];
    checked: LogVerbosity;
    overridden: boolean;
    statusKey: LogMenuStatusKey;
    statusVars?: Record<string, string>;
}

const OPTIONS: ReadonlyArray<{
    value: LogVerbosity;
    labelKey: LogVerbosityMenuOption["labelKey"];
}> = [
    { value: "off", labelKey: "menu_log_level_off" },
    { value: "low", labelKey: "menu_log_level_low" },
    { value: "medium", labelKey: "menu_log_level_medium" },
    { value: "high", labelKey: "menu_log_level_high" },
];

const LEVEL_LABEL_KEYS: Record<LogVerbosity, LogVerbosityMenuOption["labelKey"]> = {
    off: "menu_log_level_off",
    low: "menu_log_level_low",
    medium: "menu_log_level_medium",
    high: "menu_log_level_high",
};

/** 把后端视图解析成菜单需要的全部状态；不碰 React、不碰 IPC。 */
export function resolveLogVerbosityMenu(view: LogSettingsView | null): LogVerbosityMenuModel {
    const checked: LogVerbosity = view?.verbosity ?? "low";
    const overridden = view?.overridden ?? false;
    const unhealthy = view ? !view.loggingHealthy || view.hostLoggerOwned : false;
    const options: LogVerbosityMenuOption[] = OPTIONS.map((option) => ({
        ...option,
        // 启动参数/环境变量覆盖时四档全禁用，避免"改了没反应"。
        disabled: overridden,
    }));
    const statusKey: LogMenuStatusKey = overridden
        ? "menu_log_overridden"
        : unhealthy
          ? "menu_log_unavailable"
          : "menu_log_current";
    return {
        options,
        checked,
        overridden,
        statusKey,
        statusVars:
            statusKey === "menu_log_current" ? { level: LEVEL_LABEL_KEYS[checked] } : undefined,
    };
}
