/**
 * 帮助菜单里「日志详细程度」子菜单的**纯模型**。
 *
 * 【为什么单独抽出来】MenuBar.tsx 有 2000+ 行且没有测试宿主；纯函数让"四个
 * 单选项、顺序、默认档位、覆盖时禁用"有可自动执行的回归门禁。
 *
 * 【为什么没有状态行】一级触发行已经显示当前档位；再把标题、当前值、覆盖说明
 * 或健康状态塞进 SubContent 只会重复占行。用户要求子菜单里只有四档本身。
 */
import type { LogSettingsView, LogVerbosity } from "../../services/api/logSettings";

export type LogLevelLabelKey =
    "menu_log_level_off" | "menu_log_level_low" | "menu_log_level_medium" | "menu_log_level_high";

export interface LogVerbosityMenuOption {
    value: LogVerbosity;
    labelKey: LogLevelLabelKey;
    disabled: boolean;
}

export interface LogVerbosityMenuModel {
    options: LogVerbosityMenuOption[];
    checked: LogVerbosity;
    overridden: boolean;
}

const OPTIONS: ReadonlyArray<{
    value: LogVerbosity;
    labelKey: LogLevelLabelKey;
}> = [
    { value: "off", labelKey: "menu_log_level_off" },
    { value: "low", labelKey: "menu_log_level_low" },
    { value: "medium", labelKey: "menu_log_level_medium" },
    { value: "high", labelKey: "menu_log_level_high" },
];

const LEVEL_LABEL_KEYS: Record<LogVerbosity, LogLevelLabelKey> = {
    off: "menu_log_level_off",
    low: "menu_log_level_low",
    medium: "menu_log_level_medium",
    high: "menu_log_level_high",
};

/** 任一档位对应的本地化标签键（一级触发行也用它）。 */
export function logLevelLabelKey(level: LogVerbosity): LogLevelLabelKey {
    return LEVEL_LABEL_KEYS[level];
}

/** 把后端视图解析成菜单需要的全部状态；不碰 React、不碰 IPC。 */
export function resolveLogVerbosityMenu(view: LogSettingsView | null): LogVerbosityMenuModel {
    const overridden = view?.overridden ?? false;
    return {
        options: OPTIONS.map((option) => ({
            ...option,
            // 启动参数/环境变量覆盖时四档全禁用，避免"改了没反应"。
            disabled: overridden,
        })),
        checked: view?.verbosity ?? "low",
        overridden,
    };
}
