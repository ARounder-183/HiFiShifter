/**
 * 帮助菜单里的「日志详细程度」**子菜单**（唯一入口）。
 *
 * 【为什么收进子菜单】一级菜单里平铺四档会压低信息密度。这里与
 * 「选项 → 推理设备」同形：一级只有一行带当前值的触发项，展开后**只有四个互斥
 * 档位**；不放重复的标题、当前值、说明或分隔线。
 *
 * 改动即写档；失败复用 MenuBar 既有的 setNotice / AppNoticeDialog，并把选中项
 * 回滚到原档位。
 *
 * 【为什么是 hook 而不是组件】它产出的是一段**菜单条目数据**（一级的触发项 +
 * 四个单选项），而不是一张独立的菜单表面。做成组件时它只能渲染成 Radix 的
 * `DropdownMenu.Sub`，于是"帮助菜单"这一张菜单里混着两种条目来源；做成 hook 之后
 * 它返回 `AppMenuItemSpec`，与其它菜单走同一条渲染路径 —— 键盘、滚动、勾选语义
 * 全都自动一致。
 */
import { useCallback, useEffect, useState } from "react";

import type { AppMenuItemSpec } from "../../ui/Menu";
import { useI18n } from "../../i18n/I18nProvider";
import {
    getLogSettings,
    setLogSettings,
    type LogSettingsView,
    type LogVerbosity,
} from "../../services/api/logSettings";
import { listen } from "../../services/hostEvents";
import { logLevelLabelKey, resolveLogVerbosityMenu } from "./logVerbosityMenuModel";

/** 帮助菜单里那一行 + 它展开后的四个档位。 */
export interface LogVerbosityMenu {
    /** 一级菜单那一行的文案（含当前档位）。 */
    triggerLabel: string;
    /** 保存进行中：整项不可展开（避免连点造成两次写盘）。 */
    disabled: boolean;
    /** 四个互斥档位。 */
    items: AppMenuItemSpec[];
}

interface Options {
    /** 保存失败时交给 MenuBar 既有的纯通知对话框；不新增入口。 */
    onSaveFailed?: (message: string) => void;
}

export function useLogVerbosityMenu({ onSaveFailed }: Options = {}): LogVerbosityMenu {
    const { tf, tVars } = useI18n();
    const [view, setView] = useState<LogSettingsView | null>(null);
    const [selected, setSelected] = useState<LogVerbosity>("low");
    const [busy, setBusy] = useState(false);

    const apply = useCallback((next: LogSettingsView | null) => {
        setView(next);
        setSelected(next?.verbosity ?? "low");
    }, []);

    useEffect(() => {
        let disposed = false;
        void getLogSettings()
            .then((next) => {
                if (!disposed) apply(next);
            })
            .catch(() => {});
        let unsubscribe: (() => void) | undefined;
        void listen<LogSettingsView>("hs:log-settings-changed", (event) => {
            if (event.payload && typeof event.payload === "object") apply(event.payload);
        }).then((off) => {
            if (disposed) off();
            else unsubscribe = off;
        });
        return () => {
            disposed = true;
            unsubscribe?.();
        };
    }, [apply]);

    const model = resolveLogVerbosityMenu(view);

    const onChange = (value: string) => {
        const next = value as LogVerbosity;
        const previous = selected;
        setSelected(next);
        setBusy(true);
        void setLogSettings(next)
            .then((result) => {
                if (!result.ok) {
                    setSelected(previous);
                    onSaveFailed?.(
                        tf("menu_log_saved_failed") + (result.error ? ": " + result.error : ""),
                    );
                    return;
                }
                apply(result);
            })
            .catch((error: unknown) => {
                setSelected(previous);
                onSaveFailed?.(tf("menu_log_saved_failed") + ": " + String(error));
            })
            .finally(() => setBusy(false));
    };

    return {
        triggerLabel: tVars("menu_log_verbosity", {
            level: tf(logLevelLabelKey(selected)),
        }),
        disabled: busy,
        items: model.options.map((option) => ({
            key: option.value,
            label: tf(option.labelKey),
            // 【必须是 radio】四档互斥；挂成 checkbox 时读屏会念"复选框"，
            // 用户听到的是"可以多选"。
            radio: true,
            checked: selected === option.value,
            disabled: option.disabled || busy,
            onSelect: () => onChange(option.value),
        })),
    };
}
