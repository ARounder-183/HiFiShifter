/**
 * 帮助菜单里的「日志详细程度」**子菜单**（唯一入口）。
 *
 * 【为什么收进子菜单】一级菜单里平铺四档会压低信息密度。这里与
 * 「选项 → 推理设备」同形：一级只有一行带当前值的 `SubTrigger`，展开后
 * **只有四个互斥档位**；不放重复的标题、当前值、说明或分隔线。
 *
 * 改动即写档；失败复用 MenuBar 既有的 setNotice / AppNoticeDialog，
 * 并把 RadioGroup 回滚到原档位。
 */
import { useCallback, useEffect, useState } from "react";
import { DropdownMenu } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import {
    getLogSettings,
    setLogSettings,
    type LogSettingsView,
    type LogVerbosity,
} from "../../services/api/logSettings";
import { listen } from "../../services/hostEvents";
import { logLevelLabelKey, resolveLogVerbosityMenu } from "./logVerbosityMenuModel";

interface Props {
    /** 保存失败时交给 MenuBar 既有的纯通知对话框；不新增入口。 */
    onSaveFailed?: (message: string) => void;
}

export function LogVerbosityMenuItems({ onSaveFailed }: Props) {
    const { t, tf, tVars } = useI18n();
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
    const triggerText = tVars("menu_log_verbosity", {
        level: t(logLevelLabelKey(selected)),
    });

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

    return (
        <DropdownMenu.Sub>
            <DropdownMenu.SubTrigger disabled={busy}>{triggerText}</DropdownMenu.SubTrigger>
            <DropdownMenu.SubContent>
                <DropdownMenu.RadioGroup value={selected} onValueChange={onChange}>
                    {model.options.map((option) => (
                        <DropdownMenu.RadioItem
                            key={option.value}
                            value={option.value}
                            disabled={option.disabled || busy}
                        >
                            {tf(option.labelKey)}
                        </DropdownMenu.RadioItem>
                    ))}
                </DropdownMenu.RadioGroup>
            </DropdownMenu.SubContent>
        </DropdownMenu.Sub>
    );
}
