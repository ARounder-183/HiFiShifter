/**
 * 帮助菜单里的日志详细程度菜单项（**唯一入口**）。
 *
 * 形态沿用本文件 MenuBar.tsx 里语言切换的现成范例：一行只读 Label + 一个
 * RadioGroup + 四个 RadioItem。不新增对话框、不修改「关于」、不定制 macOS 原生
 * About。改动即写档；失败复用 MenuBar 既有的 setNotice / AppNoticeDialog，
 * 并把 RadioGroup 回滚到原档位。
 */
import { useCallback, useEffect, useState } from "react";
import { DropdownMenu } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import type { MessageKey } from "../../i18n/messages";
import {
    getLogSettings,
    setLogSettings,
    type LogSettingsView,
    type LogVerbosity,
} from "../../services/api/logSettings";
import { listen } from "../../services/hostEvents";
import { resolveLogVerbosityMenu } from "./logVerbosityMenuModel";

interface Props {
    /** 保存失败时交给 MenuBar 既有的纯通知对话框；不新增入口。 */
    onSaveFailed?: (message: string) => void;
}

export function LogVerbosityMenuItems({ onSaveFailed }: Props) {
    const { t, tf } = useI18n();
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
    const statusText =
        model.statusKey === "menu_log_current" && model.statusVars
            ? tf(model.statusKey).replace("{level}", t(model.statusVars.level as MessageKey))
            : t(model.statusKey);

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
        <>
            <DropdownMenu.Label>{tf("menu_log_verbosity")}</DropdownMenu.Label>
            <DropdownMenu.Label>{statusText}</DropdownMenu.Label>
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
        </>
    );
}
