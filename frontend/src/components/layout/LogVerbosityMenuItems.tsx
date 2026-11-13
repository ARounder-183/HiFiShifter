/**
 * 帮助菜单里的「日志详细程度」**子菜单**（唯一入口）。
 *
 * 【为什么收进子菜单】一级菜单里平铺 Label + 4 个单选项要占 6 行，把
 * 「打开日志文件夹 / 导出诊断信息 / 复制诊断摘要 / 关于」挤到很低的密度。
 * 项目自己的「选项 → 推理设备」已经给出范例：一级只显示一行带当前值的
 * `SubTrigger`，四个互斥档位与只读状态放进 `SubContent`。菜单项总数不变，
 * 仍没有对话框、设置面板或第二个入口。
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
    // 【为什么用 selected 而不是 view.verbosity】点击后先乐观更新触发行；
    // 覆盖态 / 日志不可用时 statusVars 为空，也不能退化成写死的"低"。
    const levelText = t(logLevelLabelKey(selected));
    const triggerText = tVars("menu_log_verbosity_current", { level: levelText });
    const statusText =
        model.statusKey === "menu_log_current"
            ? tVars("menu_log_current", { level: t(logLevelLabelKey(model.checked)) })
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
        <DropdownMenu.Sub>
            <DropdownMenu.SubTrigger disabled={busy}>{triggerText}</DropdownMenu.SubTrigger>
            <DropdownMenu.SubContent>
                <DropdownMenu.Label>{tf("menu_log_verbosity")}</DropdownMenu.Label>
                <DropdownMenu.Label>{statusText}</DropdownMenu.Label>
                <DropdownMenu.Separator />
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
