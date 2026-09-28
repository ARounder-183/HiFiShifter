/*
 * 自动备份设置对话框。
 *
 * 功能：
 * - 保存时备份开关
 * - 定时备份开关、间隔与路径模板设置
 * - 占位符快捷插入（<ProjectFolder> / <ProjectName>）
 */

import { useEffect, useRef, useState, type ChangeEvent } from "react";
import { Button, Flex, Text, TextField } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import { projectApi, type AutoBackupSettings } from "../../services/api/project";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm } from "../../ui/Field";

interface AutoBackupDialogProps {
    open: boolean;
    settings: AutoBackupSettings;
    onOpenChange: (open: boolean) => void;
    onSettingsSaved: (settings: AutoBackupSettings) => void;
}

function normalizeIntervalSec(raw: number): number {
    if (!Number.isFinite(raw)) return 300;
    return Math.max(1, Math.min(86_400, Math.floor(raw)));
}

export function AutoBackupDialog({
    open,
    settings,
    onOpenChange,
    onSettingsSaved,
}: AutoBackupDialogProps) {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;

    const [draft, setDraft] = useState<AutoBackupSettings>(settings);
    const [submitting, setSubmitting] = useState(false);
    const [errorText, setErrorText] = useState("");
    const pathInputRef = useRef<HTMLInputElement | null>(null);

    useEffect(() => {
        if (!open) {
            pathInputRef.current = null;
            return;
        }
        setDraft(settings);
        setSubmitting(false);
        setErrorText("");
    }, [open, settings]);

    function getPathInputElement(): HTMLInputElement | null {
        const input = pathInputRef.current;
        if (!input?.isConnected) {
            pathInputRef.current = null;
            return null;
        }

        return input;
    }

    function insertPathToken(token: string) {
        const input = getPathInputElement();
        if (!input) return;

        const start = input.selectionStart ?? input.value.length;
        const end = input.selectionEnd ?? input.value.length;
        const nextValue = `${input.value.slice(0, start)}${token}${input.value.slice(end)}`;

        setDraft((prev) => ({
            ...prev,
            timedBackupPathTemplate: nextValue,
        }));

        window.requestAnimationFrame(() => {
            input.focus();
            const nextPos = start + token.length;
            input.setSelectionRange(nextPos, nextPos);
        });
    }

    async function handleSave() {
        setErrorText("");
        setSubmitting(true);

        const nextSettings: AutoBackupSettings = {
            saveOnSaveEnabled: Boolean(draft.saveOnSaveEnabled),
            timedBackupEnabled: Boolean(draft.timedBackupEnabled),
            timedBackupIntervalSec: normalizeIntervalSec(Number(draft.timedBackupIntervalSec)),
            timedBackupPathTemplate: String(draft.timedBackupPathTemplate ?? "").trim(),
        };

        try {
            const result = await projectApi.saveAutoBackupSettings(nextSettings);
            if (!result?.ok) {
                setErrorText(tAny("auto_backup_save_failed"));
                return;
            }
            const saved = result.settings ?? nextSettings;
            onSettingsSaved(saved);
            onOpenChange(false);
        } catch {
            setErrorText(tAny("auto_backup_save_failed"));
        } finally {
            setSubmitting(false);
        }
    }

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("menu_auto_backup")}
            description={tAny("auto_backup_dialog_desc")}
            size="xl"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "save",
                    label: tAny("auto_backup_save_settings"),
                    intent: "primary",
                    disabled: submitting,
                    onClick: handleSave,
                },
            ]}
        >
            <AppForm labelWidth="lg">
                <label className="flex items-center gap-2 text-qt-md text-qt-text">
                    <input
                        type="checkbox"
                        checked={draft.saveOnSaveEnabled}
                        onChange={(event: ChangeEvent<HTMLInputElement>) =>
                            setDraft((prev) => ({
                                ...prev,
                                saveOnSaveEnabled: event.target.checked,
                            }))
                        }
                    />
                    <span>{tAny("auto_backup_save_on_save")}</span>
                </label>

                <label className="flex items-center gap-2 text-qt-md text-qt-text">
                    <input
                        type="checkbox"
                        checked={draft.timedBackupEnabled}
                        onChange={(event: ChangeEvent<HTMLInputElement>) =>
                            setDraft((prev) => ({
                                ...prev,
                                timedBackupEnabled: event.target.checked,
                            }))
                        }
                    />
                    <span>{tAny("auto_backup_timed")}</span>
                </label>

                <AppField label={tAny("auto_backup_interval_sec")}>
                    <Flex align="center" gap="2">
                        <TextField.Root
                            size="2"
                            type="number"
                            min={1}
                            step={1}
                            value={String(draft.timedBackupIntervalSec)}
                            onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                setDraft((prev) => ({
                                    ...prev,
                                    timedBackupIntervalSec: Number(event.target.value),
                                }))
                            }
                            style={{ width: 180 }}
                        />
                        <Text size="1" color="gray">
                            sec
                        </Text>
                    </Flex>
                </AppField>

                <AppField label={tAny("auto_backup_path_template")}>
                    <TextField.Root
                        size="2"
                        value={draft.timedBackupPathTemplate}
                        onChange={(event: ChangeEvent<HTMLInputElement>) =>
                            setDraft((prev) => ({
                                ...prev,
                                timedBackupPathTemplate: event.target.value,
                            }))
                        }
                        onFocus={(event) => {
                            pathInputRef.current = event.target as HTMLInputElement;
                        }}
                    />
                </AppField>

                <Flex gap="2" wrap="wrap" align="center">
                    <Text size="1" color="gray">
                        {tAny("auto_backup_placeholders")}
                    </Text>
                    {["<ProjectFolder>", "<ProjectName>"].map((token) => (
                        <Button
                            key={token}
                            type="button"
                            size="1"
                            variant="ghost"
                            color="gray"
                            onClick={() => insertPathToken(token)}
                        >
                            {token}
                        </Button>
                    ))}
                </Flex>

                <Text size="1" color="gray">
                    {tAny("auto_backup_time_format_hint")}
                </Text>

                {errorText ? (
                    <Text size="2" color="red">
                        {errorText}
                    </Text>
                ) : null}
            </AppForm>
        </AppDialog>
    );
}
