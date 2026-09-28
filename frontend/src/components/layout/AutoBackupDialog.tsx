/*
 * 自动备份设置对话框。
 *
 * 功能：
 * - 保存时备份开关
 * - 定时备份开关、间隔与路径模板设置
 * - 占位符快捷插入（<ProjectFolder> / <ProjectName>）
 */

import { useEffect, useRef, useState, type ChangeEvent } from "react";
import { Flex, TextField } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import { projectApi, type AutoBackupSettings } from "../../services/api/project";
import { AppButton } from "../../ui";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm, AppSwitchRow } from "../../ui/Field";
import { AppNumberField } from "../../ui";

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
    const { tf } = useI18n();

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
                setErrorText(tf("auto_backup_save_failed"));
                return;
            }
            const saved = result.settings ?? nextSettings;
            onSettingsSaved(saved);
            onOpenChange(false);
        } catch {
            setErrorText(tf("auto_backup_save_failed"));
        } finally {
            setSubmitting(false);
        }
    }

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("menu_auto_backup")}
            description={tf("auto_backup_dialog_desc")}
            size="xl"
            actions={[
                { id: "cancel", label: tf("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "save",
                    label: tf("auto_backup_save_settings"),
                    intent: "primary",
                    disabled: submitting,
                    onClick: handleSave,
                },
            ]}
        >
            <AppForm labelWidth="lg">
                <AppSwitchRow
                    control="checkbox"
                    label={tf("auto_backup_save_on_save")}
                    checked={draft.saveOnSaveEnabled}
                    onCheckedChange={(checked) =>
                        setDraft((prev) => ({
                            ...prev,
                            saveOnSaveEnabled: checked,
                        }))
                    }
                />

                <AppSwitchRow
                    control="checkbox"
                    label={tf("auto_backup_timed")}
                    checked={draft.timedBackupEnabled}
                    onCheckedChange={(checked) =>
                        setDraft((prev) => ({
                            ...prev,
                            timedBackupEnabled: checked,
                        }))
                    }
                />

                <AppField label={tf("auto_backup_interval_sec")}>
                    <AppNumberField
                        value={draft.timedBackupIntervalSec}
                        unit="integer"
                        min={1}
                        width={180}
                        suffix="sec"
                        ariaLabel={tf("auto_backup_interval_sec")}
                        onCommit={(timedBackupIntervalSec) =>
                            setDraft((prev) => ({
                                ...prev,
                                timedBackupIntervalSec,
                            }))
                        }
                    />
                </AppField>

                <AppField label={tf("auto_backup_path_template")}>
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
                    <span className="hs-type-caption">{tf("auto_backup_placeholders")}</span>
                    {["<ProjectFolder>", "<ProjectName>"].map((token) => (
                        <AppButton key={token} size="sm" onClick={() => insertPathToken(token)}>
                            {token}
                        </AppButton>
                    ))}
                </Flex>

                <span className="hs-type-caption">{tf("auto_backup_time_format_hint")}</span>

                {errorText ? (
                    <span className="hs-type-body" style={{ color: "var(--qt-danger-text)" }}>
                        {errorText}
                    </span>
                ) : null}
            </AppForm>
        </AppDialog>
    );
}
