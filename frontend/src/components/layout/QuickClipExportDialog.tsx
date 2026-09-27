import { useEffect, useMemo, useState } from "react";
import { Button, Flex, SegmentedControl, Text, TextField } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import type { MessageKey } from "../../i18n/messages";
import { coreApi, type ExportFormat } from "../../services/api/core";
import { fileBrowserApi } from "../../services/api/fileBrowser";
import { applyExtensionToFileName } from "../../utils/exportFormat";
import { buildQuickExportFileName } from "./timeline/quickExportSelection";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm } from "../../ui/Field";

interface QuickClipExportDialogProps {
    open: boolean;
    clipIds: string[];
    onOpenChange: (open: boolean) => void;
}

export function QuickClipExportDialog({ open, clipIds, onOpenChange }: QuickClipExportDialogProps) {
    const { t } = useI18n();
    const [outputDir, setOutputDir] = useState("");
    const [fileName, setFileName] = useState("");
    // 快捷导出不暴露编码参数：格式可选，参数全部沿用导出对话框的持久化设置。
    const [format, setFormat] = useState<ExportFormat>("wav");
    const [errorText, setErrorText] = useState("");
    const [submitting, setSubmitting] = useState(false);

    const exportDisabled = useMemo(
        () => submitting || clipIds.length === 0,
        [clipIds.length, submitting],
    );

    useEffect(() => {
        if (!open) return;
        let cancelled = false;
        setErrorText("");
        setSubmitting(false);

        void coreApi
            .getExportAudioDefaults()
            .then((defaults) => {
                if (cancelled || !defaults.ok) return;
                const nextFormat: ExportFormat =
                    defaults.format === "mp3" || defaults.format === "flac"
                        ? defaults.format
                        : "wav";
                setFormat(nextFormat);
                setOutputDir(defaults.projectOutputDir ?? "");
                setFileName(
                    applyExtensionToFileName(
                        buildQuickExportFileName(defaults.projectName ?? ""),
                        nextFormat,
                    ),
                );
            })
            .catch(() => {
                if (!cancelled) {
                    setFileName(buildQuickExportFileName(""));
                }
            });

        return () => {
            cancelled = true;
        };
    }, [open]);

    function handleFormatChange(next: ExportFormat) {
        setFormat(next);
        setFileName((name) => applyExtensionToFileName(name, next));
    }

    async function handleBrowse() {
        const result = await fileBrowserApi.pickDirectory();
        if (!result.ok) {
            setErrorText(t("quick_export_error_pick_directory_failed"));
            return;
        }
        if (!result.canceled && result.path) {
            setOutputDir(result.path);
            setErrorText("");
        }
    }

    async function handleExport() {
        if (clipIds.length === 0) {
            setErrorText(t("quick_export_error_no_clips"));
            return;
        }
        if (!outputDir.trim()) {
            setErrorText(t("quick_export_error_missing_output_dir"));
            return;
        }
        if (!fileName.trim()) {
            setErrorText(t("quick_export_error_missing_file_name"));
            return;
        }

        setSubmitting(true);
        setErrorText("");
        try {
            const result = await coreApi.quickExportSelectedClips({
                clipIds,
                outputDir: outputDir.trim(),
                fileName: fileName.trim(),
                format,
            });
            if (!result.ok) {
                const errorKey =
                    result.error === "quick_export_output_dir_required"
                        ? "quick_export_error_missing_output_dir"
                        : result.error === "quick_export_file_name_required"
                          ? "quick_export_error_missing_file_name"
                          : result.error === "mp3_unsupported_sample_rate"
                            ? "export_dialog_error_mp3_unsupported_sample_rate"
                            : null;
                setErrorText(
                    errorKey ? t(errorKey as MessageKey) : String(result.error ?? "Export failed"),
                );
                return;
            }
            onOpenChange(false);
        } catch (error) {
            setErrorText(error instanceof Error ? error.message : "Export failed");
        } finally {
            setSubmitting(false);
        }
    }

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={t("quick_export_title")}
            description={t("quick_export_description").replace("{n}", String(clipIds.length))}
            size="md"
            actions={[
                { id: "cancel", label: t("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "export",
                    label: submitting ? t("quick_export_submitting") : t("quick_export_confirm"),
                    intent: "primary",
                    disabled: exportDisabled,
                    onClick: handleExport,
                },
            ]}
        >
            <AppForm>
                <AppField label={t("quick_export_format")}>
                    <SegmentedControl.Root
                        value={format}
                        onValueChange={(value) => handleFormatChange(value as ExportFormat)}
                    >
                        <SegmentedControl.Item value="wav">WAV</SegmentedControl.Item>
                        <SegmentedControl.Item value="mp3">MP3</SegmentedControl.Item>
                        <SegmentedControl.Item value="flac">FLAC</SegmentedControl.Item>
                    </SegmentedControl.Root>
                </AppField>
                <AppField label={t("quick_export_file_name")}>
                    <TextField.Root
                        value={fileName}
                        onChange={(event) => setFileName(event.target.value)}
                        placeholder="quick_export.wav"
                    />
                </AppField>
                <AppField label={t("quick_export_output_dir")}>
                    <Flex gap="2">
                        <TextField.Root
                            value={outputDir}
                            onChange={(event) => setOutputDir(event.target.value)}
                        />
                        <Button type="button" variant="soft" onClick={() => void handleBrowse()}>
                            {t("quick_export_browse")}
                        </Button>
                    </Flex>
                </AppField>
                {errorText ? (
                    <Text size="2" color="red">
                        {errorText}
                    </Text>
                ) : null}
            </AppForm>
        </AppDialog>
    );
}
