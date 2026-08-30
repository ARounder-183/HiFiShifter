import { useEffect, useMemo, useState } from "react";
import { Button, Dialog, Flex, Text, TextField } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import type { MessageKey } from "../../i18n/messages";
import { coreApi } from "../../services/api/core";
import { fileBrowserApi } from "../../services/api/fileBrowser";
import { buildQuickExportFileName } from "./timeline/quickExportSelection";

interface QuickClipExportDialogProps {
    open: boolean;
    clipIds: string[];
    onOpenChange: (open: boolean) => void;
}

export function QuickClipExportDialog({ open, clipIds, onOpenChange }: QuickClipExportDialogProps) {
    const { t } = useI18n();
    const [outputDir, setOutputDir] = useState("");
    const [fileName, setFileName] = useState("");
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
                setOutputDir(defaults.projectOutputDir ?? "");
                setFileName(buildQuickExportFileName(defaults.projectName ?? ""));
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
            });
            if (!result.ok) {
                const errorKey =
                    result.error === "quick_export_output_dir_required"
                        ? "quick_export_error_missing_output_dir"
                        : result.error === "quick_export_file_name_required"
                          ? "quick_export_error_missing_file_name"
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
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content maxWidth="520px">
                <Dialog.Title>{t("quick_export_title")}</Dialog.Title>
                <Dialog.Description>
                    {t("quick_export_description").replace("{n}", String(clipIds.length))}
                </Dialog.Description>
                <Flex direction="column" gap="3" mt="4">
                    <div>
                        <Text as="label" size="2">
                            {t("quick_export_file_name")}
                        </Text>
                        <TextField.Root
                            mt="1"
                            value={fileName}
                            onChange={(event) => setFileName(event.target.value)}
                            placeholder="quick_export.wav"
                        />
                    </div>
                    <div>
                        <Text as="label" size="2">
                            {t("quick_export_output_dir")}
                        </Text>
                        <Flex gap="2" mt="1">
                            <TextField.Root
                                className="flex-1"
                                value={outputDir}
                                onChange={(event) => setOutputDir(event.target.value)}
                            />
                            <Button variant="soft" onClick={() => void handleBrowse()}>
                                {t("quick_export_browse")}
                            </Button>
                        </Flex>
                    </div>
                    {errorText ? (
                        <Text size="2" color="red">
                            {errorText}
                        </Text>
                    ) : null}
                </Flex>
                <Flex justify="end" gap="2" mt="4">
                    <Button variant="soft" color="gray" onClick={() => onOpenChange(false)}>
                        {t("cancel")}
                    </Button>
                    <Button disabled={exportDisabled} onClick={() => void handleExport()}>
                        {submitting ? t("quick_export_submitting") : t("quick_export_confirm")}
                    </Button>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}
