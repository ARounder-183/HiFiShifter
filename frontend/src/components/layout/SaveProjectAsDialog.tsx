import { useEffect, useState } from "react";
import { Flex } from "@radix-ui/themes";

import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { persistUiSettings, setProjectLastSaveFormat } from "../../features/session/sessionSlice";
import { saveProjectToPathRemote } from "../../features/session/thunks/projectThunks";
import { webApi } from "../../services/webviewApi";
import type { ProjectSaveFormat } from "../../services/api/project";
import { useI18n } from "../../i18n/I18nProvider";
import { AppButton, AppDialog } from "../../ui";

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    initialFormat?: ProjectSaveFormat;
}

const FORMATS: readonly {
    id: ProjectSaveFormat;
    titleKey: "save_as_format_hshp" | "save_as_format_json" | "save_as_format_zip";
    descKey: "save_as_desc_hshp" | "save_as_desc_json" | "save_as_desc_zip";
}[] = [
    { id: "hshp", titleKey: "save_as_format_hshp", descKey: "save_as_desc_hshp" },
    { id: "json", titleKey: "save_as_format_json", descKey: "save_as_desc_json" },
    { id: "zip", titleKey: "save_as_format_zip", descKey: "save_as_desc_zip" },
];

/**
 * 另存为格式选择。
 *
 * 格式必须先在应用内选：macOS 的 NSSavePanel 不提供可选的“文件类型”
 * ComboBox，rfd 的 filter 只表现为扩展名限制。原生对话框在这里只负责
 * 选择路径，写入格式由显式参数传给后端。
 */
export function SaveProjectAsDialog({ open, onOpenChange, initialFormat }: Props) {
    const { tf } = useI18n();
    const dispatch = useAppDispatch();
    const project = useAppSelector((state) => state.session.project);
    const [format, setFormat] = useState<ProjectSaveFormat>("hshp");
    const [error, setError] = useState<string | null>(null);
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        if (!open) return;
        setFormat(initialFormat ?? project.lastSaveFormat ?? "hshp");
        setError(null);
        setSaving(false);
    }, [open, initialFormat, project.lastSaveFormat]);

    const confirmSave = async () => {
        if (saving) return;
        setError(null);
        dispatch(setProjectLastSaveFormat(format));
        void dispatch(persistUiSettings());
        const picked = await webApi.pickProjectSavePath(format, project.name || "Untitled");
        if (!picked?.ok) {
            setError(picked?.error ?? "save_project_as_failed");
            return;
        }
        if (picked.canceled || !picked.path) return;
        setSaving(true);
        try {
            await dispatch(
                saveProjectToPathRemote({ path: picked.path, format, force: false }),
            ).unwrap();
            onOpenChange(false);
        } catch (reason) {
            const message = typeof reason === "string" ? reason : "save_project_as_failed";
            setError(
                message === "format_extension_mismatch" ? tf("save_as_format_mismatch") : message,
            );
        } finally {
            setSaving(false);
        }
    };

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("save_as_dialog_title")}
            size="md"
            actions={[
                {
                    id: "cancel",
                    label: tf("cancel"),
                    onClick: () => onOpenChange(false),
                },
                {
                    id: "continue",
                    label: tf("save_as_continue"),
                    intent: "primary",
                    disabled: saving,
                    onClick: () => void confirmSave(),
                },
            ]}
        >
            <Flex direction="column" gap="2">
                {FORMATS.map((entry) => {
                    const selected = format === entry.id;
                    return (
                        <AppButton
                            key={entry.id}
                            intent={selected ? "primary" : "default"}
                            emphasis={selected ? "soft" : "soft"}
                            aria-pressed={selected}
                            onClick={() => setFormat(entry.id)}
                            style={{
                                display: "block",
                                textAlign: "left",
                                height: "auto",
                                padding: "10px 12px",
                                whiteSpace: "normal",
                            }}
                        >
                            <div style={{ fontWeight: 600 }}>{tf(entry.titleKey)}</div>
                            <div style={{ opacity: 0.8, marginTop: 4 }}>{tf(entry.descKey)}</div>
                        </AppButton>
                    );
                })}
                {error !== null ? (
                    <div role="alert" style={{ color: "var(--qt-danger, #d33)" }}>
                        {error}
                    </div>
                ) : null}
            </Flex>
        </AppDialog>
    );
}
