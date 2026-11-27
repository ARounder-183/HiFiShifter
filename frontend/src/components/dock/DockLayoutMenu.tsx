/*
 * 「窗口 / 布局」二级菜单所触发的三个对话框 + 隐藏的导入输入框。
 *
 * 【为什么与菜单条目分成两个文件】本文件只导出组件（Fast Refresh 的约定），
 * 条目数据在 `useDockLayoutMenuItems` 里 —— 两者用 `dockLayoutDialogBridge`
 * 的模块级单值通信：菜单项置位、常驻宿主订阅。
 *
 * 【为什么必须常驻】菜单关闭即卸载，打开中的对话框不能住在菜单子树里。
 */
import { useCallback, useState } from "react";
import { TextField } from "@radix-ui/themes";

import { useAppDispatch } from "../../app/hooks";
import { useI18n } from "../../i18n/I18nProvider";
import { DockLayoutSettingsDialog } from "./DockLayoutSettingsDialog";
import {
    importLayoutJson,
    resetLayout,
    deletePreset,
    savePreset,
} from "../../features/dock/dockApi";
import { AppFileInput } from "../../ui/FileInput";
import { AppDialog } from "../../ui/Dialog";
import { AppConfirmDialog, AppNoticeDialog } from "../../ui";
import {
    dockImportInputRef,
    getDockNoticeMessage,
    getDockPresetToDelete,
    setDockDialogKind,
    setDockNotice,
    setDockNoticeMessage,
    setDockPresetToDelete,
    useDockDialogKind,
} from "./dockLayoutDialogBridge";

/**
 * 由「窗口 / 布局」二级菜单触发的三个对话框 + 隐藏的导入输入框。
 *
 * 必须由 `MenuBar` 常驻渲染（不在任何菜单表面之内）：菜单关闭即卸载，
 * 打开中的对话框不能住在那里。
 */
export function DockLayoutDialogs() {
    const { t, tf } = useI18n();
    const dispatch = useAppDispatch();

    const dialogKind = useDockDialogKind();
    const [presetDraft, setPresetDraft] = useState("");

    // 关闭即清空命名草稿与通知正文：下一次打开总是从空名开始（Escape / 取消 / 确认都经过这里）。
    const closeDialog = useCallback(() => {
        setPresetDraft("");
        setDockNoticeMessage(null);
        setDockPresetToDelete(null);
        setDockDialogKind(null);
    }, []);

    const onImportFile = useCallback(
        async (file: File | null) => {
            if (!file) return;
            const text = await file.text();
            if (!importLayoutJson(dispatch, text)) setDockNotice(tf("layout_import_failed"));
        },
        [dispatch, tf],
    );

    return (
        <>
            <DockLayoutSettingsDialog
                open={dialogKind === "settings"}
                onOpenChange={(open) => setDockDialogKind(open ? "settings" : null)}
            />

            {/* 重置确认：`confirmResetLayout` 关闭时菜单项直接重置，根本不会走到这里。 */}
            <AppConfirmDialog
                open={dialogKind === "resetConfirm"}
                onOpenChange={(open) => {
                    if (!open) closeDialog();
                }}
                title={tf("layout_reset_confirm_title")}
                message={tf("layout_reset_confirm_body")}
                confirmLabel={tf("layout_reset")}
                cancelLabel={t("cancel")}
                intent="primary"
                onConfirm={() => {
                    resetLayout(dispatch);
                    closeDialog();
                }}
            />

            {/* 删除预设确认：预设名经模块级桥接传入（菜单项已随菜单卸载）。 */}
            <AppConfirmDialog
                open={dialogKind === "deletePresetConfirm"}
                onOpenChange={(open) => {
                    if (!open) closeDialog();
                }}
                title={tf("layout_delete_preset")}
                message={tf("layout_delete_preset_confirm")}
                confirmLabel={tf("layout_delete_preset")}
                cancelLabel={t("cancel")}
                intent="danger"
                onConfirm={() => {
                    const presetToDelete = getDockPresetToDelete();
                    if (presetToDelete) deletePreset(dispatch, presetToDelete);
                    closeDialog();
                }}
            />

            <AppDialog
                open={dialogKind === "namePrompt"}
                onOpenChange={(open) => {
                    if (!open) closeDialog();
                }}
                title={tf("layout_preset_name_prompt")}
                size="sm"
                actions={[
                    { id: "cancel", label: tf("cancel"), onClick: () => closeDialog() },
                    {
                        id: "ok",
                        label: tf("ok"),
                        intent: "primary",
                        onClick: () => {
                            if (presetDraft.trim()) savePreset(dispatch, presetDraft.trim());
                            closeDialog();
                        },
                    },
                ]}
            >
                <TextField.Root
                    autoFocus
                    size="2"
                    style={{ width: "100%" }}
                    value={presetDraft}
                    onChange={(event) => setPresetDraft(event.target.value)}
                />
            </AppDialog>

            {/* 布局导入/导出的错误通知：由常驻宿主渲染（导出失败发生在菜单已关闭
                之后，二级菜单此时已卸载，对话框不能住在菜单子树里）。 */}
            <AppNoticeDialog
                open={dialogKind === "notice"}
                onOpenChange={(open) => {
                    if (!open) closeDialog();
                }}
                title={tf("status_error_prefix")}
                message={getDockNoticeMessage() ?? ""}
                closeLabel={tf("ok")}
            />

            {/*
              隐藏的文件输入：导入走浏览器原生文件选择（WebView 内可用，不必动用
              IPC）。导出不能照搬浏览器下载 —— WebView 默认拦截 `<a download>`，
              所以走原生保存对话框 + 后端写文件（见 `services/api/jsonExport.ts`）。
              常驻渲染（不随菜单关闭卸载），保证选择完成后的 change 事件有人接。
            */}
            <AppFileInput
                inputRef={dockImportInputRef}
                accept="application/json,.json"
                onFiles={(files) => void onImportFile(files[0] ?? null)}
            />
        </>
    );
}
