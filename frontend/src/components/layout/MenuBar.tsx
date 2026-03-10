import React, { useState } from "react";
import { Flex, DropdownMenu } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import {
    importAudioFromDialog,
    exportAudio,
    exportSeparated,
    openReaperFromDialog,
    openVocalShifterFromDialog,
    pickOutputPath,
    addTrackRemote,
    removeTrackRemote,
    refreshRuntime,
    clearWaveformCacheRemote,
    undoRemote,
    redoRemote,
    pasteVocalShifterClipboard,
    pasteReaperClipboard,
    saveProjectRemote,
    saveProjectAsRemote,
} from "../../features/session/sessionSlice";
import { fileBrowserApi } from "../../services/api/fileBrowser";
import { useAppTheme } from "../../theme/AppThemeProvider";
import { GlobeIcon } from "@radix-ui/react-icons";
import {
    selectMergedKeybindings,
    formatKeybinding,
} from "../../features/keybindings/keybindingsSlice";
import type { ActionId } from "../../features/keybindings/types";
import { KeybindingsDialog } from "./KeybindingsDialog";
import { ResamplerManagerDialog } from "./ResamplerManagerDialog";

interface MenuBarProps {
    onNewProject: () => void;
    onOpenProject: () => void;
    onOpenRecentProject: (projectPath: string) => void;
    onExit: () => void;
}

export const MenuBar: React.FC<MenuBarProps> = ({
    onNewProject,
    onOpenProject,
    onOpenRecentProject,
    onExit,
}) => {
    const { t, setLocale } = useI18n();
    const dispatch = useAppDispatch();
    const s = useAppSelector((state: RootState) => state.session);
    const theme = useAppTheme();
    const keybindings = useAppSelector(selectMergedKeybindings);
    const [kbDialogOpen, setKbDialogOpen] = useState(false);
    const [rsDialogOpen, setRsDialogOpen] = useState(false);

    /** 获取某个操作的快捷键显示文本 */
    function shortcutLabel(actionId: ActionId): string {
        return formatKeybinding(keybindings[actionId]);
    }

    async function handleExport() {
        const outputPath = s.outputPath?.trim();
        if (!outputPath) {
            const picked = await dispatch(pickOutputPath()).unwrap();
            if (picked.ok && !picked.canceled && picked.path) {
                await dispatch(exportAudio(picked.path));
            }
            return;
        }
        await dispatch(exportAudio(outputPath));
    }

    async function handleExportSeparated() {
        // 弹出文件夹选择对话框
        const result = await fileBrowserApi.pickDirectory();
        if (!result.ok || result.canceled || !result.path) return;
        await dispatch(exportSeparated(result.path));
    }

    return (
        <Flex
            align="center"
            className="h-8 bg-qt-panel border-b border-qt-border px-1 select-none z-50 flex-nowrap gap-1 overflow-x-auto overflow-y-hidden min-w-0 custom-scrollbar"
        >
            {/**
             * Note: @radix-ui/themes DropdownMenu.Trigger does not support asChild.
             * Use Trigger as the actual button element to avoid nesting <button>.
             */}
            {/* File Menu */}
            <DropdownMenu.Root>
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{t("menu_file")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    <DropdownMenu.Item onSelect={onNewProject}>
                        {t("menu_new_project")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("project.new")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={onOpenProject}>
                        {t("menu_open_project")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("project.open")}
                        </div>
                    </DropdownMenu.Item>

                    <DropdownMenu.Sub>
                        <DropdownMenu.SubTrigger>
                            {t("menu_recent_projects")}
                        </DropdownMenu.SubTrigger>
                        <DropdownMenu.SubContent>
                            {s.project.recent.length ? (
                                s.project.recent.slice(0, 12).map((p) => (
                                    <DropdownMenu.Item
                                        key={p}
                                        onSelect={() => onOpenRecentProject(p)}
                                    >
                                        {p}
                                    </DropdownMenu.Item>
                                ))
                            ) : (
                                <DropdownMenu.Item disabled>
                                    {t("menu_recent_empty")}
                                </DropdownMenu.Item>
                            )}
                        </DropdownMenu.SubContent>
                    </DropdownMenu.Sub>

                    <DropdownMenu.Separator />

                    <DropdownMenu.Item
                        onSelect={() => void dispatch(saveProjectRemote())}
                    >
                        {t("menu_save_project")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("project.save")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item
                        onSelect={() => void dispatch(saveProjectAsRemote())}
                    >
                        {t("menu_save_project_as")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("project.saveAs")}
                        </div>
                    </DropdownMenu.Item>

                    <DropdownMenu.Separator />

                    <DropdownMenu.Item
                        onSelect={() => dispatch(importAudioFromDialog())}
                    >
                        {t("menu_import_audio")}{" "}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item
                        onSelect={() => void dispatch(openReaperFromDialog())}
                    >
                        {t("menu_import_reaper")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item
                        onSelect={() => void dispatch(openVocalShifterFromDialog())}
                    >
                        {t("menu_import_vocalshifter")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={handleExport}>
                        {t("menu_export_audio")}{" "}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("project.export")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={handleExportSeparated}>
                        {t("menu_export_separated")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item
                        onSelect={() => dispatch(pickOutputPath())}
                    >
                        {t("menu_pick_output")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item onSelect={onExit} color="red">
                        {t("menu_exit")}
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            {/* Edit Menu */}
            <DropdownMenu.Root>
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{t("menu_edit")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    <DropdownMenu.Item
                        onSelect={() => void dispatch(undoRemote())}
                    >
                        {t("menu_undo")}{" "}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("edit.undo")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item
                        onSelect={() => void dispatch(redoRemote())}
                    >
                        {t("menu_redo")}{" "}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("edit.redo")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item>
                        {t("menu_select_all")}{" "}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            Ctrl+A
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item
                        onSelect={() => void dispatch(pasteReaperClipboard())}
                    >
                        {t("menu_paste_reaper_clipboard")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item
                        onSelect={() =>
                            void dispatch(pasteVocalShifterClipboard())
                        }
                    >
                        {t("menu_paste_vocalshifter_clipboard")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item onSelect={() => setRsDialogOpen(true)}>
                        {(t as (key: string) => string)("menu_manage_resampler")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => setKbDialogOpen(true)}>
                        {(t as (key: string) => string)("menu_keybindings")}
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            {/* Track Menu */}
            <DropdownMenu.Root>
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{t("menu_track")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    <DropdownMenu.Item
                        onSelect={() => dispatch(addTrackRemote({}))}
                    >
                        {t("track_add")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item
                        disabled={!s.selectedTrackId}
                        onSelect={() =>
                            s.selectedTrackId &&
                            dispatch(removeTrackRemote(s.selectedTrackId))
                        }
                    >
                        {t("track_remove_selected")}
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            {/* View Menu */}
            <DropdownMenu.Root>
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{t("menu_view")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    <DropdownMenu.Item
                        onSelect={() => dispatch(refreshRuntime())}
                    >
                        {t("action_refresh")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item
                        onSelect={() =>
                            void dispatch(clearWaveformCacheRemote())
                        }
                    >
                        {t("menu_clear_waveform_cache")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item onSelect={() => theme.toggleMode()}>
                        {t("theme")}:{" "}
                        {theme.mode === "dark"
                            ? t("theme_dark")
                            : t("theme_light")}
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            {/* Help Menu */}
            <DropdownMenu.Root>
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{t("menu_help")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    <DropdownMenu.Item>{t("menu_about")}</DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            <Flex ml="auto" gap="2" align="center" className="shrink-0">
                <DropdownMenu.Root>
                    <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                        <Flex align="center" gap="1">
                            <GlobeIcon width={14} height={14} />
                            <span>{t("language")}</span>
                        </Flex>
                    </DropdownMenu.Trigger>
                    <DropdownMenu.Content>
                        <DropdownMenu.Item onSelect={() => setLocale("en-US")}>
                            {t("lang_en")}
                        </DropdownMenu.Item>
                        <DropdownMenu.Item onSelect={() => setLocale("zh-CN")}>
                            {t("lang_zh")}
                        </DropdownMenu.Item>
                    </DropdownMenu.Content>
                </DropdownMenu.Root>
            </Flex>

            {/* 快捷键设置对话框 */}
            <KeybindingsDialog
                open={kbDialogOpen}
                onOpenChange={setKbDialogOpen}
            />
            {/* Resampler 管理对话框 */}
            <ResamplerManagerDialog
                open={rsDialogOpen}
                onOpenChange={setRsDialogOpen}
            />
        </Flex>
    );
};
