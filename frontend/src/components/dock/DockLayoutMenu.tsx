/*
 * 「布局」菜单的内容。
 *
 * 独立成组件而不是把 JSX 塞进 `MenuBar`：菜单项需要读一批停靠状态（面板清单、
 * 预设、是否最大化），而 `MenuBar` 已经用 `shallowEqual` 窄订阅来避免被 33Hz
 * 的播放轮询拖着重渲染。把这些订阅收在独立组件里，`MenuBar` 的订阅面就不会
 * 因为新增功能而扩大。
 *
 * 【订阅形状】只订阅 `state.dock.layout` 这一个引用，其余（面板清单、预设名）
 * 在组件内派生。若把它们写进选择器，每次都会返回新数组，`useSyncExternalStore`
 * 会因快照不稳定而反复重渲染。布局对象的引用只在真正变化时才变，因此"订阅布局
 * + 就地派生"是这里唯一正确的形状。
 *
 * 所有动作都走 `dockApi`，与快捷键、工具栏按钮共用同一份实现 —— "菜单能做的
 * 快捷键做不了"这类分叉从结构上不会出现。
 */

import { useCallback, useMemo, useRef, useState } from "react";
import { Button, Dialog, DropdownMenu, Flex, TextField } from "@radix-ui/themes";

import { store } from "../../app/store";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { useI18n } from "../../i18n/I18nProvider";
import { DockLayoutSettingsDialog } from "./DockLayoutSettingsDialog";
import {
    applyPreset,
    deletePreset,
    exportLayoutJsonFromLayout,
    importLayoutJson,
    listPanelEntriesFromLayout,
    listPresetNamesFromLayout,
    resetLayout,
    savePreset,
    togglePanelVisible,
} from "../../features/dock/dockApi";

export interface DockLayoutMenuProps {
    /** 菜单勾选前缀（与仓库既有 View 菜单同一约定）。 */
    withCheck: (active: boolean, label: string) => string;
}

/**
 * 「布局」菜单的完整宿主：触发按钮 + 菜单内容 + 由菜单触发的两个对话框。
 *
 * 【为什么对话框不放在菜单内容里】Radix 的 `DropdownMenu.Content` 在菜单关闭
 * 时会卸载其子树，而"保存预设"需要一个在菜单关闭后依然存活的输入框。把对话框
 * 与菜单并列渲染，既避开这个生命周期陷阱，也让 `MenuBar` 只需新增一行。
 */
export function DockLayoutMenu({ withCheck }: DockLayoutMenuProps) {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const dispatch = useAppDispatch();

    const [settingsOpen, setSettingsOpen] = useState(false);
    const [namePromptOpen, setNamePromptOpen] = useState(false);
    const [presetDraft, setPresetDraft] = useState("");
    const [confirmResetOpen, setConfirmResetOpen] = useState(false);
    const confirmReset = useAppSelector((s) => s.dock.settings.confirmResetLayout);

    return (
        <>
            <DropdownMenu.Root>
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{tAny("menu_layout")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    <DockLayoutMenuItems
                        withCheck={withCheck}
                        onOpenSettings={() => setSettingsOpen(true)}
                        onPromptName={() => {
                            setPresetDraft("");
                            setNamePromptOpen(true);
                        }}
                        onReset={() => {
                            if (confirmReset) setConfirmResetOpen(true);
                            else resetLayout(dispatch);
                        }}
                    />
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            <DockLayoutSettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />

            {/* 重置确认：`confirmResetLayout` 关闭时直接重置，不弹窗。 */}
            <Dialog.Root open={confirmResetOpen} onOpenChange={setConfirmResetOpen}>
                <Dialog.Content maxWidth="400px" onKeyDown={(event) => event.stopPropagation()}>
                    <Dialog.Title>{tAny("layout_reset_confirm_title")}</Dialog.Title>
                    <Dialog.Description size="2" mt="2">
                        {tAny("layout_reset_confirm_body")}
                    </Dialog.Description>
                    <Flex justify="end" gap="2" mt="4">
                        <Dialog.Close>
                            <Button variant="soft" color="gray">
                                {t("cancel")}
                            </Button>
                        </Dialog.Close>
                        <Button
                            onClick={() => {
                                resetLayout(dispatch);
                                setConfirmResetOpen(false);
                            }}
                        >
                            {tAny("layout_reset")}
                        </Button>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>

            <Dialog.Root open={namePromptOpen} onOpenChange={setNamePromptOpen}>
                <Dialog.Content maxWidth="360px" onKeyDown={(event) => event.stopPropagation()}>
                    <Dialog.Title>{tAny("layout_preset_name_prompt")}</Dialog.Title>
                    <Flex mt="3">
                        <TextField.Root
                            autoFocus
                            size="2"
                            style={{ width: "100%" }}
                            value={presetDraft}
                            onChange={(event) => setPresetDraft(event.target.value)}
                            onKeyDown={(event) => {
                                if (event.key !== "Enter") return;
                                if (presetDraft.trim()) savePreset(dispatch, presetDraft.trim());
                                setNamePromptOpen(false);
                            }}
                        />
                    </Flex>
                    <Flex justify="end" gap="2" mt="4">
                        <Dialog.Close>
                            <Button variant="soft" color="gray">
                                {t("cancel")}
                            </Button>
                        </Dialog.Close>
                        <Button
                            onClick={() => {
                                if (presetDraft.trim()) savePreset(dispatch, presetDraft.trim());
                                setNamePromptOpen(false);
                            }}
                        >
                            {"OK"}
                        </Button>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>
        </>
    );
}

interface DockLayoutMenuItemsProps {
    onOpenSettings: () => void;
    onPromptName: () => void;
    onReset: () => void;
    withCheck: (active: boolean, label: string) => string;
}

function DockLayoutMenuItems({
    onOpenSettings,
    onPromptName,
    onReset,
    withCheck,
}: DockLayoutMenuItemsProps) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;

    const layout = useAppSelector((state) => state.dock.layout);

    const entries = useMemo(() => listPanelEntriesFromLayout(layout), [layout]);
    const presetNames = useMemo(() => listPresetNamesFromLayout(layout), [layout]);

    const fileInputRef = useRef<HTMLInputElement | null>(null);

    const onExport = useCallback(() => {
        const json = exportLayoutJsonFromLayout(layout);
        const blob = new Blob([json], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = "hifishifter-layout.json";
        anchor.click();
        URL.revokeObjectURL(url);
    }, [layout]);

    const onImportFile = useCallback(
        async (file: File | null) => {
            if (!file) return;
            const text = await file.text();
            if (!importLayoutJson(dispatch, text)) window.alert(tAny("layout_import_failed"));
        },
        [dispatch, tAny],
    );

    return (
        <>
            <DropdownMenu.Sub>
                <DropdownMenu.SubTrigger>{tAny("layout_show_panels")}</DropdownMenu.SubTrigger>
                <DropdownMenu.SubContent>
                    {entries.map((entry) => (
                        <DropdownMenu.Item
                            key={entry.panelId}
                            onSelect={() =>
                                togglePanelVisible(dispatch, store.getState, entry.panelId)
                            }
                        >
                            {withCheck(entry.visible, tAny(entry.titleKey))}
                        </DropdownMenu.Item>
                    ))}
                </DropdownMenu.SubContent>
            </DropdownMenu.Sub>

            <DropdownMenu.Separator />

            {/* `浮动` / `最大化当前窗体` 两项已移除：它们是"对当前窗体"的操作，而
                这个菜单是**布局级**的（显示哪些面板、预设、导入导出）。同一能力仍在
                标签右键菜单（`DockTabMenu`）、标签上的浮动按钮、以及
                `Ctrl+Shift+F` / `Ctrl+Shift+M` 快捷键上，入口没有减少。 */}

            <DropdownMenu.Separator />

            <DropdownMenu.Sub>
                <DropdownMenu.SubTrigger>{tAny("layout_presets")}</DropdownMenu.SubTrigger>
                <DropdownMenu.SubContent>
                    {presetNames.length === 0 ? (
                        <DropdownMenu.Item disabled>{tAny("layout_no_presets")}</DropdownMenu.Item>
                    ) : (
                        presetNames.map((name) => (
                            <DropdownMenu.Item
                                key={name}
                                onSelect={() => applyPreset(dispatch, name)}
                            >
                                {withCheck(layout.activePreset === name, name)}
                            </DropdownMenu.Item>
                        ))
                    )}
                </DropdownMenu.SubContent>
            </DropdownMenu.Sub>

            <DropdownMenu.Item onSelect={onPromptName}>
                {tAny("layout_save_preset")}
            </DropdownMenu.Item>

            {presetNames.length > 0 ? (
                <DropdownMenu.Sub>
                    <DropdownMenu.SubTrigger>
                        {tAny("layout_delete_preset")}
                    </DropdownMenu.SubTrigger>
                    <DropdownMenu.SubContent>
                        {presetNames.map((name) => (
                            <DropdownMenu.Item
                                key={name}
                                color="red"
                                onSelect={() => deletePreset(dispatch, name)}
                            >
                                {name}
                            </DropdownMenu.Item>
                        ))}
                    </DropdownMenu.SubContent>
                </DropdownMenu.Sub>
            ) : null}

            <DropdownMenu.Separator />

            <DropdownMenu.Item onSelect={onExport}>{tAny("layout_export")}</DropdownMenu.Item>
            <DropdownMenu.Item onSelect={() => fileInputRef.current?.click()}>
                {tAny("layout_import")}
            </DropdownMenu.Item>

            <DropdownMenu.Separator />

            <DropdownMenu.Item onSelect={onReset}>{tAny("layout_reset")}</DropdownMenu.Item>
            <DropdownMenu.Item onSelect={onOpenSettings}>
                {tAny("layout_settings")}
            </DropdownMenu.Item>

            {/*
              隐藏的文件输入：导入布局走浏览器原生文件选择，不引入 Tauri 对话框
              命令 —— 导入的是纯文本 JSON，没有理由为它增加一条 IPC 路径。
            */}
            <input
                ref={fileInputRef}
                type="file"
                accept="application/json,.json"
                style={{ display: "none" }}
                onChange={(event) => {
                    void onImportFile(event.target.files?.[0] ?? null);
                    event.target.value = "";
                }}
            />
        </>
    );
}
