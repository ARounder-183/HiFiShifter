/*
 * 「视图」菜单里的「窗口 / 布局」二级菜单，以及由它们触发的对话框。
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
 * 【菜单与对话框为什么分两处渲染】Radix 的 `DropdownMenu.Content` 在菜单关闭
 * 时会卸载其子树，而"保存预设"需要一个在菜单关闭后依然存活的输入框。二级菜单
 * （`DockLayoutMenus`）住在视图菜单的内容里，对话框（`DockLayoutDialogs`）由
 * `MenuBar` 常驻渲染；两者用模块级的 `dockDialogKind` 通信 —— 菜单项置位，
 * 宿主读取。打开中的对话框不随菜单卸载而消失。
 *
 * 所有动作都走 `dockApi`，与快捷键、工具栏按钮共用同一份实现 —— "菜单能做的
 * 快捷键做不了"这类分叉从结构上不会出现。
 */

import { useCallback, useMemo, useState, useSyncExternalStore } from "react";
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
    reclaimDetachedForm,
} from "../../features/dock/dockApi";
import { getPanel } from "../../features/dock/panelRegistry";

/* ── 二级菜单 → 对话框宿主的通信 ─────────────────────────────────
 * 二级菜单在菜单关闭时卸载，持有不了对话框的开关状态；把三份状态提升到
 * `MenuBar` 又会让停靠 UI 的细节漏进菜单栏。这里用模块级单值桥接：菜单项
 * 置位、常驻宿主订阅。快照是原始值，`useSyncExternalStore` 不会抖。 */
type DockDialogKind = "settings" | "namePrompt" | "resetConfirm" | null;

let dockDialogKind: DockDialogKind = null;
const dockDialogListeners = new Set<() => void>();

function setDockDialogKind(kind: DockDialogKind) {
    if (dockDialogKind === kind) return;
    dockDialogKind = kind;
    dockDialogListeners.forEach((listener) => listener());
}

function useDockDialogKind(): DockDialogKind {
    return useSyncExternalStore(
        (onStoreChange) => {
            dockDialogListeners.add(onStoreChange);
            return () => {
                dockDialogListeners.delete(onStoreChange);
            };
        },
        () => dockDialogKind,
    );
}

/** 导入布局的隐藏 `<input>` 由常驻的对话框宿主持有，菜单项经此引用触发点击。
 *  输入框若随菜单内容一起卸载，文件选择期间的 change 事件就没人接了。 */
const importInputRef: { current: HTMLInputElement | null } = { current: null };

export interface DockLayoutMenusProps {
    /** 菜单勾选前缀（与仓库既有 View 菜单同一约定）。 */
    withCheck: (active: boolean, label: string) => string;
}

/**
 * 「视图」菜单内容里的两个停靠二级菜单：`窗口`（各窗体显隐）与 `布局`
 * （预设、导入导出等布局级操作）。放在 `DropdownMenu.Content` 内部使用。
 */
export function DockLayoutMenus({ withCheck }: DockLayoutMenusProps) {
    return (
        <>
            <DockWindowsMenu withCheck={withCheck} />
            <DockLayoutSubmenu withCheck={withCheck} />
        </>
    );
}

/** 「窗口」：各停靠窗体的显示开关（原「布局」菜单的「显示窗体」，随其并入视图）。 */
function DockWindowsMenu({ withCheck }: DockLayoutMenusProps) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;

    const layout = useAppSelector((state) => state.dock.layout);
    const entries = useMemo(() => listPanelEntriesFromLayout(layout), [layout]);

    return (
        <DropdownMenu.Sub>
            <DropdownMenu.SubTrigger>{tAny("menu_windows")}</DropdownMenu.SubTrigger>
            <DropdownMenu.SubContent>
                {entries.map((entry) => (
                    <DropdownMenu.Item
                        key={entry.panelId}
                        onSelect={() => togglePanelVisible(dispatch, store.getState, entry.panelId)}
                    >
                        {withCheck(entry.visible, tAny(entry.titleKey))}
                    </DropdownMenu.Item>
                ))}
            </DropdownMenu.SubContent>
        </DropdownMenu.Sub>
    );
}

/** 「布局」：预设、导入导出、重置等布局级操作（原「布局」选项卡的其余项）。 */
function DockLayoutSubmenu({ withCheck }: DockLayoutMenusProps) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;

    const layout = useAppSelector((state) => state.dock.layout);
    const confirmReset = useAppSelector((state) => state.dock.settings.confirmResetLayout);

    /** 当前在独立窗口中的窗体（用于"收回主窗口"入口）。 */
    const osWindowForms = useMemo(
        () =>
            layout.order
                .map((formId) => layout.forms[formId])
                .filter((form): form is NonNullable<typeof form> => form?.floatMode === "osWindow"),
        [layout],
    );
    const presetNames = useMemo(() => listPresetNamesFromLayout(layout), [layout]);

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

    return (
        <DropdownMenu.Sub>
            <DropdownMenu.SubTrigger>{tAny("menu_layout")}</DropdownMenu.SubTrigger>
            <DropdownMenu.SubContent>
                {/* 独立窗口中的窗体：给出"收回主窗口"的入口。
                    没有它的话，一旦卫星窗口的关闭回收没能执行（创建/登记竞态），窗体就
                    既不在主窗口、也不在任何窗口里，用户**无法从界面恢复它** —— 只能重启
                    应用（下次启动会重新打开那个窗口）。这是"面板凭空消失"的唯一出路。 */}
                {osWindowForms.length > 0 ? (
                    <DropdownMenu.Sub>
                        <DropdownMenu.SubTrigger>
                            {tAny("layout_os_windows")}
                        </DropdownMenu.SubTrigger>
                        <DropdownMenu.SubContent>
                            {osWindowForms.map((form) => (
                                <DropdownMenu.Item
                                    key={form.id}
                                    onSelect={() =>
                                        void reclaimDetachedForm(dispatch, store.getState, form.id)
                                    }
                                >
                                    {form.title ??
                                        tAny(getPanel(form.panelId)?.titleKey ?? form.panelId)}
                                </DropdownMenu.Item>
                            ))}
                        </DropdownMenu.SubContent>
                    </DropdownMenu.Sub>
                ) : null}

                <DropdownMenu.Sub>
                    <DropdownMenu.SubTrigger>{tAny("layout_presets")}</DropdownMenu.SubTrigger>
                    <DropdownMenu.SubContent>
                        {presetNames.length === 0 ? (
                            <DropdownMenu.Item disabled>
                                {tAny("layout_no_presets")}
                            </DropdownMenu.Item>
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

                <DropdownMenu.Item onSelect={() => setDockDialogKind("namePrompt")}>
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
                <DropdownMenu.Item onSelect={() => importInputRef.current?.click()}>
                    {tAny("layout_import")}
                </DropdownMenu.Item>

                <DropdownMenu.Separator />

                <DropdownMenu.Item
                    onSelect={() => {
                        // `confirmResetLayout` 关闭时直接重置，不弹窗。
                        if (confirmReset) setDockDialogKind("resetConfirm");
                        else resetLayout(dispatch);
                    }}
                >
                    {tAny("layout_reset")}
                </DropdownMenu.Item>
                <DropdownMenu.Item onSelect={() => setDockDialogKind("settings")}>
                    {tAny("layout_settings")}
                </DropdownMenu.Item>
            </DropdownMenu.SubContent>
        </DropdownMenu.Sub>
    );
}

/**
 * 由「窗口 / 布局」二级菜单触发的三个对话框 + 隐藏的导入输入框。
 *
 * 必须由 `MenuBar` 常驻渲染（不在任何 `DropdownMenu.Content` 子树内）：
 * Radix 的菜单内容在菜单关闭时卸载，打开中的对话框不能住在那里。
 */
export function DockLayoutDialogs() {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const dispatch = useAppDispatch();

    const dialogKind = useDockDialogKind();
    const [presetDraft, setPresetDraft] = useState("");

    // 关闭即清空命名草稿：下一次打开总是从空名开始（Escape / 取消 / 确认都经过这里）。
    const closeDialog = useCallback(() => {
        setPresetDraft("");
        setDockDialogKind(null);
    }, []);

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
            <DockLayoutSettingsDialog
                open={dialogKind === "settings"}
                onOpenChange={(open) => setDockDialogKind(open ? "settings" : null)}
            />

            {/* 重置确认：`confirmResetLayout` 关闭时菜单项直接重置，根本不会走到这里。 */}
            <Dialog.Root
                open={dialogKind === "resetConfirm"}
                onOpenChange={(open) => {
                    if (!open) closeDialog();
                }}
            >
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
                                closeDialog();
                            }}
                        >
                            {tAny("layout_reset")}
                        </Button>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>

            <Dialog.Root
                open={dialogKind === "namePrompt"}
                onOpenChange={(open) => {
                    if (!open) closeDialog();
                }}
            >
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
                                closeDialog();
                            }}
                        />
                    </Flex>
                    <Flex justify="end" gap="2" mt="4">
                        <Dialog.Close>
                            <Button variant="soft" color="gray">
                                {tAny("cancel")}
                            </Button>
                        </Dialog.Close>
                        <Button
                            onClick={() => {
                                if (presetDraft.trim()) savePreset(dispatch, presetDraft.trim());
                                closeDialog();
                            }}
                        >
                            {tAny("ok")}
                        </Button>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>

            {/*
              隐藏的文件输入：导入布局走浏览器原生文件选择，不引入 Tauri 对话框
              命令 —— 导入的是纯文本 JSON，没有理由为它增加一条 IPC 路径。
              常驻渲染（不随菜单关闭卸载），保证选择完成后的 change 事件有人接。
            */}
            <input
                ref={(element) => {
                    importInputRef.current = element;
                }}
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
