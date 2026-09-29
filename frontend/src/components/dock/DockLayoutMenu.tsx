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
import { DropdownMenu, TextField } from "@radix-ui/themes";

import { store } from "../../app/store";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { useI18n } from "../../i18n/I18nProvider";
import { DockLayoutSettingsDialog } from "./DockLayoutSettingsDialog";
import { exportLayoutJson } from "../../services/api/dockLayout";
import {
    addEmptyPanel,
    applyPreset,
    deletePreset,
    dissolvePanelCommand,
    exportLayoutJsonFromLayout,
    importLayoutJson,
    listPanelEntriesFromLayout,
    listPresetNamesFromLayout,
    resetLayout,
    savePreset,
    togglePanelVisible,
    reclaimDetachedForm,
} from "../../features/dock/dockApi";
import { focusForm } from "../../features/dock/dockSlice";
import { displayTitleOf, panelTitleOf } from "../../features/dock/dockPanel";
import { isPanelForm } from "../../features/dock/dockTree";
import { AppFileInput } from "../../ui/FileInput";
import { AppDialog } from "../../ui/Dialog";
import { AppConfirmDialog, AppNoticeDialog } from "../../ui";

/* ── 二级菜单 → 对话框宿主的通信 ─────────────────────────────────
 * 二级菜单在菜单关闭时卸载，持有不了对话框的开关状态；把三份状态提升到
 * `MenuBar` 又会让停靠 UI 的细节漏进菜单栏。这里用模块级单值桥接：菜单项
 * 置位、常驻宿主订阅。快照是原始值，`useSyncExternalStore` 不会抖。 */
type DockDialogKind =
    | "settings"
    | "namePrompt"
    | "resetConfirm"
    | "deletePresetConfirm"
    | "notice"
    | null;

let dockDialogKind: DockDialogKind = null;
const dockDialogListeners = new Set<() => void>();

/**
 * 待删除的预设名。与 `dockNoticeMessage` 同理：快照仍是原始值（kind），
 * 预设名在 kind 变化触发的那次重渲染里读取即可，`useSyncExternalStore` 不会抖。
 */
let dockPresetToDelete: string | null = null;

/**
 * 纯通知对话框的正文。与 `dockDialogKind` 分开存放：快照仍是原始字符串
 * （`useSyncExternalStore` 不会因对象不稳定而反复重渲染），正文在 kind 变化
 * 触发的那次重渲染里直接读取即可。
 */
let dockNoticeMessage: string | null = null;

function setDockDialogKind(kind: DockDialogKind) {
    if (dockDialogKind === kind) return;
    dockDialogKind = kind;
    dockDialogListeners.forEach((listener) => listener());
}

/** 打开错误通知（取代此前的 `window.alert`）。 */
function setDockNotice(message: string) {
    dockNoticeMessage = message;
    setDockDialogKind("notice");
}

/** 请求删除预设：菜单项只置位（记下待删名字），常驻宿主弹确认框。 */
function requestDeletePreset(name: string) {
    dockPresetToDelete = name;
    setDockDialogKind("deletePresetConfirm");
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
    const { tf } = useI18n();

    const layout = useAppSelector((state) => state.dock.layout);
    const entries = useMemo(() => listPanelEntriesFromLayout(layout), [layout]);

    return (
        <DropdownMenu.Sub>
            <DropdownMenu.SubTrigger>{tf("menu_windows")}</DropdownMenu.SubTrigger>
            <DropdownMenu.SubContent>
                {entries.map((entry) => (
                    <DropdownMenu.Item
                        key={entry.panelId}
                        onSelect={() => togglePanelVisible(dispatch, store.getState, entry.panelId)}
                    >
                        {withCheck(entry.visible, tf(entry.titleKey))}
                    </DropdownMenu.Item>
                ))}
            </DropdownMenu.SubContent>
        </DropdownMenu.Sub>
    );
}

/** 「布局」：预设、导入导出、重置等布局级操作（原「布局」选项卡的其余项）。 */
function DockLayoutSubmenu({ withCheck }: DockLayoutMenusProps) {
    const dispatch = useAppDispatch();
    const { tf } = useI18n();
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
    /** 现存的全部面板窗体（用于"面板"子菜单；顺序即创建顺序）。 */
    const panelForms = useMemo(
        () =>
            layout.order
                .map((formId) => layout.forms[formId])
                .filter((form): form is NonNullable<typeof form> => isPanelForm(form)),
        [layout],
    );
    const presetNames = useMemo(() => listPresetNamesFromLayout(layout), [layout]);

    // 导出走后端命令（原生保存对话框 + 写文件）：Tauri 的 WebView 默认拦截
    // 页面发起的下载，Blob + `<a download>` 的浏览器方案在壳内静默失败 ——
    // 用户点「导出布局」什么都不会发生（见 `services/api/dockLayout.ts`）。
    const onExport = useCallback(async () => {
        const json = exportLayoutJsonFromLayout(layout);
        try {
            const result = await exportLayoutJson(json);
            if (!result.ok && !result.canceled) {
                setDockNotice(result.error || tf("layout_export_failed"));
            }
        } catch {
            setDockNotice(tf("layout_export_failed"));
        }
    }, [layout, tf]);

    return (
        <DropdownMenu.Sub>
            <DropdownMenu.SubTrigger>{tf("menu_layout")}</DropdownMenu.SubTrigger>
            <DropdownMenu.SubContent>
                {/* 独立窗口中的窗体：给出"收回主窗口"的入口。
                    没有它的话，一旦卫星窗口的关闭回收没能执行（创建/登记竞态），窗体就
                    既不在主窗口、也不在任何窗口里，用户**无法从界面恢复它** —— 只能重启
                    应用（下次启动会重新打开那个窗口）。这是"面板凭空消失"的唯一出路。 */}
                {osWindowForms.length > 0 ? (
                    <DropdownMenu.Sub>
                        <DropdownMenu.SubTrigger>{tf("layout_os_windows")}</DropdownMenu.SubTrigger>
                        <DropdownMenu.SubContent>
                            {osWindowForms.map((form) => (
                                <DropdownMenu.Item
                                    key={form.id}
                                    onSelect={() =>
                                        void reclaimDetachedForm(dispatch, store.getState, form.id)
                                    }
                                >
                                    {displayTitleOf(layout, form.id, tf)}
                                </DropdownMenu.Item>
                            ))}
                        </DropdownMenu.SubContent>
                    </DropdownMenu.Sub>
                ) : null}

                {/* 面板：现存的容器窗体。点按聚焦；标题按内容派生（重命名过则用
                    用户文本）。仿照 os_windows 的模式：没有面板时不渲染空壳。 */}
                {panelForms.length > 0 ? (
                    <DropdownMenu.Sub>
                        <DropdownMenu.SubTrigger>{tf("layout_panels")}</DropdownMenu.SubTrigger>
                        <DropdownMenu.SubContent>
                            {panelForms.map((form) => (
                                <DropdownMenu.Item
                                    key={form.id}
                                    onSelect={() => dispatch(focusForm(form.id))}
                                >
                                    {panelTitleOf(layout, form.id, tf)}
                                </DropdownMenu.Item>
                            ))}
                        </DropdownMenu.SubContent>
                    </DropdownMenu.Sub>
                ) : null}
                <DropdownMenu.Item onSelect={() => addEmptyPanel(dispatch)}>
                    {tf("layout_new_panel")}
                </DropdownMenu.Item>
                <DropdownMenu.Item
                    color="red"
                    onSelect={() => dissolvePanelCommand(dispatch, store.getState)}
                >
                    {tf("layout_dissolve_panel")}
                </DropdownMenu.Item>

                <DropdownMenu.Sub>
                    <DropdownMenu.SubTrigger>{tf("layout_presets")}</DropdownMenu.SubTrigger>
                    <DropdownMenu.SubContent>
                        {presetNames.length === 0 ? (
                            <DropdownMenu.Item disabled>
                                {tf("layout_no_presets")}
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
                    {tf("layout_save_preset")}
                </DropdownMenu.Item>

                {presetNames.length > 0 ? (
                    <DropdownMenu.Sub>
                        <DropdownMenu.SubTrigger>
                            {tf("layout_delete_preset")}
                        </DropdownMenu.SubTrigger>
                        <DropdownMenu.SubContent>
                            {presetNames.map((name) => (
                                <DropdownMenu.Item
                                    key={name}
                                    color="red"
                                    // 不立即删除：菜单关闭即卸载，确认框住在常驻宿主里。
                                    onSelect={() => requestDeletePreset(name)}
                                >
                                    {name}
                                </DropdownMenu.Item>
                            ))}
                        </DropdownMenu.SubContent>
                    </DropdownMenu.Sub>
                ) : null}

                <DropdownMenu.Separator />

                <DropdownMenu.Item onSelect={() => void onExport()}>
                    {tf("layout_export")}
                </DropdownMenu.Item>
                <DropdownMenu.Item onSelect={() => importInputRef.current?.click()}>
                    {tf("layout_import")}
                </DropdownMenu.Item>

                <DropdownMenu.Separator />

                <DropdownMenu.Item
                    onSelect={() => {
                        // `confirmResetLayout` 关闭时直接重置，不弹窗。
                        if (confirmReset) setDockDialogKind("resetConfirm");
                        else resetLayout(dispatch);
                    }}
                >
                    {tf("layout_reset")}
                </DropdownMenu.Item>
                <DropdownMenu.Item onSelect={() => setDockDialogKind("settings")}>
                    {tf("layout_settings")}
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
    const { t, tf } = useI18n();
    const dispatch = useAppDispatch();

    const dialogKind = useDockDialogKind();
    const [presetDraft, setPresetDraft] = useState("");

    // 关闭即清空命名草稿与通知正文：下一次打开总是从空名开始（Escape / 取消 / 确认都经过这里）。
    const closeDialog = useCallback(() => {
        setPresetDraft("");
        dockNoticeMessage = null;
        dockPresetToDelete = null;
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
                    if (dockPresetToDelete) deletePreset(dispatch, dockPresetToDelete);
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
                message={dockNoticeMessage ?? ""}
                closeLabel={tf("ok")}
            />

            {/*
              隐藏的文件输入：导入走浏览器原生文件选择（WebView 内可用，不必动用
              IPC）。导出不能照搬浏览器下载 —— WebView 默认拦截 `<a download>`，
              所以走原生保存对话框 + 后端写文件（见 `services/api/dockLayout.ts`）。
              常驻渲染（不随菜单关闭卸载），保证选择完成后的 change 事件有人接。
            */}
            <AppFileInput
                inputRef={importInputRef}
                accept="application/json,.json"
                onFiles={(files) => void onImportFile(files[0] ?? null)}
            />
        </>
    );
}
