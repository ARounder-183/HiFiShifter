/*
 * 「视图」菜单里的「窗口 / 布局」二级菜单（条目数据）。
 *
 * 独立成模块而不是把条目塞进 `MenuBar`：菜单项需要读一批停靠状态（面板清单、
 * 预设、是否最大化），而 `MenuBar` 已经用 `shallowEqual` 窄订阅来避免被 33Hz
 * 的播放轮询拖着重渲染。把这些订阅收在独立模块里，`MenuBar` 的订阅面就不会
 * 因为新增功能而扩大。
 *
 * 【订阅形状】只订阅 `state.dock.layout` 这一个引用，其余（面板清单、预设名）
 * 在模块内派生。若把它们写进选择器，每次都会返回新数组，`useSyncExternalStore`
 * 会因快照不稳定而反复重渲染。布局对象的引用只在真正变化时才变，因此"订阅布局
 * + 就地派生"是这里唯一正确的形状。
 *
 * 【为什么产出的是条目数据而不是 JSX】菜单表面只有一套（`.hs-menu`），条目只有
 * 一个渲染路径。做成组件时它们只能渲染成 Radix 的 `DropdownMenu.Sub`，于是「视图」
 * 这一张菜单里混着两种条目来源 —— 键盘、滚动、勾选语义都要各写一遍。
 *
 * 【为什么与对话框宿主分成两个文件】对话框是常驻组件，本模块只导出 hook。
 * 混在一个文件里会破坏 React Fast Refresh 的"文件只导出组件"约定
 * （eslint `react-refresh/only-export-components` 会拦下）—— 与
 * `dockLayoutDialogBridge` 当初从 `DockLayoutMenu.tsx` 搬出来是同一条理由。
 *
 * 所有动作都走 `dockApi`，与快捷键、工具栏按钮共用同一份实现 —— "菜单能做的
 * 快捷键做不了"这类分叉从结构上不会出现。
 */
import { useCallback, useMemo } from "react";

import { store } from "../../app/store";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { useI18n } from "../../i18n/I18nProvider";
import { exportLayoutJson } from "../../services/api/jsonExport";
import {
    addEmptyPanel,
    applyPreset,
    dissolvePanelCommand,
    exportLayoutJsonFromLayout,
    listPanelEntriesFromLayout,
    listPresetNamesFromLayout,
    resetLayout,
    togglePanelVisible,
    reclaimDetachedForm,
} from "../../features/dock/dockApi";
import { focusForm } from "../../features/dock/dockSlice";
import { displayTitleOf, panelTitleOf } from "../../features/dock/dockPanel";
import { isPanelForm } from "../../features/dock/dockTree";
import type { AppMenuItemSpec } from "../../ui/Menu";
import {
    dockImportInputRef,
    requestDeletePreset,
    setDockDialogKind,
    setDockNotice,
} from "./dockLayoutDialogBridge";

export interface DockLayoutMenuOptions {
    /** 菜单勾选前缀（与仓库既有 View 菜单同一约定）。 */
    withCheck: (active: boolean, label: string) => string;
}

/**
 * 「视图」菜单内容里的两个停靠二级菜单：`窗口`（各窗体显隐）与 `布局`
 * （预设、导入导出等布局级操作）。
 *
 * 【返回值的形状】两个 `AppMenuItemSpec`，各自带 `items`（因此渲染成子菜单）。
 * 调用方把它们展开进视图菜单的条目数组即可，不需要知道停靠状态的任何细节。
 */
export function useDockLayoutMenuItems({ withCheck }: DockLayoutMenuOptions): AppMenuItemSpec[] {
    const dispatch = useAppDispatch();
    const { tf } = useI18n();

    const layout = useAppSelector((state) => state.dock.layout);
    const confirmReset = useAppSelector((state) => state.dock.settings.confirmResetLayout);

    const entries = useMemo(() => listPanelEntriesFromLayout(layout), [layout]);
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
    // 用户点「导出布局」什么都不会发生（见 `services/api/jsonExport.ts`）。
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

    const windowsItem: AppMenuItemSpec = {
        key: "dock-windows",
        label: tf("menu_windows"),
        items: entries.map((entry) => ({
            key: entry.panelId,
            label: withCheck(entry.visible, tf(entry.titleKey)),
            onSelect: () => togglePanelVisible(dispatch, store.getState, entry.panelId),
        })),
    };

    const layoutItem: AppMenuItemSpec = {
        key: "dock-layout",
        label: tf("menu_layout"),
        items: [
            /*
             * 独立窗口中的窗体：给出"收回主窗口"的入口。
             *
             * 没有它的话，一旦卫星窗口的关闭回收没能执行（创建/登记竞态），窗体就
             * 既不在主窗口、也不在任何窗口里，用户**无法从界面恢复它** —— 只能重启
             * 应用（下次启动会重新打开那个窗口）。这是"面板凭空消失"的唯一出路。
             */
            ...(osWindowForms.length > 0
                ? [
                      {
                          key: "dock-os-windows",
                          label: tf("layout_os_windows"),
                          items: osWindowForms.map((form) => ({
                              key: form.id,
                              label: displayTitleOf(layout, form.id, tf),
                              onSelect: () =>
                                  void reclaimDetachedForm(dispatch, store.getState, form.id),
                          })),
                      },
                  ]
                : []),
            /*
             * 面板：现存的容器窗体。点按聚焦；标题按内容派生（重命名过则用用户文本）。
             * 仿照 os_windows 的模式：没有面板时不渲染空壳。
             */
            ...(panelForms.length > 0
                ? [
                      {
                          key: "dock-panels",
                          label: tf("layout_panels"),
                          items: panelForms.map((form) => ({
                              key: form.id,
                              label: panelTitleOf(layout, form.id, tf),
                              onSelect: () => dispatch(focusForm(form.id)),
                          })),
                      },
                  ]
                : []),
            {
                key: "dock-new-panel",
                label: tf("layout_new_panel"),
                onSelect: () => addEmptyPanel(dispatch),
            },
            {
                key: "dock-dissolve-panel",
                label: tf("layout_dissolve_panel"),
                danger: true,
                onSelect: () => dissolvePanelCommand(dispatch, store.getState),
            },
            {
                key: "dock-presets",
                label: tf("layout_presets"),
                items:
                    presetNames.length === 0
                        ? [
                              {
                                  key: "dock-no-presets",
                                  label: tf("layout_no_presets"),
                                  disabled: true,
                              },
                          ]
                        : presetNames.map((name) => ({
                              key: name,
                              label: withCheck(layout.activePreset === name, name),
                              onSelect: () => applyPreset(dispatch, name),
                          })),
            },
            {
                key: "dock-save-preset",
                label: tf("layout_save_preset"),
                onSelect: () => setDockDialogKind("namePrompt"),
            },
            ...(presetNames.length > 0
                ? [
                      {
                          key: "dock-delete-preset",
                          label: tf("layout_delete_preset"),
                          items: presetNames.map((name) => ({
                              key: name,
                              label: name,
                              danger: true,
                              // 不立即删除：菜单关闭即卸载，确认框住在常驻宿主里。
                              onSelect: () => requestDeletePreset(name),
                          })),
                      },
                  ]
                : []),
            {
                key: "dock-export",
                label: tf("layout_export"),
                separatorBefore: true,
                onSelect: () => void onExport(),
            },
            {
                key: "dock-import",
                label: tf("layout_import"),
                onSelect: () => dockImportInputRef.current?.click(),
            },
            {
                key: "dock-reset",
                label: tf("layout_reset"),
                separatorBefore: true,
                onSelect: () => {
                    // `confirmResetLayout` 关闭时直接重置，不弹窗。
                    if (confirmReset) setDockDialogKind("resetConfirm");
                    else resetLayout(dispatch);
                },
            },
            {
                key: "dock-settings",
                label: tf("layout_settings"),
                onSelect: () => setDockDialogKind("settings"),
            },
        ],
    };

    return [windowsItem, layoutItem];
}
