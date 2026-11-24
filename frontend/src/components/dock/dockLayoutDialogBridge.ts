/*
 * 「布局」菜单 → 对话框宿主的通信桥。
 *
 * 【为什么单独成文件】二级菜单在菜单关闭时卸载，持有不了对话框的开关状态；
 * 把状态提升到 `MenuBar` 又会让停靠 UI 的细节漏进菜单栏。这里用模块级单值桥接：
 * 菜单项置位、常驻宿主（`DockLayoutDialogs`）订阅。快照是原始值，
 * `useSyncExternalStore` 不会抖。
 *
 * 【为什么从 `DockLayoutMenu.tsx` 里搬出来】那个文件只导出组件；再导出一个
 * 普通函数会破坏 React Fast Refresh 的"文件只导出组件"约定（eslint
 * `react-refresh/only-export-components` 会拦下）。移动端抽屉需要
 * `openDockLayoutSettingsDialog`，因此把这份共享状态放到这里 —— 两边引用的是
 * **同一个模块实例**，开关状态只有一份。
 */

import { useSyncExternalStore } from "react";

/** 由「布局」菜单触发的对话框。 */
export type DockDialogKind =
    "settings" | "namePrompt" | "resetConfirm" | "deletePresetConfirm" | "notice" | null;

let dockDialogKind: DockDialogKind = null;
const dockDialogListeners = new Set<() => void>();

/** 置位对话框类型（`null` = 全部关闭）。 */
export function setDockDialogKind(kind: DockDialogKind): void {
    if (dockDialogKind === kind) return;
    dockDialogKind = kind;
    dockDialogListeners.forEach((listener) => listener());
}

/** 常驻宿主订阅当前对话框类型。 */
export function useDockDialogKind(): DockDialogKind {
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

/**
 * 打开「布局设置」对话框。
 *
 * 【为什么导出】移动端抽屉需要一个布局设置的入口，而对话框的宿主是 `MenuBar`
 * 常驻渲染的 `DockLayoutDialogs`（见 `AppChrome` 的 `mobileMenu`）。走这里就
 * 复用了同一份开关状态 —— 在移动端再造一个对话框实例，等于让"设置"有两个互不
 * 知情的副本，改了一个另一个还显示旧值。
 */
export function openDockLayoutSettingsDialog(): void {
    setDockDialogKind("settings");
}
