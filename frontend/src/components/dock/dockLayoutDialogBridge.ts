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

/** 待删除的预设名。菜单项置位，常驻宿主在弹确认框时读取。 */
let dockPresetToDelete: string | null = null;

export function setDockPresetToDelete(name: string | null): void {
    dockPresetToDelete = name;
}

export function getDockPresetToDelete(): string | null {
    return dockPresetToDelete;
}

/**
 * 纯通知对话框的正文。
 *
 * 【为什么与 kind 分开存放】快照仍是原始字符串（`useSyncExternalStore` 不会因对象
 * 不稳定而反复重渲染），正文在 kind 变化触发的那次重渲染里直接读取即可。
 */
let dockNoticeMessage: string | null = null;

export function setDockNoticeMessage(message: string | null): void {
    dockNoticeMessage = message;
}

export function getDockNoticeMessage(): string | null {
    return dockNoticeMessage;
}

/**
 * 导入布局用的隐藏 `<input>`。
 *
 * 【为什么由桥持有】输入框必须由**常驻宿主**（`DockLayoutDialogs`）渲染：若它随菜单
 * 内容一起卸载，文件选择期间的 `change` 事件就没人接了。菜单项只需要一个"点它"的
 * 入口，因此这里共享同一个 ref 对象。
 */
export const dockImportInputRef: { current: HTMLInputElement | null } = { current: null };

/** 打开错误通知（取代此前的 `window.alert`）。 */
export function setDockNotice(message: string): void {
    setDockNoticeMessage(message);
    setDockDialogKind("notice");
}

/** 请求删除预设：菜单项只置位（记下待删名字），常驻宿主弹确认框。 */
export function requestDeletePreset(name: string): void {
    setDockPresetToDelete(name);
    setDockDialogKind("deletePresetConfirm");
}
