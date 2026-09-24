/*
 * 停靠系统的命令式 API。
 *
 * 【为什么要单独一层】菜单、快捷键、工具栏按钮，以及将来的公开插件 API，
 * 需要的是"打开记事本""把当前窗体浮起来""套用混音布局"这类动作，而不是
 * Redux action 的形状。把它们收在这里，有三点好处：
 *
 * 1. 菜单与快捷键共享同一份行为，"菜单能做的快捷键做不了"这类分叉不会出现；
 * 2. 需要多步的动作用函数表达（如"最大化"要记住原树），不必往 reducer 里塞
 *    临时字段；
 * 3. 将来开放 API 时，暴露的就是这几个函数 —— 内部实现不需要为 API 再改一次。
 *
 * 所有函数都接受 `dispatch` / `getState` 而不是自己 import store，便于测试与
 * 复用（例如在一个未挂载到主 store 的场景里）。
 */

import type { AppDispatch, RootState } from "../../app/store";
import { findMainTabset, normalizeDockLayout } from "./dockSchema";
import { collectTabsets, isFormVisible } from "./dockTree";
import { getPanel, listPanels } from "./panelRegistry";
import {
    applyDockPreset,
    closeForm,
    deleteDockPreset,
    dockFormTo,
    floatForm,
    focusForm,
    openPanel,
    resetDockLayout,
    saveDockPreset,
    setDockLayout,
    toggleMaximizeActive,
} from "./dockSlice";
import type { DockLayout } from "./dockTypes";

type GetState = () => RootState;

/** 打开面板（已打开则聚焦它）。 */
export function openPanelById(dispatch: AppDispatch, getState: GetState, panelId: string): void {
    const layout = getState().dock.layout;
    const formId = visibleFormFor(layout, panelId);
    if (formId) {
        dispatch(focusForm(formId));
        return;
    }
    dispatch(openPanel({ panelId }));
}

/** 显隐切换：可见则关闭，不可见则打开。 */
export function togglePanelVisible(
    dispatch: AppDispatch,
    getState: GetState,
    panelId: string,
): void {
    const formId = visibleFormFor(getState().dock.layout, panelId);
    if (formId) {
        dispatch(closeForm(formId));
        return;
    }
    dispatch(openPanel({ panelId }));
}

/** 按面板 id 关闭它当前可见的窗体（面板自己的关闭按钮用）。 */
export function closeFormById(dispatch: AppDispatch, getState: GetState, panelId: string): void {
    const formId = visibleFormFor(getState().dock.layout, panelId);
    if (formId) dispatch(closeForm(formId));
}

export function isPanelVisible(getState: GetState, panelId: string): boolean {
    return visibleFormFor(getState().dock.layout, panelId) !== null;
}

/**
 * 选择器工厂：某面板当前是否可见。
 *
 * 面板显隐的**唯一事实源是布局树**（`isFormVisible` 由"在树上或有浮动几何"
 * 派生），不再有第二个布尔标志。这样"工具栏按钮亮着、面板却不在屏幕上"
 * 这类自相矛盾的状态从结构上不可能出现。
 */
export function selectPanelVisible(panelId: string) {
    return (state: RootState): boolean => {
        const layout = state.dock.layout;
        for (const formId of layout.order) {
            if (layout.forms[formId]?.panelId !== panelId) continue;
            if (isFormVisible(layout, formId)) return true;
        }
        return false;
    };
}

function visibleFormFor(layout: DockLayout, panelId: string): string | null {
    for (const formId of layout.order) {
        if (layout.forms[formId]?.panelId !== panelId) continue;
        if (isFormVisible(layout, formId)) return formId;
    }
    return null;
}

/** 当前活动窗体：优先用显式的活动窗体，否则取主标签组的活动标签。 */
export function activeFormId(getState: GetState): string | null {
    const { layout, activeFormId: active } = getState().dock;
    if (active && isFormVisible(layout, active)) return active;
    return findMainTabset(layout)?.active ?? null;
}

/** 停靠 ⇄ 浮动 切换。 */
export function toggleFloatActive(dispatch: AppDispatch, getState: GetState): void {
    const formId = activeFormId(getState);
    if (!formId) return;
    const form = getState().dock.layout.forms[formId];
    if (!form) return;
    if (form.floating) {
        const main = findMainTabset(getState().dock.layout);
        if (!main) return;
        dispatch(dockFormTo({ formId, target: { kind: "tab", tabsetId: main.id } }));
        return;
    }
    dispatch(floatForm({ formId }));
}

/** 聚焦下一个/上一个窗体（在可见窗体之间循环）。 */
export function cycleFocus(dispatch: AppDispatch, getState: GetState, delta: 1 | -1): void {
    const layout = getState().dock.layout;
    const visible = layout.order.filter((formId) => isFormVisible(layout, formId));
    if (visible.length === 0) return;
    const current = activeFormId(getState);
    const index = current ? visible.indexOf(current) : -1;
    const nextIndex = (index + delta + visible.length) % visible.length;
    dispatch(focusForm(visible[nextIndex]));
}

/** 把当前窗体的标签组放大到整片工作区，再按一次还原（REAPER 的最大化）。 */
export function maximizeActive(dispatch: AppDispatch): void {
    dispatch(toggleMaximizeActive());
}

/** 当前是否处于"最大化某窗体"的临时状态。 */
export function isMaximized(getState: GetState): boolean {
    return getState().dock.maximized !== null;
}

/** 重置为出厂布局。 */
export function resetLayout(dispatch: AppDispatch): void {
    dispatch(resetDockLayout());
}

/** 把当前排布存为命名预设。 */
export function savePreset(dispatch: AppDispatch, name: string): void {
    dispatch(saveDockPreset(name));
}

export function applyPreset(dispatch: AppDispatch, name: string): void {
    dispatch(applyDockPreset(name));
}

export function deletePreset(dispatch: AppDispatch, name: string): void {
    dispatch(deleteDockPreset(name));
}

export function listPresetNames(getState: GetState): string[] {
    return listPresetNamesFromLayout(getState().dock.layout);
}

/**
 * 纯函数版本：按布局对象列出预设名（最近保存的在前）。
 *
 * 组件里应当用它 + `useAppSelector(s => s.dock.layout)`，而不是把 `getState`
 * 传进选择器 —— 后者每次都会构造新数组，`useSyncExternalStore` 会因快照不稳
 * 定而反复重渲染（React 会就此告警）。布局对象的引用只在真正变化时才变，
 * 因此"订阅布局 + 就地派生"是这里的正确形状。
 */
export function listPresetNamesFromLayout(layout: DockLayout): string[] {
    const presets = layout.presets ?? {};
    return Object.keys(presets).sort((a, b) => {
        const at = presets[a]?.createdAtMs ?? 0;
        const bt = presets[b]?.createdAtMs ?? 0;
        return bt - at;
    });
}

/** 导出布局为可分享的 JSON 文本。 */
export function exportLayoutJson(getState: GetState): string {
    return exportLayoutJsonFromLayout(getState().dock.layout);
}

export function exportLayoutJsonFromLayout(layout: DockLayout): string {
    return JSON.stringify(
        {
            kind: "hifishifter.layout",
            schema: layout.schema,
            tree: layout.tree,
            forms: layout.forms,
            order: layout.order,
            floatOrder: layout.floatOrder,
            gutters: layout.gutters,
        },
        null,
        2,
    );
}

/** 导入布局 JSON；返回是否成功（失败时布局不变）。 */
export function importLayoutJson(dispatch: AppDispatch, json: string): boolean {
    let parsed: unknown;
    try {
        parsed = JSON.parse(json);
    } catch {
        return false;
    }
    if (!parsed || typeof parsed !== "object") return false;
    const normalized = normalizeDockLayout(parsed);
    // 归一化一定会产出可用布局，所以"是否导入成功"由调用方通过布局是否变化
    // 之外的信息判断 —— 这里额外校验树非空，避免把垃圾 JSON 当成成功。
    if (collectTabsets(normalized.tree).length === 0) return false;
    dispatch(setDockLayout(normalized));
    return true;
}

export interface DockPanelEntry {
    panelId: string;
    titleKey: string;
    visible: boolean;
    formId: string | null;
}

/** 面板清单（"显示窗体"菜单用）。 */
export function listPanelEntries(getState: GetState): DockPanelEntry[] {
    return listPanelEntriesFromLayout(getState().dock.layout);
}

/** 纯函数版本，理由同 `listPresetNamesFromLayout`。 */
export function listPanelEntriesFromLayout(layout: DockLayout): DockPanelEntry[] {
    return listPanels().map((panel) => {
        const formId = visibleFormFor(layout, panel.id);
        return {
            panelId: panel.id,
            titleKey: panel.titleKey,
            visible: formId !== null,
            formId,
        };
    });
}

/** 供 UI 判断某面板是否注册过（菜单据此隐藏未注册项）。 */
export function isRegistered(panelId: string): boolean {
    return getPanel(panelId) !== undefined;
}
