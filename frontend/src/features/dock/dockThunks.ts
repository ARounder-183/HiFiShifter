/*
 * 停靠设置的读写。
 *
 * 【为什么写入要去抖】拖动分隔条、移动浮窗这类操作会在松手瞬间触发一次布局
 * 变化；但"松手"本身可能连续发生（拖几下、调几次宽度）。后端
 * `save_ui_settings` 是"读-改-写整个配置文件 + 原子替换 + 备份副本"，约 8 次
 * 文件操作，是明确的昂贵调用（见 `config.rs::save_config` 的注释）。因此布局
 * 变化合并到一个去抖窗口里写入。
 *
 * 【为什么必须先 hydrate 再允许写入】切片初始状态是出厂布局。若在读到磁盘
 * 内容之前就因为任何原因触发一次写入，用户的布局会被默认值覆盖 —— 这是
 * "打开应用发现界面被重置"这类最恼人的故障。`hydrated` 标志就是这道闸门。
 */

import { createAsyncThunk } from "@reduxjs/toolkit";

import type { AppDispatch, RootState } from "../../app/store";
import { settingsApi } from "../../services/api/settings";
import { applyDockPreset, setDockLayout } from "./dockSlice";
import { findMainTabset } from "./dockSchema";
import { insertForm } from "./dockTree";

export const loadDockSettings = createAsyncThunk("dock/loadSettings", async () => {
    const ui = await settingsApi.getUiSettings();
    const dock = (ui.dock ?? null) as { layout?: unknown } | null;
    return { settings: ui.dock ?? null, layout: dock?.layout ?? null };
});

/** 把当前设置与布局整体写回后端。 */
export const persistDockSettings = createAsyncThunk(
    "dock/persistSettings",
    async (_: void, { getState }) => {
        const { dock } = getState() as RootState;
        await settingsApi.saveUiSettings({
            // 行为选项平铺 + 一份完整布局，一次写入同时落盘（见 `dockSettings`
            // 里 `DockPersistedSettings` 的注释）。
            dock: { ...dock.settings, layout: dock.layout },
        });
    },
);

/**
 * 恢复上次的排布。
 *
 * 在 `hydrateDock` 之后调用一次，处理两件超出"归一化"范围的事：
 * - `startupLayout` 指定了预设 → 套用它（多显示器/多工种用户开机即用）；
 * - `floatRestoreOnStartup` 关闭 → 把所有浮窗收回主编辑区标签组。
 *
 * 这两件事都必须在**布局已归一化、面板已注册**之后做，所以不能塞进 reducer
 * 的 hydrate 分支里。
 */
export function finalizeDockHydration(dispatch: AppDispatch, getState: () => RootState): void {
    const state = getState();
    const { settings, layout } = state.dock;

    if (settings.startupLayout !== "last" && layout.presets?.[settings.startupLayout]) {
        dispatch(applyDockPreset(settings.startupLayout));
    }

    if (!settings.floatRestoreOnStartup) {
        const current = getState().dock.layout;
        const main = findMainTabset(current);
        if (!main) return;
        const floating = current.floatOrder.filter((id) => Boolean(current.forms[id]?.float));
        if (floating.length === 0) return;

        const forms = { ...current.forms };
        let tree = current.tree;
        for (const formId of floating) {
            const form = forms[formId];
            if (!form) continue;
            forms[formId] = { ...form, float: null };
            tree = insertForm(tree, formId, { kind: "tab", tabsetId: main.id });
        }
        dispatch(setDockLayout({ ...current, tree, forms, floatOrder: [] }));
    }
}
