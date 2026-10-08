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
import { hostMode } from "../../services/hostCapabilities";
import { applyDockPreset, setDockLayout } from "./dockSlice";
import { findMainTabset } from "./dockSchema";
import { restoreDetachedWindows } from "./dockApi";
import { insertForm } from "./dockTree";
import { MAIN_ROOT_ID } from "./dockTypes";

/**
 * 布局字段名按形态选择。
 *
 * 【为什么两个形态各存一份】ARA 窗口下限 640×400，独立 App 是整屏。插件里为了能用
 * 必然折叠面板、把分隔条拖到极端比例 —— 那份布局在大窗口里只是"挤"，但为大窗口
 * 调好的布局在 640×400 里**真的不能用**（面板被裁到最小尺寸之下、gutter 超过视口）。
 * 共用同一个字段时两个形态会互相覆盖，谁都留不住。
 *
 * 判据：**凡是"正确取值取决于窗口有多少像素"的设置分模式**；locale / 快捷键 / 主题 /
 * 设备选择这些"取决于用户是谁"的仍然共用同一个键。
 */
function dockFieldKey(): "dock" | "dockPlugin" {
    return hostMode() === "plugin" ? "dockPlugin" : "dock";
}

export const loadDockSettings = createAsyncThunk("dock/loadSettings", async () => {
    const ui = await settingsApi.getUiSettings();
    const stored = ui[dockFieldKey()] ?? null;
    return {
        settings: stored,
        // 布局的**取值**由前端的 `normalizeDockLayout` 收口，这里不做假设。
        layout: (stored as { layout?: unknown } | null)?.layout ?? null,
        // 插件形态第一次打开（还没有自己的布局）时改用更小的默认轨道头宽度：
        // 默认的 256px 会吃掉 640px 窗口的 40%（见 `PLUGIN_DEFAULT_TRACK_HEADER_PX`）。
        pluginFirstRun: stored === null && hostMode() === "plugin",
    };
});

/** 把当前设置与布局整体写回后端。 */
export const persistDockSettings = createAsyncThunk(
    "dock/persistSettings",
    async (_: void, { getState }) => {
        const { dock } = getState() as RootState;
        // 行为选项平铺 + 一份完整布局，一次写入同时落盘（见 `dockSettings`
        // 里 `DockPersistedSettings` 的注释）。
        const payload = { ...dock.settings, layout: dock.layout };
        // 显式分支而不是计算键：计算键在 `Partial<UiSettings>` 上会退化成索引签名，
        // 写错字段名也不会被类型检查抓到。
        await settingsApi.saveUiSettings(
            hostMode() === "plugin" ? { dockPlugin: payload } : { dock: payload },
        );
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

    // 【不要在这里 return】下面还有"恢复独立窗口"必须执行：早返回会让它被静默跳过
    // （曾经如此 —— 没有浮窗时独立窗口永远不会被恢复，而用户上次拆出的窗口就此
    // 消失，只能重启）。因此这里只做条件分支，不做提前退出。
    if (!settings.floatRestoreOnStartup) {
        const current = getState().dock.layout;
        const main = findMainTabset(current);
        const floating = main
            ? current.floatOrder.filter((id) => current.forms[id]?.floating === true)
            : [];
        if (main && floating.length > 0) {
            const forms = { ...current.forms };
            const mainRootId = MAIN_ROOT_ID;
            let tree = current.roots[mainRootId];
            if (!tree) return;
            for (const formId of floating) {
                const form = forms[formId];
                if (!form) continue;
                forms[formId] = { ...form, floating: false };
                tree = insertForm(tree, formId, { kind: "tab", tabsetId: main.id });
            }
            dispatch(
                setDockLayout({
                    ...current,
                    roots: { ...current.roots, [mainRootId]: tree },
                    forms,
                    floatOrder: [],
                }),
            );
        }
    }

    // 恢复上次拆出的独立窗口（必须放在"收回浮窗"之后：被收回的窗体不该再开窗口）。
    void restoreDetachedWindows(dispatch, getState);
}
