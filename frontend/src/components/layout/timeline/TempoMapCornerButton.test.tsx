/*
 * "速度映射"小按钮在两种模式下都渲染。
 *
 * 【要钉死什么】这个按钮做两件事：显示速度映射、并在没有 Tempo Map 时建一个
 * （只含 0 位置初始点 = 工程基准记录）。两件事在插件里都成立 —— Tempo Map 是
 * "随时间变化的音阶"的存储，音阶是 HiFiShifter 自有的；插件实现了
 * `set_timeline_tempo_map` 且只接受音阶轴，BPM/拍号在对话框里只读。
 * 上一轮"插件里整块不渲染"的决定已随写入命令落地而撤销。
 */
// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { combineReducers, configureStore } from "@reduxjs/toolkit";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import sessionReducer from "../../../features/session/sessionSlice";
import dockReducer from "../../../features/dock/dockSlice";
import fileBrowserReducer from "../../../features/fileBrowser/fileBrowserSlice";
import keybindingsReducer from "../../../features/keybindings/keybindingsSlice";
import notebookReducer from "../../../features/notebook/notebookSlice";
import recordingReducer from "../../../features/recording/recordingSlice";
import { enUS } from "../../../i18n/en-US";
import { I18nProvider } from "../../../i18n/I18nProvider";
import { AppThemeProvider } from "../../../theme/AppThemeProvider";
import { TempoMapCornerButton } from "./TempoMapCornerButton";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const LOCALE_KEY = "hifishifter.locale";
let container: HTMLDivElement;
let root: Root;

function buildStore() {
    return configureStore({
        reducer: combineReducers({
            session: sessionReducer,
            dock: dockReducer,
            fileBrowser: fileBrowserReducer,
            keybindings: keybindingsReducer,
            notebook: notebookReducer,
            recording: recordingReducer,
        }),
        middleware: (getDefault) => getDefault({ serializableCheck: false, immutableCheck: false }),
    });
}

beforeEach(() => {
    localStorage.setItem(LOCALE_KEY, "en-US");
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    window.__TAURI__ = { core: { invoke: async () => ({}) as never } };
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    container.remove();
    document.body.innerHTML = "";
    localStorage.removeItem(LOCALE_KEY);
    delete window.__TAURI__;
    delete window.__HFS_PLUGIN_BOOTSTRAP__;
    vi.restoreAllMocks();
});

async function render() {
    await act(async () => {
        root.render(
            <Provider store={buildStore()}>
                <AppThemeProvider>
                    <I18nProvider>
                        <TempoMapCornerButton />
                    </I18nProvider>
                </AppThemeProvider>
            </Provider>,
        );
    });
}

test("standalone keeps the tempo-map corner button", async () => {
    await render();
    expect(container.querySelector("button")).not.toBeNull();
    expect(container.querySelector("button")?.dataset.tooltip).toBe(enUS.tempo_map_show_tooltip);
});

test("plugin mode also keeps the corner button", async () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "tempo-corner" };
    await render();
    // Tempo Map 是"随时间变化的音阶"的存储，而音阶是 HiFiShifter 自有的 ——
    // 插件里这个按钮照常可用（BPM/拍号在对话框里只读，见 canEditTempoMapTempo）。
    expect(container.querySelector("button")).not.toBeNull();
    expect(container.querySelector("button")?.dataset.tooltip).toBe(enUS.tempo_map_show_tooltip);
});
