/*
 * 插件模式下的"速度映射"小按钮。
 *
 * 【要钉死什么】这个按钮做两件事：显示速度映射、并在没有 Tempo Map 时给工程建一个。
 * 插件里两件都不成立 —— BPM / 拍号是宿主权威，写入命令 `set_timeline_tempo_map` 不被
 * 支持，点下去只会得到一条被拒绝的错误（用户报障：`错误：Rejected`）。宿主窗口窄，
 * 所以整块**不渲染**，而不是留一个按不动的按钮占位。
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

test("plugin mode renders nothing: the write command is not supported there", async () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "tempo-corner" };
    await render();
    expect(container.querySelector("button")).toBeNull();
});
