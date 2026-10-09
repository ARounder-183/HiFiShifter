/**
 * 插件模式下 Transport 按钮的提示文案。
 *
 * 【要钉死什么】这两个按钮在插件里控制的是**宿主**（REAPER）的播放，提示必须说清
 * 这一点，否则用户会以为点的是本地播放。它们曾经把这句话写在 `title` 上，而同一个
 * 按钮已经有一个走词表的 `data-tooltip` —— 两个 tooltip 并存，且那一个是硬编码中文。
 *
 * 【为什么真的挂 ActionBar】提示文案是内联在工具栏按钮上的，没有可以单独挂载的
 * 组件。最小 store 即可（见 `ActionBar.metronomeMenu.test.tsx` 的同一处理）。
 */
// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { combineReducers, configureStore } from "@reduxjs/toolkit";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import sessionReducer from "../../features/session/sessionSlice";
import dockReducer from "../../features/dock/dockSlice";
import fileBrowserReducer from "../../features/fileBrowser/fileBrowserSlice";
import keybindingsReducer from "../../features/keybindings/keybindingsSlice";
import notebookReducer from "../../features/notebook/notebookSlice";
import recordingReducer from "../../features/recording/recordingSlice";
import { enUS } from "../../i18n/en-US";
import { I18nProvider } from "../../i18n/I18nProvider";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import { ActionBar } from "./ActionBar";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;

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
                        <ActionBar />
                    </I18nProvider>
                </AppThemeProvider>
            </Provider>,
        );
    });
}

/** 按 `data-tooltip` 取按钮：文案可能随模式变化，不能靠文本内容定位。 */
function tooltips(): string[] {
    return Array.from(container.querySelectorAll<HTMLElement>("[data-tooltip]")).map(
        (element) => element.dataset.tooltip ?? "",
    );
}

test("standalone transport hints stay on the local wording", async () => {
    await render();
    const hints = tooltips();
    expect(hints).toContain(enUS.action_stop);
    expect(hints).not.toContain(enUS.plugin_transport_stop);
});

test("plugin transport hints name the host instead of leaving a hardcoded title", async () => {
    // 宿主能力齐备：播放/停止按钮可用，提示才应当切换成"控制宿主"的说法。
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "transport", transportControl: true };
    await render();
    const hints = tooltips();
    expect(hints).toContain(enUS.plugin_transport_stop);
    expect(hints).toContain(enUS.plugin_transport_play);
    // 本地措辞不得残留 —— 两个 tooltip 并存正是这次要消除的问题。
    expect(hints).not.toContain(enUS.action_stop);
    // 原生 `title` 不再参与：提示只有一个来源。
    expect(container.querySelector("[title]")).toBeNull();
});

/**
 * 插件模式下音乐上下文控件的归属。
 *
 * 【要钉死什么】BPM 与拍号由宿主拥有（`render/transport.rs` 从 VST3 进程上下文读），
 * 插件写不进去；节拍器与录音键也没有对应命令。它们此前是"禁用 + 说明原因"，但宿主
 * 给的 VST 窗口本来就窄，一个永远按不动的控件只该让路 —— 现在整块**不渲染**
 * （用户反馈：界面太挤）。REAPER 自己的工具栏就显示着这些值。
 *
 * 【为什么网格还在】网格是 HiFiShifter 自有的设置（宿主没有对应概念），插件通过
 * `set_project_timeline_settings` 真正支持它，因此必须保持可用。
 */
test("plugin mode does not render the host-owned tempo, meter and metronome controls", async () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "meter", transportControl: true };
    await render();
    // BPM / 拍号的输入框整块消失（不再有"禁用但占位"的输入框）。
    const inputs = Array.from(container.querySelectorAll<HTMLInputElement>("input")).map(
        (input) => input.value,
    );
    expect(inputs).not.toContain("120");
    expect(inputs).not.toContain("4");
    // 节拍器与录音键的提示也一并消失。
    const hints = tooltips();
    expect(hints).not.toContain(enUS.action_metronome);
    expect(
        hints.some((hint) => hint.startsWith(enUS.recording_tooltip_start)),
        "录音键不该在插件里留下一个按不动的按钮",
    ).toBe(false);
    // 网格下拉没有被隐藏：它在插件里是真的能改的。
    const grid = Array.from(container.querySelectorAll<HTMLElement>('[role="combobox"]')).some(
        (element) => element.getAttribute("disabled") === null,
    );
    expect(grid, "网格下拉在插件里必须可用").toBe(true);
});

/** 宿主不提供播放控制时，整组 transport 按钮不该以"禁用"的形态占位。 */
test("plugin mode hides the transport group when the host offers no control", async () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "no-transport" };
    await render();
    const hints = tooltips();
    expect(hints).not.toContain(enUS.plugin_transport_stop);
    expect(hints).not.toContain(enUS.plugin_transport_play);
});
