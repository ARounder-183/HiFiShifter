/**
 * 状态栏"渲染中"片的显示契约。
 *
 * 【要钉死什么】
 * 1. 插件模式由 **250 ms 快照**驱动：事件丢了也要显示"渲染中 42%"；
 * 2. `render_audible_stale === true` 时左侧不显示 —— 右侧 PluginApplyStatus
 *    已经在说"正在合成 xx%"，两处重复是噪音；
 * 3. 独立 App 仍走 Redux + appStatusProgressBus，行为不变。
 */
// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { combineReducers, configureStore } from "@reduxjs/toolkit";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import sessionReducer, { setPlaybackRenderingState } from "../../features/session/sessionSlice";
import interactionReducer from "../../features/session/interactionSlice";
import { I18nProvider } from "../../i18n/I18nProvider";
import { enUS } from "../../i18n/en-US";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import { appStatusProgressBus } from "../../utils/appStatusProgressBus";
import {
    publishPluginApplyStateForTests,
    resetPluginApplyStoreForTests,
    type PluginApplyState,
} from "../../features/ara/pluginApplyStore";
import { AppStatusProgressChips } from "./AppStatusProgressChips";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const LOCALE_KEY = "hifishifter.locale";
let container: HTMLDivElement;
let root: Root;
let store: ReturnType<typeof buildStore>;

function buildStore() {
    return configureStore({
        reducer: combineReducers({
            session: sessionReducer,
            interaction: interactionReducer,
        }),
        middleware: (getDefault) => getDefault({ serializableCheck: false, immutableCheck: false }),
    });
}

function pluginState(overrides: Partial<PluginApplyState> = {}): PluginApplyState {
    return {
        generation: 1,
        applied_generation: 1,
        pending: false,
        connected: true,
        ready: true,
        error: null,
        ...overrides,
    };
}

beforeEach(() => {
    localStorage.setItem(LOCALE_KEY, "en-US");
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    resetPluginApplyStoreForTests();
    appStatusProgressBus.setRenderingProgress(null);
    appStatusProgressBus.setStretching({ active: false, clipName: null });
    appStatusProgressBus.setWaveformAnalysis({ active: false, sourcePath: null, progress: null });
    delete (window as { __HFS_PLUGIN_BOOTSTRAP__?: unknown }).__HFS_PLUGIN_BOOTSTRAP__;
    store = buildStore();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    container.remove();
    document.body.innerHTML = "";
    localStorage.removeItem(LOCALE_KEY);
    delete (window as { __HFS_PLUGIN_BOOTSTRAP__?: unknown }).__HFS_PLUGIN_BOOTSTRAP__;
    vi.restoreAllMocks();
});

async function render() {
    await act(async () => {
        root.render(
            <Provider store={store}>
                <AppThemeProvider>
                    <I18nProvider>
                        <AppStatusProgressChips />
                    </I18nProvider>
                </AppThemeProvider>
            </Provider>,
        );
    });
}

function text(): string {
    return container.textContent ?? "";
}

test("插件模式下快照的 rendering_active / render_progress 驱动渲染中片", async () => {
    (window as { __HFS_PLUGIN_BOOTSTRAP__?: unknown }).__HFS_PLUGIN_BOOTSTRAP__ = {
        version: 1,
        viewId: "test",
    };
    publishPluginApplyStateForTests(
        pluginState({ rendering_active: true, render_progress: 0.42, render_audible_stale: false }),
    );
    await render();
    expect(text()).toContain(enUS.common_rendering);
    expect(text()).toContain("42%");
});

test("渲染中片跟随用户自定义的强调色，而不是灰色", async () => {
    (window as { __HFS_PLUGIN_BOOTSTRAP__?: unknown }).__HFS_PLUGIN_BOOTSTRAP__ = {
        version: 1,
        viewId: "test",
    };
    publishPluginApplyStateForTests(
        pluginState({ rendering_active: true, render_progress: 0.42, render_audible_stale: false }),
    );
    await render();
    const chip = Array.from(container.querySelectorAll("span")).find((element) =>
        element.textContent?.includes(enUS.common_rendering),
    );
    expect(chip).toBeTruthy();
    // 状态片的颜色令牌必须仍是 accent 档；写成 neutral 会让所有「渲染中 / 分析中 /
    // 导入中」提示里唯独这一片变灰，与用户设置的强调色脱节。
    expect(chip?.style.background).toContain("--accent-3");
    expect(chip?.style.color).toContain("--accent-11");
});

test("stale 时左侧不显示，避免与右侧「正在合成」重复", async () => {
    (window as { __HFS_PLUGIN_BOOTSTRAP__?: unknown }).__HFS_PLUGIN_BOOTSTRAP__ = {
        version: 1,
        viewId: "test",
    };
    publishPluginApplyStateForTests(
        pluginState({ rendering_active: true, render_progress: 0.42, render_audible_stale: true }),
    );
    await render();
    expect(text()).not.toContain(enUS.common_rendering);
});

test("独立 App 仍读 Redux 与 appStatusProgressBus", async () => {
    store.dispatch(
        setPlaybackRenderingState({ active: true, target: "background", blocking: false }),
    );
    appStatusProgressBus.setRenderingProgress(0.42);
    await render();
    expect(text()).toContain(enUS.common_rendering);
    expect(text()).toContain("42%");
});
