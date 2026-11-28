// @vitest-environment jsdom
/*
 * 移动端工具栏的**覆盖门禁**：桌面能点的开关，手机上必须也能点到。
 *
 * 【为什么需要这条测试】用户报的"移动端仍然无法访问大部分工具栏功能，比如自动滚屏"
 * 不是某个按钮忘了适配，而是**整条通路不存在**：外壳在移动形态只渲染了菜单 sheet 与
 * 动作抽屉，而 `ACTION_META` 的 135 条动作里没有任何一条叫 autoScroll / ripple /
 * snap / dragDirection。于是"自动滚屏"在手机上既没有按钮、也没有动作 —— 没有任何
 * 报错，也没有任何日志。
 *
 * 这条测试把"控件清单"与"真实 DOM"绑在一起：
 *   1. 挂载**真实**的 `ToolbarStrip`（不是 stub），断言 sheet 里的
 *      `[data-toolbar-control]` 集合等于 `TOOLBAR_CONTROL_IDS`；
 *   2. 关掉任意一个 `data-toolbar-control`（或忘记在移动端渲染工具栏）都会让它失败；
 *   3. 反向断言：`ACTION_META` 里不得出现与工具栏开关同义的重复动作 —— 同一个开关
 *      在两处各有一份，状态一定会不同步。
 */
import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { combineReducers, configureStore } from "@reduxjs/toolkit";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import dockReducer from "../features/dock/dockSlice";
import fileBrowserReducer from "../features/fileBrowser/fileBrowserSlice";
import keybindingsReducer from "../features/keybindings/keybindingsSlice";
import { ACTION_META } from "../features/keybindings/defaultKeybindings";
import notebookReducer from "../features/notebook/notebookSlice";
import recordingReducer from "../features/recording/recordingSlice";
import sessionReducer from "../features/session/sessionSlice";
import { I18nProvider } from "../i18n/I18nProvider";
import {
    DESKTOP_FALLBACK_PROFILE,
    resetSurfaceProfileForTests,
    setSurfaceProfile,
} from "../services/surfaceProfileStore";
import { AppThemeProvider } from "../theme/AppThemeProvider";
import { ToolbarStrip } from "../components/layout/toolbar/ToolbarStrip";
import {
    PLUGIN_ONLY_TOOLBAR_CONTROLS,
    TOOLBAR_CONTROL_IDS,
} from "../components/layout/toolbar/toolbarControls";
import { AppChrome } from "./AppChrome";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;

const LOCALE_KEY = "hifishifter.locale";
let container: HTMLDivElement;
let root: ReturnType<typeof createRoot>;

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
    setSurfaceProfile({ ...DESKTOP_FALLBACK_PROFILE, form: "mobile" });
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
    resetSurfaceProfileForTests();
    vi.restoreAllMocks();
});

async function renderChrome() {
    await act(async () => {
        root.render(
            <Provider store={buildStore()}>
                <AppThemeProvider>
                    <I18nProvider>
                        <AppChrome
                            mobileMenu={<div data-testid="menu-stub" />}
                            mobileToolbar={<ToolbarStrip variant="sheet" />}
                            mobileTitle="测试工程"
                            mobileDirty={false}
                            canUndo={false}
                            canRedo={false}
                            isPlaying={false}
                            onTogglePlay={() => {}}
                            onStop={() => {}}
                            onMobileAction={() => {}}
                        >
                            <div data-testid="workspace" />
                        </AppChrome>
                    </I18nProvider>
                </AppThemeProvider>
            </Provider>,
        );
    });
}

function sheet(): HTMLElement | null {
    return container.querySelector<HTMLElement>("[data-mobile-toolbar-sheet]");
}

function controlIds(scope: ParentNode): string[] {
    return Array.from(scope.querySelectorAll("[data-toolbar-control]"))
        .map((element) => element.getAttribute("data-toolbar-control") ?? "")
        .filter((id) => id.length > 0)
        .sort();
}

/** 顶栏里的按钮。用结构定位（根容器的第一个子元素就是顶栏），不依赖文案。 */
function topBarButtons(): HTMLButtonElement[] {
    const shell = container.querySelector<HTMLElement>('[data-form="mobile"]');
    const bar = shell?.firstElementChild;
    if (!bar) throw new Error("顶栏未渲染");
    return Array.from(bar.querySelectorAll("button"));
}

test("移动端工具栏 sheet 里的控件清单与 TOOLBAR_CONTROL_IDS 完全一致", async () => {
    await renderChrome();
    const panel = sheet();
    expect(panel, "移动外壳必须挂载工具栏 sheet").not.toBeNull();
    // 插件专属控件在独立 App 里不渲染（见 `PLUGIN_ONLY_TOOLBAR_CONTROLS`）。
    const expected = TOOLBAR_CONTROL_IDS.filter(
        (id) => !PLUGIN_ONLY_TOOLBAR_CONTROLS.includes(id),
    ).sort();
    expect(controlIds(panel as HTMLElement)).toEqual(expected);
});

test("工具栏 sheet 关闭时仍挂载（那批对话框的开关状态住在里面）", async () => {
    await renderChrome();
    expect(sheet()?.getAttribute("data-mobile-toolbar-sheet")).toBe("closed");
    expect(sheet()?.className).toContain("hidden");
    // 关闭时控件仍在 DOM 里：卸载 ToolbarStrip 会让吸附网格 / 分割过渡等对话框
    // 退回到"点了没反应"。
    expect(controlIds(sheet() as HTMLElement).length).toBe(
        TOOLBAR_CONTROL_IDS.length - PLUGIN_ONLY_TOOLBAR_CONTROLS.length,
    );
});

test("顶栏的工具栏按钮打开 sheet，且三个入口互斥", async () => {
    await renderChrome();
    // 顶栏顺序：菜单 / 撤销 / 重做 / 工具栏 / 操作。按 aria-label 找（文案即无障碍名，
    // 见 `AppIconButton`），不用下标 —— 下标会随顶栏增删按钮而漂。
    const toolbarButton = topBarButtons().find(
        (button) => button.getAttribute("aria-label") === "Toolbar",
    );
    expect(toolbarButton, "顶栏必须有工具栏入口").toBeTruthy();
    await act(async () => {
        (toolbarButton as HTMLButtonElement).dispatchEvent(
            new MouseEvent("click", { bubbles: true, cancelable: true }),
        );
    });
    expect(sheet()?.getAttribute("data-mobile-toolbar-sheet")).toBe("open");
    expect(sheet()?.className).not.toContain("hidden");
    // 打开工具栏必须关掉菜单：两者都是全屏 sheet，叠在一起会互相盖住。
    expect(
        container.querySelector('[data-mobile-menu-sheet="open"]'),
        "菜单 sheet 不得与工具栏 sheet 同时打开",
    ).toBeNull();
});

test("ACTION_META 里不得有与工具栏开关同义的重复动作", async () => {
    // 同一切换在两处各有一份，状态必然不同步（用户会看到"抽屉里关了、工具栏还亮着"）。
    const forbidden = /autoscroll|ripple|playheadzoom|ignoregrouping|dragdirection/i;
    // 钢琴卷帘的"循环拖拽方向"是**另一个面**的画笔方向（与时间轴的 clip 拖拽方向
    // 不是同一个设置），因此不在这条断言的范围里。
    const duplicates = Object.keys(ACTION_META).filter(
        (id) => !id.startsWith("pianoRoll.") && forbidden.test(id),
    );
    expect(duplicates, "工具栏开关不得在动作抽屉里再有一份").toEqual([]);
});
