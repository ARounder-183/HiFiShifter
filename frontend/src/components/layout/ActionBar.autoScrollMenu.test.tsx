/**
 * 工具栏里的自动滚屏按钮：**策略**层面的回归锁。
 *
 * 【为什么另开一个文件】`AutoScrollButton.test.tsx` 验证的是组件本身（图标、菜单、
 * 回调）。但"选模式即开启"这条**策略**住在 `ActionBar` 的接线里，组件看不见它 ——
 * 少了这条测试，把 `if (!s.autoScrollEnabled) dispatch(toggleAutoScroll())` 删掉
 * 不会有任何用例变红，而用户会看到"我选了翻页式，然后什么都没发生"。
 *
 * 【为什么真的挂 ActionBar】按钮没有可以单独挂载的"带 store 的版本"；这条策略的
 * 输入是 store 里的 `autoScrollEnabled`，只能整块挂起来才测得到。挂载方式与
 * `ActionBar.metronomeMenu.test.tsx` 完全一致（同一份最小 store + 同一套桩）。
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
let store: ReturnType<typeof buildStore>;

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
    // 工具栏里的设备/运行时调用在 jsdom 里没有后端：让 invoke 一律静默成功。
    window.__TAURI__ = { core: { invoke: async () => ({}) as never } };
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
    delete window.__TAURI__;
    vi.restoreAllMocks();
});

/** 自动滚屏按钮：它是工具栏里唯一带 `data-hs-auto-scroll-icon` 的那一枚。 */
function autoScrollButton(): HTMLButtonElement {
    const icon = container.querySelector("[data-hs-auto-scroll-icon]");
    const button = icon?.closest("button");
    expect(button, "自动滚屏按钮必须出现在工具栏里").toBeTruthy();
    return button as HTMLButtonElement;
}

async function renderActionBar() {
    await act(async () => {
        root.render(
            <Provider store={store}>
                <AppThemeProvider>
                    <I18nProvider>
                        <ActionBar />
                    </I18nProvider>
                </AppThemeProvider>
            </Provider>,
        );
    });
}

async function openModeMenu() {
    await act(async () => {
        autoScrollButton().dispatchEvent(
            new MouseEvent("contextmenu", { bubbles: true, clientX: 40, clientY: 40 }),
        );
    });
    const menu = document.body.querySelector('[data-hs-context-menu="1"]');
    expect(menu, "右键必须打开模式菜单").toBeTruthy();
    return menu as HTMLElement;
}

function menuItem(menu: HTMLElement, label: string): HTMLElement {
    const item = Array.from(menu.querySelectorAll<HTMLElement>("[role=menuitemcheckbox]")).find(
        (entry) => entry.textContent?.includes(label),
    );
    expect(item, `菜单里必须有「${label}」`).toBeTruthy();
    return item as HTMLElement;
}

test("出厂：滚动式图标、未激活、未设置过", async () => {
    await renderActionBar();
    const button = autoScrollButton();
    expect(container.querySelector('[data-hs-auto-scroll-icon="continuous"]')).toBeTruthy();
    expect(button.getAttribute("aria-pressed")).toBeNull();
    expect(store.getState().session.autoScrollMode).toBe("scroll");
    expect(store.getState().session.autoScrollEnabled).toBe(false);
});

test("★ 选模式 = 立即生效：写入模式、同时开启、图标随之切换", async () => {
    await renderActionBar();
    const menu = await openModeMenu();

    await act(async () => {
        menuItem(menu, "Page").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    const session = store.getState().session;
    expect(session.autoScrollMode).toBe("page");
    // 这条就是"选模式即开启"：少了它，用户点完菜单会觉得什么都没发生。
    expect(session.autoScrollEnabled).toBe(true);
    expect(container.querySelector('[data-hs-auto-scroll-icon="page"]')).toBeTruthy();
    expect(autoScrollButton().getAttribute("aria-pressed")).toBe("true");
});

test("左键仍只是开关：不弹菜单、不改模式", async () => {
    await renderActionBar();
    const before = store.getState().session.autoScrollMode;

    await act(async () => {
        autoScrollButton().dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(store.getState().session.autoScrollEnabled).toBe(true);
    expect(store.getState().session.autoScrollMode).toBe(before);
    expect(document.body.querySelector('[data-hs-context-menu="1"]')).toBeNull();
});

test("已开启时选模式不会把它关掉", async () => {
    await renderActionBar();
    await act(async () => {
        autoScrollButton().dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(store.getState().session.autoScrollEnabled).toBe(true);

    const menu = await openModeMenu();
    await act(async () => {
        menuItem(menu, "Continuous").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(store.getState().session.autoScrollMode).toBe("scroll");
    expect(store.getState().session.autoScrollEnabled).toBe(true);
});
