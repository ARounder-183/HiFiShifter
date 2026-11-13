/**
 * 工具栏里的波纹编辑按钮：**策略**层面的回归锁（2×2 = 开关 × 方向）。
 *
 * 【要钉死什么】
 * 1. 左键只做开关 —— 不再是三档循环；
 * 2. 右键菜单只有两个**方向**项，"关闭"不在菜单里（它是开关的关闭态）；
 * 3. 选方向 = 立即开启（与相邻的"自动滚屏"同一条策略），图标随之切换；
 * 4. 已开启时选方向不会把它关掉。
 *
 * 【为什么真的挂 ActionBar】"选方向即开启"这条策略住在 ActionBar 的接线里，
 * 组件本身看不见它。挂载方式与 `ActionBar.autoScrollMenu.test.tsx` 完全一致。
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

/**
 * 工具栏上的波纹按钮。
 *
 * 【为什么排除菜单里的那个】菜单项里也画同一枚图标（让"图标 ↔ 方向"的对应关系在
 * 菜单里就学会）；菜单本身也渲染在这个容器里，因此必须按"不在菜单壳内"过滤。
 */
function rippleButton(): HTMLButtonElement {
    const button = Array.from(container.querySelectorAll<HTMLButtonElement>("button")).find(
        (entry) =>
            entry.querySelector("[data-hs-ripple-icon]") !== null &&
            entry.closest("[data-hs-context-menu]") === null,
    );
    expect(button, "波纹编辑按钮必须出现在工具栏里").toBeTruthy();
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

async function openScopeMenu() {
    await act(async () => {
        rippleButton().dispatchEvent(
            new MouseEvent("contextmenu", { bubbles: true, clientX: 40, clientY: 40 }),
        );
    });
    const menu = document.body.querySelector('[data-hs-context-menu="1"]');
    expect(menu, "右键必须打开方向菜单").toBeTruthy();
    return menu as HTMLElement;
}

function menuItem(menu: HTMLElement, label: string): HTMLElement {
    const item = Array.from(menu.querySelectorAll<HTMLElement>("[role=menuitemcheckbox]")).find(
        (entry) => entry.textContent?.includes(label),
    );
    expect(item, `菜单里必须有「${label}」`).toBeTruthy();
    return item as HTMLElement;
}

test("出厂：按轨道图标、未激活", async () => {
    await renderActionBar();
    expect(container.querySelector('[data-hs-ripple-icon="track"]')).toBeTruthy();
    // 右下角小三角：告诉用户"右键还有方向可选"。
    expect(rippleButton().querySelector("[data-hs-menu-caret]")).toBeTruthy();
    expect(rippleButton().getAttribute("aria-pressed")).toBeNull();
    expect(store.getState().session.rippleEnabled).toBe(false);
    expect(store.getState().session.rippleScope).toBe("track");
});

test("右键菜单只有两个方向项，没有「关闭」", async () => {
    await renderActionBar();
    const menu = await openScopeMenu();
    // 菜单里**没有**标题行：只有两个方向项。
    expect(menu.querySelector(".hs-menu__label")).toBeNull();
    const items = Array.from(menu.querySelectorAll<HTMLElement>("[role=menuitemcheckbox]"));
    expect(items).toHaveLength(2);
    expect(items.map((item) => item.textContent)).toEqual(["Per Track", "All Tracks"]);
    // 当前方向打勾，另一个不打。
    expect(menuItem(menu, "Per Track").getAttribute("aria-checked")).toBe("true");
    expect(menuItem(menu, "All Tracks").getAttribute("aria-checked")).toBe("false");
});

test("左键只做开关：不弹菜单、不改方向", async () => {
    await renderActionBar();
    await act(async () => {
        rippleButton().dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(store.getState().session.rippleEnabled).toBe(true);
    expect(store.getState().session.rippleScope).toBe("track");
    expect(document.body.querySelector('[data-hs-context-menu="1"]')).toBeNull();
    expect(rippleButton().getAttribute("aria-pressed")).toBe("true");
});

test("★ 选方向 = 立即生效：写入方向、同时开启、图标随之切换", async () => {
    await renderActionBar();
    const menu = await openScopeMenu();

    await act(async () => {
        menuItem(menu, "All Tracks").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    const session = store.getState().session;
    expect(session.rippleScope).toBe("all");
    // 这条就是"选方向即开启"：少了它，用户点完菜单会觉得什么都没发生。
    expect(session.rippleEnabled).toBe(true);
    expect(rippleButton().querySelector('[data-hs-ripple-icon="all"]')).toBeTruthy();
    expect(rippleButton().getAttribute("aria-pressed")).toBe("true");
});

test("已开启时选方向不会把它关掉", async () => {
    await renderActionBar();
    await act(async () => {
        rippleButton().dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(store.getState().session.rippleEnabled).toBe(true);

    const menu = await openScopeMenu();
    await act(async () => {
        menuItem(menu, "All Tracks").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(store.getState().session.rippleScope).toBe("all");
    expect(store.getState().session.rippleEnabled).toBe(true);
});
