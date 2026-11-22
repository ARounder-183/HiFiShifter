/*
 * 参数编辑器工具栏的「REAPER 音高」菜单：**只在 REAPER ARA 里出现**，且每个作用域都带计数。
 *
 * 【要钉死什么】
 * 1. 宿主不是 REAPER（或后端没查到包络 API）时**整项不渲染** —— 不是置灰。独立 App 里
 *    这两个方向没有意义，留一个永远点不动的按钮只会让"这功能坏了"变成第一印象。
 * 2. 没有选区时"选区内"那一项不出现（它恒为 0，摆一个 (0) 只会让人以为选区丢了）。
 * 3. 每一项都带计数：这是"这一下会动多少条片段"在**点击之前**唯一的防呆。
 */
// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { combineReducers, configureStore } from "@reduxjs/toolkit";
import type { Store } from "@reduxjs/toolkit";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import sessionReducer from "../../../../features/session/sessionSlice";
import dockReducer from "../../../../features/dock/dockSlice";
import keybindingsReducer from "../../../../features/keybindings/keybindingsSlice";
import { enUS } from "../../../../i18n/en-US";
import { I18nProvider } from "../../../../i18n/I18nProvider";
import { writableHostCaps, type HostCaps } from "../../../../services/pluginBootstrapFixtures";
import { settingsApi } from "../../../../services/api";
import { webApi } from "../../../../services/webviewApi";
import { AppThemeProvider } from "../../../../theme/AppThemeProvider";
import { PitchBridgeMenu } from "./PitchBridgeMenu";
import type { PianoRollPanelModel } from "./usePianoRollPanelModel";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;
let store: Store;

function buildStore() {
    return configureStore({
        reducer: combineReducers({
            session: sessionReducer,
            dock: dockReducer,
            keybindings: keybindingsReducer,
        }),
        middleware: (getDefault) => getDefault({ serializableCheck: false, immutableCheck: false }),
    });
}

const SCOPES = {
    track: ["c1", "c2"],
    selection: ["c2"],
    all: ["c1", "c2", "c3"],
};

async function mount(scopes: typeof SCOPES = SCOPES) {
    store = buildStore();
    await act(async () => {
        root.render(
            <Provider store={store}>
                <AppThemeProvider>
                    <I18nProvider>
                        <PitchBridgeMenu
                            m={{
                                // `buildStore` 用 `combineReducers` 建的 store 丢了 thunk 的
                                // dispatch 类型；这里只是把它交回面板模型本身的口径。
                                dispatch:
                                    store.dispatch as unknown as PianoRollPanelModel["dispatch"],
                                s: store.getState().session,
                                pitchBridgeScopes: scopes,
                            }}
                        />
                    </I18nProvider>
                </AppThemeProvider>
            </Provider>,
        );
    });
}

function pluginCaps(overrides: Partial<HostCaps> = {}): HostCaps {
    return writableHostCaps({ split: true, itemClipboard: true, ...overrides });
}

function setBootstrap(caps: HostCaps, hostKind: "reaper" | "ara_only" = "reaper") {
    window.__HFS_PLUGIN_BOOTSTRAP__ = {
        version: 1,
        viewId: "pitch-bridge-menu",
        hostKind,
        hostCapabilities: caps,
    };
}

/** 可点项的文案（不含锚点按钮本身）。 */
function menuLabels(): string[] {
    return Array.from(document.body.querySelectorAll(".hs-menu__item")).map(
        (el) => el.textContent ?? "",
    );
}

/** 只读行（分组标题 / 说明行）的文案：它们不是按钮，点不动，也不参与键盘导航。 */
function headingLabels(): string[] {
    return Array.from(document.body.querySelectorAll(".hs-menu__label")).map(
        (el) => el.textContent ?? "",
    );
}

/** 右键展开菜单 —— 与工具栏其它带小三角的按钮同一条协议。 */
async function openMenu() {
    const trigger = container.querySelector("button");
    expect(trigger, "工具栏按钮必须渲染出来").toBeTruthy();
    await act(async () => {
        trigger!.dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true }));
    });
}

beforeEach(() => {
    localStorage.setItem("hifishifter.locale", "en-US");
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    container.remove();
    document.body.innerHTML = "";
    localStorage.removeItem("hifishifter.locale");
    delete window.__HFS_PLUGIN_BOOTSTRAP__;
});

test("it renders nothing outside a REAPER ARA session", async () => {
    // 独立 App：没有插件引导，能力位一律为假。
    await mount();
    expect(container.querySelector("button")).toBeNull();

    // 插件里但不是 REAPER（同一个 takePitchBridge 能力位）：仍然不出现。
    setBootstrap(pluginCaps({ takePitchBridge: true }), "ara_only");
    await mount();
    expect(container.querySelector("button")).toBeNull();
});

test("the three scopes carry their counts", async () => {
    setBootstrap(pluginCaps({ takePitchBridge: true }));
    await mount();
    await openMenu();

    const headings = headingLabels();
    expect(headings).toContain(enUS.pitch_bridge_import_heading);
    expect(headings).toContain(enUS.pitch_bridge_export_heading);

    // 两个方向各三项（当前轨道 / 选区内 / 全部），计数取自作用域。
    const labels = menuLabels();
    expect(labels).toContain("Current track (2)");
    expect(labels).toContain("In selection (1)");
    expect(labels).toContain("All clips (3)");
    expect(labels.filter((label) => label === "All clips (3)")).toHaveLength(2);
});

test("the selection scope disappears when there is no selection", async () => {
    setBootstrap(pluginCaps({ takePitchBridge: true }));
    await mount({ ...SCOPES, selection: [] });
    await openMenu();

    const labels = menuLabels();
    expect(labels).not.toContain("In selection (0)");
    expect(labels).toContain("Current track (2)");
});

test("a scope with nothing in it is disabled instead of silently doing nothing", async () => {
    setBootstrap(pluginCaps({ takePitchBridge: true }));
    await mount({ track: [], selection: ["c2"], all: ["c1"] });
    await openMenu();

    const entries = Array.from(
        document.body.querySelectorAll<HTMLButtonElement>("button.hs-menu__item"),
    );
    const empty = entries.find((el) => (el.textContent ?? "").startsWith("Current track (0)"));
    expect(empty, "当前轨道 (0) 必须仍然列出（作用域是稳定的），但点不动").toBeTruthy();
    expect(empty!.disabled).toBe(true);
    const filled = entries.find((el) => (el.textContent ?? "").startsWith("All clips (1)"));
    expect(filled!.disabled).toBe(false);
});
test("the left button exports the current track while the right button opens the scopes", async () => {
    setBootstrap(pluginCaps({ takePitchBridge: true }));
    const exported = vi
        .spyOn(webApi, "exportHostTakePitch")
        .mockResolvedValue({ ok: true, clipsConverted: 2 } as never);
    await mount();

    const trigger = container.querySelector<HTMLButtonElement>("button")!;
    expect(trigger.disabled).toBe(false);
    // 同一排的按钮都是「左键立刻做事」：这枚做的事是导出当前轨道。
    expect(trigger.getAttribute("data-tooltip")).toBe(enUS.pitch_bridge_tooltip_export_track);

    await act(async () => {
        trigger.click();
    });
    await act(async () => {
        await Promise.resolve();
    });

    expect(exported).toHaveBeenCalledTimes(1);
    expect(exported.mock.calls[0][0]).toEqual(["c1", "c2"]);
    expect(exported.mock.calls[0][1]).toEqual({ tolerant: true });
    // 左键不是「开菜单」：菜单是右键的东西。
    expect(document.body.querySelector(".hs-menu__item")).toBeNull();

    await openMenu();
    expect(document.body.querySelector(".hs-menu__item")).toBeTruthy();
    exported.mockRestore();
});

test("the auto-write switch sits at the bottom of the menu and flips the session state", async () => {
    setBootstrap(pluginCaps({ takePitchBridge: true }));
    // 落盘这一跳在测试里没有后端：只关心"有没有被调用"，不关心结果。
    const saved = vi.spyOn(settingsApi, "saveUiSettings").mockResolvedValue({ ok: true } as never);
    await mount();
    await openMenu();

    const labels = menuLabels();
    // 【为什么钉"在最后"】上面六项都会立刻改写东西；这一项只改一个开关。混在同一组里会让
    // 一次右键出现七件"看起来一样"的事，其中一件点了没有任何可见变化。
    expect(labels[labels.length - 1]).toBe(enUS.pitch_bridge_auto_write);
    expect(store.getState().session.reaperAutoTakePitchWrite).toBe(false);

    const entry = Array.from(
        document.body.querySelectorAll<HTMLButtonElement>("button.hs-menu__item"),
    ).find((el) => (el.textContent ?? "").startsWith(enUS.pitch_bridge_auto_write));
    expect(entry, "自动写入必须渲染成一个可点的菜单项").toBeTruthy();
    await act(async () => {
        entry!.click();
    });

    expect(store.getState().session.reaperAutoTakePitchWrite).toBe(true);
    expect(saved).toHaveBeenCalledTimes(1);
    expect(saved.mock.calls[0][0]).toMatchObject({ reaperAutoTakePitchWrite: true });
    saved.mockRestore();
});
