/**
 * 节拍器菜单里的音量条。
 *
 * 【要钉死什么】它是裸 `<input type="range" className="qt-range">` + 手写滚轮步进。
 * `qt-range` 是给裸 range 准备的类，`AppSlider` 明确取代了它 —— 后者内建滚轮步进、
 * 精细调整修饰键与非被动滚轮守卫，调用点不再需要自己写这些。
 *
 * 【为什么真的挂 ActionBar】音量条是内联在工具栏的菜单里的，没有可以单独挂载的
 * 组件。这里用最小 store（与 `app/store.ts` 同名的切片）把它整块挂起来，只验证
 * 菜单打开后那一条滑块的行为。
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

/** 渲染主工具栏。 */
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

/** 打开节拍器菜单（右键节拍器按钮）。 */
async function openMetronomeMenu() {
    await renderActionBar();
    const trigger = container.querySelector<HTMLElement>("[data-hs-context-menu][data-tooltip]");
    const buttons = Array.from(container.querySelectorAll<HTMLElement>("button"));
    // 节拍器按钮是那个带右键菜单标记的（见 ActionBar 的 metronome 按钮）。
    const metronome = buttons.find((b) => b.closest("[data-hs-context-menu]"));
    expect(metronome ?? trigger, "metronome trigger must exist").toBeTruthy();
    await act(async () => {
        (metronome ?? trigger)!.dispatchEvent(
            new MouseEvent("contextmenu", { bubbles: true, clientX: 40, clientY: 40 }),
        );
    });
}

test("toolbar separators use a whole-device-pixel hairline at 125% DPI", async () => {
    vi.spyOn(window, "devicePixelRatio", "get").mockReturnValue(1.25);
    await renderActionBar();

    const bar = container.querySelector<HTMLElement>(".min-h-qt-bar-main");
    expect(bar, "main toolbar must exist").toBeTruthy();
    // 0.8 CSS px × 1.25 = 恰好 1 个物理像素。若退回 1px CSS，在 125% 下会覆盖
    // 1.25 个物理像素并抗锯齿 —— 正是工具栏分隔线发虚、显粗的根因。
    expect(bar!.style.getPropertyValue("--hs-toolbar-hairline")).toBe("0.8px");
});

test("the metronome volume uses AppSlider, not a bare range input", async () => {
    await openMetronomeMenu();
    const menu = document.body.querySelector('[data-hs-context-menu="1"]');
    expect(menu, "metronome menu must be open").toBeTruthy();
    expect(menu!.querySelector("input[type=range]")).toBeNull();
    expect(menu!.querySelector(".hs-slider-box [role=slider]")).toBeTruthy();
});

/** 让帧合并提交器跑完（它用 rAF，jsdom 的 rAF 是 ~16ms 的定时器）。 */
async function nextFrame() {
    await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 40));
    });
}

test("wheel steps the metronome volume by 5 percent, and by 1 with the fine modifier", async () => {
    await openMetronomeMenu();
    const slider = document.body.querySelector<HTMLElement>(".hs-slider-box");
    expect(slider, "slider box must exist").toBeTruthy();
    store.dispatch({ type: "session/setMetronomeConfig", payload: { metronomeGain: 0.5 } });
    await act(async () => {});

    const wheel = (init: WheelEventInit) =>
        act(async () => {
            slider!.dispatchEvent(
                new WheelEvent("wheel", { deltaY: -100, bubbles: true, ...init }),
            );
        });

    await wheel({});
    await nextFrame();
    // `percent` 单位语义：粗调 5%。
    expect(store.getState().session.metronomeGain).toBeCloseTo(0.55, 6);

    // 按住精细调整修饰键（默认是主修饰键）→ 精调 1%。
    await wheel({ ctrlKey: true });
    await nextFrame();
    expect(store.getState().session.metronomeGain).toBeCloseTo(0.56, 6);
});

test("the metronome menu renders inside the theme scope (Radix tokens reach it)", async () => {
    await openMetronomeMenu();
    const menu = document.body.querySelector('[data-hs-context-menu="1"]');
    expect(menu, "metronome menu must be open").toBeTruthy();
    /*
     * Radix Themes 把整套组件令牌（`--accent-9`、`--gray-a5`…）定义在
     * `.radix-themes` 元素上。弹出面若挂到 `document.body`，滑块轨道与滑块头
     * 的背景色整条失效（计算成 transparent）—— 用户看到的就是"菜单里有一行空着、
     * 只剩 60% 读数"。这条断言钉住"菜单在主题作用域内"。
     */
    expect(menu!.closest(".radix-themes"), "菜单必须挂在 .radix-themes 内").not.toBeNull();
    expect(menu!.querySelector(".hs-slider-box"), "音量滑块必须渲染出来").toBeTruthy();
});
