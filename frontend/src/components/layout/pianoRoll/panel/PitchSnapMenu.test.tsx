/**
 * 音高吸附按钮：**开关 × 模式**（与自动滚屏 / 波纹编辑同一套交互）。
 *
 * 【要钉死什么】
 * 1. 图标画**模式**、不画开关：关闭时仍然是"半音"或"音阶"的图形，没有第三种
 *    "带斜杠的关"形态；
 * 2. tooltip 固定是「功能名：模式名」，开关状态不进气泡（它由 `aria-pressed` 与
 *    按钮实心态表达）；
 * 3. 左键只做开关，右键在按钮下方展开模式菜单；
 * 4. 选模式 = 立即开启（与相邻两个按钮同一条策略）。
 */
// @vitest-environment jsdom
import { act, useRef, useState, type ComponentProps } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { combineReducers, configureStore } from "@reduxjs/toolkit";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import sessionReducer from "../../../../features/session/sessionSlice";
import dockReducer from "../../../../features/dock/dockSlice";
import { I18nProvider, useI18n } from "../../../../i18n/I18nProvider";
import { useAppDispatch, useAppSelector } from "../../../../app/hooks";
import { PitchSnapMenu } from "./PitchSnapMenu";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const LOCALE_KEY = "hifishifter.locale";
let container: HTMLDivElement;
let root: Root;
let store: ReturnType<typeof buildStore>;
let settingsOpened = 0;

function buildStore() {
    return configureStore({
        reducer: combineReducers({ session: sessionReducer, dock: dockReducer }),
        middleware: (getDefault) => getDefault({ serializableCheck: false, immutableCheck: false }),
    });
}

/** 组件吃的是面板模型切片；这里只喂它真正读的那几项。 */
function Harness() {
    const dispatch = useAppDispatch();
    const s = useAppSelector((state) => state.session);
    const { t, tf } = useI18n();
    const [menuOpen, setMenuOpen] = useState(false);
    const anchorRef = useRef<HTMLDivElement | null>(null);
    const menuRef = useRef<HTMLDivElement | null>(null);

    const m = {
        dispatch,
        s,
        t,
        tf,
        // 拖动中"临时反转吸附"由模型算；本用例只验证不反转时的形状。
        effectivePitchSnapVisual: s.pitchSnapEnabled,
        pitchSnapMenuAnchorRef: anchorRef,
        pitchSnapMenuOpen: menuOpen,
        pitchSnapMenuRef: menuRef,
        setPitchSnapMenuOpen: setMenuOpen,
        setPitchSnapOpen: () => {
            settingsOpened += 1;
        },
        pitchSnapSemitoneIcon: <svg data-hs-test-icon="semitone" />,
        pitchSnapScaleIcon: <svg data-hs-test-icon="scale" />,
    } as unknown as ComponentProps<typeof PitchSnapMenu>["m"];

    return <PitchSnapMenu m={m} />;
}

beforeEach(() => {
    localStorage.setItem(LOCALE_KEY, "en-US");
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    settingsOpened = 0;
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
    vi.restoreAllMocks();
});

async function render() {
    await act(async () => {
        root.render(
            <Provider store={store}>
                <I18nProvider>
                    <Harness />
                </I18nProvider>
            </Provider>,
        );
    });
}

function button(): HTMLButtonElement {
    const found = container.querySelector<HTMLButtonElement>("button");
    expect(found, "音高吸附按钮必须渲染出来").toBeTruthy();
    return found!;
}

async function openMenu() {
    await act(async () => {
        button().dispatchEvent(
            new MouseEvent("contextmenu", { bubbles: true, clientX: 30, clientY: 30 }),
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

test("出厂：画「半音」、未激活、tooltip 是「功能名：模式名」、带小三角", async () => {
    await render();
    expect(button().getAttribute("aria-label")).toBe("Pitch Snap: Semitone");
    expect(button().getAttribute("aria-pressed")).toBeNull();
    expect(button().querySelector('[data-hs-test-icon="semitone"]')).toBeTruthy();
    // 右下角小三角：提示"右键还有模式可选"。
    expect(button().querySelector("[data-hs-menu-caret]")).toBeTruthy();
});

test("关闭状态下图标与 tooltip 仍只表达模式（没有第三种「带斜杠」形态）", async () => {
    await render();
    const icons = Array.from(container.querySelectorAll("[data-hs-test-icon]")).map((node) =>
        node.getAttribute("data-hs-test-icon"),
    );
    expect(icons).toEqual(["semitone"]);
    // 斜杠叠加层已随三态图标一起移除：按钮里只有一枚图形。
    expect(container.querySelectorAll("svg").length).toBe(2); // 图标 + 小三角
});

test("左键只做开关：不弹菜单、不改模式、tooltip 不变", async () => {
    await render();
    await act(async () => {
        button().dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(store.getState().session.pitchSnapEnabled).toBe(true);
    expect(store.getState().session.pitchSnapUnit).toBe("semitone");
    expect(button().getAttribute("aria-label")).toBe("Pitch Snap: Semitone");
    expect(button().getAttribute("aria-pressed")).toBe("true");
    expect(document.body.querySelector('[data-hs-context-menu="1"]')).toBeNull();
});

test("右键菜单：两个模式项 + 设置项，当前项打勾，没有标题行", async () => {
    await render();
    const menu = await openMenu();
    expect(menu.querySelector(".hs-menu__label")).toBeNull();
    const items = Array.from(menu.querySelectorAll<HTMLElement>("[role=menuitemcheckbox]"));
    expect(items.map((item) => item.textContent)).toEqual(["Semitone", "Scale"]);
    expect(menuItem(menu, "Semitone").getAttribute("aria-checked")).toBe("true");
    expect(menuItem(menu, "Scale").getAttribute("aria-checked")).toBe("false");
    expect(menu.textContent).toContain("Pitch Snap Settings...");
});

test("★ 选模式 = 立即生效：写入模式、同时开启、图标与 tooltip 随之切换", async () => {
    await render();
    const menu = await openMenu();

    await act(async () => {
        menuItem(menu, "Scale").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    const session = store.getState().session;
    expect(session.pitchSnapUnit).toBe("scale");
    expect(session.pitchSnapEnabled).toBe(true);
    expect(button().querySelector('[data-hs-test-icon="scale"]')).toBeTruthy();
    expect(button().getAttribute("aria-label")).toBe("Pitch Snap: Scale");
    expect(button().getAttribute("aria-pressed")).toBe("true");
});

test("菜单里的「设置…」打开对话框并关闭菜单", async () => {
    await render();
    const menu = await openMenu();
    const settings = Array.from(menu.querySelectorAll<HTMLElement>("[role=menuitem]")).find(
        (item) => item.textContent?.includes("Pitch Snap Settings"),
    );
    expect(settings).toBeTruthy();

    await act(async () => {
        settings!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(settingsOpened).toBe(1);
    expect(document.body.querySelector('[data-hs-context-menu="1"]')).toBeNull();
});
