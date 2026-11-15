/**
 * 拖动方向按钮：**三态循环 + 右键菜单**（这枚按钮没有开关态）。
 *
 * 【要钉死什么】
 * 1. 左键依次循环本工具的全部方向（选择工具三步一圈、绘制工具两步一圈），与 `D` 键
 *    同义，且不弹菜单；
 * 1b. **高亮是派生的**：`仅 X` / `仅 Y` 亮（`aria-pressed`），`自由` 熄灭 —— 状态里没有第二个布尔；
 * 2. tooltip 固定是「功能名：方向名 (+ 快捷键)」，循环按钮的可访问状态就靠这个名字；
 * 3. 右键在按钮下方展开方向菜单，勾选态等于图标；
 * 4. **方向集随工具变化**：选择工具三项，绘制 / 直线 / 颤音只有两项（它们没有纵向约束）；
 * 5. 选方向 = 立即生效（菜单直接写入，不需要先循环过去）。
 */
// @vitest-environment jsdom
import { act, type ComponentProps } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { combineReducers, configureStore } from "@reduxjs/toolkit";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import sessionReducer, { setToolMode } from "../../../../features/session/sessionSlice";
import {
    resolveDragDirection,
    resolveDragDirectionTool,
} from "../../../../features/session/dragDirection";
import dockReducer from "../../../../features/dock/dockSlice";
import { I18nProvider, useI18n } from "../../../../i18n/I18nProvider";
import { useAppDispatch, useAppSelector } from "../../../../app/hooks";
import { DragDirectionButton } from "./DragDirectionButton";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const LOCALE_KEY = "hifishifter.locale";
let container: HTMLDivElement;
let root: Root;
let store: ReturnType<typeof buildStore>;
/** 快捷键绑定：默认给一个 `D`，个别用例换成"无"以验证气泡里的尾巴会消失。 */
let cycleKb: { key: string }[] = [{ key: "d" }];

function buildStore() {
    return configureStore({
        reducer: combineReducers({ session: sessionReducer, dock: dockReducer }),
        middleware: (getDefault) => getDefault({ serializableCheck: false, immutableCheck: false }),
    });
}

/** 组件吃的是面板模型切片；这里只喂它真正读的那几项（方向 / 作用域按模型同一条派生）。 */
function Harness() {
    const dispatch = useAppDispatch();
    const s = useAppSelector((state) => state.session);
    const { tf } = useI18n();

    const m = {
        dispatch,
        tf,
        activeDragDirection: resolveDragDirection(s),
        activeDragDirectionTool: resolveDragDirectionTool(s),
        cycleDragDirectionKb: cycleKb,
    } as unknown as ComponentProps<typeof DragDirectionButton>["m"];

    return <DragDirectionButton m={m} />;
}

beforeEach(() => {
    localStorage.setItem(LOCALE_KEY, "en-US");
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    cycleKb = [{ key: "d" }];
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

async function render(tool: "select" | "draw" = "select") {
    if (tool === "select") store.dispatch(setToolMode("select"));
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
    expect(found, "拖动方向按钮必须渲染出来").toBeTruthy();
    return found!;
}

function iconDirection(): string | null {
    return (
        container
            .querySelector("[data-hs-drag-direction-icon]")
            ?.getAttribute("data-hs-drag-direction-icon") ?? null
    );
}

async function click() {
    await act(async () => {
        button().dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
}

async function openMenu() {
    await act(async () => {
        button().dispatchEvent(
            new MouseEvent("contextmenu", { bubbles: true, clientX: 30, clientY: 30 }),
        );
    });
    const menu = document.body.querySelector('[data-hs-context-menu="1"]');
    expect(menu, "右键必须打开方向菜单").toBeTruthy();
    return menu as HTMLElement;
}

function menuItems(menu: HTMLElement): HTMLElement[] {
    return Array.from(menu.querySelectorAll<HTMLElement>("[role=menuitemcheckbox]"));
}

async function pick(menu: HTMLElement, label: string) {
    const item = menuItems(menu).find((entry) => entry.textContent?.includes(label));
    expect(item, `菜单里必须有「${label}」`).toBeTruthy();
    await act(async () => {
        item!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
}

test("出厂（选择工具 / 仅 Y）：图标画方向、按钮高亮、tooltip 带快捷键尾巴、带小三角", async () => {
    await render();
    expect(button().getAttribute("aria-label")).toBe("Drag Direction: Y Only (D)");
    expect(iconDirection()).toBe("y-only");
    // 受限取值要亮（自由才是熄灭的那一档）。
    expect(button().getAttribute("aria-pressed")).toBe("true");
    expect(button().querySelector("[data-hs-menu-caret]")).toBeTruthy();
});

test("自由：图标画自由、按钮熄灭（高亮只属于受限取值）", async () => {
    await render("draw");
    expect(button().getAttribute("aria-pressed")).toBeNull();
    expect(iconDirection()).toBe("free");
    expect(button().getAttribute("aria-label")).toBe("Drag Direction: Free (D)");
    // 没有"带斜杠的关"这种第三形态：按钮里只有方向图标 + 小三角。
    expect(container.querySelectorAll("svg").length).toBe(2);
});

test("★ 左键循环三态：仅 Y → 自由 → 仅 X → 仅 Y，图标与 tooltip 一路跟随", async () => {
    await render();
    await click();
    expect(store.getState().session.selectDragDirection).toBe("free");
    expect(iconDirection()).toBe("free");
    expect(button().getAttribute("aria-label")).toBe("Drag Direction: Free (D)");
    expect(button().getAttribute("aria-pressed")).toBeNull();

    await click();
    expect(store.getState().session.selectDragDirection).toBe("x-only");
    expect(iconDirection()).toBe("x-only");
    expect(button().getAttribute("aria-label")).toBe("Drag Direction: X Only (D)");
    expect(button().getAttribute("aria-pressed")).toBe("true");

    await click();
    expect(store.getState().session.selectDragDirection).toBe("y-only");
    expect(iconDirection()).toBe("y-only");

    // 左键从不弹菜单，也不改工具。
    expect(document.body.querySelector('[data-hs-context-menu="1"]')).toBeNull();
    expect(store.getState().session.toolMode).toBe("select");
});

test("绘制工具两态循环：自由 → 仅 X → 自由", async () => {
    await render("draw");
    await click();
    expect(store.getState().session.drawDragDirection).toBe("x-only");
    expect(iconDirection()).toBe("x-only");
    await click();
    expect(store.getState().session.drawDragDirection).toBe("free");
    expect(iconDirection()).toBe("free");
});

test("右键菜单（选择工具）：三项方向、当前项打勾、没有标题行、每项带图标", async () => {
    await render();
    const menu = await openMenu();
    expect(menu.querySelector(".hs-menu__label")).toBeNull();
    expect(menuItems(menu).map((item) => item.textContent)).toEqual(["Free", "X Only", "Y Only"]);
    expect(menuItems(menu)[2].getAttribute("aria-checked")).toBe("true");
    expect(menuItems(menu)[0].getAttribute("aria-checked")).toBe("false");
    expect(
        menuItems(menu).map(
            (item) =>
                item
                    .querySelector("[data-hs-drag-direction-icon]")
                    ?.getAttribute("data-hs-drag-direction-icon") ?? null,
        ),
    ).toEqual(["free", "x-only", "y-only"]);
});

test("右键菜单（绘制工具）：只有自由 / 仅 X —— 纵向约束对绘制没有意义", async () => {
    await render("draw");
    const menu = await openMenu();
    expect(menuItems(menu).map((item) => item.textContent)).toEqual(["Free", "X Only"]);
});

test("★ 选方向 = 立即生效：一步到位，图标与 tooltip 跟随、菜单关闭", async () => {
    await render();
    await pick(await openMenu(), "X Only");
    expect(store.getState().session.selectDragDirection).toBe("x-only");
    expect(iconDirection()).toBe("x-only");
    expect(button().getAttribute("aria-label")).toBe("Drag Direction: X Only (D)");
    expect(document.body.querySelector('[data-hs-context-menu="1"]')).toBeNull();

    // 选「自由」也是同一条路径：它就是第三种取值，不是"关"。
    await pick(await openMenu(), "Free");
    expect(store.getState().session.selectDragDirection).toBe("free");
    expect(iconDirection()).toBe("free");
});

test("快捷键未绑定时气泡尾巴消失（格式始终是「功能名：方向名」）", async () => {
    cycleKb = [{ key: "__none__" }];
    await render();
    expect(button().getAttribute("aria-label")).toBe("Drag Direction: Y Only");
});
