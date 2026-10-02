// @vitest-environment jsdom
/*
 * 停靠标签条**键盘模型**的回归测试。
 *
 * 【为什么必须有】标签条一度声明了 `role="tablist"` / `role="tab"`，却没有任何
 * 标签可聚焦，也没有方向键：屏幕阅读器会播报一个永远进不去的标签组，键盘用户
 * 完全切不了标签。**声明 ARIA 角色却不实现它的键盘契约，比不声明更糟** ——
 * 辅助技术会据此调整交互方式，然后撞上一堵墙。
 *
 * 这里锁定的契约：
 * 1. roving tabIndex：整条标签条只占一个 Tab 停留点（活动标签），关闭按钮同理；
 * 2. 方向键在标签间移动并**即时激活**（面板都是本地渲染，切换无代价）；
 * 3. `Home` / `End` 到两端，且不越界、不循环；
 * 4. `Delete` / `Backspace` 关闭当前标签（与 IDE 一致）。
 *
 * 必须真实挂载：`tabIndex` 与焦点移动都是 DOM 事实，源码文本断言看不出
 * "方向键改了 store 但焦点没跟过去"这类半吊子实现。
 */

import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider, useSelector } from "react-redux";
import { afterEach, expect, test } from "vitest";

import dockReducer, { setDockLayout } from "../../features/dock/dockSlice";
import { registerPanel, resetPanelRegistryForTests } from "../../features/dock/panelRegistry";
import type { DockTabsetNode } from "../../features/dock/dockTypes";
import type { MessageKey } from "../../i18n/messages";
import { I18nProvider } from "../../i18n/I18nProvider";
import { DockTabBar } from "./DockTabBar";

// React 19 要求显式声明这是 act() 环境，否则每次 act 都会打印一条警告。
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const TAB_IDS = ["alpha", "beta", "gamma"] as const;

/**
 * 标题键必须取自真实词典：`titleKey` 的类型是 `MessageKey`，用假键（如
 * `panel_alpha`）编译期就会失败 —— 这正是它该有的约束（面板标题曾经因为
 * 一个不存在的键把 `notebook` 直接渲染给了用户）。
 */
const TITLE_KEYS: Record<(typeof TAB_IDS)[number], MessageKey> = {
    alpha: "panel_timeline",
    beta: "panel_editor",
    gamma: "common_notebook",
};

function registerFakes(): void {
    for (const id of TAB_IDS) {
        registerPanel({
            id,
            titleKey: TITLE_KEYS[id],
            component: () => null,
            defaultWidth: 300,
            defaultHeight: 200,
            order: 10,
        });
    }
}

type TestState = { dock: ReturnType<typeof dockReducer> };

function createTestStore() {
    return configureStore({
        reducer: { dock: dockReducer },
        preloadedState: undefined,
    });
}

/** 把三个标签塞进同一个标签组，活动标签是第一个。 */
function seedLayout(store: ReturnType<typeof createTestStore>): void {
    store.dispatch(
        setDockLayout({
            schema: 2,
            roots: {
                main: {
                    t: "tabset",
                    id: "ts1",
                    tabs: [...TAB_IDS],
                    active: TAB_IDS[0],
                    collapsed: false,
                },
            },
            forms: Object.fromEntries(
                TAB_IDS.map((id) => [id, { id, panelId: id, float: null, floating: false }]),
            ),
            order: [...TAB_IDS],
            floatOrder: [],
            gutters: { timelineTrackHeaderPx: 256 },
            tabPosition: "top",
        }),
    );
}

/**
 * 真实用法里标签组节点来自 store，方向键切换活动标签后由父组件重新下发。
 * 测试里同样从 store 取，避免把"props 没更新"这种真实缺陷测没了。
 */
function Harness() {
    const node = useSelector((s: TestState) => s.dock.layout.roots.main) as DockTabsetNode;
    return <DockTabBar node={node} onToggleFloat={() => {}} compact={false} tabPosition="top" />;
}

interface Mounted {
    host: HTMLElement;
    store: ReturnType<typeof createTestStore>;
    /** 按 `data-dock-tab` 取标签元素。 */
    tab: (formId: string) => HTMLElement;
    tabs: () => HTMLElement[];
    /** 在当前活动标签上按下一个键。 */
    press: (key: string, target?: HTMLElement) => Promise<void>;
    unmount: () => Promise<void>;
}

async function mountTabBar(): Promise<Mounted> {
    resetPanelRegistryForTests();
    registerFakes();
    const store = createTestStore();
    seedLayout(store);

    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => {
        root.render(
            <Provider store={store}>
                <I18nProvider>
                    <Harness />
                </I18nProvider>
            </Provider>,
        );
    });

    const tab = (formId: string) => {
        const found = host.querySelector<HTMLElement>(`[data-dock-tab="${formId}"]`);
        if (!found) throw new Error(`标签 ${formId} 未渲染`);
        return found;
    };
    const tabs = () => Array.from(host.querySelectorAll<HTMLElement>('[role="tab"]'));
    const press = async (key: string, target?: HTMLElement) => {
        const element = target ?? (document.activeElement as HTMLElement | null);
        if (!element) throw new Error("没有可派发按键的目标元素");
        await act(async () => {
            element.dispatchEvent(
                new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }),
            );
        });
    };

    return {
        host,
        store,
        tab,
        tabs,
        press,
        unmount: async () => {
            await act(async () => root.unmount());
            host.remove();
        },
    };
}

const mounted: Mounted[] = [];
afterEach(async () => {
    while (mounted.length) await mounted.pop()?.unmount();
});

async function mount(): Promise<Mounted> {
    const m = await mountTabBar();
    mounted.push(m);
    return m;
}

function activeTabId(store: ReturnType<typeof createTestStore>): string {
    const tree = store.getState().dock.layout.roots.main as DockTabsetNode;
    return tree.active;
}

function tabIds(store: ReturnType<typeof createTestStore>): string[] {
    return (store.getState().dock.layout.roots.main as DockTabsetNode).tabs;
}

test("标签组声明为水平 tablist，且只有活动标签在 Tab 停留点上", async () => {
    const m = await mount();

    const list = m.host.querySelector('[role="tablist"]');
    expect(list?.getAttribute("aria-orientation")).toBe("horizontal");

    expect(m.tabs()).toHaveLength(3);
    for (const id of TAB_IDS) {
        const element = m.tab(id);
        expect(element.getAttribute("role")).toBe("tab");
        expect(element.getAttribute("aria-selected")).toBe(id === TAB_IDS[0] ? "true" : "false");
        // roving tabIndex：整条标签条只占一个停留点。
        expect(element.tabIndex).toBe(id === TAB_IDS[0] ? 0 : -1);
    }
});

test("关闭按钮：活动标签的进 Tab 停留点，其余不进", async () => {
    const m = await mount();

    for (const id of TAB_IDS) {
        const close = m.tab(id).querySelector<HTMLElement>("[data-dock-tab-close]");
        expect(close?.getAttribute("role")).toBe("button");
        // 关闭按钮必须有可访问名称，否则屏幕阅读器只会读出一个无名按钮。
        expect(close?.getAttribute("aria-label")).toBeTruthy();
        expect(close?.tabIndex).toBe(id === TAB_IDS[0] ? 0 : -1);
    }
});

test("方向键在标签间移动焦点并即时激活", async () => {
    const m = await mount();

    await m.press("ArrowRight", m.tab("alpha"));
    expect(activeTabId(m.store)).toBe("beta");
    expect(document.activeElement).toBe(m.tab("beta"));

    await m.press("ArrowRight", m.tab("beta"));
    expect(activeTabId(m.store)).toBe("gamma");
    expect(document.activeElement).toBe(m.tab("gamma"));

    await m.press("ArrowLeft", m.tab("gamma"));
    expect(activeTabId(m.store)).toBe("beta");
    expect(document.activeElement).toBe(m.tab("beta"));
});

test("Home / End 到两端，方向键不越界也不循环", async () => {
    const m = await mount();

    await m.press("End", m.tab("alpha"));
    expect(activeTabId(m.store)).toBe("gamma");

    // 已在最后一个：再按右键不动，也不绕回第一个（循环会让"到边界"失去反馈）。
    await m.press("ArrowRight", m.tab("gamma"));
    expect(activeTabId(m.store)).toBe("gamma");

    await m.press("Home", m.tab("gamma"));
    expect(activeTabId(m.store)).toBe("alpha");

    await m.press("ArrowLeft", m.tab("alpha"));
    expect(activeTabId(m.store)).toBe("alpha");
});

test("Delete 关闭当前标签，焦点顺序随之收敛", async () => {
    const m = await mount();

    await m.press("Delete", m.tab("alpha"));

    expect(tabIds(m.store)).toEqual(["beta", "gamma"]);
    expect(m.tabs()).toHaveLength(2);
    // 关掉活动标签后，新的活动标签必须接过那个唯一的 Tab 停留点。
    expect(m.tab(activeTabId(m.store)).tabIndex).toBe(0);
});

test("关闭按钮可用 Enter / Space 触发（role=button 的键盘等价操作）", async () => {
    const m = await mount();

    const close = m.tab("alpha").querySelector<HTMLElement>("[data-dock-tab-close]");
    if (!close) throw new Error("关闭按钮未渲染");
    await act(async () => {
        close.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    });

    expect(tabIds(m.store)).toEqual(["beta", "gamma"]);
});
