// @vitest-environment jsdom
/*
 * 单标签 compact 抓手的回归测试。
 *
 * 这里锁定三条契约：
 * 1. 只有左上角的抓手能起拖，整条 6px 安全条不再可拖 —— 用户点面板顶部边缘
 *    时不应误触拖拽；
 * 2. 抓手把指针在自身内的偏移交给 `beginTabDrag`，浮窗锚点不会沿用标签的
 *    48/12（否则浮窗会跑到指针左侧，甚至屏幕外）；
 * 3. 提示说明的是"浮动"，不是单标签下根本不存在的"重排"。
 */
import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider, useSelector } from "react-redux";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import dockReducer, { setDockLayout } from "../../features/dock/dockSlice";
import { registerPanel, resetPanelRegistryForTests } from "../../features/dock/panelRegistry";
import type { DockTabsetNode } from "../../features/dock/dockTypes";
import { I18nProvider } from "../../i18n/I18nProvider";
import { DockTabBar } from "./DockTabBar";
import { beginTabDrag } from "./dockDragController";

vi.mock("./dockDragController", async (importOriginal) => {
    const actual = await importOriginal<typeof import("./dockDragController")>();
    return { ...actual, beginTabDrag: vi.fn() };
});

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const GRIP = ".hs-dock-grip";
const HANDLE = "[data-dock-grip-handle]";

type TestState = { dock: ReturnType<typeof dockReducer> };

function createStore() {
    const store = configureStore({
        reducer: { dock: dockReducer },
        preloadedState: undefined,
    });
    store.dispatch(
        setDockLayout({
            schema: 2,
            roots: {
                main: {
                    t: "tabset",
                    id: "ts1",
                    tabs: ["alpha"],
                    active: "alpha",
                    collapsed: false,
                },
            },
            forms: {
                alpha: { id: "alpha", panelId: "alpha", float: null, floating: false },
            },
            order: ["alpha"],
            floatOrder: [],
            gutters: { timelineTrackHeaderPx: 256 },
            tabPosition: "top",
        }),
    );
    return store;
}

function Harness() {
    const node = useSelector((s: TestState) => s.dock.layout.roots.main) as DockTabsetNode;
    return <DockTabBar node={node} onToggleFloat={() => {}} compact tabPosition="top" />;
}

interface Mounted {
    host: HTMLElement;
    store: ReturnType<typeof createStore>;
    unmount: () => Promise<void>;
}

async function mount(): Promise<Mounted> {
    resetPanelRegistryForTests();
    registerPanel({
        id: "alpha",
        titleKey: "panel_timeline",
        component: () => null,
        defaultWidth: 300,
        defaultHeight: 200,
        order: 10,
    });
    const store = createStore();
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
    return {
        host,
        store,
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
beforeEach(() => {
    vi.mocked(beginTabDrag).mockClear();
});

async function mountAndTrack(): Promise<Mounted> {
    const m = await mount();
    mounted.push(m);
    return m;
}

/** jsdom 没有布局：给抓手一个真实的 24×6 矩形，指针偏移才有几何可算。 */
function stubHandleRect(handle: HTMLElement): void {
    handle.getBoundingClientRect = () =>
        ({
            left: 4,
            top: 0,
            right: 28,
            bottom: 6,
            width: 24,
            height: 6,
            x: 4,
            y: 0,
            toJSON: () => ({}),
        }) as DOMRect;
}

function pointerDown(element: HTMLElement, clientX: number, clientY: number): void {
    element.dispatchEvent(
        new MouseEvent("pointerdown", {
            bubbles: true,
            cancelable: true,
            button: 0,
            clientX,
            clientY,
        }),
    );
}

test("单标签 compact 形态只暴露左上角抓手，不再整行可拖", async () => {
    const m = await mountAndTrack();
    const strip = m.host.querySelector<HTMLElement>(GRIP);
    const handle = m.host.querySelector<HTMLElement>(HANDLE);
    if (!strip || !handle) throw new Error("compact 抓手未渲染");

    expect(strip.contains(handle)).toBe(true);
    expect(strip.getAttribute("data-dock-tabbar")).toBe("ts1");
    expect(handle.getAttribute("data-dock-tab")).toBe("alpha");
    // 6 个点完整落在 6px 条内；此前 15px 图标只露出中间两个点。
    expect(handle.querySelectorAll("circle")).toHaveLength(6);
    // compact 形态不应再渲染完整标签条。
    expect(m.host.querySelector('[role="tablist"]')).toBeNull();

    const tooltip = handle.getAttribute("data-tooltip") ?? "";
    expect(tooltip).not.toBe("");
    expect(tooltip).not.toMatch(/rearrange|重排/i);
});

test("按在安全条上不会起拖", async () => {
    const m = await mountAndTrack();
    const strip = m.host.querySelector<HTMLElement>(GRIP);
    if (!strip) throw new Error("compact 安全条未渲染");

    await act(async () => {
        pointerDown(strip, 100, 2);
    });

    expect(beginTabDrag).not.toHaveBeenCalled();
});

test("按在左上角抓手上会起拖，并把指针偏移交给浮窗锚点", async () => {
    const m = await mountAndTrack();
    const handle = m.host.querySelector<HTMLElement>(HANDLE);
    if (!handle) throw new Error("compact 抓手未渲染");
    stubHandleRect(handle);

    await act(async () => {
        pointerDown(handle, 12, 3);
    });

    expect(beginTabDrag).toHaveBeenCalledTimes(1);
    const args = vi.mocked(beginTabDrag).mock.calls[0][1];
    expect(args.offset).toEqual({ x: 8, y: 3 });
});
