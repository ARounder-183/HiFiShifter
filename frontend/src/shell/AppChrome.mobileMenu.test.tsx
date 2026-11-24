// @vitest-environment jsdom
/*
 * 移动外壳的「菜单」sheet：**桌面菜单栏本体搬到移动端**这条决定的回归测试。
 *
 * 【为什么必须真实挂载】这里锁定的三件事都只在 DOM 上成立：
 * 1. sheet 关闭时菜单节点**仍然挂载**（display:none）—— 卸载它，`MenuBar` 里那批
 *    对话框（导入、导出、转调、诊断…）就会退回到"点了没反应"；
 * 2. 打开后菜单可见、关闭后不可见，且**只有一个实例**（渲染两份会让同一个对话框
 *    被打开两次）；
 * 3. 点完一项收起 sheet，但点子菜单触发器不收起 —— 后者收起了就再也点不到下一级。
 */

import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, beforeEach, expect, test } from "vitest";

import dockReducer from "../features/dock/dockSlice";
import { I18nProvider } from "../i18n/I18nProvider";
import {
    DESKTOP_FALLBACK_PROFILE,
    setSurfaceProfile,
    resetSurfaceProfileForTests,
} from "../services/surfaceProfileStore";
import { AppChrome } from "./AppChrome";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function createStore() {
    return configureStore({ reducer: { dock: dockReducer } });
}

/** 菜单内容用真实角色标记，形状与 Radix 的输出一致（见 `mobileMenuSheet`）。 */
function MenuStub() {
    return (
        <div data-testid="menu-stub">
            <button type="button" role="menuitem" data-menu-plain="1">
                导出音频…
            </button>
            <button type="button" role="menuitem" aria-haspopup="menu" data-menu-sub="1">
                窗口
            </button>
        </div>
    );
}

interface Mounted {
    host: HTMLElement;
    unmount: () => Promise<void>;
    sheet: () => HTMLElement | null;
    click: (element: Element) => Promise<void>;
}

async function mount(): Promise<Mounted> {
    const store = createStore();
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => {
        root.render(
            <Provider store={store}>
                <I18nProvider>
                    <AppChrome
                        mobileMenu={<MenuStub />}
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
            </Provider>,
        );
    });
    return {
        host,
        sheet: () => host.querySelector<HTMLElement>("[data-mobile-menu-sheet]"),
        click: async (element: Element) => {
            await act(async () => {
                element.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
                // sheet 的收起安排在事件之后的宏任务里（见 MobileMenuSheet）。
                await new Promise((resolve) => setTimeout(resolve, 0));
            });
        },
        unmount: async () => {
            await act(async () => root.unmount());
            host.remove();
        },
    };
}

/** 顶栏里的按钮。用结构定位（根容器的第一个子元素就是顶栏），不依赖文案。 */
function topBarButtons(host: HTMLElement): HTMLButtonElement[] {
    const root = host.querySelector<HTMLElement>('[data-form="mobile"]');
    const bar = root?.firstElementChild;
    if (!bar) throw new Error("顶栏未渲染");
    return Array.from(bar.querySelectorAll("button"));
}

const mounted: Mounted[] = [];
beforeEach(() => {
    setSurfaceProfile({ ...DESKTOP_FALLBACK_PROFILE, form: "mobile" });
});
afterEach(async () => {
    while (mounted.length) await mounted.pop()?.unmount();
    resetSurfaceProfileForTests();
});

async function mountTracked(): Promise<Mounted> {
    const m = await mount();
    mounted.push(m);
    return m;
}

test("sheet 关闭时菜单仍然挂载，只是不可见", async () => {
    const m = await mountTracked();

    const sheet = m.sheet();
    expect(sheet).not.toBeNull();
    expect(sheet?.getAttribute("data-mobile-menu-sheet")).toBe("closed");
    // 挂载但隐藏：菜单里的对话框宿主必须常驻（见文件头第 1 条）。
    expect(m.host.querySelector('[data-testid="menu-stub"]')).not.toBeNull();
    expect(sheet?.className).toContain("hidden");
});

test("点顶栏的菜单按钮打开 sheet，且菜单只有一个实例", async () => {
    const m = await mountTracked();

    await m.click(topBarButtons(m.host)[0]);

    expect(m.sheet()?.getAttribute("data-mobile-menu-sheet")).toBe("open");
    expect(m.sheet()?.className).not.toContain("hidden");
    // 同一份菜单节点只渲染一次（两份会让同一个对话框被打开两次）。
    expect(m.host.querySelectorAll('[data-testid="menu-stub"]')).toHaveLength(1);
});

test("点普通菜单项收起 sheet", async () => {
    const m = await mountTracked();
    await m.click(topBarButtons(m.host)[0]);

    const item = m.host.querySelector('[data-menu-plain="1"]');
    if (!item) throw new Error("菜单项未渲染");
    await m.click(item);

    expect(m.sheet()?.getAttribute("data-mobile-menu-sheet")).toBe("closed");
});

test("点子菜单触发器不收起 sheet（否则下一级点不到）", async () => {
    const m = await mountTracked();
    await m.click(topBarButtons(m.host)[0]);

    const sub = m.host.querySelector('[data-menu-sub="1"]');
    if (!sub) throw new Error("子菜单触发器未渲染");
    await m.click(sub);

    expect(m.sheet()?.getAttribute("data-mobile-menu-sheet")).toBe("open");
});
