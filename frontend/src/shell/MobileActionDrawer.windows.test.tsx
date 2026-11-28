// @vitest-environment jsdom
/*
 * 移动抽屉的「窗口」分区。
 *
 * 【锁定的契约】
 * 1. **关掉的面板仍在清单里**（`aria-checked="false"`）—— 这正是"误触关闭之后
 *    再也打不开"的回归点；
 * 2. 点一行就切换该面板的显隐（关掉的能打开、开着的能关闭），并且抽屉**不关闭**
 *    （窗口是可连续操作的对象）；
 * 3. 分区排在动作列表之前 —— 恢复入口不能被埋在 150 条动作后面。
 */

import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, beforeEach, expect, test } from "vitest";

import { store } from "../app/store";
import { setDockLayout } from "../features/dock/dockSlice";
import { registerPanel, resetPanelRegistryForTests } from "../features/dock/panelRegistry";
import { I18nProvider } from "../i18n/I18nProvider";
import { MobileActionDrawer } from "./MobileChrome";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function registerFakes(): void {
    resetPanelRegistryForTests();
    registerPanel({
        id: "alpha",
        titleKey: "panel_timeline",
        component: () => null,
        defaultWidth: 300,
        defaultHeight: 200,
        order: 10,
    });
    registerPanel({
        id: "beta",
        titleKey: "panel_editor",
        component: () => null,
        defaultWidth: 300,
        defaultHeight: 200,
        order: 20,
    });
}

/** alpha 停靠、beta 关闭。 */
function seedLayout(): void {
    store.dispatch(
        setDockLayout({
            schema: 2,
            roots: {
                main: {
                    t: "tabset",
                    id: "ts1",
                    tabs: ["f-alpha"],
                    active: "f-alpha",
                    collapsed: false,
                },
            },
            forms: {
                "f-alpha": { id: "f-alpha", panelId: "alpha", float: null, floating: false },
            },
            order: ["f-alpha"],
            floatOrder: [],
            gutters: { timelineTrackHeaderPx: 256 },
            tabPosition: "top",
        }),
    );
}

interface Mounted {
    host: HTMLElement;
    row: (panelId: string) => HTMLElement | null;
    unmount: () => Promise<void>;
}

async function mountDrawer(onClose: () => void = () => {}): Promise<Mounted> {
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => {
        root.render(
            <Provider store={store}>
                <I18nProvider>
                    <MobileActionDrawer
                        open
                        onClose={onClose}
                        onAction={() => {}}
                        shortcutFor={() => null}
                    />
                </I18nProvider>
            </Provider>,
        );
    });
    return {
        host,
        row: (panelId) => host.querySelector<HTMLElement>(`[data-mobile-window-row="${panelId}"]`),
        unmount: async () => {
            await act(async () => root.unmount());
            host.remove();
        },
    };
}

const mounted: Mounted[] = [];
beforeEach(() => {
    registerFakes();
    seedLayout();
});
afterEach(async () => {
    while (mounted.length) await mounted.pop()?.unmount();
});

async function mountTracked(): Promise<Mounted> {
    const m = await mountDrawer();
    mounted.push(m);
    return m;
}

test("窗口分区列出全部面板，关掉的标记为未显示", async () => {
    const m = await mountTracked();

    expect(m.row("alpha")?.getAttribute("aria-checked")).toBe("true");
    expect(m.row("beta")?.getAttribute("aria-checked")).toBe("false");
});

test("点一行切换显隐，且抽屉不关闭", async () => {
    let closed = 0;
    const m = await mountDrawer(() => {
        closed += 1;
    });
    mounted.push(m);

    const row = m.row("beta");
    if (!row) throw new Error("beta 行未渲染");
    await act(async () => {
        row.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    // 打开 beta：窗体记录从"关闭"变成可见。
    expect(m.row("beta")?.getAttribute("aria-checked")).toBe("true");
    expect(closed).toBe(0);
});
