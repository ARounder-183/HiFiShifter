/*
 * 移动抽屉「窗口」分区的清单派生。
 *
 * 【锁定的契约】
 * 1. **关掉的面板仍在清单里**（visible=false）—— 这正是"关掉之后再也打不开"的
 *    那个缺陷的回归点：清单一旦只列可见面板，用户就失去了恢复入口；
 * 2. 顺序与注册表一致（面板 order），不随开关状态跳动 —— 开关一个面板时它
 *    不能在列表里换位置，否则用户下一次点到的就是另一行；
 * 3. `excludeFromWindowMenu` 的面板不出现（与桌面「视图 → 窗口」同一份规则）。
 */

import { expect, test } from "vitest";

import type { DockLayout } from "../features/dock/dockTypes";
import { registerPanel, resetPanelRegistryForTests } from "../features/dock/panelRegistry";
import { buildMobileWindowRows, windowSummary } from "./mobileWindows";

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
    registerPanel({
        id: "gamma",
        titleKey: "appearance_title",
        component: () => null,
        defaultWidth: 300,
        defaultHeight: 200,
        order: 30,
        // 设置类面板：不进窗口清单（入口在菜单里）。
        excludeFromWindowMenu: true,
    });
}

/** 只有 alpha 停靠着的布局。 */
function layoutWithAlphaOnly(): DockLayout {
    return {
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
    };
}

test("关掉的面板仍在清单里，并标记为未显示", () => {
    registerFakes();
    const rows = buildMobileWindowRows(layoutWithAlphaOnly());
    expect(rows.map((row) => row.panelId)).toEqual(["alpha", "beta"]);
    expect(rows.find((row) => row.panelId === "alpha")?.visible).toBe(true);
    expect(rows.find((row) => row.panelId === "beta")?.visible).toBe(false);
});

test("设置类面板不进窗口清单（与桌面同一份规则）", () => {
    registerFakes();
    const rows = buildMobileWindowRows(layoutWithAlphaOnly());
    expect(rows.some((row) => row.panelId === "gamma")).toBe(false);
});

test("摘要给出已显示 / 全部，没有面板时为空串", () => {
    registerFakes();
    expect(windowSummary(buildMobileWindowRows(layoutWithAlphaOnly()))).toBe("1/2");
    expect(windowSummary([])).toBe("");
});
