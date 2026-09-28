/*
 * 「视图 → 窗口」菜单的面板清单。
 *
 * 【要钉死什么】外观设置复用同一套停靠机制（浮窗、可拖动、可关闭），但它**不该**
 * 出现在「窗口」子菜单里：那个菜单列的是用户日常切换的工作面板，把低频的设置入口
 * 混进去只会稀释常用项。排除是**注册表声明**（`excludeFromWindowMenu`），不是菜单
 * 里的硬编码名单 —— 后者会让下一个设置类面板又被漏掉。
 *
 * 同时钉住反向事实：被排除的面板**仍然可以被程序化打开**（`openPanel`），
 * 否则"不进菜单"就变成了"打不开"。
 */
import { expect, test } from "vitest";

import { registerBuiltinPanels } from "../../components/dock/registerBuiltinPanels";
import { listPanelEntriesFromLayout } from "./dockApi";
import { listPanels, registerPanel, resetPanelRegistryForTests } from "./panelRegistry";
import { createDefaultDockLayout, ensureRegisteredPanels, openPanelInLayout } from "./dockSchema";
import { isFormVisible } from "./dockTree";

const noop = () => null;

function registerFakes(): void {
    registerPanel({
        id: "ordinary",
        titleKey: "panel_timeline",
        component: noop,
        defaultWidth: 400,
        defaultHeight: 300,
        order: 10,
    });
    registerPanel({
        id: "settingsLike",
        titleKey: "menu_appearance_settings",
        component: noop,
        defaultWidth: 900,
        defaultHeight: 640,
        openAsFloating: { width: 900, height: 640, anchor: "center" },
        excludeFromWindowMenu: true,
        dockable: false,
        order: 90,
    });
}

test("声明了 excludeFromWindowMenu 的面板不进窗口菜单", () => {
    resetPanelRegistryForTests();
    registerFakes();
    const layout = ensureRegisteredPanels(createDefaultDockLayout());

    const ids = listPanelEntriesFromLayout(layout).map((entry) => entry.panelId);
    expect(ids).toContain("ordinary");
    expect(ids).not.toContain("settingsLike");
});

test("被排除的面板仍可被程序化打开，且以居中浮窗出现", () => {
    resetPanelRegistryForTests();
    registerFakes();
    const base = ensureRegisteredPanels(createDefaultDockLayout());

    const opened = openPanelInLayout(base, "settingsLike");
    expect(isFormVisible(opened, "settingsLike")).toBe(true);
    expect(opened.forms.settingsLike?.floating).toBe(true);
    expect(opened.forms.settingsLike?.float?.anchor).toBe("center");
});

test("外观设置的三个声明：居中浮出、不可停靠、不进窗口菜单", () => {
    /*
     * 这三条是**用户可见的承诺**，各自有一个消费者：
     * - `anchor: "center"` → `resolveFloatRect`（该函数有独立测试）
     * - `dockable: false`  → `computeMovePatch` 跳过落点解析（见下）
     * - `excludeFromWindowMenu` → `listPanelEntriesFromLayout`（上面已测）
     *
     * 【未覆盖的部分】"拖拽时确实不产生落点"这条只在 `computeMovePatch` 里实现，
     * 而完整的拖拽会话需要指针捕获与真实停靠区，单测里起不来 —— 目前靠代码审查，
     * 没有自动化回归。改那一行时要留意这条注释。
     */
    resetPanelRegistryForTests();
    registerBuiltinPanels();

    const panel = listPanels().find((candidate) => candidate.id === "appearance");
    expect(panel, "外观设置面板未注册").toBeTruthy();
    expect(panel?.openAsFloating?.anchor).toBe("center");
    expect(panel?.dockable).toBe(false);
    expect(panel?.excludeFromWindowMenu).toBe(true);
    // 它不该出现在「窗口」菜单里，但**必须**能打开 —— 入口在「视图 → 外观设置」
    expect(panel?.singleton).toBe(true);
});

test("默认注册表里恰好只有外观设置被排除", () => {
    /*
     * 这条是"排除清单不能悄悄扩大"的守卫：每加一个被排除的面板都必须改这个测试，
     * 从而在 code review 里被看见。
     */
    resetPanelRegistryForTests();
    registerBuiltinPanels();

    const excluded = listPanels()
        .filter((panel) => panel.excludeFromWindowMenu)
        .map((panel) => panel.id)
        .sort();
    expect(excluded).toEqual(["appearance"]);
});
