/*
 * 浮窗标题栏动作矩阵的回归测试。
 *
 * 【为什么必须有】「重新停靠回主区域」按钮曾对 `dockable: false` 的外观设置
 * 面板照常渲染 —— 拖拽层辛辛苦苦抑制掉的停靠通道，被一枚按钮整个绕过。可见性
 * 收敛进 `floatTitleBarActions` 之后，"声明 → 动作"的映射终于可以直接单测，
 * 不必为四个小按钮挂起整套停靠布局。
 */
import { expect, test } from "vitest";

import {
    PANEL_APPEARANCE,
    PANEL_FILE_BROWSER,
    PANEL_NOTEBOOK,
    PANEL_TIMELINE,
    registerBuiltinPanels,
} from "./registerBuiltinPanels";
import { floatTitleBarActions } from "./floatTitleBar";
import { getPanel, resetPanelRegistryForTests } from "../../features/dock/panelRegistry";

test("外观设置：拖不进去的面板也不给「重新停靠」按钮", () => {
    resetPanelRegistryForTests();
    registerBuiltinPanels();

    const actions = floatTitleBarActions(getPanel(PANEL_APPEARANCE));
    expect(actions.redock, "dockable: false 的面板不得出现重停按钮").toBe(false);
    expect(actions.collapse).toBe(true);
    // 不可拆是声明使然（独立窗口对设置界面没有收益），保留灰色解释占位。
    expect(actions.detach).toBe("unsupported");
});

test("时间轴：可重停，但拆分是灰色解释占位（WebGL/波形缓存不可跨窗口）", () => {
    resetPanelRegistryForTests();
    registerBuiltinPanels();

    const actions = floatTitleBarActions(getPanel(PANEL_TIMELINE));
    expect(actions.redock).toBe(true);
    expect(actions.detach).toBe("unsupported");
});

test("文件浏览器与记事本：可重停、可拆分", () => {
    resetPanelRegistryForTests();
    registerBuiltinPanels();

    for (const panelId of [PANEL_FILE_BROWSER, PANEL_NOTEBOOK]) {
        const actions = floatTitleBarActions(getPanel(panelId));
        expect(actions.redock, panelId).toBe(true);
        expect(actions.detach, panelId).toBe("available");
    }
});

test("未注册面板：默认可停靠，但不给拆分动作（没有声明就没有可信解释）", () => {
    expect(floatTitleBarActions(undefined)).toEqual({
        collapse: true,
        detach: "hidden",
        redock: true,
    });
});
