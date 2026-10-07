/**
 * 内置面板默认落点的契约（"默认浮出的面板不能叠在一起"）。
 *
 * 【要钉死什么】记事本与撤销历史都是"辅助面板"：默认**关闭**，用户打开时以浮窗
 * 落在主窗口右下角一带。两个都打开时若落点相同就会完全重叠 —— 用户只能看到上面
 * 那一个，会以为另一个没打开。这里用真实注册表 + 真实布局函数验证：
 * 1. 默认布局里两者都不可见（启动时不该自己冒出来）；
 * 2. 打开后都处于浮动状态（而不是并入标签组）；
 * 3. 两者的矩形**不重叠**，且间隔不小于默认边距。
 */
import { test } from "vitest";

import {
    createDefaultDockLayout,
    ensureRegisteredPanels,
    openPanelInLayout,
} from "../../features/dock/dockSchema";
import { resolveFloatRect } from "../../features/dock/dockDropTarget";
import { getPanel } from "../../features/dock/panelRegistry";
import { isFormFloating, isFormVisible } from "../../features/dock/dockTree";
import type { DockLayout } from "../../features/dock/dockTypes";
import {
    PANEL_ARA_HOST,
    PANEL_NOTEBOOK,
    PANEL_UNDO_HISTORY,
    registerBuiltinPanels,
} from "./registerBuiltinPanels";

function assert(condition: boolean, label: string): void {
    if (!condition) throw new Error(label);
}

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

function rectOf(layout: DockLayout, formId: string, viewport: { w: number; h: number }) {
    const float = layout.forms[formId]?.float;
    if (!float) throw new Error(`${formId}: no float geometry`);
    return resolveFloatRect(float, viewport);
}

function overlaps(
    a: { x: number; y: number; w: number; h: number },
    b: { x: number; y: number; w: number; h: number },
): boolean {
    return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}

test("components/dock/registerBuiltinPanels.test.ts default float placements", () => {
    registerBuiltinPanels();
    const base = ensureRegisteredPanels(createDefaultDockLayout());

    // 1) 默认关闭：辅助面板不在启动布局里冒出来。
    assert(!isFormVisible(base, PANEL_NOTEBOOK), "notebook is closed by default");
    assert(!isFormVisible(base, PANEL_UNDO_HISTORY), "undo history is closed by default");

    // 2) 打开后都是浮窗（而不是挤进某个标签组）。
    const withNotebook = openPanelInLayout(base, PANEL_NOTEBOOK);
    const withBoth = openPanelInLayout(withNotebook, PANEL_UNDO_HISTORY);
    assert(isFormFloating(withBoth, PANEL_NOTEBOOK), "notebook opens floating");
    assert(isFormFloating(withBoth, PANEL_UNDO_HISTORY), "undo history opens floating");

    // 3) 落点错开：多个视口下都不重叠（锚点是按视口推导的语义位置）。
    for (const viewport of [
        { w: 1920, h: 1080 },
        { w: 1400, h: 900 },
        { w: 1100, h: 700 },
    ]) {
        const notebook = rectOf(withBoth, PANEL_NOTEBOOK, viewport);
        const undo = rectOf(withBoth, PANEL_UNDO_HISTORY, viewport);
        assert(
            !overlaps(notebook, undo),
            `default floats must not overlap at ${viewport.w}x${viewport.h}`,
        );
        // 水平间隔恰好一个默认边距（24）：紧邻但不贴合，两者底边对齐。
        const gap = notebook.x - (undo.x + undo.w);
        assert(gap >= 24, `expected a >=24px gap, got ${gap} at ${viewport.w}x${viewport.h}`);
        assert(notebook.y === undo.y, "the two default floats stay bottom-aligned");
    }

    // 4) 由某个控件打开时（例如从撤销/重做按钮打开操作记录）：用命令层算出的
    //    几何，而不是声明的锚点 —— 面板落在触发控件旁边。
    {
        const nearRect = { x: 300, y: 40, w: 420, h: 420, anchor: null as null };
        const opened = openPanelInLayout(base, PANEL_UNDO_HISTORY, undefined, nearRect);
        assertEqual(
            opened.forms[PANEL_UNDO_HISTORY]?.float,
            nearRect,
            "an explicit geometry wins over the declared anchor",
        );
        assertEqual(
            opened.forms[PANEL_UNDO_HISTORY]?.float?.anchor ?? null,
            null,
            "a concrete geometry clears the anchor",
        );
    }
});

/**
 * ARA 宿主会话面板的声明契约。
 *
 * 【要钉死什么】它此前是 App 与工作区之间的一整条**常驻横条**：独立 App 的多数
 * 用户从不连宿主，却一直占着工作区高度。改成面板之后，"默认关闭 + 不进窗口菜单 +
 * 不可停靠"这三条声明就是"不再常驻"的全部依据 —— 少任何一条都会让它重新变成
 * 用户无法回避的表面（例如混进「窗口」菜单就与日常面板等价了）。
 */
test("the ARA host session panel is an opt-in floating panel, never part of the work layout", () => {
    registerBuiltinPanels();
    const base = ensureRegisteredPanels(createDefaultDockLayout());
    assert(!isFormVisible(base, PANEL_ARA_HOST), "the ARA host panel is closed by default");

    const definition = getPanel(PANEL_ARA_HOST);
    if (!definition) throw new Error("the ARA host panel is registered");
    assert(
        definition.excludeFromWindowMenu === true,
        "low-frequency entries stay out of the Window menu",
    );
    assert(definition.dockable === false, "a session panel must not be woven into the layout");
    assert(definition.singleton === true, "reopening focuses the same session window");

    const opened = openPanelInLayout(base, PANEL_ARA_HOST);
    assert(isFormFloating(opened, PANEL_ARA_HOST), "the ARA host panel opens floating");
});
