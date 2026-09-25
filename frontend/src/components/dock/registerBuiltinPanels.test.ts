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
import { isFormFloating, isFormVisible } from "../../features/dock/dockTree";
import type { DockLayout } from "../../features/dock/dockTypes";
import { PANEL_NOTEBOOK, PANEL_UNDO_HISTORY, registerBuiltinPanels } from "./registerBuiltinPanels";

function assert(condition: boolean, label: string): void {
    if (!condition) throw new Error(label);
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
});
