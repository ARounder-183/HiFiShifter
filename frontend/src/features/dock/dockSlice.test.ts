import { beforeEach, test } from "vitest";

import reducer, {
    hydrateDock,
    markFormsMounted,
    setDockLayout,
    setGutterSize,
    syncRegisteredPanels,
    toggleMaximizeActive,
    floatForm,
    closeForm,
    openPanel,
} from "./dockSlice.ts";
import { registerPanel, resetPanelRegistryForTests } from "./panelRegistry.ts";
import { findTabsetOfForm, isFormVisible } from "./dockTree.ts";
import { DEFAULT_DOCK_SETTINGS } from "./dockSettings.ts";
import type { DockSplitNode } from "./dockTypes.ts";

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

function assert(condition: boolean, label: string): void {
    if (!condition) throw new Error(label);
}

function shape(node: DockSplitNode["a"]): string {
    if (node.t === "tabset") return `[${node.tabs.join(",")}]`;
    return `(${shape(node.a)}|${shape(node.b)})`;
}

const noop = () => null;

function registerFakes(): void {
    registerPanel({
        id: "timeline",
        titleKey: "panel_timeline",
        component: noop,
        defaultWidth: 900,
        defaultHeight: 400,
        preferMain: true,
        order: 10,
    });
    registerPanel({
        id: "paramEditor",
        titleKey: "panel_editor",
        component: noop,
        defaultWidth: 900,
        defaultHeight: 300,
        preferMain: true,
        singleton: false,
        order: 20,
    });
    registerPanel({
        id: "fileBrowser",
        titleKey: "panel_io",
        component: noop,
        defaultWidth: 320,
        defaultHeight: 400,
        defaultPlacement: { side: "right", sizePx: 360 },
        order: 30,
    });
    registerPanel({
        id: "notebook",
        titleKey: "notebook",
        component: noop,
        defaultWidth: 420,
        defaultHeight: 400,
        defaultPlacement: { side: "right", sizePx: 360, tabWith: "fileBrowser" },
        order: 40,
    });
}

beforeEach(() => {
    resetPanelRegistryForTests();
    registerFakes();
});

test("features/dock/dockSlice.test.ts scripted checks", async () => {
    // ── 出厂状态：两个主窗体，未 hydrate ─────────────────────────
    {
        const state = reducer(undefined, { type: "@@INIT" });
        assertEqual(shape(state.layout.tree), "([timeline]|[paramEditor])", "factory layout");
        assertEqual(state.hydrated, false, "starts unhydrated (the persist gate stays shut)");
        assertEqual(state.settings.dockModifier, DEFAULT_DOCK_SETTINGS.dockModifier, "defaults");
        assertEqual(state.mountedFormIds, [], "nothing mounted before the first visibility pass");
        assertEqual(state.maximized, null, "not maximized");
    }

    // ── hydrate：从磁盘恢复布局 + 行为选项 ───────────────────────
    {
        const persisted = {
            schema: 1,
            tree: {
                t: "split",
                id: "z1",
                dir: "row",
                ratio: 0.7,
                fixed: { side: "b", px: 380 },
                a: { t: "tabset", id: "z2", tabs: ["timeline"], active: "timeline" },
                b: { t: "tabset", id: "z3", tabs: ["fileBrowser", "notebook"], active: "notebook" },
            },
            forms: {
                timeline: { id: "timeline", panelId: "timeline" },
                paramEditor: { id: "paramEditor", panelId: "paramEditor" },
                fileBrowser: { id: "fileBrowser", panelId: "fileBrowser" },
                notebook: { id: "notebook", panelId: "notebook" },
            },
            order: ["timeline", "paramEditor", "fileBrowser", "notebook"],
            floatOrder: [],
            gutters: { timelineTrackHeaderPx: 320 },
        };

        const state = reducer(
            undefined,
            hydrateDock({ settings: { dockModifier: "alt", edgeBandPx: 40 }, layout: persisted }),
        );
        assertEqual(state.hydrated, true, "hydrated flag opens the persist gate");
        assertEqual(
            shape(state.layout.tree),
            "([timeline]|[fileBrowser,notebook])",
            "persisted arrangement restored",
        );
        assertEqual(
            (state.layout.tree as DockSplitNode).fixed,
            { side: "b", px: 380 },
            "fixed-width dock restored",
        );
        assertEqual(state.layout.gutters.timelineTrackHeaderPx, 320, "gutter width restored");
        assertEqual(state.settings.dockModifier, "alt", "behaviour option restored");
        assertEqual(state.settings.edgeBandPx, 40, "second behaviour option restored");
        // 未在磁盘上出现的选项回落到默认值，而不是 undefined。
        assertEqual(
            state.settings.showDropPreview,
            DEFAULT_DOCK_SETTINGS.showDropPreview,
            "missing option falls back to its default",
        );
    }

    // ── hydrate 后窗体 id 必须稳定（不得退化成 fileBrowser:2）──────
    //
    // 曾经这里会丢记录：`hydrateDock` 用出厂布局覆盖时，`syncRegisteredPanels`
    // 刚补上的 fileBrowser / notebook 记录一并被丢掉，于是用户打开文件浏览器
    // 会新建 `fileBrowser:2`。窗体 id 是持久化 JSON 的一部分（将来 API 与用户
    // 共享预设都要引用），重复 id 会让外部引用失效。
    {
        const state = reducer(undefined, hydrateDock({ settings: null, layout: null }));
        assertEqual(
            Object.keys(state.layout.forms).sort(),
            ["fileBrowser", "notebook", "paramEditor", "timeline"],
            "every registered panel keeps its canonical record after hydration",
        );
        const opened = reducer(state, openPanel({ panelId: "fileBrowser" }));
        assertEqual(
            findTabsetOfForm(opened.layout.tree, "fileBrowser")?.tabs,
            ["fileBrowser"],
            "opening a panel reuses its canonical form id",
        );
        assertEqual(
            Object.keys(opened.layout.forms).some((id) => id.includes(":")),
            false,
            "no suffixed duplicate forms appear",
        );
    }

    // ── hydrate：磁盘内容是垃圾 → 出厂布局，且仍然标记已 hydrate ──
    {
        const state = reducer(undefined, hydrateDock({ settings: null, layout: "not a layout" }));
        assertEqual(shape(state.layout.tree), "([timeline]|[paramEditor])", "garbage falls back");
        assertEqual(state.hydrated, true, "still hydrated so the layout can be re-saved");
    }

    // ── hydrate：面板已卸载（未注册）→ 该窗体被剔除，其余保留 ─────
    {
        const state = reducer(
            undefined,
            hydrateDock({
                settings: null,
                layout: {
                    schema: 1,
                    tree: {
                        t: "tabset",
                        id: "z1",
                        tabs: ["timeline", "ghost"],
                        active: "ghost",
                    },
                    forms: {
                        timeline: { id: "timeline", panelId: "timeline" },
                        ghost: { id: "ghost", panelId: "ghost" },
                    },
                },
            }),
        );
        assertEqual(shape(state.layout.tree), "[timeline]", "unregistered form dropped");
        assertEqual(state.layout.forms.ghost, undefined, "its record is dropped too");
    }

    // ── markFormsMounted：单调增长，重复登记不产生重复项 ──────────
    {
        let state = reducer(undefined, markFormsMounted(["timeline", "paramEditor"]));
        assertEqual(state.mountedFormIds, ["timeline", "paramEditor"], "first pass");
        state = reducer(state, markFormsMounted(["paramEditor", "notebook"]));
        assertEqual(
            state.mountedFormIds,
            ["timeline", "paramEditor", "notebook"],
            "union only: already-mounted forms are not re-added",
        );
        // 关闭面板不会把它从挂载集合里摘掉 —— 重开才是零成本的。
        state = reducer(state, closeForm("notebook"));
        assertEqual(state.mountedFormIds.includes("notebook"), true, "closed panels stay mounted");
    }

    // ── syncRegisteredPanels：补齐已注册面板为"已关闭" ────────────
    {
        let state = reducer(undefined, { type: "@@INIT" });
        assertEqual(Object.keys(state.layout.forms).sort(), ["paramEditor", "timeline"], "before");
        state = reducer(state, syncRegisteredPanels());
        assertEqual(
            Object.keys(state.layout.forms).sort(),
            ["fileBrowser", "notebook", "paramEditor", "timeline"],
            "after",
        );
        assertEqual(isFormVisible(state.layout, "notebook"), false, "added closed, not open");
    }

    // ── 打开/关闭：显隐由布局派生，且拒绝关掉最后一个可见窗体 ──────
    {
        let state = reducer(undefined, syncRegisteredPanels());
        state = reducer(state, openPanel({ panelId: "fileBrowser" }));
        assertEqual(isFormVisible(state.layout, "fileBrowser"), true, "opened");
        state = reducer(state, openPanel({ panelId: "notebook" }));
        assertEqual(
            findTabsetOfForm(state.layout.tree, "notebook")?.tabs,
            ["fileBrowser", "notebook"],
            "notebook tabs with the file browser instead of taking a second column",
        );
        state = reducer(state, closeForm("fileBrowser"));
        assertEqual(isFormVisible(state.layout, "fileBrowser"), false, "closed");
        state = reducer(state, closeForm("notebook"));
        assertEqual(isFormVisible(state.layout, "notebook"), false, "closed");
        // 只剩两个主窗体时再关就被拒绝（空布局不可渲染）。
        state = reducer(state, closeForm("paramEditor"));
        state = reducer(state, closeForm("timeline"));
        assertEqual(
            isFormVisible(state.layout, "timeline"),
            true,
            "the last visible form is protected",
        );
    }

    // ── 浮动：从树上摘除并记录几何 ───────────────────────────────
    {
        let state = reducer(undefined, syncRegisteredPanels());
        state = reducer(state, openPanel({ panelId: "fileBrowser" }));
        state = reducer(state, floatForm({ formId: "fileBrowser" }));
        assertEqual(
            isFormVisible(state.layout, "fileBrowser"),
            true,
            "still visible while floating",
        );
        assertEqual(
            findTabsetOfForm(state.layout.tree, "fileBrowser"),
            null,
            "removed from the tree",
        );
        assert(state.layout.forms.fileBrowser.float !== null, "geometry recorded");
        assertEqual(state.layout.floatOrder, ["fileBrowser"], "float z-order");
    }

    // ── 最大化：整片工作区只留当前窗体，再按一次完整还原 ───────────
    {
        let state = reducer(undefined, syncRegisteredPanels());
        state = reducer(state, openPanel({ panelId: "fileBrowser" }));
        const before = shape(state.layout.tree);
        state = reducer(state, { type: "dock/focusForm", payload: "timeline" });
        state = reducer(state, toggleMaximizeActive());
        assertEqual(shape(state.layout.tree), "[timeline]", "maximized to the focused form");
        assert(state.maximized !== null, "previous tree remembered");
        state = reducer(state, toggleMaximizeActive());
        assertEqual(shape(state.layout.tree), before, "restored exactly");
        assertEqual(state.maximized, null, "maximize state cleared");
    }

    // ── 沟槽尺寸 ────────────────────────────────────────────────
    {
        let state = reducer(undefined, { type: "@@INIT" });
        state = reducer(state, setGutterSize({ key: "timelineTrackHeaderPx", px: 333.7 }));
        assertEqual(state.layout.gutters.timelineTrackHeaderPx, 334, "gutter width rounded");
    }

    // ── setDockLayout：来自 API/导入的布局也要过归一化 ─────────────
    {
        const state = reducer(undefined, setDockLayout({ schema: 1, tree: { t: "bogus" } }));
        assertEqual(
            shape(state.layout.tree),
            "([timeline]|[paramEditor])",
            "invalid layout rejected",
        );
    }
});
