import { beforeEach, test } from "vitest";

import {
    closeFormInLayout,
    createDefaultDockLayout,
    ensureRegisteredPanels,
    findClosedFormForPanel,
    findMainTabset,
    findVisibleFormForPanel,
    MAIN_FORM_PARAM_EDITOR,
    MAIN_FORM_TIMELINE,
    migrateDockLayout,
    normalizeDockLayout,
    openPanelInLayout,
    placeForm,
} from "./dockSchema.ts";
import { collectTabsets, findTabsetOfForm, isFormVisible } from "./dockTree.ts";
import { registerPanel, resetPanelRegistryForTests } from "./panelRegistry.ts";
import type { DockSplitNode, DockTabsetNode } from "./dockTypes.ts";

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

function registerFakes(): void {
    const noop = () => null;
    registerPanel({
        id: MAIN_FORM_TIMELINE,
        titleKey: "panel_timeline",
        component: noop,
        defaultWidth: 900,
        defaultHeight: 400,
        preferMain: true,
        order: 10,
    });
    registerPanel({
        id: MAIN_FORM_PARAM_EDITOR,
        titleKey: "panel_editor",
        component: noop,
        defaultWidth: 900,
        defaultHeight: 300,
        preferMain: true,
        order: 20,
    });
    registerPanel({
        id: "fileBrowser",
        titleKey: "panel_io",
        component: noop,
        defaultWidth: 280,
        defaultHeight: 400,
        defaultPlacement: { side: "right", sizePx: 360 },
        order: 30,
    });
    registerPanel({
        id: "notebook",
        titleKey: "notebook",
        component: noop,
        defaultWidth: 360,
        defaultHeight: 400,
        defaultPlacement: { side: "right", sizePx: 360, tabWith: "fileBrowser" },
        order: 40,
    });
    registerPanel({
        id: "undoHistory",
        titleKey: "undo_history",
        component: noop,
        defaultWidth: 380,
        defaultHeight: 420,
        singleton: true,
        order: 50,
    });
}

beforeEach(() => {
    resetPanelRegistryForTests();
    registerFakes();
});

test("features/dock/dockSchema.test.ts scripted checks", async () => {
    // ── 出厂布局：上时间轴 / 下参数编辑器，其余面板不预置 ──────────
    {
        const layout = createDefaultDockLayout();
        assertEqual(shape(layout.tree), "([timeline]|[paramEditor])", "default split");
        assertEqual((layout.tree as DockSplitNode).dir, "col", "default is a vertical split");
        assertEqual((layout.tree as DockSplitNode).ratio, 0.6, "timeline keeps 60%");
        assertEqual(Object.keys(layout.forms).sort(), ["paramEditor", "timeline"], "only main forms");
        assertEqual(layout.gutters.timelineTrackHeaderPx, 256, "track header default");
    }

    // ── ensureRegisteredPanels：补齐为"已关闭"，不改变排布 ─────────
    {
        const layout = ensureRegisteredPanels(createDefaultDockLayout());
        assertEqual(
            Object.keys(layout.forms).sort(),
            ["fileBrowser", "notebook", "paramEditor", "timeline", "undoHistory"],
            "all registered panels get records",
        );
        assertEqual(shape(layout.tree), "([timeline]|[paramEditor])", "tree untouched");
        assertEqual(isFormVisible(layout, "fileBrowser"), false, "new records start closed");
    }

    // ── 归一化：剔除未注册面板 ───────────────────────────────────
    {
        const raw = {
            schema: 1,
            tree: {
                t: "split",
                id: "z1",
                dir: "row",
                ratio: 0.5,
                fixed: null,
                a: { t: "tabset", id: "z2", tabs: ["timeline"], active: "timeline" },
                b: { t: "tabset", id: "z3", tabs: ["ghost"], active: "ghost" },
            },
            forms: {
                timeline: { id: "timeline", panelId: "timeline" },
                ghost: { id: "ghost", panelId: "ghost" },
            },
            order: ["timeline", "ghost"],
        };
        const layout = normalizeDockLayout(raw);
        assertEqual(layout.forms.ghost, undefined, "unregistered panel dropped");
        assertEqual(shape(layout.tree), "[timeline]", "tabset of only ghosts collapses away");
    }

    // ── 归一化：同一窗体在树上出现两次 → 只保留首次 ────────────────
    {
        const raw = {
            schema: 1,
            tree: {
                t: "split",
                id: "z1",
                dir: "row",
                ratio: 0.5,
                fixed: null,
                a: { t: "tabset", id: "z2", tabs: ["timeline"], active: "timeline" },
                b: { t: "tabset", id: "z3", tabs: ["timeline", "paramEditor"], active: "timeline" },
            },
            forms: {
                timeline: { id: "timeline", panelId: "timeline" },
                paramEditor: { id: "paramEditor", panelId: "paramEditor" },
            },
        };
        const layout = normalizeDockLayout(raw);
        assertEqual(shape(layout.tree), "([timeline]|[paramEditor])", "duplicate form deduped");
    }

    // ── 归一化：active 失效 → 修正为首个标签 ─────────────────────
    {
        const raw = {
            schema: 1,
            tree: {
                t: "tabset",
                id: "z1",
                tabs: ["timeline", "paramEditor"],
                active: "not-there",
            },
            forms: {
                timeline: { id: "timeline", panelId: "timeline" },
                paramEditor: { id: "paramEditor", panelId: "paramEditor" },
            },
        };
        const layout = normalizeDockLayout(raw);
        assertEqual((layout.tree as DockTabsetNode).active, "timeline", "stale active repaired");
    }

    // ── 归一化：树彻底坏掉 → 回退默认（但保留窗体记录）────────────
    {
        const raw = {
            schema: 1,
            tree: { t: "nonsense" },
            forms: {
                timeline: { id: "timeline", panelId: "timeline" },
                notebook: { id: "notebook", panelId: "notebook" },
            },
            order: ["timeline", "notebook"],
        };
        const layout = normalizeDockLayout(raw);
        assertEqual(shape(layout.tree), "([timeline]|[paramEditor])", "broken tree falls back");
        assertEqual(layout.forms.notebook?.panelId, "notebook", "form records survive");
    }

    // ── 归一化：沟槽尺寸钳制 ────────────────────────────────────
    {
        const layout = normalizeDockLayout({
            schema: 1,
            tree: { t: "tabset", id: "z1", tabs: ["timeline"], active: "timeline" },
            forms: { timeline: { id: "timeline", panelId: "timeline" } },
            gutters: { timelineTrackHeaderPx: 99999, pianoRollAxisPx: -5 },
        });
        assertEqual(layout.gutters.timelineTrackHeaderPx, 560, "gutter clamped high");
        assertEqual(layout.gutters.pianoRollAxisPx, 40, "gutter clamped low");
    }

    // ── 归一化：浮动几何清洗 + 与停靠互斥 ─────────────────────────
    {
        const layout = normalizeDockLayout({
            schema: 1,
            tree: { t: "tabset", id: "z1", tabs: ["timeline"], active: "timeline" },
            forms: {
                timeline: { id: "timeline", panelId: "timeline" },
                notebook: { id: "notebook", panelId: "notebook", float: { x: "bad", y: 5, w: 1, h: 1 } },
            },
            order: ["timeline", "notebook"],
            floatOrder: ["notebook", "notebook"],
        });
        assertEqual(layout.forms.notebook?.float?.x, 120, "non-numeric float x falls back");
        assertEqual(layout.forms.notebook?.float?.w, 160, "float width clamped to its minimum");
        assertEqual(layout.floatOrder, ["notebook"], "float order deduped");
    }

    // ── 归一化：既在树上又有浮动几何 → 清掉浮动（可见性互斥）──────
    {
        const layout = normalizeDockLayout({
            schema: 1,
            tree: { t: "tabset", id: "z1", tabs: ["notebook"], active: "notebook" },
            forms: {
                timeline: { id: "timeline", panelId: "timeline" },
                notebook: { id: "notebook", panelId: "notebook", float: { x: 10, y: 10, w: 300, h: 200 } },
            },
            order: ["timeline", "notebook"],
        });
        assertEqual(layout.forms.notebook?.float, null, "docked wins over floating");
        assertEqual(layout.floatOrder, [], "float order cleaned");
    }

    // ── 迁移：来自更新版本 → 交给归一化重建 ──────────────────────
    {
        assertEqual(migrateDockLayout({ schema: 99 }), null, "future schema rejected");
        const current = { schema: 1, tree: null };
        assert(migrateDockLayout(current) === current, "current schema passes through");
        assertEqual(migrateDockLayout(null), null, "null passes through");
    }

    // ── 打开面板：按默认落点固定在右侧 ───────────────────────────
    {
        const base = ensureRegisteredPanels(createDefaultDockLayout());
        const opened = openPanelInLayout(base, "fileBrowser");
        assertEqual(collectTabsets(opened.tree).length, 3, "a third tabset appeared");
        const split = opened.tree as DockSplitNode;
        assertEqual(split.dir, "row", "workspace splits horizontally");
        assertEqual(split.fixed, { side: "b", px: 360 }, "right dock pinned to 360px");
        assertEqual(shape(split.a), "([timeline]|[paramEditor])", "main area stays intact");
        assertEqual(shape(split.b), "[fileBrowser]", "browser spans the right edge");
        assertEqual(isFormVisible(opened, "fileBrowser"), true, "browser is visible");
    }

    // ── 打开记事本：与文件浏览器并入同一组（不再并排挤占）──────────
    {
        const withBrowser = openPanelInLayout(
            ensureRegisteredPanels(createDefaultDockLayout()),
            "fileBrowser",
        );
        const withNotebook = openPanelInLayout(withBrowser, "notebook");
        const browserTabset = findTabsetOfForm(withNotebook.tree, "fileBrowser");
        assertEqual(browserTabset?.tabs, ["fileBrowser", "notebook"], "notebook tabs with browser");
        assertEqual(collectTabsets(withNotebook.tree).length, 3, "no extra column created");
    }

    // ── 单例面板已可见时再打开 = 无操作（不产生第二个时间轴）────────
    {
        const layout = ensureRegisteredPanels(createDefaultDockLayout());
        assertEqual(openPanelInLayout(layout, "timeline"), layout, "singleton open is a no-op");
    }

    // ── 重开已关闭的面板复用窗体记录（标题与 props 保留）───────────
    {
        let layout = ensureRegisteredPanels(createDefaultDockLayout());
        layout = openPanelInLayout(layout, "undoHistory");
        layout = { ...layout, forms: { ...layout.forms, undoHistory: { ...layout.forms.undoHistory, title: "我的历史" } } };
        layout = closeFormInLayout(layout, "undoHistory");
        assertEqual(isFormVisible(layout, "undoHistory"), false, "closed");
        assertEqual(findClosedFormForPanel(layout, "undoHistory"), "undoHistory", "record reused");
        const reopened = openPanelInLayout(layout, "undoHistory");
        assertEqual(reopened.forms.undoHistory?.title, "我的历史", "title survives close/reopen");
        assertEqual(isFormVisible(reopened, "undoHistory"), true, "reopened");
    }

    // ── 关闭：拒绝关掉最后一个可见窗体 ───────────────────────────
    {
        let layout = ensureRegisteredPanels(createDefaultDockLayout());
        layout = closeFormInLayout(layout, MAIN_FORM_PARAM_EDITOR);
        assertEqual(isFormVisible(layout, MAIN_FORM_PARAM_EDITOR), false, "param editor closed");
        assertEqual(shape(layout.tree), "[timeline]", "its group collapsed");
        const guarded = closeFormInLayout(layout, MAIN_FORM_TIMELINE);
        assertEqual(isFormVisible(guarded, MAIN_FORM_TIMELINE), true, "last visible form is protected");
        assert(guarded === layout, "guard returns the same layout object");
    }

    // ── findMainTabset：优先含 preferMain 面板的组 ───────────────
    {
        let layout = ensureRegisteredPanels(createDefaultDockLayout());
        layout = openPanelInLayout(layout, "fileBrowser");
        assertEqual(findMainTabset(layout)?.tabs, ["timeline"], "main tabset located");
    }

    // ── findVisibleFormForPanel ─────────────────────────────────
    {
        const layout = ensureRegisteredPanels(createDefaultDockLayout());
        assertEqual(findVisibleFormForPanel(layout, "timeline"), "timeline", "visible form found");
        assertEqual(findVisibleFormForPanel(layout, "notebook"), null, "closed panel has none");
    }

    // ── placeForm：center 并入主组 ──────────────────────────────
    {
        const layout = ensureRegisteredPanels(createDefaultDockLayout());
        const tree = placeForm(layout, "undoHistory", { side: "center" });
        assertEqual(findTabsetOfForm(tree, "timeline")?.tabs, ["timeline", "undoHistory"], "merged");
    }

    // ── placeForm：左侧拆分（无固定尺寸）────────────────────────
    {
        const layout = ensureRegisteredPanels(createDefaultDockLayout());
        const tree = placeForm(layout, "undoHistory", { side: "left" });
        assertEqual(
            shape(tree),
            "([undoHistory]|([timeline]|[paramEditor]))",
            "left dock spans the full height",
        );
    }
});
