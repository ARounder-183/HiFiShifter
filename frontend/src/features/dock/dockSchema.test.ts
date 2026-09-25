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
    resolveSyncOffsetForms,
} from "./dockSchema.ts";
import { collectTabsets, findTabsetOfForm, isFormVisible, MAX_SPLIT_RATIO, MIN_SPLIT_RATIO, addFormToTabset } from "./dockTree.ts";
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
    registerPanel({
        id: "opensAsFloat",
        titleKey: "notebook",
        component: noop,
        defaultWidth: 460,
        defaultHeight: 420,
        // 打开时以浮窗落在右下角（记事本的形态；**默认仍是关闭**）。
        openAsFloating: { width: 460, height: 420, anchor: "bottom-right" },
        order: 60,
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
        assertEqual(
            Object.keys(layout.forms).sort(),
            ["paramEditor", "timeline"],
            "only main forms",
        );
        assertEqual(layout.gutters.timelineTrackHeaderPx, 256, "track header default");
    }

    // ── ensureRegisteredPanels：补齐为"已关闭"，不改变排布 ─────────
    {
        const layout = ensureRegisteredPanels(createDefaultDockLayout());
        assertEqual(
            Object.keys(layout.forms).sort(),
            ["fileBrowser", "notebook", "opensAsFloat", "paramEditor", "timeline", "undoHistory"],
            "all registered panels get records",
        );
        assertEqual(shape(layout.tree), "([timeline]|[paramEditor])", "tree untouched");
        assertEqual(isFormVisible(layout, "fileBrowser"), false, "new records start closed");

        // ── 声明了 openAsFloating 的面板：**默认仍是关闭**，打开时才浮出 ──
        //
        // 这是用户明确要求的语义："记事本默认仍然是关闭状态"，而"悬浮在右下角"
        // 说的是**打开它时**的形态 —— 启动时不该有任何面板自己冒出来。
        {
            assertEqual(
                isFormVisible(layout, "opensAsFloat"),
                false,
                "a panel that opens as a float is still closed by default",
            );

            const opened = openPanelInLayout(layout, "opensAsFloat");
            const form = opened.forms.opensAsFloat;
            assertEqual(form?.floating, true, "opening it produces a floating window");
            assertEqual(form?.float?.w, 460, "declared width");
            assertEqual(form?.float?.h, 420, "declared height");
            assertEqual(
                form?.float?.anchor,
                "bottom-right",
                "and it is anchored to the lower-right corner",
            );
            assertEqual(
                findTabsetOfForm(opened.tree, "opensAsFloat"),
                null,
                "a floating panel takes no cell in the layout tree",
            );
            assertEqual(
                opened.floatOrder.includes("opensAsFloat"),
                true,
                "and is registered for the floating layer",
            );

            // 再次打开（已可见）不应改变任何东西。
            assertEqual(
                openPanelInLayout(opened, "opensAsFloat"),
                opened,
                "opening an already-visible singleton is a no-op",
            );
        }
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

    // ── 归一化：重复的 Zone id → 现场重编，两组都保持可用 ──────────
    //
    // 两组共用同一个 id 时（损坏的配置），`addFormToTabset` 会把窗体插进
    // **每一个**同 id 的组，而 `findZone`/`replaceZone` 只认第一个 —— 一次
    // "拖入标签组"就会同时改掉两个组。必须在归一化时重编其中一个。
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
                b: { t: "tabset", id: "z2", tabs: ["paramEditor"], active: "paramEditor" },
            },
            forms: {
                timeline: { id: "timeline", panelId: "timeline" },
                paramEditor: { id: "paramEditor", panelId: "paramEditor" },
            },
        };
        const layout = normalizeDockLayout(raw);
        const tabsets = collectTabsets(layout.tree);
        assertEqual(tabsets.length, 2, "both tabsets survive");
        assert(tabsets[0]!.id !== tabsets[1]!.id, "duplicate zone id re-minted");
        assertEqual(shape(layout.tree), "([timeline]|[paramEditor])", "tabs intact");
        // 曾经被重复 id 破坏的操作：插入只命中一个组。
        const retargeted = addFormToTabset(layout.tree, tabsets[0]!.id, "undoHistory");
        assertEqual(
            shape(retargeted),
            "([timeline,undoHistory]|[paramEditor])",
            "an insert reaches exactly one tabset",
        );

        // id 缺失的两个节点（历史上都会落成 "z0"）同样不能共享 id。
        const anonymous = normalizeDockLayout({
            schema: 1,
            tree: {
                t: "split",
                id: "z1",
                dir: "row",
                ratio: 0.5,
                fixed: null,
                a: { t: "tabset", tabs: ["timeline"], active: "timeline" },
                b: { t: "tabset", tabs: ["paramEditor"], active: "paramEditor" },
            },
            forms: {
                timeline: { id: "timeline", panelId: "timeline" },
                paramEditor: { id: "paramEditor", panelId: "paramEditor" },
            },
        });
        const anonTabsets = collectTabsets(anonymous.tree);
        assertEqual(anonTabsets.length, 2, "both anonymous tabsets survive");
        assert(
            anonTabsets[0]!.id !== anonTabsets[1]!.id,
            "missing zone ids are minted distinctly",
        );
    }

    // ── 归一化：分割比例钳制 ────────────────────────────────────
    //
    // `pruneTree` 只在子树被修剪时才顺手钳一次 ratio，两侧都完好的树不会被它
    // 碰到 —— 损坏的比例必须在 `normalizeTree` 里就地钳制，否则 `ratio: 999`
    // 一路存活到渲染，`paneStyle` 算出负的 flexGrow，整侧被挤成不可用。
    {
        const base = {
            forms: {
                timeline: { id: "timeline", panelId: "timeline" },
                paramEditor: { id: "paramEditor", panelId: "paramEditor" },
            },
        };
        const treeFor = (ratio: number) => ({
            schema: 1,
            tree: {
                t: "split",
                id: "z1",
                dir: "row",
                ratio,
                fixed: null,
                a: { t: "tabset", id: "z2", tabs: ["timeline"], active: "timeline" },
                b: { t: "tabset", id: "z3", tabs: ["paramEditor"], active: "paramEditor" },
            },
            ...base,
        });
        const tooBig = normalizeDockLayout(treeFor(999));
        assertEqual(
            (tooBig.tree as DockSplitNode).ratio,
            MAX_SPLIT_RATIO,
            "corrupt ratio clamped high",
        );
        const tooSmall = normalizeDockLayout(treeFor(-3));
        assertEqual(
            (tooSmall.tree as DockSplitNode).ratio,
            MIN_SPLIT_RATIO,
            "corrupt ratio clamped low",
        );
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
            gutters: { timelineTrackHeaderPx: 99999 },
        });
        assertEqual(layout.gutters.timelineTrackHeaderPx, 560, "gutter clamped high");

        const tooSmall = normalizeDockLayout({
            schema: 1,
            tree: { t: "tabset", id: "z1", tabs: ["timeline"], active: "timeline" },
            forms: { timeline: { id: "timeline", panelId: "timeline" } },
            gutters: { timelineTrackHeaderPx: -5 },
        });
        assertEqual(tooSmall.gutters.timelineTrackHeaderPx, 120, "gutter clamped low");
    }

    // ── 归一化：独立窗口形态 ──────────────────────────────────────
    //
    // 【为什么这里不再有"屏幕坐标"】独立窗口不单独记录屏幕位置：它就是同一个浮窗的
    // 另一种呈现，位置由 `float` 换算（见 `detachedGeometry`）。旧布局里残留的
    // `floatScreen` 字段被忽略（归一化只挑它认识的字段）。
    {
        const withDetach = normalizeDockLayout({
            schema: 1,
            tree: { t: "tabset", id: "z1", tabs: ["timeline"], active: "timeline" },
            forms: {
                timeline: { id: "timeline", panelId: "timeline" },
                notebook: {
                    id: "notebook",
                    panelId: "notebook",
                    floating: true,
                    floatMode: "osWindow",
                    float: { x: 120, y: 60, w: 460, h: 420 },
                    // 早期版本留下的字段：必须被忽略而不是复活。
                    floatScreen: { x: 120.4, y: -30.6 },
                },
            },
            order: ["timeline", "notebook"],
            floatOrder: ["notebook"],
        });
        assertEqual(withDetach.forms.notebook?.floatMode, "osWindow", "独立窗口形态随布局持久化");
        assertEqual(
            withDetach.forms.notebook?.float,
            { x: 120, y: 60, w: 460, h: 420, anchor: null },
            "位置只由浮窗几何记录（残留的 floatScreen 被忽略）",
        );

        // 未知形态回退：回 inApp。
        const garbage = normalizeDockLayout({
            schema: 1,
            tree: { t: "tabset", id: "z1", tabs: ["timeline"], active: "timeline" },
            forms: {
                timeline: { id: "timeline", panelId: "timeline" },
                notebook: {
                    id: "notebook",
                    panelId: "notebook",
                    floating: true,
                    floatMode: "bogus",
                },
            },
            order: ["timeline", "notebook"],
            floatOrder: ["notebook"],
        });
        assertEqual(garbage.forms.notebook?.floatMode, "inApp", "未知形态回退为进程内浮层");

        // 标签行位置：默认下方，显式 top 保留。
        assertEqual(withDetach.tabPosition, "bottom", "标签行默认在下方");
        const topTabs = normalizeDockLayout({
            schema: 1,
            tree: { t: "tabset", id: "z1", tabs: ["timeline"], active: "timeline" },
            forms: { timeline: { id: "timeline", panelId: "timeline" } },
            tabPosition: "top",
        });
        assertEqual(topTabs.tabPosition, "top", "显式 top 被保留");
    }

    // ── 归一化：浮动几何清洗 + 与停靠互斥 ─────────────────────────
    {
        const layout = normalizeDockLayout({
            schema: 1,
            tree: { t: "tabset", id: "z1", tabs: ["timeline"], active: "timeline" },
            forms: {
                timeline: { id: "timeline", panelId: "timeline" },
                notebook: {
                    id: "notebook",
                    panelId: "notebook",
                    float: { x: "bad", y: 5, w: 1, h: 1 },
                },
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
                notebook: {
                    id: "notebook",
                    panelId: "notebook",
                    float: { x: 10, y: 10, w: 300, h: 200 },
                    floating: true,
                },
            },
            order: ["timeline", "notebook"],
        });
        // 几何是"下次拆下来用多大"的记忆，与"此刻是否浮动"是两件事：停靠期间
        // 抹掉它，用户拆下来的浮窗就会继承停靠时那片区域的尺寸，当初调好的
        // 大小再也找不回来。
        assertEqual(layout.forms.notebook?.floating, false, "docked wins over floating");
        const remembered = layout.forms.notebook?.float;
        assertEqual(
            [remembered?.x, remembered?.y, remembered?.w, remembered?.h],
            [10, 10, 300, 200],
            "float geometry is remembered, not discarded",
        );
        assertEqual(layout.floatOrder, [], "float order cleaned");
    }

    // ── 归一化：浮窗尺寸上限 ────────────────────────────────────
    //
    // 只有下限没有上限时，损坏的落盘数据（w: 100000）会渲染出一个撑爆视口的
    // 浮窗。上限 4000 与锚点偏移的钳制（±4000）同源：正常拖放永远碰不到。
    {
        const layout = normalizeDockLayout({
            schema: 1,
            tree: { t: "tabset", id: "z1", tabs: ["timeline"], active: "timeline" },
            forms: {
                timeline: { id: "timeline", panelId: "timeline" },
                notebook: {
                    id: "notebook",
                    panelId: "notebook",
                    floating: true,
                    float: { x: 0, y: 0, w: 100000, h: 99999 },
                },
            },
            order: ["timeline", "notebook"],
            floatOrder: ["notebook"],
        });
        assertEqual(layout.forms.notebook?.float?.w, 4000, "float width clamped to the ceiling");
        assertEqual(layout.forms.notebook?.float?.h, 4000, "float height clamped to the ceiling");
    }

    // ── 迁移：来自更新版本 → 交给归一化重建 ──────────────────────
    {
        assertEqual(migrateDockLayout({ schema: 99 }), null, "future schema rejected");
        const current = { schema: 1, tree: null };
        assert(migrateDockLayout(current) === current, "current schema passes through");
        assertEqual(migrateDockLayout(null), null, "null passes through");

        // 接线：迁移是 normalizeDockLayout 的第一道工序（所有导入/恢复路径都
        // 经过它）—— 更新版本的布局必须整体回退默认，而不是被逐项修补曲解。
        const future = normalizeDockLayout({
            schema: 99,
            tree: { t: "tabset", id: "z1", tabs: ["notebook"], active: "notebook" },
            forms: { notebook: { id: "notebook", panelId: "notebook" } },
            order: ["notebook"],
        });
        assertEqual(
            shape(future.tree),
            "([timeline]|[paramEditor])",
            "a layout from a newer build rebuilds defaults",
        );
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
        layout = {
            ...layout,
            forms: {
                ...layout.forms,
                undoHistory: { ...layout.forms.undoHistory, title: "我的历史" },
            },
        };
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
        // 先关掉"默认悬浮"的那个面板：它默认是可见的，不关掉的话"最后一个可见窗体"
        // 就不是时间轴了，保护逻辑不会触发。
        layout = closeFormInLayout(layout, "opensAsFloat");
        layout = closeFormInLayout(layout, MAIN_FORM_PARAM_EDITOR);
        assertEqual(isFormVisible(layout, MAIN_FORM_PARAM_EDITOR), false, "param editor closed");
        assertEqual(shape(layout.tree), "[timeline]", "its group collapsed");
        const guarded = closeFormInLayout(layout, MAIN_FORM_TIMELINE);
        assertEqual(
            isFormVisible(guarded, MAIN_FORM_TIMELINE),
            true,
            "last visible form is protected",
        );
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
        assertEqual(
            findTabsetOfForm(tree, "timeline")?.tabs,
            ["timeline", "undoHistory"],
            "merged",
        );
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

    // ── 同步偏移适用性：两个面板都可见即可 ─────────────────────────
    //
    // 用户报告："只要时间轴与参数编辑器同屏，则这个功能应该尝试将两者对齐"。
    // 这里曾有一个**静默失效**的缺陷：调用方把"时间轴窗体 id"写成了参数编辑器
    // 自己的窗体 id，两个参数于是是同一个窗体，判定必然为假、偏移被强制为 0，
    // 整个像素对齐失效且没有任何报错。因此本组用例同时钉住"两个 id 必须不同"。
    {
        const layout = ensureRegisteredPanels(createDefaultDockLayout());
        const resolved = resolveSyncOffsetForms(layout, "timeline", "paramEditor");
        assertEqual(
            resolved,
            { timelineFormId: "timeline", paramFormId: "paramEditor" },
            "the default stacked layout resolves two DISTINCT forms",
        );

        // 并排 / 浮动同样成立：判据是"都可见"，不是"必须上下堆叠"。
        const sideBySide = normalizeDockLayout({
            schema: 1,
            tree: {
                t: "split",
                id: "z1",
                dir: "row",
                ratio: 0.5,
                fixed: null,
                a: { t: "tabset", id: "z2", tabs: ["timeline"], active: "timeline" },
                b: { t: "tabset", id: "z3", tabs: ["paramEditor"], active: "paramEditor" },
            },
            forms: {
                timeline: { id: "timeline", panelId: "timeline" },
                paramEditor: { id: "paramEditor", panelId: "paramEditor" },
            },
        });
        assertEqual(
            resolveSyncOffsetForms(sideBySide, "timeline", "paramEditor") !== null,
            true,
            "side by side still counts (the offset may be negative)",
        );

        const floated = normalizeDockLayout({
            schema: 1,
            tree: { t: "tabset", id: "z1", tabs: ["timeline"], active: "timeline" },
            forms: {
                timeline: { id: "timeline", panelId: "timeline" },
                paramEditor: {
                    id: "paramEditor",
                    panelId: "paramEditor",
                    float: { x: 40, y: 40, w: 300, h: 200 },
                    floating: true,
                },
            },
            order: ["timeline", "paramEditor"],
        });
        assertEqual(
            resolveSyncOffsetForms(floated, "timeline", "paramEditor") !== null,
            true,
            "a floating param editor over the timeline still counts",
        );

        // 时间轴不可见 → 无法对齐。
        const timelineClosed = closeFormInLayout(layout, "timeline");
        assertEqual(
            resolveSyncOffsetForms(timelineClosed, "timeline", "paramEditor"),
            null,
            "no timeline on screen means nothing to align to",
        );

        // 注入的 formId 不属于本面板 → 忽略它，按面板 id 回退。
        assertEqual(
            resolveSyncOffsetForms(layout, "timeline", "paramEditor", "fileBrowser"),
            { timelineFormId: "timeline", paramFormId: "paramEditor" },
            "an unrelated injected formId is ignored",
        );
        // 参数写反（两个面板 id 传成同一个）→ 拒绝，而不是算出一个必然错的值。
        assertEqual(
            resolveSyncOffsetForms(layout, "timeline", "timeline"),
            null,
            "resolving to the SAME form as the timeline is rejected",
        );
    }
});
