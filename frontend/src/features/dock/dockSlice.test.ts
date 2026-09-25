import { beforeEach, test } from "vitest";

import reducer, {
    closeForm,
    dockFormTo,
    floatForm,
    hydrateDock,
    markFormsMounted,
    mergeFormInto,
    openPanel,
    setDockLayout,
    setGutterSize,
    splitFormTo,
    syncRegisteredPanels,
    toggleMaximizeActive,
    toggleTabsetCollapsed,
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
        // 打开时以浮窗落在右下角；**默认仍是关闭**。
        openAsFloating: { width: 460, height: 420, anchor: "bottom-right" },
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
        // 记事本声明了 openAsFloating，因此打开它**不并入**文件浏览器所在的标签组，
        // 而是浮出来（见下方"记事本默认关闭、打开时浮在右下角"一节）。
        state = reducer(state, openPanel({ panelId: "notebook" }));
        assertEqual(
            state.layout.forms.notebook.floating,
            true,
            "the notebook opens as a float, not as a tab",
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

    // ── 浮动：最后一个停靠窗体不允许浮走 ─────────────────────────
    //
    // 树上只剩它自己时 removeForm 得到 null：若照旧浮动，同一个窗体会同时出现在
    // 树上和浮层里（两个宿主抢同一个面板宿主 div，内容来回跳）。
    {
        let state = reducer(undefined, syncRegisteredPanels());
        // 先把参数编辑器浮走，树上只剩时间轴。
        state = reducer(state, floatForm({ formId: "paramEditor" }));
        assertEqual(findTabsetOfForm(state.layout.tree, "timeline") !== null, true, "still docked");
        const before = state.layout;
        state = reducer(state, floatForm({ formId: "timeline" }));
        assertEqual(state.layout, before, "the last docked form cannot float");
        assertEqual(
            findTabsetOfForm(state.layout.tree, "timeline") !== null,
            true,
            "and it stays in the tree, not in the float layer",
        );
        assertEqual(state.layout.floatOrder, ["paramEditor"], "float layer unchanged");
    }

    // ── 浮动：只写 w/h 的几何不清锚点，写 x/y 才清 ─────────────────
    //
    // 拆到独立窗口只传 w/h（位置仍由锚点语义表达）；整体替换 x/y 才表示
    // "位置已由用户/调用方决定"。
    {
        let state = reducer(undefined, syncRegisteredPanels());
        state = reducer(state, openPanel({ panelId: "notebook" }));
        assertEqual(
            state.layout.forms.notebook.float?.anchor,
            "bottom-right",
            "opens anchored",
        );
        // resize 路径：锚点保留。
        state = reducer(state, floatForm({ formId: "notebook", geometry: { w: 500, h: 300 } }));
        assertEqual(
            state.layout.forms.notebook.float?.anchor,
            "bottom-right",
            "w/h-only geometry keeps the anchor",
        );
        // 显式位置：锚点清除（既有行为）。
        state = reducer(
            state,
            floatForm({ formId: "notebook", geometry: { x: 10, y: 20, w: 500, h: 300 } }),
        );
        assertEqual(
            state.layout.forms.notebook.float?.anchor,
            null,
            "x/y geometry clears the anchor",
        );
    }

    // ── 浮窗尺寸与停靠尺寸分别保存 ───────────────────────────────
    //
    // 用户把一个小浮窗（320×240）停进一大片区域后，面板会被撑大；此时再拆下来
    // 必须回到 320×240，而不是继承停靠区域的尺寸。为此"记住的浮窗几何"与
    // "此刻是否浮动"是两个独立字段。
    {
        let state = reducer(undefined, syncRegisteredPanels());
        state = reducer(
            state,
            floatForm({ formId: "notebook", geometry: { x: 30, y: 40, w: 320, h: 240 } }),
        );
        assertEqual(state.layout.forms.notebook.float?.w, 320, "float keeps its own size");
        assertEqual(state.layout.forms.notebook.floating, true, "floating");

        // 停靠（无论落到多大的区域）。用真正的停靠动作：`openPanel` 对"已可见"
        // 的窗体是无操作（它只是打开，不负责搬运）。
        state = reducer(
            state,
            dockFormTo({ formId: "notebook", target: { kind: "tab", tabsetId: "z2" } }),
        );
        assertEqual(state.layout.forms.notebook.floating, false, "docked");
        const remembered = state.layout.forms.notebook.float;
        assertEqual(
            [remembered?.x, remembered?.y, remembered?.w, remembered?.h],
            [30, 40, 320, 240],
            "docking must not discard the float geometry",
        );

        // 再拆下来：回到当初的浮窗尺寸。
        state = reducer(state, floatForm({ formId: "notebook" }));
        assertEqual(state.layout.forms.notebook.floating, true, "floating again");
        assertEqual(
            [state.layout.forms.notebook.float?.w, state.layout.forms.notebook.float?.h],
            [320, 240],
            "undocking restores the remembered float size",
        );
    }

    // ── 记事本：默认关闭，打开时浮在右下角 ─────────────────────────
    //
    // 用户明确要求："记事本默认仍然是关闭状态"，而"悬浮在右下角"说的是**打开它
    // 时**的形态。启动时不该有任何面板自己冒出来。
    {
        const initial = reducer(undefined, syncRegisteredPanels());
        assertEqual(
            isFormVisible(initial.layout, "notebook"),
            false,
            "the notebook is closed on startup, not floating",
        );
        assertEqual(initial.layout.floatOrder, [], "and nothing is registered as floating");

        const opened = reducer(initial, openPanel({ panelId: "notebook" }));
        assertEqual(isFormVisible(opened.layout, "notebook"), true, "opening it shows it");
        assertEqual(
            opened.layout.forms.notebook.floating,
            true,
            "and it opens as a floating window rather than docking",
        );
        assertEqual(
            opened.layout.forms.notebook.float?.anchor,
            "bottom-right",
            "anchored to the lower-right corner",
        );
        assertEqual(
            findTabsetOfForm(opened.layout.tree, "notebook"),
            null,
            "taking no cell in the layout tree",
        );
    }

    // ── 浮窗停靠后必须仍然可见（用户报告："停靠以后窗口没有被正确展示"）──
    //
    // 浮动窗体不占布局树，`moveForm` 因此"没有源"。早期实现在这种情况下原样返回
    // 树，而调用方已经把 `floating` 清成 false 并移出 `floatOrder` —— 窗体既不在
    // 树上也不再浮动，`isFormVisible` 为假，窗口凭空消失，用户只能手动重开。
    // 拖拽浮窗标题栏停靠、点浮窗上的"重新停靠"，走的都是这条路径。
    {
        const dockedTabsetId = () => {
            const state = reducer(undefined, syncRegisteredPanels());
            return state;
        };

        // ① 拖拽停靠（合并进某个组）
        {
            let state = dockedTabsetId();
            state = reducer(state, floatForm({ formId: "notebook" }));
            assertEqual(isFormVisible(state.layout, "notebook"), true, "floating to start with");
            const target = findTabsetOfForm(state.layout.tree, "timeline");
            assert(target !== null, "the timeline tab group exists");
            state = reducer(
                state,
                mergeFormInto({ formId: "notebook", referenceFormId: "timeline" }),
            );
            assertEqual(state.layout.forms.notebook.floating, false, "no longer floating");
            assertEqual(
                isFormVisible(state.layout, "notebook"),
                true,
                "docking a floating form must keep it visible",
            );
            assertEqual(
                findTabsetOfForm(state.layout.tree, "notebook")?.id,
                target!.id,
                "and it lands in the requested group",
            );
        }

        // ② 拖拽停靠（在某侧拆分）
        {
            let state = reducer(undefined, syncRegisteredPanels());
            state = reducer(state, floatForm({ formId: "notebook" }));
            state = reducer(
                state,
                splitFormTo({ formId: "notebook", referenceFormId: "timeline", side: "right" }),
            );
            assertEqual(state.layout.forms.notebook.floating, false, "no longer floating");
            assertEqual(
                isFormVisible(state.layout, "notebook"),
                true,
                "splitting a floating form in must keep it visible",
            );
        }

        // ③ 浮窗标题栏上的"重新停靠"按钮
        {
            let state = reducer(undefined, syncRegisteredPanels());
            state = reducer(state, floatForm({ formId: "notebook" }));
            state = reducer(state, openPanel({ panelId: "fileBrowser" }));
            const main = findTabsetOfForm(state.layout.tree, "timeline");
            assert(main !== null, "main group exists");
            state = reducer(
                state,
                dockFormTo({ formId: "notebook", target: { kind: "tab", tabsetId: main!.id } }),
            );
            assertEqual(
                isFormVisible(state.layout, "notebook"),
                true,
                "the redock button must keep the form visible",
            );
        }

        // ④ 停靠进一个**折叠**的组：必须自动展开，否则"停靠成功"却仍然看不见
        {
            let state = reducer(undefined, syncRegisteredPanels());
            state = reducer(state, floatForm({ formId: "notebook" }));
            const target = findTabsetOfForm(state.layout.tree, "paramEditor");
            assert(target !== null, "param editor group exists");
            state = reducer(state, toggleTabsetCollapsed({ tabsetId: target!.id }));
            assertEqual(
                findTabsetOfForm(state.layout.tree, "paramEditor")?.collapsed,
                true,
                "precondition: the target group is collapsed",
            );
            state = reducer(
                state,
                dockFormTo({ formId: "notebook", target: { kind: "tab", tabsetId: target!.id } }),
            );
            assertEqual(isFormVisible(state.layout, "notebook"), true, "the form is visible");
            assertEqual(
                findTabsetOfForm(state.layout.tree, "notebook")?.collapsed,
                false,
                "docking expands the target group",
            );
        }
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

    // ── 最大化期间打开的面板：还原时补回主组，不凭空消失 ──────────
    //
    // 最大化把整棵树换成临时的单组树，此时打开的面板落在那棵树上；还原换回
    // 原树后它既不在树上也不浮动（窗体记录还在）—— 表现为"面板不见了"。
    // 注意判据是"在临时树里出现过"：一直关闭着的面板记录（每个注册面板都有
    // 一条）绝不能被这条路径顺手打开。
    {
        let state = reducer(undefined, syncRegisteredPanels());
        state = reducer(state, { type: "dock/focusForm", payload: "timeline" });
        state = reducer(state, toggleMaximizeActive());
        state = reducer(state, openPanel({ panelId: "fileBrowser" }));
        assertEqual(
            isFormVisible(state.layout, "fileBrowser"),
            true,
            "opened while maximized",
        );
        state = reducer(state, toggleMaximizeActive());
        assertEqual(state.maximized, null, "restored");
        assertEqual(
            isFormVisible(state.layout, "fileBrowser"),
            true,
            "a panel opened during maximize survives the restore",
        );
        assertEqual(
            shape(state.layout.tree),
            "([timeline,fileBrowser]|[paramEditor])",
            "it is re-adopted into the main tabset",
        );
        // 一直关闭着的 notebook（只是有窗体记录）不能被还原动作顺手打开。
        assertEqual(
            isFormVisible(state.layout, "notebook"),
            false,
            "a closed form record must NOT be opened by the restore",
        );
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
