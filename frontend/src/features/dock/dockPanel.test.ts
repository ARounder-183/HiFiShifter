/*
 * 面板（容器窗体）的行为测试。
 *
 * 覆盖四块：`dockPanel` 的派生判定（可拆性 / 标题 / 子树根）、归一化对面板
 * 的放行与清理（孤儿根 / 跨根去重 / 环形引用）、slice 的新动作（新建空面板 /
 * 浮窗组合 / 解散 / 自动解散）、以及"关面板 = 子树休眠"的可见性语义。
 *
 * 沿用本目录的约定：单个 test() + 花括号分块 + assertEqual/assert（见
 * `dockSlice.test.ts` 头部说明），面板注册走假面板（归一化会丢弃未注册面板）。
 */

import { beforeEach, test } from "vitest";

import reducer, {
    closeForm,
    composeFloats,
    createEmptyPanel,
    dockFormTo,
    floatForm,
    setDockLayout,
    syncRegisteredPanels,
} from "./dockSlice.ts";
import { normalizeDockLayout } from "./dockSchema.ts";
import {
    collectSubtreeRootIds,
    displayTitleOf,
    isPanelDetachable,
    panelMinSize,
    panelTitleOf,
} from "./dockPanel.ts";
import { collectDockedForms, collectVisibleForms, isFormVisible, isPanelForm, rootOfForm } from "./dockTree.ts";
import { registerPanel, resetPanelRegistryForTests } from "./panelRegistry.ts";
import { DOCK_LAYOUT_SCHEMA, DOCK_PANEL_FORM, type DockLayout } from "./dockTypes.ts";

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

function assert(condition: boolean, label: string): void {
    if (!condition) throw new Error(label);
}

/** 收窄到非空（`assert` 不参与 TS 控制流分析，测试里用它接住 find 的结果）。 */
function defined<T>(value: T | null | undefined, label: string): T {
    if (value === null || value === undefined) throw new Error(label);
    return value;
}

function shape(node: DockLayout["roots"][string] | null): string {
    if (!node) return "∅";
    if (node.t === "tabset") return `[${node.tabs.join(",")}]`;
    return `(${shape(node.a)}|${shape(node.b)})`;
}

const noop = () => null;
const translate = (key: string) => key;

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
        id: "fileBrowser",
        titleKey: "panel_io",
        component: noop,
        defaultWidth: 320,
        defaultHeight: 400,
        detachable: true,
        order: 30,
    });
    registerPanel({
        id: "notebook",
        titleKey: "common_notebook",
        component: noop,
        defaultWidth: 420,
        defaultHeight: 400,
        detachable: true,
        order: 40,
    });
}

beforeEach(() => {
    resetPanelRegistryForTests();
    registerFakes();
});

/** 出厂布局 + 一枚已填充的面板（fileBrowser + notebook 并排），返回 (state, 面板窗体 id)。 */
function seedPanel(): { state: ReturnType<typeof reducer>; panelFormId: string; rootId: string } {
    let state = reducer(undefined, { type: "@@INIT" });
    state = reducer(state, syncRegisteredPanels());
    state = reducer(state, createEmptyPanel({}));
    const panelFormId = defined(
        state.layout.order.find((id) => id.startsWith(`${DOCK_PANEL_FORM}:`)),
        "empty panel form created",
    );
    const rootId = defined(state.layout.forms[panelFormId].childRootId, "panel has a root id");
    // 空面板播种：tabsetId 在空根播种路径里不参与定位，只传根 id。
    state = reducer(
        state,
        dockFormTo({ formId: "fileBrowser", target: { kind: "tab", tabsetId: rootId, rootId } }),
    );
    return { state, panelFormId, rootId };
}

/** 面板根播种后首个标签组的 id（测试辅助）。 */
function rootIdOfFirstTabset(state: ReturnType<typeof reducer>, rootId: string): string {
    const tree = state.layout.roots[rootId];
    assert(tree !== undefined, "panel root seeded");
    return tree.t === "tabset" ? tree.id : tree.a.t === "tabset" ? tree.a.id : tree.b.id;
}

test("dock panel behaviors", () => {
    // ── 新建空面板：窗体记录 + 根条目缺席（空） ────────────────────
    {
        let state = reducer(undefined, { type: "@@INIT" });
        state = reducer(state, syncRegisteredPanels());
        state = reducer(state, createEmptyPanel({}));
        const panelFormId = state.layout.order.find((id) =>
            id.startsWith(`${DOCK_PANEL_FORM}:`),
        ) as string;
        const rootId = state.layout.forms[panelFormId].childRootId as string;
        assert(isFormVisible(state.layout, panelFormId), "empty panel is visible (it renders a well)");
        assert(
            state.layout.forms[panelFormId].floating === true,
            "a new empty panel floats by default",
        );
        assert(
            state.layout.forms[panelFormId].float?.anchor === "center",
            "default float position is the centered anchor (cascade offsets keep stacks apart)",
        );
        assertEqual(state.layout.roots[rootId], undefined, "empty panel has no root entry");
        assertEqual(state.activeFormId, panelFormId, "creation focuses the new panel");

        // 往空面板里停靠一个窗体 = 播种它的第一组。
        state = reducer(
            state,
            dockFormTo({
                formId: "fileBrowser",
                target: { kind: "tab", tabsetId: rootId, rootId },
            }),
        );
        assertEqual(
            shape(state.layout.roots[rootId]),
            "[fileBrowser]",
            "first drop seeds the panel root",
        );
        assert(rootOfForm(state.layout, "fileBrowser") === rootId, "member lives in the panel root");
    }

    // ── 可拆性推导：全体成员可拆才可拆；时间轴挡住整个面板 ────────
    {
        const { state, panelFormId } = seedPanel();
        const verdict = isPanelDetachable(state.layout, panelFormId);
        assertEqual(verdict, { ok: true, blockedBy: [] }, "all-detachable members → panel detachable");

        // 塞进不可拆的时间轴后，面板整体不可拆，且能点名阻挡者。
        const rootId = state.layout.forms[panelFormId].childRootId as string;
        const next = reducer(
            state,
            dockFormTo({ formId: "timeline", target: { kind: "tab", tabsetId: rootId, rootId } }),
        );
        const blocked = isPanelDetachable(next.layout, panelFormId);
        assertEqual(blocked.ok, false, "timeline blocks the whole panel");
        assertEqual(blocked.blockedBy, ["timeline"], "blockedBy names the offender");
    }

    // ── 标题派生：活动成员标题 + 数量后缀；显式重命名优先 ─────────
    {
        const { state, panelFormId, rootId } = seedPanel();
        assertEqual(
            panelTitleOf(state.layout, panelFormId, translate),
            "panel_io",
            "single member titles the panel (no count suffix)",
        );
        let next = reducer(
            state,
            dockFormTo({ formId: "notebook", target: { kind: "tab", tabsetId: rootId, rootId } }),
        );
        next = reducer(
            next,
            dockFormTo({ formId: "fileBrowser", target: { kind: "tab", tabsetId: rootId, rootId } }),
        );
        assertEqual(
            panelTitleOf(next.layout, panelFormId, translate),
            "panel_io (2)",
            "active member title with member count",
        );
        next = reducer(next, {
            type: "dock/renameForm",
            payload: { formId: panelFormId, title: "工作区" },
        } as never);
        assertEqual(
            displayTitleOf(next.layout, panelFormId, translate),
            "工作区",
            "explicit rename wins over derivation",
        );
    }

    // ── 浮窗组合：两个浮窗 → 一个面板；几何取并集；几何记忆保留 ──
    {
        let state = reducer(undefined, { type: "@@INIT" });
        state = reducer(state, syncRegisteredPanels());
        state = reducer(
            state,
            floatForm({ formId: "fileBrowser", geometry: { x: 100, y: 100, w: 300, h: 200 } }),
        );
        state = reducer(
            state,
            floatForm({ formId: "notebook", geometry: { x: 200, y: 150, w: 300, h: 200 } }),
        );
        state = reducer(
            state,
            composeFloats({
                sourceFormId: "fileBrowser",
                targetFormId: "notebook",
                zone: "center",
                rect: { x: 100, y: 100, w: 400, h: 250 },
            }),
        );
        const panelFormId = state.layout.order.find((id) =>
            id.startsWith(`${DOCK_PANEL_FORM}:`),
        ) as string;
        const panelForm = state.layout.forms[panelFormId];
        assert(panelForm.floating === true, "composed panel floats");
        assertEqual(
            state.layout.floatOrder.at(-1),
            panelFormId,
            "panel takes the target's z slot",
        );
        assertEqual(
            shape(state.layout.roots[panelForm.childRootId as string]),
            "[notebook,fileBrowser]",
            "both members live in the panel's tree",
        );
        assertEqual(
            panelForm.float,
            { x: 100, y: 100, w: 400, h: 250, anchor: null, anchorOffsetX: 0, anchorOffsetY: 0 },
            "panel adopts the union rect",
        );
        // 目标的几何记忆保留：日后拖出来回到它当初的大小。
        assertEqual(
            state.layout.forms.notebook.float,
            { x: 200, y: 150, w: 300, h: 200, anchor: null, anchorOffsetX: 0, anchorOffsetY: 0 },
            "member keeps its remembered float geometry",
        );
        // 目标已是面板：直接停入它的树，不新建面板。
        const before = state.layout.order.length;
        state = reducer(
            state,
            floatForm({ formId: "timeline", geometry: { x: 0, y: 0, w: 400, h: 300 } }),
        );
        state = reducer(
            state,
            composeFloats({
                sourceFormId: "timeline",
                targetFormId: panelFormId,
                zone: "center",
            }),
        );
        assertEqual(state.layout.order.length, before, "no extra panel is created");
        const childRoot = state.layout.forms[panelFormId].childRootId as string;
        assert(
            (state.layout.roots[childRoot] as { t: string; tabs?: string[] }).tabs?.includes(
                "timeline",
            ) === true,
            "source docks into the existing panel's tree",
        );
    }

    // ── 解散：停靠面板 → 内容摊平进宿主组；浮动面板 → 成员还回浮动层 ──
    {
        const seeded = seedPanel();
        let state = seeded.state;
        const parentRootId = "main";
        state = reducer(
            state,
            dockFormTo({
                formId: seeded.panelFormId,
                target: { kind: "tab", tabsetId: rootIdOfFirstTabset(state, "main") },
            }),
        );
        // 此时面板停靠在主根里；解散后它的内容进主根，面板记录消失。
        state = reducer(state, { type: "dock/dissolvePanel", payload: seeded.panelFormId } as never);
        assert(state.layout.forms[seeded.panelFormId] === undefined, "dissolved panel record removed");
        assert(
            collectVisibleForms(state.layout).includes("fileBrowser"),
            "panel contents survive the dissolve",
        );
        assert(parentRootId.length > 0, "parent root referenced");
    }

    // ── 自动解散：面板失去最后一个成员 → 面板消失 ─────────────────
    {
        let state = reducer(undefined, { type: "@@INIT" });
        state = reducer(state, syncRegisteredPanels());
        state = reducer(state, createEmptyPanel({}));
        const panelFormId = state.layout.order.find((id) =>
            id.startsWith(`${DOCK_PANEL_FORM}:`),
        ) as string;
        const rootId = state.layout.forms[panelFormId].childRootId as string;
        state = reducer(
            state,
            dockFormTo({ formId: "fileBrowser", target: { kind: "tab", tabsetId: rootId, rootId } }),
        );
        // 拖走唯一成员（floatForm）→ 面板根清空 → 面板按设置自动解散。
        state = reducer(state, floatForm({ formId: "fileBrowser" }));
        assert(
            state.layout.forms[panelFormId] === undefined,
            "panel auto-dissolves once its last member leaves",
        );
        assert(
            state.layout.forms.fileBrowser?.floating === true,
            "the leaving member is floating",
        );
    }

    // ── 关面板 = 子树休眠：成员不可见但记录与内容保留 ─────────────
    {
        const seeded = seedPanel();
        let state = seeded.state;
        // 面板是浮动窗体；把它停靠进主根再关闭，成员应随之休眠。
        state = reducer(
            state,
            dockFormTo({
                formId: seeded.panelFormId,
                target: { kind: "tab", tabsetId: rootIdOfFirstTabset(state, "main") },
            }),
        );
        state = reducer(state, closeForm(seeded.panelFormId));
        assertEqual(
            isFormVisible(state.layout, "fileBrowser"),
            false,
            "members of a closed panel are invisible",
        );
        assert(
            state.layout.roots[seeded.rootId] !== undefined,
            "panel root survives close for reopen",
        );
        assert(
            state.layout.forms[seeded.panelFormId] !== undefined,
            "panel record survives close",
        );
    }

    // ── 归一化：面板放行 / 孤儿根回收 / 跨根去重 / 环形引用破除 ──
    {
        // 面板窗体不在注册表里，但持有合法 childRootId → 放行；孤儿根 → 丢弃。
        const layout = normalizeDockLayout({
            schema: DOCK_LAYOUT_SCHEMA,
            roots: {
                main: {
                    t: "tabset",
                    id: "z1",
                    tabs: [`${DOCK_PANEL_FORM}:1`, "fileBrowser"],
                    active: `${DOCK_PANEL_FORM}:1`,
                },
                r1: {
                    t: "tabset",
                    id: "z2",
                    tabs: ["notebook"],
                    active: "notebook",
                },
                r_orphan: { t: "tabset", id: "z3", tabs: ["notebook"], active: "notebook" },
            },
            forms: {
                [`${DOCK_PANEL_FORM}:1`]: {
                    id: `${DOCK_PANEL_FORM}:1`,
                    panelId: DOCK_PANEL_FORM,
                    float: null,
                    floating: false,
                    childRootId: "r1",
                },
                fileBrowser: { id: "fileBrowser", panelId: "fileBrowser", float: null },
                notebook: { id: "notebook", panelId: "notebook", float: null },
            },
            order: [`${DOCK_PANEL_FORM}:1`, "fileBrowser", "notebook"],
            floatOrder: [],
        });
        assert(layout.forms[`${DOCK_PANEL_FORM}:1`] !== undefined, "panel form kept by normalize");
        assert(layout.roots.r1 !== undefined, "referenced panel root kept");
        assert(layout.roots.r_orphan === undefined, "orphan root dropped");
        assertEqual(
            shape(layout.roots.r1),
            "[notebook]",
            "panel tree normalized intact",
        );

        // 跨根去重：同一窗体出现在两棵树里 → 只保留先归一化的那处（主根优先）。
        const deduped = normalizeDockLayout({
            schema: DOCK_LAYOUT_SCHEMA,
            roots: {
                main: { t: "tabset", id: "z1", tabs: ["notebook"], active: "notebook" },
                r1: { t: "tabset", id: "z2", tabs: ["notebook"], active: "notebook" },
            },
            forms: {
                [`${DOCK_PANEL_FORM}:1`]: {
                    id: `${DOCK_PANEL_FORM}:1`,
                    panelId: DOCK_PANEL_FORM,
                    float: null,
                    floating: false,
                    childRootId: "r1",
                },
                notebook: { id: "notebook", panelId: "notebook", float: null },
            },
            order: [`${DOCK_PANEL_FORM}:1`, "notebook"],
            floatOrder: [],
        });
        assertEqual(shape(deduped.roots.r1), "∅", "duplicate membership removed from the panel root");

        // 环形引用：面板 A 的树里有 B，B 的树里又有 A → 环被打破，不致死循环。
        const cyclic = normalizeDockLayout({
            schema: DOCK_LAYOUT_SCHEMA,
            roots: {
                main: { t: "tabset", id: "z1", tabs: ["fileBrowser"], active: "fileBrowser" },
                rA: { t: "tabset", id: "z2", tabs: [`${DOCK_PANEL_FORM}:B`], active: `${DOCK_PANEL_FORM}:B` },
                rB: { t: "tabset", id: "z3", tabs: [`${DOCK_PANEL_FORM}:A`], active: `${DOCK_PANEL_FORM}:A` },
            },
            forms: {
                [`${DOCK_PANEL_FORM}:A`]: {
                    id: `${DOCK_PANEL_FORM}:A`,
                    panelId: DOCK_PANEL_FORM,
                    float: null,
                    floating: false,
                    childRootId: "rA",
                },
                [`${DOCK_PANEL_FORM}:B`]: {
                    id: `${DOCK_PANEL_FORM}:B`,
                    panelId: DOCK_PANEL_FORM,
                    float: null,
                    floating: false,
                    childRootId: "rB",
                },
                fileBrowser: { id: "fileBrowser", panelId: "fileBrowser", float: null },
            },
            order: [`${DOCK_PANEL_FORM}:A`, `${DOCK_PANEL_FORM}:B`, "fileBrowser"],
            floatOrder: [],
        });
        // 环被打破的判据：沿"面板 → 它的树里引用的面板"走必然终止
        // （回边被剪断，不要求两侧对称摘除 —— 只要不再递归即可）。
        let depth = 0;
        let cursor = cyclic.forms[`${DOCK_PANEL_FORM}:A`].childRootId as string | null;
        while (cursor && depth < 16) {
            const tree = cyclic.roots[cursor];
            if (!tree) break;
            const nextPanel = collectDockedForms(tree).find((id) => {
                const member = cyclic.forms[id];
                return member ? isPanelForm(member) : false;
            });
            cursor = nextPanel ? ((cyclic.forms[nextPanel].childRootId as string | null) ?? null) : null;
            depth += 1;
        }
        assert(depth < 16, "panel graph terminates after the cycle is broken");
    }

    // ── 子树根集合：独立窗口挂载排除与自引用防护共用同一份计算 ──
    {
        let state = reducer(undefined, { type: "@@INIT" });
        state = reducer(state, setDockLayout(normalizeDockLayout(state.layout)));
        const { state: withPanel, panelFormId } = seedPanel();
        state = withPanel;
        const rootId = state.layout.forms[panelFormId].childRootId as string;
        assertEqual(
            [...collectSubtreeRootIds(state.layout, rootId)].sort(),
            [rootId].sort(),
            "leaf-only panel has just its own root",
        );
        assert(panelMinSize(state.layout, panelFormId, true) >= 0, "panel min size resolves");
    }
});
