/*
 * 停靠状态切片。
 *
 * 所有几何/结构计算都在 `dockTree.ts` 与 `dockSchema.ts` 里以纯函数实现，
 * 这里的 reducer 只做三件事：调用纯函数、更新非布局字段（活动窗体、设置）、
 * 以及维护 `forms` / `order` / `floatOrder` 与树的一致性。
 *
 * 【为什么布局不放在 session 切片】布局与工程会话无关（不参与工程撤销/重做，
 * 也不写进工程文件），放进 session 只会让它跟着 `persistUiSettings` 的大
 * payload 一起流动，还会与工程切换耦合。独立切片后，只有它自己的订阅者会
 * 在布局变化时重渲染。
 */

import { createSlice, type PayloadAction } from "@reduxjs/toolkit";

import {
    closeFormInLayout,
    createDefaultDockLayout,
    ensureRegisteredPanels,
    findMainTabset,
    findVisibleFormForPanel,
    normalizeDockLayout,
    openPanelInLayout,
    type DockPlacement,
} from "./dockSchema";
import {
    findTabsetOfForm,
    findZone,
    moveForm,
    removeForm,
    removeZone,
    replaceTabset,
    setActiveTab,
    setSplitRatio,
    setTabsetCollapsed,
    splitRootWith,
    type DockInsertTarget,
} from "./dockTree";
import {
    DEFAULT_DOCK_SETTINGS,
    normalizeDockSettings,
    type DockSettings,
    type ResolvedDockSettings,
} from "./dockSettings";
import {
    type DockFloatGeometry,
    type DockGutterSizes,
    type DockLayout,
    type DockPreset,
} from "./dockTypes";
import { getPanel } from "./panelRegistry";

export interface DockState {
    layout: DockLayout;
    settings: ResolvedDockSettings;
    /** 最后获得焦点的窗体（键盘操作的目标）。 */
    activeFormId: string | null;
    /** 是否已从后端读过设置（避免用默认值覆盖磁盘上的布局）。 */
    hydrated: boolean;
    /**
     * 曾经可见过的窗体 id（单调增长）。
     *
     * 停靠宿主层据此决定"哪些面板需要挂载"：挂载过就永不卸载，因此关闭再打开
     * 是零成本的、状态全在；而**从未打开过**的面板不挂载，避免启动时为一个没
     * 打开的窗口构建富文本编辑器或 WebGL 上下文。
     *
     * 放在切片里而不是组件的 ref/state：它天然是"随会话演进的累积量"，而累积
     * 逻辑写在渲染期或 effect 里都会撞上 React Compiler 的规则（渲染期不得写
     * ref；effect 里同步 setState 会触发级联渲染）。
     */
    mountedFormIds: string[];
    /**
     * 「最大化当前窗体」暂存的原树。
     *
     * 刻意**不持久化**：最大化的语义是"临时看一眼"，重启后回到用户排好的
     * 布局才符合预期。放在这里而不是往布局树里加字段，是因为它不是一种排布，
     * 而是对排布的一次临时覆盖。
     */
    maximized: { tree: DockLayout["tree"] } | null;
}

const initialState: DockState = {
    layout: createDefaultDockLayout(),
    settings: DEFAULT_DOCK_SETTINGS,
    activeFormId: null,
    hydrated: false,
    mountedFormIds: [],
    maximized: null,
};

/**
 * 安装一份布局：归一化之后补上"已注册但布局里没有"的面板记录。
 *
 * 【为什么必须补】窗体 id 是持久化 JSON 的一部分（将来公开 API 与用户共享的
 * 预设都要引用它），因此它必须稳定且可预测。少了这一步，`normalizeDockLayout`
 * 会退回到 `createDefaultDockLayout()`（只含两个主窗体），于是用户打开文件
 * 浏览器时会**新建**一个 `fileBrowser:2` 而不是复用既有的 `fileBrowser` 记录 ——
 * 布局文件里就会出现同一面板的两个 id，外部引用随之失效。
 */
function installLayout(layout: DockLayout): DockLayout {
    return ensureRegisteredPanels(layout);
}

/** 浮动窗的默认几何：错开摆放，避免新浮窗完全叠在一起。 */
function defaultFloatGeometry(formId: string, cascadeIndex: number): DockFloatGeometry {
    const panel = getPanel(formId.split(":")[0]);
    return {
        x: 140 + cascadeIndex * 28,
        y: 120 + cascadeIndex * 28,
        w: panel?.defaultWidth ?? 420,
        h: panel?.defaultHeight ?? 320,
    };
}

const dockSlice = createSlice({
    name: "dock",
    initialState,
    reducers: {
        /** 从后端设置恢复：先归一化布局，再套用行为选项。 */
        hydrateDock(
            state,
            action: PayloadAction<{ settings?: DockSettings | null; layout?: unknown }>,
        ) {
            const raw = action.payload ?? {};
            state.layout = installLayout(normalizeDockLayout(raw.layout));
            state.settings = normalizeDockSettings(raw.settings);
            state.hydrated = true;
        },
        /** 面板注册完成后补齐窗体记录（内置面板在 App 模块加载期注册）。 */
        syncRegisteredPanels(state) {
            state.layout = ensureRegisteredPanels(state.layout);
        },
        /**
         * 登记"这些窗体已经需要挂载"。
         *
         * 单调增长：已挂载的窗体即使被关闭也留在集合里（重开时零成本），
         * 因此这里只做并集，不做删除。
         */
        markFormsMounted(state, action: PayloadAction<string[]>) {
            const known = new Set(state.mountedFormIds);
            const added = action.payload.filter((formId) => !known.has(formId));
            if (added.length > 0) state.mountedFormIds = [...state.mountedFormIds, ...added];
        },
        setDockSettings(state, action: PayloadAction<DockSettings>) {
            state.settings = normalizeDockSettings({ ...state.settings, ...action.payload });
        },
        setDockLayout(state, action: PayloadAction<unknown>) {
            state.layout = installLayout(normalizeDockLayout(action.payload));
        },
        resetDockLayout(state) {
            state.layout = ensureRegisteredPanels(createDefaultDockLayout());
            state.activeFormId = null;
            state.maximized = null;
        },
        /**
         * 最大化当前窗体 / 还原。
         *
         * 实现是"把当前窗体所在的标签组替换成整棵树"，而不是给它加宽高 ——
         * 这样最大化在任意嵌套布局下都成立，且不需要改动分割比例（用户还原后
         * 尺寸分毫不变）。
         */
        toggleMaximizeActive(state) {
            if (state.maximized) {
                state.layout = { ...state.layout, tree: state.maximized.tree };
                state.maximized = null;
                return;
            }
            const formId = state.activeFormId ?? findMainTabset(state.layout)?.active ?? null;
            if (!formId) return;
            const tabset = findTabsetOfForm(state.layout.tree, formId);
            if (!tabset) return;
            state.maximized = { tree: state.layout.tree };
            state.layout = {
                ...state.layout,
                tree: { ...tabset, active: formId, collapsed: false },
            };
        },
        /** 打开面板（复用已关闭的窗体记录，或新建）。 */
        openPanel(state, action: PayloadAction<{ panelId: string; placement?: DockPlacement }>) {
            const { panelId, placement } = action.payload;
            const existing = findVisibleFormForPanel(state.layout, panelId);
            state.layout = openPanelInLayout(state.layout, panelId, placement);
            state.activeFormId = existing ?? findVisibleFormForPanel(state.layout, panelId);
        },
        closeForm(state, action: PayloadAction<string>) {
            const formId = action.payload;
            state.layout = closeFormInLayout(state.layout, formId);
            if (state.activeFormId === formId) {
                state.activeFormId =
                    state.layout.order.find((id) => Boolean(state.layout.forms[id]?.float)) ??
                    findMainTabset(state.layout)?.active ??
                    null;
            }
        },
        focusForm(state, action: PayloadAction<string>) {
            const formId = action.payload;
            if (!state.layout.forms[formId]) return;
            state.activeFormId = formId;
            // 停靠窗体获得焦点时同步把它的标签组切到它，否则"焦点在 A、显示的是 B"。
            const tabset = state.layout.tree;
            state.layout = { ...state.layout, tree: setActiveTabEverywhere(tabset, formId) };
        },
        setActiveTabOf(state, action: PayloadAction<{ tabsetId: string; formId: string }>) {
            const { tabsetId, formId } = action.payload;
            state.layout = {
                ...state.layout,
                tree: setActiveTab(state.layout.tree, tabsetId, formId),
            };
            state.activeFormId = formId;
        },
        /** 停靠到指定落点（拖动结束、菜单"停靠到…"都走这里）。 */
        dockFormTo(
            state,
            action: PayloadAction<{ formId: string; target: DockInsertTarget; focus?: boolean }>,
        ) {
            const { formId, target, focus } = action.payload;
            const form = state.layout.forms[formId];
            if (!form) return;
            const forms = { ...state.layout.forms, [formId]: { ...form, float: null } };
            const tree = moveForm(state.layout.tree, formId, target);
            state.layout = {
                ...state.layout,
                forms,
                tree,
                floatOrder: state.layout.floatOrder.filter((id) => id !== formId),
            };
            if (focus !== false) state.activeFormId = formId;
        },
        /** 浮动（从树上摘除，记录几何）。 */
        floatForm(
            state,
            action: PayloadAction<{ formId: string; geometry?: Partial<DockFloatGeometry> }>,
        ) {
            const { formId, geometry } = action.payload;
            const form = state.layout.forms[formId];
            if (!form) return;
            const index = state.layout.floatOrder.length;
            const base = form.float ?? defaultFloatGeometry(formId, index);
            const next = { ...base, ...geometry };
            const tree = removeForm(state.layout.tree, formId) ?? state.layout.tree;

            state.layout = {
                ...state.layout,
                tree,
                forms: { ...state.layout.forms, [formId]: { ...form, float: next } },
                floatOrder: [...state.layout.floatOrder.filter((id) => id !== formId), formId],
            };
            state.activeFormId = formId;
        },
        /** 更新浮动几何（拖动/缩放结束、最大化切换）。 */
        setFloatGeometry(
            state,
            action: PayloadAction<{ formId: string; geometry: Partial<DockFloatGeometry> }>,
        ) {
            const { formId, geometry } = action.payload;
            const form = state.layout.forms[formId];
            if (!form?.float) return;
            state.layout = {
                ...state.layout,
                forms: {
                    ...state.layout.forms,
                    [formId]: { ...form, float: { ...form.float, ...geometry } },
                },
            };
        },
        /** 浮动层 z 序：点击浮窗置顶。 */
        raiseFloat(state, action: PayloadAction<string>) {
            const formId = action.payload;
            if (!state.layout.forms[formId]?.float) return;
            state.layout = {
                ...state.layout,
                floatOrder: [...state.layout.floatOrder.filter((id) => id !== formId), formId],
            };
            state.activeFormId = formId;
        },
        /** 在目标组某一侧拆出新组（拖动到边缘时用）。 */
        splitFormTo(
            state,
            action: PayloadAction<{
                formId: string;
                referenceFormId: string;
                side: "left" | "right" | "top" | "bottom";
            }>,
        ) {
            const { formId, referenceFormId, side } = action.payload;
            const tabset = findZone(
                state.layout.tree,
                findTabsetIdOf(state.layout, referenceFormId) ?? "",
            );
            if (!tabset || tabset.t !== "tabset") return;
            const tree = moveForm(state.layout.tree, formId, {
                kind: "split",
                tabsetId: tabset.id,
                side,
            });
            state.layout = {
                ...state.layout,
                tree,
                forms: {
                    ...state.layout.forms,
                    [formId]: { ...state.layout.forms[formId], float: null },
                },
                floatOrder: state.layout.floatOrder.filter((id) => id !== formId),
            };
            state.activeFormId = formId;
        },
        /** 合并到目标组（拖动到中央时用）。 */
        mergeFormInto(state, action: PayloadAction<{ formId: string; referenceFormId: string }>) {
            const { formId, referenceFormId } = action.payload;
            const tabsetId = findTabsetIdOf(state.layout, referenceFormId);
            if (!tabsetId) return;
            state.layout = {
                ...state.layout,
                tree: moveForm(state.layout.tree, formId, { kind: "tab", tabsetId }),
                forms: {
                    ...state.layout.forms,
                    [formId]: { ...state.layout.forms[formId], float: null },
                },
                floatOrder: state.layout.floatOrder.filter((id) => id !== formId),
            };
            state.activeFormId = formId;
        },
        /** 把整个标签组连同它的标签搬到工作区某一侧（拖动组内空白处时用）。 */
        moveTabsetToRootSide(
            state,
            action: PayloadAction<{ tabsetId: string; side: "left" | "right" | "top" | "bottom" }>,
        ) {
            const { tabsetId, side } = action.payload;
            const node = findZone(state.layout.tree, tabsetId);
            if (!node || node.t !== "tabset" || node.tabs.length === 0) return;
            const pruned = removeZone(state.layout.tree, tabsetId);
            if (!pruned) return;

            // 摘掉整组后在根部重建，等价于"这一组连同它的所有标签一起搬到边上"。
            const anchor = node.tabs[0];
            let tree = splitRootWith(pruned, anchor, side);
            const created = findTabsetOfForm(tree, anchor);
            if (created) {
                tree = replaceTabset(tree, created.id, {
                    ...created,
                    tabs: node.tabs,
                    active: node.active,
                });
                if (node.collapsed)
                    tree = setTabsetCollapsed(tree, created.id, true, node.collapsedPx);
            }
            state.layout = { ...state.layout, tree };
            state.activeFormId = node.active;
        },
        setSplitRatioOf(
            state,
            action: PayloadAction<{
                splitId: string;
                ratio: number;
                fixed?: { side: "a" | "b"; px: number } | null;
            }>,
        ) {
            const { splitId, ratio, fixed } = action.payload;
            state.layout = {
                ...state.layout,
                tree: setSplitRatio(state.layout.tree, splitId, ratio, fixed ?? null),
            };
        },
        toggleTabsetCollapsed(
            state,
            action: PayloadAction<{ tabsetId: string; collapsedPx?: number }>,
        ) {
            const { tabsetId, collapsedPx } = action.payload;
            const node = findZone(state.layout.tree, tabsetId);
            if (!node || node.t !== "tabset") return;
            state.layout = {
                ...state.layout,
                tree: setTabsetCollapsed(state.layout.tree, tabsetId, !node.collapsed, collapsedPx),
            };
        },
        setGutterSize(state, action: PayloadAction<{ key: keyof DockGutterSizes; px: number }>) {
            const { key, px } = action.payload;
            state.layout = {
                ...state.layout,
                gutters: { ...state.layout.gutters, [key]: Math.round(px) },
            };
        },
        renameForm(state, action: PayloadAction<{ formId: string; title: string }>) {
            const { formId, title } = action.payload;
            const form = state.layout.forms[formId];
            if (!form) return;
            const trimmed = title.trim();
            state.layout = {
                ...state.layout,
                forms: {
                    ...state.layout.forms,
                    [formId]: { ...form, title: trimmed || undefined },
                },
            };
        },
        /** 把当前排布存为命名预设。 */
        saveDockPreset(state, action: PayloadAction<string>) {
            const name = action.payload.trim();
            if (!name) return;
            const preset: DockPreset = {
                name,
                tree: state.layout.tree,
                forms: state.layout.forms,
                order: state.layout.order,
                floatOrder: state.layout.floatOrder,
                gutters: state.layout.gutters,
                createdAtMs: Date.now(),
            };
            state.layout = {
                ...state.layout,
                presets: { ...state.layout.presets, [name]: preset },
                activePreset: name,
            };
        },
        applyDockPreset(state, action: PayloadAction<string>) {
            const preset = state.layout.presets?.[action.payload];
            if (!preset) return;
            const next = normalizeDockLayout({
                schema: state.layout.schema,
                tree: preset.tree,
                forms: preset.forms,
                order: preset.order,
                floatOrder: preset.floatOrder,
                gutters: preset.gutters,
                presets: state.layout.presets,
                activePreset: preset.name,
            });
            state.layout = ensureRegisteredPanels(next);
        },
        deleteDockPreset(state, action: PayloadAction<string>) {
            const name = action.payload;
            if (!state.layout.presets?.[name]) return;
            const presets = { ...state.layout.presets };
            delete presets[name];
            state.layout = {
                ...state.layout,
                presets,
                activePreset: state.layout.activePreset === name ? null : state.layout.activePreset,
            };
        },
    },
});

/** 在整棵树上把包含 formId 的标签组切到该窗体。 */
function setActiveTabEverywhere(node: DockLayout["tree"], formId: string): DockLayout["tree"] {
    if (node.t === "tabset") {
        return node.tabs.includes(formId) ? { ...node, active: formId } : node;
    }
    return {
        ...node,
        a: setActiveTabEverywhere(node.a, formId),
        b: setActiveTabEverywhere(node.b, formId),
    };
}

function findTabsetIdOf(layout: DockLayout, formId: string): string | null {
    const walk = (node: DockLayout["tree"]): string | null => {
        if (node.t === "tabset") return node.tabs.includes(formId) ? node.id : null;
        return walk(node.a) ?? walk(node.b);
    };
    return walk(layout.tree);
}

export const {
    hydrateDock,
    syncRegisteredPanels,
    markFormsMounted,
    toggleMaximizeActive,
    setDockSettings,
    setDockLayout,
    resetDockLayout,
    openPanel,
    closeForm,
    focusForm,
    setActiveTabOf,
    dockFormTo,
    floatForm,
    setFloatGeometry,
    raiseFloat,
    splitFormTo,
    mergeFormInto,
    moveTabsetToRootSide,
    setSplitRatioOf,
    toggleTabsetCollapsed,
    setGutterSize,
    renameForm,
    saveDockPreset,
    applyDockPreset,
    deleteDockPreset,
} = dockSlice.actions;

export default dockSlice.reducer;
