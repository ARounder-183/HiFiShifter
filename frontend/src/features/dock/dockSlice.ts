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
    placeForm,
    type DockPlacement,
    normalizeFloatMode,
    normalizeTabPosition,
} from "./dockSchema";
import {
    addFormToTabset,
    collectDockedForms,
    collectTabsets,
    dockForm,
    findTabsetOfForm,
    findZone,
    isPanelForm,
    makeTabset,
    nextRootId,
    nextZoneIdInLayout,
    pruneTree,
    removeForm,
    removeZone,
    replaceTabset,
    replaceZone,
    rootOfForm,
    setActiveTab,
    setSplitRatio,
    setTabsetCollapsed,
    splitRootWith,
    splitTabsetWith,
    withRoot,
    zoneIdAllocatorForLayout,
    type DockInsertTarget,
} from "./dockTree";
import {
    collectSubtreeRootIds,
    makePanelForm,
    ownerOfRoot,
    panelDepth,
    PANEL_DEFAULT_HEIGHT,
    PANEL_DEFAULT_WIDTH,
    synthesizePanelDefinition,
} from "./dockPanel";
import {
    DEFAULT_DOCK_SETTINGS,
    normalizeDockSettings,
    type DockSettings,
    type ResolvedDockSettings,
} from "./dockSettings";
import {
    type DockDropZone,
    type DockFloatGeometry,
    type DockGutterSizes,
    type DockLayout,
    type DockNode,
    type DockPreset,
    type DockFloatMode,
    type DockTabPosition,
    type DockTabsetNode,
    DOCK_PANEL_FORM,
    MAIN_ROOT_ID,
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
     * 而是对排布的一次临时覆盖。按**根**记（`rootId` + 原树）：在面板里最大化
     * 只应铺满那个面板，而不是整片主工作区。
     */
    maximized: { rootId: string; tree: DockNode } | null;
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
            // 重置 = 回到出厂排布。**预设是用户的资产，不属于"被重置的排布"**
            // —— 确认对话框承诺"已保存的预设会保留"，重置后用户仍能一键回到
            // 自己的排布。activePreset 同步清空：此刻是出厂布局，不属于任何
            // 预设；保留旧值会让布局菜单的勾选错误地暗示某预设仍然生效。
            state.layout = ensureRegisteredPanels({
                ...createDefaultDockLayout(),
                presets: state.layout.presets ?? {},
                activePreset: null,
            });
            state.activeFormId = null;
            state.maximized = null;
        },
        /**
         * 最大化当前窗体 / 还原。
         *
         * 实现是"把当前窗体所在的标签组替换成**它所在那棵根**"，而不是给它加宽高
         * —— 这样最大化在任意嵌套布局下都成立，且不需要改动分割比例（用户还原后
         * 尺寸分毫不变）。作用域是根：在面板里最大化只铺满那个面板。
         */
        toggleMaximizeActive(state) {
            if (state.maximized) {
                const { rootId, tree: restoredTree } = state.maximized;
                const temporaryTree = state.layout.roots[rootId] ?? null;
                state.maximized = null;
                // 还原时补回"最大化期间打开"的面板：它们在**临时树**上可见，换回
                // 原树后既不在树上也不浮动（窗体记录还在，floating=false）—— 表现
                // 为面板凭空消失。判据必须是"在临时树里出现过"，而不是"不在原树
                // 里"：后者会把一直关闭着的面板记录也一并打开。走"并入主组"的
                // 既有插入路径（与 center 打开落点同源）。
                const roots = { ...state.layout.roots, [rootId]: restoredTree };
                let layout: DockLayout = { ...state.layout, roots };
                if (temporaryTree) {
                    for (const formId of collectDockedForms(temporaryTree)) {
                        const form = layout.forms[formId];
                        if (!form || form.floating) continue;
                        const stillDocked = Object.values(layout.roots).some((tree) =>
                            findTabsetOfForm(tree, formId),
                        );
                        if (stillDocked) continue;
                        layout = {
                            ...layout,
                            roots: {
                                ...layout.roots,
                                [MAIN_ROOT_ID]: placeForm(layout, formId, { side: "center" }),
                            },
                        };
                    }
                }
                state.layout = layout;
                return;
            }
            const formId = state.activeFormId ?? findMainTabset(state.layout)?.active ?? null;
            if (!formId) return;
            const rootId = rootOfForm(state.layout, formId) ?? MAIN_ROOT_ID;
            const tree = state.layout.roots[rootId];
            const tabset = tree ? findTabsetOfForm(tree, formId) : null;
            if (!tree || !tabset) return;
            state.maximized = { rootId, tree };
            state.layout = {
                ...state.layout,
                roots: {
                    ...state.layout.roots,
                    [rootId]: { ...tabset, active: formId, collapsed: false },
                },
            };
        },
        /** 打开面板（复用已关闭的窗体记录，或新建）。 */
        openPanel(
            state,
            action: PayloadAction<{
                panelId: string;
                placement?: DockPlacement;
                /** 本次打开的指定浮窗几何（命令层按触发控件算出）。 */
                float?: DockFloatGeometry | null;
            }>,
        ) {
            const { panelId, placement, float } = action.payload;
            const existing = findVisibleFormForPanel(state.layout, panelId);
            state.layout = openPanelInLayout(state.layout, panelId, placement, float ?? null);
            state.activeFormId = existing ?? findVisibleFormForPanel(state.layout, panelId);
        },
        closeForm(state, action: PayloadAction<string>) {
            const formId = action.payload;
            updateLayout(state, closeFormInLayout(state.layout, formId));
            if (state.activeFormId === formId || !state.layout.forms[state.activeFormId ?? ""]) {
                state.activeFormId =
                    state.layout.order.find((id) => state.layout.forms[id]?.floating === true) ??
                    findMainTabset(state.layout)?.active ??
                    null;
            }
        },
        focusForm(state, action: PayloadAction<string>) {
            const formId = action.payload;
            if (!state.layout.forms[formId]) return;
            state.activeFormId = formId;
            // 停靠窗体获得焦点时同步把它的标签组切到它，否则"焦点在 A、显示的是 B"。
            // 多根之后要在每一棵可达的树上同步（面板里的标签组也在其中）。
            let roots = state.layout.roots;
            let changed = false;
            for (const [rootId, tree] of Object.entries(roots)) {
                const next = setActiveTabEverywhere(tree, formId);
                if (next !== tree) {
                    roots = { ...roots, [rootId]: next };
                    changed = true;
                }
            }
            if (changed) state.layout = { ...state.layout, roots };
        },
        setActiveTabOf(state, action: PayloadAction<{ tabsetId: string; formId: string }>) {
            const { tabsetId, formId } = action.payload;
            const rootId = rootIdOfZone(state.layout, tabsetId);
            if (!rootId) return;
            state.layout = withRoot(state.layout, rootId, (tree) =>
                setActiveTab(tree, tabsetId, formId),
            );
            state.activeFormId = formId;
        },
        /** 停靠到指定落点（拖动结束、菜单"停靠到…"都走这里）。 */
        dockFormTo(
            state,
            action: PayloadAction<{ formId: string; target: DockInsertTarget; focus?: boolean }>,
        ) {
            const { formId, target, focus } = action.payload;
            if (!dockFormInto(state, formId, target)) return;
            if (focus !== false) state.activeFormId = formId;
        },
        /**
         * 浮动（从所在的树上摘除，记录几何）。
         *
         * 主根不允许被清空（见下）；面板根被清空是合法的 —— 面板留在原处显示
         * 占位井，随后按设置走"失去最后一个成员 → 自动解散"的收尾。
         */
        floatForm(
            state,
            action: PayloadAction<{ formId: string; geometry?: Partial<DockFloatGeometry> }>,
        ) {
            const { formId, geometry } = action.payload;
            const form = state.layout.forms[formId];
            if (!form) return;
            const sourceRootId = rootOfForm(state.layout, formId);
            let roots = state.layout.roots;
            if (sourceRootId) {
                const pruned = removeForm(roots[sourceRootId], formId);
                if (pruned === null) {
                    // 树上只剩它自己：摘除会得到 null（见 removeForm 的约定）。若回退到
                    // 旧树继续浮动，同一个窗体会同时出现在树上和浮层里（两个宿主抢一个
                    // 面板），所以最后一个停靠窗体不允许浮走。面板根没有这条限制 ——
                    // 摘空即空面板，删条目即可。
                    if (sourceRootId === MAIN_ROOT_ID) return;
                    roots = { ...roots };
                    delete roots[sourceRootId];
                } else {
                    roots = { ...roots, [sourceRootId]: pruned };
                }
            }
            const index = state.layout.floatOrder.length;
            const base = form.float ?? defaultFloatGeometry(formId, index);
            // 显式给了位置（拖拽拆出、菜单指定）就**清除锚点**：用户/调用方已经
            // 决定了位置，不该再被"右下角"这个语义覆盖。锚点偏移随锚点一起清。
            // 只给了 w/h（resize 路径）则**保留锚点**：位置仍由锚点语义表达，
            // 见 dockApi::detachFormToWindow 的注释。
            const setsPosition =
                geometry != null && (geometry.x !== undefined || geometry.y !== undefined);
            const next = {
                ...base,
                ...geometry,
                anchor: setsPosition ? null : (base.anchor ?? null),
                anchorOffsetX: setsPosition ? 0 : base.anchorOffsetX,
                anchorOffsetY: setsPosition ? 0 : base.anchorOffsetY,
            };

            updateLayout(state, {
                ...state.layout,
                roots,
                forms: {
                    ...state.layout.forms,
                    [formId]: { ...form, float: next, floating: true },
                },
                floatOrder: [...state.layout.floatOrder.filter((id) => id !== formId), formId],
            });
            state.activeFormId = formId;
        },
        /** 更新浮动几何（拖动/缩放结束、最大化切换）。 */
        setFloatGeometry(
            state,
            action: PayloadAction<{ formId: string; geometry: Partial<DockFloatGeometry> }>,
        ) {
            const { formId, geometry } = action.payload;
            const form = state.layout.forms[formId];
            if (!form?.floating || !form.float) return;
            // 用户手动移动/缩放（或最小化/最大化）之后，锚点即失效：此后的位置由
            // 这些具体几何决定，而不是"右下角"这个语义。锚点偏移是锚点的一部分，
            // 必须一起清掉（否则它会在下一次重新挂上锚点时凭空生效）。
            // `Object.assign` 而非展开：前者保留"基础几何已提供全部必填字段"的类型，
            // 后者会因为 `Partial` 而把结果推成可选字段。
            state.layout = {
                ...state.layout,
                forms: {
                    ...state.layout.forms,
                    [formId]: {
                        ...form,
                        float: Object.assign({}, form.float, geometry, {
                            anchor: null,
                            anchorOffsetX: 0,
                            anchorOffsetY: 0,
                        }),
                    },
                },
            };
        },
        /** 浮动层 z 序：点击浮窗置顶。 */
        raiseFloat(state, action: PayloadAction<string>) {
            const formId = action.payload;
            if (!state.layout.forms[formId]?.floating) return;
            state.layout = {
                ...state.layout,
                floatOrder: [...state.layout.floatOrder.filter((id) => id !== formId), formId],
            };
            state.activeFormId = formId;
        },
        /** 在目标组某一侧拆出新组（拖动到边缘时用）。目标根从参照窗体解析。 */
        splitFormTo(
            state,
            action: PayloadAction<{
                formId: string;
                referenceFormId: string;
                side: "left" | "right" | "top" | "bottom";
            }>,
        ) {
            const { formId, referenceFormId, side } = action.payload;
            const located = locateTabsetOfForm(state.layout, referenceFormId);
            if (!located) return;
            dockFormInto(state, formId, {
                kind: "split",
                tabsetId: located.tabsetId,
                rootId: located.rootId,
                side,
            });
        },
        /** 合并到目标组（拖动到中央时用）。目标根从参照窗体解析。 */
        mergeFormInto(state, action: PayloadAction<{ formId: string; referenceFormId: string }>) {
            const { formId, referenceFormId } = action.payload;
            const located = locateTabsetOfForm(state.layout, referenceFormId);
            if (!located) return;
            dockFormInto(state, formId, {
                kind: "tab",
                tabsetId: located.tabsetId,
                rootId: located.rootId,
            });
        },
        /** 把整个标签组连同它的标签搬到**所在根**的某一侧（拖动组内空白处时用）。 */
        moveTabsetToRootSide(
            state,
            action: PayloadAction<{ tabsetId: string; side: "left" | "right" | "top" | "bottom" }>,
        ) {
            const { tabsetId, side } = action.payload;
            const rootId = rootIdOfZone(state.layout, tabsetId);
            const node = rootId ? findZone(state.layout.roots[rootId], tabsetId) : null;
            if (!rootId || !node || node.t !== "tabset" || node.tabs.length === 0) return;
            const baseTree = state.layout.roots[rootId];
            const pruned = removeZone(baseTree, tabsetId);
            if (!pruned) {
                // 整棵根就是这一组：把组内容重建为一侧的新组（主根不允许清空，
                // 但"只剩一组再搬到边上"等价于原样保留 —— 直接返回）。
                if (rootId !== MAIN_ROOT_ID) return;
                return;
            }

            // 摘掉整组后在根部重建，等价于"这一组连同它的所有标签一起搬到边上"。
            const allocate = zoneIdAllocatorForLayout(state.layout);
            const anchor = node.tabs[0];
            let tree = splitRootWith(pruned, anchor, side, allocate);
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
            updateLayout(state, {
                ...state.layout,
                roots: { ...state.layout.roots, [rootId]: tree },
            });
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
            const rootId = rootIdOfZone(state.layout, splitId);
            if (!rootId) return;
            state.layout = withRoot(state.layout, rootId, (tree) =>
                setSplitRatio(tree, splitId, ratio, fixed ?? null),
            );
        },
        toggleTabsetCollapsed(
            state,
            action: PayloadAction<{ tabsetId: string; collapsedPx?: number }>,
        ) {
            const { tabsetId, collapsedPx } = action.payload;
            const rootId = rootIdOfZone(state.layout, tabsetId);
            const node = rootId ? findZone(state.layout.roots[rootId], tabsetId) : null;
            if (!rootId || !node || node.t !== "tabset") return;
            state.layout = withRoot(state.layout, rootId, (tree) =>
                setTabsetCollapsed(tree, tabsetId, !node.collapsed, collapsedPx),
            );
        },
        setGutterSize(state, action: PayloadAction<{ key: keyof DockGutterSizes; px: number }>) {
            const { key, px } = action.payload;
            state.layout = {
                ...state.layout,
                gutters: { ...state.layout.gutters, [key]: Math.round(px) },
            };
        },
        /**
         * 设置窗体的浮动形态（进程内浮层 / 独立操作系统窗口）。
         *
         * 【为什么与 `floatForm` 分开】`floatForm` 表达"浮起来"，这里表达"浮在哪里"。
         * 独立窗口的创建/关闭是副作用（Tauri 窗口），由 `dockApi` 负责，reducer 只
         * 记录意图 —— 布局因此可以在窗口创建失败时干净地回退。
         */
        setFormFloatMode(
            state,
            action: PayloadAction<{ formId: string; floatMode: DockFloatMode }>,
        ) {
            const { formId, floatMode } = action.payload;
            const form = state.layout.forms[formId];
            if (!form) return;
            state.layout = {
                ...state.layout,
                forms: {
                    ...state.layout.forms,
                    [formId]: { ...form, floatMode: normalizeFloatMode(floatMode) },
                },
            };
        },
        /** 标签行位置（布局级偏好，随布局一起持久化）。 */
        setTabPosition(state, action: PayloadAction<DockTabPosition>) {
            state.layout = { ...state.layout, tabPosition: normalizeTabPosition(action.payload) };
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
        /**
         * 写入面板的自有配置。
         *
         * 【为什么需要它】`DockPanelProps.props` 一直被注释宣称为「未来 API
         * 面板可直接使用」，`normalizeDockLayout` 也早已持久化它、宿主也早已
         * 把它传给面板组件 —— 但**没有任何 action 能写入**。这条存储通道此前
         * 只有读的一半，面板拿到的永远是空对象。
         *
         * 第三方面板要保存自己的配置（列宽、过滤条件、展开状态）必须走这里；
         * 内置面板若要持久化面板级配置也应改用它，而不是往全局 settings 里塞。
         *
         * 浅合并：调用方传部分字段即可，未提及的键保持不变。要删除某个键，
         * 显式传 `undefined`（JSON 序列化时会被丢弃）。
         */
        setFormProps(
            state,
            action: PayloadAction<{ formId: string; props: Record<string, unknown> }>,
        ) {
            const { formId, props } = action.payload;
            const form = state.layout.forms[formId];
            if (!form) return;
            const merged: Record<string, unknown> = { ...form.props };
            for (const [key, value] of Object.entries(props)) {
                if (value === undefined) delete merged[key];
                else merged[key] = value;
            }
            state.layout = {
                ...state.layout,
                forms: {
                    ...state.layout.forms,
                    [formId]: { ...form, props: merged },
                },
            };
        },
        /** 把当前排布存为命名预设。 */
        saveDockPreset(state, action: PayloadAction<string>) {
            const name = action.payload.trim();
            if (!name) return;
            const preset: DockPreset = {
                name,
                roots: state.layout.roots,
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
                roots: preset.roots,
                forms: preset.forms,
                order: preset.order,
                floatOrder: preset.floatOrder,
                gutters: preset.gutters,
                presets: state.layout.presets,
                activePreset: preset.name,
            });
            state.layout = ensureRegisteredPanels(next);
            // 预设替换一切：旧的活动窗体若已不存在，清空以免指向幽灵记录。
            if (!state.layout.forms[state.activeFormId ?? ""]) state.activeFormId = null;
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
        /**
         * 新建一个**空面板**（不含任何窗体）。
         *
         * 「面板」是容器窗体：先分配窗体记录（`panelId` 为保留字）与它自己的布局
         * 根（此时无条目 = 空），再按落点把它放进目标根 —— 与打开普通面板共用
         * `placeForm` 的全部落点语义。空面板的根条目在放入第一个窗体时才创建，
         * 因此"新建空面板"不会在根表里留下任何待清理的残留。
         */
        createEmptyPanel(
            state,
            action: PayloadAction<{
                rootId?: string;
                /** 显式落点 → 停靠进该根。缺省 = 浮动（见下）。 */
                placement?: DockPlacement;
                /** 显式浮动几何（命令层按触发控件算出）；缺省 = 居中锚点。 */
                float?: DockFloatGeometry | null;
                /** 面板窗体的显示标题（缺省走 i18n / 内容派生）。 */
                title?: string;
            }>,
        ) {
            const { rootId, placement, float, title } = action.payload;
            const targetRootId = rootId ?? MAIN_ROOT_ID;

            const layout = state.layout;
            const newFormId = `${DOCK_PANEL_FORM}:${nextFormSuffix(layout, DOCK_PANEL_FORM)}`;
            const newRootId = nextRootId(layout);
            const form = makePanelForm(newFormId, newRootId);
            if (title && title.trim()) form.title = title.trim();

            if (placement) {
                // 显式落点：停靠进指定根（命令式入口保留这条通道，菜单默认不走）。
                // 嵌套深度上限：往深层面板里再塞面板必须先过这一关。
                const placedRootId = placement.rootId ?? targetRootId;
                if (
                    depthOfRoot(layout, placedRootId) + 1 >
                    Math.max(1, state.settings.maxPanelDepth)
                ) {
                    return;
                }
                const base: DockLayout = {
                    ...layout,
                    forms: { ...layout.forms, [newFormId]: form },
                    order: [...layout.order, newFormId],
                };
                const tree = placeForm(base, newFormId, placement);
                const pruned = pruneTree(tree);
                state.layout = {
                    ...base,
                    roots: {
                        ...base.roots,
                        [placedRootId]:
                            pruned ??
                            base.roots[placedRootId] ??
                            makeTabset(nextZoneIdInLayout(base), newFormId),
                    },
                };
                state.activeFormId = newFormId;
                return;
            }

            // ── 默认：浮动 ──────────────────────────────────────────
            // 新建空面板是"接下来要往里装窗体"的动作，让它浮在主窗口正中成为
            // 眼前的焦点（与外观设置同一条锚点语义），而不是挤进布局里占一格。
            // 位置交给**居中锚点**按当前视口推导（用户移动后锚点自然清除）；
            // 级联偏移让连续新建的面板错开 28px，不会完全叠在一起。
            const cascadeIndex = layout.floatOrder.length;
            const geometry: DockFloatGeometry = float ?? {
                x: 0,
                y: 0,
                w: PANEL_DEFAULT_WIDTH,
                h: PANEL_DEFAULT_HEIGHT,
                anchor: "center",
                anchorMarginPx: 24,
                anchorOffsetX: cascadeIndex * 28,
                anchorOffsetY: cascadeIndex * 28,
            };
            state.layout = {
                ...layout,
                forms: {
                    ...layout.forms,
                    [newFormId]: { ...form, float: geometry, floating: true },
                },
                order: [...layout.order, newFormId],
                floatOrder: [...layout.floatOrder, newFormId],
            };
            state.activeFormId = newFormId;
        },
        /**
         * 解散面板：用面板自己的树替换它在父容器中的位置。
         *
         * 组合的对偶操作。三种情形各有归宿：停靠在标签组里 → 内容摊平进那个组
         * （内容是分割树时以分割包裹其余标签）；停靠在分割一侧 → 该侧直接换成
         * 面板的树；浮动 → 成员按各自记住的浮窗几何还回浮动层。面板窗体记录与
         * 它的根条目一并删除 —— 解散就是"当作没组合过"。
         */
        dissolvePanel(state, action: PayloadAction<string>) {
            dissolvePanelForm(state, action.payload);
            state.activeFormId = state.layout.forms[state.activeFormId ?? ""]
                ? state.activeFormId
                : null;
        },
        /**
         * 浮窗 ⇄ 浮窗组合。
         *
         * 目标已是面板 → 源窗体按落点停入它的树（不新建面板，避免意外的深嵌套）；
         * 目标是叶窗体 → **提升**：新建面板把两者一起装进去。两边的浮窗几何记忆
         * 都保留 —— 日后把成员从这个面板里拖出来，会回到它当初的大小与位置
         * （这正是 `DockForm.float` 的记忆语义）。
         *
         * `rect` 是命令层按视口夹紧过的**并集矩形**（reducer 保持纯函数，不读
         * window）；缺省退化取目标浮窗的几何。
         */
        composeFloats(
            state,
            action: PayloadAction<{
                sourceFormId: string;
                targetFormId: string;
                zone: DockDropZone;
                rect?: { x: number; y: number; w: number; h: number } | null;
            }>,
        ) {
            const { sourceFormId, targetFormId, zone, rect } = action.payload;
            const source = state.layout.forms[sourceFormId];
            const target = state.layout.forms[targetFormId];
            if (!source?.floating || !target?.floating) return;
            if (sourceFormId === targetFormId) return;

            if (isPanelForm(target) && target.childRootId) {
                // 深度上限：往面板里塞面板（源是面板时）要先过这一关。
                if (
                    isPanelForm(source) &&
                    !panelDepthFits(state.layout, target.childRootId, state.settings.maxPanelDepth)
                ) {
                    return;
                }
                const rootId = target.childRootId;
                const tree = state.layout.roots[rootId];
                const allocate = zoneIdAllocatorForLayout(state.layout);
                const nextTree = tree
                    ? dockForm(
                          tree,
                          sourceFormId,
                          zone === "center"
                              ? { kind: "tab", tabsetId: mainTabsetIdOfTree(tree), rootId }
                              : { kind: "root", rootId, side: dropZoneToSideKind(zone) },
                          allocate,
                      )
                    : makeTabset(allocate(), sourceFormId);
                const pruned = pruneTree(nextTree);
                updateLayout(state, {
                    ...state.layout,
                    roots: { ...state.layout.roots, [rootId]: pruned ?? nextTree },
                    forms: {
                        ...state.layout.forms,
                        [sourceFormId]: { ...source, floating: false },
                    },
                    floatOrder: state.layout.floatOrder.filter((id) => id !== sourceFormId),
                });
                state.activeFormId = sourceFormId;
                return;
            }

            // ── 提升：目标叶窗体与源窗体共同组成一个新面板 ──────────────
            if (
                isPanelForm(source) &&
                !panelDepthFits(state.layout, null, state.settings.maxPanelDepth)
            ) {
                return;
            }
            const layout = state.layout;
            const newFormId = `${DOCK_PANEL_FORM}:${nextFormSuffix(layout, DOCK_PANEL_FORM)}`;
            const newRootId = nextRootId(layout);
            const allocate = zoneIdAllocatorForLayout(layout);
            const seedId = allocate();
            let tree: DockNode = makeTabset(seedId, targetFormId);
            tree =
                zone === "center"
                    ? addFormToTabset(tree, seedId, sourceFormId)
                    : splitTabsetWith(
                          tree,
                          seedId,
                          sourceFormId,
                          dropZoneToSideKind(zone),
                          allocate,
                      );
            const pruned = pruneTree(tree);
            if (!pruned) return;

            const union = rect ?? unionRects(source.float, target.float);
            // 浮动态渲染以 `float` 为前提（DockFloatingLayer 的空几何守卫）：
            // 两个来源都缺失（理论不可达）时给默认几何。
            const panelFloat = union ?? { x: 140, y: 120, w: 720, h: 480 };
            const forms = {
                ...layout.forms,
                [newFormId]: {
                    ...makePanelForm(newFormId, newRootId),
                    float: {
                        ...panelFloat,
                        anchor: null,
                        anchorOffsetX: 0,
                        anchorOffsetY: 0,
                    },
                    floating: true,
                },
                [targetFormId]: { ...target, floating: false },
                [sourceFormId]: { ...source, floating: false },
            };
            // 新面板顶替**目标**在 z 序里的位置：视觉上"目标窗体变成了一组"，
            // z 序不应跳动。
            const targetIndex = layout.floatOrder.indexOf(targetFormId);
            const floatOrder = layout.floatOrder.filter(
                (id) => id !== sourceFormId && id !== targetFormId,
            );
            floatOrder.splice(Math.min(Math.max(0, targetIndex), floatOrder.length), 0, newFormId);
            state.layout = {
                ...layout,
                roots: { ...layout.roots, [newRootId]: pruned },
                forms,
                order: layout.order.includes(newFormId)
                    ? layout.order
                    : [...layout.order, newFormId],
                floatOrder,
            };
            state.activeFormId = sourceFormId;
        },
    },
});

/**
 * **停靠的唯一出口**。
 *
 * 三件事必须同时成立，否则用户就会看到"停靠之后窗口没被正确展示"：
 * 1. 窗体真的落到了目标根的布局树上（`dockForm` 负责搬运 + 兜底 + 展开折叠组）；
 * 2. `floating` 被清掉（否则它同时"在树上"又"在浮动"，两个槽位抢同一个宿主，
 *    内容会落到其中一边，另一边空白）；
 * 3. `float` 几何**保留**（那是"下次拆下来用多大"的记忆，与"此刻是否浮动"无关）。
 *
 * 多根之后新增两条防线：
 * 4. **跨根搬运**——源根与目标根不是同一棵树时，先插目标、再从源根摘除（两个
 *    动作在一次 dispatch 里完成，中间态不可见）；
 * 5. **自引用与深度防护**——面板不能被塞进它自己的子树（渲染是递归的），嵌套
 *    深度受用户设置约束。
 *
 * @returns 是否真的停靠了（窗体不存在时返回 false）。
 */
function dockFormInto(state: DockState, formId: string, target: DockInsertTarget): boolean {
    const form = state.layout.forms[formId];
    if (!form) return false;
    const rootId = target.rootId ?? MAIN_ROOT_ID;

    // 防线一：面板不能落进自己的子树（主根除外——主根永远不可能属于面板）。
    if (isPanelForm(form) && form.childRootId) {
        const forbidden = collectSubtreeRootIds(state.layout, form.childRootId);
        if (forbidden.has(rootId)) return false;
    }
    // 防线二：嵌套深度上限（交互语义由设置表达，这里是 reducer 层的统一闸门）。
    if (isPanelForm(form) && !panelDepthFits(state.layout, rootId, state.settings.maxPanelDepth)) {
        return false;
    }
    // 防线三：独立窗口承载的面板只收"可拆"的成员 —— 整个面板已经在另一个
    // JS 上下文里渲染了，混进不可拆的窗体等于把那条约束击穿。
    if (collectOsWindowRootIdSet(state.layout).has(rootId)) {
        const candidate = synthesizePanelDefinition(state.layout, formId);
        if (!candidate?.detachable) return false;
    }

    const allocate = zoneIdAllocatorForLayout(state.layout);
    let roots = state.layout.roots;
    const targetTree = roots[rootId];
    const sourceRootId = rootOfForm(state.layout, formId);

    if (!targetTree) {
        // 空面板：播种它的第一个标签组。
        roots = { ...roots, [rootId]: makeTabset(allocate(), formId) };
    } else if (sourceRootId === rootId || sourceRootId === null) {
        roots = { ...roots, [rootId]: dockForm(targetTree, formId, target, allocate) };
    } else {
        // 跨根搬运：先插目标根（窗体不在那棵树里，moveForm 走"直接插入"），
        // 再从源根摘除。顺序无所谓 —— 两步都在同一次 dispatch 里。
        roots = { ...roots, [rootId]: dockForm(targetTree, formId, target, allocate) };
        const pruned = removeForm(roots[sourceRootId], formId);
        if (pruned === null) {
            if (sourceRootId !== MAIN_ROOT_ID) {
                const next = { ...roots };
                delete next[sourceRootId];
                roots = next;
            }
        } else {
            roots = { ...roots, [sourceRootId]: pruned };
        }
    }

    updateLayout(state, {
        ...state.layout,
        roots,
        forms: { ...state.layout.forms, [formId]: { ...form, floating: false } },
        floatOrder: state.layout.floatOrder.filter((id) => id !== formId),
    });
    state.activeFormId = formId;
    return true;
}

/** 在整棵树上把包含 formId 的标签组切到该窗体。 */
function setActiveTabEverywhere(node: DockNode, formId: string): DockNode {
    if (node.t === "tabset") {
        return node.tabs.includes(formId) ? { ...node, active: formId } : node;
    }
    return {
        ...node,
        a: setActiveTabEverywhere(node.a, formId),
        b: setActiveTabEverywhere(node.b, formId),
    };
}

/** zone id 在哪棵根里（多根之后这是所有"按 id 改树"操作的入口）。 */
function rootIdOfZone(layout: DockLayout, zoneId: string): string | null {
    for (const [rootId, tree] of Object.entries(layout.roots)) {
        if (tree && findZone(tree, zoneId)) return rootId;
    }
    return null;
}

/** 参照窗体所在的 {根, 标签组}（splitFormTo / mergeFormInto 的目标解析）。 */
function locateTabsetOfForm(
    layout: DockLayout,
    formId: string,
): { rootId: string; tabsetId: string } | null {
    for (const [rootId, tree] of Object.entries(layout.roots)) {
        const tabset = tree ? findTabsetOfForm(tree, formId) : null;
        if (tabset) return { rootId, tabsetId: tabset.id };
    }
    return null;
}

/** 布局根的嵌套深度：主根为 0，面板的根 = 它自己的深度。 */
function depthOfRoot(layout: DockLayout, rootId: string): number {
    if (rootId === MAIN_ROOT_ID) return 0;
    const ownerId = ownerOfRoot(layout, rootId);
    return ownerId ? panelDepth(layout, ownerId) : 0;
}

/** 面板落到 rootId 之后深度是否仍在限额内（`rootId` 为 null 表示浮出为顶层）。 */
function panelDepthFits(layout: DockLayout, rootId: string | null, maxDepth: number): boolean {
    const base = rootId === null ? 0 : depthOfRoot(layout, rootId);
    return base + 1 <= Math.max(1, maxDepth);
}

/** 当前由独立操作系统窗口承载的全部布局根。 */
function collectOsWindowRootIdSet(layout: DockLayout): Set<string> {
    const out = new Set<string>();
    for (const form of Object.values(layout.forms)) {
        if (
            isPanelForm(form) &&
            form.childRootId &&
            form.floating === true &&
            form.floatMode === "osWindow"
        ) {
            for (const nested of collectSubtreeRootIds(layout, form.childRootId)) out.add(nested);
        }
    }
    return out;
}

/**
 * 布局变更的统一出口：写回新布局，并追查"这次变更清空了哪些面板根"。
 *
 * 【为什么要在这里追查】"面板失去最后一个成员 → 自动解散"是一条跨操作的
 * 收尾规则（拖走、关掉、拆出独立窗口都会触发），而触发点分散在每个 reducer。
 * 与其在每处手工判断，不如对比变更前后根表里**消失的条目** —— 从有到无的
 * 转移本身就是判据，显式新建的空面板从未有过条目，天然不会被误伤。
 */
function updateLayout(state: DockState, next: DockLayout): void {
    const emptiedRootIds: string[] = [];
    for (const [rootId, tree] of Object.entries(state.layout.roots)) {
        if (tree !== undefined && next.roots[rootId] === undefined) emptiedRootIds.push(rootId);
    }
    state.layout = next;
    if (emptiedRootIds.length > 0 && state.settings.emptyPanelAutoDissolve) {
        dissolveEmptiedPanels(state, emptiedRootIds);
    }
}

/** 面板根刚被清空后的级联解散：面板消失可能又清空外层面板，逐层跟进。 */
function dissolveEmptiedPanels(state: DockState, emptiedRootIds: string[]): void {
    const queue = [...emptiedRootIds];
    while (queue.length > 0) {
        const rootId = queue.shift() as string;
        const ownerId = ownerOfRoot(state.layout, rootId);
        if (!ownerId) continue;
        // 解散动作可能再清空别的根：dissolvePanelForm 返回新清空的根，继续入队。
        for (const nested of dissolvePanelForm(state, ownerId)) queue.push(nested);
    }
}

/**
 * 解散一个面板窗体（从布局中彻底移除面板与其根条目）。
 *
 * @returns 本次解散**新清空**的布局根 id（供级联），通常为空。
 */
function dissolvePanelForm(state: DockState, formId: string): string[] {
    const layout = state.layout;
    const form = layout.forms[formId];
    if (!isPanelForm(form) || !form?.childRootId) return [];
    const childRootId = form.childRootId;
    const panelTree = layout.roots[childRootId] ?? null;
    const emptied: string[] = [];

    let roots = { ...layout.roots };
    let forms = { ...layout.forms };
    const order = layout.order.filter((id) => id !== formId);
    let floatOrder = layout.floatOrder.filter((id) => id !== formId);

    if (form.floating) {
        // 浮动面板：成员按各自记住的浮窗几何还回浮动层（嵌套面板整体浮出，
        // 它的根与内容原样保留 —— 解散只拆一层）。
        if (panelTree) {
            for (const memberId of collectDockedForms(panelTree)) {
                const member = forms[memberId];
                if (!member) continue;
                const remembered = member.float ?? {
                    x: 140 + order.length * 28,
                    y: 120 + order.length * 28,
                    w: 640,
                    h: 440,
                };
                forms = { ...forms, [memberId]: { ...member, floating: true, float: remembered } };
                floatOrder = [...floatOrder.filter((id) => id !== memberId), memberId];
            }
        }
    } else {
        // 停靠面板：用自己的树替换它在父容器中的位置。
        const parentRootId = rootOfForm(layout, formId);
        const host = parentRootId ? findTabsetOfForm(roots[parentRootId], formId) : null;
        if (parentRootId && host) {
            const parentTree = roots[parentRootId];
            const tabsWithout = host.tabs.filter((id) => id !== formId);
            if (!panelTree) {
                const pruned = removeForm(parentTree, formId);
                if (pruned === null) {
                    if (parentRootId !== MAIN_ROOT_ID) {
                        delete roots[parentRootId];
                        emptied.push(parentRootId);
                    }
                } else {
                    roots = { ...roots, [parentRootId]: pruned };
                }
            } else if (tabsWithout.length === 0) {
                // 宿主组除面板外没有别的标签：整组直接换成面板的树。
                roots = { ...roots, [parentRootId]: panelTree };
            } else if (panelTree.t === "tabset") {
                // 面板内容本身就是一组标签 → 摊平并入宿主组。
                const merged: DockTabsetNode = {
                    ...host,
                    tabs: [...tabsWithout, ...panelTree.tabs],
                    active: panelTree.active,
                };
                roots = { ...roots, [parentRootId]: replaceZone(parentTree, host.id, merged) };
            } else {
                // 面板内容是分割树、宿主组还有别的标签：以面板的分割方向包裹
                // —— 面板内容占先，其余标签整体排到另一侧。
                const sibling: DockTabsetNode = {
                    ...host,
                    tabs: tabsWithout,
                    active: tabsWithout.includes(host.active) ? host.active : tabsWithout[0],
                };
                const allocate = zoneIdAllocatorForLayout(layout);
                roots = {
                    ...roots,
                    [parentRootId]: replaceZone(parentTree, host.id, {
                        t: "split",
                        id: allocate(),
                        dir: panelTree.dir,
                        ratio: 0.5,
                        fixed: null,
                        a: panelTree,
                        b: sibling,
                    }),
                };
            }
        }
    }

    delete roots[childRootId];
    delete forms[formId];
    state.layout = { ...layout, roots, forms, order, floatOrder };
    return emptied;
}

/** 面板窗体的后缀分配：`__panel:2`、`__panel:3`…（与 `dockSchema.nextFormSuffix` 同一约定）。 */
function nextFormSuffix(layout: DockLayout, panelId: string): number {
    let max = 1;
    const pattern = new RegExp(`^${panelId}:(\\d+)$`);
    for (const formId of Object.keys(layout.forms)) {
        const matched = pattern.exec(formId);
        if (matched) max = Math.max(max, Number(matched[1]));
    }
    return max + 1;
}

/** 一棵树里的"主编辑区"标签组：优先含 preferMain 面板者，否则第一个（浮窗组合的落点）。 */
function mainTabsetIdOfTree(tree: DockNode): string {
    const tabsets = collectTabsets(tree);
    return tabsets[0]?.id ?? "";
}

/** 落点部位 → 分割方向（center 不会走到这里：调用方已分流）。 */
function dropZoneToSideKind(zone: DockDropZone): "left" | "right" | "top" | "bottom" {
    return zone === "left" || zone === "right" || zone === "top" || zone === "bottom"
        ? zone
        : "right";
}

/** 两个浮窗矩形的并集（缺省值兜底为后者）。 */
function unionRects(
    a: DockFloatGeometry | null | undefined,
    b: DockFloatGeometry | null | undefined,
): { x: number; y: number; w: number; h: number } | null {
    if (!a && !b) return null;
    if (!a || !b) {
        const single = (a ?? b) as { x: number; y: number; w: number; h: number };
        return { x: single.x, y: single.y, w: single.w, h: single.h };
    }
    const x = Math.min(a.x, b.x);
    const y = Math.min(a.y, b.y);
    return {
        x,
        y,
        w: Math.max(a.x + a.w, b.x + b.w) - x,
        h: Math.max(a.y + a.h, b.y + b.h) - y,
    };
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
    setFormFloatMode,
    setTabPosition,
    renameForm,
    setFormProps,
    saveDockPreset,
    applyDockPreset,
    deleteDockPreset,
    createEmptyPanel,
    dissolvePanel,
    composeFloats,
} = dockSlice.actions;

export default dockSlice.reducer;
