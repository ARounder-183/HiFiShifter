/*
 * 布局的默认值、归一化与迁移。
 *
 * 【为什么后端只存原始 JSON、校验全放这里】布局 schema 会在功能演进中反复
 * 增删字段。若后端用强类型 struct 承接，每次演进都要改 Rust、加 serde default、
 * 处理旧文件迁移 —— 而这份数据的使用者自始至终只有前端。与 `paramAxisUnits`
 * 的既有决策一致（后端只做透传存储，取值合法性由前端收口）。
 *
 * 【归一化为什么不"整体回退默认"】用户可能花了十分钟排出满意的布局，任何
 * 一个字段坏掉都不该让它全丢。这里逐项修补：剔除未注册面板、修 active、
 * 剪空组、钳制越界尺寸，只有在**树彻底不可用**时才回退默认。配置文件损坏
 * 时保留 `.bak` 兜底是同一套思路（见 `config.rs` 的 `save_config`）。
 *
 * 【为什么默认布局只含两个主窗体】面板注册发生在模块加载期，而 Redux 的
 * 初始状态在更早的模块求值期就算好了 —— 这里若去读注册表，很可能读到空表，
 * 得到一个没有时间轴的界面。因此默认布局只写死两个主窗体的常量 id，其余
 * 面板由 `ensureRegisteredPanels()` 在注册完成后补齐为"已关闭"的窗体记录。
 */

import {
    collectTabsets,
    findTabsetOfForm,
    insertForm,
    isFormVisible,
    pinChildSize,
    pruneTree,
    removeForm,
    splitRootWith,
    type DockInsertTarget,
} from "./dockTree";
import { getPanel, isPanelRegistered, listPanels } from "./panelRegistry";
import {
    DOCK_LAYOUT_SCHEMA,
    type DockForm,
    type DockPlacement,
    type DockGutterSizes,
    type DockLayout,
    type DockNode,
    type DockPreset,
    type DockRect,
    type DockSplitNode,
    type DockTabsetNode,
} from "./dockTypes";

/** 沟槽尺寸的合法区间（与面板定义里的最小值保持一致）。 */
export const GUTTER_LIMITS = {
    timelineTrackHeaderPx: { min: 120, max: 560, fallback: 256 },
} as const;

export const DEFAULT_GUTTER_SIZES: DockGutterSizes = {
    timelineTrackHeaderPx: 256,
};

/** 主编辑区面板的窗体 id（与 `registerBuiltinPanels` 的命名约定一致）。 */
export const MAIN_FORM_TIMELINE = "timeline";
export const MAIN_FORM_PARAM_EDITOR = "paramEditor";

export type { DockPlacement };

/** 出厂布局：与重构前的默认视觉一致（上时间轴 / 下参数编辑器，侧栏默认关闭）。 */
export function createDefaultDockLayout(): DockLayout {
    const tree: DockSplitNode = {
        t: "split",
        id: "z1",
        dir: "col",
        ratio: 0.6,
        fixed: null,
        a: { t: "tabset", id: "z2", tabs: [MAIN_FORM_TIMELINE], active: MAIN_FORM_TIMELINE },
        b: {
            t: "tabset",
            id: "z3",
            tabs: [MAIN_FORM_PARAM_EDITOR],
            active: MAIN_FORM_PARAM_EDITOR,
        },
    };

    return {
        schema: DOCK_LAYOUT_SCHEMA,
        tree,
        forms: {
            [MAIN_FORM_TIMELINE]: {
                id: MAIN_FORM_TIMELINE,
                panelId: MAIN_FORM_TIMELINE,
                float: null,
            },
            [MAIN_FORM_PARAM_EDITOR]: {
                id: MAIN_FORM_PARAM_EDITOR,
                panelId: MAIN_FORM_PARAM_EDITOR,
                float: null,
            },
        },
        order: [MAIN_FORM_TIMELINE, MAIN_FORM_PARAM_EDITOR],
        floatOrder: [],
        gutters: { ...DEFAULT_GUTTER_SIZES },
        presets: {},
        activePreset: null,
    };
}

/**
 * 把已注册但布局里没有的面板补成"已关闭"的窗体记录。
 *
 * 两个调用时机：面板注册完成之后（内置面板在 App 模块加载期注册），以及
 * 从磁盘恢复布局之后（用户可能卸载过某个面板，或该面板是新版本才有的）。
 * 补成"已关闭"而不是直接打开，是为了让升级不改变用户已经摆好的界面。
 */
export function ensureRegisteredPanels(layout: DockLayout): DockLayout {
    const forms = { ...layout.forms };
    const order = [...layout.order];
    let changed = false;
    for (const panel of listPanels()) {
        if (forms[panel.id]) continue;
        forms[panel.id] = { id: panel.id, panelId: panel.id, float: null };
        order.push(panel.id);
        changed = true;
    }
    return changed ? { ...layout, forms, order } : layout;
}

function clampNumber(value: unknown, min: number, max: number, fallback: number): number {
    if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
    return Math.min(max, Math.max(min, Math.round(value)));
}

function normalizeGutters(raw: unknown): DockGutterSizes {
    const input = (raw ?? {}) as Partial<DockGutterSizes>;
    return {
        timelineTrackHeaderPx: clampNumber(
            input.timelineTrackHeaderPx,
            GUTTER_LIMITS.timelineTrackHeaderPx.min,
            GUTTER_LIMITS.timelineTrackHeaderPx.max,
            GUTTER_LIMITS.timelineTrackHeaderPx.fallback,
        ),
    };
}

/** 校验并修补一棵布局树；返回 null 表示这棵子树不可用（调用方剪掉它）。 */
function normalizeTree(raw: unknown, knownForms: Set<string>, seen: Set<string>): DockNode | null {
    if (!raw || typeof raw !== "object") return null;
    // 输入来自磁盘/API，形状完全不可信：按 `Record<string, unknown>` 逐字段取值，
    // 而不是断言成联合类型 —— 后者会让 `node.t` 窄化成 `never`，后续取值全部报错。
    const node = raw as Record<string, unknown>;

    if (node.t === "tabset") {
        const tabs = Array.isArray(node.tabs) ? node.tabs : [];
        const kept: string[] = [];
        for (const tab of tabs) {
            // 三重过滤：必须是字符串、面板仍注册着、且没在别处出现过。
            if (typeof tab !== "string") continue;
            if (!knownForms.has(tab) || seen.has(tab)) continue;
            seen.add(tab);
            kept.push(tab);
        }
        if (kept.length === 0) return null;
        const active =
            typeof node.active === "string" && kept.includes(node.active) ? node.active : kept[0];
        const tabset: DockTabsetNode = {
            t: "tabset",
            id: typeof node.id === "string" && node.id ? node.id : "z0",
            tabs: kept,
            active,
        };
        if (node.collapsed === true) {
            tabset.collapsed = true;
            if (typeof node.collapsedPx === "number" && Number.isFinite(node.collapsedPx)) {
                tabset.collapsedPx = clampNumber(node.collapsedPx, 20, 400, 26);
            }
        }
        return tabset;
    }

    if (node.t === "split") {
        const a = normalizeTree(node.a, knownForms, seen);
        const b = normalizeTree(node.b, knownForms, seen);
        if (a === null) return b;
        if (b === null) return a;
        const fixedRaw = node.fixed as DockSplitNode["fixed"] | null | undefined;
        const fixed =
            fixedRaw && (fixedRaw.side === "a" || fixedRaw.side === "b")
                ? { side: fixedRaw.side, px: clampNumber(fixedRaw.px, 80, 4000, 320) }
                : null;
        return {
            t: "split",
            id: typeof node.id === "string" && node.id ? node.id : "z0",
            dir: node.dir === "col" ? "col" : "row",
            ratio: typeof node.ratio === "number" ? node.ratio : 0.5,
            fixed,
            a,
            b,
        };
    }

    return null;
}

function normalizeFloat(raw: unknown): DockForm["float"] {
    if (!raw || typeof raw !== "object") return null;
    const value = raw as Record<string, unknown>;
    const num = (key: string, min: number, fallback: number): number => {
        const candidate = value[key];
        if (typeof candidate !== "number" || !Number.isFinite(candidate)) return fallback;
        return Math.max(min, Math.round(candidate));
    };
    const float: NonNullable<DockForm["float"]> = {
        x: num("x", -4000, 120),
        y: num("y", -4000, 120),
        w: num("w", 160, 420),
        h: num("h", 120, 320),
    };
    if (value.maximized === true) float.maximized = true;
    if (value.minimized === true) float.minimized = true;
    const restore = value.restore as DockRect | null | undefined;
    if (restore && typeof restore === "object") {
        const rx = Number(restore.x);
        const ry = Number(restore.y);
        const rw = Number(restore.w);
        const rh = Number(restore.h);
        if ([rx, ry, rw, rh].every((n) => Number.isFinite(n))) {
            float.restore = { x: rx, y: ry, w: rw, h: rh };
        }
    }
    return float;
}

/** 归一化一份布局（来自磁盘、预设或 API）。 */
export function normalizeDockLayout(raw: unknown): DockLayout {
    if (!raw || typeof raw !== "object") return createDefaultDockLayout();
    const input = raw as Partial<DockLayout>;

    // ── 窗体：剔除面板已不存在的记录（插件被卸载等）────────────────
    const rawForms = (input.forms ?? {}) as Record<string, Partial<DockForm>>;
    const forms: Record<string, DockForm> = {};
    for (const [formId, form] of Object.entries(rawForms)) {
        if (!form || typeof form !== "object") continue;
        const panelId = typeof form.panelId === "string" ? form.panelId : formId;
        if (!isPanelRegistered(panelId)) continue;
        const next: DockForm = { id: formId, panelId, float: normalizeFloat(form.float) };
        if (typeof form.title === "string" && form.title.trim()) next.title = form.title;
        if (form.props && typeof form.props === "object") {
            next.props = form.props as Record<string, unknown>;
        }
        forms[formId] = next;
    }

    // 主编辑区面板缺失时补齐 —— 否则用户会得到一个没有时间轴的界面。
    for (const panelId of [MAIN_FORM_TIMELINE, MAIN_FORM_PARAM_EDITOR]) {
        if (forms[panelId] || !isPanelRegistered(panelId)) continue;
        forms[panelId] = { id: panelId, panelId, float: null };
    }

    const knownForms = new Set(Object.keys(forms));

    // ── 树 ────────────────────────────────────────────────────────
    const seen = new Set<string>();
    const normalized = normalizeTree(input.tree, knownForms, seen);
    const tree =
        pruneTree(normalized ?? createDefaultDockLayout().tree) ?? createDefaultDockLayout().tree;

    // ── 顺序：磁盘上的顺序优先，缺的按窗体表补全 ─────────────────
    const order: string[] = [];
    const rawOrder = Array.isArray(input.order) ? input.order : [];
    for (const formId of rawOrder) {
        if (typeof formId === "string" && forms[formId] && !order.includes(formId))
            order.push(formId);
    }
    for (const formId of Object.keys(forms)) {
        if (!order.includes(formId)) order.push(formId);
    }

    // 可见性互斥：浮动窗体不能同时出现在树上。
    for (const form of Object.values(forms)) {
        if (form.float && findTabsetOfForm(tree, form.id)) form.float = null;
    }

    const floatOrder: string[] = [];
    const rawFloatOrder = Array.isArray(input.floatOrder) ? input.floatOrder : [];
    for (const formId of rawFloatOrder) {
        if (forms[formId]?.float && !floatOrder.includes(formId)) floatOrder.push(formId);
    }
    for (const form of Object.values(forms)) {
        if (form.float && !floatOrder.includes(form.id)) floatOrder.push(form.id);
    }

    return {
        schema: DOCK_LAYOUT_SCHEMA,
        tree,
        forms,
        order,
        floatOrder,
        gutters: normalizeGutters(input.gutters),
        presets: normalizePresets(input.presets, knownForms),
        activePreset: typeof input.activePreset === "string" ? input.activePreset : null,
    };
}

function normalizePresets(raw: unknown, knownForms: Set<string>): Record<string, DockPreset> {
    if (!raw || typeof raw !== "object") return {};
    const out: Record<string, DockPreset> = {};
    for (const [name, value] of Object.entries(raw as Record<string, Partial<DockPreset>>)) {
        if (!value || typeof value !== "object") continue;
        const seen = new Set<string>();
        const tree = normalizeTree(value.tree, knownForms, seen);
        if (tree === null) continue;
        const pruned = pruneTree(tree);
        if (pruned === null) continue;
        const forms: Record<string, DockForm> = {};
        for (const [formId, form] of Object.entries(value.forms ?? {})) {
            if (!form || typeof form !== "object") continue;
            const panelId = typeof form.panelId === "string" ? form.panelId : formId;
            if (!knownForms.has(formId) || !isPanelRegistered(panelId)) continue;
            const next: DockForm = { id: formId, panelId, float: normalizeFloat(form.float) };
            if (typeof form.title === "string" && form.title.trim()) next.title = form.title;
            forms[formId] = next;
        }
        out[name] = {
            name,
            tree: pruned,
            forms,
            order: Array.isArray(value.order)
                ? value.order.filter((id): id is string => typeof id === "string")
                : [],
            floatOrder: Array.isArray(value.floatOrder)
                ? value.floatOrder.filter((id): id is string => typeof id === "string")
                : [],
            gutters: normalizeGutters(value.gutters),
            createdAtMs:
                typeof value.createdAtMs === "number" && Number.isFinite(value.createdAtMs)
                    ? value.createdAtMs
                    : Date.now(),
        };
    }
    return out;
}

/**
 * 迁移旧版本布局。
 *
 * v1 是首个版本，所以这里只做"版本不可识别 → 交给归一化重建"的处理；入口
 * 先于归一化存在，是为了让将来新增字段时能在此做定向搬移，而不是被迫把
 * 版本判断写进通用归一化逻辑（那会让通用逻辑越来越难读）。
 */
export function migrateDockLayout(raw: unknown): unknown {
    if (!raw || typeof raw !== "object") return raw;
    const version = (raw as { schema?: unknown }).schema;
    if (typeof version !== "number" || version <= DOCK_LAYOUT_SCHEMA) return raw;
    // 来自更新版本（用户降级了应用）：不认识就不猜，交给归一化重建。
    return null;
}

/** 主编辑区所在的标签组：优先含 `preferMain` 面板者，否则取第一个。 */
export function findMainTabset(layout: DockLayout): DockTabsetNode | null {
    const tabsets = collectTabsets(layout.tree);
    for (const tabset of tabsets) {
        const isMain = tabset.tabs.some((formId) => {
            const form = layout.forms[formId];
            return form ? getPanel(form.panelId)?.preferMain === true : false;
        });
        if (isMain) return tabset;
    }
    return tabsets[0] ?? null;
}

/** 找出该面板当前可见的窗体 id（多实例时取第一个）。 */
export function findVisibleFormForPanel(layout: DockLayout, panelId: string): string | null {
    for (const formId of layout.order) {
        if (layout.forms[formId]?.panelId !== panelId) continue;
        if (isFormVisible(layout, formId)) return formId;
    }
    return null;
}

/** 找出该面板已关闭的窗体 id（用于重开时复用记录，保住标题与 props）。 */
export function findClosedFormForPanel(layout: DockLayout, panelId: string): string | null {
    for (const formId of layout.order) {
        if (layout.forms[formId]?.panelId !== panelId) continue;
        if (!isFormVisible(layout, formId)) return formId;
    }
    return null;
}

/** 把一个窗体放到指定落点。 */
export function placeForm(layout: DockLayout, formId: string, placement: DockPlacement): DockNode {
    if (placement.side === "center" || placement.tabWith) {
        // 先尝试并入同伴组；没有同伴（或它就是 center）则并入主编辑组。
        if (placement.tabWith) {
            const companion = findVisibleFormForPanel(layout, placement.tabWith);
            const target = companion ? findTabsetOfForm(layout.tree, companion) : null;
            if (target)
                return insertForm(layout.tree, formId, { kind: "tab", tabsetId: target.id });
        }
        const main = findMainTabset(layout);
        if (!main) return layout.tree;
        return insertForm(layout.tree, formId, { kind: "tab", tabsetId: main.id });
    }

    // 边缘落点：拆整个工作区，让新组贯通全高/全宽（默认落点的正确语义）。
    const before = new Set(collectTabsets(layout.tree).map((tabset) => tabset.id));
    const next = splitRootWith(layout.tree, formId, placement.side);
    if (!placement.sizePx) return next;

    const created = collectTabsets(next).find(
        (tabset) => !before.has(tabset.id) && tabset.tabs.includes(formId),
    );
    if (!created) return next;
    return pinChildSize(next, created.id, placement.sizePx);
}

/**
 * 打开一个面板：复用已关闭的窗体记录，或新建窗体，然后按默认落点放置。
 *
 * 单例面板已可见时原样返回（调用方负责聚焦/切标签），避免重复打开产生第二个
 * 时间轴 —— 那会让用户彻底困惑。
 */
export function openPanelInLayout(
    layout: DockLayout,
    panelId: string,
    placement?: DockPlacement,
): DockLayout {
    const definition = getPanel(panelId);
    if (!definition) return layout;

    const visible = findVisibleFormForPanel(layout, panelId);
    if (visible && definition.singleton !== false) return layout;

    const existing = findClosedFormForPanel(layout, panelId);
    const formId = existing ?? `${panelId}:${nextFormSuffix(layout, panelId)}`;

    const forms = { ...layout.forms };
    const previous = forms[formId];
    forms[formId] = previous ? { ...previous, float: null } : { id: formId, panelId, float: null };
    const order = layout.order.includes(formId) ? layout.order : [...layout.order, formId];

    const base: DockLayout = { ...layout, forms, order };
    const resolved = placement ?? definition.defaultPlacement ?? { side: "center" };
    const tree = placeForm(base, formId, resolved);
    return { ...base, tree: pruneTree(tree) ?? base.tree };
}

/** 多实例窗体的后缀：`paramEditor:2`、`paramEditor:3`… */
function nextFormSuffix(layout: DockLayout, panelId: string): number {
    let max = 1;
    const pattern = new RegExp(`^${panelId}:(\\d+)$`);
    for (const formId of Object.keys(layout.forms)) {
        const matched = pattern.exec(formId);
        if (matched) max = Math.max(max, Number(matched[1]));
    }
    return max + 1;
}

/**
 * 关掉一个窗体（从树上摘除 / 清掉浮动几何），保留窗体记录。
 *
 * 拒绝关掉最后一个可见窗体：空布局无法渲染，也不该是可持久化的状态。
 */
export function closeFormInLayout(layout: DockLayout, formId: string): DockLayout {
    const form = layout.forms[formId];
    if (!form) return layout;

    const visibleCount = layout.order.filter((id) => isFormVisible(layout, id)).length;
    if (visibleCount <= 1 && isFormVisible(layout, formId)) return layout;

    const tree = removeForm(layout.tree, formId) ?? layout.tree;
    const forms = { ...layout.forms, [formId]: { ...form, float: null } };
    return { ...layout, tree, forms, floatOrder: layout.floatOrder.filter((id) => id !== formId) };
}

/** 供外部构造落点用。 */
export type { DockInsertTarget };
