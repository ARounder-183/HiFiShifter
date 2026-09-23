/*
 * 停靠布局树的纯操作。
 *
 * 全部是纯函数：输入树 + 参数 → 新树。没有 Redux、没有 DOM、没有 React。
 * 这样"停靠/拆分/合并/重排"这些最容易出错的几何逻辑可以在无 jsdom 的环境下
 * 直接单测（本仓库 vitest 未配置 jsdom，纯逻辑测试是唯一可行的自动化测试面）。
 *
 * 不变式（由 `normalizeTree` 保证，每次修改后都跑一遍）：
 * 1. 不存在空标签组 —— 摘掉最后一个标签的组会被剪掉；
 * 2. 不存在单子分割 —— 一侧被剪掉后，父分割塌缩为幸存的那一侧；
 * 3. `active` 必定是 `tabs` 的成员；
 * 4. `ratio` 已钳制到 [0.05, 0.95]（极端比例会让一侧退化成不可用的窄条）。
 */

import {
    type DockLayout,
    type DockNode,
    type DockRect,
    type DockSplitDir,
    type DockSplitNode,
    type DockTabsetNode,
} from "./dockTypes";

// ─────────────────────────────────────────────────────────────────
// 树查询：只读遍历。放在这里而不是 `dockTypes`，是为了让类型模块保持
// 纯声明（零运行时导出），也让"树怎么走"只有一处实现。
// ─────────────────────────────────────────────────────────────────

/** 查找包含指定窗体的标签组。 */
export function findTabsetOfForm(node: DockNode, formId: string): DockTabsetNode | null {
    if (node.t === "tabset") {
        return node.tabs.includes(formId) ? node : null;
    }
    return findTabsetOfForm(node.a, formId) ?? findTabsetOfForm(node.b, formId);
}

/** 深度优先收集全部标签组。 */
export function collectTabsets(node: DockNode, out: DockTabsetNode[] = []): DockTabsetNode[] {
    if (node.t === "tabset") {
        out.push(node);
    } else {
        collectTabsets(node.a, out);
        collectTabsets(node.b, out);
    }
    return out;
}

/** 深度优先收集全部分割节点。 */
export function collectSplits(node: DockNode, out: DockSplitNode[] = []): DockSplitNode[] {
    if (node.t === "split") {
        out.push(node);
        collectSplits(node.a, out);
        collectSplits(node.b, out);
    }
    return out;
}

/** 收集树上全部窗体 id（按深度优先顺序）。 */
export function collectDockedForms(node: DockNode, out: string[] = []): string[] {
    if (node.t === "tabset") {
        out.push(...node.tabs);
    } else {
        collectDockedForms(node.a, out);
        collectDockedForms(node.b, out);
    }
    return out;
}

/**
 * 窗体可见性 —— 派生而非存储。
 *
 * 在布局树里（已停靠）或有浮动几何（已浮动）即为可见；两者皆无 = 已关闭。
 * 关闭只从树上摘除并清空浮动几何，`forms` 记录保留，所以重开时标题与
 * 私有 props 都能回来。
 */
export function isFormVisible(layout: DockLayout, formId: string): boolean {
    if (layout.forms[formId]?.float) return true;
    return findTabsetOfForm(layout.tree, formId) !== null;
}

/** 当前可见的窗体 id（停靠 + 浮动），供"显示窗体"菜单使用。 */
export function collectVisibleForms(layout: DockLayout): string[] {
    const docked = collectDockedForms(layout.tree);
    const floating = layout.order.filter((id) => Boolean(layout.forms[id]?.float));
    return [...docked, ...floating.filter((id) => !docked.includes(id))];
}

/** 分割比例的合法区间：留给两侧的最小可用空间。 */
export const MIN_SPLIT_RATIO = 0.05;
export const MAX_SPLIT_RATIO = 0.95;

/** 标签组的最小尺寸（px），用于拖拽时的钳制与规范化。 */
export const TABSET_MIN_PX = 80;

export function clampRatio(value: number): number {
    if (!Number.isFinite(value)) return 0.5;
    return Math.min(MAX_SPLIT_RATIO, Math.max(MIN_SPLIT_RATIO, value));
}

/**
 * 生成 Zone id 的分配器。
 *
 * 用 `z<递增整数>` 而不是随机串：布局 JSON 是给人看、给用户导入导出的
 * （预设文件），可读的 id 让 diff 与手工排错都容易得多。起始序号取"已有
 * 最大值 + 1"，保证重新加载后不会与历史 id 冲突；一次调用内连续分配多个
 * 时由分配器自己递增，避免"新节点还没进树、扫描看不到它"导致撞号。
 */
export function zoneIdAllocator(tree: DockNode): () => string {
    let max = 0;
    const scan = (node: DockNode) => {
        const matched = /^z(\d+)$/.exec(node.id);
        if (matched) max = Math.max(max, Number(matched[1]));
        if (node.t === "split") {
            scan(node.a);
            scan(node.b);
        }
    };
    scan(tree);
    return () => {
        max += 1;
        return `z${max}`;
    };
}

/** 生成一个未占用的 Zone id。 */
export function nextZoneId(tree: DockNode): string {
    return zoneIdAllocator(tree)();
}

/** 按 id 查找 Zone。 */
export function findZone(node: DockNode, zoneId: string): DockNode | null {
    if (node.id === zoneId) return node;
    if (node.t === "split") {
        return findZone(node.a, zoneId) ?? findZone(node.b, zoneId);
    }
    return null;
}

/** 把树上某个 Zone 替换为新节点（id 不匹配则原样返回）。 */
export function replaceZone(node: DockNode, zoneId: string, next: DockNode): DockNode {
    if (node.id === zoneId) return next;
    if (node.t !== "split") return node;
    return { ...node, a: replaceZone(node.a, zoneId, next), b: replaceZone(node.b, zoneId, next) };
}

/** 替换一个标签组（用于原地改写标签集合，如整组搬家时补回全部标签）。 */
export function replaceTabset(
    node: DockNode,
    tabsetId: string,
    next: DockTabsetNode,
): DockNode {
    return replaceZone(node, tabsetId, next);
}

/**
 * 规范化：剪空组、塌缩单子分割、修正 active、钳制 ratio。
 *
 * 返回 `null` 表示整棵子树为空（调用方需要把它从父节点上摘掉）。所有会改变
 * 树结构的操作最后都过一遍这里，于是上面 4 条不变式只需在一处维护。
 */
export function pruneTree(node: DockNode): DockNode | null {
    if (node.t === "tabset") {
        const tabs = node.tabs.filter((id, index) => node.tabs.indexOf(id) === index);
        if (tabs.length === 0) return null;
        const active = tabs.includes(node.active) ? node.active : tabs[0];
        return { ...node, tabs, active };
    }

    const a = pruneTree(node.a);
    const b = pruneTree(node.b);
    if (a === null) return b;
    if (b === null) return a;
    if (a === node.a && b === node.b) return node;
    return { ...node, a, b, ratio: clampRatio(node.ratio) };
}

/**
 * 摘除一个窗体（不改变其它窗体的相对顺序）。
 *
 * 注意返回 `null` 的情形：整棵树只剩这一个窗体时，摘掉就什么都不剩了。
 * 调用方（`dockSlice`）必须拒绝把最后一个窗体关掉 —— 空布局无法渲染，
 * 也不该是一个可持久化的状态。
 */
export function removeForm(node: DockNode, formId: string): DockNode | null {
    if (node.t === "tabset") {
        if (!node.tabs.includes(formId)) return node;
        return pruneTree({ ...node, tabs: node.tabs.filter((id) => id !== formId) });
    }
    const a = removeForm(node.a, formId);
    const b = removeForm(node.b, formId);
    if (a === null) return b;
    if (b === null) return a;
    if (a === node.a && b === node.b) return node;
    return { ...node, a, b };
}

/** 摘除一个 Zone 子树（用于把整个标签组移到浮动层）。 */
export function removeZone(node: DockNode, zoneId: string): DockNode | null {
    if (node.t === "tabset") {
        return node.id === zoneId ? null : node;
    }
    if (node.id === zoneId) return null;
    const a = removeZone(node.a, zoneId);
    const b = removeZone(node.b, zoneId);
    if (a === null) return b;
    if (b === null) return a;
    if (a === node.a && b === node.b) return node;
    return { ...node, a, b };
}

/** 新建一个只含单个窗体的标签组。 */
export function makeTabset(zoneId: string, formId: string): DockTabsetNode {
    return { t: "tabset", id: zoneId, tabs: [formId], active: formId };
}

/** 把窗体并入已有标签组。`index` 缺省追加到末尾。 */
export function addFormToTabset(
    node: DockNode,
    tabsetId: string,
    formId: string,
    index?: number,
): DockNode {
    if (node.t === "tabset") {
        if (node.id !== tabsetId) return node;
        // 同一窗体重复插入只改顺序，不产生重复标签。
        const without = node.tabs.filter((id) => id !== formId);
        const at = index === undefined ? without.length : Math.max(0, Math.min(index, without.length));
        const tabs = [...without.slice(0, at), formId, ...without.slice(at)];
        return { ...node, tabs, active: formId };
    }
    return {
        ...node,
        a: addFormToTabset(node.a, tabsetId, formId, index),
        b: addFormToTabset(node.b, tabsetId, formId, index),
    };
}

/**
 * 在整个工作区的某一侧拆出新组。
 *
 * 与 `splitTabsetWith`（拆某个组）语义不同，这条用于**默认落点**：右侧停靠栏
 * 应当贴着工作区右边缘、贯通全高，而不是只在时间轴旁边开一列、下方还留着
 * 参数编辑器横跨整幅宽度。用户手动拖到某个组边缘时走的才是后者。
 */
export function splitRootWith(
    tree: DockNode,
    formId: string,
    side: "left" | "right" | "top" | "bottom",
): DockNode {
    const dir: DockSplitDir = side === "left" || side === "right" ? "row" : "col";
    const alloc = zoneIdAllocator(tree);
    const newTabset = makeTabset(alloc(), formId);
    const before = side === "left" || side === "top";
    return {
        t: "split",
        id: alloc(),
        dir,
        ratio: 0.5,
        fixed: null,
        a: before ? newTabset : tree,
        b: before ? tree : newTabset,
    };
}

/** 让包含 `childId` 的分割节点把该侧设为固定像素。 */
export function pinChildSize(tree: DockNode, childId: string, px: number): DockNode {
    const walk = (node: DockNode): DockNode => {
        if (node.t !== "split") return node;
        const next: DockSplitNode = { ...node, a: walk(node.a), b: walk(node.b) };
        if (node.a.id === childId) next.fixed = { side: "a", px };
        else if (node.b.id === childId) next.fixed = { side: "b", px };
        return next;
    };
    return walk(tree);
}

/** 在目标标签组的指定一侧拆出新组。 */
export function splitTabsetWith(
    tree: DockNode,
    tabsetId: string,
    formId: string,
    side: "left" | "right" | "top" | "bottom",
): DockNode {
    const target = findZone(tree, tabsetId);
    if (!target || target.t !== "tabset") return tree;

    const dir: DockSplitDir = side === "left" || side === "right" ? "row" : "col";
    const alloc = zoneIdAllocator(tree);
    const newTabset = makeTabset(alloc(), formId);
    const split: DockSplitNode = {
        t: "split",
        id: alloc(),
        dir,
        ratio: 0.5,
        fixed: null,
        a: side === "left" || side === "top" ? newTabset : target,
        b: side === "left" || side === "top" ? target : newTabset,
    };
    return replaceZone(tree, tabsetId, split);
}

/**
 * 在目标标签组的某一侧拆出新组，并让**新组**固定像素尺寸。
 *
 * 右侧停靠栏（文件浏览器 / 记事本）走这条路径：新组固定在右侧、宽度不随
 * 窗口缩放变化，主编辑区吸收全部变化 —— 与用户"我把它拖到 360px 就该是
 * 360px"的预期一致。
 */
export function splitTabsetWithFixedSize(
    tree: DockNode,
    tabsetId: string,
    formId: string,
    side: "left" | "right" | "top" | "bottom",
    sizePx: number,
): DockNode {
    const split = splitTabsetWith(tree, tabsetId, formId, side);
    const created = findTabsetOfFormNew(split, tree, formId);
    if (!created) return split;
    const parent = findParentSplit(split, created.id);
    if (!parent) return split;
    return replaceZone(split, parent.id, {
        ...parent,
        fixed: { side: parent.a.id === created.id ? "a" : "b", px: sizePx },
    });
}

/** 找出 `next` 相对 `prev` 新增的、包含 `formId` 的标签组。 */
function findTabsetOfFormNew(
    next: DockNode,
    prev: DockNode,
    formId: string,
): DockTabsetNode | null {
    const before = new Set(collectTabsets(prev).map((tabset) => tabset.id));
    for (const tabset of collectTabsets(next)) {
        if (!before.has(tabset.id) && tabset.tabs.includes(formId)) return tabset;
    }
    return null;
}

/** 查找某个 Zone 的父分割节点。 */
export function findParentSplit(node: DockNode, zoneId: string): DockSplitNode | null {
    if (node.t !== "split") return null;
    if (node.a.id === zoneId || node.b.id === zoneId) return node;
    return findParentSplit(node.a, zoneId) ?? findParentSplit(node.b, zoneId);
}

/**
 * 把窗体放到目标位置。
 *
 * `target` 有两种形态：并入某个标签组（可指定插入下标），或在某个标签组的
 * 某一侧拆出新组。调用方负责先把窗体从旧位置摘除（见 `moveForm`）。
 */
export type DockInsertTarget =
    | { kind: "tab"; tabsetId: string; index?: number }
    | { kind: "split"; tabsetId: string; side: "left" | "right" | "top" | "bottom" };

export function insertForm(tree: DockNode, formId: string, target: DockInsertTarget): DockNode {
    if (target.kind === "tab") {
        return addFormToTabset(tree, target.tabsetId, formId, target.index);
    }
    return splitTabsetWith(tree, target.tabsetId, formId, target.side);
}

/**
 * 移动窗体：先摘除，再插入。
 *
 * 【为什么必须一次性完成】分两步做会在中间态触发规范化 —— 若窗体是源标签组
 * 的最后一个标签，源组会被剪掉，此时若目标组恰好就是被剪掉的那个（同组内
 * 拖拽重排），插入就会落空。这里在同一次调用里算完，只把最终结果交给规范化。
 */
export function moveForm(tree: DockNode, formId: string, target: DockInsertTarget): DockNode {
    const source = collectTabsets(tree).find((tabset) => tabset.tabs.includes(formId));
    if (!source) return tree;

    // 同组内重排：只改顺序，不摘不插，避免组被剪掉的中间态。
    if (target.kind === "tab" && target.tabsetId === source.id) {
        return addFormToTabset(tree, source.id, formId, target.index);
    }

    const pruned = removeForm(tree, formId);
    if (pruned === null) return tree;

    // 目标组可能因摘除而被剪掉（源组就是目标组的情况上面已排除）。
    if (!findZone(pruned, target.tabsetId)) {
        // 目标消失：退化为并入第一个标签组，保证窗体不丢。
        const fallback = collectTabsets(pruned)[0];
        if (!fallback) return tree;
        return addFormToTabset(pruned, fallback.id, formId);
    }

    return insertForm(pruned, formId, target);
}

/** 设置分割比例（`fixed` 非空时表示某一侧固定像素）。 */
export function setSplitRatio(
    tree: DockNode,
    splitId: string,
    ratio: number,
    fixed: { side: "a" | "b"; px: number } | null,
): DockNode {
    const node = findZone(tree, splitId);
    if (!node || node.t !== "split") return tree;
    return replaceZone(tree, splitId, { ...node, ratio: clampRatio(ratio), fixed });
}

/** 切换活动标签。 */
export function setActiveTab(tree: DockNode, tabsetId: string, formId: string): DockNode {
    const node = findZone(tree, tabsetId);
    if (!node || node.t !== "tabset" || !node.tabs.includes(formId)) return tree;
    return replaceZone(tree, tabsetId, { ...node, active: formId });
}

/** 折叠/展开标签组。 */
export function setTabsetCollapsed(
    tree: DockNode,
    tabsetId: string,
    collapsed: boolean,
    collapsedPx?: number,
): DockNode {
    const node = findZone(tree, tabsetId);
    if (!node || node.t !== "tabset") return tree;
    const next: DockTabsetNode = { ...node, collapsed };
    if (collapsedPx !== undefined) next.collapsedPx = collapsedPx;
    return replaceZone(tree, tabsetId, next);
}

/** 重命名标签组内某窗体的显示标题（标题存在 forms 上，这里只做顺序无关的占位）。 */
export function moveTabWithin(tree: DockNode, tabsetId: string, formId: string, index: number): DockNode {
    return addFormToTabset(tree, tabsetId, formId, index);
}

/**
 * 计算 Zone 的矩形分割结果。
 *
 * 纯几何，供渲染与落点判定共用 —— 渲染层不该自己再算一遍，否则"看到的落点"
 * 与"实际落点"会漂移。
 */
export function splitRect(
    rect: DockRect,
    node: Pick<DockSplitNode, "dir" | "ratio" | "fixed">,
    splitterPx: number,
): [DockRect, DockRect] {
    const horizontal = node.dir === "row";
    const available = Math.max(0, (horizontal ? rect.w : rect.h) - splitterPx);

    let aSize: number;
    if (node.fixed === null) {
        aSize = available * clampRatio(node.ratio);
    } else {
        const px = Math.min(available, Math.max(0, node.fixed.px));
        aSize = node.fixed.side === "a" ? px : available - px;
    }
    const bSize = Math.max(0, available - aSize);

    if (horizontal) {
        return [
            { x: rect.x, y: rect.y, w: aSize, h: rect.h },
            { x: rect.x + aSize + splitterPx, y: rect.y, w: bSize, h: rect.h },
        ];
    }
    return [
        { x: rect.x, y: rect.y, w: rect.w, h: aSize },
        { x: rect.x, y: rect.y + aSize + splitterPx, w: rect.w, h: bSize },
    ];
}
