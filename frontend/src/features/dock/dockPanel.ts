/*
 * 面板（容器窗体）的派生属性。
 *
 * 【为什么面板不注册进 `panelRegistry`】注册表的每一项都带组件与静态声明
 * （`detachable` 等），而面板恰恰在这两点上都是**派生**的：它渲染的不是组件
 * 而是自己的树，能否拆独立窗口取决于全体子窗体。把它塞进注册表就得给注册表
 * 加"动态字段"，让最简单的扩展点背上布局相关的复杂性。于是面板走保留 id
 * （`DOCK_PANEL_FORM`）+ 本模块派生的路径；`synthesizePanelDefinition` 把
 * 派生结果包成一份普通的 `PanelDefinition`，供既有的消费方（浮动标题栏、
 * 独立窗口命令）零改动地使用。
 *
 * 全部是纯函数：输入布局 → 输出判定。没有 Redux、没有 DOM、没有 React，
 * 与 `dockTree` 同一套可单测纪律。
 */

import type { MessageKey } from "../../i18n/messages";

import type { DockForm, DockLayout, DockNode } from "./dockTypes";
import { DOCK_PANEL_FORM } from "./dockTypes";
import {
    collectDockedForms,
    collectTabsets,
    findTabsetOfForm,
    isPanelForm,
    readRoot,
} from "./dockTree";
import { getPanel, type PanelDefinition } from "./panelRegistry";

/** 窗体 id → 是否为面板（便捷重导出，调用方不必同时 import 两个模块）。 */
export const isPanel = isPanelForm;

/** 某窗体在界面上的标题：用户重命名优先，其次按内容派生（面板），再按注册表。 */
export function displayTitleOf(
    layout: DockLayout,
    formId: string,
    translate: (key: MessageKey) => string,
): string {
    const form = layout.forms[formId];
    if (!form) return formId;
    if (form.title) return form.title;
    if (isPanelForm(form) && form.childRootId) {
        return panelTitleOf(layout, formId, translate);
    }
    const definition = getPanel(form.panelId);
    return definition ? translate(definition.titleKey) : form.panelId;
}

/**
 * 面板的**派生标题**：活动子窗体的标题；成员多于一个时附带数量。
 *
 * 【为什么从内容派生】组合出来的面板若永远叫"面板"，屏幕上同时浮着三个面板时
 * 用户无法分辨谁是谁。活动子窗体的标题让它表现得像一个正常的窗口标题；数量
 * 后缀提示"这是一组"。用户显式重命名永远优先（`displayTitleOf` 已处理）。
 */
export function panelTitleOf(
    layout: DockLayout,
    formId: string,
    translate: (key: MessageKey) => string,
): string {
    const form = layout.forms[formId];
    if (!isPanelForm(form) || !form?.childRootId) return translate("dock_panel_title");
    const tree = readRoot(layout, form.childRootId);
    if (!tree) return translate("dock_panel_title");
    const members = collectDockedForms(tree);
    const firstTabset = collectTabsets(tree)[0];
    const activeId = firstTabset?.active ?? firstTabset?.tabs[0];
    const base = activeId ? displayTitleOf(layout, activeId, translate) : translate("dock_panel_title");
    return members.length > 1 ? `${base} (${members.length})` : base;
}

/**
 * 面板能否拆到**独立操作系统窗口** —— 用户规则的精确形式：
 *
 * > 当面板内的所有窗体都允许在独立窗口中打开时，则该面板也被允许在独立窗口中打开。
 *
 * 【为什么这条规则是对的而不是权宜】跨 Webview 就是另一个 JS 上下文，面板组件
 * 必须重新挂载（见 `detachedWindow` 头注）。时间轴与参数编辑器带着 WebGL 上下文
 * 与波形缓存，重挂载代价是数秒卡顿，因此它们声明了不可拆 —— 带着它们的面板
 * 同样拆不得。这条规则恰好等于"全体成员都持有可挂载证书"。空面板按空全称量词
 * 为真处理：空面板拆出去就是一个空的停靠区，没有重挂载成本。
 *
 * @returns `blockedBy` 列出不可拆成员的窗体 id（递归展开到最内层的叶子），供
 *   提示文案说清"是谁挡住了"。
 */
export function isPanelDetachable(
    layout: DockLayout,
    formId: string,
): { ok: boolean; blockedBy: string[] } {
    const blockedBy: string[] = [];
    const visit = (currentFormId: string, depth: number): boolean => {
        // 与布局归一化的环形防护互为备份：这里兜住"合法 JSON 但拼出环"的运行期判定。
        if (depth > 64) return true;
        const form = layout.forms[currentFormId];
        if (!form) return true;
        if (!isPanelForm(form) || !form.childRootId) {
            const detachable = getPanel(form.panelId)?.detachable === true;
            if (!detachable) blockedBy.push(currentFormId);
            return detachable;
        }
        const tree = readRoot(layout, form.childRootId);
        if (!tree) return true; // 空面板：没有成员，也就没有阻碍。
        return collectDockedForms(tree).every((memberId) => visit(memberId, depth + 1));
    };
    return { ok: visit(formId, 0), blockedBy };
}

/**
 * 面板的嵌套深度：1 = 停靠在主根（或浮动），每被一个面板包住再加 1。
 *
 * 交互上限（设置里的 `maxPanelDepth`）在创建/组合入口用本函数判定；归一化另有一道
 * 更高的硬上限兜底（见 `DOCK_MAX_PANEL_DEPTH_HARD`）。
 */
export function panelDepth(layout: DockLayout, formId: string): number {
    let depth = 1;
    let seen = new Set<string>([formId]);
    let owner = ownerRootIdOf(layout, formId);
    while (owner !== null) {
        const ownerFormId = ownerOfRoot(layout, owner);
        if (ownerFormId === null || seen.has(ownerFormId)) return depth;
        depth += 1;
        seen = new Set([...seen, ownerFormId]);
        owner = ownerRootIdOf(layout, ownerFormId);
    }
    return depth;
}

/** 面板窗体停靠在哪棵树里（浮动 = null）。 */
function ownerRootIdOf(layout: DockLayout, formId: string): string | null {
    for (const [rootId, tree] of Object.entries(layout.roots)) {
        if (tree && findTabsetOfForm(tree, formId)) return rootId;
    }
    return null;
}

/** 哪个面板窗体拥有这棵根（主根与孤儿根返回 null）。 */
export function ownerOfRoot(layout: DockLayout, rootId: string): string | null {
    for (const form of Object.values(layout.forms)) {
        if (isPanelForm(form) && form.childRootId === rootId) return form.id;
    }
    return null;
}

/**
 * 某棵根**自身及其子树内**的全部布局根 id（含传入的根）。
 *
 * 两个消费方：拖拽的自引用防护（"不能把自己拖进自己"需要排除整棵子树的根），
 * 以及独立窗口的挂载排除（面板拆出去了，子树里的窗体就归那个窗口渲染）。
 */
export function collectSubtreeRootIds(layout: DockLayout, rootId: string): Set<string> {
    const out = new Set<string>([rootId]);
    let frontier = [rootId];
    while (frontier.length > 0) {
        const next: string[] = [];
        for (const current of frontier) {
            const tree = readRoot(layout, current);
            if (!tree) continue;
            for (const memberId of collectDockedForms(tree)) {
                const member = layout.forms[memberId];
                if (isPanelForm(member) && member.childRootId && !out.has(member.childRootId)) {
                    out.add(member.childRootId);
                    next.push(member.childRootId);
                }
            }
        }
        frontier = next;
    }
    return out;
}

/** 全部"由独立操作系统窗口承载"的布局根（含其子树内的面板根）。 */
export function collectOsWindowRootIds(layout: DockLayout): Set<string> {
    const out = new Set<string>();
    for (const form of Object.values(layout.forms)) {
        if (
            isPanelForm(form) &&
            form.childRootId &&
            form.floating === true &&
            form.floatMode === "osWindow"
        ) {
            for (const rootId of collectSubtreeRootIds(layout, form.childRootId)) {
                out.add(rootId);
            }
        }
    }
    return out;
}

/**
 * 面板子树在某一方向上的最小像素。
 *
 * 与 `DockNodeView.subtreeMinSize` 同一语义，但递归穿过嵌套面板：面板的最小值
 * 不是注册表里的静态声明，而是它自己那棵树的最小值 —— 否则嵌套面板会被外层
 * 分割挤成不可用的窄条。
 */
export function panelMinSize(layout: DockLayout, formId: string, horizontal: boolean): number {
    const form = layout.forms[formId];
    if (!isPanelForm(form) || !form?.childRootId) return 0;
    const tree = readRoot(layout, form.childRootId);
    if (!tree) return 0;
    return subtreeMinSize(layout, tree, horizontal, 0);
}

function subtreeMinSize(
    layout: DockLayout,
    node: DockNode,
    horizontal: boolean,
    depth: number,
): number {
    if (depth > 64) return 0;
    if (node.t === "split") {
        return Math.max(
            subtreeMinSize(layout, node.a, horizontal, depth + 1),
            subtreeMinSize(layout, node.b, horizontal, depth + 1),
        );
    }
    let min = 0;
    for (const formId of node.tabs) {
        const form = layout.forms[formId];
        if (form && isPanelForm(form)) {
            min = Math.max(min, panelMinSize(layout, formId, horizontal));
            continue;
        }
        const definition = form ? getPanel(form.panelId) : undefined;
        const value = horizontal ? definition?.minWidth : definition?.minHeight;
        min = Math.max(min, value ?? 0);
    }
    return min;
}

/**
 * 面板窗体在通用代码路径里需要的"注册表定义"。
 *
 * 叶窗体原样返回注册表项；面板回填一份**派生**定义 —— 可拆性来自全体子窗体，
 * 标题键是面板的兜底文案，其余按常规默认。消费方（`floatTitleBarActions`、
 * `detachFormToWindow`）因此不需要知道自己面对的是不是面板。
 */
export function synthesizePanelDefinition(
    layout: DockLayout,
    formId: string,
): PanelDefinition | undefined {
    const form = layout.forms[formId];
    if (!isPanelForm(form)) return form ? getPanel(form.panelId) : undefined;
    return {
        id: DOCK_PANEL_FORM,
        titleKey: "dock_panel_title",
        defaultWidth: PANEL_DEFAULT_WIDTH,
        defaultHeight: PANEL_DEFAULT_HEIGHT,
        minWidth: 200,
        minHeight: 140,
        singleton: false,
        dockable: true,
        detachable: isPanelDetachable(layout, formId).ok,
        excludeFromWindowMenu: true,
    };
}

/**
 * 面板的默认浮动尺寸。
 *
 * 空面板的"新建落点"与独立窗口的兜底尺寸都从这里取 —— 两处各写一份 720×480
 * 迟早漂移。作为容器，默认给一个能直接往里排布的中等偏大尺寸。
 */
export const PANEL_DEFAULT_WIDTH = 720;
export const PANEL_DEFAULT_HEIGHT = 480;

/** 面板的直接成员窗体 id（嵌套面板计为一个成员）。 */
export function panelMemberFormIds(layout: DockLayout, formId: string): string[] {
    const form = layout.forms[formId];
    if (!isPanelForm(form) || !form?.childRootId) return [];
    const tree = readRoot(layout, form.childRootId);
    return tree ? collectDockedForms(tree) : [];
}

/** 面板窗体构造时的公共形状（`childRootId` 由调用方按新分配的根 id 填入）。 */
export function makePanelForm(formId: string, childRootId: string): DockForm {
    return { id: formId, panelId: DOCK_PANEL_FORM, float: null, floating: false, childRootId };
}
