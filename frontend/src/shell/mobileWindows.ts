/**
 * 移动抽屉的「窗口」分区：**打开 / 关闭每一个面板**的唯一入口。
 *
 * 【为什么需要这一块】桌面有「视图 → 窗口」菜单，移动端此前完全没有对应物：
 * 标签条上的关闭按钮（叉）一旦被误触，那个面板就从屏幕上消失了，而**没有任何
 * 地方能把它打开回来** —— 用户只能重启应用，甚至重启也未必恢复（布局是持久化的，
 * 关掉的面板在下次启动时仍然是关的）。这是"误触关闭"真正的破坏力所在：
 * 不是关错了一次，而是关错之后无路可退。
 *
 * 因此窗口分区承担三件事：
 * 1. **列出全部面板**（不只是打开的），每行一个开关 —— 关掉的能重新打开；
 * 2. 给出「关闭当前窗口」这个**显式**入口，替代标签上的小叉；
 * 3. 把「重置布局」放在同一处 —— 布局被拖乱之后，用户需要一个确定能回到原点的按钮。
 *
 * 【为什么是纯函数】抽屉的分区模型要能被单测直接断言（"关掉的面板仍在列表里，
 * 且 `visible === false`"），而不是靠挂载整个抽屉去数 DOM 节点。
 */

import type { DockLayout } from "../features/dock/dockTypes";
import { listPanelEntriesFromLayout } from "../features/dock/dockApi";
import type { MessageKey } from "../i18n/messages";

/** 一行窗口开关。 */
export interface MobileWindowRow {
    panelId: string;
    /** 面板标题键（与桌面「视图 → 窗口」同一份词条）。 */
    titleKey: MessageKey;
    /** 当前是否显示在屏幕上（停靠或浮动都算）。 */
    visible: boolean;
}

/**
 * 从布局派生出窗口清单。
 *
 * 【为什么直接复用 `listPanelEntriesFromLayout`】桌面「视图 → 窗口」的勾选状态、
 * 面板清单与排除规则（`excludeFromWindowMenu`）都住在那里。另写一份筛选会让
 * "手机上少一个面板 / 多一个设置面板"这类差异悄悄出现，而两处都看起来是对的。
 */
export function buildMobileWindowRows(layout: DockLayout): MobileWindowRow[] {
    return listPanelEntriesFromLayout(layout).map((entry) => ({
        panelId: entry.panelId,
        titleKey: entry.titleKey,
        visible: entry.visible,
    }));
}

/** 分区标题右侧的摘要，例如 "2/4"。没有可列的面板时返回空串。 */
export function windowSummary(rows: readonly MobileWindowRow[]): string {
    if (rows.length === 0) return "";
    const visible = rows.filter((row) => row.visible).length;
    return `${visible}/${rows.length}`;
}
