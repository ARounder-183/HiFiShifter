/**
 * 快捷键设置窗口的分类导航栏 —— 左栏固定、始终可见的一组"跳到哪个场景"的入口。
 *
 * 【解决什么问题】原窗口里 14 个分组标题排在同一条长列表上，滚动即消失：用户滚到
 * 中段就失去了"我现在在哪一组、还有哪些组"的路标。这里把分组提升为常驻的一栏，
 * 兼作搜索的范围过滤器。
 *
 * 【为什么"始终可见"重要】快捷键设置的典型用法是"我改过几条，想看看都在哪儿"。
 * 这不是一次性查找，而是反复地在几组之间跳 —— 每次跳转都要先滚回顶部重新定位的
 * 话，导航栏就退化成了一个书签。
 */
import { AppListRow } from "../../../ui";
import type { ActionMeta } from "../../../features/keybindings/types";

/** 导航栏条目的分类取值。`null` 代表"全部"。 */
export type KeybindingsNavGroup = ActionMeta["group"] | null;

export interface KeybindingsNavItem {
    group: KeybindingsNavGroup;
    /** 已本地化的显示名。 */
    label: string;
    /** 该组内的动作总数（不受搜索影响）。 */
    total: number;
    /**
     * 当前搜索下该组的命中数。`undefined` 表示没有查询（此时不显示计数徽章）。
     */
    matchCount?: number;
    /** 组内是否有被用户改过的绑定。 */
    customized: boolean;
}

export interface KeybindingsNavRailProps {
    items: KeybindingsNavItem[];
    activeGroup: KeybindingsNavGroup;
    onSelect: (group: KeybindingsNavGroup) => void;
    /** 无障碍标签（已本地化）。 */
    ariaLabel: string;
    customizedHint: string;
}

export function KeybindingsNavRail({
    items,
    activeGroup,
    onSelect,
    ariaLabel,
    customizedHint,
}: KeybindingsNavRailProps) {
    return (
        <nav
            role="listbox"
            aria-label={ariaLabel}
            /*
             * 【为什么这里可以有独立滚动】`AppDialog` 用 `bodyLayout="pane"` 之后，
             * body 不再滚动，而是提供一个确定高度的 flex 容器；本栏作为
             * `shrink-0` + `min-h-0` 的兄弟节点，可以在**不产生第二条页面级滚动条**
             * 的前提下自己滚。这也是不用 `max-h-[Npx]` 的原因 ——
             * 那会在 `designSystemGates.test.ts` 的"有界滚动盒"门禁上报错。
             */
            className="hide-v-scrollbar w-44 shrink-0 overflow-y-auto border-r border-qt-border pr-1"
        >
            {items.map((item) => {
                const dimmed = item.matchCount === 0;
                return (
                    <AppListRow
                        key={item.group ?? "__all__"}
                        role="option"
                        density="default"
                        selected={item.group === activeGroup}
                        /*
                         * 【为什么 0 命中的组仍然可点】藏起来会让"这个分类去哪了"变成
                         * 一个新问题；禁用则让点击毫无反馈，像程序坏了。保留可点、
                         * 只做视觉减淡，用户点了会看到该组的空态 —— 这是可解释的结果。
                         */
                        className={dimmed ? "opacity-50" : undefined}
                        onClick={() => onSelect(item.group)}
                        title={item.customized ? customizedHint : undefined}
                    >
                        <span className="hs-type-label flex-1 truncate">{item.label}</span>
                        {item.customized && (
                            /*
                             * 已改动标记：一个小圆点 + 无障碍说明。
                             *
                             * 【为什么需要它】这是搜索与分类都覆盖不到的第三类需求
                             * ——"我改过哪些"。用户调整完若干绑定后想复查时，一眼就能
                             * 看到动过的组，不必逐条比对绿色按钮。
                             */
                            <span
                                aria-hidden
                                className="shrink-0 rounded-qt-pill"
                                style={{
                                    width: 6,
                                    height: 6,
                                    background: "var(--qt-accent)",
                                }}
                            />
                        )}
                        {item.matchCount !== undefined && (
                            <span className="hs-type-caption shrink-0 tabular-nums">
                                {item.matchCount}
                            </span>
                        )}
                    </AppListRow>
                );
            })}
        </nav>
    );
}
