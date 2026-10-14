/**
 * 状态片原语 —— 状态栏与徽标类小标记的唯一外观来源。
 *
 * 【为什么需要它】`ParamDataLoadingChip` 当初被抽出来，注释里写明了理由
 * （避免订阅装在包裹整个应用树的大组件上）。但抽出来之后，`App.tsx` 里又
 * 原地手写了 **5 份结构完全相同**的状态片（`px-1 py-0` + `accent-3/accent-11`
 * + `fontSize: "var(--qt-fs-xs)"px`），另有一处 `green-3/green-11` 变体。
 *
 * 抽出来的组件没被复用，等于没抽。本原语把"状态片"这个形态固定下来，
 * 并提供 `tone` 表示语义，使 `App.tsx` 的 6 处可以收敛为同一个组件。
 *
 * 颜色走 Radix 的语义色阶变量（`--accent-*` / `--gray-*` / `--red-*` /
 * `--green-*` / `--amber-*`），因此自动跟随用户的强调色与明暗主题。
 */
import type { ReactNode } from "react";

import { cx } from "./cx";

/**
 * 状态语义。与 Radix 主题色阶一一对应，因此会跟随明暗主题自动取到
 * 对比度合适的色号（`:root[data-theme]` 切换时无需额外处理）。
 */
export type AppStatusTone = "neutral" | "accent" | "success" | "warning" | "danger";

const TONE_STYLE: Record<AppStatusTone, { background: string; color: string }> = {
    neutral: { background: "var(--gray-3)", color: "var(--gray-11)" },
    accent: { background: "var(--accent-3)", color: "var(--accent-11)" },
    success: { background: "var(--green-3)", color: "var(--green-11)" },
    warning: { background: "var(--amber-3)", color: "var(--amber-11)" },
    danger: { background: "var(--red-3)", color: "var(--red-11)" },
};

export interface AppStatusChipProps {
    children: ReactNode;
    tone?: AppStatusTone;
    /** 等宽数字，用于会跳动的计数，避免文字宽度抖动。 */
    tabular?: boolean;
    className?: string;
    /**
     * 悬停提示（项目自定义气泡，走 `data-tooltip` 通道）。
     *
     * 【为什么与 `title` 并存】与 `ui/ListRow.tsx` 同一取舍：`tooltip` 走项目
     * 自定义气泡（样式/换行/明暗主题一致，且同时服务于无障碍朗读），`title` 是
     * 浏览器原生提示（延迟长、样式不可控、深色主题下常撞色）。新代码用 `tooltip`。
     *
     * 【为什么多行可以是一个字符串】气泡是 `white-space: pre-wrap`
     * （`index.css:1594`），所以 `\n` 会被如实渲染成换行 —— 这正是
     * `dock/DockLayoutMenu` 的 `dockDragHint` 两行提示采用的形式。
     */
    tooltip?: string;
    /** 原生浏览器提示。新代码优先用 `tooltip`。 */
    title?: string;
}

/**
 * 状态片。
 *
 * @example
 * <AppStatusChip tone="accent">{t("common_loading")}</AppStatusChip>
 * <AppStatusChip tone="danger" tooltip={t("why_failed")}>{errorCount}</AppStatusChip>
 */
export function AppStatusChip({
    children,
    tone = "neutral",
    tabular = false,
    className,
    tooltip,
    title,
}: AppStatusChipProps) {
    const { background, color } = TONE_STYLE[tone];
    return (
        <span
            title={title}
            // 【为什么只写一个属性】与 `AppListRow` 同口径（`ui/ListRow.tsx:166`）：
            // `data-tooltip` 就是自定义通道的唯一入口，命中判据在
            // `AppTooltipProvider`（`components/AppTooltip.tsx:256`）。空串与
            // undefined 都不写，免得"命中了一个空元素"让气泡反复显示空内容。
            data-tooltip={tooltip || undefined}
            className={cx(
                "hs-type-mono shrink-0 rounded px-1 py-0 font-medium leading-4",
                tabular && "tabular-nums",
                className,
            )}
            // 状态片比正文小一档：恢复迁移前 11px（hs-type-mono 是 12px）
            style={{ background, color, fontSize: "var(--qt-fs-xs)" }}
        >
            {children}
        </span>
    );
}
