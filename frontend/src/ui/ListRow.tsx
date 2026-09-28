/**
 * 列表行原语 —— 紧凑列表（文件浏览器 / 快速搜索 / 操作记录 / 轨道头）的行外观与选中态。
 *
 * 【它解决的问题】审查发现"选中"和"悬停"这两个状态在四个列表里有**三种互不相认的
 * 编码方式**：
 *   - `color-mix(in oklab, var(--qt-highlight) N%, transparent)`，N 取 10/12/20/25
 *     （文件浏览器、快速搜索）
 *   - `bg-qt-highlight/20` + `font-medium`（操作记录）
 *   - `bg-qt-button-hover` **同时用于悬停与选中**（轨道头）—— 于是轨道头里
 *     "选中"与"悬停"在视觉上完全无法区分，只能靠一条 4px 色条的透明度暗示
 *
 * 以及行内边距三种：`py-[3px]`、`py-[4px]`、Radix `py="1"`。
 *
 * 本原语把这两个状态各归一为一档：悬停 10%、选中 22%，行高与内边距取令牌。
 * 选中态额外加 `data-selected` 供外部的键盘导航样式挂钩。
 */
import type { ReactNode } from "react";

import { cx } from "./cx";

export type AppListRowDensity = "compact" | "default";

export interface AppListRowProps {
    children: ReactNode;
    /** 选中态。 */
    selected?: boolean;
    /** 悬停/选中时的强调基调，`danger` 用于破坏性目标的悬停反馈。 */
    intent?: "default" | "danger";
    density?: AppListRowDensity;
    disabled?: boolean;
    onClick?: () => void;
    onDoubleClick?: () => void;
    onContextMenu?: (event: React.MouseEvent) => void;
    title?: string;
    className?: string;
    /** 供虚拟化列表用；普通列表省略。 */
    style?: React.CSSProperties;
    "data-testid"?: string;
}

const DENSITY_CLASS: Record<AppListRowDensity, string> = {
    // 22px 行高：文件浏览器与操作记录的历史取值（py-[3px] + 16px 行框）
    compact: "px-2 py-qt-1 min-h-[22px]",
    // 24px 行高：快速搜索的历史取值
    default: "px-2 py-qt-2 min-h-[24px]",
};

/**
 * 紧凑列表行。
 *
 * @example
 * <AppListRow selected={id === activeId} onClick={() => select(id)}>
 *   <span className="truncate">{name}</span>
 * </AppListRow>
 */
export function AppListRow({
    children,
    selected = false,
    intent = "default",
    density = "compact",
    disabled = false,
    onClick,
    onDoubleClick,
    onContextMenu,
    title,
    className,
    style,
    "data-testid": testId,
}: AppListRowProps) {
    const interactive = Boolean(onClick || onDoubleClick);

    return (
        <div
            role={interactive ? "option" : undefined}
            aria-selected={interactive ? selected : undefined}
            aria-disabled={disabled || undefined}
            data-selected={selected || undefined}
            title={title}
            style={style}
            data-testid={testId}
            onClick={disabled ? undefined : onClick}
            onDoubleClick={disabled ? undefined : onDoubleClick}
            onContextMenu={onContextMenu}
            className={cx(
                "hs-type-body group flex items-center gap-1.5",
                DENSITY_CLASS[density],
                interactive && !disabled && "cursor-pointer",
                disabled && "cursor-default opacity-50",
                intent === "danger"
                    ? "hover:bg-qt-danger-bg hover:text-qt-danger-text"
                    : "hover:bg-[color-mix(in_oklab,var(--qt-highlight)_10%,transparent)]",
                selected && intent === "default" && "bg-[color-mix(in_oklab,var(--qt-highlight)_22%,transparent)]",
                className,
            )}
        >
            {children}
        </div>
    );
}
