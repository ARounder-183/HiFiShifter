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
 * 选中态额外加 `data-selected` 供外部样式挂钩。
 *
 * 【键盘光标是第三个状态，走另一条通道】`active` 表示"键盘现在停在哪一行"
 * （roving tabindex 的活动行），它**不是**选中态：选中可能是 0..N 个，光标恒为
 * 一个。背景通道已被 hover/selected/playing 占满，再挤一档就分不出来，因此
 * 光标用 1px 内嵌强调色描边（边框通道）表达 —— 与背景正交，可叠加且各自可辨。
 */
import { forwardRef } from "react";
import type {
    CSSProperties,
    MouseEvent as ReactMouseEvent,
    PointerEvent as ReactPointerEvent,
    ReactNode,
} from "react";

import { cx } from "./cx";

export type AppListRowDensity = "compact" | "default";

export interface AppListRowProps {
    children: ReactNode;
    /** 选中态。 */
    selected?: boolean;
    /**
     * 键盘光标所在行（listbox 的 "active option"）。
     *
     * 【为什么与 `selected` 分开】两者是独立的两件事：`selected` 是"哪些被选中"
     * （可能 0..N 个，由鼠标 Ctrl/Shift 决定），`active` 是"键盘现在停在哪一行"
     * （恒为 0 或 1 个）。文件浏览器此前把 `selected` 接给鼠标多选、键盘光标没有
     * 表达，于是 ↑↓ 只能借用浏览器给 `<div tabindex>` 画的默认 outline ——
     * 一圈紧贴内容盒、与行内边距不对齐的深色 halo。
     *
     * 【为什么用描边而不是再挤一档背景色】背景通道已经被 hover(10%) /
     * selected(22%) / playing(20%) 占满，再挤一档就分不出来。描边走**边框通道**，
     * 与背景正交，可以叠加且各自可辨。
     */
    active?: boolean;
    /** 悬停/选中时的强调基调，`danger` 用于破坏性目标的悬停反馈。 */
    intent?: "default" | "danger";
    density?: AppListRowDensity;
    disabled?: boolean;
    onClick?: (event: ReactMouseEvent<HTMLDivElement>) => void;
    onDoubleClick?: (event: ReactMouseEvent<HTMLDivElement>) => void;
    onContextMenu?: (event: ReactMouseEvent) => void;
    /**
     * 指针按下（文件浏览器的拖拽起点）。
     *
     * 【为什么加这个】拖拽行可能**不可点击**（例如只能拖入时间轴的 MIDI / 工程
     * 文件），此时 `interactive` 为 false，但仍需要指针按下回调才能发起拖拽。
     */
    onPointerDown?: (event: ReactPointerEvent<HTMLDivElement>) => void;
    /** 行获得焦点。roving tabindex 下用它把"活动行"同步到真实焦点。 */
    onFocus?: () => void;
    /** roving tabindex：活动行为 `0`，其余为 `-1`；省略则不可聚焦（旧行为）。 */
    tabIndex?: number;
    /**
     * 显式列表项角色。
     *
     * 仅凭 `onClick` / `onDoubleClick` 推导会让"只能拖拽"与"暂不可用"的行在
     * listbox 里没有角色。文件浏览器对所有行显式传 `"option"`。
     */
    role?: "option";
    /**
     * 悬停提示（项目自定义气泡，走 `data-tooltip` 通道）。
     *
     * 【为什么与 `title` 并存】`title` 是浏览器原生提示：延迟长、样式不可控、
     * 深色主题下常与页面撞色。需要与全应用一致的气泡时用本字段。
     */
    tooltip?: string;
    /** 原生浏览器提示。新代码优先用 `tooltip`。 */
    title?: string;
    className?: string;
    /** 供虚拟化列表用；普通列表省略。 */
    style?: CSSProperties;
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
export const AppListRow = forwardRef<HTMLDivElement, AppListRowProps>(function AppListRow(
    {
        children,
        selected = false,
        active = false,
        intent = "default",
        density = "compact",
        disabled = false,
        onClick,
        onDoubleClick,
        onContextMenu,
        onPointerDown,
        onFocus,
        tabIndex,
        role,
        tooltip,
        title,
        className,
        style,
        "data-testid": testId,
    },
    ref,
) {
    const interactive = Boolean(onClick || onDoubleClick);
    // 显式 `role="option"`（含仅可拖拽/暂不可用的行）与可点击行一样进入列表语义，
    // 否则 listbox 里会混入没有角色、键盘也无从表达的行。
    const isOption = role === "option" || interactive;

    return (
        <div
            ref={ref}
            role={isOption ? "option" : undefined}
            aria-selected={isOption ? selected : undefined}
            aria-disabled={disabled || undefined}
            tabIndex={tabIndex}
            data-selected={selected || undefined}
            data-active={active || undefined}
            data-tooltip={tooltip}
            title={title}
            style={style}
            data-testid={testId}
            onClick={disabled ? undefined : onClick}
            onDoubleClick={disabled ? undefined : onDoubleClick}
            onContextMenu={onContextMenu}
            onPointerDown={onPointerDown}
            onFocus={onFocus}
            className={cx(
                "hs-type-body group flex items-center gap-1.5",
                DENSITY_CLASS[density],
                interactive && !disabled && "cursor-pointer",
                disabled && "cursor-default opacity-50",
                // 键盘光标走描边通道：先关掉浏览器给可聚焦元素的默认 outline
                // （在 22px 行上那是一圈与内边距不对齐的脏 halo），再画自己的。
                "focus:outline-none",
                active &&
                    "outline outline-1 outline-offset-[-1px] outline-[color:var(--qt-focus-ring)]",
                intent === "danger"
                    ? "hover:bg-qt-danger-bg hover:text-qt-danger-text"
                    : "hover:bg-[color-mix(in_oklab,var(--qt-highlight)_10%,transparent)]",
                selected &&
                    intent === "default" &&
                    "bg-[color-mix(in_oklab,var(--qt-highlight)_22%,transparent)]",
                className,
            )}
        >
            {children}
        </div>
    );
});
