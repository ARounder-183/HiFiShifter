/**
 * 工具栏原语 —— chrome 横条与其按钮的唯一外观来源。
 *
 * 【它解决的问题】审查发现 chrome 横条有 **6 种高度**（32 / 28 / 26 / 48 / 24 / 20px）、
 * **4 种按钮尺寸**（24 / 20 / 18 / 16 / 20-文字），以及 **5 种分隔线做法**：
 * Radix `<Separator size="2">`（ActionBar 7 处）、手搓 `Box` 1×18px（PianoRoll）、
 * CSS 类 `.hs-notebook-toolbar-sep`（记事本）、每标签的 `border-right`（dock 标签栏）、
 * 以及菜单里的 `my-1 border-t`。
 *
 * 这些数字本身不一定错 —— 26px 的面板条与 32px 的主工具栏是不同语境，强行拉平
 * 可能更难看。**问题在于它们是散落的魔法数字**：第 7 个横条的作者无从知道该取哪个值，
 * 于是又发明一个。本原语把高度收进 `--qt-bar-*` 令牌，分隔线收进一个组件，
 * 于是"该取哪个值"有了唯一答案，而语境差异仍可表达。
 *
 * 【为什么不像 PanelToolbar 那样只有固定高度】横条高度是**语境**属性（主工具栏 /
 * 紧凑面板 / 状态栏），不是组件的私有常量。因此由调用方从枚举里选，而不是由组件写死。
 */
import type { ReactNode } from "react";

import { AppDensityProvider } from "./density";
import { cx } from "./cx";

/** 横条高度档位，取值来自 `--qt-bar-*` 令牌。 */
export type AppToolbarHeight = "main" | "compact" | "title" | "status";

const HEIGHT_CLASS: Record<AppToolbarHeight, string> = {
    main: "h-qt-bar-main",
    compact: "h-qt-bar-compact",
    title: "h-qt-bar-title",
    status: "h-qt-bar-status",
};

export interface AppToolbarProps {
    height?: AppToolbarHeight;
    /** 左侧区域（可滚动部分）。 */
    leading?: ReactNode;
    /** 右侧区域（固定不滚动，如"更多"按钮）。 */
    trailing?: ReactNode;
    /** 底边分隔线。多数横条需要，浮层标题栏不需要（它贴着自己的内容）。 */
    border?: boolean;
    /**
     * 内容放不下时横向滚动。
     *
     * 默认 **false**（`overflow: visible`）。上一版默认开了滚动，有两个实测后果：
     *
     * 1. **吃掉高度**：26px 的横条里出现 10px 横向滚动条，24px 的按钮放不进
     *    剩下 16px —— 按钮被压扁。
     * 2. **裁剪弹层**：CSS 规范规定 `overflow-x: auto` 时 `overflow-y` 计算为
     *    `auto`，于是该容器成为**裁剪上下文**，放在工具栏里的下拉/气泡会被切掉。
     *
     * 因此"不裁剪"是更重要的默认；确实需要滚动的长工具栏显式打开，并且
     * 打开后**不要在工具栏里放弹层**。
     */
    scrollable?: boolean;
    className?: string;
}

/**
 * 工具栏容器。
 *
 * @example
 * <AppToolbar
 *     height="main"
 *     leading={<><AppToolbarGroup>…</AppToolbarGroup><AppToolbarSeparator /><AppToolbarGroup>…</AppToolbarGroup></>}
 *     trailing={<AppIconButton icon={<GearIcon />} tooltip={t("settings")} />}
 * />
 */
export function AppToolbar({
    height = "main",
    leading,
    trailing,
    border = true,
    scrollable = false,
    className,
}: AppToolbarProps) {
    return (
        <AppDensityProvider value="compact">
            <div
                className={cx(
                    "flex shrink-0 items-center justify-between gap-2 px-2",
                    HEIGHT_CLASS[height],
                    border && "border-b border-qt-border",
                    className,
                )}
                style={{ background: "var(--qt-window)" }}
            >
                {/*
                 * 默认 `overflow: visible` —— 不裁剪弹层、不产生吃掉高度的滚动条。
                 * 需要滚动的长工具栏由调用方显式打开 `scrollable`（见该属性说明）。
                 */}
                <div
                    className={cx(
                        "flex min-w-0 flex-1 items-center gap-1",
                        scrollable && "custom-scrollbar overflow-x-auto",
                    )}
                >
                    {leading}
                </div>
                {trailing ? (
                    <div className="flex shrink-0 items-center gap-1">{trailing}</div>
                ) : null}
            </div>
        </AppDensityProvider>
    );
}

/**
 * 组间分隔线。
 *
 * 取代此前 5 种做法。用 1px 竖线 + 令牌高度，不依赖 Radix `Separator`
 * （它带自己的颜色变量，与 `--qt-*` 主题体系是两套）。
 */
export function AppToolbarSeparator({ className }: { className?: string }) {
    return (
        <div aria-hidden="true" className={cx("mx-1 h-4 w-px shrink-0 bg-qt-border", className)} />
    );
}

/**
 * 一个**功能组**：换行时的最小单位。
 *
 * 【为什么必须自成一体】`shrink-0` 保证组内绝不折行（否则 `BPM:` 会和它的输入框分到
 * 两行）；`whitespace-nowrap` 保证组内文字不逐字断行。两者合起来，换行只能发生在
 * **组与组之间**，这正是"按功能分组换行"的定义。
 *
 * 【为什么分隔线在组**内部**（`divider`）而不是组与组之间】
 * 分隔线若是一个独立兄弟元素，它后面那一组换行时会把它留在**行首** —— 屏幕最左边
 * 孤零零一根竖线。CSS 没有"行首选择器"，靠测量 `offsetTop` 去补丁则要在每次
 * 重排后重新判断（实测窗口缩放时会有一次算错、之后不再重算的窗口）。
 * 把分隔线挂在组尾之后，"它跟着谁换行"由布局本身决定：组换行，分隔线跟着走，
 * 永远落在上一行的末尾 —— 一个像素的 JS 都不需要。
 *
 * `divider` 为 `false` 时组尾不带分隔线（最后那一组用）。
 */
export function AppToolbarGroup({
    children,
    divider = true,
}: {
    children: ReactNode;
    divider?: boolean;
}) {
    return (
        <div data-toolbar-group className="flex shrink-0 items-center gap-1 whitespace-nowrap">
            {children}
            {divider ? <AppToolbarSeparator /> : null}
        </div>
    );
}
