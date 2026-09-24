/**
 * 面板工具条原语 —— 面板头部**唯一**的共享外观定义。
 *
 * 【为什么需要它】`文件浏览器` 与 `记事本` 各自手写了一套头部：一个用 Radix
 * `Text size="2"` + `IconButton` + `h-8`，另一个用原生 `span text-xs` + emoji 字形
 * + `py-1.5`。结果是一大一小、一粗一细、图标与 emoji 混排，同一屏里看得出不协调。
 * 把度量收进一个原语后，"像不像一家人"就不再依赖每个面板作者的自觉。
 *
 * 【为什么没有标题槽】标题与关闭属于**窗框**（停靠时是标签行、浮动时是浮动标题栏、
 * 独立窗口时是操作系统标题栏），不属于面板。面板在这里重复画一遍就是三处重复。
 * 因此本组件只承载**面板独有的功能按钮**——那些窗框不可能知道的东西
 * （打开目录 / 刷新 / 富文本-源码-分屏切换 / 附件 / 设置）。
 */
import type { ReactNode } from "react";

/** 工具条高度（与标签行一致，避免同一窗口里出现三种不同的横条高度）。 */
export const PANEL_TOOLBAR_PX = 26;

/** 图标尺寸：统一 12px，禁止 emoji 字形（跨平台渲染差异正是"不协调"的主因）。 */
const ICON_PX = 12;

/**
 * 面板工具条容器。
 *
 * @param leading 左侧按钮组（如记事本的模式切换）；可省略。
 * @param trailing 右侧按钮组（图标按钮为主）。
 */
export function PanelToolbar({ leading, trailing }: { leading?: ReactNode; trailing?: ReactNode }) {
    return (
        <div
            className="flex shrink-0 items-center justify-between gap-2 border-b border-qt-border px-2"
            style={{ height: PANEL_TOOLBAR_PX, background: "var(--qt-window)" }}
        >
            <div className="flex min-w-0 items-center gap-1">{leading}</div>
            <div className="flex shrink-0 items-center gap-1">{trailing}</div>
        </div>
    );
}

/**
 * 工具条图标按钮。
 *
 * @param icon 图标节点（Radix 图标；尺寸由本组件统一覆盖）。
 * @param tooltip 悬停提示文本，走项目自定义 tooltip（`data-tooltip`）。
 * @param active 是否为激活态（用于可切换的按钮）。
 */
export function PanelToolbarButton({
    icon,
    tooltip,
    active = false,
    disabled = false,
    onClick,
}: {
    icon: ReactNode;
    tooltip: string;
    active?: boolean;
    disabled?: boolean;
    onClick: () => void;
}) {
    return (
        <button
            type="button"
            data-tooltip={tooltip}
            aria-label={tooltip}
            aria-pressed={active || undefined}
            disabled={disabled}
            onClick={onClick}
            className="flex items-center justify-center rounded text-qt-text-muted transition-colors hover:bg-qt-hover hover:text-qt-text disabled:cursor-default disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-qt-text-muted"
            style={{
                width: 20,
                height: 20,
                color: active ? "var(--qt-highlight)" : undefined,
                background: active
                    ? "color-mix(in srgb, var(--qt-highlight) 16%, transparent)"
                    : undefined,
            }}
        >
            {icon}
        </button>
    );
}

/**
 * 工具条文字按钮（用于"富文本 / 源码 / 分屏"这类需要可读标签的互斥切换）。
 *
 * 与 `PanelToolbarButton` 共用同一套高度、字号、圆角与 hover，只是内容为文字。
 */
export function PanelToolbarTextButton({
    label,
    tooltip,
    active = false,
    onClick,
}: {
    label: string;
    tooltip?: string;
    active?: boolean;
    onClick: () => void;
}) {
    const tip = tooltip ?? label;
    return (
        <button
            type="button"
            data-tooltip={tip}
            aria-label={tip}
            aria-pressed={active}
            onClick={onClick}
            className="rounded px-1.5 text-qt-text-muted transition-colors hover:bg-qt-hover hover:text-qt-text"
            style={{
                height: 20,
                fontSize: 11,
                lineHeight: "20px",
                color: active ? "var(--qt-highlight)" : undefined,
                background: active
                    ? "color-mix(in srgb, var(--qt-highlight) 16%, transparent)"
                    : undefined,
            }}
        >
            {label}
        </button>
    );
}

/** 图标尺寸常量出口：面板在需要自绘图标时保持一致。 */
export { ICON_PX as PANEL_TOOLBAR_ICON_PX };
