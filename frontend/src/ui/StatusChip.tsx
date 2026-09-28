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
    title?: string;
}

/**
 * 状态片。
 *
 * @example
 * <AppStatusChip tone="accent">{t("common_loading")}</AppStatusChip>
 * <AppStatusChip tone="danger">{errorCount}</AppStatusChip>
 */
export function AppStatusChip({
    children,
    tone = "neutral",
    tabular = false,
    className,
    title,
}: AppStatusChipProps) {
    const { background, color } = TONE_STYLE[tone];
    return (
        <span
            title={title}
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
