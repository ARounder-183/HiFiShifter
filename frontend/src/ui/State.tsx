/**
 * 状态展示原语：空态、加载态、状态片。
 *
 * 【为什么需要它】审查发现：
 *   - 空/加载占位文案 `<Text size="1" color="gray" className="px-3 py-4 block text-center">`
 *     在文件浏览器与快速搜索里被**逐字复制了 9 次**；
 *   - 加载指示有三种互不相干的表现：`LoadingSpinner` 组件（Tailwind 硬编码宽度）、
 *     Radix `Spinner`、以及字面量 `"..."` 文本；
 *   - 状态片抽出了 `ParamDataLoadingChip`，但 `App.tsx` 里又原地手写了 5 份结构相同的副本。
 *
 * 这三类都是"复制粘贴维持一致"的典型：第 10 个副本出现时不会有人拦。
 */
import type { ReactNode } from "react";
import { Spinner, Text } from "@radix-ui/themes";

import { cx } from "./cx";

export interface AppEmptyStateProps {
    children: ReactNode;
    /** 居中留白档位，`compact` 用于面板内部，`default` 用于整块空列表。 */
    size?: "compact" | "default";
    /**
     * 语义色调。`danger` 用于错误态（原先在文件浏览器里就地写 `color="red"`）。
     */
    tone?: "default" | "danger";
    className?: string;
}

/**
 * 空态 / 无匹配 / 加载中文案。
 *
 * @example
 * <AppEmptyState>{t("no_matching_files")}</AppEmptyState>
 * <AppEmptyState tone="danger">{t("fb_error")}</AppEmptyState>
 */
export function AppEmptyState({
    children,
    size = "default",
    tone = "default",
    className,
}: AppEmptyStateProps) {
    return (
        <Text
            size="1"
            color={tone === "danger" ? "red" : "gray"}
            className={cx("block text-center", size === "default" ? "px-3 py-4" : "px-3 py-2", className)}
        >
            {children}
        </Text>
    );
}

export interface AppBusyProps {
    /** 与文字同时显示时给出说明；只给指示器时省略。 */
    label?: ReactNode;
    /** `inline` 用于工具栏/状态栏，`block` 用于居中占位。 */
    layout?: "inline" | "block";
    className?: string;
}

/**
 * 加载指示。
 *
 * 统一到 Radix `Spinner`（跟随主题 accent 色），取代三种并存的表现。
 * 字面量 `"..."` 不是可访问的加载指示——屏幕阅读器读不出它。
 */
export function AppBusy({ label, layout = "inline", className }: AppBusyProps) {
    return (
        <span
            role="status"
            aria-live="polite"
            className={cx(
                "inline-flex items-center gap-2 text-qt-text-muted",
                layout === "block" && "w-full justify-center px-3 py-4",
                className,
            )}
        >
            <Spinner size="1" />
            {label ? <span className="hs-type-caption">{label}</span> : null}
        </span>
    );
}
