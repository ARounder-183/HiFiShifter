/*
 * 进度条：任务进度、标签文本，以及可选的取消按钮。
 *
 * 【为什么取值必须走令牌】这里曾经用 Tailwind 的固定调色板（`bg-gray-700` 轨道、
 * `bg-blue-500` 进度、`text-gray-400` 读数、`emerald` 完成标记）。固定调色板意味着
 * 这个组件**不跟随主题**：浅色主题下它是深灰轨道配白字，与周围 chrome 完全脱节；
 * 用户在 AppearanceWindow 里换强调色也影响不到它。进度条是全局共享组件（导出、
 * 渲染、分析都经由它），一处写死就是处处写死。
 */

import React from "react";
import { AppBusy } from "../ui";
import { useI18n } from "../i18n/I18nProvider";

export interface ProgressBarProps {
    percentage: number; // 0-100
    label?: string;
    completed?: boolean;
    showCancel?: boolean;
    onCancel?: () => void;
    estimatedRemaining?: number | null; // seconds
    className?: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
    percentage,
    label,
    completed = false,
    showCancel = false,
    onCancel,
    estimatedRemaining,
    className = "",
}) => {
    const { t } = useI18n();
    const clampedPercentage = Math.max(0, Math.min(100, percentage));

    const formatTime = (seconds: number): string => {
        if (seconds < 1) return "< 1s";
        if (seconds < 60) return `${Math.ceil(seconds)}s`;
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.ceil(seconds % 60);
        return `${minutes}m ${remainingSeconds}s`;
    };

    return (
        <div className={`flex flex-col gap-2 ${className}`}>
            <div className="flex items-center justify-between text-qt-md">
                <div className="flex items-center gap-2">
                    {completed ? (
                        <span
                            className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-qt-success-border text-qt-success-text"
                            aria-hidden
                        >
                            √
                        </span>
                    ) : (
                        <AppBusy />
                    )}
                    {label && <span className="text-qt-text">{label}</span>}
                    <span className="text-qt-text-muted">{clampedPercentage.toFixed(0)}%</span>
                </div>
                {estimatedRemaining !== null && estimatedRemaining !== undefined && (
                    <span className="text-qt-xs text-qt-text-muted">
                        {t("progress_est_remaining").replace(
                            "{time}",
                            formatTime(estimatedRemaining),
                        )}
                    </span>
                )}
            </div>

            <div className="relative h-2 w-full overflow-hidden rounded-full bg-qt-meter-well">
                <div
                    className="absolute left-0 top-0 h-full bg-qt-accent transition-all duration-300"
                    style={{ width: `${clampedPercentage}%` }}
                />
            </div>

            {showCancel && onCancel && (
                <button
                    onClick={onCancel}
                    className="self-end rounded px-3 py-1 text-qt-xs text-qt-text-muted hover:bg-qt-button-hover hover:text-qt-text transition-colors"
                    type="button"
                >
                    {t("progress_cancel")}
                </button>
            )}
        </div>
    );
};
