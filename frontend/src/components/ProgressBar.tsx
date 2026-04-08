/**
 * 进度条组件 (Task 6.2)
 *
 * 显示任务进度、标签文本和可选的取消按钮
 */

import React from "react";
import { LoadingSpinner } from "./LoadingSpinner";
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
            <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                    {completed ? (
                        <span
                            className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-emerald-400/80 text-emerald-300"
                            aria-hidden
                        >
                            √
                        </span>
                    ) : (
                        <LoadingSpinner size="sm" />
                    )}
                    {label && <span className="text-gray-300">{label}</span>}
                    <span className="text-gray-400">{clampedPercentage.toFixed(0)}%</span>
                </div>
                {estimatedRemaining !== null && estimatedRemaining !== undefined && (
                    <span className="text-xs text-gray-500">
                        {t("progress_est_remaining").replace(
                            "{time}",
                            formatTime(estimatedRemaining),
                        )}
                    </span>
                )}
            </div>

            <div className="relative h-2 w-full overflow-hidden rounded-full bg-gray-700">
                <div
                    className="absolute left-0 top-0 h-full bg-blue-500 transition-all duration-300"
                    style={{ width: `${clampedPercentage}%` }}
                />
            </div>

            {showCancel && onCancel && (
                <button
                    onClick={onCancel}
                    className="self-end rounded px-3 py-1 text-xs text-gray-400 hover:bg-gray-700 hover:text-gray-200 transition-colors"
                    type="button"
                >
                    {t("progress_cancel")}
                </button>
            )}
        </div>
    );
};
