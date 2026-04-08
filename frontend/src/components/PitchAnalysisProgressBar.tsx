/**
 * 音高分析进度条组件 (Tasks 3.9-3.12)
 *
 * 自动轮询后端音高分析进度，显示实时进度和预计剩余时间
 */

import React, { useEffect, useState } from "react";
import { coreApi } from "../services/api/core";
import type { PitchProgressPayload } from "../types/api";
import { ProgressBar } from "./ProgressBar";
import { useI18n } from "../i18n/I18nProvider";

export interface PitchAnalysisProgressBarProps {
    /** 轮询间隔（毫秒），默认500ms */
    pollInterval?: number;
    /** 组件类名 */
    className?: string;
    /** 完成后淡出时长（毫秒），默认1000ms */
    fadeOutDuration?: number;
}

export const PitchAnalysisProgressBar: React.FC<PitchAnalysisProgressBarProps> = ({
    pollInterval = 500,
    className = "",
    fadeOutDuration = 1000,
}) => {
    const { t } = useI18n();
    const [progress, setProgress] = useState<PitchProgressPayload | null>(null);
    const [isVisible, setIsVisible] = useState(false);
    const [isFadingOut, setIsFadingOut] = useState(false);

    useEffect(() => {
        let intervalId: ReturnType<typeof setInterval> | null = null;
        let fadeOutTimer: ReturnType<typeof setTimeout> | null = null;

        const pollProgress = async () => {
            try {
                const result = await coreApi.getPitchAnalysisProgress();

                if (result) {
                    setProgress(result);
                    setIsVisible(true);
                    setIsFadingOut(false);
                } else if (progress !== null) {
                    // 分析完成，启动淡出动画 (Task 3.12)
                    setIsFadingOut(true);
                    fadeOutTimer = setTimeout(() => {
                        setIsVisible(false);
                        setProgress(null);
                        setIsFadingOut(false);
                    }, fadeOutDuration);
                }
            } catch (error) {
                console.error("Failed to poll pitch analysis progress:", error);
            }
        };

        // 初始查询
        pollProgress();

        // 启动轮询 (Task 3.9)
        intervalId = setInterval(pollProgress, pollInterval);

        return () => {
            if (intervalId) clearInterval(intervalId);
            if (fadeOutTimer) clearTimeout(fadeOutTimer);
        };
    }, [pollInterval, fadeOutDuration, progress]);

    if (!isVisible || !progress) {
        return null;
    }

    // 计算进度百分比 (Task 3.10)
    const percentage = Math.max(0, Math.min(100, progress.progress * 100));

    // 格式化进度标签：显示当前 clip 名称和进度计数
    const completedClips = progress.completedClips ?? 0;
    const totalClips = progress.totalClips ?? 0;
    const clipCountStr = totalClips > 0 ? ` (${completedClips}/${totalClips})` : "";
    const clipNameStr = progress.currentClipName ? ` "${progress.currentClipName}"` : "";
    const label = t("pitch_analyzing_clips")
        .replace("{clipName}", clipNameStr)
        .replace("{clipCount}", clipCountStr)
        .replace("{percentage}", String(Math.round(percentage)));

    // 预计剩余时间（秒）
    const estimatedRemainingSec = progress.etaSeconds ?? null;

    return (
        <div
            className={`transition-opacity duration-${fadeOutDuration} ${
                isFadingOut ? "opacity-0" : "opacity-100"
            } ${className}`}
        >
            <ProgressBar
                percentage={percentage}
                label={label}
                estimatedRemaining={estimatedRemainingSec}
                className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-3 border border-gray-700"
            />
        </div>
    );
};
