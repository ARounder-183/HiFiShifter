/**
 * 异步音高刷新 Hook (Tasks 3.1-3.8)
 *
 * 提供音高参数异步刷新功能，包括：
 * - 启动/取消异步刷新任务
 * - 轮询任务状态和进度
 * - 预计剩余时间计算
 * - 防抖逻辑避免重复调用
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { coreApi } from "../services/api/core";
import type { PitchTaskStatusPayload } from "../types/api";

type PitchTaskStatus = PitchTaskStatusPayload["status"];

interface AsyncPitchRefreshState {
    isLoading: boolean;
    taskId: string | null;
    progress: number; // 0-100
    status: PitchTaskStatus | null;
    error: string | null;
    estimatedRemaining: number | null; // seconds
}

interface UseAsyncPitchRefreshResult extends AsyncPitchRefreshState {
    startRefresh: (rootTrackId: string) => Promise<void>;
    cancelRefresh: () => Promise<void>;
    reset: () => void;
}

const POLL_INTERVAL_MS = 500;
const DEBOUNCE_DELAY_MS = 300;

export function useAsyncPitchRefresh(): UseAsyncPitchRefreshResult {
    // Task 3.2: 定义状态
    const [state, setState] = useState<AsyncPitchRefreshState>({
        isLoading: false,
        taskId: null,
        progress: 0,
        status: null,
        error: null,
        estimatedRemaining: null,
    });

    // Task 3.7: 计算预计剩余时间
    const startTimeRef = useRef<number | null>(null);
    const pollTimerRef = useRef<number | null>(null);
    // Task 7.1: 维护 latestTaskId 引用，用于竞态处理
    const latestTaskIdRef = useRef<string | null>(null);
    // Task 3.8: 防抖逻辑
    const debounceTimerRef = useRef<number | null>(null);
    const isRefreshingRef = useRef(false);

    // Task 3.4: 实现轮询逻辑
    const pollStatus = useCallback(async (taskId: string) => {
        try {
            // Task 7.3: 检查是否为最新任务
            if (latestTaskIdRef.current !== taskId) {
                console.log(`[useAsyncPitchRefresh] Task ${taskId} is stale, stopping poll`);
                return;
            }

            const info = await coreApi.getPitchRefreshStatus(taskId);

            if (!info) {
                setState((prev) => ({
                    ...prev,
                    isLoading: false,
                    error: "Task expired or not found",
                }));
                return;
            }

            const elapsed = startTimeRef.current ? (Date.now() - startTimeRef.current) / 1000 : 0;

            let estimatedRemaining: number | null = null;
            if (info.progress > 0 && info.progress < 100 && elapsed > 0) {
                // Task 3.7: 根据已用时间和进度计算预计剩余时间
                const totalEstimated = (elapsed / info.progress) * 100;
                estimatedRemaining = Math.max(0, totalEstimated - elapsed);
            }

            setState((prev) => ({
                ...prev,
                progress: info.progress,
                status: info.status,
                error: info.error || null,
                estimatedRemaining,
            }));

            // Task 3.6: 任务完成或失败时停止轮询
            if (
                info.status === "completed" ||
                info.status === "failed" ||
                info.status === "cancelled"
            ) {
                setState((prev) => ({
                    ...prev,
                    isLoading: false,
                    taskId: null,
                }));
                latestTaskIdRef.current = null;
                if (pollTimerRef.current) {
                    clearInterval(pollTimerRef.current);
                    pollTimerRef.current = null;
                }
                startTimeRef.current = null;
            }
        } catch (error) {
            console.error("[useAsyncPitchRefresh] Poll error:", error);
            setState((prev) => ({
                ...prev,
                isLoading: false,
                error: error instanceof Error ? error.message : "Unknown error",
            }));
        }
    }, []);

    // Task 3.3: 实现 startRefresh() 函数
    const startRefresh = useCallback(
        async (rootTrackId: string) => {
            // Task 3.8: 防抖逻辑
            if (isRefreshingRef.current) {
                console.log("[useAsyncPitchRefresh] Debounce: refresh already in progress");
                return;
            }

            isRefreshingRef.current = true;

            if (debounceTimerRef.current) {
                clearTimeout(debounceTimerRef.current);
            }

            await new Promise((resolve) => {
                debounceTimerRef.current = setTimeout(resolve, DEBOUNCE_DELAY_MS);
            });

            try {
                // Task 7.1: 取消旧任务
                if (latestTaskIdRef.current) {
                    console.log(
                        `[useAsyncPitchRefresh] Cancelling previous task: ${latestTaskIdRef.current}`,
                    );
                    try {
                        await coreApi.cancelPitchTask(latestTaskIdRef.current);
                    } catch (err) {
                        console.warn("[useAsyncPitchRefresh] Failed to cancel previous task:", err);
                    }
                }

                // 停止旧的轮询定时器
                if (pollTimerRef.current) {
                    clearInterval(pollTimerRef.current);
                    pollTimerRef.current = null;
                }

                const taskId = await coreApi.startPitchRefreshTask(rootTrackId);

                latestTaskIdRef.current = taskId;
                startTimeRef.current = Date.now();

                setState({
                    isLoading: true,
                    taskId,
                    progress: 0,
                    status: "running",
                    error: null,
                    estimatedRemaining: null,
                });

                // 启动轮询定时器
                pollTimerRef.current = setInterval(() => {
                    pollStatus(taskId);
                }, POLL_INTERVAL_MS);

                // 立即执行一次轮询
                pollStatus(taskId);
            } catch (error) {
                console.error("[useAsyncPitchRefresh] Start error:", error);
                setState((prev) => ({
                    ...prev,
                    isLoading: false,
                    error: error instanceof Error ? error.message : "Failed to start refresh",
                }));
                latestTaskIdRef.current = null;
            } finally {
                isRefreshingRef.current = false;
            }
        },
        [pollStatus],
    );

    // Task 3.5: 实现 cancelRefresh() 函数
    const cancelRefresh = useCallback(async () => {
        const taskId = latestTaskIdRef.current || state.taskId;
        if (!taskId) {
            console.warn("[useAsyncPitchRefresh] No active task to cancel");
            return;
        }

        try {
            await coreApi.cancelPitchTask(taskId);

            setState((prev) => ({
                ...prev,
                isLoading: false,
                status: "cancelled",
                taskId: null,
            }));

            latestTaskIdRef.current = null;

            // 停止轮询
            if (pollTimerRef.current) {
                clearInterval(pollTimerRef.current);
                pollTimerRef.current = null;
            }

            startTimeRef.current = null;
        } catch (error) {
            console.error("[useAsyncPitchRefresh] Cancel error:", error);
            setState((prev) => ({
                ...prev,
                error: error instanceof Error ? error.message : "Failed to cancel task",
            }));
        }
    }, [state.taskId]);

    const reset = useCallback(() => {
        // 清理可能残留的轮询和防抖定时器
        if (pollTimerRef.current) {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
        }
        if (debounceTimerRef.current) {
            clearTimeout(debounceTimerRef.current);
            debounceTimerRef.current = null;
        }

        setState({
            isLoading: false,
            taskId: null,
            progress: 0,
            status: null,
            error: null,
            estimatedRemaining: null,
        });
        latestTaskIdRef.current = null;
        startTimeRef.current = null;
    }, []);

    // Task 7.2: 清理函数，组件卸载时取消活动任务
    useEffect(() => {
        return () => {
            if (pollTimerRef.current) {
                clearInterval(pollTimerRef.current);
            }
            if (debounceTimerRef.current) {
                clearTimeout(debounceTimerRef.current);
            }
            // 组件卸载时尝试取消任务
            if (latestTaskIdRef.current) {
                coreApi.cancelPitchTask(latestTaskIdRef.current).catch((err) => {
                    console.warn("[useAsyncPitchRefresh] Cleanup cancel failed:", err);
                });
            }
        };
    }, []);

    return {
        ...state,
        startRefresh,
        cancelRefresh,
        reset,
    };
}
