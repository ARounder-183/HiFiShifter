/*
 * 自动备份调度 Hook。
 *
 * 调度规则：
 * - 定时备份启用后，按间隔检查是否有新编辑。
 * - 若超过间隔但没有编辑，进入“等待编辑”状态（暂停自动触发）。
 * - 等待期间一旦检测到新编辑，立即执行一次备份并恢复周期检查。
 * - 在“新建工程 / 打开工程”后，重置计时基准。
 */

import { useEffect, useRef } from "react";
import { projectApi, type AutoBackupSettings } from "../services/api/project";

interface UseAutoBackupSchedulerParams {
    settings: AutoBackupSettings;
    paramsEpoch: number;
    projectDirty: boolean;
    status: string;
}

export function useAutoBackupScheduler({
    settings,
    paramsEpoch,
    projectDirty,
    status,
}: UseAutoBackupSchedulerParams) {
    const settingsRef = useRef(settings);
    const paramsEpochRef = useRef(paramsEpoch);
    const projectDirtyRef = useRef(projectDirty);
    const baselineRef = useRef<{ timestampMs: number; paramsEpoch: number }>({
        timestampMs: Date.now(),
        paramsEpoch,
    });
    const waitingForEditRef = useRef(false);
    const inFlightRef = useRef(false);
    const lastStatusRef = useRef(status);

    useEffect(() => {
        settingsRef.current = settings;
    }, [settings]);

    useEffect(() => {
        paramsEpochRef.current = paramsEpoch;
    }, [paramsEpoch]);

    useEffect(() => {
        projectDirtyRef.current = projectDirty;
    }, [projectDirty]);

    function resetBaseline() {
        baselineRef.current = {
            timestampMs: Date.now(),
            paramsEpoch: paramsEpochRef.current,
        };
        waitingForEditRef.current = false;
    }

    async function runTimedBackupNow() {
        if (inFlightRef.current) return;
        inFlightRef.current = true;

        try {
            await projectApi.runTimedAutoBackup(settingsRef.current.timedBackupPathTemplate);
        } catch {
            // 备份失败时仅静默跳过本轮，避免打断用户编辑流程。
        } finally {
            inFlightRef.current = false;
            resetBaseline();
        }
    }

    useEffect(() => {
        if (!settings.timedBackupEnabled) {
            waitingForEditRef.current = false;
            return;
        }
        resetBaseline();
    }, [
        settings.timedBackupEnabled,
        settings.timedBackupIntervalSec,
        settings.timedBackupPathTemplate,
    ]);

    useEffect(() => {
        const prevStatus = lastStatusRef.current;
        if (status === prevStatus) return;
        lastStatusRef.current = status;

        if (!settingsRef.current.timedBackupEnabled) return;

        if (status === "New project" || status === "Project opened") {
            resetBaseline();
        }
    }, [status]);

    useEffect(() => {
        if (!settings.timedBackupEnabled) return;

        const timerId = window.setInterval(() => {
            if (inFlightRef.current) return;
            if (waitingForEditRef.current) return;

            const intervalMs = Math.max(
                1000,
                Math.floor(settingsRef.current.timedBackupIntervalSec * 1000),
            );
            const elapsedMs = Date.now() - baselineRef.current.timestampMs;
            if (elapsedMs < intervalMs) return;

            const hasEditSinceBaseline =
                projectDirtyRef.current && paramsEpochRef.current > baselineRef.current.paramsEpoch;
            if (!hasEditSinceBaseline) {
                waitingForEditRef.current = true;
                return;
            }

            void runTimedBackupNow();
        }, 1000);

        return () => window.clearInterval(timerId);
        // eslint-disable-next-line react-hooks/exhaustive-deps -- runTimedBackupNow 每次渲染重建，加入依赖会改变定时器重启时序（既有模式）
    }, [settings.timedBackupEnabled]);

    useEffect(() => {
        if (!settings.timedBackupEnabled) return;
        if (!waitingForEditRef.current) return;

        const hasEditSinceBaseline = projectDirty && paramsEpoch > baselineRef.current.paramsEpoch;
        if (!hasEditSinceBaseline) return;

        void runTimedBackupNow();
        // eslint-disable-next-line react-hooks/exhaustive-deps -- runTimedBackupNow 每次渲染重建，加入依赖会改变检测重跑时机（既有模式）
    }, [settings.timedBackupEnabled, projectDirty, paramsEpoch]);
}
