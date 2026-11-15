/**
 * 底部状态栏里的 GAME / VAD 块分析进度。
 *
 * 进度仍由后端的内存快照提供，轮询只在块显示开启时运行。把它做成状态栏
 * 片而不是参数面板里的浮层，是为了与音高分析、波形分析共用同一处进度
 * 位置；用户不需要在两块区域之间来回找当前任务。
 */
import { useEffect, useState, useSyncExternalStore } from "react";

import { useI18n } from "../../../../i18n/I18nProvider";
import { coreApi } from "../../../../services/api";
import type { SyllableAnalysisProgressPayload } from "../../../../types/api";
import { AppStatusChip } from "../../../../ui";
import { blockAnalysisProgressStore, publishBlockAnalysisProgress } from "../blockAnalysisProgress";

export function ParamBlockAnalysisStatus(props: { readonly enabled: boolean }) {
    const { t } = useI18n();
    const [progress, setProgress] = useState<SyllableAnalysisProgressPayload | null>(null);
    const localProgress = useSyncExternalStore(
        blockAnalysisProgressStore.subscribe,
        blockAnalysisProgressStore.getSnapshot,
        blockAnalysisProgressStore.getSnapshot,
    );

    useEffect(() => {
        if (!props.enabled) {
            setProgress(null);
            return;
        }
        let disposed = false;
        let timer: number | null = null;
        const poll = async () => {
            try {
                const next = await coreApi.getSyllableAnalysisProgress();
                if (!disposed) {
                    setProgress(next);
                    // 后端空闲时清掉前端“请求刚发出”的占位；否则旧占位会
                    // 在分析已结束的状态栏里一直亮着。
                    if (next === null || !next.active) publishBlockAnalysisProgress(null);
                }
            } catch {
                if (!disposed) setProgress(null);
            }
            if (!disposed) timer = window.setTimeout(poll, 250);
        };
        void poll();
        return () => {
            disposed = true;
            if (timer !== null) window.clearTimeout(timer);
        };
    }, [props.enabled]);

    const shown = localProgress?.active ? localProgress : progress;
    if (!props.enabled || shown === null || !shown.active) return null;

    const detail = shown.clipName ? ` "${shown.clipName}"` : "";
    return (
        <AppStatusChip tone="accent">
            {t("status_analyzing_blocks")}
            {detail} {Math.round(shown.progress * 100)}%
        </AppStatusChip>
    );
}
