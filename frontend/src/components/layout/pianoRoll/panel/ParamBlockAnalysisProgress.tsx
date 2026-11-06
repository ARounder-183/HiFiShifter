/**
 * 参数编辑器左下角的音符块分析条。
 *
 * 只订阅独立的音节分析进度，不触发整个面板重渲染；块显示/编辑未开启时
 * 连轮询都不做（GAME 分块是懒加载的，不开启块层就不该付出模型推理成本）。
 */
import { useEffect, useState } from "react";

import { useI18n } from "../../../../i18n/I18nProvider";
import { coreApi } from "../../../../services/api";
import type { SyllableAnalysisProgressPayload } from "../../../../types/api";

export function ParamBlockAnalysisProgress(props: {
    readonly enabled: boolean;
    readonly visible: boolean;
}) {
    const { tf } = useI18n();
    const [progress, setProgress] = useState<SyllableAnalysisProgressPayload | null>(null);

    useEffect(() => {
        if (!props.enabled) {
            // eslint-disable-next-line react-hooks/set-state-in-effect -- 组件没有卸载，关闭块层时同步收起旧进度，避免下次开启先闪旧值。
            setProgress(null);
            return;
        }
        let disposed = false;
        let timer: number | null = null;
        const poll = async () => {
            try {
                const next = await coreApi.getSyllableAnalysisProgress();
                if (!disposed) setProgress(next);
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

    if (!props.visible || !props.enabled || progress === null || !progress.active) return null;
    const percent = Math.max(0, Math.min(100, Math.round(progress.progress * 100)));
    const algorithm = progress.algorithm.startsWith("game")
        ? "GAME"
        : tf("param_blocks_vad_fallback");
    return (
        <div
            className="pointer-events-none absolute bottom-2 left-2 z-30 w-[min(280px,calc(100%-1rem))] rounded border border-qt-border bg-qt-panel/95 px-2 py-1.5 shadow-lg"
            data-param-block-analysis-progress
        >
            <div className="flex items-center justify-between gap-2 hs-type-caption">
                <span className="truncate text-qt-text">
                    {tf("status_analyzing_blocks")}
                    {progress.clipName ? ` "${progress.clipName}"` : ""}
                </span>
                <span className="shrink-0 text-qt-text-muted">{percent}%</span>
            </div>
            <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-qt-meter-well">
                <div
                    className="h-full bg-qt-accent transition-[width] duration-200"
                    style={{ width: `${percent}%` }}
                />
            </div>
            <div className="mt-0.5 truncate hs-type-caption text-qt-text-muted">
                {algorithm}
                {progress.message ? ` · ${progress.message}` : ""}
            </div>
        </div>
    );
}
