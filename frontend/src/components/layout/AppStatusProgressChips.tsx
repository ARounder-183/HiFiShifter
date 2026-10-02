/*
 * 状态栏的高频进度指示片（变长拉伸 / 波形分析 / 播放渲染进度）。
 *
 * 【为什么单独成一个组件】这三路进度只服务状态栏显示，却曾以 React state 住在
 * `AppInner` 里 —— 每个 progress 事件都会重渲染整棵应用树（时间轴、参数编辑器、
 * 全部面板）。抽成独立组件并直接订阅外部 store（`appStatusProgressBus`）后，
 * 重渲染被限制在这几个 `<span>` 上。与 `ParamDataLoadingChip` 同一模式。
 *
 * 【为什么渲染 active 单独走 Redux】激活/目标/阻塞三个布尔镜像低频变化、
 * 且被播放轮询守卫消费，留在 Redux；本组件只订阅其中的 active 布尔
 * （播放中每 ~33ms 变的是 playheadSec，不是这三个布尔），进度百分比走总线。
 */

import { useSyncExternalStore } from "react";

import { useI18n } from "../../i18n/I18nProvider";
import { useAppSelector } from "../../app/hooks";
import { AppStatusChip } from "../../ui";
import { appStatusProgressBus } from "../../utils/appStatusProgressBus";

export function AppStatusProgressChips() {
    const { t } = useI18n();
    const status = useSyncExternalStore(
        appStatusProgressBus.subscribe,
        appStatusProgressBus.getSnapshot,
        appStatusProgressBus.getSnapshot,
    );
    const renderingActive = useAppSelector((state) => state.session.playbackRenderingActive);

    return (
        <>
            {status.stretching.active ? (
                <AppStatusChip tone="accent">
                    {t("status_stretching")}
                    {status.stretching.clipName ? ` "${status.stretching.clipName}"` : ""}
                </AppStatusChip>
            ) : null}
            {status.waveformAnalysis.active ? (
                <AppStatusChip tone="accent">
                    {t("status_analyzing_waveform")}
                    {status.waveformAnalysis.sourcePath
                        ? ` "${status.waveformAnalysis.sourcePath}"`
                        : ""}
                    {status.waveformAnalysis.progress != null
                        ? ` ${Math.round(status.waveformAnalysis.progress * 100)}%`
                        : ""}
                </AppStatusChip>
            ) : null}
            {renderingActive ? (
                <AppStatusChip tone="accent">
                    {t("common_rendering")}
                    {status.renderingProgress != null
                        ? ` ${Math.round(status.renderingProgress * 100)}%`
                        : ""}
                </AppStatusChip>
            ) : null}
        </>
    );
}
