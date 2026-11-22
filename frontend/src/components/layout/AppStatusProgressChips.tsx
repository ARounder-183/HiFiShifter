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
import { selectFolderImportProgress } from "../../features/session/interactionSlice";
import { useRenderStatus } from "../../features/ara/useRenderStatus";
import { AppStatusChip } from "../../ui";
import { appStatusProgressBus } from "../../utils/appStatusProgressBus";

export function AppStatusProgressChips() {
    const { t } = useI18n();
    const status = useSyncExternalStore(
        appStatusProgressBus.subscribe,
        appStatusProgressBus.getSnapshot,
        appStatusProgressBus.getSnapshot,
    );
    const render = useRenderStatus();
    const folderImport = useAppSelector(selectFolderImportProgress);

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
            {folderImport.active ? (
                <AppStatusChip tone="accent">
                    {t("status_importing_folder")}
                    {folderImport.total > 0 ? ` ${folderImport.done}/${folderImport.total}` : ""}
                </AppStatusChip>
            ) : null}
            {/* 【百分比只在这一处】右侧 PluginApplyStatus 只说「你听到的是不是当前
                编辑」，不再拼百分比（见那里的说明）。同一件事在两处重复会把状态栏
                撑成一条跑道，而两处判据一旦分叉就会出现"左边 40% 右边 100%"。 */}
            {/* 【为什么不再按 `stale` 让位】旧实现在 stale 时把左侧藏起来，指望右侧
                说"正在合成 N%" —— 而右侧那条百分比只在 `audible_state === "stale"`
                时才会出现，那个分支在 ARA 下**恒不成立**（有快照就永远不是 stale）。
                两边都不显示，于是"渲染中"完全没有百分比。现在百分比归左侧，与
                stale 无关。 */}
            {render.active ? (
                <AppStatusChip
                    // 【必须跟随强调色】状态片是「本插件正在工作」的提示，颜色语义与
                    // 拉伸 / 波形分析 / 导入完全一致：用户自定义的强调色。
                    tone="accent"
                    tooltip={render.pluginMode ? t("plugin_render_background_hint") : undefined}
                >
                    {t("common_rendering")}
                    {/* 【0% 是真读数】`null` 才是"没有进度可报"（Capture 没有进度
                        回调；发布阶段也不报）。把 0 也藏起来，等于让"刚开始"与
                        "不知道"长得一模一样 —— 而后者才是异常。 */}
                    {render.progress != null ? ` ${Math.round(render.progress * 100)}%` : ""}
                </AppStatusChip>
            ) : null}
        </>
    );
}
