/**
 * 状态栏"渲染中"片的唯一读数来源。
 *
 * 【为什么需要这个 hook】同一片状态在两个形态下有两个权威：
 *
 * - 独立 App：Redux 的 `playbackRenderingActive` + `appStatusProgressBus`；
 * - 插件：`plugin_get_apply_state` 的 250 ms 快照（`rendering_active` /
 *   `render_progress` / `render_audible_stale`）。
 *
 * 把"读哪一份"收在一处，状态片组件就不必同时知道两条通道；两条通道的语义
 * （stale 决定左侧片是否让位给右侧的「正在合成」，不改变状态片的强调色）也只需要在这里解释一次。
 */
import { useSyncExternalStore } from "react";

import { useAppSelector } from "../../app/hooks";
import { isPluginMode } from "../../services/hostCapabilities";
import { appStatusProgressBus } from "../../utils/appStatusProgressBus";
import { usePluginApplySnapshot } from "./pluginApplyStore";

export interface RenderStatus {
    /** 后台是否真的有一个渲染任务。 */
    active: boolean;
    /** 进度（0..1）；`null` = 没有可报告的进度。 */
    progress: number | null;
    /** 当前听到的是不是旧版本（只在插件模式有意义）。 */
    stale: boolean;
    /** 本实例是否运行在插件宿主里。 */
    pluginMode: boolean;
}

export function useRenderStatus(): RenderStatus {
    const pluginMode = isPluginMode();
    const plugin = usePluginApplySnapshot();
    const renderingActive = useAppSelector((state) => state.session.playbackRenderingActive);
    const status = useSyncExternalStore(
        appStatusProgressBus.subscribe,
        appStatusProgressBus.getSnapshot,
        appStatusProgressBus.getSnapshot,
    );
    if (pluginMode) {
        const state = plugin.state;
        const active = state?.rendering_active === true;
        const raw = state?.render_progress;
        const progress =
            active && typeof raw === "number" && Number.isFinite(raw)
                ? Math.max(0, Math.min(1, raw))
                : null;
        return {
            active,
            progress,
            stale: state?.render_audible_stale === true,
            pluginMode: true,
        };
    }
    return {
        active: renderingActive,
        progress: status.renderingProgress,
        stale: false,
        pluginMode: false,
    };
}
