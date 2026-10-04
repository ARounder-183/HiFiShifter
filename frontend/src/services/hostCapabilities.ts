// 原GUI的运行模式与能力边界；独立app保持原窗口/文件/设备行为，插件由DAW供源。
export type HostMode = "standalone" | "plugin";
/** 必须使用显式bootstrap，而不是把所有WebView2都误认为插件。 */
export function hostMode(): HostMode {
    return typeof window !== "undefined" && window.__HFS_PLUGIN_BOOTSTRAP__?.version === 1
        ? "plugin" : "standalone";
}
export function isPluginMode(): boolean { return hostMode() === "plugin"; }
export const DAW_CONTROLLED_REASON = "由 REAPER 控制；在宿主中操作文件、片段几何与播放";

/** 原编辑工具与查看操作保留；不将DAW几何操作发到独立app命令路径。 */
export function pluginAllowsAction(action: string, surface: string | null): boolean {
    if (!isPluginMode()) return true;
    if (action.startsWith("project.") || action.startsWith("transport.") || action.startsWith("recording.")
        || ["playback.toggle", "playback.stop", "playback.metronome"].includes(action)) return false;
    if (action.startsWith("track.") && !["track.selectUp", "track.selectDown", "track.toggleMute", "track.toggleSolo"].includes(action)) return false;
    if (action.startsWith("clip.") && surface !== "pianoRoll") return false;
    if (action.startsWith("edit.") && surface !== "pianoRoll") {
        return ["edit.undo", "edit.redo", "edit.selectAll", "edit.deselect", "edit.addClipsToParamSelection",
            "edit.removeClipsFromParamSelection"].includes(action);
    }
    return true;
}
