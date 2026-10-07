// 原GUI的运行模式与能力边界；独立app保持原窗口/文件/设备行为，插件由DAW供源。
import { translateOutsideReact } from "../i18n/I18nProvider";

export type HostMode = "standalone" | "plugin";
/** 必须使用显式bootstrap，而不是把所有WebView2都误认为插件。 */
export function hostMode(): HostMode {
    return typeof window !== "undefined" && window.__HFS_PLUGIN_BOOTSTRAP__?.version === 1
        ? "plugin" : "standalone";
}
export function isPluginMode(): boolean { return hostMode() === "plugin"; }
/** 仅宿主原生明确提供ARA播放请求能力时开放，不把插件模式等同支持播放控制。 */
export function canControlHostTransport(): boolean {return isPluginMode() && window.__HFS_PLUGIN_BOOTSTRAP__?.transportControl===true;}
/** 只有本次原生入口明确具备REAPER写API，才打开片段拖拽/裁切/线性拉伸。 */
export function canEditHostClips(): boolean {return isPluginMode() && window.__HFS_PLUGIN_BOOTSTRAP__?.clipEditing===true;}
/** 分割有独立原生能力门，不因允许拖拽就放开未实现的片段命令。 */
export function canSplitHostClips(): boolean {return isPluginMode() && window.__HFS_PLUGIN_BOOTSTRAP__?.clipSplitting===true;}
/** 原生完整item剪贴板及创建/删除接口已接通时开放，旧插件不会误发App私有媒体命令。 */
export function canClipboardHostClips():boolean {return isPluginMode()&&window.__HFS_PLUGIN_BOOTSTRAP__?.clipClipboard===true;}
/** 只允许插件私有参数分组，不代表能重排或创建REAPER的轨道/folder。 */
export function canGroupPluginTracks():boolean {return isPluginMode()&&window.__HFS_PLUGIN_BOOTSTRAP__?.trackGrouping===true;}
/** 独立App保留原几何编辑；旧插件或其它宿主缺写接口时保持只读。 */
export function isHostGeometryReadOnly(): boolean {return isPluginMode()&&!canEditHostClips();}
/** 文件菜单只开放明确具备宿主媒体创建能力的音频导入，不放开项目文件/设备命令。 */
export function canImportHostAudio():boolean {return isPluginMode()&&window.__HFS_PLUGIN_BOOTSTRAP__?.audioImport===true;}

/**
 * 宿主接管范围的说明文案（被禁用项的 tooltip、以及"该窗口在插件里不可用"的异常）。
 *
 * 【为什么是函数而不是常量】常量在模块加载期取值，语言就冻在那一刻的
 * localStorage 上 —— 用户切换语言后这串文案不会跟着变。调用点分散在菜单、
 * 右键菜单与命令式异常里（都不是组件），因此走 `translateOutsideReact`：
 * 它在无浏览器环境下回落英文词典，不会因为取一条文案而抛错。
 */
export function dawControlledReason(): string {
    return translateOutsideReact("plugin_daw_controlled_reason");
}

/** 原编辑工具与查看操作保留；不将DAW几何操作发到独立app命令路径。 */
export function pluginAllowsAction(action: string, surface: string | null): boolean {
    if (!isPluginMode()) return true;
    if (["playback.toggle","playback.stop"].includes(action)) return canControlHostTransport();
    if (action==="project.importMedia") return canImportHostAudio();
    if (action==="clip.split") return canSplitHostClips();
    if (action==="clip.delete"||action==="edit.pasteTracks") return canClipboardHostClips();
    // 归一化只改HFS音频处理参数，不改REAPER item几何，可在插件中继续使用。
    if (action==="clip.normalize") return true;
    if (["clip.group","clip.ungroup"].includes(action)) return canEditHostClips();
    if (action.startsWith("project.") || action.startsWith("transport.") || action.startsWith("recording.")
        || ["playback.toggle", "playback.stop", "playback.metronome"].includes(action)) return false;
    if (action.startsWith("track.") && !["track.selectUp", "track.selectDown", "track.toggleMute", "track.toggleSolo"].includes(action)) return false;
    // 共享剪贴板键的别名不决定目标；换轨后clip.paste也可能粘贴参数线。
    // 必须先解析内容/选区，再按实际事件通道拒绝宿主几何操作。
    // 共享快捷键先按剪贴板内容路由；参数剪贴板在没有clipClipboard时仍必须可用。
    if (["clip.copy", "clip.cut", "clip.paste"].includes(action)) return true;
    if (action.startsWith("clip.") && surface !== "pianoRoll") return false;
    if (action.startsWith("edit.") && surface !== "pianoRoll") {
        return ["edit.undo", "edit.redo", "edit.selectAll", "edit.deselect", "edit.addClipsToParamSelection",
            "edit.removeClipsFromParamSelection"].includes(action);
    }
    return true;
}

/** 插件共享编辑快捷键只允许发给参数面板；不向宿主轨道派发几何写操作。 */
export function pluginAllowsEditChannel(channel: string | null, op?: string): channel is string {
    return channel !== null && (!isPluginMode() || channel === "hifi:editOp"
        || (channel === "hifi:timelineEditOp" && op === "split" && canSplitHostClips())
        || (channel === "hifi:timelineEditOp" && ["copy","cut","paste","pasteTracks","delete"].includes(op??"") && canClipboardHostClips())
        || (channel === "hifi:timelineEditOp" && ["group","ungroup"].includes(op??"") && canEditHostClips()));
}
