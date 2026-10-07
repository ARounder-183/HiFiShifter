// 独立窗口能力的惰性出口；插件窗口由IPlugView拥有，不能初始化Tauri窗口API。
import { isPluginMode, DAW_CONTROLLED_REASON } from "./hostCapabilities";
/** 独立app保留原API；插件在加载Tauri模块前明确拒绝。 */
export async function loadStandaloneWindowApi(): Promise<typeof import("@tauri-apps/api/window")> {
    if (isPluginMode()) throw new Error(DAW_CONTROLLED_REASON);
    return import("@tauri-apps/api/window");
}
