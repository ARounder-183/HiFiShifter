// 原GUI统一事件口：插件使用实例内消息，独立app保留Tauri事件实现。
import { getPluginHost, type HostEvent } from "./pluginHost";

/** 订阅当前宿主事件；插件不会触碰Tauri内部回调与全局事件总线。 */
export async function listen<T>(event: string, handler: (event: HostEvent<T>) => void): Promise<() => void> {
    const host = getPluginHost();
    if (host) return host.listen(event, handler);
    const tauri = await import("@tauri-apps/api/event");
    return tauri.listen<T>(event, handler);
}

/** 插件没有独立卫星窗口，只在本实例发送显式UI事件；独立app广播方式不变。 */
export async function emit(event: string, payload?: unknown): Promise<void> {
    const host = getPluginHost();
    if (host) return host.invoke<void>("emit_ui_event", { event, payload });
    const tauri = await import("@tauri-apps/api/event");
    return tauri.emit(event, payload);
}
