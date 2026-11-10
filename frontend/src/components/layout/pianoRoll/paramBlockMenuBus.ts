/**
 * 参数块菜单上下文的跨组件总线。
 *
 * 菜单栏在面板之外，不能直接读面板的 ref；但菜单打开时必须看到与右键
 * 菜单同一份实时可用态。面板在选区 / 结构 / 根变化时发布，菜单关闭时
 * 不订阅，因此发布本身不触发任何渲染。
 */
import type { ParamBlockMenuContext } from "./paramBlockMenuModel";

let current: ParamBlockMenuContext | null = null;
const listeners = new Set<() => void>();

export function publishParamBlockMenuContext(next: ParamBlockMenuContext | null): void {
    current = next;
    for (const listener of listeners) listener();
}

export function getParamBlockMenuContext(): ParamBlockMenuContext | null {
    return current;
}

export function subscribeParamBlockMenuContext(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
}
