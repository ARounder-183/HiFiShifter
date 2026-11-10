/**
 * 参数块菜单上下文的跨组件总线。
 *
 * 参数编辑器可以多开；菜单栏在面板之外，不能直接读某个面板的 ref。
 * 旧实现只有一个全局 current：最后一个发布的面板覆盖前一个，任一面板
 * 卸载时 publish(null) 又会清掉别的面板。这里改为按 panelId 注册，并
 * 只让 active panel 的上下文出现在菜单栏。
 */
import type { ParamBlockMenuContext } from "./paramBlockMenuModel";

const contexts = new Map<string, ParamBlockMenuContext>();
const publishOrder = new Map<string, number>();
let activePanelId: string | null = null;
let nextOrder = 0;
const listeners = new Set<() => void>();

function notify(): void {
    for (const listener of listeners) listener();
}

function fallbackActivePanel(): string | null {
    let best: string | null = null;
    let bestOrder = -1;
    for (const [panelId, order] of publishOrder) {
        if (order > bestOrder) {
            best = panelId;
            bestOrder = order;
        }
    }
    return best;
}

/** 发布某个面板的上下文；null 只注销该面板，不影响其他面板。 */
export function publishParamBlockMenuContext(
    panelId: string,
    next: ParamBlockMenuContext | null,
): void {
    if (next === null) {
        contexts.delete(panelId);
        publishOrder.delete(panelId);
        if (activePanelId === panelId) activePanelId = fallbackActivePanel();
    } else {
        contexts.set(panelId, next);
        publishOrder.set(panelId, ++nextOrder);
        if (activePanelId === null || !contexts.has(activePanelId)) activePanelId = panelId;
    }
    notify();
}

/** 指针进入 / 聚焦某个面板时把它设为菜单动作目标。 */
export function activateParamBlockMenuContext(panelId: string): void {
    if (!contexts.has(panelId) || activePanelId === panelId) return;
    activePanelId = panelId;
    notify();
}

export function getParamBlockMenuContext(): ParamBlockMenuContext | null {
    return activePanelId === null ? null : (contexts.get(activePanelId) ?? null);
}

export function getActiveParamBlockMenuPanelId(): string | null {
    return activePanelId;
}

export function subscribeParamBlockMenuContext(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

/** 仅测试使用：清空注册表，避免用例间串扰。 */
export function resetParamBlockMenuBusForTests(): void {
    contexts.clear();
    publishOrder.clear();
    activePanelId = null;
    nextOrder = 0;
    notify();
}
