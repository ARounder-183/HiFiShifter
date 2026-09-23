/*
 * 拖拽期间的瞬时状态 —— 刻意**不放进 Redux**。
 *
 * 指针位置在拖拽中每秒变化上百次。放进 Redux 意味着每次 pointermove 都跑一遍
 * 全部订阅者：本应用有一个 33Hz 的播放轮询订阅着 store，菜单栏又按
 * `shallowEqual` 订阅 session —— 让这些与拖拽无关的东西跟着重渲染，正是本
 * 仓库反复踩过的坑（见 `keybindingsSlice` 里对手工 memo 的注释）。
 *
 * 因此这里是一个极小的外部 store + `useSyncExternalStore`：只有真正需要知道
 * 拖拽状态的组件（落点覆盖层、被拖的标签）订阅它，其余组件完全无感。
 */

import type { DockDropZone, DockRect } from "./dockTypes";

export type DockDragMode = "tab" | "float";

export interface DockDropTargetState {
    zoneId: string;
    zone: DockDropZone;
    rect: DockRect;
}

export interface DockDragState {
    mode: DockDragMode;
    formId: string;
    panelId: string;
    /** 指针的视口坐标。 */
    pointerX: number;
    pointerY: number;
    /** 是否已越过启动阈值（未越过时什么都不显示，避免"点一下就抖出幽灵"）。 */
    started: boolean;
    /** 当前是否按住停靠修饰键。 */
    dockIntent: boolean;
    /** 已解析的落点；`null` 表示不落任何 Zone（将浮动）。 */
    target: DockDropTargetState | null;
    /** 被拖浮窗的实时几何（`mode === "float"` 时非空）。 */
    floatRect: DockRect | null;
}

let state: DockDragState | null = null;
const listeners = new Set<() => void>();

function emit(): void {
    for (const listener of listeners) listener();
}

export function subscribeDockDrag(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

export function getDockDragState(): DockDragState | null {
    return state;
}

export function beginDockDrag(next: Omit<DockDragState, "started" | "target">): void {
    state = { ...next, started: false, target: null };
    emit();
}

export function updateDockDrag(patch: Partial<DockDragState>): void {
    if (!state) return;
    state = { ...state, ...patch };
    emit();
}

export function endDockDrag(): void {
    if (!state) return;
    state = null;
    emit();
}

/** 仅测试用。 */
export function resetDockDragForTests(): void {
    state = null;
    listeners.clear();
}
