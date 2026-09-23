/*
 * 面板渲染函数注册表 —— 让"面板定义"与"面板 props 的来源"解耦。
 *
 * 【要解决的问题】时间轴面板需要约三十个 props（MIDI 导入对话框的一整套状态、
 * 各种回调），这些状态都活在 `App.tsx` 里。若把它们搬进注册表定义，注册表就
 * 变成了 App 状态的第二份拷贝；若改成 Context 透传，就要维护一个依赖列表
 * 巨长、极易失效的 memo。
 *
 * 【做法】注册表只管**元信息**（标题、默认尺寸、落点），真正的渲染交给这里
 * 注册的闭包 —— 由 `App.tsx` 在每次渲染时写入，闭包自然捕获到最新的 props。
 * 于是 App 仍是这些状态的唯一所有者，停靠系统只负责"把它摆在哪儿"。
 *
 * 【为什么写入不需要订阅通知】写入发生在 App 的渲染过程中，而读取发生在
 * 同一趟渲染里更靠后的子组件（`DockPanelHosts`）。React 保证父组件先渲染，
 * 所以读到的一定是本趟的新值。加订阅反而会在渲染期触发额外渲染。
 */

import type { ReactNode } from "react";

export type PanelRenderFunction = () => ReactNode;

const renderers = new Map<string, PanelRenderFunction>();

/**
 * 登记（或更新）面板的渲染函数。
 *
 * 在渲染期调用是安全的：写入是幂等的，且没有订阅者会在写入时被唤醒。
 */
export function setPanelRenderer(panelId: string, render: PanelRenderFunction): void {
    renderers.set(panelId, render);
}

export function getPanelRenderer(panelId: string): PanelRenderFunction | undefined {
    return renderers.get(panelId);
}

export function clearPanelRenderer(panelId: string): void {
    renderers.delete(panelId);
}

/** 仅测试用。 */
export function resetPanelRenderersForTests(): void {
    renderers.clear();
}
