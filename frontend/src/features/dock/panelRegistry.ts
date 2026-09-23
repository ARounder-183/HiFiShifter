/*
 * 面板注册表 —— 停靠系统的唯一扩展点。
 *
 * 【为什么内置面板也走注册表】如果时间轴/记事本这些面板在停靠内核里被硬编码，
 * 那么"未来开放给用户自定义面板"就必然要重写内核。让内置面板与将来的第三方
 * 面板走完全相同的注册路径，抽象就不会被架空 —— 这是本设计对 API 化最大的
 * 一笔投资。
 *
 * 【注册表为什么是模块级 Map 而不是 Redux】面板定义里含 React 组件与函数
 * （生命周期钩子），放进 Redux 会违反可序列化约定（`app/store.ts` 已为 session
 * 开豁免，不该再开第二个）。注册发生在模块加载期，早于任何渲染，没有竞态。
 */

import type { ComponentType } from "react";

import type { DockPlacement } from "./dockTypes";

/** 面板组件收到的 props。 */
export interface DockPanelProps {
    /** 本窗体实例 id（多实例面板据此区分自己）。 */
    formId: string;
    /** 本窗体所属面板 id。 */
    panelId: string;
    /** 面板私有状态（持久化在布局里，未来 API 面板可直接使用）。 */
    props: Record<string, unknown>;
}

/**
 * 面板在 DOM 搬家前后的状态保全钩子。
 *
 * 【为什么必须有】停靠重排是通过"把已渲染好的 DOM 宿主搬到另一个容器"实现的
 * （见 `DockPanelHost`），React 组件不会卸载，所以绝大多数状态天然幸存。但
 * 浏览器在 DOM 移动时**可能**重置滚动位置，且面板内部的滚动/缩放若由命令式
 * 代码持有，就需要显式保存与恢复。这两个钩子是面板唯一需要为停靠付出的成本。
 */
export interface DockPanelLifecycle {
    /** 搬家前保存状态（返回的对象会原样交给 `afterMove`）。 */
    beforeMove?: (element: HTMLElement) => Record<string, unknown>;
    /** 搬家后恢复状态。 */
    afterMove?: (element: HTMLElement, saved: Record<string, unknown>) => void;
}

/** 面板定义。 */
export interface PanelDefinition {
    /** 唯一 id，同时是持久化 JSON 里的键 —— 一旦发布不可更改。 */
    id: string;
    /** 标题的 i18n key。 */
    titleKey: string;
    /** 标签条上的图标。 */
    icon?: ComponentType;
    component: ComponentType<DockPanelProps>;
    /** 浮动时的默认尺寸。 */
    defaultWidth: number;
    defaultHeight: number;
    minWidth?: number;
    minHeight?: number;
    /** 是否允许进入主编辑区（时间轴/参数编辑器这类"主工作区"面板）。 */
    preferMain?: boolean;
    /**
     * 是否单例（默认 true）。
     *
     * 时间轴这类"全工程唯一"的面板开两个毫无意义且会让用户困惑；参数编辑器
     * 这类"看不同片段"的面板则天然支持多实例。
     */
    singleton?: boolean;
    /** 首次打开时的落点（缺省并入主编辑区标签组）。 */
    defaultPlacement?: DockPlacement;
    /** 搬家前后的状态保全钩子（见 `DockPanelLifecycle`）。 */
    lifecycle?: DockPanelLifecycle;
    /** "显示窗体"菜单里的排序权重。 */
    order?: number;
}

const registry = new Map<string, PanelDefinition>();
const listeners = new Set<() => void>();

/** 注册一个面板。重复 id 直接覆盖并告警（热更新时会重放注册）。 */
export function registerPanel(definition: PanelDefinition): void {
    if (registry.has(definition.id)) {
        console.warn(`[dock] panel "${definition.id}" registered twice; replacing`);
    }
    registry.set(definition.id, definition);
    for (const listener of listeners) listener();
}

export function getPanel(panelId: string): PanelDefinition | undefined {
    return registry.get(panelId);
}

/** 全部已注册面板，按 `order` 再按 id 排序（顺序稳定，便于菜单展示）。 */
export function listPanels(): PanelDefinition[] {
    return [...registry.values()].sort((a, b) => {
        const orderDiff = (a.order ?? 100) - (b.order ?? 100);
        return orderDiff !== 0 ? orderDiff : a.id.localeCompare(b.id);
    });
}

export function isPanelRegistered(panelId: string): boolean {
    return registry.has(panelId);
}

/** 订阅注册表变化（布局规范化需要在面板注册后重跑一次）。 */
export function subscribePanels(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

/** 仅测试用：清空注册表。 */
export function resetPanelRegistryForTests(): void {
    registry.clear();
    listeners.clear();
}
