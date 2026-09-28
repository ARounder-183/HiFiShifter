/**
 * 贡献点注册中心 —— 扩展可以向宿主 chrome 添加条目。
 *
 * 【为什么需要它】审查时第三方能碰到 chrome 的地方**只有 Window 菜单**
 * （那条路径是数据驱动的：`listPanelEntries` 遍历注册表）。其余全是硬编码：
 * `ActionBar` 直接写死 `PANEL_FILE_BROWSER` / `PANEL_NOTEBOOK` /
 * `PANEL_UNDO_HISTORY`，`DockTabMenu` 是固定项集，菜单栏 187 个手写 item。
 * 也就是说一个第三方面板**注册了却无处可点** —— 用户只能从 Window 菜单里找到它。
 *
 * 本模块提供与 `panelRegistry` **同构**的三个贡献点（Map + 版本号 + 订阅）。
 * 沿用同一套形状是有意的：那套机制已经在面板注册上跑过生产，证明可用，
 * 不必为贡献点另发明一种。
 *
 * 【与面板注册的关键差异：作用域】面板是全局单例，贡献项通常属于某个面板
 * （"我这个面板的工具栏按钮"）或某个上下文（"剪辑右键菜单"）。因此每项带
 * `scope` 字段，宿主按 scope 取用。
 *
 * 【为什么不在这里做权限】本轮不引入权限模型（见方案 §3.5 E9）。但所有贡献都
 * 经由本模块登记，因此将来接权限层只需在这一个文件里加过滤，不必改各个宿主。
 */
import { useSyncExternalStore } from "react";
import type { ReactNode } from "react";

/** 贡献点种类。 */
export type ContributionKind = "toolbarItem" | "panelTabMenuItem" | "command";

/**
 * 作用域：贡献项挂在哪个宿主上。
 *
 * `undefined` 表示全局（所有同类宿主都会渲染）。
 */
export interface ContributionScope {
    /** 面板 id（如 `"fileBrowser"`）；面板工具栏按钮用。 */
    panelId?: string;
    /** 上下文目标的类型（如 `"clip"` / `"track"`）；右键菜单用。 */
    target?: string;
}

export interface ToolbarItemContribution extends ContributionScope {
    id: string;
    /** 排序权重，小的在前；缺省按 id 排序。 */
    order?: number;
    /**
     * 渲染这个工具栏按钮。
     *
     * 【为什么给 render 而不是给 onClick + icon】工具栏按钮通常需要反映状态
     * （"面板是否已打开"要显示激活态），而状态读取是宿主 / Redux 的事。
     * 让贡献方渲染自己的按钮，就不必把 Redux 形状泄漏进本模块。
     * 贡献方可以用 `useAppSelector` 与 `@hs/ui` 的 `AppIconButton`。
     */
    render: () => ReactNode;
}

export interface PanelTabMenuItemContribution extends ContributionScope {
    id: string;
    order?: number;
    /** 菜单项文案（已本地化）。 */
    label: string;
    /** 破坏性动作（红底）。 */
    danger?: boolean;
    /** 返回 `false` 时该项置灰。 */
    enabled?: () => boolean;
    onSelect: () => void;
}

export interface CommandContribution {
    id: string;
    /** 命令名（已本地化），用于命令面板与快捷键绑定界面。 */
    label: string;
    /** 分组名（已本地化），决定在绑定界面里归到哪一组。 */
    group?: string;
    onRun: () => void;
}

interface Registry<T> {
    items: Map<string, T>;
}

const registries: Record<ContributionKind, Registry<unknown>> = {
    toolbarItem: { items: new Map() },
    panelTabMenuItem: { items: new Map() },
    command: { items: new Map() },
};

const listeners = new Set<() => void>();
let version = 0;

function notify(): void {
    version += 1;
    for (const listener of listeners) listener();
}

/**
 * 注册一个贡献项。
 *
 * @returns 注销函数。**扩展卸载时必须调用**，否则它的按钮会留在宿主界面上
 *   指向一个已经不存在的面板。
 */
function register<T extends { id: string }>(kind: ContributionKind, item: T): () => void {
    const registry = registries[kind];
    if (registry.items.has(item.id)) {
        console.warn(`[contrib] ${kind} "${item.id}" registered twice; replacing`);
    }
    registry.items.set(item.id, item);
    notify();

    let disposed = false;
    return () => {
        if (disposed) return;
        disposed = true;
        // 只删自己那一条：期间可能已被同 id 的新注册替换，此时不应误删。
        if (registry.items.get(item.id) === item) registry.items.delete(item.id);
        notify();
    };
}

export function registerToolbarItem(item: ToolbarItemContribution): () => void {
    return register("toolbarItem", item);
}

export function registerPanelTabMenuItem(item: PanelTabMenuItemContribution): () => void {
    return register("panelTabMenuItem", item);
}

export function registerCommand(item: CommandContribution): () => void {
    return register("command", item);
}

/** 按作用域筛选（`undefined` 的作用域视为全局，所有宿主都收）。 */
function matchesScope(item: ContributionScope, scope: ContributionScope): boolean {
    if (item.panelId !== undefined && item.panelId !== scope.panelId) return false;
    if (item.target !== undefined && item.target !== scope.target) return false;
    return true;
}

function listFor<T extends ContributionScope & { id: string; order?: number }>(
    kind: ContributionKind,
    scope: ContributionScope,
): T[] {
    return [...(registries[kind].items.values() as IterableIterator<T>)]
        .filter((item) => matchesScope(item, scope))
        .sort((a, b) => {
            const diff = (a.order ?? 100) - (b.order ?? 100);
            return diff !== 0 ? diff : a.id.localeCompare(b.id);
        });
}

export function listToolbarItems(scope: ContributionScope = {}): ToolbarItemContribution[] {
    return listFor<ToolbarItemContribution>("toolbarItem", scope);
}

export function listPanelTabMenuItems(
    scope: ContributionScope = {},
): PanelTabMenuItemContribution[] {
    return listFor<PanelTabMenuItemContribution>("panelTabMenuItem", scope);
}

export function listCommands(): CommandContribution[] {
    return listFor<CommandContribution>("command", {});
}

/** 订阅注册变化（宿主据此重渲染）。 */
export function subscribeContributions(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

/** 快照：贡献表版本号。 */
export function getContributionVersion(): number {
    return version;
}

/** 仅供测试：清空所有贡献点。 */
export function resetContributionsForTests(): void {
    for (const registry of Object.values(registries)) registry.items.clear();
    notify();
}

/**
 * React 订阅钩子。
 *
 * 用 `useSyncExternalStore` 而不是把版本号放进 `useState`：注册可能发生在
 * 渲染期之外（扩展加载、面板挂载），外部 store 语义正是这个场景。
 */
export function useToolbarItems(scope: ContributionScope = {}): ToolbarItemContribution[] {
    const version = useSyncExternalStore(
        subscribeContributions,
        getContributionVersion,
        getContributionVersion,
    );
    void version; // 版本号只用于触发重算，取值本身不参与渲染
    return listToolbarItems(scope);
}

export function usePanelTabMenuItems(
    scope: ContributionScope = {},
): PanelTabMenuItemContribution[] {
    const version = useSyncExternalStore(
        subscribeContributions,
        getContributionVersion,
        getContributionVersion,
    );
    void version;
    return listPanelTabMenuItems(scope);
}
