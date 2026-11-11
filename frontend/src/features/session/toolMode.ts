/**
 * 工具身份与命令作用域的纯函数。
 *
 * 【为什么是纯函数】工具状态有两根轴：工具族（`toolMode`）与选择族子模式
 * （`selectToolMode`）。块选择不是第三种工具族，而是二者派生出来的布尔值；
 * 快捷键路由需要同一个派生结果（`ToolScope`），但作用域本身不进 Redux。
 */
import type { DrawToolMode, SelectToolMode, ToolMode } from "./sessionTypes";

/**
 * 命令路由使用的作用域。
 *
 * `block` 是“选择族 + 块子模式”的派生 scope，不是可持久化的 ToolMode。
 */
export type ToolScope = "select" | "block" | DrawToolMode;

/** 参数块选择当前是否激活。 */
export function isBlockSelectionActive(
    toolMode: ToolMode,
    selectToolMode: SelectToolMode,
): boolean {
    return toolMode === "select" && selectToolMode === "block";
}

/** 把工具族 + 选择子模式折叠成快捷键路由使用的唯一作用域。 */
export function resolveToolScope(toolMode: ToolMode, selectToolMode: SelectToolMode): ToolScope {
    return isBlockSelectionActive(toolMode, selectToolMode) ? "block" : toolMode;
}

/** 选择族内切换 range ↔ block。 */
export function toggleSelectToolMode(mode: SelectToolMode): SelectToolMode {
    return mode === "block" ? "range" : "block";
}
