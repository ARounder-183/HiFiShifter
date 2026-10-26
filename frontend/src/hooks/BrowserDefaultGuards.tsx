/*
 * 独立窗口入口用的组件形态：hook 必须在 React 组件里调用。
 *
 * 【为什么单独成文件】react-refresh 规则要求"只导出组件"的文件才能热更新；
 * hook 与组件混在一个文件里会破坏 Fast Refresh（见 eslint 的
 * react-refresh/only-export-components）。主窗口直接用 hook，独立窗口挂这个组件。
 */

import { useBrowserDefaultGuards } from "./useBrowserDefaultGuards";

export function BrowserDefaultGuards() {
    useBrowserDefaultGuards();
    return null;
}
