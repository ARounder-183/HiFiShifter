/**
 * 「点击面板留白时，焦点该不该收回列表」的判定。
 *
 * 【要修的问题】工具条背景、搜索栏留白、路径栏留白这些地方没有可聚焦元素，
 * 浏览器会把焦点丢回 `<body>` —— 于是面板的键盘模型整个失效：keydown 处理器挂在
 * 面板根上，而 `body` 不是它的后代，事件根本不经过它。用户点一下工具条背景再打字，
 * 既不会跳转、也不触发任何面板快捷键。
 *
 * 【为什么抽成纯函数】判定只看"点中的元素是不是自己管焦点的控件"，与 Redux、
 * 面板状态都无关；抽出来就能直接单测，而不必渲染整个面板（面板依赖 store / Tauri /
 * 音频上下文，本仓的键盘测试刻意不渲染它）。
 */

/**
 * 点击这些元素时焦点归它们自己 —— 面板不抢。
 *
 * - `[tabindex]:not([tabindex="-1"])` 兜住其它可 Tab 进入的控件；
 * - `[tabindex="-1"]` 的容器（列表容器自己）**不算** —— 点它的留白正是要聚焦它。
 */
export const FOCUS_OWNING_SELECTOR = [
    "input",
    "textarea",
    "select",
    "button",
    "a[href]",
    '[contenteditable="true"]',
    '[role="option"]',
    '[role="menuitem"]',
    '[role="slider"]',
    '[role="combobox"]',
    '[tabindex]:not([tabindex="-1"])',
].join(",");

/**
 * 这次点击是否应当由面板把焦点收回列表。
 *
 * @param target 指针按下的目标元素（`event.target`）。
 * @returns 目标是留白（不在任何"自己管焦点"的元素内）时返回 `true`。
 */
export function shouldPanelTakeFocus(target: Element | null): boolean {
    if (!target?.closest) return false;
    return target.closest(FOCUS_OWNING_SELECTOR) === null;
}
