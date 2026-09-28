/*
 * 停靠系统的 tooltip 文案。
 *
 * 【为什么要一个模块】"拖拽以重排 / 按住 Ctrl 可停靠"这句提示需要把**当前生效的
 * 停靠修饰键**渲染进去（Windows/Linux 是 Ctrl、macOS 是 ⌘、也可被用户改成 Alt /
 * Shift / 不按修饰键）。修饰键的可读名称由快捷键系统统一提供（`formatKeybinding`
 * 已经处理了平台差异与符号），这里只是把它取出来填进模板 —— 平台判断不能在本
 * 模块再写一遍，否则两处显示会不一致。
 */

import { formatKeybinding } from "../../features/keybindings/keybindingsSlice";
import type { DockSettings } from "../../features/dock/dockSettings";

/**
 * 停靠修饰键的可读名称。
 *
 * @returns 修饰键文本（如 `Ctrl` / `⌘` / `Alt`）；`dockModifier === "none"`
 *          （无需修饰键、始终停靠）时返回 `null`，调用方据此换一句提示。
 */
export function dockModifierLabel(
    modifier: DockSettings["dockModifier"] | undefined,
): string | null {
    // 【键名不能取 `__none__`】`isNoneBinding` 只看 `key === "__none__"`，命中后
    // `formatKeybinding` 会走"无绑定"分支返回占位符（`—`），提示因此显示成
    // "按住 - 可停靠"。修饰键绑定本来就只需要修饰键标志，键名取一个真实键名即可：
    // `modifierOnly` 分支会忽略它、直接返回修饰键名称（Ctrl / ⌘ / Alt / ⇧）。
    switch (modifier) {
        case "alt":
            return formatKeybinding({ key: "alt", alt: true, modifierOnly: true });
        case "shift":
            return formatKeybinding({ key: "shift", shift: true, modifierOnly: true });
        case "none":
            return null;
        default:
            // primary：Ctrl（Windows/Linux）/ ⌘（macOS），由 `formatKeybinding` 决定。
            return formatKeybinding({ key: "control", ctrl: true, modifierOnly: true });
    }
}

/**
 * "按住 {修饰键} 可停靠"这一句（**唯一来源**）。
 *
 * 【为什么单独成函数】这句话同时出现在两个地方：**停靠抓手**的悬停提示（第二行）
 * 与**拖拽幽灵**的提示（第二行）。两处必须永远一致 —— 用户的要求是"预览伪影的提示
 * 永远告诉用户按住修饰键可停靠"。若各写一份模板，改一处就会漏另一处。
 */
export function dockModifierHint(
    modifier: DockSettings["dockModifier"] | undefined,
    translate: (key: string) => string,
): string {
    const label = dockModifierLabel(modifier);
    if (label === null) return translate("dock_dock_hint_always");
    return translate("dock_dock_hint").replace("{modifier}", label);
}

/**
 * 停靠抓手的悬停提示（两行）：`拖拽以重排` + `按住 {修饰键} 可停靠`。
 *
 * 文案里带换行符，由自定义 tooltip 的 `white-space: pre-line` 渲染为两行
 * （见 `AppTooltip`）。**不要**改用原生 `title`：原生 tooltip 无法保证换行与
 * 主题一致，也不受本项目的样式控制。
 *
 * 特殊说明：**浮动窗口的标题栏不再使用本提示**（那里悬停不显示任何提示，见
 * `DockFloatingLayer`）；它只服务于停靠态的抓手。
 */
export function dockDragHint(
    modifier: DockSettings["dockModifier"] | undefined,
    translate: (key: string) => string,
): string {
    // 两行：第一行说明"这是拖拽重排"，第二行给出停靠方式（与幽灵提示同源）。
    return [translate("dock_drag_rearrange"), dockModifierHint(modifier, translate)].join("\n");
}
