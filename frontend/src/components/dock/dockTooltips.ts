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
 * 组装"拖拽以重排"提示（两行）。
 *
 * 文案里带 `\n`，由自定义 tooltip 的 `white-space: pre-line` 渲染为两行
 * （见 `AppTooltip`）。**不要**改用原生 `title`：原生 tooltip 无法保证换行与
 * 主题一致，也不受本项目的样式控制。
 */
export function dockDragHint(
    modifier: DockSettings["dockModifier"] | undefined,
    translate: (key: string) => string,
): string {
    const label = dockModifierLabel(modifier);
    if (label === null) return translate("dock_drag_hint_always");
    return translate("dock_drag_hint").replace("{modifier}", label);
}
