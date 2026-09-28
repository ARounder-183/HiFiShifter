/**
 * 全局快捷键抑制的模态作用域。
 *
 * 【为什么需要统一机制】此前有三个各自为政的 `document.body` 属性：
 * `data-keybindings-dialog-open`、`data-quick-search-open`、
 * `data-silence-dialog-open`，分别在 3 个组件里写入，在
 * `useKeybindings` 里读取。后果有两个：
 *
 * 1. **绝大多数对话框完全没有抑制**。42 个对话框里只有 3 个接入了这套机制，
 *    另外 39 个按空格会触发播放、按字母会触发时间轴动作 —— 键穿透到背后。
 *    这不是疏忽，而是接入成本太高：想让新对话框抑制快捷键，得先知道存在
 *    这么一个"魔法属性"，再自己写 setAttribute / removeAttribute / cleanup。
 *
 * 2. **无法表达重叠**。两个对话框同时打开时（例如导出对话框里弹出的冲突确认），
 *    各自管各自的属性，先关掉的那个会解除抑制，而后一个还开着。
 *
 * 现在改成"引用计数 + 单一属性"：任何模态表面 acquire 一次，release 一次，
 * 归零时才解除。`AppDialog` 自动接入，因此新对话框**默认就是正确的**，
 * 不需要作者知道任何细节。
 */

import { useEffect } from "react";

const suppressedScopes = new Set<symbol>();

/** 单一 body 属性，取代此前的三个。 */
export const SHORTCUT_SUPPRESSION_ATTR = "data-hs-shortcuts-suppressed";

/** 旧属性名，保留读取以兼容尚未迁移的调用点。 */
const LEGACY_ATTRS = [
    "data-keybindings-dialog-open",
    "data-quick-search-open",
    "data-silence-dialog-open",
] as const;

function syncAttribute(): void {
    if (typeof document === "undefined") return;
    if (suppressedScopes.size > 0) {
        document.body.setAttribute(SHORTCUT_SUPPRESSION_ATTR, String(suppressedScopes.size));
    } else {
        document.body.removeAttribute(SHORTCUT_SUPPRESSION_ATTR);
    }
}

/**
 * 声明"当前有模态表面打开，请抑制全局快捷键"。
 *
 * @returns 释放令牌，必须在卸载/关闭时调用一次以配对。
 */
export function acquireShortcutSuppression(): symbol {
    const token = Symbol("shortcut-scope");
    suppressedScopes.add(token);
    syncAttribute();
    return token;
}

/** 释放一个作用域。重复释放是安全的（幂等）。 */
export function releaseShortcutSuppression(token: symbol): void {
    if (!suppressedScopes.delete(token)) return;
    syncAttribute();
}

/**
 * 当前是否应抑制全局快捷键。
 *
 * `useKeybindings` 的按键入口读这一个函数即可，不必再枚举属性名。
 */
export function isShortcutSuppressed(): boolean {
    if (suppressedScopes.size > 0) return true;
    if (typeof document === "undefined") return false;
    return LEGACY_ATTRS.some((attr) => document.body.hasAttribute(attr));
}

/** 仅供测试：清空所有作用域。 */
export function resetShortcutScopesForTests(): void {
    suppressedScopes.clear();
    syncAttribute();
}

/**
 * 非 `AppDialog` 的模态表面（命令面板、浮层选择器等）用这个钩子接入。
 *
 * 调用点此前各自 `setAttribute` / `removeAttribute`，且属性名互不相同；
 * 换成钩子后，接入成本从"知道存在一个魔法属性"降到一行。
 */
export function useShortcutSuppression(enabled: boolean): void {
    useEffect(() => {
        if (!enabled) return;
        const token = acquireShortcutSuppression();
        return () => releaseShortcutSuppression(token);
    }, [enabled]);
}
