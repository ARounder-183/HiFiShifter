// @vitest-environment jsdom
/*
 * 全局快捷键抑制作用域的回归测试。
 *
 * 【为什么必须有】这套机制取代了三个各自为政的 body 属性
 * （`data-keybindings-dialog-open` / `data-quick-search-open` /
 * `data-silence-dialog-open`）。旧写法在"两个模态同时打开"时会出错：
 * 先关掉的那个会解除抑制，而后一个还开着 —— 于是快捷键穿透到仍打开的
 * 对话框背后。导出对话框里弹出的冲突确认就是这个形状。
 *
 * 因此这里锁定的契约是：**抑制状态的解除必须等最后一个持有者释放**。
 */

import { afterEach, expect, test } from "vitest";

import {
    SHORTCUT_SUPPRESSION_ATTR,
    acquireShortcutSuppression,
    isShortcutSuppressed,
    releaseShortcutSuppression,
    resetShortcutScopesForTests,
} from "./shortcutScope";

afterEach(() => {
    resetShortcutScopesForTests();
});

test("无作用域时不抑制，且不留下 body 属性", () => {
    expect(isShortcutSuppressed()).toBe(false);
    expect(document.body.hasAttribute(SHORTCUT_SUPPRESSION_ATTR)).toBe(false);
});

test("持有期间抑制，释放后恢复", () => {
    const token = acquireShortcutSuppression();
    expect(isShortcutSuppressed()).toBe(true);
    expect(document.body.hasAttribute(SHORTCUT_SUPPRESSION_ATTR)).toBe(true);

    releaseShortcutSuppression(token);
    expect(isShortcutSuppressed()).toBe(false);
    expect(document.body.hasAttribute(SHORTCUT_SUPPRESSION_ATTR)).toBe(false);
});

test("重叠作用域：先释放的那个不解除抑制（旧实现的回归点）", () => {
    const outer = acquireShortcutSuppression();
    const inner = acquireShortcutSuppression();

    // 内层先关：外层仍然打开，抑制必须保持
    releaseShortcutSuppression(inner);
    expect(isShortcutSuppressed()).toBe(true);

    releaseShortcutSuppression(outer);
    expect(isShortcutSuppressed()).toBe(false);
});

test("重复释放是幂等的，不会误解除其他作用域", () => {
    const a = acquireShortcutSuppression();
    const b = acquireShortcutSuppression();

    releaseShortcutSuppression(a);
    releaseShortcutSuppression(a); // 重复释放
    expect(isShortcutSuppressed()).toBe(true);

    releaseShortcutSuppression(b);
    expect(isShortcutSuppressed()).toBe(false);
});

test("兼容尚未迁移的旧属性：任一存在即为抑制", () => {
    document.body.setAttribute("data-silence-dialog-open", "true");
    expect(isShortcutSuppressed()).toBe(true);
    document.body.removeAttribute("data-silence-dialog-open");
    expect(isShortcutSuppressed()).toBe(false);
});
