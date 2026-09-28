/**
 * `translateOutsideReact` 自检。
 *
 * 【要钉死什么】组件之外（创建独立窗口时把面板标题交给系统标题栏）取到的必须是
 * **文案**，而不是 i18n 键 —— 用户报告过窗口标题显示成 "undo_history_title"。
 * 另外：非浏览器环境（单测）不能因为没有 localStorage 而抛错。
 */
import { test } from "vitest";

import { translateOutsideReact } from "./I18nProvider";

test("i18n/I18nProvider.test.ts translateOutsideReact", () => {
    // 面板标题键必须能翻译成人类可读文本（独立窗口标题栏用它）。
    const undoTitle = translateOutsideReact("undo_history_title");
    if (undoTitle === "undo_history_title") {
        throw new Error("undo_history_title must resolve to a label, not echo the key");
    }
    if (!undoTitle.trim()) throw new Error("translated title must not be empty");

    // 未登记的键原样返回：宁可显示键名，也不要显示空白标题。
    if (translateOutsideReact("definitely_not_a_key") !== "definitely_not_a_key") {
        throw new Error("unknown keys must fall back to the key itself");
    }
});
