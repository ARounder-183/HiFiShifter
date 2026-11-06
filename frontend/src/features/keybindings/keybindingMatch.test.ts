import { describe, expect, it } from "vitest";

import {
    canonicalKeyFromEvent,
    matchesKeybinding,
    normalizeEventKey,
    physicalKeyFromEvent,
} from "./keybindingMatch";
import type { Keybinding } from "./types";

/** 构造键盘事件（仅填充匹配所需的字段）。 */
function keyEvent(
    key: string,
    opts: {
        code?: string;
        ctrl?: boolean;
        shift?: boolean;
        alt?: boolean;
        meta?: boolean;
    } = {},
): KeyboardEvent {
    return {
        key,
        code: opts.code,
        ctrlKey: Boolean(opts.ctrl),
        shiftKey: Boolean(opts.shift),
        altKey: Boolean(opts.alt),
        metaKey: Boolean(opts.meta),
        preventDefault() {},
    } as unknown as KeyboardEvent;
}

describe("keybindingMatch — 物理键位匹配（Shift 上档字符归位）", () => {
    it("normalizeEventKey 保持既有行为（小写字符）", () => {
        expect(normalizeEventKey(keyEvent("T"))).toBe("t");
        expect(normalizeEventKey(keyEvent("-", { code: "Minus" }))).toBe("-");
    });

    it("physicalKeyFromEvent：Shift 上档字符回退到 e.code 对应的基础字符", () => {
        // US 布局 Shift+= 产出 "+"。
        expect(physicalKeyFromEvent(keyEvent("+", { code: "Equal" }))).toBe("=");
        // US 布局 Shift+- 产出 "_"。
        expect(physicalKeyFromEvent(keyEvent("_", { code: "Minus" }))).toBe("-");
        expect(physicalKeyFromEvent(keyEvent("{", { code: "BracketLeft" }))).toBe("[");
        expect(physicalKeyFromEvent(keyEvent("}", { code: "BracketRight" }))).toBe("]");
        // 字符未变形（未按 Shift）：返回 null（沿用 e.key 即可）。
        expect(physicalKeyFromEvent(keyEvent("=", { code: "Equal" }))).toBeNull();
        // 非映射键位：null。
        expect(physicalKeyFromEvent(keyEvent("a", { code: "KeyA" }))).toBeNull();
        expect(physicalKeyFromEvent(keyEvent("+", { code: "NumpadAdd" }))).toBeNull();
    });

    it("matchesKeybinding：默认 Shift 变体（key: '=' + shift）命中 US 上档事件", () => {
        const kb: Keybinding = { key: "=", shift: true };
        expect(matchesKeybinding(keyEvent("+", { code: "Equal", shift: true }), kb)).toBe(true);
        // Ctrl 按下的同事件不匹配（修饰键不符）。
        expect(
            matchesKeybinding(keyEvent("+", { code: "Equal", shift: true, ctrl: true }), kb),
        ).toBe(false);
        // 无 Shift 的事件不匹配。
        expect(matchesKeybinding(keyEvent("=", { code: "Equal" }), kb)).toBe(false);
    });

    it("matchesKeybinding：Ctrl 微调变体（key: '=' + ctrl）直接命中（字符不变形）", () => {
        const kb: Keybinding = { key: "=", ctrl: true };
        expect(matchesKeybinding(keyEvent("=", { code: "Equal", ctrl: true }), kb)).toBe(true);
        expect(matchesKeybinding(keyEvent("=", { code: "Equal" }), kb)).toBe(false);
    });

    it("matchesKeybinding：非 Shift 场景行为不变（plain '=' / '-' / ']' / '['）", () => {
        expect(matchesKeybinding(keyEvent("=", { code: "Equal" }), { key: "=" })).toBe(true);
        expect(matchesKeybinding(keyEvent("-", { code: "Minus" }), { key: "-" })).toBe(true);
        expect(matchesKeybinding(keyEvent("]", { code: "BracketRight" }), { key: "]" })).toBe(true);
        expect(matchesKeybinding(keyEvent("[", { code: "BracketLeft" }), { key: "[" })).toBe(true);
        // 无 Shift 要求的绑定不匹配带 Shift 的事件（即便物理键归位成功）。
        expect(matchesKeybinding(keyEvent("+", { code: "Equal", shift: true }), { key: "=" })).toBe(
            false,
        );
    });

    it("canonicalKeyFromEvent：录入值与默认绑定可比（'+' → '='）", () => {
        expect(canonicalKeyFromEvent(keyEvent("+", { code: "Equal" }))).toBe("=");
        expect(canonicalKeyFromEvent(keyEvent("_", { code: "Minus" }))).toBe("-");
        expect(canonicalKeyFromEvent(keyEvent("{", { code: "BracketLeft" }))).toBe("[");
        expect(canonicalKeyFromEvent(keyEvent("}", { code: "BracketRight" }))).toBe("]");
        // 字母/普通键沿用小写事件字符。
        expect(canonicalKeyFromEvent(keyEvent("T", { code: "KeyT" }))).toBe("t");
        expect(canonicalKeyFromEvent(keyEvent(" ", { code: "Space" }))).toBe("space");
    });
});
