import { describe, expect, it } from "vitest";

import type { Keybinding } from "../../../features/keybindings/types";
import { isFadeShapeCycleModifierHeld, type FadeShapeCycleEventLike } from "./fadeShapeCycle";

/** 构造事件（只带判定用到的四个字段）。 */
function event(overrides: Partial<FadeShapeCycleEventLike> = {}): FadeShapeCycleEventLike {
    return { ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, ...overrides };
}

/** 构造键位绑定。 */
function binding(overrides: Partial<Keybinding> = {}): Keybinding {
    return { key: "control", ctrl: true, alt: false, shift: false, ...overrides } as Keybinding;
}

describe("isFadeShapeCycleModifierHeld", () => {
    it("绑定缺省时不触发", () => {
        expect(isFadeShapeCycleModifierHeld(null, event({ ctrlKey: true }))).toBe(false);
        expect(isFadeShapeCycleModifierHeld(undefined, event({ ctrlKey: true }))).toBe(false);
    });

    it("常规组合：要求 Ctrl 时，Ctrl 或 ⌘ 都算命中（macOS 主修饰键语义）", () => {
        const kb = binding({ ctrl: true });
        expect(isFadeShapeCycleModifierHeld(kb, event({ ctrlKey: true }))).toBe(true);
        expect(isFadeShapeCycleModifierHeld(kb, event({ metaKey: true }))).toBe(true);
        expect(isFadeShapeCycleModifierHeld(kb, event())).toBe(false);
    });

    it("modifierOnly 单键：key = control 时只按 Ctrl 即命中", () => {
        const kb = binding({ modifierOnly: true, key: "control", ctrl: false });
        expect(isFadeShapeCycleModifierHeld(kb, event({ ctrlKey: true }))).toBe(true);
        expect(isFadeShapeCycleModifierHeld(kb, event())).toBe(false);
    });

    it("modifierOnly 单键：key = alt 时只按 Alt 即命中", () => {
        const kb = binding({ modifierOnly: true, key: "alt", ctrl: false });
        expect(isFadeShapeCycleModifierHeld(kb, event({ altKey: true }))).toBe(true);
        expect(isFadeShapeCycleModifierHeld(kb, event({ ctrlKey: true }))).toBe(false);
    });

    it("未要求的修饰键不影响判定（多按一个键不应让循环失效）", () => {
        const kb = binding({ ctrl: true });
        expect(isFadeShapeCycleModifierHeld(kb, event({ ctrlKey: true, shiftKey: true }))).toBe(
            true,
        );
        expect(isFadeShapeCycleModifierHeld(kb, event({ ctrlKey: true, altKey: true }))).toBe(true);
    });

    it("要求多个修饰键时全部满足才命中", () => {
        const kb = binding({ ctrl: true, shift: true });
        expect(isFadeShapeCycleModifierHeld(kb, event({ ctrlKey: true }))).toBe(false);
        expect(isFadeShapeCycleModifierHeld(kb, event({ ctrlKey: true, shiftKey: true }))).toBe(
            true,
        );
    });
});
