import { describe, expect, it } from "vitest";
import { DEFAULT_KEYBINDINGS } from "./defaultKeybindings";
import { resolveActionByFocus } from "./focusRouting";

function keyEvent(key: string, mods: Partial<KeyboardEvent> = {}): KeyboardEvent {
    return {
        key,
        code: key.length === 1 ? `Key${key.toUpperCase()}` : key,
        ctrlKey: false,
        shiftKey: false,
        altKey: false,
        metaKey: false,
        ...mods,
    } as KeyboardEvent;
}

describe("param block shortcuts", () => {
    it("uses G / U and leaves Escape unbound", () => {
        expect(DEFAULT_KEYBINDINGS["paramBlock.group"]).toEqual([{ key: "g" }]);
        expect(DEFAULT_KEYBINDINGS["paramBlock.ungroup"]).toEqual([{ key: "u" }]);
        expect(DEFAULT_KEYBINDINGS["paramBlock.deselect"]).toEqual([{ key: "__none__" }]);
    });

    it("routes block actions only in pianoRoll + block selection mode", () => {
        expect(resolveActionByFocus(keyEvent("g"), DEFAULT_KEYBINDINGS, "pianoRoll", "block")).toBe(
            "paramBlock.group",
        );
        expect(resolveActionByFocus(keyEvent("u"), DEFAULT_KEYBINDINGS, "pianoRoll", "block")).toBe(
            "paramBlock.ungroup",
        );
        expect(resolveActionByFocus(keyEvent("g"), DEFAULT_KEYBINDINGS, "timeline", "block")).toBe(
            "clip.group",
        );
        expect(resolveActionByFocus(keyEvent("s"), DEFAULT_KEYBINDINGS, "pianoRoll", "block")).toBe(
            "paramBlock.split",
        );
        expect(resolveActionByFocus(keyEvent("s"), DEFAULT_KEYBINDINGS, "timeline", "block")).toBe(
            "clip.split",
        );
        expect(
            resolveActionByFocus(keyEvent("s"), DEFAULT_KEYBINDINGS, "pianoRoll", "select"),
        ).toBe("clip.split");
    });
});
