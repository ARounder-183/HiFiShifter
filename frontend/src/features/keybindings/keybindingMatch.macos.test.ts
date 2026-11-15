import { describe, expect, it, vi } from "vitest";

import { matchesKeybinding } from "./keybindingMatch";

vi.mock("../../utils/platform", async (importOriginal) => {
    const actual = await importOriginal<typeof import("../../utils/platform")>();
    return { ...actual, IS_MAC: true };
});

const event = (
    key: string,
    mods: Partial<{ ctrl: boolean; meta: boolean; shift: boolean; alt: boolean }> = {},
) => ({
    key,
    ctrlKey: Boolean(mods.ctrl),
    metaKey: Boolean(mods.meta),
    shiftKey: Boolean(mods.shift),
    altKey: Boolean(mods.alt),
});

describe("macOS Delete / Backspace 匹配", () => {
    it("主 Delete（Backspace 事件）命中 delete 绑定", () => {
        expect(matchesKeybinding(event("Backspace"), { key: "delete" })).toBe(true);
    });

    it("⌘+主 Delete 命中 ctrl+delete 绑定（ctrl 在 mac 映射到 meta）", () => {
        expect(
            matchesKeybinding(event("Backspace", { meta: true }), { key: "delete", ctrl: true }),
        ).toBe(true);
        expect(
            matchesKeybinding(event("Backspace", { ctrl: true }), { key: "delete", ctrl: true }),
        ).toBe(false);
    });

    it("Fn+Delete（Delete 事件）不会被反向当成 Backspace", () => {
        expect(matchesKeybinding(event("Delete"), { key: "backspace" })).toBe(false);
        expect(matchesKeybinding(event("Delete"), { key: "delete" })).toBe(true);
    });

    it("用户自定义的 backspace 绑定仍可被主 Delete 命中", () => {
        expect(matchesKeybinding(event("Backspace"), { key: "backspace" })).toBe(true);
    });
});
