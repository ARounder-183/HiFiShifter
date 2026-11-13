import { describe, expect, it, vi } from "vitest";

import { resolveActionByFocus } from "./focusRouting";
import type { Keybinding } from "./types";

vi.mock("../../utils/platform", async (importOriginal) => {
    const actual = await importOriginal<typeof import("../../utils/platform")>();
    return { ...actual, IS_MAC: true };
});

const lists = (map: Record<string, Keybinding>) =>
    Object.fromEntries(Object.entries(map).map(([id, kb]) => [id, [kb]]));

const event = (key: string, mods: Partial<{ ctrl: boolean; shift: boolean; alt: boolean }> = {}) =>
    ({
        key,
        code: key === "Backspace" ? "Backspace" : "Delete",
        ctrlKey: false,
        // ⌘ is the primary modifier on macOS.
        metaKey: Boolean(mods.ctrl),
        shiftKey: Boolean(mods.shift),
        altKey: Boolean(mods.alt),
    }) as KeyboardEvent;

describe("macOS 焦点路由：同一个 ⌫ 的三种语义", () => {
    it("时间轴：主 Delete 命中 clip.delete", () => {
        const map = lists({ "clip.delete": { key: "delete" } });
        expect(resolveActionByFocus(event("Backspace"), map, "timeline", "select")).toBe(
            "clip.delete",
        );
    });

    it("参数编辑器选择族：主 Delete 命中 edit.initialize", () => {
        const map = lists({
            "clip.delete": { key: "delete" },
            "edit.initialize": { key: "backspace" },
        });
        expect(resolveActionByFocus(event("Backspace"), map, "pianoRoll", "select")).toBe(
            "edit.initialize",
        );
    });

    it("参数块模式：主 Delete 命中 paramBlock.delete（优先级高于 clip.delete）", () => {
        const map = lists({
            "clip.delete": { key: "delete" },
            "paramBlock.delete": { key: "delete" },
        });
        expect(resolveActionByFocus(event("Backspace"), map, "pianoRoll", "block")).toBe(
            "paramBlock.delete",
        );
    });

    it("⌘+⌫（meta）命中 track.delete 的 ctrl 绑定", () => {
        const map = lists({
            "clip.delete": { key: "delete" },
            "track.delete": { key: "delete", ctrl: true },
        });
        expect(
            resolveActionByFocus(event("Backspace", { ctrl: true }), map, "trackHeader", "select"),
        ).toBe("track.delete");
    });
});
