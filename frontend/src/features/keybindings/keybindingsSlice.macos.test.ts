import { describe, expect, it, vi } from "vitest";

import { formatKeybinding, formatKeybindingList } from "./keybindingsSlice";

vi.mock("../../utils/platform", async (importOriginal) => {
    const actual = await importOriginal<typeof import("../../utils/platform")>();
    return { ...actual, IS_MAC: true };
});

describe("macOS 快捷键展示", () => {
    it("Delete 与 Backspace 都显示 ⌫", () => {
        expect(formatKeybinding({ key: "delete" })).toBe("⌫");
        expect(formatKeybinding({ key: "backspace" })).toBe("⌫");
        expect(formatKeybinding({ key: "delete", ctrl: true })).toBe("⌘+⌫");
    });

    it("paramBlock.delete 的两个槽位折叠成一个 ⌫", () => {
        expect(formatKeybindingList([{ key: "delete" }, { key: "backspace" }])).toBe("⌫");
    });

    it("不同组合键仍用分号分隔", () => {
        expect(
            formatKeybindingList([
                { key: "z", ctrl: true, shift: true },
                { key: "y", ctrl: true },
            ]),
        ).toBe("⌘+⇧+Z;⌘+Y");
    });
});
