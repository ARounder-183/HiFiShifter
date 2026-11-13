import { describe, expect, it } from "vitest";

import { resolveFileBrowserDeleteAction } from "./fileBrowserDeleteKey";

const ev = (
    key: string,
    mods: Partial<{ metaKey: boolean; ctrlKey: boolean; altKey: boolean }> = {},
) => ({
    key,
    metaKey: Boolean(mods.metaKey),
    ctrlKey: Boolean(mods.ctrlKey),
    altKey: Boolean(mods.altKey),
});

describe("file browser Delete / Backspace 平台语义", () => {
    it("macOS：普通 ⌫ 返回上级，⌘+⌫ 删除，Fn+Delete 删除", () => {
        expect(resolveFileBrowserDeleteAction(ev("Backspace"), true)).toBe("parent");
        expect(resolveFileBrowserDeleteAction(ev("Backspace", { metaKey: true }), true)).toBe(
            "delete",
        );
        expect(resolveFileBrowserDeleteAction(ev("Delete"), true)).toBe("delete");
    });

    it("Windows/Linux：Backspace 返回上级，Delete 删除", () => {
        expect(resolveFileBrowserDeleteAction(ev("Backspace"), false)).toBe("parent");
        expect(resolveFileBrowserDeleteAction(ev("Delete"), false)).toBe("delete");
        // 非 mac 的 Ctrl+Backspace 不在这里当作删除（文本编辑语义留给输入框）。
        expect(
            resolveFileBrowserDeleteAction(ev("Backspace", { ctrlKey: true }), false),
        ).toBeNull();
    });

    it("Alt / Ctrl 组合不误触发文件删除", () => {
        expect(resolveFileBrowserDeleteAction(ev("Delete", { altKey: true }), true)).toBeNull();
        expect(resolveFileBrowserDeleteAction(ev("Delete", { ctrlKey: true }), false)).toBeNull();
    });
});
