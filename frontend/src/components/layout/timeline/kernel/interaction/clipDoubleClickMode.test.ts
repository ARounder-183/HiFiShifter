import { describe, expect, it } from "vitest";

import { resolveClipDoubleClickMode } from "./clipDoubleClickMode";
import type { Keybinding } from "../../../../../features/keybindings/types";

/** 默认绑定：`modifier.clipRangeToParamSelection` = 仅 Alt（见 defaultKeybindings）。 */
const ALT_ONLY: Keybinding = { key: "alt", modifierOnly: true, alt: true };

/**
 * 构造修饰键快照。缺省全部为 false，避免每个用例重复写四个字段。
 */
function mods(overrides: Partial<Record<"ctrlKey" | "shiftKey" | "altKey" | "metaKey", boolean>> = {}) {
    return {
        ctrlKey: overrides.ctrlKey ?? false,
        shiftKey: overrides.shiftKey ?? false,
        altKey: overrides.altKey ?? false,
        metaKey: overrides.metaKey ?? false,
    };
}

describe("resolveClipDoubleClickMode — 双击 clip 的选区写入方式", () => {
    it("按住 Alt 双击 = toggle（并入 / 挖掉该块范围）", () => {
        expect(resolveClipDoubleClickMode(ALT_ONLY, mods({ altKey: true }))).toBe("toggle");
    });

    it("不按修饰键 = replace（与旧实现的普通双击逐字一致）", () => {
        expect(resolveClipDoubleClickMode(ALT_ONLY, mods())).toBe("replace");
    });

    it("只按 Ctrl / Shift / Meta 都不是 toggle（它们各自有别的点击含义）", () => {
        expect(resolveClipDoubleClickMode(ALT_ONLY, mods({ ctrlKey: true }))).toBe("replace");
        expect(resolveClipDoubleClickMode(ALT_ONLY, mods({ shiftKey: true }))).toBe("replace");
        expect(resolveClipDoubleClickMode(ALT_ONLY, mods({ metaKey: true }))).toBe("replace");
    });

    it("绑定缺省（未提供 / null）= replace，不得抛错", () => {
        expect(resolveClipDoubleClickMode(undefined, mods({ altKey: true }))).toBe("replace");
        expect(resolveClipDoubleClickMode(null, mods({ altKey: true }))).toBe("replace");
    });

    it("用户改绑后以绑定为准：改到 Shift 则 Shift 双击是 toggle、Alt 不再是", () => {
        // 刻意用 Shift 而不是 Ctrl 做这条断言：`ctrl` 标志经 `isPrimaryModifierDown`
        // 在 macOS 上映射为 ⌘（见 utils/platform），断言会随运行平台变化。Shift 无
        // 平台映射，因此本用例在 CI（Linux）与 macOS 上结论一致。
        const rebound: Keybinding = { key: "shift", modifierOnly: true, shift: true };
        expect(resolveClipDoubleClickMode(rebound, mods({ shiftKey: true }))).toBe("toggle");
        expect(resolveClipDoubleClickMode(rebound, mods({ altKey: true }))).toBe("replace");
    });

    it("none 绑定（用户显式解绑）= replace", () => {
        const none: Keybinding = { key: "None" };
        expect(resolveClipDoubleClickMode(none, mods({ altKey: true }))).toBe("replace");
    });
});
