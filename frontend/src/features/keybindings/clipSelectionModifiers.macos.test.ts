import { describe, expect, it, vi } from "vitest";
import { resolveClipSelectionModifiers } from "./clipSelectionModifiers";
import { DEFAULT_KEYBINDINGS } from "./defaultKeybindings";

// macOS 环境模拟：IS_MAC=true 时 isPrimaryModifierDown 走 metaKey（⌘）。
// ctrl 字段绑定的平台适配（Windows: Ctrl / macOS: ⌘）由
// utils/platform.isPrimaryModifierDown 统一实现，此处验证选择修饰键
// 经过该层后的行为。
vi.mock("../../utils/platform", async (importOriginal) => {
    const actual = await importOriginal<typeof import("../../utils/platform")>();
    return {
        ...actual,
        IS_MAC: true,
        isPrimaryModifierDown: (event: { metaKey?: boolean }) => Boolean(event.metaKey),
    };
});

const MULTI = DEFAULT_KEYBINDINGS["modifier.clipMultiSelectToggle"];
const RANGE = DEFAULT_KEYBINDINGS["modifier.clipRangeSelect"];

describe("resolveClipSelectionModifiers — macOS（⌘ 为主修饰键）", () => {
    it("⌘（metaKey）命中默认多选切换绑定", () => {
        const m = resolveClipSelectionModifiers({
            event: { ctrlKey: false, shiftKey: false, altKey: false, metaKey: true },
            multiSelectToggleKb: MULTI,
            rangeSelectKb: RANGE,
        });
        expect(m.multiSelectToggleActive).toBe(true);
        expect(m.rangeSelectActive).toBe(false);
    });

    it("macOS 上物理 Ctrl（非 ⌘）不触发多选切换", () => {
        const m = resolveClipSelectionModifiers({
            event: { ctrlKey: true, shiftKey: false, altKey: false, metaKey: false },
            multiSelectToggleKb: MULTI,
            rangeSelectKb: RANGE,
        });
        expect(m.multiSelectToggleActive).toBe(false);
        expect(m.shouldPrimeSelection).toBe(true);
    });

    it("⌘+Shift：与 Windows Ctrl+Shift 相同的互斥语义", () => {
        const m = resolveClipSelectionModifiers({
            event: { ctrlKey: false, shiftKey: true, altKey: false, metaKey: true },
            multiSelectToggleKb: MULTI,
            rangeSelectKb: RANGE,
        });
        expect(m.multiSelectToggleActive).toBe(false);
        expect(m.rangeSelectActive).toBe(false);
        expect(m.shouldPrimeSelection).toBe(true);
    });
});
