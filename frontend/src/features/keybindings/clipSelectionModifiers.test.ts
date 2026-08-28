import { describe, expect, it } from "vitest";
import { resolveClipSelectionModifiers } from "./clipSelectionModifiers";
import { DEFAULT_KEYBINDINGS } from "./defaultKeybindings";
import type { Keybinding } from "./types";

const MULTI = DEFAULT_KEYBINDINGS["modifier.clipMultiSelectToggle"];
const RANGE = DEFAULT_KEYBINDINGS["modifier.clipRangeSelect"];
const NONE: Keybinding = { key: "__none__", modifierOnly: true };

function mods(
    flags: { ctrl?: boolean; shift?: boolean; alt?: boolean },
    multi: Keybinding = MULTI,
    range: Keybinding = RANGE,
) {
    return resolveClipSelectionModifiers({
        event: {
            ctrlKey: Boolean(flags.ctrl),
            shiftKey: Boolean(flags.shift),
            altKey: Boolean(flags.alt),
        },
        multiSelectToggleKb: multi,
        rangeSelectKb: range,
    });
}

describe("resolveClipSelectionModifiers — 与写死时代行为对齐", () => {
    it("仅 Ctrl（主修饰键）：多选切换生效", () => {
        const m = mods({ ctrl: true });
        expect(m.multiSelectToggleActive).toBe(true);
        expect(m.rangeSelectActive).toBe(false);
        expect(m.shouldPrimeSelection).toBe(false);
    });

    it("仅 Shift：范围选择生效", () => {
        const m = mods({ shift: true });
        expect(m.rangeSelectActive).toBe(true);
        expect(m.multiSelectToggleActive).toBe(false);
        expect(m.shouldPrimeSelection).toBe(false);
    });

    it("Ctrl+Shift 同时按下：两者互斥，退回普通单选", () => {
        const m = mods({ ctrl: true, shift: true });
        expect(m.multiSelectToggleActive).toBe(false);
        expect(m.rangeSelectActive).toBe(false);
        expect(m.shouldPrimeSelection).toBe(true);
    });

    it("物理 Alt 旁路：Alt+Ctrl / Alt+Shift 都不当作选择修饰键", () => {
        expect(mods({ ctrl: true, alt: true }).multiSelectToggleActive).toBe(false);
        expect(mods({ shift: true, alt: true }).rangeSelectActive).toBe(false);
        expect(mods({ ctrl: true, alt: true }).shouldPrimeSelection).toBe(true);
    });

    it("raw 状态不受互斥影响（供 allowSeek 等使用）", () => {
        const m = mods({ ctrl: true, shift: true });
        expect(m.multiSelectToggleRaw).toBe(true);
        expect(m.rangeSelectRaw).toBe(true);
    });

    it("绑定为“无”时对应行为关闭", () => {
        const m = mods({ shift: true }, NONE, RANGE);
        expect(m.multiSelectToggleActive).toBe(false);
        expect(m.rangeSelectActive).toBe(true);
    });

    it("用户把两个绑定配置成同一组合时互斥退回普通单选", () => {
        const m = mods({ ctrl: true }, MULTI, MULTI);
        expect(m.multiSelectToggleActive).toBe(false);
        expect(m.rangeSelectActive).toBe(false);
        expect(m.shouldPrimeSelection).toBe(true);
    });
});
