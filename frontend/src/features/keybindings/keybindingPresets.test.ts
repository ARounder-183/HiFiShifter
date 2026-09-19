/**
 * 快捷键预设单测（`./keybindingPresets`）。
 *
 * 【要锁住的约束】「拉伸」修饰键拆分后（时间轴 clip 边缘 / 参数编辑器选区边缘），
 * 预设必须把**两个** action 都设到同一键位 —— 否则应用预设后两个表面行为不一致
 * （用户按预设的键位只在一个表面生效）。同时逐项钉住各预设的拉伸键位：
 * VEGAS Pro 为 `Ctrl`，REAPER 为 `Alt`（与默认同值）。
 */
import { describe, expect, it } from "vitest";

import { createModifierOnlyBinding } from "./keybindingsSlice";
import { DEFAULT_KEYBINDINGS } from "./defaultKeybindings";
import { KEYBINDING_PRESETS, KEYBINDING_PRESET_IDS } from "./keybindingPresets";

const STRETCH_ACTIONS = ["modifier.clipStretch", "modifier.paramStretch"] as const;

/** 从修饰键绑定中读出按下的修饰键（"ctrl" / "alt" / "shift" / "none"）。 */
function modifierOf(binding: {
    ctrl?: boolean;
    alt?: boolean;
    shift?: boolean;
    key: string;
}): "ctrl" | "alt" | "shift" | "none" {
    if (binding.key === "__none__") return "none";
    if (binding.ctrl) return "ctrl";
    if (binding.alt) return "alt";
    if (binding.shift) return "shift";
    return "none";
}

describe("默认拉伸修饰键", () => {
    it("两个拉伸动作默认均为 Alt", () => {
        for (const action of STRETCH_ACTIONS) {
            expect(DEFAULT_KEYBINDINGS[action]).toEqual(
                createModifierOnlyBinding({ ctrl: false, shift: false, alt: true }),
            );
        }
    });

    it("两个动作是彼此独立的绑定项（改绑互不牵连）", () => {
        // 同一对象引用会让 `setKeybinding` 的「与默认相同则删覆盖」判定串台。
        expect(DEFAULT_KEYBINDINGS["modifier.clipStretch"]).not.toBe(
            DEFAULT_KEYBINDINGS["modifier.paramStretch"],
        );
    });
});

describe("预设中的拉伸键位", () => {
    it("VEGAS Pro：两个表面均为 Ctrl", () => {
        expect(modifierOf(KEYBINDING_PRESETS.vegasPro["modifier.clipStretch"]!)).toBe("ctrl");
        expect(modifierOf(KEYBINDING_PRESETS.vegasPro["modifier.paramStretch"]!)).toBe("ctrl");
    });

    it("REAPER：两个表面均为 Alt（与默认一致，应用预设即清除旧覆盖）", () => {
        expect(modifierOf(KEYBINDING_PRESETS.reaper["modifier.clipStretch"]!)).toBe("alt");
        expect(modifierOf(KEYBINDING_PRESETS.reaper["modifier.paramStretch"]!)).toBe("alt");
    });

    it("REAPER 的两个拉伸绑定等于默认值（setKeybinding 会据此删除覆盖项）", () => {
        // 预设值 == 默认值时 `setKeybinding` 走"移除覆盖"分支，因此必须真的一致，
        // 否则应用 REAPER 后会留下一条与默认同值的冗余覆盖。
        expect(KEYBINDING_PRESETS.reaper["modifier.clipStretch"]).toEqual(
            DEFAULT_KEYBINDINGS["modifier.clipStretch"],
        );
        expect(KEYBINDING_PRESETS.reaper["modifier.paramStretch"]).toEqual(
            DEFAULT_KEYBINDINGS["modifier.paramStretch"],
        );
    });

    it("不设拉伸键位的预设保持默认 Alt（由合并回退保证）", () => {
        for (const presetId of KEYBINDING_PRESET_IDS) {
            const preset = KEYBINDING_PRESETS[presetId];
            const setsClip = preset["modifier.clipStretch"] !== undefined;
            const setsParam = preset["modifier.paramStretch"] !== undefined;
            // 要么两个都设（表面一致），要么都不设（回退到各自默认 Alt）。
            expect(setsClip).toBe(setsParam);
        }
    });

    it("凡设了 clip 拉伸的预设必须同时设 param 拉伸（否则应用后两个表面分叉）", () => {
        for (const presetId of KEYBINDING_PRESET_IDS) {
            const preset = KEYBINDING_PRESETS[presetId];
            if (preset["modifier.clipStretch"] !== undefined) {
                expect(preset["modifier.paramStretch"]).toBeDefined();
                expect(modifierOf(preset["modifier.paramStretch"]!)).toBe(
                    modifierOf(preset["modifier.clipStretch"]!),
                );
            }
        }
    });
});
