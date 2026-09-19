/**
 * keybindingPresets.ts
 * Defines keybinding presets and their partial override entries.
 */

import type { ActionId, Keybinding } from "./types";
import { createModifierOnlyBinding } from "./keybindingsSlice";

export type KeybindingPresetId =
    | "spaceReturnPlayhead"
    | "touchpad"
    | "reaper"
    | "vegasPro"
    | "vocalShifter";

export type KeybindingPresetSelectionId = "custom" | "default" | KeybindingPresetId;

const NONE_MODIFIER_BINDING: Keybinding = { key: "__none__", modifierOnly: true };

type ModifierToken = "control" | "shift" | "alt";

function modifierBinding(modifier: ModifierToken | ModifierToken[]): Keybinding {
    const modifiers = Array.isArray(modifier) ? modifier : [modifier];
    return createModifierOnlyBinding({
        ctrl: modifiers.includes("control"),
        shift: modifiers.includes("shift"),
        alt: modifiers.includes("alt"),
    });
}

export const KEYBINDING_PRESET_IDS: KeybindingPresetId[] = [
    "spaceReturnPlayhead",
    "touchpad",
    "reaper",
    "vegasPro",
    "vocalShifter",
];

export const KEYBINDING_PRESET_SELECTION_IDS: KeybindingPresetSelectionId[] = [
    "custom",
    "default",
    ...KEYBINDING_PRESET_IDS,
];

export const KEYBINDING_PRESETS: Record<
    KeybindingPresetId,
    Partial<Record<ActionId, Keybinding>>
> = {
    spaceReturnPlayhead: {
        "playback.toggle": { key: "enter" },
        "playback.stop": { key: "space" },
    },
    touchpad: {
        "modifier.horizontalZoom": modifierBinding("shift"),
        "modifier.pianoRollVerticalZoom": modifierBinding("control"),
        "modifier.scrollHorizontal": NONE_MODIFIER_BINDING,
        "modifier.scrollVertical": modifierBinding("alt"),
        "modifier.pianoKeysVerticalScroll": NONE_MODIFIER_BINDING,
        "modifier.pianoKeysVerticalZoom": modifierBinding("alt"),
    },
    reaper: {
        "playback.toggle": { key: "enter" },
        "playback.stop": { key: "space" },
        "playback.focusCursor": { key: "'" },
        "modifier.clipSlipEdit": modifierBinding("alt"),
        // 拉伸在两个表面各自独立（时间轴 clip 边缘 / 参数编辑器选区边缘）。
        // 预设原本只设一个 `modifier.clipStretch`，两个表面都被它牵引；拆分后
        // 两个 action 都要显式设同一键位，否则应用预设后两个表面行为不一致。
        // REAPER 预设用 Alt（与 clipSlipEdit 同键，但 clip.move / clip.edge 是
        // 不同拖拽目标，与 DAW 惯例一致）。
        "modifier.clipStretch": modifierBinding("alt"),
        "modifier.paramStretch": modifierBinding("alt"),
        "modifier.horizontalZoom": NONE_MODIFIER_BINDING,
        "modifier.pianoRollVerticalZoom": modifierBinding("control"),
        "modifier.scrollHorizontal": modifierBinding("alt"),
        "modifier.scrollVertical": modifierBinding("shift"),
        "modifier.pianoKeysVerticalScroll": NONE_MODIFIER_BINDING,
        "modifier.pianoKeysVerticalZoom": modifierBinding("control"),
    },
    vegasPro: {
        "playback.toggle": { key: "enter" },
        "playback.stop": { key: "space" },
        "playback.focusCursor": { key: "\\" },
        "modifier.clipSlipEdit": modifierBinding("alt"),
        // VEGAS Pro 的拉伸修饰键是 Ctrl，两个表面都设 Ctrl。
        "modifier.clipStretch": modifierBinding("control"),
        "modifier.paramStretch": modifierBinding("control"),
        "modifier.horizontalZoom": NONE_MODIFIER_BINDING,
        "modifier.pianoRollVerticalZoom": modifierBinding("alt"),
        "modifier.scrollHorizontal": modifierBinding("shift"),
        "modifier.scrollVertical": modifierBinding("control"),
        "modifier.pianoKeysVerticalScroll": NONE_MODIFIER_BINDING,
        "modifier.pianoKeysVerticalZoom": modifierBinding("control"),
    },
    vocalShifter: {
        "playback.toggle": { key: "space" },
        "playback.stop": { key: "enter" },
        "modifier.horizontalZoom": modifierBinding("control"),
        "modifier.pianoRollVerticalZoom": modifierBinding("alt"),
        "modifier.scrollHorizontal": NONE_MODIFIER_BINDING,
        "modifier.scrollVertical": modifierBinding("shift"),
        "modifier.pianoKeysVerticalScroll": NONE_MODIFIER_BINDING,
        "modifier.pianoKeysVerticalZoom": modifierBinding("control"),
    },
};

export function isKeybindingPresetId(value: string): value is KeybindingPresetId {
    return KEYBINDING_PRESET_IDS.includes(value as KeybindingPresetId);
}
