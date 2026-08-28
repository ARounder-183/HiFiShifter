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
        "modifier.clipStretch": modifierBinding("control"),
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
        "modifier.clipStretch": modifierBinding("control"),
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
