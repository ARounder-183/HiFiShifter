/**
 * vibratoDragAdjust.ts
 * 管理直线/颤音拖拽时的键盘/滚轮调参映射与步进计算。
 */

import type { Keybinding } from "../../../features/keybindings/types";
import { isModifierActive } from "../../../features/keybindings/keybindingsSlice";
import { matchesKeybinding } from "../../../features/keybindings/useKeybindings";
import type { ParamName } from "./types";

export type VibratoAdjustTarget = "amplitude" | "frequency";
export type VibratoAdjustDirection = 1 | -1;

export type VibratoDragKeyboardBindings = {
    amplitudeIncrease: Keybinding;
    amplitudeDecrease: Keybinding;
    frequencyIncrease: Keybinding;
    frequencyDecrease: Keybinding;
};

export type VibratoDragKeyboardAdjustment = {
    target: VibratoAdjustTarget;
    direction: VibratoAdjustDirection;
};

function clearFineModifierState(event: KeyboardEvent, fineAdjustKb: Keybinding): KeyboardEvent {
    return {
        key: event.key,
        code: event.code,
        ctrlKey: fineAdjustKb.ctrl ? false : event.ctrlKey,
        metaKey: fineAdjustKb.ctrl ? false : event.metaKey,
        shiftKey: fineAdjustKb.shift ? false : event.shiftKey,
        altKey: fineAdjustKb.alt ? false : event.altKey,
    } as KeyboardEvent;
}

function matchesKeybindingAllowingFineModifier(
    event: KeyboardEvent,
    keybinding: Keybinding,
    fineAdjustKb?: Keybinding,
): boolean {
    if (matchesKeybinding(event, keybinding)) {
        return true;
    }

    if (!fineAdjustKb || !isModifierActive(fineAdjustKb, event)) {
        return false;
    }

    const normalizedEvent = clearFineModifierState(event, fineAdjustKb);
    return matchesKeybinding(normalizedEvent, keybinding);
}

export function resolveVibratoDragKeyboardAdjustment(
    event: KeyboardEvent,
    bindings: VibratoDragKeyboardBindings,
    fineAdjustKb?: Keybinding,
): VibratoDragKeyboardAdjustment | null {
    if (bindings.amplitudeIncrease.modifierOnly || bindings.amplitudeDecrease.modifierOnly) {
        return null;
    }
    if (bindings.frequencyIncrease.modifierOnly || bindings.frequencyDecrease.modifierOnly) {
        return null;
    }

    if (matchesKeybindingAllowingFineModifier(event, bindings.amplitudeIncrease, fineAdjustKb)) {
        return { target: "amplitude", direction: 1 };
    }
    if (matchesKeybindingAllowingFineModifier(event, bindings.amplitudeDecrease, fineAdjustKb)) {
        return { target: "amplitude", direction: -1 };
    }
    if (matchesKeybindingAllowingFineModifier(event, bindings.frequencyIncrease, fineAdjustKb)) {
        return { target: "frequency", direction: 1 };
    }
    if (matchesKeybindingAllowingFineModifier(event, bindings.frequencyDecrease, fineAdjustKb)) {
        return { target: "frequency", direction: -1 };
    }

    return null;
}

export function computeVibratoDragAdjustment(input: {
    editParam: ParamName;
    currentParamRange?: { min: number; max: number };
    amplitude: number;
    frequency: number;
    target: VibratoAdjustTarget;
    direction: VibratoAdjustDirection;
    steps: number;
    fineScale: number;
}): { amplitude: number; frequency: number } {
    const safeSteps = Math.max(1, Math.round(Math.abs(input.steps)));
    const safeFineScale =
        Number.isFinite(input.fineScale) && input.fineScale > 0 ? input.fineScale : 1;

    let amplitude = input.amplitude;
    let frequency = input.frequency;

    if (input.target === "amplitude") {
        const rangeSpan =
            input.editParam === "pitch"
                ? 48
                : Math.max(
                      1e-6,
                      Number(input.currentParamRange?.max ?? 1) -
                          Number(input.currentParamRange?.min ?? 0),
                  );
        const baseAmpStep = Math.max(rangeSpan / 200, 0.01);
        amplitude += input.direction * baseAmpStep * safeSteps * safeFineScale;
    } else {
        const ratio = Math.pow(1 + 0.1 * safeFineScale, safeSteps);
        frequency = input.direction > 0 ? frequency * ratio : frequency / ratio;
        frequency = Math.max(1e-4, frequency);
    }

    return { amplitude, frequency };
}
