/**
 * 参数块手势修饰键的纯策略层。
 *
 * 创建手势只锁存“创建意图”，普通 / 平直形态由当前键位实时决定：
 * - createFlat 优先；
 * - createBlock 单独按下时回到 normal；
 * - 两者都松开时保持上一次有效形态（取消只归 Escape / pointercancel）。
 */
import {
    firstBinding,
    isModifierActive,
    isNoneBinding,
} from "../../../features/keybindings/keybindingsSlice";
import type { KeybindingMap } from "../../../features/keybindings/types";

export type ParamBlockCreateShape = "normal" | "flat";

export interface ParamBlockModifierEvent {
    readonly ctrlKey: boolean;
    readonly shiftKey: boolean;
    readonly altKey: boolean;
    readonly metaKey: boolean;
}

export interface ParamBlockModifierSnapshot {
    readonly createBlock: boolean;
    readonly createFlat: boolean;
    readonly splitAtClick: boolean;
    readonly snapToggle: boolean;
}

export function resolveParamBlockModifierSnapshot(args: {
    readonly event: ParamBlockModifierEvent;
    readonly keybindings: KeybindingMap;
}): ParamBlockModifierSnapshot {
    const active = (
        action:
            | "modifier.paramBlockCreate"
            | "modifier.paramBlockCreateFlat"
            | "modifier.paramBlockSplitAtClick"
            | "modifier.clipNoSnap",
    ): boolean => {
        const binding = firstBinding(args.keybindings[action]);
        return !isNoneBinding(binding) && isModifierActive(binding, args.event);
    };
    return {
        createBlock: active("modifier.paramBlockCreate"),
        createFlat: active("modifier.paramBlockCreateFlat"),
        splitAtClick: active("modifier.paramBlockSplitAtClick"),
        snapToggle: active("modifier.clipNoSnap"),
    };
}

export function resolveParamBlockCreateShape(args: {
    readonly previous: ParamBlockCreateShape;
    readonly snapshot: ParamBlockModifierSnapshot;
}): ParamBlockCreateShape {
    if (args.snapshot.createFlat) return "flat";
    if (args.snapshot.createBlock) return "normal";
    return args.previous;
}
