import type { Keybinding } from "./types";
import { IS_MAC } from "../../utils/platform";

/**
 * 将 KeyboardEvent 的按键信息规范化为小写 key 字符串
 */
export function normalizeEventKey(e: KeyboardEvent): string {
    // 对 Space 按键特殊处理
    if (e.key === " " || e.code === "Space") return "space";
    return e.key.toLowerCase();
}

/**
 * 判断按下的按键是否匹配某个 Keybinding 定义
 */
export function matchesKeybinding(e: KeyboardEvent, kb: Keybinding): boolean {
    const key = normalizeEventKey(e);
    if (key !== kb.key) return false;

    const modKey = IS_MAC ? e.metaKey : e.ctrlKey;
    const wantCtrl = Boolean(kb.ctrl);
    const wantShift = Boolean(kb.shift);
    const wantAlt = Boolean(kb.alt);

    if (modKey !== wantCtrl) return false;
    if (e.shiftKey !== wantShift) return false;
    if (e.altKey !== wantAlt) return false;
    return true;
}

/** 修饰键状态快照（isModifierActive 等也接受该形状） */
export type ModifierEventLike = {
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
    metaKey?: boolean;
};

function clearFineModifierState(e: KeyboardEvent, fineAdjustKb: Keybinding): KeyboardEvent {
    return {
        key: e.key,
        code: e.code,
        ctrlKey: fineAdjustKb.ctrl ? false : e.ctrlKey,
        metaKey: fineAdjustKb.ctrl ? false : e.metaKey,
        shiftKey: fineAdjustKb.shift ? false : e.shiftKey,
        altKey: fineAdjustKb.alt ? false : e.altKey,
    } as KeyboardEvent;
}

export function matchesKeybindingAllowingFineModifier(
    e: KeyboardEvent,
    kb: Keybinding,
    fineAdjustKb?: Keybinding,
): boolean {
    if (matchesKeybinding(e, kb)) {
        return true;
    }
    if (!fineAdjustKb) {
        return false;
    }
    // 与 keybindingsSlice.isModifierActive 语义一致：绑定中要求按下的
    // 修饰键必须按下，未要求的修饰键允许同时按下（子集匹配）。
    const required = {
        ctrl: Boolean(fineAdjustKb.ctrl),
        shift: Boolean(fineAdjustKb.shift),
        alt: Boolean(fineAdjustKb.alt),
    };
    const pressedCtrl = IS_MAC ? e.metaKey : e.ctrlKey;
    if (
        (required.ctrl && !pressedCtrl) ||
        (required.shift && !e.shiftKey) ||
        (required.alt && !e.altKey)
    ) {
        return false;
    }
    return matchesKeybinding(clearFineModifierState(e, fineAdjustKb), kb);
}
