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
 * 物理键位 → 主键字符（仅覆盖 Shift 会改写字符、且本项目默认绑定用到
 * 的标点键）。`e.code` 是与布局无关的物理键名，Shift 按下时 `e.key`
 * 变成上档字符（US 布局 Shift+= 产出 "+"），按 `e.code` 才能还原出
 * 与默认绑定（key: "="）一致的主键。
 */
const PHYSICAL_KEY_BY_CODE: Record<string, string> = {
    equal: "=",
    minus: "-",
    bracketleft: "[",
    bracketright: "]",
};

/**
 * 从键盘事件解析「物理主键字符」：Shift 改写字符（如 US 布局 Shift+=
 * → "+"）时回退到 `e.code` 对应的基础字符；其它键位（字母、数字等）
 * 返回 null，调用方沿用 `normalizeEventKey` 即可。
 */
export function physicalKeyFromEvent(e: Pick<KeyboardEvent, "key" | "code">): string | null {
    const code = (e.code ?? "").toLowerCase();
    const base = PHYSICAL_KEY_BY_CODE[code];
    if (!base) return null;
    return e.key.toLowerCase() === base ? null : base;
}

/**
 * 录入用规范化：Shift 上档字符（"+"、"_"、"{"、"}" 等）回退到物理键位
 * 的基础字符，保证录入值与默认绑定（key: "=" + shift: true）可比、可
 * 冲突检测、可匹配。
 */
export function canonicalKeyFromEvent(e: KeyboardEvent): string {
    return physicalKeyFromEvent(e) ?? normalizeEventKey(e);
}

/**
 * 判断按下的按键是否匹配某个 Keybinding 定义
 */
export function matchesKeybinding(e: KeyboardEvent, kb: Keybinding): boolean {
    // 主键匹配同时接受「事件字符」与「物理键位字符」：Shift+标点在 US
    // 布局下 e.key 是上档字符（Shift+= → "+"），按 e.code 还原后才能
    // 命中 key: "=" 的默认绑定；非上档场景两值相同，行为不变。
    const key = normalizeEventKey(e);
    const physical = physicalKeyFromEvent(e);
    if (key !== kb.key && physical !== kb.key) return false;

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
