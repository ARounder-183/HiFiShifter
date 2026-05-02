import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import type { ActionId, Keybinding, KeybindingMap, KeybindingOverrides } from "./types";
import { DEFAULT_KEYBINDINGS, ACTION_META } from "./defaultKeybindings";
import { loadKeybindingOverrides, saveKeybindingOverrides } from "./keybindingStorage";
const IS_MAC =
    typeof navigator !== "undefined" && navigator.platform?.toLowerCase().includes("mac");
// ─── State ───────────────────────────────────────────────────────

interface KeybindingsState {
    /** 用户自定义覆盖项（与默认不同的部分） */
    overrides: KeybindingOverrides;
}

const initialState: KeybindingsState = {
    overrides: loadKeybindingOverrides(),
};

// ─── Helpers ─────────────────────────────────────────────────────

/** 合并默认映射与用户覆盖，返回完整映射表 */
export function mergeKeybindings(overrides: KeybindingOverrides): KeybindingMap {
    return { ...DEFAULT_KEYBINDINGS, ...overrides } as KeybindingMap;
}

/** 判断两个 Keybinding 是否相等 */
function keybindingEqual(a: Keybinding, b: Keybinding): boolean {
    return (
        a.key === b.key &&
        Boolean(a.ctrl) === Boolean(b.ctrl) &&
        Boolean(a.shift) === Boolean(b.shift) &&
        Boolean(a.alt) === Boolean(b.alt) &&
        Boolean(a.modifierOnly) === Boolean(b.modifierOnly)
    );
}

/** 判断绑定是否为"无" */
export function isNoneBinding(kb: Keybinding): boolean {
    return kb.key === "__none__";
}

const VIBRATO_WHEEL_MODIFIERS = new Set<ActionId>([
    "modifier.vibratoAmplitudeAdjust",
    "modifier.vibratoFrequencyAdjust",
]);

/**
 * 将 Keybinding 格式化为可读字符串，如 "Ctrl+Shift+S"
 * 如果为"无"绑定，返回本地化占位文本
 */
export function formatKeybinding(kb: Keybinding, noneLabel?: string): string {
    if (isNoneBinding(kb)) return noneLabel ?? "—";

    const isMac =
        typeof navigator !== "undefined" &&
        navigator.platform?.toLowerCase().includes("mac");

    const parts: string[] = [];
    if (kb.ctrl) parts.push(isMac ? "⌘" : "Ctrl");
    if (kb.alt) parts.push(isMac ? "⌥" : "Alt");
    if (kb.shift) parts.push(isMac ? "⇧" : "Shift");

    // modifierOnly 类型无主键，直接返回修饰键名称
    if (kb.modifierOnly) {
        return parts.length > 0 ? parts.join("+") : prettifyKey(kb.key);
    }

    // 美化特殊键名
    const keyName = kb.key.length === 1 ? kb.key.toUpperCase() : prettifyKey(kb.key);
    parts.push(keyName);
    return parts.join("+");
}

function prettifyKey(key: string): string {
    const isMac =
        typeof navigator !== "undefined" &&
        navigator.platform?.toLowerCase().includes("mac");
    const map: Record<string, string> = {
        space: "Space",
        delete: isMac ? "⌦" : "Delete",
        backspace: isMac ? "⌫" : "Backspace",
        tab: isMac ? "⇥" : "Tab",
        enter: isMac ? "↩" : "Enter",
        escape: isMac ? "⎋" : "Escape",
        arrowup: "↑",
        arrowdown: "↓",
        arrowleft: "←",
        arrowright: "→",
    };
    return map[key.toLowerCase()] ?? key.charAt(0).toUpperCase() + key.slice(1);
}

// ─── Slice ───────────────────────────────────────────────────────

const keybindingsSlice = createSlice({
    name: "keybindings",
    initialState,
    reducers: {
        /** 设置某个操作的快捷键绑定 */
        setKeybinding(state, action: PayloadAction<{ actionId: ActionId; binding: Keybinding }>) {
            const { actionId, binding } = action.payload;
            const defaultBinding = DEFAULT_KEYBINDINGS[actionId];
            if (defaultBinding && keybindingEqual(defaultBinding, binding)) {
                // 与默认值相同，移除覆盖
                delete state.overrides[actionId];
            } else {
                state.overrides[actionId] = binding;
            }
            saveKeybindingOverrides(state.overrides);
        },

        /** 重置某个操作的快捷键为默认值 */
        resetKeybinding(state, action: PayloadAction<ActionId>) {
            delete state.overrides[action.payload];
            saveKeybindingOverrides(state.overrides);
        },

        /** 重置所有快捷键为默认值 */
        resetAllKeybindings(state) {
            state.overrides = {};
            saveKeybindingOverrides(state.overrides);
        },
    },
});

export const { setKeybinding, resetKeybinding, resetAllKeybindings } = keybindingsSlice.actions;

export default keybindingsSlice.reducer;

// ─── Selectors ───────────────────────────────────────────────────

/** 获取合并后的完整快捷键映射 */
export function selectMergedKeybindings(state: { keybindings: KeybindingsState }): KeybindingMap {
    return mergeKeybindings(state.keybindings.overrides);
}

/** 获取某个操作的当前快捷键 */
export function selectKeybinding(
    state: { keybindings: KeybindingsState },
    actionId: ActionId,
): Keybinding {
    return state.keybindings.overrides[actionId] ?? DEFAULT_KEYBINDINGS[actionId];
}

/** 检测冲突：给定新绑定，返回与之冲突的 actionId 列表（排除自身）
 *  对于修饰键绑定，仅当操作类型相同时才视为冲突 */
export function findConflicts(
    overrides: KeybindingOverrides,
    actionId: ActionId,
    newBinding: Keybinding,
): ActionId[] {
    if (isNoneBinding(newBinding)) {
        // 颤音滚轮修饰键允许配置为 None，但两者同时为 None 会冲突。
        if (VIBRATO_WHEEL_MODIFIERS.has(actionId)) {
            const merged = mergeKeybindings(overrides);
            const conflicts = (Object.entries(merged) as [ActionId, Keybinding][])
                .filter(
                    ([id, binding]) =>
                        id !== actionId &&
                        VIBRATO_WHEEL_MODIFIERS.has(id) &&
                        isNoneBinding(binding),
                )
                .map(([id]) => id);
            return conflicts;
        }
        return [];
    }
    const merged = mergeKeybindings(overrides);
    const conflicts: ActionId[] = [];
    const selfMeta = ACTION_META[actionId];
    for (const [id, binding] of Object.entries(merged)) {
        if (id === actionId) continue;
        if (isNoneBinding(binding)) continue;
        if (keybindingEqual(binding, newBinding)) {
            const otherMeta = ACTION_META[id as ActionId];
            // 修饰键：仅同操作类型才冲突
            if (selfMeta?.group === "modifier" && otherMeta?.group === "modifier") {
                if (
                    selfMeta.modifierOperationType &&
                    otherMeta.modifierOperationType &&
                    selfMeta.modifierOperationType !== otherMeta.modifierOperationType
                ) {
                    continue; // 不同操作类型，不算冲突
                }
            }
            // 作用域上下文：不同作用域的绑定不冲突
            // （例如 quickSearch 弹窗内的绑定不与全局绑定冲突）
            if (selfMeta?.scopedContext !== otherMeta?.scopedContext) {
                continue;
            }
            conflicts.push(id as ActionId);
        }
    }
    return conflicts;
}

/**
 * 检测事件中某个 modifierOnly 绑定的修饰键是否按下。
 * 适用于 PointerEvent / MouseEvent / KeyboardEvent 等任何带修饰键状态的事件。
 * 如果绑定为"无"，始终返回 false。
<<<<<<< Updated upstream
=======
 * 采用子集匹配：绑定中要求为 true 的修饰键必须被按下，
 * 未要求的修饰键允许同时按下，避免组合修饰键时误判失效。
 *
 * macOS 适配：Ctrl 修饰键同时匹配 metaKey（Cmd 键）和 ctrlKey（触控板双指捏合缩放）。
 * 浏览器会将触控板捏合手势转为 ctrlKey + wheel 事件。
>>>>>>> Stashed changes
 */
export function isModifierActive(
    kb: Keybinding,
    event: {
        ctrlKey: boolean;
        shiftKey: boolean;
        altKey: boolean;
        metaKey?: boolean;
    },
): boolean {
    if (isNoneBinding(kb)) return false;
<<<<<<< Updated upstream
    if (kb.ctrl) return IS_MAC ? Boolean(event.metaKey) : event.ctrlKey;
    if (kb.alt) return event.altKey;
    if (kb.shift) return event.shiftKey;
    return false;
=======
    const required = getModifierFlags(kb);
    if (!hasAnyModifierFlags(required)) return false;

    // macOS: Ctrl 同时匹配 Cmd（metaKey，键盘快捷键）和 Ctrl（ctrlKey，触控板捏合缩放）
    const pressedCtrl = IS_MAC
        ? (Boolean(event.metaKey) || Boolean(event.ctrlKey))
        : Boolean(event.ctrlKey);
    const pressedShift = Boolean(event.shiftKey);
    const pressedAlt = Boolean(event.altKey);

    return (
        (!required.ctrl || pressedCtrl) &&
        (!required.shift || pressedShift) &&
        (!required.alt || pressedAlt)
    );
>>>>>>> Stashed changes
}
