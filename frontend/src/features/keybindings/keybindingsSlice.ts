import {
    createListenerMiddleware,
    createSlice,
    isAnyOf,
    type PayloadAction,
} from "@reduxjs/toolkit";
import type { ActionId, ActionMeta, Keybinding, KeybindingMap, KeybindingOverrides } from "./types";
import { DEFAULT_KEYBINDINGS, ACTION_META } from "./defaultKeybindings";
import { loadKeybindingOverrides, saveKeybindingOverrides } from "./keybindingStorage";
import { IS_MAC, isPrimaryModifierDown } from "../../utils/platform";
// ─── State ───────────────────────────────────────────────────────

/**
 * 本分片的状态类型。
 *
 * 导出是因为 `RootState` 由它组合而成 —— SDK 的声明产出需要能命名它
 * （否则 `tsc --emitDeclarationOnly` 报 TS4023「cannot be named」）。
 */
export interface KeybindingsState {
    /** 用户自定义覆盖项（与默认不同的部分） */
    overrides: KeybindingOverrides;
}

const initialState: KeybindingsState = {
    overrides: loadKeybindingOverrides(),
};

interface ModifierFlags {
    ctrl: boolean;
    shift: boolean;
    alt: boolean;
}

// ─── Helpers ─────────────────────────────────────────────────────

/** 合并默认映射与用户覆盖，返回完整映射表 */
export function mergeKeybindings(overrides: KeybindingOverrides): KeybindingMap {
    return { ...DEFAULT_KEYBINDINGS, ...overrides } as KeybindingMap;
}

/** 判断两个 Keybinding 是否相等 */
function keybindingEqual(a: Keybinding, b: Keybinding): boolean {
    if (isNoneBinding(a) && isNoneBinding(b)) {
        return true;
    }

    if (Boolean(a.modifierOnly) || Boolean(b.modifierOnly)) {
        if (Boolean(a.modifierOnly) !== Boolean(b.modifierOnly)) {
            return false;
        }
        const aFlags = getModifierFlags(a);
        const bFlags = getModifierFlags(b);
        return (
            aFlags.ctrl === bFlags.ctrl &&
            aFlags.shift === bFlags.shift &&
            aFlags.alt === bFlags.alt
        );
    }

    return (
        a.key === b.key &&
        Boolean(a.ctrl) === Boolean(b.ctrl) &&
        Boolean(a.shift) === Boolean(b.shift) &&
        Boolean(a.alt) === Boolean(b.alt)
    );
}

/** 判断绑定是否为"无" */
export function isNoneBinding(kb: Keybinding): boolean {
    return kb.key === "__none__";
}

function hasAnyModifierFlags(flags: ModifierFlags): boolean {
    return flags.ctrl || flags.shift || flags.alt;
}

function inferModifierFlagsFromLegacyKey(key: string): ModifierFlags {
    const lower = key.toLowerCase();
    if (
        lower === "control" ||
        lower === "ctrl" ||
        lower === "meta" ||
        lower === "command" ||
        lower === "cmd"
    ) {
        return { ctrl: true, shift: false, alt: false };
    }
    if (lower === "shift") {
        return { ctrl: false, shift: true, alt: false };
    }
    if (lower === "alt" || lower === "option") {
        return { ctrl: false, shift: false, alt: true };
    }
    return { ctrl: false, shift: false, alt: false };
}

function canonicalModifierKey(flags: ModifierFlags): string {
    if (flags.ctrl) return "control";
    if (flags.alt) return "alt";
    if (flags.shift) return "shift";
    return "__none__";
}

export function getModifierFlags(kb: Keybinding): ModifierFlags {
    const explicitFlags: ModifierFlags = {
        ctrl: Boolean(kb.ctrl),
        shift: Boolean(kb.shift),
        alt: Boolean(kb.alt),
    };

    if (!kb.modifierOnly) {
        return explicitFlags;
    }

    if (hasAnyModifierFlags(explicitFlags)) {
        return explicitFlags;
    }

    return inferModifierFlagsFromLegacyKey(kb.key);
}

export function createModifierOnlyBinding(flags: ModifierFlags): Keybinding {
    if (!hasAnyModifierFlags(flags)) {
        return { key: "__none__", modifierOnly: true };
    }
    return {
        key: canonicalModifierKey(flags),
        modifierOnly: true,
        ...(flags.ctrl ? { ctrl: true } : {}),
        ...(flags.shift ? { shift: true } : {}),
        ...(flags.alt ? { alt: true } : {}),
    };
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
    const parts: string[] = [];
    const modifierFlags = getModifierFlags(kb);
    if (modifierFlags.ctrl) parts.push(IS_MAC ? "⌘" : "Ctrl");
    if (modifierFlags.alt) parts.push(IS_MAC ? "⌥" : "Alt");
    if (modifierFlags.shift) parts.push(IS_MAC ? "⇧" : "Shift");

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
    const map: Record<string, string> = {
        space: "Space",
        delete: "Delete",
        backspace: "Backspace",
        tab: "Tab",
        enter: "Enter",
        escape: "Escape",
        arrowup: "↑",
        arrowdown: "↓",
        arrowleft: "←",
        arrowright: "→",
    };
    return map[key.toLowerCase()] ?? key.charAt(0).toUpperCase() + key.slice(1);
}

// ─── 绑定列表（`readonly Keybinding[]`）的读写辅助 ────────────────
//
// 模型见 types.ts 的 `KeybindingMap`：一个动作持有一**列表**绑定，下标 0 即
// 主绑定。下面这些函数是该列表的**唯一**读写口径 —— 数组形状的不变式只在
// `normalizeBindings` 里实现一次，其余地方一律通过它。

/** "无绑定"的单元素列表。 */
const NONE_BINDING_LIST: readonly Keybinding[] = [{ key: "__none__" }];

/** 取主绑定（下标 0）；列表为空时退化为"无"。 */
export function firstBinding(bindings: readonly Keybinding[] | undefined): Keybinding {
    return bindings?.[0] ?? NONE_BINDING_LIST[0];
}

/** 列表是否与"无绑定"等价（空，或唯一元素是 `__none__`）。 */
export function isNoneBindingList(bindings: readonly Keybinding[] | undefined): boolean {
    return !bindings || bindings.length === 0 || bindings.every(isNoneBinding);
}

/**
 * 规范化绑定列表，使其满足 `KeybindingMap` 声明的不变式。
 *
 * 规则（顺序即语义，不要重排）：
 * 1. 剔除 `null` / `undefined`（localStorage 与预设都可能带进来）；
 * 2. 去重（保留首次出现的位置 —— 下标 0 的主绑定身份因此稳定）；
 * 3. `__none__` 只允许作为**唯一**元素存在；出现在其它位置的一律删除
 *    （"第 2 个键是无"没有意义，只有"整个动作无绑定"才有）；
 * 4. 空列表回退为 `[{ key: "__none__" }]`。
 */
export function normalizeBindings(
    bindings: readonly (Keybinding | null | undefined)[],
): Keybinding[] {
    const seen: Keybinding[] = [];
    for (const binding of bindings) {
        if (!binding) continue;
        if (isNoneBinding(binding)) continue;
        if (seen.some((existing) => keybindingEqual(existing, binding))) continue;
        seen.push(binding);
    }
    if (seen.length === 0) return [...NONE_BINDING_LIST];
    return seen;
}

/**
 * 两个绑定列表是否相等（**顺序敏感**）。
 *
 * 顺序即语义（下标 0 是主绑定、菜单显示与长按重复都以它为准），因此
 * `[A, B]` 与 `[B, A]` 是两次不同的配置，不能判等。
 */
export function keybindingsEqual(a: readonly Keybinding[], b: readonly Keybinding[]): boolean {
    if (a.length !== b.length) return false;
    return a.every((binding, index) => keybindingEqual(binding, b[index]));
}

/**
 * 把一个动作的全部绑定格式化为一行（`"Ctrl+Shift+Z / Ctrl+Y"`）。
 *
 * 全部为"无"时返回 `noneLabel`（缺省 `—`）。调用方（设置面板的摘要、
 * 按钮 tooltip）用它展示**完整**配置；菜单栏那种宽度受限的位置请用
 * `formatKeybinding(firstBinding(...))` 只显示主绑定。
 */
export function formatKeybindingList(
    bindings: readonly Keybinding[] | undefined,
    noneLabel?: string,
): string {
    const parts = (bindings ?? []).filter((binding) => !isNoneBinding(binding));
    if (parts.length === 0) return noneLabel ?? "—";
    return parts.map((binding) => formatKeybinding(binding)).join(" / ");
}

// ─── Slice ───────────────────────────────────────────────────────

const keybindingsSlice = createSlice({
    name: "keybindings",
    initialState,
    reducers: {
        /**
         * 用一份**完整**的绑定列表替换某个操作的绑定。
         *
         * 【为什么只有一个写入口】"改某一个槽位 / 追加一个 / 删掉一个 / 套用
         * 预设"在数据上都只是"这个动作现在绑这些"—— 让调用方（设置面板）算出
         * 目标数组，规范化与"是否等于默认"的判断就只需在这里写一次。多开几个
         * 粒度更细的 reducer 只会让同一套不变式散到四处。
         */
        setKeybindings(
            state,
            action: PayloadAction<{ actionId: ActionId; bindings: readonly Keybinding[] }>,
        ) {
            const { actionId, bindings } = action.payload;
            const normalized = normalizeBindings(bindings);
            const defaults = DEFAULT_KEYBINDINGS[actionId];
            if (defaults && keybindingsEqual(defaults, normalized)) {
                // 与默认一致 → 不保留覆盖（改回默认即"没有自定义"）。
                delete state.overrides[actionId];
            } else {
                state.overrides[actionId] = normalized;
            }
        },

        /** 重置某个操作的快捷键为默认值 */
        resetKeybinding(state, action: PayloadAction<ActionId>) {
            delete state.overrides[action.payload];
        },

        /** 重置所有快捷键为默认值 */
        resetAllKeybindings(state) {
            state.overrides = {};
        },
    },
});

export const { setKeybindings, resetKeybinding, resetAllKeybindings } = keybindingsSlice.actions;

/**
 * 持久化收口：覆盖项落盘统一由 listener middleware 完成。
 * reducer 必须保持纯函数 —— 在 reducer 里写 localStorage 会让 DevTools
 * 的跳转/重放（以及可能的未来 SSR/快照恢复路径）反复触发副作用。
 */
export const keybindingsPersistenceMiddleware = createListenerMiddleware<{
    keybindings: { overrides: KeybindingOverrides };
}>();
keybindingsPersistenceMiddleware.startListening({
    matcher: isAnyOf(
        keybindingsSlice.actions.setKeybindings,
        keybindingsSlice.actions.resetKeybinding,
        keybindingsSlice.actions.resetAllKeybindings,
    ),
    effect: (_action, api) => {
        saveKeybindingOverrides(api.getState().keybindings.overrides);
    },
});

export default keybindingsSlice.reducer;

// ─── Selectors ───────────────────────────────────────────────────

/**
 * 获取合并后的完整快捷键映射。
 *
 * 必须按 `overrides` 引用做引用记忆化：该选择器被挂载在应用根组件与
 * MenuBar 上，若每次调用都返回新对象（`{...DEFAULT, ...overrides}`），
 * useSelector 的严格相等比较会判定"已变化"，任何 dispatch（包括播放时
 * 33Hz 的轮询、电平表事件）都会级联重渲染整个应用。
 */
const selectMergedKeybindingsCache = {
    overrides: null as KeybindingOverrides | null,
    merged: null as KeybindingMap | null,
};

export function selectMergedKeybindings(state: { keybindings: KeybindingsState }): KeybindingMap {
    const overrides = state.keybindings.overrides;
    if (
        selectMergedKeybindingsCache.merged === null ||
        selectMergedKeybindingsCache.overrides !== overrides
    ) {
        selectMergedKeybindingsCache.overrides = overrides;
        selectMergedKeybindingsCache.merged = mergeKeybindings(overrides);
    }
    return selectMergedKeybindingsCache.merged;
}

/** 获取某个操作当前生效的**全部**绑定（已合并用户覆盖）。 */
export function selectKeybindings(
    state: { keybindings: KeybindingsState },
    actionId: ActionId,
): readonly Keybinding[] {
    return state.keybindings.overrides[actionId] ?? DEFAULT_KEYBINDINGS[actionId];
}

/**
 * 获取某个操作的**主绑定**（列表下标 0）。
 *
 * 【为什么保留单绑定形态的选择器】大量调用点只关心"这一个键"：菜单栏与
 * 右键菜单的快捷键文案、修饰键手势的读取（`modifier.*` 永远只有一个绑定）、
 * 长按重复的默认基准。给它们一个单绑定入口，就不必在十几处各自写
 * `[0]`/`firstBinding`，多绑定的语义也不会渗进不需要它的地方。
 */
export function selectKeybinding(
    state: { keybindings: KeybindingsState },
    actionId: ActionId,
): Keybinding {
    return firstBinding(selectKeybindings(state, actionId));
}

/**
 * 该动作的绑定列表里是否已存在候选绑定（可排除某个槽位）。
 *
 * 用于录入时拦截"把同一个键重复绑到同一动作的另一个槽位" —— 那既没有意义，
 * 又会让规范化把它悄悄去掉、用户看不到任何反馈。跨动作的冲突另由
 * `findConflicts` 负责。
 */
export function hasDuplicateBinding(
    bindings: readonly Keybinding[],
    candidate: Keybinding,
    excludeIndex = -1,
): boolean {
    if (isNoneBinding(candidate)) return false;
    return bindings.some(
        (binding, index) => index !== excludeIndex && keybindingEqual(binding, candidate),
    );
}

/** 检测冲突：给定新绑定，返回与之冲突的 actionId 列表（排除自身） */
export function findConflicts(
    overrides: KeybindingOverrides,
    actionId: ActionId,
    newBinding: Keybinding,
): ActionId[] {
    if (isNoneBinding(newBinding)) {
        // 颤音滚轮修饰键允许配置为 None，但两者同时为 None 会冲突。
        if (VIBRATO_WHEEL_MODIFIERS.has(actionId)) {
            const merged = mergeKeybindings(overrides);
            const conflicts = (Object.entries(merged) as [ActionId, readonly Keybinding[]][])
                .filter(
                    ([id, bindings]) =>
                        id !== actionId &&
                        VIBRATO_WHEEL_MODIFIERS.has(id) &&
                        isNoneBindingList(bindings),
                )
                .map(([id]) => id);
            return conflicts;
        }
        return [];
    }
    const merged = mergeKeybindings(overrides);
    const conflicts: ActionId[] = [];
    const selfMeta = ACTION_META[actionId];
    for (const [id, bindings] of Object.entries(merged) as [ActionId, readonly Keybinding[]][]) {
        if (id === actionId) continue;
        // 一个动作只要**任一**槽位与候选相同即构成冲突；同一个动作最多记一次。
        const hit = bindings.some(
            (binding) => !isNoneBinding(binding) && keybindingEqual(binding, newBinding),
        );
        if (!hit) continue;
        {
            const otherMeta = ACTION_META[id as ActionId];
            if (isModifierMeta(selfMeta) && isModifierMeta(otherMeta)) {
                // 修饰键 vs 修饰键：需要「手势类型相同 + 生效场景有交集」才算冲突。
                // 同一修饰键在不同场景重复使用（如 Alt 同时用于音频块 Slip 与
                // 淡化包络曲率）是刻意设计，与 DAW 惯例一致，不提示冲突。
                if (
                    selfMeta.modifierOperationType !== otherMeta.modifierOperationType ||
                    !scenesIntersect(selfMeta.conflictScenes, otherMeta.conflictScenes)
                ) {
                    continue;
                }
            } else if (!isModifierMeta(selfMeta) && !isModifierMeta(otherMeta)) {
                // 键盘快捷键 vs 键盘快捷键：作用域上下文不同则不冲突。
                //
                // 这把「赋值时的冲突检测」与「运行时的焦点路由」（见
                // focusRouting.ts）对齐：不同作用域共用按键是刻意支持的设计
                // （例如「添加轨道」Ctrl+T 与参数编辑器「音高设置到」Ctrl+T、
                // 音频块删除 Delete 与轨道删除 Ctrl+Delete），它们不会在同一
                // 焦点下同时响应，而是由当前焦点决定期望的编辑目标 —— 与
                // 复制/粘贴（clip.copy 与 pianoRoll.copy 共用 Ctrl+C）同构。
                // 同样的按键在相同作用域内重复使用（例如两个 paramEditorSelect
                // 操作）才是真正的歧义，必须提示冲突。
                if (selfMeta?.scopedContext !== otherMeta?.scopedContext) {
                    continue;
                }
            } else {
                // 修饰键 vs 键盘快捷键：绑定的目标种类不同（modifierOnly 标志
                // 已在 keybindingEqual 中区分），理论上不会走到这里，保守跳过。
                continue;
            }
            conflicts.push(id as ActionId);
        }
    }
    return conflicts;
}

/** 是否为修饰键绑定（以 meta 中声明了手势类型为准） */
function isModifierMeta(meta?: ActionMeta): boolean {
    return Boolean(meta?.modifierOperationType);
}

/** 两个场景集合是否有交集；任一缺省时视为相交（保守提示冲突） */
function scenesIntersect(a?: readonly string[], b?: readonly string[]): boolean {
    if (!a || !b) return true;
    return a.some((scene) => b.includes(scene));
}

/**
 * 检测事件中某个 modifierOnly 绑定的修饰键是否按下。
 * 适用于 PointerEvent / MouseEvent / KeyboardEvent 等任何带修饰键状态的事件。
 * 如果绑定为"无"，始终返回 false。
 * 采用子集匹配：绑定中要求为 true 的修饰键必须被按下，
 * 未要求的修饰键允许同时按下，避免组合修饰键时误判失效。
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
    const required = getModifierFlags(kb);
    if (!hasAnyModifierFlags(required)) return false;

    const pressedCtrl = isPrimaryModifierDown(event);
    const pressedShift = Boolean(event.shiftKey);
    const pressedAlt = Boolean(event.altKey);

    return (
        (!required.ctrl || pressedCtrl) &&
        (!required.shift || pressedShift) &&
        (!required.alt || pressedAlt)
    );
}
