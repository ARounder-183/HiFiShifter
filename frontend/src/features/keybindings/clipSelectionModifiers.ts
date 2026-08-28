/**
 * clipSelectionModifiers.ts - 时间轴音频块"点击选择"修饰键解析。
 *
 * 把原先散落在 TrackLane / ClipItem / ClipEdgeHandles / useClipDrag 中的
 * 写死 Ctrl/Shift 选择修饰键统一收敛为可配置绑定：
 * - modifier.clipMultiSelectToggle（默认 Ctrl / macOS ⌘）：按住并点击切换多选
 * - modifier.clipRangeSelect（默认 Shift）：按住并点击从锚点范围选择
 *
 * 语义（与写死时代逐条对齐）：
 * - 两个选择修饰键互斥：同时按下（或被配置成同一组合）时都退回普通单选，
 *   避免歧义。检测基于 isModifierActive 的子集匹配（要求的修饰键按下即
 *   成立，允许额外修饰键），因此必须显式做互斥。
 * - 物理 Alt（getModifierState，而非可配置的拉伸绑定状态）按下时两者均被
 *   绕过：Alt 在时间轴上承载 Slip/拉伸等拖拽语义，此时点击沿用普通选择
 *   预备，不当作选择修饰键。物理键检测与 altPressed（可配置拉伸修饰键）
 *   解耦，避免用户把拉伸绑到 Ctrl/Shift 时误伤选择行为。
 */

import { isModifierActive } from "./keybindingsSlice";
import type { Keybinding } from "./types";

/** PointerEvent / 原生事件的修饰键子集（含可选的物理键查询） */
interface SelectionModifierEvent {
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
    metaKey?: boolean;
    nativeEvent?: { getModifierState?: (key: string) => boolean };
}

export interface ClipSelectionModifierState {
    /** 物理 Alt 是否按下（选择修饰键的总旁路） */
    altKeyDown: boolean;
    /** 多选切换修饰键原始状态（未做互斥/Alt 旁路） */
    multiSelectToggleRaw: boolean;
    /** 范围选择修饰键原始状态（未做互斥/Alt 旁路） */
    rangeSelectRaw: boolean;
    /** 多选切换修饰键生效（原始状态 且 未被 Alt 旁路/范围选择互斥） */
    multiSelectToggleActive: boolean;
    /** 范围选择修饰键生效（原始状态 且 未被 Alt 旁路/多选切换互斥） */
    rangeSelectActive: boolean;
    /** 普通点击：应预备/替换单选 */
    shouldPrimeSelection: boolean;
}

export function resolveClipSelectionModifiers(input: {
    event: SelectionModifierEvent;
    multiSelectToggleKb: Keybinding;
    rangeSelectKb: Keybinding;
}): ClipSelectionModifierState {
    const altKeyDown = Boolean(
        input.event.altKey || input.event.nativeEvent?.getModifierState?.("Alt"),
    );
    const multiSelectToggleRaw = isModifierActive(input.multiSelectToggleKb, input.event);
    const rangeSelectRaw = isModifierActive(input.rangeSelectKb, input.event);

    const multiSelectToggleActive =
        multiSelectToggleRaw && !altKeyDown && !rangeSelectRaw;
    const rangeSelectActive = rangeSelectRaw && !altKeyDown && !multiSelectToggleRaw;

    return {
        altKeyDown,
        multiSelectToggleRaw,
        rangeSelectRaw,
        multiSelectToggleActive,
        rangeSelectActive,
        shouldPrimeSelection: !multiSelectToggleActive && !rangeSelectActive,
    };
}
