/**
 * fadeShapeCycle.ts — 淡变形状循环的修饰键判定。
 *
 * 【主要内容】
 * 判定一次指针事件是否满足「循环切换淡变形状」的修饰键条件。
 *
 * 【为什么单独成模块】
 * 旧实现里这段逻辑有**两份完全相同的拷贝**（`OverlapEditLayer` 与 `FadeHitLayer`），
 * 内核又需要第三份。三份同样的判定只要有一份被改动，就会出现「同一个按键在某些
 * 淡变控件上生效、在另一些上不生效」这种极难归因的问题。
 *
 * 【判定规则】
 * 绑定的修饰键可以来自两处，取**并集**：
 * - `modifierOnly === true && key === "control" | "alt" | "shift"`：把「单键」当作
 *   修饰键（例如只按 Ctrl 就触发，不需要组合）；
 * - `ctrl` / `alt` / `shift` 布尔字段：常规组合修饰键。
 * 未要求的修饰键**不参与判定**——多按一个键不应让循环失效（否则用户在
 * Ctrl+Shift 习惯下会以为功能坏了）。
 *
 * 【macOS 注意】
 * `ctrl` 字段在 mac 上按「主修饰键」语义处理，因此判定里 `ctrlKey` 与 `metaKey`
 * 是**或**关系（⌘ 与 Ctrl 都算命中）。
 */

import type { Keybinding } from "../../../features/keybindings/types";

/** 判定所需的事件字段（结构化子集：PointerEvent / React 合成事件都能直接传入）。 */
export interface FadeShapeCycleEventLike {
    readonly ctrlKey: boolean;
    readonly metaKey: boolean;
    readonly altKey: boolean;
    readonly shiftKey: boolean;
}

/**
 * 判定该次指针事件是否满足淡变形状循环的修饰键条件。
 *
 * @param kb 键位绑定（`modifier.fadeShapeCycleClick`）；缺省视为不触发。
 * @param event 指针事件（或其修饰键字段子集）。
 * @returns 满足条件时为 true。
 */
export function isFadeShapeCycleModifierHeld(
    kb: Keybinding | null | undefined,
    event: FadeShapeCycleEventLike,
): boolean {
    if (kb == null) return false;
    const modifierOnly = kb.modifierOnly === true;
    const requiredCtrl = modifierOnly && kb.key === "control" ? true : Boolean(kb.ctrl);
    const requiredAlt = modifierOnly && kb.key === "alt" ? true : Boolean(kb.alt);
    const requiredShift = modifierOnly && kb.key === "shift" ? true : Boolean(kb.shift);
    return (
        (!requiredCtrl || event.ctrlKey || event.metaKey) &&
        (!requiredAlt || event.altKey) &&
        (!requiredShift || event.shiftKey)
    );
}
