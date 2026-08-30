/**
 * focusRouting.ts — 快捷键的焦点感知路由（冲突解决核心）
 *
 * 设计目标：不同"焦点域 / 作用域"的快捷键可以善意地共用同一组按键，
 * 由当前焦点决定期望的编辑目标 —— 与复制/粘贴（clip.copy 与
 * pianoRoll.copy 共绑 Ctrl+C，参数编辑器焦点优先参数粘贴）同一套思路。
 *
 * 匹配顺序（同一按键事件命中多个绑定时）：
 *  0. 当前焦点域内激活的作用域操作（如参数编辑器内 select 工具下的
 *     paramEditorSelect 操作；轨道头内的 trackHeaderFocus 操作）—— 最高优先。
 *  1. 全局（无 scopedContext）操作 —— 例如 track.add / clip.* / edit.*。
 *  2. 其它焦点域的作用域操作 —— 仅在没有更优匹配时兜底（例如在时间轴
 *     按 Ctrl+D，轨道头专用的 track.clone 仍是唯一匹配时也照常生效）。
 *  Infinity. 硬排除：当前焦点域下明确不应响应的作用域
 *     （paramEditorSelect 仅在参数编辑器 + select 工具下响应；
 *     quickSearch / pianoRollVibratoDrag 由各自的弹窗与拖拽表面接管）。
 *
 * 这样两个绑定键值相同但作用域不同的操作不再互相"抢占"，也不会被
 * 赋值时的冲突检测拦截（见 keybindingsSlice.findConflicts 的作用域规则）。
 */

import type { ActionId, Keybinding } from "./types";
import { ACTION_META } from "./defaultKeybindings";
import { matchesKeybinding } from "./keybindingMatch";

/** 全局快捷键路由关心的焦点域。 */
export type KeybindingFocusDomain = "pianoRoll" | "timeline" | "trackHeader" | null;

/** 硬排除的作用域：即使按键唯一匹配也绝不由此全局路由触发。 */
const HARD_EXCLUDED_CONTEXTS = new Set(["quickSearch", "pianoRollVibratoDrag"]);

/**
 * 计算作用域在当前焦点域下的优先级（越小越优先）。
 * - 0：当前焦点域内激活的作用域操作
 * - 1：全局（无作用域）
 * - 2：其它焦点域的作用域操作（兜底）
 * - Infinity：硬排除（不参与匹配）
 */
export function scopePriority(
    scopedContext: string | undefined,
    domain: KeybindingFocusDomain,
    toolMode: string,
): number {
    if (!scopedContext) return 1;
    switch (scopedContext) {
        case "paramEditorSelect":
            // 参数编辑器操作只在钢琴卷帘 + 选择工具下生效；其它焦点域硬排除。
            return domain === "pianoRoll" && toolMode === "select" ? 0 : Infinity;
        case "timelineFocus":
            return domain === "timeline" ? 0 : 2;
        case "trackHeaderFocus":
            return domain === "trackHeader" ? 0 : 2;
        default:
            return HARD_EXCLUDED_CONTEXTS.has(scopedContext) ? Infinity : 2;
    }
}

export const SCOPE_PRIORITY_MAX = Infinity;

/**
 * 在键位映射中查找匹配当前按键事件的最佳 actionId。
 *
 * 该函数替代旧的"按对象插入顺序取第一个匹配"逻辑：改为先收集全部
 * 匹配项，再按「激活作用域 > 全局 > 其它作用域 > 硬排除」排序取首位，
 * 使路由结果与键位表声明顺序无关，键值相同的作用域操作由焦点裁决
 * （与复制/粘贴的焦点分发同构）。
 *
 * @param domain 当前焦点域（pianoRoll / timeline / trackHeader / null）
 * @param toolMode 当前工具（select / draw / vibrato），用于 paramEditorSelect 激活判定
 * @param options.hardExcludeParamEditorSelect 参数编辑器作用域是否硬排除
 *   （默认按 scopePriority 的规则自动判定；无需显式传参）
 */
export function resolveActionByFocus(
    e: KeyboardEvent,
    keybindings: Readonly<Record<string, Keybinding>>,
    domain: KeybindingFocusDomain,
    toolMode: string,
): ActionId | null {
    let best: ActionId | null = null;
    let bestPriority = Infinity;
    for (const [actionId, kb] of Object.entries(keybindings) as [ActionId, Keybinding][]) {
        if (kb.modifierOnly) continue;
        if (!matchesKeybinding(e, kb)) continue;
        const priority = scopePriority(ACTION_META[actionId]?.scopedContext, domain, toolMode);
        if (priority < bestPriority) {
            bestPriority = priority;
            best = actionId;
        }
    }
    return bestPriority === Infinity ? null : best;
}

/**
 * 判断某作用域是否在给定焦点域下"激活"（用于测试与提示文案）。
 */
export function isScopeActive(
    scopedContext: string | undefined,
    domain: KeybindingFocusDomain,
    toolMode: string,
): boolean {
    return scopePriority(scopedContext, domain, toolMode) === 0;
}

/** 硬排除是否包含需要被排除的作用域（仅用于单测断言）。 */
export { HARD_EXCLUDED_CONTEXTS };
