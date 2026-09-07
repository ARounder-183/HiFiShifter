/**
 * paramShiftActions.ts — 参数线平移快捷键的 actionId → 执行参数解析。
 *
 * 12 个平移快捷键（音频块范围 / 选择范围 × 上 / 下 × 默认 / 大幅 / 微调）
 * 共用一套 id 命名规律，但**不能用 endsWith 判方向**：变体 id 以
 * "Large" / "Small" 结尾（如 pianoRoll.shiftParamDownLarge），必须用
 * includes("Down") 判定下移。集中在这里并提供单测，避免内联解析出错。
 */

import type { ParamShiftMagnitude } from "../../components/layout/pianoRoll/paramShiftStep";
import type { ActionId } from "./types";

/** 音频块（clip）范围平移：整条参数线随选中 Clip 平移（App 内执行）。 */
const CLIP_SHIFT_IDS: ReadonlySet<ActionId> = new Set<ActionId>([
    "pianoRoll.shiftParamUp",
    "pianoRoll.shiftParamDown",
    "pianoRoll.shiftParamUpLarge",
    "pianoRoll.shiftParamDownLarge",
    "pianoRoll.shiftParamUpSmall",
    "pianoRoll.shiftParamDownSmall",
]);

/** 选择范围平移：仅选区内平移（PianoRollPanel 经 hifi:editOp 执行）。 */
const SELECTION_SHIFT_IDS: ReadonlySet<ActionId> = new Set<ActionId>([
    "pianoRoll.shiftParamUpSelection",
    "pianoRoll.shiftParamDownSelection",
    "pianoRoll.shiftParamUpSelectionLarge",
    "pianoRoll.shiftParamDownSelectionLarge",
    "pianoRoll.shiftParamUpSelectionSmall",
    "pianoRoll.shiftParamDownSelectionSmall",
]);

export interface ParamShiftIntent {
    /** 是否向上移动。 */
    isUp: boolean;
    /** 变化幅度档位（决定每拍步长，见 getParamShiftStep）。 */
    magnitude: ParamShiftMagnitude;
    /**
     * 选择范围平移的 op 名（hifi:editOp 事件契约）；音频块范围平移为 null
     * （直接在 App 内执行，不经事件通道）。
     */
    selectionOp: "shiftParamUpSelection" | "shiftParamDownSelection" | null;
}

/**
 * 解析参数线平移快捷键的执行参数；非平移类 actionId 返回 null。
 */
export function resolveParamShiftIntent(actionId: ActionId): ParamShiftIntent | null {
    const isSelection = SELECTION_SHIFT_IDS.has(actionId);
    if (!isSelection && !CLIP_SHIFT_IDS.has(actionId)) {
        return null;
    }
    const isUp = !actionId.includes("Down");
    const magnitude: ParamShiftMagnitude = actionId.endsWith("Large")
        ? "coarse"
        : actionId.endsWith("Small")
          ? "fine"
          : "normal";
    const selectionOp = isSelection
        ? isUp
            ? "shiftParamUpSelection"
            : "shiftParamDownSelection"
        : null;
    return { isUp, magnitude, selectionOp };
}
