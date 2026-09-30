/**
 * 快捷键设置的共享判定与常量 —— 行组件与面板都需要的、不含 JSX 的那部分。
 *
 * 【为什么单独成文件】`KeybindingsActionRow.tsx` 需要导出组件以外的常量与函数，
 * 而 `react-refresh/only-export-components` 规则要求「一个文件只导出组件」——
 * 否则该文件的热更新会退化成整页刷新。把非组件的部分挪到这里满足规则，
 * 顺带也让 `isDefaultBinding` 可以被搜索引擎之外的调用方复用。
 */
import type { ActionMeta, Keybinding } from "../../../features/keybindings/types";
import type { AppStatusTone } from "../../../ui";

/** 修饰键手势徽章的 i18n key 与色调（四档互不相同，仍可区分手势类型）。 */
export const GESTURE_BADGES: Record<
    NonNullable<ActionMeta["modifierOperationType"]>,
    { labelKey: string; tone: AppStatusTone }
> = {
    drag: { labelKey: "kb_gesture_drag", tone: "accent" },
    click: { labelKey: "kb_gesture_click", tone: "warning" },
    wheel: { labelKey: "kb_gesture_wheel", tone: "success" },
    hold: { labelKey: "kb_gesture_hold", tone: "neutral" },
};

/**
 * 判断一条绑定是否与默认一致。
 *
 * 【为什么要 `Boolean()` 包一层】`Keybinding` 的三个修饰键字段是**可选布尔**，
 * 缺失与 `false` 语义相同。直接 `===` 比较会把 `{key:"z",ctrl:true}` 与
 * `{key:"z",ctrl:true,shift:false}` 判成不同 —— 而后者是规范化后会写出的形态。
 */
export function isDefaultBinding(current: Keybinding, fallback: Keybinding): boolean {
    return (
        current.key === fallback.key &&
        Boolean(current.ctrl) === Boolean(fallback.ctrl) &&
        Boolean(current.shift) === Boolean(fallback.shift) &&
        Boolean(current.alt) === Boolean(fallback.alt) &&
        Boolean(current.modifierOnly) === Boolean(fallback.modifierOnly)
    );
}
