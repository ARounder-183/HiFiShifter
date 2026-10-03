/**
 * 快捷键设置的共享判定与常量 —— 行组件与面板都需要的、不含 JSX 的那部分。
 *
 * 【为什么单独成文件】`KeybindingsActionRow.tsx` 需要导出组件以外的常量与函数，
 * 而 `react-refresh/only-export-components` 规则要求「一个文件只导出组件」——
 * 否则该文件的热更新会退化成整页刷新。把非组件的部分挪到这里满足规则，
 * 顺带也让 `isDefaultBinding` 可以被搜索引擎之外的调用方复用。
 */
import {
    GROUP_LABEL_KEYS,
    GROUP_NAV_LABEL_KEYS,
} from "../../../features/keybindings/defaultKeybindings";
import type { ActionMeta, Keybinding } from "../../../features/keybindings/types";
import type { AppStatusTone } from "../../../ui";

/**
 * 取一个分组在**窄位**（导航栏、搜索结果路标）里用的标签。
 *
 * 【为什么要回落】`GROUP_NAV_LABEL_KEYS` 是**部分**映射 —— 只有冗长到会溢出的分组
 * 才有短版，其余（"编辑" / "布局" / "钢琴卷帘"）短版会和长版重复。缺哪一组就
 * 用长版，于是这张表不必覆盖全部 14 组，将来加语系时也不用补满。
 *
 * 【为什么按"键是否存在"判断而不是按文案长度】文案长度要到运行时才知道，而键的
 * 有无在词典里是静态事实 —— 回落规则保持静态，行为才可预测。
 */
export function resolveGroupNavLabel(
    group: ActionMeta["group"],
    tf: (key: string) => string,
): string {
    const navKey = GROUP_NAV_LABEL_KEYS[group];
    // `tf` 查不到时返回键名本身，因此这里比对键名即可判断是否真的注册了短版。
    if (navKey) {
        const resolved = tf(navKey);
        if (resolved !== navKey) return resolved;
    }
    return tf(GROUP_LABEL_KEYS[group]);
}

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
