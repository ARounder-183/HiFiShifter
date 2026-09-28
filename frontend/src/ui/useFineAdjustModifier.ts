/**
 * 精细调整修饰键。
 *
 * 【为什么抽成一处】"按住 Ctrl 滚轮 = 更细一档"是全应用的约定，此前每个控件
 * 各自 `useAppSelector((s) => selectKeybinding(s, "modifier.paramFineAdjust"))`
 * 再自己判 `isModifierActive`，于是有的控件漏了、有的判错。收进一个 hook 后
 * 能力层（`AppNumberField` / `AppSlider`）内建它，调用方不需要知道有这回事。
 *
 * 【为什么放在 `src/ui` 而不是让它读 Redux】`src/ui` 是设计系统层，但它本来就
 * 与这个应用共生（不是通用组件库），读一个键位绑定是可以接受的 —— 换来的是
 * "精细调整无法被忘记"。
 */
import { useAppSelector } from "../app/hooks";
import { isModifierActive, selectKeybinding } from "../features/keybindings/keybindingsSlice";
import { FINE_ADJUST_ACTION_ID } from "./stepPolicy";

/** 供滚轮事件判定用的事件形状（原生与 React 事件都满足）。 */
export interface ModifierEventLike {
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
    metaKey: boolean;
}

/**
 * 返回一个判定函数：给定滚轮事件，是否应按精细步长处理。
 *
 * 判定函数身份稳定（由 `useAppSelector` 的返回值决定），可直接放进
 * `useNonPassiveWheel` 的回调里。
 */
export function useFineAdjustModifier(): (event: ModifierEventLike) => boolean {
    const binding = useAppSelector((state) => selectKeybinding(state, FINE_ADJUST_ACTION_ID));
    return (event: ModifierEventLike) => isModifierActive(binding, event);
}
