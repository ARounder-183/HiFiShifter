// 动作派发上下文：**谁**在派发这条动作。
//
// 【它解决的是什么】"长按重复"（`holdRepeat`）此前只由 `keyup` / `blur` 终止，
// 而移动端抽屉、菜单、独立窗口转发走的是**同一条** `handleKeybindingAction` ——
// 于是"点一下撤销"会在 400ms 后变成"每 50ms 撤一次，永不停止"。用户报的
// "撤销/重做几乎不可用"，一半来自这里：一次点击撤掉了整个工程的历史。
//
// 【为什么不给 `beginHoldRepeat` 加超时兜底】兜底时长只能猜。用户真的按住
// Ctrl+Z 三秒时，兜底会把长按切断；而合成派发根本没有"按住"这回事。把
// 「有没有键可按」这件事**显式建模**，是唯一不会误伤的解法。
//
// 【默认值为什么是键盘】所有既有调用点（`useKeybindings` 的真键盘、独立窗口
// 转发）都保持原行为，改动只落在**合成派发**那几处 —— 因此桌面零回归。

/** 谁发起的这次动作。 */
export type ActionDispatchSource = "keyboard" | "menu" | "mobile" | "plugin" | "shortcut";

export interface ActionDispatchContext {
    /** 发起方。只用于诊断与规则判定，不参与动作本身的路由。 */
    source: ActionDispatchSource;
    /**
     * 是否允许布防长按重复。
     *
     * **只有真键盘为 true**：长按重复的终止依赖 `keyup`，而合成派发永远不会
     * 有 `keyup`。
     */
    repeat: boolean;
}

/** 真键盘：允许长按重复。 */
export const KEYBOARD_DISPATCH: ActionDispatchContext = { source: "keyboard", repeat: true };

/**
 * 合成派发（菜单 / 抽屉 / 快捷面板 / 独立窗口转发）：**永不**布防长按重复。
 *
 * 一次点击就是一次动作。
 */
export const SYNTHETIC_DISPATCH: ActionDispatchContext = { source: "menu", repeat: false };

/** 移动端抽屉 / 工具栏按钮。 */
export const MOBILE_DISPATCH: ActionDispatchContext = { source: "mobile", repeat: false };

/** 缺省上下文（老调用点不传时用）：与改动前逐字节相同。 */
export const DEFAULT_DISPATCH: ActionDispatchContext = KEYBOARD_DISPATCH;

/** 判断某次派发是否应当布防长按重复（缺省上下文 = 键盘 = 布防）。 */
export function shouldArmHoldRepeat(ctx: ActionDispatchContext | undefined): boolean {
    return (ctx ?? DEFAULT_DISPATCH).repeat === true;
}
