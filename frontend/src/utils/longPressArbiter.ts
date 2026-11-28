/**
 * 长按仲裁器：一个时刻只能有一个长按消费者。
 *
 * 【为什么必须有它】触摸上没有"右键"，于是三件事都想抢同一段"按住不放"：
 * 1. **上下文菜单**（合成 `contextmenu`）—— 136 个 `onContextMenu` 处理器靠它复活；
 * 2. **拖动重排**（按住轨道头往上拖）—— 它的起始姿势与长按完全相同；
 * 3. **双指手势的起点**（第一根手指落下后 100ms 内落下第二根）。
 *
 * 三者同时生效的症状不是崩溃，而是**动作打架**：用户想拖轨道，结果弹出了菜单，
 * 而菜单弹出后拖动仍在继续，松手时轨道跳到了别处。这类故障极难归因，因为它只在
 * "手抖了一下"时出现。
 *
 * 因此：谁先到谁声明，其余一律放弃。声明是**单槽**的 —— 不需要计数，因为触摸上
 * 同时只可能有一个手指主导（第二根手指落下即取消第一根的长按，见 `useLongPress`）。
 */

/** 长按的消费方。 */
export type LongPressClaim = "menu" | "drag" | "gesture";

let owner: LongPressClaim | null = null;

/**
 * 尝试声明长按的所有权。
 *
 * @returns `true` = 声明成功（或本来就是自己），`false` = 已被别人占用。
 */
export function claimLongPress(kind: LongPressClaim): boolean {
    if (owner === null || owner === kind) {
        owner = kind;
        return true;
    }
    return false;
}

/** 释放所有权。只释放自己持有的那一份（别人的声明不受影响）。 */
export function releaseLongPress(kind: LongPressClaim): void {
    if (owner === kind) owner = null;
}

/** 当前持有者（诊断 / 测试用）。 */
export function longPressOwner(): LongPressClaim | null {
    return owner;
}

/**
 * 强制清空。
 *
 * 【为什么需要它】手势可能被系统打断（来电、通知下拉、WebView 崩溃恢复），那时
 * `pointerup` 永远不会到达。没有这个出口，仲裁器会永久停在"被占用"，此后所有
 * 长按都静默失效 —— 而这正是"用了一会儿之后右键菜单就不出来了"的成因。
 */
export function resetLongPressArbiter(): void {
    owner = null;
}
