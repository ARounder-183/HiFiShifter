/**
 * 时间轴渲染内核 · 滚轮缩放意图判定
 *
 * 【主要内容】
 * 把滚轮的双轴增量解析为**单调、无抖动**的缩放进 / 出意图：
 * - `resolveWheelZoomStep()` 用「主轴 + 死区累积」判定单次事件是否产生一步缩放；
 * - `WHEEL_ZOOM_DEADZONE_PX` 给出死区（CSS px）。
 *
 * 【作用：为什么不能只看 deltaY 的符号】
 * 旧实现是 `factor = deltaY < 0 ? 1.1 : 0.9`，这条规则有两个缺陷，两者都会
 * 在 Windows 上表现为**剧烈抖动**（实测：行高在同一位置 `88 → 80 → 88 → 80`
 * 往复，见下）：
 *
 * 1. **`deltaY = 0` 被判为缩小**。`0 < 0` 为假 → 取 `0.9`。而 `deltaY = 0`
 *    恰恰是「纯横向手势」（触摸板横滑）与「纵向噪声归零」的常见取值，于是
 *    一个本该缩放的横向手势被稳定地判成**缩小**。
 * 2. **没有任何幅度门限**。precision touchpad 的 ctrl+捏合会派发**小幅、
 *    逐帧变号**的增量（实测 ±0.4 ~ ±3）。逐事件取符号 ⇒ 方向逐事件翻转 ⇒
 *    缩放来回抵消，画面表现为抖动而非缩放。
 *
 * 修正分两层，且都必须有：
 * - **主轴选择**：取 `|deltaX|` 与 `|deltaY|` 的较大者定方向。纯横向手势因此
 *   按横向自身的符号缩放（单调），不再被"deltaY = 0 → 缩小"劫持。
 * - **死区 + 累积**：小于死区的增量先累积、不立即出手；**反向输入先清零累积**
 *   （真实反向立刻生效），因此对称噪声永远累积不到死区 ⇒ 完全不缩放（稳定），
 *   而真实滚轮（一格 ±100）远超死区 ⇒ 每格恰好一步。
 *
 * 【与其他模块的关系】
 * - 上游：`host/timelineKernelHost` 的 `onWheel` 每帧调用，把结果映射为
 *   `WHEEL_ZOOM_IN_FACTOR` / `WHEEL_ZOOM_OUT_FACTOR`。
 * - 下游：竖直缩放改行高（`onRowHeightChange`），水平缩放走
 *   `resolveHorizontalWheelZoom`。
 * - 独立性：纯函数 + 显式状态对象，不依赖 DOM / React / 真实 `WheelEvent`，
 *   因此可在 node 环境的 vitest 中直接驱动（本缺陷的回归测试正是靠它）。
 *
 * 【维护说明】方向约定与旧实现一致：**负增量 = 放大**（滚轮向上）。
 * `direction: -1` 放大、`1` 缩小、`0` 不变。
 */

/**
 * 缩放判定的死区（CSS px）。
 *
 * 取值权衡（实测增量分布）：
 * - 真实鼠标滚轮一格为 ±100 / ±120，触摸板捏合为 ±1 ~ ±10，**均远超死区**
 *   ——死区不会让正常缩放"变钝"；
 * - precision touchpad 的变号噪声实测 ±0.4 ~ ±3，**均低于死区**——这是它要挡住
 *   的目标区间。取 8 而非更小值，是因为噪声上界实测可达 ±3，取 4 时 `-3` 与
 *   `+4` 仍能凑出一次触发（见下方单测的噪声用例）。
 */
export const WHEEL_ZOOM_DEADZONE_PX = 8;

/** 缩放方向：-1 = 放大、1 = 缩小、0 = 本次事件不产生缩放。 */
export type WheelZoomDirection = -1 | 0 | 1;

/**
 * 滚轮缩放的累积状态。
 *
 * 特殊说明：这是**显式传入传出**的普通对象（不是模块级单例），宿主持有它、
 * 单测直接构造它——隐藏的模块级状态会让测试之间互相污染（本仓库的
 * `timelineViewportSync` 就吃过这个亏）。
 */
export interface WheelZoomAccumulator {
    /** 尚未达到死区的累积增量（CSS px，符号与方向约定一致）。 */
    pending: number;
}

/** 新建一个归零的累积状态。 */
export function createWheelZoomAccumulator(): WheelZoomAccumulator {
    return { pending: 0 };
}

/**
 * 读取本次事件用于定方向的**主轴增量**。
 *
 * 流程：取两轴绝对值，较大者胜出；完全相等或均为 0 时回退 `deltaY`
 * （鼠标滚轮只有 deltaY，回退它保持既有手感）。
 *
 * 特殊说明：只做「谁主导」的选择，**不做任何缩放**——这里返回的原始增量随后
 * 进入死区累积，因此"横向手势被稳定判成缩小"这类缺陷不会再出现。
 *
 * @param deltaX 水平增量（CSS px）。
 * @param deltaY 竖直增量（CSS px）。
 * @returns 主轴增量（可能是 0）。
 */
function dominantAxisDelta(deltaX: number, deltaY: number): number {
    const ax = Number.isFinite(deltaX) ? Math.abs(deltaX) : 0;
    const ay = Number.isFinite(deltaY) ? Math.abs(deltaY) : 0;
    if (ax > ay) return deltaX;
    return Number.isFinite(deltaY) ? deltaY : 0;
}

/**
 * 判定一次滚轮事件是否产生缩放，并推进累积状态。
 *
 * 流程：
 * 1. 取主轴增量（`dominantAxisDelta`）；非有限值归零；
 * 2. **反向先清零**：主轴增量与累积量符号相反时把累积量重置为 0
 *    ——真实反向必须立刻生效，不能让上一次手势的残余把反向吃掉；
 * 3. 累积；越过死区则产出一个方向（**饱和**在死区处，使真实滚轮"一格一步"，
 *    不会因单次大增量攒出多步）。
 *
 * 特殊说明 1：**不修改入参**，返回新的累积状态（调用方负责替换）。
 * 特殊说明 2：饱和而非「减去死区」是刻意的——减去会让一个 ±100 的滚轮格
 * 留下约 ±92 的余额，此后**任何一次微小噪声事件都会立刻触发一整步缩放**，
 * 正是抖动的一个来源。
 *
 * @param args.accumulator 上一次的累积状态。
 * @param args.deltaX 水平增量（CSS px）。
 * @param args.deltaY 竖直增量（CSS px）。
 * @param args.deadzonePx 死区（缺省 `WHEEL_ZOOM_DEADZONE_PX`；非法值回退缺省）。
 * @returns 本次的缩放方向与推进后的累积状态。
 */
export function resolveWheelZoomStep(args: {
    accumulator: WheelZoomAccumulator;
    deltaX: number;
    deltaY: number;
    deadzonePx?: number;
}): { direction: WheelZoomDirection; accumulator: WheelZoomAccumulator } {
    const deadzone =
        Number.isFinite(args.deadzonePx) && (args.deadzonePx as number) > 0
            ? (args.deadzonePx as number)
            : WHEEL_ZOOM_DEADZONE_PX;
    const previous = Number.isFinite(args.accumulator.pending) ? args.accumulator.pending : 0;
    const raw = dominantAxisDelta(args.deltaX, args.deltaY);
    const axisDelta = Number.isFinite(raw) ? raw : 0;

    // 反向输入：丢弃上一次的残余，从 0 开始累积（真实反向立刻生效）。
    const base = previous !== 0 && axisDelta !== 0 && previous > 0 !== axisDelta > 0 ? 0 : previous;
    const pending = base + axisDelta;

    if (pending >= deadzone) {
        return { direction: 1, accumulator: { pending: deadzone } };
    }
    if (pending <= -deadzone) {
        return { direction: -1, accumulator: { pending: -deadzone } };
    }
    return { direction: 0, accumulator: { pending } };
}
