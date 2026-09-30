/**
 * 手绘单周期编辑器的纯逻辑。
 *
 * 【为什么单独成文件】画笔落点、笔画补间、平滑与"从当前波形起步"都是可单测的
 * 算术；混在画布里就只能靠手感回归 —— 而"画不上 / 画歪了 / 首尾接不上"这类缺陷
 * 都不会抛错。
 *
 * 【首尾相接由谁保证】渲染端的 `sampleCycle` 把表当作**循环**（见 `vibratoCycle.ts`
 * 的 `tableAt`），因此编辑器只需把表画成一个闭合周期即可，不需要额外约定。
 */

import { CYCLE_TABLE_DEFAULT_LEN, sampleCycle } from "../../../features/vibrato/vibratoCycle";
import type { CycleSource } from "../../../features/vibrato/vibratoTypes";

/** 手绘表的默认格数（与提取路径同一长度，便于互换）。 */
export const CYCLE_EDIT_BINS = CYCLE_TABLE_DEFAULT_LEN;

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

function clampValue(value: number): number {
    return clamp(Number.isFinite(value) ? value : 0, -1, 1);
}

/**
 * 把任意周期来源采样成定长表 —— 手绘编辑器的初值。
 *
 * 【为什么不是空白】从当前波形起步，用户改的是"形状"而不是"从零画一条曲线"，
 * 顺手得多（也避免手绘一进入就把预设的波形丢光）。
 */
export function tableFromCycle(source: CycleSource, bins: number = CYCLE_EDIT_BINS): number[] {
    const n = clamp(Math.round(bins), 8, 256);
    const out = new Array<number>(n);
    for (let i = 0; i < n; i += 1) out[i] = clampValue(sampleCycle(source, i / n));
    return out;
}

/** 单点落笔：把一格设成 `value`（钳到 `[-1,1]`）。 */
export function paintCycleBin(table: readonly number[], bin: number, value: number): number[] {
    const n = table.length;
    if (n === 0) return [];
    const index = clamp(Math.round(bin), 0, n - 1);
    const out = table.slice();
    out[index] = clampValue(value);
    return out;
}

/**
 * 一次笔画：在 `from` 与 `to` 两个落点之间按线性补间填满每一格。
 *
 * 【为什么需要补间】快速划动时两次采样可能隔开好几格，只画端点会留下断线。
 * 指针事件还会用 `getCoalescedEvents` 进一步加密，但补间是兜底且更便宜。
 */
export function paintCycleSegment(
    table: readonly number[],
    from: { bin: number; value: number },
    to: { bin: number; value: number },
): number[] {
    const n = table.length;
    if (n === 0) return [];
    const a = clamp(Math.round(from.bin), 0, n - 1);
    const b = clamp(Math.round(to.bin), 0, n - 1);
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    const startValue = a === lo ? from.value : to.value;
    const endValue = a === lo ? to.value : from.value;
    const span = hi - lo;
    const out = table.slice();
    for (let i = lo; i <= hi; i += 1) {
        const t = span === 0 ? 0 : (i - lo) / span;
        out[i] = clampValue(startValue + (endValue - startValue) * t);
    }
    return out;
}

/**
 * 三点循环滑动平均（可连点）。
 *
 * 【为什么循环】表本身是首尾相接的周期，端点若不参与邻居平均，接缝处会出现
 * 一道突兀的折角 —— 正是"首尾相接"承诺要避免的东西。
 */
export function smoothCycleTable(table: readonly number[]): number[] {
    const n = table.length;
    if (n === 0) return [];
    const out = new Array<number>(n);
    for (let i = 0; i < n; i += 1) {
        const prev = table[(i - 1 + n) % n];
        const current = table[i];
        const next = table[(i + 1) % n];
        out[i] = clampValue((prev + current + next) / 3);
    }
    return out;
}

/** 画布坐标 →（格号, 值）。顶端是 `+1`，底端是 `-1`。 */
export function cycleEditorPoint(
    x: number,
    y: number,
    width: number,
    height: number,
    bins: number,
): { bin: number; value: number } {
    const n = Math.max(1, Math.round(bins));
    const w = width > 0 ? width : 1;
    const h = height > 0 ? height : 1;
    return {
        bin: clamp(Math.floor((x / w) * n), 0, n - 1),
        value: clampValue(1 - (y / h) * 2),
    };
}
