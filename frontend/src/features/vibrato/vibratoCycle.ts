/**
 * 归一化周期波形的采样。
 *
 * 所有形状都满足「约定同一起点」：`u = 0` 位于**上升的零交叉**（方波类除外，
 * 它们没有零交叉概念，`u = 0` 位于高电平起点）。这样 `startPhaseDeg = 0`
 * 对所有形状都是同一个语义，切换形状时听感起点不跳。
 *
 * `skew ∈ (0,1)` 的含义随形状变化（默认 0.5）：
 * - `triangle` 上升段占比
 * - `square` / `trapezoid` / `trill` 高电平占空比
 * - `sine` / `sawUp` / `sawDown` / `table` 不使用
 */

import type { CycleSource, WaveShape } from "./vibratoTypes";

/** `table` 采样表的长度下限 / 上限，以及新建手绘表时的默认长度。 */
export const CYCLE_TABLE_MIN_LEN = 8;
export const CYCLE_TABLE_MAX_LEN = 256;
export const CYCLE_TABLE_DEFAULT_LEN = 64;

/** 梯形 / 颤指的过渡边宽（占周期的比例）。 */
export const TRAPEZOID_EDGE = 0.12;
export const TRILL_EDGE = 0.05;

/** 把任意实数折进 `[0,1)`。 */
export function wrap01(u: number): number {
    if (!Number.isFinite(u)) return 0;
    const w = u % 1;
    return w < 0 ? w + 1 : w;
}

function clampSkew(skew: number): number {
    if (!Number.isFinite(skew)) return 0.5;
    return Math.min(0.98, Math.max(0.02, skew));
}

/** 分段线性「三角波」，`skew` 为上升段占比。值域 `[-1,1]`，在 `u=0` 处为 `-1`。 */
function triangleAt(u: number, skew: number): number {
    const s = clampSkew(skew);
    return u < s ? -1 + (2 * u) / s : 1 - (2 * (u - s)) / (1 - s);
}

/** 梯形 / 颤指：`duty` 为高电平占空比，`edge` 为过渡边宽。 */
function trapezoidAt(u: number, duty: number, edge: number): number {
    const d = clampSkew(duty);
    const e = Math.min(edge, d / 2, (1 - d) / 2);
    if (e <= 1e-6) return u < d ? 1 : -1;
    if (u < e) return u / e;
    if (u < d - e) return 1;
    if (u < d + e) return 1 - (u - (d - e)) / (2 * e);
    if (u < 1 - e) return -1;
    return -1 + (u - (1 - e)) / e;
}

function shapeAt(shape: WaveShape, skew: number, u: number): number {
    switch (shape) {
        case "sine":
            return Math.sin(2 * Math.PI * u);
        case "triangle":
            // 平移半个上升段，使 u=0 落在上升边的中点（= 零交叉且上升）。
            return triangleAt(wrap01(u + clampSkew(skew) / 2), skew);
        case "sawUp": {
            // 平移半个周期，使 u=0 位于上升段过零处。
            const phase = wrap01(u + 0.5);
            return 2 * phase - 1;
        }
        case "sawDown": {
            const phase = wrap01(u + 0.5);
            return 1 - 2 * phase;
        }
        case "square":
            return u < clampSkew(skew) ? 1 : -1;
        case "trapezoid":
            return trapezoidAt(u, skew, TRAPEZOID_EDGE);
        case "trill":
            return trapezoidAt(u, skew, TRILL_EDGE);
        default:
            return Math.sin(2 * Math.PI * u);
    }
}

/** 采样表插值：`table` 视为首尾相接的周期，线性插值。 */
function tableAt(table: readonly number[], u: number): number {
    const n = table.length;
    if (n === 0) return 0;
    if (n === 1) return Number.isFinite(table[0]) ? table[0] : 0;
    const pos = wrap01(u) * n;
    const i0 = Math.floor(pos) % n;
    const i1 = (i0 + 1) % n;
    const frac = pos - Math.floor(pos);
    const a = Number.isFinite(table[i0]) ? table[i0] : 0;
    const b = Number.isFinite(table[i1]) ? table[i1] : 0;
    return a + (b - a) * frac;
}

/**
 * 采样一个周期的波形。
 *
 * @param source 周期波形来源。
 * @param u 归一化相位，任意实数（内部取模）。
 * @returns `[-1,1]` 内的波形值（`table` 来源不强制钳制，但会过滤非有限值）。
 */
export function sampleCycle(source: CycleSource, u: number): number {
    const value =
        source.kind === "table"
            ? tableAt(source.table, u)
            : shapeAt(source.shape, source.skew, wrap01(u));
    return Number.isFinite(value) ? value : 0;
}

/** 波形是否使用 `skew`：决定预设编辑器里是否显示偏斜滑杆。 */
export function shapeUsesSkew(shape: WaveShape): boolean {
    return shape === "triangle" || shape === "square" || shape === "trapezoid" || shape === "trill";
}

/** 生成长度为 `len` 的正弦周期表（手绘编辑器的初值）。 */
export function makeSineTable(len: number = CYCLE_TABLE_DEFAULT_LEN): number[] {
    const n = Math.max(CYCLE_TABLE_MIN_LEN, Math.min(CYCLE_TABLE_MAX_LEN, Math.round(len)));
    const out = new Array<number>(n);
    for (let i = 0; i < n; i += 1) out[i] = Math.sin((2 * Math.PI * i) / n);
    return out;
}

/** 把任意表规整为合法周期表：定长、有限值、钳到 `[-1,1]`。 */
export function normalizeCycleTable(table: unknown): number[] {
    if (!Array.isArray(table) || table.length < CYCLE_TABLE_MIN_LEN) return makeSineTable();
    const n = Math.min(CYCLE_TABLE_MAX_LEN, table.length);
    const out = new Array<number>(n);
    for (let i = 0; i < n; i += 1) {
        const raw = Number(table[i]);
        out[i] = Number.isFinite(raw) ? Math.min(1, Math.max(-1, raw)) : 0;
    }
    return out;
}
