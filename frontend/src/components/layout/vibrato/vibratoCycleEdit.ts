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

/**
 * 纵轴上下各留的内缩量（CSS 像素）。
 *
 * 【为什么必须与绘制共用】值 `±1` 若正好落在画布上下边缘，1.5px 的描边会被裁掉
 * 一半，所以绘制把 `±1` 放在离边缘 `INSET` 处。命中测试曾经用整幅高度换算
 * （`value = 1 - (y/h) * 2`），于是"画出来的峰顶"与"能画到 1.0 的那一行"相差
 * 约 6.7% —— 用户在看到的峰顶落笔，写进去的却只有 `0.93`。整体缩放的手势映射
 * （`cycleRightDragTransform`）以这里为基准，偏差会被放大成"拖满一屏不等于
 * 一个八度"，所以两边必须共用同一个内缩量。
 */
export const EDITOR_VALUE_REACH_INSET = 4;

/**
 * 编辑器纵轴的半量程（值 `±1` 到中线的像素距离）。
 *
 * 退化尺寸（高度小于两倍内缩量）下退到 `1e-6` 而不是负数 —— 命测试只要求不产生
 * `NaN`，负量程会让"顶端为 +1"翻转成"顶端为 −1"。
 */
export function editorValueReach(height: number): number {
    const h = height > 0 ? height : 1;
    return Math.max(1e-6, h / 2 - EDITOR_VALUE_REACH_INSET);
}

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

// ---- 整体变换（右键拖拽） --------------------------------------------------
//
// 【为什么是「旋转 + 缩放」而不是「平移」】表的横轴是循环相位、纵轴已经是
// `[-1,1]` 的满量程：水平平移在这个模型里就是**无缝旋转**（不需要缝合端点），
// 而纵向平移在默认（顶满）的表上没有任何余量 —— 保形就拖不动，要拖得动就得
// 让值溢出盒子或把顶部压平。因此纵向取**幅度缩放**：它既总有意义，又落在
// "形状自身占了多少可用深度"这个没有其他旋钮覆盖的位置上。

/** 整体缩放的倍数下限 / 上限。 */
export const CYCLE_SCALE_MIN = 0.05;
export const CYCLE_SCALE_MAX = 4;

/** 过滤非有限值 —— 表里混进 `NaN` 时按 0 处理，不让它污染整条曲线。 */
function safeValue(value: number | undefined): number {
    return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function clampScale(factor: number): number {
    return clamp(Number.isFinite(factor) ? factor : 1, CYCLE_SCALE_MIN, CYCLE_SCALE_MAX);
}

/** 循环线性取样：`pos` 为格号，可取小数、可为负（表视为首尾相接）。 */
function sampleTableAt(table: readonly number[], pos: number): number {
    const n = table.length;
    if (n === 0) return 0;
    if (n === 1) return safeValue(table[0]);
    const wrapped = ((pos % n) + n) % n;
    const i0 = Math.floor(wrapped) % n;
    const i1 = (i0 + 1) % n;
    const frac = wrapped - Math.floor(wrapped);
    const a = safeValue(table[i0]);
    const b = safeValue(table[i1]);
    return a + (b - a) * frac;
}

/**
 * 整体旋转：正 `shiftBins` 表示**图像向右移动** `shiftBins` 格（可含小数）。
 *
 * 【为什么向右是 `new[i] = old[i - shift]`】要把画面往右挪一格，第 `i+1` 格上
 * 应当出现原来第 `i` 格的内容 —— 即新表的第 `i` 格取旧表第 `i - 1` 格。
 * 整数位移下 `frac` 恒为 0，逐位等值（无插值损失）；`shift` 为整圈时是恒等。
 */
export function rotateCycleTable(table: readonly number[], shiftBins: number): number[] {
    const n = table.length;
    if (n === 0) return [];
    const shift = Number.isFinite(shiftBins) ? shiftBins : 0;
    if (shift === 0 || shift % n === 0) return table.slice();
    const out = new Array<number>(n);
    for (let i = 0; i < n; i += 1) out[i] = clampValue(sampleTableAt(table, i - shift));
    return out;
}

/**
 * 整体缩放：绕零线乘 `factor`，逐格钳到 `[-1,1]`。
 *
 * 【倍数也在这里钳】下限 `CYCLE_SCALE_MIN` 是安全阀：`0 × 任何 = 0`，一旦缩到
 * 全零，形状就被**永久**抹平（拖回去也回不来）。留 5% 保住形状信息。
 * 放大的钳制只是防御 —— 值本来就钳到 `±1`，超过 `1/峰值` 后画面上只剩削顶。
 */
export function scaleCycleTable(table: readonly number[], factor: number): number[] {
    const n = table.length;
    if (n === 0) return [];
    const f = clampScale(factor);
    const out = new Array<number>(n);
    for (let i = 0; i < n; i += 1) out[i] = clampValue(safeValue(table[i]) * f);
    return out;
}

/**
 * 一次手势的合成变换（对**按下时的快照**调用）：先旋转再缩放，单趟产出。
 *
 * 【为什么必须对快照而不是上一帧结果】旋转是线性插值，反复插值会让曲线逐渐变平、
 * 峰值衰减；对快照重算才能保证"拖出去再拖回来"逐位复原。缩放同理：对快照乘倍数，
 * 拖到下限再拖回来能精确回到 `1×`，连乘则会滚雪球。
 *
 * 与 `scaleCycleTable(rotateCycleTable(t, s), f)` 结果一致（两步都是逐格运算）。
 */
export function transformCycleTable(
    table: readonly number[],
    shiftBins: number,
    factor: number,
): number[] {
    const n = table.length;
    if (n === 0) return [];
    const shift = Number.isFinite(shiftBins) ? shiftBins : 0;
    const f = clampScale(factor);
    const out = new Array<number>(n);
    for (let i = 0; i < n; i += 1) {
        const base = shift === 0 ? safeValue(table[i]) : sampleTableAt(table, i - shift);
        out[i] = clampValue(base * f);
    }
    return out;
}

/** 右键拖拽的变换量。 */
export interface CycleTransformSpec {
    /** 图像向右移动的格数（循环，可含小数）。 */
    rotateBins: number;
    /** 绕零线的幅度倍数。 */
    scale: number;
}

/**
 * 手势映射：累计位移 → 变换量（纯算术，不碰 DOM）。
 *
 * - 水平：**一个画布宽 = 一个整周期**（编辑器把 `n + 1` 个点铺满整幅宽），
 *   所以 `dx / width` 就是"图像移动了几个周期"，乘 `bins` 得到格数。
 *   该比例与表长无关，`bins` 由调用方给出（表长可能是提取来的非默认值）。
 * - 垂直：`2 ** (-dy / height)` —— 向上拖满一个画布高 = `2×`，向下 = `0.5×`，
 *   几何对称，且"拖回原位 = `1×`"。用几何而非线性是为了让放大 / 缩小手感相同。
 * - 退化尺寸（宽 / 高为 0）退化为不变换，避免除零与 `NaN`。
 */
export function cycleRightDragTransform(
    deltaX: number,
    deltaY: number,
    widthPx: number,
    heightPx: number,
    bins: number,
): CycleTransformSpec {
    const dx = Number.isFinite(deltaX) ? deltaX : 0;
    const dy = Number.isFinite(deltaY) ? deltaY : 0;
    const n = Math.max(1, Math.round(Number.isFinite(bins) ? bins : CYCLE_EDIT_BINS));
    const rotateBins = widthPx > 0 ? (dx / widthPx) * n : 0;
    const scale = heightPx > 0 ? clampScale(2 ** (-dy / heightPx)) : 1;
    return { rotateBins, scale };
}

/** 值 → 画布纵向像素。与 `cycleEditorPoint` 互为逆运算（同一套量程）。 */
export function cycleEditorY(value: number, height: number): number {
    const h = height > 0 ? height : 1;
    return h / 2 - clampValue(value) * editorValueReach(h);
}

/**
 * 画布坐标 →（格号, 值）。顶端是 `+1`，底端是 `-1`，量程 = `editorValueReach`。
 *
 * 【为什么值要钳制】内缩意味着 `y` 进入上下各 `INSET` 像素的边带时算出来会超过
 * `±1`；那不是"更响的颤音"，只是指针越过了峰顶。钳住才能让"把波峰拖到顶点"
 * 稳定地停在 `1.0`，而不是随边带内的像素继续往上飘。
 */
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
    const reach = editorValueReach(h);
    return {
        bin: clamp(Math.floor((x / w) * n), 0, n - 1),
        value: clampValue((h / 2 - y) / reach),
    };
}
