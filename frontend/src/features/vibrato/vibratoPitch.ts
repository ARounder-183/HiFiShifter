/**
 * 音高参数的「未设置」哨兵在颤音链路上的适配。
 *
 * 【语义】`pitch` 的参数值里 **0 = 未检测到音高**（未浊 / 静音帧）；真实音高是
 * 1..127 的半音值。后端（`commands/params.rs` 的读写两侧）与前端写入口
 * （`paramRanges.clampParamWriteValue`）都以此为准，参数编辑器里每个变换也都遵守
 * 「0 进 0 出」。
 *
 * 【为什么颤音必须单独适配】颤音的基线锚点是**从选区两端读出来的**
 * （`startValue` = 选区首帧、`endValue` = 末帧）。首 / 末帧一旦是未检测帧，锚点就
 * 成了 0：`baseline: "line"` 会把整条选区拉直成 0，中间真实唱出来的音高被整段抹掉。
 * 那不是"颤音加得不对"，是把一段音高删了 —— 而"选中一整句"几乎必然包含句首句尾
 * 的气口 / 静音，所以这是常态而不是边角。
 *
 * 本模块给出「读数据 → 变换 → 写回」要成对使用的三件事：
 *
 * 1. {@link isUnsetValue} —— 判定某帧是不是哨兵；
 * 2. {@link resolveVibratoAnchors} —— 锚点只从**已检测帧**里取；整段都未检测时返回
 *    `null`（没有可调制的对象，调用方应当放弃，而不是照着一片 0 写出 0）；
 * 3. {@link suppressUnsetValues} —— 写回前把哨兵帧还原成 0，绝不把"未检测"物化成
 *    一个具体音高（那会在气口上凭空造出一个音）。
 *
 * 第 3 件事与 `restoreDynSentinels` 对 dyn 做的是同一件事。区别只在数据来源：dyn
 * 的"未画"是相对基线的，只能靠后端回传的位图识别；pitch 的哨兵就在值本身（0），
 * 从原曲线直接推得出来，因此不需要新增协议字段。
 */

import { PITCH_PARAM_ID } from "./vibratoDepth";

/**
 * 该参数是否用 0 表示「无数据」。
 *
 * 目前只有音高。落成一个具名判定而不是散落的 `param === "pitch"`，是为了让
 * "哨兵适配"这件事有唯一的入口 —— 将来若某个参数也采用 0 哨兵，改这里一处即可。
 */
export function usesUnsetValue(param: string): boolean {
    return param === PITCH_PARAM_ID;
}

/**
 * 该帧是否为「未设置」。
 *
 * 非有限值一并算作未设置：静音 / 未分析帧在上游可能以 `NaN` 出现，把它当成一个
 * 具体音高写回去比丢掉它更糟。
 */
export function isUnsetValue(param: string, value: number): boolean {
    if (!usesUnsetValue(param)) return false;
    return !Number.isFinite(value) || value === 0;
}

/** 基线锚点：曲线首尾的**有效**值。 */
export interface VibratoAnchors {
    startValue: number;
    endValue: number;
}

/**
 * 取基线锚点。
 *
 * - 非哨兵参数：与既有行为逐字一致（首帧 / 末帧）；
 * - 哨兵参数：取**已检测帧**里的首值与末值；一段都没有时返回 `null`。
 *
 * 【为什么只换锚点就够】`buildVibratoCurve` 的 `line` / `holdStart` / `holdEnd` /
 * `average` 四种基线全部由这两个值决定（`average` 是两端均值）；把它们换成已检测帧的
 * 端点，四种模式就同时正确了，不需要在曲线内核里再开一条按参数分岔的路径。
 * `existing` 模式逐帧读原曲线，本来就与锚点无关。
 *
 * 【只有首末两帧未检测时也正确吗】是。中间已检测的帧仍落在正确的直线上；首尾那些
 * 未检测帧的取值随后由 {@link suppressUnsetValues} 还原成哨兵，不会写出直线在端点
 * 外的外推值。
 */
export function resolveVibratoAnchors(
    param: string,
    values: readonly number[],
): VibratoAnchors | null {
    if (values.length === 0) return null;
    if (!usesUnsetValue(param)) {
        return { startValue: values[0], endValue: values[values.length - 1] };
    }
    let first = -1;
    let last = -1;
    for (let i = 0; i < values.length; i += 1) {
        if (isUnsetValue(param, values[i])) continue;
        if (first < 0) first = i;
        last = i;
    }
    if (first < 0) return null;
    return { startValue: values[first], endValue: values[last] };
}

/**
 * 把「未设置」帧还原成哨兵（**就地**修改 `result`；与 `original` 按较短者对齐）。
 *
 * 幂等：对已经还原过的结果再调用一次不会变。调用时机是"变换之后、写回之前" ——
 * 与 dyn 的 `restoreDynSentinels` 完全同位。
 */
export function suppressUnsetValues(
    param: string,
    result: number[],
    original: readonly number[],
): void {
    if (!usesUnsetValue(param)) return;
    const count = Math.min(result.length, original.length);
    for (let i = 0; i < count; i += 1) {
        if (isUnsetValue(param, original[i])) result[i] = 0;
    }
}
