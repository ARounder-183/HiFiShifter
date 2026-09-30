/**
 * 深度的规范单位与按参数换算。
 *
 * 【为什么需要这一层】预设内部只存一个数（`depthCents`），但同一个 `30`
 * 在不同参数上是完全不同的量：音高上是 ±0.3 半音，动态上是 ±30% 倍率，
 * 张力（±100）上是 ±30 单位，气息（±10000）上是 ±3000。若不做换算，
 * 预设就只能在单一参数上可用。
 *
 * 【换算表】`factor` 与 `mode` 两个量就能覆盖全部情形：
 * ```
 * additive       : value = base + depthCents * factor * wave
 * multiplicative : value = base * (1 + depthCents * factor * wave)
 * ```
 *
 * | 参数族 | 例 | mode | factor | 依据 |
 * | - | - | - | - | - |
 * | 音高（半音值） | `pitch` | additive | 0.01 | 1 半音 = 100 cents |
 * | cents 类参数 | `child_pitch_offset_cents@*` | additive | 1 | 值本身就是 cents |
 * | 音级类参数 | `child_pitch_offset_degrees@*` | additive | 0.01 | 1 音级名义 = 100 cents |
 * | 乘性增益 | `dyn` / `volume` / `breath_gain` | multiplicative | 0.01 | cents/100 即深度比 |
 * | 其余原始值域 | `hifigan_tension` / `breathiness` / `pan` | additive | 半量程/100 | 按量程百分比 |
 *
 * 【乘性族的静音保护】`dyn` / `volume` / `breath_gain` 都是增益：乘性调制下
 * `base = 0` 恒为 `0`，静音帧不会被"抬起来"。历史实现在 `volume` /
 * `breath_gain` 上误用了加性，静音段会出现呼吸感。
 *
 * 【换算之外还有两件事要按参数定】同一个 `depthCents` 落在不同参数上是完全
 * 不同的量，因此除了"cents → 原生单位"，本文件还负责：
 * - {@link fullSwingCentsFor}：这个参数的"满摆幅"是多少分 —— 深度钳在它之内，
 *   否则写入口会把超出量程的部分钳平，波形顶部出现平顶；
 * - {@link depthStepCentsFor}：一个调整格走多少分 —— 恒为满摆幅的
 *   {@link FULL_SWING_DIVISIONS} 分之一，于是"从零调到满幅"在每个参数上都是
 *   同样多格，而不是音高 100 格、声像 400 格。
 */

import {
    CHILD_FORMANT_OFFSET_CENTS_RANGE,
    CHILD_PITCH_OFFSET_CENTS_RANGE,
    CHILD_PITCH_OFFSET_DEGREES_RANGE,
    isChildFormantOffsetCentsParam,
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
} from "../../components/layout/pianoRoll/childPitchOffsetParams";
import { isDynParam, VOLUME_PARAM_ID } from "../../components/layout/pianoRoll/paramRanges";
import type { StepUnit } from "../../ui/stepPolicy";
import type { DepthFamily } from "./vibratoTypes";
import { VIBRATO_LIMITS } from "./vibratoPresets";

/** 值域。`min` / `max` 均为曲线值的原生单位。 */
export interface VibratoParamRange {
    min: number;
    max: number;
}

/** 规范化单位与参数单位之间的换算方式。 */
export interface DepthMapping {
    family: DepthFamily;
    mode: "additive" | "multiplicative";
    /** `cents → 参数单位` 的乘数。 */
    factor: number;
}

/** 音高参数 id（半音值的唯一代表）。 */
export const PITCH_PARAM_ID = "pitch";

/** 音高类：曲线值是半音。 */
const SEMITONE_PARAMS = new Set<string>([PITCH_PARAM_ID]);

/** 乘性增益参数：曲线值是倍率，静音必须保持静音。 */
const RATIO_PARAMS = new Set<string>([VOLUME_PARAM_ID, "hifigan_volume", "breath_gain"]);

/** 原始值域参数在缺少描述符时的兜底量程。 */
const RAW_RANGE_FALLBACK: Record<string, VibratoParamRange> = {
    // 描述符里的真实 id 是 `hifigan_tension`（见 `renderer/chain.rs`）；`tension`
    // 是历史写法，一并保留 —— 漏掉真实 id 会让张力退回到 `{0, 2}` 的兜底量程，
    // 换算因子差 100 倍（30 分的深度只摆 0.3 个单位，肉眼看不见）。
    hifigan_tension: { min: -100, max: 100 },
    tension: { min: -100, max: 100 },
    breathiness: { min: -10000, max: 10000 },
    pan: { min: -1, max: 1 },
};

/** 1 音级在 cents 上的名义值（小二度）。 */
const CENTS_PER_SCALE_STEP = 100;

/**
 * 满摆幅被分成多少格。
 *
 * 一个调整格（滚轮一格 / 方向键一下）改变的深度 = 满摆幅 / 这个数，取 50 是
 * 让"从零调到满幅"大约需要 50 格 —— 几下就能拉到位，配合精细调整修饰键
 * （×0.1）又细到 2% 的刻度上。此前各族各写各的（音高 24 分、增益 1 分、
 * 原始值域 0.5 分），同一个"一格"在不同参数上差出几十倍。
 */
export const FULL_SWING_DIVISIONS = 50;

/**
 * 非 cents 族（增益倍率 / 原始值域）的满摆幅（cents）。
 *
 * 【为什么恰好是 100，而不是从 `range` 推】两族的 `factor` 已经把量程编进去了：
 * - 增益倍率（`dyn` / `volume` / `breath_gain`）factor = 0.01，100 分 = ±1.0 倍率；
 * - 原始值域 factor = 半量程 / 100，100 分 = ±半量程，正好铺满整个值域。
 *
 * 于是 100 分就是"把参数调制到满幅"所需的深度，与参数自身量程无关。
 */
export const NON_CENTS_FULL_SWING_CENTS = 100;

/** 判定参数所属的深度族。 */
export function depthFamilyOf(param: string): DepthFamily {
    if (isDynParam(param) || RATIO_PARAMS.has(param)) return "ratio";
    if (SEMITONE_PARAMS.has(param)) return "cents";
    if (
        isChildPitchOffsetCentsParam(param) ||
        // 音级类：曲线值是音级，1 音级名义 100 分（见 `CENTS_PER_SCALE_STEP`）。
        // 它曾经漏在这张表外，于是被当成"原始值域"，换算因子取半量程 / 100
        // （±14 音级 → 0.14）而不是 1/100 —— 同一个预设落在音级参数上的深度
        // 比落在音分参数上大 14 倍。
        isChildPitchOffsetDegreesParam(param) ||
        isChildFormantOffsetCentsParam(param) ||
        // 处理器描述符里的共振峰偏移同样是 cents。
        param === "formant_shift_cents"
    ) {
        return "cents";
    }
    return "raw";
}

/** 参数在缺少描述符时的兜底量程；返回 `undefined` 表示未知。 */
export function fallbackRangeFor(param: string): VibratoParamRange | undefined {
    if (isChildPitchOffsetCentsParam(param)) return CHILD_PITCH_OFFSET_CENTS_RANGE;
    if (isChildPitchOffsetDegreesParam(param)) return CHILD_PITCH_OFFSET_DEGREES_RANGE;
    if (isChildFormantOffsetCentsParam(param)) return CHILD_FORMANT_OFFSET_CENTS_RANGE;
    return RAW_RANGE_FALLBACK[param];
}

/**
 * 求 `depthCents → 参数单位` 的换算方式。
 *
 * @param param 参数名（可含子轨后缀）。
 * @param range 当前参数的值域（通常来自 `currentParamRange` / 处理器描述符）。
 *              仅 `raw` 族需要它来定标；缺失时退回半量程 1。
 */
export function depthMappingFor(param: string, range?: VibratoParamRange): DepthMapping {
    const family = depthFamilyOf(param);

    if (family === "ratio") {
        return { family, mode: "multiplicative", factor: 0.01 };
    }

    if (family === "cents") {
        // 值本身是 cents 的参数按 1:1；半音值与音级值按 1/100。
        const valueIsCents =
            isChildPitchOffsetCentsParam(param) ||
            isChildFormantOffsetCentsParam(param) ||
            param === "formant_shift_cents";
        return {
            family,
            mode: "additive",
            factor: valueIsCents ? 1 : 1 / CENTS_PER_SCALE_STEP,
        };
    }

    const effective = range ?? fallbackRangeFor(param) ?? { min: 0, max: 2 };
    const span = Number(effective.max) - Number(effective.min);
    const halfSpan = Number.isFinite(span) && span > 0 ? span / 2 : 1;
    return { family, mode: "additive", factor: halfSpan / 100 };
}

/** 参数的值域换算到 cents 后的半量程；无法判定时返回 `undefined`。 */
function centsHalfSpanFor(param: string, range?: VibratoParamRange): number | undefined {
    const effective = range ?? fallbackRangeFor(param);
    if (!effective) return undefined;
    const span = Number(effective.max) - Number(effective.min);
    if (!Number.isFinite(span) || span <= 0) return undefined;
    const factor = depthMappingFor(param, effective).factor;
    if (!(factor > 0)) return undefined;
    return span / 2 / factor;
}

/**
 * 参数的**满摆幅**深度（cents）—— 把这个参数调制到铺满其可表达范围所需的深度。
 *
 * 【为什么要按参数分】同一个 `depthCents` 在不同参数上是不同的量：音高上 30 分
 * 是三分之一半音（很浅），声像上 30 分是 ±0.3（很显眼），共振峰上 30 分只占
 * ±500 的 6%。把深度钳在满摆幅之内、把步长定成满摆幅的一份，才能让"一格"
 * "调到满幅"这类操作在每个参数上得到同一量级的响应。
 *
 * - cents 族（音高 / 共振峰 / 音级）：深度是**绝对音分**，以工具自身的深度上限
 *   为准；但不超过参数自己的量程 —— 共振峰只能摆 ±500，再大只会被写入口钳平，
 *   波形顶部出现平顶（拖到 1200 分并不"更颤"，只是变成方波）；
 * - 增益倍率与原始值域：恒为 {@link NON_CENTS_FULL_SWING_CENTS}。
 */
export function fullSwingCentsFor(param: string, range?: VibratoParamRange): number {
    if (depthFamilyOf(param) !== "cents") return NON_CENTS_FULL_SWING_CENTS;
    const half = centsHalfSpanFor(param, range);
    return Math.min(VIBRATO_LIMITS.depthCents.max, half ?? VIBRATO_LIMITS.depthCents.max);
}

/**
 * 一个调整格（滚轮一格 / 方向键一下）改变的深度（cents）。
 *
 * 恒为满摆幅的 {@link FULL_SWING_DIVISIONS} 分之一 —— 音高 24 分（沿用历史
 * 手感）、cents 类与音级类同样 24 分、增益倍率与原始值域 2 分（= 2%）。
 */
export function depthStepCentsFor(param: string, range?: VibratoParamRange): number {
    return fullSwingCentsFor(param, range) / FULL_SWING_DIVISIONS;
}

/**
 * 把深度钳在参数的满摆幅之内（负值保留 —— 负深度是把波形整体反相）。
 *
 * 越界不会报错，只会被写入口静默钳平，因此必须在**产生深度**的地方就钳住。
 */
export function clampDepthCentsForParam(
    depthCents: number,
    param: string,
    range?: VibratoParamRange,
): number {
    // NaN 不是"很大/很小"，而是"没有值"—— 归零，别让它传播成曲线里的 NaN。
    if (Number.isNaN(depthCents)) return 0;
    const limit = Math.min(
        VIBRATO_LIMITS.depthCents.max,
        Math.abs(fullSwingCentsFor(param, range)),
    );
    return Math.max(-limit, Math.min(limit, depthCents));
}

/**
 * 把 `depthCents` 换算成参数原生单位的「最大偏差」。
 *
 * 这是**显示与编辑用**的数值（预设编辑器、HUD 都以参数原生单位呈现，
 * 用户永远看不到 cents 这个中间量）。
 */
export function depthToParamUnit(
    depthCents: number,
    param: string,
    range?: VibratoParamRange,
): number {
    const mapping = depthMappingFor(param, range);
    return depthCents * mapping.factor;
}

/** {@link depthToParamUnit} 的逆运算：参数原生单位 → 规范 cents。 */
export function paramUnitToDepth(value: number, param: string, range?: VibratoParamRange): number {
    const mapping = depthMappingFor(param, range);
    if (!(mapping.factor > 0)) return 0;
    return value / mapping.factor;
}

/**
 * 深度在**编辑器 / HUD** 上的显示换算（cents → 显示单位）。
 *
 * 【为什么与 `depthMappingFor().factor` 分开】两者对"cents 值多少"的回答不同：
 * 曲线把音高按**半音**存（`pitch` 的曲线因子是 1/100，30 cents = 0.3 半音），
 * 而编辑器按**分**显示（揉弦深度用分直观得多）。显示若套用曲线因子，用户在
 * 音高上看到的就是 `0.3` 而不是 `30`。
 *
 * | 参数 | 显示单位 | 因子 |
 * | - | - | - |
 * | `pitch` 及 cents 类 | 分 | 1 |
 * | 音级类 | 音级 | 1/100 |
 * | 乘性增益 | 满摆幅的百分比 | 100 / 满摆幅 |
 * | 其余原始值域 | 满摆幅的百分比 | 100 / 满摆幅 |
 *
 * 【为什么原始值域也改成百分比】它原先按**原生单位**显示：声像上是 `0.3`、
 * 气声上是 `3000`，两个都是没有量纲感的裸数字，用户无从判断"这是深还是浅"。
 * 改成"占满摆幅的百分比"之后，声像与气声都是 `30%` —— 与增益倍率同一套说法，
 * 也和 HUD 上"这一笔有多深"的直觉一致。
 *
 * 必须与 {@link depthStepUnitFor} 成对使用：因子决定"显示几"，步长单位决定
 * "滚一格走多少"。
 */
export function depthDisplayFactor(param: string, range?: VibratoParamRange): number {
    // 音级类：1 音级名义 100 分。
    if (isChildPitchOffsetDegreesParam(param)) return 1 / CENTS_PER_SCALE_STEP;
    // cents 族按分显示（因子 1）；增益倍率与原始值域按"满摆幅百分比"显示。
    // 后两族的满摆幅恰好是 `NON_CENTS_FULL_SWING_CENTS`，于是显示值等于规范深度。
    if (depthFamilyOf(param) === "cents") return 1;
    const swing = fullSwingCentsFor(param, range);
    return swing > 0 ? NON_CENTS_FULL_SWING_CENTS / swing : 1;
}

/** 深度（cents）→ 编辑器显示值。 */
export function depthToDisplay(
    depthCents: number,
    param: string,
    range?: VibratoParamRange,
): number {
    return depthCents * depthDisplayFactor(param, range);
}

/** 编辑器显示值 → 深度（cents）。 */
export function displayToDepth(value: number, param: string, range?: VibratoParamRange): number {
    const factor = depthDisplayFactor(param, range);
    if (!(factor > 0)) return 0;
    return value / factor;
}

/**
 * 参数深度的**步长单位**（`AppNumberField` 的 `unit`）。
 *
 * 预设编辑器与 HUD 都以参数原生单位呈现深度，用户永远看不到 cents 这个
 * 中间量；步长也跟着原生动量级走，而不是统一给一个"整数"。
 */
export function depthStepUnitFor(param: string): StepUnit {
    if (isChildPitchOffsetDegreesParam(param)) return "scaleDegree";
    switch (depthFamilyOf(param)) {
        case "ratio":
        case "raw":
            // 两族都以"满摆幅百分比"呈现深度：增益倍率本来就是百分比，
            // 原始值域（声像 / 张力 / 气声）也按占满摆幅的比例说话。
            // 曾经原始值域走 `integer`，于是编辑器里一格 = 1 个原生单位
            // （声像上就是整条 ±1 量程），与拖拽的 0.5 分差了 200 倍。
            return "percent";
        default:
            // 音高与 cents 类参数都以分编辑 —— 揉弦深度用分比用半音直观得多。
            return "cents";
    }
}
