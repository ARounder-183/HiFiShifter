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
 * | 其余原始值域 | `tension` / `breathiness` / `pan` | additive | 半量程/100 | 按量程百分比 |
 *
 * 【乘性族的静音保护】`dyn` / `volume` / `breath_gain` 都是增益：乘性调制下
 * `base = 0` 恒为 `0`，静音帧不会被"抬起来"。历史实现在 `volume` /
 * `breath_gain` 上误用了加性，静音段会出现呼吸感。
 */

import {
    CHILD_FORMANT_OFFSET_CENTS_RANGE,
    CHILD_PITCH_OFFSET_CENTS_RANGE,
    isChildFormantOffsetCentsParam,
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
} from "../../components/layout/pianoRoll/childPitchOffsetParams";
import { isDynParam, VOLUME_PARAM_ID } from "../../components/layout/pianoRoll/paramRanges";
import type { StepUnit } from "../../ui/stepPolicy";
import type { DepthFamily } from "./vibratoTypes";

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
    tension: { min: -100, max: 100 },
    breathiness: { min: -10000, max: 10000 },
    pan: { min: -1, max: 1 },
};

/** 1 音级在 cents 上的名义值（小二度）。 */
const CENTS_PER_SCALE_STEP = 100;

/** 判定参数所属的深度族。 */
export function depthFamilyOf(param: string): DepthFamily {
    if (isDynParam(param) || RATIO_PARAMS.has(param)) return "ratio";
    if (SEMITONE_PARAMS.has(param)) return "cents";
    if (
        isChildPitchOffsetCentsParam(param) ||
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
 * 参数深度的**步长单位**（`AppNumberField` 的 `unit`）。
 *
 * 预设编辑器与 HUD 都以参数原生单位呈现深度，用户永远看不到 cents 这个
 * 中间量；步长也跟着原生动量级走，而不是统一给一个"整数"。
 */
export function depthStepUnitFor(param: string): StepUnit {
    if (isChildPitchOffsetDegreesParam(param)) return "scaleDegree";
    switch (depthFamilyOf(param)) {
        case "ratio":
            return "percent";
        case "cents":
            // 半音值参数（pitch）以 cents 编辑 —— 揉弦深度用分比用半音直观得多。
            return "cents";
        default:
            return "integer";
    }
}
