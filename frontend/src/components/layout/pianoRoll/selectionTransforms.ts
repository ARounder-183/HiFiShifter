import type { ParamName } from "./types";
import {
    GAUSSIAN_CUTOFF_SIGMAS,
    editablePitchValue,
    smoothCurveGaussian,
    smoothSigmaMsFromUnits,
} from "./paramSmoothing";

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

function isPitchParam(editParam: ParamName): boolean {
    return editParam === "pitch";
}

function isEditableValue(editParam: ParamName, value: number): boolean {
    return Number.isFinite(value) && (!isPitchParam(editParam) || value !== 0);
}

export function computeSelectionMean(values: number[], editParam: ParamName): number {
    let sum = 0;
    let count = 0;
    for (const value of values) {
        if (!isEditableValue(editParam, value)) {
            continue;
        }
        sum += value;
        count += 1;
    }
    return count > 0 ? sum / count : 0;
}

export function averageSelectionValues(
    values: number[],
    editParam: ParamName,
    strengthPercent: number,
): number[] {
    const strength = clamp((Number(strengthPercent) || 0) / 100, 0, 1);
    if (strength <= 0) {
        return values.slice();
    }
    const mean = computeSelectionMean(values, editParam);
    return values.map((value) => {
        if (!isEditableValue(editParam, value)) {
            return value;
        }
        return value + (mean - value) * strength;
    });
}

export function scaleSelectionDeviation(
    values: number[],
    editParam: ParamName,
    scale: number,
): number[] {
    const mean = computeSelectionMean(values, editParam);
    return values.map((value) => {
        if (!isEditableValue(editParam, value)) {
            return value;
        }
        return mean + (value - mean) * scale;
    });
}

// ── 右键上拖：音高残差放大 / 非音高均值中心缩放 ────────────────────────────

/** 趋势提取 σ 下限（毫秒）：极短选区兜底，防止 σ 过小使残差≈0（放大失效）。 */
export const AMPLIFY_TREND_SIGMA_MIN_MS = 40;
/** 趋势提取 σ 上限（毫秒）：超长选区的性能/语义兜底。 */
export const AMPLIFY_TREND_SIGMA_MAX_MS = 250;
/** 自适应趋势 σ = 选区时长 ÷ 该因子（设计依据：σ ≥ 颤音周期/3 时高斯对
 *  ~5.5Hz 颤音的泄漏 <1%，音符轮廓与滑音完整保留在趋势里）。 */
export const AMPLIFY_TREND_SPAN_DIVISOR = 6;
/** 哨兵防护下限：放大把有声音高推向 ≤0 时钉在该正值，绝不写入未浊哨兵 0。 */
export const AMPLIFY_PITCH_SENTINEL_FLOOR = 0.01;

export type SelectionAmplifyOptions = {
    /** 帧周期（毫秒），用于趋势 σ 的毫秒定标。缺省 5ms。 */
    framePeriodMs?: number;
    /** 显式覆盖趋势 σ；缺省按选区时长自适应（span/DIVISOR，clamp 40-250ms）。 */
    trendSigmaMs?: number;
};

export type SelectionAmplifier = {
    /** 应用放大倍率（可多次调用；趋势只预计算一次，逐 move 调用无重复卷积）。 */
    apply: (scale: number) => number[];
};

/**
 * 创建选区「幅度放大器」（右键上拖的数据层）。
 *
 * 音高参数：**残差放大** out(f) = trend(f) + (v(f) − trend(f))·scale ——
 * 趋势（音符轮廓/滑音/音程）保持不动，只有细节（颤音、抖动、重音）被缩放。
 * trend 为曲线的高斯平滑版（复用 paramSmoothing 的 trend 延拓与未浊分段；
 * 对数音高中 trend 减去全局均值的旧做法会把音程一起放大，见
 * scaleSelectionDeviation 的对照测试）。
 *
 * 防越界双守卫：
 * - 哨兵防护：结果 ≤0 时钉在 AMPLIFY_PITCH_SENTINEL_FLOOR，绝不写未浊 0；
 * - 过冲抑制：残差放大在音符台阶附近会过冲（out 越出原值域，形成
 *   "舀音"）。以逐帧包络 [min(v,trend) − B, max(v,trend) + B] 钳制，
 *   B = scale × median|残差|（鲁棒统计：平台主导的选区 B≈0 → 台阶附近
 *   几乎无过冲；颤音选区 B≈颤音深度 → 深化不受限）。
 *
 * 非音高参数：保持均值中心缩放（围绕平均值的动态范围扩张，语义本就正确）。
 */
export function createSelectionAmplifier(
    values: number[],
    editParam: ParamName,
    opts?: SelectionAmplifyOptions,
): SelectionAmplifier {
    if (isPitchParam(editParam) && values.length > 0) {
        const fpMs = Math.max(1e-6, Number(opts?.framePeriodMs) || 5);
        const spanMs = (values.length - 1) * fpMs;
        const sigmaMs =
            opts?.trendSigmaMs !== undefined
                ? Math.max(0, Number(opts.trendSigmaMs) || 0)
                : Math.min(
                      AMPLIFY_TREND_SIGMA_MAX_MS,
                      Math.max(
                          AMPLIFY_TREND_SIGMA_MIN_MS,
                          spanMs / AMPLIFY_TREND_SPAN_DIVISOR,
                      ),
                  );
        // σ 不足以形成平滑（<0.5 帧）时高斯为恒等 → trend = values → 残差 0
        // → apply 恒等（显式 trendSigmaMs=0 即"不做残差放大"）。
        const trend =
            sigmaMs / fpMs >= 0.5
                ? smoothCurveGaussian(values, {
                      sigmaMs,
                      framePeriodMs: fpMs,
                      valueFilter: editablePitchValue,
                  })
                : values.slice();
        // 鲁棒细节尺度：median|残差|（平台帧主导时趋近 0，颤音选区≈颤音深度）
        let medianAbs = 0;
        {
            const absResidual: number[] = [];
            for (let i = 0; i < values.length; i += 1) {
                if (!isEditableValue(editParam, values[i])) continue;
                absResidual.push(Math.abs(values[i] - trend[i]));
            }
            if (absResidual.length > 0) {
                absResidual.sort((a, b) => a - b);
                medianAbs =
                    absResidual.length % 2 === 1
                        ? absResidual[(absResidual.length - 1) / 2]
                        : (absResidual[absResidual.length / 2 - 1] +
                              absResidual[absResidual.length / 2]) /
                          2;
            }
        }
        return {
            apply: (scale: number) => {
                const k = Math.max(0, Number(scale) || 0);
                if (k === 1) return values.slice();
                const margin = k * medianAbs;
                return values.map((v, i) => {
                    if (!isEditableValue(editParam, v)) return v;
                    const t = trend[i];
                    const raw = t + (v - t) * k;
                    const lo = Math.min(v, t) - margin;
                    const hi = Math.max(v, t) + margin;
                    const out = Math.min(hi, Math.max(lo, raw));
                    return out > 0 ? out : AMPLIFY_PITCH_SENTINEL_FLOOR;
                });
            },
        };
    }
    // 非音高（或 legacy 开关）：均值中心缩放，无需预计算
    return {
        apply: (scale: number) => scaleSelectionDeviation(values, editParam, scale),
    };
}

/** 一次性便捷入口（单次变换用；逐 move 调用请用 createSelectionAmplifier 缓存趋势）。 */
export function amplifySelectionDeviation(
    values: number[],
    editParam: ParamName,
    scale: number,
    opts?: SelectionAmplifyOptions,
): number[] {
    return createSelectionAmplifier(values, editParam, opts).apply(scale);
}

/** smoothSelectionValues 的可选上下文：选区边界外的真实曲线值（用于边界无缝）。 */
export type SmoothSelectionOptions = {
    /** 帧周期（毫秒）；σ 以毫秒定义，经此换算为帧。缺省 5ms。 */
    framePeriodMs?: number;
    /** 选区紧邻左侧的曲线值（末位对应 values[−1]）。 */
    leftContext?: number[];
    /** 选区紧邻右侧的曲线值（首位对应 values[n]）。 */
    rightContext?: number[];
};

/**
 * 「平滑化」/ 右拖下压：对选区曲线做单次高斯平滑。
 *
 * σ 由强度以毫秒定义（u=1 → 60ms，见 paramSmoothing.smoothSigmaMsFromUnits），
 * 与帧周期解耦；**强度单位不设上限**——右键下拖超过 -50px（u>1）后按同一
 * 幂曲线继续加深，与弹窗的无限百分比显示一致。pitch=0 哨兵帧既不参与均值
 * 也不被改写；提供上下文时边界无缝（无端点内拉/台阶）。
 */
export function smoothSelectionValues(
    values: number[],
    editParam: ParamName,
    strengthUnits: number,
    opts?: SmoothSelectionOptions,
): number[] {
    const units = Math.max(0, Number(strengthUnits) || 0);
    if (units <= 0 || values.length === 0) {
        return values.slice();
    }
    return smoothCurveGaussian(values, {
        sigmaMs: smoothSigmaMsFromUnits(units),
        framePeriodMs: Math.max(1e-6, Number(opts?.framePeriodMs) || 5),
        valueFilter: isPitchParam(editParam)
            ? (v) => Number.isFinite(v) && v !== 0
            : undefined,
        leftContext: opts?.leftContext,
        rightContext: opts?.rightContext,
    });
}

/** sigmaMs → 卷积半径（帧）：调用方据此决定「平滑化」op 需要多取多少上下文帧。 */
export function smoothContextPadFrames(units: number, framePeriodMs: number): number {
    const unitsValue = Math.max(0, Number(units) || 0);
    if (unitsValue <= 0) return 0;
    const fpMs = Math.max(1e-6, Number(framePeriodMs) || 5);
    return Math.ceil((GAUSSIAN_CUTOFF_SIGMAS * smoothSigmaMsFromUnits(unitsValue)) / fpMs);
}

/**
 * 右键下拖：位移像素 → 平滑强度单位。**不设上限**：-50px = 100%，继续下拖
 * 按同一映射继续加深（σ 增大，直至 paramSmoothing 的硬上限兜底）。
 * 与弹窗百分比显示（`formatRightDragMorphPercent` = dy·2 %）保持一致：
 * units = -percent / 100。
 */
export function rightDragDownSmoothStrength(dragDelta: number): number {
    return Math.max(0, -dragDelta / 50);
}

/**
 * 右键上拖：位移像素 → 残差放大倍率（+2%/px，无上限）。+50px = ×2 倍。
 * 与 `formatRightDragMorphPercent`（dy·2 %）一致：scale = 1 + percent/100。
 */
export function rightDragUpScale(dragDelta: number): number {
    return Math.max(0, 1 + (dragDelta * 2) / 100);
}

export function transformSelectionByRightDrag(
    values: number[],
    editParam: ParamName,
    dragDelta: number,
    opts?: SmoothSelectionOptions,
): number[] {
    if (dragDelta >= 0) {
        const scale = rightDragUpScale(dragDelta);
        if (scale === 1) return values.slice();
        return amplifySelectionDeviation(values, editParam, scale, {
            framePeriodMs: opts?.framePeriodMs,
        });
    }
    return smoothSelectionValues(
        values,
        editParam,
        rightDragDownSmoothStrength(dragDelta),
        opts,
    );
}
