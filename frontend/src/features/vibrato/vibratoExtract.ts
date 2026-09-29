/**
 * 从一段已有曲线里**反推**颤音预设。
 *
 * 【用途】用户手绘（或用别的工具导入）出一段自己满意的颤音后，不必再对着
 * 五个数字猜怎么复现它 —— 直接把这段曲线拟合成预设，再在预设编辑器里微调。
 * 这是"手动绘制 → 预设"的通路。
 *
 * 【算法顺序】去趋势 → 测周期 → 测深度 → 折叠出单周期波形 → 测包络与不规则度。
 * 每一步都只用前面的结果，因此任何一步失败都能给出**具体原因**而不是笼统的
 * "没识别出来"。
 *
 * 【为什么折叠出采样表而不是只挑一个形状】手绘的颤音往往不严格是正弦；把
 * 实际的一个周期平均下来存成 `table`，比强行套一个"最接近的形状"保真得多。
 * 同时仍然给出形状提示（相关性最高的参数式形状），用户想换成参数式时一键可切。
 */

import { smoothCurveGaussian } from "../../components/layout/pianoRoll/paramSmoothing";
import { CYCLE_TABLE_DEFAULT_LEN } from "./vibratoCycle";
import { depthFamilyOf, fallbackRangeFor, PITCH_PARAM_ID } from "./vibratoDepth";
import { createVibratoPresetId } from "./vibratoPresets";
import type { VibratoParamRange } from "./vibratoDepth";
import type { VibratoPreset, WaveShape } from "./vibratoTypes";

/** 识别失败的原因。 */
export type VibratoExtractFailure =
    /** 选区太短，放不下两个周期。 */
    | "tooShort"
    /** 残差里没有稳定的周期性成分。 */
    | "noVibrato";

export interface VibratoExtractInput {
    /** 选区内的曲线值（参数原生单位）。 */
    values: readonly number[];
    framePeriodMs: number;
    param: string;
    range?: VibratoParamRange;
}

export interface VibratoExtractSuccess {
    ok: true;
    /** 已填好全部参数、但**还没有 id 与名字**的预设（由调用方命名后入库）。 */
    preset: VibratoPreset;
    /** 测得的速率（Hz）。 */
    rateHz: number;
    /** 测得的深度（cents）。 */
    depthCents: number;
    /** 折叠出的周期表与参数式形状的相关性最高的那一个。 */
    shapeHint: WaveShape;
    /** 周期性置信度 0..1（自相关峰值）。 */
    confidence: number;
}

export interface VibratoExtractFailureResult {
    ok: false;
    reason: VibratoExtractFailure;
}

export type VibratoExtractResult = VibratoExtractSuccess | VibratoExtractFailureResult;

/** 可识别的速率区间（Hz）。人声颤音几乎不会落在 2 Hz 以下或 20 Hz 以上。 */
export const EXTRACT_MIN_HZ = 2;
export const EXTRACT_MAX_HZ = 20;
/** 低于此置信度视为"没有颤音"。 */
export const EXTRACT_MIN_CONFIDENCE = 0.35;
/** 至少要有这么多个周期才敢拟合（太短时周期估计的方差过大）。 */
export const EXTRACT_MIN_CYCLES = 2;
/** 去趋势用的高斯 sigma（ms）：只留下颤音，不要歌手原本的音高运动。 */
const DETREND_SIGMA_MS = 120;
/** 深度不足这个值就当作"没有颤音"（cents）。 */
const MIN_DEPTH_CENTS = 3;

/** 包络平滑的窗长（周期数）。 */
const ENVELOPE_SMOOTH_PERIODS = 0.5;

/**
 * 参数原生单位的**偏差**换算成 cents。
 *
 * 【为什么不能复用 `depthDisplayFactor`】那一族函数是"编辑器显示单位"的换算，
 * 而这里要的是"曲线偏差 → 规范深度"。两者对音高不同：曲线把音高按半音存
 * （偏差 × 100 = 分），而编辑器按分显示（因子 1）。
 */
export function deviationToCents(
    param: string,
    deviation: number,
    range?: VibratoParamRange,
): number {
    const family = depthFamilyOf(param);
    // 乘性增益的偏差就是倍率，1 倍率 = 100 分。
    if (family === "ratio") return deviation * 100;
    if (param === PITCH_PARAM_ID) return deviation * 100;
    if (family === "cents") {
        // 值本身是 cents 的参数：子轨音分偏移、共振峰偏移；音级参数名义 100 分/级。
        return param.includes("degrees") ? deviation * 100 : deviation;
    }
    // 原始值域：按半量程定标（与曲线合成时的换算互逆）。
    const effective = range ?? fallbackRangeFor(param) ?? { min: 0, max: 2 };
    const span = Number(effective.max) - Number(effective.min);
    const halfSpan = Number.isFinite(span) && span > 0 ? span / 2 : 1;
    return (deviation / halfSpan) * 100;
}

/**
 * 归一化自相关，返回 `lag` 帧处的相关系数近似值。
 *
 * 分母用 `sum(x[i]^2)`（全段能量）而不是逐 lag 归一化：颤音信号在窗口内近似
 * 平稳，这样实现更简单，且峰值的**相对**高低仍然可用。末端重叠部分不足一半
 * 的 lag 直接跳过，避免用几个样本算出虚假的高相关。
 */
function autocorrelation(residual: readonly number[], lag: number): number {
    const n = residual.length;
    if (lag <= 0 || lag >= Math.floor(n / 2)) return 0;
    let num = 0;
    let den = 0;
    for (let i = 0; i + lag < n; i += 1) {
        num += residual[i] * residual[i + lag];
    }
    for (let i = 0; i < n; i += 1) {
        den += residual[i] * residual[i];
    }
    return den > 1e-12 ? num / den : 0;
}

/** 周期估计结果。 */
export interface PeriodEstimate {
    /** 周期（帧）。 */
    periodFrames: number;
    /** 自相关峰值 0..1。 */
    confidence: number;
}

/**
 * 用自相关估计周期。
 *
 * 【为什么要处理倍频错误】自相关在 `2L`、`3L` 处同样有峰。取全局最大常常会
 * 选中两倍周期（听感上"慢了一半"）。这里的规则是：先取全局峰，再检查它的
 * **一半**处是否有足够强的局部峰（≥ 峰值的 0.8），有就取小的那个。
 */
export function estimateVibratoPeriod(
    residual: readonly number[],
    framePeriodMs: number,
): PeriodEstimate | null {
    const n = residual.length;
    const fp = framePeriodMs > 0 ? framePeriodMs : 5;
    const lagMin = Math.max(1, Math.round(1000 / (EXTRACT_MAX_HZ * fp)));
    const lagMax = Math.min(Math.floor(n / 2) - 1, Math.round(1000 / (EXTRACT_MIN_HZ * fp)));
    if (lagMax <= lagMin) return null;

    let bestLag = -1;
    let bestValue = 0;
    for (let lag = lagMin; lag <= lagMax; lag += 1) {
        const value = autocorrelation(residual, lag);
        if (value > bestValue) {
            bestValue = value;
            bestLag = lag;
        }
    }
    if (bestLag < 0) return null;

    // 倍频修正：全局峰的一半处若有强峰，取小的那个。
    const halfLag = Math.round(bestLag / 2);
    if (halfLag >= lagMin) {
        const halfValue = autocorrelation(residual, halfLag);
        if (halfValue >= bestValue * 0.8) {
            bestLag = halfLag;
            bestValue = halfValue;
        }
    }

    // 抛物线插值：整数 lag 的分辨率在 5 ms 帧栅格上就是 ~1 Hz 的误差。
    const prev = autocorrelation(residual, bestLag - 1);
    const cur = autocorrelation(residual, bestLag);
    const next = autocorrelation(residual, bestLag + 1);
    const denom = prev - 2 * cur + next;
    const offset = Math.abs(denom) > 1e-12 ? (0.5 * (prev - next)) / denom : 0;
    const refined = bestLag + Math.max(-0.5, Math.min(0.5, offset));

    return { periodFrames: refined, confidence: bestValue };
}

/** 均值为 0 的残差。 */
function detrend(values: readonly number[], framePeriodMs: number): number[] {
    const list = Array.from(values, (value) => Number(value) || 0);
    const baseline = smoothCurveGaussian(list, {
        sigmaMs: DETREND_SIGMA_MS,
        framePeriodMs,
    });
    return list.map((value, index) => value - (baseline[index] ?? value));
}

/** 长度可变的滑动平均（边界处窗口收缩）。 */
function movingAverage(values: readonly number[], halfWidth: number): number[] {
    const n = values.length;
    const out = new Array<number>(n);
    const width = Math.max(0, Math.round(halfWidth));
    for (let i = 0; i < n; i += 1) {
        const from = Math.max(0, i - width);
        const to = Math.min(n - 1, i + width);
        let sum = 0;
        for (let j = from; j <= to; j += 1) sum += values[j];
        out[i] = sum / (to - from + 1);
    }
    return out;
}

/**
 * 按周期把残差折叠成一个周期表。
 *
 * 【相位基准就是选区起点】不做零交叉对齐：把起点当作相位 0 直接折叠，生成的
 * 预设只要 `startPhaseDeg = 0` 就能原样复现观测到的起点。做对齐全额外引入一次
 * 相位估计误差，而用户要的是"复现我画的这段"。
 *
 * 【为什么用线性分摊而不是直接落格】周期帧数几乎总是不是整数（33.33 帧 @ 6 Hz），
 * 相位每帧前进的格数因此不是整数，直接落格会在表里留下**成片的空格子** ——
 * 那些格子被填 0 之后，相邻格之间会出现接近满幅的跳变，波形变成刺。把每个样本
 * 按小数部分分摊到相邻两格，就不存在空格子。
 */
function foldCycleTable(residual: readonly number[], periodFrames: number, bins: number): number[] {
    const table = new Array<number>(bins).fill(0);
    const weights = new Array<number>(bins).fill(0);

    for (let i = 0; i < residual.length; i += 1) {
        const phase = (i / periodFrames) % 1;
        const pos = phase * bins;
        const i0 = Math.floor(pos) % bins;
        const i1 = (i0 + 1) % bins;
        const frac = pos - Math.floor(pos);
        table[i0] += residual[i] * (1 - frac);
        weights[i0] += 1 - frac;
        table[i1] += residual[i] * frac;
        weights[i1] += frac;
    }

    for (let i = 0; i < bins; i += 1) table[i] = weights[i] > 1e-9 ? table[i] / weights[i] : 0;

    // 兜底：极端参数下仍可能有空格子（样本数少于格数）。用环形最近邻插值填上，
    // 而不是留 0 —— 留 0 会制造出上面说的那种满幅跳变。
    fillEmptyBinsCircular(table, weights);

    // 去掉直流：基调由预设的 baseline 承担，波形表只表达"围绕基线的摆动"。
    const mean = table.reduce((sum, value) => sum + value, 0) / bins;
    for (let i = 0; i < bins; i += 1) table[i] -= mean;
    return table;
}

/** 环形地用最近的有权重点填充空档（线性插值，跨越首尾相接处）。 */
function fillEmptyBinsCircular(table: number[], weights: readonly number[]): void {
    const n = table.length;
    if (n === 0) return;
    if (weights.every((weight) => weight > 1e-9)) return;

    for (let i = 0; i < n; i += 1) {
        if (weights[i] > 1e-9) continue;
        let back = 1;
        while (back < n && weights[(i - back + n) % n] <= 1e-9) back += 1;
        let forward = 1;
        while (forward < n && weights[(i + forward) % n] <= 1e-9) forward += 1;
        if (back >= n && forward >= n) continue;
        const before = table[(i - back + n) % n];
        const after = table[(i + forward) % n];
        table[i] = before + ((after - before) * back) / (back + forward);
    }
}

/** 把表归一化到 `[-1,1]`（按最大绝对值）。 */
function normalizeTable(table: readonly number[]): number[] {
    let peak = 0;
    for (const value of table) peak = Math.max(peak, Math.abs(value));
    if (peak <= 1e-12) return table.map(() => 0);
    return table.map((value) => value / peak);
}

/** 参数式形状的采样（与 `vibratoCycle.sampleCycle` 的约定一致：u=0 为上升零交叉）。 */
function parametricSample(shape: WaveShape, u: number): number {
    switch (shape) {
        case "sine":
            return Math.sin(2 * Math.PI * u);
        case "triangle": {
            const t = (u + 0.25) % 1;
            return t < 0.5 ? -1 + 4 * t : 3 - 4 * t;
        }
        case "sawUp":
            return 2 * ((u + 0.5) % 1) - 1;
        case "sawDown":
            return 1 - 2 * ((u + 0.5) % 1);
        case "square":
            return u < 0.5 ? 1 : -1;
        case "trapezoid":
        case "trill":
            return u < 0.5 ? 1 : -1;
        default:
            return Math.sin(2 * Math.PI * u);
    }
}

/** 与参数式形状做相关性比对，返回最接近的那个（仅供参考的形状提示）。 */
function bestShapeMatch(normalized: readonly number[]): WaveShape {
    const shapes: WaveShape[] = ["sine", "triangle", "sawUp", "sawDown", "square"];
    let best: WaveShape = "sine";
    let bestScore = -Infinity;
    for (const shape of shapes) {
        let dot = 0;
        let norm = 0;
        for (let i = 0; i < normalized.length; i += 1) {
            const reference = parametricSample(shape, i / normalized.length);
            dot += normalized[i] * reference;
            norm += reference * reference;
        }
        const score = norm > 1e-12 ? dot / Math.sqrt(norm) : 0;
        if (score > bestScore) {
            bestScore = score;
            best = shape;
        }
    }
    return best;
}

/** 包络上升 / 下降沿：达到峰值 90% 的时间（ms）。 */
function envelopeEdgesMs(
    envelope: readonly number[],
    framePeriodMs: number,
): {
    attackMs: number;
    releaseMs: number;
} {
    const n = envelope.length;
    const peak = Math.max(...envelope);
    if (peak <= 1e-12) return { attackMs: 0, releaseMs: 0 };
    const threshold = peak * 0.9;
    let first = -1;
    let last = -1;
    for (let i = 0; i < n; i += 1) {
        if (envelope[i] >= threshold) {
            if (first < 0) first = i;
            last = i;
        }
    }
    if (first < 0) return { attackMs: 0, releaseMs: 0 };
    return {
        attackMs: first * framePeriodMs,
        releaseMs: Math.max(0, (n - 1 - last) * framePeriodMs),
    };
}

/**
 * 从一段曲线里拟合颤音预设。
 *
 * 失败时给出具体原因（太短 / 没有稳定周期），而不是笼统的"失败" —— 用户需要
 * 知道是"再选长一点"还是"这段本来就不是颤音"。
 */
export function extractVibratoPreset(input: VibratoExtractInput): VibratoExtractResult {
    const values = input.values;
    const framePeriodMs = input.framePeriodMs > 0 ? input.framePeriodMs : 5;
    if (values.length < 8) return { ok: false, reason: "tooShort" };

    const residual = detrend(values, framePeriodMs);
    const estimate = estimateVibratoPeriod(residual, framePeriodMs);
    if (!estimate) return { ok: false, reason: "tooShort" };

    const periodMs = estimate.periodFrames * framePeriodMs;
    const durationMs = values.length * framePeriodMs;
    const cycleCount = durationMs / Math.max(1e-6, periodMs);
    if (cycleCount < EXTRACT_MIN_CYCLES) return { ok: false, reason: "tooShort" };
    if (estimate.confidence < EXTRACT_MIN_CONFIDENCE) return { ok: false, reason: "noVibrato" };

    const rateHz = 1000 / periodMs;
    if (!Number.isFinite(rateHz) || rateHz < EXTRACT_MIN_HZ || rateHz > EXTRACT_MAX_HZ) {
        return { ok: false, reason: "noVibrato" };
    }

    // 深度取中段（充分发展区）的 RMS：首尾的渐入渐出会把整段 RMS 拉低。
    const from = Math.floor(residual.length * 0.2);
    const to = Math.ceil(residual.length * 0.8);
    let sumSquares = 0;
    let count = 0;
    for (let i = from; i < to; i += 1) {
        sumSquares += residual[i] * residual[i];
        count += 1;
    }
    const rms = count > 0 ? Math.sqrt(sumSquares / count) : 0;
    // 正弦的 RMS = 幅值 / √2；非正弦形状会有偏差，但归一化后的表会把它补回来。
    const amplitude = rms * Math.SQRT2;
    const depthCents = Math.abs(deviationToCents(input.param, amplitude, input.range));
    if (!Number.isFinite(depthCents) || depthCents < MIN_DEPTH_CENTS) {
        return { ok: false, reason: "noVibrato" };
    }

    // 折叠出单周期波形。
    const bins = CYCLE_TABLE_DEFAULT_LEN;
    const folded = foldCycleTable(residual, estimate.periodFrames, bins);
    const normalized = normalizeTable(folded);
    const shapeHint = bestShapeMatch(normalized);

    // 不规则度：折叠时的周期内离散度相对幅值的比例。
    let variance = 0;
    for (let i = 0; i < residual.length; i += 1) {
        const phase = (i / estimate.periodFrames) % 1;
        const bin = Math.min(bins - 1, Math.max(0, Math.floor(phase * bins)));
        const diff = residual[i] - folded[bin];
        variance += diff * diff;
    }
    const irregularity = Math.min(
        100,
        Math.round(
            (Math.sqrt(variance / Math.max(1, residual.length)) / Math.max(1e-9, amplitude)) * 100,
        ),
    );

    // 包络：平滑后的 |残差|。窗长取半个周期，恰好滤掉周期内起伏、保留渐入渐出。
    const envelope = movingAverage(
        residual.map((value) => Math.abs(value)),
        (estimate.periodFrames * ENVELOPE_SMOOTH_PERIODS) / 2,
    );
    const edges = envelopeEdgesMs(envelope, framePeriodMs);

    const preset: VibratoPreset = {
        id: createVibratoPresetId(),
        name: "",
        builtin: false,
        cycle: { kind: "table", table: normalized },
        depthCents,
        // 用"整段周期数"而不是 Hz：用户提取的就是"这一段里的这几个周期"，
        // 换个长度时保持同一听感由预设编辑器里的 Hz 模式负责。
        rateMode: "hz",
        rateHz,
        cycles: cycleCount,
        rateRampEnd: 1,
        startPhaseDeg: 0,
        attackMs: Math.round(edges.attackMs),
        attackCurve: "exp",
        releaseMs: Math.round(edges.releaseMs),
        releaseCurve: "exp",
        depthRamp: { start: 1, end: 1 },
        alignCycles: false,
        irregularity,
        biasCents: 0,
        // 提取出来的是"一段曲线上的颤音"，套回曲线时自然应当叠在已有曲线上。
        baseline: "existing",
        blend: 100,
    };

    return {
        ok: true,
        preset,
        rateHz,
        depthCents,
        shapeHint,
        confidence: estimate.confidence,
    };
}
