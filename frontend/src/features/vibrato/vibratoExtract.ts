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
import {
    refineNoteFrames,
    usesUnsetValue,
    vibratoNoteRuns,
    type VibratoNoteRun,
} from "./vibratoPitch";
import { createVibratoPresetId } from "./vibratoPresets";
import { vibratoSeedForPreset } from "./vibratoSeed";
import type { VibratoParamRange } from "./vibratoDepth";
import type { VibratoPreset, WaveShape } from "./vibratoTypes";

/** 识别失败的原因。 */
export type VibratoExtractFailure =
    /** 选区太短，放不下两个周期。 */
    | "tooShort"
    /** 残差里没有稳定的周期性成分。 */
    | "noVibrato"
    /**
     * 选区里没有可拟合的音符帧。
     *
     * 音高参数里 0 是"未检测"哨兵，浊清边界上还有跟踪器给的**低而非零**过渡帧 ——
     * 两类都不是音符（见 `vibratoPitch.ts` 的音符段判定）。整段都是这类帧时，
     * "再选长一点"与"这段不是颤音"都不对，用户需要的是知道这段里根本没有音高。
     */
    | "noPitch";

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
 * 折叠波形保留的谐波数上限（环形低通）。
 *
 * 【取值依据】真实颤音的周期形状几乎都是低次谐波主导：正弦 1 个，三角 / 梯形
 * 也就几个。保留到第 5 次谐波既能表达这些形状，又能滤掉折叠平均没消掉的观测
 * 噪声 —— 噪声主要落在高次谐波上，而它正是"坑坑洼洼"的来源。再往上放，提取出的
 * 波形就会把噪声一起存进去。
 */
const EXTRACT_MAX_HARMONICS = 5;

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

/**
 * 均值为 0 的残差。
 *
 * 【无数据帧（NaN）不参与趋势拟合】`valueFilter` 让 `smoothCurveGaussian` 跳过它们
 * 并且**不改写**这些帧（该函数本就是为"未浊帧不能混进平滑"写的，见 `paramSmoothing`
 * 的说明）；残差在无数据帧上直接记 0 —— 对自相关是中性项，对各项统计量则会被
 * `trusted` 掩码跳过。整条链路上没有任何一处需要伪造数据。
 */
function detrend(values: readonly number[], framePeriodMs: number): number[] {
    const list = Array.from(values, (value) => {
        const numeric = Number(value);
        return Number.isFinite(numeric) ? numeric : Number.NaN;
    });
    const baseline = smoothCurveGaussian(list, {
        sigmaMs: DETREND_SIGMA_MS,
        framePeriodMs,
        valueFilter: Number.isFinite,
    });
    return list.map((value, index) => {
        const center = baseline[index];
        return Number.isFinite(value) && Number.isFinite(center) ? value - center : 0;
    });
}

/**
 * 把**不是音符**的帧标成"无数据"（`NaN`）。
 *
 * 【为什么标 NaN 而不是删掉或插值】周期估计靠自相关，它要求**等间隔采样**：删帧
 * 之后"滞后 k 个样本"不再等于"k 帧"，测出的周期直接失去意义。插值补洞同样不行 ——
 * 补出来的直线与真实音高之间的落差会被去趋势放大成上千分的假残差（试过，周期估计
 * 因此彻底失效）。标成 NaN 之后：趋势拟合跳过它们（`valueFilter`），残差在那里记 0，
 * 于是对自相关是中性项、对统计量被掩码排除 —— 全链路没有一处需要伪造数据。
 *
 * 滑音 / 过渡段的剔除不在这里，而是复用 `vibratoPitch.ts` 的 {@link refineNoteFrames}：
 * "哪些帧算音符"只有一处定义，加颤音、提取预设、试听三条路径才不会各说各话。
 *
 * @param runs 可信段。`null` = 整段都是有效数据（非哨兵参数没有"未检测"这个概念）。
 */
function markNonNoteFrames(
    values: readonly number[],
    runs: readonly VibratoNoteRun[] | null,
): number[] {
    if (runs === null) return Array.from(values, (value) => Number(value));
    const trusted = new Array<boolean>(values.length).fill(false);
    for (const run of runs) {
        for (let i = run.startIndex; i < run.endIndex; i += 1) trusted[i] = true;
    }
    return values.map((value, index) => (trusted[index] ? value : Number.NaN));
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
function foldCycleTable(
    residual: readonly number[],
    periodFrames: number,
    bins: number,
    trusted?: readonly boolean[],
): number[] {
    const table = new Array<number>(bins).fill(0);
    const weights = new Array<number>(bins).fill(0);

    for (let i = 0; i < residual.length; i += 1) {
        // 非音符帧不参与：它们要么是插值出来的直线，要么根本就不是音高。
        if (trusted && !trusted[i]) continue;
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

/** 环形线性插值：把 `u ∈ [0,1)` 处的表值取出来（首尾相接）。 */
function sampleTableCircular(table: readonly number[], u: number): number {
    const n = table.length;
    if (n === 0) return 0;
    if (n === 1) return table[0];
    const wrapped = u - Math.floor(u);
    const pos = wrapped * n;
    const i0 = Math.floor(pos) % n;
    const i1 = (i0 + 1) % n;
    const frac = pos - Math.floor(pos);
    return table[i0] * (1 - frac) + table[i1] * frac;
}

/**
 * 环形低通：只保留最低的 `maxHarmonic` 次谐波（含直流）。
 *
 * 【为什么用谐波截断而不是滑动平均】滑动平均在边界要特殊处理，处理不好就在接缝
 * 处留下折角；而这里的数据本来就是**环形**的（一个周期首尾相接），用环形傅里叶
 * 截断既天然闭合，又是"理想低通"的定义本身 —— 想保留几个谐波是一个直观的旋钮，
 * 不必调窗口大小。
 */
function lowpassCycleTable(table: readonly number[], maxHarmonic: number): number[] {
    const n = table.length;
    if (n < 4) return [...table];
    const limit = Math.min(Math.floor((n - 1) / 2), Math.max(1, Math.round(maxHarmonic)));

    const out = new Array<number>(n).fill(0);
    // 直流分量。
    let mean = 0;
    for (const value of table) mean += value;
    mean /= n;
    out.fill(mean);

    for (let harmonic = 1; harmonic <= limit; harmonic += 1) {
        let cos = 0;
        let sin = 0;
        for (let i = 0; i < n; i += 1) {
            const angle = (2 * Math.PI * harmonic * i) / n;
            cos += table[i] * Math.cos(angle);
            sin += table[i] * Math.sin(angle);
        }
        const a = (2 * cos) / n;
        const b = (2 * sin) / n;
        for (let i = 0; i < n; i += 1) {
            const angle = (2 * Math.PI * harmonic * i) / n;
            out[i] += a * Math.cos(angle) + b * Math.sin(angle);
        }
    }
    return out;
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

    /*
     * 先按**音符帧**筛一遍：未检测帧与浊清边界的过渡帧不参与拟合。
     * 见 `prepareAnalysisSeries` 的说明（为什么补洞、哪些统计量只看真实帧）。
     */
    const runs = usesUnsetValue(input.param)
        ? vibratoNoteRuns(input.param, values, framePeriodMs)
        : null;
    if (runs && runs.length === 0) return { ok: false, reason: "noPitch" };
    const noteSeries = markNonNoteFrames(values, runs);
    // 音符帧判定 + 滑音剔除，与加颤音 / 试听同一套（`vibratoPitch.ts`）。
    const trusted = refineNoteFrames(noteSeries, framePeriodMs);
    if (!trusted.some(Boolean)) return { ok: false, reason: "noPitch" };
    /*
     * 用**最终**掩码再标一次：滑音帧必须从趋势拟合里也退场。只把它们排除出统计量是
     * 不够的 —— 它们仍会把高斯趋势拽偏，残差随之被污染，周期估计直接塌掉（实测：
     * 不标这一步，6 Hz 的输入被估成 98 帧 / 置信度 0.03）。
     */
    const series = noteSeries.map((value, index) => (trusted[index] ? value : Number.NaN));

    const residual = detrend(series, framePeriodMs);
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
    // 只统计**真实音符帧**：把补洞出来的插值帧算进去，等于用一条直线去稀释幅度，
    // 深度会被系统性低估（反过来，把未检测帧按 0 算则会抬成几十个半音的假深度）。
    const from = Math.floor(residual.length * 0.2);
    const to = Math.ceil(residual.length * 0.8);
    let sumSquares = 0;
    let count = 0;
    for (let i = from; i < to; i += 1) {
        if (!trusted[i]) continue;
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

    // 折叠出单周期波形（只折叠真实音符帧）。
    const bins = CYCLE_TABLE_DEFAULT_LEN;
    const folded = foldCycleTable(residual, estimate.periodFrames, bins, trusted);
    // 低通：折叠出的表仍带着观测噪声与逐周期抖动，直接存成波形会"坑坑洼洼"。
    // 真实颤音的周期形状由少数几个谐波就能表达（正弦只要 1 个，三角 / 梯形也就
    // 几个），截断高次谐波等于一次理想低通 —— 形状保留、毛刺消失，且因为是环形
    // 变换，首尾天然相接、不会在接缝处生出折角。
    const smoothed = lowpassCycleTable(folded, EXTRACT_MAX_HARMONICS);
    const normalized = normalizeTable(smoothed);
    const shapeHint = bestShapeMatch(normalized);

    // 不规则度：折叠时的周期内离散度相对幅值的比例（同样只看真实音符帧）。
    //
    // 参考值按**精确相位**在环形表上插值取得，而不是取最近的格子 —— 后者在相邻
    // 两格之间会引入最多半格的插值误差，那部分误差与"真实的逐周期抖动"混在一起，
    // 会把不规则度系统性抬高，进而让提取出的预设在应用时抖得比原曲线还厉害。
    let variance = 0;
    let varianceCount = 0;
    for (let i = 0; i < residual.length; i += 1) {
        if (!trusted[i]) continue;
        const phase = (i / estimate.periodFrames) % 1;
        const diff = residual[i] - sampleTableCircular(folded, phase);
        variance += diff * diff;
        varianceCount += 1;
    }
    const irregularity = Math.min(
        100,
        Math.round(
            (Math.sqrt(variance / Math.max(1, varianceCount)) / Math.max(1e-9, amplitude)) * 100,
        ),
    );

    // 包络：平滑后的 |残差|。窗长取半个周期，恰好滤掉周期内起伏、保留渐入渐出。
    //
    // 这里**不**排除非音符帧：补洞出来的插值段残差近乎为 0，于是气口处包络自然落到
    // 谷底，`envelopeEdgesMs` 取到的渐入 / 渐出正好落在真实音符的起止上 —— 比硬性
    // 排除更贴合"这段音是从哪开始、到哪结束"。
    const envelope = movingAverage(
        residual.map((value) => Math.abs(value)),
        (estimate.periodFrames * ENVELOPE_SMOOTH_PERIODS) / 2,
    );
    const edges = envelopeEdgesMs(envelope, framePeriodMs);

    const presetId = createVibratoPresetId();
    const preset: VibratoPreset = {
        id: presetId,
        name: "",
        builtin: false,
        cycle: { kind: "table", table: normalized },
        depthCents,
        // 按 Hz 存：提取出的音色要能在别的长度上复现同一个听感。`cycles` 只是
        // 顺手记下"这一段里大约有几个周期"，供读数参考（Hz 模式下不参与渲染）。
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
        seed: vibratoSeedForPreset({ id: presetId }),
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
