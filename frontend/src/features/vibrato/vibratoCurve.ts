/**
 * 颤音曲线生成内核。
 *
 * 【与既有实现的关系】签名与返回形状沿用 `buildVibratoDense`
 * （`{ minF, maxF, dense }`），因此可以直接塞进 `applyDenseToLiveEdit`，
 * 调用方不需要改写入路径。
 *
 * 【本文件修掉的两个缺陷】
 *
 * 1. **速率不再是"整段周期数"**。历史实现里 `t` 是 0..1 的归一化进度，
 *    `sin(2π·f·t)` 中的 `f` 因此是"整段几个周期"—— 同一段颤音拖得越长，
 *    听感越慢（拖 0.2 s 得到 15 Hz，拖 1 s 得到 3 Hz）。这里改成**按秒累积
 *    相位**：
 *    ```
 *    φ[i] = φ[i-1] + 2π · f(t) · framePeriodMs / 1000
 *    ```
 *    必须累积而不能写成 `sin(2π·f(t)·t)` —— 变速率下后者会产生相位跳变，
 *    听感是周期性的"咔哒"。
 *
 * 2. **乘性参数不再用加性调制**。`dyn` / `volume` / `breath_gain` 都是增益，
 *    乘性调制下 `base = 0` 恒为 `0`，静音帧不会被抬起来。判定见
 *    `vibratoDepth.ts`。
 */

import { sampleCycle, wrap01 } from "./vibratoCycle";
import { clampDepthCentsForParam, depthMappingFor, type VibratoParamRange } from "./vibratoDepth";
import type { EnvelopeCurve, VibratoPreset } from "./vibratoTypes";

/** 默认的帧周期，与后端 `state.rs::default_frame_period_ms` 一致。 */
export const DEFAULT_FRAME_PERIOD_MS = 5;

/** 相位抖动的最大幅度（周期数）。`irregularity = 100` 时的峰值偏移。 */
const PHASE_JITTER_CYCLES = 0.14;
/** 深度抖动的最大比例。`irregularity = 100` 时的峰值相对变化。 */
const DEPTH_JITTER_RATIO = 0.35;
/** 两路噪声的时间尺度（每秒多少个噪声格）。越大抖动越"碎"。 */
const PHASE_NOISE_RATE = 2.6;
const DEPTH_NOISE_RATE = 1.9;

export interface VibratoRenderInput {
    /** 拖拽起点 / 终点的帧与值（`baseline` 非 `existing` 时作为基线端点）。 */
    startFrame: number;
    startValue: number;
    endFrame: number;
    endValue: number;
    /**
     * 整段原值（按帧索引，`original[k]` 对应 `minF + k`）。
     * `baseline === "existing"` 时必需；`blend < 100` 时也必需。
     */
    original?: ArrayLike<number>;
    preset: VibratoPreset;
    param: string;
    framePeriodMs?: number;
    range?: VibratoParamRange;
    /**
     * 逐帧吸附，作用于**合成后的值**。
     *
     * 【保留的设计】拖拽路径开着吸附时会把整条曲线量化到半音 / 音阶格 ——
     * 这是故意的：用户可以用它快速画出一条量化的参数线。因此这里保持与
     * 历史实现完全一致的语义（对最终值吸附），不做"只吸附基线"的改动。
     * 菜单 / 对话框路径不传此回调，行为与历史一致（不吸附）。
     */
    snapFinalValue?: (value: number, frame: number) => number;
    /** 不规则度的确定性种子。预览与提交必须传同一个值，否则波形会跳。 */
    seed?: number;
    /**
     * 一并返回逐帧的深度包络（cents，恒非负）。
     *
     * 【为什么是 opt-in】拖拽预览每帧都重建整段曲线，多一个等长数组就是每帧
     * 多一次分配；而只有预设编辑器的波形预览需要把包络画出来。默认关闭。
     */
    collectEnvelope?: boolean;
}

export interface VibratoRenderResult {
    minF: number;
    maxF: number;
    /** `dense[k]` 对应帧 `minF + k`。 */
    dense: number[];
    /** 仅在 `collectEnvelope` 为 true 时给出：`|depthCents| * env`，逐帧对应 `dense`。 */
    envelope?: number[];
}

/** 整数哈希 → `[0,1)`。确定性、无浮点、跨调用稳定。 */
function hash01(n: number): number {
    let x = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b);
    x ^= x >>> 13;
    x = Math.imul(x, 0xc2b2ae35);
    x ^= x >>> 16;
    return (x >>> 0) / 4294967296;
}

/**
 * 一维值噪声，`[-1,1]`。
 *
 * 【为什么不用白噪声】逐帧独立的白噪声听起来是"嗡嗡"的宽带噪声；用 smoothstep
 * 在整点之间插值得到的是**连续起伏**，才是"摇曳"的听感。
 */
function valueNoise(x: number, seed: number): number {
    const i = Math.floor(x);
    const frac = x - i;
    const a = hash01(i + seed * 1013);
    const b = hash01(i + 1 + seed * 1013);
    const t = frac * frac * (3 - 2 * frac);
    return (a + (b - a) * t) * 2 - 1;
}

/** 包络段的形状塑形：输入 `0..1`，输出 `0..1`。 */
function shapeProgress(x: number, curve: EnvelopeCurve): number {
    const c = Math.min(1, Math.max(0, x));
    switch (curve) {
        case "exp":
            // 慢起：开头几乎不动，后段快速上升 —— 最接近真人颤音的起振。
            return c * c;
        case "s":
            return c * c * (3 - 2 * c);
        default:
            return c;
    }
}

/** 解析预设的有效速率（Hz）。 */
function effectiveRateHz(preset: VibratoPreset, durationSec: number): number {
    if (preset.rateMode === "cycles") {
        const cycles = Number.isFinite(preset.cycles) ? Math.max(0, preset.cycles) : 1;
        return durationSec > 1e-9 ? cycles / durationSec : 0;
    }
    return Number.isFinite(preset.rateHz) ? Math.max(0, preset.rateHz) : 0;
}

/** 取基线：`baseline` 模式决定颤音围绕什么摆动。 */
function baselineAt(
    preset: VibratoPreset,
    t: number,
    i: number,
    input: VibratoRenderInput,
): number {
    switch (preset.baseline) {
        case "holdStart":
            return input.startValue;
        case "holdEnd":
            return input.endValue;
        case "average":
            return (input.startValue + input.endValue) / 2;
        case "existing": {
            const original = input.original;
            const value = original ? Number(original[i]) : Number.NaN;
            return Number.isFinite(value)
                ? value
                : input.startValue + (input.endValue - input.startValue) * t;
        }
        default:
            return input.startValue + (input.endValue - input.startValue) * t;
    }
}

/**
 * 生成一段颤音曲线。
 *
 * 逐帧顺序：基线 → 累积相位（含速率渐变与相位抖动）→ 深度包络（渐入 / 渐强 /
 * 渐出，含深度抖动）→ 波形 → 合成 → 干湿混合 → 吸附。
 */
export function buildVibratoCurve(input: VibratoRenderInput): VibratoRenderResult {
    const { preset, param } = input;

    const minF = Math.min(input.startFrame, input.endFrame);
    const maxF = Math.max(input.startFrame, input.endFrame);
    const len = maxF - minF + 1;
    const span = input.endFrame - input.startFrame;
    const fp =
        Number.isFinite(input.framePeriodMs) && (input.framePeriodMs as number) > 0
            ? (input.framePeriodMs as number)
            : DEFAULT_FRAME_PERIOD_MS;

    const dense = new Array<number>(len);
    if (len <= 0) return { minF, maxF, dense };

    // 首帧与末帧之间的时长。用 `len - 1` 而不是 `len`，是为了让
    // `rateMode: "cycles"` 的 N 恰好落在首末两帧之间（"整段 N 个周期"的
    // 字面语义），`alignCycles` 的定标也才有同一个基准。
    const spanSec = (Math.max(1, len - 1) * fp) / 1000;
    const totalMs = Math.max(1, len - 1) * fp;
    const baseRateHz = effectiveRateHz(preset, spanSec);
    const rateRampEnd = Number.isFinite(preset.rateRampEnd) ? preset.rateRampEnd : 1;
    const durationSec = spanSec;

    const mapping = depthMappingFor(param, input.range);
    // 深度按**当前参数**的满摆幅钳住：预设是跨参数共用的，同一个 300 分落在
    // 声像（±1）或共振峰（±500）上远超其可表达范围，不钳的话写入口会把超出的
    // 部分钳平，波形顶部变成一条直线 —— 用户拉到 300 分并不"更颤"，只是变成
    // 方波。见 `fullSwingCentsFor`。
    const depthCents = clampDepthCentsForParam(
        Number.isFinite(preset.depthCents) ? preset.depthCents : 0,
        param,
        input.range,
    );
    const biasCents = Number.isFinite(preset.biasCents) ? preset.biasCents : 0;
    const bias = biasCents * mapping.factor;

    const irr = Math.min(
        1,
        Math.max(0, (Number.isFinite(preset.irregularity) ? preset.irregularity : 0) / 100),
    );
    const seed = Number.isFinite(input.seed) ? (input.seed as number) : 0;

    const attackMs = Math.max(0, Number.isFinite(preset.attackMs) ? preset.attackMs : 0);
    const releaseMs = Math.max(0, Number.isFinite(preset.releaseMs) ? preset.releaseMs : 0);
    const rampStart = Number.isFinite(preset.depthRamp?.start) ? preset.depthRamp.start : 1;
    const rampEnd = Number.isFinite(preset.depthRamp?.end) ? preset.depthRamp.end : 1;

    const phaseOffset = Number.isFinite(preset.startPhaseDeg) ? preset.startPhaseDeg / 360 : 0;

    // ---- 第一趟：累积相位 -------------------------------------------------
    // 单独一趟而不是与合成合并，因为「对齐整数周期」需要先知道末帧相位才能定标。
    //
    // 进度一律用**区域相对**（沿帧序号递增），而不是拖拽方向相对的 `t`：
    // 反向拖拽（从右往左画）应当得到与正向完全相同的曲线，只有基线端点
    // 的归属是方向敏感的（见 `baselineAt`）。
    const denom = len <= 1 ? 1 : len - 1;
    const phases = new Float64Array(len);
    let phase = 0;
    for (let i = 0; i < len; i += 1) {
        phases[i] = phase;
        const tc = i / denom;
        const instRate = baseRateHz * (1 + (rateRampEnd - 1) * tc);
        phase += (2 * Math.PI * instRate * fp) / 1000;
    }

    // `alignCycles`：把整段相位线性缩放到整数圈，让音符干净地落回基准线。
    // 以**末帧**相位为基准，使首帧（相位 0）与末帧都落在同一零交叉上。
    let phaseScale = 1;
    const lastPhase = phases[len - 1];
    if (preset.alignCycles && lastPhase > 1e-9) {
        const totalCycles = lastPhase / (2 * Math.PI);
        const rounded = Math.max(1, Math.round(totalCycles));
        phaseScale = rounded / totalCycles;
    }

    // ---- 第二趟：合成 -----------------------------------------------------
    const blend = Number.isFinite(preset.blend)
        ? Math.min(100, Math.max(0, preset.blend)) / 100
        : 1;
    const useOriginalBlend = blend < 1 && Boolean(input.original);
    const snap = input.snapFinalValue;
    const envelope = input.collectEnvelope ? new Array<number>(len) : undefined;

    for (let i = 0; i < len; i += 1) {
        const frame = minF + i;
        // `t` 方向敏感（基线端点归属），`tc` 区域相对（速率渐变 / 渐强 / 噪声）。
        const t = span === 0 ? 1 : (frame - input.startFrame) / span;
        const tc = i / denom;

        const base = baselineAt(preset, t, i, input);

        // 相位抖动：让周期长短不一，像真人而不是机器。
        const phaseJitter =
            irr * PHASE_JITTER_CYCLES * valueNoise(tc * PHASE_NOISE_RATE * durationSec, seed);
        const u = wrap01((phases[i] * phaseScale) / (2 * Math.PI) + phaseOffset + phaseJitter);
        const wave = sampleCycle(preset.cycle, u);

        // 深度包络：渐入 → 渐强 → 渐出。渐入 / 渐出各自以**整段时长**为上限 ——
        // 预设里存的是绝对毫秒，落到具体选区上被选区长度封顶，于是"渐入拉满"就是
        // 整条线由弱到强。
        //
        // 【两者重叠时相乘】它们各是一条独立的增益斜坡，串联起来自然是乘积
        //（与音频里两级推子串联同理）：两边都拉满时中间会凹下去，而不是突然截断。
        // 保持乘法而不是"按比例压到刚好相接"，是因为手柄的位置必须始终等于斜坡的
        // 起点/终点 —— 归一化会让另一个手柄在用户拖这一个时自己动起来。
        const tMs = i * fp;
        const maxSpan = totalMs;
        let env = 1;
        const atk = Math.min(attackMs, maxSpan);
        const rel = Math.min(releaseMs, maxSpan);
        if (atk > 0 && tMs < atk) env *= shapeProgress(tMs / atk, preset.attackCurve);
        if (rel > 0 && tMs > totalMs - rel) {
            env *= shapeProgress((totalMs - tMs) / rel, preset.releaseCurve);
        }
        env *= rampStart + (rampEnd - rampStart) * tc;
        env *=
            1 +
            irr * DEPTH_JITTER_RATIO * valueNoise(tc * DEPTH_NOISE_RATE * durationSec, seed + 7);
        env = Math.max(0, env);
        // 包络恒为非负：深度可能为负（波形反相），取绝对值后包络带仍是上下对称的
        // 幅度边界，不会被负号翻到基线下方。
        if (envelope) envelope[i] = Math.abs(depthCents) * env;

        const delta = depthCents * mapping.factor * env * wave;

        let value: number;
        if (mapping.mode === "multiplicative") {
            // 增益类：乘性调制。`base = 0`（静音）恒为 0，负半周钳到 0。
            value = Math.max(0, base * (1 + delta));
        } else {
            value = base + bias + delta;
        }

        if (useOriginalBlend) {
            const original = Number(input.original?.[i]);
            if (Number.isFinite(original)) value = original + (value - original) * blend;
        }

        dense[i] = snap ? snap(value, frame) : value;
    }

    return envelope ? { minF, maxF, dense, envelope } : { minF, maxF, dense };
}

/**
 * 估算一段预设落在给定参数上的「有效周期数」。
 *
 * 供 HUD 与预设编辑器显示（"这一段大约 6 个周期"），也供
 * `alignCycles` 的说明文案使用。
 */
export function estimateCycles(
    preset: VibratoPreset,
    frameCount: number,
    framePeriodMs: number,
): number {
    const fp = framePeriodMs > 0 ? framePeriodMs : DEFAULT_FRAME_PERIOD_MS;
    const spanSec = (Math.max(1, Math.max(0, frameCount) - 1) * fp) / 1000;
    if (spanSec <= 1e-9) return 0;
    const rate = effectiveRateHz(preset, spanSec);
    // 速率渐变下取平均速率（线性渐变时均值即两端平均）。
    const avgRate =
        rate * (1 + ((Number.isFinite(preset.rateRampEnd) ? preset.rateRampEnd : 1) - 1) / 2);
    const total = avgRate * spanSec;
    return preset.alignCycles ? Math.max(1, Math.round(total)) : total;
}
