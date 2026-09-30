/**
 * 颤音预设编辑器的纯逻辑。
 *
 * 抽出来的理由与 `customScaleDialogLogic.ts` 相同：命名映射、单位换算与预览
 * 采样都是可单测的纯函数，混在组件里就只能靠肉眼验证。
 *
 * 【为什么标签映射写成显式 Record】`keyReferenceIntegrity.test.ts` 要求
 * `labelKey` 之类的字面量必须真实存在于词条中，`` `vibrato_shape_${shape}` ``
 * 这类模板拼接过不了校验；显式表也顺带让"新增一个波形要补哪些词条"一目了然。
 */

import type { MessageKey } from "../../../i18n/messages";
import {
    BUILTIN_VIBRATO_PRESET_DESC_KEYS,
    BUILTIN_VIBRATO_PRESET_NAME_KEYS,
    type BuiltinVibratoId,
} from "../../../features/vibrato/systemPresets";
import { isBuiltinVibratoPresetId } from "../../../features/vibrato/vibratoPresets";
import { buildVibratoCurve, DEFAULT_FRAME_PERIOD_MS } from "../../../features/vibrato/vibratoCurve";
import {
    depthStepUnitFor,
    depthToDisplay,
    displayToDepth,
    paramUnitToDepth,
    type VibratoParamRange,
} from "../../../features/vibrato/vibratoDepth";
import type {
    BaselineMode,
    EnvelopeCurve,
    VibratoPreset,
    VibratoRateMode,
    WaveShape,
} from "../../../features/vibrato/vibratoTypes";

/** 翻译函数（`t` 的签名子集）。 */
export type Translate = (key: MessageKey) => string;

/** 波形形状的显示名。 */
export const WAVE_SHAPE_KEYS: Record<WaveShape, MessageKey> = {
    sine: "vibrato_shape_sine",
    triangle: "vibrato_shape_triangle",
    sawUp: "vibrato_shape_saw_up",
    sawDown: "vibrato_shape_saw_down",
    square: "vibrato_shape_square",
    trapezoid: "vibrato_shape_trapezoid",
    trill: "vibrato_shape_trill",
};

/** 包络曲线的显示名。 */
export const ENVELOPE_CURVE_KEYS: Record<EnvelopeCurve, MessageKey> = {
    linear: "vibrato_curve_linear",
    exp: "vibrato_curve_exp",
    s: "vibrato_curve_s",
};

/** 基线模式的显示名。 */
export const BASELINE_MODE_KEYS: Record<BaselineMode, MessageKey> = {
    line: "vibrato_baseline_line",
    holdStart: "vibrato_baseline_hold_start",
    holdEnd: "vibrato_baseline_hold_end",
    average: "vibrato_baseline_average",
    existing: "vibrato_baseline_existing",
};

/** 速率模式的显示名。 */
export const RATE_MODE_KEYS: Record<VibratoRateMode, MessageKey> = {
    hz: "vibrato_rate_mode_hz",
    cycles: "vibrato_rate_mode_cycles",
};

/** 波形形状的枚举顺序（与下拉框一致）。 */
export const WAVE_SHAPE_ORDER: readonly WaveShape[] = [
    "sine",
    "triangle",
    "sawUp",
    "sawDown",
    "square",
    "trapezoid",
    "trill",
];

/** 基线模式的枚举顺序。 */
export const BASELINE_MODE_ORDER: readonly BaselineMode[] = [
    "line",
    "holdStart",
    "holdEnd",
    "average",
    "existing",
];

/** 包络曲线的枚举顺序。 */
export const ENVELOPE_CURVE_ORDER: readonly EnvelopeCurve[] = ["linear", "exp", "s"];

/** 预设的显示名：系统预设走词条，用户预设用用户输入的名字。 */
export function vibratoPresetLabel(preset: VibratoPreset, t: Translate): string {
    const builtinKey = builtinIdOf(preset.id);
    if (builtinKey) return t(BUILTIN_VIBRATO_PRESET_NAME_KEYS[builtinKey]);
    return preset.name.trim();
}

/** 预设的一行说明；用户预设没有说明，返回 `undefined`。 */
export function vibratoPresetDescription(preset: VibratoPreset, t: Translate): string | undefined {
    const builtinKey = builtinIdOf(preset.id);
    return builtinKey ? t(BUILTIN_VIBRATO_PRESET_DESC_KEYS[builtinKey]) : undefined;
}

/** `builtin.<key>` → `<key>`；不是系统预设或键未知时返回 `undefined`。 */
export function builtinIdOf(id: string): BuiltinVibratoId | undefined {
    if (!isBuiltinVibratoPresetId(id)) return undefined;
    const key = id.slice(id.indexOf(".") + 1);
    return key in BUILTIN_VIBRATO_PRESET_NAME_KEYS ? (key as BuiltinVibratoId) : undefined;
}

/**
 * 预设的一行摘要（列表行 / 选择器 / HUD 共用）。
 *
 * 形如 `正弦 · 30 分 · 5.5 Hz`；按周期数模式时把频率换成周期数。
 */
export function vibratoPresetSummary(preset: VibratoPreset, t: Translate): string {
    const shapeKey =
        preset.cycle.kind === "table" ? undefined : WAVE_SHAPE_KEYS[preset.cycle.shape];
    const rate =
        preset.rateMode === "cycles"
            ? `${formatNumber(preset.cycles)} ${t("vibrato_unit_cycles")}`
            : `${formatNumber(preset.rateHz)} ${t("vibrato_unit_hz")}`;
    const parts = [
        shapeKey ? t(shapeKey) : t("vibrato_from_selection"),
        `${formatNumber(preset.depthCents)} ${t("vibrato_unit_cents")}`,
        rate,
    ];
    return parts.join(" · ");
}

/** 数值的紧凑显示：去掉无意义的小数尾巴。 */
export function formatNumber(value: number): string {
    if (!Number.isFinite(value)) return "0";
    return String(Math.round(value * 100) / 100);
}

/**
 * 把 `depthCents` 换算成当前参数在**编辑器 / HUD 上显示的单位**。
 *
 * 【与曲线换算不同】曲线把音高按半音存，编辑器按分显示 —— 用户在音高上应当
 * 看到 `30`（分）而不是 `0.3`（半音）。用 `depthToDisplay` 而不是
 * `depthToParamUnit`，否则音高上会差 100 倍。
 */
export function depthForParam(
    depthCents: number,
    param: string,
    range?: { min: number; max: number },
): number {
    return depthToDisplay(depthCents, param, range);
}

/** {@link depthForParam} 的逆运算，用于把编辑器里输入的数值写回预设。 */
export function depthToCents(
    value: number,
    param: string,
    range?: { min: number; max: number },
): number {
    return displayToDepth(value, param, range);
}

/**
 * 深度显示值的**单位标签**（`null` = 原始值域参数，不带单位后缀）。
 *
 * 【为什么必须按参数取】`depthForParam` 已经把深度换算成参数原生单位，
 * 标签必须跟着变：在 `dyn` 上显示 "30 cents" 是把百分比说成了音分。单位
 * 由 `depthStepUnitFor` 决定（它同时决定编辑器里滚轮一格走多少），两者
 * 从同一个来源取，不会各自漂移。
 */
export function depthUnitLabelKey(param: string): MessageKey | null {
    switch (depthStepUnitFor(param)) {
        case "cents":
            return "vibrato_unit_cents";
        case "percent":
            return "vibrato_unit_percent";
        case "scaleDegree":
            return "vibrato_unit_degree";
        default:
            return null;
    }
}

/** 预设编辑器波形预览的几何。 */
export interface VibratoPreviewGeometry {
    /** 采样点数（横向）。 */
    frameCount: number;
    /** 名义帧周期（仅用于把"整段时长"算出来，与真实工程无关）。 */
    framePeriodMs: number;
}

/** 预览默认窗口：约 1.6 秒，足以看清渐入、渐强与若干个周期。 */
export const PREVIEW_DEFAULT: VibratoPreviewGeometry = {
    frameCount: 320,
    framePeriodMs: DEFAULT_FRAME_PERIOD_MS,
};

/** 预览采样结果，单位一律是 **cents**（与参数无关，便于画布定标）。 */
export interface VibratoPreviewSamples {
    /** 逐点波形（cents，围绕 0 摆动）。 */
    wave: number[];
    /** 逐点包络上界（cents，恒非负）—— 渐入 / 渐强 / 渐出入画就靠它。 */
    envelope: number[];
    /** 纵轴半幅（cents）：波形与包络的绝对值上界，至少为 1 以免除零。 */
    peakCents: number;
    /**
     * 套用前的原曲线（cents，与 `wave` 同轴）。
     *
     * 仅"套用到选区"预览提供：管理器预览回答"波形长什么样"，这里回答
     * "套到这段上长什么样"，两条线并置才看得出颤音叠在哪条运动之上。
     */
    original?: number[];
}

/**
 * 生成预设的预览采样。
 *
 * 走的是与拖拽、落盘**完全相同**的 `buildVibratoCurve`（参数取 `pitch`，基线
 * 恒为 0，因此输出值 × 100 即 cents）—— 预览画的就是用户真正会得到的波形，
 * 而不是另写一份"看起来像"的公式。
 */
export function buildVibratoPreview(
    preset: VibratoPreset,
    geometry: VibratoPreviewGeometry = PREVIEW_DEFAULT,
): VibratoPreviewSamples {
    const frameCount = Math.max(2, Math.round(geometry.frameCount));
    const result = buildVibratoCurve({
        startFrame: 0,
        startValue: 0,
        endFrame: frameCount - 1,
        endValue: 0,
        // 强制 `line` 基线：预览要展示波形本身，`existing` 在没有原曲线时
        // 会退化成端点插值，画出来与用户看到的不一致。
        preset: { ...preset, baseline: "line", blend: 100 },
        param: "pitch",
        framePeriodMs: geometry.framePeriodMs,
        collectEnvelope: true,
    });

    const wave = result.dense.map((value) => value * 100);
    const envelope = (result.envelope ?? new Array(frameCount).fill(0)).slice();
    let peak = Math.max(1, ...envelope.map((value) => Math.abs(value)));
    peak = Math.max(peak, ...wave.map((value) => Math.abs(value)));
    return { wave, envelope, peakCents: peak };
}

/** "套用到选区"预览的采样结果：波形 + 原曲线 + 包络，全部同轴（cents）。 */
export interface VibratoAppliedPreview extends VibratoPreviewSamples {
    /** 套用前的原曲线（cents，已减去中心）。 */
    original: number[];
}

/**
 * 生成"套用到选区"的预览采样。
 *
 * 【与管理器预览的区别】管理器预览拿 `buildVibratoPreview`（无原曲线、基线强制
 * `line`），回答"波形长什么样"；这里喂入选区的**真实帧值**，走同一条
 * `buildVibratoCurve`，回答"套到这段上长什么样" —— 原曲线被保留（`baseline:
 * "existing"`）还是被拉直，一眼可见。
 *
 * 两条曲线都换算成 cents 并减去原曲线均值再画：否则音高上整段的上行运动会把
 * 颤音挤出画面（绝对值域是几十个半音）。减中心之后"摆动"始终居中可辨。
 *
 * @returns 原值不足两点（无数据）时返回 `null`，由调用方显示占位提示。
 */
export function buildAppliedPreview(args: {
    preset: VibratoPreset;
    original: readonly number[];
    param: string;
    framePeriodMs: number;
    range?: VibratoParamRange;
}): VibratoAppliedPreview | null {
    const values = args.original.map((value) => (Number.isFinite(value) ? Number(value) : 0));
    if (values.length < 2) return null;

    const framePeriodMs =
        Number.isFinite(args.framePeriodMs) && args.framePeriodMs > 0
            ? args.framePeriodMs
            : DEFAULT_FRAME_PERIOD_MS;

    const result = buildVibratoCurve({
        startFrame: 0,
        startValue: values[0],
        endFrame: values.length - 1,
        endValue: values[values.length - 1],
        original: values,
        preset: args.preset,
        param: args.param,
        framePeriodMs,
        range: args.range,
        collectEnvelope: true,
    });

    const toCents = (value: number) => paramUnitToDepth(value, args.param, args.range);
    const originalCents = values.map(toCents);
    const center = originalCents.reduce((sum, value) => sum + value, 0) / originalCents.length;
    const original = originalCents.map((value) => value - center);
    const wave = result.dense.map((value) => toCents(value) - center);
    const envelope = (result.envelope ?? new Array(values.length).fill(0)).map((value) =>
        Math.abs(value),
    );

    let peak = 1;
    for (const value of original) peak = Math.max(peak, Math.abs(value));
    for (const value of wave) peak = Math.max(peak, Math.abs(value));
    for (const value of envelope) peak = Math.max(peak, Math.abs(value));
    return { wave, envelope, original, peakCents: peak };
}

/**
 * 纵轴定标：按预设自身的幅度取整到一个"好看"的档位。
 *
 * 【为什么不按参数值域定标】音高的值域是几十个半音，30 cents 的颤音按那个
 * 尺度画出来就是一条直线。按预设自身幅度定标，5 分与 100 分的预设都看得清
 * 形状，真实幅度由读数负责表达。
 *
 * 【注意】这是**一次性拟合**用的（见 `fitPreviewRangeCents`），不是每帧跟着
 * 当前值走的自适应标尺 —— 后者会让波形永远填满画布，深度变化只表现为"抖动"，
 * 用户无从判断幅度大小。
 */
export function previewScaleCents(peakCents: number): number {
    const safe = Math.max(1, Number.isFinite(peakCents) ? peakCents : 1);
    return Math.ceil(safe * 1.15);
}

/**
 * 预览纵轴的"好看"档位（半幅，cents）。
 *
 * 取值成阶梯而不是连续值：同一档位下不同预设的波形高度可以直接互相比较；跨档位
 * 时轴上的刻度标签会跟着变，读数不会失真。
 */
export const PREVIEW_RANGE_LADDER: readonly number[] = [
    5, 10, 20, 25, 50, 100, 200, 300, 500, 800, 1200, 2000, 3000, 5000, 8000,
];

/**
 * 由一段波形的峰值拟合预览纵轴半幅。
 *
 * 只比峰值大 15% 再向上取到最近的档位，于是波形通常占据画布的六成上下 ——
 * 既看得清形状，又留得下"再深一点"的余地。**只在打开 / 换预设 / 点「适应」时
 * 调用**：编辑期间标尺保持不动，波形高度才等于深度，用户才能直观判断大小。
 */
export function fitPreviewRangeCents(peakCents: number): number {
    const safe = Math.abs(Number.isFinite(peakCents) ? peakCents : 0);
    const target = safe * 1.15;
    for (const step of PREVIEW_RANGE_LADDER) {
        if (target <= step) return step;
    }
    return PREVIEW_RANGE_LADDER[PREVIEW_RANGE_LADDER.length - 1];
}

/** 缩略图专用采样数：64 点足够表达形状，path 缓存也便宜。 */
export const GLYPH_FRAME_COUNT = 64;

/**
 * 由预设求缩略图的折线点（`x0,y0 x1,y1 …`，y 向下为正）。
 *
 * 抽成纯函数以便单测：归一化方式（按自身峰值）、首末点、空输入的兜底都在这里。
 * 定标与 `previewScaleCents` 同理 —— 选择场景里"形状可辨"优先于"深度可比"。
 */
export function glyphPath(preset: VibratoPreset, width: number, height: number): string {
    const { wave } = buildVibratoPreview(preset, {
        frameCount: GLYPH_FRAME_COUNT,
        framePeriodMs: 5,
    });
    let peak = 1e-9;
    for (const value of wave) peak = Math.max(peak, Math.abs(value));

    const mid = height / 2;
    const reach = height / 2 - 1;
    const step = width / Math.max(1, wave.length - 1);
    const points: string[] = [];
    for (let i = 0; i < wave.length; i += 1) {
        const x = i * step;
        const y = mid - (wave[i] / peak) * reach;
        points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }
    return points.join(" ");
}
