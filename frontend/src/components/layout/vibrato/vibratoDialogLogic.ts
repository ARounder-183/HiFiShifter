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
import { depthToDisplay, displayToDepth } from "../../../features/vibrato/vibratoDepth";
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

/**
 * 纵轴定标：按预设自身的幅度取整到一个"好看"的档位。
 *
 * 【为什么不按参数值域定标】音高的值域是几十个半音，30 cents 的颤音按那个
 * 尺度画出来就是一条直线。按预设自身幅度定标，5 分与 100 分的预设都看得清
 * 形状，真实幅度由读数负责表达。
 */
export function previewScaleCents(peakCents: number): number {
    const safe = Math.max(1, Number.isFinite(peakCents) ? peakCents : 1);
    return Math.ceil(safe * 1.15);
}
