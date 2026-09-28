/**
 * 步长策略 —— 滚轮/方向键调值的**唯一**取值来源。
 *
 * 【为什么需要它】审查发现同一个概念在不同控件里步长不同：BPM 在
 * `ActionBar` / `TempoMapRulerRow` / `ClipRateEditorDialog` 都是 1/0.1，
 * 但 `MidiTrackSelectDialog` 的 BPM 没有精细调整；滑块 `step={1}` 而滚轮粗调
 * 步长是 5（拖动与滚轮手感不一致）。
 *
 * 根因与排版那一轮相同：每个作者自己挑一个数字。这里把步长按**单位语义**
 * 固定下来，控件只说"这是 BPM"，不说"每次滚 1"。
 *
 * 【为什么 coarse/fine 成对给】精细调整修饰键（默认 Ctrl）的语义是"更细一档"，
 * 若只在个别控件实现，用户按 Ctrl 滚轮时行为就会时有时无。成对定义让
 * "有没有精细调整"不再是可选项。
 */
import type { ActionId } from "../features/keybindings/types";

/**
 * 单位语义。
 *
 * 按"一个刻度代表什么"分类，而不是按控件分类 —— 这样同一个量在应用的任何
 * 位置都拿到同一个步长。
 */
export type StepUnit =
    /** 每分钟节拍数。粗调 1 BPM，精调 0.1 BPM */
    | "bpm"
    /** 音分。粗调 1，精调 0.1 */
    | "cents"
    /** 半音 / 音级。整数 */
    | "semitone"
    /** 百分比（0–100），取整。粗调 5，精调 1 —— 用于音量、平滑度这类"整格"量 */
    | "percent"
    /**
     * 需要小数精度的百分比。粗调 1，精调 0.1，保留 2 位。
     *
     * 【为什么与 percent 分开】声道容差（设计上要 6 位小数精度）与分转换时长
     * 百分比（下界 0.01）都是**小数**百分比。若强行套用取整的 percent，滚轮一动
     * 就会把 0.35% 变成 1% —— 那是改数据，不只是改步长。
     */
    | "percentFine"
    /** 增益分贝。粗调 0.5，精调 0.1 */
    | "gainDb"
    /** 电平/阈值分贝（范围大、精度要求低）。粗调 3，精调 1 */
    | "levelDb"
    /** 播放速率倍率。粗调 0.1，精调 0.01 */
    | "rate"
    /** 毫秒。粗调 10，精调 1 */
    | "milliseconds"
    /** 秒（参数化）。粗调 0.01，精调 0.001 */
    | "seconds"
    /** 像素。整数 */
    | "pixels"
    /** 纯计数 / 索引 / 档位。整数（精调无意义，故与粗调相同） */
    | "integer";

export interface StepSpec {
    /** 无修饰键时的步长。 */
    coarse: number;
    /** 按住精细调整修饰键时的步长。 */
    fine: number;
    /** 提交时保留的小数位（同时用于避免 0.1+0.2 这类浮点噪声）。 */
    decimals: number;
}

const STEPS: Record<StepUnit, StepSpec> = {
    bpm: { coarse: 1, fine: 0.1, decimals: 1 },
    cents: { coarse: 1, fine: 0.1, decimals: 1 },
    semitone: { coarse: 1, fine: 1, decimals: 0 },
    percent: { coarse: 5, fine: 1, decimals: 0 },
    percentFine: { coarse: 1, fine: 0.1, decimals: 2 },
    gainDb: { coarse: 0.5, fine: 0.1, decimals: 1 },
    levelDb: { coarse: 3, fine: 1, decimals: 1 },
    rate: { coarse: 0.1, fine: 0.01, decimals: 2 },
    milliseconds: { coarse: 10, fine: 1, decimals: 0 },
    seconds: { coarse: 0.01, fine: 0.001, decimals: 3 },
    pixels: { coarse: 1, fine: 1, decimals: 0 },
    integer: { coarse: 1, fine: 1, decimals: 0 },
};

export function stepFor(unit: StepUnit): StepSpec {
    return STEPS[unit];
}

/**
 * 按单位语义取步长，并按是否按住精细修饰键选择粗/精。
 */
export function resolveStep(unit: StepUnit, fine: boolean): number {
    const spec = stepFor(unit);
    return fine ? spec.fine : spec.coarse;
}

/**
 * 把值对齐到步长网格并夹紧，同时消掉浮点噪声。
 *
 * 【为什么不用 `Math.round(v/step)*step` 直接算】0.1 这类步长在二进制浮点里
 * 不精确，`3 * 0.1` 会得到 `0.30000000000000004`。这里按 `decimals` 收敛到
 * 固定小数位，避免把噪声写进工程数据。
 */
export function quantizeValue(value: number, spec: StepSpec, min: number, max: number): number {
    const stepped = Math.round(value / spec.fine) * spec.fine;
    const clamped = Math.min(max, Math.max(min, stepped));
    const factor = 10 ** spec.decimals;
    return Math.round(clamped * factor) / factor;
}

/**
 * 在给定值上按步长走一格。
 *
 * @param direction `+1` 向上、`-1` 向下
 */
export function stepValue(args: {
    value: number;
    direction: 1 | -1;
    unit: StepUnit;
    fine: boolean;
    min: number;
    max: number;
}): number {
    const { value, direction, unit, fine, min, max } = args;
    const spec = stepFor(unit);
    const step = fine ? spec.fine : spec.coarse;
    return quantizeValue(value + direction * step, spec, min, max);
}

/**
 * 精细调整修饰键对应的动作 id。
 *
 * 与时间轴 / 参数编辑器共用同一个绑定，因此"按住 Ctrl 滚轮"在对话框与画布上
 * 是同一个键、同一种手感。
 */
export const FINE_ADJUST_ACTION_ID: ActionId = "modifier.paramFineAdjust";
