/**
 * fadeTooltipText — 淡入淡出编辑悬停浮标的共享文本拼装。
 *
 * 使用场景：包络线命中块、淡化区域边缘竖线、重叠区淡化控件、交叉点抓手
 * —— 一切"可被视为在编辑淡入淡出包络"的悬停目标。双侧一致的信息结构：
 *   {淡入|淡出}类型：{REAPER 形状名}
 *   长度：{主时间单位}[ / {副时间单位}]   ← 相对时长（零基点）
 *   曲率：{±0.00}
 *
 * 长度格式化走 timeFormat.formatFadeLengthTooltip（相对时长，无工程原点
 * 偏移；主副单位来自时间轴显示设置）。
 */
import { formatFadeLengthTooltip, type FadeLengthFormatContext } from "./timeFormat";

export type { FadeLengthFormatContext };

export type FadeLabelLookup = (key: string) => string;

/** 形状 id → i18n 键（与 ClipContextMenu 的 FADE_SHAPE_OPTIONS 同源）。 */
const SHAPE_LABEL_KEYS: Record<number, string> = {
    0: "fade_shape_linear",
    1: "fade_shape_fast_start",
    2: "fade_shape_fast_end",
    3: "fade_shape_fast_start_steep",
    4: "fade_shape_fast_end_steep",
    5: "fade_shape_slow_start_end",
    6: "fade_shape_slow_start_end_steep",
};

function shapeName(shape: number, t: FadeLabelLookup): string {
    // 小数变体按其基础族命名（1.1 → 快起族；REAPER 内部虽有独立编号，
    // 但对外呈现的名称与基础预设一致）。
    const normalized = Math.trunc(Number.isFinite(shape) ? shape : 0);
    return t(SHAPE_LABEL_KEYS[normalized] ?? "fade_shape_linear");
}

/**
 * 单侧淡变块（三行）。`isOut` 决定侧别文案与曲率的符号语义都沿用该侧
 * 存储值本身（dir 就是"该侧约定"），形状名直接按存储形状展示。
 */
export function buildSingleFadeInfoText(args: {
    isOut: boolean;
    shape: number;
    dir: number;
    lengthSec: number;
    formatCtx: FadeLengthFormatContext;
    t: FadeLabelLookup;
}): string {
    const sideLabel = args.isOut ? args.t("fade_out") : args.t("fade_in");
    const name = shapeName(args.shape, args.t);
    const curvature = args.t("curvature");
    const length = args.t("length");
    const sign = args.dir >= 0 ? "+" : "";
    return [
        `${sideLabel}${args.t("fade_type_label")}：${name}`,
        `${length}：${formatFadeLengthTooltip(Math.max(0, args.lengthSec), args.formatCtx)}`,
        `${curvature}：${sign}${args.dir.toFixed(2)}`,
    ].join("\n");
}

/**
 * 交叉点抓手：前一个 clip 的淡出在前、后一个 clip 的淡入在后，
 * 两块之间空一行分隔。
 */
export function buildCrossfadeGripInfoText(args: {
    earlier: { shape: number; dir: number; lengthSec: number };
    later: { shape: number; dir: number; lengthSec: number };
    formatCtx: FadeLengthFormatContext;
    t: FadeLabelLookup;
}): string {
    return [
        buildSingleFadeInfoText({ isOut: true, ...args.earlier, formatCtx: args.formatCtx, t: args.t }),
        "",
        buildSingleFadeInfoText({ isOut: false, ...args.later, formatCtx: args.formatCtx, t: args.t }),
    ].join("\n");
}

