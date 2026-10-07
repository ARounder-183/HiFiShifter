/**
 * fadeTooltipText — 淡入淡出编辑悬停浮标的共享内容拼装。
 *
 * 使用场景：包络线命中块、淡化区域边缘竖线、重叠区淡化控件、交叉点抓手
 * —— 一切"可被视为在编辑淡入淡出包络"的悬停目标。双侧一致的信息结构：
 *
 *   {淡入|淡出}类型：{REAPER 曲线图标}
 *   长度：{主时间单位}[ / {副时间单位}][ [±位移]]   ← 相对时长（零基点）
 *   曲率：{±0.00}[ [±增量]]
 *
 * 两个版本：
 * - `buildSingleFadeInfoText` / `buildCrossfadeGripInfoText`：纯文本，
 *   写入元素的 `data-tooltip` 属性即可（AppTooltipProvider 常规路径）；
 * - `buildSingleFadeInfoContent` / `buildCrossfadeGripInfoContent`：
 *   ReactNode 版，类型行以 FadeShapeIcon 内联 SVG 图标替代文字名称。
 *   经 `publishFadeRichTooltip` 注册到 AppTooltipProvider 的富内容表，
 *   元素自身带 `data-hs-rich-tooltip` 标记以便悬停命中。
 *
 * **拖拽期间逐帧发布富内容版**（面板的 fade 预览回调）：传入 `delta` 后长度与曲率
 * 两行各追加 `[±增量]`，读数随指针实时更新 —— 与吸附偏移 / 增益旋钮同一取舍。
 *
 * 长度格式化走 `timeValueText.formatDurationText`（相对时长，无工程原点
 * 偏移；主副单位来自时间轴显示设置）。
 */
import type { ReactNode } from "react";
import { createElement } from "react";
import type {HostFadeMetadata} from "../../../types/api";
import {hostFadeLabel} from "./hostFadeDisplay";

import { formatTemplate } from "../../../i18n/format";
import { formatDurationText, formatSignedDurationTextOrNull } from "./timeValueText";
import type { FadeLengthFormatContext } from "./timeFormat";
import { FadeShapeIcon } from "./FadeShapeIcon";
import { HS_TOOLTIP_CONTENT_EVENT } from "../../../components/AppTooltip";

export type { FadeLengthFormatContext };

export type FadeLabelLookup = (key: string) => string;

/**
 * 拖拽中的**增量**（每个字段都是"当前值 − 按下时的值"，带符号）。
 *
 * 缺省（`undefined`）= 悬停态，不展示任何增量。拖拽期间逐帧传入 ⇒ 内容随指针
 * 实时更新（与吸附偏移 / 增益旋钮同一取舍：悬停只给当前值，拖拽才给位移）。
 *
 * 【为什么基准是"按下时的值"而不是"上一帧的值"】位移量描述的是**本次手势**移动了
 * 多少；逐帧差分会在松手前不断归零，读不出总量。
 */
export interface FadeInfoDelta {
    /** 长度增量（秒，带符号）。 */
    readonly lengthSec?: number | null;
    /** 曲率增量（带符号）。 */
    readonly dir?: number | null;
}

/** 形状 id → i18n 键（与 ClipContextMenu 的 FADE_SHAPE_OPTIONS 同源）。 */
export const SHAPE_LABEL_KEYS: Record<number, string> = {
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
 * 曲率增量文本：`+0.20` / `-0.35`；落到显示精度之下（四舍五入为 0）时返回 null。
 *
 * 与曲率本身的显示共用两位小数精度，因此阈值就是"四舍五入到 0"——
 * 显示 `[+0.00]` 只会让人以为功能坏了。
 */
function formatDirDelta(delta: number | null | undefined): string | null {
    if (delta === null || delta === undefined || !Number.isFinite(delta)) return null;
    const rounded = Math.round(delta * 100) / 100;
    if (rounded === 0) return null;
    return `${rounded > 0 ? "+" : ""}${rounded.toFixed(2)}`;
}

/** `值[ [±增量]]`：增量缺席时只给值（悬停态）。 */
function withDelta(value: string, delta: string | null): string {
    return delta === null ? value : `${value} [${delta}]`;
}

/** 信息行内联图标的统一尺寸（与上下文菜单图标一致）。 */
const TOOLTIP_ICON_SIZE = 16;

/**
 * `label: value` 信息行。冒号与空格的语系差异（en `": "` / CJK `：`）
 * 由词典模板 `common_label_value` 决定 —— 不得在代码里硬编码任何一种冒号。
 */
function labelValue(t: FadeLabelLookup, label: string, value: string): string {
    return formatTemplate(t("common_label_value"), { label, value });
}

/** 类型行的图标节点（垂直居中对齐文本基线）。 */
function fadeIconNode(shape: number, isOut: boolean): ReactNode {
    return createElement(
        "span",
        {
            key: "icon",
            style: {
                display: "inline-flex",
                verticalAlign: "-3px",
                marginLeft: 2,
                // 淡出行镜像曲线方向，使其与画布上该侧实际走向一致。
                transform: isOut ? "scaleX(-1)" : undefined,
            },
        },
        createElement(FadeShapeIcon, { shape, size: TOOLTIP_ICON_SIZE }),
    );
}

/**
 * 单侧淡变块（三行）。`isOut` 决定侧别文案与曲率的符号语义都沿用该侧
 * 存储值本身（dir 就是"该侧约定"），形状名直接按存储形状展示。
 *
 * `delta` 缺省 = 悬停（只给当前值）；拖拽时**长度与曲率两行各追加 `[±增量]`**。
 */
export function buildSingleFadeInfoText(args: {
    isOut: boolean;
    shape: number;
    dir: number;
    lengthSec: number;
    formatCtx: FadeLengthFormatContext;
    t: FadeLabelLookup;
    hostFades?:HostFadeMetadata;
    delta?: FadeInfoDelta;
}): string {
    const sideLabel = args.isOut ? args.t("fade_out") : args.t("fade_in");
    const name = shapeName(args.shape, args.t);
    const typeLabel = formatTemplate(args.t("fade_info_side_type_label"), {
        side: sideLabel,
        type: args.t("fade_type_label"),
    });
    if (args.hostFades && args.hostFades.curve_mode !== "legacy" && args.hostFades.curve_mode !== "hifishifter") {
        return [labelValue(args.t, typeLabel, hostFadeLabel(args.hostFades, args.isOut)),
            labelValue(args.t, args.t("common_length"), lengthLine(args.lengthSec, args.formatCtx, args.delta)),
            "HiFiShifter 示意曲线；声音由 REAPER 控制"].join("\n");
    }
    return [
        labelValue(args.t, typeLabel, name),
        labelValue(args.t, args.t("common_length"), lengthLine(args.lengthSec, args.formatCtx, args.delta)),
        labelValue(args.t, args.t("common_curvature"), dirLine(args.dir, args.delta)),
    ].join("\n");
}

/** 富内容版单侧块：首行为"侧别+图标"，其余两行为纯文本。 */
export function buildSingleFadeInfoContent(args: {
    isOut: boolean;
    shape: number;
    dir: number;
    lengthSec: number;
    formatCtx: FadeLengthFormatContext;
    t: FadeLabelLookup;
    delta?: FadeInfoDelta; hostFades?:HostFadeMetadata;
}): ReactNode {
    if (args.hostFades&&args.hostFades.curve_mode!=="legacy"&&args.hostFades.curve_mode!=="hifishifter") {
        return buildSingleFadeInfoText(args).split("\n").map((row,key)=>createElement("div",{key},row));
    }
    const sideLabel = args.isOut ? args.t("fade_out") : args.t("fade_in");
    const typeLabel = formatTemplate(args.t("fade_info_side_type_label"), {
        side: sideLabel,
        type: args.t("fade_type_label"),
    });
    return [
        [typeLabel, args.t("common_value_sep"), fadeIconNode(args.shape, args.isOut)],
        [labelValue(args.t, args.t("common_length"), lengthLine(args.lengthSec, args.formatCtx, args.delta))],
        [labelValue(args.t, args.t("common_curvature"), dirLine(args.dir, args.delta))],
    ].map((row, index) =>
        createElement(
            "div",
            { key: index },
            row.map((part, partIndex) =>
                typeof part === "string" ? createElement("span", { key: partIndex }, part) : part,
            ),
        ),
    );
}

/** 长度行的值部分：`{主} / {副}[ [±位移]]`（零基点时长口径）。 */
function lengthLine(
    lengthSec: number,
    formatCtx: FadeLengthFormatContext,
    delta: FadeInfoDelta | undefined,
): string {
    return withDelta(
        formatDurationText(Math.max(0, lengthSec), formatCtx),
        formatSignedDurationTextOrNull(delta?.lengthSec, formatCtx),
    );
}

/** 曲率行的值部分：`{±0.00}[ [±增量]]`。 */
function dirLine(dir: number, delta: FadeInfoDelta | undefined): string {
    const safe = Number.isFinite(dir) ? dir : 0;
    return withDelta(`${safe >= 0 ? "+" : ""}${safe.toFixed(2)}`, formatDirDelta(delta?.dir));
}

/**
 * 交叉点抓手富内容：前一个 clip 的淡出在前、空一行、后一个 clip 的
 * 淡入在后。
 *
 * 两侧各自带自己的增量 —— 反向模式下两侧淡变按比例缩放，位移量并不相同。
 */
export function buildCrossfadeGripInfoContent(args: {
    earlier: { shape: number; dir: number; lengthSec: number; delta?: FadeInfoDelta; hostFades?:HostFadeMetadata };
    later: { shape: number; dir: number; lengthSec: number; delta?: FadeInfoDelta; hostFades?:HostFadeMetadata };
    formatCtx: FadeLengthFormatContext;
    t: FadeLabelLookup;
}): ReactNode {
    return [
        buildSingleFadeInfoContent({
            isOut: true,
            ...args.earlier,
            formatCtx: args.formatCtx,
            t: args.t,
        }),
        createElement("div", { key: "gap", style: { height: 6 } }),
        buildSingleFadeInfoContent({
            isOut: false,
            ...args.later,
            formatCtx: args.formatCtx,
            t: args.t,
        }),
    ];
}

/**
 * 把富内容注册到指定元素（AppTooltipProvider 监听同一事件维护注册表）。
 * 元素应带 `data-hs-rich-tooltip` 标记以便指针悬停命中。React 渲染期间
 * dispatch 的自定义事件同步送达 —— 注册表在下一次 pointerover/move 前
 * 必然就绪。
 */
export function publishFadeRichTooltip(element: Element | null, content: ReactNode): void {
    if (!element || typeof window === "undefined") return;
    element.setAttribute("data-hs-rich-tooltip", "1");
    window.dispatchEvent(
        new CustomEvent(HS_TOOLTIP_CONTENT_EVENT, {
            detail: { element, content },
        }),
    );
}

/**
 * 交叉点抓手：前一个 clip 的淡出在前、后一个 clip 的淡入在后，
 * 两块之间空一行分隔（纯文本版本）。
 */
export function buildCrossfadeGripInfoText(args: {
    earlier: { shape: number; dir: number; lengthSec: number; delta?: FadeInfoDelta; hostFades?:HostFadeMetadata };
    later: { shape: number; dir: number; lengthSec: number; delta?: FadeInfoDelta; hostFades?:HostFadeMetadata };
    formatCtx: FadeLengthFormatContext;
    t: FadeLabelLookup;
}): string {
    return [
        buildSingleFadeInfoText({
            isOut: true,
            ...args.earlier,
            formatCtx: args.formatCtx,
            t: args.t,
        }),
        "",
        buildSingleFadeInfoText({
            isOut: false,
            ...args.later,
            formatCtx: args.formatCtx,
            t: args.t,
        }),
    ].join("\n");
}
