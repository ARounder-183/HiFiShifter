/**
 * fadeTooltipText — 淡入淡出编辑悬停浮标的共享内容拼装。
 *
 * 使用场景：包络线命中块、淡化区域边缘竖线、重叠区淡化控件、交叉点抓手
 * —— 一切"可被视为在编辑淡入淡出包络"的悬停目标。双侧一致的信息结构：
 *
 *   {淡入|淡出}类型：{REAPER 曲线图标}
 *   长度：{主时间单位}[ / {副时间单位}]   ← 相对时长（零基点）
 *   曲率：{±0.00}
 *
 * 两个版本：
 * - `buildSingleFadeInfoText` / `buildCrossfadeGripInfoText`：纯文本，
 *   写入元素的 `data-tooltip` 属性即可（AppTooltipProvider 常规路径）；
 * - `buildSingleFadeInfoContent` / `buildCrossfadeGripInfoContent`：
 *   ReactNode 版，类型行以 FadeShapeIcon 内联 SVG 图标替代文字名称。
 *   经 `publishFadeRichTooltip` 注册到 AppTooltipProvider 的富内容表，
 *   元素自身带 `data-hs-rich-tooltip` 标记以便悬停命中。
 *
 * 长度格式化走 timeFormat.formatFadeLengthTooltip（相对时长，无工程原点
 * 偏移；主副单位来自时间轴显示设置）。
 */
import type { ReactNode } from "react";
import { createElement } from "react";

import { formatFadeLengthTooltip, type FadeLengthFormatContext } from "./timeFormat";
import { FadeShapeIcon } from "./FadeShapeIcon";
import { HS_TOOLTIP_CONTENT_EVENT } from "../../../components/AppTooltip";

export type { FadeLengthFormatContext };

export type FadeLabelLookup = (key: string) => string;

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

/** 信息行内联图标的统一尺寸（与上下文菜单图标一致）。 */
const TOOLTIP_ICON_SIZE = 16;

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

/** 富内容版单侧块：首行为"侧别+图标"，其余两行为纯文本。 */
export function buildSingleFadeInfoContent(args: {
    isOut: boolean;
    shape: number;
    dir: number;
    lengthSec: number;
    formatCtx: FadeLengthFormatContext;
    t: FadeLabelLookup;
}): ReactNode {
    const sideLabel = args.isOut ? args.t("fade_out") : args.t("fade_in");
    const curvature = args.t("curvature");
    const length = args.t("length");
    const sign = args.dir >= 0 ? "+" : "";
    return [
        [`${sideLabel}${args.t("fade_type_label")}：`, fadeIconNode(args.shape, args.isOut)],
        [`${length}：${formatFadeLengthTooltip(Math.max(0, args.lengthSec), args.formatCtx)}`],
        [`${curvature}：${sign}${args.dir.toFixed(2)}`],
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

/**
 * 交叉点抓手富内容：前一个 clip 的淡出在前、空一行、后一个 clip 的
 * 淡入在后。
 */
export function buildCrossfadeGripInfoContent(args: {
    earlier: { shape: number; dir: number; lengthSec: number };
    later: { shape: number; dir: number; lengthSec: number };
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
    earlier: { shape: number; dir: number; lengthSec: number };
    later: { shape: number; dir: number; lengthSec: number };
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
