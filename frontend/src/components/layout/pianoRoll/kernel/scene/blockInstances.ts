/**
 * 参数编辑器「音节块」实例构建（纯函数）。
 *
 * 块画成**宽一点的 MIDI 音符**：实心圆角填充 + 1 CSS px 描边 + 顶部高光。
 * 参数曲线仍在最上层（`drawGlCurves` 在块层之后），块不能抢走它的可读性。
 */
import { secToViewportPx, type TimelineAxis } from "../../../renderKernel/timelineAxis";
import type { Rgba } from "../../../renderKernel/instanceTypes";
import type { GridInstance } from "./gridInstances";

/** 块标签请求（与参数编辑器字形层的 `TextRequest` 同形；此处避免运行时依赖）。 */
export interface ParamBlockLabel {
    readonly text: string;
    readonly fontKey: string;
    readonly x: number;
    readonly y: number;
    readonly align: "left";
    readonly baseline: "middle";
    readonly rgba: Rgba;
}

export interface ParamBlockRectInput {
    readonly id: string;
    readonly startSec: number;
    readonly endSec: number;
    /** 块顶边（CSS px，视口坐标）。 */
    readonly topPx: number;
    /** 块高（CSS px）。 */
    readonly heightPx: number;
    /** 时间方向是否与相邻块连成一块（连续音节不画分隔内缩）。 */
    readonly contiguousStart?: boolean;
    readonly contiguousEnd?: boolean;
    readonly selected: boolean;
    readonly merged: boolean;
    /** 块内平均参数文字；缺省不画。 */
    readonly label?: string;
}

export interface ParamBlockInstanceArgs {
    readonly axis: TimelineAxis;
    readonly viewportHeightPx: number;
    readonly blocks: readonly ParamBlockRectInput[];
    readonly borderRgba: Rgba;
    readonly selectedRgba: Rgba;
    readonly mergedRgba: Rgba;
    readonly labelRgba?: Rgba;
    readonly selectedLabelRgba?: Rgba;
    /** 块填充色；缺省不画填充（兼容只画描边的旧调用）。 */
    readonly fillRgba?: Rgba;
    /** 鼠标悬停块的填充色；缺省复用 `fillRgba`。 */
    readonly hoverFillRgba?: Rgba;
    /** 合并块的填充色；缺省复用 `fillRgba`。 */
    readonly mergedFillRgba?: Rgba;
    /** 选中块的填充色；缺省复用 `fillRgba`。 */
    readonly selectedFillRgba?: Rgba;
    /** 顶部高光色；缺省半透白。 */
    readonly topHighlightRgba?: Rgba;
    /** 顶部高光厚度（CSS px）；0 = 不画。 */
    readonly topHighlightPx?: number;
    /** 块的圆角半径（CSS px）。 */
    readonly cornerRadiusPx?: number;
    /** 边框厚度（CSS px）；缺省 1。第一版块只是“可见的边界”，不抢曲线。 */
    readonly borderWidthPx?: number;
}

const DEFAULT_BORDER_PX = 1;
/** 块内标签：额外缩进，避免文字压住 1px 边线。 */
const LABEL_INSET_PX = 4;
/** 标签宽度不足时省略；与块最小高度一起保证小块不出现半截文字。 */
const LABEL_MIN_WIDTH_PX = 18;
const LABEL_FONT_PX = 9;
const DEFAULT_CORNER_RADIUS_PX = 4;
/** 顶部高光厚度：只描亮上缘，模拟音符块的受光面。 */

export interface ParamBlockScene {
    readonly items: readonly GridInstance[];
    readonly labels: readonly ParamBlockLabel[];
    /** 圆角填充实例；与 `items` 分开发射，保证边框压住填充。 */
    readonly fills: readonly ParamBlockFillInstance[];
    /** 顶部 1px 高光；排在填充之后、描边之前，做出音符的受光边。 */
    readonly highlights: readonly ParamBlockFillInstance[];
}

/**
 * 块填充实例：`FlatInstance` 的布局没有圆角，为了让块像音符，这里扩一个可选
 * 圆角半径；宿主写实例时把半径放进 `radius` 槽，用 BOX 模式绘制。
 */
export interface ParamBlockFillInstance {
    readonly x: number;
    readonly y: number;
    readonly w: number;
    readonly h: number;
    readonly rgba: Rgba;
    readonly radius: number;
}

/**
 * 构建块边框实例。
 *
 * 每个块产出 1 个填充 + 4 条边（顶部另有 1 条高光）。选中块用 selected 色。
 * 完全落在视口外的块整体跳过。
 */
export function buildParamBlockInstances(args: ParamBlockInstanceArgs): GridInstance[] {
    return [...buildParamBlockScene(args).items];
}

/**
 * 构建块层完整场景：边框实例 + 文字标签。
 *
 * 文字单独返回而不是并入 `GridInstance`，因为 GL 的矩形实例不带字形；
 * 宿主在同一个 GL 画布上先画块边框、再画字形，层序保持“块在曲线之下”。
 */
export function buildParamBlockScene(args: ParamBlockInstanceArgs): ParamBlockScene {
    const { axis, viewportHeightPx, blocks } = args;
    const borderPx =
        Number.isFinite(args.borderWidthPx) && (args.borderWidthPx as number) > 0
            ? (args.borderWidthPx as number)
            : DEFAULT_BORDER_PX;
    const height = Math.max(0, viewportHeightPx);
    const items: GridInstance[] = [];
    const labels: ParamBlockLabel[] = [];
    const fills: ParamBlockFillInstance[] = [];
    const highlights: ParamBlockFillInstance[] = [];
    for (const block of blocks) {
        const startSec = Math.min(block.startSec, block.endSec);
        const endSec = Math.max(block.startSec, block.endSec);
        if (!Number.isFinite(startSec) || !Number.isFinite(endSec)) continue;
        const x0 = secToViewportPx(axis, startSec);
        const x1 = secToViewportPx(axis, endSec);
        if (!(x1 - x0 > 0.5)) continue;
        if (x1 <= 0 || x0 >= axis.viewportWidthPx) continue;

        /*
         * 纵向裁剪必须按**视口交集**判断：块整体在视口上下方时直接跳过，
         * 不能把 top 夹到 0 / height-1 后画成边缘的一条假块。
         */
        const rawTop = Number.isFinite(block.topPx) ? block.topPx : 0;
        const rawBottom = rawTop + Math.max(1, block.heightPx);
        if (rawBottom <= 0 || rawTop >= height) continue;
        const top = Math.max(0, rawTop);
        const bottom = Math.min(height, rawBottom);
        const h = Math.max(1, bottom - top);
        if (h <= 0.5) continue;
        const rgba = block.selected ? args.selectedRgba : args.borderRgba;
        const fillRgba = block.selected
            ? (args.selectedFillRgba ?? args.fillRgba)
            : block.merged
              ? (args.mergedFillRgba ?? args.fillRgba)
              : args.fillRgba;
        const fillWidth = Math.max(1, x1 - x0);
        // 块相交；只在被视口截断的那一侧画边线，不能在上/下边缘伪造一条块顶。
        const clipTop = rawTop < 0;
        const clipBottom = rawBottom > height;
        if (fillRgba !== undefined) {
            fills.push({
                x: x0,
                y: top,
                w: fillWidth,
                h,
                rgba: fillRgba,
                radius: Math.max(
                    0,
                    Math.min(args.cornerRadiusPx ?? DEFAULT_CORNER_RADIUS_PX, (x1 - x0) / 2, h / 2),
                ),
            });
        }
        /*
         * 每个块都必须有左右边界，即使它与相邻音节时间相接。相接关系只影响
         * 填充是否连成一条，不能把左右轮廓一起省掉；否则块看起来没有边。
         *
         * 边界与填充在**同一条实例缓冲、同一次 draw call** 内提交：块层只是
         * 曲线帧里的一个绘制步骤，不另建 DOM / Canvas 层。
         */
        const leftEdge = Math.max(0, Math.min(x0, axis.viewportWidthPx - borderPx));
        const rightEdge = Math.max(0, Math.min(x1 - borderPx, axis.viewportWidthPx - borderPx));
        items.push({ x: leftEdge, y: top, w: borderPx, h, rgba, value: -3 });
        if (rightEdge > leftEdge + borderPx * 0.5) {
            items.push({ x: rightEdge, y: top, w: borderPx, h, rgba, value: -3 });
        } else {
            items.push({
                x: Math.max(0, Math.min(x1 - borderPx, axis.viewportWidthPx - borderPx)),
                y: top,
                w: borderPx,
                h,
                rgba,
                value: -3,
            });
        }
        const topH = args.topHighlightPx ?? 0;
        if (!clipTop) {
            items.push({ x: x0, y: top, w: fillWidth, h: borderPx, rgba, value: -3 });
            if (topH > 0) {
                highlights.push({
                    x: x0,
                    y: top,
                    w: fillWidth,
                    h: topH,
                    rgba: args.topHighlightRgba ?? rgba,
                    radius: 0,
                });
            }
        }
        if (!clipBottom) {
            items.push({
                x: x0,
                y: top + h - borderPx,
                w: fillWidth,
                h: borderPx,
                rgba,
                value: -3,
            });
        }
        if (block.merged) {
            items.push({
                x: x0,
                y: top,
                w: x1 - x0,
                h: borderPx,
                rgba: args.mergedRgba,
                value: -4,
            });
        }
        const label = block.label?.trim() ?? "";
        if (
            label.length > 0 &&
            h >= LABEL_FONT_PX + 4 &&
            x1 - x0 >= LABEL_MIN_WIDTH_PX + LABEL_INSET_PX * 2 &&
            args.labelRgba !== undefined
        ) {
            labels.push({
                text: label,
                fontKey: `${LABEL_FONT_PX}px sans-serif`,
                x: Math.max(0, x0) + LABEL_INSET_PX,
                y: top + h / 2,
                align: "left",
                baseline: "middle",
                rgba:
                    block.selected && args.selectedLabelRgba !== undefined
                        ? args.selectedLabelRgba
                        : args.labelRgba,
            });
        }
    }
    return { items, labels, fills, highlights };
}
