/**
 * 参数编辑器「音节块」实例构建（纯函数）。
 *
 * 块画成**宽一点的 MIDI 音符**：实心圆角填充 + 1 CSS px 描边 + 顶部高光。
 * 参数曲线仍在最上层（`drawGlCurves` 在块层之后），块不能抢走它的可读性。
 */
import { secToViewportPx, type TimelineAxis } from "../../../renderKernel/timelineAxis";
import { snapToDevicePx, wholeDevicePxLength } from "../../../../../utils/devicePixelLine";
import type { Rgba } from "../../../renderKernel/instanceTypes";
import {
    PARAM_BLOCK_TRANSITION_HANDLE_LINE_LENGTH_PX,
    PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX,
    paramBlockTransitionHandleY,
} from "../../paramBlocks";
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
    /** 当前被指针悬停 / 正在拖动的块；只影响手柄显示强度。 */
    readonly hovered?: boolean;
    /** 悬停在同一相接边界上；只强化该处手柄，不改变可见性规则。 */
    readonly transitionHovered?: boolean;
    readonly grouped: boolean;
    readonly transitionHandle?: {
        readonly sec: number;
        readonly leftId: string;
        readonly rightId: string;
        readonly widthSec: number;
    } | null;
    /** 块内平均参数文字；缺省不画。 */
    readonly label?: string;
    /** The visible interval starts/ends inside its source span. */
    readonly cutStart?: boolean;
    readonly cutEnd?: boolean;
    /** Display-only fallback value: hollow/dashed, never a persisted center. */
    readonly fallback?: boolean;
}

export interface ParamBlockInstanceArgs {
    readonly axis: TimelineAxis;
    readonly viewportHeightPx: number;
    readonly blocks: readonly ParamBlockRectInput[];
    readonly borderRgba: Rgba;
    readonly selectedRgba: Rgba;
    readonly groupedRgba: Rgba;
    readonly labelRgba?: Rgba;
    readonly selectedLabelRgba?: Rgba;
    /** 块填充色；缺省不画填充（兼容只画描边的旧调用）。 */
    readonly fillRgba?: Rgba;
    /** 鼠标悬停块的填充色；缺省复用 `fillRgba`。 */
    readonly hoverFillRgba?: Rgba;
    /** 合并块的填充色；缺省复用 `fillRgba`。 */
    readonly groupedFillRgba?: Rgba;
    /** 选中块的填充色；缺省复用 `fillRgba`。 */
    readonly selectedFillRgba?: Rgba;
    /** 顶部高光色；缺省半透白。 */
    readonly topHighlightRgba?: Rgba;
    /** 块间圆滑手柄色；缺省用 selectedRgba。 */
    readonly handleRgba?: Rgba;
    /** 手柄的柔光底 / 握持线色；缺省由 handleRgba 派生。 */
    readonly handleGlowRgba?: Rgba;
    readonly handleGripRgba?: Rgba;
    /** 顶部高光厚度（CSS px）；0 = 不画。 */
    readonly topHighlightPx?: number;
    /** 块的圆角半径（CSS px）。 */
    readonly cornerRadiusPx?: number;
    /** 边框厚度（CSS 意图）；缺省 1 个物理像素。 */
    readonly borderWidthPx?: number;
    /** 边框物理像素数；缺省 1。与网格 hairline 同口径。 */
    readonly borderPhysicalPx?: number;
    /** 块间过渡曲线颜色；缺省不画。 */
    readonly transitionRgba?: Rgba;
    /** 过渡曲线的端点圆点颜色；缺省复用 `transitionRgba`。 */
    readonly transitionPointRgba?: Rgba;
    /** 端点圆点半径（CSS px）；缺省 2.5。 */
    readonly transitionPointRadiusPx?: number;
}

const DEFAULT_BORDER_PHYSICAL_PX = 1;
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
    /** 相接块圆滑手柄；画在块和曲线之上，方便抓住。 */
    readonly handles: readonly ParamBlockFillInstance[];
    /** 块间过渡曲线：贝塞尔采样后的折线（视口坐标）。 */
    readonly transitions: readonly (readonly ParamBlockPoint[])[];
}

export interface ParamBlockPoint {
    readonly x: number;
    readonly y: number;
}

/** 圆滑手柄：块右边界外侧上方的圆点 + 横线，左右拖动控制过渡宽度。 */
const HANDLE_DIAMETER_PX = 7;
const HANDLE_GLOW_DIAMETER_PX = 11;
/** 过渡贝塞尔曲线的采样数；曲线很短时也足够平滑。 */
const TRANSITION_CURVE_STEPS = 24;

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

export interface SnappedBlockRect {
    readonly left: number;
    readonly right: number;
    readonly top: number;
    readonly bottom: number;
    readonly width: number;
    readonly height: number;
    readonly borderPx: number;
}

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

/**
 * 把块矩形吸附到设备像素栅格，并让四条边的厚度都是整数物理像素。
 *
 * 只在位置或线宽单方面取整都不够：位置落在半物理像素上时，同一条 1px
 * 线会因覆盖列数不同而看起来一粗一细。这里同时吸附左右/上下边界，
 * 线宽也换算成整数物理像素，填充与边框因此共享同一个矩形。
 */
export function snapParamBlockRect(args: {
    readonly x0: number;
    readonly x1: number;
    readonly rawTop: number;
    readonly rawBottom: number;
    readonly viewportWidthPx: number;
    readonly viewportHeightPx: number;
    readonly dpr: number;
    readonly borderPhysicalPx?: number;
}): SnappedBlockRect | null {
    const dpr = Number.isFinite(args.dpr) && args.dpr > 0 ? args.dpr : 1;
    const borderPhysicalPx = Math.max(
        1,
        Math.round(
            Number.isFinite(args.borderPhysicalPx)
                ? (args.borderPhysicalPx as number)
                : DEFAULT_BORDER_PHYSICAL_PX,
        ),
    );
    const borderPx = borderPhysicalPx / dpr;
    const vw = Math.max(0, args.viewportWidthPx);
    const vh = Math.max(0, args.viewportHeightPx);
    let left = clamp(snapToDevicePx(args.x0, dpr), 0, vw);
    let right = clamp(snapToDevicePx(args.x1, dpr), 0, vw);
    let top = clamp(snapToDevicePx(args.rawTop, dpr), 0, vh);
    let bottom = clamp(snapToDevicePx(args.rawBottom, dpr), 0, vh);
    if (!(right > 0) || left >= vw || !(bottom > 0) || top >= vh) return null;
    if (right - left < borderPx) {
        right = Math.min(vw, left + borderPx);
        if (right - left < borderPx) left = Math.max(0, right - borderPx);
    }
    if (bottom - top < borderPx) {
        bottom = Math.min(vh, top + borderPx);
        if (bottom - top < borderPx) top = Math.max(0, bottom - borderPx);
    }
    if (!(right > left) || !(bottom > top)) return null;
    return {
        left,
        right,
        top,
        bottom,
        width: right - left,
        height: bottom - top,
        borderPx,
    };
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
    const dpr = Number.isFinite(axis.dpr) && axis.dpr > 0 ? axis.dpr : 1;
    const borderPx =
        Number.isFinite(args.borderWidthPx) && (args.borderWidthPx as number) > 0
            ? wholeDevicePxLength(args.borderWidthPx as number, dpr)
            : (args.borderPhysicalPx ?? DEFAULT_BORDER_PHYSICAL_PX) / dpr;
    const borderPhysicalPx = Math.max(1, Math.round(borderPx * dpr));
    const height = Math.max(0, viewportHeightPx);
    const items: GridInstance[] = [];
    const labels: ParamBlockLabel[] = [];
    const fills: ParamBlockFillInstance[] = [];
    const highlights: ParamBlockFillInstance[] = [];
    const handles: ParamBlockFillInstance[] = [];
    const transitions: ParamBlockPoint[][] = [];
    /*
     * 过渡曲线需要一次算出左右块的代表高度。先按 id 建索引，避免在每个块
     * 的循环里对全部块 `find`，长工程下会退化成平方复杂度。
     */
    const blockById = new Map(blocks.map((block) => [block.id, block]));
    for (const block of blocks) {
        const startSec = Math.min(block.startSec, block.endSec);
        const endSec = Math.max(block.startSec, block.endSec);
        if (!Number.isFinite(startSec) || !Number.isFinite(endSec)) continue;
        const x0 = secToViewportPx(axis, startSec);
        const x1 = secToViewportPx(axis, endSec);
        // Only a block completely outside the horizontal viewport is absent.
        // A one-frame interval may project to less than one CSS pixel; the
        // device-pixel snap below still gives it at least one physical pixel.
        if (!(endSec > startSec)) continue;
        if (x1 < 0 || x0 > axis.viewportWidthPx) continue;
        const rawTop = Number.isFinite(block.topPx) ? block.topPx : 0;
        const rawBottom = rawTop + Math.max(1, block.heightPx);
        if (rawBottom <= 0 || rawTop >= height) continue;
        const snapped = snapParamBlockRect({
            x0,
            x1,
            rawTop,
            rawBottom,
            viewportWidthPx: axis.viewportWidthPx,
            viewportHeightPx: height,
            dpr,
            borderPhysicalPx,
        });
        if (snapped === null) continue;
        const { left, right, top, bottom, width: visibleWidth, height: h } = snapped;
        const rgba = block.selected ? args.selectedRgba : args.borderRgba;
        const fallback = block.fallback === true;
        const baseFillRgba = block.selected
            ? (args.selectedFillRgba ?? args.fillRgba)
            : block.grouped
              ? (args.groupedFillRgba ?? args.fillRgba)
              : args.fillRgba;
        const fillRgba =
            fallback && baseFillRgba !== undefined
                ? ([
                      baseFillRgba[0],
                      baseFillRgba[1],
                      baseFillRgba[2],
                      baseFillRgba[3] * 0.25,
                  ] as Rgba)
                : baseFillRgba;
        const fillWidth = visibleWidth;
        // 块相交；只在被视口截断的那一侧画边线，不能在上/下边缘伪造一条块顶。
        const clipTop = rawTop < 0;
        const clipBottom = rawBottom > height;
        if (fillRgba !== undefined) {
            fills.push({
                x: left,
                y: top,
                w: fillWidth,
                h,
                rgba: fillRgba,
                // A cut side is flat: the source block continues outside the
                // visible Clip. Rounded corners there would look like an end.
                radius:
                    block.cutStart === true || block.cutEnd === true
                        ? 0
                        : Math.max(
                              0,
                              Math.min(
                                  args.cornerRadiusPx ?? DEFAULT_CORNER_RADIUS_PX,
                                  visibleWidth / 2,
                                  h / 2,
                              ),
                          ),
            });
        }
        const cutStripRgba: Rgba = [rgba[0], rgba[1], rgba[2], Math.min(1, rgba[3] * 0.45)];
        const stripWidth = Math.min(2, Math.max(borderPx, visibleWidth / 4));
        if (block.cutStart === true) {
            highlights.push({ x: left, y: top, w: stripWidth, h, rgba: cutStripRgba, radius: 0 });
        }
        if (block.cutEnd === true) {
            highlights.push({
                x: Math.max(left, right - stripWidth),
                y: top,
                w: stripWidth,
                h,
                rgba: cutStripRgba,
                radius: 0,
            });
        }
        if (fallback) {
            // Hollow/dashed look: thin inner bars on top and bottom instead of
            // a solid fill. It is a display state only; dragging writes a real
            // curve value and the fallback style disappears on the next frame.
            for (const y of [top, Math.max(top, bottom - borderPx)]) {
                items.push({
                    x: left + borderPx,
                    y,
                    w: Math.max(0, fillWidth - borderPx * 2),
                    h: borderPx,
                    rgba: [...cutStripRgba] as Rgba,
                    value: -3,
                });
            }
        }
        /*
         * 每个块都必须有左右边界，即使它与相邻音节时间相接。相接关系只影响
         * 填充是否连成一条，不能把左右轮廓一起省掉；否则块看起来没有边。
         *
         * 边界与填充在**同一条实例缓冲、同一次 draw call** 内提交：块层只是
         * 曲线帧里的一个绘制步骤，不另建 DOM / Canvas 层。
         */
        items.push({ x: left, y: top, w: borderPx, h, rgba, value: -3 });
        items.push({
            x: Math.max(left, right - borderPx),
            y: top,
            w: borderPx,
            h,
            rgba,
            value: -3,
        });
        const topH = args.topHighlightPx ?? 0;
        if (!clipTop) {
            items.push({ x: left, y: top, w: fillWidth, h: borderPx, rgba, value: -3 });
            if (topH > 0) {
                highlights.push({
                    x: left,
                    y: top,
                    w: fillWidth,
                    h: wholeDevicePxLength(topH, dpr),
                    rgba: args.topHighlightRgba ?? rgba,
                    radius: 0,
                });
            }
        }
        if (!clipBottom) {
            items.push({
                x: left,
                y: bottom - borderPx,
                w: fillWidth,
                h: borderPx,
                rgba,
                value: -3,
            });
        }
        if (block.grouped) {
            items.push({
                x: left,
                y: top,
                w: visibleWidth,
                h: borderPx,
                rgba: args.groupedRgba,
                value: -4,
            });
        }
        const label = block.label?.trim() ?? "";
        if (
            label.length > 0 &&
            h >= LABEL_FONT_PX + 4 &&
            visibleWidth >= LABEL_MIN_WIDTH_PX + LABEL_INSET_PX * 2 &&
            args.labelRgba !== undefined
        ) {
            labels.push({
                text: label,
                fontKey: `${LABEL_FONT_PX}px sans-serif`,
                x: left + LABEL_INSET_PX,
                y: top + h / 2,
                align: "left",
                baseline: "middle",
                rgba:
                    block.selected && args.selectedLabelRgba !== undefined
                        ? args.selectedLabelRgba
                        : args.labelRgba,
            });
        }
        if (block.transitionHandle !== undefined && block.transitionHandle !== null) {
            const handleSecX = secToViewportPx(axis, block.transitionHandle.sec);
            const rightBlock = blockById.get(block.transitionHandle.rightId);
            /*
             * 手柄不是“每个相接边界”的装饰：它只属于当前正在编辑的块。
             * 块层收到完整的相接关系，但只有块自身被选中（或正在悬停该
             * 边界）才画，避免选中一个块后整排边界都冒出控件。
             */
            const visible = block.selected || block.transitionHovered === true;
            /*
             * 手柄圆点中心与右边界对齐，只向上抬到块外。以前向右偏移 18px，
             * 既容易被下一个块遮住，也会和块边缘缩放热区抢指针。
             */
            const handleX = handleSecX + PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX;
            const handleY = paramBlockTransitionHandleY(top, rightBlock?.topPx);
            if (
                visible &&
                handleX >= -HANDLE_GLOW_DIAMETER_PX &&
                handleX <= axis.viewportWidthPx + HANDLE_GLOW_DIAMETER_PX
            ) {
                const handleRgba = args.handleRgba ?? args.selectedRgba;
                const active = block.selected || block.transitionHovered === true;
                // 横线：从圆点向左伸出，读起来像“水平拖动的滑块”。
                const lineGripRgba = args.handleGripRgba ?? [
                    handleRgba[0],
                    handleRgba[1],
                    handleRgba[2],
                    0.72,
                ];
                handles.push({
                    x: handleX - PARAM_BLOCK_TRANSITION_HANDLE_LINE_LENGTH_PX,
                    y: handleY - 0.5,
                    w: PARAM_BLOCK_TRANSITION_HANDLE_LINE_LENGTH_PX,
                    h: 1,
                    rgba: lineGripRgba,
                    radius: 0.5,
                });
                // 柔光底：略大于圆点，让它在任何块底色上都可辨认。
                handles.push({
                    x: handleX - HANDLE_GLOW_DIAMETER_PX / 2,
                    y: handleY - HANDLE_GLOW_DIAMETER_PX / 2,
                    w: HANDLE_GLOW_DIAMETER_PX,
                    h: HANDLE_GLOW_DIAMETER_PX,
                    rgba: args.handleGlowRgba ?? [
                        handleRgba[0],
                        handleRgba[1],
                        handleRgba[2],
                        active ? 0.22 : 0.12,
                    ],
                    radius: HANDLE_GLOW_DIAMETER_PX / 2,
                });
                // 圆点：真正表达“这里是过渡边界”。
                handles.push({
                    x: handleX - HANDLE_DIAMETER_PX / 2,
                    y: handleY - HANDLE_DIAMETER_PX / 2,
                    w: HANDLE_DIAMETER_PX,
                    h: HANDLE_DIAMETER_PX,
                    rgba: handleRgba,
                    radius: HANDLE_DIAMETER_PX / 2,
                });
            }
            /*
             * widthSec = 0 是“硬切换”，仍画一小段很短的曲线作为可视提示，
             * 与 OpenUtau 的音高轨迹一致；真正增大平滑宽度时再向右展开。
             */
            const roomPx = Math.max(
                0,
                Math.min(
                    ((block.endSec - block.startSec) * axis.pxPerSec) / 2,
                    rightBlock
                        ? ((rightBlock.endSec - rightBlock.startSec) * axis.pxPerSec) / 2
                        : Infinity,
                ),
            );
            const transitionWidthPx = Math.min(
                roomPx,
                Math.max(Math.min(8, roomPx), block.transitionHandle.widthSec * axis.pxPerSec),
            );
            if (transitionWidthPx < 0.5) continue;
            /*
             * 贝塞尔只作用在相接点**右侧**：左侧块自己的音高线保持不变。
             * 起点圆点就是相接点，曲线向右展开；右侧块在原曲线基础上被拉向
             * 目标值，而不是被一条独立线覆盖。
             */
            const leftY = top + h / 2;
            const rightY =
                rightBlock === undefined
                    ? leftY
                    : rightBlock.topPx + Math.max(1, rightBlock.heightPx) / 2;
            if (
                handleSecX < -transitionWidthPx ||
                handleSecX > axis.viewportWidthPx + transitionWidthPx
            ) {
                continue;
            }
            const startX = handleSecX - transitionWidthPx;
            const endX = handleSecX + transitionWidthPx;
            if (args.transitionRgba !== undefined && endX >= 0 && startX <= axis.viewportWidthPx) {
                const curve: ParamBlockPoint[] = [];
                for (let index = 0; index <= TRANSITION_CURVE_STEPS; index += 1) {
                    const t = index / TRANSITION_CURVE_STEPS;
                    const oneMinusT = 1 - t;
                    // 三次贝塞尔，控制点沿水平方向；高度由两端代表值决定。
                    const cp1X = startX + (endX - startX) / 3;
                    const cp2X = startX + ((endX - startX) * 2) / 3;
                    const x =
                        oneMinusT * oneMinusT * oneMinusT * startX +
                        3 * oneMinusT * oneMinusT * t * cp1X +
                        3 * oneMinusT * t * t * cp2X +
                        t * t * t * endX;
                    const y =
                        oneMinusT * oneMinusT * oneMinusT * leftY +
                        3 * oneMinusT * oneMinusT * t * ((leftY + rightY) / 2) +
                        3 * oneMinusT * t * t * ((leftY + rightY) / 2) +
                        t * t * t * rightY;
                    curve.push({ x, y });
                }
                transitions.push(curve);
                /*
                 * 即使块被视口纵向裁掉，相接点圆点仍应可见；这里单独画
                 * 起点圆点，不依赖 `handles` 的手柄选中态。
                 */
                const pointRgba = args.transitionPointRgba ?? args.transitionRgba;
                const pointRadius = Math.max(0.5, args.transitionPointRadiusPx ?? 2.5);
                handles.push({
                    x: startX - pointRadius,
                    y: leftY - pointRadius,
                    w: pointRadius * 2,
                    h: pointRadius * 2,
                    rgba: pointRgba,
                    radius: pointRadius,
                });
            }
        }
    }
    return { items, labels, fills, highlights, handles, transitions };
}
