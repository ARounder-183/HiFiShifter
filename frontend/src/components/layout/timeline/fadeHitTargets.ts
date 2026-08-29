/**
 * fadeHitTargets.ts — 淡入淡出"拖拽命中目标"几何计算。
 *
 * 交互模型对齐主流 DAW（如 REAPER）：拖拽控件不是"整片淡入淡出区域"，
 * 而是画面上真正可见的两条线：
 *   1. 包络线：淡入=从左下到右上的斜线；淡出=从左上到右下的斜线；
 *   2. 淡化区域的边缘竖线：淡入区域最右侧 / 淡出区域最左侧。
 *
 * 本模块把这两条线离散化为小块 axis-aligned Hit 矩形，交给 DOM 层直接渲染为
 * 可按下/拖拽的透明手柄；未命中区域的事件可自然穿透到 clip body（拖拽移动 clip）。
 *
 * 几何与 timelineCanvasRenderer.drawFadeCurveStroke 完全一致（同一套
 * fadeCurveGain + bodyTop/bodyHeight 约定），保证"视觉画在哪、就能在哪抓住"。
 */
import { fadeGainSigned } from "./reaperFade";

/**
 * 包络线命中块：边长（px）。
 *
 * 12px 是"能稳定抓住 1.2px 曲线"与"不吃掉太多 Clip 内部"之间的平衡。原为
 * 16px，在矮 Clip 上沿对角线铺开会盖住大半个淡化区，且与裁短/移动手势争地。
 */
export const FADE_LINE_HIT_SIZE = 12;
/**
 * 包络线采样目标弧长步长（px）。沿对角线按等弧长采样。
 * 必须 **小于** FADE_LINE_HIT_SIZE，否则相邻命中块之间会留下空隙。
 */
export const FADE_LINE_HIT_STEP_PX = 10;
/**
 * 采样点数量上限。步长 × 上限 = 可覆盖的最大弧长（10 × 480 ≈ 4800px），
 * 与命中块改为 12px 之前的覆盖能力一致；命中块是真实 DOM 节点，上限即
 * 单个 Clip 的 DOM 开销上限，不随缩放无限增长。
 */
export const FADE_LINE_HIT_MAX_COUNT = 480;
/** 区域边缘竖线命中条宽度（px）。 */
export const FADE_EDGE_HIT_WIDTH = 6;

export type FadeHitType = "fade_in" | "fade_out";

export type FadeHitTarget =
    | {
          kind: "line";
          type: FadeHitType;
          left: number;
          top: number;
          width: number;
          height: number;
      }
    | {
          kind: "edge";
          type: FadeHitType;
          left: number;
          top: number;
          width: number;
          height: number;
      };

type Rect = { left: number; top: number; width: number; height: number };

function squareAround(centerX: number, centerY: number, size: number): Rect {
    return {
        left: centerX - size / 2,
        top: centerY - size / 2,
        width: size,
        height: size,
    };
}

/**
 * 沿包络线采样命中点。
 *
 * @param mode "in"=淡入（增益 t→g，y 从 bottom 到 top）；
 *              "out"=淡出（增益 1-t，y 从 top 到 bottom）。
 * 采样点被裁剪到 [clipXFrom, clipXTo]（用于重叠区域只保留重叠内的部分）。
 */
function sampleFadeLine(args: {
    left: number;
    right: number;
    bodyTop: number;
    bodyHeight: number;
    shape: number;
    dir: number;
    mode: "in" | "out";
    clipXFrom: number;
    clipXTo: number;
}): Array<{ x: number; y: number }> {
    const { left, right, bodyTop, bodyHeight, shape, dir, mode, clipXFrom, clipXTo } = args;
    const width = right - left;
    if (width <= 1e-6) return [];
    // 沿对角线按"弧长 = 目标步长"采样：保证相邻命中块沿 x 与沿 y 都彼此交叠，
    // 覆盖整条包络线（含陡峭短淡化与较长淡化），不产生间断。
    const arcLength = Math.hypot(width, bodyHeight);
    // 密度跟随弧长：极端缩放下弧长可达数千像素，点数下限 2、上限
    // FADE_LINE_HIT_MAX_COUNT。步长必须小于命中块边长（FADE_LINE_HIT_SIZE），
    // 否则相邻块之间会留下无法命中的空隙。
    const count = Math.max(
        2,
        Math.min(FADE_LINE_HIT_MAX_COUNT, Math.ceil(arcLength / FADE_LINE_HIT_STEP_PX)),
    );
    const points: Array<{ x: number; y: number }> = [];
    for (let index = 0; index < count; index += 1) {
        const t = index / Math.max(1, count - 1);
        const x = left + t * width;
        if (x < clipXFrom || x > clipXTo) continue;
        const gain =
            mode === "in"
                ? fadeGainSigned(shape, dir, "in", t)
                : fadeGainSigned(shape, dir, "out", t);
        const y = bodyTop + bodyHeight * (1 - gain);
        points.push({ x, y });
    }
    return points;
}

/**
 * 生成一个 clip 的淡入/淡出命中目标（包络线小方块 + 区域边缘竖条）。
 *
 * 坐标均为"目标定位容器"的局部坐标（lane 的左缘为 0,y 为 0）。
 * 可通过 clipXFrom/clipXTo 把命中目标裁剪到指定时间范围（如重叠区）。
 */
export function buildFadeHitTargets(args: {
    clipLeftPx: number;
    clipWidthPx: number;
    bodyTop: number;
    bodyHeight: number;
    fadeInPx: number;
    fadeOutPx: number;
    fadeInShape: number;
    fadeInDir: number;
    fadeOutShape: number;
    fadeOutDir: number;
    clipXFrom?: number;
    clipXTo?: number;
}): FadeHitTarget[] {
    const {
        clipLeftPx,
        clipWidthPx,
        bodyTop,
        bodyHeight,
        fadeInShape,
        fadeInDir,
        fadeOutShape,
        fadeOutDir,
    } = args;
    const solidWidth = Math.max(1, clipWidthPx);
    const clipRightPx = clipLeftPx + solidWidth;
    const clipXFrom = args.clipXFrom ?? Number.NEGATIVE_INFINITY;
    const clipXTo = args.clipXTo ?? Number.POSITIVE_INFINITY;
    const bodyH = Math.max(1, bodyHeight);
    const targets: FadeHitTarget[] = [];

    // 淡入：包络线（从左下到右上）+ 区域最右侧边缘竖线
    if (args.fadeInPx > 0) {
        const fadePx = Math.min(args.fadeInPx, solidWidth);
        const regionLeft = clipLeftPx;
        const regionRight = clipLeftPx + fadePx;
        for (const point of sampleFadeLine({
            left: regionLeft,
            right: regionRight,
            bodyTop,
            bodyHeight: bodyH,
            shape: fadeInShape,
            dir: fadeInDir,
            mode: "in",
            clipXFrom,
            clipXTo,
        })) {
            targets.push({
                kind: "line",
                type: "fade_in",
                ...squareAround(point.x, point.y, FADE_LINE_HIT_SIZE),
            });
        }
        if (regionRight >= clipXFrom && regionRight <= clipXTo) {
            targets.push({
                kind: "edge",
                type: "fade_in",
                left: regionRight - FADE_EDGE_HIT_WIDTH / 2,
                top: bodyTop,
                width: FADE_EDGE_HIT_WIDTH,
                height: bodyH,
            });
        }
    }

    // 淡出：包络线（从左上到右下）+ 区域最左侧边缘竖线
    if (args.fadeOutPx > 0) {
        const fadePx = Math.min(args.fadeOutPx, solidWidth);
        const regionLeft = clipRightPx - fadePx;
        const regionRight = clipRightPx;
        for (const point of sampleFadeLine({
            left: regionLeft,
            right: regionRight,
            bodyTop,
            bodyHeight: bodyH,
            shape: fadeOutShape,
            dir: fadeOutDir,
            mode: "out",
            clipXFrom,
            clipXTo,
        })) {
            targets.push({
                kind: "line",
                type: "fade_out",
                ...squareAround(point.x, point.y, FADE_LINE_HIT_SIZE),
            });
        }
        if (regionLeft >= clipXFrom && regionLeft <= clipXTo) {
            targets.push({
                kind: "edge",
                type: "fade_out",
                left: regionLeft - FADE_EDGE_HIT_WIDTH / 2,
                top: bodyTop,
                width: FADE_EDGE_HIT_WIDTH,
                height: bodyH,
            });
        }
    }

    return targets;
}
