/**
 * 时间轴渲染内核 · 网格实例构建
 *
 * 【主要内容】
 * 把刻度数组（`buildTimelineTicks` 产出的强 / 弱网格线，其 `contentPx` 已是内容坐标）
 * 裁剪到**构建窗口**并转成平面矩形实例：竖线位置吸附到设备像素栅格，线宽按物理像素
 * 折算（弱线 1 物理像素、强线 2 物理像素），竖直方向覆盖 `[0, contentBottomPx]`。
 *
 * 【作用】
 * 单 WebGL2 渲染器的网格由「一批纯色矩形实例 + 一次 draw call」绘制，取代既有的
 * SVG path 重建。几何以**内容坐标**产出、与视口无关（竖直方向不做视口裁剪），
 * 因此滚动帧只需更新 `u_viewOrigin` uniform，**不需要重建几何**——这是"滚动零重绘"
 * 在网格层的落点。
 *
 * 【与其他模块的关系】
 * - 上游：调用方（宿主视图）用 `buildTimelineTicks` 产出的刻度数组调用本模块，
 *   并负责按「视口 ± 横向余量」算好构建窗口（本模块只做窗口内裁剪）。
 * - 下游：GL 渲染器的 FLAT 模式把 `FlatInstance[]` 写入实例缓冲。
 * - 独立性：不依赖 DOM / React；刻度以结构化最小接口（`GridTickLike`）接收，
 *   与 `TimelineTick` 结构兼容但无编译期耦合。
 *
 * 【设计约束】
 * 1. 竖直方向恒为 `[0, contentBottomPx]`，**不按视口裁剪**：裁剪会让几何随滚动变化，
 *    破坏「滚动零重建」。屏幕外部分由 GPU 光栅化裁剪，成本可忽略。
 * 2. 线宽按**物理像素**折算：弱线 1 物理像素、强线 2 物理像素。分数 DPR 下若按
 *    CSS 像素取整，会出现"同一批线粗细不一"的观感问题。
 * 3. 矩形左缘对齐物理像素栅格（`round(x × dpr) / dpr`）：线与物理像素一一对应，
 *    不会因落点相位被抗锯齿摊成两条灰线。
 */

import type { FlatInstance, Rgba } from "./instanceTypes";

/** 弱网格线的物理像素宽度。 */
const WEAK_LINE_PHYSICAL_PX = 1;

/** 强网格线（小节线）的物理像素宽度。 */
const STRONG_LINE_PHYSICAL_PX = 2;

/**
 * 窗口外放容差（CSS px）。
 *
 * 作用：刻度位置与窗口边界同为浮点计算，恰好落在边界上的线不应因 1e-12 级误差
 * 被丢弃；容差只放宽"是否进入窗口"的判定，不改变实际绘制位置。
 */
const WINDOW_SLACK_PX = 1;

/**
 * 网格刻度输入。
 *
 * 与 `runtime/buildTimelineTicks.TimelineTick` 结构兼容（调用方可直接传入），
 * 但本模块只依赖这两个字段，避免编译期耦合到刻度构建模块。
 */
export interface GridTickLike {
    /** 内容坐标 x（CSS px），由 axis 投影得到。 */
    readonly contentPx: number;
    /** 是否作为强网格线（小节线）绘制。 */
    readonly isStrongGridLine: boolean;
}

/** 网格实例构建参数。 */
export interface GridInstanceArgs {
    /** 刻度数组（通常来自 `buildTimelineTicks`）。 */
    readonly ticks: readonly GridTickLike[];
    /** 构建窗口左缘（内容坐标 CSS px），通常 = 视口左缘 − 横向余量。 */
    readonly windowLeftPx: number;
    /** 构建窗口宽度（CSS px），通常 = 视口宽 + 2 × 横向余量。 */
    readonly windowWidthPx: number;
    /** 轨道内容底部边界（内容坐标 CSS px）；网格只画到该边界。 */
    readonly contentBottomPx: number;
    /** 设备像素比。 */
    readonly dpr: number;
    /** 弱网格线颜色。 */
    readonly weakRgba: Rgba;
    /** 强网格线（小节线）颜色。 */
    readonly strongRgba: Rgba;
}

/**
 * 构建网格线实例。
 *
 * 流程：
 * 1. 归一化窗口与 DPR（非法值回退到安全值）；
 * 2. 逐刻度判定是否落在 `[windowLeft − slack, windowRight + slack]`；
 * 3. 产出矩形实例：`x` 吸附设备像素、`w` 按物理像素折算、`y/h` 覆盖 `[0, contentBottomPx]`。
 *
 * 特殊说明：`contentBottomPx <= 0` 时直接返回空数组（没有轨道内容就不画网格），
 * 避免产出高度为 0 的退化实例污染 draw call。
 *
 * @param args 构建参数（见 GridInstanceArgs）。
 * @returns 平面矩形实例数组（顺序与输入刻度一致）。
 */
export function buildGridInstances(args: GridInstanceArgs): FlatInstance[] {
    const dpr = Number.isFinite(args.dpr) && args.dpr > 0 ? args.dpr : 1;
    const windowLeft = Number.isFinite(args.windowLeftPx) ? args.windowLeftPx : 0;
    const windowWidth = Number.isFinite(args.windowWidthPx) ? Math.max(0, args.windowWidthPx) : 0;
    const windowRight = windowLeft + windowWidth;
    const contentBottom = Number.isFinite(args.contentBottomPx)
        ? Math.max(0, args.contentBottomPx)
        : 0;
    if (contentBottom <= 0) return [];

    const out: FlatInstance[] = [];
    for (const tick of args.ticks) {
        const x = tick.contentPx;
        if (!Number.isFinite(x)) continue;
        if (x < windowLeft - WINDOW_SLACK_PX || x > windowRight + WINDOW_SLACK_PX) continue;

        const strong = tick.isStrongGridLine === true;
        const physicalWidth = strong ? STRONG_LINE_PHYSICAL_PX : WEAK_LINE_PHYSICAL_PX;
        out.push({
            // 左缘吸附物理像素栅格：线宽恒为整数物理像素，不因落点相位变粗变虚。
            x: Math.round(x * dpr) / dpr,
            y: 0,
            w: physicalWidth / dpr,
            h: contentBottom,
            rgba: strong ? args.strongRgba : args.weakRgba,
        });
    }
    return out;
}
