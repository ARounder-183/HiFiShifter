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
 * 【设计约束（与旧实现对齐，评审检查项）】
 * 1. 竖直方向恒为 `[0, contentBottomPx]`，**不按视口裁剪**：裁剪会让几何随滚动变化，
 *    破坏「滚动零重建」。屏幕外部分由 GPU 光栅化裁剪，成本可忽略。
 * 2. 线宽按 **CSS 像素**（弱 1 / 强 2），与旧实现 SVG 的 `strokeWidth` 同源。
 *    **不要**改成物理像素——Retina（dpr=2）下会只有旧实现的一半粗，观感明显偏细。
 * 3. **居中**语义：矩形左缘 = `x − 宽/2`，对应旧实现 SVG 的居中描边（`crispEdges`）。
 *    左缘仍吸附设备像素栅格（`round(left × dpr) / dpr`），保证线宽恒为整数物理像素。
 * 4. 颜色之外再叠一层整体透明度（旧实现 SVG 的 `opacity` 属性，默认 0.9）：
 *    只取颜色 alpha 会略亮。
 */

import type { FlatInstance, Rgba } from "./instanceTypes";

/**
 * 弱网格线的宽度（**CSS 像素**）。
 *
 * 与旧实现 SVG 的 `strokeWidth={1}` 一致：线宽以 CSS 像素定义，物理宽度由 DPR
 * 决定。内核曾按「1 物理像素」实现，在 Retina（dpr=2）上只有旧实现的一半粗，
 * 观感明显偏细——网格线宽必须跟旧实现同源。
 */
const WEAK_LINE_CSS_PX = 1;

/** 强网格线（小节线）的宽度（CSS 像素，对应旧实现 `strokeWidth={2}`）。 */
const STRONG_LINE_CSS_PX = 2;

/**
 * 线条整体透明度。
 *
 * 旧实现 SVG 在颜色之外还有一层 `opacity={lineOpacity}`（默认 0.9），与颜色自身的
 * alpha 相乘。内核只取颜色 alpha 会略亮，观感与旧实现不一致。
 */
const GRID_LINE_OPACITY = 0.9;

/**
 * 把 RGBA 的 alpha 再乘一层整体透明度。
 *
 * @param rgba 原始颜色。
 * @param opacity 整体透明度（0..1）。
 * @returns 应用后的颜色。
 */
function applyOpacity(rgba: Rgba, opacity: number): Rgba {
    return [rgba[0], rgba[1], rgba[2], rgba[3] * opacity];
}

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
    /**
     * 同一设备像素位置归并（键 = 吸附后的物理像素列）。
     *
     * 【为什么必须归并】刻度数组里**同一位置可能出现多条**（主/副时间单位在小节线
     * 处重合）。旧实现用 SVG path：重复坐标只是把同一段线描两次，`stroke` 不会
     * 累积透明度；而 GL 是**逐实例绘制**，同一位置的半透明矩形叠两次会让 alpha
     * 近似翻倍——这正是"网格线明显偏亮"的根因。
     *
     * 归并时保留**更强**的那条（强线优先），保证小节线不会退化成弱线宽度。
     */
    const byPosition = new Map<number, { x: number; strong: boolean }>();
    for (const tick of args.ticks) {
        const x = tick.contentPx;
        if (!Number.isFinite(x)) continue;
        if (x < windowLeft - WINDOW_SLACK_PX || x > windowRight + WINDOW_SLACK_PX) continue;

        const strong = tick.isStrongGridLine === true;
        const cssWidth = strong ? STRONG_LINE_CSS_PX : WEAK_LINE_CSS_PX;
        // 居中描边（与旧实现 SVG 的 stroke 语义一致）：矩形左缘 = 中心 − 半宽。
        // 吸附仍按设备像素栅格，保证线宽恒为整数物理像素、不因落点相位变虚。
        const snappedLeft = Math.round((x - cssWidth / 2) * dpr) / dpr;
        // 键用**中心**位置的物理像素列，不能用左缘：强线（宽 2）与弱线（宽 1）
        // 的左缘本就不同，用左缘作键会让本该重合的强弱线各画一条（又叠亮回去）。
        const key = Math.round(x * dpr);
        const existing = byPosition.get(key);
        if (existing === undefined || (strong && !existing.strong)) {
            byPosition.set(key, { x: snappedLeft, strong });
        }
    }

    for (const item of byPosition.values()) {
        const cssWidth = item.strong ? STRONG_LINE_CSS_PX : WEAK_LINE_CSS_PX;
        out.push({
            x: item.x,
            y: 0,
            w: cssWidth,
            h: contentBottom,
            rgba: applyOpacity(item.strong ? args.strongRgba : args.weakRgba, GRID_LINE_OPACITY),
        });
    }
    return out;
}
