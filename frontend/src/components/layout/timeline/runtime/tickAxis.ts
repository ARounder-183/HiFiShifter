/**
 * 刻度轴 —— 标尺与背景网格**取刻度用的视口**的唯一构造入口。
 *
 * 【为什么必须共用】时间轴与参数编辑器各自构造过一次，约定却不同：
 * - 时间轴把滚动位置量化到 `TICK_WINDOW_STEP_PX` 并把视口宽加一个步长，从而保证
 *   量化锚点 ≤ 真实滚动位置 < 锚点 + 步长，覆盖区间必然包含真实视口；
 * - 参数编辑器用**原始** `scrollLeft` 与**精确**视口宽，既无量化也无宽度补偿。
 *
 * 后者在缩放后会拿一个滞后的滚动位置去取刻度（React 的 scrollLeft 是 256px 量化
 * 提交的，最多滞后 255px），窗口不再覆盖真实视口 ⇒ 标尺文字整片消失。约定只应
 * 存在一处：这个文件。
 */

import { createTimelineAxis, type TimelineAxis } from "../../renderKernel/timelineAxis.js";
import { TICK_WINDOW_STEP_PX } from "./buildTimelineTicks.js";

/** 把真实滚动位置量化到刻度窗口锚点（负数同样成立：锚点 ≤ 真值 < 锚点 + 步长）。 */
export function quantizeTickAnchor(scrollLeftPx: number): number {
    if (!Number.isFinite(scrollLeftPx)) return 0;
    return Math.floor(scrollLeftPx / TICK_WINDOW_STEP_PX) * TICK_WINDOW_STEP_PX;
}

/**
 * 构造刻度轴。
 *
 * @param args.scrollLeftPx 当前真实滚动位置（**绘制坐标**；同步模式下可为负）。
 * @param args.viewportWidthPx 真实视口宽度（CSS px）。
 * @returns `anchorPx`（量化后的锚点，同时必须作为 `scrollLeft` 传给标尺，两处必须
 *   是同一个值，否则标尺切片的窗口与刻度生成的窗口不一致）与投影 `axis`。
 */
export function createTickAxis(args: {
    pxPerSec: number;
    scrollLeftPx: number;
    viewportWidthPx: number;
    dpr?: number;
}): { axis: TimelineAxis; anchorPx: number } {
    const anchorPx = quantizeTickAnchor(args.scrollLeftPx);
    const viewportWidthPx = Number.isFinite(args.viewportWidthPx)
        ? Math.max(0, args.viewportWidthPx)
        : 0;
    return {
        anchorPx,
        axis: createTimelineAxis({
            pxPerSec: args.pxPerSec,
            scrollLeftPx: anchorPx,
            // 宽度补一个量化步长：锚点可以落后真值接近一个步长。
            viewportWidthPx: viewportWidthPx + TICK_WINDOW_STEP_PX,
            dpr: args.dpr,
        }),
    };
}
