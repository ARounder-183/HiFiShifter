/**
 * 参数编辑器 · 值轴边缘自动滚动策略（纯函数）
 *
 * 【主要内容】
 * 1. 值轴专用的触发带宽、满带速度、饱和比例与软启动指数；
 * 2. 把「指针在上下边缘的位置 + 当前可见值域 + scrollTop 映射范围」解析为
 *    本帧 scrollTop 步长；
 * 3. 值域不可移动（span >= range）时返回 0，供驱动同时用于速度与可滚性判断。
 *
 * 【为什么单独成模块】
 * 值轴与时间轴的手感目标不同：时间轴按屏幕像素/秒思考，值轴必须按**可见值域
 * 的百分比**思考。否则同一个 scrollTop 速度在 6 半音视口里会变成 27 半音/秒，
 * 在 48 半音视口里又慢得几乎不动。把这个换算集中在一处，两个参数编辑器驱动
 * （块拖动与绘制类手势）共享同一份算术，避免再次出现「一个驱动 0.30、另一个
 * 回退到 420px/s」的分裂。
 *
 * 【与其它模块的关系】
 * - 上游：`gestures/canvasPointerDown.ts` 的两个 `createEdgeScrollDriver`；
 * - 下游：`shared/edgeAutoScroll.ts` 的通用带宽/比例形状；
 * - `verticalScrollMapping.ts` 只保留值域 ↔ scrollTop 的映射，并 re-export
 *   本模块的策略函数，旧调用方无需改 import。
 *
 * 【设计约束】
 * - 所有输入非有限时返回 0，绝不把 NaN 写进 scrollTop；
 * - 速度按「每秒移动多少个可见 span」定义，因此缩放时屏幕速度恒定；
 * - 指针拖出视口后不继续加速（maxRatio=1），值轴不需要“甩出去”的手感。
 */
import { resolveAxisEdgeScrollDeltaPx } from "../shared/edgeAutoScroll";

/** 值轴边缘带宽：只占 18px，避免块靠近上下边时误触发。 */
export const VALUE_EDGE_SCROLL_BAND_PX = 18;

/** 满带速度：每秒移动 0.12 个可见值域高度。 */
export const VALUE_EDGE_SCROLL_VIEWPORTS_PER_SEC = 0.12;

/** 拖出视口后不再加速；值轴不需要时间轴那种 1.5 倍“甩出去”的手感。 */
export const VALUE_EDGE_SCROLL_MAX_RATIO = 1;

/** 软启动：靠近带内缘时速度按 ratio^1.6 增长。 */
export const VALUE_EDGE_SCROLL_RAMP_EXPONENT = 1.6;

/** 值域是否还有可移动的居中范围；span >= range 时滚动条不应再产生位移。 */
export function valueAxisScrollable(input: {
    readonly min: number;
    readonly max: number;
    readonly span: number;
}): boolean {
    if (
        !Number.isFinite(input.min) ||
        !Number.isFinite(input.max) ||
        !Number.isFinite(input.span)
    ) {
        return false;
    }
    const range = Math.max(0, input.max - input.min);
    return range - Math.max(0, input.span) > 1e-9;
}

/**
 * 满带时每秒的 scrollTop 位移（px/s）。
 *
 * 推导：希望中心每秒移动 `viewportsPerSec × span` 个值单位；scrollTop 每 1px
 * 对应 `(range - span) / scrollRangePx` 个值单位，因此
 * `scrollTop 速度 = viewportsPerSec × span × scrollRangePx / (range - span)`。
 *
 * 特殊说明：这里返回的是**满带**速度；带内比例与软启动由
 * `valueEdgeScrollDeltaPx` 统一施加，驱动不再自己乘一次比例。
 */
export function valueEdgeScrollMaxSpeedPxPerSec(input: {
    readonly min: number;
    readonly max: number;
    readonly span: number;
    readonly scrollRangePx: number;
    readonly viewportsPerSec?: number;
}): number {
    if (
        !Number.isFinite(input.min) ||
        !Number.isFinite(input.max) ||
        !Number.isFinite(input.span) ||
        !Number.isFinite(input.scrollRangePx)
    ) {
        return 0;
    }
    const range = Math.max(0, input.max - input.min);
    const span = Math.min(range, Math.max(0, input.span));
    const movable = Math.max(0, range - span);
    const scrollRange = Math.max(0, input.scrollRangePx);
    const viewports =
        input.viewportsPerSec !== undefined && Number.isFinite(input.viewportsPerSec)
            ? Math.max(0, input.viewportsPerSec)
            : VALUE_EDGE_SCROLL_VIEWPORTS_PER_SEC;
    if (movable <= 1e-9 || scrollRange <= 0 || viewports <= 0) return 0;
    return (viewports * span * scrollRange) / movable;
}

/**
 * 值轴指针位置 → 本帧 scrollTop 步长（正数向下、负数向上）。
 *
 * 流程：
 * 1. 用可见 span 与值域范围算出满带 scrollTop 速度；
 * 2. 交给共享的 `resolveAxisEdgeScrollDeltaPx` 施加 18px 带宽、1.0 饱和比例、
 *    1.6 软启动与帧时长；
 * 3. span >= range 时满带速度为 0，函数自然返回 0。
 */
export function valueEdgeScrollDeltaPx(input: {
    readonly pointerPx: number;
    readonly topPx: number;
    readonly bottomPx: number;
    readonly frameMs: number;
    readonly min: number;
    readonly max: number;
    readonly span: number;
    readonly scrollRangePx: number;
    readonly bandPx?: number;
    readonly viewportsPerSec?: number;
    readonly maxRatio?: number;
    readonly rampExponent?: number;
}): number {
    const maxSpeedPxPerSec = valueEdgeScrollMaxSpeedPxPerSec(input);
    if (maxSpeedPxPerSec <= 0) return 0;
    return resolveAxisEdgeScrollDeltaPx({
        pointerPx: input.pointerPx,
        minPx: input.topPx,
        maxPx: input.bottomPx,
        frameMs: input.frameMs,
        maxSpeedPxPerSec,
        bandPx: input.bandPx ?? VALUE_EDGE_SCROLL_BAND_PX,
        maxRatio: input.maxRatio ?? VALUE_EDGE_SCROLL_MAX_RATIO,
        rampExponent: input.rampExponent ?? VALUE_EDGE_SCROLL_RAMP_EXPONENT,
    });
}
