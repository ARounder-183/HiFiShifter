/**
 * verticalScrollMapping.ts
 * 参数编辑器竖向滚动条与视口中心值之间的双向映射工具。
 */

export type VerticalScrollMappingInput = {
    min: number;
    max: number;
    span: number;
    scrollRangePx: number;
};

/**
 * 竖向边缘自动滚屏的目标速度：每秒移动多少个**可见值域高度**。
 *
 * 不能沿用水平轴的 420px/s：竖向 scrollTop 把整个值域映射到 1600px，
 * 常见缩放下一个视口只占值域的 10%–20%，同一个屏幕像素速度在值域上会被
 * 放大成数个视口/秒。这里按 span 归一化，保证不同缩放下手感一致。
 */
export const VALUE_EDGE_SCROLL_VIEWPORTS_PER_SEC = 0.30;

/** 值域是否还有可移动的居中范围；span >= range 时滚动条不应再产生位移。 */
export function valueAxisScrollable(input: {
    readonly min: number;
    readonly max: number;
    readonly span: number;
}): boolean {
    const range = Math.max(0, input.max - input.min);
    return range - Math.max(0, input.span) > 1e-9;
}

/**
 * 竖向边缘滚屏的最大 scrollTop 速度（px/秒）。
 *
 * 推导：希望中心每秒移动 viewportsPerSec × span 个值单位；
 * scrollTop 每 1px 对应 (range-span)/scrollRangePx 个值单位，因此
 * scrollTop 速度 = viewportsPerSec × span × scrollRangePx / (range-span)。
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

function clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
}

function resolveBounds(input: VerticalScrollMappingInput): {
    clampedSpan: number;
    minCenter: number;
    maxCenter: number;
    movableCenterRange: number;
    clampedScrollRange: number;
} {
    const range = Math.max(1e-9, input.max - input.min);
    const clampedSpan = clamp(input.span, 1e-9, range);
    const minCenter = input.min + clampedSpan / 2;
    const maxCenter = input.max - clampedSpan / 2;
    const movableCenterRange = Math.max(0, maxCenter - minCenter);
    const clampedScrollRange = Math.max(0, input.scrollRangePx);

    return {
        clampedSpan,
        minCenter,
        maxCenter,
        movableCenterRange,
        clampedScrollRange,
    };
}

export function verticalScrollTopFromCenter(
    input: VerticalScrollMappingInput & { center: number },
): number {
    const { minCenter, maxCenter, movableCenterRange, clampedScrollRange } = resolveBounds(input);
    if (movableCenterRange <= 1e-9 || clampedScrollRange <= 0) {
        return 0;
    }

    const center = clamp(input.center, minCenter, maxCenter);
    const ratio = (maxCenter - center) / movableCenterRange;
    return clamp(ratio, 0, 1) * clampedScrollRange;
}

export function centerFromVerticalScrollTop(
    input: VerticalScrollMappingInput & { scrollTop: number },
): number {
    const { minCenter, maxCenter, movableCenterRange, clampedScrollRange } = resolveBounds(input);
    if (movableCenterRange <= 1e-9 || clampedScrollRange <= 0) {
        return (minCenter + maxCenter) / 2;
    }

    const ratio = clamp(input.scrollTop / clampedScrollRange, 0, 1);
    return maxCenter - ratio * movableCenterRange;
}
