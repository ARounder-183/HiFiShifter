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

/*
 * 值轴边缘滚动的常量与速度算术已经下沉到 `valueAxisEdgeScroll.ts`：那里同时
 * 负责带宽、软启动与不可滚判断，两个手势驱动共享。这里只 re-export，保留旧
 * import 路径（`verticalScrollMapping`）不破坏现有调用方。
 */
export {
    VALUE_EDGE_SCROLL_BAND_PX,
    VALUE_EDGE_SCROLL_MAX_RATIO,
    VALUE_EDGE_SCROLL_RAMP_EXPONENT,
    VALUE_EDGE_SCROLL_VIEWPORTS_PER_SEC,
    valueAxisScrollable,
    valueEdgeScrollDeltaPx,
    valueEdgeScrollMaxSpeedPxPerSec,
} from "./valueAxisEdgeScroll";

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
