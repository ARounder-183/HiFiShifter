/** 气泡锚点相对光标的偏移（右下方跟随）。 */
export const CURSOR_OFFSET_X = 14;
export const CURSOR_OFFSET_Y = 18;
/** 气泡与窗口边缘的最小间距。 */
export const EDGE_GAP = 8;

export type AppTooltipPosition = {
    x: number;
    y: number;
};

/**
 * 单轴锚定夹紧（所有光标锚定浮动元素的定位核心）。
 *
 * 余量必须来自元素自身尺寸：文案可从 2 个字到 400px 长链接不等，若预留
 * 固定宽度（历史实现的 320px），窗口右侧的控件悬停时气泡会被整段甩到
 * 光标左侧 —— 按实测尺寸夹紧后，元素始终贴着锚点，仅在即将出窗时收回
 * 必要距离；视口装不下时钳到最小间距。
 */
export function clampAxisPosition(
    anchor: number,
    elementSize: number,
    viewportSize: number,
    offset: number,
    edgeGap: number,
): number {
    return Math.min(anchor + offset, Math.max(edgeGap, viewportSize - elementSize - edgeGap));
}

/**
 * 气泡定位夹紧（按气泡实测尺寸，X/Y 双轴）。
 */
export function clampTooltipPosition(
    position: AppTooltipPosition,
    bubbleWidth: number,
    bubbleHeight: number,
    viewportWidth: number = window.innerWidth,
    viewportHeight: number = window.innerHeight,
): AppTooltipPosition {
    return {
        x: clampAxisPosition(position.x, bubbleWidth, viewportWidth, CURSOR_OFFSET_X, EDGE_GAP),
        y: clampAxisPosition(position.y, bubbleHeight, viewportHeight, CURSOR_OFFSET_Y, EDGE_GAP),
    };
}
