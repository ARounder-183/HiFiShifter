export const AUTO_FOLLOW_PLAYHEAD_OFFSET_PX = 12;

export function computeAutoFollowScrollLeft(args: {
    playheadSec: number;
    pxPerSec: number;
    viewportWidth: number;
    contentWidth: number;
    offsetPx?: number;
}): number {
    const {
        playheadSec,
        pxPerSec,
        viewportWidth,
        contentWidth,
        offsetPx = AUTO_FOLLOW_PLAYHEAD_OFFSET_PX,
    } = args;

    const playheadX = Math.max(0, playheadSec) * Math.max(0, pxPerSec);
    // 播放跟随语义：工程右端到达画面右缘后停止平移（DAW 通用行为），
    // 因此上限仍是“工程宽 − 视口宽”。
    const maxScrollLeft = Math.max(0, Math.max(0, contentWidth) - Math.max(0, viewportWidth));
    const target = Math.max(0, playheadX - Math.max(0, offsetPx));
    return Math.min(maxScrollLeft, target);
}

/**
 * “聚焦播放光标”的水平滚动位置。
 *
 * 与播放跟随不同：聚焦必须把光标带到视口内固定偏移处，即使光标接近或
 * 位于工程末尾也要成立。自 resolveTimelineScrollRange 引入“只向右延长”
 * 的滚动范围后，原生滚动上限 = 工程宽度（右侧虚拟余量允许视口越出
 * 工程边界），因此这里的钳制上限取工程宽度本身；若沿用旧的
 * “工程宽 − 视口宽”上限，接近工程末尾的光标会被卡在画面右缘、
 * 无法正确展示。
 */
export function computeFocusCursorScrollLeft(args: {
    playheadSec: number;
    pxPerSec: number;
    /** 工程（内容）总宽度，单位 px。 */
    contentWidth: number;
    offsetPx?: number;
}): number {
    const { playheadSec, pxPerSec, contentWidth } = args;
    const offsetPx = args.offsetPx ?? AUTO_FOLLOW_PLAYHEAD_OFFSET_PX;

    const playheadX = Math.max(0, playheadSec) * Math.max(0, pxPerSec);
    // 新滚动模型：最大滚动位置 = 工程宽度。
    const maxScrollLeft = Math.max(0, contentWidth);
    const target = Math.max(0, playheadX - Math.max(0, offsetPx));
    return Math.min(maxScrollLeft, target);
}
