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
    const maxScrollLeft = Math.max(0, Math.max(0, contentWidth) - Math.max(0, viewportWidth));
    const target = Math.max(0, playheadX - Math.max(0, offsetPx));
    return Math.min(maxScrollLeft, target);
}
