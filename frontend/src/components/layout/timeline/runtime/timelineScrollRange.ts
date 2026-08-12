/**
 * 水平时间轴的“虚拟滚动范围”。
 *
 * 工程内容本身只占 [0, contentWidth]，但为了让“以鼠标/播放光标为缩放中心”
 * 在任意缩放阶段都能成立，需要允许视口向右侧越出工程边界：
 * - 左侧与 DAW 通用设计一致：最多只能与工程起点重合，不允许展示工程起点左侧区域；
 * - 右侧始终允许滚动到工程右端（缩放中心靠近工程右端时，工程左端会向左移出画面），
 *   且最大滚动位置 = 工程宽度，随缩放连续变化，避免跨过“工程宽=视口宽”时上限骤降产生跳变。
 *
 * 原生 scrollLeft 与有效滚动位置一致（最小值为 0）；内容外层仅在工程小于视口时
 * 增加右侧虚拟宽度，让原生滚动范围能够覆盖到工程右端。
 */
export function resolveTimelineScrollRange(args: { contentWidth: number; viewportWidth: number }): {
    /** 有效滚动位置的最小值（0 = 工程起点与画面左缘重合）。 */
    minScrollLeft: number;
    /** 有效滚动位置的最大值。 */
    maxScrollLeft: number;
    /** 内容外层（含右侧余量）的宽度。 */
    paddedContentWidth: number;
} {
    const viewport = Math.max(1, args.viewportWidth);
    const content = Math.max(0, args.contentWidth);
    // 最大滚动位置始终 = 工程宽度：
    // - 工程比视口小时，允许滚动到工程右端，避免缩放锚点被“内容宽度=工程全长”卡死；
    // - 工程比视口大时，允许继续向工程右侧的空余区域滚动；
    // 该公式在 content = viewport 处连续，逐步放大/缩小时不会出现滚动上限跳变。
    const maxEffective = content;
    return {
        minScrollLeft: 0,
        maxScrollLeft: maxEffective,
        paddedContentWidth: maxEffective + viewport,
    };
}

/**
 * “基于播放光标缩放”时的水平偏移计算。
 *
 * 播放光标在画面内：以播放光标当前位置为缩放锚点（不校正到鼠标位置）。
 * 播放光标不在画面内：调整水平偏移，使播放光标出现在画面正中心。
 */
export function resolvePlayheadZoomScrollLeft(args: {
    playheadSec: number;
    basePxPerSec: number;
    baseScrollLeft: number;
    nextPxPerSec: number;
    viewportWidth: number;
}): number {
    const playheadScreenX = args.playheadSec * args.basePxPerSec - args.baseScrollLeft;
    if (playheadScreenX >= 0 && playheadScreenX <= args.viewportWidth) {
        return args.playheadSec * args.nextPxPerSec - playheadScreenX;
    }
    return args.playheadSec * args.nextPxPerSec - args.viewportWidth / 2;
}
