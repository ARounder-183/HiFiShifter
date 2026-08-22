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
    /** 绘制坐标视口相对“对齐基准视口”的屏幕偏移；参数编辑器同步时为轨道头偏移。 */
    viewportOffsetPx?: number;
}): number {
    const offset = Number.isFinite(args.viewportOffsetPx) ? (args.viewportOffsetPx as number) : 0;
    const playheadScreenX =
        args.playheadSec * args.basePxPerSec - args.baseScrollLeft - offset;
    if (playheadScreenX >= 0 && playheadScreenX <= args.viewportWidth) {
        return args.playheadSec * args.nextPxPerSec - playheadScreenX - offset;
    }
    return args.playheadSec * args.nextPxPerSec - args.viewportWidth / 2 - offset;
}

function clampNumber(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

/**
 * 统一的水平滚轮缩放计算（以“秒”为单位，轨道视图与参数编辑器共用）。
 *
 * - 未启用“基于播放光标缩放”：以鼠标所在位置为锚点；
 * - 启用后：播放光标在画面内以光标当前位置为锚点，不在画面内则校正到画面正中心；
 * - 最终滚动位置使用“只向右延长”的平滑范围（maxScrollLeft = 工程宽度）。
 */
export function resolveHorizontalWheelZoom(args: {
    factor: number;
    basePxPerSec: number;
    baseScrollLeft: number;
    totalSec: number;
    viewportWidth: number;
    playheadZoomEnabled: boolean;
    playheadSec: number | null;
    anchorScreenX: number;
    minPxPerSec: number;
    maxPxPerSec: number;
    /** 有效滚动位置的最小值；轨道视图为 0，参数编辑器同步时为 -偏移。 */
    minScrollLeft?: number;
    /** 参数编辑器相对轨道视图的水平屏幕偏移（同步对齐用）；轨道视图传 0。 */
    anchorOffsetPx?: number;
}): { nextPxPerSec: number; nextScrollLeft: number } | null {
    const {
        factor,
        basePxPerSec,
        baseScrollLeft,
        totalSec,
        viewportWidth,
        playheadZoomEnabled,
        playheadSec,
        anchorScreenX,
        minPxPerSec,
        maxPxPerSec,
        minScrollLeft,
        anchorOffsetPx,
    } = args;
    if (!Number.isFinite(basePxPerSec) || basePxPerSec <= 0) return null;
    if (!Number.isFinite(factor) || factor <= 0) return null;

    const nextPxPerSec = clampNumber(basePxPerSec * factor, minPxPerSec, maxPxPerSec);
    if (Math.abs(nextPxPerSec - basePxPerSec) < 1e-9) return null;

    const viewport = Math.max(1, viewportWidth);
    let anchorX = clampNumber(anchorScreenX, 0, viewport);
    const anchorOffset = Number.isFinite(anchorOffsetPx) ? (anchorOffsetPx as number) : 0;

    let nextScrollLeft: number;
    if (playheadZoomEnabled && playheadSec != null) {
        nextScrollLeft = resolvePlayheadZoomScrollLeft({
            playheadSec: clampNumber(playheadSec, 0, totalSec),
            basePxPerSec,
            baseScrollLeft,
            nextPxPerSec,
            viewportWidth: viewport,
            viewportOffsetPx: anchorOffset,
        });
    } else {
        let anchorSec = (baseScrollLeft + anchorX) / Math.max(1e-9, basePxPerSec);
        if (anchorX - anchorOffset < 0) {
            // 光标位于参数编辑器左侧的同步对齐空白区（对应轨道视图屏幕左侧之外）时，
            // 把锚点提升到“轨道视图左缘对应的世界位置”：该世界位置在轨道视图屏幕 0 处、
            // 在参数编辑器屏幕 offset 处，两者都在屏幕内；缩放时两个视图都围绕各自的
            // 屏幕内锚点纯缩放，不再出现“缩放 + 水平移动”。
            anchorSec = (baseScrollLeft + anchorOffset) / Math.max(1e-9, basePxPerSec);
            anchorX = anchorOffset;
        }
        nextScrollLeft = anchorSec * nextPxPerSec - anchorX;
    }

    const range = resolveTimelineScrollRange({
        contentWidth: Math.max(0, totalSec) * nextPxPerSec,
        viewportWidth: viewport,
    });
    const minScroll = Number.isFinite(minScrollLeft) ? (minScrollLeft as number) : 0;
    return {
        nextPxPerSec,
        nextScrollLeft: clampNumber(nextScrollLeft, minScroll, range.maxScrollLeft),
    };
}
