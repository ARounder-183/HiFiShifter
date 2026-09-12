/**
 * 时间轴渲染内核 · 自绘滚动条几何
 *
 * 【主要内容】
 * 由「内容尺寸 / 视口尺寸 / 当前滚动位置 / 滚动上限」算出滚动条的轨道与 thumb 几何，
 * 并提供 thumb 命中测试、thumb 拖拽位移换算与轨道点击跳转。滚动条本体用 DOM 绘制
 * （拖拽命中与悬停态成本最低），但几何是纯计算，单独成模块以便单测。
 *
 * 【作用】
 * 自绘滚动取代原生 scroller 后，滚动条也必须自绘。几何计算里最容易出错的是两个
 * 边界：`maxScrollPx = 0`（内容不足一屏，thumb 应占满轨道且不可拖）与
 * `contentSize` 非法（除零）。本模块把这两类情况收敛成显式分支。
 *
 * 除几何外还承担「原生滚动条的等效交互」：拖 thumb、**点轨道翻页**——旧实现用
 * 原生滚动条时这些由浏览器提供，自绘后必须逐条补上。
 *
 * 【与其他模块的关系】
 * - 上游：宿主视图从 `ScrollKernel` 读 `contentWidthPx()/maxScrollLeft()` 等值后调用。
 * - 下游：宿主把几何写到滚动条 DOM 的 `style`，并用 `hitTestScrollbarThumb` 判定拖拽
 *   起点、用 `scrollTargetFromTrackClick` 处理轨道点击。
 * - 独立性：纯函数，不依赖 DOM / React。
 */

/** 滚动条几何（CSS px，均为相对轨道起点的坐标）。 */
export interface ScrollbarGeometry {
    /** 轨道长度（= 宿主在该轴上的可视尺寸）。 */
    readonly trackLengthPx: number;
    /** thumb 起点（相对轨道起点）。 */
    readonly thumbStartPx: number;
    /** thumb 长度。 */
    readonly thumbLengthPx: number;
    /** 是否可滚动（内容超出视口且有有效上限）。 */
    readonly scrollable: boolean;
}

/** 滚动条几何计算参数。 */
export interface ScrollbarArgs {
    /** 内容总尺寸（CSS px）。 */
    readonly contentSizePx: number;
    /** 视口尺寸（CSS px）。 */
    readonly viewportSizePx: number;
    /** 当前滚动位置（CSS px）。 */
    readonly scrollPx: number;
    /** 滚动上限（CSS px，= 0 表示不可滚动）。 */
    readonly maxScrollPx: number;
    /** thumb 最小长度（CSS px），保证极小比例下仍可抓取。 */
    readonly minThumbLengthPx?: number;
}

/** thumb 最小长度默认值。 */
const DEFAULT_MIN_THUMB_LENGTH_PX = 24;

/**
 * 计算滚动条几何。
 *
 * 流程：
 * 1. 归一化输入（非法值归零，尺寸取非负）；
 * 2. `maxScrollPx <= 0` 或视口 >= 内容 → 不可滚动：thumb 占满轨道、起点 0；
 * 3. 否则 thumb 长度 = `轨道长 × 视口 / 内容`（不小于最小长度、不大于轨道长），
 *    起点按 `scroll / maxScroll` 在「轨道长 − thumb 长」的可用行程内线性映射。
 *
 * 特殊说明：`maxScrollPx <= 0` 时**不做除法**（避免除零产生 NaN 几何）；
 * thumb 起点会被夹取到 `[0, 轨道长 − thumb 长]`，防止滚动位置越界时 thumb 画出轨道外。
 *
 * @param args 计算参数。
 * @returns 滚动条几何；调用方据此写 DOM 样式。
 */
export function computeScrollbar(args: ScrollbarArgs): ScrollbarGeometry {
    const trackLength = Math.max(0, toFinite(args.viewportSizePx));
    const contentSize = Math.max(0, toFinite(args.contentSizePx));
    const maxScroll = Math.max(0, toFinite(args.maxScrollPx));
    const scroll = Math.max(0, Math.min(maxScroll, toFinite(args.scrollPx)));
    const minThumb = Math.max(
        1,
        toFinite(args.minThumbLengthPx ?? DEFAULT_MIN_THUMB_LENGTH_PX) ||
            DEFAULT_MIN_THUMB_LENGTH_PX,
    );

    if (maxScroll <= 0 || contentSize <= trackLength || trackLength <= 0) {
        return {
            trackLengthPx: trackLength,
            thumbStartPx: 0,
            thumbLengthPx: trackLength,
            scrollable: false,
        };
    }

    const ratio = trackLength / contentSize;
    const thumbLength = Math.min(trackLength, Math.max(minThumb, trackLength * ratio));
    const travel = Math.max(0, trackLength - thumbLength);
    const thumbStart = Math.min(travel, Math.max(0, (scroll / maxScroll) * travel));
    return {
        trackLengthPx: trackLength,
        thumbStartPx: thumbStart,
        thumbLengthPx: thumbLength,
        scrollable: true,
    };
}

/**
 * 判定指针是否落在 thumb 上（用于拖拽起点识别）。
 *
 * @param pointerOffsetPx 指针相对轨道起点的偏移（CSS px）。
 * @param geometry 滚动条几何。
 * @returns 落在 thumb 上（含边界）时为 true；不可滚动时恒为 false。
 */
export function hitTestScrollbarThumb(
    pointerOffsetPx: number,
    geometry: ScrollbarGeometry,
): boolean {
    if (!geometry.scrollable) return false;
    const offset = toFinite(pointerOffsetPx);
    return (
        offset >= geometry.thumbStartPx && offset <= geometry.thumbStartPx + geometry.thumbLengthPx
    );
}

/**
 * 把 thumb 拖拽位移换算为滚动位置变化量。
 *
 * 流程：位移按「可用行程 / 轨道可用行程」的比例放大到内容滚动空间。
 *
 * 特殊说明：不可滚动时返回 0（调用方无需特判）。
 *
 * @param deltaThumbPx thumb 的位移（CSS px，向右/向下为正）。
 * @param geometry 滚动条几何。
 * @param maxScrollPx 滚动上限（CSS px）。
 * @returns 滚动位置增量（CSS px，可能为负）。
 */
export function scrollDeltaFromThumbDrag(
    deltaThumbPx: number,
    geometry: ScrollbarGeometry,
    maxScrollPx: number,
): number {
    if (!geometry.scrollable) return 0;
    const travel = geometry.trackLengthPx - geometry.thumbLengthPx;
    if (travel <= 0) return 0;
    return (toFinite(deltaThumbPx) / travel) * Math.max(0, toFinite(maxScrollPx));
}

/**
 * 点击滚动条轨道（非 thumb 区域）时的目标滚动位置。
 *
 * 【为什么要这个函数】旧实现用**原生**滚动条，点击轨道空白由浏览器按平台默认
 * 处理（大多数平台 = 向该方向翻一页，约一个视口；部分平台支持「点住轨道持续
 * 滚动」）。内核自绘滚动条后必须自己补上这个语义，否则「点轨道没反应」——
 * 用户失去除拖 thumb 与滚轮之外的第三种定位手段。
 *
 * 语义（与浏览器默认一致的最小集）：
 * - thumb **之后**（下/右方）→ 当前位置 + 一页；
 * - thumb **之前**（上/左方）→ 当前位置 − 一页；
 * - 落在 thumb 上 → 返回 `null`（那是拖拽起点，不是跳转）；
 * - 不可滚动 → 返回 `null`。
 *
 * 结果由调用方交给 `ScrollKernel` 钳制（与其它写入路径同一约定：上限不在
 * 这里算，避免两份上限来源）。
 *
 * @param pointerOffsetPx 指针相对轨道起点的偏移（CSS px）。
 * @param geometry 滚动条几何。
 * @param currentScrollPx 当前滚动位置（CSS px）。
 * @param pageSizePx 一页的跨度（CSS px，通常 = 视口尺寸）。
 * @returns 目标滚动位置；不应跳转时为 null。
 */
export function scrollTargetFromTrackClick(
    pointerOffsetPx: number,
    geometry: ScrollbarGeometry,
    currentScrollPx: number,
    pageSizePx: number,
): number | null {
    if (!geometry.scrollable) return null;
    const offset = toFinite(pointerOffsetPx);
    const thumbEnd = geometry.thumbStartPx + geometry.thumbLengthPx;
    if (offset >= geometry.thumbStartPx && offset <= thumbEnd) return null;
    const page = Math.max(0, toFinite(pageSizePx));
    if (page <= 0) return null;
    const current = toFinite(currentScrollPx);
    // 翻一页；`offset > thumbEnd` 即「thumb 之后」（下/右方）。
    return offset > thumbEnd ? current + page : current - page;
}

/** 归一化数值：非法值返回 0。 */
function toFinite(value: number): number {
    return Number.isFinite(value) ? value : 0;
}
