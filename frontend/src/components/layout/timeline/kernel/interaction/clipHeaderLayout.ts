/**
 * 时间轴 Clip header 的可视布局。
 *
 * 【重叠语义】同轨重叠时，后一个 Clip 的 header 显示在重叠区上方；前一个
 * Clip 只在它未被覆盖的宽度里排版。前者的名称与右侧 dB / 倍率信息必须整体
 * 向左移动并截断，不能伸进重叠区被后一个块面盖住。
 *
 * 【为什么需要纯函数】绘制端、命中端与行内编辑定位都用这一份结果。三处各自
 * 计算边界，迟早会出现“文字左移了，但点击区还在原处”的漂移。
 *
 * 【坐标系】所有值都是 clip 内相对坐标。`trailingOverlapPx` 从右缘向左量；
 * 返回的 `visibleRightPx` 是第一个被后序 Clip 覆盖的 x。
 */

export interface ClipHeaderLayoutArgs {
    /** Clip 自身宽度（clip 内相对坐标，CSS px）。 */
    readonly clipWidthPx: number;
    /** 右侧被后序同轨 Clip 覆盖的宽度（clip 内相对坐标，CSS px）。 */
    readonly trailingOverlapPx: number;
    /** 左侧控件区右缘（clip 内相对坐标）。 */
    readonly leadingControlsWidth: number;
    readonly showGainLabel: boolean;
    readonly showPlaybackRate: boolean;
    readonly gainLabelWidth: number;
    readonly rateLabelWidth: number;
}

export interface ClipHeaderLayout {
    readonly trailingOverlapPx: number;
    readonly visibleRightPx: number;
    readonly contentRightPx: number;
    readonly nameMinWidthPx: number;
    readonly nameLeftPx: number;
    readonly nameRightPx: number;
    readonly nameWidthPx: number;
    readonly nameVisible: boolean;
    readonly gainLabelLeftPx: number;
    readonly gainLabelRightPx: number;
    readonly gainLabelVisible: boolean;
    readonly rateLabelLeftPx: number;
    readonly rateLabelRightPx: number;
    readonly rateLabelVisible: boolean;
}

/** 与绘制端“> 12px 才画名称”同一阈值。 */
export const CLIP_HEADER_NAME_MIN_WIDTH_PX = 12;
/** 右侧标签与 clip 可见右缘的内缩。 */
export const CLIP_HEADER_RIGHT_INSET_PX = 6;
/** 速率标签与增益标签之间的间距。 */
export const CLIP_HEADER_RATE_GAP_PX = 8;
/** 名称与尾部标签之间的最小间隔。 */
export const CLIP_HEADER_NAME_GAP_PX = 4;
/** 速率徽标（行内编辑锚点）的固定宽度（CSS px）。 */
export const CLIP_HEADER_RATE_BADGE_ANCHOR_WIDTH_PX = 72;

const EPS = 1e-6;

function finiteNonNegative(value: number): number {
    return Number.isFinite(value) ? Math.max(0, value) : 0;
}

/**
 * 计算一个 Clip 可见 header 的名称区与尾部标签位置。
 *
 * 后序 Clip 的 header 不会把自己左侧的控件藏起来；因此这里完全不看
 * `leadingOverlapPx`。前序 Clip 的尾部 `visibleRightPx` 由调用方给出，名称与
 * 右侧标签都收敛在它左边。
 */
export function resolveClipHeaderLayout(args: ClipHeaderLayoutArgs): ClipHeaderLayout {
    const width = Math.max(1, Number.isFinite(args.clipWidthPx) ? args.clipWidthPx : 1);
    const trailingOverlapPx = Math.min(width, finiteNonNegative(args.trailingOverlapPx));
    const visibleRightPx = Math.max(0, width - trailingOverlapPx);
    const controls = Math.min(visibleRightPx, finiteNonNegative(args.leadingControlsWidth));
    const contentRightPx = Math.max(0, visibleRightPx - CLIP_HEADER_RIGHT_INSET_PX);

    const gainWidth = finiteNonNegative(args.gainLabelWidth);
    const rateWidth = finiteNonNegative(args.rateLabelWidth);
    const gainLabelRightPx = contentRightPx;
    const gainLabelLeftPx = Math.max(0, gainLabelRightPx - gainWidth);
    const gainLabelVisible =
        args.showGainLabel &&
        gainWidth > EPS &&
        gainLabelLeftPx >= controls - EPS &&
        gainLabelRightPx - gainLabelLeftPx > EPS &&
        gainLabelRightPx <= visibleRightPx + EPS;

    const rateLabelRightPx = Math.max(0, gainLabelLeftPx - CLIP_HEADER_RATE_GAP_PX);
    const rateLabelLeftPx = Math.max(0, rateLabelRightPx - rateWidth);
    const rateLabelVisible =
        args.showPlaybackRate &&
        rateWidth > EPS &&
        rateLabelLeftPx >= controls - EPS &&
        rateLabelRightPx - rateLabelLeftPx > EPS &&
        rateLabelRightPx <= visibleRightPx + EPS;

    const labelsLeftPx = rateLabelVisible
        ? rateLabelLeftPx
        : gainLabelVisible
          ? gainLabelLeftPx
          : contentRightPx;
    const nameLeftPx = controls;
    const nameRightPx = Math.max(
        nameLeftPx,
        Math.min(contentRightPx, labelsLeftPx - CLIP_HEADER_NAME_GAP_PX),
    );
    const nameWidthPx = Math.max(0, nameRightPx - nameLeftPx);

    return {
        trailingOverlapPx,
        visibleRightPx,
        contentRightPx,
        nameMinWidthPx: CLIP_HEADER_NAME_MIN_WIDTH_PX,
        nameLeftPx,
        nameRightPx,
        nameWidthPx,
        nameVisible: nameWidthPx > CLIP_HEADER_NAME_MIN_WIDTH_PX,
        gainLabelLeftPx,
        gainLabelRightPx,
        gainLabelVisible,
        rateLabelLeftPx,
        rateLabelRightPx,
        rateLabelVisible,
    };
}

/**
 * 控件是否完整落在自身 Clip 的可见 header 区里。
 *
 * 跨在尾部遮挡边界上的控件不画也不可点：半截按钮既看不清，也容易误触。
 */
export function isClipHeaderControlVisible(
    args: { readonly leftPx: number; readonly widthPx: number },
    layout: ClipHeaderLayout,
): boolean {
    const left = Number.isFinite(args.leftPx) ? Math.max(0, args.leftPx) : 0;
    const width = Number.isFinite(args.widthPx) ? Math.max(0, args.widthPx) : 0;
    return width > 0 && left + width <= layout.visibleRightPx + EPS;
}

/** 右对齐文字是否仍完整落在可见 header 区。 */
export function isClipHeaderRightLabelVisible(
    args: { readonly leftPx: number; readonly widthPx?: number },
    layout: ClipHeaderLayout,
): boolean {
    const left = Number.isFinite(args.leftPx) ? Math.max(0, args.leftPx) : 0;
    const width = Number.isFinite(args.widthPx ?? 0) ? Math.max(0, args.widthPx ?? 0) : 0;
    return left >= -EPS && left + width <= layout.visibleRightPx + EPS;
}
