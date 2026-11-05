/**
 * 时间轴 Clip header 的可视布局。
 *
 * 【为什么需要纯函数】重叠 Clip 的 header 文本此前只按自身矩形裁剪：
 * 后一个 Clip 的头部伸进前一个 Clip 里时，两份名称 / 徽标会画在同一片像素上。
 * 颜色怎么调都救不了，必须让绘制端与命中端共享同一条“可见 header 起点”。
 *
 * 【坐标系】`leadingOverlapPx` 是 clip 内相对左缘的前导重叠宽度。返回值
 * `nameLeftPx` / `trailingBoundPx` 也都是 clip 内相对坐标，供绘制端加
 * `clipLeft`、命中端直接用 `localX` 消费。
 *
 * 【为什么名称左界不直接等于控件右缘】前导重叠只压住 header 的**名称/尾部**，
 * 不压住左侧徽标与旋钮：那些控件仍属于后一个 Clip，并且用户需要点到它们。
 * 因此名称区取 `max(leadingControlsWidth, leadingOverlapPx)`；当重叠区宽到
 * 超过控件区时，名称自然向右退让，控件保持可点。
 */

export interface ClipHeaderLayoutArgs {
    /** Clip 自身宽度（clip 内相对坐标，CSS px）。 */
    readonly clipWidthPx: number;
    /** 前导重叠宽度（clip 内相对坐标，CSS px）。 */
    readonly leadingOverlapPx: number;
    /** 左侧控件区右缘（clip 内相对坐标）。 */
    readonly leadingControlsWidth: number;
    /** 右侧标签预留（clip 内相对坐标）。 */
    readonly trailingReservePx: number;
}

export interface ClipHeaderLayout {
    /** 被同轨前一个 Clip 遮住的 header 宽度（clip 内相对坐标）。 */
    readonly coveredWidthPx: number;
    /** 第一个**不与任何同轨前序 Clip 重叠**的 header x（clip 内相对坐标）。 */
    readonly visibleLeftPx: number;
    /** 名称区最小可用宽度；不够时不画、不命中名称。 */
    readonly nameMinWidthPx: number;
    /** 名称区左界（clip 内相对坐标）。 */
    readonly nameLeftPx: number;
    /** 名称区右界（clip 内相对坐标，**不含**右侧标签预留）。 */
    readonly nameRightPx: number;
    /** 名称区实际可用宽度（已收敛到 0）。 */
    readonly nameWidthPx: number;
    /** 名称区是否真的可绘制 / 可命中。 */
    readonly nameVisible: boolean;
}

/** 与绘制端“> 12px 才画名称”同一阈值。 */
export const CLIP_HEADER_NAME_MIN_WIDTH_PX = 12;

/**
 * 计算重叠区的 header 名称区。
 *
 * @param args 几何输入。
 * @returns 名称区边界与可见性。
 */
export function resolveClipHeaderLayout(args: ClipHeaderLayoutArgs): ClipHeaderLayout {
    const width = Number.isFinite(args.clipWidthPx) ? Math.max(1, args.clipWidthPx) : 1;
    const overlap = Number.isFinite(args.leadingOverlapPx)
        ? Math.max(0, Math.min(width, args.leadingOverlapPx))
        : 0;
    const controls = Number.isFinite(args.leadingControlsWidth)
        ? Math.max(0, args.leadingControlsWidth)
        : 0;
    const reserve = Number.isFinite(args.trailingReservePx)
        ? Math.max(0, args.trailingReservePx)
        : 0;

    const nameLeftPx = Math.max(controls, overlap);
    const nameRightPx = Math.max(0, width - reserve);
    const nameWidthPx = Math.max(0, nameRightPx - nameLeftPx);
    return {
        coveredWidthPx: overlap,
        visibleLeftPx: overlap,
        nameMinWidthPx: CLIP_HEADER_NAME_MIN_WIDTH_PX,
        nameLeftPx,
        nameRightPx,
        nameWidthPx,
        nameVisible: nameWidthPx > CLIP_HEADER_NAME_MIN_WIDTH_PX,
    };
}

/**
 * 控件是否整体落在本 Clip 的**可见 header 区**内。
 *
 * 【为什么不能只裁文本】重叠区里被盖住的不只是名称，左侧旋钮 / 徽标与右侧
 * gain/rate 标签也会被前一个 Clip 的块面压住。控件只要有任何一部分落在被遮挡段，
 * 就不能画、也不能命中，否则用户会点到“看不见的按钮”。
 *
 * @param args 控件矩形（clip 内相对坐标）。
 * @param layout 本 Clip 的 header 布局。
 * @returns 控件完整可见时为 true。
 */
export function isClipHeaderControlVisible(
    args: { readonly leftPx: number; readonly widthPx: number },
    layout: ClipHeaderLayout,
): boolean {
    const left = Number.isFinite(args.leftPx) ? args.leftPx : 0;
    const width = Number.isFinite(args.widthPx) ? Math.max(0, args.widthPx) : 0;
    return left >= layout.visibleLeftPx - 1e-6 && left + width > layout.visibleLeftPx;
}

/**
 * 右侧标签是否仍应绘制。
 *
 * 标签是右对齐的，重叠段从左侧压过来；只有重叠边界越过标签左缘时，标签才被
 * 遮挡。此函数用同一份 `visibleLeftPx` 判定，避免绘制的标签与可点区域分叉。
 */
export function isClipHeaderRightLabelVisible(
    args: { readonly leftPx: number },
    layout: ClipHeaderLayout,
): boolean {
    return args.leftPx >= layout.visibleLeftPx - 1e-6;
}
