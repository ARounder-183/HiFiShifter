/**
 * 时间轴渲染内核 · clip header 控件命中
 *
 * 【主要内容】
 * 把「clip header 内的局部坐标」映射为具体控件（静音 / 链 / 共振峰 / 增益旋钮 /
 * 增益标签 / 速率标签 / 名称区）。
 *
 * 【作用】
 * 内核模式下 clip 是**自绘**的（GL 块面 + Canvas2D 细节层），header 控件没有 DOM
 * 可点。本模块让「**看到的 = 可点的**」成立：它**直接消费
 * `buildTimelineClipVisualStyle` 的返回值**（绘制端也用同一个函数），而不是另抄
 * 一套偏移量——两份位置常量迟早漂移，而漂移的表现是「点不准」这种极难归因的问题。
 *
 * 【与其他模块的关系】
 * - 上游：宿主在 `hitTest` 判定为 `header` 分区后调用（见 `host/timelineKernelHost`）。
 * - 依赖：`ClipHeaderControlStyle` 是 `buildTimelineClipVisualStyle` 返回值的
 *   **结构化子集**（只声明本模块用到的字段），因此不需要导入运行时值，本模块保持
 *   纯函数、可直接单测。
 * - 独立性：无 DOM / React 依赖。
 *
 * 【坐标系】
 * `localX` 以 **clip 左边缘**为原点，`localY` 以 **clip 顶边**为原点——与
 * `buildTimelineClipVisualStyle` 的偏移常量同一坐标系（渲染器里是
 * `clipLeft + style.xxxOffsetX`）。
 *
 * 【判定优先级】
 * 纵向越界 → 左侧徽标（矩形）→ 增益旋钮（圆形）→ 速率标签 → 增益标签 → 名称区。
 *
 * 两个刻意的顺序决定：
 * 1. **徽标先于旋钮**：旋钮的圆形判定带容差，若先判旋钮会吃掉紧邻的链徽标左缘。
 * 2. **速率先于增益**：两个标签相邻且速率在增益左侧，先判增益会把速率的左半边吞掉。
 */

/** header 内的可交互控件；`null` = 未命中任何控件。 */
export type ClipHeaderControl =
    | "mute"
    | "formant"
    | "chain"
    | "gain-knob"
    | "gain-label"
    | "rate-label"
    | "name"
    | null;

/**
 * 控件命中所需的样式字段。
 *
 * 刻意声明为独立接口（而非导入 `buildTimelineClipVisualStyle` 的返回类型）：
 * 后者字段众多且含颜色，导入会把命中模块与配色实现绑死；结构化类型让
 * 样式对象可直接传入，同时把「命中依赖哪些几何字段」显式记录下来。
 */
export interface ClipHeaderControlStyle {
    readonly showMuteBadge: boolean;
    readonly showChainBadge: boolean;
    readonly showFormantBadge: boolean;
    readonly showGainKnob: boolean;
    readonly showGainLabel: boolean;
    readonly showPlaybackRate: boolean;
    readonly showName: boolean;
    readonly muteBadgeWidth: number;
    readonly muteBadgeHeight: number;
    readonly muteBadgeOffsetX: number;
    readonly muteBadgeOffsetY: number;
    readonly chainBadgeWidth: number;
    readonly chainBadgeHeight: number;
    readonly chainBadgeOffsetX: number;
    readonly chainBadgeOffsetY: number;
    readonly formantBadgeWidth: number;
    readonly formantBadgeHeight: number;
    readonly formantBadgeOffsetX: number;
    readonly formantBadgeOffsetY: number;
    readonly gainKnobCenterOffsetX: number;
    readonly gainKnobCenterOffsetY: number;
    readonly gainKnobRadius: number;
    /** 左侧控件区右缘（名称区的起点）。 */
    readonly leadingControlsWidth: number;
    /** 右侧预留宽度（名称区的终点）。 */
    readonly trailingReservePx: number;
    /** 增益标签像素宽度（由样式解析给出，见 `buildTimelineClipVisualStyle`）。 */
    readonly gainLabelWidth: number;
    /** 速率标签像素宽度。 */
    readonly rateLabelWidth: number;
}

/** 控件命中参数。 */
export interface ClipHeaderControlArgs {
    /** clip 内相对 x（以 clip 左边缘为 0，CSS px）。 */
    readonly localX: number;
    /** clip 内相对 y（以 clip 顶边为 0，CSS px）。 */
    readonly localY: number;
    /** clip 的像素宽度（用于右对齐标签与名称区计算）。 */
    readonly clipWidthPx: number;
    readonly style: ClipHeaderControlStyle;
    /** header 高度（CSS px）：超出则不命中任何控件。缺省 16。 */
    readonly headerHeightPx?: number;
}

/** 徽标命中容差（CSS px）：比视觉边界各外扩一点，短按更容易点中。 */
const BADGE_SLOP_PX = 1;

/** 旋钮命中容差（CSS px）：圆形判定外扩，但比徽标小以免吃掉相邻徽标。 */
const KNOB_SLOP_PX = 2;

/** 标签命中区半高（以基线中心 ± 该值）。 */
const LABEL_HIT_HALF_HEIGHT_PX = 7;

/** 标签基线中心的 y 偏移（与绘制端 `clipTop + 9` 同源）。 */
const LABEL_BASELINE_OFFSET_Y = 9;

/** 标签与 clip 右缘的间距（与绘制端 `clipWidth - width - 6` 同源）。 */
const GAIN_LABEL_RIGHT_INSET_PX = 6;

/** 速率标签与增益标签的间距（与绘制端 `gainX - width - 8` 同源）。 */
const RATE_LABEL_GAP_PX = 8;

/** 名称区的最小可用宽度（与绘制端「> 12px 才画」同源）。 */
const NAME_MIN_WIDTH_PX = 12;

/**
 * 判定点是否落在矩形内（含外扩容差）。
 *
 * @param x 点 x。@param y 点 y。
 * @param rx 矩形左上角 x。@param ry 矩形左上角 y。
 * @param rw 矩形宽。@param rh 矩形高。
 * @param slop 外扩容差（CSS px）。
 * @returns 命中为 true。
 */
function inRect(
    x: number,
    y: number,
    rx: number,
    ry: number,
    rw: number,
    rh: number,
    slop: number,
): boolean {
    return x >= rx - slop && x <= rx + rw + slop && y >= ry - slop && y <= ry + rh + slop;
}

/**
 * 命中 clip header 内的控件。
 *
 * @param args 命中参数（局部坐标 + 样式 + 宽度）。
 * @returns 命中的控件；未命中任何控件时为 null。
 */
export function hitClipHeaderControl(args: ClipHeaderControlArgs): ClipHeaderControl {
    const headerHeightPx = Number.isFinite(args.headerHeightPx)
        ? Math.max(0, args.headerHeightPx as number)
        : 16;
    // 纵向越界：落在 body 区，不属于任何 header 控件（body 有自己的手势）。
    if (args.localY < 0 || args.localY > headerHeightPx) return null;

    const style = args.style;
    const x = args.localX;
    const y = args.localY;
    const clipWidthPx = Math.max(1, args.clipWidthPx);

    // ── 左侧徽标（矩形，顺序：静音 → 共振峰 → 链）──
    // 三者互不重叠，顺序只影响「边界被容差覆盖」时的归属；取视觉上更常用的在前。
    if (
        style.showMuteBadge &&
        inRect(
            x,
            y,
            style.muteBadgeOffsetX,
            style.muteBadgeOffsetY,
            style.muteBadgeWidth,
            style.muteBadgeHeight,
            BADGE_SLOP_PX,
        )
    ) {
        return "mute";
    }
    if (
        style.showFormantBadge &&
        inRect(
            x,
            y,
            style.formantBadgeOffsetX,
            style.formantBadgeOffsetY,
            style.formantBadgeWidth,
            style.formantBadgeHeight,
            BADGE_SLOP_PX,
        )
    ) {
        return "formant";
    }
    if (
        style.showChainBadge &&
        inRect(
            x,
            y,
            style.chainBadgeOffsetX,
            style.chainBadgeOffsetY,
            style.chainBadgeWidth,
            style.chainBadgeHeight,
            BADGE_SLOP_PX,
        )
    ) {
        return "chain";
    }

    // ── 增益旋钮（圆形）──
    if (style.showGainKnob) {
        const dx = x - style.gainKnobCenterOffsetX;
        const dy = y - style.gainKnobCenterOffsetY;
        const radius = Math.max(0, style.gainKnobRadius) + KNOB_SLOP_PX;
        if (dx * dx + dy * dy <= radius * radius) return "gain-knob";
    }

    // ── 右侧标签（右对齐，基线中心 y = 9）──
    if (style.showGainLabel) {
        const gainRight = clipWidthPx - GAIN_LABEL_RIGHT_INSET_PX;
        const gainLeft = gainRight - Math.max(0, style.gainLabelWidth);
        if (style.showPlaybackRate) {
            const rateRight = gainLeft - RATE_LABEL_GAP_PX;
            const rateLeft = rateRight - Math.max(0, style.rateLabelWidth);
            if (
                x >= rateLeft - BADGE_SLOP_PX &&
                x <= rateRight + BADGE_SLOP_PX &&
                Math.abs(y - LABEL_BASELINE_OFFSET_Y) <= LABEL_HIT_HALF_HEIGHT_PX
            ) {
                return "rate-label";
            }
        }
        if (
            x >= gainLeft - BADGE_SLOP_PX &&
            x <= gainRight + BADGE_SLOP_PX &&
            Math.abs(y - LABEL_BASELINE_OFFSET_Y) <= LABEL_HIT_HALF_HEIGHT_PX
        ) {
            return "gain-label";
        }
    }

    // ── 名称区（左侧控件与右侧标签之间）──
    if (!style.showName) return null;
    const nameLeft = style.leadingControlsWidth;
    const nameRight = clipWidthPx - style.trailingReservePx + 4;
    // 可用宽度过窄时视为无名称区（绘制端同样要求 > 12px 才画名称）。
    if (nameRight - nameLeft <= NAME_MIN_WIDTH_PX) return null;
    if (x >= nameLeft && x <= nameRight) return "name";

    return null;
}
