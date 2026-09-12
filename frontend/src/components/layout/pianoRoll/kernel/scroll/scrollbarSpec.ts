/**
 * 参数编辑器内核 · 自绘滚动条几何装配。
 *
 * 【主要内容】
 * 把「视口尺寸 / 内容尺寸 / 当前滚动 / 滚动上限」装配成水平与竖直两条滚动条的几何，
 * 供宿主每帧写 DOM。
 *
 * 【作用】
 * 参数编辑器过去依赖原生滚动条（`overflow-x-scroll overflow-y-scroll`）。内核自绘
 * 滚动后浏览器不再提供滚动条，必须自己画。几何计算本身**复用时间轴内核**的
 * `computeScrollbar`——两条轴的数学完全相同，复制一份会在边界处理（内容不足一屏、
 * 除零）上分叉。
 *
 * 【竖向的特殊性】
 * 竖向滚动的「内容」不是像素内容而是**值域范围**（见 `verticalValueScroll`）：
 * 调用方传入的 `maxScrollTopPx` 即 1600px 值域范围，`verticalContentSizePx`
 * 应传「值域范围 + 视口高度」，这样 thumb 比例与拖拽行程与旧实现一致。
 * 缺省时按该式自动推出，避免调用方漏传时静默得到错误的 thumb 长度
 * （显式传入与缺省推出等价，见同目录单测）。
 *
 * 【与其他模块的关系】
 * - 上游：`pianoRollKernelHost` 每帧调用。
 * - 复用：`renderKernel/scrollbars`（几何与命中的单一来源）。
 * - 独立性：纯函数，不依赖 DOM / React。
 */

import { computeScrollbar, type ScrollbarGeometry } from "../../../renderKernel/scrollbars";

/** 装配入参。 */
export interface PianoRollScrollbarArgs {
    readonly viewportWidthPx: number;
    readonly viewportHeightPx: number;
    readonly scrollLeftPx: number;
    readonly scrollTopPx: number;
    /** 水平滚动上限（= 原生 scrollWidth − 视口宽，含同步偏移）。 */
    readonly maxScrollLeftPx: number;
    /** 竖向滚动上限（= 值域滚动范围，通常 1600）。 */
    readonly maxScrollTopPx: number;
    /**
     * 水平「内容尺寸」；缺省由 `maxScrollLeftPx + 视口宽` 推出（= 原生 scrollWidth）。
     *
     * 【为什么缺省是「上限 + 视口」而不是「内容宽」】原生的 thumb 长度是
     * `视口² / scrollWidth`，而 `scrollWidth = 内容宽 + 视口宽`（`overflow: scroll`
     * 的既有约定）。若把「内容宽」当成内容尺寸，thumb 会偏长——macOS 实测：
     * 水平 thumb 应为 `1864² / 10989 = 316.18`，用内容宽算得 `1864² / 8925 = 380.77`。
     * 竖向同理（`823² / 2423 = 279.54`，与实测一致），因此两轴统一用本条规则。
     */
    readonly horizontalContentSizePx?: number;
    /** 竖向「内容尺寸」= 值域范围 + 视口高度；缺省由 max + 视口高度推出。 */
    readonly verticalContentSizePx?: number;
}

/** 两条轴的几何。 */
export interface PianoRollScrollbarGeometries {
    readonly horizontal: ScrollbarGeometry;
    readonly vertical: ScrollbarGeometry;
}

/**
 * 解析两条滚动条的几何。
 *
 * 流程：把两轴各自的「内容尺寸 / 视口尺寸 / 当前位置 / 上限」分别交给 `computeScrollbar`。
 * 内容尺寸缺省按 `上限 + 视口尺寸` 推出（= 原生 `scrollWidth`，见入参说明）。
 *
 * 特殊说明：位置应为**绘制坐标**；`maxScrollLeftPx` 含同步偏移（原生域口径），
 * 两者口径不同是有意的——thumb 的比例取决于**原生** scrollWidth，而 thumb 的位置
 * 落在**绘制**区上。详见 `pianoRollKernelHost` 的坐标说明。
 *
 * @param args 见 `PianoRollScrollbarArgs`。
 * @returns 水平与竖直滚动条几何。
 */
export function resolvePianoRollScrollbarGeometries(
    args: PianoRollScrollbarArgs,
): PianoRollScrollbarGeometries {
    return {
        horizontal: computeScrollbar({
            contentSizePx:
                args.horizontalContentSizePx ?? args.maxScrollLeftPx + args.viewportWidthPx,
            viewportSizePx: args.viewportWidthPx,
            scrollPx: args.scrollLeftPx,
            maxScrollPx: args.maxScrollLeftPx,
        }),
        vertical: computeScrollbar({
            contentSizePx:
                args.verticalContentSizePx ?? args.maxScrollTopPx + args.viewportHeightPx,
            viewportSizePx: args.viewportHeightPx,
            scrollPx: args.scrollTopPx,
            maxScrollPx: args.maxScrollTopPx,
        }),
    };
}
