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
 * - 复用：`timeline/kernel/input/scrollbars`（几何与命中的单一来源）。
 * - 独立性：纯函数，不依赖 DOM / React。
 */

import {
    computeScrollbar,
    type ScrollbarGeometry,
} from "../../../timeline/kernel/input/scrollbars";

/** 装配入参。 */
export interface PianoRollScrollbarArgs {
    readonly viewportWidthPx: number;
    readonly viewportHeightPx: number;
    /** 时间轴内容宽度（工程秒 × pxPerSec）。 */
    readonly contentWidthPx: number;
    readonly scrollLeftPx: number;
    readonly scrollTopPx: number;
    /** 水平滚动上限（= 内容宽度，与时间轴语义一致）。 */
    readonly maxScrollLeftPx: number;
    /** 竖向滚动上限（= 值域滚动范围，通常 1600）。 */
    readonly maxScrollTopPx: number;
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
 * 竖向内容尺寸缺省时按 `maxScrollTopPx + viewportHeightPx` 推出。
 *
 * @param args 见 `PianoRollScrollbarArgs`。
 * @returns 水平与竖直滚动条几何。
 */
export function resolvePianoRollScrollbarGeometries(
    args: PianoRollScrollbarArgs,
): PianoRollScrollbarGeometries {
    return {
        horizontal: computeScrollbar({
            contentSizePx: args.contentWidthPx,
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
