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
    /** 水平滚动上限（= 原生 `scrollWidth` − 视口宽 = 内容宽；**不含**同步偏移）。 */
    readonly maxScrollLeftPx: number;
    /** 竖向滚动上限（= 值域滚动范围，通常 1600）。 */
    readonly maxScrollTopPx: number;
    /**
     * 水平「内容尺寸」；缺省由 `maxScrollLeftPx + 视口宽` 推出（= 原生 scrollWidth）。
     *
     * 【为什么缺省是「上限 + 视口」而不是「内容宽」】原生的 thumb 长度是
     * `视口² / scrollWidth`，而 `scrollWidth = 内容宽 + 视口宽`（`overflow: scroll`
     * 的既有约定）。若把「内容宽」当成内容尺寸，thumb 会偏长——macOS 实测（旧实现的
     * 原生滚动条，当时同步偏移还被算进 scrollWidth）：水平 thumb 应为
     * `1864² / 10989 = 316.18`，用内容宽算得 `1864² / 8925 = 380.77`。竖向同理
     * （`823² / 2423 = 279.54`，与实测一致），因此两轴统一用本条规则。
     */
    readonly horizontalContentSizePx?: number;
    /** 竖向「内容尺寸」= 值域范围 + 视口高度；缺省由 max + 视口高度推出。 */
    readonly verticalContentSizePx?: number;
    /**
     * 竖向值域（min / max / span）。提供时 thumb 长度改为**值域占比**口径：
     * `thumb = 轨道 × span / (max − min)`，与原生滚动条"thumb 占比 = 视口 /
     * 内容"一致 —— 竖向缩放（span 变化）时 thumb 长度随之伸缩；span 覆盖
     * 整个值域（缩放到最小）时不可滚动，宿主把它隐藏。缺省走「上限 + 视口」
     * 的旧口径（thumb 长度恒定、永不隐藏）。
     */
    readonly verticalValueDomain?: {
        readonly min: number;
        readonly max: number;
        readonly span: number;
    };
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
 * 竖向值域口径（传入 `verticalValueDomain` 时）：thumb 长度 = `轨道 × span / range`，
 * 起点仍按 `scrollTop / maxScrollTop`（可动中心的进度，0..1600 的既有映射不变）
 * 在「轨道 − thumb」的行程内线性映射 —— 视觉上与原生滚动条同构（可见占比决定
 * thumb 长度），而滚动手感（滚轮步进 / 拖拽比例、`scrollTop ↔ center` 映射）
 * 分毫未动。实现上把「内容尺寸」表达为 `轨道 × range / span`：`computeScrollbar`
 * 的 `ratio = 轨道 / 内容` 恰好约简为 `span / range`；`range ≤ span`（缩放到最小）
 * 时内容不大于视口 → 不可滚动 → 宿主隐藏 thumb。
 *
 * 特殊说明：位置与上限同为**原生坐标**（`computeScrollbar` 的 thumb 起点 = 位置 / 上限，
 * 两者必须同域）。位置若减去同步偏移，thumb 在同步模式下就永远走不到轨道末端——
 * 拖拽用的是增量（两域相减抵消）所以"能拖到"，只有画出来的起点是错的。
 * 详见 `pianoRollKernelHost` 的坐标说明。
 *
 * @param args 见 `PianoRollScrollbarArgs`。
 * @returns 水平与竖直滚动条几何。
 */
export function resolvePianoRollScrollbarGeometries(
    args: PianoRollScrollbarArgs,
): PianoRollScrollbarGeometries {
    const domain = args.verticalValueDomain;
    const range = domain ? domain.max - domain.min : 0;
    const valueProportional =
        domain != null &&
        Number.isFinite(range) &&
        range > 0 &&
        Number.isFinite(domain.span) &&
        domain.span > 0 &&
        Number.isFinite(args.viewportHeightPx) &&
        args.viewportHeightPx > 0;
    return {
        horizontal: computeScrollbar({
            contentSizePx:
                args.horizontalContentSizePx ?? args.maxScrollLeftPx + args.viewportWidthPx,
            viewportSizePx: args.viewportWidthPx,
            scrollPx: args.scrollLeftPx,
            maxScrollPx: args.maxScrollLeftPx,
        }),
        vertical: computeScrollbar(
            valueProportional && domain
                ? {
                      // 内容尺寸按值域占比反推：轨道 / 内容 = span / range。
                      // 见上方「竖向值域口径」的推导。
                      contentSizePx: (args.viewportHeightPx * range) / domain.span,
                      viewportSizePx: args.viewportHeightPx,
                      scrollPx: args.scrollTopPx,
                      maxScrollPx: args.maxScrollTopPx,
                  }
                : {
                      contentSizePx:
                          args.verticalContentSizePx ?? args.maxScrollTopPx + args.viewportHeightPx,
                      viewportSizePx: args.viewportHeightPx,
                      scrollPx: args.scrollTopPx,
                      maxScrollPx: args.maxScrollTopPx,
                  },
        ),
    };
}
