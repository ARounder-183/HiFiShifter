/**
 * 参数编辑器内核 · 竖向「值域 ↔ 像素滚动」适配。
 *
 * 【主要内容】
 * 把内核 `ScrollKernel` 的像素 `scrollTop` 与参数编辑器的**值域**视口
 * （`center` / `span`）互相转换，并给出两侧共用的滚动范围常量。
 *
 * 【作用】
 * 参数编辑器的竖向滚动语义与时间轴**根本不同**：时间轴滚的是像素行，
 * 而参数编辑器滚的是参数值域——`center` 是视口中心的值、`span` 是可见值跨度。
 * 内核的 `ScrollKernel` 只认像素，因此两者之间必须有一层适配。
 *
 * 【为什么不重写映射】
 * 真实换算仍是 `components/layout/pianoRoll/verticalScrollMapping.ts` 里的
 * `verticalScrollTopFromCenter` / `centerFromVerticalScrollTop`（既有实现、
 * 已被拖拽与滚轮路径调参验证过）。本模块只是把它接到内核像素域上：
 * 另写一套比值公式会让滚轮步进与拖拽比例的手感在两个渲染模式下分叉。
 *
 * 【与其他模块的关系】
 * - 上游：`pianoRollKernelHost` 在滚动条拖拽 / 滚轮 / 视口命令时调用。
 * - 下游：`verticalScrollMapping` 提供真正的映射算术。
 * - 独立性：纯函数，不依赖 DOM / React，可直接单测。
 */

import {
    centerFromVerticalScrollTop,
    verticalScrollTopFromCenter,
} from "../../verticalScrollMapping";

/**
 * 竖向滚动条的可滚动像素范围。
 *
 * 必须与旧实现一致：面板过去用一个 1600px 的 spacer div 撑出滚动范围
 * （`PianoRollPanel` 的 `PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX`）。
 * 改这个值会同时改变滚轮步进与拖拽比例，属手感变更，不在本次迁移范围。
 */
export const PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX = 1600;

/** 值域边界与视口跨度的公共入参。 */
export interface ValueScrollMappingArgs {
    /** 该参数值域下界。 */
    readonly min: number;
    /** 该参数值域上界。 */
    readonly max: number;
    /** 视口可见的值跨度。 */
    readonly span: number;
}

/**
 * 值域中心 → 内核像素滚动位置。
 *
 * 特殊说明：`center` 会被既有映射**钳制到可动中心域** `[min + span/2, max - span/2]`。
 * 这是刻意的有损行为——当 `span` 不足以覆盖整个值域时，视口中心不可能停在该域之外。
 * 因此本函数与 `centerFromKernelScrollTop` 只在**可动中心域内**互为逆运算
 * （见同目录单测的「往返可逆」用例）。
 *
 * @param args 值域边界、跨度，以及视口中心值。
 * @returns 内核 `scrollTop`（CSS px，已钳制到 [0, 1600]）。
 */
export function kernelScrollTopFromCenter(
    args: ValueScrollMappingArgs & { readonly center: number },
): number {
    return verticalScrollTopFromCenter({
        min: args.min,
        max: args.max,
        span: args.span,
        center: args.center,
        scrollRangePx: PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX,
    });
}

/**
 * 内核像素滚动位置 → 值域中心。
 *
 * 特殊说明：返回值恒落在可动中心域内；与 `kernelScrollTopFromCenter` 构成
 * 「域内无损、域外钳制」的一对映射。
 *
 * @param args 值域边界、跨度，以及内核 `scrollTop`。
 * @returns 视口中心值（已钳制到值域内合法区间）。
 */
export function centerFromKernelScrollTop(
    args: ValueScrollMappingArgs & { readonly scrollTop: number },
): number {
    return centerFromVerticalScrollTop({
        min: args.min,
        max: args.max,
        span: args.span,
        scrollTop: args.scrollTop,
        scrollRangePx: PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX,
    });
}
