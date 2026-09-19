/**
 * 参数编辑器内核 · 选区块实例构建（纯函数）
 *
 * 【主要内容】
 * 把选区的时间段（秒）投影成**视口坐标**下的矩形实例：每个时间段产出一个整高的
 * 半透明填充矩形 + 四条 1 CSS px 的边框。
 *
 * 【为什么选区块必须画在这个（GL 场景）层】
 * 层序契约是「选区在曲线**之下**」——`render.ts` 里那段注释写得很明确。但曲线早已
 * 迁上 GL，而选区块还留在 Canvas2D 主画布上；主画布的 DOM 顺序在 GL 场景画布**之后**，
 * 浏览器按"后画的在上"合成，层序因此被**反转**：8% 的蓝色填充盖到了曲线上，
 * 未被选中的参数线也在选区内泛蓝（用户报告"未被选择的参数线的下半部分染上了
 * 跟已选参数线一样的蓝色"）。把选区块搬进 GL 场景层、在 `drawGlCurves` **之前**
 * 发射，就恢复了迁移前的层序。
 *
 * 【为什么与网格共用同一个 GL 上下文、但用独立 program】
 * 网格几何按内容签名缓存（滚动是零重建的），而选区块**每帧都变**（选区被拖拽、
 * 滚动改变它的视口 x）。若塞进网格的实例缓冲，"几何未变就 repaint"的优化会把
 * 它冻住；共用上下文但独立缓冲则各司其职，且不必再申请一个 WebGL context
 * （上下文数量有上限）。
 *
 * 【几何与 Canvas2D 旧实现的等价性】
 * 旧代码（`render.ts`，已随本迁移删除）写作：
 * ```
 * ctx.fillRect(x0, 0, x1 - x0, h);
 * ctx.strokeRect(x0 + 0.5, 0.5, max(0, x1 - x0 - 1), h - 1);
 * ```
 * `strokeRect` 的路径以 0.5 偏移 + 1px 线宽展开后，四条边恰好各自覆盖
 * `[x0, x0+1]`、`[x1-1, x1]`（横向）与 `[0, 1]`、`[h-1, h]`（纵向）。
 * 因此这里用四个 1 CSS px 的实心矩形逐条复刻，视觉逐像素一致。
 *
 * 【与其他模块的关系】
 * - 上游：`pianoRollKernelHost` 每帧读取数据镜像的选区块字段后调用。
 * - 依赖：`renderKernel/timelineAxis`（唯一横向投影来源）、`scene/gridInstances`
 *   的 `GridInstance` 形状（复用平面矩形实例，避免再定义一套等价类型）。
 * - 下游：`writeFlatInstance` → WebGL2。
 * - 独立性：纯函数，不依赖 DOM / WebGL / React，可在 node 环境单测。
 */

import { secToViewportPx, type TimelineAxis } from "../../../renderKernel/timelineAxis";
import type { Rgba } from "../../../renderKernel/instanceTypes";
import type { GridInstance } from "./gridInstances";

/** 选区块边框厚度（CSS px）。与迁移前的 `strokeRect` 线宽一致。 */
const BORDER_PX = 1;

/** 构建参数。 */
export interface SelectionBandArgs {
    /** 当前投影（唯一横向投影来源）。 */
    readonly axis: TimelineAxis;
    /** 视口高度（CSS px）——选区块是整高的。 */
    readonly viewportHeightPx: number;
    /** 选区时间段（秒）；起止颠倒时自动归一，零宽段跳过。 */
    readonly spansSec: readonly { readonly startSec: number; readonly endSec: number }[];
    /** 填充色。 */
    readonly fillRgba: Rgba;
    /** 边框色。 */
    readonly borderRgba: Rgba;
}

/**
 * 把选区时间段投影为矩形实例序列（填充在前、边框在后）。
 *
 * 流程：逐段归一化起止 → 投影为视口 x → 产出填充矩形 → 在可见区产出四条边框。
 *
 * 特殊说明 1：**完全落在视口外**的段整体跳过（它的填充与边框都不会被看到），
 * 避免为长工程的大选区产生无用几何。
 *
 * 特殊说明 2：段的宽度小于两倍边框时（极窄选区）仍然照发——边框会重叠成一条实心
 * 竖条，这与 Canvas2D 的 `strokeRect(…, max(0, w-1), …)` 观感一致（那里也是重叠）。
 *
 * @param args 构建参数。
 * @returns 矩形实例序列；无可见选区时为空数组。
 */
export function buildSelectionBandInstances(args: SelectionBandArgs): GridInstance[] {
    const { axis, viewportHeightPx, spansSec, fillRgba, borderRgba } = args;
    const height = Number.isFinite(viewportHeightPx) ? Math.max(0, viewportHeightPx) : 0;
    if (height <= 0) return [];

    const items: GridInstance[] = [];
    for (const span of spansSec) {
        const startSec = Math.min(span.startSec, span.endSec);
        const endSec = Math.max(span.startSec, span.endSec);
        if (!Number.isFinite(startSec) || !Number.isFinite(endSec)) continue;

        const x0 = secToViewportPx(axis, startSec);
        const x1 = secToViewportPx(axis, endSec);
        const width = x1 - x0;
        if (!(width > 0)) continue;
        // 整段在视口右侧或左侧：不可见，跳过。
        if (x1 <= 0 || x0 >= axis.viewportWidthPx) continue;

        // 填充：整高。`value` 用负数占位——本层不是网格线，负值避免与"值为 0 的
        // 网格线"混淆（沿用 `GridInstance.value` 的负数约定）。
        items.push({
            x: x0,
            y: 0,
            w: width,
            h: height,
            rgba: fillRgba,
            value: -1,
        });

        // 边框：上 / 下（横向整宽）、左 / 右（纵向整高）。
        const borders: { x: number; y: number; w: number; h: number }[] = [
            { x: x0, y: 0, w: width, h: BORDER_PX },
            { x: x0, y: height - BORDER_PX, w: width, h: BORDER_PX },
            { x: x0, y: 0, w: BORDER_PX, h: height },
            { x: x1 - BORDER_PX, y: 0, w: BORDER_PX, h: height },
        ];
        for (const border of borders) {
            items.push({
                x: border.x,
                y: border.y,
                w: border.w,
                h: border.h,
                rgba: borderRgba,
                value: -1,
            });
        }
    }
    return items;
}
