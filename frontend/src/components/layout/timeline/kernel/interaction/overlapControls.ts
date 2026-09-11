/**
 * 时间轴渲染内核 · 重叠区控件解析
 *
 * 【为什么需要】
 * 内核的命中测试用二分查找定位 clip，取「最后一个 `startSec <= sec`」的那个——
 * 在重叠区里这**永远是后一个** clip。于是：
 * - 后一个 clip 的左边缘**可达**（它正好落在重叠区左缘）✓
 * - **前一个 clip 的右边缘与淡出控件完全不可达** ✗（点下去命中的是后一个 clip
 *   的 body，用户根本抓不到前一个 clip 的右缘）
 *
 * 旧实现用一个独立的 DOM 层（`OverlapEditLayer`，`z-[200]`）解决这个问题：它在
 * 重叠区**按位置**提供双方的控件——
 * - 重叠区左缘 → 后一个 clip 的左边缘 + 其淡入控件；
 * - 重叠区右缘 → 前一个 clip 的右边缘 + 其淡出控件。
 *
 * 本模块把这条「按位置解析」的规则搬进纯函数，供内核在通用命中之后改写结果。
 *
 * 【判定优先级】
 * 与旧实现的 DOM 层叠顺序一致（它**后 push 的在上**，因此这里按同样的顺序
 * 覆盖式赋值，最后命中的获胜）：
 * 1. clip 边缘（10px 带、**整行高**）最优先——淡变包络线的端点常与相邻 clip 的
 *    边缘重合，此时必须判为边缘，否则用户想裁短却改成了淡化长度；
 * 2. 淡变包络线 / 区域边缘竖线其次。
 *
 * 【几何来源】
 * 淡变命中区直接复用旧实现的 `buildFadeHitTargets`（它保证与绘制端
 * `drawFadeCurveStroke` 逐像素一致）；clip 边缘宽度与旧实现的 `clipEdgeWidthPx`
 * 一致（10px）。
 */

import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "../../constants";
import { buildFadeHitTargets } from "../../fadeHitTargets";

/** 重叠区内可命中的控件类型。 */
export type OverlapControlKind = "clip-left-edge" | "clip-right-edge" | "fade-line" | "fade-edge";

/** 重叠区控件命中结果。 */
export interface OverlapControlHit {
    readonly kind: OverlapControlKind;
    /** 控件所属的 clip（**可能是前一个 clip**——这正是本模块存在的意义）。 */
    readonly clipId: string;
    /** 淡变控件所属的一侧；clip 边缘命中时缺省。 */
    readonly fadeSide?: "in" | "out";
}

/**
 * 重叠解析所需的 clip 字段集（比 `HitTestClip` 多出淡变参数）。
 *
 * 全部字段都参与几何计算，缺一不可——淡变长度决定包络线的位置，形状与方向
 * 决定曲线的走向（曲线命中块是沿真实曲线采样的，不是直线）。
 */
export interface OverlapClip {
    readonly id: string;
    readonly startSec: number;
    readonly lengthSec: number;
    /**
     * 淡变参数（全部可选）。
     *
     * 刻意可选：数据模型里这些字段本来就允许缺省（未设过淡变的 clip），写成必填
     * 只会迫使每个调用方补零，反而掩盖"这个 clip 到底有没有淡变"。
     * 缺省一律按 0 处理（长度 0 = 无淡变，形状 0 / 方向 0 = 线性）。
     */
    readonly fadeInSec?: number;
    /** 自动交叉淡化长度（秒）：> 0 时覆盖手动值（与绘制端一致）。 */
    readonly autoFadeInSec?: number;
    readonly fadeInShape?: number;
    readonly fadeInDir?: number;
    readonly fadeOutSec?: number;
    readonly autoFadeOutSec?: number;
    readonly fadeOutShape?: number;
    readonly fadeOutDir?: number;
}

/** 命中参数。 */
export interface OverlapControlArgs {
    /** 该轨道内的 clip（顺序无关，内部按 startSec 自行配对）。 */
    readonly clips: readonly OverlapClip[];
    /** 指针的内容坐标 x（CSS px）。 */
    readonly contentX: number;
    /** 指针在该行内的局部 y（0 = 行顶）。 */
    readonly localY: number;
    readonly pxPerSec: number;
    readonly rowHeight: number;
    /** clip 边缘命中带宽度（CSS px）。缺省 10，与旧实现的 `clipEdgeWidthPx` 一致。 */
    readonly edgeWidthPx?: number;
}

/** clip 边缘命中带宽度（与旧实现 `OverlapEditLayer` 的 `clipEdgeWidthPx` 同源）。 */
const DEFAULT_EDGE_WIDTH_PX = 10;

/**
 * 淡变有效长度：自动交叉淡化（> 0 时）覆盖手动淡变（与绘制端 / 旧实现一致）。
 *
 * @param manual 手动淡变长度（秒）。
 * @param auto 自动交叉淡化长度（秒）。
 * @returns 生效的淡变长度（秒）。
 */
function effectiveFadeSec(manual: number | undefined, auto: number | undefined): number {
    const autoSec = typeof auto === "number" && Number.isFinite(auto) && auto > 0 ? auto : 0;
    if (autoSec > 0) return autoSec;
    return typeof manual === "number" && Number.isFinite(manual) && manual > 0 ? manual : 0;
}

/**
 * 判定点是否落在该 clip 的淡变命中区内。
 *
 * @param args clip 几何、淡变参数、重叠区范围与指针坐标。
 * @returns 命中时为 `"line"`（包络线）或 `"edge"`（区域边缘竖线）；未命中为 null。
 */
function hitFadeTargets(args: {
    readonly clip: OverlapClip;
    readonly side: "in" | "out";
    readonly clipLeftPx: number;
    readonly clipWidthPx: number;
    readonly fadePx: number;
    readonly bodyTop: number;
    readonly bodyHeight: number;
    readonly overlapStartPx: number;
    readonly overlapEndPx: number;
    readonly contentX: number;
    readonly localY: number;
}): "line" | "edge" | null {
    const isIn = args.side === "in";
    const targets = buildFadeHitTargets({
        clipLeftPx: args.clipLeftPx,
        clipWidthPx: args.clipWidthPx,
        bodyTop: args.bodyTop,
        bodyHeight: args.bodyHeight,
        fadeInPx: isIn ? args.fadePx : 0,
        fadeOutPx: isIn ? 0 : args.fadePx,
        fadeInShape: isIn ? (args.clip.fadeInShape ?? 0) : 0,
        fadeInDir: isIn ? (args.clip.fadeInDir ?? 0) : 0,
        fadeOutShape: isIn ? 0 : (args.clip.fadeOutShape ?? 0),
        fadeOutDir: isIn ? 0 : (args.clip.fadeOutDir ?? 0),
        // 只保留重叠区内的部分：重叠区外由 clip 自身的命中分区负责（那里不会
        // 出现"同一段空间属于两个 clip"的歧义）。
        clipXFrom: args.overlapStartPx,
        clipXTo: args.overlapEndPx,
    });
    for (const target of targets) {
        if (
            args.contentX >= target.left &&
            args.contentX <= target.left + target.width &&
            args.localY >= target.top &&
            args.localY <= target.top + target.height
        ) {
            return target.kind;
        }
    }
    return null;
}

/**
 * 解析重叠区内的控件命中。
 *
 * 流程：筛出覆盖该点的 clip → 两两配对（早 / 晚）→ 按旧实现的层叠顺序依次判定
 * （淡入 → 淡出 → 左缘 → 右缘，后者覆盖前者）→ 返回最后一次命中。
 *
 * 特殊说明：只考虑**覆盖该点**的 clip。重叠区的任何控件都要求其所属 clip 覆盖
 * 该点（边缘带贴着 clip 边界，淡变控件在 clip 内部），因此这个筛选不会漏判，
 * 却把复杂度从「轨道内所有 clip 两两配对」降到「覆盖该点的少数几个」。
 *
 * @param args 命中参数。
 * @returns 命中的控件；不在重叠区或未命中控件时为 null。
 */
export function hitOverlapControl(args: OverlapControlArgs): OverlapControlHit | null {
    const pxPerSec = Number.isFinite(args.pxPerSec) ? Math.max(1e-9, args.pxPerSec) : 1e-9;
    const rowHeight = Number.isFinite(args.rowHeight) ? Math.max(1, args.rowHeight) : 1;
    const edgeWidthPx = Number.isFinite(args.edgeWidthPx)
        ? Math.max(0, args.edgeWidthPx as number)
        : DEFAULT_EDGE_WIDTH_PX;
    const contentX = args.contentX;
    const localY = args.localY;

    const containing = args.clips.filter((item) => {
        const startPx = item.startSec * pxPerSec;
        const endPx = (item.startSec + item.lengthSec) * pxPerSec;
        return startPx <= contentX && contentX <= endPx;
    });
    // 少于两个 clip 覆盖该点 → 不存在重叠区。
    if (containing.length < 2) return null;

    const bodyTop = CLIP_HEADER_HEIGHT;
    const bodyHeight = Math.max(1, rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT);

    let result: OverlapControlHit | null = null;

    for (let i = 0; i < containing.length; i += 1) {
        for (let j = i + 1; j < containing.length; j += 1) {
            let earlier = containing[i];
            let later = containing[j];
            // 时间上「前一个」= 起点更早；起点相同时按 id 稳定排序（与旧实现一致）。
            if (
                later.startSec < earlier.startSec ||
                (later.startSec === earlier.startSec && later.id < earlier.id)
            ) {
                const swap = earlier;
                earlier = later;
                later = swap;
            }

            const earlierStartPx = earlier.startSec * pxPerSec;
            const earlierEndPx = (earlier.startSec + earlier.lengthSec) * pxPerSec;
            const laterStartPx = later.startSec * pxPerSec;
            const laterEndPx = (later.startSec + later.lengthSec) * pxPerSec;
            const overlapStartPx = Math.max(earlierStartPx, laterStartPx);
            const overlapEndPx = Math.min(earlierEndPx, laterEndPx);
            if (overlapEndPx - overlapStartPx <= 0.5) continue;

            // 1) 后一个 clip 的淡入（只取重叠区内部分）。
            const laterFadeInSec = effectiveFadeSec(later.fadeInSec, later.autoFadeInSec);
            if (laterFadeInSec > 0) {
                const kind = hitFadeTargets({
                    clip: later,
                    side: "in",
                    clipLeftPx: laterStartPx,
                    clipWidthPx: laterEndPx - laterStartPx,
                    fadePx: laterFadeInSec * pxPerSec,
                    bodyTop,
                    bodyHeight,
                    overlapStartPx,
                    overlapEndPx,
                    contentX,
                    localY,
                });
                if (kind !== null) {
                    result = {
                        kind: kind === "line" ? "fade-line" : "fade-edge",
                        clipId: later.id,
                        fadeSide: "in",
                    };
                }
            }

            // 2) 前一个 clip 的淡出（只取重叠区内部分）。
            const earlierFadeOutSec = effectiveFadeSec(earlier.fadeOutSec, earlier.autoFadeOutSec);
            if (earlierFadeOutSec > 0) {
                const kind = hitFadeTargets({
                    clip: earlier,
                    side: "out",
                    clipLeftPx: earlierStartPx,
                    clipWidthPx: earlierEndPx - earlierStartPx,
                    fadePx: earlierFadeOutSec * pxPerSec,
                    bodyTop,
                    bodyHeight,
                    overlapStartPx,
                    overlapEndPx,
                    contentX,
                    localY,
                });
                if (kind !== null) {
                    result = {
                        kind: kind === "line" ? "fade-line" : "fade-edge",
                        clipId: earlier.id,
                        fadeSide: "out",
                    };
                }
            }

            // 3) clip 边缘（最优先，覆盖上面的淡变判定）：整行高、以边界为中心。
            if (Math.abs(contentX - laterStartPx) <= edgeWidthPx / 2) {
                result = { kind: "clip-left-edge", clipId: later.id };
            }
            if (Math.abs(contentX - earlierEndPx) <= edgeWidthPx / 2) {
                result = { kind: "clip-right-edge", clipId: earlier.id };
            }
        }
    }

    return result;
}
