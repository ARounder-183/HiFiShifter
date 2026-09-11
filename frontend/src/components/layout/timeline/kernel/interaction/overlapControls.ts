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
 * 淡变命中区复用 `fadeTargets`（内部走旧实现的 `buildFadeHitTargets`，与绘制端
 * 逐像素一致）；clip 边缘宽度与旧实现的 `clipEdgeWidthPx` 一致（10px）。
 */

import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "../../constants";
import { computeCrossfadeGripPoint } from "../../crossfadeGrip";
import { effectiveFadeSec, hitClipFadeTarget, type FadeTargetClip } from "./fadeTargets";

/** 重叠区内可命中的控件类型。 */
export type OverlapControlKind =
    /** clip 边缘（归属 `clipId` 的那个 clip）。 */
    | "clip-left-edge"
    | "clip-right-edge"
    /** 淡变控件；具体是淡入还是淡出看 `fadeSide`。 */
    | "fade"
    /** 交叉淡化交点抓手：拖动它同时移动双方边缘。 */
    | "crossfade-grip";

/** 重叠区控件命中结果。 */
export interface OverlapControlHit {
    readonly kind: OverlapControlKind;
    /** 控件所属的 clip（**可能是前一个 clip**——这正是本模块存在的意义）。 */
    readonly clipId: string;
    /** 淡变控件所属的一侧；clip 边缘命中时缺省。 */
    readonly fadeSide?: "in" | "out";
    /**
     * 交叉点抓手的另一侧 clip（`clipId` 为**后一个** clip，这里为前一个）。
     * 仅 `kind === "crossfade-grip"` 时有值。
     */
    readonly partnerClipId?: string;
}

/**
 * 重叠解析所需的 clip 字段集。
 *
 * 几何字段必填，淡变参数沿用 `FadeTargetClip` 的可选语义（未设过淡变即为 0）。
 */
export interface OverlapClip extends FadeTargetClip {
    readonly id: string;
    readonly startSec: number;
    readonly lengthSec: number;
}

/** 命中参数。 */
export interface OverlapControlArgs {
    /** 该轨道内的 clip（顺序无关，内部自行配对）。 */
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

/** 交叉点抓手的命中方框边长（与旧实现的 `gripSize = 16` 同源）。 */
const GRIP_HIT_SIZE_PX = 16;

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
            const laterSide = hitClipFadeTarget({
                clip: later,
                clipLeftPx: laterStartPx,
                clipWidthPx: laterEndPx - laterStartPx,
                contentX,
                localY,
                pxPerSec,
                rowHeight,
                clipXFrom: overlapStartPx,
                clipXTo: overlapEndPx,
            });
            if (laterSide !== null) {
                result = { kind: "fade", clipId: later.id, fadeSide: laterSide };
            }

            // 2) 前一个 clip 的淡出（只取重叠区内部分）。
            const earlierSide = hitClipFadeTarget({
                clip: earlier,
                clipLeftPx: earlierStartPx,
                clipWidthPx: earlierEndPx - earlierStartPx,
                contentX,
                localY,
                pxPerSec,
                rowHeight,
                clipXFrom: overlapStartPx,
                clipXTo: overlapEndPx,
            });
            if (earlierSide !== null) {
                result = { kind: "fade", clipId: earlier.id, fadeSide: earlierSide };
            }

            // 3) clip 边缘（覆盖上面的淡变判定）：整行高、以边界为中心。
            if (Math.abs(contentX - laterStartPx) <= edgeWidthPx / 2) {
                result = { kind: "clip-left-edge", clipId: later.id };
            }
            if (Math.abs(contentX - earlierEndPx) <= edgeWidthPx / 2) {
                result = { kind: "clip-right-edge", clipId: earlier.id };
            }

            // 4) 交叉点抓手（**最高优先级**）：旧实现给它显式 `zIndex: 400`，注释
            //    写明「交叉点手柄应高于所有淡入淡出/边缘控件」。抓手的圆心就是两条
            //    真实包络曲线的交点，因此它天然落在两条包络线的命中块之上——不特殊
            //    提权的话，用户想抓抓手却总是抓到某一条曲线。
            // 前一个 clip 参与交叉的是**淡出**，后一个参与的是**淡入**。
            const earlierFadePx =
                effectiveFadeSec(earlier.fadeOutSec, earlier.autoFadeOutSec) * pxPerSec;
            const laterFadePx = effectiveFadeSec(later.fadeInSec, later.autoFadeInSec) * pxPerSec;
            const grip = computeCrossfadeGripPoint({
                earlierEndPx,
                earlierFadePx,
                earlierShape: earlier.fadeOutShape ?? 0,
                earlierDir: earlier.fadeOutDir ?? 0,
                laterStartPx,
                laterFadePx,
                laterShape: later.fadeInShape ?? 0,
                laterDir: later.fadeInDir ?? 0,
                bodyTop: CLIP_HEADER_HEIGHT,
                bodyHeight: Math.max(1, rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT),
            });
            if (grip !== null) {
                const half = GRIP_HIT_SIZE_PX / 2;
                if (Math.abs(contentX - grip.x) <= half && Math.abs(localY - grip.y) <= half) {
                    result = {
                        kind: "crossfade-grip",
                        clipId: later.id,
                        partnerClipId: earlier.id,
                    };
                }
            }
        }
    }

    return result;
}
