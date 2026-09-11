/**
 * crossfadeGrip.ts — 交叉淡化「交点抓手」的几何计算。
 *
 * 【主要内容】
 * 求前一个 clip 的淡出包络线与后一个 clip 的淡入包络线在重叠区内的**真实曲线
 * 交点**（不是两条端点连线的交点）。
 *
 * 【作用】
 * 交叉点抓手（拖动它 = 同时移动前 clip 的右缘与后 clip 的左缘，保持重叠长度
 * 不变）必须落在用户**看到的**交叉处。画布上用 `fadeCurveGain` 绘制的是曲线
 * （sine / exponential / scurve 等），直接用两端点连线求交点会在 Y 轴明显偏离。
 * 这里用二分法精确求解。
 *
 * 【与其他模块的关系】
 * - 消费方：旧实现的 `OverlapEditLayer`（DOM 命中层）与内核的重叠区解析
 *   （`kernel/interaction/overlapControls`）。抽成共享模块是为了让两者用**同一份
 *   几何**——两份实现迟早漂移，而漂移的表现是「抓手画在这里、却在那里才抓得住」。
 * - 依赖：`reaperFade.fadeGainSigned`（与绘制端同一套曲线函数）。
 * - 独立性：纯函数，无 DOM / React 依赖。
 */

import { fadeGainSigned } from "./reaperFade";

/** 交点计算参数（坐标均为时间轴内容坐标，px）。 */
export interface CrossfadeGripArgs {
    /** 前一个 clip 的右边缘 X。 */
    readonly earlierEndPx: number;
    /** 前一个 clip 淡出包络的像素宽度。 */
    readonly earlierFadePx: number;
    readonly earlierShape: number;
    readonly earlierDir: number;
    /** 后一个 clip 的左边缘 X。 */
    readonly laterStartPx: number;
    /** 后一个 clip 淡入包络的像素宽度。 */
    readonly laterFadePx: number;
    readonly laterShape: number;
    readonly laterDir: number;
    readonly bodyTop: number;
    readonly bodyHeight: number;
}

/**
 * 计算两条真实包络曲线在重叠区内的交点。
 *
 * 流程：先取两条曲线 X 区间的交集（曲线只在各自的淡化区内定义）→ 用二分法求
 * `yA - yB` 的零点（A 淡出随 x 增大而下移、B 淡入随 x 增大而上移，因此差函数
 * 在该区间内严格单调）。
 *
 * @param args 见 `CrossfadeGripArgs`。
 * @returns 交点坐标；两侧都无淡化、区间为空、或两曲线在该区间内不相交时为 null。
 */
export function computeCrossfadeGripPoint(
    args: CrossfadeGripArgs,
): { x: number; y: number } | null {
    const {
        earlierEndPx,
        earlierFadePx,
        earlierShape,
        earlierDir,
        laterStartPx,
        laterFadePx,
        laterShape,
        laterDir,
        bodyTop,
        bodyHeight,
    } = args;
    const earlierLeftPx = earlierEndPx - earlierFadePx;
    const laterRightPx = laterStartPx + laterFadePx;
    const lo = Math.max(earlierLeftPx, laterStartPx);
    const hi = Math.min(earlierEndPx, laterRightPx);
    if (hi - lo <= 0.01 || earlierFadePx <= 0 || laterFadePx <= 0) return null;

    // 两条曲线在重叠淡化区的 X 区间单调：A 淡出 y 随 x 增大而增大，
    // B 淡入 y 随 x 增大而减小，因此 yA-yB 严格单调 → 二分求零点。
    const yDiff = (x: number): number => {
        const tA = (x - earlierLeftPx) / earlierFadePx;
        const gainA = fadeGainSigned(earlierShape, earlierDir, "out", tA);
        const yA = bodyTop + bodyHeight * (1 - gainA);
        const tB = (x - laterStartPx) / laterFadePx;
        const gainB = fadeGainSigned(laterShape, laterDir, "in", tB);
        const yB = bodyTop + bodyHeight * (1 - gainB);
        return yA - yB;
    };

    let low = lo;
    let high = hi;
    const fLow = yDiff(low);
    const fHigh = yDiff(high);

    // 不相交：视觉交叉点在两个淡化区之外，不显示手柄。
    if (fLow * fHigh > 0) {
        return null;
    }

    for (let i = 0; i < 40; i += 1) {
        const mid = (low + high) / 2;
        const fMid = yDiff(mid);
        if (Math.abs(fMid) < 1e-3) {
            low = high = mid;
            break;
        }
        if (fLow * fMid < 0) {
            high = mid;
        } else {
            low = mid;
        }
    }

    const x = (low + high) / 2;
    const tA = (x - earlierLeftPx) / earlierFadePx;
    const gainA = fadeGainSigned(earlierShape, earlierDir, "out", tA);
    return {
        x,
        y: bodyTop + bodyHeight * (1 - gainA),
    };
}
