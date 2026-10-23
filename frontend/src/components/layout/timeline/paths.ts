/**
 * 淡变增益的两个方向性包装（REAPER 形状 / 曲率模型）。
 *
 * 【为什么只剩这两个】`fadeCurveGain`（旧命名曲线 → 新模型的兼容垫片）、
 * `FadeCurveTypeLegacy`（旧枚举别名）、`fadeInAreaPath` / `fadeOutAreaPath`
 * （旧面积路径）在迁移完成后已经没有任何调用点，2026-10 一并删除。
 * 新代码用 `reaperFade.ts` 的形状 id + 曲率。
 */
export type FadeCurveType = "linear" | "sine" | "exponential" | "logarithmic" | "scurve";

import { fadeGainSigned } from "./reaperFade";

/** 淡入方向：将 t ∈ [0,1] 映射为增益 ∈ [0,1]（REAPER 形状/曲率模型）。 */
export function fadeGainIn(shape: number, dir: number, t: number): number {
    return fadeGainSigned(shape, dir, "in", t);
}

/** 淡出方向：与画布一致，t 为淡出区间内的进度，增益随进度下降。 */
export function fadeGainOut(shape: number, dir: number, t: number): number {
    return fadeGainSigned(shape, dir, "out", t);
}
