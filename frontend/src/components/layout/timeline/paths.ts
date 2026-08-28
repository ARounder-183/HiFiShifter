export type FadeCurveType = "linear" | "sine" | "exponential" | "logarithmic" | "scurve";

import { FADE_CONVEX_SLIGHT, FADE_LATE_SLIGHT, FADE_S_SLIGHT, fadeGainSigned } from "./reaperFade";

/**
 * @deprecated 旧命名曲线枚举（v3/早期 v4 只读兼容）。新代码使用
 * REAPER 形状 id + 曲率（见 `reaperFade.ts`）。
 */
export type FadeCurveTypeLegacy = FadeCurveType;

/** 淡入方向：将 t ∈ [0,1] 映射为增益 ∈ [0,1]（REAPER 形状/曲率模型）。 */
export function fadeGainIn(shape: number, dir: number, t: number): number {
    return fadeGainSigned(shape, dir, "in", t);
}

/** 淡出方向：与画布一致，t 为淡出区间内的进度，增益随进度下降。 */
export function fadeGainOut(shape: number, dir: number, t: number): number {
    return fadeGainSigned(shape, dir, "out", t);
}

/**
 * @deprecated 兼容垫片：把旧命名曲线映射到新模型供面积路径等遗留调用点
 * 过渡。迁移完成后移除。
 */
export function fadeCurveGain(t: number, curve: FadeCurveType): number {
    switch (curve) {
        case "linear":
            return fadeGainSigned(0, 0, "in", t);
        case "exponential":
            return fadeGainSigned(FADE_LATE_SLIGHT, 1, "in", t);
        case "logarithmic":
            return fadeGainSigned(FADE_CONVEX_SLIGHT, 0.33, "in", t);
        case "scurve":
            return fadeGainSigned(FADE_S_SLIGHT, 0, "in", t);
        case "sine":
        default:
            // 旧 sine ≈ 轻微 S 的近似；旧 exponential √t 特性由快收 dir=1 承担。
            return fadeGainSigned(FADE_S_SLIGHT, 0, "in", t);
    }
}

/** 淡入高亮区域路径（REAPER 形状/曲率）。 */
export function fadeInAreaPath(
    width: number,
    height: number,
    steps = 24,
    shape = 0,
    dir = 0,
): string {
    if (width <= 0 || height <= 0) return "";
    const pts: Array<{ x: number; y: number }> = [];
    for (let i = 0; i < steps; i++) {
        const t = i / Math.max(1, steps - 1);
        const x = t * width;
        const g = fadeGainSigned(shape, dir, "in", t);
        const y = height * (1 - g);
        pts.push({ x, y });
    }
    // Fill the area above the fade curve so the dark emphasis sits on top.
    let d = `M 0 0`;
    for (const p of pts) d += ` L ${p.x.toFixed(2)} ${p.y.toFixed(2)}`;
    d += ` L ${width.toFixed(2)} 0 Z`;
    return d;
}

/** 淡出高亮区域路径（t=0 起点、t=1 终点，增益经 1-t 镜像）。 */
export function fadeOutAreaPath(
    width: number,
    height: number,
    steps = 24,
    shape = 0,
    dir = 0,
): string {
    if (width <= 0 || height <= 0) return "";
    const pts: Array<{ x: number; y: number }> = [];
    for (let i = 0; i < steps; i++) {
        const t = i / Math.max(1, steps - 1);
        const x = t * width;
        // fadeOut: t=0 时增益为 1，t=1 时增益为 0，故用 1-t 映射
        const g = fadeGainSigned(shape, dir, "out", t);
        const y = height * (1 - g);
        pts.push({ x, y });
    }
    // Fill the area above the fade curve so the dark emphasis sits on top.
    let d = `M 0 0`;
    for (const p of pts) d += ` L ${p.x.toFixed(2)} ${p.y.toFixed(2)}`;
    d += ` L ${width.toFixed(2)} 0 Z`;
    return d;
}
