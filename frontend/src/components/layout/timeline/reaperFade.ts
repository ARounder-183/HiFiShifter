/**
 * REAPER 风格 Clip 淡入淡出曲线数学核心（与 Rust `fade_curves.rs` 公式级一致）。
 *
 * 修改任一侧必须同步另一侧，并更新两侧的黄金锚点测试：
 * - Rust: `backend/src-tauri/src/fade_curves.rs`
 * - TS  : `reaperFade.test.ts` → `golden values match rust anchor set`
 *
 * ## 模型（依据 REAPER 实测反推，经七组默认曲率 + 视觉线性点交叉验证）
 *
 * 七个预设本质上是**同一条连续弯曲轴上的命名锚点**：
 *   g_in(t) = t^e，指数 e ∈ (0.08, 16] 沿轴滑动 ——
 *   e < 1 凸（快起向）、e = 1 直线、e > 1 凹（快收向）；
 *   S 族为对称比值曲线 g(t)=t^a/(t^a+(1-t)^a)（a≥1）。
 *
 * 曲率 dir ∈ [-1,1]（REAPER D_FADEINDIR/OUTDIR）通过方向镜像 σ
 * （淡入 +1 / 淡出 −1）归一到统一参数 u = σ·dir 再驱动轴：
 *   log2(e) = log2(p0) + k·(u − u0)
 * 其中 (p0, u0) 是该预设的锚点形态，k 是轴斜率，z 是它的视觉线性点
 * （由 log2(p0) + k·(z − u0) = 0 反解约束）。这精确复现了实测行为：
 * - 快起锚在 u=0（弯 p≈0.45），线性点在 dir≈±0.33，拉满 ±1 仍是深弯；
 * - 快收/陡峭快收锚在 |u|=1 且线性点恰在 dir=0；
 * - 线性预设自身就是这条轴的全量程视图（端点近似快起形态）。
 * 切换形状时必须把 dir 重置为新形状的方向相关默认值（DEFAULT_FADE_DIR_BY_SHAPE）。
 */

/** REAPER 标准形状 id。 */
export const FADE_LINEAR = 0;
export const FADE_CONVEX_SLIGHT = 1; // "Fast Start"
export const FADE_LATE_SLIGHT = 2; // "Fast End"
export const FADE_CONVEX_SHARP = 3; // "Fast Start Steep"
export const FADE_LATE_SHARP = 4; // "Fast End Steep"
export const FADE_S_SLIGHT = 5; // "Slow Start/End"
export const FADE_S_SHARP = 6; // "Slow Start/End Steep"

/** 指数安全范围：防止 dir 极值处视觉爆炸或数值下溢。 */
const E_MIN = 0.01;
const E_MAX = 64;

type ShapeSpec =
    | { kind: "power"; p0: number; u0: number; k: number }
    | { kind: "s"; a0: number; ks: number }
    | { kind: "equalPower" }
    | { kind: "linearPower" };

/**
 * 各预设的世界轴参数（待校准：与本机 REAPER 并排比对后微调 p0/z/k）。
 * k 已按"线性点 z 处 e=1"约束反解（见模块头注释）。
 */
const POWER_ANCHORS: Record<number, { p0: number; u0: number; k: number }> = {
    [FADE_LINEAR]: { p0: 1.0, u0: 0.0, k: 2.4 },
    [FADE_CONVEX_SLIGHT]: { p0: 0.45, u0: 0.0, k: 3.490918464985 },
    [FADE_LATE_SLIGHT]: { p0: 3.2, u0: 1.0, k: 1.6780719051126378 },
    [FADE_CONVEX_SHARP]: { p0: 0.1, u0: -1.0, k: 3.321928094887362 },
    [FADE_LATE_SHARP]: { p0: 8.0, u0: 1.0, k: 3.0 },
};
const S_ANCHORS: Record<number, { a0: number; ks: number }> = {
    [FADE_S_SLIGHT]: { a0: 2.2, ks: 0.9 },
    [FADE_S_SHARP]: { a0: 7.0, ks: 1.3 },
};

function resolveShapeSpec(shape: number): ShapeSpec {
    if (!Number.isFinite(shape)) return { kind: "linearPower" };
    const base = Math.trunc(shape);
    const hasFraction = Math.abs(shape - base) > Number.EPSILON;
    if (hasFraction) {
        // 官方小数变体：1.1 → 内部形状 7（等功率）；5.1/6.x → 锐利 S。
        if (base === 1) return { kind: "equalPower" };
        if (base === 5 || base === 6) return { kind: "s", ...S_ANCHORS[FADE_S_SHARP] };
        return { kind: "linearPower" };
    }
    const anchor = POWER_ANCHORS[base];
    if (anchor) return { kind: "power", ...anchor };
    const sAnchor = S_ANCHORS[base];
    if (sAnchor) return { kind: "s", ...sAnchor };
    return { kind: "linearPower" };
}

/** 上升视角核心：x ∈ (0,1) → 增益，u 为世界轴坐标。 */
function coreAscending(spec: ShapeSpec, u: number, x: number): number {
    if (spec.kind === "equalPower") {
        const s = Math.sin((x * Math.PI) / 2);
        return s * s;
    }
    if (spec.kind === "s") {
        const a = Math.max(1, spec.a0 * Math.pow(2, spec.ks * u));
        const xa = Math.pow(x, a);
        return xa / (xa + Math.pow(1 - x, a));
    }
    let e: number;
    if (spec.kind === "power") {
        e = Math.pow(2, Math.log2(spec.p0) + spec.k * (u - spec.u0));
    } else {
        e = Math.pow(2, 2.4 * u); // linearPower：全量程直线态
    }
    e = Math.min(E_MAX, Math.max(E_MIN, e));
    return Math.pow(x, e);
}

/**
 * 计算淡化增益（方向感知入口）。
 *
 * @param shape REAPER 形状 id
 * @param dir   该侧存储的曲率 [-1,1]（淡入/淡出各自约定）
 * @param mode  'in'（上升）/ 'out'（下降；内部完成时间镜像与 σ 符号归一）
 * @param t     该侧区间内的进度 [0,1]
 */
export function fadeGainSigned(
    shape: number,
    dir: number,
    mode: "in" | "out",
    t: number,
): number {
    const clampedT = Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0;
    if (!Number.isFinite(dir)) dir = 0;
    if (clampedT <= 0) return mode === "out" ? 1 : 0;
    if (clampedT >= 1) return mode === "out" ? 0 : 1;
    const sigma = mode === "out" ? -1 : 1;
    const u = sigma * Math.min(1, Math.max(-1, dir));
    const x = mode === "out" ? 1 - clampedT : clampedT;
    const spec = resolveShapeSpec(shape);
    const gain = coreAscending(spec, u, x);
    return Number.isFinite(gain) ? gain : clampedT;
}

/**
 * 兼容旧调用形态的淡入求值（等价 fadeGainSigned(s,d,'in',t)）。
 */
export function fadeGain(shape: number, dir: number, t: number): number {
    return fadeGainSigned(shape, dir, "in", t);
}

/**
 * 交互求解器：给定进度 t 与目标增益，反解该侧曲率 dir。
 * 幂族/S 族在固定 t 对 u 单调（含钳制平顶），扫描 + 二分总能给出确定解；
 * 目标不可达时夹到最近的边界端点。
 */
export function solveDirAt(
    shape: number,
    mode: "in" | "out",
    t: number,
    targetGain: number,
    preferDir = 0,
): number {
    const lo = -1;
    const hi = 1;
    const gLo = fadeGainSigned(shape, lo, mode, t);
    const gHi = fadeGainSigned(shape, hi, mode, t);

    const candidates: Array<{ dir: number; diff: number }> = [];
    let prevDir = lo;
    let prevGain = gLo;
    for (let step = 1; step <= 32; step += 1) {
        const dir = lo + ((hi - lo) * step) / 32;
        const gain = fadeGainSigned(shape, dir, mode, t);
        if ((prevGain - targetGain) * (gain - targetGain) <= 0 || gain === targetGain) {
            let a = prevDir;
            let b = dir;
            let ga = prevGain;
            for (let iter = 0; iter < 40; iter += 1) {
                const mid = (a + b) / 2;
                const gm = fadeGainSigned(shape, mid, mode, t);
                if ((ga - targetGain) * (gm - targetGain) <= 0) {
                    b = mid;
                } else {
                    a = mid;
                    ga = gm;
                }
            }
            candidates.push({ dir: (a + b) / 2, diff: Math.abs((a + b) / 2 - preferDir) });
        }
        prevDir = dir;
        prevGain = gain;
    }

    if (candidates.length === 0) {
        const scoreLo = Math.abs(gLo - targetGain) * 1000 + Math.abs(lo - preferDir);
        const scoreHi = Math.abs(gHi - targetGain) * 1000 + Math.abs(hi - preferDir);
        return scoreLo <= scoreHi ? lo : hi;
    }
    candidates.sort((x, y) => x.diff - y.diff);
    return Math.min(1, Math.max(-1, candidates[0].dir));
}


/**
 * 曲率编辑的基础形状解析。
 *
 * 模型修正后**任何预设都直接可弯**（线性是真实可弯的轴端视图，
 * 不再需要首次编辑时切换形状）；本函数仅把非法值归一到线性。
 * `promotedFromLinear` 保留在返回类型中以兼容既有调用点，恒为 false。
 */
export function resolveCurvatureEditBase(shape: number): {
    shape: number;
    promotedFromLinear: boolean;
} {
    if (!Number.isFinite(shape)) return { shape: FADE_LINEAR, promotedFromLinear: false };
    const base = Math.trunc(shape);
    const isKnown = base >= FADE_LINEAR && base <= FADE_S_SHARP;
    if (isKnown) return { shape, promotedFromLinear: false };
    return { shape: FADE_LINEAR, promotedFromLinear: false };
}

export type FadePresetId =
    | "linear"
    | "convexSlight"
    | "lateSlight"
    | "convexSharp"
    | "lateSharp"
    | "sSlight"
    | "sSharp";

/** 右键菜单七预设 ↔ REAPER 形状 id（顺序对齐 REAPER 7.x 菜单）。 */
export const FADE_PRESETS: ReadonlyArray<{ id: FadePresetId; shape: number }> = [
    { id: "linear", shape: FADE_LINEAR },
    { id: "convexSlight", shape: FADE_CONVEX_SLIGHT },
    { id: "lateSlight", shape: FADE_LATE_SLIGHT },
    { id: "convexSharp", shape: FADE_CONVEX_SHARP },
    { id: "lateSharp", shape: FADE_LATE_SHARP },
    { id: "sSlight", shape: FADE_S_SLIGHT },
    { id: "sSharp", shape: FADE_S_SHARP },
];

/**
 * 各形状切换时的曲率默认值（REAPER 实测整理，{淡入, 淡出}）。
 * 每次用户切换曲线类型（右键菜单或修饰键点击循环）后，dir 必须重置为
 * 此表中对应方向的值 —— 各形状的有效曲率中心不在同一位置，
 * 沿用旧曲率会得到错误的混合形态。
 */
export const DEFAULT_FADE_DIR_BY_SHAPE: Readonly<
    Record<FadePresetId, { in_: number; out: number }>
> = {
    linear: { in_: 0, out: 0 },
    convexSlight: { in_: 0, out: 0 },
    lateSlight: { in_: 1, out: -1 },
    convexSharp: { in_: -1, out: 1 },
    lateSharp: { in_: 1, out: -1 },
    sSlight: { in_: 0, out: 0 },
    sSharp: { in_: 0, out: 0 },
};

/** 形状 id → 该侧的重置曲率（UI 层便捷封装）。 */
export function defaultFadeDirFor(shape: number, isOut: boolean): number {
    const normalized = Math.trunc(Number.isFinite(shape) ? shape : 0);
    const preset =
        FADE_PRESETS.find((entry) => entry.shape === normalized) ?? FADE_PRESETS[0];
    return isOut
        ? DEFAULT_FADE_DIR_BY_SHAPE[preset.id].out
        : DEFAULT_FADE_DIR_BY_SHAPE[preset.id].in_;
}
