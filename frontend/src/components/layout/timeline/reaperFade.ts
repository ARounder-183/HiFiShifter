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

/**
 * 端点着陆窗长度（归一化进度比例，与 Rust `fade_curves.rs`
 * `FADE_LANDING_FRAC` 同步）。淡出的末尾 / 淡入的开头这段被
 * raised-cosine 窗覆盖，把曲线平滑拉回/拉离 0。
 */
export const FADE_LANDING_FRAC = 1 / 8;

/**
 * raised-cosine 着陆窗：淡出在 (1-τ, 1]、淡入在 [0, τ) 内取值；窗外恒为 1。
 * 两端窗导数均为 0，与原始曲线 C¹ 衔接（淡出 t=1 处、淡入 t=0 处零斜率）。
 */
function landingWindow(t: number, mode: "in" | "out"): number {
    const tau = FADE_LANDING_FRAC;
    let u: number;
    if (mode === "out") {
        const lo = 1 - tau;
        if (t <= lo) return 1;
        u = (t - lo) / tau;
    } else {
        if (t >= tau) return 1;
        u = (tau - t) / tau;
    }
    const c = Math.cos((u * Math.PI) / 2);
    return c * c;
}

type ShapeSpec =
    | { kind: "power"; p0: number; u0: number; k: number }
    | { kind: "s"; a0: number; ks: number }
    | { kind: "equalPower" }
    | { kind: "linearPower" };

/** 形状 id 非法（NaN / Infinity）时的退化规格。 */
const LINEAR_POWER_SPEC: ShapeSpec = { kind: "linearPower" };

/**
 * 已解析的形状规格缓存。
 *
 * 【为什么需要】`resolveShapeSpec` 是**纯查表**（输入只有形状 id），但此前每次
 * 调用都构造一个新对象字面量。它在波形几何重建的热路径上按「像素列 × 列内切片
 * × 极值」被调用——带淡变的 400 clip / 全览实测约 2.5 万次/次重建，等于每次重建
 * 白送 2.5 万次短命对象分配，落在渲染关键路径上。
 *
 * 形状 id 是 REAPER 的 0..6 加少量小数变体，取值集合极小且封闭，用 Map 缓存后
 * 命中率接近 100%，且返回的对象只被读取（`coreAscending` 不解构、不修改）。
 */
const shapeSpecCache = new Map<number, ShapeSpec>();

/**
 * 形状规格缓存的上限。
 *
 * 合法形状 id 只有 0..6 与少量小数变体，64 足够容纳全部。上限存在的意义是
 * 兜住**畸形输入**（例如损坏工程里出现任意浮点数形状 id）——缓存一旦无界，
 * 坏数据就能让它在渲染循环里无限增长。超出上限后只计算不缓存，行为不变。
 */
const SHAPE_SPEC_CACHE_MAX = 64;

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
    if (!Number.isFinite(shape)) return LINEAR_POWER_SPEC;
    const cached = shapeSpecCache.get(shape);
    if (cached !== undefined) return cached;
    const spec = computeShapeSpec(shape);
    if (shapeSpecCache.size < SHAPE_SPEC_CACHE_MAX) shapeSpecCache.set(shape, spec);
    return spec;
}

/** 形状 id → 规格的纯计算（结果被 `resolveShapeSpec` 缓存）。 */
function computeShapeSpec(shape: number): ShapeSpec {
    const base = Math.trunc(shape);
    const hasFraction = Math.abs(shape - base) > Number.EPSILON;
    if (hasFraction) {
        // 官方小数变体：1.1 → 内部形状 7（等功率）；5.1/6.x → 锐利 S。
        if (base === 1) return { kind: "equalPower" };
        if (base === 5 || base === 6) return { kind: "s", ...S_ANCHORS[FADE_S_SHARP] };
        return LINEAR_POWER_SPEC;
    }
    const anchor = POWER_ANCHORS[base];
    if (anchor) return { kind: "power", ...anchor };
    const sAnchor = S_ANCHORS[base];
    if (sAnchor) return { kind: "s", ...sAnchor };
    return LINEAR_POWER_SPEC;
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
 *
 * 端点着陆：淡出末尾/淡入开头乘 raised-cosine 窗（见模块头注释），
 * 保证端部零斜率、逐帧增益步长有界（防 Click）。
 */
export function fadeGainSigned(shape: number, dir: number, mode: "in" | "out", t: number): number {
    const clampedT = Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0;
    if (!Number.isFinite(dir)) dir = 0;
    if (clampedT <= 0) return mode === "out" ? 1 : 0;
    if (clampedT >= 1) return mode === "out" ? 0 : 1;
    const sigma = mode === "out" ? -1 : 1;
    const u = sigma * Math.min(1, Math.max(-1, dir));
    const x = mode === "out" ? 1 - clampedT : clampedT;
    const spec = resolveShapeSpec(shape);
    const gain = coreAscending(spec, u, x) * landingWindow(clampedT, mode);
    return Number.isFinite(gain) ? gain : clampedT;
}

/**
 * 兼容旧调用形态的淡入求值（等价 fadeGainSigned(s,d,'in',t)）。
 */
export function fadeGain(shape: number, dir: number, t: number): number {
    return fadeGainSigned(shape, dir, "in", t);
}

// ── 宿主连续两轴（curvature, S）→ 曲线 ────────────────────────────────────────
//
// 【为什么单独一套】REAPER ≥7.81 的淡变由 `D_FADE*DIR_NEW`（curvature）与
// `D_FADE*DIR2_NEW`（S）**两个正交轴**共同决定；`C_FADE*SHAPE` 只是由它们派生出的
// 粗分类读数（见 `hostFadeAxes.ts` 与 Rust `fade_axes.rs` 的实测说明）。宿主轴上
// **没有**形状号可用，必须直接从两根轴求值。
//
// 【模型】两根轴各自驱动一个变换，**复合**而不是相加：
//     g(x) = S_a( x^e )
//   · `e = E(curvature)`：幂族指数，只由 curvature 驱动；
//   · `a = A(S)`：S 族陡度，只由 S 驱动。
// `S_a(y) = y^a / (y^a + (1-y)^a)` 在 `a = 1` 时恰为恒等 —— 两族于是在 `(0, 0)`
// 处自然相接（都退化为线性），复合整体保持单调、端点精确。
//
// 【为什么是复合而不是按 |S| 混合】混合在预设 5 的坐标 `(0, 0.5)` 上会得到
// `0.5·线性 + 0.5·S(2.2)`，**不等于**预设 5 自己的曲线；复合则在七个实测坐标上
// **逐点精确**复现各自的预设曲线（两侧各有黄金值测试钉住）。旧实现既按
// `curvature + S` 求值（把两根**正交**轴相加），又"命中实测表就换预设曲线、差一个
// ulp 就走混合"——同一坐标上相差半个满量程，是用户报障的直接原因。
//
// 【锚点来源】五个曲率锚点就是 `POWER_ANCHORS` 各预设的 `p0`（该预设在自己的默认
// 方向上的指数）；两个 S 锚点就是 `S_ANCHORS` 的 `a0`。锚点不是新造的数，而是既有
// 预设参数 —— 表与曲线永远不会分家。

/** 一根轴上的锚点 `(坐标, 值)`，按坐标升序。 */
type AxisKnots = ReadonlyArray<readonly [number, number]>;

/** 单调三次插值的一段（Hermite 系数已预乘步长）。 */
interface HermiteSegment {
    readonly x0: number;
    readonly h: number;
    readonly y0: number;
    readonly y1: number;
    readonly m0h: number;
    readonly m1h: number;
}

/**
 * 把 `(坐标, 值)` 结点表编译成**保单调**的 C¹ 分段三次（Fritsch–Carlson）。
 *
 * 只编译一次（结点是常量），求值只剩三次多项式 —— 这个函数在波形重建的热路径上
 * 按"像素列 × 列内切片"被调用，不能每次重算切线。
 */
function compileMonotoneCubic(knots: AxisKnots): HermiteSegment[] {
    const n = knots.length;
    const h: number[] = [];
    const delta: number[] = [];
    for (let i = 0; i < n - 1; i += 1) {
        const dx = knots[i + 1][0] - knots[i][0];
        h.push(dx);
        delta.push((knots[i + 1][1] - knots[i][1]) / dx);
    }
    const m: number[] = new Array<number>(n).fill(0);
    for (let i = 1; i < n - 1; i += 1) {
        if (delta[i - 1] * delta[i] <= 0) continue;
        const w1 = 2 * h[i] + h[i - 1];
        const w2 = h[i] + 2 * h[i - 1];
        m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
    }
    // 端点切线取单侧斜率，并按保单调条件夹紧。
    const ends: ReadonlyArray<readonly [number, number]> = [
        [0, delta[0]],
        [n - 1, delta[n - 2]],
    ];
    for (const [i, d] of ends) {
        if (d === 0 || m[i] * d < 0) m[i] = 0;
        else if (Math.abs(m[i]) > Math.abs(3 * d)) m[i] = 3 * d;
    }
    return h.map((hi, i) => ({
        x0: knots[i][0],
        h: hi,
        y0: knots[i][1],
        y1: knots[i + 1][1],
        m0h: hi * m[i],
        m1h: hi * m[i + 1],
    }));
}

/** 在编译好的分段上求值；结点处返回**结点原值**（保证预设逐点精确）。 */
function sampleMonotoneCubic(segments: HermiteSegment[], x: number): number {
    const first = segments[0];
    if (x <= first.x0) return first.y0;
    const last = segments[segments.length - 1];
    if (x >= last.x0 + last.h) return last.y1;
    let lo = 0;
    let hi = segments.length - 1;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (x < segments[mid].x0 + segments[mid].h) hi = mid;
        else lo = mid + 1;
    }
    const s = segments[lo];
    const u = (x - s.x0) / s.h;
    const u2 = u * u;
    const u3 = u2 * u;
    return (
        (2 * u3 - 3 * u2 + 1) * s.y0 +
        (u3 - 2 * u2 + u) * s.m0h +
        (-2 * u3 + 3 * u2) * s.y1 +
        (u3 - u2) * s.m1h
    );
}

/** curvature → 幂族指数 e 的锚点：就是各幂族预设的 `p0`。 */
const CURVATURE_EXPONENT_KNOTS: AxisKnots = [
    [-1, POWER_ANCHORS[FADE_LATE_SHARP].p0],
    [-0.5, POWER_ANCHORS[FADE_LATE_SLIGHT].p0],
    [0, POWER_ANCHORS[FADE_LINEAR].p0],
    [0.5, POWER_ANCHORS[FADE_CONVEX_SLIGHT].p0],
    [1, POWER_ANCHORS[FADE_CONVEX_SHARP].p0],
];

/**
 * S → S 族陡度 a 的锚点：两个 S 预设的 `a0`，负半轴按 `a(-S) = 1/a(S)` 取镜像。
 *
 * `S_{1/a}` 恰是 `S_a` 的函数反演（`odds(y) = odds(x)^a`），所以负 S 是正 S 的
 * "反向 S"——单调、连续、在 `S = 0` 处与恒等相接。**负半轴没有实测数据**
 * （实测表只有 `S ∈ {0, 0.5, 1}`），此镜像是一个待取证的有界外推。
 */
const S_STEEPNESS_KNOTS: AxisKnots = [
    [-1, 1 / S_ANCHORS[FADE_S_SHARP].a0],
    [-0.5, 1 / S_ANCHORS[FADE_S_SLIGHT].a0],
    [0, 1],
    [0.5, S_ANCHORS[FADE_S_SLIGHT].a0],
    [1, S_ANCHORS[FADE_S_SHARP].a0],
];

const CURVATURE_LOG2_SEGMENTS = compileMonotoneCubic(
    CURVATURE_EXPONENT_KNOTS.map(([c, e]) => [c, Math.log2(e)] as const),
);
const S_LOG2_SEGMENTS = compileMonotoneCubic(
    S_STEEPNESS_KNOTS.map(([s, a]) => [s, Math.log2(a)] as const),
);

/** S 族对称比值曲线：`a = 1` 为恒等，`a > 1` 慢起慢收，`a < 1` 为其反向。 */
function sWarp(y: number, a: number): number {
    if (y <= 0) return 0;
    if (y >= 1) return 1;
    const ya = Math.pow(y, a);
    return ya / (ya + Math.pow(1 - y, a));
}

/**
 * 宿主连续两轴求值（与 Rust `fade_curves::fade_gain_axes` 公式级一致）。
 *
 * 单一连续函数：无分支跳变，两轴正交（`curvature` 只进指数、`S` 只进 S 族），
 * 七个实测预设坐标上逐点等于各自的预设曲线。
 */
export function fadeGainAxes(curvature: number, s: number, mode: "in" | "out", t: number): number {
    const clampedT = Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0;
    if (clampedT <= 0) return mode === "out" ? 1 : 0;
    if (clampedT >= 1) return mode === "out" ? 0 : 1;
    const c = Number.isFinite(curvature) ? Math.min(1, Math.max(-1, curvature)) : 0;
    const sv = Number.isFinite(s) ? Math.min(1, Math.max(-1, s)) : 0;
    const x = mode === "out" ? 1 - clampedT : clampedT;
    const e = Math.min(
        E_MAX,
        Math.max(E_MIN, Math.pow(2, sampleMonotoneCubic(CURVATURE_LOG2_SEGMENTS, c))),
    );
    const inner = e === 1 ? x : Math.pow(x, e);
    const a = Math.pow(2, sampleMonotoneCubic(S_LOG2_SEGMENTS, sv));
    const gain = (a === 1 ? inner : sWarp(inner, a)) * landingWindow(clampedT, mode);
    return Number.isFinite(gain) ? gain : x;
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
 * 曲率拖拽的指针环境（指针 y 到「归一化增益」的换算依据）。
 *
 * 与 `useEditDrag` 的 `fadePointerEnv` 同源：body 顶边视口 y 是"增益 = 1"的
 * 基线，往下 `bodyHeightPx` 处为增益 0。
 */
export interface FadeCurvePointerEnv {
    readonly envTopClientY: number;
    readonly bodyHeightPx: number;
}

/**
 * 把指针位置换算为包络线上的归一化坐标 `(t, gain)`。
 *
 * @param env 指针环境（body 顶边视口 y 与 body 高度）。
 * @param side 该侧淡变在时间轴上的位置与宽度（秒）。
 * @param pointerSec 指针对应的工程时间（秒）。
 * @param clientY 指针的视口 y（CSS px）。
 * @returns `t` 为淡变区内的归一化 x、`gain` 为归一化增益；该侧宽度为 0 时返回 null。
 */
export function resolveCurvePointer(
    env: FadeCurvePointerEnv,
    side: { leftSec: number; widthSec: number },
    pointerSec: number,
    clientY: number,
): { t: number; gain: number } | null {
    if (!(side.widthSec > 1e-9)) return null;
    // 端点各让出 0.1%：t 取到 0 或 1 时曲线端点的增益是常数，最近点求解会在
    // 端点整段退化成"距离 = |Δy|"，手感发死。
    const t = Math.min(0.999, Math.max(0.001, (pointerSec - side.leftSec) / side.widthSec));
    const gain = Math.min(
        1,
        Math.max(0, 1 - (clientY - env.envTopClientY) / Math.max(1, env.bodyHeightPx)),
    );
    return { t, gain };
}

/**
 * 最近点求解器：把指针位置投影到曲线族上，返回屏幕距离最近的 (t, dir)。
 *
 * 与 {@link solveDirAt}（固定 t 的竖直反解）的本质区别：反解在"平坦带"
 * （如 S 曲线中部对曲率不敏感、或目标增益超出可达范围）会把 dir 打到
 * 边界极值，指针稍一偏移就在 ±1 之间瞬变。最近点投影则把指针视为
 * 曲线族的吸引子 —— 平坦区拖拽时 dir 平滑滑向边界，永不突变。
 *
 * @param pointerX01 指针在淡化区内的归一化 x [0,1]
 * @param pointerY01 指针的归一化目标增益 [0,1]（1 = 响）
 * @param aspectYOverX y 距离的屏幕权重（= 绘制高度 / 宽度），使距离
 *        度量在非正方形淡化区里依然符合视觉直觉
 * @param gainAt 曲线求值器 `(t, dir) => gain`；缺省用本应用的曲线族
 *        （`fadeGainSigned`）。**画布画哪条曲线，这里就必须用哪条** ——
 *        插件的新轴宿主画的是宿主两轴的混合（`hostFadeGainForAxes`），
 *        用 `fadeGainSigned` 投影会让指针和包络对不上（"拖了不跟手"）。
 */
export function solveNearestCurveDir(args: {
    shape: number;
    dir: number;
    mode: "in" | "out";
    pointerX01: number;
    pointerY01: number;
    aspectYOverX?: number;
    gainAt?: (t: number, dir: number) => number;
}): { t: number; dir: number; gain: number } {
    const rawAspect = args.aspectYOverX;
    const aspect =
        typeof rawAspect === "number" && Number.isFinite(rawAspect) && rawAspect > 0
            ? rawAspect
            : 1;
    const px = Math.min(1, Math.max(0, args.pointerX01));
    const py = Math.min(1, Math.max(0, args.pointerY01));
    const dirClamp = (d: number) => Math.min(1, Math.max(-1, d));
    const gainAt =
        args.gainAt ?? ((t: number, d: number) => fadeGainSigned(args.shape, d, args.mode, t));

    const distanceOf = (t: number, d: number): number => {
        const g = gainAt(t, d);
        const dx = t - px;
        const dy = (g - py) * aspect;
        return Math.hypot(dx, dy);
    };

    // 粗扫描：(dir × t) 网格取最优。
    const DIR_STEPS = 40;
    const T_STEPS = 64;
    let bestT = 0.5;
    let bestDir = dirClamp(args.dir);
    let bestDist = Infinity;
    for (let di = 0; di <= DIR_STEPS; di += 1) {
        const d = -1 + (2 * di) / DIR_STEPS;
        for (let ti = 0; ti <= T_STEPS; ti += 1) {
            const t = ti / T_STEPS;
            const dist = distanceOf(t, d);
            if (dist < bestDist) {
                bestDist = dist;
                bestT = t;
                bestDir = d;
            }
        }
    }

    // 局部精化：围绕最优点逐步缩小步长的网格下降。
    let stepT = 1 / T_STEPS;
    let stepD = 2 / DIR_STEPS;
    for (let iter = 0; iter < 10; iter += 1) {
        for (const [dt, dd] of [
            [stepT, 0],
            [-stepT, 0],
            [0, stepD],
            [0, -stepD],
        ]) {
            const nt = Math.min(1, Math.max(0, bestT + dt));
            const nd = dirClamp(bestDir + dd);
            const dist = distanceOf(nt, nd);
            if (dist < bestDist) {
                bestDist = dist;
                bestT = nt;
                bestDir = nd;
            }
        }
        stepT /= 2;
        stepD /= 2;
    }

    return { t: bestT, dir: bestDir, gain: fadeGainSigned(args.shape, bestDir, args.mode, bestT) };
}

/**
 * 二维最近点求解器：把指针投影到 `(curvature, S)` 曲线族上。
 *
 * 【为什么需要二维】REAPER ≥7.81 的淡变曲线由 `D_FADE*DIR_NEW` 与
 * `D_FADE*DIR2_NEW` **两个**连续轴共同决定，不是一维。{@link solveNearestCurveDir}
 * 的签名是 `(t, dir) => gain`，结构上只能解一维 —— 在 S 族曲线上拖动时它永远把 S
 * 固定成常数、只动 curvature，解出的点根本不在曲线族上，表现为"拖了不跟手 / 跳变"。
 *
 * 本函数沿用一维版**完全相同的两段式**（粗扫 + 局部精化），只是网格从
 * `(dir × t)` 升成 `(curvature × S × t)`。粗扫刻意取偶数格宽，使七个实测预设
 * （`HOST_FADE_PRESET_AXES`：curvature/S 只取 0/±0.5/±1）都精确落在网格点上，
 * 不会被邻格"吸走"而画出混合曲线。
 *
 * 旧轴宿主与独立 App 仍走一维版（那里的"形状号"本身是离散的）——本函数是
 * **并列**入口，不替换它。
 */
export function solveNearestCurveAxes(args: {
    mode: "in" | "out";
    curvature: number;
    s: number;
    pointerX01: number;
    pointerY01: number;
    aspectYOverX?: number;
    /** 曲线求值器 `(t, curvature, s) => gain`；必须与画布画的是同一条曲线。 */
    gainAt: (t: number, curvature: number, s: number) => number;
}): { t: number; curvature: number; s: number; gain: number } {
    const rawAspect = args.aspectYOverX;
    const aspect =
        typeof rawAspect === "number" && Number.isFinite(rawAspect) && rawAspect > 0
            ? rawAspect
            : 1;
    const px = Math.min(1, Math.max(0, args.pointerX01));
    const py = Math.min(1, Math.max(0, args.pointerY01));
    const axisClamp = (v: number) => Math.min(1, Math.max(-1, v));
    const gainAt = args.gainAt;
    const startC = axisClamp(args.curvature);
    const startS = axisClamp(args.s);

    /**
     * 屏幕距离 + **起始轴偏离**惩罚。
     *
     * 【为什么必须带这一项】`(curvature, S)` → 曲线是**多对一**的：表外的坐标是
     * "线性形状 + 曲率"与 S 族的有界混合，很多组 `(c, S)` 会描出经过同一指针点的
     * 曲线。若只比屏幕距离，求解器会在这些等价解里随便挑一个 —— 于是用户在一根
     * **纯曲率**曲线上拖动时，S 会被无端拽到 0.4 之类的值，曲线族整个换掉（正是
     * "跳变"）。惩罚项极小（`REG` 远小于有意义的屏幕距离），只在**屏幕距离等价**
     * 的候选之间做取舍：优先保持用户原有的两根轴。
     */
    const REG = 0.005;
    const scoreOf = (t: number, c: number, s: number): number => {
        const g = gainAt(t, c, s);
        const screen = Math.hypot(t - px, (g - py) * aspect);
        return screen + REG * (Math.abs(c - startC) + Math.abs(s - startS));
    };

    // 粗扫描：(curvature × S × t) 网格取最优。偶数格宽 ⇒ 预设坐标落在网格点上。
    const C_STEPS = 16;
    const S_STEPS = 16;
    const T_STEPS = 32;
    // 起点先入场：保证结果**永不比"什么都不改"更差**（网格可能恰好错过最优点）。
    let bestT = px;
    let bestC = startC;
    let bestS = startS;
    let bestScore = scoreOf(px, startC, startS);
    for (let ci = 0; ci <= C_STEPS; ci += 1) {
        const c = -1 + (2 * ci) / C_STEPS;
        for (let si = 0; si <= S_STEPS; si += 1) {
            const s = -1 + (2 * si) / S_STEPS;
            for (let ti = 0; ti <= T_STEPS; ti += 1) {
                const t = ti / T_STEPS;
                const score = scoreOf(t, c, s);
                if (score < bestScore) {
                    bestScore = score;
                    bestT = t;
                    bestC = c;
                    bestS = s;
                }
            }
        }
    }

    // 局部精化：三轴逐步缩小步长的网格下降（含对角方向，避免轴向锯齿）。
    let stepT = 1 / T_STEPS;
    let stepC = 2 / C_STEPS;
    let stepS = 2 / S_STEPS;
    for (let iter = 0; iter < 12; iter += 1) {
        for (const [dt, dc, ds] of [
            [stepT, 0, 0],
            [-stepT, 0, 0],
            [0, stepC, 0],
            [0, -stepC, 0],
            [0, 0, stepS],
            [0, 0, -stepS],
            [0, stepC, stepS],
            [0, -stepC, -stepS],
        ]) {
            const nt = Math.min(1, Math.max(0, bestT + dt));
            const nc = axisClamp(bestC + dc);
            const ns = axisClamp(bestS + ds);
            const score = scoreOf(nt, nc, ns);
            if (score < bestScore) {
                bestScore = score;
                bestT = nt;
                bestC = nc;
                bestS = ns;
            }
        }
        stepT /= 2;
        stepC /= 2;
        stepS /= 2;
    }

    return { t: bestT, curvature: bestC, s: bestS, gain: gainAt(bestT, bestC, bestS) };
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
    const preset = FADE_PRESETS.find((entry) => entry.shape === normalized) ?? FADE_PRESETS[0];
    return isOut
        ? DEFAULT_FADE_DIR_BY_SHAPE[preset.id].out
        : DEFAULT_FADE_DIR_BY_SHAPE[preset.id].in_;
}

/**
 * 形状 id → i18n 键（与 `ClipContextMenu` 的 `FADE_SHAPE_OPTIONS` 同源）。
 *
 * 【为什么放在本模块】它描述的是"预设叫什么"，属于预设元数据；而
 * `fadeTooltipText.ts` 与 `hostFadeDisplay.ts` 都要用它来命名形状，其中
 * `fadeTooltipText` 已经依赖 `hostFadeDisplay` —— 放在那边会成环。本模块是叶子，
 * 两边都能引。
 */
export const SHAPE_LABEL_KEYS: Record<number, string> = {
    0: "fade_shape_linear",
    1: "fade_shape_fast_start",
    2: "fade_shape_fast_end",
    3: "fade_shape_fast_start_steep",
    4: "fade_shape_fast_end_steep",
    5: "fade_shape_slow_start_end",
    6: "fade_shape_slow_start_end_steep",
};
