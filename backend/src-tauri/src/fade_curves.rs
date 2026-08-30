//! REAPER 风格的 Clip 淡入淡出曲线数学核心。
//!
//! 与前端 `frontend/src/components/layout/timeline/reaperFade.ts` 保持公式级
//! 一致（双侧各有性质测试与同一组黄金值锚点），修改任一侧必须同步另一侧。
//!
//! ## 模型（依据 REAPER 实测反推）
//! 七个预设是**同一条连续弯曲轴上的命名锚点**：上升视角 g_in(t)=t^e，
//! e<1 凸（快起向）、e=1 直线、e>1 凹（快收向）；S 族为对称比值曲线
//! g(t)=t^a/(t^a+(1-t)^a)（a≥1）。曲率 dir∈[-1,1] 经方向镜像 σ
//! （淡入 +1 / 淡出 −1）归一到 u=σ·dir 驱动轴：
//! log2(e) = log2(p0) + k·(u − u0)。（详见 TS 侧模块文档。）
//!
//! ## 端点着陆（de-click landing）
//! 幂族曲线在端点斜率可能趋近无穷（e<1 时淡出末端 g(ε)=ε^e 下降极快），
//! 逐帧点采样会在 clip 末尾留下不可忽略的增益阶跃 → Click。为此对所有
//! 曲线施加 raised-cosine 着陆窗：淡出在末尾 `FADE_LANDING_FRAC` 区间
//! 内把增益平滑拉到 0、淡入在开头对称地从 0 平滑拉起（C¹ 衔接、端点
//! 零斜率）。内部采样点（黄金锚点 t≤0.75 / ≥0.25）不受影响。

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// 指数安全范围（与 TS E_MIN/E_MAX 同步）。
const E_MIN: f64 = 0.01;
const E_MAX: f64 = 64.0;

/// 幂族预设世界轴参数：(p0, u0, k)。与 TS POWER_ANCHORS 同步。
fn power_anchor(base: i64) -> Option<(f64, f64, f64)> {
    match base {
        // linear：全量程视图
        0 => Some((1.0, 0.0, 2.4)),
        // fast_start：线性点 z≈0.33（用户实测目测）
        1 => Some((0.45, 0.0, 3.490_918_464_985)),
        // fast_end
        2 => Some((3.2, 1.0, 1.678_071_905_112_637_8)),
        // steep fast_start
        3 => Some((0.10, -1.0, std::f64::consts::LOG2_10)),
        // steep fast_end
        4 => Some((8.0, 1.0, 3.0)),
        _ => None,
    }
}

/// S 族预设参数 (a0, ks)。与 TS S_ANCHORS 同步（steep 明显更陡）。
fn s_anchor(base: i64) -> Option<(f64, f64)> {
    match base {
        5 => Some((2.2, 0.9)),
        6 => Some((7.0, 1.3)),
        _ => None,
    }
}

enum ShapeSpec {
    Power { p0: f64, u0: f64, k: f64 },
    SmoothS { a0: f64, ks: f64 },
    EqualPower,
    LinearPower,
}

fn resolve_shape(shape: f64) -> ShapeSpec {
    if !shape.is_finite() {
        return ShapeSpec::LinearPower;
    }
    let base = shape.trunc() as i64;
    let has_fraction = (shape - shape.trunc()).abs() > f64::EPSILON;
    if has_fraction {
        return match base {
            1 => ShapeSpec::EqualPower,
            5 | 6 => ShapeSpec::SmoothS {
                a0: 7.0,
                ks: 1.3,
            },
            _ => ShapeSpec::LinearPower,
        };
    }
    if let Some((p0, u0, k)) = power_anchor(base) {
        return ShapeSpec::Power { p0, u0, k };
    }
    if let Some((a0, ks)) = s_anchor(base) {
        return ShapeSpec::SmoothS { a0, ks };
    }
    ShapeSpec::LinearPower
}

/// 上升视角核心：x ∈ (0,1)、世界轴坐标 u → 增益。
fn core_ascending(spec: &ShapeSpec, u: f64, x: f64) -> f64 {
    match spec {
        ShapeSpec::EqualPower => {
            let s = (x * std::f64::consts::FRAC_PI_2).sin();
            s * s
        }
        ShapeSpec::SmoothS { a0, ks } => {
            let a = (*a0 * (ks * u).exp2()).max(1.0);
            let xa = x.powf(a);
            xa / (xa + (1.0 - x).powf(a))
        }
        ShapeSpec::Power { p0, u0, k } => {
            let mut e = (p0.log2() + k * (u - *u0)).exp2();
            e = e.clamp(E_MIN, E_MAX);
            x.powf(e)
        }
        ShapeSpec::LinearPower => {
            let e = (2.4 * u).exp2().clamp(E_MIN, E_MAX);
            x.powf(e)
        }
    }
}

/// 计算淡化增益（方向感知入口；与 TS fadeGainSigned 同语义）。
///
/// - `mode_out=false` 淡入（上升）；`true` 淡出（下降；内部做时间镜像与 σ 归一）。
/// - 端点精确：淡入 g(0)=0/g(1)=1；淡出反向。
/// - 端点着陆：淡出在末尾 `FADE_LANDING_FRAC` 区间、淡入在开头同长度区间
///   乘 raised-cosine 着陆窗，保证端部零斜率、逐帧增益步长有界（防 Click）。
pub fn fade_gain_signed(shape: f64, dir: f64, mode_out: bool, t: f64) -> f64 {
    let t = if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 };
    let dir = if dir.is_finite() { dir.clamp(-1.0, 1.0) } else { 0.0 };
    if mode_out {
        if t <= 0.0 {
            return 1.0;
        }
        if t >= 1.0 {
            return 0.0;
        }
    } else if t <= 0.0 {
        return 0.0;
    } else if t >= 1.0 {
        return 1.0;
    }
    let sigma: f64 = if mode_out { -1.0 } else { 1.0 };
    let u = sigma * dir;
    let x = if mode_out { 1.0 - t } else { t };
    let spec = resolve_shape(shape);
    let gain = core_ascending(&spec, u, x) * landing_window(t, mode_out);
    if gain.is_finite() {
        gain
    } else {
        x
    }
}

/// 兼容旧调用形态的淡入求值（当前消费端均走 signed/LUT 路径，保留 API）。
#[allow(dead_code)]
pub fn fade_gain(shape: f64, dir: f64, t: f64) -> f64 {
    fade_gain_signed(shape, dir, false, t)
}

/// LUT 尺寸：音频混音用。
pub const FADE_LUT_SIZE: usize = 1024;

/// 端点着陆窗长度（归一化进度比例）。
///
/// 淡出的末尾这段 / 淡入的开头这段被 raised-cosine 窗覆盖，把曲线平滑
/// 拉回/拉离 0。取 1/8：在最恶劣的“先慢后快”组合（e≈0.04）下，10ms 级
/// 淡化（N≈480）的逐帧增益步长仍 ≤ ~2.4%，人耳不可闻；再小的 τ 会把
/// 残余阶跃压回一个采样点内。曲线中部形态（黄金锚点区间）完全不受影响。
pub const FADE_LANDING_FRAC: f64 = 0.125;

/// raised-cosine 着陆窗：淡出在 (1-τ, 1]、淡入在 [0, τ) 内取值。
///
/// - 淡出：t=1-τ 处窗值为 1（且导数 0，与原始曲线 C¹ 衔接），t=1 处为 0；
/// - 淡入：t=τ 处窗值为 1（且导数 0），t=0 处为 0；
/// - 窗外恒为 1。
#[inline]
fn landing_window(t: f64, mode_out: bool) -> f64 {
    let tau = FADE_LANDING_FRAC;
    let u = if mode_out {
        let lo = 1.0 - tau;
        if t <= lo {
            return 1.0;
        }
        (t - lo) / tau
    } else {
        if t >= tau {
            return 1.0;
        }
        (tau - t) / tau
    };
    let x = u * std::f64::consts::FRAC_PI_2;
    let c = x.cos();
    c * c
}

/// 构建一条完整淡化查表（含两端点，共 N+1 项）。
///
/// `mode_out` 决定 σ 方向镜像；表内仍按"进度升序"采样 —— 淡出消费端以
/// 剩余比例作为索引进度即可得到正确的时间镜像轨迹（见 mix.rs/mixdown.rs）。
pub fn build_fade_lut(shape: f64, dir: f64, mode_out: bool) -> Vec<f32> {
    (0..=FADE_LUT_SIZE)
        .map(|i| {
            fade_gain_signed(shape, dir, mode_out, i as f64 / FADE_LUT_SIZE as f64) as f32
        })
        .collect()
}

/// 进程内 LUT 缓存：按键为 (shape bits, dir bits, mode_out)。
/// 条目以 `Arc` 共享：命中路径只做引用计数，不克隆整表。
pub struct FadeLutCache {
    entries: Mutex<HashMap<(u64, u64, bool), Arc<Vec<f32>>>>,
}

impl FadeLutCache {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub fn lut(&self, shape: f64, dir: f64, mode_out: bool) -> Arc<Vec<f32>> {
        let key = (shape.to_bits(), dir.to_bits(), mode_out);
        if let Ok(map) = self.entries.lock() {
            if let Some(hit) = map.get(&key) {
                return hit.clone();
            }
        }
        let table = Arc::new(build_fade_lut(shape, dir, mode_out));
        if let Ok(mut map) = self.entries.lock() {
            if map.len() > 256 {
                map.clear();
            }
            map.insert(key, Arc::clone(&table));
        }
        table
    }
}

impl Default for FadeLutCache {
    fn default() -> Self {
        Self::new()
    }
}

/// 进程级共享 LUT 缓存（混音/导出共用）。
///
/// `mode_out=true` 生成"淡出侧 σ 镜像"后的表；消费端：
/// - 淡入：索引 = 进度 × N；
/// - 淡出：索引 = **剩余比例** × N（表的第 x 项对应"还剩 x 比例"时刻），
///   从而时间上从 g(x=1) 衰减到 g(x=0)，完成镜像而无需反转表。
pub fn global_fade_lut(shape: f64, dir: f64, mode_out: bool) -> Arc<Vec<f32>> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<FadeLutCache> = OnceLock::new();
    CACHE.get_or_init(FadeLutCache::new).lut(shape, dir, mode_out)
}

/// 从 LUT 取增益（index 为浮点帧位置，线性插值；越界返回边界值）。
pub fn sample_fade_lut(table: &[f32], index: f64) -> f32 {
    if table.is_empty() {
        return 1.0;
    }
    if index <= 0.0 {
        return table[0];
    }
    let last = table.len() - 1;
    if index >= last as f64 {
        return table[last];
    }
    let i = index.floor() as usize;
    let frac = (index - i as f64) as f32;
    table[i] * (1.0 - frac) + table[i + 1] * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHAPES: [f64; 9] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.1, 5.1];

    #[test]
    fn endpoints_are_exact_for_all_shapes_and_dirs() {
        for &s in &SHAPES {
            for &d in &[-1.0, -0.35, 0.0, 0.35, 1.0] {
                for out in [false, true] {
                    assert_eq!(fade_gain_signed(s, d, out, 0.0), if out { 1.0 } else { 0.0 }, "s={s} d={d} out={out}");
                    assert_eq!(fade_gain_signed(s, d, out, 1.0), if out { 0.0 } else { 1.0 }, "s={s} d={d} out={out}");
                }
            }
        }
    }

    #[test]
    fn gain_is_monotonic_in_t() {
        for &s in &SHAPES {
            for &d in &[-1.0, 0.0, 1.0] {
                for out in [false, true] {
                    let prev = fade_gain_signed(s, d, out, 0.0);
                    let mut p = prev;
                    for step in 1..=200 {
                        let t = step as f64 / 200.0;
                        let g = fade_gain_signed(s, d, out, t);
                        if out {
                            assert!(
                                g <= p + 1e-9,
                                "out must be nonincreasing s={s} d={d} t={t}"
                            );
                        } else {
                            assert!(
                                g >= p - 1e-9,
                                "in must be nondecreasing s={s} d={d} t={t}"
                            );
                        }
                        p = g;
                    }
                }
            }
        }
    }

    /// 双语（Rust/TS）一致性锚点。改公式或常数时必须同步
    /// `reaperFade.test.ts` 的 golden 用例（同一组数值）。
    #[test]
    fn golden_values_match_typescript_anchor_set() {
        // (shape, out, dir, t, expected)
        let cases: Vec<(f64, bool, f64, f64, f64)> = vec![
            (1.0, false, 0.0, 0.25, 0.5358867312681466),
            (1.0, false, 0.33, 0.25, 0.25),
            (1.0, true, -0.33, 0.75, 0.25),
            (2.0, false, 1.0, 0.25, 0.011841535675862483),
            (2.0, false, 0.0, 0.25, 0.25),
            (3.0, false, -1.0, 0.25, 0.8705505632961241),
            (4.0, false, 1.0, 0.25, 1.52587890625e-05),
            (0.0, false, -1.0, 0.25, 0.7690081847607293),
            (0.0, false, 0.0, 0.37, 0.37),
            (5.0, false, 0.0, 0.25, 0.08188949557659113),
            (5.0, false, 1.0, 0.25, 0.010876844847775608),
            (6.0, false, 0.0, 0.25, 0.0004570383912248629),
            (6.0, false, 0.0, 0.50, 0.5),
            (1.1, false, 0.0, 0.25, 0.14644660940672624),
        ];
        for (shape, out, dir, t, expected) in cases {
            let got = fade_gain_signed(shape, dir, out, t);
            assert!(
                (got - expected).abs() < 1e-9,
                "golden mismatch shape={shape} out={out} dir={dir} t={t}: got {} want {}",
                got, expected
            );
        }
    }

    #[test]
    fn lut_matches_direct_evaluation_in_and_out() {
        for out in [false, true] {
            let table = global_fade_lut(3.0, 0.35, out);
            for step in 0..=100 {
                let idx = FADE_LUT_SIZE as f64 * step as f64 / 100.0;
                let want =
                    fade_gain_signed(3.0, 0.35, out, idx / FADE_LUT_SIZE as f64) as f32;
                let got = sample_fade_lut(&table, idx);
                assert!((got - want).abs() < 5e-3, "idx={idx} out={out}");
            }
        }
    }

    /// 着陆窗在衔接点两侧连续且带符号一致（C⁰ 连续；导数零是二阶性质，
    /// 由窗函数定义保证，这里验证一阶采样步长在衔接点不突跳）。
    #[test]
    fn landing_window_is_continuous_at_junctions() {
        let eps = 1e-9;
        for &s in &SHAPES {
            for &d in &[-1.0, 0.0, 1.0] {
                let lo = 1.0 - FADE_LANDING_FRAC;
                let g_lo_m = fade_gain_signed(s, d, true, lo - eps);
                let g_lo_p = fade_gain_signed(s, d, true, lo + eps);
                assert!((g_lo_m - g_lo_p).abs() < 1e-6, "out junction s={s} d={d}");
                let g_hi_m = fade_gain_signed(s, d, false, FADE_LANDING_FRAC - eps);
                let g_hi_p = fade_gain_signed(s, d, false, FADE_LANDING_FRAC + eps);
                assert!((g_hi_m - g_hi_p).abs() < 1e-6, "in junction s={s} d={d}");
            }
        }
    }

    /// 端点锁定后的逐帧采样（淡出用 (k+1)/N 进度、淡入对称）必须：
    /// 1. 最后一帧（淡出）/ 第一帧（淡入）精确落在 0；
    /// 2. 且整条增益序列的逐帧步长有界 —— 对“先慢后快”最恶劣组合
    ///    （e 低至 ~0.04）也不得再在末尾留下不可闻阈值以上的阶跃。
    #[test]
    fn per_frame_gain_step_is_click_safe_for_slow_start_fade_outs() {
        // (shape, dir) 使淡出 u = -dir < 0 → e < 1（末端陡峭）。
        let worst_cases: &[(f64, f64)] = &[(1.0, 1.0), (1.0, 0.6), (3.0, 1.0), (3.0, 0.35)];
        for &(shape, dir) in worst_cases {
            for &frames in &[480usize, 2048, 48000] {
                let table = global_fade_lut(shape, dir, true);
                let mut prev = 1.0f32;
                let mut max_step = 0.0f32;
                for frame in 0..frames {
                    // 端点锁定约定：最后一帧进度恰为 1（与 mix.rs/mixdown.rs 一致）。
                    let consumed = (frame + 1) as f64 / frames as f64;
                    let g = sample_fade_lut(&table, consumed * FADE_LUT_SIZE as f64);
                    assert!(
                        g <= prev + 1e-6,
                        "fade-out must stay monotonically decreasing s={shape} d={dir} N={frames} frame={frame}"
                    );
                    max_step = max_step.max(prev - g);
                    prev = g;
                }
                assert_eq!(
                    prev, 0.0,
                    "last frame must be exactly silent s={shape} d={dir}"
                );
                assert!(
                    max_step < 0.025,
                    "per-frame gain step too large (click risk) s={shape} d={dir} N={frames}: {max_step}"
                );
            }
        }
    }

    #[test]
    fn per_frame_gain_step_is_click_safe_for_fast_start_fade_ins() {
        // 淡入 u = dir < 0 → e < 1（起点陡峭），对称于淡出问题。
        let worst_cases: &[(f64, f64)] = &[(1.0, -1.0), (1.0, -0.6), (3.0, -1.0), (3.0, -0.35)];
        for &(shape, dir) in worst_cases {
            for &frames in &[480usize, 2048, 48000] {
                let table = global_fade_lut(shape, dir, false);
                let mut prev = 0.0f32;
                let mut max_step = 0.0f32;
                for frame in 0..frames {
                    let consumed = (frame + 1) as f64 / frames as f64;
                    let g = sample_fade_lut(&table, consumed * FADE_LUT_SIZE as f64);
                    assert!(
                        g >= prev - 1e-6,
                        "fade-in must stay monotonically increasing s={shape} d={dir} N={frames} frame={frame}"
                    );
                    max_step = max_step.max(g - prev);
                    prev = g;
                }
                assert!(
                    max_step < 0.025,
                    "per-frame gain step too large (click risk) s={shape} d={dir} N={frames}: {max_step}"
                );
            }
        }
    }
}
