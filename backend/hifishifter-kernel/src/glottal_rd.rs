//! 声门源 LF 模型的 Rd 参数：拟合与逐谐波增益（张力 / tension 的核心）。
//!
//! # 主要内容
//! - [`LfModel`]：Liljencrants-Fant 声门流导数模型。由 Rd 求模型时序参数
//!   （Fant 1995 / Huber & Roebel 2014），并求其频谱（Doval & d'Alessandro 1997）。
//! - [`GlottalRd`]：由谐波幅度拟合 Rd；求 Rd 变化时各谐波的增益；在谐波之间插值。
//!
//! # 作用
//! 提供「张力」参数的物理模型层：张力不直接做频谱倾斜，而是解释为**声门源形状
//! 参数 Rd 的变化**——Rd 越小声门闭合越急促、频谱越亮（越"紧"）。因此张力编辑
//! 只重塑谐波结构，不改变基频、不引入噪声，感知响度也天然稳定（第一谐波恒为
//! 参考点，不重新归一化）。
//!
//! # 与其他模块的关系
//! - 由 [`crate::rd_tension`] 在 HNSep 谐波支上驱动：先拟合源 Rd，再按张力求
//!   目标 Rd 的逐谐波增益，施加到目标音高的谐波位置上。
//! - 由 [`crate::renderer::chain`] 的 `HiFiGanStage` 在 mel 分析**之前**调用，
//!   使声码器重合成出新的音色。
//! - 采样率无关：所有计算都在归一化时间（周期 T0 的比例）与 Hz 域完成。
//!
//! # 来源
//! 移植自 OpenUtau 0.1.571-beta（MIT）的 `OpenUtau.Core/Analysis/LfModel.cs` 与
//! `OpenUtau.Core/Analysis/GlottalRd.cs`；后者的 LF 模型部分又是 ciglet
//! （Kanru Hua，BSD-3-Clause）的 `cig_lfmodel_from_rd` / `cig_lfmodel_spectrum`
//! 的移植。本文件为 Rust 改写，算法与常量保持一致，仅调整了数据结构与命名。
//!
//! # 维护说明
//! - `Fit` 的 Itakura-Saito 距离与 64 点网格 + 抛物线细化是**数值契约**：
//!   改动网格范围/密度会改变所有既有工程渲染出的音色。
//! - `FlowShape` 的第一谐波归一化是「不重新归一化」约定的基础，不得移除。

use std::f64::consts::PI;

// ─── LF 声门模型 ──────────────────────────────────────────────────────────────

/// LF 模型的时序参数（以周期 T0 的比例表示）与激励幅度。
#[derive(Debug, Clone, Copy)]
pub struct LfModelParams {
    pub tp: f64,
    pub te: f64,
    pub ta: f64,
    pub t0: f64,
    pub ee: f64,
}

/// 展开后的模型系数（频谱求值用）。
#[derive(Debug, Clone, Copy, Default)]
struct LfParam {
    t0: f64,
    te: f64,
    tp: f64,
    ta: f64,
    wg: f64,
    sin_wg_te: f64,
    cos_wg_te: f64,
    e: f64,
    a_coef: f64,
    a: f64,
    e0: f64,
    scale: f64,
}

/// 声门源 LF 模型。
pub struct LfModel;

impl LfModel {
    /// 由 Rd 求 LF 时序参数（Fant 1995 / Huber & Roebel 2014 的拟合式）。
    ///
    /// 分段点 `rd = 0.21` / `rd = 2.7` 来自原实现的拟合区间划分，不要调整。
    pub fn from_rd(rd: f64, t0: f64, ee: f64) -> LfModelParams {
        let rap = if rd < 0.21 {
            1e-6
        } else if rd < 2.7 {
            (-1.0 + 4.8 * rd) / 100.0
        } else {
            0.323 / rd
        };
        let oqupp = 1.0 - 1.0 / (2.17 * rd);
        let (rkp, rgp) = if rd < 2.7 {
            let rkp = (22.4 + 11.8 * rd) / 100.0;
            let rgp = 0.25 * rkp / ((0.11 * rd) / (0.5 + 1.2 * rkp) - rap);
            (rkp, rgp)
        } else {
            let rgp = 9.3552e-3 + 596e-2 / (7.96 - 2.0 * oqupp);
            let rkp = 2.0 * rgp * oqupp - 1.0428;
            (rkp, rgp)
        };
        let tp = 1.0 / (2.0 * rgp);
        let te = tp * (rkp + 1.0);
        LfModelParams {
            tp,
            te,
            ta: rap,
            t0,
            ee,
        }
    }

    fn e_func(x: f64, p: &LfParam) -> f64 {
        1.0 - ((p.te - p.t0) * x).exp() - p.ta * x
    }

    fn a_func(x: f64, p: &LfParam) -> f64 {
        let c = p.wg * p.wg * p.sin_wg_te * p.a_coef - p.wg * p.cos_wg_te;
        p.sin_wg_te * p.a_coef * x * x + p.sin_wg_te * x + p.wg * (-x * p.te).exp() + c
    }

    fn a_deriv(x: f64, p: &LfParam) -> f64 {
        2.0 * p.sin_wg_te * p.a_coef * x + p.sin_wg_te - p.wg * p.te * (-x * p.te).exp()
    }

    /// 牛顿迭代 8 次求 a（原实现固定 8 次，不做收敛判定）。
    fn newton_search(p: &LfParam) -> f64 {
        let mut a = 0.0;
        for _ in 0..8 {
            let d = Self::a_deriv(a, p);
            if d == 0.0 || !d.is_finite() {
                break;
            }
            a -= Self::a_func(a, p) / d;
        }
        a
    }

    fn param_from_model(model: LfModelParams) -> LfParam {
        let mut scale = 1.0;
        const MAX_HZ: f64 = 800.0;
        let mut t0 = model.t0;
        // 高于 800 Hz 的 f0 用缩放后的模型近似（LF 模型的有效上限）。
        if t0 < 1.0 / MAX_HZ {
            scale = 1.0 / t0 / MAX_HZ;
            t0 = 1.0 / MAX_HZ;
        }
        let mut ret = LfParam {
            t0,
            te: t0 * model.te,
            tp: t0 * model.tp,
            ta: t0 * model.ta,
            scale,
            ..Default::default()
        };
        ret.wg = PI / ret.tp;
        ret.sin_wg_te = (ret.wg * ret.te).sin();
        ret.cos_wg_te = (ret.wg * ret.te).cos();

        let p0 = ret;
        let e = fzero(|x| Self::e_func(x, &p0), 1.0, 2.0 / (ret.ta + 1e-9));
        let e_te_t0 = (e * (ret.te - ret.t0)).exp();
        ret.a_coef = (1.0 - e_te_t0) / (e * e * ret.ta) + (ret.te - ret.t0) * e_te_t0 / (e * ret.ta);
        ret.e = e;
        ret.a = Self::newton_search(&ret);
        ret.e0 = -model.ee / ((ret.a * ret.te).exp() * ret.sin_wg_te);
        ret
    }

    /// 声门流导数频谱在给定频率（Hz）处的幅度。
    ///
    /// 移植自 ciglet 的 `cig_lfmodel_spectrum`（Doval & d'Alessandro 1997）。
    /// `freq` 为空时返回空 Vec。
    pub fn spectrum(model: LfModelParams, freq: &[f64]) -> Vec<f64> {
        let p = Self::param_from_model(model);
        let (e, a, wg, e0) = (p.e, p.a, p.wg, p.e0);
        let (sin_wg_te, cos_wg_te) = (p.sin_wg_te, p.cos_wg_te);
        let (te, ta, t0) = (p.te, p.ta, p.t0);
        let e1e_ta = e * (1.0 - e * ta);

        let mut magn = Vec::with_capacity(freq.len());
        for &f in freq {
            let omega = 2.0 * PI * f / p.scale;
            // asubipif = a - i*Omega
            let (asub_re, asub_im) = (a, -omega);

            // P1 = E0 / (asubipif^2 + wg^2)
            let den_re = asub_re * asub_re - asub_im * asub_im + wg * wg;
            let den_im = 2.0 * asub_re * asub_im;
            let den_norm = den_re * den_re + den_im * den_im;
            let (p1_re, p1_im) = if den_norm > 0.0 {
                (e0 * den_re / den_norm, -e0 * den_im / den_norm)
            } else {
                (0.0, 0.0)
            };

            // P2 = wg + exp(asubipif*Te) * (asubipif*sin_wgTe - wg*cos_wgTe)
            let (exp_re, exp_im) = exp_complex(asub_re * te, asub_im * te);
            let inner_re = asub_re * sin_wg_te - wg * cos_wg_te;
            let inner_im = asub_im * sin_wg_te;
            let mul_re = exp_re * inner_re - exp_im * inner_im;
            let mul_im = exp_re * inner_im + exp_im * inner_re;
            let (p2_re, p2_im) = (wg + mul_re, mul_im);

            // P3 = Ee * exp(-i*Omega*Te) / (i*e*Ta*Omega * (e + i*Omega))
            let (num_re, num_im) = exp_complex(0.0, -omega * te);
            let num_re = model.ee * num_re;
            let num_im = model.ee * num_im;
            // 分母 d1 = i*e*Ta*Omega, d2 = e + i*Omega
            let (d1_re, d1_im) = (0.0, e * ta * omega);
            let (d2_re, d2_im) = (e, omega);
            let den_re = d1_re * d2_re - d1_im * d2_im;
            let den_im = d1_re * d2_im + d1_im * d2_re;
            let den_norm = den_re * den_re + den_im * den_im;
            let (p3_re, p3_im) = if den_norm > 0.0 {
                (
                    (num_re * den_re + num_im * den_im) / den_norm,
                    (num_im * den_re - num_re * den_im) / den_norm,
                )
            } else {
                (0.0, 0.0)
            };

            // P4
            let (p4_re, p4_im) = if e1e_ta < 1.0 {
                // 数值稳定近似
                (0.0, -e * ta * omega)
            } else {
                let (x_re, x_im) = exp_complex(0.0, -omega * (t0 - te));
                let one_minus = (1.0 - x_re, -x_im);
                (
                    e1e_ta * one_minus.0,
                    e1e_ta * one_minus.1 - e * ta * omega,
                )
            };

            // G = P1*P2 + P3*P4
            let g_re = p1_re * p2_re - p1_im * p2_im + p3_re * p4_re - p3_im * p4_im;
            let g_im = p1_re * p2_im + p1_im * p2_re + p3_re * p4_im + p3_im * p4_re;
            let m = (g_re * g_re + g_im * g_im).sqrt();
            magn.push(if m.is_nan() { 0.0 } else { m });
        }
        magn
    }
}

/// exp(re + i*im) → (re, im)。
#[inline]
fn exp_complex(re: f64, im: f64) -> (f64, f64) {
    let e = re.exp();
    (e * im.cos(), e * im.sin())
}

/// Brent 求根（原实现为 `FZero`，移植自 scipy 风格实现）。
///
/// 区间端点同号时退化为中点（原实现行为）。
fn fzero<F: Fn(f64) -> f64>(func: F, xmin: f64, xmax: f64) -> f64 {
    const EPS: f64 = 1e-8;
    let (mut a, mut b) = (xmin, xmax);
    let mut fa = func(a);
    let mut fb = func(b);
    if fa * fb >= 0.0 {
        return (a + b) / 2.0;
    }
    let mut c;
    let mut fc;
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
        c = a;
        fc = fa;
    } else {
        c = a;
        fc = fa;
    }
    let mut d = 0.0;
    let mut mflag = true;
    while fb.abs() > EPS && (a - b).abs() > EPS {
        let s = if fa != fc && fb != fc {
            // 逆二次插值
            a * fb * fc / (fa - fb) / (fa - fc)
                + b * fa * fc / (fb - fa) / (fb - fc)
                + c * fa * fb / (fc - fa) / (fc - fb)
        } else {
            // 割线
            b - fb * (b - a) / (fb - fa)
        };
        let cond = (s < (3.0 * a + b) / 4.0 || s > b)
            || (mflag && (s - b).abs() >= (b - c).abs() * 0.5)
            || (!mflag && (s - b).abs() >= (c - d).abs() * 0.5)
            || (mflag && (b - c).abs() < EPS)
            || (!mflag && (c - d).abs() < EPS);
        let s = if cond {
            let s = (a + b) / 2.0;
            if s == b || s == a {
                break; // 已达精度极限
            }
            mflag = true;
            s
        } else {
            mflag = false;
            s
        };
        let fs = func(s);
        d = c;
        c = b;
        fc = fb;
        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }
    b
}

// ─── Rd 拟合与增益 ────────────────────────────────────────────────────────────

/// 声门源形状参数 Rd：由谐波幅度拟合，并给出移动 Rd 时的逐谐波增益。
///
/// Rd 越小 → 声门闭合越急促 → 高频越强（听感"更紧、更亮"）。
/// **第一谐波是参考点**：编辑保持它不变，只重塑其余谐波。
pub struct GlottalRd;

impl GlottalRd {
    pub const MIN_RD: f64 = 0.02;
    pub const MAX_RD: f64 = 3.0;
    /// 拟合时考察的最高谐波频率（Hz）。
    pub const MAX_FIT_HZ: f64 = 8000.0;
    const MAX_FIT_HARMONICS: usize = 80;
    const GRID_SIZE: usize = 64;
    const LIP_RADIUS_CM: f64 = 1.5;

    /// 每个网格 Rd 对应的**谐波功率形状**（`FlowShape^2`，长 `MAX_FIT_HARMONICS`）。
    ///
    /// 【为什么必须是进程级常量】这张表**不依赖输入**（谐波次数上 LF 形状与 f0 无关，
    /// 见 `flow_shape` 的说明），但 `fit` 是**逐帧**调用的。若在 `fit` 内部现算，
    /// 每帧都要重跑 64 次 LF 频谱求值（每次含 Brent 求根 + 8 次牛顿迭代）——
    /// 实测 0.658 ms/帧，占 `fit` 总耗时的 99%，1 分钟音频要多花约 6.9 秒。
    /// OpenUtau 用 `static readonly double[][] flowPower` 只算一次，此处对齐该做法。
    ///
    /// 用 `OnceLock` 而非 `const`：表的内容需要浮点运算才能得到，无法在编译期求值。
    fn flow_power_table() -> &'static [Vec<f64>] {
        static TABLE: std::sync::OnceLock<Vec<Vec<f64>>> = std::sync::OnceLock::new();
        TABLE.get_or_init(|| {
            Self::grid()
                .iter()
                .map(|&rd| {
                    Self::flow_shape(rd, 200.0, Self::MAX_FIT_HARMONICS)
                        .into_iter()
                        .map(|v| v * v)
                        .collect()
                })
                .collect()
        })
    }

    /// Rd 网格（MinRd..MaxRd 均分 GRID_SIZE 点）。
    fn grid() -> [f64; Self::GRID_SIZE] {
        let mut g = [0.0f64; Self::GRID_SIZE];
        for (i, v) in g.iter_mut().enumerate() {
            *v = Self::MIN_RD
                + (Self::MAX_RD - Self::MIN_RD) * i as f64 / (Self::GRID_SIZE - 1) as f64;
        }
        g
    }

    /// 张力（-100..100）对应的 Rd：+100 使 Rd 减半，-100 使 Rd 翻倍。
    pub fn tense_rd(rd: f64, tension: f64) -> f64 {
        (rd * 2.0f64.powf(-tension / 100.0)).clamp(Self::MIN_RD, Self::MAX_RD)
    }

    /// 声门流幅度（LF 导数频谱除以频率）在 f0 的第 1..n 次谐波上的相对值，
    /// **以第一谐波归一化**。
    ///
    /// 谐波次数上 LF 形状不依赖 f0（在模型 800 Hz 上限以下），故拟合时可用单一 f0。
    pub fn flow_shape(rd: f64, f0: f64, n: usize) -> Vec<f64> {
        if n == 0 {
            return Vec::new();
        }
        let freq: Vec<f64> = (0..n).map(|k| (k + 1) as f64 * f0).collect();
        let shape = LfModel::spectrum(LfModel::from_rd(rd, 1.0 / f0, 1.0), &freq);
        let first = shape.first().copied().unwrap_or(0.0);
        shape
            .iter()
            .enumerate()
            .map(|(k, &v)| if first > 0.0 { v / (k + 1) as f64 / first } else { 1.0 })
            .collect()
    }

    /// 活塞式唇辐射增益：`|j w L R / (R + j w L)|`。
    ///
    /// `R` / `L` 是唇半径对应的辐射阻与辐射感。
    pub fn lip_gain(hz: f64) -> f64 {
        let r = 128.0 / (9.0 * PI * PI);
        let l = 8.0 * Self::LIP_RADIUS_CM / 100.0 / (3.0 * PI * 340.0);
        let w = 2.0 * PI * hz;
        let wl = w * l;
        wl * r / (r * r + wl * wl).sqrt()
    }

    /// 由谐波幅度拟合 Rd。
    ///
    /// 流程：
    /// 1. 逐谐波除掉唇辐射，转成功率域；
    /// 2. 在 64 点 Rd 网格上以 **Itakura-Saito 距离**（功率域）比较，
    ///    模型增益由第一谐波定出；
    /// 3. 用抛物线在最优网格点附近细化。
    ///
    /// 少于 2 个谐波时返回 1.0（无法拟合）。`amplitudes` 为空同样返回 1.0。
    pub fn fit(amplitudes: &[f64], f0: f64) -> f64 {
        let n = amplitudes.len().min(Self::MAX_FIT_HARMONICS);
        if n < 2 || !(f0 > 0.0) {
            return 1.0;
        }
        let mut power = vec![0.0f64; n];
        for k in 0..n {
            let a = amplitudes[k] / Self::lip_gain((k + 1) as f64 * f0);
            power[k] = a * a + 1e-20;
        }

        // 进程级常量表（只算一次），不再逐帧重建。
        let shapes = Self::flow_power_table();

        let mut distance = vec![0.0f64; Self::GRID_SIZE];
        for (g, model) in shapes.iter().enumerate() {
            let gain = if model[0] > 0.0 { power[0] / model[0] } else { 0.0 };
            let mut sum = 0.0;
            for k in 0..n {
                let denom = model[k] * gain + 1e-30;
                let ratio = power[k] / denom;
                if ratio > 0.0 && ratio.is_finite() {
                    sum += ratio - ratio.ln() - 1.0;
                }
            }
            distance[g] = sum / n as f64;
        }

        let mut best = 0usize;
        for g in 1..Self::GRID_SIZE {
            if distance[g] < distance[best] {
                best = g;
            }
        }
        let mut index = best as f64;
        if best > 0 && best < Self::GRID_SIZE - 1 {
            let (y0, y1, y2) = (distance[best - 1], distance[best], distance[best + 1]);
            let curvature = y0 - 2.0 * y1 + y2;
            if curvature > 0.0 {
                index = best as f64 + (0.5 * (y0 - y2) / curvature).clamp(-0.5, 0.5);
            }
        }
        Self::MIN_RD + (Self::MAX_RD - Self::MIN_RD) * index / (Self::GRID_SIZE - 1) as f64
    }

    /// Rd 轨迹的平滑：未浊音帧由最近浊音帧填充（线性插值），再做滑动平均。
    ///
    /// 全未浊音时全部填 1.0。`window <= 1` 时只做填充。
    pub fn smooth(rd: &[f64], voiced: &[bool], window: usize) -> Vec<f64> {
        let n = rd.len();
        if n == 0 {
            return Vec::new();
        }
        let known: Vec<usize> = (0..n.min(voiced.len())).filter(|&i| voiced[i]).collect();
        let mut filled = vec![0.0f64; n];
        if known.is_empty() {
            filled.fill(1.0);
            return filled;
        }
        let mut j = 0usize;
        for i in 0..n {
            while j + 1 < known.len() && known[j + 1] <= i {
                j += 1;
            }
            if i <= known[0] {
                filled[i] = rd[known[0]];
            } else if i >= known[known.len() - 1] {
                filled[i] = rd[known[known.len() - 1]];
            } else {
                let (a, b) = (known[j], known[j + 1]);
                filled[i] = rd[a] + (rd[b] - rd[a]) * (i - a) as f64 / (b - a) as f64;
            }
        }
        if window <= 1 {
            return filled;
        }
        let mut smoothed = vec![0.0f64; n];
        let half = window / 2;
        for i in 0..n {
            let mut sum = 0.0;
            for w in 0..window {
                let idx = (i as isize + w as isize - half as isize).clamp(0, n as isize - 1);
                sum += filled[idx as usize];
            }
            smoothed[i] = sum / window as f64;
        }
        smoothed
    }

    /// Rd 从 `rd` 变到 `rd2` 时，f0 的第 1..n 次谐波的幅度增益（第一谐波为 1）。
    pub fn gains(rd: f64, rd2: f64, f0: f64, n: usize) -> Vec<f64> {
        let from = Self::flow_shape(rd, f0, n);
        let to = Self::flow_shape(rd2, f0, n);
        (0..n)
            .map(|k| if from[k] > 0.0 { to[k] / from[k] } else { 1.0 })
            .collect()
    }

    /// 由 f0 的逐谐波增益插值出任意频率处的增益。
    ///
    /// 谐波之间**对数线性**插值；第一谐波以下恒为 1；最后一个谐波以上取末值。
    pub fn gain_at(gains: &[f64], f0: f64, hz: f64) -> f64 {
        if gains.is_empty() || !(f0 > 0.0) {
            return 1.0;
        }
        let k = hz / f0 - 1.0;
        if k <= 0.0 {
            return 1.0;
        }
        if k >= (gains.len() - 1) as f64 {
            return gains[gains.len() - 1];
        }
        let i = k.floor() as usize;
        let t = k - i as f64;
        let a = gains[i].max(1e-9).ln();
        let b = gains[i + 1].max(1e-9).ln();
        (a * (1.0 - t) + b * t).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LF 频谱在若干 (rd, f0, freq) 上必须是有限正值。
    #[test]
    fn lf_spectrum_is_finite_and_positive() {
        for &rd in &[0.1f64, 0.5, 1.0, 2.0, 2.9] {
            for &f0 in &[80.0f64, 220.0, 500.0] {
                let freq: Vec<f64> = (1..=20).map(|k| k as f64 * f0).collect();
                let s = LfModel::spectrum(LfModel::from_rd(rd, 1.0 / f0, 1.0), &freq);
                assert_eq!(s.len(), freq.len());
                for (i, &v) in s.iter().enumerate() {
                    assert!(
                        v.is_finite() && v >= 0.0,
                        "rd={rd} f0={f0} harmonic {} gave {v}",
                        i + 1
                    );
                }
            }
        }
    }

    /// `TenseRd`：+100 减半、-100 翻倍，并钳在 [0.02, 3.0]。
    #[test]
    fn tense_rd_halves_and_doubles() {
        assert!((GlottalRd::tense_rd(1.0, 100.0) - 0.5).abs() < 1e-12);
        assert!((GlottalRd::tense_rd(1.0, -100.0) - 2.0).abs() < 1e-12);
        assert!((GlottalRd::tense_rd(1.0, 0.0) - 1.0).abs() < 1e-12);
        // 钳制
        assert!((GlottalRd::tense_rd(2.0, 100.0) - 1.0).abs() < 1e-12);
        assert_eq!(GlottalRd::tense_rd(0.001, 100.0), GlottalRd::MIN_RD);
        assert_eq!(GlottalRd::tense_rd(100.0, -100.0), GlottalRd::MAX_RD);
    }

    /// **不重新归一化的契约**：任何 Rd 变化下第一谐波增益恒为 1。
    ///
    /// 这是张力不改变感知响度的基础；若第一谐波被一起缩放，调张力会听到响度漂移。
    #[test]
    fn gains_keep_first_harmonic_at_unity() {
        for &(rd, rd2) in &[(1.0f64, 0.5f64), (1.0, 2.0), (0.5, 1.5), (2.5, 0.3)] {
            let g = GlottalRd::gains(rd, rd2, 220.0, 40);
            assert!(
                (g[0] - 1.0).abs() < 1e-9,
                "first harmonic must stay 1.0 for rd={rd}->{rd2}, got {}",
                g[0]
            );
        }
    }

    /// 张力为正（Rd 减小）时，高次谐波必须被**提升**（更亮）；
    /// 张力为负时被衰减。这是张力的方向契约。
    #[test]
    fn positive_tension_boosts_upper_harmonics() {
        let up = GlottalRd::gains(1.0, GlottalRd::tense_rd(1.0, 100.0), 220.0, 40);
        assert!(
            up[9] > 1.0,
            "tension +100 must boost harmonic 10, got {}",
            up[9]
        );
        let down = GlottalRd::gains(1.0, GlottalRd::tense_rd(1.0, -100.0), 220.0, 40);
        assert!(
            down[9] < 1.0,
            "tension -100 must attenuate harmonic 10, got {}",
            down[9]
        );
    }





    /// `Fit` 必须能从已知 Rd 生成的谐波幅度还原出该 Rd。
    #[test]
    fn fit_recovers_known_rd() {
        let f0 = 220.0;
        for &rd in &[0.4f64, 1.0, 1.8, 2.6] {
            let shape = GlottalRd::flow_shape(rd, f0, 40);
            // 造出"经唇辐射后的观测幅度"
            let amps: Vec<f64> = shape
                .iter()
                .enumerate()
                .map(|(k, &v)| v * GlottalRd::lip_gain((k + 1) as f64 * f0))
                .collect();
            let fitted = GlottalRd::fit(&amps, f0);
            assert!(
                (fitted - rd).abs() < 0.35,
                "fit({rd}) returned {fitted} (tolerance 0.35)"
            );
        }
    }

    /// 谐波不足时 `Fit` 返回中性 1.0，而不是 panic 或荒谬值。
    #[test]
    fn fit_with_too_few_harmonics_is_neutral() {
        assert_eq!(GlottalRd::fit(&[], 220.0), 1.0);
        assert_eq!(GlottalRd::fit(&[1.0], 220.0), 1.0);
        assert_eq!(GlottalRd::fit(&[1.0, 0.5], 0.0), 1.0);
    }

    /// `Smooth`：未浊音帧由邻近浊音帧填充；全未浊音 → 1.0。
    #[test]
    fn smooth_fills_unvoiced_frames() {
        let rd = vec![0.5, 0.0, 0.0, 2.0];
        let voiced = vec![true, false, false, true];
        let out = GlottalRd::smooth(&rd, &voiced, 1);
        assert!((out[0] - 0.5).abs() < 1e-9);
        assert!((out[3] - 2.0).abs() < 1e-9);
        // 中间两帧线性插值
        assert!((out[1] - 1.0).abs() < 1e-6, "got {}", out[1]);
        assert!((out[2] - 1.5).abs() < 1e-6, "got {}", out[2]);

        let none_voiced = vec![false; 4];
        let out = GlottalRd::smooth(&rd, &none_voiced, 3);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-12));
    }

    /// 滑动平均窗必须真的平滑掉逐帧抖动。
    #[test]
    fn smooth_window_reduces_jitter() {
        let rd: Vec<f64> = (0..64)
            .map(|i| if i % 2 == 0 { 0.5 } else { 2.5 })
            .collect();
        let voiced = vec![true; 64];
        let out = GlottalRd::smooth(&rd, &voiced, 9);
        let min = out.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = out.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max - min < 1.5,
            "smoothing did not reduce jitter: range {min}..{max}"
        );
    }

    /// `GainAt`：第一谐波以下为 1；谐波之间对数线性；末谐波以上取末值。
    #[test]
    fn gain_at_interpolates_between_harmonics() {
        let gains = vec![1.0, 2.0, 4.0, 8.0];
        let f0 = 100.0;
        // 第一谐波以下恒 1
        assert!((GlottalRd::gain_at(&gains, f0, 50.0) - 1.0).abs() < 1e-12);
        // 恰在谐波上
        assert!((GlottalRd::gain_at(&gains, f0, 200.0) - 2.0).abs() < 1e-9);
        // 谐波之间：1→2 的对数中点 = sqrt(2)
        let mid = GlottalRd::gain_at(&gains, f0, 150.0);
        assert!((mid - 2.0f64.sqrt()).abs() < 1e-9, "got {mid}");
        // 末谐波以上取末值
        assert!((GlottalRd::gain_at(&gains, f0, 100_000.0) - 8.0).abs() < 1e-12);
        // 空/非法输入 → 1
        assert!((GlottalRd::gain_at(&[], f0, 200.0) - 1.0).abs() < 1e-12);
        assert!((GlottalRd::gain_at(&gains, 0.0, 200.0) - 1.0).abs() < 1e-12);
    }

    /// 唇辐射增益随频率单调上升。
    #[test]
    fn lip_gain_is_monotonic_increasing() {
        let mut prev = 0.0;
        for hz in [50.0f64, 200.0, 1000.0, 4000.0, 8000.0] {
            let g = GlottalRd::lip_gain(hz);
            assert!(g > prev, "lip_gain not increasing at {hz}: {g} <= {prev}");
            assert!(g.is_finite() && g > 0.0);
            prev = g;
        }
    }
}
