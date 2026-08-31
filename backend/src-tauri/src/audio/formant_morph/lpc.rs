/*
 * formant_morph/lpc.rs - LPC 分析与极点（共振峰）参数化。
 *
 * 主要内容：
 * - 自相关 + Levinson-Durbin：求一帧信号的 LPC 系数（a[0]=1 的全极点模型）
 *   与残差能量比（供浊音 gate 使用）。
 * - Durand-Kerner 复数求根：把 LPC 多项式分解为极点。
 * - 极点 ↔ (中心频率, 带宽) 互转：共振峰的物理参数化，以及迁移后重建系数。
 * - analyze_frame：加 Hann 窗的单帧完整分析入口。
 *
 * 与其他模块的关系：
 * - 被 formant_morph/mod.rs（DSP 主流程）与 formant_morph/analysis.rs
 *   （IPC 源共振峰统计）共用，保证"UI 显示的位置 == 算法认定的位置"。
 * - 复数类型用 crate 既有依赖 num_complex。
 *
 * 维护说明：
 * - 任何返回 None / 空 vec 的路径都表示"分析失败"，调用方必须走恒等直通，
 *   严禁用兜底常量冒充分析结果（2026-06-30 路线的教训）。
 * - 所有数值路径必须防 NaN/Inf：非有限值一律丢弃而不是传播。
 */

use num_complex::Complex32;

/// FIR 阶数（多项式阶数）的求根迭代上限。
const ROOT_ITERATIONS: usize = 100;
/// 求根收敛阈值（根位移）。
const ROOT_EPSILON: f32 = 1.0e-8;
/// 认为是"实根"的虚部阈值以下（相对半径）。
const REAL_ROOT_IMAG_EPS: f32 = 1.0e-4;

/// 一个共轭极点对（正虚部侧）的物理参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pole {
    /// 中心频率（Hz）。
    pub freq_hz: f32,
    /// 3dB 带宽近似（Hz）。
    pub bandwidth_hz: f32,
    /// 极点半径（0..1）。
    pub radius: f32,
}

/// 一帧信号分解出的全部极点。
#[derive(Debug, Clone, Default)]
pub struct FramePoles {
    /// 共轭极点对（正虚部），按频率升序。
    pub pairs: Vec<Pole>,
    /// 实根（|z| < 1）。
    pub real_roots: Vec<f32>,
}

/// 单帧 LPC 分析结果。
#[derive(Debug, Clone)]
pub struct FrameLpc {
    /// LPC 系数（升幂，a[0] = 1 隐含，即返回 a[1..=order]）。
    pub coeffs: Vec<f32>,
    /// 残差能量 / 信号能量（0..1]，越小说明全极点模型拟合越好（浊音特征）。
    pub residual_ratio: f32,
}

/// 计算帧的自相关（0..=order 阶，有偏估计）。
///
/// 返回 `None`：帧能量过低（静音）、含非有限值或阶数超过帧长。
pub fn autocorrelation(frame: &[f32], order: usize) -> Option<Vec<f32>> {
    if order == 0 || frame.len() <= order {
        return None;
    }
    let mut energy = 0.0_f32;
    for &s in frame {
        if !s.is_finite() {
            return None;
        }
        energy += s * s;
    }
    if energy < 1.0e-10 {
        return None;
    }
    let mut ac = vec![0.0_f32; order + 1];
    let inv_len = 1.0 / frame.len() as f32;
    for (lag, slot) in ac.iter_mut().enumerate() {
        let mut sum = 0.0_f32;
        for i in lag..frame.len() {
            sum += frame[i] * frame[i - lag];
        }
        *slot = sum * inv_len;
    }
    Some(ac)
}

/// Levinson-Durbin 递推求 LPC 系数。
///
/// 约定：A(z) = 1 + a1·z⁻¹ + … + ap·z⁻ⁿ，残差 e[n] = x[n] + Σ a_k·x[n-k]。
///
/// 参数：`ac` 为 0..=order 阶自相关。
/// 返回 `None`：能量过低、反射系数越界（|k| ≥ 1，模型不稳定）或出现非有限值。
pub fn levinson_durbin(ac: &[f32], order: usize) -> Option<FrameLpc> {
    if order == 0 || ac.len() < order + 1 {
        return None;
    }
    if !ac.iter().all(|c| c.is_finite()) || ac[0] <= 1.0e-12 {
        return None;
    }
    let mut a = vec![0.0_f32; order + 1]; // a[1..=order]，a[0] 恒为 1（隐含）
    let mut error = ac[0];
    for i in 1..=order {
        // 正规方程 Σ_k a_k·ac[i−k] = −ac[i] ⇒ 反射系数
        // k_i = −(ac[i] + Σ_j a_j·ac[i−j]) / E_{i−1}
        let mut acc = ac[i];
        for j in 1..i {
            acc += a[j] * ac[i - j];
        }
        if !acc.is_finite() || !error.is_finite() || error <= 1.0e-12 {
            return None;
        }
        let k = -acc / error;
        if !k.is_finite() || k.abs() >= 1.0 {
            return None;
        }
        // 就地更新需要旧值：a_new[j] = a[j] + k·a[i−j]
        let prev = a.clone();
        for j in 1..i {
            a[j] = prev[j] + k * prev[i - j];
        }
        a[i] = k;
        error *= 1.0 - k * k;
    }
    Some(FrameLpc {
        coeffs: a[1..=order].to_vec(),
        residual_ratio: (error / ac[0]).clamp(1.0e-6, 1.0),
    })
}

/// 由 LPC 系数（a[1..=p]）构造首一多项式的升幂系数（供 poly_roots 使用）。
///
/// A(z) = 1 + a1·z⁻¹ + … + ap·z⁻ᵖ 两边乘 zᵖ 得 A'(z) = zᵖ + a1·zᵖ⁻¹ + … + ap，
/// 其升幂系数为 [ap, …, a1, 1]。
pub fn coeffs_to_monic(a: &[f32]) -> Vec<f32> {
    let mut c: Vec<f32> = a.iter().rev().copied().collect();
    c.push(1.0);
    c
}

/// Durand-Kerner 求首一多项式的全部复根。
///
/// 参数：`coeffs` 升幂系数，最高次必须为 1（首一）。
/// 迭代内部使用 f64：12 阶 LPC 多项式在 f32 下会因根簇碰撞发散出非有限值。
/// 返回 `None`：多项式退化 / 迭代发散 / 出现非有限根。
pub fn poly_roots(coeffs: &[f32]) -> Option<Vec<Complex32>> {
    if coeffs.len() < 2 {
        return None;
    }
    let n = coeffs.len() - 1;
    if !coeffs.iter().all(|c| c.is_finite()) || (coeffs[n] - 1.0).abs() > 1.0e-6 {
        return None;
    }
    let c64: Vec<f64> = coeffs.iter().map(|&c| c as f64).collect();

    // 初值：(0.4 + 0.9i)^k，角度分散、模长 ~0.985^k
    let init = num_complex::Complex64::new(0.4, 0.9);
    let mut z: Vec<num_complex::Complex64> = Vec::with_capacity(n);
    let mut cur = num_complex::Complex64::new(1.0, 0.0);
    for _ in 0..n {
        z.push(cur);
        cur *= init;
    }

    for _ in 0..ROOT_ITERATIONS {
        let mut max_delta = 0.0_f64;
        let mut next = z.clone();
        for i in 0..n {
            let p = eval_poly_f64(&c64, z[i]);
            let mut denom = num_complex::Complex64::new(1.0, 0.0);
            for j in 0..n {
                if j != i {
                    denom *= z[i] - z[j];
                }
            }
            if denom.norm() < 1.0e-30 {
                continue;
            }
            let delta = p / denom;
            if !delta.is_finite() {
                return None;
            }
            next[i] = z[i] - delta;
            max_delta = max_delta.max(delta.norm());
        }
        z = next;
        if max_delta < ROOT_EPSILON as f64 {
            break;
        }
    }

    if z.iter().any(|c| !c.is_finite()) {
        return None;
    }
    Some(
        z.into_iter()
            .map(|c| Complex32::new(c.re as f32, c.im as f32))
            .collect(),
    )
}

/// 升幂系数的 Horner 求值（f64 内部）：p(z) = c0 + c1·z + … + cn·zⁿ。
fn eval_poly_f64(coeffs: &[f64], z: num_complex::Complex64) -> num_complex::Complex64 {
    let mut acc = num_complex::Complex64::new(*coeffs.last().unwrap_or(&0.0), 0.0);
    for &c in coeffs.iter().rev().skip(1) {
        acc = acc * z + num_complex::Complex64::new(c, 0.0);
    }
    acc
}

/// 把复根分类为共轭极点对（正虚部）与实根。
///
/// 规则：
/// - 丢弃 |z| >= 1（不稳定极点，出现在病态帧）。
/// - 虚部绝对值 < 1e-4 的根归入实根。
/// - pairs 按频率升序。
pub fn roots_to_poles(roots: &[Complex32], sr: f32) -> FramePoles {
    if sr <= 0.0 {
        return FramePoles::default();
    }
    let mut pairs: Vec<Pole> = Vec::new();
    let mut real_roots: Vec<f32> = Vec::new();
    for z in roots {
        if !z.is_finite() || z.norm() >= 1.0 {
            continue;
        }
        if z.im.abs() < REAL_ROOT_IMAG_EPS {
            real_roots.push(z.re);
            continue;
        }
        if z.im <= 0.0 {
            continue; // 只取正虚部侧，共轭对不重复计
        }
        let freq = z.arg() * sr / (2.0 * std::f32::consts::PI);
        let bandwidth = -z.norm().ln() * sr / std::f32::consts::PI;
        pairs.push(Pole {
            freq_hz: freq,
            bandwidth_hz: bandwidth,
            radius: z.norm(),
        });
    }
    pairs.sort_by(|a, b| a.freq_hz.partial_cmp(&b.freq_hz).unwrap_or(std::cmp::Ordering::Equal));
    FramePoles { pairs, real_roots }
}

/// 由（可能已迁移的）极点重建多项式系数。
///
/// 返回 z 的升幂首一多项式（常数项在前，最高次系数为 1），其根即极点 ——
/// 与 poly_roots 的输入约定一致；对其在单位圆 e^{-jw} 求值即得
/// A(e^{-jw}) 的幅频（差一个 zᵖ 相位因子，幅度不变）。
///
/// 规则：半径 clamp 到 0.9995 保证全极点滤波器稳定；带宽下限 1 Hz 防止 r=1。
pub fn poles_to_coeffs(pairs: &[Pole], real_roots: &[f32], sr: f32) -> Vec<f32> {
    if sr <= 0.0 {
        return vec![1.0];
    }
    let mut poly: Vec<f32> = vec![1.0];
    for pair in pairs {
        let bandwidth = pair.bandwidth_hz.max(1.0);
        let radius = (-(std::f32::consts::PI * bandwidth) / sr)
            .exp()
            .min(0.9995);
        let theta = 2.0 * std::f32::consts::PI * pair.freq_hz / sr;
        // 共轭对因子：(z − re^{iθ})(z − re^{−iθ}) = z² − 2r·cosθ·z + r²
        poly = poly_mul(&poly, &[radius * radius, -2.0 * radius * theta.cos(), 1.0]);
    }
    for &x in real_roots {
        let x = x.clamp(-0.9995, 0.9995);
        // 实根因子：(z − x)
        poly = poly_mul(&poly, &[-x, 1.0]);
    }
    poly
}

/// 升幂多项式乘法。
fn poly_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

/// 预加重系数（一阶高通，抵消声源频谱倾斜，改善共振峰极点定位精度）。
pub const PREEMPHASIS: f32 = 0.97;

/// 单帧完整分析：预加重 → Hann 加窗 → 自相关 → Levinson-Durbin。
///
/// 注意：返回的系数建模的是**预加重后**的信号；构造校正滤波器时
/// 调用方必须给分子分母同时补上 (1 − PREEMPHASIS·z⁻¹) 因子，
/// 保证 H = |A_orig/A_target| 的零点结构一致。
///
/// 返回 `None`：帧能量过低、数据不足或拟合失败。
pub fn analyze_frame(frame: &[f32], sr: f32, order: usize) -> Option<FrameLpc> {
    let _ = sr;
    if order < 2 || frame.len() < order * 2 {
        return None;
    }
    let window = hann_window(frame.len());
    let windowed: Vec<f32> = frame
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let pre = if i == 0 {
                s
            } else {
                s - PREEMPHASIS * frame[i - 1]
            };
            pre * window[i]
        })
        .collect();
    let ac = autocorrelation(&windowed, order)?;
    levinson_durbin(&ac, order)
}

/// Hann 窗。
pub fn hann_window(len: usize) -> Vec<f32> {
    if len <= 1 {
        return vec![1.0; len];
    }
    let denom = (len - 1) as f32;
    (0..len)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / denom).cos())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────
// 测试
// ─────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 11_025.0;

    /// 合成稳态元音：脉冲串（F0）经两个二阶共振器**并联**求和（Klatt 并联式）。
    /// 并联求和的谱峰位于各共振器极点附近（级联式会因乘积响应使峰位偏移）。
    fn synth_vowel(f0: f32, f1: f32, f2: f32, sr: f32, secs: f32) -> Vec<f32> {
        let n = (sr * secs) as usize;
        let period = (sr / f0).round() as usize;
        let r1 = (-(std::f32::consts::PI * 80.0) / sr).exp();
        let t1 = 2.0 * std::f32::consts::PI * f1 / sr;
        let r2 = (-(std::f32::consts::PI * 100.0) / sr).exp();
        let t2 = 2.0 * std::f32::consts::PI * f2 / sr;
        let mut y1 = [0.0_f32; 2];
        let mut y2 = [0.0_f32; 2];
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let x = if i % period == 0 { 1.0 } else { 0.0 };
            // 共振器 1
            let o1 = x + 2.0 * r1 * t1.cos() * y1[0] - r1 * r1 * y1[1];
            y1[1] = y1[0];
            y1[0] = o1;
            // 共振器 2（独立激励 = 并联）
            let o2 = x + 2.0 * r2 * t2.cos() * y2[0] - r2 * r2 * y2[1];
            y2[1] = y2[0];
            y2[0] = o2;
            out.push((o1 + o2) * 0.15);
        }
        out
    }

    #[test]
    fn poly_roots_recovers_known_polynomial() {
        // (z - 0.5)(z - (0.3+0.4i))(z - (0.3-0.4i)) = z^3 - 1.1 z^2 + 0.55 z - 0.125
        let coeffs = [-0.125_f32, 0.55, -1.1, 1.0];
        let roots = poly_roots(&coeffs).unwrap();
        assert_eq!(roots.len(), 3);
        let mut real_parts: Vec<f32> = roots.iter().map(|z| z.re).collect();
        real_parts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((real_parts[0] - 0.3).abs() < 1.0e-4, "got {real_parts:?}");
        assert!((real_parts[1] - 0.3).abs() < 1.0e-4, "got {real_parts:?}");
        assert!((real_parts[2] - 0.5).abs() < 1.0e-4, "got {real_parts:?}");
    }

    #[test]
    fn poly_roots_rejects_non_monic_or_degenerate() {
        assert!(poly_roots(&[1.0, 0.5]).is_none(), "最高次必须为 1");
        assert!(poly_roots(&[]).is_none());
    }

    /// 白噪声（LCG 确定性）激励两个并联共振器：教科书式 AR 模型，
    /// LPC 理论上应精确恢复极点。
    fn synth_ar(f1: f32, f2: f32, sr: f32, secs: f32) -> Vec<f32> {
        let n = (sr * secs) as usize;
        let r1 = (-(std::f32::consts::PI * 80.0) / sr).exp();
        let t1 = 2.0 * std::f32::consts::PI * f1 / sr;
        let r2 = (-(std::f32::consts::PI * 100.0) / sr).exp();
        let t2 = 2.0 * std::f32::consts::PI * f2 / sr;
        let mut y1 = [0.0_f32; 2];
        let mut y2 = [0.0_f32; 2];
        let mut seed = 0x12345678_u32;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            // LCG → [-1, 1) 白噪声
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let x = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
            let o1 = x + 2.0 * r1 * t1.cos() * y1[0] - r1 * r1 * y1[1];
            y1[1] = y1[0];
            y1[0] = o1;
            let o2 = x + 2.0 * r2 * t2.cos() * y2[0] - r2 * r2 * y2[1];
            y2[1] = y2[0];
            y2[0] = o2;
            out.push((o1 + o2) * 0.15);
        }
        out
    }

    #[test]
    fn levinson_durbin_recovers_ar_poles() {
        // 教科书式 AR 模型（白噪声激励）下，LPC 极点应落在真值 10% 以内。
        let signal = synth_ar(800.0, 1_200.0, SR, 0.3);
        let frame_len = (SR * 0.025) as usize;
        let start = signal.len() / 2;
        let lpc = analyze_frame(&signal[start..start + frame_len], SR, 12)
            .expect("AR frame must analyze");
        let poles = roots_to_poles(&poly_roots(&coeffs_to_monic(&lpc.coeffs)).unwrap(), SR);
        let nearest = |target: f32| -> f32 {
            poles
                .pairs
                .iter()
                .map(|p| p.freq_hz)
                .fold(f32::INFINITY, |best, f| {
                    if (f - target).abs() < (best - target).abs() {
                        f
                    } else {
                        best
                    }
                })
        };
        let got_f1 = nearest(800.0);
        let got_f2 = nearest(1_200.0);
        assert!((got_f1 - 800.0).abs() / 800.0 < 0.10, "F1 {got_f1:.1}");
        assert!((got_f2 - 1_200.0).abs() / 1_200.0 < 0.10, "F2 {got_f2:.1}");
    }

    #[test]
    fn levinson_durbin_recovers_synth_vowel_formants() {
        let f1_true = 800.0_f32;
        let f2_true = 1_200.0_f32;
        let signal = synth_vowel(150.0, f1_true, f2_true, SR, 0.3);
        let frame_len = (SR * 0.025) as usize;
        // 取信号中段一帧（避开起振暂态）
        let start = signal.len() / 2;
        let frame = &signal[start..start + frame_len];
        let lpc = analyze_frame(frame, SR, 12).expect("voiced frame must analyze");
        let poles = roots_to_poles(&poly_roots(&coeffs_to_monic(&lpc.coeffs)).unwrap(), SR);

        let nearest = |target: f32| -> f32 {
            poles
                .pairs
                .iter()
                .map(|p| p.freq_hz)
                .fold(f32::INFINITY, |best, f| {
                    if (f - target).abs() < (best - target).abs() {
                        f
                    } else {
                        best
                    }
                })
        };
        let got_f1 = nearest(f1_true);
        let got_f2 = nearest(f2_true);
        assert!(
            (got_f1 - f1_true).abs() / f1_true < 0.08,
            "F1 estimate {got_f1:.1} too far from {f1_true}"
        );
        assert!(
            (got_f2 - f2_true).abs() / f2_true < 0.08,
            "F2 estimate {got_f2:.1} too far from {f2_true}"
        );
    }

    #[test]
    fn residual_ratio_is_small_for_voiced_frame() {
        let signal = synth_vowel(150.0, 800.0, 1_200.0, SR, 0.3);
        let frame_len = (SR * 0.025) as usize;
        let start = signal.len() / 2;
        let lpc = analyze_frame(&signal[start..start + frame_len], SR, 12).unwrap();
        assert!(
            lpc.residual_ratio < 0.2,
            "全极点模型应拟合良好，residual_ratio={}",
            lpc.residual_ratio
        );
    }

    #[test]
    fn poles_to_coeffs_roundtrip() {
        let sr = SR;
        let pairs = [
            Pole {
                freq_hz: 800.0,
                bandwidth_hz: 80.0,
                radius: (-(std::f32::consts::PI * 80.0) / sr).exp(),
            },
            Pole {
                freq_hz: 1_200.0,
                bandwidth_hz: 120.0,
                radius: (-(std::f32::consts::PI * 120.0) / sr).exp(),
            },
        ];
        let coeffs = poles_to_coeffs(&pairs, &[], sr);
        assert_eq!(coeffs.len(), 5, "两对共轭极点 → 4 阶多项式");
        let poles = roots_to_poles(&poly_roots(&coeffs).unwrap(), sr);
        assert_eq!(poles.pairs.len(), 2);
        assert!((poles.pairs[0].freq_hz - 800.0).abs() < 1.0, "got {:?}", poles.pairs);
        assert!((poles.pairs[1].freq_hz - 1_200.0).abs() < 1.0, "got {:?}", poles.pairs);
        assert!((poles.pairs[0].bandwidth_hz - 80.0).abs() < 1.0);
        assert!((poles.pairs[1].bandwidth_hz - 120.0).abs() < 1.0);
    }

    #[test]
    fn roots_to_poles_drops_unstable_roots() {
        // 根 1.2（不稳定）与 0.5∠1rad ± 共轭（稳定共轭对）
        let angle = 1.0_f32;
        let radius = 0.5_f32;
        let roots = vec![
            Complex32::new(1.2, 0.0),
            Complex32::new(radius * angle.cos(), radius * angle.sin()),
            Complex32::new(radius * angle.cos(), -radius * angle.sin()),
        ];
        let poles = roots_to_poles(&roots, SR);
        assert_eq!(poles.pairs.len(), 1, "不稳定根必须被丢弃");
        assert!(poles.real_roots.is_empty());
        assert!((poles.pairs[0].freq_hz - SR * angle / (2.0 * std::f32::consts::PI)).abs() < 1.0);
        assert!((poles.pairs[0].radius - radius).abs() < 1.0e-5);
    }

    #[test]
    fn analyze_frame_returns_none_on_silence() {
        let frame = vec![0.0_f32; 256];
        assert!(analyze_frame(&frame, SR, 12).is_none());
        let nan_frame = vec![f32::NAN; 256];
        assert!(analyze_frame(&nan_frame, SR, 12).is_none());
    }

    #[test]
    fn poles_to_coeffs_clamps_unstable_radius() {
        // 带宽极小 → 半径可能 >= 1，必须被 clamp 到稳定域
        let pairs = [Pole {
            freq_hz: 500.0,
            bandwidth_hz: 0.0,
            radius: 1.0,
        }];
        let coeffs = poles_to_coeffs(&pairs, &[], SR);
        let roots = poly_roots(&coeffs).unwrap();
        for z in roots {
            assert!(z.norm() < 1.0, "重建极点必须稳定，|z|={}", z.norm());
        }
    }
}
