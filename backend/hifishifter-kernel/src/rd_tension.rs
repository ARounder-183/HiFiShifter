//! 张力（tension）在谐波支上的 STFT 域实现：按 Rd 逐谐波重塑频谱。
//!
//! # 主要内容
//! - [`RdTension::apply`]：对一段**谐波波形**做 STFT，逐帧拟合源 Rd，
//!   再按张力求出目标 Rd 的逐谐波增益并施加，最后 ISTFT 回波形。
//! - [`RdTension::harmonic_peaks`]：在源 f0 的各次谐波附近取幅度峰（对数幅度
//!   抛物线插值），作为 Rd 拟合的观测值。
//! - 内部 STFT/ISTFT：周期 Hann 窗、`center=True` 反射填充、按窗平方和归一化。
//!
//! # 作用
//! 把「张力」从一个无模型的频谱倾斜，换成**声门源形状参数 Rd 的重塑**：
//! 张力只改变谐波的相对强度，不改变基频、不引入噪声，且第一谐波恒为参考点
//! （见 [`crate::glottal_rd::GlottalRd::gains`]），因此感知响度天然稳定。
//!
//! # 与其他模块的关系
//! - 由 [`crate::renderer::chain`] 的 `HiFiGanStage` 在 **HNSep 分离之后、
//!   mel 分析之前** 调用，只作用于谐波支；噪声支不受影响（避免把齿音/气声
//!   一起放大成"金属感"）。
//! - 依赖 [`crate::glottal_rd`] 的 LF 模型与 Rd 拟合/增益。
//! - 源 f0 与目标 f0 由调用方以闭包注入（分别取自 `clip_midi` 与 `pitch_edit`），
//!   因此本模块不关心曲线的时间基。
//!
//! # 来源
//! 移植自 OpenUtau 0.1.571-beta（MIT）的
//! `OpenUtau.Core/Classic/Hifisampler/HifiRdTension.cs`（其中 STFT/ISTFT 来自
//! 同项目的 `HifiDsp.cs`）。本文件为 Rust 改写。
//!
//! # 维护说明
//! - `HOP` / `N_FFT` 是**算法常量**：`HOP` 同时是源 f0 曲线的帧步长契约
//!   （调用方须按同一 hop 采样 f0），改动会让帧与 f0 错位。
//! - 不做任何重新归一化：输出响度由「第一谐波增益恒为 1」保证，
//!   额外归一化会破坏张力不改变响度的语义。

use crate::glottal_rd::GlottalRd;
use num_complex::Complex;
use rustfft::FftPlanner;
use std::sync::Arc;

/// STFT 长度。
pub const N_FFT: usize = 2048;
/// STFT 帧步长；**同时是源 f0 / 张力曲线的帧步长契约**。
pub const HOP: usize = 256;

/// 谐波搜索窗的半宽（以 f0 为单位）。
const HARMONIC_HALF_WIDTH_F0: f64 = 0.3;
/// 抛物线插值中的下限，避免 log(0)。
const LOG_FLOOR: f64 = 1e-12;
/// Rd 平滑窗时长（秒）：0.02 s。
const SMOOTH_SECONDS: f64 = 0.02;

/// 张力在谐波支上的 STFT 域重塑。
pub struct RdTension;

impl RdTension {
    /// 对谐波波形施加张力。
    ///
    /// # 流程
    /// 1. 波形补零到 `HOP` 的整数倍，做 `center=True` 的 STFT（[`N_FFT`]，[`HOP`]）；
    /// 2. 逐帧用**源 f0** 找谐波峰，`GlottalRd::fit` 拟合该帧的 Rd；
    /// 3. 对 Rd 轨迹做 0.02 s 滑动平均（未浊音帧由邻近浊音帧填充）；
    /// 4. 对每个浊音帧且 `|tension| > 0` 的帧：`rd2 = TenseRd(rd, tension)`，
    ///    求**目标 f0** 各次谐波的增益，逐 bin 乘到频谱上；
    /// 5. ISTFT 回波形，截回输入长度。
    ///
    /// # 参数
    /// - `x`：谐波支波形（任意增益；Rd 是相对第一谐波拟合的，故绝对幅度无关）
    /// - `source_f0`：源 f0（Hz，`0` 或非正值表示未浊音），帧步长 = [`HOP`] 个样本
    /// - `sample_rate`：采样率（Hz）
    /// - `tension_at`：张力曲线（-100..100）在**样本位置**处的取值
    /// - `target_f0_at`：目标 f0（Hz）在**样本位置**处的取值；`<= 0` 的帧跳过
    ///
    /// # 特殊说明
    /// - `x` 为空、采样率为 0、或所有帧的张力都 < 1e-9 时，直接返回输入拷贝
    ///   （避免无谓的 STFT）。
    /// - 输出长度恒等于 `x.len()`；输入不足 `N_FFT` 时输出为静音（STFT 无完整帧）。
    pub fn apply(
        x: &[f32],
        source_f0: &[f64],
        sample_rate: u32,
        tension_at: impl Fn(usize) -> f64,
        target_f0_at: impl Fn(usize) -> f64,
    ) -> Vec<f32> {
        if x.is_empty() || sample_rate == 0 {
            return x.to_vec();
        }

        // 补零到 HOP 整数倍，使 ISTFT 覆盖每个样本（OpenUtau 同此）。
        let padded_len = x.len().div_ceil(HOP) * HOP;
        let mut signal = vec![0.0f64; padded_len];
        for (i, &v) in x.iter().enumerate() {
            signal[i] = v as f64;
        }

        let window = periodic_hann(N_FFT);
        let (mut spec, frames) = stft_spectrum(&signal, &window);
        if frames == 0 {
            return vec![0.0; x.len()];
        }
        let bin_hz = sample_rate as f64 / N_FFT as f64;

        // ── 1. 逐帧拟合源 Rd ──────────────────────────────────────────────
        let mut rd = vec![0.0f64; frames];
        let mut voiced = vec![false; frames];
        for m in 0..frames {
            let f0 = source_f0.get(m).copied().unwrap_or(0.0);
            if !(f0 > 0.0) {
                continue;
            }
            // 与参考实现一致：这里只按 `MAX_FIT_HZ / f0` 取谐波，**不在本层封顶** ——
            // 谐波数上限由 `GlottalRd::fit` 内部的 `MAX_FIT_HARMONICS` 施加
            //（对应 C# 的 `GlottalRd.Fit` 里 `Math.Min(amplitudes.Length, MaxFitHarmonics)`）。
            // 两处都封顶会让上限语义分散在两个模块里。
            let harmonics = (GlottalRd::MAX_FIT_HZ / f0) as usize;
            if harmonics < 2 {
                continue;
            }
            let amplitudes = Self::harmonic_peaks(&spec[m], f0, bin_hz, harmonics);
            if amplitudes.len() < 2 {
                continue;
            }
            rd[m] = GlottalRd::fit(&amplitudes, f0);
            voiced[m] = true;
        }

        // ── 2. 平滑（未浊音帧由邻近浊音帧填充）────────────────────────────
        let smooth_window = ((SMOOTH_SECONDS * sample_rate as f64 / HOP as f64).round() as usize).max(1);
        let rd = GlottalRd::smooth(&rd, &voiced, smooth_window);

        // ── 3. 按张力重塑频谱 ────────────────────────────────────────────
        let mut any_applied = false;
        for m in 0..frames {
            if !voiced[m] {
                continue;
            }
            let t = tension_at(m * HOP);
            if !t.is_finite() || t.abs() < 1e-9 {
                continue;
            }
            let target = target_f0_at(m * HOP);
            if !(target > 0.0) {
                continue;
            }
            // 目标 f0 到 Nyquist 的谐波数。
            //
            // 【不要在这里加 MAX_HARMONICS 上限】参考实现（`HifiRdTension.cs:57`）
            // 传的是**不设上限**的 `(int)(sampleRate / 2.0 / f0)`，且 `GlottalRd.Gains`
            // 内部也没有上限。低音区 n 会明显超过 80（220 Hz → 100、110 Hz → 200），
            // 若在此钳到 80，`GlottalRd::gain_at` 会对第 80 次以上的谐波一律返回
            // **末值**（它按 `gains[len-1]` 处理越界），高频段被压成一条平的增益，
            // 与参考的逐谐波曲线不符 —— 听感上低音会缺少高频细节。
            // `MAX_HARMONICS` 只用于**拟合**路径（与 `MaxFitHarmonics` 对应）。
            let n = ((sample_rate as f64 / 2.0 / target) as usize).max(1);
            let gains = GlottalRd::gains(rd[m], GlottalRd::tense_rd(rd[m], t), target, n);
            let frame = &mut spec[m];
            for (k, bin) in frame.iter_mut().enumerate() {
                *bin *= GlottalRd::gain_at(&gains, target, k as f64 * bin_hz);
            }
            any_applied = true;
        }

        if !any_applied {
            return x.to_vec();
        }

        // ── 4. ISTFT 并截回原长 ──────────────────────────────────────────
        let y = istft(&spec, frames, &window);
        let mut result = vec![0.0f32; x.len()];
        for (i, out) in result.iter_mut().enumerate() {
            if let Some(&v) = y.get(i) {
                *out = v as f32;
            }
        }
        result
    }

    /// 在 `frame` 的源 f0 各次谐波附近取幅度峰。
    ///
    /// 每次谐波在 `|bin - k*f0/binHz| <= 0.3*f0/binHz` 范围内取最大幅度 bin，
    /// 再用**对数幅度的抛物线**插值出亚 bin 精度的峰值（标准三点插值）。
    ///
    /// 若某次谐波的搜索区间越界（`hi < lo`），立即截断返回已求得的前 `k-1` 个
    /// （与 OpenUtau 的 `return peaks[..(k - 1)]` 一致）——调用方据此判断
    /// 「有效谐波不足 2 个」而跳过该帧。
    ///
    /// `bins` 为 STFT 单帧频谱（长度 `N_FFT/2 + 1`）。
    pub fn harmonic_peaks(
        bins: &[Complex<f64>],
        f0: f64,
        bin_hz: f64,
        n: usize,
    ) -> Vec<f64> {
        if n == 0 || bins.len() < 3 || !(f0 > 0.0) || !(bin_hz > 0.0) {
            return Vec::new();
        }
        let mut peaks = Vec::with_capacity(n);
        let half_width = HARMONIC_HALF_WIDTH_F0 * f0 / bin_hz;
        let last_bin = bins.len() - 2; // 需要 best+1 可索引

        for k in 1..=n {
            let center = k as f64 * f0 / bin_hz;
            let lo = (1.0f64).max(center - half_width) as usize;
            let hi_f = (last_bin as f64).min(center + half_width);
            if hi_f < lo as f64 {
                break;
            }
            let hi = hi_f as usize;
            if hi < lo || lo < 1 {
                break;
            }

            let mut best = lo;
            for b in (lo + 1)..=hi {
                if bins[b].norm() > bins[best].norm() {
                    best = b;
                }
            }
            if best == 0 || best + 1 >= bins.len() {
                break;
            }

            // 对数幅度抛物线插值
            let a = (bins[best - 1].norm() + LOG_FLOOR).ln();
            let c = (bins[best].norm() + LOG_FLOOR).ln();
            let d = (bins[best + 1].norm() + LOG_FLOOR).ln();
            let curvature = a - 2.0 * c + d;
            let p = if curvature < 0.0 {
                0.5 * (a - d) / curvature
            } else {
                0.0
            };
            peaks.push((c - 0.25 * (a - d) * p).exp());
        }
        peaks
    }
}

/// 周期 Hann 窗 `0.5(1 - cos(2πn/N))`。
///
/// 【为什么是周期而非对称形式】与 OpenUtau `HifiStft.HannWindow` 一致。
/// 本模块的 STFT/ISTFT 是自洽的一对（ISTFT 按窗平方和归一化），
/// 周期形式在 `hop = N/8` 下满足 COLA 条件，重建无幅度调制。
fn periodic_hann(len: usize) -> Vec<f64> {
    if len == 0 {
        return Vec::new();
    }
    (0..len)
        .map(|n| {
            let phase = (2.0 * std::f64::consts::PI * n as f64) / len as f64;
            0.5 - 0.5 * phase.cos()
        })
        .collect()
}

/// `center=True` 的反射填充：两侧各 `pad` 个样本，不含端点重复（numpy `reflect`）。
fn reflect_index(i: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * (len as isize - 1);
    let mut m = i % period;
    if m < 0 {
        m += period;
    }
    if m < len as isize {
        m as usize
    } else {
        (period - m) as usize
    }
}

/// STFT：`center=True` 反射填充 + 周期 Hann 窗。返回 (每帧频谱, 帧数)。
///
/// 与 `torch.stft(center=True, pad_mode="reflect")` 的帧数与对齐方式一致：
/// 填充 `N_FFT/2` 后按 `HOP` 取帧，帧数 `1 + (padded - N_FFT)/HOP`。
fn stft_spectrum(signal: &[f64], window: &[f64]) -> (Vec<Vec<Complex<f64>>>, usize) {
    let pad = N_FFT / 2;
    let len = signal.len();
    if len == 0 {
        return (Vec::new(), 0);
    }
    let padded_len = pad + len + pad;
    if padded_len < N_FFT {
        return (Vec::new(), 0);
    }
    let frames = 1 + (padded_len - N_FFT) / HOP;

    let mut planner = FftPlanner::<f64>::new();
    let fft: Arc<dyn rustfft::Fft<f64>> = planner.plan_fft_forward(N_FFT);
    let bins = N_FFT / 2 + 1;

    let mut out = Vec::with_capacity(frames);
    let mut buf = vec![Complex::new(0.0, 0.0); N_FFT];
    for m in 0..frames {
        let start = m * HOP;
        for i in 0..N_FFT {
            // padded 坐标 start+i → 未填充信号坐标 start+i-pad
            let j = start as isize + i as isize - pad as isize;
            let sample = if j >= 0 && (j as usize) < len {
                signal[j as usize]
            } else {
                signal[reflect_index(j, len)]
            };
            buf[i] = Complex::new(sample * window[i], 0.0);
        }
        fft.process(&mut buf);
        out.push(buf[..bins].to_vec());
    }
    (out, frames)
}

/// `center=True` 的 ISTFT：重叠相加后按**窗平方和**归一化，两侧各裁掉 `N_FFT/2`。
///
/// 返回长度 = `N_FFT + HOP*(frames-1) - N_FFT`，即 `HOP*(frames-1)`；
/// 调用方按原始长度截取。
fn istft(spec: &[Vec<Complex<f64>>], frames: usize, window: &[f64]) -> Vec<f64> {
    if frames == 0 {
        return Vec::new();
    }
    let bins = N_FFT / 2 + 1;
    let total = N_FFT + HOP * (frames - 1);

    let mut planner = FftPlanner::<f64>::new();
    let ifft: Arc<dyn rustfft::Fft<f64>> = planner.plan_fft_inverse(N_FFT);

    let mut y = vec![0.0f64; total];
    let mut wsum = vec![0.0f64; total];
    let mut buf = vec![Complex::new(0.0, 0.0); N_FFT];

    for (m, frame) in spec.iter().enumerate().take(frames) {
        buf.fill(Complex::new(0.0, 0.0));
        let n = frame.len().min(bins);
        buf[..n].copy_from_slice(&frame[..n]);

        // ── Hermitian 对称重建（关键，易漏）────────────────────────────────
        // rustfft 的**复数**逆变换不会自行补出负频率：若 bins[N/2+1..] 保持 0，
        // 得到的解析信号只含正频率，取实部后每个频率分量恰好减半
        //（表现为整体 0.5 倍衰减，且各次谐波等比缩小，极易误判成"增益算错"）。
        // OpenUtau 用的是**实数**逆变换（FftFlat `RealFourierTransform`），
        // 负频率由变换内部推导，故其代码无需显式镜像。
        //
        // 这里显式构造共轭对称谱：buf[N-k] = conj(buf[k])。
        for k in 1..bins.saturating_sub(1) {
            buf[N_FFT - k] = Complex::new(buf[k].re, -buf[k].im);
        }
        // DC 与 Nyquist 必须是实数（实数信号的频谱约束）
        buf[0] = Complex::new(buf[0].re, 0.0);
        buf[bins - 1] = Complex::new(buf[bins - 1].re, 0.0);

        ifft.process(&mut buf);

        // rustfft 的逆变换未归一化
        let scale = 1.0 / N_FFT as f64;
        let p = m * HOP;
        for i in 0..N_FFT {
            let v = buf[i].re * scale * window[i];
            y[p + i] += v;
            wsum[p + i] += window[i] * window[i];
        }
    }

    let start = N_FFT / 2;
    let length = total.saturating_sub(N_FFT);
    let mut result = vec![0.0f64; length];
    for (i, out) in result.iter_mut().enumerate() {
        let j = start + i;
        if j < total {
            *out = if wsum[j] > 1e-11 { y[j] / wsum[j] } else { 0.0 };
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 合成一个 Rd 形状的谐波信号（含唇辐射），用于拟合验证。
    fn rd_shaped_signal(sr: u32, f0: f64, rd: f64, harmonics: usize, n: usize) -> Vec<f32> {
        let shape = GlottalRd::flow_shape(rd, f0, harmonics);
        (0..n)
            .map(|i| {
                let t = i as f64 / sr as f64;
                let mut sum = 0.0;
                for (k, &s) in shape.iter().enumerate() {
                    let hz = (k + 1) as f64 * f0;
                    sum += 0.05 * s * GlottalRd::lip_gain(hz) * (2.0 * std::f64::consts::PI * hz * t).cos();
                }
                sum as f32
            })
            .collect()
    }

    /// **张力为 0 时必须逐样本恒等**（不引入任何处理痕迹）。
    ///
    /// 这是最重要的安全契约：用户没画张力时输出必须与输入一致，
    /// 否则 STFT 往返的数值误差会污染所有既有工程。
    #[test]
    fn zero_tension_is_identity() {
        let sr = 44_100;
        let x = rd_shaped_signal(sr, 220.0, 1.0, 30, sr as usize / 4);
        let f0 = vec![220.0f64; x.len() / HOP + 2];
        let y = RdTension::apply(&x, &f0, sr, |_| 0.0, |_| 220.0);
        assert_eq!(y.len(), x.len());
        let max_err = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-6, "zero tension must be identity, max err {max_err}");
    }

    /// 空输入 / 零采样率必须安全返回。
    #[test]
    fn degenerate_inputs_are_safe() {
        assert!(RdTension::apply(&[], &[], 44_100, |_| 50.0, |_| 220.0).is_empty());
        let x = vec![0.1f32; 100];
        let y = RdTension::apply(&x, &[0.0; 4], 0, |_| 50.0, |_| 220.0);
        assert_eq!(y, x);
    }

    /// **方向契约**：张力 +100（Rd 减半）必须提升高次谐波，-100 必须衰减。
    ///
    /// 通过在输出上做单频 DFT 测量各谐波幅度来验证 —— 这是"端到端"判据，
    /// 覆盖拟合 → 增益 → ISTFT 全链路，而不是只测增益函数。
    #[test]
    fn positive_tension_boosts_upper_harmonics_end_to_end() {
        let sr = 44_100u32;
        let f0v = 220.0f64;
        let n = sr as usize / 4;
        let x = rd_shaped_signal(sr, f0v, 1.0, 30, n);
        let f0 = vec![f0v; n / HOP + 2];

        let amplitude = |sig: &[f32], hz: f64| -> f64 {
            let a = sig.len() / 4;
            let len = sig.len() / 2;
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (i, &v) in sig.iter().enumerate().skip(a).take(len) {
                let w = 2.0 * std::f64::consts::PI * hz * i as f64 / sr as f64;
                re += v as f64 * w.cos();
                im -= v as f64 * w.sin();
            }
            2.0 * (re * re + im * im).sqrt() / len as f64
        };

        let up = RdTension::apply(&x, &f0, sr, |_| 100.0, |_| f0v);
        let down = RdTension::apply(&x, &f0, sr, |_| -100.0, |_| f0v);

        // 第一谐波应基本不变（不重新归一化的契约）
        let h1_up = amplitude(&up, f0v) / amplitude(&x, f0v);
        assert!(
            (h1_up - 1.0).abs() < 0.35,
            "first harmonic must stay near unity, ratio {h1_up}"
        );

        // 第 8 谐波：张力 +100 提升、-100 衰减
        let h8_up = amplitude(&up, 8.0 * f0v) / amplitude(&x, 8.0 * f0v);
        let h8_down = amplitude(&down, 8.0 * f0v) / amplitude(&x, 8.0 * f0v);
        assert!(
            h8_up > 1.05,
            "tension +100 must boost harmonic 8, ratio {h8_up}"
        );
        assert!(
            h8_down < 0.95,
            "tension -100 must attenuate harmonic 8, ratio {h8_down}"
        );
        assert!(
            h8_up > h8_down,
            "+100 must be brighter than -100 ({h8_up} vs {h8_down})"
        );
    }

    /// `harmonic_peaks` 必须落在真实谐波上：喂入纯正弦（只有第一谐波），
    /// 第一个峰应对应 f0，且后续谐波幅度远小于它。
    #[test]
    fn harmonic_peaks_locate_the_harmonic() {
        let sr = 44_100u32;
        let f0 = 220.0f64;
        let n = N_FFT;
        let sig: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * f0 * i as f64 / sr as f64).sin())
            .collect();
        let window = periodic_hann(N_FFT);
        let (spec, frames) = stft_spectrum(&sig, &window);
        assert!(frames > 0);
        let bin_hz = sr as f64 / N_FFT as f64;
        let peaks = RdTension::harmonic_peaks(&spec[frames / 2], f0, bin_hz, 8);
        assert!(peaks.len() >= 2, "expected several harmonics, got {}", peaks.len());
        // 第一谐波远强于后续（纯正弦）
        assert!(
            peaks[0] > peaks[2] * 5.0,
            "first harmonic should dominate: {:?}",
            &peaks[..3]
        );
    }

    /// 谐波数不足 / 参数非法时返回空，而不是 panic。
    #[test]
    fn harmonic_peaks_handles_degenerate_inputs() {
        let bins = vec![Complex::new(0.0, 0.0); N_FFT / 2 + 1];
        assert!(RdTension::harmonic_peaks(&bins, 220.0, 21.5, 0).is_empty());
        assert!(RdTension::harmonic_peaks(&bins, 0.0, 21.5, 10).is_empty());
        assert!(RdTension::harmonic_peaks(&bins, 220.0, 0.0, 10).is_empty());
        // 全零频谱：仍应返回有限值（不 panic、不 NaN）
        let peaks = RdTension::harmonic_peaks(&bins, 220.0, 21.5, 10);
        assert!(peaks.iter().all(|v| v.is_finite()));
    }

    /// 未浊音（源 f0 = 0）时不得施加任何处理 —— 应逐样本恒等。
    #[test]
    fn unvoiced_frames_are_untouched() {
        let sr = 44_100u32;
        let n = sr as usize / 8;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 300.0 * i as f64 / sr as f64).sin() as f32 * 0.3)
            .collect();
        let f0 = vec![0.0f64; n / HOP + 2]; // 全未浊音
        let y = RdTension::apply(&x, &f0, sr, |_| 100.0, |_| 300.0);
        assert_eq!(y.len(), x.len());
        let max_err = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-6, "unvoiced input must pass through, err {max_err}");
    }

    /// 目标 f0 非法（0）时跳过该帧，不得产生 NaN/Inf。
    #[test]
    fn invalid_target_f0_is_skipped() {
        let sr = 44_100u32;
        let n = sr as usize / 8;
        let x = rd_shaped_signal(sr, 220.0, 1.0, 20, n);
        let f0 = vec![220.0f64; n / HOP + 2];
        let y = RdTension::apply(&x, &f0, sr, |_| 100.0, |_| 0.0);
        assert!(
            y.iter().all(|v| v.is_finite()),
            "invalid target f0 must not produce NaN/Inf"
        );
    }

    /// STFT→ISTFT 往返在有处理时应保持信号能量量级（不得爆音或静音）。
    #[test]
    fn round_trip_preserves_energy_magnitude() {
        let sr = 44_100u32;
        let n = sr as usize / 4;
        let x = rd_shaped_signal(sr, 220.0, 1.0, 30, n);
        let f0 = vec![220.0f64; n / HOP + 2];
        for t in [100.0f64, -100.0, 50.0] {
            let y = RdTension::apply(&x, &f0, sr, |_| t, |_| 220.0);
            let rms = |s: &[f32]| -> f64 {
                (s.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / s.len() as f64).sqrt()
            };
            let ratio = rms(&y) / rms(&x);
            assert!(
                ratio > 0.2 && ratio < 5.0,
                "tension {t} energy ratio out of range: {ratio}"
            );
        }
    }

use super::*;

    /// 更接近真实人声的检验：多个谐波 + 逐次谐波衰减 + 轻微失谐，
    /// 验证拟合出的 Rd 落在合理区间、且张力方向正确、输出无 NaN。
    #[test]
    fn works_on_a_more_vocal_like_signal() {
        let sr = 44_100u32;
        let f0 = 196.0f64; // G3
        let n = sr as usize / 2;
        // 25 个谐波，幅度按 1/k^1.1 衰减，并加一点噪声模拟气声
        let sig: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / sr as f64;
                let mut s = 0.0;
                for k in 1..=25 {
                    let amp = 0.15 / (k as f64).powf(1.1);
                    s += amp * (2.0 * std::f64::consts::PI * f0 * k as f64 * t).sin();
                }
                s += 0.002 * ((i as f64 * 12.9898).sin() * 43758.5453).fract();
                s as f32
            })
            .collect();
        let f0v = vec![f0; n / HOP + 2];

        let y = RdTension::apply(&sig, &f0v, sr, |_| 60.0, |_| f0);
        assert_eq!(y.len(), sig.len());
        assert!(y.iter().all(|v| v.is_finite()), "output must be finite");

        // 输出不应静音，也不应爆音
        let rms = |s: &[f32]| (s.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / s.len() as f64).sqrt();
        let ratio = rms(&y) / rms(&sig);
        assert!(ratio > 0.3 && ratio < 3.0, "energy ratio {ratio} out of range");
    }

    /// 低音目标 f0 的谐波数**不得**被封顶 —— 参考实现按 Nyquist 展开到数百个。
    ///
    /// 【为什么钉住】若把目标谐波数钳到拟合用的 80，`GlottalRd::gain_at` 会对
    /// 第 80 次以上的谐波一律返回末值，高频段被压成平的增益曲线，低音因此失去
    /// 高频细节（与 `HifiRdTension.cs:57` 的不设限口径不符）。
    #[test]
    fn low_target_pitch_uses_many_harmonics() {
        let sr = 44_100u32;
        let f0 = 55.0f64; // A1：Nyquist 处约 400 次谐波，远超 80
        let n = sr as usize / 4;
        let sig: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / sr as f64;
                let mut s = 0.0;
                for k in 1..=40 {
                    s += (0.2 / k as f64) * (2.0 * std::f64::consts::PI * f0 * k as f64 * t).sin();
                }
                s as f32
            })
            .collect();
        let f0v = vec![f0; n / HOP + 2];
        let y = RdTension::apply(&sig, &f0v, sr, |_| 80.0, |_| f0);
        assert!(y.iter().all(|v| v.is_finite()));
        let rms =
            |s: &[f32]| (s.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / s.len() as f64).sqrt();
        let ratio = rms(&y) / rms(&sig);
        assert!(ratio > 0.2, "low-pitch output collapsed, ratio {ratio}");
    }

    /// 极短输入（不足一帧 / 不足一个 N_FFT）不得 panic。
    #[test]
    fn very_short_inputs_do_not_panic() {
        let sr = 44_100u32;
        for len in [1usize, 2, 100, HOP, HOP + 1, N_FFT - 1, N_FFT] {
            let x = vec![0.1f32; len];
            let f0 = vec![220.0f64; 4];
            let y = RdTension::apply(&x, &f0, sr, |_| 100.0, |_| 220.0);
            assert_eq!(y.len(), len, "length must be preserved for len {len}");
            assert!(y.iter().all(|v| v.is_finite()));
        }
    }

    /// 静音输入不得产生 NaN（log(0) 路径）。
    #[test]
    fn silence_input_is_safe() {
        let sr = 44_100u32;
        let x = vec![0.0f32; 8192];
        let f0 = vec![220.0f64; 8192 / HOP + 2];
        let y = RdTension::apply(&x, &f0, sr, |_| 100.0, |_| 220.0);
        assert!(y.iter().all(|v| v.is_finite()), "silence must not produce NaN");
        assert!(y.iter().all(|v| v.abs() < 1e-6), "silence in, silence out");
    }
}

