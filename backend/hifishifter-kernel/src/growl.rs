//! 怒音（growl）的波形域实现：移植自 OpenUtau Hifisampler 的 GRWC 算法。
//!
//! # 来源
//! OpenUtau 0.1.571-beta（MIT）`OpenUtau.Core/Classic/Hifisampler/HifiGrowl.cs`
//! 与 [openhachimi/hifisampler](https://github.com/openhachimi/hifisampler)
//! `util/growl.py`（Apache-2.0）：两者是同一算法，OpenUtau 是上游 Python 参考实现的 C# 翻译。
//! 本文件是 Rust 改写，保留公式与常量，不依赖 scipy。
//!
//! # 算法
//! 1. 4 阶 Butterworth 高通把信号拆成 400 Hz 以上频带 `band` 与其余部分；
//! 2. 对 `band` 施加 80 Hz 方波 LFO 驱动的音高调制（±100 cents × strength）；
//! 3. 累积相位漂移先高通 20 Hz（去掉整体时移），再做线性插值重采样；
//! 4. `band` 做 RMS 回正，与低频部分相加。
//!
//! 作用域刻意只在 400 Hz 以上：基频与低次谐波不动，才能保留咬字与音高。
//!
//! # 与调用方的关系
//! 由 [`crate::renderer::chain::HiFiGanStage`] 在 HNSEP 分离后、只在谐波支上调用。
//! 曲线为 0 或缺失时由调用方跳过，本模块不做整体旁路判断。
//!
//! # 维护说明
//! - 400 Hz / 80 Hz / 100 cents 是**算法常量**，与 hifisampler 参考实现一致；
//!   改它们就是另一套音色，需重新做试听对比。
//! - 输出长度恒等于输入；全零/非有限输入返回等长的安全结果。

use num_complex::Complex;

/// GRWC 的默认 LFO 频率（Hz）。
pub const GRWC_LFO_HZ: f64 = 80.0;
/// GRWC 的默认高通截止频率（Hz）：只调制该频率以上的频带。
pub const GRWC_HIGHPASS_HZ: f64 = 400.0;
/// GRWC 的最大调制深度（cents）。
pub const GRWC_MAX_CENTS: f64 = 100.0;

/// 对整段波形施加 GRWC 怒音。
///
/// `strength` 取 `0..=1`，与 hifisampler 的 `HG / 100` 同口径。
/// 超出范围的输入按闭区间钳制；非有限值按 0 处理。
pub fn apply_growl_mono(input: &[f32], sample_rate: u32, strength: f64) -> Vec<f32> {
    if input.is_empty() || sample_rate == 0 {
        return input.to_vec();
    }
    let strength = if strength.is_finite() {
        strength.clamp(0.0, 1.0)
    } else {
        0.0
    };
    if strength <= 0.0 {
        return input.to_vec();
    }

    let mut x = Vec::with_capacity(input.len());
    x.extend(
        input
            .iter()
            .map(|&v| if v.is_finite() { f64::from(v) } else { 0.0 }),
    );

    let (band, complement) = highpass_and_complement(&x, sample_rate, GRWC_HIGHPASS_HZ);
    let lfo = square_lfo(x.len(), sample_rate, GRWC_LFO_HZ);
    let modulated = modulate_pitch(&band, sample_rate, &lfo, strength);
    let rms_orig = rms(&band);
    let rms_new = rms(&modulated);
    let gain = if rms_new > 1.0e-10 {
        rms_orig / rms_new
    } else {
        1.0
    };

    x.iter()
        .zip(complement.iter())
        .zip(modulated.iter())
        .map(|((_, &c), &m)| (c + m * gain) as f32)
        .collect()
}

/// 按逐样本强度包络施加 GRWC，输出长度恒等于输入。
///
/// 与逐块调用 [`apply_growl_mono`] 的区别：整段只做一次高通、一次 LFO 与一次
/// 插值，包络只在 `dry` 与 `wet` 之间混合。逐块调用会在每块重置滤波器与插值
/// 状态，50 ms 周期的接缝会变成可听的粗糙颤音，而且强度曲线陡变时更容易爆音。
///
/// `strength_at(i)` 返回 `0..=1`；非有限值按 0 处理。曲线全 0 时直接返回输入拷贝。
pub fn apply_growl_envelope(
    input: &[f32],
    sample_rate: u32,
    strength_at: impl Fn(usize) -> f64,
) -> Vec<f32> {
    if input.is_empty() || sample_rate == 0 {
        return input.to_vec();
    }
    let strengths: Vec<f64> = (0..input.len())
        .map(|i| {
            let v = strength_at(i);
            if v.is_finite() {
                v.clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect();
    if !strengths.iter().any(|v| *v > 0.0) {
        return input.to_vec();
    }
    let dry: Vec<f32> = input
        .iter()
        .map(|&v| if v.is_finite() { v } else { 0.0 })
        .collect();
    let wet = apply_growl_mono(&dry, sample_rate, 1.0);
    dry.iter()
        .zip(wet.iter())
        .zip(strengths.iter())
        .map(|((&a, &b), &s)| {
            let s = s as f32;
            a * (1.0 - s) + b * s
        })
        .collect()
}

/// 4 阶 Butterworth 高通（scipy `butter(4, wn, "high", output="sos")` 等价），
/// 返回 `(highpass, input - highpass)`。
fn highpass_and_complement(x: &[f64], sample_rate: u32, cutoff_hz: f64) -> (Vec<f64>, Vec<f64>) {
    let nyq = sample_rate as f64 / 2.0;
    let wn = (cutoff_hz / nyq).clamp(0.01, 0.99);
    let sos = butter_highpass_sos(4, wn);
    let high = sos_filter(&sos, x);
    let complement = x.iter().zip(high.iter()).map(|(&a, &b)| a - b).collect();
    (high, complement)
}

/// 周期方波，duty 0.5（scipy `square(2*pi*f*t)`）。
fn square_lfo(n: usize, sample_rate: u32, frequency: f64) -> Vec<f64> {
    let w = 2.0 * std::f64::consts::PI * frequency;
    (0..n)
        .map(|i| {
            let phase = (w * (i as f64 / sample_rate as f64)) % (2.0 * std::f64::consts::PI);
            if phase < std::f64::consts::PI {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// 按 LFO 调制的音高弯曲：
/// `ratio = 2^(lfo*strength*100/1200)`，累积漂移高通 20 Hz，再按 `i + drift` 插值。
fn modulate_pitch(band: &[f64], sample_rate: u32, lfo: &[f64], strength: f64) -> Vec<f64> {
    let n = band.len();
    if n == 0 {
        return Vec::new();
    }
    let mut ratio = Vec::with_capacity(n);
    let mut ratio_sum = 0.0;
    for &l in lfo {
        let r = 2.0f64.powf(l * strength * GRWC_MAX_CENTS / 1200.0);
        ratio.push(r);
        ratio_sum += r;
    }
    let ratio_mean = ratio_sum / n as f64;

    let mut drift = Vec::with_capacity(n);
    let mut cumsum = 0.0;
    for (i, &r) in ratio.iter().enumerate() {
        cumsum += r;
        drift.push((cumsum - ratio[0]) - i as f64 * ratio_mean);
    }
    if n > 100 {
        let hp_norm = 20.0 / (sample_rate as f64 / 2.0);
        let sos = butter_highpass_sos(2, hp_norm);
        drift = sos_filter(&sos, &drift);
    }

    let mut out = Vec::with_capacity(n);
    for (i, &d) in drift.iter().enumerate() {
        let idx = (i as f64 + d).clamp(0.0, (n - 1) as f64);
        let i0 = (idx.floor() as usize).min(n - 1);
        let i1 = (i0 + 1).min(n - 1);
        let frac = idx - i0 as f64;
        out.push(band[i0] + (band[i1] - band[i0]) * frac);
    }
    out
}

/// 反相 Butterworth 原型 → 双线性变换 → 二阶节。公式与 OpenUtau
/// `HifiIir.ButterHighpassSos` 逐行一致（scipy 的 sos 排列语义）。
fn butter_highpass_sos(order: usize, wn: f64) -> Vec<[f64; 6]> {
    let warped = 4.0 * (std::f64::consts::PI * wn / 2.0).tan();
    let mut poles = Vec::with_capacity(order);
    let mut gain_num = Complex::new(1.0, 0.0);
    let mut gain_den = Complex::new(1.0, 0.0);
    for k in 0..order {
        let m = -(order as i32) + 1 + 2 * k as i32;
        let angle = std::f64::consts::PI * m as f64 / (2.0 * order as f64);
        let analog = -Complex::new(angle.cos(), angle.sin());
        let hp = warped / analog;
        poles.push((4.0 + hp) / (4.0 - hp));
        gain_num *= 4.0;
        gain_den *= 4.0 - hp;
    }
    let gain = (gain_num / gain_den).re;

    let mut real_pole = None;
    let mut pairs = Vec::new();
    for p in poles {
        if p.im.abs() < 1.0e-12 {
            real_pole = Some(p.re);
        } else if p.im > 0.0 {
            pairs.push(p);
        }
    }
    pairs.sort_by(|a, b| {
        a.norm()
            .partial_cmp(&b.norm())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sections = Vec::with_capacity(order.div_ceil(2));
    if let Some(p) = real_pole {
        sections.push([1.0, -1.0, 0.0, 1.0, -p, 0.0]);
    }
    for p in pairs {
        sections.push([1.0, -2.0, 1.0, 1.0, -2.0 * p.re, p.re * p.re + p.im * p.im]);
    }
    if let Some(first) = sections.first_mut() {
        first[0] *= gain;
        first[1] *= gain;
        first[2] *= gain;
    }
    sections
}

/// 正序二阶节滤波（scipy `sosfilt`，零初始状态）。
fn sos_filter(sections: &[[f64; 6]], x: &[f64]) -> Vec<f64> {
    let mut y = x.to_vec();
    for s in sections {
        biquad(
            s[0] / s[3],
            s[1] / s[3],
            s[2] / s[3],
            s[4] / s[3],
            s[5] / s[3],
            &mut y,
        );
    }
    y
}

/// 转置直接 II 型单节，原地滤波。
fn biquad(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64, y: &mut [f64]) {
    let mut z1 = 0.0;
    let mut z2 = 0.0;
    for v in y.iter_mut() {
        let xi = *v;
        let yi = b0 * xi + z1;
        z1 = b1 * xi - a1 * yi + z2;
        z2 = b2 * xi - a2 * yi;
        *v = yi;
    }
}

fn rms(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f64, sample_rate: u32, n: usize, amp: f64) -> Vec<f32> {
        (0..n)
            .map(|i| {
                (amp * (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate as f64).sin())
                    as f32
            })
            .collect()
    }

    #[test]
    fn zero_strength_and_empty_input_are_bit_identical() {
        let input = sine(220.0, 48_000, 4_800, 0.5);
        assert_eq!(apply_growl_mono(&input, 48_000, 0.0), input);
        assert_eq!(apply_growl_mono(&input, 48_000, f64::NAN), input);
        assert!(apply_growl_mono(&[], 48_000, 1.0).is_empty());
    }

    #[test]
    fn output_length_and_finiteness_are_preserved() {
        let input = sine(220.0, 48_000, 12_345, 0.9);
        let out = apply_growl_mono(&input, 48_000, 1.0);
        assert_eq!(out.len(), input.len());
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn growl_changes_the_high_band_but_preserves_low_fundamental() {
        // 220 Hz 纯音落在 400 Hz 高通以下，GRWC 应几乎不动它；
        // 2 kHz 纯音落在通带内，应被调制并产生新的边带。
        let low = sine(220.0, 48_000, 24_000, 0.5);
        let low_out = apply_growl_mono(&low, 48_000, 1.0);
        let delta_low = low
            .iter()
            .zip(low_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(delta_low < 0.05, "低频不应被大幅调制，实测 {delta_low}");

        let high = sine(2_000.0, 48_000, 24_000, 0.5);
        let high_out = apply_growl_mono(&high, 48_000, 1.0);
        let delta_high = high
            .iter()
            .zip(high_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(delta_high > 0.1, "高频应被调制，实测 {delta_high}");
    }

    #[test]
    fn strong_parameters_stay_bounded_and_loudness_matched() {
        let input = sine(220.0, 44_100, 44_100, 0.4);
        let out = apply_growl_mono(&input, 44_100, 1.0);
        let in_rms = rms(&input.iter().map(|&v| f64::from(v)).collect::<Vec<_>>());
        let out_rms = rms(&out.iter().map(|&v| f64::from(v)).collect::<Vec<_>>());
        assert!(out.iter().all(|v| v.abs() <= 4.0), "调制后不应出现异常大值");
        assert!(
            (out_rms / in_rms - 1.0).abs() < 0.5,
            "RMS 应保持同量级，in={in_rms}, out={out_rms}"
        );
    }

    #[test]
    fn envelope_is_continuous_at_zero_strength_and_supports_ramps() {
        let input = sine(2_000.0, 48_000, 12_000, 0.4);
        let out = apply_growl_envelope(&input, 48_000, |i| {
            // 前 1/4 为 0 → 必须 bit 级等于干声；后 1/2 渐变到 1。
            if i < 3_000 {
                0.0
            } else {
                ((i - 3_000) as f64 / 9_000.0).min(1.0)
            }
        });
        assert_eq!(&out[..3_000], &input[..3_000]);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.len() == input.len());
    }
}
