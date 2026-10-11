//! Shared audio/spectral utility functions for vocoder modules.
//!
//! Deduplicates mel filterbank, windowing, padding, and resampling code
//! that was copy-pasted across nsf_hifigan_onnx, fcpe_onnx, and hnsep_onnx.

use ndarray::Array2;

/// Slaney mel scale (librosa default — matches ONNX model training).
pub fn hz_to_mel_slaney(hz: f32) -> f32 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f32).ln() / 27.0;

    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        (hz - f_min) / f_sp
    }
}

/// Inverse Slaney mel scale.
pub fn mel_to_hz_slaney(mel: f32) -> f32 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f32).ln() / 27.0;

    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_min + f_sp * mel
    }
}

/// Build a Slaney-normalized mel filterbank matrix.
pub fn mel_filterbank_slaney(
    sr: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Array2<f32> {
    let n_freqs = n_fft / 2 + 1;
    let mel_min = hz_to_mel_slaney(fmin.max(0.0));
    let mel_max = hz_to_mel_slaney(fmax.max(fmin + 1.0));

    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..(n_mels + 2) {
        let t = i as f32 / (n_mels + 1) as f32;
        mel_points.push(mel_min + (mel_max - mel_min) * t);
    }

    let mut hz_points = Vec::with_capacity(n_mels + 2);
    for &m in &mel_points {
        hz_points.push(mel_to_hz_slaney(m));
    }

    let mut fftfreqs = Vec::with_capacity(n_freqs);
    for i in 0..n_freqs {
        fftfreqs.push((i as f32) * (sr as f32) / (n_fft as f32));
    }

    let mut weights = Array2::<f32>::zeros((n_mels, n_freqs));
    for m in 0..n_mels {
        let f_left = hz_points[m];
        let f_center = hz_points[m + 1];
        let f_right = hz_points[m + 2];

        let fdiff_left = (f_center - f_left).max(1e-6);
        let fdiff_right = (f_right - f_center).max(1e-6);

        for (i, &f) in fftfreqs.iter().enumerate() {
            let lower = (f - f_left) / fdiff_left;
            let upper = (f_right - f) / fdiff_right;
            weights[[m, i]] = lower.min(upper).max(0.0);
        }

        let enorm = 2.0 / (f_right - f_left).max(1e-6);
        for i in 0..n_freqs {
            weights[[m, i]] *= enorm;
        }
    }

    weights
}

/// Reflect-index helper for padding.
pub fn reflect_index(i: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * ((len as isize) - 1);
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

/// Reflect-pad `y` with `left` and `right` samples, returning a new Vec.
pub fn reflect_pad(y: &[f32], left: usize, right: usize) -> Vec<f32> {
    if y.is_empty() {
        return vec![0.0; left + right];
    }

    let len = y.len();
    let mut out = Vec::with_capacity(left + len + right);

    for i in -(left as isize)..0 {
        out.push(y[reflect_index(i, len)]);
    }
    out.extend_from_slice(y);
    for i in (len as isize)..((len as isize) + (right as isize)) {
        out.push(y[reflect_index(i, len)]);
    }
    out
}

/// Hann window of length `len`.
pub fn hann_window(len: usize) -> Vec<f32> {
    if len == 0 {
        return vec![];
    }
    if len == 1 {
        return vec![1.0];
    }

    let denom = (len - 1) as f32;
    let mut w = Vec::with_capacity(len);
    for n in 0..len {
        let x = (2.0 * std::f32::consts::PI * (n as f32)) / denom;
        w.push(0.5 - 0.5 * x.cos());
    }
    w
}

/// Linear resample mono audio from `in_rate` to `out_rate`.
///
/// Uses pre-computed inverse ratio (multiply instead of divide in the hot loop)
/// and chunked processing to enable auto-vectorization by LLVM.
pub fn linear_resample_mono(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    if input.is_empty() || in_rate == 0 || out_rate == 0 || in_rate == out_rate {
        return input.to_vec();
    }
    if input.len() < 2 {
        return input.to_vec();
    }

    let ratio = out_rate as f64 / in_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    let inv_ratio = 1.0 / ratio;
    let input_last = input.len() - 1;

    let mut out = vec![0.0f32; out_len];

    // Process in chunks of 8 for compiler auto-vectorization.
    // LLVM can emit SIMD instructions (SSE/AVX on x86_64, NEON on ARM)
    // for this pattern when the loop body is simple enough.
    let chunk_size = 8usize;
    let num_chunks = out_len / chunk_size;

    for chunk in 0..num_chunks {
        let base = chunk * chunk_size;
        for k in 0..chunk_size {
            let idx = base + k;
            let src = idx as f64 * inv_ratio;
            let i0 = (src as usize).min(input_last.saturating_sub(1));
            let i1 = (i0 + 1).min(input_last);
            let frac = (src - i0 as f64) as f32;
            out[idx] = input[i0] + (input[i1] - input[i0]) * frac;
        }
    }

    // Tail: process remaining samples
    for idx in (num_chunks * chunk_size)..out_len {
        let src = idx as f64 * inv_ratio;
        let i0 = (src as usize).min(input_last.saturating_sub(1));
        let i1 = (i0 + 1).min(input_last);
        let frac = (src - i0 as f64) as f32;
        out[idx] = input[i0] + (input[i1] - input[i0]) * frac;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 恒等率、空输入、单点输入都必须原样返回 —— 它们是"便宜输出层"的早退分支，
    /// 一旦被改坏，48k 输出会莫名其妙地重采样甚至清空。
    #[test]
    fn resampling_short_circuits_leave_input_untouched() {
        let ramp = [0.0f32, 1.0, 2.0, 3.0];
        assert_eq!(linear_resample_mono(&ramp, 48000, 48000), ramp);
        assert!(linear_resample_mono(&[], 44100, 48000).is_empty());
        assert_eq!(linear_resample_mono(&[0.5], 44100, 48000), [0.5]);
        // 非法率同样原样返回，不制造除零。
        assert_eq!(linear_resample_mono(&ramp, 0, 48000), ramp);
        assert_eq!(linear_resample_mono(&ramp, 44100, 0), ramp);
    }

    /// 输出长度按比例取整；直流被保留（线性插值不改变常量信号）。
    #[test]
    fn resampling_length_and_endpoints_follow_the_ratio() {
        let input = vec![0.25f32; 100];
        let up = linear_resample_mono(&input, 44100, 48000);
        assert_eq!(up.len(), ((100.0_f64 * 48000.0 / 44100.0).round()) as usize);
        assert!(up.iter().all(|v| (v - 0.25).abs() < 1e-6));
        let down = linear_resample_mono(&input, 48000, 44100);
        assert_eq!(
            down.len(),
            ((100.0_f64 * 44100.0 / 48000.0).round()) as usize
        );
        assert!(down.iter().all(|v| (v - 0.25).abs() < 1e-6));
    }

    /// 2× 上采样的**斜坡**应落在原采样点之间的线性插值上；且尾部不能被外推成越界值。
    ///
    /// 【为什么要跨过 chunk_size=8】重采样内层按 8 分块 + 独立尾循环，两段的边界处理
    /// 不一致时只会在这条缝上出错 —— 取一个非 8 倍数的输出长度把它钉住。
    #[test]
    fn resampling_interpolates_a_ramp_and_clamps_the_tail() {
        let ramp = [0.0f32, 1.0, 2.0, 3.0];
        let up = linear_resample_mono(&ramp, 1, 2);
        assert_eq!(up.len(), 8);
        for (i, value) in up.iter().enumerate().take(7) {
            // 前 7 帧是斜坡上的线性插值（0, 0.5, 1.0, …）。
            let expected = (i as f32) * 0.5;
            assert!(
                (value - expected).abs() < 1e-5,
                "frame {i}: {value} != {expected}"
            );
        }
        // 单调不减：任何"尾部外推/回绕"都会破坏它。
        assert!(up.windows(2).all(|w| w[1] >= w[0]));
    }

    /// reflect 索引在两端镜像，且对任意越界输入都落在 [0, len)。
    #[test]
    fn reflect_index_mirrors_at_both_ends() {
        assert_eq!(reflect_index(-1, 4), 1);
        assert_eq!(reflect_index(-2, 4), 2);
        assert_eq!(reflect_index(4, 4), 2);
        assert_eq!(reflect_index(5, 4), 1);
        assert_eq!(reflect_index(6, 4), 0);
        for i in -50..60 {
            assert!(reflect_index(i, 4) < 4);
            assert_eq!(reflect_index(i, 1), 0);
        }
    }

    /// reflect_pad 的左右边界取自镜像，长度正确。
    #[test]
    fn reflect_pad_mirrors_without_touching_the_body() {
        let body = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(
            reflect_pad(&body, 1, 2),
            [2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
        assert_eq!(reflect_pad(&[], 3, 2).len(), 5);
    }

    /// Hann 窗：端点为零、关于中心对称、峰值不超过 1。
    #[test]
    fn hann_window_is_symmetric_and_zero_at_the_ends() {
        assert!(hann_window(0).is_empty());
        assert_eq!(hann_window(1), [1.0]);
        let w = hann_window(5);
        assert_eq!(w.len(), 5);
        assert!(w[0].abs() < 1e-6 && w[4].abs() < 1e-6);
        for i in 0..5 {
            assert!((w[i] - w[4 - i]).abs() < 1e-6, "not symmetric at {i}");
            assert!((-1e-6..=1.0 + 1e-6).contains(&w[i]));
        }
    }

    /// Slaney 域与其逆在整条频带上互为反函数。
    #[test]
    fn slaney_mel_round_trips() {
        for hz in [0.0f32, 100.0, 500.0, 999.0, 1000.0, 4000.0, 8000.0] {
            let back = mel_to_hz_slaney(hz_to_mel_slaney(hz));
            assert!((back - hz).abs() <= 1e-3 * hz.max(1.0), "hz {hz} -> {back}");
        }
    }
}
