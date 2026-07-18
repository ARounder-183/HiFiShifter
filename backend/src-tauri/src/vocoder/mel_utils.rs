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

/// Reflect-pad `y` into a pre-allocated buffer (zero-copy reuse).
pub fn reflect_pad_into(y: &[f32], left: usize, right: usize, out: &mut Vec<f32>) {
    out.clear();
    if y.is_empty() {
        out.resize(left + right, 0.0);
        return;
    }
    let len = y.len();
    out.reserve(left + len + right);
    for i in -(left as isize)..0 {
        out.push(y[reflect_index(i, len)]);
    }
    out.extend_from_slice(y);
    for i in (len as isize)..((len as isize) + (right as isize)) {
        out.push(y[reflect_index(i, len)]);
    }
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

/// Compute STFT magnitude spectrogram.
/// Returns a `Vec<Vec<f32>>` where result[freq_bin][frame_idx] is the magnitude.
pub fn stft_magnitude(signal: &[f32], n_fft: usize, win_size: usize, hop: usize, window: &[f32]) -> Result<Vec<Vec<f32>>, String> {
    use num_complex::Complex32;
    use rustfft::FftPlanner;

    if signal.len() < win_size {
        return Ok(vec![vec![0.0; 1]]);
    }

    let n_freqs = n_fft / 2 + 1;
    let n_frames = 1 + (signal.len() - win_size) / hop;

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut fft_buf = vec![Complex32::new(0.0, 0.0); n_fft];

    let mut result = vec![vec![0.0f32; n_frames]; n_freqs];

    for frame in 0..n_frames {
        let start = frame * hop;

        for i in 0..win_size.min(signal.len() - start) {
            fft_buf[i] = Complex32::new(signal[start + i] * window[i], 0.0);
        }
        for i in win_size..n_fft {
            fft_buf[i] = Complex32::new(0.0, 0.0);
        }

        fft.process(&mut fft_buf);

        for f in 0..n_freqs {
            let c = fft_buf[f];
            result[f][frame] = (c.re * c.re + c.im * c.im).sqrt();
        }
    }

    Ok(result)
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
