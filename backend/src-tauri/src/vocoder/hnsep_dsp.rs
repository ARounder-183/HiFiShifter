//! 谐波/噪声分离（HNSEP）的**频谱域**前端：STFT → mask → ISTFT。
//!
//! # 主要内容
//! - [`separate`]：完整流水线。把波形做居中 STFT，交给 `predict_mask` 得到复数掩码，
//!   再乘回频谱做 ISTFT，返回谐波分量（噪声 = 原信号 − 谐波）。
//! - [`periodic_hann`]：周期 Hann 窗（`0.5(1 - cos(2πn/N))`）。
//! - 帧数对齐逻辑：把帧数补齐到 `SEGMENT_FRAMES` 的整数倍（网络要求）。
//!
//! # 作用
//! **本模块存在的理由**：原实现使用的 HNSEP 模型是「波形域」版 —— ONNX 图内含
//! STFT、编码器、24 个 LSTM、解码器与 ISTFT，输入输出都是波形 `[1, N]`。
//! 那份模型在 GPU 上几乎没有收益（实测 CoreML 仅 1.02~1.04x），原因是：
//!
//! 1. LSTM 是串行递归结构，逐帧依赖，GPU 无法并行化；
//! 2. STFT/ISTFT 在图内（ConvTranspose 实现），算子要么不被 EP 支持而回退 CPU，
//!    要么把图切成大量碎片，kernel launch 开销吃掉收益。
//!
//! 改用 mask-only 模型后，ONNX 里**只剩 mask 网络**（纯卷积 + LSTM），
//! STFT/ISTFT 回到 Rust 侧。这与 OpenUtau 的做法一致
//! （其 `Hnsep.cs` 同样是「STFT 在宿主、只有网络在 ONNX」），
//! GPU 上因此才有可加速的密集算子。
//!
//! # 与其他模块的关系
//! - 由 [`crate::hnsep_onnx`] 调用：它负责会话、缓存与张量装配，本模块只做 DSP。
//! - 与 [`crate::rd_tension`] 是**两套独立**的 STFT：Hop/加窗/填充约定都不同
//!   （HNSEP 用 hop 512 + 零填充 + 周期 Hann；Rd 张力用 hop 256 + 反射填充 +
//!   周期 Hann）。刻意不共用，避免参数漂移破坏各自与参考实现的对齐。
//!
//! # 来源
//! 移植自 OpenUtau 0.1.571-beta（MIT）的 `OpenUtau.Core/Analysis/Hnsep.cs`
//! 中 `Separate` / `FrameScratch` / `PeriodicHann`（`Hnsep.cs:74-188`）。
//!
//! # 维护说明
//! - **加窗与填充约定是数值契约**：零填充（非反射）+ 居中（`n_fft/2`）+ 周期 Hann
//!   三者共同决定与原波形域模型的一致性。改动任何一项都会改变分离结果。
//! - ISTFT 按**窗平方和**归一化；`wsum` 过小处输出 0（不是保留原值），
//!   与参考实现一致。

use num_complex::Complex;
use rustfft::FftPlanner;
use std::sync::Arc;

/// 网络要求的帧数粒度：`n_frames` 必须是该值的整数倍。
pub const SEGMENT_FRAMES: usize = 32;

/// 周期 Hann 窗：`0.5(1 - cos(2πn/N))`。
///
/// 【为什么是周期而非对称形式】与 `torch.hann_window(N)`（默认 periodic=True）
/// 及 OpenUtau 的 `Hnsep.PeriodicHann` 一致。周期窗在 `hop = N/4` 下满足 COLA，
/// 配合下方按窗平方和归一化的 ISTFT 可无幅度调制地重建。
pub fn periodic_hann(len: usize) -> Vec<f64> {
    (0..len)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / len as f64).cos())
        .collect()
}

/// 分离流水线的一次完整执行所需的中间量。
///
/// 拆成结构体是为了让「STFT → 取 mask → ISTFT」三个阶段可以分别测试：
/// 单元测试可以用一个常量掩码替换网络，从而在**没有 ONNX 模型**的情况下
/// 验证 STFT/ISTFT 的数学（OpenUtau 的 `HnsepTest` 用同样手法）。
pub struct Separation {
    /// 输入频谱 `[frames * bins]`，行优先（每帧 `bins` 个复数）。
    pub spectrum: Vec<Complex<f64>>,
    /// 网络输入张量：`[1, 2, bins, frames]` 行优先（实部整块、虚部整块）。
    pub mask_input: Vec<f32>,
    pub bins: usize,
    pub frames: usize,
    /// 原信号长度（样本数）。
    pub sample_len: usize,
    /// 居中填充的偏移：样本 `i` 位于 `padded[n_fft/2 + sample_offset + i]`。
    pub sample_offset: usize,
    /// 参与 OLA 的总长度（含填充与两侧 `n_fft/2`）。
    pub padded_len: usize,
}

/// 把帧数补齐到 `SEGMENT_FRAMES` 的整数倍，并计算居中填充量。
///
/// 与参考实现逐行对应（`Hnsep.cs:106-113`）：
/// ```text
/// seg    = SEGMENT_FRAMES * hop
/// t1     = T + hop
/// tPad   = seg * ceil(t1 / seg) - t1
/// left   = tPad / 2 / hop * hop        // 对齐到 hop 的整数倍
/// length = T + tPad
/// frames = 1 + length / hop            // ⇒ frames 是 SEGMENT_FRAMES 的整数倍
/// ```
///
/// # 参数
/// - `sample_len`：原始样本数 `T`
/// - `hop`：帧步长（必须 > 0）
///
/// # 返回
/// `(frames, left, length)`；`hop == 0` 时返回 `(0, 0, sample_len)`。
pub fn align_frames(sample_len: usize, hop: usize) -> (usize, usize, usize) {
    if hop == 0 {
        return (0, 0, sample_len);
    }
    let seg = SEGMENT_FRAMES * hop;
    let t1 = sample_len + hop;
    // 向上取整到 seg 的整数倍，再减去 t1（这正是参考实现的 (t1-1)/seg + 1 写法）
    let t_pad = seg * t1.div_ceil(seg) - t1;
    let left = t_pad / 2 / hop * hop;
    let length = sample_len + t_pad;
    let frames = 1 + length / hop;
    (frames, left, length)
}

/// 居中 STFT（**零填充**）+ 装配网络输入张量。
///
/// 与参考实现一致：`padded` 长度 `length + n_fft`，原信号放在
/// `n_fft/2 + left` 处，两侧自然为零。窗长 == FFT 长度。
///
/// # 参数
/// - `x`：单声道波形
/// - `n_fft` / `hop`：STFT 参数（`n_fft` 必须为 2 的幂，供 rustfft 实数重建使用）
/// - `window`：长度必须等于 `n_fft`
///
/// # 错误
/// `window.len() != n_fft`、`n_fft == 0`、`hop == 0` 或 `n_fft` 非 2 的幂时返回 `Err`。
pub fn stft_mask_input(
    x: &[f32],
    n_fft: usize,
    hop: usize,
    window: &[f64],
) -> Result<Separation, String> {
    if n_fft == 0 || hop == 0 {
        return Err("hnsep stft: n_fft/hop must be non-zero".to_string());
    }
    if window.len() != n_fft {
        return Err(format!(
            "hnsep stft: window len {} != n_fft {n_fft}",
            window.len()
        ));
    }
    if !n_fft.is_power_of_two() {
        return Err(format!("hnsep stft: n_fft {n_fft} must be a power of two"));
    }

    let bins = n_fft / 2 + 1;
    let sample_len = x.len();
    let (frames, left, length) = align_frames(sample_len, hop);
    let padded_len = length + n_fft;

    if frames == 0 {
        return Ok(Separation {
            spectrum: Vec::new(),
            mask_input: Vec::new(),
            bins,
            frames: 0,
            sample_len,
            sample_offset: left,
            padded_len,
        });
    }

    // 网络输入布局：[1, 2, bins, frames]，实部整块在前、虚部整块在后。
    let mut mask_input = vec![0.0f32; 2 * bins * frames];
    let mut spectrum = vec![Complex::new(0.0, 0.0); frames * bins];

    let mut planner = FftPlanner::<f64>::new();
    let fft: Arc<dyn rustfft::Fft<f64>> = planner.plan_fft_forward(n_fft);
    let mut buf = vec![Complex::new(0.0, 0.0); n_fft];

    for m in 0..frames {
        // 帧 m 覆盖 padded[m*hop .. m*hop + n_fft]
        let base = m * hop;
        for i in 0..n_fft {
            // padded 索引 j 对应的原样本索引
            let j = base + i;
            let sample = if j >= n_fft / 2 + left && j < n_fft / 2 + left + sample_len {
                x[j - (n_fft / 2 + left)] as f64
            } else {
                0.0
            };
            buf[i] = Complex::new(sample * window[i], 0.0);
        }
        fft.process(&mut buf);

        let frame_off = m * bins;
        for k in 0..bins {
            spectrum[frame_off + k] = buf[k];
            mask_input[k * frames + m] = buf[k].re as f32;
            mask_input[(bins + k) * frames + m] = buf[k].im as f32;
        }
    }

    Ok(Separation {
        spectrum,
        mask_input,
        bins,
        frames,
        sample_len,
        sample_offset: left,
        padded_len,
    })
}

/// 把网络输出的掩码乘回频谱并做 ISTFT，返回谐波波形（长度 = 原信号长度）。
///
/// # 参数
/// - `sep`：由 [`stft_mask_input`] 得到
/// - `mask`：网络输出，布局与 `sep.mask_input` 相同（`[1,2,bins,frames]` 行优先）
/// - `n_fft` / `hop` / `window`：必须与 STFT 时逐字相同
///
/// # 错误
/// `mask.len() != 2 * bins * frames` 时返回 `Err`（形状不符属于调用方 bug，
/// 静默截断会产生错位的音频）。
///
/// # 特殊说明
/// - ISTFT 按**窗平方和**归一化；`wsum <= 1e-11` 处输出 0（与参考实现一致）。
/// - 频率轴只填充 `bins` 个点，负频率由**共轭对称**显式补出（rustfft 的复数
///   逆变换不会自行推导，缺失会让每个频率分量减半 —— 见 `rd_tension::istft`
///   的同类说明）。
pub fn istft_with_mask(
    sep: &Separation,
    mask: &[f32],
    n_fft: usize,
    hop: usize,
    window: &[f64],
) -> Result<Vec<f32>, String> {
    let bins = sep.bins;
    let frames = sep.frames;
    if frames == 0 || sep.sample_len == 0 {
        return Ok(vec![0.0; sep.sample_len]);
    }
    if mask.len() != 2 * bins * frames {
        return Err(format!(
            "hnsep istft: mask len {} != 2*{bins}*{frames} = {}",
            mask.len(),
            2 * bins * frames
        ));
    }
    if !n_fft.is_power_of_two() {
        return Err(format!("hnsep istft: n_fft {n_fft} must be a power of two"));
    }

    let mut planner = FftPlanner::<f64>::new();
    let ifft: Arc<dyn rustfft::Fft<f64>> = planner.plan_fft_inverse(n_fft);

    let total = sep.padded_len;
    let mut y = vec![0.0f64; total];
    let mut wsum = vec![0.0f64; total];
    let mut buf = vec![Complex::new(0.0, 0.0); n_fft];
    let scale = 1.0 / n_fft as f64;

    for m in 0..frames {
        let spec_off = m * bins;
        for k in 0..bins {
            let s = sep.spectrum[spec_off + k];
            let mr = mask[k * frames + m] as f64;
            let mi = mask[(bins + k) * frames + m] as f64;
            // 复数乘法：spec * mask
            buf[k] = Complex::new(s.re * mr - s.im * mi, s.re * mi + s.im * mr);
        }
        // DC 与 Nyquist 必须是实数（实数信号的频谱约束）
        buf[0] = Complex::new(buf[0].re, 0.0);
        buf[bins - 1] = Complex::new(buf[bins - 1].re, 0.0);
        // 共轭对称补出负频率（否则实部只取到解析信号的一半）
        for k in 1..bins.saturating_sub(1) {
            buf[n_fft - k] = Complex::new(buf[k].re, -buf[k].im);
        }

        ifft.process(&mut buf);

        let p = m * hop;
        for i in 0..n_fft {
            if p + i >= total {
                break;
            }
            y[p + i] += buf[i].re * scale * window[i];
            wsum[p + i] += window[i] * window[i];
        }
    }

    // 取出原信号所在区间，按窗平方和归一化
    let start = n_fft / 2 + sep.sample_offset;
    let mut out = vec![0.0f32; sep.sample_len];
    for (i, o) in out.iter_mut().enumerate() {
        let j = start + i;
        if j < total {
            *o = if wsum[j] > 1e-11 {
                (y[j] / wsum[j]) as f32
            } else {
                0.0
            };
        }
    }
    Ok(out)
}

/// 完整分离流水线：STFT → `predict_mask` → ISTFT。
///
/// `predict_mask` 接收网络输入张量（`[1,2,bins,frames]` 行优先）并返回同形状掩码。
/// 传闭包而非直接持有会话，使单元测试可用常量掩码在**无模型**下验证数学
/// （与 OpenUtau `Hnsep.Separate` 的设计一致）。
///
/// # 返回
/// 谐波分量（长度 = `x.len()`）。噪声 = `x − harmonic`。
pub fn separate(
    x: &[f32],
    n_fft: usize,
    hop: usize,
    window: &[f64],
    predict_mask: impl FnOnce(&[f32]) -> Result<Vec<f32>, String>,
) -> Result<Vec<f32>, String> {
    let sep = stft_mask_input(x, n_fft, hop, window)?;
    if sep.frames == 0 {
        return Ok(vec![0.0; x.len()]);
    }
    let mask = predict_mask(&sep.mask_input)?;
    istft_with_mask(&sep, &mask, n_fft, hop, window)
}

#[cfg(test)]
mod tests {
    use super::*;

    const N_FFT: usize = 2048;
    const HOP: usize = 512;


    /// 构造 `[1, 2, bins, frames]` 布局的**复数**掩码。
    ///
    /// 【为什么必须用这个辅助函数】布局是「实部整块 + 虚部整块」，
    /// 直接 `vec![1.0; 2*bins*frames]` 意味着实部与虚部**都是 1**，
    /// 即掩码 `1 + 1i` —— 那是幅度 ×√2、相位 +45°，**不是恒等**。
    /// 恒等掩码必须是 `(re=1, im=0)`。
    fn complex_mask(bins: usize, frames: usize, re: f32, im: f32) -> Vec<f32> {
        let mut m = vec![0.0f32; 2 * bins * frames];
        for k in 0..bins * frames {
            m[k] = re;
            m[bins * frames + k] = im;
        }
        m
    }

    /// 恒等掩码必须**逐样本**还原原信号。
    ///
    /// 这是 STFT/ISTFT 配对的正确性判据：加窗、零填充、居中与窗平方和归一化
    /// 四者只要有一处错位，往返就会有偏差或整体缩放。
    #[test]
    fn identity_mask_reconstructs_the_signal() {
        let window = periodic_hann(N_FFT);
        // 混合正弦，避开 DC/Nyquist 边界
        let n = 20_000;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / 44_100.0;
                (0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                    + 0.2 * (2.0 * std::f64::consts::PI * 1330.0 * t).sin()) as f32
            })
            .collect();

        let sep = stft_mask_input(&x, N_FFT, HOP, &window).unwrap();
        let mask = complex_mask(sep.bins, sep.frames, 1.0, 0.0);
        let y = istft_with_mask(&sep, &mask, N_FFT, HOP, &window).unwrap();

        assert_eq!(y.len(), x.len());
        let max_err = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-4,
            "identity mask must reconstruct the signal, max err {max_err}"
        );
    }

    /// 零掩码必须产出静音（不得残留原信号）。
    #[test]
    fn zero_mask_produces_silence() {
        let window = periodic_hann(N_FFT);
        let x: Vec<f32> = (0..8192).map(|i| (i as f32 * 0.01).sin()).collect();
        let sep = stft_mask_input(&x, N_FFT, HOP, &window).unwrap();
        let mask = complex_mask(sep.bins, sep.frames, 0.0, 0.0);
        let y = istft_with_mask(&sep, &mask, N_FFT, HOP, &window).unwrap();
        assert!(
            y.iter().all(|v| v.abs() < 1e-6),
            "zero mask must silence the output"
        );
    }

    /// 帧数必须是 `SEGMENT_FRAMES` 的整数倍（网络硬性要求）。
    #[test]
    fn frames_are_a_multiple_of_segment_frames() {
        for &n in &[0usize, 1, 100, 512, 2048, 8192, 44_100, 100_000] {
            let (frames, _left, length) = align_frames(n, HOP);
            assert_eq!(
                frames % SEGMENT_FRAMES,
                0,
                "n={n} gave {frames} frames (length={length}), not a multiple of {SEGMENT_FRAMES}"
            );
        }
    }

    /// 居中填充必须对齐到 hop 的整数倍，且长度足够容纳所有帧。
    #[test]
    fn padding_is_hop_aligned_and_covers_all_frames() {
        for &n in &[0usize, 1, 512, 8192, 44_100] {
            let (frames, left, length) = align_frames(n, HOP);
            assert_eq!(left % HOP, 0, "left padding must be hop-aligned (n={n})");
            assert!(length >= n, "padded length must cover the signal (n={n})");
            // 最后一帧的右端不得越过 padded 缓冲
            assert!(
                (frames - 1) * HOP + N_FFT <= length + N_FFT,
                "last frame overruns the buffer (n={n})"
            );
        }
    }

    /// 退化输入不得 panic：极短、空、以及不足一帧。
    #[test]
    fn degenerate_inputs_are_safe() {
        let window = periodic_hann(N_FFT);
        for &n in &[0usize, 1, 2, 511, 512] {
            let x = vec![0.1f32; n];
            let r = separate(&x, N_FFT, HOP, &window, |inp| {
                Ok(vec![1.0f32; inp.len()])
            });
            let y = r.unwrap();
            assert_eq!(y.len(), n, "length must be preserved for n={n}");
            assert!(y.iter().all(|v| v.is_finite()));
        }
    }

    /// 参数非法必须报错，而不是静默产出错位音频。
    #[test]
    fn invalid_params_are_rejected() {
        let window = periodic_hann(N_FFT);
        let x = vec![0.1f32; 4096];
        // 窗长不符
        assert!(stft_mask_input(&x, N_FFT, HOP, &window[..100]).is_err());
        // hop = 0
        assert!(stft_mask_input(&x, N_FFT, 0, &window).is_err());
        // n_fft = 0
        assert!(stft_mask_input(&x, 0, HOP, &[]).is_err());
        // 非 2 的幂
        assert!(stft_mask_input(&x, 1000, HOP, &vec![1.0; 1000]).is_err());

        // mask 形状不符
        let sep = stft_mask_input(&x, N_FFT, HOP, &window).unwrap();
        assert!(istft_with_mask(&sep, &[1.0f32; 10], N_FFT, HOP, &window).is_err());
    }

    /// 掩码只影响幅度时，输出应落在合理范围内（不得爆音）。
    #[test]
    fn half_gain_mask_halves_the_amplitude() {
        let window = periodic_hann(N_FFT);
        let n = 16_384;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44_100.0).sin() as f32 * 0.5)
            .collect();
        let sep = stft_mask_input(&x, N_FFT, HOP, &window).unwrap();
        let mask = complex_mask(sep.bins, sep.frames, 0.5, 0.0);
        let y = istft_with_mask(&sep, &mask, N_FFT, HOP, &window).unwrap();

        let rms = |s: &[f32]| {
            (s.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / s.len() as f64).sqrt()
        };
        let ratio = rms(&y) / rms(&x);
        assert!(
            (ratio - 0.5).abs() < 0.02,
            "0.5 mask should halve amplitude, got ratio {ratio}"
        );
    }

    /// 常量掩码走 `separate` 的完整路径，验证闭包装配无误。
    #[test]
    fn separate_with_constant_mask_matches_direct_call() {
        let window = periodic_hann(N_FFT);
        let n = 8192;
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.02).sin() * 0.4).collect();
        let y = separate(&x, N_FFT, HOP, &window, |inp| {
            assert_eq!(inp.len() % 2, 0);
            // 输入布局是 [1,2,bins,frames] 展平 ⇒ 一半实部、一半虚部
            let half = inp.len() / 2;
            let mut m = vec![0.0f32; inp.len()];
            for k in 0..half {
                m[k] = 0.25;
            }
            Ok(m)
        })
        .unwrap();
        assert_eq!(y.len(), n);

        let sep = stft_mask_input(&x, N_FFT, HOP, &window).unwrap();
        let direct = istft_with_mask(
            &sep,
            &complex_mask(sep.bins, sep.frames, 0.25, 0.0),
            N_FFT,
            HOP,
            &window,
        )
        .unwrap();
        assert_eq!(y, direct);
    }
}





