/*
 * audio/resample.rs - 带限（抗混叠）采样率转换。
 *
 * 主要内容：
 * - `resample_interleaved`：交错 PCM 的采样率转换，downsample 走带限插值
 *   （抗混叠），upsample 走同一套带限核（质量优于线性插值，且无额外分支）。
 * - `resample_mono`：单声道便捷封装。
 *
 * 与其他模块的关系：
 * - 本模块是**全项目唯一的采样率转换入口**。原 `audio_engine/io.rs` 与
 *   `audio/mixdown.rs` 各有一份逐字重复的 `linear_resample_interleaved`，
 *   已删除并统一到这里（见 P0-1）。
 * - `audio/formant_morph/decimator.rs` 保留其自有的整数抽取器：它面向
 *   LPC 极点估计，对通带边缘有更强的约束，且按整数抽取因子设计，
 *   不并入本模块（见该文件头部说明）。
 *
 * 设计要点：
 * - **为什么需要抗混叠**：线性插值在 downsample 时没有前置低通，源采样率
 *   高于输出采样率的频谱成分会折叠到可听带内（aliasing）。例如 96 kHz 源
 *   的 30 kHz 分量在降到 44.1 kHz 后会出现在 14.1 kHz 处，产生非谐波杂音。
 * - **核**：Blackman 窗 sinc。归一化方式为**逐输出样本除以权重和**，因此
 *   DC 增益恒为 1（不依赖截断长度与 cutoff），且边缘处按"仅统计有效抽头"
 *   重新归一化，等价于对称延拓，避免零填充造成的边缘衰减。
 * - **cutoff**：`min(1, out_rate/in_rate)`（相对输入 Nyquist）。downsample 时
 *   核被按 1/cutoff 拉伸，等效于先做输入域低通再插值。
 * - **性能**：核值由 LUT + 线性插值得到（LUT 覆盖核的整个支撑区间，恰好
 *   `SINC_ZERO_CROSSINGS` 个 sinc 周期，因此每周期采样点数恒定、与 cutoff
 *   无关）。避免逐抽头调用 sin/cos。
 * - **恒等直通**：`in_rate == out_rate` 或帧数 < 2 时逐样本拷贝，不做滤波。
 */

use std::f64::consts::PI;

/// 核的支撑半宽（以 sinc 过零点个数计）。越大过渡带越窄、阻带越深。
const SINC_ZERO_CROSSINGS: f64 = 12.0;

/// 支撑半宽上限（输入样本数）。极端降采样率时防止抽头数失控。
const MAX_HALF_WIDTH: f64 = 48.0;

/// 核 LUT 分辨率（覆盖 [0, half_width] 的采样点数）。
const KERNEL_LUT_SIZE: usize = 2048;

/// 低于该帧数直接原样返回（无法做插值）。
const MIN_FRAMES: usize = 2;

/// 带限插值核的查找表。
struct KernelLut {
    /// 支撑半宽（输入样本）。
    half_width: f64,
    /// 表项 [0..=KERNEL_LUT_SIZE]：第 i 项对应 |x| = i/KERNEL_LUT_SIZE * half_width。
    table: Vec<f32>,
}

impl KernelLut {
    /// 按给定 cutoff 构建 LUT。
    ///
    /// `cutoff` 相对输入 Nyquist（1.0 = 输入 Nyquist）。
    fn new(cutoff: f64) -> KernelLut {
        let cutoff = cutoff.clamp(1e-3, 1.0);
        let half_width = (SINC_ZERO_CROSSINGS / cutoff).clamp(1.0, MAX_HALF_WIDTH);

        let mut table = Vec::with_capacity(KERNEL_LUT_SIZE + 1);
        for i in 0..=KERNEL_LUT_SIZE {
            let p = i as f64 / KERNEL_LUT_SIZE as f64; // 归一化位置 [0, 1]
            let x = p * half_width; // 输入样本

            // sinc(cutoff * x)，x→0 时取极限 1。不乘 cutoff：整体增益由
            // 逐样本权重和归一化保证，DC 增益精确为 1。
            let s = if x < 1e-12 {
                1.0
            } else {
                let a = PI * cutoff * x;
                a.sin() / a
            };

            // Blackman 窗：w(0)=1，w(half_width)=0。
            let w = 0.42 + 0.5 * (PI * p).cos() + 0.08 * (2.0 * PI * p).cos();

            table.push((s * w) as f32);
        }

        KernelLut { half_width, table }
    }

    /// 核值（x 为输入样本域距离）。
    #[inline]
    fn eval(&self, x: f64) -> f32 {
        let ax = x.abs();
        if ax >= self.half_width {
            return 0.0;
        }
        let pos = ax / self.half_width * KERNEL_LUT_SIZE as f64;
        let i = pos as usize;
        // 边界保护：i 最大为 KERNEL_LUT_SIZE-1，使 i+1 合法。
        let i = i.min(KERNEL_LUT_SIZE - 1);
        let frac = (pos - i as f64) as f32;
        let a = self.table[i];
        let b = self.table[i + 1];
        a + (b - a) * frac
    }
}

/// 交错 PCM 采样率转换（带限、抗混叠）。
///
/// 参数：
/// - `input`：交错样本，长度必须是 `channels` 的整数倍。
/// - `channels`：声道数（0 视为非法，返回空）。
/// - `in_rate` / `out_rate`：输入 / 输出采样率。
///
/// 返回：交错样本，帧数为 `round(in_frames * out_rate / in_rate)`。
pub(crate) fn resample_interleaved(
    input: &[f32],
    channels: usize,
    in_rate: u32,
    out_rate: u32,
) -> Vec<f32> {
    if input.is_empty() || channels == 0 || in_rate == 0 || out_rate == 0 {
        return vec![];
    }
    let in_frames = input.len() / channels;
    if in_frames < MIN_FRAMES {
        return input.to_vec();
    }
    // 恒等直通：逐样本拷贝，不引入任何滤波与重采样误差。
    if in_rate == out_rate {
        return input.to_vec();
    }

    let ratio = out_rate as f64 / in_rate as f64;
    let out_frames = ((in_frames as f64) * ratio).round().max(1.0) as usize;

    // cutoff 相对输入 Nyquist：降采样时取输出带宽，升采样时保持输入带宽。
    let cutoff = ratio.min(1.0);
    let lut = KernelLut::new(cutoff);
    let half_width = lut.half_width;
    let reach = half_width.ceil() as i64;

    let mut out = vec![0.0f32; out_frames * channels];
    let mut acc = vec![0.0f64; channels];

    for of in 0..out_frames {
        // 输出样本在输入域中的位置（单位：输入样本）
        let t = of as f64 / ratio;
        let center = t.floor() as i64;

        for v in acc.iter_mut() {
            *v = 0.0;
        }
        let mut wsum = 0.0f64;

        let lo = center - reach;
        let hi = center + reach;
        for i in lo..=hi {
            // 越界抽头直接跳过：等价于对有效抽头重新归一化，
            // 比零填充更接近对称延拓，边缘不衰减。
            if i < 0 || i >= in_frames as i64 {
                continue;
            }
            let w = lut.eval(t - i as f64) as f64;
            if w == 0.0 {
                continue;
            }
            wsum += w;
            let base = i as usize * channels;
            for ch in 0..channels {
                acc[ch] += input[base + ch] as f64 * w;
            }
        }

        let out_base = of * channels;
        if wsum.abs() > 1e-12 {
            let inv = 1.0 / wsum;
            for ch in 0..channels {
                out[out_base + ch] = (acc[ch] * inv) as f32;
            }
        }
        // wsum == 0 只可能在没有有效抽头时发生，此时保持 0（静音）。
    }

    out
}

/// 单声道便捷封装。
#[allow(dead_code)]
pub(crate) fn resample_mono(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    resample_interleaved(input, 1, in_rate, out_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(x: &[f32]) -> f64 {
        if x.is_empty() {
            return 0.0;
        }
        let s: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        (s / x.len() as f64).sqrt()
    }

    fn sine(rate: u32, freq: f64, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|i| {
                (2.0 * PI * freq * i as f64 / rate as f64).sin() as f32
            })
            .collect()
    }

    #[test]
    fn identity_when_rates_equal() {
        let input = vec![0.1f32, -0.2, 0.3, -0.4];
        let out = resample_interleaved(&input, 2, 48000, 48000);
        assert_eq!(out, input);
    }

    #[test]
    fn empty_and_degenerate_inputs() {
        assert!(resample_interleaved(&[], 2, 48000, 44100).is_empty());
        assert!(resample_interleaved(&[0.5, 0.5], 0, 48000, 44100).is_empty());
        // 帧数 < 2 原样返回
        let one_frame = vec![0.5f32, 0.25];
        assert_eq!(resample_interleaved(&one_frame, 2, 48000, 44100), one_frame);
    }

    #[test]
    fn output_frame_count_matches_ratio() {
        let input = vec![0.0f32; 9600]; // 100 ms @ 96k, mono
        let out = resample_interleaved(&input, 1, 96000, 44100);
        let expected = (9600.0 * 44100.0 / 96000.0f64).round() as usize;
        assert_eq!(out.len(), expected);
    }

    #[test]
    fn dc_is_preserved_interior() {
        let input = vec![0.5f32; 4800];
        let out = resample_interleaved(&input, 1, 96000, 44100);
        // 跳过边缘（有效抽头重新归一化的过渡区）
        let interior = &out[64..out.len() - 64];
        for v in interior {
            assert!(
                (v - 0.5).abs() < 1e-3,
                "DC not preserved: got {v}, expected 0.5"
            );
        }
    }

    #[test]
    fn in_band_tone_amplitude_is_preserved() {
        // 1 kHz 远低于输出 Nyquist，应基本无衰减
        let input = sine(96000, 1000.0, 9600);
        let out = resample_interleaved(&input, 1, 96000, 44100);
        let interior = &out[256..out.len() - 256];
        let r = rms(interior);
        let expected = 0.5f64.sqrt(); // 单位正弦的 RMS ≈ 0.7071
        assert!(
            (r - expected).abs() < 0.02,
            "in-band tone amplitude drifted: rms={r}, expected≈{expected}"
        );
    }

    #[test]
    fn downsampling_suppresses_out_of_band_tone() {
        // 96 kHz 源上的 30 kHz 分量：远高于 44.1 kHz 的 22.05 kHz Nyquist。
        // 无抗混叠时会折叠到 |30000 - 44100| = 14.1 kHz 且保留大部分能量；
        // 有抗混叠时应被强衰减。
        let input = sine(96000, 30000.0, 9600);
        let out = resample_interleaved(&input, 1, 96000, 44100);
        let interior = &out[256..out.len() - 256];
        let r = rms(interior);
        assert!(
            r < 0.02,
            "out-of-band tone was not attenuated (aliasing): rms={r}"
        );
    }

    #[test]
    fn linear_interpolation_would_have_aliased() {
        // 对照组：直接线性插值（无低通）在同一输入上会保留大量能量，
        // 说明上面的断言确实在检验抗混叠行为而非其他因素。
        let input = sine(96000, 30000.0, 9600);
        let in_frames = input.len();
        let ratio = 44100.0f64 / 96000.0f64;
        let out_frames = ((in_frames as f64) * ratio).round() as usize;
        let mut naive = Vec::with_capacity(out_frames);
        for of in 0..out_frames {
            let t = of as f64 / ratio;
            let i0 = (t.floor() as usize).min(in_frames - 1);
            let i1 = (i0 + 1).min(in_frames - 1);
            let frac = (t - i0 as f64) as f32;
            naive.push(input[i0] + (input[i1] - input[i0]) * frac);
        }
        let r = rms(&naive[256..naive.len() - 256]);
        assert!(
            r > 0.1,
            "control group expected strong aliasing energy, got rms={r}"
        );
    }

    #[test]
    fn stereo_channels_stay_independent() {
        // 左声道 DC 0.5，右声道 DC -0.25
        let mut input = Vec::new();
        for _ in 0..4800 {
            input.push(0.5f32);
            input.push(-0.25f32);
        }
        let out = resample_interleaved(&input, 2, 96000, 44100);
        let frames = out.len() / 2;
        for f in 64..frames - 64 {
            assert!((out[f * 2] - 0.5).abs() < 1e-3);
            assert!((out[f * 2 + 1] + 0.25).abs() < 1e-3);
        }
    }

    #[test]
    fn upsampling_preserves_dc_and_length() {
        let input = vec![-0.3f32; 1000];
        let out = resample_interleaved(&input, 1, 44100, 96000);
        let expected = (1000.0 * 96000.0 / 44100.0f64).round() as usize;
        assert_eq!(out.len(), expected);
        let interior = &out[64..out.len() - 64];
        for v in interior {
            assert!((v + 0.3).abs() < 1e-3, "DC not preserved on upsampling: {v}");
        }
    }
}
