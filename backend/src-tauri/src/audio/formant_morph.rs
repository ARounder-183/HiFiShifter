/*
 * formant_morph.rs - Clip 级共振峰变形（STFT 包络扭曲版本）。
 *
 * 主要内容：
 * - apply_formant_morph_mono / apply_formant_morph_interleaved：稳定公开 API。
 * - vowel_formant_preset：元音预设（保留供 IPC 调用，不参与本模块算法）。
 * - 内部走 STFT + 倒谱包络 + F1/F2 峰位检测 + 频率扭曲映射的处理流。
 *
 * 与其他模块的关系：
 * - 由 formant_cache.rs 在重建 clip 缓存时调用。
 * - 由 audio_engine/snapshot.rs / commands/playback.rs 间接消费缓存结果。
 * - 公开签名与 2026-04-30 设计 spec 保持兼容；本模块在 2026-06-30 整体由
 *   "LPC 极点迁移"路线重写为"包络变形"路线，目标是保留原音色身份的同时
 *   能可靠把 F1/F2 拽到目标位置。
 *
 * 算法要点（重写后）：
 * 1. 帧分析：50% 重叠 Hann 窗，FFT_SIZE = 2048（采样率 ≥ 24k）/ 1024（其它），
 *    hop = FFT_SIZE / 4，相位连续性来自 75% 重叠的 OLA。
 * 2. 倒谱包络：log|X| 做 IFFT，仅保留 lifter 内的低 quefrency 部分（描述
 *    声道滤波器形状），再 FFT 回频域得到平滑包络 E(k)。激励 X(k)/E(k)
 *    完全保留（声带振动与高频音色身份不被触碰，这是"保音色"的关键）。
 * 3. F1/F2 检测：在包络上分别于 [F1_SEARCH_LO, F1_SEARCH_HI] 与
 *    [F2_SEARCH_LO, F2_SEARCH_HI] 区间寻找局部最大，要求峰值显著高于
 *    周围谷点，否则该帧记为低置信度并降权。
 * 4. 频率扭曲映射 w(f)：分段单调扭曲。0Hz、F1、F2、上界锚点 → 目标位置
 *    的对应锚点；高频区（>HIGH_FIX_HZ）保持恒等不动。锚点之间用单调
 *    PCHIP 三次插值，保证频率轴单调（不翻转）且过渡平滑。
 * 5. 新包络：E'(k) = E(w⁻¹(f_k))。逆映射用线性反查实现（包络足够平滑，
 *    误差可忽略，避免实现复杂的 Hermite 反函数）。
 * 6. 替换包络：Y(k) = X(k) * (E'(k)/E(k))^strength_eff。
 *    - strength_eff = strength * voicedness_weight，对清辅音/低能量帧自动降权。
 *    - 用幂指数混合而非线性 lerp，可保证 strength=0 严格 bypass，且响度
 *      变化更平滑。
 * 7. iSTFT：复频谱 IFFT → 加 Hann 窗（合成窗）→ overlap-add → 用 window²
 *    之和归一化。
 * 8. 整体峰值保护：output_peak / input_peak 比值受限。
 *
 * 维护说明：
 * - 严禁在内部对 X(k) 的相位做修改（否则会破坏原音色的 fine structure）。
 * - 包络 E(k) 必须加 floor 防爆零，且 floor 要在 log 域而不是线性域处理。
 * - 任何对 sample_rate / strength 极端值的入口校验，必须早于 FFT 分配。
 */
use crate::state::ClipFormantMorph;
use num_complex::Complex32;
use rustfft::FftPlanner;

// ── 公开常量（供子模块/测试参考） ────────────────────────────────────────

/// 低于该采样率直接 bypass：低于 8kHz 的素材本身就没有可靠的 F2 信息。
const MIN_SAMPLE_RATE: u32 = 8_000;
/// 输入样本不足直接 bypass：FFT 都做不满一帧。
const MIN_INPUT_SAMPLES: usize = 512;
/// strength 低于此阈值视为关闭（避免极小浮点误差触发处理）。
const STRENGTH_EPS: f32 = 1.0e-5;

/// F1 搜索区间（用于在包络上寻找当前 F1 峰位）。
const F1_SEARCH_LO_HZ: f32 = 200.0;
const F1_SEARCH_HI_HZ: f32 = 1_100.0;
/// F2 搜索区间。
const F2_SEARCH_LO_HZ: f32 = 700.0;
const F2_SEARCH_HI_HZ: f32 = 2_900.0;

/// 高于此频率的部分扭曲映射保持恒等：保留 F3 / F4 / spectral tilt（这是音色身份）。
const HIGH_FIX_HZ: f32 = 3_800.0;

/// 倒谱 lifter cutoff（quefrency bin 索引比例）：保留前 ~12% 系数描述包络。
/// 该值越小 → 包络越平滑（更不易受 F0 谐波污染），但太小会模糊 F1/F2 细节。
const LIFTER_CUTOFF_RATIO: f32 = 0.12;

/// 包络下限（线性幅度）：避免除零放大噪声。
const ENVELOPE_FLOOR: f32 = 1.0e-4;

/// 输出整体峰值上限相对输入峰值的最大放大倍数。
const OUTPUT_PEAK_RATIO_LIMIT: f32 = 1.6;

/// 元音 → 目标共振峰预设（F1, F2，单位 Hz）。
///
/// 保留供前端 / IPC 在不知道精确共振峰参数时使用；本模块算法本身只看
/// `params.target_f1_hz / target_f2_hz`，与本表无直接耦合。
#[allow(dead_code)]
pub fn vowel_formant_preset(vowel: &str) -> Option<(f64, f64)> {
    match vowel.trim().to_ascii_lowercase().as_str() {
        "a" | "aa" | "ah" | "啊" | "あ" | "ア" => Some((800.0, 1_200.0)),
        "e" | "eh" | "诶" | "欸" | "え" | "エ" => Some((500.0, 1_900.0)),
        "i" | "ee" | "yi" | "衣" | "い" | "イ" => Some((300.0, 2_300.0)),
        "o" | "oh" | "哦" | "お" | "オ" => Some((500.0, 900.0)),
        "u" | "oo" | "wu" | "乌" | "う" | "ウ" => Some((350.0, 750.0)),
        _ => None,
    }
}

// ── 公开入口 ────────────────────────────────────────────────────────────

/// 单声道 PCM 共振峰变形（公开 API，签名与重写前保持一致）。
///
/// 流程：
/// 1. 入口校验（disabled / 空输入 / 低采样率 / 短样本 / strength 接近 0）→ 直接 bypass。
/// 2. 选择 FFT 尺寸：≥24kHz 用 2048，否则用 1024。hop = FFT/4 实现 75% 重叠。
/// 3. 构建分析/合成 Hann 窗（同一窗，OLA 用 window² 归一化）。
/// 4. 计划 FFT / IFFT（rustfft，复用 planner）。
/// 5. 帧循环（步进 hop）：
///    - 取帧 + 加窗 → FFT 得到复频谱 X(k)。
///    - log|X| → IFFT → lifter → FFT → 包络 E(k)。
///    - 包络上检测 F1 / F2 峰位（带置信度）。
///    - 构造频率扭曲映射 w(f) 并采样得 E'(k)。
///    - Y(k) = X(k) * (E'(k)/E(k))^effective_strength。
///    - IFFT(Y) → 加合成窗 → 累加到 OLA buffer。
/// 6. 用 window_sum 归一化 OLA buffer，截断到原长。
/// 7. 整体峰值保护（不超过 input_peak * OUTPUT_PEAK_RATIO_LIMIT）。
///
/// 参数说明：
/// - `input`：mono PCM。
/// - `sample_rate`：采样率（Hz）。
/// - `params`：用户指定的目标 F1 / F2 / strength。
///
/// 返回：长度与 input 一致的处理后 PCM。失败 / 不适用情况下返回原始 PCM 拷贝
/// （保持调用方"总能拿到等长输出"的契约）。
pub fn apply_formant_morph_mono(
    input: &[f32],
    sample_rate: u32,
    params: &ClipFormantMorph,
) -> Result<Vec<f32>, String> {
    if !params.enabled || input.is_empty() {
        return Ok(input.to_vec());
    }
    if sample_rate < MIN_SAMPLE_RATE || input.len() < MIN_INPUT_SAMPLES {
        return Ok(input.to_vec());
    }

    let strength = (params.strength as f32).clamp(0.0, 1.0);
    if strength <= STRENGTH_EPS {
        return Ok(input.to_vec());
    }

    let target_f1 = (params.target_f1_hz as f32).clamp(180.0, 1_200.0);
    let target_f2 = (params.target_f2_hz as f32).clamp(target_f1 + 250.0, 3_200.0);

    let fft_size = if sample_rate >= 24_000 { 2048 } else { 1024 };
    let hop = fft_size / 4;
    let half = fft_size / 2 + 1;

    let analysis_window = hann_window(fft_size);
    let synthesis_window = analysis_window.clone();

    let mut planner = FftPlanner::<f32>::new();
    let fft_forward = planner.plan_fft_forward(fft_size);
    let fft_inverse = planner.plan_fft_inverse(fft_size);

    // 在样本前后各 pad 一段，让首尾帧也能被完整 OLA 覆盖。
    let pad_left = fft_size - hop;
    let pad_right = fft_size;
    let mut padded = vec![0.0_f32; pad_left + input.len() + pad_right];
    padded[pad_left..pad_left + input.len()].copy_from_slice(input);

    let mut ola = vec![0.0_f32; padded.len()];
    let mut win_sum = vec![0.0_f32; padded.len()];

    // 复用每帧 buffer，避免循环内堆分配
    let mut frame_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); fft_size];
    let mut ifft_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); fft_size];
    let mut envelope: Vec<f32> = vec![0.0; half];
    let mut warped_envelope: Vec<f32> = vec![0.0; half];
    let mut log_mag: Vec<f32> = vec![0.0; fft_size];
    let mut cepstrum: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); fft_size];

    let lifter_cutoff = ((fft_size as f32 * LIFTER_CUTOFF_RATIO) as usize).max(8);

    let mut start = 0usize;
    while start + fft_size <= padded.len() {
        // 1. 取帧 + 加窗 → 复数缓冲
        let mut frame_energy_lin = 0.0_f32;
        for i in 0..fft_size {
            let s = padded[start + i] * analysis_window[i];
            frame_buf[i] = Complex32::new(s, 0.0);
            frame_energy_lin += s * s;
        }

        // 极低能量帧：跳过包络估计，直接把加窗后的帧 OLA（保持原信号）。
        if frame_energy_lin < 1.0e-10 {
            for i in 0..fft_size {
                ola[start + i] += frame_buf[i].re * synthesis_window[i];
                win_sum[start + i] += synthesis_window[i] * synthesis_window[i];
            }
            start += hop;
            continue;
        }

        // 2. FFT
        fft_forward.process(&mut frame_buf);

        // 3. 包络估计（cepstral lifter）
        // log|X| 在 [0, fft_size) 全段：lifter 在时域是对称低通，因此需要全段 log|X|。
        for i in 0..fft_size {
            let mag = frame_buf[i].norm().max(ENVELOPE_FLOOR);
            log_mag[i] = mag.ln();
        }
        for (dst, src) in cepstrum.iter_mut().zip(log_mag.iter()) {
            *dst = Complex32::new(*src, 0.0);
        }
        fft_inverse.process(&mut cepstrum);
        // lifter：保留 quefrency 中心区，置零其余部分。注意 IFFT 的归一化在最后统一处理。
        for i in 0..fft_size {
            let in_low = i < lifter_cutoff;
            let in_high = i >= fft_size - lifter_cutoff;
            if !(in_low || in_high) {
                cepstrum[i] = Complex32::new(0.0, 0.0);
            }
        }
        // 回到频域，得到 log 域的平滑包络
        fft_forward.process(&mut cepstrum);
        // rustfft 不归一化：FFT(IFFT(x)) = N * x，因此除以 N。
        let inv_n = 1.0 / fft_size as f32;
        for i in 0..half {
            let log_env = cepstrum[i].re * inv_n;
            envelope[i] = log_env.exp().max(ENVELOPE_FLOOR);
        }

        // 4. 在包络上检测 F1 / F2
        let bin_hz = sample_rate as f32 / fft_size as f32;
        let detection = detect_f1_f2(&envelope, bin_hz);

        // 5. 构造频率扭曲映射并采样新包络
        build_warped_envelope(
            &envelope,
            &mut warped_envelope,
            bin_hz,
            detection.f1_hz,
            detection.f2_hz,
            target_f1,
            target_f2,
        );

        // 6. 频谱包络替换：Y(k) = X(k) * (E'(k)/E(k))^effective_strength
        let voicedness = detection.confidence;
        let effective_strength = (strength * voicedness).clamp(0.0, 1.0);
        if effective_strength > STRENGTH_EPS {
            for i in 0..half {
                let ratio = (warped_envelope[i] / envelope[i]).max(1.0e-6);
                let scale = ratio.powf(effective_strength);
                frame_buf[i] *= scale;
            }
            // 共轭对称：bin [half..fft_size) 是 [1..half-1] 的镜像
            for i in 1..half - 1 {
                frame_buf[fft_size - i] = frame_buf[i].conj();
            }
        }

        // 7. iFFT 回时域
        ifft_buf.copy_from_slice(&frame_buf);
        fft_inverse.process(&mut ifft_buf);

        // 8. 加合成窗 + OLA。rustfft 不归一化，因此 IFFT 输出需要除以 N。
        for i in 0..fft_size {
            let s = ifft_buf[i].re * inv_n * synthesis_window[i];
            ola[start + i] += s;
            win_sum[start + i] += synthesis_window[i] * synthesis_window[i];
        }

        start += hop;
    }

    // 9. window_sum 归一化
    for (s, w) in ola.iter_mut().zip(win_sum.iter()) {
        if *w > 1.0e-8 {
            *s /= *w;
        }
    }

    // 10. 截取与输入等长的部分
    let mut out = vec![0.0_f32; input.len()];
    out.copy_from_slice(&ola[pad_left..pad_left + input.len()]);

    // 11. 整体峰值保护（限制相对输入的最大放大倍数）
    let in_peak = peak_abs(input).max(1.0e-6);
    let out_peak = peak_abs(&out).max(1.0e-6);
    let limit = in_peak * OUTPUT_PEAK_RATIO_LIMIT;
    if out_peak > limit {
        let gain = limit / out_peak;
        for s in out.iter_mut() {
            *s *= gain;
        }
    }
    // 兜底：硬限幅
    for s in out.iter_mut() {
        if !s.is_finite() {
            *s = 0.0;
        } else if *s > 0.99 {
            *s = 0.99;
        } else if *s < -0.99 {
            *s = -0.99;
        }
    }

    Ok(out)
}

/// 多声道 PCM 共振峰变形（公开 API，签名与重写前保持一致）。
///
/// 行为约定：
/// - channels == 0 → 错误。
/// - channels == 1 → 直接走 mono 路径。
/// - channels >= 2 → 取通道平均得到 mono 分析信号、跑 mono 算法、然后把
///   `wet - dry` delta 加回每个原通道（保留通道间相对差与立体声成像）。
///
/// 这种"delta 同步"策略保证：单通道与多通道在中心声像内容上听感一致，
/// 而不会因为多通道独立分析导致 F1/F2 检测偏差或相位不同步。
pub fn apply_formant_morph_interleaved(
    input: &[f32],
    sample_rate: u32,
    channels: usize,
    params: &ClipFormantMorph,
) -> Result<Vec<f32>, String> {
    if channels == 0 {
        return Err("channels == 0".to_string());
    }
    if channels == 1 {
        return apply_formant_morph_mono(input, sample_rate, params);
    }
    if input.is_empty() || !params.enabled {
        return Ok(input.to_vec());
    }
    let frames = input.len() / channels;
    if frames == 0 {
        return Ok(input.to_vec());
    }

    let mono = average_channels_to_mono(input, channels, frames);
    let processed_mono = apply_formant_morph_mono(&mono, sample_rate, params)?;

    Ok(apply_mono_delta_to_interleaved(
        input,
        channels,
        &mono,
        &processed_mono,
    ))
}

// ── 内部辅助 ────────────────────────────────────────────────────────────

/// F1 / F2 检测结果。confidence ∈ [0, 1]，越小表示该帧越不像清晰元音。
struct FormantDetection {
    f1_hz: f32,
    f2_hz: f32,
    confidence: f32,
}

/// 在已估计的包络上检测 F1 / F2。
///
/// 流程：
/// 1. 在 [F1_SEARCH_LO_HZ, F1_SEARCH_HI_HZ] 内找包络的局部最大值（要求两侧
///    bin 都更低）。失败则用搜索区间中点作为 fallback。
/// 2. 在 [F2_SEARCH_LO_HZ, F2_SEARCH_HI_HZ] 内同样找局部最大，并要求 > F1 + 250Hz。
/// 3. 置信度：基于"峰高 / 局部谷高"对数比，clamp 到 [0, 1]。比值越大说明
///    共振峰越清晰，越是元音材料；清辅音 / 鼻音段比值小，自动降权。
///
/// 参数：`bin_hz` = sample_rate / fft_size。
fn detect_f1_f2(envelope: &[f32], bin_hz: f32) -> FormantDetection {
    let half = envelope.len();
    let bin_of = |hz: f32| -> usize {
        ((hz / bin_hz) as usize).clamp(1, half.saturating_sub(2))
    };

    let f1_lo = bin_of(F1_SEARCH_LO_HZ);
    let f1_hi = bin_of(F1_SEARCH_HI_HZ);
    let f2_lo = bin_of(F2_SEARCH_LO_HZ);
    let f2_hi = bin_of(F2_SEARCH_HI_HZ);

    let f1_bin = find_local_peak(envelope, f1_lo, f1_hi).unwrap_or((f1_lo + f1_hi) / 2);
    let f2_min = (f1_bin + ((250.0 / bin_hz).round() as usize)).max(f2_lo);
    let f2_bin = find_local_peak(envelope, f2_min, f2_hi).unwrap_or((f2_min + f2_hi) / 2);

    let f1_hz = f1_bin as f32 * bin_hz;
    let f2_hz = f2_bin as f32 * bin_hz;

    // 置信度：用 F1 峰高 / F1 周边最低点的对数比作为衡量
    let valley_lo = envelope[f1_lo.saturating_sub(0)..f1_bin]
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min)
        .max(ENVELOPE_FLOOR);
    let peak_val = envelope[f1_bin].max(ENVELOPE_FLOOR);
    let prominence = (peak_val / valley_lo).ln().max(0.0);
    // 0.05 (≈ 5%) → 置信度 0；ln(2.5) ≈ 0.916 → 置信度 1。
    let confidence = (prominence / 0.916).clamp(0.0, 1.0);

    FormantDetection {
        f1_hz,
        f2_hz,
        confidence,
    }
}

/// 在 envelope[lo..=hi] 闭区间找一个严格局部最大（两侧均更低）。
/// 找不到返回 None。
fn find_local_peak(envelope: &[f32], lo: usize, hi: usize) -> Option<usize> {
    if hi <= lo + 1 || hi >= envelope.len() {
        return None;
    }
    let mut best: Option<(usize, f32)> = None;
    for i in (lo + 1)..hi {
        let v = envelope[i];
        if v > envelope[i - 1] && v > envelope[i + 1] {
            match best {
                Some((_, bv)) if bv >= v => {}
                _ => best = Some((i, v)),
            }
        }
    }
    best.map(|(idx, _)| idx)
}

/// 构造扭曲后的包络 E'(k) = E(w⁻¹(f_k))，其中 w 是把
///   (0, F1_src, F2_src, HIGH_FIX) → (0, F1_tgt, F2_tgt, HIGH_FIX)
/// 的分段单调映射。
///
/// 实现策略：
/// - 不需要显式构造 w(f)，而是直接构造 w⁻¹：把"目标频率"映射到"源频率"。
/// - 在 [0, target_f1] 区间：源 = 0..src_f1 线性。
/// - 在 [target_f1, target_f2] 区间：源 = src_f1..src_f2 线性。
/// - 在 [target_f2, HIGH_FIX] 区间：源 = src_f2..HIGH_FIX 线性。
/// - 在 [HIGH_FIX, Nyquist] 区间：源 = 目标（恒等，保护高频音色身份）。
///
/// 这样保证频率轴单调（永不交叉），且 F1/F2 锚点精确到位。线性段在 log 域
/// 听感平滑（人耳频率分辨在中频段近似线性 / mel），再用 PCHIP 反而会引入
/// 额外形状改变，因此采用最朴素的分段线性。
fn build_warped_envelope(
    src_envelope: &[f32],
    dst_envelope: &mut [f32],
    bin_hz: f32,
    src_f1_hz: f32,
    src_f2_hz: f32,
    target_f1_hz: f32,
    target_f2_hz: f32,
) {
    let half = src_envelope.len();
    let nyquist_hz = bin_hz * (half - 1) as f32;
    let high_fix = HIGH_FIX_HZ.min(nyquist_hz - bin_hz);

    // 锚点：(target_freq → source_freq) 的分段映射。
    // 必须保持目标频率严格单调递增，且不会越界。
    let p0 = (0.0_f32, 0.0_f32);
    let p1 = (
        target_f1_hz.clamp(bin_hz, high_fix - 2.0 * bin_hz),
        src_f1_hz.clamp(bin_hz, high_fix - 2.0 * bin_hz),
    );
    let p2_target = target_f2_hz.clamp(p1.0 + bin_hz, high_fix - bin_hz);
    let p2_source = src_f2_hz.clamp(p1.1 + bin_hz, high_fix - bin_hz);
    let p2 = (p2_target, p2_source);
    let p3 = (high_fix, high_fix);

    let lerp = |x: f32, x0: f32, x1: f32, y0: f32, y1: f32| -> f32 {
        if (x1 - x0).abs() < 1.0e-6 {
            y0
        } else {
            y0 + (y1 - y0) * ((x - x0) / (x1 - x0))
        }
    };

    for k in 0..half {
        let target_hz = k as f32 * bin_hz;
        let source_hz = if target_hz <= p0.0 {
            p0.1
        } else if target_hz <= p1.0 {
            lerp(target_hz, p0.0, p1.0, p0.1, p1.1)
        } else if target_hz <= p2.0 {
            lerp(target_hz, p1.0, p2.0, p1.1, p2.1)
        } else if target_hz <= p3.0 {
            lerp(target_hz, p2.0, p3.0, p2.1, p3.1)
        } else {
            // 高频区恒等
            target_hz
        };

        // 在源包络上做线性插值
        let src_bin_f = (source_hz / bin_hz).clamp(0.0, (half - 1) as f32);
        let lo = src_bin_f.floor() as usize;
        let hi = (lo + 1).min(half - 1);
        let frac = src_bin_f - lo as f32;
        dst_envelope[k] = src_envelope[lo] * (1.0 - frac) + src_envelope[hi] * frac;
    }
}

fn hann_window(len: usize) -> Vec<f32> {
    if len <= 1 {
        return vec![1.0; len];
    }
    let denom = (len - 1) as f32;
    (0..len)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / denom).cos())
        .collect()
}

fn peak_abs(input: &[f32]) -> f32 {
    input.iter().fold(0.0_f32, |p, s| p.max(s.abs()))
}

fn average_channels_to_mono(input: &[f32], channels: usize, frames: usize) -> Vec<f32> {
    let mut mono = vec![0.0_f32; frames];
    let inv_ch = 1.0 / channels as f32;
    for frame_idx in 0..frames {
        let mut sum = 0.0_f32;
        for ch in 0..channels {
            sum += input[frame_idx * channels + ch];
        }
        mono[frame_idx] = sum * inv_ch;
    }
    mono
}

fn apply_mono_delta_to_interleaved(
    input: &[f32],
    channels: usize,
    dry_mono: &[f32],
    wet_mono: &[f32],
) -> Vec<f32> {
    let frames = dry_mono.len().min(wet_mono.len());
    let mut out = input.to_vec();
    for frame_idx in 0..frames {
        let delta = wet_mono[frame_idx] - dry_mono[frame_idx];
        for ch in 0..channels {
            let idx = frame_idx * channels + ch;
            let v = input[idx] + delta;
            out[idx] = if !v.is_finite() {
                0.0
            } else {
                v.clamp(-0.99, 0.99)
            };
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────
// 测试
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params(enabled: bool, strength: f64) -> ClipFormantMorph {
        ClipFormantMorph {
            enabled,
            target_f1_hz: 700.0,
            target_f2_hz: 1_400.0,
            strength,
        }
    }

    /// 合成一个稳态元音样本：基频 + 共振峰加权的若干谐波。
    /// 采样率默认 48kHz，时长 0.5s。
    fn synth_vowel(f0_hz: f32, f1_hz: f32, f2_hz: f32, sr: u32, secs: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        let mut out = vec![0.0_f32; n];
        // 简单做法：在 F1、F2 附近用窄带阻尼调谐振，然后给 F0 谐波叠加
        let harmonics = 30;
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let mut s = 0.0_f32;
            for h in 1..=harmonics {
                let freq = f0_hz * h as f32;
                if freq > sr as f32 * 0.45 {
                    break;
                }
                // 共振峰加权：距离 F1/F2 越近振幅越大
                let d1 = (freq - f1_hz).abs() / 200.0;
                let d2 = (freq - f2_hz).abs() / 250.0;
                let amp = (-d1 * d1).exp() + 0.6 * (-d2 * d2).exp() + 0.05;
                s += amp * (2.0 * std::f32::consts::PI * freq * t).sin();
            }
            out[i] = s * 0.05;
        }
        out
    }

    #[test]
    fn disabled_is_strict_bypass() {
        let input: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.001).sin()).collect();
        let params = default_params(false, 1.0);
        let out = apply_formant_morph_mono(&input, 48_000, &params).unwrap();
        assert_eq!(out, input, "disabled must be byte-identical bypass");
    }

    #[test]
    fn zero_strength_is_strict_bypass() {
        let input: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.001).sin()).collect();
        let params = default_params(true, 0.0);
        let out = apply_formant_morph_mono(&input, 48_000, &params).unwrap();
        assert_eq!(out, input, "strength=0 must be byte-identical bypass");
    }

    #[test]
    fn empty_input_returns_empty() {
        let input: Vec<f32> = vec![];
        let params = default_params(true, 1.0);
        let out = apply_formant_morph_mono(&input, 48_000, &params).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn low_sample_rate_is_bypass() {
        let input: Vec<f32> = vec![0.0; 2048];
        let params = default_params(true, 1.0);
        let out = apply_formant_morph_mono(&input, 4_000, &params).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn output_is_finite_and_length_preserving() {
        let input = synth_vowel(150.0, 800.0, 1_200.0, 48_000, 0.3);
        let params = default_params(true, 0.7);
        let out = apply_formant_morph_mono(&input, 48_000, &params).unwrap();
        assert_eq!(out.len(), input.len());
        for s in &out {
            assert!(s.is_finite(), "output must be finite");
            assert!(s.abs() <= 1.0, "output must be within [-1, 1]");
        }
    }

    #[test]
    fn voiced_input_changes_audibly() {
        // 输入是 a 元音 (F1=800, F2=1200)，目标 i 元音 (F1=300, F2=2300)。
        // 要求处理后与原信号有可测的差异。
        let input = synth_vowel(150.0, 800.0, 1_200.0, 48_000, 0.3);
        let params = ClipFormantMorph {
            enabled: true,
            target_f1_hz: 300.0,
            target_f2_hz: 2_300.0,
            strength: 0.8,
        };
        let out = apply_formant_morph_mono(&input, 48_000, &params).unwrap();
        let diff: f32 = input
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / input.len() as f32;
        assert!(diff > 1.0e-3, "audible difference expected, got diff={diff}");
    }

    #[test]
    fn stronger_strength_yields_larger_diff() {
        let input = synth_vowel(150.0, 800.0, 1_200.0, 48_000, 0.3);
        let mk = |s: f64| ClipFormantMorph {
            enabled: true,
            target_f1_hz: 300.0,
            target_f2_hz: 2_300.0,
            strength: s,
        };
        let weak = apply_formant_morph_mono(&input, 48_000, &mk(0.2)).unwrap();
        let strong = apply_formant_morph_mono(&input, 48_000, &mk(0.9)).unwrap();
        let diff_of = |out: &[f32]| -> f32 {
            input
                .iter()
                .zip(out.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / input.len() as f32
        };
        let weak_d = diff_of(&weak);
        let strong_d = diff_of(&strong);
        assert!(
            strong_d > weak_d * 1.15,
            "expected stronger morph to differ more; weak={weak_d} strong={strong_d}"
        );
    }

    #[test]
    fn silent_input_stays_silent() {
        let input = vec![0.0_f32; 4096];
        let params = default_params(true, 1.0);
        let out = apply_formant_morph_mono(&input, 48_000, &params).unwrap();
        assert_eq!(out.len(), input.len());
        let peak = peak_abs(&out);
        assert!(peak < 1.0e-4, "silent input must stay silent, peak={peak}");
    }

    #[test]
    fn interleaved_stereo_matches_length_and_finite() {
        let mono = synth_vowel(150.0, 800.0, 1_200.0, 48_000, 0.2);
        let mut stereo = Vec::with_capacity(mono.len() * 2);
        for &s in &mono {
            stereo.push(s);
            stereo.push(s * 0.95);
        }
        let params = default_params(true, 0.5);
        let out = apply_formant_morph_interleaved(&stereo, 48_000, 2, &params).unwrap();
        assert_eq!(out.len(), stereo.len());
        for s in &out {
            assert!(s.is_finite());
            assert!(s.abs() <= 1.0);
        }
    }

    #[test]
    fn vowel_preset_table_returns_known_values() {
        assert_eq!(vowel_formant_preset("a"), Some((800.0, 1_200.0)));
        assert_eq!(vowel_formant_preset("ee"), Some((300.0, 2_300.0)));
        assert!(vowel_formant_preset("xyz").is_none());
    }
}