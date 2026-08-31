/*
 * formant_morph/mod.rs - Clip 级共振峰迁移（LPC 极点迁移 + STFT 频域比值滤波）。
 *
 * 主要内容：
 * - apply_formant_morph_mono / apply_formant_morph_interleaved：稳定公开 API
 *   （签名与 2026-04-30 设计 spec 保持兼容，调用方零改动）。
 * - vowel_formant_preset：元音预设（保留供 IPC 调用，不参与本模块算法）。
 * - 共享常量表（分析速率 / LPC 阶数 / 候选区间 / 限幅等，调参入口集中在此）。
 *
 * 与其他模块的关系：
 * - 由 formant_cache.rs 在重建 clip 缓存时调用。
 * - 由 audio_engine/snapshot.rs / commands/playback.rs 间接消费缓存结果。
 * - audio/mixdown.rs（导出）同样经 formant_cache 间接消费。
 * - 子模块分工：
 *   - decimator.rs  抗混叠降采样到分析域
 *   - lpc.rs        自相关 / Levinson-Durbin / Durand-Kerner 求根 / 极点换算
 *   - track.rs      共振峰候选筛选与轨迹平滑
 *   - correction.rs 目标极点 / 频域比值 H(k) / 浊音 gate / 能量对齐 / 软限幅
 *   - analysis.rs   供 IPC 的源共振峰统计（与 DSP 共用同一套分析代码）
 *
 * 算法要点（2026-08-31 重写，替代旧 STFT 倒谱包络扭曲路线）：
 * 1. 分析域降采样到 ~11 kHz + 低阶 LPC(order 12)：让极点是真共振峰而非谐波
 *    （旧路线 R1：倒谱 lifter 分辨率 200-400 Hz，校正滤波器无峰/谷结构 → 搬不动）。
 * 2. 候选筛选（频率/带宽/间距/显著性）失败则该帧恒等直通，
 *    不再使用区间中点常量兜底（旧路线 R2：假锚点导致映射几乎不动）。
 * 3. F1/F2 轨迹中值 + 限速 + 插值：消除逐帧抖动（旧路线 R3）。
 * 4. STFT 域施加 H(k) = |A_orig/A_target| 幅频比值，完全保留原相位，
 *    不做帧内 IIR 重合成（规避 2026-04-30 LPC 路线的截断与不稳定）。
 * 5. 逐帧 RMS 对齐 + 软限幅，替代末端硬 clamp。
 *
 * 维护说明：
 * - 公开 API 的"等长输出"契约必须保持：失败/不适用时返回原始 PCM 拷贝。
 * - 任何新的保护逻辑不得引入逐帧乱跳的增益（必须有时间平滑）。
 */

pub(crate) mod analysis;
mod correction;
mod decimator;
mod lpc;
mod track;

use crate::state::ClipFormantMorph;
use correction::VoicingGate;
use decimator::Decimator;
use lpc::FramePoles;

// ── 共享常量 ─────────────────────────────────────────────────────────────

/// 低于该采样率直接 bypass：低于 8kHz 的素材本身就没有可靠的 F2 信息。
const MIN_SAMPLE_RATE: u32 = 8_000;
/// 输入样本不足直接 bypass：连一个分析帧都做不满。
const MIN_INPUT_SAMPLES: usize = 512;
/// strength 低于此阈值视为关闭（避免极小浮点误差触发处理）。
const STRENGTH_EPS: f32 = 1.0e-5;
/// 输出整体峰值上限相对输入峰值的最大放大倍数。
const OUTPUT_PEAK_RATIO_LIMIT: f32 = 1.6;

/// 共振峰分析目标采样率（实际为 in_rate/D，见 decimator.rs）。
const ANALYSIS_TARGET_RATE: f32 = 11_025.0;
/// LPC 阶数：~11 kHz 下 5 kHz 内约 2 极点/共振峰 + 2（Praat 同款配方）。
const LPC_ORDER: usize = 12;
/// 分析帧长（秒）。
const ANALYSIS_FRAME_SEC: f32 = 0.025;
/// 分析帧步进（秒）。
const ANALYSIS_HOP_SEC: f32 = 0.010;

/// 目标 F1/F2 允许范围：与前端 VowelChart 坐标域严格一致
/// （此前前后端范围不一致，图上可点的位置后端够不着）。
const TARGET_F1_MIN: f64 = 250.0;
const TARGET_F1_MAX: f64 = 1_000.0;
const TARGET_F2_MIN: f64 = 540.0;
const TARGET_F2_MAX: f64 = 2_600.0;
/// 目标 F2 − F1 最小间距。
const TARGET_MIN_GAP: f64 = 200.0;

// ── 公开入口 ────────────────────────────────────────────────────────────

/// 单声道 PCM 共振峰迁移（公开 API，签名与重写前保持一致）。
///
/// 流程：
/// 1. 入口校验（disabled / 空输入 / 低采样率 / 短样本 / strength 接近 0）
///    → 直接 bypass；目标参数 clamp 到与前端一致的范围。
/// 2. 分析域：抗混叠降采样 → 逐帧 LPC（order 12）→ 求根 → 极点 →
///    候选筛选（失败帧不参与）→ 轨迹提取 + 中值/限速平滑。
///    整段无任何合格候选 → 严格直通（不经过 STFT 往返）。
/// 3. STFT 域（2048/1024，hop = N/4）：每个 hop 取最近分析帧的
///    A_orig 系数与极点，把 F1/F2 极点按平滑轨迹 + strength 迁移后重建
///    A_target，施加幅频比值 H = |A_orig/A_target|（相位完全保留），
///    增益指数为时间平滑的浊音 gate；随后逐帧 RMS 对齐。
/// 4. iSTFT / OLA（window² 归一化）→ 全局峰值保护 + 软限幅。
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

    let target_f1 = params.target_f1_hz.clamp(TARGET_F1_MIN, TARGET_F1_MAX) as f32;
    let target_f2 = params
        .target_f2_hz
        .clamp(TARGET_F2_MIN, TARGET_F2_MAX)
        .max(params.target_f1_hz + TARGET_MIN_GAP) as f32;

    // ── 1. 分析域：降采样 + 逐帧 LPC / 极点 / 候选 ──────────────────────
    let Some(mut decimator) = Decimator::new(sample_rate, ANALYSIS_TARGET_RATE as u32) else {
        return Ok(input.to_vec());
    };
    let analysis_rate = decimator.analysis_rate();
    let analysis = decimator.process(input);
    let frame_len = (analysis_rate * ANALYSIS_FRAME_SEC).round() as usize;
    let hop_a = (analysis_rate * ANALYSIS_HOP_SEC).round() as usize;
    if frame_len < LPC_ORDER * 2 || analysis.len() < frame_len {
        return Ok(input.to_vec());
    }
    let n_frames = (analysis.len() - frame_len) / hop_a + 1;

    let mut candidates: Vec<Option<track::FormantCandidate>> = Vec::with_capacity(n_frames);
    let mut frame_coeffs: Vec<Vec<f32>> = Vec::with_capacity(n_frames);
    let mut frame_poles: Vec<FramePoles> = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let start = f * hop_a;
        let frame = &analysis[start..start + frame_len];
        let parsed = lpc::analyze_frame(frame, analysis_rate, LPC_ORDER).and_then(|lpc_res| {
            lpc::poly_roots(&lpc::coeffs_to_monic(&lpc_res.coeffs))
                .map(|roots| (lpc_res, lpc::roots_to_poles(&roots, analysis_rate)))
        });
        match parsed {
            Some((lpc_res, poles)) => {
                let cand = track::select_f1_f2(
                    &poles,
                    &lpc_res.coeffs,
                    analysis_rate,
                    lpc_res.residual_ratio,
                );
                candidates.push(cand);
                frame_coeffs.push(lpc_res.coeffs);
                frame_poles.push(poles);
            }
            None => {
                candidates.push(None);
                frame_coeffs.push(Vec::new());
                frame_poles.push(FramePoles::default());
            }
        }
    }

    // 整段没有任何合格共振峰候选 → 无元音素材，严格直通。
    if candidates.iter().all(|c| c.is_none()) {
        return Ok(input.to_vec());
    }

    let mut tracks = track::extract_tracks(&candidates);
    track::smooth_tracks(&mut tracks);

    // ── 2. STFT 域校正 ─────────────────────────────────────────────────
    let fft_size = if sample_rate >= 24_000 { 2_048 } else { 1_024 };
    let hop = fft_size / 4;
    let half = fft_size / 2 + 1;
    let hop_sec = hop as f32 / sample_rate as f32;
    let analysis_dur_sec = analysis.len() as f32 / analysis_rate;

    let analysis_window = hann_window(fft_size);
    let synthesis_window = analysis_window.clone();

    let mut planner = rustfft::FftPlanner::<f32>::new();
    let fft_forward = planner.plan_fft_forward(fft_size);
    let fft_inverse = planner.plan_fft_inverse(fft_size);
    let inv_n = 1.0 / fft_size as f32;

    let pad_left = fft_size - hop;
    let pad_right = fft_size;
    let mut padded = vec![0.0_f32; pad_left + input.len() + pad_right];
    padded[pad_left..pad_left + input.len()].copy_from_slice(input);

    let mut ola = vec![0.0_f32; padded.len()];
    let mut win_sum = vec![0.0_f32; padded.len()];
    let mut frame_buf: Vec<num_complex::Complex32> = vec![num_complex::Complex32::new(0.0, 0.0); fft_size];
    let mut gate = VoicingGate::new();
    let tracks_max = (tracks.len() - 1) as f32;

    let mut start = 0usize;
    while start + fft_size <= padded.len() {
        // 取帧 + 加窗
        let mut frame_energy = 0.0_f32;
        for i in 0..fft_size {
            let s = padded[start + i] * analysis_window[i];
            frame_buf[i] = num_complex::Complex32::new(s, 0.0);
            frame_energy += s * s;
        }

        // 极低能量帧：直接 OLA 原帧（保持原信号）
        if frame_energy < 1.0e-10 {
            for i in 0..fft_size {
                ola[start + i] += frame_buf[i].re * synthesis_window[i];
                win_sum[start + i] += synthesis_window[i] * synthesis_window[i];
            }
            start += hop;
            continue;
        }

        // 帧中心时间 → 分析帧索引
        let t_sec = ((start + fft_size / 2) as f32 - pad_left as f32)
            .max(0.0)
            .min(analysis_dur_sec)
            / sample_rate as f32;
        let af_f = (t_sec / ANALYSIS_HOP_SEC).clamp(0.0, tracks_max);
        let af = (af_f.round() as usize).min(frame_coeffs.len() - 1);

        let gate_val = gate.advance(
            track::interpolate_at(&tracks, af_f).map(|t| t.voiced).unwrap_or(0.0),
            hop_sec,
        );

        let can_process = gate_val > STRENGTH_EPS
            && candidates[af].is_some()
            && !frame_coeffs[af].is_empty();

        if can_process {
            fft_forward.process(&mut frame_buf);

            // 目标极点：仅迁移 F1/F2（按平滑轨迹插值 + strength），其余极点不动
            let cand = candidates[af].unwrap();
            let poles = &frame_poles[af];
            let mut moved_pairs = poles.pairs.clone();
            for p in moved_pairs.iter_mut() {
                if (p.freq_hz - cand.f1.freq_hz).abs() < 1.0 {
                    p.freq_hz = correction::moved_pole_freq(tp_f1(&tracks, af_f), target_f1, strength);
                } else if (p.freq_hz - cand.f2.freq_hz).abs() < 1.0 {
                    p.freq_hz = correction::moved_pole_freq(tp_f2(&tracks, af_f), target_f2, strength);
                }
            }
            let coeffs_target =
                monic_to_analysis(&lpc::poles_to_coeffs(&moved_pairs, &poles.real_roots, analysis_rate));

            // H(k) = |A_orig / A_target|（限幅 + 模型奈奎斯特外衰减）
            let h_db = correction::h_ratio_db(
                &frame_coeffs[af],
                &coeffs_target,
                fft_size,
                sample_rate as f32,
                analysis_rate,
            );

            // Y = X · 10^(H·gate/20)，共轭对称
            for k in 0..half {
                let scale = 10.0_f32.powf(h_db[k] * gate_val / 20.0);
                frame_buf[k] *= scale;
            }
            for k in 1..half - 1 {
                frame_buf[fft_size - k] = frame_buf[k].conj();
            }

            fft_inverse.process(&mut frame_buf);

            // 加合成窗 + 逐帧 RMS 对齐
            let dry_windowed: Vec<f32> = padded[start..start + fft_size]
                .iter()
                .zip(analysis_window.iter())
                .map(|(s, w)| s * w)
                .collect();
            let mut wet_windowed: Vec<f32> = frame_buf.iter().map(|c| c.re * inv_n).collect();
            correction::match_frame_energy(&dry_windowed, &mut wet_windowed);
            for i in 0..fft_size {
                ola[start + i] += wet_windowed[i] * synthesis_window[i];
                win_sum[start + i] += synthesis_window[i] * synthesis_window[i];
            }
        } else {
            // 非浊音 / gate 关闭：原样 OLA
            for i in 0..fft_size {
                ola[start + i] += frame_buf[i].re * synthesis_window[i];
                win_sum[start + i] += synthesis_window[i] * synthesis_window[i];
            }
        }

        start += hop;
    }

    // ── 3. 归一化 + 截断 + 保护 ────────────────────────────────────────
    for (s, w) in ola.iter_mut().zip(win_sum.iter()) {
        if *w > 1.0e-8 {
            *s /= *w;
        }
    }
    let mut out = vec![0.0_f32; input.len()];
    out.copy_from_slice(&ola[pad_left..pad_left + input.len()]);
    correction::soft_limit(&mut out, input);
    Ok(out)
}

/// 取平滑轨迹上 af_f 处的 F1（失败回退 0，调用方以 candidates 判空保护）。
fn tp_f1(tracks: &[track::TrackPoint], af_f: f32) -> f32 {
    track::interpolate_at(tracks, af_f).map(|t| t.f1_hz).unwrap_or(0.0)
}

/// 取平滑轨迹上 af_f 处的 F2。
fn tp_f2(tracks: &[track::TrackPoint], af_f: f32) -> f32 {
    track::interpolate_at(tracks, af_f).map(|t| t.f2_hz).unwrap_or(0.0)
}

/// z 的升幂首一多项式 → 分析约定系数 a[1..=p]。
///
/// A(z⁻¹) = z⁻ⁿ·P(z) ⇒ a_k = c_{n−k}（P 的升幂系数去掉最高次后取反序）。
fn monic_to_analysis(monic: &[f32]) -> Vec<f32> {
    if monic.len() < 2 {
        return Vec::new();
    }
    monic[..monic.len() - 1].iter().rev().copied().collect()
}

/// Hann 窗。
fn hann_window(len: usize) -> Vec<f32> {
    if len <= 1 {
        return vec![1.0; len];
    }
    let denom = (len - 1) as f32;
    (0..len)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / denom).cos())
        .collect()
}

/// 多声道 PCM 共振峰迁移（公开 API，签名与重写前保持一致）。
///
/// 行为约定：
/// - channels == 0 → 错误。
/// - channels == 1 → 直接走 mono 路径。
/// - channels >= 2 → 取通道平均得到 mono 分析信号、跑 mono 算法、然后把
///   `wet - dry` delta 加回每个原通道（保留通道间相对差与立体声成像）。
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

// ── 内部辅助 ────────────────────────────────────────────────────────────

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
        let input = vec![0.0; 2048];
        let params = default_params(true, 1.0);
        let out = apply_formant_morph_mono(&input, 4_000, &params).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn output_is_finite_and_length_preserving() {
        let input: Vec<f32> = (0..4_096)
            .map(|i| {
                let t = i as f32 / 48_000.0;
                ((2.0 * std::f32::consts::PI * 220.0 * t).sin()) * 0.15
            })
            .collect();
        let params = default_params(true, 0.7);
        let out = apply_formant_morph_mono(&input, 48_000, &params).unwrap();
        assert_eq!(out.len(), input.len());
        for s in &out {
            assert!(s.is_finite(), "output must be finite");
            assert!(s.abs() <= 1.0, "output must be within [-1, 1]");
        }
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
        let mono: Vec<f32> = (0..2_048)
            .map(|i| ((i as f32) * 0.03).sin() * 0.1)
            .collect();
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
    fn interleaved_zero_channels_is_error() {
        let input = vec![0.0_f32; 8];
        let params = default_params(true, 1.0);
        assert!(apply_formant_morph_interleaved(&input, 48_000, 0, &params).is_err());
    }

    #[test]
    fn vowel_preset_table_returns_known_values() {
        assert_eq!(vowel_formant_preset("a"), Some((800.0, 1_200.0)));
        assert_eq!(vowel_formant_preset("ee"), Some((300.0, 2_300.0)));
        assert!(vowel_formant_preset("xyz").is_none());
    }

    // ── 端到端闸门测试（守住"搬得动"） ───────────────────────────────────

    const TEST_SR: u32 = 48_000;

    /// 合成稳态元音：脉冲串（F0）经两个并联二阶共振器（F1/F2）。
    fn synth_vowel(f0: f32, f1: f32, f2: f32, sr: u32, secs: f32) -> Vec<f32> {
        let sr = sr as f32;
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

    /// LCG 确定性白噪声。
    fn lcg_noise(n: usize) -> Vec<f32> {
        let mut seed = 0xDEADBEEF_u32;
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
            })
            .collect()
    }

    /// 用与 DSP 完全相同的分析链测量信号的 (F1, F2)（各帧中位数）。
    fn measure_formants(signal: &[f32], sr: u32) -> Option<(f32, f32)> {
        let mut dec = decimator::Decimator::new(sr, ANALYSIS_TARGET_RATE as u32)?;
        let a_rate = dec.analysis_rate();
        let analysis = dec.process(signal);
        let frame_len = (a_rate * ANALYSIS_FRAME_SEC).round() as usize;
        let hop = (a_rate * ANALYSIS_HOP_SEC).round() as usize;
        if analysis.len() < frame_len * 2 {
            return None;
        }
        let mut got: Vec<(f32, f32)> = Vec::new();
        let mut start = 0usize;
        while start + frame_len <= analysis.len() {
            if let Some(l) = lpc::analyze_frame(&analysis[start..start + frame_len], a_rate, LPC_ORDER)
            {
                if let Some(roots) = lpc::poly_roots(&lpc::coeffs_to_monic(&l.coeffs)) {
                    let poles = lpc::roots_to_poles(&roots, a_rate);
                    if let Some(c) =
                        track::select_f1_f2(&poles, &l.coeffs, a_rate, l.residual_ratio)
                    {
                        got.push((c.f1.freq_hz, c.f2.freq_hz));
                    }
                }
            }
            start += hop;
        }
        if got.is_empty() {
            return None;
        }
        let mut f1s: Vec<f32> = got.iter().map(|g| g.0).collect();
        let mut f2s: Vec<f32> = got.iter().map(|g| g.1).collect();
        f1s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        f2s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some((f1s[f1s.len() / 2], f2s[f2s.len() / 2]))
    }

    #[test]
    fn formant_shift_moves_measured_formants_toward_target() {
        // 源 /a/(F1=800, F2=1200) → 目标 /i/(300, 2300)
        let input = synth_vowel(150.0, 800.0, 1_200.0, TEST_SR, 0.5);
        let params = ClipFormantMorph {
            enabled: true,
            target_f1_hz: 300.0,
            target_f2_hz: 2_300.0,
            strength: 0.9,
        };
        let out = apply_formant_morph_mono(&input, TEST_SR, &params).unwrap();
        let (in_f1, in_f2) = measure_formants(&input, TEST_SR).expect("输入应可测出共振峰");
        let (out_f1, out_f2) = measure_formants(&out, TEST_SR).expect("输出应可测出共振峰");
        assert!(
            (out_f1 - 300.0).abs() < (in_f1 - 300.0).abs(),
            "F1 必须向目标移动: in={in_f1:.1} out={out_f1:.1}"
        );
        assert!(
            (out_f2 - 2_300.0).abs() < (in_f2 - 2_300.0).abs(),
            "F2 必须向目标移动: in={in_f2:.1} out={out_f2:.1}"
        );
    }

    #[test]
    fn stronger_strength_moves_further() {
        let input = synth_vowel(150.0, 800.0, 1_200.0, TEST_SR, 0.5);
        let mk = |s: f64| ClipFormantMorph {
            enabled: true,
            target_f1_hz: 300.0,
            target_f2_hz: 2_300.0,
            strength: s,
        };
        let dist = |s: f64| -> f32 {
            let out = apply_formant_morph_mono(&input, TEST_SR, &mk(s)).unwrap();
            let (out_f1, _) = measure_formants(&out, TEST_SR).unwrap();
            (out_f1 - 300.0).abs()
        };
        let d1 = dist(0.3);
        let d2 = dist(0.6);
        let d3 = dist(1.0);
        assert!(
            d3 < d2 && d2 < d1,
            "强度越大应离目标越近: d(0.3)={d1:.1} d(0.6)={d2:.1} d(1.0)={d3:.1}"
        );
    }

    #[test]
    fn unvoiced_noise_is_not_amplified() {
        let input = lcg_noise(TEST_SR as usize); // 1s 白噪声
        let params = default_params(true, 0.9);
        let out = apply_formant_morph_mono(&input, TEST_SR, &params).unwrap();
        assert_eq!(out.len(), input.len());
        for s in &out {
            assert!(s.is_finite());
        }
        let in_peak = peak_abs(&input);
        let out_peak = peak_abs(&out);
        assert!(
            out_peak <= in_peak * OUTPUT_PEAK_RATIO_LIMIT + 1.0e-3,
            "噪声输入不得放大: in={in_peak} out={out_peak}"
        );
    }

    #[test]
    fn pathological_inputs_stay_bounded() {
        let cases: Vec<Vec<f32>> = vec![
            vec![0.9_f32; 8_192],                            // 直流
            (0..8_192).map(|i| if i % 2 == 0 { 0.8 } else { -0.8 }).collect(), // 方波
            (0..513).map(|i| (i as f32 * 0.05).sin() * 0.5).collect(),         // 极短
        ];
        for input in cases {
            let params = default_params(true, 1.0);
            let out = apply_formant_morph_mono(&input, TEST_SR, &params).unwrap();
            assert_eq!(out.len(), input.len());
            for s in &out {
                assert!(s.is_finite(), "病态输入输出必须有限");
                assert!(s.abs() <= 1.0);
            }
        }
    }
}
