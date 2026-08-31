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

mod decimator;

use crate::state::ClipFormantMorph;

// ── 共享常量 ─────────────────────────────────────────────────────────────

/// 低于该采样率直接 bypass：低于 8kHz 的素材本身就没有可靠的 F2 信息。
const MIN_SAMPLE_RATE: u32 = 8_000;
/// 输入样本不足直接 bypass：连一个分析帧都做不满。
const MIN_INPUT_SAMPLES: usize = 512;
/// strength 低于此阈值视为关闭（避免极小浮点误差触发处理）。
const STRENGTH_EPS: f32 = 1.0e-5;
/// 输出整体峰值上限相对输入峰值的最大放大倍数。
const OUTPUT_PEAK_RATIO_LIMIT: f32 = 1.6;

// ── 公开入口 ────────────────────────────────────────────────────────────

/// 单声道 PCM 共振峰迁移（公开 API，签名与重写前保持一致）。
///
/// 流程（Task 4 完成完整编排）：
/// 1. 入口校验（disabled / 空输入 / 低采样率 / 短样本 / strength 接近 0）
///    → 直接 bypass。
/// 2. 预加重 → 降采样 → LPC 分析 → 极点迁移 → 频域比值滤波。
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

    // TODO(Task 4)：接入完整 DSP 编排（LPC 分析 → 极点迁移 → 频域比值滤波）。
    let _ = (
        params.target_f1_hz,
        params.target_f2_hz,
        OUTPUT_PEAK_RATIO_LIMIT,
    );
    Ok(input.to_vec())
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
}
