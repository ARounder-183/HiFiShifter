//! ProcessorChain：可组合的 Stage 链。
//!
//! 每个 [`ProcessingStage`] 接收上一步输出的 PCM，返回新 PCM；
//! [`ProcessorChain`] 串联多个 Stage 并实现 [`ClipProcessor`] trait。
//!
//! 内置 Stage：
//! - [`WorldVocoderStage`]：WORLD 声码器合成
//! - [`HiFiGanStage`]：NSF-HiFiGAN 合成
//!
//! 预设链构造：[`world_chain()`]、[`hifigan_chain()`]

use super::common_params::{COMMON_MIX_PARAMS, PAN_PARAM, VOLUME_PARAM};
use super::traits::{
    ClipProcessContext, ClipProcessor, ParamDescriptor, ProcessorCapabilities, RenderContext,
    Renderer,
};

static HIFIGAN_BREATH_OPTIONS: [(&str, i32); 2] = [("Off", 0), ("On", 1)];

static HIFIGAN_PARAM_DESCRIPTORS: [ParamDescriptor; 6] = [
    ParamDescriptor {
        id: "breath_enabled",
        display_name: "Breath",
        group: "NSF-HiFiGAN",
        kind: super::traits::ParamKind::StaticEnum {
            options: &HIFIGAN_BREATH_OPTIONS,
            default_value: 0,
        },
    },
    ParamDescriptor {
        id: "breath_gain",
        display_name: "Breath Gain",
        group: "NSF-HiFiGAN",
        kind: super::traits::ParamKind::AutomationCurve {
            unit: "x",
            default_value: 1.0,
            min_value: 0.0,
            max_value: 2.0,
        },
    },
    ParamDescriptor {
        id: "hifigan_tension",
        display_name: "Tension",
        group: "NSF-HiFiGAN",
        kind: super::traits::ParamKind::AutomationCurve {
            unit: "%",
            default_value: 0.0,
            min_value: -100.0,
            max_value: 100.0,
        },
    },
    ParamDescriptor {
        id: "formant_shift_cents",
        display_name: "Formant Shift",
        group: "NSF-HiFiGAN",
        kind: super::traits::ParamKind::AutomationCurve {
            unit: "cents",
            default_value: 0.0,
            min_value: -500.0,
            max_value: 500.0,
        },
    },
    VOLUME_PARAM,
    PAN_PARAM,
];

// ─── StageContext ──────────────────────────────────────────────────────────────

/// 传递给每个 Stage 的完整上下文（持有对 [`ClipProcessContext`] 的引用）。
pub struct StageContext<'a> {
    pub clip_ctx: &'a ClipProcessContext<'a>,
}

// ─── ProcessingStage trait ────────────────────────────────────────────────────

/// 单一处理阶段，接收上一步 PCM，输出处理后 PCM。
pub trait ProcessingStage: Send + Sync {
    fn id(&self) -> &str;
    #[allow(dead_code)]
    fn display_name(&self) -> &str;
    /// Stage 自身贡献的参数描述符（可选）。
    fn param_descriptors(&self) -> &'static [ParamDescriptor] {
        &[]
    }
    /// 接收上一步 PCM，输出处理后 PCM。
    fn process(&self, input_pcm: Vec<f32>, ctx: &StageContext<'_>) -> Result<Vec<f32>, String>;
}

// ─── ProcessorChain ───────────────────────────────────────────────────────────

/// 实现 `ClipProcessor` 的 Stage 链，将多个 Stage 串联。
pub struct ProcessorChain {
    pub id: String,
    #[allow(dead_code)]
    pub display_name: String,
    pub stages: Vec<Box<dyn ProcessingStage>>,
    /// 处理器是否自行处理时间拉伸。
    /// 为 `true` 时调用方会跳过外部预拉伸，并将 `playback_rate`
    /// 通过 [`ClipProcessContext`] 传入处理器链内部。
    pub handles_time_stretch: bool,
}

impl ClipProcessor for ProcessorChain {
    fn id(&self) -> &str {
        &self.id
    }

    fn display_name(&self) -> &str {
        &self.display_name
    }

    fn is_available(&self) -> bool {
        // 链路整体可用性由各 Stage 自行控制；此处返回 true 让调用方统一判断
        true
    }

    fn capabilities(&self) -> ProcessorCapabilities {
        ProcessorCapabilities {
            handles_time_stretch: self.handles_time_stretch,
            supports_formant: false,
            supports_breathiness: self.stages.iter().any(|stage| stage.id() == "nsf_hifigan"),
        }
    }

    fn param_descriptors(&self) -> Vec<ParamDescriptor> {
        self.stages
            .iter()
            .flat_map(|s| s.param_descriptors().iter().cloned())
            .collect()
    }

    fn process(&self, ctx: &ClipProcessContext<'_>) -> Result<Vec<f32>, String> {
        let stage_ctx = StageContext { clip_ctx: ctx };
        let mut pcm = ctx.mono_pcm.to_vec();
        for stage in &self.stages {
            pcm = stage.process(pcm, &stage_ctx)?;
        }
        Ok(pcm)
    }
}

// ─── 内置 Stage 实现 ──────────────────────────────────────────────────────────

/// Stage 1a：WORLD 声码器合成。
pub struct WorldVocoderStage;

impl ProcessingStage for WorldVocoderStage {
    fn id(&self) -> &str {
        "world_vocoder"
    }

    fn display_name(&self) -> &str {
        "WORLD 声码器"
    }

    fn param_descriptors(&self) -> &'static [ParamDescriptor] {
        &COMMON_MIX_PARAMS
    }

    fn process(&self, input_pcm: Vec<f32>, ctx: &StageContext<'_>) -> Result<Vec<f32>, String> {
        let cc = ctx.clip_ctx;
        if !crate::world_vocoder::is_available() {
            return Ok(input_pcm);
        }
        let render_ctx = RenderContext {
            mono_pcm: &input_pcm,
            sample_rate: cc.sample_rate,
            seg_start_sec: cc.seg_start_sec,
            seg_end_sec: cc.seg_end_sec,
            clip_start_sec: cc.clip_start_sec,
            frame_period_ms: cc.frame_period_ms,
            pitch_edit: cc.pitch_edit,
            clip_midi: cc.clip_midi,
            clip_id: cc.clip_id,
        };
        crate::renderer::world::WorldRenderer.render(&render_ctx)
    }
}

/// Stage 1b：NSF-HiFiGAN ONNX 合成。
pub struct HiFiGanStage;

fn sample_curve_at_abs_sec(
    curve: Option<&[f32]>,
    abs_sec: f64,
    frame_period_ms: f64,
    default_value: f32,
) -> f32 {
    let Some(curve) = curve else {
        return default_value;
    };
    if curve.is_empty() {
        return default_value;
    }

    let fp = frame_period_ms.max(0.1);
    let idx_f = (abs_sec.max(0.0) * 1000.0) / fp;
    if !idx_f.is_finite() {
        return default_value;
    }
    let i0 = idx_f.floor().max(0.0) as usize;
    // 越界（超出曲线末点）直接返回默认值：i1 会被钳制到最后一个元素，
    // 若继续插值会得到 default 与末值之间的错误衰减/振荡值（与
    // hifigan / vslib 的末点越界修复保持一致）。
    if i0 >= curve.len() {
        return default_value;
    }
    let i1 = (i0 + 1).min(curve.len().saturating_sub(1));
    let frac = (idx_f - i0 as f64).clamp(0.0, 1.0) as f32;
    let a = curve.get(i0).copied().unwrap_or(default_value);
    let b = curve.get(i1).copied().unwrap_or(a);
    a + (b - a) * frac
}

/// 把噪声 stem 线性重采样到目标长度（按样本数，而非采样率比）。
///
/// 时间拉伸只作用在谐波分支上：`render_mel_stretch_with_formant` 在 mel 域把
/// 谐波拉伸到**时间轴**长度，而 HNSEP 分离出的噪声 stem 仍是**源速率**长度。
/// 若直接按 `min(harmonic, noise)` 混合：
/// - `playback_rate < 1`（拉长）时谐波比噪声长，输出被截断到噪声长度，
///   clip 拉伸出来的尾巴整段丢失（听感上就是"下一段音频被截断"）；
/// - `playback_rate > 1`（缩短）时噪声比谐波长，噪声尾部被丢掉。
///
/// 噪声是随机信号，线性重采样即可保持其统计特性，代价远低于再跑一次分离。
pub(crate) fn resample_noise_to_len(noise: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 || noise.is_empty() {
        return Vec::new();
    }
    if noise.len() == target_len {
        return noise.to_vec();
    }
    if noise.len() == 1 {
        return vec![noise[0]; target_len];
    }

    let src_last = noise.len() - 1;
    let scale = src_last as f64 / (target_len.saturating_sub(1)).max(1) as f64;
    let mut out = Vec::with_capacity(target_len);
    for idx in 0..target_len {
        let src = idx as f64 * scale;
        let i0 = (src as usize).min(src_last.saturating_sub(1));
        let i1 = (i0 + 1).min(src_last);
        let frac = (src - i0 as f64) as f32;
        out.push(noise[i0] + (noise[i1] - noise[i0]) * frac);
    }
    out
}

impl ProcessingStage for HiFiGanStage {
    fn id(&self) -> &str {
        "nsf_hifigan"
    }

    fn display_name(&self) -> &str {
        "NSF-HiFiGAN"
    }

    fn param_descriptors(&self) -> &'static [ParamDescriptor] {
        &HIFIGAN_PARAM_DESCRIPTORS
    }

    fn process(&self, input_pcm: Vec<f32>, ctx: &StageContext<'_>) -> Result<Vec<f32>, String> {
        let cc = ctx.clip_ctx;
        if !crate::nsf_hifigan_onnx::is_available() {
            return Ok(input_pcm);
        }

        let breath_enabled =
            crate::pitch_editing::extra_param_enabled(cc.extra_params, "breath_enabled");
        let formant_curve = cc
            .extra_curves
            .get("formant_shift_cents")
            .map(|v| v.as_slice());
        if !breath_enabled {
            // ── 非 Breath 路径 ──────────────────────────────────────────────
            let render_ctx = RenderContext {
                mono_pcm: &input_pcm,
                sample_rate: cc.sample_rate,
                seg_start_sec: cc.seg_start_sec,
                seg_end_sec: cc.seg_end_sec,
                clip_start_sec: cc.clip_start_sec,
                frame_period_ms: cc.frame_period_ms,
                pitch_edit: cc.pitch_edit,
                clip_midi: cc.clip_midi,
                clip_id: cc.clip_id,
            };
            let renderer = crate::renderer::hifigan::HiFiGanRenderer;
            return if (cc.playback_rate - 1.0).abs() > 1.0e-6 {
                renderer.render_mel_stretch_with_formant(
                    &render_ctx,
                    cc.playback_rate,
                    formant_curve,
                )
            } else {
                renderer.render_with_formant(&render_ctx, formant_curve)
            };
        }

        // ── Breath 路径 ─────────────────────────────────────────────────────
        if !crate::hnsep_onnx::is_available() {
            return Err("HNSEP is enabled but model is unavailable".to_string());
        }

        let (harmonic, noise) =
            crate::hnsep_onnx::infer_harmonic_noise_mono(cc.clip_id, &input_pcm, cc.sample_rate)?;

        // harmonic 直接走 HiFiGAN；时间拉伸已在处理器外部完成
        let processed_harmonic = if cc.clip_midi.is_empty() {
            if (cc.playback_rate - 1.0).abs() > 1.0e-6 {
                crate::time_stretch::time_stretch_interleaved(
                    &harmonic,
                    1,
                    cc.sample_rate,
                    cc.out_frames,
                    crate::time_stretch::resolved_external_stretch_algorithm(),
                )
            } else {
                (*harmonic).clone()
            }
        } else {
            let render_ctx = RenderContext {
                mono_pcm: &harmonic,
                sample_rate: cc.sample_rate,
                seg_start_sec: cc.seg_start_sec,
                seg_end_sec: cc.seg_end_sec,
                clip_start_sec: cc.clip_start_sec,
                frame_period_ms: cc.frame_period_ms,
                pitch_edit: cc.pitch_edit,
                clip_midi: cc.clip_midi,
                clip_id: cc.clip_id,
            };
            let renderer = crate::renderer::hifigan::HiFiGanRenderer;
            if (cc.playback_rate - 1.0).abs() > 1.0e-6 {
                renderer.render_mel_stretch_with_formant(
                    &render_ctx,
                    cc.playback_rate,
                    formant_curve,
                )?
            } else {
                renderer.render_with_formant(&render_ctx, formant_curve)?
            }
        };

        let breath_curve = cc.extra_curves.get("breath_gain").map(|v| v.as_slice());

        // Fast path: if breath_gain is uniformly zero (e.g. when computing harmonic_only
        // for BreathNoiseCache), skip noise mixing entirely and return processed_harmonic.
        let gain_is_zero = breath_curve.map_or(true, |c| {
            c.is_empty() || c.iter().all(|&v| v.abs() < f32::EPSILON)
        });
        if gain_is_zero {
            return Ok(processed_harmonic);
        }

        // 噪声 stem 与谐波对齐到同一（时间轴）长度后再混合。
        // 不能用 `min(harmonic, noise)`：那会在拉伸后把谐波尾巴裁掉。
        let noise_aligned = resample_noise_to_len(&noise, processed_harmonic.len());
        let out_len = processed_harmonic.len();

        let has_varying_curve = breath_curve.map_or(false, |c| {
            if c.len() <= 1 {
                return false;
            }
            let first = c[0];
            c.iter().any(|&v| (v - first).abs() > f32::EPSILON)
        });

        let mixed: Vec<f32> = if has_varying_curve {
            let inv_sample_rate = 1.0 / cc.sample_rate.max(1) as f64;
            processed_harmonic
                .iter()
                .zip(noise_aligned.iter())
                .take(out_len)
                .enumerate()
                .map(|(index, (&h, &n))| {
                    let abs_sec = cc.seg_start_sec + index as f64 * inv_sample_rate;
                    let gain =
                        sample_curve_at_abs_sec(breath_curve, abs_sec, cc.frame_period_ms, 1.0);
                    h + n * gain
                })
                .collect()
        } else {
            // Constant gain (typically 1.0): use uniform multiplier, auto-vectorizable
            let gain = breath_curve.and_then(|c| c.first().copied()).unwrap_or(1.0);
            if (gain - 1.0).abs() < f32::EPSILON {
                // gain == 1.0: simple addition, most common case for unity_breath
                processed_harmonic
                    .iter()
                    .zip(noise_aligned.iter())
                    .take(out_len)
                    .map(|(&h, &n)| h + n)
                    .collect()
            } else {
                processed_harmonic
                    .iter()
                    .zip(noise_aligned.iter())
                    .take(out_len)
                    .map(|(&h, &n)| h + n * gain)
                    .collect()
            }
        };

        Ok(mixed)
    }
}

// ─── 预设链构造 ───────────────────────────────────────────────────────────────

/// 构造 WORLD Vocoder 处理链。
pub fn world_chain() -> ProcessorChain {
    ProcessorChain {
        id: "world".into(),
        display_name: "WORLD Vocoder".into(),
        stages: vec![Box::new(WorldVocoderStage)],
        handles_time_stretch: false,
    }
}

/// 构造 NSF-HiFiGAN 处理链。
pub fn hifigan_chain() -> ProcessorChain {
    ProcessorChain {
        id: "nsf_hifigan".into(),
        display_name: "NSF-HiFiGAN".into(),
        stages: vec![Box::new(HiFiGanStage)],
        handles_time_stretch: false,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn hifigan_chain_no_longer_handles_time_stretch() {
        let chain = super::hifigan_chain();
        assert!(!chain.handles_time_stretch);
    }

    #[test]
    fn resample_noise_to_len_keeps_identity_length() {
        let noise = vec![0.1f32, 0.2, 0.3, 0.4];
        assert_eq!(super::resample_noise_to_len(&noise, 4), noise);
    }

    #[test]
    fn resample_noise_to_len_stretches_without_truncating() {
        // 拉伸场景（playback_rate < 1）：谐波比噪声长，噪声必须被拉长，
        // 否则 min() 会把谐波（进而整个 clip）的尾巴裁掉。
        let noise = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = super::resample_noise_to_len(&noise, 8);
        assert_eq!(out.len(), 8);
        // 端点应贴合原始端点
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[7] - 4.0).abs() < 1e-6);
        // 中间值必须来自原始信号，而不是补零
        assert!(out.iter().all(|v| *v >= 1.0 - 1e-6 && *v <= 4.0 + 1e-6));
    }

    #[test]
    fn resample_noise_to_len_shrinks_for_speedup() {
        let noise = vec![0.0f32, 1.0, 2.0, 3.0];
        let out = super::resample_noise_to_len(&noise, 2);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn resample_noise_to_len_handles_degenerate_inputs() {
        assert!(super::resample_noise_to_len(&[], 8).is_empty());
        assert!(super::resample_noise_to_len(&[0.5], 0).is_empty());
        // 单样本输入：按常数填充，不得 panic
        assert_eq!(super::resample_noise_to_len(&[0.5], 3), vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn sample_curve_beyond_end_returns_default() {
        // 回归：末点之后必须回退 default，而不是与末值插值出 0..末值 的振荡。
        // 复现场景 = 共振峰偏移点 359.15 画在曲线末点（fp=5ms，idx 6470），
        // 之后任意采样都应得到默认值 0。
        let curve = vec![0.0f32, 0.0, 0.0, 359.15]; // 末点 359.15 @ idx 3
        let fp = 5.0;
        // idx 3.0（末点本身）→ 359.15
        let at_last = super::sample_curve_at_abs_sec(Some(&curve), 3.0 * fp / 1000.0, fp, 0.0);
        assert!((at_last - 359.15).abs() < 1e-4);
        // idx ∈ [3.0, 4.0)：最后一个元素的保持区间（i0=3 在界内，
        // i1 钳到自身）→ 仍为末值，与 pitch 采样语义一致
        let hold = super::sample_curve_at_abs_sec(Some(&curve), 3.5 * fp / 1000.0, fp, 0.0);
        assert!((hold - 359.15).abs() < 1e-4);
        // idx >= 4.0（超出数组末尾）→ default 0.0（修复前为 0..末值 的振荡）
        let frac_beyond = super::sample_curve_at_abs_sec(Some(&curve), 4.5 * fp / 1000.0, fp, 0.0);
        assert_eq!(frac_beyond, 0.0);
        // 大越界同样归零（修复前 frac 小数部分导致任意非零值）
        let far = super::sample_curve_at_abs_sec(Some(&curve), 100.0, fp, 0.0);
        assert_eq!(far, 0.0);
        // 空曲线 / None → default
        assert_eq!(super::sample_curve_at_abs_sec(Some(&[]), 1.0, fp, 0.0), 0.0);
        assert_eq!(super::sample_curve_at_abs_sec(None, 1.0, fp, 0.0), 0.0);
    }

    #[test]
    fn sample_curve_interpolates_within_range() {
        let curve = vec![0.0f32, 100.0, 200.0];
        let fp = 5.0;
        // idx 0.5 → 50
        let mid = super::sample_curve_at_abs_sec(Some(&curve), 0.5 * fp / 1000.0, fp, 0.0);
        assert!((mid - 50.0).abs() < 1e-4);
        // 负时间 → idx 0 → 第一个元素
        let neg = super::sample_curve_at_abs_sec(Some(&curve), -2.0, fp, 0.0);
        assert_eq!(neg, 0.0);
    }
}
