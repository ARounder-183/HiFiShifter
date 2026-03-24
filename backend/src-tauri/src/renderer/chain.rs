//! ProcessorChain：可组合的 Stage 链。
//!
//! 每个 [`ProcessingStage`] 接收上一步输出的 PCM，返回新 PCM；
//! [`ProcessorChain`] 串联多个 Stage 并实现 [`ClipProcessor`] trait。
//!
//! 内置 Stage：
//! - [`SignalsmithTimeStretchStage`]：应用 playback_rate 时间拉伸
//! - [`WorldVocoderStage`]：WORLD 声码器合成
//! - [`HiFiGanStage`]：NSF-HiFiGAN 合成
//!
//! 预设链构造：[`world_chain()`]、[`hifigan_chain()`]

use super::traits::{
    ClipProcessContext, ClipProcessor, ParamDescriptor, ProcessorCapabilities, RenderContext,
    Renderer,
};

static HIFIGAN_BREATH_OPTIONS: [(&str, i32); 2] = [("Off", 0), ("On", 1)];

static HIFIGAN_PARAM_DESCRIPTORS: [ParamDescriptor; 5] = [
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
    ParamDescriptor {
        id: "hifigan_volume",
        display_name: "Volume",
        group: "NSF-HiFiGAN",
        kind: super::traits::ParamKind::AutomationCurve {
            unit: "x",
            default_value: 1.0,
            min_value: 0.0,
            max_value: 2.0,
        },
    },
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
    pub display_name: String,
    pub stages: Vec<Box<dyn ProcessingStage>>,
    /// 链内是否包含时间拉伸 Stage（[`SignalsmithTimeStretchStage`]）。
    /// 为 `true` 时调用方（`render_single_clip` / `render_mixdown_interleaved`）跳过外部
    /// Signalsmith 预拉伸，并将实际 `playback_rate` 通过 [`ClipProcessContext`] 传入链内。
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

/// Stage 1：Signalsmith Stretch 时间拉伸（playback_rate ≈ 1.0 时直接透传）。
pub struct SignalsmithTimeStretchStage;

impl ProcessingStage for SignalsmithTimeStretchStage {
    fn id(&self) -> &str {
        "signalsmith_stretch"
    }

    fn display_name(&self) -> &str {
        "时间拉伸 (Signalsmith)"
    }

    fn process(&self, input_pcm: Vec<f32>, ctx: &StageContext<'_>) -> Result<Vec<f32>, String> {
        let cc = ctx.clip_ctx;
        let rate = cc.playback_rate;
        if (rate - 1.0).abs() <= 1e-6 {
            return Ok(input_pcm);
        }

        // 目标帧数：优先使用 out_frames（若调用方已计算），否则由 rate 推算。
        let out_frames = if cc.out_frames > 0 {
            cc.out_frames
        } else {
            let in_frames = input_pcm.len();
            ((in_frames as f64) / rate).round().max(2.0) as usize
        };

        let stretched = crate::time_stretch::time_stretch_interleaved(
            &input_pcm,
            1, // mono
            cc.sample_rate,
            out_frames,
            crate::time_stretch::StretchAlgorithm::SignalsmithStretch,
        );
        Ok(stretched)
    }
}

/// Stage 2a：WORLD 声码器合成。
pub struct WorldVocoderStage;

impl ProcessingStage for WorldVocoderStage {
    fn id(&self) -> &str {
        "world_vocoder"
    }

    fn display_name(&self) -> &str {
        "WORLD 声码器"
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

/// Stage 2b：NSF-HiFiGAN ONNX 合成。
pub struct HiFiGanStage;

fn sample_curve_at_abs_sec(
    curve: Option<&Vec<f32>>,
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
    let i1 = (i0 + 1).min(curve.len().saturating_sub(1));
    let frac = (idx_f - i0 as f64).clamp(0.0, 1.0) as f32;
    let a = curve.get(i0).copied().unwrap_or(default_value);
    let b = curve.get(i1).copied().unwrap_or(a);
    a + (b - a) * frac
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

        let rate = cc.playback_rate;
        let needs_stretch = (rate - 1.0).abs() > 1e-6;

        let breath_enabled =
            crate::pitch_editing::extra_param_enabled(cc.extra_params, "breath_enabled");
        let formant_curve = cc.extra_curves.get("formant_shift_cents");
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
            if needs_stretch {
                // mel 域时间拉伸 + HiFiGAN 推理
                return crate::renderer::hifigan::HiFiGanRenderer.render_mel_stretch(
                    &render_ctx,
                    rate,
                    formant_curve,
                );
            } else {
                return crate::renderer::hifigan::HiFiGanRenderer
                    .render_with_formant(&render_ctx, formant_curve);
            }
        }

        // ── Breath 路径 ─────────────────────────────────────────────────────
        if !crate::hnsep_onnx::is_available() {
            return Err("HNSEP is enabled but model is unavailable".to_string());
        }

        let (harmonic, noise) =
            crate::hnsep_onnx::infer_harmonic_noise_mono(cc.clip_id, &input_pcm, cc.sample_rate)?;

        // harmonic 走 mel stretch
        let processed_harmonic = if cc.clip_midi.is_empty() {
            harmonic
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
            if needs_stretch {
                crate::renderer::hifigan::HiFiGanRenderer.render_mel_stretch(
                    &render_ctx,
                    rate,
                    formant_curve,
                )?
            } else {
                crate::renderer::hifigan::HiFiGanRenderer
                    .render_with_formant(&render_ctx, formant_curve)?
            }
        };

        // noise 走 Signalsmith Stretch 时间拉伸（noise 是非谐波信号，波形域拉伸无明显伪影）
        let stretched_noise = if needs_stretch {
            let out_frames = processed_harmonic.len();
            crate::time_stretch::time_stretch_interleaved(
                &noise,
                1,
                cc.sample_rate,
                out_frames,
                crate::time_stretch::StretchAlgorithm::SignalsmithStretch,
            )
        } else {
            noise
        };

        let breath_curve = cc.extra_curves.get("breath_gain");
        let out_len = processed_harmonic.len().min(stretched_noise.len());

        // 提取曲线存在性判断，分支走 Fast-Path
        let has_valid_curve = breath_curve.map_or(false, |c| !c.is_empty());

        let mixed: Vec<f32> = if has_valid_curve {
            // 将除法转化为乘法，移出循环
            let inv_sample_rate = 1.0 / cc.sample_rate.max(1) as f64;

            // 使用迭代器消除 memset 和 越界检查
            processed_harmonic
                .iter()
                .zip(stretched_noise.iter())
                .take(out_len)
                .enumerate()
                .map(|(index, (&h, &n))| {
                    // 使用乘法替代除法
                    let abs_sec = cc.seg_start_sec + index as f64 * inv_sample_rate;
                    let gain =
                        sample_curve_at_abs_sec(breath_curve, abs_sec, cc.frame_period_ms, 1.0);
                    h + n * gain
                })
                .collect()
        } else {
            // 无曲线时，直接 SIMD 向量化相加，gain 默认为 1.0
            processed_harmonic
                .iter()
                .zip(stretched_noise.iter())
                .take(out_len)
                .map(|(&h, &n)| h + n)
                .collect()
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
        stages: vec![
            Box::new(SignalsmithTimeStretchStage),
            Box::new(WorldVocoderStage),
        ],
        handles_time_stretch: true,
    }
}

/// 构造 NSF-HiFiGAN 处理链。
///
/// 时间拉伸由 HiFiGanStage 内部通过 mel 域线性插值完成，
/// 不再需要外部 SignalsmithTimeStretchStage。
pub fn hifigan_chain() -> ProcessorChain {
    ProcessorChain {
        id: "nsf_hifigan".into(),
        display_name: "NSF-HiFiGAN".into(),
        stages: vec![Box::new(HiFiGanStage)],
        handles_time_stretch: true,
    }
}
