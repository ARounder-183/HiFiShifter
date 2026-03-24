//! 基于 NSF-HiFiGAN ONNX 的渲染器实现。
//!
//! # F0 数据源
//!
//! `midi_at_time` 回调与 [`WorldRenderer`] 共用同一套 F0 数据源：
//! - `clip_midi`：由 Harvest 分析得到的原始 MIDI 曲线（时间轴对齐）
//! - `pitch_edit`：用户编辑的目标 MIDI 曲线（0 表示无编辑）
//!
//! 两条链路切换时无需重新分析，直接复用已有的 `clip_midi`。
//! 若 `clip_midi` 为空（Harvest 尚未完成），则跳过推理并返回原始 PCM。

use super::traits::{RenderContext, Renderer, RendererCapabilities};
use super::utils::{clip_midi_at_time, edit_midi_at_time_or_none};
use crate::state::SynthPipelineKind;

/// 基于 NSF-HiFiGAN ONNX 的渲染器。
pub struct HiFiGanRenderer;

impl Renderer for HiFiGanRenderer {
    fn id(&self) -> &str {
        "nsf_hifigan_onnx"
    }

    fn display_name(&self) -> &str {
        "NSF-HiFiGAN (ONNX)"
    }

    fn kind(&self) -> SynthPipelineKind {
        SynthPipelineKind::NsfHifiganOnnx
    }

    fn is_available(&self) -> bool {
        crate::nsf_hifigan_onnx::is_available()
    }

    fn render(&self, ctx: &RenderContext<'_>) -> Result<Vec<f32>, String> {
        self.render_with_formant(ctx, None)
    }

    fn capabilities(&self) -> RendererCapabilities {
        RendererCapabilities {
            supports_realtime: false,
            prefers_prerender: true,
            max_pitch_shift_semitones: 24.0,
        }
    }
}

impl HiFiGanRenderer {
    /// 内部实现：带共振峰偏移曲线的渲染方法。
    ///
    /// `formant_shift_curve`：共振峰偏移曲线（cents），`None` 或空表示无偏移。
    /// 曲线按 `frame_period_ms` 采样，`curve[0]` 对应绝对时间 0。
    pub fn render_with_formant(
        &self,
        ctx: &RenderContext<'_>,
        formant_shift_curve: Option<&Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        let fp = ctx.frame_period_ms;
        let clip_start = ctx.clip_start_sec;
        let pitch_edit = ctx.pitch_edit;
        let clip_midi = ctx.clip_midi;

        // clip_midi 为空时明确跳过，与 WORLD 链路行为一致。
        // Harvest 分析尚未完成时 clip_midi 可能为空，此时返回原始 PCM。
        if clip_midi.is_empty() {
            if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
                eprintln!(
                    "HiFiGanRenderer::render: clip_midi is empty (Harvest not ready?), \
                     skipping inference and returning original PCM"
                );
            }
            return Ok(ctx.mono_pcm.to_vec());
        }

        // ── 查询 per-segment 缓存 ─────────────────────────────────────────────
        // 用 clip_id + seg 范围 + pitch_edit 片段 计算 param_hash，
        // 实现离线渲染路径的推理结果复用。
        let sr = ctx.sample_rate;
        let seg_start_frame = (ctx.seg_start_sec * sr as f64).round().max(0.0) as u64;
        let seg_end_frame = (ctx.seg_end_sec * sr as f64).round().max(0.0) as u64;
        // 直接引用上下文里的 pitch_edit，不再 to_vec()
        let curves_snapshot = crate::pitch_editing::PitchCurvesSnapshot {
            frame_period_ms: fp,
            pitch_orig: &[],
            pitch_edit,
        };

        // 构建元组数组，不再 new() HashMap 并 clone() 大数组
        let extra_curves = formant_shift_curve
            .map(|c| vec![("formant_shift_cents", c.as_slice())])
            .unwrap_or_default();

        let param_hash = crate::synth_clip_cache::compute_param_hash(
            ctx.clip_id,
            seg_start_frame,
            seg_end_frame,
            sr,
            self.id(),
            &curves_snapshot,
            extra_curves,
            &std::collections::HashMap::new(),
        );
        let cache_key = crate::synth_clip_cache::SynthClipCacheKey {
            clip_id: ctx.clip_id.to_string(),
            param_hash,
        };
        // 命中缓存：直接返回 mono PCM（从 stereo 取左声道）
        {
            let mut cache = crate::synth_clip_cache::global_synth_clip_cache()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(entry) = cache.get(&cache_key) {
                let mut mono_out: Vec<f32> = entry
                    .pcm_stereo
                    .iter()
                    .step_by(2)
                    .take(ctx.mono_pcm.len())
                    .copied()
                    .collect();
                mono_out.resize(ctx.mono_pcm.len(), 0.0);
                return Ok(mono_out);
            }
        }

        // 未命中：推理后写入缓存
        // midi_at_time 回调使用 clip_midi_at_time + edit_midi_at_time_or_none
        // 的组合逻辑，与 WorldRenderer 共用同一套 F0 查询语义。
        // 区别：WORLD 返回 semitone shift，ONNX 返回目标绝对 MIDI（模型输入语义不同）。
        let chunk_sec = crate::nsf_hifigan_onnx::env_chunk_sec();
        let overlap_sec = crate::nsf_hifigan_onnx::env_overlap_sec();

        // 构造共振峰偏移回调
        let fp_local = fp.max(0.1);
        let time_to_idx_mul = 1000.0 / fp_local;

        let formant_shift_fn = move |abs_time_sec: f64| -> f32 {
            let Some(curve) = formant_shift_curve else {
                return 0.0;
            };
            if curve.is_empty() {
                return 0.0;
            }
            let idx_f = abs_time_sec.max(0.0) * time_to_idx_mul;
            if !idx_f.is_finite() {
                return 0.0;
            }
            let i0 = idx_f.floor().max(0.0) as usize;
            let i1 = (i0 + 1).min(curve.len().saturating_sub(1));
            let frac = (idx_f - i0 as f64).clamp(0.0, 1.0) as f32;
            let a = curve.get(i0).copied().unwrap_or(0.0);
            let b = curve.get(i1).copied().unwrap_or(a);
            a + (b - a) * frac
        };

        let result = crate::nsf_hifigan_onnx::infer_pitch_edit_chunked(
            ctx.mono_pcm,
            sr,
            ctx.seg_start_sec,
            move |abs_time_sec| {
                // 原始 MIDI（来自 Harvest，与 WORLD 链路共用同一数据源）
                let orig = clip_midi_at_time(fp, clip_start, clip_midi, abs_time_sec);
                if !(orig.is_finite() && orig > 0.0) {
                    return 0.0;
                }
                // 目标 MIDI：有编辑时用编辑值，否则用原始值（保持音高不变）
                let target = match edit_midi_at_time_or_none(fp, pitch_edit, abs_time_sec) {
                    Some(v) => v,
                    None => orig,
                };
                if target.is_finite() && target > 0.0 {
                    target
                } else {
                    0.0
                }
            },
            formant_shift_fn,
            chunk_sec,
            overlap_sec,
        )?;

        // 写入缓存（stereo = mono 复制到双声道）
        if !result.is_empty() {
            let mut stereo = Vec::with_capacity(result.len() * 2);
            for &v in &result {
                stereo.push(v);
                stereo.push(v);
            }
            let frames = result.len() as u64;
            let mut cache = crate::synth_clip_cache::global_synth_clip_cache()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.insert(
                cache_key,
                crate::synth_clip_cache::SynthClipCacheEntry {
                    pcm_stereo: std::sync::Arc::new(stereo),
                    frames,
                    sample_rate: sr,
                },
            );
        }

        Ok(result)
    }

    /// Mel stretch 路径：在 mel 域完成时间拉伸后推理。
    ///
    /// 与 [`render_with_formant`] 的区别：`mono_pcm` 是**未拉伸的源 PCM**，由本方法在 mel 域
    /// 完成 `playback_rate` 对应的时间拉伸，然后经 HiFiGAN 合成拉伸后的波形。
    pub fn render_mel_stretch(
        &self,
        ctx: &RenderContext<'_>,
        playback_rate: f64,
        formant_shift_curve: Option<&Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        let fp = ctx.frame_period_ms;
        let clip_start = ctx.clip_start_sec;
        let pitch_edit = ctx.pitch_edit;
        let clip_midi = ctx.clip_midi;

        if clip_midi.is_empty() {
            if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
                eprintln!(
                    "HiFiGanRenderer::render_mel_stretch: clip_midi is empty, \
                     skipping inference and returning original PCM"
                );
            }
            return Ok(ctx.mono_pcm.to_vec());
        }

        let sr = ctx.sample_rate;
        let chunk_sec = crate::nsf_hifigan_onnx::env_chunk_sec();
        let overlap_sec = crate::nsf_hifigan_onnx::env_overlap_sec();

        // 构造共振峰偏移回调
        let fp_local = fp.max(0.1);
        let time_to_idx_mul = 1000.0 / fp_local;

        let formant_shift_fn = move |abs_time_sec: f64| -> f32 {
            let Some(curve) = formant_shift_curve else {
                return 0.0;
            };
            if curve.is_empty() {
                return 0.0;
            }
            let idx_f = abs_time_sec.max(0.0) * time_to_idx_mul;
            if !idx_f.is_finite() {
                return 0.0;
            }
            let i0 = idx_f.floor().max(0.0) as usize;
            let i1 = (i0 + 1).min(curve.len().saturating_sub(1));
            let frac = (idx_f - i0 as f64).clamp(0.0, 1.0) as f32;
            let a = curve.get(i0).copied().unwrap_or(0.0);
            let b = curve.get(i1).copied().unwrap_or(a);
            a + (b - a) * frac
        };

        let result = crate::nsf_hifigan_onnx::infer_pitch_edit_chunked_mel_stretch(
            ctx.mono_pcm,
            sr,
            playback_rate,
            ctx.seg_start_sec,
            move |abs_time_sec| {
                let orig = clip_midi_at_time(fp, clip_start, clip_midi, abs_time_sec);
                if !(orig.is_finite() && orig > 0.0) {
                    return 0.0;
                }
                let target = match edit_midi_at_time_or_none(fp, pitch_edit, abs_time_sec) {
                    Some(v) => v,
                    None => orig,
                };
                if target.is_finite() && target > 0.0 {
                    target
                } else {
                    0.0
                }
            },
            formant_shift_fn,
            chunk_sec,
            overlap_sec,
        )?;

        Ok(result)
    }
}
