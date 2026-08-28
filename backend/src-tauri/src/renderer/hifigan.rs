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
//!
//! # 分块缓存
//!
//! 长音频推理使用独立的分块缓存，靠 `param_hash` 比对自然失效，
//! 不受 clip 级 `invalidate_clip_all_caches` 影响。

use super::traits::{RenderContext, Renderer, RendererCapabilities};
use super::utils::{clip_midi_at_time, edit_midi_at_time_or_none};
use crate::state::SynthPipelineKind;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// ─── 分块缓存（独立于 SynthClipCache，靠 hash 比对自然失效）───────────────────

pub struct ChunkCacheEntry {
    pub param_hash: u64,
    pub waveform: Vec<f32>,
}

static CHUNK_CACHE: OnceLock<Mutex<HashMap<(String, usize), ChunkCacheEntry>>> = OnceLock::new();

pub fn global_chunk_cache_ref() -> &'static Mutex<HashMap<(String, usize), ChunkCacheEntry>> {
    CHUNK_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 使指定 clip_id 的所有 chunk 推理缓存失效。
/// 当源文件被替换时调用，避免 HiFiGAN 推理复用旧文件的输出。
pub fn invalidate_chunk_cache_for_clip(clip_id: &str) {
    if let Ok(mut cache) = global_chunk_cache_ref().lock() {
        let keys: Vec<(String, usize)> = cache
            .keys()
            .filter(|(id, _)| id == clip_id)
            .cloned()
            .collect();
        for k in &keys {
            cache.remove(k);
        }
        if !keys.is_empty() {
            eprintln!(
                "[hifigan:cache] invalidated {} chunk(s) for clip_id={}",
                keys.len(),
                clip_id
            );
        }
    }
}

fn debug_enabled() -> bool {
    std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1")
}

fn chunk_debug(msg: &str) {
    if debug_enabled() {
        eprintln!("[hifigan-chunk] {msg}");
    }
}

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
        formant_shift_curve: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        let fp = ctx.frame_period_ms;
        let clip_start = ctx.clip_start_sec;
        let pitch_edit = ctx.pitch_edit;
        let clip_midi = ctx.clip_midi;

        eprintln!(
            "[hifigan] render_with_formant: clip_id={} samples={} seg=[{:.3},{:.3})",
            ctx.clip_id,
            ctx.mono_pcm.len(),
            ctx.seg_start_sec,
            ctx.seg_end_sec
        );

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
            .map(|c| vec![("formant_shift_cents", c)])
            .unwrap_or_default();

        let param_hash = crate::synth_clip_cache::compute_param_hash(
            ctx.clip_id,
            seg_start_frame,
            seg_end_frame,
            sr,
            self.id(),
            &curves_snapshot,
            extra_curves.clone(),
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

        let midi_fn = move |abs_time_sec| {
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
        };

        // ── 所有音频统一走分块优化路径（支持 per-chunk 缓存）─────────────
        let model_hop: u64 = 512;
        let clip_id = ctx.clip_id.to_string();

        let seg_start = seg_start_frame;
        chunk_debug(&format!(
            "chunked_opt path: clip={} samples={} seg_start_frame={}",
            clip_id,
            ctx.mono_pcm.len(),
            seg_start
        ));

        let result = crate::nsf_hifigan_onnx::infer_pitch_edit_chunked_optimized(
            ctx.mono_pcm,
            sr,
            ctx.seg_start_sec,
            midi_fn,
            formant_shift_fn,
            &|mel_start: usize, mel_end: usize| -> Option<Vec<f32>> {
                let chunk_start = seg_start + mel_start as u64 * model_hop;
                let chunk_end = seg_start + mel_end as u64 * model_hop;
                let hash = crate::synth_clip_cache::compute_param_hash(
                    &clip_id,
                    chunk_start,
                    chunk_end,
                    sr,
                    "nsf_hifigan_onnx",
                    &curves_snapshot,
                    extra_curves.iter().map(|(k, v)| (*k, *v)),
                    &std::collections::HashMap::new(),
                );
                let cache_key = (clip_id.clone(), mel_start);

                let mut cache = global_chunk_cache_ref()
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                match cache.get(&cache_key) {
                    Some(entry) if entry.param_hash == hash => {
                        chunk_debug(&format!(
                            "  chunk [{mel_start}..{mel_end}) HIT (hash={hash:016x})",
                        ));
                        Some(entry.waveform.clone())
                    }
                    Some(entry) => {
                        chunk_debug(&format!(
                            "  chunk [{mel_start}..{mel_end}) STALE (cached={:016x} current={hash:016x})",
                            entry.param_hash,
                        ));
                        cache.remove(&cache_key);
                        None
                    }
                    None => {
                        chunk_debug(&format!("  chunk [{mel_start}..{mel_end}) MISS",));
                        None
                    }
                }
            },
            &|mel_start: usize, mel_end: usize, wf: Vec<f32>| {
                let chunk_start = seg_start + mel_start as u64 * model_hop;
                let chunk_end = seg_start + mel_end as u64 * model_hop;
                let hash = crate::synth_clip_cache::compute_param_hash(
                    &clip_id,
                    chunk_start,
                    chunk_end,
                    sr,
                    "nsf_hifigan_onnx",
                    &curves_snapshot,
                    extra_curves.iter().map(|(k, v)| (*k, *v)),
                    &std::collections::HashMap::new(),
                );
                let cache_key = (clip_id.clone(), mel_start);

                chunk_debug(&format!(
                    "  chunk [{mel_start}..{mel_end}) PUT (hash={hash:016x} samples={})",
                    wf.len(),
                ));

                let mut cache = global_chunk_cache_ref()
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                cache.insert(
                    cache_key,
                    ChunkCacheEntry {
                        param_hash: hash,
                        waveform: wf,
                    },
                );
            },
        )?;
        Ok(result)
    }

    pub fn render_mel_stretch_with_formant(
        &self,
        ctx: &RenderContext<'_>,
        playback_rate: f64,
        formant_shift_curve: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        let fp = ctx.frame_period_ms;
        let clip_start = ctx.clip_start_sec;
        let pitch_edit = ctx.pitch_edit;
        let clip_midi = ctx.clip_midi;

        eprintln!(
            "[hifigan] render_mel_stretch: clip_id={} samples={} rate={:.3}",
            ctx.clip_id,
            ctx.mono_pcm.len(),
            playback_rate
        );

        if clip_midi.is_empty() {
            return Ok(ctx.mono_pcm.to_vec());
        }

        let chunk_sec = crate::nsf_hifigan_onnx::env_chunk_sec();
        let overlap_sec = crate::nsf_hifigan_onnx::env_overlap_sec();
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

        crate::nsf_hifigan_onnx::infer_pitch_edit_chunked_mel_stretch(
            ctx.mono_pcm,
            ctx.sample_rate,
            playback_rate.max(1e-6),
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
        )
    }
}
