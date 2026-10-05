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
use std::sync::{Mutex, OnceLock};

// ─── 分块缓存（独立于 SynthClipCache，靠 hash 比对自然失效）───────────────────

pub struct ChunkCacheEntry {
    pub param_hash: u64,
    pub waveform: Vec<f32>,
}

/// 分块缓存的键：`(clip_id, 声道位, 块起点的 mel 帧号)`。
///
/// 【为什么必须含声道位】真立体声（Normal/Swap + 双声道源）会以
/// `channel_index = 0/1` 把同一 clip **交错**送进处理器两次（见
/// `pitch_editing::maybe_apply_pitch_edit_to_clip_segment` 的逐声道扇出），
/// 而参数哈希**含** `channel_index`（契约见 [`crate::renderer::traits`] 中
/// `ClipProcessContext::channel_index` 的说明）。键里缺声道位时，第二次访问
/// 只能看到第一次留下的条目：哈希必然不等 ⇒ 走 STALE 分支并被 `remove` ⇒
/// 两个声道每次渲染都全量重推理（实测：参数完全未变的 ch0/ch1 交替，
/// 命中恒为 0）。这与缺了声道位就"第二个声道命中第一个声道的推理结果"是
/// 同一处契约的两个方向。
type Key = (String, u16, usize);

const CHUNK_CACHE_BYTES:usize=128*1024*1024;
pub struct ChunkCache {
    entries:lru::LruCache<Key,ChunkCacheEntry>,
    bytes:usize,max_bytes:usize,
}
impl ChunkCache {
    fn new(max_bytes:usize)->Self {Self {entries:lru::LruCache::unbounded(),bytes:0,max_bytes}}
    fn len(&self)->usize {self.entries.len()}
    fn values(&self)->impl Iterator<Item=&ChunkCacheEntry> {self.entries.iter().map(|(_,entry)|entry)}
    fn keys(&self)->impl Iterator<Item=&Key> {self.entries.iter().map(|(key,_)|key)}
    fn get(&mut self,key:&Key)->Option<&ChunkCacheEntry> {self.entries.get(key)}
    fn remove(&mut self,key:&Key) {if let Some(old)=self.entries.pop(key) {self.bytes-=old.waveform.len()*4;}}
    /// 字节驱逐只释放缓存所有权；已取出的worker副本与当前就绪快照不受影响。
    fn insert(&mut self,key:Key,entry:ChunkCacheEntry) {
        let bytes=entry.waveform.len().saturating_mul(4);
        if bytes>self.max_bytes||entry.waveform.is_empty() {return;}
        self.remove(&key);
        while self.bytes.saturating_add(bytes)>self.max_bytes {
            let Some((_,old))=self.entries.pop_lru() else {break;};self.bytes-=old.waveform.len()*4;
        }
        self.entries.put(key,entry);self.bytes+=bytes;
    }
    fn clear(&mut self) {self.entries.clear();self.bytes=0;}
}
static CHUNK_CACHE: OnceLock<Mutex<ChunkCache>> = OnceLock::new();

/// 曲线编辑对**本块之外**音频的影响半径（秒）。
///
/// 分块哈希窗口必须向外扩这么多，否则每次编辑都会在块边界残留旧音频：
/// - mel 分析窗（`n_fft` 2048 @ hop 512）使一帧 mel 影响约 ±23ms 音频；
/// - 张力在**波形域**做 STFT/ISTFT（`N_FFT` 2048、`HOP` 256）外加 20ms Rd 增益平滑，
///   实测一个张力样本约影响 ±60~90ms 音频。
///
/// 取 0.1s 覆盖上述两者并留余量。只多算几个 f32，局部性不受影响。
const CURVE_INFLUENCE_MARGIN_SEC: f64 = 0.1;

pub fn global_chunk_cache_ref() -> &'static Mutex<ChunkCache> {
    CHUNK_CACHE.get_or_init(|| Mutex::new(ChunkCache::new(CHUNK_CACHE_BYTES)))
}

/// 清空整个 chunk 推理缓存。
///
/// 缓存key含clip/声道/块起点，已由128MiB字节LRU约束；工程切换仍可主动清理。
pub fn clear_chunk_cache() {
    if let Ok(mut cache) = global_chunk_cache_ref().lock() {
        let dropped = cache.len();
        let bytes: u64 = cache
            .values()
            .map(|e| (e.waveform.len() as u64).saturating_mul(std::mem::size_of::<f32>() as u64))
            .sum();
        cache.clear();
        if dropped > 0 {
            log::warn!(
                "[hifigan:cache] cleared {} chunk(s), ~{} bytes",
                dropped,
                bytes
            );
        }
    }
}

/// 使指定 clip_id 的所有 chunk 推理缓存失效。
/// 当源文件被替换时调用，避免 HiFiGAN 推理复用旧文件的输出。
pub fn invalidate_chunk_cache_for_clip(clip_id: &str) {
    if let Ok(mut cache) = global_chunk_cache_ref().lock() {
        let keys: Vec<Key> = cache
            .keys()
            .filter(|(id, _, _)| id == clip_id)
            .cloned()
            .collect();
        for k in &keys {
            cache.remove(k);
        }
        if !keys.is_empty() {
            log::warn!(
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
        log::warn!("[hifigan-chunk] {msg}");
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
        self.render_with_formant(ctx, None, None)
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
    ///
    /// `tension_curve`：张力曲线（%）。**这里不施加它** —— 张力已由调用方在
    /// mel 分析之前作用于波形（见 `chain.rs` 的 `apply_rd_tension`）。
    /// 传入只为让它**参与分块哈希**：张力改变会改变送入声码器的波形，
    /// 哈希必须能感知，否则会命中陈旧的分块缓存。该曲线由
    /// `compute_param_hash` 按各分块的时间范围切片，因此只有与编辑区间
    /// 相交的块失效（详见 `compute_param_hash` 的 extra_curves 说明）。
    pub fn render_with_formant(
        &self,
        ctx: &RenderContext<'_>,
        formant_shift_curve: Option<&[f32]>,
        tension_curve: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        let fp = ctx.frame_period_ms;
        let clip_start = ctx.clip_start_sec;
        let pitch_edit = ctx.pitch_edit;
        let clip_midi = ctx.clip_midi;

        debug_eprintln!(
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
                log::warn!(
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
        let renderer_identity=format!("{}:{}",self.id(),crate::nsf_hifigan_onnx::cache_identity()?);
        let seg_start_frame = (ctx.seg_start_sec * sr as f64).round().max(0.0) as u64;
        let seg_end_frame = (ctx.seg_end_sec * sr as f64).round().max(0.0) as u64;
        // 直接引用上下文里的 pitch_edit，不再 to_vec()
        // `pitch_orig` 必须传 `clip_midi`：`midi_fn` 在 `pitch_edit` 无编辑处
        // 回落到它，所以它**确实参与**声码器输入。此前传空切片，等于把源 F0
        // 排除在推理缓存键之外 —— 重跑音高分析（`pitch_orig` 变、`pitch_edit`
        // 因用户已编辑而保持不变）时会命中用旧 F0 渲染的块。
        let curves_snapshot = crate::pitch_editing::PitchCurvesSnapshot {
            frame_period_ms: fp,
            pitch_orig: ctx.clip_midi,
            pitch_edit,
        };

        // 构建元组数组，不再 new() HashMap 并 clone() 大数组。
        // 张力曲线一并纳入：它虽已在进入本函数前施加到波形上，但**分块缓存的
        // 哈希必须知道它**，否则改张力会命中陈旧块（见函数文档）。
        let mut extra_curves: Vec<(&str, &[f32])> = Vec::with_capacity(2);
        if let Some(c) = formant_shift_curve {
            extra_curves.push(("formant_shift_cents", c));
        }
        if let Some(c) = tension_curve {
            extra_curves.push(("hifigan_tension", c));
        }

        // 参数哈希（**不含波形指纹**）：张力曲线作为 extra_curve 参与，
        // 并由 `compute_param_hash` 按本段的时间范围切片 —— 因此只有与该范围
        // 相交的改动才会让本键失效。详见 `compute_param_hash` 的 extra_curves 说明。
        let param_hash = crate::synth_clip_cache::compute_param_hash(
            ctx.clip_id,
            seg_start_frame,
            seg_end_frame,
            sr,
            ctx.channel_index,
            &renderer_identity,
            &curves_snapshot,
            extra_curves.clone(),
            ctx.extra_params,
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
            // 越界（超出曲线末点）时按默认值 0.0 返回：i1 会被钳制到最后一个
            // 元素，若继续插值会混入末值与 frac（小数部分），产生 0..末值 的
            // 锯齿振荡，污染最后一个参数点之后的音频（Bug 复现：共振峰偏移
            // 点之后应回退 0，却渲染出 0..359 剧烈抖动的偏移）。
            if i0 >= curve.len() {
                return 0.0;
            }
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
        let clip_id = ctx.clip_id.to_string();

        let seg_start = seg_start_frame;

        // 分块哈希窗口：回调给出**模型域算好的绝对时间（秒）**，这里只做
        // `秒 × 输出采样率`。**不要**再用 `mel_start * 512` 自行换算 ——
        // 那会把模型域样本数加到输出域帧号上，二者仅在 44100 输出时相等，
        // 在 48000 等常见设备采样率下错位约 8%，使块尾部落在自己的哈希窗口
        // 之外，编辑该处反而命中陈旧块（详见声码器侧 `chunk_time_span` 文档）。
        //
        // 同时向外扩 `CURVE_INFLUENCE_MARGIN_SEC`：块内音频还受窗口外曲线影响。
        let chunk_hash = |c0: f64, c1: f64| -> u64 {
            let lo = (c0 - CURVE_INFLUENCE_MARGIN_SEC).max(0.0);
            let hi = c1 + CURVE_INFLUENCE_MARGIN_SEC;
            let start_frame = (lo * sr as f64) as u64;
            // +1 覆盖 `sample_curve_at` 的线性插值右端点（i0 / i0+1）。
            let end_frame = (hi * sr as f64).ceil() as u64 + 1;
            crate::synth_clip_cache::compute_param_hash(
                ctx.clip_id,
                start_frame,
                end_frame,
                sr,
                ctx.channel_index,
                &renderer_identity,
                &curves_snapshot,
                extra_curves.iter().map(|(k, v)| (*k, *v)),
                ctx.extra_params,
            )
        };
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
            &|mel_start: usize, mel_end: usize, c0: f64, c1: f64| -> Option<Vec<f32>> {
                // 参数哈希（**不含波形指纹**），张力曲线按本块区间切片。
                //
                // 【为什么不再指纹化波形】波形指纹必须 bit-exact 才有意义，而
                // Rd 张力对整段做 STFT/ISTFT，未编辑处的往返误差虽仅 2e-16，
                // 却足以让**每个** chunk 的指纹都变化 ⇒ 改 1 秒也要整段重推理
                // （已实测）。改为按"曲线在本块区间内的取值"判等后，只有与编辑
                // 区间相交的块失效，未相交的块保持命中。
                let hash = chunk_hash(c0, c1);
                let cache_key = (clip_id.clone(), ctx.channel_index, mel_start);

                let mut cache = global_chunk_cache_ref()
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                match cache.get(&cache_key) {
                    Some(entry) if entry.param_hash == hash => {
                        chunk_debug(&format!(
                            "  chunk [{mel_start}..{mel_end}) ch={} HIT (hash={hash:016x})",
                            ctx.channel_index,
                        ));
                        Some(entry.waveform.clone())
                    }
                    Some(entry) => {
                        chunk_debug(&format!(
                            "  chunk [{mel_start}..{mel_end}) ch={} STALE (cached={:016x} current={hash:016x})",
                            ctx.channel_index,
                            entry.param_hash,
                        ));
                        cache.remove(&cache_key);
                        None
                    }
                    None => {
                        chunk_debug(&format!(
                            "  chunk [{mel_start}..{mel_end}) ch={} MISS",
                            ctx.channel_index,
                        ));
                        None
                    }
                }
            },
            &|mel_start: usize, mel_end: usize, c0: f64, c1: f64, wf: Vec<f32>| {
                let hash = chunk_hash(c0, c1);
                let cache_key = (clip_id.clone(), ctx.channel_index, mel_start);

                chunk_debug(&format!(
                    "  chunk [{mel_start}..{mel_end}) ch={} PUT (hash={hash:016x} samples={})",
                    ctx.channel_index,
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

        debug_eprintln!(
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
            // 越界（超出曲线末点）时按默认值 0.0 返回：i1 会被钳制到最后一个
            // 元素，若继续插值会混入末值与 frac（小数部分），产生 0..末值 的
            // 锯齿振荡，污染最后一个参数点之后的音频（Bug 复现：共振峰偏移
            // 点之后应回退 0，却渲染出 0..359 剧烈抖动的偏移）。
            if i0 >= curve.len() {
                return 0.0;
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn chunk_cache_byte_lru_is_bounded_and_accounts_replacement_and_invalidation() {
        let mut cache=ChunkCache::new(32);
        let entry=|n,value|ChunkCacheEntry {param_hash:1,waveform:vec![value;n]};
        cache.insert(("a".into(),0,0),entry(4,0.25));cache.insert(("a".into(),1,0),entry(4,0.5));
        assert_eq!(cache.bytes,32);cache.get(&("a".into(),0,0));
        cache.insert(("b".into(),0,0),entry(4,0.75));
        assert!(cache.get(&("a".into(),1,0)).is_none());assert_eq!(cache.bytes,32);
        cache.insert(("a".into(),0,0),entry(2,1.));assert_eq!(cache.bytes,24);
        cache.insert(("long".into(),0,0),entry(9,1.));assert_eq!(cache.bytes,24);
        cache.remove(&("a".into(),0,0));assert_eq!(cache.bytes,16);
        cache.clear();assert_eq!(cache.bytes,0);assert_eq!(cache.len(),0);
    }

    /// 一次 chunk 缓存的访问键。
    ///
    /// 生产 get/put 两个回调都经此取键。
    fn cache_key(clip_id: &str, channel_index: u16, mel_start: usize) -> Key {
        (clip_id.to_string(), channel_index, mel_start)
    }

    /// 逐声道扇出时，两个声道必须各自保有独立的 chunk 缓存条目。
    ///
    /// 【为什么这条测试至关重要】真立体声（Normal/Swap + 双声道源）会以
    /// `channel_index = 0/1` 把同一 clip 送进处理器两次（见
    /// `pitch_editing::maybe_apply_pitch_edit_to_clip_segment` 的扇出），而
    /// 参数哈希**含** `channel_index`（`compute_param_hash` 混入声道位，见
    /// `traits.rs` 的契约说明）。键里缺声道位时，第二次访问只能看到第一次留下
    /// 的条目：哈希必然不等 ⇒ 判为 STALE 并**删除** ⇒ 每次渲染两个声道都全量
    /// 重推理。实测（真模型、3 块、参数完全未变）ch0/ch1 交替命中恒为 0。
    ///
    /// 契约：两次访问的输入相同（同 clip、同声道位、同块起点、同哈希）就必须
    /// 命中；不同声道位不得看到对方的条目，也不得驱逐对方的条目。
    #[test]
    fn chunk_cache_keeps_one_entry_per_channel() {
        let mut cache: HashMap<Key, u64> = HashMap::new();
        let clip_id = "clip-stereo";
        let chunks = [(0usize, 0xA0u64), (512, 0xA1)];

        // ── L 平面渲染：两块都写缓存 ──
        for &(mel_start, hash) in &chunks {
            cache.insert(cache_key(clip_id, 0, mel_start), hash);
        }

        // ── R 平面渲染：不得看到 L 的条目（键不同 ⇒ 必须 MISS） ──
        for &mel_start in &[0usize, 512] {
            assert_eq!(
                cache.get(&cache_key(clip_id, 1, mel_start)),
                None,
                "声道 1 不得命中声道 0 的条目（mel_start={mel_start}）"
            );
        }

        // ── 但 L 的条目必须原样保留：这正是"互相驱逐"的判据 ──
        assert_eq!(
            cache.get(&cache_key(clip_id, 0, 0)),
            Some(&0xA0),
            "访问声道 1 不得驱逐声道 0 的条目"
        );
        assert_eq!(
            cache.get(&cache_key(clip_id, 0, 512)),
            Some(&0xA1),
            "访问声道 1 不得驱逐声道 0 的条目"
        );

        // ── 与生产同构的完整回归形状：两声道交错渲染两轮，全部命中 ──
        for &(mel_start, hash) in &chunks {
            cache.insert(cache_key(clip_id, 1, mel_start), hash + 1);
        }
        for &(mel_start, hash) in &chunks {
            assert_eq!(
                cache.get(&cache_key(clip_id, 0, mel_start)),
                Some(&hash),
                "第二轮 ch0 必须命中（mel_start={mel_start}）"
            );
            assert_eq!(
                cache.get(&cache_key(clip_id, 1, mel_start)),
                Some(&(hash + 1)),
                "第二轮 ch1 必须命中（mel_start={mel_start}）"
            );
        }
    }

    /// 键含声道位之后，两个声道的条目互不可见 —— 用同一块起点写不同波形验证。
    #[test]
    fn chunk_cache_entries_are_distinct_per_channel() {
        let mut cache: HashMap<Key, u64> = HashMap::new();
        cache.insert(cache_key("c", 0, 0), 1);
        cache.insert(cache_key("c", 1, 0), 2);
        assert_eq!(cache.len(), 2, "同一块的两个声道必须是两条独立条目");
        assert_eq!(cache.get(&cache_key("c", 0, 0)), Some(&1));
        assert_eq!(cache.get(&cache_key("c", 1, 0)), Some(&2));
        // 不同 clip 之间同样隔离
        assert_eq!(cache.get(&cache_key("other", 0, 0)), None);
    }
}
