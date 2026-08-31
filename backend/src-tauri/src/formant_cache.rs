use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::audio_engine::byte_budget_cache::ByteBudgetCache;
use crate::state::ClipFormantMorph;

const DEFAULT_CAPACITY: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FormantCacheKey {
    pub clip_id: String,
    pub source_path: PathBuf,
    pub out_rate: u32,
    pub source_start_q: i64,
    pub source_end_q: i64,
    pub reversed: bool,
    /// 缓冲域判别：`false` = 实时域（完整文件自然顺序 / 非 Loop 窗口切片，
    /// 方向可由 `reversed` 预反转）；`true` = 离线回绕平铺域（mixdown /
    /// render_single_clip 先按整文件 floor_mod 平铺再处理的 segment）。
    ///
    /// 两个域对同一 clip 的输入内容不同（实时 [0,D] 自然顺序 vs 离线
    /// 锚点起回绕、长度为 clip 消费量的平铺内容），绝不能共享缓存条目 ——
    /// 否则先计算的一方会毒化另一方（离线导出拿到未回绕的整文件音频、
    /// 或实时把平铺内容当作整文件换入），产生错误输出。
    pub tiled_wrap: bool,
    pub enabled: bool,
    pub target_f1_q: u32,
    pub target_f2_q: u32,
    pub strength_q: u32,
}

#[derive(Debug, Clone)]
pub struct FormantCacheEntry {
    pub pcm_stereo: Arc<Vec<f32>>,
    pub frames: usize,
    pub sample_rate: u32,
}

pub struct FormantCache {
    inner: ByteBudgetCache<FormantCacheKey, FormantCacheEntry>,
}

impl FormantCache {
    pub fn new(capacity: usize, budget_bytes: u64) -> Self {
        Self {
            inner: ByteBudgetCache::new(capacity, budget_bytes),
        }
    }

    pub fn get(&mut self, key: &FormantCacheKey) -> Option<&FormantCacheEntry> {
        self.inner.get(key)
    }

    pub fn insert(&mut self, key: FormantCacheKey, entry: FormantCacheEntry) {
        let weight = entry.pcm_stereo.len() as u64 * 4;
        self.inner.insert(key, entry, weight);
    }

    /// 使指定 clip_id 的所有缓存失效。
    pub fn invalidate(&mut self, clip_id: &str) {
        self.inner.invalidate_where(|k| k.clip_id == clip_id);
    }
}

static GLOBAL_FORMANT_CACHE: OnceLock<Mutex<FormantCache>> = OnceLock::new();
static FORMANT_REBUILD_GENERATIONS: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();

pub fn global_formant_cache() -> &'static Mutex<FormantCache> {
    GLOBAL_FORMANT_CACHE.get_or_init(|| {
        let budget = crate::audio_engine::byte_budget_cache::env_cache_budget_bytes() / 8;
        Mutex::new(FormantCache::new(DEFAULT_CAPACITY, budget))
    })
}

pub fn formant_debug_enabled() -> bool {
    std::env::var("HIFISHIFTER_DEBUG_FORMANT").ok().as_deref() == Some("1")
}

pub fn average_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .take(len)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .sum::<f32>()
        / len as f32
}

pub fn formant_debug_log(message: impl AsRef<str>) {
    if !formant_debug_enabled() {
        return;
    }
    let line = format!("[formant] {}", message.as_ref());
    log::warn!("{line}");
    let log_path = std::env::temp_dir().join("hifishifter-formant-debug.log");
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(log_path) {
        let _ = writeln!(file, "{line}");
    }
}

fn global_formant_rebuild_generations() -> &'static Mutex<HashMap<String, u64>> {
    FORMANT_REBUILD_GENERATIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn quantize_i64(value: f64, scale: f64) -> i64 {
    (value * scale).round() as i64
}

fn quantize_u32(value: f64, scale: f64) -> u32 {
    (value * scale).round().clamp(0.0, u32::MAX as f64) as u32
}

#[allow(clippy::too_many_arguments)]
pub fn make_formant_cache_key(
    clip_id: &str,
    source_path: &Path,
    out_rate: u32,
    source_start_sec: f64,
    source_end_sec: f64,
    reversed: bool,
    tiled_wrap: bool,
    params: &ClipFormantMorph,
) -> FormantCacheKey {
    FormantCacheKey {
        clip_id: clip_id.to_string(),
        source_path: source_path.to_path_buf(),
        out_rate,
        source_start_q: quantize_i64(source_start_sec, 1000.0),
        source_end_q: quantize_i64(source_end_sec, 1000.0),
        reversed,
        tiled_wrap,
        enabled: params.enabled,
        target_f1_q: quantize_u32(params.target_f1_hz, 10.0),
        target_f2_q: quantize_u32(params.target_f2_hz, 10.0),
        strength_q: quantize_u32(params.strength, 1000.0),
    }
}

pub fn begin_formant_rebuild_generation(clip_id: &str) -> u64 {
    let mut generations = global_formant_rebuild_generations()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let next = generations
        .get(clip_id)
        .copied()
        .unwrap_or(0)
        .saturating_add(1);
    generations.insert(clip_id.to_string(), next);
    next
}

pub fn is_current_formant_rebuild_generation(clip_id: &str, generation: u64) -> bool {
    let generations = global_formant_rebuild_generations()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    generations.get(clip_id).copied().unwrap_or(0) == generation
}

/// 使指定 clip 的所有 formant 缓存失效（例如源文件被替换时）。
pub fn invalidate_formant_cache_for_clip(clip_id: &str) {
    if let Ok(mut cache) = global_formant_cache().lock() {
        cache.invalidate(clip_id);
    }
    // 源共振峰分析缓存同步失效（键内含 clip_id）
    if let Ok(mut cache) = global_formant_analysis_cache().lock() {
        cache.retain(|k, _| k.clip_id != clip_id);
    }
}

// ── 源共振峰分析缓存（供前端可视化，与音频 FormantCache 分开） ───────────

/// 分析缓存条目键：clip + 源文件 + mtime + 消费窗口（1 ms 量化）。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FormantAnalysisKey {
    clip_id: String,
    source_path: PathBuf,
    mtime: Option<u64>,
    window_start_q: i64,
    window_end_q: i64,
}

/// 分析缓存容量上限（条目极小，仅防无限增长）。
const FORMANT_ANALYSIS_CACHE_CAP: usize = 256;

fn global_formant_analysis_cache()
-> &'static Mutex<std::collections::HashMap<FormantAnalysisKey, crate::formant_morph::analysis::FormantAnalysisSummary>>
{
    static CACHE: OnceLock<
        Mutex<std::collections::HashMap<FormantAnalysisKey, crate::formant_morph::analysis::FormantAnalysisSummary>>,
    > = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// 获取（或计算）clip 的源共振峰分析。
///
/// 流程：
/// 1. 组装缓存键（clip_id + 源路径 + mtime + 消费窗口 1ms 量化）。
/// 2. 未命中：解码源音频 → 消费窗口切片（倒放预反转，与实时域一致）→
///    通道平均为 mono → 调用与 DSP 同源的 `analyze_clip_formants`。
/// 3. 结果写入缓存并返回。
pub fn get_or_compute_formant_analysis(
    clip: &crate::state::Clip,
) -> Result<crate::formant_morph::analysis::FormantAnalysisSummary, String> {
    let source_path = clip
        .source_path
        .as_ref()
        .ok_or_else(|| "clip_has_no_source_path".to_string())?;
    let (win_start, win_end) = crate::state::clip_playback_window_sec(clip);
    let key = FormantAnalysisKey {
        clip_id: clip.id.clone(),
        source_path: PathBuf::from(source_path),
        mtime: clip.source_file_mtime,
        window_start_q: (win_start * 1000.0).round() as i64,
        window_end_q: (win_end * 1000.0).round() as i64,
    };

    {
        let cache = global_formant_analysis_cache()
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        if let Some(hit) = cache.get(&key) {
            return Ok(hit.clone());
        }
    }

    let (in_rate, in_channels, pcm) =
        crate::audio_utils::decode_audio_f32_interleaved(Path::new(source_path))?;
    let ch = (in_channels as usize).max(1);
    let frames = pcm.len() / ch;
    if frames < 2 {
        return Err("source_audio_too_short".to_string());
    }
    let total_sec = frames as f64 / in_rate as f64;
    let s0 = win_start.max(0.0).min(total_sec);
    let s1 = win_end.min(total_sec).max(s0);
    let i0 = (s0 * in_rate as f64).floor() as usize;
    let i1 = ((s1 * in_rate as f64).ceil() as usize).min(frames);
    if i1 <= i0 + 1 {
        return Err("source_slice_too_short".to_string());
    }

    let mut slice = pcm[i0 * ch..i1 * ch].to_vec();
    if clip.reversed {
        crate::mixdown::reverse_interleaved_frames(&mut slice, ch);
    }
    // 通道平均 → mono
    let n = slice.len() / ch;
    let mut mono = vec![0.0_f32; n];
    let inv_ch = 1.0 / ch as f32;
    for (frame_idx, slot) in mono.iter_mut().enumerate() {
        let mut sum = 0.0_f32;
        for c in 0..ch {
            sum += slice[frame_idx * ch + c];
        }
        *slot = sum * inv_ch;
    }

    let summary = crate::formant_morph::analysis::analyze_clip_formants(&mono, in_rate);

    let mut cache = global_formant_analysis_cache()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    if cache.len() >= FORMANT_ANALYSIS_CACHE_CAP {
        cache.clear();
    }
    cache.insert(key, summary.clone());
    Ok(summary)
}

pub fn cancel_formant_rebuild_generation(clip_id: &str) {
    let mut generations = global_formant_rebuild_generations()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let next = generations
        .get(clip_id)
        .copied()
        .unwrap_or(0)
        .saturating_add(1);
    generations.insert(clip_id.to_string(), next);
    formant_debug_log(format!(
        "cancel rebuild generation clip_id={} next={}",
        clip_id, next
    ));
}

pub fn insert_formant_cache_entry(key: FormantCacheKey, entry: FormantCacheEntry) {
    formant_debug_log(format!(
        "cache insert clip_id={} frames={} sr={}",
        key.clip_id, entry.frames, entry.sample_rate
    ));
    let mut cache = global_formant_cache()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    cache.insert(key, entry);
}

pub fn compute_formant_cache_entry_for_clip(
    clip: &crate::state::Clip,
    out_rate: u32,
) -> Result<(FormantCacheKey, FormantCacheEntry), String> {
    let params = clip
        .formant_morph
        .as_ref()
        .filter(|params| params.enabled)
        .ok_or_else(|| "formant_morph_disabled".to_string())?;
    let source_path = clip
        .source_path
        .as_ref()
        .ok_or_else(|| "clip_has_no_source_path".to_string())?;

    let (in_rate, in_channels, pcm) =
        crate::audio_utils::decode_audio_f32_interleaved(Path::new(source_path))?;
    let in_channels_usize = in_channels as usize;
    let in_frames = pcm.len() / in_channels_usize;
    if in_frames < 2 {
        return Err("source_audio_too_short".to_string());
    }

    let total_sec = crate::mixdown::clip_duration_sec_from_wav(in_rate, in_channels, &pcm)
        .ok_or_else(|| "cannot_determine_clip_duration".to_string())?;
    // 消费窗口模型（与 build_snapshot 实时分支及离线渲染完全一致）：
    //   正放 win = [ss, ss+len·r)；倒放 win = [se−len·r, se)。
    // 切片 clamp 到媒体内；缓存键使用**未 clamp** 的窗口值，与 snapshot
    // 的查找键逐字节成对（此前预计算键尾取原始 se、快照键尾取派生值，
    // 陈旧窗口的正放 clip 两键永不匹配 → 预计算白算且状态闪烁）。
    let (raw_win_start_sec, raw_win_end_sec) = crate::state::clip_playback_window_sec(clip);
    let source_start_sec = raw_win_start_sec.max(0.0);
    let source_end_sec = raw_win_end_sec.min(total_sec).max(source_start_sec);

    // Loop（循环源）：与 build_snapshot 的实时 Formant 分支保持一致 ——
    // 处理对象是**完整文件的自然顺序** PCM（方向由 mix 阶段的锚点回绕体现，
    // 不预反转），缓存键取 [0, 文件时长]。非 Loop 保持窗口切片 + 预反转。
    // 此前本函数不感知 Loop：对 Loop clip 仍按窗口切片并整体反转，产出的
    // 键 ([start,end], reversed) 永远不会被快照查找键 ([0,D], false) 命中，
    // 既白白消耗一次全量 formant 计算，还会向前端误报 rebuilding/failed 状态。
    let loop_mode = clip.loop_enabled;
    let (slice_start_sec, slice_end_sec) = if loop_mode {
        (0.0f64, total_sec)
    } else {
        if source_end_sec - source_start_sec <= 1e-9 {
            return Err("trimmed_clip_too_short".to_string());
        }
        (source_start_sec, source_end_sec)
    };

    let src_i0 = (slice_start_sec * in_rate as f64).floor().max(0.0) as usize;
    let src_i1 =
        ((slice_end_sec * in_rate as f64).ceil().max(src_i0 as f64) as usize).min(in_frames);
    if src_i1 <= src_i0 + 1 {
        return Err("source_slice_too_short".to_string());
    }

    let segment = &pcm[(src_i0 * in_channels_usize)..(src_i1 * in_channels_usize)];
    let mut segment =
        crate::mixdown::linear_resample_interleaved(segment, in_channels_usize, in_rate, out_rate);

    if clip.reversed && !loop_mode {
        // 注意：此处尚未转为立体声，通道数仍是源文件的实际通道数。
        crate::mixdown::reverse_interleaved_frames(&mut segment, in_channels_usize);
    }

    let segment = if in_channels == 1 {
        let mut stereo = Vec::with_capacity(segment.len() * 2);
        for sample in segment {
            stereo.push(sample);
            stereo.push(sample);
        }
        stereo
    } else if in_channels >= 2 {
        segment
            .chunks_exact(in_channels_usize)
            .flat_map(|chunk| [chunk[0], chunk[1]])
            .collect()
    } else {
        return Err("unsupported_channel_count".to_string());
    };

    let key = make_formant_cache_key(
        &clip.id,
        Path::new(source_path),
        out_rate,
        if loop_mode {
            0.0
        } else {
            raw_win_start_sec.max(0.0)
        },
        if loop_mode {
            // 与 snapshot 的查找键使用同一来源（优先 clip 元数据）——
            // 避免 wav 头时长与解码帧时长在 1ms 量化边界处错开键值。
            crate::state::clip_source_media_duration_sec(clip).unwrap_or(total_sec)
        } else {
            // 未 clamp 的消费窗口终点 —— 与 snapshot 查找键成对。
            raw_win_end_sec
        },
        clip.reversed && !loop_mode,
        false,
        params,
    );
    let processed =
        crate::formant_morph::apply_formant_morph_interleaved(&segment, out_rate, 2, params)?;
    formant_debug_log(format!(
        "rebuild compute clip_id={} enabled={} f1={:.1} f2={:.1} strength={:.3} frames={} diff={:.8}",
        clip.id,
        params.enabled,
        params.target_f1_hz,
        params.target_f2_hz,
        params.strength,
        processed.len() / 2,
        average_abs_diff(&segment, &processed),
    ));
    let entry = FormantCacheEntry {
        frames: processed.len() / 2,
        pcm_stereo: Arc::new(processed),
        sample_rate: out_rate,
    };
    Ok((key, entry))
}

pub fn get_or_compute_formant_audio(
    key: FormantCacheKey,
    input_stereo: &[f32],
    sample_rate: u32,
    params: &ClipFormantMorph,
) -> Result<FormantCacheEntry, String> {
    if !params.enabled {
        return Ok(FormantCacheEntry {
            pcm_stereo: Arc::new(input_stereo.to_vec()),
            frames: input_stereo.len() / 2,
            sample_rate,
        });
    }

    {
        let mut cache = global_formant_cache()
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        if let Some(entry) = cache.get(&key) {
            // 防御：命中条目的帧数必须与当前输入一致 —— 不一致说明键未覆盖
            // 某个内容维度（历史遗留条目或域泄漏），按未命中重新计算，
            // 绝不把错误长度的音频返回给调用方（会被静音截断/混入）。
            if entry.frames == input_stereo.len() / 2 && entry.sample_rate == sample_rate {
                formant_debug_log(format!(
                    "cache hit clip_id={} frames={} sr={}",
                    key.clip_id, entry.frames, entry.sample_rate
                ));
                return Ok(entry.clone());
            }
            formant_debug_log(format!(
                "cache shape-mismatch clip_id={} cached_frames={} input_frames={} → recompute",
                key.clip_id,
                entry.frames,
                input_stereo.len() / 2,
            ));
        }
    }

    let processed = crate::formant_morph::apply_formant_morph_interleaved(
        input_stereo,
        sample_rate,
        2,
        params,
    )?;
    formant_debug_log(format!(
        "cache miss compute clip_id={} enabled={} f1={:.1} f2={:.1} strength={:.3} frames={} diff={:.8}",
        key.clip_id,
        params.enabled,
        params.target_f1_hz,
        params.target_f2_hz,
        params.strength,
        processed.len() / 2,
        average_abs_diff(input_stereo, &processed),
    ));

    {
        let mut cache = global_formant_cache()
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        if let Some(entry) = cache.get(&key) {
            formant_debug_log(format!(
                "cache hit after background rebuild clip_id={} frames={} sr={}",
                key.clip_id, entry.frames, entry.sample_rate
            ));
            return Ok(entry.clone());
        }
    }

    let entry = FormantCacheEntry {
        frames: processed.len() / 2,
        pcm_stereo: Arc::new(processed),
        sample_rate,
    };
    insert_formant_cache_entry(key, entry.clone());
    Ok(entry)
}

#[cfg(test)]
mod tests {
    use super::make_formant_cache_key;
    use crate::state::ClipFormantMorph;
    use std::path::Path;

    #[test]
    fn formant_cache_key_changes_when_parameters_change() {
        let a = make_formant_cache_key(
            "clip-1",
            Path::new("demo.wav"),
            44_100,
            0.0,
            1.0,
            false,
            false,
            &ClipFormantMorph {
                enabled: true,
                target_f1_hz: 700.0,
                target_f2_hz: 1700.0,
                strength: 0.5,
            },
        );
        let b = make_formant_cache_key(
            "clip-1",
            Path::new("demo.wav"),
            44_100,
            0.0,
            1.0,
            false,
            false,
            &ClipFormantMorph {
                enabled: true,
                target_f1_hz: 750.0,
                target_f2_hz: 1700.0,
                strength: 0.5,
            },
        );
        assert_ne!(a, b);
    }

    #[test]
    fn formant_cache_key_separates_tiled_wrap_domain() {
        // 实时域（完整文件自然顺序）与离线回绕平铺域的其余键参数完全一致，
        // 仅靠 tiled_wrap 区分 —— 否则先计算的一方会毒化另一方。
        let realtime = make_formant_cache_key(
            "clip-1",
            Path::new("demo.wav"),
            44_100,
            0.0,
            10.0,
            false,
            false,
            &ClipFormantMorph {
                enabled: true,
                target_f1_hz: 700.0,
                target_f2_hz: 1700.0,
                strength: 0.5,
            },
        );
        let offline_tiled = make_formant_cache_key(
            "clip-1",
            Path::new("demo.wav"),
            44_100,
            0.0,
            10.0,
            false,
            true,
            &ClipFormantMorph {
                enabled: true,
                target_f1_hz: 700.0,
                target_f2_hz: 1700.0,
                strength: 0.5,
            },
        );
        assert_ne!(realtime, offline_tiled);
    }
}
