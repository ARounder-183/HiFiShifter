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
    eprintln!("{line}");
    let log_path = std::env::temp_dir().join("hifishifter-formant-debug.log");
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
    {
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

pub fn make_formant_cache_key(
    clip_id: &str,
    source_path: &Path,
    out_rate: u32,
    source_start_sec: f64,
    source_end_sec: f64,
    reversed: bool,
    params: &ClipFormantMorph,
) -> FormantCacheKey {
    FormantCacheKey {
        clip_id: clip_id.to_string(),
        source_path: source_path.to_path_buf(),
        out_rate,
        source_start_q: quantize_i64(source_start_sec, 1000.0),
        source_end_q: quantize_i64(source_end_sec, 1000.0),
        reversed,
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

    let source_start_sec = clip.source_start_sec.max(0.0);
    let total_sec = crate::mixdown::clip_duration_sec_from_wav(in_rate, in_channels, &pcm)
        .ok_or_else(|| "cannot_determine_clip_duration".to_string())?;
    let source_end_sec = clip.source_end_sec.min(total_sec).max(source_start_sec);
    if source_end_sec - source_start_sec <= 1e-9 {
        return Err("trimmed_clip_too_short".to_string());
    }

    let src_i0 = (source_start_sec * in_rate as f64).floor().max(0.0) as usize;
    let src_i1 = ((source_end_sec * in_rate as f64)
        .ceil()
        .max(src_i0 as f64) as usize)
        .min(in_frames);
    if src_i1 <= src_i0 + 1 {
        return Err("source_slice_too_short".to_string());
    }

    let segment = &pcm[(src_i0 * in_channels_usize)..(src_i1 * in_channels_usize)];
    let mut segment =
        crate::mixdown::linear_resample_interleaved(segment, in_channels_usize, in_rate, out_rate);

    if clip.reversed {
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
        clip.source_start_sec.max(0.0),
        clip.source_end_sec,
        clip.reversed,
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
            formant_debug_log(format!(
                "cache hit clip_id={} frames={} sr={}",
                key.clip_id, entry.frames, entry.sample_rate
            ));
            return Ok(entry.clone());
        }
    }

    let processed =
        crate::formant_morph::apply_formant_morph_interleaved(input_stereo, sample_rate, 2, params)?;
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
            &ClipFormantMorph {
                enabled: true,
                target_f1_hz: 750.0,
                target_f2_hz: 1700.0,
                strength: 0.5,
            },
        );
        assert_ne!(a, b);
    }
}
