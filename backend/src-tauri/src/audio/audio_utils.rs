use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::audio_engine::byte_budget_cache::ByteBudgetCache;

/// Decode any media file (WAV fast-path, everything else via Symphonia).
pub fn decode_audio_f32_interleaved(path: &Path) -> Result<(u32, u16, Vec<f32>), String> {
    if path.as_os_str().is_empty() {
        return Err("empty path".to_string());
    }

    let is_wav = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("wav"))
        .unwrap_or(false);

    if is_wav {
        if let Ok(v) = decode_wav_f32_interleaved_hound(path) {
            return Ok(v);
        }
    }

    crate::media::decode_media_audio_f32_interleaved(path, None)
}

fn decode_wav_f32_interleaved_hound(path: &Path) -> Result<(u32, u16, Vec<f32>), String> {
    use hound::{SampleFormat, WavReader};

    let mut reader = WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    if spec.sample_rate == 0 || spec.channels == 0 {
        return Err("invalid wav spec".to_string());
    }

    let channels = spec.channels;
    let sample_rate = spec.sample_rate;
    // hound::duration() 返回每声道的帧数（frames），总样本数 = frames * channels。
    // duration() 来自头字段，损坏文件可声称巨量帧数 → with_capacity 一次性
    // 分配失败 abort 进程；按文件大小钳制上界（总样本 ≤ 文件字节 / 最少每样本字节），
    // metadata 不可得时用 64M 样本的兜底上界。
    let min_bytes_per_sample = (spec.bits_per_sample as usize / 8).max(1);
    let file_bound = std::fs::metadata(path)
        .map(|m| (m.len() as usize).saturating_div(min_bytes_per_sample))
        .unwrap_or(1 << 22);
    let mut out: Vec<f32> =
        Vec::with_capacity(reader.duration().min(file_bound.min(u32::MAX as usize) as u32) as usize * channels as usize);

    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => {
            for s in reader.samples::<i16>() {
                let v = s.map_err(|e| e.to_string())? as f32 / i16::MAX as f32;
                out.push(v);
            }
        }
        (SampleFormat::Int, 24) => {
            // hound returns 24-bit PCM as sign-extended i32 in range [-2^23, 2^23-1].
            let denom = (1u32 << 23) as f32;
            for s in reader.samples::<i32>() {
                let v = s.map_err(|e| e.to_string())? as f32 / denom;
                out.push(v);
            }
        }
        (SampleFormat::Int, 32) => {
            for s in reader.samples::<i32>() {
                let v = s.map_err(|e| e.to_string())? as f32 / i32::MAX as f32;
                out.push(v);
            }
        }
        (SampleFormat::Float, 32) => {
            for s in reader.samples::<f32>() {
                out.push(s.map_err(|e| e.to_string())?);
            }
        }
        _ => return Err("unsupported wav format".to_string()),
    }

    Ok((sample_rate, channels, out))
}

pub struct WavInfo {
    pub sample_rate: u32,
    pub total_frames: u64, // 精确的frame总数
    pub duration_sec: f64, // 兼容性保留，从frames计算
    pub waveform_preview: Vec<f32>,
}

pub fn try_read_wav_info(path: &Path, preview_points: usize) -> Option<WavInfo> {
    // Prefer WAV fast-path via hound.
    if path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("wav"))
        .unwrap_or(false)
    {
        if let Some(info) = try_read_wav_info_hound(path, preview_points) {
            return Some(info);
        }
    }

    crate::media::probe_media(path, preview_points, None).map(|probe| WavInfo {
        sample_rate: probe.sample_rate,
        total_frames: probe.total_frames,
        duration_sec: probe.duration_sec,
        waveform_preview: probe.waveform_preview,
    })
}

/// 快速只读 sample_rate / total_frames / duration_sec，不生成 waveform_preview。
pub fn try_read_audio_header_only(path: &Path) -> Option<WavInfo> {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("wav"))
        .unwrap_or(false)
    {
        if let Some(info) = try_read_wav_info_hound(path, 0) {
            return Some(info);
        }
    }

    crate::media::probe_media(path, 0, None).map(|probe| WavInfo {
        sample_rate: probe.sample_rate,
        total_frames: probe.total_frames,
        duration_sec: probe.duration_sec,
        waveform_preview: vec![],
    })
}

fn try_read_wav_info_hound(path: &Path, preview_points: usize) -> Option<WavInfo> {
    use hound::{SampleFormat, WavReader};

    let mut reader = WavReader::open(path).ok()?;
    let spec = reader.spec();
    if spec.sample_rate == 0 || spec.channels == 0 {
        return None;
    }

    // hound::duration() 返回每声道的帧数（frames），直接就是 total_frames
    let total_frames = reader.duration() as u64;
    let duration_sec = total_frames as f64 / spec.sample_rate as f64;
    // total_samples 用于 preview 步长计算（逐样本迭代，包含所有声道）
    let total_samples = total_frames as usize * spec.channels as usize;

    let preview_len = preview_points.max(2);
    let mut preview = vec![0.0f32; preview_len];
    if total_frames == 0 || preview_points == 0 {
        return Some(WavInfo {
            sample_rate: spec.sample_rate,
            total_frames,
            duration_sec,
            waveform_preview: if preview_points == 0 { vec![] } else { preview },
        });
    }

    // Reset reader by reopening (hound doesn't support seek on all readers reliably).
    let step = (total_samples / preview_len).max(1);

    let mut idx = 0usize;
    let mut current_max = 0.0f32;
    let mut count = 0usize;

    let mut push_abs = |s: f32| {
        let a = s.abs();
        if a > current_max {
            current_max = a;
        }
        count += 1;
        if count >= step {
            preview[idx] = current_max;
            idx += 1;
            current_max = 0.0;
            count = 0;
        }
        idx < preview_len
    };

    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => {
            let scale = 1.0 / (i16::MAX as f32);
            for s in reader.samples::<i16>() {
                let v = s.ok()? as f32 * scale;
                if !push_abs(v) {
                    break;
                }
            }
        }
        (SampleFormat::Int, 24) => {
            // hound returns 24-bit PCM as sign-extended i32 in range [-2^23, 2^23-1].
            // Normalizing by i32::MAX would scale by ~1/256 and make waveform/audio nearly silent.
            let scale = 1.0 / ((1u32 << 23) as f32);
            for s in reader.samples::<i32>() {
                let v = s.ok()? as f32 * scale;
                if !push_abs(v) {
                    break;
                }
            }
        }
        (SampleFormat::Int, 32) => {
            let scale = 1.0 / (i32::MAX as f32);
            for s in reader.samples::<i32>() {
                let v = s.ok()? as f32 * scale;
                if !push_abs(v) {
                    break;
                }
            }
        }
        (SampleFormat::Float, 32) => {
            for s in reader.samples::<f32>() {
                let v = s.ok()?;
                if !push_abs(v) {
                    break;
                }
            }
        }
        _ => return None,
    }

    Some(WavInfo {
        sample_rate: spec.sample_rate,
        total_frames,
        duration_sec,
        waveform_preview: preview,
    })
}

// ─── 文件内容指纹（用于检测外部修改）─────────────────────────────────────────

/// 源文件的轻量内容指纹（FNV-1a 64-bit）。
///
/// 对文件头部 64KB + 尾部 64KB（若文件 < 128KB 则读取全文）计算哈希。
/// 用于在窗口聚焦时快速验证文件内容是否真正被外部修改，
/// 避免因云同步回写时间戳等纯元数据变更而产生误报。
pub fn compute_file_fingerprint(path: &Path) -> Option<u64> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let file_len = file.metadata().ok()?.len();

    const HEAD_TAIL_BYTES: u64 = 64 * 1024; // 64 KB

    let mut h: u64 = 14695981039346656037u64;
    let mut buf = vec![0u8; HEAD_TAIL_BYTES as usize];
    let mut hasher = |data: &[u8]| {
        for &b in data {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211u64);
        }
    };

    // 混入文件总长度
    hasher(&file_len.to_le_bytes());

    if file_len <= HEAD_TAIL_BYTES * 2 {
        let mut full = Vec::new();
        file.read_to_end(&mut full).ok()?;
        hasher(&full);
    } else {
        let n = file.read(&mut buf).ok()?;
        hasher(&buf[..n]);

        let seek_pos = file_len.saturating_sub(HEAD_TAIL_BYTES);
        std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(seek_pos)).ok()?;
        let n = file.read(&mut buf).ok()?;
        hasher(&buf[..n]);
    }

    Some(h)
}

// ─── 进程级源文件解码缓存（P1-3）─────────────────────────────────────────────

/// 已解码的源文件 PCM（**源采样率**、交错、原始通道数）。
///
/// 之所以缓存"未重采样"的原始 PCM：渲染与导出都需要在**源域**做裁剪/平铺
/// （切片边界依赖源采样率），重采样在其后进行。引擎侧另有一套"已重采样到
/// 设备采样率"的缓存（`audio_engine::resource_manager::DecodeCache`），两者
/// 面向不同需求，不互相替代。
pub struct DecodedSourceAudio {
    pub sample_rate: u32,
    pub channels: u16,
    pub pcm: Arc<Vec<f32>>,
}

/// key = (路径, 内容指纹)。指纹同时覆盖长度与首尾 64KB 内容，
/// 因此文件被替换/云同步改写时自动失效（mtime 被保留也不影响）。
type SourceCacheKey = (PathBuf, u64);

static SOURCE_CACHE: OnceLock<Mutex<ByteBudgetCache<SourceCacheKey, Arc<DecodedSourceAudio>>>> =
    OnceLock::new();

/// 条目数兜底上限（实际由字节预算主导淘汰）。
const SOURCE_CACHE_MAX_ENTRIES: usize = 128;

fn source_cache_budget_bytes() -> u64 {
    if let Some(mb) = std::env::var("HIFISHIFTER_SOURCE_CACHE_MB")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
    {
        return mb.saturating_mul(1024 * 1024);
    }
    // 与 `cache_registry::BUDGETED` 中 SourcePcmCache 的 1/4 份额保持一致。
    crate::audio_engine::byte_budget_cache::cache_budget_bytes() / 4
}

fn source_cache() -> &'static Mutex<ByteBudgetCache<SourceCacheKey, Arc<DecodedSourceAudio>>> {
    SOURCE_CACHE.get_or_init(|| {
        Mutex::new(ByteBudgetCache::new(
            SOURCE_CACHE_MAX_ENTRIES,
            source_cache_budget_bytes(),
        ))
    })
}

/// 解码源文件，命中缓存时零拷贝返回（`Arc` 共享同一份 PCM）。
///
/// 为什么需要：后台渲染、离线导出、共振峰预计算、音高分析都各自独立地调用
/// 解码。此前每次调用都重新读盘 + 解码，于是"同一源被 N 个 clip 各解码一次"
/// "同一 clip 被编辑 N 次各解码一次"，而解码大文件是秒级开销（见 P1-3）。
///
/// 调用方拿到的是 `Arc`，请共享而非复制；确需自有缓冲时自行 `to_vec()`。
pub fn decode_audio_cached_interleaved(path: &Path) -> Result<Arc<DecodedSourceAudio>, String> {
    if path.as_os_str().is_empty() {
        return Err("empty path".to_string());
    }

    // 指纹不可得（文件不存在/不可读）→ 不做缓存，交给非缓存路径报错，
    // 避免把失败结果也键成一个条目。
    let Some(fingerprint) = compute_file_fingerprint(path) else {
        let (sample_rate, channels, pcm) = decode_audio_f32_interleaved(path)?;
        return Ok(Arc::new(DecodedSourceAudio {
            sample_rate,
            channels,
            pcm: Arc::new(pcm),
        }));
    };

    let key: SourceCacheKey = (PathBuf::from(path), fingerprint);

    if let Some(hit) = source_cache()
        .lock()
        .ok()
        .and_then(|mut cache| cache.get(&key).cloned())
    {
        return Ok(hit);
    }

    let (sample_rate, channels, pcm) = decode_audio_f32_interleaved(path)?;
    let entry = Arc::new(DecodedSourceAudio {
        sample_rate,
        channels,
        pcm: Arc::new(pcm),
    });

    let weight = (entry.pcm.len() as u64)
        .saturating_mul(std::mem::size_of::<f32>() as u64)
        .saturating_add(128);

    if let Ok(mut cache) = source_cache().lock() {
        // 只缓存放得下的条目：单个文件超过预算时若仍插入，会把其它条目全部
        // 驱逐、自身随即又被淘汰，导致每次调用都清空一次缓存（负收益）。
        if weight <= cache.budget_bytes() {
            cache.insert(key, entry.clone(), weight);
        }
    }

    Ok(entry)
}

/// 运行时调整源 PCM 缓存预算（由 `cache_registry::apply_cache_budget` 调用）。
///
/// 缩容会立即按 LRU 回收，无需等待下一次插入（见 P1-7）。
pub fn set_source_cache_budget(bytes: u64) {
    if let Ok(mut cache) = source_cache().lock() {
        cache.set_budget(bytes);
    }
}

/// 丢弃某路径的所有缓存条目（路径被删除/替换后调用）。
///
/// 注意：正常的内容变更不需要调用此函数 —— 指纹已变化，旧条目不会再被命中，
/// 会随 LRU 自然淘汰。
#[allow(dead_code)]
pub fn invalidate_source_cache(path: &Path) {
    if let Ok(mut cache) = source_cache().lock() {
        cache.invalidate_where(|(p, _)| p.as_path() == path);
    }
}

/// 清空源文件解码缓存。
#[allow(dead_code)]
pub fn clear_source_cache() {
    if let Ok(mut cache) = source_cache().lock() {
        cache.clear();
    }
}

/// `(条目数, 占用字节)`，供诊断使用。
pub fn source_cache_stats() -> (usize, u64) {
    source_cache()
        .lock()
        .map(|c| (c.len(), c.total_bytes()))
        .unwrap_or((0, 0))
}

#[cfg(test)]
mod source_cache_tests {
    use super::*;
    use hound::{SampleFormat, WavSpec, WavWriter};

    fn write_wav(path: &Path, sample_rate: u32, frames: usize, amplitude: f32) {
        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(path, spec).expect("create wav");
        for i in 0..frames {
            let v = (amplitude * (i as f32 * 0.01).sin() * i16::MAX as f32) as i16;
            w.write_sample(v).expect("write sample");
        }
        w.finalize().expect("finalize wav");
    }

    fn temp_path(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let uniq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("hifishifter_srccache_{tag}_{uniq}.wav"));
        p
    }

    #[test]
    fn repeated_decode_returns_the_same_buffer() {
        let path = temp_path("repeat");
        write_wav(&path, 44_100, 2048, 0.5);

        let a = decode_audio_cached_interleaved(&path).expect("first decode");
        let b = decode_audio_cached_interleaved(&path).expect("second decode");

        assert_eq!(a.sample_rate, b.sample_rate);
        assert_eq!(a.channels, b.channels);
        assert!(
            Arc::ptr_eq(&a.pcm, &b.pcm),
            "second decode should reuse the cached buffer, not re-read the file"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn content_change_invalidates_the_entry() {
        let path = temp_path("content");
        write_wav(&path, 44_100, 2048, 0.5);

        let a = decode_audio_cached_interleaved(&path).expect("first decode");
        let a_len = a.pcm.len();

        // 重写为**不同长度**的内容：即使 mtime 分辨率不足，长度/内容指纹也不同。
        write_wav(&path, 44_100, 4096, 0.25);
        let b = decode_audio_cached_interleaved(&path).expect("decode after change");

        assert_ne!(a_len, b.pcm.len(), "should have re-decoded changed content");
        assert!(!Arc::ptr_eq(&a.pcm, &b.pcm));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_file_is_an_error_and_caches_nothing() {
        let path = temp_path("missing");
        assert!(decode_audio_cached_interleaved(&path).is_err());
    }

    #[test]
    fn empty_path_is_rejected() {
        assert!(decode_audio_cached_interleaved(Path::new("")).is_err());
    }

    #[test]
    fn invalidation_drops_the_entry() {
        let path = temp_path("invalidate");
        write_wav(&path, 44_100, 1024, 0.4);

        let a = decode_audio_cached_interleaved(&path).expect("first decode");
        invalidate_source_cache(&path);
        let b = decode_audio_cached_interleaved(&path).expect("decode after invalidate");

        assert!(!Arc::ptr_eq(&a.pcm, &b.pcm), "invalidate should force a re-read");

        let _ = std::fs::remove_file(&path);
    }
}
