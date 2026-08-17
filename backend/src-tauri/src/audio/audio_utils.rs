use std::path::Path;

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
    // hound::duration() 返回每声道的帧数（frames），总样本数 = frames * channels
    let mut out: Vec<f32> = Vec::with_capacity(reader.duration() as usize * channels as usize);

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
