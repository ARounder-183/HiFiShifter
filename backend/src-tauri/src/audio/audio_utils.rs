use std::path::Path;

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

    decode_audio_f32_interleaved_symphonia(path)
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

fn decode_audio_f32_interleaved_symphonia(path: &Path) -> Result<(u32, u16, Vec<f32>), String> {
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| e.to_string())?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| "no default track".to_string())?;

    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1)
        .max(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| e.to_string())?;

    // 预分配内存，避免 Vec 动态扩容
    let estimated_frames = track.codec_params.n_frames.unwrap_or(0) as usize;
    let mut out: Vec<f32> = Vec::with_capacity(estimated_frames * channels);
    let mut sbuf_opt: Option<symphonia::core::audio::SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(_)) => break,
            Err(e) => return Err(e.to_string()),
        };
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(Error::DecodeError(_)) => continue,
            Err(Error::IoError(_)) => break,
            Err(e) => return Err(e.to_string()),
        };

        if sample_rate == 0 {
            sample_rate = decoded.spec().rate;
        }

        // 复用缓冲区，正确比较总样本数 (Samples = Frames * Channels)
        let required_samples = decoded.capacity() * decoded.spec().channels.count();
        if sbuf_opt.is_none() || sbuf_opt.as_ref().unwrap().capacity() < required_samples {
            sbuf_opt = Some(symphonia::core::audio::SampleBuffer::<f32>::new(
                decoded.capacity() as u64,
                *decoded.spec(),
            ));
        }
        let sbuf = sbuf_opt.as_mut().unwrap();
        sbuf.copy_interleaved_ref(decoded);
        out.extend_from_slice(sbuf.samples());
    }
    Ok((
        if sample_rate == 0 { 44100 } else { sample_rate },
        channels as u16,
        out,
    ))
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

    // Fall back to Symphonia for non-WAV (or WAV variants hound can't decode).
    try_read_audio_info_symphonia(path, preview_points)
}

/// 快速只读 sample_rate / total_frames / duration_sec，不生成 waveform_preview。
///
/// - WAV：hound 单次文件打开，读 header 后直接返回，无样本扫描。
/// - 非 WAV：优先从 codec params 的 n_frames 字段获取帧数（O(1)，无需解码），
///           若容器未提供则回退到 symphonia 全量计帧（跳过 preview 生成）。
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
    try_read_duration_symphonia(path)
}

fn try_read_duration_symphonia(path: &Path) -> Option<WavInfo> {
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).ok()?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .ok()?;
    let format = probed.format;
    let track = format.default_track()?;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);

    if let Some(n_frames) = track.codec_params.n_frames {
        // 容器直接提供帧数，O(1)，无需任何解码。
        return Some(WavInfo {
            sample_rate,
            total_frames: n_frames,
            duration_sec: n_frames as f64 / sample_rate as f64,
            waveform_preview: vec![],
        });
    }

    // n_frames 不可用（如 CBR MP3）：回退到解码计帧，但跳过 preview 生成。
    try_read_audio_info_symphonia(path, 0)
}

pub fn compute_minmax_peaks(
    path: &Path,
    hop: usize,
) -> Result<crate::waveform::CachedPeaks, String> {
    if hop == 0 {
        return Err("hop must be > 0".to_string());
    }

    let is_wav = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("wav"))
        .unwrap_or(false);

    if is_wav {
        if let Ok(peaks) = compute_minmax_peaks_hound(path, hop) {
            return Ok(peaks);
        }
    }
    compute_minmax_peaks_symphonia(path, hop)
}

fn compute_minmax_peaks_hound(
    path: &Path,
    hop: usize,
) -> Result<crate::waveform::CachedPeaks, String> {
    use hound::{SampleFormat, WavReader};

    let mut reader = WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    if spec.sample_rate == 0 || spec.channels == 0 {
        return Err("invalid wav spec".to_string());
    }

    let channels = spec.channels as usize;
    // hound::duration() 返回每声道的帧数（frames），直接就是 total_frames
    let total_frames = reader.duration() as u64;

    let mut min = Vec::<f32>::new();
    let mut max = Vec::<f32>::new();
    let mut acc_min = f32::INFINITY;
    let mut acc_max = f32::NEG_INFINITY;
    let mut frame_count = 0usize;

    // 宏：将一帧的各声道极值 (ch_min, ch_max) 合入累积器，到达 hop 时输出一个 peak
    macro_rules! flush_frame {
        ($ch_min:expr, $ch_max:expr) => {{
            let cmin = $ch_min;
            let cmax = $ch_max;
            if cmin < acc_min {
                acc_min = cmin;
            }
            if cmax > acc_max {
                acc_max = cmax;
            }
            frame_count += 1;
            if frame_count >= hop {
                min.push(if acc_min.is_finite() { acc_min } else { 0.0 });
                max.push(if acc_max.is_finite() { acc_max } else { 0.0 });
                acc_min = f32::INFINITY;
                acc_max = f32::NEG_INFINITY;
                frame_count = 0;
            }
        }};
    }

    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => {
            let scale = 1.0 / (i16::MAX as f32);
            let mut buf = vec![0i16; channels];
            let mut i = 0usize;
            for s in reader.samples::<i16>() {
                buf[i] = s.map_err(|e| e.to_string())?;
                i += 1;
                if i >= channels {
                    i = 0;
                    // 取各声道极值而非求平均，避免双声道相位抵消导致波形变瘦
                    let mut ch_min = f32::INFINITY;
                    let mut ch_max = f32::NEG_INFINITY;
                    for &x in &buf {
                        let v = x as f32 * scale;
                        if v < ch_min {
                            ch_min = v;
                        }
                        if v > ch_max {
                            ch_max = v;
                        }
                    }
                    flush_frame!(ch_min, ch_max);
                }
            }
        }
        (SampleFormat::Int, 24) => {
            let scale = 1.0 / ((1u32 << 23) as f32);
            let mut buf = vec![0i32; channels];
            let mut i = 0usize;
            for s in reader.samples::<i32>() {
                buf[i] = s.map_err(|e| e.to_string())?;
                i += 1;
                if i >= channels {
                    i = 0;
                    let mut ch_min = f32::INFINITY;
                    let mut ch_max = f32::NEG_INFINITY;
                    for &x in &buf {
                        let v = x as f32 * scale;
                        if v < ch_min {
                            ch_min = v;
                        }
                        if v > ch_max {
                            ch_max = v;
                        }
                    }
                    flush_frame!(ch_min, ch_max);
                }
            }
        }
        (SampleFormat::Int, 32) => {
            let scale = 1.0 / (i32::MAX as f32);
            let mut buf = vec![0i32; channels];
            let mut i = 0usize;
            for s in reader.samples::<i32>() {
                buf[i] = s.map_err(|e| e.to_string())?;
                i += 1;
                if i >= channels {
                    i = 0;
                    let mut ch_min = f32::INFINITY;
                    let mut ch_max = f32::NEG_INFINITY;
                    for &x in &buf {
                        let v = x as f32 * scale;
                        if v < ch_min {
                            ch_min = v;
                        }
                        if v > ch_max {
                            ch_max = v;
                        }
                    }
                    flush_frame!(ch_min, ch_max);
                }
            }
        }
        (SampleFormat::Float, 32) => {
            let mut buf = vec![0f32; channels];
            let mut i = 0usize;
            for s in reader.samples::<f32>() {
                buf[i] = s.map_err(|e| e.to_string())?;
                i += 1;
                if i >= channels {
                    i = 0;
                    let mut ch_min = f32::INFINITY;
                    let mut ch_max = f32::NEG_INFINITY;
                    for &x in &buf {
                        if x < ch_min {
                            ch_min = x;
                        }
                        if x > ch_max {
                            ch_max = x;
                        }
                    }
                    flush_frame!(ch_min, ch_max);
                }
            }
        }
        _ => return Err("unsupported wav format".to_string()),
    }

    if frame_count > 0 {
        min.push(if acc_min.is_finite() { acc_min } else { 0.0 });
        max.push(if acc_max.is_finite() { acc_max } else { 0.0 });
    }

    Ok(crate::waveform::CachedPeaks {
        sample_rate: spec.sample_rate,
        hop,
        min,
        max,
        total_frames,
    })
}

fn compute_minmax_peaks_symphonia(
    path: &Path,
    hop: usize,
) -> Result<crate::waveform::CachedPeaks, String> {
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| e.to_string())?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| "no default track".to_string())?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| e.to_string())?;

    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1)
        .max(1);

    let mut min = Vec::<f32>::new();
    let mut max = Vec::<f32>::new();
    let mut acc_min = f32::INFINITY;
    let mut acc_max = f32::NEG_INFINITY;
    let mut frame_count = 0usize;
    let mut total_frames: u64 = 0;
    let mut sbuf_opt: Option<symphonia::core::audio::SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(_)) => break,
            Err(e) => return Err(e.to_string()),
        };
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(Error::DecodeError(_)) => continue,
            Err(Error::IoError(_)) => break,
            Err(e) => return Err(e.to_string()),
        };

        // 复用缓冲区，正确比较总样本数 (Samples = Frames * Channels)
        let required_samples = decoded.capacity() * decoded.spec().channels.count();
        if sbuf_opt.is_none() || sbuf_opt.as_ref().unwrap().capacity() < required_samples {
            sbuf_opt = Some(symphonia::core::audio::SampleBuffer::<f32>::new(
                decoded.capacity() as u64,
                *decoded.spec(),
            ));
        }
        let sbuf = sbuf_opt.as_mut().unwrap();
        sbuf.copy_interleaved_ref(decoded);
        let samples = sbuf.samples();

        let frames = samples.len() / channels;
        total_frames = total_frames.saturating_add(frames as u64);

        for f in 0..frames {
            let base = f * channels;
            // 取各声道极值而非求平均，避免双声道相位抵消导致波形变瘦
            let mut ch_min = f32::INFINITY;
            let mut ch_max = f32::NEG_INFINITY;
            for ch in 0..channels {
                let v = samples.get(base + ch).copied().unwrap_or(0.0);
                if v < ch_min {
                    ch_min = v;
                }
                if v > ch_max {
                    ch_max = v;
                }
            }

            if ch_min < acc_min {
                acc_min = ch_min;
            }
            if ch_max > acc_max {
                acc_max = ch_max;
            }
            frame_count += 1;
            if frame_count >= hop {
                min.push(if acc_min.is_finite() { acc_min } else { 0.0 });
                max.push(if acc_max.is_finite() { acc_max } else { 0.0 });
                acc_min = f32::INFINITY;
                acc_max = f32::NEG_INFINITY;
                frame_count = 0;
            }
        }
    }

    if frame_count > 0 {
        min.push(if acc_min.is_finite() { acc_min } else { 0.0 });
        max.push(if acc_max.is_finite() { acc_max } else { 0.0 });
    }

    Ok(crate::waveform::CachedPeaks {
        sample_rate,
        hop,
        min,
        max,
        total_frames,
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

fn try_read_audio_info_symphonia(path: &Path, preview_points: usize) -> Option<WavInfo> {
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    type SymphoniaOpen = (
        Box<dyn symphonia::core::formats::FormatReader>,
        Box<dyn symphonia::core::codecs::Decoder>,
        u32,
        usize,
    );

    fn open(path: &Path) -> Option<SymphoniaOpen> {
        let file = std::fs::File::open(path).ok()?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        let probed = symphonia::default::get_probe()
            .format(
                &hint,
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .ok()?;
        let format = probed.format;
        let track = format.default_track()?;
        let decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .ok()?;

        let sr = track
            .codec_params
            .sample_rate
            .or(track.codec_params.sample_rate)
            .unwrap_or(44100);
        let ch = track
            .codec_params
            .channels
            .map(|c| c.count())
            .unwrap_or(1)
            .max(1);

        Some((format, decoder, sr, ch))
    }

    let (mut format1, _decoder1, sample_rate, channels) = open(path)?;

    // Pass 1: count total frames
    let mut total_frames: u64 = 0;
    loop {
        let packet = match format1.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(_)) => break,
            Err(_) => return None,
        };
        total_frames = total_frames.saturating_add(packet.dur);
    }

    let duration_sec = if sample_rate > 0 {
        total_frames as f64 / sample_rate as f64
    } else {
        0.0
    };

    if preview_points == 0 || total_frames == 0 {
        return Some(WavInfo {
            sample_rate,
            total_frames,
            duration_sec,
            waveform_preview: vec![],
        });
    }

    // Pass 2: build preview.
    let (mut format2, mut decoder2, _sr2, _ch2) = open(path)?;
    let preview_len = preview_points.max(2);
    let mut preview = vec![0.0f32; preview_len];
    let step_frames = (total_frames / preview_len as u64).max(1);

    let mut idx = 0usize;
    let mut current_max = 0.0f32;
    let mut count_frames: u64 = 0;
    let mut sbuf_opt: Option<symphonia::core::audio::SampleBuffer<f32>> = None;

    loop {
        let packet = match format2.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(_)) => break,
            Err(_) => break,
        };
        let decoded = match decoder2.decode(&packet) {
            Ok(d) => d,
            Err(Error::DecodeError(_)) => continue,
            Err(Error::IoError(_)) => break,
            Err(_) => break,
        };

        // 复用缓冲区，正确比较总样本数 (Samples = Frames * Channels)
        let required_samples = decoded.capacity() * decoded.spec().channels.count();
        if sbuf_opt.is_none() || sbuf_opt.as_ref().unwrap().capacity() < required_samples {
            sbuf_opt = Some(symphonia::core::audio::SampleBuffer::<f32>::new(
                decoded.capacity() as u64,
                *decoded.spec(),
            ));
        }
        let sbuf = sbuf_opt.as_mut().unwrap();
        sbuf.copy_interleaved_ref(decoded);
        let samples = sbuf.samples();

        let frames = samples.len() / channels;
        for f in 0..frames {
            let mut frame_max = 0.0f32;
            let base = f * channels;
            for ch in 0..channels {
                let a = samples.get(base + ch).copied().unwrap_or(0.0).abs();
                if a > frame_max {
                    frame_max = a;
                }
            }
            if frame_max > current_max {
                current_max = frame_max;
            }
            count_frames += 1;
            if count_frames >= step_frames {
                preview[idx] = current_max;
                idx += 1;
                if idx >= preview_len {
                    break;
                }
                current_max = 0.0;
                count_frames = 0;
            }
        }

        if idx >= preview_len {
            break;
        }
    }

    Some(WavInfo {
        sample_rate,
        total_frames,
        duration_sec,
        waveform_preview: preview,
    })
}
