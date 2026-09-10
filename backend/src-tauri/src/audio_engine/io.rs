use std::path::Path;
use std::sync::{Arc, Mutex};

use super::resource_manager::DecodeCache;
use super::types::ResampledStereo;

// 采样率转换统一走 `crate::resample`（带限 / 抗混叠）。
// 这里保留一个别名，避免调用点写成两套名字。
pub(crate) use crate::resample::resample_interleaved as linear_resample_interleaved;

pub(crate) fn is_audio_path(path: &Path) -> bool {
    crate::media::is_media_extension(path)
}

fn read_wav_f32_interleaved(path: &Path) -> Option<(u32, u16, Vec<f32>)> {
    use hound::{SampleFormat, WavReader};

    let mut reader = WavReader::open(path).ok()?;
    let spec = reader.spec();
    if spec.sample_rate == 0 || spec.channels == 0 {
        return None;
    }

    let channels = spec.channels;
    let sample_rate = spec.sample_rate;

    // duration() 来自 WAV 头字段，损坏/恶意文件可声称数十 GB —— 直接
    // with_capacity 会一次性分配失败 abort 进程。按文件大小钳制上界
    // （交错样本数 ≤ 文件字节 / 最少每样本字节）；metadata 不可得时用
    // 64M 样本的兜底上界。
    let min_bytes_per_sample = (spec.bits_per_sample as usize / 8).max(1);
    let file_bound = std::fs::metadata(path)
        .map(|m| (m.len() as usize).saturating_div(min_bytes_per_sample))
        .unwrap_or(1 << 22);
    let declared = reader.duration() as usize;
    let mut out: Vec<f32> = Vec::with_capacity(declared.min(file_bound));

    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => {
            for s in reader.samples::<i16>() {
                let v = s.ok()? as f32 / i16::MAX as f32;
                out.push(v);
            }
        }
        (SampleFormat::Int, 24) => {
            let denom = (1u32 << 23) as f32;
            for s in reader.samples::<i32>() {
                let v = s.ok()? as f32 / denom;
                out.push(v);
            }
        }
        (SampleFormat::Int, 32) => {
            for s in reader.samples::<i32>() {
                let v = s.ok()? as f32 / i32::MAX as f32;
                out.push(v);
            }
        }
        (SampleFormat::Float, 32) => {
            for s in reader.samples::<f32>() {
                out.push(s.ok()?);
            }
        }
        _ => return None,
    }

    Some((sample_rate, channels, out))
}

pub(crate) fn decode_audio_f32_interleaved(path: &Path) -> Result<(u32, usize, Vec<f32>), String> {
    // Fast-path WAV via hound.
    if path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("wav"))
        .unwrap_or(false)
    {
        if let Some((sr, ch, pcm)) = read_wav_f32_interleaved(path) {
            return Ok((sr, ch as usize, pcm));
        }
    }

    crate::media::decode_media_audio_f32_interleaved(path, None)
        .map(|(sr, ch, pcm)| (sr, ch as usize, pcm))
}

pub(crate) fn decode_resampled_stereo(path: &Path, out_rate: u32) -> Option<ResampledStereo> {
    if !path.exists() {
        return None;
    }

    let (in_rate, in_channels, pcm) = match decode_audio_f32_interleaved(path) {
        Ok(v) => v,
        Err(e) => {
            if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
                log::error!(
                    "AudioEngine: decode failed: path={} err={} ",
                    path.display(),
                    e
                );
            }
            return None;
        }
    };
    let in_channels = in_channels.max(1);

    let resampled = linear_resample_interleaved(&pcm, in_channels, in_rate, out_rate);

    let stereo: Vec<f32> = if in_channels == 1 {
        let mut out = Vec::with_capacity(resampled.len() * 2);
        for s in &resampled {
            out.push(*s);
            out.push(*s);
        }
        out
    } else if in_channels == 2 {
        resampled
    } else {
        let frames = resampled.len() / in_channels;
        let mut out = Vec::with_capacity(frames * 2);
        for f in 0..frames {
            out.push(resampled[f * in_channels]);
            out.push(resampled[f * in_channels + 1]);
        }
        out
    };

    let frames = stereo.len() / 2;
    Some(ResampledStereo {
        sample_rate: out_rate,
        frames,
        pcm: Arc::new(stereo),
    })
}

pub(crate) fn get_resampled_stereo_cached(
    path: &Path,
    out_rate: u32,
    cache: &Arc<Mutex<DecodeCache>>,
) -> Option<ResampledStereo> {
    if !path.exists() {
        return None;
    }
    let key = (path.to_path_buf(), out_rate);
    if let Ok(mut map) = cache.lock() {
        if let Some(v) = map.get(&key) {
            return Some(v.clone());
        }
    }
    None
}

#[allow(dead_code)]
pub(crate) fn get_resampled_stereo(
    path: &Path,
    out_rate: u32,
    cache: &Arc<Mutex<DecodeCache>>,
) -> Option<ResampledStereo> {
    if !path.exists() {
        return None;
    }

    let key = (path.to_path_buf(), out_rate);
    if let Ok(mut map) = cache.lock() {
        if let Some(v) = map.get(&key) {
            return Some(v.clone());
        }
    }

    let v = decode_resampled_stereo(path, out_rate)?;

    if let Ok(mut map) = cache.lock() {
        let pcm_bytes = v.pcm_bytes();
        map.insert(key, v.clone(), pcm_bytes);
    }

    Some(v)
}
