use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::types::ResampledStereo;

pub(crate) fn linear_resample_interleaved(
    input: &[f32],
    channels: usize,
    in_rate: u32,
    out_rate: u32,
) -> Vec<f32> {
    if input.is_empty() || channels == 0 {
        return vec![];
    }
    if in_rate == out_rate {
        return input.to_vec();
    }

    let in_frames = input.len() / channels;
    if in_frames < 2 {
        return input.to_vec();
    }

    let ratio = out_rate as f64 / in_rate as f64;
    let out_frames = ((in_frames as f64) * ratio).round().max(1.0) as usize;
    let mut out = vec![0.0f32; out_frames * channels];

    for of in 0..out_frames {
        let t_in = (of as f64) / ratio;
        let i0 = t_in.floor() as isize;
        let frac = (t_in - (i0 as f64)) as f32;
        let i0 = i0.clamp(0, (in_frames - 1) as isize) as usize;
        let i1 = (i0 + 1).min(in_frames - 1);

        for ch in 0..channels {
            let a = input[i0 * channels + ch];
            let b = input[i1 * channels + ch];
            out[of * channels + ch] = a + (b - a) * frac;
        }
    }

    out
}

pub(crate) fn is_audio_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            ["wav", "mp3", "flac", "ogg", "m4a", "aac"]
                .iter()
                .any(|&ext| e.eq_ignore_ascii_case(ext))
        })
        .unwrap_or(false)
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

    let mut out: Vec<f32> = Vec::with_capacity(reader.duration() as usize);

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

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| "missing sample_rate in codec params".to_string())?;

    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

    let mut out: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(_)) => break,
            Err(Error::ResetRequired) => return Err("decoder reset required".to_string()),
            Err(e) => return Err(e.to_string()),
        };

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(Error::IoError(_)) => break,
            Err(Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.to_string()),
        };

        // 统一使用 Symphonia 高度优化的 SampleBuffer，消除手写的 Option 边界检查开销
        let spec = *decoded.spec();
        let duration = decoded.capacity() as u64;
        let mut sbuf = symphonia::core::audio::SampleBuffer::<f32>::new(duration, spec);
        sbuf.copy_interleaved_ref(decoded);
        out.extend_from_slice(sbuf.samples());
    }

    Ok((sample_rate, channels, out))
}

pub(crate) fn decode_resampled_stereo(path: &Path, out_rate: u32) -> Option<ResampledStereo> {
    if !path.exists() {
        return None;
    }

    let (in_rate, in_channels, pcm) = match decode_audio_f32_interleaved(path) {
        Ok(v) => v,
        Err(e) => {
            if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
                eprintln!(
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
    cache: &Arc<Mutex<HashMap<(PathBuf, u32), ResampledStereo>>>,
) -> Option<ResampledStereo> {
    if !path.exists() {
        return None;
    }
    let key = (path.to_path_buf(), out_rate);
    if let Ok(map) = cache.lock() {
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
    cache: &Arc<Mutex<HashMap<(PathBuf, u32), ResampledStereo>>>,
) -> Option<ResampledStereo> {
    if !path.exists() {
        return None;
    }

    let key = (path.to_path_buf(), out_rate);
    if let Ok(map) = cache.lock() {
        if let Some(v) = map.get(&key) {
            return Some(v.clone());
        }
    }

    let v = decode_resampled_stereo(path, out_rate)?;

    if let Ok(mut map) = cache.lock() {
        map.insert(key, v.clone());
    }

    Some(v)
}
