use std::path::Path;
use std::sync::{Arc, Mutex};

use super::resource_manager::DecodeCache;
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
    crate::media::is_media_extension(path)
}

/// 解码任意媒体文件为交错 f32 PCM（声道数以 `usize` 返回，音频引擎内部的习惯）。
///
/// 【为什么这里只剩一次转发】WAV 的 hound 快路径曾经在这里也有一份。多份转换函数
/// 的代价是"只修其中一份"——音高分析与波形镜像走的不是这条路，于是它们对着同一个
/// 文件整条失败。容错现在统一在 `media` 里（见那里关于 RIFF 长度的说明）。
pub(crate) fn decode_audio_f32_interleaved(path: &Path) -> Result<(u32, usize, Vec<f32>), String> {
    crate::media::decode_media_audio_f32_interleaved(path, None)
        .map(|(sample_rate, channels, pcm)| (sample_rate, channels as usize, pcm))
}

pub(crate) fn decode_resampled_stereo(path: &Path, out_rate: u32) -> Option<ResampledStereo> {
    if !path.exists() {
        return None;
    }

    let (in_rate, in_channels, pcm) = match decode_audio_f32_interleaved(path) {
        Ok(v) => v,
        Err(e) => {
            if hifishifter_kernel::log_policy::probes_enabled() {
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

    // 同采样率时**移动**刚解码的缓冲，避免 `linear_resample_interleaved` 的
    // `input.to_vec()` 整文件再拷一份 —— 长音频（1 小时立体声 ≈ 1.27GB）在解码
    // 阶段的瞬时内存会因此翻倍。跨采样率仍需输入 + 输出两份，属重采样的固有代价。
    let resampled = if in_rate == out_rate {
        pcm
    } else {
        linear_resample_interleaved(&pcm, in_channels, in_rate, out_rate)
    };

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
