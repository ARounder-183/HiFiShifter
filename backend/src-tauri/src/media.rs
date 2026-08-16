//! Media-file (audio + video container) support via dynamically-linked FFmpeg.
//!
//! The project is MIT licensed and therefore links FFmpeg **dynamically**
//! (LGPL shared libraries). No GPL/non-free components are enabled, and the
//! `ffmpeg-sys-next` `static` / `build` features must never be enabled.

use std::path::Path;
use std::sync::Once;

use ffmpeg_next as ffmpeg;
use ffmpeg::format::sample::Sample;
use ffmpeg::media;

pub const AUDIO_EXTENSIONS: &[&str] = &[
    "wav", "mp3", "flac", "ogg", "oga", "opus", "aac", "m4a", "aif", "aiff", "wma", "ac3",
    "eac3", "ape", "wv", "mp2", "mpa", "dts", "amr",
];

pub const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "m4v", "mov", "mkv", "webm", "avi", "flv", "wmv", "ts", "mts", "m2ts", "vob", "mpg",
    "mpeg", "3gp", "3g2", "ogv", "rm", "rmvb",
];

fn ext_equals(path: &Path, ext: &str) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case(ext))
        .unwrap_or(false)
}

pub fn is_audio_extension(path: &Path) -> bool {
    AUDIO_EXTENSIONS.iter().any(|e| ext_equals(path, e))
}

pub fn is_video_extension(path: &Path) -> bool {
    VIDEO_EXTENSIONS.iter().any(|e| ext_equals(path, e))
}

pub fn is_media_extension(path: &Path) -> bool {
    is_audio_extension(path) || is_video_extension(path)
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MediaAudioStream {
    pub index: usize,
    pub title: Option<String>,
    pub language: Option<String>,
    pub codec: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_sec: f64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // video metadata is available for future UI/debug surfaces
pub struct MediaProbe {
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_sec: f64,
    pub total_frames: u64,
    pub waveform_preview: Vec<f32>,
    pub has_video_stream: bool,
    pub container_format: String,
    pub audio_stream_index: usize,
    pub audio_stream_count: usize,
}

fn ensure_ffmpeg_init() -> Result<(), String> {
    static INIT: Once = Once::new();
    let mut result = Ok(());
    INIT.call_once(|| {
        if let Err(e) = ffmpeg::init() {
            result = Err(format!("ffmpeg init failed: {e}"));
        }
    });
    result
}

fn ffmpeg_err(context: &str, e: ffmpeg::Error) -> String {
    format!("{context}: {e}")
}

fn open_input(path: &Path) -> Result<ffmpeg::format::context::Input, String> {
    ensure_ffmpeg_init()?;
    ffmpeg::format::input(path).map_err(|e| ffmpeg_err("ffmpeg open failed", e))
}

fn audio_stream_index(
    ictx: &ffmpeg::format::context::Input,
    preferred: Option<usize>,
) -> Result<usize, String> {
    if let Some(index) = preferred {
        if ictx
            .streams()
            .any(|s| s.index() == index && s.parameters().medium() == media::Type::Audio)
        {
            return Ok(index);
        }
        return Err(format!("ffmpeg: audio stream {index} not found"));
    }

    ictx.streams()
        .best(media::Type::Audio)
        .map(|s| s.index())
        .ok_or_else(|| "ffmpeg: no audio stream".to_string())
}

fn duration_for_stream(ictx: &ffmpeg::format::context::Input, index: usize) -> f64 {
    let Some(stream) = ictx.streams().nth(index) else {
        return 0.0;
    };

    let tb = f64::from(stream.time_base());
    if stream.duration() > 0 && tb > 0.0 {
        stream.duration() as f64 * tb
    } else if ictx.duration() > 0 {
        // AV_TIME_BASE (1/1_000_000 s)
        ictx.duration() as f64 / 1_000_000.0
    } else {
        0.0
    }
}

pub fn list_audio_streams(path: &Path) -> Result<Vec<MediaAudioStream>, String> {
    let ictx = open_input(path)?;
    let mut out = Vec::new();

    for stream in ictx.streams() {
        if stream.parameters().medium() != media::Type::Audio {
            continue;
        }

        let index = stream.index();
        let params = stream.parameters();
        let decoder = ffmpeg::codec::context::Context::from_parameters(params.clone())
            .and_then(|ctx| ctx.decoder().audio())
            .ok();

        let codec = decoder
            .as_ref()
            .and_then(|d| d.codec())
            .map(|c| c.name().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let mut title = None;
        let mut language = None;
        for (key, value) in stream.metadata().iter() {
            if key.eq_ignore_ascii_case("title") {
                title = Some(value.to_string());
            } else if key.eq_ignore_ascii_case("language") {
                language = Some(value.to_string());
            }
        }

        out.push(MediaAudioStream {
            index,
            title,
            language,
            codec,
            sample_rate: decoder.as_ref().map(|d| d.rate()).unwrap_or(0),
            channels: decoder
                .as_ref()
                .map(|d| d.channel_layout().channels() as u16)
                .unwrap_or(0),
            duration_sec: duration_for_stream(&ictx, index),
        });
    }

    Ok(out)
}

/// Probe the first (or requested) audio stream of a media file.
///
/// When `preview_points > 0`, the complete audio stream is decoded once and a
/// downsampled min/max preview is produced (same behaviour as the WAV and
/// Symphonia paths). For header-only calls pass `preview_points = 0`.
pub fn probe_media(
    path: &Path,
    preview_points: usize,
    preferred_stream: Option<usize>,
) -> Option<MediaProbe> {
    let mut ictx = open_input(path).ok()?;
    let audio_index = audio_stream_index(&ictx, preferred_stream).ok()?;

    let container_format = ictx.format().name().to_string();
    let has_video_stream = ictx
        .streams()
        .any(|s| s.parameters().medium() == media::Type::Video);
    let audio_stream_count = ictx
        .streams()
        .filter(|s| s.parameters().medium() == media::Type::Audio)
        .count();

    let stream = ictx.streams().nth(audio_index)?;
    let params = stream.parameters();
    let context = ffmpeg::codec::context::Context::from_parameters(params).ok()?;
    let mut decoder = context.decoder().audio().ok()?;
    let sample_rate = decoder.rate().max(1);
    let channels = decoder.channel_layout().channels().max(1) as u16;
    let duration_sec = duration_for_stream(&ictx, audio_index);

    // AVStream.nb_frames is container-defined and for audio is often the
    // number of *packets* (e.g. ~19 AAC packets for a 0.4 s MP4), not the
    // number of PCM sample frames. Trusting it would create clips with a
    // near-zero length. Derive PCM frames from the stream duration instead.
    let total_frames = if sample_rate > 0 && duration_sec.is_finite() && duration_sec > 0.0 {
        (duration_sec * sample_rate as f64).round().max(0.0) as u64
    } else {
        0
    };

    let waveform_preview = if preview_points > 0 {
        compute_preview(
            &mut ictx,
            &mut decoder,
            audio_index,
            sample_rate,
            channels,
            total_frames,
            duration_sec,
            preview_points,
        )
    } else {
        Vec::new()
    };

    Some(MediaProbe {
        sample_rate,
        channels,
        duration_sec,
        total_frames,
        waveform_preview,
        has_video_stream,
        container_format,
        audio_stream_index: audio_index,
        audio_stream_count,
    })
}

fn compute_preview(
    ictx: &mut ffmpeg::format::context::Input,
    decoder: &mut ffmpeg::decoder::Audio,
    audio_index: usize,
    sample_rate: u32,
    channels: u16,
    total_frames: u64,
    duration_sec: f64,
    preview_points: usize,
) -> Vec<f32> {
    let points = preview_points.max(2);
    let estimated_frames = if total_frames > 0 {
        total_frames as usize
    } else if duration_sec > 0.0 {
        (duration_sec * sample_rate as f64).max(1.0) as usize
    } else {
        0
    };
    let estimated_samples = estimated_frames.saturating_mul(channels.max(1) as usize);

    let mut min_bucket = vec![f32::INFINITY; points];
    let mut max_bucket = vec![f32::NEG_INFINITY; points];
    let mut seen_samples = 0usize;

    let _ = visit_audio_frames(
        ictx,
        decoder,
        audio_index,
        &mut |frame: &[f32], _rate: u32, ch: u16| {
            let ch = ch.max(1) as usize;
            for &sample in frame.iter().take(frame.len() / ch * ch) {
                let idx = if estimated_samples > 0 {
                    (((seen_samples as u128) * (points as u128)) / (estimated_samples as u128))
                        as usize
                } else {
                    0
                }
                .min(points - 1);
                min_bucket[idx] = min_bucket[idx].min(sample);
                max_bucket[idx] = max_bucket[idx].max(sample);
                seen_samples = seen_samples.saturating_add(1);
            }
            Ok(())
        },
    );

    let mut preview = Vec::with_capacity(points);
    for i in 0..points {
        let min = min_bucket[i];
        let max = max_bucket[i];
        let value = if min.is_finite() && max.is_finite() {
            if max.abs() >= min.abs() { max } else { min }
        } else if min.is_finite() {
            min
        } else if max.is_finite() {
            max
        } else {
            0.0
        };
        preview.push(value);
    }
    preview
}

/// Decode the selected audio stream to interleaved f32 PCM.
pub fn decode_media_audio_f32_interleaved(
    path: &Path,
    preferred_stream: Option<usize>,
) -> Result<(u32, u16, Vec<f32>), String> {
    let mut ictx = open_input(path)?;
    let audio_index = audio_stream_index(&ictx, preferred_stream)?;
    let stream = ictx
        .streams()
        .nth(audio_index)
        .ok_or_else(|| "ffmpeg: audio stream disappeared".to_string())?;
    let params = stream.parameters();
    let context = ffmpeg::codec::context::Context::from_parameters(params)
        .map_err(|e| ffmpeg_err("ffmpeg codec context failed", e))?;
    let mut decoder = context
        .decoder()
        .audio()
        .map_err(|e| ffmpeg_err("ffmpeg audio decoder failed", e))?;

    let sample_rate = decoder.rate().max(1);
    let channels = decoder.channel_layout().channels().max(1) as u16;
    let mut out = Vec::new();

    visit_audio_frames(&mut ictx, &mut decoder, audio_index, &mut |frame, _, _| {
        out.extend_from_slice(frame);
        Ok(())
    })?;

    Ok((sample_rate, channels, out))
}

/// Iterate decoded audio frames without accumulating the whole file.
pub fn visit_media_audio_frames<F>(
    path: &Path,
    preferred_stream: Option<usize>,
    mut on_frame: F,
) -> Result<(u32, u16), String>
where
    F: FnMut(&[f32], u32, u16) -> Result<(), String>,
{
    let mut ictx = open_input(path)?;
    let audio_index = audio_stream_index(&ictx, preferred_stream)?;
    let stream = ictx
        .streams()
        .nth(audio_index)
        .ok_or_else(|| "ffmpeg: audio stream disappeared".to_string())?;
    let params = stream.parameters();
    let context = ffmpeg::codec::context::Context::from_parameters(params)
        .map_err(|e| ffmpeg_err("ffmpeg codec context failed", e))?;
    let mut decoder = context
        .decoder()
        .audio()
        .map_err(|e| ffmpeg_err("ffmpeg audio decoder failed", e))?;
    let sample_rate = decoder.rate().max(1);
    let channels = decoder.channel_layout().channels().max(1) as u16;

    visit_audio_frames(&mut ictx, &mut decoder, audio_index, &mut on_frame)?;
    Ok((sample_rate, channels))
}

fn visit_audio_frames<F>(
    ictx: &mut ffmpeg::format::context::Input,
    decoder: &mut ffmpeg::decoder::Audio,
    audio_index: usize,
    on_frame: &mut F,
) -> Result<(), String>
where
    F: FnMut(&[f32], u32, u16) -> Result<(), String>,
{
    let _ = visit_audio_frames_until(ictx, decoder, audio_index, usize::MAX, on_frame)?;
    Ok(())
}

fn visit_audio_frames_until<F>(
    ictx: &mut ffmpeg::format::context::Input,
    decoder: &mut ffmpeg::decoder::Audio,
    audio_index: usize,
    max_frames: usize,
    on_frame: &mut F,
) -> Result<bool, String>
where
    F: FnMut(&[f32], u32, u16) -> Result<(), String>,
{
    let sample_rate = decoder.rate().max(1);
    let channels = decoder.channel_layout().channels().max(1) as u16;
    let mut emitted_frames = 0usize;

    for (stream, packet) in ictx.packets() {
        if stream.index() != audio_index {
            continue;
        }

        match decoder.send_packet(&packet) {
            Ok(()) => {}
            Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => continue,
            Err(ffmpeg::Error::Eof) => break,
            Err(e) => return Err(ffmpeg_err("ffmpeg send_packet failed", e)),
        }

        loop {
            let mut decoded = ffmpeg::frame::Audio::empty();
            match decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    let mut interleaved = Vec::new();
                    append_audio_frame(&decoded, &mut interleaved)?;
                    if !interleaved.is_empty() {
                        on_frame(&interleaved, sample_rate, channels)?;
                        emitted_frames = emitted_frames.saturating_add(
                            interleaved.len() / (channels as usize).max(1),
                        );
                        if emitted_frames >= max_frames {
                            return Ok(true);
                        }
                    }
                }
                Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => break,
                Err(ffmpeg::Error::Eof) => break,
                Err(e) => return Err(ffmpeg_err("ffmpeg receive_frame failed", e)),
            }
        }
    }

    let _ = decoder.send_eof();
    loop {
        let mut decoded = ffmpeg::frame::Audio::empty();
        match decoder.receive_frame(&mut decoded) {
            Ok(()) => {
                let mut interleaved = Vec::new();
                append_audio_frame(&decoded, &mut interleaved)?;
                if !interleaved.is_empty() {
                    on_frame(&interleaved, sample_rate, channels)?;
                    emitted_frames = emitted_frames.saturating_add(
                        interleaved.len() / (channels as usize).max(1),
                    );
                    if emitted_frames >= max_frames {
                        return Ok(true);
                    }
                }
            }
            Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => continue,
            Err(ffmpeg::Error::Eof) => break,
            Err(_) => break,
        }
    }

    Ok(false)
}

/// Decode at most `max_frames` sample frames from a media file. Used by the
/// file-browser preview path so clicking a long video never decodes its whole
/// audio track synchronously.
pub fn decode_media_audio_prefix_f32(
    path: &Path,
    preferred_stream: Option<usize>,
    max_frames: usize,
) -> Result<(u32, u16, Vec<f32>), String> {
    let mut ictx = open_input(path)?;
    let audio_index = audio_stream_index(&ictx, preferred_stream)?;
    let stream = ictx
        .streams()
        .nth(audio_index)
        .ok_or_else(|| "ffmpeg: audio stream disappeared".to_string())?;
    let params = stream.parameters();
    let context = ffmpeg::codec::context::Context::from_parameters(params)
        .map_err(|e| ffmpeg_err("ffmpeg codec context failed", e))?;
    let mut decoder = context
        .decoder()
        .audio()
        .map_err(|e| ffmpeg_err("ffmpeg audio decoder failed", e))?;
    let sample_rate = decoder.rate().max(1);
    let channels = decoder.channel_layout().channels().max(1) as u16;
    let mut out = Vec::new();

    visit_audio_frames_until(
        &mut ictx,
        &mut decoder,
        audio_index,
        max_frames.max(1),
        &mut |frame, _, _| {
            out.extend_from_slice(frame);
            Ok(())
        },
    )?;

    Ok((sample_rate, channels, out))
}


/// Extract one audio stream of a media file to a WAV file next to the source.
///
/// The cache file is named `<stem>.hifi_audio_<stream>.wav` and is overwritten
/// on every call so stale extracts can never desynchronize from the source.
pub fn extract_audio_stream_to_wav(
    path: &Path,
    stream_index: usize,
) -> Result<String, String> {
    let probe = probe_media(path, 0, Some(stream_index))
        .ok_or_else(|| format!("ffmpeg failed to probe stream {stream_index}"))?;
    let sample_rate = probe.sample_rate.max(1);
    let channels = probe.channels.max(1);

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("media");
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = format!("{stem}.hifi_audio_{stream_index}.wav");
    let out_path = parent.join(&file_name);
    let temp_fallback = std::env::temp_dir()
        .join("HiFiShifterMedia")
        .join(&file_name);

    let spec = hound::WavSpec {
        channels: channels.max(1),
        sample_rate: sample_rate.max(1),
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = match hound::WavWriter::create(&out_path, spec) {
        Ok(writer) => writer,
        Err(first_error) => {
            if let Some(temp_parent) = temp_fallback.parent() {
                let _ = std::fs::create_dir_all(temp_parent);
            }
            hound::WavWriter::create(&temp_fallback, spec).map_err(|e| {
                format!(
                    "failed to create {} ({first_error}) and {}: {e}",
                    out_path.display(),
                    temp_fallback.display()
                )
            })?
        }
    };
    let out_path = if out_path.exists() { out_path } else { temp_fallback };

    visit_media_audio_frames(path, Some(stream_index), |frame, _sr, _ch| {
        for &sample in frame {
            writer.write_sample(sample).map_err(|e| e.to_string())?;
        }
        Ok(())
    })
    .map_err(|e| format!("ffmpeg stream extraction failed: {e}"))?;

    writer.finalize().map_err(|e| e.to_string())?;
    Ok(out_path.to_string_lossy().into_owned())
}

fn read_plane<T: Copy>(frame: &ffmpeg::frame::Audio, index: usize, samples: usize) -> &[T] {
    let bytes = frame.data(index);
    let count = (bytes.len() / std::mem::size_of::<T>()).min(samples);
    // FFmpeg allocates audio buffers with sufficient alignment for the sample type.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, count) }
}

fn read_packed<T: Copy>(frame: &ffmpeg::frame::Audio, samples: usize, channels: usize) -> &[T] {
    let bytes = frame.data(0);
    let count = (bytes.len() / std::mem::size_of::<T>()).min(samples.saturating_mul(channels));
    // FFmpeg allocates audio buffers with sufficient alignment for the sample type.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, count) }
}

fn append_audio_frame(frame: &ffmpeg::frame::Audio, out: &mut Vec<f32>) -> Result<(), String> {
    let samples = frame.samples();
    let channels = frame.channels().max(1) as usize;
    if samples == 0 || channels == 0 {
        return Ok(());
    }

    let planar = frame.is_planar();

    match frame.format() {
        Sample::U8(_) => {
            if planar {
                for frame_idx in 0..samples {
                    for ch in 0..channels {
                        let v = frame.plane::<u8>(ch).get(frame_idx).copied().unwrap_or(128);
                        out.push(v as f32 / 128.0 - 1.0);
                    }
                }
            } else {
                for &v in read_packed::<u8>(frame, samples, channels) {
                    out.push(v as f32 / 128.0 - 1.0);
                }
            }
        }
        Sample::I16(_) => {
            if planar {
                for frame_idx in 0..samples {
                    for ch in 0..channels {
                        let v = frame.plane::<i16>(ch).get(frame_idx).copied().unwrap_or(0);
                        out.push(v as f32 / 32768.0);
                    }
                }
            } else {
                for &v in read_packed::<i16>(frame, samples, channels) {
                    out.push(v as f32 / 32768.0);
                }
            }
        }
        Sample::I32(_) => {
            if planar {
                for frame_idx in 0..samples {
                    for ch in 0..channels {
                        let v = frame.plane::<i32>(ch).get(frame_idx).copied().unwrap_or(0);
                        out.push(v as f32 / 2_147_483_648.0);
                    }
                }
            } else {
                for &v in read_packed::<i32>(frame, samples, channels) {
                    out.push(v as f32 / 2_147_483_648.0);
                }
            }
        }
        Sample::I64(_) => {
            if planar {
                for ch in 0..channels {
                    for &v in read_plane::<i64>(frame, ch, samples) {
                        out.push((v as f64 / 9_223_372_036_854_775_808.0) as f32);
                    }
                }
            } else {
                for &v in read_packed::<i64>(frame, samples, channels) {
                    out.push((v as f64 / 9_223_372_036_854_775_808.0) as f32);
                }
            }
        }
        Sample::F32(_) => {
            if planar {
                for frame_idx in 0..samples {
                    for ch in 0..channels {
                        let v = frame.plane::<f32>(ch).get(frame_idx).copied().unwrap_or(0.0);
                        out.push(v);
                    }
                }
            } else {
                out.extend_from_slice(read_packed::<f32>(frame, samples, channels));
            }
        }
        Sample::F64(_) => {
            if planar {
                for frame_idx in 0..samples {
                    for ch in 0..channels {
                        let v = frame.plane::<f64>(ch).get(frame_idx).copied().unwrap_or(0.0);
                        out.push(v as f32);
                    }
                }
            } else {
                for &v in read_packed::<f64>(frame, samples, channels) {
                    out.push(v as f32);
                }
            }
        }
        Sample::None => {}
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_video_audio_when_test_file_provided() {
        let Ok(path) = std::env::var("HIFISHIFTER_TEST_MEDIA") else {
            return;
        };
        let probe = probe_media(Path::new(&path), 32, None).expect("probe");
        assert!(probe.has_video_stream);
        assert!(probe.sample_rate > 0);
        assert!(probe.duration_sec > 0.0);
        assert_eq!(probe.waveform_preview.len(), 32);

        let (sr, ch, pcm) = decode_media_audio_f32_interleaved(Path::new(&path), None)
            .expect("decode");
        assert!(sr > 0);
        assert!(ch > 0);
        assert!(pcm.len() >= probe.total_frames as usize * ch as usize / 2);
    }
}
