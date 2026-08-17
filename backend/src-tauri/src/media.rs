//! Media-file (audio + video container) support via Symphonia.
//!
//! Both audio-only files and audio tracks embedded in video containers are
//! decoded with Symphonia. The registry additionally registers the FDK AAC and
//! libopus adapters so AAC-HE / Opus media decode through those codecs.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use symphonia::core::codecs::audio::{AudioCodecParameters, AudioDecoderOptions};
use symphonia::core::codecs::registry::CodecRegistry;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader, Track, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::{MetadataOptions, Tag};
use symphonia::core::units::{Duration, TimeBase};

pub const AUDIO_EXTENSIONS: &[&str] = &[
    "wav", "mp3", "flac", "ogg", "oga", "opus", "aac", "m4a", "aif", "aiff", "wma", "ac3", "eac3",
    "ape", "wv", "mp2", "mpa", "dts", "amr",
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

/// Shared codec registry: every Symphonia codec enabled in `Cargo.toml`, with
/// FDK AAC replacing the built-in AAC decoder and libopus registered for Opus.
pub(crate) fn codec_registry() -> &'static CodecRegistry {
    static REGISTRY: OnceLock<CodecRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut registry = CodecRegistry::new();
        symphonia::default::register_enabled_codecs(&mut registry);

        // Registered after the native codecs on purpose: registration for the
        // same codec id at the same tier replaces the previous decoder, so FDK
        // AAC wins over Symphonia's partial native AAC implementation.
        registry.register_audio_decoder::<symphonia_adapter_fdk_aac::AacDecoder>();
        registry.register_audio_decoder::<symphonia_adapter_libopus::OpusDecoder>();
        registry
    })
}

fn open_format(path: &Path) -> Result<Box<dyn FormatReader>, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| format!("symphonia probe failed: {e}"))
}

fn select_audio_track(
    format: &dyn FormatReader,
    preferred: Option<usize>,
) -> Result<(&Track, usize), String> {
    let audio_tracks = || {
        format
            .tracks()
            .iter()
            .filter(|track| track.track_type() == Some(TrackType::Audio))
    };

    if let Some(index) = preferred {
        let track = audio_tracks()
            .nth(index)
            .ok_or_else(|| format!("symphonia: audio stream {index} not found"))?;
        return Ok((track, index));
    }

    let track = format
        .default_track(TrackType::Audio)
        .or_else(|| audio_tracks().next())
        .ok_or_else(|| "symphonia: no audio stream".to_string())?;
    let index = audio_tracks()
        .position(|candidate| candidate.id == track.id)
        .unwrap_or(0);
    Ok((track, index))
}

fn audio_params(track: &Track) -> Result<&AudioCodecParameters, String> {
    track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or_else(|| format!("symphonia: track {} is not an audio track", track.id))
}

fn codec_name(params: &AudioCodecParameters) -> String {
    codec_registry()
        .get_audio_decoder(params.codec)
        .map(|decoder| decoder.codec.info.short_name.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn duration_ticks_to_sec(time_base: Option<TimeBase>, ticks: u64, sample_rate: u32) -> f64 {
    if ticks == 0 {
        return 0.0;
    }

    if let Some(time_base) = time_base {
        return time_base
            .calc_duration(Duration::new(ticks))
            .map(|time| time.as_secs_f64())
            .unwrap_or_else(|| ticks as f64);
    }

    if sample_rate > 0 {
        ticks as f64 / sample_rate as f64
    } else {
        ticks as f64
    }
}

fn track_duration_sec(track: &Track) -> Option<f64> {
    if let (Some(time_base), Some(duration)) = (track.time_base, track.duration) {
        if let Some(time) = time_base.calc_duration(duration) {
            return Some(time.as_secs_f64());
        }
    }

    let sample_rate = audio_params(track).ok()?.sample_rate.unwrap_or(0);
    if sample_rate > 0 {
        if let Some(num_frames) = track.num_frames {
            return Some(num_frames as f64 / sample_rate as f64);
        }
    }

    None
}

/// Measure the selected track's playable duration by walking its packets.
///
/// `Packet::dur` is already gapless/trim-aware (valid frames only), so summing
/// it is preferred. If a demuxer emits packets without durations, fall back to
/// the span between the first PTS and the last PTS + decoded block duration.
fn scan_track_duration_sec(
    format: &mut dyn FormatReader,
    track_id: u32,
    time_base: Option<TimeBase>,
    sample_rate: u32,
) -> Result<f64, String> {
    let mut sum_valid_ticks: u128 = 0;
    let mut first_pts: Option<i64> = None;
    let mut last_end: i128 = 0;

    loop {
        match format.next_packet() {
            Ok(Some(packet)) => {
                if packet.track_id != track_id {
                    continue;
                }

                sum_valid_ticks = sum_valid_ticks.saturating_add(u128::from(packet.dur.get()));
                let pts = packet.pts.get();
                if first_pts.is_none() {
                    first_pts = Some(pts);
                }
                let end = i128::from(pts) + i128::from(packet.block_dur().get());
                last_end = last_end.max(end);
            }
            Ok(None) => break,
            Err(Error::ResetRequired) => {
                return Err("symphonia: decoder reset required while measuring duration".to_string())
            }
            Err(Error::IoError(_)) => break,
            Err(e) => return Err(format!("symphonia packet read failed: {e}")),
        }
    }

    let ticks = if sum_valid_ticks > 0 {
        sum_valid_ticks.min(u128::from(u64::MAX)) as u64
    } else if let Some(first) = first_pts {
        ((last_end - i128::from(first)).max(0) as u128).min(u128::from(u64::MAX)) as u64
    } else {
        0
    };

    Ok(duration_ticks_to_sec(time_base, ticks, sample_rate))
}

struct MediaHeader {
    track: Track,
    audio_stream_index: usize,
    sample_rate: u32,
    channels: u16,
    total_frames: u64,
    duration_sec: f64,
    has_video_stream: bool,
    container_format: String,
    audio_stream_count: usize,
}

fn read_media_header(path: &Path, preferred_stream: Option<usize>) -> Result<MediaHeader, String> {
    let mut format = open_format(path)?;

    let container_format = format.format_info().short_name.to_string();
    let has_video_stream = format
        .tracks()
        .iter()
        .any(|track| track.track_type() == Some(TrackType::Video));
    let audio_stream_count = format
        .tracks()
        .iter()
        .filter(|track| track.track_type() == Some(TrackType::Audio))
        .count();

    let (selected_track, audio_stream_index) =
        select_audio_track(format.as_ref(), preferred_stream)?;
    let track = selected_track.clone();
    let params = audio_params(&track)?;
    let sample_rate = params.sample_rate.unwrap_or(0);
    let channels = params
        .channels
        .as_ref()
        .map(|channels| channels.count())
        .unwrap_or(1)
        .max(1) as u16;

    let duration_sec = match track_duration_sec(&track) {
        Some(duration) => duration,
        None => scan_track_duration_sec(format.as_mut(), track.id, track.time_base, sample_rate)?,
    };

    let resolved_sample_rate = if sample_rate > 0 { sample_rate } else { 44100 };
    let total_frames = track.num_frames.unwrap_or_else(|| {
        if duration_sec.is_finite() && duration_sec > 0.0 {
            (duration_sec * resolved_sample_rate as f64)
                .round()
                .max(0.0) as u64
        } else {
            0
        }
    });

    Ok(MediaHeader {
        track,
        audio_stream_index,
        sample_rate: resolved_sample_rate,
        channels,
        total_frames,
        duration_sec,
        has_video_stream,
        container_format,
        audio_stream_count,
    })
}

fn find_tag_value(tags: &[Tag], key: &str) -> Option<String> {
    tags.iter()
        .find(|tag| tag.raw.key.eq_ignore_ascii_case(key))
        .map(|tag| tag.raw.value.to_string())
}

fn track_titles(format: &mut dyn FormatReader) -> HashMap<u32, String> {
    let mut metadata = format.metadata();
    let Some(revision) = metadata.skip_to_latest() else {
        return HashMap::new();
    };

    revision
        .per_track
        .iter()
        .filter_map(|per_track| {
            find_tag_value(&per_track.metadata.tags, "title")
                .map(|title| (per_track.track_id as u32, title))
        })
        .collect()
}

pub fn list_audio_streams(path: &Path) -> Result<Vec<MediaAudioStream>, String> {
    let mut format = open_format(path)?;
    let tracks: Vec<Track> = format
        .tracks()
        .iter()
        .filter(|track| track.track_type() == Some(TrackType::Audio))
        .cloned()
        .collect();

    if tracks.is_empty() {
        return Ok(Vec::new());
    }

    let titles = track_titles(format.as_mut());

    // Most demuxers expose a duration on the track itself. Only walk the packet
    // stream when at least one audio track is missing one.
    let mut measured_durations: HashMap<u32, f64> = HashMap::new();
    if tracks
        .iter()
        .any(|track| track_duration_sec(track).is_none())
    {
        let mut valid_ticks: HashMap<u32, u128> = HashMap::new();
        loop {
            match format.next_packet() {
                Ok(Some(packet)) => {
                    if tracks.iter().any(|track| track.id == packet.track_id) {
                        let entry = valid_ticks.entry(packet.track_id).or_insert(0);
                        *entry = entry.saturating_add(u128::from(packet.dur.get()));
                    }
                }
                Ok(None) => break,
                Err(Error::IoError(_)) | Err(Error::ResetRequired) => break,
                Err(_) => break,
            }
        }

        for track in &tracks {
            if track_duration_sec(track).is_none() {
                let ticks = valid_ticks.get(&track.id).copied().unwrap_or(0) as u64;
                let sample_rate = audio_params(track)
                    .ok()
                    .and_then(|params| params.sample_rate)
                    .unwrap_or(0);
                measured_durations.insert(
                    track.id,
                    duration_ticks_to_sec(track.time_base, ticks, sample_rate),
                );
            }
        }
    }

    let mut out = Vec::with_capacity(tracks.len());
    for (index, track) in tracks.into_iter().enumerate() {
        let params = match audio_params(&track) {
            Ok(params) => params,
            Err(_) => continue,
        };

        let duration_sec = track_duration_sec(&track)
            .or_else(|| measured_durations.get(&track.id).copied())
            .unwrap_or(0.0);

        out.push(MediaAudioStream {
            index,
            title: titles.get(&track.id).cloned(),
            language: track.language.clone(),
            codec: codec_name(params),
            sample_rate: params.sample_rate.unwrap_or(0),
            channels: params.channels.as_ref().map(|c| c.count()).unwrap_or(0) as u16,
            duration_sec,
        });
    }

    Ok(out)
}

/// Probe the first (or requested) audio stream of a media file.
///
/// When `preview_points > 0`, the complete audio stream is decoded once and a
/// downsampled min/max preview is produced (same behaviour as the WAV path).
/// For header-only calls pass `preview_points = 0`.
pub fn probe_media(
    path: &Path,
    preview_points: usize,
    preferred_stream: Option<usize>,
) -> Option<MediaProbe> {
    let header = read_media_header(path, preferred_stream).ok()?;

    let waveform_preview = if preview_points > 0 {
        compute_preview(path, &header, preview_points)
    } else {
        Vec::new()
    };

    Some(MediaProbe {
        sample_rate: header.sample_rate,
        channels: header.channels,
        duration_sec: header.duration_sec,
        total_frames: header.total_frames,
        waveform_preview,
        has_video_stream: header.has_video_stream,
        container_format: header.container_format,
        audio_stream_index: header.audio_stream_index,
        audio_stream_count: header.audio_stream_count,
    })
}

fn compute_preview(path: &Path, header: &MediaHeader, preview_points: usize) -> Vec<f32> {
    let points = preview_points.max(2);
    let estimated_frames = if header.total_frames > 0 {
        header.total_frames as usize
    } else if header.duration_sec > 0.0 {
        (header.duration_sec * header.sample_rate as f64).max(1.0) as usize
    } else {
        0
    };
    let estimated_samples = estimated_frames.saturating_mul(header.channels.max(1) as usize);

    let mut min_bucket = vec![f32::INFINITY; points];
    let mut max_bucket = vec![f32::NEG_INFINITY; points];
    let mut seen_samples = 0usize;

    let _ = decode_track_frames_until(
        path,
        Some(header.audio_stream_index),
        usize::MAX,
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
            if max.abs() >= min.abs() {
                max
            } else {
                min
            }
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
    let mut out = Vec::new();
    let (sample_rate, channels, _) =
        decode_track_frames_until(path, preferred_stream, usize::MAX, &mut |frame, _, _| {
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
    let (sample_rate, channels, _) =
        decode_track_frames_until(path, preferred_stream, usize::MAX, &mut on_frame)?;
    Ok((sample_rate, channels))
}

/// Decode at most `max_frames` sample frames from a media file. Used by the
/// file-browser preview path so clicking a long video never decodes its whole
/// audio track synchronously.
pub fn decode_media_audio_prefix_f32(
    path: &Path,
    preferred_stream: Option<usize>,
    max_frames: usize,
) -> Result<(u32, u16, Vec<f32>), String> {
    let mut out = Vec::new();
    let (sample_rate, channels, _) = decode_track_frames_until(
        path,
        preferred_stream,
        max_frames.max(1),
        &mut |frame, _, _| {
            out.extend_from_slice(frame);
            Ok(())
        },
    )?;
    Ok((sample_rate, channels, out))
}

fn decode_track_frames_until<F>(
    path: &Path,
    preferred_stream: Option<usize>,
    max_frames: usize,
    on_frame: &mut F,
) -> Result<(u32, u16, bool), String>
where
    F: FnMut(&[f32], u32, u16) -> Result<(), String>,
{
    let mut format = open_format(path)?;
    let (selected_track, _) = select_audio_track(format.as_ref(), preferred_stream)?;
    let track = selected_track.clone();
    let params = audio_params(&track)?.clone();

    let mut decoder = codec_registry()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|e| format!("symphonia audio decoder failed: {e}"))?;

    let mut sample_rate = params.sample_rate.unwrap_or(0);
    let declared_channels = params
        .channels
        .as_ref()
        .map(|c| c.count())
        .unwrap_or(1)
        .max(1) as u16;
    let track_id = track.id;
    let mut emitted_frames = 0usize;
    let mut frame_buf: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(Error::ResetRequired) => {
                return Err("symphonia: decoder reset required".to_string())
            }
            Err(Error::IoError(_)) => break,
            Err(e) => return Err(format!("symphonia packet read failed: {e}")),
        };

        if packet.track_id != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(Error::DecodeError(_)) => continue,
            Err(Error::IoError(_)) => break,
            Err(Error::ResetRequired) => {
                return Err("symphonia: decoder reset required".to_string())
            }
            Err(e) => return Err(format!("symphonia decode failed: {e}")),
        };

        if decoded.is_empty() {
            continue;
        }

        let spec = decoded.spec();
        if sample_rate == 0 {
            sample_rate = spec.rate().max(1);
        }
        let channels = spec.channels().count().max(1) as u16;

        frame_buf.clear();
        decoded.copy_to_vec_interleaved::<f32>(&mut frame_buf);

        if !frame_buf.is_empty() {
            on_frame(&frame_buf, sample_rate, channels)?;
            emitted_frames = emitted_frames.saturating_add(frame_buf.len() / channels as usize);
            if emitted_frames >= max_frames {
                return Ok((
                    if sample_rate > 0 { sample_rate } else { 44100 },
                    channels,
                    true,
                ));
            }
        }
    }

    let _ = decoder.finalize();

    Ok((
        if sample_rate > 0 { sample_rate } else { 44100 },
        declared_channels,
        false,
    ))
}

/// Extract one audio stream of a media file to a WAV file next to the source.
///
/// The cache file is named `<stem>.hifi_audio_<stream>.wav` and is overwritten
/// on every call so stale extracts can never desynchronize from the source.
pub fn extract_audio_stream_to_wav(path: &Path, stream_index: usize) -> Result<String, String> {
    let probe = probe_media(path, 0, Some(stream_index))
        .ok_or_else(|| format!("symphonia failed to probe stream {stream_index}"))?;
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
    let out_path = if out_path.exists() {
        out_path
    } else {
        temp_fallback
    };

    visit_media_audio_frames(path, Some(stream_index), |frame, _sr, _ch| {
        for &sample in frame {
            writer.write_sample(sample).map_err(|e| e.to_string())?;
        }
        Ok(())
    })
    .map_err(|e| format!("symphonia stream extraction failed: {e}"))?;

    writer.finalize().map_err(|e| e.to_string())?;
    Ok(out_path.to_string_lossy().into_owned())
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

        let (sr, ch, pcm) =
            decode_media_audio_f32_interleaved(Path::new(&path), None).expect("decode");
        assert!(sr > 0);
        assert!(ch > 0);
        assert!(pcm.len() >= probe.total_frames as usize * ch as usize / 2);
    }

    #[test]
    fn decodes_audio_when_demo_mp3_present() {
        let path =
            Path::new("third_party/signalsmith-stretch/signalsmith-stretch/web/demo/loop.mp3");
        if !path.is_file() {
            return;
        }

        let probe = probe_media(path, 16, None).expect("probe demo mp3");
        assert!(probe.sample_rate > 0);
        assert!(probe.channels > 0);
        assert!(probe.duration_sec > 0.0);
        assert_eq!(probe.waveform_preview.len(), 16);

        let (sr, ch, pcm) =
            decode_media_audio_f32_interleaved(path, None).expect("decode demo mp3");
        assert!(sr > 0);
        assert!(ch > 0);
        assert!(!pcm.is_empty());
    }
}
