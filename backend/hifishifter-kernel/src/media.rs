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
use symphonia::core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo, Track, TrackType};
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
pub fn codec_registry() -> &'static CodecRegistry {
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

/// 单遍顺序解码，沿途把 `windows` 指定的帧区间收割出来。
///
/// 每个窗口以 `(start_frame, frame_count)` 给出，回调收到 `(窗口下标, 交错 PCM,
/// 声道数, 采样率)`。窗口之间可以乱序、可以重叠、可以超出文件长度（收不到的
/// 窗口不会被回调）；解码在最后一个窗口的终点处停止。
///
/// # 为什么需要它
///
/// Symphonia 的逐包解码不保证随机访问 —— 想比较"第 4 分钟"的左右声道，就只能
/// 从文件头一路解到第 4 分钟。但**解码**和**保留**是两件事：本函数只保留窗口
/// 覆盖的那几段（默认每段 0.25 秒 × 12 段），途经的其余帧解完即弃，内存占用
/// 与文件长度无关。
///
/// 这是"非 WAV 素材只能看文件头 3 秒"那个缺陷的替代品：它让容器判定也能覆盖
/// 消费区间的**首尾**，而不只是开头。
pub fn visit_media_audio_windows<F>(
    path: &Path,
    preferred_stream: Option<usize>,
    windows: &[(u64, usize)],
    max_frames: usize,
    on_window: &mut F,
) -> Result<usize, String>
where
    F: FnMut(usize, &[f32], u16, u32) -> Result<(), String>,
{
    if windows.is_empty() || max_frames == 0 {
        return Ok(0);
    }

    // 按起点排序（保留原下标，回调要按调用方的编号汇报）。乱序输入不算错误，
    // 但顺序扫描必须有序 —— 在这里收敛掉，调用方不必操心。
    let mut ordered: Vec<(u64, usize, usize)> = windows
        .iter()
        .enumerate()
        .map(|(index, (start, len))| (*start, *len, index))
        .collect();
    ordered.sort_by_key(|(start, _, _)| *start);

    let mut harvested = 0usize;
    let mut cursor: u64 = 0;
    let mut next = 0usize;
    // 已解码但尚未被窗口消费的帧，覆盖 `[retained_start, retained_start + frames)`。
    //
    // 【为什么必须保留而不是只维护"当前累积器"】窗口之间**可以重叠**：同一
    // 音频被两个 Take 以相互重叠的消费区间引用时（切片素材的常态），两个区间的
    // 窗口可能落在同一段音频上。若只按解码游标顺序累积，走到第二个窗口时游标
    // 已经越过它的起点，就会拿**当前位置**的音频冒充该窗口 —— 静默地比较了
    // 错误的音频位置。保留一段已解码缓冲后，每个窗口都从自己的起点精确取数。
    let mut retained: Vec<f32> = Vec::new();
    let mut retained_start: u64 = 0;
    let mut channels_seen: u16 = 0;

    let result = decode_track_frames_until(
        path,
        preferred_stream,
        max_frames,
        &mut |frame: &[f32], rate: u32, channels: u16| {
            let ch = channels.max(1) as usize;
            let frame_count = frame.len() / ch;
            if frame_count == 0 {
                return Ok(());
            }

            // 声道数在同一个容器内变化（罕见）：保留缓冲的交错步长随之失效，
            // 丢弃它并重新起算。
            if channels_seen != channels {
                retained.clear();
                retained_start = cursor;
                channels_seen = channels;
            }

            retained.extend_from_slice(&frame[..frame_count * ch]);
            cursor += frame_count as u64;

            // 顺序满足所有"终点已解码到"的窗口。
            while next < ordered.len() {
                let (start, len, index) = ordered[next];
                let end = start.saturating_add(len as u64);
                if end > cursor {
                    break;
                }
                if start < retained_start {
                    // 起点已被丢弃（声道数变化等）⇒ 该窗口无法精确取数：跳过它，
                    // 不拿错位的音频充数。调用方会看到"没收到"并据此只允许
                    // 下发安全结论。
                    next += 1;
                    continue;
                }
                let from = ((start - retained_start) as usize) * ch;
                let to = from + len * ch;
                if to > retained.len() {
                    next += 1;
                    continue;
                }
                on_window(index, &retained[from..to], channels, rate)?;
                harvested += 1;
                next += 1;
            }

            // 丢弃不再被任何剩余窗口需要的保留前缀，使内存与文件长度无关。
            let keep_from = if next < ordered.len() {
                ordered[next].0
            } else {
                cursor
            };
            let drop_frames = keep_from.saturating_sub(retained_start) as usize;
            if drop_frames > 0 {
                let drop_samples = (drop_frames * ch).min(retained.len());
                retained.drain(..drop_samples);
                retained_start += (drop_samples / ch) as u64;
            }

            if next >= ordered.len() {
                // 全部窗口已收割：用哨兵错误让解码循环提前收尾。
                return Err(WINDOWS_DONE.to_string());
            }
            Ok(())
        },
    );

    match result {
        Ok(_) => Ok(harvested),
        // 提前收尾不是错误 —— 这是我们主动请求的停止。
        Err(error) if error == WINDOWS_DONE => Ok(harvested),
        Err(error) => Err(error),
    }
}

/// [`visit_media_audio_windows`] 提前收尾用的内部哨兵（不会外泄给调用方）。
const WINDOWS_DONE: &str = "__hifi_windows_done__";

/// 用容器的 seek 能力逐窗口取样，避开"从第 0 帧一路解到最后一个窗口"。
///
/// 语义与 [`visit_media_audio_windows`] 完全一致（同样的入参、同样的回调、同样
/// 按原下标汇报），只是取数方式不同：那一个是单遍顺序解码沿途收割，本函数对每个
/// 窗口 seek 到它附近的码流位置后只解出这一段。**窗口数远小于文件长度时**（默认
/// 12 个 0.25 秒窗口 vs 半小时音频）这能省掉几个数量级的解码量。
///
/// # 为什么会判不准，以及怎么兜住
///
/// 有损编码只能从关键帧起解，且 seek 是 Coarse 语义（尽力落在请求点附近）。
/// 因此 seek 后实际拿到的第一帧可能早于也可能晚于请求的窗口起点。本函数按
/// [`symphonia::core::formats::SeekedTo::actual_ts`] 报告的**实际**时间戳对齐：
/// 只从 ≥ 窗口起点的那一帧开始收割，一旦越过窗口终点立即停。
///
/// 由调用方判断"实际起点是否晚于允许的范围"来决定要不要丢弃这个窗口 —— 音频
/// 判定宁可少看一段（走保守结论），也绝不能拿**错位的音频**冒充目标位置。
/// 返回值的 `Vec<Option<u64>>` 就是每个窗口的实际起始帧（`None` = 没拿到），
/// 与 `windows` 一一对应。
///
/// 任何一步失败（不支持 seek / seek 报错 / 解不出足够样本）时返回 `Err`，调用方
/// 据此回落到顺序收割路径 —— 这是一次纯增量的性能优化，不引入新的失败模式。
pub fn visit_media_audio_windows_by_seek<F>(
    path: &Path,
    preferred_stream: Option<usize>,
    windows: &[(u64, usize)],
    sample_rate_hint: u32,
    max_decode_frames_per_window: usize,
    on_window: &mut F,
) -> Result<Vec<Option<u64>>, String>
where
    F: FnMut(usize, &[f32], u16, u32) -> Result<(), String>,
{
    if windows.is_empty() {
        return Ok(Vec::new());
    }
    let stride = max_decode_frames_per_window.max(1);

    let mut starts: Vec<Option<u64>> = vec![None; windows.len()];
    // 每个窗口独立一次 seek + 短解码。逐窗独立的实现复杂度远低于对所有窗口做
    // 全局排序与规划，而窗口数本来就只有十几个。
    for (index, (start_frame, want_frames)) in windows.iter().enumerate() {
        if *want_frames == 0 {
            continue;
        }
        let mut format = open_format(path)?;
        let (selected_track, _) = select_audio_track(format.as_ref(), preferred_stream)?;
        let track = selected_track.clone();
        let params = audio_params(&track)?.clone();
        let track_id = track.id;
        let codec_rate = params.sample_rate.unwrap_or(sample_rate_hint).max(1);
        // 求 seek 时间戳要用容器自己的采样率，否则请求的位置会系统性偏移。
        let rate_for_seek = if sample_rate_hint > 0 {
            sample_rate_hint
        } else {
            codec_rate
        };

        // Coarse：允许落在请求点附近。对抽样判定足够了 —— 我们随后按实际时间戳
        // 对齐，宁可少收也不错位。
        let Some(target_time) = symphonia::core::units::Time::try_from_secs_f64(
            *start_frame as f64 / rate_for_seek as f64,
        ) else {
            return Err("seek target out of range".to_string());
        };
        let Ok(seeked) = format.seek(
            SeekMode::Coarse,
            SeekTo::Time {
                time: target_time,
                track_id: Some(track_id),
            },
        ) else {
            return Err("seek not supported".to_string());
        };

        // 实际落点：容器报告落在哪一帧就按哪一帧算，绝不假设它等于请求值。
        // `actual_ts` 是 track timebase 下的 tick，用 Symphonia 自己的换算得到秒
        // （比手算 numer/denom 更不容易搞反），再乘采样率得到帧位。
        let seeked_frame = {
            let seconds = track
                .time_base
                .unwrap_or_default()
                .calc_time_saturating(seeked.actual_ts)
                .as_secs_f64();
            if seconds > 0.0 {
                (seconds * codec_rate as f64).round() as u64
            } else {
                0
            }
        };

        let mut decoder = codec_registry()
            .make_audio_decoder(&params, &AudioDecoderOptions::default())
            .map_err(|e| format!("symphonia audio decoder failed: {e}"))?;

        let start = *start_frame;
        let end = start.saturating_add(*want_frames as u64);
        let mut out: Vec<f32> = Vec::new();
        let mut channels_seen: u16 = 0;
        let mut rate_seen: u32 = 0;
        // 已累积的音频覆盖多少个"自 seek 落点起算"的帧位，用于算真正的起点。
        let mut cursor: u64 = seeked_frame;
        let mut first_collected_frame: Option<u64> = None;
        let mut decoded_frames = 0usize;

        'pump: loop {
            let packet = match format.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => break,
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
                Err(e) => return Err(format!("symphonia decode failed: {e}")),
            };
            if decoded.is_empty() {
                continue;
            }
            let spec = decoded.spec();
            let channels = spec.channels().count().max(1) as u16;
            let rate = if rate_for_seek > 0 {
                rate_for_seek
            } else {
                codec_rate
            };
            let mut frame_buf: Vec<f32> = Vec::new();
            decoded.copy_to_vec_interleaved::<f32>(&mut frame_buf);
            let ch = channels.max(1) as usize;
            let frame_count = frame_buf.len() / ch;
            if frame_count == 0 {
                continue;
            }
            // 声道数中途变化（罕见）会让已收集缓冲的交错步长失效。
            if channels_seen != 0 && channels_seen != channels {
                break 'pump;
            }
            channels_seen = channels;
            rate_seen = rate;

            // 逐帧决定是否收集：缓冲区内帧是均匀的，直接按帧位切片。
            for local in 0..frame_count {
                let frame_pos = cursor;
                if frame_pos >= end {
                    break 'pump;
                }
                if frame_pos >= start {
                    if first_collected_frame.is_none() {
                        first_collected_frame = Some(frame_pos);
                    }
                    let base = local * ch;
                    out.extend_from_slice(&frame_buf[base..base + ch]);
                }
                cursor += 1;
            }
            decoded_frames += frame_count;
            if decoded_frames >= stride {
                break;
            }
        }
        let _ = decoder.finalize();

        let Some(first) = first_collected_frame else {
            // 一个目标帧都没收到（seek 落到窗口之后 / 文件更短）：记为没拿到。
            continue;
        };
        let got_frames = out.len() / channels_seen.max(1) as usize;
        if got_frames == 0 {
            continue;
        }
        starts[index] = Some(first);
        on_window(index, &out, channels_seen, rate_seen)?;
    }

    Ok(starts)
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

    /// 仓库自带的非 WAV 夹具（mp3）。
    ///
    /// 相对路径在 `cargo test` 下取决于工作目录，因此从 `CARGO_MANIFEST_DIR`
    /// 反推仓库根 —— 否则这个夹具会"永远找不到"而让测试静默变成空跑。
    fn demo_mp3() -> Option<std::path::PathBuf> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("third_party/signalsmith-stretch/signalsmith-stretch/web/demo/loop.mp3");
        path.is_file().then_some(path)
    }

    #[test]
    fn window_harvest_returns_exactly_what_was_asked_for() {
        let Some(path) = demo_mp3() else {
            return;
        };
        let header = probe_media(&path, 0, None).expect("probe demo mp3");
        let total = header.total_frames;
        assert!(total > 200_000, "夹具太短，无法取中段窗口");

        let windows = vec![(total / 4, 4_000usize), (total / 2, 6_000usize)];
        let mut got: Vec<(usize, usize)> = Vec::new();
        let harvested = visit_media_audio_windows(&path, None, &windows, total as usize, &mut {
            let mut record = |index: usize, pcm: &[f32], channels: u16, _rate: u32| {
                got.push((index, pcm.len() / channels.max(1) as usize));
                Ok(())
            };
            move |index, pcm, channels, rate| record(index, pcm, channels, rate)
        })
        .expect("harvest");

        assert_eq!(harvested, 2);
        got.sort();
        assert_eq!(got, vec![(0, 4_000), (1, 6_000)], "每个窗口的帧数必须精确");
    }

    #[test]
    fn overlapping_windows_are_taken_from_their_own_starts() {
        // 回归：窗口可以重叠（同一音频被两个消费区间重叠引用）。曾经按解码游标
        // 顺序累积的实现，走到第二个窗口时游标已越过它的起点，于是拿**当前位置**
        // 的音频冒充该窗口 —— 静默比较了错误的音频位置。
        //
        // 判据：一个窗口收到的内容，不得因为它旁边还有别的窗口而改变。
        let Some(path) = demo_mp3() else {
            return;
        };
        let header = probe_media(&path, 0, None).expect("probe demo mp3");
        let total = header.total_frames;
        let start = total / 2;
        let len = 5_000usize;
        // 第二个窗口与第一个重叠一半。
        let a = (start, len);
        let b = (start + len as u64 / 2, len);

        let harvest = |windows: &[(u64, usize)]| -> Vec<(usize, Vec<f32>)> {
            let mut out: Vec<(usize, Vec<f32>)> = Vec::new();
            visit_media_audio_windows(&path, None, windows, total as usize, &mut {
                let mut record = |index: usize, pcm: &[f32], _ch: u16, _rate: u32| {
                    out.push((index, pcm.to_vec()));
                    Ok(())
                };
                move |index, pcm, ch, rate| record(index, pcm, ch, rate)
            })
            .expect("harvest");
            out.sort_by_key(|(index, _)| *index);
            out
        };

        let together = harvest(&[a, b]);
        assert_eq!(together.len(), 2, "两个窗口都必须收到");
        assert_eq!(harvest(&[a]).len(), 1);
        assert_eq!(harvest(&[b]).len(), 1);

        // 单独取与一起取，内容必须逐样本一致。
        assert_eq!(
            together[0].1,
            harvest(&[a])[0].1,
            "窗口 a 的内容不得受 b 影响"
        );
        assert_eq!(
            together[1].1,
            harvest(&[b])[0].1,
            "窗口 b 的内容不得受 a 影响"
        );
        // 重叠段确实重叠（否则上面的断言会因为两个窗口都在读同一段而失去意义）。
        let half = len / 2;
        let ch = header.channels.max(1) as usize;
        assert_eq!(
            together[0].1[half * ch..len * ch],
            together[1].1[..half * ch],
            "重叠段应当逐样本相同"
        );
    }

    #[test]
    fn windows_beyond_the_file_are_simply_not_reported() {
        let Some(path) = demo_mp3() else {
            return;
        };
        let header = probe_media(&path, 0, None).expect("probe demo mp3");
        let total = header.total_frames;
        // 起点在文件之外：收不到，但也不报错（调用方据此判定覆盖不完整）。
        let windows = vec![(total + 100_000, 1_000usize)];
        let mut count = 0usize;
        let harvested = visit_media_audio_windows(
            &path,
            None,
            &windows,
            total as usize,
            &mut |_i, _pcm, _ch, _rate| {
                count += 1;
                Ok(())
            },
        )
        .expect("harvest");
        assert_eq!(harvested, 0);
        assert_eq!(count, 0);
    }

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
