//! Convert selected HiFiShifter clips into REAPERMedia clipboard data.

use crate::reaper_parser::{
    ReaperData, ReaperIgnTempo, ReaperItem, ReaperMidiEvent, ReaperMidiSourceData, ReaperSource,
    ReaperTrack,
};
use crate::state::{Clip, TimelineState};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
pub struct ReaperExportResult {
    pub bytes: Vec<u8>,
    pub exported_clip_count: usize,
    pub skipped_clip_count: usize,
    pub track_count: usize,
}

fn fade_shape(curve: &str) -> f64 {
    match curve {
        "linear" => 0.0,
        "sine" => 1.0,
        "exponential" => 2.0,
        "logarithmic" => 3.0,
        "scurve" => 4.0,
        _ => 1.0,
    }
}

fn fade_values(curve: &str, length_sec: f64) -> Vec<f64> {
    vec![
        fade_shape(curve),
        length_sec.max(0.0),
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
}

fn source_bounds(clip: &Clip) -> (f64, f64) {
    let start = clip.source_start_sec.max(0.0);
    let mut end = clip.source_end_sec.max(start);
    if end <= start {
        end = clip
            .duration_sec
            .filter(|duration| *duration > start)
            .unwrap_or(start + clip.length_sec * clip.playback_rate.max(0.01) as f64);
    }
    (start, end.max(start))
}

/// Map a media source path to the SOURCE type REAPER accepts.
///
/// REAPER only recognises WAVE, MP3 and FLAC as named audio source types.
/// Video containers and every other format (WMA, OGG, M4A, etc.) must be
/// written as VIDEO, otherwise REAPER fails to parse the clipboard data.
fn reaper_source_type(path: &str) -> &'static str {
    let path = std::path::Path::new(path);

    if crate::media::is_video_extension(path) {
        return "VIDEO";
    }

    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("wav") => "WAVE",
        Some(ext) if ext.eq_ignore_ascii_case("mp3") => "MP3",
        Some(ext) if ext.eq_ignore_ascii_case("flac") => "FLAC",
        _ => "VIDEO",
    }
}

fn audio_source(clip: &Clip, _rate: f64, source_span_sec: f64) -> ReaperSource {
    let path = clip.source_path.clone().unwrap_or_default();
    let (start, _) = source_bounds(clip);

    if clip.reversed {
        let mut source = ReaperSource::new();
        source.source_type = "SECTION".to_string();
        source.file_path = path.clone();
        source.section_mode = 1;
        source.section_start_sec = Some(start);
        source.section_length_sec = Some(source_span_sec.max(0.0));
        source
    } else {
        let mut source = ReaperSource::new();
        source.source_type = reaper_source_type(&path).to_string();
        source.file_path = path;
        source.section_mode = 0;
        source.section_start_sec = None;
        source.section_length_sec = None;
        source
    }
}

fn midi_source(clip: &Clip, bpm: f64) -> ReaperSource {
    const PPQ: u32 = 960;

    let notes = clip.midi_note_data.as_deref().unwrap_or(&[]);
    let mut events: Vec<ReaperMidiEvent> = Vec::new();
    let mut raw_events: Vec<(u64, u8, u8, u8)> = Vec::new();

    for note in notes {
        let note_number = note.note.clamp(0.0, 127.0).round() as u8;
        let velocity = note.velocity;
        let channel = note.channel.min(15);
        let start_tick =
            (note.start_sec.max(0.0) * PPQ as f64 * bpm.max(1.0) / 60.0).round() as u64;
        let end_tick = ((note.end_sec.max(note.start_sec) * PPQ as f64 * bpm.max(1.0) / 60.0)
            .round() as u64)
            .max(start_tick);
        raw_events.push((start_tick, 0x90 | channel, note_number, velocity));
        raw_events.push((end_tick, 0x80 | channel, note_number, 0));
    }

    raw_events.sort_by_key(|event| (event.0, event.1));
    let mut cumulative_tick = 0_u64;
    for (tick, status, data1, data2) in raw_events {
        events.push(ReaperMidiEvent {
            tick_offset: tick.saturating_sub(cumulative_tick),
            status,
            data1,
            data2,
        });
        cumulative_tick = tick;
    }

    let mut source = ReaperSource::new();
    source.source_type = "MIDI".to_string();
    source.midi_source = Some(ReaperMidiSourceData {
        ticks_per_qn: PPQ,
        events,
        igntempo: Some(ReaperIgnTempo {
            ignore_project: false,
            tempo: bpm.max(1.0),
            beats: 4,
            beat_note: 4,
        }),
    });
    source
}

fn build_item(clip: &Clip, bpm: f64) -> Option<ReaperItem> {
    let mut item = ReaperItem::default();
    item.position = clip.start_sec.max(0.0);
    item.length = clip.length_sec.max(0.001);
    item.snap_offs = 0.0;
    item.is_loop = false;
    item.all_takes = false;
    // 导出“有效 fade”（自动交叉淡化覆盖手动 fade），与渲染一致。
    item.fade_in = fade_values(&clip.fade_in_curve, clip.effective_fade_in_sec());
    item.fade_out = fade_values(&clip.fade_out_curve, clip.effective_fade_out_sec());
    item.mute = vec![if clip.muted { 1 } else { 0 }, 0];
    item.selected = false;

    let rate = clip.playback_rate.max(0.01).min(100.0) as f64;
    let default_take = &mut item.default_take;
    default_take.name = clip.name.clone();
    default_take.vol_pan = vec![clip.gain as f64, 0.0, 1.0, -1.0];
    default_take.play_rate = vec![rate, 1.0, 0.0, -1.0, 0.0, 0.0025];
    default_take.chan_mode = 0;

    if let Some(ref midi_data) = clip.midi_note_data {
        if midi_data.is_empty() {
            return None;
        }
        default_take.s_offs = 0.0;
        default_take.source = Some(midi_source(clip, bpm));
        return Some(item);
    }

    let source_path = clip.source_path.as_deref().unwrap_or("").trim();
    if source_path.is_empty() {
        return None;
    }

    let (start, end) = source_bounds(clip);
    let source_span = (end - start).max(0.0);
    let required_span = clip.length_sec.max(0.0) * rate;
    item.is_loop = source_span > 1e-9 && required_span > source_span + 1e-9;

    let mut source = audio_source(clip, rate, source_span);
    if !clip.reversed {
        source.section_start_sec = None;
        source.section_length_sec = None;
        default_take.s_offs = start;
    } else {
        default_take.s_offs = 0.0;
    }
    default_take.source = Some(source);
    Some(item)
}

pub fn build_reaper_clipboard(
    timeline: &TimelineState,
    clip_ids: &[String],
) -> Result<ReaperExportResult, String> {
    let unique_ids: Vec<&String> = {
        let mut seen = std::collections::HashSet::new();
        clip_ids
            .iter()
            .filter(|id| seen.insert((*id).clone()))
            .collect()
    };

    let mut items_by_track: BTreeMap<usize, Vec<ReaperItem>> = BTreeMap::new();
    let mut skipped_clip_count = 0_usize;

    for clip_id in unique_ids {
        let Some(clip) = timeline.clips.iter().find(|clip| clip.id == *clip_id) else {
            skipped_clip_count += 1;
            continue;
        };
        let Some(track_index) = timeline
            .tracks
            .iter()
            .position(|track| track.id == clip.track_id)
        else {
            skipped_clip_count += 1;
            continue;
        };
        let Some(item) = build_item(clip, timeline.bpm) else {
            skipped_clip_count += 1;
            continue;
        };
        items_by_track
            .entry(track_index)
            .or_default()
            .push(item);
    }

    if items_by_track.is_empty() {
        return Err("reaper_export_no_supported_clips".to_string());
    }

    let track_indices: Vec<usize> = items_by_track.keys().copied().collect();
    let first_track_index = track_indices.first().copied().unwrap_or(0);

    let mut data = ReaperData::default();
    data.is_track_data = false;
    for (offset, track_index) in track_indices.iter().enumerate() {
        let mut track = ReaperTrack::default();
        track.items = items_by_track.remove(track_index).unwrap_or_default();
        track.items.sort_by(|left, right| {
            left.position
                .partial_cmp(&right.position)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        data.tracks.push(track);
        data.track_offsets
            .push(track_index.saturating_sub(first_track_index) as usize);
        if offset == 0 {
            data.track_offsets[0] = 0;
        }
    }

    let exported_clip_count = data.tracks.iter().map(|track| track.items.len()).sum();
    let bytes = crate::reaper_parser::serialize_reaper_clipboard(&data, false);
    Ok(ReaperExportResult {
        bytes,
        exported_clip_count,
        skipped_clip_count,
        track_count: data.tracks.len(),
    })
}

/// Round-trip helper used by tests.
#[cfg(test)]
pub(crate) fn parse_for_test(bytes: &[u8]) -> crate::reaper_parser::ReaperData {
    crate::reaper_parser::parse_clipboard_bytes(bytes).expect("parse exported REAPER clipboard")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi_import::MidiNoteEvent;
    use crate::state::TimelineState;

    #[test]
    fn audio_clip_roundtrips_through_reaper_clipboard() {
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Test Audio".to_string()),
            Some(1.5),
            Some(2.0),
            Some("C:/audio/test.wav".to_string()),
        );
        if let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == clip_id) {
            clip.source_start_sec = 0.25;
            clip.source_end_sec = 3.0;
            clip.playback_rate = 1.0;
            clip.gain = 0.8;
            clip.fade_in_sec = 0.01;
            clip.fade_out_sec = 0.02;
        }

        let export = build_reaper_clipboard(&timeline, &[clip_id]).unwrap();
        assert_eq!(export.exported_clip_count, 1);
        let parsed = parse_for_test(&export.bytes);
        assert_eq!(parsed.tracks.len(), 1);
        assert_eq!(parsed.tracks[0].items.len(), 1);
        let item = &parsed.tracks[0].items[0];
        assert!((item.position - 1.5).abs() < 1e-9);
        assert!((item.length - 2.0).abs() < 1e-9);
        assert!((item.default_take.s_offs - 0.25).abs() < 1e-9);
        assert_eq!(item.default_take.source.as_ref().unwrap().file_path, "C:/audio/test.wav");
    }

    fn source_type_for_path(path: &str) -> String {
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Source Type Test".to_string()),
            Some(0.0),
            Some(1.0),
            Some(path.to_string()),
        );

        let export = build_reaper_clipboard(&timeline, &[clip_id]).unwrap();
        let parsed = parse_for_test(&export.bytes);
        parsed.tracks[0].items[0]
            .default_take
            .source
            .as_ref()
            .unwrap()
            .source_type
            .clone()
    }

    #[test]
    fn audio_source_type_follows_reaper_clipboard_conventions() {
        assert_eq!(source_type_for_path("C:/audio/song.wav"), "WAVE");
        assert_eq!(source_type_for_path("C:/audio/song.MP3"), "MP3");
        assert_eq!(source_type_for_path("C:/audio/song.flac"), "FLAC");
        assert_eq!(source_type_for_path("C:/video/movie.mp4"), "VIDEO");
        assert_eq!(source_type_for_path("C:/video/movie.mkv"), "VIDEO");
        assert_eq!(source_type_for_path("C:/audio/song.wma"), "VIDEO");
        assert_eq!(source_type_for_path("C:/audio/song.ogg"), "VIDEO");
        assert_eq!(source_type_for_path("C:/audio/song"), "VIDEO");
    }

    #[test]
    fn midi_clip_roundtrips_through_reaper_clipboard() {
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Test MIDI".to_string()),
            Some(0.0),
            Some(2.0),
            None,
        );
        if let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == clip_id) {
            clip.midi_note_data = Some(vec![MidiNoteEvent {
                start_sec: 0.5,
                end_sec: 1.0,
                note: 60.0,
                velocity: 100,
                channel: 0,
            }]);
            clip.source_path = None;
        }

        let export = build_reaper_clipboard(&timeline, &[clip_id]).unwrap();
        assert_eq!(export.exported_clip_count, 1);
        let parsed = parse_for_test(&export.bytes);
        let item = &parsed.tracks[0].items[0];
        let source = item.default_take.source.as_ref().unwrap();
        assert_eq!(source.source_type, "MIDI");
        assert!(!source.midi_source.as_ref().unwrap().events.is_empty());
    }

    #[test]
    fn unsupported_clip_is_skipped() {
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("No source".to_string()),
            Some(0.0),
            Some(1.0),
            None,
        );
        if let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == clip_id) {
            clip.source_path = None;
            clip.midi_note_data = None;
        }
        let result = build_reaper_clipboard(&timeline, &[clip_id]);
        assert!(result.is_err());
    }
}
