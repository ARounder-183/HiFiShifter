//! Convert selected HiFiShifter clips into REAPERMedia clipboard data.

use crate::reaper_parser::{
    ReaperData, ReaperIgnTempo, ReaperItem, ReaperMidiEvent, ReaperMidiSourceData, ReaperSource,
    ReaperTrack,
};
use crate::state::{Clip, ClipTake, TimelineState};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
pub struct ReaperExportResult {
    pub bytes: Vec<u8>,
    pub exported_clip_count: usize,
    pub skipped_clip_count: usize,
    pub track_count: usize,
}

/// 写出 FADEIN/FADEOUT 数组。
///
/// 布局对照 REAPER 7.x 实测样本（详见 reaper_parser::reaper_fade_shape_dir）：
/// `[shape, 手动长度, 自动交叉淡化长度, 取整形状(镜像槽), 自动selector, 曲率dir, 0]`。
/// 索引 3 在官方样本中存放取整后的基础形状（如形状 1.1 时写 1），保持一致；
/// 第二曲率参数（索引 6）语义未公开，固定写 0。
/// 手动/自动分别来自 Clip 的 `fade_*_sec` 与 `auto_fade_*_sec`，
/// 与应用内"有效 fade"模型一一对应，不再像旧实现那样合并成一个手动值。
pub(crate) fn fade_values(
    shape: f64,
    dir: f64,
    manual_length_sec: f64,
    auto_length_sec: f64,
) -> Vec<f64> {
    let shape = if shape.is_finite() { shape } else { 0.0 };
    let mirror_base = shape.trunc().clamp(0.0, 255.0);
    vec![
        shape,
        manual_length_sec.max(0.0),
        auto_length_sec.max(0.0),
        mirror_base,
        if auto_length_sec > 1e-9 { 1.0 } else { 0.0 },
        dir.clamp(-1.0, 1.0),
        0.0,
    ]
}

fn source_bounds(take: &ClipTake, clip_length_sec: f64) -> (f64, f64) {
    // 正放：保留负 source_start（导出为负 SOFFS = REAPER 左延伸 item 的前导静音）。
    // 倒放走 SECTION 域（RPP 的 SECTION STARTPOS 不能为负），钳制到 ≥0。
    let start = if take.reversed {
        take.source_start_sec.max(0.0)
    } else {
        take.source_start_sec
    };
    let mut end = take.source_end_sec.max(start);
    if end <= start {
        end = take
            .duration_sec
            .filter(|duration| *duration > start)
            .unwrap_or_else(|| start + clip_length_sec * take.playback_rate.max(0.01) as f64);
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

fn audio_source(take: &ClipTake, _rate: f64, source_span_sec: f64) -> ReaperSource {
    let path = take.source_path.clone().unwrap_or_default();
    let (start, _) = source_bounds(take, 0.0);

    if take.reversed {
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

fn midi_source(take: &ClipTake, bpm: f64) -> ReaperSource {
    const PPQ: u32 = 960;

    let notes = take.midi_note_data.as_deref().unwrap_or(&[]);
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

fn fill_reaper_take(
    dest: &mut crate::reaper_parser::ReaperTake,
    take: &ClipTake,
    clip_length_sec: f64,
    bpm: f64,
    is_item_default: bool,
    output_playback_rate: f32,
) -> bool {
    let rate = output_playback_rate.max(0.01).min(100.0) as f64;
    dest.name = take.name.clone();
    // REAPER 的字段语义：
    // - Item 默认 take：`VOLPAN <item trim> <pan> <take volume> <pan law>`。
    // - 显式 take：`TAKEVOLPAN <pan> <take volume> <pan law>`，
    //   默认值为 `0 1 -1`；不能沿用 Item 默认 take 的四元组写法。
    dest.vol_pan = if is_item_default {
        vec![1.0, 0.0, take.gain as f64, -1.0]
    } else {
        vec![0.0, take.gain as f64, -1.0]
    };
    dest.play_rate = vec![rate, 1.0, 0.0, -1.0, 0.0, 0.0025];
    dest.chan_mode = 0;

    if let Some(ref midi_data) = take.midi_note_data {
        if midi_data.is_empty() {
            return false;
        }
        dest.s_offs = 0.0;
        dest.source = Some(midi_source(take, bpm));
        return true;
    }

    let source_path = take.source_path.as_deref().unwrap_or("").trim();
    if source_path.is_empty() {
        return false;
    }

    if take.reversed && take.loop_enabled {
        // 倒放 + Loop：回绕发生在整个媒体文件上（与引擎及正放 Loop 的
        // "循环原始音频文件"语义一致）。用覆盖全媒体的 SECTION 承载回绕域，
        // SOFFS 承载倒放相位锚点。
        let media_dur = take
            .duration_sec
            .filter(|d| d.is_finite() && *d > 1e-9)
            .or_else(|| {
                take.duration_frames
                    .zip(take.source_sample_rate)
                    .filter(|(frames, sr)| *sr > 0 && *frames > 0)
                    .map(|(frames, sr)| frames as f64 / sr as f64)
            });
        if let Some(d) = media_dur {
            let anchor = take.source_end_sec.min(d).rem_euclid(d);
            let mut source = ReaperSource::new();
            source.source_type = "SECTION".to_string();
            source.file_path = source_path.to_string();
            source.section_mode = 1;
            source.section_start_sec = Some(0.0);
            source.section_length_sec = Some(d);
            dest.s_offs = (d - anchor).rem_euclid(d);
            dest.source = Some(source);
            return true;
        }
        // 媒体时长未知：退化为下方通用反向路径（尽力而为）。
    }

    let (start, end) = source_bounds(take, clip_length_sec);
    let source_span = (end - start).max(0.0);

    let mut source = audio_source(take, rate, source_span);
    if take.reversed {
        // 反向：SECTION MODE 1 承载源窗口，SOFFS 置 0。
        dest.s_offs = 0.0;
    } else {
        // 正向：plain SOURCE + SOFFS 承载进入锚点（可为负 = 前导静音）。
        // Loop 的回绕发生在整个媒体文件上（REAPER 原生 Loop source 语义），
        // 无需 SECTION；非 Loop 超出媒体的 LENGTH 部分由 REAPER 渲染静音。
        source.section_start_sec = None;
        source.section_length_sec = None;
        dest.s_offs = start;
    }
    dest.source = Some(source);
    true
}

fn build_item(clip: &Clip, bpm: f64) -> Option<ReaperItem> {
    let mut working = clip.clone();
    working.normalize_takes();
    if working.takes.is_empty() {
        return None;
    }
    let active_idx = working.active_take_index().min(working.takes.len() - 1);

    let mut item = ReaperItem::default();
    item.position = working.start_sec.max(0.0);
    item.length = working.length_sec.max(0.001);
    // SnapOffset：相对 Clip 起点的偏移，与 REAPER SNAPOFFS 同语义直传。
    item.snap_offs = working.snap_offset_sec.max(0.0);
    // LOOP 是 REAPER 的 Item 级标志：多 take Clip 各 take 的 loop_enabled
    // 不同时，往返只能保留 active take 的值（导入端也按 item 级读取并共享
    // 给全部 take）。这是 RPP 格式的表达力边界，非实现遗漏。
    item.is_loop = working.takes[active_idx].loop_enabled;
    item.all_takes = false;
    // 手动长度与自动交叉淡化长度分槽写出（REAPER 索引 1 / 索引 2 +
    // selector），形状与曲率原样导出，保证 REAPER 渲染与应用一致。
    item.fade_in = fade_values(
        working.fade_in_shape,
        working.fade_in_dir,
        working.fade_in_sec,
        working.auto_fade_in_sec,
    );
    item.fade_out = fade_values(
        working.fade_out_shape,
        working.fade_out_dir,
        working.fade_out_sec,
        working.auto_fade_out_sec,
    );
    item.mute = vec![if working.muted { 1 } else { 0 }, 0];
    item.selected = false;

    let active_take = working.takes[active_idx].clone();
    // RPP 格式没有 Item 级 PLAYRATE 行：take 的 PLAYRATE 就是该 take 的
    // 有效播放速率（导入端也按此读取）。因此：
    // - default take（= active take）必须写 **组合有效速率**
    //   `clip_rate × take_rate` —— 与 develop 单窗口时代导出
    //   `clip.playback_rate` 的行为一致，否则带速率的 item 往返即丢速；
    // - 显式 take 同样写各自组合速率，保证在 REAPER 里切换 take 时
    //   听到的速度与 HiFiShifter 渲染一致。
    let clip_rate = if working.clip_playback_rate.is_finite() && working.clip_playback_rate > 1e-6 {
        working.clip_playback_rate
    } else {
        1.0
    };
    if !fill_reaper_take(
        &mut item.default_take,
        &active_take,
        working.length_sec,
        bpm,
        true,
        clip_rate * active_take.playback_rate,
    ) {
        return None;
    }

    // 显式 TAKE 块列出除 active 之外的全部 take（active 已由 default_take
    // 承载）。此前实现固定 skip(1)：当 active 不是第一个 take 时会把
    // active 重复写一份、而真正的第一个 take 被静默丢弃，再导入即得到
    // 错误的 take 集合。所有显式 take 均不打 SEL 标记 —— 导入端在无 SEL
    // 时回退到 default_take，恰好还原当前 active 选择。
    for (idx, take) in working.takes.iter().enumerate() {
        if idx == active_idx {
            continue;
        }
        let mut dest = crate::reaper_parser::ReaperTake::default();
        if fill_reaper_take(
            &mut dest,
            take,
            working.length_sec,
            bpm,
            false,
            clip_rate * take.playback_rate,
        ) {
            dest.selected = false;
            item.takes.push(dest);
        }
    }

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
        items_by_track.entry(track_index).or_default().push(item);
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

/// 测试辅助：按导入端约定从 SECTION take 还原倒放锚点（区间末端 − SOFFS）。
#[cfg(test)]
fn compute_anchor_from_section_for_test(take: &crate::reaper_parser::ReaperTake) -> f64 {
    let src = take.source.as_ref().expect("source present");
    let end = src.section_start_sec.unwrap_or(0.0)
        + src
            .section_length_sec
            .filter(|len| len.is_finite() && *len > 0.0)
            .unwrap_or(0.0);
    end - take.s_offs.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi_import::MidiNoteEvent;
    use crate::state::TimelineState;

    #[test]
    fn multi_take_export_uses_reaper_takevolpan_layout() {
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Multi Take".to_string()),
            Some(0.0),
            Some(2.0),
            Some("C:/audio/a.wav".to_string()),
        );
        if let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == clip_id) {
            // 直接写投影字段后必须 sync：takes 是磁盘/导出权威，
            // 否则 normalize_takes 会用陈旧 Take 覆盖这些值。
            let mut second = clip.active_take().clone();
            second.id = crate::state::new_id("take");
            second.name = "Second".to_string();
            second.source_path = Some("C:/audio/b.wav".to_string());
            clip.add_take(second);
            clip.gain = 0.75;
            clip.sync_take_from_flat();
        }

        let export = build_reaper_clipboard(&timeline, &[clip_id]).unwrap();
        let parsed = parse_for_test(&export.bytes);
        let item = &parsed.tracks[0].items[0];
        // Item 默认 take：VOLPAN <trim> <pan> <take volume> <pan law>。
        assert_eq!(item.default_take.vol_pan.len(), 4);
        assert!((item.default_take.vol_pan[2] - 0.75).abs() < 1e-9);
        // 显式 take：TAKEVOLPAN <pan> <take volume> <pan law>，默认音量 1。
        assert_eq!(item.takes.len(), 1);
        assert_eq!(item.takes[0].vol_pan.len(), 3);
        assert_eq!(item.takes[0].vol_pan[0], 0.0);
        assert_eq!(item.takes[0].vol_pan[1], 1.0);
        assert_eq!(item.takes[0].vol_pan[2], -1.0);
    }

    #[test]
    fn multi_take_export_preserves_rates_and_nonzero_active_index() {
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Rates".to_string()),
            Some(0.0),
            Some(2.0),
            Some("C:/audio/a.wav".to_string()),
        );
        if let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == clip_id) {
            // takes = [Rates(1.0), Second(1.0)]；随后把 Clip 级倍率设为 2.0，
            // 并给第二个 take 自身速率 1.5（组合有效速率 3.0），再把 active
            // 切到第二个 take —— 覆盖“active 非首位 + 非 1 速率”的导出路径。
            let mut second = clip.active_take().clone();
            second.id = crate::state::new_id("take");
            second.name = "Second".to_string();
            second.source_path = Some("C:/audio/b.wav".to_string());
            clip.add_take(second);
            clip.clip_playback_rate = 2.0;
            clip.playback_rate = 2.0;
            clip.sync_take_from_flat();
            let second_id = clip.takes[1].id.clone();
            clip.switch_active_take(&second_id)
                .expect("second take exists");
            clip.takes[1].playback_rate = 1.5;
        }

        let export = build_reaper_clipboard(&timeline, &[clip_id]).unwrap();
        let parsed = parse_for_test(&export.bytes);
        let item = &parsed.tracks[0].items[0];
        // active take（Second）作为 default take：PLAYRATE = clip 级 × take 自身。
        assert_eq!(item.default_take.name, "Second");
        assert!((item.default_take.play_rate[0] - 3.0).abs() < 1e-9);
        // 显式 TAKE 块只含非 active 的第一个 take，且不打 SEL 标记：
        // 导入端在无 SEL 时回退 default_take，恰好还原 active 选择。
        assert_eq!(item.takes.len(), 1);
        assert_eq!(item.takes[0].name, "Rates");
        assert!((item.takes[0].play_rate[0] - 2.0).abs() < 1e-9);
        assert!(!item.takes[0].selected);
    }

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
            // 本测试考察普通裁剪 Clip 的往返，须显式关闭 add_clip 的
            // 进程级默认 Loop；写完投影统一 sync 回 Take 权威数据。
            clip.loop_enabled = false;
            clip.sync_take_from_flat();
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
        assert_eq!(
            item.default_take.source.as_ref().unwrap().file_path,
            "C:/audio/test.wav"
        );
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
            // MIDI 内容写在投影上，须 sync 回 Take 权威数据供导出读取。
            clip.sync_take_from_flat();
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

    #[test]
    fn looping_clip_roundtrips_loop_flag_and_section_window() {
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Looped".to_string()),
            Some(0.0),
            Some(6.0),
            Some("C:/audio/loop.wav".to_string()),
        );
        if let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == clip_id) {
            clip.duration_sec = Some(4.0);
            // 循环窗口是媒体的一个子区间：导出必须用 SECTION 表达，
            // 否则 REAPER 会在媒体末尾而不是窗口末尾回绕。
            clip.source_start_sec = 1.0;
            clip.source_end_sec = 3.0;
            clip.playback_rate = 1.0;
            clip.loop_enabled = true;
            clip.sync_take_from_flat();
        }

        let export = build_reaper_clipboard(&timeline, &[clip_id.clone()]).unwrap();
        let parsed = parse_for_test(&export.bytes);
        let item = &parsed.tracks[0].items[0];
        assert!(item.is_loop, "loop flag must be exported");
        let source = item.default_take.source.as_ref().unwrap();
        // 正向 Loop = plain SOURCE + SOFFS（进入锚点）：
        // REAPER 原生 Loop source 在整个媒体上回绕，无需 SECTION。
        assert_eq!(source.source_type, "WAVE");
        assert!((item.default_take.s_offs - 1.0).abs() < 1e-9);

        // 非 Loop 的同窗口 clip 不应推断出 LOOP。
        {
            let clip = timeline
                .clips
                .iter_mut()
                .find(|clip| clip.id == clip_id)
                .expect("clip exists");
            clip.loop_enabled = false;
            clip.length_sec = 2.0;
            // 同步纪律：改写投影后必须写回 Take 权威数据。
            clip.sync_take_from_flat();
        }
        let export2 = build_reaper_clipboard(&timeline, &[clip_id.clone()]).unwrap();
        let parsed2 = parse_for_test(&export2.bytes);
        let item2 = &parsed2.tracks[0].items[0];
        assert!(!item2.is_loop, "short non-loop clip must not infer loop");
        assert_eq!(
            item2.default_take.source.as_ref().unwrap().source_type,
            "WAVE"
        );
    }

    #[test]
    fn negative_soffs_silence_tail_roundtrips() {
        // LENGTH 覆盖整个可见区间。导出必须逐字保留负 SOFFS 与 LOOP 0 ——
        // 不得把静音尾巴推断成 LOOP，也不得把 SOFFS 钳到 0。
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Left Extended".to_string()),
            Some(20.0),
            Some(16.81342267992402),
            Some("C:/audio/Vocal-1-3.wav".to_string()),
        );
        if let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == clip_id) {
            clip.duration_sec = Some(15.25232042998004);
            clip.source_start_sec = -1.56110224994398;
            clip.source_end_sec = 15.25232042998004;
            clip.playback_rate = 1.0;
            clip.loop_enabled = false;
            clip.sync_take_from_flat();
        }

        let export = build_reaper_clipboard(&timeline, &[clip_id]).unwrap();
        let parsed = parse_for_test(&export.bytes);
        let item = &parsed.tracks[0].items[0];
        assert!(
            (item.default_take.s_offs - (-1.56110224994398)).abs() < 1e-9,
            "negative SOFFS must survive export verbatim, got {}",
            item.default_take.s_offs
        );
        assert!(!item.is_loop, "silence-tail non-loop clip must stay LOOP 0");
        assert!((item.length - 16.81342267992402).abs() < 1e-9);
    }

    #[test]
    fn reversed_loop_clip_exports_whole_file_section_with_anchor() {
        // 倒放 + Loop：SECTION 必须覆盖整个媒体文件（回绕域），
        // SOFFS 承载倒放相位锚点：SOFFS = D − floor_mod(min(source_end, D), D)。
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Reversed Loop".to_string()),
            Some(0.0),
            Some(6.0),
            Some("C:/audio/loop.wav".to_string()),
        );
        if let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == clip_id) {
            clip.duration_sec = Some(4.0);
            clip.reversed = true;
            clip.loop_enabled = true;
            // split 可能产生的环绕窗口（start > end）：引擎只按
            // floor_mod(min(source_end, D), D) 取倒放锚点。
            clip.source_start_sec = 3.0;
            clip.source_end_sec = 1.0;
            clip.sync_take_from_flat();
        }

        let export = build_reaper_clipboard(&timeline, &[clip_id]).unwrap();
        let parsed = parse_for_test(&export.bytes);
        let item = &parsed.tracks[0].items[0];
        assert!(item.is_loop, "loop flag must be exported");
        let source = item.default_take.source.as_ref().unwrap();
        assert_eq!(
            source.section_mode, 1,
            "reversal must be expressed via SECTION MODE"
        );
        assert!(
            source.section_start_sec == Some(0.0) && source.section_length_sec == Some(4.0),
            "SECTION must cover the whole media file, got {:?}..{:?}",
            source.section_start_sec,
            source.section_length_sec
        );
        // φ = mod(min(1, 4), 4) = 1 ⇒ SOFFS = 4 − 1 = 3。
        assert!(
            (item.default_take.s_offs - 3.0).abs() < 1e-9,
            "reverse anchor phase must round-trip, got {}",
            item.default_take.s_offs
        );

        // 导入端按同一约定还原倒放锚点：anchor = 区间末端 − SOFFS = 4 − 3 = 1 = φ。
        let take = &item.default_take;
        let anchor = super::compute_anchor_from_section_for_test(take);
        assert!((anchor - 1.0).abs() < 1e-9);
    }
}
