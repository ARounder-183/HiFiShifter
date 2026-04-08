// MIDI 文件解析模块。
//
// 使用 midly crate 解析标准 MIDI 文件（.mid / .midi），
// 提取轨道信息和音符事件，用于导入到 pitch_edit。

use std::fs;
use std::path::Path;

use midly::{MetaMessage, MidiMessage, Smf, TrackEventKind};

/// 单个 MIDI 轨道的摘要信息
#[derive(Debug, Clone, serde::Serialize)]
pub struct MidiTrackInfo {
    /// 轨道索引（从 0 开始）
    pub index: usize,
    /// 轨道名称（从 Meta 事件中提取，可能为空）
    pub name: String,
    /// 该轨道中的音符数量
    pub note_count: usize,
    /// 最低音高 (MIDI note number)
    pub min_note: u8,
    /// 最高音高 (MIDI note number)
    pub max_note: u8,
}

/// 单个音符事件
#[derive(Debug, Clone, Copy)]
pub struct MidiNoteEvent {
    /// 起始时间（秒）
    pub start_sec: f64,
    /// 结束时间（秒）
    pub end_sec: f64,
    /// MIDI note number (0-127)
    pub note: u8,
    /// 力度 (0-127)
    #[allow(dead_code)]
    pub velocity: u8,
}

/// MIDI 文件解析结果
pub struct MidiParseResult {
    pub tracks: Vec<MidiTrackInfo>,
    /// 每个轨道的音符事件列表
    pub track_notes: Vec<Vec<MidiNoteEvent>>,
}

/// 解析 MIDI 文件，返回轨道信息和音符事件。
///
/// `fallback_bpm`：当 MIDI 文件不包含 Tempo 事件时，使用此值作为
/// 默认 BPM。传 `None` 则沿用 120 BPM 默认值。
pub fn parse_midi_file(path: &Path, fallback_bpm: Option<f64>) -> Result<MidiParseResult, String> {
    let data = fs::read(path).map_err(|e| format!("io_error: {}", e))?;
    parse_midi_data(&data, fallback_bpm)
}

/// 从字节数据解析 MIDI，返回轨道信息和音符事件。
///
/// `fallback_bpm`：当 MIDI 数据本身不包含 Tempo 事件时，使用此值作为
/// 默认 BPM（而非硬编码 120）。传 `None` 则沿用 120 BPM 默认值。
pub fn parse_midi_bytes(data: &[u8], fallback_bpm: Option<f64>) -> Result<MidiParseResult, String> {
    parse_midi_data(data, fallback_bpm)
}

fn parse_midi_data(data: &[u8], fallback_bpm: Option<f64>) -> Result<MidiParseResult, String> {
    let smf = Smf::parse(data).map_err(|e| format!("midi_parse_error: {}", e))?;

    // 解析 tempo map（用于将 tick 转换为秒）
    let ticks_per_beat = match smf.header.timing {
        midly::Timing::Metrical(tpb) => tpb.as_int() as f64,
        midly::Timing::Timecode(fps, sub) => {
            // 对于 SMPTE 时间码，按 fps * sub 转换
            let fps_val = match fps.as_int() {
                24 => 24.0,
                25 => 25.0,
                29 => 29.97,
                30 => 30.0,
                other => other as f64,
            };
            fps_val * sub as f64
        }
    };

    // 收集全局 tempo 事件（主要在第一个轨道中，但也可能散布在任何轨道）
    let mut tempo_events: Vec<(u64, f64)> = Vec::new(); // (abs_tick, microseconds_per_beat)

    // 对于 Format 0，所有事件都在第一个轨道中。
    // 对于 Format 1，tempo 事件通常在第一个轨道。
    for track in &smf.tracks {
        let mut abs_tick: u64 = 0;
        for event in track {
            abs_tick += event.delta.as_int() as u64;
            if let TrackEventKind::Meta(MetaMessage::Tempo(tempo)) = event.kind {
                tempo_events.push((abs_tick, tempo.as_int() as f64));
            }
        }
    }

    // 如果没有 tempo 事件，使用 fallback_bpm 或默认 120 BPM
    if tempo_events.is_empty() {
        let us_per_beat = match fallback_bpm {
            Some(bpm) if bpm > 0.0 && bpm.is_finite() => 60_000_000.0 / bpm,
            _ => 500_000.0, // 120 BPM
        };
        tempo_events.push((0, us_per_beat));
    }
    tempo_events.sort_by_key(|&(tick, _)| tick);

    let is_smpte = matches!(smf.header.timing, midly::Timing::Timecode(_, _));

    let track_count = smf.tracks.len();
    let mut all_tracks = Vec::with_capacity(track_count);
    let mut all_track_notes = Vec::with_capacity(track_count);

    for (track_idx, track) in smf.tracks.iter().enumerate() {
        let mut track_name = String::new();
        let mut notes: Vec<MidiNoteEvent> = Vec::new();
        // 记录正在发声的音符: 索引即 key -> (start_sec, velocity)
        let mut active_notes: [Option<(f64, u8)>; 128] = [None; 128];
        let mut abs_tick: u64 = 0;

        for event in track {
            abs_tick += event.delta.as_int() as u64;

            match event.kind {
                TrackEventKind::Meta(MetaMessage::TrackName(name_bytes)) => {
                    if track_name.is_empty() {
                        track_name = String::from_utf8_lossy(name_bytes).into_owned();
                    }
                }
                TrackEventKind::Midi { message, .. } => match message {
                    MidiMessage::NoteOn { key, vel } => {
                        let note = key.as_int();
                        let velocity = vel.as_int();
                        let current_sec =
                            tick_to_sec(abs_tick, ticks_per_beat, &tempo_events, is_smpte);

                        if velocity == 0 {
                            // NoteOn with velocity 0 等同于 NoteOff
                            if let Some((start_sec, start_vel)) = active_notes[note as usize].take()
                            {
                                notes.push(MidiNoteEvent {
                                    start_sec,
                                    end_sec: current_sec,
                                    note,
                                    velocity: start_vel,
                                });
                            }
                        } else {
                            // 如果已有同音高的音符在发声，先关闭它
                            if let Some((start_sec, start_vel)) = active_notes[note as usize].take()
                            {
                                notes.push(MidiNoteEvent {
                                    start_sec,
                                    end_sec: current_sec,
                                    note,
                                    velocity: start_vel,
                                });
                            }
                            active_notes[note as usize] = Some((current_sec, velocity));
                        }
                    }
                    MidiMessage::NoteOff { key, .. } => {
                        let note = key.as_int();
                        if let Some((start_sec, start_vel)) = active_notes[note as usize].take() {
                            let end_sec =
                                tick_to_sec(abs_tick, ticks_per_beat, &tempo_events, is_smpte);
                            notes.push(MidiNoteEvent {
                                start_sec,
                                end_sec,
                                note,
                                velocity: start_vel,
                            });
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        // 关闭所有未结束的音符（在轨道末尾）
        let end_sec = tick_to_sec(abs_tick, ticks_per_beat, &tempo_events, is_smpte);
        for (note_idx, note_data) in active_notes.iter().enumerate() {
            if let Some((start_sec, velocity)) = *note_data {
                notes.push(MidiNoteEvent {
                    start_sec,
                    end_sec,
                    note: note_idx as u8,
                    velocity,
                });
            }
        }

        // 按起始时间排序
        notes.sort_by(|a, b| {
            a.start_sec
                .partial_cmp(&b.start_sec)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let note_count = notes.len();
        let (min_note, max_note) = if notes.is_empty() {
            (0, 0)
        } else {
            notes.iter().fold((127, 0), |(curr_min, curr_max), n| {
                (curr_min.min(n.note), curr_max.max(n.note))
            })
        };

        all_tracks.push(MidiTrackInfo {
            index: track_idx,
            name: track_name,
            note_count,
            min_note,
            max_note,
        });
        all_track_notes.push(notes);
    }

    Ok(MidiParseResult {
        tracks: all_tracks,
        track_notes: all_track_notes,
    })
}

/// 将 MIDI 音符事件写入 pitch_edit 帧数组。
///
/// - `notes`: 要写入的音符事件列表（已按时间排序）
/// - `frame_period_ms`: 每帧的时间间隔（毫秒）
/// - `pitch_edit`: 目标 pitch_edit 帧数组（就地修改）
/// - `offset_sec`: 时间偏移量（秒）
///
/// 采用阶梯式写入：音符持续期间内所有帧直接设为该音符的 note number。
/// 音符之间的间隙保持原有值不变。
/// 重叠音符时取最高音。
///
/// 返回写入的帧数量。
pub fn write_notes_to_pitch_edit(
    notes: &[MidiNoteEvent],
    frame_period_ms: f64,
    pitch_edit: &mut [f32],
    offset_sec: f64,
) -> usize {
    // 避免后续除 0 或无效浮点引发崩溃
    if frame_period_ms <= 0.0 || !frame_period_ms.is_finite() {
        return 0;
    }

    let mut touched = 0usize;
    let total_frames = pitch_edit.len();

    for note in notes {
        let start_sec = note.start_sec + offset_sec;
        let end_sec = note.end_sec + offset_sec;

        if start_sec < 0.0 || !start_sec.is_finite() || !end_sec.is_finite() {
            continue;
        }

        let start_frame = ((start_sec * 1000.0) / frame_period_ms).round() as usize;
        let end_frame = ((end_sec * 1000.0) / frame_period_ms).round() as usize;

        if start_frame >= total_frames {
            continue;
        }

        let end_frame = end_frame.min(total_frames);
        let note_value = note.note as f32;

        for frame in start_frame..end_frame {
            // 重叠音符时取最高音
            let current = pitch_edit[frame];
            if note_value > current || current <= 0.0 {
                pitch_edit[frame] = note_value;
                touched += 1;
            }
        }
    }

    touched
}

/// 将 MIDI tick 转换为秒。
fn tick_to_sec(tick: u64, ticks_per_beat: f64, tempo_events: &[(u64, f64)], is_smpte: bool) -> f64 {
    if is_smpte {
        // SMPTE: tick 直接对应秒
        return tick as f64 / ticks_per_beat;
    }

    // Metrical timing: 需要根据 tempo map 分段计算
    let mut sec = 0.0;
    let mut last_tick: u64 = 0;
    let mut current_tempo: f64 = 500_000.0; // 默认 120 BPM

    for &(tempo_tick, tempo_us) in tempo_events {
        if tempo_tick >= tick {
            break;
        }
        // 从 last_tick 到 tempo_tick 之间的时间
        let delta_ticks = tempo_tick.saturating_sub(last_tick) as f64;
        sec += (delta_ticks / ticks_per_beat) * (current_tempo / 1_000_000.0);
        last_tick = tempo_tick;
        current_tempo = tempo_us;
    }

    // 从最后一个 tempo 变化点到目标 tick
    let delta_ticks = tick.saturating_sub(last_tick) as f64;
    sec += (delta_ticks / ticks_per_beat) * (current_tempo / 1_000_000.0);

    sec
}
