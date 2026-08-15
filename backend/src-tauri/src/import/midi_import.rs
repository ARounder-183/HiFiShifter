// MIDI 文件解析模块。
//
// 使用 midly crate 解析标准 MIDI 文件（.mid / .midi），
// 提取轨道信息和音符事件，用于导入到 pitch_edit。
//
// 弯音轮支持（per MIDI 1.0 Specification）：
// - Pitch Bend Change (EnH): 14-bit 分辨率，中心 8192 / 00H 40H，范围 0-16383
// - Pitch Bend Sensitivity: RPN 00 00，默认 ±2 半音，可通过 CC 101/100 → CC 6/38 调整
// - 弯音轮偏移在解析时直接烘焙到音符音高中：弯音轮事件会即时切分当前正在发声的音符，
//   使每段音符携带正确的弯音轮偏移后的音高值，无需额外参数存储。

use std::fs;
use std::path::Path;

use midly::{MetaMessage, MidiMessage, Smf, TrackEventKind};
use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MidiNoteEvent {
    /// 起始时间（秒）
    #[serde(alias = "startSec")]
    pub start_sec: f64,
    /// 结束时间（秒）
    #[serde(alias = "endSec")]
    pub end_sec: f64,
    /// MIDI note number (0.0-127.0)，已包含弯音轮偏移
    pub note: f32,
    /// 力度 (0-127)
    #[allow(dead_code)]
    #[serde(alias = "velocity")]
    pub velocity: u8,
    /// MIDI 通道 (0-15)
    #[serde(alias = "channel")]
    pub channel: u8,
}

/// MIDI 文件解析结果
pub struct MidiParseResult {
    pub tracks: Vec<MidiTrackInfo>,
    /// 每个轨道的音符事件列表
    pub track_notes: Vec<Vec<MidiNoteEvent>>,
    /// MIDI 初始 BPM（第一个 Tempo 事件的 BPM，或回退默认值）
    pub initial_bpm: f64,
    /// MIDI 文件是否包含实际的 Tempo 事件（非回退值）
    pub has_tempo: bool,
    /// 全局 Tempo 事件：(abs_tick, 微秒/拍)。
    pub tempo_events: Vec<(u64, f64)>,
    /// 全局拍号事件：(abs_tick, 分子, 分母)。
    pub time_signature_events: Vec<(u64, u32, u32)>,
    /// 全局音阶（调号）事件：(abs_tick, 升降号数 -7..=7)。
    pub key_signature_events: Vec<(u64, i8)>,
    /// 每四分音符 tick 数（SMTPE 时为每秒 tick 数）。
    pub ticks_per_beat: f64,
    pub is_smpte: bool,
}

/// 将弯音轮原始值和弯音灵敏度转换为半音偏移量。
#[inline]
fn raw_pb_to_semitones(raw: i16, range_semitones: f32) -> f32 {
    (raw as f32 - 8192.0) / 8192.0 * range_semitones
}

/// 在弯音轮事件发生时，切分当前通道上所有正在发声的音符。
///
/// 对于每个在该通道上正在发声的音符，关闭当前段（用旧的弯音轮偏移写入音高），
/// 并立即以当前时间开启新段。这样弯音轮的连续变化被正确地反映到音符数据中。
fn split_active_notes_on_channel(
    active_notes: &mut [Option<(f64, u8, u8)>; 128],
    notes: &mut Vec<MidiNoteEvent>,
    channel: u8,
    split_time_sec: f64,
    channel_pb: &[i16; 16],
    channel_bend_range: &[f32; 16],
) {
    for note_idx in 0..128u8 {
        if let Some((start_sec, velocity, note_ch)) = active_notes[note_idx as usize] {
            if note_ch == channel && start_sec < split_time_sec {
                let pb_semitones =
                    raw_pb_to_semitones(channel_pb[channel as usize], channel_bend_range[channel as usize]);
                let adjusted_note = (note_idx as f32 + pb_semitones).clamp(0.0, 127.0);
                notes.push(MidiNoteEvent {
                    start_sec,
                    end_sec: split_time_sec,
                    note: adjusted_note,
                    velocity,
                    channel: note_ch,
                });
                // 开启新段（从当前时间继续）
                active_notes[note_idx as usize] = Some((split_time_sec, velocity, note_ch));
            }
        }
    }
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

    // 收集全局 tempo 事件
    let mut tempo_events: Vec<(u64, f64)> = Vec::new(); // (abs_tick, microseconds_per_beat)
    for track in &smf.tracks {
        let mut abs_tick: u64 = 0;
        for event in track {
            abs_tick += event.delta.as_int() as u64;
            if let TrackEventKind::Meta(MetaMessage::Tempo(tempo)) = event.kind {
                tempo_events.push((abs_tick, tempo.as_int() as f64));
            }
        }
    }

    // 收集全局拍号事件（FF 58 04 nn dd cc bb）。
    // 注意：MIDI 规格中 dd 是“2 的指数”（2^dd = 以几分音符为一拍），
    // 例：dd=2 → 四分音符（分母 4）。midly 的 TimeSignature 第二个字段
    // 就是原始指数字节，需要转换成实际分母值。
    let mut time_signature_events: Vec<(u64, u32, u32)> = Vec::new();
    // 收集全局音阶/调号事件（FF 59 02 sf mi）
    let mut key_signature_events: Vec<(u64, i8)> = Vec::new();
    for track in &smf.tracks {
        let mut abs_tick: u64 = 0;
        for event in track {
            abs_tick += event.delta.as_int() as u64;
            match event.kind {
                TrackEventKind::Meta(MetaMessage::TimeSignature(num, den_pow2, _cc, _bb)) => {
                    let denominator = (2u32)
                        .checked_pow(den_pow2 as u32)
                        .filter(|d| matches!(d, 1 | 2 | 4 | 8 | 16 | 32))
                        .unwrap_or(4);
                    time_signature_events.push((abs_tick, num as u32, denominator));
                }
                TrackEventKind::Meta(MetaMessage::KeySignature(sf, _mi)) => {
                    key_signature_events.push((abs_tick, sf));
                }
                _ => {}
            }
        }
    }
    time_signature_events.sort_by_key(|&(tick, _, _)| tick);
    key_signature_events.sort_by_key(|&(tick, _)| tick);

    let has_tempo = !tempo_events.is_empty();

    // 如果没有 tempo 事件，使用 fallback_bpm 或默认 120 BPM
    if !has_tempo {
        let us_per_beat = match fallback_bpm {
            Some(bpm) if bpm > 0.0 && bpm.is_finite() => 60_000_000.0 / bpm,
            _ => 500_000.0, // 120 BPM
        };
        tempo_events.push((0, us_per_beat));
    }
    tempo_events.sort_by_key(|&(tick, _)| tick);

    // 提取初始 BPM
    let initial_bpm = {
        let first_us = tempo_events.first().map(|&(_, us)| us).unwrap_or(500_000.0);
        if first_us > 0.0 && first_us.is_finite() {
            60_000_000.0 / first_us
        } else {
            120.0
        }
    };

    let is_smpte = matches!(smf.header.timing, midly::Timing::Timecode(_, _));

    let track_count = smf.tracks.len();
    let mut all_tracks = Vec::with_capacity(track_count);
    let mut all_track_notes = Vec::with_capacity(track_count);

    for (track_idx, track) in smf.tracks.iter().enumerate() {
        let mut track_name = String::new();
        let mut notes: Vec<MidiNoteEvent> = Vec::new();

        // 记录正在发声的音符: 索引即 key -> (start_sec, velocity, channel)
        let mut active_notes: [Option<(f64, u8, u8)>; 128] = [None; 128];
        let mut abs_tick: u64 = 0;

        // ── 每通道弯音轮状态 ──
        // 当前弯音轮原始值 (0-16383, 8192=中心)
        let mut channel_pb: [i16; 16] = [8192; 16];
        // 每通道弯音灵敏度（半音），默认 ±2 半音，可通过 RPN 00 00 修改
        let mut channel_bend_range: [f32; 16] = [2.0; 16];
        // RPN 参数号选择状态
        let mut rpn_msb: [u8; 16] = [0x7F; 16];
        let mut rpn_lsb: [u8; 16] = [0x7F; 16];
        // 暂存的 Data Entry LSB（cents 部分）
        let mut pending_bend_range_cents: [Option<f32>; 16] = [None; 16];

        for event in track {
            abs_tick += event.delta.as_int() as u64;

            match event.kind {
                TrackEventKind::Meta(MetaMessage::TrackName(name_bytes)) => {
                    if track_name.is_empty() {
                        track_name = String::from_utf8_lossy(name_bytes).into_owned();
                    }
                }
                TrackEventKind::Midi { channel, message } => {
                    let ch = channel.as_int();
                    match message {
                        MidiMessage::NoteOn { key, vel } => {
                            let raw_note = key.as_int() as f32;
                            let velocity = vel.as_int();
                            let current_sec =
                                tick_to_sec(abs_tick, ticks_per_beat, &tempo_events, is_smpte);

                            if velocity == 0 {
                                // NoteOn with velocity 0 等同于 NoteOff
                                if let Some((start_sec, start_vel, note_ch)) =
                                    active_notes[raw_note as usize].take()
                                {
                                    let pb_semitones = raw_pb_to_semitones(
                                        channel_pb[note_ch as usize],
                                        channel_bend_range[note_ch as usize],
                                    );
                                    let adjusted_note =
                                        (raw_note + pb_semitones).clamp(0.0, 127.0);
                                    notes.push(MidiNoteEvent {
                                        start_sec,
                                        end_sec: current_sec,
                                        note: adjusted_note,
                                        velocity: start_vel,
                                        channel: note_ch,
                                    });
                                }
                            } else {
                                // 如果已有同音高的音符在发声，先关闭它
                                if let Some((start_sec, start_vel, prev_ch)) =
                                    active_notes[raw_note as usize].take()
                                {
                                    let pb_semitones = raw_pb_to_semitones(
                                        channel_pb[prev_ch as usize],
                                        channel_bend_range[prev_ch as usize],
                                    );
                                    let adjusted_note =
                                        (raw_note + pb_semitones).clamp(0.0, 127.0);
                                    notes.push(MidiNoteEvent {
                                        start_sec,
                                        end_sec: current_sec,
                                        note: adjusted_note,
                                        velocity: start_vel,
                                        channel: prev_ch,
                                    });
                                }
                                active_notes[raw_note as usize] =
                                    Some((current_sec, velocity, ch));
                            }
                        }
                        MidiMessage::NoteOff { key, .. } => {
                            let note = key.as_int();
                            if let Some((start_sec, start_vel, note_ch)) =
                                active_notes[note as usize].take()
                            {
                                let end_sec = tick_to_sec(
                                    abs_tick, ticks_per_beat, &tempo_events, is_smpte,
                                );
                                let pb_semitones = raw_pb_to_semitones(
                                    channel_pb[note_ch as usize],
                                    channel_bend_range[note_ch as usize],
                                );
                                let adjusted_note =
                                    (note as f32 + pb_semitones).clamp(0.0, 127.0);
                                notes.push(MidiNoteEvent {
                                    start_sec,
                                    end_sec,
                                    note: adjusted_note,
                                    velocity: start_vel,
                                    channel: note_ch,
                                });
                            }
                        }
                        MidiMessage::PitchBend { bend } => {
                            let current_sec =
                                tick_to_sec(abs_tick, ticks_per_beat, &tempo_events, is_smpte);
                            // 切分当前通道上所有正在发声的音符（让已发声的部分用旧弯音值关闭，
                            // 并在当前时间开启新段）
                            split_active_notes_on_channel(
                                &mut active_notes,
                                &mut notes,
                                ch,
                                current_sec,
                                &channel_pb,
                                &channel_bend_range,
                            );
                            // 更新弯音轮值
                            let raw = bend.0.as_int() as i16;
                            channel_pb[ch as usize] = raw;
                        }
                        MidiMessage::Controller { controller, value } => {
                            let ctrl = controller.as_int();
                            let val = value.as_int();
                            match ctrl {
                                // RPN MSB (CC 101)
                                101 => {
                                    rpn_msb[ch as usize] = val;
                                }
                                // RPN LSB (CC 100)
                                100 => {
                                    rpn_lsb[ch as usize] = val;
                                }
                                // Data Entry MSB (CC 6)
                                6 => {
                                    if rpn_msb[ch as usize] == 0 && rpn_lsb[ch as usize] == 0 {
                                        let semitones = val as f32;
                                        let cents =
                                            pending_bend_range_cents[ch as usize].unwrap_or(0.0);
                                        channel_bend_range[ch as usize] =
                                            (semitones + cents / 100.0).max(0.0);
                                        pending_bend_range_cents[ch as usize] = None;
                                    }
                                }
                                // Data Entry LSB (CC 38)
                                38 => {
                                    if rpn_msb[ch as usize] == 0 && rpn_lsb[ch as usize] == 0 {
                                        pending_bend_range_cents[ch as usize] = Some(val as f32);
                                        let current_msb =
                                            channel_bend_range[ch as usize].trunc();
                                        channel_bend_range[ch as usize] =
                                            (current_msb + val as f32 / 100.0).max(0.0);
                                    }
                                }
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        // 关闭所有未结束的音符（在轨道末尾）
        let end_sec = tick_to_sec(abs_tick, ticks_per_beat, &tempo_events, is_smpte);
        for (note_idx, note_data) in active_notes.iter().enumerate() {
            if let Some((start_sec, velocity, ch)) = *note_data {
                let pb_semitones =
                    raw_pb_to_semitones(channel_pb[ch as usize], channel_bend_range[ch as usize]);
                let adjusted_note = (note_idx as f32 + pb_semitones).clamp(0.0, 127.0);
                notes.push(MidiNoteEvent {
                    start_sec,
                    end_sec,
                    note: adjusted_note,
                    velocity,
                    channel: ch,
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
            (0.0f32, 0.0f32)
        } else {
            notes.iter().fold((127.0f32, 0.0f32), |(curr_min, curr_max), n| {
                (curr_min.min(n.note), curr_max.max(n.note))
            })
        };

        all_tracks.push(MidiTrackInfo {
            index: track_idx,
            name: track_name,
            note_count,
            min_note: min_note as u8,
            max_note: max_note as u8,
        });
        all_track_notes.push(notes);
    }

    Ok(MidiParseResult {
        tracks: all_tracks,
        track_notes: all_track_notes,
        initial_bpm,
        has_tempo,
        tempo_events,
        time_signature_events,
        key_signature_events,
        ticks_per_beat,
        is_smpte,
    })
}

/// 将 MIDI 调号（升降号数 -7..=7）映射为 HiFiShifter 内置音阶键名。
fn key_signature_to_scale_key(sf: i8) -> Option<String> {
    let key = match sf {
        0 => "C",
        1 => "G",
        2 => "D",
        3 => "A",
        4 => "E",
        5 => "B",
        6 => "Gb", // F# → Gb
        7 => "Db", // C# → Db
        -1 => "F",
        -2 => "Bb",
        -3 => "Eb",
        -4 => "Ab",
        -5 => "Db",
        -6 => "Gb",
        -7 => "B", // Cb → B
        _ => return None,
    };
    Some(key.to_string())
}

/// 由 MIDI 事件构建 Tempo Map 变化点列表（时间锚定，秒）。
///
/// - `import_tempo` / `import_time_signature` / `import_key_signature` 控制导入哪些参数；
/// - 事件时间用 MIDI 自身的 Tempo 事件积分换算为秒；
/// - 返回的列表已按位置升序，第一个点位于 0（始终存在）；
/// - 仅当存在“0 之后的实际变化”时返回 Some（否则 None，交由调用方按工程基准处理）。
#[allow(clippy::too_many_arguments)]
pub fn build_tempo_map_points_from_midi(
    result: &MidiParseResult,
    import_tempo: bool,
    import_time_signature: bool,
    import_key_signature: bool,
    fallback_bpm: f64,
    fallback_beats_per_bar: u32,
    fallback_denominator: u32,
) -> Option<Vec<crate::state::TempoPointData>> {
    // 收集事件（位置为秒；跳过被关闭的导入类型）。
    struct Event {
        sec: f64,
        bpm: Option<f64>,
        numerator: Option<u32>,
        denominator: Option<u32>,
        scale_key: Option<String>,
    }
    let to_sec = |tick: u64| -> f64 {
        tick_to_sec(tick, result.ticks_per_beat, &result.tempo_events, result.is_smpte)
    };

    let mut events: Vec<Event> = Vec::new();
    if import_tempo {
        for &(tick, us_per_beat) in &result.tempo_events {
            if us_per_beat > 0.0 && us_per_beat.is_finite() {
                events.push(Event {
                    sec: to_sec(tick),
                    bpm: Some(60_000_000.0 / us_per_beat),
                    numerator: None,
                    denominator: None,
                    scale_key: None,
                });
            }
        }
    }
    if import_time_signature {
        for &(tick, num, den) in &result.time_signature_events {
            events.push(Event {
                sec: to_sec(tick),
                bpm: None,
                numerator: Some(num.clamp(1, 32)),
                denominator: Some(if matches!(den, 1 | 2 | 4 | 8 | 16 | 32) {
                    den
                } else {
                    4
                }),
                scale_key: None,
            });
        }
    }
    if import_key_signature {
        for &(tick, sf) in &result.key_signature_events {
            if let Some(key) = key_signature_to_scale_key(sf) {
                events.push(Event {
                    sec: to_sec(tick),
                    bpm: None,
                    numerator: None,
                    denominator: None,
                    scale_key: Some(key),
                });
            }
        }
    }
    events.sort_by(|a, b| a.sec.partial_cmp(&b.sec).unwrap_or(std::cmp::Ordering::Equal));

    // 逐事件合并到变化点（同一位置合并多个参数）。
    let mut merged: Vec<Event> = Vec::new();
    for event in events {
        let sec = event.sec.max(0.0);
        if let Some(last) = merged.last_mut() {
            if (last.sec - sec).abs() < 1e-6 {
                last.bpm = event.bpm.or(last.bpm);
                last.numerator = event.numerator.or(last.numerator);
                last.denominator = event.denominator.or(last.denominator);
                last.scale_key = event.scale_key.or_else(|| last.scale_key.clone());
                continue;
            }
        }
        merged.push(event);
    }

    if merged.is_empty() {
        return None;
    }

    // 计算每个位置生效的参数（向前继承；无前值用 fallback）。
    let mut current_bpm = fallback_bpm.clamp(10.0, 960.0);
    let mut current_num = fallback_beats_per_bar.clamp(1, 32);
    let mut current_den: u32 = if matches!(fallback_denominator, 1 | 2 | 4 | 8 | 16 | 32) {
        fallback_denominator
    } else {
        4
    };

    let mut points: Vec<crate::state::TempoPointData> = Vec::new();
    let mut index = 0usize;
    for event in &merged {
        if let Some(bpm) = event.bpm {
            current_bpm = bpm.clamp(10.0, 960.0);
        }
        if let Some(num) = event.numerator {
            current_num = num;
        }
        if let Some(den) = event.denominator {
            current_den = den;
        }
        points.push(crate::state::TempoPointData {
            id: format!("midi_tp_{index}"),
            position_sec: event.sec,
            bpm: current_bpm,
            numerator: Some(current_num),
            denominator: Some(current_den),
            scale: event.scale_key.as_ref().map(|key| crate::state::TempoScaleData {
                key: Some(key.clone()),
                name: None,
                notes: None,
            }),
        });
        index += 1;
    }

    // 确保 0 位置点存在。
    if points.first().map(|p| p.position_sec > 1e-9).unwrap_or(false) {
        points.insert(
            0,
            crate::state::TempoPointData {
                id: "midi_tp_0".to_string(),
                position_sec: 0.0,
                bpm: fallback_bpm.clamp(10.0, 960.0),
                numerator: Some(fallback_beats_per_bar.clamp(1, 32)),
                denominator: Some(if matches!(fallback_denominator, 1 | 2 | 4 | 8 | 16 | 32) {
                    fallback_denominator
                } else {
                    4
                }),
                scale: None,
            },
        );
    }

    // 始终返回列表（至少包含 0 位置点）；调用方判断是否存在“0 之后的实际变化”。
    Some(points)
}

/// 在写入 MIDI 音符之前，清除 pitch_edit 中将被音符覆盖的帧范围。
///
/// 这确保已有的 pitch 编辑不会阻止新导入的 MIDI 音符（例如旧的高音不会阻挡新的低音）。
/// "最高音优先"规则仍然适用于同一批次内重叠的音符。
pub fn clear_pitch_edit_range_for_notes(
    notes: &[MidiNoteEvent],
    frame_period_ms: f64,
    pitch_edit: &mut [f32],
    offset_sec: f64,
) {
    if frame_period_ms <= 0.0 || !frame_period_ms.is_finite() || notes.is_empty() {
        return;
    }
    let total_frames = pitch_edit.len();
    let mut min_frame = usize::MAX;
    let mut max_frame = 0usize;

    for note in notes {
        let start_sec = note.start_sec + offset_sec;
        let end_sec = note.end_sec + offset_sec;
        if start_sec < 0.0 || !start_sec.is_finite() || !end_sec.is_finite() {
            continue;
        }
        let sf = ((start_sec * 1000.0) / frame_period_ms).round() as usize;
        let ef = (((end_sec * 1000.0) / frame_period_ms).round() as usize).min(total_frames);
        if sf < total_frames {
            min_frame = min_frame.min(sf);
            max_frame = max_frame.max(ef);
        }
    }

    if min_frame < max_frame && max_frame <= total_frames {
        for frame in min_frame..max_frame {
            pitch_edit[frame] = 0.0;
        }
    }
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
/// 弯音轮偏移已在解析阶段直接写入 note 值中。
///
/// 返回写入的帧数量。
pub fn write_notes_to_pitch_edit(
    notes: &[MidiNoteEvent],
    frame_period_ms: f64,
    pitch_edit: &mut [f32],
    offset_sec: f64,
) -> usize {
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
        let note_value = note.note;

        for frame in start_frame..end_frame {
            let current = pitch_edit[frame];
            if note_value > current || current <= 0.0 {
                pitch_edit[frame] = note_value;
                touched += 1;
            }
        }
    }

    touched
}

/// 填充 pitch_edit 中音符之间的空隙。
///
/// 从第一个非零帧到最后一个非零帧，将值为 0 的帧用前一个非零值填充。
/// 不填充第一个音符之前和最后一个音符之后的区域。
/// 返回填充的帧数量。
pub fn fill_gaps_in_pitch_edit(pitch_edit: &mut [f32]) -> usize {
    let total_frames = pitch_edit.len();
    if total_frames == 0 {
        return 0;
    }

    // 找到第一个非零帧
    let first_nonzero = match pitch_edit.iter().position(|&v| v > 0.0) {
        Some(pos) => pos,
        None => return 0,
    };

    // 找到最后一个非零帧
    let last_nonzero = match pitch_edit.iter().rposition(|&v| v > 0.0) {
        Some(pos) => pos,
        None => return 0,
    };

    if first_nonzero >= last_nonzero {
        return 0;
    }

    let mut filled = 0usize;
    let mut last_pitch: f32 = 0.0;

    for frame in first_nonzero..=last_nonzero {
        let current = pitch_edit[frame];
        if current > 0.0 {
            last_pitch = current;
        } else if last_pitch > 0.0 {
            pitch_edit[frame] = last_pitch;
            filled += 1;
        }
    }

    filled
}

/// 将 MIDI tick 转换为秒。
fn tick_to_sec(tick: u64, ticks_per_beat: f64, tempo_events: &[(u64, f64)], is_smpte: bool) -> f64 {
    if is_smpte {
        return tick as f64 / ticks_per_beat;
    }

    let mut sec = 0.0;
    let mut last_tick: u64 = 0;
    let mut current_tempo: f64 = 500_000.0; // 默认 120 BPM

    for &(tempo_tick, tempo_us) in tempo_events {
        if tempo_tick >= tick {
            break;
        }
        let delta_ticks = tempo_tick.saturating_sub(last_tick) as f64;
        sec += (delta_ticks / ticks_per_beat) * (current_tempo / 1_000_000.0);
        last_tick = tempo_tick;
        current_tempo = tempo_us;
    }

    let delta_ticks = tick.saturating_sub(last_tick) as f64;
    sec += (delta_ticks / ticks_per_beat) * (current_tempo / 1_000_000.0);

    sec
}

/// 将 MIDI 轨道的音符按音高拆分为不重叠的组。
///
/// 检测在时间上重叠的音符，将其拆分到不同的组中，
/// 使得每个组内部的音符在时间轴上互不重叠。
/// 音高高的音符优先分配到编号较小的组。
///
/// 返回一个 Vec，每个元素是一组不重叠的音符。
pub fn split_notes_into_non_overlapping_groups(
    notes: &[MidiNoteEvent],
) -> Vec<Vec<MidiNoteEvent>> {
    if notes.is_empty() {
        return vec![];
    }

    // 按起始时间排序，起始时间相同时按音高降序（高音在前）
    let mut sorted: Vec<&MidiNoteEvent> = notes.iter().collect();
    sorted.sort_by(|a, b| {
        a.start_sec
            .partial_cmp(&b.start_sec)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.note.partial_cmp(&a.note).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut groups: Vec<Vec<MidiNoteEvent>> = vec![];
    let mut group_end_times: Vec<f64> = vec![];

    for note in sorted {
        let mut placed = false;
        // 尝试放入已有的组（不重叠即可放入）
        for (gi, &end_time) in group_end_times.iter().enumerate() {
            if note.start_sec >= end_time - 1e-9 {
                groups[gi].push(*note);
                group_end_times[gi] = group_end_times[gi].max(note.end_sec);
                placed = true;
                break;
            }
        }
        if !placed {
            // 创建新组
            groups.push(vec![*note]);
            group_end_times.push(note.end_sec);
        }
    }

    groups
}
