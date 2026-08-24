use crate::pitch_clip;
use crate::pitch_editing::transpose_midi_by_scale_steps;
use crate::state::{AppState, TimelineState};

use midly::{Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind};
use serde::Deserialize;
use std::fs;
use std::io::Write;

// ── 请求数据结构 ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct MidiExportTrackEntry {
    #[serde(alias = "trackId")]
    pub track_id: String,
    #[serde(alias = "rootTrackId")]
    pub root_track_id: String,
    pub name: String,
    #[serde(alias = "startSec")]
    pub start_sec: f64,
    #[serde(alias = "endSec")]
    pub end_sec: f64,
    /// 当 track 级音高数据不可用时的 fallback clip ID（非 Compose 轨道）
    #[serde(alias = "clipId", default)]
    pub clip_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MidiExportRequest {
    #[serde(alias = "outputPath")]
    pub output_path: String,
    pub tracks: Vec<MidiExportTrackEntry>,
    pub bpm: f64,
    #[serde(alias = "beatsPerBar")]
    pub beats_per_bar: u32,
    #[serde(alias = "baseScale")]
    pub base_scale: String,
    #[serde(alias = "projectScaleNotes")]
    #[allow(dead_code)]
    pub project_scale_notes: Vec<u8>,
}

// ── 内部结构 ──────────────────────────────────────────────────────────────────

const TICKS_PER_BEAT: u16 = 480;
const DEFAULT_VELOCITY: u8 = 100;
const PITCH_BEND_RANGE_CENTS: f32 = 200.0; // ±2 半音 = ±200 音分
const GM_LEAD_1_SQUARE: u8 = 80; // General MIDI Program: Lead 1 (Square)
const CHILD_CENTS_PREFIX: &str = "child_pitch_offset_cents@";
const CHILD_DEGREES_PREFIX: &str = "child_pitch_offset_degrees@";

fn delta_u28(t: u64) -> midly::num::u28 {
    midly::num::u28::new(t.min(0x0FFF_FFFF) as u32)
}

// ── 通道分配 ──────────────────────────────────────────────────────────────────

fn assign_channels(track_count: usize) -> Vec<u8> {
    const MELODIC: [u8; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15];
    (0..track_count)
        .map(|i| MELODIC[i % MELODIC.len()])
        .collect()
}

// ── Scale → KeySignature ─────────────────────────────────────────────────────

fn scale_to_key_signature(base_scale: &str) -> (i8, bool) {
    match base_scale {
        "C" => (0, false),
        "G" => (1, false),
        "D" => (2, false),
        "A" => (3, false),
        "E" => (4, false),
        "B" => (5, false),
        "F#" => (6, false),
        "C#" => (7, false),
        "F" => (-1, false),
        "Bb" => (-2, false),
        "Eb" => (-3, false),
        "Ab" => (-4, false),
        "Db" => (-5, false),
        "Gb" => (-6, false),
        "Cb" => (-7, false),
        "Am" => (0, true),
        "Em" => (1, true),
        "Bm" => (2, true),
        "F#m" => (3, true),
        "C#m" => (4, true),
        "G#m" => (5, true),
        "D#m" => (6, true),
        "A#m" => (7, true),
        "Dm" => (-1, true),
        "Gm" => (-2, true),
        "Cm" => (-3, true),
        "Fm" => (-4, true),
        "Bbm" => (-5, true),
        "Ebm" => (-6, true),
        "Abm" => (-7, true),
        _ => (0, false),
    }
}

// ── 音分 → 弯音轮 raw 值 ─────────────────────────────────────────────────────

fn cents_to_bend_raw(cents_diff: f32) -> u16 {
    let raw = (cents_diff / PITCH_BEND_RANGE_CENTS * 8192.0 + 8192.0).round() as i32;
    raw.clamp(0, 16383) as u16
}

// ── Pitch 曲线 → MIDI 轨道事件（滑音合并 + 弯音轮） ──────────────────────────

/// 弯音轮覆盖范围（半音数）。
const MAX_NOTE_SPAN_SEMITONES: f32 = 1.0;
/// 最小音符时长（毫秒），短于此值的音符会被丢弃。
const MIN_NOTE_DURATION_MS: f64 = 100.0;
/// 平滑窗口大小（帧数），用于减少逐帧抖动。
const SMOOTH_WINDOW: usize = 3;
/// 稳定性检测：音分阈值。|pitch - round(pitch)| * 100 < 此值 → 视为稳定。
const STABLE_CENTS_THRESHOLD: f32 = 10.0;
/// 稳定性检测：需要连续多少帧稳定才触发切分。
const STABILITY_FRAMES: usize = 8;

/// 将一帧音高数据转为 MIDI 轨道事件。
///
/// 算法（弯音轮滑音合并）：
/// - 先用移动平均平滑曲线，减少逐帧抖动
/// - 扫描有声音段，跟踪当前候选音符内的 min/max 音高
/// - 当音高跨度超过 MAX_NOTE_SPAN_SEMITONES（4 半音）时，关闭当前候选并开始新候选
/// - 当音高在某个半音附近稳定（|cents| < 15 且连续 ≥ STABILITY_FRAMES 帧）且该半音
///   与当前候选的基准半音不同时，关闭候选并开始新候选——保证稳定音高处产生干净音符
/// - 每个候选音符的基准半音 = round((min+max)/2)，弯音轮按帧跟随实际音高
/// - 丢弃短于 MIN_NOTE_DURATION_MS 的音符
fn pitch_curve_to_track_events(
    pitch_values: &[f32],
    frame_period_ms: f64,
    start_sec: f64,
    channel: u8,
    tick_converter: &TempoTickConverter,
) -> Vec<TrackEvent<'static>> {
    if pitch_values.is_empty() {
        return Vec::new();
    }

    let fp = frame_period_ms.max(1e-6);
    let n = pitch_values.len();

    // ── Phase 1: 平滑 ──────────────────────────────────────────────────────
    let smoothed = smooth_curve(pitch_values, SMOOTH_WINDOW);

    // ── Phase 2: 构建音符候选 ──────────────────────────────────────────────
    struct NoteCandidate {
        start_idx: usize,
        end_idx: usize,   // exclusive
        min_pitch: f32,
        max_pitch: f32,
    }

    let mut candidates: Vec<NoteCandidate> = Vec::new();
    let mut i: usize = 0;

    while i < n {
        // 跳过静音帧
        let p = smoothed[i];
        if !p.is_finite() || p <= 0.0 {
            i += 1;
            continue;
        }

        // 开始一个新的候选
        let cand_start = i;
        let mut cand_min = p;
        let mut cand_max = p;
        let mut stable_count: usize = 0;
        let mut stable_semi: Option<u8> = None;

        i += 1;
        while i < n {
            let cur = smoothed[i];
            if !cur.is_finite() || cur <= 0.0 {
                // 静音 → 结束候选
                break;
            }

            // 检查音高跨度
            let new_min = cand_min.min(cur);
            let new_max = cand_max.max(cur);
            if new_max - new_min > MAX_NOTE_SPAN_SEMITONES {
                // 跨度超过弯音轮范围 → 结束候选
                break;
            }

            // 稳定性检测
            let cur_semi = cur.round();
            let cur_cents = (cur - cur_semi).abs() * 100.0;
            if cur_cents < STABLE_CENTS_THRESHOLD {
                let cur_semi_u8 = (cur_semi as i32).clamp(0, 127) as u8;
                if stable_semi == Some(cur_semi_u8) {
                    stable_count += 1;
                    if stable_count >= STABILITY_FRAMES {
                        // 检查稳定半音是否与当前候选的基准不同
                        let current_base = ((cand_min + cand_max) / 2.0).round();
                        if (cur_semi - current_base).abs() >= 1.0 {
                            // 在新半音处稳定 → 回到稳定开始帧，结束候选
                            let rollback = stable_count.saturating_sub(1);
                            i = i.saturating_sub(rollback);
                            break;
                        }
                    }
                } else {
                    stable_semi = Some(cur_semi_u8);
                    stable_count = 1;
                }
            } else {
                stable_count = 0;
                stable_semi = None;
            }

            cand_min = new_min;
            cand_max = new_max;
            i += 1;
        }

        let end_idx = i;
        let dur_ms = (end_idx - cand_start) as f64 * fp;
        if dur_ms >= MIN_NOTE_DURATION_MS {
            candidates.push(NoteCandidate {
                start_idx: cand_start,
                end_idx,
                min_pitch: cand_min,
                max_pitch: cand_max,
            });
        }
    }

    // ── Phase 3: 生成 MIDI 事件 ────────────────────────────────────────────
    let mut events: Vec<(u64, TrackEvent<'static>)> = Vec::new();

    // Program Change: Lead 1 (Square) at tick 0
    events.push((
        0,
        TrackEvent {
            delta: delta_u28(0),
            kind: TrackEventKind::Midi {
                channel: midly::num::u4::new(channel),
                message: MidiMessage::ProgramChange {
                    program: midly::num::u7::new(GM_LEAD_1_SQUARE),
                },
            },
        },
    ));

    let has_pitch_bend = candidates.iter().any(|c| {
        let span = c.max_pitch - c.min_pitch;
        span > 0.01
            || smoothed[c.start_idx..c.end_idx]
                .iter()
                .any(|&p| p > 0.0 && (p - p.round()).abs() * 100.0 > 0.5)
    });

    // RPN: Pitch Bend Sensitivity = ±2 semitones
    if has_pitch_bend {
        let cc = |ctrl: u8, val: u8| TrackEvent {
            delta: delta_u28(0),
            kind: TrackEventKind::Midi {
                channel: midly::num::u4::new(channel),
                message: MidiMessage::Controller {
                    controller: midly::num::u7::new(ctrl),
                    value: midly::num::u7::new(val),
                },
            },
        };
        events.push((0, cc(101, 0)));
        events.push((0, cc(100, 0)));
        events.push((0, cc(6, 2)));
        events.push((0, cc(38, 0)));
    }

    for cand in &candidates {
        let base_semi = ((cand.min_pitch + cand.max_pitch) / 2.0)
            .round()
            .clamp(0.0, 127.0) as u8;

        let note_start_sec = start_sec + cand.start_idx as f64 * fp / 1000.0;
        let note_end_sec = start_sec + cand.end_idx as f64 * fp / 1000.0;
        let start_tick = tick_converter.sec_to_ticks(note_start_sec);
        let end_tick = tick_converter.sec_to_ticks(note_end_sec);

        // NoteOn
        events.push((
            start_tick,
            TrackEvent {
                delta: delta_u28(0),
                kind: TrackEventKind::Midi {
                    channel: midly::num::u4::new(channel),
                    message: MidiMessage::NoteOn {
                        key: midly::num::u7::new(base_semi),
                        vel: midly::num::u7::new(DEFAULT_VELOCITY),
                    },
                },
            },
        ));

        // NoteOff
        events.push((
            end_tick,
            TrackEvent {
                delta: delta_u28(0),
                kind: TrackEventKind::Midi {
                    channel: midly::num::u4::new(channel),
                    message: MidiMessage::NoteOff {
                        key: midly::num::u7::new(base_semi),
                        vel: midly::num::u7::new(0),
                    },
                },
            },
        ));

        // 逐帧弯音轮（仅当 has_pitch_bend 时）
        if has_pitch_bend {
            let mut last_bend_raw: Option<u16> = None;
            for fi in cand.start_idx..cand.end_idx {
                let p = smoothed[fi];
                if !p.is_finite() || p <= 0.0 {
                    continue;
                }
                let cents_diff = (p - base_semi as f32) * 100.0;
                let cents_clamped =
                    cents_diff.clamp(-PITCH_BEND_RANGE_CENTS, PITCH_BEND_RANGE_CENTS);
                let bend_raw = cents_to_bend_raw(cents_clamped);
                if last_bend_raw != Some(bend_raw) {
                    last_bend_raw = Some(bend_raw);
                    let t = start_sec + fi as f64 * fp / 1000.0;
                    let tick = tick_converter.sec_to_ticks(t);
                    events.push((
                        tick,
                        TrackEvent {
                            delta: delta_u28(0),
                            kind: TrackEventKind::Midi {
                                channel: midly::num::u4::new(channel),
                                message: MidiMessage::PitchBend {
                                    bend: midly::PitchBend(midly::num::u14::new(bend_raw)),
                                },
                            },
                        },
                    ));
                }
            }
        }
    }

    // ── Phase 4: 排序 + delta 编码 ──────────────────────────────────────────
    events.sort_by_key(|(tick, _)| *tick);
    let mut result: Vec<TrackEvent<'static>> = Vec::with_capacity(events.len() + 1);
    let mut prev_tick: u64 = 0;
    for (tick, mut ev) in events {
        let delta = tick.saturating_sub(prev_tick);
        prev_tick = tick;
        ev.delta = delta_u28(delta);
        result.push(ev);
    }

    result
}

/// 移动平均平滑，减少逐帧抖动。
fn smooth_curve(values: &[f32], window: usize) -> Vec<f32> {
    if values.is_empty() || window <= 1 {
        return values.to_vec();
    }
    let half = window / 2;
    let n = values.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        let slice = &values[start..end];
        let mut sum: f32 = 0.0;
        let mut count: usize = 0;
        for &v in slice {
            if v.is_finite() && v > 0.0 {
                sum += v;
                count += 1;
            }
        }
        // 仅对原本有声音的帧做平滑，静音帧保持静音
        out[i] = if values[i] > 0.0 && count > 0 {
            sum / count as f32
        } else {
            values[i]
        };
    }
    out
}

// ── Conductor Track ──────────────────────────────────────────────────────────

/// Tempo Map 感知的 秒→tick 转换器（积分各段 BPM）。
///
/// 构建时一次性积分各段 BPM 得到每个变化点处的累计 tick 数（O(n)），
/// 之后每次查询用二分查找定位所在段（O(log n)），避免在逐帧弯音导出
/// 等热点路径中退化为 O(points × frames)。
struct TempoTickConverter {
    /// (sec, bpm)，按 sec 升序，首点位于 0。
    points: Vec<(f64, f64)>,
    /// cumulative_ticks[i] = 从 0 到 points[i].sec 的累计 tick 数。
    cumulative_ticks: Vec<f64>,
}

impl TempoTickConverter {
    fn new(timeline: &crate::state::TimelineState, fallback_bpm: f64) -> Self {
        let mut points: Vec<(f64, f64)> = Vec::new();
        if let Some(map) = timeline.tempo_map.as_ref() {
            for p in map {
                points.push((p.position_sec.max(0.0), p.bpm.clamp(10.0, 960.0)));
            }
        }
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        if points.is_empty() || points[0].0 > 1e-9 {
            points.insert(0, (0.0, fallback_bpm.max(10.0).min(960.0)));
        }

        let mut cumulative_ticks = Vec::with_capacity(points.len());
        let mut cumulative = 0.0f64;
        cumulative_ticks.push(cumulative);
        let mut last_sec = points[0].0;
        let mut last_bpm = points[0].1;
        for &(point_sec, point_bpm) in points.iter().skip(1) {
            cumulative +=
                (point_sec - last_sec).max(0.0) * (last_bpm / 60.0) * TICKS_PER_BEAT as f64;
            cumulative_ticks.push(cumulative);
            last_sec = point_sec;
            last_bpm = point_bpm;
        }

        Self {
            points,
            cumulative_ticks,
        }
    }

    fn sec_to_ticks(&self, sec: f64) -> u64 {
        let sec = sec.max(0.0);
        // 最后一个 position_sec < sec 的点：其 bpm 作用于 (points[idx-1].sec, sec)。
        let idx = self
            .points
            .partition_point(|&(point_sec, _)| point_sec < sec);
        let (base_ticks, last_sec, bpm) = if idx == 0 {
            (0.0, 0.0, self.points[0].1)
        } else {
            let (last_sec, bpm) = self.points[idx - 1];
            (self.cumulative_ticks[idx - 1], last_sec, bpm)
        };
        (base_ticks + (sec - last_sec).max(0.0) * (bpm / 60.0) * TICKS_PER_BEAT as f64).round()
            as u64
    }
}

/// 由 Tempo Map 构建 Conductor 轨道（变速/变拍/变调全部写入 meta 事件）。
fn make_conductor_track(
    timeline: &crate::state::TimelineState,
    fallback_bpm: f64,
    fallback_beats_per_bar: u32,
    fallback_base_scale: &str,
) -> Vec<TrackEvent<'static>> {
    let converter = TempoTickConverter::new(timeline, fallback_bpm);

    let mut events: Vec<TrackEvent<'static>> = vec![TrackEvent {
        delta: delta_u28(0),
        kind: TrackEventKind::Meta(MetaMessage::TrackName(b"Conductor")),
    }];

    let mut last_tick: u64 = 0;
    let mut push_meta = |events: &mut Vec<TrackEvent<'static>>,
                         tick: u64,
                         kind: midly::MetaMessage<'static>| {
        events.push(TrackEvent {
            delta: delta_u28(tick.saturating_sub(last_tick)),
            kind: TrackEventKind::Meta(kind),
        });
        last_tick = tick;
    };

    let Some(map) = timeline.tempo_map.as_ref() else {
        // 与 TempoTickConverter::new 的钳制保持一致（10-960）。
        let fallback_bpm = fallback_bpm.max(10.0).min(960.0);
        let tempo_us_per_beat = (60_000_000.0 / fallback_bpm.max(1.0)).round() as u32;
        let (sharps_flats, major_minor) = scale_to_key_signature(fallback_base_scale);
        push_meta(
            &mut events,
            0,
            MetaMessage::Tempo(midly::num::u24::new(tempo_us_per_beat.min(16_777_215))),
        );
        push_meta(
            &mut events,
            0,
            MetaMessage::TimeSignature(
                fallback_beats_per_bar.clamp(1, 32) as u8,
                2,
                24,
                8,
            ),
        );
        push_meta(
            &mut events,
            0,
            MetaMessage::KeySignature(sharps_flats, major_minor),
        );
        events.push(TrackEvent {
            delta: delta_u28(0),
            kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
        });
        return events;
    };

    // 0 位置点：初始速度 / 拍号 / 调号。
    let first = &map[0];
    let (first_num, first_den) = crate::state::TimelineState::effective_time_signature_at(map, 0);
    let tempo_us_per_beat = (60_000_000.0 / first.bpm.max(1.0)).round() as u32;
    push_meta(
        &mut events,
        0,
        MetaMessage::Tempo(midly::num::u24::new(tempo_us_per_beat.min(16_777_215))),
    );
    push_meta(
        &mut events,
        0,
        MetaMessage::TimeSignature(
            first_num.clamp(1, 32) as u8,
            denominator_to_midi_pow2(first_den),
            24,
            8,
        ),
    );
    let mut prev_scale: Option<String> = None;
    if let Some(scale) = first.scale.as_ref() {
        if let Some(key) = scale.key.as_deref() {
            let (sf, mi) = scale_to_key_signature(key);
            push_meta(&mut events, 0, MetaMessage::KeySignature(sf, mi));
            prev_scale = Some(key.to_string());
        }
    }
    // 初始点音阶为“跟随工程音阶”（null）时，回退写入工程的调号，
    // 否则消费方（DAW）会按默认 C 大调理解整首乐曲。
    if prev_scale.is_none() {
        let (sf, mi) = scale_to_key_signature(fallback_base_scale);
        push_meta(&mut events, 0, MetaMessage::KeySignature(sf, mi));
    }

    let mut prev_bpm = first.bpm;
    let mut prev_num = first_num;
    let mut prev_den = first_den;

    for (index, point) in map.iter().enumerate().skip(1) {
        let tick = converter.sec_to_ticks(point.position_sec);
        if (point.bpm - prev_bpm).abs() > 1e-9 {
            let us = (60_000_000.0 / point.bpm.max(1.0)).round() as u32;
            push_meta(
                &mut events,
                tick,
                MetaMessage::Tempo(midly::num::u24::new(us.min(16_777_215))),
            );
            prev_bpm = point.bpm;
        }
        // 拍号跟随之前的拍号时解析为实际生效值；仅在实际值变化时写入。
        let (eff_num, eff_den) =
            crate::state::TimelineState::effective_time_signature_at(map, index);
        if eff_num != prev_num || eff_den != prev_den {
            push_meta(
                &mut events,
                tick,
                MetaMessage::TimeSignature(
                    eff_num.clamp(1, 32) as u8,
                    denominator_to_midi_pow2(eff_den),
                    24,
                    8,
                ),
            );
            prev_num = eff_num;
            prev_den = eff_den;
        }
        let scale_key = point.scale.as_ref().and_then(|s| s.key.clone());
        if scale_key != prev_scale {
            if let Some(key) = scale_key.as_deref() {
                let (sf, mi) = scale_to_key_signature(key);
                push_meta(&mut events, tick, MetaMessage::KeySignature(sf, mi));
            }
            prev_scale = scale_key;
        }
    }

    events.push(TrackEvent {
        delta: delta_u28(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });
    events
}

/// 拍号分母 → MIDI TimeSignature 的 2 的指数（4 → 2 表示四分音符）。
fn denominator_to_midi_pow2(denominator: u32) -> u8 {
    match denominator {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        16 => 4,
        32 => 5,
        _ => 2,
    }
}

// ── 读取 pitch（含子轨偏移） ─────────────────────────────────────────────────

fn read_pitch_for_track(
    timeline: &crate::state::TimelineState,
    entry: &MidiExportTrackEntry,
) -> Result<(Vec<f32>, f64), String> {
    let root_id = &entry.root_track_id;
    let params = timeline
        .params_by_root_track
        .get(root_id)
        .ok_or_else(|| format!("no_params_for_root: {root_id}"))?;

    let fp = params.frame_period_ms.max(1e-6);
    let start_frame = (entry.start_sec * 1000.0 / fp).floor() as usize;
    let end_frame = (entry.end_sec * 1000.0 / fp).ceil() as usize;
    let frame_count = end_frame.saturating_sub(start_frame);

    if frame_count == 0 {
        return Err("empty_range".into());
    }

    // 读取主音高（与 get_param_frames 逻辑一致：edit==0 && orig!=0 → orig）
    let mut pitch: Vec<f32> = Vec::with_capacity(frame_count);
    for i in 0..frame_count {
        let idx = start_frame + i;
        let o = params.pitch_orig.get(idx).copied().unwrap_or(0.0);
        let e_raw = params.pitch_edit.get(idx).copied().unwrap_or(0.0);
        let v = if e_raw == 0.0 && o != 0.0 { o } else { e_raw };
        pitch.push(v);
    }

    // 如果是子轨，应用子轨音高偏移（Tempo Map 感知：按帧时刻解析生效音阶）
    if entry.track_id != entry.root_track_id {
        let cents_key = format!("{CHILD_CENTS_PREFIX}{}", entry.track_id);
        let degrees_key = format!("{CHILD_DEGREES_PREFIX}{}", entry.track_id);

        let cents_curve = params.extra_curves.get(&cents_key);
        let degrees_curve = params.extra_curves.get(&degrees_key);
        let scale_segments = timeline.scale_segments();
        let mut cursor = crate::pitch_editing::ScaleSegmentsCursor::new(&scale_segments);

        for i in 0..frame_count {
            let frame_idx = start_frame + i;

            let cents_offset = cents_curve
                .and_then(|c| c.get(frame_idx).copied())
                .unwrap_or(0.0) as f64;
            let degree_offset = degrees_curve
                .and_then(|c| c.get(frame_idx).copied())
                .unwrap_or(0.0) as f64;

            if cents_offset.abs() > 1e-9 || degree_offset.abs() > 1e-9 {
                let v = pitch[i] as f64 + cents_offset / 100.0;
                let scale_notes = cursor.notes_at((frame_idx as f64) * fp / 1000.0);
                let v = transpose_midi_by_scale_steps(v, degree_offset, scale_notes);
                pitch[i] = v as f32;
            }
        }
    }

    Ok((pitch, fp))
}

// ── 从单个 clip 读取音高（非 Compose 轨道的 fallback） ────────────────────────

const EXPORT_FRAME_PERIOD_MS: f64 = 5.0;

/// 当 track 级 `TrackParamsState` 不可用时，直接从 clip 的 FCPE 缓存
/// （或 MIDI note data）计算该 clip 在时间线区间上的音高曲线。
fn read_pitch_for_clip(
    timeline: &TimelineState,
    entry: &MidiExportTrackEntry,
) -> Result<(Vec<f32>, f64), String> {
    let clip_id = entry
        .clip_id
        .as_ref()
        .ok_or_else(|| "no_clip_id_for_fallback".to_string())?;
    let clip = timeline
        .clips
        .iter()
        .find(|c| c.id == *clip_id)
        .ok_or_else(|| format!("clip_not_found: {clip_id}"))?;

    let fp = EXPORT_FRAME_PERIOD_MS.max(1e-6);
    let clip_len = clip.length_sec.max(0.0);
    let target_frames = ((clip_len * 1000.0) / fp).round().max(1.0) as usize;

    // ── MIDI clip 路径 ──
    if let Some(ref notes) = clip.midi_note_data {
        let mut midi_curve = vec![0.0f32; target_frames];

        let pr = clip.playback_rate as f64;
        let pr_valid = if pr.is_finite() && pr > 0.0 { pr } else { 1.0 };
        // 消费窗口（非 Loop 倒放锚定 se：win=[se−len·r, se]）——
        // 与引擎 emit / assemble 的窗口模型一致。
        let src_end = if clip.source_end_sec > 0.0 {
            clip.source_end_sec
        } else {
            notes
                .iter()
                .map(|n| n.end_sec)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(clip_len)
                .max(clip_len)
        };
        let src_start = if !clip.loop_enabled && clip.reversed {
            src_end - clip_len * pr_valid
        } else {
            clip.source_start_sec
        };
        let src_total_len = src_end - src_start;

        for note in notes {
            // Loop（循环源）：按媒体时长 D 的锚点回绕放置音符（与实时引擎 /
            // assemble 的放置算法一致）。不能用窗口比较过滤可见性 —— split
            // 产生的"环绕窗口"（start > end）会把所有音符误判为越界。
            if let Some(placement) = crate::state::place_note_occurrence_in_loop(
                clip,
                note.start_sec,
                note.end_sec,
                fp,
            ) {
                let note_value = note.note as f32;
                let mut cycle_offset = 0usize;
                while cycle_offset < target_frames {
                    let write_start = cycle_offset + placement.first_start_frame;
                    let write_end = (cycle_offset
                        + placement.first_start_frame
                        + placement.len_frames)
                        .min(target_frames);
                    if write_start >= write_end {
                        break;
                    }
                    for frame in write_start..write_end {
                        let current = midi_curve[frame];
                        if note_value > current || current <= 0.0 {
                            midi_curve[frame] = note_value;
                        }
                    }
                    cycle_offset += placement.cycle_frames;
                }
                continue;
            }

            if note.end_sec <= src_start || note.start_sec >= src_end {
                continue;
            }
            let rel_start = (note.start_sec - src_start).max(0.0);
            let rel_end = (note.end_sec - src_start).min(src_total_len);
            if rel_end <= rel_start {
                continue;
            }
            let (eff_start, eff_end) = if clip.reversed {
                (
                    (src_total_len - rel_end).max(0.0),
                    (src_total_len - rel_start).min(src_total_len),
                )
            } else {
                (rel_start, rel_end)
            };
            if eff_end <= eff_start {
                continue;
            }
            let note_start_frame = ((eff_start / pr_valid * 1000.0) / fp).round() as usize;
            let note_end_frame = ((eff_end / pr_valid * 1000.0) / fp).round() as usize;
            let write_end = note_end_frame.min(target_frames);
            if note_start_frame < write_end {
                let note_value = note.note as f32;
                for frame in note_start_frame..write_end {
                    let current = midi_curve[frame];
                    if note_value > current || current <= 0.0 {
                        midi_curve[frame] = note_value;
                    }
                }
            }
        }

        if clip.midi_fill_gaps && target_frames > 0 {
            let first = midi_curve.iter().position(|&v| v > 0.0);
            let last = midi_curve.iter().rposition(|&v| v > 0.0);
            if let (Some(f), Some(l)) = (first, last) {
                if f < l {
                    let mut last_pitch: f32 = 0.0;
                    for i in f..=l {
                        let current = midi_curve[i];
                        if current > 0.0 {
                            last_pitch = current;
                        } else if last_pitch > 0.0 {
                            midi_curve[i] = last_pitch;
                        }
                    }
                }
            }
        }

        return Ok((midi_curve, fp));
    }

    // ── 音频 clip 路径：按需运行 FCPE ──
    let full_midi = pitch_clip::compute_clip_pitch_midi(timeline, clip, &entry.root_track_id, fp)
        .ok_or_else(|| format!("fcpe_failed_for_clip: {clip_id}"))?;

    let pr = clip.playback_rate as f64;
    let pr = if pr.is_finite() && pr > 0.0 { pr } else { 1.0 };

    // 非 Loop 倒放：传入真实消费窗口 [se−len·r, se]（输出在下方整体反转）。
    let (trim_src_start, trim_src_end) = crate::state::clip_pitch_trim_window_sec(clip);
    let mut curve = pitch_clip::trim_and_resample_midi(
        &full_midi,
        fp,
        trim_src_start,
        trim_src_end,
        pr,
        clip_len,
        clip.loop_enabled,
        crate::state::clip_source_media_duration_sec(clip),
        clip.reversed && clip.loop_enabled,
    );

    if clip.reversed && !clip.loop_enabled && !curve.is_empty() {
        // 非 Loop 倒放：曲线整体反转（既有约定）。
        // Loop 模式的倒放方向已在 trim_and_resample_midi 内部按
        // "从 source_end 向下环绕"处理，无需再反转。
        curve.reverse();
    }

    Ok((curve, fp))
}

// ── 选择音高来源 ─────────────────────────────────────────────────────────────

/// 选择导出用的音高数据来源。
///
/// - Compose 轨道：优先使用 track 级曲线（包含 pitch_edit、子轨偏移等）。
/// - 非 Compose 轨道：track 级曲线通常只是全零占位，因此当请求带有
///   `clip_id` 时，自动退回 clip 级读取：MIDI clip 直接展开音符，音频 clip
///   按需运行 FCPE。这样用户无需手动开启 Compose 也能完成导出。
fn read_pitch_for_export(
    timeline: &TimelineState,
    entry: &MidiExportTrackEntry,
) -> Result<(Vec<f32>, f64), String> {
    let compose_enabled = timeline
        .tracks
        .iter()
        .find(|track| track.id == entry.root_track_id)
        .map(|track| track.compose_enabled)
        .unwrap_or(false);

    let track_has_pitch = timeline
        .params_by_root_track
        .get(&entry.root_track_id)
        .map(|params| {
            params.pitch_orig.iter().any(|value| *value > 0.0)
                || params.pitch_edit.iter().any(|value| *value > 0.0)
        })
        .unwrap_or(false);

    if entry.clip_id.is_some() && (!compose_enabled || !track_has_pitch) {
        match read_pitch_for_clip(timeline, entry) {
            Ok(values) => return Ok(values),
            // clip 级读取失败时继续尝试 track 级数据，避免丢失已有的音高编辑。
            Err(_) => {}
        }
    }

    read_pitch_for_track(timeline, entry)
}

// ── 主入口 ────────────────────────────────────────────────────────────────────

pub(super) fn export_pitch_to_midi(
    state: &AppState,
    request: MidiExportRequest,
) -> serde_json::Value {
    let timeline = match state.timeline.lock() {
        Ok(guard) => guard,
        Err(e) => {
            return serde_json::json!({"ok": false, "error": format!("lock_failed: {}", e)});
        }
    };

    let channels = assign_channels(request.tracks.len());
    let mut midi_tracks: Vec<Vec<TrackEvent<'static>>> =
        Vec::with_capacity(request.tracks.len() + 1);

    // Track 0: Conductor（Tempo Map 感知：变速/变拍/变调写入 meta 事件）
    midi_tracks.push(make_conductor_track(
        &timeline,
        request.bpm,
        request.beats_per_bar,
        &request.base_scale,
    ));
    let tick_converter = TempoTickConverter::new(&timeline, request.bpm);

    let mut any_pitch_data = false;

    for (idx, entry) in request.tracks.iter().enumerate() {
        let channel = channels[idx];

        let name_bytes: &'static [u8] = Box::leak(entry.name.clone().into_bytes().into_boxed_slice());

        let (pitch_values, fp) = match read_pitch_for_export(&timeline, entry) {
            Ok(v) => v,
            Err(_) => {
                midi_tracks.push(vec![
                    TrackEvent {
                        delta: delta_u28(0),
                        kind: TrackEventKind::Meta(MetaMessage::TrackName(name_bytes)),
                    },
                    TrackEvent {
                        delta: delta_u28(0),
                        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
                    },
                ]);
                continue;
            }
        };

        let has_pitch = pitch_values.iter().any(|v| *v > 0.0);
        any_pitch_data = any_pitch_data || has_pitch;

        let mut events = vec![TrackEvent {
            delta: delta_u28(0),
            kind: TrackEventKind::Meta(MetaMessage::TrackName(name_bytes)),
        }];

        if has_pitch {
            let track_events = pitch_curve_to_track_events(
                &pitch_values,
                fp,
                entry.start_sec,
                channel,
                &tick_converter,
            );
            events.extend(track_events);
        }

        events.push(TrackEvent {
            delta: delta_u28(0),
            kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
        });

        midi_tracks.push(events);
    }

    if !any_pitch_data {
        let all_no_pitch = request.tracks.iter().all(|entry| {
            timeline
                .params_by_root_track
                .get(&entry.root_track_id)
                .map(|p| {
                    !p.pitch_edit.iter().any(|v| *v > 0.0)
                        && !p.pitch_orig.iter().any(|v| *v > 0.0)
                })
                .unwrap_or(true)
        });
        if all_no_pitch {
            return serde_json::json!({"ok": false, "error": "no_pitch_data"});
        }
    }

    // 写入 SMF
    let smf = Smf {
        header: Header::new(
            Format::Parallel,
            Timing::Metrical(midly::num::u15::new(TICKS_PER_BEAT)),
        ),
        tracks: midi_tracks,
    };

    let mut file = match fs::File::create(&request.output_path) {
        Ok(f) => f,
        Err(e) => {
            return serde_json::json!({"ok": false, "error": format!("io_error: {}", e)});
        }
    };

    match smf.write_std(&mut file) {
        Ok(_) => {
            let _ = file.flush();
            serde_json::json!({"ok": true})
        }
        Err(e) => {
            serde_json::json!({"ok": false, "error": format!("midi_write_error: {}", e)})
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TempoTickConverter;
    use crate::state::{TempoPointData, TimelineState};

    fn point(id: &str, position_sec: f64, bpm: f64) -> TempoPointData {
        TempoPointData {
            id: id.to_string(),
            position_sec,
            bpm,
            numerator: Some(4),
            denominator: Some(4),
            scale: None,
        }
    }

    fn converter_with(points: &[TempoPointData], fallback_bpm: f64) -> TempoTickConverter {
        let mut timeline = TimelineState::default();
        timeline.tempo_map = Some(points.to_vec());
        TempoTickConverter::new(&timeline, fallback_bpm)
    }

    #[test]
    fn fallback_converts_at_constant_tempo() {
        let converter = TempoTickConverter::new(&TimelineState::default(), 90.0);
        // 90 BPM = 1.5 beat/s × 480 ticks/beat = 720 ticks/s。
        assert_eq!(converter.sec_to_ticks(0.0), 0);
        assert_eq!(converter.sec_to_ticks(1.0), 720);
        assert_eq!(converter.sec_to_ticks(2.5), 1_800);
        assert_eq!(converter.sec_to_ticks(-5.0), 0);
    }

    #[test]
    fn integrates_each_segment_at_its_own_bpm() {
        let points = [
            point("a", 0.0, 120.0),
            point("b", 10.0, 60.0),
            point("c", 20.0, 180.0),
        ];
        let converter = converter_with(&points, 120.0);
        assert_eq!(converter.sec_to_ticks(0.0), 0);
        // 变化点边界处仍按前一段的 BPM 计算。
        assert_eq!(converter.sec_to_ticks(10.0), 9_600);
        assert_eq!(converter.sec_to_ticks(15.0), 12_000);
        assert_eq!(converter.sec_to_ticks(20.0), 14_400);
        assert_eq!(converter.sec_to_ticks(21.0), 15_840);
    }

    #[test]
    fn sorts_points_before_integration() {
        let points = [
            point("c", 20.0, 180.0),
            point("b", 10.0, 60.0),
            point("a", 0.0, 120.0),
        ];
        let converter = converter_with(&points, 120.0);
        assert_eq!(converter.sec_to_ticks(15.0), 12_000);
        assert_eq!(converter.sec_to_ticks(21.0), 15_840);
    }

    #[test]
    fn duplicate_positions_keep_the_last_bpm_after_the_boundary() {
        let points = [
            point("a", 0.0, 120.0),
            point("b", 10.0, 60.0),
            point("b2", 10.0, 180.0),
            point("c", 20.0, 120.0),
        ];
        let converter = converter_with(&points, 120.0);
        assert_eq!(converter.sec_to_ticks(10.0), 9_600);
        assert_eq!(converter.sec_to_ticks(15.0), 16_800);
        assert_eq!(converter.sec_to_ticks(21.0), 24_960);
    }

    #[test]
    fn inserts_fallback_point_when_map_does_not_start_at_zero() {
        let points = [point("a", 5.0, 120.0)];
        let converter = converter_with(&points, 90.0);
        assert_eq!(converter.sec_to_ticks(4.0), 2_880);
        assert_eq!(converter.sec_to_ticks(5.0), 3_600);
        assert_eq!(converter.sec_to_ticks(6.0), 4_560);
    }

    #[test]
    fn non_compose_export_reads_midi_clip_fallback() {
        use super::{read_pitch_for_export, EXPORT_FRAME_PERIOD_MS, MidiExportTrackEntry};
        use crate::midi_import::MidiNoteEvent;

        let mut timeline = TimelineState::default();
        let root_id = timeline.tracks[0].id.clone();
        timeline.ensure_params_for_root(&root_id);
        assert!(timeline.params_by_root_track.contains_key(&root_id));

        let clip_id = timeline.add_clip(
            Some(root_id.clone()),
            Some("MIDI Clip".to_string()),
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
        }

        let entry = MidiExportTrackEntry {
            track_id: root_id.clone(),
            root_track_id: root_id,
            name: "MIDI Clip".to_string(),
            start_sec: 0.0,
            end_sec: 2.0,
            clip_id: Some(clip_id),
        };

        // 即使 Compose 关闭且 track 级曲线为空，也应自动走 clip 级读取。
        let (pitch, fp) = read_pitch_for_export(&timeline, &entry).expect("clip fallback export");
        assert_eq!(fp, EXPORT_FRAME_PERIOD_MS);
        assert!(pitch.iter().any(|value| *value > 0.0));
    }
}
