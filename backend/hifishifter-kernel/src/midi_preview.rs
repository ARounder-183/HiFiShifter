//! MIDI 试听：只用于编辑参考的轻量合成。
//!
//! 这里没有采样器、没有 GM 还原、没有外部插件调用。它只回答一个问题：
//! “这一段 MIDI 大致是什么旋律/和声？”因此合成器故意保持极简（正弦/三角/锯齿），
//! 默认低音量，并且与 mixdown、导出、渲染缓存完全隔离。
//!
//! 合成是无状态的：第 n 帧的相位由“该帧相对音符起点的秒数”直接求出，
//! 不保存振荡器/包络状态。这样 seek、loop、块边界都不会留下尾音或爆音，
//! 也满足音频线程“无分配、无锁”的要求。

use crate::fade_curves::{global_fade_lut, sample_fade_lut, FADE_LUT_SIZE};
use crate::midi_take::{MidiExpressionKind, MidiPreviewTimbre, MidiTakeData};
use crate::state::{clip_loop_period_sec, clip_playback_window_sec, TimelineState};
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct MidiPreviewSchedule {
    pub clips: Vec<MidiPreviewClip>,
}

#[derive(Debug, Clone)]
pub struct MidiPreviewClip {
    pub clip_id: String,
    /// 工程时间轴起点（秒）。
    pub start_sec: f64,
    pub length_sec: f64,
    /// 源域消费窗口与方向。
    pub source_start_sec: f64,
    pub source_end_sec: f64,
    pub playback_rate: f64,
    pub reversed: bool,
    pub loop_enabled: bool,
    pub loop_period_sec: Option<f64>,
    pub notes: Vec<PreviewNote>,
    pub settings: crate::midi_take::MidiPreviewSettings,
    /// Clip 的 item × take 增益，再乘轨道父链增益。
    pub gain: f32,
    /// 有效淡入/淡出（自动交叉淡化优先，与音频渲染同口径）。
    pub fade_in_sec: f64,
    pub fade_out_sec: f64,
    pub fade_in_lut: Option<Arc<Vec<f32>>>,
    pub fade_out_lut: Option<Arc<Vec<f32>>>,
}

#[derive(Debug, Clone)]
pub struct PreviewNote {
    pub start_sec: f64,
    pub end_sec: f64,
    pub note: f32,
    pub velocity: u8,
    pub channel: u8,
    /// 同一通道的弯音事件，时间已换算为相对音符播放起点的秒数。
    ///
    /// 正放时从 start_sec 计；倒放时从 end_sec 计。Loop 下按回绕周期取模，
    /// 因此跨过循环边界的音符也能拿到正确的前一圈弯音状态。
    pub bends: Vec<(f64, f32)>,
    /// 音符起点处已经生效的弯音值（没有事件时为 0）。
    pub initial_bend: f32,
}

impl MidiPreviewSchedule {
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// 从时间线收集启用试听的 MIDI Take。
    ///
    /// 只读 role 无关：音高参考也可以试听，但默认关闭；普通 MIDI Take 默认开启。
    /// 这里同时消费 Clip 的 item/take 增益、静音、轨道父链静音/solo、有效淡变、
    /// Loop/倒放/速率/裁剪窗口。MIDI Take 因此不是一套“旁路参数”，而是与音频
    /// Clip 共用同一组时间轴属性。
    pub fn from_timeline(timeline: &TimelineState) -> Self {
        let track_outputs = crate::state::compute_track_output_gains(&timeline.tracks);
        let mut clips = Vec::new();
        for clip in &timeline.clips {
            if clip.muted {
                continue;
            }
            let Some(data) = clip.midi_take.as_ref() else {
                continue;
            };
            if !data.preview.enabled || data.notes.is_empty() {
                continue;
            }
            let track_output = track_outputs
                .get(&clip.track_id)
                .copied()
                .unwrap_or(crate::state::TrackOutputGain {
                    gain: 1.0,
                    muted: false,
                    solo_ok: true,
                });
            if track_output.muted || !track_output.solo_ok {
                continue;
            }
            let gain = crate::gain::clamp_gain(clip.audible_gain() * track_output.gain);
            if gain <= 0.0 {
                continue;
            }
            let length = clip.length_sec.max(0.0);
            if !(length.is_finite() && length > 1e-9) {
                continue;
            }
            let rate = if clip.playback_rate.is_finite() && clip.playback_rate > 1e-6 {
                clip.playback_rate as f64
            } else {
                1.0
            };
            let (source_start_sec, source_end_sec) = clip_playback_window_sec(clip);
            let loop_period_sec = clip
                .loop_enabled
                .then(|| clip_loop_period_sec(clip))
                .flatten()
                .or_else(|| {
                    clip.loop_enabled
                        .then_some((source_end_sec - source_start_sec).abs())
                })
                .filter(|value| value.is_finite() && *value > 1e-9);
            let loop_enabled = clip.loop_enabled && loop_period_sec.is_some();
            let fade_in_sec = clip.effective_fade_in_sec().clamp(0.0, length);
            let fade_out_sec = clip.effective_fade_out_sec().clamp(0.0, length);
            let fade_in_lut =
                (fade_in_sec > 1e-9).then(|| global_fade_lut(clip.fade_in_shape, clip.fade_in_dir, false));
            let fade_out_lut =
                (fade_out_sec > 1e-9).then(|| global_fade_lut(clip.fade_out_shape, clip.fade_out_dir, true));
            let notes = data
                .notes
                .iter()
                .filter(|note| !note.muted)
                .map(|note| {
                    let (bends, initial_bend) =
                        preview_bends(data, note, clip.reversed, loop_period_sec);
                    PreviewNote {
                        start_sec: note.start_sec,
                        end_sec: note.end_sec,
                        note: note.note,
                        velocity: note.velocity,
                        channel: note.channel,
                        bends,
                        initial_bend,
                    }
                })
                .collect();
            clips.push(MidiPreviewClip {
                clip_id: clip.id.clone(),
                start_sec: clip.start_sec.max(0.0),
                length_sec: length,
                source_start_sec,
                source_end_sec,
                playback_rate: rate,
                reversed: clip.reversed,
                loop_enabled,
                loop_period_sec,
                notes,
                settings: data.preview,
                gain,
                fade_in_sec,
                fade_out_sec,
                fade_in_lut,
                fade_out_lut,
            });
        }
        Self { clips }
    }
}

/// 把一条音符的表情事件换算到“相对音符播放起点”的坐标系。
fn preview_bends(
    data: &MidiTakeData,
    note: &crate::midi_import::MidiNoteEvent,
    reversed: bool,
    cycle: Option<f64>,
) -> (Vec<(f64, f32)>, f32) {
    let channel = note.channel.min(15);
    let range = data.pitch_bend_ranges[channel as usize];
    let note_len = (note.end_sec - note.start_sec).max(0.0);
    let semitones =
        |value: i16| -> f32 { (value as f32 / 8192.0 * range).clamp(-96.0, 96.0) };
    let mut bends = Vec::new();
    for event in &data.expression {
        let MidiExpressionKind::PitchBend {
            channel: event_channel,
            value,
        } = event.kind
        else {
            continue;
        };
        if event_channel != channel {
            continue;
        }
        let offset = if reversed {
            note.end_sec - event.time_sec
        } else {
            event.time_sec - note.start_sec
        };
        let offset = match cycle {
            Some(cycle) => offset.rem_euclid(cycle),
            None => offset,
        };
        if offset >= -1e-9 && offset <= note_len + 1e-9 {
            bends.push((offset.max(0.0), semitones(value)));
        }
    }
    bends.sort_by(|a, b| a.0.total_cmp(&b.0));
    // 起点处已生效的弯音：正放取音符起点之前最近的一条；倒放取音符终点之前
    // 最近的一条（倒放从终点开始向下读源）。Loop 下若没有“之前”的事件，
    // 取整圈最后一条，避免循环接缝处弯音突然归零。
    let reference = if reversed { note.end_sec } else { note.start_sec };
    let before = data
        .expression
        .iter()
        .filter(|event| event.time_sec <= reference + 1e-9)
        .filter_map(|event| match event.kind {
            MidiExpressionKind::PitchBend {
                channel: event_channel,
                value,
            } if event_channel == channel => Some(semitones(value)),
            _ => None,
        })
        .last();
    let initial_bend = before.unwrap_or_else(|| {
        if cycle.is_some() {
            data.expression
                .iter()
                .filter_map(|event| match event.kind {
                    MidiExpressionKind::PitchBend {
                        channel: event_channel,
                        value,
                    } if event_channel == channel => Some(semitones(value)),
                    _ => None,
                })
                .last()
                .unwrap_or(0.0)
        } else {
            0.0
        }
    });
    (bends, initial_bend)
}

#[inline]
fn midi_freq(note: f32) -> f32 {
    440.0 * 2.0_f32.powf((note - 69.0) / 12.0)
}

#[inline]
fn gain_from_db(db: f32) -> f32 {
    if db.is_finite() {
        10.0_f32.powf(db / 20.0)
    } else {
        0.0
    }
}

/// 源域时间映射（与音频 clip 的 Loop/倒放/速率语义同构）。
fn source_time(clip: &MidiPreviewClip, local_sec: f64) -> Option<f64> {
    let rate = clip.playback_rate.max(1e-6);
    if let Some(total) = clip
        .loop_period_sec
        .filter(|value| value.is_finite() && *value > 1e-9)
        .filter(|_| clip.loop_enabled)
    {
        let anchored = if clip.reversed {
            // 与音频路径 / place_note_occurrence_in_loop 同约定：倒放锚点先
            // clamp 到内容总时长，再做 floor_mod；否则超界 source_end 会让
            // 试听与时间轴音符块错开一个固定相位。
            clip.source_end_sec.min(total) - local_sec * rate
        } else {
            clip.source_start_sec + local_sec * rate
        };
        return Some(anchored.rem_euclid(total));
    }
    let span = (clip.source_end_sec - clip.source_start_sec).abs().max(0.0);
    if span <= 1e-9 {
        return None;
    }
    let rel = (local_sec * rate).min(span);
    Some(if clip.reversed {
        clip.source_end_sec - rel
    } else {
        clip.source_start_sec + rel
    })
}

/// 当前源位置落在这条音符内部的“播放年龄”（秒）；不在音符内返回 None。
///
/// Loop 使用圆周距离而不是简单的区间包含：一个横跨回绕边界的音符
/// （例如 9..11 秒、周期 10 秒）在下一圈开头仍应继续响，而不是被边界截断。
fn note_age(clip: &MidiPreviewClip, src_sec: f64, note: &PreviewNote) -> Option<f64> {
    let len = note.end_sec - note.start_sec;
    if !(len > 1e-9) {
        return None;
    }
    if clip.loop_enabled {
        let cycle = clip.loop_period_sec?;
        if !(cycle > 1e-9) {
            return None;
        }
        let age = if clip.reversed {
            (note.end_sec - src_sec).rem_euclid(cycle)
        } else {
            (src_sec - note.start_sec).rem_euclid(cycle)
        };
        if age < len || len >= cycle - 1e-9 {
            Some(age)
        } else {
            None
        }
    } else {
        let age = if clip.reversed {
            note.end_sec - src_sec
        } else {
            src_sec - note.start_sec
        };
        (age >= 0.0 && age < len).then_some(age)
    }
}

fn voice_sample(note: &PreviewNote, age: f64, timbre: MidiPreviewTimbre) -> f32 {
    let length = (note.end_sec - note.start_sec).max(1e-6);
    let age = age.clamp(0.0, length);
    let remain = (length - age).max(0.0);
    // ADSR：5ms attack、60ms decay -> 0.7、末 30ms release。全部由时间直接求值。
    let attack = (age / 0.005).min(1.0) as f32;
    let decay = if age < 0.065 {
        1.0 - 0.3 * ((age - 0.005).max(0.0) / 0.06) as f32
    } else {
        0.7
    };
    let release = (remain / 0.03).min(1.0) as f32;
    let env = attack * decay * release;
    let velocity = (note.velocity as f32 / 127.0).max(0.02);
    let bend = note
        .bends
        .iter()
        .take_while(|(time, _)| *time <= age + 1e-9)
        .last()
        .map(|(_, semitones)| *semitones)
        .unwrap_or(note.initial_bend);
    let freq = midi_freq(note.note + bend);
    let phase = std::f64::consts::TAU * freq as f64 * age;
    let wave = match timbre {
        MidiPreviewTimbre::SoftSine => phase.sin() as f32,
        MidiPreviewTimbre::Triangle => (2.0 / std::f64::consts::PI * phase.sin().asin()) as f32,
        MidiPreviewTimbre::Saw => {
            let x = phase / std::f64::consts::TAU;
            (2.0 * (x - (x + 0.5).floor())) as f32
        }
    };
    (wave * env * velocity * 0.25).clamp(-1.0, 1.0)
}

/// Clip 有效淡变在局部秒处的增益。
fn clip_fade_gain(clip: &MidiPreviewClip, local_sec: f64, sample_rate: u32) -> f32 {
    let mut gain = 1.0f32;
    let frame = 1.0 / sample_rate.max(1) as f64;
    if clip.fade_in_sec > 1e-9 && local_sec < clip.fade_in_sec {
        let progress = ((local_sec + frame) / clip.fade_in_sec).clamp(0.0, 1.0);
        gain *= match clip.fade_in_lut.as_deref() {
            Some(lut) => sample_fade_lut(lut, progress * FADE_LUT_SIZE as f64),
            None => progress as f32,
        };
    }
    if clip.fade_out_sec > 1e-9 {
        let zone_start = (clip.length_sec - clip.fade_out_sec).max(0.0);
        if local_sec >= zone_start {
            let progress = ((local_sec - zone_start + frame) / clip.fade_out_sec).clamp(0.0, 1.0);
            gain *= match clip.fade_out_lut.as_deref() {
                Some(lut) => sample_fade_lut(lut, progress * FADE_LUT_SIZE as f64),
                None => 1.0 - progress as f32,
            };
        }
    }
    gain
}

/// 单帧的试听样本（所有 Clip 共用，避免 interleaved/planar 两条路径各写一份）。
fn frame_sample(clip: &MidiPreviewClip, local_sec: f64, sample_rate: u32) -> f32 {
    let fade_gain = clip_fade_gain(clip, local_sec, sample_rate);
    if fade_gain <= 0.0 {
        return 0.0;
    }
    let Some(src) = source_time(clip, local_sec) else {
        return 0.0;
    };
    let mut sample = 0.0f32;
    let mut voices = 0u16;
    for note in &clip.notes {
        if voices >= clip.settings.polyphony.max(1) {
            break;
        }
        if let Some(age) = note_age(clip, src, note) {
            voices = voices.saturating_add(1);
            sample += voice_sample(note, age, clip.settings.timbre);
        }
    }
    sample * fade_gain
}

/// 把一个立体声 interleaved 块叠加试听。pos0/pos1 是工程帧。
pub fn mix_into_interleaved(
    scratch: &mut [f32],
    schedule: &MidiPreviewSchedule,
    pos0: u64,
    pos1: u64,
    sample_rate: u32,
) {
    if schedule.is_empty() || sample_rate == 0 || pos1 <= pos0 {
        return;
    }
    let sr = sample_rate as f64;
    let frames = (pos1 - pos0) as usize;
    if scratch.len() < frames * 2 {
        return;
    }
    for clip in &schedule.clips {
        let start = (clip.start_sec * sr).round() as u64;
        let end = ((clip.start_sec + clip.length_sec) * sr).round() as u64;
        let overlap_start = start.max(pos0);
        let overlap_end = end.min(pos1);
        if overlap_end <= overlap_start {
            continue;
        }
        let gain = gain_from_db(clip.settings.gain_db) * clip.gain;
        if gain <= 0.0 {
            continue;
        }
        for frame in overlap_start..overlap_end {
            let local = (frame - start) as f64 / sr;
            let sample = frame_sample(clip, local, sample_rate);
            if sample == 0.0 {
                continue;
            }
            let idx = (frame - pos0) as usize * 2;
            if idx + 1 < scratch.len() {
                scratch[idx] += sample * gain;
                scratch[idx + 1] += sample * gain;
            }
        }
    }
}

/// 插件输出面：直接写前两个声道平面。planes 可为单声道。
pub fn mix_into_planar(
    planes: [*mut f32; 2],
    frames: usize,
    schedule: &MidiPreviewSchedule,
    pos0: u64,
    pos1: u64,
    sample_rate: u32,
) {
    if schedule.is_empty() || sample_rate == 0 || frames == 0 || pos1 <= pos0 {
        return;
    }
    let sr = sample_rate as f64;
    for clip in &schedule.clips {
        let start = (clip.start_sec * sr).round() as u64;
        let end = ((clip.start_sec + clip.length_sec) * sr).round() as u64;
        let overlap_start = start.max(pos0);
        let overlap_end = end.min(pos1);
        if overlap_end <= overlap_start {
            continue;
        }
        let gain = gain_from_db(clip.settings.gain_db) * clip.gain;
        if gain <= 0.0 {
            continue;
        }
        for frame in overlap_start..overlap_end {
            let local = (frame - start) as f64 / sr;
            let sample = frame_sample(clip, local, sample_rate);
            if sample == 0.0 {
                continue;
            }
            let out = (frame - pos0) as usize;
            if out >= frames {
                continue;
            }
            for plane in planes.iter().filter(|plane| !plane.is_null()) {
                // SAFETY: 调用方保证每个非空平面至少可写 frames 帧。
                unsafe {
                    *plane.add(out) += sample * gain;
                }
            }
        }
    }
}

/// 从单个 MIDI Take 构建一条独立试听轨（预览编辑器/测试用）。
pub fn clip_from_take(
    clip_id: String,
    start_sec: f64,
    length_sec: f64,
    data: &MidiTakeData,
) -> Option<MidiPreviewClip> {
    if !data.preview.enabled || data.notes.is_empty() {
        return None;
    }
    let content_length = data.effective_length_sec().max(length_sec).max(0.1);
    Some(MidiPreviewClip {
        clip_id,
        start_sec,
        length_sec,
        source_start_sec: 0.0,
        source_end_sec: content_length,
        playback_rate: 1.0,
        reversed: false,
        loop_enabled: false,
        loop_period_sec: Some(content_length),
        notes: data
            .notes
            .iter()
            .filter(|note| !note.muted)
            .map(|note| PreviewNote {
                start_sec: note.start_sec,
                end_sec: note.end_sec,
                note: note.note,
                velocity: note.velocity,
                channel: note.channel,
                bends: Vec::new(),
                initial_bend: 0.0,
            })
            .collect(),
        settings: data.preview,
        gain: 1.0,
        fade_in_sec: 0.0,
        fade_out_sec: 0.0,
        fade_in_lut: None,
        fade_out_lut: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi_import::MidiNoteEvent;
    use crate::midi_take::{MidiTakeData, MidiTakeRole};

    fn note() -> MidiNoteEvent {
        MidiNoteEvent {
            start_sec: 0.0,
            end_sec: 0.2,
            note: 69.0,
            velocity: 127,
            channel: 0,
            muted: false,
            release_velocity: 0,
        }
    }

    fn clip_for(data: &MidiTakeData) -> MidiPreviewClip {
        let mut clip = clip_from_take("c".into(), 0.0, 1.0, data).unwrap();
        clip.settings.enabled = true;
        clip
    }

    #[test]
    fn preview_mixes_only_when_enabled() {
        let mut data = MidiTakeData::new_midi_take(vec![note()], 1.0, "x".into());
        data.preview.enabled = true;
        let schedule = MidiPreviewSchedule {
            clips: vec![clip_for(&data)],
        };
        let mut out = vec![0.0f32; 4800];
        mix_into_interleaved(&mut out, &schedule, 0, 2400, 48_000);
        assert!(out.iter().any(|v| v.abs() > 1e-6));
        data.preview.enabled = false;
        assert!(clip_from_take("c".into(), 0.0, 1.0, &data).is_none());
    }

    #[test]
    fn preview_is_silent_before_the_note() {
        let mut data = MidiTakeData::new_midi_take(vec![note()], 1.0, "x".into());
        data.preview.enabled = true;
        let schedule = MidiPreviewSchedule {
            clips: vec![clip_from_take("c".into(), 1.0, 1.0, &data).unwrap()],
        };
        let mut out = vec![0.0f32; 4800];
        mix_into_interleaved(&mut out, &schedule, 0, 2400, 48_000);
        assert!(out.iter().all(|v| v.abs() < 1e-9));
    }

    #[test]
    fn preview_respects_clip_gain_and_mute() {
        let mut data = MidiTakeData::new_midi_take(vec![note()], 1.0, "x".into());
        data.preview.enabled = true;
        let mut clip = clip_for(&data);
        clip.gain = 0.0;
        let schedule = MidiPreviewSchedule { clips: vec![clip] };
        let mut out = vec![0.0f32; 4800];
        mix_into_interleaved(&mut out, &schedule, 0, 2400, 48_000);
        assert!(out.iter().all(|v| v.abs() < 1e-9));
    }

    #[test]
    fn preview_loop_repeats_notes_and_carries_across_the_wrap() {
        let note = MidiNoteEvent {
            start_sec: 0.0,
            end_sec: 0.1,
            note: 60.0,
            velocity: 127,
            channel: 0,
            muted: false,
            release_velocity: 0,
        };
        let mut data = MidiTakeData::new_midi_take(vec![note], 0.2, "x".into());
        data.preview.enabled = true;
        let mut clip = clip_for(&data);
        clip.length_sec = 0.6;
        clip.loop_enabled = true;
        clip.loop_period_sec = Some(0.2);
        clip.source_start_sec = 0.0;
        clip.source_end_sec = 0.2;
        let schedule = MidiPreviewSchedule { clips: vec![clip] };
        let mut out = vec![0.0f32; 28_800];
        mix_into_interleaved(&mut out, &schedule, 0, 14_400, 48_000);
        let first = out[..4_800].iter().any(|v| v.abs() > 1e-6);
        let second = out[4_800..9_600].iter().any(|v| v.abs() > 1e-6);
        assert!(first && second, "each loop cycle must sound");
    }

    #[test]
    fn preview_fade_is_applied() {
        let mut data = MidiTakeData::new_midi_take(vec![note()], 1.0, "x".into());
        data.preview.enabled = true;
        let mut clip = clip_for(&data);
        clip.fade_in_sec = 1.0;
        clip.fade_in_lut = Some(global_fade_lut(0.0, 0.0, false));
        let schedule = MidiPreviewSchedule { clips: vec![clip] };
        let mut out = vec![0.0f32; 4800];
        mix_into_interleaved(&mut out, &schedule, 0, 2400, 48_000);
        assert!(out.iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn midi_take_role_does_not_change_preview_eligibility() {
        let mut data = MidiTakeData::new_midi_take(vec![note()], 1.0, "x".into());
        data.role = MidiTakeRole::PitchReference;
        data.preview.enabled = true;
        assert!(clip_from_take("c".into(), 0.0, 1.0, &data).is_some());
    }
    #[test]
    fn timeline_schedule_uses_clip_and_track_parameters() {
        let mut timeline = crate::state::TimelineState::default();
        let mut data = MidiTakeData::new_midi_take(vec![note()], 0.2, "x".into());
        data.preview.enabled = true;
        data.preview.gain_db = 0.0;
        let clip_id = timeline.add_clip(
            Some("track_main".to_string()),
            Some("midi".to_string()),
            Some(0.0),
            Some(0.6),
            None,
        );
        let clip = timeline
            .clips
            .iter_mut()
            .find(|clip| clip.id == clip_id)
            .unwrap();
        clip.set_midi_take(data);
        clip.loop_enabled = true;
        clip.source_end_sec = 0.2;
        clip.clip_gain = 0.5;
        timeline.tracks[0].volume = 0.5;
        let schedule = MidiPreviewSchedule::from_timeline(&timeline);
        assert_eq!(schedule.clips.len(), 1);
        assert!((schedule.clips[0].gain - 0.25).abs() < 1e-6);
        assert!(schedule.clips[0].loop_enabled);
        assert_eq!(schedule.clips[0].loop_period_sec, Some(0.2));

        timeline.tracks[0].muted = true;
        assert!(MidiPreviewSchedule::from_timeline(&timeline).is_empty());
    }

}
