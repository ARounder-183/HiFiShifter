//! MIDI 试听：只用于编辑参考的轻量合成。
//!
//! 这里没有采样器、没有 GM 还原、没有外部插件调用。它只回答一个问题：
//! “这一段 MIDI 大致是什么旋律/和声？”因此合成器故意保持极简（正弦/三角/锯齿），
//! 默认低音量，并且与 mixdown、导出、渲染缓存完全隔离。
//!
//! 合成是无状态的：第 n 帧的相位由“该帧相对音符起点的秒数”直接求出，
//! 不保存振荡器/包络状态。这样 seek、loop、块边界都不会留下尾音或爆音，
//! 也满足音频线程“无分配、无锁”的要求。

use crate::midi_take::{MidiExpressionKind, MidiPreviewTimbre, MidiTakeData};
use crate::state::TimelineState;

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
}

#[derive(Debug, Clone)]
pub struct PreviewNote {
    pub start_sec: f64,
    pub end_sec: f64,
    pub note: f32,
    pub velocity: u8,
    pub channel: u8,
    /// 同一通道的弯音事件（时间, 半音偏移），已按音符窗口裁剪。
    pub bends: Vec<(f64, f32)>,
    pub muted: bool,
}

impl MidiPreviewSchedule {
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// 从时间线收集启用试听的 MIDI Take。
    ///
    /// 只读 role 无关：音高参考也可以试听，但默认关闭；普通 MIDI Take 默认开启。
    pub fn from_timeline(timeline: &TimelineState) -> Self {
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
            let length = data.effective_length_sec();
            if !(length.is_finite() && length > 1e-9) {
                continue;
            }
            let rate = if clip.playback_rate.is_finite() && clip.playback_rate > 1e-6 {
                clip.playback_rate as f64
            } else {
                1.0
            };
            let notes = data
                .notes
                .iter()
                .filter(|note| !note.muted)
                .map(|note| {
                    let bends = data
                        .expression
                        .iter()
                        .filter_map(|event| match event.kind {
                            MidiExpressionKind::PitchBend { channel, value }
                                if channel == note.channel
                                    && event.time_sec >= note.start_sec
                                    && event.time_sec <= note.end_sec =>
                            {
                                Some((
                                    event.time_sec,
                                    value as f32 / 8192.0
                                        * data.pitch_bend_ranges[note.channel.min(15) as usize],
                                ))
                            }
                            _ => None,
                        })
                        .collect();
                    PreviewNote {
                        start_sec: note.start_sec,
                        end_sec: note.end_sec,
                        note: note.note,
                        velocity: note.velocity,
                        channel: note.channel,
                        bends,
                        muted: note.muted,
                    }
                })
                .collect();
            clips.push(MidiPreviewClip {
                clip_id: clip.id.clone(),
                start_sec: clip.start_sec.max(0.0),
                length_sec: clip.length_sec.max(0.0),
                source_start_sec: clip.source_start_sec,
                source_end_sec: clip.source_end_sec,
                playback_rate: rate,
                reversed: clip.reversed,
                loop_enabled: clip.loop_enabled,
                loop_period_sec: clip
                    .loop_period_sec
                    .filter(|value| value.is_finite() && *value > 1e-9)
                    .or(Some(length)),
                notes,
                settings: data.preview,
            });
        }
        Self { clips }
    }
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
    let total = clip.loop_period_sec?;
    if total <= 1e-9 {
        return None;
    }
    let rate = clip.playback_rate.max(1e-6);
    if clip.loop_enabled {
        let t = local_sec * rate;
        let anchored = if clip.reversed {
            clip.source_end_sec - t
        } else {
            clip.source_start_sec + t
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

fn voice_sample(note: &PreviewNote, src_sec: f64, timbre: MidiPreviewTimbre) -> f32 {
    if src_sec < note.start_sec || src_sec >= note.end_sec {
        return 0.0;
    }
    let age = (src_sec - note.start_sec).max(0.0);
    let remain = (note.end_sec - src_sec).max(0.0);
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
        .take_while(|(time, _)| *time <= src_sec)
        .last()
        .map(|(_, semitones)| *semitones)
        .unwrap_or(0.0);
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
        let gain = gain_from_db(clip.settings.gain_db);
        if gain <= 0.0 {
            continue;
        }
        for frame in overlap_start..overlap_end {
            let local = (frame - start) as f64 / sr;
            let Some(src) = source_time(clip, local) else {
                continue;
            };
            let mut sample = 0.0f32;
            let mut voices = 0u16;
            for note in &clip.notes {
                if voices >= clip.settings.polyphony.max(1) {
                    break;
                }
                if src >= note.start_sec && src < note.end_sec {
                    voices = voices.saturating_add(1);
                    sample += voice_sample(note, src, clip.settings.timbre);
                }
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
        let gain = gain_from_db(clip.settings.gain_db);
        if gain <= 0.0 {
            continue;
        }
        for frame in overlap_start..overlap_end {
            let local = (frame - start) as f64 / sr;
            let Some(src) = source_time(clip, local) else {
                continue;
            };
            let mut sample = 0.0f32;
            let mut voices = 0u16;
            for note in &clip.notes {
                if voices >= clip.settings.polyphony.max(1) {
                    break;
                }
                if src >= note.start_sec && src < note.end_sec {
                    voices = voices.saturating_add(1);
                    sample += voice_sample(note, src, clip.settings.timbre);
                }
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
    Some(MidiPreviewClip {
        clip_id,
        start_sec,
        length_sec,
        source_start_sec: 0.0,
        source_end_sec: length_sec.max(data.effective_length_sec()),
        playback_rate: 1.0,
        reversed: false,
        loop_enabled: false,
        loop_period_sec: Some(length_sec.max(data.effective_length_sec())),
        notes: data
            .notes
            .iter()
            .map(|note| PreviewNote {
                start_sec: note.start_sec,
                end_sec: note.end_sec,
                note: note.note,
                velocity: note.velocity,
                channel: note.channel,
                bends: Vec::new(),
                muted: note.muted,
            })
            .collect(),
        settings: data.preview,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi_import::MidiNoteEvent;
    use crate::midi_take::MidiTakeData;

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

    #[test]
    fn preview_mixes_only_when_enabled() {
        let mut data = MidiTakeData::new_midi_take(vec![note()], 1.0, "x".into());
        data.preview.enabled = true;
        let schedule = MidiPreviewSchedule {
            clips: vec![clip_from_take("c".into(), 0.0, 1.0, &data).unwrap()],
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
}