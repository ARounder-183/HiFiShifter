//! MIDI Take 的数据模型、迁移与派生算法。
//!
//! ## 这一层为什么存在
//!
//! 历史上 `ClipTake.midi_note_data` 同时承担两件事：保存音符内容，以及
//! “覆盖这一段原始音高线”。前者是**内容**，后者是**角色**。两者混在一个
//! `Option<Vec<MidiNoteEvent>>` 里，于是任何看到非空音符的消费者都只能猜：
//! 这是一条 MIDI 片段，还是一条音高参考块？
//!
//! 本模块把两者拆开：
//!
//! - [`MidiTakeData`] 是 Take 级 MIDI 内容的权威；
//! - [`MidiTakeRole`] 是它相对音高组装的角色；
//! - `ClipTake::midi_note_data` 降为**旧版本兼容镜像**。
//!
//! 所有时间仍是**源域秒**，与音频 Take、内核渲染和现有 `MidiNoteEvent` 完全
//! 同一把尺子。这里刻意不引入 beats 时钟：MIDI 导入后应当和音频一样，在工程
//! BPM 改变时保持绝对时间不变。

use crate::midi_import::MidiNoteEvent;
use serde::{Deserialize, Serialize};

/// MIDI Take 的用途。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MidiTakeRole {
    /// 普通 MIDI 片段：显示音符、可试听，但**不**参与 `pitch_orig` 组装。
    #[default]
    MidiTake,
    /// 音高参考块：把 MIDI 压成一条（或按通道选择的）单声部参考线，覆盖原声。
    PitchReference,
}

/// 音高参考的压平策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MidiFlattenMode {
    /// 同时发声的音符取最高音（旧行为）。
    #[default]
    Highest,
    /// 同时发声的音符取最低音。
    Lowest,
    /// 只取指定通道，再在通道内取最高音。
    Channel(u8),
}

/// 音高参考角色的选项。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PitchReferenceOptions {
    /// 是否填满首尾音符之间的空隙（旧 `midi_fill_gaps`）。
    #[serde(default)]
    pub fill_gaps: bool,
    /// 复调压平策略。
    #[serde(default)]
    pub flatten: MidiFlattenMode,
}

/// 不影响音符配对的 MIDI 表情/控制事件。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MidiExpressionEvent {
    pub time_sec: f64,
    pub kind: MidiExpressionKind,
}

/// 一类 MIDI 表情事件。保留原始语义，不执行任何设备逻辑。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MidiExpressionKind {
    ControlChange {
        channel: u8,
        controller: u8,
        value: u8,
    },
    /// -8192..=8191（MIDI 1.0 的 14-bit 弯音，中心为 0）。
    PitchBend {
        channel: u8,
        value: i16,
    },
    ProgramChange {
        channel: u8,
        program: u8,
    },
    ChannelAftertouch {
        channel: u8,
        value: u8,
    },
    PolyAftertouch {
        channel: u8,
        key: u8,
        value: u8,
    },
    SysEx {
        data: Vec<u8>,
    },
    Meta {
        kind: u8,
        data: Vec<u8>,
    },
}

/// MIDI 试听设置。只影响监听，不进入 `pitch_orig`、混音导出或渲染缓存。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MidiPreviewSettings {
    #[serde(default = "default_preview_enabled")]
    pub enabled: bool,
    #[serde(default = "default_preview_gain_db")]
    pub gain_db: f32,
    #[serde(default)]
    pub timbre: MidiPreviewTimbre,
    #[serde(default = "default_preview_polyphony")]
    pub polyphony: u16,
}

fn default_preview_enabled() -> bool {
    true
}
fn default_preview_gain_db() -> f32 {
    -12.0
}
fn default_preview_polyphony() -> u16 {
    32
}
impl Default for MidiPreviewSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            gain_db: -12.0,
            timbre: MidiPreviewTimbre::default(),
            polyphony: 32,
        }
    }
}

/// 试听音色。不是 GM 还原，只提供三种轻量参考音色。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MidiPreviewTimbre {
    #[default]
    SoftSine,
    Triangle,
    Saw,
}

pub fn default_pitch_bend_ranges() -> [f32; 16] {
    [2.0; 16]
}

/// Take 级 MIDI 内容的**权威**表示。
///
/// 音符的 `start_sec` / `end_sec` 是 Take 源域秒；允许重叠（复调）。
/// `length_sec` 是内容总长度，空 MIDI item 也仍然可循环/裁剪。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MidiTakeData {
    #[serde(default)]
    pub role: MidiTakeRole,
    #[serde(default)]
    pub notes: Vec<MidiNoteEvent>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub expression: Vec<MidiExpressionEvent>,
    #[serde(default)]
    pub length_sec: f64,
    #[serde(default)]
    pub source_bpm: f64,
    #[serde(default)]
    pub ticks_per_beat: u16,
    #[serde(default = "default_pitch_bend_ranges")]
    pub pitch_bend_ranges: [f32; 16],
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tempo_map: Vec<MidiTempoEvent>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub time_signatures: Vec<MidiTimeSignatureEvent>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub key_signatures: Vec<MidiKeySignatureEvent>,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub pitch_reference: PitchReferenceOptions,
    /// 运行时缓存键；不落盘。加载后由 [`Self::refresh_hash`] 重算。
    #[serde(default, skip_serializing)]
    pub content_hash: String,
    /// 试听设置。与角色正交：音高参考默认不试听，MIDI Take 默认试听。
    #[serde(default)]
    pub preview: MidiPreviewSettings,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MidiTempoEvent {
    pub time_sec: f64,
    pub bpm: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MidiTimeSignatureEvent {
    pub time_sec: f64,
    pub numerator: u32,
    pub denominator: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MidiKeySignatureEvent {
    pub time_sec: f64,
    pub sharps_flats: i8,
    #[serde(default)]
    pub minor: bool,
}

impl MidiTakeData {
    /// 从旧音高参考音符迁移。旧数据没有角色概念，因此固定为音高参考。
    pub fn from_legacy_notes(notes: Vec<MidiNoteEvent>, fill_gaps: bool, name: String) -> Self {
        let length_sec = notes.iter().map(|n| n.end_sec).fold(0.0f64, f64::max);
        Self {
            role: MidiTakeRole::PitchReference,
            notes,
            expression: Vec::new(),
            length_sec,
            source_bpm: 0.0,
            ticks_per_beat: 0,
            pitch_bend_ranges: default_pitch_bend_ranges(),
            tempo_map: Vec::new(),
            time_signatures: Vec::new(),
            key_signatures: Vec::new(),
            name,
            pitch_reference: PitchReferenceOptions {
                fill_gaps,
                flatten: MidiFlattenMode::Highest,
            },
            content_hash: String::new(),
            preview: MidiPreviewSettings {
                enabled: false,
                ..MidiPreviewSettings::default()
            },
        }
    }

    pub fn new_midi_take(notes: Vec<MidiNoteEvent>, length_sec: f64, name: String) -> Self {
        let mut data = Self {
            role: MidiTakeRole::MidiTake,
            notes,
            length_sec,
            name,
            ..Default::default()
        };
        data.normalize();
        data.refresh_hash();
        data
    }

    pub fn is_pitch_reference(&self) -> bool {
        self.role == MidiTakeRole::PitchReference
    }

    pub fn is_midi_take(&self) -> bool {
        self.role == MidiTakeRole::MidiTake
    }

    pub fn effective_length_sec(&self) -> f64 {
        if self.length_sec.is_finite() && self.length_sec > 1e-9 {
            return self.length_sec;
        }
        self.notes
            .iter()
            .map(|n| n.end_sec)
            .fold(0.0f64, f64::max)
            .max(0.0)
    }

    /// 旧版本兼容镜像只允许承载音高参考；普通 MIDI Take 必须显式返回空。
    pub fn legacy_mirror_notes(&self) -> Option<Vec<MidiNoteEvent>> {
        if self.is_pitch_reference() {
            Some(self.baked_notes())
        } else {
            None
        }
    }

    /// 把所有时间/音高/力度整理成合法、稳定排序的数据。坏事件丢弃。
    pub fn normalize(&mut self) {
        self.notes.retain(|n| {
            n.start_sec.is_finite()
                && n.end_sec.is_finite()
                && n.start_sec >= 0.0
                && n.end_sec > n.start_sec
                && n.note.is_finite()
                && (0.0..=127.0).contains(&n.note)
                && n.channel < 16
                && n.velocity <= 127
        });
        for note in &mut self.notes {
            note.end_sec = note.end_sec.max(note.start_sec + 1e-6);
        }
        self.notes.sort_by(|a, b| {
            a.start_sec
                .total_cmp(&b.start_sec)
                .then_with(|| a.note.total_cmp(&b.note))
                .then_with(|| a.channel.cmp(&b.channel))
        });
        self.expression.retain(|event| {
            event.time_sec.is_finite()
                && event.time_sec >= 0.0
                && match &event.kind {
                    MidiExpressionKind::PitchBend { channel, value } => {
                        *channel < 16 && (-8192..=8191).contains(value)
                    }
                    MidiExpressionKind::ControlChange {
                        channel,
                        controller,
                        value,
                    } => *channel < 16 && *controller <= 127 && *value <= 127,
                    MidiExpressionKind::ProgramChange { channel, program } => {
                        *channel < 16 && *program <= 127
                    }
                    MidiExpressionKind::ChannelAftertouch { channel, value } => {
                        *channel < 16 && *value <= 127
                    }
                    MidiExpressionKind::PolyAftertouch { channel, key, value } => {
                        *channel < 16 && *key <= 127 && *value <= 127
                    }
                    MidiExpressionKind::SysEx { .. } | MidiExpressionKind::Meta { .. } => true,
                }
        });
        self.expression
            .sort_by(|a, b| a.time_sec.total_cmp(&b.time_sec));
        self.length_sec = self.effective_length_sec();
        for range in &mut self.pitch_bend_ranges {
            if !range.is_finite() || *range <= 0.0 || *range > 96.0 {
                *range = 2.0;
            }
        }
        if !self.source_bpm.is_finite() || self.source_bpm <= 0.0 {
            self.source_bpm = 0.0;
        }
        if self.name.is_empty() {
            self.name = "MIDI".to_string();
        }
    }

    /// 重新计算内容哈希（音符 + 表情 + 长度 + tempo/拍号/调号）。
    pub fn refresh_hash(&mut self) {
        let canonical = serde_json::json!({
            "notes": self.notes,
            "expression": self.expression,
            "length_sec": self.length_sec,
            "source_bpm": self.source_bpm,
            "ticks_per_beat": self.ticks_per_beat,
            "tempo_map": self.tempo_map,
            "time_signatures": self.time_signatures,
            "key_signatures": self.key_signatures,
        });
        let bytes = serde_json::to_vec(&canonical).unwrap_or_default();
        self.content_hash = blake3::hash(&bytes).to_hex().to_string();
    }

    /// 把弯音烘焙进音符并切分，得到旧音高路径可直接消费的单段音高事件。
    ///
    /// 新路径的音符保存**名义音高**；任何需要“音高线”的消费者（旧镜像、
    /// 参数编辑器基线、音高参考压平）都通过本函数取得已经含弯音的值。
    pub fn baked_notes(&self) -> Vec<MidiNoteEvent> {
        if !self
            .expression
            .iter()
            .any(|event| matches!(event.kind, MidiExpressionKind::PitchBend { .. }))
        {
            return self.notes.clone();
        }
        let mut out = Vec::new();
        for note in &self.notes {
            let channel = note.channel.min(15);
            let range = self.pitch_bend_ranges[channel as usize];
            let mut points: Vec<(f64, f32)> = self
                .expression
                .iter()
                .filter_map(|event| match event.kind {
                    MidiExpressionKind::PitchBend {
                        channel: event_channel,
                        value,
                    } if event_channel == channel
                        && event.time_sec <= note.end_sec
                        && event.time_sec > note.start_sec =>
                    {
                        Some((
                            event.time_sec,
                            (value as f32 / 8192.0 * range).clamp(-96.0, 96.0),
                        ))
                    }
                    _ => None,
                })
                .collect();
            points.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut cursor = note.start_sec;
            let mut bend = self
                .expression
                .iter()
                .filter(|event| event.time_sec <= note.start_sec)
                .filter_map(|event| match event.kind {
                    MidiExpressionKind::PitchBend {
                        channel: event_channel,
                        value,
                    } if event_channel == channel => {
                        Some((value as f32 / 8192.0 * range).clamp(-96.0, 96.0))
                    }
                    _ => None,
                })
                .last()
                .unwrap_or(0.0);
            for (time, next_bend) in points {
                if time > cursor + 1e-9 {
                    out.push(MidiNoteEvent {
                        start_sec: cursor,
                        end_sec: time,
                        note: (note.note + bend).clamp(0.0, 127.0),
                        velocity: note.velocity,
                        channel,
                        muted: note.muted,
                        release_velocity: note.release_velocity,
                    });
                }
                cursor = time;
                bend = next_bend;
            }
            if note.end_sec > cursor + 1e-9 {
                out.push(MidiNoteEvent {
                    start_sec: cursor,
                    end_sec: note.end_sec,
                    note: (note.note + bend).clamp(0.0, 127.0),
                    velocity: note.velocity,
                    channel,
                    muted: note.muted,
                    release_velocity: note.release_velocity,
                });
            }
        }
        out.sort_by(|a, b| a.start_sec.total_cmp(&b.start_sec));
        out
    }

    /// 按倍率缩放所有时间（BPM 模式/工程 BPM 对齐用）。
    pub fn scale_time(&mut self, factor: f64) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        for note in &mut self.notes {
            note.start_sec *= factor;
            note.end_sec *= factor;
        }
        for event in &mut self.expression {
            event.time_sec *= factor;
        }
        for event in &mut self.tempo_map {
            event.time_sec *= factor;
        }
        for event in &mut self.time_signatures {
            event.time_sec *= factor;
        }
        for event in &mut self.key_signatures {
            event.time_sec *= factor;
        }
        self.length_sec *= factor;
        self.refresh_hash();
    }

    /// 去掉开头空白：所有事件减去第一个音符/事件的时间（不会产生负时间）。
    pub fn trim_leading_silence(&mut self) {
        let first = self
            .notes
            .iter()
            .map(|note| note.start_sec)
            .chain(self.expression.iter().map(|event| event.time_sec))
            .fold(f64::INFINITY, f64::min);
        if !first.is_finite() {
            return;
        }
        let shift = |time: &mut f64| {
            *time = (*time - first).max(0.0);
        };
        for note in &mut self.notes {
            shift(&mut note.start_sec);
            shift(&mut note.end_sec);
        }
        for event in &mut self.expression {
            shift(&mut event.time_sec);
        }
        for event in &mut self.tempo_map {
            shift(&mut event.time_sec);
        }
        for event in &mut self.time_signatures {
            shift(&mut event.time_sec);
        }
        for event in &mut self.key_signatures {
            shift(&mut event.time_sec);
        }
        self.length_sec = (self.length_sec - first).max(0.0);
        self.refresh_hash();
    }

    /// 把另一条轨道的 MIDI 内容并入本 Take（多轨合并）。保留通道与表情。
    pub fn merge_from(&mut self, other: &MidiTakeData, time_offset_sec: f64) {
        for note in &other.notes {
            let mut note = *note;
            note.start_sec += time_offset_sec;
            note.end_sec += time_offset_sec;
            self.notes.push(note);
        }
        for event in &other.expression {
            let mut event = event.clone();
            event.time_sec += time_offset_sec;
            self.expression.push(event);
        }
        self.length_sec = self
            .length_sec
            .max(other.effective_length_sec() + time_offset_sec);
        self.normalize();
        self.refresh_hash();
    }

    /// 按 [`MidiFlattenMode`] 压平成不重叠的单声部事件。
    ///
    /// 这是“音高参考”的显式烘焙操作：它只影响派生视图/旧镜像，不改写
    /// [`Self::notes`]，因此切换角色或取消音高参考永远不会破坏原始 MIDI。
    pub fn flatten_to_note_events(&self) -> Vec<MidiNoteEvent> {
        let source = self.baked_notes();
        if source.is_empty() {
            return Vec::new();
        }
        let mut bounds: Vec<f64> = Vec::with_capacity(source.len() * 2);
        for note in &source {
            bounds.push(note.start_sec);
            bounds.push(note.end_sec);
        }
        bounds.sort_by(f64::total_cmp);
        bounds.dedup_by(|a, b| (*a - *b).abs() <= 1e-9);
        let mut out: Vec<MidiNoteEvent> = Vec::new();
        for pair in bounds.windows(2) {
            let start = pair[0];
            let end = pair[1];
            if end <= start + 1e-9 {
                continue;
            }
            let mut active: Vec<&MidiNoteEvent> = source
                .iter()
                .filter(|note| note.start_sec < end - 1e-9 && note.end_sec > start + 1e-9)
                .filter(|note| !note.muted)
                .collect();
            if active.is_empty() {
                continue;
            }
            if let MidiFlattenMode::Channel(want) = self.pitch_reference.flatten {
                active.retain(|note| note.channel == want.min(15));
                if active.is_empty() {
                    continue;
                }
            }
            let selected = match self.pitch_reference.flatten {
                MidiFlattenMode::Lowest => active
                    .iter()
                    .min_by(|a, b| a.note.total_cmp(&b.note))
                    .copied(),
                _ => active
                    .iter()
                    .max_by(|a, b| a.note.total_cmp(&b.note))
                    .copied(),
            };
            let Some(selected) = selected else {
                continue;
            };
            if let Some(last) = out.last_mut() {
                if last.channel == selected.channel
                    && (last.end_sec - start).abs() <= 1e-6
                    && (last.note - selected.note).abs() <= 1e-6
                {
                    last.end_sec = end;
                    continue;
                }
            }
            out.push(MidiNoteEvent {
                start_sec: start,
                end_sec: end,
                note: selected.note,
                velocity: selected.velocity,
                channel: selected.channel,
                muted: false,
                release_velocity: selected.release_velocity,
            });
        }
        out
    }
}

impl Default for MidiTakeData {
    fn default() -> Self {
        Self {
            role: MidiTakeRole::MidiTake,
            notes: Vec::new(),
            expression: Vec::new(),
            length_sec: 0.0,
            source_bpm: 0.0,
            ticks_per_beat: 0,
            pitch_bend_ranges: default_pitch_bend_ranges(),
            tempo_map: Vec::new(),
            time_signatures: Vec::new(),
            key_signatures: Vec::new(),
            name: "MIDI".to_string(),
            pitch_reference: PitchReferenceOptions::default(),
            content_hash: String::new(),
            preview: MidiPreviewSettings::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(start: f64, end: f64, pitch: f32, channel: u8) -> MidiNoteEvent {
        MidiNoteEvent {
            start_sec: start,
            end_sec: end,
            note: pitch,
            velocity: 100,
            channel,
            muted: false,
            release_velocity: 0,
        }
    }

    #[test]
    fn legacy_notes_migrate_to_pitch_reference_even_when_empty() {
        let data = MidiTakeData::from_legacy_notes(vec![note(0.0, 1.0, 60.0, 0)], true, "x".into());
        assert_eq!(data.role, MidiTakeRole::PitchReference);
        assert!(data.pitch_reference.fill_gaps);
        assert_eq!(data.legacy_mirror_notes().unwrap().len(), 1);
    }

    #[test]
    fn midi_take_never_writes_a_legacy_mirror() {
        let data = MidiTakeData::new_midi_take(vec![note(0.0, 1.0, 60.0, 0)], 1.0, "x".into());
        assert!(data.legacy_mirror_notes().is_none());
    }

    #[test]
    fn flatten_highest_and_lowest_are_role_aware() {
        let mut data = MidiTakeData::new_midi_take(
            vec![note(0.0, 1.0, 60.0, 0), note(0.25, 0.75, 72.0, 1)],
            1.0,
            "x".into(),
        );
        assert_eq!(data.flatten_to_note_events()[1].note, 72.0);
        data.pitch_reference.flatten = MidiFlattenMode::Lowest;
        assert!(data
            .flatten_to_note_events()
            .iter()
            .all(|event| event.note <= 60.0));
    }

    #[test]
    fn pitch_bend_is_baked_without_mutating_the_source_notes() {
        let mut data = MidiTakeData::new_midi_take(vec![note(0.0, 1.0, 60.0, 0)], 1.0, "x".into());
        data.expression.push(MidiExpressionEvent {
            time_sec: 0.5,
            kind: MidiExpressionKind::PitchBend {
                channel: 0,
                value: 4096,
            },
        });
        let baked = data.baked_notes();
        assert_eq!(baked.len(), 2);
        assert_eq!(data.notes[0].note, 60.0);
        assert!(baked[1].note > 60.0);
    }

    #[test]
    fn normalize_drops_non_finite_and_out_of_range_events() {
        let mut data = MidiTakeData::default();
        data.notes.push(note(f64::NAN, 1.0, 60.0, 0));
        data.notes.push(note(0.0, 1.0, 200.0, 0));
        data.notes.push(note(0.0, 1.0, 60.0, 0));
        data.length_sec = 0.0;
        data.normalize();
        assert_eq!(data.notes.len(), 1);
        assert_eq!(data.length_sec, 1.0);
    }
}
