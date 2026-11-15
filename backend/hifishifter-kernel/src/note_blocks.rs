//! 音符块来源与工程时间轴投影的共享实现。
//!
//! 参数块与静音检测必须消费同一份来源优先级：
//! 1. `syllable_analysis`（GAME/VAD 分析结果）；
//! 2. `game_block_baseline`；
//! 3. `midi_note_data`。
//!
//! 这里只读，不触发分析、不解码音频，也不写 timeline。

use crate::param_blocks::projection::{project_span_part, split_loop_span_parts, ClipProjection};
use crate::param_blocks::types::{BlockLabel, TakeSpan};
use crate::state::{Clip, ClipTake};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteBlockSource {
    Syllable,
    GameBaseline,
    MidiNote,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NoteBlockSpan {
    pub start_sec: f64,
    pub end_sec: f64,
    /// Representative MIDI note from the source analysis; MIDI notes carry it,
    /// audio estimates may have none when the span has no voiced frames.
    pub midi_note: Option<f32>,
    /// Stable full-source event identity; independent of Clip trimming.
    pub event_id: Option<String>,
}

/// 与 param_blocks/base.rs 完全相同的来源优先级。
pub fn source_note_spans(take: &ClipTake) -> Option<(Vec<NoteBlockSpan>, NoteBlockSource)> {
    if let Some(analysis) = take
        .syllable_analysis
        .as_ref()
        .filter(|analysis| !analysis.source_spans_sec.is_empty())
    {
        return Some((
            analysis
                .source_spans_sec
                .iter()
                .map(|span| NoteBlockSpan {
                    start_sec: span.start_sec,
                    end_sec: span.end_sec,
                    midi_note: span.midi_note,
                    event_id: span.event_id.clone().or_else(|| {
                        Some(crate::pitch_analysis::source_event_id(
                            span.start_sec,
                            span.end_sec,
                        ))
                    }),
                })
                .collect(),
            NoteBlockSource::Syllable,
        ));
    }
    if let Some(baseline) = take
        .game_block_baseline
        .as_ref()
        .filter(|baseline| !baseline.source_spans_sec.is_empty())
    {
        return Some((
            baseline
                .source_spans_sec
                .iter()
                .map(|span| NoteBlockSpan {
                    start_sec: span.start_sec,
                    end_sec: span.end_sec,
                    midi_note: span.midi_note,
                    event_id: span.event_id.clone().or_else(|| {
                        Some(crate::pitch_analysis::source_event_id(
                            span.start_sec,
                            span.end_sec,
                        ))
                    }),
                })
                .collect(),
            NoteBlockSource::GameBaseline,
        ));
    }
    if let Some(notes) = take
        .midi_note_data
        .as_ref()
        .filter(|notes| !notes.is_empty())
    {
        return Some((
            notes
                .iter()
                .filter(|note| note.end_sec > note.start_sec)
                .map(|note| NoteBlockSpan {
                    start_sec: note.start_sec,
                    end_sec: note.end_sec,
                    midi_note: Some(note.note),
                    event_id: Some(crate::pitch_analysis::source_event_id(
                        note.start_sec,
                        note.end_sec,
                    )),
                })
                .collect(),
            NoteBlockSource::MidiNote,
        ));
    }
    None
}

/// 源域音符块 → 工程时间轴绝对秒（复用参数块投影）。
pub fn projected_note_spans(clip: &Clip, take: &ClipTake, frame_period_ms: f64) -> Vec<(f64, f64)> {
    let Some((spans, _source)) = source_note_spans(take) else {
        return Vec::new();
    };
    let projection = ClipProjection::from_clip_take(clip, take);
    let frame_period_ms = frame_period_ms.max(0.1);
    let mut out = Vec::new();
    for span in spans {
        let base = TakeSpan {
            clip_id: clip.id.clone(),
            take_id: take.id.clone(),
            cycle: 0,
            source_start_sec: span.start_sec,
            source_end_sec: span.end_sec,
            syllable_id: None,
            label: BlockLabel::Syllable,
        };
        for part in split_loop_span_parts(&projection, &base) {
            for interval in project_span_part(&projection, &part, frame_period_ms) {
                if interval.cycle < part.span.cycle {
                    continue;
                }
                let start = interval.start_frame as f64 * frame_period_ms / 1000.0;
                let end = interval.end_frame_exclusive as f64 * frame_period_ms / 1000.0;
                let clip_start = clip.start_sec.max(0.0);
                let clip_end = clip_start + clip.length_sec.max(0.0);
                let start = start.max(clip_start);
                let end = end.min(clip_end);
                if end > start {
                    out.push((start, end));
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::TimelineState;

    #[test]
    fn source_priority_prefers_syllable_then_baseline_then_midi() {
        let mut timeline: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"root","name":"Root","order":0}],
            "clips":[],"bpm":120,"project_sec":4
        }))
        .unwrap();
        let clip_id = timeline.add_clip(
            Some("root".into()),
            Some("clip".into()),
            Some(0.0),
            Some(1.0),
            None,
        );
        let clip = timeline
            .clips
            .iter_mut()
            .find(|clip| clip.id == clip_id)
            .unwrap();
        let take = clip.takes.first_mut().unwrap();
        take.midi_note_data = Some(vec![crate::midi_import::MidiNoteEvent {
            start_sec: 0.0,
            end_sec: 0.5,
            note: 60.0,
            velocity: 100,
            channel: 0,
        }]);
        assert_eq!(
            source_note_spans(take).unwrap().1,
            NoteBlockSource::MidiNote
        );
        take.game_block_baseline = Some(crate::param_blocks::types::GameBlockBaseline {
            revision: 1,
            algorithm: "test".into(),
            algorithm_version: 1,
            source_spans_sec: vec![crate::pitch_analysis::SyllableSpan {
                start_sec: 0.1,
                end_sec: 0.2,
                midi_note: None,
                event_id: Some(crate::pitch_analysis::source_event_id(0.1, 0.2)),
            }],
            analyzed_window_sec: [0.0, 0.0],
        });
        assert_eq!(
            source_note_spans(take).unwrap().1,
            NoteBlockSource::GameBaseline
        );
        take.syllable_analysis = Some(crate::pitch_analysis::TakeSyllableAnalysis {
            algorithm: "test".into(),
            algorithm_version: 1,
            source_digest: "d".into(),
            source_spans_sec: vec![crate::pitch_analysis::SyllableSpan {
                start_sec: 0.2,
                end_sec: 0.3,
                midi_note: None,
                event_id: Some(crate::pitch_analysis::source_event_id(0.2, 0.3)),
            }],
            analyzed_window_sec: [0.0, 0.0],
        });
        assert_eq!(
            source_note_spans(take).unwrap().1,
            NoteBlockSource::Syllable
        );
    }
}
