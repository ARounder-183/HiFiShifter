//! 从 Take 源域分析生成 base 块。
//!
//! 职责：只读 Take 的 `syllable_analysis` / `game_block_baseline` / `midi_note_data`，
//! 按 active Take 与 Clip 几何投影成 base 块。
//! 关系：`compose` 消费本文件的 `BaseBlock`；本文件不写任何状态。
//! 维护约束：没有分析结果的音频 Clip 绝不伪造“整段一块”。

use super::projection::{normalize_loop_span, project_span, ClipProjection, ProjectedInterval};
use super::types::{BlockLabel, TakeKey, TakeSpan};
use crate::state::{Clip, ClipTake, TimelineState};

#[derive(Clone, Debug)]
pub struct BaseBlock {
    pub id: String,
    pub span: TakeSpan,
    pub z_order: u32,
    pub projected: Vec<ProjectedInterval>,
    pub rate: f64,
    pub projection: ClipProjection,
}

fn stable_id(take: &ClipTake, clip: &Clip, span: &TakeSpan) -> String {
    let identity = format!(
        "{}|{}|{}|{:08}|{:08}|{:?}",
        take.id,
        clip.id,
        span.cycle,
        (span.source_start_sec * 1000.0).round() as i64,
        (span.source_end_sec * 1000.0).round() as i64,
        span.label
    );
    format!("base:{}", blake3::hash(identity.as_bytes()).to_hex())
}

fn source_spans(take: &ClipTake) -> Option<(Vec<(f64, f64)>, BlockLabel)> {
    if let Some(analysis) = take
        .syllable_analysis
        .as_ref()
        .filter(|analysis| !analysis.source_spans_sec.is_empty())
    {
        return Some((
            analysis
                .source_spans_sec
                .iter()
                .map(|span| (span.start_sec, span.end_sec))
                .collect(),
            BlockLabel::Syllable,
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
                .map(|span| (span.start_sec, span.end_sec))
                .collect(),
            BlockLabel::GameBaseline,
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
                .map(|note| (note.start_sec, note.end_sec))
                .collect(),
            BlockLabel::MidiNote,
        ));
    }
    None
}

fn normalized_spans(mut spans: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    spans.retain(|(start, end)| start.is_finite() && end.is_finite() && *end > *start);
    spans.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let mut out: Vec<(f64, f64)> = Vec::new();
    for (start, end) in spans {
        match out.last_mut() {
            Some(last) if start < last.1 - 1e-9 => {
                // 重叠：保留更宽者；等宽时保留先出现者。
                if end - start > last.1 - last.0 {
                    *last = (start, end);
                }
            }
            _ => out.push((start, end)),
        }
    }
    out
}

pub fn base_blocks_for_root(
    timeline: &TimelineState,
    root_track_id: &str,
    frame_period_ms: f64,
) -> Vec<BaseBlock> {
    let mut out = Vec::new();
    for (z_order, clip) in timeline.clips.iter().enumerate() {
        if timeline.resolve_root_track_id(&clip.track_id).as_deref() != Some(root_track_id) {
            continue;
        }
        let take = if clip.takes.is_empty() {
            ClipTake::from_clip(clip)
        } else {
            clip.active_take().clone()
        };
        let Some((spans, label)) = source_spans(&take) else {
            continue;
        };
        let projection = ClipProjection::from_clip_take(clip, &take);
        for (index, (start_sec, end_sec)) in normalized_spans(spans).into_iter().enumerate() {
            let identity = format!(
                "span:{}:{}:{:08}",
                take.id,
                index,
                (start_sec * 1000.0).round() as i64
            );
            let base_span = TakeSpan {
                clip_id: clip.id.clone(),
                take_id: take.id.clone(),
                cycle: 0,
                source_start_sec: start_sec,
                source_end_sec: end_sec,
                // 稳定源域身份：重分析边界微移时仍能按重叠比例匹配。
                syllable_id: Some(identity.clone()),
                label,
            };
            let parts = normalize_loop_span(&projection, &base_span);
            let parts_len = parts.len();
            for (part_index, part) in parts.into_iter().enumerate() {
                let syllable_id = if parts_len > 1 {
                    Some(format!("{}:part{}", identity, part_index))
                } else {
                    Some(identity.clone())
                };
                for interval in project_span(&projection, &part, frame_period_ms) {
                    if interval.cycle < part.cycle {
                        continue;
                    }
                    let mut cycle_span = part.clone();
                    cycle_span.cycle = interval.cycle;
                    cycle_span.syllable_id = syllable_id.clone();
                    out.push(BaseBlock {
                        id: stable_id(&take, clip, &cycle_span),
                        span: cycle_span,
                        z_order: z_order as u32,
                        projected: vec![interval],
                        rate: projection.effective_rate,
                        projection: projection.clone(),
                    });
                }
            }
        }
    }
    out
}

/// 当前 root 组是否还有音频 Clip 在等待音节分析。
pub fn root_analysis_pending(timeline: &TimelineState, root_track_id: &str) -> bool {
    timeline.clips.iter().any(|clip| {
        if timeline.resolve_root_track_id(&clip.track_id).as_deref() != Some(root_track_id) {
            return false;
        }
        let take = if clip.takes.is_empty() {
            ClipTake::from_clip(clip)
        } else {
            clip.active_take().clone()
        };
        clip.source_path.is_some()
            && take
                .midi_note_data
                .as_ref()
                .is_none_or(|notes| notes.is_empty())
            && take
                .syllable_analysis
                .as_ref()
                .is_none_or(|analysis| analysis.source_spans_sec.is_empty())
    })
}

pub fn take_key_for_block(
    timeline: &TimelineState,
    clip_id: &str,
    take_id: &str,
) -> Option<TakeKey> {
    let clip = timeline.clips.iter().find(|clip| clip.id == clip_id)?;
    if clip.takes.iter().any(|take| take.id == take_id) {
        Some(TakeKey {
            clip_id: clip_id.to_owned(),
            take_id: take_id.to_owned(),
        })
    } else {
        None
    }
}
