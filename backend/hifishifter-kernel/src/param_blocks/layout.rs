//! 参数块视图的布局键。
//!
//! 职责：回答“这份视图由哪份 Clip/Take 几何与哪份分析生成”。
//! 关系：`compose` 把结果写进 `ParamBlockView.layout_key`；缓存与前端冲突检测共用。
//! 维护约束：键必须覆盖所有会改变投影结果的输入，但不能包含 edit revision。

use super::base::clip_layer_order;
use crate::state::TimelineState;

/// 视图缓存：只有布局键或编辑 revision 变化才重新 compose。
#[derive(Clone, Debug)]
pub struct ParamBlockViewCache {
    pub layout_key: String,
    pub edit_revision: u64,
    pub view: super::ParamBlockView,
}

impl ParamBlockViewCache {
    pub fn matches(&self, layout_key: &str, edit_revision: u64) -> bool {
        self.layout_key == layout_key && self.edit_revision == edit_revision
    }
}

pub fn layout_key(timeline: &TimelineState, root_track_id: &str, frame_period_ms: f64) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"param_blocks_layout_v1");
    hasher.update(&((frame_period_ms.max(0.1) * 1000.0).round() as u32).to_le_bytes());
    hasher.update(root_track_id.as_bytes());
    let mut clips: Vec<&crate::state::Clip> = timeline
        .clips
        .iter()
        .filter(|clip| {
            timeline.resolve_root_track_id(&clip.track_id).as_deref() == Some(root_track_id)
        })
        .collect();
    clips.sort_by(|a, b| clip_layer_order(a, b));
    for clip in clips {
        hasher.update(clip.id.as_bytes());
        hasher.update(&clip.start_sec.to_le_bytes());
        hasher.update(&clip.length_sec.to_le_bytes());
        hasher.update(&clip.duration_sec.unwrap_or(0.0).to_le_bytes());
        hasher.update(&(clip.clip_playback_rate as f64).to_le_bytes());
        hasher.update(&[u8::from(clip.reversed), u8::from(clip.loop_enabled)]);
        hasher.update(&clip.loop_period_sec.unwrap_or(0.0).to_le_bytes());
        let take = if clip.takes.is_empty() {
            crate::state::ClipTake::from_clip(clip)
        } else {
            clip.active_take().clone()
        };
        hasher.update(take.id.as_bytes());
        hasher.update(&take.source_start_sec.to_le_bytes());
        hasher.update(&take.source_end_sec.to_le_bytes());
        hasher.update(&(take.playback_rate as f64).to_le_bytes());
        hasher.update(&[u8::from(take.reversed), u8::from(take.loop_enabled)]);
        hasher.update(&take.duration_sec.unwrap_or(0.0).to_le_bytes());
        hasher.update(&take.duration_frames.unwrap_or(0).to_le_bytes());
        hasher.update(&take.source_sample_rate.unwrap_or(0).to_le_bytes());
        if let Some(analysis) = &take.syllable_analysis {
            hasher.update(analysis.algorithm.as_bytes());
            hasher.update(&analysis.algorithm_version.to_le_bytes());
            hasher.update(analysis.source_digest.as_bytes());
            let window = analysis.analyzed_window_sec;
            hasher.update(&window[0].to_le_bytes());
            hasher.update(&window[1].to_le_bytes());
            for span in &analysis.source_spans_sec {
                hasher.update(&span.start_sec.to_le_bytes());
                hasher.update(&span.end_sec.to_le_bytes());
                hasher.update(&span.midi_note.unwrap_or(f32::NAN).to_le_bytes());
            }
        }
        if let Some(baseline) = &take.game_block_baseline {
            hasher.update(&baseline.revision.to_le_bytes());
            let window = baseline.analyzed_window_sec;
            hasher.update(&window[0].to_le_bytes());
            hasher.update(&window[1].to_le_bytes());
            for span in &baseline.source_spans_sec {
                hasher.update(&span.start_sec.to_le_bytes());
                hasher.update(&span.end_sec.to_le_bytes());
                hasher.update(&span.midi_note.unwrap_or(f32::NAN).to_le_bytes());
            }
        }
        if let Some(notes) = &take.midi_note_data {
            for note in notes {
                hasher.update(&note.start_sec.to_le_bytes());
                hasher.update(&note.end_sec.to_le_bytes());
                hasher.update(&note.note.to_le_bytes());
            }
        }
    }
    format!("layout:{}", hasher.finalize().to_hex())
}
