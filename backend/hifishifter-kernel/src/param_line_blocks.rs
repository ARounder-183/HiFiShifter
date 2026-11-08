//! One persisted interval type for analysis and hand editing. Frame ranges are
//! half-open and belong to a root parameter line, never to an audio Clip.
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
pub mod geometry;

pub fn timeline_equal_except_blocks(
    left: &crate::state::TimelineState,
    right: &crate::state::TimelineState,
) -> bool {
    let strip = |mut snapshot: crate::state::TimelineState| {
        for params in snapshot.params_by_root_track.values_mut() {
            params.param_line_blocks = None;
        }
        snapshot
            .params_by_root_track
            .retain(|_, params| !params.is_empty_project_data());
        serde_json::to_value(snapshot).ok()
    };
    strip(left.clone()) == strip(right.clone())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ParamLineBlock {
    pub id: String,
    pub start_frame: u32,
    pub end_frame_exclusive: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transition_frames: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ParamLineCurvePatch {
    pub param: String,
    pub start_frame: u32,
    pub values: Vec<f32>,
}

pub fn validate_curve_patches(patches: &[ParamLineCurvePatch]) -> Result<(), String> {
    let mut total = 0usize;
    for patch in patches {
        total = total.saturating_add(patch.values.len());
        if patch.param.is_empty()
            || patch.param.len() > 128
            || total > 1_000_000
            || patch.start_frame as usize + patch.values.len() > 1_000_000
            || patch.values.iter().any(|value| !value.is_finite())
        {
            return Err("invalid parameter block curve patch".into());
        }
    }
    Ok(())
}

pub fn apply_curve_patches(
    timeline: &crate::state::TimelineState,
    root: &str,
    patches: &[ParamLineCurvePatch],
) -> Result<crate::state::TimelineState, String> {
    validate_curve_patches(patches)?;
    struct Candidate(std::sync::Mutex<crate::state::TimelineState>);
    impl crate::editor::ParamHost for Candidate {
        fn timeline(&self) -> &std::sync::Mutex<crate::state::TimelineState> {
            &self.0
        }
        fn checkpoint_timeline(&self, _: &crate::state::TimelineState, _: crate::state::HistoryOp) {
        }
        fn mark_dirty(&self) {}
        fn publish_timeline(&self, _: crate::state::TimelineState) {}
    }
    let candidate = Candidate(std::sync::Mutex::new(timeline.clone()));
    for patch in patches {
        let result = crate::editor::params::set_param_frames_with_present(
            &candidate,
            root.to_owned(),
            patch.param.clone(),
            patch.start_frame,
            patch.values.clone(),
            None,
            Some(false),
        );
        if result["ok"] != true {
            return Err(result["error"]
                .as_str()
                .unwrap_or("block curve edit failed")
                .into());
        }
    }
    Ok(candidate.0.into_inner().unwrap())
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GameBlockBaseline {
    pub revision: u64,
    pub segments: Vec<ParamLineBlock>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ParamLineBlockSet {
    pub schema_version: u32,
    pub revision: u64,
    pub frame_period_ms: f64,
    pub blocks: Vec<ParamLineBlock>,
    pub game_baseline: GameBlockBaseline,
}

impl Default for ParamLineBlockSet {
    fn default() -> Self {
        Self {
            schema_version: 1,
            revision: 0,
            frame_period_ms: 5.0,
            blocks: Vec::new(),
            game_baseline: GameBlockBaseline::default(),
        }
    }
}

fn validate_intervals(blocks: &[ParamLineBlock]) -> Result<(), String> {
    if blocks.len() > 20000 {
        return Err("parameter block count exceeded".into());
    }
    let mut ids = BTreeSet::new();
    let mut previous_end = 0;
    for block in blocks {
        if block.id.is_empty()
            || block.id.len() > 512
            || !ids.insert(&block.id)
            || block.end_frame_exclusive <= block.start_frame
            || block.end_frame_exclusive > 1_000_000
            || block.start_frame < previous_end
        {
            return Err("invalid, overlapping or duplicate parameter blocks".into());
        }
        previous_end = block.end_frame_exclusive;
    }
    Ok(())
}

impl ParamLineBlockSet {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1
            || !self.frame_period_ms.is_finite()
            || self.frame_period_ms <= 0.0
        {
            return Err("invalid parameter block schema or frame period".into());
        }
        validate_intervals(&self.blocks)?;
        validate_intervals(&self.game_baseline.segments)
    }

    pub fn replace(
        &self,
        expected_revision: u64,
        blocks: Vec<ParamLineBlock>,
    ) -> Result<Self, String> {
        self.replace_with_baseline(expected_revision, blocks, None)
    }

    pub fn replace_with_baseline(
        &self,
        expected_revision: u64,
        mut blocks: Vec<ParamLineBlock>,
        baseline: Option<Vec<ParamLineBlock>>,
    ) -> Result<Self, String> {
        if self.revision != expected_revision {
            return Err("Conflict: parameter blocks changed".into());
        }
        blocks.sort_by(|a, b| {
            a.start_frame
                .cmp(&b.start_frame)
                .then(a.end_frame_exclusive.cmp(&b.end_frame_exclusive))
                .then(a.id.cmp(&b.id))
        });
        let mut next = self.clone();
        if let Some(mut segments) = baseline {
            segments.sort_by_key(|block| (block.start_frame, block.end_frame_exclusive));
            if segments != next.game_baseline.segments {
                next.game_baseline.revision += 1;
                next.game_baseline.segments = segments;
            }
        }
        next.revision +=
            u64::from(blocks != self.blocks || next.game_baseline != self.game_baseline);
        next.blocks = blocks;
        next.validate()?;
        Ok(next)
    }
}

fn place(mut basis: Vec<ParamLineBlock>, target: ParamLineBlock) -> Vec<ParamLineBlock> {
    let mut result = Vec::new();
    for block in basis.drain(..) {
        if block.id == target.id {
            continue;
        }
        if block.end_frame_exclusive <= target.start_frame
            || block.start_frame >= target.end_frame_exclusive
        {
            result.push(block);
            continue;
        }
        let left = block.start_frame < target.start_frame;
        let right = block.end_frame_exclusive > target.end_frame_exclusive;
        if left {
            let mut kept = block.clone();
            kept.end_frame_exclusive = target.start_frame;
            kept.transition_frames = Some(0);
            result.push(kept);
        }
        if right {
            let mut kept = block;
            if left {
                let identity = format!("{}:right:{}", kept.id, target.id);
                kept.id = format!("block:{}", blake3::hash(identity.as_bytes()).to_hex());
            }
            kept.start_frame = target.end_frame_exclusive;
            result.push(kept);
        }
    }
    result.push(target);
    result.sort_by_key(|block| (block.start_frame, block.end_frame_exclusive));
    result
}

fn without_range(blocks: Vec<ParamLineBlock>, start: u32, end: u32) -> Vec<ParamLineBlock> {
    place(
        blocks,
        ParamLineBlock {
            id: "__analysis_range__".into(),
            start_frame: start,
            end_frame_exclusive: end,
            transition_frames: None,
        },
    )
    .into_iter()
    .filter(|block| block.id != "__analysis_range__")
    .collect()
}

/// The projection uses the parameter/audio window and loop placement helpers.
/// Only this generator needs source/Clip facts; authored blocks remain plain intervals.
pub fn project_analysis(
    clip: &crate::state::Clip,
    digest: &str,
    fp: f64,
    spans: &[crate::pitch_analysis::SyllableSpan],
) -> Vec<ParamLineBlock> {
    use crate::editor::param_edit::EditAnchor;
    let frame = |sec: f64| (sec * 1000.0 / fp).round().clamp(0.0, 1_000_000.0) as u32;
    let first = frame(clip.start_sec);
    let last = frame(clip.start_sec + clip.length_sec);
    if last <= first {
        return Vec::new();
    }
    let (lo, hi) = crate::state::clip_playback_window_sec(clip);
    let anchor = EditAnchor {
        root: String::new(),
        project_start_sec: clip.start_sec,
        project_duration_sec: clip.length_sec,
        source_start_sec: lo,
        source_duration_sec: hi - lo,
        reversed: clip.reversed,
    };
    let mut intervals = Vec::new();
    for span in spans {
        if let Some(placement) =
            crate::state::place_note_occurrence_in_loop(clip, span.start_sec, span.end_sec, fp)
        {
            let cycle = placement.cycle_frames as i64;
            let mut local = placement.first_start_frame as i64 - cycle;
            while local < (last - first) as i64 && intervals.len() < 20000 {
                let start = (first as i64 + local).max(first as i64) as u32;
                let end = (first as i64 + local + placement.len_frames as i64)
                    .min(last as i64)
                    .max(first as i64) as u32;
                if end > start {
                    intervals.push((start, end));
                }
                local += cycle;
            }
        } else if let (Some(a), Some(b)) = (
            anchor.source_to_project(span.start_sec.max(lo)),
            anchor.source_to_project(span.end_sec.min(hi)),
        ) {
            let start = frame(a.min(b)).max(first);
            let end = frame(a.max(b)).min(last);
            if end > start {
                intervals.push((start, end));
            }
        }
    }
    intervals.sort_unstable();
    // Keep the detected boundaries and fill only the uncovered visible gaps.
    let mut complete = Vec::new();
    let mut cursor = first;
    for (start, end) in intervals {
        if start > cursor {
            complete.push((cursor, start));
        }
        if end > cursor {
            complete.push((start.max(cursor), end));
            cursor = end;
        }
    }
    if cursor < last {
        complete.push((cursor, last));
    }
    complete
        .into_iter()
        .map(|(start, end)| {
            // UI namespaces change at every plugin launch, unlike persisted identities.
            let clip_id = clip
                .id
                .strip_prefix("hfs-ui-")
                .and_then(|id| id.splitn(3, '-').nth(2))
                .unwrap_or(&clip.id);
            let identity = format!("{digest}|{clip_id}|{start}|{end}");
            ParamLineBlock {
                id: format!("block:{}", blake3::hash(identity.as_bytes()).to_hex()),
                start_frame: start,
                end_frame_exclusive: end,
                transition_frames: None,
            }
        })
        .collect()
}

impl ParamLineBlockSet {
    pub fn apply_analysis(
        &self,
        clip: &crate::state::Clip,
        digest: &str,
        spans: &[crate::pitch_analysis::SyllableSpan],
    ) -> Result<Self, String> {
        let incoming = project_analysis(clip, digest, self.frame_period_ms, spans);
        if incoming.is_empty()
            || incoming.iter().all(|block| {
                self.game_baseline
                    .segments
                    .iter()
                    .any(|old| old.id == block.id)
            })
        {
            return Ok(self.clone());
        }
        let start = (clip.start_sec * 1000.0 / self.frame_period_ms)
            .round()
            .max(0.0) as u32;
        let end = ((clip.start_sec + clip.length_sec) * 1000.0 / self.frame_period_ms)
            .round()
            .max(0.0) as u32;
        let relevant =
            |block: &&ParamLineBlock| block.start_frame < end && block.end_frame_exclusive > start;
        let original = self
            .game_baseline
            .segments
            .iter()
            .filter(relevant)
            .collect::<Vec<_>>();
        let current = self.blocks.iter().filter(relevant).collect::<Vec<_>>();
        let pristine = original == current;
        let mut baseline = without_range(self.game_baseline.segments.clone(), start, end);
        let mut blocks = if pristine {
            without_range(self.blocks.clone(), start, end)
        } else {
            self.blocks.clone()
        };
        for block in incoming {
            baseline = place(baseline, block.clone());
            if pristine {
                blocks = place(blocks, block);
            } else if original.is_empty() {
                let mut free = vec![block];
                for occupied in &self.blocks {
                    free = place(free, occupied.clone())
                        .into_iter()
                        .filter(|part| part.id != occupied.id)
                        .collect();
                }
                for part in free {
                    blocks = place(blocks, part);
                }
            }
        }
        self.replace_with_baseline(self.revision, blocks, Some(baseline))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn block(id: &str, start: u32, end: u32) -> ParamLineBlock {
        ParamLineBlock {
            id: id.into(),
            start_frame: start,
            end_frame_exclusive: end,
            transition_frames: None,
        }
    }
    #[test]
    fn records_round_trip_without_audio_clip_or_parent_ids() {
        let set = ParamLineBlockSet::default()
            .replace(0, vec![block("a", 0, 10)])
            .unwrap();
        let encoded = serde_json::to_string(&set).unwrap();
        assert!(!encoded.contains("clip"));
        assert_eq!(
            serde_json::from_str::<ParamLineBlockSet>(&encoded).unwrap(),
            set
        );
        assert_eq!(set.replace(1, set.blocks.clone()).unwrap().revision, 1);
        assert!(set.replace(0, Vec::new()).is_err());
        assert!(set.replace(1, Vec::new()).unwrap().blocks.is_empty());
    }
    #[test]
    fn invalid_layouts_never_replace_committed_blocks() {
        let set = ParamLineBlockSet::default();
        assert!(set
            .replace(0, vec![block("a", 0, 20), block("b", 10, 30)])
            .is_err());
        assert!(set
            .replace(0, vec![block("a", 0, 10), block("a", 10, 20)])
            .is_err());
        assert!(set.replace(0, vec![block("a", 2, 2)]).is_err());
        assert!(serde_json::from_str::<ParamLineBlock>(
            r#"{"id":"a","startFrame":0.5,"endFrameExclusive":2}"#
        )
        .is_err());
    }
    #[test]
    fn layout_edits_preserve_the_game_baseline() {
        let mut set = ParamLineBlockSet::default();
        set.game_baseline.segments = vec![block("original", 0, 10)];
        let edited = set.replace(0, vec![block("manual", 3, 9)]).unwrap();
        assert_eq!(edited.game_baseline, set.game_baseline);
    }

    #[test]
    fn analysis_projects_through_clip_time_and_never_overwrites_user_ranges() {
        let mut clip:crate::state::Clip=serde_json::from_value(serde_json::json!({
            "id":"clip","track_id":"root","name":"source","start_sec":2.0,"length_sec":1.0,
            "takes":[{"id":"take","source_path":"source","source_start_sec":0.0,"source_end_sec":1.0}]
        })).unwrap();
        clip.normalize_takes();
        let spans = vec![
            crate::pitch_analysis::SyllableSpan {
                start_sec: 0.1,
                end_sec: 0.5,
            },
            crate::pitch_analysis::SyllableSpan {
                start_sec: 0.5,
                end_sec: 0.9,
            },
        ];
        let set = ParamLineBlockSet::default();
        let generated = set.apply_analysis(&clip, "digest", &spans).unwrap();
        assert_eq!(
            generated
                .blocks
                .iter()
                .map(|block| (block.start_frame, block.end_frame_exclusive))
                .collect::<Vec<_>>(),
            vec![(400, 420), (420, 500), (500, 580), (580, 600)]
        );
        let manual = set.replace(0, vec![block("manual", 430, 550)]).unwrap();
        let with_analysis = manual.apply_analysis(&clip, "digest", &spans).unwrap();
        assert!(with_analysis.blocks.iter().any(|block| block.id == "manual"
            && block.start_frame == 430
            && block.end_frame_exclusive == 550));
        let mut deleted = generated.clone();
        deleted.blocks.clear();
        assert!(
            deleted
                .apply_analysis(&clip, "digest", &spans)
                .unwrap()
                .blocks
                .is_empty(),
            "cached analysis must not resurrect deleted blocks"
        );
    }
}
