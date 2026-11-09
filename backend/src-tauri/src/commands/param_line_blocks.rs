use crate::state::{AppState, HistoryOp};
use hifishifter_kernel::param_line_blocks::{
    validate_curve_patches, ParamLineBlock, ParamLineBlockSet, ParamLineCurvePatch,
};
use tauri::Emitter;

pub(super) fn get(state: &AppState, root: &str) -> Result<serde_json::Value, String> {
    let timeline = state
        .timeline
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if !timeline.tracks.iter().any(|track| track.id == root) {
        return Err("unknown root parameter line".into());
    }
    let stored = timeline
        .params_by_root_track
        .get(root)
        .and_then(|params| params.param_line_blocks.as_ref());
    let layout = stored.cloned().unwrap_or(ParamLineBlockSet {
        frame_period_ms: timeline.frame_period_ms(),
        ..Default::default()
    });
    Ok(
        serde_json::json!({"ok":true,"initialized":stored.is_some(),"layout":layout,
        "curveRevision":state.param_curve_revision.load(std::sync::atomic::Ordering::Acquire)}),
    )
}

pub(super) fn set(
    state: &AppState,
    root: String,
    expected: u64,
    blocks: Vec<ParamLineBlock>,
    baseline: Option<Vec<ParamLineBlock>>,
    checkpoint: bool,
    patches: Vec<ParamLineCurvePatch>,
    expected_curve_revision: Option<u64>,
    expected_geometry_signature: Option<String>,
) -> Result<serde_json::Value, String> {
    validate_curve_patches(&patches)?;
    let (layout, audio_timeline, curve_revision) = {
        let mut timeline = state
            .timeline
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !timeline.tracks.iter().any(|track| track.id == root) {
            return Err("unknown root parameter line".into());
        }
        if !patches.is_empty()
            && expected_geometry_signature
                .as_deref()
                .is_some_and(|signature| {
                    signature
                        != hifishifter_kernel::pitch_analysis::build_root_pitch_layout_key(
                            &timeline, &root,
                        )
                })
        {
            return Err("Conflict: parameter geometry changed".into());
        }
        if !patches.is_empty()
            && expected_curve_revision
                != Some(
                    state
                        .param_curve_revision
                        .load(std::sync::atomic::Ordering::Acquire),
                )
        {
            return Err("Conflict: parameter curve changed".into());
        }
        let current = timeline
            .params_by_root_track
            .get(&root)
            .and_then(|params| params.param_line_blocks.clone())
            .unwrap_or(ParamLineBlockSet {
                frame_period_ms: timeline.frame_period_ms(),
                ..Default::default()
            });
        let next = current.replace_with_baseline(expected, blocks, baseline)?;
        let candidate = Candidate(std::sync::Mutex::new(timeline.clone()));
        for patch in &patches {
            let result = hifishifter_kernel::editor::params::set_param_frames_with_present(
                &candidate,
                root.clone(),
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
        if checkpoint && (next != current || !patches.is_empty()) {
            state.checkpoint_block_layout(&timeline);
        }
        let mut candidate = candidate.0.into_inner().unwrap();
        candidate.ensure_params_for_root(&root);
        candidate
            .params_by_root_track
            .get_mut(&root)
            .unwrap()
            .param_line_blocks = Some(next.clone());
        *timeline = candidate;
        let audio = if patches.is_empty() {
            None
        } else {
            state.bump_timeline_version();
            Some(timeline.clone())
        };
        (
            next,
            audio,
            state
                .param_curve_revision
                .load(std::sync::atomic::Ordering::Acquire),
        )
    };
    if let Some(timeline) = audio_timeline {
        state.audio_engine.update_timeline(timeline);
    }
    state.mark_project_dirty_and_retitle();
    if let Some(app) = state.app_handle.get() {
        let _ = app.emit(
            "param_line_blocks_updated",
            serde_json::json!({"rootTrackId":root,"layout":layout}),
        );
    }
    Ok(
        serde_json::json!({"ok":true,"initialized":true,"layout":layout,"curveRevision":curve_revision}),
    )
}

struct Candidate(std::sync::Mutex<crate::state::TimelineState>);
impl hifishifter_kernel::editor::ParamHost for Candidate {
    fn timeline(&self) -> &std::sync::Mutex<crate::state::TimelineState> {
        &self.0
    }
    fn checkpoint_timeline(&self, _: &crate::state::TimelineState, _: HistoryOp) {}
    fn mark_dirty(&self) {}
    fn publish_timeline(&self, _: crate::state::TimelineState) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    #[test]
    fn param_line_app_save_undo_and_atomic_conflict() {
        let state = AppState::default();
        let root = {
            let mut tl = state.timeline.lock().unwrap();
            tl.add_track(None, None, None);
            tl.tracks[0].id.clone()
        };
        let initialized = set(
            &state,
            root.clone(),
            0,
            vec![],
            None,
            false,
            vec![],
            None,
            None,
        )
        .unwrap();
        let generation = state.timeline_version.load(Ordering::Acquire);
        let block = ParamLineBlock {
            id: "manual".into(),
            start_frame: 10,
            end_frame_exclusive: 30,
            transition_frames: None,
        };
        let edited = set(
            &state,
            root.clone(),
            initialized["layout"]["revision"].as_u64().unwrap(),
            vec![block.clone()],
            None,
            true,
            vec![],
            None,
            None,
        )
        .unwrap();
        state.undo_timeline();
        let undone = get(&state, &root).unwrap();
        assert_eq!(undone["layout"]["blocks"], serde_json::json!([]));
        assert!(
            undone["layout"]["revision"].as_u64().unwrap()
                > edited["layout"]["revision"].as_u64().unwrap()
        );
        assert_eq!(state.timeline_version.load(Ordering::Acquire), generation);
        state.redo_timeline();
        let redone = get(&state, &root).unwrap();
        assert_eq!(redone["layout"]["blocks"][0]["id"], "manual");
        let history = state.history_depths();
        assert!(set(
            &state,
            root.clone(),
            redone["layout"]["revision"].as_u64().unwrap(),
            vec![],
            None,
            true,
            vec![ParamLineCurvePatch {
                param: "volume".into(),
                start_frame: 10,
                values: vec![0.5]
            }],
            Some(u64::MAX),
            None
        )
        .is_err());
        assert_eq!(get(&state, &root).unwrap()["layout"], redone["layout"]);
        assert_eq!(history, state.history_depths());
    }
}
