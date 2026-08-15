//! Native cross-process timeline clipboard commands.
//!
//! The data is a msgpack-encoded `ProjectFragment` transported through
//! `crate::system_clipboard`.  Copying happens in the backend so no WebView
//! clipboard permission is required, and pasting works across two running
//! HiFiShifter processes.

use crate::project_fragment::{
    build_clip_fragment, build_track_fragment, merge_project_fragment, FragmentMergeOptions,
    FragmentTrackPlacement, ProjectFragment, ProjectFragmentKind,
};
use crate::state::AppState;
use crate::system_clipboard;
use serde_json::json;

fn current_project_name(state: &AppState) -> String {
    state
        .project
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .name
        .clone()
}

fn write_fragment(fragment: &ProjectFragment) -> Result<(), String> {
    system_clipboard::write_bytes(&fragment.encode()?)
}

fn read_fragment() -> Result<ProjectFragment, String> {
    let bytes =
        system_clipboard::read_bytes()?.ok_or_else(|| "timeline_clipboard_empty".to_string())?;
    ProjectFragment::decode(&bytes)
}

fn paste_fragment(state: &AppState, fragment: ProjectFragment) -> serde_json::Value {
    let result = {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        state.checkpoint_timeline(&tl);
        let playhead_sec = tl.playhead_sec.max(0.0);
        let track_placement = if fragment.kind == ProjectFragmentKind::Clips {
            FragmentTrackPlacement::PlaceOnSelected
        } else {
            FragmentTrackPlacement::AppendAtEnd
        };
        let merge = match merge_project_fragment(
            &mut tl,
            &fragment,
            FragmentMergeOptions {
                anchor_sec: Some(playhead_sec),
                track_placement,
            },
        ) {
            Ok(merge) => merge,
            Err(error) => return json!({ "ok": false, "error": error }),
        };

        for clip_id in &merge.created_clip_ids {
            if let Some(clip) = tl.clips.iter_mut().find(|clip| clip.id == *clip_id) {
                crate::state::TimelineState::populate_clip_file_metadata(clip);
            }
        }
        state.audio_engine.update_timeline(tl.clone());

        let mut payload = tl.to_payload();
        payload.created_clip_ids = Some(merge.created_clip_ids.clone());
        payload.created_track_ids = Some(merge.created_track_ids.clone());
        payload.project = Some(state.project_meta_payload());

        let midi_root_tracks: Vec<String> = merge
            .created_clip_ids
            .iter()
            .filter_map(|clip_id| tl.clips.iter().find(|clip| clip.id == *clip_id))
            .filter(|clip| clip.midi_note_data.is_some())
            .filter_map(|clip| tl.resolve_root_track_id(&clip.track_id))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        drop(tl);

        for root_id in &midi_root_tracks {
            crate::pitch_analysis::maybe_schedule_pitch_orig(state, root_id);
        }

        let mut json = serde_json::to_value(&payload).unwrap_or_default();
        json["sourceProject"] = json!(fragment.source_project_name);
        json["importedTrackCount"] = json!(merge.imported_track_count);
        json["importedClipCount"] = json!(merge.imported_clip_count);
        json
    };

    result
}

pub(super) fn copy_timeline_clips(state: &AppState, clip_ids: Vec<String>) -> serde_json::Value {
    let timeline = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    let fragment = match build_clip_fragment(&timeline, &clip_ids, current_project_name(state)) {
        Ok(fragment) => fragment,
        Err(error) => return json!({ "ok": false, "error": error }),
    };
    drop(timeline);

    match write_fragment(&fragment) {
        Ok(()) => json!({
            "ok": true,
            "kind": "clips",
            "clipCount": fragment.timeline.clips.len(),
            "trackCount": fragment.timeline.tracks.len(),
        }),
        Err(error) => json!({ "ok": false, "error": error }),
    }
}

pub(super) fn copy_timeline_tracks(state: &AppState, track_ids: Vec<String>) -> serde_json::Value {
    let timeline = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    let fragment = match build_track_fragment(&timeline, &track_ids, current_project_name(state)) {
        Ok(fragment) => fragment,
        Err(error) => return json!({ "ok": false, "error": error }),
    };
    drop(timeline);

    match write_fragment(&fragment) {
        Ok(()) => json!({
            "ok": true,
            "kind": "tracks",
            "clipCount": fragment.timeline.clips.len(),
            "trackCount": fragment.timeline.tracks.len(),
        }),
        Err(error) => json!({ "ok": false, "error": error }),
    }
}

pub(super) fn paste_timeline_clipboard(state: &AppState) -> serde_json::Value {
    match read_fragment() {
        Ok(fragment) => paste_fragment(state, fragment),
        Err(error) => json!({ "ok": false, "error": error }),
    }
}

pub(super) fn has_timeline_clipboard() -> serde_json::Value {
    match read_fragment() {
        Ok(fragment) => json!({
            "ok": true,
            "available": true,
            "kind": fragment.kind,
            "clipCount": fragment.timeline.clips.len(),
            "trackCount": fragment.timeline.tracks.len(),
            "sourceProject": fragment.source_project_name,
        }),
        Err(_) => json!({ "ok": true, "available": false }),
    }
}
