//! Project-fragment serialization and merge helpers.
//!
//! A `ProjectFragment` is a self-contained piece of a HiFiShifter project:
//! tracks, clips, root-track parameter curves and (for clip selections) the
//! automation slices that belong to each copied clip.  The same type is used
//! for native cross-process timeline copy/paste, whole-track copy/paste, and
//! merging another project into the current one.

use crate::state::{new_id, Clip, LinkedParamCurvesPayload, TimelineState, Track};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectFragmentKind {
    Clips,
    Tracks,
    Project,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectFragment {
    pub version: u32,
    pub kind: ProjectFragmentKind,
    pub source_project_name: String,
    pub timeline: TimelineState,
    /// Per-source-clip automation slices for clip selections.  For whole-track
    /// and whole-project fragments the complete `timeline.params_by_root_track`
    /// is used instead.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub linked_params_by_clip: BTreeMap<String, LinkedParamCurvesPayload>,
}

impl ProjectFragment {
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        rmp_serde::to_vec_named(self).map_err(|e| format!("clipboard_serialize_failed: {}", e))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let fragment: Self =
            rmp_serde::from_slice(bytes).map_err(|e| format!("clipboard_parse_failed: {}", e))?;
        if !matches!(fragment.version, 1 | 2) {
            return Err(format!(
                "clipboard_parse_failed: unsupported version {}",
                fragment.version
            ));
        }
        Ok(fragment)
    }
}

fn clip_end_sec(clip: &Clip) -> f64 {
    (clip.start_sec + clip.length_sec).max(0.0)
}

fn max_clip_end_sec(timeline: &TimelineState) -> f64 {
    timeline
        .clips
        .iter()
        .map(clip_end_sec)
        .fold(0.0_f64, f64::max)
}

fn minimal_timeline_shell(source: &TimelineState) -> TimelineState {
    let mut timeline = TimelineState::default();
    timeline.bpm = source.bpm;
    timeline.playhead_sec = source.playhead_sec;
    timeline.project_sec = source.project_sec;
    timeline.project_scale_notes = source.project_scale_notes.clone();
    timeline.selected_track_id = None;
    timeline.selected_clip_id = None;
    timeline.next_track_order = 1;
    timeline
}

fn expand_track_ids_with_ancestors(
    timeline: &TimelineState,
    track_ids: &[String],
) -> BTreeSet<String> {
    let mut selected: BTreeSet<String> = BTreeSet::new();
    for track_id in track_ids {
        let mut cursor = Some(track_id.clone());
        while let Some(id) = cursor {
            if !selected.insert(id.clone()) {
                break;
            }
            cursor = timeline
                .tracks
                .iter()
                .find(|track| track.id == id)
                .and_then(|track| track.parent_id.clone());
        }
    }
    selected
}

fn expand_track_ids_with_ancestors_and_descendants(
    timeline: &TimelineState,
    track_ids: &[String],
) -> BTreeSet<String> {
    let mut selected = expand_track_ids_with_ancestors(timeline, track_ids);

    let mut queue: Vec<String> = track_ids.to_vec();
    while let Some(id) = queue.pop() {
        for child in timeline
            .tracks
            .iter()
            .filter(|track| track.parent_id.as_deref() == Some(id.as_str()))
            .map(|track| track.id.clone())
        {
            if selected.insert(child.clone()) {
                queue.push(child);
            }
        }
    }

    selected
}

fn ordered_selected_tracks(timeline: &TimelineState, selected: &BTreeSet<String>) -> Vec<Track> {
    let mut tracks: Vec<Track> = timeline
        .tracks
        .iter()
        .filter(|track| selected.contains(&track.id))
        .cloned()
        .collect();
    tracks.sort_by_key(|track| track.order);
    tracks
}

fn selected_clips(timeline: &TimelineState, clip_ids: &[String]) -> Vec<Clip> {
    let unique: Vec<String> = {
        let mut seen = BTreeSet::new();
        clip_ids
            .iter()
            .filter(|id| seen.insert((*id).clone()))
            .cloned()
            .collect()
    };
    timeline
        .clips
        .iter()
        .filter(|clip| unique.contains(&clip.id))
        .cloned()
        .collect()
}

pub fn build_clip_fragment(
    timeline: &TimelineState,
    clip_ids: &[String],
    source_project_name: String,
) -> Result<ProjectFragment, String> {
    let clips = selected_clips(timeline, clip_ids);
    if clips.is_empty() {
        return Err("no_clips_selected".to_string());
    }

    // Smart hierarchy preservation: when the selection covers every clip in
    // each affected root track, the user is transferring whole track groups
    // (e.g. Ctrl+A followed by Ctrl+C). In that case produce a full Tracks
    // fragment so the root track, child tracks, full parameter curves and
    // pitch_orig are all pasted together.
    let selected_ids: BTreeSet<String> = clips.iter().map(|clip| clip.id.clone()).collect();
    let mut affected_roots: BTreeSet<String> = BTreeSet::new();
    for clip in &clips {
        if let Some(root) = timeline.resolve_root_track_id(&clip.track_id) {
            affected_roots.insert(root);
        }
    }
    let full_subtree_selection = !affected_roots.is_empty()
        && affected_roots.iter().all(|root| {
            timeline
                .clips
                .iter()
                .filter(|clip| {
                    timeline.resolve_root_track_id(&clip.track_id).as_deref() == Some(root.as_str())
                })
                .all(|clip| selected_ids.contains(&clip.id))
        });
    if full_subtree_selection {
        let root_ids: Vec<String> = affected_roots.into_iter().collect();
        return build_track_fragment(timeline, &root_ids, source_project_name);
    }

    let owning_track_ids: BTreeSet<String> =
        clips.iter().map(|clip| clip.track_id.clone()).collect();
    let owning_track_ids_vec: Vec<String> = owning_track_ids.iter().cloned().collect();
    let selected_track_ids = expand_track_ids_with_ancestors(timeline, &owning_track_ids_vec);

    let mut fragment_timeline = minimal_timeline_shell(timeline);
    fragment_timeline.tracks = ordered_selected_tracks(timeline, &selected_track_ids);
    fragment_timeline.clips = clips
        .into_iter()
        .map(|mut clip| {
            clip.clear_waveform_preview_caches();
            clip
        })
        .collect();
    fragment_timeline.project_sec = max_clip_end_sec(&fragment_timeline).max(4.0).ceil();
    fragment_timeline.next_track_order = fragment_timeline.tracks.len() as i32 + 1;

    let mut source = timeline.clone();
    let mut linked = BTreeMap::new();
    for clip in &fragment_timeline.clips {
        let Some(root_track_id) = source.resolve_root_track_id(&clip.track_id) else {
            continue;
        };
        if let Some(params) = source.extract_linked_params_from_root_range(
            &root_track_id,
            clip.start_sec,
            clip.length_sec,
        ) {
            linked.insert(clip.id.clone(), params);
        }
    }

    Ok(ProjectFragment {
        version: 2,
        kind: ProjectFragmentKind::Clips,
        source_project_name,
        timeline: fragment_timeline,
        linked_params_by_clip: linked,
    })
}

pub fn build_track_fragment(
    timeline: &TimelineState,
    track_ids: &[String],
    source_project_name: String,
) -> Result<ProjectFragment, String> {
    if track_ids.is_empty() {
        return Err("no_tracks_selected".to_string());
    }

    let selected_track_ids = expand_track_ids_with_ancestors_and_descendants(timeline, track_ids);
    let tracks = ordered_selected_tracks(timeline, &selected_track_ids);
    if tracks.is_empty() {
        return Err("track_not_found".to_string());
    }

    let mut fragment_timeline = minimal_timeline_shell(timeline);
    fragment_timeline.tracks = tracks;
    fragment_timeline.clips = timeline
        .clips
        .iter()
        .filter(|clip| selected_track_ids.contains(&clip.track_id))
        .cloned()
        .map(|mut clip| {
            clip.clear_waveform_preview_caches();
            clip
        })
        .collect();
    fragment_timeline.project_sec = max_clip_end_sec(&fragment_timeline).max(4.0).ceil();
    fragment_timeline.next_track_order = fragment_timeline.tracks.len() as i32 + 1;

    for (root_id, params) in &timeline.params_by_root_track {
        if selected_track_ids.contains(root_id) {
            fragment_timeline
                .params_by_root_track
                .insert(root_id.clone(), params.clone());
        }
    }

    Ok(ProjectFragment {
        version: 2,
        kind: ProjectFragmentKind::Tracks,
        source_project_name,
        timeline: fragment_timeline,
        linked_params_by_clip: BTreeMap::new(),
    })
}

pub fn build_project_fragment(
    timeline: TimelineState,
    source_project_name: String,
) -> ProjectFragment {
    let mut fragment_timeline = timeline;
    for clip in &mut fragment_timeline.clips {
        clip.clear_waveform_preview_caches();
    }
    fragment_timeline.project_sec = max_clip_end_sec(&fragment_timeline).max(4.0).ceil();
    ProjectFragment {
        version: 2,
        kind: ProjectFragmentKind::Project,
        source_project_name,
        timeline: fragment_timeline,
        linked_params_by_clip: BTreeMap::new(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentTrackPlacement {
    /// Append every imported root track at the end of the current track list.
    /// Used by track/project paste and by "Paste as New Tracks".
    AppendAtEnd,
    /// Put every copied clip onto the currently selected track. Ancestor-only
    /// tracks in the fragment are not recreated.
    SelectedTrackOnly,
    /// Preserve the relative order of the source clip tracks starting at the
    /// currently selected track. Missing target tracks are created as roots.
    #[allow(dead_code)]
    SelectedTracksRelative,
}

#[derive(Debug, Clone, Copy)]
pub struct FragmentMergeOptions {
    /// When set, the earliest imported clip is shifted to this absolute
    /// timeline position. Relative clip spacing is preserved.
    pub anchor_sec: Option<f64>,
    pub track_placement: FragmentTrackPlacement,
}

#[derive(Debug, Clone, Default)]
pub struct FragmentMergeResult {
    pub created_track_ids: Vec<String>,
    pub created_clip_ids: Vec<String>,
    pub imported_track_count: usize,
    pub imported_clip_count: usize,
}

fn push_cloned_track(
    timeline: &mut TimelineState,
    source: &Track,
    parent_id: Option<String>,
) -> String {
    let id = new_id("track");
    let order = timeline.next_track_order;
    timeline.next_track_order += 1;

    let mut track = source.clone();
    track.id = id.clone();
    track.parent_id = parent_id;
    track.order = order;
    timeline.tracks.push(track);
    id
}

fn track_id_at_index(timeline: &TimelineState, index: usize) -> Option<String> {
    timeline.tracks.get(index).map(|track| track.id.clone())
}

fn remap_group_id(group_map: &mut HashMap<String, String>, old_group_id: &str) -> String {
    group_map
        .entry(old_group_id.to_string())
        .or_insert_with(|| new_id("group"))
        .clone()
}

/// Merge `fragment` into `timeline`. All imported IDs are remapped so the
/// operation is safe to repeat and never collides with existing objects.
pub fn merge_project_fragment(
    timeline: &mut TimelineState,
    fragment: &ProjectFragment,
    options: FragmentMergeOptions,
) -> Result<FragmentMergeResult, String> {
    let source_timeline = &fragment.timeline;
    if source_timeline.tracks.is_empty() && source_timeline.clips.is_empty() {
        return Ok(FragmentMergeResult::default());
    }

    let min_start = source_timeline
        .clips
        .iter()
        .map(|clip| clip.start_sec)
        .fold(f64::INFINITY, f64::min);
    let time_offset_sec = if min_start.is_finite() {
        options
            .anchor_sec
            .map(|anchor| anchor - min_start)
            .unwrap_or(0.0)
    } else {
        0.0
    };

    let mut track_id_map: HashMap<String, String> = HashMap::new();
    let source_tracks = source_timeline.tracks.clone();

    let mut source_roots: Vec<&Track> = source_tracks
        .iter()
        .filter(|track| track.parent_id.is_none())
        .collect();
    source_roots.sort_by_key(|track| track.order);

    match options.track_placement {
        FragmentTrackPlacement::AppendAtEnd => {
            for root in &source_roots {
                let new_id = push_cloned_track(timeline, root, None);
                track_id_map.insert(root.id.clone(), new_id);
            }
            for source in source_tracks
                .iter()
                .filter(|track| track.parent_id.is_some())
            {
                let Some(mapped_parent) = source
                    .parent_id
                    .as_ref()
                    .and_then(|parent| track_id_map.get(parent).cloned())
                else {
                    continue;
                };
                let new_id = push_cloned_track(timeline, source, Some(mapped_parent));
                track_id_map.insert(source.id.clone(), new_id);
            }
        }
        FragmentTrackPlacement::SelectedTrackOnly => {
            // Map every source track that actually contains clips onto the
            // currently selected track. Ancestor-only tracks are ignored, so
            // pasting a child clip never creates an extra child track.
            let clip_track_ids: BTreeSet<String> = source_timeline
                .clips
                .iter()
                .map(|clip| clip.track_id.clone())
                .collect();
            let target_track_id = timeline
                .selected_track_id
                .clone()
                .or_else(|| timeline.tracks.first().map(|track| track.id.clone()))
                .unwrap_or_else(|| {
                    push_cloned_track(timeline, &TimelineState::default().tracks[0], None)
                });
            let target_root_id = timeline
                .resolve_root_track_id(&target_track_id)
                .unwrap_or_else(|| target_track_id.clone());
            for track_id in clip_track_ids {
                track_id_map.insert(track_id, target_track_id.clone());
            }
            // Forced track-fragment flattening: full root params are written
            // into the target track's root group even though hierarchy is not
            // recreated.
            for source_root_id in source_timeline.params_by_root_track.keys() {
                track_id_map
                    .entry(source_root_id.clone())
                    .or_insert_with(|| target_root_id.clone());
            }
        }
        FragmentTrackPlacement::SelectedTracksRelative => {
            let selected_index = timeline
                .selected_track_id
                .as_ref()
                .and_then(|selected| {
                    timeline
                        .tracks
                        .iter()
                        .position(|track| track.id == *selected)
                })
                .unwrap_or(timeline.tracks.len());

            let mut clip_tracks: Vec<&Track> = source_tracks
                .iter()
                .filter(|track| {
                    source_timeline
                        .clips
                        .iter()
                        .any(|clip| clip.track_id == track.id)
                })
                .collect();
            clip_tracks.sort_by_key(|track| track.order);

            for (offset, source_track) in clip_tracks.iter().enumerate() {
                let target_index = selected_index.saturating_add(offset);
                let mapped_id = match track_id_at_index(timeline, target_index) {
                    Some(existing) => existing,
                    None => push_cloned_track(timeline, source_track, None),
                };
                track_id_map.insert(source_track.id.clone(), mapped_id);
            }
        }
    }

    let mut clip_id_map: HashMap<String, String> = HashMap::new();
    let mut group_id_map: HashMap<String, String> = HashMap::new();
    let mut created_clip_ids = Vec::new();

    for source_clip in &source_timeline.clips {
        let Some(mapped_track_id) = track_id_map.get(&source_clip.track_id).cloned() else {
            continue;
        };
        let mut clip = source_clip.clone();
        clip.reconcile_legacy_fade_fields();
        clip.id = new_id("clip");
        clip.track_id = mapped_track_id;
        clip.group_id = source_clip
            .group_id
            .as_ref()
            .map(|group| remap_group_id(&mut group_id_map, group));
        clip.start_sec = (clip.start_sec + time_offset_sec).max(0.0);
        // 先物化 Take → 投影，再统一清波形缓存；顺序颠倒会让 normalize_takes
        // 把 Take 里残留的旧预览写回投影，清空失效并随片段携带大数据。
        clip.normalize_takes();
        clip.clear_waveform_preview_caches();
        clip.remap_take_ids();

        let created_id = clip.id.clone();
        clip_id_map.insert(source_clip.id.clone(), created_id.clone());
        timeline.ensure_project_end_sec(clip_end_sec(&clip));
        timeline.clips.push(clip);
        created_clip_ids.push(created_id);
    }

    for (source_root_id, params) in &source_timeline.params_by_root_track {
        if let Some(mapped_root_id) = track_id_map.get(source_root_id) {
            timeline
                .params_by_root_track
                .insert(mapped_root_id.clone(), params.clone());
            // Full-track fragments carry pitch_orig/pitch_edit. Ensure the
            // target root is in compose mode so those curves actually render.
            if params.pitch_edit_user_modified {
                if let Some(track) = timeline
                    .tracks
                    .iter_mut()
                    .find(|track| track.id == *mapped_root_id)
                {
                    track.compose_enabled = true;
                }
            }
        }
    }

    for (source_clip_id, linked_params) in &fragment.linked_params_by_clip {
        let Some(mapped_clip_id) = clip_id_map.get(source_clip_id) else {
            continue;
        };
        let Some(mapped_clip) = timeline
            .clips
            .iter()
            .find(|clip| clip.id == *mapped_clip_id)
        else {
            continue;
        };
        let Some(root_track_id) = timeline.resolve_root_track_id(&mapped_clip.track_id) else {
            continue;
        };
        timeline.apply_linked_params_to_root_range(
            &root_track_id,
            mapped_clip.start_sec,
            linked_params,
        );
    }

    for disabled_group in &source_timeline.disabled_group_ids {
        if let Some(mapped) = group_id_map.get(disabled_group).cloned() {
            timeline.disabled_group_ids.insert(mapped);
        }
    }

    let created_track_ids: Vec<String> = source_tracks
        .iter()
        .filter_map(|track| track_id_map.get(&track.id).cloned())
        .collect();

    if let Some(first_created_clip_id) = created_clip_ids.first() {
        timeline.selected_clip_id = Some(first_created_clip_id.clone());
        if let Some(clip) = timeline
            .clips
            .iter()
            .find(|clip| clip.id == *first_created_clip_id)
        {
            timeline.selected_track_id = Some(clip.track_id.clone());
            timeline.playhead_sec = clip.start_sec;
        }
    } else if let Some(first_mapped_track_id) = source_tracks
        .first()
        .and_then(|track| track_id_map.get(&track.id).cloned())
    {
        timeline.selected_track_id = Some(first_mapped_track_id);
    }

    let imported_track_count = track_id_map.values().collect::<BTreeSet<_>>().len();
    let imported_clip_count = created_clip_ids.len();
    Ok(FragmentMergeResult {
        created_track_ids,
        created_clip_ids,
        imported_track_count,
        imported_clip_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(name: &str, track_id: &str, start_sec: f64, length_sec: f64) -> Clip {
        Clip {
            id: format!("clip_{name}"),
            takes: vec![],
            active_take_id: None,
            clip_playback_rate: 1.0,
            group_id: None,
            track_id: track_id.to_string(),
            name: name.to_string(),
            start_sec,
            length_sec,
            color: "blue".to_string(),
            source_path: Some(format!("C:/audio/{name}.wav")),
            source_path_relative: None,
            duration_sec: Some(length_sec),
            duration_frames: None,
            source_sample_rate: Some(48_000),
            source_file_mtime: None,
            source_file_size: None,
            source_file_fingerprint: None,
            waveform_preview: None,
            pitch_range: None,
            gain: 1.0,
            muted: false,
            source_start_sec: 0.0,
            source_end_sec: length_sec,
            playback_rate: 1.0,
            reversed: false,
            loop_enabled: false,
            snap_offset_sec: 0.0,
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,
            fade_in_curve: "sine".to_string(),
            fade_out_curve: "sine".to_string(),
            fade_in_shape: 0.0,
            fade_out_shape: 0.0,
            fade_in_dir: 0.0,
            fade_out_dir: 0.0,
            auto_fade_in_sec: 0.0,
            auto_fade_out_sec: 0.0,
            extra_curves: None,
            extra_params: None,
            formant_morph: None,
            midi_note_data: None,
            midi_fill_gaps: false,
        }
    }

    fn timeline_with_child() -> TimelineState {
        let mut tl = TimelineState::default();
        let root = tl.tracks[0].id.clone();
        tl.ensure_params_for_root(&root);
        let child = tl.add_track(Some("Child".to_string()), Some(root.clone()), None);
        let sibling = tl.add_track(Some("Sibling".to_string()), Some(root.clone()), None);
        tl.clips.push(clip("a", &root, 1.0, 2.0));
        tl.clips.push(clip("b", &child, 3.0, 1.5));
        tl.clips.push(clip("c", &sibling, 5.0, 1.0));
        tl
    }

    #[test]
    fn clip_fragment_keeps_ancestors_but_not_untouched_siblings() {
        let tl = timeline_with_child();
        let child_id = tl
            .tracks
            .iter()
            .find(|t| t.name == "Child")
            .unwrap()
            .id
            .clone();
        let clip_id = tl
            .clips
            .iter()
            .find(|c| c.track_id == child_id)
            .unwrap()
            .id
            .clone();
        let fragment = build_clip_fragment(&tl, &[clip_id], "src".into()).unwrap();
        assert_eq!(fragment.timeline.clips.len(), 1);
        assert_eq!(fragment.timeline.tracks.len(), 2);
        assert!(fragment.timeline.tracks.iter().any(|t| t.name == "Main"));
        assert!(fragment.timeline.tracks.iter().any(|t| t.name == "Child"));
        assert!(!fragment.timeline.tracks.iter().any(|t| t.name == "Sibling"));
        assert_eq!(fragment.linked_params_by_clip.len(), 1);
    }

    #[test]
    fn merge_remaps_ids_and_shifts_clips_to_anchor() {
        let source = timeline_with_child();
        let clip_id = source.clips[1].id.clone();
        let fragment = build_clip_fragment(&source, &[clip_id], "src".into()).unwrap();
        let source_clip_id = fragment.timeline.clips[0].id.clone();

        let mut target = TimelineState::default();
        target.playhead_sec = 10.0;
        let merge = merge_project_fragment(
            &mut target,
            &fragment,
            FragmentMergeOptions {
                anchor_sec: Some(10.0),
                track_placement: FragmentTrackPlacement::SelectedTrackOnly,
            },
        )
        .unwrap();

        assert_eq!(merge.imported_clip_count, 1);
        let pasted = target
            .clips
            .iter()
            .find(|c| c.id == merge.created_clip_ids[0])
            .unwrap();
        assert_ne!(pasted.id, source_clip_id);
        assert!((pasted.start_sec - 10.0).abs() < 1e-9);
        assert_eq!(target.selected_clip_id.as_deref(), Some(pasted.id.as_str()));
    }

    #[test]
    fn partial_child_clip_pastes_onto_selected_track_without_extra_tracks() {
        let source = timeline_with_child();
        let child_id = source
            .tracks
            .iter()
            .find(|track| track.name == "Child")
            .unwrap()
            .id
            .clone();
        let clip_id = source
            .clips
            .iter()
            .find(|clip| clip.track_id == child_id)
            .unwrap()
            .id
            .clone();
        let fragment = build_clip_fragment(&source, &[clip_id], "src".into()).unwrap();
        assert_eq!(fragment.kind, ProjectFragmentKind::Clips);

        let mut target = TimelineState::default();
        let target_root = target.tracks[0].id.clone();
        let target_child =
            target.add_track(Some("Target Child".to_string()), Some(target_root), None);
        target.selected_track_id = Some(target_child.clone());
        let track_count_before = target.tracks.len();

        let merge = merge_project_fragment(
            &mut target,
            &fragment,
            FragmentMergeOptions {
                anchor_sec: Some(0.0),
                track_placement: FragmentTrackPlacement::SelectedTrackOnly,
            },
        )
        .unwrap();

        assert_eq!(target.tracks.len(), track_count_before);
        assert_eq!(merge.imported_clip_count, 1);
        let pasted = target
            .clips
            .iter()
            .find(|clip| clip.id == merge.created_clip_ids[0])
            .unwrap();
        assert_eq!(pasted.track_id, target_child);
    }

    #[test]
    fn pasting_pitch_edit_enables_compose_on_target_root() {
        let mut source = timeline_with_child();
        let root_id = source
            .tracks
            .iter()
            .find(|track| track.parent_id.is_none())
            .unwrap()
            .id
            .clone();
        source.ensure_params_for_root(&root_id);
        if let Some(entry) = source.params_by_root_track.get_mut(&root_id) {
            entry.pitch_edit.resize(1600, 61.0);
            entry.pitch_edit_user_modified = true;
        }
        let child_id = source
            .tracks
            .iter()
            .find(|track| track.name == "Child")
            .unwrap()
            .id
            .clone();
        let clip_id = source
            .clips
            .iter()
            .find(|clip| clip.track_id == child_id)
            .unwrap()
            .id
            .clone();
        let fragment = build_clip_fragment(&source, &[clip_id.clone()], "src".into()).unwrap();
        assert!(!fragment.linked_params_by_clip[&clip_id]
            .pitch_edit
            .is_empty());

        let mut target = TimelineState::default();
        let target_root = target.tracks[0].id.clone();
        target.selected_track_id = Some(target_root.clone());
        assert!(!target.tracks[0].compose_enabled);
        merge_project_fragment(
            &mut target,
            &fragment,
            FragmentMergeOptions {
                anchor_sec: Some(0.0),
                track_placement: FragmentTrackPlacement::SelectedTrackOnly,
            },
        )
        .unwrap();
        assert!(
            target
                .tracks
                .iter()
                .find(|track| track.id == target_root)
                .unwrap()
                .compose_enabled
        );
    }

    #[test]
    fn full_subtree_clip_selection_becomes_a_tracks_fragment() {
        let source = timeline_with_child();
        let all_clip_ids: Vec<String> = source.clips.iter().map(|clip| clip.id.clone()).collect();
        let fragment = build_clip_fragment(&source, &all_clip_ids, "src".into()).unwrap();
        assert_eq!(fragment.kind, ProjectFragmentKind::Tracks);
        assert_eq!(fragment.timeline.tracks.len(), 3);
        assert_eq!(fragment.timeline.clips.len(), 3);
        assert!(fragment.timeline.params_by_root_track.keys().any(|root| {
            source
                .tracks
                .iter()
                .any(|track| track.parent_id.is_none() && track.id == *root)
        }));
    }

    #[test]
    fn multi_track_clip_paste_maps_clip_tracks_relative_to_selection() {
        let source = timeline_with_child();
        let child_id = source
            .tracks
            .iter()
            .find(|track| track.name == "Child")
            .unwrap()
            .id
            .clone();
        let sibling_id = source
            .tracks
            .iter()
            .find(|track| track.name == "Sibling")
            .unwrap()
            .id
            .clone();
        let ids = vec![
            source
                .clips
                .iter()
                .find(|clip| clip.track_id == child_id)
                .unwrap()
                .id
                .clone(),
            source
                .clips
                .iter()
                .find(|clip| clip.track_id == sibling_id)
                .unwrap()
                .id
                .clone(),
        ];
        let fragment = build_clip_fragment(&source, &ids, "src".into()).unwrap();
        assert_eq!(fragment.kind, ProjectFragmentKind::Clips);

        let mut target = TimelineState::default();
        target.selected_track_id = Some(target.tracks[0].id.clone());
        let merge = merge_project_fragment(
            &mut target,
            &fragment,
            FragmentMergeOptions {
                anchor_sec: Some(0.0),
                track_placement: FragmentTrackPlacement::SelectedTracksRelative,
            },
        )
        .unwrap();

        assert_eq!(target.tracks.len(), 2);
        assert_eq!(merge.imported_clip_count, 2);
        assert!(target.tracks.iter().all(|track| track.parent_id.is_none()));
    }

    #[test]
    fn fragment_messagepack_roundtrip() {
        let tl = TimelineState::default();
        let fragment = build_track_fragment(&tl, &[tl.tracks[0].id.clone()], "src".into()).unwrap();
        let bytes = fragment.encode().unwrap();
        let decoded = ProjectFragment::decode(&bytes).unwrap();
        assert_eq!(decoded.kind, ProjectFragmentKind::Tracks);
        assert_eq!(decoded.timeline.tracks.len(), 1);
    }
}
