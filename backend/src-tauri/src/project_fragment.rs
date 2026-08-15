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
    #[serde(default)]
    pub linked_params_by_clip: BTreeMap<String, LinkedParamCurvesPayload>,
}

impl ProjectFragment {
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        rmp_serde::to_vec_named(self).map_err(|e| format!("clipboard_serialize_failed: {}", e))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let fragment: Self =
            rmp_serde::from_slice(bytes).map_err(|e| format!("clipboard_parse_failed: {}", e))?;
        if fragment.version != 1 {
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

    let owning_track_ids: BTreeSet<String> =
        clips.iter().map(|clip| clip.track_id.clone()).collect();
    let owning_track_ids_vec: Vec<String> = owning_track_ids.iter().cloned().collect();
    let selected_track_ids = expand_track_ids_with_ancestors(timeline, &owning_track_ids_vec);

    let mut fragment_timeline = minimal_timeline_shell(timeline);
    fragment_timeline.tracks = ordered_selected_tracks(timeline, &selected_track_ids);
    fragment_timeline.clips = clips
        .into_iter()
        .map(|mut clip| {
            clip.waveform_preview = None;
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
        version: 1,
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
            clip.waveform_preview = None;
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
        version: 1,
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
        clip.waveform_preview = None;
    }
    fragment_timeline.project_sec = max_clip_end_sec(&fragment_timeline).max(4.0).ceil();
    ProjectFragment {
        version: 1,
        kind: ProjectFragmentKind::Project,
        source_project_name,
        timeline: fragment_timeline,
        linked_params_by_clip: BTreeMap::new(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentTrackPlacement {
    /// Append every imported root track at the end of the current track list.
    AppendAtEnd,
    /// Re-use tracks starting at the currently selected track and create new
    /// child/root tracks only when required. Mirrors clip paste behavior.
    PlaceOnSelected,
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
        FragmentTrackPlacement::PlaceOnSelected => {
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

            for (offset, root) in source_roots.iter().enumerate() {
                let target_index = selected_index.saturating_add(offset);
                let mapped_id = match track_id_at_index(timeline, target_index) {
                    Some(existing) => existing,
                    None => push_cloned_track(timeline, root, None),
                };
                track_id_map.insert(root.id.clone(), mapped_id);
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
    }

    let mut clip_id_map: HashMap<String, String> = HashMap::new();
    let mut group_id_map: HashMap<String, String> = HashMap::new();
    let mut created_clip_ids = Vec::new();

    for source_clip in &source_timeline.clips {
        let Some(mapped_track_id) = track_id_map.get(&source_clip.track_id).cloned() else {
            continue;
        };
        let mut clip = source_clip.clone();
        clip.id = new_id("clip");
        clip.track_id = mapped_track_id;
        clip.group_id = source_clip
            .group_id
            .as_ref()
            .map(|group| remap_group_id(&mut group_id_map, group));
        clip.start_sec = (clip.start_sec + time_offset_sec).max(0.0);
        clip.waveform_preview = None;

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

    if let Some(first_mapped_track_id) = source_tracks
        .first()
        .and_then(|track| track_id_map.get(&track.id).cloned())
    {
        timeline.selected_track_id = Some(first_mapped_track_id);
    }
    if let Some(first_created_clip_id) = created_clip_ids.first() {
        timeline.selected_clip_id = Some(first_created_clip_id.clone());
        if let Some(clip) = timeline
            .clips
            .iter()
            .find(|clip| clip.id == *first_created_clip_id)
        {
            timeline.playhead_sec = clip.start_sec;
        }
    }

    let imported_track_count = source_tracks.len();
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
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,
            fade_in_curve: "sine".to_string(),
            fade_out_curve: "sine".to_string(),
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
                track_placement: FragmentTrackPlacement::PlaceOnSelected,
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
    fn fragment_messagepack_roundtrip() {
        let tl = TimelineState::default();
        let fragment = build_track_fragment(&tl, &[tl.tracks[0].id.clone()], "src".into()).unwrap();
        let bytes = fragment.encode().unwrap();
        let decoded = ProjectFragment::decode(&bytes).unwrap();
        assert_eq!(decoded.kind, ProjectFragmentKind::Tracks);
        assert_eq!(decoded.timeline.tracks.len(), 1);
    }
}
