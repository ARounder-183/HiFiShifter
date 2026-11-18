//! Root-track scoped optimistic concurrency (`EditScope`) for real-time human-agent co-editing.
//!
//! Instead of rejecting an entire Agent batch whenever a human touches an unrelated track,
//! `ScopeRevisions` hashes each root-track subtree (its tracks, clips/takes, and root-owned
//! parameter curves/blocks) independently alongside `tempo_revision`.

use super::draft::Failure;
use crate::state::AppState;
use hifishifter_kernel::state::TimelineState;
use hifishifter_mcp::{EditScope, Operation};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone)]
pub(super) struct ScopeRevisions {
    pub tempo_revision: String,
    pub track_revisions: BTreeMap<String, String>,
}

impl ScopeRevisions {
    pub fn to_json(&self) -> Value {
        json!({
            "tempo_revision": self.tempo_revision,
            "track_revisions": self.track_revisions,
        })
    }
}

pub(super) fn subtree_track_ids(tl: &TimelineState, root_track_id: &str) -> HashSet<String> {
    let mut set = HashSet::new();
    for track in &tl.tracks {
        if tl.resolve_root_track_id(&track.id).as_deref() == Some(root_track_id) {
            set.insert(track.id.clone());
        }
    }
    set
}

pub(super) fn compute_scope_revisions(
    state: &AppState,
    tl: &TimelineState,
    instance: &str,
) -> ScopeRevisions {
    let runtime = hifishifter_kernel::time_stretch::current_runtime_stretch_settings();
    let tempo_payload = json!({
        "instance": instance,
        "bpm": tl.bpm,
        "project_sec": tl.project_sec,
        "project_scale_notes": tl.project_scale_notes,
        "tempo_map": tl.tempo_map,
        "stretch_algo": runtime.effective_algorithm(),
        "mel_stretch": runtime.effective_hifigan_mel_stretch(),
        "base_scale": state.project.lock().unwrap_or_else(|e| e.into_inner()).base_scale,
    });
    let tempo_hash = blake3::hash(tempo_payload.to_string().as_bytes());
    let tempo_revision = format!("tempo:{}", &tempo_hash.to_hex()[..16]);

    let mut track_revisions = BTreeMap::new();
    for root in tl.tracks.iter().filter(|t| t.parent_id.is_none()) {
        let member_ids = subtree_track_ids(tl, &root.id);
        let mut sub_tl = TimelineState::default();
        sub_tl.tracks = tl
            .tracks
            .iter()
            .filter(|t| member_ids.contains(&t.id))
            .cloned()
            .collect();
        sub_tl.clips = tl
            .clips
            .iter()
            .filter(|c| member_ids.contains(&c.track_id))
            .cloned()
            .collect();
        sub_tl.params_by_root_track.clear();
        if let Some(params) = tl.params_by_root_track.get(&root.id) {
            sub_tl
                .params_by_root_track
                .insert(root.id.clone(), params.clone());
        }
        let fp = hifishifter_kernel::state::timeline_fingerprint(&sub_tl);
        let mut hasher = blake3::Hasher::new();
        hasher.update(instance.as_bytes());
        hasher.update(root.id.as_bytes());
        hasher.update(&fp.to_le_bytes());
        let hex = hasher.finalize();
        track_revisions.insert(root.id.clone(), format!("trk:{}", &hex.to_hex()[..16]));
    }
    ScopeRevisions {
        tempo_revision,
        track_revisions,
    }
}

/// Ensure all operations in `operations` stay strictly inside `expected_scope`.
pub(super) fn validate_operations_fit_scope(
    snapshot: &TimelineState,
    scope: &EditScope,
    operations: &[Operation],
) -> Result<(), Failure> {
    let mut created_track_aliases = HashSet::new();
    let mut created_clip_aliases = HashSet::new();
    let allowed_existing_tracks: HashSet<String> = match scope {
        EditScope::IsolatedNewTracks { .. } => HashSet::new(),
        EditScope::RootTrack { root_track_id, .. } => {
            let is_root = snapshot
                .tracks
                .iter()
                .any(|t| t.id == *root_track_id && t.parent_id.is_none());
            if !is_root {
                return Err(Failure::new(
                    "TRACK_NOT_FOUND",
                    format!("root track '{root_track_id}' does not exist"),
                ));
            }
            subtree_track_ids(snapshot, root_track_id)
        }
    };
    let allowed_existing_clips: HashSet<String> = snapshot
        .clips
        .iter()
        .filter(|c| allowed_existing_tracks.contains(&c.track_id))
        .map(|c| c.id.clone())
        .collect();

    let check_track = |id: &str, aliases: &HashSet<String>| -> Result<(), Failure> {
        if aliases.contains(id) || allowed_existing_tracks.contains(id) {
            Ok(())
        } else {
            Err(Failure::new(
                "INVALID_ARGUMENT",
                format!("operation references track '{id}' outside declared expected_scope"),
            ))
        }
    };
    let check_clip = |id: &str, aliases: &HashSet<String>| -> Result<(), Failure> {
        if aliases.contains(id) || allowed_existing_clips.contains(id) {
            Ok(())
        } else {
            Err(Failure::new(
                "INVALID_ARGUMENT",
                format!("operation references clip '{id}' outside declared expected_scope"),
            ))
        }
    };

    for op in operations {
        match op {
            Operation::SetTempo { .. } | Operation::SetProjectLength { .. } => {
                return Err(Failure::new(
                    "INVALID_ARGUMENT",
                    "project-wide tempo/length edits require expected_revision, not expected_scope",
                ));
            }
            Operation::CreateTrack {
                client_id,
                parent_id,
                ..
            } => {
                match (scope, parent_id) {
                    (EditScope::RootTrack { .. }, None) => {
                        return Err(Failure::new(
                            "INVALID_ARGUMENT",
                            "root_track scope cannot create a sibling root track; pass parent_id or use isolated_new_tracks",
                        ));
                    }
                    (_, Some(pid)) => check_track(pid, &created_track_aliases)?,
                    (EditScope::IsolatedNewTracks { .. }, None) => {}
                }
                created_track_aliases.insert(client_id.clone());
            }
            Operation::PatchTrack { track_id, .. } | Operation::DuplicateTrack { track_id } => {
                check_track(track_id, &created_track_aliases)?;
            }
            Operation::RemoveTrack { track_id } => {
                if let EditScope::RootTrack { root_track_id, .. } = scope {
                    if track_id == root_track_id {
                        return Err(Failure::new(
                            "INVALID_ARGUMENT",
                            "removing the scoped root track itself requires expected_revision",
                        ));
                    }
                }
                check_track(track_id, &created_track_aliases)?;
            }
            Operation::MoveTrack {
                track_id,
                parent_id,
                ..
            } => {
                check_track(track_id, &created_track_aliases)?;
                match (scope, parent_id) {
                    (EditScope::RootTrack { .. }, None) => {
                        return Err(Failure::new(
                            "INVALID_ARGUMENT",
                            "cannot promote child track to root inside root_track scope",
                        ));
                    }
                    (_, Some(pid)) => check_track(pid, &created_track_aliases)?,
                    (EditScope::IsolatedNewTracks { .. }, None) => {}
                }
            }
            Operation::ImportClip {
                client_id,
                track_id,
                ..
            }
            | Operation::ImportCandidateTakes {
                client_id,
                track_id,
                ..
            } => {
                check_track(track_id, &created_track_aliases)?;
                created_clip_aliases.insert(client_id.clone());
            }
            Operation::PatchClip { clip_id, .. }
            | Operation::SplitClip { clip_id, .. }
            | Operation::RemoveClip { clip_id }
            | Operation::SetActiveTake { clip_id, .. }
            | Operation::RenameTake { clip_id, .. }
            | Operation::DuplicateTake { clip_id, .. }
            | Operation::RemoveTake { clip_id, .. }
            | Operation::ExplodeTakes { clip_id } => {
                check_clip(clip_id, &created_clip_aliases)?;
            }
            Operation::MoveClip {
                clip_id, track_id, ..
            } => {
                check_clip(clip_id, &created_clip_aliases)?;
                if let Some(tid) = track_id {
                    check_track(tid, &created_track_aliases)?;
                }
            }
            Operation::PackTakes { clip_ids }
            | Operation::DuplicateClips { clip_ids, .. }
            | Operation::GroupClips { clip_ids, .. } => {
                for cid in clip_ids {
                    check_clip(cid, &created_clip_aliases)?;
                }
            }
            Operation::WriteCurve { track_id, .. }
            | Operation::RestoreCurve { track_id, .. }
            | Operation::SetStatic { track_id, .. }
            | Operation::TransformCurve { track_id, .. }
            | Operation::SetParamBlocks { track_id, .. } => {
                check_track(track_id, &created_track_aliases)?;
            }
        }
    }
    Ok(())
}

pub(super) fn check_scope_matches(
    live_scopes: &ScopeRevisions,
    expected: &EditScope,
) -> Result<(), Failure> {
    match expected {
        EditScope::IsolatedNewTracks { tempo_revision } => {
            if live_scopes.tempo_revision != *tempo_revision {
                return Err(Failure::new(
                    "REVISION_CONFLICT",
                    "project tempo or stretch settings changed; refresh context and retry",
                ));
            }
        }
        EditScope::RootTrack {
            root_track_id,
            track_revision,
            tempo_revision,
        } => {
            if live_scopes.tempo_revision != *tempo_revision {
                return Err(Failure::new(
                    "REVISION_CONFLICT",
                    "project tempo or stretch settings changed; refresh context and retry",
                ));
            }
            if live_scopes.track_revisions.get(root_track_id) != Some(track_revision) {
                return Err(Failure::new(
                    "REVISION_CONFLICT",
                    format!("root track '{root_track_id}' changed concurrently; refresh context and retry"),
                ));
            }
        }
    }
    Ok(())
}

/// Splice only the scoped changes from `candidate` onto `live` while preserving
/// concurrent edits made on other root tracks!
pub(super) fn splice_scoped_candidate_into_live(
    live: &mut TimelineState,
    base_snapshot: &TimelineState,
    candidate: TimelineState,
    scope: &EditScope,
) {
    match scope {
        EditScope::IsolatedNewTracks { .. } => {
            let base_track_ids: HashSet<&str> =
                base_snapshot.tracks.iter().map(|t| t.id.as_str()).collect();
            let base_clip_ids: HashSet<&str> =
                base_snapshot.clips.iter().map(|c| c.id.as_str()).collect();
            let mut next_order = live
                .tracks
                .iter()
                .map(|t| t.order)
                .max()
                .unwrap_or(-1)
                .saturating_add(1);
            let mut new_roots = HashSet::new();
            for mut track in candidate.tracks {
                if !base_track_ids.contains(track.id.as_str()) {
                    if track.parent_id.is_none() {
                        new_roots.insert(track.id.clone());
                    }
                    track.order = next_order;
                    next_order = next_order.saturating_add(1);
                    live.tracks.push(track);
                }
            }
            live.next_track_order = live.next_track_order.max(next_order);
            for clip in candidate.clips {
                if !base_clip_ids.contains(clip.id.as_str()) {
                    live.clips.push(clip);
                }
            }
            for root_id in new_roots {
                if let Some(params) = candidate.params_by_root_track.get(&root_id) {
                    live.params_by_root_track.insert(root_id, params.clone());
                }
            }
            if candidate.project_sec > live.project_sec {
                live.project_sec = candidate.project_sec;
            }
        }
        EditScope::RootTrack { root_track_id, .. } => {
            let old_live_members = subtree_track_ids(live, root_track_id);
            let new_cand_members = subtree_track_ids(&candidate, root_track_id);
            let insert_pos = live
                .tracks
                .iter()
                .position(|t| t.id == *root_track_id)
                .unwrap_or(live.tracks.len());
            live.tracks.retain(|t| !old_live_members.contains(&t.id));
            let replacement_tracks: Vec<_> = candidate
                .tracks
                .into_iter()
                .filter(|t| new_cand_members.contains(&t.id))
                .collect();
            let pos = insert_pos.min(live.tracks.len());
            live.tracks.splice(pos..pos, replacement_tracks);
            for (idx, track) in live.tracks.iter_mut().enumerate() {
                track.order = idx as i32;
            }
            live.next_track_order = live.tracks.len() as i32;

            live.clips
                .retain(|c| !old_live_members.contains(&c.track_id));
            for clip in candidate.clips {
                if new_cand_members.contains(&clip.track_id) {
                    live.clips.push(clip);
                }
            }
            match candidate.params_by_root_track.get(root_track_id) {
                Some(params) => {
                    live.params_by_root_track
                        .insert(root_track_id.clone(), params.clone());
                }
                None => {
                    live.params_by_root_track.remove(root_track_id);
                }
            }
            if candidate.project_sec > live.project_sec {
                live.project_sec = candidate.project_sec;
            }
        }
    }
}
