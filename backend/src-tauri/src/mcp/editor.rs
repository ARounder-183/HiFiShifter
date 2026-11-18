//! Core App MCP request executor (`EditorRequest` -> structured JSON response).
//!
//! Operates against [`McpRuntimeHost`] so production (`tauri::AppHandle`) and
//! headless tests (`HeadlessMcpHost`) share the exact same CAS, scoped-CAS,
//! idempotency ledger, readiness barrier, acoustic preview, and notebook logic.

use super::{
    draft::{self, Draft, Failure},
    host::{McpRuntimeHost, TauriRuntimeHost},
    notebook, preview, scope, AuditEntry, McpControl,
};
use crate::state::AppState;
use hifishifter_kernel::{
    editor::params,
    state::{HistoryOp, TimelineState},
};
use hifishifter_mcp::*;
use serde_json::{json, Value};
use std::{
    collections::HashSet,
    io::Write,
    path::Path,
    sync::atomic::Ordering,
    time::{Duration, Instant},
};
use tauri::Manager;

fn settings(state: &AppState) -> Value {
    let runtime = hifishifter_kernel::time_stretch::current_runtime_stretch_settings();
    json!({
        "scope": "project/app",
        "effective_stretch_algorithm": runtime.effective_algorithm(),
        "effective_hifigan_mel_stretch": runtime.effective_hifigan_mel_stretch(),
        "project": state.project_meta_payload()
    })
}

fn stamp(state: &AppState, tl: &TimelineState, instance: &str) -> String {
    let project = settings(state);
    let config = blake3::hash(project.to_string().as_bytes());
    format!(
        "{instance}:{}:{}:{:016x}:{}",
        state.timeline_version.load(Ordering::Acquire),
        state.param_curve_revision.load(Ordering::Acquire),
        hifishifter_kernel::state::timeline_fingerprint(tl),
        &config.to_hex()[..16]
    )
}

pub(super) fn current_revision(state: &AppState, instance: &str) -> String {
    let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    stamp(state, &tl, instance)
}

fn conflict() -> Failure {
    Failure::new(
        McpErrorCode::RevisionConflict.as_str(),
        "The project changed. Read current context and retry with a new idempotency key.",
    )
}

fn encode<T: serde::Serialize>(value: T) -> Result<Value, Failure> {
    serde_json::to_value(value)
        .map_err(|e| Failure::new(McpErrorCode::SerializationFailed.as_str(), e.to_string()))
}

pub(super) fn view_snapshot(app: &tauri::AppHandle) -> Value {
    let state = app.state::<AppState>();
    let control = app.state::<McpControl>();
    let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    let revision = stamp(&state, &tl, &control.instance_id);
    let mut timeline = tl.to_payload();
    timeline.project = Some(state.project_meta_payload());
    json!({ "ok": true, "revision": revision, "timeline": timeline })
}

pub(super) fn execute(app: &tauri::AppHandle, request: EditorRequest) -> Value {
    execute_with_host(&TauriRuntimeHost(app), request)
}

pub(super) fn execute_with_host(host: &dyn McpRuntimeHost, request: EditorRequest) -> Value {
    match run(host, request) {
        Ok(value) => value,
        Err(error) => error.value(),
    }
}

fn run(host: &dyn McpRuntimeHost, request: EditorRequest) -> Result<Value, Failure> {
    let state = host.state();
    let control = host.control();
    let instance = &control.instance_id;

    match request {
        EditorRequest::Context => {
            let (revision, scopes) = {
                let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
                (
                    stamp(state, &tl, instance),
                    scope::compute_scope_revisions(state, &tl, instance),
                )
            };
            let seen = control
                .view_revision
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            Ok(json!({
                "ok": true,
                "instance_id": instance,
                "revision": revision,
                "scope_revisions": scopes.to_json(),
                "gui_revision": seen,
                "view_sync_pending": seen.as_ref() != Some(&revision),
                "settings": settings(state),
                "frame_period_ms": 5.0,
                "algorithm_status": crate::commands::get_onnx_status(),
                "discovery_dir": super::discovery_dir().map(|p| p.to_string_lossy().into_owned()),
                "next_actions": [
                    "hfs_get_edit_capabilities",
                    "hfs_query",
                    "hfs_edit",
                    "hfs_wait_ready",
                    "hfs_render_preview",
                    "hfs_notebook"
                ]
            }))
        }
        EditorRequest::Capabilities(target) => {
            let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            let scopes = scope::compute_scope_revisions(state, &tl, instance);
            let tracks: Vec<_> = tl
                .tracks
                .iter()
                .filter(|t| target.track_id.as_ref().is_none_or(|id| *id == t.id))
                .collect();
            if target.track_id.is_some() && tracks.is_empty() {
                return Err(Failure::new(
                    McpErrorCode::TrackNotFound.as_str(),
                    "unknown target track",
                ));
            }
            let mut entries = Vec::new();
            for track in tracks {
                let root = tl.resolve_root_track_id(&track.id).ok_or_else(|| {
                    Failure::new(McpErrorCode::InvalidHierarchy.as_str(), "invalid root")
                })?;
                let owner = tl.tracks.iter().find(|t| t.id == root).ok_or_else(|| {
                    Failure::new(McpErrorCode::TrackNotFound.as_str(), "missing root")
                })?;
                let kind = hifishifter_kernel::state::SynthPipelineKind::from_track_algo(
                    &owner.pitch_analysis_algo,
                );
                let mut names: Vec<String> =
                    hifishifter_kernel::renderer::all_param_descriptors(kind)
                        .into_iter()
                        .map(|p| p.id.into())
                        .collect();
                names.insert(0, "pitch".into());
                if track.parent_id.is_some() {
                    for prefix in [
                        "child_pitch_offset_cents@",
                        "child_pitch_offset_degrees@",
                        "child_formant_offset_cents@",
                    ] {
                        names.push(format!("{prefix}{}", track.id));
                    }
                }
                let descriptors: Vec<Value> = names
                    .iter()
                    .map(|name| draft::parameter(&tl, &track.id, name, false))
                    .collect::<Result<_, _>>()?;
                entries.push(json!({
                    "track": track,
                    "root_track_id": root,
                    "track_scope_revision": scopes.track_revisions.get(&root),
                    "effective_algorithm": owner.pitch_analysis_algo.id(),
                    "parameters": descriptors,
                }));
            }
            Ok(json!({
                "ok": true,
                "revision": stamp(state, &tl, instance),
                "scope_revisions": scopes.to_json(),
                "tracks": entries,
                "algorithms": ["nsf_hifigan_onnx", "world_dll", "vslib", "none"],
                "track_properties": {
                    "volume": {
                        "unit": "linear_gain",
                        "min": hifishifter_kernel::gain::GAIN_MIN,
                        "max": hifishifter_kernel::gain::GAIN_MAX
                    },
                    "algorithm": { "scope": "root_track_effective" },
                    "compose_enabled": { "scope": "root_track_effective" }
                },
                "algorithm_status": {
                    "hifigan": crate::commands::get_onnx_status(),
                    "vslib": crate::commands::get_vslib_status()
                },
                "time_unit": "project_seconds",
                "curve_frame_period_ms": tl.frame_period_ms(),
                "settings": settings(state),
            }))
        }
        EditorRequest::Query(q) => {
            let snapshot = state
                .timeline
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let revision = stamp(state, &snapshot, instance);
            let offset = q.offset.unwrap_or(0);
            let limit = q.limit.unwrap_or(100).clamp(1, 500);
            let data = match q.kind {
                QueryKind::Timeline => {
                    let mut v = snapshot.to_payload_lite();
                    v.clips = v.clips.into_iter().skip(offset).take(limit).collect();
                    encode(v)?
                }
                QueryKind::Tracks => encode(
                    snapshot
                        .tracks
                        .iter()
                        .filter(|t| q.track_id.as_ref().is_none_or(|id| *id == t.id))
                        .skip(offset)
                        .take(limit)
                        .collect::<Vec<_>>(),
                )?,
                QueryKind::Clips => encode(
                    snapshot
                        .to_payload()
                        .clips
                        .into_iter()
                        .filter(|c| {
                            q.track_id.as_ref().is_none_or(|id| *id == c.track_id)
                                && q.clip_id.as_ref().is_none_or(|id| *id == c.id)
                        })
                        .skip(offset)
                        .take(limit)
                        .collect::<Vec<_>>(),
                )?,
                QueryKind::Curve => {
                    let id = q.track_id.as_ref().ok_or_else(|| {
                        Failure::new(McpErrorCode::InvalidArgument.as_str(), "track_id required")
                    })?;
                    let param = q.param.as_ref().ok_or_else(|| {
                        Failure::new(McpErrorCode::InvalidArgument.as_str(), "param required")
                    })?;
                    draft::parameter(&snapshot, id, param, false)?;
                    let count = q.frame_count.unwrap_or(200);
                    if count == 0 || count > 20000 {
                        return Err(Failure::new(
                            McpErrorCode::InvalidArgument.as_str(),
                            "frame_count must be 1..20000",
                        ));
                    }
                    encode(params::get_param_frames(
                        &Draft::new(snapshot),
                        id.clone(),
                        param.clone(),
                        q.start_frame.unwrap_or(0),
                        count,
                        None,
                        Some(false),
                        Some(true),
                    ))?
                }
                QueryKind::Static => {
                    let id = q.track_id.ok_or_else(|| {
                        Failure::new(McpErrorCode::InvalidArgument.as_str(), "track_id required")
                    })?;
                    let param = q.param.ok_or_else(|| {
                        Failure::new(McpErrorCode::InvalidArgument.as_str(), "param required")
                    })?;
                    draft::parameter(&snapshot, &id, &param, false)?;
                    encode(params::get_static_param(&Draft::new(snapshot), id, param))?
                }
                QueryKind::ParamBlocks => {
                    let id = q.track_id.ok_or_else(|| {
                        Failure::new(McpErrorCode::InvalidArgument.as_str(), "track_id required")
                    })?;
                    let root = snapshot
                        .resolve_root_track_id(&id)
                        .ok_or_else(|| Failure::new(McpErrorCode::TrackNotFound.as_str(), id))?;
                    let layout = snapshot
                        .params_by_root_track
                        .get(&root)
                        .map(|p| p.param_block_edits.clone())
                        .unwrap_or_default();
                    let view = hifishifter_kernel::param_blocks::compose_param_block_view(
                        &snapshot,
                        &root,
                        &layout,
                        snapshot.frame_period_ms(),
                    )
                    .map_err(|error| {
                        Failure::new(McpErrorCode::ParamBlockInvalid.as_str(), error)
                    })?;
                    json!({ "root_track_id": root, "layout": layout, "view": view })
                }
                QueryKind::Settings => settings(state),
                QueryKind::Playback => host.playback_state()?,
                QueryKind::History => state.history_state_json(),
                QueryKind::Render => {
                    build_render_status(state, &snapshot, q.clip_id.as_deref(), offset, limit)
                }
                QueryKind::Notebook => {
                    let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
                    json!({
                        "markdown": p.notes_markdown,
                        "byte_len": p.notes_markdown.len(),
                        "agent_sections": notebook::list_agent_sections(&p.notes_markdown),
                    })
                }
                QueryKind::Ledger => {
                    let audit = control.audit.lock().unwrap_or_else(|e| e.into_inner());
                    let items: Vec<_> = audit
                        .iter()
                        .rev()
                        .skip(offset)
                        .take(limit)
                        .cloned()
                        .collect();
                    json!({
                        "total_retained": audit.len(),
                        "entries": items,
                    })
                }
            };
            Ok(json!({
                "ok": true,
                "revision": revision,
                "offset": offset,
                "limit": limit,
                "data": data
            }))
        }
        EditorRequest::Edit(request) => execute_edit(host, request),
        EditorRequest::History(request) => {
            let (target, intent) = {
                let history = state
                    .timeline_history
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                match request.action {
                    HistoryAction::Undo => (
                        history.position.checked_sub(1).ok_or_else(|| {
                            Failure::new(McpErrorCode::HistoryEmpty.as_str(), "nothing to undo")
                        })?,
                        hifishifter_kernel::state::HistoryJumpIntent::Undo,
                    ),
                    HistoryAction::Redo => {
                        let target = history.position + 1;
                        if target >= history.records.len() {
                            return Err(Failure::new(
                                McpErrorCode::HistoryEmpty.as_str(),
                                "nothing to redo",
                            ));
                        }
                        (target, hifishifter_kernel::state::HistoryJumpIntent::Redo)
                    }
                }
            };
            let payload = state.set_history_position_guarded(
                target,
                intent,
                Some((&request.expected_revision, instance)),
            );
            if !payload.ok {
                return Err(conflict());
            }
            let rev = current_revision(state, instance);
            host.notify_project_changed(&rev);
            Ok(json!({
                "ok": payload.ok,
                "revision": rev,
                "history": state.history_state_json()
            }))
        }
        EditorRequest::Transport(request) => {
            if let Some(position) = request.position_sec {
                draft::finite(position, 0.0, 86400.0, "position_sec")?;
            }
            Ok(host.transport(&request))
        }
        EditorRequest::Save(request) => {
            if current_revision(state, instance) != request.expected_revision {
                return Err(conflict());
            }
            let path = Path::new(&request.path);
            if !path.is_absolute() || path.extension().and_then(|s| s.to_str()) != Some("hshp") {
                return Err(Failure::new(
                    McpErrorCode::InvalidArgument.as_str(),
                    "an absolute .hshp path is required",
                ));
            }
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Project");
            let mut snapshot =
                crate::commands::project::build_project_file_snapshot(state, path, name);
            snapshot.save_undo_history = false;
            if current_revision(state, instance) != request.expected_revision {
                return Err(conflict());
            }
            let bytes =
                hifishifter_kernel::project::serialize_project_file_for_path(&snapshot, path)
                    .map_err(|e| Failure::new(McpErrorCode::SaveFailed.as_str(), e))?;
            let mut file = tempfile::NamedTempFile::new_in(path.parent().ok_or_else(|| {
                Failure::new(
                    McpErrorCode::SaveFailed.as_str(),
                    "missing parent directory",
                )
            })?)
            .map_err(|e| Failure::new(McpErrorCode::SaveFailed.as_str(), e.to_string()))?;
            file.write_all(&bytes)
                .and_then(|_| file.as_file().sync_all())
                .map_err(|e| Failure::new(McpErrorCode::SaveFailed.as_str(), e.to_string()))?;
            file.persist_noclobber(path)
                .map_err(|e| Failure::new(McpErrorCode::SaveFailed.as_str(), e.to_string()))?;
            Ok(json!({
                "ok": true,
                "path": request.path,
                "saved_revision": request.expected_revision,
                "current_revision": current_revision(state, instance),
                "mode": "snapshot_copy",
                "includes_undo_history": false
            }))
        }
        EditorRequest::Settings(request) => {
            if let SettingsChange::SetProjectStretch {
                expected_revision,
                algorithm_override,
                hifigan_mel_stretch_override,
            } = request.change
            {
                let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
                if stamp(state, &tl, instance) != expected_revision {
                    return Err(conflict());
                }
                let algorithm = algorithm_override.map(|a| match a {
                    StretchAlgorithm::Signalsmith => {
                        hifishifter_kernel::time_stretch::UserStretchAlgorithm::Signalsmith
                    }
                    StretchAlgorithm::Soundtouch => {
                        hifishifter_kernel::time_stretch::UserStretchAlgorithm::Soundtouch
                    }
                    StretchAlgorithm::Linear => {
                        hifishifter_kernel::time_stretch::UserStretchAlgorithm::Linear
                    }
                });
                {
                    let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
                    p.stretch_algorithm_override = algorithm;
                    p.hifigan_mel_stretch_override = hifigan_mel_stretch_override;
                }
                hifishifter_kernel::time_stretch::update_project_stretch_overrides(
                    algorithm,
                    hifigan_mel_stretch_override,
                );
                for clip in &tl.clips {
                    hifishifter_kernel::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
                }
                state.audio_engine.update_timeline(tl.clone());
                drop(tl);
                state.mark_project_dirty_and_retitle();
                let rev = current_revision(state, instance);
                host.notify_project_changed(&rev);
            }
            Ok(json!({
                "ok": true,
                "revision": current_revision(state, instance),
                "data": settings(state),
                "undoable": false
            }))
        }
        EditorRequest::View(request) => {
            if let Some(s) = request.focus_start_sec {
                draft::finite(s, 0.0, 86400.0, "focus_start_sec")?;
            }
            if let Some(e) = request.focus_end_sec {
                draft::finite(e, 0.0, 86400.0, "focus_end_sec")?;
            }
            if let (Some(s), Some(e)) = (request.focus_start_sec, request.focus_end_sec) {
                if e <= s {
                    return Err(Failure::new(
                        McpErrorCode::InvalidArgument.as_str(),
                        "focus_end_sec must exceed focus_start_sec",
                    ));
                }
            }
            {
                let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(id) = &request.selected_track_id {
                    if !tl.tracks.iter().any(|t| t.id == *id) {
                        return Err(Failure::new(McpErrorCode::TrackNotFound.as_str(), id));
                    }
                }
                if let Some(id) = &request.selected_clip_id {
                    if !tl.clips.iter().any(|c| c.id == *id) {
                        return Err(Failure::new(McpErrorCode::ClipNotFound.as_str(), id));
                    }
                }
                if let Some(param) = &request.active_param {
                    let id = request
                        .selected_track_id
                        .as_ref()
                        .or(tl.selected_track_id.as_ref())
                        .ok_or_else(|| {
                            Failure::new(
                                McpErrorCode::TrackNotFound.as_str(),
                                "select a track first",
                            )
                        })?;
                    draft::parameter(&tl, id, param, false)?;
                }
                if let Some(ids) = &request.reference_root_track_ids {
                    if ids.len() > 1000
                        || ids.iter().any(|id| {
                            !tl.tracks
                                .iter()
                                .any(|t| t.id == *id && t.parent_id.is_none())
                        })
                    {
                        return Err(Failure::new(
                            McpErrorCode::InvalidArgument.as_str(),
                            "reference tracks must be root tracks",
                        ));
                    }
                }
            }
            Ok(host.request_view(request))
        }
        EditorRequest::Wait(request) => execute_wait(host, request),
        EditorRequest::RenderPreview(request) => {
            let rev = current_revision(state, instance);
            preview::execute_render_preview(state, &rev, request)
        }
        EditorRequest::Notebook(request) => execute_notebook(host, request),
    }
}

fn execute_edit(host: &dyn McpRuntimeHost, request: EditRequest) -> Result<Value, Failure> {
    let state = host.state();
    let control = host.control();
    let instance = &control.instance_id;

    if request.idempotency_key.is_empty() || request.idempotency_key.len() > 128 {
        return Err(Failure::new(
            McpErrorCode::InvalidArgument.as_str(),
            "idempotency_key must contain 1..128 characters",
        ));
    }
    if request.expected_revision.is_none() && request.expected_scope.is_none() {
        return Err(Failure::new(
            McpErrorCode::InvalidArgument.as_str(),
            "provide either expected_revision or expected_scope",
        ));
    }
    let hash = blake3::hash(
        serde_json::to_vec(&request)
            .map_err(|e| Failure::new(McpErrorCode::InvalidArgument.as_str(), e.to_string()))?
            .as_slice(),
    )
    .to_hex()
    .to_string();

    // Serializes MCP retries; human edits are protected independently by timeline CAS / scoped CAS.
    let mut ledger = control.ledger.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, previous, value)) = ledger
        .iter()
        .find(|(key, _, _)| *key == request.idempotency_key)
    {
        if *previous != hash {
            return Err(Failure::new(
                McpErrorCode::IdempotencyConflict.as_str(),
                "key was already used with a different request",
            ));
        }
        let mut result = value.clone();
        result["replayed"] = json!(true);
        return Ok(result);
    }

    let snapshot = {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(expected_rev) = &request.expected_revision {
            if stamp(state, &tl, instance) != *expected_rev {
                return Err(conflict());
            }
        }
        if let Some(expected_scope) = &request.expected_scope {
            let live_scopes = scope::compute_scope_revisions(state, &tl, instance);
            scope::check_scope_matches(&live_scopes, expected_scope)?;
        }
        tl.clone()
    };

    if let Some(expected_scope) = &request.expected_scope {
        scope::validate_operations_fit_scope(&snapshot, expected_scope, &request.operations)?;
    }

    let old_tracks: HashSet<String> = snapshot.tracks.iter().map(|t| t.id.clone()).collect();
    let old_clips: HashSet<String> = snapshot.clips.iter().map(|c| c.id.clone()).collect();
    let (candidate, created) = draft::apply(
        snapshot.clone(),
        &request.operations,
        state.ui_settings_snapshot().lock_param_lines,
    )?;
    let created_tracks: Vec<String> = candidate
        .tracks
        .iter()
        .filter(|t| !old_tracks.contains(&t.id))
        .map(|t| t.id.clone())
        .collect();
    let created_clips: Vec<String> = candidate
        .clips
        .iter()
        .filter(|c| !old_clips.contains(&c.id))
        .map(|c| c.id.clone())
        .collect();

    let roots: Vec<String>;
    let committed;
    {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(expected_rev) = &request.expected_revision {
            if stamp(state, &tl, instance) != *expected_rev {
                return Err(conflict());
            }
        }
        if let Some(expected_scope) = &request.expected_scope {
            let live_scopes = scope::compute_scope_revisions(state, &tl, instance);
            scope::check_scope_matches(&live_scopes, expected_scope)?;
        }

        let before_fp = hifishifter_kernel::state::timeline_fingerprint(&tl);
        let mut next_live = tl.clone();
        if let Some(expected_scope) = &request.expected_scope {
            scope::splice_scoped_candidate_into_live(
                &mut next_live,
                &snapshot,
                candidate,
                expected_scope,
            );
        } else {
            let mut full = candidate;
            // Transport/selection may have changed while the draft was being decoded.
            full.playhead_sec = tl.playhead_sec;
            full.selected_track_id = tl
                .selected_track_id
                .clone()
                .filter(|id| full.tracks.iter().any(|t| t.id == *id));
            full.selected_clip_id = tl
                .selected_clip_id
                .clone()
                .filter(|id| full.clips.iter().any(|c| c.id == *id));
            next_live = full;
        }

        committed = hifishifter_kernel::state::timeline_fingerprint(&next_live) != before_fp;
        if committed {
            if state.suppress_checkpoints.load(Ordering::Acquire) {
                return Err(Failure::new(
                    McpErrorCode::EditorBusy.as_str(),
                    "A GUI undo group is active; finish the current gesture before retrying",
                ));
            }
            state.checkpoint_timeline(&tl, HistoryOp::Batch);
            *tl = next_live;
            {
                let mut rendering =
                    hifishifter_kernel::clip_rendering_state::global_clip_rendering_state()
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                for clip in &tl.clips {
                    if hifishifter_kernel::pitch_editing::does_clip_need_processor_render(
                        &tl,
                        clip,
                        clip.start_sec,
                    ) {
                        rendering.set_state(
                            &clip.id,
                            hifishifter_kernel::clip_rendering_state::ClipRenderingState::Pending,
                            0.0,
                            None,
                        );
                    } else {
                        rendering.remove_state(&clip.id);
                    }
                }
            }
            let mut project = state.project.lock().unwrap_or_else(|e| e.into_inner());
            state.sync_project_record_from_tempo_map(&mut tl, &mut project);
        }
        roots = tl
            .tracks
            .iter()
            .filter(|t| t.parent_id.is_none())
            .map(|t| t.id.clone())
            .collect();
        state.audio_engine.update_timeline(tl.clone());
    }

    let (new_rev, new_scopes) = {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        (
            stamp(state, &tl, instance),
            scope::compute_scope_revisions(state, &tl, instance),
        )
    };
    if committed {
        hifishifter_kernel::hnsep_onnx::clear_separation_cache();
        for root in roots {
            host.schedule_pitch_analysis(&root);
        }
        host.notify_project_changed(&new_rev);
        let mut audit = control.audit.lock().unwrap_or_else(|e| e.into_inner());
        let seq = control
            .next_audit_seq
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
        audit.push_back(AuditEntry {
            sequence: seq,
            timestamp_unix_ms: hifishifter_mcp::discovery::now_unix_ms(),
            idempotency_key: request.idempotency_key.clone(),
            label: request.label.clone(),
            operation_count: request.operations.len(),
            revision: new_rev.clone(),
            created_track_ids: created_tracks.clone(),
            created_clip_ids: created_clips.clone(),
        });
        while audit.len() > 128 {
            audit.pop_front();
        }
    }
    let result = json!({
        "ok": true,
        "committed": committed,
        "revision": new_rev,
        "scope_revisions": new_scopes.to_json(),
        "created_ids": created,
        "created_track_ids": created_tracks,
        "created_clip_ids": created_clips,
        "view_sync_pending": committed,
        "render_state": "check_app_status",
        "replayed": false,
    });
    ledger.push_back((request.idempotency_key, hash, result.clone()));
    while ledger.len() > 128 {
        ledger.pop_front();
    }
    Ok(result)
}

fn execute_wait(host: &dyn McpRuntimeHost, request: WaitRequest) -> Result<Value, Failure> {
    if request.targets.is_empty() {
        return Err(Failure::new(
            McpErrorCode::InvalidArgument.as_str(),
            "targets must be non-empty",
        ));
    }
    let state = host.state();
    let control = host.control();
    let instance = &control.instance_id;
    let timeout_ms = request.timeout_ms.unwrap_or(5_000).min(30_000);

    // Validate requested track/clip IDs up front and optionally kick off pitch analysis.
    let target_roots: Vec<String> = {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(tids) = &request.track_ids {
            let mut roots = Vec::new();
            for tid in tids {
                let root = tl.resolve_root_track_id(tid).ok_or_else(|| {
                    Failure::new(McpErrorCode::TrackNotFound.as_str(), tid.clone())
                })?;
                if !roots.contains(&root) {
                    roots.push(root);
                }
            }
            roots
        } else {
            tl.tracks
                .iter()
                .filter(|t| t.parent_id.is_none() && t.compose_enabled)
                .map(|t| t.id.clone())
                .collect()
        }
    };
    if let Some(cids) = &request.clip_ids {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        for cid in cids {
            if !tl.clips.iter().any(|c| c.id == *cid) {
                return Err(Failure::new(
                    McpErrorCode::ClipNotFound.as_str(),
                    cid.clone(),
                ));
            }
        }
    }
    if request.targets.contains(&WaitTarget::PitchAnalysis)
        && request.trigger_analysis.unwrap_or(true)
    {
        for root in &target_roots {
            host.schedule_pitch_analysis(root);
        }
    }

    let started = Instant::now();
    let deadline = started + Duration::from_millis(timeout_ms);
    loop {
        let snapshot = state
            .timeline
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let rev = stamp(state, &snapshot, instance);

        let mut all_ready = true;
        let mut status = json!({});

        if request.targets.contains(&WaitTarget::PitchAnalysis) {
            let inflight = state
                .pitch_inflight
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let draft_tl = Draft::new(snapshot.clone());
            let mut pending_roots = Vec::new();
            for root in &target_roots {
                let busy_inflight = inflight
                    .iter()
                    .any(|k| k == root || k.starts_with(&format!("{root}:")));
                let frames = params::get_param_frames(
                    &draft_tl,
                    root.clone(),
                    "pitch".into(),
                    0,
                    1,
                    None,
                    Some(false),
                    Some(true),
                );
                if busy_inflight || frames.analysis_pending == Some(true) {
                    pending_roots.push(root.clone());
                }
            }
            let ready = pending_roots.is_empty();
            if !ready {
                all_ready = false;
            }
            status["pitch_analysis"] = json!({
                "ready": ready,
                "checked_root_tracks": target_roots,
                "pending_root_tracks": pending_roots,
            });
        }

        if request.targets.contains(&WaitTarget::ClipRender) {
            let auto_bg = state.ui_settings_snapshot().auto_background_render;
            let allowed_tracks: Option<HashSet<String>> = request.track_ids.as_ref().map(|tids| {
                let mut s = HashSet::new();
                for tid in tids {
                    if let Some(r) = snapshot.resolve_root_track_id(tid) {
                        s.extend(scope::subtree_track_ids(&snapshot, &r));
                    }
                }
                s
            });
            let manager = hifishifter_kernel::clip_rendering_state::global_clip_rendering_state()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let mut pending_clips = Vec::new();
            let mut failed_clips = Vec::new();
            let mut ready_clips = 0usize;
            for clip in &snapshot.clips {
                if let Some(cids) = &request.clip_ids {
                    if !cids.contains(&clip.id) {
                        continue;
                    }
                } else if let Some(tset) = &allowed_tracks {
                    if !tset.contains(&clip.track_id) {
                        continue;
                    }
                }
                let needed = hifishifter_kernel::pitch_editing::does_clip_need_processor_render(
                    &snapshot,
                    clip,
                    clip.start_sec,
                );
                if !needed {
                    ready_clips += 1;
                    continue;
                }
                match manager
                    .get_state(&clip.id)
                    .map(|i| (&i.state, i.error.clone()))
                {
                    Some((
                        hifishifter_kernel::clip_rendering_state::ClipRenderingState::Ready,
                        _,
                    )) => {
                        ready_clips += 1;
                    }
                    Some((
                        hifishifter_kernel::clip_rendering_state::ClipRenderingState::Failed,
                        err,
                    )) => {
                        failed_clips.push(json!({ "clip_id": clip.id, "error": err }));
                    }
                    _ => {
                        pending_clips.push(clip.id.clone());
                    }
                }
            }
            // Terminal failure stops waiting immediately so Agents do not hang until timeout.
            let settled = pending_clips.is_empty() || !failed_clips.is_empty() || !auto_bg;
            if !settled {
                all_ready = false;
            }
            status["clip_render"] = json!({
                "ready": pending_clips.is_empty() && failed_clips.is_empty(),
                "ready_clips": ready_clips,
                "pending_clips": pending_clips,
                "failed_clips": failed_clips,
                "auto_background_render": auto_bg,
            });
            if !pending_clips.is_empty() && !auto_bg {
                all_ready = false;
            }
        }

        if request.targets.contains(&WaitTarget::GuiView) {
            let seen = control
                .view_revision
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let gui_ready = seen.as_ref() == Some(&rev);
            if !gui_ready {
                all_ready = false;
            }
            status["gui_view"] = json!({
                "ready": gui_ready,
                "gui_revision": seen,
                "expected_revision": rev,
            });
        }

        let render_aborted = status
            .get("clip_render")
            .and_then(|v| v.get("failed_clips"))
            .and_then(Value::as_array)
            .is_some_and(|arr| !arr.is_empty());

        if all_ready || render_aborted || Instant::now() >= deadline {
            return Ok(json!({
                "ok": true,
                "ready": all_ready && !render_aborted,
                "timed_out": !all_ready && !render_aborted,
                "elapsed_ms": started.elapsed().as_millis() as u64,
                "revision": rev,
                "status": status,
            }));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn execute_notebook(host: &dyn McpRuntimeHost, request: NotebookRequest) -> Result<Value, Failure> {
    let state = host.state();
    let instance = &host.control().instance_id;
    match request.action {
        NotebookAction::Read => {
            let rev = current_revision(state, instance);
            let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
            Ok(json!({
                "ok": true,
                "revision": rev,
                "markdown": p.notes_markdown,
                "byte_len": p.notes_markdown.len(),
                "agent_sections": notebook::list_agent_sections(&p.notes_markdown),
            }))
        }
        NotebookAction::UpsertAgentSection {
            expected_revision,
            section_id,
            title,
            markdown_body,
        } => {
            {
                let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(expected) = &expected_revision {
                    if stamp(state, &tl, instance) != *expected {
                        return Err(conflict());
                    }
                }
                let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
                let updated = notebook::upsert_fenced_section(
                    &p.notes_markdown,
                    &section_id,
                    &title,
                    &markdown_body,
                )?;
                p.notes_markdown = updated;
                p.notes_history_sealed = true;
                p.dirty = true;
            }
            state.mark_project_dirty_and_retitle();
            let rev = current_revision(state, instance);
            host.notify_notebook_changed(&rev);
            let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
            Ok(json!({
                "ok": true,
                "revision": rev,
                "section_id": section_id,
                "byte_len": p.notes_markdown.len(),
                "agent_sections": notebook::list_agent_sections(&p.notes_markdown),
            }))
        }
        NotebookAction::RemoveAgentSection {
            expected_revision,
            section_id,
        } => {
            let removed;
            {
                let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(expected) = &expected_revision {
                    if stamp(state, &tl, instance) != *expected {
                        return Err(conflict());
                    }
                }
                let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
                let (updated, did_remove) =
                    notebook::remove_fenced_section(&p.notes_markdown, &section_id)?;
                removed = did_remove;
                if removed {
                    p.notes_markdown = updated;
                    p.notes_history_sealed = true;
                    p.dirty = true;
                }
            }
            if removed {
                state.mark_project_dirty_and_retitle();
            }
            let rev = current_revision(state, instance);
            if removed {
                host.notify_notebook_changed(&rev);
            }
            Ok(json!({
                "ok": true,
                "removed": removed,
                "revision": rev,
                "section_id": section_id,
            }))
        }
    }
}

fn build_render_status(
    state: &AppState,
    snapshot: &TimelineState,
    clip_filter: Option<&str>,
    offset: usize,
    limit: usize,
) -> Value {
    let manager = hifishifter_kernel::clip_rendering_state::global_clip_rendering_state()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let clips: Vec<Value> = snapshot
        .clips
        .iter()
        .filter(|c| clip_filter.is_none_or(|id| id == c.id))
        .skip(offset)
        .take(limit)
        .map(|clip| {
            let required = hifishifter_kernel::pitch_editing::does_clip_need_processor_render(
                snapshot,
                clip,
                clip.start_sec,
            );
            let info = manager.get_state(&clip.id);
            json!({
                "clip_id": clip.id,
                "processing_required": required,
                "state": if required {
                    info.map(|i| json!(i.state)).unwrap_or(json!("NotObserved"))
                } else {
                    json!("NotRequired")
                },
                "progress": info.map(|i| i.progress),
                "error": info.and_then(|i| i.error.as_ref()),
            })
        })
        .collect();
    json!({
        "clips": clips,
        "algorithm_status": crate::commands::get_onnx_status(),
        "auto_background_render": state.ui_settings_snapshot().auto_background_render,
    })
}

pub(crate) fn revision_for_history(state: &AppState, tl: &TimelineState, instance: &str) -> String {
    stamp(state, tl, instance)
}

#[cfg(test)]
#[path = "editor_tests.rs"]
mod tests;
