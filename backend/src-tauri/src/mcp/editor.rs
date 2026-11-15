use super::{
    draft::{self, Draft, Failure},
    McpControl,
};
use crate::state::AppState;
use hifishifter_kernel::{
    editor::params,
    state::{HistoryOp, TimelineState},
};
use hifishifter_mcp::*;
use serde_json::{json, Value};
use std::{io::Write, path::Path, sync::atomic::Ordering};
use tauri::{Emitter, Manager};

fn settings(state: &AppState) -> Value {
    let runtime = hifishifter_kernel::time_stretch::current_runtime_stretch_settings();
    json!({"scope":"project/app","effective_stretch_algorithm":runtime.effective_algorithm(),"effective_hifigan_mel_stretch":runtime.effective_hifigan_mel_stretch(),"project":state.project_meta_payload()})
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
fn current(state: &AppState, instance: &str) -> String {
    let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    stamp(state, &tl, instance)
}
fn conflict() -> Failure {
    Failure::new(
        "REVISION_CONFLICT",
        "The project changed. Read current context and retry with a new idempotency key.",
    )
}
fn encode<T: serde::Serialize>(value: T) -> Result<Value, Failure> {
    serde_json::to_value(value).map_err(|e| Failure::new("SERIALIZATION_FAILED", e.to_string()))
}

pub(super) fn view_snapshot(app: &tauri::AppHandle) -> Value {
    let state = app.state::<AppState>();
    let control = app.state::<McpControl>();
    let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    let revision = stamp(&state, &tl, &control.instance_id);
    let mut timeline = tl.to_payload();
    timeline.project = Some(state.project_meta_payload());
    json!({"ok":true,"revision":revision,"timeline":timeline})
}
fn notify(app: &tauri::AppHandle) {
    let state = app.state::<AppState>();
    let control = app.state::<McpControl>();
    let revision = current(&state, &control.instance_id);
    let _ = app.emit("mcp_project_changed", json!({"revision":revision}));
    crate::commands::playback::request_background_render(app);
}

pub(super) fn execute(app: &tauri::AppHandle, request: EditorRequest) -> Value {
    match run(app, request) {
        Ok(value) => value,
        Err(error) => error.value(),
    }
}
fn run(app: &tauri::AppHandle, request: EditorRequest) -> Result<Value, Failure> {
    let state = app.state::<AppState>();
    let control = app.state::<McpControl>();
    let instance = &control.instance_id;
    match request {
        EditorRequest::Context => {
            let revision = current(&state, instance);
            let seen = control
                .view_revision
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            Ok(
                json!({"ok":true,"instance_id":instance,"revision":revision,"gui_revision":seen,"view_sync_pending":seen.as_ref()!=Some(&revision),"settings":settings(&state),"frame_period_ms":5.0,"algorithm_status":crate::commands::get_onnx_status(),"next_actions":["hfs_get_edit_capabilities","hfs_query","hfs_edit"]}),
            )
        }
        EditorRequest::Capabilities(target) => {
            let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            let tracks: Vec<_> = tl
                .tracks
                .iter()
                .filter(|t| target.track_id.as_ref().is_none_or(|id| *id == t.id))
                .collect();
            if target.track_id.is_some() && tracks.is_empty() {
                return Err(Failure::new("TRACK_NOT_FOUND", "unknown target track"));
            }
            let mut entries = Vec::new();
            for track in tracks {
                let root = tl
                    .resolve_root_track_id(&track.id)
                    .ok_or_else(|| Failure::new("INVALID_HIERARCHY", "invalid root"))?;
                let owner = tl
                    .tracks
                    .iter()
                    .find(|t| t.id == root)
                    .ok_or_else(|| Failure::new("TRACK_NOT_FOUND", "missing root"))?;
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
                entries.push(json!({"track":track,"root_track_id":root,"effective_algorithm":owner.pitch_analysis_algo.id(),"parameters":descriptors}));
            }
            Ok(
                json!({"ok":true,"revision":stamp(&state,&tl,instance),"tracks":entries,"algorithms":["nsf_hifigan_onnx","world_dll","vslib","none"],
                    "track_properties":{"volume":{"unit":"linear_gain","min":hifishifter_kernel::gain::GAIN_MIN,"max":hifishifter_kernel::gain::GAIN_MAX},"algorithm":{"scope":"root_track_effective"},"compose_enabled":{"scope":"root_track_effective"}},
                    "algorithm_status":{"hifigan":crate::commands::get_onnx_status(),"vslib":crate::commands::get_vslib_status()},
                    "time_unit":"project_seconds","curve_frame_period_ms":tl.frame_period_ms(),"settings":settings(&state)}),
            )
        }
        EditorRequest::Query(q) => {
            let snapshot = {
                state
                    .timeline
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
            };
            let revision = stamp(&state, &snapshot, instance);
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
                    let id = q
                        .track_id
                        .as_ref()
                        .ok_or_else(|| Failure::new("INVALID_ARGUMENT", "track_id required"))?;
                    let param = q
                        .param
                        .as_ref()
                        .ok_or_else(|| Failure::new("INVALID_ARGUMENT", "param required"))?;
                    draft::parameter(&snapshot, id, param, false)?;
                    let count = q.frame_count.unwrap_or(200);
                    if count == 0 || count > 20000 {
                        return Err(Failure::new(
                            "INVALID_ARGUMENT",
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
                    let id = q
                        .track_id
                        .ok_or_else(|| Failure::new("INVALID_ARGUMENT", "track_id required"))?;
                    let param = q
                        .param
                        .ok_or_else(|| Failure::new("INVALID_ARGUMENT", "param required"))?;
                    draft::parameter(&snapshot, &id, &param, false)?;
                    encode(params::get_static_param(&Draft::new(snapshot), id, param))?
                }
                QueryKind::ParamBlocks => {
                    let id = q
                        .track_id
                        .ok_or_else(|| Failure::new("INVALID_ARGUMENT", "track_id required"))?;
                    let root = snapshot
                        .resolve_root_track_id(&id)
                        .ok_or_else(|| Failure::new("TRACK_NOT_FOUND", id))?;
                    let layout = snapshot
                        .params_by_root_track
                        .get(&root)
                        .and_then(|p| p.param_line_blocks.clone())
                        .unwrap_or_default();
                    json!({"root_track_id":root,"layout":layout})
                }
                QueryKind::Settings => settings(&state),
                QueryKind::Playback => encode(crate::commands::get_playback_state(app.state()))?,
                QueryKind::History => state.history_state_json(),
                QueryKind::Render => {
                    let manager =
                        hifishifter_kernel::clip_rendering_state::global_clip_rendering_state()
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                    let clips:Vec<Value>=snapshot.clips.iter().filter(|c|q.clip_id.as_ref().is_none_or(|id|*id==c.id)).skip(offset).take(limit).map(|clip|{
                        let required=hifishifter_kernel::pitch_editing::does_clip_need_processor_render(&snapshot,clip,clip.start_sec);
                        let info=manager.get_state(&clip.id);
                        json!({"clip_id":clip.id,"processing_required":required,
                            "state":if required{info.map(|i|json!(i.state)).unwrap_or(json!("NotObserved"))}else{json!("NotRequired")},
                            "progress":info.map(|i|i.progress),"error":info.and_then(|i|i.error.as_ref())})
                    }).collect();
                    json!({"clips":clips,"algorithm_status":crate::commands::get_onnx_status(),"auto_background_render":state.ui_settings_snapshot().auto_background_render})
                }
            };
            Ok(json!({"ok":true,"revision":revision,"offset":offset,"limit":limit,"data":data}))
        }
        EditorRequest::Edit(request) => {
            if request.idempotency_key.is_empty() || request.idempotency_key.len() > 128 {
                return Err(Failure::new(
                    "INVALID_ARGUMENT",
                    "idempotency_key must contain 1..128 characters",
                ));
            }
            let hash = blake3::hash(
                serde_json::to_vec(&request)
                    .map_err(|e| Failure::new("INVALID_ARGUMENT", e.to_string()))?
                    .as_slice(),
            )
            .to_hex()
            .to_string();
            // Serializes MCP retries; human edits are protected independently by the timeline CAS.
            let mut ledger = control.ledger.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((_, previous, value)) = ledger
                .iter()
                .find(|(key, _, _)| *key == request.idempotency_key)
            {
                if *previous != hash {
                    return Err(Failure::new(
                        "IDEMPOTENCY_CONFLICT",
                        "key was already used with a different request",
                    ));
                }
                let mut result = value.clone();
                result["replayed"] = json!(true);
                return Ok(result);
            }
            let snapshot = {
                let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
                if stamp(&state, &tl, instance) != request.expected_revision {
                    return Err(conflict());
                }
                tl.clone()
            };
            let old_tracks: std::collections::HashSet<String> =
                snapshot.tracks.iter().map(|t| t.id.clone()).collect();
            let old_clips: std::collections::HashSet<String> =
                snapshot.clips.iter().map(|c| c.id.clone()).collect();
            let (mut candidate, created) = draft::apply(
                snapshot,
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
                if stamp(&state, &tl, instance) != request.expected_revision {
                    return Err(conflict());
                }
                committed = hifishifter_kernel::state::timeline_fingerprint(&candidate)
                    != hifishifter_kernel::state::timeline_fingerprint(&tl);
                if committed {
                    if state.suppress_checkpoints.load(Ordering::Acquire) {
                        return Err(Failure::new("EDITOR_BUSY","A GUI undo group is active; finish the current gesture before retrying"));
                    }
                    // Transport/selection may have changed while the draft was being decoded.
                    candidate.playhead_sec = tl.playhead_sec;
                    candidate.selected_track_id = tl
                        .selected_track_id
                        .clone()
                        .filter(|id| candidate.tracks.iter().any(|t| t.id == *id));
                    candidate.selected_clip_id = tl
                        .selected_clip_id
                        .clone()
                        .filter(|id| candidate.clips.iter().any(|c| c.id == *id));
                    state.checkpoint_timeline(&tl, HistoryOp::Batch);
                    *tl = candidate;
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
                                rendering.set_state(&clip.id,hifishifter_kernel::clip_rendering_state::ClipRenderingState::Pending,0.0,None);
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
            if committed {
                hifishifter_kernel::hnsep_onnx::clear_separation_cache();
                for root in roots {
                    crate::pitch_analysis::maybe_schedule_pitch_orig(&state.timeline, &root);
                }
                notify(app);
            }
            let result = json!({"ok":true,"committed":committed,"revision":current(&state,instance),"created_ids":created,"created_track_ids":created_tracks,"created_clip_ids":created_clips,"view_sync_pending":committed,"render_state":"check_app_status","replayed":false});
            ledger.push_back((request.idempotency_key, hash, result.clone()));
            while ledger.len() > 128 {
                ledger.pop_front();
            }
            Ok(result)
        }
        EditorRequest::History(request) => {
            // Keep the timeline lock across revision validation and the history transition.
            // The existing history entry point locks internally; a fingerprint recheck is added there.
            let (target, intent) = {
                let history = state
                    .timeline_history
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                match request.action {
                    HistoryAction::Undo => (
                        history
                            .position
                            .checked_sub(1)
                            .ok_or_else(|| Failure::new("HISTORY_EMPTY", "nothing to undo"))?,
                        hifishifter_kernel::state::HistoryJumpIntent::Undo,
                    ),
                    HistoryAction::Redo => {
                        let target = history.position + 1;
                        if target >= history.records.len() {
                            return Err(Failure::new("HISTORY_EMPTY", "nothing to redo"));
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
            notify(app);
            Ok(
                json!({"ok":payload.ok,"revision":current(&state,instance),"history":state.history_state_json()}),
            )
        }
        EditorRequest::Transport(request) => {
            if let Some(position) = request.position_sec {
                draft::finite(position, 0.0, 86400.0, "position_sec")?;
            }
            let result = match request.action {
                TransportAction::Play => {
                    crate::commands::play_original(app.state(), request.position_sec.unwrap_or(0.0))
                }
                TransportAction::Stop => crate::commands::stop_audio(app.state()),
                TransportAction::Seek => {
                    crate::commands::set_transport(app.state(), request.position_sec, None)
                }
            };
            Ok(result)
        }
        EditorRequest::Save(request) => {
            if current(&state, instance) != request.expected_revision {
                return Err(conflict());
            }
            let path = Path::new(&request.path);
            if !path.is_absolute() || path.extension().and_then(|s| s.to_str()) != Some("hshp") {
                return Err(Failure::new(
                    "INVALID_ARGUMENT",
                    "an absolute .hshp path is required",
                ));
            }
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Project");
            let mut snapshot =
                crate::commands::project::build_project_file_snapshot(&state, path, name);
            // This endpoint exports a snapshot. A GUI save remains available for archive/history collection.
            snapshot.save_undo_history = false;
            if current(&state, instance) != request.expected_revision {
                return Err(conflict());
            }
            let bytes =
                hifishifter_kernel::project::serialize_project_file_for_path(&snapshot, path)
                    .map_err(|e| Failure::new("SAVE_FAILED", e))?;
            let mut file = tempfile::NamedTempFile::new_in(
                path.parent()
                    .ok_or_else(|| Failure::new("SAVE_FAILED", "missing parent directory"))?,
            )
            .map_err(|e| Failure::new("SAVE_FAILED", e.to_string()))?;
            file.write_all(&bytes)
                .and_then(|_| file.as_file().sync_all())
                .map_err(|e| Failure::new("SAVE_FAILED", e.to_string()))?;
            file.persist_noclobber(path)
                .map_err(|e| Failure::new("SAVE_FAILED", e.to_string()))?;
            Ok(
                json!({"ok":true,"path":request.path,"saved_revision":request.expected_revision,"current_revision":current(&state,instance),"mode":"snapshot_copy","includes_undo_history":false}),
            )
        }
        EditorRequest::Settings(request) => {
            if let SettingsChange::SetProjectStretch {
                expected_revision,
                algorithm_override,
                hifigan_mel_stretch_override,
            } = request.change
            {
                let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
                if stamp(&state, &tl, instance) != expected_revision {
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
                notify(app);
            }
            Ok(
                json!({"ok":true,"revision":current(&state,instance),"data":settings(&state),"undoable":false}),
            )
        }
        EditorRequest::View(request) => {
            {
                let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(id) = &request.selected_track_id {
                    if !tl.tracks.iter().any(|t| t.id == *id) {
                        return Err(Failure::new("TRACK_NOT_FOUND", id));
                    }
                }
                if let Some(id) = &request.selected_clip_id {
                    if !tl.clips.iter().any(|c| c.id == *id) {
                        return Err(Failure::new("CLIP_NOT_FOUND", id));
                    }
                }
                if let Some(param) = &request.active_param {
                    let id = request
                        .selected_track_id
                        .as_ref()
                        .or(tl.selected_track_id.as_ref())
                        .ok_or_else(|| Failure::new("TRACK_NOT_FOUND", "select a track first"))?;
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
                            "INVALID_ARGUMENT",
                            "reference tracks must be root tracks",
                        ));
                    }
                }
            }
            Ok(super::request_view(app, request))
        }
    }
}

pub(crate) fn revision_for_history(state: &AppState, tl: &TimelineState, instance: &str) -> String {
    stamp(state, tl, instance)
}
