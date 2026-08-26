//! Import another HiFiShifter project into the currently open project.
//!
//! Unlike `open_project` this never replaces the current timeline.  Imported
//! tracks are appended (or aligned to the playhead), all IDs are remapped and
//! source paths are resolved relative to the imported project file first.

use crate::project::{load_project_file, project_name_from_path, resolve_source_paths_on_open};
use crate::project_fragment::{
    build_project_fragment, merge_project_fragment, FragmentMergeOptions, FragmentTrackPlacement,
};
use crate::state::AppState;
use std::fs;
use std::path::PathBuf;
use tauri::{State, Window};

use super::core::get_timeline_state_from_ref;

fn update_window_title(window: &Window, name: &str, dirty: bool) {
    let suffix = if dirty { "*" } else { "" };
    let title = format!("HiFiShifter - {}{}", name, suffix);
    let _ = window.set_title(&title);
}

pub(super) fn import_project_dialog() -> serde_json::Value {
    let picked = rfd::FileDialog::new()
        .add_filter("HiFiShifter Project", &["hshp", "hsp", "json"])
        .pick_file();
    match picked {
        None => serde_json::json!({"ok": true, "canceled": true}),
        Some(path) => {
            serde_json::json!({"ok": true, "canceled": false, "path": path.display().to_string()})
        }
    }
}

#[allow(clippy::too_many_lines)]
pub(super) fn import_project(
    state: State<'_, AppState>,
    window: Window,
    project_path: String,
    place_at_playhead: Option<bool>,
    import_tempo_map: Option<bool>,
) -> serde_json::Value {
    let path = PathBuf::from(&project_path);
    let bytes = fs::read(&path).unwrap_or_default();
    let parsed = load_project_file(&bytes);
    let Ok(mut pf) = parsed else {
        return serde_json::json!({"ok": false, "error": "import_project_parse_failed"});
    };

    let (resolved_timeline, missing_files) = resolve_source_paths_on_open(pf.timeline, &path);
    let mut timeline = resolved_timeline;
    // 旧项目兼容迁移（仅 v3 及更早，同 open_project）：se==0 哨兵展开。
    if pf.version < 4 {
        for clip in &mut timeline.clips {
            if clip.source_end_sec == 0.0 {
                clip.source_end_sec = clip.duration_sec.unwrap_or(clip.length_sec);
            }
        }
    }
    // v4 迁移：旧工程 Clip 不携带 loop_enabled，按"为新的音频块启用循环"设置
    // 补齐 —— 仅限有源媒体的音频 Clip（纯 MIDI 块保持关闭，与导入器约定一致，
    // 见 open_project 的同名迁移注释）。
    if pf.version < 4 {
        let default_loop = crate::config::loop_new_clips_default();
        for clip in &mut timeline.clips {
            if clip.source_path.is_some() {
                clip.loop_enabled = default_loop;
            }
        }
    }
    // 非 Loop 存储窗口规范化（同 open_project，见其注释）。
    for clip in &mut timeline.clips {
        crate::state::normalize_nonloop_source_window(clip);
    }
    timeline.sync_clip_takes_from_flat();

    let imported_notes = std::mem::take(&mut pf.notes_markdown);
    let imported_tempo_map = timeline.tempo_map.take();
    let imported_project_name = pf.name.clone();
    let source_name = if imported_project_name.trim().is_empty() {
        project_name_from_path(&path)
    } else {
        imported_project_name
    };

    let fragment = build_project_fragment(timeline, source_name.clone());
    if fragment.timeline.tracks.is_empty() && fragment.timeline.clips.is_empty() {
        if !imported_notes.trim().is_empty() {
            let mut project = state.project.lock().unwrap_or_else(|e| e.into_inner());
            project.notes_markdown = if project.notes_markdown.trim().is_empty() {
                imported_notes
            } else {
                format!(
                    "{}

## {} — {}

{}",
                    project.notes_markdown, "Imported", source_name, imported_notes
                )
            };
            project.dirty = true;
            update_window_title(&window, &project.name, project.dirty);
        }
        let mut json =
            serde_json::to_value(get_timeline_state_from_ref(&state)).unwrap_or_default();
        json["ok"] = serde_json::json!(true);
        json["empty"] = serde_json::json!(true);
        json["sourceProject"] = serde_json::json!(source_name);
        if !missing_files.is_empty() {
            json["missing_files"] = serde_json::json!(missing_files);
        }
        return json;
    }

    let (merge_result, scale_signature_before, tempo_map_imported, tempo_map_skipped) = {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        state.checkpoint_timeline(&tl);

        let scale_signature_before = tl.render_scale_signature();
        let anchor_sec = if place_at_playhead.unwrap_or(false) {
            Some(tl.playhead_sec.max(0.0))
        } else {
            None
        };

        let merge = match merge_project_fragment(
            &mut tl,
            &fragment,
            FragmentMergeOptions {
                anchor_sec,
                track_placement: FragmentTrackPlacement::AppendAtEnd,
            },
        ) {
            Ok(merge) => merge,
            Err(error) => return serde_json::json!({"ok": false, "error": error}),
        };

        for clip_id in &merge.created_clip_ids {
            if let Some(clip) = tl.clips.iter_mut().find(|clip| clip.id == *clip_id) {
                crate::state::TimelineState::populate_clip_file_metadata(clip);
            }
        }

        let affected_roots: std::collections::HashSet<String> = merge
            .created_clip_ids
            .iter()
            .filter_map(|clip_id| tl.clips.iter().find(|clip| clip.id == *clip_id))
            .filter_map(|clip| tl.resolve_root_track_id(&clip.track_id))
            .collect();
        for clip in &tl.clips {
            if tl
                .resolve_root_track_id(&clip.track_id)
                .as_ref()
                .map(|root| affected_roots.contains(root))
                .unwrap_or(false)
            {
                crate::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
                crate::formant_cache::invalidate_formant_cache_for_clip(&clip.id);
            }
        }

        let mut map_imported = false;
        let mut skipped = false;
        if let Some(mut points) = imported_tempo_map {
            if place_at_playhead.unwrap_or(false) {
                skipped = true;
            } else if tl.tempo_map.is_some() {
                skipped = true;
            } else if import_tempo_map.unwrap_or(true) {
                for point in &mut points {
                    point.id = crate::state::new_id("tp");
                }
                tl.tempo_map = Some(points);
                tl.normalize_tempo_map();
                map_imported = true;
            } else {
                skipped = true;
            }
        }

        if map_imported {
            let mut project = state.project.lock().unwrap_or_else(|e| e.into_inner());
            state.sync_project_record_from_tempo_map(&mut tl, &mut project);
            drop(project);
        }

        state.audio_engine.update_timeline(tl.clone());
        let scale_signature_after = tl.render_scale_signature();

        let mut payload = tl.to_payload();
        payload.created_clip_ids = Some(merge.created_clip_ids.clone());
        payload.created_track_ids = Some(merge.created_track_ids.clone());
        payload.project = Some(state.project_meta_payload());
        payload.missing_files = if missing_files.is_empty() {
            None
        } else {
            Some(missing_files.clone())
        };

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
            crate::pitch_analysis::maybe_schedule_pitch_orig(&state, root_id);
        }
        if let Some(handle) = state.app_handle.get() {
            crate::commands::playback::request_background_render(handle);
        }

        let mut json = serde_json::to_value(&payload).unwrap_or_default();
        json["sourceProject"] = serde_json::json!(source_name);
        json["importedTrackCount"] = serde_json::json!(merge.imported_track_count);
        json["importedClipCount"] = serde_json::json!(merge.imported_clip_count);
        json["tempoMapImported"] = serde_json::json!(map_imported);
        json["tempoMapSkipped"] = serde_json::json!(skipped);

        if scale_signature_before != scale_signature_after {
            for clip in &payload.clips {
                crate::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
            }
            if let Some(handle) = state.app_handle.get() {
                crate::commands::playback::request_background_render(handle);
            }
        }

        (merge, scale_signature_before, map_imported, skipped)
    };

    // Append imported notebook notes (if any) without replacing the current
    // notebook. This is deliberately done outside the timeline undo scope.
    if !imported_notes.trim().is_empty() {
        let mut project = state.project.lock().unwrap_or_else(|e| e.into_inner());
        let separator = format!("\n\n## {} — {}\n\n", "Imported", source_name);
        project.notes_markdown = if project.notes_markdown.trim().is_empty() {
            imported_notes
        } else {
            format!("{}{}{}", project.notes_markdown, separator, imported_notes)
        };
        project.dirty = true;
        update_window_title(&window, &project.name, project.dirty);
    }

    let mut json = serde_json::to_value(get_timeline_state_from_ref(&state)).unwrap_or_default();
    json["sourceProject"] = serde_json::json!(source_name);
    json["importedTrackCount"] = serde_json::json!(merge_result.imported_track_count);
    json["importedClipCount"] = serde_json::json!(merge_result.imported_clip_count);
    json["tempoMapImported"] = serde_json::json!(tempo_map_imported);
    json["tempoMapSkipped"] = serde_json::json!(tempo_map_skipped);
    if !missing_files.is_empty() {
        json["missing_files"] = serde_json::json!(missing_files);
    }
    let _ = scale_signature_before;
    json
}
