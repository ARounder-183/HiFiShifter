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
use crate::system_clipboard::{self, ClipboardCacheEntry};
use serde_json::json;

fn current_project_name(state: &AppState) -> String {
    state
        .project
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .name
        .clone()
}

fn fragment_summary(fragment: &ProjectFragment) -> String {
    let clip_count = fragment.timeline.clips.len();
    let track_count = fragment.timeline.tracks.len();
    match fragment.kind {
        ProjectFragmentKind::Clips => format!(
            "HiFiShifter: {} clip(s) copied ({} track path(s)). Paste in HiFiShifter.",
            clip_count, track_count
        ),
        ProjectFragmentKind::Tracks => format!(
            "HiFiShifter: {} track(s) copied ({} clip(s)). Paste in HiFiShifter.",
            track_count, clip_count
        ),
        ProjectFragmentKind::Project => format!(
            "HiFiShifter: project fragment copied ({} track(s), {} clip(s)). Paste in HiFiShifter.",
            track_count, clip_count
        ),
    }
}

fn fragment_summary_with_reaper(
    fragment: &ProjectFragment,
    reaper: Option<&crate::reaper_export::ReaperExportResult>,
) -> String {
    let mut summary = fragment_summary(fragment);
    if let Some(reaper) = reaper {
        if reaper.exported_clip_count > 0 {
            summary.push_str(&format!(
                " REAPERMedia: {} clip(s) exported. Paste in REAPER.",
                reaper.exported_clip_count
            ));
        }
    }
    summary
}

/// Record the OS clipboard state right after a successful HiFiShifter
/// write, so `has_timeline_clipboard` can answer without reopening the
/// clipboard while the sequence number is unchanged.
fn record_written_clipboard(
    kind: ProjectFragmentKind,
    clip_count: usize,
    track_count: usize,
    source_project_name: &str,
    reaper_available: bool,
) {
    if let Some(seq) = system_clipboard::clipboard_seq_num() {
        system_clipboard::write_clipboard_cache(ClipboardCacheEntry {
            seq,
            hifi_kind: Some(match kind {
                ProjectFragmentKind::Clips => "clips".to_string(),
                ProjectFragmentKind::Tracks => "tracks".to_string(),
                ProjectFragmentKind::Project => "project".to_string(),
            }),
            hifi_clip_count: clip_count as u64,
            hifi_track_count: track_count as u64,
            hifi_source_project: Some(source_project_name.to_string()),
            reaper_available,
        });
    }
}

fn build_reaper_clipboard(
    timeline: &crate::state::TimelineState,
    clip_ids: &[String],
) -> Option<crate::reaper_export::ReaperExportResult> {
    crate::reaper_export::build_reaper_clipboard(timeline, clip_ids).ok()
}

fn write_fragment(
    fragment: &ProjectFragment,
    reaper: Option<&crate::reaper_export::ReaperExportResult>,
) -> Result<(), String> {
    let summary = fragment_summary_with_reaper(fragment, reaper);
    let bytes = fragment.encode()?;
    system_clipboard::write_bytes_with_reaper(&bytes, &summary, reaper.map(|r| r.bytes.as_slice()))
}

fn read_fragment() -> Result<ProjectFragment, String> {
    let bytes =
        system_clipboard::read_bytes()?.ok_or_else(|| "timeline_clipboard_empty".to_string())?;
    ProjectFragment::decode(&bytes)
}

fn paste_placement(mode: Option<&str>) -> FragmentTrackPlacement {
    match mode.unwrap_or("selected") {
        "new_tracks" | "tracks" => FragmentTrackPlacement::AppendAtEnd,
        _ => FragmentTrackPlacement::SelectedTrackOnly,
    }
}

fn paste_fragment(
    state: &AppState,
    fragment: ProjectFragment,
    mode: Option<String>,
) -> serde_json::Value {
    let result = {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        state.checkpoint_timeline(&tl);
        let playhead_sec = tl.playhead_sec.max(0.0);
        let track_placement = paste_placement(mode.as_deref());
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

        // Pasted clips get brand-new IDs, but linked parameter curves may be
        // written into an existing target root. Invalidate every clip on all
        // affected root tracks so playback never reuses a stale render whose
        // hash was computed before the paste.
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

        // Clip fragments only carry sliced automation (no pitch_orig). If the
        // target root has no analyzed pitch_orig yet, schedule analysis so
        // pasted pitch edits render correctly even before the next reopen.
        if fragment.kind == ProjectFragmentKind::Clips {
            for root_id in &affected_roots {
                crate::pitch_analysis::maybe_schedule_pitch_orig(state, root_id);
            }
        }
        for root_id in &midi_root_tracks {
            crate::pitch_analysis::maybe_schedule_pitch_orig(state, root_id);
        }
        if let Some(handle) = state.app_handle.get() {
            crate::commands::playback::request_background_render(handle);
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

    // The native HiFiShifter copy now also publishes REAPERMedia data so the
    // same selection can be pasted directly in REAPER.
    let reaper_clip_ids: Vec<String> = fragment
        .timeline
        .clips
        .iter()
        .map(|clip| clip.id.clone())
        .collect();
    let reaper = if reaper_clip_ids.is_empty() {
        None
    } else {
        build_reaper_clipboard(&fragment.timeline, &reaper_clip_ids)
    };
    drop(timeline);

    match write_fragment(&fragment, reaper.as_ref()) {
        Ok(()) => {
            record_written_clipboard(
                fragment.kind,
                fragment.timeline.clips.len(),
                fragment.timeline.tracks.len(),
                &fragment.source_project_name,
                reaper.is_some(),
            );
            json!({
                "ok": true,
                "kind": fragment.kind,
                "clipCount": fragment.timeline.clips.len(),
                "trackCount": fragment.timeline.tracks.len(),
                "reaperExportedClipCount": reaper.as_ref().map_or(0, |r| r.exported_clip_count),
                "reaperSkippedClipCount": reaper.as_ref().map_or(0, |r| r.skipped_clip_count),
                "reaperTrackCount": reaper.as_ref().map_or(0, |r| r.track_count),
            })
        }
        Err(error) => json!({ "ok": false, "error": error }),
    }
}

pub(super) fn copy_timeline_tracks(state: &AppState, track_ids: Vec<String>) -> serde_json::Value {
    let timeline = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    let fragment = match build_track_fragment(&timeline, &track_ids, current_project_name(state)) {
        Ok(fragment) => fragment,
        Err(error) => return json!({ "ok": false, "error": error }),
    };

    // Track copy includes every clip on the copied track (sub-)tree.
    let reaper_clip_ids: Vec<String> = fragment
        .timeline
        .clips
        .iter()
        .map(|clip| clip.id.clone())
        .collect();
    let reaper = if reaper_clip_ids.is_empty() {
        None
    } else {
        build_reaper_clipboard(&fragment.timeline, &reaper_clip_ids)
    };
    drop(timeline);

    match write_fragment(&fragment, reaper.as_ref()) {
        Ok(()) => {
            record_written_clipboard(
                fragment.kind,
                fragment.timeline.clips.len(),
                fragment.timeline.tracks.len(),
                &fragment.source_project_name,
                reaper.is_some(),
            );
            json!({
                "ok": true,
                "kind": "tracks",
                "clipCount": fragment.timeline.clips.len(),
                "trackCount": fragment.timeline.tracks.len(),
                "reaperExportedClipCount": reaper.as_ref().map_or(0, |r| r.exported_clip_count),
                "reaperSkippedClipCount": reaper.as_ref().map_or(0, |r| r.skipped_clip_count),
                "reaperTrackCount": reaper.as_ref().map_or(0, |r| r.track_count),
            })
        }
        Err(error) => json!({ "ok": false, "error": error }),
    }
}

pub(super) fn paste_timeline_clipboard(
    state: &AppState,
    mode: Option<String>,
) -> serde_json::Value {
    match read_fragment() {
        Ok(fragment) => paste_fragment(state, fragment, mode),
        Err(hifi_error) => {
            // No (or invalid) HiFiShifter clipboard data: fall back to the
            // REAPERMedia format so items copied in REAPER paste natively.
            let fallback = super::reaper_clipboard::paste_reaper_clipboard(state, None, None);
            if fallback.get("ok").and_then(serde_json::Value::as_bool) == Some(true) {
                return fallback;
            }
            let reaper_error = fallback
                .get("error")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("reaper_clipboard_empty");
            json!({ "ok": false, "error": format!("{}; Reaper clipboard: {}", hifi_error, reaper_error) })
        }
    }
}

pub(super) fn has_timeline_clipboard() -> serde_json::Value {
    // 剪贴板序列号未变化时用本进程缓存应答：避免前端每 2 秒的可用性轮询
    // 反复打开系统剪贴板（打开是全局独占操作，会放大与剪贴板管理器、
    // RDP、其它应用乃至复制/剪切写入之间的竞争窗口，正是 Clip 复制/剪切
    // 随机失败的来源之一）。
    if let Some(cache) = system_clipboard::read_clipboard_cache_if_current() {
        if cache.hifi_clip_count > 0 || cache.hifi_track_count > 0 {
            return json!({
                "ok": true,
                "available": true,
                "kind": cache.hifi_kind,
                "clipCount": cache.hifi_clip_count,
                "trackCount": cache.hifi_track_count,
                "sourceProject": cache.hifi_source_project,
            });
        }
        if cache.reaper_available {
            return json!({ "ok": true, "available": true, "kind": "reaper" });
        }
        return json!({ "ok": true, "available": false });
    }

    match read_fragment() {
        Ok(fragment) => {
            if let Some(seq) = system_clipboard::clipboard_seq_num() {
                system_clipboard::write_clipboard_cache(ClipboardCacheEntry {
                    seq,
                    hifi_kind: Some(match fragment.kind {
                        ProjectFragmentKind::Clips => "clips".to_string(),
                        ProjectFragmentKind::Tracks => "tracks".to_string(),
                        ProjectFragmentKind::Project => "project".to_string(),
                    }),
                    hifi_clip_count: fragment.timeline.clips.len() as u64,
                    hifi_track_count: fragment.timeline.tracks.len() as u64,
                    hifi_source_project: Some(fragment.source_project_name.clone()),
                    // 其它 HiFiShifter 进程的复制可能同时带 REAPERMedia，
                    // 用真实探测保证缓存标志准确。
                    reaper_available: system_clipboard::has_reaper_format(),
                });
            }
            json!({
                "ok": true,
                "available": true,
                "kind": fragment.kind,
                "clipCount": fragment.timeline.clips.len(),
                "trackCount": fragment.timeline.tracks.len(),
                "sourceProject": fragment.source_project_name,
            })
        }
        Err(_) => {
            let reaper = super::reaper_clipboard::has_reaper_clipboard();
            if let Some(seq) = system_clipboard::clipboard_seq_num() {
                system_clipboard::write_clipboard_cache(ClipboardCacheEntry {
                    seq,
                    hifi_kind: None,
                    hifi_clip_count: 0,
                    hifi_track_count: 0,
                    hifi_source_project: None,
                    reaper_available: reaper,
                });
            }
            if reaper {
                json!({ "ok": true, "available": true, "kind": "reaper" })
            } else {
                json!({ "ok": true, "available": false })
            }
        }
    }
}
