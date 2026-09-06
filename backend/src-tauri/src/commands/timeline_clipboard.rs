//! Native cross-process timeline clipboard commands.
//!
//! The data is a msgpack-encoded `ProjectFragment` transported through
//! `crate::system_clipboard`.  Copying happens in the backend so no WebView
//! clipboard permission is required, and pasting works across two running
//! HiFiShifter processes.
//!
//! 单剪贴板纪律：`OBJECT_FORMAT` 槽位是整个应用的**唯一**逻辑剪贴板，时间轴
//! Clip 与参数线载荷互相覆盖（最后复制的获胜），读取方只按自己的格式解析，
//! 解析失败一律视为"没有可粘贴的内容"（细节进日志），由上层给出统一的
//! 剪贴板为空提示并继续走 REAPER 回退 —— 不做跨类型判别。

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
    // 写入即替换整个逻辑剪贴板：参数线等其它载荷随 empty() 一并清除
    // （system_clipboard::write_contents），"只同时存在一个"由写入保证。
    system_clipboard::write_bytes_with_reaper(&bytes, &summary, reaper.map(|r| r.bytes.as_slice()))
}

fn read_fragment() -> Result<ProjectFragment, String> {
    let bytes =
        system_clipboard::read_bytes()?.ok_or_else(|| "timeline_clipboard_empty".to_string())?;
    // 槽位内容无法按时间轴载荷解析（被参数线复制覆盖 / 损坏 / 未知版本）时，
    // 一律按"没有可粘贴的内容"处理并继续上层 REAPER 回退；解析细节仅进日志，
    // 不把裸 serde 错误透给用户。
    ProjectFragment::decode(&bytes).map_err(|error| {
        log::warn!("[hifishifter] timeline clipboard payload unusable: {error}");
        "timeline_clipboard_empty".to_string()
    })
}

/// 判断槽位字节是否为参数线载荷（前端 `writeSystemClipboardObject` 写入的
/// JSON，`version:1` + `kind:"param"`）。
fn is_param_payload(bytes: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .ok()
        .and_then(|value| {
            let kind = value.get("kind")?.as_str()?.to_string();
            let version = value.get("version")?.as_u64()?;
            (kind == "param" && version == 1).then_some(kind)
        })
        .is_some()
}

/// 剪贴板当前载荷类型探测（粘贴的内容路由依据，last-copy-wins）。
///
/// 缓存优先：本进程写入后序号未变化时直接用缓存应答，不打开系统剪贴板
/// （打开是全局独占操作，见 system_clipboard 模块注释）。缓存失效时真实
/// 读取并按"时间轴载荷优先，其次参数线 JSON"判别，成功后回填缓存。
/// kind："clips" | "tracks" | "project" | "param" | null（空/外来/无法识别）。
pub(super) fn clipboard_kind() -> serde_json::Value {
    if let Some(cache) = system_clipboard::read_clipboard_cache_if_current() {
        return json!({ "ok": true, "kind": cache.hifi_kind });
    }

    let bytes = match system_clipboard::read_bytes() {
        Ok(Some(bytes)) if !bytes.is_empty() => bytes,
        _ => return json!({ "ok": true, "kind": null }),
    };

    if let Ok(fragment) = ProjectFragment::decode(&bytes) {
        let kind = match fragment.kind {
            ProjectFragmentKind::Clips => "clips",
            ProjectFragmentKind::Tracks => "tracks",
            ProjectFragmentKind::Project => "project",
        };
        if let Some(seq) = system_clipboard::clipboard_seq_num() {
            system_clipboard::write_clipboard_cache(ClipboardCacheEntry {
                seq,
                hifi_kind: Some(kind.to_string()),
                hifi_clip_count: fragment.timeline.clips.len() as u64,
                hifi_track_count: fragment.timeline.tracks.len() as u64,
                hifi_source_project: Some(fragment.source_project_name.clone()),
                reaper_available: system_clipboard::has_reaper_format(),
            });
        }
        return json!({ "ok": true, "kind": kind });
    }

    if is_param_payload(&bytes) {
        if let Some(seq) = system_clipboard::clipboard_seq_num() {
            system_clipboard::write_clipboard_cache(ClipboardCacheEntry {
                seq,
                hifi_kind: Some("param".to_string()),
                // 参数线载荷不参与时间轴可用性判定（counts 为 0 时
                // has_timeline_clipboard 视为时间轴不可用）。
                hifi_clip_count: 0,
                hifi_track_count: 0,
                hifi_source_project: None,
                reaper_available: system_clipboard::has_reaper_format(),
            });
        }
        return json!({ "ok": true, "kind": "param" });
    }

    json!({ "ok": true, "kind": null })
}

fn paste_placement(mode: Option<&str>) -> FragmentTrackPlacement {
    match mode.unwrap_or("selected") {
        "new_tracks" | "tracks" => FragmentTrackPlacement::AppendAtEnd,
        // 粘贴需求：起始目标轨道 = 当前选中轨道；多轨数据按显示顺序向下
        // 扩展，轨道不足自动新建（轨道/工程片段粘贴仍走 AppendAtEnd）。
        _ => FragmentTrackPlacement::SelectedTracksRelative,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 参数线载荷判别：与前端 writeSystemClipboardObject 写入的 JSON 契约一致
    /// （version:1 + kind:"param"）。
    #[test]
    fn param_payload_is_recognized() {
        let json = serde_json::json!({
            "version": 1,
            "kind": "param",
            "param": "pitch",
            "framePeriodMs": 5,
            "values": [123],
        });
        assert!(is_param_payload(json.to_string().as_bytes()));
    }

    #[test]
    fn non_param_payloads_are_rejected() {
        // 其它 kind / 版本不符 / 非 JSON（如时间轴 MessagePack 载荷）都不算参数线。
        let clip_json = serde_json::json!({ "version": 1, "kind": "clip", "param": "x" });
        assert!(!is_param_payload(clip_json.to_string().as_bytes()));
        let wrong_version = serde_json::json!({ "version": 2, "kind": "param" });
        assert!(!is_param_payload(wrong_version.to_string().as_bytes()));
        assert!(!is_param_payload(b"{\"kind\": \"param\"")); // 截断 JSON
        assert!(!is_param_payload(&[0x7b, 0x00, 0xff])); // 二进制
        assert!(!is_param_payload(b""));
    }
}
