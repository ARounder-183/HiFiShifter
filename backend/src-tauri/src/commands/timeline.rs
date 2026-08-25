use crate::state::AppState;
use crate::state::SplitTransitionDurationUnit;
use crate::state::SplitTransitionMode;
use crate::state::SplitTransitionOptions;
use base64::Engine;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use tauri::Emitter;
use tauri::Manager;
use tauri::State;
use uuid::Uuid;

use super::common::ensure_temp_dir;

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ClipFormantStatusPayload {
    clip_id: String,
    status: String,
}

fn emit_clip_formant_status(app: &tauri::AppHandle, clip_id: &str, status: &str) {
    let _ = app.emit(
        "clip_formant_status",
        ClipFormantStatusPayload {
            clip_id: clip_id.to_string(),
            status: status.to_string(),
        },
    );
}

fn clip_formant_rebuild_needs_refresh(
    before: Option<&crate::state::Clip>,
    after: &crate::state::Clip,
) -> bool {
    let before_enabled = before
        .and_then(|clip| clip.formant_morph.as_ref())
        .map(|params| params.enabled)
        .unwrap_or(false);
    let after_enabled = after
        .formant_morph
        .as_ref()
        .map(|params| params.enabled)
        .unwrap_or(false);

    if !before_enabled && !after_enabled {
        return false;
    }

    let before_params = before.and_then(|clip| clip.formant_morph.as_ref());
    let after_params = after.formant_morph.as_ref();
    let params_changed = before_params != after_params;
    let source_changed = before
        .map(|clip| {
            clip.source_path != after.source_path
                || (clip.source_start_sec - after.source_start_sec).abs() > 1e-9
                || (clip.source_end_sec - after.source_end_sec).abs() > 1e-9
                || clip.reversed != after.reversed
        })
        .unwrap_or(after_enabled);

    params_changed || source_changed
}

fn schedule_clip_formant_rebuild(state: &AppState, clip: crate::state::Clip) {
    let Some(app) = state.app_handle.get().cloned() else {
        return;
    };
    let clip_id = clip.id.clone();
    let Some(formant) = clip.formant_morph.as_ref() else {
        crate::formant_cache::cancel_formant_rebuild_generation(&clip_id);
        return;
    };
    if !formant.enabled {
        crate::formant_cache::cancel_formant_rebuild_generation(&clip_id);
        return;
    }

    let generation = crate::formant_cache::begin_formant_rebuild_generation(&clip_id);
    if let Some(formant) = clip.formant_morph.as_ref() {
        crate::formant_cache::formant_debug_log(format!(
            "schedule rebuild clip_id={} generation={} f1={:.1} f2={:.1} strength={:.3} source={} range=[{:.3},{:.3}] reversed={}",
            clip_id,
            generation,
            formant.target_f1_hz,
            formant.target_f2_hz,
            formant.strength,
            clip.source_path.as_deref().unwrap_or(""),
            clip.source_start_sec,
            clip.source_end_sec,
            clip.reversed,
        ));
    }
    emit_clip_formant_status(&app, &clip_id, "rebuilding");

    let out_rate = state.audio_engine.sample_rate_hz().max(8_000);
    std::thread::spawn(move || {
        let result = crate::formant_cache::compute_formant_cache_entry_for_clip(&clip, out_rate);

        if !crate::formant_cache::is_current_formant_rebuild_generation(&clip_id, generation) {
            crate::formant_cache::formant_debug_log(format!(
                "discard stale rebuild clip_id={} generation={}",
                clip_id, generation
            ));
            return;
        }

        match result {
            Ok((key, entry)) => {
                crate::formant_cache::formant_debug_log(format!(
                    "rebuild ready clip_id={} generation={} frames={} sr={}",
                    clip_id, generation, entry.frames, entry.sample_rate
                ));
                crate::formant_cache::insert_formant_cache_entry(key, entry);
                emit_clip_formant_status(&app, &clip_id, "ready");
                let state = app.state::<AppState>();
                let timeline = match state.timeline.lock() {
                    Ok(guard) => guard.clone(),
                    Err(poisoned) => poisoned.into_inner().clone(),
                };
                state.audio_engine.update_timeline(timeline);
            }
            Err(error) => {
                crate::formant_cache::formant_debug_log(format!(
                    "rebuild failed clip_id={} generation={} error={}",
                    clip_id, generation, error
                ));
                emit_clip_formant_status(&app, &clip_id, "failed");
            }
        }
    });
}

// ===================== dialogs / io =====================

pub(super) fn import_audio_bytes(
    state: State<'_, AppState>,
    file_name: String,
    base64_data: String,
    track_id: Option<Option<String>>,
    start_sec: Option<f64>,
) -> crate::models::TimelineStatePayload {
    if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
        eprintln!(
            "import_audio_bytes(file_name={}, base64_len={}, track_id={:?}, start_sec={:?})",
            file_name,
            base64_data.len(),
            track_id,
            start_sec
        );
    }
    let engine = base64::engine::general_purpose::STANDARD;
    let bytes = engine.decode(base64_data.as_bytes()).unwrap_or_default();

    let ext = Path::new(&file_name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("bin");
    let tmp_dir = ensure_temp_dir().ok();
    let path = tmp_dir.unwrap_or_else(std::env::temp_dir).join(format!(
        "{}_{}.{}",
        "import",
        Uuid::new_v4().simple(),
        ext
    ));

    let _ = fs::write(&path, &bytes);

    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    let resolved_track_id: Option<String> = match track_id {
        None => None,
        Some(Some(id)) => Some(id),
        Some(None) => Some(tl.add_track(Some("Track".to_string()), None, None)),
    };

    tl.import_audio_item(&path.display().to_string(), resolved_track_id, start_sec);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(super) fn import_audio_item(
    state: State<'_, AppState>,
    audio_path: String,
    track_id: Option<Option<String>>,
    start_sec: Option<f64>,
    media_audio_stream_index: Option<usize>,
) -> crate::models::TimelineStatePayload {
    if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
        eprintln!(
            "import_audio_item(audio_path={}, track_id={:?}, start_sec={:?}, stream={:?})",
            audio_path, track_id, start_sec, media_audio_stream_index
        );
    }

    // 多音轨视频：用户显式选择某条音轨时，先把该音轨抽取为源文件旁边的
    // WAV 缓存，再走普通导入流程。未指定时保留原始媒体路径，由 Symphonia
    // 在需要时直接解码默认音轨。
    let source_path = if let Some(stream_index) = media_audio_stream_index {
        if crate::media::is_video_extension(std::path::Path::new(&audio_path)) {
            match crate::media::extract_audio_stream_to_wav(
                std::path::Path::new(&audio_path),
                stream_index,
            ) {
                Ok(path) => path,
                Err(error) => {
                    let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
                    let mut payload = tl.to_payload();
                    payload.project = Some(state.project_meta_payload());
                    payload.ok = false;
                    payload.created_clip_ids = Some(vec![]);
                    payload.missing_files = Some(vec![error]);
                    return payload;
                }
            }
        } else {
            audio_path.clone()
        }
    } else {
        audio_path.clone()
    };

    let source_meta = std::path::Path::new(&source_path);
    if source_meta.exists()
        && crate::audio_utils::try_read_audio_header_only(source_meta).is_none()
    {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        let mut payload = tl.to_payload();
        payload.project = Some(state.project_meta_payload());
        payload.ok = false;
        payload.created_clip_ids = Some(vec![]);
        payload.missing_files = Some(vec![format!(
            "media_has_no_audio_or_unsupported_codec: {}",
            source_meta.display()
        )]);
        return payload;
    }

    {
        let mut rt = state.runtime.lock().unwrap_or_else(|e| e.into_inner());
        rt.audio_loaded = true;
    }

    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    let resolved_track_id: Option<String> = match track_id {
        None => None,
        Some(Some(id)) => Some(id),
        Some(None) => Some(tl.add_track(Some("Track".to_string()), None, None)),
    };

    tl.import_audio_item(&source_path, resolved_track_id, start_sec);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

// ===================== timeline CRUD =====================

pub(super) fn add_track(
    state: State<'_, AppState>,
    name: Option<String>,
    parent_track_id: Option<String>,
    index: Option<usize>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    tl.add_track(name, parent_track_id, index);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(super) fn remove_track(
    state: State<'_, AppState>,
    track_id: String,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);

    // 删除前：BFS 收集将被删除的轨道 ID 及其关联的 clip ID，用于后续清理全局缓存。
    let (clip_ids_to_clean, root_track_ids_to_clean) = {
        let mut to_remove = vec![track_id.clone()];
        let mut idx = 0;
        while idx < to_remove.len() {
            let cur = to_remove[idx].clone();
            for child in tl
                .tracks
                .iter()
                .filter(|t| t.parent_id.as_deref() == Some(cur.as_str()))
                .map(|t| t.id.clone())
            {
                to_remove.push(child);
            }
            idx += 1;
        }
        let remove_set: std::collections::HashSet<&str> =
            to_remove.iter().map(|s| s.as_str()).collect();
        let clip_ids: Vec<String> = tl
            .clips
            .iter()
            .filter(|c| remove_set.contains(c.track_id.as_str()))
            .map(|c| c.id.clone())
            .collect();
        (clip_ids, to_remove)
    };

    tl.remove_track(&track_id);
    state.audio_engine.update_timeline(tl.clone());

    // 清理被删除 clip 的全局合成缓存和渲染状态，防止内存泄漏和旧数据残留。
    for clip_id in &clip_ids_to_clean {
        crate::synth_clip_cache::invalidate_clip_all_caches(clip_id);
    }

    // 将锁的获取移到循环外部，避免 O(N) 的锁争用开销
    if let Ok(mut mgr) = crate::clip_rendering_state::global_clip_rendering_state().lock() {
        for clip_id in &clip_ids_to_clean {
            mgr.remove_state(clip_id);
        }
    }

    // 清理被删除轨道的 pitch_timeline_snapshot，防止增量分析数据残留。
    if let Ok(mut snapshot_map) = state.pitch_timeline_snapshot.lock() {
        for root_id in &root_track_ids_to_clean {
            snapshot_map.remove(root_id);
        }
    }

    // 清理 pitch_inflight 中包含被删轨道 ID 的去重 key。
    if let Ok(mut inflight) = state.pitch_inflight.lock() {
        inflight.retain(|key| {
            !root_track_ids_to_clean
                .iter()
                .any(|tid| key.contains(tid.as_str()))
        });
    }

    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(super) fn duplicate_track(
    state: State<'_, AppState>,
    track_id: String,
    parent_track_id: Option<String>,
    target_index: Option<usize>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    // 以 targetIndex 是否存在作为“复制拖动”放置语义的开关：
    // - Some(index)：克隆子树移动到指定位置。注意 parentTrackId 为 null
    //   （serde 反序列化为 None）代表根层级，是完全合法的放置目标，
    //   绝不能作为回退到默认行为的条件——否则所有根级拖放都会退化成
    //   “紧贴源轨道克隆”。
    // - None：未提供位置（右键菜单“克隆轨道”），保持紧贴源轨道。
    let new_track_ids = match target_index {
        Some(index) => tl.duplicate_track_to(&track_id, parent_track_id, index),
        None => tl.duplicate_track(&track_id),
    };
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.created_track_ids = Some(new_track_ids);
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(super) fn move_track(
    state: State<'_, AppState>,
    track_id: String,
    target_index: usize,
    parent_track_id: Option<String>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    tl.move_track(&track_id, target_index, parent_track_id);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(super) fn set_track_state(
    state: State<'_, AppState>,
    track_id: String,
    muted: Option<bool>,
    solo: Option<bool>,
    volume: Option<f32>,
    compose_enabled: Option<bool>,
    pitch_analysis_algo: Option<String>,
    color: Option<String>,
    name: Option<String>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    let algo = pitch_analysis_algo.as_deref().map(|s| match s {
        "world_dll" | "world" => crate::state::PitchAnalysisAlgo::WorldDll,
        "nsf_hifigan_onnx" | "nsf_hifigan" | "onnx" => {
            crate::state::PitchAnalysisAlgo::NsfHifiganOnnx
        }
        "vslib" | "vocalshifter_vslib" => crate::state::PitchAnalysisAlgo::VocalShifterVslib,
        "none" => crate::state::PitchAnalysisAlgo::None,
        _ => crate::state::PitchAnalysisAlgo::Unknown,
    });
    tl.set_track_state(
        &track_id,
        muted,
        solo,
        volume,
        compose_enabled,
        algo,
        color,
        name,
    );
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(super) fn select_track(
    state: State<'_, AppState>,
    track_id: String,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    tl.select_track(&track_id);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(super) fn set_project_length(
    state: State<'_, AppState>,
    project_sec: f64,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    tl.set_project_length(project_sec);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(super) fn add_clip(
    state: State<'_, AppState>,
    track_id: Option<String>,
    name: Option<String>,
    start_sec: Option<f64>,
    length_sec: Option<f64>,
    source_path: Option<String>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    tl.add_clip(track_id, name, start_sec, length_sec, source_path);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(super) fn create_clips_bulk(
    state: State<'_, AppState>,
    payload: crate::state::CreateClipsBulkPayload,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    let created_clip_ids = tl.create_clips_bulk(&payload);
    state.audio_engine.update_timeline(tl.clone());
    let mut timeline_payload = tl.to_payload();
    timeline_payload.created_clip_ids = Some(created_clip_ids);
    timeline_payload.project = Some(state.project_meta_payload());
    timeline_payload
}

pub(super) fn remove_clip(
    state: State<'_, AppState>,
    clip_id: String,
) -> crate::models::TimelineStatePayload {
    remove_clips(state, vec![clip_id])
}

pub(super) fn remove_clips(
    state: State<'_, AppState>,
    clip_ids: Vec<String>,
) -> crate::models::TimelineStatePayload {
    for clip_id in &clip_ids {
        crate::formant_cache::cancel_formant_rebuild_generation(clip_id);
    }
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);

    // 波纹编辑：删除前的被删除剪辑信息（用于计算平移量与轨道归属）。
    let (ripple_mode, ripple_link) = ripple_settings(&state);
    let mut affected_root_tracks: HashSet<String> = HashSet::new();
    let mut origin = f64::INFINITY;
    let mut old_right_edge = f64::NEG_INFINITY;
    let mut removed_tracks: HashSet<String> = HashSet::new();
    let mut found_any = false;

    for clip_id in &clip_ids {
        if let Some(clip) = tl.clips.iter().find(|c| c.id == *clip_id) {
            found_any = true;
            origin = origin.min(clip.start_sec);
            old_right_edge = old_right_edge.max(clip.start_sec + clip.length_sec);
            removed_tracks.insert(clip.track_id.clone());
            if let Some(root_id) = tl.resolve_root_track_id(&clip.track_id) {
                affected_root_tracks.insert(root_id);
            }
        }
    }

    let edited: Vec<&str> = clip_ids.iter().map(|s| s.as_str()).collect();
    tl.remove_clips(&clip_ids);

    // 波纹编辑：删除后“收拢”时间轴——后续剪辑左移（origin - old_right_edge）。
    if ripple_mode != crate::state::RippleMode::Off && found_any {
        let delta = origin - old_right_edge;
        let affected_tracks = match ripple_mode {
            crate::state::RippleMode::All => None,
            crate::state::RippleMode::Track => Some(removed_tracks),
            crate::state::RippleMode::Off => None,
        };
        let shifted = tl.ripple_shift_clips(
            &edited,
            affected_tracks.as_ref(),
            origin,
            delta,
            ripple_link,
        );
        for clip_id in shifted {
            if let Some(clip) = tl.clips.iter().find(|c| c.id == clip_id) {
                if let Some(root_id) = tl.resolve_root_track_id(&clip.track_id) {
                    affected_root_tracks.insert(root_id);
                }
            }
        }
    }

    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    drop(tl);
    for root_id in affected_root_tracks {
        crate::pitch_analysis::maybe_schedule_pitch_orig(&state, &root_id);
    }
    payload
}

pub(super) fn move_clip(
    state: State<'_, AppState>,
    clip_id: String,
    start_sec: f64,
    track_id: Option<String>,
    move_linked_params: Option<bool>,
) -> crate::models::TimelineStatePayload {
    move_clips(
        state,
        vec![crate::state::MoveClipPayload {
            clip_id,
            start_sec,
            track_id,
        }],
        move_linked_params,
    )
}

pub(super) fn move_clips(
    state: State<'_, AppState>,
    moves: Vec<crate::state::MoveClipPayload>,
    move_linked_params: Option<bool>,
) -> crate::models::TimelineStatePayload {
    let move_linked_params = move_linked_params.unwrap_or(false);
    let payload = {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        state.checkpoint_timeline(&tl);

        // 波纹编辑：记录被编辑剪辑的移动前状态（起点 / 右边缘 / 原轨道）。
        let (ripple_mode, ripple_link) = ripple_settings(&state);
        let mut affected_root_tracks: HashSet<String> = HashSet::new();
        let before: HashMap<String, (String, f64, f64)> = moves
            .iter()
            .filter_map(|m| {
                tl.clips.iter().find(|c| c.id == m.clip_id).map(|c| {
                    (
                        c.id.clone(),
                        (c.track_id.clone(), c.start_sec, c.length_sec),
                    )
                })
            })
            .collect();

        // 移动前轨道所属 root（换轨时旧 root 也需要重分析音高）。
        for (track_id, _, _) in before.values() {
            if let Some(root_id) = tl.resolve_root_track_id(track_id) {
                affected_root_tracks.insert(root_id);
            }
        }

        tl.move_clips(&moves, move_linked_params);

        // 波纹编辑：以“编辑区右边缘的变化量”平移后续剪辑。
        if ripple_mode != crate::state::RippleMode::Off && !before.is_empty() {
            let origin = before
                .values()
                .map(|(_, s, _)| *s)
                .fold(f64::INFINITY, f64::min);
            let old_right_edge = before
                .values()
                .map(|(_, s, l)| *s + *l)
                .fold(f64::NEG_INFINITY, f64::max);
            let new_right_edge = moves
                .iter()
                .filter_map(|m| {
                    tl.clips
                        .iter()
                        .find(|c| c.id == m.clip_id)
                        .map(|c| c.start_sec + c.length_sec)
                })
                .fold(f64::NEG_INFINITY, f64::max);
            let delta = new_right_edge - old_right_edge;

            let affected_tracks = match ripple_mode {
                crate::state::RippleMode::All => None,
                crate::state::RippleMode::Track => Some(
                    before
                        .values()
                        .map(|(t, _, _)| t.clone())
                        .collect::<HashSet<String>>(),
                ),
                crate::state::RippleMode::Off => None,
            };
            let edited: Vec<&str> = before.keys().map(|s| s.as_str()).collect();
            let shifted = tl.ripple_shift_clips(
                &edited,
                affected_tracks.as_ref(),
                origin,
                delta,
                ripple_link,
            );
            for clip_id in shifted {
                if let Some(clip) = tl.clips.iter().find(|c| c.id == clip_id) {
                    if let Some(root_id) = tl.resolve_root_track_id(&clip.track_id) {
                        affected_root_tracks.insert(root_id);
                    }
                }
            }
        }

        // 移动后的新 root 也应重分析。
        for m in &moves {
            if let Some(clip) = tl.clips.iter().find(|c| c.id == m.clip_id) {
                if let Some(root_id) = tl.resolve_root_track_id(&clip.track_id) {
                    affected_root_tracks.insert(root_id);
                }
            }
        }

        state.audio_engine.update_timeline(tl.clone());
        let mut payload = tl.to_payload();
        payload.project = Some(state.project_meta_payload());
        drop(tl);

        for root_id in affected_root_tracks {
            crate::pitch_analysis::maybe_schedule_pitch_orig(&state, &root_id);
        }
        payload
    };
    payload
}

pub(super) fn get_clip_linked_params(
    state: State<'_, AppState>,
    clip_id: String,
) -> serde_json::Value {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    match tl.extract_clip_linked_params(&clip_id) {
        Some(linked_params) => serde_json::json!({
            "ok": true,
            "linkedParams": linked_params,
        }),
        None => serde_json::json!({
            "ok": false,
            "error": "clip_not_found",
        }),
    }
}

pub(super) fn apply_clip_linked_params(
    state: State<'_, AppState>,
    clip_id: String,
    linked_params: crate::state::LinkedParamCurvesPayload,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    tl.apply_linked_params_to_clip(&clip_id, &linked_params);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

#[allow(clippy::too_many_arguments)]

pub(super) fn set_clip_state(
    state: State<'_, AppState>,
    clip_id: String,
    name: Option<String>,
    start_sec: Option<f64>,
    length_sec: Option<f64>,
    gain: Option<f32>,
    muted: Option<bool>,
    source_start_sec: Option<f64>,
    source_end_sec: Option<f64>,
    playback_rate: Option<f32>,
    reversed: Option<bool>,
    loop_enabled: Option<bool>,
    snap_offset_sec: Option<f64>,
    fade_in_sec: Option<f64>,
    fade_out_sec: Option<f64>,
    fade_in_curve: Option<String>,
    fade_out_curve: Option<String>,
    auto_fade_in_sec: Option<f64>,
    auto_fade_out_sec: Option<f64>,
    color: Option<String>,
    formant_morph: Option<crate::state::ClipFormantMorph>,
    checkpoint: Option<bool>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    let previous_clip = tl.clips.iter().find(|clip| clip.id == clip_id).cloned();
    // checkpoint 默认为 true，但可以通过传递 false 来抑制 undo checkpoint
    // 这在 undo group 内进行多次操作时很有用
    let do_checkpoint = checkpoint.unwrap_or(true);
    if do_checkpoint {
        state.checkpoint_timeline(&tl);
    }
    tl.patch_clip_state(
        &clip_id,
        crate::state::ClipStatePatch {
            name,
            start_sec,
            length_sec,
            gain,
            muted,
            source_start_sec,
            source_end_sec,
            playback_rate,
            reversed,
            loop_enabled,
            snap_offset_sec,
            fade_in_sec,
            fade_out_sec,
            fade_in_curve,
            fade_out_curve,
            auto_fade_in_sec,
            auto_fade_out_sec,
            color,
            formant_morph,
        },
    );
    // 波纹编辑（自动跟进）：当起点/长度改变（右边缘位移）时，平移后续剪辑。
    let mut ripple_root_track_ids: HashSet<String> = HashSet::new();
    if start_sec.is_some() || length_sec.is_some() {
        let (ripple_mode, ripple_link) = ripple_settings(&state);
        if ripple_mode != crate::state::RippleMode::Off {
            if let Some(before_clip) = previous_clip.as_ref() {
                let old_end = before_clip.start_sec + before_clip.length_sec;
                if let Some(after) = tl.clips.iter().find(|c| c.id == clip_id) {
                    let new_end = after.start_sec + after.length_sec;
                    let delta = new_end - old_end;
                    if delta.abs() > 1e-9 {
                        let affected_tracks = match ripple_mode {
                            crate::state::RippleMode::All => None,
                            crate::state::RippleMode::Track => Some(HashSet::from([
                                before_clip.track_id.clone(),
                            ])),
                            crate::state::RippleMode::Off => None,
                        };
                        let shifted = tl.ripple_shift_clips(
                            &[clip_id.as_str()],
                            affected_tracks.as_ref(),
                            before_clip.start_sec,
                            delta,
                            ripple_link,
                        );
                        for shifted_id in shifted {
                            if let Some(clip) = tl.clips.iter().find(|c| c.id == shifted_id) {
                                if let Some(root_id) = tl.resolve_root_track_id(&clip.track_id) {
                                    ripple_root_track_ids.insert(root_id);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let next_clip = tl.clips.iter().find(|clip| clip.id == clip_id).cloned();
    let root_track_id = tl
        .clips
        .iter()
        .find(|c| c.id == clip_id)
        .map(|c| c.track_id.clone())
        .and_then(|tid| tl.resolve_root_track_id(&tid));
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    drop(tl);

    if let Some(root_id) = root_track_id {
        crate::pitch_analysis::maybe_schedule_pitch_orig(&state, &root_id);
    }
    for root_id in ripple_root_track_ids {
        crate::pitch_analysis::maybe_schedule_pitch_orig(&state, &root_id);
    }
    if let Some(next_clip) = next_clip {
        if clip_formant_rebuild_needs_refresh(previous_clip.as_ref(), &next_clip) {
            schedule_clip_formant_rebuild(&state, next_clip);
        }
    }
    payload
}

pub(super) fn set_clips_state_bulk(
    state: State<'_, AppState>,
    updates: Vec<crate::state::BulkClipStatePatch>,
    checkpoint: Option<bool>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    if checkpoint.unwrap_or(true) {
        state.checkpoint_timeline(&tl);
    }
    // 波纹编辑（自动跟进，防御性）：批量更新可能携带起点/长度（尺寸）变更时生效。
    // 前端当前的 set_clips_state_bulk 只传 gain/muted/fades，不会触发；该路径用于覆盖
    // 未来“多选尺寸调整”类调用。每个被改尺寸的剪辑按“右边缘位移”独立平移后续剪辑。
    let resized_before: Vec<(String, String, f64, f64)> = updates
        .iter()
        .filter(|u| u.patch.start_sec.is_some() || u.patch.length_sec.is_some())
        .filter_map(|u| {
            tl.clips.iter().find(|c| c.id == u.clip_id).map(|c| {
                (
                    u.clip_id.clone(),
                    c.track_id.clone(),
                    c.start_sec,
                    c.start_sec + c.length_sec,
                )
            })
        })
        .collect();

    tl.patch_clips_state(&updates);
    let mut root_track_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for update in &updates {
        if let Some(clip) = tl.clips.iter().find(|c| c.id == update.clip_id) {
            if let Some(root) = tl.resolve_root_track_id(&clip.track_id) {
                root_track_ids.insert(root);
            }
        }
    }

    let (ripple_mode, ripple_link) = ripple_settings(&state);
    let mut ripple_roots: HashSet<String> = HashSet::new();
    if ripple_mode != crate::state::RippleMode::Off && !resized_before.is_empty() {
        // 区域化波纹：把被批量重设尺寸的剪辑视为一个“编辑区域”，以区域最左起点
        // 为原点、区域右缘净位移为平移量，一次平移后续剪辑；避免多个成员各自波纹
        // 时后置跟随剪辑被重复平移（与 move/delete 的“单次波纹”语义一致）。
        let origin = resized_before
            .iter()
            .map(|(_, _, s, _)| *s)
            .fold(f64::INFINITY, f64::min);
        let old_right_edge = resized_before
            .iter()
            .map(|(_, _, _, e)| *e)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut new_right_edge = old_right_edge;
        for (clip_id, _, _, _) in &resized_before {
            if let Some(after) = tl.clips.iter().find(|c| c.id == *clip_id) {
                new_right_edge = new_right_edge.max(after.start_sec + after.length_sec);
            }
        }
        let delta = new_right_edge - old_right_edge;
        if delta.abs() > 1e-9 {
            let edited: Vec<&str> = resized_before
                .iter()
                .map(|(id, _, _, _)| id.as_str())
                .collect();
            let affected_tracks = match ripple_mode {
                crate::state::RippleMode::All => None,
                crate::state::RippleMode::Track => Some(
                    resized_before
                        .iter()
                        .map(|(_, track_id, _, _)| track_id.clone())
                        .collect::<HashSet<String>>(),
                ),
                crate::state::RippleMode::Off => None,
            };
            let shifted = tl.ripple_shift_clips(
                &edited,
                affected_tracks.as_ref(),
                origin,
                delta,
                ripple_link,
            );
            for shifted_id in shifted {
                if let Some(clip) = tl.clips.iter().find(|c| c.id == shifted_id) {
                    if let Some(root) = tl.resolve_root_track_id(&clip.track_id) {
                        ripple_roots.insert(root);
                    }
                }
            }
        }
    }

    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    drop(tl);
    for root in &root_track_ids {
        crate::pitch_analysis::maybe_schedule_pitch_orig(&state, root);
    }
    for root in ripple_roots {
        crate::pitch_analysis::maybe_schedule_pitch_orig(&state, &root);
    }
    payload
}

pub(super) fn duplicate_clips_bulk(
    state: State<'_, AppState>,
    payload: crate::state::DuplicateClipsBulkPayload,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    let created_clip_ids = tl.duplicate_clips_bulk(&payload);
    state.audio_engine.update_timeline(tl.clone());
    let mut timeline_payload = tl.to_payload();
    timeline_payload.created_clip_ids = Some(created_clip_ids);
    timeline_payload.project = Some(state.project_meta_payload());
    timeline_payload
}

pub(super) fn replace_clip_source(
    state: State<'_, AppState>,
    clip_ids: Vec<String>,
    new_source_path: String,
    replace_same_source: Option<bool>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);

    // 收集被替换 clip 的旧源路径
    let old_paths: Vec<String> = tl
        .clips
        .iter()
        .filter(|c| clip_ids.contains(&c.id) && c.source_path.is_some())
        .map(|c| c.source_path.clone().unwrap())
        .collect();

    // ★ 关键：先发送缓存失效命令到引擎，再发送 update_timeline。
    // 引擎按 FIFO 顺序处理，确保 build_snapshot 之前解码/拉伸缓存已被清空，
    // 避免 snapshot 中使用旧文件的解码数据。
    state.audio_engine.evict_source_path(&new_source_path);
    for old_path in &old_paths {
        if old_path != &new_source_path {
            state.audio_engine.evict_source_path(old_path);
        }
    }

    tl.replace_clip_sources(
        &clip_ids,
        &new_source_path,
        replace_same_source.unwrap_or(false),
    );
    state.audio_engine.update_timeline(tl.clone());

    // 使波形峰值内存缓存失效：新路径和所有旧路径都需要清理
    state.invalidate_waveform_cache_for_path(&new_source_path);
    for old_path in &old_paths {
        if old_path != &new_source_path {
            state.invalidate_waveform_cache_for_path(old_path);
        }
    }

    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

/// 检查所有已导入的媒体源文件是否被外部修改或删除。
/// 前端在窗口重新获得焦点时调用此命令，以便提示用户做出相应处理。
pub(super) fn check_source_files_changed(
    state: State<'_, AppState>,
) -> crate::models::CheckSourceFilesChangedPayload {
    let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    tl.check_source_files_changed()
}

const MAX_SOURCE_MATCH_CANDIDATES_PER_CLIP: usize = 200;

/// 取路径的小写扩展名（不含点）。
fn path_extension_key(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
}

fn collect_source_file_match_candidates(
    dir: &Path,
    targets_by_name: &HashMap<std::ffi::OsString, Vec<(String, Option<u64>)>>,
    targets_by_extension: &HashMap<String, Vec<(String, Option<u64>)>>,
    mode: crate::models::SearchSourceFileMode,
    out: &mut HashMap<String, Vec<crate::models::SourceFileMatchCandidatePayload>>,
) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        if file_type.is_dir() {
            collect_source_file_match_candidates(
                &path,
                targets_by_name,
                targets_by_extension,
                mode,
                out,
            );
            continue;
        }
        if !file_type.is_file() {
            continue;
        }

        let expected_targets = match mode {
            crate::models::SearchSourceFileMode::ByFileName => {
                let Some(file_name) = path.file_name() else {
                    continue;
                };
                let Some(targets) = targets_by_name.get(file_name) else {
                    continue;
                };
                targets
            }
            crate::models::SearchSourceFileMode::ByExtensionHash => {
                let Some(ext) = path_extension_key(&path) else {
                    continue;
                };
                let Some(targets) = targets_by_extension.get(&ext) else {
                    continue;
                };
                targets
            }
        };

        // 同一文件只计算一次指纹；哈希逻辑与“重新捕获缺失媒体”检测完全一致。
        let actual_fingerprint = crate::audio_utils::compute_file_fingerprint(&path);
        let candidate_path = path.display().to_string();

        for (clip_id, expected_fingerprint) in expected_targets {
            let candidates = out.entry(clip_id.clone()).or_default();

            match mode {
                crate::models::SearchSourceFileMode::ByFileName => {
                    if candidates.len() >= MAX_SOURCE_MATCH_CANDIDATES_PER_CLIP {
                        continue;
                    }
                    let exact_hash = match (expected_fingerprint, actual_fingerprint) {
                        (Some(expected), Some(actual)) => *expected == actual,
                        _ => false,
                    };
                    candidates.push(crate::models::SourceFileMatchCandidatePayload {
                        path: candidate_path.clone(),
                        exact_hash,
                    });
                }
                crate::models::SearchSourceFileMode::ByExtensionHash => {
                    // 扩展名 + 哈希模式只展示内容指纹完全一致的候选。
                    let exact_hash = match (expected_fingerprint, actual_fingerprint) {
                        (Some(expected), Some(actual)) => *expected == actual,
                        _ => false,
                    };
                    if !exact_hash {
                        continue;
                    }
                    if candidates.len() >= MAX_SOURCE_MATCH_CANDIDATES_PER_CLIP {
                        continue;
                    }
                    candidates.push(crate::models::SourceFileMatchCandidatePayload {
                        path: candidate_path.clone(),
                        exact_hash: true,
                    });
                }
            }
        }
    }
}

/// 在指定文件夹及其子文件夹中搜索候选源文件。
///
/// - `ByFileName`：按源文件的完整文件名精确匹配每个候选文件，并使用与
///   “重新捕获缺失媒体”检测完全相同的 `compute_file_fingerprint` 逻辑计算
///   指纹；哈希完全一致的候选会标为 `exact_hash = true` 且排在前面。
/// - `ByExtensionHash`：遍历文件夹中所有扩展名与源文件一致的候选文件，逐个
///   计算内容指纹，只把指纹与 clip 记录的源文件指纹完全一致的候选展示出来
///   （可能较慢，适合文件被改名但内容未变的情形）。
///
/// 文件夹遍历和指纹计算放在 blocking task 中执行，避免阻塞前端 IPC 线程。
pub(super) async fn search_source_file_replacements(
    state: State<'_, AppState>,
    folder_path: String,
    clip_ids: Vec<String>,
    search_mode: crate::models::SearchSourceFileMode,
) -> Result<crate::models::SearchSourceFileMatchesPayload, String> {
    let mut targets_by_name: HashMap<std::ffi::OsString, Vec<(String, Option<u64>)>>;
    let mut targets_by_extension: HashMap<String, Vec<(String, Option<u64>)>>;
    {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        targets_by_name = HashMap::new();
        targets_by_extension = HashMap::new();
        for clip_id in clip_ids {
            let Some(clip) = tl.clips.iter().find(|clip| clip.id == clip_id) else {
                continue;
            };
            let Some(source_path) = clip
                .source_path
                .as_deref()
                .map(str::trim)
                .filter(|path| !path.is_empty())
            else {
                continue;
            };
            let source_path = Path::new(source_path);
            if let Some(file_name) = source_path.file_name() {
                targets_by_name
                    .entry(file_name.to_os_string())
                    .or_default()
                    .push((clip_id.clone(), clip.source_file_fingerprint));
            }
            if let Some(ext) = path_extension_key(source_path) {
                targets_by_extension
                    .entry(ext)
                    .or_default()
                    .push((clip_id.clone(), clip.source_file_fingerprint));
            }
        }
    }

    let folder_path = folder_path.trim().to_string();
    tauri::async_runtime::spawn_blocking(move || {
        let mut matches: HashMap<String, Vec<crate::models::SourceFileMatchCandidatePayload>> =
            targets_by_name
                .values()
                .flatten()
                .chain(targets_by_extension.values().flatten())
                .map(|(clip_id, _)| (clip_id.clone(), Vec::new()))
                .collect();
        let root = Path::new(&folder_path);
        if root.is_dir() {
            collect_source_file_match_candidates(
                root,
                &targets_by_name,
                &targets_by_extension,
                search_mode,
                &mut matches,
            );
        }
        for candidates in matches.values_mut() {
            candidates.sort_by(|left, right| {
                right
                    .exact_hash
                    .cmp(&left.exact_hash)
                    .then_with(|| left.path.cmp(&right.path))
            });
        }
        crate::models::SearchSourceFileMatchesPayload { matches }
    })
    .await
    .map_err(|error| format!("Failed to join source match search task: {error}"))
}

pub(super) fn split_clip(
    state: State<'_, AppState>,
    clip_id: String,
    split_sec: f64,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    let root_track_id = tl
        .clips
        .iter()
        .find(|c| c.id == clip_id)
        .map(|c| c.track_id.clone())
        .and_then(|tid| tl.resolve_root_track_id(&tid));
    let options = split_transition_options(&state);
    let _ = tl.split_clip_with_transition(&clip_id, split_sec, &options);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    drop(tl);
    if let Some(root_id) = root_track_id {
        crate::pitch_analysis::maybe_schedule_pitch_orig(&state, &root_id);
    }
    payload
}

pub(super) fn split_clips_at(
    state: State<'_, AppState>,
    clip_ids: Vec<String>,
    split_sec: f64,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    let root_ids: Vec<String> = clip_ids
        .iter()
        .filter_map(|cid| tl.clips.iter().find(|c| c.id == *cid))
        .map(|c| c.track_id.clone())
        .filter_map(|tid| tl.resolve_root_track_id(&tid))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let options = split_transition_options(&state);
    tl.split_clips_at_with_transition(&clip_ids, split_sec, &options);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    drop(tl);
    for root_id in root_ids {
        crate::pitch_analysis::maybe_schedule_pitch_orig(&state, &root_id);
    }
    payload
}

/// 读取波纹编辑（自动跟进）模式与“参数线是否随剪辑一起平移”的设置。
///
/// 返回 `(模式, 是否平移参数线)`。参数线跟随开关读取全局“锁定参数线”
/// 设置，与前端拖拽移动时传入的 `moveLinkedParams` 语义保持一致。
fn ripple_settings(
    state: &State<'_, AppState>,
) -> (crate::state::RippleMode, bool) {
    let settings = if let Some(dir) = state.config_dir.get() {
        let mut settings = crate::config::load_ui_settings(dir);
        settings.normalize_ripple_mode();
        settings
    } else {
        crate::config::UiSettings::default()
    };
    (
        crate::state::RippleMode::from_str(&settings.ripple_mode),
        settings.lock_param_lines,
    )
}

fn split_transition_options(state: &State<'_, AppState>) -> SplitTransitionOptions {
    let settings = if let Some(dir) = state.config_dir.get() {
        let mut settings = crate::config::load_ui_settings(dir);
        settings.normalize_split_transition();
        settings
    } else {
        crate::config::UiSettings::default()
    };

    let mode = if settings.split_transition_mode == "overlap" {
        SplitTransitionMode::ExtendOverlap
    } else {
        SplitTransitionMode::FadeOnly
    };

    SplitTransitionOptions {
        enabled: settings.split_transition_enabled,
        mode,
        duration_unit: if settings.split_transition_duration_unit == "percent" {
            SplitTransitionDurationUnit::Percent
        } else {
            SplitTransitionDurationUnit::Seconds
        },
        duration_sec: settings.split_transition_duration_sec,
        duration_percent: settings.split_transition_duration_percent,
        curve: Some(settings.split_transition_curve),
        overlap_fades: settings.auto_crossfade
            || settings.split_transition_overlap_crossfade == "always",
    }
}

pub(super) fn glue_clips(
    state: State<'_, AppState>,
    clip_ids: Vec<String>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    // Collect root track IDs before gluing
    let root_ids: Vec<String> = clip_ids
        .iter()
        .filter_map(|clip_id| {
            tl.clips
                .iter()
                .find(|c| c.id == *clip_id)
                .map(|c| c.track_id.clone())
                .and_then(|tid| tl.resolve_root_track_id(&tid))
        })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    tl.glue_clips(&clip_ids);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    drop(tl);
    for root_id in root_ids {
        crate::pitch_analysis::maybe_schedule_pitch_orig(&state, &root_id);
    }
    payload
}

pub(super) fn convert_clips_to_pitch_reference(
    state: State<'_, AppState>,
    clip_ids: Vec<String>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    // 收集 root track IDs 用于后续 pitch 分析调度
    let root_ids: Vec<String> = clip_ids
        .iter()
        .filter_map(|clip_id| {
            tl.clips
                .iter()
                .find(|c| c.id == *clip_id)
                .map(|c| c.track_id.clone())
                .and_then(|tid| tl.resolve_root_track_id(&tid))
        })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    tl.convert_clips_to_pitch_reference(&clip_ids);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    drop(tl);
    for root_id in root_ids {
        crate::pitch_analysis::maybe_schedule_pitch_orig(&state, &root_id);
    }
    payload
}

pub(super) fn update_pitch_reference(
    state: State<'_, AppState>,
    clip_ids: Vec<String>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    // 收集 root track IDs 用于后续 pitch 分析调度
    let root_ids: Vec<String> = clip_ids
        .iter()
        .filter_map(|clip_id| {
            tl.clips
                .iter()
                .find(|c| c.id == *clip_id)
                .map(|c| c.track_id.clone())
                .and_then(|tid| tl.resolve_root_track_id(&tid))
        })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    tl.update_pitch_reference_from_track_params(&clip_ids);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    drop(tl);
    for root_id in root_ids {
        crate::pitch_analysis::maybe_schedule_pitch_orig(&state, &root_id);
    }
    payload
}

pub(super) fn select_clip(
    state: State<'_, AppState>,
    clip_id: Option<String>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    tl.select_clip(clip_id);
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(super) fn get_track_summary(
    state: State<'_, AppState>,
    track_id: Option<String>,
) -> serde_json::Value {
    // Minimal placeholder summary; waveform is empty until audio pipeline is migrated.
    let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    let tid = track_id
        .or_else(|| tl.selected_track_id.clone())
        .or_else(|| tl.tracks.first().map(|t| t.id.clone()))
        .unwrap_or_default();

    let clip_count = tl.clips.iter().filter(|c| c.track_id == tid).count();

    serde_json::json!({
        "ok": true,
        "track_id": tid,
        "clip_count": clip_count,
        "waveform_preview": [],
        "pitch_range": {"min": -24, "max": 24}
    })
}

/// 设置工程 Tempo Map（None = 清除）。
///
/// 前端是 Tempo Map 的唯一编辑入口；本命令：
/// - 校验并规范化变化点（排序、钳制、确保 0 位置点）；
/// - 幂等：载荷规范化后与当前 Tempo Map 完全一致时不产生撤销快照、
///   不更新引擎、不失效缓存、不触发后台预渲染；
/// - 同步工程基准 BPM / 每小节拍数（与 0 位置点一致）；
/// - 实际生效音阶发生变化时失效渲染缓存并触发后台预渲染
///   （子轨道“度数差”等依赖音阶的渲染需要重建）；Tempo / 拍号变化
///   或创建/清除仅含工程基准初始点的 Tempo Map 不会触发重渲染。
pub(super) fn set_timeline_tempo_map(
    state: State<'_, AppState>,
    tempo_map: Option<Vec<crate::models::TempoPointPayload>>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    let had_map = tl.tempo_map.is_some();

    // 比较“实际生效音阶”签名（仅音阶变化才需要失效渲染缓存；
    // 创建/清除仅含工程基准初始点的 Tempo Map 不会触发重新渲染）。
    let scale_signature_before = tl.render_scale_signature();

    // 先在候选副本上应用载荷与规范化，便于与当前状态做幂等比较
    // （此时不产生撤销快照、不修改真实状态）。
    let mut incoming = tl.clone();
    incoming.tempo_map = tempo_map.map(|points| {
        points
            .into_iter()
            .map(|p| crate::state::TempoPointData {
                id: p.id,
                position_sec: p.position_sec,
                bpm: p.bpm,
                numerator: p.numerator,
                denominator: p.denominator,
                scale: p.scale.map(|s| crate::state::TempoScaleData {
                    key: s.key,
                    name: s.name,
                    notes: s.notes,
                }),
            })
            .collect()
    });
    incoming.normalize_tempo_map();

    // 首次创建 Tempo Map：初始点即工程基准记录，音阶为空时物化为工程音阶。
    if !had_map {
        if let Some(points) = incoming.tempo_map.as_mut() {
            if let Some(first) = points.first_mut() {
                if first.scale.is_none() {
                    let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
                    first.scale = Some(crate::state::tempo_scale_data_from_project(&p));
                }
            }
        }
    }

    // ★ 幂等提交：内容与当前完全一致时不产生撤销快照。
    // 典型场景：内联编辑/对话框确认时内容未修改 —— 若每次都无条件
    // checkpoint，一次编辑会产生两个撤销步，用户需要按两次撤销。
    if incoming.tempo_map == tl.tempo_map {
        let mut payload = tl.to_payload();
        payload.project = Some(state.project_meta_payload());
        return payload;
    }

    state.checkpoint_timeline(&tl);
    *tl = incoming;

    // 同步工程基准 BPM / 拍号 / 音阶（与 0 位置点一致，初始点即工程基准记录）。
    // 音阶同步回写会更新 tl.project_scale_notes，因此引擎快照必须在其后获取。
    {
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        state.sync_project_record_from_tempo_map(&mut tl, &mut p);
    }

    let scale_signature_after = tl.render_scale_signature();

    state.audio_engine.update_timeline(tl.clone());

    let scale_changed = scale_signature_before != scale_signature_after;
    if scale_changed {
        for clip in &tl.clips {
            crate::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
        }
    }

    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());

    // ★ 触发后台预渲染前必须先释放时间线锁：
    // `request_background_render` → `start_background_render` 内部会再次锁定
    // `state.timeline`（克隆时间线以收集待渲染 clip）。std Mutex 不可重入，
    // 若此处仍持有 `tl`，命令线程会在自己的锁上自我死锁，整个应用
    // “未响应”（后台预渲染开启时必现）。
    drop(tl);

    if scale_changed {
        if let Some(handle) = state.app_handle.get() {
            crate::commands::playback::request_background_render(handle);
        }
    }

    payload
}
