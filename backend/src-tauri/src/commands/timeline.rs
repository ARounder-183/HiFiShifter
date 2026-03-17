use crate::state::AppState;
use base64::Engine;
use std::fs;
use std::path::Path;
use tauri::State;
use uuid::Uuid;

use super::common::ensure_temp_dir;

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
) -> crate::models::TimelineStatePayload {
    if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
        eprintln!(
            "import_audio_item(audio_path={}, track_id={:?}, start_sec={:?})",
            audio_path, track_id, start_sec
        );
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

    tl.import_audio_item(&audio_path, resolved_track_id, start_sec);
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
    tl.remove_track(&track_id);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(super) fn duplicate_track(
    state: State<'_, AppState>,
    track_id: String,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    tl.duplicate_track(&track_id);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
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

pub(super) fn remove_clip(
    state: State<'_, AppState>,
    clip_id: String,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    tl.remove_clip(&clip_id);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(super) fn move_clip(
    state: State<'_, AppState>,
    clip_id: String,
    start_sec: f64,
    track_id: Option<String>,
    move_linked_params: Option<bool>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    tl.move_clip(
        &clip_id,
        start_sec,
        track_id,
        move_linked_params.unwrap_or(false),
    );
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(super) fn move_clips(
    state: State<'_, AppState>,
    moves: Vec<crate::state::MoveClipPayload>,
    move_linked_params: Option<bool>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    tl.move_clips(&moves, move_linked_params.unwrap_or(false));
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
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
    fade_in_sec: Option<f64>,
    fade_out_sec: Option<f64>,
    fade_in_curve: Option<String>,
    fade_out_curve: Option<String>,
    color: Option<String>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
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
            fade_in_sec,
            fade_out_sec,
            fade_in_curve,
            fade_out_curve,
            color,
        },
    );
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(super) fn replace_clip_source(
    state: State<'_, AppState>,
    clip_ids: Vec<String>,
    new_source_path: String,
    replace_same_source: Option<bool>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    tl.replace_clip_sources(
        &clip_ids,
        &new_source_path,
        replace_same_source.unwrap_or(false),
    );
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(super) fn split_clip(
    state: State<'_, AppState>,
    clip_id: String,
    split_sec: f64,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    tl.split_clip(&clip_id, split_sec);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(super) fn glue_clips(
    state: State<'_, AppState>,
    clip_ids: Vec<String>,
) -> crate::models::TimelineStatePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl);
    tl.glue_clips(&clip_ids);
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
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
