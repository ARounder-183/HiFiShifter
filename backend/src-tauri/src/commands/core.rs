use crate::state::AppState;
use tauri::State;

pub(super) fn ping() -> serde_json::Value {
    serde_json::json!({ "ok": true, "message": "pong" })
}

pub(super) fn get_runtime_info(state: State<'_, AppState>) -> crate::models::RuntimeInfoPayload {
    state.runtime_info()
}

pub(super) fn consume_startup_project_path(state: State<'_, AppState>) -> serde_json::Value {
    let path = state.take_pending_startup_project_path();
    serde_json::json!({ "ok": true, "path": path })
}

pub(super) fn set_ui_locale(state: State<'_, AppState>, locale: String) -> serde_json::Value {
    let locale = locale.trim();
    let lower = locale.to_lowercase();
    let normalized = if locale.eq_ignore_ascii_case("zh-CN")
        || locale.eq_ignore_ascii_case("zh_CN")
        || lower.starts_with("zh")
    {
        "zh-CN".to_string()
    } else if locale.eq_ignore_ascii_case("ja-JP")
        || locale.eq_ignore_ascii_case("ja_JP")
        || lower.starts_with("ja")
    {
        "ja-JP".to_string()
    } else if locale.eq_ignore_ascii_case("ko-KR")
        || locale.eq_ignore_ascii_case("ko_KR")
        || lower.starts_with("ko")
    {
        "ko-KR".to_string()
    } else {
        // Default to en-US for unknown values.
        "en-US".to_string()
    };

    {
        let mut guard = state.ui_locale.write().unwrap_or_else(|e| e.into_inner());
        *guard = normalized.clone();
    }

    serde_json::json!({"ok": true, "locale": normalized})
}

pub(super) fn get_timeline_state(
    state: State<'_, AppState>,
) -> crate::models::TimelineStatePayload {
    let tl = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(crate) fn get_timeline_state_from_ref(state: &AppState) -> crate::models::TimelineStatePayload {
    let tl = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(super) fn set_transport(
    state: State<'_, AppState>,
    playhead_sec: Option<f64>,
    bpm: Option<f64>,
) -> serde_json::Value {
    if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
        eprintln!(
            "set_transport(playhead_sec={:?}, bpm={:?})",
            playhead_sec, bpm
        );
    }
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    let prev_bpm = tl.bpm;
    if let Some(v) = playhead_sec {
        tl.playhead_sec = v.max(0.0);
    }
    if let Some(v) = bpm {
        if v.is_finite() && v > 0.0 {
            // BPM is project-affecting: checkpoint for undo.
            state.checkpoint_timeline(&tl);
            tl.bpm = v;
        }
    }

    // Keep realtime engine transport aligned.
    state.audio_engine.seek_sec(tl.playhead_sec);
    if (tl.bpm - prev_bpm).abs() > 1e-9 {
        state.audio_engine.update_timeline(tl.clone());
    }

    serde_json::json!({"ok": true, "playhead_sec": tl.playhead_sec, "bpm": tl.bpm })
}

// ===================== undo / redo =====================

pub(super) fn undo_timeline(state: State<'_, AppState>) -> crate::models::TimelineStatePayload {
    state.undo_timeline()
}

pub(super) fn redo_timeline(state: State<'_, AppState>) -> crate::models::TimelineStatePayload {
    state.redo_timeline()
}
