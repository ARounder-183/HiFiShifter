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

/// Lightweight timeline state for regular frontend polls.
/// Skips waveform_preview, pitch_range, and midi_note_data to reduce clone+serialize cost.
pub(super) fn get_timeline_state_lite(
    state: State<'_, AppState>,
) -> crate::models::TimelineStatePayload {
    // to_payload_lite 接受引用：直接在锁内构建 payload，
    // 省掉整棵 TimelineState（含全部参数曲线）的深克隆。
    let mut payload = {
        let tl = state
            .timeline
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        tl.to_payload_lite()
    };
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
            // 与 Tempo Map 规范化/前端 clampBpm 一致：钳制到 10-960，
            // 否则这里可直接把 Tempo Map 初始点 BPM 写出合法范围。
            let clamped = v.clamp(10.0, 960.0);
            tl.bpm = clamped;
            // Tempo Map 存在时，工程 BPM 与 0 位置点保持一致。
            if let Some(points) = tl.tempo_map.as_mut() {
                if let Some(first) = points.first_mut() {
                    first.bpm = clamped;
                }
            }
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
