use crate::config::RecordingSettings;
use crate::models::TimelineStatePayload;
use crate::recording::{self, RecordingFinishedInfo};
use crate::state::AppState;
use tauri::State;

pub(super) fn get_recording_settings(state: State<'_, AppState>) -> RecordingSettings {
    recording::load_settings(state.inner())
}

pub(super) fn save_recording_settings(
    state: State<'_, AppState>,
    settings: RecordingSettings,
) -> serde_json::Value {
    let normalized = recording::save_settings(state.inner(), &settings);
    serde_json::json!({ "ok": true, "settings": normalized })
}

pub(super) fn get_recording_devices() -> serde_json::Value {
    serde_json::json!({
        "ok": true,
        "devices": recording::enumerate_devices(),
    })
}

pub(super) fn get_recording_apps() -> serde_json::Value {
    serde_json::json!({
        "ok": true,
        "apps": recording::enumerate_applications(),
    })
}

pub(super) fn start_recording(state: State<'_, AppState>, start_sec: f64) -> serde_json::Value {
    match recording::start(state.inner(), start_sec.max(0.0)) {
        Ok(info) => serde_json::json!({
            "ok": true,
            "startSec": info.start_sec,
            "outputPath": info.output_path,
        }),
        Err(error) => serde_json::json!({ "ok": false, "error": error }),
    }
}

pub(super) fn get_recording_state(state: State<'_, AppState>) -> recording::RecordingStatePayload {
    recording::current_state(state.inner())
}

pub(super) fn stop_recording(state: State<'_, AppState>) -> serde_json::Value {
    // 停止录音后立即停止时间轴播放（录音导入前保持安静状态）。
    state.audio_engine.stop();

    let finished = match recording::stop(state.inner()) {
        Ok(finished) => finished,
        Err(error) => return serde_json::json!({ "ok": false, "error": error }),
    };

    match import_finished_recording(state.inner(), &finished) {
        Ok(timeline) => serde_json::json!({
            "ok": true,
            "timeline": timeline,
            "recording": {
                "startSec": finished.start_sec,
                "durationSec": finished.duration_sec,
                "sampleRate": finished.sample_rate,
                "channels": finished.channels,
                "peak": finished.peak,
                "outputPath": finished.output_path,
            },
        }),
        Err(error) => serde_json::json!({ "ok": false, "error": error }),
    }
}

fn import_finished_recording(
    state: &AppState,
    finished: &RecordingFinishedInfo,
) -> Result<TimelineStatePayload, String> {
    let mut timeline = state
        .timeline
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    state.checkpoint_timeline(&timeline);

    let start_sec = finished.start_sec.max(0.0);
    let end_sec = start_sec + finished.duration_sec.max(0.0);
    let selected_track_id = timeline.selected_track_id.clone();
    let recording_track_name = recording_track_name(state);

    // 选中轨在录音范围内完全为空时，直接使用选中轨；否则在选中轨正下方新建轨道。
    let selected_is_empty = selected_track_id.as_ref().is_some_and(|track_id| {
        timeline.tracks.iter().any(|track| &track.id == track_id)
            && !timeline.clips.iter().any(|clip| {
                clip.track_id == *track_id
                    && clip.start_sec < end_sec
                    && clip.start_sec + clip.length_sec > start_sec
            })
    });

    let target_track_id = if selected_is_empty {
        selected_track_id.unwrap_or_else(|| {
            timeline.add_track(Some(recording_track_name.clone()), None, None)
        })
    } else if let Some(selected) = selected_track_id {
        let mut root_tracks: Vec<_> = timeline
            .tracks
            .iter()
            .filter(|track| track.parent_id.is_none())
            .collect();
        root_tracks.sort_by_key(|track| track.order);
        let root_order: Vec<String> = root_tracks.iter().map(|track| track.id.clone()).collect();
        let selected_root = timeline
            .resolve_root_track_id(&selected)
            .unwrap_or_else(|| selected.clone());
        let insert_index = root_order
            .iter()
            .position(|id| *id == selected_root)
            .map(|index| index + 1)
            .unwrap_or(root_order.len());
        timeline.add_track(
            Some(recording_track_name.clone()),
            None,
            Some(insert_index),
        )
    } else {
        timeline.add_track(Some(recording_track_name), None, None)
    };

    timeline.import_audio_item(
        &finished.output_path,
        Some(target_track_id.clone()),
        Some(start_sec),
    );

    let settings = recording::load_settings(state);
    let imported_clip_id = timeline
        .clips
        .iter()
        .filter(|clip| {
            clip.track_id == target_track_id
                && clip.source_path.as_deref() == Some(finished.output_path.as_str())
        })
        .last()
        .map(|clip| clip.id.clone());

    if settings.auto_normalize {
        if let Some(clip) = imported_clip_id
            .as_ref()
            .and_then(|id| timeline.clips.iter_mut().find(|clip| &clip.id == id))
        {
            if finished.peak.is_finite() && finished.peak > 0.0001 {
                clip.gain = (1.0 / finished.peak).clamp(1.0, 4.0);
            }
        }
    }

    // 将轨道选择自动切到新录音所在的轨道，并选中新导入的 clip。
    if let Some(clip_id) = imported_clip_id {
        timeline.select_clip(Some(clip_id));
    } else {
        timeline.select_track(&target_track_id);
    }

    // 录音完成后把播放光标定位到录音末尾：随 payload 返回给前端，
    // 前端再通过 pendingPlayheadRevealSec 机制在超出画面时滚动视图。
    timeline.playhead_sec = end_sec.max(0.0);

    state.audio_engine.update_timeline(timeline.clone());

    let mut payload = timeline.to_payload();
    payload.project = Some(state.project_meta_payload());
    Ok(payload)
}

fn recording_track_name(state: &AppState) -> String {
    let locale = state
        .ui_locale
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_else(|err| err.into_inner().clone());
    match locale.as_str() {
        "zh-CN" => "录音".to_string(),
        "zh-TW" => "錄音".to_string(),
        "ja-JP" => "録音".to_string(),
        "ko-KR" => "녹음".to_string(),
        _ => "Recording".to_string(),
    }
}
