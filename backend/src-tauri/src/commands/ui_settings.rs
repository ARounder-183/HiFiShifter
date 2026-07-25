use crate::config::UiSettings;
use crate::state::AppState;
use tauri::State;

pub(super) fn get_ui_settings(state: State<'_, AppState>) -> UiSettings {
    let settings = if let Some(dir) = state.config_dir.get() {
        crate::config::load_ui_settings(dir)
    } else {
        UiSettings::default()
    };
    crate::time_stretch::update_global_stretch_defaults(
        settings.default_stretch_algorithm,
        settings.default_hifigan_mel_stretch,
    );
    // Apply EP settings on load — all three ONNX models
    crate::nsf_hifigan_onnx::update_ort_ep(&settings.ort_ep, settings.ort_device_id);
    crate::hnsep_onnx::update_ort_ep(&settings.ort_ep, settings.ort_device_id);
    crate::fcpe_onnx::update_ort_ep(&settings.ort_ep, settings.ort_device_id);
    // Sync background render setting
    crate::commands::playback::AUTO_BG_RENDER_ENABLED
        .store(settings.auto_background_render, std::sync::atomic::Ordering::Relaxed);
    settings
}

pub(super) fn save_ui_settings(
    state: State<'_, AppState>,
    settings: UiSettings,
) -> serde_json::Value {
    // Check if EP choice has changed
    let prev_ep = if let Some(dir) = state.config_dir.get() {
        crate::config::load_ui_settings(dir).ort_ep
    } else {
        "auto".to_string()
    };

    if let Some(dir) = state.config_dir.get() {
        crate::config::save_ui_settings(dir, &settings);
    }
    crate::time_stretch::update_global_stretch_defaults(
        settings.default_stretch_algorithm,
        settings.default_hifigan_mel_stretch,
    );
    crate::commands::playback::AUTO_BG_RENDER_ENABLED
        .store(settings.auto_background_render, std::sync::atomic::Ordering::Relaxed);

    let ep_changed = prev_ep != settings.ort_ep;

    if ep_changed {
        crate::nsf_hifigan_onnx::update_ort_ep(&settings.ort_ep, settings.ort_device_id);
        crate::hnsep_onnx::update_ort_ep(&settings.ort_ep, settings.ort_device_id);
        crate::fcpe_onnx::update_ort_ep(&settings.ort_ep, settings.ort_device_id);

        let timeline = state.timeline.lock().unwrap_or_else(|e| e.into_inner()).clone();
        for clip in &timeline.clips {
            crate::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
        }
        state.audio_engine.update_timeline(timeline);
    }
    
    serde_json::json!({ "ok": true })
}
