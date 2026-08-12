use crate::config::UiSettings;
use crate::state::AppState;
use tauri::State;

pub(super) fn get_ui_settings(state: State<'_, AppState>) -> UiSettings {
    let mut settings = if let Some(dir) = state.config_dir.get() {
        crate::config::load_ui_settings(dir)
    } else {
        UiSettings::default()
    };
    settings.normalize_split_transition();
    settings.normalize_time_display();
    crate::time_stretch::update_global_stretch_defaults(
        settings.default_stretch_algorithm,
        settings.default_hifigan_mel_stretch,
    );
    // Apply EP settings on load — all three ONNX models
    crate::nsf_hifigan_onnx::update_ort_ep(&settings.ort_ep, settings.ort_device_id);
    crate::hnsep_onnx::update_ort_ep(&settings.ort_ep, settings.ort_device_id);
    crate::fcpe_onnx::update_ort_ep(&settings.ort_ep, settings.ort_device_id);
    // Sync background render setting
    crate::commands::playback::AUTO_BG_RENDER_ENABLED.store(
        settings.auto_background_render,
        std::sync::atomic::Ordering::Relaxed,
    );
    settings
}

pub(super) fn save_ui_settings(
    state: State<'_, AppState>,
    mut settings: UiSettings,
) -> serde_json::Value {
    settings.normalize_split_transition();
    settings.normalize_time_display();
    let prev_settings = if let Some(dir) = state.config_dir.get() {
        crate::config::load_ui_settings(dir)
    } else {
        UiSettings::default()
    };
    let prev_ep = prev_settings.ort_ep.clone();

    if let Some(dir) = state.config_dir.get() {
        crate::config::save_ui_settings(dir, &settings);
    }
    crate::time_stretch::update_global_stretch_defaults(
        settings.default_stretch_algorithm,
        settings.default_hifigan_mel_stretch,
    );
    crate::commands::playback::AUTO_BG_RENDER_ENABLED.store(
        settings.auto_background_render,
        std::sync::atomic::Ordering::Relaxed,
    );

    let ep_changed = prev_ep != settings.ort_ep;

    // Changing the global stretch defaults only affects the current project when
    // the project inherits the corresponding setting. Compute the effective value
    // so unrelated global edits (e.g. theme-only saves) do not invalidate renders.
    let effective_stretch_changed = {
        let project = state.project.lock().unwrap_or_else(|e| e.into_inner());
        let algorithm_before = project
            .stretch_algorithm_override
            .unwrap_or(prev_settings.default_stretch_algorithm);
        let algorithm_after = project
            .stretch_algorithm_override
            .unwrap_or(settings.default_stretch_algorithm);
        let mel_before = project
            .hifigan_mel_stretch_override
            .unwrap_or(prev_settings.default_hifigan_mel_stretch);
        let mel_after = project
            .hifigan_mel_stretch_override
            .unwrap_or(settings.default_hifigan_mel_stretch);
        algorithm_before != algorithm_after || mel_before != mel_after
    };

    if ep_changed {
        crate::nsf_hifigan_onnx::update_ort_ep(&settings.ort_ep, settings.ort_device_id);
        crate::hnsep_onnx::update_ort_ep(&settings.ort_ep, settings.ort_device_id);
        crate::fcpe_onnx::update_ort_ep(&settings.ort_ep, settings.ort_device_id);
    }

    if ep_changed || effective_stretch_changed {
        let timeline = state
            .timeline
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        for clip in &timeline.clips {
            crate::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
        }
        state.audio_engine.update_timeline(timeline);
        if let Some(handle) = state.app_handle.get() {
            crate::commands::playback::request_background_render(handle);
        }
    }

    serde_json::json!({ "ok": true })
}
