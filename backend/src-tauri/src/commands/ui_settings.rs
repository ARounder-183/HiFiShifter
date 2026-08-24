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
    // Sync "loop for new clips" default (used by importers / legacy project migration)
    crate::config::set_loop_new_clips_default(settings.loop_new_clips);
    settings
}

pub(super) fn save_ui_settings(
    state: State<'_, AppState>,
    settings_value: serde_json::Value,
) -> serde_json::Value {
    // 前端可能只发送变更字段（部分保存，如单个 MIDI 导入选项）。
    // UiSettings 的所有字段都带 serde default，直接用部分对象反序列化
    // 会把未发送字段重置为默认值，覆盖磁盘上的其他设置 —— 因此以
    // 现有设置为基础做 JSON 级合并，再反序列化应用。
    let prev_settings = if let Some(dir) = state.config_dir.get() {
        crate::config::load_ui_settings(dir)
    } else {
        UiSettings::default()
    };
    let merged_value = match serde_json::to_value(&prev_settings) {
        Ok(mut base) => {
            if let (serde_json::Value::Object(base_obj), serde_json::Value::Object(patch_obj)) =
                (&mut base, &settings_value)
            {
                for (key, value) in patch_obj {
                    if key == "timelineSnap" {
                        // 嵌套设置做深度合并，避免部分保存时清空其它吸附选项。
                        match base_obj.get_mut("timelineSnap") {
                            Some(serde_json::Value::Object(base_nested)) => {
                                if let serde_json::Value::Object(patch_nested) = value {
                                    for (nested_key, nested_value) in patch_nested {
                                        base_nested.insert(nested_key.clone(), nested_value.clone());
                                    }
                                }
                            }
                            _ => {
                                base_obj.insert(key.clone(), value.clone());
                            }
                        }
                    } else {
                        base_obj.insert(key.clone(), value.clone());
                    }
                }
            }
            base
        }
        Err(_) => settings_value,
    };
    let mut settings: UiSettings =
        serde_json::from_value(merged_value).unwrap_or_else(|_| prev_settings.clone());

    settings.normalize_split_transition();
    settings.normalize_time_display();
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
    crate::config::set_loop_new_clips_default(settings.loop_new_clips);

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
