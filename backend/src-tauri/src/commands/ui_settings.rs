use crate::config::UiSettings;
use crate::state::AppState;
use tauri::State;

pub(super) fn get_ui_settings(state: State<'_, AppState>) -> UiSettings {
    if let Some(dir) = state.config_dir.get() {
        crate::config::load_ui_settings(dir)
    } else {
        UiSettings::default()
    }
}

pub(super) fn save_ui_settings(
    state: State<'_, AppState>,
    settings: UiSettings,
) -> serde_json::Value {
    if let Some(dir) = state.config_dir.get() {
        crate::config::save_ui_settings(dir, &settings);
    }
    serde_json::json!({ "ok": true })
}
