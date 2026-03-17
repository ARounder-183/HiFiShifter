use crate::project::CustomScale;
use std::fs;
use std::path::Path;

/// UI 设置（持久化到 app_config.json）
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct UiSettings {
    #[serde(default = "default_true")]
    pub auto_crossfade: bool,
    #[serde(default = "default_true")]
    pub grid_snap: bool,
    #[serde(default = "default_grid_size")]
    pub grid_size: String,
    #[serde(default)]
    pub pitch_snap: bool,
    #[serde(default = "default_pitch_snap_unit")]
    pub pitch_snap_unit: String,
    #[serde(default)]
    pub pitch_snap_tolerance_cents: u32,
    #[serde(default)]
    pub playhead_zoom: bool,
    #[serde(default)]
    pub auto_scroll: bool,
    #[serde(default)]
    pub show_clipboard_preview: bool,
    #[serde(default = "default_true")]
    pub lock_param_lines: bool,
    #[serde(default = "default_drag_direction")]
    pub drag_direction: String,
    #[serde(default = "default_drag_direction")]
    pub select_drag_direction: String,
    #[serde(default = "default_draw_drag_direction")]
    pub draw_drag_direction: String,
    #[serde(default = "default_draw_drag_direction")]
    pub line_vibrato_drag_direction: String,
    #[serde(default, alias = "edgeSmoothnessPercent")]
    pub smoothness_percent: u32,
    #[serde(default = "default_scale_highlight_mode")]
    pub scale_highlight_mode: String,
    #[serde(default)]
    pub custom_scale_presets: Vec<CustomScale>,
}

fn default_true() -> bool {
    true
}
fn default_pitch_snap_unit() -> String {
    "semitone".to_string()
}
fn default_grid_size() -> String {
    "1/4".to_string()
}
fn default_drag_direction() -> String {
    "y-only".to_string()
}
fn default_draw_drag_direction() -> String {
    "free".to_string()
}

fn default_scale_highlight_mode() -> String {
    "off".to_string()
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            auto_crossfade: true,
            grid_snap: true,
            grid_size: default_grid_size(),
            pitch_snap: false,
            pitch_snap_unit: default_pitch_snap_unit(),
            pitch_snap_tolerance_cents: 0,
            playhead_zoom: false,
            auto_scroll: false,
            show_clipboard_preview: true,
            lock_param_lines: true,
            drag_direction: default_drag_direction(),
            select_drag_direction: default_drag_direction(),
            draw_drag_direction: default_draw_drag_direction(),
            line_vibrato_drag_direction: default_draw_drag_direction(),
            smoothness_percent: 0,
            scale_highlight_mode: default_scale_highlight_mode(),
            custom_scale_presets: Vec::new(),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct AppConfig {
    #[serde(default)]
    recent: Vec<String>,
    #[serde(default)]
    ui: UiSettings,
}

fn load_config(config_dir: &Path) -> AppConfig {
    let path = config_dir.join("app_config.json");
    let Ok(data) = fs::read_to_string(&path) else {
        return AppConfig::default();
    };
    serde_json::from_str::<AppConfig>(&data).unwrap_or_default()
}

fn save_config(config_dir: &Path, cfg: &AppConfig) {
    let path = config_dir.join("app_config.json");
    if let Ok(data) = serde_json::to_string_pretty(cfg) {
        let _ = fs::write(&path, data);
    }
}

/// 从 config dir 读取最近工程列表；读取失败时返回空列表。
pub fn load_recent(config_dir: &Path) -> Vec<String> {
    load_config(config_dir).recent
}

/// 将最近工程列表写入 config dir；写入失败时静默忽略。
/// 保留现有配置中的其他字段（如 UI 设置）。
pub fn save_recent(config_dir: &Path, recent: &[String]) {
    let mut cfg = load_config(config_dir);
    cfg.recent = recent.to_vec();
    save_config(config_dir, &cfg);
}

/// 从 config dir 读取 UI 设置。
pub fn load_ui_settings(config_dir: &Path) -> UiSettings {
    load_config(config_dir).ui
}

/// 将 UI 设置写入 config dir；保留现有配置中的其他字段。
pub fn save_ui_settings(config_dir: &Path, ui: &UiSettings) {
    let mut cfg = load_config(config_dir);
    cfg.ui = ui.clone();
    save_config(config_dir, &cfg);
}
