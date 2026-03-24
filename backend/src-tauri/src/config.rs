use crate::project::CustomScale;
use std::fs;
use std::path::Path;

/// UI 设置（持久化到 app_config.json）
///
/// 该文件负责管理应用的可序列化配置项，包括 UI 相关的偏好
/// 以及窗口状态。窗口状态用于在程序重启后恢复上次的窗口尺寸、位置和最大化/全屏状态。
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
    #[serde(default = "default_true")]
    pub param_editor_seek_playhead: bool,
    #[serde(default = "default_true")]
    pub show_clipboard_preview: bool,
    #[serde(default = "default_true")]
    pub show_param_value_popup: bool,
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
            param_editor_seek_playhead: true,
            show_clipboard_preview: true,
            show_param_value_popup: true,
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

/// 持久化配置根结构。
#[derive(serde::Serialize, serde::Deserialize, Default, Clone, Debug)]
struct AppConfig {
    #[serde(default)]
    recent: Vec<String>,
    #[serde(default)]
    ui: UiSettings,
    /// 持久化的窗口状态（可选）。
    #[serde(default)]
    window: WindowState,
}

/// 窗口状态（持久化）
#[derive(serde::Serialize, serde::Deserialize, Default, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct WindowState {
    /// 窗口左上角 x（屏幕坐标，逻辑像素）
    pub x: Option<i32>,
    /// 窗口左上角 y（屏幕坐标，逻辑像素）
    pub y: Option<i32>,
    /// 窗口宽度（逻辑像素）
    pub width: Option<f64>,
    /// 窗口高度（逻辑像素）
    pub height: Option<f64>,
    /// 是否最大化
    pub maximized: Option<bool>,
    /// 是否全屏
    pub fullscreen: Option<bool>,
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

/// 读取持久化的窗口状态，如果不存在则返回默认值
pub fn load_window_state(config_dir: &Path) -> WindowState {
    load_config(config_dir).window
}

/// 将窗口状态写回配置文件（保留其他字段）
pub fn save_window_state(config_dir: &Path, ws: &WindowState) {
    let mut cfg = load_config(config_dir);
    cfg.window = ws.clone();
    save_config(config_dir, &cfg);
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

// ─── 外部 Resampler 注册表持久化 ──────────────────────────────────────────────

const RESAMPLER_REGISTRY_FILE: &str = "resampler_registry.json";

/// 从 config dir 加载已注册的外部 Resampler 列表；读取失败时返回空注册表。
pub fn load_resampler_registry(config_dir: &Path) -> crate::state::ResamplerRegistry {
    let path = config_dir.join(RESAMPLER_REGISTRY_FILE);
    let Ok(data) = fs::read_to_string(&path) else {
        return crate::state::ResamplerRegistry::default();
    };
    let Ok(mut registry) = serde_json::from_str::<crate::state::ResamplerRegistry>(&data) else {
        eprintln!("[config] 解析 resampler_registry.json 失败，使用空注册表");
        return crate::state::ResamplerRegistry::default();
    };
    // 刷新可用性：检查 exe 文件是否仍然存在
    registry.refresh_availability();
    registry
}

/// 将外部 Resampler 注册表写入 config dir；写入失败时静默忽略。
pub fn save_resampler_registry(config_dir: &Path, registry: &crate::state::ResamplerRegistry) {
    let path = config_dir.join(RESAMPLER_REGISTRY_FILE);
    if let Ok(data) = serde_json::to_string_pretty(registry) {
        let _ = fs::write(&path, data);
    }
}
