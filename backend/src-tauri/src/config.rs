use crate::project::CustomScale;
use crate::time_stretch::UserStretchAlgorithm;
use std::fs;
use std::path::Path;

// 最小合理窗口尺寸与坐标阈值，用于校验从磁盘读取到的窗口状态，避免异常值导致窗口无法显示。
const MIN_WINDOW_WIDTH: f64 = 200.0;
const MIN_WINDOW_HEIGHT: f64 = 160.0;
// 某些平台/环境会把不可用的位置写成 -32768 之类的哨兵值，认为这是无效坐标。
const INVALID_COORD_MIN: i32 = -32000;
// 也拒绝极端大的坐标值（防止溢出或误写入极端数值）
const MAX_COORD_ABS: i32 = 1_000_000;

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
    #[serde(default)]
    pub quick_search_auto_normalize: bool,
    #[serde(default)]
    pub visible_reference_root_track_ids: Vec<String>,
    #[serde(default)]
    pub default_stretch_algorithm: UserStretchAlgorithm,
    #[serde(default = "default_hifigan_mel_stretch")]
    pub default_hifigan_mel_stretch: bool,
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
    #[serde(default)]
    pub ignore_grouping: bool,
    #[serde(default = "default_midi_import_position")]
    pub midi_import_position: String,
    #[serde(default)]
    pub midi_fill_gaps: bool,
    #[serde(default = "default_true")]
    pub midi_multi_track_merge: bool,
    #[serde(default)]
    pub midi_import_bpm_as_project: bool,
    #[serde(default = "default_midi_note_bpm_mode")]
    pub midi_note_bpm_mode: String,
    #[serde(default)]
    pub midi_specified_bpm: Option<f64>,
    #[serde(default = "default_true")]
    pub midi_close_leading_gap: bool,
    #[serde(default = "default_midi_import_target")]
    pub midi_import_target: String,
    /// ONNX Runtime execution provider preference ("Auto", "CUDA", "CPU").
    /// Persisted so the user's GPU/CPU choice survives restarts.
    #[serde(default = "default_ort_ep")]
    pub ort_ep: String,
    /// Selected CUDA device ID for GPU inference.
    /// Persisted so the user's GPU selection survives restarts.
    /// Set to 0 by default; users with multiple GPUs can change it.
    #[serde(default = "default_cuda_device_id")]
    pub cuda_device_id: i32,
    /// 后台预渲染：启用后，当编辑操作使渲染缓存失效时，
    /// 立即在后台启动预渲染，而不是等到用户按下播放时才渲染。
    /// 用户可在渲染进行中随时开始播放已渲染完成的部分。
    /// Background pre-render: when enabled, immediately start
    /// rendering in the background after editing invalidates the
    /// render cache. Users can play already-rendered content
    /// at any time during rendering.
    #[serde(default)]
    pub auto_background_render: bool,
}

fn default_ort_ep() -> String {
    "Auto".to_string()
}

fn default_cuda_device_id() -> i32 {
    0
}

/// 导出音频设置（持久化到 app_config.json）
///
/// 用于记住导出窗口中不同导出类型的输出目录与文件名设置。
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ExportSettings {
    #[serde(default)]
    pub project_output_dir: Option<String>,
    #[serde(default)]
    pub project_file_name: Option<String>,
    #[serde(default)]
    pub separated_output_dir: Option<String>,
    #[serde(default)]
    pub separated_file_name_pattern: Option<String>,
    #[serde(default = "default_export_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_export_bit_depth")]
    pub bit_depth: u32,
}

/// 自动备份设置（持久化到 app_config.json）
///
/// - `save_on_save_enabled`: 手动保存/另存为时，保存前先轮换目标文件为备份副本。
/// - `timed_backup_enabled`: 是否启用定时备份。
/// - `timed_backup_interval_sec`: 定时备份判定间隔（秒）。
/// - `timed_backup_path_template`: 备份目标路径模板，支持占位符与时间格式。
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AutoBackupSettings {
    #[serde(default = "default_true")]
    pub save_on_save_enabled: bool,
    #[serde(default)]
    pub timed_backup_enabled: bool,
    #[serde(default = "default_timed_backup_interval_sec")]
    pub timed_backup_interval_sec: u32,
    #[serde(default = "default_timed_backup_path_template")]
    pub timed_backup_path_template: String,
}

fn default_timed_backup_interval_sec() -> u32 {
    300
}

fn default_timed_backup_path_template() -> String {
    "<ProjectFolder>/HiFiShifter Backup/<ProjectName>_%Y-%m-%d-%H-%M-%S.hshp".to_string()
}

impl Default for AutoBackupSettings {
    fn default() -> Self {
        Self {
            save_on_save_enabled: true,
            timed_backup_enabled: false,
            timed_backup_interval_sec: default_timed_backup_interval_sec(),
            timed_backup_path_template: default_timed_backup_path_template(),
        }
    }
}

impl AutoBackupSettings {
    pub fn normalized(&self) -> Self {
        let interval = self.timed_backup_interval_sec.clamp(1, 86_400);
        let template = {
            let trimmed = self.timed_backup_path_template.trim();
            if trimmed.is_empty() {
                default_timed_backup_path_template()
            } else {
                trimmed.to_string()
            }
        };

        Self {
            save_on_save_enabled: self.save_on_save_enabled,
            timed_backup_enabled: self.timed_backup_enabled,
            timed_backup_interval_sec: interval,
            timed_backup_path_template: template,
        }
    }
}

fn default_export_sample_rate() -> u32 {
    48_000
}

fn default_export_bit_depth() -> u32 {
    32
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            project_output_dir: None,
            project_file_name: None,
            separated_output_dir: None,
            separated_file_name_pattern: None,
            sample_rate: default_export_sample_rate(),
            bit_depth: default_export_bit_depth(),
        }
    }
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

fn default_hifigan_mel_stretch() -> bool {
    true
}

fn default_scale_highlight_mode() -> String {
    "off".to_string()
}

fn default_midi_import_position() -> String {
    "selection".to_string()
}

fn default_midi_note_bpm_mode() -> String {
    "midi".to_string()
}

fn default_midi_import_target() -> String {
    "pitchParam".to_string()
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
            quick_search_auto_normalize: false,
            visible_reference_root_track_ids: Vec::new(),
            default_stretch_algorithm: UserStretchAlgorithm::default(),
            default_hifigan_mel_stretch: default_hifigan_mel_stretch(),
            drag_direction: default_drag_direction(),
            select_drag_direction: default_drag_direction(),
            draw_drag_direction: default_draw_drag_direction(),
            line_vibrato_drag_direction: default_draw_drag_direction(),
            smoothness_percent: 0,
            scale_highlight_mode: default_scale_highlight_mode(),
            custom_scale_presets: Vec::new(),
            ignore_grouping: false,
            midi_import_position: default_midi_import_position(),
            midi_fill_gaps: false,
            midi_multi_track_merge: true,
            midi_import_bpm_as_project: false,
            midi_note_bpm_mode: default_midi_note_bpm_mode(),
            midi_specified_bpm: None,
            midi_close_leading_gap: true,
            midi_import_target: default_midi_import_target(),
            ort_ep: default_ort_ep(),
            cuda_device_id: default_cuda_device_id(),
            auto_background_render: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::UiSettings;
    use crate::time_stretch::UserStretchAlgorithm;

    #[test]
    fn ui_settings_defaults_to_signalsmith_and_hifigan_mel_stretch_on() {
        let settings = UiSettings::default();
        assert_eq!(
            settings.default_stretch_algorithm,
            UserStretchAlgorithm::Signalsmith
        );
        assert!(settings.default_hifigan_mel_stretch);
    }
}

/// 持久化配置根结构。
#[derive(serde::Serialize, serde::Deserialize, Default, Clone, Debug)]
struct AppConfig {
    #[serde(default)]
    recent: Vec<String>,
    #[serde(default)]
    ui: UiSettings,
    #[serde(default)]
    export: ExportSettings,
    #[serde(default)]
    auto_backup: AutoBackupSettings,
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
fn sanitize_window_state(mut ws: WindowState) -> WindowState {
    // 宽高校验：必须是有限数且不小于最小尺寸，过大的值视为异常
    if let Some(w) = ws.width {
        if !w.is_finite() || w < MIN_WINDOW_WIDTH || w > 100_000.0 {
            ws.width = None;
        }
    }
    if let Some(h) = ws.height {
        if !h.is_finite() || h < MIN_WINDOW_HEIGHT || h > 100_000.0 {
            ws.height = None;
        }
    }

    // 坐标校验：拒绝典型的哨兵值（如 -32768）或极端不合理的坐标
    if let Some(x) = ws.x {
        if x <= INVALID_COORD_MIN || x.abs() > MAX_COORD_ABS {
            ws.x = None;
        }
    }
    if let Some(y) = ws.y {
        if y <= INVALID_COORD_MIN || y.abs() > MAX_COORD_ABS {
            ws.y = None;
        }
    }

    ws
}

pub fn load_window_state(config_dir: &Path) -> WindowState {
    let ws = load_config(config_dir).window;
    sanitize_window_state(ws)
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

/// 从 config dir 读取导出设置。
pub fn load_export_settings(config_dir: &Path) -> ExportSettings {
    load_config(config_dir).export
}

/// 将导出设置写入 config dir；保留现有配置中的其他字段。
pub fn save_export_settings(config_dir: &Path, export: &ExportSettings) {
    let mut cfg = load_config(config_dir);
    cfg.export = export.clone();
    save_config(config_dir, &cfg);
}

/// 从 config dir 读取自动备份设置。
pub fn load_auto_backup_settings(config_dir: &Path) -> AutoBackupSettings {
    load_config(config_dir).auto_backup.normalized()
}

/// 将自动备份设置写入 config dir；保留现有配置中的其他字段。
pub fn save_auto_backup_settings(config_dir: &Path, settings: &AutoBackupSettings) {
    let mut cfg = load_config(config_dir);
    cfg.auto_backup = settings.normalized();
    save_config(config_dir, &cfg);
}
