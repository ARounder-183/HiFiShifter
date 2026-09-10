use crate::project::CustomScale;
use crate::time_stretch::UserStretchAlgorithm;
use std::fs;
use std::path::Path;

/// 时间轴吸附/网格设置（对标 REAPER Snap/Grid Settings）。
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct TimelineSnapSettings {
    #[serde(default = "default_true")]
    pub grid_visible: bool,
    #[serde(default = "default_grid_min_spacing_px")]
    pub grid_min_spacing_px: u32,
    #[serde(default)]
    pub swing_enabled: bool,
    #[serde(default)]
    pub swing_percent: u32,
    #[serde(default = "default_true")]
    pub adjust_clips_on_swing_change: bool,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_snap_distance_px")]
    pub snap_distance_px: u32,
    #[serde(default)]
    pub snap_relative_to_grid: bool,
    /// 拖拽时显示吸附竖线高亮（纯视觉开关，不影响吸附行为）。
    #[serde(default = "default_true")]
    pub snap_highlight_enabled: bool,
    #[serde(default = "default_true")]
    pub snap_clips_to_selection_markers_cursor: bool,
    #[serde(default = "default_true")]
    pub snap_clips_to_grid: bool,
    #[serde(default = "default_true")]
    pub snap_selection_to_selection_markers_cursor: bool,
    #[serde(default = "default_true")]
    pub snap_selection_to_grid: bool,
    #[serde(default = "default_true")]
    pub snap_cursor_to_selection_markers_cursor: bool,
    #[serde(default = "default_true")]
    pub snap_cursor_to_grid: bool,
    #[serde(
        default = "default_true",
        alias = "gridSnapFollowsGridVisibility",
        alias = "grid_snap_follows_grid_visibility"
    )]
    pub snap_follows_grid_visibility: bool,
    #[serde(default)]
    pub snap_to_grid_any_distance: bool,
    #[serde(default)]
    pub use_independent_snap_spacing: bool,
    #[serde(default = "default_grid_size")]
    pub snap_spacing: String,
    #[serde(default = "default_grid_min_spacing_px")]
    pub snap_spacing_min_px: u32,
    #[serde(default = "default_true")]
    pub snap_clip_edges: bool,
    #[serde(default = "default_true")]
    pub snap_clip_snap_offset: bool,
    #[serde(default = "default_true")]
    pub snap_across_tracks: bool,
    #[serde(default)]
    pub snap_track_distance: u32,
    #[serde(default = "default_true")]
    pub snap_razor_edits: bool,
    #[serde(default)]
    pub snap_to_project_sample_rate: bool,
    #[serde(default = "default_true")]
    pub snap_clips_to_source_media: bool,
    #[serde(default)]
    pub force_selections_to_multiples: bool,
    #[serde(default = "default_grid_size")]
    pub selection_multiple: String,
    #[serde(default = "default_true")]
    pub sync_arrange_and_midi_grid: bool,
}

fn default_grid_min_spacing_px() -> u32 {
    8
}

fn default_snap_distance_px() -> u32 {
    4
}

impl Default for TimelineSnapSettings {
    fn default() -> Self {
        Self {
            grid_visible: true,
            grid_min_spacing_px: default_grid_min_spacing_px(),
            swing_enabled: false,
            swing_percent: 0,
            adjust_clips_on_swing_change: true,
            enabled: true,
            snap_distance_px: default_snap_distance_px(),
            snap_relative_to_grid: false,
            snap_highlight_enabled: true,
            snap_clips_to_selection_markers_cursor: true,
            snap_clips_to_grid: true,
            snap_selection_to_selection_markers_cursor: true,
            snap_selection_to_grid: true,
            snap_cursor_to_selection_markers_cursor: true,
            snap_cursor_to_grid: true,
            snap_follows_grid_visibility: true,
            snap_to_grid_any_distance: false,
            use_independent_snap_spacing: false,
            snap_spacing: default_grid_size(),
            snap_spacing_min_px: default_grid_min_spacing_px(),
            snap_clip_edges: true,
            snap_clip_snap_offset: true,
            snap_across_tracks: true,
            snap_track_distance: 0,
            snap_razor_edits: true,
            snap_to_project_sample_rate: false,
            snap_clips_to_source_media: true,
            force_selections_to_multiples: false,
            selection_multiple: default_grid_size(),
            sync_arrange_and_midi_grid: true,
        }
    }
}

impl TimelineSnapSettings {
    fn valid_grid(value: &str) -> bool {
        matches!(
            value,
            "1/1"
                | "1/2"
                | "1/4"
                | "1/8"
                | "1/16"
                | "1/32"
                | "1/64"
                | "1/1d"
                | "1/2d"
                | "1/4d"
                | "1/8d"
                | "1/16d"
                | "1/32d"
                | "1/64d"
                | "1/1t"
                | "1/2t"
                | "1/4t"
                | "1/8t"
                | "1/16t"
                | "1/32t"
                | "1/64t"
        )
    }

    pub fn normalize(&mut self) {
        self.grid_min_spacing_px = self.grid_min_spacing_px.clamp(2, 200);
        self.swing_percent = self.swing_percent.clamp(0, 100);
        self.snap_distance_px = self.snap_distance_px.clamp(0, 200);
        self.snap_spacing_min_px = self.snap_spacing_min_px.clamp(2, 200);
        self.snap_track_distance = self.snap_track_distance.clamp(0, 32);
        if !Self::valid_grid(&self.snap_spacing) {
            self.snap_spacing = default_grid_size();
        }
        if !Self::valid_grid(&self.selection_multiple) {
            self.selection_multiple = default_grid_size();
        }
    }
}

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
/// 静音检测对话框的上次使用参数。
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SilenceDetectSettings {
    #[serde(default = "default_silence_method")]
    pub method: String,
    #[serde(default = "default_silence_threshold")]
    pub threshold_db: f64,
    #[serde(default)]
    pub adaptive: bool,
    #[serde(default = "default_silence_min_silence")]
    pub min_silence_ms: f64,
    #[serde(default)]
    pub min_sound_ms: f64,
    #[serde(default = "default_silence_padding")]
    pub padding_ms: f64,
    #[serde(default = "default_silence_cut_fade")]
    pub cut_fade_ms: f64,
    #[serde(default = "default_silence_action")]
    pub action: String,
    #[serde(default = "default_true_value")]
    pub delete_silent_clips: bool,
    #[serde(default)]
    pub sync_all_takes: bool,
}

impl Default for SilenceDetectSettings {
    fn default() -> Self {
        Self {
            method: default_silence_method(),
            threshold_db: default_silence_threshold(),
            adaptive: false,
            min_silence_ms: default_silence_min_silence(),
            min_sound_ms: 0.0,
            padding_ms: default_silence_padding(),
            cut_fade_ms: default_silence_cut_fade(),
            action: default_silence_action(),
            delete_silent_clips: true,
            sync_all_takes: false,
        }
    }
}

fn default_silence_method() -> String {
    "rms".to_string()
}
fn default_silence_threshold() -> f64 {
    -50.0
}
fn default_silence_min_silence() -> f64 {
    120.0
}
fn default_silence_padding() -> f64 {
    10.0
}
fn default_silence_cut_fade() -> f64 {
    5.0
}
fn default_silence_action() -> String {
    "close".to_string()
}
fn default_true_value() -> bool {
    true
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct UiSettings {
    #[serde(default = "default_true")]
    pub auto_crossfade: bool,
    /// 空间足够时在 Clip 内显示全部 Take 波形。
    #[serde(default = "default_true")]
    pub show_all_takes: bool,
    #[serde(default = "default_true")]
    pub split_transition_enabled: bool,
    #[serde(default = "default_split_transition_mode")]
    pub split_transition_mode: String,
    #[serde(default = "default_split_transition_duration_unit")]
    pub split_transition_duration_unit: String,
    #[serde(default = "default_split_transition_duration_sec")]
    pub split_transition_duration_sec: f64,
    #[serde(default = "default_split_transition_duration_percent")]
    pub split_transition_duration_percent: f64,
    #[serde(default = "default_split_transition_curve")]
    pub split_transition_curve: String,
    #[serde(default = "default_split_transition_overlap_crossfade")]
    pub split_transition_overlap_crossfade: String,
    #[serde(default = "default_true", alias = "gridSnap", alias = "grid_snap")]
    pub snap_enabled: bool,
    #[serde(default = "default_grid_size")]
    pub grid_size: String,
    /// 完整时间轴吸附/网格设置。
    #[serde(default)]
    pub timeline_snap: TimelineSnapSettings,
    /// Tempo Map 标尺行可见性（默认开启）。
    #[serde(default = "default_true")]
    pub tempo_map_visible: bool,
    #[serde(default = "default_primary_time_unit")]
    pub primary_time_unit: String,
    #[serde(default = "default_secondary_time_unit")]
    pub secondary_time_unit: String,
    #[serde(default = "default_ruler_label_spacing_px")]
    pub ruler_label_spacing_px: u32,
    #[serde(default = "default_true")]
    pub show_playhead_time_in_track_header: bool,
    #[serde(default = "default_true")]
    pub param_editor_sync_timeline: bool,
    #[serde(default = "default_true")]
    pub param_editor_timeline_click_select_track: bool,
    #[serde(default)]
    pub pitch_snap: bool,
    #[serde(default = "default_pitch_snap_unit")]
    pub pitch_snap_unit: String,
    #[serde(default)]
    pub pitch_snap_tolerance_cents: u32,
    /// 音阶吸附使用的音阶（默认 C）。与前端 pitchSnapScale 对应。
    #[serde(default = "default_pitch_snap_scale")]
    pub pitch_snap_scale: String,
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

    // ── 节拍器（Metronome）──
    /// 是否启用（传输栏节拍器按钮 / playback.metronome 快捷键切换）。
    #[serde(default)]
    pub metronome_enabled: bool,
    /// 音量 0..1（默认 0.5）。
    #[serde(default = "default_metronome_gain")]
    pub metronome_gain: f64,
    /// 细分模式：grid（跟随网格标尺，默认）/ beat（仅每拍）/ bar（仅小节首）。
    #[serde(default = "default_metronome_mode")]
    pub metronome_mode: String,
    /// 是否强调小节首（重音音色）。
    #[serde(default = "default_true")]
    pub metronome_accent: bool,
    /// 音色：click（默认）/ woodblock / beep（程序化合成）。
    #[serde(default = "default_metronome_sound")]
    pub metronome_sound: String,

    /// 静音检测对话框的上次使用参数。
    #[serde(default)]
    pub silence_detect_options: SilenceDetectSettings,

    // ── 主总线（Master bus）── 见 `audio/master_bus.rs` 与 P0-2
    /// 主总线软削波开关。关闭后退回"裸求和 + 末端硬截断"的旧行为。
    #[serde(default = "default_true")]
    pub master_soft_clip_enabled: bool,
    /// 主总线软削波膝值：|x| ≤ knee 线性通过。会被钳制到合法范围。
    #[serde(default = "default_master_soft_clip_knee")]
    pub master_soft_clip_knee: f32,

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
    /// 波纹编辑（自动跟进）模式：off / track / all（对应 REAPER 的 Ripple Editing）。
    /// - `off`：关闭（默认）。
    /// - `track`：仅被编辑的轨道上的后续剪辑一起跟进。
    /// - `all`：所有轨道上位于编辑点之后的剪辑一起跟进。
    #[serde(default = "default_ripple_mode")]
    pub ripple_mode: String,

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
    /// 指定 BPM 模式使用的 BPM 值（默认 120，与前端输入框初始值一致）。
    #[serde(default = "default_midi_specified_bpm")]
    pub midi_specified_bpm: Option<f64>,
    #[serde(default = "default_true")]
    pub midi_close_leading_gap: bool,
    /// MIDI 导入目标（统一弹窗）：pitchRef = 创建音高参考块，pitchParam = 导入到音高参数。
    /// 两个导入场景分别持久化、分别存储：
    /// - 时间轴场景（文件菜单导入 / 拖拽到轨道视图）：默认 pitchRef；
    /// - 参数编辑器场景（编辑器内导入按钮 / 拖拽到编辑器内）：默认 pitchParam。
    /// （旧版单一字段 midiImportTarget 已移除，不再做读取兼容。）
    #[serde(default = "default_midi_import_target_timeline")]
    pub midi_import_target_menu: String,
    #[serde(default = "default_midi_import_target_timeline")]
    pub midi_import_target_drag_drop: String,
    #[serde(default = "default_midi_import_target_param_editor")]
    pub midi_import_target_param_editor: String,
    #[serde(default = "default_midi_import_target_param_editor")]
    pub midi_import_target_reaper_clipboard: String,
    /// MIDI 导入为 Tempo Map（默认关闭）。
    #[serde(default)]
    pub midi_import_as_tempo_map: bool,
    /// 导入 Tempo（默认开启）。
    #[serde(default = "default_true")]
    pub midi_import_tempo_map_tempo: bool,
    /// 导入拍号（默认开启）。
    #[serde(default = "default_true")]
    pub midi_import_tempo_map_time_signature: bool,
    /// 导入音阶（默认关闭：大多数 MIDI 文件只写默认 C 大调）。
    #[serde(default)]
    pub midi_import_tempo_map_key_signature: bool,
    /// ONNX Runtime execution provider preference ("auto", "cpu", "gpu").
    /// Persisted so the user's GPU/CPU choice survives restarts.
    #[serde(default = "default_ort_ep")]
    pub ort_ep: String,
    /// DirectML device ID override (Windows only).
    /// When set, DirectML will use the specified GPU adapter index.
    /// When `None`, DirectML auto-selects the best GPU using
    /// PerformancePreference::HighPerformance + DeviceFilter::Gpu.
    /// GPU adapter indices are 0-based and match DXGI adapter enumeration order.
    #[serde(default)]
    pub ort_device_id: Option<i32>,
    /// GPU 设备 ID（对应前端 gpuDeviceId，0 表示默认）。
    #[serde(default)]
    pub gpu_device_id: i32,
    /// 后台预渲染：启用后，当编辑操作使渲染缓存失效时，
    /// 立即在后台启动预渲染，而不是等到用户按下播放时才渲染。
    /// 用户可在渲染进行中随时开始播放已渲染完成的部分。
    /// Background pre-render: when enabled, immediately start
    /// rendering in the background after editing invalidates the
    /// render cache. Users can play already-rendered content
    /// at any time during rendering.
    #[serde(default = "default_true")]
    pub auto_background_render: bool,
    /// 自动重新加载已修改的媒体文件（默认开启）。
    /// 启用后，窗口重新获得焦点并检测到媒体内容变化时，
    /// 后端会在后台直接重新加载原路径，无需弹出确认窗口。
    #[serde(default = "default_true")]
    pub auto_reload_modified_media: bool,
    /// 为新的音频块启用循环（Loop / 循环源，默认开启）。
    ///
    /// 作用范围（仅影响"新 Clip"，绝不自动修改时间轴上已有 Clip）：
    /// - 导入新媒体作为 Clip（文件导入/拖放/录音等）时的初始 Loop 属性；
    /// - 打开旧版本工程（v3 及更早，Clip 不携带 loop_enabled 字段）时的迁移默认值；
    /// - REAPER 工程导入 / REAPER 剪贴板粘贴中未显式写出 LOOP 标记的 ITEM 的默认值；
    /// - VocalShifter 等其他格式导入生成的音频 Clip 的默认值。
    #[serde(default = "default_true")]
    pub loop_new_clips: bool,
    /// 同步编辑所有 Take：启用后，对 active Take 的内容级编辑
    /// （源偏移、播放速率、倒放、Loop、增益）会尝试同步到同一 Clip 的其余 Take。
    #[serde(default = "default_true")]
    pub sync_edits_across_takes: bool,
}

/// "为新的音频块启用循环"的进程级生效值（默认 true）。
///
/// 由 `commands::ui_settings::get_ui_settings` / `save_ui_settings` 在加载与
/// 保存设置时同步；供 `TimelineState::add_clip`、各格式 importer、旧工程
/// 迁移等无法访问 AppState 的创建点读取。
pub static LOOP_NEW_CLIPS_DEFAULT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// 读取当前生效的"新 Clip 默认 Loop 属性"。
pub fn loop_new_clips_default() -> bool {
    LOOP_NEW_CLIPS_DEFAULT.load(std::sync::atomic::Ordering::Relaxed)
}

/// 同步"新 Clip 默认 Loop 属性"的进程级生效值。
pub fn set_loop_new_clips_default(enabled: bool) {
    LOOP_NEW_CLIPS_DEFAULT.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// "同步编辑所有 Take"的进程级生效值（默认开启）。
pub static SYNC_EDITS_ACROSS_TAKES: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

pub fn sync_edits_across_takes() -> bool {
    SYNC_EDITS_ACROSS_TAKES.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_sync_edits_across_takes(enabled: bool) {
    SYNC_EDITS_ACROSS_TAKES.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

fn default_ort_ep() -> String {
    "auto".to_string()
}

/// 导出音频设置（持久化到 app_config.json）
///
/// 用于记住导出窗口中不同导出类型的输出目录与文件名设置。
///
/// 新增的格式 / 编码参数字段沿用本文件既有惯例以字符串存枚举值
/// （如 `"wav"`、`"tpdf"`），解析由 `crate::encode` 的 `from_name`
/// 宽松完成，非法值回退默认，避免手改配置破坏整体加载。
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
    /// 输出格式："wav" | "mp3" | "flac"（缺省 → wav）。
    #[serde(default)]
    pub format: Option<String>,
    /// 导出声道："stereo" | "mono"（缺省 → stereo）。
    #[serde(default)]
    pub channel_mode: Option<String>,
    /// 抖动："none" | "tpdf"（缺省 → none）。
    #[serde(default)]
    pub dither: Option<String>,
    /// MP3 码率模式："cbr" | "vbr"（缺省 → vbr）。
    #[serde(default)]
    pub mp3_mode: Option<String>,
    #[serde(default)]
    pub mp3_bitrate_kbps: Option<u32>,
    #[serde(default)]
    pub mp3_vbr_quality_index: Option<u8>,
    #[serde(default)]
    pub mp3_tags: Option<crate::encode::Mp3Tags>,
    /// FLAC 位深："i16" | "i24"（缺省 → i24）。
    #[serde(default)]
    pub flac_bit_depth: Option<String>,
    #[serde(default)]
    pub flac_compression_level: Option<u8>,
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

/// 录音设置（持久化到 app_config.json）
///
/// - `source_device`：录音源 ID。`"default"` 表示系统默认录音设备；
///   `"input:<name>"` 表示具体录音设备；`"loopback:<name>"` 表示系统声音回环
///   （Windows WASAPI 下可直接捕获播放器输出）。
/// - `path_template`：输出路径模板，支持 `<ProjectFolder>`、`<ProjectName>`
///   与 strftime 时间格式（例如 `%Y-%m-%d-%H-%M-%S`）。
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct RecordingSettings {
    #[serde(default = "default_recording_source_device")]
    pub source_device: String,
    /// "device" (microphone), "loopback" (system sound) or "application".
    #[serde(default = "default_recording_capture_mode")]
    pub capture_mode: String,
    /// Output device used for system-sound capture: "default" or an endpoint
    /// id / legacy device name.
    #[serde(default = "default_recording_loopback_device")]
    pub loopback_device: String,
    /// Selected application id, e.g. "pid:1234".
    #[serde(default)]
    pub capture_app_id: String,
    /// Display name of the selected application (UI convenience only).
    #[serde(default)]
    pub capture_app_name: String,
    /// Executable name of the selected application, used to re-match the app
    /// after it restarts with a new PID.
    #[serde(default)]
    pub capture_app_process: String,
    #[serde(default = "default_recording_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_recording_bit_depth")]
    pub bit_depth: u32,
    #[serde(default = "default_recording_channels")]
    pub channels: u16,
    #[serde(default = "default_recording_gain_db")]
    pub input_gain_db: f32,
    #[serde(default)]
    pub monitor_enabled: bool,
    #[serde(default = "default_recording_gain_db")]
    pub monitor_gain_db: f32,
    #[serde(default)]
    pub countdown_sec: u32,
    #[serde(default)]
    pub auto_normalize: bool,
    #[serde(default)]
    pub auto_stop_at_selection_end: bool,
    #[serde(default = "default_recording_path_template")]
    pub path_template: String,
}

fn default_recording_source_device() -> String {
    "default".to_string()
}

fn default_recording_capture_mode() -> String {
    "device".to_string()
}

fn default_recording_loopback_device() -> String {
    "default".to_string()
}

fn default_recording_sample_rate() -> u32 {
    48_000
}

fn default_recording_bit_depth() -> u32 {
    24
}

fn default_recording_channels() -> u16 {
    2
}

fn default_recording_gain_db() -> f32 {
    0.0
}

fn default_recording_path_template() -> String {
    "<ProjectFolder>/HiFiShifter Record/%Y-%m-%d-%H-%M-%S.wav".to_string()
}

impl Default for RecordingSettings {
    fn default() -> Self {
        Self {
            source_device: default_recording_source_device(),
            capture_mode: default_recording_capture_mode(),
            loopback_device: default_recording_loopback_device(),
            capture_app_id: String::new(),
            capture_app_name: String::new(),
            capture_app_process: String::new(),
            sample_rate: default_recording_sample_rate(),
            bit_depth: default_recording_bit_depth(),
            channels: default_recording_channels(),
            input_gain_db: default_recording_gain_db(),
            monitor_enabled: false,
            monitor_gain_db: default_recording_gain_db(),
            countdown_sec: 0,
            auto_normalize: false,
            auto_stop_at_selection_end: false,
            path_template: default_recording_path_template(),
        }
    }
}

impl RecordingSettings {
    pub fn normalized(&self) -> Self {
        let source_device = {
            let trimmed = self.source_device.trim();
            if trimmed.is_empty() {
                default_recording_source_device()
            } else {
                trimmed.to_string()
            }
        };
        let capture_mode = {
            let trimmed = self.capture_mode.trim();
            match trimmed {
                "loopback" | "application" => trimmed.to_string(),
                // Migrate legacy `source_device = "loopback:<name>"` values
                // saved by older builds into the new capture mode.
                _ if source_device.starts_with("loopback:") => "loopback".to_string(),
                _ => default_recording_capture_mode(),
            }
        };
        let loopback_device = {
            let trimmed = self.loopback_device.trim();
            if trimmed.is_empty() {
                if let Some(name) = source_device.strip_prefix("loopback:") {
                    name.trim().to_string()
                } else {
                    default_recording_loopback_device()
                }
            } else {
                trimmed.to_string()
            }
        };
        let sample_rate = if (8_000..=192_000).contains(&self.sample_rate) {
            self.sample_rate
        } else {
            default_recording_sample_rate()
        };
        let bit_depth = match self.bit_depth {
            16 | 32 => self.bit_depth,
            _ => 24,
        };
        let channels = if self.channels == 1 { 1 } else { 2 };
        let input_gain_db = self.input_gain_db.clamp(-24.0, 24.0);
        let monitor_gain_db = self.monitor_gain_db.clamp(-24.0, 24.0);
        let path_template = {
            let trimmed = self.path_template.trim();
            if trimmed.is_empty() {
                default_recording_path_template()
            } else {
                trimmed.to_string()
            }
        };

        Self {
            source_device,
            capture_mode,
            loopback_device,
            capture_app_id: self.capture_app_id.trim().to_string(),
            capture_app_name: self.capture_app_name.trim().to_string(),
            capture_app_process: self.capture_app_process.trim().to_string(),
            sample_rate,
            bit_depth,
            channels,
            input_gain_db,
            monitor_enabled: self.monitor_enabled,
            monitor_gain_db,
            countdown_sec: self.countdown_sec.clamp(0, 10),
            auto_normalize: self.auto_normalize,
            auto_stop_at_selection_end: self.auto_stop_at_selection_end,
            path_template,
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
            format: None,
            channel_mode: None,
            dither: None,
            mp3_mode: None,
            mp3_bitrate_kbps: None,
            mp3_vbr_quality_index: None,
            mp3_tags: None,
            flac_bit_depth: None,
            flac_compression_level: None,
        }
    }
}

fn default_true() -> bool {
    true
}
/// 主总线软削波膝值的默认值，与 `audio/master_bus.rs` 的常量保持单一来源。
fn default_master_soft_clip_knee() -> f32 {
    crate::master_bus::DEFAULT_SOFT_CLIP_KNEE
}
fn default_pitch_snap_unit() -> String {
    "semitone".to_string()
}
fn default_pitch_snap_scale() -> String {
    "C".to_string()
}
fn default_grid_size() -> String {
    "1/4".to_string()
}
fn default_primary_time_unit() -> String {
    "barBeats".to_string()
}
fn default_secondary_time_unit() -> String {
    "clock".to_string()
}
fn default_ruler_label_spacing_px() -> u32 {
    110
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

fn default_ripple_mode() -> String {
    "off".to_string()
}

fn default_metronome_gain() -> f64 {
    0.5
}

fn default_metronome_mode() -> String {
    "grid".to_string()
}

fn default_metronome_sound() -> String {
    "click".to_string()
}

fn default_split_transition_mode() -> String {
    "overlap".to_string()
}

fn default_split_transition_duration_sec() -> f64 {
    0.01
}

fn default_split_transition_duration_unit() -> String {
    "seconds".to_string()
}

fn default_split_transition_duration_percent() -> f64 {
    1.0
}

/// 分割过渡“淡化曲线”：新版 REAPER 预设 id；"keep"（默认）= 分割后保留
/// 原 Clip 的淡化曲线类型，不再修改。
fn default_split_transition_curve() -> String {
    "keep".to_string()
}

fn default_split_transition_overlap_crossfade() -> String {
    "auto".to_string()
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

/// 指定 BPM 模式的默认值（与前端输入框初始值一致）。
fn default_midi_specified_bpm() -> Option<f64> {
    Some(120.0)
}

/// 时间轴 MIDI 导入场景（文件菜单 / 拖拽到轨道视图）的默认导入目标。
fn default_midi_import_target_timeline() -> String {
    "pitchRef".to_string()
}

/// 参数编辑器 MIDI 导入场景（编辑器导入按钮 / 拖拽到编辑器内）的默认导入目标。
fn default_midi_import_target_param_editor() -> String {
    "pitchParam".to_string()
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            auto_crossfade: true,
            show_all_takes: true,
            split_transition_enabled: true,
            split_transition_mode: default_split_transition_mode(),
            split_transition_duration_unit: default_split_transition_duration_unit(),
            split_transition_duration_sec: default_split_transition_duration_sec(),
            split_transition_duration_percent: default_split_transition_duration_percent(),
            split_transition_curve: default_split_transition_curve(),
            split_transition_overlap_crossfade: default_split_transition_overlap_crossfade(),
            snap_enabled: true,
            grid_size: default_grid_size(),
            timeline_snap: TimelineSnapSettings::default(),
            tempo_map_visible: true,
            primary_time_unit: default_primary_time_unit(),
            secondary_time_unit: default_secondary_time_unit(),
            ruler_label_spacing_px: default_ruler_label_spacing_px(),
            show_playhead_time_in_track_header: true,
            param_editor_sync_timeline: true,
            param_editor_timeline_click_select_track: true,
            pitch_snap: false,
            pitch_snap_unit: default_pitch_snap_unit(),
            pitch_snap_tolerance_cents: 0,
            pitch_snap_scale: default_pitch_snap_scale(),
            playhead_zoom: false,
            auto_scroll: false,
            param_editor_seek_playhead: true,
            show_clipboard_preview: true,
            show_param_value_popup: true,
            lock_param_lines: true,
            metronome_enabled: false,
            metronome_gain: default_metronome_gain(),
            metronome_mode: default_metronome_mode(),
            metronome_accent: true,
            metronome_sound: default_metronome_sound(),
            silence_detect_options: SilenceDetectSettings::default(),
            master_soft_clip_enabled: true,
            master_soft_clip_knee: default_master_soft_clip_knee(),
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
            ripple_mode: default_ripple_mode(),

            midi_import_position: default_midi_import_position(),
            midi_fill_gaps: false,
            midi_multi_track_merge: true,
            midi_import_bpm_as_project: false,
            midi_note_bpm_mode: default_midi_note_bpm_mode(),
            midi_specified_bpm: default_midi_specified_bpm(),
            midi_close_leading_gap: true,
            midi_import_target_menu: default_midi_import_target_timeline(),
            midi_import_target_drag_drop: default_midi_import_target_timeline(),
            midi_import_target_param_editor: default_midi_import_target_param_editor(),
            midi_import_target_reaper_clipboard: default_midi_import_target_param_editor(),
            midi_import_as_tempo_map: false,
            midi_import_tempo_map_tempo: true,
            midi_import_tempo_map_time_signature: true,
            midi_import_tempo_map_key_signature: false,
            ort_ep: default_ort_ep(),
            ort_device_id: None,
            gpu_device_id: 0,
            auto_background_render: true,
            auto_reload_modified_media: true,
            loop_new_clips: true,
            sync_edits_across_takes: true,
        }
    }
}

impl UiSettings {
    /// 规范化分割过渡相关设置，避免损坏/越界的持久化值影响编辑行为。
    pub fn normalize_split_transition(&mut self) {
        if !["fade", "overlap"].contains(&self.split_transition_mode.as_str()) {
            self.split_transition_mode = default_split_transition_mode();
        }
        if !self.split_transition_duration_sec.is_finite() {
            self.split_transition_duration_sec = default_split_transition_duration_sec();
        } else {
            self.split_transition_duration_sec =
                self.split_transition_duration_sec.clamp(0.001, 10.0);
        }
        if !["seconds", "percent"].contains(&self.split_transition_duration_unit.as_str()) {
            self.split_transition_duration_unit = default_split_transition_duration_unit();
        }
        if !self.split_transition_duration_percent.is_finite() {
            self.split_transition_duration_percent = default_split_transition_duration_percent();
        } else {
            self.split_transition_duration_percent =
                self.split_transition_duration_percent.clamp(0.01, 100.0);
        }
        // 新版预设 id（"keep" + FADE_PRESETS）；旧命名曲线（淡化模型重构前）
        // 在此迁移到等价预设：exponential→lateSlight、logarithmic→convexSlight、
        // sine/scurve→sSlight、linear→linear；其余未知值回退默认 "keep"。
        self.split_transition_curve = match self.split_transition_curve.as_str() {
            "keep" | "linear" | "convexSlight" | "lateSlight" | "convexSharp"
            | "lateSharp" | "sSlight" | "sSharp" => self.split_transition_curve.clone(),
            "exponential" => "lateSlight".to_string(),
            "logarithmic" => "convexSlight".to_string(),
            "sine" | "scurve" => "sSlight".to_string(),
            _ => default_split_transition_curve(),
        };
        if !["auto", "always"].contains(&self.split_transition_overlap_crossfade.as_str()) {
            self.split_transition_overlap_crossfade = default_split_transition_overlap_crossfade();
        }
        self.normalize_time_display();
        self.timeline_snap.normalize();
        self.normalize_ripple_mode();
    }

    /// 规范化波纹编辑模式，避免损坏/未知的持久化值影响编辑行为。
    pub fn normalize_ripple_mode(&mut self) {
        if !["off", "track", "all"].contains(&self.ripple_mode.as_str()) {
            self.ripple_mode = default_ripple_mode();
        }
    }

    /// 规范化时间轴时间显示相关设置，避免损坏/越界的持久化值影响界面。
    pub fn normalize_time_display(&mut self) {
        const VALID_UNITS: [&str; 4] = ["barBeats", "barDivisions", "seconds", "clock"];
        if !VALID_UNITS.contains(&self.primary_time_unit.as_str()) {
            self.primary_time_unit = default_primary_time_unit();
        }
        if self.secondary_time_unit != "none"
            && !VALID_UNITS.contains(&self.secondary_time_unit.as_str())
        {
            self.secondary_time_unit = default_secondary_time_unit();
        }
        self.ruler_label_spacing_px = self.ruler_label_spacing_px.clamp(40, 320);
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
        assert!(settings.sync_edits_across_takes);
        assert!(settings.split_transition_enabled);
        assert_eq!(settings.split_transition_mode, "overlap");
        assert_eq!(settings.split_transition_duration_unit, "seconds");
        assert!((settings.split_transition_duration_sec - 0.01).abs() < 1e-12);
        assert!((settings.split_transition_duration_percent - 1.0).abs() < 1e-12);
        assert_eq!(settings.split_transition_curve, "keep");
        assert_eq!(settings.split_transition_overlap_crossfade, "auto");
    }

    #[test]
    fn ui_settings_normalizes_split_transition_values() {
        let mut settings = UiSettings {
            split_transition_mode: "bogus".to_string(),
            split_transition_duration_unit: "frames".to_string(),
            split_transition_duration_sec: 9999.0,
            split_transition_duration_percent: 999.0,
            split_transition_curve: "nope".to_string(),
            split_transition_overlap_crossfade: "sometimes".to_string(),
            ..UiSettings::default()
        };
        settings.normalize_split_transition();
        assert_eq!(settings.split_transition_mode, "overlap");
        assert_eq!(settings.split_transition_duration_unit, "seconds");
        assert!((settings.split_transition_duration_sec - 10.0).abs() < 1e-12);
        assert!((settings.split_transition_duration_percent - 100.0).abs() < 1e-12);
        assert_eq!(settings.split_transition_curve, "keep");
        assert_eq!(settings.split_transition_overlap_crossfade, "auto");
    }

    #[test]
    fn ui_settings_migrates_legacy_split_transition_curve_names() {
        // 旧命名曲线（淡化模型重构前）→ 新版预设 id；未知值回退 "keep"。
        let migrate = |curve: &str| {
            let mut settings = UiSettings {
                split_transition_curve: curve.to_string(),
                ..UiSettings::default()
            };
            settings.normalize_split_transition();
            settings.split_transition_curve
        };
        assert_eq!(migrate("keep"), "keep");
        assert_eq!(migrate("linear"), "linear");
        assert_eq!(migrate("lateSharp"), "lateSharp");
        assert_eq!(migrate("sine"), "sSlight");
        assert_eq!(migrate("scurve"), "sSlight");
        assert_eq!(migrate("exponential"), "lateSlight");
        assert_eq!(migrate("logarithmic"), "convexSlight");
        assert_eq!(migrate("bogus"), "keep");
    }

    #[test]
    fn ui_settings_normalizes_ripple_mode() {
        let settings = UiSettings::default();
        assert_eq!(settings.ripple_mode, "off");

        let mut bogus = UiSettings {
            ripple_mode: "sideways".to_string(),
            ..UiSettings::default()
        };
        bogus.normalize_ripple_mode();
        assert_eq!(bogus.ripple_mode, "off");

        let mut track_mode = UiSettings {
            ripple_mode: "track".to_string(),
            ..UiSettings::default()
        };
        track_mode.normalize_ripple_mode();
        assert_eq!(track_mode.ripple_mode, "track");
    }

    // ── 窗口状态：清洗 / 合并 / 显示器交集校验 ────────────────────────────────

    use super::{
        centered_position, clamp_size_to_bounds, load_window_state, merge_window_state,
        overlap_size, sanitize_window_state, save_window_state, PhysicalRect, WindowState,
        MIN_VISIBLE_OVERLAP_H, MIN_VISIBLE_OVERLAP_W,
    };

    #[test]
    fn sanitize_window_state_rejects_degenerate_sizes() {
        // 历史版本最小化时落盘的 width=0.0 必须被清洗
        let ws = WindowState {
            x: Some(100),
            y: Some(100),
            width: Some(0.0),
            height: Some(-5.0),
            maximized: Some(false),
            fullscreen: Some(false),
        };
        let clean = sanitize_window_state(ws);
        assert_eq!(clean.width, None);
        assert_eq!(clean.height, None);
        assert_eq!(clean.x, Some(100));
        assert_eq!(clean.y, Some(100));

        let bad = WindowState {
            width: Some(f64::NAN),
            height: Some(100_000_000.0),
            ..WindowState::default()
        };
        let clean = sanitize_window_state(bad);
        assert_eq!(clean.width, None);
        assert_eq!(clean.height, None);

        let ok = WindowState {
            width: Some(1280.0),
            height: Some(800.0),
            ..WindowState::default()
        };
        let clean = sanitize_window_state(ok);
        assert_eq!(clean.width, Some(1280.0));
        assert_eq!(clean.height, Some(800.0));
    }

    #[test]
    fn sanitize_window_state_coordinate_rules() {
        // 多显示器布局下 -25600 是合法坐标（左侧副屏），不得被阈值误杀 ——
        // 是否在已连接显示器内由恢复侧交集校验负责
        let ws = WindowState {
            x: Some(-25600),
            y: Some(-25600),
            ..WindowState::default()
        };
        let clean = sanitize_window_state(ws);
        assert_eq!(clean.x, Some(-25600));
        assert_eq!(clean.y, Some(-25600));

        // Windows 最小化停泊哨兵（物理 -32000 经低倍缩放换算后仍 ≤ -32000
        // 的原始值）与超界坐标必须拒绝
        let ws = WindowState {
            x: Some(-32000),
            y: Some(-32000),
            ..WindowState::default()
        };
        let clean = sanitize_window_state(ws);
        assert_eq!(clean.x, None);
        assert_eq!(clean.y, None);

        let ws = WindowState {
            x: Some(5_000_000),
            y: Some(-5_000_000),
            ..WindowState::default()
        };
        let clean = sanitize_window_state(ws);
        assert_eq!(clean.x, None);
        assert_eq!(clean.y, None);
    }

    #[test]
    fn merge_window_state_keeps_last_good_geometry_when_capture_unavailable() {
        // 最小化/最大化/全屏/隐藏关闭时几何捕获不到（全 None）→ 保留上一次好值
        let prev = WindowState {
            x: Some(100),
            y: Some(200),
            width: Some(1280.0),
            height: Some(800.0),
            maximized: Some(false),
            fullscreen: Some(false),
        };
        let incoming = WindowState {
            maximized: Some(true),
            ..WindowState::default()
        };
        let merged = merge_window_state(prev, incoming);
        assert_eq!(merged.x, Some(100));
        assert_eq!(merged.y, Some(200));
        assert_eq!(merged.width, Some(1280.0));
        assert_eq!(merged.height, Some(800.0));
        assert_eq!(merged.maximized, Some(true));
    }

    #[test]
    fn merge_window_state_self_heals_poisoned_geometry() {
        // 历史污染文件（width=0.0）在下一次保存时被清洗掉
        let prev = WindowState {
            x: Some(-25600),
            y: Some(-25600),
            width: Some(0.0),
            height: Some(0.0),
            maximized: Some(false),
            fullscreen: Some(false),
        };
        let incoming = WindowState {
            maximized: Some(false),
            ..WindowState::default()
        };
        let merged = merge_window_state(prev, incoming);
        assert_eq!(merged.width, None);
        assert_eq!(merged.height, None);
        // 坐标在合法值域内保留（屏幕外与否交给恢复侧交集校验）
        assert_eq!(merged.x, Some(-25600));
        assert_eq!(merged.y, Some(-25600));
    }

    #[test]
    fn overlap_and_fallback_geometry_helpers() {
        let monitor = PhysicalRect {
            x: 0.0,
            y: 0.0,
            width: 1920.0,
            height: 1080.0,
        };
        // 最小化停泊矩形与显示器不相交
        let parked = PhysicalRect {
            x: -32000.0,
            y: -32000.0,
            width: 1280.0,
            height: 800.0,
        };
        assert_eq!(overlap_size(&parked, &monitor), (0.0, 0.0));

        // 部分可见但低于阈值 → 判定屏幕外
        let barely = PhysicalRect {
            x: monitor.width - 100.0,
            y: 0.0,
            width: 1280.0,
            height: 800.0,
        };
        let (ow, oh) = overlap_size(&barely, &monitor);
        assert_eq!((ow, oh), (100.0, 800.0));
        assert!(ow < MIN_VISIBLE_OVERLAP_W || oh < MIN_VISIBLE_OVERLAP_H);

        // 完整在屏内
        let inside = PhysicalRect {
            x: 10.0,
            y: 10.0,
            width: 800.0,
            height: 600.0,
        };
        assert_eq!(overlap_size(&inside, &monitor), (800.0, 600.0));

        // 尺寸钳制与居中回退
        assert_eq!(
            clamp_size_to_bounds(4000.0, 3000.0, &monitor),
            (1920.0, 1080.0)
        );
        assert_eq!(centered_position((800.0, 600.0), &monitor), (560, 240));
        // 窗口大于显示器时贴左上角
        assert_eq!(centered_position((3840.0, 2160.0), &monitor), (0, 0));
    }

    #[test]
    fn save_window_state_round_trip_merges_and_self_heals() {
        let dir = std::env::temp_dir().join(format!(
            "hifishifter_cfg_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("create temp config dir");

        // 正常保存 → 读回一致
        save_window_state(
            &dir,
            &WindowState {
                x: Some(120),
                y: Some(48),
                width: Some(1440.0),
                height: Some(900.0),
                maximized: Some(false),
                fullscreen: Some(false),
            },
        );
        let ws = load_window_state(&dir);
        assert_eq!(ws.x, Some(120));
        assert_eq!(ws.y, Some(48));
        assert_eq!(ws.width, Some(1440.0));
        assert_eq!(ws.height, Some(900.0));

        // 模拟历史污染文件（0.0 尺寸 / 停泊坐标）直接落盘
        let poisoned = r#"{"window":{"x":-25600,"y":-25600,"width":0.0,"height":0.0,"maximized":false,"fullscreen":false}}"#;
        std::fs::write(dir.join("app_config.json"), poisoned).expect("write poisoned config");

        // 最小化关闭：几何捕获不到（全 None）→ 合并保留污染几何 → 清洗尺寸
        save_window_state(
            &dir,
            &WindowState {
                maximized: Some(true),
                ..WindowState::default()
            },
        );
        let ws = load_window_state(&dir);
        assert_eq!(ws.width, None);
        assert_eq!(ws.height, None);
        assert_eq!(ws.x, Some(-25600));
        assert_eq!(ws.y, Some(-25600));
        assert_eq!(ws.maximized, Some(true));

        // 恢复正常关闭 → 几何被新值覆盖
        save_window_state(
            &dir,
            &WindowState {
                x: Some(10),
                y: Some(20),
                width: Some(1280.0),
                height: Some(720.0),
                maximized: Some(false),
                fullscreen: Some(false),
            },
        );
        let ws = load_window_state(&dir);
        assert_eq!((ws.x, ws.y), (Some(10), Some(20)));
        assert_eq!((ws.width, ws.height), (Some(1280.0), Some(720.0)));

        let _ = std::fs::remove_dir_all(&dir);
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
    #[serde(default)]
    recording: RecordingSettings,
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
    match serde_json::from_str::<AppConfig>(&data) {
        Ok(cfg) => cfg,
        Err(e) => {
            // 解析失败不能无痕回退默认值：那会静默丢掉全部用户设置。
            log::error!("app_config.json parse failed ({e}); trying .bak fallback");
            let bak = config_dir.join("app_config.json.bak");
            if let Ok(bak_data) = fs::read_to_string(&bak) {
                if let Ok(cfg) = serde_json::from_str::<AppConfig>(&bak_data) {
                    return cfg;
                }
            }
            AppConfig::default()
        }
    }
}

fn save_config(config_dir: &Path, cfg: &AppConfig) {
    let path = config_dir.join("app_config.json");
    if let Ok(data) = serde_json::to_string_pretty(cfg) {
        // 原子写：先写临时文件再替换，避免进程中断留下半截 JSON，
        // 下次启动解析失败导致全部设置静默回退默认值。
        let tmp = config_dir.join("app_config.json.tmp");
        match fs::write(&tmp, &data).and_then(|()| {
            // Windows 上 rename 不能覆盖已存在目标，先删旧文件。
            if path.exists() {
                let _ = fs::remove_file(&path);
            }
            fs::rename(&tmp, &path)
        }) {
            Ok(()) => {
                // 保留上一份成功写入的副本，供解析失败时兜底恢复。
                let _ = fs::write(config_dir.join("app_config.json.bak"), &data);
            }
            Err(e) => {
                log::error!("app_config.json save failed: {e}");
                let _ = fs::remove_file(&tmp);
            }
        }
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

    // 坐标校验：拒绝典型的哨兵值（如 -32768）或极端不合理的坐标。
    // 注意：-25600 这类"中等偏移"在多显示器布局下是合法坐标（如左侧
    // 副屏），不能靠阈值过滤 —— 是否落在已连接显示器内由恢复侧的
    // 显示器交集校验负责（见 PhysicalRect / overlap_size）。
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

/// 保存时合并：几何字段为 None（最小化/最大化/全屏/隐藏窗口时捕获不到
/// 有意义的值）时保留上一次的好值；标志位总是更新。合并后再清洗一次，
/// 保证任何情况下落盘的窗口状态都不会是明显的垃圾值（如历史版本在
/// 最小化时落盘的 width=0.0）。
fn merge_window_state(mut prev: WindowState, incoming: WindowState) -> WindowState {
    if incoming.x.is_some() {
        prev.x = incoming.x;
    }
    if incoming.y.is_some() {
        prev.y = incoming.y;
    }
    if incoming.width.is_some() {
        prev.width = incoming.width;
    }
    if incoming.height.is_some() {
        prev.height = incoming.height;
    }
    if incoming.maximized.is_some() {
        prev.maximized = incoming.maximized;
    }
    if incoming.fullscreen.is_some() {
        prev.fullscreen = incoming.fullscreen;
    }
    sanitize_window_state(prev)
}

pub fn load_window_state(config_dir: &Path) -> WindowState {
    let ws = load_config(config_dir).window;
    sanitize_window_state(ws)
}

/// 将窗口状态写回配置文件（保留其他字段）。
pub fn save_window_state(config_dir: &Path, ws: &WindowState) {
    let mut cfg = load_config(config_dir);
    cfg.window = merge_window_state(cfg.window.clone(), ws.clone());
    save_config(config_dir, &cfg);
}

// ── 窗口恢复的显示器交集校验（纯函数，lib.rs 在启动时调用） ────────────────

/// 物理像素矩形（f64，便于交集/钳制运算）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicalRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// 恢复的窗口与显示器的可见重叠阈值：窗口至少要有这么大的区域落在某台
/// 已连接显示器内，否则视为"屏幕外"，回退主显示器内居中。取值只需保证
/// 用户能看到并拖到标题栏即可，不必覆盖大部分窗口。
pub const MIN_VISIBLE_OVERLAP_W: f64 = 120.0;
pub const MIN_VISIBLE_OVERLAP_H: f64 = 60.0;

/// 两矩形的相交宽高（不相交时为 (0, 0)）。
pub fn overlap_size(a: &PhysicalRect, b: &PhysicalRect) -> (f64, f64) {
    let w = (a.x + a.width).min(b.x + b.width) - a.x.max(b.x);
    let h = (a.y + a.height).min(b.y + b.height) - a.y.max(b.y);
    (w.max(0.0), h.max(0.0))
}

/// 把窗口尺寸钳制到显示器边界内（窗口不允许大于显示器）。
pub fn clamp_size_to_bounds(width: f64, height: f64, bounds: &PhysicalRect) -> (f64, f64) {
    (
        width.min(bounds.width).max(1.0),
        height.min(bounds.height).max(1.0),
    )
}

/// 在显示器内居中放置窗口；窗口不小于显示器时贴左上角。
/// 返回物理像素坐标（配合 `Position::Physical` 使用）。
pub fn centered_position(window: (f64, f64), bounds: &PhysicalRect) -> (i32, i32) {
    let x = bounds.x + ((bounds.width - window.0) / 2.0).max(0.0);
    let y = bounds.y + ((bounds.height - window.1) / 2.0).max(0.0);
    (x.round() as i32, y.round() as i32)
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

/// 从 config dir 读取录音设置。
pub fn load_recording_settings(config_dir: &Path) -> RecordingSettings {
    load_config(config_dir).recording.normalized()
}

/// 将录音设置写入 config dir；保留现有配置中的其他字段。
pub fn save_recording_settings(config_dir: &Path, settings: &RecordingSettings) {
    let mut cfg = load_config(config_dir);
    cfg.recording = settings.normalized();
    save_config(config_dir, &cfg);
}
