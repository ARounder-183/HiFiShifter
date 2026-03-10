use crate::audio_engine::AudioEngine;
use crate::audio_utils::try_read_wav_info;
use crate::clip_pitch_cache::ClipPitchCache;
use crate::models::{
    ModelConfig, ModelConfigPayload, PitchRange, ProjectMetaPayload, RuntimeInfoPayload,
    TimelineClip, TimelineStatePayload, TimelineTrack,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use uuid::Uuid;

// ─── 外部 Resampler 注册表 ──────────────────────────────────────────────────

/// 单个 flag 参数定义（如 B=气声, g=性别/共振峰, t=时值偏移）。
///
/// 每个 flag 参数会在前端参数面板中生成一个独立的 AutomationCurve 滑块/曲线，
/// 并在调用 resampler 时拼接进 flags 字符串。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlagParam {
    /// Flag 字母（如 "B"、"g"、"t"、"H"）。
    pub key: String,
    /// 显示名称（如 "气声 (Breathiness)"）。
    pub display_name: String,
    /// 参数最小值。
    pub min_value: f64,
    /// 参数最大值。
    pub max_value: f64,
    /// 默认值（即 orig 值，不编辑时使用此值）。
    pub default_value: f64,
}

/// 外部 Resampler 注册条目（持久化到用户配置）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResamplerEntry {
    /// 唯一 ID（UUID 或用户自定义短名）。
    pub id: String,
    /// 显示名称（如 "Moresampler"、"TIPS"、"tn_fnds"）。
    pub display_name: String,
    /// 可执行文件绝对路径。
    pub exe_path: PathBuf,
    /// 默认 flags 字符串（如 "B50Y0H0"），用于不在 flag_params 中的额外 flags。
    pub default_flags: String,
    /// 结构化的 flag 参数列表（用户可在前端通过 + 号添加/删除）。
    #[serde(default)]
    pub flag_params: Vec<FlagParam>,
    /// 是否可用（exe 存在且可执行）。
    pub available: bool,
}

/// 外部 Resampler 注册表（运行时存在 AppState 中，持久化到 config 目录）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResamplerRegistry {
    pub entries: HashMap<String, ResamplerEntry>,
}

impl ResamplerRegistry {
    /// 添加或更新一个 resampler 条目。
    pub fn register(&mut self, entry: ResamplerEntry) {
        self.entries.insert(entry.id.clone(), entry);
    }

    /// 移除指定 ID 的条目。返回是否存在并被移除。
    pub fn remove(&mut self, id: &str) -> bool {
        self.entries.remove(id).is_some()
    }

    /// 按 ID 查找条目。
    pub fn get(&self, id: &str) -> Option<&ResamplerEntry> {
        self.entries.get(id)
    }

    /// 刷新所有条目的可用性（检查 exe 文件是否存在）。
    pub fn refresh_availability(&mut self) {
        for entry in self.entries.values_mut() {
            entry.available = entry.exe_path.exists();
        }
    }

    /// 返回所有条目列表（按显示名称排序）。
    pub fn list(&self) -> Vec<&ResamplerEntry> {
        let mut entries: Vec<_> = self.entries.values().collect();
        entries.sort_by(|a, b| a.display_name.cmp(&b.display_name));
        entries
    }
}

fn default_frame_period_ms() -> f64 {
    5.0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PitchAnalysisAlgo {
    #[default]
    WorldDll,
    NsfHifiganOnnx,
    #[serde(rename = "vslib")]
    VocalShifterVslib,
    /// 外部 Resampler（ID = registry entry key）。
    ExternalResampler(String),
    None,
    #[serde(other)]
    Unknown,
}

/// 合成链路类型，独立于 PitchAnalysisAlgo，面向声码器选择。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SynthPipelineKind {
    WorldVocoder,
    NsfHifiganOnnx,
    /// VocalShifter vslib 原生声码器（仅限 Windows，需 vslib feature）。
    #[cfg(feature = "vslib")]
    VocalShifterVslib,
    /// 外部 UTAU Resampler（ID = registry entry key）。
    ExternalResampler(String),
}

impl SynthPipelineKind {
    /// 从 Track 的分析算法推断合成链路类型。
    pub fn from_track_algo(algo: &PitchAnalysisAlgo) -> Self {
        match algo {
            PitchAnalysisAlgo::NsfHifiganOnnx => Self::NsfHifiganOnnx,
            #[cfg(feature = "vslib")]
            PitchAnalysisAlgo::VocalShifterVslib => Self::VocalShifterVslib,
            PitchAnalysisAlgo::ExternalResampler(id) => Self::ExternalResampler(id.clone()),
            _ => Self::WorldVocoder,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrackParamsState {
    #[serde(default = "default_frame_period_ms")]
    pub frame_period_ms: f64,

    #[serde(default)]
    pub pitch_orig: Vec<f32>,
    #[serde(default)]
    pub pitch_edit: Vec<f32>,

    #[serde(default)]
    pub pitch_edit_user_modified: bool,

    #[serde(default)]
    pub tension_orig: Vec<f32>,
    #[serde(default)]
    pub tension_edit: Vec<f32>,

    #[serde(skip)]
    pub pitch_orig_key: Option<String>,

    /// 由 Reaper 导入产生的待应用音高偏移（半音）。
    /// 当 pitch_orig 分析完成后，pitch_edit = pitch_orig + 此偏移。
    #[serde(skip)]
    pub pending_pitch_offset: Option<Vec<f32>>,

    /// 声码器专属逐帧自动化曲线（key = ParamDescriptor::id）。
    /// 例："formant_shift_cents", "volume" 等；缺失 key = 使用参数默认值。
    #[serde(default)]
    pub extra_curves: HashMap<String, Vec<f32>>,

    /// 声码器专属静态参数（key = ParamDescriptor::id，值为枚举整数转 f64）。
    /// 例："synth_mode" = 1.0（SYNTHMODE_MF）。
    #[serde(default)]
    pub extra_params: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Track {
    pub id: String,
    pub name: String,
    pub parent_id: Option<String>,
    pub order: i32,
    pub muted: bool,
    pub solo: bool,
    pub volume: f32,

    #[serde(default)]
    pub compose_enabled: bool,

    #[serde(default)]
    pub pitch_analysis_algo: PitchAnalysisAlgo,

    /// 轨道主题色，hex 字符串，如 "#4f8ef7"
    #[serde(default)]
    pub color: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Clip {
    pub id: String,
    pub track_id: String,
    pub name: String,
    pub start_sec: f64,
    pub length_sec: f64,
    pub color: String,

    pub source_path: Option<String>,
    pub duration_sec: Option<f64>,           // 兼容性保留
    pub duration_frames: Option<u64>,        // 精确的frame总数
    pub source_sample_rate: Option<u32>,     // 源文件采样率
    pub waveform_preview: Option<Vec<f32>>,
    pub pitch_range: Option<PitchRange>,

    pub gain: f32,
    pub muted: bool,
    #[serde(alias = "trim_start_sec")]
    pub source_start_sec: f64,
    #[serde(alias = "trim_end_sec")]
    pub source_end_sec: f64,
    pub playback_rate: f32,
    pub fade_in_sec: f64,
    pub fade_out_sec: f64,
    /// 淡入曲线类型（linear/sine/exponential/logarithmic/scurve），默认 sine
    #[serde(default = "default_fade_curve")]
    pub fade_in_curve: String,
    /// 淡出曲线类型（linear/sine/exponential/logarithmic/scurve），默认 sine
    #[serde(default = "default_fade_curve")]
    pub fade_out_curve: String,

    /// Clip 级别的声码器曲线覆盖（None = 使用 Track 级别的 extra_curves）。
    #[serde(default)]
    pub extra_curves: Option<HashMap<String, Vec<f32>>>,

    /// Clip 级别的声码器静态参数覆盖（None = 使用 Track 级别的 extra_params）。
    #[serde(default)]
    pub extra_params: Option<HashMap<String, f64>>,
}

#[derive(Debug, Clone, Default)]
pub struct ClipStatePatch {
    pub name: Option<String>,
    pub start_sec: Option<f64>,
    pub length_sec: Option<f64>,
    pub gain: Option<f32>,
    pub muted: Option<bool>,
    pub source_start_sec: Option<f64>,
    pub source_end_sec: Option<f64>,
    pub playback_rate: Option<f32>,
    pub fade_in_sec: Option<f64>,
    pub fade_out_sec: Option<f64>,
    pub fade_in_curve: Option<String>,
    pub fade_out_curve: Option<String>,
    pub color: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct RuntimeState {
    pub device: String,
    pub model_loaded: bool,
    pub audio_loaded: bool,
    pub has_synthesized: bool,

    pub synthesized_wav_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineState {
    pub tracks: Vec<Track>,
    pub clips: Vec<Clip>,
    pub selected_track_id: Option<String>,
    pub selected_clip_id: Option<String>,
    pub bpm: f64,
    pub playhead_sec: f64,
    pub project_sec: f64,

    #[serde(default)]
    pub params_by_root_track: BTreeMap<String, TrackParamsState>,

    pub next_track_order: i32,
}

#[derive(Debug, Clone, Default)]
pub struct TimelineHistory {
    pub undo: Vec<TimelineState>,
    pub redo: Vec<TimelineState>,
}

#[derive(Debug, Clone)]
pub struct ProjectState {
    pub name: String,
    pub path: Option<String>,
    pub dirty: bool,
    pub recent: Vec<String>,
    pub allow_close: bool,
}

impl Default for ProjectState {
    fn default() -> Self {
        Self {
            name: "Untitled".to_string(),
            path: None,
            dirty: false,
            recent: Vec::new(),
            allow_close: false,
        }
    }
}

impl Default for TimelineState {
    fn default() -> Self {
        let track_id = "track_main".to_string();
        Self {
            tracks: vec![Track {
                id: track_id.clone(),
                name: "Main".to_string(),
                parent_id: None,
                order: 0,
                muted: false,
                solo: false,
                volume: 0.9,

                compose_enabled: false,
                pitch_analysis_algo: PitchAnalysisAlgo::default(),
                color: String::new(),
            }],
            clips: vec![],
            selected_track_id: Some(track_id),
            selected_clip_id: None,
            bpm: 120.0,
            playhead_sec: 0.0,
            project_sec: 32.0, // 64 beats @ 120 BPM = 32 sec

            params_by_root_track: BTreeMap::new(),
            next_track_order: 1,
        }
    }
}

impl TimelineState {
    pub fn resolve_root_track_id(&self, track_id: &str) -> Option<String> {
        if track_id.trim().is_empty() {
            return None;
        }
        let mut cur = track_id.to_string();
        let mut safety = 0;
        loop {
            let parent = self
                .tracks
                .iter()
                .find(|t| t.id == cur)
                .and_then(|t| t.parent_id.clone());
            match parent {
                Some(p) if !p.trim().is_empty() => {
                    cur = p;
                }
                _ => return Some(cur),
            }
            safety += 1;
            if safety > 2048 {
                return Some(cur);
            }
        }
    }

    pub fn frame_period_ms(&self) -> f64 {
        default_frame_period_ms()
    }

    pub fn project_duration_sec(&self) -> f64 {
        self.project_sec.max(0.0)
    }

    pub fn target_param_frames(&self, frame_period_ms: f64) -> usize {
        let fp = frame_period_ms.max(0.1);
        let sec = self.project_duration_sec();
        let frames = (sec * 1000.0 / fp).ceil();
        if !(frames.is_finite() && frames > 0.0) {
            return 1;
        }
        (frames as usize).max(1)
    }

    pub fn ensure_params_for_root(&mut self, root_track_id: &str) {
        let fp = self.frame_period_ms();
        let target = self.target_param_frames(fp);
        
        // Calculate expected cache key to detect when timeline changed
        let expected_key = crate::pitch_analysis::build_root_pitch_key(self, root_track_id);
        
        let entry = self
            .params_by_root_track
            .entry(root_track_id.to_string())
            .or_insert_with(|| TrackParamsState {
                frame_period_ms: fp,
                ..TrackParamsState::default()
            });

        entry.frame_period_ms = fp;
        
        // CRITICAL FIX: Detect stale pitch curves and clear them when clip/timeline changes.
        // This prevents old pitch data from being displayed after clip replacement or timeline edits.
        let key_changed = entry.pitch_orig_key.as_deref() != Some(&expected_key);
        
        if key_changed && entry.pitch_orig_key.is_some() {
            // Timeline/clip configuration changed - clear orig curves to force re-analysis
            entry.pitch_orig.clear();
            entry.pitch_orig_key = None;
            // 仅当用户未手动编辑时才清空 pitch_edit，保护用户的编辑成果
            if !entry.pitch_edit_user_modified {
                entry.pitch_edit.clear();
            }
            
            if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
                eprintln!(
                    "state: [INVALIDATE] Cleared stale pitch curves for root_track={} (key changed, user_modified={})",
                    root_track_id, entry.pitch_edit_user_modified
                );
            }
        }

        #[allow(clippy::ptr_arg)]
        fn resize_curve(v: &mut Vec<f32>, target: usize, fill: f32) {
            if v.len() < target {
                v.extend(std::iter::repeat_n(fill, target - v.len()));
            } else if v.len() > target {
                v.truncate(target);
            }
        }

        resize_curve(&mut entry.pitch_orig, target, 0.0);
        resize_curve(&mut entry.pitch_edit, target, 0.0);
        resize_curve(&mut entry.tension_orig, target, 0.0);
        resize_curve(&mut entry.tension_edit, target, 0.0);

        // Backward compatibility: older projects didn't have `pitch_edit_user_modified`.
        // Infer it if we detect a meaningful difference between edit and orig.
        if !entry.pitch_edit_user_modified {
            let len = entry.pitch_orig.len().min(entry.pitch_edit.len());
            let mut i = 0usize;
            let stride = 1usize; // keep it simple; curves are not huge.
            while i < len {
                let o = entry.pitch_orig[i];
                let e = entry.pitch_edit[i];
                if e.is_finite() && e > 0.0 {
                    if !(o.is_finite() && o > 0.0) {
                        entry.pitch_edit_user_modified = true;
                        break;
                    }
                    if (e - o).abs() > 1e-3 {
                        entry.pitch_edit_user_modified = true;
                        break;
                    }
                }
                i += stride;
            }
        }
    }
}

/// Timeline snapshot for incremental pitch refresh
///
/// Stores a snapshot of the timeline state at the time of last pitch analysis
/// to enable detection of which clips have changed and need re-analysis.
#[derive(Debug, Clone)]
pub struct TimelineSnapshot {
    /// Mapping from clip ID to cache key
    pub clips: HashMap<String, String>,
    /// BPM at the time of analysis
    pub bpm: f64,
    /// Frame period used for analysis
    pub frame_period_ms: f64,
}

pub struct AppState {
    pub timeline: std::sync::Mutex<TimelineState>,
    pub timeline_history: std::sync::Mutex<TimelineHistory>,
    pub project: std::sync::Mutex<ProjectState>,
    pub runtime: std::sync::Mutex<RuntimeState>,

    /// Current UI locale reported by the frontend (e.g. "en-US", "zh-CN").
    /// Used to localize native dialogs implemented in Rust.
    pub ui_locale: RwLock<String>,

    pub waveform_cache_dir: std::sync::Mutex<PathBuf>,
    pub waveform_cache: std::sync::Mutex<
        std::collections::HashMap<String, std::sync::Arc<crate::waveform::CachedPeaks>>,
    >,

    // Set in Tauri setup. Used for async notifications.
    pub app_handle: OnceLock<tauri::AppHandle>,

    // De-dup background pitch analysis jobs (keyed by rootTrackId + analysis key).
    pub pitch_inflight: std::sync::Mutex<std::collections::HashSet<String>>,
    
    // Current pitch analysis progress (for polling from frontend)
    pub pitch_analysis_progress: std::sync::RwLock<Option<crate::pitch_analysis::PitchOrigAnalysisProgressEvent>>,

    // Clip-level pitch analysis cache for performance optimization
    pub clip_pitch_cache: Arc<Mutex<ClipPitchCache>>,
    
    // Timeline snapshot for incremental pitch refresh (keyed by root_track_id)
    pub pitch_timeline_snapshot: Mutex<HashMap<String, TimelineSnapshot>>,

    pub audio_engine: AudioEngine,

    /// App config directory for persisting recent projects etc.
    pub config_dir: OnceLock<std::path::PathBuf>,

    /// 外部 Resampler 注册表（持久化到 config 目录）。
    pub resampler_registry: Mutex<ResamplerRegistry>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            timeline: std::sync::Mutex::new(TimelineState::default()),
            timeline_history: std::sync::Mutex::new(TimelineHistory::default()),
            project: std::sync::Mutex::new(ProjectState::default()),
            runtime: std::sync::Mutex::new(RuntimeState {
                device: "tauri".to_string(),
                synthesized_wav_path: None,
                ..RuntimeState::default()
            }),

            ui_locale: RwLock::new("en-US".to_string()),

            waveform_cache_dir: std::sync::Mutex::new(
                crate::waveform_disk_cache::default_cache_dir(),
            ),
            waveform_cache: std::sync::Mutex::new(std::collections::HashMap::new()),

            app_handle: OnceLock::new(),
            pitch_inflight: std::sync::Mutex::new(std::collections::HashSet::new()),
            pitch_analysis_progress: std::sync::RwLock::new(None),
            clip_pitch_cache: Arc::new(Mutex::new(ClipPitchCache::new(100))),
            pitch_timeline_snapshot: Mutex::new(HashMap::new()),

            audio_engine: AudioEngine::new(),
            config_dir: OnceLock::new(),
            resampler_registry: Mutex::new(ResamplerRegistry::default()),
        }
    }
}

impl AppState {
    pub fn get_or_compute_waveform_peaks(
        &self,
        source_path: &str,
        hop: usize,
    ) -> Result<std::sync::Arc<crate::waveform::CachedPeaks>, String> {
        if source_path.trim().is_empty() {
            return Err("empty source_path".to_string());
        }

        let cache_key = format!("{}|{}", source_path, hop);

        {
            let cache = self
                .waveform_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(found) = cache.get(&cache_key) {
                return Ok(found.clone());
            }
        }

        // Disk cache (best-effort): if present, load and populate the in-memory cache.
        let cache_dir = {
            self.waveform_cache_dir
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
        };
        let disk_path = crate::waveform_disk_cache::cache_file_path(&cache_dir, source_path, hop);
        if let Some(found) = crate::waveform_disk_cache::try_load_peaks(&disk_path) {
            if found.hop == hop {
                let found = std::sync::Arc::new(found);
                let mut cache = self
                    .waveform_cache
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                cache.insert(cache_key.clone(), found.clone());
                return Ok(found);
            }
        }

        let peaks = crate::waveform::CachedPeaks::compute(std::path::Path::new(source_path), hop)?;

        // Save to disk cache (best-effort; ignore failures).
        let _ = crate::waveform_disk_cache::save_peaks(&disk_path, &peaks);

        let peaks = std::sync::Arc::new(peaks);
        let mut cache = self
            .waveform_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        cache.insert(cache_key, peaks.clone());
        Ok(peaks)
    }

    pub fn clear_waveform_cache(&self) -> crate::waveform_disk_cache::ClearStats {
        {
            let mut cache = self
                .waveform_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.clear();
        }

        let cache_dir = {
            self.waveform_cache_dir
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
        };
        crate::waveform_disk_cache::clear_dir(&cache_dir)
    }

    pub fn project_meta_payload(&self) -> ProjectMetaPayload {
        let p = self
            .project
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        ProjectMetaPayload {
            name: p.name,
            path: p.path,
            dirty: p.dirty,
            recent: p.recent,
        }
    }

    pub fn checkpoint_timeline(&self, snapshot: &TimelineState) {
        let mut h = self
            .timeline_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        h.undo.push(snapshot.clone());
        if h.undo.len() > 100 {
            h.undo.remove(0);
        }
        h.redo.clear();
        drop(h);

        let (name, was_clean) = {
            let mut p = self.project.lock().unwrap_or_else(|e| e.into_inner());
            let was_clean = !p.dirty;
            p.dirty = true;
            (p.name.clone(), was_clean)
        };

        // 仅在首次变脏时更新窗口标题（添加 * 号）
        if was_clean {
            if let Some(handle) = self.app_handle.get() {
                use tauri::Manager;
                if let Some(win) = handle.get_webview_window("main") {
                    let title = format!("HiFiShifter - {}*", name);
                    let _ = win.set_title(&title);
                }
            }
        }
    }

    pub fn clear_history(&self) {
        let mut h = self
            .timeline_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        h.undo.clear();
        h.redo.clear();
    }

    pub fn undo_timeline(&self) -> TimelineStatePayload {
        let mut tl = self.timeline.lock().unwrap_or_else(|e| e.into_inner());
        let mut h = self
            .timeline_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(prev) = h.undo.pop() else {
            let mut payload = tl.to_payload();
            payload.project = Some(self.project_meta_payload());
            return payload;
        };
        h.redo.push(tl.clone());
        *tl = prev;
        drop(h);
        self.audio_engine.update_timeline(tl.clone());
        let mut payload = tl.to_payload();
        payload.project = Some(self.project_meta_payload());
        payload
    }

    pub fn redo_timeline(&self) -> TimelineStatePayload {
        let mut tl = self.timeline.lock().unwrap_or_else(|e| e.into_inner());
        let mut h = self
            .timeline_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(next) = h.redo.pop() else {
            let mut payload = tl.to_payload();
            payload.project = Some(self.project_meta_payload());
            return payload;
        };
        h.undo.push(tl.clone());
        *tl = next;
        drop(h);
        self.audio_engine.update_timeline(tl.clone());
        let mut payload = tl.to_payload();
        payload.project = Some(self.project_meta_payload());
        payload
    }
}

fn new_id(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::new_v4().simple())
}

fn default_clip_color() -> String {
    "emerald".to_string()
}

fn default_fade_curve() -> String {
    "sine".to_string()
}

impl TimelineState {
    fn ensure_project_end_sec(&mut self, end_sec: f64) {
        if !(end_sec.is_finite()) {
            return;
        }
        // Only extend; never shrink automatically.
        // Use ceil so the ruler/grid has room for the full clip.
        let target = end_sec.max(4.0).ceil();
        if target > self.project_sec {
            self.project_sec = target;
        }
    }

    pub fn to_payload(&self) -> TimelineStatePayload {
        let tracks_payload = build_track_payload(&self.tracks);
        let clips_payload = self
            .clips
            .iter()
            .map(|c| TimelineClip {
                id: c.id.clone(),
                track_id: c.track_id.clone(),
                name: c.name.clone(),
                start_sec: c.start_sec,
                length_sec: c.length_sec,
                color: c.color.clone(),
                source_path: c.source_path.clone(),
                duration_sec: c.duration_sec,
                duration_frames: c.duration_frames,
                source_sample_rate: c.source_sample_rate,
                waveform_preview: c.waveform_preview.clone(),
                pitch_range: c.pitch_range.clone(),
                gain: Some(c.gain),
                muted: Some(c.muted),
                source_start_sec: Some(c.source_start_sec),
                source_end_sec: Some(c.source_end_sec),
                playback_rate: Some(c.playback_rate),
                fade_in_sec: Some(c.fade_in_sec),
                fade_out_sec: Some(c.fade_out_sec),
                fade_in_curve: Some(c.fade_in_curve.clone()),
                fade_out_curve: Some(c.fade_out_curve.clone()),
            })
            .collect::<Vec<_>>();

        TimelineStatePayload {
            ok: true,
            tracks: tracks_payload,
            clips: clips_payload,
            selected_track_id: self.selected_track_id.clone(),
            selected_clip_id: self.selected_clip_id.clone(),
            bpm: self.bpm,
            playhead_sec: self.playhead_sec,
            project_sec: Some(self.project_sec),
            project: None,
        }
    }

    pub fn add_track(
        &mut self,
        name: Option<String>,
        parent_track_id: Option<String>,
        index: Option<usize>,
    ) -> String {
        let id = new_id("track");
        let order = self.next_track_order;
        self.next_track_order += 1;

        // 预设轨道颜色调色板，循环分配
        const TRACK_COLORS: &[&str] = &[
            "#4f8ef7", // 蓝
            "#a78bfa", // 紫
            "#34d399", // 绿
            "#fb923c", // 橙
            "#f472b6", // 粉
            "#38bdf8", // 天蓝
            "#facc15", // 黄
            "#f87171", // 红
        ];
        let color_index = self.tracks.len() % TRACK_COLORS.len();
        let color = TRACK_COLORS[color_index].to_string();

        let track = Track {
            id: id.clone(),
            name: name.unwrap_or_else(|| "Track".to_string()),
            parent_id: parent_track_id,
            order,
            muted: false,
            solo: false,
            volume: 0.9,

            compose_enabled: false,
            pitch_analysis_algo: PitchAnalysisAlgo::default(),
            color,
        };
        self.tracks.push(track);

        // Best-effort insert ordering: we encode ordering using `order`, but for now
        // we accept `index` by nudging orders for the same parent.
        if let Some(i) = index {
            self.reorder_siblings(&id, i);
        }

        self.selected_track_id = Some(id.clone());
        id
    }

    fn reorder_siblings(&mut self, track_id: &str, target_index: usize) {
        let parent_id = self
            .tracks
            .iter()
            .find(|t| t.id == track_id)
            .and_then(|t| t.parent_id.clone());
        let mut siblings: Vec<_> = self
            .tracks
            .iter()
            .filter(|t| t.parent_id == parent_id && t.id != track_id)
            .cloned()
            .collect();
        siblings.sort_by_key(|t| t.order);
        let target_index = target_index.min(siblings.len());

        // Pull this track out and rebuild orders.
        let mut rebuilt: Vec<String> = siblings.into_iter().map(|t| t.id).collect();
        rebuilt.insert(target_index, track_id.to_string());

        for (i, tid) in rebuilt.iter().enumerate() {
            if let Some(t) = self.tracks.iter_mut().find(|t| &t.id == tid) {
                t.order = i as i32;
            }
        }
        self.next_track_order = rebuilt.len() as i32 + 1;
    }

    pub fn remove_track(&mut self, track_id: &str) {
        // Remove clips first.
        self.clips.retain(|c| c.track_id != track_id);

        // Remove descendants.
        let mut to_remove = vec![track_id.to_string()];
        let mut idx = 0;
        while idx < to_remove.len() {
            let cur = to_remove[idx].clone();
            for child in self
                .tracks
                .iter()
                .filter(|t| t.parent_id.as_deref() == Some(cur.as_str()))
                .map(|t| t.id.clone())
                .collect::<Vec<_>>()
            {
                to_remove.push(child);
            }
            idx += 1;
        }
        self.tracks.retain(|t| !to_remove.contains(&t.id));

        if self.selected_track_id.as_deref() == Some(track_id) {
            self.selected_track_id = self.tracks.first().map(|t| t.id.clone());
        }
        if let Some(cid) = self.selected_clip_id.clone() {
            if !self.clips.iter().any(|c| c.id == cid) {
                self.selected_clip_id = None;
            }
        }
    }

    pub fn move_track(
        &mut self,
        track_id: &str,
        target_index: usize,
        parent_track_id: Option<String>,
    ) {
        if let Some(t) = self.tracks.iter_mut().find(|t| t.id == track_id) {
            t.parent_id = parent_track_id;
        }
        self.reorder_siblings(track_id, target_index);
    }

    pub fn set_track_state(
        &mut self,
        track_id: &str,
        muted: Option<bool>,
        solo: Option<bool>,
        volume: Option<f32>,
        compose_enabled: Option<bool>,
        pitch_analysis_algo: Option<PitchAnalysisAlgo>,
        color: Option<String>,
    ) {
        if let Some(t) = self.tracks.iter_mut().find(|t| t.id == track_id) {
            if let Some(v) = muted {
                t.muted = v;
            }
            if let Some(v) = solo {
                t.solo = v;
            }
            if let Some(v) = volume {
                t.volume = v.clamp(0.0, 1.0);
            }

            if let Some(v) = compose_enabled {
                t.compose_enabled = v;
            }
            if let Some(v) = pitch_analysis_algo {
                t.pitch_analysis_algo = v;
            }
            if let Some(v) = color {
                t.color = v;
            }
        }
    }

    pub fn select_track(&mut self, track_id: &str) {
        if self.tracks.iter().any(|t| t.id == track_id) {
            self.selected_track_id = Some(track_id.to_string());
        }
    }

    pub fn set_project_length(&mut self, project_sec: f64) {
        if project_sec.is_finite() {
            self.project_sec = project_sec.max(4.0);
        }
    }

    pub fn add_clip(
        &mut self,
        track_id: Option<String>,
        name: Option<String>,
        start_sec: Option<f64>,
        length_sec: Option<f64>,
        source_path: Option<String>,
    ) -> String {
        let track_id = track_id
            .or_else(|| self.selected_track_id.clone())
            .or_else(|| self.tracks.first().map(|t| t.id.clone()))
            .unwrap_or_else(|| self.add_track(Some("Main".to_string()), None, None));

        if !self.tracks.iter().any(|t| t.id == track_id) {
            // Create missing track.
            self.tracks.push(Track {
                id: track_id.clone(),
                name: "Track".to_string(),
                parent_id: None,
                order: self.next_track_order,
                muted: false,
                solo: false,
                volume: 0.9,

                compose_enabled: false,
                pitch_analysis_algo: PitchAnalysisAlgo::default(),
                color: String::new(),
            });
            self.next_track_order += 1;
        }

        // If this is a new clip referencing an existing audio source, inherit cached metadata
        // (duration + waveform preview) from any existing clip that already has it.
        let inherited = source_path.as_deref().and_then(|sp| {
            self.clips
                .iter()
                .find(|c| c.source_path.as_deref() == Some(sp) && c.waveform_preview.is_some())
                .map(|c| {
                    (
                        c.duration_sec,
                        c.duration_frames,
                        c.source_sample_rate,
                        c.waveform_preview.clone(),
                        c.pitch_range.clone(),
                    )
                })
        });

        let id = new_id("clip");
        let ss = start_sec.unwrap_or(self.playhead_sec).max(0.0);
        let ls = length_sec.unwrap_or(4.0).max(0.01);
        self.ensure_project_end_sec(ss + ls);
        let clip = Clip {
            id: id.clone(),
            track_id: track_id.clone(),
            name: name.unwrap_or_else(|| "Clip".to_string()),
            start_sec: ss,
            length_sec: ls,
            color: default_clip_color(),
            source_path,
            duration_sec: inherited.as_ref().and_then(|v| v.0),
            duration_frames: inherited.as_ref().and_then(|v| v.1),
            source_sample_rate: inherited.as_ref().and_then(|v| v.2),
            waveform_preview: inherited.as_ref().and_then(|v| v.3.clone()),
            pitch_range: inherited
                .as_ref()
                .and_then(|v| v.4.clone())
                .or(Some(PitchRange {
                    min: -24.0,
                    max: 24.0,
                })),
            gain: 1.0,
            muted: false,
            source_start_sec: 0.0,
            source_end_sec: inherited.as_ref().and_then(|v| v.0).unwrap_or(ls),
            playback_rate: 1.0,
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,
            fade_in_curve: default_fade_curve(),
            fade_out_curve: default_fade_curve(),
            extra_curves: None,
            extra_params: None,
        };
        self.clips.push(clip);
        self.selected_clip_id = Some(id.clone());
        self.playhead_sec = ss;
        id
    }

    pub fn remove_clip(&mut self, clip_id: &str) {
        self.clips.retain(|c| c.id != clip_id);
        if self.selected_clip_id.as_deref() == Some(clip_id) {
            self.selected_clip_id = None;
        }
    }

    pub fn move_clip(&mut self, clip_id: &str, start_sec: f64, track_id: Option<String>) {
        let mut end_sec: Option<f64> = None;
        if let Some(c) = self.clips.iter_mut().find(|c| c.id == clip_id) {
            c.start_sec = start_sec.max(0.0);
            if let Some(tid) = track_id {
                if self.tracks.iter().any(|t| t.id == tid) {
                    c.track_id = tid;
                }
            }
            end_sec = Some(c.start_sec + c.length_sec);
        }
        if let Some(v) = end_sec {
            self.ensure_project_end_sec(v);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_clip_state(
        &mut self,
        clip_id: &str,
        length_sec: Option<f64>,
        gain: Option<f32>,
        muted: Option<bool>,
        source_start_sec: Option<f64>,
        source_end_sec: Option<f64>,
        playback_rate: Option<f32>,
        fade_in_sec: Option<f64>,
        fade_out_sec: Option<f64>,
    ) {
        self.patch_clip_state(
            clip_id,
            ClipStatePatch {
                name: None,
                start_sec: None,
                length_sec,
                gain,
                muted,
                source_start_sec,
                source_end_sec,
                playback_rate,
                fade_in_sec,
                fade_out_sec,
                fade_in_curve: None,
                fade_out_curve: None,
                color: None,
            },
        );
    }

    pub fn patch_clip_state(&mut self, clip_id: &str, patch: ClipStatePatch) {
        let mut end_sec: Option<f64> = None;
        if let Some(c) = self.clips.iter_mut().find(|c| c.id == clip_id) {
            if let Some(v) = patch.name {
                c.name = v;
            }
            if let Some(v) = patch.start_sec {
                c.start_sec = v.max(0.0);
            }
            if let Some(v) = patch.length_sec {
                c.length_sec = v.max(0.0);
            }
            if let Some(v) = patch.gain {
                c.gain = v.clamp(0.0, 2.0);
            }
            if let Some(v) = patch.muted {
                c.muted = v;
            }
            if let Some(v) = patch.source_start_sec {
                if v.is_finite() {
                    // Negative values are allowed (slip-edit past the source start -> leading silence).
                    // Keep a reasonable bound to avoid accidental extreme values.
                    c.source_start_sec = v.clamp(-1_000_000.0, 1_000_000.0);
                }
            }
            if let Some(v) = patch.source_end_sec {
                c.source_end_sec = v.max(0.0);
            }
            if let Some(v) = patch.playback_rate {
                c.playback_rate = v.clamp(0.1, 10.0);
            }
            if let Some(v) = patch.fade_in_sec {
                c.fade_in_sec = v.max(0.0);
            }
            if let Some(v) = patch.fade_out_sec {
                c.fade_out_sec = v.max(0.0);
            }
            if let Some(v) = patch.fade_in_curve {
                c.fade_in_curve = v;
            }
            if let Some(v) = patch.fade_out_curve {
                c.fade_out_curve = v;
            }
            if let Some(v) = patch.color {
                c.color = v;
            }

            end_sec = Some(c.start_sec + c.length_sec);
        }

        if let Some(v) = end_sec {
            self.ensure_project_end_sec(v);
        }
    }

    pub fn split_clip(&mut self, clip_id: &str, split_sec: f64) {
        let Some(idx) = self.clips.iter().position(|c| c.id == clip_id) else {
            return;
        };
        let clip = self.clips[idx].clone();
        let start = clip.start_sec;
        let end = clip.start_sec + clip.length_sec;
        let split = split_sec.clamp(start, end);
        if split <= start + 1e-6 || split >= end - 1e-6 {
            return;
        }

        self.ensure_project_end_sec(end);

        let left_len = split - start;
        let right_len = end - split;

        // 计算左 clip 的 playback_rate，用于更新 source_end_sec
        let left_rate = {
            let r = self.clips[idx].playback_rate as f64;
            if r.is_finite() && r > 0.0 { r } else { 1.0 }
        };

        self.clips[idx].length_sec = left_len;
        // 更新左 clip 的 source_end_sec: 切分点对应的源时间
        {
            let orig_src_end = self.clips[idx].source_end_sec;
            let new_src_end = self.clips[idx].source_start_sec + left_len * left_rate;
            self.clips[idx].source_end_sec =
                new_src_end.clamp(self.clips[idx].source_start_sec, orig_src_end);
        }
        // Fade semantics on split:
        // - fade-in is anchored to the original start, so only the left clip should keep it.
        // - fade-out is anchored to the original end, so only the right clip should keep it.
        // Clamp fades to the new clip lengths.
        self.clips[idx].fade_in_sec = self.clips[idx].fade_in_sec.min(left_len.max(0.0));
        self.clips[idx].fade_out_sec = 0.0;

        let mut right = clip;
        right.id = new_id("clip");
        right.start_sec = split;
        right.length_sec = right_len;
        right.fade_in_sec = 0.0;
        right.fade_out_sec = right.fade_out_sec.min(right_len.max(0.0));

        // Preserve the original audio offset: the right clip should continue from where the left ended.
        // trim_* are in sec (source time), while playback_rate scales source progress per timeline time.
        let rate = right.playback_rate as f64;
        let rate = if rate.is_finite() && rate > 0.0 {
            rate
        } else {
            1.0
        };
        if right.source_start_sec.is_finite() {
            right.source_start_sec =
                (right.source_start_sec + left_len * rate).clamp(-1_000_000.0, 1_000_000.0);
        }
        self.clips.push(right);
    }

    pub fn glue_clips(&mut self, clip_ids: &[String]) {
        if clip_ids.len() < 2 {
            return;
        }
        let mut selected: Vec<Clip> = self
            .clips
            .iter()
            .filter(|c| clip_ids.contains(&c.id))
            .cloned()
            .collect();
        if selected.len() < 2 {
            return;
        }
        let track_id = selected[0].track_id.clone();
        if selected.iter().any(|c| c.track_id != track_id) {
            return;
        }
        selected.sort_by(|a, b| a.start_sec.total_cmp(&b.start_sec));
        let Some(first) = selected.first() else {
            return;
        };
        let start = first.start_sec;
        let end = selected
            .iter()
            .map(|c| c.start_sec + c.length_sec)
            .fold(start, f64::max);

        self.ensure_project_end_sec(end);

        let mut glued = first.clone();
        glued.id = new_id("clip");
        glued.name = "Glued".to_string();
        glued.start_sec = start;
        glued.length_sec = (end - start).max(0.01);

        self.clips.retain(|c| !clip_ids.contains(&c.id));
        self.clips.push(glued.clone());
        self.selected_clip_id = Some(glued.id);
    }

    pub fn select_clip(&mut self, clip_id: Option<String>) {
        match clip_id {
            None => self.selected_clip_id = None,
            Some(id) => {
                if self.clips.iter().any(|c| c.id == id) {
                    self.selected_clip_id = Some(id);
                }
            }
        }
    }

    pub fn import_audio_item(
        &mut self,
        audio_path: &str,
        track_id: Option<String>,
        start_sec: Option<f64>,
    ) {
        let name = Path::new(audio_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("Audio")
            .to_string();

        let mut duration_sec: Option<f64> = None;
        let mut duration_frames: Option<u64> = None;
        let mut source_sample_rate: Option<u32> = None;
        let mut waveform_preview: Option<Vec<f32>> = None;

        match try_read_wav_info(Path::new(audio_path), 4096) {
            Some(info) => {
                if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
                    let mut max_amp = 0.0f32;
                    for &v in info.waveform_preview.iter() {
                        if v.is_finite() {
                            max_amp = max_amp.max(v.abs());
                        }
                    }
                    let head: Vec<String> = info
                        .waveform_preview
                        .iter()
                        .take(8)
                        .map(|v| format!("{:.4}", v))
                        .collect();
                    eprintln!(
                        "import_audio_item: audio_info ok: total_frames={}, sample_rate={}, duration_sec={:.6}, preview_len={}, preview_max={:.4}, preview_head=[{}]",
                        info.total_frames,
                        info.sample_rate,
                        info.duration_sec,
                        info.waveform_preview.len(),
                        max_amp,
                        head.join(", ")
                    );
                }
                duration_sec = Some(info.duration_sec);
                duration_frames = Some(info.total_frames);
                source_sample_rate = Some(info.sample_rate);
                waveform_preview = Some(info.waveform_preview);
            }
            None => {
                if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
                    let exists = Path::new(audio_path).exists();
                    eprintln!(
                        "import_audio_item: audio_info FAILED: path_exists={} path={}",
                        exists, audio_path
                    );
                }
            }
        }

        // 使用精确的frame计算length_sec（直接用秒，不依赖BPM）
        let computed_length_sec = if let (Some(frames), Some(sr)) = (duration_frames, source_sample_rate) {
            frames as f64 / sr as f64
        } else {
            duration_sec.unwrap_or(4.0)
        };

        let clip_id = self.add_clip(
            track_id,
            Some(name),
            start_sec,
            Some(computed_length_sec),
            Some(audio_path.to_string()),
        );

        // DEBUG: 打印导入clip时的关键参数
        if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
            eprintln!(
                "import_audio_item: clip created: clip_id={}, duration_frames={:?}, sample_rate={:?}, computed_length_sec={:.6}",
                &clip_id[..8.min(clip_id.len())],
                duration_frames,
                source_sample_rate,
                computed_length_sec
            );
        }

        if let Some(c) = self.clips.iter_mut().find(|c| c.id == clip_id) {
            c.duration_sec = duration_sec;
            c.duration_frames = duration_frames;
            c.source_sample_rate = source_sample_rate;
            c.waveform_preview = waveform_preview;
        }
    }
}

fn build_track_payload(tracks: &[Track]) -> Vec<TimelineTrack> {
    // Group by parent and keep stable ordering by `order`.
    let mut by_parent: HashMap<Option<String>, Vec<Track>> = HashMap::new();
    for t in tracks.iter().cloned() {
        by_parent.entry(t.parent_id.clone()).or_default().push(t);
    }
    for v in by_parent.values_mut() {
        v.sort_by_key(|t| t.order);
    }

    // Roots in order.
    let roots = by_parent.get(&None).cloned().unwrap_or_else(Vec::new);

    let mut out: Vec<TimelineTrack> = Vec::with_capacity(tracks.len());

    fn dfs(
        t: &Track,
        depth: u32,
        by_parent: &HashMap<Option<String>, Vec<Track>>,
        out: &mut Vec<TimelineTrack>,
    ) {
        fn algo_name(a: &PitchAnalysisAlgo) -> String {
            match a {
                PitchAnalysisAlgo::WorldDll => "world_dll".to_string(),
                PitchAnalysisAlgo::NsfHifiganOnnx => "nsf_hifigan_onnx".to_string(),
                PitchAnalysisAlgo::VocalShifterVslib => "vslib".to_string(),
                PitchAnalysisAlgo::ExternalResampler(id) => format!("external_resampler:{}", id),
                PitchAnalysisAlgo::None => "none".to_string(),
                PitchAnalysisAlgo::Unknown => "unknown".to_string(),
            }
        }

        let children = by_parent
            .get(&Some(t.id.clone()))
            .cloned()
            .unwrap_or_else(Vec::new);
        let child_ids = children.iter().map(|c| c.id.clone()).collect::<Vec<_>>();

        out.push(TimelineTrack {
            id: t.id.clone(),
            name: t.name.clone(),
            parent_id: t.parent_id.clone(),
            depth: Some(depth),
            child_track_ids: Some(child_ids),
            muted: t.muted,
            solo: t.solo,
            volume: t.volume,
            compose_enabled: t.compose_enabled,
            pitch_analysis_algo: algo_name(&t.pitch_analysis_algo),
            color: t.color.clone(),
        });

        for c in children {
            dfs(&c, depth + 1, by_parent, out);
        }
    }

    for r in roots {
        dfs(&r, 0, &by_parent, &mut out);
    }

    // Any orphans (missing parent) appended.
    if out.len() != tracks.len() {
        let mut seen: BTreeMap<String, bool> = BTreeMap::new();
        for t in &out {
            seen.insert(t.id.clone(), true);
        }
        for t in tracks {
            if !seen.contains_key(&t.id) {
                out.push(TimelineTrack {
                    id: t.id.clone(),
                    name: t.name.clone(),
                    parent_id: t.parent_id.clone(),
                    depth: Some(0),
                    child_track_ids: Some(vec![]),
                    muted: t.muted,
                    solo: t.solo,
                    volume: t.volume,
                    compose_enabled: t.compose_enabled,
                    pitch_analysis_algo: match &t.pitch_analysis_algo {
                        PitchAnalysisAlgo::WorldDll => "world_dll".to_string(),
                        PitchAnalysisAlgo::NsfHifiganOnnx => "nsf_hifigan_onnx".to_string(),
                        PitchAnalysisAlgo::VocalShifterVslib => "vslib".to_string(),
                        PitchAnalysisAlgo::ExternalResampler(id) => format!("external_resampler:{}", id),
                        PitchAnalysisAlgo::None => "none".to_string(),
                        PitchAnalysisAlgo::Unknown => "unknown".to_string(),
                    },
                    color: t.color.clone(),
                });
            }
        }
    }

    out
}

impl AppState {
    pub fn runtime_info(&self) -> RuntimeInfoPayload {
        let rt = self.runtime.lock().unwrap_or_else(|e| e.into_inner());
        let pb = self.audio_engine.snapshot_state();

        RuntimeInfoPayload {
            ok: true,
            device: rt.device.clone(),
            model_loaded: rt.model_loaded,
            audio_loaded: rt.audio_loaded,
            has_synthesized: rt.has_synthesized,
            is_playing: Some(pb.is_playing),
            playback_target: pb.target.clone(),
            timeline: None,
        }
    }

    pub fn model_config_ok(&self) -> ModelConfigPayload {
        ModelConfigPayload {
            ok: true,
            config: ModelConfig {
                audio_sample_rate: 44100,
                audio_num_mel_bins: 128,
                hop_size: 512,
                fmin: 40.0,
                fmax: 16000.0,
            },
        }
    }
}
