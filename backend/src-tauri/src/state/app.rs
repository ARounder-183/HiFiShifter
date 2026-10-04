//! `AppState`：时间线状态的**运行时容器**。
//!
//! 【为什么单独成文件】它持有 Tauri 句柄（`app_handle` / `events`）、设备层句柄
//! （`audio_engine`）、波形缓存与渲染缓存的进程级状态、以及录制会话。这些**都不是**
//! 时间线数据模型的一部分 —— 模型（`model.rs`）要搬进 `hifishifter-kernel`，
//! 而内核不认识 Tauri，也不认识音频设备。
//!
//! 【为什么搬走的是它】实测：从 `mixdown` 出发的依赖闭包会因为 `state` 而卷进
//! `project` / `notebook_assets` / `hfspeaks_v2` / `temp_manager` / `media` /
//! `recording`（43 模块 / 2.57 MB），根源就是本文件引用了它们。切出去之后，
//! 内核闭包才第一次可测。

use super::model::*;
use crate::audio_engine::AudioEngine;
use crate::models::{
    ModelConfig, ModelConfigPayload, ProjectMetaPayload, RuntimeInfoPayload, TimelineStatePayload,
};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock, RwLock};

/// `AppState` 的状态字段集合（`Default` 里构造、`AppState::new` 之外无处构造）。
#[derive(Debug, Clone, Default)]
pub struct RuntimeState {
    pub device: String,
    pub model_loaded: bool,
    pub audio_loaded: bool,
    pub has_synthesized: bool,

    pub synthesized_wav_path: Option<String>,
}

/// 波形 mipmap 计算的 RAII inflight 标记。
///
/// 构造时把 `key` 插入 `waveform_inflight`，Drop（包括 panic 展开与提前
/// 返回路径）时移除并唤醒全部 condvar 等待者。此前手工 insert/remove 的
/// 写法一旦在计算或写盘过程中 panic，标记会永久残留，该文件的后续所有
/// 请求都会在无超时的 condvar 等待中永久挂起。
struct WaveformInflightGuard<'a> {
    state: &'a AppState,
    key: String,
}

impl<'a> WaveformInflightGuard<'a> {
    /// 立即插入标记（无论此前是否已有——HashSet insert 幂等）。
    fn acquire(state: &'a AppState, key: &str) -> Self {
        let mut inflight = state
            .waveform_inflight
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        inflight.insert(key.to_string());
        Self {
            state,
            key: key.to_string(),
        }
    }
}

impl Drop for WaveformInflightGuard<'_> {
    fn drop(&mut self) {
        self.state.remove_waveform_inflight(&self.key);
    }
}

pub struct AppState {
    pub timeline: std::sync::Mutex<TimelineState>,
    pub timeline_version: std::sync::atomic::AtomicU64,
    pub timeline_history: std::sync::Mutex<TimelineHistory>,
    pub project: std::sync::Mutex<ProjectState>,
    pub runtime: std::sync::Mutex<RuntimeState>,

    /// Current UI locale reported by the frontend (e.g. "en-US", "zh-CN").
    /// Used to localize native dialogs implemented in Rust.
    pub ui_locale: RwLock<String>,

    /// When true, `checkpoint_timeline` calls are suppressed.
    /// Used by begin_undo_group / end_undo_group to group multiple
    /// backend operations into a single undo entry.
    pub suppress_checkpoints: std::sync::atomic::AtomicBool,

    pub waveform_cache_dir: std::sync::Mutex<PathBuf>,

    /// V2 多级 mipmap 波形缓存 (key = source_path)
    pub waveform_cache_v2: std::sync::Mutex<crate::hfspeaks_v2::WaveformPeakCache>,

    /// Inflight deduplication for waveform peak computation.
    /// When a file is being computed, its source_path is in this set.
    /// Other threads calling get_or_compute for the same path will wait
    /// on the Condvar until computation finishes, then read from cache.
    pub waveform_inflight: std::sync::Mutex<std::collections::HashSet<String>>,
    pub waveform_inflight_cv: std::sync::Condvar,

    /// In-memory cache of clipboard MIDI bytes, keyed by GUID (first 8 bytes of blake3 hash as hex).
    pub clipboard_midi_cache: std::sync::Mutex<std::collections::HashMap<String, Vec<u8>>>,

    /// 进程内 UI 设置缓存（`ripple_settings` / `split_transition_options`
    /// 在持有 timeline 锁的每个拖拽 tick 里读取；若每次都走磁盘读 +
    /// JSON 解析，会在全局锁内做慢 IO）。由 get/save_ui_settings 维护，
    /// 首次访问时兜底从磁盘加载。
    pub cached_ui_settings: std::sync::RwLock<Option<std::sync::Arc<crate::config::UiSettings>>>,

    // Set in Tauri setup. Used for async notifications.
    pub app_handle: OnceLock<tauri::AppHandle>,

    /// 内核事件出口。内核模块（含本文件里的后台任务）用它发进度事件，
    /// 从而不必认识 Tauri；由 `app_events::install` 在 setup 时注入。
    pub events: OnceLock<hifishifter_kernel::events::SharedEventSink>,

    // De-dup background pitch analysis jobs (keyed by rootTrackId + analysis key).
    pub pitch_inflight: std::sync::Mutex<std::collections::HashSet<String>>,

    // Current pitch analysis progress (for polling from frontend)
    pub pitch_analysis_progress:
        std::sync::RwLock<Option<crate::pitch_analysis::PitchOrigAnalysisProgressEvent>>,

    // Clip-level pitch analysis cache for performance optimization

    // Timeline snapshot for incremental pitch refresh (keyed by root_track_id)
    pub audio_engine: AudioEngine,

    /// 传输命令串行化锁（叶子锁：先取它再取 timeline，绝无反向持有）。
    ///
    /// Tauri 同步命令在阻塞线程池上并发执行，`set_transport` / `play_original`
    /// / `stop_audio` / `get_playback_state` 之间没有先后保证。快速连续
    /// 播放/暂停/停止时，乱序执行会让 stop 杀掉刚启动的播放、或与 play 的
    /// `timeline.playhead_sec` 读改交错，前端随即陷入"引擎实际已停但仍在
    /// 轮询推进光标"的分裂状态（播放光标跳变的根源之一）。把四个传输命令
    /// 放到同一把锁内全序执行：用户点击顺序（前端逐个 await）即后端执行
    /// 顺序，状态机不再有交错态。锁内最重的工作是 play_original 的节拍器
    /// 响点表重建（毫秒级），30Hz 的 get_playback_state 等锁开销可忽略；
    /// play_original 派生的渲染线程不持有也不需要这把锁。
    pub transport_lock: std::sync::Mutex<()>,

    /// 正在进行的录音会话（非录制时为 None）。
    pub recording: std::sync::Mutex<Option<crate::recording::ActiveRecording>>,
    /// 录音启动互斥（CAS）：设备就绪等待最长 8s，不能放在 `recording`
    /// 锁内，否则 meter 轮询（current_state）会同步阻塞整个等待期。
    pub recording_starting: std::sync::atomic::AtomicBool,

    /// App config directory for persisting recent projects etc.
    pub config_dir: OnceLock<std::path::PathBuf>,

    /// 启动参数传入的待打开工程路径（一次性消费）。
    pub pending_startup_project_path: Mutex<Option<String>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self::with_audio_engine(AudioEngine::new())
    }
}

impl AppState {
    /// 共享字段装配：产品Default仍建立真实引擎，测试fixture显式注入无设备引擎。
    fn with_audio_engine(audio_engine: AudioEngine) -> Self {
        Self {
            timeline: std::sync::Mutex::new(TimelineState::default()),
            timeline_version: std::sync::atomic::AtomicU64::new(0),
            timeline_history: std::sync::Mutex::new(TimelineHistory::default()),
            project: std::sync::Mutex::new(ProjectState::default()),
            runtime: std::sync::Mutex::new(RuntimeState {
                device: "tauri".to_string(),
                synthesized_wav_path: None,
                ..RuntimeState::default()
            }),

            ui_locale: RwLock::new("en-US".to_string()),

            suppress_checkpoints: std::sync::atomic::AtomicBool::new(false),

            waveform_cache_dir: std::sync::Mutex::new(crate::hfspeaks_v2::default_cache_dir()),
            waveform_cache_v2: std::sync::Mutex::new(
                crate::hfspeaks_v2::WaveformPeakCache::default(),
            ),

            waveform_inflight: std::sync::Mutex::new(std::collections::HashSet::new()),
            waveform_inflight_cv: std::sync::Condvar::new(),
            clipboard_midi_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            cached_ui_settings: std::sync::RwLock::new(None),

            app_handle: OnceLock::new(),
            events: OnceLock::new(),
            pitch_inflight: std::sync::Mutex::new(std::collections::HashSet::new()),
            pitch_analysis_progress: std::sync::RwLock::new(None),

            audio_engine,
            transport_lock: std::sync::Mutex::new(()),
            recording: std::sync::Mutex::new(None),
            recording_starting: std::sync::atomic::AtomicBool::new(false),
            config_dir: OnceLock::new(),
            pending_startup_project_path: Mutex::new(None),
        }
    }
}

/// 仅命令状态回归使用的显式fixture；没有ambient模式，也不影响其他Default调用。
#[cfg(test)]
pub(crate) fn command_test_state_without_audio_output() -> AppState {
    AppState::with_audio_engine(crate::audio_engine::command_test_support::detached_engine())
}

impl AppState {
    pub fn bump_timeline_version(&self) -> u64 {
        self.timeline_version
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
            .saturating_add(1)
    }

    pub fn set_pending_startup_project_path(&self, path: Option<String>) {
        let mut guard = self
            .pending_startup_project_path
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *guard = path;
    }

    pub fn take_pending_startup_project_path(&self) -> Option<String> {
        let mut guard = self
            .pending_startup_project_path
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        guard.take()
    }

    /// 使指定源路径的波形峰值缓存失效（内存缓存 + inflight 标记）。
    ///
    /// 当源文件被替换（即使路径相同但内容变化）时调用，确保下次请求波形
    /// 数据时重新从磁盘/文件计算，而非返回旧文件的缓存峰值。
    pub fn invalidate_waveform_cache_for_path(&self, source_path: &str) {
        {
            let mut cache_v2 = self
                .waveform_cache_v2
                .lock()
                .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());
            cache_v2.remove(source_path);
        }
        // 同时移除 inflight 标记（如果正在计算中，也让它失效重新计算）
        self.remove_waveform_inflight(source_path);
    }

    pub fn clear_waveform_cache(&self) -> crate::hfspeaks_v2::ClearStats {
        // 清理 v2 内存缓存
        {
            let mut cache_v2 = self
                .waveform_cache_v2
                .lock()
                .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());
            cache_v2.clear();
        }

        let cache_dir = {
            self.waveform_cache_dir
                .lock()
                .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner())
                .clone()
        };
        crate::hfspeaks_v2::clear_cache_dir(&cache_dir)
    }

    /// 获取或计算 v2 多级 mipmap 峰值数据
    ///
    /// 优先从内存缓存读取，其次从磁盘缓存读取，最后计算。
    /// 使用 inflight 去重：如果另一线程正在计算同一文件，当前线程会等待
    /// 其完成后直接从缓存读取，避免重复计算和重复进度事件。
    /// 首次计算时会通过 Tauri 事件推送进度（waveform_analysis_progress）
    pub fn get_or_compute_waveform_peaks_v2(
        &self,
        source_path: &str,
    ) -> Result<std::sync::Arc<crate::hfspeaks_v2::HfsPeakFile>, String> {
        if source_path.trim().is_empty() {
            return Err("empty source_path".to_string());
        }

        // ── 1. 检查内存缓存 ──
        {
            let mut cache = self
                .waveform_cache_v2
                .lock()
                .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());
            if let Some(found) = cache.get(source_path) {
                // 缓存命中：发送 cached 状态事件
                if let Some(handle) = self.app_handle.get() {
                    use tauri::Emitter;
                    let _ = handle.emit(
                        "waveform_analysis_progress",
                        serde_json::json!({
                            "sourcePath": source_path,
                            "progress": 1.0,
                            "status": "cached",
                        }),
                    );
                }
                return Ok(found.clone() as std::sync::Arc<crate::hfspeaks_v2::HfsPeakFile>);
            }
        }

        // ── 2. Inflight 去重检查 ──
        // 如果另一线程已在计算同一文件，等待它完成后从缓存读取
        {
            let inflight = self
                .waveform_inflight
                .lock()
                .unwrap_or_else(|e| e.into_inner());

            if inflight.contains(source_path) {
                // 另一线程正在计算此文件，等待 Condvar 通知
                let key = source_path.to_string();
                let _guard = self
                    .waveform_inflight_cv
                    .wait_while(inflight, |set| set.contains(&*key))
                    .unwrap_or_else(|e| e.into_inner());

                // 计算已完成，从缓存读取
                let mut cache = self
                    .waveform_cache_v2
                    .lock()
                    .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());
                if let Some(found) = cache.get(source_path) {
                    if let Some(handle) = self.app_handle.get() {
                        use tauri::Emitter;
                        let _ = handle.emit(
                            "waveform_analysis_progress",
                            serde_json::json!({
                                "sourcePath": source_path,
                                "progress": 1.0,
                                "status": "cached",
                            }),
                        );
                    }
                    return Ok(found.clone());
                }
                // 极端情况：前一线程计算失败未放入缓存，由下方的 guard
                // 登记本线程为新的计算者后继续重算。
            }
        }

        // RAII 计算者标记：构造即插入 inflight，Drop（含 panic 展开）时移除
        // 并唤醒全部等待者。此前手工 insert/remove 的写法一旦在计算或写盘
        // 过程中 panic，标记会永久残留，该文件的后续所有请求都会在无超时
        // 的 condvar 等待中永久挂起。
        let _inflight_guard = WaveformInflightGuard::acquire(self, source_path);

        // ── 3. 磁盘缓存 ──
        let cache_dir = {
            self.waveform_cache_dir
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
        };

        let hfs_cache = crate::hfspeaks_v2::HfsPeaksCache::new(cache_dir);
        let path = std::path::Path::new(source_path);

        // 尝试从磁盘加载
        if let Some(cached) = hfs_cache.try_load(path) {
            let cached: std::sync::Arc<crate::hfspeaks_v2::HfsPeakFile> =
                std::sync::Arc::new(cached);
            {
                let mut cache = self
                    .waveform_cache_v2
                    .lock()
                    .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());
                cache.insert(source_path, cached.clone());
            }
            // 磁盘缓存命中：发送 cached 状态事件
            if let Some(handle) = self.app_handle.get() {
                use tauri::Emitter;
                let _ = handle.emit(
                    "waveform_analysis_progress",
                    serde_json::json!({
                        "sourcePath": source_path,
                        "progress": 1.0,
                        "status": "cached",
                    }),
                );
            }
            // inflight 标记由 _inflight_guard 的 Drop 移除并通知等待线程
            return Ok(cached);
        }

        // ── 4. 计算新的峰值数据 ──
        // 发送 computing 状态事件（进度 0）
        let source_path_owned = source_path.to_string();
        if let Some(handle) = self.app_handle.get() {
            use tauri::Emitter;
            let _ = handle.emit(
                "waveform_analysis_progress",
                serde_json::json!({
                    "sourcePath": &source_path_owned,
                    "progress": 0.0,
                    "status": "computing",
                }),
            );
        }

        // 构建进度回调：通过 app_handle emit 事件
        let app_handle_for_cb = self.app_handle.get().cloned();
        let source_path_for_cb = source_path_owned.clone();
        let progress_cb = move |progress: f32| {
            if let Some(ref handle) = app_handle_for_cb {
                use tauri::Emitter;
                let _ = handle.emit(
                    "waveform_analysis_progress",
                    serde_json::json!({
                        "sourcePath": &source_path_for_cb,
                        "progress": progress.clamp(0.0, 1.0),
                        "status": "computing",
                    }),
                );
            }
        };

        // 计算新的峰值数据（带进度回调）
        let result =
            crate::hfspeaks_v2::compute_mipmap_peaks_with_progress(path, Some(progress_cb));

        // 如果计算失败，移除 inflight 标记并返回错误
        let peaks = match result {
            Ok(p) => p,
            Err(e) => {
                // 必须发送终态事件：前端依赖 done/failed 隐藏“正在分析波形”状态，
                // 缺失/无法读取的文件如果没有终态事件会导致进度提示永久停留。
                if let Some(handle) = self.app_handle.get() {
                    use tauri::Emitter;
                    let _ = handle.emit(
                        "waveform_analysis_progress",
                        serde_json::json!({
                            "sourcePath": &source_path_owned,
                            "progress": 1.0,
                            "status": "failed",
                            "error": e,
                        }),
                    );
                }
                // inflight 标记由 _inflight_guard 的 Drop 移除并通知等待线程
                return Err(e);
            }
        };

        // 保存到磁盘缓存
        if let Err(e) = hfs_cache.save(path, &peaks) {
            log::error!("Warning: failed to save v2 peaks cache: {}", e);
        }

        // 发送 done 状态事件
        if let Some(handle) = self.app_handle.get() {
            use tauri::Emitter;
            let _ = handle.emit(
                "waveform_analysis_progress",
                serde_json::json!({
                    "sourcePath": &source_path_owned,
                    "progress": 1.0,
                    "status": "done",
                }),
            );
        }

        let peaks = std::sync::Arc::new(peaks);
        {
            let mut cache = self
                .waveform_cache_v2
                .lock()
                .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());
            cache.insert(source_path, peaks.clone());
        }
        // 移除 inflight 标记并通知等待线程（由 _inflight_guard 的 Drop 完成）
        Ok(peaks)
    }

    /// 辅助方法：从 inflight 集合中移除 source_path 并通知所有等待线程
    fn remove_waveform_inflight(&self, source_path: &str) {
        let mut inflight = self
            .waveform_inflight
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        inflight.remove(source_path);
        self.waveform_inflight_cv.notify_all();
    }

    /// 读取 UI 设置（带进程内缓存；磁盘读 + JSON 解析只在缓存未建立时发生）。
    pub fn ui_settings_snapshot(&self) -> std::sync::Arc<crate::config::UiSettings> {
        if let Some(cached) = self
            .cached_ui_settings
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            return cached.clone();
        }
        let mut settings = if let Some(dir) = self.config_dir.get() {
            crate::config::load_ui_settings(dir)
        } else {
            crate::config::UiSettings::default()
        };
        settings.normalize_ripple_mode();
        settings.normalize_split_transition();
        settings.normalize_time_display();
        let arc = std::sync::Arc::new(settings);
        *self
            .cached_ui_settings
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Some(arc.clone());
        arc
    }

    /// 用最新设置刷新进程内缓存（get_ui_settings / save_ui_settings 调用）。
    pub fn store_ui_settings_cache(&self, settings: &crate::config::UiSettings) {
        *self
            .cached_ui_settings
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Some(std::sync::Arc::new(settings.clone()));
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
            notes_markdown: p.notes_markdown,
            base_scale: p.base_scale,
            use_custom_scale: p.use_custom_scale,
            custom_scale: p.custom_scale,
            beats_per_bar: p.beats_per_bar,
            time_signature_denominator: p.time_signature_denominator,
            grid_size: p.grid_size,
            stretch_algorithm_override: p.stretch_algorithm_override,
            hifigan_mel_stretch_override: p.hifigan_mel_stretch_override,
            save_undo_history: p.save_undo_history,
        }
    }

    /// 设置当前工程的「保存时是否写出 UNDO 数据」开关（工程级设置）。
    ///
    /// 属于工程数据：改动会置脏并在首次变脏时刷新窗口标题，但不打撤销点
    /// （它不是时间线内容）。返回更新后的工程元数据供前端同步。
    pub fn set_project_save_undo_history(&self, enabled: bool) -> ProjectMetaPayload {
        let changed = {
            let mut p = self.project.lock().unwrap_or_else(|e| e.into_inner());
            if p.save_undo_history == enabled {
                false
            } else {
                p.save_undo_history = enabled;
                true
            }
        };
        if changed {
            self.mark_project_dirty_and_retitle();
        }
        self.project_meta_payload()
    }

    /// 打一个撤销点：把当前时间线状态登记为新的「第 N 步之后的状态」。
    ///
    /// 必须在**修改时间线之前**调用：`snapshot` 即该操作执行前的实时时间线。
    /// 同一次用户操作只打一次点（`begin_undo_group` 包裹的批量操作只在组首
    /// 打点，见 `suppress_checkpoints`）。
    pub fn checkpoint_timeline(&self, snapshot: &TimelineState, op: HistoryOp) {
        self.push_checkpoint(snapshot, op.key().to_string());
    }

    /// 当前记事本内容（`ProjectState.notes_markdown` 的读取封装）。
    fn current_notes_markdown(&self) -> Option<String> {
        Some(self.current_notes_value())
    }

    /// 读取当前记事本内容（空填空串）。
    pub(crate) fn current_notes_value(&self) -> String {
        let p = self.project.lock().unwrap_or_else(|e| e.into_inner());
        p.notes_markdown.clone()
    }

    /// 写入记事本内容（不触碰历史，供跳转路径恢复用）。
    fn apply_notes_markdown(&self, markdown: String) {
        let mut p = self.project.lock().unwrap_or_else(|e| e.into_inner());
        p.notes_markdown = markdown;
    }

    // ── 记事本附件 ──────────────────────────────────────────────────────────

    /// 附件登记表快照。
    pub fn notebook_assets_snapshot(&self) -> crate::notebook_assets::NotebookAssetMap {
        self.project
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .notebook_assets
            .clone()
    }

    /// 仍被引用的附件 id 集合：**当前正文 ∪ 全部历史记录的正文**。
    ///
    /// 必须并入历史：撤销/重做会把正文换成更早的版本，若只按当前正文清理，
    /// 撤销回去的图片就会变成空白（而撤销本应无损）。历史记录里已经存了每一步
    /// 的记事本内容（见 `HistoryRecord.notes_markdown`），直接复用。
    pub fn notebook_referenced_ids(&self) -> std::collections::HashSet<String> {
        let mut ids = crate::notebook_assets::referenced_asset_ids(&self.current_notes_value());
        {
            let h = self
                .timeline_history
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            for record in &h.records {
                if let Some(markdown) = &record.notes_markdown {
                    ids.extend(crate::notebook_assets::referenced_asset_ids(markdown));
                }
            }
        }
        ids
    }

    /// 保存时的附件整理：把不再被引用的条目从登记表移除，返回移除条数。
    ///
    /// 字节内嵌之后这里不再碰磁盘 —— 只是把没人引用的条目丢掉，工程文件随之变小。
    pub fn prune_notebook_assets(&self) -> usize {
        let keep = self.notebook_referenced_ids();
        let mut p = self.project.lock().unwrap_or_else(|e| e.into_inner());
        crate::notebook_assets::prune_unreferenced(&mut p.notebook_assets, &keep)
    }

    /// 清空记事本附件（新建工程时调用）。
    pub fn reset_notebook_assets(&self) {
        let mut p = self.project.lock().unwrap_or_else(|e| e.into_inner());
        p.notebook_assets.clear();
    }

    /// 打点的公共实现：与记事本无关的改动（绝大多数操作）。
    fn push_checkpoint(&self, snapshot: &TimelineState, label: String) {
        self.push_checkpoint_with_notes(snapshot, label, None);
    }

    /// 打点的公共实现（`op` 为语言无关的操作 key，见 HistoryOp）。
    ///
    /// `notes_markdown`：`Some(..)` 表示**本次操作之前的记事本内容** —— 它
    /// 就是即将被打点的那个状态（state N）的记事本，因此记在**旧记录**上。
    /// 传 `None` 表示"用现场值补齐"，绝大多数非记事本操作走这一条。
    fn push_checkpoint_with_notes(
        &self,
        snapshot: &TimelineState,
        label: String,
        notes_markdown: Option<String>,
    ) {
        // When suppress_checkpoints is active (inside an undo group),
        // skip pushing to the undo stack so multiple operations become
        // a single undo entry.
        if self
            .suppress_checkpoints
            .load(std::sync::atomic::Ordering::Acquire)
        {
            self.mark_project_dirty_and_retitle();
            return;
        }
        {
            let mut h = self
                .timeline_history
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let mut supplied_notes=notes_markdown;
            hifishifter_kernel::editor::history::checkpoint(&mut h,snapshot,label,
                || supplied_notes.take().or_else(|| self.current_notes_markdown()));
        }

        self.bump_timeline_version();
        // 历史变化广播：前端「操作记录」窗口与撤销/重做可用性据此实时刷新。
        self.emit_history_state();
        self.mark_project_dirty_and_retitle();
    }

    /// 标记工程已修改，并在首次变脏时更新窗口标题（添加 * 号）。
    pub(crate) fn mark_project_dirty_and_retitle(&self) {
        let (name, was_clean) = {
            let mut p = self.project.lock().unwrap_or_else(|e| e.into_inner());
            let was_clean = !p.dirty;
            p.dirty = true;
            (p.name.clone(), was_clean)
        };
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
        {
            let mut h = self
                .timeline_history
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            h.records.clear();
            h.position = 0;
            h.started_at_ms = now_unix_ms();
        }
        // 新工程 / 打开工程：历史已清空，广播新的「操作记录」。
        self.emit_history_state();
    }

    /// 当前撤销/重做步数（「撤销 / 重做」可用性判定的权威来源）。
    pub fn history_depths(&self) -> (usize, usize) {
        let h = self
            .timeline_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        history_depths_of(&h)
    }

    /// 当前所处历史位置（= 已应用的步数）。
    pub fn history_position(&self) -> usize {
        let h = self
            .timeline_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        h.position
    }

    /// 「操作记录」载荷：当前位置 + 状态链（每条只带语言无关的 key 与时刻）。
    ///
    /// 历史为空（新建 / 打开工程后尚无编辑）时补一条「初始化状态」行，
    /// 保证窗口始终有内容可看。
    pub fn history_state_json(&self) -> serde_json::Value {
        let h = self
            .timeline_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (undo_depth, redo_depth) = history_depths_of(&h);
        // started_at_ms == 0（进程启动后既未清空历史也未打点）时用当前时刻，
        // 避免「初始化状态」行显示 1970 年。
        let started_at_ms = if h.started_at_ms == 0 {
            now_unix_ms()
        } else {
            h.started_at_ms
        };
        let records: Vec<serde_json::Value> = if h.records.is_empty() {
            vec![serde_json::json!({ "label": null, "atMs": started_at_ms })]
        } else {
            h.records
                .iter()
                .map(|record| {
                    serde_json::json!({
                        "label": record.label,
                        "atMs": record.at_ms,
                    })
                })
                .collect()
        };
        serde_json::json!({
            "ok": true,
            "position": h.position,
            "undoDepth": undo_depth,
            "redoDepth": redo_depth,
            "records": records,
        })
    }

    /// 向前端广播「操作记录」与撤销/重做可用性。
    ///
    /// 打点、清空历史、撤销/重做、跳转之后调用：窗口与按钮由此实时刷新，
    /// 不需要额外轮询。
    pub fn emit_history_state(&self) {
        let Some(handle) = self.app_handle.get() else {
            return;
        };
        use tauri::Emitter;
        let _ = handle.emit("history_state", self.history_state_json());
    }

    /// Begin an undo group: push the current state once and suppress further checkpoints.
    ///
    /// `label` 由调用方给出（语言无关的 key，前端本地化）：批量导入等场景
    /// 能给出比「批量操作」更准确的名字；缺省用 HistoryOp::Batch。
    pub fn begin_undo_group(&self, label: Option<String>) -> TimelineStatePayload {
        let tl = self.timeline.lock().unwrap_or_else(|e| e.into_inner());
        // Force a checkpoint even if suppress was already active (defensive)
        self.suppress_checkpoints
            .store(false, std::sync::atomic::Ordering::Release);
        self.push_checkpoint(
            &tl,
            label.unwrap_or_else(|| HistoryOp::Batch.key().to_string()),
        );
        self.suppress_checkpoints
            .store(true, std::sync::atomic::Ordering::Release);
        let mut payload = tl.to_payload();
        payload.project = Some(self.project_meta_payload());
        let (undo_depth, redo_depth) = self.history_depths();
        payload.undo_depth = Some(undo_depth);
        payload.redo_depth = Some(redo_depth);
        payload
    }

    /// End the undo group: re-enable checkpoints.
    pub fn end_undo_group(&self) -> serde_json::Value {
        self.suppress_checkpoints
            .store(false, std::sync::atomic::Ordering::Release);
        serde_json::json!({ "ok": true })
    }

    /// 写入记事本内容，并保证撤销栈里始终只有**一步**「编辑记事本」。
    ///
    /// ## 合并规则（结构化，与时间无关）
    ///
    /// 历史最前沿一步的 key 就是「编辑记事本」时，本次写入**并入该步**：
    /// 不打新点、只更新 `ProjectState.notes_markdown`。这条步的快照（时间线 +
    /// 记事本）由既有的惰性补齐机制在下一次打点/跳转时用实时状态填上，撤销
    /// 回它之前的那一步就是整段编辑开始前的样子。
    ///
    /// 只有**其它操作介入**（剪辑、参数、导入…任何会 `push_checkpoint` 的路径）
    /// 或**历史跳转**离开前沿时，这一步才落定；之后的记事本输入再开新步。
    ///
    /// 【为什么不用时间窗口】早期实现靠前端 700ms 停手超时 + 失焦来"收尾"，
    /// 用户只要停顿超过阈值（思考、看内容、切窗口查资料）就被切成新的一步，
    /// 一场长编辑会话能产生几十上百条「编辑记事本」记录——既撤不回编辑前的
    /// 状态，还会把真正的剪辑/参数历史挤出 `MAX_UNDO_HISTORY` 上限。结构化
    /// 合并完全不看时间：连续写记事本=一步，被别的操作打断=另起一步，语义
    /// 与「拖拽参数线」等手势类操作一致，且不需要前后端协商任何开窗/收尾
    /// 时序（前端只管发文本）。
    ///
    /// 记事本不在时间线上，所以这一步**不**触发音频/合成层的任何失效。
    pub fn set_notes_markdown(&self, markdown: String) {
        // 判定"前沿是否已是记事本步"必须在打点之前：`push_checkpoint_*` 会
        // 追加新记录，读时就分不清了。
        let merged_into_existing = self.history_top_is_notes_edit();
        if !merged_into_existing {
            let before = self.current_notes_markdown();
            let tl = self.timeline.lock().unwrap_or_else(|e| e.into_inner());
            self.push_checkpoint_with_notes(&tl, HistoryOp::EditNotes.key().to_string(), before);
        }
        let mut p = self.project.lock().unwrap_or_else(|e| e.into_inner());
        if p.notes_markdown != markdown {
            p.notes_markdown = markdown;
            p.dirty = true;
        }
        // 分节闸门是一次性的：无论本次是否真的另起了一步，都消费掉。
        p.notes_history_sealed = false;
        drop(p);
        self.emit_history_state();
    }

    /// 关闭记事本编辑的"结构性合并"窗口：下一次写入必然另起一个撤销步。
    ///
    /// 后端默认把连续写入并入前沿那一步（无论用户打字多久），这对纯文本
    /// 记事本足够；富文本编辑器有自己的细粒度撤销栈，但仍需要一个"这一节
    /// 到此为止"的显式信号 —— 切换视图模式、失焦、关闭面板、保存前调用。
    pub fn seal_notes_history(&self) {
        {
            let mut p = self.project.lock().unwrap_or_else(|e| e.into_inner());
            p.notes_history_sealed = true;
        }
        self.emit_history_state();
    }

    /// 历史最前沿的一步是否是「编辑记事本」。
    ///
    /// 仅当该步**就是当前所处位置**（打点后位置恒指向它）且 label 匹配时为
    /// 真。历史为空 / 前沿是初始状态行 / 已跳转离开（此时前沿 label 是其它
    /// 操作或 None）均为假。分节闸门置位时同样为假 —— 这正是"另起一步"的
    /// 实现方式。
    fn history_top_is_notes_edit(&self) -> bool {
        if self
            .project
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .notes_history_sealed
        {
            return false;
        }
        let h = self
            .timeline_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        h.records
            .get(h.position)
            .and_then(|record| record.label.as_deref())
            == Some(HistoryOp::EditNotes.key())
    }

    /// 撤销一步（跳到当前位置之前）。
    pub fn undo_timeline(&self) -> TimelineStatePayload {
        let target = self.history_position().saturating_sub(1);
        self.set_history_position(target, HistoryJumpIntent::Undo)
    }

    /// 重做一步（跳到当前位置之后）。
    pub fn redo_timeline(&self) -> TimelineStatePayload {
        let target = self.history_position().saturating_add(1);
        self.set_history_position(target, HistoryJumpIntent::Redo)
    }

    /// 登记「边缘拉伸」步骤带来的选区变化（作用于**当前所处的那一步**）。
    ///
    /// 前端在曲线回写成功之后调用：那一刻新步已追加、`position` 指向它，因此
    /// 后端无需任何位置推断。只接受 `ParamCurve` 步 —— 写回被抑制（`suppress_
    /// checkpoints`，例如处于撤销组内）时当前位置可能是别的操作甚至是初始状态
    /// 行，此时**拒绝登记**而不是把选区快照挂到无关步骤上（那会让撤销那一步时
    /// 莫名改掉用户的选区）。
    ///
    /// 两个入参都是**帧制**的选区：每项 `[startFrame, frameCount]`（半开区间）。
    ///
    /// @returns `ok = false` + `reason` 表示没有登记（步骤不匹配）；前端无需重试，
    ///   这种情况本就意味着这次手势没有产生可撤销的步骤。
    pub fn record_param_selection_step(
        &self,
        before: Vec<[f32; 2]>,
        after: Vec<[f32; 2]>,
    ) -> serde_json::Value {
        let mut h = self
            .timeline_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let position = h.position;
        let Some(record) = h.records.get_mut(position) else {
            return serde_json::json!({ "ok": false, "reason": "no-step" });
        };
        if record.label.as_deref() != Some(HistoryOp::ParamCurve.key()) {
            return serde_json::json!({ "ok": false, "reason": "step-is-not-param-curve" });
        }
        // 归一：丢掉非有限值，并把帧号 / 帧数取整（选区的单位是整数帧，见
        // `ParamSelectionStep` 的说明）。前端已经发整数，这里再取一次是为了让
        // "整数帧"这条不变式钉在写入侧，不依赖调用方自律。
        let filtered = |ranges: Vec<[f32; 2]>| -> Vec<[f32; 2]> {
            ranges
                .into_iter()
                .filter(|r| r[0].is_finite() && r[1].is_finite())
                .map(|r| [r[0].round(), r[1].max(0.0).round()])
                .collect()
        };
        record.param_selection = Some(ParamSelectionStep {
            before: filtered(before),
            after: filtered(after),
        });
        serde_json::json!({ "ok": true })
    }

    /// 跳到历史中的第 `target` 个状态（「操作记录」窗口双击 / 撤销 / 重做
    /// 共用此入口）。
    ///
    /// 越界或原地不动时返回 `ok = false`：前端不套用任何快照，界面零刷新
    /// 零变更（与空栈撤销/重做的语义一致）；载荷仍带回权威的历史状态，
    /// 前端可借机纠正可能落后的镜像。
    ///
    /// `intent` 决定带回哪一侧的选区快照（见 `param_selection_restore_for`）。
    pub fn set_history_position(
        &self,
        target: usize,
        intent: HistoryJumpIntent,
    ) -> TimelineStatePayload {
        let mut tl = self.timeline.lock().unwrap_or_else(|e| e.into_inner());
        let mut h = self
            .timeline_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if target >= h.records.len() || target == h.position {
            let (undo_depth, redo_depth) = history_depths_of(&h);
            drop(h);
            let mut payload = tl.to_payload();
            payload.project = Some(self.project_meta_payload());
            payload.ok = false;
            payload.undo_depth = Some(undo_depth);
            payload.redo_depth = Some(redo_depth);
            return payload;
        }
        // 离开当前位置：先用实时时间线补齐该位置的快照，之后跳回才有依据。
        let current_position = h.position;
        if let Some(current) = h.records.get_mut(current_position) {
            current.state = Some(tl.clone());
            // 当前位置若还不知道自己的记事本内容，此刻的现场值就是它的值 ——
            // 之后从这里再往前跳时，记事本才能正确回到这一步的样子。
            if current.notes_markdown.is_none() {
                current.notes_markdown = Some(self.current_notes_value());
            }
        }
        let Some(next_state) = h.records.get(target).and_then(|r| r.state.clone()) else {
            // 目标记录尚未补齐（理论上不可达：非当前位置的记录在离开时即已
            // 补齐）。宁可原地不动，也不套用一个不完整的状态。
            let (undo_depth, redo_depth) = history_depths_of(&h);
            drop(h);
            let mut payload = tl.to_payload();
            payload.project = Some(self.project_meta_payload());
            payload.ok = false;
            payload.undo_depth = Some(undo_depth);
            payload.redo_depth = Some(redo_depth);
            return payload;
        };
        let scale_before = tl.render_scale_signature();
        *tl = next_state;
        h.position = target;
        // 记事本：跳转恢复目标记录的快照时连带恢复它记录的记事本内容
        // （HistoryRecord.notes_markdown 在离开每一步时惰性补齐，见下）。
        // 早期实现在每次撤销都用后端"当前值"覆盖前端，于是"记事本有内容时，
        // 为任意操作执行撤销都会清空记事本"；改为每步各自记录后，撤销/重做
        // 跨过记事本编辑就能正确回到该步的内容。
        let target_notes = h.records.get(target).and_then(|r| r.notes_markdown.clone());
        if let Some(notes) = target_notes.clone() {
            self.apply_notes_markdown(notes);
        }
        // 选区（「边缘拉伸」步才有）：与记事本同一套路 —— 谁记得谁负责带回，
        // 前端只管无脑套用。`None` = 这一步与选区无关，前端不得改动选区。
        let param_selection_restore = param_selection_restore_for(&h, target, intent);
        let (undo_depth, redo_depth) = history_depths_of(&h);
        drop(h);
        self.bump_timeline_version();
        self.emit_history_state();
        // 恢复的快照可能改变 clip 的 active take：hnsep 分离缓存键只含
        // clip_id+采样率+样本数，等长 Take 会命中彼此的 harmonic/noise
        // stem（气声路径串音）。撤销/重做属低频操作，整体清空与
        // set_clip_active_take 命令路径的失效策略一致。
        crate::hnsep_onnx::clear_separation_cache();
        // 恢复的时间线快照可能带有 Tempo Map 初始点（工程基准记录）：
        // 同步工程 BPM/拍号/音阶，避免回退后工程记录与 Tempo Map 分叉。
        {
            let mut p = self.project.lock().unwrap_or_else(|e| e.into_inner());
            self.sync_project_record_from_tempo_map(&mut tl, &mut p);
        }
        self.audio_engine.update_timeline(tl.clone());
        self.invalidate_render_caches_if_scale_changed(&tl, &scale_before);
        let mut payload = tl.to_payload();
        payload.project = Some(self.project_meta_payload());
        payload.undo_depth = Some(undo_depth);
        payload.redo_depth = Some(redo_depth);
        // 回给前端：跨过记事本编辑则带上该步的记事本，否则留 None 让前端保留现场。
        payload.notes_markdown = target_notes;
        payload.param_selection_restore = param_selection_restore;
        payload
    }

    /// 实际生效音阶发生变化时失效所有渲染缓存，并在「后台预渲染」启用时
    /// 触发后台渲染（与直接编辑 Tempo Map 的路径一致；撤销/重做恢复的快照
    /// 同样走这里 —— 引擎的 clip 差分检测不会覆盖 Tempo Map）。
    fn invalidate_render_caches_if_scale_changed(&self, tl: &TimelineState, scale_before: &str) {
        let scale_after = tl.render_scale_signature();
        if scale_before == scale_after {
            return;
        }
        for clip in &tl.clips {
            crate::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
        }
        if let Some(handle) = self.app_handle.get() {
            let _ = crate::commands::playback::request_background_render(handle);
        }
    }

    /// 从 Tempo Map 0 位置初始点同步“工程基准记录”（BPM / 拍号 / 音阶）。
    ///
    /// 初始点即工程基准记录（与 `set_timeline_tempo_map` 的双向同步约定一致），
    /// 撤销/重做恢复时间线快照后也必须重新同步，否则工程记录与 Tempo Map
    /// 会永久分叉（例如撤销音阶修改后工程仍显示旧音阶，保存/重开也无法自愈）。
    /// 仅在实际值变化时写回并标记工程 dirty。
    pub fn sync_project_record_from_tempo_map(&self, tl: &mut TimelineState, p: &mut ProjectState) {
        let Some(first) = tl
            .tempo_map
            .as_ref()
            .and_then(|points| points.first())
            .cloned()
        else {
            return;
        };
        let mut changed = false;

        let bpm = first.bpm.clamp(10.0, 960.0);
        if (tl.bpm - bpm).abs() > 1e-9 {
            tl.bpm = bpm;
            changed = true;
        }
        let beats = first.numerator.unwrap_or(4).clamp(1, 32);
        if p.beats_per_bar != beats {
            p.beats_per_bar = beats;
            changed = true;
        }
        let denominator = match first.denominator {
            Some(d) if matches!(d, 1 | 2 | 4 | 8 | 16 | 32) => d,
            _ => p.time_signature_denominator,
        };
        if p.time_signature_denominator != denominator {
            p.time_signature_denominator = denominator;
            changed = true;
        }

        if let Some(scale) = first.scale.as_ref() {
            if let Some(key) = scale.key.as_deref() {
                if p.base_scale != key || p.use_custom_scale {
                    p.base_scale = key.to_string();
                    p.use_custom_scale = false;
                    p.custom_scale = None;
                    changed = true;
                }
                if let Some(notes) = scale_notes_for_key(key) {
                    tl.project_scale_notes = notes;
                }
            } else if let Some(notes) = scale.notes.as_ref() {
                let mut normalized: Vec<u8> = notes.iter().map(|n| n % 12).collect();
                normalized.sort_unstable();
                normalized.dedup();
                if !normalized.is_empty() {
                    let name = scale
                        .name
                        .clone()
                        .filter(|n| !n.trim().is_empty())
                        .unwrap_or_else(|| "Custom Scale".to_string());
                    let same = p.use_custom_scale
                        && p.custom_scale.as_ref().map(|c| (&c.name, &c.notes))
                            == Some((&name, &normalized));
                    if !same {
                        p.custom_scale = Some(crate::project::CustomScale {
                            id: p
                                .custom_scale
                                .as_ref()
                                .map(|c| c.id.clone())
                                .unwrap_or_else(|| new_id("cs")),
                            name,
                            notes: normalized.clone(),
                        });
                        p.use_custom_scale = true;
                        changed = true;
                    }
                    tl.project_scale_notes = normalized;
                }
            }
        }
        if changed {
            p.dirty = true;
        }
    }
}

impl AppState {
    pub fn runtime_info(&self) -> RuntimeInfoPayload {
        let rt = self.runtime.lock().unwrap_or_else(|e| e.into_inner());
        let pb = self.audio_engine.snapshot_state();

        // Report the execution provider the live vocoder session actually runs
        // on.  This used to be a compile-time constant, so the menu claimed
        // "GPU (CoreML)" even when every session had fallen back to CPU.
        let gpu_backend = crate::nsf_hifigan_onnx::active_backend_name();

        RuntimeInfoPayload {
            ok: true,
            device: rt.device.clone(),
            model_loaded: rt.model_loaded,
            audio_loaded: rt.audio_loaded,
            has_synthesized: rt.has_synthesized,
            is_playing: Some(pb.is_playing),
            playback_target: pb.target.clone(),
            timeline: None,
            gpu_backend: gpu_backend.to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 与 `model.rs` 的测试同形的辅助：取某个 clip 的 `start_sec`。
    ///
    /// 【为什么两边各持一份】测试辅助函数不跨模块共享 —— 让 `model.rs` 保持自包含，
    /// 它才能整体搬进内核 crate 而不拖着 app 层的测试工具。
    fn find_clip_start(timeline: &TimelineState, clip_id: &str) -> f64 {
        timeline
            .clips
            .iter()
            .find(|clip| clip.id == clip_id)
            .map(|clip| clip.start_sec)
            .unwrap_or(f64::NAN)
    }
    /// 记事本必须能被撤销/重做恢复，且**与记事本无关的撤销不得清空它**。
    ///
    /// 回归对象 1：早期实现里记事本只存在前端、后端只在保存时看到它，于是
    /// 「记事本有内容时，为任意操作执行撤销都会清空记事本」。
    ///
    /// 回归对象 2：合并曾靠前端 700ms 停手超时，长编辑会话（停顿 > 阈值）
    /// 会产生大量「编辑记事本」记录，既撤不回编辑前状态、还会把其它操作的
    /// 历史挤出撤销栈上限。现在合并在后端按**历史结构**进行：前沿一步就是
    /// 「编辑记事本」时，后续写入无限并入 —— 只有其它操作介入或跳转离开
    /// 才落定。时间完全不参与。
    #[test]
    fn notes_edit_is_recorded_in_history_and_kept_across_other_undos() {
        let state = AppState::default();
        assert_eq!(state.current_notes_value(), "");

        // 一段连续输入：任意多次调用（间隔长短无关——合并不看时间）只产生
        // **一个**撤销步。停顿用例刻意不引入 sleep：结构化合并在逻辑上就
        // 不可能因停顿分裂，sleep 只会拖慢测试。
        for (index, next) in ["H", "He", "Hel", "Hello, wor", "Hello, world!"]
            .iter()
            .enumerate()
        {
            state.set_notes_markdown(next.to_string());
            let (undo_depth, _) = state.history_depths();
            assert_eq!(undo_depth, 1, "第 {index} 次写入后仍必须只有一步");
        }
        assert_eq!(state.current_notes_value(), "Hello, world!");

        // 再来一个**非记事本**操作：移动 clip。它必须正常落定记事本步、
        // 另起新步 —— 这是"合并被其它操作打断"的断言。
        {
            let mut tl = state.timeline.lock().unwrap();
            let track_id = tl.tracks[0].id.clone();
            let clip_id = tl.add_clip(Some(track_id), None, Some(0.0), Some(1.0), None);
            drop(tl);
            let prev_sec = find_clip_start(&state.timeline.lock().unwrap(), &clip_id);
            let mut tl = state.timeline.lock().unwrap();
            state.checkpoint_timeline(&tl, HistoryOp::MoveClip);
            tl.clips
                .iter_mut()
                .find(|clip| clip.id == clip_id)
                .unwrap()
                .start_sec = prev_sec + 5.0;
        }
        let (undo_depth, _) = state.history_depths();
        assert_eq!(undo_depth, 2, "其它操作必须另起一步");

        // 其它操作介入后，记事本输入再开**新的一步**（前一步已落定）。
        state.set_notes_markdown("Hello, world! -- second session".to_string());
        let (undo_depth, _) = state.history_depths();
        assert_eq!(undo_depth, 3);

        // 撤销第二次记事本会话：回到第一会话结束时的内容。
        let payload = state.undo_timeline();
        assert!(payload.ok);
        assert_eq!(payload.notes_markdown, Some("Hello, world!".to_string()));
        assert_eq!(state.current_notes_value(), "Hello, world!");

        // 撤销 clip 移动：记事本原样保留（回归对象 1 的核心断言）。
        let payload = state.undo_timeline();
        assert!(payload.ok);
        assert_eq!(
            state.current_notes_value(),
            "Hello, world!",
            "无关操作的撤销清空了记事本 —— 回归"
        );

        // 再撤销一步 = 跨过第一段记事本编辑：恢复到编辑**开始之前**的内容。
        // 整段长会话（5 次写入）只占这一步 —— 撤销一次即回到编辑前状态。
        let payload = state.undo_timeline();
        assert!(payload.ok);
        assert_eq!(payload.notes_markdown, Some(String::new()));
        assert_eq!(state.current_notes_value(), "");

        // 重做：记事本回到编辑后的内容。
        let payload = state.redo_timeline();
        assert!(payload.ok);
        assert_eq!(payload.notes_markdown, Some("Hello, world!".to_string()));
        assert_eq!(state.current_notes_value(), "Hello, world!");
    }

    /// 「边缘拉伸」的选区快照：随**产生它的那一步**记录，撤销/重做/跳转由载荷带回；
    /// 位置被复用（新检查点丢弃重做分支）时必须清空，不得继承旧分支的快照。
    ///
    /// 这是"撤销后曲线回来了、选区却没回到拉伸前"的根治点：前端不再按撤销深度
    /// 推断"我这一步是第几步"，因此不存在镜像滞后 / 分支裁剪导致的位置错位。
    #[test]
    fn param_selection_step_rides_on_its_history_step() {
        let state = AppState::default();
        let before: Vec<[f32; 2]> = vec![[1.0, 2.0]];
        let after: Vec<[f32; 2]> = vec![[1.0, 3.0]];

        // 第 1 步：参数曲线（拉伸回写的首块打点），随后登记该步的选区变化。
        {
            let tl = state.timeline.lock().unwrap();
            state.checkpoint_timeline(&tl, HistoryOp::ParamCurve);
        }
        assert_eq!(
            state.record_param_selection_step(before.clone(), after.clone())["ok"],
            serde_json::json!(true),
            "当前步是参数曲线步时应登记成功",
        );

        // 撤销这一步：载荷带回拉伸**前**的选区。
        let payload = state.undo_timeline();
        assert!(payload.ok);
        assert_eq!(payload.param_selection_restore, Some(before.clone()));

        // 跳回该步所在状态（「操作记录」双击）：带回拉伸**后**的选区。
        let payload = state.set_history_position(1, HistoryJumpIntent::Jump);
        assert!(payload.ok);
        assert_eq!(payload.param_selection_restore, Some(after.clone()));

        // 再撤销一次（重做栈因此非空）—— 下一步要验证"位置被复用"。
        let payload = state.undo_timeline();
        assert!(payload.ok);
        assert_eq!(payload.param_selection_restore, Some(before.clone()));

        // 位置被复用：此时做一次**别的**检查点（丢弃重做分支、复用第 1 个位置），
        // 新步不得继承旧分支的选区快照。
        {
            let tl = state.timeline.lock().unwrap();
            state.checkpoint_timeline(&tl, HistoryOp::MoveClip);
        }
        let payload = state.undo_timeline();
        assert!(payload.ok);
        assert_eq!(
            payload.param_selection_restore, None,
            "被复用的位置必须清空旧分支的选区快照",
        );

        // 与选区无关的步骤不得触碰选区（载荷不带该字段）。
        let payload = state.redo_timeline();
        assert!(payload.ok);
        assert_eq!(
            payload.param_selection_restore, None,
            "非拉伸步重做时不得改动选区",
        );

        // 登记接口只接受参数曲线步：当前位置是 MoveClip 时应拒绝。
        assert_eq!(
            state.record_param_selection_step(before, after)["ok"],
            serde_json::json!(false),
        );
    }

    /// 登记接口把选区归一为**整数帧**：非有限项丢弃、帧号取整、帧数下钳 0。
    ///
    /// 前端本来就发整数（选区内部单位即帧），这里再归一一次是为了把"整数帧"
    /// 这条不变式钉在**写入侧** —— 不依赖调用方自律，也让磁盘上的历史文件
    /// 永远是规整的整数。
    #[test]
    fn record_param_selection_step_normalizes_to_integer_frames() {
        let state = AppState::default();
        {
            let tl = state.timeline.lock().unwrap();
            state.checkpoint_timeline(&tl, HistoryOp::ParamCurve);
        }
        assert_eq!(
            state.record_param_selection_step(
                // 非有限项被丢弃；10.4 → 10、20.6 → 21。
                vec![[10.4, 20.6], [f32::NAN, 3.0], [f32::INFINITY, 3.0]],
                // 起点可以为负（选区被拖到 0 左侧时随数据越界），帧数不夹到负。
                vec![[-5.0, -7.2]],
            )["ok"],
            serde_json::json!(true),
        );

        let h = state.timeline_history.lock().unwrap();
        let step = h
            .records
            .iter()
            .find_map(|record| record.param_selection.as_ref())
            .expect("选区注记已登记");
        assert_eq!(step.before, vec![[10.0, 21.0]], "取整并丢弃非有限项");
        assert_eq!(step.after, vec![[-5.0, 0.0]], "帧数下钳 0，起点不夹");
    }
}
