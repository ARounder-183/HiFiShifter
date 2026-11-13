// 命令层共用工具函数
use crate::state::AppState;
use serde::Serialize;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PlaybackRenderingStateEvent {
    pub(crate) active: bool,
    pub(crate) progress: Option<f64>,
    pub(crate) target: Option<String>,
    /// 本轮渲染 pass 的标识。
    ///
    /// 【为什么需要它】进度在**同一轮内**单调不减（见 `renderer::progress` 的
    /// 单调闸门），但**跨轮允许重新起始** —— 打开大工程时音高分析逐批解锁，
    /// 每批各跑一轮，每轮都从 0% 涨到 100% 才是正确反馈。前端只看到
    /// `active`/`progress` 时无法区分"新一轮开始"与"同一轮回退"，
    /// 于是要么误杀真实的重新起始，要么把回退显示给用户。带上 pass 序号后，
    /// 前端按"pass 变了 ⇒ 允许重新起始"判定，不需要任何超时启发式。
    pub(crate) pass: Option<u64>,
}

/// 用户操作「面包屑」白名单：只记操作类型，绝不记工程内容或完整路径原文。
/// 归「中」档（Info），一会话最多几十条。
pub(crate) fn breadcrumb(command: &str) {
    const BREADCRUMBS: &[&str] = &[
        "open_project_dialog",
        "save_project",
        "save_project_as",
        "import_audio_dialog",
        "import_audio_advanced",
        "export_audio_advanced",
        "start_background_render",
        "cancel_background_render",
        "export_diagnostics",
        "start_recording",
        "stop_recording",
        "clear_waveform_cache",
        "set_ui_locale",
        "set_log_settings",
    ];
    if BREADCRUMBS.contains(&command) {
        log::info!("[ui] action={command}");
    }
}

pub(crate) fn guard_command<T>(name: &str, fallback: T, f: impl FnOnce() -> T) -> T {
    breadcrumb(name);
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => {
            log::error!("command panicked: {name}");
            fallback
        }
    }
}

/// 同步命令返回类型的"panic 失败形态"。
///
/// 【为什么要有这个 trait】同步 Tauri 命令在 WebView IPC 线程内联执行，panic 越过
/// FFI 边界会 abort 整个 App；`guard_command` 需要一个与返回类型同型的 fallback，
/// 而 payload 大多没有 `Default`（且默认零值未必是"失败"语义）。把每种 payload
/// 的失败形态集中在 trait 实现里，命令包装层就只需一行 `common::guarded(...)`，
/// 不必各自发明一份错误数据。
pub(crate) trait PanicFallback: Sized {
    /// panic 被拦下后返回给前端的失败形态：`ok=false` / 空集合 / 带 error 原因。
    fn panic_fallback() -> Self;
}

/// 同步命令的统一 panic 守卫。用法：`common::guarded("命令名", || 原命令体)`。
///
/// 【覆盖面】`commands.rs` 的全部同步命令接线完毕（此前 23 条读路径 / 配置命令
/// 只有裸委托，panic 会越过 FFI 边界 abort 整个 App）；`process_audio` 另走
/// `spawn_blocking`，由任务边界结构性遏制。新增同步命令必须在这里接守卫，
/// 新增返回类型必须实现 `PanicFallback`。
pub(crate) fn guarded<T: PanicFallback>(name: &str, f: impl FnOnce() -> T) -> T {
    guard_command(name, T::panic_fallback(), f)
}

mod fallbacks {
    use super::PanicFallback;
    use std::collections::BTreeMap;

    impl PanicFallback for crate::models::RuntimeInfoPayload {
        fn panic_fallback() -> Self {
            Self {
                ok: false,
                device: String::new(),
                model_loaded: false,
                audio_loaded: false,
                has_synthesized: false,
                is_playing: None,
                playback_target: None,
                timeline: None,
                gpu_backend: String::new(),
            }
        }
    }

    impl PanicFallback for crate::models::ProjectMetaPayload {
        fn panic_fallback() -> Self {
            Self {
                name: String::new(),
                path: None,
                dirty: false,
                recent: vec![],
                notes_markdown: String::new(),
                base_scale: String::new(),
                use_custom_scale: false,
                custom_scale: None,
                beats_per_bar: 0,
                time_signature_denominator: 0,
                grid_size: String::new(),
                stretch_algorithm_override: None,
                save_undo_history: false,
                hifigan_mel_stretch_override: None,
            }
        }
    }

    impl PanicFallback for crate::config::AutoBackupSettings {
        fn panic_fallback() -> Self {
            Self {
                save_on_save_enabled: false,
                timed_backup_enabled: false,
                timed_backup_interval_sec: 0,
                timed_backup_path_template: String::new(),
            }
        }
    }

    impl PanicFallback for crate::config::RecordingSettings {
        fn panic_fallback() -> Self {
            Self {
                source_device: String::new(),
                capture_mode: String::new(),
                loopback_device: String::new(),
                capture_app_id: String::new(),
                capture_app_name: String::new(),
                capture_app_process: String::new(),
                sample_rate: 0,
                bit_depth: 0,
                channels: 0,
                input_gain_db: 0.0,
                monitor_enabled: false,
                monitor_gain_db: 0.0,
                countdown_sec: 0,
                auto_normalize: false,
                auto_stop_at_selection_end: false,
                path_template: String::new(),
            }
        }
    }

    impl PanicFallback for crate::recording::RecordingStatePayload {
        fn panic_fallback() -> Self {
            Self {
                active: false,
                elapsed_sec: 0.0,
                level: 0.0,
                peak: 0.0,
                start_sec: None,
                output_path: None,
                error: Some("panic in command: get_recording_state".to_string()),
            }
        }
    }

    impl PanicFallback for crate::models::CheckSourceFilesChangedPayload {
        fn panic_fallback() -> Self {
            Self { changed: vec![] }
        }
    }

    impl PanicFallback for crate::models::ParamFramesPayload {
        fn panic_fallback() -> Self {
            Self {
                geometry_signature: None,
                ok: false,
                root_track_id: String::new(),
                param: String::new(),
                frame_period_ms: 0.0,
                start_frame: 0,
                orig: vec![],
                edit: vec![],
                edited: vec![],
                authority_revision: 0,
                baseline_revision: 0,
                geometry_revision: 0,
                binary: None,
                reference_kind: crate::models::ParamReferenceKind::DefaultValue,
                analysis_pending: None,
                analysis_progress: None,
                pitch_edit_user_modified: None,
                pitch_edit_backend_available: None,
                edit_sentinel: None,
                dyn_orig_key: None,
            }
        }
    }

    impl PanicFallback for crate::models::StaticParamValuePayload {
        fn panic_fallback() -> Self {
            Self {
                ok: false,
                root_track_id: String::new(),
                param: String::new(),
                value: 0.0,
            }
        }
    }

    impl PanicFallback for crate::models::ModelConfigPayload {
        fn panic_fallback() -> Self {
            Self {
                ok: false,
                config: crate::models::ModelConfig {
                    audio_sample_rate: 0,
                    audio_num_mel_bins: 0,
                    hop_size: 0,
                    fmin: 0.0,
                    fmax: 0.0,
                },
            }
        }
    }

    impl PanicFallback for crate::models::PlaybackStatePayload {
        fn panic_fallback() -> Self {
            Self {
                ok: false,
                is_playing: false,
                waiting_for_render: false,
                target: None,
                base_sec: 0.0,
                position_sec: 0.0,
                duration_sec: 0.0,
            }
        }
    }

    impl PanicFallback for super::super::onnx_status::OnnxStatusPayload {
        fn panic_fallback() -> Self {
            Self {
                compiled: false,
                available: false,
                error: Some("panic in command: get_onnx_status".to_string()),
                ep_choice: String::new(),
            }
        }
    }

    impl PanicFallback for crate::nsf_hifigan_onnx::OnnxDiagnosticInfo {
        fn panic_fallback() -> Self {
            Self {
                compiled: false,
                available: false,
                error: Some("panic in command: get_onnx_diagnostic".to_string()),
                ep_choice: String::new(),
                active_ep: String::new(),
                onnx_version: None,
                providers: None,
                gpu_diagnostic: None,
            }
        }
    }

    impl PanicFallback for super::super::vslib_status::VslibStatusPayload {
        fn panic_fallback() -> Self {
            Self {
                compiled: false,
                available: false,
                version: None,
                error: Some("panic in command: get_vslib_status".to_string()),
            }
        }
    }

    impl PanicFallback for super::super::synth::ExportAudioDefaultsPayload {
        fn panic_fallback() -> Self {
            Self {
                ok: false,
                project_name: String::new(),
                documents_dir: String::new(),
                project_output_dir: String::new(),
                project_file_name: String::new(),
                separated_output_dir: String::new(),
                separated_file_name: String::new(),
                sample_rate: 0,
                bit_depth: 0,
                format: crate::encode::OutputFormat::default(),
                encoder: crate::encode::OutputSpec::default(),
            }
        }
    }

    impl PanicFallback for super::super::synth::ExportAudioPlanPayload {
        fn panic_fallback() -> Self {
            Self {
                ok: false,
                mode: super::super::synth::ExportAudioMode::Project,
                targets: vec![],
                existing_paths: vec![],
            }
        }
    }

    impl PanicFallback for Vec<crate::search::TranslitResult> {
        fn panic_fallback() -> Self {
            vec![]
        }
    }

    impl PanicFallback for Vec<super::super::processor_caps::ParamDescriptorDto> {
        fn panic_fallback() -> Self {
            vec![]
        }
    }

    impl PanicFallback for BTreeMap<String, String> {
        fn panic_fallback() -> Self {
            BTreeMap::new()
        }
    }
}

/// 同步 Tauri 命令的 panic 守卫（`serde_json::Value` 版）。
///
/// 【为什么必须接在每条同步命令上】Tauri 的同步命令在 WebView 的 IPC 线程上
/// **内联**执行 —— panic 会一路 unwind 穿过 Win32 消息循环的 `extern "system"`
/// 边界，Rust 1.71+ 语义下等于 abort 整个 App。async 命令已被 tokio 的
/// 任务边界结构性遏制，同步命令只能靠这层显式守卫。
pub(crate) fn guard_json_command(
    name: &str,
    f: impl FnOnce() -> serde_json::Value,
) -> serde_json::Value {
    guard_command(
        name,
        serde_json::json!({"ok": false, "error": format!("panic in command: {name}")}),
        f,
    )
}

/// `TimelineStatePayload` 的"命令失败"形态（panic 守卫与各处错误路径共用）。
///
/// 【为什么错误挂在 `missing_files`】本 payload 没有独立 error 字段，前端既有
/// 惯例是把失败信息放进 `missing_files`（见 midi 导入的 error_payload）——
/// panic 守卫沿用同一惯例，前端不需要新增分支。
pub(crate) fn timeline_error_payload(error: &str) -> crate::models::TimelineStatePayload {
    crate::models::TimelineStatePayload {
        ok: false,
        tracks: vec![],
        clips: vec![],
        created_clip_ids: None,
        created_track_ids: None,
        selected_track_id: None,
        selected_clip_id: None,
        bpm: 120.0,
        playhead_sec: 0.0,
        project_sec: None,
        project: None,
        missing_files: Some(vec![error.to_string()]),
        disabled_group_ids: vec![],
        tempo_map: None,
        undo_depth: None,
        redo_depth: None,
        notes_markdown: None,
        param_selection_restore: None,
    }
}

/// 波形命令的 panic 守卫。
///
/// 【为什么保留而未删除】它是 `guard_command` 的同款守卫，只是返回值类型不同
/// （`WaveformPeaksSegmentPayload` 而不是 `serde_json::Value`）。目前**没有调用点**
/// —— 也就是说波形命令实际上没有这层遏制。这是"写了安全网但没接线"，删除它等于
/// 把这个事实埋掉；保留 + 本注解把它留在明面上，等一次专门的判断（接线或明确删除）。
#[allow(
    dead_code,
    reason = "已实现但尚未接线的 panic 守卫；保留以暴露该缺口，见上方说明"
)]
pub(crate) fn guard_waveform_command(
    name: &str,
    f: impl FnOnce() -> super::waveform::WaveformPeaksSegmentPayload,
) -> super::waveform::WaveformPeaksSegmentPayload {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => {
            log::error!("command panicked: {name}");
            super::waveform::WaveformPeaksSegmentPayload {
                ok: false,
                min: vec![],
                max: vec![],
            }
        }
    }
}

pub(crate) fn ok_bool() -> serde_json::Value {
    serde_json::json!({ "ok": true })
}

/// 在系统文件管理器中定位一批路径：存在的文件被**多选高亮**，全是目录时打开
/// 第一个目录。
///
/// 【为什么走 Rust 而不是前端 `@tauri-apps/plugin-opener`】`opener:default` 实际
/// 只授予 `open-url` / `reveal-item-in-dir` / `default-urls`，**不含 `open_path`**；
/// 而 Rust 侧的 `OpenerExt` 调用不过 ACL。本仓既有 4 处同款做法
/// （`diagnostics.rs` 的日志目录与 zip、`cache.rs` 的缓存目录与导出产物）。
///
/// 【为什么先过滤不存在的项】`reveal_items_in_dir` 内部会 `canonicalize`，
/// 任何一个缺失路径都会让整批 reveal 失败 —— 分轨导出时个别目标被跳过 / 写入失败
/// 就会连累其余文件。
pub(crate) fn reveal_paths_in_file_manager(
    app: &tauri::AppHandle,
    paths: Vec<String>,
    empty_error: &str,
) -> serde_json::Value {
    use tauri_plugin_opener::OpenerExt;

    let mut files: Vec<PathBuf> = Vec::new();
    let mut dirs: Vec<PathBuf> = Vec::new();
    for raw in paths {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let path = PathBuf::from(trimmed);
        if path.is_file() {
            files.push(path);
        } else if path.is_dir() {
            dirs.push(path);
        }
    }

    if !files.is_empty() {
        return match app.opener().reveal_items_in_dir(files.iter()) {
            Ok(()) => serde_json::json!({ "ok": true, "count": files.len() }),
            Err(e) => {
                // 多选失败（个别文件管理器不支持）→ 至少打开首个文件的所在目录。
                if let Some(parent) = files[0].parent() {
                    let _ = app
                        .opener()
                        .open_path(parent.to_string_lossy(), None::<&str>);
                }
                serde_json::json!({
                    "ok": false,
                    "error": format!("reveal files failed: {e}"),
                })
            }
        };
    }

    if let Some(dir) = dirs.first() {
        return match app.opener().open_path(dir.to_string_lossy(), None::<&str>) {
            Ok(()) => serde_json::json!({ "ok": true, "path": dir.to_string_lossy() }),
            Err(e) => serde_json::json!({
                "ok": false,
                "error": format!("open folder failed: {e}"),
            }),
        };
    }

    serde_json::json!({ "ok": false, "error": empty_error })
}

/// 用系统默认程序打开一个路径（文件或目录）。
///
/// 与 [`reveal_paths_in_file_manager`] 同一 ACL 理由：必须走 Rust 侧 `OpenerExt`。
pub(crate) fn open_path_with_default_app(app: &tauri::AppHandle, path: &str) -> serde_json::Value {
    use tauri_plugin_opener::OpenerExt;

    let trimmed = path.trim();
    if trimmed.is_empty() {
        return serde_json::json!({ "ok": false, "error": "empty path" });
    }
    match app.opener().open_path(trimmed, None::<&str>) {
        Ok(()) => serde_json::json!({ "ok": true }),
        Err(e) => serde_json::json!({
            "ok": false,
            "error": format!("open path failed: {e}"),
        }),
    }
}

pub(crate) fn ensure_temp_dir() -> std::io::Result<PathBuf> {
    let dir = std::env::temp_dir().join("hifishifter");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub(crate) fn new_temp_wav_path(prefix: &str) -> Result<PathBuf, String> {
    let dir = ensure_temp_dir().map_err(|e| e.to_string())?;
    Ok(dir.join(format!("{}_{}.wav", prefix, Uuid::new_v4().simple())))
}

pub(crate) fn render_timeline_to_wav(
    state: &AppState,
    output_path: &Path,
    start_sec: f64,
    end_sec: Option<f64>,
) -> Result<crate::mixdown::MixdownResult, String> {
    let timeline = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    crate::mixdown::render_mixdown_to_file(
        &timeline,
        output_path,
        crate::mixdown::MixdownOptions {
            sample_rate: 44100,
            start_sec,
            end_sec,
            stretch: crate::time_stretch::resolved_external_stretch_algorithm(),
            apply_pitch_edit: true,
            // 临时渲染固定 32-bit float WAV（内部用途，不受导出设置影响）。
            output: crate::encode::OutputSpec::wav_32f(),
            quality_preset: crate::mixdown::QualityPreset::Export,
            cancel_flag: None,
            progress: None,
            cache_stats: None,
        },
    )
}
