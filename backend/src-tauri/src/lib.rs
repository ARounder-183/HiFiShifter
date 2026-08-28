mod audio_engine;
#[path = "audio/audio_utils.rs"]
mod audio_utils;
#[path = "pitch/clip_pitch_cache.rs"]
mod clip_pitch_cache;
#[path = "pitch/clip_rendering_state.rs"]
mod clip_rendering_state;
mod fade_curves;
pub(crate) mod commands;
mod formant_cache;
#[path = "audio/formant_morph.rs"]
mod formant_morph;
#[path = "audio/hifigan_tension.rs"]
mod hifigan_tension;
mod launch_args;
mod media;
#[path = "audio/mixdown.rs"]
mod mixdown;
mod models;
mod pitch_analysis;
#[path = "pitch/pitch_clip.rs"]
mod pitch_clip;
#[path = "pitch/pitch_config.rs"]
mod pitch_config;
mod pitch_editing;
#[path = "pitch/pitch_progress.rs"]
mod pitch_progress;
mod recording;
mod renderer;
mod synth_clip_cache;

#[cfg(feature = "onnx")]
#[path = "vocoder/ort_session.rs"]
mod vocoder_ort_session;

#[cfg(target_os = "windows")]
#[path = "vocoder/gpu_info.rs"]
mod gpu_info;

#[cfg(not(target_os = "windows"))]
#[path = "vocoder/gpu_info_stub.rs"]
mod gpu_info;

#[cfg(target_os = "windows")]
#[path = "vocoder/dml_adapters.rs"]
mod dml_adapters;

#[cfg(not(target_os = "windows"))]
#[path = "vocoder/dml_adapters_stub.rs"]
mod dml_adapters;

#[cfg(feature = "onnx")]
#[path = "vocoder/mel_utils.rs"]
mod mel_utils;

#[cfg(feature = "onnx")]
#[path = "vocoder/nsf_hifigan_onnx.rs"]
mod nsf_hifigan_onnx;
#[cfg(not(feature = "onnx"))]
#[path = "vocoder/nsf_hifigan_onnx_stub.rs"]
mod nsf_hifigan_onnx_stub;
#[cfg(not(feature = "onnx"))]
use nsf_hifigan_onnx_stub as nsf_hifigan_onnx;

#[cfg(feature = "onnx")]
#[path = "vocoder/hnsep_onnx.rs"]
mod hnsep_onnx;
#[cfg(not(feature = "onnx"))]
#[path = "vocoder/hnsep_onnx_stub.rs"]
mod hnsep_onnx_stub;
#[cfg(not(feature = "onnx"))]
use hnsep_onnx_stub as hnsep_onnx;

#[cfg(feature = "onnx")]
#[path = "vocoder/fcpe_onnx.rs"]
mod fcpe_onnx;
#[cfg(not(feature = "onnx"))]
#[path = "vocoder/fcpe_onnx_stub.rs"]
mod fcpe_onnx_stub;
#[cfg(not(feature = "onnx"))]
use fcpe_onnx_stub as fcpe_onnx;

mod config;
#[path = "audio/hfspeaks_v2.rs"]
mod hfspeaks_v2;
#[cfg(target_os = "linux")]
mod linux_clipboard;
#[path = "import/midi_import.rs"]
mod midi_import;
mod project;
mod project_fragment;
#[path = "import/reaper_export.rs"]
mod reaper_export;
#[path = "import/reaper_import.rs"]
mod reaper_import;
#[path = "import/reaper_parser.rs"]
mod reaper_parser;
#[path = "audio/soundtouch.rs"]
mod soundtouch;
#[path = "audio/sstretch.rs"]
mod sstretch;
mod state;
#[path = "vocoder/streaming_world.rs"]
mod streaming_world;
mod system_clipboard;
mod temp_manager;
#[path = "audio/time_stretch.rs"]
mod time_stretch;
#[path = "import/vocalshifter_clipboard.rs"]
mod vocalshifter_clipboard;
#[path = "import/vocalshifter_import.rs"]
mod vocalshifter_import;
#[cfg(all(feature = "vslib", target_os = "windows"))]
#[path = "vocoder/vslib.rs"]
mod vslib;
#[path = "vocoder/world_vocoder.rs"]
mod world_vocoder;

/// Internal pure-function exports used by integration tests (tests/).
///
/// Gated behind the `__test-internals` feature, which is part of `default`
/// so plain `cargo test` exercises the same code paths as CI
/// (`cargo test --features __test-internals`).
///
/// NOTE: this module used to exist so pure-function regressions could run
/// through integration targets because the lib unit-test harness could not
/// start on Windows (no manifest link channel). That limitation is gone —
/// `.cargo/config.toml` delay-loads comctl32.dll, so `cargo test --lib`
/// runs natively; the re-exports are kept for the integration targets.
#[cfg(feature = "__test-internals")]
pub mod __test_internals {
    pub use crate::pitch_clip::trim_and_resample_midi;
    // REAPER export round-trips: rate/multi-take export regressions run via
    // the integration targets (loop_semantics / reaper_export_rates).
    pub use crate::reaper_export::build_reaper_clipboard;
    pub use crate::reaper_parser::parse_clipboard_bytes;
    pub use crate::state::{
        Clip, SplitTransitionDurationUnit, SplitTransitionMode, SplitTransitionOptions,
        TimelineState,
    };

    /// Consumed playback window (forward [ss, ss+len·r) / reverse [se−len·r, se)).
    pub fn playback_window_sec(c: &Clip) -> (f64, f64) {
        crate::state::clip_playback_window_sec(c)
    }

    /// Directional leading silence (forward: window start; reverse: window
    /// end past the media end).
    pub fn leading_silence_sec(c: &Clip, media_total_sec: Option<f64>) -> f64 {
        crate::state::clip_leading_silence_sec(c, media_total_sec)
    }

    /// Window arguments for trim_and_resample_midi (non-loop reverse is
    /// redirected to [se−len·r, se]).
    pub fn pitch_trim_window_sec(c: &Clip) -> (f64, f64) {
        crate::state::clip_pitch_trim_window_sec(c)
    }
}

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tauri::Manager;

static NSF_HIFIGAN_MODEL_DIR: OnceLock<PathBuf> = OnceLock::new();
static HNSEP_MODEL_DIR: OnceLock<PathBuf> = OnceLock::new();
static FCPE_ONNX_PATH: OnceLock<PathBuf> = OnceLock::new();

pub fn nsf_hifigan_model_dir() -> Option<&'static Path> {
    NSF_HIFIGAN_MODEL_DIR.get().map(|p| p.as_path())
}

pub fn hnsep_model_dir() -> Option<&'static Path> {
    HNSEP_MODEL_DIR.get().map(|p| p.as_path())
}

pub fn fcpe_onnx_path() -> Option<&'static Path> {
    FCPE_ONNX_PATH.get().map(|p| p.as_path())
}

pub fn nsf_hifigan_onnx_probe() -> Result<String, String> {
    // Probe ONNX model availability.
    #[cfg(feature = "onnx")]
    {
        nsf_hifigan_onnx::probe_load().map(|_| "ok".to_string())
    }
    #[cfg(not(feature = "onnx"))]
    {
        Err("onnx feature disabled".to_string())
    }
}

/// Run the inference-device benchmark and return the serialized results.
/// Used by the in-app benchmark dialog and by the `--benchmark` CLI flag.
pub fn run_vocoder_benchmark_cli() -> Result<String, String> {
    #[cfg(feature = "onnx")]
    {
        let results = nsf_hifigan_onnx::run_benchmark()?;
        serde_json::to_string_pretty(&results)
            .map_err(|e| format!("failed to serialize benchmark results: {e}"))
    }
    #[cfg(not(feature = "onnx"))]
    {
        Err("onnx feature disabled".to_string())
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(state::AppState::default())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // ── AppImage Mesa/EGL driver path ──────────────────────────
            // When running inside an AppImage, Mesa's libEGL is bundled
            // along with its DRI drivers under usr/lib/dri/.  Tell Mesa
            // where to find them so WebKit2GTK can create its EGL display.
            #[cfg(target_os = "linux")]
            if let Ok(appdir) = std::env::var("APPDIR") {
                let dri_dir = format!("{appdir}/usr/lib/dri");
                if std::path::Path::new(&dri_dir).is_dir() {
                    std::env::set_var("LIBGL_DRIVERS_PATH", &dri_dir);
                    eprintln!("[setup] LIBGL_DRIVERS_PATH={dri_dir}");
                }
            }

            // 打包后的应用：从 resource_dir 查找内嵌的 ONNX 模型
            if let Ok(res_dir) = app.path().resource_dir() {
                let p = res_dir.join("models").join("nsf_hifigan");
                let has_model = p.join("pc_nsf_hifigan.onnx").exists()
                    || p.join("pc_nsf_hifigan_coreml.onnx").exists();
                if has_model && p.join("config.json").exists() {
                    let _ = NSF_HIFIGAN_MODEL_DIR.set(p);
                }
            }

            if let Ok(res_dir) = app.path().resource_dir() {
                let p = res_dir.join("models").join("hnsep");
                if p.join("hnsep.onnx").exists() {
                    let _ = HNSEP_MODEL_DIR.set(p);
                }
            }

            if let Ok(res_dir) = app.path().resource_dir() {
                let p = res_dir.join("models").join("fcpe").join("fcpe.onnx");
                if p.exists() {
                    let _ = FCPE_ONNX_PATH.set(p);
                }
            }

            let state = app.state::<state::AppState>();

            // 从进程启动参数中解析工程路径（双击文件关联场景）。
            let startup_project_path =
                launch_args::extract_project_path_from_args(std::env::args_os());
            state.set_pending_startup_project_path(startup_project_path);

            // Expose app handle for background workers.
            let _ = state.app_handle.set(app.handle().clone());

            // 将 app_handle 传递给 audio engine worker，使其能向前端推送事件。
            state.audio_engine.set_app_handle(app.handle().clone());

            // Prefer the OS-level app cache dir so peaks persist across runs.
            let base = app
                .path()
                .app_cache_dir()
                .unwrap_or_else(|_| hfspeaks_v2::default_cache_dir());
            let dir = base.join("hifishifter").join("waveform_peaks_cache");
            {
                let mut d = state
                    .waveform_cache_dir
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                *d = dir.clone();
            }
            let _ = hfspeaks_v2::ensure_cache_dir(&dir);

            // 加载持久化的最近工程列表
            if let Ok(cfg_base) = app.path().app_config_dir() {
                let cfg_dir = cfg_base.join("HiFiShifter");
                let _ = std::fs::create_dir_all(&cfg_dir);
                let recent = crate::config::load_recent(&cfg_dir);
                {
                    let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
                    p.recent = recent;
                }
                let _ = state.config_dir.set(cfg_dir);
            }

            // 启动即同步"为新的音频块启用循环"的进程级默认值：拖放导入、
            // 打开 v<4 工程的迁移等都可能在 get_ui_settings 之前发生，
            // 不能假设前端已先拉取过设置。
            if let Some(cfg_dir) = state.config_dir.get() {
                let ui = crate::config::load_ui_settings(cfg_dir);
                crate::config::set_loop_new_clips_default(ui.loop_new_clips);
                crate::config::set_sync_edits_across_takes(ui.sync_edits_across_takes);
            }

            // 尝试恢复上次运行时保存的窗口状态（非强制性）
            if let Some(cfg_dir) = state.config_dir.get() {
                if let Some(win) = app.get_webview_window("main") {
                    let ws = crate::config::load_window_state(cfg_dir);
                    // 应用尺寸与位置（非最大化/全屏状态先应用尺寸/位置，再切换最大化）
                    if let (Some(w), Some(h)) = (ws.width, ws.height) {
                        let _ = win.set_size(tauri::Size::Logical(tauri::LogicalSize {
                            width: w,
                            height: h,
                        }));
                    }
                    if let (Some(x), Some(y)) = (ws.x, ws.y) {
                        let _ =
                            win.set_position(tauri::Position::Logical(tauri::LogicalPosition {
                                x: x as f64,
                                y: y as f64,
                            }));
                    }
                    if ws.fullscreen.unwrap_or(false) {
                        let _ = win.set_fullscreen(true);
                    } else if ws.maximized.unwrap_or(false) {
                        let _ = win.maximize();
                    } else {
                        let _ = win.set_fullscreen(false);
                    }
                }
            }

            // 启动时清理上次遗留的临时文件（后台线程，不阻塞启动）
            temp_manager::cleanup_stale_temp_files();

            Ok(())
        })
        // 在窗口事件中监听 CloseRequested，保存窗口状态到配置目录
        .on_window_event(|win, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                // 仅针对主窗口保存状态
                if win.label() != "main" {
                    return;
                }

                let maximized = win.is_maximized().unwrap_or(false);
                let fullscreen = win.is_fullscreen().unwrap_or(false);
                let mut x_opt = None;
                let mut y_opt = None;
                let mut w_opt = None;
                let mut h_opt = None;
                if let Ok(pos) = win.outer_position() {
                    x_opt = Some(pos.x);
                    y_opt = Some(pos.y);
                }
                if let Ok(size) = win.inner_size() {
                    w_opt = Some(size.width as f64);
                    h_opt = Some(size.height as f64);
                }

                if let Some(cfg_dir) = win.app_handle().state::<state::AppState>().config_dir.get()
                {
                    let ws = crate::config::WindowState {
                        x: x_opt,
                        y: y_opt,
                        width: w_opt,
                        height: h_opt,
                        maximized: Some(maximized),
                        fullscreen: Some(fullscreen),
                    };
                    crate::config::save_window_state(cfg_dir, &ws);
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::ping,
            commands::get_runtime_info,
            commands::consume_startup_project_path,
            commands::set_ui_locale,
            commands::get_timeline_state,
            commands::get_timeline_state_lite,
            commands::set_transport,
            commands::close_window,
            commands::undo_timeline,
            commands::redo_timeline,
            commands::begin_undo_group,
            commands::end_undo_group,
            commands::get_project_meta,
            commands::new_project,
            commands::open_project_dialog,
            commands::open_project,
            commands::import_project_dialog,
            commands::import_project,
            commands::save_project,
            commands::save_project_as,
            commands::save_project_to_path,
            commands::get_auto_backup_settings,
            commands::save_auto_backup_settings,
            commands::run_timed_auto_backup,
            commands::get_recording_settings,
            commands::save_recording_settings,
            commands::get_recording_devices,
            commands::get_recording_apps,
            commands::start_recording,
            commands::stop_recording,
            commands::get_recording_state,
            commands::set_project_base_scale,
            commands::set_project_custom_scale,
            commands::set_project_stretch_settings,
            commands::set_project_timeline_settings,
            commands::set_timeline_tempo_map,
            commands::open_audio_dialog,
            commands::open_audio_dialog_multi,
            commands::open_audio_dialog_for_source,
            commands::get_media_audio_streams,
            commands::pick_output_path,
            commands::pick_directory,
            commands::open_midi_dialog,
            commands::get_root_mix_waveform_peaks_segment,
            commands::get_track_mix_waveform_peaks_segment,
            commands::clear_waveform_cache,
            commands::get_waveform_mipmap_binary,
            commands::preload_waveform_mipmap,
            commands::batch_get_waveform_mipmap,
            commands::get_waveform_manifest,
            commands::get_waveform_tiles_binary,
            commands::import_audio_item,
            commands::import_audio_bytes,
            commands::add_track,
            commands::remove_track,
            commands::duplicate_track,
            commands::move_track,
            commands::set_track_state,
            commands::select_track,
            commands::set_project_length,
            commands::get_track_summary,
            commands::get_param_frames,
            commands::set_param_frames,
            commands::restore_param_frames,
            commands::add_clip,
            commands::create_clips_bulk,
            commands::get_static_param,
            commands::set_static_param,
            commands::remove_clip,
            commands::remove_clips,
            commands::move_clip,
            commands::move_clips,
            commands::get_clip_linked_params,
            commands::apply_clip_linked_params,
            commands::set_clip_state,
            commands::set_clips_state_bulk,
            commands::set_clip_active_take,
            commands::cycle_clip_takes,
            commands::pack_clips_into_takes,
            commands::explode_clip_takes,
            commands::duplicate_clip_take,
            commands::remove_clip_take,
            commands::rename_clip_take,
            commands::add_clip_take_from_media,
            commands::import_media_files_as_takes,
            commands::duplicate_clips_bulk,
            commands::replace_clip_source,
            commands::check_source_files_changed,
            commands::search_source_file_replacements,
            commands::split_clip,
            commands::split_clips_at,
            commands::glue_clips,
            commands::group_clips,
            commands::ungroup_clips,
            commands::toggle_group_disabled,
            commands::convert_clips_to_pitch_reference,
            commands::update_pitch_reference,
            commands::select_clip,
            commands::copy_timeline_clips,
            commands::copy_timeline_tracks,
            commands::paste_timeline_clipboard,
            commands::has_timeline_clipboard,
            commands::write_system_clipboard_object,
            commands::read_system_clipboard_object,
            commands::load_default_model,
            commands::load_model,
            commands::set_pitch_shift,
            commands::process_audio,
            commands::synthesize,
            commands::save_synthesized,
            commands::save_separated,
            commands::export_audio_advanced,
            commands::cancel_export_audio,
            commands::get_export_audio_defaults,
            commands::preview_export_audio_plan,
            commands::quick_export_selected_clips,
            commands::play_original,
            commands::stop_audio,
            commands::get_playback_state,
            commands::start_background_render,
            commands::cancel_background_render,
            commands::debug_realtime_render_stats,
            commands::get_pitch_analysis_progress,
            commands::get_onnx_status,
            commands::get_onnx_diagnostic,
            commands::run_vocoder_benchmark,
            commands::get_gpu_devices,
            commands::get_dml_adapters,
            commands::clear_pitch_cache,
            commands::get_pitch_cache_stats,
            commands::list_directory,
            commands::get_audio_file_info,
            commands::read_audio_preview,
            commands::search_files_recursive,
            commands::open_vocalshifter_dialog,
            commands::import_vocalshifter_project,
            commands::paste_vocalshifter_clipboard,
            commands::open_reaper_dialog,
            commands::import_reaper_project,
            commands::paste_reaper_clipboard,
            commands::has_reaper_clipboard,
            commands::clear_cache,
            commands::get_processor_params,
            commands::get_midi_tracks,
            commands::read_midi_clipboard_to_memory,
            commands::import_midi_to_pitch,
            commands::import_midi_as_clip,
            commands::replace_midi_clip_data,
            commands::pick_midi_output_path,
            commands::export_pitch_to_midi,
            commands::get_ui_settings,
            commands::save_ui_settings,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                // Shut down audio engine: stop meter thread, send Shutdown to
                // worker threads, and drop the channel sender so all worker
                // threads exit their recv loops.
                let state = app_handle.state::<state::AppState>();
                crate::recording::shutdown(state.inner());
                state.audio_engine.shutdown();

                // Force-drop all ONNX sessions to release GPU memory before exit.
                crate::nsf_hifigan_onnx::drop_shared_session();
                crate::fcpe_onnx::drop_shared_session();
                crate::hnsep_onnx::drop_shared_session();
            }
        });
}
