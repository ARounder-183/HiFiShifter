mod audio_engine;
#[path = "audio/audio_utils.rs"]
mod audio_utils;
#[path = "audio/ffmpeg_utils.rs"]
mod ffmpeg_utils;
#[path = "pitch/clip_pitch_cache.rs"]
mod clip_pitch_cache;
#[path = "pitch/clip_rendering_state.rs"]
mod clip_rendering_state;
mod commands;
#[path = "audio/hifigan_tension.rs"]
mod hifigan_tension;
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
mod renderer;
mod synth_clip_cache;

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

mod config;
#[path = "audio/hfspeaks_v2.rs"]
mod hfspeaks_v2;
#[path = "import/midi_import.rs"]
mod midi_import;
mod project;
#[path = "import/reaper_import.rs"]
mod reaper_import;
#[path = "import/reaper_parser.rs"]
mod reaper_parser;
#[path = "audio/sstretch.rs"]
mod sstretch;
mod state;
#[path = "vocoder/streaming_world.rs"]
mod streaming_world;
mod temp_manager;
#[path = "audio/time_stretch.rs"]
mod time_stretch;
#[path = "import/vocalshifter_clipboard.rs"]
mod vocalshifter_clipboard;
#[path = "import/vocalshifter_import.rs"]
mod vocalshifter_import;
#[cfg(feature = "vslib")]
#[path = "vocoder/vslib.rs"]
mod vslib;
#[path = "audio/waveform.rs"]
mod waveform;
#[path = "audio/waveform_disk_cache.rs"]
mod waveform_disk_cache;
#[path = "vocoder/world.rs"]
mod world;
#[path = "vocoder/world_vocoder.rs"]
mod world_vocoder;

use tauri::Manager;

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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(state::AppState::default())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // 打包后的应用：从 resource_dir 查找内嵌的 ONNX 模型
            if std::env::var_os("HIFISHIFTER_NSF_HIFIGAN_MODEL_DIR").is_none() {
                if let Ok(res_dir) = app.path().resource_dir() {
                    let p = res_dir.join("models").join("nsf_hifigan");
                    if p.join("pc_nsf_hifigan.onnx").exists() && p.join("config.json").exists() {
                        std::env::set_var("HIFISHIFTER_NSF_HIFIGAN_MODEL_DIR", &p);
                    }
                }
            }

            if std::env::var_os("HIFISHIFTER_HNSEP_MODEL_DIR").is_none() {
                if let Ok(res_dir) = app.path().resource_dir() {
                    let p = res_dir.join("models").join("hnsep");
                    if p.join("hnsep.onnx").exists() {
                        std::env::set_var("HIFISHIFTER_HNSEP_MODEL_DIR", &p);
                    }
                }
            }

            let state = app.state::<state::AppState>();

            // Expose app handle for background workers.
            let _ = state.app_handle.set(app.handle().clone());

            // 将 app_handle 传递给 audio engine worker，使其能向前端推送事件。
            state.audio_engine.set_app_handle(app.handle().clone());

            // Prefer the OS-level app cache dir so peaks persist across runs.
            let base = app
                .path()
                .app_cache_dir()
                .unwrap_or_else(|_| waveform_disk_cache::default_cache_dir());
            let dir = base.join("hifishifter").join("waveform_peaks_cache");
            {
                let mut d = state
                    .waveform_cache_dir
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                *d = dir.clone();
            }
            let _ = waveform_disk_cache::ensure_dir(&dir);

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

            // 启动时清理上次遗留的临时文件（后台线程，不阻塞启动）
            temp_manager::cleanup_stale_temp_files();

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::ping,
            commands::get_runtime_info,
            commands::set_ui_locale,
            commands::get_timeline_state,
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
            commands::save_project,
            commands::save_project_as,
            commands::set_project_base_scale,
            commands::set_project_custom_scale,
            commands::set_project_timeline_settings,
            commands::open_audio_dialog,
            commands::open_audio_dialog_multi,
            commands::pick_output_path,
            commands::pick_directory,
            commands::open_midi_dialog,
            commands::get_root_mix_waveform_peaks_segment,
            commands::get_track_mix_waveform_peaks_segment,
            commands::clear_waveform_cache,
            commands::get_waveform_mipmap_binary,
            commands::preload_waveform_mipmap,
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
            commands::get_static_param,
            commands::set_static_param,
            commands::remove_clip,
            commands::move_clip,
            commands::move_clips,
            commands::get_clip_linked_params,
            commands::apply_clip_linked_params,
            commands::set_clip_state,
            commands::replace_clip_source,
            commands::split_clip,
            commands::glue_clips,
            commands::select_clip,
            commands::load_default_model,
            commands::load_model,
            commands::set_pitch_shift,
            commands::process_audio,
            commands::synthesize,
            commands::save_synthesized,
            commands::save_separated,
            commands::play_original,
            commands::stop_audio,
            commands::get_playback_state,
            commands::debug_realtime_render_stats,
            commands::get_pitch_analysis_progress,
            commands::get_onnx_status,
            commands::get_onnx_diagnostic,
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
            commands::clear_cache,
            commands::get_processor_params,
            commands::get_midi_tracks,
            commands::import_midi_to_pitch,
            commands::get_ui_settings,
            commands::save_ui_settings // TODO: 异步音高刷新命令暂时禁用，等待基础设施完成
                                       // commands::start_pitch_refresh_task,
                                       // commands::get_pitch_refresh_status,
                                       // commands::cancel_pitch_task
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
