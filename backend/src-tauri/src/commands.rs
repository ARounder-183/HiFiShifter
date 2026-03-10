// 命令门面（Facade）
//
// 约束：
// - `#[tauri::command]` 只允许出现在本文件中（作为前端 invoke 的稳定入口）。
// - 具体实现按领域拆分在 `backend/src-tauri/src/commands/*.rs`，并通过本文件转发调用。
// - 拆分模块中的函数请保持 `pub(super)` / `pub(crate)`，避免被当成公共 API 直接依赖。

#[path = "commands/common.rs"]
mod common;
#[path = "commands/core.rs"]
mod core;
#[path = "commands/dialogs.rs"]
mod dialogs;
#[path = "commands/debug.rs"]
mod debug;
#[path = "commands/params.rs"]
mod params;
#[path = "commands/playback.rs"]
mod playback;
#[path = "commands/project.rs"]
mod project;
#[path = "commands/synth.rs"]
mod synth;
#[path = "commands/timeline.rs"]
mod timeline;
#[path = "commands/waveform.rs"]
mod waveform;
#[path = "commands/pitch_progress.rs"]
mod pitch_progress;
#[path = "commands/onnx_status.rs"]
mod onnx_status;
#[path = "commands/pitch_cache.rs"]
mod pitch_cache;
#[path = "commands/file_browser.rs"]
mod file_browser;
#[path = "commands/vocalshifter.rs"]
mod vocalshifter;
#[path = "commands/vocalshifter_clipboard.rs"]
mod vocalshifter_clipboard;
#[path = "commands/reaper.rs"]
mod reaper;
#[path = "commands/reaper_clipboard.rs"]
mod reaper_clipboard;
#[path = "commands/cache.rs"]
mod cache;
#[path = "commands/processor_caps.rs"]
mod processor_caps;
#[path = "commands/resampler_registry.rs"]
mod resampler_registry;
#[path = "commands/midi.rs"]
mod midi;
// TODO: 异步音高刷新功能未完成，缺少必要的状态管理和依赖
// #[path = "commands/pitch_refresh_async.rs"]
// mod pitch_refresh_async;

use crate::state::AppState;
use tauri::{State, Window};

// This is used by the window close handler (crate-internal), not a tauri command.
pub(crate) use project::save_project_to_path_inner;

// ===================== core =====================

#[tauri::command(rename_all = "camelCase")]
pub fn ping() -> serde_json::Value {
    core::ping()
}

#[tauri::command(rename_all = "camelCase")]
pub fn get_runtime_info(state: State<'_, AppState>) -> crate::models::RuntimeInfoPayload {
    core::get_runtime_info(state)
}

#[tauri::command(rename_all = "camelCase")]
pub fn set_ui_locale(state: State<'_, AppState>, locale: String) -> serde_json::Value {
    core::set_ui_locale(state, locale)
}

#[tauri::command(rename_all = "camelCase")]
pub fn get_timeline_state(state: State<'_, AppState>) -> crate::models::TimelineStatePayload {
    core::get_timeline_state(state)
}

#[tauri::command(rename_all = "camelCase")]
pub fn set_transport(
    state: State<'_, AppState>,
    playhead_sec: Option<f64>,
    bpm: Option<f64>,
) -> serde_json::Value {
    core::set_transport(state, playhead_sec, bpm)
}
#[tauri::command(rename_all = "camelCase")]
pub fn undo_timeline(state: State<'_, AppState>) -> crate::models::TimelineStatePayload {
    core::undo_timeline(state)
}

#[tauri::command(rename_all = "camelCase")]
pub fn redo_timeline(state: State<'_, AppState>) -> crate::models::TimelineStatePayload {
    core::redo_timeline(state)
}

// ===================== project =====================

#[tauri::command(rename_all = "camelCase")]
pub fn close_window(window: Window) -> serde_json::Value {
    project::close_window(window)
}

#[tauri::command(rename_all = "camelCase")]
pub fn get_project_meta(state: State<'_, AppState>) -> crate::models::ProjectMetaPayload {
    project::get_project_meta(state)
}

#[tauri::command(rename_all = "camelCase")]
pub fn new_project(state: State<'_, AppState>, window: Window) -> crate::models::TimelineStatePayload {
    project::new_project(state, window)
}

#[tauri::command(rename_all = "camelCase")]
pub fn open_project_dialog() -> serde_json::Value {
    project::open_project_dialog()
}

#[tauri::command(rename_all = "camelCase")]
pub fn open_project(
    state: State<'_, AppState>,
    window: Window,
    project_path: String,
) -> crate::models::TimelineStatePayload {
    project::open_project(state, window, project_path)
}

#[tauri::command(rename_all = "camelCase")]
pub fn save_project(state: State<'_, AppState>, window: Window) -> serde_json::Value {
    project::save_project(state, window)
}

#[tauri::command(rename_all = "camelCase")]
pub fn save_project_as(state: State<'_, AppState>, window: Window) -> serde_json::Value {
    project::save_project_as(state, window)
}

// ===================== dialogs =====================

#[tauri::command(rename_all = "camelCase")]
pub fn open_audio_dialog() -> serde_json::Value {
    dialogs::open_audio_dialog()
}

#[tauri::command(rename_all = "camelCase")]
pub fn pick_output_path() -> serde_json::Value {
    dialogs::pick_output_path()
}

#[tauri::command(rename_all = "camelCase")]
pub fn pick_directory() -> serde_json::Value {
    dialogs::pick_directory()
}

#[tauri::command(rename_all = "camelCase")]
pub fn open_midi_dialog() -> serde_json::Value {
    dialogs::open_midi_dialog()
}

// ===================== waveform =====================

#[tauri::command(rename_all = "camelCase")]
pub fn get_waveform_peaks_segment(
    state: State<'_, AppState>,
    source_path: String,
    start_sec: f64,
    duration_sec: f64,
    columns: usize,
) -> crate::waveform::WaveformPeaksSegmentPayload {
    waveform::get_waveform_peaks_segment(state, source_path, start_sec, duration_sec, columns)
}

#[tauri::command(rename_all = "camelCase")]
pub fn get_root_mix_waveform_peaks_segment(
    state: State<'_, AppState>,
    track_id: String,
    start_sec: f64,
    duration_sec: f64,
    columns: usize,
) -> crate::waveform::WaveformPeaksSegmentPayload {
    waveform::get_root_mix_waveform_peaks_segment(state, track_id, start_sec, duration_sec, columns)
}

#[tauri::command(rename_all = "camelCase")]
pub fn get_track_mix_waveform_peaks_segment(
    state: State<'_, AppState>,
    track_id: String,
    start_sec: f64,
    duration_sec: f64,
    columns: usize,
) -> crate::waveform::WaveformPeaksSegmentPayload {
    waveform::get_track_mix_waveform_peaks_segment(state, track_id, start_sec, duration_sec, columns)
}

#[tauri::command(rename_all = "camelCase")]
pub fn clear_waveform_cache(state: State<'_, AppState>) -> serde_json::Value {
    waveform::clear_waveform_cache(state)
}

// ===================== timeline =====================

#[tauri::command(rename_all = "camelCase")]
pub fn import_audio_item(
    state: State<'_, AppState>,
    audio_path: String,
    track_id: Option<Option<String>>,
    start_sec: Option<f64>,
) -> crate::models::TimelineStatePayload {
    timeline::import_audio_item(state, audio_path, track_id, start_sec)
}
#[tauri::command(rename_all = "camelCase")]
pub fn import_audio_bytes(
    state: State<'_, AppState>,
    file_name: String,
    base64_data: String,
    track_id: Option<Option<String>>,
    start_sec: Option<f64>,
) -> crate::models::TimelineStatePayload {
    timeline::import_audio_bytes(state, file_name, base64_data, track_id, start_sec)
}
#[tauri::command(rename_all = "camelCase")]
pub fn add_track(
    state: State<'_, AppState>,
    name: Option<String>,
    parent_track_id: Option<String>,
    index: Option<usize>,
) -> crate::models::TimelineStatePayload {
    timeline::add_track(state, name, parent_track_id, index)
}

#[tauri::command(rename_all = "camelCase")]
pub fn remove_track(state: State<'_, AppState>, track_id: String) -> crate::models::TimelineStatePayload {
    timeline::remove_track(state, track_id)
}

#[tauri::command(rename_all = "camelCase")]
pub fn move_track(
    state: State<'_, AppState>,
    track_id: String,
    target_index: usize,
    parent_track_id: Option<String>,
) -> crate::models::TimelineStatePayload {
    timeline::move_track(state, track_id, target_index, parent_track_id)
}

#[tauri::command(rename_all = "camelCase")]
pub fn set_track_state(
    state: State<'_, AppState>,
    track_id: String,
    muted: Option<bool>,
    solo: Option<bool>,
    volume: Option<f32>,
    compose_enabled: Option<bool>,
    pitch_analysis_algo: Option<String>,
    color: Option<String>,
) -> crate::models::TimelineStatePayload {
    timeline::set_track_state(
        state,
        track_id,
        muted,
        solo,
        volume,
        compose_enabled,
        pitch_analysis_algo,
        color,
    )
}

#[tauri::command(rename_all = "camelCase")]
pub fn select_track(state: State<'_, AppState>, track_id: String) -> crate::models::TimelineStatePayload {
    timeline::select_track(state, track_id)
}

#[tauri::command(rename_all = "camelCase")]
pub fn set_project_length(
    state: State<'_, AppState>,
    project_sec: f64,
) -> crate::models::TimelineStatePayload {
    timeline::set_project_length(state, project_sec)
}
#[tauri::command(rename_all = "camelCase")]
pub fn get_track_summary(state: State<'_, AppState>, track_id: Option<String>) -> serde_json::Value {
    timeline::get_track_summary(state, track_id)
}

#[tauri::command(rename_all = "camelCase")]
pub fn add_clip(
    state: State<'_, AppState>,
    track_id: Option<String>,
    name: Option<String>,
    start_sec: Option<f64>,
    length_sec: Option<f64>,
    source_path: Option<String>,
) -> crate::models::TimelineStatePayload {
    timeline::add_clip(state, track_id, name, start_sec, length_sec, source_path)
}
#[tauri::command(rename_all = "camelCase")]
pub fn remove_clip(state: State<'_, AppState>, clip_id: String) -> crate::models::TimelineStatePayload {
    timeline::remove_clip(state, clip_id)
}

#[tauri::command(rename_all = "camelCase")]
pub fn move_clip(
    state: State<'_, AppState>,
    clip_id: String,
    start_sec: f64,
    track_id: Option<String>,
) -> crate::models::TimelineStatePayload {
    timeline::move_clip(state, clip_id, start_sec, track_id)
}
#[tauri::command(rename_all = "camelCase")]
#[allow(clippy::too_many_arguments)]
pub fn set_clip_state(
    state: State<'_, AppState>,
    clip_id: String,
    name: Option<String>,
    start_sec: Option<f64>,
    length_sec: Option<f64>,
    gain: Option<f32>,
    muted: Option<bool>,
    source_start_sec: Option<f64>,
    source_end_sec: Option<f64>,
    playback_rate: Option<f32>,
    fade_in_sec: Option<f64>,
    fade_out_sec: Option<f64>,
    fade_in_curve: Option<String>,
    fade_out_curve: Option<String>,
    color: Option<String>,
) -> crate::models::TimelineStatePayload {
    timeline::set_clip_state(
        state,
        clip_id,
        name,
        start_sec,
        length_sec,
        gain,
        muted,
        source_start_sec,
        source_end_sec,
        playback_rate,
        fade_in_sec,
        fade_out_sec,
        fade_in_curve,
        fade_out_curve,
        color,
    )
}
#[tauri::command(rename_all = "camelCase")]
pub fn split_clip(
    state: State<'_, AppState>,
    clip_id: String,
    split_sec: f64,
) -> crate::models::TimelineStatePayload {
    timeline::split_clip(state, clip_id, split_sec)
}
#[tauri::command(rename_all = "camelCase")]
pub fn glue_clips(
    state: State<'_, AppState>,
    clip_ids: Vec<String>,
) -> crate::models::TimelineStatePayload {
    timeline::glue_clips(state, clip_ids)
}

#[tauri::command(rename_all = "camelCase")]
pub fn select_clip(
    state: State<'_, AppState>,
    clip_id: Option<String>,
) -> crate::models::TimelineStatePayload {
    timeline::select_clip(state, clip_id)
}

// ===================== params =====================

#[tauri::command(rename_all = "camelCase")]
pub fn get_param_frames(
    state: State<'_, AppState>,
    track_id: String,
    param: String,
    start_frame: u32,
    frame_count: u32,
    stride: Option<u32>,
) -> crate::models::ParamFramesPayload {
    params::get_param_frames(state, track_id, param, start_frame, frame_count, stride)
}

#[tauri::command(rename_all = "camelCase")]
pub fn set_param_frames(
    state: State<'_, AppState>,
    track_id: String,
    param: String,
    start_frame: u32,
    values: Vec<f32>,
    checkpoint: Option<bool>,
) -> serde_json::Value {
    params::set_param_frames(state, track_id, param, start_frame, values, checkpoint)
}

#[tauri::command(rename_all = "camelCase")]
pub fn restore_param_frames(
    state: State<'_, AppState>,
    track_id: String,
    param: String,
    start_frame: u32,
    frame_count: u32,
    checkpoint: Option<bool>,
) -> serde_json::Value {
    params::restore_param_frames(state, track_id, param, start_frame, frame_count, checkpoint)
}

#[tauri::command(rename_all = "camelCase")]
pub fn get_static_param(
    state: State<'_, AppState>,
    track_id: String,
    param: String,
) -> crate::models::StaticParamValuePayload {
    params::get_static_param(state, track_id, param)
}

#[tauri::command(rename_all = "camelCase")]
pub fn set_static_param(
    state: State<'_, AppState>,
    track_id: String,
    param: String,
    value: f64,
    checkpoint: Option<bool>,
) -> serde_json::Value {
    params::set_static_param(state, track_id, param, value, checkpoint)
}

// ===================== synth =====================

#[tauri::command(rename_all = "camelCase")]
pub fn load_default_model(state: State<'_, AppState>) -> crate::models::ModelConfigPayload {
    synth::load_default_model(state)
}

#[tauri::command(rename_all = "camelCase")]
pub fn load_model(state: State<'_, AppState>, model_dir: String) -> crate::models::ModelConfigPayload {
    synth::load_model(state, model_dir)
}

#[tauri::command(rename_all = "camelCase")]
pub fn set_pitch_shift(semitones: f64) -> serde_json::Value {
    synth::set_pitch_shift(semitones)
}

#[tauri::command(rename_all = "camelCase")]
pub fn process_audio(state: State<'_, AppState>, audio_path: String) -> crate::models::ProcessAudioPayload {
    synth::process_audio(state, audio_path)
}

#[tauri::command(rename_all = "camelCase")]
pub fn synthesize(state: State<'_, AppState>) -> crate::models::SynthesizePayload {
    synth::synthesize(state)
}

#[tauri::command(rename_all = "camelCase")]
pub fn save_synthesized(state: State<'_, AppState>, output_path: String) -> serde_json::Value {
    synth::save_synthesized(state, output_path)
}

#[tauri::command(rename_all = "camelCase")]
pub fn save_separated(state: State<'_, AppState>, output_dir: String) -> serde_json::Value {
    synth::save_separated(state, output_dir)
}

// ===================== playback =====================

#[tauri::command(rename_all = "camelCase")]
pub fn play_original(state: State<'_, AppState>, start_sec: f64) -> serde_json::Value {
    playback::play_original(state, start_sec)
}

#[tauri::command(rename_all = "camelCase")]
pub fn stop_audio(state: State<'_, AppState>) -> serde_json::Value {
    playback::stop_audio(state)
}

#[tauri::command(rename_all = "camelCase")]
pub fn get_playback_state(state: State<'_, AppState>) -> crate::models::PlaybackStatePayload {
    playback::get_playback_state(state)
}

// ===================== debug =====================

#[tauri::command(rename_all = "camelCase")]
pub fn debug_realtime_render_stats(
    state: State<'_, AppState>,
) -> crate::models::DebugRealtimeRenderStatsPayload {
    debug::debug_realtime_render_stats(state)
}

// ===================== pitch_progress =====================

#[tauri::command(rename_all = "camelCase")]
pub fn get_pitch_analysis_progress(
    state: State<'_, AppState>,
) -> Result<Option<crate::pitch_analysis::PitchProgressPayload>, String> {
    pitch_progress::get_pitch_analysis_progress(state)
}

// ===================== onnx_status =====================

#[tauri::command(rename_all = "camelCase")]
pub fn get_onnx_status() -> onnx_status::OnnxStatusPayload {
    onnx_status::get_onnx_status()
}

#[tauri::command(rename_all = "camelCase")]
pub fn get_onnx_diagnostic() -> crate::nsf_hifigan_onnx::OnnxDiagnosticInfo {
    onnx_status::get_onnx_diagnostic_info()
}

// ===================== pitch_cache =====================

#[tauri::command(rename_all = "camelCase")]
pub fn clear_pitch_cache(state: State<'_, AppState>) -> serde_json::Value {
    pitch_cache::clear_pitch_cache(state)
}

#[tauri::command(rename_all = "camelCase")]
pub fn get_pitch_cache_stats(state: State<'_, AppState>) -> pitch_cache::PitchCacheStatsPayload {
    pitch_cache::get_pitch_cache_stats(state)
}

// ===================== file_browser =====================

#[tauri::command(rename_all = "camelCase")]
pub fn list_directory(dir_path: String) -> Result<Vec<file_browser::FileEntry>, String> {
    file_browser::list_directory(dir_path)
}

#[tauri::command(rename_all = "camelCase")]
pub fn get_audio_file_info(file_path: String) -> Result<file_browser::AudioFileInfo, String> {
    file_browser::get_audio_file_info(file_path)
}

#[tauri::command(rename_all = "camelCase")]
pub fn read_audio_preview(
    file_path: String,
    max_frames: Option<u32>,
) -> Result<file_browser::AudioPreviewData, String> {
    file_browser::read_audio_preview(file_path, max_frames)
}

#[tauri::command(rename_all = "camelCase")]
pub fn search_files_recursive(
    dir_path: String,
    query: String,
) -> Result<Vec<file_browser::FileEntry>, String> {
    file_browser::search_files_recursive(dir_path, query)
}

// ===================== vocalshifter =====================

#[tauri::command(rename_all = "camelCase")]
pub fn open_vocalshifter_dialog() -> serde_json::Value {
    vocalshifter::open_vocalshifter_dialog()
}

#[tauri::command(rename_all = "camelCase")]
pub fn import_vocalshifter_project(
    state: State<'_, AppState>,
    window: Window,
    vsp_path: String,
) -> serde_json::Value {
    vocalshifter::import_vocalshifter_project(state.inner(), &window, vsp_path)
}

#[tauri::command(rename_all = "camelCase")]
pub fn paste_vocalshifter_clipboard(state: State<'_, AppState>) -> serde_json::Value {
    vocalshifter_clipboard::paste_vocalshifter_clipboard(state.inner())
}

// ===================== reaper =====================

#[tauri::command(rename_all = "camelCase")]
pub fn open_reaper_dialog() -> serde_json::Value {
    reaper::open_reaper_dialog()
}

#[tauri::command(rename_all = "camelCase")]
pub fn import_reaper_project(
    state: State<'_, AppState>,
    window: Window,
    rpp_path: String,
) -> serde_json::Value {
    reaper::import_reaper_project(state.inner(), &window, rpp_path)
}

#[tauri::command(rename_all = "camelCase")]
pub fn paste_reaper_clipboard(state: State<'_, AppState>) -> serde_json::Value {
    reaper_clipboard::paste_reaper_clipboard(state.inner())
}

// ===================== cache =====================

#[tauri::command(rename_all = "camelCase")]
pub fn clear_cache(state: State<'_, AppState>) -> Result<u64, String> {
    cache::clear_cache(state)
}

// ===================== processor_caps =====================

#[tauri::command(rename_all = "camelCase")]
pub fn get_processor_params(algo: String) -> Vec<processor_caps::ParamDescriptorDto> {
    processor_caps::get_processor_params(algo)
}

// ===================== resampler_registry =====================

#[tauri::command(rename_all = "camelCase")]
pub fn list_resamplers(state: tauri::State<'_, crate::state::AppState>) -> resampler_registry::ResamplerListPayload {
    resampler_registry::list_resamplers(state)
}

#[tauri::command(rename_all = "camelCase")]
pub fn add_resampler(
    state: tauri::State<'_, crate::state::AppState>,
    display_name: String,
    exe_path: String,
    default_flags: Option<String>,
    flag_params: Option<Vec<resampler_registry::FlagParamPayload>>,
) -> resampler_registry::ResamplerOpPayload {
    resampler_registry::add_resampler(state, display_name, exe_path, default_flags, flag_params)
}

#[tauri::command(rename_all = "camelCase")]
pub fn remove_resampler(
    state: tauri::State<'_, crate::state::AppState>,
    id: String,
) -> resampler_registry::ResamplerOpPayload {
    resampler_registry::remove_resampler(state, id)
}

#[tauri::command(rename_all = "camelCase")]
pub fn update_resampler(
    state: tauri::State<'_, crate::state::AppState>,
    id: String,
    display_name: Option<String>,
    exe_path: Option<String>,
    default_flags: Option<String>,
    flag_params: Option<Vec<resampler_registry::FlagParamPayload>>,
) -> resampler_registry::ResamplerOpPayload {
    resampler_registry::update_resampler(state, id, display_name, exe_path, default_flags, flag_params)
}

#[tauri::command(rename_all = "camelCase")]
pub fn scan_resamplers(
    state: tauri::State<'_, crate::state::AppState>,
    directory: String,
) -> resampler_registry::ResamplerListPayload {
    resampler_registry::scan_resamplers(state, directory)
}

#[tauri::command(rename_all = "camelCase")]
pub async fn browse_resampler_exe() -> Option<String> {
    resampler_registry::browse_resampler_exe().await
}

// ===================== midi =====================

#[tauri::command(rename_all = "camelCase")]
pub fn get_midi_tracks(midi_path: String) -> serde_json::Value {
    midi::get_midi_tracks(midi_path)
}

#[tauri::command(rename_all = "camelCase")]
pub fn import_midi_to_pitch(
    state: State<'_, AppState>,
    midi_path: String,
    track_index: Option<usize>,
    offset_sec: Option<f64>,
) -> serde_json::Value {
    midi::import_midi_to_pitch(state.inner(), midi_path, track_index, offset_sec)
}

// ===================== pitch_refresh_async (暂时禁用) =====================
// TODO: 需要实现以下功能才能启用：
// 1. 在 state.rs 中添加 PitchTaskInfo 和 PitchTaskStatus 类型
// 2. 在 AppState 中添加 pitch_refresh_tasks 字段
// 3. 在 Cargo.toml 中添加 tokio 依赖
// 4. 将 pitch_analysis.rs 中的相关函数改为 pub
//
// #[tauri::command(rename_all = "camelCase")]
// pub async fn start_pitch_refresh_task(
//     root_track_id: String,
//     state: State<'_, AppState>,
// ) -> Result<String, String> {
//     pitch_refresh_async::start_pitch_refresh_task(root_track_id, state).await
// }
//
// #[tauri::command(rename_all = "camelCase")]
// pub fn get_pitch_refresh_status(
//     task_id: String,
//     state: State<'_, AppState>,
// ) -> Result<pitch_refresh_async::PitchTaskStatusPayload, String> {
//     pitch_refresh_async::get_pitch_refresh_status(task_id, state)
// }
//
// #[tauri::command(rename_all = "camelCase")]
// pub fn cancel_pitch_task(
//     task_id: String,
//     state: State<'_, AppState>,
// ) -> Result<(), String> {
//     pitch_refresh_async::cancel_pitch_task(task_id, state)
// }
