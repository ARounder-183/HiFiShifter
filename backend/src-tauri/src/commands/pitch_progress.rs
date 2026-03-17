use crate::pitch_analysis::PitchProgressPayload;
use crate::state::AppState;

/// Get current pitch analysis progress (Task 3.6)
pub(super) fn get_pitch_analysis_progress(
    state: tauri::State<AppState>,
) -> Result<Option<PitchProgressPayload>, String> {
    // 优先返回 pitch_clip 批次进度（包含 clip 名称信息）
    if let Some(batch) = crate::pitch_clip::get_clip_pitch_batch_progress() {
        return Ok(Some(PitchProgressPayload {
            root_track_id: String::new(),
            progress: batch.progress,
            eta_seconds: None,
            current_clip_name: batch.current_clip_name,
            completed_clips: batch.completed_clips,
            total_clips: batch.total_clips,
        }));
    }

    // 回退到旧的 pitch_analysis_progress（单 clip 路径）
    let progress = state
        .pitch_analysis_progress
        .read()
        .map_err(|e| format!("Failed to read progress: {}", e))?;

    Ok(progress.as_ref().map(|p| PitchProgressPayload::from(p)))
}
