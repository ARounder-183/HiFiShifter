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

    // 回退到旧的 pitch_analysis_progress（单 clip 路径）。
    // 毒化恢复与本仓其余锁访问同一策略（一次 panic 不应让这条轮询命令永久失败）。
    let progress = state
        .pitch_analysis_progress
        .read()
        .unwrap_or_else(|e| e.into_inner());

    Ok(progress.as_ref().map(PitchProgressPayload::from))
}

/// 音符分块（GAME / VAD）的独立进度。前端只在参数块工具/显示开启时轮询。
pub(super) fn get_syllable_analysis_progress(
) -> Option<crate::pitch_analysis::SyllableAnalysisProgress> {
    crate::pitch_analysis::syllable_analysis_progress()
}

/// 按需读取 per-clip 源音高曲线（独立窗口的 Clip 音高参考线懒加载）。
///
/// 主窗口在分析完成后通过 clip_pitch_data 事件把曲线写进 session；卫星窗口
/// 晚打开时事件已经发生、快照又刻意剔除大数组，因此只能显式向这里要一次。
pub(super) fn get_clip_pitch_data(
    state: tauri::State<'_, AppState>,
    clip_ids: Vec<String>,
) -> serde_json::Value {
    // 毒化恢复与本仓其余锁访问同一策略（见 commands.rs 的 export_pitch_to_midi）。
    let timeline = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    let mut payloads = Vec::new();
    for clip_id in clip_ids {
        let Some(clip) = timeline.clips.iter().find(|clip| clip.id == clip_id) else {
            continue;
        };
        if let Some(payload) = crate::audio_engine::build_clip_pitch_data_payload(&timeline, clip) {
            payloads.push(payload);
        }
    }
    serde_json::json!({ "ok": true, "payloads": payloads })
}
