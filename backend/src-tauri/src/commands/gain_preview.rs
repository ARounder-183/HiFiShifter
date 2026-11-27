//! 拖动增益的瞬时试听命令。
//!
//! 【为什么独立成模块】这条通道刻意**不取 timeline 锁、不 checkpoint、不返回
//! 全量 payload**：它只把手势值投递给音频 worker。若复用 set_clip_state /
//! set_clips_state_bulk，逐帧提交会克隆整份时间线、写 undo、重建快照，UI 与
//! 音频都会被拖垮。权威提交仍然只在手势结束时发生一次。

use hifishifter_kernel::engine_command::GainPreviewTarget;
use tauri::State;

use crate::state::AppState;

#[tauri::command(rename_all = "camelCase")]
pub fn set_gain_preview(
    state: State<'_, AppState>,
    gesture_id: String,
    seq: u64,
    targets: Vec<GainPreviewTarget>,
) -> Result<(), String> {
    state
        .audio_engine
        .set_gain_preview(gesture_id, seq, targets);
    Ok(())
}

#[tauri::command(rename_all = "camelCase")]
pub fn clear_gain_preview(state: State<'_, AppState>, gesture_id: String) -> Result<(), String> {
    state.audio_engine.clear_gain_preview(gesture_id);
    Ok(())
}
