//! 原参数命令的Tauri薄适配；实际曲线/默认值/互转逻辑与插件共享，不维护两套。
use crate::state::{AppState, HistoryOp, TimelineState};
use hifishifter_kernel::editor::{params, ParamHost};
use tauri::State;

impl ParamHost for AppState {
    fn timeline(&self) -> &std::sync::Mutex<TimelineState> { &self.timeline }
    fn checkpoint_timeline(&self, timeline: &TimelineState, operation: HistoryOp) {
        AppState::checkpoint_timeline(self, timeline, operation);
    }
    fn mark_dirty(&self) { self.mark_project_dirty_and_retitle(); }
    fn publish_timeline(&self, timeline: TimelineState) { self.audio_engine.update_timeline(timeline); }
}

/// 读原/编辑帧，保持既有二进制、stride与sentinel返回格式。
pub(super) fn get_param_frames(state: State<'_,AppState>, track_id:String, param:String,
    start_frame:u32, frame_count:u32, stride:Option<u32>, binary:Option<bool>,
    with_sentinel:Option<bool>) -> crate::models::ParamFramesPayload {
    params::get_param_frames(&*state,track_id,param,start_frame,frame_count,stride,binary,with_sentinel)
}
/// 写参数；checkpoint只控制undo，dirty副作用由共享逻辑分别记录。
pub(crate) fn set_param_frames(state:&AppState, track_id:String, param:String, start_frame:u32,
    values:Vec<f32>, checkpoint:Option<bool>) -> serde_json::Value {
    params::set_param_frames(state,track_id,param,start_frame,values,checkpoint)
}
/// 恢复原参数，不改变原分块checkpoint契约。
pub(crate) fn restore_param_frames(state:&AppState,track_id:String,param:String,start_frame:u32,
    frame_count:u32,checkpoint:Option<bool>) -> serde_json::Value {
    params::restore_param_frames(state,track_id,param,start_frame,frame_count,checkpoint)
}
/// 音量与动态转换继续使用原权威基线补偿实现。
pub(super) fn convert_mix_param(state:State<'_,AppState>,track_id:String,from:String,
    ranges:Vec<crate::commands::ConvertRange>) -> serde_json::Value {
    params::convert_mix_param(&*state,track_id,from,ranges)
}
/// 读静态参数及默认值。
pub(super) fn get_static_param(state:State<'_,AppState>,track_id:String,param:String)
    -> crate::models::StaticParamValuePayload { params::get_static_param(&*state,track_id,param) }
/// 写静态参数，独立设备更新保持既有路径。
pub(crate) fn set_static_param(state:&AppState,track_id:String,param:String,value:f64,
    checkpoint:Option<bool>) -> serde_json::Value {
    params::set_static_param(state,track_id,param,value,checkpoint)
}
/// 共享原参数时间域映射；插件是否允许几何操作由插件命令准入控制。
pub(crate) fn stretch_track_linked_params(state:&AppState,track_id:String,
    mappings:Vec<crate::state::StretchLinkedRangeSec>,checkpoint:Option<bool>) -> serde_json::Value {
    params::stretch_track_linked_params(state,track_id,mappings,checkpoint)
}