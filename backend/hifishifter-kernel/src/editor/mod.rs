//! 独立app与插件共享的原编辑命令；平台只适配undo、dirty与音频发布副作用。
pub mod assembly;
pub mod capabilities;
pub mod history;
pub mod host;
pub mod host_pcm;
pub mod host_timeline;
pub mod midi_import;
pub mod param_authority;
pub mod param_edit;
pub mod param_geometry;
pub mod params;
pub mod selection;
pub mod settings;
pub mod waveform;
use crate::state::{HistoryOp, TimelineState};
pub use selection::{ParamSelectionWindow, SelectionFrameRange};
use std::sync::Mutex;

/// 互转选区段，保持原Tauri命令JSON形状和帧语义。
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConvertRange {
    pub start_frame: u32,
    pub frame_count: u32,
}

/// 一次参数写入的**显式意图**：范围来自命令本身，不依赖旧几何的整份数组差分。
///
/// ARA 插件用它把"用户确实写了哪几帧"记进会话；独立 App 可以忽略（默认 no-op）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParamEditIntent {
    SetFrames {
        track_id: String,
        param: String,
        start_frame: usize,
        frame_count: usize,
    },
    RestoreFrames {
        track_id: String,
        param: String,
        start_frame: usize,
        frame_count: usize,
    },
}

/// 原参数命令所需的最小运行时边界；不包含Tauri、窗口或设备句柄。
pub trait ParamHost {
    fn timeline(&self) -> &Mutex<TimelineState>;
    fn checkpoint_timeline(&self, timeline: &TimelineState, operation: HistoryOp);
    /// 即使checkpoint=false的成功尾块也必须标记编辑代次与dirty。
    fn mark_dirty(&self);
    /// 独立app更新设备快照；插件只排后台自动渲染，不能在这里跑设备/推理。
    fn publish_timeline(&self, timeline: TimelineState);
    /// 记录一笔显式参数写入意图；默认实现供独立 App 忽略。
    fn note_param_edit(&self, _intent: ParamEditIntent) {}

    /// 插件权威读入口。返回 Some 时内核不再读/写整条显示数组。
    ///
    /// 独立 App 保持 None，走原有 project-frame 数组语义；ARA 插件实现这一组方法。
    fn read_authority_frames(
        &self,
        _track_id: &str,
        _param: &str,
        _start_frame: u32,
        _frame_count: u32,
        _stride: Option<u32>,
        _with_sentinel: Option<bool>,
        _binary: Option<bool>,
    ) -> Option<Result<crate::models::ParamFramesPayload, String>> {
        None
    }

    fn write_authority_frames(
        &self,
        _track_id: &str,
        _param: &str,
        _start_frame: u32,
        _values: &[f32],
        _present: Option<&[bool]>,
        _checkpoint: bool,
    ) -> Option<Result<serde_json::Value, String>> {
        None
    }

    fn restore_authority_frames(
        &self,
        _track_id: &str,
        _param: &str,
        _start_frame: u32,
        _frame_count: u32,
        _checkpoint: bool,
    ) -> Option<Result<serde_json::Value, String>> {
        None
    }

    /// 当前权威 revision（插件用于在 UI 载荷里给前端水位）。
    fn authority_revision(&self) -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct Host {
        timeline: Mutex<TimelineState>,
        history: Mutex<Vec<TimelineState>>,
        published: Mutex<Option<TimelineState>>,
        dirty: AtomicBool,
    }
    impl ParamHost for Host {
        fn timeline(&self) -> &Mutex<TimelineState> {
            &self.timeline
        }
        fn checkpoint_timeline(&self, timeline: &TimelineState, _operation: HistoryOp) {
            self.history.lock().unwrap().push(timeline.clone());
        }
        fn mark_dirty(&self) {
            self.dirty.store(true, Ordering::Release);
        }
        fn publish_timeline(&self, timeline: TimelineState) {
            *self.published.lock().unwrap() = Some(timeline);
        }
    }
    fn host() -> Host {
        let timeline = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"shared editor","order":0,"pitch_analysis_algo":"none"}],
            "clips":[],"bpm":120,"project_sec":1
        }))
        .unwrap();
        Host {
            timeline: Mutex::new(timeline),
            history: Mutex::new(vec![]),
            published: Mutex::new(None),
            dirty: AtomicBool::new(false),
        }
    }
    /// 缺少非checkpoint副作用会使最后一笔无法自动应用；既有undo步数不能被增加。
    #[test]
    fn noncheckpoint_tail_updates_curve_and_publication_without_extra_undo() {
        let host = host();
        assert_eq!(
            params::set_param_frames(
                &host,
                "track".into(),
                "pitch".into(),
                0,
                vec![60.],
                Some(true)
            )["ok"],
            true
        );
        host.dirty.store(false, Ordering::Release);
        assert_eq!(
            params::set_param_frames(
                &host,
                "track".into(),
                "pitch".into(),
                1,
                vec![64.],
                Some(false)
            )["ok"],
            true
        );
        assert!(host.dirty.load(Ordering::Acquire));
        assert_eq!(host.history.lock().unwrap().len(), 1);
        assert_eq!(
            &host.timeline.lock().unwrap().params_by_root_track["track"].pitch_edit[..2],
            &[60., 64.]
        );
        assert_eq!(
            &host
                .published
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .params_by_root_track["track"]
                .pitch_edit[..2],
            &[60., 64.]
        );
    }
    /// 平滑/恢复/静态写入必须经过同一共享命令，不因平台hook换掉原参数结果。
    #[test]
    fn static_parameters_and_restore_share_original_command_semantics() {
        let host = host();
        assert_eq!(
            params::set_static_param(
                &host,
                "track".into(),
                "test_parameter".into(),
                0.75,
                Some(false)
            )["ok"],
            true
        );
        let value = params::get_static_param(&host, "track".into(), "test_parameter".into());
        assert!(value.ok);
        assert_eq!(value.value, 0.75);
        assert!(host.history.lock().unwrap().is_empty());
        params::set_param_frames(
            &host,
            "track".into(),
            "pitch".into(),
            0,
            vec![60.],
            Some(false),
        );
        let result =
            params::restore_param_frames(&host, "track".into(), "pitch".into(), 0, 1, Some(false));
        assert_eq!(result["ok"], true);
        let timeline = host.timeline.lock().unwrap();
        let curve = &timeline.params_by_root_track["track"];
        assert_eq!(curve.pitch_edit[0], curve.pitch_orig[0]);
    }
}
