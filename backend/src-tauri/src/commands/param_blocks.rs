//! 参数块命令：源域编辑集 + 派生视图。
//!
//! 职责：App 侧唯一的参数块读写入口。曲线补丁与块布局在同一事务提交。
//! 关系：内核 `param_blocks` 负责投影/合成/编辑；本文件只做锁、CAS、历史与事件。
//! 维护约束：绝不把 `ParamBlockView` 写进工程文件；它只是传输缓存。

use crate::state::AppState;
use hifishifter_kernel::param_blocks::{
    apply_curve_patches, compose_param_block_view, validate_curve_patches, ParamBlockCurvePatch,
    ParamBlockEditSet, ParamBlockView, UserParamBlock,
};
use tauri::Emitter;

fn view_payload(
    view: &ParamBlockView,
    edits: &ParamBlockEditSet,
    curve_revision: u64,
) -> serde_json::Value {
    serde_json::json!({
        "ok": true,
        "editRevision": view.edit_revision,
        "layoutKey": view.layout_key,
        "framePeriodMs": view.frame_period_ms,
        "blocks": view.blocks,
        "orphanCount": view.orphan_count,
        "analysisPending": view.analysis_pending,
        "curveRevision": curve_revision,
        "edits": edits,
    })
}

pub(super) fn get(state: &AppState, root: &str) -> Result<serde_json::Value, String> {
    let mut timeline = state
        .timeline
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if !timeline.tracks.iter().any(|track| track.id == root) {
        return Err("unknown root parameter line".into());
    }
    let frame_period_ms = timeline.frame_period_ms();
    let edits = timeline
        .params_by_root_track
        .get(root)
        .map(|params| params.param_block_edits.clone())
        .unwrap_or_default();
    let view = compose_param_block_view(&timeline, root, &edits, frame_period_ms)?;
    timeline.ensure_params_for_root(root);
    timeline
        .params_by_root_track
        .get_mut(root)
        .unwrap()
        .param_block_view = Some(view.clone());
    Ok(view_payload(
        &view,
        &edits,
        state
            .param_curve_revision
            .load(std::sync::atomic::Ordering::Acquire),
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn set(
    state: &AppState,
    root: String,
    expected_edit_revision: u64,
    edits: Vec<UserParamBlock>,
    checkpoint: bool,
    patches: Vec<ParamBlockCurvePatch>,
    expected_curve_revision: Option<u64>,
    expected_geometry_signature: Option<String>,
) -> Result<serde_json::Value, String> {
    validate_curve_patches(&patches)?;
    let (view, edits_for_payload, audio_timeline, curve_revision) = {
        let mut timeline = state
            .timeline
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !timeline.tracks.iter().any(|track| track.id == root) {
            return Err("unknown root parameter line".into());
        }
        if !patches.is_empty()
            && expected_geometry_signature
                .as_deref()
                .is_some_and(|signature| {
                    signature
                        != hifishifter_kernel::pitch_analysis::build_root_pitch_layout_key(
                            &timeline, &root,
                        )
                })
        {
            return Err("Conflict: parameter geometry changed".into());
        }
        if !patches.is_empty()
            && expected_curve_revision
                != Some(
                    state
                        .param_curve_revision
                        .load(std::sync::atomic::Ordering::Acquire),
                )
        {
            return Err("Conflict: parameter curve changed".into());
        }
        let current = timeline
            .params_by_root_track
            .get(&root)
            .map(|params| params.param_block_edits.clone())
            .unwrap_or_default();
        let next = current.replace(expected_edit_revision, edits)?;
        let mut candidate = if patches.is_empty() {
            timeline.clone()
        } else {
            apply_curve_patches(&timeline, &root, &patches)?
        };
        let frame_period_ms = candidate.frame_period_ms();
        let view = compose_param_block_view(&candidate, &root, &next, frame_period_ms)?;
        if checkpoint && (next.blocks != current.blocks || !patches.is_empty()) {
            state.checkpoint_block_layout(&timeline);
        }
        candidate.ensure_params_for_root(&root);
        let params = candidate.params_by_root_track.get_mut(&root).unwrap();
        params.param_block_edits = next.clone();
        params.param_block_view = Some(view.clone());
        *timeline = candidate;
        let audio = if patches.is_empty() {
            None
        } else {
            state.bump_timeline_version();
            Some(timeline.clone())
        };
        (
            view,
            next,
            audio,
            state
                .param_curve_revision
                .load(std::sync::atomic::Ordering::Acquire),
        )
    };
    if let Some(timeline) = audio_timeline {
        state.audio_engine.update_timeline(timeline);
    }
    state.mark_project_dirty_and_retitle();
    if let Some(app) = state.app_handle.get() {
        let _ = app.emit(
            "param_block_view_updated",
            serde_json::json!({"rootTrackId":root,"view":view}),
        );
    }
    Ok(view_payload(&view, &edits_for_payload, curve_revision))
}

pub(super) fn restore_game(
    state: &AppState,
    root: String,
    start_frame: u32,
    end_frame_exclusive: u32,
    checkpoint: bool,
) -> Result<serde_json::Value, String> {
    let (view, edits_for_payload, curve_revision) = {
        let mut timeline = state
            .timeline
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !timeline.tracks.iter().any(|track| track.id == root) {
            return Err("unknown root parameter line".into());
        }
        let current = timeline
            .params_by_root_track
            .get(&root)
            .map(|params| params.param_block_edits.clone())
            .unwrap_or_default();
        let next = hifishifter_kernel::param_blocks::editing::restore_game_blocks(
            &timeline,
            &root,
            &current,
            start_frame,
            end_frame_exclusive,
            timeline.frame_period_ms(),
        )?;
        let view = compose_param_block_view(&timeline, &root, &next, timeline.frame_period_ms())?;
        if checkpoint {
            state.checkpoint_block_layout(&timeline);
        }
        timeline.ensure_params_for_root(&root);
        let params = timeline.params_by_root_track.get_mut(&root).unwrap();
        params.param_block_edits = next.clone();
        params.param_block_view = Some(view.clone());
        (
            view,
            next,
            state
                .param_curve_revision
                .load(std::sync::atomic::Ordering::Acquire),
        )
    };
    state.mark_project_dirty_and_retitle();
    if let Some(app) = state.app_handle.get() {
        let _ = app.emit(
            "param_block_view_updated",
            serde_json::json!({"rootTrackId":root,"view":view}),
        );
    }
    Ok(view_payload(&view, &edits_for_payload, curve_revision))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hifishifter_kernel::param_blocks::{BlockGeometry, BlockLabel, TakeSpan};
    use hifishifter_kernel::pitch_analysis::{SyllableSpan, TakeSyllableAnalysis};

    fn analysis(source_digest: String) -> TakeSyllableAnalysis {
        TakeSyllableAnalysis {
            algorithm: "test".into(),
            algorithm_version: 1,
            source_digest,
            source_spans_sec: vec![
                SyllableSpan {
                    start_sec: 0.0,
                    end_sec: 0.5,
                },
                SyllableSpan {
                    start_sec: 0.5,
                    end_sec: 1.0,
                },
            ],
        }
    }

    /// 一条 root 轨道 + 一条带源路径的 Clip，**还没有**音节分析。
    fn state_without_analysis() -> (AppState, String, String, String) {
        let state = AppState::default();
        let root = {
            let mut timeline = state.timeline.lock().unwrap();
            timeline.add_track(None, None, None);
            let root = timeline.tracks[0].id.clone();
            timeline.add_clip(
                Some(root.clone()),
                Some("clip".into()),
                Some(0.0),
                Some(2.0),
                Some("a.wav".into()),
            );
            root
        };
        let (clip_id, take_id) = {
            let timeline = state.timeline.lock().unwrap();
            (
                timeline.clips[0].id.clone(),
                timeline.clips[0].active_take().id.clone(),
            )
        };
        (state, root, clip_id, take_id)
    }

    /// 同上，并把分析结果写进 active take（摘要按真实素材算，与命令层同源）。
    fn state_with_clip() -> (AppState, String, String, String) {
        let (state, root, clip_id, take_id) = state_without_analysis();
        {
            let mut timeline = state.timeline.lock().unwrap();
            let frame_period_ms = timeline.frame_period_ms();
            let clip = timeline
                .clips
                .iter_mut()
                .find(|clip| clip.id == clip_id)
                .unwrap();
            let digest = hifishifter_kernel::param_blocks::source_digest(clip, frame_period_ms);
            clip.takes[0].syllable_analysis = Some(analysis(digest));
        }
        (state, root, clip_id, take_id)
    }

    /// 打一个撤销点（快照 = 当前时间线）。
    fn checkpoint(state: &AppState) {
        let timeline = state.timeline.lock().unwrap();
        state.checkpoint_block_layout(&timeline);
    }

    fn param_block_view_of(state: &AppState, root: &str) -> Option<ParamBlockView> {
        let timeline = state.timeline.lock().unwrap();
        timeline
            .params_by_root_track
            .get(root)
            .and_then(|params| params.param_block_view.clone())
    }

    /// 一个覆盖 [source_start, source_end] 的用户块。
    fn user_block(
        clip_id: &str,
        take_id: &str,
        source_start: f64,
        source_end: f64,
    ) -> UserParamBlock {
        UserParamBlock {
            id: "manual".into(),
            geometry: BlockGeometry::Take {
                members: vec![TakeSpan {
                    clip_id: clip_id.to_string(),
                    take_id: take_id.to_string(),
                    cycle: 0,
                    source_start_sec: source_start,
                    source_end_sec: source_end,
                    syllable_id: None,
                    label: BlockLabel::Manual,
                }],
            },
            deleted: false,
            manual: true,
            value: Some(60.0),
            transition_out_sec: None,
            created_seq: 1,
            base_cover: None,
            grouped_from: None,
        }
    }

    #[test]
    fn empty_edit_set_is_accepted_and_serialized_with_blocks_array() {
        let (state, root, _clip_id, _take_id) = state_with_clip();
        let initial = get(&state, &root).unwrap();
        let revision = initial["editRevision"].as_u64().unwrap();
        let edited = set(
            &state,
            root,
            revision,
            Vec::new(),
            false,
            Vec::new(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(edited["edits"]["blocks"], serde_json::json!([]));
    }

    #[test]
    fn app_block_edit_is_atomic_and_survives_undo() {
        let (state, root, clip_id, take_id) = state_with_clip();
        let initial = get(&state, &root).unwrap();
        assert_eq!(initial["blocks"].as_array().unwrap().len(), 2);
        let revision = initial["editRevision"].as_u64().unwrap();
        let edited = set(
            &state,
            root.clone(),
            revision,
            vec![UserParamBlock {
                id: "manual".into(),
                geometry: BlockGeometry::Take {
                    members: vec![TakeSpan {
                        clip_id: clip_id.clone(),
                        take_id: take_id.clone(),
                        cycle: 0,
                        source_start_sec: 0.0,
                        source_end_sec: 1.0,
                        syllable_id: None,
                        label: BlockLabel::Manual,
                    }],
                },
                deleted: false,
                manual: true,
                value: Some(60.0),
                transition_out_sec: None,
                created_seq: 1,
                base_cover: None,
                grouped_from: None,
            }],
            true,
            Vec::new(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(edited["blocks"].as_array().unwrap().len(), 1);
        let generation = state
            .timeline_version
            .load(std::sync::atomic::Ordering::Acquire);
        state.undo_timeline();
        let undone = get(&state, &root).unwrap();
        assert_eq!(undone["blocks"].as_array().unwrap().len(), 2);
        assert_eq!(
            state
                .timeline_version
                .load(std::sync::atomic::Ordering::Acquire),
            generation
        );
        state.redo_timeline();
        assert_eq!(
            get(&state, &root).unwrap()["blocks"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert!(set(&state, root, 0, Vec::new(), true, Vec::new(), None, None).is_err());
    }

    /*
     * 【一次提交 = 一个撤销步】曲线补丁与块布局必须在同一个 `set` 里落地。
     *
     * 前端的一次块拖拽同时改块中心（`edits`）与曲线（`patches`）；拆成两次写
     * 就会产生两步历史，用户要按两次撤销才能回到原样。
     */
    #[test]
    fn one_set_with_edits_and_patches_is_one_history_step() {
        let (state, root, clip_id, take_id) = state_with_clip();
        let revision = get(&state, &root).unwrap()["editRevision"]
            .as_u64()
            .unwrap();
        let curve_revision = state
            .param_curve_revision
            .load(std::sync::atomic::Ordering::Acquire);
        assert_eq!(state.history_position(), 0);

        let edited = set(
            &state,
            root.clone(),
            revision,
            vec![user_block(&clip_id, &take_id, 0.0, 1.0)],
            true,
            vec![ParamBlockCurvePatch {
                param: "pitch".into(),
                start_frame: 0,
                values: vec![61.0, 61.0],
            }],
            Some(curve_revision),
            None,
        )
        .unwrap();
        assert_eq!(edited["blocks"].as_array().unwrap().len(), 1);
        assert_eq!(state.history_position(), 1, "一次 set 只能产生一个撤销步",);
        assert_eq!(
            state.timeline.lock().unwrap().params_by_root_track[&root]
                .pitch_edit
                .get(..2),
            Some(&[61.0f32, 61.0][..]),
        );

        // 撤销一次：块与曲线同时回退。
        assert!(state.undo_timeline().ok);
        assert_eq!(state.history_position(), 0);
        assert_eq!(
            get(&state, &root).unwrap()["blocks"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_ne!(
            state.timeline.lock().unwrap().params_by_root_track[&root]
                .pitch_edit
                .get(..2),
            Some(&[61.0f32, 61.0][..]),
        );
    }

    /*
     * 【撤销不得删除分析结果】音节分析不进撤销栈（它是素材派生物），但历史快照
     * 存的是整份时间线。于是"记录里还没有分析、实时时间线已经有了"这个中间态
     * 会让一次撤销把分析一起回退 —— 块整片消失，而前端已经记下"分析过了"，
     * 不会自己补回来。
     */
    #[test]
    fn undo_does_not_delete_syllable_analysis() {
        let (state, root, clip_id, _take_id) = state_without_analysis();
        // 分析完成之前的两个撤销点：两条记录的快照里都没有分析。
        checkpoint(&state);
        checkpoint(&state);
        // 分析完成（不进撤销栈）。
        {
            let mut timeline = state.timeline.lock().unwrap();
            let frame_period_ms = timeline.frame_period_ms();
            let clip = timeline
                .clips
                .iter_mut()
                .find(|clip| clip.id == clip_id)
                .unwrap();
            let digest = hifishifter_kernel::param_blocks::source_digest(clip, frame_period_ms);
            clip.takes[0].syllable_analysis = Some(analysis(digest));
        }
        // `get` 会顺手把派生视图写进缓存 —— 正好用来验证"缓存不得随快照恢复"。
        assert_eq!(
            get(&state, &root).unwrap()["blocks"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        checkpoint(&state);

        assert!(state.undo_timeline().ok);
        assert!(
            param_block_view_of(&state, &root).is_none(),
            "撤销不得把快照里的过期派生视图带回来",
        );

        assert!(state.undo_timeline().ok);
        assert_eq!(
            get(&state, &root).unwrap()["blocks"]
                .as_array()
                .unwrap()
                .len(),
            2,
            "撤销跨过分析完成点之后，块必须仍然存在",
        );
        let timeline = state.timeline.lock().unwrap();
        assert!(timeline.clips[0].takes[0].syllable_analysis.is_some());
    }

    /*
     * 负例：撤销一次"替换素材"不能把**新素材**的分段贴到旧素材上。
     * 判据是 `source_digest`，所以只要分析摘要对不上就不搬运。
     */
    #[test]
    fn analysis_is_not_carried_across_a_source_swap() {
        let (state, _root, clip_id, _take_id) = state_without_analysis();
        checkpoint(&state);
        checkpoint(&state);
        {
            let mut timeline = state.timeline.lock().unwrap();
            let clip = timeline
                .clips
                .iter_mut()
                .find(|clip| clip.id == clip_id)
                .unwrap();
            clip.takes[0].syllable_analysis = Some(analysis("other-media".into()));
        }
        checkpoint(&state);

        assert!(state.undo_timeline().ok);
        assert!(state.undo_timeline().ok);
        let timeline = state.timeline.lock().unwrap();
        assert!(timeline.clips[0].takes[0].syllable_analysis.is_none());
    }
}
