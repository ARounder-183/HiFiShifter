//! 参数块的唯一模型：Take 源域身份 + Track 用户覆盖层 + 派生投影视图。
//!
//! 职责：本模块是 App、ARA、VST Capture 共用的唯一参数块实现。
//! 关系：曲线仍由 `editor::param_authority` 以 root 工程帧为权威；本模块只负责
//! 离散分段的身份、编辑与显示投影。
//! 维护约束：`ParamBlockView` 永远不是权威；保存工程前必须清空。任何旧版
//! 绝对帧块模型、merge/interval 记录都不得在这里复活。

pub mod base;
pub mod compose;
pub mod editing;
pub mod layout;
pub mod projection;
pub mod types;
pub mod validate;

pub use compose::{compose_param_block_view, project_single_user_block};
pub use editing::DeletedParamBlockSnapshot;
pub use types::{
    BlockGeometry, BlockLabel, GameBlockBaseline, ParamBlockEditSet, ParamBlockView,
    ProjectedBlockKind, ProjectedMember, ProjectedParamBlock, RootSpan, TakeKey, TakeSpan,
    UserParamBlock,
};
pub use validate::{validate_edit_set, validate_view};

use serde::{Deserialize, Serialize};

/// 参数曲线补丁：与块布局在同一事务提交。
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ParamBlockCurvePatch {
    pub param: String,
    pub start_frame: u32,
    pub values: Vec<f32>,
}

pub fn validate_curve_patches(patches: &[ParamBlockCurvePatch]) -> Result<(), String> {
    let mut total = 0usize;
    for patch in patches {
        total = total.saturating_add(patch.values.len());
        if patch.param.is_empty()
            || patch.param.len() > 128
            || total > 1_000_000
            || patch.start_frame as usize + patch.values.len() > 1_000_000
            || patch.values.iter().any(|value| !value.is_finite())
        {
            return Err("Invalid: parameter block curve patch".into());
        }
    }
    Ok(())
}

pub fn apply_curve_patches(
    timeline: &crate::state::TimelineState,
    root: &str,
    patches: &[ParamBlockCurvePatch],
) -> Result<crate::state::TimelineState, String> {
    validate_curve_patches(patches)?;
    struct Candidate(std::sync::Mutex<crate::state::TimelineState>);
    impl crate::editor::ParamHost for Candidate {
        fn timeline(&self) -> &std::sync::Mutex<crate::state::TimelineState> {
            &self.0
        }
        fn checkpoint_timeline(&self, _: &crate::state::TimelineState, _: crate::state::HistoryOp) {
        }
        fn mark_dirty(&self) {}
        fn publish_timeline(&self, _: crate::state::TimelineState) {}
    }
    let candidate = Candidate(std::sync::Mutex::new(timeline.clone()));
    for patch in patches {
        let result = crate::editor::params::set_param_frames_with_present(
            &candidate,
            root.to_owned(),
            patch.param.clone(),
            patch.start_frame,
            patch.values.clone(),
            None,
            Some(false),
        );
        if result["ok"] != true {
            return Err(result["error"]
                .as_str()
                .unwrap_or("block curve edit failed")
                .into());
        }
    }
    Ok(candidate.0.into_inner().unwrap())
}

/// 判断两份时间线是否只在参数块编辑/视图上不同。
///
/// 【用途】纯块边界编辑不触发音频重渲染：音频指纹比较时把编辑集与派生视图
/// 一起剥掉；两者都不参与声音。
pub fn timeline_equal_except_blocks(
    left: &crate::state::TimelineState,
    right: &crate::state::TimelineState,
) -> bool {
    let strip = |mut snapshot: crate::state::TimelineState| {
        for params in snapshot.params_by_root_track.values_mut() {
            params.param_block_edits = ParamBlockEditSet::empty();
            params.param_block_view = None;
        }
        snapshot
            .params_by_root_track
            .retain(|_, params| !params.is_empty_project_data());
        serde_json::to_value(snapshot).ok()
    };
    strip(left.clone()) == strip(right.clone())
}
