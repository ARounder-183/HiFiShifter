//! 参数块编辑集与派生视图的校验。
//!
//! 职责：把所有“非法输入绝不进入权威”的判断收在一处。
//! 关系：`types::ParamBlockEditSet::replace` 与 `compose` 都调用本文件。
//! 维护约束：错误字符串统一以 `Invalid: ` 开头；绝不 panic。

use super::types::{BlockGeometry, ParamBlockEditSet, ParamBlockView, UserParamBlock};

pub const MAX_BLOCKS: usize = 20_000;
pub const MAX_MEMBERS: usize = 20_000;
pub const MAX_GROUP_DEPTH: usize = 8;

pub fn validate_edit_set(edits: &ParamBlockEditSet) -> Result<(), String> {
    if edits.schema_version != 1 {
        return Err("Invalid: parameter block schema version".into());
    }
    let mut total_blocks = 0usize;
    let mut ids = std::collections::BTreeSet::new();
    for block in &edits.blocks {
        validate_block(block, 0, &mut total_blocks, &mut ids)?;
    }
    if total_blocks > MAX_BLOCKS {
        return Err("Invalid: parameter block count exceeded".into());
    }
    Ok(())
}

fn validate_block(
    block: &UserParamBlock,
    depth: usize,
    total_blocks: &mut usize,
    ids: &mut std::collections::BTreeSet<String>,
) -> Result<(), String> {
    if depth > MAX_GROUP_DEPTH {
        return Err("Invalid: parameter block group depth".into());
    }
    *total_blocks = total_blocks.saturating_add(1);
    if *total_blocks > MAX_BLOCKS {
        return Err("Invalid: parameter block count exceeded".into());
    }
    if block.id.is_empty() || block.id.len() > 512 || !ids.insert(block.id.clone()) {
        return Err("Invalid: parameter block id".into());
    }
    if block.deleted && block.grouped_from.is_some() {
        return Err("Invalid: deleted grouped block".into());
    }
    match &block.geometry {
        BlockGeometry::Take { members } => {
            if members.is_empty() || members.len() > MAX_MEMBERS {
                return Err("Invalid: parameter block members".into());
            }
            for member in members {
                if member.clip_id.is_empty()
                    || member.take_id.is_empty()
                    || !member.source_start_sec.is_finite()
                    || !member.source_end_sec.is_finite()
                    || member.source_end_sec <= member.source_start_sec
                {
                    return Err("Invalid: parameter block source span".into());
                }
            }
        }
        BlockGeometry::Root { span } => {
            if block.deleted {
                return Err("Invalid: deleted root block".into());
            }
            if !span.start_sec.is_finite()
                || !span.end_sec.is_finite()
                || span.end_sec <= span.start_sec
            {
                return Err("Invalid: parameter block root span".into());
            }
        }
    }
    if block
        .transition_out_sec
        .is_some_and(|width| !width.is_finite() || width < 0.0)
    {
        return Err("Invalid: parameter block transition".into());
    }
    if block.value.is_some_and(|value| !value.is_finite()) {
        return Err("Invalid: parameter block value".into());
    }
    if let Some(children) = &block.grouped_from {
        if children.len() < 2 {
            return Err("Invalid: parameter block group members".into());
        }
        for child in children {
            validate_block(child, depth + 1, total_blocks, ids)?;
        }
    }
    Ok(())
}

pub fn validate_view(view: &ParamBlockView) -> Result<(), String> {
    if view.schema_version != 1 || !view.frame_period_ms.is_finite() || view.frame_period_ms <= 0.0
    {
        return Err("Invalid: parameter block view schema".into());
    }
    if view.blocks.len() > MAX_BLOCKS {
        return Err("Invalid: parameter block view count".into());
    }
    let mut ids = std::collections::BTreeSet::new();
    let mut previous_end = 0u32;
    for block in &view.blocks {
        if block.id.is_empty()
            || block.id.len() > 512
            || !ids.insert(&block.id)
            || block.end_frame_exclusive <= block.start_frame
            || block.start_frame < previous_end
            || block.end_frame_exclusive > 1_000_000
        {
            return Err("Invalid: parameter block view interval".into());
        }
        if block.transition_frames.is_some_and(|width| {
            width > block.end_frame_exclusive.saturating_sub(block.start_frame)
        }) {
            return Err("Invalid: parameter block view transition".into());
        }
        previous_end = block.end_frame_exclusive;
    }
    Ok(())
}
