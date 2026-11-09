//! 参数块的源域编辑原语。
//!
//! 职责：所有编辑都只改写 `ParamBlockEditSet`，绝不写绝对帧。
//! 关系：`compose` 消费编辑结果；App/插件只通过本文件改块。
//! 维护约束：新块必须经过 `place_user_block`，由它裁剪被覆盖的旧用户块。

use super::projection::{inverse_project, ClipProjection};
use super::types::{
    BlockGeometry, BlockLabel, ParamBlockEditSet, ProjectedBlockKind, ProjectedParamBlock,
    RootSpan, TakeSpan, UserParamBlock,
};
use crate::state::TimelineState;

fn new_id(prefix: &str, identity: &str) -> String {
    format!("{prefix}:{}", blake3::hash(identity.as_bytes()).to_hex())
}

fn next_seq(edits: &ParamBlockEditSet) -> u64 {
    edits.next_seq()
}

fn subtract_interval(start: f64, end: f64, cut_start: f64, cut_end: f64) -> Vec<(f64, f64)> {
    if cut_end <= start + 1e-12 || cut_start >= end - 1e-12 {
        return vec![(start, end)];
    }
    let mut out = Vec::new();
    if start < cut_start {
        out.push((start, cut_start));
    }
    if end > cut_end {
        out.push((cut_end, end));
    }
    out
}

/// 用 target 覆盖旧用户块；被部分覆盖的块保留未覆盖成员。
pub fn place_user_block(
    edits: &ParamBlockEditSet,
    target: UserParamBlock,
) -> Result<ParamBlockEditSet, String> {
    let mut blocks: Vec<UserParamBlock> = Vec::new();
    for block in &edits.blocks {
        if block.id == target.id {
            continue;
        }
        // 墓碑只抑制 base，不参与可见用户块的互相裁剪。
        if block.deleted || target.deleted {
            blocks.push(block.clone());
            continue;
        }
        let remaining = match (&block.geometry, &target.geometry) {
            (BlockGeometry::Take { members }, BlockGeometry::Take { members: cuts }) => {
                let mut kept = Vec::new();
                for member in members {
                    let mut pieces = vec![(member.source_start_sec, member.source_end_sec)];
                    for cut in cuts {
                        if cut.clip_id != member.clip_id
                            || cut.take_id != member.take_id
                            || cut.cycle != member.cycle
                        {
                            continue;
                        }
                        let mut next = Vec::new();
                        for (start, end) in pieces {
                            next.extend(subtract_interval(
                                start,
                                end,
                                cut.source_start_sec,
                                cut.source_end_sec,
                            ));
                        }
                        pieces = next;
                    }
                    for (start, end) in pieces {
                        if end > start + 1e-12 {
                            let mut part = member.clone();
                            part.source_start_sec = start;
                            part.source_end_sec = end;
                            kept.push(part);
                        }
                    }
                }
                if kept.is_empty() {
                    None
                } else {
                    let mut next = block.clone();
                    next.geometry = BlockGeometry::Take { members: kept };
                    Some(next)
                }
            }
            (BlockGeometry::Root { span }, BlockGeometry::Root { span: cut }) => {
                let pieces =
                    subtract_interval(span.start_sec, span.end_sec, cut.start_sec, cut.end_sec);
                if pieces.is_empty() {
                    None
                } else {
                    let mut next = block.clone();
                    let (start, end) = pieces[0];
                    next.geometry = BlockGeometry::Root {
                        span: RootSpan {
                            start_sec: start,
                            end_sec: end,
                        },
                    };
                    Some(next)
                }
            }
            _ => Some(block.clone()),
        };
        if let Some(block) = remaining {
            blocks.push(block);
        }
    }
    blocks.push(target);
    let mut next = edits.clone();
    next.blocks = blocks;
    next.revision = edits.revision.saturating_add(1);
    next.validate()?;
    Ok(next)
}

fn take_members(selected: &[ProjectedParamBlock]) -> Vec<TakeSpan> {
    let mut out: Vec<TakeSpan> = Vec::new();
    for block in selected {
        for member in &block.members {
            let span = TakeSpan {
                clip_id: member.clip_id.clone(),
                take_id: member.take_id.clone(),
                cycle: member.cycle,
                source_start_sec: member.source_start_sec,
                source_end_sec: member.source_end_sec,
                syllable_id: member.syllable_id.clone(),
                label: BlockLabel::Manual,
            };
            if !out.iter().any(|existing| {
                existing.clip_id == span.clip_id
                    && existing.take_id == span.take_id
                    && existing.cycle == span.cycle
                    && (existing.source_start_sec - span.source_start_sec).abs() < 1e-9
                    && (existing.source_end_sec - span.source_end_sec).abs() < 1e-9
            }) {
                out.push(span);
            }
        }
    }
    out
}

pub fn merge_blocks(
    edits: &ParamBlockEditSet,
    selected: &[ProjectedParamBlock],
    frame_period_ms: f64,
) -> Result<ParamBlockEditSet, String> {
    if selected.len() < 2 {
        return Err("Invalid: merge needs two blocks".into());
    }
    let mut ordered = selected.to_vec();
    ordered.sort_by_key(|block| block.start_frame);
    for pair in ordered.windows(2) {
        if pair[0].end_frame_exclusive != pair[1].start_frame {
            return Err("Invalid: merge blocks are not adjacent".into());
        }
    }
    let members = take_members(&ordered);
    let geometry = if members.is_empty() {
        let start = ordered.first().map(|block| block.start_frame).unwrap_or(0);
        let end = ordered
            .last()
            .map(|block| block.end_frame_exclusive)
            .unwrap_or(0);
        BlockGeometry::Root {
            span: RootSpan {
                start_sec: start as f64 * frame_period_ms.max(0.1) / 1000.0,
                end_sec: end as f64 * frame_period_ms.max(0.1) / 1000.0,
            },
        }
    } else {
        BlockGeometry::Take { members }
    };
    let values: Vec<Option<f64>> = ordered.iter().map(|block| block.value).collect();
    let value = values
        .first()
        .copied()
        .flatten()
        .filter(|first| values.iter().all(|value| *value == Some(*first)));
    let manual = ordered
        .iter()
        .all(|block| matches!(block.kind, ProjectedBlockKind::Manual));
    let transition_out_sec = ordered.last().and_then(|block| {
        block
            .transition_frames
            .map(|frames| frames as f64 * frame_period_ms.max(0.1) / 1000.0)
    });
    let target = UserParamBlock {
        id: new_id("ublock", &format!("merge:{}", ordered[0].id)),
        geometry,
        deleted: false,
        manual,
        value,
        transition_out_sec,
        created_seq: next_seq(edits),
    };
    place_user_block(edits, target)
}

pub fn split_block(
    timeline: &TimelineState,
    edits: &ParamBlockEditSet,
    block_id: &str,
    frame: u32,
    frame_period_ms: f64,
) -> Result<ParamBlockEditSet, String> {
    let block = edits
        .blocks
        .iter()
        .find(|block| block.id == block_id)
        .ok_or_else(|| "Invalid: block not found".to_string())?;
    match &block.geometry {
        BlockGeometry::Root { span } => {
            let split_sec = frame as f64 * frame_period_ms.max(0.1) / 1000.0;
            if split_sec <= span.start_sec + 1e-9 || split_sec >= span.end_sec - 1e-9 {
                return Err("Invalid: split outside block".into());
            }
            let left = UserParamBlock {
                id: new_id("ublock", &format!("split:{}:{frame}:left", block.id)),
                geometry: BlockGeometry::Root {
                    span: RootSpan {
                        start_sec: span.start_sec,
                        end_sec: split_sec,
                    },
                },
                deleted: false,
                manual: block.manual,
                value: block.value,
                transition_out_sec: Some(0.0),
                created_seq: next_seq(edits),
            };
            let right = UserParamBlock {
                id: new_id("ublock", &format!("split:{}:{frame}:right", block.id)),
                geometry: BlockGeometry::Root {
                    span: RootSpan {
                        start_sec: split_sec,
                        end_sec: span.end_sec,
                    },
                },
                deleted: false,
                manual: block.manual,
                value: block.value,
                transition_out_sec: block.transition_out_sec,
                created_seq: next_seq(edits).saturating_add(1),
            };
            let next = place_user_block(edits, left)?;
            place_user_block(&next, right)
        }
        BlockGeometry::Take { members } => {
            let mut ordered = members.clone();
            ordered.sort_by(|a, b| a.source_start_sec.total_cmp(&b.source_start_sec));
            let mut split_index = None;
            let mut split_source = None;
            for (index, member) in ordered.iter().enumerate() {
                let Some((projection, _)) = timeline
                    .clips
                    .iter()
                    .find(|clip| clip.id == member.clip_id)
                    .and_then(|clip| {
                        clip.take(&member.take_id)
                            .map(|take| (ClipProjection::from_clip_take(clip, take), take))
                    })
                else {
                    continue;
                };
                let intervals: Vec<_> =
                    super::projection::project_span(&projection, member, frame_period_ms)
                        .into_iter()
                        .filter(|interval| interval.cycle == member.cycle)
                        .collect();
                let start = intervals.first().map(|interval| interval.start_frame);
                let end = intervals
                    .last()
                    .map(|interval| interval.end_frame_exclusive);
                if let (Some(start), Some(end)) = (start, end) {
                    if frame > start && frame < end {
                        split_index = Some(index);
                        split_source =
                            inverse_project(&projection, frame, member.cycle, frame_period_ms);
                        break;
                    }
                }
            }
            let (Some(index), Some(source_split)) = (split_index, split_source) else {
                return Err("Invalid: split outside block".into());
            };
            let containing = &ordered[index];
            if source_split <= containing.source_start_sec + 1e-9
                || source_split >= containing.source_end_sec - 1e-9
            {
                return Err("Invalid: split outside block".into());
            }
            let mut left_members = ordered[..index].to_vec();
            let mut left_part = containing.clone();
            left_part.source_end_sec = source_split;
            left_members.push(left_part);
            let mut right_members = Vec::new();
            let mut right_part = containing.clone();
            right_part.source_start_sec = source_split;
            right_members.push(right_part);
            right_members.extend_from_slice(&ordered[index + 1..]);
            let left = UserParamBlock {
                id: new_id("ublock", &format!("split:{}:{frame}:left", block.id)),
                geometry: BlockGeometry::Take {
                    members: left_members,
                },
                deleted: false,
                manual: block.manual,
                value: block.value,
                transition_out_sec: Some(0.0),
                created_seq: next_seq(edits),
            };
            let right = UserParamBlock {
                id: new_id("ublock", &format!("split:{}:{frame}:right", block.id)),
                geometry: BlockGeometry::Take {
                    members: right_members,
                },
                deleted: false,
                manual: block.manual,
                value: block.value,
                transition_out_sec: block.transition_out_sec,
                created_seq: next_seq(edits).saturating_add(1),
            };
            let next = place_user_block(edits, left)?;
            place_user_block(&next, right)
        }
    }
}

pub fn resize_block(
    timeline: &TimelineState,
    edits: &ParamBlockEditSet,
    block_id: &str,
    edge: &str,
    target_frame: u32,
    frame_period_ms: f64,
) -> Result<ParamBlockEditSet, String> {
    let block = edits
        .blocks
        .iter()
        .find(|block| block.id == block_id)
        .ok_or_else(|| "Invalid: block not found".to_string())?;
    let mut next = block.clone();
    match &mut next.geometry {
        BlockGeometry::Root { span } => {
            let sec = target_frame as f64 * frame_period_ms.max(0.1) / 1000.0;
            if edge == "left" {
                span.start_sec = sec.max(0.0);
            } else {
                span.end_sec = sec.max(span.start_sec + frame_period_ms.max(0.1) / 1000.0);
            }
            if span.end_sec <= span.start_sec {
                return Err("Invalid: resized block too short".into());
            }
        }
        BlockGeometry::Take { members } => {
            if members.is_empty() {
                return Err("Invalid: block has no members".into());
            }
            let index = if edge == "left" { 0 } else { members.len() - 1 };
            let member = &mut members[index];
            let Some(clip) = timeline.clips.iter().find(|clip| clip.id == member.clip_id) else {
                return Err("Invalid: member clip missing".into());
            };
            let Some(take) = clip.take(&member.take_id) else {
                return Err("Invalid: member take missing".into());
            };
            let projection = ClipProjection::from_clip_take(clip, take);
            let source = inverse_project(&projection, target_frame, member.cycle, frame_period_ms)
                .ok_or_else(|| "Invalid: resize target outside clip".to_string())?;
            if edge == "left" {
                member.source_start_sec = source.min(member.source_end_sec - 1e-6);
            } else {
                member.source_end_sec = source.max(member.source_start_sec + 1e-6);
            }
        }
    }
    place_user_block(edits, next)
}

pub fn delete_blocks(
    edits: &ParamBlockEditSet,
    selected: &[ProjectedParamBlock],
) -> Result<ParamBlockEditSet, String> {
    let mut next = edits.clone();
    let mut tombstones: Vec<UserParamBlock> = Vec::new();
    for block in selected {
        if let Some(user) = edits.blocks.iter().find(|user| user.id == block.id) {
            next.blocks.retain(|candidate| candidate.id != user.id);
            for member in user.members() {
                tombstones.push(UserParamBlock {
                    id: new_id(
                        "tomb",
                        &format!(
                            "{}:{}:{}:{}",
                            user.id, member.clip_id, member.take_id, member.cycle
                        ),
                    ),
                    geometry: BlockGeometry::Take {
                        members: vec![member.clone()],
                    },
                    deleted: true,
                    manual: false,
                    value: None,
                    transition_out_sec: None,
                    created_seq: next.next_seq(),
                });
            }
        } else {
            for member in &block.members {
                tombstones.push(UserParamBlock {
                    id: new_id(
                        "tomb",
                        &format!(
                            "{}:{}:{}:{}",
                            block.id, member.clip_id, member.take_id, member.cycle
                        ),
                    ),
                    geometry: BlockGeometry::Take {
                        members: vec![TakeSpan {
                            clip_id: member.clip_id.clone(),
                            take_id: member.take_id.clone(),
                            cycle: member.cycle,
                            source_start_sec: member.source_start_sec,
                            source_end_sec: member.source_end_sec,
                            syllable_id: member.syllable_id.clone(),
                            label: BlockLabel::Manual,
                        }],
                    },
                    deleted: true,
                    manual: false,
                    value: None,
                    transition_out_sec: None,
                    created_seq: next.next_seq(),
                });
            }
        }
    }
    for tombstone in tombstones {
        next = place_user_block(&next, tombstone)?;
    }
    next.revision = edits.revision.saturating_add(1);
    next.validate()?;
    Ok(next)
}

pub fn create_block(
    timeline: &TimelineState,
    root_track_id: &str,
    edits: &ParamBlockEditSet,
    start_frame: u32,
    end_frame: u32,
    value: Option<f64>,
    frame_period_ms: f64,
) -> Result<ParamBlockEditSet, String> {
    if end_frame <= start_frame {
        return Err("Invalid: created block too short".into());
    }
    let start_sec = start_frame as f64 * frame_period_ms.max(0.1) / 1000.0;
    let end_sec = end_frame as f64 * frame_period_ms.max(0.1) / 1000.0;
    let mut members = Vec::new();
    for clip in &timeline.clips {
        if timeline.resolve_root_track_id(&clip.track_id).as_deref() != Some(root_track_id) {
            continue;
        }
        let take = if clip.takes.is_empty() {
            crate::state::ClipTake::from_clip(clip)
        } else {
            clip.active_take().clone()
        };
        let projection = ClipProjection::from_clip_take(clip, &take);
        let clip_start = clip.start_sec;
        let clip_end = clip.start_sec + clip.length_sec;
        if clip_end <= start_sec || clip_start >= end_sec {
            continue;
        }
        let a = start_sec.max(clip_start);
        let b = end_sec.min(clip_end);
        let Some(source_start) = inverse_project(
            &projection,
            ((a * 1000.0 / frame_period_ms.max(0.1)).round()) as u32,
            0,
            frame_period_ms,
        ) else {
            continue;
        };
        let Some(source_end) = inverse_project(
            &projection,
            ((b * 1000.0 / frame_period_ms.max(0.1)).round()) as u32,
            0,
            frame_period_ms,
        ) else {
            continue;
        };
        let (start, end) = (source_start.min(source_end), source_start.max(source_end));
        if end > start + 1e-9 {
            members.push(TakeSpan {
                clip_id: clip.id.clone(),
                take_id: take.id.clone(),
                cycle: 0,
                source_start_sec: start,
                source_end_sec: end,
                syllable_id: None,
                label: BlockLabel::Manual,
            });
        }
    }
    let geometry = if members.is_empty() {
        BlockGeometry::Root {
            span: RootSpan { start_sec, end_sec },
        }
    } else {
        BlockGeometry::Take { members }
    };
    let target = UserParamBlock {
        id: new_id(
            "ublock",
            &format!("create:{start_frame}:{end_frame}:{}", next_seq(edits)),
        ),
        geometry,
        deleted: false,
        manual: true,
        value,
        transition_out_sec: None,
        created_seq: next_seq(edits),
    };
    place_user_block(edits, target)
}

pub fn set_transition(
    edits: &ParamBlockEditSet,
    block: &ProjectedParamBlock,
    width_sec: f64,
) -> Result<ParamBlockEditSet, String> {
    let mut next = edits.clone();
    if let Some(user) = next.blocks.iter_mut().find(|user| user.id == block.id) {
        user.transition_out_sec = Some(width_sec.max(0.0));
        next.revision = edits.revision.saturating_add(1);
        next.validate()?;
        return Ok(next);
    }
    // base 块：先物化为同源用户块，再写过渡，绝不改 base。
    let members: Vec<TakeSpan> = block
        .members
        .iter()
        .map(|member| TakeSpan {
            clip_id: member.clip_id.clone(),
            take_id: member.take_id.clone(),
            cycle: member.cycle,
            source_start_sec: member.source_start_sec,
            source_end_sec: member.source_end_sec,
            syllable_id: member.syllable_id.clone(),
            label: BlockLabel::Manual,
        })
        .collect();
    if members.is_empty() {
        return Err("Invalid: transition target has no source".into());
    }
    let target = UserParamBlock {
        id: new_id("ublock", &format!("transition:{}", block.id)),
        geometry: BlockGeometry::Take { members },
        deleted: false,
        manual: false,
        value: block.value,
        transition_out_sec: Some(width_sec.max(0.0)),
        created_seq: next_seq(&next),
    };
    place_user_block(&next, target)
}

pub fn restore_game_blocks(
    timeline: &TimelineState,
    root_track_id: &str,
    edits: &ParamBlockEditSet,
    start_frame: u32,
    end_frame: u32,
    frame_period_ms: f64,
) -> Result<ParamBlockEditSet, String> {
    let start_sec = start_frame as f64 * frame_period_ms.max(0.1) / 1000.0;
    let end_sec = end_frame as f64 * frame_period_ms.max(0.1) / 1000.0;
    let mut next = edits.clone();
    next.blocks.retain(|block| match &block.geometry {
        BlockGeometry::Root { span } => span.end_sec <= start_sec || span.start_sec >= end_sec,
        BlockGeometry::Take { members } => {
            // 用户块按**投影帧**判断是否落在还原选区；源域秒与工程秒不能直接比较。
            let mut intersects = false;
            for member in members {
                let Some(clip) = timeline.clips.iter().find(|clip| clip.id == member.clip_id)
                else {
                    continue;
                };
                let Some(take) = clip.take(&member.take_id) else {
                    continue;
                };
                let projection = ClipProjection::from_clip_take(clip, take);
                if super::projection::project_span(&projection, member, frame_period_ms)
                    .iter()
                    .any(|interval| {
                        interval.end_frame_exclusive > start_frame
                            && interval.start_frame < end_frame
                    })
                {
                    intersects = true;
                    break;
                }
            }
            !intersects
        }
    });
    for clip in &timeline.clips {
        if timeline.resolve_root_track_id(&clip.track_id).as_deref() != Some(root_track_id) {
            continue;
        }
        let take = if clip.takes.is_empty() {
            crate::state::ClipTake::from_clip(clip)
        } else {
            clip.active_take().clone()
        };
        let Some(baseline) = take.game_block_baseline.as_ref() else {
            continue;
        };
        let projection = ClipProjection::from_clip_take(clip, &take);
        for (index, span) in baseline.source_spans_sec.iter().enumerate() {
            let candidate = TakeSpan {
                clip_id: clip.id.clone(),
                take_id: take.id.clone(),
                cycle: 0,
                source_start_sec: span.start_sec,
                source_end_sec: span.end_sec,
                syllable_id: Some(format!("game:{index}:{}", span.start_sec)),
                label: BlockLabel::GameBaseline,
            };
            let intersects =
                super::projection::project_span(&projection, &candidate, frame_period_ms)
                    .iter()
                    .any(|interval| {
                        interval.end_frame_exclusive > start_frame
                            && interval.start_frame < end_frame
                    });
            if !intersects {
                continue;
            }
            let target = UserParamBlock {
                id: new_id(
                    "ublock",
                    &format!("restore:{}:{}:{}", clip.id, index, span.start_sec),
                ),
                geometry: BlockGeometry::Take {
                    members: vec![candidate],
                },
                deleted: false,
                manual: false,
                value: None,
                transition_out_sec: None,
                created_seq: next.next_seq(),
            };
            next = place_user_block(&next, target)?;
        }
    }
    next.revision = edits.revision.saturating_add(1);
    next.validate()?;
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::param_blocks::compose::compose_param_block_view;
    use crate::state::TimelineState;

    fn timeline() -> TimelineState {
        let mut timeline: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"root","name":"Root","order":0}],
            "clips":[{
                "id":"clip","track_id":"root","name":"clip","start_sec":0.0,"length_sec":2.0,
                "takes":[{"id":"take","source_path":"a.wav","source_start_sec":0.0,"source_end_sec":2.0,
                    "syllable_analysis":{"algorithm":"test","algorithmVersion":1,"sourceDigest":"d",
                        "sourceSpansSec":[{"startSec":0.0,"endSec":0.5},{"startSec":0.5,"endSec":1.0}]}}]
            }],
            "project_sec":4,"bpm":120
        })).unwrap();
        timeline.clips[0].normalize_takes();
        timeline
    }

    fn base_view(timeline: &TimelineState) -> crate::param_blocks::ParamBlockView {
        compose_param_block_view(timeline, "root", &ParamBlockEditSet::default(), 5.0).unwrap()
    }

    #[test]
    fn merge_split_resize_delete_and_restore_are_source_anchored() {
        let timeline = timeline();
        let view = base_view(&timeline);
        let merged = merge_blocks(&ParamBlockEditSet::default(), &view.blocks, 5.0).unwrap();
        let merged_view = compose_param_block_view(&timeline, "root", &merged, 5.0).unwrap();
        assert_eq!(merged_view.blocks.len(), 1);
        assert_eq!(
            merged_view.blocks[0].kind,
            crate::param_blocks::ProjectedBlockKind::Merged
        );

        let split = split_block(&timeline, &merged, &merged_view.blocks[0].id, 50, 5.0).unwrap();
        let split_view = compose_param_block_view(&timeline, "root", &split, 5.0).unwrap();
        assert_eq!(split_view.blocks.len(), 2);
        assert_eq!(split_view.blocks[0].start_frame, 0);
        assert_eq!(split_view.blocks[0].end_frame_exclusive, 50);
        assert_eq!(split_view.blocks[1].start_frame, 50);

        let resized = resize_block(
            &timeline,
            &split,
            &split_view.blocks[1].id,
            "right",
            150,
            5.0,
        )
        .unwrap();
        let resized_view = compose_param_block_view(&timeline, "root", &resized, 5.0).unwrap();
        assert_eq!(resized_view.blocks[1].end_frame_exclusive, 150);

        let deleted = delete_blocks(&split, &split_view.blocks).unwrap();
        let deleted_view = compose_param_block_view(&timeline, "root", &deleted, 5.0).unwrap();
        assert!(deleted_view.blocks.is_empty());

        let restored = restore_game_blocks(&timeline, "root", &deleted, 0, 200, 5.0).unwrap();
        let restored_view = compose_param_block_view(&timeline, "root", &restored, 5.0).unwrap();
        assert_eq!(restored_view.blocks.len(), 2);
    }

    #[test]
    fn created_block_over_take_is_a_take_span_not_an_absolute_interval() {
        let timeline = timeline();
        let created = create_block(
            &timeline,
            "root",
            &ParamBlockEditSet::default(),
            20,
            80,
            Some(61.0),
            5.0,
        )
        .unwrap();
        let block = &created.blocks[0];
        match &block.geometry {
            BlockGeometry::Take { members } => {
                assert_eq!(members[0].clip_id, "clip");
                assert!((members[0].source_start_sec - 0.1).abs() < 1e-9);
                assert!((members[0].source_end_sec - 0.4).abs() < 1e-9);
            }
            BlockGeometry::Root { .. } => panic!("take block must not be a root block"),
        }
    }

    #[test]
    fn transition_materializes_a_base_block_instead_of_mutating_base() {
        let timeline = timeline();
        let view = base_view(&timeline);
        let next = set_transition(&ParamBlockEditSet::default(), &view.blocks[0], 0.02).unwrap();
        let next_view = compose_param_block_view(&timeline, "root", &next, 5.0).unwrap();
        assert_eq!(next_view.blocks[0].transition_frames, Some(4));
    }
}
