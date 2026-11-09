//! 参数块的源域编辑原语。
//!
//! 职责：所有编辑都只改写 \`ParamBlockEditSet\`，绝不写绝对帧。
//! 关系：\`compose\` 消费编辑结果；App/插件只通过本文件改块。
//! 维护约束：新块必须经过 \`place_user_block\`，由它裁剪被覆盖的旧用户块。

use super::projection::{inverse_project, project_span, ClipProjection};
use super::types::{
    BlockGeometry, BlockLabel, ParamBlockEditSet, ProjectedBlockKind, ProjectedParamBlock,
    RootSpan, TakeSpan, UserParamBlock,
};
use crate::state::{ClipTake, TimelineState};

const EPS: f64 = 1e-9;
const MAX_LOOP_CYCLES: u32 = 20_000;

fn new_id(prefix: &str, identity: &str) -> String {
    format!("{prefix}:{}", blake3::hash(identity.as_bytes()).to_hex())
}

fn next_seq(edits: &ParamBlockEditSet) -> u64 {
    edits.next_seq()
}

fn subtract_interval(start: f64, end: f64, cut_start: f64, cut_end: f64) -> Vec<(f64, f64)> {
    if cut_end <= start + EPS || cut_start >= end - EPS {
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

fn active_take(clip: &crate::state::Clip) -> ClipTake {
    if clip.takes.is_empty() {
        ClipTake::from_clip(clip)
    } else {
        clip.active_take().clone()
    }
}

fn same_member(left: &TakeSpan, right: &TakeSpan) -> bool {
    left.clip_id == right.clip_id
        && left.take_id == right.take_id
        && left.cycle == right.cycle
        && (left.source_start_sec - right.source_start_sec).abs() <= 1e-7
        && (left.source_end_sec - right.source_end_sec).abs() <= 1e-7
}

fn same_member_set(left: &[TakeSpan], right: &[TakeSpan]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .all(|member| right.iter().any(|other| same_member(member, other)))
}

fn root_piece_id(block: &UserParamBlock, index: usize, start: f64, end: f64) -> String {
    if index == 0 {
        block.id.clone()
    } else {
        new_id(
            "ublock",
            &format!("root-piece:{}:{start:.9}:{end:.9}", block.id),
        )
    }
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
        match (&block.geometry, &target.geometry) {
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
                        if end > start + EPS {
                            let mut part = member.clone();
                            part.source_start_sec = start;
                            part.source_end_sec = end;
                            kept.push(part);
                        }
                    }
                }
                if !kept.is_empty() {
                    let mut next = block.clone();
                    next.geometry = BlockGeometry::Take { members: kept };
                    blocks.push(next);
                }
            }
            (BlockGeometry::Root { span }, BlockGeometry::Root { span: cut }) => {
                let pieces =
                    subtract_interval(span.start_sec, span.end_sec, cut.start_sec, cut.end_sec);
                for (index, (start, end)) in pieces.into_iter().enumerate() {
                    if end <= start + EPS {
                        continue;
                    }
                    let mut next = block.clone();
                    next.id = root_piece_id(block, index, start, end);
                    next.geometry = BlockGeometry::Root {
                        span: RootSpan {
                            start_sec: start,
                            end_sec: end,
                        },
                    };
                    blocks.push(next);
                }
            }
            _ => blocks.push(block.clone()),
        }
    }
    blocks.push(target);
    let mut next = edits.clone();
    next.blocks = blocks;
    next.revision = edits.revision.saturating_add(1);
    next.validate()?;
    Ok(next)
}

fn projected_members_to_spans(
    timeline: &TimelineState,
    block: &ProjectedParamBlock,
    frame_period_ms: f64,
) -> Result<Vec<TakeSpan>, String> {
    if block.members.is_empty() {
        return Err("Invalid: block has no source".into());
    }
    let mut out = Vec::new();
    for member in &block.members {
        let Some(clip) = timeline.clips.iter().find(|clip| clip.id == member.clip_id) else {
            return Err("Invalid: member clip missing".into());
        };
        let Some(take) = clip.take(&member.take_id) else {
            return Err("Invalid: member take missing".into());
        };
        let projection = ClipProjection::from_clip_take(clip, take);
        let source_a = inverse_project(
            &projection,
            member.start_frame,
            member.cycle,
            frame_period_ms,
        );
        let source_b = inverse_project(
            &projection,
            member.end_frame_exclusive,
            member.cycle,
            frame_period_ms,
        );
        let (start, end) = match (source_a, source_b) {
            (Some(a), Some(b)) => {
                let lo = a.min(b).max(member.source_start_sec);
                let hi = a.max(b).min(member.source_end_sec);
                if hi > lo + EPS {
                    (lo, hi)
                } else {
                    (member.source_start_sec, member.source_end_sec)
                }
            }
            _ => (member.source_start_sec, member.source_end_sec),
        };
        if end > start + EPS {
            out.push(TakeSpan {
                clip_id: member.clip_id.clone(),
                take_id: member.take_id.clone(),
                cycle: member.cycle,
                source_start_sec: start,
                source_end_sec: end,
                syllable_id: member.syllable_id.clone(),
                label: BlockLabel::Manual,
            });
        }
    }
    if out.is_empty() {
        return Err("Invalid: block has no source".into());
    }
    Ok(out)
}

fn materialize_projected_fragment(
    timeline: &TimelineState,
    edits: &ParamBlockEditSet,
    block: &ProjectedParamBlock,
    frame_period_ms: f64,
) -> Result<(ParamBlockEditSet, UserParamBlock), String> {
    let parent_id = block.user_block_id.as_deref().unwrap_or(block.id.as_str());
    let Some(parent) = edits
        .blocks
        .iter()
        .find(|candidate| candidate.id == parent_id)
        .cloned()
    else {
        return Err("Invalid: block not found".into());
    };
    if parent.grouped_from.is_some() {
        return Ok((edits.clone(), parent));
    }
    let fragment = projected_members_to_spans(timeline, block, frame_period_ms)?;
    if let BlockGeometry::Take { members } = &parent.geometry {
        if same_member_set(members, &fragment) {
            return Ok((edits.clone(), parent));
        }
    }

    let mut next = edits.clone();
    next.blocks.retain(|candidate| candidate.id != parent.id);
    next.revision = edits.revision.saturating_add(1);

    let mut remaining_members = Vec::new();
    if let BlockGeometry::Take { members } = &parent.geometry {
        for member in members {
            let mut pieces = vec![(member.source_start_sec, member.source_end_sec)];
            for cut in &fragment {
                if cut.clip_id != member.clip_id
                    || cut.take_id != member.take_id
                    || cut.cycle != member.cycle
                {
                    continue;
                }
                let mut split = Vec::new();
                for (start, end) in pieces {
                    split.extend(subtract_interval(
                        start,
                        end,
                        cut.source_start_sec,
                        cut.source_end_sec,
                    ));
                }
                pieces = split;
            }
            for (start, end) in pieces {
                if end > start + EPS {
                    let mut part = member.clone();
                    part.source_start_sec = start;
                    part.source_end_sec = end;
                    remaining_members.push(part);
                }
            }
        }
    }
    if !remaining_members.is_empty() {
        let mut sibling = parent.clone();
        sibling.geometry = BlockGeometry::Take {
            members: remaining_members,
        };
        next = place_user_block(&next, sibling)?;
    }

    let child_id = if block.id == parent.id || next.blocks.iter().any(|item| item.id == block.id) {
        new_id(
            "fragment",
            &format!(
                "{}:{}:{}",
                parent.id, block.start_frame, block.end_frame_exclusive
            ),
        )
    } else {
        block.id.clone()
    };
    let child = UserParamBlock {
        id: child_id,
        geometry: BlockGeometry::Take { members: fragment },
        deleted: false,
        manual: parent.manual,
        value: parent.value,
        transition_out_sec: parent.transition_out_sec,
        created_seq: next.next_seq(),
        grouped_from: None,
    };
    let next = place_user_block(&next, child.clone())?;
    Ok((next, child))
}

/// 确保一个投影块可以按普通用户块编辑；base 与多 run 片段在这里物化。
pub fn ensure_editable_user_block(
    timeline: &TimelineState,
    edits: &ParamBlockEditSet,
    block: &ProjectedParamBlock,
    frame_period_ms: f64,
) -> Result<(ParamBlockEditSet, UserParamBlock), String> {
    if let Some(direct) = edits.blocks.iter().find(|item| item.id == block.id) {
        return Ok((edits.clone(), direct.clone()));
    }
    if block
        .user_block_id
        .as_deref()
        .is_some_and(|id| id != block.id)
    {
        return materialize_projected_fragment(timeline, edits, block, frame_period_ms);
    }
    let members = projected_members_to_spans(timeline, block, frame_period_ms)?;
    let target = UserParamBlock {
        id: block
            .user_block_id
            .clone()
            .unwrap_or_else(|| block.id.clone()),
        geometry: BlockGeometry::Take { members },
        deleted: false,
        manual: matches!(block.kind, ProjectedBlockKind::Manual),
        value: block.value,
        transition_out_sec: block
            .transition_frames
            .map(|frames| frames as f64 * frame_period_ms.max(0.1) / 1000.0),
        created_seq: next_seq(edits),
        grouped_from: None,
    };
    let next = place_user_block(edits, target.clone())?;
    Ok((next, target))
}

pub fn group_blocks(
    timeline: &TimelineState,
    edits: &ParamBlockEditSet,
    selected: &[ProjectedParamBlock],
    frame_period_ms: f64,
) -> Result<ParamBlockEditSet, String> {
    if selected.len() < 2 {
        return Err("Invalid: group needs two blocks".into());
    }
    let mut ordered = selected.to_vec();
    ordered.sort_by_key(|block| block.start_frame);
    for pair in ordered.windows(2) {
        if pair[0].end_frame_exclusive != pair[1].start_frame {
            return Err("Invalid: group blocks are not adjacent".into());
        }
    }

    let mut working = edits.clone();
    let mut snapshots = Vec::new();
    for block in &ordered {
        let (next, user) = ensure_editable_user_block(timeline, &working, block, frame_period_ms)?;
        working = next;
        snapshots.push(user);
    }
    if snapshots.iter().all(|block| block.id == snapshots[0].id) {
        return Err("Invalid: blocks are already grouped".into());
    }
    let has_root = snapshots
        .iter()
        .any(|block| matches!(block.geometry, BlockGeometry::Root { .. }));
    let has_take = snapshots.iter().any(UserParamBlock::is_take);
    if has_root && has_take {
        return Err("Invalid: cannot group root and take blocks together".into());
    }

    let members: Vec<TakeSpan> = snapshots
        .iter()
        .flat_map(|block| block.members().iter().cloned())
        .collect();
    let geometry = if !has_take {
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
    let values: Vec<Option<f64>> = snapshots.iter().map(|block| block.value).collect();
    let value = values
        .first()
        .copied()
        .flatten()
        .filter(|first| values.iter().all(|value| *value == Some(*first)));
    let manual = snapshots.iter().all(|block| block.manual);
    let transition_out_sec = snapshots.last().and_then(|block| block.transition_out_sec);
    let target = UserParamBlock {
        id: new_id("ublock", &format!("group:{}", ordered[0].id)),
        geometry,
        deleted: false,
        manual,
        value,
        transition_out_sec,
        created_seq: working.next_seq(),
        grouped_from: Some(snapshots),
    };
    place_user_block(&working, target)
}

pub fn ungroup_user_block(
    edits: &ParamBlockEditSet,
    block_id: &str,
) -> Result<ParamBlockEditSet, String> {
    let block = edits
        .blocks
        .iter()
        .find(|item| item.id == block_id)
        .ok_or_else(|| "Invalid: block not found".to_string())?;
    let Some(children) = block.grouped_from.clone() else {
        return Err("Invalid: block is not grouped".into());
    };
    if children.len() < 2 {
        return Err("Invalid: grouped block has no members".into());
    }
    let mut next = edits.clone();
    next.blocks.retain(|item| item.id != block_id);
    next.revision = edits.revision.saturating_add(1);
    for child in children {
        next = place_user_block(&next, child)?;
    }
    next.revision = edits.revision.saturating_add(1);
    next.validate()?;
    Ok(next)
}

fn find_projected_block(
    timeline: &TimelineState,
    edits: &ParamBlockEditSet,
    block_id: &str,
    frame_period_ms: f64,
) -> Result<ProjectedParamBlock, String> {
    if let Some(user) = edits.blocks.iter().find(|block| block.id == block_id) {
        if let Some(projected) =
            super::compose::project_single_user_block(timeline, user, frame_period_ms)?
        {
            return Ok(projected);
        }
    }
    let roots: std::collections::BTreeSet<&str> = timeline
        .tracks
        .iter()
        .map(|track| track.id.as_str())
        .collect();
    for root in roots {
        if let Ok(view) =
            super::compose::compose_param_block_view(timeline, root, edits, frame_period_ms)
        {
            if let Some(block) = view.blocks.into_iter().find(|block| block.id == block_id) {
                return Ok(block);
            }
        }
    }
    Err("Invalid: block not found".into())
}

/** 一个成员在本帧域里的可见区间（与 `compose` 的 run 同源）。 */
struct SplitMemberInterval<'a> {
    index: usize,
    member: &'a TakeSpan,
    projection: ClipProjection,
    start_frame: u32,
    end_frame_exclusive: u32,
}

/** 要把 `frame` 挪多少帧才能严格落进该成员内部（0 = 已经在内）。 */
fn frames_to_inside(interval: &SplitMemberInterval<'_>, frame: u32) -> i64 {
    if frame <= interval.start_frame {
        interval.start_frame as i64 + 1 - frame as i64
    } else if frame >= interval.end_frame_exclusive {
        frame as i64 - (interval.end_frame_exclusive as i64 - 1)
    } else {
        0
    }
}

/**
 * 找出"切在哪一个成员上"，并把切点钳进它内部。
 *
 * 【为什么需要兜底】合成合并 run 的判据是 `start <= last.end + 1`（相差 1 帧也并），
 * 因此块的 `[start_frame, end_frame_exclusive)` 里可能存在**成员内无帧**的缝。
 * 旧实现只做"哪个成员严格包含该帧"，落进缝里就返回 `split outside block` ——
 * 于是"菜单可点、播放光标在块内，却切不了"。这里改为：先找严格包含的成员；找不到
 * 就取**帧距离最近**的成员并把切点钳进它的内部。
 *
 * 钳位后仍然由该成员的投影反算源域切点，所以切出来的两半首尾相接、无重叠 ——
 * 与"正好点在成员内"的结果只差不到 1 帧。
 */
fn resolve_split_member(
    timeline: &TimelineState,
    members: &[TakeSpan],
    frame: u32,
    frame_period_ms: f64,
) -> Option<(usize, f64)> {
    let mut intervals: Vec<SplitMemberInterval<'_>> = Vec::new();
    for (index, member) in members.iter().enumerate() {
        let Some(clip) = timeline.clips.iter().find(|clip| clip.id == member.clip_id) else {
            continue;
        };
        let Some(take) = clip.take(&member.take_id) else {
            continue;
        };
        let projection = ClipProjection::from_clip_take(clip, take);
        let projected: Vec<_> = project_span(&projection, member, frame_period_ms)
            .into_iter()
            .filter(|interval| interval.cycle == member.cycle)
            .collect();
        let (Some(first), Some(last)) = (projected.first(), projected.last()) else {
            continue;
        };
        intervals.push(SplitMemberInterval {
            index,
            member,
            projection,
            start_frame: first.start_frame,
            end_frame_exclusive: last.end_frame_exclusive,
        });
    }
    if intervals.is_empty() {
        return None;
    }

    let mut hit = intervals
        .iter()
        .find(|interval| frames_to_inside(interval, frame) == 0);
    let mut split_frame = frame;
    if hit.is_none() {
        let nearest = intervals
            .iter()
            .min_by_key(|interval| frames_to_inside(interval, frame))?;
        let clamped = frame
            .max(nearest.start_frame.saturating_add(1))
            .min(nearest.end_frame_exclusive.saturating_sub(1));
        if clamped <= nearest.start_frame || clamped >= nearest.end_frame_exclusive {
            return None;
        }
        split_frame = clamped;
        hit = intervals
            .iter()
            .find(|interval| interval.index == nearest.index);
    }
    let hit = hit?;
    let source_sec = inverse_project(
        &hit.projection,
        split_frame,
        hit.member.cycle,
        frame_period_ms,
    )?;
    Some((hit.index, source_sec))
}

pub fn split_block(
    timeline: &TimelineState,
    edits: &ParamBlockEditSet,
    block_id: &str,
    frame: u32,
    frame_period_ms: f64,
) -> Result<ParamBlockEditSet, String> {
    let projected = find_projected_block(timeline, edits, block_id, frame_period_ms)?;
    let (working, block) =
        ensure_editable_user_block(timeline, edits, &projected, frame_period_ms)?;
    if block.grouped_from.is_some() {
        return Err("Invalid: grouped block must be ungrouped first".into());
    }
    match &block.geometry {
        BlockGeometry::Root { span } => {
            let split_sec = frame as f64 * frame_period_ms.max(0.1) / 1000.0;
            if split_sec <= span.start_sec + EPS || split_sec >= span.end_sec - EPS {
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
                created_seq: next_seq(&working),
                grouped_from: None,
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
                created_seq: next_seq(&working).saturating_add(1),
                grouped_from: None,
            };
            let next = place_user_block(&working, left)?;
            place_user_block(&next, right)
        }
        BlockGeometry::Take { members } => {
            let mut ordered = members.clone();
            ordered.sort_by(|a, b| a.source_start_sec.total_cmp(&b.source_start_sec));
            let Some((index, source_split)) =
                resolve_split_member(timeline, &ordered, frame, frame_period_ms)
            else {
                return Err("Invalid: split outside block".into());
            };
            let containing = &ordered[index];
            if source_split <= containing.source_start_sec + EPS
                || source_split >= containing.source_end_sec - EPS
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
                created_seq: next_seq(&working),
                grouped_from: None,
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
                created_seq: next_seq(&working).saturating_add(1),
                grouped_from: None,
            };
            let next = place_user_block(&working, left)?;
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
    let projected = find_projected_block(timeline, edits, block_id, frame_period_ms)?;
    let (working, block) =
        ensure_editable_user_block(timeline, edits, &projected, frame_period_ms)?;
    if block.grouped_from.is_some() {
        return Err("Invalid: grouped block must be ungrouped first".into());
    }
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
            let mut indices: Vec<usize> = (0..members.len()).collect();
            indices.sort_by(|left, right| {
                members[*left]
                    .source_start_sec
                    .total_cmp(&members[*right].source_start_sec)
                    .then(
                        members[*left]
                            .source_end_sec
                            .total_cmp(&members[*right].source_end_sec),
                    )
            });
            let index = if edge == "left" {
                *indices.first().unwrap_or(&0)
            } else {
                *indices.last().unwrap_or(&0)
            };
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
    place_user_block(&working, next)
}

fn tombstone_matches(tombstone: &UserParamBlock, member: &TakeSpan) -> bool {
    if !tombstone.deleted {
        return false;
    }
    let BlockGeometry::Take { members } = &tombstone.geometry else {
        return false;
    };
    members.len() == 1 && same_member(&members[0], member)
}

fn make_tombstone(user: &UserParamBlock, member: &TakeSpan, seq: u64) -> UserParamBlock {
    UserParamBlock {
        id: new_id(
            "tomb",
            &format!(
                "{}:{}:{}:{}:{:.9}:{:.9}",
                user.id,
                member.clip_id,
                member.take_id,
                member.cycle,
                member.source_start_sec,
                member.source_end_sec
            ),
        ),
        geometry: BlockGeometry::Take {
            members: vec![member.clone()],
        },
        deleted: true,
        manual: false,
        value: None,
        transition_out_sec: None,
        created_seq: seq,
        grouped_from: None,
    }
}

#[derive(Clone, Debug)]
pub struct DeletedParamBlockSnapshot {
    pub block: UserParamBlock,
    pub previous_edits: ParamBlockEditSet,
}

/// 删除块并返回精确到源区间的可逆快照。base 与多 run 片段先物化。
pub fn delete_blocks(
    timeline: &TimelineState,
    edits: &ParamBlockEditSet,
    selected: &[ProjectedParamBlock],
    frame_period_ms: f64,
) -> Result<(ParamBlockEditSet, Vec<DeletedParamBlockSnapshot>), String> {
    let mut working = edits.clone();
    let mut snapshots = Vec::new();
    let mut removed = std::collections::BTreeSet::new();
    for block in selected {
        let (next, user) = ensure_editable_user_block(timeline, &working, block, frame_period_ms)?;
        working = next;
        if !removed.insert(user.id.clone()) {
            continue;
        }
        let before = working.clone();
        working.blocks.retain(|candidate| candidate.id != user.id);
        snapshots.push(DeletedParamBlockSnapshot {
            block: user.clone(),
            previous_edits: before,
        });
    }
    if snapshots.is_empty() {
        return Err("Invalid: block not found".into());
    }
    let mut tombstones = Vec::new();
    for snapshot in &snapshots {
        for member in snapshot.block.members() {
            if working
                .blocks
                .iter()
                .any(|candidate| tombstone_matches(candidate, member))
            {
                continue;
            }
            tombstones.push(make_tombstone(&snapshot.block, member, working.next_seq()));
        }
    }
    for tombstone in tombstones {
        working = place_user_block(&working, tombstone)?;
    }
    working.revision = edits.revision.saturating_add(1);
    working.validate()?;
    Ok((working, snapshots))
}

/// 按快照恢复删除的块，只移除与快照成员精确匹配的墓碑。
pub fn restore_deleted_blocks(
    edits: &ParamBlockEditSet,
    snapshots: &[DeletedParamBlockSnapshot],
) -> Result<ParamBlockEditSet, String> {
    if snapshots.is_empty() {
        return Err("Invalid: no deleted blocks".into());
    }
    let mut next = edits.clone();
    for snapshot in snapshots {
        next.blocks.retain(|candidate| {
            !snapshot
                .block
                .members()
                .iter()
                .any(|member| tombstone_matches(candidate, member))
        });
        next = place_user_block(&next, snapshot.block.clone())?;
    }
    next.revision = edits.revision.saturating_add(1);
    next.validate()?;
    Ok(next)
}

fn source_spans_for_timeline_range(
    clip: &crate::state::Clip,
    start_frame: u32,
    end_frame: u32,
    frame_period_ms: f64,
) -> Vec<TakeSpan> {
    let take = active_take(clip);
    let projection = ClipProjection::from_clip_take(clip, &take);
    let start_sec = start_frame as f64 * frame_period_ms.max(0.1) / 1000.0;
    let end_sec = end_frame as f64 * frame_period_ms.max(0.1) / 1000.0;
    let rate = projection.effective_rate.max(1e-9);
    let max_cycles = if projection.loop_enabled {
        projection
            .loop_cycle_sec
            .map(|cycle| {
                ((clip.length_sec.max(0.0) / (cycle / rate).max(1e-9)).ceil() as u32)
                    .saturating_add(2)
                    .min(MAX_LOOP_CYCLES)
            })
            .unwrap_or(1)
    } else {
        1
    };
    let mut out = Vec::new();
    for cycle in 0..max_cycles {
        let (cycle_start, cycle_end) = if projection.loop_enabled {
            let cycle_sec = projection.loop_cycle_sec.unwrap_or(0.0) / rate;
            if cycle_sec <= EPS {
                continue;
            }
            (
                clip.start_sec + cycle as f64 * cycle_sec,
                clip.start_sec + (cycle + 1) as f64 * cycle_sec,
            )
        } else {
            (clip.start_sec, clip.start_sec + clip.length_sec)
        };
        let a = start_sec.max(cycle_start).max(clip.start_sec);
        let b = end_sec.min(cycle_end).min(clip.start_sec + clip.length_sec);
        if b <= a + EPS {
            continue;
        }
        let a_frame = projection.to_frame(a, frame_period_ms);
        let mut b_frame = projection.to_frame(b, frame_period_ms);
        let ends_at_cycle_boundary = projection.loop_enabled && b >= cycle_end - EPS;
        if ends_at_cycle_boundary {
            b_frame = b_frame.saturating_sub(1);
        }
        let Some(source_a) = inverse_project(&projection, a_frame, cycle, frame_period_ms) else {
            continue;
        };
        let Some(source_b) = inverse_project(&projection, b_frame, cycle, frame_period_ms) else {
            continue;
        };
        let mut source_end = source_a.max(source_b);
        if ends_at_cycle_boundary {
            source_end += frame_period_ms.max(0.1) / 1000.0 * rate;
        }
        let source_start = source_a.min(source_b);
        if source_end > source_start + EPS {
            out.push(TakeSpan {
                clip_id: clip.id.clone(),
                take_id: take.id.clone(),
                cycle,
                source_start_sec: source_start,
                source_end_sec: source_end,
                syllable_id: None,
                label: BlockLabel::Manual,
            });
        }
    }
    out
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
        if clip.start_sec + clip.length_sec <= start_sec || clip.start_sec >= end_sec {
            continue;
        }
        members.extend(source_spans_for_timeline_range(
            clip,
            start_frame,
            end_frame,
            frame_period_ms,
        ));
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
        grouped_from: None,
    };
    place_user_block(edits, target)
}

pub fn set_transition(
    timeline: &TimelineState,
    edits: &ParamBlockEditSet,
    block: &ProjectedParamBlock,
    width_sec: f64,
    frame_period_ms: f64,
) -> Result<ParamBlockEditSet, String> {
    let (mut next, user) = ensure_editable_user_block(timeline, edits, block, frame_period_ms)?;
    let Some(target) = next.blocks.iter_mut().find(|item| item.id == user.id) else {
        return Err("Invalid: transition target not found".into());
    };
    target.transition_out_sec = Some(width_sec.max(0.0));
    next.revision = edits.revision.saturating_add(1);
    next.validate()?;
    Ok(next)
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
                if project_span(&projection, member, frame_period_ms)
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
        let take = active_take(clip);
        let Some(baseline) = take.game_block_baseline.as_ref() else {
            continue;
        };
        let projection = ClipProjection::from_clip_take(clip, &take);
        let max_cycles = if projection.loop_enabled {
            projection
                .loop_cycle_sec
                .map(|cycle| {
                    ((clip.length_sec.max(0.0)
                        / (cycle / projection.effective_rate.max(1e-9)).max(1e-9))
                    .ceil() as u32)
                        .saturating_add(2)
                        .min(MAX_LOOP_CYCLES)
                })
                .unwrap_or(1)
        } else {
            1
        };
        for (index, span) in baseline.source_spans_sec.iter().enumerate() {
            for cycle in 0..max_cycles {
                let candidate = TakeSpan {
                    clip_id: clip.id.clone(),
                    take_id: take.id.clone(),
                    cycle,
                    source_start_sec: span.start_sec,
                    source_end_sec: span.end_sec,
                    syllable_id: Some(format!("game:{index}:{}", span.start_sec)),
                    label: BlockLabel::GameBaseline,
                };
                let intersects = project_span(&projection, &candidate, frame_period_ms)
                    .into_iter()
                    .filter(|interval| interval.cycle == cycle)
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
                        &format!("restore:{}:{}:{}:{}", clip.id, index, span.start_sec, cycle),
                    ),
                    geometry: BlockGeometry::Take {
                        members: vec![candidate],
                    },
                    deleted: false,
                    manual: false,
                    value: None,
                    transition_out_sec: None,
                    created_seq: next.next_seq(),
                    grouped_from: None,
                };
                next = place_user_block(&next, target)?;
            }
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
    use crate::param_blocks::ParamBlockView;
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

    fn base_view(timeline: &TimelineState) -> ParamBlockView {
        compose_param_block_view(timeline, "root", &ParamBlockEditSet::default(), 5.0).unwrap()
    }

    #[test]
    fn empty_edit_set_serializes_blocks_array() {
        let value = serde_json::to_value(ParamBlockEditSet::default()).unwrap();
        assert_eq!(value["blocks"], serde_json::json!([]));
    }

    #[test]
    fn base_blocks_materialize_split_resize_and_delete() {
        let timeline = timeline();
        let view = base_view(&timeline);
        let split = split_block(
            &timeline,
            &ParamBlockEditSet::default(),
            &view.blocks[0].id,
            50,
            5.0,
        )
        .unwrap();
        let split_view = compose_param_block_view(&timeline, "root", &split, 5.0).unwrap();
        assert_eq!(split_view.blocks.len(), 3);
        assert_eq!(split_view.blocks[0].end_frame_exclusive, 50);
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
        let (deleted, snapshots) =
            delete_blocks(&timeline, &resized, &resized_view.blocks, 5.0).unwrap();
        assert_eq!(snapshots.len(), 2);
        let deleted_view = compose_param_block_view(&timeline, "root", &deleted, 5.0).unwrap();
        assert!(deleted_view.blocks.is_empty());
        let restored = restore_deleted_blocks(&deleted, &snapshots).unwrap();
        let restored_view = compose_param_block_view(&timeline, "root", &restored, 5.0).unwrap();
        assert_eq!(restored_view.blocks.len(), 2);
    }

    #[test]
    fn group_and_ungroup_restore_original_source_blocks() {
        let timeline = timeline();
        let view = base_view(&timeline);
        let grouped =
            group_blocks(&timeline, &ParamBlockEditSet::default(), &view.blocks, 5.0).unwrap();
        let grouped_view = compose_param_block_view(&timeline, "root", &grouped, 5.0).unwrap();
        assert_eq!(grouped_view.blocks.len(), 1);
        assert_eq!(
            grouped_view.blocks[0].kind,
            crate::param_blocks::ProjectedBlockKind::Grouped
        );
        let ungrouped = ungroup_user_block(&grouped, &grouped_view.blocks[0].id).unwrap();
        let ungrouped_view = compose_param_block_view(&timeline, "root", &ungrouped, 5.0).unwrap();
        assert_eq!(ungrouped_view.blocks.len(), 2);
        assert_eq!(ungrouped.blocks[0].id, view.blocks[0].id);
        assert_eq!(ungrouped.blocks[1].id, view.blocks[1].id);
    }

    #[test]
    fn grouped_blocks_reject_split_and_resize() {
        let timeline = timeline();
        let view = base_view(&timeline);
        let grouped =
            group_blocks(&timeline, &ParamBlockEditSet::default(), &view.blocks, 5.0).unwrap();
        let grouped_view = compose_param_block_view(&timeline, "root", &grouped, 5.0).unwrap();
        assert!(split_block(&timeline, &grouped, &grouped_view.blocks[0].id, 50, 5.0).is_err());
        assert!(resize_block(
            &timeline,
            &grouped,
            &grouped_view.blocks[0].id,
            "right",
            150,
            5.0
        )
        .is_err());
    }

    #[test]
    fn root_block_middle_cut_keeps_both_pieces() {
        let timeline = timeline();
        let target = UserParamBlock {
            id: "root".into(),
            geometry: BlockGeometry::Root {
                span: RootSpan {
                    start_sec: 0.0,
                    end_sec: 1.0,
                },
            },
            deleted: false,
            manual: true,
            value: Some(60.0),
            transition_out_sec: None,
            created_seq: 1,
            grouped_from: None,
        };
        let cut = UserParamBlock {
            id: "cut".into(),
            geometry: BlockGeometry::Root {
                span: RootSpan {
                    start_sec: 0.4,
                    end_sec: 0.6,
                },
            },
            deleted: false,
            manual: true,
            value: Some(61.0),
            transition_out_sec: None,
            created_seq: 2,
            grouped_from: None,
        };
        let next = place_user_block(&ParamBlockEditSet::default(), target).unwrap();
        let next = place_user_block(&next, cut).unwrap();
        let view = compose_param_block_view(&timeline, "root", &next, 5.0).unwrap();
        assert!(view.blocks.iter().any(|block| block.start_frame == 0));
        assert!(view
            .blocks
            .iter()
            .any(|block| block.end_frame_exclusive == 200));
    }

    /// 两个**相差 1 帧**的成员：合成会把它们并成一个 run，并在帧 100 处留下
    /// "块内有、成员内无"的缝（合并容差是 `start <= last.end + 1`）。
    fn gap_member_edits() -> ParamBlockEditSet {
        let member = |start: f64, end: f64| TakeSpan {
            clip_id: "clip".into(),
            take_id: "take".into(),
            cycle: 0,
            source_start_sec: start,
            source_end_sec: end,
            syllable_id: None,
            label: BlockLabel::Manual,
        };
        ParamBlockEditSet {
            schema_version: 1,
            revision: 1,
            blocks: vec![UserParamBlock {
                id: "user".into(),
                geometry: BlockGeometry::Take {
                    members: vec![member(0.0, 0.5), member(0.505, 1.0)],
                },
                deleted: false,
                manual: false,
                value: Some(60.0),
                transition_out_sec: None,
                created_seq: 1,
                grouped_from: None,
            }],
        }
    }

    /*
     * 【run 合并容差的缝】旧实现只找"严格包含该帧的成员"，落进缝里就
     * `split outside block` —— 菜单可点、播放光标在块内，却切不了。
     * 兜底规则：取帧距离最近的成员，把切点钳进它内部。
     */
    #[test]
    fn split_inside_a_merged_run_gap() {
        let timeline = timeline();
        let edits = gap_member_edits();
        let view = compose_param_block_view(&timeline, "root", &edits, 5.0).unwrap();
        assert_eq!(view.blocks.len(), 1);
        assert_eq!(
            (
                view.blocks[0].start_frame,
                view.blocks[0].end_frame_exclusive
            ),
            (0, 200)
        );
        // 多成员 run 不是可逆编组：它照样可以切分。
        assert_eq!(
            view.blocks[0].kind,
            crate::param_blocks::ProjectedBlockKind::User
        );

        // 帧 100 落在缝里。
        let split = split_block(&timeline, &edits, &view.blocks[0].id, 100, 5.0).unwrap();
        let split_view = compose_param_block_view(&timeline, "root", &split, 5.0).unwrap();
        assert_eq!(split_view.blocks.len(), 2);
        assert_eq!(split_view.blocks[0].start_frame, 0);
        assert_eq!(split_view.blocks[1].end_frame_exclusive, 200);

        // 切完的右半仍然可以再切（回归"多成员块切一刀之后就切不动了"）。
        let again = split_block(&timeline, &split, &split_view.blocks[1].id, 150, 5.0).unwrap();
        let again_view = compose_param_block_view(&timeline, "root", &again, 5.0).unwrap();
        assert_eq!(again_view.blocks.len(), 3);
    }

    #[test]
    fn create_block_covers_loop_cycles() {
        let mut timeline: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"root","name":"Root","order":0}],
            "clips":[{
                "id":"clip","track_id":"root","name":"clip","start_sec":0.0,"length_sec":4.0,
                "loop_enabled":true,"loop_period_sec":1.0,
                "takes":[{"id":"take","source_path":"a.wav","source_start_sec":0.0,"source_end_sec":1.0,"loop_enabled":true}]
            }],
            "project_sec":8,"bpm":120
        })).unwrap();
        timeline.clips[0].normalize_takes();
        let created = create_block(
            &timeline,
            "root",
            &ParamBlockEditSet::default(),
            0,
            800,
            Some(60.0),
            5.0,
        )
        .unwrap();
        let block = &created.blocks[0];
        assert!(block.members().len() >= 4);
        assert!(block.members().iter().any(|member| member.cycle == 3));
    }

    #[test]
    fn transition_materializes_base_block() {
        let timeline = timeline();
        let view = base_view(&timeline);
        let next = set_transition(
            &timeline,
            &ParamBlockEditSet::default(),
            &view.blocks[0],
            0.02,
            5.0,
        )
        .unwrap();
        let next_view = compose_param_block_view(&timeline, "root", &next, 5.0).unwrap();
        assert_eq!(next_view.blocks[0].transition_frames, Some(4));
    }
}
