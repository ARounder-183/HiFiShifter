//! 参数块的源域编辑原语。
//!
//! 职责：所有编辑都只改写 \`ParamBlockEditSet\`，绝不写绝对帧。
//! 关系：\`compose\` 消费编辑结果；App/插件只通过本文件改块。
//! 维护约束：新块必须经过 \`place_user_block\`，由它裁剪被覆盖的旧用户块。

use super::projection::{
    normalize_loop_span, project_span, source_at_projected_frame, ClipProjection,
};
use super::types::{
    BaseCover, BlockGeometry, BlockLabel, ParamBlockEditSet, ProjectedBlockKind, ProjectedMember,
    ProjectedParamBlock, RootSpan, TakeSpan, UserParamBlock,
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
fn trim_block_against(existing: &UserParamBlock, target: &UserParamBlock) -> Vec<UserParamBlock> {
    if let BlockGeometry::Group { children } = &target.geometry {
        let mut out = vec![existing.clone()];
        for child in children {
            out = out
                .into_iter()
                .flat_map(|block| trim_block_against(&block, child))
                .collect();
            if out.is_empty() {
                break;
            }
        }
        return out;
    }
    if let BlockGeometry::Group { children } = &existing.geometry {
        let mut out = Vec::new();
        for child in children {
            out.extend(trim_block_against(child, target));
        }
        if out.is_empty() {
            return Vec::new();
        }
        if out.len() == 1 {
            return out;
        }
        let mut group = existing.clone();
        group.geometry = BlockGeometry::Group { children: out };
        return vec![group];
    }
    match (&existing.geometry, &target.geometry) {
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
            if kept.is_empty() {
                return Vec::new();
            }
            let mut next = existing.clone();
            next.geometry = BlockGeometry::Take { members: kept };
            vec![next]
        }
        (BlockGeometry::Root { span }, BlockGeometry::Root { span: cut }) => {
            let mut out = Vec::new();
            for (index, (start, end)) in
                subtract_interval(span.start_sec, span.end_sec, cut.start_sec, cut.end_sec)
                    .into_iter()
                    .enumerate()
            {
                if end <= start + EPS {
                    continue;
                }
                let mut next = existing.clone();
                next.id = root_piece_id(existing, index, start, end);
                next.geometry = BlockGeometry::Root {
                    span: RootSpan {
                        start_sec: start,
                        end_sec: end,
                    },
                };
                out.push(next);
            }
            out
        }
        _ => vec![existing.clone()],
    }
}

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
        blocks.extend(trim_block_against(block, &target));
    }
    blocks.push(target);
    let mut next = edits.clone();
    next.blocks = blocks;
    next.revision = edits.revision.saturating_add(1);
    next.validate()?;
    Ok(next)
}

fn projected_members_to_spans(
    _timeline: &TimelineState,
    block: &ProjectedParamBlock,
    _frame_period_ms: f64,
) -> Result<Vec<TakeSpan>, String> {
    if block.members.is_empty() {
        return Err("Invalid: block has no source".into());
    }
    let mut out = Vec::new();
    for member in &block.members {
        let start = member.visible_source_start_sec;
        let end = member.visible_source_end_sec;
        if end <= start + EPS {
            continue;
        }
        let full = (start - member.source_start_sec).abs() <= 1e-7
            && (end - member.source_end_sec).abs() <= 1e-7;
        out.push(TakeSpan {
            clip_id: member.clip_id.clone(),
            take_id: member.take_id.clone(),
            cycle: member.cycle,
            source_start_sec: start,
            source_end_sec: end,
            // 片段不再拥有完整基块身份；保留 syllable_id 会让 BaseCover.full
            // 误伤同一基块的其他片段。
            syllable_id: if full {
                member.syllable_id.clone()
            } else {
                None
            },
            label: member.label.unwrap_or(BlockLabel::Manual),
        });
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
    if parent.is_group() {
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
        base_cover: parent.base_cover.clone(),
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
    let members = if block.kind == ProjectedBlockKind::Base {
        block
            .base_cover
            .as_ref()
            .map(|cover| cover.spans.clone())
            .unwrap_or_else(|| {
                projected_members_to_spans(timeline, block, frame_period_ms).unwrap_or_default()
            })
    } else {
        projected_members_to_spans(timeline, block, frame_period_ms)?
    };
    if members.is_empty() {
        return Err("Invalid: block has no source".into());
    }
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
        base_cover: block.base_cover.clone(),
    };
    let next = place_user_block(edits, target.clone())?;
    Ok((next, target))
}

fn projected_blocks_overlap(selected: &[ProjectedParamBlock]) -> bool {
    let mut intervals: Vec<(u32, u32)> = selected
        .iter()
        .map(|block| (block.start_frame, block.end_frame_exclusive))
        .collect();
    intervals.sort_unstable();
    intervals.windows(2).any(|pair| pair[1].0 < pair[0].1)
}

fn first_projected_frame(
    timeline: &TimelineState,
    block: &UserParamBlock,
    frame_period_ms: f64,
) -> u32 {
    super::compose::project_single_user_block(timeline, block, frame_period_ms)
        .ok()
        .flatten()
        .map(|projected| projected.start_frame)
        .unwrap_or(u32::MAX)
}

/// 顶层替换：删除指定 id 后原样放入 replacements。组编辑专用。
///
/// 不能走 `place_user_block`：Group 没有可裁剪的成员，旧顶层块必须显式移除，
/// 否则组会和被它包含的原块同时存在。
fn replace_top_level_blocks(
    edits: &ParamBlockEditSet,
    remove_ids: &std::collections::BTreeSet<String>,
    replacements: Vec<UserParamBlock>,
) -> Result<ParamBlockEditSet, String> {
    let mut next = edits.clone();
    next.blocks
        .retain(|block| !remove_ids.contains(block.id.as_str()));
    next.blocks.extend(replacements);
    next.blocks.sort_by(|a, b| {
        a.created_seq
            .cmp(&b.created_seq)
            .then_with(|| a.id.cmp(&b.id))
    });
    next.revision = edits.revision.saturating_add(1);
    next.validate()?;
    Ok(next)
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
    if projected_blocks_overlap(selected) {
        return Err("Invalid: group blocks overlap".into());
    }
    /*
     * 多 run 逻辑块在一次选择里会出现多个 ProjectedParamBlock。先按逻辑 id
     * 收敛成一个块，再物化；否则每个 run 都会各自 materialize 一次，父块
     * 的剩余成员会被拆成兄弟块，组里就会漏进未选中的片段。
     */
    let mut by_logical: std::collections::BTreeMap<String, ProjectedParamBlock> =
        std::collections::BTreeMap::new();
    for block in selected {
        let id = block
            .user_block_id
            .clone()
            .unwrap_or_else(|| block.id.clone());
        match by_logical.get_mut(&id) {
            Some(existing) => {
                existing.start_frame = existing.start_frame.min(block.start_frame);
                existing.end_frame_exclusive =
                    existing.end_frame_exclusive.max(block.end_frame_exclusive);
                existing.members.extend(block.members.iter().cloned());
            }
            None => {
                let mut copy = block.clone();
                copy.id = id.clone();
                copy.user_block_id = Some(id.clone());
                by_logical.insert(id, copy);
            }
        }
    }
    let mut ordered: Vec<ProjectedParamBlock> = by_logical.into_values().collect();
    ordered.sort_by_key(|block| block.start_frame);
    if ordered.len() < 2 {
        return Err("Invalid: group needs two blocks".into());
    }

    let mut working = edits.clone();
    let mut ensured_ids = std::collections::BTreeSet::new();
    let mut children: Vec<UserParamBlock> = Vec::new();
    for block in &ordered {
        let (next, user) = ensure_editable_user_block(timeline, &working, block, frame_period_ms)?;
        working = next;
        ensured_ids.insert(user.id.clone());
        if user.is_group() {
            for child in user.flatten_children() {
                if !children.iter().any(|existing| existing.id == child.id) {
                    children.push(child.clone());
                }
            }
        } else if !children.iter().any(|existing| existing.id == user.id) {
            children.push(user);
        }
    }
    if children.len() < 2 {
        return Err("Invalid: blocks are already grouped".into());
    }
    children.sort_by_key(|child| first_projected_frame(timeline, child, frame_period_ms));

    let has_root = children
        .iter()
        .any(|block| matches!(block.geometry, BlockGeometry::Root { .. }));
    let has_take = children.iter().any(UserParamBlock::is_take);
    if has_root && has_take {
        return Err("Invalid: cannot group root and take blocks together".into());
    }
    if children.iter().any(UserParamBlock::is_group) {
        return Err("Invalid: nested group".into());
    }

    let values: Vec<Option<f64>> = children.iter().map(|block| block.value).collect();
    let value = values
        .first()
        .copied()
        .flatten()
        .filter(|first| values.iter().all(|value| *value == Some(*first)));
    let manual = children.iter().any(|block| block.manual);
    let transition_out_sec = children.last().and_then(|block| block.transition_out_sec);
    let target = UserParamBlock {
        id: new_id("ublock", &format!("group:{}", ordered[0].id)),
        geometry: BlockGeometry::Group { children },
        deleted: false,
        manual,
        value,
        transition_out_sec,
        created_seq: working.next_seq(),
        base_cover: None,
    };
    working = replace_top_level_blocks(&working, &ensured_ids, vec![target])?;
    Ok(working)
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
    let children = block
        .children()
        .ok_or_else(|| "Invalid: block is not grouped".to_string())?
        .to_vec();
    if children.len() < 2 {
        return Err("Invalid: grouped block has no members".into());
    }
    let mut remove = std::collections::BTreeSet::new();
    remove.insert(block_id.to_owned());
    replace_top_level_blocks(edits, &remove, children)
}

fn span_from_projected_member(member: &ProjectedMember) -> TakeSpan {
    TakeSpan {
        clip_id: member.clip_id.clone(),
        take_id: member.take_id.clone(),
        cycle: member.cycle,
        source_start_sec: member.source_start_sec,
        source_end_sec: member.source_end_sec,
        syllable_id: member.syllable_id.clone(),
        label: member.label.unwrap_or(BlockLabel::Manual),
    }
}

fn member_projection<'a>(
    timeline: &'a TimelineState,
    member: &ProjectedMember,
) -> Result<(&'a crate::state::Clip, ClipProjection), String> {
    let clip = timeline
        .clips
        .iter()
        .find(|clip| clip.id == member.clip_id)
        .ok_or_else(|| "Invalid: member clip missing".to_string())?;
    let take = clip
        .take(&member.take_id)
        .ok_or_else(|| "Invalid: member take missing".to_string())?;
    Ok((clip, ClipProjection::from_clip_take(clip, take)))
}

fn source_at_member_frame(
    timeline: &TimelineState,
    member: &ProjectedMember,
    frame: u32,
    frame_period_ms: f64,
) -> Result<f64, String> {
    let (_, projection) = member_projection(timeline, member)?;
    source_at_projected_frame(&projection, frame, member.cycle, frame_period_ms)
        .ok_or_else(|| "Invalid: source at frame".to_string())
}

fn member_is_reversed(timeline: &TimelineState, member: &ProjectedMember) -> bool {
    if (member.frame_source_start_sec - member.frame_source_end_sec).abs() > 1e-12 {
        return member.frame_source_start_sec > member.frame_source_end_sec;
    }
    member_projection(timeline, member)
        .map(|(_, projection)| projection.reversed)
        .unwrap_or(false)
}

fn min_source_width(
    timeline: &TimelineState,
    member: &ProjectedMember,
    frame_period_ms: f64,
) -> f64 {
    let rate = member_projection(timeline, member)
        .map(|(_, projection)| projection.effective_rate)
        .unwrap_or(1.0)
        .max(1e-9);
    frame_period_ms.max(0.1) / 1000.0 * rate
}

fn order_spans_by_projection(
    timeline: &TimelineState,
    spans: Vec<TakeSpan>,
    frame_period_ms: f64,
) -> Vec<TakeSpan> {
    let mut ranked: Vec<(u32, TakeSpan)> = spans
        .into_iter()
        .map(|span| {
            let rank = timeline
                .clips
                .iter()
                .find(|clip| clip.id == span.clip_id)
                .and_then(|clip| clip.take(&span.take_id).map(|take| (clip, take)))
                .map(|(clip, take)| {
                    let projection = ClipProjection::from_clip_take(clip, take);
                    project_span(&projection, &span, frame_period_ms)
                        .into_iter()
                        .find(|interval| interval.cycle == span.cycle)
                        .map(|interval| interval.start_frame)
                        .unwrap_or(u32::MAX)
                })
                .unwrap_or(u32::MAX);
            (rank, span)
        })
        .collect();
    ranked.sort_by_key(|(rank, _)| *rank);
    ranked.into_iter().map(|(_, span)| span).collect()
}

fn normalize_spans_for_projection(timeline: &TimelineState, span: &TakeSpan) -> Vec<TakeSpan> {
    let Some(clip) = timeline.clips.iter().find(|clip| clip.id == span.clip_id) else {
        return vec![span.clone()];
    };
    let Some(take) = clip.take(&span.take_id) else {
        return vec![span.clone()];
    };
    normalize_loop_span(&ClipProjection::from_clip_take(clip, take), span)
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

/// 按分割帧把 baseCover 拆成左右两份；跨切点的 span 两侧都降为 full:false。
fn split_base_cover(
    timeline: &TimelineState,
    cover: &Option<BaseCover>,
    frame: u32,
    frame_period_ms: f64,
) -> (Option<BaseCover>, Option<BaseCover>) {
    let Some(cover) = cover else {
        return (None, None);
    };
    let mut left_spans: Vec<TakeSpan> = Vec::new();
    let mut right_spans: Vec<TakeSpan> = Vec::new();
    let mut left_full = cover.full;
    let mut right_full = cover.full;
    for span in &cover.spans {
        let Some(clip) = timeline.clips.iter().find(|clip| clip.id == span.clip_id) else {
            left_spans.push(span.clone());
            right_spans.push(span.clone());
            left_full = false;
            right_full = false;
            continue;
        };
        let Some(take) = clip.take(&span.take_id) else {
            left_spans.push(span.clone());
            right_spans.push(span.clone());
            left_full = false;
            right_full = false;
            continue;
        };
        let projection = ClipProjection::from_clip_take(clip, take);
        let intervals = project_span(&projection, span, frame_period_ms);
        let mut crossing = None;
        let mut side: Option<&str> = None;
        for interval in intervals {
            if interval.start_frame < frame && frame < interval.end_frame_exclusive {
                crossing = Some(interval);
                break;
            }
            if interval.end_frame_exclusive <= frame {
                side = Some(if side == Some("right") {
                    "both"
                } else {
                    "left"
                });
            } else if interval.start_frame >= frame {
                side = Some(if side == Some("left") {
                    "both"
                } else {
                    "right"
                });
            }
        }
        if let Some(interval) = crossing {
            let a = source_at_projected_frame(
                &projection,
                interval.start_frame,
                interval.cycle,
                frame_period_ms,
            );
            let b = source_at_projected_frame(
                &projection,
                interval.end_frame_exclusive,
                interval.cycle,
                frame_period_ms,
            );
            let cut =
                source_at_projected_frame(&projection, frame, interval.cycle, frame_period_ms);
            if let (Some(a), Some(b), Some(cut)) = (a, b, cut) {
                let lo = a.min(b);
                let hi = a.max(b);
                let cut = cut.clamp(lo, hi);
                let mut left = span.clone();
                left.source_start_sec = lo;
                left.source_end_sec = cut;
                let mut right = span.clone();
                right.source_start_sec = cut;
                right.source_end_sec = hi;
                left_spans.push(left);
                right_spans.push(right);
                left_full = false;
                right_full = false;
                continue;
            }
        }
        match side {
            Some("left") => left_spans.push(span.clone()),
            Some("right") => right_spans.push(span.clone()),
            _ => {
                left_spans.push(span.clone());
                right_spans.push(span.clone());
                left_full = false;
                right_full = false;
            }
        }
    }
    (
        (!left_spans.is_empty()).then_some(BaseCover {
            spans: left_spans,
            full: left_full,
        }),
        (!right_spans.is_empty()).then_some(BaseCover {
            spans: right_spans,
            full: right_full,
        }),
    )
}

fn split_take_parts(
    timeline: &TimelineState,
    working: &ParamBlockEditSet,
    block: &UserParamBlock,
    projected: &ProjectedParamBlock,
    frame: u32,
    frame_period_ms: f64,
) -> Result<(UserParamBlock, UserParamBlock), String> {
    let members = &projected.members;
    if members.is_empty() {
        return Err("Invalid: block has no members".into());
    }
    let inside = members
        .iter()
        .position(|member| member.start_frame < frame && frame < member.end_frame_exclusive);
    let (left_members, right_members) = if let Some(index) = inside {
        let hit = &members[index];
        let left_frames = frame.saturating_sub(hit.start_frame);
        let right_frames = hit.end_frame_exclusive.saturating_sub(frame);
        if left_frames < 1 || right_frames < 1 || hit.end_frame_exclusive - hit.start_frame <= 1 {
            return Err("Invalid: split too close to edge".into());
        }
        let source = source_at_member_frame(timeline, hit, frame, frame_period_ms)?;
        let reversed = member_is_reversed(timeline, hit);
        let mut left_part = span_from_projected_member(hit);
        let mut right_part = span_from_projected_member(hit);
        if reversed {
            left_part.source_start_sec = source;
            left_part.source_end_sec = hit.source_end_sec;
            right_part.source_start_sec = hit.source_start_sec;
            right_part.source_end_sec = source;
        } else {
            left_part.source_start_sec = hit.source_start_sec;
            left_part.source_end_sec = source;
            right_part.source_start_sec = source;
            right_part.source_end_sec = hit.source_end_sec;
        }
        let mut left: Vec<TakeSpan> = members[..index]
            .iter()
            .map(span_from_projected_member)
            .collect();
        left.push(left_part);
        let mut right = vec![right_part];
        right.extend(members[index + 1..].iter().map(span_from_projected_member));
        (left, right)
    } else {
        let left: Vec<TakeSpan> = members
            .iter()
            .filter(|member| member.end_frame_exclusive <= frame)
            .map(span_from_projected_member)
            .collect();
        let right: Vec<TakeSpan> = members
            .iter()
            .filter(|member| member.start_frame >= frame)
            .map(span_from_projected_member)
            .collect();
        if left.is_empty() || right.is_empty() {
            return Err("Invalid: split at block edge".into());
        }
        (left, right)
    };
    let (left_cover, right_cover) =
        split_base_cover(timeline, &block.base_cover, frame, frame_period_ms);
    let left = UserParamBlock {
        id: new_id("ublock", &format!("split:{}:{frame}:left", block.id)),
        geometry: BlockGeometry::Take {
            members: left_members,
        },
        deleted: false,
        manual: block.manual,
        value: block.value,
        transition_out_sec: Some(0.0),
        created_seq: next_seq(working),
        base_cover: left_cover,
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
        created_seq: next_seq(working).saturating_add(1),
        base_cover: right_cover,
    };
    Ok((left, right))
}

fn split_root_parts(
    working: &ParamBlockEditSet,
    block: &UserParamBlock,
    span: &RootSpan,
    frame: u32,
    frame_period_ms: f64,
) -> Result<(UserParamBlock, UserParamBlock), String> {
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
        created_seq: next_seq(working),
        base_cover: None,
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
        created_seq: next_seq(working).saturating_add(1),
        base_cover: None,
    };
    Ok((left, right))
}

fn split_leaf_parts(
    timeline: &TimelineState,
    working: &ParamBlockEditSet,
    block: &UserParamBlock,
    projected: &ProjectedParamBlock,
    frame: u32,
    frame_period_ms: f64,
) -> Result<(UserParamBlock, UserParamBlock), String> {
    match &block.geometry {
        BlockGeometry::Root { span } => {
            split_root_parts(working, block, span, frame, frame_period_ms)
        }
        BlockGeometry::Take { .. } => {
            split_take_parts(timeline, working, block, projected, frame, frame_period_ms)
        }
        BlockGeometry::Group { .. } => Err("Invalid: nested group".into()),
    }
}

fn split_group_block(
    timeline: &TimelineState,
    edits: &ParamBlockEditSet,
    group: &UserParamBlock,
    frame: u32,
    frame_period_ms: f64,
) -> Result<ParamBlockEditSet, String> {
    let mut ordered = group
        .flatten_children()
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    ordered.sort_by_key(|child| first_projected_frame(timeline, child, frame_period_ms));
    let working = edits.clone();
    let mut left: Vec<UserParamBlock> = Vec::new();
    let mut right: Vec<UserParamBlock> = Vec::new();
    for child in ordered {
        let Some(projected) =
            super::compose::project_logical_user_block(timeline, &child, frame_period_ms)?
        else {
            continue;
        };
        if projected.end_frame_exclusive <= frame {
            left.push(child);
        } else if projected.start_frame >= frame {
            right.push(child);
        } else {
            let (left_part, right_part) = split_leaf_parts(
                timeline,
                &working,
                &child,
                &projected,
                frame,
                frame_period_ms,
            )?;
            left.push(left_part);
            right.push(right_part);
        }
    }
    if left.is_empty() || right.is_empty() {
        return Err("Invalid: split at block edge".into());
    }
    let mut remove = std::collections::BTreeSet::new();
    remove.insert(group.id.clone());
    let mut replacements = Vec::new();
    let first_seq = working.next_seq();
    replacements.push(group_side(
        timeline,
        group,
        left,
        &format!("split:{}:{frame}:left", group.id),
        frame_period_ms,
        first_seq,
    ));
    replacements.push(group_side(
        timeline,
        group,
        right,
        &format!("split:{}:{frame}:right", group.id),
        frame_period_ms,
        first_seq.saturating_add(1),
    ));
    replace_top_level_blocks(&working, &remove, replacements)
}

fn group_side(
    timeline: &TimelineState,
    original: &UserParamBlock,
    mut children: Vec<UserParamBlock>,
    identity: &str,
    frame_period_ms: f64,
    created_seq: u64,
) -> UserParamBlock {
    children.sort_by_key(|child| first_projected_frame(timeline, child, frame_period_ms));
    if children.len() == 1 {
        let mut child = children.remove(0);
        if original.value.is_some() {
            child.value = original.value;
        }
        return child;
    }
    UserParamBlock {
        id: new_id("ublock", identity),
        geometry: BlockGeometry::Group { children },
        deleted: false,
        manual: original.manual,
        value: original.value,
        transition_out_sec: original.transition_out_sec,
        created_seq,
        base_cover: None,
    }
}

fn resize_single_leaf(
    timeline: &TimelineState,
    edits: &ParamBlockEditSet,
    child: &UserParamBlock,
    edge: &str,
    target_frame: u32,
    frame_period_ms: f64,
) -> Result<Vec<UserParamBlock>, String> {
    let temp = ParamBlockEditSet {
        schema_version: 2,
        revision: edits.revision,
        blocks: vec![child.clone()],
    };
    let Some(projected) =
        super::compose::project_single_user_block(timeline, child, frame_period_ms)?
    else {
        return Ok(Vec::new());
    };
    let resized = resize_block(
        timeline,
        &temp,
        &projected.id,
        edge,
        target_frame,
        frame_period_ms,
    )?;
    Ok(resized.blocks)
}

fn resize_group_block(
    timeline: &TimelineState,
    edits: &ParamBlockEditSet,
    group: &UserParamBlock,
    edge: &str,
    target_frame: u32,
    frame_period_ms: f64,
) -> Result<ParamBlockEditSet, String> {
    let mut children = group
        .flatten_children()
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    children.sort_by_key(|child| first_projected_frame(timeline, child, frame_period_ms));
    if children.is_empty() {
        return Err("Invalid: group has no children".into());
    }
    let Some(first_projected) =
        super::compose::project_single_user_block(timeline, &children[0], frame_period_ms)?
    else {
        return Err("Invalid: group child not projected".into());
    };
    let Some(last_projected) = super::compose::project_single_user_block(
        timeline,
        children.last().unwrap(),
        frame_period_ms,
    )?
    else {
        return Err("Invalid: group child not projected".into());
    };
    if edge == "right" && target_frame <= first_projected.start_frame.saturating_add(1) {
        let group_projected =
            super::compose::project_single_user_block(timeline, group, frame_period_ms)?
                .ok_or_else(|| "Invalid: block not found".to_string())?;
        return Ok(delete_blocks(timeline, edits, &[group_projected], frame_period_ms)?.0);
    }
    if edge == "left" && target_frame >= last_projected.end_frame_exclusive.saturating_sub(1) {
        let group_projected =
            super::compose::project_single_user_block(timeline, group, frame_period_ms)?
                .ok_or_else(|| "Invalid: block not found".to_string())?;
        return Ok(delete_blocks(timeline, edits, &[group_projected], frame_period_ms)?.0);
    }

    let mut kept: Vec<UserParamBlock> = Vec::new();
    if edge == "right" {
        for child in &children {
            let Some(projected) =
                super::compose::project_single_user_block(timeline, child, frame_period_ms)?
            else {
                continue;
            };
            if projected.end_frame_exclusive <= target_frame {
                kept.push(child.clone());
            } else if projected.start_frame >= target_frame {
                break;
            } else {
                let (left_part, _) = split_leaf_parts(
                    timeline,
                    edits,
                    child,
                    &projected,
                    target_frame,
                    frame_period_ms,
                )?;
                kept.push(left_part);
                break;
            }
        }
        if kept.is_empty() {
            let group_projected =
                super::compose::project_single_user_block(timeline, group, frame_period_ms)?
                    .ok_or_else(|| "Invalid: block not found".to_string())?;
            return Ok(delete_blocks(timeline, edits, &[group_projected], frame_period_ms)?.0);
        }
        if target_frame >= last_projected.end_frame_exclusive
            && kept.last().is_some_and(|child| {
                super::compose::project_single_user_block(timeline, child, frame_period_ms)
                    .ok()
                    .flatten()
                    .is_some_and(|projected| projected.end_frame_exclusive < target_frame)
            })
        {
            let last = kept.pop().unwrap();
            let mut resized = resize_single_leaf(
                timeline,
                edits,
                &last,
                "right",
                target_frame,
                frame_period_ms,
            )?;
            kept.append(&mut resized);
        }
    } else {
        for child in children.iter().rev() {
            let Some(projected) =
                super::compose::project_single_user_block(timeline, child, frame_period_ms)?
            else {
                continue;
            };
            if projected.start_frame >= target_frame {
                kept.push(child.clone());
            } else if projected.end_frame_exclusive <= target_frame {
                break;
            } else {
                let (_, right_part) = split_leaf_parts(
                    timeline,
                    edits,
                    child,
                    &projected,
                    target_frame,
                    frame_period_ms,
                )?;
                kept.push(right_part);
                break;
            }
        }
        kept.reverse();
        if kept.is_empty() {
            let group_projected =
                super::compose::project_single_user_block(timeline, group, frame_period_ms)?
                    .ok_or_else(|| "Invalid: block not found".to_string())?;
            return Ok(delete_blocks(timeline, edits, &[group_projected], frame_period_ms)?.0);
        }
        if target_frame <= first_projected.start_frame
            && kept.first().is_some_and(|child| {
                super::compose::project_single_user_block(timeline, child, frame_period_ms)
                    .ok()
                    .flatten()
                    .is_some_and(|projected| projected.start_frame > target_frame)
            })
        {
            let first = kept.remove(0);
            let mut resized = resize_single_leaf(
                timeline,
                edits,
                &first,
                "left",
                target_frame,
                frame_period_ms,
            )?;
            resized.append(&mut kept);
            kept = resized;
        }
    }

    let mut remove = std::collections::BTreeSet::new();
    remove.insert(group.id.clone());
    if kept.is_empty() {
        return replace_top_level_blocks(edits, &remove, Vec::new());
    }
    let kept_ids: std::collections::BTreeSet<String> =
        kept.iter().map(|child| child.id.clone()).collect();
    let mut tombstones: Vec<UserParamBlock> = Vec::new();
    for child in &children {
        if kept_ids.contains(&child.id) {
            continue;
        }
        if child.manual && child.base_cover.is_none() {
            continue;
        }
        if !child.is_take() {
            continue;
        }
        for member in child.members() {
            if !tombstones
                .iter()
                .any(|tombstone| tombstone_matches(tombstone, member))
            {
                tombstones.push(make_tombstone(
                    child,
                    member,
                    edits.next_seq().saturating_add(tombstones.len() as u64),
                ));
            }
        }
    }
    let side = group_side(
        timeline,
        group,
        kept,
        &format!("resize:{}", group.id),
        frame_period_ms,
        edits.next_seq(),
    );
    let mut replacements = vec![side];
    replacements.extend(tombstones);
    replace_top_level_blocks(edits, &remove, replacements)
}

pub fn split_block(
    timeline: &TimelineState,
    edits: &ParamBlockEditSet,
    block_id: &str,
    frame: u32,
    frame_period_ms: f64,
) -> Result<ParamBlockEditSet, String> {
    let projected = find_projected_block(timeline, edits, block_id, frame_period_ms)?;
    if frame <= projected.start_frame || frame >= projected.end_frame_exclusive {
        return Err("Invalid: split outside block".into());
    }
    let (working, block) =
        ensure_editable_user_block(timeline, edits, &projected, frame_period_ms)?;
    if let BlockGeometry::Group { .. } = &block.geometry {
        return split_group_block(timeline, &working, &block, frame, frame_period_ms);
    }
    let projected = super::compose::project_single_user_block(timeline, &block, frame_period_ms)?
        .ok_or_else(|| "Invalid: block not found".to_string())?;
    let (left, right) = split_leaf_parts(
        timeline,
        &working,
        &block,
        &projected,
        frame,
        frame_period_ms,
    )?;
    let next = place_user_block(&working, left)?;
    place_user_block(&next, right)
}

fn edge_clip_bounds(
    clip: &crate::state::Clip,
    projection: &ClipProjection,
    cycle: u32,
    frame_period_ms: f64,
) -> (u32, u32) {
    let clip_start = projection.to_frame(clip.start_sec, frame_period_ms);
    let clip_end = projection.to_frame(clip.start_sec + clip.length_sec, frame_period_ms);
    if projection.loop_enabled {
        if let Some(cycle_sec) = projection.loop_cycle_sec {
            let cycle_timeline = cycle_sec / projection.effective_rate.max(1e-9);
            let cycle_start = projection.to_frame(
                projection.project_start_sec + cycle as f64 * cycle_timeline,
                frame_period_ms,
            );
            let cycle_end = projection.to_frame(
                projection.project_start_sec + (cycle as f64 + 1.0) * cycle_timeline,
                frame_period_ms,
            );
            return (clip_start.max(cycle_start), clip_end.min(cycle_end));
        }
    }
    (clip_start, clip_end)
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
    if let BlockGeometry::Group { .. } = &block.geometry {
        return resize_group_block(
            timeline,
            &working,
            &block,
            edge,
            target_frame,
            frame_period_ms,
        );
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
            return place_user_block(&working, next);
        }
        BlockGeometry::Take { .. } => {}
        // 上面已把 Group 分流到 resize_group_block；这里再出现 Group 只可能是
        // 新增枚举变体 / 前期分流被改动。返回明确错误而不是 panic（渲染线程）。
        BlockGeometry::Group { .. } => {
            return Err("Invalid: group block must use the group resize path".into());
        }
    }
    let members = projected.members.clone();
    if members.is_empty() {
        return Err("Invalid: block has no members".into());
    }
    let edge_member = if edge == "left" {
        members.first().unwrap()
    } else {
        members.last().unwrap()
    };
    let (clip, projection) = member_projection(timeline, edge_member)?;
    let (min_frame, max_frame) =
        edge_clip_bounds(clip, &projection, edge_member.cycle, frame_period_ms);
    let mut target = target_frame;
    let mut next_members: Vec<TakeSpan> = members.iter().map(span_from_projected_member).collect();
    if edge == "right" {
        if target <= projected.start_frame + 1 {
            return Err("Invalid: resized block too short".into());
        }
        if target > max_frame {
            target = max_frame;
        }
        if target >= projected.end_frame_exclusive {
            let last = members.last().unwrap();
            let source = source_at_member_frame(timeline, last, target, frame_period_ms)?;
            let reversed = member_is_reversed(timeline, last);
            let width = min_source_width(timeline, last, frame_period_ms);
            let span = next_members.last_mut().unwrap();
            if reversed {
                if source >= span.source_end_sec - width {
                    return Err("Invalid: resized block too short".into());
                }
                span.source_start_sec = source;
            } else {
                if source <= span.source_start_sec + width {
                    return Err("Invalid: resized block too short".into());
                }
                span.source_end_sec = source;
            }
        } else if let Some(index) = members
            .iter()
            .position(|member| member.start_frame < target && target < member.end_frame_exclusive)
        {
            let hit = &members[index];
            let source = source_at_member_frame(timeline, hit, target, frame_period_ms)?;
            let reversed = member_is_reversed(timeline, hit);
            let width = min_source_width(timeline, hit, frame_period_ms);
            let span = &mut next_members[index];
            if reversed {
                if source <= span.source_start_sec + width {
                    next_members.truncate(index);
                } else {
                    span.source_start_sec = source;
                    next_members.truncate(index + 1);
                }
            } else if source >= span.source_end_sec - width {
                next_members.truncate(index);
            } else {
                span.source_end_sec = source;
                next_members.truncate(index + 1);
            }
        } else {
            let kept: Vec<usize> = members
                .iter()
                .enumerate()
                .filter(|(_, member)| member.end_frame_exclusive <= target)
                .map(|(index, _)| index)
                .collect();
            if kept.is_empty() {
                return Err("Invalid: resized block too short".into());
            }
            next_members = kept
                .iter()
                .map(|index| span_from_projected_member(&members[*index]))
                .collect();
            let last_index = *kept.last().unwrap();
            let last = &members[last_index];
            if last.end_frame_exclusive < target {
                let source = source_at_member_frame(timeline, last, target, frame_period_ms)?;
                let reversed = member_is_reversed(timeline, last);
                let width = min_source_width(timeline, last, frame_period_ms);
                let span = next_members.last_mut().unwrap();
                let can_extend = if reversed {
                    source < span.source_start_sec - width
                } else {
                    source > span.source_end_sec + width
                };
                if can_extend {
                    if reversed {
                        span.source_start_sec = source;
                    } else {
                        span.source_end_sec = source;
                    }
                }
            }
        }
    } else {
        if target >= projected.end_frame_exclusive.saturating_sub(1) {
            return Err("Invalid: resized block too short".into());
        }
        if target < min_frame {
            target = min_frame;
        }
        if target <= projected.start_frame {
            let first = members.first().unwrap();
            let source = source_at_member_frame(timeline, first, target, frame_period_ms)?;
            let reversed = member_is_reversed(timeline, first);
            let width = min_source_width(timeline, first, frame_period_ms);
            let span = next_members.first_mut().unwrap();
            if reversed {
                if source <= span.source_start_sec + width {
                    return Err("Invalid: resized block too short".into());
                }
                span.source_end_sec = source;
            } else {
                if source >= span.source_end_sec - width {
                    return Err("Invalid: resized block too short".into());
                }
                span.source_start_sec = source;
            }
        } else if let Some(index) = members
            .iter()
            .position(|member| member.start_frame < target && target < member.end_frame_exclusive)
        {
            let hit = &members[index];
            let source = source_at_member_frame(timeline, hit, target, frame_period_ms)?;
            let reversed = member_is_reversed(timeline, hit);
            let width = min_source_width(timeline, hit, frame_period_ms);
            let span = &mut next_members[index];
            if reversed {
                if source >= span.source_end_sec - width {
                    next_members.drain(..=index);
                } else {
                    span.source_end_sec = source;
                    next_members.drain(..index);
                }
            } else if source <= span.source_start_sec + width {
                next_members.drain(..=index);
            } else {
                span.source_start_sec = source;
                next_members.drain(..index);
            }
        } else {
            let kept: Vec<usize> = members
                .iter()
                .enumerate()
                .filter(|(_, member)| member.start_frame >= target)
                .map(|(index, _)| index)
                .collect();
            if kept.is_empty() {
                return Err("Invalid: resized block too short".into());
            }
            next_members = kept
                .iter()
                .map(|index| span_from_projected_member(&members[*index]))
                .collect();
            let first_index = kept[0];
            let first = &members[first_index];
            if first.start_frame > target {
                let source = source_at_member_frame(timeline, first, target, frame_period_ms)?;
                let reversed = member_is_reversed(timeline, first);
                let width = min_source_width(timeline, first, frame_period_ms);
                let span = next_members.first_mut().unwrap();
                let can_extend = if reversed {
                    source > span.source_end_sec + width
                } else {
                    source < span.source_start_sec - width
                };
                if can_extend {
                    if reversed {
                        span.source_end_sec = source;
                    } else {
                        span.source_start_sec = source;
                    }
                }
            }
        }
    }
    let expanded: Vec<TakeSpan> = next_members
        .iter()
        .flat_map(|span| normalize_spans_for_projection(timeline, span))
        .collect();
    let ordered = order_spans_by_projection(timeline, expanded, frame_period_ms);
    if ordered.is_empty() {
        return Err("Invalid: resized block too short".into());
    }
    next.geometry = BlockGeometry::Take { members: ordered };
    place_user_block(&working, next)
}

fn augment_base_cover_for_delete(
    timeline: &TimelineState,
    edits: &ParamBlockEditSet,
    user: &UserParamBlock,
    frame_period_ms: f64,
) -> Option<super::types::BaseCover> {
    let mut owned = user.base_cover.clone()?;
    let roots: std::collections::BTreeSet<String> = timeline
        .tracks
        .iter()
        .map(|track| track.id.clone())
        .collect();
    for root in roots {
        let Ok(view) =
            super::compose::compose_param_block_view(timeline, &root, edits, frame_period_ms)
        else {
            continue;
        };
        let Some(target) = view.blocks.iter().find(|block| {
            block.id == user.id || block.user_block_id.as_deref() == Some(user.id.as_str())
        }) else {
            continue;
        };
        /*
         * 目标块自己会遮住它覆盖的基块；要找出这些基块，必须把目标块从编辑集里
         * 临时移除后再合成一次，否则只能看到目标块之外露出的基块余段。
         */
        let mut without_target = edits.clone();
        without_target.blocks.retain(|block| block.id != user.id);
        let Ok(base_view) = super::compose::compose_param_block_view(
            timeline,
            &root,
            &without_target,
            frame_period_ms,
        ) else {
            continue;
        };
        for base in base_view.blocks.iter().filter(|block| {
            block.kind == ProjectedBlockKind::Base
                && block.start_frame < target.end_frame_exclusive
                && block.end_frame_exclusive > target.start_frame
        }) {
            let Some(cover) = base.base_cover.as_ref() else {
                continue;
            };
            for span in &cover.spans {
                let already_contained = owned.spans.iter().any(|existing| {
                    existing.clip_id == span.clip_id
                        && existing.take_id == span.take_id
                        && existing.cycle == span.cycle
                        && existing.source_start_sec <= span.source_start_sec + 1e-7
                        && existing.source_end_sec >= span.source_end_sec - 1e-7
                });
                if !already_contained {
                    owned.spans.push(span.clone());
                }
            }
        }
    }
    Some(owned)
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
        base_cover: user.base_cover.clone(),
    }
}

#[derive(Clone, Debug)]
pub struct DeletedParamBlockSnapshot {
    pub block: UserParamBlock,
    /// 组删除时为 base 派生 child 生成的墓碑；恢复时按这些 id 精确移除。
    pub child_tombstones: Vec<UserParamBlock>,
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
        let owned = augment_base_cover_for_delete(timeline, &working, &user, frame_period_ms);
        working.blocks.retain(|candidate| candidate.id != user.id);
        let mut snapshot_block = user.clone();
        if owned.is_some() {
            snapshot_block.base_cover = owned;
        }
        let mut child_tombstones = Vec::new();
        for child in user.flatten_children() {
            if child.manual && child.base_cover.is_none() {
                continue;
            }
            if child.base_cover.as_ref().is_some_and(|cover| !cover.full) {
                continue;
            }
            if child.is_group() {
                continue;
            }
            for member in child.members() {
                if !child_tombstones
                    .iter()
                    .any(|tombstone: &UserParamBlock| tombstone_matches(tombstone, member))
                {
                    child_tombstones.push(make_tombstone(
                        child,
                        member,
                        working
                            .next_seq()
                            .saturating_add(child_tombstones.len() as u64),
                    ));
                }
            }
        }
        snapshots.push(DeletedParamBlockSnapshot {
            block: snapshot_block,
            child_tombstones,
        });
    }
    if snapshots.is_empty() {
        return Err("Invalid: block not found".into());
    }
    let mut tombstones = Vec::new();
    for snapshot in &snapshots {
        // 纯手工块删除后必须让基块回来：它没有 base_cover，也不需要墓碑。
        if snapshot.block.manual && snapshot.block.base_cover.is_none() {
            continue;
        }
        if snapshot
            .block
            .base_cover
            .as_ref()
            .is_some_and(|cover| !cover.full)
        {
            continue;
        }
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
    for snapshot in &snapshots {
        for tombstone in &snapshot.child_tombstones {
            if !working
                .blocks
                .iter()
                .any(|candidate| candidate.id == tombstone.id)
            {
                working = place_user_block(&working, tombstone.clone())?;
            }
        }
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
        let child_ids: std::collections::BTreeSet<&str> = snapshot
            .child_tombstones
            .iter()
            .map(|tombstone| tombstone.id.as_str())
            .collect();
        next.blocks.retain(|candidate| {
            !child_ids.contains(candidate.id.as_str())
                && !snapshot
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
        let cycle_timeline = if projection.loop_enabled {
            projection.loop_cycle_sec.unwrap_or(0.0) / rate
        } else {
            clip.length_sec.max(0.0)
        };
        let cycle_start = projection.to_frame(
            clip.start_sec + cycle as f64 * cycle_timeline,
            frame_period_ms,
        );
        let cycle_end = projection.to_frame(
            clip.start_sec + (cycle as f64 + 1.0) * cycle_timeline,
            frame_period_ms,
        );
        let a = start_frame
            .max(cycle_start)
            .max(projection.to_frame(clip.start_sec, frame_period_ms));
        let b = end_frame
            .min(cycle_end)
            .min(projection.to_frame(clip.start_sec + clip.length_sec, frame_period_ms));
        if b <= a {
            continue;
        }
        /*
         * 覆盖整个周期时，源域是 [anchor, media] + [0, anchor] 两段；只算
         * source_at(a/b) 会得到 [anchor, media]，漏掉回绕后的尾部。
         */
        if projection.loop_enabled && !projection.reversed && a <= cycle_start && b >= cycle_end {
            let anchor = projection.source_window_start_sec;
            let media = projection
                .media_duration_sec
                .unwrap_or(anchor + cycle_timeline * rate);
            if media > anchor + EPS {
                out.push(TakeSpan {
                    clip_id: clip.id.clone(),
                    take_id: take.id.clone(),
                    cycle,
                    source_start_sec: anchor,
                    source_end_sec: media,
                    syllable_id: None,
                    label: BlockLabel::Manual,
                });
            }
            if anchor > EPS {
                out.push(TakeSpan {
                    clip_id: clip.id.clone(),
                    take_id: take.id.clone(),
                    cycle,
                    source_start_sec: 0.0,
                    source_end_sec: anchor,
                    syllable_id: None,
                    label: BlockLabel::Manual,
                });
            }
            continue;
        }
        let Some(source_a) = source_at_projected_frame(&projection, a, cycle, frame_period_ms)
        else {
            continue;
        };
        let Some(source_b) = source_at_projected_frame(&projection, b, cycle, frame_period_ms)
        else {
            continue;
        };
        /*
         * 正放时 source_a > source_b 表示区间跨过媒体末尾回绕；必须拆成
         * [source_a, media] + [0, source_b]，否则会被当成一条从 0 到 media 的巨段。
         */
        if projection.loop_enabled && !projection.reversed && source_a > source_b + EPS {
            let media = projection
                .media_duration_sec
                .unwrap_or(projection.source_window_start_sec + cycle_timeline * rate);
            if media > source_a + EPS {
                out.push(TakeSpan {
                    clip_id: clip.id.clone(),
                    take_id: take.id.clone(),
                    cycle,
                    source_start_sec: source_a,
                    source_end_sec: media,
                    syllable_id: None,
                    label: BlockLabel::Manual,
                });
            }
            if source_b > EPS {
                out.push(TakeSpan {
                    clip_id: clip.id.clone(),
                    take_id: take.id.clone(),
                    cycle,
                    source_start_sec: 0.0,
                    source_end_sec: source_b,
                    syllable_id: None,
                    label: BlockLabel::Manual,
                });
            }
            continue;
        }
        let mut lo = source_a.min(source_b);
        let mut hi = source_a.max(source_b);
        if let Some(media) = projection.media_duration_sec {
            lo = lo.max(0.0);
            hi = hi.min(media);
        }
        if hi <= lo + EPS {
            continue;
        }
        let span = TakeSpan {
            clip_id: clip.id.clone(),
            take_id: take.id.clone(),
            cycle,
            source_start_sec: lo,
            source_end_sec: hi,
            syllable_id: None,
            label: BlockLabel::Manual,
        };
        out.extend(normalize_loop_span(&projection, &span));
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
    create_block_with_id(
        timeline,
        root_track_id,
        edits,
        start_frame,
        end_frame,
        value,
        frame_period_ms,
        None,
    )
}

/// 带调用方 id 的创建入口：UI 的预览 / 选择 / 交接必须命中同一个实际块 id。
#[allow(clippy::too_many_arguments)]
pub fn create_block_with_id(
    timeline: &TimelineState,
    root_track_id: &str,
    edits: &ParamBlockEditSet,
    start_frame: u32,
    end_frame: u32,
    value: Option<f64>,
    frame_period_ms: f64,
    block_id: Option<&str>,
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
        id: block_id.map(str::to_owned).unwrap_or_else(|| {
            new_id(
                "ublock",
                &format!("create:{start_frame}:{end_frame}:{}", next_seq(edits)),
            )
        }),
        geometry,
        deleted: false,
        manual: true,
        value,
        transition_out_sec: None,
        created_seq: next_seq(edits),
        base_cover: None,
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

fn keep_take_members_outside_range(
    timeline: &TimelineState,
    members: &[TakeSpan],
    start_frame: u32,
    end_frame: u32,
    frame_period_ms: f64,
) -> Vec<TakeSpan> {
    let mut out = Vec::new();
    for member in members {
        let Some(clip) = timeline.clips.iter().find(|clip| clip.id == member.clip_id) else {
            continue;
        };
        let Some(take) = clip.take(&member.take_id) else {
            continue;
        };
        let projection = ClipProjection::from_clip_take(clip, take);
        for interval in project_span(&projection, member, frame_period_ms) {
            if interval.cycle != member.cycle {
                continue;
            }
            let mut pieces: Vec<(u32, u32)> = Vec::new();
            if interval.start_frame < start_frame {
                pieces.push((
                    interval.start_frame,
                    interval.end_frame_exclusive.min(start_frame),
                ));
            }
            if interval.end_frame_exclusive > end_frame {
                pieces.push((
                    interval.start_frame.max(end_frame),
                    interval.end_frame_exclusive,
                ));
            }
            for (a, b) in pieces {
                if b <= a {
                    continue;
                }
                let Some(source_a) =
                    source_at_projected_frame(&projection, a, member.cycle, frame_period_ms)
                else {
                    continue;
                };
                let Some(source_b) =
                    source_at_projected_frame(&projection, b, member.cycle, frame_period_ms)
                else {
                    continue;
                };
                let span = TakeSpan {
                    clip_id: member.clip_id.clone(),
                    take_id: member.take_id.clone(),
                    cycle: member.cycle,
                    source_start_sec: source_a.min(source_b),
                    source_end_sec: source_a.max(source_b),
                    syllable_id: member.syllable_id.clone(),
                    label: member.label,
                };
                out.extend(normalize_loop_span(&projection, &span));
            }
        }
    }
    out
}

pub fn restore_game_blocks(
    timeline: &TimelineState,
    root_track_id: &str,
    edits: &ParamBlockEditSet,
    start_frame: u32,
    end_frame: u32,
    frame_period_ms: f64,
) -> Result<ParamBlockEditSet, String> {
    let mut next = edits.clone();
    let mut kept_blocks = Vec::new();
    for block in &next.blocks {
        match &block.geometry {
            BlockGeometry::Root { span } => {
                let start_sec = start_frame as f64 * frame_period_ms.max(0.1) / 1000.0;
                let end_sec = end_frame as f64 * frame_period_ms.max(0.1) / 1000.0;
                for (index, (a, b)) in
                    subtract_interval(span.start_sec, span.end_sec, start_sec, end_sec)
                        .into_iter()
                        .enumerate()
                {
                    if b <= a + EPS {
                        continue;
                    }
                    let mut part = block.clone();
                    if index > 0 {
                        part.id = new_id(
                            "ublock",
                            &format!("restore-piece:{}:{a:.9}:{b:.9}", block.id),
                        );
                    }
                    part.geometry = BlockGeometry::Root {
                        span: RootSpan {
                            start_sec: a,
                            end_sec: b,
                        },
                    };
                    kept_blocks.push(part);
                }
            }
            BlockGeometry::Take { members } => {
                let kept = keep_take_members_outside_range(
                    timeline,
                    members,
                    start_frame,
                    end_frame,
                    frame_period_ms,
                );
                if kept.is_empty() {
                    continue;
                }
                let mut part = block.clone();
                part.geometry = BlockGeometry::Take { members: kept };
                kept_blocks.push(part);
            }
            BlockGeometry::Group { children } => {
                let mut kept_children = Vec::new();
                let start_sec = start_frame as f64 * frame_period_ms.max(0.1) / 1000.0;
                let end_sec = end_frame as f64 * frame_period_ms.max(0.1) / 1000.0;
                for child in children {
                    match &child.geometry {
                        BlockGeometry::Take { members } => {
                            let kept = keep_take_members_outside_range(
                                timeline,
                                members,
                                start_frame,
                                end_frame,
                                frame_period_ms,
                            );
                            if !kept.is_empty() {
                                let mut part = child.clone();
                                part.geometry = BlockGeometry::Take { members: kept };
                                kept_children.push(part);
                            }
                        }
                        BlockGeometry::Root { span } => {
                            for (index, (a, b)) in
                                subtract_interval(span.start_sec, span.end_sec, start_sec, end_sec)
                                    .into_iter()
                                    .enumerate()
                            {
                                if b <= a + EPS {
                                    continue;
                                }
                                let mut part = child.clone();
                                if index > 0 {
                                    part.id = new_id(
                                        "ublock",
                                        &format!("restore-group:{}:{a:.9}:{b:.9}", child.id),
                                    );
                                }
                                part.geometry = BlockGeometry::Root {
                                    span: RootSpan {
                                        start_sec: a,
                                        end_sec: b,
                                    },
                                };
                                kept_children.push(part);
                            }
                        }
                        BlockGeometry::Group { .. } => {}
                    }
                }
                if kept_children.is_empty() {
                    continue;
                }
                if kept_children.len() == 1 {
                    let mut child = kept_children.remove(0);
                    if block.value.is_some() {
                        child.value = block.value;
                    }
                    kept_blocks.push(child);
                } else {
                    let mut part = block.clone();
                    part.geometry = BlockGeometry::Group {
                        children: kept_children,
                    };
                    kept_blocks.push(part);
                }
            }
        }
    }
    next.blocks = kept_blocks;
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
                for interval in project_span(&projection, &candidate, frame_period_ms) {
                    if interval.cycle != cycle {
                        continue;
                    }
                    let a = interval.start_frame.max(start_frame);
                    let b = interval.end_frame_exclusive.min(end_frame);
                    if b <= a {
                        continue;
                    }
                    let Some(source_a) =
                        source_at_projected_frame(&projection, a, cycle, frame_period_ms)
                    else {
                        continue;
                    };
                    let Some(source_b) =
                        source_at_projected_frame(&projection, b, cycle, frame_period_ms)
                    else {
                        continue;
                    };
                    let member = TakeSpan {
                        clip_id: clip.id.clone(),
                        take_id: take.id.clone(),
                        cycle,
                        source_start_sec: source_a.min(source_b),
                        source_end_sec: source_a.max(source_b),
                        syllable_id: Some(format!("game:{index}:{}:{cycle}", span.start_sec)),
                        label: BlockLabel::GameBaseline,
                    };
                    let target = UserParamBlock {
                        id: new_id(
                            "ublock",
                            &format!(
                                "restore:{}:{}:{}:{}:{}:{}",
                                clip.id, index, span.start_sec, cycle, a, b
                            ),
                        ),
                        geometry: BlockGeometry::Take {
                            members: vec![member.clone()],
                        },
                        deleted: false,
                        manual: false,
                        value: None,
                        transition_out_sec: None,
                        created_seq: next.next_seq(),
                        base_cover: Some(super::types::BaseCover {
                            spans: vec![member],
                            full: false,
                        }),
                    };
                    next = place_user_block(&next, target)?;
                }
            }
        }
    }
    next.revision = edits.revision.saturating_add(1);
    next.validate()?;
    Ok(next)
}

/// 多段范围还原：每一段独立执行一次 GAME 基线还原，段间保持原样。
pub fn restore_game_block_ranges(
    timeline: &TimelineState,
    root_track_id: &str,
    edits: &ParamBlockEditSet,
    ranges: &[(u32, u32)],
    frame_period_ms: f64,
) -> Result<ParamBlockEditSet, String> {
    let mut next = edits.clone();
    for (start_frame, end_frame) in ranges {
        if end_frame <= start_frame {
            continue;
        }
        next = restore_game_blocks(
            timeline,
            root_track_id,
            &next,
            *start_frame,
            *end_frame,
            frame_period_ms,
        )?;
    }
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
        // 基块衬底下，resize 后视图包含左半、右半、右侧基块余段三块。
        assert_eq!(snapshots.len(), 3);
        let deleted_view = compose_param_block_view(&timeline, "root", &deleted, 5.0).unwrap();
        // 基块衬底仍会显示（用户删掉的是自己的编辑，不是原始曲线）；只要求
        // 视图里不再有任何用户块。与下面 split 视图的过滤判据保持同一口径。
        assert!(deleted_view
            .blocks
            .iter()
            .all(|block| block.kind == ProjectedBlockKind::Base));
        let restored = restore_deleted_blocks(&deleted, &snapshots).unwrap();
        let restored_view = compose_param_block_view(&timeline, "root", &restored, 5.0).unwrap();
        assert_eq!(restored_view.blocks.len(), 3);
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
    fn group_non_adjacent_blocks_without_filling_gap() {
        let timeline = timeline();
        let edits = ParamBlockEditSet {
            schema_version: 2,
            revision: 1,
            blocks: vec![
                UserParamBlock {
                    id: "left".into(),
                    geometry: BlockGeometry::Take {
                        members: vec![TakeSpan {
                            clip_id: "clip".into(),
                            take_id: "take".into(),
                            cycle: 0,
                            source_start_sec: 0.0,
                            source_end_sec: 0.2,
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
                },
                UserParamBlock {
                    id: "right".into(),
                    geometry: BlockGeometry::Take {
                        members: vec![TakeSpan {
                            clip_id: "clip".into(),
                            take_id: "take".into(),
                            cycle: 0,
                            source_start_sec: 0.6,
                            source_end_sec: 0.8,
                            syllable_id: None,
                            label: BlockLabel::Manual,
                        }],
                    },
                    deleted: false,
                    manual: true,
                    value: Some(60.0),
                    transition_out_sec: None,
                    created_seq: 2,
                    base_cover: None,
                },
            ],
        };
        let view = compose_param_block_view(&timeline, "root", &edits, 5.0).unwrap();
        let selected: Vec<ProjectedParamBlock> = view
            .blocks
            .iter()
            .filter(|block| block.kind != ProjectedBlockKind::Base)
            .cloned()
            .collect();
        assert_eq!(selected.len(), 2);
        let grouped = group_blocks(&timeline, &edits, &selected, 5.0).unwrap();
        let grouped_view = compose_param_block_view(&timeline, "root", &grouped, 5.0).unwrap();
        let runs: Vec<_> = grouped_view
            .blocks
            .iter()
            .filter(|block| block.kind != ProjectedBlockKind::Base)
            .collect();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].start_frame, 0);
        assert_eq!(runs[0].end_frame_exclusive, 40);
        assert_eq!(runs[1].start_frame, 120);
        assert_eq!(runs[1].end_frame_exclusive, 160);
        let group_id = runs[0].user_block_id.clone().unwrap();
        assert_eq!(runs[1].user_block_id.as_deref(), Some(group_id.as_str()));
        assert!(runs
            .iter()
            .all(|block| !(block.start_frame < 80 && block.end_frame_exclusive > 80)));
        let ungrouped = ungroup_user_block(&grouped, &group_id).unwrap();
        let ungrouped_view = compose_param_block_view(&timeline, "root", &ungrouped, 5.0).unwrap();
        let restored: Vec<_> = ungrouped_view
            .blocks
            .iter()
            .filter(|block| block.kind != ProjectedBlockKind::Base)
            .collect();
        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0].start_frame, 0);
        assert_eq!(restored[1].start_frame, 120);
    }

    #[test]
    fn grouped_blocks_split_and_resize() {
        let timeline = timeline();
        let view = base_view(&timeline);
        let grouped =
            group_blocks(&timeline, &ParamBlockEditSet::default(), &view.blocks, 5.0).unwrap();
        let grouped_view = compose_param_block_view(&timeline, "root", &grouped, 5.0).unwrap();
        let group_id = grouped_view.blocks[0].id.clone();
        let split = split_block(&timeline, &grouped, &group_id, 50, 5.0).unwrap();
        let split_view = compose_param_block_view(&timeline, "root", &split, 5.0).unwrap();
        assert_eq!(
            split_view
                .blocks
                .iter()
                .filter(|block| block.kind != crate::param_blocks::ProjectedBlockKind::Base)
                .count(),
            2
        );
        assert!(
            split
                .blocks
                .iter()
                .filter(|block| block.is_group())
                .flat_map(|block| block.flatten_children())
                .count()
                >= 2
        );

        let resized = resize_block(&timeline, &grouped, &group_id, "right", 150, 5.0).unwrap();
        let resized_view = compose_param_block_view(&timeline, "root", &resized, 5.0).unwrap();
        // 组伸缩会重建容器（新 id），因此按结果几何断言而不是按 id 找回。
        let user_blocks: Vec<_> = resized_view
            .blocks
            .iter()
            .filter(|block| block.kind != crate::param_blocks::ProjectedBlockKind::Base)
            .collect();
        assert_eq!(user_blocks.len(), 1);
        assert_eq!(user_blocks[0].end_frame_exclusive, 150);
    }

    #[test]
    fn group_delete_creates_child_tombstones_and_restores() {
        let timeline = timeline();
        let base = base_view(&timeline);
        let grouped =
            group_blocks(&timeline, &ParamBlockEditSet::default(), &base.blocks, 5.0).unwrap();
        let grouped_view = compose_param_block_view(&timeline, "root", &grouped, 5.0).unwrap();
        let (deleted, snapshots) =
            delete_blocks(&timeline, &grouped, &grouped_view.blocks, 5.0).unwrap();
        assert_eq!(snapshots.len(), 1);
        assert!(snapshots[0].child_tombstones.len() >= 2);
        let deleted_view = compose_param_block_view(&timeline, "root", &deleted, 5.0).unwrap();
        assert!(deleted_view.blocks.is_empty(), "base must stay hidden");
        let restored = restore_deleted_blocks(&deleted, &snapshots).unwrap();
        let restored_view = compose_param_block_view(&timeline, "root", &restored, 5.0).unwrap();
        assert_eq!(restored_view.blocks.len(), 1);
        assert_eq!(
            restored_view.blocks[0].kind,
            crate::param_blocks::ProjectedBlockKind::Grouped
        );
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
            base_cover: None,
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
            base_cover: None,
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
            schema_version: 2,
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
                base_cover: None,
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
        // 两个 member 之间的缝会露出一格基块衬底：视图是
        // [user 0..100, base 100..101, user 101..200]，不再是"合并成一块"。
        // 与下面 split_view 的判据一致，只数用户块。
        let user_blocks: Vec<_> = view
            .blocks
            .iter()
            .filter(|block| block.kind != ProjectedBlockKind::Base)
            .collect();
        assert_eq!(user_blocks.len(), 2);
        assert_eq!(user_blocks[0].start_frame, 0);
        assert_eq!(user_blocks[1].end_frame_exclusive, 200);
        // 多成员 run 不是可逆编组：它照样可以切分。
        assert!(user_blocks
            .iter()
            .all(|block| block.kind == ProjectedBlockKind::User));

        // 旧实现把两个 member 并成一块、缝里（帧 100）也能切；现在缝由一格
        // base 衬底覆盖，两侧各是一个用户块，切点必须落在用户块内部。取左半
        // 的帧 50，仍守住"多成员块切一刀之后还能再切"这条回归。
        let split = split_block(&timeline, &edits, &user_blocks[0].id, 50, 5.0).unwrap();
        let split_view = compose_param_block_view(&timeline, "root", &split, 5.0).unwrap();
        // 切完左半：0..50、50..100 两块，加上缝里的 base 与右半 101..200，
        // 用户块共三块。
        let user_blocks: Vec<_> = split_view
            .blocks
            .iter()
            .filter(|block| block.kind != ProjectedBlockKind::Base)
            .collect();
        assert_eq!(user_blocks.len(), 3);
        assert_eq!(user_blocks[0].start_frame, 0);
        assert_eq!(user_blocks[2].end_frame_exclusive, 200);

        // 切完的右半仍然可以再切（回归"多成员块切一刀之后就切不动了"）。
        let right_user = split_view
            .blocks
            .iter()
            .find(|block| {
                block.kind != ProjectedBlockKind::Base && block.end_frame_exclusive == 200
            })
            .unwrap();
        let again = split_block(&timeline, &split, &right_user.id, 150, 5.0).unwrap();
        let again_view = compose_param_block_view(&timeline, "root", &again, 5.0).unwrap();
        assert_eq!(
            again_view
                .blocks
                .iter()
                .filter(|block| block.kind != ProjectedBlockKind::Base)
                .count(),
            4
        );
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
