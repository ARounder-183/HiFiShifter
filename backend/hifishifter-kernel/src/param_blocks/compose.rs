//! base 分段与用户覆盖层的合成。
//!
//! 职责：抑制、优先级、重叠消解、过渡、视图校验。
//! 关系：`base` 提供 base 块，`projection` 提供投影，`layout` 提供布局键。
//! 维护约束：用户显式编辑 > base 分析；deleted 墓碑只抑制 base，不显示。

use super::base::{base_blocks_for_root, root_analysis_pending};
use super::layout::layout_key;
use super::projection::{
    normalize_loop_span, project_span, source_at_projected_frame, ClipProjection,
};
use super::types::{
    BaseCover, BlockGeometry, ParamBlockEditSet, ParamBlockView, ProjectedBlockKind,
    ProjectedMember, ProjectedParamBlock, TakeSpan, UserParamBlock,
};
use super::validate::validate_view;
use crate::state::TimelineState;

const ANCHOR_TOLERANCE_SEC: f64 = 0.005;
const ANCHOR_OVERLAP_RATIO: f64 = 0.5;

#[derive(Clone, Debug)]
struct Candidate {
    id: String,
    user_block_id: Option<String>,
    start: u32,
    end: u32,
    kind: ProjectedBlockKind,
    members: Vec<ProjectedMember>,
    value: Option<f64>,
    transition_width_sec: Option<f64>,
    transition_frames: Option<u32>,
    rate: f64,
    note: Option<f32>,
    cut_start: bool,
    cut_end: bool,
    source_key: Option<String>,
    priority: u8,
    seq: u64,
    z_order: u32,
    base_cover: Option<BaseCover>,
}

fn member_source_key(member: &ProjectedMember) -> String {
    format!("{}|{}|{}", member.clip_id, member.take_id, member.cycle)
}

fn first_member_source_key(members: &[ProjectedMember]) -> Option<String> {
    members.first().map(member_source_key)
}

/// Find the source note with the largest overlap with this member's source span.
fn source_note_for_member(timeline: &TimelineState, member: &ProjectedMember) -> Option<f32> {
    let clip = timeline
        .clips
        .iter()
        .find(|clip| clip.id == member.clip_id)?;
    let take = clip.take(&member.take_id)?;
    let mut spans: Vec<(f64, f64, Option<f32>)> = Vec::new();
    if let Some(analysis) = &take.syllable_analysis {
        spans.extend(
            analysis
                .source_spans_sec
                .iter()
                .map(|span| (span.start_sec, span.end_sec, span.midi_note)),
        );
    }
    if spans.is_empty() {
        if let Some(baseline) = &take.game_block_baseline {
            spans.extend(
                baseline
                    .source_spans_sec
                    .iter()
                    .map(|span| (span.start_sec, span.end_sec, span.midi_note)),
            );
        }
    }
    if spans.is_empty() {
        if let Some(notes) = &take.midi_note_data {
            spans.extend(
                notes
                    .iter()
                    .map(|note| (note.start_sec, note.end_sec, Some(note.note))),
            );
        }
    }
    let lo = member.visible_source_start_sec.min(member.source_start_sec);
    let hi = member.visible_source_end_sec.max(member.source_end_sec);
    let mut best: Option<(f64, f32)> = None;
    for (start, end, note) in spans {
        let overlap = end.min(hi) - start.max(lo);
        if overlap <= 0.0 {
            continue;
        }
        if let Some(pitch) = note {
            if best.map_or(true, |(width, _)| overlap > width) {
                best = Some((overlap, pitch));
            }
        }
    }
    best.map(|(_, pitch)| pitch)
}

fn interval_width(start: u32, end: u32) -> u32 {
    end.saturating_sub(start)
}

fn member_from(
    span: &TakeSpan,
    start: u32,
    end: u32,
    projection: &ClipProjection,
    frame_period_ms: f64,
    cut_start: bool,
    cut_end: bool,
) -> ProjectedMember {
    let frame_source_start_sec =
        source_at_projected_frame(projection, start, span.cycle, frame_period_ms)
            .unwrap_or(span.source_start_sec);
    let frame_source_end_sec =
        source_at_projected_frame(projection, end, span.cycle, frame_period_ms)
            .unwrap_or(span.source_end_sec);
    ProjectedMember {
        clip_id: span.clip_id.clone(),
        take_id: span.take_id.clone(),
        cycle: span.cycle,
        source_start_sec: span.source_start_sec,
        source_end_sec: span.source_end_sec,
        syllable_id: span.syllable_id.clone(),
        label: Some(span.label),
        start_frame: start,
        end_frame_exclusive: end,
        frame_source_start_sec,
        frame_source_end_sec,
        visible_source_start_sec: frame_source_start_sec.min(frame_source_end_sec),
        visible_source_end_sec: frame_source_start_sec.max(frame_source_end_sec),
        cut_start,
        cut_end,
    }
}

/// 把用户块投影成连续的候选块。返回 (候选, orphan 成员数)。
fn project_take_members(
    timeline: &TimelineState,
    id: &str,
    members: &[TakeSpan],
    kind: ProjectedBlockKind,
    value: Option<f64>,
    transition_width_sec: Option<f64>,
    priority: u8,
    created_seq: u64,
    base_cover: Option<BaseCover>,
    frame_period_ms: f64,
) -> (Vec<Candidate>, usize) {
    let mut projected: Vec<(u32, u32, ProjectedMember, f64)> = Vec::new();
    let mut orphan = 0usize;
    for member in members {
        let Some(clip) = timeline.clips.iter().find(|clip| clip.id == member.clip_id) else {
            orphan += 1;
            continue;
        };
        let Some(take) = clip.take(&member.take_id) else {
            orphan += 1;
            continue;
        };
        let projection = ClipProjection::from_clip_take(clip, take);
        let mut matched = false;
        for part in normalize_loop_span(&projection, member) {
            for interval in project_span(&projection, &part, frame_period_ms) {
                /*
                 * 用户成员是周期特定的：cycle=N 只画第 N 个周期。基块
                 * 在 base_candidates 中用 < 收集后续周期；用户块若也用 <，
                 * 一个 cycle=0 成员会吞掉整段 Loop 后面的基块。
                 */
                if interval.cycle != part.cycle {
                    continue;
                }
                matched = true;
                projected.push((
                    interval.start_frame,
                    interval.end_frame_exclusive,
                    member_from(
                        &part,
                        interval.start_frame,
                        interval.end_frame_exclusive,
                        &projection,
                        frame_period_ms,
                        interval.cut_start,
                        interval.cut_end,
                    ),
                    projection.effective_rate,
                ));
            }
        }
        if !matched {
            orphan += 1;
        }
    }
    if projected.is_empty() {
        return (Vec::new(), orphan);
    }
    projected.sort_by_key(|(start, end, _, _)| (*start, *end));
    let mut runs: Vec<(u32, u32, Vec<ProjectedMember>, f64)> = Vec::new();
    for (start, end, member, rate) in projected {
        match runs.last_mut() {
            Some(run) if start <= run.1 => {
                run.1 = run.1.max(end);
                run.2.push(member);
            }
            _ => runs.push((start, end, vec![member], rate)),
        }
    }
    let count = runs.len();
    let mut out = Vec::new();
    for (index, (start, end, run_members, rate)) in runs.into_iter().enumerate() {
        let candidate_id = if count == 1 {
            id.to_owned()
        } else {
            format!("{id}:{index}")
        };
        out.push(Candidate {
            id: candidate_id,
            user_block_id: Some(id.to_owned()),
            start,
            end,
            kind,
            members: run_members,
            value,
            transition_width_sec,
            transition_frames: None,
            rate,
            note: None,
            cut_start: false,
            cut_end: false,
            source_key: None,
            priority,
            seq: created_seq,
            z_order: 0,
            base_cover: base_cover.clone(),
        });
    }
    (out, orphan)
}

fn project_group_block(
    timeline: &TimelineState,
    block: &UserParamBlock,
    children: &[UserParamBlock],
    frame_period_ms: f64,
) -> (Vec<Candidate>, usize) {
    let leaves = block.flatten_children();
    let manual = block.manual || leaves.iter().any(|child| child.manual);
    let priority = if manual { 4 } else { 3 };
    let value = block.value.or_else(|| block.group_common_value());
    let transition_width_sec = block
        .transition_out_sec
        .or_else(|| children.last().and_then(|child| child.transition_out_sec));
    let has_root = leaves
        .iter()
        .any(|child| matches!(child.geometry, BlockGeometry::Root { .. }));
    if has_root {
        /*
         * 根块组必须按每个 child 生成独立 run；min..max 包围盒会把子块之间
         * 的空白也填成组，等于凭空制造几何。
         */
        let mut runs: Vec<(u32, u32)> = leaves
            .iter()
            .filter_map(|child| match child.geometry {
                BlockGeometry::Root { ref span } => {
                    let start = (span.start_sec.max(0.0) * 1000.0 / frame_period_ms.max(0.1))
                        .round() as u32;
                    let end =
                        (span.end_sec.max(0.0) * 1000.0 / frame_period_ms.max(0.1)).round() as u32;
                    (end > start).then_some((start, end))
                }
                _ => None,
            })
            .collect();
        runs.sort_by_key(|(start, end)| (*start, *end));
        if runs.is_empty() {
            return (Vec::new(), 0);
        }
        let count = runs.len();
        return (
            runs.into_iter()
                .enumerate()
                .map(|(index, (start, end))| Candidate {
                    id: if count == 1 {
                        block.id.clone()
                    } else {
                        format!("{}:{}", block.id, index)
                    },
                    user_block_id: Some(block.id.clone()),
                    start,
                    end,
                    kind: ProjectedBlockKind::Grouped,
                    members: Vec::new(),
                    value,
                    transition_width_sec,
                    transition_frames: None,
                    rate: 1.0,
                    note: None,
                    cut_start: false,
                    cut_end: false,
                    source_key: None,
                    priority,
                    seq: block.created_seq,
                    z_order: 0,
                    base_cover: None,
                })
                .collect(),
            0,
        );
    }
    let members = block.leaf_members();
    project_take_members(
        timeline,
        &block.id,
        &members,
        ProjectedBlockKind::Grouped,
        value,
        transition_width_sec,
        priority,
        block.created_seq,
        None,
        frame_period_ms,
    )
}

/// 把用户块投影成连续的候选块。返回 (候选, orphan 成员数)。
fn project_user_block(
    timeline: &TimelineState,
    block: &UserParamBlock,
    frame_period_ms: f64,
) -> (Vec<Candidate>, usize) {
    match &block.geometry {
        BlockGeometry::Root { span } => {
            let start =
                (span.start_sec.max(0.0) * 1000.0 / frame_period_ms.max(0.1)).round() as u32;
            let end = (span.end_sec.max(0.0) * 1000.0 / frame_period_ms.max(0.1)).round() as u32;
            if end <= start {
                return (Vec::new(), 0);
            }
            (
                vec![Candidate {
                    id: block.id.clone(),
                    user_block_id: Some(block.id.clone()),
                    start,
                    end,
                    kind: if block.manual {
                        ProjectedBlockKind::Manual
                    } else {
                        ProjectedBlockKind::User
                    },
                    members: Vec::new(),
                    value: block.value,
                    transition_width_sec: block.transition_out_sec,
                    transition_frames: None,
                    rate: 1.0,
                    note: None,
                    cut_start: false,
                    cut_end: false,
                    source_key: None,
                    priority: if block.manual { 4 } else { 3 },
                    seq: block.created_seq,
                    z_order: 0,
                    base_cover: block.base_cover.clone(),
                }],
                0,
            )
        }
        BlockGeometry::Take { members } => project_take_members(
            timeline,
            &block.id,
            members,
            if block.manual {
                ProjectedBlockKind::Manual
            } else {
                ProjectedBlockKind::User
            },
            block.value,
            block.transition_out_sec,
            if block.manual { 4 } else { 3 },
            block.created_seq,
            block.base_cover.clone(),
            frame_period_ms,
        ),
        BlockGeometry::Group { children } => {
            project_group_block(timeline, block, children, frame_period_ms)
        }
    }
}

/// 把一个用户块单独投影成视图块，供编辑原语在不知道完整 view 时定位几何。
pub fn project_single_user_block(
    timeline: &TimelineState,
    block: &UserParamBlock,
    frame_period_ms: f64,
) -> Result<Option<ProjectedParamBlock>, String> {
    let (candidates, _) = project_user_block(timeline, block, frame_period_ms);
    let Some(candidate) = candidates
        .into_iter()
        .find(|candidate| candidate.id == block.id)
        .or_else(|| {
            project_user_block(timeline, block, frame_period_ms)
                .0
                .into_iter()
                .next()
        })
    else {
        return Ok(None);
    };
    Ok(Some(ProjectedParamBlock {
        id: candidate.id,
        user_block_id: candidate.user_block_id,
        start_frame: candidate.start,
        end_frame_exclusive: candidate.end,
        transition_frames: candidate.transition_frames,
        kind: candidate.kind,
        members: candidate.members,
        value: candidate.value,
        source_note: candidate.note,
        source_key: candidate.source_key,
        cut_start: candidate.cut_start,
        cut_end: candidate.cut_end,
        base_cover: candidate.base_cover,
    }))
}

/// 把逻辑块的全部可见 run 合成一个投影块。组算法 / 切分应使用它，
/// 而不是只看第一个 run。
pub fn project_logical_user_block(
    timeline: &TimelineState,
    block: &UserParamBlock,
    frame_period_ms: f64,
) -> Result<Option<ProjectedParamBlock>, String> {
    let (candidates, _) = project_user_block(timeline, block, frame_period_ms);
    if candidates.is_empty() {
        return Ok(None);
    }
    let start = candidates
        .iter()
        .map(|candidate| candidate.start)
        .min()
        .unwrap_or(0);
    let end = candidates
        .iter()
        .map(|candidate| candidate.end)
        .max()
        .unwrap_or(0);
    let members = candidates
        .iter()
        .flat_map(|candidate| candidate.members.iter().cloned())
        .collect();
    let first = &candidates[0];
    Ok(Some(ProjectedParamBlock {
        id: block.id.clone(),
        user_block_id: Some(block.id.clone()),
        start_frame: start,
        end_frame_exclusive: end,
        transition_frames: first.transition_frames,
        kind: first.kind,
        members,
        value: block.value,
        source_note: first.note,
        source_key: first.source_key.clone(),
        cut_start: first.cut_start,
        cut_end: first.cut_end,
        base_cover: block.base_cover.clone(),
    }))
}

fn anchor_matches(base: &TakeSpan, member: &TakeSpan) -> bool {
    if base.syllable_id.is_some()
        && member.syllable_id.is_some()
        && base.syllable_id == member.syllable_id
    {
        return true;
    }
    let same_edges = (base.source_start_sec - member.source_start_sec).abs()
        <= ANCHOR_TOLERANCE_SEC
        && (base.source_end_sec - member.source_end_sec).abs() <= ANCHOR_TOLERANCE_SEC;
    if same_edges {
        return true;
    }
    let overlap = base.source_end_sec.min(member.source_end_sec)
        - base.source_start_sec.max(member.source_start_sec);
    let width = (base.source_end_sec - base.source_start_sec)
        .min(member.source_end_sec - member.source_start_sec);
    overlap > 0.0 && width > 1e-9 && overlap / width >= ANCHOR_OVERLAP_RATIO
}

fn base_span_of(candidate: &Candidate) -> Option<&TakeSpan> {
    candidate
        .base_cover
        .as_ref()
        .and_then(|cover| cover.spans.first())
}

/// 递归收集组内全部叶子的 base_cover；组本身不携带 base_cover。
fn collect_base_covers(block: &UserParamBlock) -> Vec<&BaseCover> {
    let mut covers: Vec<&BaseCover> = Vec::new();
    if let Some(cover) = block.base_cover.as_ref() {
        covers.push(cover);
    }
    if let BlockGeometry::Group { children } = &block.geometry {
        for child in children {
            covers.extend(collect_base_covers(child));
        }
    }
    covers
}

/// 收集所有会“拥有”这个基块的覆盖区间。
///
/// 普通手工块没有 base_cover，不在这里出现；它只通过占用区间覆盖基块，
/// 因此会留下两侧剩余段。
fn cover_intervals_for_base(
    timeline: &TimelineState,
    base: &Candidate,
    edits: &ParamBlockEditSet,
    frame_period_ms: f64,
) -> Vec<(u32, u32)> {
    let Some(base_span) = base_span_of(base) else {
        return Vec::new();
    };
    let mut intervals = Vec::new();
    for block in &edits.blocks {
        /*
         * 旧版墓碑没有 base_cover：它只有“删除即抑制基块”的语义。为了兼容旧
         * 工程，把 deleted + 无 base_cover 的块按它自己的成员投影区间当作覆盖。
         * 组则递归收集全部叶子的 base_cover。
         */
        let covers = collect_base_covers(block);
        if covers.is_empty() {
            if block.deleted {
                for span in block.members() {
                    if span.clip_id != base_span.clip_id
                        || span.take_id != base_span.take_id
                        || span.cycle != base_span.cycle
                    {
                        continue;
                    }
                    let Some(clip) = timeline.clips.iter().find(|clip| clip.id == span.clip_id)
                    else {
                        continue;
                    };
                    let Some(take) = clip.take(&span.take_id) else {
                        continue;
                    };
                    let projection = ClipProjection::from_clip_take(clip, take);
                    for interval in project_span(&projection, span, frame_period_ms) {
                        if interval.cycle != span.cycle {
                            continue;
                        }
                        let start = base.start.max(interval.start_frame);
                        let end = base.end.min(interval.end_frame_exclusive);
                        if end > start {
                            intervals.push((start, end));
                        }
                    }
                }
            }
            continue;
        }
        for cover in covers {
            for span in &cover.spans {
                if span.clip_id != base_span.clip_id
                    || span.take_id != base_span.take_id
                    || span.cycle != base_span.cycle
                {
                    continue;
                }
                if cover.full
                    && anchor_matches(base_span, span)
                    && span.source_start_sec <= base_span.source_start_sec + 1e-7
                    && span.source_end_sec >= base_span.source_end_sec - 1e-7
                {
                    return vec![(base.start, base.end)];
                }
                let Some(clip) = timeline.clips.iter().find(|clip| clip.id == span.clip_id) else {
                    continue;
                };
                let Some(take) = clip.take(&span.take_id) else {
                    continue;
                };
                let projection = ClipProjection::from_clip_take(clip, take);
                for interval in project_span(&projection, span, frame_period_ms) {
                    if interval.cycle != span.cycle {
                        continue;
                    }
                    let start = base.start.max(interval.start_frame);
                    let end = base.end.min(interval.end_frame_exclusive);
                    if end > start {
                        intervals.push((start, end));
                    }
                }
            }
        }
    }
    intervals
}

fn subtract_span(start: u32, end: u32, occupied: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut pieces = vec![(start, end)];
    for (occupied_start, occupied_end) in occupied {
        let mut next = Vec::new();
        for (piece_start, piece_end) in pieces {
            if *occupied_end <= piece_start || *occupied_start >= piece_end {
                next.push((piece_start, piece_end));
                continue;
            }
            if piece_start < *occupied_start {
                next.push((piece_start, *occupied_start));
            }
            if piece_end > *occupied_end {
                next.push((*occupied_end, piece_end));
            }
        }
        pieces = next;
        if pieces.is_empty() {
            break;
        }
    }
    pieces
}

fn base_piece_candidate(
    base: &Candidate,
    start: u32,
    end: u32,
    timeline: &TimelineState,
    frame_period_ms: f64,
) -> Option<Candidate> {
    let member = base.members.first()?;
    let base_span = base_span_of(base)?;
    let clip = timeline
        .clips
        .iter()
        .find(|clip| clip.id == member.clip_id)?;
    let take = clip.take(&member.take_id)?;
    let projection = ClipProjection::from_clip_take(clip, take);
    let a = source_at_projected_frame(&projection, start, member.cycle, frame_period_ms)?;
    let b = source_at_projected_frame(&projection, end, member.cycle, frame_period_ms)?;
    let (lo, hi) = (a.min(b), a.max(b));
    // A positive frame interval is visible truth. Rounding may collapse the
    // source coordinates; one frame is the smallest editable local interval.
    let (source_start, mut source_end) = if hi > lo + 1e-12 {
        (lo, hi)
    } else {
        (
            lo,
            lo + frame_period_ms.max(0.1) / 1000.0 * projection.effective_rate.max(1e-9),
        )
    };
    if source_end <= source_start {
        source_end = source_start + 1e-12;
    }
    let piece_span = TakeSpan {
        clip_id: member.clip_id.clone(),
        take_id: member.take_id.clone(),
        cycle: member.cycle,
        source_start_sec: source_start,
        source_end_sec: source_end,
        syllable_id: None,
        label: member.label.unwrap_or(base_span.label),
    };
    let mut piece = base.clone();
    piece.id = format!("{}:{}:{}", base.id, start, end);
    piece.start = start;
    piece.end = end;
    piece.members = vec![member_from(
        &piece_span,
        start,
        end,
        &projection,
        frame_period_ms,
        piece.cut_start && start == base.start,
        piece.cut_end && end == base.end,
    )];
    piece.base_cover = Some(BaseCover {
        spans: vec![piece_span],
        full: false,
    });
    Some(piece)
}

fn trim_member(
    timeline: &TimelineState,
    member: &ProjectedMember,
    start: u32,
    end: u32,
    frame_period_ms: f64,
) -> ProjectedMember {
    let Some(clip) = timeline.clips.iter().find(|clip| clip.id == member.clip_id) else {
        let mut next = member.clone();
        next.start_frame = start;
        next.end_frame_exclusive = end;
        return next;
    };
    let Some(take) = clip.take(&member.take_id) else {
        let mut next = member.clone();
        next.start_frame = start;
        next.end_frame_exclusive = end;
        return next;
    };
    let projection = ClipProjection::from_clip_take(clip, take);
    let a = source_at_projected_frame(&projection, start, member.cycle, frame_period_ms);
    let b = source_at_projected_frame(&projection, end, member.cycle, frame_period_ms);
    let mut next = member.clone();
    next.cut_start = member.start_frame == start && member.cut_start;
    next.cut_end = member.end_frame_exclusive == end && member.cut_end;
    next.start_frame = start;
    next.end_frame_exclusive = end;
    if let Some(a) = a {
        next.frame_source_start_sec = a;
    }
    if let Some(b) = b {
        next.frame_source_end_sec = b;
    }
    next.visible_source_start_sec = next.frame_source_start_sec.min(next.frame_source_end_sec);
    next.visible_source_end_sec = next.frame_source_start_sec.max(next.frame_source_end_sec);
    next
}

pub fn compose_param_block_view(
    timeline: &TimelineState,
    root_track_id: &str,
    edits: &ParamBlockEditSet,
    frame_period_ms: f64,
) -> Result<ParamBlockView, String> {
    let frame_period_ms = frame_period_ms.max(0.1);
    let base_blocks = base_blocks_for_root(timeline, root_track_id, frame_period_ms);
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut orphan_count = 0usize;

    for base in &base_blocks {
        let interval = base.projected[0];
        let candidate = Candidate {
            id: base.id.clone(),
            user_block_id: None,
            start: interval.start_frame,
            end: interval.end_frame_exclusive,
            kind: ProjectedBlockKind::Base,
            members: vec![member_from(
                &base.span,
                interval.start_frame,
                interval.end_frame_exclusive,
                &base.projection,
                frame_period_ms,
                interval.cut_start,
                interval.cut_end,
            )],
            value: None,
            note: base.note,
            cut_start: interval.cut_start,
            cut_end: interval.cut_end,
            source_key: Some(format!(
                "{}|{}|{}",
                base.span.clip_id, base.span.take_id, interval.cycle
            )),
            transition_width_sec: None,
            transition_frames: None,
            rate: base.rate,
            priority: 1,
            seq: 0,
            z_order: base.z_order,
            base_cover: Some(BaseCover {
                spans: vec![base.span.clone()],
                full: true,
            }),
        };
        let covers = cover_intervals_for_base(timeline, &candidate, edits, frame_period_ms);
        for (start, end) in subtract_span(candidate.start, candidate.end, &covers) {
            if end <= start {
                continue;
            }
            if start == candidate.start && end == candidate.end {
                candidates.push(candidate.clone());
            } else if let Some(piece) =
                base_piece_candidate(&candidate, start, end, timeline, frame_period_ms)
            {
                candidates.push(piece);
            }
        }
    }

    for block in edits.blocks.iter().filter(|block| !block.deleted) {
        let (projected, orphan) = project_user_block(timeline, block, frame_period_ms);
        orphan_count += orphan;
        candidates.extend(projected);
    }

    candidates.sort_by(|a, b| {
        b.priority
            .cmp(&a.priority)
            .then(b.seq.cmp(&a.seq))
            .then(b.z_order.cmp(&a.z_order))
            .then(a.id.cmp(&b.id))
    });

    let mut occupied: Vec<(u32, u32)> = Vec::new();
    let mut accepted: Vec<Candidate> = Vec::new();
    for candidate in candidates {
        let pieces = subtract_span(candidate.start, candidate.end, &occupied);
        occupied.push((candidate.start, candidate.end));
        let piece_count = pieces.len();
        for (index, (start, end)) in pieces.into_iter().enumerate() {
            if end <= start {
                continue;
            }
            let trimmed = piece_count > 1 || start != candidate.start || end != candidate.end;
            if candidate.kind == ProjectedBlockKind::Base && trimmed {
                if let Some(mut piece) =
                    base_piece_candidate(&candidate, start, end, timeline, frame_period_ms)
                {
                    piece.id = format!("{}:{}", piece.id, index);
                    accepted.push(piece);
                }
                continue;
            }
            let mut piece = candidate.clone();
            piece.start = start;
            piece.end = end;
            if trimmed {
                piece.id = format!("{}:{}:{}", candidate.id, start, end);
                if piece_count > 1 {
                    piece.id = format!("{}:{}", piece.id, index);
                }
            }
            piece.members = piece
                .members
                .iter()
                .filter(|member| member.end_frame_exclusive > start && member.start_frame < end)
                .map(|member| {
                    trim_member(
                        timeline,
                        member,
                        member.start_frame.max(start),
                        member.end_frame_exclusive.min(end),
                        frame_period_ms,
                    )
                })
                .collect();
            accepted.push(piece);
        }
    }

    accepted.sort_by_key(|candidate| (candidate.start, candidate.end, candidate.id.clone()));
    for candidate in &mut accepted {
        if candidate.source_key.is_none() {
            candidate.source_key = first_member_source_key(&candidate.members);
        }
        if candidate.members.len() == 1 && !matches!(candidate.kind, ProjectedBlockKind::Grouped) {
            candidate.note = source_note_for_member(timeline, &candidate.members[0]);
        }
        candidate.cut_start = candidate
            .members
            .first()
            .is_some_and(|member| member.cut_start);
        candidate.cut_end = candidate
            .members
            .last()
            .is_some_and(|member| member.cut_end);
    }

    // 过渡只作用于相邻块，宽度受本块与右块限制。
    for index in 0..accepted.len().saturating_sub(1) {
        let (left, right) = accepted.split_at_mut(index + 1);
        let prev = &mut left[index];
        let next = &right[0];
        if prev.end != next.start || prev.source_key.is_none() || prev.source_key != next.source_key
        {
            continue;
        }
        let Some(width_sec) = prev.transition_width_sec else {
            continue;
        };
        if width_sec <= 1e-9 {
            continue;
        }
        let width = (width_sec * 1000.0 / frame_period_ms / prev.rate.max(1e-9)).round() as u32;
        let max_width =
            interval_width(prev.start, prev.end).min(interval_width(next.start, next.end) / 2);
        if width > 0 && max_width > 0 {
            prev.transition_frames = Some(width.min(max_width));
        }
    }

    let blocks: Vec<ProjectedParamBlock> = accepted
        .into_iter()
        .map(|candidate| ProjectedParamBlock {
            id: candidate.id,
            user_block_id: candidate.user_block_id,
            start_frame: candidate.start,
            end_frame_exclusive: candidate.end,
            transition_frames: candidate.transition_frames,
            kind: candidate.kind,
            members: candidate.members,
            value: candidate.value,
            source_note: candidate.note,
            source_key: candidate.source_key,
            cut_start: candidate.cut_start,
            cut_end: candidate.cut_end,
            base_cover: candidate.base_cover,
        })
        .collect();

    let view = ParamBlockView {
        schema_version: 1,
        root_track_id: root_track_id.to_owned(),
        edit_revision: edits.revision,
        layout_key: layout_key(timeline, root_track_id, frame_period_ms),
        frame_period_ms,
        blocks,
        orphan_count,
        analysis_pending: root_analysis_pending(timeline, root_track_id),
    };
    validate_view(&view)?;
    Ok(view)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::param_blocks::types::{BlockGeometry, BlockLabel, UserParamBlock};
    use crate::state::TimelineState;

    fn timeline(start_sec: f64, source_start_sec: f64) -> TimelineState {
        let mut timeline: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"root","name":"Root","order":0}],
            "clips":[{
                "id":"clip","track_id":"root","name":"clip","start_sec":start_sec,"length_sec":2.0,
                "takes":[{"id":"take","source_path":"a.wav","source_start_sec":source_start_sec,
                    "source_end_sec":source_start_sec + 2.0,
                    "syllable_analysis":{"algorithm":"test","algorithmVersion":1,"sourceDigest":"d","analyzedWindowSec":[0.0,4.0],
                        "sourceSpansSec":[{"startSec":0.0,"endSec":0.5},{"startSec":0.5,"endSec":1.0}]}}]
            }],
            "project_sec":4,"bpm":120
        })).unwrap();
        timeline.clips[0].normalize_takes();
        timeline
    }

    fn merged_edits() -> ParamBlockEditSet {
        ParamBlockEditSet {
            schema_version: 2,
            revision: 1,
            blocks: vec![UserParamBlock {
                id: "merged".into(),
                geometry: BlockGeometry::Take {
                    members: vec![
                        TakeSpan {
                            clip_id: "clip".into(),
                            take_id: "take".into(),
                            cycle: 0,
                            source_start_sec: 0.0,
                            source_end_sec: 0.5,
                            syllable_id: Some("span:take:0:0".into()),
                            label: BlockLabel::Syllable,
                        },
                        TakeSpan {
                            clip_id: "clip".into(),
                            take_id: "take".into(),
                            cycle: 0,
                            source_start_sec: 0.5,
                            source_end_sec: 1.0,
                            syllable_id: Some("span:take:1:500".into()),
                            label: BlockLabel::Syllable,
                        },
                    ],
                },
                deleted: false,
                manual: false,
                value: None,
                transition_out_sec: None,
                created_seq: 1,
                base_cover: None,
            }],
        }
    }

    #[test]
    fn compose_keeps_the_left_cut_block_at_clip_start() {
        let mut song = timeline(2.0, 1.0);
        song.clips[0].takes[0].syllable_analysis =
            Some(crate::pitch_analysis::TakeSyllableAnalysis {
                algorithm: "test".into(),
                algorithm_version: 1,
                source_digest: "d".into(),
                source_spans_sec: vec![crate::pitch_analysis::SyllableSpan {
                    start_sec: 1.0,
                    end_sec: 1.5,
                    midi_note: Some(60.0),
                }],
                analyzed_window_sec: [1.0, 3.0],
            });
        let view =
            compose_param_block_view(&song, "root", &ParamBlockEditSet::default(), 5.0).unwrap();
        assert_eq!(view.blocks[0].start_frame, 400);
        assert_eq!(view.blocks[0].end_frame_exclusive, 500);
        assert_eq!(view.blocks[0].source_note, Some(60.0));
    }

    #[test]
    fn overlapping_spans_trim_the_later_start_instead_of_dropping_it() {
        let base = super::super::base::base_blocks_for_root(&timeline(0.0, 0.0), "root", 5.0);
        assert_eq!(base.len(), 2);
        let mut song = timeline(0.0, 0.0);
        song.clips[0].takes[0].syllable_analysis =
            Some(crate::pitch_analysis::TakeSyllableAnalysis {
                algorithm: "test".into(),
                algorithm_version: 1,
                source_digest: "d".into(),
                source_spans_sec: vec![
                    crate::pitch_analysis::SyllableSpan {
                        start_sec: 0.0,
                        end_sec: 0.6,
                        midi_note: None,
                    },
                    crate::pitch_analysis::SyllableSpan {
                        start_sec: 0.4,
                        end_sec: 0.9,
                        midi_note: None,
                    },
                ],
                analyzed_window_sec: [0.0, 2.0],
            });
        let normalized = super::super::base::base_blocks_for_root(&song, "root", 5.0);
        assert_eq!(normalized.len(), 2);
        assert_eq!(normalized[0].span.source_start_sec, 0.0);
        assert_eq!(normalized[0].span.source_end_sec, 0.6);
        assert_eq!(normalized[1].span.source_start_sec, 0.6);
        assert_eq!(normalized[1].span.source_end_sec, 0.9);
    }

    #[test]
    fn one_frame_base_remainder_keeps_an_editable_source_interval() {
        let mut timeline = timeline(0.0, 0.0);
        timeline.clips[0].normalize_takes();
        timeline.clips[0].takes[0].syllable_analysis =
            Some(crate::pitch_analysis::TakeSyllableAnalysis {
                algorithm: "test".into(),
                algorithm_version: 1,
                source_digest: "d".into(),
                source_spans_sec: vec![crate::pitch_analysis::SyllableSpan {
                    start_sec: 0.0,
                    end_sec: 0.4,
                    midi_note: Some(64.0),
                }],
                analyzed_window_sec: [0.0, 2.0],
            });
        let edits = ParamBlockEditSet {
            schema_version: 2,
            revision: 1,
            blocks: vec![UserParamBlock {
                id: "cover".into(),
                geometry: BlockGeometry::Root {
                    span: super::super::types::RootSpan {
                        start_sec: 0.0,
                        end_sec: 0.396,
                    },
                },
                deleted: false,
                manual: false,
                value: None,
                transition_out_sec: None,
                created_seq: 1,
                base_cover: None,
            }],
        };
        let view = compose_param_block_view(&timeline, "root", &edits, 5.0).unwrap();
        let remainder = view
            .blocks
            .iter()
            .find(|block| {
                block.kind == ProjectedBlockKind::Base
                    && block.end_frame_exclusive - block.start_frame == 1
            })
            .expect("one-frame base remainder");
        assert!(remainder.members[0].source_end_sec > remainder.members[0].source_start_sec);
        assert_eq!(remainder.source_note, Some(64.0));
    }

    #[test]
    fn full_base_cover_only_suppresses_the_range_it_still_covers() {
        let mut song = timeline(0.0, 0.0);
        song.clips[0].takes[0].syllable_analysis =
            Some(crate::pitch_analysis::TakeSyllableAnalysis {
                algorithm: "test".into(),
                algorithm_version: 1,
                source_digest: "d".into(),
                source_spans_sec: vec![crate::pitch_analysis::SyllableSpan {
                    start_sec: 0.9,
                    end_sec: 1.6,
                    midi_note: None,
                }],
                analyzed_window_sec: [0.0, 2.0],
            });
        let edits = ParamBlockEditSet {
            schema_version: 2,
            revision: 1,
            blocks: vec![UserParamBlock {
                id: "partial".into(),
                geometry: BlockGeometry::Take {
                    members: vec![TakeSpan {
                        clip_id: "clip".into(),
                        take_id: "take".into(),
                        cycle: 0,
                        source_start_sec: 0.5,
                        source_end_sec: 1.2,
                        syllable_id: None,
                        label: BlockLabel::Syllable,
                    }],
                },
                deleted: false,
                manual: false,
                value: None,
                transition_out_sec: None,
                created_seq: 1,
                base_cover: Some(BaseCover {
                    spans: vec![TakeSpan {
                        clip_id: "clip".into(),
                        take_id: "take".into(),
                        cycle: 0,
                        source_start_sec: 0.5,
                        source_end_sec: 1.2,
                        syllable_id: None,
                        label: BlockLabel::Syllable,
                    }],
                    full: true,
                }),
            }],
        };
        let view = compose_param_block_view(&song, "root", &edits, 5.0).unwrap();
        assert!(view
            .blocks
            .iter()
            .any(|block| block.kind == ProjectedBlockKind::Base && block.start_frame == 240));
    }

    #[test]
    fn base_blocks_follow_slip_and_merge_suppresses_base() {
        let song = timeline(0.0, 0.0);
        let base =
            compose_param_block_view(&song, "root", &ParamBlockEditSet::default(), 5.0).unwrap();
        assert_eq!(base.blocks.len(), 2);
        assert!(base
            .blocks
            .iter()
            .all(|block| block.kind == ProjectedBlockKind::Base));

        let merged = compose_param_block_view(&song, "root", &merged_edits(), 5.0).unwrap();
        assert_eq!(merged.blocks.len(), 1);
        // 两个相邻成员合成一个 run：这是几何事实，**不是**可逆编组 —— 它照样可以
        // 切分 / 伸缩，所以 `kind` 必须是 `User`（见 `ProjectedBlockKind` 的文档）。
        assert_eq!(merged.blocks[0].kind, ProjectedBlockKind::User);
        assert_eq!(merged.blocks[0].start_frame, 0);
        assert_eq!(merged.blocks[0].end_frame_exclusive, 200);

        // Alt+Slip：只改源窗口。把 Clip 放在 1.0s，源窗口滑到 0.5s，
        // 合并块成员整体 +0.5s，投影应落在工程 1.0..2.0。
        let slipped = timeline(1.0, 0.5);
        let mut edits = merged_edits();
        if let BlockGeometry::Take { members } = &mut edits.blocks[0].geometry {
            for member in members {
                member.source_start_sec += 0.5;
                member.source_end_sec += 0.5;
            }
        }
        let shifted = compose_param_block_view(&slipped, "root", &edits, 5.0).unwrap();
        assert_eq!(shifted.blocks.len(), 1);
        assert_eq!(shifted.blocks[0].start_frame, 200);
        assert_eq!(shifted.blocks[0].end_frame_exclusive, 400);
    }

    fn crossing_timeline(reverse: bool) -> TimelineState {
        let mut a = serde_json::json!({
            "id":"A","track_id":"root","name":"A","start_sec":0.0,"length_sec":2.0,
            "takes":[{"id":"take-a","source_path":"a.wav","source_start_sec":0.0,
                "source_end_sec":2.0,
                "syllable_analysis":{"algorithm":"test","algorithmVersion":1,"sourceDigest":"a","analyzedWindowSec":[0.0,4.0],
                    "sourceSpansSec":[{"startSec":0.0,"endSec":0.5},{"startSec":0.5,"endSec":1.0}]}}]
        });
        let mut b = serde_json::json!({
            "id":"B","track_id":"root","name":"B","start_sec":0.2,"length_sec":1.0,
            "takes":[{"id":"take-b","source_path":"b.wav","source_start_sec":0.0,
                "source_end_sec":1.0,
                "syllable_analysis":{"algorithm":"test","algorithmVersion":1,"sourceDigest":"b","analyzedWindowSec":[0.0,4.0],
                    "sourceSpansSec":[{"startSec":0.0,"endSec":0.5}]}}]
        });
        let clips = if reverse {
            vec![b.take(), a.take()]
        } else {
            vec![a.take(), b.take()]
        };
        let mut timeline: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"root","name":"Root","order":0}],
            "clips": clips,
            "project_sec":4,"bpm":120
        }))
        .unwrap();
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }
        timeline
    }

    #[test]
    fn crossing_clip_order_is_independent_of_storage_order() {
        let ab = compose_param_block_view(
            &crossing_timeline(false),
            "root",
            &ParamBlockEditSet::default(),
            5.0,
        )
        .unwrap();
        let ba = compose_param_block_view(
            &crossing_timeline(true),
            "root",
            &ParamBlockEditSet::default(),
            5.0,
        )
        .unwrap();
        let ranges = |view: &ParamBlockView| {
            view.blocks
                .iter()
                .map(|block| (block.start_frame, block.end_frame_exclusive))
                .collect::<Vec<_>>()
        };
        assert_eq!(ranges(&ab), vec![(0, 40), (40, 140), (140, 200)]);
        assert_eq!(ranges(&ab), ranges(&ba));
        assert_eq!(
            super::super::layout::layout_key(&crossing_timeline(false), "root", 5.0),
            super::super::layout::layout_key(&crossing_timeline(true), "root", 5.0)
        );
        assert!(validate_view(&ab).is_ok());
    }

    #[test]
    fn deleted_tombstone_hides_base_without_showing_a_block() {
        let song = timeline(0.0, 0.0);
        let edits = ParamBlockEditSet {
            schema_version: 2,
            revision: 1,
            blocks: vec![UserParamBlock {
                id: "tomb".into(),
                geometry: BlockGeometry::Take {
                    members: vec![TakeSpan {
                        clip_id: "clip".into(),
                        take_id: "take".into(),
                        cycle: 0,
                        source_start_sec: 0.0,
                        source_end_sec: 0.5,
                        syllable_id: Some("span:take:0:0".into()),
                        label: BlockLabel::Syllable,
                    }],
                },
                deleted: true,
                manual: false,
                value: None,
                transition_out_sec: None,
                created_seq: 1,
                base_cover: None,
            }],
        };
        let view = compose_param_block_view(&song, "root", &edits, 5.0).unwrap();
        assert_eq!(view.blocks.len(), 1);
        assert_eq!(view.blocks[0].start_frame, 100);
    }

    #[test]
    fn root_block_is_absolute_and_does_not_follow_slip() {
        let song = timeline(0.0, 0.0);
        let edits = ParamBlockEditSet {
            schema_version: 2,
            revision: 1,
            blocks: vec![UserParamBlock {
                id: "root".into(),
                geometry: BlockGeometry::Root {
                    span: crate::param_blocks::types::RootSpan {
                        start_sec: 1.0,
                        end_sec: 2.0,
                    },
                },
                deleted: false,
                manual: true,
                value: Some(60.0),
                transition_out_sec: None,
                created_seq: 1,
                base_cover: None,
            }],
        };
        let view = compose_param_block_view(&song, "root", &edits, 5.0).unwrap();
        let manual = view
            .blocks
            .iter()
            .find(|block| block.kind == ProjectedBlockKind::Manual)
            .unwrap();
        assert_eq!(manual.start_frame, 200);
        assert_eq!(manual.end_frame_exclusive, 400);
    }
}
