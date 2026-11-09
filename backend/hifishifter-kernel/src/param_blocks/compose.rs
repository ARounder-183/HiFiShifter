//! base 分段与用户覆盖层的合成。
//!
//! 职责：抑制、优先级、重叠消解、过渡、视图校验。
//! 关系：`base` 提供 base 块，`projection` 提供投影，`layout` 提供布局键。
//! 维护约束：用户显式编辑 > base 分析；deleted 墓碑只抑制 base，不显示。

use super::base::{base_blocks_for_root, root_analysis_pending};
use super::layout::layout_key;
use super::projection::{project_span, ClipProjection};
use super::types::{
    BlockGeometry, ParamBlockEditSet, ParamBlockView, ProjectedBlockKind, ProjectedMember,
    ProjectedParamBlock, TakeSpan, UserParamBlock,
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
    priority: u8,
    seq: u64,
    z_order: u32,
}

fn interval_width(start: u32, end: u32) -> u32 {
    end.saturating_sub(start)
}

fn member_from(span: &TakeSpan, start: u32, end: u32) -> ProjectedMember {
    ProjectedMember {
        clip_id: span.clip_id.clone(),
        take_id: span.take_id.clone(),
        cycle: span.cycle,
        source_start_sec: span.source_start_sec,
        source_end_sec: span.source_end_sec,
        syllable_id: span.syllable_id.clone(),
        start_frame: start,
        end_frame_exclusive: end,
    }
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
                    kind: if block.grouped_from.is_some() {
                        ProjectedBlockKind::Grouped
                    } else if block.manual {
                        ProjectedBlockKind::Manual
                    } else {
                        ProjectedBlockKind::User
                    },
                    members: Vec::new(),
                    value: block.value,
                    transition_width_sec: block.transition_out_sec,
                    transition_frames: None,
                    rate: 1.0,
                    priority: if block.manual { 4 } else { 3 },
                    seq: block.created_seq,
                    z_order: 0,
                }],
                0,
            )
        }
        BlockGeometry::Take { members } => {
            let mut projected: Vec<(u32, u32, ProjectedMember, f64)> = Vec::new();
            let mut orphan = 0usize;
            for member in members {
                let Some(clip) = timeline.clips.iter().find(|clip| clip.id == member.clip_id)
                else {
                    orphan += 1;
                    continue;
                };
                let Some(take) = clip.take(&member.take_id) else {
                    orphan += 1;
                    continue;
                };
                let projection = ClipProjection::from_clip_take(clip, take);
                let mut matched = false;
                for interval in project_span(&projection, member, frame_period_ms) {
                    if interval.cycle != member.cycle {
                        continue;
                    }
                    matched = true;
                    projected.push((
                        interval.start_frame,
                        interval.end_frame_exclusive,
                        member_from(member, interval.start_frame, interval.end_frame_exclusive),
                        projection.effective_rate,
                    ));
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
                    Some(run) if start <= run.1.saturating_add(1) => {
                        run.1 = run.1.max(end);
                        run.2.push(member);
                    }
                    _ => runs.push((start, end, vec![member], rate)),
                }
            }
            let count = runs.len();
            let mut out = Vec::new();
            for (index, (start, end, run_members, rate)) in runs.into_iter().enumerate() {
                let id = if count == 1 {
                    block.id.clone()
                } else {
                    format!("{}:{}", block.id, index)
                };
                out.push(Candidate {
                    id,
                    user_block_id: Some(block.id.clone()),
                    start,
                    end,
                    kind: if block.grouped_from.is_some() {
                        ProjectedBlockKind::Grouped
                    } else if block.manual {
                        ProjectedBlockKind::Manual
                    } else {
                        ProjectedBlockKind::User
                    },
                    members: run_members,
                    value: block.value,
                    transition_width_sec: block.transition_out_sec,
                    transition_frames: None,
                    rate,
                    priority: if block.manual { 4 } else { 3 },
                    seq: block.created_seq,
                    z_order: 0,
                });
            }
            (out, orphan)
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
    let overlap = base
        .source_end_sec
        .min(member.source_end_sec)
        .max(base.source_start_sec.max(member.source_start_sec));
    let width = (base.source_end_sec - base.source_start_sec)
        .min(member.source_end_sec - member.source_start_sec);
    overlap > 0.0 && width > 1e-9 && overlap / width >= ANCHOR_OVERLAP_RATIO
}

fn suppressed_by_user(base: &TakeSpan, edits: &ParamBlockEditSet) -> bool {
    edits.blocks.iter().any(|block| {
        block.members().iter().any(|member| {
            member.clip_id == base.clip_id
                && member.take_id == base.take_id
                && member.cycle == base.cycle
                && member.overlaps(base)
                && anchor_matches(base, member)
        })
    })
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
        if suppressed_by_user(&base.span, edits) {
            continue;
        }
        let interval = base.projected[0];
        candidates.push(Candidate {
            id: base.id.clone(),
            user_block_id: None,
            start: interval.start_frame,
            end: interval.end_frame_exclusive,
            kind: ProjectedBlockKind::Base,
            members: vec![member_from(
                &base.span,
                interval.start_frame,
                interval.end_frame_exclusive,
            )],
            value: None,
            transition_width_sec: None,
            transition_frames: None,
            rate: base.rate,
            priority: 1,
            seq: 0,
            z_order: base.z_order,
        });
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
            let mut piece = candidate.clone();
            piece.start = start;
            piece.end = end;
            if piece_count > 1 || start != candidate.start || end != candidate.end {
                piece.id = format!("{}:{}:{}", candidate.id, start, end);
                if piece_count > 1 {
                    piece.id = format!("{}:{}", piece.id, index);
                }
            }
            for member in &mut piece.members {
                member.start_frame = member.start_frame.max(start);
                member.end_frame_exclusive = member.end_frame_exclusive.min(end);
            }
            piece.members.retain(|member| {
                member.end_frame_exclusive > member.start_frame
                    && member.start_frame >= start
                    && member.end_frame_exclusive <= end
            });
            accepted.push(piece);
        }
    }

    accepted.sort_by_key(|candidate| (candidate.start, candidate.end, candidate.id.clone()));

    // 过渡只作用于相邻块，宽度受本块与右块限制。
    for index in 0..accepted.len().saturating_sub(1) {
        let (left, right) = accepted.split_at_mut(index + 1);
        let prev = &mut left[index];
        let next = &right[0];
        if prev.end != next.start {
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
                    "syllable_analysis":{"algorithm":"test","algorithmVersion":1,"sourceDigest":"d",
                        "sourceSpansSec":[{"startSec":0.0,"endSec":0.5},{"startSec":0.5,"endSec":1.0}]}}]
            }],
            "project_sec":4,"bpm":120
        })).unwrap();
        timeline.clips[0].normalize_takes();
        timeline
    }

    fn merged_edits() -> ParamBlockEditSet {
        ParamBlockEditSet {
            schema_version: 1,
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
                grouped_from: None,
            }],
        }
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

    #[test]
    fn deleted_tombstone_hides_base_without_showing_a_block() {
        let song = timeline(0.0, 0.0);
        let edits = ParamBlockEditSet {
            schema_version: 1,
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
                grouped_from: None,
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
            schema_version: 1,
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
                grouped_from: None,
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
