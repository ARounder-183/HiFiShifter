//! Apply a whole Clip geometry transaction to root intervals and the GAME baseline.
//! Sources always come from the same immutable snapshot (including swaps/copies).
use super::{ParamLineBlock, ParamLineBlockSet};
use crate::editor::param_geometry::{
    ClipGeometry, GeometryDelta, GeometryFrame, GeometryKey, GeometryOp,
};
use std::collections::BTreeMap;

fn piece(block: &ParamLineBlock, start: u32, end: u32, tag: &str) -> ParamLineBlock {
    let mut result = block.clone();
    result.start_frame = start;
    result.end_frame_exclusive = end;
    if start != block.start_frame || end != block.end_frame_exclusive || !tag.is_empty() {
        let identity = format!("{}|{start}|{end}|{tag}", block.id);
        result.id = format!("block:{}", blake3::hash(identity.as_bytes()).to_hex());
    }
    if end != block.end_frame_exclusive {
        result.transition_frames = Some(0);
    }
    result
}

fn frame(sec: f64, fp: f64) -> u32 {
    (sec.max(0.0) * 1000.0 / fp).round() as u32
}

fn erase(blocks: Vec<ParamLineBlock>, start: u32, end: u32) -> Vec<ParamLineBlock> {
    blocks
        .into_iter()
        .flat_map(|block| {
            if end <= block.start_frame || start >= block.end_frame_exclusive {
                return vec![block];
            }
            let mut parts = Vec::new();
            if block.start_frame < start {
                parts.push(piece(&block, block.start_frame, start, ""));
            }
            if block.end_frame_exclusive > end {
                parts.push(piece(&block, end, block.end_frame_exclusive, ""));
            }
            parts
        })
        .collect()
}

/// Crop only confirmed shortened windows. A split or an overlapping Clip protects
/// its occupied time; incomplete ARA inventories can never remove intervals.
pub fn crop_windows(
    old: &BTreeMap<GeometryKey, ClipGeometry>,
    current: &GeometryFrame,
) -> Vec<(String, f64, f64)> {
    if !current.complete {
        return Vec::new();
    }
    let mut result = Vec::new();
    for clip in &current.clips {
        let Some(previous) = old.get(&clip.key) else {
            continue;
        };
        if previous.root != clip.root
            || previous.source_id != clip.source_id
            || clip.project_duration >= previous.project_duration - 1e-9
            || (clip.rate - previous.rate).abs() > 1e-9
        {
            continue;
        }
        let mut remaining = vec![(previous.project_start, previous.project_end())];
        for coverage in current
            .clips
            .iter()
            .filter(|item| item.root == previous.root)
        {
            remaining = remaining
                .into_iter()
                .flat_map(|(start, end)| {
                    if coverage.project_end() <= start || coverage.project_start >= end {
                        return vec![(start, end)];
                    }
                    let mut parts = Vec::new();
                    if start < coverage.project_start {
                        parts.push((start, coverage.project_start));
                    }
                    if end > coverage.project_end() {
                        parts.push((coverage.project_end(), end));
                    }
                    parts
                })
                .collect();
        }
        result.extend(
            remaining
                .into_iter()
                .map(|(start, end)| (previous.root.clone(), start, end)),
        );
    }
    result
}

pub fn apply_geometry(
    layouts: &mut BTreeMap<String, ParamLineBlockSet>,
    deltas: &[GeometryDelta],
    crops: &[(String, f64, f64)],
) -> Result<bool, String> {
    let active: Vec<_> = deltas
        .iter()
        .filter(|delta| {
            delta.old.is_valid()
                && delta.new.is_valid()
                && (delta.old_root != delta.new_root
                    || delta.old != delta.new
                    || matches!(delta.op, GeometryOp::Copy { .. }))
                && (delta.op.is_move_or_stretch() || matches!(delta.op, GeometryOp::Copy { .. }))
        })
        .collect();
    if active.is_empty() && crops.is_empty() {
        return Ok(false);
    }
    let basis = layouts.clone();
    let mut candidate = basis.clone();
    for delta in &active {
        if let Some(source) = basis.get(&delta.old_root) {
            candidate
                .entry(delta.new_root.clone())
                .or_insert_with(|| ParamLineBlockSet {
                    frame_period_ms: source.frame_period_ms,
                    ..Default::default()
                });
        }
    }
    for baseline in [false, true] {
        let mut kept = BTreeMap::new();
        let mut moved = Vec::new();
        for (root, layout) in &basis {
            let fp = layout.frame_period_ms;
            let blocks = if baseline {
                &layout.game_baseline.segments
            } else {
                &layout.blocks
            };
            let source_deltas: Vec<_> = active
                .iter()
                .filter(|d| d.old_root == *root && d.op.is_move_or_stretch())
                .collect();
            let mut output = Vec::new();
            for block in blocks {
                let mut cuts = vec![block.start_frame, block.end_frame_exclusive];
                for delta in &source_deltas {
                    cuts.extend(
                        [
                            frame(delta.old.start_sec, fp),
                            frame(delta.old.end_sec(), fp),
                        ]
                        .into_iter()
                        .filter(|cut| *cut > block.start_frame && *cut < block.end_frame_exclusive),
                    );
                }
                cuts.sort_unstable();
                cuts.dedup();
                for span in cuts.windows(2) {
                    let target = source_deltas
                        .iter()
                        .filter(|d| {
                            span[0] >= frame(d.old.start_sec, fp)
                                && span[1] <= frame(d.old.end_sec(), fp)
                        })
                        .max_by_key(|d| (d.priority, d.order));
                    if let Some(delta) = target {
                        moved.push((**delta, piece(block, span[0], span[1], ""), fp));
                    } else {
                        output.push(piece(block, span[0], span[1], ""));
                    }
                }
            }
            kept.insert(root.clone(), output);
        }
        // Clear destinations after extracting every source, so swaps never cascade.
        for delta in &active {
            if let Some(layout) = candidate.get(&delta.new_root) {
                let blocks = kept.remove(&delta.new_root).unwrap_or_default();
                kept.insert(
                    delta.new_root.clone(),
                    erase(
                        blocks,
                        frame(delta.new.start_sec, layout.frame_period_ms),
                        frame(delta.new.end_sec(), layout.frame_period_ms),
                    ),
                );
            }
            if matches!(delta.op, GeometryOp::Copy { .. }) {
                if let Some(layout) = basis.get(&delta.old_root) {
                    let blocks = if baseline {
                        &layout.game_baseline.segments
                    } else {
                        &layout.blocks
                    };
                    let start = frame(delta.old.start_sec, layout.frame_period_ms);
                    let end = frame(delta.old.end_sec(), layout.frame_period_ms);
                    for block in blocks {
                        let a = start.max(block.start_frame);
                        let b = end.min(block.end_frame_exclusive);
                        if b > a {
                            moved.push((
                                *delta,
                                piece(block, a, b, "copy"),
                                layout.frame_period_ms,
                            ));
                        }
                    }
                }
            }
        }
        moved.sort_by_key(|(delta, _, _)| (delta.priority, delta.order));
        for (delta, block, source_fp) in moved {
            let Some(layout) = candidate.get(&delta.new_root) else {
                continue;
            };
            let scale = delta.new.duration_sec / delta.old.duration_sec;
            let map = |f: u32| {
                frame(
                    delta.new.start_sec
                        + (f as f64 * source_fp / 1000.0 - delta.old.start_sec) * scale,
                    layout.frame_period_ms,
                )
            };
            let mut mapped = block.clone();
            mapped.start_frame = map(block.start_frame);
            mapped.end_frame_exclusive = map(block.end_frame_exclusive);
            mapped.transition_frames = block.transition_frames.map(|width| {
                ((width as f64 * source_fp / layout.frame_period_ms) * scale).round() as u32
            });
            if mapped.end_frame_exclusive <= mapped.start_frame {
                continue;
            }
            if matches!(delta.op, GeometryOp::Copy { .. }) || delta.old_root != delta.new_root {
                mapped = piece(
                    &mapped,
                    mapped.start_frame,
                    mapped.end_frame_exclusive,
                    delta.key.as_str(),
                );
            }
            let blocks = kept.remove(&delta.new_root).unwrap_or_default();
            let mut blocks = erase(blocks, mapped.start_frame, mapped.end_frame_exclusive);
            blocks.push(mapped);
            kept.insert(delta.new_root.clone(), blocks);
        }
        for (root, start, end) in crops {
            if let Some(layout) = candidate.get(root) {
                let blocks = kept.remove(root).unwrap_or_default();
                kept.insert(
                    root.clone(),
                    erase(
                        blocks,
                        frame(*start, layout.frame_period_ms),
                        frame(*end, layout.frame_period_ms),
                    ),
                );
            }
        }
        for (root, layout) in &mut candidate {
            let mut blocks = kept.remove(root).unwrap_or_default();
            blocks.sort_by_key(|b| (b.start_frame, b.end_frame_exclusive));
            if baseline {
                layout.game_baseline.segments = blocks;
            } else {
                layout.blocks = blocks;
            }
        }
    }
    let mut changed = false;
    for (root, layout) in &mut candidate {
        let previous = basis.get(root);
        if previous.map(|old| &old.blocks) != Some(&layout.blocks)
            || previous.map(|old| &old.game_baseline.segments)
                != Some(&layout.game_baseline.segments)
        {
            layout.revision = layout.revision.saturating_add(1);
            layout.game_baseline.revision = layout.game_baseline.revision.saturating_add(1);
            changed = true;
        }
        layout.validate()?;
    }
    if changed {
        *layouts = candidate;
    }
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::param_geometry::GeometryWindow;
    fn block(id: &str, a: u32, b: u32) -> ParamLineBlock {
        ParamLineBlock {
            id: id.into(),
            start_frame: a,
            end_frame_exclusive: b,
            transition_frames: None,
        }
    }
    fn layout(blocks: Vec<ParamLineBlock>) -> ParamLineBlockSet {
        ParamLineBlockSet {
            blocks: blocks.clone(),
            game_baseline: super::super::GameBlockBaseline {
                revision: 0,
                segments: blocks,
            },
            ..Default::default()
        }
    }
    fn delta(a: f64, b: f64, c: f64, d: f64, order: u64) -> GeometryDelta {
        let old = GeometryWindow::new(a, b - a);
        let new = GeometryWindow::new(c, d - c);
        GeometryDelta {
            key: GeometryKey::clip(&format!("c{order}")),
            old_root: "root".into(),
            new_root: "root".into(),
            old,
            new,
            op: GeometryOp::Stretch { old, new },
            priority: 1,
            order,
        }
    }
    fn spans(layouts: &BTreeMap<String, ParamLineBlockSet>) -> Vec<(u32, u32)> {
        layouts["root"]
            .blocks
            .iter()
            .map(|b| (b.start_frame, b.end_frame_exclusive))
            .collect()
    }
    #[test]
    fn param_line_geometry_swaps_from_one_basis_and_maps_baseline() {
        let mut layouts = BTreeMap::from([(
            "root".into(),
            layout(vec![block("a", 0, 200), block("b", 400, 600)]),
        )]);
        apply_geometry(
            &mut layouts,
            &[delta(0., 1., 2., 3., 0), delta(2., 3., 0., 1., 1)],
            &[],
        )
        .unwrap();
        assert_eq!(spans(&layouts), vec![(0, 200), (400, 600)]);
        assert_eq!(layouts["root"].blocks[0].id, "b");
        assert_eq!(
            layouts["root"].blocks,
            layouts["root"].game_baseline.segments
        );
    }
    #[test]
    fn param_line_geometry_moves_only_overlapping_portion_and_crops_both_lists() {
        let mut layouts = BTreeMap::from([("root".into(), layout(vec![block("whole", 0, 600)]))]);
        apply_geometry(&mut layouts, &[delta(1., 2., 4., 6., 0)], &[]).unwrap();
        assert_eq!(spans(&layouts), vec![(0, 200), (400, 600), (800, 1200)]);
        apply_geometry(&mut layouts, &[], &[("root".into(), 4.5, 5.5)]).unwrap();
        assert_eq!(
            spans(&layouts),
            vec![(0, 200), (400, 600), (800, 900), (1100, 1200)]
        );
        assert_eq!(
            layouts["root"].blocks,
            layouts["root"].game_baseline.segments
        );
    }
    #[test]
    fn param_line_geometry_copy_preserves_source_and_has_distinct_ids() {
        let mut copy = delta(0., 1., 2., 3., 0);
        copy.op = GeometryOp::Copy {
            source: copy.old,
            target: copy.new,
        };
        let mut layouts = BTreeMap::from([("root".into(), layout(vec![block("a", 0, 200)]))]);
        apply_geometry(&mut layouts, &[copy], &[]).unwrap();
        assert_eq!(spans(&layouts), vec![(0, 200), (400, 600)]);
        assert_ne!(layouts["root"].blocks[0].id, layouts["root"].blocks[1].id);
    }
    fn clip(key: &str, start: f64, len: f64) -> ClipGeometry {
        ClipGeometry {
            key: GeometryKey::clip(key),
            root: "root".into(),
            project_start: start,
            project_duration: len,
            source_start: start,
            source_duration: len,
            rate: 1.,
            reversed: false,
            loop_enabled: false,
            source_id: "source".into(),
            last_seen_epoch: 0,
        }
    }
    #[test]
    fn param_line_crop_protects_split_and_incomplete_inventory() {
        let old = clip("a", 0., 3.);
        let entries = BTreeMap::from([(old.key.clone(), old)]);
        let split = GeometryFrame::complete(vec![clip("a", 0., 1.), clip("b", 1., 2.)]);
        assert!(crop_windows(&entries, &split).is_empty());
        let trim = GeometryFrame::complete(vec![clip("a", 0., 1.)]);
        assert_eq!(crop_windows(&entries, &trim), vec![("root".into(), 1., 3.)]);
        assert!(crop_windows(&entries, &GeometryFrame::partial(trim.clips)).is_empty());
    }
}
