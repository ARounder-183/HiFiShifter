//! 宿主几何变化 → 参数 run 放置变换。
//!
//! 锁定只影响**一次几何事务**：关闭时只更新几何日志；开启时只改
//! `placement` 与边界切分，`values` 逐字节不动。裁剪/分割/倒放/Loop 不产生
//! Move/Stretch，因此权威哈希必须保持不变。

use super::param_authority::{ParamAuthority, PlacedRun, RootParamAuthority, RunPlacement};
use crate::state::Clip;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const EPS: f64 = 1e-9;

/// 一条 Clip 在项目时间轴上的几何事实。身份键优先 item GUID，ARA-only 用稳定指纹。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ClipGeometry {
    pub key: String,
    pub root: String,
    pub project_start: f64,
    pub project_duration: f64,
    pub source_start: f64,
    pub source_duration: f64,
    pub rate: f64,
    pub reversed: bool,
    pub loop_enabled: bool,
    pub source_id: String,
    #[serde(default)]
    pub last_seen_epoch: u64,
}

impl ClipGeometry {
    pub fn project_end(&self) -> f64 {
        self.project_start + self.project_duration
    }

    pub fn source_end(&self) -> f64 {
        self.source_start + self.source_duration
    }

    pub fn is_well_formed(&self) -> bool {
        self.project_start.is_finite()
            && self.project_duration.is_finite()
            && self.project_duration > 0.0
            && self.source_start.is_finite()
            && self.source_duration.is_finite()
            && self.source_duration > 0.0
            && self.rate.is_finite()
            && self.rate > 0.0
            && !self.source_id.is_empty()
    }

    pub fn from_clip(clip: &Clip, root: &str, key: String) -> Self {
        let source_start = clip.source_start_sec.max(0.0);
        let source_end = clip
            .source_end_sec
            .max(source_start + clip.length_sec.max(0.0) * clip.playback_rate.abs() as f64);
        let source_duration = (source_end - source_start).max(0.0);
        let rate = if clip.length_sec > 0.0 && source_duration > 0.0 {
            source_duration / clip.length_sec
        } else {
            1.0
        };
        Self {
            key,
            root: root.to_owned(),
            project_start: clip.start_sec,
            project_duration: clip.length_sec,
            source_start,
            source_duration,
            rate,
            reversed: clip.reversed,
            loop_enabled: clip.loop_enabled,
            source_id: clip.source_path.clone().unwrap_or_default(),
            last_seen_epoch: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeometryWindow {
    pub start_sec: f64,
    pub duration_sec: f64,
}

impl GeometryWindow {
    pub fn new(start_sec: f64, duration_sec: f64) -> Self {
        Self {
            start_sec,
            duration_sec,
        }
    }
    pub fn end_sec(self) -> f64 {
        self.start_sec + self.duration_sec
    }
    pub fn is_valid(self) -> bool {
        self.start_sec.is_finite() && self.duration_sec.is_finite() && self.duration_sec > 0.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GeometryOp {
    Replace,
    Move {
        old: GeometryWindow,
        new: GeometryWindow,
    },
    Stretch {
        old: GeometryWindow,
        new: GeometryWindow,
    },
    TrimOrNone,
    /// 原生/宿主复制：把源窗口搬一份到目标窗口。
    Copy {
        source: GeometryWindow,
        target: GeometryWindow,
    },
}

impl GeometryOp {
    pub fn is_move_or_stretch(&self) -> bool {
        matches!(self, Self::Move { .. } | Self::Stretch { .. })
    }
}

/// 与独立 App 对齐的分类：
/// * rate 与工程时长同时变化 => Stretch；
/// * 源窗不变、工程起点变化 => Move；
/// * 其余（裁剪/滑源/倒放/Loop/Section/换素材）=> 不搬参数。
pub fn classify(old: &ClipGeometry, new: &ClipGeometry) -> GeometryOp {
    if old.source_id != new.source_id {
        return GeometryOp::Replace;
    }
    if !old.is_well_formed() || !new.is_well_formed() {
        return GeometryOp::TrimOrNone;
    }
    let rate_changed = (new.rate - old.rate).abs() > EPS;
    let source_start_changed = (new.source_start - old.source_start).abs() > EPS;
    let source_duration_changed = (new.source_duration - old.source_duration).abs() > EPS;
    if rate_changed && (new.project_duration - old.project_duration).abs() > EPS {
        return GeometryOp::Stretch {
            old: GeometryWindow::new(old.project_start, old.project_duration),
            new: GeometryWindow::new(new.project_start, new.project_duration),
        };
    }
    if !source_start_changed
        && !source_duration_changed
        && (old.root != new.root || (new.project_start - old.project_start).abs() > EPS)
    {
        return GeometryOp::Move {
            old: GeometryWindow::new(old.project_start, old.project_duration),
            new: GeometryWindow::new(new.project_start, new.project_duration),
        };
    }
    GeometryOp::TrimOrNone
}

/// 一个已确认的几何变化。`priority` 用于重叠窗口的确定性裁决。
#[derive(Clone, Debug, PartialEq)]
pub struct GeometryDelta {
    pub key: String,
    pub old_root: String,
    pub new_root: String,
    pub old: GeometryWindow,
    pub new: GeometryWindow,
    pub op: GeometryOp,
    pub priority: u64,
    pub order: u64,
}

impl GeometryDelta {
    pub fn from_geometry(old: &ClipGeometry, new: &ClipGeometry, order: u64) -> Option<Self> {
        let op = classify(old, new);
        if !op.is_move_or_stretch() {
            return None;
        }
        let (old_window, new_window) = match &op {
            GeometryOp::Move { old, new } | GeometryOp::Stretch { old, new } => (*old, *new),
            _ => return None,
        };
        if !old_window.is_valid() || !new_window.is_valid() {
            return None;
        }
        Some(Self {
            key: new.key.clone(),
            old_root: old.root.clone(),
            new_root: new.root.clone(),
            old: old_window,
            new: new_window,
            op,
            priority: 0,
            order,
        })
    }

    fn contains_old(&self, sec: f64) -> bool {
        // 曲线最后一个采样点可以精确落在 clip 结束边界上；它仍属于这条 run，
        // 必须跟着旧窗口一起变换。若用半开区间，末值会留在原位形成“残影”。
        sec >= self.old.start_sec - EPS && sec <= self.old.end_sec() + EPS
    }

    fn rank(&self) -> (u64, u64) {
        (self.priority, self.order)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TransformReport {
    pub affected_runs: usize,
    pub split_runs: usize,
    pub cleared_runs: usize,
    pub dropped_frames: usize,
    pub unknown: usize,
    pub applied_deltas: usize,
}

impl TransformReport {
    pub fn is_empty(&self) -> bool {
        self.affected_runs == 0 && self.split_runs == 0 && self.cleared_runs == 0
    }
}

/// 把全部 Move/Stretch 当作一个原子事务：要么全部应用，要么权威保持原样。
pub fn apply_geometry_transaction(
    authority: &mut ParamAuthority,
    deltas: &[GeometryDelta],
) -> Result<TransformReport, String> {
    let active = deltas
        .iter()
        .filter(|delta| {
            delta.op.is_move_or_stretch()
                && delta.old.is_valid()
                && delta.new.is_valid()
                && !delta.old_root.is_empty()
                && !delta.new_root.is_empty()
        })
        .collect::<Vec<_>>();
    let copies = deltas
        .iter()
        .filter(|delta| matches!(delta.op, GeometryOp::Copy { .. }))
        .collect::<Vec<_>>();
    if active.is_empty() && copies.is_empty() {
        return Ok(TransformReport::default());
    }
    let mut report = TransformReport::default();
    let mut kept_by_root = BTreeMap::<String, RootParamAuthority>::new();
    let mut moved_by_root = BTreeMap::<String, RootParamAuthority>::new();
    let roots = authority
        .roots
        .keys()
        .cloned()
        .chain(
            active
                .iter()
                .flat_map(|d| [d.old_root.clone(), d.new_root.clone()]),
        )
        .collect::<BTreeSet<_>>();
    let mut seq_cursor = authority.next_seq.max(1);

    for root in roots {
        let source = authority.roots.get(&root).cloned().unwrap_or_default();
        let root_deltas = active
            .iter()
            .copied()
            .filter(|delta| delta.old_root == root)
            .collect::<Vec<_>>();
        for (key, runs) in &source.runs {
            for run in runs {
                let (segments, split) = partition_run(run, &root_deltas);
                report.split_runs += split;
                for (target, segment) in segments {
                    if let Some(delta) = target {
                        let Some(transformed) = transform_segment(segment, delta, &mut seq_cursor)
                        else {
                            report.dropped_frames += 1;
                            continue;
                        };
                        report.affected_runs += 1;
                        moved_by_root
                            .entry(delta.new_root.clone())
                            .or_default()
                            .runs
                            .entry(key.clone())
                            .or_default()
                            .push(transformed);
                    } else {
                        kept_by_root
                            .entry(root.clone())
                            .or_default()
                            .runs
                            .entry(key.clone())
                            .or_default()
                            .push(segment);
                    }
                }
            }
        }
    }

    // App 的批量语义：目标窗口内的旧编辑先被清除；搬过去的 run 最后插入。
    for delta in &active {
        if let Some(root) = kept_by_root.get_mut(&delta.new_root) {
            for runs in root.runs.values_mut() {
                let mut kept = Vec::new();
                for run in runs.drain(..) {
                    let before = kept.len();
                    clear_run_window(run, delta.new.start_sec, delta.new.end_sec(), &mut kept);
                    if kept.len() == before {
                        // 这条 run 完全落在目标窗口里。
                        report.cleared_runs += 1;
                    } else if kept.len() > before + 1 {
                        report.cleared_runs += 1;
                    }
                }
                *runs = kept;
            }
            root.runs.retain(|_, runs| !runs.is_empty());
        }
    }
    for (root, moved) in moved_by_root {
        let target = kept_by_root.entry(root).or_default();
        for (key, mut runs) in moved.runs {
            target.runs.entry(key).or_default().append(&mut runs);
        }
    }

    for root in kept_by_root.values_mut() {
        for runs in root.runs.values_mut() {
            runs.sort_by_key(|run| (run.placement.start_sec.to_bits(), run.seq, run.id));
        }
    }
    let mut candidate = ParamAuthority {
        frame_period_ms: authority.frame_period_ms,
        roots: kept_by_root,
        revision: authority.revision,
        next_seq: seq_cursor.max(authority.next_seq),
    };
    candidate.roots.retain(|_, root| !root.is_empty());
    let sanitize = candidate.sanitize();
    if sanitize.dropped_runs > 0 {
        return Err(format!(
            "geometry transaction produced {} invalid run(s); the authority was left unchanged",
            sanitize.dropped_runs
        ));
    }
    // 复制必须在移动后的目标上执行，但源数据取事务开始前的快照。
    if !copies.is_empty() {
        let source_snapshot = authority.clone();
        for delta in &copies {
            let GeometryOp::Copy { source, target } = delta.op else {
                continue;
            };
            report.affected_runs += copy_runs_from_window(
                &source_snapshot,
                &mut candidate,
                &delta.old_root,
                source,
                &delta.new_root,
                target,
            );
            report.applied_deltas += 1;
        }
    }
    candidate.revision = authority.revision.saturating_add(1);
    candidate.next_seq = candidate.next_seq.max(authority.next_seq);
    *authority = candidate;
    report.applied_deltas = active.len();
    Ok(report)
}

fn copy_runs_from_window(
    source: &ParamAuthority,
    target: &mut ParamAuthority,
    source_root: &str,
    source_window: GeometryWindow,
    target_root: &str,
    target_window: GeometryWindow,
) -> usize {
    let mut copied = RootParamAuthority::default();
    let mut count = 0usize;
    let Some(root) = source.root(source_root) else {
        return 0;
    };
    let scale = target_window.duration_sec / source_window.duration_sec;
    for (key, runs) in &root.runs {
        for run in runs {
            let Some(segment) =
                run.slice_by_project_sec(source_window.start_sec, source_window.end_sec())
            else {
                continue;
            };
            let seq = target.next_seq.max(1);
            target.next_seq = seq.saturating_add(1);
            copied
                .runs
                .entry(key.clone())
                .or_default()
                .push(PlacedRun::new(
                    seq,
                    seq,
                    segment.values,
                    RunPlacement {
                        start_sec: target_window.start_sec
                            + (segment.placement.start_sec - source_window.start_sec) * scale,
                        seconds_per_frame: segment.placement.seconds_per_frame * scale,
                    },
                ));
            count += 1;
        }
    }
    let target_root_authority = target.root_mut(target_root);
    for runs in target_root_authority.runs.values_mut() {
        let mut kept = Vec::new();
        for run in runs.drain(..) {
            clear_run_window(
                run,
                target_window.start_sec,
                target_window.end_sec(),
                &mut kept,
            );
        }
        *runs = kept;
    }
    for (key, mut runs) in copied.runs {
        target_root_authority
            .runs
            .entry(key)
            .or_default()
            .append(&mut runs);
    }
    target_root_authority
        .runs
        .retain(|_, runs| !runs.is_empty());
    target.roots.retain(|_, root| !root.is_empty());
    count
}

/// 复制/粘贴：把源工程窗口内的 run 映射到目标窗口，目标范围先清。
pub fn copy_geometry_window(
    authority: &mut ParamAuthority,
    source_root: &str,
    source_window: GeometryWindow,
    target_root: &str,
    target_window: GeometryWindow,
) -> Result<TransformReport, String> {
    if !source_window.is_valid() || !target_window.is_valid() {
        return Err("invalid copy window".into());
    }
    if (source_window.duration_sec - target_window.duration_sec).abs() <= EPS {
        // 纯位置复制也走同一仿射公式。
    }
    let source = authority.root(source_root).cloned().unwrap_or_default();
    let mut copied = RootParamAuthority::default();
    let mut report = TransformReport {
        applied_deltas: 1,
        ..Default::default()
    };
    let scale = target_window.duration_sec / source_window.duration_sec;
    for (key, runs) in &source.runs {
        for run in runs {
            let Some(segment) =
                run.slice_by_project_sec(source_window.start_sec, source_window.end_sec())
            else {
                continue;
            };
            let seq = authority.next_seq.max(1);
            authority.next_seq = seq.saturating_add(1);
            let new_start = target_window.start_sec
                + (segment.placement.start_sec - source_window.start_sec) * scale;
            copied
                .runs
                .entry(key.clone())
                .or_default()
                .push(PlacedRun::new(
                    seq,
                    seq,
                    segment.values,
                    RunPlacement {
                        start_sec: new_start,
                        seconds_per_frame: segment.placement.seconds_per_frame * scale,
                    },
                ));
            report.affected_runs += 1;
        }
    }
    let target = authority.root_mut(target_root);
    for runs in target.runs.values_mut() {
        let mut kept = Vec::new();
        for run in runs.drain(..) {
            clear_run_window(
                run,
                target_window.start_sec,
                target_window.end_sec(),
                &mut kept,
            );
        }
        *runs = kept;
        report.cleared_runs += 1;
    }
    for (key, mut runs) in copied.runs {
        target.runs.entry(key).or_default().append(&mut runs);
    }
    target.runs.retain(|_, runs| !runs.is_empty());
    authority.roots.retain(|_, root| !root.is_empty());
    authority.revision = authority.revision.saturating_add(1);
    Ok(report)
}

/// 运行时几何日志：不存档，可由宿主清单重建。只比较“旧日志 → 当前事实”。
#[derive(Clone, Debug, Default)]
pub struct GeometryJournal {
    entries: BTreeMap<String, ClipGeometry>,
    epoch: u64,
    unknown_keys: BTreeSet<String>,
}

impl GeometryJournal {
    pub fn entries(&self) -> &BTreeMap<String, ClipGeometry> {
        &self.entries
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn unknown_keys(&self) -> &BTreeSet<String> {
        &self.unknown_keys
    }

    /// 与当前清单比较，产出 Move/Stretch；新 key 不复制、消失 key 不删除权威。
    pub fn diff(&mut self, current: &[ClipGeometry]) -> Vec<GeometryDelta> {
        self.epoch = self.epoch.saturating_add(1);
        let mut seen = BTreeSet::new();
        let mut deltas = Vec::new();
        for (order, geometry) in current.iter().enumerate() {
            if geometry.key.is_empty() || !geometry.is_well_formed() {
                self.unknown_keys.insert(geometry.key.clone());
                continue;
            }
            seen.insert(geometry.key.clone());
            match self.entries.get(&geometry.key) {
                Some(old) => {
                    if let Some(mut delta) =
                        GeometryDelta::from_geometry(old, geometry, order as u64)
                    {
                        delta.priority = if old.root == geometry.root { 1 } else { 2 };
                        deltas.push(delta);
                    }
                }
                None => {
                    // 新身份：只接受**内容窗口完全一致**的唯一原 Clip 作为复制源。
                    // 分割子段源窗是父段子集，不会命中；身份歧义则宁可少复制。
                    let candidates = self
                        .entries
                        .values()
                        .filter(|old| {
                            old.source_id == geometry.source_id
                                && (old.source_start - geometry.source_start).abs() <= 1e-6
                                && (old.source_duration - geometry.source_duration).abs() <= 1e-6
                                && (old.rate - geometry.rate).abs() <= 1e-9
                                && (old.project_start - geometry.project_start).abs() > 1e-6
                        })
                        .collect::<Vec<_>>();
                    if let [old] = candidates.as_slice() {
                        deltas.push(GeometryDelta {
                            key: geometry.key.clone(),
                            old_root: old.root.clone(),
                            new_root: geometry.root.clone(),
                            old: GeometryWindow::new(old.project_start, old.project_duration),
                            new: GeometryWindow::new(
                                geometry.project_start,
                                geometry.project_duration,
                            ),
                            op: GeometryOp::Copy {
                                source: GeometryWindow::new(
                                    old.project_start,
                                    old.project_duration,
                                ),
                                target: GeometryWindow::new(
                                    geometry.project_start,
                                    geometry.project_duration,
                                ),
                            },
                            priority: 0,
                            order: order as u64,
                        });
                    }
                }
            }
        }
        self.entries.retain(|key, _| seen.contains(key));
        deltas
    }

    /// 只有事务成功后调用：把当前几何写入日志。
    pub fn commit(&mut self, current: &[ClipGeometry], epoch: u64) {
        self.entries = current
            .iter()
            .filter(|geometry| !geometry.key.is_empty() && geometry.is_well_formed())
            .map(|geometry| {
                let mut geometry = geometry.clone();
                geometry.last_seen_epoch = epoch;
                (geometry.key.clone(), geometry)
            })
            .collect();
        self.epoch = epoch.max(self.epoch);
        self.unknown_keys.clear();
    }

    /// 锁定关闭/事务失败时只更新日志（I5）。
    pub fn observe_only(&mut self, current: &[ClipGeometry]) {
        let epoch = self.epoch.saturating_add(1);
        self.commit(current, epoch);
    }
}

fn partition_run<'a>(
    run: &PlacedRun,
    deltas: &[&'a GeometryDelta],
) -> (Vec<(Option<&'a GeometryDelta>, PlacedRun)>, usize) {
    let mut out = Vec::new();
    if run.values.is_empty() {
        return (out, 0);
    }
    let spf = run.placement.seconds_per_frame;
    let mut groups: Vec<(Option<&GeometryDelta>, usize, usize)> = Vec::new();
    for index in 0..run.values.len() {
        let sec = run.placement.start_sec + index as f64 * spf;
        let selected = deltas
            .iter()
            .copied()
            .filter(|delta| delta.contains_old(sec))
            .max_by_key(|delta| delta.rank());
        match groups.last_mut() {
            Some((current, _, last)) if *current == selected && *last + 1 == index => {
                *last = index;
            }
            _ => groups.push((selected, index, index)),
        }
    }
    let mut split = 0usize;
    for (target, first, last) in groups {
        if first > 0 || last + 1 < run.values.len() {
            split += 1;
        }
        out.push((
            target,
            PlacedRun {
                id: run.id,
                seq: run.seq,
                values: run.values[first..=last].to_vec(),
                placement: RunPlacement {
                    start_sec: run.placement.start_sec + first as f64 * spf,
                    seconds_per_frame: spf,
                },
            },
        ));
    }
    (out, split)
}

fn transform_segment(
    segment: PlacedRun,
    delta: &GeometryDelta,
    seq_cursor: &mut u64,
) -> Option<PlacedRun> {
    if segment.values.is_empty() || !delta.old.is_valid() || !delta.new.is_valid() {
        return None;
    }
    let scale = delta.new.duration_sec / delta.old.duration_sec;
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let start = delta.new.start_sec + (segment.placement.start_sec - delta.old.start_sec) * scale;
    if !start.is_finite() || start < 0.0 {
        return None;
    }
    let seq = (*seq_cursor).max(1);
    *seq_cursor = seq.saturating_add(1);
    Some(PlacedRun {
        // id 只用于诊断；seq 是覆盖顺序。
        id: segment.id,
        seq,
        values: segment.values,
        placement: RunPlacement {
            start_sec: start,
            seconds_per_frame: segment.placement.seconds_per_frame * scale,
        },
    })
}

fn clear_run_window(run: PlacedRun, lo: f64, hi: f64, out: &mut Vec<PlacedRun>) {
    if run.values.is_empty() {
        return;
    }
    let run_lo = run.placement.start_sec;
    let run_hi = run.placement.span_end_sec(run.values.len());
    if run_hi <= lo + EPS || run_lo >= hi - EPS {
        out.push(run);
        return;
    }
    if let Some(left) = run.slice_by_project_sec(run_lo, lo - EPS) {
        if left.end_sec() < lo - EPS {
            out.push(left);
        }
    }
    if let Some(right) = run.slice_by_project_sec(hi, run_hi) {
        if right.placement.start_sec >= hi - EPS {
            out.push(right);
        }
    }
}

/// 让 seq 在几何搬移后保持“后写者胜”：返回给事务调用方统一分配。
pub fn allocate_geometry_sequences(authority: &mut ParamAuthority, count: usize) {
    for _ in 0..count {
        authority.next_seq = authority.next_seq.saturating_add(1).max(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::param_authority::ParamAuthority;

    fn clip(start: f64, len: f64, source_start: f64, source_len: f64, rate: f64) -> ClipGeometry {
        ClipGeometry {
            key: "clip".into(),
            root: "root".into(),
            project_start: start,
            project_duration: len,
            source_start,
            source_duration: source_len,
            rate,
            reversed: false,
            loop_enabled: false,
            source_id: "src".into(),
            last_seen_epoch: 0,
        }
    }

    fn authority_with_run() -> ParamAuthority {
        let mut authority = ParamAuthority::default();
        authority
            .write_range("root", "pitch", 0, &[61.0; 20], &[true; 20], 0.0)
            .unwrap();
        authority
    }

    fn delta(old: &ClipGeometry, new: &ClipGeometry) -> GeometryDelta {
        GeometryDelta::from_geometry(old, new, 0).unwrap()
    }

    #[test]
    fn move_only_changes_placement() {
        let old = clip(0.0, 0.1, 0.0, 0.1, 1.0);
        let new = clip(0.02, 0.1, 0.0, 0.1, 1.0);
        let mut authority = authority_with_run();
        let values = authority.roots["root"].runs["pitch_edit"][0].values.clone();
        let report = apply_geometry_transaction(&mut authority, &[delta(&old, &new)]).unwrap();
        assert_eq!(report.affected_runs, 1);
        assert_eq!(authority.roots["root"].runs["pitch_edit"][0].values, values);
        assert!(
            (authority.roots["root"].runs["pitch_edit"][0]
                .placement
                .start_sec
                - 0.02)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn partial_move_splits_into_outside_and_moved_segments() {
        let old = clip(0.0, 0.03, 0.0, 0.03, 1.0);
        let new = clip(0.05, 0.03, 0.0, 0.03, 1.0);
        let mut authority = authority_with_run();
        apply_geometry_transaction(&mut authority, &[delta(&old, &new)]).unwrap();
        let runs = &authority.roots["root"].runs["pitch_edit"];
        assert!(runs.len() >= 2, "run must split at the clip boundary");
        assert!(runs
            .iter()
            .any(|run| (run.placement.start_sec - 0.05).abs() < 1e-9));
    }

    #[test]
    fn stretch_scales_placement_but_not_values() {
        let old = clip(0.0, 0.1, 0.0, 0.1, 1.0);
        let new = clip(0.0, 0.2, 0.0, 0.1, 0.5);
        let mut authority = authority_with_run();
        let values = authority.roots["root"].runs["pitch_edit"][0].values.clone();
        apply_geometry_transaction(&mut authority, &[delta(&old, &new)]).unwrap();
        let run = authority.roots["root"].runs["pitch_edit"]
            .iter()
            .find(|run| run.placement.start_sec.abs() < 1e-9)
            .unwrap();
        assert_eq!(run.values, values);
        assert!((run.placement.seconds_per_frame - 0.01).abs() < 1e-9);
    }

    #[test]
    fn target_window_is_cleared_before_moved_runs_are_inserted() {
        let mut authority = ParamAuthority::default();
        authority
            .write_range("root", "pitch", 0, &[61.0; 10], &[true; 10], 0.0)
            .unwrap();
        authority
            .write_range("root", "dyn", 50, &[0.75; 10], &[true; 10], -1.0)
            .unwrap();
        let old = clip(0.0, 0.05, 0.0, 0.05, 1.0);
        let new = clip(0.25, 0.05, 0.0, 0.05, 1.0);
        apply_geometry_transaction(&mut authority, &[delta(&old, &new)]).unwrap();
        let view = crate::editor::param_authority::compose_root_view(
            "root",
            &authority,
            &crate::editor::param_authority::RootBaseline {
                frame_period_ms: 5.0,
                pitch_orig: vec![60.0; 120],
                pitch_target: vec![60.0; 120],
                dyn_orig: vec![],
                tension_orig: vec![],
                pitch_key: Some("k".into()),
                dyn_key: None,
                ready: true,
                revision: 1,
                analysis_pending: false,
            },
            0,
        );
        assert_eq!(
            view.params.extra_curves["dyn"][50], -1.0,
            "target old edit must be cleared"
        );
        assert_eq!(view.params.pitch_edit[50], 61.0);
    }

    #[test]
    fn trim_split_reverse_and_loop_do_not_mutate_authority() {
        let authority = authority_with_run();
        let before = authority.content_hash();
        let old = clip(0.0, 0.05, 0.0, 0.05, 1.0);
        let trim = clip(0.01, 0.03, 0.01, 0.03, 1.0);
        let reverse = ClipGeometry {
            reversed: true,
            ..old.clone()
        };
        let looped = ClipGeometry {
            loop_enabled: true,
            ..old.clone()
        };
        let deltas = vec![
            GeometryDelta::from_geometry(&old, &trim, 0),
            GeometryDelta::from_geometry(&old, &reverse, 1),
            GeometryDelta::from_geometry(&old, &looped, 2),
        ];
        for delta in deltas {
            assert!(delta.is_none());
        }
        assert_eq!(authority.content_hash(), before);
    }

    #[test]
    fn cross_root_move_moves_runs_between_roots() {
        let old = clip(0.0, 0.1, 0.0, 0.1, 1.0);
        let new = ClipGeometry {
            root: "other".into(),
            ..old.clone()
        };
        let mut authority = authority_with_run();
        apply_geometry_transaction(&mut authority, &[delta(&old, &new)]).unwrap();
        assert!(authority
            .roots
            .get("root")
            .is_none_or(|root| !root.has_edits("pitch_edit")));
        assert!(authority.roots["other"].has_edits("pitch_edit"));
    }

    #[test]
    fn copy_maps_window_and_clears_target() {
        let mut authority = authority_with_run();
        let before = authority.content_hash();
        let report = copy_geometry_window(
            &mut authority,
            "root",
            GeometryWindow::new(0.0, 0.03),
            "root",
            GeometryWindow::new(0.10, 0.03),
        )
        .unwrap();
        assert_eq!(report.affected_runs, 1);
        assert_ne!(authority.content_hash(), before);
        let view = crate::editor::param_authority::compose_root_view(
            "root",
            &authority,
            &crate::editor::param_authority::RootBaseline {
                frame_period_ms: 5.0,
                pitch_orig: vec![60.0; 40],
                pitch_target: vec![60.0; 40],
                dyn_orig: vec![],
                tension_orig: vec![],
                pitch_key: Some("k".into()),
                dyn_key: None,
                ready: true,
                revision: 1,
                analysis_pending: false,
            },
            0,
        );
        assert_eq!(view.params.pitch_edit[20], 61.0);
    }

    #[test]
    fn journal_diff_is_one_shot_and_commit_advances_epoch() {
        let mut journal = GeometryJournal::default();
        let old = clip(0.0, 0.05, 0.0, 0.05, 1.0);
        journal.commit(std::slice::from_ref(&old), 1);
        let new = clip(0.02, 0.05, 0.0, 0.05, 1.0);
        assert_eq!(journal.diff(std::slice::from_ref(&new)).len(), 1);
        journal.commit(std::slice::from_ref(&new), 2);
        assert!(journal.diff(std::slice::from_ref(&new)).is_empty());
    }
}
