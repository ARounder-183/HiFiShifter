//! 宿主几何变化 → 参数 run 放置变换。
//!
//! 锁定只影响**一次几何事务**：关闭时只更新几何日志；开启时只改
//! `placement` 与边界切分，`values` 逐字节不动。
//!
//! # 分类契约（这一节就是"锁定参数线"的全部语义）
//!
//! 一次几何变化只有三种可能，判据只看**宿主给的三个事实**：
//!
//! | 变化 | 判据 | 参数线 |
//! |---|---|---|
//! | 移动 | 源窗口不变、工程起点变 | 整段平移 |
//! | 拉伸 | **播放倍率**变了、工程时长也变了 | 按新旧时长比缩放 |
//! | 其余 | 以上都不成立 | **原地不动** |
//!
//! "其余"包含裁剪、**分割**、滑源、倒放、Loop 开关、换素材 —— 它们改变的是
//! "这条 clip 露出素材的哪一段"，不是"素材被拉长压短"。用户把一条 item 一刀切两半，
//! 两半各自仍然播放原来那段时间上的原速内容：**分割不是压缩**，参数线一根都不该动。
//!
//! 【为什么倍率必须是宿主读数而不是除出来的】`源窗口跨度 / 工程时长` 这个商只在
//! "源窗口就是消费窗口"时等于倍率。**循环源**（REAPER 的 `B_LOOPSRC`，新建 item
//! 的默认值）的源窗口是 `[锚点, 媒体末)` —— 它比消费窗口长得多，于是任何一次长度
//! 变化都会让这个商变化。据此判成拉伸，就会把整条参数线压进新长度里：用户看到
//! "分割一个 item，它的参数线被压缩到半段之内"。倍率是宿主自己报的量，分割与裁剪
//! 都不改它，所以按它分类，分割就不可能冒充拉伸。
//!
//! 【为什么阈值不是 1e-9】见 [`crate::host_value`]：f32 往返与采样量化会带来
//! ~1e-7 的相对误差，用绝对阈值判"倍率变没变"会把舍入当成编辑。

use super::param_authority::{ParamAuthority, PlacedRun, RootParamAuthority, RunPlacement};
use crate::state::Clip;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const EPS: f64 = 1e-9;

/// 宿主几何的**稳定身份**。
///
/// 【为什么不用 `clip.id`】呈现层的 id 会在 `ara-region-{key}` 与 `ara-item-{guid}`
/// 之间迁移（见 `ara::model::remap_and_log` 的认领循环）。用 id 做键，一次迁移就会让
/// 几何日志"少一条、多一条"，下一次移动因此算不出 Move —— 用户看到的就是"锁开了
/// 跟没开一样"。
///
/// 【为什么优先 item GUID】GUID 跨会话存活、且**移动不会改变它**；region key 只在本次
/// 会话内稳定；`clip:{id}` 只是最后兜底（`ara-clip-N` 形态）。
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GeometryKey(String);

impl GeometryKey {
    pub fn item(guid: &str) -> Self {
        Self(format!("item:{guid}"))
    }
    pub fn region(key: u64) -> Self {
        Self(format!("region:{key}"))
    }
    pub fn clip(id: &str) -> Self {
        Self(format!("clip:{id}"))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Display for GeometryKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// 一条 Clip 在项目时间轴上的几何事实。身份键优先 item GUID，ARA-only 用稳定指纹。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ClipGeometry {
    pub key: GeometryKey,
    pub root: String,
    pub project_start: f64,
    pub project_duration: f64,
    pub source_start: f64,
    /// 源窗口的跨度（秒）。**循环源的窗口是 `[锚点, 媒体末)`**，因此它不等于
    /// `project_duration × rate` —— 只有非循环 clip 才有那个等式。
    pub source_duration: f64,
    /// 这条 clip 的**有效播放倍率**（`Clip::playback_rate`，即 item 倍率 × take 倍率）。
    ///
    /// 【为什么不能由 `source_duration / project_duration` 除出来】那个商只在
    /// "源窗口 == 消费窗口"时等于倍率，而循环源的源窗口是 `[锚点, 媒体末)`：它比
    /// 消费窗口长得多，于是任何一次长度变化（分割、裁剪）都会让这个商变化。分类器
    /// 据此判成一次 Stretch，把整条参数线压进新长度里 —— 用户看到的正是"分割一个
    /// item，它的参数线被压缩到半段之内"。倍率是宿主自己给的读数，分割/裁剪不改它。
    ///
    /// 【取值口径】与渲染键同一个函数（[`crate::render_key::clip_playback_rate`]）：
    /// 非法值按 1.0，与实时引擎一致。两处各写一份，就会出现"渲染按 0.5 倍速、
    /// 几何按 1.0 分类"的错位。
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

    /// 一行几何读数：`pos=…+… src=[…+…] rate=… loop=… rev=…`。
    ///
    /// 【为什么要它】"锁定参数线没生效"这一类报障，唯一的判据是**这一帧的两个窗口
    /// 差在哪**。旧日志只写 `deltas=[item:{GUID}:stretch]` —— 它说"判成了拉伸"，
    /// 却不说"哪一项让它这么判"。于是排查只能从代码反推，而真正的答案（循环源的
    /// 窗口不随长度收缩 ⇒ 除出来的倍率翻倍）恰恰藏在被丢掉的那几个数里。
    pub fn trace_facts(&self) -> String {
        format!(
            "pos={:.6}+{:.6} src=[{:.6}+{:.6}] rate={:.6} loop={} rev={}",
            self.project_start,
            self.project_duration,
            self.source_start,
            self.source_duration,
            self.rate,
            self.loop_enabled,
            self.reversed
        )
    }

    pub fn from_clip(clip: &Clip, root: &str, key: GeometryKey) -> Self {
        let source_start = clip.source_start_sec.max(0.0);
        let source_end = clip
            .source_end_sec
            .max(source_start + clip.length_sec.max(0.0) * clip.playback_rate.abs() as f64);
        let source_duration = (source_end - source_start).max(0.0);
        Self {
            key,
            root: root.to_owned(),
            project_start: clip.start_sec,
            project_duration: clip.length_sec,
            source_start,
            source_duration,
            // 【宿主读数，不是除出来的】见 `rate` 字段的说明。
            rate: crate::render_key::clip_playback_rate(clip),
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

/// 与独立 App 对齐的分类（契约见模块文档）：
/// * **倍率**与工程时长同时变化 => Stretch；
/// * 源窗不变、工程起点变化 => Move；
/// * 其余（裁剪/**分割**/滑源/倒放/Loop/Section/换素材）=> 不搬参数。
///
/// 【为什么全部比较都走 `host_value::compatible`】这些量来自宿主 f64 与模型 f32
/// 的往返，1e-7 量级的相对误差是**常态**而不是编辑。判据偏严的代价不是"少刷新
/// 一次"，而是把一次分割当成一次拉伸（见模块文档与 `ClipGeometry::rate`）。
pub fn classify(old: &ClipGeometry, new: &ClipGeometry) -> GeometryOp {
    use crate::host_value::compatible;
    if old.source_id != new.source_id {
        return GeometryOp::Replace;
    }
    if !old.is_well_formed() || !new.is_well_formed() {
        return GeometryOp::TrimOrNone;
    }
    let rate_changed = !compatible(old.rate, new.rate);
    let source_start_changed = !compatible(old.source_start, new.source_start);
    let source_duration_changed = !compatible(old.source_duration, new.source_duration);
    if rate_changed && !compatible(old.project_duration, new.project_duration) {
        return GeometryOp::Stretch {
            old: GeometryWindow::new(old.project_start, old.project_duration),
            new: GeometryWindow::new(new.project_start, new.project_duration),
        };
    }
    if !source_start_changed
        && !source_duration_changed
        && (old.root != new.root || !compatible(old.project_start, new.project_start))
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
    pub key: GeometryKey,
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
    // 【没有任何编辑时，几何事务必须是纯 no-op】否则一次"移动了 clip、但这条轨还没画过线"
    // 也会把 authority.revision 推进一格 —— 那个代次是渲染守卫的一部分，于是每次宿主
    // 移动都会让在途渲染变成 superseded（"尚未应用"的另一条成因）。
    if authority.roots.is_empty() {
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
    // 【残影守卫】Move/Stretch 的每一帧都必须落到新位置；任何一帧被丢弃（例如目标
    // 起点为负）都会在旧位置留下"搬不走的残影"。宁可整笔不生效，也不要留半条线。
    if report.dropped_frames > 0 {
        return Err(format!(
            "geometry transaction would drop {} frame(s); the authority was left unchanged",
            report.dropped_frames
        ));
    }
    report.applied_deltas += active.len();
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

/// 两条几何是否描述**同一份素材、同样长的一段**（与"同一时刻"无关）。
///
/// 【为什么这是配对判据】宿主在移动 / 跨轨拖动时可能销毁并重建 region，clip 身份因此
/// 变化，但"源文件 + 源窗口 + 速率 + 方向 + 工程时长"逐项不变。满足这条的
/// "消失 + 出现"就是一次 **移动**，而不是"复制"。
///
/// 【为什么工程时长也必须在判据里】移动与复制都**整条**搬同一段素材，长度不变。
/// 长度变了就不是"同一条 clip 换了个身份"，而是一次同时改了长度与位置的编辑 ——
/// 那种情形按移动处理会把曲线**整体平移却不缩放**（长度变了、曲线间距没变），
/// 比不做更糟。宁可这一帧不配对，等下一帧几何稳定后再算。
fn same_content(a: &ClipGeometry, b: &ClipGeometry) -> bool {
    use crate::host_value::compatible;
    a.source_id == b.source_id
        && compatible(a.source_start, b.source_start)
        && compatible(a.source_duration, b.source_duration)
        && compatible(a.rate, b.rate)
        && compatible(a.project_duration, b.project_duration)
        && a.reversed == b.reversed
}

/// 一帧几何事实。
///
/// 【为什么需要 `complete`】"这一帧没看到某个 clip"有五种正常来源：呈现层过滤、
/// scope 暂时为空、工作区身份迁移窗口、轨道全 mute、`is_well_formed` 失败。把它们
/// 一律当成"clip 被删了"会让几何日志被**空帧清空** —— 那正是"锁定失效"的头号成因。
/// 只有枚举方明确声明"这一帧覆盖了全部对象"时，才允许丢弃没见到的条目。
#[derive(Clone, Debug, Default)]
pub struct GeometryFrame {
    pub clips: Vec<ClipGeometry>,
    pub complete: bool,
}

impl GeometryFrame {
    pub fn new(clips: Vec<ClipGeometry>, complete: bool) -> Self {
        Self { clips, complete }
    }

    /// 完整帧（宿主清单枚举成功时的常态）。
    pub fn complete(clips: Vec<ClipGeometry>) -> Self {
        Self {
            clips,
            complete: true,
        }
    }

    /// 不完整帧：只允许**新增/更新**，绝不剪枝。
    pub fn partial(clips: Vec<ClipGeometry>) -> Self {
        Self {
            clips,
            complete: false,
        }
    }
}

/// 运行时几何日志：不存档，可由宿主清单重建。只比较“旧日志 → 当前事实”。
#[derive(Clone, Debug, Default)]
pub struct GeometryJournal {
    entries: BTreeMap<GeometryKey, ClipGeometry>,
    epoch: u64,
    unknown_keys: BTreeSet<GeometryKey>,
}

impl GeometryJournal {
    pub fn entries(&self) -> &BTreeMap<GeometryKey, ClipGeometry> {
        &self.entries
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn unknown_keys(&self) -> &BTreeSet<GeometryKey> {
        &self.unknown_keys
    }

    /// 与这一帧几何比较，产出 Move / Stretch / Copy。
    ///
    /// 【三条与旧实现的实质差别】
    /// 1. **不在 `diff` 里剪枝**。"没看到"不等于"被删了"；日志的推进只发生在
    ///    `commit` / `observe_only`（即"这一轮真的处理完了"），且完整帧才丢条目。
    /// 2. **消失 ↔ 出现会配对成 Move**。身份迁移（region → item GUID）与"宿主销毁重建
    ///    region 的移动"都表现为"一条消失、一条出现"，内容窗口完全一致。旧实现把它
    ///    当 Copy，于是源保留 ⇒ 旧位置永远留着一条线。
    /// 3. **Copy 只在源仍在当前帧时成立**。真正的复制/粘贴，源必然还在；源不在就不是
    ///    复制，是移动。
    pub fn diff(&mut self, frame: &GeometryFrame) -> Vec<GeometryDelta> {
        self.epoch = self.epoch.saturating_add(1);
        let mut current = BTreeMap::<GeometryKey, (usize, &ClipGeometry)>::new();
        for (order, geometry) in frame.clips.iter().enumerate() {
            if geometry.key.is_empty() || !geometry.is_well_formed() {
                self.unknown_keys.insert(geometry.key.clone());
                continue;
            }
            current.insert(geometry.key.clone(), (order, geometry));
        }
        let mut deltas = Vec::new();
        // ① 键相同的：正常分类（Move / Stretch / 其余丢弃）。
        for (key, (order, geometry)) in &current {
            let Some(old) = self.entries.get(key) else {
                continue;
            };
            if let Some(mut delta) = GeometryDelta::from_geometry(old, geometry, *order as u64) {
                delta.priority = if old.root == geometry.root { 1 } else { 2 };
                deltas.push(delta);
            }
        }
        // ② 消失 ↔ 出现：按内容窗口配对。
        let vanished = self
            .entries
            .iter()
            .filter(|(key, _)| !current.contains_key(*key))
            .map(|(_, geometry)| geometry)
            .collect::<Vec<_>>();
        let appeared = current
            .iter()
            .filter(|(key, _)| !self.entries.contains_key(*key))
            .map(|(_, (order, geometry))| (*order, *geometry))
            .collect::<Vec<_>>();
        let mut consumed = BTreeSet::<usize>::new();
        let mut upgraded = BTreeSet::<GeometryKey>::new();
        let mut paired = BTreeSet::<GeometryKey>::new();
        let mut ambiguous = BTreeSet::<GeometryKey>::new();
        for (order, fresh) in &appeared {
            let candidates = vanished
                .iter()
                .enumerate()
                .filter(|(index, old)| !consumed.contains(index) && same_content(old, fresh))
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            // 多个候选 ⇒ 不配对（宁可不动，也不要错搬）。但它仍然"可能是一次移动"，
            // 因此绝不能在同一帧里再把它当成复制 —— 否则源保留、旧位置永远有线。
            if candidates.len() > 1 {
                ambiguous.insert(fresh.key.clone());
                continue;
            }
            let [index] = candidates.as_slice() else {
                continue;
            };
            let old = vanished[*index];
            consumed.insert(*index);
            paired.insert(fresh.key.clone());
            let moved = old.root != fresh.root
                || !crate::host_value::compatible(fresh.project_start, old.project_start);
            if moved {
                let window_old = GeometryWindow::new(old.project_start, old.project_duration);
                let window_new = GeometryWindow::new(fresh.project_start, fresh.project_duration);
                if window_old.is_valid() && window_new.is_valid() {
                    deltas.push(GeometryDelta {
                        key: fresh.key.clone(),
                        old_root: old.root.clone(),
                        new_root: fresh.root.clone(),
                        old: window_old,
                        new: window_new,
                        op: GeometryOp::Move {
                            old: window_old,
                            new: window_new,
                        },
                        priority: if old.root == fresh.root { 1 } else { 2 },
                        order: *order as u64,
                    });
                }
            } else {
                // 只是身份升级（同一位置、同一内容）：不产生 delta，只换键。
                upgraded.insert(fresh.key.clone());
            }
        }
        // ③ 复制：源必须**仍在当前帧**里。
        for (order, fresh) in &appeared {
            if upgraded.contains(&fresh.key)
                || paired.contains(&fresh.key)
                || ambiguous.contains(&fresh.key)
            {
                continue;
            }
            let sources = current
                .values()
                .filter(|(_, geometry)| geometry.key != fresh.key && same_content(geometry, fresh))
                .map(|(_, geometry)| *geometry)
                .collect::<Vec<_>>();
            let [source] = sources.as_slice() else {
                continue;
            };
            if crate::host_value::compatible(source.project_start, fresh.project_start) {
                continue;
            }
            deltas.push(GeometryDelta {
                key: fresh.key.clone(),
                old_root: source.root.clone(),
                new_root: fresh.root.clone(),
                old: GeometryWindow::new(source.project_start, source.project_duration),
                new: GeometryWindow::new(fresh.project_start, fresh.project_duration),
                op: GeometryOp::Copy {
                    source: GeometryWindow::new(source.project_start, source.project_duration),
                    target: GeometryWindow::new(fresh.project_start, fresh.project_duration),
                },
                priority: 0,
                order: *order as u64,
            });
        }
        deltas
    }

    /// 只有事务成功后调用：把这一帧几何并入日志。
    ///
    /// 完整帧整份替换（消失的条目随之丢弃）；不完整帧只做**并集**（绝不因为"这一轮
    /// 没枚举到"而丢掉一条活着的 clip）。
    pub fn commit(&mut self, frame: &GeometryFrame, epoch: u64) {
        let mut next = if frame.complete {
            BTreeMap::new()
        } else {
            self.entries.clone()
        };
        for geometry in &frame.clips {
            if geometry.key.is_empty() || !geometry.is_well_formed() {
                continue;
            }
            let mut geometry = geometry.clone();
            geometry.last_seen_epoch = epoch;
            next.insert(geometry.key.clone(), geometry);
        }
        self.entries = next;
        self.epoch = epoch.max(self.epoch);
        self.unknown_keys.clear();
    }

    /// 锁定关闭/事务失败时只更新日志（I5）。
    pub fn observe_only(&mut self, frame: &GeometryFrame) {
        let epoch = self.epoch.saturating_add(1);
        self.commit(frame, epoch);
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
    use crate::editor::param_authority::{compose_root_view, ParamAuthority, RootBaseline};

    /// 一份"什么都没编辑"的基线（逐帧 60）。
    fn baseline(count: usize) -> RootBaseline {
        RootBaseline {
            frame_period_ms: 5.0,
            pitch_orig: vec![60.0; count],
            dyn_orig: Vec::new(),
            tension_orig: Vec::new(),
            pitch_key: Some("k".into()),
            dyn_key: None,
            ready: true,
            revision: 1,
            analysis_pending: false,
            ..Default::default()
        }
    }

    fn clip(start: f64, len: f64, source_start: f64, source_len: f64, rate: f64) -> ClipGeometry {
        ClipGeometry {
            key: GeometryKey::clip("clip"),
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

    /// 一条真实 `Clip`：走产品同一条反序列化 + 物化路径（默认值只有一个来源）。
    fn clip_from(value: serde_json::Value) -> Clip {
        let mut clip: Clip = serde_json::from_value(value).unwrap();
        clip.normalize_takes();
        clip
    }

    /// 循环源 item：媒体 10 s，源窗口恒为 `[锚点, 媒体末)`（与 item 长度无关）。
    fn loop_item(length: f64) -> Clip {
        clip_from(serde_json::json!({
            "id": "ara-item-{A}", "track_id": "root", "name": "vocal",
            "start_sec": 0.0, "length_sec": length,
            "takes": [{
                "id": "take", "name": "vocal", "source_path": "src",
                "source_start_sec": 0.0, "source_end_sec": 10.0,
                "playback_rate": 1.0, "duration_sec": 10.0, "loop_enabled": true
            }]
        }))
    }

    fn geometry_of(clip: &Clip) -> ClipGeometry {
        ClipGeometry::from_clip(clip, "root", GeometryKey::item("{A}"))
    }

    /// **分割不是压缩。**
    ///
    /// 【为什么这条必须存在】REAPER 新建 item 默认打开"循环源"（`B_LOOPSRC=1`），
    /// 于是源窗口是 `[锚点, 媒体末)` —— 它**不随 item 长度收缩**。旧实现把倍率算成
    /// `源窗口跨度 / 工程时长`，分割之后这个商从 1.0 变成 2.0，分类器据此判成一次
    /// Stretch，把整条参数线压进左半段。用户报的就是这个："分割 item，参数线被压缩
    /// 到该 item 范围内"。
    ///
    /// 判据取"没有 delta"，因为那正是 `锁定参数线` 开启时唯一正确的处置：分割只改变
    /// "这条 clip 露出素材的哪一段"，素材本身一个采样都没有被拉长压短。
    #[test]
    fn a_loop_source_split_is_not_a_stretch() {
        let parent = loop_item(10.0);
        let mut left = parent.clone();
        left.length_sec = 5.0;
        let (old, new) = (geometry_of(&parent), geometry_of(&left));
        assert_eq!(old.rate, 1.0, "倍率取宿主读数");
        assert_eq!(new.rate, 1.0, "分割不改变播放倍率");
        assert_eq!(
            new.source_duration, old.source_duration,
            "循环源的源窗口不随长度收缩 —— 正是它让'除出来的倍率'失真"
        );
        assert!(
            GeometryDelta::from_geometry(&old, &new, 0).is_none(),
            "分割不是拉伸：{old:?} -> {new:?}"
        );
    }

    /// 同一个 item 的**右边缘裁剪**同理：素材没被拉伸，曲线留在原地。
    #[test]
    fn a_loop_source_trim_is_not_a_stretch() {
        let parent = loop_item(10.0);
        let mut trimmed = parent.clone();
        trimmed.length_sec = 7.5;
        assert!(
            GeometryDelta::from_geometry(&geometry_of(&parent), &geometry_of(&trimmed), 0)
                .is_none()
        );
    }

    /// 真正的拉伸（宿主改 `D_PLAYRATE` 并同步改长度）仍然必须缩放曲线。
    ///
    /// 【为什么循环源也要测】倍率改口径之后，唯一的风险是"把真的拉伸也当成裁剪"。
    /// 这条用例把那个方向钉死：1.0 倍 → 0.5 倍、10 s → 20 s，曲线间距必须翻倍。
    #[test]
    fn a_real_stretch_of_a_loop_source_still_scales_the_curve() {
        let parent = loop_item(10.0);
        let mut stretched = parent.clone();
        stretched.length_sec = 20.0;
        stretched.playback_rate = 0.5;
        let (old, new) = (geometry_of(&parent), geometry_of(&stretched));
        assert_eq!(old.rate, 1.0);
        assert_eq!(new.rate, 0.5);
        let mut authority = authority_with_run();
        let before = authority.roots["root"].runs["pitch_edit"][0]
            .placement
            .seconds_per_frame;
        let delta = GeometryDelta::from_geometry(&old, &new, 0).expect("拉伸必须被识别");
        assert!(matches!(delta.op, GeometryOp::Stretch { .. }), "{delta:?}");
        apply_geometry_transaction(&mut authority, &[delta]).unwrap();
        let run = &authority.roots["root"].runs["pitch_edit"][0];
        assert!(
            (run.placement.seconds_per_frame - before * 2.0).abs() < 1e-9,
            "帧间距必须按新旧时长比放大：{} -> {}",
            before,
            run.placement.seconds_per_frame
        );
    }

    /// 非循环 item 的分割同样什么都不搬（回归护栏：修循环源不能反过来放松这一条）。
    #[test]
    fn a_plain_split_is_not_a_stretch_either() {
        let parent = clip_from(serde_json::json!({
            "id": "ara-item-{A}", "track_id": "root", "name": "vocal",
            "start_sec": 0.0, "length_sec": 10.0,
            "takes": [{
                "id": "take", "source_path": "src", "source_start_sec": 0.0,
                "source_end_sec": 10.0, "playback_rate": 1.0, "duration_sec": 10.0
            }]
        }));
        let mut left = parent.clone();
        left.length_sec = 5.0;
        left.takes[0].source_end_sec = 5.0;
        left.source_end_sec = 5.0;
        assert!(
            GeometryDelta::from_geometry(&geometry_of(&parent), &geometry_of(&left), 0).is_none()
        );
    }

    /// **分割本身什么都不搬。**
    ///
    /// 【为什么这条必须存在】用户报的"分割之后右半段的参数线被重置"，根因是插件侧
    /// 把分割当成一次几何事件、并试图把父段的曲线"喂"给右半段。这条用例钉死的是
    /// 相反的事实：分割之后宿主给出的是**两条新的 clip**（源窗各自是父段的一半），
    /// 而参数 run 活在工程时间上 —— 它既没被删、也没被移动，**根本不需要任何处理**。
    ///
    /// 判据取"分割前后 compose_root_view 逐帧相等"，因为那正是用户看见的东西。
    #[test]
    fn a_split_moves_nothing_and_needs_no_lineage() {
        let mut authority = authority_with_run();
        let before = compose_root_view("root", &authority, &baseline(20), 0);
        // 宿主把 [0.0, 0.2) 一刀切成 [0.0, 0.1) 与 [0.1, 0.2)：两条新 clip，源窗各半。
        // 它们与父段**没有**共同的内容窗口（各自只有一半），因此几何日志不会把它们
        // 配对成 Move —— 这里显式断言这一点。
        let parent = clip(0.0, 0.2, 0.0, 0.2, 1.0);
        let left = clip(0.0, 0.1, 0.0, 0.1, 1.0);
        let right = clip(0.1, 0.1, 0.1, 0.1, 1.0);
        assert!(
            GeometryDelta::from_geometry(&parent, &left, 0).is_none(),
            "分割出的左半段不能被当成一次裁剪之外的几何事件"
        );
        assert!(
            GeometryDelta::from_geometry(&parent, &right, 0).is_none(),
            "分割出的右半段同理"
        );
        let report = apply_geometry_transaction(&mut authority, &[]).unwrap();
        assert!(report.is_empty(), "没有 delta 就不该有任何搬运");
        let after = compose_root_view("root", &authority, &baseline(20), 0);
        assert_eq!(
            before.params.pitch_edit, after.params.pitch_edit,
            "分割前后曲线必须逐帧相等"
        );
        assert_eq!(authority.revision, before.authority_revision);
    }

    /// 分割之后拖动**右半段**：只有右半段的线跟着走，左半段原地不动。
    ///
    /// 【为什么这是锁关闭语义的完整表达】"线跟着素材走"只在用户真的搬动素材时发生；
    /// 分割本身不搬。这条用例把两个动作分开验证：先分割（什么都不变），再移动右半段
    /// （只有右半段那一段线被平移，且跨过分割点的那条 run 被切干净）。
    #[test]
    fn moving_one_half_after_a_split_leaves_the_other_half_in_place() {
        let mut authority = ParamAuthority::default();
        // 显式钉住帧周期：本用例的窗口算术全部以 10 ms 为单位。
        authority.frame_period_ms = 10.0;
        // 一条横跨分割点的 run：工程 [0.0, 0.3)，30 帧 × 10 ms。
        authority
            .write_range("root", "pitch", 0, &[61.0; 30], &[true; 30], 0.0)
            .unwrap();
        // 宿主在 0.1 s 处分割；右半段 [0.1, 0.2) 被拖到 [0.4, 0.5)。
        let old = clip(0.1, 0.1, 0.1, 0.1, 1.0);
        let new = clip(0.4, 0.1, 0.1, 0.1, 1.0);
        apply_geometry_transaction(&mut authority, &[delta(&old, &new)]).unwrap();

        let runs = &authority.roots["root"].runs["pitch_edit"];
        // 左半段（含分割点左侧）必须原地不动。
        let head = runs
            .iter()
            .find(|run| run.placement.start_sec.abs() < 1e-9)
            .expect("左半段必须留在原位");
        // 切点按半开区间取：正好落在 0.10 的那一帧属于**被搬走**的那一段。
        assert_eq!(head.values, vec![61.0; 10], "[0.00, 0.09] 共 10 帧");
        // 右半段必须出现在新位置。
        let moved = runs
            .iter()
            .find(|run| (run.placement.start_sec - 0.4).abs() < 1e-9)
            .expect("右半段必须跟着素材走");
        // 被搬走的窗口两端都是闭的（`contains_old` 的约定）：0.10 与 0.20 都跟着走。
        assert_eq!(moved.values, vec![61.0; 11], "[0.10, 0.20] 共 11 帧");
        // 旧位置不得留下残影。
        assert!(
            !runs
                .iter()
                .any(|run| (run.placement.start_sec - 0.1).abs() < 1e-9),
            "搬走的那一段不能在旧位置留残影"
        );
        // 分割点右侧 0.2 s 之后的部分（不在被拖动的窗口里）必须留在原位。
        let tail = runs
            .iter()
            .find(|run| (run.placement.start_sec - 0.21).abs() < 1e-9)
            .expect("窗口之外的尾巴必须留在原位");
        assert_eq!(tail.values, vec![61.0; 9], "[0.21, 0.29] 共 9 帧");
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
                dyn_orig: vec![],
                tension_orig: vec![],
                pitch_key: Some("k".into()),
                dyn_key: None,
                ready: true,
                revision: 1,
                analysis_pending: false,

                ..Default::default()
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
                dyn_orig: vec![],
                tension_orig: vec![],
                pitch_key: Some("k".into()),
                dyn_key: None,
                ready: true,
                revision: 1,
                analysis_pending: false,

                ..Default::default()
            },
            0,
        );
        assert_eq!(view.params.pitch_edit[20], 61.0);
    }

    #[test]
    fn journal_diff_is_one_shot_and_commit_advances_epoch() {
        let mut journal = GeometryJournal::default();
        let old = clip(0.0, 0.05, 0.0, 0.05, 1.0);
        journal.commit(&GeometryFrame::complete(vec![old.clone()]), 1);
        let new = clip(0.02, 0.05, 0.0, 0.05, 1.0);
        assert_eq!(
            journal
                .diff(&GeometryFrame::complete(vec![new.clone()]))
                .len(),
            1
        );
        journal.commit(&GeometryFrame::complete(vec![new.clone()]), 2);
        assert!(journal.diff(&GeometryFrame::complete(vec![new])).is_empty());
    }

    /// G1：空帧**不得**剪掉日志 —— 这是"锁定失效"的头号成因。
    #[test]
    fn empty_frame_never_prunes_the_journal() {
        let mut journal = GeometryJournal::default();
        let old = clip(0.0, 0.05, 0.0, 0.05, 1.0);
        journal.commit(&GeometryFrame::complete(vec![old.clone()]), 1);
        assert!(journal.diff(&GeometryFrame::partial(Vec::new())).is_empty());
        assert_eq!(journal.entries().len(), 1, "partial empty frame pruned");
        assert!(journal
            .diff(&GeometryFrame::complete(Vec::new()))
            .is_empty());
        assert_eq!(journal.entries().len(), 1, "diff must not prune at all");
        journal.commit(&GeometryFrame::complete(Vec::new()), 2);
        assert!(
            journal.entries().is_empty(),
            "complete empty frame must prune"
        );
    }

    /// G2：不完整帧只允许新增/更新，绝不能因为"没枚举到"丢掉活着的条目。
    #[test]
    fn incomplete_frame_keeps_unseen_entries() {
        let mut journal = GeometryJournal::default();
        let a = clip(0.0, 0.05, 0.0, 0.05, 1.0);
        let b = ClipGeometry {
            key: GeometryKey::clip("b"),
            ..a.clone()
        };
        journal.commit(&GeometryFrame::complete(vec![a.clone(), b.clone()]), 1);
        journal.commit(&GeometryFrame::partial(vec![a.clone()]), 2);
        assert_eq!(journal.entries().len(), 2, "partial commit must merge");
        journal.commit(&GeometryFrame::complete(vec![a]), 3);
        assert_eq!(journal.entries().len(), 1, "complete commit must replace");
    }

    /// G4：身份变了、素材没变 ⇒ 一次 **Move**（不是 Copy）。
    #[test]
    fn a_move_with_new_identity_becomes_a_move_not_a_copy() {
        let mut journal = GeometryJournal::default();
        let old = ClipGeometry {
            key: GeometryKey::region(7),
            ..clip(0.0, 0.05, 0.0, 0.05, 1.0)
        };
        journal.commit(&GeometryFrame::complete(vec![old.clone()]), 1);
        let new = ClipGeometry {
            key: GeometryKey::item("{GUID}"),
            ..clip(0.20, 0.05, 0.0, 0.05, 1.0)
        };
        let deltas = journal.diff(&GeometryFrame::complete(vec![new]));
        assert_eq!(deltas.len(), 1);
        assert!(matches!(deltas[0].op, GeometryOp::Move { .. }));
        let mut authority = authority_with_run();
        apply_geometry_transaction(&mut authority, &deltas).unwrap();
        // 旧窗口被清空、新窗口接上；窗口之外的样本原地保留（那是另一段时间）。
        let runs = &authority.roots["root"].runs["pitch_edit"];
        assert!(
            runs.iter()
                .all(|run| run.placement.start_sec >= 0.05 - 1e-9),
            "no sample may stay inside the old window: {runs:?}"
        );
        assert!(
            runs.iter()
                .any(|run| (run.placement.start_sec - 0.20).abs() < 1e-9),
            "the moved segment must land on the new position"
        );
    }

    /// 身份变了、源窗口一样、但**长度**变了 ⇒ 不是一次移动。
    ///
    /// 【为什么不配对】宿主在"销毁并重建 region"的同时顺手裁短了这条 clip。若配对判据
    /// 只看源窗口，它会被判成一次 Move —— 于是曲线被**整体平移却不缩放**：长度变了、
    /// 曲线间距没变，比不动更糟。宁可这一帧不配对，等几何稳定后再算。
    #[test]
    fn a_length_change_with_a_new_identity_is_not_a_move() {
        let mut journal = GeometryJournal::default();
        let old = ClipGeometry {
            key: GeometryKey::item("{OLD}"),
            ..clip(0.0, 0.10, 0.0, 0.10, 1.0)
        };
        journal.commit(&GeometryFrame::complete(vec![old]), 1);
        let fresh = ClipGeometry {
            key: GeometryKey::item("{NEW}"),
            ..clip(0.05, 0.05, 0.0, 0.10, 1.0)
        };
        assert!(
            journal
                .diff(&GeometryFrame::complete(vec![fresh]))
                .is_empty(),
            "长度变了就不是同一条 clip 换了身份"
        );
    }

    /// G5：只是身份升级（位置不变）⇒ 不产生 delta。
    #[test]
    fn an_identity_upgrade_produces_no_delta() {
        let mut journal = GeometryJournal::default();
        let old = ClipGeometry {
            key: GeometryKey::region(7),
            ..clip(0.0, 0.05, 0.0, 0.05, 1.0)
        };
        journal.commit(&GeometryFrame::complete(vec![old]), 1);
        let new = ClipGeometry {
            key: GeometryKey::item("{GUID}"),
            ..clip(0.0, 0.05, 0.0, 0.05, 1.0)
        };
        assert!(journal
            .diff(&GeometryFrame::complete(vec![new.clone()]))
            .is_empty());
        journal.commit(&GeometryFrame::complete(vec![new]), 2);
        assert_eq!(journal.entries().len(), 1);
    }

    /// G6：真正的复制（源仍在）⇒ Copy，源曲线逐字节不变。
    #[test]
    fn a_duplicate_keeps_the_source_curve() {
        let mut journal = GeometryJournal::default();
        let source = clip(0.0, 0.05, 0.0, 0.05, 1.0);
        journal.commit(&GeometryFrame::complete(vec![source.clone()]), 1);
        let copy = ClipGeometry {
            key: GeometryKey::item("{COPY}"),
            ..clip(0.20, 0.05, 0.0, 0.05, 1.0)
        };
        let deltas = journal.diff(&GeometryFrame::complete(vec![source, copy]));
        assert_eq!(deltas.len(), 1);
        assert!(matches!(deltas[0].op, GeometryOp::Copy { .. }));
    }

    /// G7：同内容多条同时出现 ⇒ 不猜。
    #[test]
    fn ambiguous_pairs_are_not_guessed() {
        let mut journal = GeometryJournal::default();
        let a = clip(0.0, 0.05, 0.0, 0.05, 1.0);
        let b = ClipGeometry {
            key: GeometryKey::clip("b"),
            ..clip(0.10, 0.05, 0.0, 0.05, 1.0)
        };
        journal.commit(&GeometryFrame::complete(vec![a, b]), 1);
        let a2 = ClipGeometry {
            key: GeometryKey::item("{A}"),
            ..clip(0.30, 0.05, 0.0, 0.05, 1.0)
        };
        let b2 = ClipGeometry {
            key: GeometryKey::item("{B}"),
            ..clip(0.40, 0.05, 0.0, 0.05, 1.0)
        };
        let deltas = journal.diff(&GeometryFrame::complete(vec![a2, b2]));
        assert!(
            deltas.is_empty(),
            "two same-content pairs must not be guessed: {deltas:?}"
        );
    }

    /// G9：applied_deltas 必须把 Copy 也算进去。
    #[test]
    fn geometry_transaction_reports_every_delta() {
        let mut authority = authority_with_run();
        let source = clip(0.0, 0.03, 0.0, 0.03, 1.0);
        let copy = ClipGeometry {
            key: GeometryKey::clip("copy"),
            ..clip(0.10, 0.03, 0.0, 0.03, 1.0)
        };
        let delta = GeometryDelta {
            key: copy.key.clone(),
            old_root: "root".into(),
            new_root: "root".into(),
            old: GeometryWindow::new(0.0, 0.03),
            new: GeometryWindow::new(0.10, 0.03),
            op: GeometryOp::Copy {
                source: GeometryWindow::new(0.0, 0.03),
                target: GeometryWindow::new(0.10, 0.03),
            },
            priority: 0,
            order: 0,
        };
        let report = apply_geometry_transaction(&mut authority, &[delta]).unwrap();
        assert_eq!(report.applied_deltas, 1);
        let _ = source;
    }
}
