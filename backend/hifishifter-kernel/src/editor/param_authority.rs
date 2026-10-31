//! 项目时间的参数编辑权威。
//!
//! 这里只存用户**显式编辑**的稀疏 run；显示数组永远由
//! `(当前分析基线 + 本权威)` 纯函数合成。源域曲线、锚层、差分捕获都不属于
//! 这一层——它们会让“这一笔钉在哪个几何上”变成第二份真相。
//!
//! 关键边界：
//! * `values` 只含非默认值；pitch 0 / tension 0 / dyn −1 / volume 1.0 都是“未编辑”；
//! * `placement` 是项目秒，允许非 1:1 的 `seconds_per_frame`，拉伸只改放置、不重采样；
//! * 所有写入口都以 `(start_frame, values, present)` 显式表达意图，绝不从显示数组差分；
//! * `revision` 单调递增，前端据此丢弃在途旧响应。

use crate::state::{Clip, TrackParamsState};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const PITCH_EDIT: &str = "pitch_edit";
pub const TENSION_EDIT: &str = "tension_edit";
pub const EXTRA_DYN: &str = "extra:dyn";
pub const EXTRA_VOLUME: &str = "extra:volume";
pub const EXTRA_BREATH_GAIN: &str = "extra:breath_gain";

const DEFAULT_FRAME_PERIOD_MS: f64 = 5.0;
const MAX_RUN_VALUES: usize = 1_000_000;
const MAX_RUNS_PER_PARAM: usize = 65_536;
const EPS: f64 = 1e-9;

fn default_frame_period_ms() -> f64 {
    DEFAULT_FRAME_PERIOD_MS
}

/// 一个参数键的默认值（“未编辑”语义）。
///
/// 这个函数是唯一默认值表：写路径、校验、显示都无法各自解释“什么算编辑”。
pub fn default_value_for_key(key: &str) -> f32 {
    match key {
        PITCH_EDIT | TENSION_EDIT => 0.0,
        EXTRA_DYN => -1.0,
        EXTRA_VOLUME | EXTRA_BREATH_GAIN => 1.0,
        _ => 0.0,
    }
}

/// `true` = 这个样本表达的是用户编辑，而不是“回到基线”。
pub fn is_edited_value(key: &str, value: f32, default_value: f32) -> bool {
    if !value.is_finite() {
        return false;
    }
    match key {
        PITCH_EDIT | TENSION_EDIT => value != 0.0,
        EXTRA_DYN => value >= 0.0,
        EXTRA_VOLUME | EXTRA_BREATH_GAIN => (value - 1.0).abs() > 1e-6,
        _ => (value - default_value).abs() > 1e-6,
    }
}

/// 把 UI 参数名规范化为权威键。
pub fn canonical_param_key(param: &str) -> String {
    match param {
        "pitch" | PITCH_EDIT => PITCH_EDIT.to_owned(),
        "tension" | TENSION_EDIT => TENSION_EDIT.to_owned(),
        "dyn" | EXTRA_DYN => EXTRA_DYN.to_owned(),
        "volume" | EXTRA_VOLUME => EXTRA_VOLUME.to_owned(),
        "breath_gain" | EXTRA_BREATH_GAIN => EXTRA_BREATH_GAIN.to_owned(),
        other if other.starts_with("extra:") => other.to_owned(),
        other => format!("extra:{other}"),
    }
}

/// 权威键 → 前端/内核参数名。
pub fn api_param_name(key: &str) -> &str {
    match key {
        PITCH_EDIT => "pitch",
        TENSION_EDIT => "tension",
        EXTRA_DYN => "dyn",
        EXTRA_VOLUME => "volume",
        EXTRA_BREATH_GAIN => "breath_gain",
        other => other.strip_prefix("extra:").unwrap_or(other),
    }
}

/// 一条 run 在项目时间轴上的放置。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RunPlacement {
    /// 第 0 个 value 的工程秒。
    pub start_sec: f64,
    /// 每个 value 占用的工程秒；创建时 = frame_period_ms / 1000。
    pub seconds_per_frame: f64,
}

impl RunPlacement {
    pub fn end_sec(&self, value_count: usize) -> f64 {
        self.start_sec + value_count.saturating_sub(1) as f64 * self.seconds_per_frame
    }

    /// 区间（半开）覆盖到最后一个样本之后一个采样周期；用于“目标范围清理”。
    pub fn span_end_sec(&self, value_count: usize) -> f64 {
        self.start_sec + value_count as f64 * self.seconds_per_frame
    }
}

/// 一条显式编辑 run。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PlacedRun {
    /// 诊断/去重。几何切分后仍沿用原 id 的后缀只在诊断里体现。
    pub id: u64,
    /// 同一帧重叠时后写者胜。
    pub seq: u64,
    /// 只含显式编辑值；绝不存默认值/原始值。
    pub values: Vec<f32>,
    pub placement: RunPlacement,
}

impl PlacedRun {
    pub fn new(id: u64, seq: u64, values: Vec<f32>, placement: RunPlacement) -> Self {
        Self {
            id,
            seq,
            values,
            placement,
        }
    }

    pub fn end_sec(&self) -> f64 {
        self.placement.end_sec(self.values.len())
    }

    pub fn span_end_sec(&self) -> f64 {
        self.placement.span_end_sec(self.values.len())
    }

    /// 在某个项目秒处线性插值；越界返回 None（不跨 run、不跨空洞）。
    pub fn value_at_project_sec(&self, sec: f64) -> Option<f32> {
        if !sec.is_finite() || self.values.is_empty() || self.placement.seconds_per_frame <= 0.0 {
            return None;
        }
        let pos = (sec - self.placement.start_sec) / self.placement.seconds_per_frame;
        if pos < -EPS || pos > self.values.len().saturating_sub(1) as f64 + EPS {
            return None;
        }
        let lo = pos.floor().max(0.0) as usize;
        if lo >= self.values.len() {
            return self.values.last().copied();
        }
        let hi = (lo + 1).min(self.values.len() - 1);
        let t = (pos - lo as f64).clamp(0.0, 1.0) as f32;
        Some(self.values[lo] * (1.0 - t) + self.values[hi] * t)
    }

    /// 按项目时间切出 `[lo, hi]` 内的样本。切点按最近 run 帧取整。
    ///
    /// 结果保留原始 value 字节；边界样本不重采样，因此不会在多次几何操作里累积误差。
    pub fn slice_by_project_sec(&self, lo: f64, hi: f64) -> Option<PlacedRun> {
        if self.values.is_empty() || hi < lo - EPS {
            return None;
        }
        let spf = self.placement.seconds_per_frame;
        let first = ((lo - self.placement.start_sec) / spf).ceil().max(0.0) as usize;
        let last = ((hi - self.placement.start_sec) / spf).floor();
        if !last.is_finite() || last < first as f64 {
            return None;
        }
        let last = (last as usize).min(self.values.len() - 1);
        if first > last {
            return None;
        }
        Some(PlacedRun {
            id: self.id,
            seq: self.seq,
            values: self.values[first..=last].to_vec(),
            placement: RunPlacement {
                start_sec: self.placement.start_sec + first as f64 * spf,
                seconds_per_frame: spf,
            },
        })
    }
}

/// 每条根轨的显式编辑集合。
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RootParamAuthority {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub runs: BTreeMap<String, Vec<PlacedRun>>,
}

impl RootParamAuthority {
    pub fn is_empty(&self) -> bool {
        self.runs.values().all(Vec::is_empty)
    }

    pub fn run_count(&self) -> usize {
        self.runs.values().map(Vec::len).sum()
    }

    pub fn sample_count(&self) -> usize {
        self.runs
            .values()
            .flat_map(|runs| runs.iter())
            .map(|run| run.values.len())
            .sum()
    }

    pub fn has_edits(&self, key: &str) -> bool {
        self.runs.get(key).is_some_and(|runs| !runs.is_empty())
    }
}

/// 参数编辑权威。每条根轨一份，项目时间域，单调 revision。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ParamAuthority {
    #[serde(default = "default_frame_period_ms")]
    pub frame_period_ms: f64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub roots: BTreeMap<String, RootParamAuthority>,
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub next_seq: u64,
}

impl Default for ParamAuthority {
    fn default() -> Self {
        Self {
            frame_period_ms: DEFAULT_FRAME_PERIOD_MS,
            roots: BTreeMap::new(),
            revision: 0,
            next_seq: 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthoritySanitizeReport {
    pub dropped_runs: usize,
    pub dropped_samples: usize,
}

/// 单条 run 的校验错误。
#[derive(Clone, Debug, PartialEq)]
pub struct InvalidRun {
    pub reason: String,
}

impl ParamAuthority {
    pub fn frame_period_sec(&self) -> f64 {
        self.frame_period_ms / 1000.0
    }

    pub fn is_pristine(&self) -> bool {
        self.roots.values().all(RootParamAuthority::is_empty)
    }

    pub fn root(&self, root: &str) -> Option<&RootParamAuthority> {
        self.roots.get(root)
    }

    pub fn root_mut(&mut self, root: &str) -> &mut RootParamAuthority {
        self.roots.entry(root.to_owned()).or_default()
    }

    pub fn run_count(&self) -> usize {
        self.roots.values().map(RootParamAuthority::run_count).sum()
    }

    pub fn sample_count(&self) -> usize {
        self.roots
            .values()
            .map(RootParamAuthority::sample_count)
            .sum()
    }

    /// 序列化内容哈希。用于 I5/I7/I8 这类“操作前后权威必须逐字相同”的断言。
    pub fn content_hash(&self) -> String {
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        blake3::hash(&bytes).to_hex().to_string()
    }

    /// 全量校验。默认值/NaN/超长 run 都返回错误；调用方可以改用 `Self::sanitize`
    /// 做逐 run 隔离。
    pub fn validate(&self) -> Result<(), String> {
        if !self.frame_period_ms.is_finite() || !(0.1..=1000.0).contains(&self.frame_period_ms) {
            return Err("invalid authority frame period".into());
        }
        for (root, authority) in &self.roots {
            for (key, runs) in &authority.runs {
                if runs.len() > MAX_RUNS_PER_PARAM {
                    return Err(format!("too many runs for {root}/{key}"));
                }
                let default_value = default_value_for_key(key);
                for run in runs {
                    validate_run(key, run, default_value)
                        .map_err(|error| format!("{root}/{key}: {}", error.reason))?;
                }
            }
        }
        Ok(())
    }

    /// 逐 run 隔离坏数据：坏 run 被丢弃，其所在根轨与其它参数不受影响。
    pub fn sanitize(&mut self) -> AuthoritySanitizeReport {
        let mut report = AuthoritySanitizeReport {
            dropped_runs: 0,
            dropped_samples: 0,
        };
        if !self.frame_period_ms.is_finite() || !(0.1..=1000.0).contains(&self.frame_period_ms) {
            self.frame_period_ms = DEFAULT_FRAME_PERIOD_MS;
        }
        let mut max_seq = self.next_seq.max(1);
        for authority in self.roots.values_mut() {
            for (key, runs) in &mut authority.runs {
                let default_value = default_value_for_key(key);
                runs.retain(|run| {
                    let ok = validate_run(key, run, default_value).is_ok();
                    if !ok {
                        report.dropped_runs += 1;
                        report.dropped_samples += run.values.len();
                    } else {
                        max_seq = max_seq.max(run.seq.saturating_add(1));
                    }
                    ok
                });
                normalize_runs(runs);
            }
            authority.runs.retain(|_, runs| !runs.is_empty());
        }
        self.roots.retain(|_, authority| !authority.is_empty());
        self.next_seq = max_seq.max(1);
        report
    }

    /// 区间替换写：`present[i] == false` 表示该帧回到基线（删除编辑）。
    ///
    /// 这是唯一写入口。它先切掉所有与 `[start_frame, start_frame+len)` 相交的旧 run，
    /// 再按连续 `present=true` 段创建新 run；不读 `pitch_orig`，也不做整条数组差分。
    pub fn write_range(
        &mut self,
        root: &str,
        param: &str,
        start_frame: usize,
        values: &[f32],
        present: &[bool],
        default_value: f32,
    ) -> Result<u64, String> {
        if values.len() != present.len() {
            return Err("values/present length mismatch".into());
        }
        if values.is_empty() {
            return Ok(self.revision);
        }
        if values.len() > MAX_RUN_VALUES {
            return Err("parameter write exceeds frame budget".into());
        }
        let key = canonical_param_key(param);
        let fp = self.frame_period_sec();
        if !fp.is_finite() || fp <= 0.0 {
            return Err("invalid authority frame period".into());
        }
        let start_sec = start_frame as f64 * fp;
        let end_sec = (start_frame.saturating_add(values.len())) as f64 * fp;
        let existing = self
            .roots
            .get(root)
            .and_then(|authority| authority.runs.get(&key))
            .cloned()
            .unwrap_or_default();
        let before = runs_signature(&existing);
        let mut replaced = Vec::new();
        for run in existing {
            split_outside_range(run, start_sec, end_sec, &mut replaced);
        }
        let mut changed = before != runs_signature(&replaced);

        let mut i = 0usize;
        while i < values.len() {
            if !present[i] || !is_edited_value(&key, values[i], default_value) {
                i += 1;
                continue;
            }
            let begin = i;
            while i + 1 < values.len()
                && present[i + 1]
                && is_edited_value(&key, values[i + 1], default_value)
            {
                i += 1;
            }
            let seq = self.alloc_seq();
            replaced.push(PlacedRun::new(
                seq,
                seq,
                values[begin..=i].to_vec(),
                RunPlacement {
                    start_sec: start_sec + begin as f64 * fp,
                    seconds_per_frame: fp,
                },
            ));
            changed = true;
            i += 1;
        }
        normalize_runs(&mut replaced);
        let authority = self.roots.entry(root.to_owned()).or_default();
        if replaced.is_empty() {
            authority.runs.remove(&key);
        } else {
            authority.runs.insert(key, replaced);
        }
        if authority.is_empty() {
            self.roots.remove(root);
        }
        if changed {
            self.bump_revision();
        }
        Ok(self.revision)
    }

    /// 删除区间内所有编辑（右键重置/初始化）。
    pub fn restore_range(
        &mut self,
        root: &str,
        param: &str,
        start_frame: usize,
        frame_count: usize,
    ) -> u64 {
        if frame_count == 0 {
            return self.revision;
        }
        let key = canonical_param_key(param);
        let fp = self.frame_period_sec();
        let start_sec = start_frame as f64 * fp;
        let end_sec = start_frame.saturating_add(frame_count) as f64 * fp;
        let Some(authority) = self.roots.get_mut(root) else {
            return self.revision;
        };
        let Some(runs) = authority.runs.get_mut(&key) else {
            return self.revision;
        };
        let before = runs_signature(runs);
        let mut kept = Vec::new();
        for run in runs.drain(..) {
            split_outside_range(run, start_sec, end_sec, &mut kept);
        }
        normalize_runs(&mut kept);
        *runs = kept;
        let changed = before != runs_signature(runs);
        if runs.is_empty() {
            authority.runs.remove(&key);
        }
        if authority.is_empty() {
            self.roots.remove(root);
        }
        if changed {
            self.bump_revision();
        }
        self.revision
    }

    fn alloc_seq(&mut self) -> u64 {
        let seq = self.next_seq.max(1);
        self.next_seq = self.next_seq.saturating_add(1).max(seq + 1);
        seq
    }

    fn bump_revision(&mut self) {
        self.revision = self.revision.saturating_add(1);
    }

    /// 重映射根轨身份。唯一映射搬移整份权威；旧根保留为空时删除。
    pub fn remap_roots(&mut self, mapping: &BTreeMap<String, String>) {
        let mut moved = BTreeMap::new();
        for (old, authority) in std::mem::take(&mut self.roots) {
            let target = mapping.get(&old).cloned().unwrap_or(old);
            if let Some(existing) = moved.get_mut(&target) {
                merge_root_authorities(existing, authority);
            } else {
                moved.insert(target, authority);
            }
        }
        self.roots = moved;
        self.roots.retain(|_, authority| !authority.is_empty());
        self.bump_revision();
    }
}

/// 合成输出：显示数组 + 编辑位图。
#[derive(Clone, Debug)]
pub struct RootParamView {
    pub root: String,
    pub params: TrackParamsState,
    /// 与 params 逐帧对齐；没有条目的参数 = 全部 false。
    pub edited: BTreeMap<String, Vec<bool>>,
    pub authority_revision: u64,
    pub baseline_revision: u64,
    pub geometry_revision: u64,
}

impl RootParamView {
    pub fn edited_mask(&self, param: &str) -> Vec<bool> {
        let key = canonical_param_key(param);
        let count = self
            .params
            .pitch_edit
            .len()
            .max(self.params.tension_edit.len())
            .max(
                self.params
                    .extra_curves
                    .values()
                    .map(Vec::len)
                    .max()
                    .unwrap_or(0),
            );
        self.edited
            .get(&key)
            .map(|mask| {
                let mut out = mask.clone();
                out.resize(count, false);
                out
            })
            .unwrap_or_else(|| vec![false; count])
    }

    pub fn view_hash(&self) -> String {
        let bytes = serde_json::to_vec(&(
            &self.params,
            &self.edited,
            self.authority_revision,
            self.baseline_revision,
            self.geometry_revision,
        ))
        .unwrap_or_default();
        blake3::hash(&bytes).to_hex().to_string()
    }
}

/// 一条根轨的分析基线。仅由分析流水线发布，几何变化不得改写它。
#[derive(Clone, Debug, Default)]
pub struct RootBaseline {
    pub frame_period_ms: f64,
    pub pitch_orig: Vec<f32>,
    /// 分析给出的目标曲线（未编辑时显示层就是它，渲染器也需要它）。
    pub pitch_target: Vec<f32>,
    pub dyn_orig: Vec<f32>,
    pub tension_orig: Vec<f32>,
    pub pitch_key: Option<String>,
    pub dyn_key: Option<String>,
    pub ready: bool,
    pub revision: u64,
    /// 分析未完成时保留上一份完整基线；该标记只用于 UI 提示，不参与合成。
    pub analysis_pending: bool,
}

impl RootBaseline {
    pub fn from_params(params: &TrackParamsState) -> Self {
        Self {
            frame_period_ms: if params.frame_period_ms.is_finite() && params.frame_period_ms > 0.0 {
                params.frame_period_ms
            } else {
                DEFAULT_FRAME_PERIOD_MS
            },
            pitch_orig: params.pitch_orig.clone(),
            pitch_target: params.pitch_edit.clone(),
            dyn_orig: params.dyn_orig.clone(),
            tension_orig: params.tension_orig.clone(),
            pitch_key: params.pitch_orig_key.clone(),
            dyn_key: params.dyn_orig_key.clone(),
            ready: !params.pitch_orig.is_empty()
                || !params.pitch_edit.is_empty()
                || !params.dyn_orig.is_empty()
                || !params.tension_orig.is_empty(),
            revision: 1,
            analysis_pending: false,
        }
    }

    pub fn from_display(params: &TrackParamsState) -> Self {
        let mut baseline = Self::from_params(params);
        if baseline.pitch_orig.is_empty() && !params.pitch_edit.is_empty() {
            baseline.pitch_orig = params.pitch_edit.clone();
            baseline.pitch_target = params.pitch_edit.clone();
        }
        if baseline.dyn_orig.is_empty() {
            if let Some(curve) = params.extra_curves.get("dyn") {
                if curve.iter().any(|value| *value >= 0.0) {
                    baseline.dyn_orig = curve.clone();
                }
            }
        }
        baseline.ready = true;
        baseline
            .pitch_key
            .get_or_insert_with(|| "display".to_owned());
        baseline.dyn_key.get_or_insert_with(|| "display".to_owned());
        baseline.revision = 1;
        baseline
    }

    pub fn from_timeline(timeline: &crate::state::TimelineState) -> BTreeMap<String, Self> {
        timeline
            .params_by_root_track
            .iter()
            .map(|(root, params)| (root.clone(), Self::from_params(params)))
            .collect()
    }

    pub fn frame_count(&self) -> usize {
        self.pitch_orig
            .len()
            .max(self.pitch_target.len())
            .max(self.dyn_orig.len())
            .max(self.tension_orig.len())
    }

    pub fn is_ready(&self) -> bool {
        self.ready
            && self.frame_period_ms.is_finite()
            && (0.1..=1000.0).contains(&self.frame_period_ms)
            && (self.pitch_key.is_some()
                || !self.pitch_orig.is_empty()
                || !self.pitch_target.is_empty())
    }

    pub fn is_ready_for(&self, key: &str) -> bool {
        match key {
            PITCH_EDIT => self.is_ready(),
            EXTRA_DYN => self.ready && self.dyn_key.is_some(),
            TENSION_EDIT => self.ready,
            _ => self.ready,
        }
    }

    /// 基线层的显示数组：未编辑帧就是派生原始值。
    pub fn display(&self) -> TrackParamsState {
        let count = self.frame_count();
        let pitch_orig = if self.pitch_orig.is_empty() && !self.pitch_target.is_empty() {
            self.pitch_target.clone()
        } else {
            self.pitch_orig.clone()
        };
        let mut params = TrackParamsState {
            frame_period_ms: self.frame_period_ms,
            pitch_orig: pitch_orig.clone(),
            dyn_orig: self.dyn_orig.clone(),
            tension_orig: self.tension_orig.clone(),
            pitch_orig_key: self.pitch_key.clone(),
            dyn_orig_key: self.dyn_key.clone(),
            ..Default::default()
        };
        params.pitch_edit =
            if self.pitch_target.len() == pitch_orig.len() && !self.pitch_target.is_empty() {
                self.pitch_target.clone()
            } else {
                pitch_orig.clone()
            };
        params.tension_edit = vec![0.0; count];
        params
            .extra_curves
            .insert("dyn".to_owned(), vec![-1.0; count]);
        params
    }
}

/// 由权威 + 基线合成唯一显示视图。纯函数：不读设置、不写状态、不依赖调用顺序。
pub fn compose_root_view(
    root: &str,
    authority: &ParamAuthority,
    baseline: &RootBaseline,
    geometry_revision: u64,
) -> RootParamView {
    let mut params = baseline.display();
    let fp = if authority.frame_period_ms.is_finite() && authority.frame_period_ms > 0.0 {
        authority.frame_period_ms
    } else {
        baseline.frame_period_ms
    };
    // 权威 run 可以比分析基线更长（先画线、后分析，或冷启动只有存档 run）。
    // 显示帧数必须由两者共同决定，否则 compose 会在 run 的第一帧前就 break。
    let authority_frames = authority
        .root(root)
        .map(|root_authority| {
            root_authority
                .runs
                .values()
                .flat_map(|runs| runs.iter())
                .map(|run| ((run.end_sec() * 1000.0 / fp.max(0.1)).ceil() as usize).saturating_add(1))
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0);
    // 【上限】run 的工程位置来自存档/宿主，理论上可以很大；显示数组必须按
    // 真实工程帧数分配，绝不能由一条越界的 run 决定长度（否则一次恢复就能
    // 申请出数 GB）。1M 帧与内核既有的"参数帧预算"同量级。
    const MAX_VIEW_FRAMES: usize = 1_000_000;
    let count = baseline
        .frame_count()
        .max(params.pitch_edit.len())
        .max(params.tension_edit.len())
        .max(authority_frames)
        .min(MAX_VIEW_FRAMES);
    if count > 0 {
        params.pitch_edit.resize(count, 0.0);
        params.tension_edit.resize(count, 0.0);
        params
            .extra_curves
            .insert("dyn".to_owned(), vec![-1.0; count]);
    }
    let mut edited: BTreeMap<String, Vec<bool>> = BTreeMap::new();
    let Some(root_authority) = authority.roots.get(root) else {
        params.pitch_edit_user_modified = false;
        return RootParamView {
            root: root.to_owned(),
            params,
            edited,
            authority_revision: authority.revision,
            baseline_revision: baseline.revision,
            geometry_revision,
        };
    };
    for (key, runs) in &root_authority.runs {
        if count == 0 {
            break;
        }
        let mut ordered = runs.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|run| (run.seq, run.id));
        let mask = edited
            .entry(key.clone())
            .or_insert_with(|| vec![false; count]);
        for run in ordered {
            for frame in 0..count {
                let sec = frame as f64 * fp / 1000.0;
                let Some(mut value) = run.value_at_project_sec(sec) else {
                    continue;
                };
                if !is_edited_value(key, value, default_value_for_key(key)) {
                    continue;
                }
                match key.as_str() {
                    PITCH_EDIT => params.pitch_edit[frame] = value,
                    TENSION_EDIT => params.tension_edit[frame] = value,
                    EXTRA_DYN => {
                        value = value.max(0.0);
                        params
                            .extra_curves
                            .entry("dyn".to_owned())
                            .or_insert_with(|| vec![-1.0; count])[frame] = value;
                    }
                    other => {
                        let name = other.strip_prefix("extra:").unwrap_or(other).to_owned();
                        params
                            .extra_curves
                            .entry(name)
                            .or_insert_with(|| vec![default_value_for_key(other); count])[frame] =
                            value;
                    }
                }
                mask[frame] = true;
            }
        }
    }
    params.pitch_edit_user_modified = root_authority.has_edits(PITCH_EDIT);
    RootParamView {
        root: root.to_owned(),
        params,
        edited,
        authority_revision: authority.revision,
        baseline_revision: baseline.revision,
        geometry_revision,
    }
}

/// 从根视图切出 Clip 局部数组。切窗而不重建曲线，是 App 根数组语义的直接对应。
pub fn compose_clip_params(
    view: &RootParamView,
    clip: &Clip,
    frame_period_ms: f64,
) -> TrackParamsState {
    let fp = if frame_period_ms.is_finite() && frame_period_ms > 0.0 {
        frame_period_ms
    } else {
        view.params.frame_period_ms
    };
    let start = ((clip.start_sec.max(0.0) * 1000.0 / fp).floor()) as usize;
    let count = ((clip.length_sec.max(0.0) * 1000.0 / fp).ceil() as usize).max(1);
    let end = start.saturating_add(count);
    let slice = |values: &[f32], pad: f32| -> Vec<f32> {
        (start..end)
            .map(|index| values.get(index).copied().unwrap_or(pad))
            .collect()
    };
    // 【空数组的语义是“这一层没有基线”，不是一列零】clip 局部数组是渲染器
    // 判断“是否有原始值”的唯一入口：把空的 dyn_orig 补成 0 会让 DYN 的 -1
    // 哨兵解析到 0 dB，整条 clip 被静音；pitch_orig 补零也会让“缺 F0”伪装成
    // 合法源曲线。非空但较短的曲线仍按 0 补齐，避免越界采样。
    let slice_optional = |values: &[f32]| -> Vec<f32> {
        if values.is_empty() {
            Vec::new()
        } else {
            slice(values, 0.0)
        }
    };
    let mut out = TrackParamsState {
        frame_period_ms: fp,
        pitch_orig: slice_optional(&view.params.pitch_orig),
        pitch_edit: slice(&view.params.pitch_edit, 0.0),
        pitch_edit_user_modified: view.params.pitch_edit_user_modified,
        tension_orig: slice_optional(&view.params.tension_orig),
        tension_edit: slice(&view.params.tension_edit, 0.0),
        ..Default::default()
    };
    out.dyn_orig = slice_optional(&view.params.dyn_orig);
    for (key, curve) in &view.params.extra_curves {
        out.extra_curves.insert(
            key.clone(),
            slice(curve, default_value_for_key(&canonical_param_key(key))),
        );
    }
    out.extra_params = view.params.extra_params.clone();
    out
}

fn validate_run(key: &str, run: &PlacedRun, default_value: f32) -> Result<(), InvalidRun> {
    if run.values.is_empty() {
        return Err(InvalidRun {
            reason: "empty run".into(),
        });
    }
    if run.values.len() > MAX_RUN_VALUES {
        return Err(InvalidRun {
            reason: "run exceeds frame budget".into(),
        });
    }
    if !run.placement.start_sec.is_finite() || run.placement.start_sec < 0.0 {
        return Err(InvalidRun {
            reason: "invalid run start".into(),
        });
    }
    if !run.placement.seconds_per_frame.is_finite() || run.placement.seconds_per_frame <= 0.0 {
        return Err(InvalidRun {
            reason: "invalid run seconds_per_frame".into(),
        });
    }
    if run
        .values
        .iter()
        .any(|value| !value.is_finite() || value.abs() > 10000.0)
    {
        return Err(InvalidRun {
            reason: "non-finite or out-of-range value".into(),
        });
    }
    if run
        .values
        .iter()
        .any(|value| !is_edited_value(key, *value, default_value))
    {
        return Err(InvalidRun {
            reason: "run contains a default / unedited value".into(),
        });
    }
    Ok(())
}

fn runs_signature(runs: &[PlacedRun]) -> Vec<(u64, u64, usize, u64, u64)> {
    runs.iter()
        .map(|run| {
            (
                run.id,
                run.seq,
                run.values.len(),
                run.placement.start_sec.to_bits(),
                run.placement.seconds_per_frame.to_bits(),
            )
        })
        .collect()
}

fn normalize_runs(runs: &mut Vec<PlacedRun>) {
    runs.sort_by_key(|run| (run.placement.start_sec.to_bits(), run.seq, run.id));
    runs.dedup_by(|a, b| {
        a.id == b.id && a.seq == b.seq && a.placement == b.placement && a.values == b.values
    });
}

/// 切掉 run 与 `[lo, hi)` 相交的样本；区间外的左右残段保留。
fn split_outside_range(run: PlacedRun, lo: f64, hi: f64, out: &mut Vec<PlacedRun>) {
    if run.values.is_empty() {
        return;
    }
    let run_lo = run.placement.start_sec;
    let run_hi = run.span_end_sec();
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

fn merge_root_authorities(target: &mut RootParamAuthority, source: RootParamAuthority) {
    for (key, mut runs) in source.runs {
        target.runs.entry(key).or_default().append(&mut runs);
    }
    for runs in target.runs.values_mut() {
        normalize_runs(runs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline(count: usize) -> RootBaseline {
        RootBaseline {
            frame_period_ms: 5.0,
            pitch_orig: vec![60.0; count],
            pitch_target: vec![60.0; count],
            dyn_orig: vec![0.5; count],
            tension_orig: vec![0.0; count],
            pitch_key: Some("pitch".into()),
            dyn_key: Some("dyn".into()),
            ready: true,
            revision: 1,
            analysis_pending: false,
        }
    }

    #[test]
    fn empty_authority_displays_the_current_baseline() {
        let authority = ParamAuthority::default();
        let view = compose_root_view("root", &authority, &baseline(8), 0);
        assert_eq!(view.params.pitch_edit, vec![60.0; 8]);
        assert_eq!(view.params.tension_edit, vec![0.0; 8]);
        assert_eq!(view.params.extra_curves["dyn"], vec![-1.0; 8]);
        assert!(!view.params.pitch_edit_user_modified);
    }

    #[test]
    fn write_pitch_and_dyn_composes_mask_and_display() {
        let mut authority = ParamAuthority::default();
        authority
            .write_range(
                "root",
                "pitch",
                2,
                &[64.0, 65.0, 66.0],
                &[true, true, true],
                0.0,
            )
            .unwrap();
        authority
            .write_range("root", "dyn", 4, &[0.25], &[true], -1.0)
            .unwrap();
        let view = compose_root_view("root", &authority, &baseline(8), 3);
        assert_eq!(&view.params.pitch_edit[2..5], &[64.0, 65.0, 66.0]);
        assert_eq!(view.params.extra_curves["dyn"][4], 0.25);
        assert_eq!(
            view.edited_mask("pitch"),
            [false, false, true, true, true, false, false, false]
        );
        assert_eq!(
            view.edited_mask("dyn"),
            [false, false, false, false, true, false, false, false]
        );
        assert!(view.params.pitch_edit_user_modified);
        assert_eq!(view.authority_revision, authority.revision);
        assert_eq!(view.geometry_revision, 3);
    }

    #[test]
    fn default_values_do_not_create_runs() {
        let mut authority = ParamAuthority::default();
        authority
            .write_range(
                "root",
                "pitch",
                0,
                &[0.0, 5.0, 0.0],
                &[true, true, true],
                0.0,
            )
            .unwrap();
        assert_eq!(authority.root("root").unwrap().runs[PITCH_EDIT].len(), 1);
        authority.restore_range("root", "pitch", 0, 10);
        assert!(authority.roots.is_empty());
    }

    #[test]
    fn later_seq_wins_on_overlap_and_replacement_removes_old_edits() {
        let mut authority = ParamAuthority::default();
        authority
            .write_range("root", "pitch", 0, &[61.0, 62.0, 63.0], &[true; 3], 0.0)
            .unwrap();
        authority
            .write_range("root", "pitch", 1, &[70.0], &[true], 0.0)
            .unwrap();
        let view = compose_root_view("root", &authority, &baseline(5), 0);
        assert_eq!(&view.params.pitch_edit[..3], &[61.0, 70.0, 63.0]);
        assert!(view.edited_mask("pitch")[0]);
    }

    #[test]
    fn restore_returns_frames_to_baseline_and_preserves_surrounding_edits() {
        let mut authority = ParamAuthority::default();
        authority
            .write_range("root", "pitch", 0, &[61.0; 6], &[true; 6], 0.0)
            .unwrap();
        authority.restore_range("root", "pitch", 2, 2);
        let view = compose_root_view("root", &authority, &baseline(8), 0);
        assert_eq!(
            &view.params.pitch_edit[..6],
            &[61.0, 61.0, 60.0, 60.0, 61.0, 61.0]
        );
        assert_eq!(
            view.edited_mask("pitch")[..6],
            [true, true, false, false, true, true]
        );
    }

    #[test]
    fn empty_range_is_noop_and_oversize_is_rejected() {
        let mut authority = ParamAuthority::default();
        let revision = authority.revision;
        assert_eq!(
            authority
                .write_range("root", "pitch", 3, &[], &[], 0.0)
                .unwrap(),
            revision
        );
        let values = vec![61.0; MAX_RUN_VALUES + 1];
        let present = vec![true; MAX_RUN_VALUES + 1];
        assert!(authority
            .write_range("root", "pitch", 0, &values, &present, 0.0)
            .is_err());
        assert_eq!(authority.revision, revision);
    }

    #[test]
    fn content_hash_is_stable_and_revision_is_monotonic() {
        let mut authority = ParamAuthority::default();
        let before = authority.content_hash();
        authority
            .write_range("root", "pitch", 0, &[61.0], &[true], 0.0)
            .unwrap();
        assert_ne!(before, authority.content_hash());
        let revision = authority.revision;
        authority.restore_range("none", "pitch", 0, 10);
        assert_eq!(authority.revision, revision, "no-op restore must not bump");
        authority.restore_range("root", "pitch", 0, 1);
        assert!(authority.revision > revision);
    }

    #[test]
    fn a_bad_run_is_quarantined_alone() {
        let mut authority = ParamAuthority::default();
        authority
            .write_range("root", "pitch", 0, &[61.0], &[true], 0.0)
            .unwrap();
        authority.root_mut("root").runs.insert(
            "pitch_edit".into(),
            vec![PlacedRun::new(
                9,
                9,
                vec![0.0],
                RunPlacement {
                    start_sec: 0.0,
                    seconds_per_frame: 0.005,
                },
            )],
        );
        let report = authority.sanitize();
        assert_eq!(report.dropped_runs, 1);
        assert!(authority.roots.is_empty());
    }
}
