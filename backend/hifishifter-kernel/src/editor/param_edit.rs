//! 参数编辑的稀疏权威表示：只记录用户显式改过的帧。
//!
//! 分析派生的 pitch_orig / dyn_orig / tension_orig 与用户编辑共用同一个
//! TrackParamsState，但二者的生命周期相反：前者每次几何变化都要重新分析，后者
//! 必须跨越移动、拉伸、裁剪和分割保持源域身份。把它们都当成整条曲线数组存储，
//! 任何一次几何刷新都会用旧的分析快照覆盖新的分析结果，用户看到的就是参数线回到原值。
//!
//! 本模块把用户显式编辑收敛为稀疏的连续 run：未出现的帧永远没有编辑，显示时再
//! 与当前分析基线合成。独立 App 与 ARA 插件共用同一套提取、重映射和合成原语。

use crate::state::TrackParamsState;
use serde::{Deserialize, Serialize};

/// 一条连续的用户编辑 run（闭区间，按帧索引）。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EditRun {
    pub first_frame: usize,
    pub values: Vec<f32>,
}

impl EditRun {
    pub fn end_frame(&self) -> usize {
        self.first_frame
            .saturating_add(self.values.len().saturating_sub(1))
    }

    pub fn contains(&self, frame: f64) -> bool {
        frame.is_finite() && frame >= self.first_frame as f64 && frame <= self.end_frame() as f64
    }

    /// 在 run 内线性取样。绝不跨 run 插值：两个 run 之间的帧是“没有编辑”，
    /// 用相邻 run 的端点补洞会把用户画的断点连成一条线。
    pub fn sample(&self, frame: f64, key: &str) -> Option<f32> {
        if self.values.is_empty() || !self.contains(frame) {
            return None;
        }
        let index = frame - self.first_frame as f64;
        let lo = index.floor().clamp(0.0, (self.values.len() - 1) as f64) as usize;
        let hi = (lo + 1).min(self.values.len() - 1);
        let fraction = (index - lo as f64).clamp(0.0, 1.0) as f32;
        let a = self.values[lo];
        let b = self.values[hi];
        if key == "pitch_edit" && (a <= 0.0 || b <= 0.0) {
            Some(a.max(b).max(0.0))
        } else {
            Some(a + (b - a) * fraction)
        }
    }
}

/// 一条用户显式编辑曲线，按源帧（插件）或工程帧（独立 App）索引。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EditCurve {
    pub frame_ms: f64,
    pub runs: Vec<EditRun>,
}

impl EditCurve {
    pub fn new(frame_ms: f64) -> Self {
        Self {
            frame_ms,
            runs: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    pub fn end_frame(&self) -> Option<usize> {
        self.runs.last().map(EditRun::end_frame)
    }

    pub fn value_at(&self, frame: f64, key: &str) -> Option<f32> {
        if !frame.is_finite() {
            return None;
        }
        let index = self
            .runs
            .partition_point(|run| (run.end_frame() as f64) < frame);
        self.runs.get(index).and_then(|run| run.sample(frame, key))
    }

    /// 展开成逐帧 map，便于捕获时做并集/删除证据。
    pub fn frame_values(&self) -> std::collections::BTreeMap<usize, f32> {
        let mut out = std::collections::BTreeMap::new();
        for run in &self.runs {
            for (offset, value) in run.values.iter().copied().enumerate() {
                out.insert(run.first_frame + offset, value);
            }
        }
        out
    }

    /// 从逐帧 map 重建稀疏 run（只保留默认值之外的样本）。
    pub fn from_frame_values(
        frame_ms: f64,
        key: &str,
        values: std::collections::BTreeMap<usize, f32>,
    ) -> Self {
        let mut curve = Self::new(frame_ms);
        for (frame, value) in values {
            if !is_edited_value(key, value, None, Some(default_pad_for_key(key))) {
                continue;
            }
            match curve.runs.last_mut() {
                Some(run) if run.end_frame().saturating_add(1) == frame => {
                    run.values.push(value);
                }
                _ => curve.runs.push(EditRun {
                    first_frame: frame,
                    values: vec![value],
                }),
            }
        }
        curve
    }

    pub fn covers(&self, first: usize, last: usize) -> bool {
        !self.runs.is_empty()
            && self
                .runs
                .iter()
                .any(|run| run.first_frame <= first && run.end_frame() >= last)
    }

    /// 用一批逐帧值替换 [first_frame, first_frame + values.len())。
    ///
    /// 只保留“不是默认值”的帧。用户把某帧画回默认值等价于删除该帧的编辑；
    /// 这是独立 App 既有语义（保存层与 compact_params 都按此精简）。
    pub fn replace_range(&mut self, first_frame: usize, values: &[f32], key: &str) {
        if values.is_empty() {
            return;
        }
        let last = first_frame.saturating_add(values.len().saturating_sub(1));
        let mut next = Vec::<EditRun>::new();
        for run in &self.runs {
            if run.end_frame() < first_frame || run.first_frame > last {
                next.push(run.clone());
                continue;
            }
            if run.first_frame < first_frame {
                next.push(EditRun {
                    first_frame: run.first_frame,
                    values: run.values[..first_frame - run.first_frame].to_vec(),
                });
            }
            if run.end_frame() > last {
                let skip = last + 1 - run.first_frame;
                next.push(EditRun {
                    first_frame: last + 1,
                    values: run.values[skip..].to_vec(),
                });
            }
        }
        for (offset, value) in values.iter().copied().enumerate() {
            if !is_edited_value(key, value, None, None) {
                continue;
            }
            let frame = first_frame + offset;
            match next.last_mut() {
                Some(run) if run.end_frame().saturating_add(1) == frame => {
                    run.values.push(value);
                }
                _ => next.push(EditRun {
                    first_frame: frame,
                    values: vec![value],
                }),
            }
        }
        next.retain(|run| !run.values.is_empty());
        next.sort_by_key(|run| run.first_frame);
        self.runs = next;
    }

    /// 删除一段帧上的编辑。
    pub fn remove_range(&mut self, first_frame: usize, last_frame: usize) {
        if first_frame > last_frame {
            return;
        }
        let mut next = Vec::new();
        for run in &self.runs {
            if run.end_frame() < first_frame || run.first_frame > last_frame {
                next.push(run.clone());
                continue;
            }
            if run.first_frame < first_frame {
                next.push(EditRun {
                    first_frame: run.first_frame,
                    values: run.values[..first_frame - run.first_frame].to_vec(),
                });
            }
            if run.end_frame() > last_frame {
                let skip = last_frame + 1 - run.first_frame;
                next.push(EditRun {
                    first_frame: last_frame + 1,
                    values: run.values[skip..].to_vec(),
                });
            }
        }
        next.retain(|run| !run.values.is_empty());
        self.runs = next;
    }

    /// 平移所有帧。返回 false 表示越界（调用方应保留旧曲线）。
    pub fn shift_by(&mut self, delta_frames: i64) -> bool {
        let mut shifted = Vec::with_capacity(self.runs.len());
        for run in &self.runs {
            let first = run.first_frame as i64 + delta_frames;
            if first < 0 {
                return false;
            }
            shifted.push(EditRun {
                first_frame: first as usize,
                values: run.values.clone(),
            });
        }
        self.runs = shifted;
        true
    }

    /// 把一段源窗口的编辑按比例映射到新窗口（拉伸）。
    ///
    /// 目标帧在旧窗口之外时不产生编辑；旧窗口内按比例取样。返回值仍是稀疏 run。
    pub fn stretch_window(
        &self,
        old_first: f64,
        old_last: f64,
        new_first: f64,
        new_last: f64,
        key: &str,
    ) -> Self {
        if !(old_first.is_finite()
            && old_last.is_finite()
            && new_first.is_finite()
            && new_last.is_finite())
            || old_last <= old_first
            || new_last <= new_first
        {
            return self.clone();
        }
        let first = new_first.floor().max(0.0) as usize;
        let last = new_last.ceil().max(0.0) as usize;
        if last >= 1_000_000 {
            return self.clone();
        }
        let mut out = Self::new(self.frame_ms);
        let mut start: Option<usize> = None;
        let mut values = Vec::new();
        for frame in first..=last {
            let fraction = (frame as f64 - new_first) / (new_last - new_first);
            let value = if (0.0..=1.0).contains(&fraction) {
                let source = old_first + fraction * (old_last - old_first);
                self.value_at(source, key)
            } else {
                None
            };
            match (start, value) {
                (None, Some(value)) => {
                    start = Some(frame);
                    values.push(value);
                }
                (Some(_), Some(value)) => values.push(value),
                (Some(begin), None) => {
                    out.runs.push(EditRun {
                        first_frame: begin,
                        values: std::mem::take(&mut values),
                    });
                    start = None;
                }
                (None, None) => {}
            }
        }
        if let Some(begin) = start {
            out.runs.push(EditRun {
                first_frame: begin,
                values,
            });
        }
        out
    }

    /// 反转窗口内的帧顺序（倒放）。
    pub fn reverse_window(&self, first: f64, last: f64, key: &str) -> Self {
        if last <= first {
            return self.clone();
        }
        let begin = first.floor().max(0.0) as usize;
        let end = last.ceil().max(0.0) as usize;
        if end >= 1_000_000 {
            return self.clone();
        }
        let mut out = Self::new(self.frame_ms);
        let mut start: Option<usize> = None;
        let mut values = Vec::new();
        for frame in begin..=end {
            let mirrored = first + (last - frame as f64);
            let value = self.value_at(mirrored, key);
            match (start, value) {
                (None, Some(value)) => {
                    start = Some(frame);
                    values.push(value);
                }
                (Some(_), Some(value)) => values.push(value),
                (Some(begin), None) => {
                    out.runs.push(EditRun {
                        first_frame: begin,
                        values: std::mem::take(&mut values),
                    });
                    start = None;
                }
                (None, None) => {}
            }
        }
        if let Some(begin) = start {
            out.runs.push(EditRun {
                first_frame: begin,
                values,
            });
        }
        out
    }
}

/// 一条轨道的全部显式编辑；frame_ms 与 TrackParamsState.frame_period_ms 同源。
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct EditCurveSet {
    pub frame_ms: f64,
    pub curves: std::collections::BTreeMap<String, EditCurve>,
}

impl EditCurveSet {
    pub fn is_empty(&self) -> bool {
        self.curves.is_empty()
    }

    pub fn get(&self, key: &str) -> Option<&EditCurve> {
        self.curves.get(key)
    }

    pub fn has_pitch(&self) -> bool {
        self.curves
            .get("pitch_edit")
            .is_some_and(|curve| !curve.is_empty())
    }
}

/// 这个值是否表达“用户显式编辑”。
///
/// orig 只用于 pitch：未编辑的 pitch_edit 是 pitch_orig 的逐帧副本，数值相等
/// 不等于用户意图。pad 是曲线无数据时的填充值。
pub fn is_edited_value(key: &str, value: f32, orig: Option<f32>, pad: Option<f32>) -> bool {
    if !value.is_finite() {
        return false;
    }
    match key {
        "pitch_edit" => {
            if value <= 0.0 {
                return false;
            }
            orig.is_none_or(|orig| !(orig.is_finite() && orig > 0.0) || (value - orig).abs() > 1e-3)
        }
        "tension_edit" => value.abs() > 1e-6,
        "extra:dyn" => value >= 0.0,
        _ => {
            let pad = pad.unwrap_or_else(|| default_pad_for_key(key));
            (value - pad).abs() > 1e-6
        }
    }
}

/// 与图集 / 渲染参数层共用的“无数据”值。
pub fn default_pad_for_key(key: &str) -> f32 {
    if key == "extra:volume" || key == "extra:breath_gain" {
        1.0
    } else if key == "extra:dyn" {
        -1.0
    } else {
        0.0
    }
}

fn scan_curve<F>(
    values: &[f32],
    key: &str,
    frame_ms: f64,
    curves: &mut std::collections::BTreeMap<String, EditCurve>,
    edited: F,
) where
    F: Fn(usize, f32) -> bool,
{
    let mut curve = EditCurve::new(frame_ms);
    let mut start: Option<usize> = None;
    for (frame, value) in values.iter().copied().enumerate() {
        let hit = edited(frame, value);
        match (start, hit) {
            (None, true) => start = Some(frame),
            (Some(begin), false) => {
                curve.runs.push(EditRun {
                    first_frame: begin,
                    values: values[begin..frame].to_vec(),
                });
                start = None;
            }
            _ => {}
        }
    }
    if let Some(begin) = start {
        curve.runs.push(EditRun {
            first_frame: begin,
            values: values[begin..].to_vec(),
        });
    }
    if !curve.is_empty() {
        curves.insert(key.to_owned(), curve);
    }
}

/// 从当前轨道视图提取“用户显式编辑”。
///
/// 绝不读取 pitch_orig / dyn_orig / tension_orig 作为可写曲线：它们只作为 pitch
/// 判定的参考（未编辑的 pitch_edit 是它的副本）。
pub fn extract_edit_curves(params: &TrackParamsState) -> EditCurveSet {
    let frame_ms = if params.frame_period_ms.is_finite() && params.frame_period_ms > 0.0 {
        params.frame_period_ms
    } else {
        5.0
    };
    let mut curves = std::collections::BTreeMap::new();

    if params.pitch_edit_user_modified {
        scan_curve(
            &params.pitch_edit,
            "pitch_edit",
            frame_ms,
            &mut curves,
            |frame, value| {
                is_edited_value(
                    "pitch_edit",
                    value,
                    params.pitch_orig.get(frame).copied(),
                    None,
                )
            },
        );
    }
    scan_curve(
        &params.tension_edit,
        "tension_edit",
        frame_ms,
        &mut curves,
        |_, value| is_edited_value("tension_edit", value, None, None),
    );
    for (name, values) in &params.extra_curves {
        let key = format!("extra:{name}");
        let pad = default_pad_for_key(&key);
        scan_curve(values, &key, frame_ms, &mut curves, |_, value| {
            is_edited_value(&key, value, None, Some(pad))
        });
    }

    EditCurveSet { frame_ms, curves }
}

/// 把一份编辑值合成到显示数组：未编辑帧取 defaults，编辑帧覆盖。
///
/// map_frame 把输出帧映射到编辑曲线的帧号；返回的布尔数组标记哪些输出帧来自
/// 显式编辑。map_frame 返回 None 表示该帧没有源映射（保持默认）。
pub fn compose_curve<F>(
    key: &str,
    defaults: &[f32],
    edit: Option<&EditCurve>,
    mut map_frame: F,
) -> (Vec<f32>, Vec<bool>)
where
    F: FnMut(usize) -> Option<f64>,
{
    let mut values = defaults.to_vec();
    let mut edited = vec![false; values.len()];
    let Some(edit) = edit else {
        return (values, edited);
    };
    for frame in 0..values.len() {
        let Some(source_frame) = map_frame(frame) else {
            continue;
        };
        if let Some(value) = edit.value_at(source_frame, key) {
            values[frame] = value;
            edited[frame] = true;
        }
    }
    (values, edited)
}

/// 把一条曲线投影到另一组帧上，只保留有编辑的区间。独立 App 的拉伸路径可复用。
pub fn resample_curve<F>(
    curve: &EditCurve,
    key: &str,
    out_first: usize,
    out_last: usize,
    mut map_frame: F,
) -> EditCurve
where
    F: FnMut(usize) -> Option<f64>,
{
    let mut out = EditCurve::new(curve.frame_ms);
    let mut start: Option<usize> = None;
    let mut values = Vec::new();
    for frame in out_first..=out_last {
        let hit = map_frame(frame).and_then(|source| curve.value_at(source, key));
        match (start, hit) {
            (None, Some(value)) => {
                start = Some(frame);
                values.push(value);
            }
            (Some(_), Some(value)) => values.push(value),
            (Some(begin), None) => {
                out.runs.push(EditRun {
                    first_frame: begin,
                    values: std::mem::take(&mut values),
                });
                start = None;
            }
            (None, None) => {}
        }
    }
    if let Some(begin) = start {
        out.runs.push(EditRun {
            first_frame: begin,
            values,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(frame_ms: f64) -> TrackParamsState {
        TrackParamsState {
            frame_period_ms: frame_ms,
            ..Default::default()
        }
    }

    #[test]
    fn extracts_only_pitch_frames_that_differ_from_original() {
        let mut p = params(5.0);
        p.pitch_orig = vec![60.0, 61.0, 62.0, 0.0];
        p.pitch_edit = vec![60.0, 63.0, 62.0, 64.0];
        p.pitch_edit_user_modified = true;
        let edits = extract_edit_curves(&p);
        let pitch = edits.get("pitch_edit").unwrap();
        assert_eq!(pitch.runs.len(), 2);
        assert_eq!(pitch.runs[0].first_frame, 1);
        assert_eq!(pitch.runs[0].values, vec![63.0]);
        assert_eq!(pitch.runs[1].first_frame, 3);
        assert_eq!(pitch.runs[1].values, vec![64.0]);
    }

    #[test]
    fn pitch_edit_equal_to_orig_is_not_an_edit_even_when_flag_is_set() {
        let mut p = params(5.0);
        p.pitch_orig = vec![60.0; 4];
        p.pitch_edit = vec![60.0; 4];
        p.pitch_edit_user_modified = true;
        assert!(extract_edit_curves(&p).is_empty());
    }

    #[test]
    fn extracts_dyn_tension_and_extras_with_their_own_pads() {
        let mut p = params(5.0);
        p.extra_curves.insert("dyn".into(), vec![-1.0, 0.5, 0.75]);
        p.extra_curves.insert("volume".into(), vec![1.0, 0.8, 1.0]);
        p.tension_edit = vec![0.0, 0.25, 0.0];
        let edits = extract_edit_curves(&p);
        assert_eq!(
            edits.get("extra:dyn").unwrap().runs[0].values,
            vec![0.5, 0.75]
        );
        assert_eq!(edits.get("extra:volume").unwrap().runs[0].first_frame, 1);
        assert_eq!(
            edits.get("tension_edit").unwrap().runs[0].values,
            vec![0.25]
        );
    }

    #[test]
    fn runs_do_not_interpolate_across_gaps() {
        let curve = EditCurve {
            frame_ms: 5.0,
            runs: vec![
                EditRun {
                    first_frame: 0,
                    values: vec![1.0],
                },
                EditRun {
                    first_frame: 10,
                    values: vec![3.0],
                },
            ],
        };
        assert_eq!(curve.value_at(0.0, "tension_edit"), Some(1.0));
        assert_eq!(curve.value_at(5.0, "tension_edit"), None);
        assert_eq!(curve.value_at(10.0, "tension_edit"), Some(3.0));
    }

    #[test]
    fn replace_range_splits_and_removes_only_covered_frames() {
        let mut curve = EditCurve {
            frame_ms: 5.0,
            runs: vec![EditRun {
                first_frame: 10,
                values: vec![1.0; 10],
            }],
        };
        curve.remove_range(12, 15);
        assert_eq!(curve.runs[0].first_frame, 10);
        assert_eq!(curve.runs[0].values.len(), 2);
        assert_eq!(curve.runs[1].first_frame, 16);
        assert_eq!(curve.runs[1].values.len(), 4);
    }

    #[test]
    fn compose_marks_only_explicit_frames() {
        let edit = EditCurve {
            frame_ms: 5.0,
            runs: vec![EditRun {
                first_frame: 2,
                values: vec![9.0],
            }],
        };
        let (values, edited) = compose_curve("tension_edit", &[0.0; 5], Some(&edit), |frame| {
            Some(frame as f64)
        });
        assert_eq!(values, vec![0.0, 0.0, 9.0, 0.0, 0.0]);
        assert_eq!(edited, vec![false, false, true, false, false]);
    }

    #[test]
    fn stretch_window_keeps_edits_sparse_and_scales_the_run() {
        let curve = EditCurve {
            frame_ms: 5.0,
            runs: vec![EditRun {
                first_frame: 0,
                values: vec![60.0, 62.0, 64.0],
            }],
        };
        let stretched = curve.stretch_window(0.0, 2.0, 0.0, 4.0, "pitch_edit");
        assert_eq!(stretched.value_at(0.0, "pitch_edit"), Some(60.0));
        assert_eq!(stretched.value_at(1.0, "pitch_edit"), Some(61.0));
        assert_eq!(stretched.value_at(2.0, "pitch_edit"), Some(62.0));
        assert_eq!(stretched.value_at(3.0, "pitch_edit"), Some(63.0));
        assert_eq!(stretched.value_at(4.0, "pitch_edit"), Some(64.0));
    }

    #[test]
    fn reverse_window_mirrors_run_order() {
        let curve = EditCurve {
            frame_ms: 5.0,
            runs: vec![EditRun {
                first_frame: 0,
                values: vec![60.0, 62.0, 64.0],
            }],
        };
        let reversed = curve.reverse_window(0.0, 2.0, "pitch_edit");
        assert_eq!(reversed.value_at(0.0, "pitch_edit"), Some(64.0));
        assert_eq!(reversed.value_at(1.0, "pitch_edit"), Some(62.0));
        assert_eq!(reversed.value_at(2.0, "pitch_edit"), Some(60.0));
    }
}
