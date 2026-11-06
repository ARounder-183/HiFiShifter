//! 参数编辑的稀疏权威表示：只记录用户显式改过的帧。
//!
//! 分析派生的 pitch_orig / dyn_orig / tension_orig 与用户编辑共用同一个
//! TrackParamsState，但二者的生命周期相反：前者每次几何变化都要重新分析，后者
//! 必须跨越移动、拉伸、裁剪和分割保持源域身份。把它们都当成整条曲线数组存储，
//! 任何一次几何刷新都会用旧的分析快照覆盖新的分析结果，用户看到的就是参数线回到原值。
//!
//! 本模块把用户显式编辑收敛为稀疏的连续 run：未出现的帧永远没有编辑，显示时再
//! 与当前分析基线合成。独立 App 与 ARA 插件共用同一套提取、重映射和合成原语。
//!
//! 【为什么 run 需要锚】「锁定参数线」的两种语义只差一件事：这条编辑**跟着素材走**
//! 还是**钉在工程位置**。旧模型把这件事存在权威级（一份 `Placement`），于是"移动之后
//! 再画一笔"这一笔没有自己的位置可钉 —— 它只能继承整份权威的冻结窗口，而那窗口
//! 描述的是**上一次**的几何。锚把它下沉到 run：每个 run 记住自己被画在哪一段工程
//! 时间上，锁定 = 没有锚（跟随素材），不锁定 = 有锚（钉住）。开关因此是纯元数据
//! 操作：只改 run 的 `anchor` 字段，源帧与数值一格不动。

use crate::state::TrackParamsState;
use serde::{Deserialize, Serialize};

/// 一条编辑 run 在工程时间上的**锚**：它被画在哪一段工程时间 / 哪一段源时间上。
///
/// 【为什么锚是纯数据】它只描述一次映射（工程窗口 ⇄ 源窗口），任何时刻都能重新算出
/// 一一对应，因此"钉住 / 不钉住"是可逆的：拨开关只改 `EditRun::anchor` 这个下标，
/// 曲线本身一个字节都不动。旧模型把"钉在哪"存成权威级的 `frozen: Vec<RegionGeometry>`，
/// 于是它只能描述**建立权威那一刻**的几何 —— 之后新画的笔画无处可钉，只能被解释成
/// "跟随素材"，这正是"移动之后再画一笔，线落在错地方"的成因。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EditAnchor {
    /// 这条锚落在哪条根轨上（工程时间的坐标属于一条根轨）。
    pub root: String,
    pub project_start_sec: f64,
    pub project_duration_sec: f64,
    pub source_start_sec: f64,
    pub source_duration_sec: f64,
    /// 倒放：工程时间增、源位置减。
    #[serde(default)]
    pub reversed: bool,
}

impl EditAnchor {
    /// 窗口是否可用于映射（有限、正长度）。
    pub fn is_valid(&self) -> bool {
        [
            self.project_start_sec,
            self.project_duration_sec,
            self.source_start_sec,
            self.source_duration_sec,
        ]
        .into_iter()
        .all(f64::is_finite)
            && self.project_duration_sec > 0.0
            && self.source_duration_sec > 0.0
    }

    /// 工程秒 → 源秒。窗口外返回 `None`（这一帧没有编辑），绝不钳到端点 ——
    /// 钳位会把第一格编辑复制到整条空白区。
    pub fn project_to_source(&self, project_sec: f64) -> Option<f64> {
        if !self.is_valid() || !project_sec.is_finite() {
            return None;
        }
        let end = self.project_start_sec + self.project_duration_sec;
        // 右端是**闭**的：输出的最后一格常常正好落在窗口终点上（ceil 取整）。
        if project_sec < self.project_start_sec - 1e-9 || project_sec > end + 1e-9 {
            return None;
        }
        let clamped = project_sec.clamp(self.project_start_sec, end);
        let fraction = (clamped - self.project_start_sec) / self.project_duration_sec;
        Some(if self.reversed {
            self.source_start_sec + self.source_duration_sec * (1.0 - fraction)
        } else {
            self.source_start_sec + self.source_duration_sec * fraction
        })
    }

    /// 源秒 → 工程秒。源窗口外返回 `None`。
    pub fn source_to_project(&self, source_sec: f64) -> Option<f64> {
        if !self.is_valid() || !source_sec.is_finite() {
            return None;
        }
        let end = self.source_start_sec + self.source_duration_sec;
        if source_sec < self.source_start_sec - 1e-9 || source_sec > end + 1e-9 {
            return None;
        }
        let clamped = source_sec.clamp(self.source_start_sec, end);
        let fraction = (clamped - self.source_start_sec) / self.source_duration_sec;
        Some(if self.reversed {
            self.project_start_sec + self.project_duration_sec * (1.0 - fraction)
        } else {
            self.project_start_sec + self.project_duration_sec * fraction
        })
    }

    /// 同一个窗口（同一根、同一段工程/源区间、同方向）。
    pub fn equivalent(&self, other: &Self) -> bool {
        self.root == other.root
            && self.reversed == other.reversed
            && [
                (self.project_start_sec, other.project_start_sec),
                (self.project_duration_sec, other.project_duration_sec),
                (self.source_start_sec, other.source_start_sec),
                (self.source_duration_sec, other.source_duration_sec),
            ]
            .into_iter()
            .all(|(a, b)| (a - b).abs() <= 1e-9)
    }
}

/// 工程秒 → 源秒（显式给定锚）。
pub fn anchor_project_to_source(anchor: &EditAnchor, project_sec: f64) -> Option<f64> {
    anchor.project_to_source(project_sec)
}

/// 源秒 → 工程秒（显式给定锚）。
pub fn anchor_source_to_project(anchor: &EditAnchor, source_sec: f64) -> Option<f64> {
    anchor.source_to_project(source_sec)
}

/// 一条连续的用户编辑 run（闭区间，按帧索引）。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EditRun {
    pub first_frame: usize,
    pub values: Vec<f32>,
    /// 该 run 的锚（索引到 `EditCurveSet::anchors`）；`None` = 跟随素材。
    ///
    /// 【为什么是下标而不是内联结构】同一条 clip 上的一笔通常覆盖几十个 run
    /// （pitch / tension / dyn 各一条曲线），它们共享同一个窗口。内联会把同一份
    /// 窗口数据重复几十遍，而 `u32` 让"这一笔钉在哪"变成一个可以整体重指的指针 ——
    /// 锁定开关因此是 O(run 数) 的下标清零，不是数据搬运。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<u32>,
}

impl EditRun {
    /// 跟随素材的 run（独立 App 与锁定模式都用它）。
    pub fn new(first_frame: usize, values: Vec<f32>) -> Self {
        Self {
            first_frame,
            values,
            anchor: None,
        }
    }

    /// 钉在某个锚上的 run。
    pub fn anchored(first_frame: usize, values: Vec<f32>, anchor: u32) -> Self {
        Self {
            first_frame,
            values,
            anchor: Some(anchor),
        }
    }

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

    /// 末端帧（**所有层**的最大值）。
    ///
    /// 【为什么不是 `runs.last()`】run 按 `(first_frame, anchor)` 排序，跨层可以重叠；
    /// 最后一条的末端不一定是整条曲线的末端。用 `max` 才是"这条曲线覆盖到哪"。
    pub fn end_frame(&self) -> Option<usize> {
        self.runs.iter().map(EditRun::end_frame).max()
    }

    /// 一条曲线里属于某个锚层（`None` = 跟随素材）的 run。
    ///
    /// 【为什么按层取而不是"取全部"】同一个源帧可能同时存在"钉住的旧笔画"与
    /// "跟随素材的新笔画"。合成时必须分两层各自映射，否则其中一层会被另一层的
    /// 工程位置解释，用户看到的就是"拨一下开关，线跑到别处"。
    pub fn runs_in_layer(&self, anchor: Option<u32>) -> impl Iterator<Item = &EditRun> {
        self.runs.iter().filter(move |run| run.anchor == anchor)
    }

    /// 只保留某一层的 run（其它层原样丢弃）。
    pub fn layer(&self, anchor: Option<u32>) -> Self {
        Self {
            frame_ms: self.frame_ms,
            runs: self.runs_in_layer(anchor).cloned().collect(),
        }
    }

    /// 某一层里是否真的有一条编辑。
    pub fn has_layer(&self, anchor: Option<u32>) -> bool {
        self.runs_in_layer(anchor).next().is_some()
    }

    /// 任意层里取样（旧存档迁移与 App 用）。
    ///
    /// 【为什么倒序线性扫】`partition_point` 要求 run 按 `end_frame` 单调，而跨层重叠时
    /// 这个前提不成立（同一层的 run 会被别的层隔开），它会跳过真正覆盖该帧的那一条。
    /// 倒序找到的第一条是"后写入的那一段"，与"最新意图优先"一致。
    pub fn value_at(&self, frame: f64, key: &str) -> Option<f32> {
        if !frame.is_finite() {
            return None;
        }
        self.runs
            .iter()
            .rev()
            .find(|run| run.contains(frame))
            .and_then(|run| run.sample(frame, key))
    }

    /// 只在某一层里取样。
    ///
    /// 一层里的 run 通常只有几条（一笔 = 一条），线性扫描既正确又便宜；
    /// 跨层交错让任何基于"全局有序"的二分都失去前提。
    pub fn value_at_in_layer(&self, frame: f64, key: &str, anchor: Option<u32>) -> Option<f32> {
        if !frame.is_finite() {
            return None;
        }
        self.runs_in_layer(anchor)
            .find(|run| run.contains(frame))
            .and_then(|run| run.sample(frame, key))
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
        Self::from_frame_values_in_layer(frame_ms, key, None, values)
    }

    /// 从逐帧 map 重建**某一层**的稀疏 run。
    pub fn from_frame_values_in_layer(
        frame_ms: f64,
        key: &str,
        anchor: Option<u32>,
        values: std::collections::BTreeMap<usize, f32>,
    ) -> Self {
        let mut curve = Self::new(frame_ms);
        for (frame, value) in values {
            if !is_edited_value(key, value, None, Some(default_pad_for_key(key))) {
                continue;
            }
            match curve.runs.last_mut() {
                Some(run) if run.anchor == anchor && run.end_frame().saturating_add(1) == frame => {
                    run.values.push(value);
                }
                _ => curve.runs.push(EditRun {
                    first_frame: frame,
                    values: vec![value],
                    anchor,
                }),
            }
        }
        curve
    }

    /// 展开成逐帧 map，只取某一层。
    pub fn frame_values_in_layer(
        &self,
        anchor: Option<u32>,
    ) -> std::collections::BTreeMap<usize, f32> {
        let mut out = std::collections::BTreeMap::new();
        for run in self.runs_in_layer(anchor) {
            for (offset, value) in run.values.iter().copied().enumerate() {
                out.insert(run.first_frame + offset, value);
            }
        }
        out
    }

    /// 这条曲线用到的所有层：`None`（跟随素材）在最前，其余按锚下标升序。
    pub fn layers(&self) -> Vec<Option<u32>> {
        let mut out: Vec<Option<u32>> = Vec::new();
        for run in &self.runs {
            if !out.contains(&run.anchor) {
                out.push(run.anchor);
            }
        }
        out.sort_by_key(|anchor| (anchor.is_some(), anchor.unwrap_or(0)));
        out
    }

    /// 某一层的 run（便利入口；与 `runs_in_layer` 同义，供校验与诊断遍历）。
    pub fn layer_slice(&self, anchor: Option<u32>) -> Vec<&EditRun> {
        self.runs_in_layer(anchor).collect()
    }

    /// 规范化某一层：按帧排序、合并相邻、裁掉重叠（后写者胜）。
    ///
    /// 【为什么必须能裁】`set_lock_mode` 会把"无锚层"整体钉到某个锚上；若那个锚上
    /// 已经有笔画，两层就落在同一个层里。不裁的话 `validate` 会认为这条曲线非法，
    /// 而它本该是"用户最后画的那一笔赢"。
    pub fn normalize_layer(&mut self, anchor: Option<u32>) {
        let mut mine: Vec<EditRun> = self
            .runs
            .iter()
            .filter(|run| run.anchor == anchor)
            .cloned()
            .collect();
        mine.retain(|run| !run.values.is_empty());
        mine.sort_by_key(|run| run.first_frame);
        let mut merged: Vec<EditRun> = Vec::with_capacity(mine.len());
        for run in mine {
            let mut first = run.first_frame;
            let mut values = run.values;
            if let Some(previous) = merged.last_mut() {
                let previous_end = previous.end_frame();
                if first <= previous_end {
                    // 重叠：后写者胜 —— 裁掉被覆盖的头部。
                    let skip = previous_end + 1 - first;
                    if skip >= values.len() {
                        continue;
                    }
                    values.drain(..skip);
                    first = previous_end + 1;
                }
                if first == previous_end + 1 {
                    previous.values.extend(values);
                    continue;
                }
            }
            merged.push(EditRun {
                first_frame: first,
                values,
                anchor,
            });
        }
        self.runs.retain(|run| run.anchor != anchor);
        self.runs.extend(merged);
        self.runs.sort_by_key(|run| (run.first_frame, run.anchor));
    }

    /// 规范化所有层。
    pub fn normalize(&mut self) {
        for anchor in self.layers() {
            self.normalize_layer(anchor);
        }
    }

    /// 把给定源帧从**所有**层里删掉。
    ///
    /// 【为什么是"所有层"】用户在同一段素材上再画一次，意图是覆盖那几帧，不是
    /// 再叠一份。只清"当前层"会让旧锚层的同一批帧继续投影到另一个工程位置 ——
    /// 那正是"移动之后线还在旧地方"的形状。
    pub fn remove_frames_everywhere(&mut self, frames: &std::collections::BTreeSet<usize>) {
        if frames.is_empty() {
            return;
        }
        let mut next: Vec<EditRun> = Vec::with_capacity(self.runs.len());
        for run in &self.runs {
            let mut start: Option<usize> = None;
            let mut values: Vec<f32> = Vec::new();
            for (offset, value) in run.values.iter().copied().enumerate() {
                let frame = run.first_frame + offset;
                if frames.contains(&frame) {
                    if let Some(begin) = start.take() {
                        next.push(EditRun {
                            first_frame: begin,
                            values: std::mem::take(&mut values),
                            anchor: run.anchor,
                        });
                    }
                    continue;
                }
                if start.is_none() {
                    start = Some(frame);
                }
                values.push(value);
            }
            if let Some(begin) = start {
                next.push(EditRun {
                    first_frame: begin,
                    values,
                    anchor: run.anchor,
                });
            }
        }
        next.retain(|run| !run.values.is_empty());
        self.runs = next;
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
        self.replace_range_in_layer(first_frame, values, key, None);
    }

    /// 与 [`Self::replace_range`] 相同，但只动一层（其它层原样保留）。
    pub fn replace_range_in_layer(
        &mut self,
        first_frame: usize,
        values: &[f32],
        key: &str,
        anchor: Option<u32>,
    ) {
        if values.is_empty() {
            return;
        }
        let last = first_frame.saturating_add(values.len().saturating_sub(1));
        let mut next = Vec::<EditRun>::new();
        for run in &self.runs {
            if run.anchor != anchor || run.end_frame() < first_frame || run.first_frame > last {
                next.push(run.clone());
                continue;
            }
            if run.first_frame < first_frame {
                next.push(EditRun {
                    first_frame: run.first_frame,
                    values: run.values[..first_frame - run.first_frame].to_vec(),
                    anchor,
                });
            }
            if run.end_frame() > last {
                let skip = last + 1 - run.first_frame;
                next.push(EditRun {
                    first_frame: last + 1,
                    values: run.values[skip..].to_vec(),
                    anchor,
                });
            }
        }
        for (offset, value) in values.iter().copied().enumerate() {
            if !is_edited_value(key, value, None, None) {
                continue;
            }
            let frame = first_frame + offset;
            match next.last_mut() {
                Some(run) if run.anchor == anchor && run.end_frame().saturating_add(1) == frame => {
                    run.values.push(value);
                }
                _ => next.push(EditRun {
                    first_frame: frame,
                    values: vec![value],
                    anchor,
                }),
            }
        }
        next.retain(|run| !run.values.is_empty());
        next.sort_by_key(|run| (run.first_frame, run.anchor));
        self.runs = next;
    }

    /// 删除一段帧上的编辑（所有层）。
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
                    anchor: run.anchor,
                });
            }
            if run.end_frame() > last_frame {
                let skip = last_frame + 1 - run.first_frame;
                next.push(EditRun {
                    first_frame: last_frame + 1,
                    values: run.values[skip..].to_vec(),
                    anchor: run.anchor,
                });
            }
        }
        next.retain(|run| !run.values.is_empty());
        self.runs = next;
    }

    /// 删除某一层里的一段帧上的编辑。
    pub fn remove_range_in_layer(
        &mut self,
        first_frame: usize,
        last_frame: usize,
        anchor: Option<u32>,
    ) {
        if first_frame > last_frame {
            return;
        }
        let mut next = Vec::new();
        for run in &self.runs {
            if run.anchor != anchor || run.end_frame() < first_frame || run.first_frame > last_frame
            {
                next.push(run.clone());
                continue;
            }
            if run.first_frame < first_frame {
                next.push(EditRun {
                    first_frame: run.first_frame,
                    values: run.values[..first_frame - run.first_frame].to_vec(),
                    anchor,
                });
            }
            if run.end_frame() > last_frame {
                let skip = last_frame + 1 - run.first_frame;
                next.push(EditRun {
                    first_frame: last_frame + 1,
                    values: run.values[skip..].to_vec(),
                    anchor,
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
                anchor: run.anchor,
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
                    out.runs
                        .push(EditRun::new(begin, std::mem::take(&mut values)));
                    start = None;
                }
                (None, None) => {}
            }
        }
        if let Some(begin) = start {
            out.runs.push(EditRun::new(begin, values));
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
                    out.runs
                        .push(EditRun::new(begin, std::mem::take(&mut values)));
                    start = None;
                }
                (None, None) => {}
            }
        }
        if let Some(begin) = start {
            out.runs.push(EditRun::new(begin, values));
        }
        out
    }
}

/// 一条轨道的全部显式编辑；frame_ms 与 TrackParamsState.frame_period_ms 同源。
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct EditCurveSet {
    pub frame_ms: f64,
    /// 去重后的锚表。同一条 clip 上的一笔只存一份窗口。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub anchors: Vec<EditAnchor>,
    pub curves: std::collections::BTreeMap<String, EditCurve>,
}

impl EditCurveSet {
    pub fn is_empty(&self) -> bool {
        self.curves.is_empty()
    }

    /// 某个窗口的锚下标；没有就追加一条（按窗口去重）。
    pub fn intern_anchor(&mut self, anchor: &EditAnchor) -> u32 {
        if let Some(index) = self
            .anchors
            .iter()
            .position(|known| known.equivalent(anchor))
        {
            return index as u32;
        }
        self.anchors.push(anchor.clone());
        (self.anchors.len() - 1) as u32
    }

    /// 锚表里的一条。
    pub fn anchor(&self, index: u32) -> Option<&EditAnchor> {
        self.anchors.get(index as usize)
    }

    /// 曲线里是否真的存在某一层。
    pub fn has_layer(&self, key: &str, anchor: Option<u32>) -> bool {
        self.curves
            .get(key)
            .is_some_and(|curve| curve.has_layer(anchor))
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

/// 把一条曲线里属于某个锚层（`None` = 无锚层）的 run 取出来。
pub fn runs_for_anchor(curve: &EditCurve, anchor: Option<u32>) -> impl Iterator<Item = &EditRun> {
    curve.runs_in_layer(anchor)
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
                curve
                    .runs
                    .push(EditRun::new(begin, values[begin..frame].to_vec()));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(begin) = start {
        curve
            .runs
            .push(EditRun::new(begin, values[begin..].to_vec()));
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

    // 【为什么只信这个标记】`pitch_edit` 在未编辑时是**分析目标曲线**（等于
    // `pitch_orig`，或用户导入的 MIDI 目标），它与原始值不同不代表用户改过 ——
    // 拿它做差分会把"分析结果"整条当成编辑写回。标记由每一次真正的写入设置。
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

    EditCurveSet {
        frame_ms,
        anchors: Vec::new(),
        curves,
    }
}

/// 把一份编辑值合成到显示数组：未编辑帧取 defaults，编辑帧覆盖。
///
/// map_frame 把输出帧映射到编辑曲线的帧号；返回的布尔数组标记哪些输出帧来自
/// 显式编辑。map_frame 返回 None 表示该帧没有源映射（保持默认）。
pub fn compose_curve<F>(
    key: &str,
    defaults: &[f32],
    edit: Option<&EditCurve>,
    map_frame: F,
) -> (Vec<f32>, Vec<bool>)
where
    F: FnMut(usize) -> Option<f64>,
{
    compose_curve_in_layer(key, defaults, edit, None, map_frame)
}

/// 与 [`compose_curve`] 相同，但只应用**某一层**的 run。
///
/// 【为什么要分层】锁定（无锚）与不锁定（有锚）是两层独立的编辑：无锚层跟着
/// 当前几何走，有锚层钉在各自的工程窗口上。把两层混在一次取样里，其中一层就会
/// 被另一层的映射解释 —— 这正是"拨一下开关，线跑到别处"的形状。
pub fn compose_curve_in_layer<F>(
    key: &str,
    defaults: &[f32],
    edit: Option<&EditCurve>,
    anchor: Option<u32>,
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
    if !edit.has_layer(anchor) {
        return (values, edited);
    }
    for frame in 0..values.len() {
        let Some(source_frame) = map_frame(frame) else {
            continue;
        };
        if let Some(value) = edit.value_at_in_layer(source_frame, key, anchor) {
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
                out.runs
                    .push(EditRun::new(begin, std::mem::take(&mut values)));
                start = None;
            }
            (None, None) => {}
        }
    }
    if let Some(begin) = start {
        out.runs.push(EditRun::new(begin, values));
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
                    anchor: None,
                },
                EditRun {
                    first_frame: 10,
                    values: vec![3.0],
                    anchor: None,
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
                anchor: None,
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
                anchor: None,
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
                anchor: None,
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
                anchor: None,
            }],
        };
        let reversed = curve.reverse_window(0.0, 2.0, "pitch_edit");
        assert_eq!(reversed.value_at(0.0, "pitch_edit"), Some(64.0));
        assert_eq!(reversed.value_at(1.0, "pitch_edit"), Some(62.0));
        assert_eq!(reversed.value_at(2.0, "pitch_edit"), Some(60.0));
    }
    fn anchor(root: &str, start: f64, duration: f64, source: f64, reversed: bool) -> EditAnchor {
        EditAnchor {
            root: root.to_owned(),
            project_start_sec: start,
            project_duration_sec: duration,
            source_start_sec: source,
            source_duration_sec: duration,
            reversed,
        }
    }

    #[test]
    fn an_anchor_maps_project_and_source_in_both_directions() {
        let forward = anchor("root", 2.0, 4.0, 10.0, false);
        assert_eq!(forward.project_to_source(2.0), Some(10.0));
        assert_eq!(forward.project_to_source(4.0), Some(12.0));
        assert_eq!(forward.project_to_source(6.0), Some(14.0));
        assert_eq!(forward.project_to_source(6.5), None);
        assert_eq!(forward.source_to_project(12.0), Some(4.0));

        let reversed = anchor("root", 2.0, 4.0, 10.0, true);
        assert_eq!(reversed.project_to_source(2.0), Some(14.0));
        assert_eq!(reversed.project_to_source(6.0), Some(10.0));
        assert_eq!(reversed.source_to_project(14.0), Some(2.0));
    }

    #[test]
    fn an_anchored_run_keeps_its_project_position_after_a_move() {
        // 用户在第 4 秒画了一格 64；clip 随后被移到第 10 秒。
        let mut curve = EditCurve::new(5.0);
        curve.runs.push(EditRun::anchored(800, vec![64.0], 0));
        let pinned = anchor("root", 4.0, 0.005, 4.0, false);
        // 钉住的这一笔仍然只出现在它被画的那一段工程时间上。
        let (values, edited) =
            compose_curve_in_layer("pitch_edit", &[60.0; 3], Some(&curve), Some(0), |frame| {
                pinned
                    .project_to_source(4.0 + frame as f64 * 0.005)
                    .map(|source_sec| source_sec * 1000.0 / 5.0)
            });
        assert_eq!(values, vec![64.0, 60.0, 60.0]);
        assert_eq!(edited, vec![true, false, false]);
        // 无锚层看不到它：锁定状态下这一笔必须跟着素材，而不是留在旧位置。
        assert_eq!(curve.value_at_in_layer(800.0, "pitch_edit", None), None);
        assert_eq!(
            curve.value_at_in_layer(800.0, "pitch_edit", Some(0)),
            Some(64.0)
        );
    }

    #[test]
    fn two_runs_over_the_same_source_with_different_anchors_stay_apart() {
        let mut curve = EditCurve::new(5.0);
        curve.runs.push(EditRun::new(0, vec![61.0]));
        curve.runs.push(EditRun::anchored(0, vec![64.0], 0));
        // "任意层"取样按**合成时的优先级**回答：锚层压过无锚层（见 `Rank`）。
        // 分层读取仍然各拿各的 —— 这才是"两层互不覆盖"的真正判据。
        assert_eq!(curve.value_at(0.0, "pitch_edit"), Some(64.0));
        assert_eq!(curve.value_at_in_layer(0.0, "pitch_edit", None), Some(61.0));
        assert_eq!(
            curve.value_at_in_layer(0.0, "pitch_edit", Some(0)),
            Some(64.0)
        );
        assert_eq!(curve.runs_in_layer(None).count(), 1);
        assert_eq!(curve.runs_in_layer(Some(0)).count(), 1);
    }

    #[test]
    fn anchors_are_deduplicated_by_window() {
        let mut set = EditCurveSet::default();
        let first = set.intern_anchor(&anchor("root", 1.0, 2.0, 3.0, false));
        let again = set.intern_anchor(&anchor("root", 1.0, 2.0, 3.0, false));
        let other = set.intern_anchor(&anchor("root", 5.0, 2.0, 3.0, false));
        assert_eq!(first, again);
        assert_ne!(first, other);
        assert_eq!(set.anchors.len(), 2);
    }

    #[test]
    fn a_layer_survives_range_edits_and_shifts() {
        let mut curve = EditCurve::new(5.0);
        curve.runs.push(EditRun::anchored(10, vec![64.0, 65.0], 3));
        curve.remove_range_in_layer(11, 11, Some(3));
        assert_eq!(curve.runs.len(), 1);
        assert_eq!(curve.runs[0].values, vec![64.0]);
        assert!(curve.shift_by(5));
        assert_eq!(curve.runs[0].first_frame, 15);
        assert_eq!(curve.runs[0].anchor, Some(3));
        // 另一层不受影响。
        curve.remove_range_in_layer(0, 100, None);
        assert_eq!(curve.runs.len(), 1);
    }

    #[test]
    fn normalize_layer_merges_adjacent_and_trims_overlap() {
        let mut curve = EditCurve::new(5.0);
        curve.runs.push(EditRun::new(0, vec![1.0, 2.0]));
        curve.runs.push(EditRun::new(2, vec![3.0]));
        curve.runs.push(EditRun::new(1, vec![9.0, 9.0]));
        curve.normalize();
        assert_eq!(curve.runs.len(), 1, "相邻的 run 必须合并成一条");
        // 重叠段保留**先写入**的那一条（它是这一段原本的采样），后写入的只保留
        // 未被覆盖的尾部 —— 这样"把两层钉进同一层"永远不会丢任何一帧数据。
        assert_eq!(curve.runs[0].values, vec![1.0, 2.0, 9.0]);
    }

    #[test]
    fn remove_frames_everywhere_clears_every_layer() {
        let mut curve = EditCurve::new(5.0);
        curve.runs.push(EditRun::new(0, vec![1.0; 5]));
        curve.runs.push(EditRun::anchored(0, vec![2.0; 5], 0));
        let frames = std::collections::BTreeSet::from([1_usize, 3]);
        curve.remove_frames_everywhere(&frames);
        for layer in [None, Some(0)] {
            let values = curve.frame_values_in_layer(layer);
            assert_eq!(values.keys().copied().collect::<Vec<_>>(), vec![0, 2, 4]);
        }
    }

    #[test]
    fn end_frame_is_the_max_over_all_layers() {
        let mut curve = EditCurve::new(5.0);
        curve.runs.push(EditRun::new(0, vec![1.0]));
        curve.runs.push(EditRun::anchored(10, vec![2.0, 2.0], 0));
        assert_eq!(curve.end_frame(), Some(11), "末端取所有层的最大值");
    }

    #[test]
    fn value_at_in_layer_is_correct_with_interleaved_layers() {
        let mut curve = EditCurve::new(5.0);
        curve.runs.push(EditRun::new(0, vec![1.0, 1.0]));
        curve.runs.push(EditRun::anchored(0, vec![2.0, 2.0], 0));
        curve.runs.push(EditRun::new(10, vec![3.0]));
        curve.runs.sort_by_key(|run| (run.first_frame, run.anchor));
        assert_eq!(curve.value_at_in_layer(0.0, "pitch_edit", None), Some(1.0));
        assert_eq!(
            curve.value_at_in_layer(0.0, "pitch_edit", Some(0)),
            Some(2.0)
        );
        assert_eq!(curve.value_at_in_layer(10.0, "pitch_edit", None), Some(3.0));
    }

    #[test]
    fn layers_lists_follow_first_then_anchors_in_order() {
        let mut curve = EditCurve::new(5.0);
        curve.runs.push(EditRun::anchored(0, vec![1.0], 2));
        curve.runs.push(EditRun::new(0, vec![1.0]));
        curve.runs.push(EditRun::anchored(0, vec![1.0], 0));
        assert_eq!(curve.layers(), vec![None, Some(0), Some(2)]);
    }
}
