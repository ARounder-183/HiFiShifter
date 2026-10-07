//! 区域参数的源坐标权威；原GUI项目帧只是投影，宿主移动/拉伸不重新采样写坏原始曲线。
//! 身份来自真实ARA region→modification/source边；不从名字/文件路径/会话序号猜冷恢复归属。

use crate::ara::time_map::SourceTimeMap;
use hifishifter_kernel::state::{TimelineState, TrackParamsState};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct RegionIdentity {
    #[serde(skip)]
    pub key: u64,
    /// REAPER实际item GUID跨mute撤销/区域重建保持；旧归档没有此字段仍可读。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item: Option<String>,
    pub source: String,
    pub modification: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct RegionGeometry {
    pub project_start: f64,
    pub project_duration: f64,
    pub source_start: f64,
    pub source_duration: f64,
}
impl RegionGeometry {
    /// JSON f64往返可能偏一ULP；只忽略数值舍入，不把真实移动/倍率变化当相同布局。
    fn equivalent(&self, other: &Self) -> bool {
        [
            (self.project_start, other.project_start),
            (self.project_duration, other.project_duration),
            (self.source_start, other.source_start),
            (self.source_duration, other.source_duration),
        ]
        .into_iter()
        .all(|(a, b)| near(a, b))
    }
    /// 几何只接受已经确认的ARA秒域；未确认raw marker不得在此成为渲染权威。
    pub fn map(&self) -> Result<SourceTimeMap, String> {
        SourceTimeMap::linear(
            self.project_start,
            self.project_duration,
            self.source_start,
            self.source_duration,
        )
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct SourceCurve {
    basis: RegionGeometry,
    first_frame: usize,
    frame_ms: f64,
    #[serde(with = "arc_values")]
    values: Arc<Vec<f32>>,
    #[serde(skip)]
    reservation: Option<Arc<crate::render::budget::Reservation>>,
}
mod arc_values {
    use super::*;
    pub fn serialize<S: serde::Serializer>(
        value: &Arc<Vec<f32>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value.as_ref().serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Arc<Vec<f32>>, D::Error> {
        let values = Vec::<f32>::deserialize(deserializer)?;
        if values.len() > 1_000_000 || values.iter().any(|v| !v.is_finite() || v.abs() > 10000.) {
            return Err(serde::de::Error::custom("invalid source parameter samples"));
        }
        Ok(Arc::new(values))
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct RegionParameters {
    pub identity: RegionIdentity,
    pub root: String,
    pub current: RegionGeometry,
    /// 历史父范围保留供恢复，但不能与最近活图的真正父候选一起制造嵌套拆分歧义。
    #[serde(default = "default_live")]
    live: bool,
    template: TrackParamsState,
    curves: BTreeMap<String, SourceCurve>,
}
fn default_live() -> bool {
    true
}
/// 项目绝对时间的空白编辑；只保存有效片段，clip源曲线不会被旧位置的整轨数组冒充。
#[derive(Clone, Debug, Serialize, Deserialize)]
struct GapSpan {
    first_frame: usize,
    #[serde(with = "arc_values")]
    values: Arc<Vec<f32>>,
    #[serde(skip)]
    reservation: Option<Arc<crate::render::budget::Reservation>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct GapCurve {
    frame_ms: f64,
    spans: Vec<GapSpan>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct GapParameters {
    template: TrackParamsState,
    curves: BTreeMap<String, GapCurve>,
}
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub(crate) struct ParameterAtlas {
    pub regions: BTreeMap<String, RegionParameters>,
    /// 仅由原生SplitMediaItem回执登记，不允许GUI按源路径伪造分割继承关系。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    split_parents: BTreeMap<String, String>,
    /// 原生复制回执确认的新item GUID；剪切后原region可销毁，seed仍独立保留源basis。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) copy_seeds: BTreeMap<String, RegionParameters>,
    /// 与源basis并存的整轨空白编辑；根轨道身份由同一冷恢复映射重绑定。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) gaps: BTreeMap<String, GapParameters>,
}
impl ParameterAtlas {
    /// 剪贴板反序列化的seed必须复验并登记内存额度，不能让外部数据绕过source curve预算。
    pub(crate) fn reserve_copy_seed(seed: RegionParameters) -> Result<RegionParameters, String> {
        let mut atlas = Self::default();
        atlas.regions.insert("clipboard-seed".into(), seed);
        let mut checked = atlas.reserve_restored()?;
        Ok(checked.regions.remove("clipboard-seed").unwrap())
    }
    /// 在原生加载新item前登记确切GUID谱系，支持跨root与新的source/modification身份。
    pub(crate) fn register_copy(
        &mut self,
        item: &str,
        root: &str,
        current: RegionGeometry,
        seed: Option<RegionParameters>,
    ) -> Result<(), String> {
        if item.len() != 38 || !item.starts_with('{') || !item.ends_with('}') || root.is_empty() {
            return Err("invalid host copy identity/root".into());
        }
        current.map()?;
        if let Some(mut seed) = seed {
            if self.copy_seeds.contains_key(item) || self.copy_seeds.len() >= 16384 {
                return Err("duplicate or excessive pending copy identity".into());
            }
            seed.identity.item = Some(item.to_owned());
            seed.identity.key = 0;
            seed.root = root.to_owned();
            seed.current = current;
            seed.live = false;
            self.copy_seeds.insert(item.to_owned(), seed);
            if let Err(error) = self.validate() {
                self.copy_seeds.remove(item);
                return Err(error);
            }
        }
        Ok(())
    }
    /// 确切item谱系用于同源重叠曲线的无歧义继承；既有basis仍共享且不重新采样。
    pub(crate) fn split_seed(&self, parent: &str) -> Option<RegionParameters> {
        self.regions
            .values()
            .find(|record| record.live && record.identity.item.as_deref() == Some(parent))
            .cloned()
    }
    /// 宿主分割可同步重入ARA回流、先缩短左段；保留调用前的父basis供右段继承。
    pub(crate) fn register_split(
        &mut self,
        parent: &str,
        right: &str,
        seed: Option<RegionParameters>,
    ) -> Result<(), String> {
        if parent == right || parent.len() != 38 || right.len() != 38 {
            return Err("invalid host split GUID pair".into());
        }
        if self.split_parents.len() >= 16384 || self.regions.len() >= 16384 {
            return Err("split lineage budget exceeded".into());
        }
        if let Some(mut seed) = seed {
            seed.live = false;
            self.regions.insert(format!("split-basis:{right}"), seed);
        }
        self.split_parents
            .insert(right.to_owned(), parent.to_owned());
        Ok(())
    }
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty() && self.copy_seeds.is_empty() && self.gaps.is_empty()
    }
    /// 同一源曲线可由历史/区域共享，额度按reservation身份去重，而不是按序列化值重复计数。
    pub fn accounted_curve_bytes(&self) -> usize {
        let mut seen = std::collections::BTreeSet::new();
        let source = self
            .regions
            .values()
            .chain(self.copy_seeds.values())
            .flat_map(|region| region.curves.values())
            .filter_map(|curve| curve.reservation.as_ref());
        let gaps = self
            .gaps
            .values()
            .flat_map(|root| root.curves.values())
            .flat_map(|curve| &curve.spans)
            .filter_map(|span| span.reservation.as_ref());
        source
            .chain(gaps)
            .filter(|reservation| seen.insert(Arc::as_ptr(reservation) as usize))
            .map(|reservation| reservation.bytes())
            .sum()
    }
    /// 冷绑定布局完全相同时保留GUI原整轨数组（含无音频处编辑），音频仍用区域源basis。
    pub fn same_layout(&self, other: &Self) -> bool {
        self.regions.len() == other.regions.len()
            && self.regions.values().all(|old| {
                other.regions.values().any(|new| {
                    old.identity.source == new.identity.source
                        && old.identity.modification == new.identity.modification
                        && (old.identity.item.is_none() || old.identity.item == new.identity.item)
                        && old.root == new.root
                        && old.current.equivalent(&new.current)
                })
            })
    }
    /// 有界state解码后为源曲线重新登记共享512MiB预算；clone共享Arc收费，不按renderer重复收费。
    pub fn reserve_restored(mut self) -> Result<Self, String> {
        self.validate()?;
        for record in self
            .regions
            .values_mut()
            .chain(self.copy_seeds.values_mut())
        {
            for curve in record.curves.values_mut() {
                if curve.reservation.is_none() {
                    curve.reservation = Some(Arc::new(
                        crate::render::budget::global_budget()
                            .reserve(curve.values.len() * 4)
                            .ok_or("source parameter memory budget exceeded")?,
                    ));
                }
            }
        }
        for root in self.gaps.values_mut() {
            for curve in root.curves.values_mut() {
                for span in &mut curve.spans {
                    if span.reservation.is_none() {
                        span.reservation = Some(Arc::new(
                            crate::render::budget::global_budget()
                                .reserve(span.values.len() * 4)
                                .ok_or("gap parameter memory budget exceeded")?,
                        ));
                    }
                }
            }
        }
        Ok(self)
    }
    /// 只换活图投影与真实key，源basis保持；拆分同modification的新region可继承唯一父范围。
    pub fn follow_geometry(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
    ) -> Result<Self, String> {
        let mut followed = self.clone();
        for record in followed.regions.values_mut() {
            record.live = false;
        }
        for clip in &timeline.clips {
            let identity = identities
                .get(&clip.id)
                .ok_or("missing actual ARA parameter identity")?;
            let root = timeline
                .resolve_root_track_id(&clip.track_id)
                .ok_or("unknown parameter root")?;
            let geometry = geometry(clip)?;
            if let Some(previous) = self.find(identity, &root, &geometry)? {
                let mut record = previous.clone();
                record.identity = identity.clone();
                record.root = root;
                record.current = geometry;
                record.live = true;
                followed.regions.insert(clip.id.clone(), record);
            }
        }
        // 两段已经绑定后不再保留临时父副本，避免反复分割累计收费/序列化体积。
        let bound = followed
            .regions
            .values()
            .filter(|record| record.live)
            .filter_map(|record| record.identity.item.clone())
            .collect::<std::collections::BTreeSet<_>>();
        followed.regions.retain(|id, _| {
            id.strip_prefix("split-basis:")
                .is_none_or(|right| !bound.contains(right))
        });
        followed
            .split_parents
            .retain(|right, _| !bound.contains(right));
        followed.copy_seeds.retain(|item, _| !bound.contains(item));
        followed.validate()?;
        Ok(followed)
    }
    /// 冷恢复只在本组件真实范围内绑定；重复持久身份且源窗口相同仍拒绝，不能按旧key猜。
    pub fn rebind(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
    ) -> Result<Self, String> {
        let mut bound = Self::default();
        bound.copy_seeds = self.copy_seeds.clone();
        bound.gaps = self.gaps.clone();
        for record in self.regions.values().filter(|record| record.live) {
            let candidates = timeline
                .clips
                .iter()
                .filter(|clip| {
                    identities.get(&clip.id).is_some_and(|id| {
                        id.source == record.identity.source
                            && id.modification == record.identity.modification
                            && (record.identity.item.is_none() || id.item == record.identity.item)
                    }) && timeline.resolve_root_track_id(&clip.track_id).as_deref()
                        == Some(record.root.as_str())
                })
                .collect::<Vec<_>>();
            let matches = if candidates.len() == 1 {
                candidates
            } else {
                candidates
                    .into_iter()
                    .filter(|clip| {
                        geometry(clip).is_ok_and(|geometry| {
                            near(geometry.source_start, record.current.source_start)
                                && near(geometry.source_duration, record.current.source_duration)
                        })
                    })
                    .collect()
            };
            let [clip] = matches.as_slice() else {
                return Err(
                    "ARA source parameter identity missing or ambiguous in assigned scope".into(),
                );
            };
            if bound.regions.contains_key(&clip.id) {
                return Err("ARA source parameter records collapse into one region".into());
            }
            let mut record = record.clone();
            record.identity = identities[&clip.id].clone();
            record.current = geometry(clip)?;
            record.live = true;
            bound.regions.insert(clip.id.clone(), record);
        }
        bound.validate()?;
        Ok(bound)
    }
    /// GUI仍用原track网格：显示投影可选重叠区，音频始终保留每region自己的参数。
    pub fn project_roots(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
    ) -> Result<BTreeMap<String, TrackParamsState>, String> {
        let clips = self.project(timeline, identities)?;
        let mut roots = BTreeMap::<String, TrackParamsState>::new();
        for (root, gaps) in &self.gaps {
            if !timeline.tracks.iter().any(|track| &track.id == root) {
                continue;
            }
            let mut params = gaps.template.clone();
            for (key, curve) in &gaps.curves {
                set_curve(
                    &mut params,
                    key,
                    gap_values(curve, key, gaps.template.frame_period_ms)?,
                );
            }
            roots.insert(root.clone(), params);
        }
        let mut order = timeline.clips.iter().collect::<Vec<_>>();
        order.sort_by_key(|clip| Some(&clip.id) == timeline.selected_clip_id.as_ref());
        for clip in order {
            let Some(params) = clips.get(&clip.id) else {
                continue;
            };
            let root = timeline
                .resolve_root_track_id(&clip.track_id)
                .ok_or("unknown parameter root")?;
            let geometry = geometry(clip)?;
            let Some((begin, end)) = covered_frame_range(&geometry, params.frame_period_ms)? else {
                continue;
            };
            let entry = roots.entry(root).or_insert_with(|| {
                let mut entry = params.clone();
                clear_curves(&mut entry);
                entry
            });
            if entry.frame_period_ms != params.frame_period_ms {
                return Err("Conflict: source parameter frame periods differ within root".into());
            }
            for (key, values) in parameter_curves(params) {
                if values.is_empty() {
                    continue;
                }
                let mut merged = parameter_curves(entry)
                    .into_iter()
                    .find(|(name, _)| *name == key)
                    .map(|(_, v)| v.to_vec())
                    .unwrap_or_default();
                merged.resize(end.max(merged.len().saturating_sub(1)) + 1, pad(&key));
                merged[begin..=end].copy_from_slice(&values[begin..=end]);
                set_curve(entry, &key, merged);
            }
        }
        Ok(roots)
    }
    /// 接受当前投影上的真实编辑；未改动的曲线继续保留原始源坐标basis。
    pub fn capture_changes(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
        previous_view: &BTreeMap<String, TrackParamsState>,
    ) -> Result<Self, String> {
        self.capture_inner(timeline, identities, Some(previous_view))
    }
    pub fn capture(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
    ) -> Result<Self, String> {
        self.capture_inner(timeline, identities, None)
    }
    /// 只把相对旧GUI投影的delta送回源authority；重叠选区外保留该region自己的值。
    fn capture_inner(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
        previous_view: Option<&BTreeMap<String, TrackParamsState>>,
    ) -> Result<Self, String> {
        let mut candidate = self.clone();
        candidate.capture_gaps(timeline, previous_view)?;
        for clip in &timeline.clips {
            let identity = identities
                .get(&clip.id)
                .ok_or("missing actual ARA parameter identity")?;
            if identity.key == 0
                || identity.source.is_empty()
                || identity.modification.is_empty()
                || clip.source_path.as_deref() != Some(identity.source.as_str())
            {
                return Err("invalid ARA parameter identity".into());
            }
            let root = timeline
                .resolve_root_track_id(&clip.track_id)
                .ok_or("unknown parameter root")?;
            let Some(params) = timeline.params_by_root_track.get(&root) else {
                candidate.regions.remove(&clip.id);
                continue;
            };
            let geometry = geometry(clip)?;
            let (begin, end) = frame_range(&geometry, params.frame_period_ms)?;
            let previous = self.find(identity, &root, &geometry)?;
            let mut curves = BTreeMap::new();
            for (key, values) in parameter_curves(params) {
                if values.is_empty() {
                    continue;
                }
                let old = previous.and_then(|previous| previous.curves.get(&key));
                let view = previous_view
                    .and_then(|view| view.get(&root))
                    .filter(|view| view.frame_period_ms == params.frame_period_ms);
                let old_view = view.and_then(|view| {
                    parameter_curves(view)
                        .into_iter()
                        .find(|(name, _)| *name == key)
                        .map(|(_, values)| values)
                });
                if let (Some(old), Some(old_view)) = (old, old_view) {
                    if values == old_view {
                        curves.insert(key, old.clone());
                        continue;
                    }
                }
                let adjusted = if let (Some(old), Some(old_view)) = (old, old_view) {
                    let mut region = old.project(&geometry, &key, params.frame_period_ms)?;
                    let selected = timeline
                        .selected_clip_id
                        .as_ref()
                        .and_then(|id| timeline.clips.iter().find(|selected| selected.id == *id))
                        .filter(|selected| {
                            timeline
                                .resolve_root_track_id(&selected.track_id)
                                .as_deref()
                                == Some(root.as_str())
                                && selected.id != clip.id
                        });
                    for frame in begin..=end {
                        let incoming = values.get(frame).copied().unwrap_or_else(|| pad(&key));
                        let prior = old_view.get(frame).copied().unwrap_or_else(|| pad(&key));
                        let time = frame as f64 * params.frame_period_ms / 1000.;
                        let covered = selected.is_some_and(|selected| {
                            time >= selected.start_sec
                                && time <= selected.start_sec + selected.length_sec
                        });
                        if incoming != prior && !covered {
                            region[frame] = incoming;
                        }
                    }
                    Some(region)
                } else {
                    None
                };
                let values = adjusted.as_deref().unwrap_or(values);
                let unchanged = old.is_some_and(|old| {
                    old.project(&geometry, &key, params.frame_period_ms)
                        .is_ok_and(|projected| {
                            (begin..=end).all(|frame| {
                                projected[frame]
                                    == values.get(frame).copied().unwrap_or_else(|| pad(&key))
                            })
                        })
                });
                let curve = if unchanged {
                    old.unwrap().clone()
                } else {
                    if values.len() > 1_000_000
                        || values.iter().any(|v| !v.is_finite() || v.abs() > 10000.)
                    {
                        return Err("invalid source parameter samples".into());
                    }
                    let first = begin.saturating_sub(1);
                    let last = (end + 2).min(values.len());
                    let bytes = last.saturating_sub(first) * 4;
                    let reservation = crate::render::budget::global_budget()
                        .reserve(bytes)
                        .ok_or("source parameter memory budget exceeded")?;
                    let samples = if first < last {
                        values[first..last].to_vec()
                    } else {
                        Vec::new()
                    };
                    SourceCurve {
                        basis: geometry.clone(),
                        first_frame: first,
                        frame_ms: params.frame_period_ms,
                        values: Arc::new(samples),
                        reservation: Some(Arc::new(reservation)),
                    }
                };
                curves.insert(key, curve);
            }
            let mut template = params.clone();
            clear_curves(&mut template);
            candidate.regions.insert(
                clip.id.clone(),
                RegionParameters {
                    identity: identity.clone(),
                    root,
                    current: geometry,
                    live: true,
                    template,
                    curves,
                },
            );
            if let Some(item) = &identity.item {
                candidate.copy_seeds.remove(item);
            }
        }
        candidate.validate()?;
        Ok(candidate)
    }
    /// 仅更新current几何，不把投影结果当新源数据；返回每个clip独立参数供worker冻结。
    pub fn project(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
    ) -> Result<BTreeMap<String, TrackParamsState>, String> {
        self.project_in_domain(timeline, identities, false)
    }
    /// 音频直接投影到region局部零点，不经GUI项目网格再次插值；纯移动保持相同合成参数。
    pub fn project_local(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
    ) -> Result<BTreeMap<String, TrackParamsState>, String> {
        self.project_in_domain(timeline, identities, true)
    }
    fn project_in_domain(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
        local: bool,
    ) -> Result<BTreeMap<String, TrackParamsState>, String> {
        self.validate()?;
        let mut projected = BTreeMap::new();
        for clip in &timeline.clips {
            let identity = identities
                .get(&clip.id)
                .ok_or("missing actual ARA parameter identity")?;
            let root = timeline
                .resolve_root_track_id(&clip.track_id)
                .ok_or("unknown parameter root")?;
            let mut geometry = geometry(clip)?;
            let Some(record) = self.find(identity, &root, &geometry)? else {
                continue;
            };
            if local {
                geometry.project_start = 0.;
            }
            let mut params = record.template.clone();
            let frame_ms = params.frame_period_ms;
            for (key, curve) in &record.curves {
                set_curve(&mut params, key, curve.project(&geometry, key, frame_ms)?);
            }
            params.pitch_orig_key = None;
            params.dyn_orig_key = None;
            projected.insert(clip.id.clone(), params);
        }
        Ok(projected)
    }
    /// 原region key优先；拆分新key只能沿同modification/source与同root的唯一父源范围继承。
    fn find(
        &self,
        identity: &RegionIdentity,
        root: &str,
        geometry: &RegionGeometry,
    ) -> Result<Option<&RegionParameters>, String> {
        // 新GUID只能来自真实原生复制回执；不依赖新source/modification是否沿用旧身份。
        if let Some(seed) = identity
            .item
            .as_ref()
            .and_then(|item| self.copy_seeds.get(item))
        {
            return Ok(Some(seed));
        }
        let related = self.regions.values().filter(|record| {
            record.identity.source == identity.source
                && record.identity.modification == identity.modification
        });
        if let Some(item) = &identity.item {
            let matches = related
                .clone()
                .filter(|record| record.identity.item.as_ref() == Some(item))
                .collect::<Vec<_>>();
            if let Some(record) = matches.iter().copied().find(|record| record.live) {
                return Ok(Some(record));
            }
            if let Some(record) = matches.first() {
                return Ok(Some(record));
            }
            if let Some(parent) = self.split_parents.get(item) {
                let candidates = self
                    .regions
                    .values()
                    .filter(|record| {
                        record.identity.item.as_ref() == Some(parent)
                            && record.identity.source == identity.source
                            && record.root == root
                            && geometry.source_start >= record.current.source_start - 1e-6
                            && geometry.source_start + geometry.source_duration
                                <= record.current.source_start
                                    + record.current.source_duration
                                    + 1e-6
                    })
                    .collect::<Vec<_>>();
                let live = candidates
                    .iter()
                    .copied()
                    .filter(|record| record.live)
                    .collect::<Vec<_>>();
                let candidates = if live.is_empty() { candidates } else { live };
                match candidates.as_slice() {
                    [] => {}
                    [record] => return Ok(Some(*record)),
                    _ => return Err("Conflict: ambiguous confirmed split ancestry".into()),
                }
            }
        }
        if identity.key != 0 {
            if let Some(record) = related
                .clone()
                .find(|record| record.live && record.identity.key == identity.key)
            {
                return Ok(Some(record));
            }
        }
        let candidates = related
            .filter(|record| {
                record.identity.key != 0
                    && record.root == root
                    && geometry.source_start
                        < record.current.source_start + record.current.source_duration
                    && geometry.source_start + geometry.source_duration
                        > record.current.source_start
            })
            .collect::<Vec<_>>();
        let live = candidates
            .iter()
            .copied()
            .filter(|record| record.live)
            .collect::<Vec<_>>();
        let candidates = if live.is_empty() { candidates } else { live };
        match candidates.as_slice() {
            [] => Ok(None),
            [record] => Ok(Some(*record)),
            _ => Err("Conflict: ambiguous source parameter ancestry".into()),
        }
    }
    /// 当前先保持原安全数量边界；source basis可序列化，但反序列化后也必须复验。
    pub fn validate(&self) -> Result<(), String> {
        if self.regions.len() > 16384 {
            return Err("parameter atlas region budget exceeded".into());
        }
        if self.copy_seeds.len() > 16384
            || self
                .copy_seeds
                .iter()
                .any(|(item, seed)| item.len() != 38 || seed.identity.item.as_deref() != Some(item))
        {
            return Err("invalid pending copy lineage".into());
        }
        if self.split_parents.len() > 16384
            || self
                .split_parents
                .iter()
                .any(|(right, parent)| right == parent || right.len() != 38 || parent.len() != 38)
        {
            return Err("invalid split lineage".into());
        }
        let mut bytes = 0_usize;
        if self.gaps.len() > 16384 {
            return Err("gap parameter root budget exceeded".into());
        }
        for (root, gaps) in &self.gaps {
            if root.is_empty() || !gaps.template.frame_period_ms.is_finite() {
                return Err("invalid gap parameter root".into());
            }
            for curve in gaps.curves.values() {
                if !curve.frame_ms.is_finite()
                    || !(0.1..=1000.).contains(&curve.frame_ms)
                    || curve.spans.len() > 16384
                {
                    return Err("invalid gap parameter frame domain".into());
                }
                let mut last = 0;
                for span in &curve.spans {
                    let end = span
                        .first_frame
                        .checked_add(span.values.len())
                        .ok_or("gap frame overflow")?;
                    if span.values.is_empty()
                        || span.first_frame < last
                        || end > 1_000_000
                        || span
                            .values
                            .iter()
                            .any(|value| !value.is_finite() || value.abs() > 10000.)
                    {
                        return Err("invalid gap parameter samples".into());
                    }
                    last = end;
                    bytes = bytes
                        .checked_add(span.values.len() * 4)
                        .ok_or("gap byte overflow")?;
                    if bytes > 64 * 1024 * 1024 {
                        return Err("parameter atlas exceeds 64MiB".into());
                    }
                }
            }
        }
        for record in self.regions.values().chain(self.copy_seeds.values()) {
            if record.identity.source.is_empty()
                || record.identity.modification.is_empty()
                || record.root.is_empty()
                || !record.template.frame_period_ms.is_finite()
                || record.template.frame_period_ms < 0.1
                || record.template.frame_period_ms > 1000.
                || record
                    .template
                    .extra_params
                    .values()
                    .any(|value| !value.is_finite())
            {
                return Err("invalid source parameter record".into());
            }
            record.current.map()?;
            for curve in record.curves.values() {
                curve.basis.map()?;
                frame_range(&curve.basis, curve.frame_ms)?;
                if curve.first_frame >= 1_000_000
                    || curve
                        .first_frame
                        .checked_add(curve.values.len())
                        .is_none_or(|end| end > 1_000_000)
                    || curve
                        .values
                        .iter()
                        .any(|v| !v.is_finite() || v.abs() > 10000.)
                {
                    return Err("invalid source parameter samples".into());
                }
                bytes = bytes
                    .checked_add(curve.values.len() * 4)
                    .ok_or("parameter atlas byte overflow")?;
                if bytes > 64 * 1024 * 1024 {
                    return Err("parameter atlas exceeds 64MiB".into());
                }
            }
        }
        Ok(())
    }

    /// 只捕获clip外的用户delta；被移动clip遮住的旧空白编辑保持隐藏，移开后仍存在。
    fn capture_gaps(
        &mut self,
        timeline: &TimelineState,
        previous: Option<&BTreeMap<String, TrackParamsState>>,
    ) -> Result<(), String> {
        let mut coverage = BTreeMap::<String, Vec<RegionGeometry>>::new();
        for clip in &timeline.clips {
            let root = timeline
                .resolve_root_track_id(&clip.track_id)
                .ok_or("unknown gap parameter root")?;
            coverage.entry(root).or_default().push(geometry(clip)?);
        }
        self.capture_gap_roots(&timeline.params_by_root_track, &coverage, previous)
    }
    /// 旧归档整轨数组按保存时布局分离空白层，不能用冷重开后的移动位置猜旧归属。
    pub(crate) fn migrate_legacy_gaps(
        &mut self,
        params: &BTreeMap<String, TrackParamsState>,
    ) -> Result<(), String> {
        if !self.gaps.is_empty() || self.regions.is_empty() {
            return Ok(());
        }
        let mut coverage = BTreeMap::<String, Vec<RegionGeometry>>::new();
        for record in self.regions.values().filter(|record| record.live) {
            coverage
                .entry(record.root.clone())
                .or_default()
                .push(record.current.clone());
        }
        self.capture_gap_roots(params, &coverage, None)?;
        self.validate()
    }
    fn capture_gap_roots(
        &mut self,
        params_by_root: &BTreeMap<String, TrackParamsState>,
        coverage: &BTreeMap<String, Vec<RegionGeometry>>,
        previous: Option<&BTreeMap<String, TrackParamsState>>,
    ) -> Result<(), String> {
        for (root, params) in params_by_root {
            let mut template = params.clone();
            clear_curves(&mut template);
            let old = self.gaps.get(root);
            let mut curves = old.map(|old| old.curves.clone()).unwrap_or_default();
            let count = parameter_curves(params)
                .iter()
                .map(|(_, values)| values.len())
                .max()
                .unwrap_or(0);
            if count > 1_000_000 {
                return Err("gap parameter frame budget exceeded".into());
            }
            let mut owned = vec![false; count];
            for geometry in coverage.get(root).into_iter().flatten() {
                if let Some((begin, end)) = covered_frame_range(geometry, params.frame_period_ms)? {
                    if begin < count {
                        owned[begin..=end.min(count - 1)].fill(true);
                    }
                }
            }
            for (key, values) in parameter_curves(params) {
                let prior = previous
                    .and_then(|previous| previous.get(root))
                    .filter(|prior| prior.frame_period_ms == params.frame_period_ms)
                    .and_then(|prior| {
                        parameter_curves(prior)
                            .into_iter()
                            .find(|(name, _)| name == &key)
                            .map(|(_, values)| values)
                    });
                if prior == Some(values) && curves.contains_key(&key) {
                    continue;
                }
                if values.is_empty() {
                    curves.remove(&key);
                    continue;
                }
                let (mut samples, mut valid) = if let Some(curve) = curves.get(&key) {
                    dense_gap(curve, &key, params.frame_period_ms, values.len())?
                } else {
                    (vec![pad(&key); values.len()], vec![false; values.len()])
                };
                if samples.len() < values.len() {
                    samples.resize(values.len(), pad(&key));
                    valid.resize(values.len(), false);
                }
                for (frame, incoming) in values.iter().copied().enumerate() {
                    if owned[frame] {
                        continue;
                    }
                    let changed = prior.is_none_or(|prior| {
                        incoming != prior.get(frame).copied().unwrap_or_else(|| pad(&key))
                    });
                    if changed {
                        samples[frame] = incoming;
                        valid[frame] = true;
                    }
                }
                let mut spans = Vec::new();
                let mut frame = 0;
                while frame < valid.len() {
                    if !valid[frame] {
                        frame += 1;
                        continue;
                    }
                    let first = frame;
                    while frame < valid.len() && valid[frame] {
                        frame += 1;
                    }
                    let reservation = crate::render::budget::global_budget()
                        .reserve((frame - first) * 4)
                        .ok_or("gap parameter memory budget exceeded")?;
                    spans.push(GapSpan {
                        first_frame: first,
                        values: Arc::new(samples[first..frame].to_vec()),
                        reservation: Some(Arc::new(reservation)),
                    });
                }
                curves.insert(
                    key,
                    GapCurve {
                        frame_ms: params.frame_period_ms,
                        spans,
                    },
                );
            }
            self.gaps
                .insert(root.clone(), GapParameters { template, curves });
        }
        Ok(())
    }
}

/// 用本体真实消费窗口而不是take媒体全长；宿主ARA已经把总倍率投影到active take。
fn geometry(clip: &hifishifter_kernel::state::Clip) -> Result<RegionGeometry, String> {
    if clip.reversed {
        return Err("reverse parameter map unsupported".into());
    }
    let value = RegionGeometry {
        project_start: clip.start_sec,
        project_duration: clip.length_sec,
        source_start: clip.source_start_sec,
        source_duration: clip.length_sec * clip.playback_rate as f64,
    };
    value.map()?;
    Ok(value)
}
fn near(a: f64, b: f64) -> bool {
    a.is_finite()
        && b.is_finite()
        && (a - b).abs() <= 8. * f64::EPSILON * a.abs().max(b.abs()).max(1.)
}
fn frame_range(geometry: &RegionGeometry, frame_ms: f64) -> Result<(usize, usize), String> {
    if !frame_ms.is_finite() || frame_ms < 0.1 || frame_ms > 1000. || geometry.project_start < 0. {
        return Err("invalid source parameter frame domain".into());
    }
    let first = (geometry.project_start * 1000. / frame_ms).floor();
    let last = ((geometry.project_start + geometry.project_duration) * 1000. / frame_ms).ceil();
    if last >= 1_000_000. || !last.is_finite() {
        return Err("source parameter projection frame budget exceeded".into());
    }
    Ok((first as usize, last as usize))
}
/// 显示合并只拥有实际落在clip内的格点；向外取整的哨兵不能覆盖空白或重叠编辑。
fn covered_frame_range(
    geometry: &RegionGeometry,
    frame_ms: f64,
) -> Result<Option<(usize, usize)>, String> {
    frame_range(geometry, frame_ms)?;
    let begin = (geometry.project_start * 1000. / frame_ms).ceil() as usize;
    let end =
        ((geometry.project_start + geometry.project_duration) * 1000. / frame_ms).floor() as usize;
    Ok((begin <= end).then_some((begin, end)))
}
fn gap_values(curve: &GapCurve, key: &str, frame_ms: f64) -> Result<Vec<f32>, String> {
    Ok(dense_gap(curve, key, frame_ms, 0)?.0)
}
/// 空白层只在每个已保存片段内部按绝对时间重采样，不跨过原clip占用区插值。
fn dense_gap(
    curve: &GapCurve,
    key: &str,
    frame_ms: f64,
    count: usize,
) -> Result<(Vec<f32>, Vec<bool>), String> {
    if !frame_ms.is_finite() || !(0.1..=1000.).contains(&frame_ms) {
        return Err("invalid gap frame period".into());
    }
    let duration = curve
        .spans
        .last()
        .map(|span| (span.first_frame + span.values.len() - 1) as f64 * curve.frame_ms)
        .unwrap_or(0.);
    let extent = if curve.spans.is_empty() {
        0
    } else {
        (duration / frame_ms).ceil() as usize + 1
    };
    let count = count.max(extent);
    if count > 1_000_000 {
        return Err("gap projection frame budget exceeded".into());
    }
    let mut values = vec![pad(key); count];
    let mut valid = vec![false; count];
    for span in &curve.spans {
        let begin = (span.first_frame as f64 * curve.frame_ms / frame_ms).ceil() as usize;
        let end = ((span.first_frame + span.values.len() - 1) as f64 * curve.frame_ms / frame_ms)
            .floor() as usize;
        if begin > end {
            continue;
        }
        for frame in begin..=end {
            let index = (frame as f64 * frame_ms / curve.frame_ms - span.first_frame as f64)
                .clamp(0., (span.values.len() - 1) as f64);
            let lo = index.floor() as usize;
            let hi = (lo + 1).min(span.values.len() - 1);
            let fraction = (index - lo as f64) as f32;
            values[frame] = span.values[lo] + (span.values[hi] - span.values[lo]) * fraction;
            valid[frame] = true;
        }
    }
    Ok((values, valid))
}
pub(super) fn pad(key: &str) -> f32 {
    if key == "extra:volume" || key == "extra:breath_gain" {
        1.
    } else if key == "extra:dyn" {
        -1.
    } else {
        0.
    }
}
pub(super) fn parameter_curves(params: &TrackParamsState) -> Vec<(String, &[f32])> {
    let mut curves = vec![
        ("pitch_orig".into(), params.pitch_orig.as_slice()),
        ("pitch_edit".into(), params.pitch_edit.as_slice()),
        ("tension_orig".into(), params.tension_orig.as_slice()),
        ("tension_edit".into(), params.tension_edit.as_slice()),
        ("dyn_orig".into(), params.dyn_orig.as_slice()),
    ];
    curves.extend(
        params
            .extra_curves
            .iter()
            .map(|(key, values)| (format!("extra:{key}"), values.as_slice())),
    );
    curves
}
fn clear_curves(params: &mut TrackParamsState) {
    params.pitch_orig.clear();
    params.pitch_edit.clear();
    params.tension_orig.clear();
    params.tension_edit.clear();
    params.dyn_orig.clear();
    params.extra_curves.clear();
}
pub(super) fn set_curve(params: &mut TrackParamsState, key: &str, values: Vec<f32>) {
    match key {
        "pitch_orig" => params.pitch_orig = values,
        "pitch_edit" => params.pitch_edit = values,
        "tension_orig" => params.tension_orig = values,
        "tension_edit" => params.tension_edit = values,
        "dyn_orig" => params.dyn_orig = values,
        _ => {
            if let Some(key) = key.strip_prefix("extra:") {
                params.extra_curves.insert(key.into(), values);
            }
        }
    }
}
impl SourceCurve {
    /// 只生成项目网格投影，原values/basis保持不可变，连续宿主变换不累积插值损失。
    fn project(
        &self,
        geometry: &RegionGeometry,
        key: &str,
        frame_ms: f64,
    ) -> Result<Vec<f32>, String> {
        let (begin, end) = frame_range(geometry, frame_ms)?;
        let current = geometry.map()?;
        let basis = self.basis.map()?;
        let mut result = vec![pad(key); end + 1];
        for frame in begin..=end {
            // 音频/显示的外侧网格哨兵使用真实边界值，不能把未落在clip内的格点置零后插入尾部。
            let project = (frame as f64 * frame_ms / 1000.).clamp(
                geometry.project_start,
                geometry.project_start + geometry.project_duration,
            );
            let Some(previous) = current.previous_project_time(&basis, project) else {
                continue;
            };
            let index = previous * 1000. / self.frame_ms - self.first_frame as f64;
            if !index.is_finite() || index < 0. || self.values.is_empty() {
                continue;
            }
            let lo = (index.floor() as usize).min(self.values.len() - 1);
            let hi = (lo + 1).min(self.values.len() - 1);
            let fraction = (index - lo as f64).clamp(0., 1.) as f32;
            let a = self.values[lo];
            let b = self.values[hi];
            result[frame] = if key == "pitch_edit" && (a <= 0. || b <= 0.) {
                a.max(b).max(0.)
            } else {
                a + (b - a) * fraction
            };
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn continuous_track() -> TimelineState {
        let mut timeline = host(1.003, 0.197, 0., 0.197);
        timeline.project_sec = 4.;
        timeline.params_by_root_track.insert(
            "a".into(),
            TrackParamsState {
                frame_period_ms: 5.,
                pitch_edit_user_modified: true,
                pitch_edit: vec![60.; 801],
                tension_edit: vec![60.; 801],
                extra_curves: std::collections::HashMap::from([(
                    "hifigan_tension".into(),
                    vec![60.; 801],
                )]),
                ..Default::default()
            },
        );
        timeline
    }
    /// 空白编辑参与保存/回流；非网格clip两侧不能重新变成零，局部音频哨兵也保持边界值。
    #[test]
    fn gap_layer_keeps_continuous_curves_and_non_grid_boundary_samples() {
        let timeline = continuous_track();
        let ids = identities();
        let atlas = ParameterAtlas::default().capture(&timeline, &ids).unwrap();
        let bytes = serde_json::to_vec(&atlas).unwrap();
        let restored: ParameterAtlas = serde_json::from_slice(&bytes).unwrap();
        let restored = restored
            .reserve_restored()
            .unwrap()
            .rebind(&timeline, &ids)
            .unwrap();
        let view = restored.project_roots(&timeline, &ids).unwrap();
        assert_eq!(view["a"].tension_edit.len(), 801);
        for (frame, value) in view["a"].tension_edit.iter().enumerate() {
            assert_eq!(*value, 60., "legacy tension frame {frame}");
        }
        for (frame, value) in view["a"].extra_curves["hifigan_tension"].iter().enumerate() {
            assert_eq!(*value, 60., "HiFiGAN tension frame {frame}");
        }
        let local = restored.project_local(&timeline, &ids).unwrap();
        assert_eq!(local["clip"].tension_edit.last(), Some(&60.));
    }
    /// 空白层留在项目绝对时间，源曲线随clip移动；原clip占用区不留下假的旧源曲线。
    #[test]
    fn gap_layer_preserves_gap_edits_after_clip_moves_and_new_gap_edit_deltas() {
        let initial = continuous_track();
        let ids = identities();
        let atlas = ParameterAtlas::default().capture(&initial, &ids).unwrap();
        let mut moved = host(2.203, 0.197, 0., 0.197);
        moved.project_sec = 4.;
        let followed = atlas.follow_geometry(&moved, &ids).unwrap();
        let before = followed.project_roots(&moved, &ids).unwrap();
        assert_eq!(before["a"].tension_edit[200], 60.);
        assert_eq!(before["a"].tension_edit[210], 0.);
        assert_eq!(before["a"].tension_edit[450], 60.);
        moved.params_by_root_track = before.clone();
        moved
            .params_by_root_track
            .get_mut("a")
            .unwrap()
            .tension_edit[210] = 45.;
        let changed = followed.capture_changes(&moved, &ids, &before).unwrap();
        assert_eq!(
            changed.project_roots(&moved, &ids).unwrap()["a"].tension_edit[210],
            45.
        );
        assert_eq!(
            changed.project_local(&moved, &ids).unwrap()["clip"]
                .tension_edit
                .last(),
            Some(&60.)
        );
    }
    /// 旧归档迁移使用保存时的源布局，而不是用当前已移动布局误吞旧空白编辑。
    #[test]
    fn gap_layer_migrates_legacy_track_arrays_using_saved_layout() {
        let initial = continuous_track();
        let ids = identities();
        let mut legacy = ParameterAtlas::default().capture(&initial, &ids).unwrap();
        legacy.gaps.clear();
        legacy
            .migrate_legacy_gaps(&initial.params_by_root_track)
            .unwrap();
        let moved = host(2.203, 0.197, 0., 0.197);
        let projected = legacy.project_roots(&moved, &ids).unwrap();
        assert_eq!(projected["a"].tension_edit[200], 60.);
        assert_eq!(projected["a"].tension_edit[210], 0.);
        assert_eq!(projected["a"].tension_edit[450], 60.);
    }
    /// 剪切可销毁原region；新source/modification与跨root仍按真实新GUID继承，而非路径猜父。
    #[test]
    fn copied_item_seed_survives_original_removal_and_new_host_identities() {
        let original = edited();
        let ids = identities();
        let atlas = ParameterAtlas::default().capture(&original, &ids).unwrap();
        let expected = atlas.project_local(&original, &ids).unwrap()["clip"].clone();
        let seed = atlas.regions["clip"].clone();
        let mut copied = host(3.125, 1., 0., 1.);
        copied.tracks[0].id = "b".into();
        copied.clips[0].track_id = "b".into();
        copied.clips[0].id = "copy".into();
        let item = "{11111111-2222-3333-4444-555555555555}";
        let new_ids = BTreeMap::from([(
            "copy".into(),
            RegionIdentity {
                key: 99,
                item: Some(item.into()),
                source: "new-source".into(),
                modification: "new-mod".into(),
            },
        )]);
        let mut pending = ParameterAtlas::default();
        pending
            .register_copy(item, "b", geometry(&copied.clips[0]).unwrap(), Some(seed))
            .unwrap();
        let serialized = serde_json::to_vec(&pending).unwrap();
        let restored: ParameterAtlas = serde_json::from_slice(&serialized).unwrap();
        let restored = restored.reserve_restored().unwrap();
        let followed = restored.follow_geometry(&copied, &new_ids).unwrap();
        assert!(followed.copy_seeds.is_empty());
        assert_eq!(followed.regions["copy"].root, "b");
        assert_eq!(
            followed.project_local(&copied, &new_ids).unwrap()["copy"].pitch_edit,
            expected.pitch_edit
        );
        assert_eq!(
            followed.project_local(&copied, &new_ids).unwrap()["copy"].extra_curves["volume"],
            expected.extra_curves["volume"]
        );
    }
    /// 连续拆分只沿最近活图的唯一父范围，保留历史父记录但不让它阻塞第二次拆分。
    #[test]
    fn nested_splits_follow_live_ancestry_without_losing_history_or_guessing_overlaps() {
        let original = edited();
        let atlas = ParameterAtlas::default()
            .capture(&original, &identities())
            .unwrap();
        let mut first = host(1., 0.5, 0., 0.5);
        first.clips[0].id = "left".into();
        let mut right = host(1.5, 0.5, 0.5, 0.5).clips.remove(0);
        right.id = "right".into();
        first.clips.push(right);
        first.project_sec = 2.;
        let ids = BTreeMap::from([
            (
                "left".into(),
                RegionIdentity {
                    key: 42,
                    item: None,
                    source: "source".into(),
                    modification: "mod".into(),
                },
            ),
            (
                "right".into(),
                RegionIdentity {
                    key: 43,
                    item: None,
                    source: "source".into(),
                    modification: "mod".into(),
                },
            ),
        ]);
        let followed = atlas.follow_geometry(&first, &ids).unwrap();
        assert_eq!(followed.regions.len(), 3);
        assert!(!followed.regions["clip"].live);
        assert!(followed.regions["left"].live);
        let mut next = host(1., 0.25, 0., 0.25);
        next.clips[0].id = "quarter-a".into();
        let mut second = host(1.25, 0.25, 0.25, 0.25).clips.remove(0);
        second.id = "quarter-b".into();
        next.clips.push(second);
        next.clips.push(first.clips[1].clone());
        next.project_sec = 2.;
        let mut next_ids = ids.clone();
        next_ids.remove("left");
        next_ids.insert(
            "quarter-a".into(),
            RegionIdentity {
                key: 44,
                item: None,
                source: "source".into(),
                modification: "mod".into(),
            },
        );
        next_ids.insert(
            "quarter-b".into(),
            RegionIdentity {
                key: 45,
                item: None,
                source: "source".into(),
                modification: "mod".into(),
            },
        );
        let nested = followed.follow_geometry(&next, &next_ids).unwrap();
        let local = nested.project_local(&next, &next_ids).unwrap();
        assert_eq!(local["quarter-a"].pitch_edit, &[60., 61.]);
        assert_eq!(local["quarter-b"].pitch_edit, &[61., 62.]);
        assert_eq!(local["right"].pitch_edit, &[62., 63., 64.]);
        assert!(!nested.regions["left"].live);
        // 两条真正活且重叠的不同编辑仍有歧义，不能选最小范围/最后写入来掩盖。
        let mut ambiguous = followed.clone();
        ambiguous.regions.get_mut("clip").unwrap().live = true;
        assert!(ambiguous
            .follow_geometry(&next, &next_ids)
            .unwrap_err()
            .contains("ambiguous"));
    }
    fn host(start: f64, duration: f64, source_start: f64, source_duration: f64) -> TimelineState {
        let mut timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0}],"bpm":120,"project_sec":start+duration,
            "clips":[{"id":"clip","track_id":"a","name":"voice","start_sec":start,"length_sec":duration,
                "takes":[{"id":"take","source_path":"source","source_start_sec":source_start,
                    "source_end_sec":source_start+source_duration,"playback_rate":source_duration/duration}]}]
        })).unwrap();
        timeline.clips[0].normalize_takes();
        timeline
    }
    fn identities() -> BTreeMap<String, RegionIdentity> {
        BTreeMap::from([(
            "clip".into(),
            RegionIdentity {
                key: 41,
                item: None,
                source: "source".into(),
                modification: "mod".into(),
            },
        )])
    }
    /// 同源同窗口的多个item重建ARA key后仍按真实GUID归属，不走歧义源祖先猜测。
    #[test]
    fn ui_inventory_item_identity_survives_mute_region_recreation_with_same_source_windows() {
        let mut initial = edited();
        let mut other = initial.clips[0].clone();
        other.id = "other".into();
        other.start_sec = 3.;
        initial.clips.push(other);
        initial.project_sec = 4.;
        let params = initial.params_by_root_track.get_mut("a").unwrap();
        params.pitch_edit.resize(17, 0.);
        params.pitch_orig.resize(17, 0.);
        params.pitch_edit[12..17].fill(70.);
        params.pitch_orig[12..17].fill(57.);
        let mut ids = identities();
        ids.get_mut("clip").unwrap().item = Some("item-a".into());
        ids.insert(
            "other".into(),
            RegionIdentity {
                key: 42,
                item: Some("item-b".into()),
                source: "source".into(),
                modification: "mod".into(),
            },
        );
        let atlas = ParameterAtlas::default().capture(&initial, &ids).unwrap();
        ids.get_mut("clip").unwrap().key = 141;
        ids.get_mut("other").unwrap().key = 142;
        let followed = atlas.follow_geometry(&initial, &ids).unwrap();
        let projected = followed.project_local(&initial, &ids).unwrap();
        assert_eq!(projected["clip"].pitch_edit[0], 60.);
        assert_eq!(projected["other"].pitch_edit[0], 70.);
        let bytes = serde_json::to_vec(&followed).unwrap();
        let restored: ParameterAtlas = serde_json::from_slice(&bytes).unwrap();
        assert!(restored.rebind(&initial, &ids).is_ok());
    }
    fn edited() -> TimelineState {
        let mut timeline = host(1., 1., 0., 1.);
        timeline.params_by_root_track.insert(
            "a".into(),
            TrackParamsState {
                frame_period_ms: 250.,
                pitch_edit_user_modified: true,
                pitch_orig: vec![0., 0., 0., 0., 57., 57., 57., 57., 57.],
                pitch_edit: vec![0., 0., 0., 0., 60., 61., 62., 63., 64.],
                extra_curves: std::collections::HashMap::from([(
                    "volume".into(),
                    vec![1., 1., 1., 1., 0.5, 0.6, 0.7, 0.8, 0.9],
                )]),
                ..Default::default()
            },
        );
        timeline
    }
    #[test]
    fn audio_local_projection_is_identical_after_non_grid_movement_and_maps_stretch_once() {
        let ids = identities();
        let atlas = ParameterAtlas::default().capture(&edited(), &ids).unwrap();
        let original = atlas.project_local(&edited(), &ids).unwrap();
        let moved = atlas
            .project_local(&host(512.013, 1., 0., 1.), &ids)
            .unwrap();
        assert_eq!(original["clip"].pitch_edit, moved["clip"].pitch_edit);
        assert_eq!(&moved["clip"].pitch_edit, &[60., 61., 62., 63., 64.]);
        let stretched = atlas
            .project_local(&host(512.013, 2., 0., 1.), &ids)
            .unwrap();
        assert_eq!(
            &stretched["clip"].pitch_edit,
            &[60., 60.5, 61., 61.5, 62., 62.5, 63., 63.5, 64.]
        );
    }
    /// 一个可见root数组不能把重叠的另一region当同一编辑；只修改可见delta对应的选区。
    #[test]
    fn overlapping_selection_edits_do_not_recapture_another_regions_visible_projection() {
        let mut initial = edited();
        let mut other = initial.clips[0].clone();
        other.id = "other".into();
        other.start_sec = 3.;
        initial.clips.push(other);
        initial.project_sec = 4.;
        let params = initial.params_by_root_track.get_mut("a").unwrap();
        params.pitch_edit.resize(17, 0.);
        params.pitch_orig.resize(17, 0.);
        params.pitch_edit[12..17].copy_from_slice(&[67., 68., 69., 70., 71.]);
        params.pitch_orig[12..17].fill(57.);
        let mut ids = identities();
        ids.insert(
            "other".into(),
            RegionIdentity {
                key: 42,
                item: None,
                source: "source".into(),
                modification: "other-mod".into(),
            },
        );
        let atlas = ParameterAtlas::default().capture(&initial, &ids).unwrap();
        let mut overlap = initial.clone();
        overlap.clips[1].start_sec = 1.;
        overlap.project_sec = 2.;
        overlap.selected_clip_id = Some("clip".into());
        let atlas = atlas.follow_geometry(&overlap, &ids).unwrap();
        let before = atlas.project_roots(&overlap, &ids).unwrap();
        overlap.params_by_root_track = before.clone();
        let same = atlas
            .capture_changes(&overlap, &ids, &before)
            .unwrap()
            .project(&overlap, &ids)
            .unwrap();
        assert_eq!(
            &same["other"].pitch_edit[4..9],
            &[67., 68., 69., 70., 71.],
            "只换GUI投影不能改另一region"
        );
        overlap
            .params_by_root_track
            .get_mut("a")
            .unwrap()
            .pitch_edit[5] = 72.;
        let changed = atlas
            .capture_changes(&overlap, &ids, &before)
            .unwrap()
            .project(&overlap, &ids)
            .unwrap();
        assert_eq!(changed["clip"].pitch_edit[5], 72.);
        assert_eq!(changed["other"].pitch_edit[5], 68.);
    }
    #[test]
    fn source_curves_follow_movement_crop_and_forward_stretch_instead_of_old_project_frames() {
        let atlas = ParameterAtlas::default()
            .capture(&edited(), &identities())
            .unwrap();
        let moved = atlas.project(&host(3., 1., 0., 1.), &identities()).unwrap();
        assert_eq!(
            &moved["clip"].pitch_edit[12..17],
            &[60., 61., 62., 63., 64.]
        );
        assert_eq!(moved["clip"].pitch_edit[4], 0.);
        assert_eq!(moved["clip"].extra_curves["volume"][4], 1.);
        let stretched = atlas.project(&host(1., 2., 0., 1.), &identities()).unwrap();
        assert_eq!(
            &stretched["clip"].pitch_edit[4..13],
            &[60., 60.5, 61., 61.5, 62., 62.5, 63., 63.5, 64.]
        );
        let cropped = atlas
            .project(&host(2., 1., 0.5, 0.5), &identities())
            .unwrap();
        assert_eq!(
            &cropped["clip"].pitch_edit[8..13],
            &[62., 62.5, 63., 63.5, 64.]
        );
    }
    #[test]
    fn repeated_geometry_roundtrips_do_not_recapture_and_blur_unchanged_source_curves() {
        let ids = identities();
        let original = edited();
        let atlas = ParameterAtlas::default().capture(&original, &ids).unwrap();
        let mut changed = host(0.013, 1.731, 0., 1.);
        changed.params_by_root_track.insert(
            "a".into(),
            atlas
                .project(&changed, &ids)
                .unwrap()
                .remove("clip")
                .unwrap(),
        );
        let accepted = atlas.capture(&changed, &ids).unwrap();
        let back = accepted.project(&original, &ids).unwrap();
        assert_eq!(
            &back["clip"].pitch_edit[4..9],
            &[60., 61., 62., 63., 64.],
            "不可把已插值投影反复当新源数据"
        );
    }
    #[test]
    fn unrelated_modification_cannot_inherit_curves_from_a_shared_source_path() {
        let atlas = ParameterAtlas::default()
            .capture(&edited(), &identities())
            .unwrap();
        let mut ids = identities();
        ids.get_mut("clip").unwrap().modification = "other".into();
        let projected = atlas.project(&host(1., 1., 0., 1.), &ids).unwrap();
        assert!(projected.is_empty());
    }
    /// 同源同窗口的不同编辑重叠时，原生分割谱系只能继承确切父item；序列化仍保留归属。
    #[test]
    fn host_split_confirmed_lineage_preserves_right_pitch_with_overlapping_source_ancestry() {
        let parent = "{11111111-1111-1111-1111-111111111111}";
        let right = "{33333333-3333-3333-3333-333333333333}";
        let mut initial = edited();
        let mut other = initial.clips[0].clone();
        other.id = "other".into();
        other.start_sec = 3.;
        initial.clips.push(other);
        initial.project_sec = 4.;
        let params = initial.params_by_root_track.get_mut("a").unwrap();
        params.pitch_edit.resize(17, 0.);
        params.pitch_orig.resize(17, 57.);
        params.pitch_edit[12..17].fill(72.);
        let mut ids = identities();
        ids.get_mut("clip").unwrap().item = Some(parent.into());
        ids.insert(
            "other".into(),
            RegionIdentity {
                key: 42,
                item: Some("{55555555-5555-5555-5555-555555555555}".into()),
                source: "source".into(),
                modification: "mod".into(),
            },
        );
        let mut atlas = ParameterAtlas::default().capture(&initial, &ids).unwrap();
        let seed = atlas.split_seed(parent);
        // 模拟宿主在SplitMediaItem返回前已经同步缩短左段，不能事后再从左段猜右段basis。
        let left = host(1., 0.5, 0., 0.5);
        atlas = atlas.follow_geometry(&left, &ids).unwrap();
        atlas.register_split(parent, right, seed).unwrap();
        let mut split = host(1.5, 0.5, 0.5, 0.5);
        split.clips[0].id = "right".into();
        let right_ids = BTreeMap::from([(
            "right".into(),
            RegionIdentity {
                key: 43,
                item: Some(right.into()),
                source: "source".into(),
                modification: "new-mod".into(),
            },
        )]);
        let followed = atlas.follow_geometry(&split, &right_ids).unwrap();
        let local = followed.project_local(&split, &right_ids).unwrap();
        assert_eq!(&local["right"].pitch_edit[..3], &[62., 63., 64.]);
        assert!(followed.split_parents.is_empty());
        assert!(!followed
            .regions
            .keys()
            .any(|id| id.starts_with("split-basis:")));
        let restored: ParameterAtlas =
            serde_json::from_slice(&serde_json::to_vec(&followed).unwrap()).unwrap();
        assert!(restored.rebind(&split, &right_ids).is_ok());
    }
}
