//! 区域参数的源坐标权威；原GUI项目帧只是投影，宿主移动/拉伸不重新采样写坏原始曲线。
//! 身份来自真实ARA region→modification/source边；不从名字/文件路径/会话序号猜冷恢复归属。

use crate::ara::time_map::{SourceTimeMap, TimeAnchor};
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
    /// **跨会话稳定的 region 签名**（见 [`ParameterAtlas::ara_region_signature`]）。
    ///
    /// 【为什么与 `item` 并存】`item` 只在有原生扩展时才有值；ARA-only 宿主上它恒为
    /// `None`，于是谱系、身份认领全部走不通。签名只用 ARA 自己给的量，任何宿主都有。
    /// 旧归档没有这个字段，`#[serde(default)]` 让它读作空串，随即退回 `item` 路径 ——
    /// 向后兼容。
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub signature: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct RegionGeometry {
    pub project_start: f64,
    pub project_duration: f64,
    /// **消费窗口起点**（源域升序下界）；倒放时是 `source_end − span`，不是存储的
    /// `clip.source_start_sec`（后者在倒放里只是编辑痕迹，不参与消费数学）。
    pub source_start: f64,
    pub source_duration: f64,
    /// 倒放：项目时间增、源位置减。缺省 false 让旧归档按正放读，语义不变。
    #[serde(default)]
    pub reversed: bool,
}
impl RegionGeometry {
    /// JSON f64往返可能偏一ULP；只忽略数值舍入，不把真实移动/倍率变化当相同布局。
    fn equivalent(&self, other: &Self) -> bool {
        self.reversed == other.reversed
            && [
                (self.project_start, other.project_start),
                (self.project_duration, other.project_duration),
                (self.source_start, other.source_start),
                (self.source_duration, other.source_duration),
            ]
            .into_iter()
            .all(|(a, b)| near(a, b))
    }
    /// 几何只接受已经确认的ARA秒域；未确认raw marker不得在此成为渲染权威。
    ///
    /// 倒放建**降序源轴**（项目起点 ⇒ 消费窗口终点），正放建升序源轴。两者共用
    /// `SourceTimeMap` 的同一份区间插值，只是锚点方向相反。
    pub fn map(&self) -> Result<SourceTimeMap, String> {
        if self.reversed {
            SourceTimeMap::piecewise(vec![
                TimeAnchor {
                    project_sec: self.project_start,
                    source_sec: self.source_start + self.source_duration,
                },
                TimeAnchor {
                    project_sec: self.project_start + self.project_duration,
                    source_sec: self.source_start,
                },
            ])
        } else {
            SourceTimeMap::linear(
                self.project_start,
                self.project_duration,
                self.source_start,
                self.source_duration,
            )
        }
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
    /// **ARA-only 宿主的谱系**：右半段签名 → 父段签名。
    ///
    /// 【为什么需要平行的一套】上面的 `split_parents` / `copy_seeds` 都以宿主 item GUID
    /// 为键，而 item GUID 只在有**原生扩展**时才拿得到（`region_items` 由
    /// `refresh_host_geometry` 填，ARA-only 宿主上恒空）。于是 Cubase / Nuendo /
    /// Studio One 上每条分割都走不了显式谱系，只能落到 `find` 的几何包含兜底 ——
    /// 两个半段共享同一个 (source, modification) 且窗口相邻，兜底层把它们都判成
    /// 候选，报 `Conflict: ambiguous source parameter ancestry`。
    ///
    /// 这一套改用**region 签名**作键（见 `ara_region_signature`）：它只用 ARA 自己给的量
    /// （source persistentID + modification persistentID + 播放几何），不依赖任何原生扩展，
    /// 因此在 ARA-only 宿主上同样稳定。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    ara_split_parents: BTreeMap<String, String>,
    /// ARA-only 宿主的复制谱系：新片段签名 → 种子的完整参数记录。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    ara_copy_seeds: BTreeMap<String, RegionParameters>,
}
impl ParameterAtlas {
    /// 一条 region 的**跨会话稳定签名**。
    ///
    /// 【为什么用这三个量】`ARAPlaybackRegionProperties` 没有 persistentID
    /// （`ARAInterface.h:1012-1073`），所以 region 身份在会话内只能用 model-ref key，
    /// 而 key 跨会话（存档重放）不保证相同。能跨会话稳定的是它引用的两个 persistentID
    /// 加上播放几何：`audioModificationPersistentID` 在宿主复制/分割一个 item 时会
    /// 被克隆成新的，`audioSourcePersistentID` 指向同一个文件，播放起止时间则是
    /// 时间线上的位置。
    ///
    /// 【为什么包含播放时间】只靠 (source, modification) 不足以区分同一文件上的多个
    /// 片段；而 `durationInPlaybackTime` 会随裁剪变化，`startInPlaybackTime` 会随移动
    /// 变化 —— 两者都在签名里，于是"同一个片段"的判据是"同源、同修改、同位置"。
    /// 宿主确实移动了它时签名会变，那正是一次需要重新认领的编辑。
    pub(crate) fn ara_region_signature(
        source: &str,
        modification: &str,
        play_start: f64,
        play_duration: f64,
    ) -> String {
        format!(
            "ara:{modification}:{source}:{}:{}",
            play_start.to_bits(),
            play_duration.to_bits()
        )
    }

    /// 登记一次 ARA-only 宿主的**分割**谱系。
    ///
    /// 与 `register_split` 同口径（seed 是父段完整窗口、右半段靠包含判定继承），
    /// 差别只在键：这里用 region 签名而不是 item GUID。
    pub(crate) fn register_ara_split(
        &mut self,
        parent_signature: &str,
        right_signature: &str,
        seed: Option<RegionParameters>,
    ) -> Result<(), String> {
        if parent_signature == right_signature || parent_signature.is_empty() {
            return Err("invalid ARA split signature pair".into());
        }
        if self.ara_split_parents.len() >= 16384 || self.regions.len() >= 16384 {
            return Err("ARA split lineage budget exceeded".into());
        }
        if let Some(mut seed) = seed {
            seed.live = false;
            self.regions
                .insert(format!("ara-split-basis:{right_signature}"), seed);
        }
        self.ara_split_parents
            .insert(right_signature.to_owned(), parent_signature.to_owned());
        Ok(())
    }

    /// 查一个 ARA region 签名的父段签名（分割谱系）。
    pub(crate) fn ara_split_parent(&self, right_signature: &str) -> Option<&str> {
        self.ara_split_parents
            .get(right_signature)
            .map(String::as_str)
    }

    /// 登记一次 ARA-only 宿主的**复制**谱系（种子即父段完整参数）。
    ///
    /// 【为什么读者已就位、写者暂缺】`ara_copy_seed` 已经在 `find` 的解析顺序里
    /// （复制谱系早于几何兜底），但 ARA-only 上"哪两条 region 是一次复制"**没有可核实的
    /// 信号** —— item GUID 版的 `register_copy` 靠 REAPER 的 take 身份，而 ARA 表达不到。
    /// 按几何猜会把"同一素材被放两次"误认成复制，那正是前几轮刚修掉的歧义类缺陷。
    /// 所以保留入口、等一条真实判据，而不是现在硬塞一个猜测。
    #[allow(dead_code)]
    pub(crate) fn register_ara_copy(
        &mut self,
        signature: &str,
        seed: RegionParameters,
    ) -> Result<(), String> {
        if signature.is_empty() {
            return Err("invalid ARA copy signature".into());
        }
        if self.ara_copy_seeds.len() >= 16384 {
            return Err("ARA copy lineage budget exceeded".into());
        }
        self.ara_copy_seeds.insert(signature.to_owned(), seed);
        Ok(())
    }

    /// 取一个 ARA region 签名的复制种子。
    pub(crate) fn ara_copy_seed(&self, signature: &str) -> Option<&RegionParameters> {
        self.ara_copy_seeds.get(signature)
    }

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
    /// ARA-only 版 `split_seed`：按 region 签名找父段记录。
    ///
    /// 【为什么另写一个而不是合并】`split_seed` 按 item GUID 匹配，而 ARA-only 宿主的
    /// 记录 `item` 是 `None` —— 合进来会让两条判据在同一次查找里竞争，命中哪一个
    /// 取决于记录形状而不是宿主类型。分开之后"有 GUID 走 GUID、没有走签名"是显式的。
    ///
    /// 【为什么不再要求 `live`】分割检测（`ModelHandle::register_host_split_lineage`）
    /// 在**参数投影之前**跑，但父段在上一轮 `follow_geometry` 里就已被置为非活 ——
    /// 实测（探针）：
    ///
    /// ```text
    /// after split: [("L", live=true), ("P", live=false), ("R", live=true)]
    /// ara_split_seed(P) = None          ← 父段刚死，seed 就没了
    /// ```
    ///
    /// 于是每次分割都登记不了显式谱系、只能落几何兜底 —— 而那正是歧义的来源。
    /// 非活记录恰恰**就是**"分割前的父段"，正是要用的那一条。
    ///
    /// 【为什么取窗口最大的一条】同签名理论上唯一（签名含播放几何），但 `split-basis`
    /// 副本与历史记录可能同签名。取源窗**最大**的那条 = 最近/最完整的祖先，与几何兜底
    /// "包含判定"的取向一致；并列时按 id 稳定取第一条，避免非确定性。
    pub(crate) fn ara_split_seed(&self, parent_signature: &str) -> Option<RegionParameters> {
        self.regions
            .values()
            .filter(|record| record.identity.signature == parent_signature)
            .max_by(|a, b| {
                a.current
                    .source_duration
                    .partial_cmp(&b.current.source_duration)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
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
    /// 测试用：右半段登记的父 item（宿主分割谱系）。
    #[cfg(test)]
    pub(crate) fn split_parent_of(&self, right: &str) -> Option<&str> {
        self.split_parents.get(right).map(String::as_str)
    }
    /// 测试用：右半段是否保留了父段的完整窗口 seed（`find` 的包含判定靠它）。
    #[cfg(test)]
    pub(crate) fn has_split_basis(&self, right: &str) -> bool {
        self.regions.contains_key(&format!("split-basis:{right}"))
    }
    /// 同一源曲线可由历史/区域共享，额度按reservation身份去重，而不是按序列化值重复计数。
    // 额度统计诊断入口，保留供预算核对。
    #[cfg(test)]
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
            // 【为什么身份/根缺席也只跳过】与下面几何失败同一条纪律：本函数在**整份**
            // 接受路径上跑（`capture_render_input`、工作区接受都经过它）。此前这两处
            // `ok_or(...)?` 排在几何 `continue` **之前**，于是"身份表这一轮还没跟上"
            // （region 已建、派生表未同步的窗口）会让整份投影失败 —— 而那条降级通道
            // 明明就在下面几行。挪到同一口径：这一条片段跳过，其余照常。
            let Some(identity) = identities.get(&clip.id) else {
                continue;
            };
            let Some(root) = timeline.resolve_root_track_id(&clip.track_id) else {
                continue;
            };
            // 【为什么几何失败只跳过而不是报错】本函数在**整份**接受路径上跑
            // （`capture_render_input`、工作区接受都经过它）。一个片段若在这里 `Err`，
            // 整份文档就渲染不出来 —— 与 `ensure_loaded` 里那句 `any(reversed)` 是同一个
            // "把局部状态升级成全局故障"的错。几何不可解（退化窗口等）时如实跳过它。
            let Ok(geometry) = geometry(clip) else {
                continue;
            };
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
            // 【为什么两个前缀都要认】`split-basis:` 是 item 版（`register_split`），
            // `ara-split-basis:` 是 ARA 签名版（`register_ara_split`）。此前这里只
            // `strip_prefix("split-basis:")`，而 `"ara-split-basis:x"` 不以它开头 ⇒
            // 返回 `None` ⇒ `is_none_or` 为真 ⇒ **ARA 版副本永不回收**：每次分割都留下
            // 一条带完整曲线的记录，长会话里无界增长，`validate()` 到 16384 条后
            // 整份图集被拒。
            id.strip_prefix("split-basis:")
                .or_else(|| id.strip_prefix("ara-split-basis:"))
                .is_none_or(|right| !bound.contains(right))
        });
        followed
            .split_parents
            .retain(|right, _| !bound.contains(right));
        followed.copy_seeds.retain(|item, _| !bound.contains(item));
        // 【为什么回收"被活记录覆盖"的历史祖先】`follow_geometry` 只把历史记录置
        // `live = false`、从不删除，于是它们随**分割次数**无界累积（每次分割都留下
        // 分割前的父段）。而几何兜底（`find`）会把这些死记录一并当作候选 ——
        // 累积越多，"多候选 ⇒ 猜不出"越常见，`regions` 也会在 16384 条时被
        // `validate()` 整份拒绝。
        //
        // 【判据为什么是"被活记录的**并集**覆盖"】一次分割的父段窗口恰好等于左右两半
        // 的并集：父段 [0,1) 被 left [0,0.5) 与 right [0.5,1) 合起来盖满，但**不被其中
        // 任何一条单独**包含。所以判据必须是并集，逐条包含判定永远回收不掉分割祖先
        // （那是本次实现过程中实测到的：逐条判定对分割场景一次都不成立）。
        //
        // 【为什么删掉是安全的】父段窗口若被同源同修改的活记录并集完全盖住，它对兜底层
        // 就不提供任何信息：任何落在它内部的查询窗口必然也落在活记录里。删掉它不改变
        // 任何解析结果，只让候选池变小。
        //
        // 【为什么不动"未被覆盖"的死记录】那可能是用户删掉的片段，撤销/重做会把它
        // 带回来；它仍是重做后那条片段的祖先。留着是对的（代价是有限的）。
        let mut coverage: BTreeMap<(String, String), Vec<(f64, f64)>> = BTreeMap::new();
        for record in followed.regions.values().filter(|record| record.live) {
            coverage
                .entry((
                    record.identity.source.clone(),
                    record.identity.modification.clone(),
                ))
                .or_default()
                .push((
                    record.current.source_start,
                    record.current.source_start + record.current.source_duration,
                ));
        }
        for intervals in coverage.values_mut() {
            intervals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut merged: Vec<(f64, f64)> = Vec::with_capacity(intervals.len());
            for (start, end) in intervals.drain(..) {
                match merged.last_mut() {
                    // 相邻/相接也合并：分割产生的是**紧接**的两段。
                    Some(last) if start <= last.1 + 1e-6 => last.1 = last.1.max(end),
                    _ => merged.push((start, end)),
                }
            }
            *intervals = merged;
        }
        followed.regions.retain(|id, record| {
            if record.live || id.starts_with("split-basis:") || id.starts_with("ara-split-basis:") {
                return true;
            }
            let end = record.current.source_start + record.current.source_duration;
            let covered = coverage
                .get(&(
                    record.identity.source.clone(),
                    record.identity.modification.clone(),
                ))
                .is_some_and(|intervals| {
                    intervals.iter().any(|(start, stop)| {
                        record.current.source_start >= *start - 1e-6 && end <= *stop + 1e-6
                    })
                });
            !covered
        });
        // 【为什么也要裁剪签名谱系】`ara_split_parents` / `ara_copy_seeds` 以**签名**为键，
        // 而签名含播放几何 —— 一条 region 被移动/裁剪后它的签名就变了，旧条目再也不会
        // 被查到，却一直留着（每次分割/复制加一条，无界）。只保留"右半段签名仍对应某条
        // 记录"的条目：那正是 `find` 唯一会去查的那些。
        let live_signatures: std::collections::BTreeSet<String> = followed
            .regions
            .values()
            .map(|record| record.identity.signature.clone())
            .filter(|signature| !signature.is_empty())
            .collect();
        followed
            .ara_split_parents
            .retain(|right, _| live_signatures.contains(right));
        followed
            .ara_copy_seeds
            .retain(|signature, _| live_signatures.contains(signature));
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
                // 倒放与正放都参与绑定：几何的 `reversed` 让候选按真实消费窗口比较，
                // 方向不同即窗口不同，自然区分。
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
                            geometry.reversed == record.current.reversed
                                && near(geometry.source_start, record.current.source_start)
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
            // 与 `capture_changes` 同一条理由：几何不可解的片段只跳过它自己，
            // 绝不能让一条片段使整份投影失败（本函数在渲染输入路径上）。
            //
            // 【为什么根解析也排到几何之后】此前 `ok_or("unknown parameter root")?` 排在
            // 几何 `continue` **之前**，于是"轨道身份这一轮还查不到"会让整份投影失败 ——
            // 而按设计它本该被下面这条 `continue` 兜住。两处口径统一到"局部跳过"。
            let Ok(geometry) = geometry(clip) else {
                continue;
            };
            let Some(root) = timeline.resolve_root_track_id(&clip.track_id) else {
                continue;
            };
            let Some((begin, end)) = covered_frame_range(&geometry, params.frame_period_ms)? else {
                continue;
            };
            let entry = roots.entry(root).or_insert_with(|| {
                let mut entry = params.clone();
                clear_curves(&mut entry);
                entry
            });
            if entry.frame_period_ms != params.frame_period_ms {
                // 【为什么不再整份失败】同一根轨道上两条片段声明了不同的帧周期，
                // 合并它们的曲线会把值按错误的时间轴插值。此前这里 `Err` ⇒ 整份投影
                // 失败 ⇒ 整份编辑被拒。改为跳过这一条片段：它的曲线本次不并入根视图，
                // 其余片段照常。
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Warn,
                    "ara.atlas.frame-period-mismatch",
                    format_args!(
                        "[ara] clip {} declares {} ms frames but its root already uses {} ms; \
                         its curves are not merged into the root view",
                        clip.id, params.frame_period_ms, entry.frame_period_ms
                    ),
                );
                continue;
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
            // 【为什么身份缺席只跳过】与几何不可解同一纪律（见下方注释）。此前这里
            // `ok_or(...)?` 会把"身份表这一轮还没跟上"升级成整份捕获失败，于是用户的
            // 一次编辑被整份拒绝。跳过它：该片段这次不落曲线，其余片段照常。
            let Some(identity) = identities.get(&clip.id) else {
                continue;
            };
            if identity.key == 0
                || identity.source.is_empty()
                || identity.modification.is_empty()
                || clip.source_path.as_deref() != Some(identity.source.as_str())
            {
                // 身份自相矛盾（key 为 0 / 源为空 / 与 clip 的源不符）时同样只跳过这一条：
                // 拿一个不可信的身份去写曲线，会把参数挂到错误的片段上 —— 那比不写更糟。
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Warn,
                    "ara.atlas.invalid-identity",
                    format_args!(
                        "[ara] clip {} has an invalid parameter identity this round; its curves \
                         are not captured (other clips are unaffected)",
                        clip.id
                    ),
                );
                continue;
            }
            let Some(root) = timeline.resolve_root_track_id(&clip.track_id) else {
                continue;
            };
            let Some(params) = timeline.params_by_root_track.get(&root) else {
                candidate.regions.remove(&clip.id);
                continue;
            };
            // 几何不可解只跳过该片段：否则一条退化片段会让整份捕获失败。
            let Ok(geometry) = geometry(clip) else {
                continue;
            };
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
            // 【为什么身份/根缺席也只跳过】与下面几何失败同一条纪律。此前这两处
            // `ok_or(...)?` 排在几何 `continue` 之前，于是"身份表/轨道身份这一轮还没
            // 跟上"会让整个渲染输入准备失败（本函数在 `capture_render_input` 上）⇒
            // 快照永不发布 ⇒ 声音停在旧版本。跳过这一条片段即可。
            let Some(identity) = identities.get(&clip.id) else {
                continue;
            };
            let Some(root) = timeline.resolve_root_track_id(&clip.track_id) else {
                continue;
            };
            // 几何不可解只跳过该片段：否则整个渲染输入准备都会失败
            // （本函数在 `capture_render_input` 上）。
            let Ok(mut geometry) = geometry(clip) else {
                continue;
            };
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
        // 【ARA-only 的复制谱系】没有 item GUID 时（Cubase / Nuendo / Studio One），
        // 复制种子按 region 签名索引 —— 同一套判定，换一个任何宿主都有的键。
        if !identity.signature.is_empty() {
            if let Some(seed) = self.ara_copy_seed(&identity.signature) {
                return Ok(Some(seed));
            }
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
                    // 【为什么不再报错】多候选说明"祖先猜不出"，与几何兜底层同一条纪律：
                    // 返回 `None`（该片段不投影曲线）远好于把一次内部推断失败变成用户的
                    // "音频块编辑被拒绝"（见 `log_ambiguous_ancestry`）。
                    _ => {
                        Self::log_ambiguous_ancestry("confirmed split", identity, candidates.len());
                        return Ok(None);
                    }
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
        // 【ARA-only 的分割谱系】没有 item GUID 时走签名版：`ara_split_parents` 记的是
        // "右半段签名 → 父段签名"，父段记录按**签名**匹配而不是 item。
        //
        // 【为什么这一层必须在几何兜底之前】Cubase 上报错最多的
        // `Conflict: ambiguous source parameter ancestry` 就来自几何兜底：两个半段共享
        // 同一个 (source, modification) 且窗口相邻，兜底层把它们都判成候选。有了显式谱系，
        // 这里直接命中父段，不再靠猜。
        if !identity.signature.is_empty() {
            if let Some(parent_signature) = self.ara_split_parent(&identity.signature) {
                let parent_signature = parent_signature.to_owned();
                let candidates = self
                    .regions
                    .values()
                    .filter(|record| {
                        record.identity.signature == parent_signature
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
                    // 同上：ARA 签名谱系也猜不出时降级为"不投影曲线"，不拒绝整份编辑。
                    _ => {
                        Self::log_ambiguous_ancestry("ara split", identity, candidates.len());
                        return Ok(None);
                    }
                }
            }
        }
        // 【兜底层：没有谱系时按源窗猜祖先 —— 判据必须是**包含**而不是**重叠**】
        //
        // 这一层只在没有显式谱系（`split_parents`）可用时才跑，因此必须保守：
        //   * **重叠**会把**兄弟半段**判成合法祖先。分割后两半的源窗恰好相邻/相接，
        //     重叠判据让右半段继承左半段的 `SourceCurve.basis`，曲线被整体平移一个
        //     分割长度（用户报障：分割含倒放 Take 的 Item 后偏移；几何正确、曲线错位）。
        //     改成包含后，只有真正**包住**本片段窗口的祖先才合格，兄弟半段被排除。
        //   * 多候选说明"猜不出"。返回 `None`（不画曲线）远好于返回**错**曲线；
        //     也远好于把"猜不出"当 `Err` 抛给用户 —— 那会变成一次
        //     "音频块编辑被拒绝"，让用户为一个内部推断失败负责。
        //
        // 这里保留"优先活记录、必要时退回非活记录"的口径：非活记录正是**分割前**的
        // 父段，用它才能让两半继承到正确祖先。只有显式谱系（F1.2）铺开后才应进一步收紧。
        let candidates = related
            .filter(|record| {
                record.identity.key != 0
                    && record.root == root
                    && geometry.source_start >= record.current.source_start - 1e-6
                    && geometry.source_start + geometry.source_duration
                        <= record.current.source_start + record.current.source_duration + 1e-6
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
            // 【为什么不再报错 —— 用户报障的直接修复】两条候选都合格说明"祖先猜不出"。
            // 此前这里返回 `Err`，于是**一个片段的曲线身份对不上**被放大成**整份编辑被
            // 拒绝**：用户在 Cubase 上看到的就是 `Conflict: ambiguous source parameter
            // ancestry`，而它其实只是一次内部推断的失败，不该由用户为一个插件内部问题负责。
            //
            // 返回 `None` 的语义是"这个片段本次不投影曲线"：几何、PCM、其余片段全部照常
            // 渲染。这正是本层上方注释早就写明的口径（"返回 None 远好于返回 Err"），
            // 也是与 `geometry()` 不可解时 `continue` 完全同一条纪律。
            //
            // 【为什么降级是安全的】真歧义（两条活记录都包含本窗口）本来就没有正确答案：
            // 取最小范围或"最后写入"都会把**错**的曲线画到片段上，那比不画更糟 ——
            // 用户会看到参数被莫名改动，却没有任何提示。留一行限流日志，让"某个片段
            // 丢了曲线"在日志里可复盘，而不是让用户去猜。
            _ => {
                Self::log_ambiguous_ancestry("geometric containment", identity, candidates.len());
                Ok(None)
            }
        }
    }
    /// 祖先推断多候选时的限流留痕。
    ///
    /// 【为什么是日志而不是错误】见几何兜底层的长注释：多候选说明"猜不出"，
    /// 返回 `None`（该片段本次不投影曲线）远好于把一次内部推断失败变成用户的
    /// "音频块编辑被拒绝"。但静默同样不行 —— "某个片段的曲线没跟上"必须是
    /// 日志里可复盘的事实，否则用户只会看到"参数不生效"而无从判断原因。
    ///
    /// 限流键固定，避免每帧投影都对同一片段重复刷屏。
    fn log_ambiguous_ancestry(kind: &str, identity: &RegionIdentity, candidates: usize) {
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            "ara.atlas.ambiguous-ancestry",
            format_args!(
                "[ara] parameter ancestry is ambiguous ({kind}): {candidates} candidate(s) for \
                 source={} modification={}; this clip projects no curves until an edit \
                 disambiguates it (geometry and audio are unaffected)",
                identity.source, identity.modification
            ),
        );
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
            // 覆盖只用到**时间线跨度**（`covered_frame_range` 只看 project_start/duration），
            // 与方向无关 —— 倒放 clip 也必须计入覆盖，否则它的跨度会被当成空白层，
            // 用户在其上的编辑被误捕获进 gap 层。
            coverage.entry(root).or_default().push(RegionGeometry {
                project_start: clip.start_sec,
                project_duration: clip.length_sec,
                source_start: clip.source_start_sec,
                source_duration: clip.length_sec * clip.playback_rate as f64,
                reversed: clip.reversed && !clip.loop_enabled,
            });
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

/// 用**真实消费窗口**而不是 take 媒体全长；宿主 ARA 已经把总倍率投影到 active take。
///
/// 【倒放为什么要重算窗口】倒放的消费窗口是 `[se−len·r, se)`（`clip_playback_window_sec`），
/// 而 `clip.source_start_sec` 在倒放里只是编辑痕迹、**不参与消费数学**。若照搬
/// `source_start_sec` 建升序映射，曲线会被整体镜像 —— 与波形/音频对不上。
/// 这里取真实窗口并把 `reversed` 交给 [`RegionGeometry::map`] 建降序源轴。
///
/// 公开给同 crate 的剪贴板粘贴路径，避免"窗口/方向"这套规则被抄成两份。
pub(crate) fn geometry(clip: &hifishifter_kernel::state::Clip) -> Result<RegionGeometry, String> {
    // Loop 有自己的回绕锚点语义，不适用"窗口终点递减"模型；沿用既有的正放近似。
    let reversed = clip.reversed && !clip.loop_enabled;
    let (source_start, source_duration) = if reversed {
        let (window_start, window_end) = hifishifter_kernel::state::clip_playback_window_sec(clip);
        (window_start, window_end - window_start)
    } else {
        (
            clip.source_start_sec,
            clip.length_sec * clip.playback_rate as f64,
        )
    };
    let value = RegionGeometry {
        project_start: clip.start_sec,
        project_duration: clip.length_sec,
        source_start,
        source_duration,
        reversed,
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
    if !frame_ms.is_finite() || !(0.1..=1000.).contains(&frame_ms) || geometry.project_start < 0. {
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
                signature: String::new(),
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
    /// **ARA-only 宿主（无 item GUID）的分割谱系必须按签名命中。**
    ///
    /// 【为什么这是回归测试】`find` 的解析顺序此前只认 item GUID：`copy_seeds`、
    /// 精确 item、`split_parents` 三条路全都要求 `identity.item` 有值，而它在
    /// Cubase / Nuendo / Studio One 上恒为 `None`（`region_items` 需要原生扩展）。
    /// 于是每一次分割都落到几何包含兜底 —— 两个半段共享同一个 (source, modification)
    /// 且窗口相邻，兜底层把它们都判成候选，报
    /// `Conflict: ambiguous source parameter ancestry`，整份编辑被拒。
    ///
    /// 这条测试摆出"有原生扩展时能解析、没有时也能解析"的同一形状，差别只在 item。
    #[test]
    fn an_ara_only_split_resolves_by_signature_without_a_host_item_guid() {
        let original = edited();
        // 父段的 identity 必须带上签名，`ara_split_seed` 才认得出它 —— 生产路径由
        // `parameter_identities_locked` 填（见 `DocumentSession`）。
        let parent_signature = ParameterAtlas::ara_region_signature("source", "mod", 0.0, 1.0);
        let mut parent_ids = identities();
        if let Some(identity) = parent_ids.get_mut("clip") {
            identity.signature = parent_signature.clone();
        }
        let atlas = ParameterAtlas::default()
            .capture(&original, &parent_ids)
            .unwrap();
        // 父段（分割前）有曲线，因此有可继承的 seed。
        assert!(atlas.regions["clip"].live);

        // 一次宿主分割：父段消失，左右两半出现。**都不带 item GUID** —— ARA-only 常态。
        let right_signature = ParameterAtlas::ara_region_signature("source", "mod", 0.5, 0.5);
        let mut split = atlas.clone();
        let seed = split.ara_split_seed(&parent_signature).expect("父段有曲线");
        split
            .register_ara_split(&parent_signature, &right_signature, Some(seed))
            .expect("谱系登记");

        let mut timeline = host(1.5, 0.5, 0.5, 0.5);
        timeline.clips[0].id = "right".into();
        timeline.project_sec = 2.;
        let ids = BTreeMap::from([(
            "right".to_string(),
            RegionIdentity {
                key: 43,
                item: None,
                source: "source".into(),
                modification: "mod".into(),
                signature: right_signature.clone(),
            },
        )]);
        let followed = split.follow_geometry(&timeline, &ids).unwrap();
        // 谱系命中父段 ⇒ 右半段继承到曲线，而不是报 ambiguity。
        assert!(
            followed.regions.contains_key("right"),
            "右半段必须拿到一条记录：{:?}",
            followed.regions.keys().collect::<Vec<_>>()
        );
        assert_eq!(
            split.ara_split_parent(&right_signature),
            Some(parent_signature.as_str())
        );
    }

    /// 没有谱系、也没有 item GUID 时，几何兜底仍然工作（不回归成硬错误）。
    #[test]
    fn an_ara_only_region_without_lineage_still_falls_back_to_geometry() {
        let original = edited();
        let atlas = ParameterAtlas::default()
            .capture(&original, &identities())
            .unwrap();
        let timeline = host(1., 0.5, 0., 0.5);
        let ids = BTreeMap::from([(
            "clip".to_string(),
            RegionIdentity {
                key: 1,
                item: None,
                source: "source".into(),
                modification: "mod".into(),
                signature: ParameterAtlas::ara_region_signature("source", "mod", 0.0, 0.5),
            },
        )]);
        // 一个被父窗口包含的片段：唯一候选 ⇒ 解析成功，不报错。
        let followed = atlas.follow_geometry(&timeline, &ids).unwrap();
        assert!(followed.regions.contains_key("clip"));
    }

    /// **用户报障的回归测试：两次嵌套分割后，一条新片段被两条已死祖先同时包含。**
    ///
    /// 【为什么必须单独钉住】单次分割的测试**全部通过**而用户仍然报错，因为触发形状
    /// 是"两次以上嵌套分割"：`follow_geometry` 对历史父记录只置 `live = false`、
    /// 从不删除，于是祖先会累积。一条源窗跨在两个历史祖先之间的新片段（复制/粘贴、
    /// 撤销重做后宿主重建、或窗口恰好落在两者交集内）会让几何兜底同时命中 P 与 L，
    /// 多候选 ⇒ 此前抛 `Conflict: ambiguous source parameter ancestry` ⇒ 整份编辑被拒。
    ///
    /// 现在必须：**不报错**，且该片段**不投影曲线**（宁可不画，不可画错）。
    #[test]
    fn a_region_spanning_two_dead_ancestors_is_not_an_error() {
        let original = edited();
        let atlas = ParameterAtlas::default()
            .capture(&original, &identities())
            .unwrap();
        let with_signature = |id: &str, key: u64, ps: f64, pd: f64| {
            (
                id.to_string(),
                RegionIdentity {
                    key,
                    item: None,
                    source: "source".into(),
                    modification: "mod".into(),
                    signature: ParameterAtlas::ara_region_signature("source", "mod", ps, pd),
                },
            )
        };

        // 第一次分割：父段 [0,1) -> 左 [0,0.5) + 右 [0.5,1)。
        let mut first = host(1., 0.5, 0., 0.5);
        first.clips[0].id = "left".into();
        let mut right = host(1.5, 0.5, 0.5, 0.5).clips.remove(0);
        right.id = "right".into();
        first.clips.push(right);
        first.project_sec = 2.;
        let first_ids = BTreeMap::from([
            with_signature("left", 42, 1.0, 0.5),
            with_signature("right", 43, 1.5, 0.5),
        ]);
        let a1 = atlas.follow_geometry(&first, &first_ids).unwrap();

        // 第二次分割：左半段 [0,0.5) -> [0,0.25) + [0.25,0.5)。
        let mut second = host(1., 0.25, 0., 0.25);
        second.clips[0].id = "quarter-a".into();
        let mut qb = host(1.25, 0.25, 0.25, 0.25).clips.remove(0);
        qb.id = "quarter-b".into();
        second.clips.push(qb);
        second.clips.push(first.clips[1].clone());
        second.project_sec = 2.;
        let second_ids = BTreeMap::from([
            with_signature("quarter-a", 44, 1.0, 0.25),
            with_signature("quarter-b", 45, 1.25, 0.25),
            with_signature("right", 43, 1.5, 0.5),
        ]);
        let a2 = a1.follow_geometry(&second, &second_ids).unwrap();
        // 两级祖先都被各自后代的并集盖满 ⇒ 全部回收，不再累积。
        assert!(
            !a2.regions.contains_key("clip") && !a2.regions.contains_key("left"),
            "嵌套分割的祖先必须被回收，否则每次分割都在候选池里加一条死记录：{:?}",
            a2.regions.keys().collect::<Vec<_>>()
        );
        assert!(a2.regions["quarter-a"].live);
        assert!(a2.regions["quarter-b"].live);
        assert!(a2.regions["right"].live);

        // 【歧义的可达形状】并集回收消掉了"分割累积"，但**未被覆盖**的死记录仍然
        // 存在（用户删掉的片段、撤销/重做会带回来的片段）。两条这样的记录若窗口
        // 重叠，几何兜底就会同时命中它们 —— 那正是用户报的
        // `Conflict: ambiguous source parameter ancestry`。
        //
        // 摆出该形状：同源同修改的两条片段（同一窗口、不同工程位置），用户把它们
        // **都删了**（时间线变空 ⇒ 两条记录都变非活，且无活记录可覆盖它们）。
        let mut two: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0}],"bpm":120,"project_sec":2.0,"clips":[]
        }))
        .unwrap();
        two.clips.push(host(1., 1., 0., 1.).clips.remove(0));
        let mut second_copy = host(3., 1., 0., 1.).clips.remove(0);
        second_copy.id = "clip2".into();
        two.clips.push(second_copy);
        let two_ids = BTreeMap::from([
            with_signature("clip", 41, 1.0, 1.0),
            with_signature("clip2", 42, 3.0, 1.0),
        ]);
        let atlas = ParameterAtlas::default()
            .capture(&edited(), &identities())
            .unwrap()
            .follow_geometry(&two, &two_ids)
            .unwrap();
        assert!(atlas.regions["clip"].live && atlas.regions["clip2"].live);

        // 全删：两条记录都变非活，没有任何活记录覆盖它们 ⇒ 都留着（撤销要能找回）。
        let empty: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0}],"bpm":120,"project_sec":2.0,"clips":[]
        }))
        .unwrap();
        let orphaned = atlas.follow_geometry(&empty, &BTreeMap::new()).unwrap();
        assert!(!orphaned.regions["clip"].live && !orphaned.regions["clip2"].live);
        assert_eq!(orphaned.regions.len(), 2, "未被覆盖的死记录必须保留");

        // 新片段出现，窗口 [0.1,0.9) 同时落在上面两条死记录内 ⇒ 真歧义。
        let mut third: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0}],"bpm":120,"project_sec":4.0,"clips":[]
        }))
        .unwrap();
        third.clips.push(host(1., 0.8, 0.1, 0.8).clips.remove(0));
        let third_ids = BTreeMap::from([with_signature("clip", 43, 1.0, 0.8)]);
        let resolved = orphaned
            .follow_geometry(&third, &third_ids)
            .expect("猜不出祖先不得让整份编辑失败（用户报障的直接修复）");
        // 新片段沿用了 id "clip"；关键是它**没有**被认领成 live（即没拿到任何一条
        // 祖先的曲线），而不是"键不存在"。
        assert!(
            !resolved.regions["clip"].live,
            "猜不出祖先的片段不得拿到曲线（两条死记录仍应保持非活）：{:?}",
            resolved
                .regions
                .iter()
                .map(|(k, r)| (k.as_str(), r.live))
                .collect::<Vec<_>>()
        );
        assert!(!resolved.regions["clip2"].live);
        // 投影与编辑接受都必须成功（此前它们都在这里抛 Conflict）。
        resolved
            .project_roots(&third, &third_ids)
            .expect("歧义不得让整份投影失败");
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
                    signature: String::new(),
                },
            ),
            (
                "right".into(),
                RegionIdentity {
                    key: 43,
                    item: None,
                    source: "source".into(),
                    modification: "mod".into(),
                    signature: String::new(),
                },
            ),
        ]);
        let followed = atlas.follow_geometry(&first, &ids).unwrap();
        // 【为什么是 2 而不是 3】父段 [0,1) 被左右两半的**并集**恰好盖满，因此作为
        // 冗余祖先被回收（见 `follow_geometry` 的并集覆盖判据）。此前它作为一条
        // 非活记录永久留着 —— 那正是祖先随分割次数无界累积、几何兜底越来越容易
        // 多候选的根源。
        assert_eq!(followed.regions.len(), 2);
        assert!(
            !followed.regions.contains_key("clip"),
            "被活记录并集覆盖的父段必须回收：{:?}",
            followed.regions.keys().collect::<Vec<_>>()
        );
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
                signature: String::new(),
            },
        );
        next_ids.insert(
            "quarter-b".into(),
            RegionIdentity {
                key: 45,
                item: None,
                source: "source".into(),
                modification: "mod".into(),
                signature: String::new(),
            },
        );
        let nested = followed.follow_geometry(&next, &next_ids).unwrap();
        let local = nested.project_local(&next, &next_ids).unwrap();
        assert_eq!(local["quarter-a"].pitch_edit, &[60., 61.]);
        assert_eq!(local["quarter-b"].pitch_edit, &[61., 62.]);
        assert_eq!(local["right"].pitch_edit, &[62., 63., 64.]);
        // 两级祖先都被后代并集盖满 ⇒ 回收，不再作为死记录留在候选池里。
        assert!(
            !nested.regions.contains_key("left") && !nested.regions.contains_key("clip"),
            "嵌套分割的祖先必须回收：{:?}",
            nested.regions.keys().collect::<Vec<_>>()
        );
        // 【歧义本身的可达形状与降级口径】已由
        // `a_region_spanning_two_dead_ancestors_is_not_an_error` 覆盖：并集回收消掉了
        // "分割累积"，但**未被覆盖**的死记录（用户删掉的片段）仍会重叠，那里验证
        // "猜不出 ⇒ 不投影曲线而不是拒绝整份编辑"。
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
                signature: String::new(),
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
                signature: String::new(),
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
    /// 倒放 clip **参与**参数投影：同一源曲线经降序窗口投影后，项目时间上的取值相对正放
    /// **镜像**（源位置随时间递减）。这条把"倒放无曲线"的老口径钉死为历史。
    #[test]
    fn a_reversed_clip_projects_the_source_curve_mirrored() {
        let ids = identities();
        let forward = edited();
        let atlas = ParameterAtlas::default().capture(&forward, &ids).unwrap();
        let forward_view = atlas.project_local(&forward, &ids).unwrap()["clip"]
            .pitch_edit
            .clone();
        assert!(!forward_view.is_empty());

        // 同一源窗口 [0,1)、同一项目 [1,2)，只是消费方向相反。
        let mut reversed = forward.clone();
        reversed.clips[0].reversed = true;
        let reversed_view = atlas.project_local(&reversed, &ids).unwrap()["clip"]
            .pitch_edit
            .clone();

        let expected = forward_view.iter().rev().copied().collect::<Vec<_>>();
        assert_eq!(reversed_view, expected, "倒放投影必须是正放投影的时间镜像");
    }

    /// 倒放几何的源窗口取自真实消费窗口（`[se−len·r, se)`），而不是存储的
    /// `source_start_sec`，并且方向位被记进 `RegionGeometry`。
    #[test]
    fn reversed_geometry_uses_the_consumption_window_and_marks_direction() {
        let mut clip = edited().clips[0].clone();
        clip.reversed = true;
        clip.source_start_sec = 7.; // 陈旧/编辑痕迹，不参与倒放消费数学
        clip.source_end_sec = 1.;
        clip.length_sec = 1.;
        clip.playback_rate = 1.;
        let geometry = geometry(&clip).unwrap();
        assert!(geometry.reversed);
        assert!((geometry.source_start - 0.).abs() < 1e-12);
        assert!((geometry.source_duration - 1.).abs() < 1e-12);
        // 降序映射：项目起点 ⇒ 源窗口终点。
        let map = geometry.map().unwrap();
        assert!((map.project_to_source(clip.start_sec).unwrap() - 1.).abs() < 1e-12);
        assert!(
            (map.project_to_source(clip.start_sec + clip.length_sec)
                .unwrap()
                - 0.)
                .abs()
                < 1e-12
        );
    }

    /// 一条几何不可解的片段只跳过它自己，绝不把局部状态升级成整份投影/捕获失败。
    #[test]
    fn a_degenerate_clip_is_skipped_without_failing_the_whole_projection() {
        let mut ids = identities();
        let mut timeline = edited();
        // 退化窗口：跨度为零 ⇒ 无法建映射。此前靠"倒放即跳过"回避，现在靠几何失败兜底。
        let mut degenerate = timeline.clips[0].clone();
        degenerate.id = "degenerate".into();
        degenerate.start_sec = 3.;
        degenerate.length_sec = 0.;
        degenerate.source_start_sec = 0.;
        degenerate.source_end_sec = 0.;
        timeline.clips.push(degenerate);
        timeline.project_sec = 4.;
        ids.insert(
            "degenerate".into(),
            RegionIdentity {
                key: 99,
                item: None,
                source: "source".into(),
                modification: "mod".into(),
                signature: String::new(),
            },
        );
        let atlas = ParameterAtlas::default().capture(&timeline, &ids).unwrap();
        let projected = atlas.project_local(&timeline, &ids).unwrap();
        assert!(projected.contains_key("clip"), "正放 clip 必须照常投影");
        assert!(!projected.contains_key("degenerate"), "退化 clip 无曲线");
        assert!(!atlas
            .capture(&timeline, &ids)
            .unwrap()
            .regions
            .contains_key("degenerate"));
        assert!(atlas.follow_geometry(&timeline, &ids).is_ok());
        assert!(atlas.rebind(&timeline, &ids).is_ok());
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
                signature: String::new(),
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
                signature: String::new(),
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
                signature: String::new(),
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
