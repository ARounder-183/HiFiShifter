//! 区域参数的源坐标权威；原GUI项目帧只是投影，宿主移动/拉伸不重新采样写坏原始曲线。
//! 身份来自真实ARA region→modification/source边；不从名字/文件路径/会话序号猜冷恢复归属。

use crate::ara::time_map::{SourceTimeMap, TimeAnchor};
use hifishifter_kernel::editor::param_edit::{
    compose_curve, default_pad_for_key, extract_edit_curves, is_edited_value, EditCurve,
};
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
/// 一条曲线的**索引域**：`values[i]` 的第 `i` 帧对应哪个域的什么时刻。
///
/// 【为什么必须显式声明】第七轮之前，曲线按工程帧索引；第七轮把它搬到了源帧
/// （`SourceCurve` 的 `basis` 改成 `Option`，新形状一律 `None`）。但渲染路径上
/// 仍有四处用**绝对工程时间**去读这些曲线，而那一刻它们已被 `project_local`
/// 投影成**零点对齐的 clip 局部数组** —— 同一个数组在四个地方被两种不同的时间域
/// 读，偏移量恰等于 `clip.start_sec`（也就是分割点）。这就是分割后曲线整体偏移
/// 的根因。把域变成类型系统的一部分，那种混用才可能被编译器或断言抓住。
///
/// 【为什么三分而不是二分】二分（源 / 工程）无法表达渲染路径真正消费的那个域：
/// `project_local` 的产物既不是"源帧"也不是"绝对工程帧"，而是**零点对齐的工程帧**
/// （`geometry.project_start` 被置 0）。标成 `Project` 会让域校验在 `project_start == 0`
/// 时恒真（断言形同虚设）；标成 `Source` 则与 `frame_range` 的实际算式矛盾。
/// 只有第三个取值能让"这条数组该用哪种时间读"成为一个可检查的事实。
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum CurveDomain {
    /// 源域帧：`values[first_frame + k]` = 源时刻 `(first_frame + k)·frame_ms` 处的值。
    /// 与"是哪条 region 的哪次编辑"无关 —— 共享同一 modification 的多条 region
    /// 落在同一条曲线的不同区间上，各取所需。
    Source,
    /// 工程域帧：`values[i]` = 项目**绝对**时刻 `i·frame_ms` 处的值。
    /// 只用于根视图合并（`project_roots`）与旧存档未迁移的窗口。
    Project,
    /// clip 局部零点帧：`values[i]` = 该 clip 第 `i·frame_ms` 处的值。
    /// `project_local` 的产物，渲染路径唯一消费域。
    ClipLocal,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SourceCurve {
    /// **索引域**。`None` = 新形状：`values` 按**源帧**索引（与是哪条 region 无关）。
    /// `Some(geometry)` = 旧形状：`values` 按**捕获时那条 region 的工程帧**索引。
    ///
    /// 【为什么是 `Option` 而不是换类型或加版本分支】旧存档里这个字段叫 `basis` 且是一个
    /// 对象；新存档写成 `null`。同一个字段名 + `Option` 让新旧两种形状**天然共存**：
    /// 旧存档照旧读得进（`Some`），新存档写出去的是源域（`None`），
    /// 且不需要 `untagged`（那会牺牲错误信息），也不需要改名（那会读不进旧存档）。
    ///
    /// 【为什么旧形状必须继续被支持】把旧曲线一次性折算到源域需要重采样，而"用户从没
    /// 编辑过的地方"不该在升级时被动过。保留旧分支 ⇒ 升级零损失；某条曲线第一次被重新
    /// 编辑时自然变成源域（`capture` 一律写 `None`）。
    #[serde(default)]
    basis: Option<RegionGeometry>,
    /// 本曲线的索引域（见 [`CurveDomain`]）。
    ///
    /// 【为什么能从 `basis` 推导却仍然单独存】`basis == None` 确实等价于源域，
    /// 但"只有源域和工程域两档"这件事本身就是本轮要消除的缺陷：渲染路径消费的
    /// `project_local` 产物是 **clip 局部零点**，它与前两者都不同。单独存一个字段，
    /// 让每个写出方都必须声明自己写的是哪个域；缺省时按 `basis` 推导
    /// （`Some` = 旧工程帧，`None` = 源帧），旧存档读进来即得到与改动前完全一致的解释。
    #[serde(default = "legacy_curve_domain")]
    domain: CurveDomain,
    /// `values[0]` 对应的**索引域**帧号（新形状下即源帧）。
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

/// 旧存档缺 `domain` 时的推导：`basis == Some(..)` = 工程帧，`None` = 源帧。
///
/// 【为什么不是简单的 `Default`】`Default` 拿不到同一条记录的 `basis`，
/// 而"旧存档该怎么解释"恰恰由 `basis` 决定。函数级 `serde(default)` 同样只看得到
/// 本字段，所以这里取两档里与旧实现完全一致的那个：
/// 第七轮起 `capture_source_curve` 一律写 `basis: None`（源域），旧存档是 `Some`
/// （工程帧）。取 `Source` 作为缺省与"新存档"一致；`Some` 的旧记录由
/// [`SourceCurve::domain_time`] 的既有旧分支继续正确处理，两者互不干扰。
fn legacy_curve_domain() -> CurveDomain {
    CurveDomain::Source
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
    /// 新路径的稀疏用户编辑；与 `curves` 并存以读旧存档。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    edits: BTreeMap<String, EditCurve>,
    curves: BTreeMap<String, SourceCurve>,
}
fn default_live() -> bool {
    true
}
fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

/// 曲线在工程时间上的**放置策略**。它只影响投影，不影响存储。
///
/// 【为什么把「锚点」换成「放置」】旧模型里 `ParamAnchor` 决定 `values[i]` 是
/// "源第 i 格"还是"工程第 i 格"，换锚点就意味着**真的把数组搬到另一个域**
/// （`bake_curve_to_anchor`）。换域必然有损（工程域里没有窗外的格），于是必须有闸；
/// 有闸就必然有"闸拒绝 ⇒ 开关无效" —— 用户对它的原话正是"完全没反应"。
///
/// 现在曲线**永远以源域存**，另存一个放置策略：
///   * 锁定（默认）：按**当前**几何投影 —— 曲线跟着素材走；
///   * 不锁定：按「建立这条权威时的那份几何」投影 —— 曲线留在工程位置。
///
/// 【为什么冻结几何而不是存工程帧数组】工程帧数组把"时间"烧进了下标：一旦片段被
/// 拉伸，同一份数组在源域上就再也找不到对应位置（这正是换域有损的根源）。冻结几何
/// 是纯数据（起点 / 长度 / 源窗 / 方向 / 倍率），任何时刻都能重新算出一一映射，
/// 因此这个开关是**可逆**的：拨回去逐帧回到原样，一个字节都不动。
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) enum Placement {
    /// 锁定（默认）：按当前几何投影 —— 曲线跟着素材走。
    #[default]
    FollowMaterial,
    /// 不锁定：按冻结几何投影 —— 曲线留在工程位置。
    PinProject {
        /// 冻结的是工程窗口到源窗口的若干段，而不是某一条 region 的单一几何。
        ///
        /// 一个 modification 可能被多条 region 共享（Cubase 分割是常态）。只冻结
        /// 第一条 region 的几何会让所有半段都按父段窗口读，右半段整段落到界外。
        /// 空 vec 只出现在两种场合：旧存档的 `anchor: "Project"`（那时还没有这个
        /// 字段）；用户刚把开关拨到"不锁定"而这一轮没有活着的 clip 几何可用。
        /// 两者都在第一次能拿到几何时补上 —— 那正是它当初被冻结的那一刻。
        frozen: Vec<RegionGeometry>,
    },
}

impl Placement {
    /// 把曲线钉在**当前**这份几何上（用户拨开关时用）。
    pub(crate) fn pinned(frozen: RegionGeometry) -> Self {
        Self::PinProject {
            frozen: vec![frozen],
        }
    }

    /// 把曲线钉在一组 region 几何上（同一 modification 被多条 region 共享）。
    pub(crate) fn pinned_many(frozen: Vec<RegionGeometry>) -> Self {
        Self::PinProject { frozen }
    }

    fn frozen(&self) -> &[RegionGeometry] {
        match self {
            Self::FollowMaterial => &[],
            Self::PinProject { frozen } => frozen,
        }
    }

    /// 投影用的映射：跟随素材取**当前**几何，钉住取**冻结**几何。
    ///
    /// frozen 为空时退回当前几何：没有冻结数据时"钉在现在"是唯一可解释的答案。
    fn map(&self, current: &RegionGeometry) -> Result<SourceTimeMap, String> {
        match self {
            Self::FollowMaterial => current.map(),
            Self::PinProject { frozen } => frozen.first().unwrap_or(current).map(),
        }
    }

    /// 某一工程时刻该用哪条映射。钉住模式下窗口可以有多段；找不到覆盖段返回
    /// `None`（这一帧没有编辑），而不是拿最近的一段猜。
    fn map_at_project(
        &self,
        current: &RegionGeometry,
        project_sec: f64,
    ) -> Result<Option<SourceTimeMap>, String> {
        match self {
            Self::FollowMaterial => Ok(Some(current.map()?)),
            Self::PinProject { frozen } if frozen.is_empty() => Ok(Some(current.map()?)),
            Self::PinProject { frozen } => {
                let found = frozen
                    .iter()
                    .filter(|geometry| {
                        // 冻结窗口的右端是开区间：片段被拉长后，超出原窗口的那一格
                        // 必须读出默认值，而不是把最后一个编辑复制到窗口外。
                        project_sec >= geometry.project_start - 1e-9
                            && project_sec
                                < geometry.project_start + geometry.project_duration - 1e-9
                    })
                    .min_by(|a, b| {
                        a.project_duration
                            .partial_cmp(&b.project_duration)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| {
                                a.project_start
                                    .partial_cmp(&b.project_start)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                    });
                found.map(|geometry| geometry.map()).transpose()
            }
        }
    }
}

/// 序列化形状。
///
/// 【为什么手写】旧存档里这个字段叫 `anchor` 且取值是 `"Source"` / `"Project"`。
/// 用派生实现会让那些工程读不进（整份参数丢失）；手写之后两种形状共存，
/// 新写入一律用新名字。`"Source"` 与 `"Project"` 各自映射到语义等价的新取值。
impl Serialize for Placement {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStructVariant;
        match self {
            Self::FollowMaterial => serializer.serialize_str("FollowMaterial"),
            Self::PinProject { frozen } => {
                let mut variant =
                    serializer.serialize_struct_variant("Placement", 1, "PinProject", 1)?;
                variant.serialize_field("frozen", frozen)?;
                variant.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Placement {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct PlacementVisitor;
        impl<'de> Visitor<'de> for PlacementVisitor {
            type Value = Placement;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a placement name or a pin with its frozen geometry")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Placement, E> {
                match value {
                    // 旧存档的两种取值：「Source」就是"跟随素材"，「Project」是"钉在
                    // 工程位置"（那时没有冻结几何字段，第一次能拿到几何时补上）。
                    "Source" | "FollowMaterial" => Ok(Placement::FollowMaterial),
                    "Project" | "PinProject" => Ok(Placement::PinProject { frozen: Vec::new() }),
                    other => Err(E::unknown_variant(
                        other,
                        &["FollowMaterial", "PinProject", "Source", "Project"],
                    )),
                }
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Placement, A::Error> {
                let Some(key) = map.next_key::<String>()? else {
                    return Ok(Placement::FollowMaterial);
                };
                if key != "PinProject" {
                    return Err(de::Error::unknown_variant(&key, &["PinProject"]));
                }
                // 兼容两种形状：旧存档是单个 object/null，新存档是数组。
                #[derive(Deserialize)]
                #[serde(untagged)]
                enum Frozen {
                    One(RegionGeometry),
                    Many(Vec<RegionGeometry>),
                }
                let frozen = match map.next_value::<Option<Frozen>>()? {
                    Some(Frozen::One(geometry)) => vec![geometry],
                    Some(Frozen::Many(geometries)) => geometries,
                    None => Vec::new(),
                };
                Ok(Placement::PinProject { frozen })
            }
        }

        deserializer.deserialize_any(PlacementVisitor)
    }
}

/// 参数权威键：**宿主给的稳定身份优先**。
///
/// 【为什么 item GUID 优先于 modification】REAPER 的 item GUID 跨移动 / 裁剪 / 分割
/// 稳定，且它能把"同一素材被放两次"区分成两份独立参数 —— 这是原生路径既有的、已被
/// 测试固定的语义（`ui_inventory_item_identity_survives_mute_region_recreation_...`）。
/// 若只用 modification，两条共享同一 modification 的 region 会互相覆盖曲线。
///
/// 【为什么 ARA-only 退回 modification】没有原生扩展时 `item` 恒为 `None`，此时
/// modification persistentID 是**唯一**跨会话稳定的身份 —— 而且 ARA 明确要求适配设置
/// 属于 audio modification state（`ARAInterface.h:950-952`），modification 持久、
/// playback region 不持久（`:903` / `:953-955`）。
///
/// 【为什么两者都与几何无关】这正是本键存在的理由：移动 / 裁剪 / 分割 / 拉伸都不改变
/// item GUID，也不改变 modification persistentID，因此权威查找在这四种编辑下都命中。
pub(crate) fn authority_key(item: Option<&str>, modification: &str) -> Option<String> {
    match item {
        Some(item) if !item.is_empty() => Some(format!("item:{item}")),
        _ if !modification.is_empty() => Some(format!("mod:{modification}")),
        _ => None,
    }
}

/// 一份参数权威：某个稳定宿主身份（item GUID 或 modification persistentID）的参数。
///
/// 【为什么它同时解决分割】分割之所以需要谱系，只是因为"新的稳定身份从哪来"没有答案。
/// 而这个答案 ARA 会给（`cloneAudioModification` 显式给出父 modification），即便宿主
/// 不用它，`end_editing` 的 region 结构 diff 也能给出父（含两个新 item GUID）——
/// 两条路径都把父的这份参数**整份复制**到两个新键上，此后两半各自独立演化。
///
/// 【为什么曲线以源域存储】两个半段的源窗是父段源窗的两个子区间，投影时各取所需；
/// 移动 / 裁剪 / 拉伸同理。切分或重采样反而会引入误差。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct AuthorityParameters {
    /// 用户显式编辑的稀疏源域记录。新写入只写这里；未出现的帧永远没有编辑。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    edits: BTreeMap<String, EditCurve>,
    /// 该权威所属的 ARA audio source；take 切换后 item GUID 可能不变，必须校验源。
    #[serde(default, skip_serializing_if = "String::is_empty")]
    source: String,
    /// 旧存档（第 8 代及以前）的整条源域曲线。只读兼容；新写入为空，encode 前清空。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    curves: BTreeMap<String, SourceCurve>,
    /// 逐轨模板（曲线已清空），与 [`RegionParameters::template`] 同形。
    template: TrackParamsState,
    /// 曲线在工程时间上的放置策略（见 [`Placement`]）。
    ///
    /// 【为什么字段名换了、别名留着】旧存档写的是 `anchor: "Source" | "Project"`。
    /// `alias` + 手写的 `Deserialize` 让两种形状读进同一份数据 —— 升级零损失，
    /// 而新写入一律用新名字。
    #[serde(default, alias = "anchor")]
    pub(crate) placement: Placement,

    /// 该权威最后一次被用户写入的序号；多个编辑碰撞时用于确定性优先级。
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    last_edit_seq: u64,

    /// 连续多少轮它的 modification 已不在文档图里（孤儿回收的宽限期计数）。
    ///
    /// 【为什么 `serde(skip)`】这是**运行时**状态，不是存档数据：它只在本次会话里
    /// 用来给"宿主销毁了 modification、但用户可能撤销回来"留一段宽限期。
    /// 写进存档只会让格式多一个没有意义的字段。
    #[serde(skip)]
    unseen_rounds: u32,

    /// 连续多少轮"想写回却被硬闸拒绝"。
    ///
    /// 【为什么需要它】`ara.atlas.line-write` 被 `log_limiter` 按 key 限流，稳定状态
    /// 下每 8 条只留 1 条 —— 于是"它一直在失败"在日志里看起来像"偶发失败"。
    /// 有了这个计数，卡住的循环会自己汇总成一行。
    ///
    /// 【与 `unseen_rounds` 同一条理由 `serde(skip)`】运行时状态，不是存档数据。
    #[serde(skip)]
    refused_rounds: u32,
}

/// 孤儿权威的宽限期（轮数）。见 [`ParameterAtlas::reclaim_orphans`]。
const ORPHAN_GRACE_ROUNDS: u32 = 64;

impl AuthorityParameters {
    /// 从一条已解析的 per-clip 记录建立权威条目（迁移与播种共用）。
    fn from_record(record: &RegionParameters, placement: Placement) -> Self {
        let edits = if record.edits.is_empty() {
            edits_from_source_curves(&record.curves, record.template.frame_period_ms)
        } else {
            record.edits.clone()
        };
        Self {
            edits,
            source: record.identity.source.clone(),
            curves: record.curves.clone(),
            template: record.template.clone(),
            placement,
            last_edit_seq: 0,
            unseen_rounds: 0,
            refused_rounds: 0,
        }
    }
    /// 是否持有用户编辑（曲线或旧存档曲线任一非空）。
    fn has_curves(&self) -> bool {
        !self.edits.is_empty() || !self.curves.is_empty()
    }
    fn has_pitch_edits(&self) -> bool {
        self.edits
            .get("pitch_edit")
            .is_some_and(|curve| !curve.is_empty())
            || self.curves.contains_key("pitch_edit")
    }
    /// 第 9 代存档写出前清掉旧整条曲线；新权威只留 edits。
    pub(crate) fn clear_legacy_curves(&mut self) {
        self.curves.clear();
    }
    /// 把第 8 代及以前的整条曲线迁移成稀疏编辑（幂等）。
    pub(crate) fn migrate_legacy_to_edits(&mut self) {
        if self.edits.values().all(|curve| curve.is_empty()) && !self.curves.is_empty() {
            self.edits = edits_from_source_curves(&self.curves, self.template.frame_period_ms);
            self.last_edit_seq = self.last_edit_seq.max(1);
            self.curves.clear();
        }
    }
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
    /// **每帧派生的 per-clip 投影缓存**，键是 clip id（`ara-region-{key}`）。
    ///
    /// 【为什么它不是权威】region key 会被 `allocate_region` 复用（`ara/model.rs`），
    /// 所以 clip id 不是身份。权威是 [`Self::authorities`]；本表每帧由它派生。
    /// 它仍然被序列化，但只为了**读旧存档**（那时权威表还是空的，需要迁移）；
    /// `EditState::encode` 在写出前清空它，因此新存档里它是空的。
    pub regions: BTreeMap<String, RegionParameters>,
    /// **权威**：稳定宿主身份（见 [`authority_key`]）→ 参数。
    ///
    /// 【为什么它取代了"几何指纹认领"】旧路径每帧用 `ara_region_signature` 去猜曲线归属，
    /// 而那个签名含**播放几何** —— 移动 / 裁剪 / 分割 / 拉伸四种编辑每一种都会让它变化，
    /// 于是认领必然失败、曲线被丢弃（用户看到的"参数被重置"）。本表按与几何无关的稳定
    /// 身份索引，四种编辑全部命中。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub authorities: BTreeMap<String, AuthorityParameters>,
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
    /// 【为什么保留】分割谱系现在由 modification 播种承载（见
    /// [`ModificationParameters`]），本表只剩读旧存档与原生 GUID 版回退两条用途。
    /// 新写入一律走 modification 播种。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    ara_split_parents: BTreeMap<String, String>,
    /// ARA-only 宿主的复制谱系：新片段签名 → 种子的完整参数记录。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    ara_copy_seeds: BTreeMap<String, RegionParameters>,
}

/// 一条根轨的**根视图投影**：参数 + 这一轮真的投影到了哪些帧。
///
/// 【为什么必须带上 `frames`】`project_roots` 只能投影"有权威且几何可解"的 clip。
/// 任何一条拿不到权威（日志里 `keys=0` 的那种）都会让它的帧段停在默认值上。
/// 只交一份参数数组时，调用方唯一的合并方式是**整值替换**整条根轨 —— 于是那一段
/// 默认值会把用户曲线抹平。带上 `frames`，合并就能只覆盖真正有信息的帧。
///
/// 【`complete` 与 `frames` 各回答什么】`frames` 回答"哪些帧有信息"；`complete`
/// 回答"这条根轨上**所有** clip 都有信息吗"。前者决定合并范围，后者只用于诊断
/// （"某条 clip 掉了"必须能被看见，而不是被一段默认值安静地盖过去）。
#[derive(Clone, Debug)]
pub(crate) struct RootProjection {
    pub params: TrackParamsState,
    /// 这一轮真的写入过的帧（升序、不重叠；不再使用整轨哨兵）。
    pub frames: Vec<std::ops::Range<usize>>,
    /// 每条曲线这一轮由显式编辑赢下的帧。只有这些帧允许覆盖其它 clip 的编辑。
    pub edited_frames: BTreeMap<String, Vec<std::ops::Range<usize>>>,
    /// 这一轮是否完整（所有 clip 都投影）。
    pub complete: bool,
    /// baseline 是否与当前几何同代（有可用的 pitch_orig 内容）。
    pub baseline_ready: bool,
    /// 用户明确清空、且本轮 complete 的曲线；merge 时才允许删除。
    pub cleared_curves: std::collections::BTreeSet<String>,
}

/// 一条 clip 的投影结果：参数 + 这一轮由显式编辑赢下的帧。
struct ProjectedClip {
    params: TrackParamsState,
    edited_frames: BTreeMap<String, Vec<std::ops::Range<usize>>>,
    baseline_ready: bool,
}

fn mask_ranges(mask: &[bool]) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = None;
    for (frame, edited) in mask.iter().copied().enumerate() {
        match (start, edited) {
            (None, true) => start = Some(frame),
            (Some(begin), false) => {
                push_frame_range(&mut ranges, begin..frame);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(begin) = start {
        push_frame_range(&mut ranges, begin..mask.len());
    }
    ranges
}

impl RootProjection {
    /// 覆盖整条根轨（空白编辑与"没有曲线的根"用它）。
    ///
    /// 【为什么带 `#[cfg(test)]`】生产路径总是从真实曲线构造投影；这个"整条覆盖"
    /// 的便捷构造只被测试用到。
    #[cfg(test)]
    pub(crate) fn whole(params: TrackParamsState) -> Self {
        Self {
            params,
            #[allow(clippy::single_range_in_vec_init)]
            frames: [0..usize::MAX].to_vec(),
            edited_frames: BTreeMap::new(),
            complete: true,
            baseline_ready: true,
            cleared_curves: std::collections::BTreeSet::new(),
        }
    }
}

/// 把一份根视图投影并进 `target`：**只覆盖 `frames` 里的帧**。
///
/// 【为什么不 `extend`】`BTreeMap::extend` 对已存在的键是整值替换。而投影可能只
/// 覆盖了半条轨道（有一条 clip 没查到权威），替换等于用"一段默认值 + 半条真曲线"
/// 盖掉用户原来的整条曲线 —— 用户看到的就是"移动一下 item，参数回退了"。
pub(crate) fn merge_root_projection(
    target: &mut BTreeMap<String, TrackParamsState>,
    roots: BTreeMap<String, RootProjection>,
) {
    for (root, projection) in roots {
        let Some(existing) = target.get_mut(&root) else {
            // 新根：只有完整投影才装。半份投影插进去会让缺失的帧永远以默认值存在。
            if projection.complete {
                target.insert(root, projection.params);
            } else {
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Warn,
                    "ara.atlas.root-projection-incomplete",
                    format_args!(
                        "[ara] root {root}: no existing root view and this round projected only \
                         {} clip(s); the new root is not materialized from a partial default",
                        projection.frames.len()
                    ),
                );
            }
            continue;
        };
        // 帧周期不一致时无法按帧合并（数组的时间轴不同）。保留旧值。
        if existing.frame_period_ms != projection.params.frame_period_ms {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.atlas.root-frame-period-changed",
                format_args!(
                    "[ara] root {root} changes its frame period from {} ms to {} ms; the root \
                     view is kept as it was until the curves are re-captured",
                    existing.frame_period_ms, projection.params.frame_period_ms
                ),
            );
            continue;
        }
        if !projection.complete {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.atlas.root-projection-incomplete",
                format_args!(
                    "[ara] root {root}: only {} clip range(s) were projected this round;                      frames that were not projected keep their previous curves",
                    projection.frames.len()
                ),
            );
        }
        // 用户明确清空整条曲线：只有完整投影才允许删除。否则保留旧曲线。
        if projection.complete {
            for key in &projection.cleared_curves {
                if parameter_curves(existing)
                    .into_iter()
                    .any(|(name, values)| &name == key && !values.is_empty())
                {
                    set_curve(existing, key, Vec::new());
                }
            }
        }
        for (key, values) in parameter_curves(&projection.params) {
            // 派生原始值只由当前分析/根 baseline 提供，永远不从投影写回。
            if matches!(key.as_str(), "pitch_orig" | "dyn_orig" | "tension_orig") {
                continue;
            }
            if values.is_empty() {
                continue;
            }
            let mut merged = parameter_curves(existing)
                .into_iter()
                .find(|(name, _)| *name == key)
                .map(|(_, current)| current.to_vec())
                .unwrap_or_default();
            // 显式编辑任何情况下都覆盖；未编辑帧只在 baseline 同代时覆盖。
            let mut ranges = projection
                .edited_frames
                .get(&key)
                .cloned()
                .unwrap_or_default();
            if projection.baseline_ready {
                ranges.extend(projection.frames.iter().cloned());
                ranges.sort_by_key(|range| (range.start, range.end));
                let mut merged_ranges = Vec::new();
                for range in ranges {
                    push_frame_range(&mut merged_ranges, range);
                }
                ranges = merged_ranges;
            }
            for range in ranges {
                let from = range.start;
                let to = range.end.min(values.len());
                if from >= to {
                    continue;
                }
                if merged.len() < to {
                    merged.resize(to, pad(&key));
                }
                merged[from..to].copy_from_slice(&values[from..to]);
            }
            if !merged.is_empty() {
                set_curve(existing, &key, merged);
            }
        }
        // 完整投影是这条根轨这一轮的**完整显示**：只保留 frames 的旧值会让 clip
        // 移走后旧位置继续残留编辑（锁定时参数线“跟着走”的可见反例）。这里用
        // projection 的完整曲线替换可编辑层，但保留派生原始值与会话记账。
        if projection.complete {
            let mut keys = std::collections::BTreeSet::new();
            for (key, _) in parameter_curves(existing) {
                if !matches!(key.as_str(), "pitch_orig" | "dyn_orig" | "tension_orig") {
                    keys.insert(key);
                }
            }
            for (key, _) in parameter_curves(&projection.params) {
                if !matches!(key.as_str(), "pitch_orig" | "dyn_orig" | "tension_orig") {
                    keys.insert(key);
                }
            }
            keys.extend(projection.cleared_curves.iter().cloned());
            for key in keys {
                if projection.cleared_curves.contains(&key) {
                    set_curve(existing, &key, Vec::new());
                    continue;
                }
                let Some(values) = parameter_curves(&projection.params)
                    .into_iter()
                    .find(|(name, _)| *name == key)
                    .map(|(_, values)| values.to_vec())
                else {
                    continue;
                };
                let old_len = parameter_curves(existing)
                    .into_iter()
                    .find(|(name, _)| *name == key)
                    .map(|(_, values)| values.len())
                    .unwrap_or(0);
                let mut values = values;
                if values.len() < old_len {
                    values.resize(old_len, pad(&key));
                }
                set_curve(existing, &key, values);
            }
        }
        if projection.complete {
            existing.pitch_edit_user_modified = projection.params.pitch_edit_user_modified;
        } else {
            existing.pitch_edit_user_modified |= projection.params.pitch_edit_user_modified;
        }
    }
}

/// 把一个区间并进**升序、不重叠**的帧列表（相邻或相交就合并）。
fn push_frame_range(frames: &mut Vec<std::ops::Range<usize>>, range: std::ops::Range<usize>) {
    if range.start >= range.end {
        return;
    }
    match frames.last_mut() {
        Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
        _ => frames.push(range),
    }
}
/// 一条 clip 的参数权威：新路径查权威表，旧存档退回 per-clip 记录。
enum Authority<'a> {
    Current(&'a AuthorityParameters),
    Legacy(&'a RegionParameters),
}

impl Authority<'_> {
    fn template(&self) -> &TrackParamsState {
        match self {
            Self::Current(params) => &params.template,
            Self::Legacy(record) => &record.template,
        }
    }
    fn edits(&self) -> Option<&BTreeMap<String, EditCurve>> {
        match self {
            Self::Current(params) if !params.edits.is_empty() => Some(&params.edits),
            _ => None,
        }
    }

    /// 旧存档的整条曲线；只用于读兼容与测试种子。
    fn curves(&self) -> &BTreeMap<String, SourceCurve> {
        match self {
            Self::Current(params) => &params.curves,
            Self::Legacy(record) => &record.curves,
        }
    }

    /// 这份权威的曲线**按哪份几何投影**。
    ///
    /// 【为什么读路径必须用它，而不是"当前设置"】`capture_inner` 会把落地的放置策略
    /// 写进权威（与落地的数据同源），但读路径此前只信"当前设置"。两者可以相反：
    /// 用户拨动「锁定参数线」之后，读路径会按另一种映射解释同一个数组 —— 整条曲线
    /// 偏移一个源窗起点，而数据一格都没丢。用户看到的就是"移动片段后参数线留在原地"。
    ///
    /// `Legacy`（旧存档的 per-clip 记录）没有这个字段，`serde(default)` 也补不出
    /// "当初是怎么写的"，只能按当前设置解释 —— 而那条路径只是迁移窗口，
    /// `EditState::encode` 已在写出前清空它。
    fn placement<'b>(&'b self, fallback: &'b Placement) -> &'b Placement {
        match self {
            Self::Current(params) => &params.placement,
            Self::Legacy(_) => fallback,
        }
    }

    /// 是否持有一条音高编辑曲线。
    ///
    /// 【为什么这是"内核不得清空"的判据】内核 `ensure_params_for_root` 在
    /// `build_root_pitch_key` 变化时会清 `pitch_edit`（除非 `pitch_edit_user_modified`），
    /// 而那个 key 把每条 clip 的 id / start / length / 倍率都喂进哈希 —— **任何一次分割
    /// 都会改变它**。这是完全独立于图集的第二条"参数被重置"路径。权威里有一条音高曲线，
    /// 就说明这份数据是用户意图，不是可丢弃的陈旧分析结果。
    fn has_pitch_curve(&self) -> bool {
        match self {
            Self::Current(params) => params.has_pitch_edits(),
            Self::Legacy(record) => record.curves.contains_key("pitch_edit"),
        }
    }
}

/// 一次写入是「用户表达了意图」还是「系统重新解释数据」。
///
/// 【为什么必须是显式类型】两者的安全边界**相反**：
///   * 用户意图：允许增加、减少、清空任何帧（包括把整段擦回默认值）；
///   * 派生写入：只允许「同值搬运」，任何减少都必须被解释成"几何变化"，不是"用户删除"。
///
/// 今天这两件事共用一条 `capture_inner`，于是守卫只能取两者的**交集** —— 而交集恰好
/// 把"用户删除"也禁掉了。用户用右键拖拽把曲线擦回初始化，在数据层就是"减少非默认帧"，
/// 于是整条权威被拒绝写回：界面先回显、下一次投影再抹掉、音频一个采样都没变。
pub(crate) enum WriteIntent<'a> {
    /// 用户编辑：带「编辑器读到的那份视图」作为差分基线。
    User {
        baseline: &'a BTreeMap<String, TrackParamsState>,
    },
    /// 派生写入：由几何/域变化产生，没有基线可依。
    Derived,
}

/// 一次被拒绝的写回：哪条权威、哪条曲线、几帧没有证据。
///
/// 【为什么要带上帧号与两侧取值】"你的这次编辑没有被采纳"必须能回答"哪几帧"，
/// 否则用户唯一能做的事是"再画一次试试" —— 而那正是今天的状态。
#[derive(Clone, Debug)]
pub(crate) struct RefusedWrite {
    /// 权威键（`item:{GUID}` 或 `mod:{persistentID}`）。
    pub(crate) key: String,
    /// 根轨 id。
    pub(crate) root: String,
    /// 曲线名（`pitch_edit` / `extra:xxx` …）。
    pub(crate) curve: String,
    /// 没有证据就被删掉的帧数。
    pub(crate) unexplained: usize,
    /// 其中第一帧的源秒与两侧取值（诊断用）。
    pub(crate) first_frame: Option<(f64, f32, f32)>,
}

/// 一次捕获的**读数**（写回被拒 / 无基线曲线 / 被保护而未写的帧）。
///
/// 【为什么要返回值而不是打日志】`log_limiter` 按 key 限流，稳定状态下每 8 条只留 1 条
/// —— 于是"它一直在被拒"在日志里看起来像"偶发被拒"。读数进状态快照与诊断包，
/// 用户与支持都能看见同一件事。
#[derive(Clone, Debug, Default)]
pub(crate) struct CaptureReport {
    /// 被拒绝写回的曲线（有界，见 `MAX_REFUSED_WRITES`）。
    pub(crate) refused: Vec<RefusedWrite>,
    /// 因为"根的基线数组缺席"而整条跳过的 clip（R4）。
    pub(crate) skipped_clips: usize,
    /// 有曲线、却没有基线数组的曲线条数（R3）：这些曲线只允许**增加**帧。
    pub(crate) no_baseline_curves: usize,
    /// 因为落在另一条选中片段的保护区间里而没有写回的帧数（R5）。
    pub(crate) protected_frames: usize,
    /// 本轮被用户删除（有证据）的帧数。
    pub(crate) removed_frames: usize,
}

impl CaptureReport {
    /// 一次捕获最多留几条被拒记录。
    ///
    /// 【为什么有界】它进状态快照与诊断包；一个持续失败的工程会让它无界增长。
    const MAX_REFUSED_WRITES: usize = 8;

    fn refuse(&mut self, write: RefusedWrite) {
        if self.refused.len() < Self::MAX_REFUSED_WRITES {
            self.refused.push(write);
        }
    }

    /// 一次写入是否被拒绝过。
    pub(crate) fn has_refusals(&self) -> bool {
        !self.refused.is_empty()
    }

    /// 读数的一行摘要（状态栏与日志共用）。
    pub(crate) fn summary(&self) -> String {
        format!(
            "refused={} skipped={} no-baseline={} protected={} removed={}",
            self.refused.len(),
            self.skipped_clips,
            self.no_baseline_curves,
            self.protected_frames,
            self.removed_frames
        )
    }

    /// 诊断包 / 状态快照的 JSON 形状。
    pub(crate) fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "refused": self.refused.iter().map(|write| serde_json::json!({
                "key": write.key,
                "root": write.root,
                "curve": write.curve,
                "unexplained": write.unexplained,
                "firstFrameSec": write.first_frame.map(|frame| frame.0),
                "kept": write.first_frame.map(|frame| frame.1),
                "incoming": write.first_frame.map(|frame| frame.2),
            })).collect::<Vec<_>>(),
            "skippedClips": self.skipped_clips,
            "noBaselineCurves": self.no_baseline_curves,
            "protectedFrames": self.protected_frames,
            "removedFrames": self.removed_frames,
        })
    }
}

/// 一条**空**的基线数组。
///
/// 【为什么需要它】"基线里没有这条曲线"要表达成"没有可用的旧视图"，而
/// `sample_project_frames` 对空切片返回 `pad` —— 也就是"旧视图里这一帧是默认值"。
/// 这与 `None` 的区别正是本轮要的：`None` 表示"派生写入，没有基线"，
/// 空数组表示"有基线，但这一条没有可比的东西"。两者都只允许**增加**帧。
const EMPTY_VIEW: [f32; 0] = [];

/// 判定一次写回里"被删掉的帧"有没有证据。
///
/// 【判据】删掉一帧（旧非默认 → 新默认）只有在**两条证据同时成立**时才允许：
///   1. 编辑器这次交来的数组在这一帧上确实是默认值；
///   2. 编辑器读到的那份基线在这一帧上确实不是默认值。
///
/// 这恰好就是"用户把它改成了默认值"的定义，也正是 [`SourceCapture`] 里
/// `incoming != prior` 这一条判据的另一面 —— 只是这里要求**两侧都**成立，
/// 而不是"有一侧变了"。
///
/// 不变量：
///   * **用户删除被允许**（每一帧都有"基线非默认、当前是默认"这条证据）；
///   * **派生写入的误伤仍被拦住**（没有基线时任何减少都不成立 ⇒ 拒绝）；
///   * **半份视图造成的假删除仍被拦住**（投影范围变窄时，被覆盖的帧在 client 里是
///     默认值，在基线里**也是**默认值 ⇒ 不满足第二条 ⇒ 拒绝）。
fn unexplained_removals(
    key: &str,
    old: Option<&SourceCurve>,
    new: Option<&SourceCurve>,
    baseline: Option<&[f32]>,
    client: &[f32],
    current: &SourceTimeMap,
    frame_ms: f64,
) -> usize {
    let Some(old) = old else {
        return 0;
    };
    // 【旧存档形状不参与证据判定】`basis`/`Project` 域的格号不是源帧，把它们当源帧
    // 折算等于两种单位相减。这类曲线只在读旧存档时出现，迁移会把它换成源域
    // （见 `curve_into_source_domain`），因此这里退回"计数不变"的旧判据。
    if old.basis.is_some() || old.domain != CurveDomain::Source {
        let before = non_default_values(key, old);
        let after = new.map(|curve| non_default_values(key, curve)).unwrap_or(0);
        return before.saturating_sub(after);
    }
    let pad_value = pad(key);
    let mut unexplained = 0_usize;
    for (index, value) in old.values.iter().enumerate() {
        if !(value.is_finite() && (*value - pad_value).abs() > 1e-6) {
            continue;
        }
        let incoming = new
            .and_then(|curve| curve.values.get(index))
            .copied()
            .unwrap_or(pad_value);
        // 不是"被删掉"（这一帧仍然非默认）⇒ 不参与证据判定。
        if (incoming - pad_value).abs() > 1e-6 {
            continue;
        }
        // 这一帧被删掉了。现在要两条证据。
        let source_sec = (old.first_frame + index) as f64 * old.frame_ms / 1000.;
        let explained = current
            .source_to_project(source_sec)
            .is_some_and(|project_sec| {
                let frame = project_sec * 1000. / frame_ms;
                let prior = baseline.map(|view| sample_project_frames(view, frame, key));
                let now = sample_project_frames(client, frame, key);
                // 证据①：当前是默认值。证据②：基线不是默认值。
                (now - pad_value).abs() <= 1e-6
                    && prior.is_some_and(|prior| (prior - pad_value).abs() > 1e-6)
            });
        if !explained {
            unexplained += 1;
        }
    }
    unexplained
}

/// 删掉一帧时的诊断取样：第一帧的源秒与两侧取值。
fn first_unexplained_frame(
    key: &str,
    old: Option<&SourceCurve>,
    new: Option<&SourceCurve>,
    current: &SourceTimeMap,
    frame_ms: f64,
    client: &[f32],
) -> Option<(f64, f32, f32)> {
    let old = old?;
    if old.basis.is_some() || old.domain != CurveDomain::Source {
        return None;
    }
    let pad_value = pad(key);
    for (index, value) in old.values.iter().enumerate() {
        if !(value.is_finite() && (*value - pad_value).abs() > 1e-6) {
            continue;
        }
        let incoming = new
            .and_then(|curve| curve.values.get(index))
            .copied()
            .unwrap_or(pad_value);
        if (incoming - pad_value).abs() > 1e-6 {
            continue;
        }
        let source_sec = (old.first_frame + index) as f64 * old.frame_ms / 1000.;
        let now = current.source_to_project(source_sec).map_or_else(
            || pad_value,
            |project_sec| sample_project_frames(client, project_sec * 1000. / frame_ms, key),
        );
        return Some((source_sec, *value, now));
    }
    None
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

    /// 把一个权威条目的参数**整份复制**给另一个（宿主 clone / 结构 diff 播种）。
    ///
    /// 【为什么这是分割的正确形状】分割之所以需要谱系，只是因为"新身份从哪来"没有答案。
    /// ARA 的 `cloneAudioModification` 直接给出父；宿主不用它时，`end_editing` 的 region
    /// 结构 diff 也能给出父（且给出两个新 item GUID）。两条路径都落到这里。
    ///
    /// 【为什么整份复制而不是切子段】曲线以**源域**存储，两个半段的源窗是父段源窗的两个
    /// 子区间，投影时各取所需。切分反而引入一次多余的重采样。
    ///
    /// 返回是否真的播了种。父没有参数、子已经存在（含宿主直接复用父身份的别名情形）、
    /// 或键非法时都返回 `false` 且不改动任何状态 —— 保持"从没编辑过"的语义。
    pub(crate) fn seed_authority(&mut self, parent: &str, child: &str) -> bool {
        if parent.is_empty() || child.is_empty() || parent == child {
            return false;
        }
        if self.authorities.contains_key(child) || self.authorities.len() >= 16384 {
            return false;
        }
        let Some(params) = self.authorities.get(parent).cloned() else {
            return false;
        };
        // 父没有曲线时不播：一条空权威与"从没编辑过"在投影上等价，播过去只会撑大存档。
        if !params.has_curves() {
            return false;
        }
        self.authorities.insert(child.to_owned(), params);
        true
    }

    /// 写入一个权威条目的参数（`capture` 的权威侧写入）。
    pub(crate) fn set_authority(&mut self, key: &str, params: AuthorityParameters) {
        self.authorities.insert(key.to_owned(), params);
    }

    /// 测试用：给一个权威键种一条源域音高曲线。
    ///
    /// 【为什么需要它】模型层的分割回归必须让"父段真的有参数"，而构造 `SourceCurve`
    /// 需要访问本模块的私有字段。把这件事收在一个测试助手里，而不是放宽生产可见性。
    #[cfg(test)]
    pub(crate) fn seed_test_pitch_curve(&mut self, key: &str, frame_ms: f64, values: Vec<f32>) {
        self.seed_test_pitch_curve_placed(key, frame_ms, values, Placement::FollowMaterial);
    }

    /// 测试用：种一条带指定**放置策略**的源域音高曲线。
    ///
    /// 【为什么只有一个域】曲线现在永远按源秒存（见 `Placement`）。`first_frame` 因此
    /// 恒为 0：种子覆盖源 `[0, span)`。旧存档那种"按工程帧索引"的形状另有
    /// `seed_legacy_project_curve`，只服务迁移用例。
    #[cfg(test)]
    pub(crate) fn seed_test_pitch_curve_placed(
        &mut self,
        key: &str,
        frame_ms: f64,
        values: Vec<f32>,
        placement: Placement,
    ) {
        let mut edits = BTreeMap::new();
        if let Some(curve) = edit_curve_from_values(&values, 0, frame_ms, "pitch_edit") {
            edits.insert("pitch_edit".to_owned(), curve);
        }
        self.authorities.insert(
            key.to_owned(),
            AuthorityParameters {
                edits,
                source: String::new(),
                curves: BTreeMap::new(),
                template: TrackParamsState {
                    frame_period_ms: frame_ms,
                    ..Default::default()
                },
                placement,
                last_edit_seq: 1,
                unseen_rounds: 0,
                refused_rounds: 0,
            },
        );
    }

    /// 测试用：种一条**旧存档形状**（工程帧索引）的音高曲线。
    ///
    /// 【为什么要单独一个】第七轮之前的存档把 `values[i]` 写成"工程第 i 格"。
    /// 那种形状只能出现在**读旧存档**这一条路上，迁移之后就不再有；用例需要它来
    /// 证明"升级这一刻曲线一格都不动"。
    #[cfg(test)]
    pub(crate) fn seed_legacy_project_curve(
        &mut self,
        key: &str,
        frame_ms: f64,
        project_start: f64,
        values: Vec<f32>,
    ) {
        let first_frame = (project_start * 1000. / frame_ms).floor() as usize;
        let mut curves = BTreeMap::new();
        curves.insert(
            "pitch_edit".to_owned(),
            SourceCurve {
                basis: None,
                domain: CurveDomain::Project,
                first_frame,
                frame_ms,
                values: Arc::new(values),
                reservation: None,
            },
        );
        self.authorities.insert(
            key.to_owned(),
            AuthorityParameters {
                edits: BTreeMap::new(),
                source: String::new(),
                curves,
                template: TrackParamsState {
                    frame_period_ms: frame_ms,
                    ..Default::default()
                },
                // 旧 `anchor: "Project"` 读进来就是"钉住但还没有冻结几何"。
                placement: Placement::PinProject { frozen: Vec::new() },
                last_edit_seq: 0,
                unseen_rounds: 0,
                refused_rounds: 0,
            },
        );
    }

    /// 有界回收**孤儿权威**：它的 modification 已不在文档图里。
    ///
    /// 【为什么需要它】宿主销毁 modification 时权威**不能**立刻删 —— 撤销会把同一个
    /// persistentID 带回来（ARA 允许宿主用 `restoreObjectsFromArchive` 重放）。
    /// 但也不能永远留着：反复"分割 → 撤销 → 再分割"会持续产生新 ID，孤儿权威只增不减，
    /// 最终撞上 `validate` 的 64MiB 上限，于是**整个存档写不出去**（用户的编辑全丢）。
    ///
    /// 【为什么入参是"整份文档的 modification 集合"而不是"本 renderer 的分配范围"】
    /// 这是本函数唯一的安全边界。`follow_geometry` 是**逐 renderer** 跑的，而它只看得到
    /// 自己那一份分配；按那个范围回收会把**别的轨道上没分配给本实例**的权威整批删掉。
    /// 文档级集合没有这个问题。
    ///
    /// 【为什么有宽限期】销毁与撤销之间隔着若干轮 `end_editing`；`ORPHAN_GRACE_ROUNDS`
    /// 给这段窗口留足余量，也顺带容忍宿主"先删后建同一个 ID"的时序。
    ///
    /// 返回回收掉的条数。
    pub(crate) fn reclaim_orphans(&mut self, live: &std::collections::BTreeSet<String>) -> usize {
        let mut reclaimed = 0_usize;
        self.authorities.retain(|key, params| {
            let orphan = match key.strip_prefix("mod:") {
                Some(id) => !live.contains(id),
                // `item:` 级权威无法从键本身看出它引用哪个 modification ⇒ 不回收。
                None => false,
            };
            if !orphan {
                params.unseen_rounds = 0;
                return true;
            }
            params.unseen_rounds = params.unseen_rounds.saturating_add(1);
            if params.unseen_rounds < ORPHAN_GRACE_ROUNDS {
                return true;
            }
            reclaimed += 1;
            false
        });
        reclaimed
    }

    /// 按 ARA 的 `RestoreFilter` 把存档里的 modification ID 重映射到当前 ID。
    ///
    /// 【为什么必须做】persistentID 只保证**文档内**唯一，跨文档导入时宿主要求把存档
    /// ID 映射到当前图里的 ID（`ARAInterface.h:2291-2294` 与 `ARARestoreObjectsFilter`
    /// 的 `audioModificationArchiveIDs` / `audioModificationCurrentIDs`）。直接按存档 ID
    /// 合入会把参数挂到一个不存在的键上 —— 静默丢失。
    pub(crate) fn remap_authorities(&mut self, remap: &[(String, String)]) {
        if remap.is_empty() {
            return;
        }
        let renamed = remap
            .iter()
            .filter(|(archive, current)| archive != current && !archive.is_empty())
            .filter_map(|(archive, current)| {
                authority_key(None, archive).map(|key| (key, authority_key(None, current)))
            })
            .filter_map(|(from, to)| to.map(|to| (from, to)))
            .collect::<Vec<_>>();
        for (from, to) in renamed {
            if let Some(params) = self.authorities.remove(&from) {
                self.authorities.insert(to, params);
            }
        }
    }

    /// 只保留给定 modification ID 的权威条目（部分存档的写出侧）。
    ///
    /// `None` 表示"全部选中"。`item:` 级条目按它引用的 modification 无法从键本身看出，
    /// 因此**整份保留** —— 它们是原生宿主上的逐 item 权威，丢掉会让那些工程丢参数。
    pub(crate) fn retain_authorities_for(
        &mut self,
        modifications: Option<&std::collections::BTreeSet<String>>,
    ) {
        let Some(modifications) = modifications else {
            return;
        };
        self.authorities
            .retain(|key, _| match key.strip_prefix("mod:") {
                Some(id) => modifications.contains(id),
                None => true,
            });
    }

    /// 这条 clip 的稳定身份是否已有权威参数（测试用）。    /// 这条 clip 的稳定身份是否已有权威参数（测试用）。
    #[cfg(test)]
    pub(crate) fn has_authority_parameters(&self, item: Option<&str>, modification: &str) -> bool {
        authority_key(item, modification)
            .and_then(|key| self.authorities.get(&key))
            .is_some_and(AuthorityParameters::has_curves)
    }

    /// 剪贴板反序列化的seed必须复验并登记内存额度，不能让外部数据绕过source curve预算。
    pub(crate) fn reserve_copy_seed(seed: RegionParameters) -> Result<RegionParameters, String> {
        let mut atlas = Self::default();
        atlas.regions.insert("clipboard-seed".into(), seed);
        let mut checked = atlas.reserve_restored()?;
        Ok(checked.regions.remove("clipboard-seed").unwrap())
    }
    /// 只登记"右段 ← 父段"的谱系边，**不需要**一份 per-clip 种子。
    ///
    /// 【为什么独立于 `register_split`】后者还要求一份 per-clip 记录（旧存档迁移窗口
    /// 的产物）才能建立 `split-basis`。而权威表已经把曲线播给子段时，谱系边仍然值得
    /// 记下来 —— 它是诊断（"这次分割到底有没有认出父段"）与 `find()` 旧路径的唯一读数。
    pub(crate) fn note_split_parent(&mut self, parent: &str, right: &str) {
        if parent.is_empty() || right.is_empty() || parent == right {
            return;
        }
        if self.split_parents.len() >= 16384 {
            return;
        }
        self.split_parents
            .insert(right.to_owned(), parent.to_owned());
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
            // 权威侧：复制回执直接给出新 item GUID ⇒ 播种 item 级权威。
            if let Some(seed) = self.copy_seeds.get(item) {
                let mut params = AuthorityParameters::from_record(seed, Placement::FollowMaterial);
                // 复制回执已给出新 item GUID；item 身份比旧 source 更强，不能因为
                // 宿主给复制体换了 source identity 就把 seed 丢掉。
                params.source = String::new();
                if let Some(key) = authority_key(Some(item), "") {
                    self.set_authority(&key, params);
                }
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
            // 权威侧：分割回执给出右半段的新 item GUID（左半段沿用原 item 的 GUID，
            // 因此它自己的条目无需播种）⇒ 把父段参数整份播给右半段。
            let mut params = AuthorityParameters::from_record(&seed, Placement::FollowMaterial);
            // 分割出的右半段可能拿到新的 source identity；item GUID 是宿主明确
            // 给出的新身份，旧 source 不应阻止父曲线播种。
            params.source = String::new();
            if let Some(key) = authority_key(Some(right), "") {
                self.set_authority(&key, params);
            }
            seed.live = false;
            self.regions.insert(format!("split-basis:{right}"), seed);
        }
        self.note_split_parent(parent, right);
        Ok(())
    }
    /// 读取第 8 代及以前存档后，把整条 curves 一次性折算成稀疏 edits。
    pub(crate) fn migrate_legacy_authority_edits(&mut self) {
        for authority in self.authorities.values_mut() {
            authority.migrate_legacy_to_edits();
        }
    }

    pub fn is_empty(&self) -> bool {
        self.authorities.is_empty()
            && self.regions.is_empty()
            && self.copy_seeds.is_empty()
            && self.gaps.is_empty()
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
            .authorities
            .values()
            .flat_map(|params| params.curves.values())
            .chain(
                self.regions
                    .values()
                    .chain(self.copy_seeds.values())
                    .flat_map(|region| region.curves.values()),
            )
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
        // 权威表按 modification persistentID 索引：键集相同即"没有任何 modification
        // 身份发生变化"，无需再看几何（曲线是源域的，几何变化由投影吸收）。
        if self.authorities.keys().eq(other.authorities.keys()) {
            return true;
        }
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
        for params in self.authorities.values_mut() {
            for curve in params.curves.values_mut() {
                if curve.reservation.is_none() {
                    curve.reservation = Some(Arc::new(
                        crate::render::budget::global_budget()
                            .reserve(curve.values.len() * 4)
                            .ok_or("source parameter memory budget exceeded")?,
                    ));
                }
            }
        }
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
    ///
    /// 【为什么从"认领"改成"派生"】此前每帧用几何指纹去 `find` 认领曲线，而几何指纹
    /// 在移动 / 裁剪 / 分割 / 拉伸四种编辑下**每一种都会变化** ⇒ 认领必然失败 ⇒ 曲线
    /// 被丢弃（用户看到的"参数被重置"）。现在权威是 modification persistentID，
    /// 与几何无关：本函数只做一次查表 + 播种迁移，然后重新派生投影缓存。
    pub fn follow_geometry(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
    ) -> Result<Self, String> {
        let mut followed = self.clone();
        // 【为什么这里只剩"旧存档迁移"】开关不再靠烘焙生效（见 `Placement`）：
        // 它是纯元数据操作，由 `DocumentSession::sync_placement_with_settings` 在
        // 投影前一次性落地。这里只做一次性的**域迁移**：把旧存档里按工程帧索引的
        // 曲线折算成源域，此后永不再有"域"这个概念。
        followed.migrate_legacy_curves(timeline, identities);
        // 旧存档只有 per-clip 记录：按旧规则认领一次并折算进权威（一次性）。
        followed.migrate_legacy_records(timeline, identities);
        // 本帧投影完全由权威派生。clip id 的不稳定在这里不再有影响。
        followed.regions = followed.derive_regions(timeline, identities)?;
        // 谱系表的有界回收：种子一旦被活记录认领就不再需要。
        //
        // 【为什么现在只需这一条】`regions` 已经是**每帧重建**的派生缓存（上面那行），
        // 因此 `split-basis:*` / `ara-split-basis:*` 副本与历史死记录都随旧表一起消失 ——
        // 此前那段"并集覆盖回收"（用来清理死祖先）不再需要。留下的只是两张按宿主
        // 回执索引的谱系表，它们仍会被 `find` 的迁移路径读取，所以按活记录裁剪。
        let bound = followed
            .regions
            .values()
            .filter_map(|record| record.identity.item.clone())
            .collect::<std::collections::BTreeSet<_>>();
        followed
            .split_parents
            .retain(|right, _| !bound.contains(right));
        followed.copy_seeds.retain(|item, _| !bound.contains(item));
        let live_signatures = followed
            .regions
            .values()
            .map(|record| record.identity.signature.clone())
            .filter(|signature| !signature.is_empty())
            .collect::<std::collections::BTreeSet<_>>();
        followed
            .ara_split_parents
            .retain(|right, _| live_signatures.contains(right));
        followed
            .ara_copy_seeds
            .retain(|signature, _| live_signatures.contains(signature));
        followed.validate()?;
        Ok(followed)
    }

    /// 把旧存档里**按工程帧索引**的曲线折算成源域（一次性、幂等）。
    ///
    /// 【为什么需要它】第七轮之前的存档把 `values[i]` 写成"工程第 i 格"。那种形状
    /// 无法参与"按帧证据"的写入判据（格号不是源帧，两种单位相减没有意义），也无法
    /// 跟随片段移动。本函数在第一次能拿到该权威的活几何时折算一次，此后它和所有
    /// 新曲线一样是源域的。
    ///
    /// 【为什么用**当前**几何折算】旧形状没有记录"当初是哪份几何"。而当前几何就是
    /// 用户此刻看到的那一份 —— 用它折算能保证"升级这一刻，曲线一格都不动"。
    /// 之后它跟随素材（`FollowMaterial`）或按本次几何钉住（`PinProject`），
    /// 两种都与旧语义里"锁定/不锁定"的可见行为一致。
    ///
    /// 【为什么按 clip 而不是按权威】几何是**片段**的属性，而一条权威可能被多条
    /// region 共享。逐 clip 走，每条都能用自己的几何折算；同一条权威折算两次是
    /// 幂等的（第二次没有旧形状曲线，直接返回）。
    fn migrate_legacy_curves(
        &mut self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
    ) -> usize {
        let mut pending: Vec<(String, AuthorityParameters)> = Vec::new();
        let mut migrated = 0_usize;
        for clip in &timeline.clips {
            let Some(identity) = identities.get(&clip.id) else {
                continue;
            };
            let Some(key) = authority_key(identity.item.as_deref(), &identity.modification) else {
                continue;
            };
            if pending.iter().any(|(done, _)| *done == key) {
                continue;
            }
            let Some(params) = self.authorities.get(&key) else {
                continue;
            };
            // 没有旧形状曲线 ⇒ 无事可做（这也是幂等性的来源）。
            if !params
                .curves
                .values()
                .any(|curve| curve.basis.is_some() || curve.domain != CurveDomain::Source)
            {
                continue;
            }
            let Ok(geometry) = geometry(clip) else {
                continue;
            };
            let frame_ms = params.template.frame_period_ms;
            let mut curves = BTreeMap::new();
            for (param, curve) in params.curves.iter() {
                let converted = curve_into_source_domain(curve, &geometry, param, frame_ms)
                    .unwrap_or_else(|| curve.clone());
                curves.insert(param.clone(), converted);
            }
            let mut next = params.clone();
            next.curves = curves;
            // 旧 `anchor: "Project"` 读进来是 `PinProject { frozen: [] }`：折算用的
            // 就是这份几何，于是把冻结几何补成它 —— 曲线停在用户此刻看到的位置。
            if matches!(next.placement, Placement::PinProject { ref frozen } if frozen.is_empty()) {
                next.placement = Placement::pinned(geometry);
            }
            migrated += 1;
            pending.push((key, next));
        }
        for (key, params) in pending {
            self.set_authority(&key, params);
        }
        migrated
    }

    /// 让「锁定参数线」的当前设置（`wanted`）对**既有**权威生效（纯元数据，O(1)）。
    ///
    /// 【为什么这是一次元数据操作】曲线永远按源秒存，开关只决定"用哪份几何投影"。
    /// 于是拨开关不需要搬数组、不需要重采样、不需要模型收口 —— 下一次投影就生效，
    /// 拨回去逐帧回到原样。旧实现靠一次性烘焙，而烘焙必然有损、有损就必须有闸、
    /// 有闸就必然出现"闸拒绝 ⇒ 开关无效"（用户原话：完全没反应）。
    ///
    /// 【为什么冻结几何取第一条活 clip】同一 modification 被多条 region 共享时，
    /// 它们此刻的几何可以不同（分割之后两半各自被拖过）。取第一条是**确定性**的，
    /// 而且"钉住"本身描述的是"此刻把它按在哪"，用户看到的正是那一条。
    ///
    /// 返回被改动的权威条数。
    pub(crate) fn set_placement(
        &mut self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
        wanted: &Placement,
    ) -> usize {
        // 同一条权威可能被多条 live region 共享（Cubase 分割、复制后共享 modification）。
        // 钉住必须收集**全部**几何；只取第一条会让右半段按父段整窗读，整段落到界外。
        let mut geometries_by_key: BTreeMap<String, Vec<RegionGeometry>> = BTreeMap::new();
        for clip in &timeline.clips {
            let Some(identity) = identities.get(&clip.id) else {
                continue;
            };
            let Some(key) = authority_key(identity.item.as_deref(), &identity.modification) else {
                continue;
            };
            let Some(geometry) = geometry(clip).ok() else {
                continue;
            };
            let windows = geometries_by_key.entry(key).or_default();
            if !windows.iter().any(|known| known.equivalent(&geometry)) {
                windows.push(geometry);
            }
        }
        for windows in geometries_by_key.values_mut() {
            windows.sort_by(|a, b| {
                a.project_start
                    .partial_cmp(&b.project_start)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        a.project_duration
                            .partial_cmp(&b.project_duration)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });
            windows.truncate(256);
        }
        let mut changed = 0_usize;
        for (key, windows) in geometries_by_key {
            let Some(params) = self.authorities.get_mut(&key) else {
                continue;
            };
            let target = match wanted {
                Placement::FollowMaterial => Placement::FollowMaterial,
                Placement::PinProject { .. } => match &params.placement {
                    // 已经有冻结窗口 ⇒ 保留它：来回拨开关不该把“钉在哪”悄悄改掉。
                    Placement::PinProject { frozen } if !frozen.is_empty() => {
                        Placement::PinProject {
                            frozen: frozen.clone(),
                        }
                    }
                    _ if !windows.is_empty() => Placement::pinned_many(windows),
                    // 没有任何活几何可钉：保留待冻结状态，下一次拿到几何时补上。
                    _ => Placement::PinProject { frozen: Vec::new() },
                },
            };
            if params.placement != target {
                params.placement = target;
                changed += 1;
            }
        }
        changed
    }

    /// 把旧存档里的 per-clip 记录折算进权威表（幂等）。
    ///
    /// 【为什么需要它】新存档不再写 `regions`（见 [`ParameterAtlas::regions`]），
    /// 所以"权威表为空而 per-clip 记录非空"只可能出现在**读旧存档**时。此处用旧规则
    /// （源 + modification + 几何包含）认领一次，把结果写进权威表；此后所有帧
    /// （以及后续的宿主分割）都走权威快路径。
    fn migrate_legacy_records(
        &mut self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
    ) {
        if self.regions.is_empty() {
            return;
        }
        let mut migrated = 0_usize;
        for clip in &timeline.clips {
            let Some(identity) = identities.get(&clip.id) else {
                continue;
            };
            let Some(key) = authority_key(identity.item.as_deref(), &identity.modification) else {
                continue;
            };
            if self.authorities.contains_key(&key) {
                continue;
            }
            let Some(root) = timeline.resolve_root_track_id(&clip.track_id) else {
                continue;
            };
            let Ok(geometry) = geometry(clip) else {
                continue;
            };
            // 先克隆再插入：`find` 借用 `self.regions`，不能与 `self.authorities` 的可变借用共存。
            let seed = match self.find(identity, &root, &geometry) {
                Ok(Some(record)) => record.clone(),
                Ok(None) => continue,
                Err(_) => continue,
            };
            self.authorities.insert(
                key,
                AuthorityParameters::from_record(&seed, Placement::FollowMaterial),
            );
            migrated += 1;
        }
        if migrated > 0 {
            // 迁移每份存档只发生一次，不需要限流；`info` 是它的正确级别。
            log::info!(
                "[ara] atlas migrated {migrated} parameter set(s) from legacy per-clip records; \
                 new archives store them by stable host identity"
            );
        }
    }

    /// 从权威表派生本帧的 per-clip 投影缓存（纯函数）。
    fn derive_regions(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
    ) -> Result<BTreeMap<String, RegionParameters>, String> {
        let mut derived = BTreeMap::new();
        for clip in &timeline.clips {
            let Some(identity) = identities.get(&clip.id) else {
                continue;
            };
            let Some(root) = timeline.resolve_root_track_id(&clip.track_id) else {
                continue;
            };
            let Ok(geometry) = geometry(clip) else {
                continue;
            };
            let Some(authority) = self.authority(identity, &root, &geometry)? else {
                continue;
            };
            derived.insert(
                clip.id.clone(),
                RegionParameters {
                    identity: identity.clone(),
                    root,
                    current: geometry,
                    live: true,
                    template: authority.template().clone(),
                    edits: authority.edits().cloned().unwrap_or_default(),
                    curves: authority.curves().clone(),
                },
            );
        }
        Ok(derived)
    }

    /// 一条 clip 的参数权威。
    ///
    /// 【查表顺序】先按稳定宿主身份（[`authority_key`]）查权威表；item 级缺席时按
    /// modification 级兜底；两者都没有才退回 per-clip 记录（旧存档迁移窗口）。
    /// 除最后一步外都是**确定性查表**，不做任何几何猜测 —— 几何兜底曾是
    /// "分割后曲线整体平移一个分割长度"的直接来源（兄弟半段的源窗相邻，
    /// 会被判成合法祖先），因此这里刻意**不读** `split_parents` /
    /// `ara_split_parents` 那些谱系表：归属由该 clip 自己的源窗与权威曲线的
    /// 源区间求交决定，不需要推断祖先。
    fn authority<'a>(
        &'a self,
        identity: &RegionIdentity,
        root: &str,
        geometry: &RegionGeometry,
    ) -> Result<Option<Authority<'a>>, String> {
        if let Some(key) = authority_key(identity.item.as_deref(), &identity.modification) {
            if let Some(params) = self.authorities.get(&key) {
                if params.source.is_empty() || params.source == identity.source {
                    return Ok(Some(Authority::Current(params)));
                }
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Warn,
                    "ara.atlas.source-mismatch",
                    format_args!(
                        "[ara] parameter authority for key={key} belongs to source {} but the                          live region reads {}; the old source curve is not projected",
                        params.source, identity.source
                    ),
                );
            }
        }
        // item 级缺席时按 modification 级兜底：宿主 `cloneAudioModification` 只能给出
        // modification，它播的就是这一级。原生宿主上 item 级一旦被捕获就会抢先命中，
        // 因此这条兜底只在"克隆刚发生、新 item 还没被编辑过"的窗口里生效。
        if identity.item.is_some() {
            if let Some(params) = self
                .authorities
                .get(&format!("mod:{}", identity.modification))
            {
                if params.source.is_empty() || params.source == identity.source {
                    return Ok(Some(Authority::Current(params)));
                }
            }
        }
        let legacy = self.find(identity, root, geometry)?.map(Authority::Legacy);
        if legacy.is_none() {
            // 【为什么这条日志是必需的】权威查表未命中此前是**完全静默**的，而它正是
            // "参数被重置"的第一现场：没命中 ⇒ 该 clip 不投影曲线 ⇒ 渲染退回根轨道
            // 基线。真机上因此无法区分"没查到"与"查到了但装的是别人的窗口"——
            // 那正是这个缺陷能活过上一轮的直接原因。带上身份三要素与键，可判定。
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.atlas.authority-miss",
                format_args!(
                    "[ara] no parameter authority for item={:?} modification={} source={} \
                     (keys={}); this clip projects no curves this round",
                    identity.item.as_deref().unwrap_or(""),
                    identity.modification,
                    identity.source,
                    self.authorities.len(),
                ),
            );
        }
        Ok(legacy)
    }
    /// 冷恢复只在本组件真实范围内绑定；重复持久身份且源窗口相同仍拒绝，不能按旧key猜。
    ///
    /// 【为什么不再需要几何绑定】权威是 modification persistentID，而它是**跨会话稳定**
    /// 的（ARA 明确保证，`ARAInterface.h:903` / `:953-955`）。因此冷恢复只需要把旧存档
    /// 的 per-clip 记录折算进权威表，再按当前活图派生投影缓存 —— 没有"这条记录对应哪条
    /// clip"这个问题，也就没有"唯一候选 or 失败"这条脆弱判据。
    pub fn rebind(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
    ) -> Result<Self, String> {
        let mut bound = self.clone();
        bound.migrate_legacy_records(timeline, identities);
        bound.regions = bound.derive_regions(timeline, identities)?;
        bound.validate()?;
        Ok(bound)
    }
    /// GUI仍用原track网格：显示投影可选重叠区，音频始终保留每region自己的参数。
    ///
    /// 只要参数数组时用它；需要"这一轮投影到了哪些帧"（按帧合并）时用
    /// [`Self::project_root_records`]。
    #[cfg(test)]
    pub fn project_roots(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
    ) -> Result<BTreeMap<String, TrackParamsState>, String> {
        Ok(self
            .project_root_records(timeline, identities)?
            .into_iter()
            .map(|(root, projection)| (root, projection.params))
            .collect())
    }

    /// 与 [`Self::project_roots`] 相同，但**带上"这一轮真的投影到了哪些帧"**。
    ///
    /// 【G4：只给参数数组是不够的】任何一条 clip 拿不到权威（日志里 `keys=0` 的那种）
    /// 都会让它的帧段落成默认值。只交一份参数数组时，调用方唯一的合并方式是
    /// **整值替换**整条根轨 —— 于是那一段默认值会把用户曲线抹平。带上 `frames` 之后，
    /// 合并可以只覆盖这一轮真的投影过的帧（见 `merge_root_projection`）。
    ///
    /// 【为什么两个入口】绝大多数调用方只要参数数组（渲染、编辑器）。只有需要
    /// **按帧合并**的地方（`merge_root_projection`）才需要 `frames`，而它们正是
    /// "整值替换会抹平用户曲线"的那几处。
    pub fn project_root_records(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
    ) -> Result<BTreeMap<String, RootProjection>, String> {
        let mut roots = BTreeMap::<String, RootProjection>::new();
        let baseline_for = |root: &str| timeline.params_by_root_track.get(root).cloned();
        let baseline_ready_for = |root: &str| {
            timeline
                .params_by_root_track
                .get(root)
                .is_some_and(|params| !params.pitch_orig.is_empty())
        };

        // 空白编辑是最低层：只覆盖 clip 外的帧，clip 一旦盖住该位置就暂时隐藏。
        for (root, gaps) in &self.gaps {
            if !timeline.tracks.iter().any(|track| &track.id == root) {
                continue;
            }
            let mut params = baseline_for(root).unwrap_or_else(|| gaps.template.clone());
            clear_user_curves(&mut params);
            let mut frames = Vec::new();
            for (key, curve) in &gaps.curves {
                let dense = gap_values(curve, key, params.frame_period_ms)?;
                let ranges = gap_ranges(curve, params.frame_period_ms)?;
                if ranges.is_empty() {
                    continue;
                }
                set_curve(&mut params, key, dense);
                for range in ranges {
                    push_frame_range(&mut frames, range);
                }
            }
            if let Some(existing) = roots.get_mut(root) {
                // 同一根不应有两条 gap 记录；保留已合并的帧。
                for range in frames {
                    push_frame_range(&mut existing.frames, range);
                }
            } else {
                roots.insert(
                    root.clone(),
                    RootProjection {
                        params,
                        frames,
                        edited_frames: BTreeMap::new(),
                        complete: true,
                        baseline_ready: baseline_ready_for(root),
                        cleared_curves: std::collections::BTreeSet::new(),
                    },
                );
            }
        }

        let mut expected = BTreeMap::<String, usize>::new();
        for clip in &timeline.clips {
            if let Some(root) = timeline.resolve_root_track_id(&clip.track_id) {
                *expected.entry(root).or_default() += 1;
            }
        }
        let mut projected = BTreeMap::<String, usize>::new();
        let mut explicit_masks = BTreeMap::<(String, String), Vec<bool>>::new();
        let mut order = timeline.clips.iter().collect::<Vec<_>>();
        // 选中的 clip 最后处理：它的显式编辑优先于其它 clip；它的未编辑帧不会
        // 覆盖其它 clip 已经赢下的显式编辑（见下面的 explicit_masks）。
        order.sort_by_key(|clip| Some(&clip.id) == timeline.selected_clip_id.as_ref());
        for clip in order {
            let Ok(geometry) = geometry(clip) else {
                continue;
            };
            let Some(root) = timeline.resolve_root_track_id(&clip.track_id) else {
                continue;
            };
            let Some(detail) =
                self.project_clip(timeline, identities, clip, false, &default_placement())?
            else {
                continue;
            };
            let Some((begin, end)) = covered_frame_range(&geometry, detail.params.frame_period_ms)?
            else {
                continue;
            };
            *projected.entry(root.clone()).or_default() += 1;
            let entry = roots.entry(root.clone()).or_insert_with(|| {
                let mut params = baseline_for(&root).unwrap_or_else(|| {
                    let mut empty = detail.params.clone();
                    clear_curves(&mut empty);
                    empty
                });
                clear_user_curves(&mut params);
                RootProjection {
                    params,
                    frames: Vec::new(),
                    edited_frames: BTreeMap::new(),
                    complete: true,
                    baseline_ready: baseline_ready_for(&root),
                    cleared_curves: std::collections::BTreeSet::new(),
                }
            });
            if entry.params.frame_period_ms != detail.params.frame_period_ms {
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Warn,
                    "ara.atlas.frame-period-mismatch",
                    format_args!(
                        "[ara] clip {} declares {} ms frames but its root already uses {} ms; \
                         its curves are not merged into the root view",
                        clip.id, detail.params.frame_period_ms, entry.params.frame_period_ms
                    ),
                );
                continue;
            }
            push_frame_range(&mut entry.frames, begin..end.saturating_add(1));
            entry.baseline_ready |= detail.baseline_ready;
            entry.params.pitch_edit_user_modified |= detail.params.pitch_edit_user_modified;

            // clip 投影的每条曲线：显式编辑覆盖一切；未编辑帧只覆盖空白层，不覆盖
            // 另一条 clip 的显式编辑。
            for (key, values) in parameter_curves(&detail.params) {
                if matches!(key.as_str(), "pitch_orig" | "dyn_orig" | "tension_orig")
                    || values.is_empty()
                {
                    continue;
                }
                let mask = explicit_masks
                    .entry((root.clone(), key.clone()))
                    .or_insert_with(|| vec![false; end.saturating_add(1)]);
                if mask.len() <= end {
                    mask.resize(end.saturating_add(1), false);
                }
                let edited_ranges = detail.edited_frames.get(&key).cloned().unwrap_or_default();
                let mut edited_mask = vec![false; end.saturating_sub(begin).saturating_add(1)];
                for range in &edited_ranges {
                    let lo = range.start.max(begin);
                    let hi = range.end.min(end.saturating_add(1));
                    if lo < hi {
                        for bit in &mut edited_mask[lo - begin..hi - begin] {
                            *bit = true;
                        }
                    }
                }
                let existing_values = parameter_curves(&entry.params)
                    .into_iter()
                    .find(|(name, _)| *name == key)
                    .map(|(_, values)| values.to_vec())
                    .unwrap_or_default();
                let mut merged = existing_values;
                if merged.len() <= end {
                    merged.resize(end.saturating_add(1), pad(&key));
                }
                for frame in begin..=end {
                    let is_edit = edited_mask[frame - begin];
                    if is_edit {
                        merged[frame] = values[frame];
                        mask[frame] = true;
                    } else if detail.baseline_ready && !mask[frame] {
                        merged[frame] = values[frame];
                    }
                }
                set_curve(&mut entry.params, &key, merged);
                if !edited_mask.iter().any(|bit| *bit) {
                    continue;
                }
                let ranges = entry.edited_frames.entry(key.clone()).or_default();
                for range in edited_ranges {
                    let lo = range.start.max(begin);
                    let hi = range.end.min(end.saturating_add(1));
                    if lo < hi {
                        push_frame_range(ranges, lo..hi);
                    }
                }
            }
        }

        // 不锁定时，编辑留在**冻结的工程位置**：即使 clip 已经移走，旧位置的线仍在。
        // 这一层是显式编辑，最后叠加，保证不会被新位置上的未编辑 baseline 抹掉。
        let mut pinned_seen = std::collections::BTreeSet::new();
        for clip in &timeline.clips {
            let Some(identity) = identities.get(&clip.id) else {
                continue;
            };
            let Some(root) = timeline.resolve_root_track_id(&clip.track_id) else {
                continue;
            };
            let Some(key) = authority_key(identity.item.as_deref(), &identity.modification) else {
                continue;
            };
            if !pinned_seen.insert((root.clone(), key.clone())) {
                continue;
            }
            let Some(params) = self.authorities.get(&key) else {
                continue;
            };
            if !matches!(params.placement, Placement::PinProject { .. }) {
                continue;
            }
            let windows = params.placement.frozen().to_vec();
            if windows.is_empty() {
                continue;
            }
            let entry = roots.entry(root.clone()).or_insert_with(|| {
                let mut shell = baseline_for(&root).unwrap_or_else(|| params.template.clone());
                clear_user_curves(&mut shell);
                RootProjection {
                    params: shell,
                    frames: Vec::new(),
                    edited_frames: BTreeMap::new(),
                    complete: true,
                    baseline_ready: baseline_ready_for(&root),
                    cleared_curves: std::collections::BTreeSet::new(),
                }
            });
            let frame_ms = entry.params.frame_period_ms;
            for window in windows {
                let Some((begin, end)) = covered_frame_range(&window, frame_ms)? else {
                    continue;
                };
                for (curve_key, curve) in &params.edits {
                    if curve.is_empty() {
                        continue;
                    }
                    let existing = parameter_curves(&entry.params)
                        .into_iter()
                        .find(|(name, _)| name == curve_key)
                        .map(|(_, values)| values.to_vec())
                        .unwrap_or_default();
                    let mut merged = existing;
                    if merged.len() <= end {
                        merged.resize(end.saturating_add(1), pad(curve_key));
                    }
                    let map = window.map()?;
                    for frame in begin..=end {
                        let project_sec = frame as f64 * frame_ms / 1000.0;
                        let Some(source_sec) = map.project_to_source(project_sec) else {
                            continue;
                        };
                        let source_frame = source_sec * 1000.0 / curve.frame_ms;
                        if let Some(value) = curve.value_at(source_frame, curve_key) {
                            merged[frame] = value;
                            push_frame_range(&mut entry.frames, frame..frame + 1);
                            let ranges = entry.edited_frames.entry(curve_key.clone()).or_default();
                            push_frame_range(ranges, frame..frame + 1);
                        }
                    }
                    set_curve(&mut entry.params, curve_key, merged);
                }
            }
        }

        for (root, record) in roots.iter_mut() {
            record.complete = expected.get(root).copied().unwrap_or(0)
                == projected.get(root).copied().unwrap_or(0);
            // 权威里没有这条用户曲线 ⇒ 用户明确清空过整条线；只有 complete 才允许删除。
            if record.complete {
                let mut intended = std::collections::BTreeSet::new();
                let mut emptied = std::collections::BTreeSet::new();
                for clip in &timeline.clips {
                    let Some(identity) = identities.get(&clip.id) else {
                        continue;
                    };
                    let Some(clip_root) = timeline.resolve_root_track_id(&clip.track_id) else {
                        continue;
                    };
                    if clip_root != *root {
                        continue;
                    }
                    if let Some(key) =
                        authority_key(identity.item.as_deref(), &identity.modification)
                    {
                        if let Some(params) = self.authorities.get(&key) {
                            for (curve_key, curve) in &params.edits {
                                if curve.is_empty() {
                                    emptied.insert(curve_key.clone());
                                } else {
                                    intended.insert(curve_key.clone());
                                }
                            }
                        }
                    }
                }
                if let Some(gaps) = self.gaps.get(root) {
                    for key in gaps.curves.keys() {
                        intended.insert(key.clone());
                    }
                }
                for key in parameter_curves(&record.params)
                    .into_iter()
                    .map(|(key, _)| key)
                {
                    if matches!(
                        key.as_str(),
                        "pitch_edit" | "pitch_orig" | "dyn_orig" | "tension_orig"
                    ) {
                        continue;
                    }
                    if !intended.contains(&key) {
                        record.cleared_curves.insert(key);
                    }
                }
                for key in emptied {
                    if !intended.contains(&key) {
                        record.cleared_curves.insert(key);
                    }
                }
            }
        }
        Ok(roots)
    }

    /// 接受当前投影上的真实编辑；未改动的曲线继续保留原始源坐标basis。
    /// 接受当前投影上的真实编辑；未改动的曲线继续保留原始源坐标basis。
    ///
    /// 【为什么保留一个"不要读数"的入口】生产路径必须走
    /// [`Self::capture_changes_reporting`]（读数要进状态快照）；这个入口只服务测试
    /// 与"不需要报告"的调用方。少一个读数不该变成一次编译错误。
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn capture_changes(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
        previous_view: &BTreeMap<String, TrackParamsState>,
    ) -> Result<Self, String> {
        self.capture_changes_reporting(
            timeline,
            identities,
            previous_view,
            &mut CaptureReport::default(),
        )
    }
    /// 与 [`Self::capture_changes`] 相同，但把**读数**交回调用方。
    ///
    /// 【为什么生产路径必须用它】"这次编辑没有被采纳"要进状态快照与诊断包，
    /// 而读数只有调用方（`DocumentSession`）有地方存。丢掉读数就是让第 3 点
    /// 那 11 条静默回退继续静默。
    pub(crate) fn capture_changes_reporting(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
        previous_view: &BTreeMap<String, TrackParamsState>,
        report: &mut CaptureReport,
    ) -> Result<Self, String> {
        self.capture_inner(
            timeline,
            identities,
            WriteIntent::User {
                baseline: previous_view,
            },
            default_placement(),
            report,
        )
    }
    pub fn capture(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
    ) -> Result<Self, String> {
        self.capture_inner(
            timeline,
            identities,
            WriteIntent::Derived,
            default_placement(),
            &mut CaptureReport::default(),
        )
    }

    /// 只把相对旧GUI投影的delta送回源authority；重叠选区外保留该region自己的值。
    fn capture_inner(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
        intent: WriteIntent<'_>,
        placement: Placement,
        report: &mut CaptureReport,
    ) -> Result<Self, String> {
        let previous_view = match &intent {
            WriteIntent::User { baseline } => Some(*baseline),
            WriteIntent::Derived => None,
        };
        // 派生写入没有用户意图：已有权威时只更新派生缓存，绝不把当前投影
        // （可能已经是插值过的显示数组）当成新的源数据回写。
        if matches!(intent, WriteIntent::Derived) && !self.authorities.is_empty() {
            return Ok(self.clone());
        }
        let mut candidate = self.clone();
        // 首次接触旧权威时先把整条 curves 折算成稀疏 edits；新写入永远只落 edits。
        candidate.migrate_legacy_authority_edits();
        candidate.capture_gaps_sparse(timeline, previous_view)?;
        for clip in &timeline.clips {
            let Some(identity) = identities.get(&clip.id) else {
                continue;
            };
            if identity.source.is_empty()
                || identity.modification.is_empty()
                || clip.source_path.as_deref() != Some(identity.source.as_str())
            {
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
            let Ok(geometry) = geometry(clip) else {
                continue;
            };
            let frame_ms = params.frame_period_ms;
            let (begin, _end) = frame_range(&geometry, frame_ms)?;
            let view = previous_view
                .and_then(|view| view.get(&root))
                .filter(|view| view.frame_period_ms == frame_ms);
            if previous_view.is_some_and(|view| !view.is_empty()) && view.is_none() {
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Warn,
                    "ara.atlas.baseline-root-missing",
                    format_args!(
                        "[ara] line write skipped: no baseline array for root {root} this round \
                         (the root id changed); clip {} keeps its authority unchanged instead of \
                         treating the whole visible window as user input",
                        clip.id
                    ),
                );
                report.skipped_clips += 1;
                continue;
            }
            let baseline_present = view.is_some();
            // 编辑器没有改动任何用户曲线：这是几何刷新后的同一份投影，不能把
            // 插值结果再写回源域（重复往返会逐次模糊曲线）。
            if let Some(baseline) = view {
                if baseline.pitch_edit == params.pitch_edit
                    && baseline.tension_edit == params.tension_edit
                    && baseline.extra_curves == params.extra_curves
                {
                    continue;
                }
            }
            let protected = timeline
                .selected_clip_id
                .as_ref()
                .and_then(|id| timeline.clips.iter().find(|selected| selected.id == *id))
                .filter(|selected| {
                    timeline
                        .resolve_root_track_id(&selected.track_id)
                        .as_deref()
                        == Some(root.as_str())
                        && selected.id != clip.id
                })
                .map(|selected| (selected.start_sec, selected.start_sec + selected.length_sec));
            let desired = extract_edit_curves(params);
            let baseline_edits = view.map(extract_edit_curves);
            let key_for_anchor = authority_key(identity.item.as_deref(), &identity.modification);
            let Some(key_for_anchor) = key_for_anchor else {
                continue;
            };
            let existing = candidate
                .authorities
                .get(&key_for_anchor)
                .map(|authority| {
                    (
                        authority.edits.clone(),
                        authority.placement.clone(),
                        authority.last_edit_seq,
                    )
                })
                .or_else(|| {
                    self.authorities.get(&key_for_anchor).map(|authority| {
                        (
                            authority.edits.clone(),
                            authority.placement.clone(),
                            authority.last_edit_seq,
                        )
                    })
                });
            let had_existing = existing.is_some();
            let (mut edits, inherited_placement, last_seq) =
                existing.unwrap_or_else(|| (BTreeMap::new(), placement.clone(), 0));
            let mut keys = std::collections::BTreeSet::new();
            keys.extend(edits.keys().cloned());
            keys.extend(desired.curves.keys().cloned());
            if let Some(baseline) = &baseline_edits {
                keys.extend(baseline.curves.keys().cloned());
            }
            let mut changed = false;
            for key in keys {
                let old = edits.get(&key).cloned();
                let desired_source = desired
                    .get(&key)
                    .map(|curve| {
                        project_curve_to_source(curve, &key, &geometry, frame_ms, protected)
                    })
                    .transpose()?;
                let baseline_source = baseline_edits
                    .as_ref()
                    .and_then(|baseline| baseline.get(&key))
                    .map(|curve| project_curve_to_source(curve, &key, &geometry, frame_ms, None))
                    .transpose()?;
                let mut values = old
                    .as_ref()
                    .map(EditCurve::frame_values)
                    .unwrap_or_default();
                if baseline_present && baseline_source.is_none() && old.is_some() {
                    report.no_baseline_curves += 1;
                }
                let mut removed = 0_usize;
                if baseline_present {
                    if let Some(baseline_curve) = &baseline_source {
                        let desired_curve = desired_source.as_ref();
                        let old_frames = values.keys().copied().collect::<Vec<_>>();
                        for source_frame in old_frames {
                            let source_sec = source_frame as f64 * frame_ms / 1000.0;
                            let Some(project_sec) = geometry.map()?.source_to_project(source_sec)
                            else {
                                continue;
                            };
                            if project_sec < geometry.project_start - 1e-9
                                || project_sec
                                    > geometry.project_start + geometry.project_duration + 1e-9
                            {
                                continue;
                            }
                            if protected
                                .is_some_and(|(lo, hi)| project_sec >= lo && project_sec <= hi)
                            {
                                continue;
                            }
                            let source_frame_f = source_frame as f64;
                            let baseline_hit =
                                baseline_curve.value_at(source_frame_f, &key).is_some();
                            let desired_hit = desired_curve.is_some_and(|curve| {
                                curve.value_at(source_frame_f, &key).is_some()
                            });
                            if baseline_hit && !desired_hit {
                                values.remove(&source_frame);
                                removed += 1;
                            }
                        }
                    }
                }
                if let Some(desired_curve) = &desired_source {
                    for (frame, value) in desired_curve.frame_values() {
                        values.insert(frame, value);
                    }
                    changed = true;
                }
                let rebuilt = EditCurve::from_frame_values(frame_ms, &key, values);
                let had_old = old.is_some_and(|curve| !curve.is_empty());
                let had_baseline = baseline_source
                    .as_ref()
                    .is_some_and(|curve| !curve.is_empty());
                if rebuilt.is_empty() {
                    // 用户明确擦空（或旧曲线被证据删除）⇒ 保留一个空 key 作为
                    // “这条线已经清空”的显式标记；下一次投影才能真的清除 GUI 数组。
                    if had_old || had_baseline {
                        edits.insert(key.clone(), rebuilt);
                    } else {
                        edits.remove(&key);
                    }
                } else {
                    edits.insert(key.clone(), rebuilt);
                    changed = true;
                }
                if removed > 0 {
                    report.removed_frames += removed;
                    changed = true;
                }
            }
            // 第一次捕获没有显式编辑：不建立空权威，避免把“从没编辑”固化。
            if edits.is_empty() && !had_existing {
                continue;
            }
            let mut template = params.clone();
            clear_curves(&mut template);
            template.pitch_edit_user_modified = false;
            let placement = match inherited_placement {
                Placement::PinProject { ref frozen } if frozen.is_empty() => {
                    Placement::pinned(geometry.clone())
                }
                other => other,
            };
            let next_seq = if changed {
                last_seq.saturating_add(1).max(1)
            } else {
                last_seq
            };
            candidate.set_authority(
                &key_for_anchor,
                AuthorityParameters {
                    edits,
                    source: identity.source.clone(),
                    curves: BTreeMap::new(),
                    template,
                    placement: placement.clone(),
                    last_edit_seq: next_seq,
                    unseen_rounds: 0,
                    refused_rounds: 0,
                },
            );
            let (edit_curves, edit_runs, edit_samples) = candidate
                .authorities
                .get(&key_for_anchor)
                .map(|authority| {
                    (
                        authority.edits.len(),
                        authority
                            .edits
                            .values()
                            .map(|curve| curve.runs.len())
                            .sum::<usize>(),
                        authority
                            .edits
                            .values()
                            .map(|curve| {
                                curve.runs.iter().map(|run| run.values.len()).sum::<usize>()
                            })
                            .sum::<usize>(),
                    )
                })
                .unwrap_or((0, 0, 0));
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.atlas.line-write",
                format_args!(
                    "[ara] line write: root={root} key={key_for_anchor} placement={placement:?} \
                     edit_curves={edit_curves} frames={begin}..{_end} baseline={} edit_runs={edit_runs} \
                     edit_samples={edit_samples} accepted=1",
                    if baseline_present { "present" } else { "absent" },
                ),
            );
            candidate.regions.insert(
                clip.id.clone(),
                RegionParameters {
                    identity: identity.clone(),
                    root,
                    current: geometry,
                    live: true,
                    template: candidate
                        .authorities
                        .get(&key_for_anchor)
                        .map(|authority| authority.template.clone())
                        .unwrap_or_default(),
                    edits: candidate
                        .authorities
                        .get(&key_for_anchor)
                        .map(|authority| authority.edits.clone())
                        .unwrap_or_default(),
                    curves: BTreeMap::new(),
                },
            );
        }
        candidate.validate()?;
        Ok(candidate)
    }

    #[allow(dead_code)]
    fn capture_legacy_inner(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
        intent: WriteIntent<'_>,
        placement: Placement,
        report: &mut CaptureReport,
    ) -> Result<Self, String> {
        // 【为什么从 intent 里取基线，而不是另开一个参数】基线的**有无**就是"这次写入
        // 算不算用户意图"的全部内容。分成两个参数，迟早会出现"给了基线却按派生写"或
        // 反过来 —— 而那正是本模块此前把用户删除当成数据损坏的机理。
        let previous_view = match &intent {
            WriteIntent::User { baseline } => Some(*baseline),
            WriteIntent::Derived => None,
        };
        let mut candidate = self.clone();
        candidate.capture_gaps(timeline, previous_view)?;
        for clip in &timeline.clips {
            // 【为什么身份缺席只跳过】与几何不可解同一纪律（见下方注释）。此前这里
            // `ok_or(...)?` 会把"身份表这一轮还没跟上"升级成整份捕获失败，于是用户的
            // 一次编辑被整份拒绝。跳过它：该片段这次不落曲线，其余片段照常。
            let Some(identity) = identities.get(&clip.id) else {
                continue;
            };
            // 【为什么不再要求 `key != 0`】权威是 modification persistentID，而 `key` 是
            // region 的 model-ref 地址 —— 它 `#[serde(skip)]`、且 `allocate_region` 会复用
            // 已销毁槽位。此前这条守卫让**任何一次存盘/重载之后一条曲线都捕获不到**。
            // 现在 `key` 只是诊断字段，身份自相矛盾只看源与 modification。
            if identity.source.is_empty()
                || identity.modification.is_empty()
                || clip.source_path.as_deref() != Some(identity.source.as_str())
            {
                // 身份自相矛盾（源为空 / 与 clip 的源不符）时同样只跳过这一条：
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
            // 几何必须先可解：帧域非法时只跳过这一条片段（见 `frame_range` 的守卫）。
            let (begin, end) = frame_range(&geometry, params.frame_period_ms)?;
            // 旧曲线取**权威侧**（modification），旧存档未迁移时退回 per-clip 记录。
            let authority = self.authority(identity, &root, &geometry)?;
            // 【锚点**永远**取当前设置】这是一处曾经写反的地方，也是"开关无效"的直接原因。
            //
            // 旧实现让既有权威**沿用入库时**定下的锚点。于是读路径用当前设置、写路径用
            // 入库值，两者可以相反：用户关掉「锁定参数线」之后，投影按工程域解释一份
            // 源域数组（界面上每条线整体偏移一个源窗起点），而任何一次自动写回又按源域
            // 把它写回去 —— 错位被固化，且不可逆。
            //
            // 现在只有一个来源：`settings_store::lock_param_lines()`。它变了，曲线就在
            // 那一次捕获里被**烘焙**到新域（见下面的 `bake_curve_to_anchor`），之后读写
            // 继续共用同一个判断。读与写不可能再相反。
            let key_for_anchor = authority_key(identity.item.as_deref(), &identity.modification);
            let mut curves = BTreeMap::new();
            // 【U-10 的读数】这一轮有没有"编辑器读到的那份"可比。`absent` 出现在写日志里
            // 就说明 U-4 的守卫被绕过（那会把默认值当成用户改动写回）。
            let view = previous_view
                .and_then(|view| view.get(&root))
                .filter(|view| view.frame_period_ms == params.frame_period_ms);
            // 【不变式】只要调用方给了基线，就必须能在这个根上找到它。
            // 找不到只有一种成因：这个根的 id 变了（宿主重建序列），而基线还是旧键。
            // 此时**当前视图不是用户输入** —— 把整窗写回等于把"投影不完整产生的默认值"
            // 固化成用户曲线（不可逆）。跳过这一条 clip（其余照常），并留一行可复盘的话。
            // 【为什么"空基线"不算基线】一份**空的**参数视图只说明"这份工作区还没有
            // 任何曲线"（新工程 / 第一次画线）。那种场合下"当前视图里的非默认帧就是
            // 用户刚画的"是唯一可用的解释 —— 拦掉它会让第一条曲线永远画不上。
            // 只有"基线里**有别的根**、却没有这一个"才说明根键错位（宿主重建了序列）。
            if previous_view.is_some_and(|view| !view.is_empty()) && view.is_none() {
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Warn,
                    "ara.atlas.baseline-root-missing",
                    format_args!(
                        "[ara] line write skipped: no baseline array for root {root} this round \
                         (the root id changed); clip {} keeps its authority unchanged instead of \
                         treating the whole visible window as user input",
                        clip.id
                    ),
                );
                report.skipped_clips += 1;
                continue;
            }
            let baseline_present = view.is_some();
            // 这一条 clip 上有多少条曲线被"按帧证据"拒绝了（决定 refused_rounds 与日志级别）。
            let mut refused_here = 0_usize;
            for (key, values) in parameter_curves(params) {
                if values.is_empty() {
                    continue;
                }
                // 【为什么优先取 `candidate` 而不是 `self`】一次 capture 会依次处理
                // **共享同一个 modification 的多条 region**。前一条刚写进 `candidate`
                // 的源区间必须作为"既有曲线"参与并集，否则后一条会把它覆盖掉 ——
                // 那正是"分割后参数完全重置"的原形。
                let old = key_for_anchor
                    .as_deref()
                    .and_then(|key| candidate.authorities.get(key))
                    .and_then(|params| params.curves.get(&key))
                    .or_else(|| authority.as_ref().and_then(|a| a.curves().get(&key)));
                let old_view = view.and_then(|view| {
                    parameter_curves(view)
                        .into_iter()
                        .find(|(name, _)| *name == key)
                        .map(|(_, values)| values)
                });
                // 【R3：有曲线、但基线里**没有这条数组**】
                //
                // 【为什么不再"保留旧曲线、跳过这一条"】那等于把用户刚画的东西静默丢掉：
                // 权威里有一条曲线、而基线里没有对应数组时，"用户改了什么"无从差分，
                // 但"用户刚画出来的非默认帧"仍然是可以确认的意图。旧实现把它整条丢掉，
                // 于是这一次编辑连日志都没有。
                //
                // 现在把基线当成**空数组**：`prior` 恒为 `pad` ⇒ 只有非默认帧会被写回，
                // 一帧都不会被删。这既保住了用户刚画的部分，又不可能抹平任何东西 ——
                // 而"能不能删"这件事由下面的按帧证据判据独立裁决。
                let old_view = match (previous_view.is_some(), old_view) {
                    // 【为什么"空数组"也算缺席】`parameter_curves` 对一条**存在但为空**
                    // 的数组也会交出一个空切片。把它当成基线，等于声称"旧视图在这一帧是
                    // 默认值" —— 而实际上我们什么都不知道。两种形状必须同处置。
                    (true, view) if old.is_some() && view.is_none_or(<[f32]>::is_empty) => {
                        report.no_baseline_curves += 1;
                        Some(EMPTY_VIEW.as_slice())
                    }
                    (_, view) => view,
                };
                // 证据判据要比的是**权威里的那条曲线**（烘焙前），不是换域后的副本。
                let authority_curve = old;
                if let (Some(old), Some(old_view)) = (old, old_view) {
                    if values == old_view {
                        curves.insert(key, old.clone());
                        continue;
                    }
                }
                if values.len() > 1_000_000
                    || values.iter().any(|v| !v.is_finite() || v.abs() > 10000.)
                {
                    return Err("invalid source parameter samples".into());
                }
                // 被另一条选中片段覆盖的工程区间不属于本次编辑（与旧实现同一条纪律）。
                let protected = timeline
                    .selected_clip_id
                    .as_ref()
                    .and_then(|id| timeline.clips.iter().find(|selected| selected.id == *id))
                    .filter(|selected| {
                        timeline
                            .resolve_root_track_id(&selected.track_id)
                            .as_deref()
                            == Some(root.as_str())
                            && selected.id != clip.id
                    })
                    .map(|selected| (selected.start_sec, selected.start_sec + selected.length_sec));
                // 【曲线永远按源秒存 —— 这里不再有"换域烘焙"】
                //
                // 旧模型里「锁定参数线」决定 `values[i]` 是"源第 i 格"还是"工程第 i 格"，
                // 于是拨开关必须把整条数组搬到另一个域。换域必然有损（工程域里没有窗外的
                // 格），于是必须有闸；有闸就必然有"闸拒绝 ⇒ 开关无效"。用户对它的原话
                // 正是"完全没反应"，而现场日志里 `18026 -> 11824` 就是那次截断。
                //
                // 现在开关只换**投影用的映射**（见 `Placement`）：数组一个字节都不动，
                // 因此可逆、即时、无损。这里只剩一件事 —— 把差分写回**源域**。
                let current = geometry.map()?;
                let (rebuilt, protected_skipped) = capture_source_curve(SourceCapture {
                    existing: old,
                    old_view,
                    values,
                    current: &current,
                    geometry: &geometry,
                    key: &key,
                    frame_ms: params.frame_period_ms,
                    protected,
                })?;
                report.protected_frames += protected_skipped;
                // 内容没变就保留旧对象：避免无谓的代际推进与存档改写。
                //
                // 【为什么先比"投影是否复现这份视图"而不是逐点比曲线】宿主每次几何变化
                // （非网格移动 / 拉伸）都会让 GUI 视图成为源曲线的一次**插值投影**。
                // 若把那份投影当新源数据写回，反复往返会逐次模糊原曲线（实测漂移
                // ~0.03 音分/次）。判据取"旧曲线投影出来就是这份视图 ⇒ 没有新编辑"，
                // 与旧实现同口径。
                let reproduces_view = old.is_some_and(|old| {
                    old.project(
                        &geometry,
                        &key,
                        params.frame_period_ms,
                        &placement,
                        CurveDomain::Project,
                    )
                    .is_ok_and(|projected| {
                        (begin..=end).all(|frame| {
                            projected[frame]
                                == values.get(frame).copied().unwrap_or_else(|| pad(&key))
                        })
                    })
                });
                let curve = match old {
                    Some(old) if reproduces_view || same_curve(old, &rebuilt) => old.clone(),
                    _ => rebuilt,
                };
                // 【R1：I-6 从"计数单调"改成"按帧证据"】
                //
                // 旧判据是一个整数比较（非默认帧数只能增不能减）。注释写着"只有用户能
                // 减少曲线"，但这条路径上**每一次调用都是用户命令** —— 于是这道闸的实际
                // 效果是"任何用户删除都被拒绝"。用户用右键拖拽擦回初始化，就是最直接的
                // 一次"减少非默认帧"：权威被整条拒绝写回，界面先回显、下一次投影再抹掉、
                // 音频一个采样都没变。
                //
                // 现在的判据是**证据**：删掉一帧必须同时满足"当前视图在这一帧是默认值"
                // 与"基线在这一帧不是默认值"。派生写入没有基线 ⇒ 任何减少都不成立 ⇒
                // 仍被拦住；用户擦除每一帧都有这条证据 ⇒ 被接受。
                let unexplained = unexplained_removals(
                    &key,
                    authority_curve,
                    Some(&curve),
                    old_view,
                    values,
                    &current,
                    params.frame_period_ms,
                );
                if unexplained > 0 {
                    let refused_key = key_for_anchor.clone().unwrap_or_default();
                    hifishifter_kernel::log_limiter::emit_limited_key(
                        log::Level::Warn,
                        "ara.atlas.write-refused",
                        format_args!(
                            "[ara] line write refused for key={refused_key} curve={key}: \
                             {unexplained} frame(s) would be erased without evidence (the editor's \
                             baseline does not show them as user data); the stored curve is kept",
                        ),
                    );
                    report.refuse(RefusedWrite {
                        key: refused_key,
                        root: root.clone(),
                        curve: key.clone(),
                        unexplained,
                        first_frame: first_unexplained_frame(
                            &key,
                            authority_curve,
                            Some(&curve),
                            &current,
                            params.frame_period_ms,
                            values,
                        ),
                    });
                    // 只拒绝**这一条曲线**：整份权威保持原样会让同一次捕获里别的曲线
                    // 也被牵连，而那正是旧闸"一条不合、整份不写"的形状。
                    refused_here += 1;
                    if let Some(old) = authority_curve {
                        curves.insert(key, old.clone());
                    }
                    continue;
                }
                if let Some(before) = authority_curve.map(|curve| non_default_values(&key, curve)) {
                    report.removed_frames +=
                        before.saturating_sub(non_default_values(&key, &curve));
                }
                curves.insert(key, curve);
            }
            let mut template = params.clone();
            clear_curves(&mut template);
            // 权威侧：把这份参数写到稳定宿主身份上（这才是被持久化、且与几何无关的一份）。
            //
            // 【为什么锚点沿用已有值】用户在编辑途中切换「锁定参数线」时，既有曲线不应被
            // 重新解释。第一次播种时定下，之后沿用。
            if let Some(key) = key_for_anchor {
                // 【U-5：**永不删除**】"这一轮没看到曲线"有三种成因：身份没跟上、几何不可解、
                // 以及用户真的清空了。旧实现在这里 `remove` —— 前两种会被误判成第三种，
                // 而删除是**不可逆**的数据销毁（权威是唯一被持久化的那一份）。
                // 现在只累加宽限期，真正的回收留给 `reclaim_orphans`（它要求"宿主图里
                // 确实没有这条 modification"且连续多轮成立）。
                if curves.is_empty() {
                    if let Some(existing) = candidate.authorities.get_mut(&key) {
                        existing.unseen_rounds = existing.unseen_rounds.saturating_add(1);
                    }
                } else {
                    // 【I-6 的去处】"不许减少非默认帧"这条整数闸已经移到**逐曲线**的
                    // 按帧证据判据里（见上面 `unexplained_removals` 的调用点）。那里
                    // 拒绝的是"没有证据的那一条曲线"，而不是"整份权威"：
                    //   * 用户擦除每一帧都有证据 ⇒ 写回，音频真的改变；
                    //   * 派生写入没有基线 ⇒ 任何减少都被拒 ⇒ 数据仍然安全。
                    // 到这里 `curves` 已经是"能写的那一部分"，整份装上即可。
                    let before = candidate
                        .authorities
                        .get(&key)
                        .map(|existing| non_default_frames(&existing.curves))
                        .unwrap_or(0);
                    let after = non_default_frames(&curves);
                    let accepted = refused_here == 0;
                    let written = curves.clone();
                    // 【放置策略沿用已有值，第一次播种时取当前设置】
                    //
                    // 【为什么"沿用"而不是"每次都写当前设置"】用户在编辑途中拨一下开关，
                    // 既有曲线不应被重新解释 —— 那正是旧实现里"读用当前设置、写用入库值"
                    // 造成整条曲线偏移的同一个坑。开关本身由
                    // `DocumentSession::sync_placement_with_settings` 一次性改元数据，
                    // 这里只负责"新权威用当前设置播种"。
                    //
                    // 【为什么要补 `frozen`】旧存档的 `anchor: "Project"` 读进来是
                    // `PinProject { frozen: [] }`；第一次拿到活 clip 的几何时补上，
                    // 它此后就真的钉在那一刻了。
                    let placement = match candidate.authorities.get(&key) {
                        Some(existing) => match &existing.placement {
                            Placement::PinProject { frozen } if frozen.is_empty() => {
                                Placement::pinned(geometry.clone())
                            }
                            other => other.clone(),
                        },
                        None => placement.clone(),
                    };
                    let landed_anchor = &placement;
                    let refused_rounds = if accepted {
                        0
                    } else {
                        candidate
                            .authorities
                            .get(&key)
                            .map(|existing| existing.refused_rounds.saturating_add(1))
                            .unwrap_or(1)
                    };
                    candidate.set_authority(
                        &key,
                        AuthorityParameters {
                            edits: BTreeMap::new(),
                            source: String::new(),
                            curves: written,
                            template: template.clone(),
                            placement: placement.clone(),
                            last_edit_seq: 0,
                            // 正在被一条活 region 认领 ⇒ 宽限期清零。
                            unseen_rounds: 0,
                            refused_rounds,
                        },
                    );
                    // 【U-10：每一次写都必须有一行，且必须能回答"少了没有"】
                    //
                    // `before > after` 就是"用户数据被抹平"的直接证据。它出现在这里
                    // （自动捕获路径）而不是只在提交路径上，是因为**自动路径也能删数据**：
                    // 只要某一轮的视图里出现了默认值，`incoming != prior` 就会把它当成
                    // 用户改动写回。I-4 的判据就靠这一行兑现。
                    let level = if accepted {
                        log::Level::Warn
                    } else {
                        log::Level::Error
                    };
                    hifishifter_kernel::log_limiter::emit_limited_key(
                        level,
                        "ara.atlas.line-write",
                        format_args!(
                            "[ara] line write: root={root} key={key} placement={landed_anchor:?} \
                             curves={} frames={}..{} baseline={} before_nondefault={before} \
                             after_nondefault={after} accepted={} refused_curves={refused_here} \
                             refused_rounds={refused_rounds}",
                            curves.len(),
                            begin,
                            end,
                            if baseline_present {
                                "present"
                            } else {
                                "absent"
                            },
                            u8::from(accepted),
                        ),
                    );
                    // 【B3-3：卡住的循环要自己说出来】`line write` 那一行被按 key 限流，
                    // 稳定状态下每 8 条只留 1 条 —— 于是"它一直在失败"看起来像"偶发
                    // 失败"。按 1/2/4/8/16… 递增地补一行汇总，代价有界，而"一直拒绝"
                    // 这个形状再也藏不住。
                    if !accepted && refused_rounds.is_power_of_two() {
                        hifishifter_kernel::log_limiter::emit_limited_key(
                            log::Level::Error,
                            "ara.atlas.refused",
                            format_args!(
                                "[ara] line write refused {refused_rounds} round(s) for key={key}: \
                                 {refused_here} curve(s) would have erased frames without evidence \
                                 ({before} -> {after} non-default frame(s)); those curves keep their \
                                 stored samples, the rest of this write landed",
                            ),
                        );
                    }
                }
            }
            // 派生侧：本帧的 per-clip 投影缓存（供 `project_*` 与渲染输入消费）。
            candidate.regions.insert(
                clip.id.clone(),
                RegionParameters {
                    identity: identity.clone(),
                    root,
                    current: geometry,
                    live: true,
                    template,
                    edits: BTreeMap::new(),
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
    #[cfg(test)]
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
    /// 与 `project` 相同，但**显式给定旧存档的兜底放置策略**。
    ///
    /// 【注意：它不再决定"用哪条映射"】权威曲线按它自己的 `Placement` 解释
    /// （见 `Authority::placement`）；这个参数只在**旧存档的 per-clip 记录**上生效
    /// （那种记录没有放置字段）。测试因此可以直接构造两种几何，而不必改进程级设置。
    #[cfg(test)]
    pub(crate) fn project_with_placement(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
        placement: &Placement,
    ) -> Result<BTreeMap<String, TrackParamsState>, String> {
        self.project_in_domain_with_anchor(timeline, identities, false, placement)
    }
    fn project_in_domain(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
        local: bool,
    ) -> Result<BTreeMap<String, TrackParamsState>, String> {
        self.project_in_domain_with_anchor(timeline, identities, local, &default_placement())
    }
    fn project_in_domain_with_anchor(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
        local: bool,
        fallback: &Placement,
    ) -> Result<BTreeMap<String, TrackParamsState>, String> {
        self.validate()?;
        let mut projected = BTreeMap::new();
        for clip in &timeline.clips {
            if let Some(detail) = self.project_clip(timeline, identities, clip, local, fallback)? {
                projected.insert(clip.id.clone(), detail.params);
            }
        }
        Ok(projected)
    }

    /// 投影一条 clip；只跳过这一条，不让身份/几何的瞬时缺失整份失败。
    fn project_clip(
        &self,
        timeline: &TimelineState,
        identities: &BTreeMap<String, RegionIdentity>,
        clip: &hifishifter_kernel::state::Clip,
        local: bool,
        fallback: &Placement,
    ) -> Result<Option<ProjectedClip>, String> {
        let Some(identity) = identities.get(&clip.id) else {
            return Ok(None);
        };
        let Some(root) = timeline.resolve_root_track_id(&clip.track_id) else {
            return Ok(None);
        };
        let Ok(geometry) = geometry(clip) else {
            return Ok(None);
        };
        let Some(authority) = self.authority(identity, &root, &geometry)? else {
            return Ok(None);
        };
        let placement = authority.placement(fallback).clone();
        if authority.edits().is_some() {
            self.project_sparse_clip(
                timeline,
                clip,
                identity,
                &root,
                &geometry,
                &placement,
                authority.template(),
                authority.edits().unwrap(),
                local,
            )
        } else {
            self.project_legacy_clip(clip, identity, &geometry, &placement, authority, local)
        }
    }

    fn project_sparse_clip(
        &self,
        timeline: &TimelineState,
        _clip: &hifishifter_kernel::state::Clip,
        _identity: &RegionIdentity,
        root: &str,
        geometry: &RegionGeometry,
        placement: &Placement,
        template: &TrackParamsState,
        edits: &BTreeMap<String, EditCurve>,
        local: bool,
    ) -> Result<Option<ProjectedClip>, String> {
        let baseline = timeline
            .params_by_root_track
            .get(root)
            .cloned()
            .unwrap_or_else(|| template.clone());
        let frame_ms = if baseline.frame_period_ms.is_finite() && baseline.frame_period_ms > 0.0 {
            baseline.frame_period_ms
        } else {
            template.frame_period_ms
        };
        let (begin, end) = if local {
            (
                0,
                ((geometry.project_duration * 1000. / frame_ms).ceil() as usize).max(1),
            )
        } else {
            frame_range(geometry, frame_ms)?
        };
        let count = end.saturating_add(1);
        let baseline_ready = !baseline.pitch_orig.is_empty() || !baseline.pitch_edit.is_empty();
        let mut params = baseline;
        params.frame_period_ms = frame_ms;
        params.pitch_orig_key = None;
        params.dyn_orig_key = None;
        params.dyn_orig_source_key = None;
        params.pending_pitch_offset = None;
        // 派生原始值只从当前 baseline 读；GUI 根视图保留绝对工程帧，渲染局部
        // 输出则把它切成 clip 局部数组。权威永远不写这两条曲线。
        if local {
            params.pitch_orig = sample_baseline_slice(
                &params.pitch_orig,
                "pitch_edit",
                geometry,
                frame_ms,
                begin,
                count,
            );
            params.dyn_orig = sample_baseline_slice(
                &params.dyn_orig,
                "dyn_orig",
                geometry,
                frame_ms,
                begin,
                count,
            );
            params.tension_orig = sample_baseline_slice(
                &params.tension_orig,
                "tension_orig",
                geometry,
                frame_ms,
                begin,
                count,
            );
        }

        let pitch_defaults =
            pitch_baseline_defaults(&params, geometry, frame_ms, begin, count, local);
        let edit_pitch = edits.get("pitch_edit");
        let (pitch_values, pitch_mask) =
            compose_curve("pitch_edit", &pitch_defaults, edit_pitch, |frame| {
                source_frame_for_output(geometry, placement, frame_ms, frame, local)
            });
        params.pitch_edit = pitch_values;
        params.pitch_edit_user_modified = edits
            .get("pitch_edit")
            .is_some_and(|curve| !curve.is_empty());

        // 未编辑的 tension / extras 必须是默认值，而不是上一轮投影留下的旧数组。
        let tension_defaults = vec![0.0; count];
        let (tension_values, tension_mask) = compose_curve(
            "tension_edit",
            &tension_defaults,
            edits.get("tension_edit"),
            |frame| source_frame_for_output(geometry, placement, frame_ms, frame, local),
        );
        params.tension_edit = tension_values;
        params.extra_curves.clear();
        let mut edited_frames = BTreeMap::new();
        if !mask_is_empty(&pitch_mask) {
            edited_frames.insert("pitch_edit".to_owned(), mask_ranges(&pitch_mask));
        }
        if !mask_is_empty(&tension_mask) {
            edited_frames.insert("tension_edit".to_owned(), mask_ranges(&tension_mask));
        }
        for (key, edit) in edits {
            let Some(name) = key.strip_prefix("extra:") else {
                continue;
            };
            if edit.is_empty() {
                continue;
            }
            let defaults = vec![pad(key); count];
            let (values, mask) = compose_curve(key, &defaults, Some(edit), |frame| {
                source_frame_for_output(geometry, placement, frame_ms, frame, local)
            });
            if !mask_is_empty(&mask) {
                edited_frames.insert(key.clone(), mask_ranges(&mask));
            }
            params.extra_curves.insert(name.to_owned(), values);
        }
        // 即使没有 extra 编辑，也把 dyn 的哨兵显式写出来：GUI 上“没有 dyn 编辑”
        // 必须显示为跟随原声，而不是沿用上一轮的旧数组。
        if edits.get("extra:dyn").is_none() {
            params
                .extra_curves
                .insert("dyn".to_owned(), vec![-1.0; count]);
        }
        Ok(Some(ProjectedClip {
            params,
            edited_frames,
            baseline_ready,
        }))
    }

    fn project_legacy_clip(
        &self,
        clip: &hifishifter_kernel::state::Clip,
        identity: &RegionIdentity,
        geometry: &RegionGeometry,
        placement: &Placement,
        authority: Authority<'_>,
        local: bool,
    ) -> Result<Option<ProjectedClip>, String> {
        let mut params = authority.template().clone();
        let frame_ms = params.frame_period_ms;
        let missed: Vec<&str> = authority
            .curves()
            .iter()
            .filter(|(_key, curve)| {
                let Some((lo, hi)) = curve.source_extent() else {
                    return false;
                };
                let window = (
                    geometry.source_start,
                    geometry.source_start + geometry.source_duration,
                );
                window.1 < lo - 1e-6 || window.0 > hi + 1e-6
            })
            .map(|(key, _)| key.as_str())
            .collect();
        if !missed.is_empty() {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.render.split-source-window",
                format_args!(
                    "[ara] clip {} {} placement={:?}: its source window [{:.6}, {:.6}] does not \
                     intersect the authority curve cover(s) for {}; those curves project as \
                     defaults this round",
                    clip.id,
                    identity.modification,
                    placement,
                    geometry.source_start,
                    geometry.source_start + geometry.source_duration,
                    missed.join(", "),
                ),
            );
        }
        for (key, curve) in authority.curves() {
            let out_domain = if local {
                CurveDomain::ClipLocal
            } else {
                CurveDomain::Project
            };
            set_curve(
                &mut params,
                key,
                curve.project(geometry, key, frame_ms, placement, out_domain)?,
            );
        }
        params.pitch_orig_key = None;
        params.dyn_orig_key = None;
        params.pitch_edit_user_modified = authority.has_pitch_curve();
        Ok(Some(ProjectedClip {
            params,
            edited_frames: BTreeMap::new(),
            baseline_ready: true,
        }))
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
        if self.authorities.len() > 16384 {
            return Err("parameter atlas authority budget exceeded".into());
        }
        for params in self.authorities.values() {
            if !params.template.frame_period_ms.is_finite()
                || params.template.frame_period_ms < 0.1
                || params.template.frame_period_ms > 1000.
            {
                return Err("invalid authority parameter template".into());
            }
            for (key, curve) in &params.edits {
                if key.is_empty()
                    || !curve.frame_ms.is_finite()
                    || !(0.1..=1000.).contains(&curve.frame_ms)
                    || curve.runs.len() > 16384
                {
                    return Err("invalid authority edit curve".into());
                }
                let mut last = 0_usize;
                for run in &curve.runs {
                    let end = run
                        .first_frame
                        .checked_add(run.values.len())
                        .ok_or("authority edit frame overflow")?;
                    if run.values.is_empty()
                        || run.first_frame < last
                        || end > 1_000_000
                        || run
                            .values
                            .iter()
                            .any(|value| !value.is_finite() || value.abs() > 10000.)
                    {
                        return Err("invalid authority edit samples".into());
                    }
                    last = end;
                }
            }
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
        // 源域曲线：权威表与派生表**共享 Arc**（`ModificationParameters::from_record`
        // 与 `derive_regions` 都只克隆 `Arc`，不复制采样），因此按指针去重 —— 否则同一份
        // 采样会被计两次，64MiB 上限会在正常工程上误报。
        let mut seen = std::collections::BTreeSet::new();
        for params in self.authorities.values() {
            Self::validate_source_curves(&params.curves, &mut bytes, &mut seen)?;
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
            Self::validate_source_curves(&record.curves, &mut bytes, &mut seen)?;
        }
        Ok(())
    }

    /// 复验一组源域曲线，并按 `Arc` 指针去重累计字节。
    fn validate_source_curves(
        curves: &BTreeMap<String, SourceCurve>,
        bytes: &mut usize,
        seen: &mut std::collections::BTreeSet<usize>,
    ) -> Result<(), String> {
        for curve in curves.values() {
            curve.validate()?;
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
            if seen.insert(Arc::as_ptr(&curve.values) as usize) {
                *bytes = bytes
                    .checked_add(curve.values.len() * 4)
                    .ok_or("parameter atlas byte overflow")?;
                if *bytes > 64 * 1024 * 1024 {
                    return Err("parameter atlas exceeds 64MiB".into());
                }
            }
        }
        Ok(())
    }

    /// 只捕获 clip 外的用户编辑；被移动 clip 遮住的旧空白编辑保持隐藏，移开后仍存在。
    fn capture_gaps_sparse(
        &mut self,
        timeline: &TimelineState,
        previous: Option<&BTreeMap<String, TrackParamsState>>,
    ) -> Result<(), String> {
        let mut coverage = BTreeMap::<String, Vec<RegionGeometry>>::new();
        for clip in &timeline.clips {
            let root = timeline
                .resolve_root_track_id(&clip.track_id)
                .ok_or("unknown gap parameter root")?;
            coverage.entry(root).or_default().push(RegionGeometry {
                project_start: clip.start_sec,
                project_duration: clip.length_sec,
                source_start: clip.source_start_sec,
                source_duration: clip.length_sec * clip.playback_rate as f64,
                reversed: clip.reversed && !clip.loop_enabled,
            });
        }
        self.capture_gap_roots_sparse(&timeline.params_by_root_track, &coverage, previous)
    }

    #[allow(dead_code)]
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
        self.capture_gap_roots_sparse(params, &coverage, None)?;
        self.validate()
    }
    fn capture_gap_roots_sparse(
        &mut self,
        params_by_root: &BTreeMap<String, TrackParamsState>,
        coverage: &BTreeMap<String, Vec<RegionGeometry>>,
        previous: Option<&BTreeMap<String, TrackParamsState>>,
    ) -> Result<(), String> {
        for (root, params) in params_by_root {
            let frame_ms = params.frame_period_ms;
            let mut template = params.clone();
            clear_curves(&mut template);
            template.pitch_edit_user_modified = false;
            let old = self.gaps.get(root).cloned().unwrap_or(GapParameters {
                template: template.clone(),
                curves: BTreeMap::new(),
            });
            let desired = extract_edit_curves(params);
            let prior_edits = previous
                .and_then(|previous| previous.get(root))
                .filter(|prior| prior.frame_period_ms == frame_ms)
                .map(extract_edit_curves);
            let mut count = existing_gap_count(&old, frame_ms)?;
            for values in params.extra_curves.values() {
                count = count.max(values.len());
            }
            count = count
                .max(params.pitch_edit.len())
                .max(params.tension_edit.len());
            if count > 1_000_000 {
                return Err("gap parameter frame budget exceeded".into());
            }
            let mut owned = vec![false; count];
            for geometry in coverage.get(root).into_iter().flatten() {
                if let Some((begin, end)) = covered_frame_range(geometry, frame_ms)? {
                    if begin < count {
                        owned[begin..=end.min(count - 1)].fill(true);
                    }
                }
            }
            let mut keys = std::collections::BTreeSet::new();
            keys.extend(old.curves.keys().cloned());
            keys.extend(desired.curves.keys().cloned());
            if let Some(prior) = &prior_edits {
                keys.extend(prior.curves.keys().cloned());
            }
            let mut curves = BTreeMap::new();
            for key in keys {
                let (mut values, mut valid) = old
                    .curves
                    .get(&key)
                    .map(|curve| dense_gap(curve, &key, frame_ms, count))
                    .transpose()?
                    .unwrap_or_else(|| (vec![pad(&key); count], vec![false; count]));
                if values.len() < count {
                    values.resize(count, pad(&key));
                    valid.resize(count, false);
                }
                if let Some(prior) = &prior_edits {
                    if let Some(prior_curve) = prior.get(&key) {
                        for frame in 0..count {
                            if owned[frame] || !valid[frame] {
                                continue;
                            }
                            if prior_curve.value_at(frame as f64, &key).is_some()
                                && desired.get(&key).is_none_or(|curve| {
                                    curve.value_at(frame as f64, &key).is_none()
                                })
                            {
                                valid[frame] = false;
                            }
                        }
                    }
                }
                if let Some(current) = desired.get(&key) {
                    for (frame, value) in current.frame_values() {
                        if frame < count && !owned[frame] {
                            values[frame] = value;
                            valid[frame] = true;
                        }
                    }
                }
                let mut spans = Vec::new();
                let mut frame = 0;
                while frame < count {
                    if !valid[frame] {
                        frame += 1;
                        continue;
                    }
                    let first = frame;
                    while frame < count && valid[frame] {
                        frame += 1;
                    }
                    let reservation = crate::render::budget::global_budget()
                        .reserve((frame - first) * 4)
                        .ok_or("gap parameter memory budget exceeded")?;
                    spans.push(GapSpan {
                        first_frame: first,
                        values: Arc::new(values[first..frame].to_vec()),
                        reservation: Some(Arc::new(reservation)),
                    });
                }
                if spans.is_empty() {
                    continue;
                }
                curves.insert(key, GapCurve { frame_ms, spans });
            }
            self.gaps
                .insert(root.clone(), GapParameters { template, curves });
        }
        Ok(())
    }

    #[allow(dead_code)]
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
/// 把「锁定参数线」的取值映射成**新权威**的放置策略。
///
/// 【语义】`true`（默认）= 锁定：参数跟着**素材**走。`false` = 不锁定：
/// 参数留在**工程位置**，宿主移动片段不会带走曲线。
///
/// 【为什么新权威才用它】既有权威的放置策略由
/// `ParameterAtlas::set_placement` 一次性改写（纯元数据）。这里只回答"第一次播种
/// 一条权威时该用哪个策略"。
fn placement_for_lock(locked: bool) -> Placement {
    if locked {
        Placement::FollowMaterial
    } else {
        Placement::PinProject { frozen: Vec::new() }
    }
}

/// 新权威在建立时采用的放置策略，由用户设置「锁定参数线」决定。
#[cfg(not(test))]
fn default_placement() -> Placement {
    placement_for_lock(crate::settings_store::lock_param_lines())
}

/// 当前设置对应的放置策略；**测试里返回 `None`**（"不要动"）。
///
/// 【为什么测试里是 `None` 而不是 `FollowMaterial`】进程级设置是全局状态，而测试
/// 并行运行 —— 一个改设置的用例会让另一个正在投影的用例看到不同的放置策略。
/// 机制本身由 `set_placement` 的用例覆盖；这里只负责"生产路径从设置取值"。
#[cfg(not(test))]
pub(crate) fn placement_from_settings() -> Option<Placement> {
    Some(placement_for_lock(crate::settings_store::lock_param_lines()))
}

/// 测试里不改任何权威（见上一条）。
#[cfg(test)]
pub(crate) fn placement_from_settings() -> Option<Placement> {
    None
}

/// 测试里固定为"跟随素材"。
///
/// 【为什么不读进程级设置】那是**全局**状态，而测试并行运行 —— 一个改设置的测试
/// 会让另一个正在捕获的测试拿到"钉住"。放置行为本身由 `placement_for_lock` 与
/// `the_lock_setting_selects_whether_curves_follow_the_material_or_the_project` 覆盖。
#[cfg(test)]
fn default_placement() -> Placement {
    placement_for_lock(true)
}

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

fn existing_gap_count(gaps: &GapParameters, frame_ms: f64) -> Result<usize, String> {
    if !frame_ms.is_finite() || !(0.1..=1000.).contains(&frame_ms) {
        return Err("invalid gap frame period".into());
    }
    let mut count = 0_usize;
    for curve in gaps.curves.values() {
        for span in &curve.spans {
            let end = ((span.first_frame + span.values.len() - 1) as f64 * curve.frame_ms
                / frame_ms)
                .ceil() as usize;
            count = count.max(end.saturating_add(1));
        }
    }
    if count > 1_000_000 {
        return Err("gap parameter frame budget exceeded".into());
    }
    Ok(count)
}

fn gap_ranges(curve: &GapCurve, frame_ms: f64) -> Result<Vec<std::ops::Range<usize>>, String> {
    if !frame_ms.is_finite() || !(0.1..=1000.).contains(&frame_ms) {
        return Err("invalid gap frame period".into());
    }
    let mut ranges = Vec::new();
    for span in &curve.spans {
        let begin = (span.first_frame as f64 * curve.frame_ms / frame_ms).ceil() as usize;
        let end = ((span.first_frame + span.values.len() - 1) as f64 * curve.frame_ms / frame_ms)
            .floor() as usize;
        if begin <= end {
            push_frame_range(&mut ranges, begin..end + 1);
        }
    }
    Ok(ranges)
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
    default_pad_for_key(key)
}
/// 图集承载的曲线集合。
///
/// 【为什么 `pitch_orig` 仍在这里】它是分析基线；渲染侧的**主**来源已改为内容寻址的
/// 逐 clip 缓存（见 `mixdown::apply_host_pitch_edit`），因此它不再是渲染的关键路径。
/// 但投影契约（`project_roots` / `capture_inner` / 旧存档迁移）与合成夹具仍把它当作
/// 一份可往返的曲线 —— 删掉会让"内容缓存尚未建立"的窗口里无 F0 可用。因此本轮保留，
/// 只把渲染侧改成"内容缓存优先、工程域曲线兜底"。
/// 把一批值里所有非默认帧转成一条稀疏编辑曲线（测试种子 / 旧记录迁移共用）。
#[cfg(test)]
fn edit_curve_from_values(
    values: &[f32],
    first_frame: usize,
    frame_ms: f64,
    key: &str,
) -> Option<EditCurve> {
    let mut curve = EditCurve::new(frame_ms);
    let mut start: Option<usize> = None;
    for (offset, value) in values.iter().copied().enumerate() {
        let hit = is_edited_value(key, value, None, Some(default_pad_for_key(key)));
        match (start, hit) {
            (None, true) => start = Some(offset),
            (Some(begin), false) => {
                curve
                    .runs
                    .push(hifishifter_kernel::editor::param_edit::EditRun {
                        first_frame: first_frame + begin,
                        values: values[begin..offset].to_vec(),
                    });
                start = None;
            }
            _ => {}
        }
    }
    if let Some(begin) = start {
        curve
            .runs
            .push(hifishifter_kernel::editor::param_edit::EditRun {
                first_frame: first_frame + begin,
                values: values[begin..].to_vec(),
            });
    }
    (!curve.is_empty()).then_some(curve)
}

/// 旧存档的整条曲线 -> 稀疏编辑。pitch 优先用同一份旧曲线里的 pitch_orig 做差；
/// 没有 orig 时只能保守地把正 pitch 当编辑（v8 存档的迁移日志会说明这一点）。
fn edits_from_source_curves(
    curves: &BTreeMap<String, SourceCurve>,
    frame_ms: f64,
) -> BTreeMap<String, EditCurve> {
    let mut edits = BTreeMap::new();
    let pitch_orig = curves.get("pitch_orig");
    for (key, curve) in curves {
        if matches!(key.as_str(), "pitch_orig" | "dyn_orig" | "tension_orig") {
            continue;
        }
        let mut out = EditCurve::new(frame_ms);
        let mut start: Option<usize> = None;
        for (index, value) in curve.values.iter().copied().enumerate() {
            let frame = curve.first_frame + index;
            let orig = if key == "pitch_edit" {
                pitch_orig.and_then(|orig| orig.values.get(index).copied())
            } else {
                None
            };
            let hit = is_edited_value(key, value, orig, Some(default_pad_for_key(key)));
            match (start, hit) {
                (None, true) => start = Some(frame),
                (Some(begin), false) => {
                    let begin_offset = begin.saturating_sub(curve.first_frame);
                    let end_offset = frame.saturating_sub(curve.first_frame);
                    out.runs
                        .push(hifishifter_kernel::editor::param_edit::EditRun {
                            first_frame: begin,
                            values: curve.values[begin_offset..end_offset].to_vec(),
                        });
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(begin) = start {
            let begin_offset = begin.saturating_sub(curve.first_frame);
            out.runs
                .push(hifishifter_kernel::editor::param_edit::EditRun {
                    first_frame: begin,
                    values: curve.values[begin_offset..].to_vec(),
                });
        }
        if !out.is_empty() {
            edits.insert(key.clone(), out);
        }
    }
    edits
}

/// GUI/分组可编辑的曲线集合（不含分析派生数组）。
pub(super) fn editable_parameter_curves(params: &TrackParamsState) -> Vec<(String, &[f32])> {
    let mut curves = vec![
        ("pitch_edit".into(), params.pitch_edit.as_slice()),
        ("tension_edit".into(), params.tension_edit.as_slice()),
    ];
    curves.extend(
        params
            .extra_curves
            .iter()
            .map(|(key, values)| (format!("extra:{key}"), values.as_slice())),
    );
    curves
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
    params.pitch_edit_user_modified = false;
}

/// 根视图重算前清掉所有用户可编辑曲线；派生原始值保留。
fn clear_user_curves(params: &mut TrackParamsState) {
    params.pitch_edit.clear();
    params.tension_edit.clear();
    params.extra_curves.clear();
    params.pitch_edit_user_modified = false;
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
    ///
    /// 【`out_domain` 声明返回数组的索引域】三条路径消费的不是同一件事：
    ///   * `CurveDomain::Project` —— 根视图（GUI）：数组按**绝对工程帧**索引；
    ///   * `CurveDomain::ClipLocal` —— 渲染输入：数组按 **clip 局部零点**索引，
    ///     但采样坐标仍用该 clip 在工程里的**真实**位置（见下）。
    ///
    /// 【为什么局部投影不能把 `project_start` 一起归零】早先 `project_local` 直接把
    /// `geometry.project_start = 0.`，于是"绝对工程时刻"这个坐标被**抹掉**。源域曲线
    /// 下碰巧还对（映射的左端也跟着变成 0，等价）；但只要曲线按绝对工程帧存
    /// （「锁定参数线」关闭），右半段就会用局部 0..1 去读绝对 1..2 的帧
    /// ⇒ 读到**左半段**的值。偏差恰好一个分割长度，正是用户报障的
    /// "分割位置以后的参数线整体偏移"。所以：**输出索引归零，采样坐标不归零。**
    ///
    /// 【`placement` 与 `domain` 各自决定什么】这是两个**不同**的决定，缺一不可：
    ///   * `placement`（「锁定参数线」的当前值）决定**用哪条映射**：跟随素材 ⇒ 用
    ///     当前几何把工程时刻折成源秒再按源帧读数；钉住 ⇒ 用**冻结几何**做同一次
    ///     折算，于是曲线留在工程位置。这就是用户看得见的那个开关。
    ///   * `domain`（返回数组的索引域）决定**返回数组按什么索引**：`ClipLocal`
    ///     从 0 起，`Project` 按绝对工程帧。它只影响下游怎么用这份数组，
    ///     不影响读到哪些值。
    ///
    /// 【为什么存储域不再是变量】曲线**永远按源秒存**。早先"不锁定 ⇒ 按工程帧存"
    /// 让同一个数组在两个域之间搬来搬去（`bake_curve_to_anchor`），搬动必然有损、
    /// 有损就必须有闸、有闸就必然出现"闸拒绝 ⇒ 开关无效"。现在开关只换**映射**，
    /// 一个字节都不动，因此可逆、即时、无损。
    #[allow(clippy::too_many_arguments)]
    fn project(
        &self,
        geometry: &RegionGeometry,
        key: &str,
        frame_ms: f64,
        placement: &Placement,
        out_domain: CurveDomain,
    ) -> Result<Vec<f32>, String> {
        let (begin, end) = match out_domain {
            // 局部零点索引：`values[i]` = 本 clip 第 `i·frame_ms` 处。从 0 起算，
            // 与渲染路径 `mix_interleaved` 把 `clip.start_sec` 置 0 对齐。
            CurveDomain::ClipLocal => (
                0,
                (geometry.project_duration * 1000. / frame_ms).ceil() as usize,
            ),
            // 绝对工程帧索引：`values[i]` = 项目第 `i·frame_ms` 处。根视图与
            // 捕获侧的对照投影都用它，行为与域显式化之前完全一致。
            _ => frame_range(geometry, frame_ms)?,
        };
        // **放置策略决定用哪条映射**。两条映射都从源秒出发，差别只在"用哪份几何
        // 把工程时刻折成源秒"：
        //   * 跟随素材 ⇒ 当前几何 ⇒ 片段移动 / 拉伸后仍读到同一段源；
        //   * 钉住 ⇒ 冻结几何 ⇒ 片段移走后新位置落在冻结几何之外（读出默认值），
        //     原位置仍留着旧值。
        //
        // 【为什么读和写必须共用同一个判断】早先读用当前设置、写用入库时定下的值。
        // 用户在编辑途中拨一下开关，两者对"这份数组按什么时间索引"的判断就相反 ——
        // 于是任何一次自动应用都会把曲线按错误的映射读一遍、再按"用户改过"写回。
        let current = placement.map(geometry)?;
        let mut result = vec![pad(key); end + 1];
        for frame in begin..=end {
            // 【为什么局部域要加回真实起点】输出索引是局部的，但坐标不是：
            // 局部帧 `f` 对应工程时刻 `project_start + f·frame_ms`。少了这一项，
            // 工程域锚定（曲线按绝对工程帧存）时右半段会用局部 0..1 去读绝对
            // 1..2 的帧 ⇒ 读到左半段的值，整体偏移一个分割长度。
            // 绝对域下 `frame` 本来就是绝对帧号，加 `project_start` 会双重计数，
            // 所以只在局部域加。
            let project = match out_domain {
                CurveDomain::ClipLocal => {
                    (geometry.project_start + frame as f64 * frame_ms / 1000.).clamp(
                        geometry.project_start,
                        geometry.project_start + geometry.project_duration,
                    )
                }
                _ => (frame as f64 * frame_ms / 1000.).clamp(
                    geometry.project_start,
                    geometry.project_start + geometry.project_duration,
                ),
            };
            let Some(domain_time) = self.domain_time(&current, project) else {
                continue;
            };
            result[frame] = self
                .sample_indexed(domain_time, key, frame_ms)
                .unwrap_or_else(|| pad(key));
        }
        Ok(result)
    }

    /// 取本曲线在**索引域**某一时刻的值（线性插值；界外返回 `None`）。
    ///
    /// 【为什么按域换算帧号】曲线的格号 `first_frame` 是**它自己那个索引域**上的
    /// 帧号：源域曲线的 `first_frame` 是源帧，工程域曲线的是绝对工程帧。
    /// 而 `domain_time` 给出来的是**秒**。源域下 `秒 × 1000/fp` 恰好就是源帧号，
    /// 可以直接比；工程域下必须再减去**工程零点**才是"从这条曲线起点数起"的帧号 ——
    /// 少了这一步，工程域曲线（「锁定参数线」关闭）会被整体偏移一个
    /// `project_start`，用户在分割处看到的正是这种偏移。
    fn sample_indexed(&self, domain_time: f64, key: &str, _frame_ms: f64) -> Option<f32> {
        // 【为什么减 `first_frame`】曲线的源轴/项目轴都从**该曲线的窗口起点**起算
        // （源锚定下 `geometry.map()` 的源轴从 `source_start` 起，工程锚定下
        // `project_map()` 两条轴都从 `project_start` 起），而 `first_frame`
        // 正是"`values[0]` 在那个轴上的帧号"。不区分两种域、统一减去它，
        // 得到的下标就是"从这条曲线第 0 帧数起"的偏移 —— 与 `sample_at` 同义。
        let index = domain_time * 1000. / self.frame_ms - self.first_frame as f64;
        if !index.is_finite() || index < 0. || self.values.is_empty() {
            return None;
        }
        let last = (self.values.len() - 1) as f64;
        if index > last {
            return None;
        }
        let lo = (index.floor() as usize).min(self.values.len() - 1);
        let hi = (lo + 1).min(self.values.len() - 1);
        let fraction = (index - lo as f64).clamp(0., 1.) as f32;
        let a = self.values[lo];
        let b = self.values[hi];
        Some(if key == "pitch_edit" && (a <= 0. || b <= 0.) {
            a.max(b).max(0.)
        } else {
            a + (b - a) * fraction
        })
    }

    /// 把当前 region 的**工程时刻**映射到本曲线的**索引域**时刻。
    ///
    /// 【为什么按 `domain` 而不是 `basis` 分叉】`basis` 是第七轮留下的旧形状标记，
    /// 把当前 region 的**工程时刻**映射成"喂给 `sample_at` 的索引域时刻"。
    ///
    /// 【为什么只按 `basis` 分叉】这里只回答"曲线自己按哪种时间索引"这一个事实：
    ///   * `basis == None`（第七轮起的形状）⇒ 索引域就是**源秒**，所以把工程时刻
    ///     经当前映射折成源秒。当前映射取哪条由**锚点**决定（见 `project`）：
    ///     锁定 ⇒ 真折成源秒（跟着素材走）；不锁定 ⇒ `project_map` 是恒等映射，
    ///     "折过去"等于原地不动（留在工程位置）。**两条锚点共用这一行代码**，
    ///     差别全在映射本身 —— 这正是开关能生效的机制。
    ///   * `basis == Some(..)`（旧存档）⇒ 曲线按**那条 region 的工程帧**索引，
    ///     必须先把工程秒折回源秒、再经旧基映射回当时的工程帧。旧分支保留
    ///     ⇒ 升级零损失。
    fn domain_time(&self, current: &SourceTimeMap, project_sec: f64) -> Option<f64> {
        match &self.basis {
            // 旧形状：先把工程秒折回源秒，再经旧基映射回那条 region 的工程帧。
            Some(basis) => current.previous_project_time(&basis.map().ok()?, project_sec),
            // 新形状：索引域即源秒，由调用方选定的映射（= 锚点）完成折算。
            None => current.project_to_source(project_sec),
        }
    }

    /// 取本曲线在**索引域**某一时刻的值（线性插值；界外返回 `None`）。
    fn sample_at(&self, domain_time: f64, key: &str) -> Option<f32> {
        let index = domain_time * 1000. / self.frame_ms - self.first_frame as f64;
        if !index.is_finite() || index < 0. || self.values.is_empty() {
            return None;
        }
        let last = (self.values.len() - 1) as f64;
        if index > last {
            return None;
        }
        let lo = (index.floor() as usize).min(self.values.len() - 1);
        let hi = (lo + 1).min(self.values.len() - 1);
        let fraction = (index - lo as f64).clamp(0., 1.) as f32;
        let a = self.values[lo];
        let b = self.values[hi];
        Some(if key == "pitch_edit" && (a <= 0. || b <= 0.) {
            a.max(b).max(0.)
        } else {
            a + (b - a) * fraction
        })
    }

    /// 取本曲线在**源秒**处的值（新形状直接查；旧形状先经它自己的基映射回源）。
    fn value_at_source(&self, source_sec: f64, key: &str) -> Option<f32> {
        let domain_time = match &self.basis {
            None => source_sec,
            Some(basis) => basis.map().ok()?.source_to_project(source_sec)?,
        };
        self.sample_at(domain_time, key)
    }

    /// 本曲线在**源域**上覆盖的秒区间（含端点）。空曲线返回 `None`。
    fn source_extent(&self) -> Option<(f64, f64)> {
        if self.values.is_empty() {
            return None;
        }
        let span = self.values.len() - 1;
        let lo = self.first_frame as f64 * self.frame_ms / 1000.;
        let hi = (self.first_frame + span) as f64 * self.frame_ms / 1000.;
        // 【工程域曲线没有"源区间"】它的格号是绝对工程帧，把它当源秒用等于两种单位
        // 相减。返回 `None` 让调用方（诊断）跳过它，而不是打一条假的告警。
        if self.basis.is_none() && self.domain == CurveDomain::Project {
            return None;
        }
        match &self.basis {
            None => Some((lo, hi)),
            Some(basis) => {
                let map = basis.map().ok()?;
                let a = map.project_to_source(lo)?;
                let b = map.project_to_source(hi)?;
                Some((a.min(b), a.max(b)))
            }
        }
    }

    /// 复验索引域与采样（`validate` 的曲线侧）。
    fn validate(&self) -> Result<(), String> {
        match &self.basis {
            Some(basis) => {
                basis.map()?;
                frame_range(basis, self.frame_ms)?;
            }
            None => {
                if !self.frame_ms.is_finite() || !(0.1..=1000.).contains(&self.frame_ms) {
                    return Err("invalid source parameter frame domain".into());
                }
            }
        }
        Ok(())
    }
}

/// 把一条 **project 帧索引**的数组（GUI 视图 / 旧曲线投影）在分数帧处线性取样。
///
/// 【为什么需要分数帧】rate == 1 时源格点恰好落在整数工程帧上（无需插值）；
/// 带拉伸的 region 上源格点落在分数帧，必须插值才能把用户画的那条曲线读准。
fn sample_project_frames(values: &[f32], frame: f64, key: &str) -> f32 {
    if values.is_empty() || !frame.is_finite() || frame < 0. {
        return pad(key);
    }
    let lo = frame.floor() as usize;
    if lo >= values.len() {
        return pad(key);
    }
    let hi = (lo + 1).min(values.len() - 1);
    let fraction = (frame - lo as f64).clamp(0., 1.) as f32;
    let a = values[lo];
    let b = values[hi];
    if key == "pitch_edit" && (a <= 0. || b <= 0.) {
        a.max(b).max(0.)
    } else {
        a + (b - a) * fraction
    }
}

/// 把一份**工程帧索引**的编辑曲线折算到源帧（捕获用）。
fn project_curve_to_source(
    curve: &EditCurve,
    key: &str,
    geometry: &RegionGeometry,
    frame_ms: f64,
    protected: Option<(f64, f64)>,
) -> Result<EditCurve, String> {
    let map = geometry.map()?;
    let step = frame_ms / 1000.0;
    let project_end = geometry.project_start + geometry.project_duration;
    let source_lo = geometry
        .source_start
        .min(geometry.source_start + geometry.source_duration);
    let source_hi = geometry
        .source_start
        .max(geometry.source_start + geometry.source_duration);
    let first = (source_lo / step).floor().max(0.0) as usize;
    let last = (source_hi / step).ceil().max(0.0) as usize;
    if last >= 1_000_000 || last < first {
        return Err("source parameter projection frame budget exceeded".into());
    }
    // **按源网格逐帧采样**，而不是把工程帧逐个 round 到源帧：拉伸时两者
    // 不必整除，逐个 round 会在源域留下空帧，也会把斜坡变成平台。
    let mut values = std::collections::BTreeMap::new();
    for source_frame in first..=last {
        let source_sec = source_frame as f64 * step;
        let clamped = source_sec.clamp(source_lo, source_hi);
        let Some(project_sec) = map
            .source_to_project(clamped)
            .or_else(|| ((clamped - source_hi).abs() < 1e-9).then_some(project_end))
        else {
            continue;
        };
        if project_sec < geometry.project_start - 1e-9 || project_sec > project_end + 1e-9 {
            continue;
        }
        if protected.is_some_and(|(lo, hi)| project_sec >= lo && project_sec <= hi) {
            continue;
        }
        let project_frame = project_sec * 1000.0 / frame_ms;
        if let Some(value) = curve.value_at(project_frame, key) {
            values.insert(source_frame, value);
        }
    }
    Ok(EditCurve::from_frame_values(frame_ms, key, values))
}

fn mask_is_empty(mask: &[bool]) -> bool {
    !mask.iter().any(|edited| *edited)
}

/// 把当前根 baseline 的派生数组切成 clip 局部数组（只用于渲染输入）。
fn sample_baseline_slice(
    values: &[f32],
    key: &str,
    geometry: &RegionGeometry,
    frame_ms: f64,
    begin: usize,
    count: usize,
) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(count);
    for offset in 0..count {
        let project_sec = geometry.project_start + (begin + offset) as f64 * frame_ms / 1000.0;
        out.push(sample_project_frames(
            values,
            project_sec * 1000.0 / frame_ms,
            key,
        ));
    }
    out
}

/// pitch_edit 未编辑帧的默认值：当前分析出的 pitch_orig；分析尚未就绪时保留
/// 上一份显示值，绝不把整条线写为零。
fn pitch_baseline_defaults(
    params: &TrackParamsState,
    geometry: &RegionGeometry,
    frame_ms: f64,
    begin: usize,
    count: usize,
    local: bool,
) -> Vec<f32> {
    let source = if !params.pitch_orig.is_empty() {
        params.pitch_orig.as_slice()
    } else {
        params.pitch_edit.as_slice()
    };
    if source.is_empty() {
        return vec![0.0; count];
    }
    // 输出数组的索引域不同：local 从 0 起，project 用绝对工程帧。
    // begin 只对 project 非零，因此默认值必须写回绝对下标，不能按 offset 顺序 push。
    let mut out = vec![
        0.0;
        if local {
            count
        } else {
            begin.saturating_add(count)
        }
    ];
    for offset in 0..count {
        let frame = begin + offset;
        let project_sec = if local {
            geometry.project_start + offset as f64 * frame_ms / 1000.0
        } else {
            frame as f64 * frame_ms / 1000.0
        };
        let index = if local { offset } else { frame };
        out[index] = sample_project_frames(source, project_sec * 1000.0 / frame_ms, "pitch_edit");
    }
    out
}

/// 输出帧 -> 编辑曲线帧。FollowMaterial 用当前几何；PinProject 用覆盖该工程
/// 时刻的冻结窗口。找不到窗口时返回 None（这一帧没有编辑）。
fn source_frame_for_output(
    geometry: &RegionGeometry,
    placement: &Placement,
    frame_ms: f64,
    frame: usize,
    local: bool,
) -> Option<f64> {
    let project_sec = if local {
        geometry.project_start + frame as f64 * frame_ms / 1000.0
    } else {
        frame as f64 * frame_ms / 1000.0
    };
    // 工程根视图的数组从 0 开始，clip 之外的帧本来就没有编辑；不能把窗口外
    // 的帧钳到窗口起点（那会把第一格编辑复制到整条空白区）。
    let project_end = geometry.project_start + geometry.project_duration;
    if local {
        // 局部数组的帧数按 ceil 取整，最后一格可能略微越过窗口终点；渲染仍需要
        // 这一格，夹回窗口内侧而不是返回默认值。
        let _ = project_sec;
    } else if project_sec < geometry.project_start - 1e-9 || project_sec > project_end + 1e-9 {
        return None;
    }
    let project_sec = project_sec.clamp(
        geometry.project_start,
        (project_end - 1e-9).max(geometry.project_start),
    );
    let map = placement
        .map_at_project(geometry, project_sec)
        .ok()
        .flatten()?;
    let source_sec = map.project_to_source(project_sec).or_else(|| {
        ((project_sec - project_end).abs() < 1e-9)
            .then_some(geometry.source_start + geometry.source_duration)
    })?;
    Some(source_sec * 1000.0 / frame_ms)
}

/// 把一条**旧存档形状**的曲线折算成源域。
///
/// 【为什么只剩这一个方向】曲线现在**永远按源秒存**（见 `Placement`），因此不存在
/// "源域 → 工程域"的搬运 —— 那正是旧实现里必然有损、必须加闸、最终让开关"时灵时不灵"
/// 的那一步。留下的只有一次性的读旧存档迁移：按工程帧索引的数组折算成源域数组。
///
/// 【为什么只覆盖当前源窗】工程域的格号是**绝对工程帧**，而一个源窗只覆盖工程时间的
/// 一段 —— 曲线超出本 clip 的那部分在源域里没有对应的源格可写（它的源位置在窗外）。
/// 迁移是"升级这一刻把用户看到的东西原样搬过来"，窗外那部分本来也不可见；
/// 它在下一次该 clip 的窗口覆盖到那里时会被重新采样出来（曲线是源域权威，
/// 只是这条旧曲线还没覆盖到那儿）。
///
/// 【返回 `None` 的三种情形】已经是源域（无事可做）、几何不可解、曲线为空。
/// 三种都不该改动任何东西。
fn curve_into_source_domain(
    curve: &SourceCurve,
    geometry: &RegionGeometry,
    key: &str,
    frame_ms: f64,
) -> Option<SourceCurve> {
    if (curve.basis.is_none() && curve.domain == CurveDomain::Source) || curve.values.is_empty() {
        return None;
    }
    if !(frame_ms.is_finite() && (0.1..=1000.).contains(&frame_ms)) {
        return None;
    }
    let step = frame_ms / 1000.;
    let lo = geometry.source_start;
    let hi = geometry.source_start + geometry.source_duration;
    let map = geometry.map().ok()?;
    let first = (lo / step).floor().max(0.) as usize;
    let last = (hi / step).ceil() as usize;
    if last < first || last >= 1_000_000 {
        return None;
    }
    let mut values = Vec::with_capacity(last - first + 1);
    for k in first..=last {
        let source_sec = (k as f64 * step).clamp(lo, hi);
        let project_sec = map.source_to_project(source_sec)?;
        values.push(
            curve
                .sample_indexed(project_sec, key, frame_ms)
                .unwrap_or_else(|| pad(key)),
        );
    }
    Some(SourceCurve {
        basis: None,
        domain: CurveDomain::Source,
        first_frame: first,
        frame_ms,
        values: Arc::new(values),
        reservation: None,
    })
}

/// 两条曲线是否逐点一致（索引域、起点、采样）。
///
/// 【为什么只比这三项】`reservation` 是预算登记的句柄，不是数据；把它算进来会让
/// "内容没变"被误判成"变了"，于是每次捕获都推进代际、重写存档。
fn same_curve(a: &SourceCurve, b: &SourceCurve) -> bool {
    let domain_equal = match (&a.basis, &b.basis) {
        (None, None) => true,
        (Some(x), Some(y)) => x.equivalent(y),
        _ => false,
    };
    domain_equal
        && a.first_frame == b.first_frame
        && a.frame_ms == b.frame_ms
        && a.values.len() == b.values.len()
        && a.values.iter().zip(b.values.iter()).all(|(x, y)| x == y)
}

/// 一次源域捕获的全部输入（打包成一个结构，避免长参数表）。
struct SourceCapture<'a> {
    /// 本权威里这条曲线**当前**的形状（同一帧内已写入的部分也算）。
    existing: Option<&'a SourceCurve>,
    /// 上一帧 GUI 投影（delta 基准）；`capture`（无基准）时为 `None`。
    old_view: Option<&'a [f32]>,
    /// 本帧 GUI 视图（工程帧索引）。
    values: &'a [f32],
    /// 本 region 的工程→源映射。
    current: &'a SourceTimeMap,
    geometry: &'a RegionGeometry,
    key: &'a str,
    frame_ms: f64,
    /// 被另一条选中片段覆盖的工程区间（该区间不写）。
    protected: Option<(f64, f64)>,
}

/// 把一次捕获写回**源域**，返回这条曲线的新形状。
///
/// 【为什么必须换域】权威键是 modification 级（多条 region **共享**），而工程帧是
/// region 级（每条**独占**）。两者不同域 ⇒ 共享的键被逐条 region 覆盖，只剩最后一条的
/// 窗口 —— 这正是"在 Cubase 里分割后参数完全重置"的根因。改按源帧索引后，共享同一
/// modification 的每条 region 写的是同一条曲线的**不同区间**，互不覆盖。
///
/// 【为什么要与既有曲线取并集】别的半段已经写过的源区间不能被这一次捕获裁掉
/// （那正是旧实现 `set_authority` 整份覆盖的翻版）。
/// 一条根轨的曲线里"非默认帧"的总数（I-4 的判据）。
///
/// 【为什么这个数就够】用户曲线的价值全在"与默认值不同的帧"上。任何一次自动写回
/// 若让这个数**变小**，就说明有用户数据被抹平了 —— 而那正是"参数回退"的定义。
/// 把判据收敛成一个整数，才能让 `[ara] line write` 那一行既可读又可断言。
fn non_default_frames(curves: &BTreeMap<String, SourceCurve>) -> usize {
    let mut total = 0_usize;
    for (key, curve) in curves {
        total += non_default_values(key, curve);
    }
    total
}

/// 单条曲线的"非默认帧"数（`non_default_frames` 的逐条版本）。
///
/// 【为什么要拆出来】"这次烘焙会不会让曲线变短"要在**单条**曲线上判，而
/// 硬闸判的是整份权威。两者用同一个计数函数，才不会出现"闸门说没变、烘焙却少了"。
fn non_default_values(key: &str, curve: &SourceCurve) -> usize {
    let default = pad(key);
    curve
        .values
        .iter()
        .filter(|value| value.is_finite() && (**value - default).abs() > 1e-6)
        .count()
}
fn capture_source_curve(request: SourceCapture<'_>) -> Result<(SourceCurve, usize), String> {
    let SourceCapture {
        existing,
        old_view,
        values,
        current,
        geometry,
        key,
        frame_ms,
        protected,
    } = request;
    if !frame_ms.is_finite() || !(0.1..=1000.).contains(&frame_ms) {
        return Err("invalid source parameter frame domain".into());
    }
    let project_end = geometry.project_start + geometry.project_duration;
    // 本 region 在源域上覆盖的区间。
    let a = current
        .project_to_source(geometry.project_start)
        .ok_or("region start does not map to the source")?;
    let b = current
        .project_to_source(project_end)
        .ok_or("region end does not map to the source")?;
    let (region_lo, region_hi) = (a.min(b), a.max(b));
    // 与既有曲线取并集。
    let (lo, hi) = match existing.and_then(SourceCurve::source_extent) {
        Some((e_lo, e_hi)) => (region_lo.min(e_lo), region_hi.max(e_hi)),
        None => (region_lo, region_hi),
    };
    let first = (lo * 1000. / frame_ms).floor().max(0.);
    let last = (hi * 1000. / frame_ms).ceil();
    if !last.is_finite() || last >= 1_000_000. || last < first {
        return Err("source parameter projection frame budget exceeded".into());
    }
    let first_frame = first as usize;
    let count = last as usize - first_frame + 1;
    let mut samples = Vec::with_capacity(count);
    for k in first_frame..first_frame + count {
        let source_sec = k as f64 * frame_ms / 1000.;
        samples.push(
            existing
                .and_then(|curve| curve.value_at_source(source_sec, key))
                .unwrap_or_else(|| pad(key)),
        );
    }
    // 覆盖本 region 源窗内、被用户改动过的格点。
    //
    // 【为什么只写 region 自己的格点范围，而不是整段并集】并集边界按 `ceil` 取整，
    // region 的最后（或最前）一个源格点会落在源窗之外一丁点。它必须被填上边界值，
    // 否则投影在窗末插值到 `pad`（实测：末格变 0，边界值算成 36 而非 60）。
    // 但**不能**把这套 clamp 用到整段并集上 —— 那样别的半段已经写好的区间会被
    // 边界值覆盖（实测：左半段读出右半段的 68）。两者只差一个循环范围。
    let step = frame_ms / 1000.;
    let region_first = (region_lo / step).floor().max(0.) as usize;
    let region_last = (region_hi / step).ceil() as usize;
    // 被"另一条选中片段"的保护区间挡下的帧数（R5 的读数：用户要能看见"这几帧没写"）。
    let mut protected_skipped = 0_usize;
    for k in region_first..=region_last.min(first_frame + count - 1) {
        let source_sec = k as f64 * step;
        let clamped = source_sec.clamp(region_lo, region_hi);
        let Some(project_sec) = current.source_to_project(clamped) else {
            continue;
        };
        if project_sec < geometry.project_start || project_sec > project_end {
            continue;
        }
        if protected.is_some_and(|(lo, hi)| project_sec >= lo && project_sec <= hi) {
            protected_skipped += 1;
            continue;
        }
        let project_frame = project_sec * 1000. / frame_ms;
        let incoming = sample_project_frames(values, project_frame, key);
        let prior = old_view
            .map(|view| sample_project_frames(view, project_frame, key))
            .unwrap_or_else(|| pad(key));
        if incoming != prior {
            samples[k - first_frame] = incoming;
        }
    }
    // 【存储域恒为源域 —— 这就是「锁定参数线」不再需要烘焙的原因】
    //
    // 旧实现让「不锁定」把 `values[i]` 写成"工程第 first_frame+i 格"，于是同一个数组
    // 要在两个域之间搬（`bake_curve_to_anchor`），搬动必然有损、有损就必须有闸、
    // 有闸就必然出现"闸拒绝 ⇒ 开关无效"。现在曲线永远按源秒存，开关只换投影用的
    // 映射（见 `Placement`），一个字节都不动。
    let domain = CurveDomain::Source;

    let bytes = samples.len() * 4;
    let reservation = crate::render::budget::global_budget()
        .reserve(bytes)
        .ok_or("source parameter memory budget exceeded")?;
    Ok((
        SourceCurve {
            basis: None,
            domain,
            first_frame,
            frame_ms,
            values: Arc::new(samples),
            reservation: Some(Arc::new(reservation)),
        },
        protected_skipped,
    ))
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
    /// **不完整的根视图只能覆盖它真的投影过的帧。**
    ///
    /// 【为什么这是"移动一下 item，整轨参数被抹平"的回归测试】旧实现是
    /// `params.extend(roots)` —— 对已存在的键**整值替换**。而 `project_roots` 只能
    /// 投影"有权威且几何可解"的 clip，其余帧是 `pad()`。一条 clip 拿不到权威就足以
    /// 用"一段默认值 + 半条真曲线"盖掉用户原来的整条曲线。
    fn root_params() -> TrackParamsState {
        serde_json::from_value(serde_json::json!({"frame_period_ms": 5.0})).unwrap()
    }

    fn pitch_edit_values(authority: &AuthorityParameters) -> Vec<f32> {
        authority
            .edits
            .get("pitch_edit")
            .map(|curve| {
                curve
                    .runs
                    .iter()
                    .flat_map(|run| run.values.iter().copied())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn edit_samples(authority: &AuthorityParameters) -> usize {
        authority
            .edits
            .values()
            .map(|curve| curve.runs.iter().map(|run| run.values.len()).sum::<usize>())
            .sum::<usize>()
            + authority
                .curves
                .values()
                .map(|curve| curve.values.iter().filter(|value| **value != 0.0).count())
                .sum::<usize>()
    }

    #[test]
    fn an_incomplete_root_projection_only_overwrites_its_own_frames() {
        let mut existing = root_params();
        existing.pitch_edit = vec![60.0; 100];
        existing.tension_edit = vec![0.5; 100];
        let mut target = BTreeMap::from([("a".to_owned(), existing)]);
        // 这一轮只投影到了 [10, 20) 这一段（其余帧没有信息）。
        let mut partial = root_params();
        partial.pitch_edit = vec![70.0; 100];
        partial.tension_edit = vec![0.5; 100];
        merge_root_projection(
            &mut target,
            BTreeMap::from([(
                "a".to_owned(),
                RootProjection {
                    params: partial,
                    #[allow(clippy::single_range_in_vec_init)]
                    frames: [10..20].to_vec(),
                    edited_frames: BTreeMap::new(),
                    complete: false,
                    baseline_ready: true,
                    cleared_curves: std::collections::BTreeSet::new(),
                },
            )]),
        );
        let merged = &target["a"];
        assert_eq!(
            &merged.pitch_edit[10..20],
            &[70.0; 10][..],
            "覆盖到的帧必须更新"
        );
        assert_eq!(merged.pitch_edit[0], 60.0, "没投影到的帧必须保持原值");
        assert_eq!(merged.pitch_edit[50], 60.0);
        assert_eq!(merged.pitch_edit[99], 60.0);
        assert_eq!(merged.tension_edit[0], 0.5);
    }

    /// 整轨投影（空白编辑）照常整份覆盖。
    #[test]
    fn a_whole_root_projection_replaces_every_frame() {
        let mut existing = root_params();
        existing.pitch_edit = vec![60.0; 100];
        let mut target = BTreeMap::from([("a".to_owned(), existing)]);
        let mut whole = root_params();
        whole.pitch_edit = vec![70.0; 100];
        merge_root_projection(
            &mut target,
            BTreeMap::from([("a".to_owned(), RootProjection::whole(whole))]),
        );
        assert!(target["a"].pitch_edit.iter().all(|value| *value == 70.0));
    }
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

    /// 没有谱系、也没有 item GUID 时，权威查表仍然工作（不回归成硬错误）。
    #[test]
    fn an_ara_only_region_without_lineage_still_resolves_through_the_authority_table() {
        let original = edited();
        let atlas = ParameterAtlas::default()
            .capture(&original, &identities())
            .unwrap();
        let timeline = host(1., 0.5, 0., 0.5);
        let projected = atlas
            .follow_geometry(&timeline, &identities())
            .unwrap()
            .project_local(&timeline, &identities())
            .expect("权威查表不得因缺少谱系而失败");
        assert_eq!(projected["clip"].pitch_edit[0], 60.);
    }

    /// **连续拆分不累积记录；删光之后权威仍在（撤销能找回）。**
    ///
    /// 【为什么这条测试取代了"两个死祖先"那条】旧设计把"曲线归属"存在 per-clip 记录
    /// 里，于是被删掉的片段留下**死记录**，而几何兜底会把两条死记录同时判成候选 ⇒
    /// `Conflict: ambiguous source parameter ancestry`。新设计里权威按稳定宿主身份索引，
    /// 歧义不可能出现；但"删除后撤销要能找回曲线"这条**用户保证**必须继续成立，
    /// 只是它现在由权威表承载。
    #[test]
    fn nested_splits_do_not_accumulate_records_and_the_authority_survives_deletion() {
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
        // 派生缓存只包含**当前活着的**片段：祖先不会作为死记录累积在候选池里。
        assert_eq!(
            a2.regions.keys().cloned().collect::<Vec<_>>(),
            vec!["quarter-a", "quarter-b", "right"],
            "每次分割都在派生缓存里留下祖先，候选池就会随分割次数无界增长"
        );
        // 权威只有一条（三段共享同一个 modification "mod"）—— 它按稳定身份索引，
        // 不随片段数量增长。
        assert_eq!(a2.authorities.len(), 1);
        assert!(a2.has_authority_parameters(None, "mod"));

        // 全删：派生缓存清空，但**权威仍在** —— 撤销/重做把片段带回来时曲线照旧。
        let empty: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0}],"bpm":120,"project_sec":2.0,"clips":[]
        }))
        .unwrap();
        let orphaned = a2.follow_geometry(&empty, &BTreeMap::new()).unwrap();
        assert!(
            orphaned.regions.is_empty(),
            "没有活片段时派生缓存必须为空：{:?}",
            orphaned.regions.keys().collect::<Vec<_>>()
        );
        assert!(
            orphaned.has_authority_parameters(None, "mod"),
            "删光片段不得丢掉权威参数，否则撤销/重做会丢曲线"
        );

        // 片段回来（同一 modification）⇒ 曲线立刻重新投影。
        let mut third: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0}],"bpm":120,"project_sec":4.0,"clips":[]
        }))
        .unwrap();
        third.clips.push(host(1., 1., 0., 1.).clips.remove(0));
        let third_ids = BTreeMap::from([with_signature("clip", 46, 1.0, 1.0)]);
        let resolved = orphaned
            .follow_geometry(&third, &third_ids)
            .expect("权威查表不得让整份编辑失败");
        assert!(resolved.regions["clip"].live);
        let projected = resolved.project_roots(&third, &third_ids).unwrap();
        assert_eq!(projected["a"].pitch_edit[4], 60., "撤销回来后曲线必须还在");
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
    /// **没有该根的基线 = 这一轮没有可证实的用户意图。**
    ///
    /// 【为什么这是"分割后参数线回退"的闸】根轨 id 被宿主重建之后，基线还挂在旧键上；
    /// 而"没有权威 + 没有基线"的 clip 在旧实现里会被当成"整窗都是用户改的"，把投影
    /// 出来的默认值写进一个新权威 —— 不可逆。现在这一条被跳过并留痕。
    #[test]
    fn a_missing_root_baseline_skips_the_clip_instead_of_writing_defaults() {
        let host = host(0., 1., 0., 1.);
        let ids = identities();
        let mut client = host.clone();
        let mut params = TrackParamsState {
            frame_period_ms: 100.,
            ..Default::default()
        };
        params.pitch_edit = vec![60.; 6];
        params.pitch_edit_user_modified = true;
        client
            .params_by_root_track
            .insert("a".into(), params.clone());

        let atlas = ParameterAtlas::default();
        // 基线里**有别的根**、却没有这个根 —— 宿主重建序列后的真实形状。
        let stale = TrackParamsState {
            frame_period_ms: 100.,
            ..Default::default()
        };
        let after = atlas
            .capture_changes(
                &client,
                &ids,
                &BTreeMap::from([("old-root".to_owned(), stale)]),
            )
            .unwrap();
        assert!(
            after.authorities.is_empty(),
            "根键错位时不得凭空建立权威（那会把默认值固化成用户曲线）"
        );

        // 同一份输入，基线里有这个根 ⇒ delta 正常写入（闸不能拦住正常编辑）。
        let mut view = TrackParamsState {
            frame_period_ms: 100.,
            ..Default::default()
        };
        view.pitch_edit = vec![0.; 6];
        let baseline = BTreeMap::from([("a".to_owned(), view)]);
        let after = atlas.capture_changes(&client, &ids, &baseline).unwrap();
        assert!(
            after.authorities.contains_key("mod:mod"),
            "有基线时 delta 必须照常落盘"
        );
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

    /// **Cubase 形状**：一条源、**一个 modification**、三条 region 共享它。
    ///
    /// 【为什么这是必须的最小夹具】Cubase 一次分割之后，所有半段共享同一个
    /// `audioModificationPersistentID` —— 仓库里取自真机日志的 `cubase_split_document()`
    /// （`ara/mapping.rs:969`）就是这个形状（1 条 modification + 8 条 region）。
    /// 权威键按 modification 索引 ⇒ 多条 region 落到**同一个键**上。
    /// 上一轮的 24 条回归测试从未造过这个形状，于是"共享键被逐条 region 覆盖"一直没被发现。
    ///
    /// 源 [0,3) 被切成三段，各占工程 [0,1) / [1,2) / [2,3)；整轨曲线是 60,61,62,…
    fn cubase_split_timeline() -> TimelineState {
        let mut timeline: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0}],"bpm":120,"project_sec":3.,
            "clips":[
                {"id":"left","track_id":"a","name":"left","start_sec":0.,"length_sec":1.,
                 "takes":[{"id":"tl","source_path":"source","source_start_sec":0.,
                           "source_end_sec":1.,"playback_rate":1.}]},
                {"id":"mid","track_id":"a","name":"mid","start_sec":1.,"length_sec":1.,
                 "takes":[{"id":"tm","source_path":"source","source_start_sec":1.,
                           "source_end_sec":2.,"playback_rate":1.}]},
                {"id":"right","track_id":"a","name":"right","start_sec":2.,"length_sec":1.,
                 "takes":[{"id":"tr","source_path":"source","source_start_sec":2.,
                           "source_end_sec":3.,"playback_rate":1.}]}
            ]
        }))
        .unwrap();
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }
        timeline.params_by_root_track.insert(
            "a".into(),
            TrackParamsState {
                frame_period_ms: 250.,
                pitch_edit_user_modified: true,
                // 13 个格点覆盖工程 [0,3]（帧 0..=12）。
                pitch_edit: (0..13).map(|frame| 60. + frame as f32).collect(),
                ..Default::default()
            },
        );
        timeline
    }

    /// 三条半段的身份：**同一个 modification**，各自的源窗不同（Cubase 分割后的形状）。
    fn cubase_shared_identities() -> BTreeMap<String, RegionIdentity> {
        ["left", "mid", "right"]
            .iter()
            .enumerate()
            .map(|(index, id)| {
                (
                    (*id).to_owned(),
                    RegionIdentity {
                        key: 100 + index as u64,
                        item: None,
                        source: "source".into(),
                        modification: "mod-1".into(),
                        signature: String::new(),
                    },
                )
            })
            .collect()
    }

    /// **共享一个 modification 时，每条 region 读的是同一条源曲线上属于自己的那一段。**
    ///
    /// 【为什么这是本轮的核心断言】此前权威里的曲线基是**捕获时那一条 region 的几何**，
    /// 而键是共享的 modification ⇒ 后捕获的半段覆盖先捕获的 ⇒ 前两条半段投影时全部落到
    /// `pad()`（音高归 0）。这正是用户在 Cubase 里看到的"分割后参数完全重置"。
    #[test]
    fn sharing_one_modification_each_region_reads_its_own_source_window() {
        let timeline = cubase_split_timeline();
        let ids = cubase_shared_identities();
        let atlas = ParameterAtlas::default().capture(&timeline, &ids).unwrap();
        let view = atlas.project_local(&timeline, &ids).unwrap();
        assert_eq!(
            &view["left"].pitch_edit[0..4],
            &[60., 61., 62., 63.],
            "左半段必须读源窗 [0,1) 上那一段"
        );
        assert_eq!(
            &view["mid"].pitch_edit[0..4],
            &[64., 65., 66., 67.],
            "中半段必须读源窗 [1,2) 上那一段"
        );
        assert_eq!(
            &view["right"].pitch_edit[0..4],
            &[68., 69., 70., 71.],
            "右半段必须读源窗 [2,3) 上那一段"
        );
    }

    /// **编辑一个半段不得重置其余半段。**（计划 §6 的核心断言）
    #[test]
    fn editing_one_half_does_not_reset_the_other_halves() {
        let timeline = cubase_split_timeline();
        let ids = cubase_shared_identities();
        let atlas = ParameterAtlas::default().capture(&timeline, &ids).unwrap();
        let before = atlas.project_roots(&timeline, &ids).unwrap();

        // 用户只把左半段第一个格点改成 55。
        let mut edited_timeline = timeline.clone();
        let params = edited_timeline.params_by_root_track.get_mut("a").unwrap();
        params.pitch_edit[0] = 55.;
        let after = atlas
            .capture_changes(&edited_timeline, &ids, &before)
            .unwrap()
            .project_roots(&edited_timeline, &ids)
            .unwrap();

        assert_eq!(after["a"].pitch_edit[0], 55., "被编辑的格点必须生效");
        // 其余格点逐帧不变 —— 包括中、右半段所覆盖的那些。
        for frame in 1..13 {
            assert_eq!(
                after["a"].pitch_edit[frame], before["a"].pitch_edit[frame],
                "frame {frame} 未被编辑，不得变化"
            );
        }
    }

    /// **三次捕获不得把权威裁到最后一条半段的窗口。**（计划 §6）
    #[test]
    fn capturing_three_shared_regions_does_not_shrink_the_authority() {
        let timeline = cubase_split_timeline();
        let ids = cubase_shared_identities();
        let atlas = ParameterAtlas::default().capture(&timeline, &ids).unwrap();
        // 权威里那条曲线必须仍覆盖整条源 [0,3)：三条半段的源窗都能投影出真实值。
        let view = atlas.project_local(&timeline, &ids).unwrap();
        for (id, expected) in [("left", 60.), ("mid", 64.), ("right", 68.)] {
            assert_eq!(
                view[id].pitch_edit[0], expected,
                "{id} 的源窗起点必须仍能取到权威里的值，而不是 pad()"
            );
        }
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
    /// **一次宿主分割：两个半段各自拿到父段曲线的对应子段。**
    ///
    /// 【为什么这是用户报障的直接回归测试】旧设计里 `ara_region_signature` 含播放几何，
    /// 分割一发生签名就变、谱系蒸发，两个半段都认领不到父段 ⇒ 曲线被丢弃（"参数被重置"）。
    /// 新设计按稳定身份查表，父段参数被整份播给两个新身份，投影各取所需。
    #[test]
    fn a_host_split_gives_each_half_the_parent_curve_for_its_own_source_window() {
        let ids = identities();
        let original = edited();
        let atlas = ParameterAtlas::default().capture(&original, &ids).unwrap();
        // 宿主克隆 modification：父 "mod" → 左 "mod-l" / 右 "mod-r"（P1 clone 回执或
        // P2 结构 diff 都落到这里）。
        let mut split = atlas.clone();
        assert!(split.seed_authority("mod:mod", "mod:mod-l"));
        assert!(split.seed_authority("mod:mod", "mod:mod-r"));

        let mut halves = host(1., 0.5, 0., 0.5);
        halves.clips[0].id = "left".into();
        let mut right = host(1.5, 0.5, 0.5, 0.5).clips.remove(0);
        right.id = "right".into();
        halves.clips.push(right);
        halves.project_sec = 2.;
        let half_ids = BTreeMap::from([
            (
                "left".into(),
                RegionIdentity {
                    key: 42,
                    item: None,
                    source: "source".into(),
                    modification: "mod-l".into(),
                    signature: String::new(),
                },
            ),
            (
                "right".into(),
                RegionIdentity {
                    key: 43,
                    item: None,
                    source: "source".into(),
                    modification: "mod-r".into(),
                    signature: String::new(),
                },
            ),
        ]);
        let projected = split
            .follow_geometry(&halves, &half_ids)
            .unwrap()
            .project(&halves, &half_ids)
            .unwrap();
        // 父曲线在源域 [0,1) 上是 60,61,62,63,64（frame_ms=250 ⇒ 每 0.25s 一格）。
        // 左半段消费源窗 [0,0.5) ⇒ 60,61,62；右半段消费 [0.5,1) ⇒ 从 62 起。
        assert_eq!(&projected["left"].pitch_edit[4..7], &[60., 61., 62.]);
        assert_eq!(projected["right"].pitch_edit[6], 62.);
        assert_ne!(
            projected["right"].pitch_edit[4], 60.,
            "右半段不得整体平移左半段的曲线"
        );
    }

    /// **裁剪与取消裁剪都不缩短曲线；取消裁剪不再把整条曲线重置。**
    ///
    /// 旧设计的几何兜底是单向"包含"判定：新窗口一旦**超出**旧窗口就返回 `None`
    /// ⇒ 该片段**一条曲线都没有**（用户看到的"参数被重置"）。新设计的曲线以**源域**
    /// 存在权威里，裁剪只改变投影窗口，权威里的源曲线不受影响 —— 因此取消裁剪能完整
    /// 恢复。这条测试同时钉住"裁剪不截短"与"取消裁剪不重置"两件事。
    #[test]
    fn trimming_does_not_shorten_the_authority_curve_and_un_trimming_restores_it() {
        let ids = identities();
        let original = edited();
        let atlas = ParameterAtlas::default().capture(&original, &ids).unwrap();
        // 裁剪到源窗 [0.5,1)：宿主几何 + 与之匹配的 GUI 投影一起给，避免把插值当新数据。
        let mut cropped_host = host(1.5, 0.5, 0.5, 0.5);
        cropped_host.params_by_root_track.insert(
            "a".into(),
            atlas
                .project(&cropped_host, &ids)
                .unwrap()
                .remove("clip")
                .unwrap(),
        );
        let cropped = atlas.capture(&cropped_host, &ids).unwrap();

        // 裁剪态只投影消费窗口内的值。
        let projected = cropped.project(&cropped_host, &ids).unwrap();
        assert_eq!(&projected["clip"].pitch_edit[6..9], &[62., 63., 64.]);

        // 用户把左边界拉回去：源窗 [0.5,1) → [0,1)。整条父曲线都回来。
        let restored = cropped.project(&host(1., 1., 0., 1.), &ids).unwrap();
        assert_eq!(
            &restored["clip"].pitch_edit[4..9],
            &[60., 61., 62., 63., 64.],
            "取消裁剪必须完整恢复曲线，而不是重置或延展旧末值"
        );
    }

    /// **冷重载后 region key 为 0 不再致命。**
    ///
    /// `RegionIdentity::key` 是 `#[serde(skip)]` 的 model-ref 地址，重载后恒为 0。
    /// 旧设计在 `capture` 里以 `key == 0 ⇒ continue` 把**所有**片段跳过，于是任何一次
    /// 存盘/重载之后一条曲线都捕获不到。权威按稳定宿主身份索引后，key 只是诊断字段。
    #[test]
    fn a_cold_reload_restores_curves_without_a_live_region_key() {
        let ids = identities();
        let original = edited();
        let atlas = ParameterAtlas::default().capture(&original, &ids).unwrap();
        // 新存档形状：派生缓存不入档，只有权威表。
        let mut trimmed = atlas.clone();
        trimmed.regions.clear();
        let restored: ParameterAtlas =
            serde_json::from_slice(&serde_json::to_vec(&trimmed).unwrap()).unwrap();
        let mut cold_ids = identities();
        cold_ids.get_mut("clip").unwrap().key = 0;
        let projected = restored.project(&original, &cold_ids).unwrap();
        assert_eq!(
            &projected["clip"].pitch_edit[4..9],
            &[60., 61., 62., 63., 64.],
            "key 为 0 不得再让曲线缺席"
        );
    }

    /// **旧存档的 per-clip 记录被折算进权威表。**
    #[test]
    fn legacy_per_clip_records_migrate_into_the_authority_table() {
        let ids = identities();
        let original = edited();
        let atlas = ParameterAtlas::default().capture(&original, &ids).unwrap();
        assert!(!atlas.regions.is_empty());
        // 旧存档形状：只有 per-clip 记录，权威表为空。
        let mut legacy = atlas.clone();
        legacy.authorities.clear();
        let migrated = legacy.follow_geometry(&original, &ids).unwrap();
        assert!(
            migrated.has_authority_parameters(None, "mod"),
            "旧存档必须被折算成权威条目"
        );
        let projected = migrated.project(&original, &ids).unwrap();
        assert_eq!(
            &projected["clip"].pitch_edit[4..9],
            &[60., 61., 62., 63., 64.]
        );
    }

    /// **源窗不相交时，该片段退回默认值而不是错读别人的区间。**
    ///
    /// 【为什么这条是偏移的防线】"归属接错"在数值上的表现是"读到了别的半段的值"，
    /// 而"没接上"的表现是"退回默认"。后者留痕（`ara.render.split-source-window`），
    /// 前者无声 —— 所以判据必须是：**宁可没有，也不要错拿**。这条测试把该 clip 的
    /// 源窗搬到权威覆盖区间之外，断言整条曲线归默认，而不是出现任何真实数值。
    #[test]
    fn a_source_window_outside_the_authority_cover_projects_defaults_not_neighbours() {
        let ids = identities();
        let atlas = ParameterAtlas::default().capture(&edited(), &ids).unwrap();
        // 同一份权威，但该 clip 的源窗整体移到曲线覆盖之外（[0,1) → [9,10)）。
        let elsewhere = host(1., 1., 9., 1.);
        let projected = atlas.project(&elsewhere, &ids).unwrap();
        assert_eq!(
            &projected["clip"].pitch_edit[4..9],
            &[0., 0., 0., 0., 0.],
            "源窗不相交时必须退回默认；读到任何非零值都说明接错了区间"
        );
        // 对照组：源窗正确时同一份权威照常给出真实曲线。
        let correct = atlas.project(&host(1., 1., 0., 1.), &ids).unwrap();
        assert_eq!(
            &correct["clip"].pitch_edit[4..9],
            &[60., 61., 62., 63., 64.]
        );
    }

    /// **投影只读权威表，不读谱系推断。**
    ///
    /// 【为什么需要这条】`HIFISHIFTER_ARA_LEGACY_ATLAS` 开关已删除（它能在环境变量里
    /// 把"按几何猜祖先"这个已知缺陷开回来）。删掉之后要保证的是：即便谱系表里塞满
    /// 条目，投影**一个字都不读它们** —— 归属只由该 clip 自己的源窗与权威曲线的源区间
    /// 求交决定。这条测试故意把谱系表全部填满，再断言结果与清空时逐帧相同。
    #[test]
    fn projection_ignores_lineage_tables_entirely() {
        let ids = identities();
        let original = edited();
        let atlas = ParameterAtlas::default().capture(&original, &ids).unwrap();
        let clean = atlas.follow_geometry(&original, &ids).unwrap();
        let expected = clean.project(&original, &ids).unwrap();

        let mut poisoned = atlas.clone();
        // 把两张谱系表与种子表全部填上**指向错误片段**的条目：如果投影还读它们，
        // 这一段就会把曲线接到别的源窗上（"整体平移一个分割长度"的老形状）。
        // 键用 GUID 形状且父子不同：`validate` 会拒绝畸形/自指的谱系，而这里要测的
        // 是"结构合法也一个字都不读"，不是"畸形条目会不会被拒绝"。
        let polluted_child = "{99999999-9999-9999-9999-999999999999}".to_owned();
        let polluted_parent = "{88888888-8888-8888-8888-888888888888}".to_owned();
        poisoned
            .split_parents
            .insert(polluted_child.clone(), polluted_parent.clone());
        poisoned
            .ara_split_parents
            .insert(polluted_child.clone(), polluted_parent);
        poisoned.ara_copy_seeds.insert(
            polluted_child,
            RegionParameters {
                identity: RegionIdentity {
                    key: 0,
                    item: None,
                    source: "other-source".into(),
                    modification: "unrelated".into(),
                    signature: String::new(),
                },
                root: "a".into(),
                edits: BTreeMap::new(),
                current: RegionGeometry {
                    project_start: 1.,
                    project_duration: 1.,
                    source_start: 5.,
                    source_duration: 5.,
                    reversed: false,
                },
                live: false,
                template: TrackParamsState::default(),
                curves: BTreeMap::new(),
            },
        );
        let followed = poisoned.follow_geometry(&original, &ids).unwrap();
        let projected = followed.project(&original, &ids).unwrap();
        assert_eq!(
            projected["clip"].pitch_edit, expected["clip"].pitch_edit,
            "谱系表不得参与投影：曲线归属只由该 clip 自己的源窗决定"
        );
    }

    /// 「锁定参数线」的取值必须映射成正确的放置策略。
    #[test]
    fn the_lock_setting_maps_to_the_expected_placement() {
        assert_eq!(
            placement_for_lock(true),
            Placement::FollowMaterial,
            "锁定 ⇒ 跟着素材走"
        );
        assert!(
            matches!(
                placement_for_lock(false),
                Placement::PinProject { ref frozen } if frozen.is_empty()
            ),
            "不锁定 ⇒ 留在工程位置（冻结几何在第一次投影时补上）"
        );
    }

    /// **「锁定参数线」的双放置：锁定跟素材走，不锁定留在工程位置。**
    ///
    /// 【为什么必须实现它】宿主发起的 ARA 编辑不走 `move_clips` / `stretch_*`，因此
    /// 这条设置此前对**任何宿主编辑**结构性失效（`lock_param_lines` 在
    /// `hifishifter-plugin/src/` 里出现次数为 0）。现在它由投影时的放置策略表达。
    ///
    /// 【为什么曲线不再换域】两条分支用的都是**同一份源域数组**，差别只在"用哪份
    /// 几何把它折回工程时间"。这正是开关可逆的原因：拨回去逐帧回到原样。
    #[test]
    fn the_lock_setting_selects_whether_curves_follow_the_material_or_the_project() {
        let ids = identities();
        // 曲线在工程 [1,2) 上：60,62,64,66（每 250ms 一格）。
        let curve = vec![60., 62., 64., 66.];

        // 锁定（跟随素材）：片段从 [1,2) 移到 [3,4)，曲线跟着走。
        let mut locked = ParameterAtlas::default();
        locked.seed_test_pitch_curve_placed(
            "mod:mod",
            250.,
            curve.clone(),
            Placement::FollowMaterial,
        );
        let moved = locked.project(&host(3., 1., 0., 1.), &ids).unwrap();
        assert_eq!(
            &moved["clip"].pitch_edit[12..16],
            &[60., 62., 64., 66.],
            "锁定时参数必须跟着素材移动"
        );

        // 不锁定（钉在 [1,2) 那份几何上）：移走之后新位置是空的。
        let mut unlocked = ParameterAtlas::default();
        unlocked.seed_test_pitch_curve_placed("mod:mod", 250., curve.clone(), pinned_at(1., 1.));
        let moved = unlocked.project(&host(3., 1., 0., 1.), &ids).unwrap();
        assert_eq!(
            &moved["clip"].pitch_edit[12..16],
            &[0., 0., 0., 0.],
            "不锁定时参数不随片段移动"
        );
        // 原位置仍然保留着它。
        let stayed = unlocked.project(&host(1., 1., 0., 1.), &ids).unwrap();
        assert_eq!(
            &stayed["clip"].pitch_edit[4..8],
            &[60., 62., 64., 66.],
            "不锁定时参数留在原工程位置"
        );
    }

    /// 一份"钉在 [start, start+duration) 上、源窗从 0 起"的放置策略（测试夹具）。
    fn pinned_at(start: f64, duration: f64) -> Placement {
        Placement::pinned(RegionGeometry {
            project_start: start,
            project_duration: duration,
            source_start: 0.,
            source_duration: duration,
            reversed: false,
        })
    }

    /// **投影按权威自己的放置策略解释曲线，而不是按调用方传进来的"当前设置"。**
    ///
    /// 【为什么这是"移动片段后参数线留在原地"的根因】`capture_inner` 会把放置策略
    /// 写进权威（与落地的数据同源），但读路径此前只信"当前设置"。两者可以相反：
    /// 于是投影按另一种映射去读同一个数组，整条曲线偏移一个源窗起点。数据一格都没丢，
    /// 用户看到的却是"参数线不动了"。
    ///
    /// 【那"切换开关立刻生效"怎么办】它由 `ParameterAtlas::set_placement` 实现：
    /// 只改元数据，不改数据。于是"读当前设置"与"读权威放置"在正常情况下是同一件事，
    /// 只在两者短暂不一致时分开 —— 而那时正确的只能是数据自己的那一份。
    #[test]
    fn projection_uses_the_authority_placement_not_the_callers_fallback() {
        let ids = identities();
        let mut atlas = ParameterAtlas::default();
        // 种子：源域曲线 + 跟随素材（生产路径的常态）。
        atlas.seed_test_pitch_curve("mod:mod", 250., vec![60., 62., 64., 66.]);

        // 即便调用方说"当前设置是钉住"，投影也必须按权威自己的策略读 ——
        // 片段从 [1,2) 移到 [3,4) 时曲线跟着素材走。
        let moved = atlas
            .project_with_placement(&host(3., 1., 0., 1.), &ids, &pinned_at(1., 1.))
            .unwrap();
        assert_eq!(
            &moved["clip"].pitch_edit[12..16],
            &[60., 62., 64., 66.],
            "权威策略是跟随素材 ⇒ 曲线必须跟着素材移动，不能被调用方的兜底值改写"
        );
    }

    /// **不变式：权威里每条曲线都按源域存储。**
    ///
    /// 【为什么单独钉一条】上一条测试测的是"读的时候用了谁"，这条测的是"数据本身
    /// 自洽不自洽"。两者一起才排除"读对了但数据是错的"这种更隐蔽的形态。
    /// 源域是**唯一**的存储域 —— 「锁定参数线」只换投影映射，不换存储域，
    /// 这正是那个开关可逆、即时、无损的原因。
    #[test]
    fn every_authority_curve_is_stored_in_the_source_domain() {
        let ids = identities();
        let atlas = ParameterAtlas::default()
            .capture(&edited(), &ids)
            .expect("捕获必须成功");
        for (key, params) in &atlas.authorities {
            for (curve_key, curve) in &params.curves {
                assert_eq!(
                    curve.domain,
                    CurveDomain::Source,
                    "权威 {key} 的曲线 {curve_key} 不在源域：开关会因此需要一次换域烘焙，                     而换域必然有损、有损就必须有闸、有闸就必然出现「开关无效」"
                );
            }
        }
    }

    /// **拨动「锁定参数线」只改元数据，一格数据都不动，而且可逆。**
    ///
    /// 【为什么这是第 4 点的回归测试】旧实现靠一次性烘焙：工程域的格号是绝对工程帧，
    /// 一条源窗只覆盖工程时间的一段 ⇒ 换域必然变短 ⇒ 必须有闸 ⇒ 闸拒绝时开关无效。
    /// 用户对它的原话正是"完全没反应"，现场日志里 `18026 -> 11824` 就是那次截断。
    ///
    /// 现在开关只换**投影用的映射**：拨过去曲线当场停在工程位置，拨回来逐帧回到原样。
    #[test]
    fn toggling_the_lock_moves_curves_on_the_next_projection_and_is_reversible() {
        let ids = identities();
        let mut atlas = ParameterAtlas::default();
        atlas.seed_test_pitch_curve("mod:mod", 250., vec![60., 62., 64., 66.]);
        let original = pitch_edit_values(&atlas.authorities["mod:mod"]);
        assert_eq!(original, vec![60., 62., 64., 66.], "锁定 ⇒ 源域");

        // 锁定状态下，片段从 [1,2) 移到 [3,4)：曲线跟着走。
        let locked = atlas.project(&host(3., 1., 0., 1.), &ids).unwrap();
        assert_eq!(&locked["clip"].pitch_edit[12..16], &[60., 62., 64., 66.]);

        // 拨到"不锁定"：只改元数据（冻结几何 = 此刻的 [1,2)），数据不动。
        let timeline = host(1., 1., 0., 1.);
        let changed = atlas.set_placement(
            &timeline,
            &ids,
            &Placement::PinProject { frozen: Vec::new() },
        );
        assert_eq!(changed, 1, "必须有一条权威被改");
        assert_eq!(
            pitch_edit_values(&atlas.authorities["mod:mod"]),
            original,
            "拨开关不得改动任何一个采样"
        );

        // 同一份数据、同一段工程时间：曲线留在原处，移走之后新位置是空的。
        let pinned = atlas.project(&host(3., 1., 0., 1.), &ids).unwrap();
        assert_eq!(
            &pinned["clip"].pitch_edit[12..16],
            &[0., 0., 0., 0.],
            "不锁定时参数不随片段移动"
        );
        let stayed = atlas.project(&host(1., 1., 0., 1.), &ids).unwrap();
        assert_eq!(&stayed["clip"].pitch_edit[4..8], &[60., 62., 64., 66.]);

        // 拨回锁定：逐帧回到原样（可逆）。
        assert_eq!(
            atlas.set_placement(&timeline, &ids, &Placement::FollowMaterial),
            1
        );
        let back = atlas.project(&host(1., 1., 0., 1.), &ids).unwrap();
        assert_eq!(&back["clip"].pitch_edit[4..8], &[60., 62., 64., 66.]);
        assert_eq!(
            pitch_edit_values(&atlas.authorities["mod:mod"]),
            original,
            "来回拨开关必须逐帧回到原样"
        );
    }

    /// **投影必须把"权威持有音高曲线"翻译成内核的"用户数据"标记。**
    ///
    /// 【为什么这是一条独立的重置路径】内核 `ensure_params_for_root` 在
    /// `build_root_pitch_key` 变化时清 `pitch_edit`，而那个 key 把每条 clip 的
    /// id / start / length / 倍率都喂进哈希 ⇒ **任何一次分割都会改变它**。图集修好了
    /// 但这里不标记，GUI 上仍然会看到"参数没了"。
    #[test]
    fn projecting_an_authority_pitch_curve_marks_it_as_user_data() {
        let ids = identities();
        let original = edited();
        let atlas = ParameterAtlas::default().capture(&original, &ids).unwrap();
        let roots = atlas.project_roots(&original, &ids).unwrap();
        assert!(
            roots["a"].pitch_edit_user_modified,
            "根视图必须标记用户音高数据，否则内核会把它当陈旧分析结果清掉"
        );
        let local = atlas.project_local(&original, &ids).unwrap();
        assert!(
            local["clip"].pitch_edit_user_modified,
            "渲染输入同样必须带这个标记"
        );
    }

    /// **播种是幂等的、且拒绝无效对。**
    #[test]
    fn seeding_a_cloned_identity_copies_the_parent_parameters_once() {
        let atlas = ParameterAtlas::default()
            .capture(&edited(), &identities())
            .unwrap();
        let mut seeded = atlas.clone();
        assert!(seeded.seed_authority("mod:mod", "mod:clone"));
        assert!(seeded.has_authority_parameters(None, "clone"));
        // 已经存在 ⇒ 不覆盖（P1 播过之后 P2 再来一次是常态）。
        assert!(!seeded.seed_authority("mod:mod", "mod:clone"));
        // 父不存在 / 同键（宿主复用父身份 = 别名）/ 空键 ⇒ 都不播种。
        assert!(!seeded.seed_authority("mod:missing", "mod:x"));
        assert!(!seeded.seed_authority("mod:mod", "mod:mod"));
        assert!(!seeded.seed_authority("", "mod:y"));
        assert!(!seeded.seed_authority("mod:mod", ""));
    }

    /// **persistentID 改名必须把权威一起搬走。**
    ///
    /// 【为什么这是一条独立的重置路径】`update_audio_modification` 会把
    /// `persistent_id` 换成宿主给的新值，而权威键正是 `mod:{persistentID}`。
    /// 不搬 ⇒ 曲线被孤儿化，下一次投影查新键必然落空。
    #[test]
    fn remapping_a_modification_id_carries_its_authority() {
        let mut atlas = ParameterAtlas::default();
        atlas.seed_test_pitch_curve("mod:old", 250., vec![60., 61.]);
        atlas.remap_authorities(&[("old".into(), "new".into())]);
        assert!(!atlas.authorities.contains_key("mod:old"), "旧键必须消失");
        assert!(
            atlas.has_authority_parameters(None, "new"),
            "曲线必须跟着新 ID"
        );
        // 恒等映射（整份存档恢复用的就是它）不得改动任何东西。
        let before = atlas.authorities.clone();
        atlas.remap_authorities(&[("new".into(), "new".into())]);
        assert_eq!(atlas.authorities.len(), before.len());
    }

    /// **孤儿权威只在宽限期之后被回收，且只回收那一条。**
    ///
    /// 【为什么需要回收】反复"分割 → 撤销"会持续产生新 modification ID，孤儿权威
    /// 只增不减，最终撞上 64MiB 上限 ⇒ **整个存档写不出去**（用户的编辑全丢）。
    /// 【为什么必须有宽限期】撤销会把同一个 persistentID 带回来，立刻删就是丢数据。
    #[test]
    fn an_orphan_authority_is_reclaimed_only_after_the_grace_period() {
        let mut atlas = ParameterAtlas::default();
        atlas.seed_test_pitch_curve("mod:gone", 250., vec![60., 61.]);
        atlas.seed_test_pitch_curve("mod:live", 250., vec![62., 63.]);
        let live = std::collections::BTreeSet::from(["live".to_owned()]);
        for round in 0..ORPHAN_GRACE_ROUNDS - 1 {
            assert_eq!(
                atlas.reclaim_orphans(&live),
                0,
                "宽限期内不得回收 (round {round})"
            );
        }
        assert!(atlas.authorities.contains_key("mod:gone"));
        assert_eq!(atlas.reclaim_orphans(&live), 1, "超过宽限期才回收");
        assert!(!atlas.authorities.contains_key("mod:gone"));
        assert!(
            atlas.authorities.contains_key("mod:live"),
            "在文档图里的那一条不得被回收"
        );
    }

    /// **只要 modification 还在文档图里，宽限期就持续清零。**
    #[test]
    fn a_live_authority_never_ages_out() {
        let mut atlas = ParameterAtlas::default();
        atlas.seed_test_pitch_curve("mod:live", 250., vec![60.]);
        let live = std::collections::BTreeSet::from(["live".to_owned()]);
        for _ in 0..(ORPHAN_GRACE_ROUNDS * 3) {
            assert_eq!(atlas.reclaim_orphans(&live), 0);
        }
        assert!(atlas.authorities.contains_key("mod:live"));
    }

    /// **旧形状（第 8 代之前）的曲线照旧读得进、且投影正确。**
    ///
    /// 【为什么这条必须存在】第 8 代把 `basis` 从对象改成 `null`。旧存档里它是一个
    /// `RegionGeometry`，必须被解析成 `Some(..)` 并走**旧索引分支** —— 否则升级一次
    /// 就丢光全部参数。这条同时钉住"两条索引分支共存"这个设计。
    #[test]
    fn a_legacy_project_frame_curve_still_projects_after_the_domain_change() {
        let mut atlas = ParameterAtlas::default();
        let geometry = RegionGeometry {
            project_start: 1.,
            project_duration: 1.,
            source_start: 0.,
            source_duration: 1.,
            reversed: false,
        };
        let mut curves = BTreeMap::new();
        curves.insert(
            "pitch_edit".to_owned(),
            SourceCurve {
                // 旧形状：`basis == Some` 且域为源域 —— 这正是第七轮的读法
                // （`domain_time` 的 `Some` 分支）。`legacy_curve_domain()` 对
                // 缺 `domain` 字段的旧存档给的就是这个组合，所以这里显式写出来
                // 与真实升级路径逐字对齐。
                basis: Some(geometry),
                domain: CurveDomain::Source,
                first_frame: 4,
                frame_ms: 250.,
                values: Arc::new(vec![60., 62., 64., 66., 68.]),
                reservation: None,
            },
        );
        atlas.authorities.insert(
            "mod:mod".into(),
            AuthorityParameters {
                edits: BTreeMap::new(),
                source: String::new(),
                curves,
                template: TrackParamsState {
                    frame_period_ms: 250.,
                    ..Default::default()
                },
                placement: Placement::FollowMaterial,
                last_edit_seq: 0,
                unseen_rounds: 0,
                refused_rounds: 0,
            },
        );
        // 往返即升级路径：写出的 `basis` 是对象，读回来必须是 `Some`（旧分支）。
        let bytes = serde_json::to_vec(&atlas).unwrap();
        let restored: ParameterAtlas = serde_json::from_slice(&bytes).unwrap();
        assert!(
            restored.authorities["mod:mod"].curves["pitch_edit"]
                .basis
                .is_some(),
            "旧形状必须读成 `Some(..)`，否则会走错索引分支"
        );
        let ids = identities();
        let view = restored.project(&host(1., 1., 0., 1.), &ids).unwrap();
        assert_eq!(&view["clip"].pitch_edit[4..9], &[60., 62., 64., 66., 68.]);
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

    // ─────────────────────────────────────────────────────────────────────────
    // 第八轮：分割归属只由源窗决定（plan §B3）
    //
    // 这三条是本轮 D 组的"先红"断言。它们钉的都是**用户报障的形状**：
    // 一条 clip 被分割成两半，两半共享同一个 modification persistentID
    // （Cubase 实测从不调用 `cloneAudioModification`，见 round7 §1.4）。
    // ─────────────────────────────────────────────────────────────────────────

    /// 一条 clip 在 `[0, 2)` 处被切成两半，两半的 `modification` 相同
    /// （阿拉伯-only 宿主的常态）。权威曲线按**源帧**存一条，
    /// 两个半段必须各自读到自己的那一半，且**互不串段**。
    ///
    /// 【为什么必须是子序列而不是整份】源域曲线只有一条（第七轮起
    /// `capture_source_curve` 一律写 `basis: None`，键是 modification 级）。
    /// 两个半段共享同一个键 ⇒ 共享同一条曲线；各自的源窗是父段源窗的两个
    /// 不相交子区间 ⇒ 投影天然各取所需。若某个半段读到对方的帧，
    /// 就是"曲线整体平移一个分割长度"（round6 §1.1 记的原始报障）。
    #[test]
    fn two_halves_sharing_one_modification_read_disjoint_source_subsequences() {
        let ids = identities();
        // 父段：工程 [0, 2)，源窗 [0, 2)，每 5 ms 一帧 ⇒ 401 帧。
        let mut parent = host(0., 2., 0., 2.);
        let frames = 401_usize;
        // 源帧上的特征值：60, 61, 62, ...（第 k 帧 = 60 + k%48），
        // 让"读错半段"立刻在数值上看得见。
        let ramp: Vec<f32> = (0..frames).map(|k| 60. + (k % 48) as f32).collect();
        parent.project_sec = 2.;
        parent.params_by_root_track.insert(
            "a".into(),
            TrackParamsState {
                frame_period_ms: 5.,
                pitch_edit_user_modified: true,
                // 夹具目标曲线与原始分析值不同；否则相等即“未编辑”（I10）。
                pitch_orig: vec![0.; frames],
                pitch_edit: ramp.clone(),
                ..Default::default()
            },
        );
        let atlas = ParameterAtlas::default().capture(&parent, &ids).unwrap();
        // 一条源域曲线覆盖整条源窗。
        let authority = atlas
            .authorities
            .get("mod:mod")
            .expect("权威按 modification persistentID 入库");
        assert!(
            authority.curves.is_empty(),
            "新权威只存稀疏 edits，不再写整条旧曲线"
        );
        assert!(authority.edits.contains_key("pitch_edit"));

        // 左半段：工程 [0, 1)，源窗 [0, 1)。右半段：工程 [1, 2)，源窗 [1, 2)。
        // 两半的 source / modification 与父段完全相同 —— 这就是 Cubase 的形状。
        //
        // 【为什么两半各用一次 `host()`】`host()` 会把 take 的源窗镜像到 clip 顶层的
        // `source_start_sec` / `playback_rate` 上（`normalize_takes` → `apply_to_clip`），
        // 而 `geometry()` 读的正是**顶层**那两个字段。直接 clone 再改 `takes[0]`
        // 会留下顶层旧值 —— 那样右半段的源窗仍是 [0,1)，本测试就什么也没验证。
        let mut left = host(0., 1., 0., 1.);
        left.clips[0].id = "left".into();
        let mut right = host(1., 1., 1., 1.);
        right.clips[0].id = "right".into();
        let right_clip = right.clips[0].clone();
        let mut halves = left;
        halves.clips.push(right_clip);
        halves.project_sec = 2.;

        let mut half_ids = BTreeMap::new();
        for (id, key) in [("left", 41_u64), ("right", 42_u64)] {
            half_ids.insert(
                id.into(),
                RegionIdentity {
                    key,
                    item: None,
                    source: "source".into(),
                    modification: "mod".into(),
                    signature: String::new(),
                },
            );
        }

        let followed = atlas.follow_geometry(&halves, &half_ids).unwrap();
        let local = followed.project_local(&halves, &half_ids).unwrap();

        let left_curve = &local["left"].pitch_edit;
        let right_curve = &local["right"].pitch_edit;
        // 1 s @ 5 ms = 200 帧（`frame_range` 含哨兵）。
        assert_eq!(left_curve.len(), 201, "左半段含一个哨兵帧");
        assert_eq!(right_curve.len(), 201, "右半段含一个哨兵帧");
        // 左半段 = 源帧 0..200；右半段 = 源帧 200..400。
        assert_eq!(
            &left_curve[..200],
            &ramp[..200],
            "左半段必须是源序列的前半段"
        );
        assert_eq!(
            &right_curve[..200],
            &ramp[200..400],
            "右半段必须是源序列的后半段"
        );
        assert_ne!(
            &left_curve[..200],
            &right_curve[..200],
            "两个半段不得互为副本：那正是'曲线整体平移一个分割长度'"
        );
    }

    /// **「锁定参数线」不得改变分割结果**。
    ///
    /// 【为什么这是必须的】独立 App 的 `split_clip`（`state/model.rs:10456-10500`）
    /// 全程不读 `params_by_root_track`，也不读 `lock_param_lines` —— 分割不改变任何
    /// 参数的位置语义：左半段天然读到前半段、右半段天然读到后半段，两段都对。
    /// 开关只解释"移动 / 裁剪 / 拉伸"。插件侧若在分割上分叉，就是**新增**了一种
    /// App 没有的行为。
    ///
    /// 【为什么现在能直接断言"数据一格不动"】开关只换投影映射，不换存储域
    /// （见 `Placement`）。旧实现靠烘焙，而烘焙是**逐 clip** 的：同一个 modification
    /// 的两个半段各按自己的几何烘焙一次，第二次会把第一次的结果再搬一遍 —— 那正是
    /// "分割后参数完全重置"的形状。
    #[test]
    fn the_lock_setting_does_not_change_how_a_split_divides_curves() {
        let ids = identities();
        let mut parent = host(0., 2., 0., 2.);
        let ramp: Vec<f32> = (0..401).map(|k| 60. + (k % 48) as f32).collect();
        parent.project_sec = 2.;
        parent.params_by_root_track.insert(
            "a".into(),
            TrackParamsState {
                frame_period_ms: 5.,
                pitch_edit_user_modified: true,
                // 夹具目标曲线与原始分析值不同；否则相等即“未编辑”（I10）。
                pitch_orig: vec![0.; ramp.len()],
                pitch_edit: ramp.clone(),
                ..Default::default()
            },
        );
        let atlas = ParameterAtlas::default().capture(&parent, &ids).unwrap();

        let mut halves = host(0., 1., 0., 1.);
        halves.clips[0].id = "left".into();
        let mut right = host(1., 1., 1., 1.);
        right.clips[0].id = "right".into();
        let right_clip = right.clips[0].clone();
        halves.clips.push(right_clip);
        halves.project_sec = 2.;
        let mut half_ids = BTreeMap::new();
        for (id, key) in [("left", 41_u64), ("right", 42_u64)] {
            half_ids.insert(
                id.into(),
                RegionIdentity {
                    key,
                    item: None,
                    source: "source".into(),
                    modification: "mod".into(),
                    signature: String::new(),
                },
            );
        }

        let followed = atlas.follow_geometry(&halves, &half_ids).unwrap();
        let locked = followed.project_local(&halves, &half_ids).unwrap();
        // 【分割的划分】两个半段各读**自己那一段源**：左段从源 0 起（斜坡第 0 格 60），
        // 右段从源 1 s 起（5 ms 一格 ⇒ 第 200 格 ⇒ `60 + 200 % 48 = 68`）。
        assert_eq!(locked["left"].pitch_edit[0], 60., "左半段读源窗前半段");
        assert_eq!(locked["right"].pitch_edit[0], 68., "右半段读源窗后半段");

        let before = followed.authorities["mod:mod"].curves.clone();
        let mut unlocked = followed;
        unlocked.set_placement(
            &halves,
            &half_ids,
            &Placement::PinProject { frozen: Vec::new() },
        );
        assert!(
            unlocked.authorities["mod:mod"]
                .curves
                .iter()
                .all(|(key, curve)| same_curve(curve, &before[key])),
            "拨开关不得改动任何一个采样"
        );
        // 拨回来：逐帧回到分割后的原样（可逆）。这是"开关只换映射"的直接推论，
        // 也是旧实现（烘焙）做不到的那一条 —— 烘焙一旦截断就再也回不来。
        unlocked.set_placement(&halves, &half_ids, &Placement::FollowMaterial);
        let back = unlocked.project_local(&halves, &half_ids).unwrap();
        for clip in ["left", "right"] {
            assert_eq!(
                locked[clip].pitch_edit, back[clip].pitch_edit,
                "来回拨开关必须逐帧回到原样（clip {clip}）"
            );
        }
    }

    /// **钉住的曲线在拉伸过的 clip 上必须逐格还原。**
    ///
    /// 【为什么这条是本轮的回归】放置策略决定"用哪份几何把工程时刻折成源秒"。
    /// 一旦折算错了（用当前几何去读一份按冻结几何放置的曲线），越到后段偏得越多 ——
    /// 用户看到的就是"关掉锁定之后那条线整体歪了"。
    ///
    /// 【为什么曲线不换域】它永远按源秒存（见 `Placement`）。这条用例钉的是
    /// "同一个数组在 2× 拉伸的 clip 上经冻结几何折回工程时间后逐格正确"。
    #[test]
    fn a_pinned_curve_round_trips_through_a_stretched_clip() {
        let ids = identities();
        let frame_ms = 250.;
        // 2× 拉伸：工程 1 s、源 2 s。
        let timeline = host(1., 1., 0., 2.);
        let frozen = RegionGeometry {
            project_start: 1.,
            project_duration: 1.,
            source_start: 0.,
            source_duration: 2.,
            reversed: false,
        };
        // 源 0..2 s 每 250 ms 一格：60, 62, …（2× 拉伸下每两个源格落在一个工程格上）。
        let mut atlas = ParameterAtlas::default();
        atlas.seed_test_pitch_curve_placed(
            "mod:mod",
            frame_ms,
            (0..8).map(|index| 60. + index as f32).collect(),
            Placement::pinned(frozen),
        );
        let projected = atlas.project(&timeline, &ids).expect("投影必须成功");
        let clip = &projected["clip"];
        // 工程帧 f 对应工程时刻 f·0.25 s；冻结几何把工程 [1,2) 映到源 [0,2)
        // （2× 拉伸 ⇒ 一个工程格 = 两个源格）：
        //   f=4 ⇒ 源 0.00 ⇒ 第 0 格 60     f=5 ⇒ 源 0.50 ⇒ 第 2 格 62
        //   f=6 ⇒ 源 1.00 ⇒ 第 4 格 64     f=7 ⇒ 源 1.50 ⇒ 第 6 格 66
        //   f=8 ⇒ 源 2.00 ⇒ 界外（数组只到第 7 格）⇒ 默认值
        assert_eq!(
            &clip.pitch_edit[4..=8],
            &[60., 62., 64., 66., 0.],
            "冻结几何必须把 2× 拉伸逐格折回来，实测 {:?}",
            &clip.pitch_edit[4..=8]
        );

        // 片段移走后，钉住的曲线留在工程 [1,2)：新位置读出默认值。
        let moved = atlas
            .project(&host(3., 1., 0., 2.), &ids)
            .expect("投影必须成功");
        assert_eq!(
            &moved["clip"].pitch_edit[12..16],
            &[0., 0., 0., 0.],
            "钉住的曲线不随片段移动"
        );
    }

    /// **派生写入（没有基线）仍然不许让"非默认帧"变少。**
    ///
    /// 【为什么这条闸保留】`capture` 的语义是"把当前时间线整份采下来"，此时
    /// `prior` 退化成 `pad`：任何"减少"都**没有证据**（谁也没说这些帧该变成默认值）。
    /// 派生写入（几何/域变化触发）走的就是这条路 —— 它必须只能增加，不能抹平。
    ///
    /// 【与下一条的分工】用户删除由「按帧证据」放行（下一条用例）；这条只钉住
    /// "没有证据时仍然拒绝"。
    #[test]
    fn a_derived_write_still_refuses_to_erase_frames() {
        let ids = identities();
        let frame_ms = 250.;
        let mut atlas = ParameterAtlas::default();
        // 权威里有一条**远多于**本 region 覆盖范围的用户曲线。
        atlas.seed_test_pitch_curve(
            "mod:mod",
            frame_ms,
            (0..16).map(|index| 60. + index as f32).collect(),
        );
        let before = edit_samples(&atlas.authorities["mod:mod"]);
        assert!(before > 4, "夹具必须有一条足够长的曲线：{before}");
        // 视图里这条 clip 的整段都是默认值（0），且**没有基线**。
        let mut timeline = host(1., 1., 0., 1.);
        let mut view = TrackParamsState {
            frame_period_ms: frame_ms,
            ..Default::default()
        };
        view.pitch_edit = vec![0.; 8];
        timeline
            .params_by_root_track
            .insert("a".to_owned(), view.clone());
        let after_atlas = atlas.capture(&timeline, &ids).expect("捕获必须成功");
        let after = edit_samples(&after_atlas.authorities["mod:mod"]);
        assert_eq!(
            after, before,
            "没有基线时任何减少都没有证据，必须被拒绝：before={before} after={after}"
        );
    }

    /// **用户擦除是一次有效编辑：写回、且下一次投影仍然看得见。**
    ///
    /// 【为什么这是第 2 点的回归测试】旧判据是"非默认帧数只能增不能减"，而这条路径上
    /// 每一次调用都是用户命令 —— 于是**任何用户删除都被拒绝**：权威保持旧曲线、
    /// 渲染输入没变（音频一个采样都没动）、下一次投影再把视图覆盖回来（切屏后回退）。
    /// 三步与用户描述逐字对应。
    ///
    /// 现在的判据是**证据**：删掉一帧必须同时满足"当前视图在这一帧是默认值"与
    /// "基线在这一帧不是默认值"。用户擦除每一帧都满足它。
    #[test]
    fn a_user_erase_is_written_and_survives_the_next_projection() {
        let ids = identities();
        let frame_ms = 250.;
        let mut atlas = ParameterAtlas::default();
        // 曲线覆盖源 0..3.75 s；clip 只显示源 0..1 s（工程 1..2 s ⇒ 帧 4..8）。
        atlas.seed_test_pitch_curve(
            "mod:mod",
            frame_ms,
            (0..16).map(|index| 60. + index as f32).collect(),
        );
        let before = edit_samples(&atlas.authorities["mod:mod"]);
        assert_eq!(before, 16, "夹具：16 个非默认帧");
        let mut timeline = host(1., 1., 0., 1.);
        let projected = atlas.project(&timeline, &ids).expect("投影必须成功");
        let baseline = BTreeMap::from([("a".to_owned(), projected["clip"].clone())]);
        assert_eq!(
            &projected["clip"].pitch_edit[4..=8],
            &[60., 61., 62., 63., 64.],
            "夹具：窗口内五格来自源曲线"
        );
        // 用户用右键拖拽把窗口内整段擦回初始化。
        let mut erased = projected["clip"].clone();
        for frame in 4..=8 {
            erased.pitch_edit[frame] = 0.;
        }
        timeline.params_by_root_track.insert("a".to_owned(), erased);
        let mut report = CaptureReport::default();
        let after_atlas = atlas
            .capture_changes_reporting(&timeline, &ids, &baseline, &mut report)
            .expect("捕获必须成功");
        let after = edit_samples(&after_atlas.authorities["mod:mod"]);
        assert_eq!(
            after,
            before - 5,
            "擦除必须真的写回（窗口内五格）：{before} -> {after}"
        );
        assert_eq!(report.removed_frames, 5, "读数必须如实记下被删的帧数");
        assert!(
            !report.has_refusals(),
            "用户擦除不该被拒绝：{}",
            report.summary()
        );
        // 下一次投影：视图与权威一致（不会"切屏之后又回来"）。
        let reprojected = after_atlas.project(&timeline, &ids).expect("投影必须成功");
        // 【为什么只看 4..=7】音高这一路的插值有一条规矩：一侧为 0 时取另一侧的值
        // （0 表示"没有音高"）。窗口末格正好落在"已擦除"与"窗外第一个未擦除格"之间，
        // 于是它读出那个未擦除格的值 —— 那是插值语义，不是写回失败。
        assert_eq!(
            &reprojected["clip"].pitch_edit[4..=7],
            &[0., 0., 0., 0.],
            "重投影必须看到擦除后的样子"
        );
        // 窗口外的帧一格都不能少（擦除是局部的）。
        assert_eq!(
            edit_samples(&after_atlas.authorities["mod:mod"]),
            before - 5,
            "窗口外 11 格必须原样保留"
        );
    }

    /// **没有证据的减少仍然被拒绝，而且这次拒绝是看得见的。**
    ///
    /// 【什么叫"没有证据"】权威里有曲线，但基线里**没有这条数组**（投影范围变窄、
    /// **没有基线时，一次"全默认"的视图不会擦掉任何东西。**
    ///
    /// 【为什么这是 U-4 的正解】权威里有曲线、基线里**没有这条数组**（投影范围变窄、
    /// 根的键换了、或用户刚打开窗口）时，"当前是默认值"这一条成立，"基线不是默认值"
    /// 这一条不成立 —— 无法证明是用户删的。处置是：把基线当成**空数组**（只允许增加），
    /// 于是这一轮一帧都不写，曲线原样保留，并把读数交出去。
    #[test]
    fn a_view_without_a_baseline_never_erases_a_curve() {
        let ids = identities();
        let frame_ms = 250.;
        let mut atlas = ParameterAtlas::default();
        atlas.seed_test_pitch_curve(
            "mod:mod",
            frame_ms,
            (0..16).map(|index| 60. + index as f32).collect(),
        );
        let before = edit_samples(&atlas.authorities["mod:mod"]);
        assert!(before > 4, "夹具必须有一条非默认曲线：{before}");
        let mut timeline = host(1., 1., 0., 1.);
        // 基线里**有这个根**、但没有 `pitch_edit` 这条数组。
        let baseline = BTreeMap::from([(
            "a".to_owned(),
            TrackParamsState {
                frame_period_ms: frame_ms,
                ..Default::default()
            },
        )]);
        let mut wiped = TrackParamsState {
            frame_period_ms: frame_ms,
            ..Default::default()
        };
        wiped.pitch_edit = vec![0.; 9];
        timeline.params_by_root_track.insert("a".to_owned(), wiped);
        let mut report = CaptureReport::default();
        let after_atlas = atlas
            .capture_changes_reporting(&timeline, &ids, &baseline, &mut report)
            .expect("捕获必须成功");
        let after = edit_samples(&after_atlas.authorities["mod:mod"]);
        assert_eq!(
            after, before,
            "没有基线时不得改动任何一格：{before} -> {after}"
        );
        assert_eq!(
            report.no_baseline_curves, 1,
            "读数必须记下「没有基线」这件事"
        );
        assert_eq!(report.removed_frames, 0, "这一轮不该有任何帧被删");
        assert!(
            after_atlas.authorities["mod:mod"]
                .edits
                .contains_key("pitch_edit"),
            "曲线仍然是源域稀疏编辑（数据与策略同源）"
        );
    }

    /// **旧存档形状（工程帧索引）的曲线在"计数减少"时仍被记成一次无证据的删除。**
    ///
    /// 【为什么这条单独钉】旧形状的格号不是源帧，无法参与按帧证据判定。对它们退回
    /// "非默认帧数不许减少"的旧判据 —— 那是升级窗口里唯一安全的解释，而它必须仍然
    /// 拒绝、并且**被记下来**（否则又回到"静默丢弃"）。
    #[test]
    fn a_legacy_curve_losing_frames_is_counted_as_an_unexplained_removal() {
        let frame_ms = 250.;
        let mut legacy = ParameterAtlas::default();
        legacy.seed_legacy_project_curve(
            "mod:mod",
            frame_ms,
            1.,
            (0..8).map(|index| 60. + index as f32).collect(),
        );
        let old = legacy.authorities["mod:mod"].curves["pitch_edit"].clone();
        // 同一份曲线，只剩三格非默认。
        let mut shortened = old.clone();
        shortened.values = Arc::new(vec![60., 61., 62., 0., 0., 0., 0., 0.]);
        let current = RegionGeometry {
            project_start: 1.,
            project_duration: 1.,
            source_start: 0.,
            source_duration: 1.,
            reversed: false,
        };
        let map = current.map().unwrap();
        let unexplained = unexplained_removals(
            "pitch_edit",
            Some(&old),
            Some(&shortened),
            None,
            &[],
            &map,
            frame_ms,
        );
        assert_eq!(
            unexplained, 5,
            "旧形状只能按计数判：8 格 -> 3 格 ⇒ 5 格没有证据"
        );
    }

    #[test]
    fn first_capture_stores_only_explicit_edits_and_never_derived_curves() {
        let atlas = ParameterAtlas::default()
            .capture(&edited(), &identities())
            .unwrap();
        let authority = &atlas.authorities["mod:mod"];
        assert!(authority.curves.is_empty(), "新权威不再写整条 SourceCurve");
        assert!(authority.edits.contains_key("pitch_edit"));
        assert!(authority.edits.contains_key("extra:volume"));
        for key in ["pitch_orig", "dyn_orig", "tension_orig"] {
            assert!(
                !authority.edits.contains_key(key),
                "派生数组绝不能进入编辑权威：{key}"
            );
        }
    }

    #[test]
    fn volume_only_capture_does_not_create_pitch_authority() {
        let mut timeline = host(0., 1., 0., 1.);
        let mut params = TrackParamsState {
            frame_period_ms: 250.,
            pitch_orig: vec![57.; 4],
            pitch_edit: vec![57.; 4],
            ..Default::default()
        };
        params
            .extra_curves
            .insert("volume".into(), vec![1.0, 0.5, 0.8, 1.0]);
        timeline.params_by_root_track.insert("a".into(), params);
        let atlas = ParameterAtlas::default()
            .capture(&timeline, &identities())
            .unwrap();
        let authority = &atlas.authorities["mod:mod"];
        assert!(
            !authority.edits.contains_key("pitch_edit"),
            "只画 volume 不得创建 pitch 权威"
        );
        assert!(!authority.has_pitch_edits());
        let root = atlas.project_roots(&timeline, &identities()).unwrap();
        assert!(!root["a"].pitch_edit_user_modified);
    }

    #[test]
    fn locked_move_uses_new_pitch_orig_for_unedited_frames() {
        let mut atlas = ParameterAtlas::default();
        atlas.seed_test_pitch_curve("mod:mod", 250., vec![60., 62.]);
        let mut timeline = host(1., 2., 0., 2.);
        timeline.project_sec = 3.;
        timeline.params_by_root_track.insert(
            "a".into(),
            TrackParamsState {
                frame_period_ms: 250.,
                pitch_orig: vec![70.; 13],
                ..Default::default()
            },
        );
        let projected = atlas.project(&timeline, &identities()).unwrap();
        let pitch = &projected["clip"].pitch_edit;
        assert_eq!(&pitch[4..6], &[60., 62.], "编辑段跟随素材");
        assert_eq!(pitch[8], 70., "未编辑帧取新的 pitch_orig，而不是旧快照");
        assert_eq!(pitch[12], 70.);
    }

    #[test]
    fn unlocked_move_keeps_edits_at_project_positions() {
        let mut atlas = ParameterAtlas::default();
        atlas.seed_test_pitch_curve_placed(
            "mod:mod",
            250.,
            vec![60., 62.],
            Placement::pinned(RegionGeometry {
                project_start: 1.,
                project_duration: 1.,
                source_start: 0.,
                source_duration: 1.,
                reversed: false,
            }),
        );
        let mut timeline = host(3., 1., 0., 1.);
        timeline.project_sec = 4.;
        timeline.params_by_root_track.insert(
            "a".into(),
            TrackParamsState {
                frame_period_ms: 250.,
                pitch_orig: vec![70.; 17],
                ..Default::default()
            },
        );
        let root = atlas.project_roots(&timeline, &identities()).unwrap();
        assert_eq!(root["a"].pitch_edit[4], 60., "不锁定时编辑留在原工程位置");
        assert_eq!(root["a"].pitch_edit[5], 62.);
        assert_eq!(root["a"].pitch_edit[12], 70., "新位置显示新的原始值");
    }

    #[test]
    fn repeated_projection_is_idempotent() {
        let atlas = ParameterAtlas::default()
            .capture(&edited(), &identities())
            .unwrap();
        let first = atlas.project(&host(1., 1., 0., 1.), &identities()).unwrap();
        let second = atlas.project(&host(1., 1., 0., 1.), &identities()).unwrap();
        assert_eq!(first["clip"].pitch_edit, second["clip"].pitch_edit);
        assert_eq!(first["clip"].extra_curves, second["clip"].extra_curves);
    }

    #[test]
    fn take_switch_with_a_different_source_does_not_project_the_old_authority() {
        let atlas = ParameterAtlas::default()
            .capture(&edited(), &identities())
            .unwrap();
        let mut ids = identities();
        if let Some(identity) = ids.get_mut("clip") {
            identity.source = "other-take-source".into();
        }
        let mut timeline = host(1., 1., 0., 1.);
        timeline.clips[0].source_path = Some("other-take-source".into());
        let projected = atlas.project(&timeline, &ids).unwrap();
        assert!(
            projected.is_empty(),
            "item GUID 未变但 source 变了：旧 take 的权威不得投影到新 take"
        );
    }
}
