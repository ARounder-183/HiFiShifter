//! 本实例在宿主里**挂在哪里**：某条轨道的 FX 链、某条 take 的 FX 链、还是没有。
//!
//! # 为什么必须把它显式建模
//!
//! REAPER 的 ARA 支持两种插入位置：轨道 FX（`MediaTrack` 的 FX 链）与媒体项 take FX
//! （`MediaItem_Take` 的 FX 链）。插件侧的对象模型从第一天起就是"文档 = 工程、
//! region sequence ≈ 轨道、playback region ≈ clip"，而轨道 FX 下这三层恰好与宿主的
//! 对象一一对应 —— 于是所有"找宿主对象"的代码都可以只问一句"我的轨道是谁"。
//!
//! take FX 打破了这条对应：一个实例只拥有**一条 take** 的音频，而同一轨道上可以有任意
//! 多条 take 各有自己的实例。此时"我的轨道"是 item 的**宿主轨**，"我的宿主对象"是
//! **一条 take**，两者不再等价。凡是拿前者代替后者的地方（旁通、清单、导入、建轨、
//! 写回）都会指错对象，而且**查询失败会被折成 `Unknown` 或静默跳过** —— 功能看起来
//! 存在，却从不生效，日志里也没有一行能指认位置。这个模块就是那个"位置"的唯一来源。
//!
//! # 判定顺序
//!
//! 先用强证据，再退回弱证据：
//!
//! 1. `parent(2)` 是一条经 `ValidatePtr2` 核实的 take，且它的 `TakeFX_*` 链里有本插件
//!    ⇒ take 侧候选；
//! 2. `parent(1)` 是一条经核实的轨道，且它的 `TrackFX_*` 链里有本插件 ⇒ 轨道侧候选；
//! 3. 两条都不成立 ⇒ [`HostPlacement::Unbound`]（行为与改动前完全一致：不改写、不门控）。
//!
//! 两条都成立时按 [`resolve`] 的歧义规则裁决 —— 见那里的说明。
//!
//! # 为什么每次绑定后重算
//!
//! 宿主可能在工程重开时把同一个处理器实例复用到另一份文档（`rebind_generation` 计数
//! 的就是这件事），放置位置也可能随之改变；同一个实例甚至可能先作为轨道 FX 插入、
//! 之后被用户剪到某条 take 上。缓存一个永不失效的 `placement` 会把第一次的结论带到
//! 下一次，而那次错误的后果是"take 实例去读轨道链"（回到今天的 bug）。

use super::{checked, fx_state, HostVtbl, NativeHost};
use std::ffi::{c_void, CStr};

/// 本实例在宿主里挂在哪里。
///
/// 指针以 `usize` 保存（与 `HostClipTarget` 同一条纪律）：它要跨线程读，而裸指针不是
/// `Send`。每次使用前都由调用方经宿主自己的 `ValidatePtr2` 复核归属。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum HostPlacement {
    /// FX 在某条轨道的 FX 链上；ARA 内容覆盖这条轨（及其 folder 后代）的 item。
    TrackFx { track: usize },
    /// FX 在某条 take 的 FX 链上；ARA 内容只覆盖这一条 take。
    TakeFx {
        track: usize,
        item: usize,
        take: usize,
    },
    /// 没有原生宿主（ARA-only / Capture）或尚未求值。
    ///
    /// 【为什么它是一个**一等**取值而不是 `Option::None`】"没有位置"必须与"位置是轨道"
    /// 区分开：前者要回到改动前的全部行为（不门控、不改写），后者要按轨道规则收窄。
    #[default]
    Unbound,
}

impl HostPlacement {
    /// 稳定的语言无关编码（日志与诊断包用）。
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::TrackFx { .. } => "track-fx",
            Self::TakeFx { .. } => "take-fx",
            Self::Unbound => "unbound",
        }
    }

    /// 本实例是否挂在某条 take 的 FX 链上。
    pub(crate) fn is_take_fx(self) -> bool {
        matches!(self, Self::TakeFx { .. })
    }

    /// 本实例的宿主轨道（没有位置时为 `None`）。
    pub(crate) fn track(self) -> Option<usize> {
        match self {
            Self::TrackFx { track } | Self::TakeFx { track, .. } => Some(track),
            Self::Unbound => None,
        }
    }

    /// 本实例绑定的宿主 item（只有 take FX 有）。
    pub(crate) fn item(self) -> Option<usize> {
        match self {
            Self::TakeFx { item, .. } => Some(item),
            _ => None,
        }
    }

    /// 本实例绑定的宿主 take（只有 take FX 有）。
    pub(crate) fn take(self) -> Option<usize> {
        match self {
            Self::TakeFx { take, .. } => Some(take),
            _ => None,
        }
    }
}

/// 放置位置缓存：结论 + "是否求值过"。
///
/// 【为什么带 `resolved`】`Unbound` 既是"没问过"也是"问过，确实没有"。
/// 前者要触发一次惰性求值，后者不能（否则每次旁通采样都往宿主线程跑一趟完整探测）。
#[derive(Default)]
pub(super) struct PlacementCache {
    pub value: HostPlacement,
    pub resolved: bool,
}

/// take 侧的一轮探测读数。
pub(super) struct TakeProbe {
    pub track: usize,
    pub item: usize,
    pub take: usize,
    /// item 在工程里的播放区间（起点, 终点）；读不出来时 `None`。
    ///
    /// 【为什么需要它】歧义裁决要回答"这一轮分配到的 region 是不是都落在这条 take 的
    /// item 里" —— 那是唯一能区分"某个 item 上的 take 实例"与"同一条轨上的轨道实例"
    /// 的证据。
    pub span: Option<(f64, f64)>,
    pub facts: fx_state::ChainFacts,
}

/// 轨道侧的一轮探测读数。
pub(super) struct TrackProbe {
    pub track: usize,
    pub facts: fx_state::ChainFacts,
}

/// 两条链的一轮探测结果。
///
/// 【为什么"没匹配到"也要留着】诊断行要说清"我看了哪两条链、各自有几个 FX、其中几个
/// 是我"。把它们丢掉，下一次报障又只能靠猜。
#[derive(Default)]
pub(super) struct PlacementProbe {
    pub take: Option<TakeProbe>,
    pub track: Option<TrackProbe>,
}

impl PlacementProbe {
    /// 两条链的读数摘要（诊断行用）：`track=0/0 take=2/2`（总数/匹配数）。
    pub(super) fn chains_summary(&self) -> String {
        let side = |facts: Option<&fx_state::ChainFacts>| match facts {
            Some(facts) => format!("{}/{}", facts.count, facts.matching),
            None => "n/a".to_string(),
        };
        format!(
            "track={} take={}",
            side(self.track.as_ref().map(|probe| &probe.facts)),
            side(self.take.as_ref().map(|probe| &probe.facts)),
        )
    }

    /// 这一轮是否"两条链里都有本插件"（诊断用）。
    pub(super) fn ambiguous(&self) -> bool {
        self.take
            .as_ref()
            .is_some_and(|probe| probe.facts.matching > 0)
            && self
                .track
                .as_ref()
                .is_some_and(|probe| probe.facts.matching > 0)
    }
}

/// 由探测结果与**本轮**分配到的 region 区间裁决放置位置。
///
/// 【歧义规则】两条链里都有本插件（同一轨同时挂了轨道 FX 实例与 take FX 实例）时，
/// 只有"本轮分配到的 region **全部**落在 `parent(2)` 那条 take 的 item 内"才判 take FX。
/// 判据的方向是不对称的：把轨道实例误判成 take FX 会让它的清单只剩一条 take（整轨
/// 消失，用户立刻看见），把 take 实例误判成轨道 FX 只是退回今天的保守行为。所以
/// **没有证据就选轨道**。
pub(super) fn resolve(probe: &PlacementProbe, regions: &[(f64, f64)]) -> HostPlacement {
    let take = probe.take.as_ref().filter(|probe| probe.facts.matching > 0);
    let track = probe
        .track
        .as_ref()
        .filter(|probe| probe.facts.matching > 0);
    match (take, track) {
        (Some(take), None) => HostPlacement::TakeFx {
            track: take.track,
            item: take.item,
            take: take.take,
        },
        (Some(take), Some(track)) => {
            if regions_inside(regions, take.span) {
                HostPlacement::TakeFx {
                    track: take.track,
                    item: take.item,
                    take: take.take,
                }
            } else {
                HostPlacement::TrackFx { track: track.track }
            }
        }
        (None, Some(track)) => HostPlacement::TrackFx { track: track.track },
        (None, None) => HostPlacement::Unbound,
    }
}

/// 这一轮的 region 区间是否**全部**落在 `span` 内。
///
/// 【判据的容差】region 与 item 的边界在宿主里是同一批样本点算出来的，浮点重算会有
/// 1e-9 量级差；这里给 1 ms 的余量（远小于任何有意义的剪辑精度），既不放过"整组在
/// 别的 item 上"，也不因为末位误差把 take FX 判成轨道 FX。
fn regions_inside(regions: &[(f64, f64)], span: Option<(f64, f64)>) -> bool {
    const TOLERANCE_SEC: f64 = 1e-3;
    let Some((start, end)) = span else {
        // 读不到 item 区间 ⇒ 没有证据 ⇒ 不走 take 分支。
        return false;
    };
    if regions.is_empty() {
        return false;
    }
    regions.iter().all(|(region_start, region_end)| {
        region_start.is_finite()
            && region_end.is_finite()
            && *region_start >= start - TOLERANCE_SEC
            && *region_end <= end + TOLERANCE_SEC
    })
}

impl NativeHost {
    /// 缓存里的放置位置（**不发**任何宿主调用）。
    ///
    /// 【为什么允许一个"旧"读数】读它的地方是能力位与清单作用域，两者都在同一条刷新
    /// 链上被重新求值（[`Self::resolve_placement`] 就在它们前面）。为一次能力位查询
    /// 再往宿主线程跑一趟毫无收益。
    pub(crate) fn placement(&self) -> HostPlacement {
        self.placement
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .value
    }

    /// 放置位置是否**求值过**（诊断用：`false` = 还停在"未求值"的默认值上）。
    pub(crate) fn placement_resolved(&self) -> bool {
        self.placement
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .resolved
    }

    /// 探一次两条 FX 链，回答"我可能挂在哪"（只读，不改缓存）。
    pub(super) fn probe_placement(&self, authorized: &impl Fn() -> bool) -> PlacementProbe {
        let mut probe = PlacementProbe::default();
        let Some(api) = self.fx_state.as_ref() else {
            return probe;
        };
        let pointer = self._interface.0 as *mut c_void;
        let table = unsafe { &**pointer.cast::<*const HostVtbl>() };
        // take 侧：`parent(2)` 必须是一条真实 take，而且它的 FX 链要能被读到。
        if let Some(take_api) = api.take.as_ref() {
            let take = checked(authorized, || unsafe { (table.parent)(pointer, 2) })
                .unwrap_or(std::ptr::null_mut());
            if !take.is_null() && self.validate_object(take, c"MediaItem_Take*", authorized) {
                if let Some(row) = self.take_chain_owner(take, authorized) {
                    let chain = fx_state::TakeChain::new(take_api, take, authorized);
                    if let Ok(facts) = fx_state::probe(&chain, crate::CLASS_NAME) {
                        probe.take = Some(TakeProbe {
                            track: row.0,
                            item: row.1,
                            take: take as usize,
                            span: self.item_span(row.1, authorized),
                            facts,
                        });
                    }
                }
            }
        }
        // 轨道侧：`parent(1)` 必须是一条真实轨道。
        let track = checked(authorized, || unsafe { (table.parent)(pointer, 1) })
            .unwrap_or(std::ptr::null_mut());
        if !track.is_null() && self.validate_object(track, c"MediaTrack*", authorized) {
            let chain = fx_state::TrackChain::new(&api.track, track, authorized);
            if let Ok(facts) = fx_state::probe(&chain, crate::CLASS_NAME) {
                probe.track = Some(TrackProbe {
                    track: track as usize,
                    facts,
                });
            }
        }
        probe
    }

    /// 一条 take 的 (宿主轨, item) 指针。
    ///
    /// 【为什么要绕这一下】take FX 的 `parent(1)` 可能**不是** item 的宿主轨（也可能
    /// 直接是 null，见方案 E0）。轨道的唯一可靠来源是 `GetMediaItemTake_Item` →
    /// `GetMediaItem_Track` 这条边，而不是"当前选中轨道"。
    fn take_chain_owner(
        &self,
        take: *mut c_void,
        authorized: &impl Fn() -> bool,
    ) -> Option<(usize, usize)> {
        let geometry = self.geometry.as_ref()?;
        let item = checked(authorized, || unsafe { (geometry.item)(take) }).ok()?;
        if item.is_null() {
            return None;
        }
        let getter = self.item_track?;
        let track = checked(authorized, || unsafe { (getter)(item) }).ok()?;
        if track.is_null() {
            return None;
        }
        Some((track as usize, item as usize))
    }

    /// 重新求值并缓存放置位置；返回本次结论。
    ///
    /// `regions` 是本实例这一轮分配到的 ARA region 的播放时间区间，只在"两条链都有本
    /// 插件"时用来消除歧义（见 [`resolve`]）。调用方是绑定/清单刷新那条路径 ——
    /// 它是唯一同时知道"region 落在哪"与"宿主对象是谁"的层。
    pub(crate) fn resolve_placement(
        &self,
        regions: &[(f64, f64)],
        authorized: &impl Fn() -> bool,
    ) -> HostPlacement {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if !self.on_host_thread() {
            return self
                .on_host(|| self.resolve_placement(regions, authorized))
                .unwrap_or(HostPlacement::Unbound);
        }
        let probe = self.probe_placement(authorized);
        let placement = resolve(&probe, regions);
        let previous = {
            let mut cache = self
                .placement
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let previous = cache.value;
            cache.value = placement;
            cache.resolved = true;
            previous
        };
        if previous != placement {
            self.report_placement(placement, &probe, regions);
        }
        placement
    }

    /// 丢弃缓存，让下一轮重新求值（重绑、或宿主把实例换到另一条链上时）。
    pub(crate) fn forget_placement(&self) {
        let mut cache = self
            .placement
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache.value = HostPlacement::Unbound;
        cache.resolved = false;
        *self
            .placement_reported
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }

    /// 只在读数**变化**时输出的 WARN：这一行是下一次报障的自证材料。
    ///
    /// 【为什么按指纹去重】`resolve_placement` 会跟着清单刷新跑（10 Hz 量级）；无条件
    /// 输出会把它变成第二条刷屏源，而刷屏的代价是**真正的读数变化**被埋掉。
    fn report_placement(
        &self,
        placement: HostPlacement,
        probe: &PlacementProbe,
        regions: &[(f64, f64)],
    ) {
        let bound = probe
            .take
            .as_ref()
            .filter(|probe| probe.facts.matching > 0 && regions_inside(regions, probe.span));
        let verdict = match (placement.is_take_fx(), probe.ambiguous(), bound.is_some()) {
            (true, true, true) => {
                "take-fx, both chains matched, every region is inside the bound take"
            }
            (false, true, false) => {
                "track-fx, both chains matched, regions are not all inside the take"
            }
            _ => placement.as_str(),
        };
        let detail = format!(
            "{verdict}; chains: {}; regions={}",
            probe.chains_summary(),
            regions.len(),
        );
        {
            let mut reported = self
                .placement_reported
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if reported.as_deref() == Some(detail.as_str()) {
                return;
            }
            *reported = Some(detail.clone());
        }
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            &format!("ara.placement:{detail}"),
            format_args!(
                "[ara] placement: {detail} (track={}, item={}, take={})",
                placement.track().unwrap_or(0),
                placement.item().unwrap_or(0),
                placement.take().unwrap_or(0),
            ),
        );
    }

    /// 一个链指针还活着吗；活着就返回它，否则 `None`。
    ///
    /// 与 [`Self::validate_object`] 同一判据，供"每条链都重取一次直接 parent"的
    /// 旁通读路径使用（见 `fx_bypass_state`）。
    pub(super) fn live_chain_owner(
        &self,
        pointer: *mut c_void,
        kind: &CStr,
        authorized: &impl Fn() -> bool,
    ) -> Option<*mut c_void> {
        if pointer.is_null() || !self.validate_object(pointer, kind, authorized) {
            return None;
        }
        Some(pointer)
    }

    /// `ValidatePtr2` 复核；宿主没提供校验入口时按"未核实但可用"处理。
    ///
    /// 【为什么允许缺省】`ValidatePtr2` 是可选增强（与 `GeometryApi` 同一条纪律）：
    /// 老宿主上它可能缺席，而"因此连旁通都读不到"是把可选增强变成硬依赖。缺省时与
    /// 改动前的行为一致 —— 改动前从不校验这条指针。
    fn validate_object(
        &self,
        object: *mut c_void,
        kind: &CStr,
        authorized: &impl Fn() -> bool,
    ) -> bool {
        let Some(validate) = self.validate else {
            return true;
        };
        let Ok(project) = self.project(authorized) else {
            return false;
        };
        checked(authorized, || unsafe {
            (validate)(project, object, kind.as_ptr())
        })
        .unwrap_or(false)
    }

    /// item 在工程里的播放区间（`D_POSITION` + `D_LENGTH`）；读不出来时 `None`。
    fn item_span(&self, item: usize, authorized: &impl Fn() -> bool) -> Option<(f64, f64)> {
        let api = self.geometry.as_ref()?;
        let item = item as *mut c_void;
        let read = |field: &CStr| -> Option<f64> {
            checked(authorized, || unsafe {
                (api.item_value)(item, field.as_ptr())
            })
            .ok()
            .filter(|value| value.is_finite())
        };
        let start = read(c"D_POSITION")?;
        let length = read(c"D_LENGTH")?;
        (length > 0.0).then_some((start, start + length))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(
        take: Option<(usize, usize, usize, usize, Option<(f64, f64)>)>,
        track: Option<(usize, usize)>,
    ) -> PlacementProbe {
        PlacementProbe {
            take: take.map(|(track_id, item, take_id, matching, span)| TakeProbe {
                track: track_id,
                item,
                take: take_id,
                span,
                facts: fx_state::ChainFacts {
                    count: matching as i32,
                    matching,
                },
            }),
            track: track.map(|(track_id, matching)| TrackProbe {
                track: track_id,
                facts: fx_state::ChainFacts {
                    count: matching as i32,
                    matching,
                },
            }),
        }
    }

    /// 只有 take 链里有本插件 ⇒ take FX。这是本轮修复的主路径。
    #[test]
    fn a_take_chain_match_is_take_fx() {
        let resolved = resolve(&probe(Some((10, 20, 30, 1, Some((0.0, 4.0)))), None), &[]);
        assert_eq!(
            resolved,
            HostPlacement::TakeFx {
                track: 10,
                item: 20,
                take: 30
            }
        );
    }

    /// 只有轨道链里有本插件 ⇒ 轨道 FX（改动前的形态，必须原样保留）。
    #[test]
    fn a_track_chain_match_alone_is_track_fx() {
        let resolved = resolve(&probe(None, Some((10, 1))), &[(0.0, 4.0)]);
        assert_eq!(resolved, HostPlacement::TrackFx { track: 10 });
    }

    /// 两条都有：region 全在 take 的 item 内 ⇒ take FX。
    #[test]
    fn an_ambiguous_probe_resolves_by_region_containment() {
        let both = probe(Some((10, 20, 30, 1, Some((2.0, 6.0)))), Some((10, 1)));
        assert_eq!(
            resolve(&both, &[(2.0, 4.0), (4.0, 6.0)]),
            HostPlacement::TakeFx {
                track: 10,
                item: 20,
                take: 30
            }
        );
    }

    /// 两条都有，但 region 有一条落在 item 之外 ⇒ **退回轨道 FX**。
    ///
    /// 【为什么这条是安全边界】把轨道实例误判成 take FX 会让整条轨的清单只剩一条
    /// take（用户立刻看不见别的片段）；反过来的代价只是回到今天的保守行为。
    #[test]
    fn an_ambiguous_probe_never_takes_the_take_side_without_evidence() {
        let both = probe(Some((10, 20, 30, 1, Some((2.0, 6.0)))), Some((10, 1)));
        // 一条 region 落在 item 之外。
        assert_eq!(
            resolve(&both, &[(2.0, 4.0), (6.5, 8.0)]),
            HostPlacement::TrackFx { track: 10 }
        );
        // 没有 region ⇒ 没有证据。
        assert_eq!(resolve(&both, &[]), HostPlacement::TrackFx { track: 10 });
        // 读不到 item 区间 ⇒ 没有证据。
        let unknown_span = probe(Some((10, 20, 30, 1, None)), Some((10, 1)));
        assert_eq!(
            resolve(&unknown_span, &[(2.0, 4.0)]),
            HostPlacement::TrackFx { track: 10 }
        );
    }

    /// 两条链都没有本插件 ⇒ `Unbound`（行为与改动前完全一致）。
    #[test]
    fn no_match_anywhere_is_unbound() {
        assert_eq!(
            resolve(&PlacementProbe::default(), &[(0.0, 1.0)]),
            HostPlacement::Unbound
        );
        // 链在、但里面没有本插件（别的插件同样占着链）也必须算"没有"。
        let foreign = PlacementProbe {
            take: Some(TakeProbe {
                track: 10,
                item: 20,
                take: 30,
                span: Some((0.0, 4.0)),
                facts: fx_state::ChainFacts {
                    count: 3,
                    matching: 0,
                },
            }),
            track: Some(TrackProbe {
                track: 10,
                facts: fx_state::ChainFacts {
                    count: 2,
                    matching: 0,
                },
            }),
        };
        assert_eq!(resolve(&foreign, &[(0.0, 1.0)]), HostPlacement::Unbound);
    }

    /// 诊断摘要必须能把"哪条链有几个、其中几个是我"说清楚。
    #[test]
    fn the_probe_summary_names_both_chains() {
        let both = probe(Some((10, 20, 30, 2, Some((0.0, 4.0)))), Some((10, 1)));
        assert_eq!(both.chains_summary(), "track=1/1 take=2/2");
        assert!(both.ambiguous());
        let summary = PlacementProbe::default().chains_summary();
        assert_eq!(summary, "track=n/a take=n/a");
    }
}
