//! 宿主是否仍在把本实例的音频放进信号链（FX 旁通 / 离线）。
//!
//! # 为什么需要它
//!
//! ARA 没有"旁通"这个概念：`ARAPlugInExtensionInstance` 只声明播放渲染器 / 编辑渲染器 /
//! 编辑视图三个接口，宿主旁通一个 FX 时**根本不再调用**播放渲染器 —— 插件侧看到的只是
//! "这一块没人来取"。VST3 与 CLAP 同样没有旁通查询：VST3 的 `IComponent` 只有
//! `setActive` / `setProcessing`，而旁通是宿主在**自己**的信号链里做的；CLAP 1.x 也没有
//! 旁通扩展。所以"我是不是被旁通了"只能问**宿主自己的**原生 API。
//!
//! REAPER 给出**两条平行**的这一族读数：轨道的 `TrackFX_*` 与 take 的 `TakeFX_*`，
//! 各有四件（`GetCount` / `GetEnabled` / `GetOffline` / `GetFXName`），外加各自的
//! 整链总开关（轨道的 `I_FXEN`、take 的 `B_TAKEFX`）。本模块把它们收成一个**链**的
//! 只读视图（[`FxChain`]），于是"读哪条链"由调用方按放置位置选，"怎么判"只有一份。
//!
//! # 为什么是三态而不是布尔
//!
//! "读不出来"与"没有被旁通"必须分开。前者（旧宿主、API 缺失、授权收回、这条链上找不到
//! 本插件的 FX）一旦被当成后者，插件就会在一个用户已经旁通的实例上继续跑神经合成 ——
//! 这正是本模块要消除的浪费；反过来把它当成"已旁通"，用户会**听不到声音**却没有任何提示。
//! 三态让两个方向都不会被静默猜错：`Unknown` 走改动前的行为（照常渲染）。
//!
//! # 为什么"查不到"绝不跨链回退
//!
//! 轨道 FX 与 take FX 是两条**互不相干**的链。take 侧查不到就去看轨道侧，等于把
//! "同一条轨上另一个实例的开关"当成自己的（见 `HostPlacement`）—— 两个方向都会判错：
//! 轨道实例被旁通时 take 实例被误判旁通（用户听不到），反之则继续白算。所以
//! `read_chain` 只认它拿到的那一条链，匹配数为 0 就是 `Unknown`。

use super::checked;
use std::ffi::{c_char, c_void};

/// `int TrackFX_GetCount(MediaTrack*)` / `int TakeFX_GetCount(MediaItem_Take*)`
pub(super) type FxCount = unsafe extern "C" fn(*mut c_void) -> i32;
/// `bool TrackFX_GetEnabled(MediaTrack*, int)` / `bool TakeFX_GetEnabled(MediaItem_Take*, int)`
pub(super) type FxEnabled = unsafe extern "C" fn(*mut c_void, i32) -> bool;
/// `bool TrackFX_GetOffline(MediaTrack*, int)` / `bool TakeFX_GetOffline(MediaItem_Take*, int)`
pub(super) type FxOffline = unsafe extern "C" fn(*mut c_void, i32) -> bool;
/// `bool TrackFX_GetFXName(MediaTrack*, int, char*, int)` / 同形的 take 版
pub(super) type FxName = unsafe extern "C" fn(*mut c_void, i32, *mut c_char, i32) -> bool;
/// `double GetMediaTrackInfo_Value(MediaTrack*, const char*)`
pub(super) type TrackValue = unsafe extern "C" fn(*mut c_void, *const c_char) -> f64;
/// `double GetMediaItemTakeInfo_Value(MediaItem_Take*, const char*)`
pub(super) type TakeValue = unsafe extern "C" fn(*mut c_void, *const c_char) -> f64;

/// 轨道 FX 链的四件读数 + 整链总开关（`I_FXEN`）。
pub(super) struct TrackChainApi {
    pub count: FxCount,
    pub enabled: FxEnabled,
    pub offline: FxOffline,
    pub name: FxName,
    pub track_value: TrackValue,
}

/// take FX 链的四件读数 + 整链总开关（`B_TAKEFX`）。
pub(super) struct TakeChainApi {
    pub count: FxCount,
    pub enabled: FxEnabled,
    pub offline: FxOffline,
    pub name: FxName,
    pub take_value: TakeValue,
}

/// 读 FX 旁通所需的入口。
///
/// 【为什么 take 侧是可选的】`TakeFX_*` 是后来才进宿主的一族；缺任何一个就整体降级成
/// `None`（⇒ take 链上的实例一律 `Unknown` ⇒ 照常渲染），而不是拿轨道链顶替。
pub(super) struct FxStateApi {
    pub track: TrackChainApi,
    pub take: Option<TakeChainApi>,
}

/// 本实例在宿主里此刻是否被旁通。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FxBypass {
    /// 宿主会把我们的输出放进信号链。
    Active,
    /// 用户旁通了它（或把它离线了，或关掉了整条 FX 链）：宿主不会读我们的输出。
    Bypassed,
    /// 读不出来。**必须**按改动前的行为处理，不得猜成任何一种。
    Unknown,
}

impl FxBypass {
    /// 稳定的语言无关编码（诊断包与日志用）。
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Bypassed => "bypassed",
            Self::Unknown => "unknown",
        }
    }

    /// "确定被旁通"的唯一判断入口。
    pub(crate) fn is_bypassed(self) -> bool {
        matches!(self, Self::Bypassed)
    }

    /// 跨线程传递用的编码（`AtomicU8` 的载荷）。
    ///
    /// 【为什么需要编码而不是直接存枚举】读侧可能在音频线程、写侧在宿主线程；为读一个
    /// 三态量取锁毫无必要，而"一个字节"是三态量能有的最小共享形式。
    ///
    /// 【为什么 `Unknown` 必须是 0】它是 `AtomicU8` 的默认值，而默认值必须等于
    /// **改动前的行为**（照常渲染）。任何把默认值改成"旁通"的改动都会让所有不支持这
    /// 一族读数的宿主（Cubase / Studio One / 任何非 REAPER 宿主）当场静音。
    pub(crate) fn encode(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::Active => 1,
            Self::Bypassed => 2,
        }
    }

    /// 解码。**任何未定义值都必须落回 `Unknown`** —— 见 [`Self::encode`]。
    pub(crate) fn decode(raw: u8) -> Self {
        match raw {
            1 => Self::Active,
            2 => Self::Bypassed,
            _ => Self::Unknown,
        }
    }
}

/// 一条 FX 链的只读视图；轨道与 take 各实现一次。
///
/// 【为什么要有这层】"哪条链"与"怎么判"是两件事。判定（[`decide`]）是纯函数、
/// 已经被用例钉死；把链变成 trait 之后，两条链共用同一份判定与同一份错误处理，
/// 不存在"轨道那份修了、take 那份忘了修"的漂移面。
pub(super) trait FxChain {
    fn count(&self) -> Result<i32, String>;
    /// `index` 处的 FX 显示名写进 `buffer`；宿主拒绝时返回 `false`。
    fn name(&self, index: i32, buffer: &mut [i8]) -> Result<bool, String>;
    fn enabled(&self, index: i32) -> Result<bool, String>;
    fn offline(&self, index: i32) -> Result<bool, String>;
    /// 整条链的总开关。
    fn chain_enabled(&self) -> Result<bool, String>;
}

/// 轨道 FX 链（`TrackFX_*` + `I_FXEN`）。
pub(super) struct TrackChain<'a> {
    api: &'a TrackChainApi,
    track: *mut c_void,
    authorized: &'a dyn Fn() -> bool,
}

impl<'a> TrackChain<'a> {
    pub(super) fn new(
        api: &'a TrackChainApi,
        track: *mut c_void,
        authorized: &'a dyn Fn() -> bool,
    ) -> Self {
        Self {
            api,
            track,
            authorized,
        }
    }
}

impl FxChain for TrackChain<'_> {
    fn count(&self) -> Result<i32, String> {
        checked(self.authorized, || unsafe { (self.api.count)(self.track) })
    }
    fn name(&self, index: i32, buffer: &mut [i8]) -> Result<bool, String> {
        let size = buffer.len() as i32;
        checked(self.authorized, || unsafe {
            (self.api.name)(self.track, index, buffer.as_mut_ptr(), size)
        })
    }
    fn enabled(&self, index: i32) -> Result<bool, String> {
        checked(self.authorized, || unsafe {
            (self.api.enabled)(self.track, index)
        })
    }
    fn offline(&self, index: i32) -> Result<bool, String> {
        checked(self.authorized, || unsafe {
            (self.api.offline)(self.track, index)
        })
    }
    fn chain_enabled(&self) -> Result<bool, String> {
        let value = checked(self.authorized, || unsafe {
            (self.api.track_value)(self.track, c"I_FXEN".as_ptr())
        })?;
        Ok(value != 0.0)
    }
}

/// take FX 链（`TakeFX_*` + `B_TAKEFX`）。
pub(super) struct TakeChain<'a> {
    api: &'a TakeChainApi,
    take: *mut c_void,
    authorized: &'a dyn Fn() -> bool,
}

impl<'a> TakeChain<'a> {
    pub(super) fn new(
        api: &'a TakeChainApi,
        take: *mut c_void,
        authorized: &'a dyn Fn() -> bool,
    ) -> Self {
        Self {
            api,
            take,
            authorized,
        }
    }
}

impl FxChain for TakeChain<'_> {
    fn count(&self) -> Result<i32, String> {
        checked(self.authorized, || unsafe { (self.api.count)(self.take) })
    }
    fn name(&self, index: i32, buffer: &mut [i8]) -> Result<bool, String> {
        let size = buffer.len() as i32;
        checked(self.authorized, || unsafe {
            (self.api.name)(self.take, index, buffer.as_mut_ptr(), size)
        })
    }
    fn enabled(&self, index: i32) -> Result<bool, String> {
        checked(self.authorized, || unsafe {
            (self.api.enabled)(self.take, index)
        })
    }
    fn offline(&self, index: i32) -> Result<bool, String> {
        checked(self.authorized, || unsafe {
            (self.api.offline)(self.take, index)
        })
    }
    /// take 侧的整链开关是 take 自己的字段 `B_TAKEFX`（`I_FXEN` 的对应物）。
    ///
    /// 【为什么不用 `TakeFX_GetEnabled(take, -1)`】两个读数表达同一件事，但
    /// `GetMediaItemTakeInfo_Value` 与轨道侧的 `GetMediaTrackInfo_Value` 是同一条
    /// 读取纪律（同族入口、同一种"读不到就当 0"的失败面），选它让两条链的失败形态
    /// 一致。真机复核（方案 E2）若发现 `B_TAKEFX` 不表达整链开关，只需改这一处。
    fn chain_enabled(&self) -> Result<bool, String> {
        let value = checked(self.authorized, || unsafe {
            (self.api.take_value)(self.take, c"B_TAKEFX".as_ptr())
        })?;
        Ok(value != 0.0)
    }
}

/// 一条链上一轮枚举的读数（放置位置探测与诊断用）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ChainFacts {
    /// 宿主报告的 FX 总数。
    pub count: i32,
    /// 其中按类名匹配到本插件的个数。
    pub matching: usize,
}

/// 数一条链上有几个本插件的 FX（**不**读 enabled/offline，只回答"我在不在里面"）。
///
/// 【为什么与 [`read_chain`] 分开】放置位置探测只关心"这条链里有没有我"；把
/// enabled/chain 开关一起读进来会让一次探测产生十来个多余宿主调用，而探测发生在
/// 每个绑定与每次惰性求值上。
pub(super) fn probe(chain: &dyn FxChain, class_name: &str) -> Result<ChainFacts, String> {
    let count = chain.count()?;
    if count <= 0 {
        return Ok(ChainFacts {
            count: count.max(0),
            matching: 0,
        });
    }
    let needle = class_name.to_ascii_lowercase();
    let mut buffer = [0_i8; 256];
    let mut matching = 0_usize;
    for index in 0..count {
        if !chain.name(index, &mut buffer)? {
            continue;
        }
        // SAFETY: 宿主把 NUL 结尾的名字写进我们持有的缓冲区。
        let name = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .to_ascii_lowercase();
        if name.contains(&needle) {
            matching += 1;
        }
    }
    Ok(ChainFacts { count, matching })
}

/// 在**一条**链上读一次旁通结论。
///
/// 调用方保证：`authorized` 在整段读取期间可反复求值；`chain` 指向本实例**真正挂靠**
/// 的那条链（由 [`super::placement::HostPlacement`] 决定）。**不得**在查不到时换一条链
/// 重试 —— 见模块文档。
pub(super) fn read_chain(chain: &dyn FxChain, class_name: &str) -> Result<FxBypass, String> {
    let count = chain.count()?;
    if count <= 0 {
        return Ok(FxBypass::Unknown);
    }
    let needle = class_name.to_ascii_lowercase();
    let mut found = 0_usize;
    let mut any_enabled = false;
    let mut all_offline = true;
    let mut buffer = [0_i8; 256];
    for index in 0..count {
        if !chain.name(index, &mut buffer)? {
            continue;
        }
        // SAFETY: 宿主把 NUL 结尾的名字写进我们持有的缓冲区。
        let name = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .to_ascii_lowercase();
        if !name.contains(&needle) {
            continue;
        }
        found += 1;
        any_enabled |= chain.enabled(index)?;
        all_offline &= chain.offline(index)?;
    }
    if found == 0 {
        return Ok(FxBypass::Unknown);
    }
    let chain_enabled = chain.chain_enabled()?;
    Ok(decide(found, any_enabled, all_offline, chain_enabled))
}

/// 由四份读数得出结论。
///
/// 入参都是**已经核实过的**事实：
/// * `found` —— 这条链上有几个 FX 是本插件（按类名匹配到的个数）；
/// * `any_enabled` —— 其中是否至少有一个是启用的；
/// * `all_offline` —— 其中是否全部处于离线（offline = 不加载、不处理）；
/// * `chain_enabled` —— 这条链的总开关（轨道 `I_FXEN` / take `B_TAKEFX`）。
///
/// 判定取"**全部**证据都指向旁通才报旁通"：
/// * 一个都没匹配到 ⇒ 不知道（可能是宿主改了显示名，也可能这条链上挂的是别的实例）；
/// * 只要有一个同类 FX 是启用的 ⇒ 活跃（同一条链上挂两个本插件实例时宿主只会把区域交给
///   其中一个，"至少有一个在用"才是安全的答案）；
/// * 全部离线、或整条链被关掉 ⇒ 旁通 —— 这两种状态下宿主都拿不到我们的输出。
pub(crate) fn decide(
    found: usize,
    any_enabled: bool,
    all_offline: bool,
    chain_enabled: bool,
) -> FxBypass {
    if found == 0 {
        return FxBypass::Unknown;
    }
    if !chain_enabled || all_offline || !any_enabled {
        return FxBypass::Bypassed;
    }
    FxBypass::Active
}

/// 把**多个实例**的旁通读数合成"这份工程此刻还会不会出声"。
///
/// 【为什么需要合成，而不是让它们共享一格】旁通是**每个 FX** 的属性，而一个 ARA 文档
/// 里可以有多条轨道、每轨一个实例。把结论写进一个共享格子，等于让"轨 1 被旁通"和
/// "轨 2 没被旁通"互相覆盖：读数每一轮都在两个值之间跳，而每次跳都会让渲染输入失效
/// ⇒ 每个实例每 100 ms 重算一整次神经合成（用户看到的就是"无限渲染"）。
///
/// 【判据：只有**全部**实例都确定被旁通，才算旁通】
/// * 任何一个 `Active` ⇒ 有人会来取音频，必须渲染；
/// * 任何一个 `Unknown` ⇒ "读不出来"必须按改动前的行为（照常渲染），不得被别人的
///   `Bypassed` 带着一起跳过 —— 否则那个实例的用户会静音；
/// * 一个实例都没有 ⇒ 无从判断。
pub(crate) fn combine(verdicts: &[FxBypass]) -> FxBypass {
    if verdicts.is_empty() {
        return FxBypass::Unknown;
    }
    if verdicts.iter().all(|verdict| verdict.is_bypassed()) {
        return FxBypass::Bypassed;
    }
    if verdicts.contains(&FxBypass::Active) {
        return FxBypass::Active;
    }
    // 剩下的组合里一定既有 `Bypassed` 又有 `Unknown`：任何一个实例读不出来就不许跳过。
    FxBypass::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 一条纯内存的 FX 链：把宿主的五件读数换成五个可断言的值。
    ///
    /// 【为什么值得造它】[`read_chain`] 的判据（名字匹配 → 逐 FX enabled/offline →
    /// 整链开关）此前只能靠真机验证；有了它，两条链共用的那份判定在没有宿主的前提下
    /// 也能被逐值钉住。
    struct FakeChain {
        entries: Vec<(&'static str, bool, bool)>,
        chain_enabled: bool,
    }

    impl FakeChain {
        fn new(entries: &[(&'static str, bool, bool)]) -> Self {
            Self {
                entries: entries.to_vec(),
                chain_enabled: true,
            }
        }
    }

    impl FxChain for FakeChain {
        fn count(&self) -> Result<i32, String> {
            Ok(self.entries.len() as i32)
        }
        fn name(&self, index: i32, buffer: &mut [i8]) -> Result<bool, String> {
            for (slot, byte) in buffer
                .iter_mut()
                .zip(self.entries[index as usize].0.as_bytes())
            {
                *slot = *byte as i8;
            }
            Ok(true)
        }
        fn enabled(&self, index: i32) -> Result<bool, String> {
            Ok(self.entries[index as usize].1)
        }
        fn offline(&self, index: i32) -> Result<bool, String> {
            Ok(self.entries[index as usize].2)
        }
        fn chain_enabled(&self) -> Result<bool, String> {
            Ok(self.chain_enabled)
        }
    }

    /// 一条轨上找不到本插件的 FX ⇒ 不知道。绝不猜成"活跃"（那会让下面的门失效）也
    /// 绝不猜成"旁通"（那会让用户听不到声音）。
    #[test]
    fn no_matching_fx_is_unknown_not_a_guess() {
        assert_eq!(decide(0, false, false, true), FxBypass::Unknown);
        assert_eq!(decide(0, true, false, false), FxBypass::Unknown);
    }

    /// 只要有一个同类实例是启用的，就按活跃处理 —— 同轨两个实例时宿主只会把区域交给
    /// 其中一个，"至少有一个在用"是唯一安全的答案。
    #[test]
    fn one_enabled_instance_keeps_the_renderer_active() {
        assert_eq!(decide(2, true, false, true), FxBypass::Active);
        assert_eq!(decide(1, true, false, true), FxBypass::Active);
    }

    /// 三种"宿主不会读我们的输出"都必须报旁通。
    #[test]
    fn every_way_the_host_stops_reading_us_reports_bypass() {
        // 单个 FX 被旁通。
        assert_eq!(decide(1, false, false, true), FxBypass::Bypassed);
        // FX 离线（不加载）。
        assert_eq!(decide(1, true, true, true), FxBypass::Bypassed);
        // 整条 FX 链被关掉。
        assert_eq!(decide(1, true, false, false), FxBypass::Bypassed);
    }

    /// 一条只有**别的**插件的链，与一条同名但离线/旁通的链，判据必须分开：
    /// 前者是"我不知道"，后者是"宿主不会读我"。
    ///
    /// 【为什么这条是 take FX 那次故障的回归】take 链上本插件之前，最容易被写成
    /// "读轨道链"或"读不到就当活跃"。两者都会让旁通门在 take FX 上失效，而现象
    /// 与本次报障完全一致（旁通了还在渲染）。
    #[test]
    fn a_chain_without_our_class_name_never_borrows_a_verdict() {
        let other = FakeChain::new(&[("VST3: Someone Else", true, false)]);
        assert_eq!(read_chain(&other, "hifishifter"), Ok(FxBypass::Unknown));
        let empty = FakeChain::new(&[]);
        assert_eq!(read_chain(&empty, "hifishifter"), Ok(FxBypass::Unknown));
        let bypassed = FakeChain::new(&[("VST3: HiFiShifter", false, false)]);
        assert_eq!(read_chain(&bypassed, "hifishifter"), Ok(FxBypass::Bypassed));
        let active = FakeChain::new(&[("VST3: HiFiShifter", true, false)]);
        assert_eq!(read_chain(&active, "hifishifter"), Ok(FxBypass::Active));
    }

    /// 两条链的**同一条**判定：只要本插件在链上，整链总开关就说话。
    ///
    /// 【为什么值得单测】`chain_enabled` 是唯一一个"每个 FX 都正常、但整条链被关掉"
    /// 的读数。它错了，用户会在 take FX 窗口里关掉整条链、插件照旧跑满 CPU。
    #[test]
    fn the_chain_toggle_closes_the_gate_even_when_every_fx_is_enabled() {
        let mut chain = FakeChain::new(&[("VST3: HiFiShifter", true, false)]);
        chain.chain_enabled = false;
        assert_eq!(read_chain(&chain, "hifishifter"), Ok(FxBypass::Bypassed));
    }

    /// `probe` 只回答"我在不在里面"，不读 enabled/offline。
    #[test]
    fn the_placement_probe_counts_matches_only() {
        let chain = FakeChain::new(&[
            ("VST3: HiFiShifter", false, true),
            ("VST3: Other", true, false),
            ("VST3: HiFiShifter (mono)", true, false),
        ]);
        assert_eq!(
            probe(&chain, "hifishifter"),
            Ok(ChainFacts {
                count: 3,
                matching: 2
            })
        );
        assert_eq!(
            probe(&FakeChain::new(&[]), "hifishifter"),
            Ok(ChainFacts {
                count: 0,
                matching: 0
            })
        );
    }

    /// `FxBypass` 的编码是**跨线程契约**：写侧（`sample_host_bypass`）与读侧
    /// （`prepare` / 后台作业的门）必须对同一个字节给出同一个结论。默认值 0
    /// 必须是 `Unknown` —— 那是"照改动前渲染"的唯一保证。
    #[test]
    fn the_bypass_encoding_never_turns_unknown_into_a_verdict() {
        assert_eq!(FxBypass::decode(0), FxBypass::Unknown);
        assert_eq!(FxBypass::decode(1), FxBypass::Active);
        assert_eq!(FxBypass::decode(2), FxBypass::Bypassed);
        // 任何未来可能出现的值都必须落回"未知"，不得被当成"活跃"或"旁通"。
        for raw in [3_u8, 7, 200, 255] {
            assert_eq!(FxBypass::decode(raw), FxBypass::Unknown, "raw={raw}");
        }
        // 往返：三态各自编码再解码必须回到自己。
        for state in [FxBypass::Unknown, FxBypass::Active, FxBypass::Bypassed] {
            assert_eq!(FxBypass::decode(state.encode()), state);
        }
        // 只有"确定旁通"能挡住渲染。
        assert!(FxBypass::decode(2).is_bypassed());
        assert!(!FxBypass::decode(0).is_bypassed());
        assert!(!FxBypass::decode(1).is_bypassed());
    }

    /// **合成判据：只有全部实例都确定被旁通，才算这份工程不会出声。**
    ///
    /// 【为什么这条必须单测】它是"跳过整次合成"的唯一门。判宽了（把 Unknown 当旁通）
    /// 会让读不出读数的宿主静音；判窄了（要求任意一个就算）会让"轨 1 旁通、轨 2 正常"
    /// 的工程永远不渲染。两个方向都是用户直接可感知的故障。
    #[test]
    fn only_an_unanimous_bypass_skips_the_render() {
        use FxBypass::{Active, Bypassed, Unknown};
        assert_eq!(combine(&[]), Unknown, "没有实例 ⇒ 无从判断");
        assert_eq!(combine(&[Bypassed]), Bypassed);
        assert_eq!(combine(&[Bypassed, Bypassed]), Bypassed);
        // 只要有一个还在用，就必须渲染。
        assert_eq!(combine(&[Bypassed, Active]), Active);
        assert_eq!(combine(&[Active]), Active);
        // 读不出来必须按改动前的行为：不得被别人的结论带着一起跳过。
        assert_eq!(combine(&[Bypassed, Unknown]), Unknown);
        assert_eq!(combine(&[Unknown]), Unknown);
        assert_eq!(combine(&[Unknown, Active]), Active);
        // 只有 `Bypassed` 能挡住渲染。
        assert!(combine(&[Bypassed, Bypassed]).is_bypassed());
        assert!(!combine(&[Bypassed, Unknown]).is_bypassed());
        assert!(!combine(&[Bypassed, Active]).is_bypassed());
    }

    /// 三态的编码是诊断契约的一部分：改字面量等于改日志与诊断包的读法。
    #[test]
    fn the_state_names_are_stable() {
        assert_eq!(FxBypass::Active.as_str(), "active");
        assert_eq!(FxBypass::Bypassed.as_str(), "bypassed");
        assert_eq!(FxBypass::Unknown.as_str(), "unknown");
        assert!(FxBypass::Bypassed.is_bypassed());
        assert!(!FxBypass::Unknown.is_bypassed());
        assert!(!FxBypass::Active.is_bypassed());
    }
}
