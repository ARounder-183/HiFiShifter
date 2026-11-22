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
//! 各有四件（`GetCount` / `GetEnabled` / `GetOffline` / `GetFXName`）。本模块把它们
//! 收成一个**链**的只读视图（[`FxChain`]），于是"读哪条链"由调用方按放置位置选，
//! "怎么判"只有一份。
//!
//! # 两条链**并不对称**：整链总开关只有轨道侧有
//!
//! 轨道有 `I_FXEN`（"fx enabled, 0=bypassed"），take 侧**没有**对应物：
//! `GetMediaItemTakeInfo_Value` 的字段表里没有任何整链开关（`TAKEFX_NCH` 只是内部
//! 通道数），`TakeFX_*` 一族也只有逐 FX 的 `GetEnabled` / `GetOffline`。
//!
//! 这条不对称曾经被"对称美学"抹平过：take 链照轨道的样子去读 `B_TAKEFX`，而宿主对
//! **未知**参数一律返回 0 —— 于是一个**完全没有被旁通**的 take FX 实例被判成旁通，
//! 渲染入口整段跳过，用户听到的是**彻底的静音**。教训写进 [`ChainGate`]：读不到
//! 就是读不到，绝不用"0"这个默认值替宿主下结论。
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

/// 轨道 FX 链的四件读数 + 整链总开关（`I_FXEN`）。
pub(super) struct TrackChainApi {
    pub count: FxCount,
    pub enabled: FxEnabled,
    pub offline: FxOffline,
    pub name: FxName,
    pub track_value: TrackValue,
}

/// take FX 链的四件读数。
///
/// 【为什么**没有**整链总开关】take 侧根本没有这个读数（见模块文档）：`TakeFX_*` 没有
/// `GetChainEnabled`，`GetMediaItemTakeInfo_Value` 也没有对应字段。曾经凭"轨道有、
/// take 也该有"去读了一个不存在的 `B_TAKEFX`，宿主对未知参数返回 0 ⇒ 一律判成"整链
/// 被关掉" ⇒ take FX 下**整段静音**。宁可承认"这条链没有总开关"，也不许再猜。
pub(super) struct TakeChainApi {
    pub count: FxCount,
    pub enabled: FxEnabled,
    pub offline: FxOffline,
    pub name: FxName,
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

/// 一条链**整链总开关**的读数。
///
/// `Unknown` 是"**没有这个读数**"：take 链恒为它，轨道链在 API 缺失 / 读取被拒时也是它。
/// 与 [`FxBypass::Unknown`] 同一条纪律 —— 不得被当成"关掉"。判据里把它读作"这一位
/// 不投票"：只要每个 FX 自己的 enabled / offline 都说"在用"，门就是开的。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ChainGate {
    /// 宿主明确报告这条链是通的。
    Open,
    /// 宿主明确报告这条链被关掉（轨道 `I_FXEN == 0`）。
    Closed,
    /// 没有这个读数（take 链，或这一轮读不出来）。
    Unknown,
}

impl ChainGate {
    /// 稳定的语言无关编码（诊断用）。
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Closed => "closed",
            Self::Unknown => "unknown",
        }
    }

    /// 只有**宿主明确说**关掉才算关 —— `Unknown` 不算。
    pub(super) fn is_closed(self) -> bool {
        matches!(self, Self::Closed)
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
    ///
    /// 【为什么缓冲是 `c_char` 而不是 `i8`】`char` 的符号性是**目标平台**决定的：
    /// x86_64 上是 `i8`，而 aarch64 Linux 上是 `u8`。写死 `i8` 只在 x86_64 上
    /// 恰好等于 `c_char`，交叉编译到 linux-aarch64 时就会和 [`FxName`] 的签名
    /// 对不上（`*mut i8` vs `*mut u8`）。缓冲区必须跟随 `c_char` 本身。
    fn name(&self, index: i32, buffer: &mut [c_char]) -> Result<bool, String>;
    fn enabled(&self, index: i32) -> Result<bool, String>;
    fn offline(&self, index: i32) -> Result<bool, String>;
    /// 整条链的总开关。没有这个读数的链返回 [`ChainGate::Unknown`]。
    fn chain_enabled(&self) -> Result<ChainGate, String>;
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
    fn name(&self, index: i32, buffer: &mut [c_char]) -> Result<bool, String> {
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
    fn chain_enabled(&self) -> Result<ChainGate, String> {
        let value = checked(self.authorized, || unsafe {
            (self.api.track_value)(self.track, c"I_FXEN".as_ptr())
        })?;
        Ok(if value != 0.0 {
            ChainGate::Open
        } else {
            ChainGate::Closed
        })
    }
}

/// take FX 链（只有 `TakeFX_*` 四件；take 侧没有整链总开关）。
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
    fn name(&self, index: i32, buffer: &mut [c_char]) -> Result<bool, String> {
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
    /// take 侧**没有**整链总开关这个读数（见模块文档与 [`TakeChainApi`]）。
    ///
    /// 【为什么是"没有读数"而不是"链是开的"】两种答案都让门保持敞开，但一个是事实、
    /// 一个是断言。诊断里必须能看出"这一位没有投票"，否则下一次同样的故障又要从头查。
    fn chain_enabled(&self) -> Result<ChainGate, String> {
        Ok(ChainGate::Unknown)
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
    let mut buffer = [0 as c_char; 256];
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

/// 一条链上一轮读取的完整读数：结论 **和** 它凭什么。
///
/// 【为什么要连"凭什么"一起交出来】旁通门一旦判错，用户看到的是"完全没有声音"或
/// "旁通了还在跑满 CPU"，而日志里只有一行结论。把四份原始读数留在同一行里，
/// 下一次同类报障第一眼就能看出是哪一位判错了 —— take FX 那次静音正是这样：
/// 结论"被旁通"，而四位事实里唯一的反常是"整链总开关 = 关"。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ChainReading {
    /// 结论。
    pub verdict: FxBypass,
    /// 这条链上有几个 FX、其中几个是本插件。
    pub facts: ChainFacts,
    /// 匹配到的 FX 里有几个是启用的。
    pub enabled: usize,
    /// 匹配到的 FX 里有几个处于离线。
    pub offline: usize,
    /// 整链总开关的读数（take 链恒为 [`ChainGate::Unknown`]）。
    pub gate: ChainGate,
}

impl ChainReading {
    /// 没有读数（链是空的 / 这条链上没有本插件）。
    fn unknown(facts: ChainFacts) -> Self {
        Self {
            verdict: FxBypass::Unknown,
            facts,
            enabled: 0,
            offline: 0,
            gate: ChainGate::Unknown,
        }
    }

    /// 一行诊断：`found=1/1 enabled=1 offline=0 gate=unknown`。
    ///
    /// 【为什么要带上分母】`found=1` 与 `found=1/3` 是两种完全不同的处境：后者说明
    /// 这条链上还挂着别人的 FX。排查"读错链"时这一点比结论本身更有用。
    pub(super) fn summary(self) -> String {
        format!(
            "found={}/{} enabled={} offline={} gate={}",
            self.facts.matching,
            self.facts.count,
            self.enabled,
            self.offline,
            self.gate.as_str()
        )
    }
}

/// 在**一条**链上读一次旁通结论。
///
/// 调用方保证：`authorized` 在整段读取期间可反复求值；`chain` 指向本实例**真正挂靠**
/// 的那条链（由 [`super::placement::HostPlacement`] 决定）。**不得**在查不到时换一条链
/// 重试 —— 见模块文档。
pub(super) fn read_chain(chain: &dyn FxChain, class_name: &str) -> Result<ChainReading, String> {
    let count = chain.count()?;
    if count <= 0 {
        return Ok(ChainReading::unknown(ChainFacts {
            count: count.max(0),
            matching: 0,
        }));
    }
    let needle = class_name.to_ascii_lowercase();
    let mut found = 0_usize;
    let mut enabled = 0_usize;
    let mut offline = 0_usize;
    let mut buffer = [0 as c_char; 256];
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
        if chain.enabled(index)? {
            enabled += 1;
        }
        if chain.offline(index)? {
            offline += 1;
        }
    }
    if found == 0 {
        return Ok(ChainReading::unknown(ChainFacts { count, matching: 0 }));
    }
    // 【整链总开关排在逐 FX 读数之后读】它是唯一一个"每个 FX 都正常、但整条链被关掉"
    // 的读数，只在确实匹配到本插件时才值得问一次宿主。
    let gate = chain.chain_enabled()?;
    Ok(ChainReading {
        verdict: decide(found, enabled, offline, gate),
        facts: ChainFacts {
            count,
            matching: found,
        },
        enabled,
        offline,
        gate,
    })
}

/// 由几份原始读数得出结论。
///
/// 入参都是**已经核实过的**事实：
/// * `found` —— 这条链上有几个 FX 是本插件（按类名匹配到的个数）；
/// * `enabled` —— 其中几个是启用的；
/// * `offline` —— 其中几个处于离线（offline = 不加载、不处理）；
/// * `gate` —— 这条链的整链总开关；[`ChainGate::Unknown`] = 没有这个读数。
///
/// 判定取"**全部**证据都指向旁通才报旁通"：
/// * 一个都没匹配到 ⇒ 不知道（可能是宿主改了显示名，也可能这条链上挂的是别的实例）；
/// * 只要有一个同类 FX 是启用的 ⇒ 活跃（同一条链上挂两个本插件实例时宿主只会把区域交给
///   其中一个，"至少有一个在用"才是安全的答案）；
/// * 全部离线、或整条链**被明确关掉** ⇒ 旁通 —— 这两种状态下宿主都拿不到我们的输出；
/// * [`ChainGate::Unknown`] ⇒ 这一位不投票。它不是"关掉"，而是"这条链没有这个开关"。
pub(crate) fn decide(found: usize, enabled: usize, offline: usize, gate: ChainGate) -> FxBypass {
    if found == 0 {
        return FxBypass::Unknown;
    }
    if gate.is_closed() || offline >= found || enabled == 0 {
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

    /// 一条纯内存的 FX 链：把宿主的读数换成几个可断言的值。
    ///
    /// 【为什么值得造它】[`read_chain`] 的判据（名字匹配 → 逐 FX enabled/offline →
    /// 整链开关）此前只能靠真机验证；有了它，两条链共用的那份判定在没有宿主的前提下
    /// 也能被逐值钉住。
    struct FakeChain {
        entries: Vec<(&'static str, bool, bool)>,
        gate: ChainGate,
    }

    impl FakeChain {
        fn new(entries: &[(&'static str, bool, bool)]) -> Self {
            Self {
                entries: entries.to_vec(),
                gate: ChainGate::Open,
            }
        }

        /// 没有整链总开关的链（take FX 的唯一形态）。
        fn without_master_switch(entries: &[(&'static str, bool, bool)]) -> Self {
            Self {
                entries: entries.to_vec(),
                gate: ChainGate::Unknown,
            }
        }
    }

    impl FxChain for FakeChain {
        fn count(&self) -> Result<i32, String> {
            Ok(self.entries.len() as i32)
        }
        fn name(&self, index: i32, buffer: &mut [c_char]) -> Result<bool, String> {
            for (slot, byte) in buffer
                .iter_mut()
                .zip(self.entries[index as usize].0.as_bytes())
            {
                *slot = *byte as c_char;
            }
            Ok(true)
        }
        fn enabled(&self, index: i32) -> Result<bool, String> {
            Ok(self.entries[index as usize].1)
        }
        fn offline(&self, index: i32) -> Result<bool, String> {
            Ok(self.entries[index as usize].2)
        }
        fn chain_enabled(&self) -> Result<ChainGate, String> {
            Ok(self.gate)
        }
    }

    /// 一条轨上找不到本插件的 FX ⇒ 不知道。绝不猜成"活跃"（那会让下面的门失效）也
    /// 绝不猜成"旁通"（那会让用户听不到声音）。
    #[test]
    fn no_matching_fx_is_unknown_not_a_guess() {
        assert_eq!(decide(0, 0, 0, ChainGate::Open), FxBypass::Unknown);
        assert_eq!(decide(0, 1, 0, ChainGate::Closed), FxBypass::Unknown);
        assert_eq!(
            read_chain(&FakeChain::new(&[]), "hifishifter"),
            Ok(ChainReading::unknown(ChainFacts {
                count: 0,
                matching: 0
            }))
        );
    }

    /// 只要有一个同类实例是启用的，就按活跃处理 —— 同轨两个实例时宿主只会把区域交给
    /// 其中一个，"至少有一个在用"是唯一安全的答案。
    #[test]
    fn one_enabled_instance_keeps_the_renderer_active() {
        assert_eq!(decide(2, 1, 0, ChainGate::Open), FxBypass::Active);
        assert_eq!(decide(1, 1, 0, ChainGate::Open), FxBypass::Active);
    }

    /// 三种"宿主不会读我们的输出"都必须报旁通。
    #[test]
    fn every_way_the_host_stops_reading_us_reports_bypass() {
        // 单个 FX 被旁通。
        assert_eq!(decide(1, 0, 0, ChainGate::Open), FxBypass::Bypassed);
        // FX 离线（不加载）。
        assert_eq!(decide(1, 1, 1, ChainGate::Open), FxBypass::Bypassed);
        // 整条 FX 链被关掉。
        assert_eq!(decide(1, 1, 0, ChainGate::Closed), FxBypass::Bypassed);
    }

    /// **整链总开关"没有读数"不是"关掉"。**
    ///
    /// 【为什么这条是 take FX 那次静音的回归】take 链没有 `I_FXEN` 的对应物，而旧实现
    /// 按轨道的形状去读了一个不存在的参数（`B_TAKEFX`）。宿主对**未知**参数一律返回
    /// 0.0，于是"整链总开关"读成 0 ⇒ 判旁通 ⇒ `prepare()` 直接返回 ⇒ **一点声音都
    /// 没有**，而日志里只有一行 `unknown -> bypassed`。判据必须能表达"这一位不存在"。
    #[test]
    fn a_chain_without_a_master_switch_never_closes_the_gate() {
        let take = FakeChain::without_master_switch(&[("VST3: HiFiShifter", true, false)]);
        let reading = read_chain(&take, "hifishifter").unwrap();
        assert_eq!(reading.verdict, FxBypass::Active);
        assert_eq!(reading.gate, ChainGate::Unknown);
        // 逐 FX 的读数照旧说话：用户真的旁通了那个 FX，就必须报旁通。
        let bypassed = FakeChain::without_master_switch(&[("VST3: HiFiShifter", false, false)]);
        assert_eq!(
            read_chain(&bypassed, "hifishifter").unwrap().verdict,
            FxBypass::Bypassed
        );
        // 离线同理。
        let offline = FakeChain::without_master_switch(&[("VST3: HiFiShifter", true, true)]);
        assert_eq!(
            read_chain(&offline, "hifishifter").unwrap().verdict,
            FxBypass::Bypassed
        );
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
        assert_eq!(
            read_chain(&other, "hifishifter").unwrap().verdict,
            FxBypass::Unknown
        );
        let empty = FakeChain::new(&[]);
        assert_eq!(
            read_chain(&empty, "hifishifter").unwrap().verdict,
            FxBypass::Unknown
        );
        let bypassed = FakeChain::new(&[("VST3: HiFiShifter", false, false)]);
        assert_eq!(
            read_chain(&bypassed, "hifishifter").unwrap().verdict,
            FxBypass::Bypassed
        );
        let active = FakeChain::new(&[("VST3: HiFiShifter", true, false)]);
        assert_eq!(
            read_chain(&active, "hifishifter").unwrap().verdict,
            FxBypass::Active
        );
    }

    /// 两条链的**同一条**判定：只要本插件在链上，整链总开关就说话。
    ///
    /// 【为什么值得单测】`chain_enabled` 是唯一一个"每个 FX 都正常、但整条链被关掉"
    /// 的读数。它错了，用户会在 FX 链窗口里关掉整条链、插件照旧跑满 CPU。
    #[test]
    fn the_chain_toggle_closes_the_gate_even_when_every_fx_is_enabled() {
        let mut chain = FakeChain::new(&[("VST3: HiFiShifter", true, false)]);
        chain.gate = ChainGate::Closed;
        assert_eq!(
            read_chain(&chain, "hifishifter").unwrap().verdict,
            FxBypass::Bypassed
        );
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

    /// 诊断行必须把"凭什么"写全：结论、匹配数/总数、启用数、离线数、整链开关。
    ///
    /// 【为什么值得钉住字符串】它是排查"没声音"时唯一的一手证据；把 `gate=unknown`
    /// 写成 `gate=open` 会让下一个人以为宿主明确说了"链是通的"。
    #[test]
    fn the_reading_names_the_fact_that_decided_it() {
        let take = FakeChain::without_master_switch(&[("VST3: HiFiShifter", true, false)]);
        assert_eq!(
            read_chain(&take, "hifishifter").unwrap().summary(),
            "found=1/1 enabled=1 offline=0 gate=unknown"
        );
        let mixed = FakeChain::new(&[
            ("VST3: HiFiShifter", true, false),
            ("VST3: Other", true, false),
        ]);
        assert_eq!(
            read_chain(&mixed, "hifishifter").unwrap().summary(),
            "found=1/2 enabled=1 offline=0 gate=open"
        );
        assert_eq!(ChainGate::Open.as_str(), "open");
        assert_eq!(ChainGate::Closed.as_str(), "closed");
        assert_eq!(ChainGate::Unknown.as_str(), "unknown");
        assert!(!ChainGate::Unknown.is_closed());
        assert!(ChainGate::Closed.is_closed());
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
