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
//! REAPER 给出这一族读数：`TrackFX_GetCount` / `TrackFX_GetEnabled` /
//! `TrackFX_GetOffline` / `TrackFX_GetFXName`，以及轨道的 `I_FXEN`（整条 FX 链的总开关）。
//! 本模块只把它们收成一个三态结论；判定本身是纯函数（[decide]），可以在没有宿主的前提下
//! 逐值断言。
//!
//! # 为什么是三态而不是布尔
//!
//! "读不出来"与"没有被旁通"必须分开。前者（旧宿主、API 缺失、授权收回、这条轨上找不到
//! 本插件的 FX）一旦被当成后者，插件就会在一个用户已经旁通的实例上继续跑神经合成 ——
//! 这正是本模块要消除的浪费；反过来把它当成"已旁通"，用户会**听不到声音**却没有任何提示。
//! 三态让两个方向都不会被静默猜错：`Unknown` 走改动前的行为（照常渲染）。

use super::checked;
use std::ffi::{c_char, c_void};

/// `int TrackFX_GetCount(MediaTrack*)`
pub(super) type FxCount = unsafe extern "C" fn(*mut c_void) -> i32;
/// `bool TrackFX_GetEnabled(MediaTrack*, int fx)`
pub(super) type FxEnabled = unsafe extern "C" fn(*mut c_void, i32) -> bool;
/// `bool TrackFX_GetOffline(MediaTrack*, int fx)`
pub(super) type FxOffline = unsafe extern "C" fn(*mut c_void, i32) -> bool;
/// `bool TrackFX_GetFXName(MediaTrack*, int fx, char* buf, int buf_sz)`
pub(super) type FxName = unsafe extern "C" fn(*mut c_void, i32, *mut c_char, i32) -> bool;
/// `double GetMediaTrackInfo_Value(MediaTrack*, const char* parmname)`
pub(super) type TrackValue = unsafe extern "C" fn(*mut c_void, *const c_char) -> f64;

/// 读 FX 旁通所需的入口；缺任何一个都整体不可用（单独降级没有意义）。
pub(super) struct FxStateApi {
    pub count: FxCount,
    pub enabled: FxEnabled,
    pub offline: FxOffline,
    pub name: FxName,
    pub track_value: TrackValue,
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

/// 由四份读数得出结论。
///
/// 入参都是**已经核实过的**事实：
/// * `found` —— 这条轨上有几个 FX 是本插件（按类名匹配到的个数）；
/// * `any_enabled` —— 其中是否至少有一个是启用的；
/// * `all_offline` —— 其中是否全部处于离线（offline = 不加载、不处理）；
/// * `chain_enabled` —— 轨道的 FX 链总开关（`I_FXEN`）。
///
/// 判定取"**全部**证据都指向旁通才报旁通"：
/// * 一个都没匹配到 ⇒ 不知道（可能是宿主改了显示名，也可能这条轨上挂的是别的实例）；
/// * 只要有一个同类 FX 是启用的 ⇒ 活跃（同一轨道挂两个本插件实例时宿主只会把区域交给
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

/// 在**宿主线程**上读一次旁通结论。
///
/// 调用方保证：`authorized` 在整段读取期间可反复求值；`track` 是本实例的直接 parent
/// 轨道指针（`HostVtbl::parent(interface, 1)`），由调用方在同一线程上取得。
pub(super) fn read(
    api: &FxStateApi,
    track: *mut c_void,
    class_name: &str,
    authorized: &impl Fn() -> bool,
) -> Result<FxBypass, String> {
    let count = checked(authorized, || unsafe { (api.count)(track) })?;
    if count <= 0 {
        return Ok(FxBypass::Unknown);
    }
    let needle = class_name.to_ascii_lowercase();
    let mut found = 0_usize;
    let mut any_enabled = false;
    let mut all_offline = true;
    let mut buffer = [0_i8; 256];
    for index in 0..count {
        let read_name = checked(authorized, || unsafe {
            (api.name)(track, index, buffer.as_mut_ptr(), buffer.len() as i32)
        })?;
        if !read_name {
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
        let enabled = checked(authorized, || unsafe { (api.enabled)(track, index) })?;
        let offline = checked(authorized, || unsafe { (api.offline)(track, index) })?;
        any_enabled |= enabled;
        all_offline &= offline;
    }
    if found == 0 {
        return Ok(FxBypass::Unknown);
    }
    let chain_enabled = checked(authorized, || unsafe {
        (api.track_value)(track, c"I_FXEN".as_ptr())
    })? != 0.0;
    Ok(decide(found, any_enabled, all_offline, chain_enabled))
}

#[cfg(test)]
mod tests {
    use super::*;

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
