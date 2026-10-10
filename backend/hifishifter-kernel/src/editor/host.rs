//! 编辑后端抽象：两侧共用的"我是哪个宿主、我能做什么"。
//!
//! 【为什么下沉到 kernel】独立 app 与 ARA 插件都要回答同一个问题：当前后端能读、能写
//! 哪些东西。此前插件把这套判断写死在 `ReaperHost` 的 `Option` bundle 里，于是
//! "没有 REAPER" 与 "有 REAPER 但缺某个 API" 无法区分，两种退化形态都退成"空时间线"。
//! 把**能力词汇表与后端身份**抽到内核，两侧共用同一份定义，插件侧再各自实现。
//!
//! 【边界】这里只放词汇表与身份（[`HostKind`] / [`HostCapabilities`] / [`EditorHost`]），
//! **不放任何宿主 API 调用** —— 那属于各后端的实现（插件侧的 REAPER 适配器、ARA-only
//! 后端，app 侧的文件后端）。
//!
//! 【绝不按宿主名分支】[`EditorHost::app_name`] 只用于诊断（"你是哪个宿主"），任何
//! 能力判断都必须走 [`EditorHost::capabilities`]。宿主名是任意字符串，不可枚举。

use serde::Serialize;

/// 当前编辑后端的身份。
///
/// 【为什么需要显式身份，而不是"一堆 `Option` 的布尔组合"】能力位能回答"有没有这个
/// API"，却回答不了"我到底是哪个后端"。前端要据此决定**只读横幅**与历史面板的措辞，
/// 诊断包也要能直接读出后端种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HostKind {
    /// REAPER 的原生 VST3/ARA 扩展：有 item GUID，可写回片段几何。
    Reaper,
    /// 仅提供 ARA 2.0 的通用宿主：能看 / 能听 / 能渲染 / 能调参，片段几何只读。
    AraOnly,
}

impl HostKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            HostKind::Reaper => "reaper",
            HostKind::AraOnly => "ara_only",
        }
    }

    /// 片段几何是否允许写回宿主。
    ///
    /// 【为什么挂在身份上而不是只放能力位】ARA 2.0 的契约不允许插件改写 region 几何，
    /// 所以"ARA-only ⇒ 只读"是**身份蕴含**的结论，不是某一台宿主碰巧缺了某个 API。
    /// 两者分开表达会让"ARA-only 但能力位写错成可写"变成一个可编译的状态。
    pub const fn clip_write_allowed(self) -> bool {
        matches!(self, HostKind::Reaper)
    }
}

/// 一个后端能做什么。每个字段对应一条**独立的**用户可见操作族。
///
/// 【为什么是固定字段而不是 `HashMap<&str, bool>`】能力位是编译期契约：加一个能力就该
/// 让所有后端实现显式表态，而不是默默读到 `None` 当成"不支持"。固定字段让漏掉的那一
/// 个后端编译不过。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HostCapabilities {
    /// 宿主提供实时播放位置 / 播放状态。
    pub transport: bool,
    /// 宿主提供片段几何（位置 / 长度 / 源窗 / 倍率）。
    pub clip_geometry: bool,
    /// 宿主允许写回片段几何。
    pub clip_write: bool,
    /// 宿主允许分割片段。
    pub split: bool,
    /// 撤销历史由宿主权威栈承载（`false` ⇒ 走插件自管的本地栈）。
    pub undo_stack: bool,
    /// 宿主允许导入音频（会建轨、建 item）。
    pub media_import: bool,
    /// 宿主允许新建一条宿主轨道。
    pub track_create: bool,
    /// 宿主提供完整的 item 剪贴板。
    pub item_clipboard: bool,
    /// 宿主提供淡变形状轴（`fade_axes_v2` 那个布尔之外，"有没有"这一层）。
    pub fade_axes: bool,
}

impl HostCapabilities {
    /// 什么都不支持：用于"尚未绑定任何后端"。
    pub const fn none() -> Self {
        Self {
            transport: false,
            clip_geometry: false,
            clip_write: false,
            split: false,
            undo_stack: false,
            media_import: false,
            track_create: false,
            item_clipboard: false,
            fade_axes: false,
        }
    }

    /// ARA-only 后端：能看 / 能听 / 能渲染 / 能调参，片段几何只读。
    ///
    /// 【为什么 `transport` 是 true】VST3 的 `ProcessContext` 提供播放状态与工程时间，
    /// 与 ARA 无关，任何宿主都给了 —— 所以非 REAPER 宿主上播放光标照常走。
    /// 【为什么其余是 false】ARA 2.0 不允许插件改写 region 几何，也没有 item 剪贴板 /
    /// 建轨 / 导入的宿主级入口。这些能力在 ARA-only 后端上是**诚实的"没有"**，不是
    /// "还没实现"——前端据此禁用对应控件，而不是让用户点了之后撞一个字符串错误。
    pub const fn ara_only() -> Self {
        Self {
            transport: true,
            clip_geometry: true,
            clip_write: false,
            split: false,
            undo_stack: false,
            media_import: false,
            track_create: false,
            item_clipboard: false,
            fade_axes: false,
        }
    }
}

/// 一个编辑后端：报告身份与能力，并提供诊断用的宿主名 / 版本。
///
/// 【为什么是 trait】REAPER 适配器与 ARA-only 后端是两条独立的实现，插件启动时按
/// "宿主有没有给出原生扩展"二选一。trait 让调用点只面对"当前后端"，不必知道背后是哪
/// 一套 API。
pub trait EditorHost: Send + Sync {
    /// 后端身份。
    fn kind(&self) -> HostKind;

    /// 后端能力。
    fn capabilities(&self) -> HostCapabilities;

    /// 宿主应用名（诊断用；**绝不用于分支逻辑**）。
    fn app_name(&self) -> Option<&str> {
        None
    }

    /// 宿主原始版本串（诊断用）。
    fn app_version(&self) -> Option<&str> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ARA-only 后端的能力契约：能看/能听/能渲染/能调参，但**写不进去**。
    ///
    /// 【为什么钉死】这是"非 REAPER 宿主上编辑只读"这条用户可见承诺的唯一出处。
    /// 一旦有人把某个写能力位改成 true，界面就会给出一个必然失败的入口。
    #[test]
    fn ara_only_reports_read_only_geometry_but_keeps_observation() {
        let caps = HostCapabilities::ara_only();
        assert!(caps.transport, "播放光标在任意 ARA 宿主上都要走");
        assert!(caps.clip_geometry, "ARA 直接给出 region 几何");
        assert!(!caps.clip_write, "ARA 2.0 不允许插件改 region 几何");
        assert!(!caps.split);
        assert!(!caps.undo_stack, "本地栈承载撤销");
        assert!(!caps.media_import);
        assert!(!caps.track_create);
        assert!(!caps.item_clipboard);
        assert!(!caps.fade_axes);
        assert!(!HostKind::AraOnly.clip_write_allowed());
        assert!(HostKind::Reaper.clip_write_allowed());
    }

    /// 未绑定后端时是"全否"，不是"全有"。
    #[test]
    fn unbound_capabilities_are_all_false() {
        let caps = HostCapabilities::none();
        assert_eq!(caps, HostCapabilities::default());
        assert!(!caps.transport && !caps.clip_geometry && !caps.clip_write);
    }

    /// 能力位序列化成前端认得的 camelCase。
    #[test]
    fn capabilities_serialize_in_camel_case() {
        let json = serde_json::to_value(HostCapabilities::ara_only()).unwrap();
        assert_eq!(json["clipWrite"], false);
        assert_eq!(json["clipGeometry"], true);
        assert_eq!(json["undoStack"], false);
    }

    /// 身份序列化成 snake_case 字符串。
    #[test]
    fn host_kind_serializes_as_a_stable_token() {
        assert_eq!(
            serde_json::to_value(HostKind::AraOnly).unwrap(),
            serde_json::json!("ara_only")
        );
        assert_eq!(HostKind::Reaper.as_str(), "reaper");
    }
}
