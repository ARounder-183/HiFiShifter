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
    /// 宿主既不提供原生扩展、也没有把本实例绑定到 ARA。
    ///
    /// 【为什么这是一等身份而不是"未绑定"】它描述的是宿主**把音频交给我们的方式**：
    /// 不是逐个 region 授权（那是 ARA），而是把播放中的音频从 VST3 输入总线流过来。
    /// 插件据此把音频捕获进自己的时间轴（Transfer 通路）。把它与"尚未绑定任何后端"
    /// 混为一谈，会让前端在**受支持**的形态下显示"会话不可用"。
    Transfer,
}

impl HostKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            HostKind::Reaper => "reaper",
            HostKind::AraOnly => "ara_only",
            HostKind::Transfer => "transfer",
        }
    }

    /// 片段几何是否允许写回宿主。
    ///
    /// 【为什么挂在身份上而不是只放能力位】ARA 2.0 的契约不允许插件改写 region 几何，
    /// 所以"ARA-only ⇒ 只读"是**身份蕴含**的结论，不是某一台宿主碰巧缺了某个 API。
    /// 两者分开表达会让"ARA-only 但能力位写错成可写"变成一个可编译的状态。
    /// Transfer 同理：**根本没有宿主片段可写** —— 素材是我们自己录进来的。
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
    /// 插件能**导出处理结果到文件**。
    ///
    /// 【为什么它属于能力位而不是恒真】导出只读已授权的 PCM 再写用户选定的文件，
    /// 不需要任何宿主写接口 —— 因此它在 ARA-only 宿主上也成立。把它显式列出来，
    /// 是为了让前端的 `project.export` 判据有**一个来源**（此前它被 `project.*`
    /// 前缀规则整段拒掉，`Ctrl+E` 因此是死的）。
    pub export_audio: bool,
    /// 插件能导出**纯数据文件**（布局 / 主题 / 颤音预设）。
    ///
    /// 【为什么与 `export_audio` 分开】前者要渲染音频，后者只把前端算好的 JSON
    /// 写成文件；两者失败面完全不同，混用会让"JSON 导出能用"被音频导出拖住。
    pub export_data: bool,
    /// 后端从 VST3 输入总线取得素材（Transfer 通路）。
    ///
    /// 【为什么它是一个能力位而不是恒真的"是不是 Transfer 形态"】它与 `HostKind::Transfer`
    /// 回答的是两个不同的问题：身份说"我是哪个后端"，这一位说"这条通路**当前可用**"。
    /// 在 Transfer 后端上它可能是 `false` —— 宿主从不提供 `ProcessContext`（拿不到工程
    /// 位置）时录制无法工作，前端要据此**禁用按钮并说明原因**，而不是让用户对着一个
    /// 点得动、永远不进度的控件。
    pub transfer: bool,
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
            export_audio: false,
            export_data: false,
            transfer: false,
        }
    }

    /// ARA-only 后端：能看 / 能听 / 能渲染 / 能调参 / 能导出，片段几何只读。
    ///
    /// 【为什么 `transport` 是 true】VST3 的 `ProcessContext` 提供播放状态与工程时间，
    /// 与 ARA 无关，任何宿主都给了 —— 所以非 REAPER 宿主上播放光标照常走。
    /// 【为什么导出是 true】导出只读已授权的 PCM（`ARAAudioAccessControllerInterface`
    /// 对读取**没有角色限制**）再写用户选定的文件，不碰任何宿主写接口。
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
            export_audio: true,
            export_data: true,
            transfer: false,
        }
    }

    /// Transfer 后端（宿主既无原生扩展、也没绑定 ARA）。
    ///
    /// 【为什么 `transport` 是 true】与 `ara_only` 同一条理由：VST3 的 `ProcessContext`
    /// 与 ARA 无关，任何宿主都提供播放状态与工程时间。而且对 Transfer 来说它不只是
    /// "光标能走"—— 它是**录制能工作的前提**：音频按 `projectTimeSamples` 对齐到
    /// 插件时间轴，没有它整条通路不成立。
    ///
    /// 【为什么 `clip_geometry` 是 false，而 `ara_only` 是 true】这是两种形态最本质的
    /// 差别。ARA 宿主交给我们 region 几何（位置 / 长度 / 源窗），我们只是不能改；
    /// Transfer 宿主**一个片段都不告诉我们** —— 时间轴上的片段是我们在录制时按
    /// 连续性自己切出来的。声明 true 会让前端去读一份不存在的几何。
    ///
    /// 【为什么导出仍是 true】与 `ara_only` 同源：导出只渲染自己的 PCM 再写用户选定的
    /// 文件，不碰任何宿主接口。
    ///
    /// 【为什么其余是 false】没有 item 剪贴板、没有建轨、没有导入、没有宿主撤销栈 ——
    /// 宿主连片段都不给我们，这些概念在它那里不存在。这是**诚实的"没有"**。
    pub const fn transfer_backend() -> Self {
        Self {
            transport: true,
            clip_geometry: false,
            clip_write: false,
            split: false,
            undo_stack: false,
            media_import: false,
            track_create: false,
            item_clipboard: false,
            fade_axes: false,
            export_audio: true,
            export_data: true,
            transfer: true,
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
        assert_eq!(
            serde_json::to_value(HostKind::Transfer).unwrap(),
            serde_json::json!("transfer")
        );
        assert_eq!(HostKind::Transfer.as_str(), "transfer");
    }

    /// Transfer 后端的能力契约：**能录、能听、能导出，但没有任何宿主片段几何**。
    ///
    /// 【为什么钉死 `clip_geometry == false`】这是 Transfer 与 ARA-only 最本质的差别，
    /// 也是最容易顺手写错的一位（照抄 `ara_only()` 就会写成 true）。写成 true 的后果
    /// 不是崩溃，而是前端去读一份**从来不存在**的宿主几何：时间线空着，却没有任何
    /// 解释。这条断言把那个错误变成编译期/测试期可见。
    #[test]
    fn transfer_backend_captures_audio_but_owns_no_host_geometry() {
        let caps = HostCapabilities::transfer_backend();
        assert!(caps.transfer, "这一位就是本后端存在的理由");
        assert!(
            caps.transport,
            "ProcessContext 的工程时间是录制对齐的前提，不是可选装饰"
        );
        assert!(
            !caps.clip_geometry,
            "Transfer 宿主一个片段都不给 —— 片段是我们按录制连续性切出来的"
        );
        assert!(!caps.clip_write);
        assert!(!caps.split);
        assert!(!caps.undo_stack, "本地栈承载撤销");
        assert!(!caps.media_import);
        assert!(!caps.track_create);
        assert!(!caps.item_clipboard);
        assert!(!caps.fade_axes);
        assert!(caps.export_audio, "导出只渲染自己的 PCM，不碰宿主接口");
        assert!(caps.export_data);
        assert!(!HostKind::Transfer.clip_write_allowed());
    }

    /// `transfer` 这一位只在 Transfer 后端上为真 —— 它不能靠"是不是别的后端"推出来。
    #[test]
    fn only_the_transfer_backend_claims_the_capture_capability() {
        assert!(!HostCapabilities::none().transfer);
        assert!(!HostCapabilities::ara_only().transfer);
        assert!(HostCapabilities::transfer_backend().transfer);
    }
}
