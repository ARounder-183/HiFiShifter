//! 端口层：**平台无关**的两个 trait 与一个能力自述。
//!
//! 【为什么 Windows 上有一批 §dead_code§】本文件的消费者是 macOS/Linux 的两个原语
//! （自定义 scheme、导航白名单、§FloatingWindow§ 降级）。Windows 用 WebView2 的虚拟
//! 主机，走不到它们 —— 但它们**必须**继续编译，否则等到那个平台时才腐烂。
#![cfg_attr(windows, allow(dead_code))]
//!
//! 与 `editor::host_key_relay` 同构的三层：
//!
//! | 层 | 是什么 | 有没有 `cfg` |
//! |---|---|---|
//! | Policy（`client.rs`） | 校验、分流、回执泵、预算、撤销门、事件队列 | **没有** |
//! | Port（本文件） | 两个 trait：谁收消息、谁给平台能力 | **没有** |
//! | Primitive（`windows.rs` / `macos.rs` / `linux.rs`） | 建原生视图、注入脚本、把 JS 回调转给 Policy | 各文件自己 |
//!
//! 【为什么 trait 里没有 `hwnd`/`NSView`/`GtkWidget`】三个平台的句柄类型互不相干，
//! 而 Policy 只需要一个**不透明整数**：它被原样交给文件对话框与让路原语，
//! 谁认识它由各自的实现决定。用整数而不是泛型，是因为 `ViewState` 只能存
//! `Box<dyn EditorHost>`（一个具体类型），泛型会让它退化成三个 `cfg` 分支。

/// 本编辑器视图的平台句柄：Windows `HWND`、macOS 容器 `NSView*`、Linux `GtkWidget*`。
///
/// 【为什么是 `usize` 而不是裸指针】它会被存进 `EditorClient` 并转交给
/// `browser_files`（那里的四个平台服务只认一个整数）；裸指针会让这些结构体
/// 全部变成 `!Send` 的传播源，而它们本来就不需要解引用 —— 解引用发生在平台实现内部。
pub(crate) type NativeWindow = usize;

/// 平台自己说它能做到哪一步。
///
/// 【为什么不许上层到处 `cfg`】"能不能内嵌""要不要提示用户"是**平台事实**，
/// 不是调用点的偏好。让平台自报一次，Policy 与 `view.rs` 只读这一个枚举，
/// 新增平台时不必回头改判定。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GuiFidelity {
    /// 完整内嵌：视图就在宿主给的 parent 里。
    Embedded,
    /// 只能开自己的顶层窗口（Wayland 兜底、以及 GTK 只能跑在私有线程上的宿主）。
    ///
    /// 【为什么它也是一种"降级"而不只是"另一种样子"】泵不在宿主 UI 线程上，
    /// 于是 `flush_dirty` / `host_undo.tick()` 只能由宿主的下一次 `IPlugView` 回调补做，
    /// 宿主 Undo 的合并粒度会变粗。功能可用，语义有损 —— 所以它要能被看见（日志一行）。
    FloatingWindow,
    /// 没有 GUI：宿主显示空面板，日志里有一行**可搜到的**原因。
    Unavailable(&'static str),
}

impl GuiFidelity {
    /// 诊断/日志用的稳定短名。
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            GuiFidelity::Embedded => "embedded",
            GuiFidelity::FloatingWindow => "floating-window",
            GuiFidelity::Unavailable(reason) => reason,
        }
    }
}

/// 把一条 JSON 交给页面。
///
/// 【为什么入参是**已序列化的字符串**】WebKit 的结构化 `postMessage` 会走
/// `NSDictionary`/`JSCValue` 转换，`u64` 的 id 在转换里丢精度（`9007199254740993` 变成
/// `…992`），回执就对不上了。字符串是三条链路（WebView2 / WKWebView / WebKitGTK）上
/// 唯一逐字节一致的形式，所以 Policy 只产出字符串，端口只负责搬运。
pub(crate) trait EditorPort {
    /// 返回是否真的投递出去了。页面未就绪或投递失败都返回 `false`。
    ///
    /// 【为什么不让端口自己排队】排队策略（上限、丢最旧、就绪门）是**判定**，
    /// 属于 Policy；端口只回答"这一条现在能不能发"。
    fn post_message(&self, text: &str) -> bool;

    /// 平台级错误提示：Windows 画一个 `STATIC` 子窗口，其它平台写日志并让页面显示一条横幅。
    ///
    /// 【为什么是端口职责】"怎么让用户看见一句错误"完全是平台的（Windows 用子窗口、
    /// macOS/Linux 只能走页面），而"什么时候要报错"是 Policy 的。
    fn show_error(&self, text: &str);

    /// 拆掉原生侧的一切（控制器 / 页面 / 窗口）。必须可重入。
    fn close(&self);
}

/// 宿主（`view.rs`）持有的一份视图句柄。
///
/// 【为什么不要求 `Send`】VST3 保证 `IPlugView` 的全部方法都来自调用 `attached` 的
/// 同一线程（`docs/thread-model.md`），所以这个句柄**只属于那条线程**；给它加 `Send`
/// 只会逼 macOS/Linux 的实现去藏 `Rc`（见 `linux.rs` 的注册表），换不来任何真实保证。
/// 需要"从任意线程投递"的地方（情形 3 的 GTK 线程）走的是通道，不是这个句柄。
pub(crate) trait EditorHost {
    fn fidelity(&self) -> GuiFidelity;

    /// 视图尺寸变化。macOS 的实现负责把物理像素换算成逻辑点。
    fn resize(&self, width: i32, height: i32) -> Result<(), String>;

    /// 宿主明确交付焦点时进入 WebView。
    fn focus(&self) -> Result<(), String>;

    /// 宿主投递的共享编辑键（Ctrl+C/X/V/Z/Y）。返回是否消费。
    fn forward_edit_key(&self, key: char, mods: i16, down: bool) -> bool;

    /// 宿主侧 20 ms 补做：只在**泵不在宿主线程上**时才有意义（见 `GuiFidelity`）。
    ///
    /// 【为什么放在 trait 上而不是 Policy 里】它要碰宿主 API（`flush_dirty` /
    /// `host_undo.tick()`），而那必须发生在宿主 UI 线程 —— 情形 3 里 Policy 在另一条
    /// 线程上，够不着。
    fn host_tick(&self);

    /// 关窗。必须可重入、可在已关闭状态下重复调用。
    fn close(&self);
}
