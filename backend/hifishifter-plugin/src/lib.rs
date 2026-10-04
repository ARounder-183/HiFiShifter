//! HiFiShifter 的 ARA 插件侧适配层。
//!
//! 目标（设计文档 §5.1）：让 HiFiShifter 以 ARA2 插件形式挂在 DAW 里，时间线内容
//! 随 DAW 走，并**复用本体已经存在的离线内核**，而不是另写一套渲染。
//!
//! 本 crate 目前只有两块：
//! - [`ara`]：ARA 文档模型 → `TimelineState` 的映射（探针 Task 3 的产品化版本）；
//! - [`render`]：把 `TimelineState` 交给本体内核离线渲染的薄封装。
//!
//! 边界：这里**不**做设备 I/O，也**不**依赖 Tauri IPC。插件**只**依赖
//! `hifishifter-kernel` —— 这是"插件不把 WebView2 / Tauri 带进 DAW 进程"真正成立的时刻，
//! 由 `tests/no_tauri_in_dependency_tree.rs` 钉住。宿主回调替代 cpal 的职责是下一步的事。
//!
//! # 结构
//!
//! - [`vst3`]：手写的最小 VST3 模块 ABI（工厂 + 组件 + 处理器 + 最小编辑控制器）。
//!   三条 ABI 硬事实写在它的文件头，都是探针期用崩溃换来的。
//! - [`runtime`]：进程级的 ARA 工厂与 companion 关联。
//! - [`ara`]：宿主模型回调的累积（`model`）与 `ARA → TimelineState` 的映射（`mapping`）。
//! - [`render`]：把 `TimelineState` 交给内核离线渲染的薄封装。
//!
//! 【为什么 VST3 外壳要手写】`ara2-bridge` 给的是 ARA↔VST3 的**桥**，不是可加载模块。
//! 模块入口（`GetPluginFactory` / `InitDll` / `ExitDll`）与最小组件必须自备。

pub mod ara;
pub mod render;
mod diagnostics;
mod audio_abi;
mod ara_entry;
#[cfg(test)]
mod test_host;
#[cfg(test)]
mod test_allocator;
mod runtime;
mod vst3;

use std::ffi::c_void;

/// VST3 类名。**必须与 ARA 工厂的 plugInName 一致** —— ARA 的 VST3 配对规则要求如此。
pub(crate) const CLASS_NAME: &str = "HiFiShifter";
/// 版本号（展示用）。
pub(crate) const VERSION: &str = env!("CARGO_PKG_VERSION");
/// ARA 工厂的持久 ID。
pub(crate) const FACTORY_ID: &str = "com.hifishifter.ara";
/// ARA 文档归档 ID。
pub(crate) const ARCHIVE_ID: &str = "com.hifishifter.ara.archive";

/// 写一行诊断信息。
///
/// 【为什么不用 `println!`】插件跑在宿主进程里，stdout 通常无处可去。`log` 会走宿主
/// 已装配的日志后端；宿主没装时它是静默丢弃，不会因为"日志写不出去"影响编曲。
pub(crate) fn log_line(message: &str) {
    log::info!("[vst3] {message}");
}

/// VST3 模块入口：返回 `IPluginFactory*`。
///
/// 【为什么先 `runtime()`】工厂查询可能早于任何组件创建；先把 ARA 工厂建好，
/// 后面 VST3 侧查 `IPlugInEntryPoint` 时才有东西可给。
#[no_mangle]
pub extern "system" fn GetPluginFactory() -> *mut c_void {
    diagnostics::init();
    let _ = runtime::runtime();
    vst3::get_plugin_factory()
}

/// VST3 模块初始化入口（REAPER 在 `LoadLibrary` 之后显式调用）。
///
/// 【为什么初始化失败也返回 `true`】让宿主把类列出来，失败原因记在日志里。返回
/// `false` 会让 REAPER 直接跳过插件 —— 那就什么信息都拿不到了。探针期实测过这个取舍。
#[no_mangle]
pub extern "system" fn InitDll() -> bool {
    diagnostics::init();
    let _ = runtime::runtime();
    true
}

/// VST3 模块卸载入口。
#[no_mangle]
pub extern "system" fn ExitDll() -> bool {
    true
}
