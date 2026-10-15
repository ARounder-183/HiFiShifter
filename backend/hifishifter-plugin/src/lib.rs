//! HiFiShifter 的 ARA 插件侧适配层。
//!
//! 目标（设计文档 §5.1）：让 HiFiShifter 以 ARA2 插件形式挂在 DAW 里，时间线内容
//! 随 DAW 走，并**复用本体已经存在的离线内核**，而不是另写一套渲染。
//!
//! 本 crate 目前只有两块：
//! - [`ara`]：ARA 文档模型 → `TimelineState` 的映射（探针 Task 3 的产品化版本）；
//! - [`render`]：把 `TimelineState` 交给本体内核离线渲染的薄封装。
//!
//! 边界：不做设备I/O，不依赖Tauri IPC/app runtime。二期增加原生WebView2内嵌原GUI，
//! 不接管宿主消息循环；A5-v2禁止Tauri/wry/app/cpal，由依赖树守卫钉住。
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

// ── clippy 策略 ─────────────────────────────────────────────────────────────
//
// 【为什么要有这一块】CI 的 clippy 步骤对**整个 workspace** 判死（`-D warnings`），
// 而本 crate 此前从未被 lint 过 —— 它的 build.rs 要求 ARA/VST3 SDK，工作区里其余
// crate 不需要，于是它的告警长期无人看见。能修的都已在源码里修掉；剩下的是
// **有意识接受的风格类 lint**，逐条写在这里，让缺陷类 lint 保持默认告警级别。
//
// 每一条都是判断，不是省事：
#![allow(
    // 播放快照按份 `Box` 分配：`AtomicPtr<PlaybackSnapshot>` 直接指向盒内对象，
    // 换成 `Vec<PlaybackSnapshot>` 会在扩容时搬移元素，让已发布的裸指针失效。
    // 测试夹具里的 `Vec<Box<u8>>` 同理 —— 那是"不透明宿主句柄"的占位，不是容器。
    clippy::vec_box,
    // 逐元素数值循环里索引是语义的一部分（相邻样本、跨通道步长、原地读写），
    // 改成迭代器往往要引入 windows()/zip()，反而更难核对边界。
    clippy::needless_range_loop,
    // 闭包/迭代器签名的类型确实复杂，抽 `type` 别名会让"它到底是什么"更难读。
    clippy::type_complexity,
    // 测试模块紧邻被测代码、生产代码随后。重排只是纯移动 diff
    // （`render/extension.rs` 一处就是 1700 行），没有语义收益。
    clippy::items_after_test_module,
    // 测试等待循环用 `while <不变量>` 表达"轮询到满足为止"，超时由循环体内的
    // deadline 断言负责；改写成 `loop` 会把停止条件藏得更深。
    clippy::while_immutable_condition,
    // `let mut x = X::default(); x.a = 1;` 与结构体更新语法等价；后者在字段多、
    // 默认值集中定义时更难读（要来回对照 Default 实现）。
    clippy::field_reassign_with_default,
)]

pub mod ara;
mod ara_entry;
mod audio_abi;
mod component_state;
mod diagnostics;
mod editor;
mod fade;
pub mod render;
mod runtime;
mod settings_store;
mod state_channel;
mod state_stream;
mod sync;
#[cfg(test)]
mod test_allocator;
#[cfg(test)]
mod test_host;
mod transfer;
mod vst3;

use std::ffi::c_void;

/// VST3 类名。**必须与 ARA 工厂的 plugInName 一致** —— ARA 的 VST3 配对规则要求如此。
pub(crate) const CLASS_NAME: &str = "HiFiShifter";
pub(crate) mod host;
/// 版本号（展示用）。
///
/// 【为什么不是 `CARGO_PKG_VERSION`】产品版本的**单一来源**是
/// `src-tauri/tauri.conf.json`，由 `build.rs` 注入为 `HFS_PRODUCT_VERSION`。读本 crate
/// 的 `CARGO_PKG_VERSION` 会引入**第二份副本**：只改一处就会让插件与独立 App 的
/// 「关于」对话框报出不同版本（已经真实发生过一次）。`tests/version_parity.rs`
/// 再把 Cargo.toml / package.json 等其余副本钉在这一个值上。
pub(crate) const VERSION: &str = env!("HFS_PRODUCT_VERSION");
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

/// FFI 入口的 panic 遏制原语：Rust 1.71+ 起 panic 越过 `extern "system"` 边界 =
/// abort，宿主（DAW）进程随之崩溃。**每一个被宿主调用的入口**都必须把自己的
/// panic 拦成明确的失败值 —— 这不是音频回调的特例，而是全部 ABI 面的规则。
///
/// 【为什么 fallback 由调用方给】不同入口的"诚实失败值"不同（QI 该回
/// `K_NO_INTERFACE`、refcount 该回计数、视图创建该回 null），原语不该替调用方猜。
/// 拦下的 panic 统一记入日志（带 payload 文本），诊断有据可查。
pub(crate) fn guard_ffi<R>(fallback: R, f: impl FnOnce() -> R) -> R {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic payload".to_string());
            log::error!("[ffi-guard] panic contained at FFI boundary: {message}");
            fallback
        }
    }
}

/// VST3 模块入口：返回 `IPluginFactory*`。
///
/// 【为什么先 `runtime()`】工厂查询可能早于任何组件创建；先把 ARA 工厂建好，
/// 后面 VST3 侧查 `IPlugInEntryPoint` 时才有东西可给。
#[no_mangle]
pub extern "system" fn GetPluginFactory() -> *mut c_void {
    crate::guard_ffi(std::ptr::null_mut(), || {
        diagnostics::init();
        editor::resources::initialize_models();
        let _ = runtime::runtime();
        vst3::get_plugin_factory()
    })
}

/// VST3 模块初始化入口（REAPER 在 `LoadLibrary` 之后显式调用）。
///
/// 【为什么初始化失败也返回 `true`】让宿主把类列出来，失败原因记在日志里。返回
/// `false` 会让 REAPER 直接跳过插件 —— 那就什么信息都拿不到了。探针期实测过这个取舍。
/// panic 除外：`guard_ffi` 已把 panic 原文记进日志，此时模块状态未知，宁可让宿主
/// 跳过也不要带着坏 runtime 插进 N 个轨道。
#[no_mangle]
pub extern "system" fn InitDll() -> bool {
    crate::guard_ffi(false, || {
        diagnostics::init();
        editor::resources::initialize_models();
        let _ = runtime::runtime();
        true
    })
}

/// VST3 模块卸载入口。
#[no_mangle]
pub extern "system" fn ExitDll() -> bool {
    crate::guard_ffi(false, editor::shutdown)
}
