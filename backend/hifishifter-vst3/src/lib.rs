//! HiFiShifter 的 VST3 插件外壳。
//!
//! 【边界】本 crate 只认识 VST3 3.8 的模块 ABI（入口符号、工厂、组件/处理器、
//! 编辑控制器、连接点、`IPlugView`）与 ARA↔VST3 的 COM 入口；时间/缓冲/状态/资源
//! 规则一律经 `hifishifter-plugin`（格式无关实现层）与 `hifishifter-plugin-core`。
//! 渲染调度、快照输出与编辑器生命周期都在共享层落地，本 crate 只做 ABI 翻译。
//! ABI 词汇（`TResult` / TUID 布局 / `FUnknown` 前缀）来自共享层的
//! `hifishifter_plugin::vst3_abi` —— REAPER 原生扩展等共享代码也消费它。
//!
//! 【能力边界】宿主支持 ARA 时，本插件以 ARA2 形式挂在 DAW 里；不支持时仍是常规
//! VST3（捕获通路）。不链接 VST3 SDK 的 C++ 运行时（ABI 面手写，见 `vst3` 模块
//! 文档）；不带 Tauri/wry/app runtime。
//!
//! 【为什么本 crate 是独立的外壳】外壳曾住在共享层（`hifishifter-plugin`）里，
//! 后果是其他外壳的产物被链接并导出 VST3 入口符号、CLAP 构建被迫检出 VST3 SDK，
//! 这正是拆分外壳的由来。

// ── clippy 策略 ─────────────────────────────────────────────────────────────
//
// 【为什么要有这一块】CI 的 clippy 步骤对整个 workspace 判死（`-D warnings`）。
// 壳是从共享层整体搬来的（见 `vst3.rs` / `view.rs` / `connection.rs` 的文件头），
// 沿用原 crate 对同一段代码的既有豁免；每一条都是判断，不是省事。
#![allow(
    // 播放快照按份 `Box` 分配的测试夹具（`Vec<Box<u8>>` 是"不透明宿主句柄"的占位）。
    clippy::vec_box,
    // 逐元素数值循环里索引是语义的一部分（相邻样本、跨通道步长、原地读写）。
    clippy::needless_range_loop,
    // 闭包/迭代器签名的类型确实复杂，抽 `type` 别名会让"它到底是什么"更难读。
    clippy::type_complexity,
    // 测试模块紧邻被测代码、生产代码随后。重排只是纯移动 diff，没有语义收益。
    clippy::items_after_test_module,
    // 测试等待循环用 `while <不变量>` 表达"轮询到满足为止"，超时由循环体内的
    // deadline 断言负责；改写成 `loop` 会把停止条件藏得更深。
    clippy::while_immutable_condition,
    // `let mut x = X::default(); x.a = 1;` 与结构体更新语法等价；后者在字段多、
    // 默认值集中定义时更难读（要来回对照 Default 实现）。
    clippy::field_reassign_with_default,
)]

pub mod ara_entry;
pub mod connection;
pub mod view;
pub mod vst3;

use std::ffi::c_void;

/// VST3 类名。**必须与 ARA 工厂的 plugInName 一致** —— ARA 的 VST3 配对规则要求如此。
pub use hifishifter_plugin::CLASS_NAME;

/// 产品版本（展示用）。
///
/// 【为什么不是 `CARGO_PKG_VERSION`】产品版本的**单一来源**是
/// `src-tauri/tauri.conf.json`，由 `build.rs` 注入为 `HFS_PRODUCT_VERSION`。
/// `tests/version_parity.rs` 把所有副本钉在这一个值上。
pub(crate) const VERSION: &str = env!("HFS_PRODUCT_VERSION");

/// 【为什么测试二进制要在进入 main 之前关掉异步预热】libtest 的退出没有收尾钩子，
/// 预热线程在"建 ONNX 会话"中途被拆掉时，整个测试二进制以 `0xC0000005`
/// （Windows）/ SIGSEGV（Linux）退出 —— 即使每条用例都通过。
/// `hifishifter-plugin` 与 `hifishifter-clap` 的测试各有同一份兜底。
#[cfg(test)]
mod test_prewarm_guard {
    #[ctor::ctor(unsafe)]
    fn disarm_async_session_prewarm() {
        hifishifter_kernel::vocoder_ort_session::set_async_session_prewarm(false);
    }
}

/// VST3 bundle 形态的资源根登记 + 进程级初始化。
///
/// 【为什么资源布局在这里而不是共享层】VST3 的模块是 bundle（`Contents/Resources`），
/// 而资源根的唯一实现 `resource_root` 需要知道"本模块在哪、是哪种形态"。CLAP 在它
/// 自己的 `entry_init` 里登记 Sibling 形态 —— 这是两个外壳的**契约差异**（bundle vs
/// 单文件），不是实现漂移。
///
/// 【顺序与 `hifishifter-clap` 的 `entry_init` 逐字同序】资源根 → 进程状态（迁移）→
/// 日志 → 模型 → ARA 运行时。差异只有两处：本壳登记 BundleContents；ARA 运行时
/// 必须在工厂可查询前建好（CLAP 经 `companion_for` 惰性建立也可以，但为对称同样
/// 在入口建立）。
fn register_bundle_layout_and_initialize() {
    // 【为什么资源布局在进程状态之前】`initialize_process_state` 只登记迁移钩子并跑
    // 一次性迁移；资源根要在任何"读本模块旁边的资源"之前就位。
    if let Ok(module) = hifishifter_plugin::editor::resources::module_path() {
        let _ = hifishifter_plugin_core::resource_root::set_layout(
            hifishifter_plugin_core::resource_root::ResourceLayout::BundleContents,
        );
        let _ = hifishifter_plugin_core::resource_root::set_module_path(module);
    }
    hifishifter_plugin::initialize_process_state();
}

/// 模块级初始化：三个平台的入口做的是同一件事，所以只有这一份实现。
///
/// 【为什么失败也要返回 `true`】让宿主把类列出来，原因记在日志里。返回 `false` 会让
/// 宿主直接跳过插件 —— 那就什么信息都拿不到了。探针期实测过这个取舍。
/// panic 除外：`guard_ffi` 已把 panic 原文记进日志，此时模块状态未知，宁可让宿主
/// 跳过也不要带着坏 runtime 插进 N 个轨道。
fn module_initialize() -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        // 【为什么显式登记】格式标签是日志文件名与会话头的唯一来源；VST3 虽是共享层
        // 的历史默认值，显式登记能让"两个格式走同一套逻辑"这件事在入口处可见、且
        // 不会被将来的默认值改动悄悄破坏。
        hifishifter_plugin::set_log_tag("vst3");
        register_bundle_layout_and_initialize();
        hifishifter_plugin::diagnostics::init();
        // 【为什么打这一行】宿主扫描插件也会走到这里。有了这一行，日志就能区分
        // "宿主只是扫描了我"与"宿主真的在用我" —— 前者之后再没有任何行是正常的。
        hifishifter_plugin::log_module_loaded();
        hifishifter_plugin::initialize_resources();
        // 【为什么在模块入口建 ARA 运行时】工厂查询可能早于任何组件创建；先把 ARA
        // 工厂建好，后面 VST3 侧查 `IPlugInEntryPoint` 时才有东西可给。
        let _ = hifishifter_plugin::runtime::runtime();
        true
    })
}

/// 模块级卸载：宿主在最后一次释放模块前的清理点。
///
/// 收尾顺序与 `hifishifter-clap` 的 `entry_deinit` 相同：共享层的
/// `shutdown_module`（编辑器 + 慢车道）先走，再把还攒在缓冲里的日志落盘。
fn module_shutdown() -> bool {
    let result = hifishifter_plugin::shutdown_module();
    hifishifter_plugin::diagnostics::flush_pending();
    hifishifter_plugin::diagnostics::write_session_footer("shutdown");
    result
}

/// VST3 模块入口：返回 `IPluginFactory*`。
///
/// 【为什么先 `runtime()`】工厂查询可能早于任何组件创建；先把 ARA 工厂建好，
/// 后面 VST3 侧查 `IPlugInEntryPoint` 时才有东西可给。
#[no_mangle]
pub extern "system" fn GetPluginFactory() -> *mut c_void {
    hifishifter_plugin_core::guard_ffi(std::ptr::null_mut(), || {
        hifishifter_plugin::set_log_tag("vst3");
        register_bundle_layout_and_initialize();
        hifishifter_plugin::diagnostics::init();
        hifishifter_plugin::log_module_loaded();
        // 【为什么把 IID 布局写进日志】它是"插件能不能和宿主说上话"的前置条件，却完全
        // 不可见：发错布局的失败形态是宿主回一个 null 指针，日志里只剩一行
        // `available=false`。把它钉在会话头附近，排查时第一眼就能排除一整类猜测。
        hifishifter_plugin::log_line(&format!(
            "iid layout: {} ({})",
            hifishifter_plugin::vst3_abi::iid_layout(),
            std::env::consts::OS
        ));
        hifishifter_plugin::initialize_resources();
        let _ = hifishifter_plugin::runtime::runtime();
        vst3::get_plugin_factory()
    })
}

/// VST3 模块初始化入口（Windows：REAPER 在 `LoadLibrary` 之后显式调用）。
#[no_mangle]
pub extern "system" fn InitDll() -> bool {
    hifishifter_plugin_core::guard_ffi(false, module_initialize)
}

/// VST3 模块卸载入口（Windows）。
#[no_mangle]
pub extern "system" fn ExitDll() -> bool {
    hifishifter_plugin_core::guard_ffi(false, module_shutdown)
}

// ── DLL_PROCESS_ATTACH：VEGAS 家族包裹层的线程 DPI 提升 ─────────────────────
//
// 【为什么是 DllMain 而不是 InitDll】`InitDll` 是 VST3 约定入口，REAPER 显式调它；
// 而 ACID/VEGAS 家族的 sfvstwrap 只 LoadLibrary、不保证调它。提升的意义取决于
// "赶在包裹窗口创建之前" —— 入口越早越好，DllMain 是我们能拿到的最早一点。
//
// 【loader lock 纪律】这里只允许 kernel32 的无同步操作（见
// `host_identity::promote_wrapper_thread_dpi` 的说明）：映像名查询 + 线程上下文
// 读/写，不碰 COM、不开线程、不做文件 I/O。结果推迟到附着点的 `editor DPI facts`
// 行再落日志。
#[cfg(windows)]
#[no_mangle]
extern "system" fn DllMain(
    _module: *mut core::ffi::c_void,
    reason: u32,
    _reserved: *mut core::ffi::c_void,
) -> i32 {
    // DLL_PROCESS_ATTACH（winbase.h）；DLL_THREAD_ATTACH 不动 —— 提升只对
    // "加载线程将来会创建包裹窗口"这一次有意义；DETACH 也不动（卸载清理是
    // ExitDll 的职责，loader lock 里不做任何有副作用的事）。
    const DLL_PROCESS_ATTACH: u32 = 1;
    if reason == DLL_PROCESS_ATTACH {
        hifishifter_plugin::host_identity::promote_wrapper_thread_dpi();
    }
    1 // TRUE：永不阻断加载
}

// ── Unix 模块入口 ───────────────────────────────────────────────────────────
//
// 【为什么需要它们】VST3 的模块 ABI 不是"有 `GetPluginFactory` 就行"：Windows 的
// 约定是 `InitDll`/`ExitDll`，Linux 是 `ModuleEntry`/`ModuleExit`，macOS 是
// `bundleEntry`/`bundleExit`（且必须先有 `Contents/Info.plist`，宿主靠 CFBundle
// 找本模块的二进制）。宿主查不到入口符号时**不会报错**，表现是"这个插件 0 个类" ——
// 与符号拼错完全同一个症状，所以三组入口都显式导出，不赌宿主的宽容度。
//
// 【为什么 macOS 的参数是 `*mut c_void`】签名里的 `CFBundleRef` 就是一个指针；
// 本模块不需要读它（不查 CFBundle 元数据），声明成裸指针即可，省掉对
// CoreFoundation 的链接依赖。

/// Linux 模块入口。SDK 会在 `dlopen` 之后调用它（参数是模块句柄，本模块不用）。
#[cfg(unix)]
#[no_mangle]
pub extern "C" fn ModuleEntry(_module: *mut c_void) -> bool {
    // 【为什么守卫写在**入口**而不是只写在它调用的函数里】panic 越过 extern 边界
    // 是 abort；把它放在边界上，才不依赖"每个被调者都记得包一层"。
    hifishifter_plugin_core::guard_ffi(false, module_initialize)
}

/// Linux 模块卸载入口。
#[cfg(unix)]
#[no_mangle]
pub extern "C" fn ModuleExit() -> bool {
    hifishifter_plugin_core::guard_ffi(false, module_shutdown)
}

/// macOS bundle 入口（`CFBundleRef` 以裸指针传入）。
#[cfg(target_os = "macos")]
#[no_mangle]
pub extern "C" fn bundleEntry(_bundle: *mut c_void) -> bool {
    hifishifter_plugin_core::guard_ffi(false, module_initialize)
}

/// macOS bundle 卸载入口。
#[cfg(target_os = "macos")]
#[no_mangle]
pub extern "C" fn bundleExit() -> bool {
    hifishifter_plugin_core::guard_ffi(false, module_shutdown)
}
