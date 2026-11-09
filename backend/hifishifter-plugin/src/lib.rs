//! HiFiShifter 的 VST3 / ARA 插件适配层。
//!
//! 【两种加载方式，同一份代码】宿主支持 ARA 时，本插件以 ARA2 形式挂在 DAW 里，
//! 时间线内容随 DAW 走；宿主**不**支持 ARA 时，它仍然是一个常规 VST3 插件，
//! 主要靠 [`capture`]（捕获）通路工作 —— 所以本 crate 与它的日志/诊断措辞都不该
//! 把"插件"等同于"ARA 插件"。
//!
//! 目标（设计文档 §5.1）：**复用本体已经存在的离线内核**，而不是另写一套渲染。
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
    // 【为什么允许 private_interfaces】本 crate 同时是 VST3 的 cdylib 与 CLAP 外壳的
    // rlib：`ExtensionOwner` / `DocumentSession` 必须 `pub`，兄弟外壳才能把 ARA 绑定
    // 转给**同一份**实现；而它们的公开方法签名里会带出本 crate 内部的协作类型
    // （快照发布器、区域键……）。那些类型不属于对外契约，也不该为了消一条 lint 被
    // 提升成 `pub` —— 那会把实现细节变成别人可以依赖的 API。真正的边界由
    // `hifishifter-clap` 只调用少数几个方法这一事实约束。
    private_interfaces,
    // `let mut x = X::default(); x.a = 1;` 与结构体更新语法等价；后者在字段多、
    // 默认值集中定义时更难读（要来回对照 Default 实现）。
    clippy::field_reassign_with_default,
)]

pub mod ara;
mod ara_entry;
mod audio_abi;

pub use audio_abi::BufferError;
pub mod audio_block;
/// 音频回调的跨格式读数（panic / 慢块 / 写满审计开关）。
pub mod audio_stats;
mod capture;
pub use hifishifter_plugin_core::component_state;
pub mod block_context;
/// 两种格式共用的**唯一**文件日志实现；格式外壳在自己的模块入口调 `init()`。
pub mod diagnostics;
pub mod editor;
#[allow(missing_docs)]
pub mod fade;
#[allow(missing_docs)]
pub mod host_identity;
pub use hifishifter_plugin_core::plugin_log_name;
pub mod render;
pub mod runtime;
mod settings_store;
mod startup;
mod state_channel;
mod state_stream;
pub use hifishifter_plugin_core::sync;
#[cfg(test)]
mod test_allocator;
#[cfg(test)]
mod test_host;
mod vst3;

use std::ffi::c_void;

/// VST3 类名。**必须与 ARA 工厂的 plugInName 一致** —— ARA 的 VST3 配对规则要求如此。
pub use hifishifter_plugin_core::CLASS_NAME;
pub(crate) mod host;
/// 版本号（展示用）。
///
/// 【为什么不是 `CARGO_PKG_VERSION`】产品版本的**单一来源**是
/// `src-tauri/tauri.conf.json`，由 `build.rs` 注入为 `HFS_PRODUCT_VERSION`。读本 crate
/// 的 `CARGO_PKG_VERSION` 会引入**第二份副本**：只改一处就会让插件与独立 App 的
/// 「关于」对话框报出不同版本（已经真实发生过一次）。`tests/version_parity.rs`
/// 再把 Cargo.toml / package.json 等其余副本钉在这一个值上。
pub(crate) const VERSION: &str = env!("HFS_PRODUCT_VERSION");
/// ARA 文档归档 ID。
pub use hifishifter_plugin_core::ARCHIVE_ID;
/// ARA 工厂的持久 ID。
pub use hifishifter_plugin_core::FACTORY_ID;

/// 「关于」对话框与诊断包里的**形态标识**。
///
/// 【为什么不能写死 "ARA plugin"】宿主不提供 ARA 时，本插件仍然可作为常规 VST3
/// 使用（主要靠捕获通路）。写死 ARA 会让用户拿着一张与实际运行形态不符的截图报障，
/// 而开发者第一眼要判断的正是"这份日志来自哪条通路"。
///
/// 【判据为什么是"有没有捕获存储且未绑定 ARA"】捕获存储只在"宿主没走 ARA"的实例上
/// 才有意义；已绑定 ARA 时几何由宿主给，捕获通路不适用。
pub(crate) fn plugin_form_label(
    owner: Option<&crate::render::extension::ExtensionOwner>,
) -> &'static str {
    // 【为什么按标签选】同一份共享代码同时跑在两种外壳里，把 "VST3" 写死会让 CLAP
    // 会话的「关于」与诊断包自称 VST3 —— 用户照着截图报障时，第一句话就是错的。
    let capture = owner.is_some_and(|owner| !owner.is_ara_bound());
    match (log_tag(), capture) {
        ("clap", true) => "CLAP plug-in (capture)",
        ("clap", false) => "CLAP / ARA plug-in",
        (_, true) => "VST3 plug-in (capture)",
        (_, false) => "VST3 / ARA plug-in",
    }
}

/// 本 cdylib 的**插件类型标签**（`vst3` / `clap` / 未来的 `au` …）。
///
/// 【为什么是格式身份的唯一登记点】`hifishifter-plugin` 是所有外壳共用的共享层。
/// 同一份事实要出现在三处：日志行前缀（`log_line`）、「关于」/诊断包的形态标签、
/// 以及**日志文件名与会话头**（`diagnostics`）。三处各自猜一次格式，就必然分叉 ——
/// 此前 CLAP 会话里出现过 `[vst3] ARA factory built`，正是因为它只改了日志前缀，
/// 没改日志系统对「我是谁」的判断。现在统一由这一处 `set_log_tag` 登记，其余全部派生。
///
/// 新增插件格式时只需在自己的 cdylib 入口调一次 `set_log_tag("<tag>")`；文件名、
/// 会话头、轮转、诊断包都会自动跟着走，无需改动共享层或命名模块。
static LOG_TAG: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();

/// 登记插件类型标签；重复登记忽略（第一个赢，与模块初始化的一次性语义一致）。
///
/// 【为什么必须由各 cdylib 在自己最早的入口登记】共享层的日志可能在任何组件创建之前
/// 就要写；晚一步登记，最早那几行就会归到默认文件名下。
pub fn set_log_tag(tag: &'static str) {
    let _ = LOG_TAG.set(tag);
}

/// 当前插件类型标签；未登记时按 VST3（历史默认，也是本 crate 最初的唯一外壳）。
///
/// 【为什么公开】`diagnostics` 用它拼日志文件名与会话头；外壳自己也可以读回它做自检。
pub fn log_tag() -> &'static str {
    LOG_TAG.get().copied().unwrap_or("vst3")
}

/// 会话头里的产品标识：日志放在一起时要能一眼分清是哪一种插件格式。
///
/// 由格式标签派生，因此新增格式不需要改这里。
pub fn log_product() -> String {
    format!("HiFiShifter {} plug-in", log_tag().to_ascii_uppercase())
}

/// 解析并记录宿主身份（进程证据链；两种格式共用同一条判据）。
///
/// 【为什么 CLAP 也要调】身份解析的证据（进程映像、祖先链、环境变量）与格式无关，
/// 而日志文件名 `plugin-<format>-<host>.log` 里的宿主段正是由它决定的 —— 不调就等于每次
/// CLAP 报障都只能拿到 `plugin-clap-unknown.log`。
pub fn resolve_host_identity() {
    crate::editor::connection::resolve_and_record_host_identity();
}

/// 用户级目录的一次性布局迁移（**每个宿主进程只跑一次**）。
///
/// 【为什么由插件来跑】用户可能只装插件、从不打开独立 App（REAPER 用户很常见）。
/// 那时迁移只能由插件触发，否则他们的设置与 PCM 暂存永远留在
/// `com.arounder.hifishifter\` 那一层下面。
///
/// 进程级初始化：产品版本 + 用户目录布局迁移（**两种格式共用同一份**）。
///
/// 【为什么单独成一个函数】CLAP 外壳没有 bundle 形态的资源目录，但它同样需要这一整段：
/// 用户可能只装插件、从不打开独立 App。此前 CLAP 的 `entry_init` 注册的是一个**空钩子**，
/// 于是 CLAP-only 用户的数据永远停在旧布局；更糟的是 `set_layout_migration_hook` 是
/// 进程级 `OnceLock` —— 谁先登记谁赢，CLAP 先加载时会把 VST3 真正的那一份挡掉。
/// 现在两个外壳登记的是同一个函数指针，谁先谁后都一样。
///
/// 【为什么必须早于 `diagnostics::init()`】日志初始化会创建
/// `<本地根>\logs` 并立刻写文件；迁移的判据是"新旧根同时存在就绝不合并"，
/// 日志目录先被建出来，旧日志就再也搬不过来了。
pub fn initialize_process_state() {
    // 【为什么在钩子之后才迁移】日志初始化会先建 logs 目录，而迁移判据是
    // 「新旧根同时存在就不合并」；钩子必须先登记，否则每个入口都要重复这段顺序。
    hifishifter_plugin_core::set_layout_migration_hook(|| {
        hifishifter_kernel::app_paths_migrate::ensure_layout_v2().log_summary("plugin");
    });
    hifishifter_plugin_core::set_product_version(crate::VERSION);
    hifishifter_plugin_core::ensure_layout_v2_once();
}

/// 登记本模块的推理模型与声码器资源（**两种格式共用同一份**）。
///
/// 【为什么公开】VST3 在模块入口调它；CLAP 的模块入口此前没有这一步 —— 模型要等到
/// 第一次打开编辑器、建会话时才登记，而"从没开过窗口就渲染/导出"的会话会先撞上
/// `models unavailable`。登记是幂等的，早调没有副作用。
pub fn initialize_resources() {
    crate::editor::resources::initialize_models();
}

/// VST3 bundle 形态的模块初始化：先登记资源根，再跑进程级初始化。
fn ensure_layout_v2_once() {
    // 【为什么资源布局在这里登记】VST3 的模块是 bundle（`Contents/Resources`），
    // 而资源根的唯一实现 `resource_root` 需要知道"本模块在哪、是哪种形态"。
    // 这里与 `set_product_version` 同属"进程级、只登记一次"的事实，放在同一处。
    // CLAP 在它自己的 `entry_init` 里登记 Sibling 形态（见 `hifishifter-clap`）。
    if let Ok(module) = crate::editor::resources::module_path() {
        let _ = hifishifter_plugin_core::resource_root::set_layout(
            hifishifter_plugin_core::resource_root::ResourceLayout::BundleContents,
        );
        let _ = hifishifter_plugin_core::resource_root::set_module_path(module);
    }
    initialize_process_state();
}

/// 写一行诊断信息。
///
/// 【为什么不用 `println!`】插件跑在宿主进程里，stdout 通常无处可去。`log` 会走宿主
/// 已装配的日志后端；宿主没装时它是静默丢弃，不会因为"日志写不出去"影响编曲。
pub(crate) fn log_line(message: &str) {
    log::info!("[{}] {message}", log_tag());
}

/// GUI 生命周期的诊断行：**必须是 WARN 级**。
///
/// 【为什么不能是 `log_line` 的 INFO】release 的默认级别是 `Warn`（理由见
/// `diagnostics.rs`：宿主回调的打点会把用户磁盘写满）。而 GUI 这几行恰恰是
/// "用户报一个空面板时唯一能拿到的东西" —— 一次真实的 Linux 报障里，日志里
/// **一条 `[gui]` 都没有**，于是"到底有没有 attach""走的是哪个后端""页面有没有起来"
/// 全部只能靠猜。这类行每次会话只有几条，用 WARN 换现场可诊断，代价可以忽略。
pub(crate) fn gui_log_line(message: &str) {
    log::warn!("[gui] {message}");
}

/// 模块入口的“我加载了”诊断行（**两种格式共用**）。
///
/// 【为什么是 WARN 级】默认日志级别是 Warn，而这一行回答的是“宿主只是扫描了我”
/// 还是“宿主真的在用我”：用户报“插件不见了”时，第一件要确认的就是模块有没有被
/// 加载、加载的是哪个版本与哪个目标平台。CLAP 外壳此前一条模块级记录都没有，
/// 于是同一句诊断在两个格式的日志里只能查到一半。
pub fn log_module_loaded() {
    gui_log_line(&format!(
        "module loaded: {} ({} {})",
        crate::VERSION,
        std::env::consts::OS,
        std::env::consts::ARCH
    ));
}

/// FFI 入口的 panic 遏制原语：Rust 1.71+ 起 panic 越过 `extern "system"` 边界 =
/// abort，宿主（DAW）进程随之崩溃。**每一个被宿主调用的入口**都必须把自己的
/// panic 拦成明确的失败值 —— 这不是音频回调的特例，而是全部 ABI 面的规则。
///
/// 【为什么 fallback 由调用方给】不同入口的"诚实失败值"不同（QI 该回
/// `K_NO_INTERFACE`、refcount 该回计数、视图创建该回 null），原语不该替调用方猜。
/// 拦下的 panic 统一记入日志（带 payload 文本），诊断有据可查。
pub(crate) fn guard_ffi<R>(fallback: R, f: impl FnOnce() -> R) -> R {
    hifishifter_plugin_core::guard_ffi(fallback, f)
}

/// VST3 模块入口：返回 `IPluginFactory*`。
///
/// 【为什么先 `runtime()`】工厂查询可能早于任何组件创建；先把 ARA 工厂建好，
/// 后面 VST3 侧查 `IPlugInEntryPoint` 时才有东西可给。
#[no_mangle]
pub extern "system" fn GetPluginFactory() -> *mut c_void {
    crate::guard_ffi(std::ptr::null_mut(), || {
        // 【为什么显式登记】格式标签是日志文件名与会话头的唯一来源；VST3 虽是默认
        // 值，显式登记能让"两个格式走同一套逻辑"这件事在入口处可见、且不会被将来
        // 的默认值改动悄悄破坏。
        crate::set_log_tag("vst3");
        crate::ensure_layout_v2_once();
        diagnostics::init();
        // 【为什么在工厂入口打一行】宿主扫描插件也会走到这里。有了这一行，日志就能
        // 区分"宿主只是扫描了我"与"宿主真的在用我" —— 前者之后再没有任何行是正常的。
        crate::log_module_loaded();
        // 【为什么把 IID 布局写进日志】它是"插件能不能和宿主说上话"的前置条件，却完全
        // 不可见：发错布局的失败形态是宿主回一个 null 指针，日志里只剩一行
        // `available=false`。把它钉在会话头附近，排查时第一眼就能排除一整类猜测。
        crate::log_line(&format!(
            "iid layout: {} ({})",
            crate::vst3::iid_layout(),
            std::env::consts::OS
        ));
        editor::resources::initialize_models();
        let _ = runtime::runtime();
        vst3::get_plugin_factory()
    })
}

/// 模块级初始化：三个平台的入口做的是同一件事，所以只有这一份实现。
///
/// 【为什么失败也要返回 `true`】让宿主把类列出来，原因记在日志里。返回 `false` 会让
/// 宿主直接跳过插件 —— 那就什么信息都拿不到了。探针期实测过这个取舍。
/// panic 除外：`guard_ffi` 已把 panic 原文记进日志，此时模块状态未知，宁可让宿主
/// 跳过也不要带着坏 runtime 插进 N 个轨道。
fn module_initialize() -> bool {
    crate::guard_ffi(false, || {
        crate::set_log_tag("vst3");
        crate::ensure_layout_v2_once();
        diagnostics::init();
        crate::log_module_loaded();
        editor::resources::initialize_models();
        let _ = runtime::runtime();
        true
    })
}

/// 模块级卸载：宿主在最后一次释放模块前的清理点。
fn module_shutdown() -> bool {
    // 先收尾 GUI 与慢车道（共享实现），再把还攒在缓冲里的日志落盘。
    let result = shutdown_module();
    diagnostics::flush_pending();
    result
}

/// 模块卸载的**唯一**实现：两个格式的模块入口都调它。
///
/// 【为什么卸载不能只拆窗口】宿主在最后一次释放模块前的清理点上有三类实体要收：
///   * 编辑器（窗口类、WebView2 回调、私有 GTK 线程）—— `editor::shutdown`；
///   * 慢车道任务（以分钟计）—— 它们持有本模块的代码页，卸载后下一次执行就是野跳转；
///   * 日志缓冲（宽限期内攒着的行）—— 由调用方在本函数返回后落盘。
///
/// 此前只有 VST3 的 `ExitDll` 做了前两件，CLAP 的 `entry_deinit` 只拆了编辑器：
/// 同一个宿主动作（卸载模块）在两种格式下留下不同的残留，而残留的形态恰好是
/// "卸载之后崩溃"，最难归因。
///
/// 【返回值为什么是"编辑器是否已全部关闭"】它是 VST3 `ExitDll` 的历史返回值，
/// CLAP 不需要它，但两个格式必须给出同一个结论（诊断行也据此判断能否正常卸载）。
pub fn shutdown_module() -> bool {
    // 先收尾 GUI（它会写诊断行），等慢车道清空。
    crate::guard_ffi(false, || {
        let editors = editor::shutdown();
        // 【为什么是限时】慢车道任务以分钟计；等不满就放弃。模块已被
        // `editor::pin_module`（慢车道任务开始 / 编辑器附着时）钉住，
        // 宿主随后的 FreeLibrary 不会真的卸载 —— 这里只是给"正常情况"
        // （没有后台工作）一个干净的收尾。
        let lane_idle = editor::slow_lane::wait_until_idle(std::time::Duration::from_secs(2));
        if !lane_idle {
            crate::gui_log_line(
                "module shutdown: slow lane still busy; the module stays resident until it finishes",
            );
        }
        editors
    })
}

/// VST3 模块初始化入口（Windows：REAPER 在 `LoadLibrary` 之后显式调用）。
#[no_mangle]
pub extern "system" fn InitDll() -> bool {
    crate::guard_ffi(false, module_initialize)
}

/// VST3 模块卸载入口（Windows）。
#[no_mangle]
pub extern "system" fn ExitDll() -> bool {
    crate::guard_ffi(false, module_shutdown)
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
    crate::guard_ffi(false, module_initialize)
}

/// Linux 模块卸载入口。
#[cfg(unix)]
#[no_mangle]
pub extern "C" fn ModuleExit() -> bool {
    crate::guard_ffi(false, module_shutdown)
}

/// macOS bundle 入口（`CFBundleRef` 以裸指针传入）。
#[cfg(target_os = "macos")]
#[no_mangle]
pub extern "C" fn bundleEntry(_bundle: *mut c_void) -> bool {
    crate::guard_ffi(false, module_initialize)
}

/// macOS bundle 卸载入口。
#[cfg(target_os = "macos")]
#[no_mangle]
pub extern "C" fn bundleExit() -> bool {
    crate::guard_ffi(false, module_shutdown)
}
