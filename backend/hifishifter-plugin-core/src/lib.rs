//! HiFiShifter 插件的格式无关契约。
//!
//! 【边界】本 crate 不认识任何插件格式的 ABI 类型：不出现 VST3 的 `ProcessContext` /
//! `AudioBusBuffers`、CLAP 的 `clap_process_t`、AU 的 `AudioBufferList`，也不链接
//! 任何 companion SDK。格式外壳负责把宿主给的时间与缓冲**在边界转换一次**，再调用这里。
//!
//! 【为什么 trait 都是窄接口】宿主能力是可选给的：缺席必须是一等状态，不能拿 0/false
//! 冒充。每个 trait 只回答一个格式差异问题，避免 core 反过来依赖某个具体宿主协议。

#![deny(missing_docs)]
#![allow(
    // 历史模块的字段逐个赋值读起来更直接；禁止重排既有语义。
    clippy::field_reassign_with_default,
    // 测试模块紧邻被测代码，重排只是纯移动 diff。
    clippy::items_after_test_module,
    // 数值循环里的索引是采样位置/通道步长的一部分。
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::vec_box,
    clippy::while_immutable_condition,
)]

#[allow(missing_docs)]
pub mod component_state;
pub mod format;
pub mod local_data;
#[allow(missing_docs)]
pub mod plugin_log_name;
pub mod resource_root;
#[allow(missing_docs)]
pub mod sync;

use std::path::Path;
use std::sync::OnceLock;

/// 产品显示名。**必须**与各格式 descriptor 及 ARA 工厂的 plugInName 一致。
pub const CLASS_NAME: &str = "HiFiShifter";
/// ARA 工厂的持久 ID（发布后不可改）。
pub const FACTORY_ID: &str = "com.hifishifter.ara";
/// ARA 文档归档 ID（发布后不可改）。
pub const ARCHIVE_ID: &str = "com.hifishifter.ara.archive";

/// 进程级产品版本，由各 cdylib 在模块入口登记一次。
static PRODUCT_VERSION: OnceLock<&'static str> = OnceLock::new();

/// 登记产品版本。**必须由每个 cdylib 的模块入口调用**，取注入的 `HFS_PRODUCT_VERSION`。
pub fn set_product_version(version: &'static str) {
    let _ = PRODUCT_VERSION.set(version);
}

/// 产品版本；未登记时明确返回空串，绝不猜一个 crate 版本。
pub fn product_version() -> &'static str {
    PRODUCT_VERSION.get().copied().unwrap_or("")
}

/// FFI 入口的 panic 遏制原语。
///
/// 【为什么放在 core】Rust 1.71+ 起 panic 越过 `extern` 边界即 abort；每个格式外壳的
/// 每个宿主回调都要拦成明确的失败值。共用一个实现能让静态门禁逐 crate 核对同一规则。
pub fn guard_ffi<R>(fallback: R, f: impl FnOnce() -> R) -> R {
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

/// 存储布局迁移钩子。
///
/// 【为什么是钩子而不是 core 直接依赖内核】布局迁移属于应用核心，而 core 只是契约层，
/// 不该为了一个一次性调用把整个推理内核拉进依赖图。各格式 cdylib 注入同一个函数，
/// 宿主进程里仍只跑一次。
static LAYOUT_MIGRATION: OnceLock<fn()> = OnceLock::new();

/// 登记用户级目录的一次性布局迁移；重复登记同一函数无副作用。
pub fn set_layout_migration_hook(hook: fn()) {
    let _ = LAYOUT_MIGRATION.set(hook);
}

/// 用户级目录的一次性布局迁移（**每个宿主进程只跑一次**）。
pub fn ensure_layout_v2_once() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if let Some(hook) = LAYOUT_MIGRATION.get() {
            hook();
        } else {
            log::warn!("[plugin] storage layout migration hook is not registered");
        }
    });
}

/// 判断资源目录里是否带齐前端入口。用于打包自检，与平台无关。
pub fn has_frontend_entry(dir: &Path) -> bool {
    dir.join("plugin.html").is_file()
}
