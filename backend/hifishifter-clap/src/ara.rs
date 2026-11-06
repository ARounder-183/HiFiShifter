//! CLAP 的 ARA 发现层（阶段 2 前置）。
//!
//! 【能力边界】阶段 1 的 descriptor 不声明 `ara:supported`，entry 不暴露
//! `ClapAraEntry`，实例不返回 `clap_ara_plugin_extension`：三者都如实缺席。
//! 原因不是"忘了接"，而是共享 ARA 文档模型/渲染器仍在 VST3 外壳里，交一个没有
//! 控制器 backing 的 `ARAFactory` 会让宿主按 ARA 契约写穿空指针。
//!
//! 本模块把 `ARACLAP.h` 钉死的四个扩展 ID 收在一处，供阶段 2 接线时复用；
//! `tests/ara_discovery.rs` 用 `ara2-bridge-companion` 的**真实** CLAP 类型
//! 核对这些常量与工厂发现表的 ABI，确保接线那天不会从错误的 ID 开始。

use crate::abi::clap_plugin_t;
use std::ffi::{c_char, c_void};

/// entry 级 ARA 工厂扩展 ID（`ARACLAP.h` 的稳定 ID）。
pub const CLAP_EXT_ARA_FACTORY: &str = "org.ara-audio.ara.factory/2";
/// entry 级 draft 兼容 ID（上游注明 2026 年可能移除，只作可选兼容）。
pub const CLAP_EXT_ARA_FACTORY_COMPAT: &str = "org.ara-audio.ara.factory.draft/2";
/// 稳定实例扩展 ID。
pub const CLAP_EXT_ARA_PLUGIN_EXTENSION: &str = "org.ara-audio.ara.pluginextension/2";
/// draft 兼容 ID（上游注明 2026 年可能移除，只作可选兼容）。
pub const CLAP_EXT_ARA_PLUGIN_EXTENSION_COMPAT: &str = "org.ara-audio.ara.pluginextension.draft/2";

/// 返回静态 C 字符串，避免每次扩展查询分配。
fn static_id(id: &str) -> *const c_char {
    match id {
        CLAP_EXT_ARA_FACTORY => c"org.ara-audio.ara.factory/2".as_ptr().cast(),
        CLAP_EXT_ARA_FACTORY_COMPAT => c"org.ara-audio.ara.factory.draft/2".as_ptr().cast(),
        CLAP_EXT_ARA_PLUGIN_EXTENSION => c"org.ara-audio.ara.pluginextension/2".as_ptr().cast(),
        _ => c"org.ara-audio.ara.pluginextension.draft/2".as_ptr().cast(),
    }
}

/// ARA 扩展查询：阶段 1 一律返回 null。
///
/// 【为什么连 entry 级工厂也不给】descriptor 不声明 `ara:supported`，宿主不该来问；
/// 万一问了，返回 null 也比交一个没有文档控制器 backing 的 `ARAFactory` 安全 ——
/// 宿主会按 ARA 契约去调用工厂回调，那是一次空指针写穿。
pub fn get_extension(_plugin: *const clap_plugin_t, id: &str) -> *const c_void {
    if matches!(
        id,
        CLAP_EXT_ARA_FACTORY
            | CLAP_EXT_ARA_FACTORY_COMPAT
            | CLAP_EXT_ARA_PLUGIN_EXTENSION
            | CLAP_EXT_ARA_PLUGIN_EXTENSION_COMPAT
    ) {
        let _ = static_id(id);
    }
    std::ptr::null()
}

/// 生命周期观察点：共享 controller 绑定接线后逐一转发给 companion adapter。
pub fn attach(_plugin: *const clap_plugin_t) {}
/// 激活观察点。
pub fn observe_activation(_plugin: *const clap_plugin_t) {}
/// 停活观察点。
pub fn observe_deactivation(_plugin: *const clap_plugin_t) {}
/// 处理观察点。
pub fn observe_processing(_plugin: *const clap_plugin_t) {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_factory_id_is_format_suffixed() {
        assert_eq!(crate::entry::ARA_FACTORY_ID, "com.hifishifter.ara.clap");
    }
    #[test]
    fn an_unwired_instance_extension_returns_null_instead_of_a_fake_interface() {
        assert!(get_extension(std::ptr::null(), CLAP_EXT_ARA_PLUGIN_EXTENSION).is_null());
        assert!(get_extension(std::ptr::null(), "clap.state").is_null());
    }
}
