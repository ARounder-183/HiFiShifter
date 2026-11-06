//! CLAP 的 ARA 发现层（阶段 2）。
//!
//! 【能力边界】entry 级 `ClapAraEntry` 暴露进程级工厂，宿主可以读取 ARA 元数据；
//! 实例级 `clap_ara_plugin_extension` 需要把每个 CLAP 实例绑到产品 ARA 文档控制器，
//! 那要求共享的文档模型/渲染器在 core 中可用。此处如实把未接线的实例扩展返回 null，
//! 而不是交一个会让宿主写穿的假接口。黑盒探针把这条断言出来。

use crate::abi::clap_plugin_t;
use std::ffi::{c_char, c_void};

/// 稳定实例扩展 ID。
pub const CLAP_EXT_ARA_PLUGIN_EXTENSION: &str = "org.ara-audio.ara.pluginextension/2";
/// draft 兼容 ID（上游注明 2026 年可能移除，只作可选兼容）。
pub const CLAP_EXT_ARA_PLUGIN_EXTENSION_COMPAT: &str = "org.ara-audio.ara.pluginextension.draft/2";

/// 返回静态 C 字符串，避免每次扩展查询分配。
fn static_id(id: &str) -> *const c_char {
    match id {
        CLAP_EXT_ARA_PLUGIN_EXTENSION => c"org.ara-audio.ara.pluginextension/2".as_ptr().cast(),
        _ => c"org.ara-audio.ara.pluginextension.draft/2".as_ptr().cast(),
    }
}

/// 实例扩展目前未接线；返回 null 是诚实的失败值。
pub fn get_extension(_plugin: *const clap_plugin_t, id: &str) -> *const c_void {
    if id == CLAP_EXT_ARA_PLUGIN_EXTENSION || id == CLAP_EXT_ARA_PLUGIN_EXTENSION_COMPAT {
        let _ = static_id(id);
        return std::ptr::null();
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
