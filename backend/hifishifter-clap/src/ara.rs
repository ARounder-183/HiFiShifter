//! CLAP 的 ARA 接入：entry 级工厂表 + 实例级扩展。
//!
//! 【实现唯一性】本模块只做 CLAP 的 **ABI 适配**：把 ARACLAP.h 的两个回调转成对
//! [hifishifter_plugin::ara::binder::AraBinder] 的调用。角色校验、文档查找、代数协商
//! 检查、companion 复位与幂等复用都在那份共享实现里 —— VST3 外壳调的是同一个方法，
//! 两个格式因此不会在"绑到哪份文档、给不给接口"这类判断上分叉。
//!
//! 【为什么不直接用 companion 的 CLAP 适配器】它的 extension_builder 入参只有两个
//! 角色位，**不把控制器引用交出来**；而绑定必须按控制器身份找到本产品的文档会话
//! （同一进程里可以同时有多份文档）。少这一个实参，就只能靠"猜最近一份文档"，那是把
//! 正确性交给时序。这里改用 companion 的**同一套** ClapAraPluginExtension 结构体与
//! ClapAraEntry 工厂表，只把回调接到共享 binder 上。

use crate::abi::clap_plugin_t;
use ara2_bridge::companion::clap::sys::{ClapAraPluginExtension, ClapPlugin};
/// ARACLAP.h 钉死的四个扩展 ID（stable + draft 兼容）。对外只此一份，测试直接比对。
pub use ara2_bridge::companion::clap::sys::{
    CLAP_EXT_ARA_FACTORY, CLAP_EXT_ARA_FACTORY_COMPAT, CLAP_EXT_ARA_PLUGIN_EXTENSION,
    CLAP_EXT_ARA_PLUGIN_EXTENSION_COMPAT,
};
use ara2_bridge::companion::clap::ClapAraEntry;
use ara2_bridge::companion::CompanionRoles;
use ara2_bridge::sys::{
    ARADocumentControllerRef, ARAFactory, ARAPlugInExtensionInstance, ARAPlugInInstanceRoleFlags,
};
use hifishifter_plugin::ara::binder::AraBinder;
use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

/// entry 级 ARA 工厂表；进程内只建一次（工厂本身泄漏到进程结束）。
///
/// 【为什么失败也缓存】OnceLock 只跑一次。宿主可能反复查询扩展；每次重试构建工厂
/// 既没意义也会刷日志，而失败原因已经写在日志里。
fn ara_entry() -> Option<&'static ClapAraEntry<'static>> {
    static ENTRY: OnceLock<Option<ClapAraEntry<'static>>> = OnceLock::new();
    ENTRY
        .get_or_init(|| {
            let Some(runtime) = hifishifter_plugin::runtime::runtime() else {
                log::warn!("[clap][ara] runtime unavailable; the entry publishes no ARA factory");
                return None;
            };
            let association = match runtime.companion_for(crate::abi::PLUGIN_ID_STR) {
                Ok(association) => association,
                Err(error) => {
                    log::error!("[clap][ara] companion association failed: {error}");
                    return None;
                }
            };
            match ClapAraEntry::new([association]) {
                Ok(entry) => {
                    log::info!(
                        "[clap][ara] entry publishes {} ARA association(s) for {}",
                        entry.factory_count(),
                        crate::abi::PLUGIN_ID_STR
                    );
                    Some(entry)
                }
                Err(error) => {
                    log::error!("[clap][ara] ARA factory table rejected: {error}");
                    None
                }
            }
        })
        .as_ref()
}

/// entry 级 get_factory 的 ARA 分支：只认工厂扩展 ID。
///
/// 返回 null 表示"本模块没有 ARA 工厂"（运行时建不起来），宿主据此把插件当普通 CLAP
/// 效果器；**绝不能**返回一张没有文档控制器的表 —— 宿主会按 ARA 契约去调回调。
pub fn entry_extension(id: &str) -> *const c_void {
    if !matches!(id, CLAP_EXT_ARA_FACTORY | CLAP_EXT_ARA_FACTORY_COMPAT) {
        return std::ptr::null();
    }
    ara_entry().map_or(std::ptr::null(), |entry| entry.extension(id))
}

/// 实例级扩展表。回调本身与实例无关，状态一律经 plugin->plugin_data 取。
static PLUGIN_EXT: ClapAraPluginExtension = ClapAraPluginExtension {
    get_factory: Some(ara_get_factory),
    bind_to_document_controller: Some(ara_bind),
};

/// 取本实例的绑定器；未接 ARA（运行时不可用）时为 None。
fn binder_for(plugin: *const clap_plugin_t) -> Option<&'static Arc<AraBinder>> {
    // SAFETY: 调用方是宿主按 CLAP 契约传入的活插件记录；instance 自己会判空。
    unsafe { crate::instance(plugin) }.and_then(|instance| instance.ara.as_ref())
}

unsafe extern "C" fn ara_get_factory(plugin: *const ClapPlugin) -> *const ARAFactory {
    hifishifter_plugin_core::guard_ffi(std::ptr::null(), || {
        let Some(binder) = binder_for(plugin.cast()) else {
            return std::ptr::null();
        };
        binder.factory_raw().cast()
    })
}

unsafe extern "C" fn ara_bind(
    plugin: *const ClapPlugin,
    controller: ARADocumentControllerRef,
    known: ARAPlugInInstanceRoleFlags,
    assigned: ARAPlugInInstanceRoleFlags,
) -> *const ARAPlugInExtensionInstance {
    hifishifter_plugin_core::guard_ffi(std::ptr::null(), || {
        let Some(binder) = binder_for(plugin.cast()) else {
            log::error!("[clap][ara] bind request on an instance without an ARA binder");
            return std::ptr::null();
        };
        binder
            .bind(controller.cast(), known, assigned)
            .cast()
    })
}

/// 实例级 ARA 扩展查询。
pub fn get_extension(plugin: *const clap_plugin_t, id: &str) -> *const c_void {
    if !matches!(
        id,
        CLAP_EXT_ARA_PLUGIN_EXTENSION | CLAP_EXT_ARA_PLUGIN_EXTENSION_COMPAT
    ) {
        return std::ptr::null();
    }
    if binder_for(plugin).is_none() {
        return std::ptr::null();
    }
    std::ptr::from_ref(&PLUGIN_EXT).cast()
}

/// 在 plugin->init 时接上本实例的 ARA 绑定器。
///
/// 【为什么在 init 里接】CLAP 的 ARA 契约要求宿主在 activate 之前完成
/// bindToDocumentController，而 init 是唯一确定早于它的实例回调。这里只建绑定器
/// （不建 companion 绑定对象：那个必须惰性建在真正 bind 的线程上，见 CompanionSlot）。
/// # Safety
/// `plugin` 必须是宿主按 CLAP 契约传入的、尚未 destroy 的插件记录。
pub unsafe fn attach(plugin: *const clap_plugin_t) {
    if plugin.is_null() {
        return;
    }
    // SAFETY: 本函数只在 plugin_init 里调用，实例仍归宿主所有。
    let instance = unsafe { crate::instance_mut(plugin) };
    if instance.ara.is_some() {
        return;
    }
    let Some(runtime) = hifishifter_plugin::runtime::runtime() else {
        log::warn!("[clap][ara] runtime unavailable; this instance stays capture-only");
        return;
    };
    let association = match runtime.companion_for(crate::abi::PLUGIN_ID_STR) {
        Ok(association) => association,
        Err(error) => {
            log::error!("[clap][ara] companion association failed: {error}");
            return;
        }
    };
    // 复用实例的扩展所有权：文档绑定、renderer 分配与编辑器会话都挂在同一份上。
    let owner = Arc::clone(&instance.owner);
    match AraBinder::new(
        association,
        CompanionRoles::all(),
        crate::abi::PLUGIN_ID_STR,
        owner,
    ) {
        Ok(binder) => {
            instance.ara = Some(Arc::new(binder));
            log::info!("[clap][ara] instance bound to the shared ARA binder");
        }
        Err(error) => log::error!("[clap][ara] ARA binder rejected: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 共享实现必须能被 CLAP 外壳**命名**：ARA 绑定走的是 VST3 那一份 AraBinder，
    /// 不是复制品。这条用例钉住公开面（类型可达、可 Arc 持有）。
    #[test]
    fn the_shared_ara_binder_is_reachable_from_the_clap_shell() {
        let binder: Option<Arc<AraBinder>> = None;
        assert!(binder.is_none());
    }
    #[test]
    fn the_factory_id_is_format_suffixed() {
        assert_eq!(crate::entry::ARA_FACTORY_ID, "com.hifishifter.ara.clap");
    }
    /// 没接 ARA 的实例必须如实返回 null，而不是一张假接口表。
    #[test]
    fn an_unwired_instance_extension_returns_null_instead_of_a_fake_interface() {
        assert!(get_extension(std::ptr::null(), CLAP_EXT_ARA_PLUGIN_EXTENSION).is_null());
        assert!(get_extension(std::ptr::null(), "clap.state").is_null());
    }
    /// entry 级只认 ARA 工厂 ID：别的 ID 不能拿到工厂表。
    #[test]
    fn the_entry_only_answers_the_ara_factory_ids() {
        assert!(entry_extension("clap.plugin-factory").is_null());
        assert!(entry_extension(CLAP_EXT_ARA_PLUGIN_EXTENSION).is_null());
    }
}
