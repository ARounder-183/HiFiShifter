//! CLAP 模块入口：descriptor、工厂、实例生命周期与扩展分发。
//!
//! 【为什么 ARA 工厂在 init 时建立】ARACLAP.h 要求工厂指针与 `get_plugin_id` 在
//! `deinit()` 之前常驻；`ClapAraEntry` 用 Box 固定地址并泄漏到进程结束。

use crate::abi::cstr;
use crate::abi::*;
use crate::{audio, gui, state};
use ara2_bridge::companion::clap::sys::{
    CLAP_EXT_ARA_FACTORY, CLAP_EXT_ARA_FACTORY_COMPAT, CLAP_PLUGIN_FACTORY_ID,
};
use ara2_bridge::companion::clap::ClapAraEntry;
use ara2_bridge::companion::CompanionFactory;
use ara2_bridge::core::{ApiGeneration, PlaybackTransformationFlags};
use ara2_bridge::plugin::{Factory as AraFactory, FactoryBuilder, FactoryCapabilities};
use hifishifter_plugin_core::resource_root::{self, ResourceLayout};
use std::ffi::{c_char, c_void, CStr};
use std::sync::OnceLock;

/// 产品版本（每个格式 cdylib 各自注入，但来自同一个 tauri.conf.json）。
pub const VERSION: &str = env!("HFS_PRODUCT_VERSION");
/// ARA 工厂持久 ID：按 D8 默认带格式后缀，避免与 VST3 工厂在同一进程里配对歧义。
pub const ARA_FACTORY_ID: &str = "com.hifishifter.ara.clap";
/// ARA 文档归档 ID。
pub const ARA_ARCHIVE_ID: &str = "com.hifishifter.ara.archive.clap";

static ARA_ENTRY: OnceLock<Option<&'static ClapAraEntry<'static>>> = OnceLock::new();
static MODULE_PATH: OnceLock<String> = OnceLock::new();

static FEATURES: StaticPtrArray = StaticPtrArray([
    cstr(FEATURE_AUDIO_EFFECT),
    cstr(FEATURE_PITCH_CORRECTION),
    cstr(FEATURE_STEREO),
    cstr(FEATURE_ARA_SUPPORTED),
    std::ptr::null(),
]);
static DESCRIPTOR: clap_plugin_descriptor_t = clap_plugin_descriptor_t {
    clap_version: CLAP_VERSION,
    id: cstr(PLUGIN_ID),
    name: cstr(PLUGIN_NAME),
    vendor: cstr(VENDOR),
    url: cstr(URL),
    manual_url: cstr(URL),
    support_url: cstr(URL),
    version: { c"0.2.0-beta".as_ptr().cast() },
    description: cstr(DESCRIPTION),
    features: FEATURES.0.as_ptr(),
};

static FACTORY: clap_plugin_factory_t = clap_plugin_factory_t {
    get_plugin_count: Some(factory_count),
    get_plugin_descriptor: Some(factory_descriptor),
    create_plugin: Some(create_plugin),
};
static RENDER_EXT: crate::render_ext::RenderExt = crate::render_ext::RenderExt::new();
static PORTS_EXT: crate::ports::PortsExt = crate::ports::PortsExt::new();
static TIMER_EXT: crate::timer::TimerExt = crate::timer::TimerExt::new();

/// CLAP 唯一必须导出的动态库入口符号。
#[no_mangle]
pub static clap_entry: clap_plugin_entry_t = clap_plugin_entry_t {
    clap_version: CLAP_VERSION,
    init: Some(entry_init),
    deinit: Some(entry_deinit),
    get_factory: Some(entry_get_factory),
};

unsafe extern "C" fn entry_init(plugin_path: *const c_char) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        if !plugin_path.is_null() {
            if let Ok(path) = unsafe { CStr::from_ptr(plugin_path) }.to_str() {
                let _ = MODULE_PATH.set(path.to_owned());
                resource_root::set_module_path(path).ok();
            }
        }
        resource_root::set_layout(ResourceLayout::Sibling).ok();
        hifishifter_plugin_core::set_product_version(VERSION);
        hifishifter_plugin_core::set_layout_migration_hook(|| {});
        hifishifter_plugin_core::ensure_layout_v2_once();
        let _ = ara_entry();
        true
    })
}

unsafe extern "C" fn entry_deinit() {
    hifishifter_plugin_core::guard_ffi((), || {
        audio::set_render_mode(CLAP_RENDER_REALTIME);
        log::info!("[clap] module shutdown");
    })
}

unsafe extern "C" fn entry_get_factory(factory_id: *const c_char) -> *const c_void {
    hifishifter_plugin_core::guard_ffi(std::ptr::null(), || {
        if factory_id.is_null() {
            return std::ptr::null();
        }
        let Ok(id) = unsafe { CStr::from_ptr(factory_id) }.to_str() else {
            return std::ptr::null();
        };
        if id == CLAP_PLUGIN_FACTORY_ID {
            return (&raw const FACTORY).cast::<c_void>();
        }
        if id == CLAP_EXT_ARA_FACTORY || id == CLAP_EXT_ARA_FACTORY_COMPAT {
            if let Some(entry) = ara_entry() {
                return entry.as_raw().cast::<c_void>();
            }
        }
        std::ptr::null()
    })
}

fn ara_entry() -> Option<&'static ClapAraEntry<'static>> {
    ARA_ENTRY
        .get_or_init(|| {
            let factory: &'static AraFactory = Box::leak(Box::new(
                FactoryBuilder::new(ARA_FACTORY_ID, ARA_ARCHIVE_ID)
                    .display(
                        "HiFiShifter",
                        "HiFiShifter",
                        "https://github.com/ARounder-183/HiFiShifter",
                        VERSION,
                    )
                    .generations(ApiGeneration::V2Draft, ApiGeneration::V2Final)
                    .capabilities(
                        FactoryCapabilities::default().with_playback_transformations(
                            PlaybackTransformationFlags::TIMESTRETCH
                                | PlaybackTransformationFlags::REFLECT_TEMPO,
                        ),
                    )
                    .build()
                    .ok()?,
            ));
            // SAFETY: factory 泄漏到进程结束，其 ARAFactory 与 companion 关联同寿。
            let association = unsafe {
                CompanionFactory::from_raw(crate::abi::PLUGIN_ID_STR, &*factory.as_raw()).ok()?
            };
            let entry = ClapAraEntry::new([association]).ok()?;
            Some(Box::leak(Box::new(entry)))
        })
        .as_ref()
        .copied()
}

unsafe extern "C" fn factory_count(_factory: *const clap_plugin_factory_t) -> u32 {
    hifishifter_plugin_core::guard_ffi(0, || 1)
}
unsafe extern "C" fn factory_descriptor(
    _f: *const clap_plugin_factory_t,
    index: u32,
) -> *const clap_plugin_descriptor_t {
    hifishifter_plugin_core::guard_ffi(std::ptr::null(), || {
        if index == 0 {
            &raw const DESCRIPTOR
        } else {
            std::ptr::null()
        }
    })
}

/// 一个 CLAP 实例的全部可变状态。
pub struct Instance {
    pub state: state::InstanceState,
    pub host: *const clap_host_t,
    pub plugin: *const clap_plugin_t,
    /// 阶段 2 的实例级 ARA 适配器；未接线前恒为 None。
    pub ara: Option<()>,
}

unsafe extern "C" fn create_plugin(
    _factory: *const clap_plugin_factory_t,
    host: *const clap_host_t,
    plugin_id: *const c_char,
) -> *const clap_plugin_t {
    hifishifter_plugin_core::guard_ffi(std::ptr::null(), || {
        if plugin_id.is_null() || host.is_null() {
            return std::ptr::null();
        }
        if unsafe { CStr::from_ptr(plugin_id) }.to_bytes() != b"com.hifishifter.clap" {
            return std::ptr::null();
        }
        if !unsafe { clap_version_is_compatible((*host).clap_version) } {
            return std::ptr::null();
        }
        let instance = Box::new(Instance {
            state: state::InstanceState::default(),
            host,
            plugin: std::ptr::null(),
            ara: None,
        });
        let plugin = Box::new(clap_plugin_t {
            desc: &raw const DESCRIPTOR,
            plugin_data: Box::into_raw(instance).cast(),
            init: Some(plugin_init),
            destroy: Some(plugin_destroy),
            activate: Some(plugin_activate),
            deactivate: Some(plugin_deactivate),
            start_processing: Some(start_processing),
            stop_processing: Some(stop_processing),
            reset: Some(reset_processing),
            process: Some(plugin_process),
            get_extension: Some(plugin_get_extension),
            on_main_thread: Some(on_main_thread),
        });
        let ptr: *const clap_plugin_t = Box::into_raw(plugin);
        unsafe { crate::instance_mut(ptr) }.plugin = ptr;
        ptr
    })
}

fn clap_version_is_compatible(v: clap_version_t) -> bool {
    v.major >= 1
}

/// 从 `clap_plugin_t` 取实例；空指针或已销毁返回 None。
/// # Safety
/// `plugin` 必须指向本模块创建的、尚未 destroy 的 `clap_plugin_t`。
pub unsafe fn instance(plugin: *const clap_plugin_t) -> Option<&'static Instance> {
    if plugin.is_null() {
        return None;
    }
    let data = unsafe { (*plugin).plugin_data };
    if data.is_null() {
        return None;
    }
    Some(unsafe { &*data.cast::<Instance>() })
}
/// 可变版本；只在本模块生命周期回调里使用。
/// # Safety
/// 仅允许在本 crate 的生命周期回调里调用；不得同时持有另一个可变引用。
pub unsafe fn instance_mut(plugin: *const clap_plugin_t) -> &'static mut Instance {
    unsafe { &mut *(*plugin).plugin_data.cast::<Instance>() }
}

unsafe extern "C" fn plugin_init(plugin: *const clap_plugin_t) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        let Some(host) =
            (unsafe { plugin.as_ref() }).and_then(|p| unsafe { instance(p) }.map(|i| i.host))
        else {
            return false;
        };
        let _ = host;
        crate::ara::attach(plugin);
        true
    })
}
unsafe extern "C" fn plugin_destroy(plugin: *const clap_plugin_t) {
    hifishifter_plugin_core::guard_ffi((), || {
        if plugin.is_null() {
            return;
        }
        let data = unsafe { (*plugin).plugin_data };
        if !data.is_null() {
            unsafe { drop(Box::from_raw(data.cast::<Instance>())) };
        }
        unsafe { drop(Box::from_raw(plugin.cast_mut())) };
    })
}
unsafe extern "C" fn plugin_activate(
    plugin: *const clap_plugin_t,
    sample_rate: f64,
    min: u32,
    max: u32,
) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        if !sample_rate.is_finite() || sample_rate <= 0.0 || min == 0 || max < min {
            return false;
        }
        audio::set_sample_rate(sample_rate);
        crate::ara::observe_activation(plugin);
        true
    })
}
unsafe extern "C" fn plugin_deactivate(plugin: *const clap_plugin_t) {
    hifishifter_plugin_core::guard_ffi((), || crate::ara::observe_deactivation(plugin))
}
unsafe extern "C" fn start_processing(plugin: *const clap_plugin_t) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        crate::ara::observe_processing(plugin);
        true
    })
}
unsafe extern "C" fn stop_processing(_plugin: *const clap_plugin_t) {
    hifishifter_plugin_core::guard_ffi((), || ())
}
unsafe extern "C" fn reset_processing(_plugin: *const clap_plugin_t) {
    hifishifter_plugin_core::guard_ffi((), || ())
}

unsafe extern "C" fn plugin_process(
    plugin: *const clap_plugin_t,
    process: *const clap_process_t,
) -> i32 {
    hifishifter_plugin_core::guard_ffi(CLAP_PROCESS_ERROR, || {
        if process.is_null() {
            return CLAP_PROCESS_ERROR;
        }
        let Some(process) = (unsafe { process.as_ref() }) else {
            return CLAP_PROCESS_ERROR;
        };
        let frame = audio::to_transport_frame(process);
        log::trace!(
            "[clap] block frames={} trusted={} playing={:?} offline={}",
            process.frames_count,
            frame.position_trusted,
            frame.playing,
            frame.offline
        );
        crate::ara::observe_processing(plugin);
        if unsafe { audio::process_block(process) } {
            CLAP_PROCESS_CONTINUE
        } else {
            CLAP_PROCESS_ERROR
        }
    })
}

unsafe extern "C" fn plugin_get_extension(
    plugin: *const clap_plugin_t,
    id: *const c_char,
) -> *const c_void {
    hifishifter_plugin_core::guard_ffi(std::ptr::null(), || {
        if id.is_null() {
            return std::ptr::null();
        }
        let Ok(id) = unsafe { CStr::from_ptr(id) }.to_str() else {
            return std::ptr::null();
        };
        match id {
            "clap.state" => return (&raw const state::STATE_EXT).cast(),
            "clap.gui" => return (&raw const gui::GUI_EXT).cast(),
            "clap.render" => return (&raw const RENDER_EXT).cast(),
            "clap.audio-ports" => return (&raw const PORTS_EXT).cast(),
            "clap.timer-support" => return (&raw const TIMER_EXT).cast(),
            _ => {}
        }
        crate::ara::get_extension(plugin, id)
    })
}

unsafe extern "C" fn on_main_thread(_plugin: *const clap_plugin_t) {
    hifishifter_plugin_core::guard_ffi((), || ())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_descriptor_advertises_optional_ara_but_never_requires_it() {
        let features = unsafe { std::slice::from_raw_parts(DESCRIPTOR.features, 4) };
        let strings: Vec<&str> = features
            .iter()
            .map(|p| unsafe { CStr::from_ptr(*p) }.to_str().unwrap())
            .collect();
        assert!(strings.contains(&"ara:supported"));
        assert!(!strings.contains(&"ara:required"));
    }
    #[test]
    fn the_entry_declares_clap_1_1_9() {
        assert_eq!(clap_entry.clap_version, CLAP_VERSION)
    }
}
