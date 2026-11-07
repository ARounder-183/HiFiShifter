//! CLAP 模块入口：descriptor、工厂、实例生命周期与扩展分发。
//!
//! 【为什么阶段 1 不暴露 ARA】`ara:supported` 意味着宿主可以按 ARACLAP.h 去要
//! ARAFactory 与实例扩展。当前共享文档模型仍在 VST3 外壳里，交一个不能创建控制器的
//! 工厂会让宿主写穿空指针；所以 descriptor 不声明 ARA，entry 也不给 ARA factory。

use crate::abi::cstr;
use crate::abi::*;
use crate::{audio, state};
/// 标准插件工厂 ID。
const CLAP_PLUGIN_FACTORY_ID: &str = "clap.plugin-factory";
use hifishifter_plugin_core::resource_root::{self, ResourceLayout};
use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// 产品版本（每个格式 cdylib 各自注入，但来自同一个 tauri.conf.json）。
pub const VERSION: &str = env!("HFS_PRODUCT_VERSION");
/// 阶段 2 定稿的 ARA 工厂 ID（带格式后缀，避免与 VST3 同进程配对歧义）。
pub const ARA_FACTORY_ID: &str = "com.hifishifter.ara.clap";
/// 阶段 2 定稿的 ARA 文档归档 ID。
pub const ARA_ARCHIVE_ID: &str = "com.hifishifter.ara.archive.clap";

static MODULE_PATH: OnceLock<String> = OnceLock::new();

static FEATURES: StaticPtrArray = StaticPtrArray([
    cstr(FEATURE_AUDIO_EFFECT),
    cstr(FEATURE_PITCH_CORRECTION),
    cstr(FEATURE_STEREO),
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
    version: { cstr(PRODUCT_VERSION_CSTR) },
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
        resource_root::set_layout(ResourceLayout::for_clap()).ok();
        hifishifter_plugin_core::set_product_version(VERSION);
        hifishifter_plugin_core::set_layout_migration_hook(|| {});
        hifishifter_plugin_core::ensure_layout_v2_once();
        true
    })
}

unsafe extern "C" fn entry_deinit() {
    hifishifter_plugin_core::guard_ffi((), || {
        crate::log_host::clear();
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
        std::ptr::null()
    })
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
    /// 本实例的音频时间状态（采样率 / render 模式 / transport 观察值）。
    /// 绝不放到进程级 static：同一宿主进程里多实例会互相覆盖。
    pub audio: audio::AudioState,
    pub host: *const clap_host_t,
    pub plugin: *const clap_plugin_t,
    /// 阶段 2 的实例级 ARA 适配器；未接线前恒为 None。
    pub ara: Option<()>,
    /// `plugin->init` 是否已成功。CLAP 规定每个实例只调一次。
    pub initialized: AtomicBool,
    /// `plugin->activate` 是否已成功；`process` 必须在此之后。
    pub active: AtomicBool,
    /// `start_processing` / `stop_processing` 之间。
    pub processing: AtomicBool,
    /// `process` 是否正在执行（防同一实例并发进入音频回调）。
    pub in_process: AtomicBool,
    /// 捕获宿主：与 VST3 共用同一份运行时（判定、投影、sidecar）。
    pub capture: crate::capture::CaptureHost,
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
            audio: audio::AudioState::default(),
            host,
            plugin: std::ptr::null(),
            ara: None,
            initialized: AtomicBool::new(false),
            active: AtomicBool::new(false),
            processing: AtomicBool::new(false),
            in_process: AtomicBool::new(false),
            capture: crate::capture::CaptureHost::default(),
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
        let Some(instance) = (unsafe { crate::instance(plugin) }) else {
            return false;
        };
        // CLAP: init() is called exactly once per instance, on the main thread.
        // A second call is a host bug; returning false is safer than re-running
        // side effects (logger registration, ARA attach) twice.
        if instance.initialized.swap(true, Ordering::AcqRel) {
            log::warn!("[clap] duplicate plugin.init rejected");
            return false;
        }
        if instance.host.is_null() {
            return false;
        }
        crate::log_host::install(instance.host);
        crate::ara::attach(plugin);
        log::info!("[clap] instance initialized");
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
            if let Some(instance) = unsafe { crate::instance(plugin) } {
                // destroy() requires !active. Refuse to free a live audio instance.
                if instance.active.load(Ordering::Acquire) {
                    log::error!("[clap] destroy called while active; leaking instead of crashing");
                    return;
                }
            }
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
        let Some(instance) = (unsafe { crate::instance(plugin) }) else {
            return false;
        };
        if !instance.initialized.load(Ordering::Acquire) {
            log::error!("[clap] activate before init rejected");
            return false;
        }
        if instance.active.swap(true, Ordering::AcqRel) {
            log::error!("[clap] duplicate activate rejected");
            return false;
        }
        if !sample_rate.is_finite() || sample_rate <= 0.0 || min == 0 || max < min {
            instance.active.store(false, Ordering::Release);
            return false;
        }
        instance.audio.set_sample_rate(sample_rate);
        // 捕获拒绝在没有工程采样率的实例上武装：它是"工程帧 ↔ 秒"换算的唯一依据。
        instance.capture.note_activate(sample_rate);
        crate::ara::observe_activation(plugin);
        log::info!("[clap] activated at {sample_rate} Hz, block {min}..={max}");
        true
    })
}
unsafe extern "C" fn plugin_deactivate(plugin: *const clap_plugin_t) {
    hifishifter_plugin_core::guard_ffi((), || {
        let Some(instance) = (unsafe { crate::instance(plugin) }) else {
            return;
        };
        if instance.processing.load(Ordering::Acquire) {
            log::error!("[clap] deactivate while processing rejected");
            return;
        }
        if !instance.active.swap(false, Ordering::AcqRel) {
            log::warn!("[clap] deactivate on an inactive instance");
            return;
        }
        instance.audio.set_render_mode(CLAP_RENDER_REALTIME);
        crate::ara::observe_deactivation(plugin);
    })
}
unsafe extern "C" fn start_processing(plugin: *const clap_plugin_t) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        let Some(instance) = (unsafe { crate::instance(plugin) }) else {
            return false;
        };
        if !instance.active.load(Ordering::Acquire)
            || instance.processing.swap(true, Ordering::AcqRel)
        {
            log::error!("[clap] start_processing in an invalid state rejected");
            return false;
        }
        crate::ara::observe_processing(plugin);
        true
    })
}
unsafe extern "C" fn stop_processing(plugin: *const clap_plugin_t) {
    hifishifter_plugin_core::guard_ffi((), || {
        if let Some(instance) = unsafe { crate::instance(plugin) } {
            instance.processing.store(false, Ordering::Release);
        }
    })
}
unsafe extern "C" fn reset_processing(plugin: *const clap_plugin_t) {
    hifishifter_plugin_core::guard_ffi((), || {
        // reset() clears processing state but keeps the activation. Nothing is
        // buffered across blocks in this shell, so there is no DSP state to drop.
        if let Some(instance) = unsafe { crate::instance(plugin) } {
            let _ = instance;
        }
    })
}

unsafe extern "C" fn plugin_process(
    plugin: *const clap_plugin_t,
    process: *const clap_process_t,
) -> i32 {
    hifishifter_plugin_core::guard_ffi(CLAP_PROCESS_ERROR, || {
        if process.is_null() {
            return CLAP_PROCESS_ERROR;
        }
        let Some(instance) = (unsafe { crate::instance(plugin) }) else {
            return CLAP_PROCESS_ERROR;
        };
        // [audio-thread & active_state & processing_state]: refuse any other
        // combination instead of silently reading a half-initialised state.
        if !instance.active.load(Ordering::Acquire) || !instance.processing.load(Ordering::Acquire)
        {
            log::error!("[clap] process outside the active/processing state rejected");
            return CLAP_PROCESS_ERROR;
        }
        if instance.in_process.swap(true, Ordering::AcqRel) {
            log::error!("[clap] concurrent process on one instance rejected");
            return CLAP_PROCESS_ERROR;
        }
        let result = (|| {
            let Some(process) = (unsafe { process.as_ref() }) else {
                return CLAP_PROCESS_ERROR;
            };
            // 【为什么只在这里读一次 host transport】`to_transport_frame` 会推进
            // 块计数与读数序号；此前它在这里和 `capture.on_process` 里各读一次，
            // 于是每个音频块被记成两块（诊断读数翻倍，前端的乱序判据也被加快）。
            // 现在唯一的读点在 `capture.on_process`，它自己负责写日志。
            crate::ara::observe_processing(plugin);
            if unsafe { audio::process_block(process) } {
                // 【顺序】先让宿主缓冲就位（in-place 时输入与输出是同一块），再喂捕获：
                // 捕获只读输入平面，写完自己的段；它不改输出，因此直通不受影响。
                unsafe { instance.capture.on_process(process, &instance.audio) };
                CLAP_PROCESS_CONTINUE
            } else {
                CLAP_PROCESS_ERROR
            }
        })();
        instance.in_process.store(false, Ordering::Release);
        result
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
            "clap.render" => return (&raw const RENDER_EXT).cast(),
            "clap.audio-ports" => return (&raw const PORTS_EXT).cast(),
            "clap.timer-support" => return (&raw const TIMER_EXT).cast(),
            "clap.latency" => return (&raw const crate::latency_tail::LATENCY_EXT).cast(),
            "clap.tail" => return (&raw const crate::latency_tail::TAIL_EXT).cast(),
            "clap.gui" => return (&raw const crate::gui::GUI_EXT).cast(),
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
    use std::sync::atomic::AtomicBool;

    /// 造一个与宿主给的形状完全一致的实例记录，用于直接跑生命周期回调。
    /// 返回的 `clap_plugin_t` 保持在栈上；测试结束前手动释放 `Instance`。
    fn fixture() -> (clap_plugin_t, *mut Instance) {
        let instance = Box::into_raw(Box::new(Instance {
            state: state::InstanceState::default(),
            audio: audio::AudioState::default(),
            host: std::ptr::null(),
            plugin: std::ptr::null(),
            ara: None,
            initialized: AtomicBool::new(false),
            active: AtomicBool::new(false),
            processing: AtomicBool::new(false),
            in_process: AtomicBool::new(false),
            capture: crate::capture::CaptureHost::default(),
        }));
        let plugin = clap_plugin_t {
            desc: std::ptr::null(),
            plugin_data: instance.cast(),
            init: None,
            destroy: None,
            activate: None,
            deactivate: None,
            start_processing: None,
            stop_processing: None,
            reset: None,
            process: None,
            get_extension: None,
            on_main_thread: None,
        };
        (plugin, instance)
    }
    fn release(plugin: &mut clap_plugin_t, instance: *mut Instance) {
        plugin.plugin_data = std::ptr::null_mut();
        unsafe { drop(Box::from_raw(instance)) };
    }

    #[test]
    fn activate_requires_init_and_rejects_duplicates() {
        let (mut plugin, instance) = fixture();
        let ptr: *const clap_plugin_t = &raw const plugin;
        // 未 init 就 activate：必须拒绝。
        assert!(!unsafe { plugin_activate(ptr, 48_000.0, 1, 512) });
        // 用一个假的 host 让 init 通过（log_host::install 会安全跳过空扩展）。
        let host = crate::abi::clap_host_t {
            clap_version: CLAP_VERSION,
            host_data: std::ptr::null_mut(),
            name: c"probe".as_ptr(),
            vendor: c"probe".as_ptr(),
            url: c"probe".as_ptr(),
            version: c"1".as_ptr(),
            get_extension: None,
            request_restart: None,
            request_process: None,
            request_callback: None,
        };
        unsafe { (*instance).host = &raw const host };
        assert!(unsafe { plugin_init(ptr) });
        assert!(!unsafe { plugin_init(ptr) }, "duplicate init must fail");
        assert!(unsafe { plugin_activate(ptr, 48_000.0, 1, 512) });
        assert!(
            !unsafe { plugin_activate(ptr, 48_000.0, 1, 512) },
            "duplicate activate must fail"
        );
        assert!(unsafe { start_processing(ptr) });
        assert!(
            !unsafe { start_processing(ptr) },
            "duplicate start_processing must fail"
        );
        unsafe {
            stop_processing(ptr);
            plugin_deactivate(ptr);
        }
        release(&mut plugin, instance);
    }

    #[test]
    fn process_is_rejected_before_start_processing() {
        let (mut plugin, instance) = fixture();
        let ptr: *const clap_plugin_t = &raw const plugin;
        let host = crate::abi::clap_host_t {
            clap_version: CLAP_VERSION,
            host_data: std::ptr::null_mut(),
            name: c"probe".as_ptr(),
            vendor: c"probe".as_ptr(),
            url: c"probe".as_ptr(),
            version: c"1".as_ptr(),
            get_extension: None,
            request_restart: None,
            request_process: None,
            request_callback: None,
        };
        unsafe { (*instance).host = &raw const host };
        assert!(unsafe { plugin_init(ptr) });
        assert!(unsafe { plugin_activate(ptr, 48_000.0, 1, 512) });
        let process = clap_process_t {
            steady_time: 0,
            frames_count: 0,
            transport: std::ptr::null(),
            audio_inputs: std::ptr::null(),
            audio_outputs: std::ptr::null_mut(),
            audio_inputs_count: 0,
            audio_outputs_count: 0,
            in_events: std::ptr::null(),
            out_events: std::ptr::null(),
        };
        assert_eq!(
            unsafe { plugin_process(ptr, &raw const process) },
            CLAP_PROCESS_ERROR,
            "process before start_processing must fail"
        );
        assert!(unsafe { start_processing(ptr) });
        assert_eq!(
            unsafe { plugin_process(ptr, &raw const process) },
            CLAP_PROCESS_CONTINUE
        );
        unsafe {
            stop_processing(ptr);
            plugin_deactivate(ptr);
        }
        release(&mut plugin, instance);
    }

    /// **端到端验收**：宿主播放 → 页面按捕获 → 录制 → 停止 → 时间线上出现片段。
    ///
    /// 【为什么这条必须存在】前面的用例各自只验证一段（存储、协议、音频适配）。
    /// 用户报障的形态恰恰是"每一段都绿、合起来什么都不动"——接线处的错误（`process`
    /// 没喂捕获、页面命令走错实例、回执形状不对）只有把整条链跑一遍才看得见。
    /// 这里走的是与真实编辑器同一条路：C ABI 的 process 回调 + `protocol::reply`。
    #[test]
    fn a_host_playback_produces_a_clip_through_the_plugin_entry() {
        let (mut plugin, instance) = fixture();
        let ptr: *const clap_plugin_t = &raw const plugin;
        let host = crate::abi::clap_host_t {
            clap_version: CLAP_VERSION,
            host_data: std::ptr::null_mut(),
            name: c"probe".as_ptr(),
            vendor: c"probe".as_ptr(),
            url: c"probe".as_ptr(),
            version: c"1".as_ptr(),
            get_extension: None,
            request_restart: None,
            request_process: None,
            request_callback: None,
        };
        unsafe { (*instance).host = &raw const host };
        assert!(unsafe { plugin_init(ptr) });
        assert!(unsafe { plugin_activate(ptr, 48_000.0, 1, 512) });
        assert!(unsafe { start_processing(ptr) });

        let reply = |command: &str, args: serde_json::Value| {
            let request = serde_json::json!({ "id": 1, "viewId": "v", "version": 1, "command": command, "args": args });
            let capture = &unsafe { crate::instance(ptr) }.expect("instance").capture;
            crate::editor::protocol::reply("v", &request, capture)
        };

        // 页面在用户按键之前就能读到"能录"。
        assert_eq!(
            reply("capture_status", serde_json::Value::Null)["value"]["canCapture"],
            true
        );
        assert_eq!(reply("capture_start", serde_json::Value::Null)["ok"], true);

        // 宿主播放：每块 480 帧、位置按块长推进。
        let left: Vec<f32> = (0..480).map(|i| (i as f32 * 0.05).sin()).collect();
        let right = left.clone();
        let mut out_left = vec![0.0_f32; 480];
        let mut out_right = vec![0.0_f32; 480];
        for block in 0..40_i64 {
            let mut transport = crate::abi::clap_event_transport_t {
                header: crate::abi::clap_event_header_t {
                    size: std::mem::size_of::<crate::abi::clap_event_transport_t>() as u32,
                    time: 0,
                    space_id: 0,
                    type_: crate::abi::CLAP_EVENT_TRANSPORT,
                    flags: 0,
                },
                flags: crate::abi::transport_flags::HAS_SECONDS_TIMELINE
                    | crate::abi::transport_flags::IS_PLAYING,
                song_pos_beats: 0,
                song_pos_seconds: (block as f64 * 480.0 / 48_000.0
                    * crate::abi::CLAP_SECTIME_FACTOR)
                    .round() as i64,
                tempo: 120.0,
                tempo_inc: 0.0,
                loop_start_beats: 0,
                loop_end_beats: 0,
                loop_start_seconds: 0,
                loop_end_seconds: 0,
                bar_start: 0,
                bar_number: 0,
                tsig_num: 4,
                tsig_denom: 4,
            };
            let mut in_planes: [*mut f32; 2] =
                [left.as_ptr() as *mut f32, right.as_ptr() as *mut f32];
            let mut out_planes: [*mut f32; 2] = [out_left.as_mut_ptr(), out_right.as_mut_ptr()];
            let input = crate::abi::clap_audio_buffer_t {
                data32: in_planes.as_mut_ptr(),
                data64: std::ptr::null_mut(),
                channel_count: 2,
                latency: 0,
                constant_mask: 0,
            };
            let mut output = crate::abi::clap_audio_buffer_t {
                data32: out_planes.as_mut_ptr(),
                data64: std::ptr::null_mut(),
                channel_count: 2,
                latency: 0,
                constant_mask: 0,
            };
            let process = clap_process_t {
                steady_time: block * 480,
                frames_count: 480,
                transport: &mut transport,
                audio_inputs: &input,
                audio_outputs: &mut output,
                audio_inputs_count: 1,
                audio_outputs_count: 1,
                in_events: std::ptr::null(),
                out_events: std::ptr::null(),
            };
            assert_eq!(
                unsafe { plugin_process(ptr, &raw const process) },
                CLAP_PROCESS_CONTINUE
            );
        }

        let status = reply("capture_status", serde_json::Value::Null);
        assert_eq!(status["value"]["state"], "recording", "{status}");
        // 光标跟着宿主走：40 块 × 480 帧 = 0.4 s。
        let transport = reply("get_playback_state", serde_json::Value::Null);
        let position = transport["value"]["position_sec"].as_f64().unwrap_or(-1.0);
        assert!((position - 0.392).abs() < 0.02, "position was {position}");

        let stopped = reply("capture_stop", serde_json::Value::Null);
        assert_eq!(stopped["value"]["status"]["takeCount"], 1, "{stopped}");
        let timeline = reply("get_timeline_state", serde_json::Value::Null);
        let clips = timeline["value"]["clips"].as_array().expect("clips");
        assert_eq!(clips.len(), 1, "expected exactly one clip: {timeline}");
        let source = clips[0]["source_path"].as_str().expect("source path");
        let args = serde_json::json!({ "sourcePath": source, "level": 0 });
        let mipmap = reply("get_waveform_mipmap_binary", args);
        assert!(
            mipmap["value"]
                .as_str()
                .is_some_and(|data| !data.is_empty()),
            "the clip must serve waveform peaks: {mipmap}"
        );
        // 时间线版本必须前进，否则编辑器永远不会重取 —— 用户看到的就是"没有片段"。
        let apply = reply("plugin_get_apply_state", serde_json::Value::Null);
        assert_eq!(apply["value"]["generation"], serde_json::json!(1));

        unsafe {
            stop_processing(ptr);
            plugin_deactivate(ptr);
        }
        release(&mut plugin, instance);
    }

    #[test]
    fn a_capture_only_descriptor_does_not_advertise_ara() {
        let features = unsafe { std::slice::from_raw_parts(DESCRIPTOR.features, 3) };
        let strings: Vec<&str> = features
            .iter()
            .map(|p| unsafe { CStr::from_ptr(*p) }.to_str().unwrap())
            .collect();
        assert!(!strings.contains(&"ara:supported"));
        assert!(!strings.contains(&"ara:required"));
    }
    #[test]
    fn the_entry_declares_clap_1_1_9() {
        assert_eq!(clap_entry.clap_version, CLAP_VERSION)
    }
}
