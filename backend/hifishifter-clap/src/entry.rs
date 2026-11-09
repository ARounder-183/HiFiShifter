//! CLAP 模块入口：descriptor、工厂、实例生命周期与扩展分发。
//!
//! 【ARA 的两级入口】entry 级工厂表在 `crate::ara::entry_extension`，实例级扩展在
//! `crate::ara::get_extension`；两者背后是 VST3 那份共享 binder。descriptor 因此声明
//! `ara:supported`：按 ARACLAP.h 定稿协议发现的宿主据此才会在 entry 上要工厂、
//! 在实例上要扩展并 bind。
//!
//! 【为什么不声明 `ara:required`】本插件在没有 ARA 的宿主里仍是可用的普通效果器
//! （捕获形态）。`required` 会让那些宿主直接拒绝加载它 —— 那是把“支持 ARA”说成
//! “只会 ARA”。
//!
//! 【REAPER 的发现路径**不**被支持】REAPER 6.71+ 不查 entry 级 ARA 工厂、也不检查
//! `ara:supported`，而是在 `plugin->init` 后直接向实例查询 2022 私有草案 ID
//! `com.celemony.ara.plugin_extension.draft/1`。该草案的宿主实现在需要真正渲染时会
//! 崩在自己的代码里（`reaper.exe+0x5a677`，详见 `crate::ara` 的模块文档），所以
//! `crate::ara::get_extension` **不应答**它：REAPER 里的 CLAP 插件以 Capture 形态
//! 稳定运行，需要 ARA 时用 VST3 形态。`ara:supported` 只服务于按 ARACLAP.h 定稿
//! 协议发现的宿主。

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
    // 【为什么这一条要保留】ARACLAP.h 规定按定稿协议发现的宿主先看 descriptor 的
    // `ara:supported`，再查 entry 级工厂与实例级扩展。REAPER 6.71+ 走的是更早的
    // 私有草案路径，它既不读这个 feature、也不查 entry 级工厂 —— 那条路径上的宿主
    // 缺陷让插件无法安全应答（见 `crate::ara`），因此这里声明的是"定稿协议下支持
    // ARA"，不是"REAPER 的 CLAP ARA 可用"。
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
    // 【为什么第一件事就是登记标签】格式标签是共享层判断「我是谁」的唯一依据：日志
    // 行前缀、日志文件名（plugin-clap-<host>.log）与会话头全部由它派生。晚一步登记，
    // 最早那几行就会挂到默认的 VST3 文件名下 —— 那正是此前两种格式日志混在一起的成因。
    hifishifter_plugin::set_log_tag("clap");
    hifishifter_plugin_core::guard_ffi(false, || {
        // 【为什么资源根与进程状态排在日志初始化之前】`initialize_process_state` 会跑
        // 用户目录的一次性布局迁移，而迁移的判据是"新旧根同时存在就绝不合并"；
        // 日志初始化会先创建 `<本地根>\logs`，先装日志就等于把旧目录的日志永远留在原地。
        // VST3 的入口顺序（`ensure_layout_v2_once` → `diagnostics::init`）正是这个顺序，
        // CLAP 此前是反的 —— 两个格式在同一个宿主动作上得到不同的用户目录结果。
        if !plugin_path.is_null() {
            if let Ok(path) = unsafe { CStr::from_ptr(plugin_path) }.to_str() {
                let _ = MODULE_PATH.set(path.to_owned());
                resource_root::set_module_path(path).ok();
            }
        }
        resource_root::set_layout(ResourceLayout::for_clap()).ok();
        hifishifter_plugin::initialize_process_state();
        // 【为什么在共享层装日志器】文件日志只有一份实现（格式、会话头、轮转、级别、
        // panic hook）。CLAP 外壳只补一条可选的宿主转发 sink（见 log_host::install）。
        hifishifter_plugin::diagnostics::init();
        // 【为什么不再自己注册空钩子】`set_layout_migration_hook` 是进程级 OnceLock：
        // 空钩子会挡住 VST3 真正的那一份，而 CLAP-only 用户则永远不迁移。
        // 现在两个外壳登记同一个函数指针（见 `hifishifter_plugin::initialize_process_state`，
        // 它已在上面资源根就位之后调用）。
        // 模型登记与 VST3 的模块入口同一步：早调幂等，晚调会让"没开过窗口就导出"的
        // 会话先撞上 models unavailable。
        hifishifter_plugin::initialize_resources();

        // 【为什么也在这里建 ARA 运行时】VST3 的模块入口（`GetPluginFactory`）会建它；
        // 迟建只会把失败推后到第一次 `get_factory`，而那时宿主已经在等一张表。
        let _ = hifishifter_plugin::runtime::runtime();
        true
    })
}

unsafe extern "C" fn entry_deinit() {
    hifishifter_plugin_core::guard_ffi((), || {
        // 【为什么调共享的卸载序列】窗口类与 WebView2 回调都指向本模块的代码页；宿主
        // 随后的 `FreeLibrary` 若真的卸载它，下一条窗口消息就是野跳转。共享序列还负责
        // 等慢车道任务清空 —— 它们同样持有本模块的代码页。VST3 的 `ExitDll` 调的是
        // 同一个函数。
        if !hifishifter_plugin::shutdown_module() {
            log::warn!("[clap] module deinit with live editor windows; the module stays resident");
        }
        crate::log_host::clear();
        log::info!("[clap] module shutdown");
        // 把宽限期内仍攒在缓冲里的早期日志行落盘：缓冲只在「下一条记录到达」时落地，
        // 宿主不再打日志就退出的会话会把它整段丢掉。
        hifishifter_plugin::diagnostics::flush_pending();
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
        // ARA 工厂表（entry 级）：与 VST3 共用同一个 ARAFactory，只是关联名不同。
        crate::ara::entry_extension(id)
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
    /// 本实例的音频时间状态（采样率 / render 模式 / transport 观察值）。
    /// 绝不放到进程级 static：同一宿主进程里多实例会互相覆盖。
    pub audio: audio::AudioState,
    pub host: *const clap_host_t,
    pub plugin: *const clap_plugin_t,
    /// 本实例的扩展所有权：文档绑定、renderer 分配、捕获存储与编辑器会话都挂在这里。
    ///
    /// 【为什么每个实例都有，而不只是 ARA 实例】它是**共享语义**的唯一载体：捕获
    /// 观察、输出仲裁、编辑会话、参数与时间线都在 `ExtensionOwner` 上。此前 CLAP 只有
    /// ARA 实例才建它，于是 Capture 形态只能另起一套（`CaptureHost` + `RenderEngine`）
    /// —— 同一个命令在两种形态下有两种行为。
    pub owner: std::sync::Arc<hifishifter_plugin::render::extension::ExtensionOwner>,
    /// 实例级 ARA 绑定器；运行时不可用时为 None（实例退回 Capture 形态）。
    pub ara: Option<std::sync::Arc<hifishifter_plugin::ara::binder::AraBinder>>,
    /// `plugin->init` 是否已成功。CLAP 规定每个实例只调一次。
    pub initialized: AtomicBool,
    /// `plugin->activate` 是否已成功；`process` 必须在此之后。
    pub active: AtomicBool,
    /// `start_processing` / `stop_processing` 之间。
    pub processing: AtomicBool,
    /// `process` 是否正在执行（防同一实例并发进入音频回调）。
    pub in_process: AtomicBool,
    /// 本实例在进程内的租约。
    ///
    /// 【为什么必须持有它】`live_owners()` 只看这张表：没有租约的实例对宿主侧心跳、
    /// 编辑器链路与诊断读数都不存在。VST3 由组件的 `route` 字段持有同一份东西。
    pub route: hifishifter_plugin::editor::RouteLease,
    /// `clap.timer-support` 注册到的定时器 id；`u32::MAX` = 没挂上。
    ///
    /// 【为什么每个实例一个】宿主按实例回调 `on_timer`；共用 id 会让一个实例的节拍
    /// 去驱动另一个实例的心跳，而它们可能在不同线程/不同文档上。
    pub timer_id: std::sync::atomic::AtomicU32,
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
        let requested = unsafe { CStr::from_ptr(plugin_id) }
            .to_string_lossy()
            .into_owned();
        // 【为什么宿主身份要在第一行日志之前解析】日志文件名里的宿主段由它决定；
        // 晚一步解析，最早那几行会写进 `plugin-clap-unknown.log`，随后日志系统
        // 换名重建 —— 一次会话留下两个文件、两个会话头。VST3 在组件构造里就解析，
        // 这里与它对齐（判据本身是共享的 `resolve_host_identity`）。
        hifishifter_plugin::resolve_host_identity();

        // 【为什么每个实例建一次就记一行】"宿主为一份 ARA 文档建了几个实例"是判断
        // 宿主 ARA 实现是否完整的**唯一**外部读数：ARACLAP.h 规定一个 clap_plugin
        // 实例只承担一组角色，多角色必须多实例。宿主只建一个却要求多个角色时，
        // 现场就只剩"没有 clip"或"宿主崩溃"两种结局，而这一行是唯一能区分的证据。
        log::info!("[clap] create_plugin id={requested}");
        if requested.as_bytes() != b"com.hifishifter.clap" {
            log::warn!("[clap] create_plugin rejected unknown id={requested}");
            return std::ptr::null();
        }
        if !unsafe { clap_version_is_compatible((*host).clap_version) } {
            log::warn!("[clap] create_plugin rejected incompatible clap version from host");
            return std::ptr::null();
        }
        // 宿主在 `clap_host_t` 里自报名字与版本（CLAP 契约要求这两个字段非空）。
        // 这是 CLAP 侧唯一诚实的宿主身份来源：VST3 用 IHostApplication，两者最终都进
        // `ExtensionOwner::host_app_name`，读者与分支判据完全一致（且绝不按名字分支）。
        let host_name = unsafe { clap_host_string((*host).name) };
        let host_version = unsafe { clap_host_string((*host).version) };

        let owner = std::sync::Arc::new({
            let owner = hifishifter_plugin::render::extension::ExtensionOwner::default();
            owner.note_host_self_report(host_name.as_deref(), host_version.as_deref());
            owner
        });
        // 与 VST3 的组件构造同一步：把音频线程首块会读的一次性开关预热掉。
        owner.warm_audio_caches();
        let route = hifishifter_plugin::editor::RouteLease::new(&owner);
        let instance = Box::new(Instance {
            audio: audio::AudioState::default(),
            host,
            plugin: std::ptr::null(),
            owner,
            ara: None,
            initialized: AtomicBool::new(false),
            active: AtomicBool::new(false),
            processing: AtomicBool::new(false),
            in_process: AtomicBool::new(false),
            route,
            timer_id: std::sync::atomic::AtomicU32::new(u32::MAX),
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

/// 读一个宿主提供的 CLAP 字符串；空指针或非 UTF-8 一律 `None`。
///
/// 【为什么不 `unwrap`】这些字段由宿主进程写入。一个坏指针不该让插件在扫描阶段崩掉
/// —— 身份只是诊断事实，缺了它插件照常工作。
/// # Safety
/// `text` 必须是宿主在本次调用期间仍然有效的 NUL 结尾字符串，或空指针。
unsafe fn clap_host_string(text: *const c_char) -> Option<String> {
    if text.is_null() {
        return None;
    }
    let bytes = unsafe { CStr::from_ptr(text) }.to_str().ok()?.trim();
    (!bytes.is_empty()).then(|| bytes.to_owned())
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
        // 【为什么只走这一条】宿主身份（以及日志文件名里的宿主段）由共享层
        // `resolve_host_identity` 的进程证据链唯一决定，与 VST3 完全同一判据。
        // CLAP 的 `clap_host_t.name` 已经在 `create_plugin` 里交给
        // `ExtensionOwner::note_host_self_report` 作为诊断事实；如果在这里再直调
        // `note_host_application`，就会与证据链形成两个改名源 —— 一个报 "probe"、
        // 一个报 "unknown"，日志文件被反复重建、会话头叠成一串。
        hifishifter_plugin::resolve_host_identity();
        // SAFETY: 本函数就是宿主调用的 plugin.init，plugin 是本实例的记录。
        unsafe { crate::ara::attach(plugin) };
        // 【为什么节拍挂在 init 而不是 activate】它是"编辑器命令要回到宿主主线程"的
        // 唯一驱动（`host::marshal` 的泵），而编辑器可以在**未激活**的实例上打开
        // （宿主停播时打开 FX 窗口是常态）。VST3 的 REAPER 定时器同样挂在宿主适配器
        // 建立时，而不是 `setActive` 时。
        //
        // 【没有 timer-support 时】改走 `request_callback` 自挂：先要一次，之后每次
        // `on_main_thread` 结束再要下一次。没有这一条，那种宿主上 `on_main_thread`
        // 永远不会被调用（宿主只在被请求后回调）。
        if !unsafe { crate::timer::acquire(plugin) } {
            unsafe { crate::timer::request_main_thread(instance.host) };
        }
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
                // 宿主没按契约先调 `gui.destroy` 时，窗口与链路必须随实例一起消失：
                // 窗口类留在已卸载的模块里，下一条窗口消息就是野跳转。
                unsafe { crate::gui::forget(plugin) };
                unsafe { crate::timer::release(plugin) };

                // 组件终止：停掉会话 actor 与准备 worker（它们持有指向 owner 的强引用，
                // 不先关闭邮箱线程就不会退出）。与 VST3 的 terminate 同一个方法。
                instance.owner.component_terminated();
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
        // 【为什么接受域与 VST3 完全相同】它是共享常量（SUPPORTED_SAMPLE_RATE）：
        // 同一个工程在两种格式下必须得到同一个结论。此前 CLAP 只查"有限且大于 0"，
        // 于是 4 kHz / 400 kHz 在 CLAP 里被接受、在 VST3 里被拒绝。
        if !hifishifter_plugin::render::extension::SUPPORTED_SAMPLE_RATE.contains(&sample_rate) {
            log::warn!(
                "[clap] activate refused: sampleRate={sample_rate} is outside {}..={}",
                hifishifter_plugin::render::extension::SUPPORTED_SAMPLE_RATE.start(),
                hifishifter_plugin::render::extension::SUPPORTED_SAMPLE_RATE.end()
            );
            instance.active.store(false, Ordering::Release);
            return false;
        }
        instance.audio.set_sample_rate(sample_rate);
        // 【为什么单块上限也要记】它是诊断包里的 maxSamplesPerBlock（"宿主声明的
        // 块上限"），VST3 在 setupProcessing 里记同一个读数。不记的话 CLAP 会话里
        // 它恒为 0，排查"块长异常"时读不到宿主到底声明了什么。
        instance.owner.note_max_samples_per_block(max as i32);
        // 采样率、捕获渲染基础设施与离线意图：与 VST3 的 setupProcessing 同一条共享
        // 序列（见 `ExtensionOwner::setup_processing`）。CLAP 没有单独的 setup 回调，
        // 所以这三件事都由 activate 承担。
        if let Err(error) = instance
            .owner
            .setup_processing(sample_rate, instance.audio.is_offline())
        {
            log::warn!("[clap] offline preparation at activate failed: {error}");
        }
        // 激活序列（恢复挂起预览 / 清上一遍几何 / 主动重同步）：与 VST3 的
        // setActive(true) 是同一个方法。ARA 的绑定窗口也在这里关闭。
        instance.owner.component_activated();
        // 宿主侧节拍已在 `plugin.init` 挂上（见那里的说明）；激活不需要再动它。
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
        // 节拍与实例同生共死（`plugin.destroy` 才注销）：宿主可能只是暂时失活，
        // 而编辑器窗口仍然开着。
        instance.owner.component_deactivated();
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
        instance
            .owner
            .processing_started(instance.audio.is_offline());
        true
    })
}
unsafe extern "C" fn stop_processing(plugin: *const clap_plugin_t) {
    hifishifter_plugin_core::guard_ffi((), || {
        if let Some(instance) = unsafe { crate::instance(plugin) } {
            instance.processing.store(false, Ordering::Release);
            // 处理停止序列（收尾 pass、冻结宿主时钟、清流几何）：与 VST3 的
            // setProcessing(false) 是同一个方法。
            instance.owner.processing_stopped();
        }
    })
}
unsafe extern "C" fn reset_processing(plugin: *const clap_plugin_t) {
    hifishifter_plugin_core::guard_ffi((), || {
        // `clap/plugin.h` 的 reset 是"丢掉会跨 process 调用残留的状态"，宿主在定位、
        // 循环回卷与离线分段之间调它。本插件跨块残留的只有会话边界（流几何 / 每流播放位 /
        // rung latch / 捕获时钟）—— 与 `processing_stopped` 走同一个原子量序列。
        if let Some(instance) = unsafe { crate::instance(plugin) } {
            instance.owner.processing_reset();
        }
    })
}

/// 宿主音频回调：**只**做 panic 遏制、慢块计量与结果码映射。
///
/// 【为什么用 `catch_unwind` 而不是 `guard_ffi`】音频回调要把"被拦下的 panic"计进
/// 共享读数（`audio_stats`，VST3 的 `audio_process` 写的是同一组原子量）。`guard_ffi`
/// 内部自己 catch，外面再包一层就永远看不到 panic —— 计数恒为 0。FFI 门禁接受
/// `catch_unwind`（见 scripts/checks/check-ffi-guards.py 的 GUARD_MARKERS）。
unsafe extern "C" fn plugin_process(
    plugin: *const clap_plugin_t,
    process: *const clap_process_t,
) -> i32 {
    let started = std::time::Instant::now();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        plugin_process_body(plugin, process)
    }));
    hifishifter_plugin::audio_stats::note_block_elapsed(started.elapsed());
    match outcome {
        Ok(code) => code,
        Err(payload) => {
            hifishifter_plugin::audio_stats::note_panic();
            let message = payload
                .downcast_ref::<&str>()
                .map(|text| (*text).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic payload".to_owned());
            log::error!("[clap] audio callback panicked: {message}");
            // CLAP 的语义：返回错误 ⇒ 宿主必须丢弃本块输出。
            CLAP_PROCESS_ERROR
        }
    }
}

unsafe fn plugin_process_body(
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
            // 【为什么整条通路只有一处】捕获观察、相位判定、门控与快照输出都在
            // `ExtensionOwner::process_audio` 里（VST3 调的是同一个方法）。外壳只翻译
            // 缓冲与 transport，再把结局映射成 CLAP 的返回码 —— 于是两种格式不可能在
            // “这一块该出什么声”上分叉，两种**形态**（ARA / Capture）也不会。
            use hifishifter_plugin::render::extension::RenderOutcome;
            let _ = instance;
            match unsafe { crate::process::process_audio(plugin, process) } {
                // 宿主给的数据不可用 / 形状不受支持：如实上报。
                RenderOutcome::Invalid | RenderOutcome::Unsupported => CLAP_PROCESS_ERROR,
                // 写满整块 / 清零 / 直通：本块已按语义处理完。
                RenderOutcome::Done => CLAP_PROCESS_CONTINUE,
                // 【为什么 NotReady 也算失败】它只出现在离线流上（快照还没就绪）。
                // VST3 对同一结局回 `kResultFalse`，让宿主知道这一帧不可用；CLAP 的
                // 错误码语义正是"输出必须丢弃"。回成功会让宿主把静音**当成产物**
                // 烘焙进去 —— 那是数据正确性失败，不是可诊断的降级。
                RenderOutcome::NotReady => CLAP_PROCESS_ERROR,
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

unsafe extern "C" fn on_main_thread(plugin: *const clap_plugin_t) {
    hifishifter_plugin_core::guard_ffi((), || {
        // 宿主按 `clap_host.request_callback` 把控制权交回主线程：这正是"投递泵 +
        // 宿主侧周期工作"要的那条线程。内容与 VST3 的宿主节拍**同一份实现**。
        hifishifter_plugin::editor::host_main_thread_tick();
        // 【为什么这里要再请求一次】没有 `clap.timer-support` 的宿主不会自己再来；
        // 每次回调结束时补一次请求，就把"宿主主线程上的周期工作"接了起来。有 timer
        // 时不再请求（那条路更稳定，也不需要每次重新排队）。
        if let Some(instance) = unsafe { crate::instance(plugin) } {
            if instance.timer_id.load(Ordering::Acquire) == u32::MAX {
                unsafe { crate::timer::request_main_thread(instance.host) };
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    /// 造一个与宿主给的形状完全一致的实例记录，用于直接跑生命周期回调。
    /// 返回的 `clap_plugin_t` 保持在栈上；测试结束前手动释放 `Instance`。
    fn fixture() -> (clap_plugin_t, *mut Instance) {
        let owner =
            std::sync::Arc::new(hifishifter_plugin::render::extension::ExtensionOwner::default());
        let route = hifishifter_plugin::editor::RouteLease::new(&owner);
        let instance = Box::into_raw(Box::new(Instance {
            audio: audio::AudioState::default(),
            host: std::ptr::null(),
            plugin: std::ptr::null(),
            owner,
            ara: None,
            initialized: AtomicBool::new(false),
            active: AtomicBool::new(false),
            processing: AtomicBool::new(false),
            in_process: AtomicBool::new(false),
            route,
            timer_id: std::sync::atomic::AtomicU32::new(u32::MAX),
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

    /// **未绑定文档的 ARA 实例必须把宿主原声原样送出去**。
    ///
    /// 【为什么这条必须存在】CLAP 的 ARA 实例现在整块走**共享渲染通路**
    /// （`ExtensionOwner::process_audio`）。那条路在“没有文档”时的正确结论是**直通**，
    /// 而不是静音 —— 宿主只是把一个普通插入插在轨上，用户必须听见原声。这条用例把
    /// “共享通路确实接上了”与“退化结论正确”两件事一起钉住：它同时断言实例有绑定器
    /// （否则测的只是回退通路）与输出逐样本等于输入。
    #[test]
    fn an_ara_instance_without_a_document_passes_host_audio_through() {
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
        assert!(
            unsafe { (*instance).ara.is_some() },
            "this case only means something when the instance has an ARA binder"
        );
        assert!(unsafe { plugin_activate(ptr, 48_000.0, 1, 512) });
        assert!(unsafe { start_processing(ptr) });

        const FRAMES: usize = 256;
        let left: Vec<f32> = (0..FRAMES).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let right: Vec<f32> = (0..FRAMES).map(|i| (i as f32 * 0.03).cos() * 0.5).collect();
        let mut out_left = vec![0.0_f32; FRAMES];
        let mut out_right = vec![0.0_f32; FRAMES];
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
            song_pos_seconds: 0,
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
        let mut in_planes: [*mut f32; 2] = [left.as_ptr() as *mut f32, right.as_ptr() as *mut f32];
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
            steady_time: 0,
            frames_count: FRAMES as u32,
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
        assert_eq!(
            out_left, left,
            "the host signal must reach the output unchanged"
        );
        assert_eq!(
            out_right, right,
            "the host signal must reach the output unchanged"
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

        // 【为什么走视图端口】编辑命令只有一套实现（`EditorPort` → 会话 / 捕获命令）；
        // 直接调协议层的命令表测的会是另一条早已删除的路。
        let port = hifishifter_plugin::editor::port::EditorPort::new(
            std::sync::Arc::clone(&unsafe { crate::instance(ptr) }.expect("instance").owner),
            "v",
        );
        let reply = |command: &str, args: serde_json::Value| {
            let request = serde_json::json!({ "id": 1, "viewId": "v", "version": 1, "command": command, "args": args });
            port.reply(&request)
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
        assert_eq!(apply["value"]["mode"], "capture", "{apply}");
        // 【为什么看 host_version 而不是 generation】前端判“要不要重取时间线”读的是
        // host_version（`pluginApplyStore.poll`），generation 只是渲染阶段的计数。
        // 两个字段各有各的含义，这里钉的是**界面真的会刷新**那一条。
        assert!(
            apply["value"]["host_version"].as_u64().unwrap_or(0) >= 1,
            "the host version must advance after a capture, or the timeline never refreshes: {apply}"
        );

        unsafe {
            stop_processing(ptr);
            plugin_deactivate(ptr);
        }
        release(&mut plugin, instance);
    }

    /// **非 ARA（Capture）形态下，普通播放也必须驱动播放光标。**
    ///
    /// 【为什么这条必须存在】宿主采样通道（`ExtensionOwner::observe_transport`）此前
    /// 只由 VST3 外壳调用：CLAP 的 `process` 通路里根本没有这个调用点。于是现场出现
    /// 一个很奇怪的分裂 —— **捕获时光标跟随，普通播放时不动**：前者走
    /// `publish_capture_block`，后者本该走宿主采样通道，而那条通道在 CLAP 上从未接通。
    /// 两种表现并存让它看起来像前端问题，实际是整条通道缺失。
    ///
    /// 这条用例刻意**不调用 capture_start**：它要证明的就是"没有捕获时也有位置"。
    #[test]
    fn a_plain_host_playback_moves_the_playhead_without_capturing() {
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
        let owner =
            std::sync::Arc::clone(&unsafe { crate::instance(ptr) }.expect("instance").owner);
        let port =
            hifishifter_plugin::editor::port::EditorPort::new(std::sync::Arc::clone(&owner), "v");
        let reply = |command: &str| {
            let request = serde_json::json!({ "id": 1, "viewId": "v", "version": 1, "command": command, "args": serde_json::Value::Null });
            port.reply(&request)
        };

        // 一开始没有任何宿主块：位置是 0，且没在播放。
        assert_eq!(reply("get_playback_state")["value"]["is_playing"], false);

        const FRAMES: usize = 480;
        let mut out_left = vec![0.0_f32; FRAMES];
        let mut out_right = vec![0.0_f32; FRAMES];
        let mut feed = |playing: bool, block: i64| {
            let seconds = block as f64 * FRAMES as f64 / 48_000.0;
            let mut transport = crate::abi::clap_event_transport_t {
                header: crate::abi::clap_event_header_t {
                    size: std::mem::size_of::<crate::abi::clap_event_transport_t>() as u32,
                    time: 0,
                    space_id: 0,
                    type_: crate::abi::CLAP_EVENT_TRANSPORT,
                    flags: 0,
                },
                flags: crate::abi::transport_flags::HAS_SECONDS_TIMELINE
                    | if playing {
                        crate::abi::transport_flags::IS_PLAYING
                    } else {
                        0
                    },
                song_pos_beats: 0,
                song_pos_seconds: (seconds * crate::abi::CLAP_SECTIME_FACTOR).round() as i64,
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
            let live: Vec<f32> = (0..FRAMES).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
            let mut in_planes: [*mut f32; 2] =
                [live.as_ptr() as *mut f32, live.as_ptr() as *mut f32];
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
                steady_time: block * FRAMES as i64,
                frames_count: FRAMES as u32,
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
        };

        // 宿主播放 40 块 ≈ 0.4 s（没有捕获，也没有 ARA 文档）。
        for block in 0..40_i64 {
            feed(true, block);
        }
        let playing = reply("get_playback_state");
        assert_eq!(
            playing["value"]["is_playing"], true,
            "宿主在播放必须如实报出：{playing}"
        );
        let position = playing["value"]["position_sec"].as_f64().unwrap_or(-1.0);
        let expected = 39.0 * FRAMES as f64 / 48_000.0;
        assert!(
            (position - expected).abs() < 0.02,
            "播放光标必须跟随宿主位置（期望 {expected:.3}s，实际 {position:.3}s）：{playing}"
        );
        assert_eq!(
            playing["value"]["position_source"], "block",
            "位置应来自音频块通道：{playing}"
        );

        // 宿主停播：播放位必须如实回落，而不是被闩死。
        feed(false, 40);
        assert_eq!(
            reply("get_playback_state")["value"]["is_playing"],
            false,
            "宿主停播必须如实回落"
        );

        unsafe {
            stop_processing(ptr);
            plugin_deactivate(ptr);
        }
        release(&mut plugin, instance);
    }

    /// **实例级 ARA 扩展必须真的发得出来**。
    ///
    /// 【为什么这条必须存在】entry 级工厂表与实例级扩展是两条独立通路：宿主先按工厂
    /// 建文档，再对实例要 `ARACLAP.h` 的插件扩展去 bind。少了后者，宿主拿到的永远是
    /// null —— 表现是"工程里有这个插件，但 ARA 一条 region 都不给"，而日志里只有
    /// 工厂发现成功的痕迹。这里走与真实宿主同一条 C ABI 路径，钉住"实例确实交出了
    /// 扩展表"这一事实。
    #[test]
    fn an_initialised_instance_exposes_the_ara_plugin_extension() {
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
        let id = c"org.ara-audio.ara.pluginextension/2".as_ptr();
        let extension = unsafe { plugin_get_extension(ptr, id) };
        assert!(
            !extension.is_null(),
            "an initialised instance must expose the ARA plug-in extension"
        );
        // 同一个指针必须稳定：宿主可能反复查询，每次返回新表会让它拿到的回调对不上。
        assert_eq!(
            extension,
            unsafe { plugin_get_extension(ptr, id) },
            "the extension table address must be stable"
        );
        unsafe {
            plugin_deactivate(ptr);
        }
        release(&mut plugin, instance);
    }

    /// **descriptor 必须声明 `ara:supported`**。
    ///
    /// 【为什么这条必须存在】按 ARACLAP.h 定稿协议发现的宿主只在看到该 feature 时
    /// 才去查两级 ARA 扩展。REAPER 的旧草案路径虽然不依赖它，但 descriptor 仍须如实
    /// 声明能力，否则定稿宿主会完全跳过 ARA 发现。
    ///
    /// 【为什么不声明 `ara:required`】见模块文档：没有 ARA 的宿主里本插件仍是可用的
    /// 普通效果器。`required` 会让那些宿主直接拒绝加载。
    #[test]
    fn the_descriptor_advertises_ara_support_without_requiring_it() {
        let features = unsafe { std::slice::from_raw_parts(DESCRIPTOR.features, 4) };
        let strings: Vec<&str> = features
            .iter()
            .map(|p| unsafe { CStr::from_ptr(*p) }.to_str().unwrap())
            .collect();
        assert!(
            strings.contains(&"ara:supported"),
            "a host only asks for the ARA extensions when this feature is present: {strings:?}"
        );
        assert!(
            !strings.contains(&"ara:required"),
            "the plug-in stays usable as a plain effect outside ARA hosts"
        );
    }
    #[test]
    fn the_entry_declares_clap_1_1_9() {
        assert_eq!(clap_entry.clap_version, CLAP_VERSION)
    }

    /// **CLAP 状态机不变量**：宿主停播后仍持续送块时，捕获不得自行收成 take。
    ///
    /// 【为什么钉这条】现场日志里 REAPER 7.85 的 CLAP 通路每次只播约 1 s，随后
    /// kPlaying 落 0、transport 位置回到这一遍的起点并持续送静音块；用户看到的就是
    /// "捕获时长走到 1 s 又自动终止"。这条用例按现场形状（beats 轴 + 停播后位置回绕 +
    /// 静音输入 + 非音频心跳）走一遍，把"停播不结束录制"这个设计意图钉住：
    /// 无论现场那次自动收口最终来自上限、预览心跳还是别的路径，状态机本身都不允许
    /// 宿主的一次停播把这一遍 promote 掉。
    ///
    /// 【为什么用 beats 轴】REAPER 的 CLAP transport 实测只给 HAS_BEATS_TIMELINE +
    /// HAS_TEMPO；只认 seconds 的夹具测的是另一条路。
    ///
    /// 【判据】一次 capture_start 之后，宿主中间的停播/重播都属于同一遍录制：
    /// 在用户按 capture_stop 之前，takeCount 必须是 0，且状态保持 armed。
    #[test]
    fn a_host_pause_does_not_promote_a_one_second_take() {
        let (mut plugin, instance) = fixture();
        let ptr: *const clap_plugin_t = &raw const plugin;
        let host = crate::abi::clap_host_t {
            clap_version: CLAP_VERSION,
            host_data: std::ptr::null_mut(),
            name: c"REAPER".as_ptr(),
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
        let owner =
            std::sync::Arc::clone(&unsafe { crate::instance(ptr) }.expect("instance").owner);
        let port =
            hifishifter_plugin::editor::port::EditorPort::new(std::sync::Arc::clone(&owner), "v");
        let reply = |command: &str, args: serde_json::Value| {
            let request = serde_json::json!({ "id": 1, "viewId": "v", "version": 1, "command": command, "args": args });
            port.reply(&request)
        };
        assert_eq!(reply("capture_start", serde_json::Value::Null)["ok"], true);

        let frames = 512_u32;
        let rate = 48_000.0_f64;
        let left: Vec<f32> = vec![0.0_f32; frames as usize];
        let right = left.clone();
        let mut out_left = vec![0.0_f32; frames as usize];
        let mut out_right = vec![0.0_f32; frames as usize];
        let mut block_index = 0_i64;
        let mut feed = |playing: bool, position_frames: i64| {
            // REAPER 给拍轴：music 秒 -> beats = 秒 * tempo/60。
            let seconds = position_frames as f64 / rate;
            let beats = seconds * 120.0 / 60.0;
            let mut transport = crate::abi::clap_event_transport_t {
                header: crate::abi::clap_event_header_t {
                    size: std::mem::size_of::<crate::abi::clap_event_transport_t>() as u32,
                    time: 0,
                    space_id: 0,
                    type_: crate::abi::CLAP_EVENT_TRANSPORT,
                    flags: 0,
                },
                flags: crate::abi::transport_flags::HAS_BEATS_TIMELINE
                    | crate::abi::transport_flags::HAS_TEMPO
                    | if playing {
                        crate::abi::transport_flags::IS_PLAYING
                    } else {
                        0
                    },
                song_pos_beats: (beats * crate::abi::CLAP_BEATTIME_FACTOR).round() as i64,
                song_pos_seconds: 0,
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
                steady_time: block_index * frames as i64,
                frames_count: frames,
                transport: &mut transport,
                audio_inputs: &input,
                audio_outputs: &mut output,
                audio_inputs_count: 1,
                audio_outputs_count: 1,
                in_events: std::ptr::null(),
                out_events: std::ptr::null(),
            };
            block_index += 1;
            assert_eq!(
                unsafe { plugin_process(ptr, &raw const process) },
                CLAP_PROCESS_CONTINUE
            );
        };

        // 播 1 s（约 94 块 @512/48k）。
        for i in 0..94_i64 {
            feed(true, i * frames as i64);
        }
        // 停播后 REAPER 把 transport 位置重置回这一遍的起点，并持续送静音块
        // （现场日志：playing=0 时 raw 从 429264 回到 381648，music 同步回 15.902）。
        let pass_start = 0_i64;
        for _ in 0..470_i64 {
            feed(false, pass_start);
        }
        // 非音频线程心跳：补给线 + 自动 promote 都挂在这里。
        for _ in 0..10 {
            owner.capture().service();
            owner.tick_preview_quiet();
        }

        let status = reply("capture_status", serde_json::Value::Null);
        let take_count = status["value"]["takeCount"].as_u64().unwrap_or(u64::MAX);
        let state = status["value"]["state"].as_str().unwrap_or("?").to_owned();
        assert_eq!(take_count, 0, "宿主停播不得把这一遍收成 take：{status}");
        assert!(
            state == "recording" || state == "standby",
            "停播后仍应保持武装，实际 state={state}：{status}"
        );

        // 用户真正按停止才产生唯一一个 take。
        let stopped = reply("capture_stop", serde_json::Value::Null);
        assert_eq!(stopped["value"]["status"]["takeCount"], 1, "{stopped}");
        unsafe {
            stop_processing(ptr);
            plugin_deactivate(ptr);
        }
        release(&mut plugin, instance);
    }

    /// **宿主根本没播放时按下捕获，不得开始计时、不得自己收口。**
    ///
    /// 【为什么与上一节并存】上一节钉的是"播过一遍、停播后不 promote"；这一节钉的是
    /// 现场报障的正身：capture_start 之后宿主一直是 kPlaying=0（REAPER 停播时仍持续
    /// process 固定光标处的静音块）。此前 note_stream_geometry 在 capture_start 重置
    /// 流几何后的**第一块**上无条件伪造一个起播上升沿，打开 1 秒有界起点窗口 ⇒ 静音
    /// 块被逐块记账，界面上"计时走到 1 s 就自动停"，take 里 first_signal_at=none。
    /// 修复后即时给出 kPlaying 的宿主（REAPER）一律以它为准：停播块 transport_active
    /// 恒为假，capturedFrames 必须一直是 0，用户按停止之前 takeCount 必须是 0。
    #[test]
    fn a_stopped_host_never_starts_capture_counting() {
        let (mut plugin, instance) = fixture();
        let ptr: *const clap_plugin_t = &raw const plugin;
        let host = crate::abi::clap_host_t {
            clap_version: CLAP_VERSION,
            host_data: std::ptr::null_mut(),
            name: c"REAPER".as_ptr(),
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
        let owner =
            std::sync::Arc::clone(&unsafe { crate::instance(ptr) }.expect("instance").owner);
        let port =
            hifishifter_plugin::editor::port::EditorPort::new(std::sync::Arc::clone(&owner), "v");
        let reply = |command: &str, args: serde_json::Value| {
            let request = serde_json::json!({ "id": 1, "viewId": "v", "version": 1, "command": command, "args": args });
            port.reply(&request)
        };
        assert_eq!(reply("capture_start", serde_json::Value::Null)["ok"], true);

        let frames = 512_u32;
        let rate = 48_000.0_f64;
        let silent: Vec<f32> = vec![0.0_f32; frames as usize];
        let mut out_left = vec![0.0_f32; frames as usize];
        let mut out_right = vec![0.0_f32; frames as usize];
        let mut block_index = 0_i64;
        let mut feed = |playing: bool, position_frames: i64, live: &[f32]| {
            let seconds = position_frames as f64 / rate;
            let beats = seconds * 120.0 / 60.0;
            let mut transport = crate::abi::clap_event_transport_t {
                header: crate::abi::clap_event_header_t {
                    size: std::mem::size_of::<crate::abi::clap_event_transport_t>() as u32,
                    time: 0,
                    space_id: 0,
                    type_: crate::abi::CLAP_EVENT_TRANSPORT,
                    flags: 0,
                },
                flags: crate::abi::transport_flags::HAS_BEATS_TIMELINE
                    | crate::abi::transport_flags::HAS_TEMPO
                    | if playing {
                        crate::abi::transport_flags::IS_PLAYING
                    } else {
                        0
                    },
                song_pos_beats: (beats * crate::abi::CLAP_BEATTIME_FACTOR).round() as i64,
                song_pos_seconds: 0,
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
                [live.as_ptr() as *mut f32, live.as_ptr() as *mut f32];
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
                steady_time: block_index * frames as i64,
                frames_count: frames,
                transport: &mut transport,
                audio_inputs: &input,
                audio_outputs: &mut output,
                audio_inputs_count: 1,
                audio_outputs_count: 1,
                in_events: std::ptr::null(),
                out_events: std::ptr::null(),
            };
            block_index += 1;
            assert_eq!(
                unsafe { plugin_process(ptr, &raw const process) },
                CLAP_PROCESS_CONTINUE
            );
        };

        // 宿主一直停播：持续送固定光标处的静音块（远超旧实现那 1 秒窗口）。
        for _ in 0..400_i64 {
            feed(false, 381_648, &silent);
        }
        for _ in 0..10 {
            owner.capture().service();
            owner.tick_preview_quiet();
        }

        let status = reply("capture_status", serde_json::Value::Null);
        assert_eq!(
            status["value"]["takeCount"].as_u64().unwrap_or(u64::MAX),
            0,
            "停播期间不得产生任何 take：{status}"
        );
        assert!(
            status["value"]["recording"].is_null(),
            "宿主没播放 ⇒ 仍是待命，根本不该有录制读数（计时必须停在 0）：{status}"
        );
        assert_eq!(
            status["value"]["state"].as_str().unwrap_or("?"),
            "standby",
            "停播时状态必须是待命而不是 recording：{status}"
        );

        // 用户真正按播放：这一遍才开始积累素材。
        let live: Vec<f32> = (0..frames as usize)
            .map(|i| (i as f32 / 64.0).sin() * 0.25)
            .collect();
        for i in 0..94_i64 {
            feed(true, 381_648 + i * frames as i64, &live);
        }
        // 用户按停止 ⇒ 唯一一个 take，且带真实内容。
        let stopped = reply("capture_stop", serde_json::Value::Null);
        assert_eq!(stopped["value"]["status"]["takeCount"], 1, "{stopped}");
        unsafe {
            stop_processing(ptr);
            plugin_deactivate(ptr);
        }
        release(&mut plugin, instance);
    }

    /// **`clap.render.set(OFFLINE)` 必须真的走到共享 setup 序列。**
    ///
    /// 【为什么这条必须存在】`clap.render.set` 是 CLAP 里"宿主声明这一段要离线渲染"
    /// 的唯一时刻，等价于 VST3 的 `setupProcessing(mode=2)`。它此前只记了一个布尔量：
    /// 进程模式恒为 0 ⇒ `offline_pass_active` 恒假 ⇒ 离线准备不要求用户曲线 ⇒
    /// bounce 产物里是没处理的声音（正是 `capture_render_input` 文档里那条失败）。
    #[test]
    fn an_offline_render_mode_runs_the_shared_setup_sequence() {
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
        let owner = unsafe { &(*instance).owner };
        assert_eq!(owner.setup_process_mode(), 0, "实时激活必须是实时模式");
        // SAFETY: 本实例已 init + activate；`clap.render.set` 允许在任何时候调用。
        let set = RENDER_EXT.set.expect("clap.render.set must be exported");
        assert!(unsafe { set(ptr, crate::abi::CLAP_RENDER_OFFLINE) });
        assert_eq!(
            owner.setup_process_mode(),
            2,
            "离线模式必须落进共享的 setup 序列（与 VST3 的 setupProcessing(mode=2) 同一条）"
        );
        assert!(owner.capture().project_rate() > 0.0, "采样率必须由同一次 setup 落定");
        unsafe {
            stop_processing(ptr);
            plugin_deactivate(ptr);
        }
        release(&mut plugin, instance);
    }

    /// **64-bit 块必须被记成"不支持的精度"。**
    ///
    /// 【为什么这条必须存在】"静音"在界面上必须能说清原因（`CaptureStore::unsupported_precision`）。
    /// VST3 在 process 入口记这个读数，CLAP 此前从不记 —— 于是 CLAP 会话里
    /// 那个位永远是"32-bit"，用户只看到"没有声音"而没有任何原因。
    #[test]
    fn a_sixty_four_bit_block_is_recorded_as_unsupported_precision() {
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
        let mut left = vec![1.0_f64; 4];
        let mut planes = [left.as_mut_ptr()];
        let buffer = crate::abi::clap_audio_buffer_t {
            data32: std::ptr::null_mut(),
            data64: planes.as_mut_ptr(),
            channel_count: 1,
            latency: 0,
            constant_mask: 0,
        };
        let process = clap_process_t {
            steady_time: 0,
            frames_count: 4,
            transport: std::ptr::null(),
            audio_inputs: &buffer,
            audio_outputs: &buffer as *const _ as *mut _,
            audio_inputs_count: 1,
            audio_outputs_count: 1,
            in_events: std::ptr::null(),
            out_events: std::ptr::null(),
        };
        assert_eq!(
            unsafe { plugin_process(ptr, &raw const process) },
            CLAP_PROCESS_CONTINUE,
            "64-bit 宽度必须静音但在场，而不是让宿主把插件踢出链路"
        );
        assert_eq!(left, vec![0.0; 4], "64-bit 输出必须被清零");
        assert!(
            unsafe { (*instance).owner.capture().precision_unsupported() },
            "精度必须记账，否则界面无法解释为什么是静音"
        );
        unsafe {
            stop_processing(ptr);
            plugin_deactivate(ptr);
        }
        release(&mut plugin, instance);
    }

    /// **读不出来的存档必须把恢复状态标成失败。**
    ///
    /// 【为什么这条必须存在】`restore_state` 失败时只写一行日志，界面与诊断包都不知道
    /// 参数没读回来 —— 用户看到的是"参数没了，日志里一行错都没有"。VST3 在
    /// `IComponent::setState` 里调 `note_restore_failed`，CLAP 此前不调。
    #[test]
    fn an_unreadable_payload_marks_the_restore_as_failed() {
        struct Cursor {
            data: Vec<u8>,
            at: usize,
        }
        unsafe extern "C" fn read_cursor(
            stream: *const crate::state::clap_istream_t,
            buffer: *mut std::ffi::c_void,
            size: u64,
        ) -> i64 {
            // SAFETY: ctx 由调用方指向本帧内的 Cursor。
            let cursor = unsafe { &mut *(*stream).ctx.cast::<Cursor>() };
            let remaining = cursor.data.len().saturating_sub(cursor.at);
            if remaining == 0 {
                return 0;
            }
            let count = (size as usize).min(remaining);
            // SAFETY: buffer 至少可写 count 字节（宿主契约）。
            unsafe {
                std::ptr::copy_nonoverlapping(
                    cursor.data.as_ptr().add(cursor.at),
                    buffer.cast::<u8>(),
                    count,
                )
            };
            cursor.at += count;
            count as i64
        }
        let (mut plugin, instance) = fixture();
        let ptr: *const clap_plugin_t = &raw const plugin;
        let mut cursor = Cursor {
            data: b"this is not a component state".to_vec(),
            at: 0,
        };
        let stream = crate::state::clap_istream_t {
            ctx: (&raw mut cursor).cast(),
            read: Some(read_cursor),
        };
        assert!(
            unsafe { crate::state::STATE_EXT.load.expect("clap.state.load")(ptr, &raw const stream) },
            "读不出来的存档不能让宿主认为整份工程加载失败"
        );
        assert_eq!(
            unsafe { (*instance).owner.restore_status_name() },
            "failed",
            "失败必须可见：界面据此报 restore_unresolved"
        );
        release(&mut plugin, instance);
    }

    /// **没有 `clap.timer-support` 的宿主必须被 `request_callback` 驱动起来。**
    ///
    /// 【为什么这条必须存在】`clap_plugin.on_main_thread` 只有被请求之后才会被宿主
    /// 调用。没有 timer-support 时若一次都不请求，`host_main_thread_tick` 在那类宿主
    /// 上一个环节都不跑 —— 而它们恰好是"播放头不跟随、工程不标脏"最容易出现的地方。
    /// 这条用例钉住"先要一次、每次回调再要下一次"的自挂节拍。
    #[test]
    fn a_host_without_timer_support_is_driven_by_request_callback() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        unsafe extern "C" fn count(_host: *const clap_host_t) {
            CALLS.fetch_add(1, Ordering::SeqCst);
        }
        let (mut plugin, instance) = fixture();
        let ptr: *const clap_plugin_t = &raw const plugin;
        // `get_extension: None` ⇒ 宿主没有 clap.timer-support（也没有 clap.log）。
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
            request_callback: Some(count),
        };
        unsafe { (*instance).host = &raw const host };
        let before = CALLS.load(Ordering::SeqCst);
        assert!(unsafe { plugin_init(ptr) });
        assert_eq!(
            CALLS.load(Ordering::SeqCst),
            before + 1,
            "没有 timer-support 时必须先向宿主要一次主线程回调"
        );
        unsafe { on_main_thread(ptr) };
        assert_eq!(
            CALLS.load(Ordering::SeqCst),
            before + 2,
            "每次 on_main_thread 结束都要再要一次，否则节拍只来一次"
        );
        release(&mut plugin, instance);
    }
}
