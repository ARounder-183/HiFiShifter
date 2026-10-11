//! HiFiShifter 的手写 VST3 模块 ABI：产品级插件外壳（工厂 + 组件/音频处理器 +
//! 编辑控制器 + 连接点）。
//!
//! REAPER 等宿主加载本 cdylib、把它识别为 ARA 插件，并把 ARA 对象推给
//! `ara2-bridge` 的文档控制器（见 `model.rs` 的计数）。渲染调度、快照输出与
//! 编辑器生命周期都在本文件与 `render/`、`editor/` 模块里落地。
//!
//! 这里为什么手写 vtbl 而不引入 `vst3-sys`：本机 cargo 缓存里没有 VST3 绑定，
//! 且只用得到少量接口。vtbl 的槽位顺序、结构体布局与调用约定都按锁定的
//! VST3 SDK `v3.8.0_build_66` 头文件逐条对齐（`ipluginbase.h` / `ivstcomponent.h` /
//! `ivstaudioprocessor.h` / `funknown.h`）。ARA 侧的接口（`IMainFactory`、
//! `IPlugInEntryPoint`、`IPlugInEntryPoint2`）不在这里实现，而是转发给
//! `ara2-bridge-companion` 的 C++ 适配器，避免重复实现引用计数语义。

use crate::ara_entry::HostEntry;
use crate::render::extension::ExtensionOwner;
use ara2_bridge::companion::vst3::ffi::{
    ara2_vst3_interface_id, ara2_vst3_main_factory_category, Ara2Vst3InterfaceId,
    Ara2Vst3InterfaceKind, ARA2_VST3_OK,
};
use ara2_bridge::companion::vst3::Vst3MainFactoryAdapter;
use ara2_bridge::companion::{CompanionProcessorBinding, CompanionRoles};
use std::ffi::{c_char, c_void};
use std::mem::offset_of;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;

/// VST3 `tresult`（HRESULT 风格）。
pub type TResult = i32;
/// VST3 16 字节接口/类标识。
pub type Tuid = [u8; 16];

/// `kResultOk`。
pub const K_RESULT_OK: TResult = 0;
/// `kResultFalse`。
pub const K_RESULT_FALSE: TResult = 1;
/// `kNotImplemented`。
pub const K_NOT_IMPLEMENTED: TResult = 0x8000_4001u32 as i32;
/// `kNoInterface`。
pub const K_NO_INTERFACE: TResult = 0x8000_4002u32 as i32;
/// `kInvalidArgument`。
pub const K_INVALID_ARGUMENT: TResult = 0x8007_0057u32 as i32;
/// `PClassInfo::kManyInstances`。
const K_MANY_INSTANCES: i32 = 0x7FFF_FFFF;

/// 处理器组件类 ID（探针自定，稳定即可）。
const PROCESSOR_CID: Tuid = *b"HFSARAProbe.Prc1";
/// 编辑器控制器类 ID（探针自定，稳定即可）。
const CONTROLLER_CID: Tuid = *b"HFSARAProbe.Ctl1";
/// ARA 主工厂类 ID（探针自定，稳定即可）。
const FACTORY_CID: Tuid = *b"HFSARAProbe.Fac1";

/// `Component Controller Class`（`kVstComponentControllerClass`）。
const COMPONENT_CONTROLLER_CLASS: &str = "Component Controller Class";

/// 从 `DECLARE_CLASS_IID` 的四个 32 位字构造 TUID（Windows VST3 布局）。
///
/// 这是实测结果，不是推断：Windows 上 VST3 SDK 的 `fplatform.h` 把 `COM_COMPATIBLE`
/// 定义为 1，其 `INLINE_UID` 展开等价于标准 GUID 内存布局 ——
/// 第 1 个字小端、第 2 个字拆成两个 u16 各自小端、第 3/4 个字大端。
/// 依据：REAPER 请求 `IPluginFactory2` 时传出的字节
/// `50B607004BF20B4CA464EDB9F00B2ABB`，正是
/// `{0007B650-F24B-4C0B-A464-EDB9F00B2ABB}` 的 GUID 布局。
pub(crate) const fn uid_guid(words: [u32; 4]) -> Tuid {
    let [l1, l2, l3, l4] = words;
    [
        (l1 & 0xFF) as u8,
        ((l1 >> 8) & 0xFF) as u8,
        ((l1 >> 16) & 0xFF) as u8,
        ((l1 >> 24) & 0xFF) as u8,
        ((l2 >> 16) & 0xFF) as u8,
        ((l2 >> 24) & 0xFF) as u8,
        (l2 & 0xFF) as u8,
        ((l2 >> 8) & 0xFF) as u8,
        ((l3 >> 24) & 0xFF) as u8,
        ((l3 >> 16) & 0xFF) as u8,
        ((l3 >> 8) & 0xFF) as u8,
        (l3 & 0xFF) as u8,
        ((l4 >> 24) & 0xFF) as u8,
        ((l4 >> 16) & 0xFF) as u8,
        ((l4 >> 8) & 0xFF) as u8,
        (l4 & 0xFF) as u8,
    ]
}

/// `COM_COMPATIBLE=0` 时的备用布局：每个字大端展开。
const fn uid_be(words: [u32; 4]) -> Tuid {
    let [l1, l2, l3, l4] = words;
    [
        ((l1 >> 24) & 0xFF) as u8,
        ((l1 >> 16) & 0xFF) as u8,
        ((l1 >> 8) & 0xFF) as u8,
        (l1 & 0xFF) as u8,
        ((l2 >> 24) & 0xFF) as u8,
        ((l2 >> 16) & 0xFF) as u8,
        ((l2 >> 8) & 0xFF) as u8,
        (l2 & 0xFF) as u8,
        ((l3 >> 24) & 0xFF) as u8,
        ((l3 >> 16) & 0xFF) as u8,
        ((l3 >> 8) & 0xFF) as u8,
        (l3 & 0xFF) as u8,
        ((l4 >> 24) & 0xFF) as u8,
        ((l4 >> 16) & 0xFF) as u8,
        ((l4 >> 8) & 0xFF) as u8,
        (l4 & 0xFF) as u8,
    ]
}

const IID_FUNKNOWN: [u32; 4] = [0x0000_0000, 0x0000_0000, 0xC000_0000, 0x0000_0046];
const IID_IPLUGIN_BASE: [u32; 4] = [0x2288_8DDB, 0x156E_45AE, 0x8358_B348, 0x0819_0625];
const IID_IPLUGIN_FACTORY: [u32; 4] = [0x7A4D_811C, 0x5211_4A1F, 0xAED9_D2EE, 0x0B43_BF9F];
const IID_IPLUGIN_FACTORY2: [u32; 4] = [0x0007_B650, 0xF24B_4C0B, 0xA464_EDB9, 0xF00B_2ABB];
const IID_ICOMPONENT: [u32; 4] = [0xE831_FF31, 0xF2D5_4301, 0x928E_BBEE, 0x2569_7802];
const IID_IAUDIO_PROCESSOR: [u32; 4] = [0x4204_3F99, 0xB7DA_453C, 0xA569_E79D, 0x9AAE_C33D];
const IID_IEDIT_CONTROLLER: [u32; 4] = [0xDCD7_BBE3, 0x7742_448D, 0xA874_AACC, 0x979C_759E];

/// 判断宿主传入的 16 字节 IID 是否等于给定接口（四个 32 位字）。
///
/// 同时接受 GUID 布局与逐字大端布局：Windows 实测是前者，但换一种宿主构建方式
/// 就可能变成后者。接受两种可以避免"因为字节序猜错而误判为 ABI 不兼容"，
/// 对探针而言多接受的错误匹配没有实际风险。
pub(crate) unsafe fn iid_matches(iid: *const u8, words: [u32; 4]) -> bool {
    if iid.is_null() {
        return false;
    }
    // SAFETY: 调用方保证 iid 指向一个可读的 16 字节 TUID。
    let actual = unsafe { std::slice::from_raw_parts(iid, 16) };
    actual == uid_guid(words) || actual == uid_be(words)
}

/// 把 16 字节标识转成十六进制，便于在日志里核对宿主实际请求的 IID。
fn hex16(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(32);
    for byte in bytes.iter().take(16) {
        out.push_str(&format!("{byte:02X}"));
    }
    out
}

/// 读取宿主传入的 16 字节 IID 并转成十六进制（空指针时返回 `"<null>"`）。
unsafe fn hex_iid(iid: *const u8) -> String {
    if iid.is_null() {
        return "<null>".to_owned();
    }
    // SAFETY: 调用方保证 iid 指向可读的 16 字节 TUID。
    hex16(unsafe { std::slice::from_raw_parts(iid, 16) })
}

/// 读取 companion 侧声明的 ARA 接口 IID（四个 32 位字）。
fn ara_iid_words(kind: Ara2Vst3InterfaceKind) -> Option<[u32; 4]> {
    let mut id = Ara2Vst3InterfaceId { words: [0; 4] };
    // SAFETY: 输出槽可写，kind 是合法的枚举值。
    let result = unsafe { ara2_vst3_interface_id(kind, &mut id) };
    if result == ARA2_VST3_OK {
        Some(id.words)
    } else {
        None
    }
}

/// `PFactoryInfo`。
#[repr(C)]
pub struct PFactoryInfo {
    /// 厂商名（64 字节）。
    pub vendor: [c_char; 64],
    /// 厂商 URL（256 字节）。
    pub url: [c_char; 256],
    /// 联系邮箱（128 字节）。
    pub email: [c_char; 128],
    /// `PFactoryInfo::FactoryFlags`。
    pub flags: i32,
}

/// `PClassInfo`。
#[repr(C)]
pub struct PClassInfo {
    /// 类 ID。
    pub cid: Tuid,
    /// 基数（多实例用 `kManyInstances`）。
    pub cardinality: i32,
    /// 类类别（32 字节，如 `Audio Module Class`）。
    pub category: [c_char; 32],
    /// 类名（64 字节，必须等于 ARA 工厂的 plugInName）。
    pub name: [c_char; 64],
}

/// `PClassInfo2`（`IPluginFactory2::getClassInfo2`）。
#[repr(C)]
pub struct PClassInfo2 {
    /// 类 ID。
    pub cid: Tuid,
    /// 基数。
    pub cardinality: i32,
    /// 类类别。
    pub category: [c_char; 32],
    /// 类名。
    pub name: [c_char; 64],
    /// 类别标志。
    pub class_flags: u32,
    /// 子类别（128 字节，ARA 约定 `Fx|OnlyARA`）。
    pub sub_categories: [c_char; 128],
    /// 厂商覆盖（64 字节）。
    pub vendor: [c_char; 64],
    /// 版本（64 字节）。
    pub version: [c_char; 64],
    /// SDK 版本（64 字节）。
    pub sdk_version: [c_char; 64],
}

/// `Vst::BusInfo`。
#[repr(C)]
pub struct BusInfo {
    /// `Vst::MediaTypes`。
    pub media_type: i32,
    /// `Vst::BusDirections`。
    pub direction: i32,
    /// 通道数。
    pub channel_count: i32,
    /// 总线名（UTF-16，128 个 char16）。
    pub name: [u16; 128],
    /// `Vst::BusTypes`。
    pub bus_type: i32,
    /// `Vst::BusInfo::BusFlags`。
    pub flags: u32,
}

/// `Vst::RoutingInfo`。
#[repr(C)]
pub struct RoutingInfo {
    /// `Vst::MediaTypes`。
    pub media_type: i32,
    /// 总线下标。
    pub bus_index: i32,
    /// 通道（-1 表示全部）。
    pub channel: i32,
}

/// `Vst::ProcessSetup`。
#[repr(C)]
pub struct ProcessSetup {
    /// `Vst::ProcessModes`。
    pub process_mode: i32,
    /// `Vst::SymbolicSampleSizes`。
    pub symbolic_sample_size: i32,
    /// 单块最大样本数。
    pub max_samples_per_block: i32,
    /// 采样率。
    pub sample_rate: f64,
}

/// 把 UTF-8 字符串按 NUL 补齐写入定长 `c_char` 缓冲。
fn copy_str<const N: usize>(dst: &mut [c_char; N], src: &str) {
    for (slot, byte) in dst.iter_mut().zip(src.bytes()) {
        *slot = byte as c_char;
    }
}

/// 把 C 字符串指针按 NUL 补齐写入定长缓冲。
unsafe fn copy_str_ptr<const N: usize>(dst: &mut [c_char; N], src: *const c_char) {
    if src.is_null() {
        return;
    }
    for (index, slot) in dst.iter_mut().enumerate() {
        if index + 1 >= N {
            break;
        }
        // SAFETY: 调用方保证 src 是以 NUL 结尾的可读 C 字符串。
        let byte = unsafe { *src.add(index) };
        if byte == 0 {
            break;
        }
        *slot = byte;
    }
}

// ---------------------------------------------------------------------------
// 工厂对象
// ---------------------------------------------------------------------------

/// `IPluginFactory3` 的超集 vtbl（只 advertise `FUnknown`/`IPluginFactory`/`IPluginFactory2`）。
///
/// 注意：`IPluginFactory` 直接继承自 `FUnknown`（**不是** `IPluginBase`），
/// 因此 release 之后紧接 `getFactoryInfo`，中间没有 `initialize`/`terminate`。
#[repr(C)]
pub struct PluginFactoryVtbl {
    /// `FUnknown::queryInterface`。
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const u8, *mut *mut c_void) -> TResult,
    /// `FUnknown::addRef`。
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    /// `FUnknown::release`。
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    /// `IPluginFactory::getFactoryInfo`。
    pub get_factory_info: unsafe extern "system" fn(*mut c_void, *mut PFactoryInfo) -> TResult,
    /// `IPluginFactory::countClasses`。
    pub count_classes: unsafe extern "system" fn(*mut c_void) -> i32,
    /// `IPluginFactory::getClassInfo`。
    pub get_class_info: unsafe extern "system" fn(*mut c_void, i32, *mut PClassInfo) -> TResult,
    /// `IPluginFactory::createInstance`。
    pub create_instance: unsafe extern "system" fn(
        *mut c_void,
        *const c_char,
        *const c_char,
        *mut *mut c_void,
    ) -> TResult,
    /// `IPluginFactory2::getClassInfo2`。
    pub get_class_info2: unsafe extern "system" fn(*mut c_void, i32, *mut PClassInfo2) -> TResult,
    /// `IPluginFactory3::getClassInfoUnicode`（探针不 adverise 该接口）。
    pub get_class_info_unicode: unsafe extern "system" fn(*mut c_void, i32, *mut c_void) -> TResult,
    /// `IPluginFactory3::setHostContext`（探针不 advertise 该接口）。
    pub set_host_context: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
}

/// 进程级单例工厂对象。
#[repr(C)]
struct PluginFactoryObj {
    vtbl: *const PluginFactoryVtbl,
    refcount: AtomicI32,
}

// SAFETY: 工厂对象只在进程内共享；vtbl 指向常驻静态表，引用计数是原子的。
unsafe impl Send for PluginFactoryObj {}
// SAFETY: 同上，见 Send 说明。
unsafe impl Sync for PluginFactoryObj {}

static FACTORY: PluginFactoryObj = PluginFactoryObj {
    vtbl: &FACTORY_VTBL,
    refcount: AtomicI32::new(1),
};

unsafe extern "system" fn factory_query_interface(
    this: *mut c_void,
    iid: *const u8,
    obj: *mut *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_NO_INTERFACE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_NO_INTERFACE, || {
            if obj.is_null() {
                return K_INVALID_ARGUMENT;
            }
            // SAFETY: obj 是宿主提供的输出槽。
            unsafe { *obj = std::ptr::null_mut() };
            if unsafe { iid_matches(iid, IID_FUNKNOWN) }
                || unsafe { iid_matches(iid, IID_IPLUGIN_FACTORY) }
                || unsafe { iid_matches(iid, IID_IPLUGIN_FACTORY2) }
            {
                let factory = this as *const PluginFactoryObj;
                // SAFETY: this 是形如 PluginFactoryObj 的常驻单例。
                unsafe { (*factory).refcount.fetch_add(1, Ordering::AcqRel) };
                // SAFETY: 输出槽非空。
                unsafe { *obj = this };
                return K_RESULT_OK;
            }
            crate::log_line(&format!("factory queryInterface miss iid={}", unsafe {
                hex_iid(iid)
            }));
            K_NO_INTERFACE
        })
    }
}

unsafe extern "system" fn factory_add_ref(this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || {
            let factory = this as *const PluginFactoryObj;
            // SAFETY: this 是形如 PluginFactoryObj 的常驻单例。
            unsafe { (*factory).refcount.fetch_add(1, Ordering::AcqRel) as u32 + 1 }
        })
    }
}

unsafe extern "system" fn factory_release(this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || {
            let factory = this as *const PluginFactoryObj;
            // SAFETY: this 是形如 PluginFactoryObj 的常驻单例；探针不销毁它。
            let previous = unsafe { (*factory).refcount.fetch_sub(1, Ordering::AcqRel) };
            (previous - 1) as u32
        })
    }
}

unsafe extern "system" fn factory_get_factory_info(
    _this: *mut c_void,
    info: *mut PFactoryInfo,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            if info.is_null() {
                return K_INVALID_ARGUMENT;
            }
            // SAFETY: info 是宿主提供的可写 PFactoryInfo。
            let info = unsafe { &mut *info };
            copy_str(&mut info.vendor, "HiFiShifter");
            copy_str(&mut info.url, "https://github.com/ARounder-183/HiFiShifter");
            copy_str(&mut info.email, "");
            // kUnicode = 1 << 4
            info.flags = 1 << 4;
            K_RESULT_OK
        })
    }
}

unsafe extern "system" fn factory_count_classes(_this: *mut c_void) -> i32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || {
            // 0: 音频处理器组件，1: ARA 主工厂类，2: 编辑器控制器。
            crate::log_line("factory countClasses -> 3");
            3
        })
    }
}

/// 以下两个类共用的静态元数据。
const AUDIO_MODULE_CLASS: &str = "Audio Module Class";

unsafe fn fill_class_info(info: *mut PClassInfo, index: i32) {
    // SAFETY: info 是宿主提供的可写 PClassInfo；先清零再写字段。
    let info = unsafe { &mut *info };
    // SAFETY: 整个结构体是 POD，可直接清零。
    unsafe { std::ptr::write_bytes(info as *mut PClassInfo, 0, 1) };
    info.cardinality = K_MANY_INSTANCES;
    match index {
        0 => {
            info.cid = PROCESSOR_CID;
            copy_str(&mut info.category, AUDIO_MODULE_CLASS);
            copy_str(&mut info.name, crate::CLASS_NAME);
        }
        2 => {
            info.cid = CONTROLLER_CID;
            copy_str(&mut info.category, COMPONENT_CONTROLLER_CLASS);
            copy_str(&mut info.name, crate::CLASS_NAME);
        }
        1 => {
            info.cid = FACTORY_CID;
            // SAFETY: 适配器返回常驻 C 字符串（或 null）。
            unsafe { copy_str_ptr(&mut info.category, ara2_vst3_main_factory_category()) };
            if info.category[0] == 0 {
                copy_str(&mut info.category, "ARA Main Factory Class");
            }
            copy_str(&mut info.name, crate::CLASS_NAME);
        }
        _ => {}
    }
}

unsafe extern "system" fn factory_get_class_info(
    _this: *mut c_void,
    index: i32,
    info: *mut PClassInfo,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            if info.is_null() || !(0..=2).contains(&index) {
                return K_INVALID_ARGUMENT;
            }
            // SAFETY: info 是宿主提供的可写 PClassInfo。
            unsafe { fill_class_info(info, index) };
            crate::log_line(&format!("factory getClassInfo({index}) ok"));
            K_RESULT_OK
        })
    }
}

unsafe extern "system" fn factory_get_class_info2(
    _this: *mut c_void,
    index: i32,
    info: *mut PClassInfo2,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            if info.is_null() || !(0..=2).contains(&index) {
                return K_INVALID_ARGUMENT;
            }
            // SAFETY: info 是宿主提供的可写 PClassInfo2。
            let info = unsafe { &mut *info };
            // SAFETY: POD 清零。
            unsafe { std::ptr::write_bytes(info as *mut PClassInfo2, 0, 1) };
            info.cardinality = K_MANY_INSTANCES;
            match index {
                0 => {
                    info.cid = PROCESSOR_CID;
                    copy_str(&mut info.category, AUDIO_MODULE_CLASS);
                    copy_str(&mut info.name, crate::CLASS_NAME);
                    // 二期编辑会话在宿主本进程，不宣称跨进程processor/controller分布。
                    info.class_flags = 0;
                    copy_str(&mut info.sub_categories, "Fx|OnlyARA");
                    copy_str(&mut info.vendor, "HiFiShifter");
                    copy_str(&mut info.version, crate::VERSION);
                    copy_str(&mut info.sdk_version, "VST 3.8.0");
                }
                2 => {
                    info.cid = CONTROLLER_CID;
                    copy_str(&mut info.category, COMPONENT_CONTROLLER_CLASS);
                    copy_str(&mut info.name, crate::CLASS_NAME);
                    copy_str(&mut info.vendor, "HiFiShifter");
                    copy_str(&mut info.version, crate::VERSION);
                    copy_str(&mut info.sdk_version, "VST 3.8.0");
                }
                1 => {
                    info.cid = FACTORY_CID;
                    // SAFETY: 适配器返回常驻 C 字符串（或 null）。
                    unsafe { copy_str_ptr(&mut info.category, ara2_vst3_main_factory_category()) };
                    if info.category[0] == 0 {
                        copy_str(&mut info.category, "ARA Main Factory Class");
                    }
                    copy_str(&mut info.name, crate::CLASS_NAME);
                    copy_str(&mut info.vendor, "HiFiShifter");
                    copy_str(&mut info.version, crate::VERSION);
                    copy_str(&mut info.sdk_version, "VST 3.8.0");
                }
                _ => {}
            }
            K_RESULT_OK
        })
    }
}

unsafe extern "system" fn factory_get_class_info_unicode(
    _this: *mut c_void,
    _index: i32,
    _info: *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || K_NOT_IMPLEMENTED)
    }
}

unsafe extern "system" fn factory_set_host_context(
    _this: *mut c_void,
    _context: *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || K_RESULT_OK)
    }
}

unsafe extern "system" fn factory_create_instance(
    _this: *mut c_void,
    cid: *const c_char,
    iid: *const c_char,
    obj: *mut *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            if obj.is_null() {
                return K_INVALID_ARGUMENT;
            }
            // SAFETY: obj 是宿主提供的输出槽。
            unsafe { *obj = std::ptr::null_mut() };
            if cid.is_null() || iid.is_null() {
                return K_INVALID_ARGUMENT;
            }
            // SAFETY: cid 指向宿主从 getClassInfo 拿到的 16 字节 TUID。
            let cid = unsafe { std::slice::from_raw_parts(cid as *const u8, 16) };
            crate::log_line(&format!(
                "factory createInstance cid={} iid={}",
                hex16(cid),
                unsafe { hex_iid(iid as *const u8) }
            ));
            if cid == PROCESSOR_CID {
                let Some(processor) = Processor::create() else {
                    return K_NO_INTERFACE;
                };
                let base = Box::into_raw(processor) as *mut c_void;
                // SAFETY: base 是新分配的 Processor，iid/obj 来自宿主。
                let result = unsafe { component_query_interface(base, iid as *const u8, obj) };
                if result != K_RESULT_OK {
                    // 宿主没有要任何我们实现的接口：按 VST3 约定销毁刚创建的对象。
                    // SAFETY: 引用计数仍为 1，且未把指针交给宿主。
                    unsafe { drop(Box::from_raw(base as *mut Processor)) };
                } else {
                    // queryInterface 已交出一个宿主引用；工厂必须消耗自己的初始引用。
                    // SAFETY: 宿主 owning 引用仍保留对象，此处仅减少工厂的那一份。
                    unsafe { processor_release(base as *mut Processor) };
                }
                return result;
            }
            if cid == FACTORY_CID {
                let Some(adapter) = new_main_factory_adapter() else {
                    return K_NO_INTERFACE;
                };
                // 宿主会按它请求的 IID（ARA::IMainFactory 或 FUnknown）使用该对象，
                // 适配器自身的 queryInterface 已支持两者，因此直接把主接口交出去。
                let raw = adapter.as_raw();
                // 探针把适配器泄漏给宿主管理引用计数，直到进程结束。
                std::mem::forget(adapter);
                // SAFETY: obj 是宿主提供的输出槽。
                unsafe { *obj = raw };
                return K_RESULT_OK;
            }
            if cid == CONTROLLER_CID {
                let controller = Box::into_raw(Box::new(EditController::new())) as *mut c_void;
                // SAFETY: controller 是新分配的控制器对象。
                let result =
                    unsafe { edit_controller_query_interface(controller, iid as *const u8, obj) };
                if result != K_RESULT_OK {
                    // 宿主没有要任何我们实现的接口：按 VST3 约定销毁刚创建的对象。
                    // SAFETY: 引用计数仍为 1，且未把指针交给宿主。
                    unsafe { drop(Box::from_raw(controller as *mut EditController)) };
                } else {
                    // queryInterface 已交出一个宿主引用；工厂必须消耗自己的初始引用。
                    // 此前漏掉这一步：每交付一个控制器实例就永久泄漏一份
                    // ConnectionState（含 Arc 与两个 Mutex），宿主反复插拔 FX 时慢性增长。
                    // SAFETY: 宿主 owning 引用仍保留对象，此处仅减少工厂的那一份。
                    unsafe { edit_controller_release(controller) };
                }
                return result;
            }
            K_INVALID_ARGUMENT
        })
    }
}

static FACTORY_VTBL: PluginFactoryVtbl = PluginFactoryVtbl {
    query_interface: factory_query_interface,
    add_ref: factory_add_ref,
    release: factory_release,
    get_factory_info: factory_get_factory_info,
    count_classes: factory_count_classes,
    get_class_info: factory_get_class_info,
    create_instance: factory_create_instance,
    get_class_info2: factory_get_class_info2,
    get_class_info_unicode: factory_get_class_info_unicode,
    set_host_context: factory_set_host_context,
};

/// 返回进程级单例工厂指针（每次调用增加一个引用，符合 VST3 约定）。
pub fn get_plugin_factory() -> *mut c_void {
    crate::log_line("GetPluginFactory called");
    FACTORY.refcount.fetch_add(1, Ordering::AcqRel);
    &FACTORY as *const PluginFactoryObj as *mut c_void
}

// ---------------------------------------------------------------------------
// 处理器组件对象（IComponent + IAudioProcessor + ARA 入口点转发）
// ---------------------------------------------------------------------------

/// `IComponent` 的 vtbl（含继承自 `IPluginBase` / `FUnknown` 的槽位）。
#[repr(C)]
pub struct ComponentVtbl {
    /// `FUnknown::queryInterface`。
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const u8, *mut *mut c_void) -> TResult,
    /// `FUnknown::addRef`。
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    /// `FUnknown::release`。
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    /// `IPluginBase::initialize`。
    pub initialize: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    /// `IPluginBase::terminate`。
    pub terminate: unsafe extern "system" fn(*mut c_void) -> TResult,
    /// `IComponent::getControllerClassId`。
    pub get_controller_class_id: unsafe extern "system" fn(*mut c_void, *mut u8) -> TResult,
    /// `IComponent::setIoMode`。
    pub set_io_mode: unsafe extern "system" fn(*mut c_void, i32) -> TResult,
    /// `IComponent::getBusCount`。
    pub get_bus_count: unsafe extern "system" fn(*mut c_void, i32, i32) -> i32,
    /// `IComponent::getBusInfo`。
    pub get_bus_info:
        unsafe extern "system" fn(*mut c_void, i32, i32, i32, *mut BusInfo) -> TResult,
    /// `IComponent::getRoutingInfo`。
    pub get_routing_info:
        unsafe extern "system" fn(*mut c_void, *mut RoutingInfo, *mut RoutingInfo) -> TResult,
    /// `IComponent::activateBus`。
    pub activate_bus: unsafe extern "system" fn(*mut c_void, i32, i32, i32, i8) -> TResult,
    /// `IComponent::setActive`。
    pub set_active: unsafe extern "system" fn(*mut c_void, i8) -> TResult,
    /// `IComponent::setState`。
    pub set_state: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    /// `IComponent::getState`。
    pub get_state: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
}

/// `IAudioProcessor` 的 vtbl（独立的子对象，避免与 `IComponent` 槽位冲突）。
#[repr(C)]
pub struct AudioProcessorVtbl {
    /// `FUnknown::queryInterface`。
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const u8, *mut *mut c_void) -> TResult,
    /// `FUnknown::addRef`。
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    /// `FUnknown::release`。
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    /// `IAudioProcessor::setBusArrangements`。
    pub set_bus_arrangements:
        unsafe extern "system" fn(*mut c_void, *mut u64, i32, *mut u64, i32) -> TResult,
    /// `IAudioProcessor::getBusArrangement`。
    pub get_bus_arrangement: unsafe extern "system" fn(*mut c_void, i32, i32, *mut u64) -> TResult,
    /// `IAudioProcessor::canProcessSampleSize`。
    pub can_process_sample_size: unsafe extern "system" fn(*mut c_void, i32) -> TResult,
    /// `IAudioProcessor::getLatencySamples`。
    pub get_latency_samples: unsafe extern "system" fn(*mut c_void) -> u32,
    /// `IAudioProcessor::setupProcessing`。
    pub setup_processing: unsafe extern "system" fn(*mut c_void, *mut ProcessSetup) -> TResult,
    /// `IAudioProcessor::setProcessing`。
    pub set_processing: unsafe extern "system" fn(*mut c_void, i8) -> TResult,
    /// `IAudioProcessor::process`。
    pub process: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    /// `IAudioProcessor::getTailSamples`。
    pub get_tail_samples: unsafe extern "system" fn(*mut c_void) -> u32,
}

/// 处理器对象。
///
/// 前三个字段（组件 vtbl 指针、共享引用计数、处理器子对象 vtbl 指针）是布局契约：
/// `IComponent*` 指向对象首地址，`IAudioProcessor*` 指向 `audio_vtbl` 字段地址。
/// 其余字段只被 Rust 侧使用。
#[repr(C)]
struct Processor {
    component_vtbl: *const ComponentVtbl,
    refcount: AtomicI32,
    audio_vtbl: *const AudioProcessorVtbl,
    entry: Option<HostEntry>,
    extension_owner: Arc<ExtensionOwner>,
    active: AtomicI32,
    connection_vtbl: *const crate::editor::connection::ConnectionVtbl,
    connection: crate::editor::connection::ConnectionState,
    route: crate::editor::routing::RouteLease,
}

impl Processor {
    /// 建立处理器：为每个实例单独建一个 companion 处理器绑定 + ARA 入口适配器，
    /// 这样多个实例不会互相抢同一次绑定。
    fn create() -> Option<Box<Processor>> {
        let runtime = crate::runtime::runtime()?;
        let binding =
            CompanionProcessorBinding::new([runtime.companion.clone()], CompanionRoles::all())
                .ok()?;
        let extension_owner = Arc::new(ExtensionOwner::default());
        let route = crate::editor::routing::RouteLease::new(&extension_owner);
        let entry = HostEntry::new(binding, crate::CLASS_NAME, extension_owner.clone()).ok()?;
        Some(Box::new(Processor {
            component_vtbl: &COMPONENT_VTBL,
            refcount: AtomicI32::new(1),
            audio_vtbl: &AUDIO_VTBL,
            entry: Some(entry),
            extension_owner,
            active: AtomicI32::new(0),
            connection_vtbl: &PROCESSOR_CONNECTION_VTBL,
            connection: Default::default(),
            route,
        }))
    }
}

/// 处理器子对象相对对象基址的偏移。
fn audio_offset() -> usize {
    offset_of!(Processor, audio_vtbl)
}

/// 从 `IAudioProcessor*` 回到对象基址。
unsafe fn base_from_audio(audio: *mut c_void) -> *mut Processor {
    // SAFETY: audio 由 audio_ptr 生成，指向 Processor 的 audio_vtbl 字段。
    unsafe { (audio as *mut u8).sub(audio_offset()) as *mut Processor }
}

/// 取 `IAudioProcessor*`（即 audio_vtbl 字段地址）。
unsafe fn audio_ptr(processor: *mut Processor) -> *mut c_void {
    // SAFETY: processor 是活动对象，addr_of 不产生引用。
    unsafe { std::ptr::addr_of!((*processor).audio_vtbl) as *mut c_void }
}

/// 递增共享引用计数。
unsafe fn processor_add_ref(processor: *mut Processor) -> u32 {
    // SAFETY: processor 是活动对象。
    unsafe { (*processor).refcount.fetch_add(1, Ordering::AcqRel) as u32 + 1 }
}

/// 递减共享引用计数，归零时销毁对象。
unsafe fn processor_release(processor: *mut Processor) -> u32 {
    // SAFETY: processor 是活动对象。
    let previous = unsafe { (*processor).refcount.fetch_sub(1, Ordering::AcqRel) };
    let remaining = previous - 1;
    if remaining == 0 {
        unsafe {
            (*processor).extension_owner.stop_channel();
        }
        unsafe {
            (*processor).extension_owner.stop_editor();
        }
        // SAFETY: 引用计数归零，且没有其它持有者。
        unsafe { drop(Box::from_raw(processor)) };
    }
    remaining as u32
}

unsafe extern "system" fn component_query_interface(
    this: *mut c_void,
    iid: *const u8,
    obj: *mut *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_NO_INTERFACE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_NO_INTERFACE, || {
            if obj.is_null() {
                return K_INVALID_ARGUMENT;
            }
            // SAFETY: obj 是宿主提供的输出槽。
            unsafe { *obj = std::ptr::null_mut() };
            let processor = this as *mut Processor;
            if unsafe { crate::editor::connection::is_connection(iid) } {
                unsafe {
                    processor_add_ref(processor);
                    *obj = std::ptr::addr_of_mut!((*processor).connection_vtbl).cast();
                }
                return K_RESULT_OK;
            }
            // FUnknown / IPluginBase / IComponent 共用对象基址。
            if unsafe { iid_matches(iid, IID_FUNKNOWN) }
                || unsafe { iid_matches(iid, IID_IPLUGIN_BASE) }
                || unsafe { iid_matches(iid, IID_ICOMPONENT) }
            {
                if unsafe { iid_matches(iid, IID_ICOMPONENT) } {
                    crate::log_line("component queryInterface hit IComponent");
                }
                // SAFETY: processor 是活动对象。
                unsafe { processor_add_ref(processor) };
                // SAFETY: 输出槽非空。
                unsafe { *obj = this };
                return K_RESULT_OK;
            }
            if unsafe { iid_matches(iid, IID_IAUDIO_PROCESSOR) } {
                crate::log_line("component queryInterface hit IAudioProcessor");
                // SAFETY: processor 是活动对象。
                unsafe { processor_add_ref(processor) };
                // SAFETY: 输出槽非空。
                unsafe { *obj = audio_ptr(processor) };
                return K_RESULT_OK;
            }
            // ARA 入口点：转发给 companion 适配器，由它做 IID 匹配与引用计数。
            // 关键：必须先按宿主**实际请求的 IID** 判断，不能无条件转发 ——
            // companion 的 query_interface(kind) 是按给定 kind 去问"你支持这个 ARA 接口吗"，
            // 它永远会成功。若宿主问的是别的接口（实测会问 IEditController），
            // 无条件返回 ARA 指针会让宿主用错误的 vtable 调用方法而崩溃。
            // SAFETY: processor 是活动对象；entry 在对象生命周期内有效。
            if let Some(entry) = unsafe { (*processor).entry.as_ref() } {
                for kind in [
                    Ara2Vst3InterfaceKind::PluginEntry,
                    Ara2Vst3InterfaceKind::PluginEntry2,
                ] {
                    let Some(words) = ara_iid_words(kind) else {
                        continue;
                    };
                    if unsafe { iid_matches(iid, words) } {
                        if let Ok(interface) = entry.query_interface(kind) {
                            crate::log_line(&format!(
                                "component queryInterface ARA hit iid={}",
                                unsafe { hex_iid(iid) }
                            ));
                            // SAFETY: 输出槽非空。
                            unsafe { *obj = interface };
                            return K_RESULT_OK;
                        }
                    }
                }
            }
            crate::log_line(&format!("component queryInterface miss iid={}", unsafe {
                hex_iid(iid)
            }));
            K_NO_INTERFACE
        })
    }
}

unsafe extern "system" fn component_add_ref(this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || {
            // SAFETY: this 是组件接口指针（对象基址）。
            unsafe { processor_add_ref(this as *mut Processor) }
        })
    }
}

unsafe extern "system" fn component_release(this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || {
            // SAFETY: this 是组件接口指针（对象基址）。
            unsafe { processor_release(this as *mut Processor) }
        })
    }
}

unsafe extern "system" fn component_initialize(this: *mut c_void, context: *mut c_void) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            // QI/GetApi可重入释放宿主组件引用；短引用只保活本次调用的插件存储，不保活project/take。
            struct CallReference(*mut c_void);
            impl Drop for CallReference {
                fn drop(&mut self) {
                    unsafe {
                        component_release(self.0);
                    }
                }
            }
            unsafe {
                component_add_ref(this);
            }
            let _call_reference = CallReference(this);
            let owner = unsafe { (*this.cast::<Processor>()).extension_owner.clone() };
            if owner.is_closed() {
                return K_RESULT_FALSE;
            }
            unsafe {
                (*(this as *mut Processor)).connection.initialize(context);
            }
            if owner.is_closed() {
                // QI可能先重入close再返回新引用；拒绝初始化时不能把该引用留在closed connection。
                unsafe {
                    (*this.cast::<Processor>()).connection.close();
                }
                return K_RESULT_FALSE;
            }
            unsafe {
                owner.bind_host_adapter(context);
            }
            if owner.is_closed() {
                unsafe {
                    (*this.cast::<Processor>()).connection.close();
                }
                return K_RESULT_FALSE;
            }
            // 走到这里说明组件没有在 QI/bind 期间被重入关闭；此时才读一次宿主应用名
            // （只用于诊断，见 `connection::host_application_name`）。
            unsafe {
                (*this.cast::<Processor>()).connection.record_host_name();
            }
            crate::log_line("IComponent::initialize");
            K_RESULT_OK
        })
    }
}

unsafe extern "system" fn component_terminate(this: *mut c_void) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            unsafe {
                let processor = &*(this as *mut Processor);
                processor.connection.close();
                processor.extension_owner.stop_editor();
            }
            crate::log_line("IComponent::terminate");
            K_RESULT_OK
        })
    }
}

unsafe extern "system" fn component_get_controller_class_id(
    _this: *mut c_void,
    class_id: *mut u8,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            if class_id.is_null() {
                return K_INVALID_ARGUMENT;
            }
            // 探针提供一个最小编辑器控制器：REAPER 需要它才能完成 ARA 插入。
            // SAFETY: class_id 是宿主提供的可写 TUID。
            unsafe { std::ptr::copy_nonoverlapping(CONTROLLER_CID.as_ptr(), class_id, 16) };
            crate::log_line("IComponent::getControllerClassId -> HFSARAProbe.Ctl1");
            K_RESULT_OK
        })
    }
}

unsafe extern "system" fn component_set_io_mode(_this: *mut c_void, _mode: i32) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || K_RESULT_OK)
    }
}

unsafe extern "system" fn component_get_bus_count(
    _this: *mut c_void,
    media_type: i32,
    _direction: i32,
) -> i32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || if media_type == 0 { 1 } else { 0 })
    }
}

unsafe extern "system" fn component_get_bus_info(
    _this: *mut c_void,
    media_type: i32,
    direction: i32,
    index: i32,
    bus: *mut BusInfo,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            if bus.is_null() || media_type != 0 || index != 0 {
                return K_INVALID_ARGUMENT;
            }
            // SAFETY: bus 是宿主提供的可写 BusInfo。
            let bus = unsafe { &mut *bus };
            // SAFETY: POD 清零。
            unsafe { std::ptr::write_bytes(bus as *mut BusInfo, 0, 1) };
            bus.media_type = 0;
            bus.direction = direction;
            // 如实报告**协商结果**：`setBusArrangements` 接受宿主给出的任意扬声器布局
            // （值是位掩码：kMono=1、kStereo=3、5.1=63…），声道数即置位数。此前
            // `1 => 1, _ => 2` 的映射只认 mono/stereo —— 宿主协商 5.1 时
            // `getBusArrangement` 回 6、这里却报 2，注释与两处实现互相矛盾。渲染面已
            // 泛化到任意声道数（`copy_block` 只处理前两个声道，`audio_abi::clear_outputs`
            // 按 `num_channels` 清零其余），所以这里照实报即可。未协商时
            // `bus_arrangement` 默认 kStereo(3) ⇒ 2。
            bus.channel_count = if _this.is_null() {
                2
            } else {
                // SAFETY: `_this` 是存活 component（即 `Processor`）。
                unsafe {
                    (*_this.cast::<Processor>())
                        .extension_owner
                        .bus_arrangement(direction)
                }
                .count_ones() as i32
            };
            bus.bus_type = 0;
            bus.flags = 1; // kDefaultActive
            bus.name[0] = b'M' as u16;
            K_RESULT_OK
        })
    }
}

unsafe extern "system" fn component_get_routing_info(
    _this: *mut c_void,
    _in_info: *mut RoutingInfo,
    _out_info: *mut RoutingInfo,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || K_NOT_IMPLEMENTED)
    }
}

unsafe extern "system" fn component_activate_bus(
    _this: *mut c_void,
    _media_type: i32,
    _direction: i32,
    _index: i32,
    _state: i8,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || K_RESULT_OK)
    }
}

unsafe extern "system" fn component_set_active(this: *mut c_void, state: i8) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            let processor = this as *mut Processor;
            // SAFETY: processor 是活动对象。
            unsafe { (*processor).active.store(state as i32, Ordering::SeqCst) };
            crate::log_line(&format!("IComponent::setActive({state})"));
            K_RESULT_OK
        })
    }
}

unsafe extern "system" fn component_set_state(this: *mut c_void, state: *mut c_void) -> TResult {
    if this.is_null() {
        return K_INVALID_ARGUMENT;
    }
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let bytes = unsafe { crate::state_stream::read_state(state) }?;
        let owner = unsafe { &(*this.cast::<Processor>()).extension_owner };
        owner.restore_state(&bytes)?;
        Ok::<_, String>(())
    }));
    if matches!(outcome, Ok(Ok(()))) {
        K_RESULT_OK
    } else {
        K_RESULT_FALSE
    }
}

unsafe extern "system" fn component_get_state(this: *mut c_void, state: *mut c_void) -> TResult {
    if this.is_null() {
        return K_INVALID_ARGUMENT;
    }
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let owner = unsafe { &(*this.cast::<Processor>()).extension_owner };
        let bytes = owner.encode_state()?;
        unsafe { crate::state_stream::write_state(state, &bytes) }
    }));
    if matches!(outcome, Ok(Ok(()))) {
        K_RESULT_OK
    } else {
        K_RESULT_FALSE
    }
}

unsafe extern "system" fn audio_query_interface(
    this: *mut c_void,
    iid: *const u8,
    obj: *mut *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_NO_INTERFACE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_NO_INTERFACE, || {
            // SAFETY: this 是 IAudioProcessor 子对象指针。
            let base = unsafe { base_from_audio(this) } as *mut c_void;
            // SAFETY: 转发到基址查询。
            unsafe { component_query_interface(base, iid, obj) }
        })
    }
}

unsafe extern "system" fn audio_add_ref(this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || {
            // SAFETY: this 是 IAudioProcessor 子对象指针。
            unsafe { processor_add_ref(base_from_audio(this)) }
        })
    }
}

unsafe extern "system" fn audio_release(this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || {
            // SAFETY: this 是 IAudioProcessor 子对象指针。
            unsafe { processor_release(base_from_audio(this)) }
        })
    }
}

unsafe extern "system" fn audio_set_bus_arrangements(
    this: *mut c_void,
    inputs: *mut u64,
    num_inputs: i32,
    outputs: *mut u64,
    num_outputs: i32,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            if num_inputs != 1 || num_outputs != 1 {
                return K_RESULT_FALSE;
            }
            if this.is_null() || inputs.is_null() || outputs.is_null() {
                return K_INVALID_ARGUMENT;
            }
            // SAFETY: 宿主按数量提供合法布局数组；本批声明一进一出。
            // 【为什么接受任意布局】此前只认 `kMono`(1) 与 `kStereo`(3)，而多数宿主在
            // `setBusArrangements` 返回 `kResultFalse` 时会把插件**踢出插槽**（用户看到
            // "插件不见了"，且没有提示）。5.1 / 7.1 / 任意环绕轨道因此完全不可用。
            // 现在的取舍：接受宿主给的布局、内部只处理前两个声道、其余保持静音 ——
            // 宁可部分工作，也不要从插槽里消失（渲染面见 `PlaybackSnapshot::copy_block`，
            // 清零点见 `audio_abi::clear_outputs`）。
            let (input, output) = unsafe { (*inputs, *outputs) };
            if input == 0 || output == 0 {
                return K_RESULT_FALSE;
            }
            // 记下协商结果：`getBusArrangement` 必须如实回吐，否则宿主按错误声道数组织缓冲。
            // SAFETY: this 为存活音频接口。
            unsafe { &(*base_from_audio(this)).extension_owner }
                .set_bus_arrangements(input, output);
            K_RESULT_OK
        })
    }
}

unsafe extern "system" fn audio_get_bus_arrangement(
    this: *mut c_void,
    direction: i32,
    _index: i32,
    arrangement: *mut u64,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            if this.is_null() || arrangement.is_null() {
                return K_INVALID_ARGUMENT;
            }
            // SAFETY: this 为存活音频接口；arrangement 是宿主提供的可写 SpeakerArrangement。
            // 回吐**协商结果**（未协商时默认 kStereo），不再无条件写死 kStereo。
            let negotiated =
                unsafe { &(*base_from_audio(this)).extension_owner }.bus_arrangement(direction);
            unsafe { *arrangement = negotiated };
            K_RESULT_OK
        })
    }
}

unsafe extern "system" fn audio_can_process_sample_size(
    _this: *mut c_void,
    symbolic_sample_size: i32,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            if symbolic_sample_size == 0 {
                K_RESULT_OK
            } else {
                K_RESULT_FALSE
            }
        })
    }
}

unsafe extern "system" fn audio_get_latency_samples(_this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || 0)
    }
}

unsafe extern "system" fn audio_setup_processing(
    this: *mut c_void,
    setup: *mut ProcessSetup,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            if this.is_null() || setup.is_null() {
                return K_INVALID_ARGUMENT;
            }
            // SAFETY: 宿主提供调用期间可读的 SDK setup 结构。
            let setup = unsafe { &*setup };
            if !setup.sample_rate.is_finite()
                || setup.sample_rate <= 0.0
                || setup.max_samples_per_block <= 0
                || !(0..=2).contains(&setup.process_mode)
            {
                return K_INVALID_ARGUMENT;
            }
            // 【为什么不再按采样率拒绝】此前只认 44.1/48k，其余 `kResultFalse` —— 而多数宿主
            // 收到 `kResultFalse` 会把插件**整条踢出链路**：96k/192k 工程里用户看到"插件不见了"，
            // 没有任何提示。内部渲染仍按 44.1/48k 出快照，其余率由
            // `PlaybackSnapshot::copy_block` 做线性重采样（降级、可听、有日志）。真正的多率
            // 渲染管线是独立一期工作；**先保证插件在场**。
            if setup.symbolic_sample_size != 0 {
                // 【为什么返回 OK 而不是 `kResultFalse`】`kResultFalse` 会让多数宿主把插件
                // **整条踢出链路**（用户看到"插件不见了"）。我们读不了 f64 平面，但可以在
                // `process` 里按实际宽度静音（`silence_unsupported_sample_size`）—— 静音但在场
                // 是可诊断的降级，优于静默消失。
                crate::log_line(&format!(
                "[vst3] setupProcessing: sampleSize={} is not 32-bit float; the plugin stays in the \
                 chain and outputs silence (see silence_unsupported_sample_size).",
                setup.symbolic_sample_size
            ));
                return K_RESULT_OK;
            }
            // 【接受域与 `render::source` 对齐（8k..=384k）】此前下限是 44.1k，理由是"内部快照
            // 永不降采样"。但宿主率低于 44.1k 时返回 `kResultFalse` 会让多数宿主把插件**整条
            // 踢出链路**（用户看到"插件不见了"，与 `symbolic_sample_size` 同一类失败）。源读取域
            // 本就接受 8k（`render/source.rs`），这里与它对齐：接受后用
            // `PlaybackSnapshot::copy_block` 的线性重采样出音。**低于 44.1k 时是降采样**，线性
            // 插值没有抗混叠滤波 ⇒ 有明显镜像，所以单独记一条警告 —— 可诊断的降级优于静默消失。
            // 真正按宿主率渲染内部快照（消除这次降采样）是独立一期工作。
            if !(8000.0..=384000.0).contains(&setup.sample_rate) {
                crate::log_line(&format!(
                    "[vst3] setupProcessing refused: sampleRate={} is outside 8000..=384000",
                    setup.sample_rate
                ));
                return K_RESULT_FALSE;
            }
            if (setup.sample_rate - 44100.0).abs() > 1e-6
                && (setup.sample_rate - 48000.0).abs() > 1e-6
            {
                let aliasing = if setup.sample_rate < 44100.0 {
                    " The host rate is below the internal snapshot rate, so this is a downsample with \
                 no anti-aliasing filter; expect audible imaging."
                } else {
                    ""
                };
                crate::log_line(&format!(
                "[vst3] sampleRate={} is not natively rendered; output will be linearly resampled \
                 from the internal 44.1/48k snapshots.{aliasing}",
                setup.sample_rate
            ));
            }
            // 记下权威采样率：`process` 用它（而非 `ProcessContext::sample_rate`）选快照，
            // 保证"setup 接受过的率"在音频线程上不会再被拒（见 `ExtensionOwner::setup_sample_rate`）。
            // SAFETY: this 为存活音频接口。
            unsafe { &(*base_from_audio(this)).extension_owner }
                .set_setup_sample_rate(setup.sample_rate);
            crate::log_line(&format!(
                "IAudioProcessor::setupProcessing mode={} rate={}",
                setup.process_mode, setup.sample_rate
            ));
            if setup.process_mode == 2 {
                // SAFETY: this为存活音频接口；SDK明确setup在UI线程/禁用状态调用。
                let owner = unsafe { &(*base_from_audio(this)).extension_owner };
                // 【为什么在这里同步等待】离线渲染会把当时听不到的声音烘焙进产物：等得不够久
                // ⇒ 产物里是静音，这是数据正确性失败。等待时长与理由见 `OFFLINE_PREPARE_TIMEOUT`。
                //
                // 【为什么失败返回 `kResultOk` 而不是 `kResultFalse`】`kResultFalse` 会让多数
                // 宿主把插件**整个踢出链路**（用户看到"插件不见了"，没有任何提示）。返回 OK 则
                // 保留插件：离线渲染输出静音，且日志里有一条明确的原因 —— 可诊断的降级优于
                // 静默的消失。
                let started = std::time::Instant::now();
                if let Err(error) = owner.prepare_offline_until(
                    started + crate::render::extension::OFFLINE_PREPARE_TIMEOUT,
                ) {
                    crate::log_line(&format!(
                    "Offline preparation failed after {:?}: {error}. Keeping the plugin in the chain; \
                     offline output will be silent until preparation succeeds.",
                    started.elapsed()
                ));
                    return K_RESULT_OK;
                }
            }
            K_RESULT_OK
        })
    }
}

unsafe extern "system" fn audio_set_processing(this: *mut c_void, state: i8) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            // SDK 允许从音频线程调用此函数；禁止触发同步文件日志。
            if this.is_null() {
                return K_INVALID_ARGUMENT;
            }
            if state == 0 {
                unsafe { &(*base_from_audio(this)).extension_owner }.record_processing_stop();
            }
            K_RESULT_OK
        })
    }
}

/// 音频回调里被 `catch_unwind` 拦下的 panic 次数。
///
/// 【为什么是原子计数而不是日志】音频线程不能做文件 IO（见
/// `realtime_callbacks_do_not_enter_the_file_logger`），而 panic 现场又必须留下痕迹。
/// 计一次数，由下一次 UI 轮询/诊断包落盘。
static AUDIO_PANIC_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 音频回调被拦下的 panic 次数（诊断用）。
pub(crate) fn audio_panic_count() -> u64 {
    AUDIO_PANIC_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

/// 宿主音频回调：完整渲染调度 + PCM 快照输出都在 `audio_process_inner` 里完成；
/// 本层只负责 panic 遏制。
unsafe extern "system" fn audio_process(this: *mut c_void, data: *mut c_void) -> TResult {
    // 【为什么必须 catch_unwind】本回调由**宿主音频线程**调用，panic 越过
    // `extern "system"` 边界是 UB，而且会带走整个 DAW 会话。`component_set_state` /
    // `component_get_state` 早就包了这一层，音频路径此前没有 —— 一条最不该崩的路径
    // 反而最没有保护。捕获后输出静音（不留上一次的残留），并只做一次原子计数：
    // 音频线程不得写文件、不得加锁等待。
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        audio_process_inner(this, data)
    }));
    match outcome {
        Ok(result) => result,
        Err(_) => {
            AUDIO_PANIC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // SAFETY: 与 audio_process_inner 同一份宿主提供的 ProcessData；清空输出
            // 失败（形状非法）也只能返回 OK —— 此时输出交给宿主处置，不能二次 panic。
            let _ = unsafe { crate::audio_abi::clear_outputs(data.cast()) };
            K_RESULT_OK
        }
    }
}

unsafe fn audio_process_inner(this: *mut c_void, data: *mut c_void) -> TResult {
    // SAFETY: VST3 宿主按 SDK 布局提供回调数据，函数内部处理空指针与非法字段。
    if this.is_null() {
        return K_INVALID_ARGUMENT;
    }
    match unsafe { crate::audio_abi::validate(data.cast()) } {
        Ok(()) => {
            // SAFETY: 宿主持有 Processor，validate已验证当前数据和音频总线。
            let data = unsafe { &mut *data.cast::<crate::audio_abi::ProcessData>() };
            if data.num_samples == 0 || data.num_outputs == 0 {
                return K_RESULT_OK;
            }
            // SAFETY: this 是存活的音频接口子对象。
            let owner = unsafe { &(*base_from_audio(this)).extension_owner };
            if !data.process_context.is_null() {
                owner.observe_transport(unsafe { &*data.process_context }, data.process_mode);
            }
            if owner.is_editor_only() {
                // 纯编辑角色不是歌曲播放renderer；保留宿主已经处理的fade/gain/前级FX，包括停播监听。
                return match unsafe { crate::audio_abi::pass_through(data) } {
                    Ok(()) => K_RESULT_OK,
                    Err(crate::audio_abi::BufferError::InvalidArgument) => K_INVALID_ARGUMENT,
                    Err(crate::audio_abi::BufferError::UnsupportedFormat) => {
                        if unsafe { crate::audio_abi::silence_unsupported_sample_size(data) } {
                            K_RESULT_OK
                        } else {
                            K_RESULT_FALSE
                        }
                    }
                };
            }
            // 【为什么不用 expect】这里在**音频线程**上，panic 会跨越 VST3 的 FFI 边界
            // 展开进 REAPER 的音频回调。`validate` 已经查过布局，但校验与使用之间若出现
            // 任何不一致（例如宿主给了我们没预料到的总线形状），宁可返回错误码让宿主把
            // 这一帧当静音，也不能让整个 DAW 崩掉。
            if unsafe { crate::audio_abi::clear_outputs(data) }.is_err() {
                return K_RESULT_FALSE;
            }
            if owner.host_item_muted() {
                return K_RESULT_OK;
            }
            if data.process_context.is_null() {
                owner.snapshots[0].misses.fetch_add(1, Ordering::Relaxed);
                return if data.process_mode == 2 {
                    K_RESULT_FALSE
                } else {
                    K_RESULT_OK
                };
            }
            // SAFETY: VST3 processContext 的完整 SDK 结构在当前回调期间存活。
            let context = unsafe { &*data.process_context };
            // REAPER停播也会process固定光标位置；只离线导出允许没有kPlaying的供音。
            if data.process_mode != 2 && context.state & (1 << 1) == 0 {
                return K_RESULT_OK;
            }
            // 按 `setupProcessing` 的**权威率**选快照：非原生率（96k/192k…）取最近的已声明
            // 原生快照，`copy_block` 会把它线性重采样到宿主输出率。此前这里用
            // `context.sample_rate` 且只认 44.1/48k，其余 `kResultFalse` —— 与 setup 的接受域
            // 不一致，96k 工程里会持续静音甚至被踢出链路（见 `setup_sample_rate`）。
            let publisher = &owner.snapshots[owner.snapshot_index_for_rate(context.sample_rate)];
            // 输出率取宿主在 processContext 里报的率（那才是宿主缓冲的真实率）；它缺失/
            // 非法时退回 `setupProcessing` 的声明，避免 0 触发无意义的巨量重采样。
            let output_rate = if context.sample_rate.is_finite() && context.sample_rate > 0.0 {
                context.sample_rate
            } else {
                owner.setup_sample_rate()
            };
            // SAFETY: 缓冲经 clear_outputs 校验，publisher 由 owner 保留。
            let ready = unsafe {
                publisher.copy_block(
                    context.project_time_samples,
                    output_rate.max(1.0) as u32,
                    &mut *data.outputs,
                    data.num_samples as usize,
                )
            };
            if data.process_mode == 2 && !ready {
                return K_RESULT_FALSE;
            }
            K_RESULT_OK
        }
        Err(crate::audio_abi::BufferError::InvalidArgument) => K_INVALID_ARGUMENT,
        Err(crate::audio_abi::BufferError::UnsupportedFormat) => {
            // 样本宽度不支持（kSample64）时按实际宽度静音并留在链路里；其余形状问题
            // （多总线等）仍按失败上报 —— 见 `silence_unsupported_sample_size`。
            if unsafe {
                crate::audio_abi::silence_unsupported_sample_size(
                    data.cast::<crate::audio_abi::ProcessData>(),
                )
            } {
                K_RESULT_OK
            } else {
                K_RESULT_FALSE
            }
        }
    }
}

unsafe extern "system" fn audio_get_tail_samples(_this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || 0)
    }
}

static COMPONENT_VTBL: ComponentVtbl = ComponentVtbl {
    query_interface: component_query_interface,
    add_ref: component_add_ref,
    release: component_release,
    initialize: component_initialize,
    terminate: component_terminate,
    get_controller_class_id: component_get_controller_class_id,
    set_io_mode: component_set_io_mode,
    get_bus_count: component_get_bus_count,
    get_bus_info: component_get_bus_info,
    get_routing_info: component_get_routing_info,
    activate_bus: component_activate_bus,
    set_active: component_set_active,
    set_state: component_set_state,
    get_state: component_get_state,
};

static AUDIO_VTBL: AudioProcessorVtbl = AudioProcessorVtbl {
    query_interface: audio_query_interface,
    add_ref: audio_add_ref,
    release: audio_release,
    set_bus_arrangements: audio_set_bus_arrangements,
    get_bus_arrangement: audio_get_bus_arrangement,
    can_process_sample_size: audio_can_process_sample_size,
    get_latency_samples: audio_get_latency_samples,
    setup_processing: audio_setup_processing,
    set_processing: audio_set_processing,
    process: audio_process,
    get_tail_samples: audio_get_tail_samples,
};

// ---------------------------------------------------------------------------
// 编辑器控制器对象：内嵌原GUI的原生view入口，编辑会话关联另由connection提供。
// ---------------------------------------------------------------------------

/// `IEditController` 的 vtbl（`IPluginBase` 之后是 13 个控制器方法）。
///
/// 目前控制器参数尚未使用；createView返回真实原生视图。REAPER需要getControllerClassId
/// 指向一个真实类，
/// 否则会判定插入失败并卸载模块。
#[repr(C)]
pub struct EditControllerVtbl {
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const u8, *mut *mut c_void) -> TResult,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    pub initialize: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    pub terminate: unsafe extern "system" fn(*mut c_void) -> TResult,
    pub set_component_state: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    pub set_state: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    pub get_state: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    pub get_parameter_count: unsafe extern "system" fn(*mut c_void) -> i32,
    pub get_parameter_info: unsafe extern "system" fn(*mut c_void, i32, *mut c_void) -> TResult,
    pub get_param_string_by_value:
        unsafe extern "system" fn(*mut c_void, u32, f64, *mut u16) -> TResult,
    pub get_param_value_by_string:
        unsafe extern "system" fn(*mut c_void, u32, *mut u16, *mut f64) -> TResult,
    pub normalized_param_to_plain: unsafe extern "system" fn(*mut c_void, u32, f64) -> f64,
    pub plain_param_to_normalized: unsafe extern "system" fn(*mut c_void, u32, f64) -> f64,
    pub get_param_normalized: unsafe extern "system" fn(*mut c_void, u32) -> f64,
    pub set_param_normalized: unsafe extern "system" fn(*mut c_void, u32, f64) -> TResult,
    pub set_component_handler: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    pub create_view: unsafe extern "system" fn(*mut c_void, *const c_char) -> *mut c_void,
}

#[repr(C)]
struct EditController {
    vtbl: *const EditControllerVtbl,
    refcount: AtomicI32,
    connection_vtbl: *const crate::editor::connection::ConnectionVtbl,
    connection: crate::editor::connection::ConnectionState,
    editor_link: Arc<crate::editor::routing::EditorLink>,
}

impl EditController {
    fn new() -> Self {
        Self {
            vtbl: &EDIT_CONTROLLER_VTBL,
            refcount: AtomicI32::new(1),
            connection_vtbl: &CONTROLLER_CONNECTION_VTBL,
            connection: Default::default(),
            editor_link: Arc::new(Default::default()),
        }
    }
}

unsafe extern "system" fn edit_controller_query_interface(
    this: *mut c_void,
    iid: *const u8,
    obj: *mut *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_NO_INTERFACE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_NO_INTERFACE, || {
            if obj.is_null() {
                return K_INVALID_ARGUMENT;
            }
            // SAFETY: obj 是宿主提供的输出槽。
            unsafe { *obj = std::ptr::null_mut() };
            if unsafe { crate::editor::connection::is_connection(iid) } {
                let controller = this as *mut EditController;
                unsafe {
                    edit_controller_add_ref(this);
                    *obj = std::ptr::addr_of_mut!((*controller).connection_vtbl).cast();
                }
                return K_RESULT_OK;
            }
            if unsafe { iid_matches(iid, IID_FUNKNOWN) }
                || unsafe { iid_matches(iid, IID_IPLUGIN_BASE) }
                || unsafe { iid_matches(iid, IID_IEDIT_CONTROLLER) }
            {
                let controller = this as *const EditController;
                // SAFETY: this 是活动控制器对象。
                unsafe { (*controller).refcount.fetch_add(1, Ordering::AcqRel) };
                // SAFETY: 输出槽非空。
                unsafe { *obj = this };
                return K_RESULT_OK;
            }
            crate::log_line(&format!("controller queryInterface miss iid={}", unsafe {
                hex_iid(iid)
            }));
            K_NO_INTERFACE
        })
    }
}

unsafe extern "system" fn edit_controller_add_ref(this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || {
            let controller = this as *const EditController;
            // SAFETY: this 是活动控制器对象。
            unsafe { (*controller).refcount.fetch_add(1, Ordering::AcqRel) as u32 + 1 }
        })
    }
}

unsafe extern "system" fn edit_controller_release(this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || {
            let controller = this as *mut EditController;
            // SAFETY: this 是活动控制器对象。
            let previous = unsafe { (*controller).refcount.fetch_sub(1, Ordering::AcqRel) };
            let remaining = previous - 1;
            if remaining == 0 {
                // SAFETY: 引用计数归零，且没有其它持有者。
                unsafe { drop(Box::from_raw(controller)) };
            }
            remaining as u32
        })
    }
}

unsafe extern "system" fn edit_controller_initialize(
    this: *mut c_void,
    context: *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            unsafe {
                (*(this as *mut EditController))
                    .connection
                    .initialize(context);
            }
            crate::log_line("IEditController::initialize");
            K_RESULT_OK
        })
    }
}

unsafe extern "system" fn edit_controller_terminate(this: *mut c_void) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            unsafe {
                let controller = &*(this as *mut EditController);
                controller.editor_link.clear();
                let _ = controller
                    .editor_link
                    .set_component_handler(std::ptr::null_mut());
                controller.connection.close();
            }
            crate::log_line("IEditController::terminate");
            K_RESULT_OK
        })
    }
}

unsafe extern "system" fn edit_controller_set_component_state(
    _this: *mut c_void,
    _state: *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || K_NOT_IMPLEMENTED)
    }
}

unsafe extern "system" fn edit_controller_set_state(
    _this: *mut c_void,
    _state: *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || K_NOT_IMPLEMENTED)
    }
}

unsafe extern "system" fn edit_controller_get_state(
    _this: *mut c_void,
    _state: *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || K_NOT_IMPLEMENTED)
    }
}

unsafe extern "system" fn edit_controller_get_parameter_count(_this: *mut c_void) -> i32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || 0)
    }
}

unsafe extern "system" fn edit_controller_get_parameter_info(
    _this: *mut c_void,
    _index: i32,
    _info: *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || K_NOT_IMPLEMENTED)
    }
}

unsafe extern "system" fn edit_controller_get_param_string_by_value(
    _this: *mut c_void,
    _id: u32,
    _value: f64,
    _string: *mut u16,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || K_NOT_IMPLEMENTED)
    }
}

unsafe extern "system" fn edit_controller_get_param_value_by_string(
    _this: *mut c_void,
    _id: u32,
    _string: *mut u16,
    _value: *mut f64,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || K_NOT_IMPLEMENTED)
    }
}

unsafe extern "system" fn edit_controller_normalized_param_to_plain(
    _this: *mut c_void,
    _id: u32,
    value: f64,
) -> f64 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0.0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0.0, || value)
    }
}

unsafe extern "system" fn edit_controller_plain_param_to_normalized(
    _this: *mut c_void,
    _id: u32,
    value: f64,
) -> f64 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0.0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0.0, || value)
    }
}

unsafe extern "system" fn edit_controller_get_param_normalized(
    _this: *mut c_void,
    _id: u32,
) -> f64 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0.0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0.0, || 0.0)
    }
}

unsafe extern "system" fn edit_controller_set_param_normalized(
    _this: *mut c_void,
    _id: u32,
    _value: f64,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || K_NOT_IMPLEMENTED)
    }
}

unsafe extern "system" fn edit_controller_set_component_handler(
    this: *mut c_void,
    handler: *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            if this.is_null() {
                return K_INVALID_ARGUMENT;
            }
            let controller = unsafe { &*(this as *const EditController) };
            if controller
                .editor_link
                .set_component_handler(handler)
                .is_err()
            {
                return K_RESULT_FALSE;
            }
            crate::log_line(&format!(
                "IEditController::setComponentHandler({})",
                if handler.is_null() {
                    "null"
                } else {
                    "connected"
                }
            ));
            K_RESULT_OK
        })
    }
}

unsafe extern "system" fn edit_controller_create_view(
    this: *mut c_void,
    name: *const c_char,
) -> *mut c_void {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `std::ptr::null_mut()` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(std::ptr::null_mut(), || {
            if this.is_null()
                || name.is_null()
                || unsafe { std::ffi::CStr::from_ptr(name) }.to_bytes() != b"editor"
            {
                return std::ptr::null_mut();
            }
            crate::log_line("IEditController::createView -> native editor");
            crate::editor::create_view_with_link(
                unsafe { &*(this as *const EditController) }
                    .editor_link
                    .clone(),
            )
        })
    }
}

static EDIT_CONTROLLER_VTBL: EditControllerVtbl = EditControllerVtbl {
    query_interface: edit_controller_query_interface,
    add_ref: edit_controller_add_ref,
    release: edit_controller_release,
    initialize: edit_controller_initialize,
    terminate: edit_controller_terminate,
    set_component_state: edit_controller_set_component_state,
    set_state: edit_controller_set_state,
    get_state: edit_controller_get_state,
    get_parameter_count: edit_controller_get_parameter_count,
    get_parameter_info: edit_controller_get_parameter_info,
    get_param_string_by_value: edit_controller_get_param_string_by_value,
    get_param_value_by_string: edit_controller_get_param_value_by_string,
    normalized_param_to_plain: edit_controller_normalized_param_to_plain,
    plain_param_to_normalized: edit_controller_plain_param_to_normalized,
    get_param_normalized: edit_controller_get_param_normalized,
    set_param_normalized: edit_controller_set_param_normalized,
    set_component_handler: edit_controller_set_component_handler,
    create_view: edit_controller_create_view,
};

/// 从真实connection子对象回到对应拥有者，所有接口共享同一个COM引用计数。
unsafe fn processor_from_connection(this: *mut c_void) -> *mut Processor {
    unsafe {
        this.cast::<u8>()
            .sub(offset_of!(Processor, connection_vtbl))
            .cast()
    }
}
unsafe fn controller_from_connection(this: *mut c_void) -> *mut EditController {
    unsafe {
        this.cast::<u8>()
            .sub(offset_of!(EditController, connection_vtbl))
            .cast()
    }
}
unsafe extern "system" fn pc_query(
    this: *mut c_void,
    iid: *const u8,
    out: *mut *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || unsafe {
            component_query_interface(processor_from_connection(this).cast(), iid, out)
        })
    }
}
unsafe extern "system" fn pc_add(this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || unsafe {
            processor_add_ref(processor_from_connection(this))
        })
    }
}
unsafe extern "system" fn pc_release(this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || unsafe {
            processor_release(processor_from_connection(this))
        })
    }
}
unsafe extern "system" fn pc_connect(this: *mut c_void, other: *mut c_void) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            let processor = unsafe { &*processor_from_connection(this) };
            let result = unsafe { processor.connection.connect(other) };
            if result == K_RESULT_OK {
                let sent = processor.connection.send(Some(processor.route.token()));
                crate::log_line(&format!("processor connection route sent result={sent}"));
            }
            result
        })
    }
}
unsafe extern "system" fn pc_disconnect(this: *mut c_void, other: *mut c_void) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            unsafe { &*processor_from_connection(this) }
                .connection
                .disconnect(other)
        })
    }
}
unsafe extern "system" fn pc_notify(this: *mut c_void, message: *mut c_void) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            let processor = unsafe { &*processor_from_connection(this) };
            match unsafe { crate::editor::connection::read(message) } {
                Ok(crate::editor::connection::Message::RequestRoute) => {
                    processor.connection.send(Some(processor.route.token()))
                }
                _ => K_RESULT_FALSE,
            }
        })
    }
}
unsafe extern "system" fn cc_query(
    this: *mut c_void,
    iid: *const u8,
    out: *mut *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || unsafe {
            edit_controller_query_interface(controller_from_connection(this).cast(), iid, out)
        })
    }
}
unsafe extern "system" fn cc_add(this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || unsafe {
            edit_controller_add_ref(controller_from_connection(this).cast())
        })
    }
}
unsafe extern "system" fn cc_release(this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || unsafe {
            edit_controller_release(controller_from_connection(this).cast())
        })
    }
}
unsafe extern "system" fn cc_connect(this: *mut c_void, other: *mut c_void) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            let controller = unsafe { &*controller_from_connection(this) };
            let result = unsafe { controller.connection.connect(other) };
            if result == K_RESULT_OK {
                let sent = controller.connection.send(None);
                crate::log_line(&format!(
                    "controller connection route requested result={sent}"
                ));
            }
            result
        })
    }
}
unsafe extern "system" fn cc_disconnect(this: *mut c_void, other: *mut c_void) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            let controller = unsafe { &*controller_from_connection(this) };
            let result = controller.connection.disconnect(other);
            if result == K_RESULT_OK {
                controller.editor_link.clear();
            }
            result
        })
    }
}
unsafe extern "system" fn cc_notify(this: *mut c_void, message: *mut c_void) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            let controller = unsafe { &*controller_from_connection(this) };
            match unsafe { crate::editor::connection::read(message) } {
                Ok(crate::editor::connection::Message::Route { pid, token }) => {
                    match controller.editor_link.bind(pid, &token) {
                        Ok(()) => {
                            crate::log_line("controller editor route bound to actual processor");
                            K_RESULT_OK
                        }
                        Err(error) => {
                            crate::log_line(&format!("controller editor route rejected: {error}"));
                            K_RESULT_FALSE
                        }
                    }
                }
                _ => K_RESULT_FALSE,
            }
        })
    }
}
static PROCESSOR_CONNECTION_VTBL: crate::editor::connection::ConnectionVtbl =
    crate::editor::connection::ConnectionVtbl {
        base: crate::editor::connection::UnknownVtbl {
            query: pc_query,
            add: pc_add,
            release: pc_release,
        },
        connect: pc_connect,
        disconnect: pc_disconnect,
        notify: pc_notify,
    };
static CONTROLLER_CONNECTION_VTBL: crate::editor::connection::ConnectionVtbl =
    crate::editor::connection::ConnectionVtbl {
        base: crate::editor::connection::UnknownVtbl {
            query: cc_query,
            add: cc_add,
            release: cc_release,
        },
        connect: cc_connect,
        disconnect: cc_disconnect,
        notify: cc_notify,
    };

/// 建立 ARA 主工厂适配器（`ARA::IMainFactory`）。
pub fn new_main_factory_adapter() -> Option<Vst3MainFactoryAdapter> {
    let runtime = crate::runtime::runtime()?;
    Vst3MainFactoryAdapter::new(crate::CLASS_NAME, runtime.companion.clone()).ok()
}

#[cfg(test)]
mod audio_boundary_tests;
#[cfg(test)]
mod lifetime_tests;
#[cfg(test)]
mod realtime_tests;
