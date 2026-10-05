//! HiFiShifter ARA 探针 / Task 2 Step 3 —— 手写的最小 VST3 模块 ABI。
//!
//! 目标只有一个：让 REAPER 能加载这个 cdylib、把它识别为 ARA 插件，并把 ARA 对象
//! 推给 `ara2-bridge` 的文档控制器（见 `model.rs` 的计数）。
//!
//! 这里为什么手写 vtbl 而不引入 `vst3-sys`：本机 cargo 缓存里没有 VST3 绑定，
//! 且探针只用到极少接口。vtbl 的槽位顺序、结构体布局与调用约定都按锁定的
//! VST3 SDK `v3.8.0_build_66` 头文件逐条对齐（`ipluginbase.h` / `ivstcomponent.h` /
//! `ivstaudioprocessor.h` / `funknown.h`）。ARA 侧的接口（`IMainFactory`、
//! `IPlugInEntryPoint`、`IPlugInEntryPoint2`）不在这里实现，而是转发给
//! `ara2-bridge-companion` 的 C++ 适配器，避免重复实现引用计数语义。
//!
//! 产品外壳从一次性探针迁入；尚未完成的渲染/生命周期事项见 Phase 3a 计划。

use ara2_bridge::companion::vst3::ffi::{
    ara2_vst3_interface_id, ara2_vst3_main_factory_category, Ara2Vst3InterfaceId,
    Ara2Vst3InterfaceKind, ARA2_VST3_OK,
};
use ara2_bridge::companion::vst3::Vst3MainFactoryAdapter;
use crate::ara_entry::HostEntry;
use ara2_bridge::companion::{CompanionProcessorBinding, CompanionRoles};
use std::ffi::{c_char, c_void};
use std::mem::offset_of;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use crate::render::extension::ExtensionOwner;

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
    pub query_interface: unsafe extern "system" fn(*mut c_void, *const u8, *mut *mut c_void) -> TResult,
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
    crate::log_line(&format!(
        "factory queryInterface miss iid={}",
        unsafe { hex_iid(iid) }
    ));
    K_NO_INTERFACE
}

unsafe extern "system" fn factory_add_ref(this: *mut c_void) -> u32 {
    let factory = this as *const PluginFactoryObj;
    // SAFETY: this 是形如 PluginFactoryObj 的常驻单例。
    unsafe { (*factory).refcount.fetch_add(1, Ordering::AcqRel) as u32 + 1 }
}

unsafe extern "system" fn factory_release(this: *mut c_void) -> u32 {
    let factory = this as *const PluginFactoryObj;
    // SAFETY: this 是形如 PluginFactoryObj 的常驻单例；探针不销毁它。
    let previous = unsafe { (*factory).refcount.fetch_sub(1, Ordering::AcqRel) };
    (previous - 1) as u32
}

unsafe extern "system" fn factory_get_factory_info(
    _this: *mut c_void,
    info: *mut PFactoryInfo,
) -> TResult {
    if info.is_null() {
        return K_INVALID_ARGUMENT;
    }
    // SAFETY: info 是宿主提供的可写 PFactoryInfo。
    let info = unsafe { &mut *info };
    copy_str(&mut info.vendor, "HiFiShifter");
    copy_str(&mut info.url, "https://example.invalid");
    copy_str(&mut info.email, "");
    // kUnicode = 1 << 4
    info.flags = 1 << 4;
    K_RESULT_OK
}

unsafe extern "system" fn factory_count_classes(_this: *mut c_void) -> i32 {
    // 0: 音频处理器组件，1: ARA 主工厂类，2: 编辑器控制器。
    crate::log_line("factory countClasses -> 3");
    3
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
    if info.is_null() || index < 0 || index > 2 {
        return K_INVALID_ARGUMENT;
    }
    // SAFETY: info 是宿主提供的可写 PClassInfo。
    unsafe { fill_class_info(info, index) };
    crate::log_line(&format!("factory getClassInfo({index}) ok"));
    K_RESULT_OK
}

unsafe extern "system" fn factory_get_class_info2(
    _this: *mut c_void,
    index: i32,
    info: *mut PClassInfo2,
) -> TResult {
    if info.is_null() || index < 0 || index > 2 {
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
}

unsafe extern "system" fn factory_get_class_info_unicode(
    _this: *mut c_void,
    _index: i32,
    _info: *mut c_void,
) -> TResult {
    K_NOT_IMPLEMENTED
}

unsafe extern "system" fn factory_set_host_context(
    _this: *mut c_void,
    _context: *mut c_void,
) -> TResult {
    K_RESULT_OK
}

unsafe extern "system" fn factory_create_instance(
    _this: *mut c_void,
    cid: *const c_char,
    iid: *const c_char,
    obj: *mut *mut c_void,
) -> TResult {
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
        let result = unsafe { edit_controller_query_interface(controller, iid as *const u8, obj) };
        if result != K_RESULT_OK {
            // SAFETY: 引用计数仍为 1，且未把指针交给宿主。
            unsafe { drop(Box::from_raw(controller as *mut EditController)) };
        }
        return result;
    }
    K_INVALID_ARGUMENT
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
    pub query_interface: unsafe extern "system" fn(*mut c_void, *const u8, *mut *mut c_void) -> TResult,
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
    pub get_bus_info: unsafe extern "system" fn(*mut c_void, i32, i32, i32, *mut BusInfo) -> TResult,
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
    pub query_interface: unsafe extern "system" fn(*mut c_void, *const u8, *mut *mut c_void) -> TResult,
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
    connection_vtbl:*const crate::editor::connection::ConnectionVtbl,
    connection:crate::editor::connection::ConnectionState,
    route:crate::editor::routing::RouteLease,
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
        let route=crate::editor::routing::RouteLease::new(&extension_owner);
        let entry = HostEntry::new(binding, crate::CLASS_NAME, extension_owner.clone()).ok()?;
        Some(Box::new(Processor {
            component_vtbl: &COMPONENT_VTBL,
            refcount: AtomicI32::new(1),
            audio_vtbl: &AUDIO_VTBL,
            entry: Some(entry),
            extension_owner,
            active: AtomicI32::new(0),
            connection_vtbl:&PROCESSOR_CONNECTION_VTBL,
            connection:Default::default(),
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
        unsafe { (*processor).extension_owner.stop_channel(); }
        unsafe { (*processor).extension_owner.stop_editor(); }
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
    if obj.is_null() {
        return K_INVALID_ARGUMENT;
    }
    // SAFETY: obj 是宿主提供的输出槽。
    unsafe { *obj = std::ptr::null_mut() };
    let processor = this as *mut Processor;
    if unsafe { crate::editor::connection::is_connection(iid) } {
        unsafe { processor_add_ref(processor); *obj=std::ptr::addr_of_mut!((*processor).connection_vtbl).cast(); }
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
    crate::log_line(&format!(
        "component queryInterface miss iid={}",
        unsafe { hex_iid(iid) }
    ));
    K_NO_INTERFACE
}

unsafe extern "system" fn component_add_ref(this: *mut c_void) -> u32 {
    // SAFETY: this 是组件接口指针（对象基址）。
    unsafe { processor_add_ref(this as *mut Processor) }
}

unsafe extern "system" fn component_release(this: *mut c_void) -> u32 {
    // SAFETY: this 是组件接口指针（对象基址）。
    unsafe { processor_release(this as *mut Processor) }
}

unsafe extern "system" fn component_initialize(
    this: *mut c_void,
    context: *mut c_void,
) -> TResult {
    // QI/GetApi可重入释放宿主组件引用；短引用只保活本次调用的插件存储，不保活project/take。
    struct CallReference(*mut c_void);
    impl Drop for CallReference {
        fn drop(&mut self) {unsafe {component_release(self.0);}}
    }
    unsafe {component_add_ref(this);}
    let _call_reference=CallReference(this);
    let owner=unsafe {(*this.cast::<Processor>()).extension_owner.clone()};
    if owner.is_closed() {return K_RESULT_FALSE;}
    unsafe { (*(this as *mut Processor)).connection.initialize(context); }
    if owner.is_closed() {
        // QI可能先重入close再返回新引用；拒绝初始化时不能把该引用留在closed connection。
        unsafe {(*this.cast::<Processor>()).connection.close();}
        return K_RESULT_FALSE;
    }
    unsafe {owner.bind_reaper_host(context);}
    if owner.is_closed() {
        unsafe {(*this.cast::<Processor>()).connection.close();}
        return K_RESULT_FALSE;
    }
    crate::log_line("IComponent::initialize");
    K_RESULT_OK
}

unsafe extern "system" fn component_terminate(this: *mut c_void) -> TResult {
    unsafe {let processor=&*(this as *mut Processor);processor.connection.close();processor.extension_owner.stop_editor();}
    crate::log_line("IComponent::terminate");
    K_RESULT_OK
}

unsafe extern "system" fn component_get_controller_class_id(
    _this: *mut c_void,
    class_id: *mut u8,
) -> TResult {
    if class_id.is_null() {
        return K_INVALID_ARGUMENT;
    }
    // 探针提供一个最小编辑器控制器：REAPER 需要它才能完成 ARA 插入。
    // SAFETY: class_id 是宿主提供的可写 TUID。
    unsafe { std::ptr::copy_nonoverlapping(CONTROLLER_CID.as_ptr(), class_id, 16) };
    crate::log_line("IComponent::getControllerClassId -> HFSARAProbe.Ctl1");
    K_RESULT_OK
}

unsafe extern "system" fn component_set_io_mode(_this: *mut c_void, _mode: i32) -> TResult {
    K_RESULT_OK
}

unsafe extern "system" fn component_get_bus_count(
    _this: *mut c_void,
    media_type: i32,
    _direction: i32,
) -> i32 {
    if media_type == 0 {
        1
    } else {
        0
    }
}

unsafe extern "system" fn component_get_bus_info(
    _this: *mut c_void,
    media_type: i32,
    direction: i32,
    index: i32,
    bus: *mut BusInfo,
) -> TResult {
    if bus.is_null() || media_type != 0 || index != 0 {
        return K_INVALID_ARGUMENT;
    }
    // SAFETY: bus 是宿主提供的可写 BusInfo。
    let bus = unsafe { &mut *bus };
    // SAFETY: POD 清零。
    unsafe { std::ptr::write_bytes(bus as *mut BusInfo, 0, 1) };
    bus.media_type = 0;
    bus.direction = direction;
    bus.channel_count = 2;
    bus.bus_type = 0;
    bus.flags = 1; // kDefaultActive
    bus.name[0] = b'M' as u16;
    K_RESULT_OK
}

unsafe extern "system" fn component_get_routing_info(
    _this: *mut c_void,
    _in_info: *mut RoutingInfo,
    _out_info: *mut RoutingInfo,
) -> TResult {
    K_NOT_IMPLEMENTED
}

unsafe extern "system" fn component_activate_bus(
    _this: *mut c_void,
    _media_type: i32,
    _direction: i32,
    _index: i32,
    _state: i8,
) -> TResult {
    K_RESULT_OK
}

unsafe extern "system" fn component_set_active(this: *mut c_void, state: i8) -> TResult {
    let processor = this as *mut Processor;
    // SAFETY: processor 是活动对象。
    unsafe { (*processor).active.store(state as i32, Ordering::SeqCst) };
    crate::log_line(&format!("IComponent::setActive({state})"));
    K_RESULT_OK
}

unsafe extern "system" fn component_set_state(
    this: *mut c_void,
    state: *mut c_void,
) -> TResult {
    if this.is_null() { return K_INVALID_ARGUMENT; }
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let bytes = unsafe { crate::state_stream::read_state(state) }?;
        let owner = unsafe { &(*this.cast::<Processor>()).extension_owner };
        owner.restore_state(&bytes)?;
        Ok::<_, String>(())
    }));
    if matches!(outcome, Ok(Ok(()))) { K_RESULT_OK } else { K_RESULT_FALSE }
}

unsafe extern "system" fn component_get_state(
    this: *mut c_void,
    state: *mut c_void,
) -> TResult {
    if this.is_null() { return K_INVALID_ARGUMENT; }
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let owner = unsafe { &(*this.cast::<Processor>()).extension_owner };
        let bytes = owner.encode_state()?;
        unsafe { crate::state_stream::write_state(state, &bytes) }
    }));
    if matches!(outcome, Ok(Ok(()))) { K_RESULT_OK } else { K_RESULT_FALSE }
}

unsafe extern "system" fn audio_query_interface(
    this: *mut c_void,
    iid: *const u8,
    obj: *mut *mut c_void,
) -> TResult {
    // SAFETY: this 是 IAudioProcessor 子对象指针。
    let base = unsafe { base_from_audio(this) } as *mut c_void;
    // SAFETY: 转发到基址查询。
    unsafe { component_query_interface(base, iid, obj) }
}

unsafe extern "system" fn audio_add_ref(this: *mut c_void) -> u32 {
    // SAFETY: this 是 IAudioProcessor 子对象指针。
    unsafe { processor_add_ref(base_from_audio(this)) }
}

unsafe extern "system" fn audio_release(this: *mut c_void) -> u32 {
    // SAFETY: this 是 IAudioProcessor 子对象指针。
    unsafe { processor_release(base_from_audio(this)) }
}

unsafe extern "system" fn audio_set_bus_arrangements(
    _this: *mut c_void,
    inputs: *mut u64,
    num_inputs: i32,
    outputs: *mut u64,
    num_outputs: i32,
) -> TResult {
    if num_inputs != 1 || num_outputs != 1 {
        return K_RESULT_FALSE;
    }
    if inputs.is_null() || outputs.is_null() {
        return K_INVALID_ARGUMENT;
    }
    // SAFETY: 宿主按数量提供合法布局数组；本批只声明一进一出 stereo。
    if unsafe { *inputs != 3 || *outputs != 3 } {
        return K_RESULT_FALSE;
    }
    K_RESULT_OK
}

unsafe extern "system" fn audio_get_bus_arrangement(
    _this: *mut c_void,
    _direction: i32,
    _index: i32,
    arrangement: *mut u64,
) -> TResult {
    if arrangement.is_null() {
        return K_INVALID_ARGUMENT;
    }
    // SAFETY: arrangement 是宿主提供的可写 SpeakerArrangement。
    unsafe { *arrangement = 3 }; // kStereo
    K_RESULT_OK
}

unsafe extern "system" fn audio_can_process_sample_size(
    _this: *mut c_void,
    symbolic_sample_size: i32,
) -> TResult {
    if symbolic_sample_size == 0 {
        K_RESULT_OK
    } else {
        K_RESULT_FALSE
    }
}

unsafe extern "system" fn audio_get_latency_samples(_this: *mut c_void) -> u32 {
    0
}

unsafe extern "system" fn audio_setup_processing(
    this: *mut c_void,
    setup: *mut ProcessSetup,
) -> TResult {
    if this.is_null()||setup.is_null() {
        return K_INVALID_ARGUMENT;
    }
    // SAFETY: 宿主提供调用期间可读的 SDK setup 结构。
    let setup = unsafe { &*setup };
    if !setup.sample_rate.is_finite() || setup.sample_rate <= 0.0
        || setup.max_samples_per_block <= 0 || !(0..=2).contains(&setup.process_mode)
    {
        return K_INVALID_ARGUMENT;
    }
    if setup.symbolic_sample_size != 0 || ![44100.0, 48000.0].contains(&setup.sample_rate) {
        return K_RESULT_FALSE;
    }
    crate::log_line(&format!("IAudioProcessor::setupProcessing mode={} rate={}",setup.process_mode,setup.sample_rate));
    if setup.process_mode==2 {
        // SAFETY: this为存活音频接口；SDK明确setup在UI线程/禁用状态调用。
        let owner=unsafe {&(*base_from_audio(this)).extension_owner};
        if let Err(error)=owner.prepare_offline_until(std::time::Instant::now()+std::time::Duration::from_secs(180)) {
            crate::log_line(&format!("Offline preparation failed: {error}"));return K_RESULT_FALSE;
        }
    }
    K_RESULT_OK
}

unsafe extern "system" fn audio_set_processing(this: *mut c_void, state: i8) -> TResult {
    // SDK 允许从音频线程调用此函数；禁止触发同步文件日志。
    if this.is_null() {return K_INVALID_ARGUMENT;}
    if state==0 {unsafe {&(*base_from_audio(this)).extension_owner}.record_processing_stop();}
    K_RESULT_OK
}

/// 初始化宿主缓冲的安全边界；实际 PCM 快照输出在 Phase 3a Task 14 接入。
unsafe extern "system" fn audio_process(this: *mut c_void, data: *mut c_void) -> TResult {
    // SAFETY: VST3 宿主按 SDK 布局提供回调数据，函数内部处理空指针与非法字段。
    if this.is_null() {return K_INVALID_ARGUMENT;}
    match unsafe { crate::audio_abi::validate(data.cast()) } {
        Ok(()) => {
            // SAFETY: 宿主持有 Processor，validate已验证当前数据和音频总线。
            let data = unsafe { &mut *data.cast::<crate::audio_abi::ProcessData>() };
            if data.num_samples == 0 || data.num_outputs == 0 { return K_RESULT_OK; }
            // SAFETY: this 是存活的音频接口子对象。
            let owner = unsafe { &(*base_from_audio(this)).extension_owner };
            if !data.process_context.is_null() {
                owner.observe_transport(unsafe {&*data.process_context},data.process_mode);
            }
            if owner.is_editor_only() {
                // 纯编辑角色不是歌曲播放renderer；保留宿主已经处理的fade/gain/前级FX，包括停播监听。
                return match unsafe {crate::audio_abi::pass_through(data)} {
                    Ok(())=>K_RESULT_OK,Err(crate::audio_abi::BufferError::InvalidArgument)=>K_INVALID_ARGUMENT,
                    Err(crate::audio_abi::BufferError::UnsupportedFormat)=>K_RESULT_FALSE,
                };
            }
            unsafe {crate::audio_abi::clear_outputs(data).expect("validated audio buffers");}
            if data.process_context.is_null() { owner.snapshots[0].misses.fetch_add(1, Ordering::Relaxed); return if data.process_mode==2 {K_RESULT_FALSE} else {K_RESULT_OK}; }
            // SAFETY: VST3 processContext 的完整 SDK 结构在当前回调期间存活。
            let context = unsafe { &*data.process_context };
            // REAPER停播也会process固定光标位置；只离线导出允许没有kPlaying的供音。
            if data.process_mode!=2 && context.state & (1<<1)==0 {return K_RESULT_OK;}
            let publisher = match context.sample_rate {
                44100.0 => &owner.snapshots[0], 48000.0 => &owner.snapshots[1], _ => return K_RESULT_FALSE,
            };
            // SAFETY: 缓冲经 clear_outputs 校验，publisher 由 owner 保留。
            let ready=unsafe { publisher.copy_block(context.project_time_samples, context.sample_rate as u32, &mut *data.outputs, data.num_samples as usize) };
            if data.process_mode==2&&!ready {return K_RESULT_FALSE;}
            K_RESULT_OK
        },
        Err(crate::audio_abi::BufferError::InvalidArgument) => K_INVALID_ARGUMENT,
        Err(crate::audio_abi::BufferError::UnsupportedFormat) => K_RESULT_FALSE,
    }
}

unsafe extern "system" fn audio_get_tail_samples(_this: *mut c_void) -> u32 {
    0
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
    connection_vtbl:*const crate::editor::connection::ConnectionVtbl,
    connection:crate::editor::connection::ConnectionState,
    editor_link:Arc<crate::editor::routing::EditorLink>,
}

impl EditController {
    fn new() -> Self {
        Self {
            vtbl: &EDIT_CONTROLLER_VTBL,
            refcount: AtomicI32::new(1),
            connection_vtbl:&CONTROLLER_CONNECTION_VTBL,
            connection:Default::default(),
            editor_link:Arc::new(Default::default()),
        }
    }
}

unsafe extern "system" fn edit_controller_query_interface(
    this: *mut c_void,
    iid: *const u8,
    obj: *mut *mut c_void,
) -> TResult {
    if obj.is_null() {
        return K_INVALID_ARGUMENT;
    }
    // SAFETY: obj 是宿主提供的输出槽。
    unsafe { *obj = std::ptr::null_mut() };
    if unsafe { crate::editor::connection::is_connection(iid) } {
        let controller=this as *mut EditController;
        unsafe { edit_controller_add_ref(this); *obj=std::ptr::addr_of_mut!((*controller).connection_vtbl).cast(); }
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
    crate::log_line(&format!(
        "controller queryInterface miss iid={}",
        unsafe { hex_iid(iid) }
    ));
    K_NO_INTERFACE
}

unsafe extern "system" fn edit_controller_add_ref(this: *mut c_void) -> u32 {
    let controller = this as *const EditController;
    // SAFETY: this 是活动控制器对象。
    unsafe { (*controller).refcount.fetch_add(1, Ordering::AcqRel) as u32 + 1 }
}

unsafe extern "system" fn edit_controller_release(this: *mut c_void) -> u32 {
    let controller = this as *mut EditController;
    // SAFETY: this 是活动控制器对象。
    let previous = unsafe { (*controller).refcount.fetch_sub(1, Ordering::AcqRel) };
    let remaining = previous - 1;
    if remaining == 0 {
        // SAFETY: 引用计数归零，且没有其它持有者。
        unsafe { drop(Box::from_raw(controller)) };
    }
    remaining as u32
}

unsafe extern "system" fn edit_controller_initialize(
    this: *mut c_void,
    context: *mut c_void,
) -> TResult {
    unsafe { (*(this as *mut EditController)).connection.initialize(context); }
    crate::log_line("IEditController::initialize");
    K_RESULT_OK
}

unsafe extern "system" fn edit_controller_terminate(this: *mut c_void) -> TResult {
    unsafe { let controller=&*(this as *mut EditController); controller.editor_link.clear(); controller.connection.close(); }
    crate::log_line("IEditController::terminate");
    K_RESULT_OK
}

unsafe extern "system" fn edit_controller_set_component_state(
    _this: *mut c_void,
    _state: *mut c_void,
) -> TResult {
    K_NOT_IMPLEMENTED
}

unsafe extern "system" fn edit_controller_set_state(
    _this: *mut c_void,
    _state: *mut c_void,
) -> TResult {
    K_NOT_IMPLEMENTED
}

unsafe extern "system" fn edit_controller_get_state(
    _this: *mut c_void,
    _state: *mut c_void,
) -> TResult {
    K_NOT_IMPLEMENTED
}

unsafe extern "system" fn edit_controller_get_parameter_count(_this: *mut c_void) -> i32 {
    0
}

unsafe extern "system" fn edit_controller_get_parameter_info(
    _this: *mut c_void,
    _index: i32,
    _info: *mut c_void,
) -> TResult {
    K_NOT_IMPLEMENTED
}

unsafe extern "system" fn edit_controller_get_param_string_by_value(
    _this: *mut c_void,
    _id: u32,
    _value: f64,
    _string: *mut u16,
) -> TResult {
    K_NOT_IMPLEMENTED
}

unsafe extern "system" fn edit_controller_get_param_value_by_string(
    _this: *mut c_void,
    _id: u32,
    _string: *mut u16,
    _value: *mut f64,
) -> TResult {
    K_NOT_IMPLEMENTED
}

unsafe extern "system" fn edit_controller_normalized_param_to_plain(
    _this: *mut c_void,
    _id: u32,
    value: f64,
) -> f64 {
    value
}

unsafe extern "system" fn edit_controller_plain_param_to_normalized(
    _this: *mut c_void,
    _id: u32,
    value: f64,
) -> f64 {
    value
}

unsafe extern "system" fn edit_controller_get_param_normalized(
    _this: *mut c_void,
    _id: u32,
) -> f64 {
    0.0
}

unsafe extern "system" fn edit_controller_set_param_normalized(
    _this: *mut c_void,
    _id: u32,
    _value: f64,
) -> TResult {
    K_NOT_IMPLEMENTED
}

unsafe extern "system" fn edit_controller_set_component_handler(
    _this: *mut c_void,
    _handler: *mut c_void,
) -> TResult {
    K_RESULT_OK
}

unsafe extern "system" fn edit_controller_create_view(
    this: *mut c_void,
    name: *const c_char,
) -> *mut c_void {
    if this.is_null() || name.is_null() || unsafe { std::ffi::CStr::from_ptr(name) }.to_bytes() != b"editor" {
        return std::ptr::null_mut();
    }
    crate::log_line("IEditController::createView -> native editor");
    crate::editor::create_view_with_link(unsafe { &*(this as *const EditController) }.editor_link.clone())
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
unsafe fn processor_from_connection(this:*mut c_void)->*mut Processor {
    unsafe { this.cast::<u8>().sub(offset_of!(Processor,connection_vtbl)).cast() }
}
unsafe fn controller_from_connection(this:*mut c_void)->*mut EditController {
    unsafe { this.cast::<u8>().sub(offset_of!(EditController,connection_vtbl)).cast() }
}
unsafe extern "system" fn pc_query(this:*mut c_void,iid:*const u8,out:*mut *mut c_void)->TResult {
    unsafe { component_query_interface(processor_from_connection(this).cast(),iid,out) }
}
unsafe extern "system" fn pc_add(this:*mut c_void)->u32 { unsafe { processor_add_ref(processor_from_connection(this)) } }
unsafe extern "system" fn pc_release(this:*mut c_void)->u32 { unsafe { processor_release(processor_from_connection(this)) } }
unsafe extern "system" fn pc_connect(this:*mut c_void,other:*mut c_void)->TResult {
    let processor=unsafe { &*processor_from_connection(this) };
    let result=unsafe { processor.connection.connect(other) };
    if result==K_RESULT_OK {
        let sent=processor.connection.send(Some(processor.route.token()));
        crate::log_line(&format!("processor connection route sent result={sent}"));
    }
    result
}
unsafe extern "system" fn pc_disconnect(this:*mut c_void,other:*mut c_void)->TResult {
    unsafe { &*processor_from_connection(this) }.connection.disconnect(other)
}
unsafe extern "system" fn pc_notify(this:*mut c_void,message:*mut c_void)->TResult {
    let processor=unsafe { &*processor_from_connection(this) };
    match unsafe { crate::editor::connection::read(message) } {
        Ok(crate::editor::connection::Message::RequestRoute)=>processor.connection.send(Some(processor.route.token())),
        _=>K_RESULT_FALSE,
    }
}
unsafe extern "system" fn cc_query(this:*mut c_void,iid:*const u8,out:*mut *mut c_void)->TResult {
    unsafe { edit_controller_query_interface(controller_from_connection(this).cast(),iid,out) }
}
unsafe extern "system" fn cc_add(this:*mut c_void)->u32 { unsafe { edit_controller_add_ref(controller_from_connection(this).cast()) } }
unsafe extern "system" fn cc_release(this:*mut c_void)->u32 { unsafe { edit_controller_release(controller_from_connection(this).cast()) } }
unsafe extern "system" fn cc_connect(this:*mut c_void,other:*mut c_void)->TResult {
    let controller=unsafe { &*controller_from_connection(this) };
    let result=unsafe { controller.connection.connect(other) };
    if result==K_RESULT_OK {
        let sent=controller.connection.send(None);
        crate::log_line(&format!("controller connection route requested result={sent}"));
    }
    result
}
unsafe extern "system" fn cc_disconnect(this:*mut c_void,other:*mut c_void)->TResult {
    let controller=unsafe { &*controller_from_connection(this) };
    let result=controller.connection.disconnect(other);
    if result==K_RESULT_OK { controller.editor_link.clear(); }
    result
}
unsafe extern "system" fn cc_notify(this:*mut c_void,message:*mut c_void)->TResult {
    let controller=unsafe { &*controller_from_connection(this) };
    match unsafe { crate::editor::connection::read(message) } {
        Ok(crate::editor::connection::Message::Route {pid,token})=>match controller.editor_link.bind(pid,&token) {
            Ok(())=>{ crate::log_line("controller editor route bound to actual processor"); K_RESULT_OK },
            Err(error)=>{ crate::log_line(&format!("controller editor route rejected: {error}")); K_RESULT_FALSE },
        },
        _=>K_RESULT_FALSE,
    }
}
static PROCESSOR_CONNECTION_VTBL:crate::editor::connection::ConnectionVtbl=crate::editor::connection::ConnectionVtbl {
    base:crate::editor::connection::UnknownVtbl {query:pc_query,add:pc_add,release:pc_release},
    connect:pc_connect,disconnect:pc_disconnect,notify:pc_notify,
};
static CONTROLLER_CONNECTION_VTBL:crate::editor::connection::ConnectionVtbl=crate::editor::connection::ConnectionVtbl {
    base:crate::editor::connection::UnknownVtbl {query:cc_query,add:cc_add,release:cc_release},
    connect:cc_connect,disconnect:cc_disconnect,notify:cc_notify,
};

/// 建立 ARA 主工厂适配器（`ARA::IMainFactory`）。
pub fn new_main_factory_adapter() -> Option<Vst3MainFactoryAdapter> {
    let runtime = crate::runtime::runtime()?;
    Vst3MainFactoryAdapter::new(crate::CLASS_NAME, runtime.companion.clone()).ok()
}

#[cfg(test)]
mod lifetime_tests {
    use super::*;
    use ara2_bridge::companion::vst3::ffi::ara2_vst3_release;
    use std::sync::atomic::AtomicU32;

    /// 从真实工厂创建组件，返回其宿主 owning COM 引用。
    fn create_component() -> *mut c_void {
        let iid = uid_guid(IID_ICOMPONENT);
        let mut component = std::ptr::null_mut();
        // SAFETY: CID/IID 和输出槽均在同步工厂调用期间存活。
        let result = unsafe {
            factory_create_instance(std::ptr::null_mut(), PROCESSOR_CID.as_ptr().cast(), iid.as_ptr().cast(), &raw mut component)
        };
        assert_eq!(result, K_RESULT_OK);
        component
    }

    /// 安全RED：只观察原生refcount，不在尚未保活的实现中重入最终release触发悬空。
    #[test]
    fn task38a_initialize_owns_a_short_component_reference_before_querying_host() {
        let component=create_component();let host=crate::host::reaper::ReaperFixture::new();
        let observed=Arc::new(AtomicI32::new(0));let count=observed.clone();let raw=component as usize;
        *host.hook.borrow_mut()=Some(("api:GetPlayPositionEx".into(),Box::new(move || {
            count.store(unsafe {(*((raw as *mut c_void).cast::<Processor>())).refcount.load(Ordering::Acquire)},Ordering::Release);
        })));
        let result=unsafe {component_initialize(component,host.context())};
        let count=observed.load(Ordering::Acquire);assert_eq!(unsafe {component_release(component)},0);
        assert_eq!(result,K_RESULT_OK);assert_eq!(count,2,"宿主owning引用之外，本次初始化应独立保活插件存储");
    }

    /// terminate撤销授权与最终release不同；初始化不得把closed组件伪报成成功或重装host引用。
    #[test]
    fn task38a_initialize_reentrant_terminate_is_rejected_and_does_not_reinstall_host() {
        for terminate in [true,false] {
        let component=create_component();let host=crate::host::reaper::ReaperFixture::new();let raw=component as usize;
        let owner=unsafe {Arc::downgrade(&(*component.cast::<Processor>()).extension_owner)};
        *host.hook.borrow_mut()=Some(("api:GetPlayPositionEx".into(),Box::new(move || {
            if terminate {unsafe {component_terminate(raw as *mut c_void);}} else {owner.upgrade().unwrap().stop_editor();}
        })));
        let result=unsafe {component_initialize(component,host.context())};
        let closed=unsafe {(*component.cast::<Processor>()).extension_owner.is_closed()};
        let references=host.references();let calls=host.calls();assert_eq!(unsafe {component_release(component)},0);
        assert_eq!(result,K_RESULT_FALSE);assert!(closed);assert_eq!(references,1);
        assert_eq!(calls.last().unwrap(),"api:GetPlayPositionEx");
        }
    }

    /// 完成保活后才执行真正的重入最终release；调用结束不能泄漏该短引用。
    #[test]
    fn task38a_initialize_survives_reentrant_final_release_and_drops_its_short_reference() {
        let component=create_component();let host=crate::host::reaper::ReaperFixture::new();let raw=component as usize;
        let weak=unsafe {Arc::downgrade(&(*component.cast::<Processor>()).extension_owner)};let during=weak.clone();
        let remaining=Arc::new(AtomicU32::new(99));let observed=remaining.clone();
        let alive=Arc::new(std::sync::atomic::AtomicBool::new(false));let observed_alive=alive.clone();
        *host.hook.borrow_mut()=Some(("api:GetPlayPositionEx".into(),Box::new(move || {
            let count=unsafe {component_release(raw as *mut c_void)};
            observed.store(count,Ordering::Release);observed_alive.store(during.upgrade().is_some(),Ordering::Release);
        })));
        assert_eq!(unsafe {component_initialize(component,host.context())},K_RESULT_OK);
        assert_eq!(remaining.load(Ordering::Acquire),1);assert!(alive.load(Ordering::Acquire));
        assert!(weak.upgrade().is_none());assert_eq!(host.references(),1);
    }

    #[test]
    fn task38a_initialize_of_a_closed_component_does_not_call_host() {
        let component=create_component();let host=crate::host::reaper::ReaperFixture::new();
        unsafe {component_terminate(component);}host.reset();
        let result=unsafe {component_initialize(component,host.context())};
        let calls=host.calls();assert_eq!(unsafe {component_release(component)},0);
        assert_eq!(result,K_RESULT_FALSE);assert!(calls.is_empty());assert_eq!(host.references(),1);
    }

    /// IHostApplication QI itself可重入terminate；失败初始化不能留下后来才存入的host owning引用。
    #[test]
    fn task38a_initialize_reentry_during_host_application_qi_releases_the_returned_host_reference() {
        let component=create_component();let host=crate::host::reaper::ReaperFixture::new();host.enable_connection_host();
        let raw=component as usize;
        *host.hook.borrow_mut()=Some(("QI".into(),Box::new(move || {unsafe {component_terminate(raw as *mut c_void);}})));
        let result=unsafe {component_initialize(component,host.context())};let refs=host.references();let calls=host.calls();
        assert_eq!(unsafe {component_release(component)},0);assert_eq!(result,K_RESULT_FALSE);
        assert_eq!(refs,1,"关闭回调之后QI成功返回的引用也必须立即回收");assert_eq!(calls,["QI"]);
    }

    /// 工厂不能遗留自己的初始引用，否则组件和扩展永远不会释放。
    #[test]
    fn host_component_release_drops_the_extension_owner() {
        let component = create_component();
        // SAFETY: component 是工厂返回的有效 Processor 基址。
        let owner = unsafe { Arc::downgrade(&(*component.cast::<Processor>()).extension_owner) };
        // SAFETY: 消耗唯一宿主组件引用，此后不再使用 component。
        assert_eq!(unsafe { component_release(component) }, 0);
        assert!(owner.upgrade().is_none());
    }

    /// native ARA entry 被宿主持有时，组件先释放也不能让 extension storage 悬空。
    #[test]
    fn host_entry_reference_keeps_the_extension_owner_alive() {
        let component = create_component();
        // SAFETY: 组件仍存活，query_interface 返回新 owning COM 引用。
        let (owner, entry) = unsafe {
            let processor = &*component.cast::<Processor>();
            (Arc::downgrade(&processor.extension_owner), processor.entry.as_ref().unwrap()
                .query_interface(Ara2Vst3InterfaceKind::PluginEntry2).unwrap())
        };
        // SAFETY: 消耗宿主组件引用，entry 仍被单独持有。
        assert_eq!(unsafe { component_release(component) }, 0);
        assert!(owner.upgrade().is_some());
        let mut remaining = 0;
        // SAFETY: 最后消耗 entry owning COM 引用，不再访问其指针。
        assert_eq!(unsafe { ara2_vst3_release(entry, &raw mut remaining) }, ARA2_VST3_OK);
        assert_eq!(remaining, 0);
        assert!(owner.upgrade().is_none());
    }

    /// 真COM组件、真实route租约与native entry同时持有；释放入口不能牵连同文档/另一文档actor。
    #[test]
    fn task34_real_processor_release_revokes_routes_without_retaining_document() {
        use crate::editor::{routing::EditorLink,session::{UiRequest,UiSink}};
        use ara2_bridge::{core::ApiGeneration,plugin::ExtensionRoles};
        use std::sync::{mpsc,atomic::AtomicBool};
        let model=crate::ara::model::ModelHandle::new();let document=model.session();let weak_document=Arc::downgrade(&document);
        let other_model=crate::ara::model::ModelHandle::new();let other_document=other_model.session();
        let components=[create_component(),create_component(),create_component()];
        let mut links=Vec::new();let mut editors=Vec::new();let mut sinks=Vec::new();let mut receivers=Vec::new();
        for (index,component) in components.iter().enumerate() {
            // SAFETY: 三个工厂返回的COM owning引用在循环期间均存活。
            let processor=unsafe {&*component.cast::<Processor>()};
            processor.extension_owner.bind_to_document(if index==2 {other_document.clone()} else {document.clone()},
                ApiGeneration::V2Final,ExtensionRoles::all(),ExtensionRoles::EDITOR_RENDERER,None).unwrap();
            let link=Arc::new(EditorLink::default());link.bind(std::process::id() as i64,processor.route.token()).unwrap();
            let editor=processor.extension_owner.editor_session().unwrap();let (reply,rx)=mpsc::channel();let (events,_)=mpsc::sync_channel(8);
            let sink=UiSink {view_id:format!("task34-com-{index}"),reply,events,closed:Arc::new(AtomicBool::new(false))};
            editor.enqueue(UiRequest {id:1,command:"get_ui_settings".into(),args:serde_json::json!({}),sink:sink.clone(),link:Some(link.clone())}).unwrap();
            assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(3)).unwrap()["ok"],true);
            links.push(link);editors.push(editor);sinks.push(sink);receivers.push(rx);
        }
        assert!(Arc::ptr_eq(&editors[0],&editors[1]));assert!(!Arc::ptr_eq(&editors[0],&editors[2]));
        // ARA native entry保留owner，弱引用仍能升级也必须撤销processor入口。
        let (weak_owner,entry)=unsafe {let processor=&*components[0].cast::<Processor>();
            (Arc::downgrade(&processor.extension_owner),processor.entry.as_ref().unwrap().query_interface(Ara2Vst3InterfaceKind::PluginEntry2).unwrap())};
        assert_eq!(unsafe {component_release(components[0])},0);assert!(weak_owner.upgrade().is_some());
        assert!(links[0].owner().is_err());assert!(sinks[0].closed.load(Ordering::Acquire));assert!(!sinks[1].closed.load(Ordering::Acquire));
        assert!(editors[0].enqueue(UiRequest {id:2,command:"get_ui_settings".into(),args:serde_json::json!({}),sink:sinks[0].clone(),link:Some(links[0].clone())}).is_err());
        document.close();assert!(sinks[1].closed.load(Ordering::Acquire));
        assert!(links[1].owner().is_err());assert!(links[2].owner().is_ok());
        editors[2].enqueue(UiRequest {id:3,command:"get_ui_settings".into(),args:serde_json::json!({}),sink:sinks[2].clone(),link:Some(links[2].clone())}).unwrap();
        assert_eq!(receivers[2].recv_timeout(std::time::Duration::from_secs(3)).unwrap()["ok"],true);
        drop(document);drop(model);assert!(weak_document.upgrade().is_none(),"native entry/actor/route不能强持document");
        let mut remaining=0;assert_eq!(unsafe {ara2_vst3_release(entry,&raw mut remaining)},ARA2_VST3_OK);assert_eq!(remaining,0);
        assert!(weak_owner.upgrade().is_none());
        assert_eq!(unsafe {component_release(components[1])},0);assert_eq!(unsafe {component_release(components[2])},0);
        other_document.close();drop(other_document);drop(other_model);
    }

    /// 实际IComponent getState/setState经过IBStream ABI；短读写也必须保存共享actor最新值的组件范围。
    #[test]
    fn task34_real_component_state_stream_flushes_and_saves_only_its_scope() {
        use crate::editor::{routing::EditorLink,session::{UiRequest,UiSink}};
        use ara2_bridge::{core::ApiGeneration,plugin::ExtensionRoles};
        use std::sync::{mpsc,atomic::AtomicBool};
        #[repr(C)]
        struct Stream {vtable:*const Vtable,bytes:Vec<u8>,position:usize}
        #[repr(C)]
        struct Vtable {query:usize,add_ref:usize,release:usize,
            read:unsafe extern "system" fn(*mut c_void,*mut c_void,i32,*mut i32)->i32,
            write:unsafe extern "system" fn(*mut c_void,*mut c_void,i32,*mut i32)->i32,seek:usize,tell:usize}
        unsafe extern "system" fn read(this:*mut c_void,buffer:*mut c_void,count:i32,actual:*mut i32)->i32 {
            let stream=unsafe {&mut *this.cast::<Stream>()};let size=(count as usize).min(13).min(stream.bytes.len()-stream.position);
            unsafe {std::ptr::copy_nonoverlapping(stream.bytes.as_ptr().add(stream.position),buffer.cast(),size);*actual=size as i32;}
            stream.position+=size;K_RESULT_OK
        }
        unsafe extern "system" fn write(this:*mut c_void,buffer:*mut c_void,count:i32,actual:*mut i32)->i32 {
            let stream=unsafe {&mut *this.cast::<Stream>()};let size=(count as usize).min(11);
            stream.bytes.extend_from_slice(unsafe {std::slice::from_raw_parts(buffer.cast::<u8>(),size)});
            unsafe {*actual=size as i32;}K_RESULT_OK
        }
        let vtable=Vtable {query:0,add_ref:0,release:0,read,write,seek:0,tell:0};
        let (model,old_owners,ids)=crate::editor::session::tests::workspace_fixture();let document=model.session();
        for owner in old_owners {owner.stop_editor();}
        let components=[create_component(),create_component()];let mut links=Vec::new();
        for (index,component) in components.iter().enumerate() {
            // SAFETY: 工厂返回的processor COM owning引用仍由本测试持有。
            let processor=unsafe {&*component.cast::<Processor>()};
            let raw=processor.extension_owner.bind_to_document(document.clone(),ApiGeneration::V2Final,ExtensionRoles::all(),ExtensionRoles::PLAYBACK_RENDERER|ExtensionRoles::EDITOR_RENDERER,None).unwrap();
            let key=(&*ids[index] as *const u8) as u64;
            unsafe {let ext=&*raw;((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(ext.playbackRendererRef,key as *mut _);}
            let link=Arc::new(EditorLink::default());link.bind(std::process::id() as i64,processor.route.token()).unwrap();links.push(link);
        }
        let editor=document.editor_session().unwrap();let (reply,rx)=mpsc::channel();let (events,_)=mpsc::sync_channel(128);
        let sink=UiSink {view_id:"task34-state-stream".into(),reply,events,closed:Arc::new(AtomicBool::new(false))};
        let request=|id,command:&str,args| {
            editor.enqueue(UiRequest {id,command:command.into(),args,sink:sink.clone(),link:Some(links[0].clone())}).unwrap();
            let response=rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();assert_eq!(response["ok"],true,"{response}");response["value"].clone()
        };
        let timeline=request(1,"get_timeline_state",serde_json::json!({}));
        for (index,volume) in [0.5,0.25].into_iter().enumerate() {
            request(2,"set_track_state",serde_json::json!({"trackId":timeline["tracks"][index]["id"],"volume":volume}));
        }
        for (index,id) in ["track","b"].into_iter().enumerate() {
            let mut stream=Stream {vtable:&vtable,bytes:Vec::new(),position:0};
            assert_eq!(unsafe {(COMPONENT_VTBL.get_state)(components[index],(&raw mut stream).cast())},K_RESULT_OK);
            let length=u32::from_le_bytes(stream.bytes[..4].try_into().unwrap()) as usize;assert_eq!(stream.bytes.len(),length+4);
            let saved:serde_json::Value=serde_json::from_slice(&stream.bytes[4..]).unwrap();
            assert_eq!(saved["version"],2);assert_eq!(saved["edits"]["tracks"].as_array().unwrap().len(),1);
            assert_eq!(saved["edits"]["tracks"][0]["id"],id);assert_eq!(saved["edits"]["tracks"][0]["volume"],if index==0 {0.5} else {0.25});
            assert_eq!(saved["edits"]["bindings"].as_object().unwrap().len(),1);
            assert_eq!(unsafe {(COMPONENT_VTBL.set_state)(components[index],(&raw mut stream).cast())},K_RESULT_OK);
        }
        document.close();for component in components {assert_eq!(unsafe {component_release(component)},0);}
    }
}

#[cfg(test)]
mod audio_boundary_tests {
    use super::*;
    use crate::audio_abi::{AudioBusBuffers, ProcessData};

    /// SDK纯editor renderer透传宿主音频，不能把已含fade/gain的输入换成自己的旧快照。
    #[test]
    fn editor_only_renderer_preserves_host_fades_in_all_process_modes_without_allocations() {
        let model=crate::ara::model::ModelHandle::new();let mut processor=Processor::create().unwrap();
        processor.extension_owner.bind_to_document(model.session(),ara2_bridge::core::ApiGeneration::V2Final,
            ara2_bridge::plugin::ExtensionRoles::all(),ara2_bridge::plugin::ExtensionRoles::EDITOR_RENDERER,None).unwrap();
        processor.extension_owner.snapshots[0].publish(crate::render::snapshot::PlaybackSnapshot {
            sample_rate:44100,origin_sample:0,left:vec![0.9;4],right:vec![0.9;4],_reservation:None}).unwrap();
        for mode in [0,1,2] {
            let mut source_left=[0.0,0.1,0.2,0.3];let mut source_right=[0.3,0.2,0.1,0.0];
            let mut input_planes=[source_left.as_mut_ptr(),source_right.as_mut_ptr()];
            let mut input=AudioBusBuffers {num_channels:2,silence_flags:0,channel_buffers:input_planes.as_mut_ptr()};
            let mut left=[9.0;5];let mut right=[9.0;5];let mut output_planes=[left.as_mut_ptr(),right.as_mut_ptr()];
            let mut output=AudioBusBuffers {num_channels:2,silence_flags:3,channel_buffers:output_planes.as_mut_ptr()};
            // context可选/停播也必须透传；preview为空时不能凭播放门禁覆盖宿主输入。
            let mut data=ProcessData {process_mode:mode,num_samples:4,num_inputs:1,num_outputs:1,
                inputs:&raw mut input,outputs:&raw mut output,..Default::default()};
            crate::test_allocator::begin();
            let result=unsafe {(AUDIO_VTBL.process)(audio_ptr(&mut *processor),(&raw mut data).cast())};
            let allocations=crate::test_allocator::end();assert_eq!(result,K_RESULT_OK);assert_eq!(allocations,0);
            assert_eq!(left,[0.0,0.1,0.2,0.3,9.0]);assert_eq!(right,[0.3,0.2,0.1,0.0,9.0]);assert_eq!(output.silence_flags,0);
        }
    }

    /// 相同bus/相同plane的in-place不能先清零；silent或inactive输入则明确补零。
    #[test]
    fn editor_only_passthrough_handles_in_place_silence_and_inactive_planes() {
        let model=crate::ara::model::ModelHandle::new();let mut processor=Processor::create().unwrap();
        processor.extension_owner.bind_to_document(model.session(),ara2_bridge::core::ApiGeneration::V2Final,
            ara2_bridge::plugin::ExtensionRoles::all(),ara2_bridge::plugin::ExtensionRoles::EDITOR_RENDERER,None).unwrap();
        let mut left=[0.05,0.1,123.0];let mut right=[0.2,0.3,456.0];let mut planes=[left.as_mut_ptr(),right.as_mut_ptr()];
        let mut bus=AudioBusBuffers {num_channels:2,silence_flags:0,channel_buffers:planes.as_mut_ptr()};
        let mut data=ProcessData {num_samples:2,num_inputs:1,num_outputs:1,inputs:&raw mut bus,outputs:&raw mut bus,..Default::default()};
        assert_eq!(unsafe {(AUDIO_VTBL.process)(audio_ptr(&mut *processor),(&raw mut data).cast())},K_RESULT_OK);
        assert_eq!(left,[0.05,0.1,123.0]);assert_eq!(right,[0.2,0.3,456.0]);
        bus.silence_flags=1;planes[1]=std::ptr::null_mut();
        let mut out_left=[9.0;3];let mut out_right=[9.0;3];let mut out_planes=[out_left.as_mut_ptr(),out_right.as_mut_ptr()];
        let mut output=AudioBusBuffers {num_channels:2,silence_flags:0,channel_buffers:out_planes.as_mut_ptr()};data.inputs=&raw mut bus;data.outputs=&raw mut output;
        assert_eq!(unsafe {(AUDIO_VTBL.process)(audio_ptr(&mut *processor),(&raw mut data).cast())},K_RESULT_OK);
        assert_eq!(out_left,[0.0,0.0,9.0]);assert_eq!(out_right,[0.0,0.0,9.0]);assert_eq!(output.silence_flags,3);
        bus.channel_buffers=std::ptr::null_mut();out_left=[7.0;3];data.inputs=&raw mut bus;
        assert_eq!(unsafe {(AUDIO_VTBL.process)(audio_ptr(&mut *processor),(&raw mut data).cast())},K_INVALID_ARGUMENT);
        assert_eq!(out_left,[7.0;3]);
    }

    /// 真正 process 在有效快照下输出 PCM，且观察区间不分配或释放任何内存。
    #[test]
    fn process_outputs_the_snapshot_without_allocating_or_deallocating() {
        let mut processor=Processor::create().unwrap();
        processor.extension_owner.snapshots[0].publish(crate::render::snapshot::PlaybackSnapshot {
            sample_rate:44100,origin_sample:10,left:vec![0.1,0.2],right:vec![0.3,0.4],_reservation:None,
        }).unwrap();
        let mut left=[8.0_f32;3]; let mut right=[8.0_f32;3];
        let mut planes=[left.as_mut_ptr(),right.as_mut_ptr()];
        let mut bus=AudioBusBuffers {num_channels:2,silence_flags:0,channel_buffers:planes.as_mut_ptr()};
        let mut context=crate::audio_abi::ProcessContext {state:1<<1,sample_rate:44100.0,project_time_samples:11,..Default::default()};
        let mut data=ProcessData {num_samples:2,num_outputs:1,outputs:&raw mut bus,process_context:&raw mut context,..Default::default()};
        crate::test_allocator::begin();
        // SAFETY: 实际 processor 音频子对象和完整 SDK 缓冲都在回调期间存活。
        let result=unsafe {(AUDIO_VTBL.process)(audio_ptr(&mut *processor),(&raw mut data).cast())};
        let allocations=crate::test_allocator::end();
        assert_eq!(result,K_RESULT_OK);
        assert_eq!(left,[0.2,0.0,8.0]); assert_eq!(right,[0.4,0.0,8.0]);
        assert_eq!(bus.silence_flags,0);
        assert_eq!(allocations,0);
    }

    /// 真实SDK入口验证离线setup等完整后台发布；普通setup不受doc事务/worker阻塞。
    #[test]
    fn offline_setup_waits_for_current_snapshot_and_reprepares_after_edit_revision_changes() {
        use std::sync::mpsc;
        let (model,owners,_ids)=crate::editor::session::tests::workspace_fixture();let document=model.session();
        let pcm=document.edit_sources.lock().unwrap()["ara://source"].clone();document.sources.lock().unwrap().insert("ara://source".into(),pcm);
        let mut processor=Processor::create().unwrap();processor.extension_owner=owners[0].clone();
        // SAFETY: Box保留稳定处理器地址到离线setup测试线程join。
        let audio=unsafe {audio_ptr(&mut *processor)};let held=document.transaction.lock().unwrap();owners[0].prepare();
        let mut normal=ProcessSetup {process_mode:0,symbolic_sample_size:0,max_samples_per_block:1024,sample_rate:44100.};
        assert_eq!(unsafe {(AUDIO_VTBL.setup_processing)(audio,&raw mut normal)},K_RESULT_OK,"普通setup不能等待doc/推理");
        let pointer=audio as usize;let (done,rx)=mpsc::channel();
        let job=std::thread::spawn(move || {let mut setup=ProcessSetup {process_mode:2,symbolic_sample_size:0,max_samples_per_block:1024,sample_rate:44100.};
            // SAFETY: 主测试保留稳定Box/接口/owner到join；此线程模拟SDK的非实时setup调用。
            let result=unsafe {(AUDIO_VTBL.setup_processing)(pointer as *mut c_void,&raw mut setup)};done.send(result).unwrap();});
        assert!(rx.recv_timeout(std::time::Duration::from_millis(20)).is_err(),"不能在后台未完成时成功返回");drop(held);
        assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(3)).unwrap(),K_RESULT_OK);job.join().unwrap();
        assert!(owners[0].snapshots.iter().all(|snapshot|snapshot.is_ready()));
        {let _transaction=document.transaction.lock().unwrap();let mut edits=document.edits.lock().unwrap();edits.revision+=1;
            let mut track=document.timeline.lock().unwrap().as_ref().unwrap().tracks[0].clone();track.volume=0.5;edits.tracks=vec![track];}
        let mut setup=ProcessSetup {process_mode:2,symbolic_sample_size:0,max_samples_per_block:1024,sample_rate:44100.};
        assert_eq!(unsafe {(AUDIO_VTBL.setup_processing)(audio,&raw mut setup)},K_RESULT_OK);
        let mut left=[9_f32;5];let mut right=[9_f32;5];let mut planes=[left.as_mut_ptr(),right.as_mut_ptr()];
        let mut bus=AudioBusBuffers {num_channels:2,silence_flags:0,channel_buffers:planes.as_mut_ptr()};
        let mut context=crate::audio_abi::ProcessContext {sample_rate:44100.,..Default::default()};
        let mut data=ProcessData {process_mode:2,num_samples:4,num_outputs:1,outputs:&raw mut bus,process_context:&raw mut context,..Default::default()};
        assert_eq!(unsafe {(AUDIO_VTBL.process)(audio,(&raw mut data).cast())},K_RESULT_OK);
        document.close();
        for (actual,want) in left[..4].iter().zip([0.05,0.1,0.15,0.2]) {assert!((*actual-want).abs()<1e-6);}
        assert_eq!(left,right);assert_eq!(left[4],9.);
    }
    /// 离线缺快照/上下文必须显式失败；实时仍安全静音，全部回调保持零分配。
    #[test]
    fn offline_missing_snapshot_is_failure_not_successful_silence() {
        let mut processor=Processor::create().unwrap();let mut left=[9_f32;4];let mut right=[9_f32;4];
        let mut planes=[left.as_mut_ptr(),right.as_mut_ptr()];let mut bus=AudioBusBuffers {num_channels:2,silence_flags:0,channel_buffers:planes.as_mut_ptr()};
        let mut context=crate::audio_abi::ProcessContext {sample_rate:44100.,..Default::default()};
        let mut data=ProcessData {process_mode:2,num_samples:4,num_outputs:1,outputs:&raw mut bus,process_context:&raw mut context,..Default::default()};
        crate::test_allocator::begin();let missing=unsafe {(AUDIO_VTBL.process)(audio_ptr(&mut *processor),(&raw mut data).cast())};
        data.process_context=std::ptr::null_mut();let no_context=unsafe {(AUDIO_VTBL.process)(audio_ptr(&mut *processor),(&raw mut data).cast())};
        data.process_mode=0;let realtime=unsafe {(AUDIO_VTBL.process)(audio_ptr(&mut *processor),(&raw mut data).cast())};
        let allocations=crate::test_allocator::end();assert_eq!((missing,no_context,realtime),(K_RESULT_FALSE,K_RESULT_FALSE,K_RESULT_OK));
        assert_eq!(allocations,0);assert_eq!(left,[0.;4]);assert_eq!(right,[0.;4]);
    }
    /// 用真实接口子对象调用 process，避免测试绕过产品 vtable。
    fn run(data: *mut ProcessData) -> TResult {
        let mut processor = Processor::create().unwrap();
        // SAFETY: 本测试持有处理器与 SDK 同布局数据，调用期间缓冲存活。
        unsafe { (AUDIO_VTBL.process)(audio_ptr(&mut *processor), data.cast()) }
    }

    /// 停播重复process同一帧必须静音；离线导出没有kPlaying也必须保留音频。
    #[test]
    fn idle_transport_is_silent_but_offline_export_still_reads_snapshot() {
        let mut processor=Processor::create().unwrap();
        processor.extension_owner.snapshots[0].publish(crate::render::snapshot::PlaybackSnapshot {
            sample_rate:44100,origin_sample:0,left:vec![0.25;4],right:vec![0.5;4],_reservation:None,
        }).unwrap();
        let mut left=[9_f32;4];let mut right=[9_f32;4];let mut planes=[left.as_mut_ptr(),right.as_mut_ptr()];
        let mut bus=AudioBusBuffers {num_channels:2,silence_flags:0,channel_buffers:planes.as_mut_ptr()};
        let mut context=crate::audio_abi::ProcessContext {sample_rate:44100.,..Default::default()};
        let mut data=ProcessData {num_samples:4,num_outputs:1,outputs:&raw mut bus,process_context:&raw mut context,..Default::default()};
        for _ in 0..3 {
            assert_eq!(unsafe {(AUDIO_VTBL.process)(audio_ptr(&mut *processor),(&raw mut data).cast())},K_RESULT_OK);
            assert_eq!(left,[0.;4],"停播不能循环输出光标位置的快照块");assert_eq!(right,[0.;4]);
        }
        data.process_mode=2;
        assert_eq!(unsafe {(AUDIO_VTBL.process)(audio_ptr(&mut *processor),(&raw mut data).cast())},K_RESULT_OK);
        assert_eq!(left,[0.25;4]);assert_eq!(right,[0.5;4]);
    }

    /// 空实现留下旧音频；写错块长则越界覆盖尾哨兵。
    #[test]
    fn process_clears_exactly_the_requested_frames() {
        let mut left = [0.75_f32, 0.75, 123.0];
        let mut right = [-0.5_f32, -0.5, 456.0];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut output = AudioBusBuffers {
            num_channels: 2, silence_flags: 0, channel_buffers: planes.as_mut_ptr(),
        };
        let mut data = ProcessData {
            num_samples: 2, num_outputs: 1, outputs: &raw mut output, ..Default::default()
        };
        assert_eq!(run(&raw mut data), K_RESULT_OK);
        assert_eq!(left, [0.0, 0.0, 123.0]);
        assert_eq!(right, [0.0, 0.0, 456.0]);
        assert_eq!(output.silence_flags, 3);
    }

    /// 非法宿主数据不能被空实现伪报成功。
    #[test]
    fn process_rejects_invalid_counts_and_missing_bus_storage() {
        assert_eq!(run(std::ptr::null_mut()), K_INVALID_ARGUMENT);
        for data in [
            ProcessData { num_samples: -1, ..Default::default() },
            ProcessData { num_outputs: -1, ..Default::default() },
            ProcessData { num_inputs: -1, ..Default::default() },
            ProcessData { num_samples: 1, num_outputs: 1, ..Default::default() },
            ProcessData { num_samples: 1, num_inputs: 1, ..Default::default() },
        ] {
            let mut data = data;
            assert_eq!(run(&raw mut data), K_INVALID_ARGUMENT);
        }
    }

    /// 零帧参数 flush 与无总线合法，不应解引用空地址。
    #[test]
    fn process_accepts_zero_frame_flush_and_no_output() {
        let mut flush = ProcessData { num_outputs: 1, ..Default::default() };
        assert_eq!(run(&raw mut flush), K_RESULT_OK);
        let mut empty = ProcessData { num_samples: 32, ..Default::default() };
        assert_eq!(run(&raw mut empty), K_RESULT_OK);
    }

    /// 64 位样本不支持时必须拒绝，不能按 f32 写出破坏的样本。
    #[test]
    fn process_rejects_sample64_without_writing() {
        let mut samples = [0.25_f64, 0.5];
        let mut planes = [samples.as_mut_ptr(), samples.as_mut_ptr()];
        let mut output = AudioBusBuffers {
            num_channels: 2, silence_flags: 0, channel_buffers: planes.as_mut_ptr().cast(),
        };
        let mut data = ProcessData {
            symbolic_sample_size: 1, num_samples: 2, num_outputs: 1,
            outputs: &raw mut output, ..Default::default()
        };
        assert_eq!(run(&raw mut data), K_RESULT_FALSE);
        assert_eq!(samples, [0.25, 0.5]);
    }

    /// SDK 允许 inactive plane 为 null，但有通道时 plane 数组不能缺失。
    #[test]
    fn process_skips_inactive_planes_but_rejects_missing_arrays() {
        let mut samples = [1.0_f32, 2.0];
        let mut planes = [samples.as_mut_ptr(), std::ptr::null_mut()];
        let mut output = AudioBusBuffers {
            num_channels: 2, silence_flags: 0, channel_buffers: planes.as_mut_ptr(),
        };
        let mut data = ProcessData {
            num_samples: 2, num_outputs: 1, outputs: &raw mut output, ..Default::default()
        };
        assert_eq!(run(&raw mut data), K_RESULT_OK);
        assert_eq!(samples, [0.0, 0.0]);
        assert_eq!(output.silence_flags, 3);
        output.channel_buffers = std::ptr::null_mut();
        data.outputs = &raw mut output;
        assert_eq!(run(&raw mut data), K_INVALID_ARGUMENT);
    }

    /// 所有输入形状先校验，非法输入不能导致输出缓冲被部分写入。
    #[test]
    fn process_rejects_invalid_input_before_touching_output() {
        let mut input_planes = [std::ptr::null_mut(), std::ptr::null_mut()];
        for (channels, missing_array, expected) in [
            (-1, false, K_INVALID_ARGUMENT),
            (1, false, K_RESULT_FALSE),
            (2, true, K_INVALID_ARGUMENT),
            (2, false, K_RESULT_OK),
            (0, true, K_RESULT_OK),
        ] {
            let mut left = [0.25_f32, 0.5];
            let mut right = [0.75_f32, 1.0];
            let mut output_planes = [left.as_mut_ptr(), right.as_mut_ptr()];
            let mut input = AudioBusBuffers {
                num_channels: channels, silence_flags: 0,
                channel_buffers: if missing_array { std::ptr::null_mut() } else { input_planes.as_mut_ptr() },
            };
            let mut output = AudioBusBuffers {
                num_channels: 2, silence_flags: 0, channel_buffers: output_planes.as_mut_ptr(),
            };
            let mut data = ProcessData {
                num_samples: 2, num_inputs: 1, num_outputs: 1,
                inputs: &raw mut input, outputs: &raw mut output, ..Default::default()
            };
            assert_eq!(run(&raw mut data), expected);
            if expected != K_RESULT_OK {
                assert_eq!(left, [0.25, 0.5]);
                assert_eq!(right, [0.75, 1.0]);
                assert_eq!(output.silence_flags, 0);
            } else {
                assert_eq!(left, [0.0, 0.0]);
                assert_eq!(right, [0.0, 0.0]);
            }
        }
    }

    /// 未实现多总线或非 stereo 时不能假装完成协商。
    #[test]
    fn bus_negotiation_rejects_unsupported_arrangements() {
        let mut processor = Processor::create().unwrap();
        let mut stereo = 3_u64;
        let mut mono = 1_u64;
        // SAFETY: 真实处理器和布局数组在协商期间存活。
        unsafe {
            let this = audio_ptr(&mut *processor);
            assert_eq!(audio_set_bus_arrangements(this, &raw mut stereo, 1, &raw mut stereo, 1), K_RESULT_OK);
            assert_eq!(audio_set_bus_arrangements(this, &raw mut mono, 1, &raw mut stereo, 1), K_RESULT_FALSE);
            assert_eq!(audio_set_bus_arrangements(this, &raw mut stereo, 0, &raw mut stereo, 1), K_RESULT_FALSE);
            assert_eq!(audio_set_bus_arrangements(this, std::ptr::null_mut(), 1, &raw mut stereo, 1), K_INVALID_ARGUMENT);
        }
    }

    /// 非法 setup 不能让后续回调使用无效采样率或块尺寸。
    #[test]
    fn setup_rejects_invalid_rate_block_and_format() {
        let mut processor = Processor::create().unwrap();
        // SAFETY: 真实处理器和 setup POD 在调用期间存活。
        unsafe {
            let this = audio_ptr(&mut *processor);
            assert_eq!(audio_setup_processing(this, std::ptr::null_mut()), K_INVALID_ARGUMENT);
            for (rate, block, format, expected) in [
                (44100.0, 512, 0, K_RESULT_OK),
                (f64::NAN, 512, 0, K_INVALID_ARGUMENT),
                (0.0, 512, 0, K_INVALID_ARGUMENT),
                (44100.0, -1, 0, K_INVALID_ARGUMENT),
                (44100.0, 512, 1, K_RESULT_FALSE),
                (96000.0, 512, 0, K_RESULT_FALSE),
            ] {
                let mut setup = ProcessSetup {
                    process_mode: 0, symbolic_sample_size: format,
                    max_samples_per_block: block, sample_rate: rate,
                };
                assert_eq!(audio_setup_processing(this, &raw mut setup), expected);
            }
        }
    }
}

#[cfg(test)]
mod realtime_tests {
    use super::*;
    use std::cell::Cell;

    std::thread_local! {
        static IN_REALTIME_CALLBACK: Cell<bool> = const { Cell::new(false) };
        static REALTIME_LOGS: Cell<usize> = const { Cell::new(0) };
    }

    struct ObservingLogger;

    impl log::Log for ObservingLogger {
        fn enabled(&self, _: &log::Metadata<'_>) -> bool { true }
        fn log(&self, _: &log::Record<'_>) {
            if IN_REALTIME_CALLBACK.with(Cell::get) {
                REALTIME_LOGS.with(|count| count.set(count.get() + 1));
            }
        }
        fn flush(&self) {}
    }

    /// setProcessing 和 process 都可在音频线程运行，不能触发同步文件日志。
    #[test]
    fn realtime_callbacks_do_not_enter_the_file_logger() {
        static LOGGER: ObservingLogger = ObservingLogger;
        log::set_logger(&LOGGER).unwrap();
        log::set_max_level(log::LevelFilter::Info);
        let mut processor = Processor::create().expect("processor creation");
        // SAFETY: 取真实音频处理器子对象地址，而非组件基址。
        let this = unsafe { audio_ptr(&mut *processor) };
        let mut left = [1.0_f32; 2];
        let mut right = [1.0_f32; 2];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = crate::audio_abi::AudioBusBuffers {
            num_channels: 2, silence_flags: 0, channel_buffers: planes.as_mut_ptr(),
        };
        let mut data = crate::audio_abi::ProcessData {
            num_samples: 2, num_outputs: 1, outputs: &raw mut bus, ..Default::default()
        };
        IN_REALTIME_CALLBACK.with(|flag| flag.set(true));
        // SAFETY: 处理器在两次状态回调期间存活。
        unsafe {
            assert_eq!(audio_set_processing(this, 1), K_RESULT_OK);
            assert_eq!((AUDIO_VTBL.process)(this, (&raw mut data).cast()), K_RESULT_OK);
            assert_eq!(audio_set_processing(this, 0), K_RESULT_OK);
        }
        IN_REALTIME_CALLBACK.with(|flag| flag.set(false));
        assert_eq!(REALTIME_LOGS.with(Cell::get), 0);
        assert_eq!(left, [0.0; 2]);
        assert_eq!(right, [0.0; 2]);
    }
}
