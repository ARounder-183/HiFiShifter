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
use ara2_bridge::companion::CompanionRoles;
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

/// 处理器 / 控制器 / ARA 工厂的类 ID。
///
/// 【为什么是 GUID 而不是 16 个 ASCII 字符】ASCII 形式（如 `HFSARAProbe.Prc1`）看着像
/// 名字，实际只是 16 个字节：谁"顺手改一个字符"ID 就变了，却看不出这是一次身份变更。
/// GUID 有明确的字形与生成方式，能对着工具核对；编码沿用本文件的 [`uid_guid`]
/// （Windows VST3 的 GUID 内存布局，见该函数的三段注释与实测依据）。
///
/// 【这三个值一旦发布就再也不能改】它们写在宿主工程里，换掉等于让所有已存在的实例
/// 认不出来（表现为"插件丢失"）。当前仍是开发期（从未分发），因此这一轮把探针时代的
/// 名字一并换掉；此后的任何改动都必须视为破坏性变更。
const PROCESSOR_CID: Tuid = uid_guid([0x9C3A_4F17, 0x1E8B, 0x4D02, 0xB7A1_3C6E]);
/// 编辑器控制器类 ID。
const CONTROLLER_CID: Tuid = uid_guid([0x9C3A_4F18, 0x1E8B, 0x4D02, 0xB7A1_3C6E]);
/// ARA 主工厂类 ID。
const FACTORY_CID: Tuid = uid_guid([0x9C3A_4F19, 0x1E8B, 0x4D02, 0xB7A1_3C6E]);

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
/// `IPluginFactory3`（`ipluginbase.h`）。它只比 `IPluginFactory2` 多一个
/// `getClassInfoUnicode`，但 Steinberg 自家宿主（Cubase / Nuendo）会**主动查询**它：
/// 实测日志里 `factory queryInterface miss iid=ABA2554523C1574E9B12291036878931` 就是
/// 它。此前我们回 `K_NO_INTERFACE`，等于对宿主说"这个工厂没有你认识的那版接口"。
const IID_IPLUGIN_FACTORY3: [u32; 4] = [0x4555_A2AB, 0xC123_4E57, 0x9B12_2910, 0x3687_8931];
const IID_ICOMPONENT: [u32; 4] = [0xE831_FF31, 0xF2D5_4301, 0x928E_BBEE, 0x2569_7802];
const IID_IAUDIO_PROCESSOR: [u32; 4] = [0x4204_3F99, 0xB7DA_453C, 0xA569_E79D, 0x9AAE_C33D];
const IID_IEDIT_CONTROLLER: [u32; 4] = [0xDCD7_BBE3, 0x7742_448D, 0xA874_AACC, 0x979C_759E];

/// `Vst::IProcessContextRequirements`（`ivstaudioprocessor.h`，VST 3.7 起）。
///
/// 【为什么这个 IID 是必需的，而不是"锦上添花"】SDK 对该接口的原文：
/// "To get accurate process context information (Vst::ProcessContext), it is now required
/// to implement this interface and return the desired bit mask of flags which your audio
/// effect needs. **If you do not implement this interface, you may not get any information
/// at all of the process function!**" 此前插件从不实现它，却依赖 `ProcessContext` 的
/// `kPlaying` 与 `projectTimeSamples` —— 那是在赌宿主发一件我们没声明要的东西。
/// 3.7 之前的 SDK 会无条件发全量，所以旧行为在旧宿主上"碰巧能用"。
const IID_IPROCESS_CONTEXT_REQUIREMENTS: [u32; 4] =
    [0x2A654303, 0xEF764E3D, 0x95B5FE83, 0x730EF6D0];

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

/// 记录一次"查询未命中"，同一接口每 10 秒最多一条。
///
/// 【为什么要限流】宿主会成批查询它认识的每一个接口（Cubase 一次会话里就有几十条），
/// 全部记 INFO 会把真正有用的行冲掉。**首次仍然记** —— 未命中的接口清单是判断"宿主
/// 想用什么"的第一手证据（例如 Cubase 查询 `IPluginFactory3` 而未命中）；重复项交给
/// 内核限流器汇总成 `[throttled] N message(s) suppressed`，信息不丢，只是不重复。
///
/// 【为什么按接口分桶】不同接口的未命中是不同的事实。用一个粗桶会让"新出现的未命中"
/// 被已有噪音吞掉 —— 那正是需要看见的时刻。
fn log_query_interface_miss(owner: &str, iid: *const u8) {
    // SAFETY: 调用方保证 iid 指向可读的 16 字节 TUID（或为空）。
    let iid = unsafe { hex_iid(iid) };
    hifishifter_kernel::log_limiter::emit_limited_key(
        log::Level::Info,
        &format!("vst3-qi-miss:{owner}:{iid}"),
        format_args!("[vst3] {owner} queryInterface miss iid={iid}"),
    );
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
    /// 子类别（128 字节）。当前是 `"Fx"`（见 `class_declaration` 的说明）。
    pub sub_categories: [c_char; 128],
    /// 厂商覆盖（64 字节）。
    pub vendor: [c_char; 64],
    /// 版本（64 字节）。
    pub version: [c_char; 64],
    /// SDK 版本（64 字节）。
    pub sdk_version: [c_char; 64],
}

/// `PClassInfoW`（`IPluginFactory3::getClassInfoUnicode`）。
///
/// 与 [`PClassInfo2`] 同形，差别只在 `name` / `vendor` / `version` / `sdk_version`
/// 是 UTF-16（`char16`）。
#[repr(C)]
pub struct PClassInfoW {
    /// 类 ID。
    pub cid: Tuid,
    /// 基数。
    pub cardinality: i32,
    /// 类类别。
    pub category: [c_char; 32],
    /// 类名（UTF-16，64 个 char16）。
    pub name: [u16; 64],
    /// 类别标志。
    pub class_flags: u32,
    /// 子类别（128 字节）。当前是 `"Fx"`（见 `class_declaration` 的说明）。
    pub sub_categories: [c_char; 128],
    /// 厂商覆盖（UTF-16）。
    pub vendor: [u16; 64],
    /// 版本（UTF-16）。
    pub version: [u16; 64],
    /// SDK 版本（UTF-16）。
    pub sdk_version: [u16; 64],
}

/// 一个 VST3 类的声明内容 —— `getClassInfo` / `getClassInfo2` /
/// `getClassInfoUnicode` 三个入口共用**同一份事实**。
///
/// 【为什么必须共用】ARA 契约要求处理器类的 `PClassInfo.name` 与
/// `ARAFactory.plugInName` 一致，宿主据此把 ARA 主工厂与音频处理器配成一对
/// （见 `ARAVST3.h` 的 `IMainFactory` 说明）。三个入口各写一份字面量时，任何一处
/// 漂移都会让宿主配不上对 —— 而症状是"插件能加载但从不进入 ARA"，极难定位。
struct ClassDeclaration {
    cid: Tuid,
    category: String,
    name: &'static str,
    class_flags: u32,
    sub_categories: &'static str,
    vendor: &'static str,
    version: &'static str,
    sdk_version: &'static str,
}

/// 类别索引 → 声明内容；越界返回 `None`（宿主只该问 0..=2）。
///
/// 索引与 `factory_count_classes` 的注释一致：0 = 音频处理器组件，1 = ARA 主工厂，
/// 2 = 编辑器控制器。
fn class_declaration(index: i32) -> Option<ClassDeclaration> {
    /// 所有类共用的 SDK 版本串。
    const SDK_VERSION: &str = "VST 3.8.0";
    const VENDOR: &str = "HiFiShifter";

    let (cid, category, sub_categories, class_flags) = match index {
        0 => (
            PROCESSOR_CID,
            AUDIO_MODULE_CLASS.to_owned(),
            // 【为什么不再是 `Fx|OnlyARA`】`kOnlyARA` 的语义是"必须 ARA 才能工作，
            // 作为普通插入插件不工作"（`ivstaudioprocessor.h`）。这在**当时**是事实：
            // 非 ARA 插入时没有文档、没有授权 PCM，`audio_process` 只能输出静音。
            //
            // 现在它不再是事实 —— 插件有了 **Capture** 通路：在非 ARA 实例上，
            // 音频从 VST3 输入总线实时（或离线）捕获，按 `ProcessContext` 的
            // `projectTimeSamples` 对到插件自己的时间轴上，再照常处理并输出。
            // 于是"作为普通插入插件"是**真的能工作**的，继续声明 `OnlyARA`
            // 会变成一句会误导宿主的谎：多数宿主据此不把插件列进普通 FX 列表，
            // 用户连试都试不了。
            //
            // 【为什么绝不能顺手加 `kNoOfflineProcess`】bounce / 导出是 Capture 的
            // 第二条（且质量最好的）通路：宿主以 `kOffline` 快于实时地跑完整个区间。
            // 声明"我不参与离线处理"等于自断一臂。
            //
            // 【与 ARA 配对的关系】ARA 契约要求处理器类名与 `ARAFactory.plugInName`
            // 一致（见 `ClassDeclaration` 的说明）；subCategories 不参与那个配对，
            // 改它不会让 ARA 入口失效。
            "Fx",
            0,
        ),
        1 => {
            // 主工厂的类别串由适配器给出（与 shim 的常量同源）；取不到时退回规范字面量。
            let mut buffer = [0 as c_char; 32];
            // SAFETY: 适配器返回常驻 NUL 结尾 C 字符串（或 null），缓冲足够。
            unsafe { copy_str_ptr(&mut buffer, ara2_vst3_main_factory_category()) };
            let category = if buffer[0] == 0 {
                "ARA Main Factory Class".to_owned()
            } else {
                let bytes = buffer
                    .iter()
                    .take_while(|byte| **byte != 0)
                    .map(|byte| *byte as u8)
                    .collect::<Vec<u8>>();
                String::from_utf8_lossy(&bytes).into_owned()
            };
            (FACTORY_CID, category, "", 0)
        }
        2 => (CONTROLLER_CID, COMPONENT_CONTROLLER_CLASS.to_owned(), "", 0),
        _ => return None,
    };
    Some(ClassDeclaration {
        cid,
        category,
        name: crate::CLASS_NAME,
        class_flags,
        sub_categories,
        vendor: VENDOR,
        version: crate::VERSION,
        sdk_version: SDK_VERSION,
    })
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

/// 把 `&str` 按 UTF-16 写入定长 `char16` 缓冲（`PClassInfoW` 的字段用）。
///
/// 与 [`copy_str`] 同一纪律：只填前 N 个码元，剩余留给调用方清零的尾部。
fn copy_wide_str<const N: usize>(dst: &mut [u16; N], src: &str) {
    for (slot, unit) in dst.iter_mut().zip(src.encode_utf16()) {
        *slot = unit;
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
    /// `IPluginFactory3::getClassInfoUnicode`。
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
            // 接受的接口 → 名字。**命中与未命中都记**：未命中的清单是判断"宿主想用
            // 什么"的第一手证据，命中的则回答"宿主拿走了哪一版工厂接口"。
            let accepted = [
                (IID_FUNKNOWN, "FUnknown"),
                (IID_IPLUGIN_FACTORY, "IPluginFactory"),
                (IID_IPLUGIN_FACTORY2, "IPluginFactory2"),
                (IID_IPLUGIN_FACTORY3, "IPluginFactory3"),
            ];
            let Some(name) = accepted
                .iter()
                .find(|(words, _)| unsafe { iid_matches(iid, *words) })
                .map(|(_, name)| *name)
            else {
                log_query_interface_miss("factory", iid);
                return K_NO_INTERFACE;
            };
            let factory = this as *const PluginFactoryObj;
            // SAFETY: this 是形如 PluginFactoryObj 的常驻单例。
            unsafe { (*factory).refcount.fetch_add(1, Ordering::AcqRel) };
            // SAFETY: 输出槽非空。
            unsafe { *obj = this };
            crate::log_line(&format!("factory queryInterface hit {name}"));
            K_RESULT_OK
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
    // `PClassInfo` 是精简版：只有 cid / cardinality / category / name。
    let Some(declaration) = class_declaration(index) else {
        return;
    };
    info.cid = declaration.cid;
    copy_str(&mut info.category, &declaration.category);
    copy_str(&mut info.name, declaration.name);
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
            let Some(declaration) = class_declaration(index) else {
                return K_INVALID_ARGUMENT;
            };
            info.cid = declaration.cid;
            copy_str(&mut info.category, &declaration.category);
            copy_str(&mut info.name, declaration.name);
            // 二期编辑会话在宿主本进程，不宣称跨进程 processor/controller 分布。
            info.class_flags = declaration.class_flags;
            copy_str(&mut info.sub_categories, declaration.sub_categories);
            copy_str(&mut info.vendor, declaration.vendor);
            copy_str(&mut info.version, declaration.version);
            copy_str(&mut info.sdk_version, declaration.sdk_version);
            K_RESULT_OK
        })
    }
}

/// `IPluginFactory3::getClassInfoUnicode`。
///
/// 【为什么必须真的实现】Steinberg 自家宿主会**主动查询** `IPluginFactory3`（实测
/// Cubase 日志里 `factory queryInterface miss iid=ABA25545…`）。此前这个入口回
/// `K_NOT_IMPLEMENTED`，且 `factory_query_interface` 也不接受该接口 —— 对宿主而言
/// 就是"这个工厂没有你认识的那版接口"。名字走 UTF-16 是这套接口存在的唯一理由，
/// 因此内容与 [`factory_get_class_info2`] **同源**（见 [`class_declaration`]），
/// 不另写一份字面量。
unsafe extern "system" fn factory_get_class_info_unicode(
    _this: *mut c_void,
    index: i32,
    info: *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            if info.is_null() || !(0..=2).contains(&index) {
                return K_INVALID_ARGUMENT;
            }
            // SAFETY: info 是宿主提供的可写 PClassInfoW。
            let info = unsafe { &mut *info.cast::<PClassInfoW>() };
            // SAFETY: POD 清零。
            unsafe { std::ptr::write_bytes(info as *mut PClassInfoW, 0, 1) };
            info.cardinality = K_MANY_INSTANCES;
            let Some(declaration) = class_declaration(index) else {
                return K_INVALID_ARGUMENT;
            };
            info.cid = declaration.cid;
            copy_str(&mut info.category, &declaration.category);
            copy_wide_str(&mut info.name, declaration.name);
            info.class_flags = declaration.class_flags;
            copy_str(&mut info.sub_categories, declaration.sub_categories);
            copy_wide_str(&mut info.vendor, declaration.vendor);
            copy_wide_str(&mut info.version, declaration.version);
            copy_wide_str(&mut info.sdk_version, declaration.sdk_version);
            crate::log_line(&format!("factory getClassInfoUnicode({index}) ok"));
            K_RESULT_OK
        })
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
                // VST3 契约：`createInstance` 只交付宿主**确实请求**的接口；请求的 IID
                // 不受支持时必须销毁刚建的对象并回 `kNoInterface`。此前这里无条件交出
                // 主接口 —— 宿主若问的是别的接口，就会拿着**错误的 vtable** 去调用方法，
                // 而故障点会落在宿主的代码里（实测 Cubase 15 在 `araservice.dll` 内
                // 空指针写入崩溃，我们的模块既不在栈上、也不在崩溃进程的模块表里，
                // 正是"宿主用了不属于它的接口"这类问题的典型形态）。
                //
                // 适配器支持的两个接口：`ARA::IMainFactory`（自身）与 `FUnknown`（基类，
                // 同一对象基址）。两者都直接把主接口交出去。
                let supported = unsafe { iid_matches(iid as *const u8, IID_FUNKNOWN) }
                    || ara_iid_words(Ara2Vst3InterfaceKind::MainFactory)
                        .is_some_and(|words| unsafe { iid_matches(iid as *const u8, words) });
                if !supported {
                    // `adapter` 在此 drop：消耗它唯一的 COM 引用，宿主拿不到对象。
                    crate::log_line("factory createInstance ARA main factory: unsupported iid");
                    return K_NO_INTERFACE;
                }
                // 【为什么记这一行】创建 ARA 主工厂是宿主**明确要走 ARA 路径**的第一个
                // 信号（Cubase 实测：加载前创建了它两次）。把它与"之后是否真的绑定了
                // 文档控制器"放在同一条时间线上，一眼就能看出宿主走到哪一步停住。
                crate::log_line("factory createInstance ARA main factory");
                let raw = adapter.as_raw();
                // 交付成功：把适配器泄漏给宿主管理引用计数，直到最终 release。
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

/// `Vst::IProcessContextRequirements` 的 vtbl。
///
/// 【为什么是**独立子对象**而不是给 `AudioProcessorVtbl` 追加第 12 个槽】
/// SDK 的原声明是 `class IProcessContextRequirements : public FUnknown` ——
/// 它**不**继承 `IAudioProcessor`（对比 `IComponentHandler2 : public FUnknown`、
/// `IComponentHandler3 : public FUnknown`，这是 VST3 对"扩展接口"的一贯写法）。
/// 因此它的 vtable 只有 4 个槽：`FUnknown` 的 3 个 + `getProcessContextRequirements`。
/// 若误按"继承 IAudioProcessor"去追加第 12 槽，宿主会在**槽 3** 上调
/// `getProcessContextRequirements`，而那里是 `setBusArrangements` —— 宿主会拿着
/// 一个返回值当位掩码用。这类错误在宿主里表现为随机的图优化异常，极难归因。
///
/// 返回指针沿用 `connection_vtbl` 的既有手法：指向对象内嵌的 vtbl 字段，
/// 由 `base_from_context_requirements` 反解回 `Processor`。
#[repr(C)]
pub struct ProcessContextRequirementsVtbl {
    pub base: crate::editor::connection::UnknownVtbl,
    /// `IProcessContextRequirements::getProcessContextRequirements`。
    pub get_process_context_requirements: unsafe extern "system" fn(*mut c_void) -> u32,
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
    /// `IProcessContextRequirements` 子对象（见该 vtbl 的说明）。
    ///
    /// 【为什么每个实例一份而不是共享静态】子对象身份必须与拥有者一一对应：
    /// 宿主可能对同一模块的多个实例各查一次，而 `queryInterface` 返回的指针
    /// 要能被反解回**正确的那个** `Processor`。共享一个静态对象会让两个实例
    /// 拿到同一个地址。
    context_requirements_vtbl: *const ProcessContextRequirementsVtbl,
    route: crate::editor::routing::RouteLease,
}

impl Processor {
    /// 建立处理器：为每个实例单独建一个 companion 处理器绑定 + ARA 入口适配器，
    /// 这样多个实例不会互相抢同一次绑定。
    fn create() -> Option<Box<Processor>> {
        let runtime = crate::runtime::runtime()?;
        let extension_owner = Arc::new(ExtensionOwner::default());
        let route = crate::editor::routing::RouteLease::new(&extension_owner);
        // 【为什么把 factory/roles 交给 HostEntry】companion 的 "already bound" 是
        // 处理器级的：宿主重开工程会复用同一个 `Processor`，而上一份文档关闭后
        // 这个一次性标志仍然为真。`HostEntry` 必须在同一个位置重建一个等价绑定，
        // 所以它需要原始构造参数，而不是只拿一次创建好的对象。
        let entry = HostEntry::new(
            runtime.companion.clone(),
            CompanionRoles::all(),
            crate::CLASS_NAME,
            extension_owner.clone(),
        )
        .ok()?;
        Some(Box::new(Processor {
            component_vtbl: &COMPONENT_VTBL,
            refcount: AtomicI32::new(1),
            audio_vtbl: &AUDIO_VTBL,
            entry: Some(entry),
            extension_owner,
            active: AtomicI32::new(0),
            connection_vtbl: &PROCESSOR_CONNECTION_VTBL,
            connection: Default::default(),
            context_requirements_vtbl: &PROCESS_CONTEXT_REQUIREMENTS_VTBL,
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
            // `IProcessContextRequirements`（VST 3.7 起 mandatory）。宿主在 Setup Done 时
            // 查一次并缓存，因此这里必须**先于**通用未命中日志返回，否则日志会把它
            // 记成"宿主查了个我们不认识的接口"。
            if unsafe { iid_matches(iid, IID_IPROCESS_CONTEXT_REQUIREMENTS) } {
                crate::log_line("component queryInterface hit IProcessContextRequirements");
                unsafe {
                    processor_add_ref(processor);
                    *obj = std::ptr::addr_of_mut!((*processor).context_requirements_vtbl).cast();
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
            log_query_interface_miss("component", iid);
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
            crate::log_line("IComponent::getControllerClassId -> controller class id");
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
            // SAFETY: processor 是活动对象。
            let owner = unsafe { (*processor).extension_owner.clone() };
            if state != 0 {
                // 【为什么在 setActive 上重建宿主适配器】组件初始化线程与宿主真正
                // 运行模型/UI 回调的线程可能不同（工程加载线程 vs UI 线程），而
                // `NativeHost` 的所有查询都要求回到它绑定的线程。旧适配器线程不匹配
                // 时，`refresh_ui_inventory` / `refresh_host_geometry` 会静默失败，
                // 表现为"等待宿主音频"。这里用组件持有的 `IHostApplication` 引用
                // 在**当前**线程重建一次；失败只降级为 ARA-only，不会比现状更糟。
                let host = unsafe { (*processor).connection.host_raw() };
                if !host.is_null()
                    && (owner.native_host().is_none() || owner.host_adapter_thread_mismatch())
                {
                    unsafe { owner.rebind_host_adapter(host) };
                }
                // 【为什么记这一行】`setActive` 是 ARA 绑定窗口的关闭点：契约规定
                // `bindToDocumentControllerWithRoles` 只能在它之前调用（ARAVST3.h 的
                // `IPlugInEntryPoint2`）。`ara_bound` 为假就是"宿主把本插件当普通插入
                // 效果器"的直接证据 —— Cubase 实测正是如此，而此前日志里没有任何一行
                // 直接说出这件事，得靠对比两个会话才能推断。
                owner.note_component_activated();
                // 后台/前台切换后宿主可能已重建几何或恢复源访问；主动枚举一次，
                // 不等下一次 UI tick 的节流窗口。
                owner.resume_resync();
                crate::log_line(&format!(
                    "IComponent::setActive({state}): ara_bound={}",
                    owner.is_ara_bound()
                ));
            } else {
                crate::log_line(&format!("IComponent::setActive({state})"));
            }
            K_RESULT_OK
        })
    }
}

unsafe extern "system" fn component_set_state(this: *mut c_void, state: *mut c_void) -> TResult {
    if this.is_null() || state.is_null() {
        return K_INVALID_ARGUMENT;
    }
    // 【为什么恢复失败也回 OK】`setState` 返回失败时，宿主可能中止该实例的恢复，
    // 或者在重开工程时把插件标成"状态损坏"。一份读不出来的旧存档只该丢掉那份
    // 编辑，不该让整份工程加载失败。
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let bytes = match unsafe { crate::state_stream::read_state(state) } {
            Ok(bytes) => bytes,
            Err(error) => {
                log::warn!("[state] component setState ignored: unreadable stream: {error}");
                return;
            }
        };
        let owner = unsafe { &(*this.cast::<Processor>()).extension_owner };
        if let Err(error) = owner.restore_state(&bytes) {
            // 【为什么仍回 OK】见上面的说明：一份读不出来的旧存档只该丢掉那份编辑，
            // 不该让整份工程加载失败。但"丢掉"必须是**可见的**：把真因记进恢复状态，
            // 界面报 `restore_unresolved`，诊断包里能读到原因。
            owner.note_restore_failed(&error);
            log::error!(
                "[state] component setState ignored: {error}; the host document load continues"
            );
        }
    }));
    if outcome.is_ok() {
        K_RESULT_OK
    } else {
        K_RESULT_FALSE
    }
}

unsafe extern "system" fn component_get_state(this: *mut c_void, state: *mut c_void) -> TResult {
    if this.is_null() || state.is_null() {
        return K_INVALID_ARGUMENT;
    }
    // 【为什么编码失败也回 OK】`encode_state` 的契约是"永远给出一份可用载荷"；
    // 即使它意外失败，也要写一份合法的空状态，而不是让宿主把这次保存当成失败。
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let owner = unsafe { &(*this.cast::<Processor>()).extension_owner };
        let bytes = owner.encode_state().unwrap_or_else(|error| {
            log::error!("[state] component getState degraded to an empty payload: {error}");
            Vec::new()
        });
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
            // 【为什么 0 也算不支持】宿主提出"这条总线不要了"时，正确的回答是
            // `kResultFalse`（"我做不到动态增减总线"），宿主于是沿用我们声明的 1 进 1 出。
            // 假装接受会让 `getBusCount` 与 `getBusInfo` 互相矛盾 —— 对宿主撒谎。
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
            // SAFETY: this 是存活的音频接口；setupProcessing 在 process 之前调用。
            let owner = unsafe { &(*base_from_audio(this)).extension_owner };
            // 诊断：宿主声明的单块上限（不是分配上限；当前实现不依赖它分配）。
            owner.note_max_samples_per_block(setup.max_samples_per_block);
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
            let owner = unsafe { &(*base_from_audio(this)).extension_owner };
            owner.set_setup_sample_rate(setup.sample_rate);
            // 【为什么在这里装载 Capture 的渲染基础设施】`setupProcessing` 是 SDK 保证
            // 在任何 `process` 之前、且在宿主 UI 线程上被调用的唯一时刻，而宿主时钟、
            // 合成队列与 `prepare_owner` 都是"第一次 process 之前必须就位"的东西。
            // 此前它们由 WebView 的 20ms 定时器装载 —— 于是**没开窗口**的会话里宿主位置
            // 根本没有地方落：插件 UI 的播放光标永远是 0，而用户以为宿主没在播放。
            owner.arm_capture_rendering();
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

/// 输出写满审计是否开启（HIFISHIFTER_BUFFER_AUDIT=1）。
///
/// 【为什么用环境变量而不是设置项】它是排查工具，不是产品行为；用户永远不会打开它，
/// 而我们要能在不改构建的情况下让用户跑一遍。读一次就缓存。
fn buffer_audit_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HIFISHIFTER_BUFFER_AUDIT")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// 把输出平面预填成 NaN（审计哨兵）。
///
/// # Safety
/// data 由宿主提供；只按声明长度写，形状非法时由后续的 validate 拦下。
unsafe fn poison_outputs(data: *mut c_void) {
    if data.is_null() {
        return;
    }
    // SAFETY: 调用方保证 data 指向宿主提供的 ProcessData。
    let process = unsafe { &*data.cast::<crate::audio_abi::ProcessData>() };
    if process.outputs.is_null() || process.num_outputs <= 0 || process.num_samples <= 0 {
        return;
    }
    let frames = process.num_samples as usize;
    for bus in 0..process.num_outputs as usize {
        // SAFETY: 宿主声明了 num_outputs 个总线。
        let bus_ref = unsafe { &*process.outputs.add(bus) };
        if bus_ref.channel_buffers.is_null() || bus_ref.num_channels <= 0 {
            continue;
        }
        for channel in 0..bus_ref.num_channels as usize {
            // SAFETY: 宿主声明了 num_channels 个平面，每个至少 num_samples 帧。
            let plane = unsafe { *bus_ref.channel_buffers.add(channel) };
            if plane.is_null() {
                continue;
            }
            for index in 0..frames {
                // SAFETY: index < frames。
                unsafe { *plane.add(index) = f32::NAN };
            }
        }
    }
}

/// 扫描输出平面里仍未被写过的帧，并按档位记一次读数。
///
/// # Safety
/// 与 poison_outputs 相同。
unsafe fn report_unwritten(this: *mut c_void, data: *mut c_void) {
    if data.is_null() {
        return;
    }
    // SAFETY: 调用方保证 this 是存活的音频接口子对象。
    let owner = unsafe { &(*base_from_audio(this)).extension_owner };
    // SAFETY: 同一份 ProcessData。
    let process = unsafe { &*data.cast::<crate::audio_abi::ProcessData>() };
    if process.outputs.is_null() || process.num_outputs <= 0 || process.num_samples <= 0 {
        return;
    }
    let frames = process.num_samples as usize;
    let mut unwritten = 0_usize;
    let mut channels = 0_usize;
    for bus in 0..process.num_outputs as usize {
        // SAFETY: 宿主声明了 num_outputs 个总线。
        let bus_ref = unsafe { &*process.outputs.add(bus) };
        if bus_ref.channel_buffers.is_null() || bus_ref.num_channels <= 0 {
            continue;
        }
        for channel in 0..bus_ref.num_channels as usize {
            // SAFETY: 宿主声明了 num_channels 个平面。
            let plane = unsafe { *bus_ref.channel_buffers.add(channel) };
            if plane.is_null() {
                continue;
            }
            channels = channels.max(channel + 1);
            for index in 0..frames {
                // SAFETY: index < frames。
                if unsafe { *plane.add(index) }.is_nan() {
                    unwritten += 1;
                }
            }
        }
    }
    owner.note_buffer_audit(channels, unwritten, frames);
}

/// 宿主音频回调：完整渲染调度 + PCM 快照输出都在 `audio_process_inner` 里完成；
/// 本层只负责 panic 遏制。
unsafe extern "system" fn audio_process(this: *mut c_void, data: *mut c_void) -> TResult {
    // 【为什么必须 catch_unwind】本回调由**宿主音频线程**调用，panic 越过
    // `extern "system"` 边界是 UB，而且会带走整个 DAW 会话。`component_set_state` /
    // `component_get_state` 早就包了这一层，音频路径此前没有 —— 一条最不该崩的路径
    // 反而最没有保护。捕获后输出静音（不留上一次的残留），并只做一次原子计数：
    // 音频线程不得写文件、不得加锁等待。
    // 【输出写满审计（HIFISHIFTER_BUFFER_AUDIT=1）】把输出平面预填成 NaN，回调返回后
    // 扫一遍：任何仍是 NaN 的帧都说明某条分支**没有写满整块**，于是宿主会把上一次的
    // 内容（in-place 时甚至是**输入原声**）当作这一次的输出播出去。
    //
    // 【为什么这值得一次额外扫描】"输出缓冲残留"是用户提出的方向之一。逐条看分支可以
    // 论证它在立体声下不成立，但论证不是证据；这条审计把"到底有没有漏写、漏在哪一档"
    // 变成一次播放就能读到的一行。默认关闭（每块两次内存扫描，约 2 µs）。
    let auditing = buffer_audit_enabled();
    if auditing {
        // SAFETY: data 由宿主提供；形状非法时下面的 validate 会拦下，不会走到输出。
        unsafe { poison_outputs(data) };
    }
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        audio_process_inner(this, data)
    }));
    if auditing {
        // SAFETY: 与 audio_process_inner 同一份宿主提供的 ProcessData。
        unsafe { report_unwritten(this, data) };
    }
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
    let shape = unsafe { crate::audio_abi::validate(data.cast()) };
    // 【为什么 Capture 的观察与捕获在形状判定**之前】** 见 `ExtensionOwner::capture_observe`：
    // 录制只依赖"宿主给的位置"与"宿主给的输入"，与输出总线的形状无关。此前它在下面的
    // `num_outputs == 0` 早退之后 —— 于是"宿主没接我们的输出"这一个无关事实会让插件
    // 既录不到东西，又让 `sawContext` 永不置位（界面上的录制键一直是灰的）。
    //
    // 只跳过 `InvalidArgument`（data 本身非法）：那种情况下连字段读取都不该做。
    let mut evidence = None;
    if !matches!(shape, Err(crate::audio_abi::BufferError::InvalidArgument)) {
        // SAFETY: 宿主持有 Processor；data 的字段布局由 `validate` 的调用契约保证。
        let probe = unsafe { &mut *data.cast::<crate::audio_abi::ProcessData>() };
        // SAFETY: this 是存活的音频接口子对象。
        let owner = unsafe { &(*base_from_audio(this)).extension_owner };
        // 【T23】`process` 是宿主**一定会周期性到达**的少数入口之一，因此它是"宿主不再
        // 回调"这条死路在插件侧最可靠的出口。本函数只读两个原子量并按需置位，
        // 真正的收口仍由 ARA 控制器在下一次回调里执行（`runtime_mut` 的头部）。
        //
        // 【为什么在音频线程上也敢调】它不取任何 `Mutex` 的阻塞路径（文档槽用
        // `try_lock`，拿不到就跳过）—— 下一次块（10 ms 后）会再来。
        owner.request_settle_if_overdue();
        // 【为什么精度要记下来】非 32-bit 时插件按设计静音，而"静音"在界面上必须能
        // 说清原因（见 `CaptureStore::unsupported_precision`）。这一行与形状判定无关，
        // 所以在 `InvalidArgument` 之外一律记录。
        owner.capture().note_precision(probe.symbolic_sample_size);
        // 【为什么必须要求 32-bit 浮点】非 32-bit 时 `channelBuffers32` 那个联合体成员
        // 实际是 `channelBuffers64`，按 `f32` 读会读到垃圾。
        if probe.symbolic_sample_size == 0 && !probe.process_context.is_null() {
            // SAFETY: VST3 processContext 的完整 SDK 结构在当前回调期间存活。
            let context = unsafe { &*probe.process_context };
            owner.observe_transport(context, probe.process_mode);
            evidence = owner.capture_observe(probe, context);
        }
    }
    match shape {
        Ok(()) => {
            // SAFETY: 宿主持有 Processor，validate已验证当前数据和音频总线。
            let data = unsafe { &mut *data.cast::<crate::audio_abi::ProcessData>() };
            if data.num_samples == 0 || data.num_outputs == 0 {
                return K_RESULT_OK;
            }
            // SAFETY: this 是存活的音频接口子对象。
            let owner = unsafe { &(*base_from_audio(this)).extension_owner };
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
            // 【为什么不再无条件先清输出 —— 这是"插件在链路里却没有一点输出"的直接原因】
            // 宿主**可以合法地把同一个缓冲同时当输入和输出**（in-place：`inputs` 与
            // `outputs` 指向同一份内存，甚至同一个 `AudioBusBuffers`）。先清输出就把
            // **输入**一起抹掉了，而下面两条通路都要读它：直通会把零送出去、分层拷贝在
            // 快照范围外也会把零送出去。于是插件在链路里、却一点声音都没有。
            //
            // 现在的纪律是：**只有"本通路确实不写整块"时才清**（见 `layered` 与停止/静音
            // 分支）。写满整块的路径（直通、分层拷贝）自己覆盖每一帧，不需要也不允许先清。
            if owner.host_item_muted() {
                // 【为什么不再直接 clear】宿主静音是一次"有声 → 静音"的源切换；
                // 直接清零会在块首留下一次阶跃（正是用户听到的"咔"）。与门控清零
                // 走同一条去咔路径：从上一块尾巴淡到零，其余帧写零。
                // 该函数自己写满整块，因此**不能**再调 `clear_outputs`。
                let stream =
                    crate::capture::recorder::ProcessStream::from_mode(data.process_mode);
                let frames = data.num_samples as usize;
                let outputs = data.outputs;
                if outputs.is_null() {
                    let _ = unsafe { crate::audio_abi::clear_outputs(data) };
                } else {
                    unsafe { owner.apply_gate_fade_out(stream, &mut *outputs, frames) };
                }
                return K_RESULT_OK;
            }
            if data.process_context.is_null() {
                owner.snapshots[0].misses.fetch_add(1, Ordering::Relaxed);
                // 没有工程位置就无从选择输出内容：与改动前一致地静音。
                let _ = unsafe { crate::audio_abi::clear_outputs(data) };
                return if data.process_mode == 2 {
                    K_RESULT_FALSE
                } else {
                    K_RESULT_OK
                };
            }
            // SAFETY: VST3 processContext 的完整 SDK 结构在当前回调期间存活。
            let context = unsafe { &*data.process_context };
            // REAPER停播也会process固定光标位置；只离线导出允许没有kPlaying的供音。
            // 停播时不输出任何东西 ⇒ 必须清（否则输出里留着上一块的残留）。
            //
            // 【为什么还要看"位置在不在前进"】`kPlaying` 与位置不是原子更新的：宿主从
            // "停止"切到"播放"的那几块里，位置已经在走、播放位可能还没置上。此前只按
            // 播放位判断，于是**每次播放都吞掉开头**（用户报障：播放后约 1 秒没有输出）。
            // 位置在前进是宿主无关的硬事实 —— 真在播就必然前进（见
            // `ExtensionOwner::capture_transport_running`）。
            //
            // 【为什么这一支必须排在 `capture_stage` **之前**】门控判"要静音"时若已经
            // 写过快照，清输出就清不掉那一块的内容 —— 于是每次停播/移动光标都会漏出一
            // 小块**新位置**的音频（用户报障："很短、内容不对的一声"）。先判、后写，
            // "清输出"才是真的清。
            // 【为什么不能再以 `kPlaying == 0` 为前提】FL Studio 停播时仍可能保持
            // `kPlaying == 1`；此时门控必须由“本块是否真的在推进”（`allow_output`）
            // 独立决定。`capture_should_emit` 对 ARA 实例仍返回 `kPlaying`，所以
            // “停播即静音”的既有语义不变。
            let playing = context.state & crate::audio_abi::PROCESS_CONTEXT_PLAYING != 0;
            if data.process_mode != 2 && !owner.capture_should_emit(evidence, playing) {
                owner.note_gate_cleared(
                    evidence.map_or(crate::capture::recorder::ProcessStream::Realtime, |ev| {
                        ev.stream
                    }),
                );
                // 【为什么要先去咔再清零】起播窗口里 `allow_output` 可能逐块翻转
                // （`Resting` 重复抑制），于是出现"有声 → 清零 → 有声"。直接从有声
                // 阶跃到零就是用户听到的"咔"。
                //
                // 【为什么用 `apply_gate_fade_out` 而不是"先去咔再 `clear_outputs`"】
                // 后者先把整块写零，去咔产物随即被销毁 —— 那不是去咔，是把去咔白做
                // 一遍。本函数自己按"尾巴 → 0"写满整块，因此**不能再调用**清零。
                unsafe {
                    owner.apply_gate_fade_out(
                        evidence.map_or(crate::capture::recorder::ProcessStream::Realtime, |ev| {
                            ev.stream
                        }),
                        &mut *data.outputs,
                        data.num_samples as usize,
                    );
                }
                return K_RESULT_OK;
            }
            // 【Capture 通路】在快照输出之前插入。它要么自己写完输出（`Handled`），要么
            // 告诉下面"读哪个发布器"以及"区间外要不要直通宿主原声"（`Snapshot`）。
            // 未录制且没有素材时它是 `PassThrough` —— 插件在录制之前是完全透明的，
            // 而不是把整条轨静音（见 `ExtensionOwner::capture_output`）。
            let (snapshot_source, layered) = match owner.capture_stage(data, evidence) {
                crate::render::extension::CaptureStage::Handled => return K_RESULT_OK,
                crate::render::extension::CaptureStage::Snapshot { source, layered } => {
                    (source, layered)
                }
            };
            // 【为什么分层拷贝前不能清】它写满整块（快照范围内取快照、范围外取**输入**），
            // 而 in-place 宿主下"输入"就是这块缓冲本身 —— 先清等于把要拷的东西删掉。
            if !layered && unsafe { crate::audio_abi::clear_outputs(data) }.is_err() {
                // 【为什么不用 expect】这里在**音频线程**上，panic 会跨越 VST3 的 FFI 边界
                // 展开进宿主的音频回调。宁可返回错误码让宿主把这一帧当静音。
                return K_RESULT_FALSE;
            }
            // 按 `setupProcessing` 的**权威率**选快照：非原生率（96k/192k…）取最近的已声明
            // 原生快照，`copy_block` 会把它线性重采样到宿主输出率。此前这里用
            // `context.sample_rate` 且只认 44.1/48k，其余 `kResultFalse` —— 与 setup 的接受域
            // 不一致，96k 工程里会持续静音甚至被踢出链路（见 `setup_sample_rate`）。
            //
            // 原始磁带只有一份（它按捕获时的工程率录制），因此不参与按率选择。
            let publisher = match snapshot_source {
                crate::render::extension::SnapshotSource::Processed => {
                    &owner.snapshots[owner.snapshot_index_for_rate(context.sample_rate)]
                }
                crate::render::extension::SnapshotSource::Raw => &owner.raw_tape,
            };
            // 输出率取宿主在 processContext 里报的率（那才是宿主缓冲的真实率）；它缺失/
            // 非法时退回 `setupProcessing` 的声明，避免 0 触发无意义的巨量重采样。
            let output_rate = if context.sample_rate.is_finite() && context.sample_rate > 0.0 {
                context.sample_rate
            } else {
                owner.setup_sample_rate()
            };
            // SAFETY: 缓冲经 validate 校验；`layered` 时未清但由 `copy_block_layered_at`
            // 逐帧覆盖，其余路径已由上面的 clear_outputs 清过。publisher 由 owner 保留。
            // 【B0】本块分层拷贝的覆盖读数。两段（循环切分）各自写一份，这里合并。
            let mut layered_probe = crate::render::snapshot::LayeredProbe::default();
            let ready = unsafe {
                if layered {
                    // 【为什么要重新取一次输入视图】Capture 要"录过的那一段被替换、其余
                    // 保持宿主原声"，而"范围外"这个区别只在写之前知道（见
                    // `copy_block_layered_at`）。这里不分配：`capture_view` 只借用宿主缓冲。
                    let input = crate::audio_abi::capture_view(
                        data,
                        context.project_time_samples,
                        context.sample_rate,
                        context.state & crate::audio_abi::PROCESS_CONTEXT_PLAYING != 0,
                    );
                    match evidence {
                        Some(evidence) => {
                            let mut ready = true;
                            for part in evidence.split.iter() {
                                let mut part_probe = crate::render::snapshot::LayeredProbe::default();
                                ready &= publisher.copy_block_layered_at_probed(
                                    part.position,
                                    output_rate.max(1.0) as u32,
                                    &mut *data.outputs,
                                    part.input_offset,
                                    part.frames,
                                    &input.planes,
                                    part.input_offset,
                                    &mut part_probe,
                                );
                                layered_probe.merge(part_probe);
                            }
                            ready
                        }
                        None => publisher.copy_block_layered_at_probed(
                            context.project_time_samples,
                            output_rate.max(1.0) as u32,
                            &mut *data.outputs,
                            0,
                            data.num_samples as usize,
                            &input.planes,
                            0,
                            &mut layered_probe,
                        ),
                    }
                } else {
                    match evidence {
                        Some(evidence) => {
                            let mut ready = true;
                            for part in evidence.split.iter() {
                                ready &= publisher.copy_block_at(
                                    part.position,
                                    output_rate.max(1.0) as u32,
                                    &mut *data.outputs,
                                    part.input_offset,
                                    part.frames,
                                );
                            }
                            ready
                        }
                        None => publisher.copy_block(
                            context.project_time_samples,
                            output_rate.max(1.0) as u32,
                            &mut *data.outputs,
                            data.num_samples as usize,
                        ),
                    }
                }
            };
            // 【为什么在快照写完以后】接缝去咔需要"上一块输出的尾巴"与"这一块的新信号"
            // 做交叉淡化；写完再混，才不会把新源的第一帧直接硬切到旧源的最后帧。
            let declick_stream = evidence.map_or(
                crate::capture::recorder::ProcessStream::Realtime,
                |evidence| evidence.stream,
            );
            if layered {
                owner.note_layered_probe(declick_stream, layered_probe);
            }
            // 【这里曾经有一条"整块直通宿主原声"的保护，现已删除】
            //
            // 它的判据是"读游标跳过了超过 50 ms"，动作是**把整块**（宿主块最大可达 6 s）
            // 换成宿主输入。而直通本身就是换了一个信号源 —— 前后各一次硬切换，用户听到
            // 的那一下"咔"其实是这个保护动作自己。
            //
            // 现在位置只有一个权威（`StreamGeometry` 的 R1–R4），跳变只可能发生在 pass
            // 边界，而 pass 边界本来就要重新开始；接缝去咔的一次交叉淡化就够。
            unsafe {
                owner.apply_output_declick(
                    declick_stream,
                    &mut *data.outputs,
                    data.num_samples as usize,
                )
            };
            if data.process_mode == 2 && !ready {
                return K_RESULT_FALSE;
            }
            // 【为什么 in-place 时要把掩码压平】见 `input_output_share_bus`：快照写下的
            // "这一段是静音"会在下一块被我们自己的捕获读成"宿主没给输入"。掩码只是提示，
            // 压平成"非静音"永远是安全的（最坏情况是宿主少省一次算力）。
            if crate::audio_abi::input_output_share_bus(data) {
                // SAFETY: 输出总线已在上面校验过；同一份结构在回调期间存活。
                unsafe { (*data.outputs).silence_flags = 0 };
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
    /// 控制器先于路由拿到状态时的暂存。
    ///
    /// 【为什么需要】宿主可以只走 `IEditController` 通道；`setComponentState` /
    /// `setState` 往往发生在 `createView` 与视图连接之前，此时 `editor_link` 还没有
    /// owner。先存下来，等 `setComponentHandler` / `createView` 时补投。
    pending_state: std::sync::Mutex<Option<Vec<u8>>>,
}

impl EditController {
    fn new() -> Self {
        Self {
            vtbl: &EDIT_CONTROLLER_VTBL,
            refcount: AtomicI32::new(1),
            connection_vtbl: &CONTROLLER_CONNECTION_VTBL,
            connection: Default::default(),
            editor_link: Arc::new(Default::default()),
            pending_state: std::sync::Mutex::new(None),
        }
    }

    /// 把暂存状态交给 owner；owner 还不可用时原样放回。
    fn flush_pending_state(&self) {
        let bytes = self
            .pending_state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        let Some(bytes) = bytes else {
            return;
        };
        match self.editor_link.owner() {
            Ok(owner) => {
                if let Err(error) = owner.restore_state(&bytes) {
                    log::error!(
                        "[state] controller pending state ignored: {error}; the host load continues"
                    );
                }
            }
            Err(_) => {
                *self
                    .pending_state
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = Some(bytes);
            }
        }
    }

    /// 控制器通道的状态载荷：owner 可用时镜像组件状态，否则用暂存/空状态。
    fn state_bytes(&self) -> Vec<u8> {
        if let Ok(owner) = self.editor_link.owner() {
            return owner.encode_state().unwrap_or_else(|error| {
                log::error!("[state] controller getState degraded to an empty payload: {error}");
                Vec::new()
            });
        }
        self.pending_state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
            .unwrap_or_else(|| {
                crate::state_channel::EditState::default()
                    .encode()
                    .unwrap_or_default()
            })
    }

    /// 控制器通道的恢复：owner 可用就直接应用，否则暂存。
    fn restore_bytes(&self, bytes: Vec<u8>) {
        if bytes.is_empty() {
            return;
        }
        match self.editor_link.owner() {
            Ok(owner) => {
                if let Err(error) = owner.restore_state(&bytes) {
                    log::error!(
                        "[state] controller setState ignored: {error}; the host load continues"
                    );
                }
            }
            Err(_) => {
                *self
                    .pending_state
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = Some(bytes);
            }
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
            log_query_interface_miss("controller", iid);
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
    this: *mut c_void,
    state: *mut c_void,
) -> TResult {
    if this.is_null() || state.is_null() {
        return K_INVALID_ARGUMENT;
    }
    // 【为什么不再回 `E_NOTIMPL`】少数宿主只走控制器通道；回 not-implemented 会让
    // 它们保存/恢复不到任何编辑。这里与 `IComponent::setState` 共用同一份恢复逻辑。
    crate::guard_ffi(K_RESULT_FALSE, || {
        let controller = unsafe { &*(this as *const EditController) };
        match unsafe { crate::state_stream::read_state(state) } {
            Ok(bytes) => {
                controller.restore_bytes(bytes);
                K_RESULT_OK
            }
            Err(error) => {
                log::warn!("[state] controller setComponentState ignored: {error}");
                K_RESULT_OK
            }
        }
    })
}

unsafe extern "system" fn edit_controller_set_state(
    this: *mut c_void,
    state: *mut c_void,
) -> TResult {
    if this.is_null() || state.is_null() {
        return K_INVALID_ARGUMENT;
    }
    crate::guard_ffi(K_RESULT_FALSE, || {
        let controller = unsafe { &*(this as *const EditController) };
        match unsafe { crate::state_stream::read_state(state) } {
            Ok(bytes) => {
                controller.restore_bytes(bytes);
                K_RESULT_OK
            }
            Err(error) => {
                log::warn!("[state] controller setState ignored: {error}");
                K_RESULT_OK
            }
        }
    })
}

unsafe extern "system" fn edit_controller_get_state(
    this: *mut c_void,
    state: *mut c_void,
) -> TResult {
    if this.is_null() || state.is_null() {
        return K_INVALID_ARGUMENT;
    }
    crate::guard_ffi(K_RESULT_FALSE, || {
        let controller = unsafe { &*(this as *const EditController) };
        let bytes = controller.state_bytes();
        match unsafe { crate::state_stream::write_state(state, &bytes) } {
            Ok(()) => K_RESULT_OK,
            Err(error) => {
                log::warn!("[state] controller getState could not write: {error}");
                K_RESULT_FALSE
            }
        }
    })
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
            // 宿主连接控制器后，路由通常已经建立；补投早到的控制器状态。
            controller.flush_pending_state();
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
            let controller = unsafe { &*(this as *const EditController) };
            let view = crate::editor::create_view_with_link(controller.editor_link.clone());
            // 视图创建后路由可能已可用；把早到的控制器状态补投一次（不可用则放回）。
            controller.flush_pending_state();
            view
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

/// 从 `IProcessContextRequirements` 子对象回到 `Processor`（手法同 `connection_vtbl`）。
unsafe fn base_from_context_requirements(this: *mut c_void) -> *mut Processor {
    unsafe {
        this.cast::<u8>()
            .sub(offset_of!(Processor, context_requirements_vtbl))
            .cast()
    }
}

// ── 我们要宿主提供哪些 `ProcessContext` 字段 ────────────────────────────────────
//
// 【为什么必须精确，而不是"全都要"】这个掩码在 Setup Done 时被问一次，
// **之后不可更改**（SDK 原文 "It cannot be changed afterwards."）。两个方向都有代价：
//
// * 要得**太少** ⇒ 那个字段的有效位永远不置位。代价是**可检测的**（我们本来就读有效位，
//   读不到就退回上一有效读数），且只影响需要它的那一个功能。
// * 要得**太多** ⇒ 宿主为**每一个音频块**多做一份用不上的计算（和弦识别、MIDI 时钟
//   分辨率都要按块算）。代价是**不可检测的**：没有任何日志、没有错误，只表现为
//   "这个插件让工程变慢了"。
//
// 因此这里只要**真正消费**的字段，并逐位注明消费点。将来新增需求时改这一处即可 ——
// 代价只是"本次会话里旧实例拿不到"，而不是"永远拿不到"。

/// `kSystemTimeValid`：`TransportClock` 的诊断读数（`render/transport.rs` 的 `observe`）。
const NEED_SYSTEM_TIME: u32 = 1 << 0;
/// `kContTimeValid`：诊断读数；也是"这一块是不是循环回卷"的旁证。
const NEED_CONTINUOUS_TIME_SAMPLES: u32 = 1 << 1;
/// `kProjectTimeMusicValid`：小节/拍显示，以及 Capture 的区间按小节对齐。
const NEED_PROJECT_TIME_MUSIC: u32 = 1 << 2;
/// `kBarPositionValid`：网格吸附（Capture 的 punch 区间要吸到小节线）。
const NEED_BAR_POSITION_MUSIC: u32 = 1 << 3;
/// `kCycleValid`：Capture 的"只录循环区间"需要循环起止。
const NEED_CYCLE_MUSIC: u32 = 1 << 4;
/// `kTempoValid`：`TransportClock::tempo()` 的**唯一**来源。
const NEED_TEMPO: u32 = 1 << 6;
/// `kTimeSigValid`：`TransportClock::time_signature()` 的唯一来源。
const NEED_TIME_SIGNATURE: u32 = 1 << 7;
/// `kPlaying` / `kCycleActive` / `kRecording`。**没有这一位，`kPlaying` 是未定义的**，
/// 而 `audio_process_inner` 与 Capture 录制都靠它区分"停播"与"播放"。
const NEED_TRANSPORT_STATE: u32 = 1 << 10;

/// `IProcessContextRequirements::getProcessContextRequirements` 的返回值。
///
/// 【为什么是**并集**而不是按当前模式二选一】这个值在 Setup Done 时被问一次，
/// 而那时我们还**不知道**宿主会不会绑定 ARA（ARA 绑定允许发生在这个调用之前，
/// 见 `ARAVST3.h:165`；但宿主也可以完全不绑）。两条路径的需求不同：
/// ARA 路径要 tempo / 拍号，Capture 路径要 transportState / 循环。
/// 二选一必然有一半场景拿不到数据，所以取并集。
pub(crate) fn process_context_requirements() -> u32 {
    NEED_SYSTEM_TIME
        | NEED_CONTINUOUS_TIME_SAMPLES
        | NEED_PROJECT_TIME_MUSIC
        | NEED_BAR_POSITION_MUSIC
        | NEED_CYCLE_MUSIC
        | NEED_TEMPO
        | NEED_TIME_SIGNATURE
        | NEED_TRANSPORT_STATE
}

unsafe extern "system" fn pcr_query(
    this: *mut c_void,
    iid: *const u8,
    out: *mut *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_NO_INTERFACE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_NO_INTERFACE, || unsafe {
            component_query_interface(base_from_context_requirements(this).cast(), iid, out)
        })
    }
}

unsafe extern "system" fn pcr_add_ref(this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        crate::guard_ffi(0, || unsafe {
            processor_add_ref(base_from_context_requirements(this))
        })
    }
}

unsafe extern "system" fn pcr_release(this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        crate::guard_ffi(0, || unsafe {
            processor_release(base_from_context_requirements(this))
        })
    }
}

/// `IProcessContextRequirements::getProcessContextRequirements`。
///
/// 【为什么忽略 `this`】需求集是**模块级**常量，不随实例变化。`this` 仍参与
/// `addRef`/`release`（引用计数必须落在拥有者上），但这里不需要反解对象。
unsafe extern "system" fn pcr_get_process_context_requirements(_this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        // 返回 0 的语义是"什么都不要"，是安全的降级：宿主少发信息，我们按有效位退回默认。
        crate::guard_ffi(0, process_context_requirements)
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
static PROCESS_CONTEXT_REQUIREMENTS_VTBL: ProcessContextRequirementsVtbl =
    ProcessContextRequirementsVtbl {
        base: crate::editor::connection::UnknownVtbl {
            query: pcr_query,
            add: pcr_add_ref,
            release: pcr_release,
        },
        get_process_context_requirements: pcr_get_process_context_requirements,
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
mod context_requirements_tests;
#[cfg(test)]
mod factory_declaration_tests;
#[cfg(test)]
mod host_handshake_tests;
#[cfg(test)]
mod lifetime_tests;
#[cfg(test)]
mod realtime_tests;
