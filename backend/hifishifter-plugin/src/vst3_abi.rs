//! VST3 的 ABI **词汇**：类型、结果码、TUID 布局与 `FUnknown` vtable 前缀。
//!
//! 【为什么 VST3 的 ABI 词汇住在共享实现层】REAPER 的原生扩展（`host/reaper/`）、
//! `IComponentHandler2` 的 `setDirty` 桥（`editor/routing.rs`）、Linux 上向
//! `IPlugFrame` 查 `IRunLoop`（`editor/gui/linux.rs`）都是**共享层代码**在消费 VST3
//! 的 COM 面——这些宿主能力只经 VST3 暴露（见 `docs/host-extension-boundary.md`
//! 的「宿主原生编辑只有 VST3 有」），而依赖方向（格式外壳 → 共享层）决定了壳
//! 实现不了这份词汇。因此只有这里是 VST3 ABI 允许出现在共享层的地方：
//!
//! * 只允许"类型 + 常量 + 纯函数"——任何带状态、带工厂、带生命周期的东西都属于
//!   `hifishifter-vst3` 壳 crate（工厂 / 组件 / 处理器 / `IPlugView` /
//!   `IConnectionPoint` / 模块入口）；
//! * 判据：本模块不引用 `runtime` / `render` / `editor` 的任何项，也不持有任何
//!   `static` 状态。
//!
//! 【可见性】全部 `pub`：共享层内部用它，`hifishifter-vst3` 外壳也用它（壳把宿主
//! 给的 COM 指针转成对共享层调用时，两端说的是同一种字节布局——三份
//! `UnknownVtbl` 曾经各自定义，本模块是它们合一后的唯一一份）。

use std::ffi::c_void;

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

/// 从 `DECLARE_CLASS_IID` 的四个 32 位字构造 TUID（Windows VST3 布局）。
///
/// 这是实测结果，不是推断：Windows 上 VST3 SDK 的 `fplatform.h` 把 `COM_COMPATIBLE`
/// 定义为 1，其 `INLINE_UID` 展开等价于标准 GUID 内存布局 ——
/// 第 1 个字小端、第 2 个字拆成两个 u16 各自小端、第 3/4 个字大端。
/// 依据：REAPER 请求 `IPluginFactory2` 时传出的字节
/// `50B607004BF20B4CA464EDB9F00B2ABB`，正是
/// `{0007B650-F24B-4C0B-A464-EDB9F00B2ABB}` 的 GUID 布局。
pub const fn uid_guid(words: [u32; 4]) -> Tuid {
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
        (l3 >> 24) as u8,
        ((l3 >> 16) & 0xFF) as u8,
        ((l3 >> 8) & 0xFF) as u8,
        (l3 & 0xFF) as u8,
        ((l4 >> 24) & 0xFF) as u8,
        ((l4 >> 16) & 0xFF) as u8,
        ((l4 >> 8) & 0xFF) as u8,
        (l4 & 0xFF) as u8,
    ]
}

/// `COM_COMPATIBLE=0` 时的布局：每个字大端展开。
///
/// 【为什么不是"备用"而是"另一半平台的主布局"】Linux / macOS 的 VST3 就是按它比较
/// IID 的（见 [`platform_iid`]）。此前它只被 `iid_matches` 用来"收"，从未被用来"发"。
pub const fn uid_be(words: [u32; 4]) -> Tuid {
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

/// 发给宿主的 IID 字节：跟着平台的 `COM_COMPATIBLE` 走。
///
/// 【为什么必须分平台，而不是"两种都发一遍"】`queryInterface` 是**逐字节**比较，
/// 而 `INLINE_UID`（`pluginterfaces/base/funknown.h:36-60`）按 `COM_COMPATIBLE`
/// 展开成两种布局：
///
/// * Windows（`fplatform.h:78`，`COM_COMPATIBLE 1`）—— COM/GUID 布局，即 [`uid_guid`]；
/// * Linux / macOS（`fplatform.h:158` / `:226`，`COM_COMPATIBLE 0`）—— **每字大端**，
///   即 [`uid_be`]。
///
/// 此前全部"插件 → 宿主"的查询都发的是 `uid_guid`。宿主的常量里根本没有那一串，
/// 于是**一次都没命中过**，而失败形态是"回一个 null 指针"——不崩、不报错，只有一行
/// `available=false`。实测（Linux REAPER）：`IReaperHostApplication` 拿不到 ⇒ ARA
/// 认领不到宿主 item；`IHostApplication` 拿不到 ⇒ 编辑器与处理器断链；
/// `IComponentHandler2` 拿不到 ⇒ 工程永不标脏。判据：REAPER 的二进制里那六个 IID
/// 只有大端那一串，且 `scripts/vst3/unix/host_iid_probe.c` 的握手探针能稳定复现。
///
/// 【为什么不在类型层面禁掉 `uid_guid`】类 ID（`PROCESSOR_CID` 等）是我们自己造的
/// 16 字节身份，宿主只做"原样保存、原样回传"，两种布局都自洽；它们**不能**跟着平台变
/// （已经写进用户工程）。所以这里只约束"发给宿主去比较的接口 IID"。
pub const fn platform_iid(words: [u32; 4]) -> Tuid {
    if cfg!(windows) {
        uid_guid(words)
    } else {
        uid_be(words)
    }
}

/// 另一种布局的同一个 IID：**只**给"宿主可能是别人转发的"兜底用。
///
/// 【为什么保留它】Wine / CrossOver / 各类 wrapper 进程转发 VST3 调用时，转发层可能
/// 按它自己那一套布局比较 IID。`NativeHost::from_context` 因此会再试一次另一种布局，
/// 命中时大声记一行 —— 那种情况必须看得见，不能被"反正能跑"掩盖。
pub const fn alternate_iid(words: [u32; 4]) -> Tuid {
    if cfg!(windows) {
        uid_be(words)
    } else {
        uid_guid(words)
    }
}

/// 本构建把 IID 编成哪种布局（日志用：让人一眼看出产物是按哪边编的）。
pub const fn iid_layout() -> &'static str {
    if cfg!(windows) {
        "com-guid"
    } else {
        "big-endian"
    }
}

/// 判断宿主传入的 16 字节 IID 是否等于给定接口（四个 32 位字）。
///
/// 同时接受 GUID 布局与逐字大端布局：Windows 实测是前者，但换一种宿主构建方式
/// 就可能变成后者。接受两种可以避免"因为字节序猜错而误判为 ABI 不兼容"，
/// 对探针而言多接受的错误匹配没有实际风险。
///
/// # Safety
///
/// `iid` 必须指向可读的 16 字节 TUID（或为空；空指针恒返回 `false`）。
pub unsafe fn iid_matches(iid: *const u8, words: [u32; 4]) -> bool {
    if iid.is_null() {
        return false;
    }
    // SAFETY: 调用方保证 iid 指向一个可读的 16 字节 TUID。
    let actual = unsafe { std::slice::from_raw_parts(iid, 16) };
    actual == uid_guid(words) || actual == uid_be(words)
}

/// `FUnknown` 的三项（一切 VST3/COM 接口的 vtable 前缀）。
///
/// 【为什么在这里而不是各接口文件里】`editor/connection.rs`、`editor/view.rs` 与
/// `editor/gui/linux.rs` 曾经各自定义一份逐字段相同的结构；三份"长得一样"的定义
/// 正是分叉的温床（谁先多加一个字段，另外两处就要靠人工同步）。现在合一：一切
/// "按 vtable 前缀操作陌生 COM 对象"的代码都引用这一份。
#[repr(C)]
pub struct UnknownVtbl {
    /// `FUnknown::queryInterface`（IID 是 16 字节 TUID 的裸指针）。
    pub query: unsafe extern "system" fn(*mut c_void, *const u8, *mut *mut c_void) -> TResult,
    /// `FUnknown::addRef`。
    pub add: unsafe extern "system" fn(*mut c_void) -> u32,
    /// `FUnknown::release`。
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
}

#[cfg(test)]
#[path = "vst3/iid_layout_tests.rs"]
mod iid_layout_tests;
