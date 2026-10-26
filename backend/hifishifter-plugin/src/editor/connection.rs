//! 锁定SDK的IConnectionPoint/IHostApplication/IMessage ABI；消息允许宿主放置代理。
use crate::vst3::{
    iid_matches, TResult, K_INVALID_ARGUMENT, K_NOT_IMPLEMENTED, K_NO_INTERFACE, K_RESULT_FALSE,
    K_RESULT_OK,
};
use std::ffi::{c_char, c_void, CStr};
use std::sync::Mutex;
pub(crate) const IID_CONNECTION: [u32; 4] = [0x70A4156F, 0x6E6E4026, 0x989148BF, 0xAA60D8D1];
const IID_MESSAGE: [u32; 4] = [0x936F033B, 0xC6C047DB, 0xBB0882F8, 0x13C1E613];
const ROUTE_ID: &CStr = c"HiFiShifter.Editor.Route.v1";
const REQUEST_ID: &CStr = c"HiFiShifter.Editor.RequestRoute.v1";
#[repr(C)]
pub(crate) struct UnknownVtbl {
    pub query: unsafe extern "system" fn(*mut c_void, *const u8, *mut *mut c_void) -> TResult,
    pub add: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
}
#[repr(C)]
pub(crate) struct ConnectionVtbl {
    pub base: UnknownVtbl,
    pub connect: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    pub disconnect: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    pub notify: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
}
#[repr(C)]
struct HostVtbl {
    base: UnknownVtbl,
    name: unsafe extern "system" fn(*mut c_void, *mut u16) -> TResult,
    /// `IHostApplication::createInstance`。
    ///
    /// 【为什么保留但不再调用】本仓改为自带 `IMessage`（见 `ConnectionState::send`）：
    /// 这条工厂是**可选**的，宿主回 `kNotImplemented` 时整条编辑器路由会静默失败。
    /// 槽位留着，这张表才是 `IHostApplication` 的完整 ABI 镜像 —— 没有它，
    /// 下一次有人往表里补字段时就得重新数一遍偏移。
    #[allow(dead_code)]
    create:
        unsafe extern "system" fn(*mut c_void, *const u8, *const u8, *mut *mut c_void) -> TResult,
}
#[repr(C)]
struct MessageVtbl {
    base: UnknownVtbl,
    id: unsafe extern "system" fn(*mut c_void) -> *const c_char,
    set_id: unsafe extern "system" fn(*mut c_void, *const c_char),
    attributes: unsafe extern "system" fn(*mut c_void) -> *mut c_void,
}
#[repr(C)]
struct AttributesVtbl {
    base: UnknownVtbl,
    set_int: unsafe extern "system" fn(*mut c_void, *const c_char, i64) -> TResult,
    get_int: unsafe extern "system" fn(*mut c_void, *const c_char, *mut i64) -> TResult,
    set_float: unsafe extern "system" fn(*mut c_void, *const c_char, f64) -> TResult,
    get_float: unsafe extern "system" fn(*mut c_void, *const c_char, *mut f64) -> TResult,
    set_string: unsafe extern "system" fn(*mut c_void, *const c_char, *const u16) -> TResult,
    get_string: unsafe extern "system" fn(*mut c_void, *const c_char, *mut u16, u32) -> TResult,
    set_binary:
        unsafe extern "system" fn(*mut c_void, *const c_char, *const c_void, u32) -> TResult,
    get_binary: unsafe extern "system" fn(
        *mut c_void,
        *const c_char,
        *mut *const c_void,
        *mut u32,
    ) -> TResult,
}
struct Owned(usize);
impl Owned {
    unsafe fn retain(pointer: *mut c_void) -> Self {
        if !pointer.is_null() {
            unsafe {
                ((**pointer.cast::<*const UnknownVtbl>()).add)(pointer);
            }
        }
        Self(pointer as usize)
    }
    fn ptr(&self) -> *mut c_void {
        self.0 as *mut c_void
    }
}
impl Drop for Owned {
    fn drop(&mut self) {
        let ptr = self.ptr();
        if !ptr.is_null() {
            unsafe {
                ((**ptr.cast::<*const UnknownVtbl>()).release)(ptr);
            }
        }
    }
}

/// 宿主应用名（`IHostApplication::getName`）。
///
/// 【为什么单独存一份】报障时最需要的一条信息是"你在哪个宿主里"——而这份信息此前只有
/// 一个**声明过却从未调用**的 vtable 槽位（`HostVtbl::name`）。存下来供诊断包读取。
///
/// 【为什么绝不用于分支】按宿主名分支会让"没见过的宿主"退化成不可用；一切差异都应由
/// 能力探测（`BootstrapFlags` / `NativeHost::from_context` 的 Option bundle）决定。
static HOST_NAME: Mutex<Option<String>> = Mutex::new(None);

/// 读回宿主应用名（未识别为 `None`）。只读，供诊断输出。
pub(crate) fn host_application_name() -> Option<String> {
    HOST_NAME.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// 调用 `IHostApplication::getName` 并记录；失败/空名一律静默（诊断信息不是硬依赖）。
///
/// # Safety
/// `host` 必须是 `IHostApplication`（`initialize` 的实参）且持有一份 owning 引用。
unsafe fn record_host_name(host: *mut c_void) {
    if host.is_null() {
        return;
    }
    // SAFETY: host 指向 IHostApplication vtable；name 的签名为 `TResult(String128)`，
    // 即写入一个 128 个 UTF-16 单元的调用方缓冲（VST3 SDK 的 String128）。
    let table = unsafe { &**host.cast::<*const HostVtbl>() };
    let mut buffer = empty_host_name_buffer();
    if unsafe { (table.name)(host, buffer.as_mut_ptr().cast()) } != K_RESULT_OK {
        return;
    }
    let Some(name) = decode_string128(&buffer) else {
        return;
    };
    *HOST_NAME.lock().unwrap_or_else(|e| e.into_inner()) = Some(name.clone());
    crate::log_line(&format!("host application (vst3 name): {name}"));
    // 宿主名不是「问一次」就能得到的：包装器进程（sfvstwrap / fruity-wrapper）会冒充
    // 宿主，Cubase 加载期只给 unknown。真正的判定在 `host_identity` 里，靠证据链
    // （环境变量 / ARA 名 / 进程映像 / 祖先链 / VST3 名）取第一个能认出的。
    resolve_and_record_host_identity();
}

/// 收集全部证据、定出宿主、把日志切到 `plugin-<slug>.log`。
///
/// 【为什么可以重复调用】Cubase 在加载期只给 unknown，真正使用时的名字更晚才到。
/// 第一次拿不到就留 unknown，后续调用（首次 process）再解析一次；一旦得到非 unknown
/// 的结果就不再重算 —— 否则日志文件名会来回跳。
pub(crate) fn resolve_and_record_host_identity() -> crate::host_identity::HostIdentity {
    let mut evidence = crate::host_identity::process_evidence();
    if let Some(name) = host_application_name() {
        evidence.push(crate::host_identity::Evidence {
            source: crate::host_identity::EvidenceSource::Vst3Name,
            name,
        });
    }
    let identity = crate::host_identity::resolve_identity(&evidence);
    crate::log_line(&identity.describe());
    if identity.display != "unknown" {
        // 日志按宿主分文件：拿到名字的这一刻把日志切过去
        //（此前攒下的行会一起补写进去，见 `diagnostics::note_host_application`）。
        crate::diagnostics::note_host_application(&identity.display);
    }
    identity
}

/// 装 `IHostApplication::getName` 结果的缓冲。
///
/// 【为什么类型随平台变】VST3 的 `String128` **不是一种编码**：
///
/// `cpp
/// #if SMTG_OS_WINDOWS
/// typedef char16 String128[128];   // UTF-16
/// #else
/// typedef char   String128[128];   // UTF-8
/// #endif
/// `
///
/// 按 UTF-16 解 Linux 的缓冲会得到一串乱码 ⇒ `from_utf16` 失败 ⇒ 宿主名永远是
/// `unknown` ⇒ 日志文件名永远停在 `plugin-unknown.log`。用户报的"日志没有按宿主名分类"
/// 就是这一条：这段代码只在 Windows 上被跑过。
#[cfg(windows)]
type HostNameBuffer = [u16; 128];
// 【为什么留 256 字节】宿主写多少字节由**它**的 `String128` 定义决定，而不是由我们的
// 解码方式决定：Windows 是 128 个 UTF-16 单元（256 字节），Unix 是 128 字节。缓冲比
// 两者都大，才不会被"对方用的是另一种编码"写穿 —— 那会是一次栈溢出，在 ARA 宿主里
// 表现为一个语焉不详的 `ARA Error`。之前这里只有 128 字节，正好踩中。
#[cfg(not(windows))]
type HostNameBuffer = [u8; 256];

/// 一份清零的 `String128` 缓冲。
///
/// 【为什么要一个函数】缓冲长度是类型的一部分（Windows 是 128 个 UTF-16 单元、Unix 是
/// 256 字节），而 `[0; N]` 里的 N 必须跟着平台走。写在一处，另一边改长度时编译器会
/// 直接指着这里报错 —— 而不是让某个平台的构建悄悄坏掉（此前正是这样：Windows 侧写着
/// `[0; 256]`，在 `[u16; 128]` 上根本编译不过）。
fn empty_host_name_buffer() -> HostNameBuffer {
    #[cfg(windows)]
    {
        [0u16; 128]
    }
    #[cfg(not(windows))]
    {
        [0u8; 256]
    }
}

/// `String128` → `String`；空串/非法编码返回 `None`。
///
/// 【为什么单独抽出来】宿主名是纯诊断信息，绝不能因为一次读名失败影响加载；把"怎么
/// 解释这段缓冲"独立出来，边界就能单测，不必造一个真实宿主接口。
#[cfg(windows)]
fn decode_string128(buffer: &HostNameBuffer) -> Option<String> {
    let end = buffer
        .iter()
        .position(|unit| *unit == 0)
        .unwrap_or(buffer.len());
    let name = String::from_utf16(&buffer[..end]).ok()?;
    (!name.is_empty()).then_some(name)
}

#[cfg(not(windows))]
fn decode_string128(buffer: &HostNameBuffer) -> Option<String> {
    // 【为什么这里有两种编码】VST3 的 `String128` 在 Unix 上是 `char[128]`（UTF-8），
    // 但宿主不一定照办：实测 REAPER/Linux 按 **UTF-16LE** 写（"REAPER" 写成
    // `R\0E\0A\0P\0E\0R\0`）。只按 UTF-8 解会停在第一个 NUL 上得到 "R"，
    // 于是宿主名永远是 `unknown`、日志文件名永远停在 `plugin-unknown.log` ——
    // 症状与"根本没拿到宿主名"完全同形，这正是它藏了这么久的原因。
    if looks_like_utf16le(buffer) {
        if let Some(name) = decode_utf16le(buffer) {
            return Some(name);
        }
    }
    let end = buffer
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(buffer.len());
    let name = std::str::from_utf8(&buffer[..end]).ok()?.to_owned();
    (!name.is_empty()).then_some(name)
}

/// 这段缓冲是不是 UTF-16LE 写出来的？
///
/// 判据只有一条：**第二个字节是 0**。ASCII 名字的 UTF-16LE 一定长成 `R\0E\0…`；
/// 而 UTF-8 名字的第二个字节不可能是 0（空名除外，空名两种解都得到空串）。
/// 单字符 UTF-8 名（"X"）也会命中，但两种解都得到 "X"，没有区别。
#[cfg(not(windows))]
fn looks_like_utf16le(buffer: &HostNameBuffer) -> bool {
    buffer.first().is_some_and(|byte| *byte != 0) && buffer.get(1) == Some(&0)
}

/// 按 UTF-16LE 解到第一个 0 单元；空串/非法代理对返回 `None`。
#[cfg(not(windows))]
fn decode_utf16le(buffer: &HostNameBuffer) -> Option<String> {
    let mut units = Vec::new();
    let mut index = 0;
    while index + 1 < buffer.len() {
        let unit = u16::from_le_bytes([buffer[index], buffer[index + 1]]);
        if unit == 0 {
            break;
        }
        units.push(unit);
        index += 2;
    }
    if units.is_empty() {
        return None;
    }
    String::from_utf16(&units)
        .ok()
        .filter(|name| !name.is_empty())
}

#[cfg(test)]
mod host_name_tests {
    use super::*;

    /// 未连接宿主时读不到名字 —— 诊断包据此报 `null`，而不是一个编出来的占位名。
    #[test]
    fn the_host_name_is_absent_before_any_host_is_recorded() {
        // 注意：本用例不写 `HOST_NAME`（进程级静态），只读。
        let name = host_application_name();
        assert!(name.is_none() || !name.unwrap().is_empty());
    }

    /// **不变量：`String128` 按平台的真实编码解码。**
    ///
    /// 【为什么钉死】Linux 上按 UTF-16 解会得到乱码，宿主名永远是 `unknown`，日志文件
    /// 永远叫 `plugin-unknown.log` —— 一个"什么都没坏、但诊断全废"的静默缺陷。
    #[cfg(windows)]
    #[test]
    fn string128_decoding_stops_at_the_terminator() {
        let mut buffer = [0u16; 128];
        for (slot, unit) in "TestHost".encode_utf16().enumerate() {
            buffer[slot] = unit;
        }
        assert_eq!(decode_string128(&buffer).as_deref(), Some("TestHost"));
        // 全是 NUL ⇒ 空名 ⇒ None（不是 `Some("")`）。
        assert_eq!(decode_string128(&[0u16; 128]), None);
        // 没有终结符时取整段。
        //
        // 【为什么用 128 个 'A' 而不是两个字面量】`HostNameBuffer` 在 Windows 上是
        // `[u16; 128]`，长度是类型的一部分；两元素字面量在这里**编译不过**（与 Unix
        // 那条用例同一个坑的镜像）。
        assert_eq!(
            decode_string128(&[b'A' as u16; 128]).map(|name| name.len()),
            Some(128)
        );
        // 非法 UTF-16（孤立高位代理）⇒ None，不 panic。
        let mut lone_surrogate = [0u16; 128];
        lone_surrogate[0] = 0xD800;
        assert_eq!(decode_string128(&lone_surrogate), None);
    }

    /// Unix：`String128` 是 `char[128]`（UTF-8），不是 UTF-16。
    #[cfg(not(windows))]
    #[test]
    fn string128_decoding_stops_at_the_terminator() {
        let mut buffer = [0u8; 256];
        for (slot, byte) in "Reaper".bytes().enumerate() {
            buffer[slot] = byte;
        }
        assert_eq!(decode_string128(&buffer).as_deref(), Some("Reaper"));
        assert_eq!(decode_string128(&[0u8; 256]), None);

        // 宿主按 **UTF-16LE** 写（实测 REAPER/Linux）：必须整名读回，而不是只读到 "R"。
        // 这一条与上面那条一起，钉住"同一个缓冲的两种合法写法"。
        let mut utf16 = [0u8; 256];
        for (slot, unit) in "REAPER".encode_utf16().enumerate() {
            utf16[slot * 2] = unit as u8;
            utf16[slot * 2 + 1] = (unit >> 8) as u8;
        }
        assert_eq!(decode_string128(&utf16).as_deref(), Some("REAPER"));

        // 非 ASCII 的宿主名（中文/日文宿主名并不罕见）必须原样解出来。
        let mut wide = [0u8; 256];
        for (slot, byte) in "宿主".as_bytes().iter().enumerate() {
            wide[slot] = *byte;
        }
        assert_eq!(decode_string128(&wide).as_deref(), Some("宿主"));

        // 非法 UTF-8 ⇒ None，不 panic。
        //
        // 【为什么不用 `&[0xFF, 0xFE, 0]` 字面量】`HostNameBuffer` 在 Unix 上是
        // `[u8; 256]`，长度是类型的一部分；三元素字面量在 Linux 上**编译不过**
        // （这条用例此前只在 Windows 上跑过）。
        let mut invalid = [0u8; 256];
        invalid[0] = 0xFF;
        invalid[1] = 0xFE;
        assert_eq!(decode_string128(&invalid), None);
    }
}
#[derive(Default)]
pub(crate) struct ConnectionState {
    host: Mutex<usize>,
    peer: Mutex<usize>,
}
impl ConnectionState {
    /// 宿主提供message对象，不能把Rust Arc裸指针当跨代理协议。
    ///
    /// 【为什么不再 queryInterface】VST3 的 `initialize (IHostApplication*)` 契约里，
    /// 实参**就是**宿主应用接口。此前为了拿一份 owning 引用去 QI，等于把"编辑器能不能
    /// 连上处理器"绑死在 IID 的字节布局上 —— 布局一错，宿主名与整条消息通道一起死，
    /// 而用户看到的只是每个命令都回一句 "FX editor not connected to its processor yet"。
    /// 直接 addRef 就够了：它只碰 FUnknown 的前三个槽，与任何 IID 无关。
    pub unsafe fn initialize(&self, context: *mut c_void) {
        if context.is_null() {
            return;
        }
        // SAFETY: initialize 的实参按 VST3 契约是 IHostApplication*；addRef 只读 vtable
        // 第 1 槽，不依赖任何 IID。
        let owned = unsafe { Owned::retain(context) };
        let old = std::mem::replace(
            &mut *self.host.lock().unwrap_or_else(|e| e.into_inner()),
            owned.0,
        );
        std::mem::forget(owned);
        drop(Owned(old));
    }
    /// 读一次宿主应用名（`IHostApplication::getName`）。
    ///
    /// 【为什么由调用方在**健康路径**上触发】`initialize` 期间的宿主调用可能重入关闭
    /// 组件（见 `component_initialize` 的重入测试）。读名是一次额外的宿主调用，不该
    /// 发生在"已经被重入关闭"的那条路径上 —— 那时组件正在拒绝初始化，任何多余的宿主
    /// 交互都没有意义。
    pub fn record_host_name(&self) {
        let host = *self.host.lock().unwrap_or_else(|e| e.into_inner()) as *mut c_void;
        // SAFETY: host 是本状态持有的 IHostApplication 引用（initialize 里 addRef 登记）。
        unsafe { record_host_name(host) };
    }
    /// 本组件持有的宿主应用接口（`IHostApplication*`）。
    ///
    /// 【为什么可以在 `setActive` 上重新绑定】`NativeHost::from_context` 只需要一个
    /// `FUnknown*` 去 QI `IReaperHostApplication`。组件初始化时 QI 成功得到的
    /// `IHostApplication` 引用在本组件生命周期内一直有效；宿主若在另一个线程上
    /// 调用 `setActive`，用同一个引用重建适配器就能把线程亲和性纠正到实际模型线程。
    pub(crate) fn host_raw(&self) -> *mut c_void {
        *self.host.lock().unwrap_or_else(|e| e.into_inner()) as *mut c_void
    }
    pub unsafe fn connect(&self, peer: *mut c_void) -> TResult {
        if peer.is_null() {
            return K_INVALID_ARGUMENT;
        }
        let owned = unsafe { Owned::retain(peer) };
        let old = std::mem::replace(
            &mut *self.peer.lock().unwrap_or_else(|e| e.into_inner()),
            owned.0,
        );
        std::mem::forget(owned);
        drop(Owned(old));
        K_RESULT_OK
    }
    pub fn disconnect(&self, peer: *mut c_void) -> TResult {
        let old = {
            let mut known = self.peer.lock().unwrap_or_else(|e| e.into_inner());
            if *known != peer as usize {
                return K_RESULT_FALSE;
            }
            std::mem::take(&mut *known)
        };
        drop(Owned(old));
        K_RESULT_OK
    }
    pub fn close(&self) {
        let peer = std::mem::take(&mut *self.peer.lock().unwrap_or_else(|e| e.into_inner()));
        let host = std::mem::take(&mut *self.host.lock().unwrap_or_else(|e| e.into_inner()));
        drop(Owned(peer));
        drop(Owned(host));
    }
    /// 在无内部锁时调用宿主，notify可同步重入另一个connection并请求回信。
    ///
    /// 【为什么消息对象由**我们**分配】`IMessage` 只是 6 个槽位（`FUnknown` 3 个 +
    /// 自己的 3 个），规范**建议**插件经 `IHostApplication::createInstance` 申请，
    /// 但这不是协议要求。实测宿主分两类：实现了消息工厂的（REAPER），与只回
    /// `kNotImplemented` 的。后者让这条握手静默失败 —— 而握手是编辑器与处理器之间
    /// **唯一**的身份来源，失败的表现是界面空着、每条命令都回
    /// 「FX editor not connected to its processor yet」。自带一份消息对象，
    /// 就把这个变量从「宿主是否实现某个可选工厂」变成「不需要」。
    pub fn send(&self, token: Option<&str>) -> TResult {
        let peer = {
            let peer = self.peer.lock().unwrap_or_else(|e| e.into_inner());
            unsafe { Owned::retain(*peer as *mut c_void) }
        };
        if peer.0 == 0 {
            // 【为什么是 Warn】这是「编辑器为什么没连上」的第一现场。此前的记录走
            // `log_line`（INFO），而 release 的默认级别是 Warn —— 写在 INFO 里等于没写，
            // 于是「宿主没 connect」与「宿主拒绝消息」在用户日志里完全同形。
            log::warn!("[gui] editor route: no peer connection; the route message was not sent");
            return K_RESULT_FALSE;
        }
        let direction = if token.is_some() {
            "processor->controller"
        } else {
            "controller->processor"
        };
        unsafe {
            let message = OwnedMessage::allocate(if token.is_some() {
                ROUTE_ID
            } else {
                REQUEST_ID
            });
            if let Some(token) = token {
                if !(*message).set_route(token) {
                    release_message(message.cast());
                    log::warn!("[gui] editor route: the route token was rejected ({direction})");
                    return K_RESULT_FALSE;
                }
                (*message).set_process_id(std::process::id() as i64);
            }
            let result =
                ((**peer.ptr().cast::<*const ConnectionVtbl>()).notify)(peer.ptr(), message.cast());
            release_message(message.cast());
            if result != K_RESULT_OK {
                log::warn!("[gui] editor route: {direction} notify rejected result={result:#x}");
            }
            result
        }
    }
}

/// 我们自己实现的 `Vst::IMessage`（`FUnknown` + `getMessageID` / `setMessageID` /
/// `getAttributes`）。
///
/// 【为什么引用计数从 1 起】`send` 是唯一的初始持有者：把裸指针交给 `notify`，
/// 返回后立刻 release 自己那一份。宿主若想长期保留，按 COM 规则必须先 `addRef` ——
/// 这也是它必须是堆对象、而不是栈上临时量的原因。
///
/// 【为什么 id 是 `*const c_char` 而不是 `String`】两个消息 ID 都是编译期常量
/// （`HiFiShifter.Editor.Route.v1` / `….RequestRoute.v1`），宿主只在 `notify`
/// 期间读它；存静态指针既省一次分配，也让「谁拥有这块内存」没有歧义。
#[repr(C)]
struct OwnedMessage {
    vtbl: *const MessageVtbl,
    refs: std::sync::atomic::AtomicU32,
    id: *const c_char,
    attrs: *mut OwnedAttributes,
}

impl OwnedMessage {
    /// 分配一条消息（引用计数 1，属性表 1 份引用）。
    fn allocate(id: &'static CStr) -> *mut OwnedMessage {
        let attrs = Box::into_raw(Box::new(OwnedAttributes {
            vtbl: &OWNED_ATTRIBUTES_VTBL,
            refs: std::sync::atomic::AtomicU32::new(1),
            route: Vec::new(),
            pid: 0,
        }));
        Box::into_raw(Box::new(Self {
            vtbl: &OWNED_MESSAGE_VTBL,
            refs: std::sync::atomic::AtomicU32::new(1),
            id: id.as_ptr(),
            attrs,
        }))
    }

    /// 写入路由令牌（UTF-16，无终结符 —— 终结由 `getString` 负责补）。
    ///
    /// # Safety
    /// `self` 必须是一条仍然存活的消息。
    unsafe fn set_route(&self, token: &str) -> bool {
        if self.attrs.is_null() {
            return false;
        }
        // SAFETY: attrs 由 allocate 分配，与消息同寿。
        let attrs = unsafe { &mut *self.attrs };
        attrs.route = token.encode_utf16().collect();
        true
    }

    /// # Safety
    /// 同上。
    unsafe fn set_process_id(&self, pid: i64) {
        if self.attrs.is_null() {
            return;
        }
        // SAFETY: 同上。
        unsafe { (*self.attrs).pid = pid };
    }
}

/// 消息的属性表。
///
/// 【为什么与消息分开分配】`getAttributes` 交出去的是**另一个** COM 接口：宿主会对它
/// 单独 `addRef` / `release`。让它们各自计数，是唯一不会在「宿主先放消息、后放属性」
/// 这种顺序下把内存释放两次的形状。
#[repr(C)]
struct OwnedAttributes {
    vtbl: *const AttributesVtbl,
    refs: std::sync::atomic::AtomicU32,
    /// 路由令牌（UTF-16，无终结符）。
    route: Vec<u16>,
    pid: i64,
}

const ROUTE_ATTRIBUTE: &CStr = c"hfs_editor_route";
const PID_ATTRIBUTE: &CStr = c"hfs_process_id";

/// `FUnknown` 的 IID（与 `vst3.rs` 里那份同值；本模块不导出它，避免两处耦合）。
const IID_UNKNOWN: [u32; 4] = [0x0000_0000, 0x0000_0000, 0xC000_0000, 0x0000_0046];

unsafe extern "system" fn message_query(
    this: *mut c_void,
    iid: *const u8,
    out: *mut *mut c_void,
) -> TResult {
    crate::guard_ffi(K_RESULT_FALSE, || {
        if out.is_null() || iid.is_null() || this.is_null() {
            return K_INVALID_ARGUMENT;
        }
        // SAFETY: out 是调用方提供的输出槽。
        unsafe { *out = std::ptr::null_mut() };
        // SAFETY: iid 指向 16 字节 TUID；两个候选 IID 都是本模块的常量。
        let hit = unsafe { iid_matches(iid, IID_MESSAGE) || iid_matches(iid, IID_UNKNOWN) };
        if !hit {
            return K_NO_INTERFACE;
        }
        // SAFETY: this 是 OwnedMessage；交出自己那一份的副本引用。
        unsafe {
            (*this.cast::<OwnedMessage>())
                .refs
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            *out = this;
        }
        K_RESULT_OK
    })
}

unsafe extern "system" fn message_add_ref(this: *mut c_void) -> u32 {
    crate::guard_ffi(0, || {
        // SAFETY: this 是仍然存活的消息对象。
        unsafe {
            (*this.cast::<OwnedMessage>())
                .refs
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
                + 1
        }
    })
}

unsafe extern "system" fn message_release(this: *mut c_void) -> u32 {
    crate::guard_ffi(0, || unsafe { release_message(this) })
}

/// 递减消息的引用计数；归零时连同属性表一起释放。
///
/// # Safety
/// `this` 必须来自 `OwnedMessage::allocate`（或它的某个已 addRef 的副本）。
unsafe fn release_message(this: *mut c_void) -> u32 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: 调用方保证指针来自 allocate。
    let message = unsafe { &mut *this.cast::<OwnedMessage>() };
    let previous = message
        .refs
        .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    if previous > 1 {
        return previous - 1;
    }
    let attrs = message.attrs;
    // SAFETY: 计数归零，没有其它持有者。
    unsafe { drop(Box::from_raw(this.cast::<OwnedMessage>())) };
    unsafe { release_attributes(attrs.cast()) };
    0
}

unsafe extern "system" fn message_id(this: *mut c_void) -> *const c_char {
    crate::guard_ffi(std::ptr::null(), || {
        // SAFETY: this 是仍然存活的消息对象。
        unsafe { (*this.cast::<OwnedMessage>()).id }
    })
}

/// `setMessageID` 对我们是空操作：消息 ID 决定这条消息的**语义**，而我们只发两种、
/// 都在分配时定好了。让宿主（或某个代理）改写它，就等于允许对端把「路由」读成
/// 「请求路由」，那是没有意义的组合。
unsafe extern "system" fn message_set_id(_this: *mut c_void, _id: *const c_char) {
    // 空操作也要带守卫：thread-model §1.1 的合同是「每个宿主可调入口自带 panic
    // 遏制」，不因为函数体平凡而豁免。
    crate::guard_ffi((), || {})
}

unsafe extern "system" fn message_attributes(this: *mut c_void) -> *mut c_void {
    crate::guard_ffi(std::ptr::null_mut(), || {
        if this.is_null() {
            return std::ptr::null_mut();
        }
        // SAFETY: this 是仍然存活的消息对象。
        let attrs = unsafe { (*this.cast::<OwnedMessage>()).attrs };
        if attrs.is_null() {
            return std::ptr::null_mut();
        }
        // SAFETY: attrs 与消息同寿；交出去的那一份由调用方 release。
        unsafe {
            (*attrs)
                .refs
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
        attrs.cast()
    })
}

unsafe extern "system" fn attributes_query(
    this: *mut c_void,
    iid: *const u8,
    out: *mut *mut c_void,
) -> TResult {
    crate::guard_ffi(K_RESULT_FALSE, || {
        if out.is_null() || iid.is_null() || this.is_null() {
            return K_INVALID_ARGUMENT;
        }
        // SAFETY: out 是调用方提供的输出槽。
        unsafe { *out = std::ptr::null_mut() };
        // SAFETY: iid 指向 16 字节 TUID。
        if !unsafe { iid_matches(iid, IID_ATTRIBUTE_LIST) || iid_matches(iid, IID_UNKNOWN) } {
            return K_NO_INTERFACE;
        }
        // SAFETY: this 是 OwnedAttributes。
        unsafe {
            (*this.cast::<OwnedAttributes>())
                .refs
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            *out = this;
        }
        K_RESULT_OK
    })
}

unsafe extern "system" fn attributes_add_ref(this: *mut c_void) -> u32 {
    crate::guard_ffi(0, || {
        // SAFETY: this 是仍然存活的属性表。
        unsafe {
            (*this.cast::<OwnedAttributes>())
                .refs
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
                + 1
        }
    })
}

unsafe extern "system" fn attributes_release(this: *mut c_void) -> u32 {
    crate::guard_ffi(0, || unsafe { release_attributes(this) })
}

/// # Safety
/// `this` 必须来自 `OwnedMessage::allocate` 的属性表（或它的某个已 addRef 的副本）。
unsafe fn release_attributes(this: *mut c_void) -> u32 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: 调用方保证指针来自 allocate。
    let attrs = unsafe { &mut *this.cast::<OwnedAttributes>() };
    let previous = attrs.refs.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    if previous > 1 {
        return previous - 1;
    }
    // SAFETY: 计数归零，没有其它持有者。
    unsafe { drop(Box::from_raw(this.cast::<OwnedAttributes>())) };
    0
}

/// 读一个以 NUL 结尾的 UTF-16 串；超出上限即拒绝（不做无界读）。
///
/// # Safety
/// `value` 必须是调用方可读的、以 0 结尾的 `TChar*`（Windows 上是 char16）。
unsafe fn read_tchar(value: *const u16, limit: usize) -> Option<Vec<u16>> {
    if value.is_null() {
        return None;
    }
    let mut units = Vec::new();
    for index in 0..=limit {
        // SAFETY: 调用方保证可读到第一个 0 为止；limit 是防御上界。
        let unit = unsafe { *value.add(index) };
        if unit == 0 {
            return Some(units);
        }
        units.push(unit);
    }
    None
}

unsafe extern "system" fn attributes_set_int(
    this: *mut c_void,
    key: *const c_char,
    value: i64,
) -> TResult {
    crate::guard_ffi(K_RESULT_FALSE, || {
        if this.is_null() || key.is_null() {
            return K_INVALID_ARGUMENT;
        }
        // SAFETY: key 是调用方给的 NUL 结尾 C 串。
        if unsafe { CStr::from_ptr(key) } != PID_ATTRIBUTE {
            return K_RESULT_FALSE;
        }
        // SAFETY: this 是仍然存活的属性表。
        unsafe { (*this.cast::<OwnedAttributes>()).pid = value };
        K_RESULT_OK
    })
}

unsafe extern "system" fn attributes_get_int(
    this: *mut c_void,
    key: *const c_char,
    value: *mut i64,
) -> TResult {
    crate::guard_ffi(K_RESULT_FALSE, || {
        if this.is_null() || key.is_null() || value.is_null() {
            return K_INVALID_ARGUMENT;
        }
        // SAFETY: key 是调用方给的 NUL 结尾 C 串。
        if unsafe { CStr::from_ptr(key) } != PID_ATTRIBUTE {
            return K_RESULT_FALSE;
        }
        // SAFETY: value 是调用方提供的输出槽。
        unsafe { *value = (*this.cast::<OwnedAttributes>()).pid };
        K_RESULT_OK
    })
}

/// `setString` 的 `TChar*` 在 Windows 上是 UTF-16、且**以 NUL 结尾**（无长度参数）。
/// 这与 `ivstattributes.h` 的声明逐一对应：错一位就会把内存读穿。
unsafe extern "system" fn attributes_set_string(
    this: *mut c_void,
    key: *const c_char,
    value: *const u16,
) -> TResult {
    crate::guard_ffi(K_RESULT_FALSE, || {
        if this.is_null() || key.is_null() {
            return K_INVALID_ARGUMENT;
        }
        // SAFETY: key 是调用方给的 NUL 结尾 C 串。
        if unsafe { CStr::from_ptr(key) } != ROUTE_ATTRIBUTE {
            return K_RESULT_FALSE;
        }
        // SAFETY: 路由令牌由我们自己写入，最长 64 个 UTF-16 单元；留一倍余量。
        let Some(units) = (unsafe { read_tchar(value, 256) }) else {
            return K_INVALID_ARGUMENT;
        };
        // SAFETY: this 是仍然存活的属性表。
        unsafe { (*this.cast::<OwnedAttributes>()).route = units };
        K_RESULT_OK
    })
}

/// `getString` 的 size 单位是**字节**（不是字符），且调用方保证缓冲够大。
unsafe extern "system" fn attributes_get_string(
    this: *mut c_void,
    key: *const c_char,
    buffer: *mut u16,
    size_in_bytes: u32,
) -> TResult {
    crate::guard_ffi(K_RESULT_FALSE, || {
        if this.is_null() || key.is_null() || buffer.is_null() {
            return K_INVALID_ARGUMENT;
        }
        // SAFETY: key 是调用方给的 NUL 结尾 C 串。
        if unsafe { CStr::from_ptr(key) } != ROUTE_ATTRIBUTE {
            return K_RESULT_FALSE;
        }
        let capacity = (size_in_bytes as usize) / 2;
        if capacity == 0 {
            return K_INVALID_ARGUMENT;
        }
        // SAFETY: this 是仍然存活的属性表；缓冲容量由调用方保证。
        unsafe {
            let attrs = &*this.cast::<OwnedAttributes>();
            let copy = attrs.route.len().min(capacity - 1);
            std::ptr::copy_nonoverlapping(attrs.route.as_ptr(), buffer, copy);
            *buffer.add(copy) = 0;
        }
        K_RESULT_OK
    })
}

// 数值/二进制属性我们既不发布也不消费（属性表只承载「编辑器路由」一个字符串键，
// 见 attributes_query），四个入口恒回未实现。入口平凡不代表可以裸奔：守卫是
// thread-model §1.1 的合同，漏一个 check-ffi-guards.py 就全量红。
unsafe extern "system" fn attributes_set_float(
    _this: *mut c_void,
    _key: *const c_char,
    _value: f64,
) -> TResult {
    crate::guard_ffi(K_NOT_IMPLEMENTED, || K_NOT_IMPLEMENTED)
}

unsafe extern "system" fn attributes_get_float(
    _this: *mut c_void,
    _key: *const c_char,
    _value: *mut f64,
) -> TResult {
    crate::guard_ffi(K_NOT_IMPLEMENTED, || K_NOT_IMPLEMENTED)
}

unsafe extern "system" fn attributes_set_binary(
    _this: *mut c_void,
    _key: *const c_char,
    _data: *const c_void,
    _size: u32,
) -> TResult {
    crate::guard_ffi(K_NOT_IMPLEMENTED, || K_NOT_IMPLEMENTED)
}

unsafe extern "system" fn attributes_get_binary(
    _this: *mut c_void,
    _key: *const c_char,
    _data: *mut *const c_void,
    _size: *mut u32,
) -> TResult {
    crate::guard_ffi(K_NOT_IMPLEMENTED, || K_NOT_IMPLEMENTED)
}

/// `Vst::IAttributeList` 的 IID。
const IID_ATTRIBUTE_LIST: [u32; 4] = [0x1E5F_0AEB, 0xCC7F_4533, 0xA254_4011, 0x38AD_5EE4];

static OWNED_MESSAGE_VTBL: MessageVtbl = MessageVtbl {
    base: UnknownVtbl {
        query: message_query,
        add: message_add_ref,
        release: message_release,
    },
    id: message_id,
    set_id: message_set_id,
    attributes: message_attributes,
};

static OWNED_ATTRIBUTES_VTBL: AttributesVtbl = AttributesVtbl {
    base: UnknownVtbl {
        query: attributes_query,
        add: attributes_add_ref,
        release: attributes_release,
    },
    set_int: attributes_set_int,
    get_int: attributes_get_int,
    set_float: attributes_set_float,
    get_float: attributes_get_float,
    set_string: attributes_set_string,
    get_string: attributes_get_string,
    set_binary: attributes_set_binary,
    get_binary: attributes_get_binary,
};

impl Drop for ConnectionState {
    fn drop(&mut self) {
        self.close();
    }
}

pub(crate) enum Message {
    RequestRoute,
    Route { pid: i64, token: String },
}
/// 读取宿主拥有的活message；严格按UTF16字节容量和NUL边界处理属性。
pub(crate) unsafe fn read(message: *mut c_void) -> Result<Message, String> {
    if message.is_null() {
        return Err("null VST3 message".into());
    }
    let table = unsafe { &**message.cast::<*const MessageVtbl>() };
    let id = unsafe { (table.id)(message) };
    if id.is_null() {
        return Err("null message ID".into());
    }
    let id = unsafe { CStr::from_ptr(id) };
    if id == REQUEST_ID {
        return Ok(Message::RequestRoute);
    }
    if id != ROUTE_ID {
        return Err("unknown VST3 message".into());
    }
    let attrs = unsafe { (table.attributes)(message) };
    if attrs.is_null() {
        return Err("route attributes missing".into());
    }
    let table = unsafe { &**attrs.cast::<*const AttributesVtbl>() };
    let mut pid = 0_i64;
    let mut wide = [0_u16; 128];
    if unsafe { (table.get_int)(attrs, c"hfs_process_id".as_ptr(), &mut pid) } != K_RESULT_OK
        || unsafe {
            (table.get_string)(attrs, c"hfs_editor_route".as_ptr(), wide.as_mut_ptr(), 256)
        } != K_RESULT_OK
    {
        return Err("route attributes invalid".into());
    }
    let length = wide
        .iter()
        .position(|c| *c == 0)
        .ok_or("unterminated route token")?;
    Ok(Message::Route {
        pid,
        token: String::from_utf16(&wide[..length]).map_err(|e| e.to_string())?,
    })
}
/// 判断IID仅在真实queryInterface入口使用，不把其它VST对象误当connection。
pub(crate) unsafe fn is_connection(iid: *const u8) -> bool {
    unsafe { iid_matches(iid, IID_CONNECTION) }
}

#[cfg(test)]
mod owned_message_tests {
    use super::*;

    const TOKEN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    /// **不变量：自带的消息对象与自带的 reader 使用同一套 ABI。**
    ///
    /// 【为什么钉死】这两个半边的字段顺序、`TChar` 宽度、`getString` 的字节语义
    /// 只要有一处对不上，表现都是「握手静默失败」—— 而它不是崩溃，是一整块空界面。
    #[test]
    fn an_owned_message_round_trips_the_route_attributes() {
        // SAFETY: 消息由 allocate 分配，read 只在本线程内读取它。
        unsafe {
            let message = OwnedMessage::allocate(ROUTE_ID);
            assert!((*message).set_route(TOKEN), "route token must be accepted");
            (*message).set_process_id(4321);
            match read(message.cast()) {
                Ok(Message::Route { pid, token }) => {
                    assert_eq!(pid, 4321);
                    assert_eq!(token, TOKEN);
                }
                Ok(Message::RequestRoute) => panic!("the message ID was not preserved"),
                Err(error) => panic!("the owned message was not readable: {error}"),
            }
            release_message(message.cast());
        }
    }

    /// **不变量：消息与属性表各自计数。**
    ///
    /// 【为什么必须分开】宿主完全可以先放消息、后放属性（或反序）。共用一份计数时，
    /// 先放的那一个会把另一个一起释放 —— 那是 use-after-free，只在宿主恰好这个顺序时出现。
    #[test]
    fn message_and_attribute_references_are_counted_separately() {
        // SAFETY: 全程单线程，指针都在本用例里产生与释放。
        unsafe {
            let message = OwnedMessage::allocate(ROUTE_ID);
            let attrs = message_attributes(message.cast());
            assert!(!attrs.is_null());
            assert_eq!(
                (*attrs.cast::<OwnedAttributes>())
                    .refs
                    .load(std::sync::atomic::Ordering::Acquire),
                2,
                "getAttributes 必须交出自己的一份引用"
            );
            // 宿主先放消息：属性表还剩 getAttributes 那一份。
            release_message(message.cast());
            assert_eq!(
                (*attrs.cast::<OwnedAttributes>())
                    .refs
                    .load(std::sync::atomic::Ordering::Acquire),
                1,
                "消息释放不能连带把属性表放掉"
            );
            attributes_release(attrs);
        }
    }

    /// 未知属性名必须被拒绝，而不是写进一个随便的位置。
    #[test]
    fn unknown_attribute_keys_are_refused() {
        // SAFETY: 同上。
        unsafe {
            let message = OwnedMessage::allocate(ROUTE_ID);
            let attrs = message_attributes(message.cast());
            let wide: Vec<u16> = TOKEN.encode_utf16().chain(Some(0)).collect();
            assert_eq!(
                attributes_set_string(attrs, c"not_a_route".as_ptr(), wide.as_ptr()),
                K_RESULT_FALSE
            );
            let mut pid = 7_i64;
            assert_eq!(
                attributes_get_int(attrs, c"not_a_pid".as_ptr(), &mut pid),
                K_RESULT_FALSE
            );
            assert_eq!(pid, 7, "被拒绝的读不能改动输出槽");
            attributes_release(attrs);
            release_message(message.cast());
        }
    }
}

#[cfg(test)]
mod host_handshake_tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    use std::sync::{Arc, Mutex};

    /// 一个**不实现** `IHostApplication::createInstance` 的宿主。
    ///
    /// 【为什么这是关键样本】它正是现场里「界面能开、时间轴全空」的那类宿主：
    /// 连接点齐全，消息工厂缺席。此前 `send` 会在这里返回 `K_RESULT_FALSE`，
    /// 于是编辑器永远没有 owner，每条命令都回「FX editor not connected to its
    /// processor yet」。
    #[repr(C)]
    struct FakeHost {
        vtbl: *const HostVtbl,
        refs: AtomicU32,
    }

    /// 宿主放在两个组件之间的代理连接点。
    #[repr(C)]
    struct FakePeer {
        vtbl: *const ConnectionVtbl,
        refs: AtomicU32,
        seen: Arc<Mutex<Option<(String, i64)>>>,
    }

    unsafe extern "system" fn fake_query(
        _this: *mut c_void,
        _iid: *const u8,
        out: *mut *mut c_void,
    ) -> TResult {
        if !out.is_null() {
            // SAFETY: out 是调用方提供的输出槽。
            unsafe { *out = std::ptr::null_mut() };
        }
        K_NO_INTERFACE
    }

    unsafe extern "system" fn fake_add_ref(_this: *mut c_void) -> u32 {
        2
    }

    unsafe extern "system" fn fake_release(_this: *mut c_void) -> u32 {
        1
    }

    unsafe extern "system" fn fake_host_name(_this: *mut c_void, _name: *mut u16) -> TResult {
        K_RESULT_OK
    }

    /// **这里就是那个兼容性分水岭**：真实宿主在这一格回 `kNotImplemented`。
    unsafe extern "system" fn fake_host_create(
        _this: *mut c_void,
        _cid: *const u8,
        _iid: *const u8,
        out: *mut *mut c_void,
    ) -> TResult {
        if !out.is_null() {
            // SAFETY: out 是调用方提供的输出槽。
            unsafe { *out = std::ptr::null_mut() };
        }
        K_NOT_IMPLEMENTED
    }

    unsafe extern "system" fn fake_connect(_this: *mut c_void, _other: *mut c_void) -> TResult {
        K_RESULT_OK
    }

    unsafe extern "system" fn fake_disconnect(_this: *mut c_void, _other: *mut c_void) -> TResult {
        K_RESULT_OK
    }

    unsafe extern "system" fn fake_notify(this: *mut c_void, message: *mut c_void) -> TResult {
        // SAFETY: this 是 FakePeer；message 由 send 构造并在 notify 期间有效。
        let peer = unsafe { &*(this as *const FakePeer) };
        match unsafe { read(message) } {
            Ok(Message::Route { pid, token }) => {
                *peer.seen.lock().unwrap() = Some((token, pid));
                K_RESULT_OK
            }
            _ => K_RESULT_FALSE,
        }
    }

    static FAKE_HOST_VTBL: HostVtbl = HostVtbl {
        base: UnknownVtbl {
            query: fake_query,
            add: fake_add_ref,
            release: fake_release,
        },
        name: fake_host_name,
        create: fake_host_create,
    };

    static FAKE_PEER_VTBL: ConnectionVtbl = ConnectionVtbl {
        base: UnknownVtbl {
            query: fake_query,
            add: fake_add_ref,
            release: fake_release,
        },
        connect: fake_connect,
        disconnect: fake_disconnect,
        notify: fake_notify,
    };

    const TOKEN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    /// **不变量：宿主没有消息工厂时，路由仍然送达。**
    ///
    /// 【为什么钉死】这一格是「非 ARA 宿主里编辑器完全不可用」的直接原因。修复之后，
    /// 整条链路不再依赖宿主的可选能力：我们自带消息对象，宿主只需要实现它本来就必须
    /// 实现的 `IConnectionPoint`。
    #[test]
    fn a_route_is_delivered_even_when_the_host_has_no_message_factory() {
        let seen = Arc::new(Mutex::new(None));
        let host = Box::new(FakeHost {
            vtbl: &FAKE_HOST_VTBL,
            refs: AtomicU32::new(1),
        });
        let peer = Box::new(FakePeer {
            vtbl: &FAKE_PEER_VTBL,
            refs: AtomicU32::new(1),
            seen: Arc::clone(&seen),
        });
        let state = ConnectionState::default();
        // SAFETY: 两个假对象都活到本用例结束；ConnectionState 只持有它们的引用。
        unsafe {
            let host_ptr = (&*host as *const FakeHost).cast::<c_void>() as *mut c_void;
            let peer_ptr = (&*peer as *const FakePeer).cast::<c_void>() as *mut c_void;
            state.initialize(host_ptr);
            assert_eq!(state.connect(peer_ptr), K_RESULT_OK);
            assert_eq!(
                state.send(Some(TOKEN)),
                K_RESULT_OK,
                "宿主没有消息工厂时也必须送得出去"
            );
        }
        let captured = seen
            .lock()
            .unwrap()
            .clone()
            .expect("代理必须收到一条可读的 Route 消息");
        assert_eq!(captured.0, TOKEN);
        assert_eq!(captured.1, std::process::id() as i64);
    }

    /// **不变量：没有对端时如实失败，而不是假装发过。**
    ///
    /// 【为什么钉死】静默返回失败正是这条链路此前最难查的形态：日志里什么都没有，
    /// 而用户看到的只是「编辑器没连上」。现在它至少有一步明确的拒绝。
    #[test]
    fn a_route_without_a_peer_is_refused() {
        let host = Box::new(FakeHost {
            vtbl: &FAKE_HOST_VTBL,
            refs: AtomicU32::new(1),
        });
        let state = ConnectionState::default();
        // SAFETY: 同上。
        unsafe {
            let host_ptr = (&*host as *const FakeHost).cast::<c_void>() as *mut c_void;
            state.initialize(host_ptr);
            assert_eq!(state.send(Some(TOKEN)), K_RESULT_FALSE);
        }
    }
}

#[cfg(test)]
mod ffi_guard_tests {
    use super::*;

    /// **不变量：每个宿主可调入口都自带 panic 遏制，且对垃圾指针无害。**
    ///
    /// 【为什么钉死】thread-model §1.1 的合同是「panic 越过 FFI 边界 = abort = DAW
    /// 崩溃」。编辑器消息这一批入口是后加的，此前漏包守卫让全量门禁红了一轮；
    /// 这里把五个入口对非法指针的行为钉住——既验证守卫在（不 panic），
    /// 也验证 stub 的诚实返回值不被将来顺手改掉。
    #[test]
    fn the_trivial_entries_survive_garbage_pointers() {
        // SAFETY: 传入的"宿主指针"全是悬空/垃圾值，正是要用守卫兜住的情形；
        // 五个入口都不解引用 this，panic 与否只取决于守卫是否存在。
        // 悬空指针用 std::ptr::dangling_mut 造出来：只比较/传递、绝不解引用。
        let garbage_a = std::ptr::dangling_mut::<c_void>();
        let garbage_b = std::ptr::dangling_mut::<c_void>();
        let garbage_c = std::ptr::dangling_mut::<c_void>();
        unsafe {
            message_set_id(garbage_a, c"anything".as_ptr());

            assert_eq!(
                attributes_set_float(garbage_b, c"not_a_route".as_ptr(), 1.5),
                K_NOT_IMPLEMENTED
            );
            assert_eq!(
                attributes_get_float(garbage_b, std::ptr::null(), std::ptr::null_mut()),
                K_NOT_IMPLEMENTED
            );
            assert_eq!(
                attributes_set_binary(std::ptr::null_mut(), std::ptr::null(), std::ptr::null(), 0),
                K_NOT_IMPLEMENTED
            );
            assert_eq!(
                attributes_get_binary(
                    garbage_c,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut()
                ),
                K_NOT_IMPLEMENTED
            );
        }
    }
}
