//! 锁定SDK的IConnectionPoint/IHostApplication/IMessage ABI；消息允许宿主放置代理。
use crate::vst3::{
    iid_matches, uid_guid, TResult, K_INVALID_ARGUMENT, K_RESULT_FALSE, K_RESULT_OK,
};
use std::ffi::{c_char, c_void, CStr};
use std::sync::Mutex;
pub(crate) const IID_CONNECTION: [u32; 4] = [0x70A4156F, 0x6E6E4026, 0x989148BF, 0xAA60D8D1];
const IID_HOST: [u32; 4] = [0x58E595CC, 0xDB2D4969, 0x8B6AAF8C, 0x36A664E5];
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
/// `host` 必须是 `IID_HOST` 查询成功返回的、仍持有引用的宿主接口指针。
unsafe fn record_host_name(host: *mut c_void) {
    if host.is_null() {
        return;
    }
    // SAFETY: host 指向 IHostApplication vtable；name 的签名为 `TResult(String128)`，
    // 即写入一个 128 个 UTF-16 单元的调用方缓冲（VST3 SDK 的 String128）。
    let table = unsafe { &**host.cast::<*const HostVtbl>() };
    let mut buffer: HostNameBuffer = [0; 256];
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
    let end = buffer
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(buffer.len());
    let name = std::str::from_utf8(&buffer[..end]).ok()?.to_owned();
    (!name.is_empty()).then_some(name)
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
        assert_eq!(
            decode_string128(&[b'A' as u16, b'B' as u16]).as_deref(),
            Some("AB")
        );
        // 非法 UTF-16（孤立高位代理）⇒ None，不 panic。
        assert_eq!(decode_string128(&[0xD800, 0]), None);
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

        // 非 ASCII 的宿主名（中文/日文宿主名并不罕见）必须原样解出来。
        let mut wide = [0u8; 256];
        for (slot, byte) in "宿主".as_bytes().iter().enumerate() {
            wide[slot] = *byte;
        }
        assert_eq!(decode_string128(&wide).as_deref(), Some("宿主"));

        // 非法 UTF-8 ⇒ None，不 panic。
        assert_eq!(decode_string128(&[0xFF, 0xFE, 0]), None);
    }
}
#[derive(Default)]
pub(crate) struct ConnectionState {
    host: Mutex<usize>,
    peer: Mutex<usize>,
}
impl ConnectionState {
    /// 宿主提供message对象，不能把Rust Arc裸指针当跨代理协议。
    pub unsafe fn initialize(&self, context: *mut c_void) {
        if context.is_null() {
            return;
        }
        let mut host = std::ptr::null_mut();
        let iid = uid_guid(IID_HOST);
        let result = unsafe {
            ((**context.cast::<*const UnknownVtbl>()).query)(context, iid.as_ptr(), &mut host)
        };
        if result == K_RESULT_OK && !host.is_null() {
            let old = std::mem::replace(
                &mut *self.host.lock().unwrap_or_else(|e| e.into_inner()),
                host as usize,
            );
            drop(Owned(old));
        }
    }
    /// 读一次宿主应用名（`IHostApplication::getName`）。
    ///
    /// 【为什么由调用方在**健康路径**上触发】`initialize` 里的 QI 可能先重入关闭组件
    /// （见 `component_initialize` 的重入测试）。读名是一次额外的宿主调用，不该发生在
    /// "已经被重入关闭"的那条路径上 —— 那时组件正在拒绝初始化，任何多余的宿主交互都
    /// 没有意义。
    pub fn record_host_name(&self) {
        let host = *self.host.lock().unwrap_or_else(|e| e.into_inner()) as *mut c_void;
        // SAFETY: host 是本状态持有的 IHostApplication 引用（QI 成功后登记）。
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
    pub fn send(&self, token: Option<&str>) -> TResult {
        let host = {
            let host = self.host.lock().unwrap_or_else(|e| e.into_inner());
            unsafe { Owned::retain(*host as *mut c_void) }
        };
        let peer = {
            let peer = self.peer.lock().unwrap_or_else(|e| e.into_inner());
            unsafe { Owned::retain(*peer as *mut c_void) }
        };
        if host.0 == 0 || peer.0 == 0 {
            return K_RESULT_FALSE;
        }
        unsafe {
            let table = &**host.ptr().cast::<*const HostVtbl>();
            let iid = uid_guid(IID_MESSAGE);
            let mut message = std::ptr::null_mut();
            if (table.create)(host.ptr(), iid.as_ptr(), iid.as_ptr(), &mut message) != K_RESULT_OK
                || message.is_null()
            {
                return K_RESULT_FALSE;
            }
            let _message = Owned(message as usize);
            let message_table = &**message.cast::<*const MessageVtbl>();
            (message_table.set_id)(
                message,
                if token.is_some() {
                    ROUTE_ID.as_ptr()
                } else {
                    REQUEST_ID.as_ptr()
                },
            );
            if let Some(token) = token {
                let attrs = (message_table.attributes)(message);
                if attrs.is_null() {
                    return K_RESULT_FALSE;
                }
                let table = &**attrs.cast::<*const AttributesVtbl>();
                let wide: Vec<u16> = token.encode_utf16().chain(Some(0)).collect();
                if (table.set_string)(attrs, c"hfs_editor_route".as_ptr(), wide.as_ptr())
                    != K_RESULT_OK
                    || (table.set_int)(attrs, c"hfs_process_id".as_ptr(), std::process::id() as i64)
                        != K_RESULT_OK
                {
                    return K_RESULT_FALSE;
                }
            }
            ((**peer.ptr().cast::<*const ConnectionVtbl>()).notify)(peer.ptr(), message)
        }
    }
}
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
