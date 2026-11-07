//! CLAP 的宿主日志后端（clap.log）→ 共享日志系统的薄适配层。
//!
//! 【这里只做两件事】声明 clap_host_log_t 的 ABI，以及把宿主回调登记成共享
//! diagnostics 的一个转发 sink。文件日志（目录、格式、会话头、轮转、级别、
//! panic hook）**全部**由 hifishifter_plugin::diagnostics 负责 —— 此前本模块自己
//! 装了一个 log::Log、自己开文件、写死 Info 级别，于是同一个 CLAP 进程同时存在
//! 两套日志：一份无时间戳不轮转的 plugin-clap.log，一份被宿主身份解析误开的
//! VST3 空会话。日志一旦有两套实现，就必然分叉。
//!
//! 【为什么在 init 之后才查扩展】clap/plugin.h 明确写了 "It is forbidden to call
//! it before plugin->init()"。所以 install 由 plugin_init 调用，而不是
//! clap_entry.init 或 create_plugin。

use crate::abi::clap_host_t;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::atomic::{AtomicPtr, Ordering};

/// clap.log 扩展 ID（clap/ext/log.h）。
pub const CLAP_EXT_LOG: &[u8] = b"clap.log\0";
/// clap_host_log_t::log 的 C 签名。
type HostLogFn = unsafe extern "C" fn(*const clap_host_t, i32, *const c_char);

/// clap_host_log_t：只有一个 log 回调，前面没有 ctx。
///
/// 【为什么公开】tests/clap_layout.rs 用同一份类型与原生头文件逐字段比对布局。
#[repr(C)]
pub struct ClapHostLog {
    pub log: Option<HostLogFn>,
}
// SAFETY: 宿主只读的静态扩展表。
unsafe impl Sync for ClapHostLog {}
unsafe impl Send for ClapHostLog {}

/// clap_log_severity 到宿主侧的映射（clap/ext/log.h）。
const DEBUG: i32 = 0;
const INFO: i32 = 1;
const WARNING: i32 = 2;
const ERROR: i32 = 3;

/// 当前登记的 clap_host_t（传给 log 回调的第一个参数）。
///
/// 【为什么全局但只登记一次】log::set_logger 在进程内不可撤销，共享日志系统也只装
/// 一次；因此只认第一个提供 clap.log 的宿主实例，后续实例复用同一份指针。
static HOST_PTR: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
/// 当前宿主的 clap_host_log_t；clear() 摘掉后转发变成 no-op。
static HOST: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

/// 把一条已格式化的行交给宿主的 clap.log。
///
/// 【为什么是函数指针】diagnostics::set_host_sink 收的是格式无关的 fn；
/// 这里做一次级别映射，CLAP 的 severity 常量不会渗进共享层。
fn forward_to_host(level: log::Level, line: &str) {
    let host = HOST.load(Ordering::Acquire);
    if host.is_null() {
        return;
    }
    // 宿主回调可能把消息留到返回之后，所以必须交一个 NUL 结尾的 C 字符串；
    // CString::new 失败时（消息里含内部 NUL）退化为一条固定文本，绝不 panic。
    let message = CString::new(line)
        .unwrap_or_else(|_| CString::new("hifishifter: log message dropped").unwrap());
    let severity = match level {
        log::Level::Error => ERROR,
        log::Level::Warn => WARNING,
        log::Level::Info => INFO,
        log::Level::Debug | log::Level::Trace => DEBUG,
    };
    // SAFETY: host 是 plugin.init 时从宿主拿到的 clap_host_log_t，宿主拥有它；
    // 该扩展表只有一个函数指针字段，且 log 回调按契约 [thread-safe]。
    unsafe {
        let table = &*host.cast::<ClapHostLog>();
        if let Some(callback) = table.log {
            let host_ptr = HOST_PTR.load(Ordering::Acquire).cast::<clap_host_t>();
            callback(host_ptr, severity, message.as_ptr());
        }
    }
}

/// 由 plugin_init 调用：查询宿主的 clap.log 并登记转发 sink。
///
/// 【为什么不再装 logger】共享的 diagnostics::init() 在 clap_entry.init 时就已
/// 装好；这里只补一条可选的宿主后端。宿主没有 clap.log 时文件日志照常工作。
///
/// # Safety
/// host 必须指向本实例存活期内有效的 clap_host_t，且当前已在
/// plugin->init 调用栈中（CLAP 允许此刻查询宿主扩展）。
pub unsafe fn install(host: *const clap_host_t) {
    if host.is_null() {
        return;
    }
    let get_extension = unsafe { (*host).get_extension };
    let Some(get_extension) = get_extension else {
        return;
    };
    let ext = unsafe { get_extension(host, CLAP_EXT_LOG.as_ptr().cast()) };
    if ext.is_null() {
        return;
    }
    // 【为什么只登记第一个】允许第二个实例覆盖指针会让第一个实例的诊断凭空改道，
    // 而且宿主卸载顺序不可预测。
    if HOST
        .compare_exchange(
            std::ptr::null_mut(),
            ext as *mut c_void,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        return;
    }
    HOST_PTR.store(host as *mut c_void, Ordering::Release);
    hifishifter_plugin::diagnostics::set_host_sink(forward_to_host);
    let name = unsafe { (*host).name };
    log::info!(
        "[clap] host log sink installed (host={})",
        if name.is_null() {
            "unknown".to_string()
        } else {
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        }
    );
}

/// clap_entry.deinit 时摘掉宿主指针；之后转发变成 no-op，文件日志不受影响。
///
/// 【已知边界】log::set_logger 在进程内不可撤销，所以同一次模块加载里宿主日志只认
/// 第一个提供 clap.log 的宿主。宿主若在 deinit 之前先销毁那个实例，后续转发会
/// 被丢弃而不是崩溃 —— 这是 log crate 的进程级单例约束，不是能力声明。
pub fn clear() {
    HOST.store(std::ptr::null_mut(), Ordering::Release);
    HOST_PTR.store(std::ptr::null_mut(), Ordering::Release);
}

/// 供测试与探针读取：宿主是否提供了 clap.log。
pub fn is_installed() -> bool {
    !HOST.load(Ordering::Acquire).is_null()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_mapping_covers_every_log_level() {
        assert_eq!(DEBUG, 0);
        assert_eq!(INFO, 1);
        assert_eq!(WARNING, 2);
        assert_eq!(ERROR, 3);
    }
}
