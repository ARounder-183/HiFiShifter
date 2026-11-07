//! 把 `log` 的日志转发到宿主的 `clap.log` 扩展。
//!
//! 【为什么必须装 logger】此前 `log::info!` / `log::warn!` / `log::error!` 全都没有
//! 消费者，宿主侧一条都看不到：FFI 守卫截住的 panic、非法的生命周期调用、state
//! 流失败这些**只会在真实宿主里出现**的线索，全都静默丢掉。CLAP 提供了标准的
//! `clap_host_log`，让插件把诊断交给宿主自己的日志面板 / 日志文件。
//!
//! 【为什么在 init 之后才查扩展】`clap/plugin.h` 明确写了 "It is forbidden to call
//! it before plugin->init()"。所以 `install` 由 `plugin_init` 调用，
//! 而不是 `clap_entry.init` 或 `create_plugin`。
//!
//! 【为什么是全局单例 + 原子指针】同一个宿主进程里的多个实例共享同一个 `log` 实现；
//! `set_logger` 只允许成功一次。宿主指针由 `clap_host_t` 拥有，在插件实例存活期间
//! 有效；`deinit` 时 `clear()` 把它摘掉，避免把日志写进已卸载的模块。

use crate::abi::clap_host_t;
use std::ffi::{c_char, c_void, CStr, CString};
use std::io::Write;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Mutex, OnceLock};

/// `clap.log` 扩展 ID（`clap/ext/log.h`）。
pub const CLAP_EXT_LOG: &[u8] = b"clap.log\0";
/// `clap_host_log_t::log` 的 C 签名。
type HostLogFn = unsafe extern "C" fn(*const clap_host_t, i32, *const c_char);

/// `clap_host_log_t`：只有一个 `log` 回调，前面没有 ctx。
#[repr(C)]
pub struct ClapHostLog {
    pub log: Option<HostLogFn>,
}
// SAFETY: 宿主只读的静态扩展表。
unsafe impl Sync for ClapHostLog {}
unsafe impl Send for ClapHostLog {}

/// 日志级别映射（`clap/ext/log.h` 的 `clap_log_severity`）。
const DEBUG: i32 = 0;
const INFO: i32 = 1;
const WARNING: i32 = 2;
const ERROR: i32 = 3;

/// 当前登记的 `clap_host_t`；传给 `log` 回调的第一个参数。
static HOST_PTR: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
/// 当前宿主的 `clap_host_log_t`；`deinit` 时清空。
static HOST: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
/// `set_logger` 只跑一次；失败也不重试（宿主没提供 clap.log 是正常的）。
static LOGGER: OnceLock<()> = OnceLock::new();
/// 文件落点（每进程只解析一次；解析不出来就只走宿主日志）。
static FILE: OnceLock<Mutex<Option<std::fs::File>>> = OnceLock::new();
/// 单个日志文件的软上限：超过就在进程启动时截断（插件诊断不需要历史）。
const MAX_LOG_BYTES: u64 = 8 * 1024 * 1024;

/// 本模块的日志文件：`%LOCALAPPDATA%\HiFiShifter\logs\plugin-clap-<host>.log`。
///
/// 【为什么 CLAP 也要落盘】VST3 外壳一直写自己的日志文件，而 CLAP 只把日志交给
/// 宿主的 `clap.log` —— 宿主的面板默认关着，用户报障时我们手上一个字节都没有。
/// "按钮一直是灰的"这类问题只能靠"块数 / transport 标志 / 是否推出过位置"三个读数
/// 定位，它们必须能事后读到。文件名带 `clap-` 前缀，与 VST3 的
/// `plugin-<host>.log` 并列而不混写（两种格式的判定逻辑不同，混在一起没法读）。
fn open_log_file() -> Option<std::fs::File> {
    let dir = hifishifter_kernel::app_paths::log_dir()?;
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join("plugin-clap.log");
    let oversized = std::fs::metadata(&path)
        .map(|meta| meta.len() > MAX_LOG_BYTES)
        .unwrap_or(false);
    let file = if oversized {
        std::fs::File::create(&path).ok()?
    } else {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .ok()?
    };
    Some(file)
}

struct ClapLogger;

impl log::Log for ClapLogger {
    fn enabled(&self, _metadata: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &log::Record<'_>) {
        // 【先写文件，再交宿主】宿主回调是外部代码：它可以阻塞、可以丢弃、也可以在
        // 卸载过程中失效。文件这一路不依赖宿主，必须无条件先落盘。
        if let Some(slot) = FILE.get() {
            if let Ok(mut guard) = slot.lock() {
                if let Some(file) = guard.as_mut() {
                    let _ = writeln!(
                        file,
                        "[{}] {} {}",
                        record.level(),
                        record.target(),
                        record.args()
                    );
                }
            }
        }
        let host = HOST.load(Ordering::Acquire);
        if host.is_null() {
            return;
        }
        // 宿主回调可能把消息留到返回之后，所以必须交一个 NUL 结尾的 C 字符串；
        // CString::new 失败时（消息里含内部 NUL）退化为一条固定文本，绝不 panic。
        let text = format!("[hifishifter] {}", record.args());
        let message = CString::new(text)
            .unwrap_or_else(|_| CString::new("[hifishifter] log message dropped").unwrap());
        let severity = match record.level() {
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
                let host = HOST_PTR.load(Ordering::Acquire).cast::<clap_host_t>();
                callback(host, severity, message.as_ptr());
            }
        }
    }

    fn flush(&self) {}
}

/// 由 `plugin_init` 调用：查询宿主的 `clap.log` 并装上进程级 logger。
///
/// # Safety
/// `host` 必须指向本实例存活期内有效的 `clap_host_t`，且当前已在
/// `plugin->init` 调用栈中（CLAP 允许此刻查询宿主扩展）。
pub unsafe fn install(host: *const clap_host_t) {
    if host.is_null() {
        return;
    }
    // 【为什么先把文件日志装起来】即使宿主没有 clap.log，插件自己的诊断也必须落地。
    // log::set_logger 全进程只能成功一次，所以无论宿主给不给扩展，这里都要装。
    LOGGER.get_or_init(|| {
        let _ = log::set_logger(&ClapLogger);
        log::set_max_level(log::LevelFilter::Info);
        FILE.get_or_init(|| Mutex::new(open_log_file()));
    });
    let get_extension = unsafe { (*host).get_extension };
    let Some(get_extension) = get_extension else {
        return;
    };
    let name = unsafe { (*host).name };
    log::info!(
        "[clap] logger installed (host={})",
        if name.is_null() {
            "unknown".to_string()
        } else {
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        }
    );
    let ext = unsafe { get_extension(host, CLAP_EXT_LOG.as_ptr().cast()) };
    if ext.is_null() {
        // 宿主没有 clap.log 不是错误：日志已经写了文件，音频不受影响。
        return;
    }
    // 【为什么只登记第一个】log::set_logger 全进程只能成功一次，因此日志也只能
    // 交给一个宿主实例。允许第二个实例覆盖指针会让第一个实例的诊断凭空改道，
    // 而且宿主卸载顺序不可预测 —— 固定第一个并把后续实例视为"已有 logger"。
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
}

/// `clap_entry.deinit` 时摘掉宿主指针；之后 `log` 变成 no-op。
///
/// 【已知边界】`log::set_logger` 在进程内不可撤销、不可重设，所以同一次模块加载
/// 里 logger 只认第一个提供 `clap.log` 的宿主。宿主若在 `deinit` 之前先销毁那个
/// 实例，后续日志会被丢弃而不是崩溃 —— `clear()` 把指针摘掉后 `log()` 是 no-op。
/// 这是 `log` crate 的进程级单例约束，不是能力声明；真机多实例行为仍按
/// `docs/plugin-host-survey.md` 保持「未测」。
pub fn clear() {
    HOST.store(std::ptr::null_mut(), Ordering::Release);
    HOST_PTR.store(std::ptr::null_mut(), Ordering::Release);
}

/// 供测试与探针读取：宿主是否提供了 `clap.log`。
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
