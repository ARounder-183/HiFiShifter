//! 全局日志基础设施。
//!
//! 设计：
//! - `log` crate 作为唯一门面，应用代码统一使用 `log::info!` / `log::warn!` /
//!   `log::error!` / `log::debug!`（热路径仍用 `debug_eprintln!`，编译期裁剪）。
//! - [`StderrLogger`] 把带级别的日志行写到 stderr；随后 tee 线程从 stderr 读取，
//!   补充时间戳后写入日志文件并回显到真正的控制台。这样第三方库（ORT、
//!   vslib、cpal 等）直接写 stderr 的输出也会被一并捕获。
//! - 默认把日志写到平台标准日志目录，可用 `--log-file=<path>` 指定路径、
//!   `--log-file=-` 显式关闭。
//! - panic hook 会把 panic 详情写入日志（含 backtrace），保证 `panic = "abort"`
//!   下用户仍能提交带现场信息的日志。
//!
//! 【目录、行格式、会话头、轮转都不在这里实现】它们住在
//! [`hifishifter_kernel::logfile`] —— ARA 插件要产出**同一形状**的日志，两边各写
//! 一套必然分叉（插件那份此前就没有日期、没有版本号、也不轮转）。本模块只保留
//! App 独有的那部分：**stderr 管道 + tee 线程**（为的是顺带捕获第三方库直接写
//! stderr 的输出，插件跑在宿主进程里没有这个需求）。

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use hifishifter_kernel::logfile::RotatingLog;

/// App 的日志文件名。**与插件的 `plugin.log` 不同**，这是有意的：两个形态的日志
/// 现在落在同一个目录里，文件名不同才不会互相覆盖，轮转也各按自己的文件名进行。
const LOG_FILE_NAME: &str = "hifishifter.log";

/// stderr 管道缓冲（字节），避免 ORT 初始化等密集输出在 tee 线程启动前阻塞主线程。
const PIPE_BUFFER_BYTES: usize = 1024 * 1024;

static LOG_FILE_PATH: OnceLock<PathBuf> = OnceLock::new();

/// 当前日志文件路径（未启用文件日志时为 `None`）。
pub fn log_file() -> Option<&'static Path> {
    LOG_FILE_PATH.get().map(|p| p.as_path())
}

/// 当前日志所在目录（未启用文件日志时为 `None`）。
pub fn log_dir() -> Option<&'static Path> {
    LOG_FILE_PATH.get().and_then(|p| p.parent()).map(Path::new)
}

/// 当前日志文件及历史轮转（当前在前；仅返回存在的文件）。
/// 供诊断包导出使用。
pub fn log_files() -> Vec<PathBuf> {
    LOG_FILE_PATH
        .get()
        .map(|current| hifishifter_kernel::logfile::rotated_files(current))
        .unwrap_or_default()
}

/// `--log-file` 参数的解析结果。
pub enum LogFileChoice {
    /// `--log-file=-`：显式关闭文件日志。
    Disabled,
    /// 未传 `--log-file`：写入平台默认日志目录。
    Default,
    /// `--log-file=<path>`：写入指定路径。
    Explicit(PathBuf),
}

/// 从进程启动参数解析 `--log-file <path>` / `--log-file=<path>`。
pub fn choice_from_args(args: &[String]) -> LogFileChoice {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if let Some(path) = arg.strip_prefix("--log-file=") {
            if path.is_empty() || path == "-" {
                return LogFileChoice::Disabled;
            }
            return LogFileChoice::Explicit(std::path::PathBuf::from(path));
        }
        if arg == "--log-file" {
            return match iter.next() {
                Some(path) if !path.is_empty() && path != "-" => {
                    LogFileChoice::Explicit(std::path::PathBuf::from(path))
                }
                _ => LogFileChoice::Disabled,
            };
        }
    }
    LogFileChoice::Default
}

/// 初始化日志系统：安装 stderr logger 与 panic hook，并按 `choice` 启动
/// stderr → 日志文件的 tee 线程。
pub fn init_logging(choice: LogFileChoice) {
    install_stderr_logger();
    install_panic_hook();

    let log_path = match choice {
        // `--log-file=-`：用户显式关闭文件日志，属正常路径，静默返回 ——
        // 不得落入下方“默认目录未解析”的警告分支（那会误导用户以为出错）。
        LogFileChoice::Disabled => return,
        LogFileChoice::Explicit(path) => Some(path),
        LogFileChoice::Default => default_log_file_path(),
    };
    let Some(log_path) = log_path else {
        log::warn!("file logging unavailable (default log dir unresolved)");
        return;
    };

    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = LOG_FILE_PATH.set(log_path.clone());

    if spawn_stderr_tee(log_path.clone()) {
        log::info!(
            "HiFiShifter v{} ({} {}) starting; log file: {}",
            crate::build_info::display_version(),
            std::env::consts::OS,
            std::env::consts::ARCH,
            log_path.display()
        );
    } else {
        log::warn!(
            "failed to open log file, continuing console-only: {}",
            log_path.display()
        );
    }
}

/// 平台默认日志目录（与 Tauri `app_log_dir` 的目录约定一致）。
///
/// 【为什么转发到内核】目录解析是"日志"这件事的一部分，不是 App 的能力。插件要落进
/// **同一个目录**，就必须用同一份实现 —— 否则两个形态的日志会各写各的（这正是此前
/// 的状况：插件多了 `HiFiShifter` 一层子目录）。可用环境变量 `HIFISHIFTER_LOG_DIR`
/// 覆盖，两个宿主同名同义。
fn default_log_dir() -> Option<PathBuf> {
    hifishifter_kernel::logfile::log_dir()
}

fn default_log_file_path() -> Option<PathBuf> {
    Some(default_log_dir()?.join(LOG_FILE_NAME))
}

// ── log 门面的 stderr logger ────────────────────────────────────────

struct StderrLogger;

/// 第三方库经 `log` 门面输出的 info 级日志过于啰嗦（例如 symphonia 的 MP3
/// demuxer 每次解析都会输出 "using xing header for duration"，一次会话可产生
/// 数百条），按 target 前缀把这些库的 Warn 以下日志丢弃；Warn 及以上保留。
const DEMOTED_TARGETS: &[(&str, log::LevelFilter)] = &[("symphonia", log::LevelFilter::Warn)];

fn is_demoted(target: &str, level: log::Level) -> bool {
    DEMOTED_TARGETS.iter().any(|(prefix, min_level)| {
        target.starts_with(prefix) && level_filter_of(level) > *min_level
    })
}

impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        if is_demoted(record.metadata().target(), record.level()) {
            return;
        }
        // 单次 writeln 保证多线程下行不交错；时间戳由 tee 线程统一补充。
        //
        // 【为什么把 target 也写进来】落盘后的最终形状由
        // `hifishifter_kernel::logfile::format_record` 定义，插件产出的是
        // `[ts] [LEVEL] target: message`。这里少写 target，App 与插件的日志就会
        // 在同一目录里长得不一样 —— 而"两份日志不像同一个产品"正是本次要修的问题。
        let mut stderr = std::io::stderr().lock();
        let _ = writeln!(
            stderr,
            "[{:<5}] {}: {}",
            record.level(),
            record.target(),
            record.args()
        );
    }

    fn flush(&self) {
        let _ = std::io::stderr().flush();
    }
}

/// 日志级别：环境变量 `HIFISHIFTER_LOG`（error|warn|info|debug|trace）优先，
/// 缺省 debug 构建为 Debug、release 构建为 Info。
fn max_level_from_env() -> log::LevelFilter {
    if let Ok(value) = std::env::var("HIFISHIFTER_LOG") {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => return log::LevelFilter::Off,
            "error" => return log::LevelFilter::Error,
            "warn" => return log::LevelFilter::Warn,
            "info" => return log::LevelFilter::Info,
            "debug" => return log::LevelFilter::Debug,
            "trace" => return log::LevelFilter::Trace,
            _ => {}
        }
    }
    if cfg!(debug_assertions) {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    }
}

fn install_stderr_logger() {
    let level = max_level_from_env();
    let _ = log::set_boxed_logger(Box::new(StderrLogger));
    log::set_max_level(level);
}

// ── panic hook ──────────────────────────────────────────────────────

fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let thread = std::thread::current();
        let thread_name = thread.name().unwrap_or("<unnamed>");
        let location = info.location().map(|l| l.to_string()).unwrap_or_default();
        let payload = panic_payload_as_str(info.payload());
        let backtrace = std::backtrace::Backtrace::force_capture();
        log::error!("PANIC: thread '{thread_name}' panicked at {location}: {payload}\n{backtrace}");
        // `panic = "abort"` 下 hook 返回后进程立即中止；短暂等待让 tee 线程
        // 把上面的错误行从管道落盘。
        std::thread::sleep(std::time::Duration::from_millis(150));
    }));
}

fn panic_payload_as_str(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

// ── 限流日志 ────────────────────────────────────────────────────────
//
// 由 lib.rs 的 `log_warn_limited!` / `log_error_limited!` 使用：
// 同一调用点在限流窗口内最多输出一条，防止循环 / 回调 / 逐轮询路径上的
// 错误或警告把日志文件刷满、挤占轮转额度。窗口内被抑制的条数会在该调用点
// 的下一条输出之前以 `[throttled]` 汇总行补记。

// ── 限流（实现已搬到内核）────────────────────────────────────────────────────
//
// 内核模块（渲染器 / 音高分析 / 缓存）也要用限流日志，而 `macro_rules!` 是文本作用域的、
// 搬不过去；实现本身又与宿主无关（只用 `log` crate 与一张进程级哈希表）。
// 所以实现落在 `hifishifter-kernel::log_limiter`，这里只转发 ——
// **两边共用同一份限流状态**，同一调用点不会被 app 与内核各算一次窗口。
pub use hifishifter_kernel::log_limiter::emit_limited;

/// 把 `log::Level` 映射成对应的 `LevelFilter`。
///
/// 【为什么留在这里】它服务于 `max_level_from_env` / 日志级别判定，
/// 与限流实现无关（后者已在内核）。
fn level_filter_of(level: log::Level) -> log::LevelFilter {
    match level {
        log::Level::Error => log::LevelFilter::Error,
        log::Level::Warn => log::LevelFilter::Warn,
        log::Level::Info => log::LevelFilter::Info,
        log::Level::Debug => log::LevelFilter::Debug,
        log::Level::Trace => log::LevelFilter::Trace,
    }
}

// ── stderr → 日志文件 tee ───────────────────────────────────────────

/// 把 stderr 重定向进管道并启动 tee 线程；返回 tee 是否成功启动。
#[cfg(windows)]
fn spawn_stderr_tee(log_path: PathBuf) -> bool {
    use std::os::windows::io::FromRawHandle;

    // Save the original stderr so the tee thread can still echo to console.
    let saved = unsafe { libc::dup(2) };
    if saved < 0 {
        return false;
    }

    let mut fds = [0i32; 2];
    if unsafe { libc::pipe(fds.as_mut_ptr(), PIPE_BUFFER_BYTES as _, libc::O_BINARY) } != 0 {
        unsafe { libc::close(saved) };
        return false;
    }

    // Replace fd 2 with the pipe's write end.
    unsafe {
        libc::dup2(fds[1], 2);
        libc::close(fds[1]);
    }

    let read_handle = unsafe { libc::get_osfhandle(fds[0]) };
    let console_handle = unsafe { libc::get_osfhandle(saved) };
    if read_handle == -1 || console_handle == -1 {
        return false;
    }

    let reader = unsafe { std::fs::File::from_raw_handle(read_handle as *mut _) };
    let console = unsafe { std::fs::File::from_raw_handle(console_handle as *mut _) };
    std::thread::Builder::new()
        .name("stderr-log-tee".to_string())
        .spawn(move || tee_loop(BufReader::new(reader), console, log_path))
        .is_ok()
}

/// 把 stderr 重定向进管道并启动 tee 线程；返回 tee 是否成功启动。
#[cfg(unix)]
fn spawn_stderr_tee(log_path: PathBuf) -> bool {
    use std::os::fd::FromRawFd;

    // Save the original stderr so the tee thread can still echo to console.
    let saved = unsafe { libc::dup(2) };
    if saved < 0 {
        return false;
    }

    let mut fds = [0i32; 2];
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        unsafe { libc::close(saved) };
        return false;
    }

    unsafe {
        libc::dup2(fds[1], 2);
        libc::close(fds[1]);
    }

    let reader = unsafe { std::fs::File::from_raw_fd(fds[0]) };
    let console = unsafe { std::fs::File::from_raw_fd(saved) };
    std::thread::Builder::new()
        .name("stderr-log-tee".to_string())
        .spawn(move || tee_loop(BufReader::new(reader), console, log_path))
        .is_ok()
}

/// 逐行读取 stderr 管道，补时间戳后写入日志文件并回显控制台。
///
/// 【为什么时间戳在**这里**补，而不是在 logger 里】tee 捕获的不只是 `log` 门面的
/// 输出，还有第三方库（ORT / vslib / cpal）直接写 stderr 的行 —— 那些行没有级别、
/// 没有 target，唯一的共同处理就是补时间戳。格式与轮转都来自内核共享实现。
fn tee_loop(mut reader: BufReader<std::fs::File>, mut console: std::fs::File, log_path: PathBuf) {
    let mut writer = match RotatingLog::open(log_path, "HiFiShifter", crate::build_info::version())
    {
        Ok(w) => w,
        Err(_) => return,
    };

    let mut line_buf = String::new();
    loop {
        line_buf.clear();
        match reader.read_line(&mut line_buf) {
            Ok(0) => break,
            Ok(_) => {
                let body = line_buf.trim_end_matches(['\n', '\r']);
                let stamped = hifishifter_kernel::logfile::stamp(body);
                let _ = writer.write_line(&stamped);
                let _ = console.write_all(stamped.as_bytes());
                let _ = console.write_all(b"\n");
                let _ = console.flush();
            }
            Err(_) => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn third_party_info_is_demoted() {
        // symphonia 的 info/debug 被丢弃
        assert!(is_demoted(
            "symphonia_bundle_mp3::demuxer",
            log::Level::Info
        ));
        assert!(is_demoted("symphonia_core::io", log::Level::Debug));
        // symphonia 的 warn/error 保留
        assert!(!is_demoted(
            "symphonia_bundle_mp3::demuxer",
            log::Level::Warn
        ));
        assert!(!is_demoted(
            "symphonia_bundle_mp3::demuxer",
            log::Level::Error
        ));
        // 其他 target 不受影响
        assert!(!is_demoted("audio_engine", log::Level::Info));
        assert!(!is_demoted("symphonicax::noisy", log::Level::Info)); // 前缀不匹配
    }
}
