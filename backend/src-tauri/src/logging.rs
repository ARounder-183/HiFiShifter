//! 全局日志基础设施。
//!
//! 设计：
//! - `log` crate 作为唯一门面，应用代码统一使用 `log::info!` / `log::warn!` /
//!   `log::error!` / `log::debug!`（热路径仍用 `debug_eprintln!`，编译期裁剪）。
//! - [`StderrLogger`] 把带级别的日志行写到 stderr；随后 tee 线程从 stderr 读取，
//!   补充时间戳后写入日志文件并回显到真正的控制台。这样第三方库（ORT、
//!   vslib、cpal 等）直接写 stderr 的输出也会被一并捕获。
//! - 默认把日志写到平台标准日志目录（与 Tauri `app_log_dir` 的约定一致），
//!   可用 `--log-file=<path>` 指定路径、`--log-file=-` 显式关闭。
//! - 文件按大小轮转：`hifishifter.log` 超过上限后挪为 `hifishifter.1.log`，
//!   最多保留 [`MAX_ROTATED_LOGS`] 份历史。
//! - panic hook 会把 panic 详情写入日志（含 backtrace），保证 `panic = "abort"`
//!   下用户仍能提交带现场信息的日志。

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// 与 tauri.conf.json 的 `identifier` 保持一致（用于推导默认日志目录）。
const APP_IDENTIFIER: &str = "com.arounder.hifishifter";

const LOG_FILE_NAME: &str = "hifishifter.log";
/// 单个日志文件大小上限，超过后轮转。
const MAX_LOG_FILE_BYTES: u64 = 8 * 1024 * 1024;
/// 保留的历史日志份数（hifishifter.1.log 起）。
const MAX_ROTATED_LOGS: usize = 3;
/// stderr 管道缓冲（字节），避免 ORT 初始化等密集输出在 tee 线程启动前阻塞主线程。
const PIPE_BUFFER_BYTES: usize = 1024 * 1024;
/// 每写入多少行检查一次轮转。
const ROTATION_CHECK_INTERVAL: u64 = 256;

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
    let Some(current) = LOG_FILE_PATH.get() else {
        return Vec::new();
    };
    let mut files = vec![current.clone()];
    for i in 1..=MAX_ROTATED_LOGS {
        let rotated = rotated_path(current, i);
        if rotated.exists() {
            files.push(rotated);
        }
    }
    files
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
        log::warn!("failed to open log file, continuing console-only: {}", log_path.display());
    }
}

/// 平台默认日志目录（与 Tauri `app_log_dir` 的目录约定一致）。
/// 可用环境变量 `HIFISHIFTER_LOG_DIR` 覆盖。
fn default_log_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("HIFISHIFTER_LOG_DIR") {
        let dir = dir.trim();
        if !dir.is_empty() {
            return Some(PathBuf::from(dir));
        }
    }
    #[cfg(windows)]
    {
        let base = std::env::var_os("LOCALAPPDATA")?;
        Some(PathBuf::from(base).join(APP_IDENTIFIER).join("logs"))
    }
    #[cfg(target_os = "macos")]
    {
        let home = std::env::var_os("HOME")?;
        Some(PathBuf::from(home).join("Library").join("Logs").join(APP_IDENTIFIER))
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let base = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))?;
        Some(base.join(APP_IDENTIFIER).join("logs"))
    }
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
    DEMOTED_TARGETS
        .iter()
        .any(|(prefix, min_level)| target.starts_with(prefix) && level_filter_of(level) > *min_level)
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
        let mut stderr = std::io::stderr().lock();
        let _ = writeln!(stderr, "[{:<5}] {}", record.level(), record.args());
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
        log::error!(
            "PANIC: thread '{thread_name}' panicked at {location}: {payload}\n{backtrace}"
        );
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

/// 限流窗口：同一调用点在该窗口内最多输出一条。
const RATE_LIMIT_WINDOW: Duration = Duration::from_secs(10);

struct RateLimitState {
    last_emit: Option<Instant>,
    suppressed: u64,
}

enum EmitDecision {
    Emit { previously_suppressed: u64 },
    Suppress,
}

type RateLimitMap = HashMap<(&'static str, u32), RateLimitState>;

static RATE_LIMITER: OnceLock<Mutex<RateLimitMap>> = OnceLock::new();

fn rate_limiter() -> &'static Mutex<RateLimitMap> {
    RATE_LIMITER.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 限流判定的纯逻辑（便于单元测试）：窗口外首条放行并结算上一窗口的
/// 抑制计数；窗口内后续条一律抑制。
fn limited_decision(state: &mut RateLimitState, now: Instant, window: Duration) -> EmitDecision {
    match state.last_emit {
        Some(last) if now.duration_since(last) < window => {
            state.suppressed += 1;
            EmitDecision::Suppress
        }
        _ => {
            let previously_suppressed = std::mem::take(&mut state.suppressed);
            state.last_emit = Some(now);
            EmitDecision::Emit { previously_suppressed }
        }
    }
}

fn short_source(file: &str) -> &str {
    file.rsplit(['/', '\\']).next().unwrap_or(file)
}

fn level_filter_of(level: log::Level) -> log::LevelFilter {
    match level {
        log::Level::Error => log::LevelFilter::Error,
        log::Level::Warn => log::LevelFilter::Warn,
        log::Level::Info => log::LevelFilter::Info,
        log::Level::Debug => log::LevelFilter::Debug,
        log::Level::Trace => log::LevelFilter::Trace,
    }
}

/// 限流输出入口。级别被过滤时不做任何计数（没有真正被"抑制"的内容）。
pub fn emit_limited(
    level: log::Level,
    file: &'static str,
    line: u32,
    args: std::fmt::Arguments<'_>,
) {
    let filter = level_filter_of(level);
    if filter > log::STATIC_MAX_LEVEL || filter > log::max_level() {
        return;
    }

    let decision = {
        let mut map = rate_limiter().lock().unwrap_or_else(|e| e.into_inner());
        let state = map
            .entry((file, line))
            .or_insert_with(|| RateLimitState { last_emit: None, suppressed: 0 });
        limited_decision(state, Instant::now(), RATE_LIMIT_WINDOW)
    };

    if let EmitDecision::Emit { previously_suppressed } = decision {
        if previously_suppressed > 0 {
            log::log!(
                level,
                "[throttled] {}:{line} — {previously_suppressed} message(s) suppressed",
                short_source(file),
            );
        }
        log::log!(level, "{args}");
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

/// 逐行读取 stderr 管道，加时间戳后写入日志文件并回显控制台。
fn tee_loop(mut reader: BufReader<std::fs::File>, mut console: std::fs::File, log_path: PathBuf) {
    let mut writer = match RotatingLog::open(log_path) {
        Ok(w) => w,
        Err(_) => return,
    };

    let mut line_buf = String::new();
    loop {
        line_buf.clear();
        match reader.read_line(&mut line_buf) {
            Ok(0) => break,
            Ok(_) => {
                let ts = chrono::Local::now().format("%H:%M:%S%.3f");
                let body = line_buf.trim_end_matches(['\n', '\r']);
                let stamped = format!("[{ts}] {body}\n");
                let _ = writer.write_line(&stamped);
                let _ = console.write_all(stamped.as_bytes());
                let _ = console.flush();
            }
            Err(_) => break,
        }
    }
}

/// 追加写入的日志文件，带按大小轮转。
struct RotatingLog {
    path: PathBuf,
    file: std::fs::File,
    bytes_written: u64,
    lines_since_check: u64,
}

impl RotatingLog {
    fn open(path: PathBuf) -> std::io::Result<Self> {
        let file = Self::open_with_rotation(&path)?;
        // 追加模式下文件可能已接近上限（上次会话遗留）：按真实长度初始化
        // 计数，否则本轮要再写满一个 MAX 才会触发轮转，文件可超出上限近一倍。
        let bytes_written = file.metadata().map(|m| m.len()).unwrap_or(0);
        Ok(Self { path, file, bytes_written, lines_since_check: 0 })
    }

    /// 打开（必要时先轮转）日志文件并写入会话头。
    fn open_with_rotation(path: &Path) -> std::io::Result<std::fs::File> {
        if std::fs::metadata(path).map(|m| m.len() >= MAX_LOG_FILE_BYTES).unwrap_or(false) {
            Self::shift_files(path);
        }
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        let _ = writeln!(
            file,
            "==== HiFiShifter v{} ({} {}) log started at {} ====",
            crate::build_info::display_version(),
            std::env::consts::OS,
            std::env::consts::ARCH,
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f")
        );
        let _ = file.flush();
        Ok(file)
    }

    /// `hifishifter.log → hifishifter.1.log → … → hifishifter.{MAX}.log`，
    /// 超出份数的历史删除。
    fn shift_files(path: &Path) {
        for i in (1..MAX_ROTATED_LOGS).rev() {
            let from = rotated_path(path, i);
            if !from.exists() {
                continue;
            }
            let to = rotated_path(path, i + 1);
            let _ = std::fs::remove_file(&to);
            let _ = std::fs::rename(&from, &to);
        }
        if path.exists() {
            let to = rotated_path(path, 1);
            let _ = std::fs::remove_file(&to);
            let _ = std::fs::rename(path, &to);
        }
    }

    fn write_line(&mut self, line: &str) -> std::io::Result<()> {
        self.file.write_all(line.as_bytes())?;
        self.file.flush()?;
        self.bytes_written += line.len() as u64;
        self.lines_since_check += 1;
        if self.lines_since_check >= ROTATION_CHECK_INTERVAL
            && self.bytes_written >= MAX_LOG_FILE_BYTES
        {
            self.lines_since_check = 0;
            self.file = Self::open_with_rotation(&self.path)?;
            // 轮转后是新文件（仅含会话头），同样按真实长度重置计数。
            self.bytes_written = self.file.metadata().map(|m| m.len()).unwrap_or(0);
        }
        Ok(())
    }
}

fn rotated_path(path: &Path, index: usize) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(LOG_FILE_NAME);
    let rotated = file_name.split_once('.').map_or_else(
        || format!("{file_name}.{index}.log"),
        |(stem, ext)| format!("{stem}.{index}.{ext}"),
    );
    path.with_file_name(rotated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limited_decision_first_call_emits() {
        let mut state = RateLimitState { last_emit: None, suppressed: 0 };
        let now = Instant::now();
        match limited_decision(&mut state, now, RATE_LIMIT_WINDOW) {
            EmitDecision::Emit { previously_suppressed } => {
                assert_eq!(previously_suppressed, 0);
            }
            EmitDecision::Suppress => panic!("first call must emit"),
        }
    }

    #[test]
    fn limited_decision_suppresses_within_window() {
        let mut state = RateLimitState { last_emit: None, suppressed: 0 };
        let now = Instant::now();
        assert!(matches!(
            limited_decision(&mut state, now, RATE_LIMIT_WINDOW),
            EmitDecision::Emit { .. }
        ));
        for i in 1..=5u64 {
            let at = now + Duration::from_millis(100 * i);
            assert!(matches!(
                limited_decision(&mut state, at, RATE_LIMIT_WINDOW),
                EmitDecision::Suppress
            ));
        }
        assert_eq!(state.suppressed, 5);
    }

    #[test]
    fn limited_decision_flushes_suppressed_count_after_window() {
        let mut state = RateLimitState { last_emit: None, suppressed: 0 };
        let now = Instant::now();
        assert!(matches!(
            limited_decision(&mut state, now, RATE_LIMIT_WINDOW),
            EmitDecision::Emit { .. }
        ));
        for i in 1..=3u64 {
            let at = now + Duration::from_millis(100 * i);
            assert!(matches!(
                limited_decision(&mut state, at, RATE_LIMIT_WINDOW),
                EmitDecision::Suppress
            ));
        }
        // 窗口过期后的下一条：放行，并携带上一窗口累计的抑制条数。
        let after_window = now + RATE_LIMIT_WINDOW + Duration::from_secs(1);
        match limited_decision(&mut state, after_window, RATE_LIMIT_WINDOW) {
            EmitDecision::Emit { previously_suppressed } => {
                assert_eq!(previously_suppressed, 3);
            }
            EmitDecision::Suppress => panic!("call after window must emit"),
        }
        assert_eq!(state.suppressed, 0);
    }

    #[test]
    fn limited_decision_re_arms_window_after_flush() {
        let mut state = RateLimitState { last_emit: None, suppressed: 0 };
        let now = Instant::now();
        assert!(matches!(
            limited_decision(&mut state, now, RATE_LIMIT_WINDOW),
            EmitDecision::Emit { .. }
        ));
        // 放行后重新进入窗口期：紧随其后的调用再次被抑制。
        assert!(matches!(
            limited_decision(&mut state, now + Duration::from_millis(1), RATE_LIMIT_WINDOW),
            EmitDecision::Suppress
        ));
    }

    #[test]
    fn short_source_strips_paths() {
        assert_eq!(short_source("src\\commands\\playback.rs"), "playback.rs");
        assert_eq!(short_source("src/renderer/vslib_processor.rs"), "vslib_processor.rs");
        assert_eq!(short_source("plain.rs"), "plain.rs");
    }

    #[test]
    fn third_party_info_is_demoted() {
        // symphonia 的 info/debug 被丢弃
        assert!(is_demoted("symphonia_bundle_mp3::demuxer", log::Level::Info));
        assert!(is_demoted("symphonia_core::io", log::Level::Debug));
        // symphonia 的 warn/error 保留
        assert!(!is_demoted("symphonia_bundle_mp3::demuxer", log::Level::Warn));
        assert!(!is_demoted("symphonia_bundle_mp3::demuxer", log::Level::Error));
        // 其他 target 不受影响
        assert!(!is_demoted("audio_engine", log::Level::Info));
        assert!(!is_demoted("symphonicax::noisy", log::Level::Info)); // 前缀不匹配
    }
}
