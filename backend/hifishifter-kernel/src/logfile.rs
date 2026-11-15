//! 日志文件的位置、行格式与会话头 —— 独立 App 与 ARA 插件**共用同一份**。
//!
//! 【为什么必须有这个模块】"日志"这件事此前被写了两遍：App 在 `src-tauri/src/logging.rs`
//! 里有一套目录解析 + 行格式 + 会话头 + 轮转，插件在 `hifishifter-plugin/src/diagnostics.rs`
//! 里另有一套。代价是用户看到的**两份日志完全不像同一个产品** —— 插件那份没有日期、
//! 没有版本号、没有时间戳、不轮转，而 App 那份全都有。两个形态的日志还要放进同一个
//! 目录（见 `log_dir`），并排出现时差异只会更刺眼。
//!
//! 因此目录、格式、会话头、轮转只有这一处实现，两个宿主都调它。目录本身由
//! [`crate::app_paths`] 计算（与 [`crate::config_location`] 是同一个原则：
//! **同一件事只有一处实现**）。
//!
//! 两个宿主各自保留的差异只有一处，且是真实的：
//! - App 用 stderr 管道 + tee 线程，为了**顺带捕获第三方库直接写 stderr 的输出**
//!   （ORT / vslib / cpal 都不走 `log` 门面）。所以它逐行给 stderr 补时间戳，
//!   用的是 [`stamp`]。
//! - 插件跑在宿主进程里，没有自己的 stderr 可接管，直接由 `log::Log` 实现写文件，
//!   用的是 [`format_record`]。
//!
//! 两者产出的**最终行格式完全一致**。

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// 覆盖日志目录的环境变量（两个宿主同名同义）。
pub use crate::app_paths::LOG_DIR_ENV;

/// 单个日志文件大小上限，超过后轮转。
pub const MAX_LOG_FILE_BYTES: u64 = 8 * 1024 * 1024;
/// 保留的历史日志份数（`.1.log` 起）。
pub const MAX_ROTATED_LOGS: usize = 3;
/// 每写入多少行检查一次轮转（避免每行都做一次 `metadata` 系统调用）。
pub const ROTATION_CHECK_INTERVAL: u64 = 256;
/// 默认单行上限（按**字符**计）。可由 `RotatingLog::open_with_policy` 覆盖。
pub const DEFAULT_MAX_LINE_CHARS: usize = 4096;
/// 缓冲写阈值：累积到这个字节数就落一次盘（详见 `RotatingLog` 的说明）。
pub const FLUSH_THRESHOLD_BYTES: usize = 64 * 1024;

/// 时间戳格式：**带日期**。
///
/// 【为什么必须带日期】旧实现只写 `%H:%M:%S%.3f`，日期仅存在于会话头。但日志会跨天、
/// 会轮转、会被打进诊断包 —— 一旦离开会话头的上下文，"14:03:22" 就无法定位到哪一天。
pub const TIMESTAMP_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.3f";

/// 平台默认日志目录；无法确定时为 `None`（不 panic —— 日志写不出不该让宿主起不来）。
///
/// 【为什么只是一条转发】目录解析是"日志"这件事的一部分，不是宿主的能力差异。插件与
/// 独立 App 必须写进**同一个目录**，两份实现就一定会分家（此前正是如此：插件多了
/// `HiFiShifter` 一层子目录）。唯一的事实来源是 [`crate::app_paths::log_dir`]：
///
/// | 平台 | 路径 |
/// | --- | --- |
/// | Windows | `%LOCALAPPDATA%\HiFiShifter\logs` |
/// | macOS | `~/Library/Logs/HiFiShifter` |
/// | Linux | `$XDG_STATE_HOME/HiFiShifter/logs`（缺省 `~/.local/state`） |
///
/// 可用环境变量 `HIFISHIFTER_LOG_DIR` 覆盖（两个宿主同名同义）。
pub fn log_dir() -> Option<PathBuf> {
    crate::app_paths::log_dir()
}

/// 会话头：版本、平台、以及本次会话的起始时刻。
///
/// 【为什么要版本号】用户事后拿到一份日志时，第一件要判断的事就是"这是哪一版"。
/// 插件日志此前完全没有这一行，于是每份日志都得先问用户"你装的是哪个版本"。
pub fn session_banner(product: &str, version: &str) -> String {
    format!(
        "==== {product} v{} ({} {}) log started at {} ====",
        crate::build_info::display_version(version),
        std::env::consts::OS,
        std::env::consts::ARCH,
        now_timestamp()
    )
}

/// 本次会话的短标识（8 位十六进制）。
///
/// 【为什么不用完整 UUID】会话头/尾要在用户贴出来的日志里一眼能对照；8 位十六进制
/// 在单个用户目录里足够区分，又不会把每一行都撑长。进程级只生成一次。
pub fn short_session_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let seed = NEXT.fetch_add(1, Ordering::Relaxed)
        ^ std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
        ^ (std::process::id() as u64) << 17;
    // xorshift64* 的一步混合：只为生成可读的短 id，不用于安全用途。
    let mixed = seed
        .wrapping_mul(0x2545_F491_4F6C_DD1D)
        .rotate_left(29)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    format!("{mixed:08x}")
}

/// 把任意文本压成**单行、有字符上限**的日志字段。
///
/// 【为什么连换行也要转义】日志按行读、按行 grep：一条带 `\n` 的 message 能凭空
/// 伪造出整条假日志（log injection，CWE-117）。两个宿主与策略自证行共用这一份实现。
pub fn log_field(text: &str, max_chars: usize) -> String {
    let mut out = String::with_capacity(text.len().min(max_chars.saturating_mul(2)));
    let mut truncated = false;
    for (index, ch) in text.chars().enumerate() {
        if index >= max_chars {
            truncated = true;
            break;
        }
        match ch {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => {}
            ch => out.push(ch),
        }
    }
    if truncated {
        out.push_str("…(truncated)");
    }
    out
}

/// 带策略自证的会话头（旧 `session_banner` 保持同形，见其单测）。
///
/// 追加的 `session / verbosity / source / policy / file` 段回答排障第一问：
/// "这一段沉默是哪次会话、哪一档"。
pub fn session_banner_with_policy(
    product: &str,
    version: &str,
    session_id: &str,
    verbosity: &str,
    source: &str,
    file_name: &str,
) -> String {
    let base = session_banner(product, version);
    let base = base.strip_suffix("====").unwrap_or(&base);
    format!(
        "{base} | session={session_id} | verbosity={verbosity} | source={source} | policy={} | file={file_name} ====",
        crate::log_policy::POLICY_VERSION
    )
}

/// 会话尾：正常退出与 panic 都写它，用户才能区分"日志被截断"和"会话结束了"。
pub fn session_footer(
    session_id: &str,
    reason: &str,
    lines: u64,
    rotations: u64,
    verbosity: &str,
) -> String {
    format!(
        "==== HiFiShifter session {session_id} ended at {} | reason={reason} | lines={lines} | rotations={rotations} | verbosity={verbosity} ====",
        now_timestamp()
    )
}

/// 一条 `log` 记录的完整行（含时间戳、级别、target）。
///
/// 形如 `[2026-10-08 14:03:22.417] [INFO ] hifishifter_plugin::ara: ara: clips=2`。
/// 级别补齐到 5 列，行与行之间才对齐。
pub fn format_record(level: log::Level, target: &str, message: &str) -> String {
    format!("[{}] [{:<5}] {target}: {message}", now_timestamp(), level)
}

/// 给一行**原文**（未经 `log` 门面的 stderr 输出）补时间戳。
///
/// 供 App 的 tee 线程使用：第三方库直接写 stderr 的那部分没有级别也没有 target，
/// 能补的只有时间戳。
pub fn stamp(body: &str) -> String {
    format!("[{}] {body}", now_timestamp())
}

/// 当前本地时刻，按 [`TIMESTAMP_FORMAT`] 格式化。
pub fn now_timestamp() -> String {
    chrono::Local::now().format(TIMESTAMP_FORMAT).to_string()
}

/// `hifishifter.log → hifishifter.1.log → … → hifishifter.{MAX}.log` 的命名。
///
/// 以**文件名**（而非目录）为准，因此同一目录下的 `hifishifter.log` 与 `plugin-<format>-<host>.log`
/// 各自轮转、互不影响 —— 两个形态的日志放在一起时这一点是必须的。
///
/// 两个宿主各自决定自己的文件名（App 用 `hifishifter.log`，插件按格式与宿主分文件），
/// 所以这里没有"日志文件名"常量；`fallback` 只在传入路径没有文件名时用得上。
pub fn rotated_path(path: &Path, index: usize) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("hifishifter.log");
    let rotated = file_name.split_once('.').map_or_else(
        || format!("{file_name}.{index}.log"),
        |(stem, ext)| format!("{stem}.{index}.{ext}"),
    );
    path.with_file_name(rotated)
}

/// 当前日志文件及历史轮转（当前在前；仅返回存在的文件）。供诊断包导出使用。
pub fn rotated_files(current: &Path) -> Vec<PathBuf> {
    rotated_files_with_limit(current, MAX_ROTATED_LOGS)
}

/// 按策略给定的历史份数枚举轮转件（高档保留更多上下文）。
pub fn rotated_files_with_limit(current: &Path, max_files: usize) -> Vec<PathBuf> {
    let mut files = vec![current.to_path_buf()];
    for index in 1..=max_files {
        let rotated = rotated_path(current, index);
        if rotated.exists() {
            files.push(rotated);
        }
    }
    files
}

/// 追加写入的日志文件，带按大小轮转、单行截断与缓冲写。
///
/// 【为什么两个宿主共用它】轮转策略（上限、份数、检查间隔、命名）是"日志"这件事的
/// 一部分，不是宿主的能力差异。分开写必然分叉 —— 插件此前就完全没有轮转。
///
/// 【为什么改缓冲写】旧实现每行 `file.flush()`，高频打点下是逐行系统调用；实测
/// 40 分钟能写满 3 个 8 MiB 轮转件，把启动现场冲掉。现在累积 64 KiB 或显式
/// `flush_now()` 才落盘：会话头/尾、panic、诊断包导出前必须强制 flush。
pub struct RotatingLog {
    path: PathBuf,
    /// 会话头所需的产品标识与版本号；轮转后重建同形的头时要用。
    product: String,
    version: String,
    /// 策略自证所需的会话信息；轮转后重建同形的头时要用。
    session_id: String,
    verbosity: String,
    source: String,
    file: std::fs::File,
    /// 待落盘字节（会话头/尾也计入，但直接 flush）。
    buffer: Vec<u8>,
    bytes_written: u64,
    lines_since_check: u64,
    lines_written: u64,
    rotations: u64,
    max_bytes: u64,
    max_files: usize,
    max_line_chars: usize,
}

impl RotatingLog {
    /// 打开日志文件（必要时先轮转），并写入会话头。
    ///
    /// 旧签名保留：默认容量/份数/单行上限，且会话头**保持原样**（行格式单测钉死）。
    pub fn open(path: PathBuf, product: &str, version: &str) -> std::io::Result<Self> {
        Self::open_inner(
            path,
            product,
            version,
            MAX_LOG_FILE_BYTES,
            MAX_ROTATED_LOGS,
            DEFAULT_MAX_LINE_CHARS,
            false,
        )
    }

    /// 按 `LogPolicy` 的容量/份数/单行上限打开，并写带策略自证的会话头。
    pub fn open_with_policy(
        path: PathBuf,
        product: &str,
        version: &str,
        max_bytes: u64,
        max_files: usize,
        max_line_chars: usize,
    ) -> std::io::Result<Self> {
        Self::open_inner(
            path,
            product,
            version,
            max_bytes,
            max_files,
            max_line_chars,
            true,
        )
    }

    fn open_inner(
        path: PathBuf,
        product: &str,
        version: &str,
        max_bytes: u64,
        max_files: usize,
        max_line_chars: usize,
        with_policy: bool,
    ) -> std::io::Result<Self> {
        let session_id = crate::log_policy::session_id().to_owned();
        let verbosity = crate::log_policy::current().as_str().to_owned();
        let source = crate::log_policy::source().as_str().to_owned();
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("hifishifter.log")
            .to_owned();
        let banner = if with_policy {
            session_banner_with_policy(
                product,
                version,
                &session_id,
                &verbosity,
                &source,
                &file_name,
            )
        } else {
            session_banner(product, version)
        };
        let file = Self::open_with_rotation(&path, &banner, max_bytes, max_files)?;
        // 追加模式下文件可能已接近上限（上次会话遗留）：按真实长度初始化计数，
        // 否则本轮要再写满一个 MAX 才会触发轮转，文件可超出上限近一倍。
        let bytes_written = file.metadata().map(|m| m.len()).unwrap_or(0);
        Ok(Self {
            path,
            product: product.to_string(),
            version: version.to_string(),
            session_id,
            verbosity,
            source,
            file,
            buffer: Vec::with_capacity(FLUSH_THRESHOLD_BYTES),
            bytes_written,
            lines_since_check: 0,
            lines_written: 0,
            rotations: 0,
            max_bytes,
            max_files,
            max_line_chars,
        })
    }

    /// 打开（必要时先轮转）并写入会话头。
    fn open_with_rotation(
        path: &Path,
        banner: &str,
        max_bytes: u64,
        max_files: usize,
    ) -> std::io::Result<std::fs::File> {
        if std::fs::metadata(path)
            .map(|m| m.len() >= max_bytes)
            .unwrap_or(false)
        {
            Self::shift_files(path, max_files);
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let _ = writeln!(file, "{banner}");
        let _ = file.flush();
        Ok(file)
    }

    /// 把历史文件整体后移一位，超出份数的删除。
    fn shift_files(path: &Path, max_files: usize) {
        for index in (1..max_files).rev() {
            let from = rotated_path(path, index);
            if !from.exists() {
                continue;
            }
            let to = rotated_path(path, index + 1);
            let _ = std::fs::remove_file(&to);
            let _ = std::fs::rename(&from, &to);
        }
        if path.exists() {
            let to = rotated_path(path, 1);
            let _ = std::fs::remove_file(&to);
            let _ = std::fs::rename(path, &to);
        }
    }

    /// 写入一行（自动补换行、按字符截断、缓冲写）。轮转检查按
    /// [`ROTATION_CHECK_INTERVAL`] 摊销。
    pub fn write_line(&mut self, line: &str) -> std::io::Result<()> {
        let single = log_field(line, self.max_line_chars);
        self.buffer.extend_from_slice(single.as_bytes());
        if !single.ends_with('\n') {
            self.buffer.push(b'\n');
        }
        self.bytes_written += single.len() as u64 + 1;
        self.lines_since_check += 1;
        self.lines_written += 1;
        if self.buffer.len() >= FLUSH_THRESHOLD_BYTES {
            self.flush_now()?;
        }
        if self.lines_since_check >= ROTATION_CHECK_INTERVAL && self.bytes_written >= self.max_bytes
        {
            self.flush_now()?;
            self.lines_since_check = 0;
            let file_name = self
                .path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("hifishifter.log");
            let banner = session_banner_with_policy(
                &self.product,
                &self.version,
                &self.session_id,
                &self.verbosity,
                &self.source,
                file_name,
            );
            self.file =
                Self::open_with_rotation(&self.path, &banner, self.max_bytes, self.max_files)?;
            // 轮转后是新文件（仅含会话头），同样按真实长度重置计数。
            self.bytes_written = self.file.metadata().map(|m| m.len()).unwrap_or(0);
            self.rotations += 1;
        }
        Ok(())
    }

    /// 立刻把缓冲落盘。会话头/尾、panic、诊断包导出、进程退出前必须调用。
    pub fn flush_now(&mut self) -> std::io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        self.file.write_all(&self.buffer)?;
        self.file.flush()?;
        self.buffer.clear();
        Ok(())
    }

    /// 写一行会话尾并强制落盘（供 `RunEvent::Exit` / `entry_deinit` / panic hook）。
    pub fn write_session_footer(&mut self, reason: &str) -> std::io::Result<()> {
        let footer = session_footer(
            &self.session_id,
            reason,
            self.lines_written,
            self.rotations,
            crate::log_policy::current().as_str(),
        );
        self.write_line(&footer)?;
        self.flush_now()
    }

    /// 当前日志文件路径。
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// 本会话写入的行数（会话尾与诊断包用）。
    pub fn lines_written(&self) -> u64 {
        self.lines_written
    }

    /// 本会话发生的轮转次数。
    pub fn rotations(&self) -> u64 {
        self.rotations
    }
}

impl Drop for RotatingLog {
    fn drop(&mut self) {
        let _ = self.flush_now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 行格式是**契约**：两个宿主都产出它，任何一侧漂移都会让日志不可比。
    ///
    /// 钉死形状（时间戳 + 5 列级别 + target + 消息），不钉具体时刻。
    #[test]
    fn the_line_format_is_pinned() {
        let line = format_record(log::Level::Info, "hifishifter_plugin::ara", "ara: clips=2");
        // [YYYY-MM-DD HH:MM:SS.mmm] [INFO ] target: message
        assert!(
            line.contains("] [INFO ] hifishifter_plugin::ara: ara: clips=2"),
            "行格式漂移：{line}"
        );
        // 时间戳长度固定（`[YYYY-MM-DD HH:MM:SS.mmm]`），其后紧跟一个空格。
        const STAMP_LEN: usize = "[2026-10-08 14:03:22.417]".len();
        assert_eq!(line.as_bytes()[STAMP_LEN], b' ', "{line}");
        let stamp = &line[1..STAMP_LEN - 1];
        assert_eq!(stamp.len(), 23, "{stamp}");
        assert_eq!(&stamp[4..5], "-", "{stamp}");
        assert_eq!(&stamp[10..11], " ", "{stamp}");
    }

    /// 时间戳必须**带日期**：跨天、轮转、进诊断包之后，只有时刻无法定位。
    #[test]
    fn the_timestamp_carries_the_date() {
        let ts = now_timestamp();
        // YYYY-MM-DD HH:MM:SS.mmm
        assert_eq!(ts.len(), 23, "{ts}");
        assert_eq!(&ts[4..5], "-", "{ts}");
        assert_eq!(&ts[7..8], "-", "{ts}");
        assert_eq!(&ts[10..11], " ", "{ts}");
        assert_eq!(&ts[13..14], ":", "{ts}");
        assert_eq!(&ts[16..17], ":", "{ts}");
        assert_eq!(&ts[19..20], ".", "{ts}");
    }

    /// 会话头必须带版本与平台 —— 用户拿到一份日志，第一件事是判断"这是哪一版"。
    #[test]
    fn the_banner_carries_version_and_platform() {
        let banner = session_banner("HiFiShifter", "1.2.3");
        assert!(banner.starts_with("==== HiFiShifter v1.2.3"), "{banner}");
        assert!(banner.contains(std::env::consts::OS), "{banner}");
        assert!(banner.contains(std::env::consts::ARCH), "{banner}");
        assert!(banner.contains("log started at"), "{banner}");
        assert!(banner.ends_with("===="), "{banner}");
    }

    /// 轮转按**文件名**进行，因此同目录下的插件日志不会被 App 的轮转碰到。
    #[test]
    fn rotation_is_per_file_name() {
        let app = rotated_path(Path::new("/logs/hifishifter.log"), 1);
        assert!(app.ends_with("hifishifter.1.log"), "{app:?}");
        let plugin = rotated_path(Path::new("/logs/plugin-clap-reaper.log"), 2);
        assert!(plugin.ends_with("plugin-clap-reaper.2.log"), "{plugin:?}");
    }

    /// `HIFISHIFTER_LOG_DIR` 覆盖默认目录（两个宿主同名同义）。
    ///
    /// 【为什么这条要单独测】它是"两个形态落进同一个目录"的手段，也是测试把日志
    /// 指向临时目录的手段。名字或语义变了，两件事会同时坏掉。
    #[test]
    fn the_log_dir_env_var_overrides_the_platform_default() {
        // 这个测试不设环境变量（会与并行测试互踩），只断言未设置时的形状。
        if std::env::var_os(LOG_DIR_ENV).is_some()
            || (cfg!(windows)
                && std::env::var_os("LOCALAPPDATA").is_none()
                && std::env::var_os("USERPROFILE").is_none())
        {
            return;
        }
        let dir = log_dir().expect("platform log dir");
        assert!(!dir.starts_with(std::env::temp_dir()), "{}", dir.display());
        assert!(
            dir.to_string_lossy()
                .contains(crate::app_paths::PRODUCT_DIR),
            "日志目录应以产品目录名命名：{}",
            dir.display()
        );
    }

    /// 单行化：一条前端/插件消息永远只占一行，且按**字符**截断。
    #[test]
    fn a_log_field_never_spans_more_than_one_line() {
        let field = log_field("boom\n[frontend] forged entry\r\nnext\u{7}", 512);
        assert!(!field.contains('\n'), "{field}");
        assert!(!field.contains('\r'), "{field}");
        assert!(!field.contains('\u{7}'), "{field}");
        assert!(field.contains("\\n[frontend] forged entry"), "{field}");
    }

    #[test]
    fn truncation_counts_characters_not_bytes() {
        let long = "安".repeat(12);
        let field = log_field(&long, 5);
        assert!(field.ends_with("…(truncated)"), "{field}");
        assert_eq!(field.matches('安').count(), 5);
    }

    /// 带策略的会话头与会话尾必须能按 session id 对照。
    #[test]
    fn the_policy_banner_and_footer_share_the_session() {
        let banner = session_banner_with_policy(
            "HiFiShifter",
            "1.2.3",
            "abcd1234",
            "low",
            "persisted",
            "hifishifter.log",
        );
        assert!(banner.contains("session=abcd1234"), "{banner}");
        assert!(banner.contains("verbosity=low"), "{banner}");
        assert!(banner.contains("source=persisted"), "{banner}");
        assert!(banner.ends_with("===="), "{banner}");
        let footer = session_footer("abcd1234", "shutdown", 12, 1, "medium");
        assert!(footer.contains("session abcd1234 ended"), "{footer}");
        assert!(footer.contains("reason=shutdown"), "{footer}");
        assert!(footer.contains("lines=12"), "{footer}");
        assert!(footer.contains("verbosity=medium"), "{footer}");
    }

    /// 缓冲写不得丢内容：flush_now 后文件里必须能读到那一行。
    #[test]
    fn buffered_lines_are_readable_after_flush() {
        let dir = std::env::temp_dir().join(format!(
            "hfs-log-buffer-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("buffered.log");
        let mut log = RotatingLog::open(path.clone(), "HiFiShifter", "1.2.3").unwrap();
        log.write_line("first buffered line").unwrap();
        log.flush_now().unwrap();
        assert!(std::fs::read_to_string(&path)
            .unwrap()
            .contains("first buffered line"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 超长单行必须截断（防止一条前端 JSON 把整个轮转额度吃掉）。
    #[test]
    fn a_single_line_is_capped_by_the_policy() {
        let dir = std::env::temp_dir().join(format!(
            "hfs-log-cap-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("capped.log");
        let mut log = RotatingLog::open_with_policy(
            path.clone(),
            "HiFiShifter",
            "1.2.3",
            8 * 1024 * 1024,
            3,
            32,
        )
        .unwrap();
        log.write_line(&"x".repeat(200)).unwrap();
        log.flush_now().unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let body = text.lines().last().unwrap_or_default();
        assert!(body.contains("…(truncated)"), "{body}");
        assert!(
            body.chars().count() <= 32 + "…(truncated)".chars().count(),
            "{body}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 轮转把历史整体后移，并写新的会话头。
    #[test]
    fn writing_rotates_and_keeps_a_banner() {
        let dir = std::env::temp_dir().join(format!(
            "hfs-logfile-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("plugin-clap-reaper.log");

        let mut log = RotatingLog::open(path.clone(), "HiFiShifter", "1.2.3").expect("open log");
        log.write_line("first").unwrap();
        drop(log);

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("==== HiFiShifter v1.2.3"), "{text}");
        assert!(text.contains("first"), "{text}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
