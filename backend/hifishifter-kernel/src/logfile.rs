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

/// 缓冲的**最长存活时间**：到点就落盘，不看字节数。
///
/// 【为什么字节阈值一个人不够】"进程被卡住"不是一条日志，它是一个时刻：此后再没有任何
/// 记录到达，于是 64 KiB 的阈值永远填不满，缓冲里的最后一段随进程一起消失。现场两次
/// `AppHangB1` 的日志尾都是**一行看起来很正常的启动行**，而真正卡住的那一分钟全在
/// 缓冲里 —— 那正是唯一能回答"卡在哪把锁上"的证据。
///
/// 【为什么是 2 秒】一次挂起判定（`HFS_STALL_MS` 默认 2 s）加上 Windows 把窗口画成
/// "未响应"的时间量级。2 秒的代价是空闲期每 2 秒一次 `write` + `flush`，可以忽略。
pub const FLUSH_MAX_AGE: std::time::Duration = std::time::Duration::from_secs(2);

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

/// 剥掉一行**已经带前缀**的日志原文的前缀（`[ts] [LVL ] target: `）。
///
/// 【为什么必须剥】前端把自己的格式化行整行回传，后端再经 `log` 门面写一次 ——
/// 同一行于是有两个时间戳、两个级别、两个 target。现场 `hifishifter.log` 的 13 586
/// 行里有 1 841 行如此（13.5%），其中 1 448 行落在
/// `backend_lib::commands::diagnostics` 这个**假 target** 下。双前缀不只是难看：
/// 按行 grep 出来的次数与真实发生次数不再一致，按 target 统计也会把前端行算进后端。
///
/// 【为什么是"剥前缀"而不是"重新格式化"】我们只能确定**自己写出来的那种前缀**
/// （`format_record` 的形状）该剥；其余一律原样保留，宁可多留一个方括号，也不猜。
pub fn strip_prefix(line: &str) -> &str {
    let mut rest = line.trim_start();
    let Some(after_open) = rest.strip_prefix('[') else {
        return rest;
    };
    let Some((stamp, tail)) = after_open.split_once(']') else {
        return rest;
    };
    if !looks_like_timestamp(stamp) {
        return rest;
    }
    rest = tail.trim_start();
    if let Some(after_level_open) = rest.strip_prefix('[') {
        if let Some((level, tail)) = after_level_open.split_once(']') {
            if is_level_tag(level.trim()) {
                rest = tail.trim_start();
                // 【必须按 ": " 切】target 自己带 `::`（`crate::module`），
                // 按第一个 `:` 切会把 `hifishifter_kernel` 当成 target，
                // 剩下的 `::vocoder…` 就留在"消息"里了。target 不含空格，
                // 因此第一个 ": " 一定落在 target 与消息之间。
                if let Some((target, tail)) = rest.split_once(": ") {
                    if is_target_tag(target) {
                        rest = tail.trim_start();
                    }
                }
            }
        }
    }
    rest
}

/// `YYYY-MM-DD HH:MM:SS.mmm`（`TIMESTAMP_FORMAT`）—— 只认形状，不解析时间。
fn looks_like_timestamp(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() < 19 {
        return false;
    }
    let digits = |range: std::ops::Range<usize>| bytes[range].iter().all(u8::is_ascii_digit);
    digits(0..4)
        && bytes[4] == b'-'
        && digits(5..7)
        && bytes[7] == b'-'
        && digits(8..10)
        && bytes[10] == b' '
        && digits(11..13)
        && bytes[13] == b':'
        && digits(14..16)
        && bytes[16] == b':'
        && digits(17..19)
}

fn is_level_tag(text: &str) -> bool {
    matches!(text, "TRACE" | "DEBUG" | "INFO" | "WARN" | "ERROR")
}

/// target 是 `crate::module` 形状：不含空白、只由标识符字符与 `::` 组成。
fn is_target_tag(text: &str) -> bool {
    !text.is_empty()
        && !text.contains(char::is_whitespace)
        && text
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | ':' | '-' | '.'))
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
    /// 上一次落盘的时刻。缓冲的存活时间从它算起（见 [`FLUSH_MAX_AGE`]）。
    last_flush: std::time::Instant,
    /// 缓冲最长存活多久。发布值是 [`FLUSH_MAX_AGE`]；测试把它缩短。
    flush_after: std::time::Duration,
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
            last_flush: std::time::Instant::now(),
            flush_after: FLUSH_MAX_AGE,
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
        // 【两道闸：字节数与时间】字节数挡住"刷屏把磁盘打满"，时间挡住"刷屏之后突然
        // 什么都不写了" —— 后者正是挂起的形状（见 `FLUSH_MAX_AGE`）。
        if self.buffer.len() >= FLUSH_THRESHOLD_BYTES
            || self.last_flush.elapsed() >= self.flush_after
        {
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
        self.last_flush = std::time::Instant::now();
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

    /// 测试专用：把"缓冲最长存活多久"缩短到毫秒级。
    #[cfg(test)]
    pub fn set_flush_after_for_test(&mut self, after: std::time::Duration) {
        self.flush_after = after;
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

    /// **回归：回放的前端行不许带两个前缀。**
    ///
    /// 现场 `hifishifter.log` 的 13 586 行里有 1 841 行是
    /// `[ts] [ts] [WARN ] target: …`，其中 1 448 行落在
    /// `backend_lib::commands::diagnostics` 这个假 target 下。剥掉既有前缀之后，
    /// 按行 grep 出来的次数才等于真实发生次数。
    #[test]
    fn an_already_formatted_line_loses_its_prefix() {
        let formatted = format_record(
            log::Level::Warn,
            "hifishifter_kernel::vocoder_ort_session",
            "session open failed",
        );
        assert_eq!(strip_prefix(&formatted), "session open failed");
        // 空格对齐的级别（`[WARN ]`）与更宽的（`[ERROR]`）都要认。
        assert_eq!(
            strip_prefix("[2026-11-25 20:07:17.054] [ERROR] backend_lib: boom"),
            "boom"
        );
        // 没有前缀的行原样返回：宁可多留一个方括号，也不猜。
        assert_eq!(strip_prefix("[frontend] boom"), "[frontend] boom");
        assert_eq!(strip_prefix("plain message"), "plain message");
        // 形状不对（不是时间戳）时不许剥：那可能是用户消息本身。
        assert_eq!(
            strip_prefix("[2026-11-25] [WARN ] x: y"),
            "[2026-11-25] [WARN ] x: y"
        );
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

    /// **不变量：安静下来之后，缓冲必须自己落盘。**
    ///
    /// 【为什么这条是"挂起也能留证据"的判据】挂起的形状是"写完最后一行的两秒内再没有
    /// 任何记录"。没有这条闸，那一段就永远在缓冲里，用户的日志尾停在一个看起来完全
    /// 正常的行上 —— 现场两次 `AppHangB1` 都是这样。
    #[test]
    fn a_quiet_log_reaches_disk_within_the_deadline() {
        let dir = std::env::temp_dir().join(format!(
            "hfs-log-age-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("age.log");
        let mut log = RotatingLog::open(path.clone(), "HiFiShifter", "1.2.3").unwrap();
        log.set_flush_after_for_test(std::time::Duration::from_millis(10));
        log.write_line("first line").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        // 这一行"到达"时才发现已经超时：落盘由写入路径触发，不需要定时器。
        log.write_line("second line").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("first line"), "超时之后必须已经落盘：{text}");
        assert!(text.contains("second line"));
        drop(log);
        let _ = std::fs::remove_dir_all(&dir);
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
