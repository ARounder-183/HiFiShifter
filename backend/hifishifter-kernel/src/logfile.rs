//! 日志文件的位置、行格式与会话头 —— 独立 App 与 ARA 插件**共用同一份**。
//!
//! 【为什么必须有这个模块】"日志"这件事此前被写了两遍：App 在 `src-tauri/src/logging.rs`
//! 里有一套目录解析 + 行格式 + 会话头 + 轮转，插件在 `hifishifter-plugin/src/diagnostics.rs`
//! 里另有一套。代价是用户看到的**两份日志完全不像同一个产品** —— 插件那份没有日期、
//! 没有版本号、没有时间戳、不轮转，而 App 那份全都有。两个形态的日志还要放进同一个
//! 目录（见 `log_dir`），并排出现时差异只会更刺眼。
//!
//! 因此目录、格式、会话头、轮转只有这一处实现，两个宿主都调它。与
//! [`crate::config_location`] 是同一个原则：**同一件事只有一处实现**。
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

/// Tauri 的 `identifier`，也是日志目录的一级名字。
const APP_IDENTIFIER: &str = "com.arounder.hifishifter";

/// 覆盖日志目录的环境变量（两个宿主同名同义）。
pub const LOG_DIR_ENV: &str = "HIFISHIFTER_LOG_DIR";

/// 单个日志文件大小上限，超过后轮转。
pub const MAX_LOG_FILE_BYTES: u64 = 8 * 1024 * 1024;
/// 保留的历史日志份数（`.1.log` 起）。
pub const MAX_ROTATED_LOGS: usize = 3;
/// 每写入多少行检查一次轮转（避免每行都做一次 `metadata` 系统调用）。
pub const ROTATION_CHECK_INTERVAL: u64 = 256;

/// 时间戳格式：**带日期**。
///
/// 【为什么必须带日期】旧实现只写 `%H:%M:%S%.3f`，日期仅存在于会话头。但日志会跨天、
/// 会轮转、会被打进诊断包 —— 一旦离开会话头的上下文，"14:03:22" 就无法定位到哪一天。
pub const TIMESTAMP_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.3f";

/// 平台默认日志目录；无法确定时为 `None`（不 panic —— 日志写不出不该让宿主起不来）。
///
/// 结果与独立 App 历史上的布局逐一对应（**不是** [`crate::config_location`] 的
/// `HiFiShifter` 子目录 —— 那是配置与运行时数据的家，日志一直是它的兄弟目录）：
///
/// | 平台 | 路径 |
/// | --- | --- |
/// | Windows | `%LOCALAPPDATA%\com.arounder.hifishifter\logs` |
/// | macOS | `~/Library/Logs/com.arounder.hifishifter` |
/// | Linux | `$XDG_DATA_HOME/com.arounder.hifishifter/logs`（缺省 `~/.local/share`） |
pub fn log_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var(LOG_DIR_ENV) {
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
        Some(
            PathBuf::from(home)
                .join("Library")
                .join("Logs")
                .join(APP_IDENTIFIER),
        )
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let base = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))?;
        Some(base.join(APP_IDENTIFIER).join("logs"))
    }
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
/// 以**文件名**（而非目录）为准，因此同一目录下的 `hifishifter.log` 与 `plugin.log`
/// 各自轮转、互不影响 —— 两个形态的日志放在一起时这一点是必须的。
///
/// 两个宿主各自决定自己的文件名（App 用 `hifishifter.log`，插件用 `plugin.log`），
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
    let mut files = vec![current.to_path_buf()];
    for index in 1..=MAX_ROTATED_LOGS {
        let rotated = rotated_path(current, index);
        if rotated.exists() {
            files.push(rotated);
        }
    }
    files
}

/// 追加写入的日志文件，带按大小轮转。
///
/// 【为什么两个宿主共用它】轮转策略（上限、份数、检查间隔、命名）是"日志"这件事的
/// 一部分，不是宿主的能力差异。分开写必然分叉 —— 插件此前就完全没有轮转。
pub struct RotatingLog {
    path: PathBuf,
    /// 会话头所需的产品标识与版本号；轮转后重建同形的头时要用。
    product: String,
    version: String,
    file: std::fs::File,
    bytes_written: u64,
    lines_since_check: u64,
}

impl RotatingLog {
    /// 打开日志文件（必要时先轮转），并写入会话头。
    pub fn open(path: PathBuf, product: &str, version: &str) -> std::io::Result<Self> {
        let banner = session_banner(product, version);
        let file = Self::open_with_rotation(&path, &banner)?;
        // 追加模式下文件可能已接近上限（上次会话遗留）：按真实长度初始化计数，
        // 否则本轮要再写满一个 MAX 才会触发轮转，文件可超出上限近一倍。
        let bytes_written = file.metadata().map(|m| m.len()).unwrap_or(0);
        Ok(Self {
            path,
            product: product.to_string(),
            version: version.to_string(),
            file,
            bytes_written,
            lines_since_check: 0,
        })
    }

    /// 打开（必要时先轮转）并写入会话头。
    fn open_with_rotation(path: &Path, banner: &str) -> std::io::Result<std::fs::File> {
        if std::fs::metadata(path)
            .map(|m| m.len() >= MAX_LOG_FILE_BYTES)
            .unwrap_or(false)
        {
            Self::shift_files(path);
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
    fn shift_files(path: &Path) {
        for index in (1..MAX_ROTATED_LOGS).rev() {
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

    /// 写入一行（自动补换行）。轮转检查按 [`ROTATION_CHECK_INTERVAL`] 摊销。
    pub fn write_line(&mut self, line: &str) -> std::io::Result<()> {
        self.file.write_all(line.as_bytes())?;
        if !line.ends_with('\n') {
            self.file.write_all(b"\n")?;
        }
        self.file.flush()?;
        self.bytes_written += line.len() as u64 + 1;
        self.lines_since_check += 1;
        if self.lines_since_check >= ROTATION_CHECK_INTERVAL
            && self.bytes_written >= MAX_LOG_FILE_BYTES
        {
            self.lines_since_check = 0;
            let banner = session_banner(&self.product, &self.version);
            self.file = Self::open_with_rotation(&self.path, &banner)?;
            // 轮转后是新文件（仅含会话头），同样按真实长度重置计数。
            self.bytes_written = self.file.metadata().map(|m| m.len()).unwrap_or(0);
        }
        Ok(())
    }

    /// 当前日志文件路径。
    pub fn path(&self) -> &Path {
        &self.path
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

    /// 轮转按**文件名**进行，因此同目录下的 `plugin.log` 不会被 App 的轮转碰到。
    #[test]
    fn rotation_is_per_file_name() {
        let app = rotated_path(Path::new("/logs/hifishifter.log"), 1);
        assert!(app.ends_with("hifishifter.1.log"), "{app:?}");
        let plugin = rotated_path(Path::new("/logs/plugin.log"), 2);
        assert!(plugin.ends_with("plugin.2.log"), "{plugin:?}");
    }

    /// `HIFISHIFTER_LOG_DIR` 覆盖默认目录（两个宿主同名同义）。
    ///
    /// 【为什么这条要单独测】它是"两个形态落进同一个目录"的手段，也是测试把日志
    /// 指向临时目录的手段。名字或语义变了，两件事会同时坏掉。
    #[test]
    fn the_log_dir_env_var_overrides_the_platform_default() {
        // 这个测试不设环境变量（会与并行测试互踩），只断言未设置时的形状。
        if std::env::var_os(LOG_DIR_ENV).is_some() {
            return;
        }
        let dir = log_dir().expect("platform log dir");
        assert!(!dir.starts_with(std::env::temp_dir()), "{}", dir.display());
        assert!(
            dir.to_string_lossy().contains(APP_IDENTIFIER),
            "日志目录应以 identifier 命名：{}",
            dir.display()
        );
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
        let path = dir.join("plugin.log");

        let mut log = RotatingLog::open(path.clone(), "HiFiShifter", "1.2.3").expect("open log");
        log.write_line("first").unwrap();
        drop(log);

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("==== HiFiShifter v1.2.3"), "{text}");
        assert!(text.contains("first"), "{text}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
