//! 插件 DLL 的轻量诊断日志。
//!
//! 【为什么默认就该写日志】原先只有在 `HIFISHIFTER_ARA_LOG` 有值时才写文件 ——
//! 而那是探针脚本专用的环境变量，普通用户根本不会设。于是"插件出问题时去 Help →
//! 打开日志目录拿日志"这条路是断的：那里既没有日志，菜单项本身也会因为命令未实现
//! 而报错。REAPER 不提供插件日志后端，所以文件日志是唯一的排查手段。
//!
//! 位置与独立 App 的日志同源（`%LOCALAPPDATA%\HiFiShifter\logs`，见 App 的
//! `logging.rs`）：用户已经知道去那里找，两个形态的日志也该放在一起。
//!
//! 覆盖顺序：`HIFISHIFTER_ARA_LOG`（探针的精确落点）> `HIFISHIFTER_LOG_DIR`
//! （与 App 同名同义）> 平台默认。

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Once, OnceLock};

static INIT: Once = Once::new();
static LOGGER: FileLogger = FileLogger;
static LOG_PATH: OnceLock<PathBuf> = OnceLock::new();

struct FileLogger;

impl log::Log for FileLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let Some(path) = log_path() else {
            return;
        };
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(
                file,
                "{}",
                format_log_line(record.level(), record.target(), &record.args().to_string())
            );
        }
    }

    fn flush(&self) {}
}

/// 解析日志文件路径（结果只算一次）。
fn log_path() -> Option<&'static PathBuf> {
    Some(LOG_PATH.get_or_init(|| {
        for variable in ["HIFISHIFTER_ARA_LOG", "HIFISHIFTER_LOG_DIR"] {
            if let Some(raw) = std::env::var_os(variable) {
                let raw = raw.to_string_lossy().trim().to_owned();
                if raw.is_empty() {
                    continue;
                }
                let path = PathBuf::from(&raw);
                // `HIFISHIFTER_LOG_DIR` 是**目录**（与 App 同名同义）；
                // `HIFISHIFTER_ARA_LOG` 是探针用的**文件**路径。
                return if variable == "HIFISHIFTER_LOG_DIR" {
                    path.join("plugin.log")
                } else {
                    path
                };
            }
        }
        hifishifter_kernel::config_location::local_data_subdir("logs").join("plugin.log")
    }))
}

/// 日志文件所在目录（供 Help 菜单展示 / 打开）。
pub fn log_directory() -> PathBuf {
    log_path()
        .and_then(|path| path.parent().map(PathBuf::from))
        .unwrap_or_else(|| hifishifter_kernel::config_location::local_data_subdir("logs"))
}

/// 安装一次进程级日志器；宿主已有日志器时保留宿主的实现。
pub fn init() {
    INIT.call_once(|| {
        if log::set_logger(&LOGGER).is_ok() {
            log::set_max_level(log::LevelFilter::Info);
        }
        // 目录不存在就创建：用户第一次打开 Help 菜单之前，日志文件必须已经可写。
        if let Some(dir) = log_path().and_then(|path| path.parent()) {
            let _ = std::fs::create_dir_all(dir);
        }
    });
}

/// 把日志记录转换成稳定的一行，便于隔离 REAPER 采集后直接 diff。
fn format_log_line(level: log::Level, target: &str, message: &str) -> String {
    format!("[{level}] {target}: {message}")
}

#[cfg(test)]
mod tests {
    #[test]
    fn formats_a_record_for_the_probe_log() {
        assert_eq!(
            super::format_log_line(log::Level::Info, "hifishifter_plugin::ara", "ara: clips=2"),
            "[INFO] hifishifter_plugin::ara: ara: clips=2"
        );
    }

    /// 日志目录必须落在本机数据目录下，而不是临时目录 —— 临时目录会在会话进行中
    /// 被磁盘清理，用户事后拿不到日志。
    #[test]
    fn the_default_log_directory_is_not_the_temp_directory() {
        // 只有在没有任何环境变量覆盖时才检查默认值。
        if std::env::var_os("HIFISHIFTER_ARA_LOG").is_some()
            || std::env::var_os("HIFISHIFTER_LOG_DIR").is_some()
        {
            return;
        }
        let dir = super::log_directory();
        assert!(
            !dir.starts_with(std::env::temp_dir()),
            "日志目录落到了临时目录：{}",
            dir.display()
        );
        assert!(
            dir.ends_with("logs"),
            "日志目录应以 logs 收尾：{}",
            dir.display()
        );
    }
}
