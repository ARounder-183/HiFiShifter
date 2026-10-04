//! 插件 DLL 的轻量诊断日志。日志路径由宿主进程环境变量指定。

use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Once;

static INIT: Once = Once::new();
static LOGGER: FileLogger = FileLogger;

struct FileLogger;

impl log::Log for FileLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let Some(path) = std::env::var_os("HIFISHIFTER_ARA_LOG") else {
            return;
        };
        if path.is_empty() {
            return;
        }
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

/// 安装一次进程级日志器；宿主已有日志器时保留宿主的实现。
pub fn init() {
    INIT.call_once(|| {
        if log::set_logger(&LOGGER).is_ok() {
            log::set_max_level(log::LevelFilter::Info);
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
}
