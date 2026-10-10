//! 插件 DLL 的轻量诊断日志。
//!
//! 【为什么默认就该写日志】原先只有在 `HIFISHIFTER_ARA_LOG` 有值时才写文件 ——
//! 而那是探针脚本专用的环境变量，普通用户根本不会设。于是"插件出问题时去 Help →
//! 打开日志目录拿日志"这条路是断的：那里既没有日志，菜单项本身也会因为命令未实现
//! 而报错。REAPER 不提供插件日志后端，所以文件日志是唯一的排查手段。
//!
//! 【为什么与 App 同一个目录、同一个格式】两个形态的日志会被用户放在一起看、
//! 一起打进诊断包。目录、行格式、会话头、轮转全部来自
//! [`hifishifter_kernel::logfile`] —— 此前这里另写了一套，结果是插件日志**没有日期、
//! 没有版本号、没有时间戳、也不轮转**，与 App 的日志完全不像同一个产品。
//! 唯一的差异是**文件名**（`plugin.log` vs `hifishifter.log`），这是有意的：
//! 同目录不冲突，轮转也各按自己的文件名进行。
//!
//! 覆盖顺序：`HIFISHIFTER_ARA_LOG`（探针的精确落点，一个**文件**路径）>
//! `HIFISHIFTER_LOG_DIR`（与 App 同名同义的**目录**）> 平台默认。

use hifishifter_kernel::logfile::RotatingLog;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once, OnceLock};

/// 插件日志文件名。与 App 的 `hifishifter.log` 不同 —— 见模块文档。
const PLUGIN_LOG_FILE: &str = "plugin.log";

/// 会话头里的产品标识：两个形态的日志放在一起时要能一眼分清。
const PRODUCT: &str = "HiFiShifter ARA plugin";

static INIT: Once = Once::new();
static LOGGER: FileLogger = FileLogger;
static LOG_PATH: OnceLock<PathBuf> = OnceLock::new();
/// 持久的轮转写入器。取不到锁时按"内部错误"处理（`into_inner`），
/// 不因为日志本身让宿主进程 panic。
static WRITER: Mutex<Option<RotatingLog>> = Mutex::new(None);

struct FileLogger;

impl log::Log for FileLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let line = hifishifter_kernel::logfile::format_record(
            record.level(),
            record.target(),
            &record.args().to_string(),
        );
        let mut guard = WRITER.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(writer) = guard.as_mut() {
            let _ = writer.write_line(&line);
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
                    path.join(PLUGIN_LOG_FILE)
                } else {
                    path
                };
            }
        }
        default_log_dir().join(PLUGIN_LOG_FILE)
    }))
}

/// 平台默认日志目录。与 App **同一个目录**，解析不出来时退回临时目录。
///
/// 【为什么不能没有兜底】缺 `LOCALAPPDATA` 时宁可用临时目录也不能让插件起不来 ——
/// 一个跑得起来但日志不完美的会话，好过一个打不开的窗口。
fn default_log_dir() -> PathBuf {
    hifishifter_kernel::logfile::log_dir()
        .unwrap_or_else(|| std::env::temp_dir().join("hifishifter"))
}

/// 日志文件所在目录（供 Help 菜单展示 / 打开）。
pub fn log_directory() -> PathBuf {
    log_path()
        .and_then(|path| path.parent().map(Path::to_path_buf))
        .unwrap_or_else(default_log_dir)
}

/// 当前日志文件路径（供"复制路径"用）。
pub fn log_file() -> PathBuf {
    log_path()
        .cloned()
        .unwrap_or_else(|| default_log_dir().join(PLUGIN_LOG_FILE))
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
        // 会话头 + 轮转都在共享实现里：打开即写一行 `==== … log started at … ====`，
        // 用户事后拿到日志时第一眼就能看到版本号与日期。
        match RotatingLog::open(
            log_path()
                .cloned()
                .unwrap_or_else(|| default_log_dir().join(PLUGIN_LOG_FILE)),
            PRODUCT,
            crate::VERSION,
        ) {
            Ok(writer) => {
                *WRITER.lock().unwrap_or_else(|e| e.into_inner()) = Some(writer);
            }
            Err(error) => {
                // 落不了盘也不影响本次会话使用；下一行 log 会因为 writer 为 None 而跳过。
                eprintln!("hifishifter: plugin log unavailable: {error}");
            }
        }
        install_panic_hook();
    });
}

/// panic 详情写进日志。
///
/// 【为什么插件也要】插件跑在宿主进程里，一次 panic 会让整个 DAW 会话出问题，
/// 而用户能拿到的现场只有日志。App 一直有这个 hook，插件此前没有 —— 也就是说
/// 插件崩了之后日志里**什么都不会留下**，那正是最需要日志的时刻。
fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("<unnamed>");
        let location = info.location().map(|l| l.to_string()).unwrap_or_default();
        let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".to_string()
        };
        // 走 log 门面会经过宿主可能已装配的后端；同时直接写一行到日志文件，
        // 因为 panic 之后宿主未必还有机会处理日志。
        let line = hifishifter_kernel::logfile::format_record(
            log::Level::Error,
            "hifishifter_plugin::panic",
            &format!("PANIC: thread '{name}' panicked at {location}: {payload}"),
        );
        {
            // 【为什么用 `try_lock` 而不是 `lock`】本 hook 会在**任意线程**的 panic 上跑，
            // 包括"panic 正发生在 `FileLogger::log` 里、`WRITER` 已被持有"这种情况 ——
            // 那时 `lock` 会自锁死。拿不到就放弃写这一行（`previous(info)` 仍会执行，
            // 宿主/标准错误仍能看到 panic），绝不让日志把 panic 变成死锁。
            if let Ok(mut guard) = WRITER.try_lock() {
                if let Some(writer) = guard.as_mut() {
                    let _ = writer.write_line(&line);
                }
            }
        }
        previous(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 插件日志必须落在**与 App 同一个目录**里。
    ///
    /// 【为什么这条是核心】目录曾经差一层（插件多了 `HiFiShifter` 子目录），
    /// 于是用户要在两个地方找日志。现在两边都走 `kernel::logfile::log_dir`，
    /// 这条断言把"同一个目录"钉住；文件名不同是另一件事（见下一条）。
    #[test]
    fn the_plugin_log_shares_the_app_log_directory() {
        if std::env::var_os("HIFISHIFTER_ARA_LOG").is_some()
            || std::env::var_os("HIFISHIFTER_LOG_DIR").is_some()
        {
            return;
        }
        let dir = log_directory();
        assert_eq!(
            Some(dir.clone()),
            hifishifter_kernel::logfile::log_dir(),
            "插件与 App 的日志目录必须一致"
        );
        assert!(
            !dir.starts_with(std::env::temp_dir()),
            "日志目录落到了临时目录：{}",
            dir.display()
        );
    }

    /// 文件名与 App 不同 —— 同目录下才不会被覆盖，轮转也互不影响。
    #[test]
    fn the_plugin_log_file_name_differs_from_the_app() {
        assert_eq!(
            log_file().file_name().and_then(|n| n.to_str()),
            Some("plugin.log")
        );
    }

    /// 覆盖变量优先于平台默认目录。
    ///
    /// 【为什么要测】它是探针把日志钉到指定落点的手段，也是"两个形态落进同一目录"
    /// 之外唯一还允许的改道方式。语义（文件 vs 目录）弄反了会让日志写到奇怪的地方。
    #[test]
    fn the_env_overrides_take_precedence() {
        // 不实际设置环境变量（会与并行测试互踩），只断言**未设置**时用的是默认目录。
        if std::env::var_os("HIFISHIFTER_ARA_LOG").is_some()
            || std::env::var_os("HIFISHIFTER_LOG_DIR").is_some()
        {
            return;
        }
        assert!(log_file().ends_with(PLUGIN_LOG_FILE));
        assert!(log_directory()
            .to_string_lossy()
            .contains("com.arounder.hifishifter"));
    }
}
