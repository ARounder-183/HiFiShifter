//! 插件 DLL 的轻量诊断日志。
//!
//! 【为什么默认就该写日志】原先只有在 `HIFISHIFTER_PLUGIN_LOG` 有值时才写文件 ——
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
//! 覆盖顺序：`HIFISHIFTER_PLUGIN_LOG`（精确落点，一个**文件**路径）>
//! `HIFISHIFTER_LOG_DIR`（与 App 同名同义的**目录**）> 平台默认。
//!
//! 【为什么这两个变量不再叫 `HIFISHIFTER_ARA_*`】宿主不提供 ARA 时本插件仍然可作为
//! 常规 VST3 使用（主要靠捕获通路），"ARA" 只描述其中一条通路。同一个前缀下还有几个
//! 探针/开发用的变量（`_PCM_PROBE` / `_TRANSPORT_PROBE` / `_EDITOR_ASSETS` …），
//! 它们一并改成了 `HIFISHIFTER_PLUGIN_*` —— 这组变量此前只存在于开发者脚本里，
//! 没有对外承诺，因此不保留旧名。

use hifishifter_kernel::logfile::RotatingLog;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once, OnceLock};

/// 未知宿主时的日志文件名。真正的文件名由 `plugin_log_name::log_file_name` 决定
/// （`plugin-<slug>.log`）；App 的 `hifishifter.log` 是另一回事，见模块文档。
const PLUGIN_LOG_FILE: &str = "plugin-unknown.log";

/// 会话头里的产品标识：两个形态的日志放在一起时要能一眼分清。
///
/// 【为什么不再写 `ARA plugin`】宿主不提供 ARA 时，这个插件仍然可作为常规 VST3
/// 使用（主要靠捕获通路）。写死 ARA 会让日志与「关于」对不上实际运行形态 ——
/// 用户拿着截图报障时，开发者会先怀疑"你装的到底是哪个"。这里写的是**构建形态**，
/// 运行时具体走 ARA 还是捕获由 `get_about_info` 的 `host` 字段如实给出。
const PRODUCT: &str = "HiFiShifter VST3 / ARA plugin";

static INIT: Once = Once::new();
static LOGGER: FileLogger = FileLogger;
/// 最终日志路径。宿主名要到 `setActive` 才拿得到，而日志在 DLL 初始化时就要能写 ——
/// 因此这里**可变**：先按"未知宿主"确定一次，宿主名到达后改写并重建写入器。
static LOG_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);
/// 宿主名（文件名用的原文）。由 `note_host_application` 写入。
static HOST_NAME: Mutex<Option<String>> = Mutex::new(None);
/// 写入器确定之前攒下的日志行。上限见 `MAX_PENDING_LINES`。
static PENDING: Mutex<Vec<String>> = Mutex::new(Vec::new());
/// 攒多少行就放弃等待宿主名、直接落到 `plugin-unknown.log`。
const MAX_PENDING_LINES: usize = 512;
/// 日志器装好之后的宽限期：超过它还没等到宿主名就直接落 unknown。
///
/// 【为什么需要】纯 VST3（未绑定 ARA）时 `IHostApplication` 也可能拿不到名字 ——
/// 若只按"缓冲满"才落盘，一个安静的会话可能一行日志都不写。
const HOST_NAME_GRACE: std::time::Duration = std::time::Duration::from_secs(3);
/// 日志器装好的时刻（见 `HOST_NAME_GRACE`）。
static STARTED: OnceLock<std::time::Instant> = OnceLock::new();
/// 持久的轮转写入器。取不到锁时按"内部错误"处理（`into_inner`），
/// 不因为日志本身让宿主进程 panic。
static WRITER: Mutex<Option<RotatingLog>> = Mutex::new(None);

struct FileLogger;

impl log::Log for FileLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        // 【为什么不再写死 Info】写死会把 `HIFISHIFTER_LOG=debug` 静默降级回 Info ——
        // 用户以为打开了详细日志，实际什么都没多。跟着已配置的上限走。
        metadata.level() <= log::max_level()
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
        if guard.is_none() {
            // 落点未定（宿主名还没到、也没过宽限期）：先攒着，别急着创建
            // `plugin-unknown.log` —— 那会让"按宿主分文件"在真拿到宿主名之前就定死。
            let buffered = {
                let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
                if pending.len() < MAX_PENDING_LINES {
                    pending.push(line.clone());
                    true
                } else {
                    false
                }
            };
            // 缓冲满、或已经等过了宽限期：直接落到 unknown，宁可文件名不完美也不能丢日志。
            let timed_out = STARTED
                .get()
                .is_some_and(|started| started.elapsed() >= HOST_NAME_GRACE);
            if buffered && !timed_out {
                return;
            }
            match open_writer() {
                Ok(mut writer) => {
                    drain_pending_into(&mut writer);
                    *guard = Some(writer);
                }
                Err(error) => {
                    eprintln!("hifishifter: plugin log unavailable: {error}");
                    return;
                }
            }
        }
        if let Some(writer) = guard.as_mut() {
            let _ = writer.write_line(&line);
        }
    }

    fn flush(&self) {}
}

/// 环境变量给出的精确落点（探针用）。设了它就不再按宿主分文件。
fn env_override_path() -> Option<PathBuf> {
    for variable in ["HIFISHIFTER_PLUGIN_LOG", "HIFISHIFTER_LOG_DIR"] {
        if let Some(raw) = std::env::var_os(variable) {
            let raw = raw.to_string_lossy().trim().to_owned();
            if raw.is_empty() {
                continue;
            }
            let path = PathBuf::from(&raw);
            // `HIFISHIFTER_LOG_DIR` 是**目录**（与 App 同名同义）；
            // `HIFISHIFTER_PLUGIN_LOG` 是探针用的**文件**路径。
            return Some(if variable == "HIFISHIFTER_LOG_DIR" {
                path.join(PLUGIN_LOG_FILE)
            } else {
                path
            });
        }
    }
    None
}

/// 当前应当使用的日志文件路径。
///
/// 【为什么不缓存成 OnceLock】宿主名要到 `setActive` 才拿得到，而在此之前日志已经
/// 开始写了。路径必须能随宿主名"改名"一次，见 `note_host_application`。
fn log_path() -> PathBuf {
    if let Some(path) = env_override_path() {
        return path;
    }
    let host = HOST_NAME
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let file = crate::plugin_log_name::log_file_name(host.as_deref());
    default_log_dir().join(file)
}

/// 按当前路径开一个写入器（会话头由 `RotatingLog::open` 写）。
fn open_writer() -> Result<RotatingLog, String> {
    let path = log_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let writer = RotatingLog::open(path, PRODUCT, crate::VERSION).map_err(|e| e.to_string())?;
    *LOG_PATH.lock().unwrap_or_else(|e| e.into_inner()) = Some(writer.path().to_path_buf());
    Ok(writer)
}

/// 把攒下的日志行补写进刚打开的写入器。
fn drain_pending_into(writer: &mut RotatingLog) {
    let pending: Vec<String> = {
        let mut guard = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *guard)
    };
    for line in pending {
        let _ = writer.write_line(&line);
    }
}

/// 记下宿主应用名，并把日志切到 `plugin-<slug>.log`。
///
/// 【为什么必须在起日志之后才调】`IHostApplication::getName` 只有 `setActive` 之后
/// 可用，而日志在 DLL 加载时就要能写。于是先攒着、拿到名字时再一次性落地：
/// 会话头会开在**正确的文件**里，早期几行也不会丢。
pub fn note_host_application(name: &str) {
    {
        *HOST_NAME.lock().unwrap_or_else(|e| e.into_inner()) = Some(name.to_owned());
    }
    let mut guard = WRITER.lock().unwrap_or_else(|e| e.into_inner());
    match open_writer() {
        Ok(mut writer) => {
            drain_pending_into(&mut writer);
            *guard = Some(writer);
        }
        Err(error) => eprintln!("hifishifter: plugin log unavailable: {error}"),
    }
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
    log_file()
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(default_log_dir)
}

/// 当前日志文件路径（供"复制路径"用）。
pub fn log_file() -> PathBuf {
    // 【为什么优先读实际打开的路径】`log_path()` 是"按当前宿主名应当用的路径"，
    // 而写入器一旦打开就以它为准（宿主名可能再变，但文件已经在那儿了）。
    // 两者在"名字还没到"的那段时间里不同，报给用户的必须是**真的写着日志**那个。
    if let Some(path) = LOG_PATH
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
    {
        return path;
    }
    log_path()
}

/// 日志级别：release 默认 `Warn`，debug 默认 `Info`，`HIFISHIFTER_LOG` 可覆盖。
///
/// 【为什么 release 不能默认 Info】插件跑在宿主进程里，日志落在用户磁盘上。此前
/// 无条件设成 `Info`，而宿主回调的节奏（每次重算、每次清单刷新、每次几何查询）都会
/// 打点 —— 实测一次 40 分钟的会话写满 3 个 8 MiB 轮转件。那些点对开发者有用、对
/// 用户只是磁盘占用与 I/O。需要时用 `HIFISHIFTER_LOG=info`（或 `debug`/`trace`）
/// 打开，排查现场的第一步就是它。
///
/// 取值与 `log` crate 的过滤名一致：`off`/`error`/`warn`/`info`/`debug`/`trace`。
fn log_level_from_env() -> log::LevelFilter {
    if let Ok(value) = std::env::var("HIFISHIFTER_LOG") {
        if let Ok(level) = value
            .trim()
            .to_ascii_lowercase()
            .parse::<log::LevelFilter>()
        {
            return level;
        }
    }
    if cfg!(debug_assertions) {
        log::LevelFilter::Info
    } else {
        log::LevelFilter::Warn
    }
}

/// 安装一次进程级日志器；宿主已有日志器时保留宿主的实现。
pub fn init() {
    INIT.call_once(|| {
        if log::set_logger(&LOGGER).is_ok() {
            log::set_max_level(log_level_from_env());
        }
        let _ = STARTED.set(std::time::Instant::now());
        // 目录不存在就创建：用户第一次打开 Help 菜单之前，日志文件必须已经可写。
        let dir = log_path();
        if let Some(dir) = dir.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        // 【为什么这里不直接开文件】宿主名（`IHostApplication::getName`）要到
        // `setActive` 之后才有，而按宿主分文件要求文件名里带它。此处只把 logger 装上：
        // 早期日志行先进内存缓冲，`note_host_application` 到达时再一次性落到
        // `plugin-<slug>.log`（见该函数与 `FileLogger::log` 的说明）。
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
        if std::env::var_os("HIFISHIFTER_PLUGIN_LOG").is_some()
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
    ///
    /// 具体形状是 `plugin-<宿主>.log` / `plugin-unknown.log`（见 `plugin_log_name`）；
    /// 这里只钉住"插件前缀 + 与 App 的 `hifishifter.log` 不同名"。
    #[test]
    fn the_plugin_log_file_name_differs_from_the_app() {
        let name = log_file().file_name().and_then(|n| n.to_str()).map(str::to_owned);
        let name = name.expect("log file name");
        assert!(name.starts_with("plugin-"), "{name}");
        assert!(name.ends_with(".log"), "{name}");
        assert_ne!(name, "hifishifter.log");
    }

    /// 覆盖变量优先于平台默认目录。
    ///
    /// 【为什么要测】它是探针把日志钉到指定落点的手段，也是"两个形态落进同一目录"
    /// 之外唯一还允许的改道方式。语义（文件 vs 目录）弄反了会让日志写到奇怪的地方。
    #[test]
    fn the_env_overrides_take_precedence() {
        // 不实际设置环境变量（会与并行测试互踩），只断言**未设置**时用的是默认目录。
        if std::env::var_os("HIFISHIFTER_PLUGIN_LOG").is_some()
            || std::env::var_os("HIFISHIFTER_LOG_DIR").is_some()
        {
            return;
        }
        let name = log_file().file_name().and_then(|n| n.to_str()).unwrap_or_default().to_owned();
        assert!(name.starts_with("plugin-"), "{name}");
        assert!(log_directory()
            .to_string_lossy()
            .contains("com.arounder.hifishifter"));
    }
}
