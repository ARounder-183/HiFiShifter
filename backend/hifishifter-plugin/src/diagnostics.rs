//! 插件的**唯一**文件日志实现：VST3、CLAP 以及后续所有格式共用。
//!
//! 【为什么只能有一份】此前 CLAP 外壳在 `log_host.rs` 里另写了一套：固定文件名、
//! 无时间戳、无会话头、无轮转、写死 Info；而共享层的宿主身份解析又会去调本模块的
//! `note_host_application`，于是一个 CLAP 进程**同时**在 VST3 的文件里开一个空会话、
//! 在自己的文件里写无格式的行。用户拿到的是"所有时期的日志混在一起、分不出哪次运行、
//! 也分不出哪份二进制"。日志一旦出现两套实现，就必然出现这种交叉污染。
//!
//! 因此目录、行格式、会话头、轮转、级别、panic hook 全部来自这里与
//! [`hifishifter_kernel::logfile`]；格式外壳唯一的差异是**自己那条宿主的日志后端**
//! （CLAP 的 `clap.log`），那通过一个格式无关的转发 sink 登记进来，而不是另装一个
//! `log::Log`。
//!
//! 文件名 = `plugin-<format>-<host>.log`（见 `plugin_log_name`）。format 来自唯一的
//! 格式登记点 [`crate::log_tag`]，新增格式不需要改本模块。
//!
//! 覆盖顺序：`HIFISHIFTER_PLUGIN_LOG`（精确落点，一个**文件**路径）>
//! `HIFISHIFTER_LOG_DIR`（与 App 同名同义的**目录**）> 平台默认。

use hifishifter_kernel::logfile::RotatingLog;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once, OnceLock};

/// 宿主日志后端的签名：把一条**已格式化**的行交给宿主自己的日志面板。
///
/// 级别用 `log::Level` 而不是 CLAP 的 severity 常量 —— 后者是 CLAP 私有细节，不该
/// 渗进共享层；外壳在自己的 sink 里做一次映射即可。
pub type HostLogSink = fn(log::Level, &str);

/// 宿主是否已发起本格式的 ARA 握手。
///
/// 【为什么这个读数在共享层】"这个宿主有没有把本插件当 ARA 插件"是报障里最常被
/// 问、却最难自证的一件事。格式外壳在任一 ARA 发现路径触发时登记这里：ARA SDK 2.3
/// 的 entry 级工厂查询或实例级扩展查询。诊断包与会话日志据此**如实**报告，而不是
/// 让读者去猜插件装没装对。
static ARA_FACTORY_QUERIED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 登记"宿主已发起本格式的 ARA 握手"（格式外壳调用，每进程一次即可）。
///
/// 名字保留 `factory_queried` 是为了不动既有 JSON 字段 `factoryQueriedByHost`；
/// 语义已扩展为"任一发现路径触发了 ARA 入口"。
pub fn note_ara_factory_queried() {
    ARA_FACTORY_QUERIED.store(true, std::sync::atomic::Ordering::Release);
}

/// ARA 能力读数（供诊断包 system_info 与会话日志使用）。
pub fn ara_discovery_json() -> serde_json::Value {
    let active = ARA_FACTORY_QUERIED.load(std::sync::atomic::Ordering::Acquire);
    serde_json::json!({
        // 兼容键：字段名沿用历史约定，语义已扩展为"任一 ARA 发现路径被触发"。
        "factoryQueriedByHost": active,
        "araActive": active,
        "format": crate::log_tag(),
        "note": if active {
            "the host started an ARA handshake for this plug-in format"
        } else {
            "no ARA handshake was started for this plug-in format; the plug-in runs in Capture mode unless the host uses a different discovery path"
        },
    })
}

/// 由格式外壳登记的宿主日志后端（可缺省）。
///
/// 【为什么是 `OnceLock` 而不是可覆盖的指针】`log::set_logger` 全进程只能成功一次，
/// 日志也只能交给一个宿主实例；允许第二个实例覆盖会让第一个实例的诊断凭空改道。
static HOST_SINK: OnceLock<HostLogSink> = OnceLock::new();

/// 登记宿主日志后端；重复登记忽略（第一个赢）。装不装都不影响文件日志。
pub fn set_host_sink(sink: HostLogSink) {
    let _ = HOST_SINK.set(sink);
}

/// 会话头里的产品标识由格式标签派生（见 `crate::log_product`）。
fn product() -> String {
    crate::log_product()
}

static INIT: Once = Once::new();
static LOGGER: FileLogger = FileLogger;
/// 最终日志路径。宿主名要到 `setActive` 才拿得到，而日志在 DLL 初始化时就要能写 ——
/// 因此这里**可变**：先按"未知宿主"确定一次，宿主名到达后改写并重建写入器。
static LOG_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);
/// 宿主名（文件名用的原文）。由 `note_host_application` 写入。
static HOST_NAME: Mutex<Option<String>> = Mutex::new(None);
/// 写入器确定之前攒下的日志行。上限见 `MAX_PENDING_LINES`。
static PENDING: Mutex<Vec<String>> = Mutex::new(Vec::new());
/// 攒多少行就放弃等待宿主名、直接落到 `plugin-<format>-unknown.log`。
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
        // 【先交宿主，再写文件】宿主面板是实时的，文件是事后取证的；两路都无条件走，
        // 互不阻塞对方的判断。
        if let Some(sink) = HOST_SINK.get() {
            sink(record.level(), &line);
        }
        let mut guard = WRITER.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            // 缓冲满、或已经等过了宽限期：直接落到 unknown，宁可文件名不完美也不能丢日志。
            let timed_out = STARTED
                .get()
                .is_some_and(|started| started.elapsed() >= HOST_NAME_GRACE);
            // 【为什么显式路径不缓冲】设了 `HIFISHIFTER_PLUGIN_LOG`/`HIFISHIFTER_LOG_DIR`
            // 的人要的就是**那个文件**：这时还按"等宿主名"攒着，等于让排查者对着一个
            // 空目录猜。实测过一个合成宿主：宿主名没到、宽限期又没新记录触发，
            // 整段缓冲就随进程退出一起消失了。
            let explicit = has_explicit_override();
            // 【顺序很重要】只有"确实要攒着"时才入队：先入队再决定要不要落盘，
            // 会让同一条日志既进缓冲又被 `drain_pending_into` 写一遍 —— 输出里出现
            // 两条一模一样、时间戳也一样的行。这个缺陷真的发生过。
            // 【为什么 GUI 冻结诊断不走缓冲】`gui_log_line` 产出的行以 `[gui] ` 开头，
            // 用来定位"宿主冻住"；冻结恰好发生在启动后一两秒内，那时写入器还没落定，
            // 整段缓冲会随进程一起消失。用 `starts_with` 而不是 `contains`：CLAP 的
            // `[clap][gui] ...` 只是普通视图日志，不该绕过缓冲、在 unknown 文件里
            // 留下半段会话。
            let urgent = record.args().to_string().starts_with("[gui] ");
            if !timed_out && !explicit && !urgent {
                let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
                if pending.len() < MAX_PENDING_LINES {
                    // 落点未定（宿主名还没到、也没过宽限期）：先攒着，别急着创建
                    // `plugin-<format>-unknown.log` —— 那会让"按宿主分文件"在真拿到宿主名之前就定死。
                    pending.push(line.clone());
                    return;
                }
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

/// 精确文件落点（`HIFISHIFTER_PLUGIN_LOG`，探针用）。设了它就完全固定，
/// 不再按格式/宿主分文件。
///
/// 【为什么只有它关缓冲】设了精确路径的人要的就是**那个文件**，让排查者对着一个
/// 空目录猜是不可接受的。而 `HIFISHIFTER_LOG_DIR` 只是改目录：宿主名尚未到达时
/// 仍应把早期行攒在内存里，等 `note_host_application` 一次性补进正确文件 ——
/// 否则每次宿主改名都会在 unknown 文件里留半段、再在真名文件里留半段。
fn explicit_file() -> Option<PathBuf> {
    std::env::var_os("HIFISHIFTER_PLUGIN_LOG")
        .map(|raw| raw.to_string_lossy().trim().to_owned())
        .filter(|raw| !raw.is_empty())
        .map(PathBuf::from)
}

/// 日志目录覆盖（`HIFISHIFTER_LOG_DIR`，与 App 同名同义）。**只改目录**，
/// 文件名仍按格式 + 当前宿主名生成 —— 否则宿主名到达时 `note_host_application`
/// 的改名会被这个固定名吞掉。
fn log_dir_override() -> Option<PathBuf> {
    std::env::var_os("HIFISHIFTER_LOG_DIR")
        .map(|raw| raw.to_string_lossy().trim().to_owned())
        .filter(|raw| !raw.is_empty())
        .map(PathBuf::from)
}

/// 精确文件落点是否关掉了"等宿主名"的内存缓冲。
fn has_explicit_override() -> bool {
    explicit_file().is_some()
}

/// 当前应当使用的日志文件路径。
///
/// 【为什么不缓存成 OnceLock】宿主名只有 `setActive` / `clap_host_t.name` 之后
/// 拿得到，而在此之前日志已经开始写了。路径必须能随宿主名"改名"一次，见
/// `note_host_application`。
fn log_path() -> PathBuf {
    if let Some(path) = explicit_file() {
        return path;
    }
    let host = HOST_NAME.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let file = hifishifter_plugin_core::plugin_log_name::log_file_name(
        Some(crate::log_tag()),
        host.as_deref(),
    );
    let dir = log_dir_override().unwrap_or_else(default_log_dir);
    dir.join(file)
}

/// 按当前路径开一个写入器（会话头由 `RotatingLog::open` 写）。
fn open_writer() -> Result<RotatingLog, String> {
    let path = log_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let writer = RotatingLog::open(path, &product(), crate::VERSION).map_err(|e| e.to_string())?;
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

/// 把还在缓冲里的日志行落盘。
///
/// 【为什么必须有】缓冲只在"下一条记录到达且已过宽限期"时才会落盘。宿主不给名字、
/// 之后也不再有任何日志的进程（合成宿主、无头探针、崩在 `setActive` 之前的宿主）
/// 会把整段缓冲**随退出一起丢掉** —— 而那段恰恰是最需要的东西。卸载入口调一次，
/// 成本是一行 `write`。
pub fn flush_pending() {
    let mut guard = WRITER.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let Ok(mut writer) = open_writer() else {
            return;
        };
        drain_pending_into(&mut writer);
        *guard = Some(writer);
        return;
    }
    if let Some(writer) = guard.as_mut() {
        let _ = writer.write_line(""); // 触发一次落盘（write_line 内部按行写）
    }
}

/// 记下宿主应用名，并把日志切到 `plugin-<format>-<host>.log`。
///
/// 【为什么必须在起日志之后才调】宿主名只有 `setActive` / `clap_host_t.name` 之后
/// 可用，而日志在模块加载时就要能写。于是先攒着、拿到名字时再一次性落地：
/// 会话头会开在**正确的文件**里，早期几行也不会丢。
///
/// 【为什么必须幂等】CLAP 的每个实例 `init` 都会解析一次宿主身份，VST3 也可能在
/// 加载期先给 `unknown`、之后补一个真名。每次都重建写入器会在文件里留下一串
/// **只有会话头、没有内容**的空会话 —— 用户看到的就是"日志被反复截断"。同一个
/// 名字重复到达时直接返回；`unknown` 不落名，等真名来。
pub fn note_host_application(name: &str) {
    if name.is_empty() || name.eq_ignore_ascii_case("unknown") {
        return;
    }
    // 【为什么先取 WRITER 再比对】宿主名与写入器都是同一份会话状态：只有
    // "名字未变且写入器已落地"才是幂等返回。名字已登记但写入器还没打开时仍要
    // 在这里把缓冲里的早期日志补进正确文件，否则那一段会话会永久留在内存里。
    let mut guard = WRITER.lock().unwrap_or_else(|e| e.into_inner());
    {
        let mut current = HOST_NAME.lock().unwrap_or_else(|e| e.into_inner());
        if current.as_deref() == Some(name) && guard.is_some() {
            return;
        }
        *current = Some(name.to_owned());
    }
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
    if let Some(path) = LOG_PATH.lock().unwrap_or_else(|e| e.into_inner()).clone() {
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
        // `plugin-<format>-<host>.log`（见该函数与 `FileLogger::log` 的说明）。
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
        if let Some(sink) = HOST_SINK.get() {
            sink(log::Level::Error, &line);
        }
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

    /// 文件名必须同时带格式与宿主 —— 这是"日志不再混在一起"的契约。
    #[test]
    fn the_log_file_name_carries_both_dimensions() {
        let clap =
            hifishifter_plugin_core::plugin_log_name::log_file_name(Some("clap"), Some("REAPER"));
        let vst3 =
            hifishifter_plugin_core::plugin_log_name::log_file_name(Some("vst3"), Some("REAPER"));
        assert_eq!(clap, "plugin-clap-reaper.log");
        assert_ne!(clap, vst3, "同一宿主里两种格式不能共用一个文件");
    }

    /// 会话头由格式标签派生，新增格式不需要改这里。
    #[test]
    fn the_product_banner_names_the_format() {
        assert!(product().contains(&crate::log_tag().to_ascii_uppercase()));
    }

    /// 插件日志必须落在**与 App 同一个目录**里。
    ///
    /// 【为什么这条是核心】目录曾经差一层（插件多了 `HiFiShifter` 子目录），
    /// 于是用户要在两个地方找日志。现在两边都走 `kernel::logfile::log_dir`，
    /// 这条断言把"同一个目录"钉住；文件名不同是另一件事（见下一条）。
    #[test]
    fn the_plugin_log_shares_the_app_log_directory() {
        if std::env::var_os("HIFISHIFTER_PLUGIN_LOG").is_some()
            || std::env::var_os("HIFISHIFTER_LOG_DIR").is_some()
            || (cfg!(windows)
                && std::env::var_os("LOCALAPPDATA").is_none()
                && std::env::var_os("USERPROFILE").is_none())
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
    /// 具体形状是 `plugin-<format>-<host>.log`（见 `plugin_log_name`）；
    /// 这里只钉住"插件前缀 + 与 App 的 `hifishifter.log` 不同名"。
    #[test]
    fn the_plugin_log_file_name_differs_from_the_app() {
        let name = log_file()
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_owned);
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
            || (cfg!(windows)
                && std::env::var_os("LOCALAPPDATA").is_none()
                && std::env::var_os("USERPROFILE").is_none())
        {
            return;
        }
        let name = log_file()
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_owned();
        assert!(name.starts_with("plugin-"), "{name}");
        // 目录里必须出现**产品目录名**（v2 布局）；历史上这里断言的是 identifier 那
        // 一层，它正是"用户拿到一串反向域名"的成因。
        assert!(log_directory()
            .to_string_lossy()
            .contains(hifishifter_kernel::app_paths::PRODUCT_DIR));
    }
}
