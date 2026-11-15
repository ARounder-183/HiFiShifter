//! 日志详细程度的**唯一**事实来源。
//!
//! 【为什么必须在内核】同一份代码跑在三个进程形态里（独立 App / VST3-ARA / CLAP），
//! 过去 App 的 `logging.rs` 与插件的 `diagnostics.rs` 各写了一份级别解析、各有一个
//! 出厂默认值（release 下 App=Info、插件=Warn），五语手册却统一写成 `info`。
//! 用户选择档位时，"四档是什么""谁覆盖谁""改档后哪些通道要关"必须在**一处**回答，
//! 否则帮助菜单只是一个看起来能点的控件。
//!
//! 【为什么不用 `log::set_logger` 热切换】logger 是进程级单例，只能成功安装一次
//! （插件在宿主已装 logger 时还会直接失败）。运行时改档只能改 `log::set_max_level`
//! 与各 sink 共同读取的原子量。本模块的 `CURRENT` 就是那个原子量。
//!
//! 【为什么这个文件不依赖 Tauri / React / settings_store】日志初始化早于 Tauri setup、
//! 早于前端 store、插件甚至早于 DLL 的组件创建（见 `app_paths_migrate` 的顺序约束）。
//! 策略读取只能依赖 `log`、`serde`、标准库与 `app_paths`。

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

/// 出厂默认档位。**用户可见的默认值 = 低**。
pub const DEFAULT_VERBOSITY: LogVerbosity = LogVerbosity::Low;
/// 新策略文件格式版本；兼容升级用（未来值回落默认，不丢弃用户的"不要日志"选择）。
pub const POLICY_VERSION: u32 = 1;
/// 新环境变量（两个宿主通用）。
pub const VERBOSITY_ENV: &str = "HIFISHIFTER_LOG_VERBOSITY";
/// 兼容旧环境变量；取值 `off|error|warn|info|debug|trace`。
pub const LEGACY_LOG_ENV: &str = "HIFISHIFTER_LOG";

/// 用户可选的四档详细程度。
///
/// serde 的 `rename_all` 决定 `log_settings.json` 与命令载荷里的字面量；
/// 前端 `LogVerbosity` 联合类型逐字对应，不做第二份映射。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogVerbosity {
    /// 什么都不产生。
    Off,
    /// 故障信号（Error / Warn）。
    #[default]
    Low,
    /// 低 + 会话边界、用户操作、关键生命周期。
    Medium,
    /// 中 + 逐回调 / 逐块诊断、探针、环境明细。
    High,
}

impl LogVerbosity {
    pub const ALL: [Self; 4] = [Self::Off, Self::Low, Self::Medium, Self::High];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|level| level.as_str() == value.trim().to_ascii_lowercase())
    }

    /// 档位 → `log` 门面的唯一映射。
    pub fn level_filter(self) -> log::LevelFilter {
        match self {
            Self::Off => log::LevelFilter::Off,
            Self::Low => log::LevelFilter::Warn,
            Self::Medium => log::LevelFilter::Info,
            Self::High => log::LevelFilter::Debug,
        }
    }

    /// 档位 → 完整策略的唯一映射。禁止宿主与前端再写 `if verbosity == "high"`。
    pub fn policy(self) -> LogPolicy {
        let (max_file_bytes, max_files, detail_cap_chars) = match self {
            Self::Off => (0, 0, 0),
            Self::Low => (8 * 1024 * 1024, 3, 1000),
            Self::Medium => (8 * 1024 * 1024, 3, 2000),
            // 排障期保留更多上下文；诊断包与帮助菜单的状态行按策略读数展示。
            Self::High => (8 * 1024 * 1024, 5, 4000),
        };
        let enabled = self != Self::Off;
        LogPolicy {
            verbosity: self,
            level_filter: self.level_filter(),
            file_enabled: enabled,
            stderr_tee: enabled,
            host_sink: enabled,
            probes: self == Self::High,
            frontend_accept: FrontendAccept {
                error: enabled,
                warn: enabled && self != Self::Low,
                info: self == Self::Medium || self == Self::High,
                debug: self == Self::High,
            },
            detail_cap_chars,
            max_single_line_chars: if enabled { 4096 } else { 0 },
            max_file_bytes,
            max_files,
        }
    }
}

/// 前端四个级别是否被接收（低档只回传 error，见策略矩阵）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrontendAccept {
    pub error: bool,
    pub warn: bool,
    pub info: bool,
    pub debug: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogPolicy {
    pub verbosity: LogVerbosity,
    pub level_filter: log::LevelFilter,
    pub file_enabled: bool,
    pub stderr_tee: bool,
    pub host_sink: bool,
    pub probes: bool,
    pub frontend_accept: FrontendAccept,
    pub detail_cap_chars: usize,
    pub max_single_line_chars: usize,
    pub max_file_bytes: u64,
    pub max_files: usize,
}

/// 生效档位来自哪里。UI 据此把四个单选项禁用并给出说明（不新增入口）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PolicySource {
    Default,
    Persisted,
    /// 新的 `HIFISHIFTER_LOG_VERBOSITY` 或 `--log-verbosity`。
    Env,
    /// 旧 `HIFISHIFTER_LOG` 兼容映射。
    EnvLegacy,
    /// App 的 `--log-verbosity=<...>` 显式覆盖。
    Cli,
}

impl PolicySource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Persisted => "persisted",
            Self::Env => "env",
            Self::EnvLegacy => "env",
            Self::Cli => "cli",
        }
    }
    /// 启动参数/环境变量覆盖时，帮助菜单的四档必须禁用（改了不会生效）。
    pub fn overrides_ui(self) -> bool {
        matches!(self, Self::Env | Self::EnvLegacy | Self::Cli)
    }
}

/// 一条档位变更（内存环形，导出到诊断包的 `policy_changes.jsonl`）。
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PolicyChange {
    pub at: String,
    pub from: LogVerbosity,
    pub to: LogVerbosity,
    pub source: String,
}

/// 进程级当前档位。`0..3` 与 `LogVerbosity::ALL` 的下标一致。
static CURRENT: AtomicU8 = AtomicU8::new(LogVerbosity::Low as u8);
static SOURCE: Mutex<PolicySource> = Mutex::new(PolicySource::Default);
static OVERRIDE_HINT: OnceLock<String> = OnceLock::new();
static HOST_LOGGER_OWNED: AtomicBool = AtomicBool::new(false);
static LOGGING_HEALTHY: AtomicBool = AtomicBool::new(true);
static LAST_ERROR: Mutex<String> = Mutex::new(String::new());
static SESSION_ID: OnceLock<String> = OnceLock::new();
static CHANGES: Mutex<Vec<PolicyChange>> = Mutex::new(Vec::new());
const CHANGE_CAPACITY: usize = 64;

fn lock_or_recover<'a, T>(mutex: &'a Mutex<T>) -> std::sync::MutexGuard<'a, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// 旧环境变量 → 四档的兼容映射（**唯一**实现）。
pub fn legacy_env_verbosity(value: &str) -> Option<LogVerbosity> {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" => Some(LogVerbosity::Off),
        "error" | "warn" => Some(LogVerbosity::Low),
        "info" => Some(LogVerbosity::Medium),
        "debug" | "trace" => Some(LogVerbosity::High),
        _ => None,
    }
}

/// 从 `--log-verbosity=<x>` / `--log-verbosity x` 解析显式覆盖。
pub fn cli_verbosity(args: &[String]) -> Option<LogVerbosity> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if let Some(value) = arg.strip_prefix("--log-verbosity=") {
            return LogVerbosity::parse(value);
        }
        if arg == "--log-verbosity" {
            return iter.next().and_then(|value| LogVerbosity::parse(value));
        }
    }
    None
}

/// 优先级链的**纯**实现，便于单测注入任意组合（不触碰进程级环境变量）。
///
/// `① CLI/probe 显式覆盖 > ② 旧 HIFISHIFTER_LOG > ③ 持久化设置 > ④ 默认 Low`。
pub fn resolve_initial(
    cli: Option<LogVerbosity>,
    new_env: Option<&str>,
    legacy_env: Option<&str>,
    persisted: Option<LogVerbosity>,
) -> (LogVerbosity, PolicySource) {
    if let Some(level) = cli {
        return (level, PolicySource::Cli);
    }
    if let Some(value) = new_env {
        if let Some(level) = LogVerbosity::parse(value) {
            return (level, PolicySource::Env);
        }
    }
    if let Some(value) = legacy_env {
        if let Some(level) = legacy_env_verbosity(value) {
            return (level, PolicySource::EnvLegacy);
        }
    }
    match persisted {
        Some(level) => (level, PolicySource::Persisted),
        None => (DEFAULT_VERBOSITY, PolicySource::Default),
    }
}

fn env_value(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

/// 启动期装载策略：读持久化文件 → 应用优先级链 → 安装 `log::set_max_level`。
///
/// 两个宿主都在日志器安装之前调用它；`load_initial` 是幂等的（`INIT` 只跑一次），
/// 插件与 CLAP 外壳先后调用不会重复写变更历史。
pub fn load_initial(args: &[String]) -> LogPolicy {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let persisted = crate::log_settings::load_or_default().verbosity;
        let (level, source) = resolve_initial(
            cli_verbosity(args),
            env_value(VERBOSITY_ENV).as_deref(),
            env_value(LEGACY_LOG_ENV).as_deref(),
            Some(persisted),
        );
        *lock_or_recover(&SOURCE) = source;
        if source.overrides_ui() {
            let _ = OVERRIDE_HINT.set(format!(
                "startup override active: {source:?} -> {level:?}; clear {} / {} / --log-verbosity to use the Help menu",
                VERBOSITY_ENV,
                LEGACY_LOG_ENV,
            ));
            if source == PolicySource::EnvLegacy {
                eprintln!("hifishifter: HIFISHIFTER_LOG overrides the UI log verbosity (use Help > Log verbosity after removing it)");
            }
        }
        CURRENT.store(level as u8, Ordering::SeqCst);
        // session id 只生成一次；会话头、会话尾、诊断包都读它。
        let id = crate::logfile::short_session_id();
        let _ = SESSION_ID.set(id);
        apply_to_log_facade(level);
        record_change(level, level, "startup");
    });
    current().policy()
}

/// 把当前档位写进 `log` 门面。**唯一**允许调用 `set_max_level` 的入口。
fn apply_to_log_facade(level: LogVerbosity) {
    log::set_max_level(level.level_filter());
}

/// 运行时热切换（帮助菜单调用）。覆盖态下 UI 会禁用四档；后端仍显式拒绝，
/// 避免卫星窗口或旧版本前端绕过。返回 `false` 表示持久化失败。
pub fn apply(verbosity: LogVerbosity, source: PolicySource) -> bool {
    if lock_or_recover(&SOURCE).overrides_ui() {
        return false;
    }
    if crate::log_settings::save(verbosity).is_err() {
        return false;
    }
    set_current(verbosity, source);
    true
}

/// 只改内存态（测试、探针、未来的临时提级共用；不写盘）。
pub fn set_current(verbosity: LogVerbosity, source: PolicySource) {
    let previous = current();
    CURRENT.store(verbosity as u8, Ordering::SeqCst);
    *lock_or_recover(&SOURCE) = source;
    apply_to_log_facade(verbosity);
    // 在新档生效**之后**写自证行：低档也能看见"用户把档位调走了"，
    // 否则改成「无」之后日志里没有任何东西能解释随后的沉默。
    if previous != verbosity {
        log::warn!(
            "[log] verbosity {}→{} source={} at={}",
            previous.as_str(),
            verbosity.as_str(),
            source.as_str(),
            crate::logfile::now_timestamp()
        );
        record_change(previous, verbosity, source.as_str());
    }
}

fn record_change(from: LogVerbosity, to: LogVerbosity, source: &str) {
    let mut changes = lock_or_recover(&CHANGES);
    if changes.len() >= CHANGE_CAPACITY {
        changes.remove(0);
    }
    changes.push(PolicyChange {
        at: crate::logfile::now_timestamp(),
        from,
        to,
        source: source.to_owned(),
    });
}

pub fn current() -> LogVerbosity {
    LogVerbosity::ALL
        .into_iter()
        .find(|level| *level as u8 == CURRENT.load(Ordering::SeqCst))
        .unwrap_or(DEFAULT_VERBOSITY)
}

pub fn source() -> PolicySource {
    *lock_or_recover(&SOURCE)
}

pub fn overridden() -> bool {
    source().overrides_ui()
}

pub fn override_hint() -> Option<&'static str> {
    OVERRIDE_HINT.get().map(String::as_str)
}

pub fn session_id() -> &'static str {
    SESSION_ID.get().map(String::as_str).unwrap_or("00000000")
}

pub fn policy() -> LogPolicy {
    current().policy()
}

pub fn file_enabled() -> bool {
    policy().file_enabled
}

pub fn probes_enabled() -> bool {
    policy().probes && log::max_level() >= log::LevelFilter::Debug
}

/// 诊断包 `log_policy.json` 的内容：当前策略 + 来源 + 会话 + 健康度。
pub fn diagnostics_json() -> serde_json::Value {
    let policy = self::policy();
    serde_json::json!({
        "policyVersion": POLICY_VERSION,
        "verbosity": policy.verbosity.as_str(),
        "source": source().as_str(),
        "overridden": overridden(),
        "overrideHint": override_hint(),
        "sessionId": session_id(),
        "levelFilter": format!("{:?}", policy.level_filter),
        "fileEnabled": policy.file_enabled,
        "stderrTee": policy.stderr_tee,
        "hostSink": policy.host_sink,
        "probes": policy.probes,
        "frontendAccept": {
            "error": policy.frontend_accept.error,
            "warn": policy.frontend_accept.warn,
            "info": policy.frontend_accept.info,
            "debug": policy.frontend_accept.debug,
        },
        "detailCapChars": policy.detail_cap_chars,
        "maxSingleLineChars": policy.max_single_line_chars,
        "maxFileBytes": policy.max_file_bytes,
        "maxFiles": policy.max_files,
        "loggingHealthy": logging_healthy(),
        "hostLoggerOwned": host_logger_owned(),
        "lastError": last_error(),
    })
}

/// 诊断包 `policy_changes.jsonl` 的内容（内存环形最近 64 条）。
pub fn diagnostics_changes_jsonl() -> String {
    change_log()
        .into_iter()
        .map(|change| {
            serde_json::json!({
                "at": change.at,
                "from": change.from.as_str(),
                "to": change.to.as_str(),
                "source": change.source,
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn change_log() -> Vec<PolicyChange> {
    lock_or_recover(&CHANGES).clone()
}

// ── 日志健康度（G5 / G14 / G15）────────────────────────────────────────────

/// 插件 `log::set_logger` 已被宿主抢先占用：文件日志不可用，必须明确回报。
pub fn mark_host_logger_owned() {
    HOST_LOGGER_OWNED.store(true, Ordering::SeqCst);
    mark_unhealthy("host process already installed a global logger; file logging unavailable");
}

pub fn host_logger_owned() -> bool {
    HOST_LOGGER_OWNED.load(Ordering::SeqCst)
}

pub fn mark_unhealthy(reason: &str) {
    LOGGING_HEALTHY.store(false, Ordering::SeqCst);
    *lock_or_recover(&LAST_ERROR) = crate::logfile::log_field(reason, 512);
}

pub fn logging_healthy() -> bool {
    LOGGING_HEALTHY.load(Ordering::SeqCst)
}

pub fn last_error() -> Option<String> {
    let error = lock_or_recover(&LAST_ERROR).clone();
    (!error.is_empty()).then_some(error)
}

/// 仅测试用：把进程级策略状态复位（依赖 crate 的单测也要能调它）。
pub fn reset_for_tests() {
    CURRENT.store(DEFAULT_VERBOSITY as u8, Ordering::SeqCst);
    *lock_or_recover(&SOURCE) = PolicySource::Default;
    HOST_LOGGER_OWNED.store(false, Ordering::SeqCst);
    LOGGING_HEALTHY.store(true, Ordering::SeqCst);
    lock_or_recover(&LAST_ERROR).clear();
    lock_or_recover(&CHANGES).clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_verbosity_is_low() {
        assert_eq!(DEFAULT_VERBOSITY, LogVerbosity::Low);
        assert_eq!(LogVerbosity::default(), LogVerbosity::Low);
        assert_eq!(LogVerbosity::Low.as_str(), "low");
    }

    #[test]
    fn four_verbosities_map_to_expected_level_filters() {
        assert_eq!(LogVerbosity::Off.level_filter(), log::LevelFilter::Off);
        assert_eq!(LogVerbosity::Low.level_filter(), log::LevelFilter::Warn);
        assert_eq!(LogVerbosity::Medium.level_filter(), log::LevelFilter::Info);
        assert_eq!(LogVerbosity::High.level_filter(), log::LevelFilter::Debug);
    }

    #[test]
    fn off_policy_disables_every_output_channel() {
        let policy = LogVerbosity::Off.policy();
        assert!(!policy.file_enabled);
        assert!(!policy.stderr_tee);
        assert!(!policy.host_sink);
        assert!(!policy.probes);
        assert!(!policy.frontend_accept.error);
        assert!(!policy.frontend_accept.warn);
        assert!(!policy.frontend_accept.info);
        assert!(!policy.frontend_accept.debug);
        assert_eq!(policy.max_file_bytes, 0);
        assert_eq!(policy.max_files, 0);
    }

    #[test]
    fn frontend_accept_follows_the_four_level_matrix() {
        assert!(LogVerbosity::Low.policy().frontend_accept.error);
        assert!(!LogVerbosity::Low.policy().frontend_accept.warn);
        assert!(LogVerbosity::Medium.policy().frontend_accept.warn);
        assert!(!LogVerbosity::Medium.policy().frontend_accept.debug);
        assert!(LogVerbosity::High.policy().frontend_accept.debug);
        assert!(LogVerbosity::High.policy().probes);
        assert!(!LogVerbosity::Medium.policy().probes);
    }

    #[test]
    fn legacy_env_log_maps_to_the_same_verbosity() {
        assert_eq!(legacy_env_verbosity("off"), Some(LogVerbosity::Off));
        assert_eq!(legacy_env_verbosity("error"), Some(LogVerbosity::Low));
        assert_eq!(legacy_env_verbosity("WARN"), Some(LogVerbosity::Low));
        assert_eq!(legacy_env_verbosity("info"), Some(LogVerbosity::Medium));
        assert_eq!(legacy_env_verbosity("debug"), Some(LogVerbosity::High));
        assert_eq!(legacy_env_verbosity("trace"), Some(LogVerbosity::High));
        assert_eq!(legacy_env_verbosity("nonsense"), None);
    }

    #[test]
    fn priority_cli_beats_env_beats_persisted_beats_default() {
        assert_eq!(
            resolve_initial(None, None, None, None),
            (LogVerbosity::Low, PolicySource::Default)
        );
        assert_eq!(
            resolve_initial(None, None, None, Some(LogVerbosity::High)),
            (LogVerbosity::High, PolicySource::Persisted)
        );
        assert_eq!(
            resolve_initial(None, None, Some("info"), Some(LogVerbosity::Off)),
            (LogVerbosity::Medium, PolicySource::EnvLegacy)
        );
        assert_eq!(
            resolve_initial(None, Some("off"), Some("debug"), Some(LogVerbosity::High)),
            (LogVerbosity::Off, PolicySource::Env)
        );
        assert_eq!(
            resolve_initial(
                Some(LogVerbosity::High),
                Some("off"),
                Some("info"),
                Some(LogVerbosity::Off)
            ),
            (LogVerbosity::High, PolicySource::Cli)
        );
    }

    #[test]
    fn invalid_persisted_value_falls_back_to_low() {
        assert_eq!(LogVerbosity::parse("loud"), None);
        assert_eq!(LogVerbosity::parse("  HIGH "), Some(LogVerbosity::High));
        assert_eq!(LogVerbosity::parse(""), None);
        // 非法值由 log_settings::load_or_default 转成 Low；见该模块单测。
        assert_eq!(resolve_initial(None, None, None, None).0, LogVerbosity::Low);
    }

    #[test]
    fn the_cli_flag_is_parsed_in_both_forms() {
        let equals = vec!["HiFiShifter".to_owned(), "--log-verbosity=high".to_owned()];
        let separate = vec![
            "HiFiShifter".to_owned(),
            "--log-verbosity".to_owned(),
            "off".to_owned(),
        ];
        assert_eq!(cli_verbosity(&equals), Some(LogVerbosity::High));
        assert_eq!(cli_verbosity(&separate), Some(LogVerbosity::Off));
        assert_eq!(cli_verbosity(&[]), None);
    }

    #[test]
    fn apply_records_a_change_and_switches_the_log_level() {
        // 只测纯状态，不触碰进程级 log::set_max_level 的其它用例（并行测试会互踩）。
        reset_for_tests();
        set_current(LogVerbosity::Off, PolicySource::Persisted);
        assert_eq!(current(), LogVerbosity::Off);
        assert_eq!(source(), PolicySource::Persisted);
        set_current(LogVerbosity::High, PolicySource::Persisted);
        let changes = change_log();
        let last = changes.last().expect("a change was recorded");
        assert_eq!(last.from, LogVerbosity::Off);
        assert_eq!(last.to, LogVerbosity::High);
        assert_eq!(current(), LogVerbosity::High);
        reset_for_tests();
    }

    #[test]
    fn an_explicit_override_disables_the_menu_options() {
        assert!(PolicySource::Cli.overrides_ui());
        assert!(PolicySource::Env.overrides_ui());
        assert!(PolicySource::EnvLegacy.overrides_ui());
        assert!(!PolicySource::Persisted.overrides_ui());
        assert!(!PolicySource::Default.overrides_ui());
    }

    #[test]
    fn health_flags_are_reported_instead_of_swallowed() {
        reset_for_tests();
        assert!(logging_healthy());
        mark_host_logger_owned();
        assert!(host_logger_owned());
        assert!(!logging_healthy());
        assert!(last_error().unwrap_or_default().contains("global logger"));
        reset_for_tests();
    }
}
