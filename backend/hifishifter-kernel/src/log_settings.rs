//! `log_settings.json` 的读写：独立于 `UiSettings` 的轻量日志策略文件。
//!
//! 【为什么不塞进 app_config.json】日志在 `main.rs:40` / DLL 入口就初始化，而
//! `UiSettings` 要等 Tauri setup（`lib.rs:472`）与前端 `get_ui_settings` 才到。
//! 把"无日志"这个明确选择放进巨型配置里，用户每次启动都会被先写一段日志。
//! 独立文件还避免 `persistUiSettings` 的 70+ 字段全量序列化把档位写回默认。
//!
//! 格式（`schemaVersion=1`，未知字段忽略）：
//! ```json
//! {"schemaVersion":1,"verbosity":"low","updatedAt":"...","updatedBy":"ui",
//!  "temporaryUntil":null,"lastSessionId":null}
//! ```
//! 非法值 / 解析失败 / 未来版本一律回落 `Low`，并写一行 stderr 自证（日志系统
//! 自身失效的最后通道，不属常规日志）。

use crate::log_policy::{LogVerbosity, DEFAULT_VERBOSITY, POLICY_VERSION};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// 文件名常量：**只在这里定义一次**（静态门禁会检查）。
pub const LOG_SETTINGS_FILE_NAME: &str = "log_settings.json";

/// 进程内写锁；跨进程由 `log_settings.json.lock` 的排他文件锁串行化。
static WRITE_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LogSettingsFile {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub verbosity: LogVerbosity,
    #[serde(default)]
    pub updated_at: Option<String>,
    #[serde(default)]
    pub updated_by: Option<String>,
    /// 预留给未来的「临时提级自动回落」；本批不启用（见方案 §12 开放问题 6）。
    #[serde(default)]
    pub temporary_until: Option<String>,
    #[serde(default)]
    pub last_session_id: Option<String>,
}

fn default_schema_version() -> u32 {
    POLICY_VERSION
}

impl Default for LogSettingsFile {
    fn default() -> Self {
        Self {
            schema_version: POLICY_VERSION,
            verbosity: DEFAULT_VERBOSITY,
            updated_at: None,
            updated_by: None,
            temporary_until: None,
            last_session_id: None,
        }
    }
}

/// 设置文件路径：与 `app_config.json` 同一个用户配置目录（两个宿主共用）。
///
/// 解析不出用户目录时退回系统临时目录：宁可策略不完美，也不能让插件起不来。
pub fn settings_path() -> PathBuf {
    base_dir().join(LOG_SETTINGS_FILE_NAME)
}

fn base_dir() -> PathBuf {
    crate::app_paths::config_dir().unwrap_or_else(|| {
        std::env::temp_dir()
            .join(crate::app_paths::PRODUCT_DIR.to_lowercase())
            .join("config")
    })
}

/// 纯解析：便于单测覆盖缺文件 / 损坏 / 非法值 / 未来版本，不触碰真实用户目录。
pub fn parse(contents: &str) -> LogSettingsFile {
    match serde_json::from_str::<LogSettingsFile>(contents) {
        Ok(file) if file.schema_version == POLICY_VERSION => {
            // serde 对未知枚举值直接报错；落到下方统一回退 Low。
            file
        }
        Ok(_) => {
            eprintln!(
                "hifishifter: {LOG_SETTINGS_FILE_NAME} has an unsupported schemaVersion; using {DEFAULT_VERBOSITY:?}"
            );
            LogSettingsFile::default()
        }
        Err(error) => {
            eprintln!(
                "hifishifter: {LOG_SETTINGS_FILE_NAME} unreadable ({error}); using {DEFAULT_VERBOSITY:?}"
            );
            LogSettingsFile::default()
        }
    }
}

pub fn load_from(dir: &Path) -> LogSettingsFile {
    let path = dir.join(LOG_SETTINGS_FILE_NAME);
    match std::fs::read_to_string(&path) {
        Ok(contents) => parse(&contents),
        Err(_) => LogSettingsFile::default(),
    }
}

/// 缺文件 / 损坏 / 非法值都得到「低」，绝不静默沿用旧的错误值。
pub fn load_or_default() -> LogSettingsFile {
    load_from(&base_dir())
}

/// 取进程内写锁 + 跨进程排他文件锁。
///
/// 独立 App 与 REAPER 里的插件可能同时改档；两层锁缺一不可（与
/// `config.rs::update_config` 同一纪律）。锁文件本身永不被读写。
fn with_write_lock<F, T>(dir: &Path, f: F) -> T
where
    F: FnOnce() -> T,
{
    let _guard = WRITE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _file_lock = acquire_lock(dir);
    f()
}

fn acquire_lock(dir: &Path) -> Option<std::fs::File> {
    let _ = std::fs::create_dir_all(dir);
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join(format!("{LOG_SETTINGS_FILE_NAME}.lock")))
        .ok()?;
    file.lock().ok().map(|()| file)
}

/// 写盘（原子替换）。`save` 是 UI 改档的唯一下盘入口；返回结构化错误供命令层回报。
pub fn save(verbosity: LogVerbosity) -> Result<LogSettingsFile, String> {
    save_to(&base_dir(), verbosity)
}

pub fn save_to(dir: &Path, verbosity: LogVerbosity) -> Result<LogSettingsFile, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create config dir failed: {e}"))?;
    with_write_lock(dir, || write_settings(dir, verbosity))
}

fn write_settings(dir: &Path, verbosity: LogVerbosity) -> Result<LogSettingsFile, String> {
    let current = load_from(dir);
    let mut file = current.clone();
    file.schema_version = POLICY_VERSION;
    file.verbosity = verbosity;
    file.updated_at = Some(crate::logfile::now_timestamp());
    file.updated_by = Some("ui".to_owned());
    file.last_session_id = Some(crate::log_policy::session_id().to_owned());
    // 临时提级本批不启用，但预留字段不得被旧值污染。
    file.temporary_until = None;
    let data = serde_json::to_string_pretty(&file)
        .map_err(|e| format!("serialize log settings failed: {e}"))?;
    let target = dir.join(LOG_SETTINGS_FILE_NAME);
    let tmp = dir.join(format!("{LOG_SETTINGS_FILE_NAME}.tmp"));
    std::fs::write(&tmp, &data).map_err(|e| format!("write log settings failed: {e}"))?;
    match crate::util::replace_file_atomically(&tmp, &target) {
        Ok(()) => Ok(file),
        Err(error) => {
            let _ = std::fs::remove_file(&tmp);
            Err(format!("replace log settings failed: {error}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hfs-log-settings-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_missing_file_yields_the_low_default() {
        let dir = temp("missing");
        assert_eq!(load_from(&dir).verbosity, LogVerbosity::Low);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn damaged_and_invalid_values_fall_back_to_low() {
        assert_eq!(parse("{not json").verbosity, LogVerbosity::Low);
        assert_eq!(
            parse(r#"{"schemaVersion":1,"verbosity":"loud"}"#).verbosity,
            LogVerbosity::Low
        );
        assert_eq!(
            parse(r#"{"schemaVersion":1,"verbosity":"off"}"#).verbosity,
            LogVerbosity::Off
        );
        assert_eq!(
            parse(r#"{"schemaVersion":99,"verbosity":"high"}"#).verbosity,
            LogVerbosity::Low
        );
    }

    #[test]
    fn unknown_fields_are_ignored_for_forward_compatibility() {
        let parsed =
            parse(r#"{"schemaVersion":1,"verbosity":"medium","futureFlag":true,"updatedAt":"x"}"#);
        assert_eq!(parsed.verbosity, LogVerbosity::Medium);
        assert_eq!(parsed.updated_at.as_deref(), Some("x"));
    }

    #[test]
    fn saving_atomically_replaces_the_file_and_is_readable_immediately() {
        let dir = temp("atomic");
        save_to(&dir, LogVerbosity::Off).unwrap();
        assert_eq!(load_from(&dir).verbosity, LogVerbosity::Off);
        save_to(&dir, LogVerbosity::High).unwrap();
        let file = load_from(&dir);
        assert_eq!(file.verbosity, LogVerbosity::High);
        assert_eq!(file.updated_by.as_deref(), Some("ui"));
        assert_eq!(file.temporary_until, None);
        assert!(
            !dir.join(format!("{LOG_SETTINGS_FILE_NAME}.tmp")).exists(),
            "临时文件必须已替换走"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_independent_directories_do_not_overwrite_each_other() {
        let first = temp("first");
        let second = temp("second");
        save_to(&first, LogVerbosity::Off).unwrap();
        save_to(&second, LogVerbosity::High).unwrap();
        assert_eq!(load_from(&first).verbosity, LogVerbosity::Off);
        assert_eq!(load_from(&second).verbosity, LogVerbosity::High);
        let _ = std::fs::remove_dir_all(&first);
        let _ = std::fs::remove_dir_all(&second);
    }
}
