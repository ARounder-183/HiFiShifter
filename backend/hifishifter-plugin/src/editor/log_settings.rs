//! 插件侧日志详细程度命令的**唯一**实现。
//!
//! 独立 App 的 `commands/log_settings.rs` 与这里逐字段同形，但插件跑在宿主进程里、
//! 没有 Tauri `AppHandle`：读侧用 `diagnostics` 的落点与健康度，写侧通过内核事件
//! 出口广播 `hs:log-settings-changed`，让当前编辑器实例与前端 `frontendLog` 同步。
//!
//! `get_log_settings` 必须能在「无」档下工作（只读，不写日志），否则用户无法改回来。

use hifishifter_kernel::log_policy::{self, LogVerbosity, PolicySource};
use serde_json::{json, Value};

/// 插件侧的 `LogSettingsView`：与 App 侧前端类型同形。
pub fn view(override_error: Option<&str>) -> Value {
    let policy = log_policy::policy();
    let log_file = crate::diagnostics::log_file();
    let log_dir = crate::diagnostics::log_directory();
    let (files, total_bytes) = file_stats(&log_file, policy.max_files);
    let changes: Vec<Value> = log_policy::change_log()
        .into_iter()
        .map(|change| {
            json!({
                "at": change.at,
                "from": change.from.as_str(),
                "to": change.to.as_str(),
                "source": change.source,
            })
        })
        .collect();
    let mut out = json!({
        "ok": override_error.is_none(),
        "verbosity": policy.verbosity.as_str(),
        "persistedVerbosity": hifishifter_kernel::log_settings::load_or_default().verbosity.as_str(),
        "source": log_policy::source().as_str(),
        "overridden": log_policy::overridden(),
        "overrideHint": log_policy::override_hint(),
        "fileEnabled": policy.file_enabled,
        "fileWritable": policy.file_enabled && !log_policy::host_logger_owned(),
        "loggingHealthy": log_policy::logging_healthy(),
        "hostLoggerOwned": log_policy::host_logger_owned(),
        "lastError": log_policy::last_error().or_else(|| override_error.map(str::to_owned)),
        "logDir": log_dir.to_string_lossy(),
        "logFile": log_file.to_string_lossy(),
        "sessionId": log_policy::session_id(),
        "files": files,
        "totalBytes": total_bytes,
        "maxFileBytes": policy.max_file_bytes,
        "maxFiles": policy.max_files,
        "policyVersion": hifishifter_kernel::log_policy::POLICY_VERSION,
        "recentChanges": changes,
    });
    if let Some(error) = override_error {
        out["error"] = json!(error);
    }
    out
}

fn file_stats(log_file: &std::path::Path, max_files: usize) -> (Vec<Value>, u64) {
    let mut files = Vec::new();
    let mut total = 0_u64;
    for path in hifishifter_kernel::logfile::rotated_files_with_limit(log_file, max_files) {
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let modified = meta
            .modified()
            .ok()
            .map(|time| {
                let dt: chrono::DateTime<chrono::Local> = time.into();
                dt.format(hifishifter_kernel::logfile::TIMESTAMP_FORMAT)
                    .to_string()
            })
            .unwrap_or_default();
        total = total.saturating_add(meta.len());
        files.push(json!({ "name": name, "bytes": meta.len(), "modifiedAt": modified }));
    }
    (files, total)
}

pub fn get_log_settings() -> Value {
    crate::diagnostics::flush_writer_for_diagnostics();
    view(None)
}

/// 与 App 侧命令同形：参数是前端 `{verbosity}` 载荷（JSON 值）。
pub fn set_log_settings(input: &Value) -> Value {
    let Some(raw) = input["verbosity"].as_str() else {
        return view(Some("verbosity missing"));
    };
    let Some(level) = LogVerbosity::parse(raw) else {
        return view(Some(&format!("invalid log verbosity: {raw}")));
    };
    if log_policy::overridden() {
        return view(Some(
            "log verbosity is controlled by a startup argument or environment variable",
        ));
    }
    if !log_policy::apply(level, PolicySource::Persisted) {
        return view(Some("failed to save log settings"));
    }
    let payload = view(None);
    // 内核事件出口按文档私有 id 分流；策略变更是所有会话共享的进程级事实，
    // 因此借用 `emit_ui_event` 的既有广播路径（前端 hostEvents.listen 同形）。
    hifishifter_kernel::events::events().emit("hs:log-settings-changed", payload.clone());
    payload
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_invalid_verbosity_is_reported_as_a_structured_failure() {
        let value = set_log_settings(&json!({"verbosity": "loud"}));
        assert_eq!(value["ok"], false);
        assert!(value["error"]
            .as_str()
            .unwrap_or_default()
            .contains("invalid"));
    }

    #[test]
    fn the_view_contains_the_health_fields_the_help_menu_reads() {
        let value = view(None);
        for key in [
            "verbosity",
            "overridden",
            "loggingHealthy",
            "hostLoggerOwned",
            "files",
        ] {
            assert!(!value[key].is_null(), "missing {key}");
        }
    }
}
