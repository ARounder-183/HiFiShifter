//! 日志详细程度命令（独立 App 侧）。
//!
//! 【为什么不放在 `commands/diagnostics.rs`】帮助菜单的四档单选项是用户设置，
//! 不是诊断导出；实现只转发内核策略层，命令层不含任何"判档"逻辑。
//! 插件侧 `editor/commands.rs` 走同形命令，前端无需知道宿主形态。
//!
//! `get_log_settings` 在「无」档下**仍然可用**（只读命令，不写日志），
//! 否则用户无法把档位改回来。

use hifishifter_kernel::log_policy::{self, LogVerbosity, PolicySource};
use serde_json::{json, Value};

/// 把当前策略、落点、健康度与只读文件清单汇总成前端唯一的 `LogSettingsView`。
///
/// 数据结构在 `frontend/src/services/api/logSettings.ts` 有逐字段同形的类型；
/// 两个宿主都调同一个 `view()`，避免 App 与插件各拼一份 JSON。
pub fn view(override_error: Option<&str>) -> Value {
    let policy = log_policy::policy();
    let (files, total_bytes) = crate::logging::log_file_stats();
    let log_file = crate::logging::log_file().map(|p| p.to_string_lossy().into_owned());
    let log_dir = crate::logging::log_dir().map(|p| p.to_string_lossy().into_owned());
    let overridden = log_policy::overridden();
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
        "overridden": overridden,
        "overrideHint": log_policy::override_hint(),
        "fileEnabled": policy.file_enabled,
        "fileWritable": policy.file_enabled && crate::logging::log_file().is_some(),
        "loggingHealthy": log_policy::logging_healthy(),
        "hostLoggerOwned": log_policy::host_logger_owned(),
        "lastError": log_policy::last_error().or_else(|| override_error.map(str::to_owned)),
        "logDir": log_dir,
        "logFile": log_file,
        "sessionId": log_policy::session_id(),
        "files": files.into_iter().map(|(name, bytes, modified_at)| json!({
            "name": name,
            "bytes": bytes,
            "modifiedAt": modified_at,
        })).collect::<Vec<_>>(),
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

/// 只读：帮助菜单挂载时取一次，之后靠 `hs:log-settings-changed` 同步。
pub(super) fn get_log_settings() -> Value {
    // 诊断包导出前必须把 tee 的缓冲落盘，否则"刚发生的错误"不在文件里。
    let _ = crate::logging::flush_logs();
    view(None)
}

/// 写入：解析档位 → 拒绝启动覆盖 → 持久化 + 热切换 → 自证行 → 广播事件。
pub(super) fn set_log_settings(verbosity: String) -> Value {
    let Some(level) = LogVerbosity::parse(&verbosity) else {
        return view(Some(&format!("invalid log verbosity: {verbosity}")));
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
    emit_changed(&payload);
    payload
}

/// 主窗口与卫星窗口共用同一份策略：广播事件让两边的 `frontendLog` 同步。
fn emit_changed(payload: &Value) {
    if let Some(handle) = crate::app_events::app_handle() {
        use tauri::Emitter;
        let _ = handle.emit("hs:log-settings-changed", payload.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_verbosity_is_a_structured_failure() {
        let value = view(Some("invalid log verbosity: loud"));
        assert_eq!(value["ok"], false);
        assert_eq!(value["error"], "invalid log verbosity: loud");
    }

    #[test]
    fn the_view_never_omits_the_status_fields() {
        let value = view(None);
        for key in [
            "verbosity",
            "persistedVerbosity",
            "source",
            "overridden",
            "fileEnabled",
            "loggingHealthy",
            "hostLoggerOwned",
            "policyVersion",
            "recentChanges",
            "files",
            "totalBytes",
        ] {
            assert!(!value[key].is_null(), "missing {key}");
        }
    }
}
