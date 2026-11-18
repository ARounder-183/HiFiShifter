//! Local MCP instance discovery registry.
//!
//! Multiple HiFiShifter windows may run concurrently on one machine. Each active
//! MCP server writes an atomic `<registry_dir>/<instance_id>.json` lease so local
//! agents and the `--mcp-stdio` bridge can discover open projects without port
//! guessing or clobbering `LOCAL_PORT` silently.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InstanceRecord {
    pub instance_id: String,
    pub pid: u32,
    pub endpoint: String,
    pub port: u16,
    pub project_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_path: Option<String>,
    pub updated_at_unix_ms: u64,
}

pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn record_path(registry_dir: &Path, instance_id: &str) -> PathBuf {
    let safe: String = instance_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    registry_dir.join(format!("{safe}.json"))
}

pub fn write_instance_record(
    registry_dir: &Path,
    record: &InstanceRecord,
) -> Result<PathBuf, String> {
    std::fs::create_dir_all(registry_dir).map_err(|e| e.to_string())?;
    let target = record_path(registry_dir, &record.instance_id);
    let tmp = registry_dir.join(format!(".{}.{}.tmp", record.instance_id, record.pid));
    let bytes = serde_json::to_vec_pretty(record).map_err(|e| e.to_string())?;
    std::fs::write(&tmp, bytes).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &target).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        e.to_string()
    })?;
    Ok(target)
}

pub fn remove_instance_record(registry_dir: &Path, instance_id: &str) {
    let _ = std::fs::remove_file(record_path(registry_dir, instance_id));
}

pub fn list_instance_records(registry_dir: &Path) -> Vec<InstanceRecord> {
    let Ok(entries) = std::fs::read_dir(registry_dir) else {
        return Vec::new();
    };
    let mut records = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        if let Ok(bytes) = std::fs::read(&path) {
            if let Ok(rec) = serde_json::from_slice::<InstanceRecord>(&bytes) {
                records.push(rec);
            }
        }
    }
    records.sort_by(|a, b| b.updated_at_unix_ms.cmp(&a.updated_at_unix_ms));
    records
}

/// Resolve the best local MCP endpoint from explicit URL, project path match,
/// newest registered instance, or the canonical loopback default.
pub fn resolve_endpoint(
    registry_dir: Option<&Path>,
    explicit_url: Option<&str>,
    project_filter: Option<&str>,
) -> String {
    if let Some(url) = explicit_url {
        if !url.trim().is_empty() {
            return url.trim().to_string();
        }
    }
    if let Some(dir) = registry_dir {
        let records = list_instance_records(dir);
        if let Some(target_project) = project_filter {
            if let Some(hit) = records.iter().find(|r| {
                r.project_path
                    .as_deref()
                    .is_some_and(|p| p == target_project)
                    || r.project_name == target_project
            }) {
                return hit.endpoint.clone();
            }
        }
        if let Some(newest) = records.first() {
            return newest.endpoint.clone();
        }
    }
    format!("http://127.0.0.1:{}/mcp", crate::server::LOCAL_PORT)
}
