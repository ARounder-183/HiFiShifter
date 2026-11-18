//! App-owned MCP lifecycle, discovery lease registration, and GUI view bridge.
//! Disabled by default unless `--enable-mcp` is passed; never loads a second project.

mod draft;
mod editor;
mod host;
mod notebook;
mod preview;
mod scope;

use hifishifter_mcp::{
    discovery::{self, InstanceRecord},
    EditorBackend, EditorRequest,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    sync::{atomic::AtomicU64, Arc, Mutex},
};
use tauri::{Emitter, Manager};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AuditEntry {
    pub sequence: u64,
    pub timestamp_unix_ms: u64,
    pub idempotency_key: String,
    pub label: Option<String>,
    pub operation_count: usize,
    pub revision: String,
    pub created_track_ids: Vec<String>,
    pub created_clip_ids: Vec<String>,
}

struct Running {
    endpoint: String,
    port: u16,
    cancel: CancellationToken,
}

pub(crate) struct McpControl {
    running: Mutex<Option<Running>>,
    pub instance_id: String,
    pub(super) ledger: Mutex<std::collections::VecDeque<(String, String, Value)>>,
    pub(super) audit: Mutex<std::collections::VecDeque<AuditEntry>>,
    pub(super) next_audit_seq: AtomicU64,
    pub(super) view_revision: Mutex<Option<String>>,
    view_requests: Mutex<std::collections::HashMap<String, std::sync::mpsc::Sender<Value>>>,
}

impl Default for McpControl {
    fn default() -> Self {
        Self {
            running: Mutex::new(None),
            instance_id: uuid::Uuid::new_v4().to_string(),
            ledger: Mutex::new(Default::default()),
            audit: Mutex::new(Default::default()),
            next_audit_seq: AtomicU64::new(0),
            view_revision: Mutex::new(None),
            view_requests: Mutex::new(Default::default()),
        }
    }
}

pub(super) fn discovery_dir() -> Option<PathBuf> {
    hifishifter_kernel::app_paths::local_data_dir().map(|root| root.join("mcp").join("instances"))
}

pub(super) fn sync_discovery_record(state: &crate::state::AppState, control: &McpControl) {
    let Some(dir) = discovery_dir() else {
        return;
    };
    let (endpoint, port) = {
        let guard = control.running.lock().unwrap_or_else(|e| e.into_inner());
        let Some(r) = guard.as_ref() else {
            return;
        };
        (r.endpoint.clone(), r.port)
    };
    let (project_name, project_path) = {
        let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        (p.name.clone(), p.path.clone())
    };
    let record = InstanceRecord {
        instance_id: control.instance_id.clone(),
        pid: std::process::id(),
        endpoint,
        port,
        project_name,
        project_path,
        updated_at_unix_ms: discovery::now_unix_ms(),
    };
    let _ = discovery::write_instance_record(&dir, &record);
}

impl McpControl {
    pub(crate) fn stop(&self) {
        if let Some(r) = self
            .running
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            r.cancel.cancel();
        }
        if let Some(dir) = discovery_dir() {
            discovery::remove_instance_record(&dir, &self.instance_id);
        }
        self.view_requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    pub(crate) fn status(&self) -> Value {
        let guard = self.running.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            None => json!({
                "ok": true,
                "enabled": false,
                "instance_id": self.instance_id,
            }),
            Some(r) => json!({
                "ok": true,
                "enabled": true,
                "instance_id": self.instance_id,
                "port": r.port,
                "endpoint": r.endpoint,
                "client_config": {
                    "mcpServers": {
                        "HiFiShifter": { "url": r.endpoint }
                    }
                }
            }),
        }
    }
}

pub(crate) fn set_enabled(app: &tauri::AppHandle, enabled: bool) -> Value {
    set_enabled_with_port(app, enabled, Some(hifishifter_mcp::server::LOCAL_PORT))
}

pub(crate) fn set_enabled_with_port(
    app: &tauri::AppHandle,
    enabled: bool,
    requested_port: Option<u16>,
) -> Value {
    let control = app.state::<McpControl>();
    if !enabled {
        control.stop();
        return control.status();
    }
    let mut guard = control.running.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_some() {
        drop(guard);
        return control.status();
    }
    let bind_port = requested_port.unwrap_or(0);
    let (listener, address) = match hifishifter_mcp::server::bind_local(bind_port) {
        Ok(v) => v,
        Err(e) => {
            return json!({
                "ok": false,
                "code": "MCP_BIND_FAILED",
                "message": format!("Cannot bind 127.0.0.1:{bind_port}: {e}")
            });
        }
    };
    let actual_port = address.port();
    let endpoint = format!("http://{address}/mcp");
    let cancel = CancellationToken::new();
    *guard = Some(Running {
        endpoint,
        port: actual_port,
        cancel: cancel.clone(),
    });
    drop(guard);
    sync_discovery_record(&app.state::<crate::state::AppState>(), &control);
    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        let backend = Arc::new(AppBackend(handle.clone()));
        if let Err(error) = hifishifter_mcp::server::serve(listener, backend, cancel).await {
            log::error!("MCP server stopped: {error}");
            handle.state::<McpControl>().stop();
            let _ = handle.emit(
                "mcp_status_changed",
                json!({ "enabled": false, "code": "MCP_SERVER_FAILED" }),
            );
        }
    });
    control.status()
}

/// Parse `--enable-mcp` and `--mcp-port=<u16|auto>` startup CLI flags.
pub(crate) fn maybe_start_from_args<I, S>(app: &tauri::AppHandle, args: I)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut enable = false;
    let mut port: Option<u16> = Some(hifishifter_mcp::server::LOCAL_PORT);
    for raw in args {
        let s = raw.as_ref().trim();
        if s == "--enable-mcp" {
            enable = true;
        } else if let Some(val) = s.strip_prefix("--mcp-port=") {
            enable = true;
            if val.eq_ignore_ascii_case("auto") || val == "0" {
                port = Some(0);
            } else if let Ok(parsed) = val.parse::<u16>() {
                port = Some(parsed);
            }
        }
    }
    if enable {
        let status = set_enabled_with_port(app, true, port);
        if status.get("ok").and_then(Value::as_bool) == Some(true) {
            log::info!(
                "MCP enabled via CLI at {}",
                status["endpoint"].as_str().unwrap_or_default()
            );
        } else {
            log::warn!("MCP CLI startup failed: {status}");
        }
    }
}

struct AppBackend(tauri::AppHandle);
impl EditorBackend for AppBackend {
    fn execute(
        &self,
        request: EditorRequest,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Value> + Send>> {
        let app = self.0.clone();
        Box::pin(async move {
            match tauri::async_runtime::spawn_blocking(move || editor::execute(&app, request)).await
            {
                Ok(value) => value,
                Err(error) => json!({
                    "ok": false,
                    "code": hifishifter_mcp::McpErrorCode::EditorTaskFailed.as_str(),
                    "message": error.to_string()
                }),
            }
        })
    }
}

pub(crate) fn refresh_view(app: &tauri::AppHandle) -> Value {
    editor::view_snapshot(app)
}

pub(crate) fn acknowledge_view(app: &tauri::AppHandle, revision: String) -> Value {
    if revision.len() > 256 {
        return json!({ "ok": false });
    }
    *app.state::<McpControl>()
        .view_revision
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = Some(revision);
    json!({ "ok": true })
}

pub(crate) fn revision_for_history(
    state: &crate::state::AppState,
    tl: &crate::state::TimelineState,
    instance: &str,
) -> String {
    editor::revision_for_history(state, tl, instance)
}

pub(super) fn request_view(app: &tauri::AppHandle, request: hifishifter_mcp::ViewRequest) -> Value {
    let control = app.state::<McpControl>();
    let id = uuid::Uuid::new_v4().to_string();
    let (send, receive) = std::sync::mpsc::channel();
    {
        let mut pending = control
            .view_requests
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if pending.len() >= 8 {
            return json!({ "ok": false, "code": "VIEW_BUSY" });
        }
        pending.insert(id.clone(), send);
    }
    if app
        .emit(
            "mcp_view_request",
            json!({ "request_id": id, "request": request }),
        )
        .is_err()
    {
        control
            .view_requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&id);
        return json!({ "ok": false, "code": "VIEW_UNAVAILABLE" });
    }
    let result = receive
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap_or_else(|_| {
            json!({
                "ok": false,
                "code": "VIEW_UNAVAILABLE",
                "message": "The App editor did not acknowledge this request"
            })
        });
    control
        .view_requests
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&id);
    result
}

pub(crate) fn reply_view(app: &tauri::AppHandle, request_id: String, result: Value) -> Value {
    let pending = app
        .state::<McpControl>()
        .view_requests
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&request_id);
    json!({ "ok": pending.is_some_and(|send| send.send(result).is_ok()) })
}
