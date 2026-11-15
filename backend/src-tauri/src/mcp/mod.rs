//! App-owned MCP lifecycle. Disabled by default; never loads a second project.
mod draft;
mod editor;

use hifishifter_mcp::{EditorBackend, EditorRequest};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tauri::{Emitter, Manager};
use tokio_util::sync::CancellationToken;

struct Running {
    endpoint: String,
    cancel: CancellationToken,
}
pub(crate) struct McpControl {
    running: Mutex<Option<Running>>,
    pub instance_id: String,
    pub(super) ledger: Mutex<std::collections::VecDeque<(String, String, Value)>>,
    pub(super) view_revision: Mutex<Option<String>>,
    view_requests: Mutex<std::collections::HashMap<String, std::sync::mpsc::Sender<Value>>>,
}
impl Default for McpControl {
    fn default() -> Self {
        Self {
            running: Mutex::new(None),
            instance_id: uuid::Uuid::new_v4().to_string(),
            ledger: Mutex::new(Default::default()),
            view_revision: Mutex::new(None),
            view_requests: Mutex::new(Default::default()),
        }
    }
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
        self.view_requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
    pub(crate) fn status(&self) -> Value {
        let guard = self.running.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            None => json!({"ok":true,"enabled":false,"instance_id":self.instance_id}),
            Some(r) => {
                json!({"ok":true,"enabled":true,"instance_id":self.instance_id,"endpoint":r.endpoint,
                "client_config":{"mcpServers":{"HiFiShifter":{"url":r.endpoint}}}})
            }
        }
    }
}

pub(crate) fn set_enabled(app: &tauri::AppHandle, enabled: bool) -> Value {
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
    let (listener, address) = match hifishifter_mcp::server::bind_local(
        hifishifter_mcp::server::LOCAL_PORT,
    ) {
        Ok(v) => v,
        Err(e) => {
            return json!({"ok":false,"code":"MCP_BIND_FAILED","message":format!("Cannot bind 127.0.0.1:{}: {e}",hifishifter_mcp::server::LOCAL_PORT)})
        }
    };
    let endpoint = format!("http://{address}/mcp");
    let cancel = CancellationToken::new();
    *guard = Some(Running {
        endpoint,
        cancel: cancel.clone(),
    });
    drop(guard);
    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        let backend = Arc::new(AppBackend(handle.clone()));
        if let Err(error) = hifishifter_mcp::server::serve(listener, backend, cancel).await {
            log::error!("MCP server stopped: {error}");
            handle.state::<McpControl>().stop();
            let _ = handle.emit(
                "mcp_status_changed",
                json!({"enabled":false,"code":"MCP_SERVER_FAILED"}),
            );
        }
    });
    control.status()
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
                Err(error) => {
                    json!({"ok":false,"code":"EDITOR_TASK_FAILED","message":error.to_string()})
                }
            }
        })
    }
}

pub(crate) fn refresh_view(app: &tauri::AppHandle) -> Value {
    editor::view_snapshot(app)
}
pub(crate) fn acknowledge_view(app: &tauri::AppHandle, revision: String) -> Value {
    if revision.len() > 256 {
        return json!({"ok":false});
    }
    *app.state::<McpControl>()
        .view_revision
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = Some(revision);
    json!({"ok":true})
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
            return json!({"ok":false,"code":"VIEW_BUSY"});
        }
        pending.insert(id.clone(), send);
    }
    if app
        .emit(
            "mcp_view_request",
            json!({"request_id":id,"request":request}),
        )
        .is_err()
    {
        control
            .view_requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&id);
        return json!({"ok":false,"code":"VIEW_UNAVAILABLE"});
    }
    let result=receive.recv_timeout(std::time::Duration::from_secs(5)).unwrap_or_else(|_|json!({"ok":false,"code":"VIEW_UNAVAILABLE","message":"The App editor did not acknowledge this request"}));
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
    json!({"ok":pending.is_some_and(|send|send.send(result).is_ok())})
}
