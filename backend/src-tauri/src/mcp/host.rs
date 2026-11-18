//! Runtime host abstraction decoupling the MCP state machine from `tauri::AppHandle`.
//!
//! Production executes against [`TauriRuntimeHost`], while deterministic end-to-end
//! tests execute the exact same `editor::execute_with_host` state machine against
//! [`HeadlessMcpHost`] without spawning a WebView window.

use super::{draft::Failure, McpControl};
use crate::state::AppState;
use hifishifter_mcp::{TransportAction, TransportRequest, ViewRequest};
use serde_json::{json, Value};
use tauri::{Emitter, Manager};

pub(super) trait McpRuntimeHost: Sync {
    fn state(&self) -> &AppState;
    fn control(&self) -> &McpControl;
    fn notify_project_changed(&self, revision: &str);
    fn notify_notebook_changed(&self, revision: &str);
    fn schedule_pitch_analysis(&self, root_track_id: &str);
    fn request_view(&self, request: ViewRequest) -> Value;
    fn transport(&self, request: &TransportRequest) -> Value;
    fn playback_state(&self) -> Result<Value, Failure>;
}

pub(super) struct TauriRuntimeHost<'a>(pub &'a tauri::AppHandle);

impl McpRuntimeHost for TauriRuntimeHost<'_> {
    fn state(&self) -> &AppState {
        self.0.state::<AppState>().inner()
    }

    fn control(&self) -> &McpControl {
        self.0.state::<McpControl>().inner()
    }

    fn notify_project_changed(&self, revision: &str) {
        let _ = self
            .0
            .emit("mcp_project_changed", json!({ "revision": revision }));
        crate::commands::playback::request_background_render(self.0);
        super::sync_discovery_record(self.state(), self.control());
    }

    fn notify_notebook_changed(&self, revision: &str) {
        let _ = self.0.emit(
            "mcp_project_changed",
            json!({ "revision": revision, "notebook": true }),
        );
        super::sync_discovery_record(self.state(), self.control());
    }

    fn schedule_pitch_analysis(&self, root_track_id: &str) {
        crate::pitch_analysis::maybe_schedule_pitch_orig(&self.state().timeline, root_track_id);
    }

    fn request_view(&self, request: ViewRequest) -> Value {
        super::request_view(self.0, request)
    }

    fn transport(&self, request: &TransportRequest) -> Value {
        match request.action {
            TransportAction::Play => {
                if let Some(position) = request.position_sec {
                    let seeked =
                        crate::commands::set_transport(self.0.state(), Some(position), None);
                    if seeked.get("ok").and_then(Value::as_bool) == Some(false) {
                        return seeked;
                    }
                }
                crate::commands::play_original(self.0.state())
            }
            TransportAction::Stop => crate::commands::stop_audio(self.0.state()),
            TransportAction::Seek => {
                crate::commands::set_transport(self.0.state(), request.position_sec, None)
            }
        }
    }

    fn playback_state(&self) -> Result<Value, Failure> {
        serde_json::to_value(crate::commands::get_playback_state(self.0.state()))
            .map_err(|e| Failure::new("SERIALIZATION_FAILED", e.to_string()))
    }
}

#[cfg(test)]
pub(super) struct HeadlessMcpHost {
    pub app_state: AppState,
    pub mcp_control: McpControl,
    pub notifications: std::sync::Mutex<Vec<String>>,
    pub scheduled_roots: std::sync::Mutex<Vec<String>>,
    pub last_view: std::sync::Mutex<Value>,
}

#[cfg(test)]
impl Default for HeadlessMcpHost {
    fn default() -> Self {
        Self {
            app_state: AppState::default(),
            mcp_control: McpControl::default(),
            notifications: std::sync::Mutex::new(Vec::new()),
            scheduled_roots: std::sync::Mutex::new(Vec::new()),
            last_view: std::sync::Mutex::new(json!({
                "ok": true,
                "selected_track_id": "track_main",
                "selected_clip_id": null,
                "active_param": "pitch",
                "reference_root_track_ids": [],
                "tool_mode": "select",
                "focus_start_sec": null,
                "focus_end_sec": null,
            })),
        }
    }
}

#[cfg(test)]
impl McpRuntimeHost for HeadlessMcpHost {
    fn state(&self) -> &AppState {
        &self.app_state
    }

    fn control(&self) -> &McpControl {
        &self.mcp_control
    }

    fn notify_project_changed(&self, revision: &str) {
        self.notifications
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(revision.to_string());
    }

    fn notify_notebook_changed(&self, revision: &str) {
        self.notifications
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(format!("notebook:{revision}"));
    }

    fn schedule_pitch_analysis(&self, root_track_id: &str) {
        self.scheduled_roots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(root_track_id.to_string());
    }

    fn request_view(&self, request: ViewRequest) -> Value {
        let mut view = self.last_view.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(id) = request.selected_track_id {
            view["selected_track_id"] = json!(id);
            self.app_state
                .timeline
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .selected_track_id = view["selected_track_id"].as_str().map(str::to_owned);
        }
        if let Some(id) = request.selected_clip_id {
            view["selected_clip_id"] = json!(id);
            self.app_state
                .timeline
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .selected_clip_id = view["selected_clip_id"].as_str().map(str::to_owned);
        }
        if let Some(param) = request.active_param {
            view["active_param"] = json!(param);
        }
        if let Some(refs) = request.reference_root_track_ids {
            view["reference_root_track_ids"] = json!(refs);
        }
        if let Some(mode) = request.tool_mode {
            view["tool_mode"] = json!(mode);
        }
        if let Some(s) = request.focus_start_sec {
            view["focus_start_sec"] = json!(s);
            if request.seek_playhead == Some(true) {
                self.app_state
                    .timeline
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .playhead_sec = s;
            }
        }
        if let Some(e) = request.focus_end_sec {
            view["focus_end_sec"] = json!(e);
        }
        view.clone()
    }

    fn transport(&self, request: &TransportRequest) -> Value {
        let mut tl = self
            .app_state
            .timeline
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(pos) = request.position_sec {
            tl.playhead_sec = pos;
        }
        json!({
            "ok": true,
            "playing": matches!(request.action, TransportAction::Play),
            "playhead_sec": tl.playhead_sec,
        })
    }

    fn playback_state(&self) -> Result<Value, Failure> {
        let tl = self
            .app_state
            .timeline
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        Ok(json!({
            "is_playing": false,
            "playhead_sec": tl.playhead_sec,
        }))
    }
}
