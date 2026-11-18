use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
};

use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::Response,
};
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    tool, tool_handler, tool_router, ServerHandler,
};
use tokio_util::sync::CancellationToken;

use crate::{contract::*, EditorBackend};

#[derive(Clone)]
struct EditorServer {
    backend: Arc<dyn EditorBackend>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl EditorServer {
    fn new(backend: Arc<dyn EditorBackend>) -> Self {
        Self {
            backend,
            tool_router: Self::tool_router(),
        }
    }
    async fn call(&self, request: EditorRequest) -> CallToolResult {
        let value = self.backend.execute(request).await;
        let failed = value.get("ok").and_then(serde_json::Value::as_bool) == Some(false);
        let mut result = CallToolResult::structured(value);
        result.is_error = Some(failed);
        result
    }
    #[tool(
        description = "Read the visible HiFiShifter project's current revision, per-root-track scope_revisions, effective App settings and connection state. Call this first. No audio or model inference is run."
    )]
    async fn hfs_get_context(&self) -> CallToolResult {
        self.call(EditorRequest::Context).await
    }
    #[tool(
        description = "Discover all track properties, available algorithms, root-owned curves, child-track offsets, static parameters, units, bounds and disabled dependencies. Pitch and virtual child parameters are included."
    )]
    async fn hfs_get_edit_capabilities(
        &self,
        Parameters(request): Parameters<TargetRequest>,
    ) -> CallToolResult {
        self.call(EditorRequest::Capabilities(request)).await
    }
    #[tool(
        description = "Read timeline, tracks, clips/takes, curve samples and edit masks, static values, parameter blocks, effective settings, playback, render, undo history, notebook or MCP audit ledger. Results are paginated; curve windows are bounded. Times are project seconds and curve indices use the returned frame_period_ms."
    )]
    async fn hfs_query(&self, Parameters(request): Parameters<QueryRequest>) -> CallToolResult {
        self.call(EditorRequest::Query(request)).await
    }
    #[tool(
        description = "Atomically edit the currently visible project. Supports tracks/hierarchy/settings, audio placement/crop/stretch/fades/takes, multi-candidate take packing, every available parameter curve and static value, pitch flatten/shift/hermite/vibrato/smooth, scale-preserving tempo and parameter blocks. Accepts expected_revision (whole-project CAS) or expected_scope (root-track scoped CAS for concurrent human-agent co-editing) plus idempotency_key; one batch is one undo step."
    )]
    async fn hfs_edit(&self, Parameters(request): Parameters<EditRequest>) -> CallToolResult {
        self.call(EditorRequest::Edit(request)).await
    }
    #[tool(
        description = "Undo or redo a project operation using the same history as the GUI. Requires the current revision so a concurrent human edit is not accidentally undone."
    )]
    async fn hfs_history(&self, Parameters(request): Parameters<HistoryRequest>) -> CallToolResult {
        self.call(EditorRequest::History(request)).await
    }
    #[tool(
        description = "Play, stop or seek the App's audible transport. Does not edit audio. Playback and editing use the same visible project."
    )]
    async fn hfs_transport(
        &self,
        Parameters(request): Parameters<TransportRequest>,
    ) -> CallToolResult {
        self.call(EditorRequest::Transport(request)).await
    }
    #[tool(
        description = "Save the editable project to a new absolute .hshp path without opening a dialog. Existing destinations are refused. Audio previews are not a replacement for this project file."
    )]
    async fn hfs_save_project(
        &self,
        Parameters(request): Parameters<SaveRequest>,
    ) -> CallToolResult {
        self.call(EditorRequest::Save(request)).await
    }
    #[tool(
        description = "Read App defaults and effective project stretch settings, or set the complete project override (null inherits defaults). Scope is project-wide, not per clip. Follows the same behavior as the App's stretch preferences; does not silently change synthesis algorithms."
    )]
    async fn hfs_settings(
        &self,
        Parameters(request): Parameters<SettingsRequest>,
    ) -> CallToolResult {
        self.call(EditorRequest::Settings(request)).await
    }
    #[tool(
        description = "Read or change the actual visible editor's selected track/clip, active parameter, reference tracks, drawing tool or viewport time focus. Uses the live GUI and returns its acknowledged state; fails clearly when the view is unavailable. No audio editing occurs."
    )]
    async fn hfs_editor_view(
        &self,
        Parameters(request): Parameters<ViewRequest>,
    ) -> CallToolResult {
        self.call(EditorRequest::View(request)).await
    }
    #[tool(
        description = "Wait until requested asynchronous subsystems (pitch_analysis, clip_render, gui_view) become ready or until timeout_ms expires, optionally triggering source pitch analysis on target tracks. Use this barrier instead of polling hfs_query in a loop."
    )]
    async fn hfs_wait_ready(&self, Parameters(request): Parameters<WaitRequest>) -> CallToolResult {
        self.call(EditorRequest::Wait(request)).await
    }
    #[tool(
        description = "Render a bounded phrase window (<= 30s) of the project or selected tracks offline, optionally writing a .wav preview file, and return deterministic acoustic QA metrics (peak/RMS dBFS, clipping, silent dropouts, and clip-boundary click/step checks) so Agents can self-verify audio quality."
    )]
    async fn hfs_render_preview(
        &self,
        Parameters(request): Parameters<RenderPreviewRequest>,
    ) -> CallToolResult {
        self.call(EditorRequest::RenderPreview(request)).await
    }
    #[tool(
        description = "Read the project's Markdown Notebook or atomically upsert/remove a fenced Agent review section (e.g. needs_review checklist, OOV notes, candidate explanations) without overwriting human-written notes."
    )]
    async fn hfs_notebook(
        &self,
        Parameters(request): Parameters<NotebookRequest>,
    ) -> CallToolResult {
        self.call(EditorRequest::Notebook(request)).await
    }
}

/// Export the JSON Schema bundle for every MCP tool input contract and error enum.
pub fn export_contract_schemas() -> serde_json::Value {
    serde_json::json!({
        "server": "HiFiShifter",
        "version": env!("CARGO_PKG_VERSION"),
        "error_codes": schemars::schema_for!(McpErrorCode),
        "tools": {
            "hfs_get_edit_capabilities": schemars::schema_for!(TargetRequest),
            "hfs_query": schemars::schema_for!(QueryRequest),
            "hfs_edit": schemars::schema_for!(EditRequest),
            "hfs_history": schemars::schema_for!(HistoryRequest),
            "hfs_transport": schemars::schema_for!(TransportRequest),
            "hfs_save_project": schemars::schema_for!(SaveRequest),
            "hfs_settings": schemars::schema_for!(SettingsRequest),
            "hfs_editor_view": schemars::schema_for!(ViewRequest),
            "hfs_wait_ready": schemars::schema_for!(WaitRequest),
            "hfs_render_preview": schemars::schema_for!(RenderPreviewRequest),
            "hfs_notebook": schemars::schema_for!(NotebookRequest),
        }
    })
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for EditorServer {
    fn get_info(&self) -> ServerConfig {
        let mut info = ServerConfig::default();
        info.instructions = Some("Control the open, visible HiFiShifter project. Start with hfs_get_context and hfs_get_edit_capabilities. Read revision (or scope_revisions for root-track concurrent co-editing), then send an atomic hfs_edit with explicit IDs and idempotency_key. Use hfs_wait_ready instead of sleep-polling for pitch analysis or synthesis renders, hfs_render_preview for acoustic QA & WAV previews, and hfs_notebook to post review checklists for the human. All time positions are project seconds; source offsets are file seconds.".into());
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info.name = "HiFiShifter".into();
        info.server_info.version = env!("CARGO_PKG_VERSION").into();
        info
    }
}

#[derive(Clone)]
struct LocalEndpoint {
    authority: String,
}

async fn validate_local_request(
    State(endpoint): State<LocalEndpoint>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Native MCP clients omit Origin. Browsers must never get access to this local editor endpoint.
    if request.headers().contains_key(header::ORIGIN) {
        return Err(StatusCode::FORBIDDEN);
    }
    if request
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        != Some(endpoint.authority.as_str())
    {
        return Err(StatusCode::FORBIDDEN);
    }
    Ok(next.run(request).await)
}

/// Listener is allocated by the App so enabling can synchronously report its real port.
pub async fn serve(
    listener: std::net::TcpListener,
    backend: Arc<dyn EditorBackend>,
    cancel: CancellationToken,
) -> Result<(), String> {
    let address = listener.local_addr().map_err(|e| e.to_string())?;
    if !address.ip().is_loopback() {
        return Err("MCP requires a loopback listener".into());
    }
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    let listener = tokio::net::TcpListener::from_std(listener).map_err(|e| e.to_string())?;
    let service = StreamableHttpService::new(
        move || Ok(EditorServer::new(backend.clone())),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default()
            .with_legacy_session_mode(true)
            .with_json_response(true)
            .with_max_request_body_bytes(4 * 1024 * 1024)
            .with_cancellation_token(cancel.child_token()),
    );
    let router =
        axum::Router::new()
            .nest_service("/mcp", service)
            .layer(middleware::from_fn_with_state(
                LocalEndpoint {
                    authority: address.to_string(),
                },
                validate_local_request,
            ));
    axum::serve(listener, router)
        .with_graceful_shutdown(cancel.cancelled_owned())
        .await
        .map_err(|e| e.to_string())
}

pub const LOCAL_PORT: u16 = 39393;

pub fn bind_local(port: u16) -> std::io::Result<(std::net::TcpListener, SocketAddr)> {
    let listener = std::net::TcpListener::bind((IpAddr::V4(std::net::Ipv4Addr::LOCALHOST), port))?;
    let address = listener.local_addr()?;
    Ok((listener, address))
}
